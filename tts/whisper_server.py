"""
Run : 
CUDA_VISIBLE_DEVICES=0 python3 tts/whisper_server.py --port 8000 --model large-v3

Whisper NLL REST server (vLLM-backed) + optional CTC + WeSpeaker similarity

- POST /score: {tokens, text, ref_audio_path?} -> {nll, transcript, ctc?, sim?}
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import List, Optional
import math

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch.nn.functional import cross_entropy
import uvicorn

import whisper  # type: ignore
import whisper.audio as _wa

# vLLM (for fast transcription)
from vllm import LLM, SamplingParams  # type: ignore

# ---------------------------------------------------------------------------
# CLI / model loading
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Whisper NLL REST server (vLLM-backed)")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--model", type=str, default="large-v3", help="Whisper model name (e.g., large-v3 or openai/whisper-large-v3)")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:2, cpu … (for NLL path)")
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2", help="XCodec-2 repo or path")
parser.add_argument("--lang", type=str, default="ko", help="Language hint for vLLM decoder prompt (e.g., ko, en, auto)")
parser.add_argument("--max-tokens", type=int, default=200, help="Max decode tokens (vLLM)")
parser.add_argument("--temperature", type=float, default=0.0, help="vLLM temperature")
parser.add_argument("--top-p", type=float, default=1.0, help="vLLM top_p")
parser.add_argument("--ctc-model", type=str, default="kresnik/wav2vec2-large-xlsr-korean", help="HF CTC model id for optional CTC loss. Leave empty to disable.")
parser.add_argument("--nan-to-null", action="store_true", help="Return null for non-finite numbers instead of 422 error")

# NEW: WeSpeaker options
parser.add_argument("--ws-model", type=str, default="chinese", help="WeSpeaker model id (e.g., chinese). Empty to disable similarity.")
parser.add_argument("--ws-min-sec", type=float, default=1.0, help="Min seconds for similarity audio (pad if shorter).")
args, _ = parser.parse_known_args()

DEVICE = args.device
SAMPLE_RATE = 16000  # Whisper/XCodec-2 common SR

# Map short names (e.g., "large-v3") to HF IDs for vLLM
HF_MODEL_ID = args.model if "/" in args.model else f"openai/whisper-{args.model}"

# Load Whisper once (for NLL computation)
print(f"Loading Whisper '{args.model}' on {DEVICE} for NLL …")
WHISPER = whisper.load_model(args.model, device=DEVICE).eval()


def _get_mel_bins(model) -> int:
    if hasattr(model, "n_mels"):
        return int(model.n_mels)
    if hasattr(model, "conv1"):
        return int(model.conv1.weight.shape[1])
    if hasattr(model, "encoder") and hasattr(model.encoder, "conv1"):
        return int(model.encoder.conv1.weight.shape[1])
    return 80


REQ_BINS = _get_mel_bins(WHISPER)

# Load XCodec-2 once (for token→wav)
print("Loading XCodec-2 …")
from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore

CODEC = XCodec2Model.from_pretrained(args.codec).to(DEVICE).eval()

# Tokenizer for NLL (teacher forcing)
TOKENIZER = whisper.tokenizer.get_tokenizer(multilingual=True, task="transcribe")

# vLLM LLM for fast transcription
print(f"Loading vLLM LLM '{HF_MODEL_ID}' …")
LLM_ENGINE = LLM(
    model=HF_MODEL_ID,
    dtype="float16",
    enforce_eager=True,
    max_model_len=448,      # Whisper uses a small decode context
    max_num_seqs=512,
    limit_mm_per_prompt={"audio": 1},
)
S_PARAMS = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
)

# Optional: CTC model (HuggingFace Transformers)
CTC_PROCESSOR = None
CTC_MODEL = None
if args.ctc_model:
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
        print(f"Loading CTC model '{args.ctc_model}' …")
        CTC_PROCESSOR = Wav2Vec2Processor.from_pretrained(args.ctc_model)
        CTC_MODEL = Wav2Vec2ForCTC.from_pretrained(args.ctc_model).to(DEVICE).eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load CTC model '{args.ctc_model}': {e}")

# NEW: WeSpeaker model (for speaker similarity)
WS_MODEL = None
if args.ws_model:
    try:
        import wespeaker
        print(f"Loading WeSpeaker model '{args.ws_model}' …")
        WS_MODEL = wespeaker.load_model(args.ws_model)
    except Exception as e:
        print(f"[WARN] Failed to load WeSpeaker model '{args.ws_model}': {e}")
        WS_MODEL = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_wav_np(wav: np.ndarray) -> np.ndarray:
    """Make audio JSON/compute-safe: remove NaN/Inf, clamp to [-1,1], ensure 1D float32."""
    w = wav.astype(np.float32, copy=False)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.ndim > 1:
        w = w.reshape(-1)
    np.clip(w, -1.0, 1.0, out=w)
    return w

def _decoder_prompt(lang: str) -> str:
    parts = ["<|startoftranscript|>"]
    lang = (lang or "auto").lower()
    if lang != "auto":
        parts.append(f"<|{lang}|>") 
    parts.extend(["<|transcribe|>", "<|notimestamps|>"]) 
    return "".join(parts)

def _finite_or_null(x: float) -> Optional[float]:
    try:
        xf = float(x)
    except Exception:
        return None
    return xf if math.isfinite(xf) else None


DEC_PROMPT = _decoder_prompt(args.lang)


@torch.inference_mode()
def tokens_to_wav(tokens: List[int]) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, None, :]
    wav = CODEC.decode_code(t)[0, 0]
    return wav.float().cpu()


@torch.inference_mode()
def whisper_nll(wav: torch.Tensor, text: str) -> float:
    # Whisper expects float32 numpy audio at 16 kHz
    audio_np = _sanitize_wav_np(wav.numpy())
    mel = _wa.log_mel_spectrogram(_wa.pad_or_trim(audio_np), n_mels=REQ_BINS)
    mel = torch.as_tensor(mel, device=DEVICE)[None]
    tgt = torch.tensor([TOKENIZER.sot] + TOKENIZER.encode(text) + [TOKENIZER.eot], device=DEVICE)[None]
    enc = WHISPER.encoder(mel)
    logits = WHISPER.decoder(tgt[:, :-1], enc)
    loss = cross_entropy(logits.view(-1, logits.size(-1)), tgt[:, 1:].view(-1))
    return float(loss)


def vllm_transcribe(wav: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    # Build a vLLM multi-modal prompt: audio in encoder prompt, text control tokens in decoder prompt
    prompt = {
        "encoder_prompt": {"prompt": "", "multi_modal_data": {"audio": (_sanitize_wav_np(wav).astype(np.float32), sr)}},
        "decoder_prompt": DEC_PROMPT,
    }
    try:
        out = LLM_ENGINE.generate([prompt], S_PARAMS, use_tqdm=False)[0]
        return out.outputs[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"vLLM transcription failed: {e}")


@torch.inference_mode()
def ctc_loss_value(wav: np.ndarray, text: str) -> float:
    if CTC_MODEL is None or CTC_PROCESSOR is None:
        raise RuntimeError("CTC model not loaded. Provide --ctc-model to enable CTC loss.")

    # 1) Prepare acoustic inputs
    proc_inputs = CTC_PROCESSOR(_sanitize_wav_np(wav), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = (proc_inputs["input_values"] if isinstance(proc_inputs, dict) else proc_inputs.input_values).to(DEVICE)
    attention_mask = (proc_inputs.get("attention_mask") if isinstance(proc_inputs, dict) else getattr(proc_inputs, "attention_mask", None))
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # 2) Prepare labels
    labels = None
    try:
        ctx = getattr(CTC_PROCESSOR, "as_target_processor", None)
        if ctx is not None:
            with ctx():
                lab = CTC_PROCESSOR(text, return_tensors="pt", padding=True)
                labels = lab["input_ids"] if isinstance(lab, dict) else lab.input_ids
    except Exception:
        labels = None

    if labels is None:
        tokenizer = getattr(CTC_PROCESSOR, "tokenizer", CTC_PROCESSOR)
        lab = tokenizer(text, return_tensors="pt", padding=True)
        labels = lab["input_ids"] if isinstance(lab, dict) else lab.input_ids

    pad_id = getattr(getattr(CTC_PROCESSOR, "tokenizer", None), "pad_token_id", None)
    if pad_id is not None:
        labels = labels.masked_fill(labels == pad_id, -100)
    labels = labels.to(DEVICE)

    outputs = CTC_MODEL(input_values=input_values, attention_mask=attention_mask, labels=labels)
    return float(outputs.loss)

# ---------------------- NEW: WeSpeaker similarity helpers -------------------

def _ensure_16k_mono_tensor(wav_t: torch.Tensor, sr: int, min_sec: float) -> torch.Tensor:
    """
    wav_t: [T] or [1,T] float tensor on CPU
    returns: [1,T'] float32 at 16kHz mono, padded to min_sec if shorter
    """
    w = wav_t.float().view(1, -1)
    if sr != SAMPLE_RATE:
        w = torchaudio.functional.resample(w, sr, SAMPLE_RATE)
    min_len = int(min_sec * SAMPLE_RATE)
    if w.size(1) < min_len:
        w = torch.nn.functional.pad(w, (0, min_len - w.size(1)))
    return w

def _load_and_prepare_ref(path: str, min_sec: float) -> torch.Tensor:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ref_audio_path not found: {path}")
    w, sr = torchaudio.load(path)  # [C,T]
    w = w.mean(0, keepdim=True)    # mono
    return _ensure_16k_mono_tensor(w.squeeze(0), sr, min_sec)

def _save_tmp_wav_and_return_path(wav_1xT: torch.Tensor) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    torchaudio.save(tmp.name, wav_1xT, SAMPLE_RATE)
    return tmp.name

@torch.inference_mode()
def _wespeaker_normed_embedding_from_path(model, path: str) -> torch.Tensor:
    emb = model.extract_embedding(path)  # np.ndarray [D]
    e = torch.tensor(emb, dtype=torch.float32)
    return torch.nn.functional.normalize(e, dim=-1)

@torch.inference_mode()
def speaker_similarity_with_ref(decoded_wav_t: torch.Tensor, ref_audio_path: str) -> float:
    """
    decoded_wav_t: [T] float32 CPU tensor at 16k (from tokens_to_wav)
    ref_audio_path: path to reference audio (any sr/channels)
    returns: cosine similarity in [-1, 1]
    """
    if WS_MODEL is None:
        raise RuntimeError("WeSpeaker model not loaded. Pass --ws-model to enable similarity.")

    refs_wav = _load_and_prepare_ref(ref_audio_path, args.ws_min_sec)         # [1,T]
    dec_wav  = _ensure_16k_mono_tensor(decoded_wav_t, SAMPLE_RATE, args.ws_min_sec)

    dec_tmp = _save_tmp_wav_and_return_path(dec_wav)
    ref_tmp = _save_tmp_wav_and_return_path(refs_wav)
    try:
        e_dec = _wespeaker_normed_embedding_from_path(WS_MODEL, dec_tmp)
        e_ref = _wespeaker_normed_embedding_from_path(WS_MODEL, ref_tmp)
        sim = float(torch.dot(e_dec, e_ref).item())
    finally:
        try:
            os.remove(dec_tmp)
        except Exception:
            pass
        try:
            os.remove(ref_tmp)
        except Exception:
            pass
    return sim

# ---------------------------------------------------------------------------
# FastAPI definitions
# ---------------------------------------------------------------------------
app = FastAPI(title="Whisper NLL server (vLLM)", version="0.4.0")


class ScoreRequest(BaseModel):
    tokens: List[int] = Field(..., description="Speech token ids (<|s_xxx|>)")
    text: str = Field(..., description="Ground-truth text for NLL")
    ref_audio_path: Optional[str] = Field(None, description="Reference audio path for speaker similarity (WeSpeaker)")


class ScoreResponse(BaseModel):
    nll: Optional[float] = None
    transcript: str
    ctc: Optional[float] = None              # Optional CTC loss
    sim: Optional[float] = None              # NEW: speaker cosine similarity (WeSpeaker), if ref provided


@app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
async def score(req: ScoreRequest):
    try:
        # 1) Decode XCodec-2 tokens → wav
        wav = tokens_to_wav(req.tokens)  # [T] CPU float
        if wav.numel() == 0:
            raise HTTPException(422, detail="Decoded audio is empty")

        # Sanitize audio for downstream steps
        wav_np = _sanitize_wav_np(wav.numpy())
        wav_t = torch.from_numpy(wav_np)

        # 2) NLL via Whisper teacher-forced CE (torch path)
        nll_val = whisper_nll(wav_t, req.text)
        if not math.isfinite(nll_val):
            if args.nan_to_null:
                nll_val = None
            else:
                raise HTTPException(422, detail="Non-finite NLL computed (check audio/text)")

        # 3) Transcript via vLLM Whisper (fast)
        transcript = vllm_transcribe(wav_np, SAMPLE_RATE)

        # 4) Optional: CTC loss (silently drop if non-finite)
        ctc_val = None
        if CTC_MODEL is not None:
            try:
                ctc_val = ctc_loss_value(wav_np, req.text)
                if not math.isfinite(ctc_val):
                    print("[WARN] CTC produced non-finite value; omitting from response")
                    ctc_val = None
            except Exception as e:
                print(f"[WARN] CTC loss failed: {e}")

        # 5) NEW: Optional speaker similarity with reference audio
        sim_val = None
        if req.ref_audio_path:
            try:
                sim_val = speaker_similarity_with_ref(wav_t, req.ref_audio_path)
                if not math.isfinite(sim_val):
                    print("[WARN] WeSpeaker similarity non-finite; omitting from response")
                    sim_val = None
            except Exception as e:
                # 유사도 계산 실패는 치명적이지 않으므로 경고만 남기고 계속
                print(f"[WARN] Speaker similarity failed: {e}")
                sim_val = None

        # JSON safety for NaN/Inf if requested
        if args.nan_to_null:
            nll_safe = None if (nll_val is None or (isinstance(nll_val, float) and not math.isfinite(nll_val))) else nll_val
            ctc_safe = None if (ctc_val is None or (isinstance(ctc_val, float) and not math.isfinite(ctc_val))) else ctc_val
            sim_safe = None if (sim_val is None or (isinstance(sim_val, float) and not math.isfinite(sim_val))) else sim_val
        else:
            nll_safe, ctc_safe, sim_safe = nll_val, ctc_val, sim_val

        return ScoreResponse(nll=nll_safe, transcript=transcript, ctc=ctc_safe, sim=sim_safe)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "whisper": args.model,
        "vllm": HF_MODEL_ID,
        "ctc_loaded": CTC_MODEL is not None,
        "wespeaker_loaded": WS_MODEL is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
