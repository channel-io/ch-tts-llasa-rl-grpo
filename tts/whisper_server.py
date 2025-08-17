"""
Lightweight REST server that returns Whisper NLL (and transcript).

Now backed by **vLLM** for fast transcription, while keeping the same
endpoints. Optionally computes **CTC loss** (via a separate CTC ASR model)
without changing paths.

Usage
-----
```bash
# GPU 3 only
CUDA_VISIBLE_DEVICES=3 python whisper_server.py --port 8000 --model large-v3 \
  --ctc-model jonatasgrosman/wav2vec2-large-xlsr-53-korean  # (옵션) CTC loss 포함 시
```

Client (reward function) can POST JSON:
```json
{
  "tokens": [123, 456, ...],
  "text": "안녕하세요"
}
```
Response:
```json
{
  "nll": 4.7321,
  "transcript": "안녕하세요",
  "ctc": 12.3456   // present only if --ctc-model is provided
}
```

You may also POST raw *wav* bytes to `/score_wav` if you want to keep
decoding client‑side.  (Endpoint kept as a comment for compatibility;
this file defines the original `/score` and `/healthz` endpoints only.)
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional
import math

import numpy as np
import torch
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
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2", help="XCodec‑2 repo or path")
parser.add_argument("--lang", type=str, default="ko", help="Language hint for vLLM decoder prompt (e.g., ko, en, auto)")
parser.add_argument("--max-tokens", type=int, default=200, help="Max decode tokens (vLLM)")
parser.add_argument("--temperature", type=float, default=0.0, help="vLLM temperature")
parser.add_argument("--top-p", type=float, default=1.0, help="vLLM top_p")
parser.add_argument("--ctc-model", type=str, default="kresnik/wav2vec2-large-xlsr-korean", help="HF CTC model id for optional CTC loss (e.g., jonatasgrosman/wav2vec2-large-xlsr-53-korean). Leave empty to disable.")
parser.add_argument("--nan-to-null", action="store_true", help="Return null for non-finite numbers instead of 422 error")
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

# Load XCodec‑2 once (for token→wav)
print("Loading XCodec‑2 …")
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
    # Access by key for HF compatibility
    input_values = (proc_inputs["input_values"] if isinstance(proc_inputs, dict) else proc_inputs.input_values).to(DEVICE)
    attention_mask = (proc_inputs.get("attention_mask") if isinstance(proc_inputs, dict) else getattr(proc_inputs, "attention_mask", None))
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # 2) Prepare labels (robust to HF version differences)
    labels = None
    try:
        # Preferred path (works on newer transformers)
        ctx = getattr(CTC_PROCESSOR, "as_target_processor", None)
        if ctx is not None:
            with ctx():
                lab = CTC_PROCESSOR(text, return_tensors="pt", padding=True)
                labels = lab["input_ids"] if isinstance(lab, dict) else lab.input_ids
    except Exception:
        labels = None

    if labels is None:
        # Fallback: call tokenizer directly
        tokenizer = getattr(CTC_PROCESSOR, "tokenizer", CTC_PROCESSOR)
        lab = tokenizer(text, return_tensors="pt", padding=True)
        labels = lab["input_ids"] if isinstance(lab, dict) else lab.input_ids

    # Mask pad tokens to -100 so they don't contribute to the loss
    pad_id = getattr(getattr(CTC_PROCESSOR, "tokenizer", None), "pad_token_id", None)
    if pad_id is not None:
        labels = labels.masked_fill(labels == pad_id, -100)
    labels = labels.to(DEVICE)

    # 3) Forward pass with labels → returns CTC loss
    outputs = CTC_MODEL(input_values=input_values, attention_mask=attention_mask, labels=labels)
    return float(outputs.loss)


# ---------------------------------------------------------------------------
# FastAPI definitions
# ---------------------------------------------------------------------------
app = FastAPI(title="Whisper NLL server (vLLM)", version="0.3.0")


class ScoreRequest(BaseModel):
    tokens: List[int] = Field(..., description="Speech token ids (<|s_xxx|>)")
    text: str = Field(..., description="Ground‑truth text for NLL")


class ScoreResponse(BaseModel):
    nll: Optional[float] = None
    transcript: str
    ctc: Optional[float] = None  # Optional CTC loss (if --ctc-model is provided)


@app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
async def score(req: ScoreRequest):
    try:
        # 1) Decode XCodec-2 tokens → wav
        wav = tokens_to_wav(req.tokens)
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

        # If nan-to-null is requested, coerce non-finite numbers to None for JSON safety
        if args.nan_to_null:
            nll_safe = None if (nll_val is None or (isinstance(nll_val, float) and not math.isfinite(nll_val))) else nll_val
            ctc_safe = None if (ctc_val is None or (isinstance(ctc_val, float) and not math.isfinite(ctc_val))) else ctc_val
        else:
            nll_safe, ctc_safe = nll_val, ctc_val

        return ScoreResponse(nll=nll_safe, transcript=transcript, ctc=ctc_safe)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
