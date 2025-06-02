"""Lightweight REST server that returns Whisper NLL (and transcript).

Usage
-----
```bash
# GPU 3 only
CUDA_VISIBLE_DEVICES=3 python whisper_server.py --port 8000 --model large-v3
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
  "transcript": "안녕하세요"
}
```

You may also POST raw *wav* bytes to `/score_wav` if you want to keep
decoding client‑side.
"""

from __future__ import annotations

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

import torch
import whisper  # type: ignore
import whisper.audio as _wa
from torch.nn.functional import cross_entropy

# ---------------------------------------------------------------------------
# CLI / model loading
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Whisper NLL REST server")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--model", type=str, default="large-v3", help="Whisper model name")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:2, cpu …")
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2", help="XCodec‑2 repo or path")
args, _ = parser.parse_known_args()

DEVICE = args.device

# Load Whisper once
print(f"Loading Whisper '{args.model}' on {DEVICE} …")
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

# Load XCodec‑2 once
print("Loading XCodec‑2 …")
from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore
CODEC = XCodec2Model.from_pretrained(args.codec).to(DEVICE).eval()

# Tokenizer
TOKENIZER = whisper.tokenizer.get_tokenizer(multilingual=True, task="transcribe")

# ---------------------------------------------------------------------------
# FastAPI definitions
# ---------------------------------------------------------------------------
app = FastAPI(title="Whisper NLL server", version="0.1.0")


class ScoreRequest(BaseModel):
    tokens: List[int] = Field(..., description="Speech token ids (<|s_xxx|>)")
    text: str = Field(..., description="Ground‑truth text for NLL")


class ScoreResponse(BaseModel):
    nll: float
    transcript: str


@torch.inference_mode()
def tokens_to_wav(tokens: List[int]) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, None, :]
    wav = CODEC.decode_code(t)[0, 0]
    return wav.float().cpu()


@torch.inference_mode()
def whisper_nll(wav: torch.Tensor, text: str) -> float:
    mel = _wa.log_mel_spectrogram(_wa.pad_or_trim(wav.numpy()), n_mels=REQ_BINS)
    mel = torch.as_tensor(mel, device=DEVICE)[None]
    tgt = torch.tensor([TOKENIZER.sot] + TOKENIZER.encode(text) + [TOKENIZER.eot], device=DEVICE)[None]
    enc = WHISPER.encoder(mel)
    logits = WHISPER.decoder(tgt[:, :-1], enc)
    return float(cross_entropy(logits.view(-1, logits.size(-1)), tgt[:, 1:].view(-1)))


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    try:
        wav = tokens_to_wav(req.tokens)
        nll_val = whisper_nll(wav, req.text)
        # greedy transcript (quick)
        transcript = WHISPER.transcribe(audio=wav.numpy(), fp16=torch.cuda.is_available())["text"].strip()
        return ScoreResponse(nll=nll_val, transcript=transcript)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
