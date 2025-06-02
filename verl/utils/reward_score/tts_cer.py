# Reward calculation – **Whisper server client**
# Uses external REST server (`whisper_server.py`) running on GPU 3
# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache 2.0 License.
"""Shaped reward for LLaSA-TTS.

Changes
-------
* Whisper inference is **off-loaded** to a dedicated server (default
  `http://localhost:8000`).  No GPU usage inside the RL workers.
* Worker sends only **speech token IDs + text** → minimal network payload.
* Response: `{ nll, transcript }` – we reuse both for reward.
"""

from __future__ import annotations

import os, re, warnings, json, time
from functools import lru_cache
from typing import List

import numpy as np
import requests
import torch
from jiwer import cer

# ---------------------------------------------------------------------------
# 1.  XCodec-2 decode (CPU, tiny) – gives optional wav dump for debugging.
# ---------------------------------------------------------------------------
from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore


@lru_cache(maxsize=1)
def _load_codec(path: str = "HKUST-Audio/xcodec2") -> XCodec2Model:
    return XCodec2Model.from_pretrained(path).eval()


def _parse_ids(token_str: str) -> List[int]:
    return [int(t) for t in re.findall(r"<\|s_(\d+)\|>", token_str)]

# ---------------------------------------------------------------------------
# 2.  Whisper server client helpers
# ---------------------------------------------------------------------------
SERVER = os.getenv("WHISPER_SERVER", "http://localhost:8000")
SCORE_URL = f"{SERVER.rstrip('/')}/score"
HEALTH_URL = f"{SERVER.rstrip('/')}/healthz"

# quick health cache to avoid hitting server each call
_last_health = 0.0

def _check_server():
    global _last_health
    if time.time() - _last_health < 30:
        return
    try:
        requests.get(HEALTH_URL, timeout=2)
        _last_health = time.time()
    except Exception as e:
        raise RuntimeError(f"Whisper server not reachable at {SERVER}: {e}")


def _remote_whisper(tokens: List[int], text: str):
    _check_server()
    payload = {"tokens": tokens, "text": text}
    r = requests.post(SCORE_URL, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()  # {nll, transcript}

# ---------------------------------------------------------------------------
# 3.  Reward computation (same shaping)
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    beta_c: float = 3.0,
    tau_n: float = 3.0,
    lambda_c: float = 0.6,
    lambda_n: float = 0.4,
    debug_dump: bool = False,
) -> float:
    """Return reward in [0,1] using remote Whisper."""

    ids = _parse_ids(solution_str)

    try:
        resp = _remote_whisper(ids, ground_truth)
        nll = float(resp["nll"])
        transcript = resp.get("transcript", "")
    except Exception as e:
        warnings.warn(f"Whisper server error: {e}; CER-only fallback")
        nll = None
        transcript = ""

    # CER utility
    hyp = transcript if transcript else ground_truth  # in worst case CER=0
    c = float(cer(ground_truth, hyp))
    cer_u = 1.0 - np.tanh(beta_c * c)

    # NLL utility
    if nll is not None:
        nll_u = float(np.exp(-nll / tau_n))
    else:
        nll_u = 1e-9

    denom = lambda_c / cer_u + lambda_n / nll_u
    reward = (lambda_c + lambda_n) / denom if denom > 0 else 0.0

    print(f"\033[92mCER: {c:.3f}, NLL: {nll}, Reward: {reward:.4f}\033[0m")
    return max(0.0, min(1.0, reward))

# CLI quick test
if __name__ == "__main__":
    import sys
    print(json.dumps({"reward": compute_score("cli", sys.argv[1], sys.argv[2])}, indent=2, ensure_ascii=False))
