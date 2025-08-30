from __future__ import annotations

import io
import os
from typing import Optional

def _truthy(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    v = v.strip().strip('"').strip("'").lower()
    return v in {"1", "true", "yes", "on", "y"}

# Default follows env; routes may override via ?dry=0|1
_DRY_DEFAULT = _truthy("DRY_MODE", "1")

def transcribe_audio_bytes(filename: str, data: bytes, *, dry_mode: Optional[bool] = None) -> str:
    """
    Transcribe audio. If dry_mode is None, falls back to env default.
    """
    use_dry = _DRY_DEFAULT if dry_mode is None else dry_mode
    if use_dry:
        return "Stub transcript (DRY_MODE=1): candidate introductions and key project summaries…"

    # Live path via OpenAI Whisper
    from openai import OpenAI
    client = OpenAI()

    # The SDK accepts file-like objects; name helps format detection
    buff = io.BytesIO(data)
    buff.name = filename or "audio.m4a"

    # Text output; adjust model if you prefer a later Whisper variant
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=buff,
        response_format="text"
    )
    # The SDK returns text directly when response_format="text"
    if isinstance(resp, str):
        return resp.strip()

    # Fallback for SDKs that return an object with .text
    text = getattr(resp, "text", "") or ""
    return str(text).strip()

