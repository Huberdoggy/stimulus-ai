from __future__ import annotations

import io
from typing import Optional

from openai import OpenAI


def transcribe_audio_bytes(filename: str, data: bytes, *, dry_mode: Optional[bool] = None) -> str:
    """
    Transcribe audio bytes to text (Live-only; DRY removed in P1.5).

    Args:
        filename: Original filename (used to hint format; also set on the in-memory buffer).
        data:     Raw audio bytes.

    Returns:
        Transcript text (stripped). Returns "" on empty or unparsable audio.
    """
    # Live-only: ignore dry_mode (kept for signature compatibility).
    client = OpenAI()

    # OpenAI SDK expects a file-like object; BytesIO works if it has a .name
    buff = io.BytesIO(data)
    # Preserve a filename-ish name to help the server infer format if needed
    buff.name = filename or "audio"

    # Request plain text directly; avoids extra JSON parsing steps
    # If you later prefer a different model variant, change here.
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=buff,
        response_format="text",
    )

    # Some SDK builds return a raw string; others return an object with .text
    if isinstance(resp, str):
        return resp.strip()

    text = getattr(resp, "text", "") or ""
    return str(text).strip()
