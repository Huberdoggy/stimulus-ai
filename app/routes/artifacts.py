from __future__ import annotations

import os
import json
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import JSONResponse

from ..services.ingest import extract_text_from_bytes, normalize_resume_text
from ..services.transcribe import transcribe_audio_bytes

router = APIRouter()

# All saved under app/static/uploads/<kind>/<candidate_id>/*
_APP_STATIC = os.path.join("app", "static")
_UPLOAD_ROOT = os.path.join(_APP_STATIC, "uploads")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_filename(name: Optional[str]) -> str:
    base = (name or "upload.bin").strip()
    # Very light sanitization; keeps extensions intact.
    base = base.replace("\\", "_").replace("/", "_")
    return base or "upload.bin"


@router.post("/resume")
async def upload_resume(
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
    schema_json: Optional[str] = Form(None),
    dry: Optional[int] = Query(None, description="1=stub, 0=live (optional override)")
):
    """
    Upload a resume-like artifact, extract text, and normalize to 'claims'.
    If schema_json is provided, the front-end can map later; for now we just echo back.
    """
    data: bytes = await file.read()
    filename = _safe_filename(file.filename)

    # Save original file
    dest_dir = os.path.join(_UPLOAD_ROOT, "resumes", candidate_id)
    _ensure_dir(dest_dir)
    dest_path = os.path.join(dest_dir, filename)
    with open(dest_path, "wb") as f:
        f.write(data)

    # Extract + normalize
    try:
        text = extract_text_from_bytes(filename, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")

    claims = normalize_resume_text(text)

    result = {
        "ok": True,
        "path": f"/static/uploads/resumes/{candidate_id}/{filename}",
        "text_chars": len(text),
        "claims_count": len(claims),
        "claims": claims,  # keep for now; UI only shows counts
        "dry_mode": None if dry is None else bool(dry),
    }

    # Echo parsed schema if the UI posted one (no mapping done in this route)
    if schema_json:
        try:
            _ = json.loads(schema_json)
            result["schema_received"] = True
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid schema JSON: {e}")

    return JSONResponse(result)


@router.post("/audio")
async def upload_audio(
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
    dry: Optional[int] = Query(None, description="1=stub, 0=live (optional override)")
):
    """
    Upload audio and transcribe. Honors ?dry=0|1; defaults to env if omitted.
    """
    data: bytes = await file.read()
    filename = _safe_filename(file.filename)

    # Save original audio
    dest_dir = os.path.join(_UPLOAD_ROOT, "audio", candidate_id)
    _ensure_dir(dest_dir)
    audio_path = os.path.join(dest_dir, filename)
    with open(audio_path, "wb") as f:
        f.write(data)

    # Transcribe (thread through dry override)
    dry_override: Optional[bool] = None if dry is None else bool(dry)
    transcript: str = transcribe_audio_bytes(filename, data, dry_mode=dry_override)

    # Save transcript
    name, _ext = os.path.splitext(filename)
    txt_name = f"{name}.txt"
    txt_path = os.path.join(dest_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return JSONResponse({
        "ok": True,
        "audio_path": f"/static/uploads/audio/{candidate_id}/{filename}",
        "transcript_path": f"/static/uploads/audio/{candidate_id}/{txt_name}",
        "transcript_chars": len(transcript),
        "preview": transcript[:240],
        "dry_mode": dry_override,
    })
