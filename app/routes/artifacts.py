from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

# Keep your module layout — these exist in your project
from ..services.ingest import extract_text_from_bytes, normalize_resume_text
from ..services.transcribe import transcribe_audio_bytes  # type: ignore[import]

router = APIRouter()

# ---------- Paths ----------
_APP_STATIC = os.path.join("app", "static")
_UPLOAD_ROOT = os.path.join(_APP_STATIC, "uploads")
_RESUMES = os.path.join(_UPLOAD_ROOT, "resumes")
_AUDIO = os.path.join(_UPLOAD_ROOT, "audio")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _safe_cand(c: str) -> str:
    c = (c or "").strip().replace(" ", "_")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", c):
        raise HTTPException(400, "Invalid candidate_id. Use letters, numbers, - and _ only.")
    return c

def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _web_path(abs_under_static: str) -> str:
    if not abs_under_static.startswith(_APP_STATIC):
        raise HTTPException(400, "Path must be under app/static")
    rel = abs_under_static[len(_APP_STATIC):].lstrip(os.sep).replace(os.sep, "/")
    return f"/static/{rel}"

# ---------- Discovery helpers (for UI picker) ----------
def _list_candidates() -> List[str]:
    names: set[str] = set()
    for base in (_RESUMES, _AUDIO):
        if os.path.isdir(base):
            for name in os.listdir(base):
                if os.path.isdir(os.path.join(base, name)):
                    names.add(name)
    return sorted(names, key=str.lower)

def _latest_claims_json(cand: str) -> Optional[str]:
    base = os.path.join(_RESUMES, cand)
    if not os.path.isdir(base):
        return None
    best: Optional[str] = None
    best_m = -1.0
    for fn in os.listdir(base):
        if fn.endswith(".claims.json"):
            p = os.path.join(base, fn)
            m = os.path.getmtime(p)
            if m > best_m:
                best, best_m = p, m
    return best

def _latest_transcript_txt(cand: str) -> Optional[str]:
    base = os.path.join(_AUDIO, cand)
    if not os.path.isdir(base):
        return None
    best: Optional[str] = None
    best_m = -1.0
    for fn in os.listdir(base):
        if fn.endswith(".txt"):
            p = os.path.join(base, fn)
            m = os.path.getmtime(p)
            if m > best_m:
                best, best_m = p, m
    return best

# ---------- Picker endpoints ----------
@router.get("/candidates")
def list_candidates() -> Dict[str, Any]:
    lst = _list_candidates()
    return {"candidates": lst, "count": len(lst)}

@router.get("/for/{candidate_id}")
def load_latest_for_candidate(candidate_id: str) -> Dict[str, Any]:
    cand = _safe_cand(candidate_id)

    claims_json_path = _latest_claims_json(cand)
    transcript_txt_path = _latest_transcript_txt(cand)

    claims: List[Dict[str, Any]] = []
    resume_path_web: Optional[str] = None
    claims_json_web: Optional[str] = None

    if claims_json_path and os.path.exists(claims_json_path):
        try:
            with open(claims_json_path, "r", encoding="utf-8") as f:
                claims = json.load(f)
            claims_json_web = _web_path(claims_json_path)
            # Infer original resume filename (strip ".claims.json")
            raw_resume = claims_json_path[:-len(".claims.json")]
            if os.path.exists(raw_resume):
                resume_path_web = _web_path(raw_resume)
        except Exception as e:
            raise HTTPException(500, f"Failed to load claims.json: {e}")

    transcript_preview = ""
    transcript_chars = 0
    transcript_web: Optional[str] = None
    if transcript_txt_path and os.path.exists(transcript_txt_path):
        try:
            t = open(transcript_txt_path, "r", encoding="utf-8", errors="ignore").read()
            transcript_chars = len(t)
            transcript_preview = (t.strip()[:400] or "")
            transcript_web = _web_path(transcript_txt_path)
        except Exception as e:
            raise HTTPException(500, f"Failed to read transcript: {e}")

    return {
        "candidate_id": cand,
        "latest": {
            "resume_path": resume_path_web,
            "claims_json_path": claims_json_web,
            "claims_count": len(claims),
            "claims": claims,  # provided so the UI can build evidence without re-upload
            "transcript_path": transcript_web,
            "transcript_chars": transcript_chars,
            "transcript_preview": transcript_preview,
        },
        "ok": True
    }

# ---------- Upload: resume (persists claims.json) ----------
@router.post("/resume")
async def upload_resume(
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
    dry: Optional[int] = Query(default=1, ge=0, le=1),
) -> JSONResponse:
    cand = _safe_cand(candidate_id)
    dest_dir = os.path.join(_RESUMES, cand)
    _ensure_dir(dest_dir)

    orig_name = os.path.basename(file.filename or "resume")
    safe_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", orig_name)
    filename = f"{_stamp()}__{safe_name}"
    abs_path = os.path.join(dest_dir, filename)

    try:
        data = await file.read()
        with open(abs_path, "wb") as f:
            f.write(data)
    except Exception as e:
        raise HTTPException(400, f"Failed to save resume: {e}")

    # Extract + normalize
    try:
        text = extract_text_from_bytes(filename, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")
    claims = normalize_resume_text(text)

    # Persist claims.json next to the resume
    claims_json_path = abs_path + ".claims.json"
    try:
        with open(claims_json_path, "w", encoding="utf-8") as f:
            json.dump(claims, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(500, f"Failed to write claims.json: {e}")

    return JSONResponse({
        "ok": True,
        "path": _web_path(abs_path),
        "text_chars": len(text),
        "claims_count": len(claims),
        "claims": claims,
        "claims_json_path": _web_path(claims_json_path),
        "dry_mode": None if dry is None else bool(dry),
    })

# ---------- Upload: audio (persists transcript .txt) ----------
@router.post("/audio")
async def upload_audio(
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
    dry: Optional[int] = Query(default=1, ge=0, le=1),
) -> JSONResponse:
    cand = _safe_cand(candidate_id)
    dest_dir = os.path.join(_AUDIO, cand)
    _ensure_dir(dest_dir)

    orig_name = os.path.basename(file.filename or "audio")
    safe_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", orig_name)
    filename = f"{_stamp()}__{safe_name}"
    abs_audio = os.path.join(dest_dir, filename)

    try:
        data = await file.read()
        with open(abs_audio, "wb") as f:
            f.write(data)
    except Exception as e:
        raise HTTPException(400, f"Failed to save audio: {e}")

    # Correct signature: transcribe_audio_bytes(filename: str, data: bytes, *, dry_mode: Optional[bool])
    dry_override: Optional[bool] = None if dry is None else bool(dry)
    try:
        transcript: str = transcribe_audio_bytes(filename=filename, data=data, dry_mode=dry_override)
    except TypeError:
        # Fallback for prior signature variants: (filename, data) or (filename, data, dry)
        try:
            transcript = transcribe_audio_bytes(filename, data)  # type: ignore[misc]
        except Exception as e2:
            raise HTTPException(500, f"Transcription failed: {e2}")
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

    txt_path = os.path.splitext(abs_audio)[0] + ".txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript or "")
    except Exception as e:
        raise HTTPException(500, f"Failed to write transcript: {e}")

    return JSONResponse({
        "ok": True,
        "audio_path": _web_path(abs_audio),
        "transcript_path": _web_path(txt_path),
        "transcript_chars": len(transcript or ""),
        "preview": (transcript or "").strip()[:400],
        "dry_mode": dry_override,
    })
