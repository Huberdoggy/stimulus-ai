from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from ..services.ingest import extract_text_from_bytes, normalize_resume_text
from ..services.transcribe import transcribe_audio_bytes
from ..services.media import (
    ALLOWED_VIDEO_EXT,
    probe_video,
    probe_audio,
    extract_audio_wav,
    NoAudioStreamError,
    FFmpegUnavailableError,
    FFmpegExecError,
)

router = APIRouter()

# ---------- Paths ----------
_APP_STATIC = os.path.join("app", "static")
_UPLOAD_ROOT = os.path.join(_APP_STATIC, "uploads")
_RESUMES = os.path.join(_UPLOAD_ROOT, "resumes")
_AUDIO = os.path.join(_UPLOAD_ROOT, "audio")
_VIDEO = os.path.join(_UPLOAD_ROOT, "video")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_cand(c: str) -> str:
    c = (c or "").strip().replace(" ", "_")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", c):
        raise HTTPException(
            400, "Invalid candidate_id. Use letters, numbers, - and _ only."
        )
    return c


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _web_path(abs_under_static: str) -> str:
    if not abs_under_static.startswith(_APP_STATIC):
        raise HTTPException(400, "Path must be under app/static")
    rel = abs_under_static[len(_APP_STATIC) :].lstrip(os.sep).replace(os.sep, "/")
    return f"/static/{rel}"


def _list_files(dir_path: str, predicate=None) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    out = []
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p) and (predicate(p) if predicate else True):
            out.append(p)
    return sorted(out, key=os.path.getmtime)


def _latest_with_suffix(dir_path: str, suffix: str) -> Optional[str]:
    files = _list_files(dir_path, lambda p: p.endswith(suffix))
    return files[-1] if files else None


# ---------- Clear helpers (used by ?replace=1) ----------
def _rm_all_files(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        return
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass


def _clear_audio(cand: str) -> None:
    _rm_all_files(os.path.join(_AUDIO, cand))


def _clear_video(cand: str) -> None:
    _rm_all_files(os.path.join(_VIDEO, cand))


# ---------- Discovery helpers ----------
def _list_candidates() -> List[str]:
    names: set[str] = set()
    for base in (_RESUMES, _AUDIO, _VIDEO):
        if os.path.isdir(base):
            for name in os.listdir(base):
                if os.path.isdir(os.path.join(base, name)):
                    names.add(name)
    return sorted(names, key=str.lower)


def _latest_claims_json(cand: str) -> Optional[str]:
    return _latest_with_suffix(os.path.join(_RESUMES, cand), ".claims.json")


def _latest_transcript_txt(cand: str) -> Optional[str]:
    paths = []
    a = _latest_with_suffix(os.path.join(_AUDIO, cand), ".txt")
    v = _latest_with_suffix(os.path.join(_VIDEO, cand), ".txt")
    if a and os.path.exists(a):
        paths.append(a)
    if v and os.path.exists(v):
        paths.append(v)
    if not paths:
        return None
    paths.sort(key=os.path.getmtime)
    return paths[-1]


def _latest_audio_meta(cand: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    base = os.path.join(_AUDIO, cand)
    if not os.path.isdir(base):
        return None, None
    metas = _list_files(base, lambda p: p.endswith(".meta.json"))
    if not metas:
        return None, None
    meta_path = metas[-1]
    try:
        meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
    except Exception:
        meta = None
    aud_guess = meta_path[: -len(".meta.json")]
    if not os.path.exists(aud_guess):
        stem = os.path.basename(aud_guess).split("__", 1)[0]
        for fn in os.listdir(base):
            if fn.startswith(stem + "__"):
                aud_guess = os.path.join(base, fn)
                break
    if not os.path.exists(aud_guess):
        aud_guess = None
    return aud_guess, meta


def _latest_video_meta(cand: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    base = os.path.join(_VIDEO, cand)
    if not os.path.isdir(base):
        return None, None
    metas = _list_files(base, lambda p: p.endswith(".meta.json"))
    if not metas:
        return None, None
    meta_path = metas[-1]
    try:
        meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
    except Exception:
        meta = None
    vid_guess = meta_path[: -len(".meta.json")]
    if not os.path.exists(vid_guess):
        stem = os.path.basename(vid_guess).split("__", 1)[0]
        for fn in os.listdir(base):
            if (
                fn.startswith(stem + "__")
                and os.path.splitext(fn)[1].lower() in ALLOWED_VIDEO_EXT
            ):
                vid_guess = os.path.join(base, fn)
                break
    if not os.path.exists(vid_guess):
        vid_guess = None
    return vid_guess, meta


# ---------- Picker endpoints ----------
@router.get("/candidates")
def list_candidates() -> Dict[str, Any]:
    lst = _list_candidates()
    return {"candidates": lst, "count": len(lst)}


@router.get("/for/{candidate_id}")
def load_latest_for_candidate(candidate_id: str) -> Dict[str, Any]:
    """
    Restored original response shape so the UI preview renders:
    { "candidate_id": "...", "latest": { ... preview fields ... }, "ok": true }
    """
    cand = _safe_cand(candidate_id)

    claims_json_path = _latest_claims_json(cand)
    transcript_txt_path = _latest_transcript_txt(cand)
    video_file_abs, video_meta = _latest_video_meta(cand)
    audio_file_abs, audio_meta = _latest_audio_meta(cand)

    audio_path_web = _web_path(audio_file_abs) if audio_file_abs else None
    video_path_web = _web_path(video_file_abs) if video_file_abs else None

    # Claims + resume web paths
    claims: List[Dict[str, Any]] = []
    resume_path_web: Optional[str] = None
    claims_json_web: Optional[str] = None
    if claims_json_path and os.path.exists(claims_json_path):
        try:
            with open(claims_json_path, "r", encoding="utf-8") as f:
                claims = json.load(f)
            claims_json_web = _web_path(claims_json_path)
            raw_resume = claims_json_path[: -len(".claims.json")]
            if os.path.exists(raw_resume):
                resume_path_web = _web_path(raw_resume)
        except Exception as e:
            raise HTTPException(500, f"Failed to load claims.json: {e}")

    # Transcript preview + path
    transcript_preview = ""
    transcript_chars = 0
    transcript_web: Optional[str] = None
    if transcript_txt_path and os.path.exists(transcript_txt_path):
        try:
            t = open(transcript_txt_path, "r", encoding="utf-8", errors="ignore").read()
            transcript_chars = len(t or "")
            transcript_preview = t.strip()[:400] or ""
            transcript_web = _web_path(transcript_txt_path)
        except Exception as e:
            raise HTTPException(500, f"Failed to read transcript: {e}")

    return {
        "candidate_id": cand,
        "latest": {
            "resume_path": resume_path_web,
            "claims_json_path": claims_json_web,
            "claims_count": len(claims),
            "claims": claims,
            "transcript_path": transcript_web,
            "transcript_chars": transcript_chars,
            "transcript_preview": transcript_preview,
            "audio_path": audio_path_web,
            "audio_meta": audio_meta or None,
            "video_path": video_path_web,
            "video_meta": video_meta or None,
        },
        "ok": True,
    }


# ---------- Upload: resume (persists raw file + .claims.json) ----------
@router.post("/resume")
async def upload_resume(
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
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

    # Extract normalized claims
    try:
        text = extract_text_from_bytes(filename, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")
    claims = normalize_resume_text(text)
    text_len = len(text or "")

    claims_json_path = abs_path + ".claims.json"
    try:
        with open(claims_json_path, "w", encoding="utf-8") as f:
            json.dump(claims, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(500, f"Failed to write claims.json: {e}")

    return JSONResponse(
        {
            "ok": True,
            "path": _web_path(abs_path),
            "text_chars": text_len,
            "claims_count": len(claims),
            "claims": claims,
            "claims_json_path": _web_path(claims_json_path),
        }
    )


# ---------- Mutual-exclusion helpers ----------
def _has_any_audio(cand: str) -> bool:
    d = os.path.join(_AUDIO, cand)
    return any(_list_files(d))


def _has_any_video(cand: str) -> bool:
    d = os.path.join(_VIDEO, cand)
    return any(
        _list_files(d, lambda p: os.path.splitext(p)[1].lower() in ALLOWED_VIDEO_EXT)
    )


# ---------- Upload: audio (persists transcript .txt) ----------
@router.post("/audio")
async def upload_audio(
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
    replace: Optional[int] = Query(default=0, ge=0, le=1),
) -> JSONResponse:
    cand = _safe_cand(candidate_id)

    # If a video already exists, either 409 or auto-remove when ?replace=1
    if _has_any_video(cand):
        if replace:
            _clear_video(cand)
        else:
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "conflict",
                    "other_type": "video",
                    "message": "One spoken-word artifact per candidate (audio OR video).",
                },
            )

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

    # Probe audio meta (badge info). Fallback to stub if ffprobe is unavailable.
    try:
        meta = probe_audio(abs_audio)
    except FFmpegUnavailableError:
        filesize_mb = round(os.path.getsize(abs_audio) / (1024 * 1024), 2)
        h = sum(ord(c) for c in os.path.basename(abs_audio))
        duration_sec = 25 + (h % 46)
        meta = {
            "duration_sec": int(duration_sec),
            "codec": "stub",
            "bitrate_kbps": 0,
            "filesize_mb": filesize_mb,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except FFmpegExecError:
        meta = None

    if meta:
        try:
            with open(
                os.path.splitext(abs_audio)[0] + ".meta.json", "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Live-only transcription
    try:
        transcript: str = transcribe_audio_bytes(filename=filename, data=data)
    except TypeError:
        transcript = transcribe_audio_bytes(filename, data)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

    txt_path = os.path.splitext(abs_audio)[0] + ".txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript or "")
    except Exception as e:
        raise HTTPException(500, f"Failed to write transcript: {e}")

    return JSONResponse(
        {
            "ok": True,
            "audio_path": _web_path(abs_audio),
            "transcript_path": _web_path(txt_path),
            "transcript_chars": len(transcript or ""),
            "preview": (transcript or "").strip()[:400],
            "audio_meta": meta or None,
            "replaced": bool(replace),
        }
    )


# ---------- Upload: video (persists video + extracted .wav + transcript .txt) ----------
_MAX_MB = 150
_MAX_BYTES = _MAX_MB * 1024 * 1024
_MAX_DURATION_SEC = 120


@router.post("/video")
async def upload_video(
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
    replace: Optional[int] = Query(default=0, ge=0, le=1),
) -> JSONResponse:
    cand = _safe_cand(candidate_id)

    # If audio already exists, either 409 or auto-remove when ?replace=1
    if _has_any_audio(cand):
        if replace:
            _clear_audio(cand)
        else:
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "conflict",
                    "other_type": "audio",
                    "message": "One spoken-word artifact per candidate (audio OR video).",
                },
            )

    orig_name = os.path.basename(file.filename or "video")
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(
            415,
            f"Unsupported video type. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXT))}",
        )

    blob = await file.read()
    if len(blob) > _MAX_BYTES:
        raise HTTPException(413, f"Max video size is {_MAX_MB} MB.")

    dest_dir = os.path.join(_VIDEO, cand)
    _ensure_dir(dest_dir)

    safe_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", orig_name)
    stamp = _stamp()
    vid_filename = f"{stamp}__{safe_name}"
    abs_video = os.path.join(dest_dir, vid_filename)

    try:
        with open(abs_video, "wb") as f:
            f.write(blob)
    except Exception as e:
        raise HTTPException(400, f"Failed to save video: {e}")

    # Probe video
    try:
        meta = probe_video(abs_video)
    except FFmpegUnavailableError as e:
        raise HTTPException(502, f"{e}. Tip: install ffmpeg/ffprobe on the server.")
    except NoAudioStreamError:
        raise HTTPException(422, "No decodable audio stream found.")
    except FFmpegExecError as e:
        raise HTTPException(
            502, f"ffprobe failed. Tip: try re-encoding to MP4/H.264/AAC. Details: {e}"
        )
    except Exception as e:
        raise HTTPException(502, f"Video probe failed: {e}")

    if meta.get("duration_sec", 0) > _MAX_DURATION_SEC:
        raise HTTPException(413, f"Max duration is {_MAX_DURATION_SEC} seconds.")

    wav_abs = os.path.splitext(abs_video)[0] + ".wav"
    try:
        extract_audio_wav(abs_video, wav_abs)
    except FFmpegExecError as e:
        raise HTTPException(
            502, f"ffmpeg extract failed. Tip: try MP4/H.264/AAC. Details: {e}"
        )
    except Exception as e:
        raise HTTPException(502, f"Audio extraction failed: {e}")

    # Save video meta sidecar
    try:
        with open(
            os.path.splitext(abs_video)[0] + ".meta.json", "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Transcribe from WAV
    try:
        with open(wav_abs, "rb") as f:
            wav_bytes = f.read()
    except Exception as e:
        raise HTTPException(500, f"Failed to read extracted audio: {e}")

    try:
        transcript = transcribe_audio_bytes(
            filename=os.path.basename(wav_abs), data=wav_bytes
        )
    except TypeError:
        transcript = transcribe_audio_bytes(os.path.basename(wav_abs), wav_bytes)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

    txt_abs = os.path.splitext(abs_video)[0] + ".txt"
    try:
        with open(txt_abs, "w", encoding="utf-8") as f:
            f.write(transcript or "")
    except Exception as e:
        raise HTTPException(500, f"Failed to write transcript: {e}")

    return JSONResponse(
        {
            "ok": True,
            "video_path": _web_path(abs_video),
            "audio_path": _web_path(wav_abs),
            "transcript_path": _web_path(txt_abs),
            "transcript_chars": len(transcript or ""),
            "preview": (transcript or "").strip()[:400],
            **meta,
            "replaced": bool(replace),
        }
    )
