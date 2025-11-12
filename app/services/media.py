from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Dict, Set

ALLOWED_VIDEO_EXT: Set[str] = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


class FFmpegUnavailableError(RuntimeError): ...


class NoAudioStreamError(RuntimeError): ...


class FFmpegExecError(RuntimeError): ...


def _need(tool: str):
    if not shutil.which(tool):
        raise FFmpegUnavailableError(f"{tool} not available on PATH")


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as e:
        raise FFmpegExecError(e.output.decode("utf-8", errors="ignore"))


def probe_video(path: str) -> Dict:
    """
    Returns {duration_sec, codec, bitrate_kbps, filesize_mb, created_at}
    Raises NoAudioStreamError if no decodable audio stream is present.
    """
    _need("ffprobe")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # format (duration + size)
    fmt = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            path,
        ]
    )
    fmt_j = json.loads(fmt or "{}").get("format", {}) if fmt else {}
    dur = float(fmt_j.get("duration") or 0.0)
    size_bytes = int(fmt_j.get("size") or 0)
    mb = round(size_bytes / (1024 * 1024), 2)

    # video codec (v:0)
    v = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,bit_rate",
            "-of",
            "json",
            path,
        ]
    )
    vj = json.loads(v or "{}").get("streams", []) if v else []
    vcodec = (vj[0].get("codec_name") if vj else None) or "unknown"
    vbit = int((vj[0].get("bit_rate") or 0) if vj else 0)

    # audio probe (a:0) — ensure exists
    a = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,bit_rate",
            "-of",
            "json",
            path,
        ]
    )
    aj = json.loads(a or "{}").get("streams", []) if a else []
    if not aj:
        raise NoAudioStreamError("No audio stream")

    abit = int((aj[0].get("bit_rate") or 0) if aj else 0)

    # bitrate heuristic
    kbps = 0
    if vbit or abit:
        kbps = int(round((vbit + abit) / 1000)) if (vbit or abit) else 0
    elif dur > 0 and size_bytes > 0:
        kbps = int(round((size_bytes * 8) / 1000 / dur))

    return {
        "duration_sec": int(round(dur)),
        "codec": vcodec,
        "bitrate_kbps": kbps,
        "filesize_mb": mb,
        "created_at": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
        ),
    }


def probe_audio(path: str) -> Dict:
    """
    Lightweight audio probe using ffprobe.
    Returns {duration_sec, codec, bitrate_kbps, filesize_mb, created_at}
    Raises NoAudioStreamError if no decodable audio stream exists.
    """
    _need("ffprobe")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # container duration + size
    fmt = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            path,
        ]
    )
    fmt_j = json.loads(fmt or "{}").get("format", {}) if fmt else {}
    dur = float(fmt_j.get("duration") or 0.0)
    size_bytes = int(fmt_j.get("size") or 0)
    mb = round(size_bytes / (1024 * 1024), 2)

    # audio stream info
    a = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,bit_rate",
            "-of",
            "json",
            path,
        ]
    )
    aj = json.loads(a or "{}").get("streams", []) if a else []
    if not aj:
        raise NoAudioStreamError("No audio stream")
    acodec = (aj[0].get("codec_name") if aj else None) or "unknown"
    abit = int((aj[0].get("bit_rate") or 0) if aj else 0)
    kbps = (
        int(round(abit / 1000))
        if abit
        else (int(round((size_bytes * 8) / 1000 / dur)) if (size_bytes and dur) else 0)
    )

    return {
        "duration_sec": int(round(dur)),
        "codec": acodec,
        "bitrate_kbps": kbps,
        "filesize_mb": mb,
        "created_at": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
        ),
    }


def extract_audio_wav(input_path: str, output_path: str) -> None:
    """
    Extract mono, 16kHz PCM WAV via ffmpeg.
    """
    _need("ffmpeg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        output_path,
    ]
    _run(cmd)
