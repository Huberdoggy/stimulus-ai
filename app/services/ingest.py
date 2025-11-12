from __future__ import annotations

import io
import os
import re
import hashlib
from typing import Dict, List, Optional

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    from docx import Document as DocxDocument  # type: ignore
except Exception:
    DocxDocument = None

_STOP = {
    "the",
    "and",
    "of",
    "to",
    "a",
    "in",
    "for",
    "on",
    "as",
    "with",
    "by",
    "or",
    "an",
    "be",
    "at",
    "from",
    "is",
    "are",
    "was",
    "were",
    "it",
    "that",
    "this",
    "these",
    "those",
}


def _ext(name: Optional[str]) -> str:
    if not name:
        return ""
    return os.path.splitext(name)[1].lower()


def extract_text_from_bytes(filename: Optional[str], data: bytes) -> str:
    ext = _ext(filename)
    if ext == ".pdf" and PdfReader:
        buff = io.BytesIO(data)
        reader = PdfReader(buff)
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)
    if ext == ".docx" and DocxDocument:
        buff = io.BytesIO(data)
        doc = DocxDocument(buff)
        return "\n".join(p.text for p in doc.paragraphs)
    # fallback: assume utf-8 text-like content
    return data.decode("utf-8", errors="ignore")


def _mk_claim(text: str, line_no: int) -> Dict:
    cid = hashlib.sha1(f"{line_no}:{text}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": cid,
        "text": text,
        "source_ref": {"kind": "resume", "line": line_no},
        "tags": [],
    }


def _dedupe_claims(claims: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for c in claims:
        key = c["text"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def normalize_resume_text(text: str) -> List[Dict]:
    """
    Produce atomic 'claims' with minimal structure:
    [{id, text, source_ref:{kind:'resume', line:int}}]
    """
    claims: List[Dict] = []
    lines = [ln.strip() for ln in text.splitlines()]
    line_no = 0
    for ln in lines:
        line_no += 1
        if not ln:
            continue
        # prefer bullet- or sentence-like units between 30 and 240 chars
        if len(ln) < 30 and not re.search(r"[•\-–]\s+\w", ln):
            continue
        if len(ln) > 240:
            # split mega-lines on sentence boundaries
            parts = re.split(r"(?<=[\.\!\?])\s+", ln)
            for p in parts:
                p = p.strip()
                if 30 <= len(p) <= 240:
                    claims.append(_mk_claim(p, line_no))
        else:
            claims.append(_mk_claim(ln, line_no))
    return _dedupe_claims(claims)
