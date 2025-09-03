from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI

# Router keeps its own prefix; main.py includes WITHOUT an extra prefix.
router = APIRouter(prefix="/evidence", tags=["evidence"])

# ---------- Project paths (robust) ----------
def _find_app_dir(here: Path) -> Path:
    # Walk up until we hit a folder literally named "app"
    for p in [here] + list(here.parents):
        if p.name == "app":
            return p
    # Fallback: assume .../app two levels up (repo/app)
    return here.parents[1] / "app"

HERE = Path(__file__).resolve()
APP_DIR = _find_app_dir(HERE)               # .../app
ROOT_DIR = APP_DIR.parent                   # repo root
STATIC_DIR = APP_DIR / "static"             # .../app/static
UPLOADS_DIR = STATIC_DIR / "uploads"
RESUME_DIR = UPLOADS_DIR / "resumes"
AUDIO_DIR = UPLOADS_DIR / "audio"

# ---------- Env / model selection ----------
def truthy(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    v = v.strip().strip('"').strip("'").lower()
    return v in {"1", "true", "yes", "on", "y"}

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MODEL_LIVE = os.getenv("LLM_MODEL_LIVE", "") or LLM_MODEL
DRY_MODE_DEFAULT = truthy("DRY_MODE", "1")

# ---------- Validation ----------
ALLOWED_CAND = re.compile(r"^[a-z0-9_\-]+$", re.IGNORECASE)

def _safe_cand(cand: str) -> str:
    cand = (cand or "").strip().lower().replace(" ", "_")
    if not ALLOWED_CAND.fullmatch(cand):
        raise HTTPException(400, "Invalid candidate_id.")
    return cand

# ---------- Artifact discovery ----------
def _latest_claims_json(cand: str) -> Optional[Path]:
    base = RESUME_DIR / cand
    if not base.exists():
        return None
    best: Optional[Path] = None
    best_mtime = -1.0
    for p in base.iterdir():
        if p.is_file() and p.name.endswith(".claims.json"):
            m = p.stat().st_mtime
            if m > best_mtime:
                best, best_mtime = p, m
    return best

def _latest_transcript_txt(cand: str) -> Optional[Path]:
    base = AUDIO_DIR / cand
    if not base.exists():
        return None
    best: Optional[Path] = None
    best_mtime = -1.0
    for p in base.iterdir():
        if p.is_file() and p.suffix == ".txt":
            m = p.stat().st_mtime
            if m > best_mtime:
                best, best_mtime = p, m
    return best

def _read_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(500, f"Failed to read {fp.name}: {e}")

def _load_resume_claims(fp: Path) -> List[Dict[str, Any]]:
    try:
        items = json.loads(_read_text(fp))
    except Exception as e:
        raise HTTPException(500, f"Failed to parse claims.json: {e}")
    out: List[Dict[str, Any]] = []
    for c in items if isinstance(items, list) else []:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        sref = c.get("source_ref") or {}
        line = sref.get("line")
        cid = str(c.get("id") or f"r{len(out)+1:04d}")
        if not text:
            continue
        out.append({"id": cid, "text": text, "source": "resume", "line": line})
    return out

# ---------- Transcript → claims ----------
_SENT_SPLIT = re.compile(
    r"(?<=[\.\!\?])\s+|[\r\n]+|(?:\n[\-\*\u2022]\s+)",  # sentence end, any newline, or bullet line
    re.UNICODE,
)

def _chunk_text(s: str, max_len: int = 280, min_len: int = 60) -> List[str]:
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    if len(s) <= max_len:
        return [s]
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        j = min(i + max_len, n)
        cut = s.rfind(" ", i + min_len, j)
        if cut == -1:
            cut = j
        out.append(s[i:cut].strip())
        i = cut
    return [t for t in out if len(t) >= min_len or t == out[-1]]

def _sentences(text: str) -> List[str]:
    raw = [re.sub(r"\s+", " ", (seg or "").strip()) for seg in _SENT_SPLIT.split(text or "")]
    raw = [s for s in raw if s]
    out: List[str] = []
    for s in raw:
        if len(s) < 20:
            continue
        if len(s) <= 400:
            out.append(s)
        else:
            out.extend(_chunk_text(s, max_len=280, min_len=60))
    # de-dup adjacent
    dedup: List[str] = []
    for s in out:
        if not dedup or s != dedup[-1]:
            dedup.append(s)
    if not dedup and (text or "").strip():
        first = re.sub(r"\s+", " ", (text or "")).strip()[:240]
        if len(first) >= 40:
            dedup.append(first)
    return dedup[:200]

def _resolve_static_web_path(web_path: str) -> Optional[Path]:
    """
    Accept '/static/uploads/audio/...txt' or 'app/static/uploads/audio/...txt'.
    Return a filesystem Path if it exists under app/static/uploads/audio, else None.
    """
    if not web_path:
        return None
    rel = unquote(web_path).lstrip("/")
    if rel.startswith("app/"):
        rel = rel[4:]
    path = (APP_DIR / rel).resolve()
    audio_root = AUDIO_DIR.resolve()
    if not str(path).startswith(str(audio_root)) or not path.exists():
        return None
    return path

def _claims_from_transcript_file(fp: Path) -> List[Dict[str, Any]]:
    txt = _read_text(fp)
    claims: List[Dict[str, Any]] = []
    for idx, s in enumerate(_sentences(txt), start=1):
        claims.append({"id": f"t{idx:04d}", "text": s, "source": "transcript", "line": idx})
    return claims

# ---------- Request model ----------
class EvidenceIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    candidate_id: str
    jd_schema: Dict[str, Any] = Field(default_factory=dict, alias="schema")
    claims: Optional[List[Dict[str, Any]]] = None
    transcript_path: Optional[str] = None

# ---------- LLM prompt ----------
SYSTEM_PROMPT = (
    "You are a JSON-only evidence matcher. You receive:\n"
    "(1) a JD schema with themes and requirements, and\n"
    "(2) a list of candidate claims (id, text, source: \"resume\" or \"transcript\", and optional line).\n"
    "Your job: For EACH requirement in EACH theme, select up to TWO supporting claims from the list and produce:\n"
    "• evidence: array of {claim_id, hit_terms[]} where hit_terms are the EXACT tokens or short substrings copied verbatim from that claim’s text that justify the match.\n"
    "• open_questions: array of up to 2 concise questions to clarify gaps ONLY if evidence is weak or missing.\n\n"
    "Strict rules:\n"
    "• Do not invent text. Only use claims provided.\n"
    "• Prefer source diversity when two candidates are comparable: if both resume and transcript fit, choose one of each.\n"
    "• Keep hit_terms minimal and meaningful: 1–8 items, each 1–40 chars, must appear verbatim in the claim text (case-insensitive allowed) and relate directly to the requirement.\n"
    "• Limit evidence to 0–2 items per requirement.\n"
    "• If multiple candidates are equally good, tie-break deterministically in this order:\n"
    "  (a) higher semantic relevance to the requirement,\n"
    "  (b) source diversity (resume+transcript over two of the same),\n"
    "  (c) earlier occurrence in the provided claims array.\n"
    "• If you select 0 evidence items for a requirement, include at least 1 open question for that requirement.\n"
    "• Adapter keys MUST be the EXACT theme names from the provided schema.\n"
    "• GLOBAL CONSTRAINT: If any transcript claims are present in the input, you MUST include at least one transcript-based evidence item somewhere in the output. If all transcript claims are weak, select the single most relevant transcript claim and include an open question explaining the gap.\n"
    "• Output JSON ONLY. No extra keys, no markdown.\n\n"
    "Output JSON schema (theme keys must be actual names):\n"
    "{\n"
    "  \"adapter\": {\n"
    "    \"<Theme Name>\": [\n"
    "      {\n"
    "        \"requirement\": \"\",\n"
    "        \"evidence\": [\n"
    "          {\"claim_id\": \"\", \"hit_terms\": [\"\", \"\", \"…\"]}\n"
    "        ],\n"
    "        \"open_questions\": [\"\", \"\"]\n"
    "      }\n"
    "    ]\n"
    "  }\n"
    "}"
)

def _build_llm_payload(jd_schema: Dict[str, Any], llm_claims: List[Dict[str, Any]], theme_names: List[str]) -> str:
    payload = {
        "schema": {
            "themes": [
                {"name": (t.get("name") or "").strip(), "requirements": list(t.get("requirements") or [])}
                for t in (jd_schema.get("themes") or [])
            ]
        },
        "claims": [
            {k: v for k, v in c.items() if k in {"id", "text", "source", "line"}}
            for c in llm_claims
        ],
        "_theme_keys": theme_names,
        "_sources_present": {
            "resume": sum(1 for c in llm_claims if c.get("source") == "resume"),
            "transcript": sum(1 for c in llm_claims if c.get("source") == "transcript"),
        },
    }
    return json.dumps(payload, ensure_ascii=False)

_CID_RE = re.compile(r"^\s*([rt])\s*0*(\d+)\s*$", re.I)
def _normalize_cid(cid: Any) -> Optional[str]:
    if cid is None:
        return None
    s = str(cid).strip().lower()
    m = _CID_RE.match(s)
    if not m:
        return s or None
    prefix, num = m.group(1), int(m.group(2))
    return f"{prefix}{num:04d}"

def _resolve_and_score(adapter_in: Dict[str, Any], jd: Dict[str, Any], id2claim: Dict[str, Dict[str, Any]]):
    adapter: Dict[str, List[Dict[str, Any]]] = {}
    total_reqs = 0
    hit_reqs = 0
    by_theme: Dict[str, float] = {}
    used_any_transcript = False

    for t in (jd.get("themes") or []):
        tname = (t.get("name") or "").strip() or "Theme"
        rows_llm = adapter_in.get(tname) or []
        if not isinstance(rows_llm, list):
            rows_llm = []
        out_rows: List[Dict[str, Any]] = []

        for row in rows_llm:
            requirement = row.get("requirement")
            evs = []
            for e in row.get("evidence", []):
                raw_cid = e.get("claim_id")
                cid = _normalize_cid(raw_cid)
                c = id2claim.get(cid or "") or id2claim.get(str(raw_cid) if raw_cid is not None else "")
                if not c:
                    continue
                hits = e.get("hit_terms") or []
                kind = c.get("source")
                if kind == "transcript":
                    used_any_transcript = True
                source_ref = {"kind": kind, "line": c.get("line")}
                evs.append({
                    "source_ref": source_ref,
                    "snippet": c.get("text", ""),
                    "hit_terms": [h for h in hits if isinstance(h, str) and h.strip()][:8],
                })
            oq = [q for q in (row.get("open_questions") or []) if isinstance(q, str) and q.strip()][:2]
            out_rows.append({
                "requirement": requirement,
                "evidence": evs[:2],
                "open_questions": oq,
            })

        total_reqs += len(out_rows)
        theme_hits = sum(1 for r in out_rows if (r.get("evidence") or []))
        by_theme[tname] = round((theme_hits / max(1, len(out_rows))) * 100.0, 1)
        hit_reqs += theme_hits
        adapter[tname] = out_rows

    overall = round((hit_reqs / max(1, total_reqs)) * 100.0, 1)
    coverage = {"by_theme": by_theme, "overall": overall}
    counts = {"requirements": total_reqs, "hit": hit_reqs}
    return adapter, coverage, counts, used_any_transcript

# ---------- Core route ----------
@router.post("/build")
def build_evidence(
    payload: EvidenceIn,
    dry: int | None = Query(None, description="1=stub, 0=live"),
):
    cand = _safe_cand(payload.candidate_id)
    jd = payload.jd_schema or {}
    themes = jd.get("themes") or []
    if not isinstance(themes, list) or not themes:
        raise HTTPException(status_code=400, detail="Schema missing themes.")

    # Resume claims
    if payload.claims and len(payload.claims) > 0:
        resume_claims: List[Dict[str, Any]] = []
        for c in payload.claims:
            text = (c.get("text") or "").strip()
            if not text:
                continue
            sref = c.get("source_ref") or {}
            src = (sref.get("kind") or c.get("source") or "resume").lower()
            line = sref.get("line", c.get("line"))
            cid = str(c.get("id") or f"r{len(resume_claims)+1:04d}")
            resume_claims.append({"id": cid, "text": text, "source": src, "line": line})
    else:
        resume_claims = []
        cjson = _latest_claims_json(cand)
        if cjson:
            resume_claims = _load_resume_claims(cjson)

    # Transcript: resolve provided path; if it yields zero claims, fall back to latest
    transcript_claims: List[Dict[str, Any]] = []
    if payload.transcript_path:
        p = _resolve_static_web_path(payload.transcript_path)
        if p:
            transcript_claims = _claims_from_transcript_file(p)
    if not transcript_claims:
        ttxt = _latest_transcript_txt(cand)
        if ttxt:
            transcript_claims = _claims_from_transcript_file(ttxt)

    # Merge (transcript first to help diversity tie-break)
    claims_llm = transcript_claims + resume_claims
    id2claim: Dict[str, Dict[str, Any]] = {c["id"]: c for c in claims_llm}
    theme_names = [(t.get("name") or "").strip() or "Theme" for t in themes]
    transcript_exists = any(c.get("source") == "transcript" for c in claims_llm)

    # DRY vs LIVE
    use_dry = DRY_MODE_DEFAULT if dry is None else bool(dry)
    if use_dry:
        adapter: Dict[str, List[Dict[str, Any]]] = {}
        total_reqs = 0
        for t in themes:
            tname = (t.get("name") or "").strip() or "Theme"
            reqs = [r for r in (t.get("requirements") or []) if str(r).strip()]
            total_reqs += len(reqs)
            adapter[tname] = [{"requirement": r, "evidence": [], "open_questions": []} for r in reqs]
        coverage = {"by_theme": {k: 0.0 for k in adapter.keys()}, "overall": 0.0}
        counts = {"requirements": total_reqs, "hit": 0}
        return {"adapter": adapter, "coverage": coverage, "counts": counts}

    # Live LLM call (JSON-only) with optional nudge
    client = OpenAI()
    model = LLM_MODEL_LIVE or LLM_MODEL

    def call_llm(extra_hint: str = "") -> Dict[str, Any]:
        user_payload = _build_llm_payload(jd, claims_llm, theme_names)
        system_hint = (
            SYSTEM_PROMPT
            + "\n\nUse these exact theme keys: " + json.dumps(theme_names, ensure_ascii=False)
            + "\nSources present summary: "
            + json.dumps(
                {
                    "resume": sum(1 for c in claims_llm if c.get("source") == "resume"),
                    "transcript": sum(1 for c in claims_llm if c.get("source") == "transcript"),
                },
                ensure_ascii=False,
            )
            + (("\n" + extra_hint) if extra_hint else "")
        )
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            temperature=0,
            top_p=1,
            messages=[
                {"role": "system", "content": system_hint},
                {
                    "role": "user",
                    "content": (
                        'Below is the payload. Use it as the only source of truth. '
                        'Return only the JSON specified in the System Prompt’s "Output JSON schema".\n\n'
                        + user_payload
                    ),
                },
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not m:
                raise HTTPException(502, "LLM did not return valid JSON.")
            return json.loads(m.group(0))

    data = call_llm()
    adapter_in = data.get("adapter") or {}
    if not isinstance(adapter_in, dict):
        raise HTTPException(502, "LLM JSON missing 'adapter' object.")

    expected_keys = set(theme_names)
    got_keys = set(adapter_in.keys())
    if not expected_keys.intersection(got_keys):
        raise HTTPException(502, f"LLM adapter keys mismatch. Got {sorted(got_keys)}; expected some of {sorted(expected_keys)}.")

    adapter, coverage, counts, used_any_transcript = _resolve_and_score(adapter_in, jd, id2claim)

    # Optional nudge if no transcript was used but exists
    if transcript_exists and not used_any_transcript:
        data2 = call_llm(
            extra_hint="Your previous JSON contained zero transcript-based evidence even though transcript claims were provided. "
                       "Revise and return JSON that includes at least one transcript-based evidence item (choose the most relevant)."
        )
        adapter_in2 = data2.get("adapter") or {}
        if expected_keys.intersection(set(adapter_in2.keys())):
            adapter2, coverage2, counts2, used_any_transcript2 = _resolve_and_score(adapter_in2, jd, id2claim)
            if used_any_transcript2:
                adapter, coverage, counts = adapter2, coverage2, counts2

    return {"adapter": adapter, "coverage": coverage, "counts": counts}
