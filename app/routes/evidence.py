from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict  # Pydantic v2

# Router owns its prefix; main.py should include this WITHOUT an extra prefix.
router = APIRouter(prefix="/evidence", tags=["evidence"])

# ---------- Project paths ----------
APP_DIR = Path(__file__).resolve().parents[2]  # .../app/routes -> app/
STATIC_DIR = APP_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESUME_DIR = UPLOADS_DIR / "resumes"
AUDIO_DIR = UPLOADS_DIR / "audio"

ALLOWED_CAND = re.compile(r"^[a-z0-9_\-]+$", re.IGNORECASE)

# ---------- Text + token utils (deterministic, no model calls) ----------
_STOP = {
    "the","and","of","to","a","in","for","on","as","with","by","or","an","be","at","from",
    "is","are","was","were","it","that","this","these","those","you","we","they","i","our",
    "their","your","will","can","must","should","may","but","not","than","then","so","if",
    "while","when","where","which","who","whom","into","over","under","between","within",
}

# phrase-level canonical replacements done *before* tokenization
_PHRASES = [
    (re.compile(r"\blarge language models?\b", re.I), " llm "),
    (re.compile(r"\bregular expressions?\b", re.I),    " regex "),
    (re.compile(r"\bcommand[-\s]?line\b", re.I),       " cli "),
    (re.compile(r"\bhigh[-\s]?school\b", re.I),        " highschool "),
    (re.compile(r"\bquality checks?\b", re.I),         " audit "),
    (re.compile(r"\bdata markups?\b", re.I),           " annotation "),
    (re.compile(r"\bdata analysis\b", re.I),           " analysis "),
    (re.compile(r"\bU\.?S\.?(?:-based)?\b", re.I),     " us "),
]

# token-level canonical mapping (after tokenization)
_CANON = {
    # annotation / labeling
    "annotations":"annotate","annotation":"annotate","annotating":"annotate","annotate":"annotate",
    "label":"annotate","labels":"annotate","labeling":"annotate","markup":"annotate","markups":"annotate",
    # writing / grammar
    "writing":"write","written":"write","wrote":"write","writes":"write","write":"write",
    "grammar":"grammar","grammatically":"grammar",
    # audit / quality
    "audits":"audit","audit":"audit","qa":"audit","quality":"quality","check":"audit","checks":"audit",
    # improvements
    "improvements":"improve","improvement":"improve","improving":"improve","improve":"improve",
    "tooling":"tool","tools":"tool","tool":"tool","bug":"bug","bugs":"bug","report":"report","reporting":"report",
    "suggest":"suggest","suggestions":"suggest","suggesting":"suggest",
    # decisions / ambiguity
    "judgment":"judgment","judgements":"judgment","judgments":"judgment","decision":"decision","decisions":"decision",
    "logical":"logic","logic":"logic","ambiguity":"ambiguous","ambiguous":"ambiguous",
    # independence / ownership / depth
    "independently":"independent","independent":"independent","ownership":"own","owning":"own","own":"own",
    "dive":"deep","diving":"deep","deep":"deep",
    # skills / tech
    "python":"python","unix":"unix","cli":"cli","html":"html","xml":"xml","markdown":"markdown","json":"json",
    "csv":"csv","rtf":"rtf","regex":"regex","llm":"llm","cefr":"cefr","c2":"c2",
    # culture / research
    "research":"research","synthesize":"synthesize","plagiarism":"plagiarism","integrity":"integrity",
    "culture":"culture","society":"society","norms":"norms","us":"us",
    # misc
    "technical":"technical","science":"science","stem":"stem","evaluation":"evaluation","testing":"testing",
}

# single-token "skills" that can match with just 1 overlap (crisp tokens)
_SKILL_TOKENS = {"python","unix","cli","html","xml","markdown","json","csv","rtf","regex","llm","cefr","c2"}

def _pre_normalize(text: str) -> str:
    s = " " + (text or "") + " "
    for pat, repl in _PHRASES:
        s = pat.sub(repl, s)
    return s

def _stemish(t: str) -> str:
    # Light stemming: plurals/verb endings, keep ≥3 chars
    for suf in ("ing","ized","ises","ers","ies","ied","izes","ings","ed","es","s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t

def _tok(s: str) -> List[str]:
    s = _pre_normalize(s).lower()
    raw = re.findall(r"[a-z0-9\+\#]+", s)  # allow c2, cli, json, etc.
    out: List[str] = []
    for r in raw:
        if r in _STOP or len(r) < 2:
            continue
        t = _CANON.get(r, r)
        t = _CANON.get(_stemish(t), t)  # try mapping again after stem
        if t and t not in _STOP and len(t) >= 2:
            out.append(t)
    return out

def _sentences(text: str) -> List[str]:
    raw = re.split(r"(?:\n[\-\*\u2022]\s+)|(?<=[\.\!\?])\s+", (text or "").strip())
    out: List[str] = []
    for s in raw:
        s = re.sub(r"\s+", " ", (s or "").strip())
        if 30 <= len(s) <= 300:
            out.append(s)
    # de-dup adjacent
    dedup: List[str] = []
    for s in out:
        if not dedup or s != dedup[-1]:
            dedup.append(s)
    return dedup

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

# ---------- Artifact loading (fallbacks) ----------
def _safe_cand(cand: str) -> str:
    cand = (cand or "").strip().lower().replace(" ", "_")
    if not ALLOWED_CAND.fullmatch(cand):
        raise HTTPException(400, "Invalid candidate_id.")
    return cand

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

def _load_claims_from_file(fp: Path) -> List[Dict[str, Any]]:
    try:
        items = json.loads(fp.read_text(encoding="utf-8"))
        out: List[Dict[str, Any]] = []
        for c in items if isinstance(items, list) else []:
            if isinstance(c, dict) and "text" in c:
                out.append(c)
        return out
    except Exception as e:
        raise HTTPException(500, f"Failed to read claims.json: {e}")

def _read_transcript_txt(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(500, f"Failed to read transcript: {e}")

def _claims_from_transcript_webpath(candidate_id: str, transcript_path: Optional[str]) -> List[Dict[str, Any]]:
    """Accept a web path like /static/uploads/audio/<cand>/<file>.txt, validate, read."""
    if not transcript_path:
        return []
    rel = transcript_path.lstrip("/")
    expected_root = (STATIC_DIR / "uploads" / "audio" / candidate_id).resolve()
    path = (APP_DIR / rel).resolve()
    if not str(path).startswith(str(expected_root)):
        raise HTTPException(400, "Invalid transcript path.")
    if not path.exists():
        return []
    txt = _read_transcript_txt(path)
    claims = []
    for idx, sentence in enumerate(_sentences(txt), start=1):
        claims.append({
            "id": f"t{idx:04d}",
            "text": sentence,
            "source_ref": {"kind": "transcript", "line": idx},
            "tags": [],
        })
    return claims

# ---------- Request model (Pydantic v2) ----------
class EvidenceIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    candidate_id: str
    jd_schema: Dict[str, Any] = Field(default_factory=dict, alias="schema")  # accept JSON key "schema"
    claims: Optional[List[Dict[str, Any]]] = None
    transcript_path: Optional[str] = None

# ---------- Matching ----------
def _score_requirement(req: str, claim_pairs: List[Tuple[Dict[str, Any], List[str]]]) -> List[Dict[str, Any]]:
    """Return top evidence rows (snippet+source_ref+score) for a requirement."""
    rtoks = _tok(req)
    if not rtoks:
        return []

    req_lc = _pre_normalize(req).lower()

    best: List[Dict[str, Any]] = []
    for c, ctoks in claim_pairs:
        if not ctoks:
            continue
        text = (c.get("text") or "")
        text_lc = _pre_normalize(text).lower()

        # base similarity
        score = _jaccard(rtoks, ctoks)

        # substring presence boost (strong lexical hint)
        if len(req_lc) >= 8 and req_lc in text_lc:
            score = max(score, 0.9)

        # small boost if any skill token overlaps (skills are crisp)
        if set(rtoks) & _SKILL_TOKENS & set(ctoks):
            score = max(score, 0.5, score)

        # gate: jaccard or ≥2 overlaps or 1 skill token overlap
        overlap = len(set(rtoks) & set(ctoks))
        passes = (score >= 0.30) or (overlap >= 2) or (overlap == 1 and (set(rtoks) & _SKILL_TOKENS))
        if passes:
            best.append({
                "source_ref": c.get("source_ref") or {},
                "snippet": text.strip(),
                "score": round(float(score), 3),
            })

    # return top 2 by score (stable)
    return sorted(best, key=lambda x: x["score"], reverse=True)[:2]

# ---------- Route ----------
@router.post("/build")
def build_evidence(payload: EvidenceIn):
    cand = _safe_cand(payload.candidate_id)
    jd = payload.jd_schema or {}
    themes = jd.get("themes") or []
    if not isinstance(themes, list) or not themes:
        raise HTTPException(status_code=400, detail="Schema missing themes.")

    used_fallback = {"claims": False, "transcript": False}

    # --- 1) Resume claims: use payload if present; otherwise load latest *.claims.json
    if payload.claims and len(payload.claims) > 0:
        resume_claims: List[Dict[str, Any]] = list(payload.claims)
    else:
        resume_claims = []
        cjson = _latest_claims_json(cand)
        if cjson:
            resume_claims = _load_claims_from_file(cjson)
            used_fallback["claims"] = True

    # --- 2) Transcript sentences: build from provided web path; if none provided, try latest
    transcript_claims: List[Dict[str, Any]] = _claims_from_transcript_webpath(cand, payload.transcript_path)
    if not transcript_claims and not payload.transcript_path:
        ttxt = _latest_transcript_txt(cand)
        if ttxt:
            txt = _read_transcript_txt(ttxt)
            for idx, s in enumerate(_sentences(txt), start=1):
                transcript_claims.append({
                    "id": f"t{idx:04d}",
                    "text": s,
                    "source_ref": {"kind": "transcript", "line": idx},
                    "tags": [],
                })
            used_fallback["transcript"] = True

    # --- 3) Merge artifacts
    claims: List[Dict[str, Any]] = resume_claims + transcript_claims

    # --- 4) Tokenize once for performance
    claim_pairs: List[Tuple[Dict[str, Any], List[str]]] = [(c, _tok(c.get("text",""))) for c in claims]

    adapter: Dict[str, List[Dict[str, Any]]] = {}
    total_reqs = 0
    hit_reqs = 0
    by_theme: Dict[str, float] = {}

    for t in themes:
        tname = (t.get("name") or "").strip() or "Theme"
        reqs = [r for r in (t.get("requirements") or []) if str(r).strip()]
        total_reqs += len(reqs)
        rows: List[Dict[str, Any]] = []

        theme_hits = 0
        for req in reqs:
            evid = _score_requirement(str(req), claim_pairs)
            if evid:
                theme_hits += 1
            rows.append({
                "requirement": req,
                "evidence": [{"source_ref": e["source_ref"], "snippet": e["snippet"]} for e in evid],
                "open_questions": [],
            })

        hit_reqs += theme_hits
        by_theme[tname] = round((theme_hits / max(1, len(reqs))) * 100.0, 1)
        adapter[tname] = rows

    overall = round((hit_reqs / max(1, total_reqs)) * 100.0, 1)

    resp: Dict[str, Any] = {
        "adapter": adapter,
        "coverage": {"by_theme": by_theme, "overall": overall},
        "counts": {"requirements": total_reqs, "hit": hit_reqs},
    }

    if not claims:
        resp["warning"] = "No artifacts found (resume claims/transcript)."
    if used_fallback["claims"] or used_fallback["transcript"]:
        resp["fallback_used"] = used_fallback

    return resp
