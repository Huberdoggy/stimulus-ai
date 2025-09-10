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

# Router: no prefix here; main.py applies /evidence
router = APIRouter(tags=["evidence"])

# ---------- Project paths ----------
def _find_app_dir(here: Path) -> Path:
    for p in [here] + list(here.parents):
        if p.name == "app":
            return p
    return here.parents[1] / "app"

HERE = Path(__file__).resolve()
APP_DIR = _find_app_dir(HERE)
ROOT_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESUME_DIR = UPLOADS_DIR / "resumes"
AUDIO_DIR = UPLOADS_DIR / "audio"
VIDEO_DIR = UPLOADS_DIR / "video"  # NEW: allow transcripts saved next to video artifacts
LEXICON_PATH = STATIC_DIR / "lexicon.json"

# ---------- Env / model selection ----------
def truthy(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    v = v.strip().strip('"').strip("'").lower()
    return v in {"1", "true", "yes", "on", "y"}

MODEL_LIVE = os.getenv("LLM_MODEL_LIVE", "gpt-4o-mini")
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
    """
    Find the newest transcript .txt under either audio/<cand> or video/<cand>.
    """
    candidates: List[Path] = []
    for base in (AUDIO_DIR / cand, VIDEO_DIR / cand):
        if base.exists():
            for p in base.iterdir():
                if p.is_file() and p.suffix == ".txt":
                    candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

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
    words = re.findall(r"\S+\s*", s or "")
    out: List[str] = []
    buf = ""
    for w in words:
        if len(buf) + len(w) > max_len and len(buf) >= min_len:
            out.append(buf.strip())
            buf = w
        else:
            buf += w
    if buf.strip():
        out.append(buf.strip())
    return out

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
    Accepts /static/uploads/(audio|video)/... .txt and returns a filesystem Path,
    restricted to the corresponding directory under app/static/uploads.
    """
    if not web_path:
        return None
    rel = unquote(web_path).lstrip("/")
    if rel.startswith("app/"):
        rel = rel[4:]
    path = (APP_DIR / rel).resolve()

    audio_root = AUDIO_DIR.resolve()
    video_root = VIDEO_DIR.resolve()
    # Allow either audio or video transcript locations
    allowed = (str(path).startswith(str(audio_root)) or str(path).startswith(str(video_root)))
    if not allowed or not path.exists():
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
    "• open_questions: array of up to 2 concise, ACTIONABLE questions (no vague phrases like 'tell me more') to clarify gaps ONLY if evidence is weak or missing.\n\n"
    "Strict rules:\n"
    "• Do not invent text. Only use claims provided.\n"
    "• Prefer source diversity when two candidates are comparable.\n"
    "• If a requirement has TWO evidence items AND both resume and transcript claims exist, choose ONE from each source (resume+transcript) unless no transcript claim is reasonably relevant—in that case include 1 transcript-based open question.\n"
    "• Keep hit_terms minimal and meaningful: 1–8 items, each 1–40 chars, must appear verbatim in the claim text (case-insensitive allowed) and relate directly to the requirement.\n"
    "• Limit evidence to 0–2 items per requirement. If all transcript claims are weak, select the single most relevant transcript claim and include an open question explaining the gap.\n"
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

# ---------- DRY (lexicon) helpers ----------
def _load_lexicon() -> Dict[str, Any]:
    if LEXICON_PATH.exists():
        try:
            return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Sensible defaults if missing
    return {
        "stop": ["the","and","of","to","a","in","for","on","as","with","by","or","an","be","at","from","is","are","was","were","it","that","this","these","those"],
        "aliases": {
            "leadership": ["led","managed","oversaw","mentored","coached","owner","owned"],
            "delivery": ["delivered","shipped","launched","deployed","released","rolled out"],
            "metrics": ["kpi","sla","target","throughput","csat","nps","uptime","conversion","revenue","churn","accuracy","latency"],
            "ai": ["llm","prompt","chatgpt","openai","anthropic","gemini","whisper","embedding","vector","rag","transformer"],
            "data": ["sql","python","pandas","etl","pipeline","analytics","insights","dashboard","bi"],
            "product": ["roadmap","backlog","user stories","ux","ui","prototype","mvp","discovery","requirements","stakeholder"],
            "ops": ["process","sop","workflow","incident","on-call","runbook","rca","sre","monitoring","alerting"],
            "compliance": ["hipaa","gdpr","soc 2","pci","osha","ferpa","sox"],
            "education": ["curriculum","instruction","classroom","students","lesson","assessment"],
            "healthcare": ["patient","clinical","emr","ehr","medication","safety","care"],
            "sales": ["pipeline","prospecting","quota","crm","deal","renewal","upsell","cold call"]
        }
    }

def _tokens(s: str) -> List[str]:
    return [w.lower() for w in re.compile(r"[A-Za-z0-9\+\#\.\-']+").findall(s or "")]

def _expand_aliases(tokens: List[str], lex: Dict[str, Any]) -> List[str]:
    aliases = lex.get("aliases", {})
    out = set(tokens)
    for canon, alts in aliases.items():
        alts_set = {a.lower() for a in (alts or [])}
        if any(t in alts_set or t == canon for t in tokens):
            out.add(canon)
    return list(out)

def _score(req_tokens: set[str], claim_tokens: set[str], stop: set[str]) -> tuple[int, List[str]]:
    inter = [t for t in claim_tokens.intersection(req_tokens) if t not in stop]
    return len(inter), inter[:8]

def _dry_lexicon_adapter(jd: Dict[str, Any], claims: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    lex = _load_lexicon()
    stop = set(lex.get("stop", []))
    # Pre-tokenize claims w/ alias expansion
    c_vecs = []
    for c in claims:
        toks = _expand_aliases(_tokens(c.get("text","")), lex)
        c_vecs.append((c, set(toks)))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for t in (jd.get("themes") or []):
        tname = (t.get("name") or "").strip() or "Theme"
        rows: List[Dict[str, Any]] = []
        for req in (t.get("requirements") or []):
            rtoks = _expand_aliases(_tokens(req), lex)
            rt = set(rtoks)
            # Score all claims
            scored = []
            for c, toks in c_vecs:
                s, hits = _score(rt, toks, stop)
                if s > 0:
                    scored.append((s, c, hits))
            scored.sort(key=lambda x: (-x[0], (x[1].get("source") != "transcript"), x[1].get("id","")))

            # Choose up to 2, enforce diversity when possible
            chosen: List[Dict[str, Any]] = []
            have_resume = any(c.get("source") == "resume" for _, c, _ in scored)
            have_trans  = any(c.get("source") == "transcript" for _, c, _ in scored)

            for s, c, hits in scored:
                if len(chosen) >= 2:
                    break
                # If both sources exist overall and we already picked resume, favor transcript next (and vice versa)
                if len(chosen) == 1 and have_resume and have_trans:
                    if chosen[0]["source_ref"]["kind"] == c.get("source"):
                        continue
                chosen.append({
                    "source_ref": {"kind": c.get("source"), "line": c.get("line")},
                    "snippet": c.get("text",""),
                    "hit_terms": hits
                })

            # Actionable OQs if nothing chosen
            oq = []
            if not chosen:
                oq = [f"Share a concrete example for “{req}” (scope, tools, and metric/outcome)."]

            rows.append({"requirement": req, "evidence": chosen, "open_questions": oq})
        out[tname] = rows
    return out

# ---------- Core route ----------
@router.post("/build")
def build_evidence(
    payload: EvidenceIn,
    dry: int | None = Query(None, description="1=stub (lexicon), 0=live"),
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

    # Transcript claims
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

    use_dry = DRY_MODE_DEFAULT if dry is None else bool(dry)
    if use_dry:
        adapter = _dry_lexicon_adapter(jd, claims_llm)
        # Coverage
        cov_by = {}
        total = 0
        hit = 0
        for t in themes:
            tname = (t.get("name") or "").strip() or "Theme"
            rows = adapter.get(tname, [])
            total += len(rows)
            th = sum(1 for r in rows if (r.get("evidence") or []))
            cov_by[tname] = round((th / max(1, len(rows))) * 100.0, 1)
            hit += th
        coverage = {"by_theme": cov_by, "overall": round((hit / max(1, total)) * 100.0, 1)}
        counts = {"requirements": total, "hit": hit}
        return {"adapter": adapter, "coverage": coverage, "counts": counts}

    # Live LLM call (JSON-only) with diversity & actionable OQ rules baked in
    client = OpenAI()

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
            model=MODEL_LIVE,
            temperature=0,
            top_p=1,
            messages=[
                {"role": "system", "content": system_hint},
                {"role": "user", "content":
                    'Below is the payload. Use it as the only source of truth. '
                    'Return only the JSON specified in the System Prompt’s "Output JSON schema".\n\n'
                    + user_payload
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

    # Diversity nudge: if transcript exists and not used anywhere, or a requirement shows two same-source hits where both sources exist
    needs_diversity_pass = transcript_exists and not used_any_transcript
    if not needs_diversity_pass:
        for tname, rows in adapter.items():
            for r in rows:
                evs = r.get("evidence") or []
                kinds = {e["source_ref"]["kind"] for e in evs}
                if len(evs) >= 2 and transcript_exists and kinds == {"resume"}:
                    needs_diversity_pass = True
                    break
            if needs_diversity_pass:
                break

    if needs_diversity_pass:
        data2 = call_llm(
            extra_hint="Ensure requirements with two evidence items use resume+transcript when both sources exist. "
                       "If no reasonable transcript exists, include one transcript-based open question instead."
        )
        adapter_in2 = data2.get("adapter") or {}
        if expected_keys.intersection(set(adapter_in2.keys())):
            adapter2, coverage2, counts2, used_any_transcript2 = _resolve_and_score(adapter_in2, jd, id2claim)
            if used_any_transcript2:
                adapter, coverage, counts = adapter2, coverage2, counts2

    return {"adapter": adapter, "coverage": coverage, "counts": counts}
