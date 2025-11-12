from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI

# Router: main.py applies /evidence prefix
router = APIRouter(tags=["evidence"])

# ---------------- Path constants ----------------
APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESUME_DIR = UPLOADS_DIR / "resumes"
AUDIO_DIR = UPLOADS_DIR / "audio"
VIDEO_DIR = UPLOADS_DIR / "video"

# Cache
CACHE_ROOT = STATIC_DIR / "cache"
EVIDENCE_DIR = CACHE_ROOT / "evidence"


# ---------------- Env / model selection ----------------
def truthy(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    v = v.strip().strip('"').strip("'").lower()
    return v in {"1", "true", "yes", "on", "y"}


MODEL_LIVE = os.getenv("LLM_MODEL_LIVE", "gpt-4o-mini")
LLM_SEED = int(os.getenv("LLM_SEED", "42"))
LLM_DEBUG = truthy("LLM_DEBUG", "0")  # optional fingerprint print

# 7 days TTL
EVID_TTL_SECONDS = 7 * 24 * 60 * 60
EVIDENCE_PROMPT_V = os.getenv("EVIDENCE_PROMPT_V", "p1.5-evidence-2025-10-05")

# opportunistic purge cadence
_PURGE_MIN_INTERVAL_S = 600
_last_purge_ts = 0.0


def ensure_cache_dirs() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.time()


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # NEW: ensure nested dirs exist
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _is_expired(path: Path, ttl_seconds: int, now: Optional[float] = None) -> bool:
    if not path.exists():
        return True
    if now is None:
        now = _now()
    try:
        age = now - path.stat().st_mtime
    except FileNotFoundError:
        return True
    return age > ttl_seconds


def purge_cache(
    ttl_seconds: int = EVID_TTL_SECONDS,
    scan_limit: int = 50,
    max_runtime_ms: int = 25,
    min_interval_s: int = _PURGE_MIN_INTERVAL_S,
) -> int:
    global _last_purge_ts
    now = _now()
    if now - _last_purge_ts < min_interval_s:
        return 0
    _last_purge_ts = now
    ensure_cache_dirs()
    removed = 0
    deadline = now + (max_runtime_ms / 1000.0)
    try:
        # Recurse into nested layout and pick only evidence files
        entries = list(EVIDENCE_DIR.rglob("*.evidence.json"))
    except FileNotFoundError:
        return 0

    # oldest first (by mtime)
    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    entries.sort(key=mtime)

    for p in entries:
        if removed >= scan_limit or _now() > deadline:
            break
        if _is_expired(p, ttl_seconds, now):
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed


# --------------- Helpers: hashing / signatures ---------------
import hashlib  # pyright: ignore


def _safe_component(s: str, maxlen: int = 60) -> str:
    # Keep only filename-safe chars; trim length
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s or "")
    return s[:maxlen]


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cfg_signature(
    model: str, temperature: float, top_p: float, seed: Optional[int]
) -> str:
    payload = {
        "model": model,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": int(seed) if seed is not None else None,
        "penalties": {"presence": 0.0, "frequency": 0.0},
        "n": 1,
        "stream": False,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _schema_signature_from_schema(schema: Dict[str, Any]) -> str:
    # Only the deterministic parts: theme names and requirement lists
    themes = schema.get("themes") or []
    norm = []
    for t in themes:
        name = (t.get("name") or "").strip()
        reqs = list(t.get("requirements") or [])
        norm.append({"name": name, "requirements": reqs})
    return _sha256_text(
        json.dumps(norm, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _artifact_manifest(
    candidate_id: str,
    resume_claims_path: Optional[Path],
    transcript_path: Optional[Path],
    claims_inline: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    man: Dict[str, Any] = {"candidate_id": candidate_id}

    def file_entry(p: Optional[Path]) -> Optional[Dict[str, Any]]:
        if not p or not p.exists():
            return None
        try:
            data = p.read_bytes()
        except Exception:
            return {"path": str(p), "exists": p.exists()}
        h = hashlib.sha1(data).hexdigest()
        st = p.stat()
        return {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime), "sha1": h}

    man["resume_claims"] = file_entry(resume_claims_path)
    man["transcript_txt"] = file_entry(transcript_path)
    if claims_inline is not None:
        try:
            blob = json.dumps(claims_inline, sort_keys=True, ensure_ascii=False)
        except Exception:
            blob = str(claims_inline)
        man["claims_inline_sha1"] = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    return man


def _artifact_signature(manifest: Dict[str, Any]) -> str:
    return _sha256_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def _evidence_cache_path(
    jd_sig: str, art_sig: str, model: str, cfg_sig: str, prompt_ver: str
) -> Path:
    # Shard into subdirs to avoid single-filename >255 bytes
    jd_dir = EVIDENCE_DIR / f"jd={jd_sig}"
    cfg_dir = jd_dir / f"cfg={cfg_sig}"
    mod_dir = cfg_dir / f"model={_safe_component(model)}"
    pv_dir = mod_dir / f"pv={_safe_component(prompt_ver)}"
    return pv_dir / f"art={art_sig}.evidence.json"


def _lazy_load_json(path: Path, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    if _is_expired(path, ttl_seconds):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


# ---------------- Artifact discovery (mirrors your existing) ----------------
ALLOWED_CAND = re.compile(r"^[a-z0-9_\-]+$", re.IGNORECASE)


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
        raise HTTPException(500, f"Invalid claims JSON: {e}")
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        text = (it.get("text") or "").strip()
        if not text:
            continue
        source_ref = (
            it.get("source_ref") if isinstance(it.get("source_ref"), dict) else {}
        )
        kind = (source_ref.get("kind") or "resume").strip() or "resume"
        line = it.get("line")
        if line is None:
            line = source_ref.get("line")
        sr_payload: Dict[str, Any] = {"kind": kind}
        if line is not None:
            sr_payload["line"] = line
        out.append(
            {
                "id": f"r{i:04d}",
                "text": text,
                "source": kind,
                "line": line,
                "source_ref": sr_payload,
            }
        )
    return out


def _chunk_text(text: str, max_len: int = 280, min_len: int = 60) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # try to cut at sentence boundary near max_len
        cut = text.rfind(". ", 0, max_len)
        if cut == -1:
            cut = text.rfind(", ", 0, max_len)
        if cut == -1:
            cut = max_len
        frag, text = text[: cut + 1].strip(), text[cut + 1 :].strip()
        if len(frag) >= min_len:
            chunks.append(frag)
    # dedupe consecutive
    dedup: List[str] = []
    for s in chunks:
        if not dedup or s != dedup[-1]:
            dedup.append(s)
    return dedup[:200]


def _claims_from_transcript_file(fp: Path) -> List[Dict[str, Any]]:
    txt = _read_text(fp)
    fragments = _chunk_text(txt, max_len=240, min_len=60)
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(fragments, 1):
        sr_payload = {"kind": "transcript", "line": i}
        out.append(
            {
                "id": f"t{i:04d}",
                "text": s,
                "source": "transcript",
                "line": i,
                "source_ref": sr_payload,
            }
        )
    return out


def _resolve_static_web_path(web_path: str) -> Optional[Path]:
    """
    Accepts a page-relative /static/... URL and returns an absolute Path if it exists.
    """
    try:
        wp = web_path.split("?")[0]
    except Exception:
        wp = web_path
    if not wp.startswith("/static/"):
        return None
    p = (APP_DIR / wp.lstrip("/")).resolve()
    try:
        p.relative_to(STATIC_DIR)
    except Exception:
        return None
    return p if p.exists() else None


# ---------------- LLM prompt (unchanged) ----------------
SYSTEM_PROMPT = (
    "You are a JSON-only evidence matcher. You receive:\n"
    "(1) a JD schema with themes and requirements, and\n"
    "(2) a list of candidate claims (id, text, source: \"resume\" or \"transcript\", and optional line).\n"
    "Your job: For EACH requirement in EACH theme, select up to TWO supporting claims from the list and produce:\n"
    "• evidence: array of {claim_id, hit_terms[]} where hit_terms are the EXACT tokens or short substrings copied verbatim from that claim’s text that justify the match.\n"
    "• open_questions: array of up to 2 concise, ACTIONABLE questions (no fluff like 'tell me more') to clarify gaps ONLY if evidence is weak or missing.\n\n"
    "Strict rules:\n"
    "• Do not invent text. Only use claims provided.\n"
    "• Prefer source diversity when two candidates are comparable.\n"
    "• If a requirement has TWO evidence items AND both resume and transcript are present, include one resume claim and one transcript claim when relevant—in that case include 1 transcript-based open question.\n"
    "• Keep hit_terms minimal and meaningful: 1–8 items, each 1–4 words (case-insensitive allowed) and relate directly to the requirement.\n"
    "• Limit evidence to 0–2 items per requirement. If all transcript claims are weak, prefer a resume claim and include an open question explaining the gap.\n"
    "• Output JSON ONLY. No extra keys, no markdown.\n\n"
    "Output JSON schema (theme keys must be actual names):\n"
    "{\n"
    "  \"adapter\": {\n"
    "    \"<Theme Name>\": [\n"
    "      {\n"
    "        \"requirement\": \"\",\n"
    "        \"evidence\": [{\"claim_id\": \"r0001\", \"hit_terms\": [\"term\"]}],\n"
    "        \"open_questions\": [\"\", \"\"]\n"
    "      }\n"
    "    ]\n"
    "  }\n"
    "}"
)


def _build_llm_payload(
    jd_schema: Dict[str, Any], llm_claims: List[Dict[str, Any]], theme_names: List[str]
) -> str:
    payload = {
        "schema": {
            "themes": [
                {
                    "name": (t.get("name") or "").strip(),
                    "requirements": list(t.get("requirements") or []),
                }
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


def _deterministic_select(
    adapter_in: Dict[str, Any],
    jd: Dict[str, Any],
    id2claim: Dict[str, Dict[str, Any]],
    k: int = 2,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float], Dict[str, int]]:
    adapter: Dict[str, List[Dict[str, Any]]] = {}
    total_reqs = 0
    hit_reqs = 0
    by_theme: Dict[str, float] = {}

    themes = jd.get("themes") or []
    for t in themes:
        tname = (t.get("name") or "").strip() or "Theme"
        rows_llm = adapter_in.get(tname) or []
        if not isinstance(rows_llm, list):
            rows_llm = []
        out_rows: List[Dict[str, Any]] = []

        for row in rows_llm:
            requirement = row.get("requirement")
            # build candidate evidence with validation against claim text
            scored: List[Tuple[int, int, str, Dict[str, Any]]] = []
            used_ids = set()
            for e in row.get("evidence", []):
                raw_cid = e.get("claim_id")
                cid = _normalize_cid(raw_cid)
                c = id2claim.get(cid or "") or id2claim.get(
                    str(raw_cid) if raw_cid is not None else ""
                )
                if not c:
                    continue
                text = c.get("text") or ""
                hits = [
                    h
                    for h in (e.get("hit_terms") or [])
                    if isinstance(h, str) and h.strip()
                ]
                # accept only if all hit_terms appear in claim text (case-insensitive)
                ok = True
                uniq = set()
                low = text.lower()
                for h in hits:
                    hs = h.strip()
                    if not hs:
                        continue
                    if hs.lower() not in low:
                        ok = False
                        break
                    uniq.add(hs.lower())
                if not ok:
                    continue
                if cid in used_ids:
                    continue
                used_ids.add(cid)

                # force a string id for sorting
                cid_str = str(c.get("id") or "")
                raw_sr = (
                    c.get("source_ref") if isinstance(c.get("source_ref"), dict) else {}
                )
                source_ref: Dict[str, Any] = {}
                if isinstance(raw_sr, dict):
                    kind_val = raw_sr.get("kind")
                    if kind_val:
                        source_ref["kind"] = kind_val
                    if raw_sr.get("line") is not None:
                        source_ref["line"] = raw_sr.get("line")
                kind = source_ref.get("kind") or (c.get("source") or "artifact")
                source_ref["kind"] = kind
                line_val = c.get("line")
                if line_val is not None and "line" not in source_ref:
                    source_ref["line"] = line_val

                scored.append(
                    (
                        -len(uniq),
                        len(text),
                        cid_str,
                        {
                            "claim_id": c.get("id"),
                            "hit_terms": hits,
                            "snippet": text,
                            "source": kind,
                            "source_ref": source_ref,
                        },
                    )
                )

            # stable deterministic selection
            scored.sort()
            selected = [item[3] for item in scored[:k]]

            oq = [
                q
                for q in (row.get("open_questions") or [])
                if isinstance(q, str) and q.strip()
            ][:2]

            out_rows.append(
                {"requirement": requirement, "evidence": selected, "open_questions": oq}
            )

        total_reqs += len(out_rows)
        theme_hits = sum(1 for r in out_rows if (r.get("evidence") or []))
        by_theme[tname] = round((theme_hits / max(1, len(out_rows))) * 100.0, 1)
        hit_reqs += theme_hits
        adapter[tname] = out_rows

    overall = round((hit_reqs / max(1, total_reqs)) * 100.0, 1)
    counts = {"requirements": total_reqs, "hit": hit_reqs}
    return adapter, by_theme, counts


# ---------------- Request model ----------------
class EvidenceIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    candidate_id: str
    jd_schema: Dict[str, Any] = Field(default_factory=dict, alias="schema")
    claims: Optional[List[Dict[str, Any]]] = None
    transcript_path: Optional[str] = None


# ---------------- Core route (Live-only; DRY removed) ----------------
@router.post("/build")
def build_evidence(payload: EvidenceIn):
    cand = _safe_cand(payload.candidate_id)
    jd = payload.jd_schema or {}
    themes = jd.get("themes") or []
    if not isinstance(themes, list) or not themes:
        raise HTTPException(status_code=400, detail="Schema missing themes.")

    # Build claims from resume and transcript if not supplied
    resume_claims: List[Dict[str, Any]] = []
    resume_claims_fp: Optional[Path] = None
    p = _latest_claims_json(cand)
    if p:
        resume_claims_fp = p
        resume_claims = _load_resume_claims(p)

    transcript_claims: List[Dict[str, Any]] = []
    transcript_fp: Optional[Path] = None
    if payload.transcript_path:
        tp = _resolve_static_web_path(payload.transcript_path)
        if tp:
            transcript_fp = tp
            transcript_claims = _claims_from_transcript_file(tp)
    if not transcript_claims:
        ttxt = _latest_transcript_txt(cand)
        if ttxt:
            transcript_fp = ttxt
            transcript_claims = _claims_from_transcript_file(ttxt)

    # Inline claims (if provided AND non-empty) take precedence for evidence computation.
    # The UI currently POSTs an empty list by default, so treat [] as "no override" (fall back to server artifacts).
    if isinstance(payload.claims, list):
        has_real_claim = any(isinstance(c, dict) and c.get("id") for c in payload.claims)
        inline_claims = payload.claims if has_real_claim else None
    else:
        inline_claims = None

    # Merge claims (transcript first to help diversity tie-break)
    claims_llm = (
        inline_claims
        if inline_claims is not None
        else (transcript_claims + resume_claims)
    )
    id2claim: Dict[str, Dict[str, Any]] = {c["id"]: c for c in claims_llm}
    theme_names = [(t.get("name") or "").strip() or "Theme" for t in themes]

    # ---------------- Cache key build ----------------
    ensure_cache_dirs()
    purge_cache()

    jd_sig = _schema_signature_from_schema(jd)
    man = _artifact_manifest(cand, resume_claims_fp, transcript_fp, inline_claims)
    art_sig = _artifact_signature(man)
    cfg_sig = _cfg_signature(MODEL_LIVE, temperature=0.0, top_p=1.0, seed=LLM_SEED)
    cache_path = _evidence_cache_path(
        jd_sig, art_sig, MODEL_LIVE, cfg_sig, EVIDENCE_PROMPT_V
    )

    # Case 0: try cache
    cached = _lazy_load_json(cache_path, EVID_TTL_SECONDS)
    if cached is not None:
        return cached

    # ---------------- Live LLM call (strict) ----------------
    client = OpenAI()
    user_payload = _build_llm_payload(jd, claims_llm, theme_names)
    system_hint = (
        SYSTEM_PROMPT
        + "\n\nUse these exact theme keys: "
        + json.dumps(theme_names, ensure_ascii=False)
        + "\nSources present summary: "
        + json.dumps(
            {
                "resume": sum(1 for c in claims_llm if c.get("source") == "resume"),
                "transcript": sum(
                    1 for c in claims_llm if c.get("source") == "transcript"
                ),
            },
            ensure_ascii=False,
        )
    )

    resp = client.chat.completions.create(
        model=MODEL_LIVE,
        temperature=0.0,
        top_p=1.0,
        seed=LLM_SEED,
        n=1,
        messages=[
            {"role": "system", "content": system_hint},
            {
                "role": "user",
                "content": 'Below is the payload. Use it as the only source of truth. '
                'Return only the JSON specified in the System Prompt’s "Output JSON schema".\n\n'
                + user_payload,
            },
        ],
    )
    if LLM_DEBUG:
        print(
            "[LLM EVIDENCE] system_fingerprint:",
            getattr(resp, "system_fingerprint", None),
        )

    content = (resp.choices[0].message.content or "").strip()
    try:
        adapter_in = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise HTTPException(500, "LLM returned non-JSON content.")
        adapter_in = json.loads(m.group(0))

    # ---------------- Deterministic post-processing ----------------
    adapter, by_theme, counts = _deterministic_select(
        adapter_in.get("adapter") or {}, jd, id2claim
    )
    overall = round((counts["hit"] / max(1, counts["requirements"])) * 100.0, 1)
    coverage = {"by_theme": by_theme, "overall": overall}

    out = {"adapter": adapter, "coverage": coverage, "counts": counts}

    # Persist to cache (atomic)
    payload_obj = {
        "created_at": int(_now()),
        "jd_sig": jd_sig,
        "artifacts_sig": art_sig,
        "model": MODEL_LIVE,
        "cfg_sig": cfg_sig,
        "prompt_ver": EVIDENCE_PROMPT_V,
        "adapter": adapter,
        "coverage": coverage,
        "counts": counts,
    }
    _atomic_write_json(cache_path, payload_obj)
    return out
