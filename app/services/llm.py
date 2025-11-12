import os
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict
from pydantic import BaseModel

# ======================== P1.5: Determinism + Cache =========================
# - Live-only (DRY removed at call path; we keep the parameter for compatibility)
# - Strict params on compile: temperature=0, top_p=1, optional seed
# - 7-day schema cache with lazy eviction + bounded sweep, atomic writes
# - No prompt edits in evidence.py (this file is safe to change)
# ============================================================================


# ------------------------ env helpers ------------------------
def truthy(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    v = v.strip().strip('"').strip("'").lower()
    return v in {"1", "true", "yes", "on", "y"}


# Single model knob (live only)
MODEL_LIVE = os.getenv("LLM_MODEL_LIVE", "gpt-4o-mini")

# Seed (optional; if unsupported by SDK, remove/ignore at the call site)
LLM_SEED = int(os.getenv("LLM_SEED", "42"))

LLM_DEBUG = truthy("LLM_DEBUG", "0")  # optional fingerprint print

# Cache TTL: 7 days
SCHEMA_TTL_SECONDS = 7 * 24 * 60 * 60

# Prompt version tag (not a prompt change; just a cache discriminator)
COMPILER_PROMPT_V = os.getenv("COMPILER_PROMPT_V", "p1.5-compiler-2025-10-05")

# Cache dirs
_THIS_FILE = Path(__file__).resolve()
_APP_ROOT = _THIS_FILE.parents[1]  # .../app
_CACHE_ROOT = _APP_ROOT / "static" / "cache"
_SCHEMAS_DIR = _CACHE_ROOT / "schemas"

# Opportunistic purge cadence
_PURGE_MIN_INTERVAL_S = 600  # 10 minutes
_last_purge_ts = 0.0


def ensure_cache_dirs() -> None:
    _SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.time()


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # NEW
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _is_expired(path: Path, ttl_seconds: int, now: float | None = None) -> bool:
    if not path.exists():
        return True
    if now is None:
        now = _now()
    try:
        age = now - path.stat().st_mtime
    except FileNotFoundError:
        return True
    return age > ttl_seconds


def _lazy_load_schema(path: Path, ttl_seconds: int) -> dict | None:
    """Return cached schema dict or None. Deletes expired files on sight."""
    if not path.exists():
        return None
    if _is_expired(path, ttl_seconds):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return blob.get("schema")
    except Exception:
        # Corrupt file → remove
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def purge_cache(
    ttl_seconds: int = SCHEMA_TTL_SECONDS,
    scan_limit: int = 50,
    max_runtime_ms: int = 25,
    min_interval_s: int = _PURGE_MIN_INTERVAL_S,
) -> int:
    """Small, bounded sweep of expired schema cache files; returns files removed."""
    global _last_purge_ts
    now = _now()
    if now - _last_purge_ts < min_interval_s:
        return 0
    _last_purge_ts = now

    ensure_cache_dirs()
    removed = 0
    deadline = now + (max_runtime_ms / 1000.0)
    try:
        entries = list(_SCHEMAS_DIR.iterdir())
    except FileNotFoundError:
        return 0
    # Oldest first
    entries.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)

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


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cfg_signature(model: str, temperature: float, top_p: float, seed: int | None) -> str:
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


def _schema_cache_path(jd_hash: str, model: str, cfg_sig: str, prompt_ver: str) -> Path:
    fname = f"jd={jd_hash}__model={model}__cfg={cfg_sig}__pv={prompt_ver}.schema.json"
    return _SCHEMAS_DIR / fname


# ------------------------ schema guard ------------------------
class SixtySecondJD(BaseModel):
    role_title: str
    company: str | None = None
    themes: list[Dict]  # [{name, requirements:[], success_indicators:[]}]


# ------------------------ normalization ------------------------
_PLACEHOLDERS = {"not specified", "unspecified", "n/a", "na", "none", "null", ""}


def _normalize_company(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in _PLACEHOLDERS:
        return None
    return s


# ------------------------ domain-agnostic system prompt ------------------------
# NOTE: This is your existing compiler prompt as-is.
_SYSTEM_PROMPT = """
You are a domain-agnostic, JSON-only compiler. Extract exclusively what appears in the Job Description (JD).
Return ONE JSON object with keys:
- role_title (string)
- company (string OR null)
- themes (array of objects, each with keys: name, requirements (array), success_indicators (array))

Rules:
- Use ONLY facts present in the JD (verbatim or precise paraphrase). Never invent or generalize.
- Derive 3–6 domain-appropriate theme names from the JD itself:
  • Prefer concise noun phrases taken from section headers or clear clusters of responsibilities/skills.
  • Themes must reflect the domain (e.g., “Patient Care & Safety", “Guest Experience & Front Desk”, “Sales Pipeline & Prospecting”,
    “Operations & Inventory”, “Regulatory Compliance”, “Instruction & Classroom Management”, “Creative Development”, etc.).
  • Do NOT force any technology-specific theme unless the JD mentions it.
- requirements: short, testable bullets that actually appear in the JD (≤120 chars each). Include named tools, regulations,
  standards, or techniques ONLY if present (e.g., EMR systems, OMS/POS, CRM, Python, regex, LLMs, HTML/XML/Markdown, JSON/CSV/RTF).
- success_indicators: observable outcomes implied by the JD (quality bars, SLA/throughput, accuracy, compliance, satisfaction/CSAT,
  audit pass, inter-rater consistency, revenue/targets, safety metrics, etc.). Avoid vague platitudes.
- company: if absent in the JD, set to null. Do NOT output placeholders like "Not Specified", "N/A", or "None".
- Output RAW JSON only (no markdown, no code fences, no commentary).
"""


# ------------------------ compiler (Live-only) ------------------------
def compile_60s_jd(raw_text: str, *, dry_mode: bool | None = None) -> dict:
    """
    Compile a JD into the 60-second schema.
    P1.5:
      - Live-only (dry_mode is ignored)
      - Strict params: temperature=0, top_p=1, optional seed
      - 7-day schema cache with lazy eviction + bounded sweep, atomic writes
    """
    # Live-only; ignore dry_mode
    _ = dry_mode

    # Cache preflight (ensure dirs + small sweep)
    ensure_cache_dirs()
    purge_cache()

    # Build cache key
    jd_norm = _normalize_ws(raw_text)
    jd_hash = _sha256_text(jd_norm)
    cfg_sig = _cfg_signature(MODEL_LIVE, temperature=0.0, top_p=1.0, seed=LLM_SEED)
    cache_path = _schema_cache_path(jd_hash, MODEL_LIVE, cfg_sig, COMPILER_PROMPT_V)

    # Case 0: try cache (lazy eviction)
    cached = _lazy_load_schema(cache_path, SCHEMA_TTL_SECONDS)
    if cached is not None:
        return cached

    # ---- live call (strict) ----
    from openai import OpenAI

    client = OpenAI()

    system = _SYSTEM_PROMPT
    user = f"JD:\n{raw_text}"

    resp = client.chat.completions.create(
        model=MODEL_LIVE,
        temperature=0.0,
        top_p=1.0,
        seed=LLM_SEED,  # if unsupported by your SDK version, remove this line
        n=1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    if LLM_DEBUG:
        print(
            "[LLM COMPILE] system_fingerprint:", getattr(resp, "system_fingerprint", None)
        )

    # coerce to string
    content = resp.choices[0].message.content or ""

    # Lenient JSON extraction
    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))

    # Normalize + validate
    data["company"] = _normalize_company(data.get("company"))
    if "role_title" not in data or "themes" not in data:
        raise ValueError("Model returned incomplete schema.")

    cleaned_themes = []
    for t in data.get("themes", []):
        name = (t.get("name") or "").strip()
        reqs = [r for r in (t.get("requirements") or []) if str(r).strip()]
        succ = [s for s in (t.get("success_indicators") or []) if str(s).strip()]
        if name and (reqs or succ):
            cleaned_themes.append(
                {"name": name, "requirements": reqs, "success_indicators": succ}
            )
    data["themes"] = cleaned_themes

    # Final shape guard
    SixtySecondJD(
        **{
            "role_title": data["role_title"],
            "company": data.get("company"),
            "themes": data["themes"],
        }
    )

    # Persist to cache (atomic)
    payload = {
        "created_at": int(_now()),
        "jd_hash": jd_hash,
        "model": MODEL_LIVE,
        "cfg_sig": cfg_sig,
        "prompt_ver": COMPILER_PROMPT_V,
        "schema": data,
    }
    _atomic_write_json(cache_path, payload)
    return data
