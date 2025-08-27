import os
import json
import re
from typing import Dict
from pydantic import BaseModel

# ------------------------ env helpers ------------------------

def truthy(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    v = v.strip().strip('"').strip("'").lower()
    return v in {"1", "true", "yes", "on", "y"}

# Frugal default; allow stronger model for Live
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MODEL_LIVE = os.getenv("LLM_MODEL_LIVE", "") or LLM_MODEL

DRY_MODE_DEFAULT = truthy("DRY_MODE", "1")

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
  • Themes must reflect the domain (e.g., “Patient Care & Safety”, “Guest Experience & Front Desk”, “Sales Pipeline & Prospecting”,
    “Operations & Inventory”, “Regulatory Compliance”, “Instruction & Classroom Management”, “Creative Development”, etc.).
  • Do NOT force any technology-specific theme unless the JD mentions it.
- requirements: short, testable bullets that actually appear in the JD (≤120 chars each). Include named tools, regulations,
  standards, or techniques ONLY if present (e.g., EMR systems, OSHA, POS, CRM, Python, regex, LLMs, HTML/XML/Markdown, JSON/CSV/RTF).
- success_indicators: observable outcomes implied by the JD (quality bars, SLA/throughput, accuracy, compliance, satisfaction/CSAT,
  audit pass, inter-rater consistency, revenue/targets, safety metrics, etc.). Avoid vague platitudes.
- company: if absent in the JD, set to null. Do NOT output placeholders like "Not Specified", "N/A", or "None".
- Output RAW JSON only (no markdown, no code fences, no commentary).
""".strip()

# ------------------------ compiler ------------------------

def compile_60s_jd(raw_text: str, *, dry_mode: bool | None = None) -> dict:
    """
    Compile a JD into the 60-second schema. DRY mode returns an echo-stub.
    Live mode uses a strict, domain-agnostic prompt for high fidelity across roles.
    """
    use_dry = DRY_MODE_DEFAULT if dry_mode is None else dry_mode

    if use_dry:
        # DRY: echo first non-empty line for visibility; use a generic theme label.
        first_line = next((ln.strip() for ln in raw_text.splitlines() if ln.strip()), "")
        stub = {
            "role_title": first_line or "60-Second JD (Stub)",
            "company": None,
            "themes": [{
                "name": "Core Responsibilities",
                "requirements": [
                    "Summarize key responsibilities from the JD text",
                    "Follow provided guidelines and process steps"
                ],
                "success_indicators": [
                    "Consistent output meeting stated quality bars",
                    "On-time completion within expectations"
                ],
            }],
        }
        return stub

    # ---- live call (high fidelity, cross-domain) ----
    from openai import OpenAI
    client = OpenAI()
    model = LLM_MODEL_LIVE or LLM_MODEL

    system = _SYSTEM_PROMPT
    user = f"JD:\n{raw_text}"

    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()

    # Parse JSON defensively
    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))

    # Normalize company and clean empty items
    data["company"] = _normalize_company(data.get("company"))

    if "role_title" not in data or "themes" not in data:
        raise ValueError("Model returned incomplete schema.")

    cleaned_themes = []
    for t in data.get("themes", []):
        name = (t.get("name") or "").strip()
        reqs = [r for r in (t.get("requirements") or []) if str(r).strip()]
        succ = [s for s in (t.get("success_indicators") or []) if str(s).strip()]
        if name and (reqs or succ):
            cleaned_themes.append({"name": name, "requirements": reqs, "success_indicators": succ})
    data["themes"] = cleaned_themes

    # Validate shape
    SixtySecondJD(**{
        "role_title": data["role_title"],
        "company": data.get("company"),
        "themes": data["themes"],
    })

    return data
