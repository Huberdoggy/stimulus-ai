import os
import json
from typing import Dict
from pydantic import BaseModel


def truthy(name: str, default: str = "1") -> bool:
  v = os.getenv(name, default)
  if v is None:
      return False
  v = v.strip().strip('"').strip("'").lower()
  return v in {"1", "true", "yes", "on", "y"}


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini") # small + frugal
DRY_MODE_DEFAULT = truthy("DRY_MODE", "1")  # default ON

# Minimal schema typing for sanity (keeps the UI predictable)

class SixtySecondJD(BaseModel):
  role_title: str
  company: str | None = None
  themes: list[Dict] # [{name, requirements:[],success_indicators:[]}]
  

def compile_60s_jd(raw_text: str, *, dry_mode: bool | None = None) -> dict:
  """Return schema. If dry_mode is None, use DRY_MODE_DEFAULT."""
  use_dry = DRY_MODE_DEFAULT if dry_mode is None else dry_mode

  # Helpful DRY behavior: echo the first line into role_title so the UI reflects input
  if use_dry:
      first_line = next((ln.strip() for ln in raw_text.splitlines() if ln.strip()), "")
      stub = {
          "role_title": first_line or "60-Second JD (Stub)",
          "company": None,
          "themes": [
              {
                  "name": "Discovery & Scoping",
                  "requirements": [
                      "Translate business goals into AI use cases",
                      "Map data accessibility and privacy constraints"
                  ],
                  "success_indicators": [
                      "Clarity on problem framing and constraints",
                      "Stakeholder sign-off on scope"
                  ],
              }
          ],
      }
      return stub

  # -- Live call path (frugal) --
  from openai import OpenAI
  client = OpenAI()
  system = (
      "Return JSON with keys: role_title, company, "
      "themes:[{name, requirements[], success_indicators[]}] — no prose."
  )
  resp = client.chat.completions.create(
      model=LLM_MODEL, temperature=0.1,
      messages=[{"role":"system","content":system},
                {"role":"user","content":f"Job Description:\n{raw_text}"}]
  )
  content = resp.choices[0].message.content.strip() # pyright: ignore
  return json.loads(content)

