import os
import json
from typing import Dict
from openai import OpenAI
from pydantic import BaseModel

# Optional: respect external model override and DRY_MODE

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini") # small + frugal

DRY_MODE = os.getenv("DRY_MODE", "1") == "1"

# Minimal schema typing for sanity (keeps the UI predictable)

class SixtySecondJD(BaseModel):
  role_title: str
  company: str | None = None
  themes: list[Dict] # [{name, requirements:[],success_indicators:[]}]

STUB = SixtySecondJD(
role_title="AI Business Solutions — Solution Engineer",
company="Contoso",
themes=[
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
  },
  
  {
    "name": "Prototyping",
    "requirements": [
      "Rapid build of narrow proof of concept",
      "Measure outcome vs. baseline"
    ],
    "success_indicators": [
      "Working demo that survives a live click-through",
      "Early signal on feasibility/cost"
    ],
  },
],
)

def compile_60s_jd(raw_text: str) -> dict:

  """
  Returns a dict that conforms to SixtySecondJD.
  DRY_MODE returns a deterministic stub to avoid token spend.
  """

  if DRY_MODE:
    return json.loads(STUB.model_dump_json())
    #return json.loads(STUB.json()) # Deprecated

  # Live call path (OpenAI) — keep it frugal and deterministic
  
  client = OpenAI()
  
  system = (
  "Return JSON with keys: role_title, company, "
    "themes:[{name, requirements[], success_indicators[]}] — no prose."
  )
  
  resp = client.chat.completions.create(
    model=LLM_MODEL,
    temperature=0.1,
    messages=[
      {"role": "system", "content": system},
      {"role": "user", "content": f"Job Description:\n{raw_text}"}
    ],
  )
  
  # Defensive parse (model returns a JSON string)
  
  content = resp.choices[0].message.content.strip() # pyright: ignore
  
  return json.loads(content)
