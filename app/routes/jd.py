from fastapi import APIRouter
from pydantic import BaseModel
from ..services.llm import compile_60s_jd

router = APIRouter()


class JDIn(BaseModel):
    text: str


@router.post("/60s")
def sixty_seconds_jd(payload: JDIn):
    """
    Compile a Job Description (JD) into the 60-second schema.
    P1.5: Live-only; no DRY mode flags or branches.
    """
    schema = compile_60s_jd(payload.text)
    return {"schema": schema}
