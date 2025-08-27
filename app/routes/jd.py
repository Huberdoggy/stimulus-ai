from fastapi import APIRouter

from pydantic import BaseModel

from ..services.llm import compile_60s_jd

router = APIRouter()

class JDIn(BaseModel):
    text: str

@router.post("/60s")

def sixty_seconds_jd(payload: JDIn):
    schema = compile_60s_jd(payload.text)
    return {"schema": schema}