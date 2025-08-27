from fastapi import APIRouter
from fastapi import Query
from pydantic import BaseModel
from ..services.llm import compile_60s_jd

router = APIRouter()

class JDIn(BaseModel):
    text: str

@router.post("/60s")
def sixty_seconds_jd(payload: JDIn, dry: int | None = Query(None, description="1=stub, 0=live")):
    # dry=None → use global default; 1 → force stub; 0 → force live
    force = None if dry is None else bool(dry)
    schema = compile_60s_jd(payload.text, dry_mode=force)
    return {"schema": schema}
