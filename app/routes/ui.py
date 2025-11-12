from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/adapter", response_class=HTMLResponse)
def adapter_card(request: Request):
    # Render a clean shell; the browser will call /jd/60s (DRY or Live) when you click Compile
    return templates.TemplateResponse("adapter_card.html", {"request": request})
