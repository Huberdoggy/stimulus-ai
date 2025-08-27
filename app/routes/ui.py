from fastapi import APIRouter, Request

from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")

@router.get("/adapter", response_class=HTMLResponse)

def adapter_card(request: Request):

# Simple starter: sample payload (DRY demo)

    sample = {
        "role_title": "AI Business Solutions — Solution Engineer",
        "company": "Contoso",
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

    return templates.TemplateResponse("adapter_card.html",
                                      {"request": request, "schema":sample}
                                     )