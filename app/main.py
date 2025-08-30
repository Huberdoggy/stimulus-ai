from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .routes.jd import router as jd_router
from .routes.ui import router as ui_router
from .routes.artifacts import router as artifacts_router
from .routes.evidence import router as evidence_router  # ✅ add

app = FastAPI(title="Stimulus AI (48h POC)")

# Static assets (images, css)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Permissive CORS for quick demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/ui/adapter", status_code=307)

# Routers
app.include_router(jd_router,        prefix="/jd",        tags=["jd"])
app.include_router(ui_router,        prefix="/ui",        tags=["ui"])
app.include_router(artifacts_router, prefix="/artifacts", tags=["artifacts"])
app.include_router(evidence_router, tags=["evidence"])  # ✅ add - no prefix
