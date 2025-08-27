from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from .routes.jd import router as jd_router

from .routes.ui import router as ui_router

app = FastAPI(title="Stimulus AI (48h POC)")

# Permissive CORS for quick demo

app.add_middleware(

CORSMiddleware,

allow_origins=["*"], allow_credentials=True,

allow_methods=["*"], allow_headers=["*"]

)

@app.get("/health")

def health():

    return {"status": "ok"}

app.include_router(jd_router, prefix="/jd", tags=["jd"])

app.include_router(ui_router, prefix="/ui", tags=["ui"])