from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .routes.jd import router as jd_router
from .routes.ui import router as ui_router
from fastapi.responses import RedirectResponse


app = FastAPI(title="Stimulus AI (48h POC)")

# Optional - In place in the event I opt to use an external CSS style sheet
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Permissive CORS for quick demo

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
@app.get("/", include_in_schema=False)

def home():
    # 307 preserves method if I ever POST to "/" (nice safety); 302/308 also fine here
    return RedirectResponse(url="/ui/adapter", status_code=307)
    
def health():
    return {"status": "ok"}

app.include_router(jd_router, prefix="/jd", tags=["jd"])
app.include_router(ui_router, prefix="/ui", tags=["ui"])