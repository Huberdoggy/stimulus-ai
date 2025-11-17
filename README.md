# 🔌⚡ Stimulus.AI 🔋

Stimulus AI turns résumés, transcripts, and job descriptions into a shared “evidence map” so recruiters can see exactly where each requirement is satisfied. The FastAPI app you see here bundles the full workflow: compile a JD schema, upload artifacts, transcribe spoken samples, and drive the animated adapter UI. Highlights:

- **Evidence-first UX:** `/ui/adapter` renders a deterministic coverage map with token-level highlights, badges for codecs/durations, and live health checks.
- **Media ingest pipeline:** ffmpeg handles video/audio probing, WAV extraction, and upload caches under `app/static/uploads/`.
- **LLM orchestration:** Deterministic JD/evidence compilers run with caching in `app/static/cache/` and respect OpenAI settings pulled from `.env`.
- **Single-node friendly:** Everything (uploads, caches, templates, static assets) lives inside `app/`; only requires Python + ffmpeg + your API key.

---

## 🎥 Watch in Action

https://github.com/user-attachments/assets/05408a41-cbde-4c77-8563-ef4c75ce16a7

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.11+ (repo ships with `.python-version` pointing to `stimulus-ai-venv` for pyenv users).
- ffmpeg/ffprobe installed on the OS path (`ffmpeg -version` should succeed).
- OpenAI API access (models: `gpt-4o-mini` for schema/evidence, `whisper-1` for transcription).

### Quickstart
1. **Clone & enter the repo**
   ```bash
   git clone https://github.com/Huberdoggy/stimulus-ai.git stimulus-ai
   cd stimulus-ai
   ```
2. **Create/activate the virtualenv**
   ```bash
   pyenv virtualenv 3.11 stimulus-ai-venv   # or python -m venv .venv
   pyenv local stimulus-ai-venv             # activates via .python-version
   ```
3. **Install Python deps**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment**
   ```bash
   cp .env.example .env
   # edit PORT / OPENAI_API_KEY / LLM_* values as needed
   ```
   The app now auto-loads `.env` on startup—no need to export variables manually.
5. **Run the API (localhost only)**
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port ${PORT:-8000}
   ```
   Visit `http://127.0.0.1:8000` and you’ll be redirected to `/ui/adapter`.

### Workflow Tips
- Paste any JD into the left pane and click **Compile** to generate the schema (caching keeps repeat runs instant).
- Upload résumés (`PDF/DOCX/TXT/MD`) plus either an audio OR video artifact per candidate; ffmpeg writes artifacts under `app/static/uploads/` and sidecar metadata/transcripts alongside them.
- Hit **Build Evidence** once you have a JD schema and artifacts; coverage data and caches live under `app/static/cache/`.
- Clearing caches/uploads is as simple as removing the directories inside `app/static/{uploads,cache}`—they’re auto-created when needed and already ignored by git.

---

## 👤 Authorship and Entitlements

Stimulus.AI was created by [Kyle Huber](https://linkedin.com/in/kyle-james-my-filenames) in partnership with [Travis McBurney](https://linkedin.com/in/travis-mcburney). Together, the pair embarked a journey; designing a tool to serve as an ***adapter*** between AI-shaped job candidate insight and recruiter-shaped operational needs.

What you now have before you is the result of those endeavors.
Regardless of which end of the spectrum you fall, we hope that Stimulus.AI augments your existing processes with additional insights, enrichment, and efficiency.

Terms of use (and the *"as is"* disclaimer) are futher detailed in the [LICENSE](./LICENSE.md)
