# MIGRATION CHECKLIST

## Scope of Work:

### Goals:
1) Ensure the Stimulus.AI program continues to function as designed, post-migration from Replit to local machine.
2) Run a formatting pass; using Black from the virtual environment as is standard convention per [AGENTS.md](../AGENTS.md)
3) Enumerate on the following components of our documentation;ensure an appropriate balance is struck between precise technical instruction, clarity, and detailing our current feature implementations:
    - [primary description (first heading)](../README.md#stimulusai)
    - [Setup & Usage](../README.md#️-setup--usage)
    - The dot env template [.env.example](../.env.example)

### Acceptance Tests (To Be Performed Interactively By Kyle)
| Action | Expectation |
| --- | --- |
| Entrypoint command from superseded Replit config is run:</br>```python -m uvicorn app.main:app --host 127.0.0.1 --port ${PORT:-8000}```</br>Web server - up; desired: relevant code is scoped to listen on localhost only | Accessible at http://127.0.0.1:8000; automatic redirect to the **`/ui/adapter`** endpoint is still functional. |
| All configuration options are declared in local **`.env`** | Seamless transition; existing program logic is equipped to anticipate these; fallback defaults remain in place. |
| Entire flow is run from a clean slate; new multi-media artifacts uploaded | OS-level ffmpeg handles parsing/transcoding as expected, caches are lazily recreated on the filesystem, all 4 tests outlined in the earlier [stability_roadmap](./stability_roadmap.md#edge-case-tests-minimal-combos) continue to come back green. |