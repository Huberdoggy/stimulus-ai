# STABILITY ROADMAP

## Start Here

**Review the following files in the dirty working tree:**

- [artifacts.py](https://replit.com/@KJHuber/StimulusAI#app/routes/artifacts.py)
- [evidence.py](https://replit.com/@KJHuber/StimulusAI#app/routes/evidence.py)
- [jd.py](https://replit.com/@KJHuber/StimulusAI#app/routes/jd.py)
- [llm.py](https://replit.com/@KJHuber/StimulusAI#app/services/llm.py)
- [transcribe.py](https://replit.com/@KJHuber/StimulusAI#app/services/transcribe.py)

## Scope of Work:

### Goals:
1) **(PRIORITY) Fix broken back-end logic in the current state.** Relevant files modified since last commit hash ***ed242815e12f84620311d3abce18df593539e947*** are listed above.
   - Symptoms observed -> Test artifacts previously used as control variables now return 0% match. Can't progress past Test 1 (below)
3) **Make results consistent from run to run, and remove old *“Dry mode”* paths no longer in use.**

- Intended additions

  - A 7-day cache so we can reuse past results when nothing has changed.

  - Stricter model settings (turn “randomness” off).

  - Deterministic post-processing (we sort and select results in one fixed way every time).

  - Auto-cleanup for expired cache files (no cron jobs; it cleans itself when you use the app).

- Intended removals

  - All Dry/Live backend switches — the app will just run Live always.

  - The lexicon file (done) and any code that depended on it.


- New folders (auto-created as needed):

  - app/static/cache/schemas/

  - app/static/cache/evidence/

### Edge-Case Tests (minimal combos)

| Test | Why It Matters | Actions I Will Perform | Expectation
| --- | --- | --- | --- |
| Test 1 — Cache “hit” (same everything) | Proves -> reuse everything works, no new model calls). | Run the app with Test JD + Resume A. Let it finish. Run the exact same thing again (no changes). | Expect: the second run is much faster and the numbers are identical.
| Test 2 — New candidate file, same JD (names/order must not change) | Proves -> reuse schema, recompute only evidence). | Keep the same Test JD. Replace Resume A with a slightly different file (Resume B) or change just that one artifact. Run again. | Expect: The domain/theme names and their order match the first run exactly. The total requirement count (denominator) is the same. Coverage numbers may change only because the résumé content changed — not because sections changed. |
Test 3 — New JD (compiles once, then reuses) | Proves -> fresh schema, then reuse.| Make a small, meaningful edit to the Test JD (e.g., add or remove one requirement line). Run with Resume A. Run the same edited JD a second time with Resume A again. | First run compiles a new schema; second run is faster and identical. If I later revert to the original Test JD, it reuses that schema again (if within 7 days). |
Test 4 — Expired cache cleans itself (no cron jobs) | Proves -> auto-cleanup removes old cache on use. | Create a harmless “dummy” cache file and modify older than 7 days, then trigger the app to clean it.<br># Create a tiny dummy file:<br>``` echo '{}' > app/static/cache/schemas/ttl_probe.schema.json ```<br># Pretend it’s older than 7 days (set its timestamp into the past): ``` touch -d "8 days ago" app/static/cache/schemas/ttl_probe.schema.json ```<br># Confirm it's there and looks "old": ``` ls -la app/static/cache/schemas ```<br>Trigger cleanup by running the app once. Program performs a small sweep before work, and deletes expired files on access. | Dummy file has vanished |

### Undesired for this Sprint:
- UI changes
- Pull requests
- Running tests on my behalf (the API key has been intentionally omitted from your sandboxed environment).