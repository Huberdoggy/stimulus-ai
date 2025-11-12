# Welcome Codex!

You’re joining a build that rewards taste as much as technical precision. Stimulus AI wasn’t spun up as another “let’s throw a model at hiring” experiment — it’s a living prototype that proves how well-curated evidence can outclass traditional résumés. We’ve reached the point where the architecture is mature enough to run, but lean enough that a strong full-stack developer can shape its next form with very little friction.

If you care about clean component boundaries, atomic commits, and pipelines that explain themselves, you’ll fit right in. The groundwork is solid — the kind of project where a senior engineer can immediately leave fingerprints that matter. Think of this as your invitation to take a beautifully drafted melody and orchestrate the rhythm section around it.

**Product Mission & Technical Snapshot**

Stimulus AI transforms hiring from a paperwork exercise into a proof-based evaluation. The system ingests candidate artifacts — résumés, audio, and video — and measures them against a compiled job description to produce visualized “evidence maps.” Recruiters see, at a glance, where each candidate demonstrates the required skills and where gaps exist.

**What’s implemented today:**

- A fully operational front-end with animated load states, DOM intro, and video preview thumbnails with codec badges.
- ffmpeg-based upload and transcoding, allowing multi-format media handling.
- An LLM pipeline that compiles job requirements into structured schemas and matches candidate evidence accordingly.
- Real-time coverage visualization — nested bullets and token highlights display what evidence supports each requirement.
- Modularized architecture under ***app/***

**Why it matters:**
Stimulus doesn’t just show that a candidate can talk about a skill — it shows where that skill lives inside their real work samples. It’s a recruiter’s microscope and a candidate’s amplifier rolled into one.

## Your Mission:
- Fortify the foundations and help us move from prototype to production-grade reliability without losing the elegance that makes the tool human.

### \!\! Standing Directives \!\!

#### Sprint Initiatives:
- Consult documentation under ***milestones/*** at the start of each new session. This directory will contain specific action items requiring your attention.

#### Concerning files that live outside of app/

| Okay to touch | No touch zones |
| --- | --- |
| [README](./README.md) -> ensure this stays up to date (for internal personnel only). | **`.env`** ***( [example template is permissible](./.env.example) )*** |
| [requirements.txt](./requirements.txt) as needed. | Anything not explicitly mentioned as being 'okay'. |

- Python files should be formatted in accordance with Black conventions; use commandline arguments specified in ***`.vscode/settings.json`***.
