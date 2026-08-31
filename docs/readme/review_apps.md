# Review Apps

[Overview](../../README.md) | [Onboarding and Experience](onboarding_and_experience.md) | **Review Apps** | [Research and Data](research_and_data.md) | [Status and Setup](status_and_setup.md)

---

## Human Annotation Tool — ✅ Complete

Measures overlap between project-team annotations and LLM-Judge VIF Labels across 10 Schwartz value dimensions. Annotators score the shared subset before seeing the LLM-Judge VIF Labels; the tool then computes Cohen's κ and Fleiss' κ. This bounded sample does not establish independent human validation or label accuracy.

**Run the tool:**
```sh
uv run shiny run src/annotation_tool/app.py
```

Open `http://127.0.0.1:8000` in your browser.

**Features:**
- Displays persona context (name, age, profession, culture, Core Values, collapsible bio)
- Shows Journal Entries with nudge/response threading
- 10-value LLM-Judge VIF Label grid with `-1` (Conflict), `0` (neutral), and
  `+1`, with CSS tooltips for Schwartz value definitions
- Progress tracking per annotator
- Annotations persisted to `logs/annotations/<annotator>.parquet`
- **Analysis & Metrics panel** — Computes inter-annotator agreement (Cohen's κ, Fleiss' κ)
- **Export functionality** — CSV, Parquet, and Markdown report formats
- **Comparison view** — Inline display of project-team annotations and LLM-Judge VIF Labels for review

**Key files:**
- `src/annotation_tool/app.py` — Main Shiny application
- `src/annotation_tool/data_loader.py` — Loads entries from wrangled files
- `src/annotation_tool/annotation_store.py` — Persists annotations with file locking
- `src/annotation_tool/agreement_metrics.py` — Kappa calculations and export
- `src/annotation_tool/components/` — Modular UI components (scoring grid, comparison view, analysis)
- `src/annotation_tool/state.py` — Centralized state management
- `docs/pipeline/annotation_tool_plan.md` — Full implementation plan

## Drift Inspection App — ✅ Complete

The read-only Python Shiny app compares Runs 1–3 for three frozen Weekly Drift
Reviewer setups: `gpt-5.4-mini` at reasoning effort `none`, `gpt-5.6-luna` at
reasoning effort `none`, and `gpt-5.6-luna` at reasoning effort `low`. It shows
complete development results, persona-level outcomes, Journal Entries,
AI-reviewed LLM-Judge Conflict Labels, Weekly Drift Reviewer Decisions, and Run
variability without merging Runs or calculating a majority vote.
The first two setups are historical comparisons; `gpt-5.6-luna` at reasoning
effort `low` is the fixed Weekly Drift Reviewer model contract.

**Run the app:**

```sh
uv run shiny run --host 127.0.0.1 --port 8000 --no-dev-mode \
  src/drift_review_app/app.py
```

Open `http://127.0.0.1:8000`. The `drift-review-app` entry in
`.claude/launch.json` runs the same command.

**Features:**

- filters for known Drift status and Core Value before persona selection
- complete development summaries for known Drift hits, false Drift alerts,
  coverage, and all preserved Runs
- persona scoreboards with exact known Drift and Drift alert spans
- side-by-side Journal Entries, LLM-Judge Conflict Labels, Weekly Drift
  Reviewer Decisions, cited evidence, and verified weekly cutoffs
- fail-closed checks for prompt hashes, setup identities, model identifiers,
  reasoning effort, joins, counts, and aggregate parity
- no model or provider API calls; all inputs are committed research files

`railway.json` deploys the app with `Dockerfile.review_app`; `railway up` starts
a Railway deployment from the repository. The container uses Railway's `PORT`
and needs no database or persistent volume.

See [`docs/demo/weekly_drift_review_app.md`](../../docs/demo/weekly_drift_review_app.md)
for the review contract, input boundary, launch options, and frozen files.

**Key files:**

- `src/drift_review_app/app.py` — Shiny interface
- `src/drift_review_app/data.py` — frozen-input loading and validation
- `src/drift_rules.py` — shared deterministic Drift rules
- `Dockerfile.review_app`, `requirements-review-app.txt`, and `railway.json` —
  deployment boundary

## Runtime Demo Review App — 🧪 Deprecated Experimental Compatibility

A sibling Shiny app for inspecting the deprecated VIF Critic (Offline) crash/rut/evolution compatibility path. It remains available for historical demonstrations but does not execute the approved Weekly Drift Reviewer and Drift Detector runtime.

**Run the app:**
```sh
uv run shiny run src/demo_tool/app.py
```

Open `http://127.0.0.1:8000` in your browser when running via `shiny run`.

To launch the same app directly with Python:
```sh
uv run python src/demo_tool/app.py
```

Open `http://127.0.0.1:8001` when running the file directly.

**Features:**
- Persona browser with full timeline, nudges, responses, and collapsible bio
- Checkpoint catalog sourced from `logs/experiments/artifacts`, `models/vif`, and `logs/experiments`
- Cached output loading for previously run persona/checkpoint pairs
- End-to-end runtime execution via `src.coach.runtime.run_weekly_coach_cycle`
- Detector input source toggle between **LLM-Judge VIF Labels** and **VIF Critic Predictions**
- Detector comparison across **Baseline**, **EMA**, **CUSUM**, **Cosine**, **Control Chart**, and **KL Div**, with per-Journal Entry detector-vote counts (not the five-pass LLM-Judge reference)
- A six-tab result canvas. Its old `Weekly Digest` tab name is a compatibility label.
- A live Coach Digest response with `weekly_mirror`, `tension_explanation`, and `reflective_question` sections

**Coach Digest response:** `src/coach/llm_client.py` builds the provider-backed
callable that `src/demo_tool/runtime_bridge.py` injects into
`run_weekly_coach_cycle`. `TWINKL_COACH_PROVIDER` selects `openai` (default) or
`gemini`; `TWINKL_COACH_MODEL` overrides the per-provider default model
(`gpt-5.6-luna` or `gemini-3.5-flash`). OpenAI calls use reasoning effort
`none`. When the selected provider's API key is absent, the provider is
unrecognised, or the request fails, the app keeps its structured output without
a response. The app stays runnable offline. The `src.coach.runtime` and
`src.coach.weekly_digest` CLIs do not call a live Coach Digest LLM. They render
and persist the prompt only.

**Generated files:** The app writes persona/checkpoint-specific runtime bundles under `logs/exports/demo_tool_runs/<persona_id>/<checkpoint-stem>-<hash>/`.

See [`docs/demo/review_app.md`](../../docs/demo/review_app.md) for the full workflow
and file layout. This app is distinct from the read-only
[`Drift Inspection App`](../../docs/demo/weekly_drift_review_app.md), which compares
frozen Weekly Drift Reviewer Runs and does not execute the VIF Critic (Offline) runtime.

**Key files:**
- `src/demo_tool/app.py` — Main Shiny demo application
- `src/demo_tool/data_loader.py` — Persona catalog and chronological timeline loading
- `src/demo_tool/runtime_bridge.py` — Checkpoint discovery and Coach Digest runtime wrapper
- `src/coach/llm_client.py` — Provider-backed Coach Digest response adapters (Gemini, OpenAI)
- `src/demo_tool/multi_drift.py` — Multi-detector comparison bundle for LLM-Judge VIF Label and VIF Critic Prediction views
- `src/demo_tool/state.py` — Centralized reactive UI state
