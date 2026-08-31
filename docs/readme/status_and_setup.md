# Status and Setup

[Overview](../../README.md) | [Onboarding and Experience](onboarding_and_experience.md) | [Review Apps](review_apps.md) | [Research and Data](research_and_data.md) | **Status and Setup**

---

## Known Gaps

| Capability | Status | Note |
|---|---|---|
| Onboarding (SVBWS Values Assessment) | 🧪 Experimental | The React POC implements the complete local, user-facing flow and a versioned Profile. Manual Experience synchronizes the confirmed Profile with the in-memory Python boundary. Production multi-user storage and generalized persistence remain outside the capstone. |
| Coach Digest validation depth | ⚠️ Partial | The same five accepted key-week responses appear in the Persona replay fixtures and the [evaluation manifest](../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json). All five passed all Coach Digest Validations. Coach Digest Evals scored mean correctness `4.80`, specificity `5.00`, non-prescriptive tone `5.00`, and tension honesty `4.60`; all reflective questions passed, with no failed verdicts or review flags. These scores are same-model AI review, not human validation. Cross-provider evaluator options and the deterministic 42-Drift/42-control study are implemented, but no paid independent-provider result is committed. Future human calibration of the AI review remains incomplete. |
| Experience and Inspect completion | 🚧 In Progress | The shared app, five saved Persona replays, manual Journal Entries, displayed nudges, Weekly Drift Detection, Coach Digest, Inspect, privacy notice, confirmed session deletion, and release checks are implemented. Coach Digest feedback, longitudinal Core Value history, and the final professor walkthrough evidence remain open. |
| Displayed nudge user evidence | ⚠️ Not collected | Displayed nudge implementation is complete. A future external pilot can measure response rate, continued journaling, and perceived relevance. Saved replays and regression tests do not establish those user outcomes. |
| Embedding Explorer | ✅ Complete | Interactive 3D visualization of VIF Critic (Offline) embeddings |
| Drift Detector validation and deployment approval | ⚠️ Not claimed | The deterministic Drift Detector and Luna-low Weekly Drift Reviewer runtime are complete and wired for the capstone POC, with versioned receipts and fail-closed abstention. The evidence is AI-reviewed synthetic development evidence; no fresh final test was run, so no deployment approval is claimed. |
| Journaling anomaly radar | ❌ Not Started | Cadence/gap detection beyond the current prototype-router tooling |
| Goal-aligned inspiration feed | ❌ Not Started | External API integration |

For the full breakdown, see the [Implementation Status](../../docs/prd.md#implementation-status) table in prd.md.

## Common Commands

Examples below use `uv run` so they pick up the project environment directly. Activating `.venv` manually also works.

- Launch the annotation tool: `uv run shiny run src/annotation_tool/app.py`
- Launch the Experience and Inspect Python boundary: `uv run uvicorn src.demo.api:app --port 8000`
- Launch the React development server from `frontend/onboarding/`: `npm run dev`
- Run the React checks from `frontend/onboarding/`: `npm test`, `npm run typecheck`, and `npm run build`
- Regenerate the Experience and Inspect contracts: `uv run python -m src.demo.export_contract_schema`
- Launch the Drift Inspection App: `uv run shiny run --host 127.0.0.1 --port 8000 --no-dev-mode src/drift_review_app/app.py`
- Deploy the Drift Inspection App to Railway: `railway up`
- Launch the Runtime Demo Review App: `uv run shiny run src/demo_tool/app.py`
- Run the Runtime Demo Review App directly on port `8001`: `uv run python src/demo_tool/app.py`
- Run the paid Weekly Drift Detection path: `uv run python -m src.coach.weekly_drift_runtime --persona-id 0a2fe15c --execute`
- Reproduce the deprecated VIF Critic (Offline) compatibility path: `uv run python -m src.coach.runtime --persona-id 0a2fe15c --checkpoint-path logs/experiments/artifacts/.../selected_checkpoint.pt`
- Build compatibility Weekly Drift Detection output from persisted LLM-Judge VIF Labels: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c`
- Build compatibility Weekly Drift Detection output from saved VIF Critic Predictions: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c --signals-path logs/exports/weekly_coach/0a2fe15c_vif_timeline.parquet`
- Train the VIF Critic (Offline) with CLI overrides and LR-finder export: `uv run python -m src.vif.train --grad-clip 1.0 --lr-find-output-path logs/exports/lr_find.png`
- Run the BNN baseline: `uv run python -m src.vif.train_bnn --epochs 10 --batch-size 16`
- Generate the embedding explorer without auto-opening a browser: `uv run python -m src.vif.extract_embeddings --checkpoint logs/experiments/artifacts/.../selected_checkpoint.pt --no-browser`
- Prepare a deterministic consensus pilot bundle: `uv run python scripts/journalling/twinkl_754_prepare_consensus.py --pilot-size 50 --pilot-hard-dimensions security,hedonism,stimulation`
- Reproduce the default consensus-label Drift EDA with runtime-compatible week bins: `uv run python scripts/drift/trajectory_eda.py`
- Compare persisted LLM-Judge VIF Labels with week bins anchored to the first Journal Entry: `uv run python scripts/drift/trajectory_eda.py --labels judge --week-mode persona_anchor`
- Estimate the LLM context baseline cost without making API calls: `uv run python scripts/experiments/llm_critic_baseline.py estimate --split test --context-arms student_visible human_context`
- Re-score the frozen Weekly Drift Reviewer model comparison: `uv run python -m scripts.experiments.compare_twinkl_52zz_models score`
- Re-score the Luna higher-reasoning comparison: `uv run python -m scripts.experiments.compare_twinkl_ck3w_luna_higher_reasoning score`
- Replay recall-aware checkpoint selection from saved traces without retraining: `uv run python scripts/experiments/replay_recall_aware_checkpoint_selection.py`
- Build the deterministic Coach Digest Drift/control target catalog without provider calls: `uv run python scripts/experiments/run_coach_drift_control_eval.py`
- Dry-run cross-provider Coach Digest Evals over the committed five-response manifest: `uv run python -m src.evals.coach_narrative_judge --manifest logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json --judge-provider gemini`
- After paid Drift/control generation and Coach Digest Evals, build the saved comparison report: `uv run python -m src.evals.coach_drift_control_report --manifest logs/experiments/reports/coach_digest_drift_control/judge_sample_manifest.json --eval-metrics logs/experiments/reports/coach_digest_drift_control/evals/metrics.json --out logs/experiments/reports/coach_digest_drift_control/comparison`
- Regenerate the capstone report figures: `MPLCONFIGDIR=/tmp/twinkl-matplotlib uv run python scripts/capstone/generate_report_figures.py`
- Render the capstone report PDF: `quarto render docs/capstone_report/capstone_project_report.md --to pdf`

The Drift EDA accepts `--labels {consensus,judge}` (default: `consensus`) and `--week-mode {runtime,persona_anchor}` (default: `runtime`). The LLM baseline exposes `estimate`, `run`, `score`, and `report`; `run` writes dry-run records unless `--execute` is supplied.

The `twinkl-52zz` model-comparison runner exposes `prepare`, `estimate`, `run`,
and `score`; `run` requires `--execute` and accepts
`--model-key {all,gpt_5_4_mini,gpt_5_6_luna}`. The Luna reasoning-effort runner
exposes `prepare`, `smoke`, `run`, and `score`; its paid `smoke` and `run`
commands require `--execute`. Both runners accept `--root` and `--config`.
Those global options must precede the subcommand.

The Drift/control runner accepts source and output overrides through
`--episodes-parquet`, `--case-outcomes-parquet`, `--wrangled-dir`,
`--parquet-path`, `--output-dir`, `--manifest-out`, and `--targets-out`.
`--group {drift,control,both}`, `--limit`, and `--seed` control selection.
`--resume` preserves completed target IDs, and `--execute` authorizes paid
Weekly Drift Reviewer and Coach Digest calls. See the
[Coach Digest test and eval guide](../../docs/evals/coach_narrative_test_and_eval_guide.md)
before using either paid command.

## Setup

This repo uses `uv` and `pyproject.toml` for dependency management.

1. Install `uv`:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   (Or see https://docs.astral.sh/uv/getting-started/installation/ for other methods)

2. Create the virtual environment:
   ```sh
   uv venv
   ```
3. Activate it when you want an interactive shell (Fish shell preferred in this repo):
   ```sh
   source .venv/bin/activate.fish
   ```
   Bash/Zsh fallback:
   ```sh
   source .venv/bin/activate
   ```
4. For provider-backed commands, create a `.env` file in the project root with
   the key for each selected provider:
   ```sh
   OPENAI_API_KEY=your-api-key-here
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

   Gemini also accepts `GOOGLE_API_KEY`. `TWINKL_COACH_PROVIDER` selects
   `openai` or `gemini` for Coach Digest generation, and
   `TWINKL_COACH_MODEL` overrides that provider's default. Coach Digest Evals
   can select an independent evaluator with `--judge-provider` and
   `--judge-model`.

   The Drift Inspection App reads committed files and does not require an API
   key.

### Installing dependencies

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

- Install everything from the lockfile:
  ```sh
  uv sync
  ```
- Install the development group with pytest, pytest-asyncio, and Ruff:
  ```sh
  uv sync --group dev
  ```

### Running tests

Install the dev dependencies first:

```sh
uv sync --group dev
```

Run the full pytest suite:

```sh
uv run pytest
```

Run Ruff on the Python files touched by a change:

```sh
uv run ruff check path/to/changed_file.py tests/path/to/changed_test.py
```

Run the pinned MyPy version on Python files whose typed behavior changed:

```sh
uv run --with 'mypy==2.3.0' mypy path/to/changed_file.py
```

The repository still contains historical notebook and test lint debt, so a
repo-wide `uv run ruff check .` is diagnostic rather than a clean gate.

Run the deterministic local end-to-end smoke pipeline only:

```sh
uv run pytest tests/e2e -q
```

This smoke test exercises the offline path `synthetic_data -> wrangled markdown -> consolidated LLM-Judge VIF Labels -> VIF Critic (Offline) training` using tiny local fixtures and a mock text encoder, so it does not require live LLM calls.

### Adding a dependency

Use `uv add` to both install into the environment and record it in
`pyproject.toml`:

```sh
uv add <package>
```

Pin an exact version if desired:

```sh
uv add "<package>==<version>"
```

After adding, `uv` updates `uv.lock` automatically.

### Exporting requirements.txt (optional)

Only needed for legacy tooling or platforms that require it:

```sh
uv export -o requirements.txt
```
