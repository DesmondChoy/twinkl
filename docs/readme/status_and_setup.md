# Status and Setup

[Overview](../../README.md) | [Onboarding and Experience](onboarding_and_experience.md) | [Review Apps](review_apps.md) | [Research and Data](research_and_data.md) | **Status and Setup**

---

## Known Gaps

| Capability | Status | Note |
|---|---|---|
| Onboarding (SVBWS Values Assessment) | 🧪 Experimental | The React POC implements the complete local, user-facing flow and a versioned Profile. Manual Experience synchronizes the confirmed Profile with the in-memory Python boundary. Production multi-user storage and generalized persistence remain outside the capstone. |
| Coach Digest validation depth | ⚠️ Partial | All 27 saved replay weeks have baseline responses from the [prompt-4.4 voice refresh and repairs](../../logs/experiments/reports/coach_voice_refresh_20260909/report.md). Twenty-two eligible weeks have [paired responses with and without North Star Moment context](../north_star/demo_coach_comparison.md), with versioned inputs and checks. Ordinary live generation uses prompt `4.7`, complete displayed text from selected Journal Entries, exact quotations, three nonempty fields, and one generated question. Mechanical checks do not establish factual accuracy: the [source-context comparison](../../logs/experiments/reports/coach_context_eval_20260913/report.md) records attribution errors and missed evaluator flags. Historical August scores describe five earlier responses, not the current replay. The independent-provider and 42-Drift/42-control evaluation workstream is Done under its accepted tooling scope, with no committed paid result. Human calibration and the external user pilot are closed outside capstone scope without execution. |
| Experience and Inspect completion | 🚧 In Progress | The shared app, five saved Persona replays, manual Journal Entries, displayed nudges, Weekly Drift Detection, Coach Digest, Inspect, privacy notice, confirmed session deletion, and release checks are implemented. Longitudinal Core Value history is Done: the Reviewed week selector opens retained results, Coach Digest responses, and linked Inspect events across session restore and recomputation. Coach Digest feedback and the final professor walkthrough evidence remain open. |
| Displayed nudge user evidence | ⚠️ Not collected | Displayed nudge implementation is complete. The external user pilot is Not done and closed outside capstone scope. Response rate, continued journaling, and perceived relevance remain unmeasured. Saved replays and regression tests do not establish those user outcomes. |
| North Star Moment | 🧪 AI evaluation and saved replay | Full eligible history supplies exact supportive-action quotations and no-card outcomes across all five saved Personas. The [v4 Run 1 comparison](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md) covers 501 weeks from 105 synthetic Personas, with 38 reassessed cases and 463 retained observations. Human review and real-user benefit remain unestablished. An accepted selection appears after the Coach Digest narrative and question. Saved replay can switch between paired generated responses; manual Experience preserves its original response and can recover an eligible historical Moment. Live execution uses a pinned US$1 allowance shared across sessions and restarts; invalid or exhausted budgets fail closed. |
| Embedding Explorer | ✅ Complete | Interactive 3D visualization of VIF Critic (Offline) embeddings |
| Drift Detector validation and deployment approval | ⚠️ Not claimed | The deterministic Drift Detector and Luna-low Weekly Drift Reviewer runtime are complete and wired for the capstone POC, with versioned receipts and fail-closed abstention. Prompt `4.0` includes the selected Core Values' definitions and core motivations. Its [comparison with v3](../../logs/experiments/reports/experiment_review_2026-09-07_twinkl_j3k7_core_value_definitions.md) measures changes in detections across three Runs per variant; it does not establish accuracy. Evidence remains AI-reviewed synthetic development evidence without a fresh final test or deployment approval. |
| Journaling anomaly radar | ❌ Not Started | Cadence/gap detection beyond the current prototype-router tooling |
| Goal-aligned inspiration feed | ❌ Not Started | External API integration |

The [capstone closeout decisions](../prd.md#capstone-closeout-decisions) record
the final Done or Not done status and accepted scope for history, independent
evaluation, and human studies. For the full breakdown, see the
[Implementation Status](../prd.md#implementation-status) table.

## Common Commands

Run these commands from the repository root with `.venv` activated, except
where a frontend directory is specified. `uv run` uses the project environment.

- Launch the annotation tool: `uv run shiny run src/annotation_tool/app.py`
- Launch the Experience and Inspect Python boundary: `uv run uvicorn src.demo.api:app --port 8000`
- Launch the React development server from `frontend/onboarding/`: `npm run dev`
- Run the React checks from `frontend/onboarding/`: `npm test`, `npm run test:watch`, `npm run typecheck`, and `npm run build`
- Run deterministic Chromium checks from `frontend/onboarding/`: `npm run test:e2e` (install the browser once with `npx playwright install chromium`)
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
- Verify the prompt-v4 GPT-5.6/GPT-6 Luna comparisons without API calls: `uv run python -m scripts.experiments.replay_luna_comparison q6pt verify` and `uv run python -m scripts.experiments.replay_luna_comparison gv3x verify`
- Replay recall-aware checkpoint selection from saved traces without retraining: `uv run python scripts/experiments/replay_recall_aware_checkpoint_selection.py`
- Inspect Coach completion options without provider calls or file writes: `uv run python -m scripts.coach.complete_scenario_coach --help`
- Build the deterministic Coach Digest Drift/control target catalog without provider calls: `uv run python scripts/experiments/run_coach_drift_control_eval.py`
- Dry-run cross-provider Coach Digest Evals over the historical August five-response manifest: `uv run python -m src.evals.coach_narrative_judge --manifest logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json --judge-provider gemini`
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

The prompt-v4 model-comparison runners, `compare_twinkl_q6pt_luna`
(`none`/`low`) and `compare_twinkl_gv3x_luna_higher` (`medium`/`high`/`xhigh`),
expose `prepare`, `verify`, `smoke`, `run`, and `score` with `--config` and
`--execute`. Use `replay_luna_comparison` for offline `verify` and `score`
after the live model migration. It checks the recorded Reviewer source and
allows only the model ID change. `score` rewrites the saved `metrics.json`.
`smoke` and `run` make paid calls, require `--execute`, and resume from
recorded terminal responses under the configured spend ceiling.

The Drift/control runner accepts source and output overrides through
`--episodes-parquet`, `--case-outcomes-parquet`, `--wrangled-dir`,
`--parquet-path`, `--output-dir`, `--manifest-out`, and `--targets-out`.
`--group {drift,control,both}`, `--limit`, and `--seed` control selection.
`--resume` preserves completed target IDs, and `--execute` authorizes paid
Weekly Drift Reviewer and Coach Digest calls. See the
[Coach Digest test and eval guide](../../docs/evals/coach_narrative_test_and_eval_guide.md)
before using either paid command.

### Weekly Drift and North Star Moment experiments

The experiment runners expose the following commands and options:

| Module (`python -m …`) | Commands | Options |
|---|---|---|
| `scripts.experiments.weekly_drift_definitions` | `prepare`, `run`, `report`, `verify` | `--output PATH` selects the frozen experiment directory. `run --limit N` executes at most N incomplete requests and retains them for resumption. |
| `scripts.experiments.nsm_experiment` | `prepare`, `run`, `report`, `verify` | `--record PATH` selects the consolidated original study JSON. `--concurrency N` accepts 1–32, default 8. |
| `scripts.experiments.nsm_targeted_update` | `audit`, `prepare`, `run`, `report`, `verify` | `--output PATH` selects a separate update directory; it cannot be the original study directory. `--concurrency N` accepts 1–32, default 8. |

These runners' `run` commands make paid provider calls for incomplete requests;
they have no `--execute` flag. Completed receipts are reused on resume.
`verify` checks frozen inputs and code without provider calls. `report` also
makes no provider calls but writes derived results. `audit` writes the targeted
update's impact audit, and `prepare` freezes inputs and may perform local Nomic
encoding. Preserve the published records when choosing output paths.

Reproduce the frozen 105-Persona cohort without provider calls or file writes:

```sh
uv run --no-sync python -m scripts.experiments.north_star_cohort_selection
```

Verify the saved Weekly Drift definitions study, or inspect runner options:

```sh
uv run --no-sync python -m scripts.experiments.weekly_drift_definitions verify
uv run --no-sync python -m scripts.experiments.nsm_targeted_update --help
uv run --no-sync python -m scripts.experiments.nsm_experiment --help
```

Exact NSM verification requires the recorded study source versions. The
published targeted-update revision `f7e14ebb` matches its 15 frozen code hashes;
the current application contracts differ, so running its `verify` command
against the current checkout rejects that mismatch. Use the
[methodology's reproduction guidance](../north_star/nsm_experiment_methodology.md)
for the original study and the
[targeted-update report](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md)
for its retained observations, audit, and regrading procedure. The saved replay
export instead validates the completed-study hash and matching semantic inputs;
see [Experience and Inspect](../demo/experience_inspect_app.md).

The `north_star_phase0*`, `north_star_luna`, `north_star_encoder_probe`,
and `north_star_integration` scripts support historical
preparation and diagnostics. Their experiment outputs do not supply the current
NSM comparison; use the two `nsm_*` runners and linked frozen records above.

Historical `north_star_saved_checks` outputs are retained as research records;
the maintained commands are the `nsm_*` runners above. Historical receipt tests
replay temporary source archives at the published study revisions; current
runtime tests use this checkout.
The full suite requires local Git history containing those recorded revisions.

### Saved Persona Coach Digest responses

All 27 replay weeks have baseline Coach Digest responses; 22 weeks with an
accepted North Star Moment also have paired responses. Saved responses use their
recorded prompt and validation policy. Rebuild public scenarios without provider
calls:

```sh
uv run python -m scripts.export_demo_experiments
```

`uv run python -m src.demo.scenarios` rebuilds only the scenario bundles from
existing compact North Star Moment records. Both exporters verify provenance
and input compatibility before retaining saved responses.

The Coach runners prepare frozen plans without provider calls by default. Use
a fresh directory under the repository for each plan; their defaults name
preserved historical runs whose input or policy hashes may differ from the
current checkout:

```sh
uv run python -m scripts.coach.complete_scenario_coach \
  --output logs/experiments/reports/coach_completion_local
uv run python -m scripts.coach.refresh_scenario_coach \
  --output logs/experiments/reports/coach_refresh_local
uv run python -m scripts.coach.compare_scenario_coach \
  --output logs/experiments/reports/coach_comparison_local
```

| Runner | Purpose | Options |
|---|---|---|
| `complete_scenario_coach` | Retain compatible responses and generate missing weeks | `--output PATH`; `--execute` permits paid generation; `--repair-requirements PATH` supplies a frozen JSON mapping of case keys to repair instructions. |
| `refresh_scenario_coach` | Generate a complete replacement baseline with preserved attempts | `--output PATH`; `--execute`; `--apply` installs a complete validated fixture; `--prior-run PATH` and a nonempty `--repair-requirements PATH` must be supplied together to repair selected cases while retaining other compatible responses. |
| `compare_scenario_coach` | Generate paired responses with and without selected North Star Moment context | `--output PATH`; `--execute`; `--apply`; `--prior-run PATH`; `--editorial-repairs PATH`; repeatable `--case scenario::week-start` limits a pilot. Applying a pilot requires compatible coverage for the other eligible weeks. |

These commands use Luna-none and record requests, attempts, validation, and
provider usage. They make no Weekly Drift Reviewer or North Star Moment calls.
`--execute` makes paid Coach Digest calls; `--apply` installs accepted saved
responses without requiring further generation. Run the scenario exporter
separately after installing responses. Frozen plans reject incompatible
inputs or generation policies. See the [Coach Digest runbook](../evals/coach_narrative_test_and_eval_guide.md#3a-complete-saved-replay-weeks-and-retain-compatible-responses)
and [comparison guide](../north_star/demo_coach_comparison.md) for retry limits,
repair formats, and current saved evidence.

The historical `generate_approved_judge_sample.py --reuse-scenario-key-weeks`
command targets five key weeks and replaces its response fixture. Its default
fixture path is unsuitable for maintaining the 27-week replay. The preserved
August manifest is an evaluation subset with a different Persona roster.

### Coach Digest source-context experiment

Prepare the fixed short-excerpt/complete-entry comparison without provider calls:

```sh
uv run python -m scripts.experiments.run_coach_context_eval \
  --out /tmp/twinkl-coach-context
```

The required `--out PATH` receives the frozen plan, inputs, and source hashes.
Adding `--execute` makes six Coach Digest generation calls and nine AI evaluator
calls. This comparison uses archived Coach prompt `4.6` and evaluator `3.1`;
it does not follow ordinary prompt `4.7` generation. Completed calls are reused
on resume. The [runbook](../evals/coach_narrative_test_and_eval_guide.md#source-context-comparison)
describes compact receipts, source verification, and the limits of the result.

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

   Onboarding, saved Persona replay, and the Drift Inspection App require no
   provider key. Start the live Python boundary with
   `uv run uvicorn src.demo.api:app --env-file .env --port 8000` to load `.env`.
   Live North Star Moment review uses the policy in
   [`config/evals/north_star_live_v1.json`](../../config/evals/north_star_live_v1.json)
   and its private ledger under `logs/exports/demo_tool_runs/north_star/`.
   The US$1 cap applies across sessions and restarts, with at most two attempts
   per exact request and no SDK retries. Missing provider configuration or
   invalid/exhausted accounting leaves the weekly result available without a
   Moment.

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

### Frontend and browser checks

Install frontend dependencies from the committed lockfile and run the checks:

```sh
cd frontend/onboarding
npm ci
npm test
npm run typecheck
npm run build
npx playwright install chromium
npm run test:e2e
```

`npm run test:watch` keeps unit tests running during editing. Playwright supports
`npm run test:e2e -- --project=desktop-chromium` or `--project=narrow-chromium`
for one viewport. It builds the app, starts the controlled Python boundary on
port `8765`, and uses test doubles for model calls. The port must be free.
Failures retain screenshots and traces under `frontend/onboarding/test-results/`.
The checks cover replay, Inspect context and reload, onboarding, manual writing,
closed-week review, Coach Digest retry, historical Moment recovery, source-dialog
focus restoration, and session deletion. They establish application behavior,
not model quality or human validation.

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
