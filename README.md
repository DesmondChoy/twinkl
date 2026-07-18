# Twinkl

Twinkl is an "inner compass" that helps users align their daily behavior with their long-term values. Unlike traditional journaling apps that summarize moods and topics, Twinkl maintains a dynamic self-model of the user's declared priorities and surfaces tensions when behavior drifts from intent. It answers the question: *"Am I living in line with what I said I value?"*

> **Project status:** Twinkl is under active development as an academic capstone. Status markers in each section indicate capability maturity:
> ✅ Complete · 🧪 Experimental · ⚠️ Partial · 📋 Specified · ❌ Not Started
>
> See [Known Gaps](#known-gaps) for the summary and [Implementation Status](docs/prd.md#implementation-status) for the full breakdown.

## Documentation Guide

- [`docs/prd.md`](docs/prd.md) — authoritative product intent and implementation status
- [`docs/architecture/e2e_architecture.md`](docs/architecture/e2e_architecture.md) — high-level end-to-end architecture map with unresolved product decisions called out
- [`docs/vif/01_concepts_and_roadmap.md`](docs/vif/01_concepts_and_roadmap.md), [`docs/vif/02_system_architecture.md`](docs/vif/02_system_architecture.md), [`docs/vif/03_model_training.md`](docs/vif/03_model_training.md), [`docs/vif/04_uncertainty_logic.md`](docs/vif/04_uncertainty_logic.md) — VIF design, runtime, training, and uncertainty logic
- [`docs/pipeline/pipeline_specs.md`](docs/pipeline/pipeline_specs.md), [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md), [`docs/pipeline/consensus_rejudging_instructions.md`](docs/pipeline/consensus_rejudging_instructions.md) — data generation, label datasets, and consensus diagnostics
- [`docs/drift/trajectory_eda.md`](docs/drift/trajectory_eda.md) — historical Drift-definition analysis comparing five-pass LLM-Judge consensus with persisted labels
- [`docs/evals/drift_v1_student_visible_target.md`](docs/evals/drift_v1_student_visible_target.md) — historical five-Drift development result and withheld former final-test score
- [`docs/weekly/weekly_digest_generation.md`](docs/weekly/weekly_digest_generation.md) — Weekly Digest contract and runtime CLI
- [`docs/demo/weekly_drift_review_app.md`](docs/demo/weekly_drift_review_app.md) — read-only Drift inspection of the frozen Weekly Drift Reviewer comparison Runs
- [`docs/demo/review_app.md`](docs/demo/review_app.md) — Runtime Demo Review App for the local VIF Critic-to-Weekly-Digest path
- [`docs/future_work/README.md`](docs/future_work/README.md) — exploratory directions, including OpenClaw integration research

## Value Identity Function (VIF) — 🧪 Experimental

The VIF is Twinkl's core evaluative engine. It compares what users *do* (daily Journal Entries) against what they *value* (their declared priorities) across the ten Schwartz value dimensions.

Key properties:
- **Vector-valued**: Tracks multiple life dimensions simultaneously, preserving trade-offs (e.g., "work goals crushed, but sleep suffered")
- **Uncertainty-aware**: Holds back judgment when the situation is complex or data is sparse
- **Trajectory-aware**: Detects patterns over time rather than reacting to single Journal Entries

These are implemented properties of the current stack, not target-only aspirations. The VIF Critic path includes ordinal MLP heads with MC Dropout, a BNN baseline, config-driven frozen encoders with `nomic-embed-text-v1.5` as the active default, corrected-split experiment logging, alternate-checkpoint retention, checkpoint discovery, runtime timeline reconstruction, weekly aggregation, and Weekly Coach-facing Drift routing. The 69-run / 133-config archive keeps `run_019`-`run_021` BalancedSoftmax as the historical corrected-split reference. The consensus-label branch `run_048`-`run_050` and recall-aware alternate-checkpoint reruns `run_051`-`run_056` remain diagnostic rather than reference replacements. See [`logs/experiments/index.md`](logs/experiments/index.md) for the live board.

**Current Drift contract:** Drift is two consecutive Conflicts for the same Core Value. The Weekly Drift Reviewer model contract is fixed at `gpt-5.6-luna` with reasoning effort `low`. Across three frozen development Runs, that setup found a median 23/42 known Drifts, produced 4 false Drift alerts, and had `0.637` coverage on the 42 Drifts across 36 Drift trajectories in the 292 resolved cases from the [`twinkl-qtwz` complete development review](logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md). These are AI-reviewed synthetic development results, not human validation, a fresh final test, or deployment approval. The approved staged architecture uses the Weekly Drift Reviewer without VIF Critic input followed by the deterministic Drift Detector. The VIF Critic remains essential to offline comparison, independent review, and retraining; its predictions do not enter the Weekly Drift Reviewer or user-facing Drift decision. The current runtime remains an experimental predecessor that emits `stable`, `crash`, `rut`, `evolution`, and `high_uncertainty`. See the [`twinkl-52zz` report](logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md), [`docs/architecture/drift_detection.md`](docs/architecture/drift_detection.md), and [`docs/evals/drift_detection_eval.md`](docs/evals/drift_detection_eval.md).

**Current runtime behavior:** A local VIF Critic checkpoint can drive the full offline path from per-Journal Entry predictions to validated weekly aggregates, structured prototype-router JSON, and a Weekly Digest. A shared schema in `src/vif/weekly_schema.py` defines the producer/consumer column contract and fails early when a required weekly column is missing. See `docs/vif/`, [`docs/weekly/weekly_digest_generation.md`](docs/weekly/weekly_digest_generation.md), and [`docs/demo/review_app.md`](docs/demo/review_app.md).

### Automated Experiment Logging & Review

The experiment log tracks VIF Critic training runs. The canonical frontier driver, `scripts/experiments/critic_training_v4_review.py`, writes metadata, configurations, model capacity, selection traces, alternate checkpoints, and evaluation metrics to `logs/experiments/`.

An AI **experiment-review skill** acts as an autonomous data science partner to process these runs. Rather than mechanically tuning hyperparameters, it synthesizes results to provide research-backed insights and hypotheses.

**To trigger it:** Point any capable LLM at `.claude/skills/experiment-review/SKILL.md` and ask it to read the skill and run it via the instructions.

**What it does:** 
- **Intelligent Backfilling**: Reads `git` logs and configuration diffs to reconstruct the rationale for past runs, automatically backfilling missing provenance and observations.
- **Data Science Partner**: Synthesizes interacting variables (e.g., encoder choice vs model capacity) to form hypotheses about the model's fundamental understanding of the task.
- **Research Colleague**: Actively browses the web for state-of-the-art literature to validate its recommendations for next-step experiments.
- **Reporting**: Produces a structured analysis of metric trade-offs (e.g., hedging vs minority recall), logs compact circumplex summaries, and maintains a leaderboard of the best models.

### LLM Context Baseline

The frozen-holdout baseline in `scripts/experiments/llm_critic_baseline.py` compares small OpenAI models under three explicit context setups: `student_visible`, `human_context`, and the upper-bound-only `full_judge_context`. On the 221-row test split, the `human_context` setup improves QWK and mean minority recall over `run_020`, while the VIF Critic retains higher `recall_-1` and lower hedging. The LLM is useful for LLM-Judge target repair or inference fallback diagnostics, not as a drop-in VIF Critic replacement. See [`docs/vif/03_model_training.md`](docs/vif/03_model_training.md) for the CLI and interpretation.

## Synthetic Data Generation — ✅ Complete

Bootstraps training data for value tagging and reward modeling. Generates realistic, longitudinal Journal Entries from synthetic personas with known Schwartz value profiles.

Key features:
- Personas with 1-2 assigned values expressed through concrete life details (not labels)
- Longitudinal Journal Entries that exhibit value tensions, Conflicts, and ambiguity
- Parallel asynchronous generation workflow
- Configurable tone, verbosity, and reflection mode per entry
- 🧪 Two-way conversational journaling with nudge system (LLM-based classification determines when/how to nudge, LLM generates contextual follow-up questions)

**🧪 Experimental**: Assessing whether nudging helps improve VIF data signal quality. See `docs/pipeline/annotation_guidelines.md` for the study methodology.

See `docs/pipeline/pipeline_specs.md` for implementation details.

### Current Dataset

| Metric | Value |
|--------|-------|
| Personas | 204 |
| Journal Entries | 1,651 |
| Avg entries/persona | 8.1 |
| Entries with generated nudges | 1,028 (62.3%) |

**Demographics:** 6 cultures, 9 professions, and 5 standard age brackets.

**Schwartz Value Distribution** (personas can have 1-2 values):
| Value | Personas | % |
|-------|----------|---|
| Power | 37 | 18% |
| Security | 36 | 18% |
| Hedonism | 33 | 16% |
| Universalism | 32 | 16% |
| Conformity | 28 | 14% |
| Tradition | 28 | 14% |
| Achievement | 25 | 12% |
| Benevolence | 25 | 12% |
| Self-Direction | 24 | 12% |
| Stimulation | 24 | 12% |

**LLM-Judge Label Distribution** (16,510 per-dimension labels across 1,651 Journal Entries):
| Label | Count | % |
|-------|-------|---|
| -1 | 1,165 | 7.1% |
| 0 | 12,535 | 75.9% |
| +1 | 2,810 | 17.0% |

Most generated nudges still use the standard three-category taxonomy (`tension_surfacing`, `elaboration`, `clarification`); a small number of older one-off labels remain in legacy raw artifacts.

See [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md) for parquet schemas and query examples.

## LLM-Judge Labeling Workflow — ✅ Complete

Scores synthetic Journal Entries against the 10 Schwartz value dimensions to create VIF Critic training labels. The workflow uses Claude Code subagents for parallel, consistent scoring.

**Workflow:**
```
Registry Check → Auto-Wrangle → Parallel Labeling (subagents) → Validation → Consolidation
```

**Usage:** Run `/judge` in Claude Code to execute the full workflow. The skill:
1. Checks the registry for pending work (`logs/registry/personas.parquet`)
2. Auto-wrangles raw synthetic data if needed (`logs/synthetic_data/` → `logs/wrangled/`)
3. Spawns parallel subagents — one per persona — each scoring all Journal Entries
4. Validates JSON output against Pydantic models
5. Consolidates labels to `logs/judge_labels/judge_labels.parquet`

**Scoring:** Each Journal Entry receives a 10-dimensional vector with values `{-1, 0, +1}` indicating Conflict, neutrality, or alignment with each Schwartz value. **Rationales** explain each non-zero score. Most Journal Entries have 1-3 non-zero scores.

**Data outputs:** See [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md) for parquet file schemas, example Polars queries, and analytics guidance.

**Consensus reference labels:** [`logs/judge_labels/consensus_labels.parquet`](logs/judge_labels/consensus_labels.parquet) stores the five-pass LLM-Judge resolver output, confidence tiers, agreement counts, and label-change flags. It remains diagnostic rather than the mainline VIF Critic training target. For Drift v1, a strict Conflict is `alignment_<value> == -1`; the resolver first chooses neutral versus non-neutral, then polarity among non-neutral votes. The agreement fields are confidence metadata, not full class distributions; actual `P(-1)`, `P(0)`, and `P(+1)` targets require the per-pass LLM-Judge vote files. This LLM-Judge reference is distinct from the six-detector comparison's detector-vote count. The orchestration guide lives in [`docs/pipeline/consensus_rejudging_instructions.md`](docs/pipeline/consensus_rejudging_instructions.md), and the stability-first report lives in [`logs/exports/twinkl_754/consensus_rejudging_report.md`](logs/exports/twinkl_754/consensus_rejudging_report.md). It is label provenance and diagnostic evidence only: it must not be used as a Drift target, threshold-selection input, or final test set.

**Key files:**
- `.claude/commands/judge.md` — Skill entry point
- `.claude/skills/judge/orchestration.md` — Detailed workflow
- `.claude/skills/judge/annotation_guide.md` — Scorability heuristics and calibration examples
- `.claude/skills/judge/rubric.md` — Schwartz value reference for scoring

**Primary Generation Method:**
- `docs/pipeline/claude_gen_instructions.md` — Instructions for Claude Code to generate synthetic data using parallel subagents

**Experimentation Scripts/Modules** (for prompt iteration and testing):
- `src/synthetic/generation.py` — One-way generation primitives (context, date sampling, banned-term guards)
- `src/synthetic/batch_preparation.py` — Baseline snapshots and frozen-holdout manifests for targeted data-lift experiments
- `src/synthetic/batch_verification.py` — Raw-batch acceptance checks and spot-check export generation
- `src/nudge/decision.py` + `src/nudge/generation.py` — Two-way conversational nudging logic
- `scripts/journalling/generation_sanity_check.py` — Quick local sanity checks
- `scripts/journalling/twinkl_681_5_freeze_baseline.py` / `scripts/journalling/twinkl_691_2_prepare_batch.py` — Example baseline-freeze wrappers
- `scripts/journalling/twinkl_681_5_verify_batch.py` / `scripts/journalling/twinkl_691_2_verify_batch.py` — Example targeted-batch verification wrappers
- `scripts/journalling/twinkl_754_prepare_consensus.py` / `twinkl_754_validate_results.py` / `twinkl_754_merge_pass_results.py` / `twinkl_754_summarize_consensus.py` — Consensus rerun bundle preparation, validation, merge, and stability-first reporting

## Human Annotation Tool — ✅ Complete

Validates LLM-Judge labels via blind human annotation across 10 Schwartz value dimensions. Annotators provide independent scores without seeing LLM-Judge labels first; the annotation tool then computes agreement metrics (Cohen's κ, Fleiss' κ).

**Run the tool:**
```sh
uv run shiny run src/annotation_tool/app.py
```

Open `http://127.0.0.1:8000` in your browser.

**Features:**
- Displays persona context (name, age, profession, culture, Core Values, collapsible bio)
- Shows Journal Entries with nudge/response threading
- 10-value scoring grid with -1 (misaligned) / 0 (neutral) / +1 (aligned) with CSS tooltips for Schwartz value definitions
- Progress tracking per annotator
- Annotations persisted to `logs/annotations/<annotator>.parquet`
- **Analysis & Metrics panel** — Computes inter-annotator agreement (Cohen's κ, Fleiss' κ)
- **Export functionality** — CSV, Parquet, and Markdown report formats
- **Comparison view** — Inline display of human vs LLM-Judge labels for review

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

See [`docs/demo/weekly_drift_review_app.md`](docs/demo/weekly_drift_review_app.md)
for the review contract, input boundary, launch options, and frozen files.

**Key files:**

- `src/drift_review_app/app.py` — Shiny interface
- `src/drift_review_app/data.py` — frozen-input loading and validation
- `src/drift_rules.py` — shared deterministic Drift rules
- `Dockerfile.review_app`, `requirements-review-app.txt`, and `railway.json` —
  deployment boundary

## Runtime Demo Review App — 🧪 Experimental

A sibling Shiny app for showcasing the end-to-end runtime flow on top of existing wrangled personas and local VIF Critic checkpoints. It lets you browse persona details, read the full Journal Entry timeline, choose a checkpoint, run the live VIF Critic → prototype router → Weekly Digest cycle, and inspect the resulting files in one place. This app also generates a live Weekly Coach reflection when a provider API key is available.

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
- Detector input source toggle between **LLM-Judge labels** and **VIF Critic predictions**
- Detector comparison across **Baseline**, **EMA**, **CUSUM**, **Cosine**, **Control Chart**, and **KL Div**, with per-Journal Entry detector-vote counts (not the five-pass LLM-Judge reference)
- A six-tab result canvas: Overview, per-Journal Entry VIF Critic outputs, weekly signals, Drift, Weekly Digest, and detector comparison
- Live Weekly Coach reflection rendered in the Weekly Digest tab, with `weekly_mirror`, `tension_explanation`, and `reflective_question` sections

**Weekly Coach reflection:** `src/coach/llm_client.py` builds the provider-backed
callable that `src/demo_tool/runtime_bridge.py` injects into
`run_weekly_coach_cycle`. `TWINKL_COACH_PROVIDER` selects `openai` (default) or
`gemini`; `TWINKL_COACH_MODEL` overrides the per-provider default model
(`gpt-5.4-mini` or `gemini-2.5-flash`). When the selected provider's API key is
absent, the provider is unrecognised, or the request fails, the app degrades to a
numeric-only Weekly Digest instead of erroring, so it stays runnable offline. The
`src.coach.runtime` and `src.coach.weekly_digest` CLIs do not call a live Weekly
Coach LLM; they render and persist the prompt only.

**Generated files:** The app writes persona/checkpoint-specific runtime bundles under `logs/exports/demo_tool_runs/<persona_id>/<checkpoint-stem>-<hash>/`.

See [`docs/demo/review_app.md`](docs/demo/review_app.md) for the full workflow
and file layout. This app is distinct from the read-only
[`Drift Inspection App`](docs/demo/weekly_drift_review_app.md), which compares
frozen Weekly Drift Reviewer Runs and does not execute the VIF Critic runtime.

**Key files:**
- `src/demo_tool/app.py` — Main Shiny demo application
- `src/demo_tool/data_loader.py` — Persona catalog and chronological timeline loading
- `src/demo_tool/runtime_bridge.py` — Checkpoint discovery and Weekly Coach runtime wrapper
- `src/coach/llm_client.py` — Provider-backed Weekly Coach reflection adapters (Gemini, OpenAI)
- `src/demo_tool/multi_drift.py` — Multi-detector comparison bundle for LLM-Judge-label and VIF Critic-prediction views
- `src/demo_tool/state.py` — Centralized reactive UI state

## Evaluation Workflow — ⚠️ Partial

Sequential validation workflow for the VIF with four stages:
1. **LLM-Judge Validation** — Training data quality (Cohen's κ > 0.60)
2. **Value Modeling** — VIF Critic learns value hierarchies correctly
3. **Drift Detection** — Drift Detector finds Drift without unacceptable false alerts
4. **Explanation Quality** — Explanations are grounded and useful

See [`docs/evals/overview.md`](docs/evals/overview.md) for the full evaluation workflow and current status.

## Embedding Explorer — ✅ Complete

An interactive 3D visualization that lets you explore the VIF Critic's internal embedding space. By projecting high-dimensional hidden-layer activations and SBERT text embeddings into 3D via PCA and t-SNE, the explorer reveals how the VIF Critic organizes Journal Entries — whether Journal Entries with similar value profiles cluster together, how prediction errors distribute across the space, and where the model is most uncertain.

This is useful for building intuition about what the VIF Critic has learned: do Conflict predictions occupy distinct regions? Are hard dimensions (stimulation, hedonism) scattered differently than easy ones? Does the hidden-layer structure differ meaningfully from the raw text embeddings?

**Generate and open:**
```sh
uv run python -m src.vif.extract_embeddings \
  --checkpoint logs/experiments/artifacts/.../BalancedSoftmax/selected_checkpoint.pt
```

**Features:**
- 4 projection spaces: Hidden Layer / SBERT Embedding × PCA / t-SNE
- 5 color modes: Data Split, Prediction, Ground Truth, Persona, Uncertainty
- Per-dimension filtering across all 10 Schwartz values
- Click-to-inspect: view journal text, predictions vs ground truth, and uncertainty per entry
- Persona trajectory lines (toggle-able) showing temporal progression through embedding space
- Adjustable bloom glow, auto-rotation, full orbit controls

**Output:** Self-contained HTML file (`viz/embedding_explorer.html`, ~3MB) with embedded Three.js and all 1,651 data points. No server required.

The CLI accepts `--output` for a different HTML path, `--perplexity` for the
t-SNE setting (default `30.0`), and `--no-browser` to suppress automatic browser
opening.

**Key files:**
- `src/vif/extract_embeddings.py` — Extraction script and HTML template
- `viz/embedding_explorer.html` — Generated visualization (gitignored)

## Known Gaps

| Capability | Status | Note |
|---|---|---|
| Onboarding (BWS Values Assessment) | 📋 Specified | Flow designed; not yet implemented |
| Weekly Coach validation depth | ⚠️ Partial | Weekly Digest generation, runtime files, and demo review flow exist; the approved Weekly Drift Reviewer and Drift Detector path is not wired, a fresh final test is missing, and Weekly Coach evaluation remains incomplete |
| Nudge signal quality validation | 🧪 Experimental | Annotation study and downstream usefulness checks remain in progress |
| Embedding Explorer | ✅ Complete | Interactive 3D visualization of VIF Critic embeddings |
| Drift Detector validity and production wiring | 🧪 Experimental | The Weekly Drift Reviewer model contract is fixed at `gpt-5.6-luna` with reasoning effort `low`, and its decisions without VIF Critic input feed the deterministic Drift Detector. Predefined operating criteria, a fresh final test, deployment approval, and production wiring remain pending. |
| Journaling anomaly radar | ❌ Not Started | Cadence/gap detection beyond the current prototype-router tooling |
| Goal-aligned inspiration feed | ❌ Not Started | External API integration |

For the full breakdown, see the [Implementation Status](docs/prd.md#implementation-status) table in prd.md.

## Common Commands

Examples below use `uv run` so they pick up the project environment directly. Activating `.venv` manually also works.

- Launch the annotation tool: `uv run shiny run src/annotation_tool/app.py`
- Launch the Drift Inspection App: `uv run shiny run --host 127.0.0.1 --port 8000 --no-dev-mode src/drift_review_app/app.py`
- Deploy the Drift Inspection App to Railway: `railway up`
- Launch the Runtime Demo Review App: `uv run shiny run src/demo_tool/app.py`
- Run the Runtime Demo Review App directly on port `8001`: `uv run python src/demo_tool/app.py`
- Run a local VIF Critic checkpoint through the full Weekly Coach path: `uv run python -m src.coach.runtime --persona-id 0a2fe15c --checkpoint-path logs/experiments/artifacts/.../selected_checkpoint.pt`
- Build a Weekly Digest from the default persisted LLM-Judge labels: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c`
- Build a Weekly Digest from saved VIF Critic predictions: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c --signals-path logs/exports/weekly_coach/0a2fe15c_vif_timeline.parquet`
- Train the mainline VIF Critic with CLI overrides and LR-finder export: `uv run python -m src.vif.train --grad-clip 1.0 --lr-find-output-path logs/exports/lr_find.png`
- Run the BNN baseline: `uv run python -m src.vif.train_bnn --epochs 10 --batch-size 16`
- Generate the embedding explorer without auto-opening a browser: `uv run python -m src.vif.extract_embeddings --checkpoint logs/experiments/artifacts/.../selected_checkpoint.pt --no-browser`
- Prepare a deterministic consensus pilot bundle: `uv run python scripts/journalling/twinkl_754_prepare_consensus.py --pilot-size 50 --pilot-hard-dimensions security,hedonism,stimulation`
- Reproduce the default consensus-label Drift EDA with runtime-compatible week bins: `uv run python scripts/drift/trajectory_eda.py`
- Compare persisted LLM-Judge labels with week bins anchored to the first Journal Entry: `uv run python scripts/drift/trajectory_eda.py --labels judge --week-mode persona_anchor`
- Estimate the LLM context baseline cost without making API calls: `uv run python scripts/experiments/llm_critic_baseline.py estimate --split test --context-arms student_visible human_context`
- Re-score the frozen Weekly Drift Reviewer model comparison: `uv run python -m scripts.experiments.compare_twinkl_52zz_models score`
- Re-score the frozen Luna reasoning-effort comparison: `uv run python -m scripts.experiments.compare_twinkl_52zz_luna_reasoning score`
- Replay recall-aware checkpoint selection from saved traces without retraining: `uv run python scripts/experiments/replay_recall_aware_checkpoint_selection.py`

The Drift EDA accepts `--labels {consensus,judge}` (default: `consensus`) and `--week-mode {runtime,persona_anchor}` (default: `runtime`). The LLM baseline exposes `estimate`, `run`, `score`, and `report`; `run` writes dry-run records unless `--execute` is supplied.

The `twinkl-52zz` model-comparison runner exposes `prepare`, `estimate`, `run`,
and `score`; `run` requires `--execute` and accepts
`--model-key {all,gpt_5_4_mini,gpt_5_6_luna}`. The Luna reasoning-effort runner
exposes `prepare`, `smoke`, `run`, and `score`; its paid `smoke` and `run`
commands require `--execute`. Both runners accept `--root` and `--config`.
Those global options must precede the subcommand.

# Setup

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
4. For commands that call OpenAI, create a `.env` file in the project root with
   your OpenAI API key:
   ```sh
   OPENAI_API_KEY=your-api-key-here
   ```

   The Drift Inspection App reads committed files and does not require an API
   key.

## Installing dependencies

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

- Install everything from the lockfile:
  ```sh
  uv sync
  ```
- Install the development group with pytest, pytest-asyncio, and Ruff:
  ```sh
  uv sync --group dev
  ```

## Running tests

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

The repository still contains historical notebook and test lint debt, so a
repo-wide `uv run ruff check .` is diagnostic rather than a clean gate.

Run the deterministic local end-to-end smoke pipeline only:

```sh
uv run pytest tests/e2e -q
```

This smoke test exercises the offline path `synthetic_data -> wrangled markdown -> consolidated LLM-Judge labels -> VIF Critic training` using tiny local fixtures and a mock text encoder, so it does not require live LLM calls.

## Adding a dependency

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

## Exporting requirements.txt (optional)

Only needed for legacy tooling or platforms that require it:

```sh
uv export -o requirements.txt
```
