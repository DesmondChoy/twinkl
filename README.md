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
- [`docs/pipeline/pipeline_specs.md`](docs/pipeline/pipeline_specs.md), [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md), [`docs/pipeline/consensus_rejudging_instructions.md`](docs/pipeline/consensus_rejudging_instructions.md) — data generation, label surfaces, and consensus diagnostics
- [`docs/drift/trajectory_eda.md`](docs/drift/trajectory_eda.md) — empirical drift-definition analysis, consensus-vs-persisted comparisons, and benchmark candidate tables
- [`docs/weekly/weekly_digest_generation.md`](docs/weekly/weekly_digest_generation.md) — weekly Coach digest contract and runtime CLI
- [`docs/demo/review_app.md`](docs/demo/review_app.md) — Shiny review UI for the local end-to-end runtime
- [`docs/future_work/README.md`](docs/future_work/README.md) — exploratory directions, including OpenClaw integration research

## Value Identity Function (VIF) — 🧪 Experimental

The VIF is Twinkl's core evaluative engine. It compares what users *do* (daily journal entries) against what they *value* (their declared priorities) across the ten Schwartz value dimensions.

Key properties:
- **Vector-valued**: Tracks multiple life dimensions simultaneously, preserving trade-offs (e.g., "work goals crushed, but sleep suffered")
- **Uncertainty-aware**: Holds back judgment when the situation is complex or data is sparse
- **Trajectory-aware**: Detects patterns over time rather than reacting to single entries

These are implemented properties of the current stack, not target-only aspirations. The Critic path includes ordinal MLP heads with MC Dropout, a BNN baseline, config-driven frozen encoders with `nomic-embed-text-v1.5` as the active default, corrected-split experiment logging, candidate-checkpoint retention, checkpoint discovery, runtime timeline reconstruction, weekly aggregation, and Coach-facing drift routing. The 56-run / 120-config archive keeps `run_019`-`run_021` BalancedSoftmax as the active corrected-split frontier. The consensus-label branch `run_048`-`run_050` and recall-aware candidate reruns `run_051`-`run_056` remain diagnostic rather than frontier replacements. See [`logs/experiments/index.md`](logs/experiments/index.md) for the live board.

**Current drift contract:** Drift v1 means a sustained conflict episode: two consecutive consensus `-1` reference labels on a declared core or high-weight value. The runtime target is rolling soft `P(-1)` evidence under uncertainty gating, delivered through the weekly Coach digest. The current code path is an experimental predecessor that still emits `stable`, `crash`, `rut`, `evolution`, and `high_uncertainty`; it does not yet implement the selected soft-evidence detector. The review UI also compares six exploratory rule-based detectors against either judge labels or Critic predictions. See [`docs/drift/trajectory_eda.md`](docs/drift/trajectory_eda.md), [`docs/evals/drift_detection_eval.md`](docs/evals/drift_detection_eval.md), and [`docs/demo/review_app.md`](docs/demo/review_app.md).

**Current runtime behavior:** A local checkpoint can drive the full offline path from per-entry VIF signals to validated weekly aggregates, structured prototype drift output, and a weekly digest artifact. A shared schema in `src/vif/weekly_schema.py` defines the producer/consumer column contract and fails early when a required weekly column is missing. See `docs/vif/`, [`docs/weekly/weekly_digest_generation.md`](docs/weekly/weekly_digest_generation.md), and [`docs/demo/review_app.md`](docs/demo/review_app.md).

### Automated Experiment Logging & Review

An automated logging system tracks VIF training experiments. The canonical frontier driver, `scripts/experiments/critic_training_v4_review.py`, writes metadata, configurations, model capacity, selection traces, candidate checkpoints, and evaluation metrics to `logs/experiments/`.

An AI **experiment-review skill** acts as an autonomous data science partner to process these runs. Rather than mechanically tuning hyperparameters, it synthesizes results to provide research-backed insights and hypotheses.

**To trigger it:** Point any capable LLM at `.claude/skills/experiment-review/SKILL.md` and ask it to read the skill and run it via the instructions.

**What it does:** 
- **Intelligent Backfilling**: Reads `git` logs and configuration diffs to reconstruct the rationale for past runs, automatically backfilling missing provenance and observations.
- **Data Science Partner**: Synthesizes interacting variables (e.g., encoder choice vs model capacity) to form hypotheses about the model's fundamental understanding of the task.
- **Research Colleague**: Actively browses the web for state-of-the-art literature to validate its recommendations for next-step experiments.
- **Reporting**: Produces a structured analysis of metric trade-offs (e.g., hedging vs minority recall), logs compact circumplex summaries, and maintains a leaderboard of the best models.

### LLM Critic Context Baseline

The frozen-holdout baseline in `scripts/experiments/llm_critic_baseline.py` compares small OpenAI models under three explicit context contracts: `student_visible`, `human_context`, and the upper-bound-only `full_judge_context`. On the 221-row test split, the strongest `human_context` arm improves QWK and mean minority recall over `run_020`, while the MLP retains higher `recall_-1` and lower hedging. The LLM therefore serves as a teacher, oracle, or fallback diagnostic rather than a drop-in replacement for the local Critic. See [`docs/vif/03_model_training.md`](docs/vif/03_model_training.md) for the command surface and interpretation.

## Synthetic Data Generation — ✅ Complete

Bootstraps training data for value tagging and reward modeling. Generates realistic, longitudinal journal entries from synthetic personas with known Schwartz value profiles.

Key features:
- Personas with 1-2 assigned values expressed through concrete life details (not labels)
- Longitudinal entries that exhibit value drift, conflicts, and ambiguity
- Parallel async pipeline for efficient generation
- Configurable tone, verbosity, and reflection mode per entry
- 🧪 Two-way conversational journaling with nudge system (LLM-based classification determines when/how to nudge, LLM generates contextual follow-up questions)

**🧪 Experimental**: Assessing whether nudging helps improve VIF data signal quality. See `docs/pipeline/annotation_guidelines.md` for the study methodology.

See `docs/pipeline/pipeline_specs.md` for implementation details.

### Current Dataset

| Metric | Value |
|--------|-------|
| Personas | 204 |
| Journal entries | 1,651 |
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

**Judge Label Distribution** (16,510 per-dimension labels across 1,651 entries):
| Label | Count | % |
|-------|-------|---|
| -1 | 1,165 | 7.1% |
| 0 | 12,535 | 75.9% |
| +1 | 2,810 | 17.0% |

Most generated nudges still use the standard three-category taxonomy (`tension_surfacing`, `elaboration`, `clarification`); a small number of older one-off labels remain in legacy raw artifacts.

See [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md) for parquet schemas and query examples.

## Judge Labeling Pipeline — ✅ Complete

Scores synthetic journal entries against the 10 Schwartz value dimensions to create training labels for the VIF. The pipeline uses Claude Code subagents for parallel, consistent scoring.

**Workflow:**
```
Registry Check → Auto-Wrangle → Parallel Labeling (subagents) → Validation → Consolidation
```

**Usage:** Run `/judge` in Claude Code to execute the full pipeline. The skill:
1. Checks the registry for pending work (`logs/registry/personas.parquet`)
2. Auto-wrangles raw synthetic data if needed (`logs/synthetic_data/` → `logs/wrangled/`)
3. Spawns parallel subagents — one per persona — each scoring all entries
4. Validates JSON output against Pydantic models
5. Consolidates labels to `logs/judge_labels/judge_labels.parquet`

**Scoring:** Each entry receives a 10-dimensional vector with values `{-1, 0, +1}` indicating misalignment, neutrality, or alignment with each Schwartz value. **Rationales** explain each non-zero score. Most entries have 1-3 non-zero scores.

**Data outputs:** See [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md) for parquet file schemas, example Polars queries, and analytics guidance.

**Consensus reference surface:** [`logs/judge_labels/consensus_labels.parquet`](logs/judge_labels/consensus_labels.parquet) stores the 5-pass consensus rerun with per-dimension confidence tiers, agreement counts, and label-change flags. It remains diagnostic rather than the mainline Critic training target, and it supplies the strict label-side reference for drift v1. The `consensus_agreement_*` fields are confidence metadata, not full class distributions; actual `P(-1)`, `P(0)`, and `P(+1)` targets require the per-pass vote files. The orchestration guide lives in [`docs/pipeline/consensus_rejudging_instructions.md`](docs/pipeline/consensus_rejudging_instructions.md), and the stability-first report lives in [`logs/exports/twinkl_754/consensus_rejudging_report.md`](logs/exports/twinkl_754/consensus_rejudging_report.md).

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

Validates LLM Judge labels via blind human annotation across 10 Schwartz value dimensions. Annotators provide independent scores without seeing Judge labels first; the system then computes agreement metrics (Cohen's κ, Fleiss' κ).

**Run the tool:**
```sh
uv run shiny run src/annotation_tool/app.py
```

Open `http://127.0.0.1:8000` in your browser.

**Features:**
- Displays persona context (name, age, profession, culture, core values, collapsible bio)
- Shows journal entries with nudge/response threading
- 10-value scoring grid with -1 (misaligned) / 0 (neutral) / +1 (aligned) with CSS tooltips for Schwartz value definitions
- Progress tracking per annotator
- Annotations persisted to `logs/annotations/<annotator>.parquet`
- **Analysis & Metrics panel** — Computes inter-annotator agreement (Cohen's κ, Fleiss' κ)
- **Export functionality** — CSV, Parquet, and Markdown report formats
- **Comparison view** — Inline display of human vs judge labels for review

**Key files:**
- `src/annotation_tool/app.py` — Main Shiny application
- `src/annotation_tool/data_loader.py` — Loads entries from wrangled files
- `src/annotation_tool/annotation_store.py` — Persists annotations with file locking
- `src/annotation_tool/agreement_metrics.py` — Kappa calculations and export
- `src/annotation_tool/components/` — Modular UI components (scoring grid, comparison view, analysis)
- `src/annotation_tool/state.py` — Centralized state management
- `docs/pipeline/annotation_tool_plan.md` — Full implementation plan

## Demo Review App — 🧪 Experimental

A sibling Shiny app for showcasing the end-to-end runtime flow on top of existing wrangled personas and local Critic checkpoints. It lets you browse persona details, read the full journal timeline, choose a checkpoint, run the live Critic -> drift -> weekly digest cycle, and inspect the resulting artifacts in one place.

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
- Cached artifact loading for previously run persona/checkpoint pairs
- End-to-end runtime execution via `src.coach.runtime.run_weekly_coach_cycle`
- Detector input source toggle between **Judge labels** and **Critic predictions**
- Detector comparison across **Baseline**, **EMA**, **CUSUM**, **Cosine**, **Control Chart**, and **KL Div**, with per-entry detector-vote counts (not five-pass Judge consensus)
- A six-tab result canvas with Overview, per-entry Critic outputs, weekly signals, drift payloads, weekly digest markdown, and detector comparison

**Generated artifacts:** The app writes persona/checkpoint-specific runtime bundles under `logs/exports/demo_tool_runs/<persona_id>/<checkpoint-stem>-<hash>/`.

See [`docs/demo/review_app.md`](docs/demo/review_app.md) for the full workflow and artifact layout.

**Key files:**
- `src/demo_tool/app.py` — Main Shiny demo application
- `src/demo_tool/data_loader.py` — Persona catalog and chronological timeline loading
- `src/demo_tool/runtime_bridge.py` — Checkpoint discovery and weekly Coach runtime wrapper
- `src/demo_tool/multi_drift.py` — Multi-detector comparison bundle for judge-label and Critic-signal views
- `src/demo_tool/state.py` — Centralized reactive UI state

## Evaluation Pipeline — ⚠️ Partial

Sequential validation pipeline for the VIF with four stages:
1. **Judge Validation** — Training data quality (Cohen's κ > 0.60)
2. **Value Modeling** — Critic learns value hierarchies correctly
3. **Drift Detection** — Triggers fire accurately on misalignment
4. **Explanation Quality** — Explanations are grounded and useful

See [`docs/evals/overview.md`](docs/evals/overview.md) for the full pipeline overview and current status.

## Embedding Explorer — ✅ Complete

An interactive 3D visualization that lets you explore the VIF critic's internal embedding space. By projecting high-dimensional hidden-layer activations and SBERT text embeddings into 3D via PCA and t-SNE, the explorer reveals how the critic organizes journal entries — whether entries with similar value profiles cluster together, how prediction errors distribute across the space, and where the model is most uncertain.

This is useful for building intuition about what the critic has learned: do misaligned entries occupy distinct regions? Are hard dimensions (stimulation, hedonism) scattered differently than easy ones? Does the hidden-layer structure differ meaningfully from the raw text embeddings?

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
| Weekly Coach validation depth | ⚠️ Partial | Digest generation, runtime artifacts, and demo review flow exist; trigger calibration and narrative evaluation still need broader validation |
| Nudge signal quality validation | 🧪 Experimental | Annotation study and downstream usefulness checks remain in progress |
| Embedding Explorer | ✅ Complete | Interactive 3D visualization of critic embedding space |
| Drift threshold calibration | 🧪 Experimental | Consensus-label EDA provides sustained-conflict and conflict-heavy-week candidates; the soft-evidence detector, threshold calibration, and unbiased held-out benchmark remain incomplete |
| Journaling anomaly radar | ❌ Not Started | Cadence/gap detection beyond the current drift tooling |
| Goal-aligned inspiration feed | ❌ Not Started | External API integration |

For the full breakdown, see the [Implementation Status](docs/prd.md#implementation-status) table in prd.md.

## Common Commands

Examples below use `uv run` so they pick up the project environment directly. Activating `.venv` manually also works.

- Launch the annotation tool: `uv run shiny run src/annotation_tool/app.py`
- Launch the demo review UI: `uv run shiny run src/demo_tool/app.py`
- Run the demo review UI directly on port `8001`: `uv run python src/demo_tool/app.py`
- Run a local checkpoint through the full weekly Coach path: `uv run python -m src.coach.runtime --persona-id 0a2fe15c --checkpoint-path logs/experiments/artifacts/.../selected_checkpoint.pt`
- Build a digest from the default persisted Judge labels: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c`
- Build a digest from saved Critic signals: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c --signals-path logs/exports/weekly_coach/0a2fe15c_vif_timeline.parquet`
- Train the mainline Critic with CLI overrides and LR-finder export: `uv run python -m src.vif.train --grad-clip 1.0 --lr-find-output-path logs/exports/lr_find.png`
- Run the BNN baseline: `uv run python -m src.vif.train_bnn --epochs 10 --batch-size 16`
- Generate the embedding explorer without auto-opening a browser: `uv run python -m src.vif.extract_embeddings --checkpoint logs/experiments/artifacts/.../selected_checkpoint.pt --no-browser`
- Prepare a deterministic consensus pilot bundle: `uv run python scripts/journalling/twinkl_754_prepare_consensus.py --pilot-size 50 --pilot-hard-dimensions security,hedonism,stimulation`
- Reproduce the default consensus-label drift EDA with runtime-compatible week bins: `uv run python scripts/drift/trajectory_eda.py`
- Compare persisted Judge labels with first-entry-anchored week bins: `uv run python scripts/drift/trajectory_eda.py --labels judge --week-mode persona_anchor`
- Estimate the LLM Critic baseline cost without making API calls: `uv run python scripts/experiments/llm_critic_baseline.py estimate --split test --context-arms student_visible human_context`
- Replay recall-aware checkpoint selection from saved traces without retraining: `uv run python scripts/experiments/replay_recall_aware_checkpoint_selection.py`

The drift EDA accepts `--labels {consensus,judge}` (default: `consensus`) and `--week-mode {runtime,persona_anchor}` (default: `runtime`). The LLM baseline exposes `estimate`, `run`, `score`, and `report`; `run` writes dry-run records unless `--execute` is supplied.

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
4. Create a `.env` file in the project root with your OpenAI API key:
   ```sh
   OPENAI_API_KEY=your-api-key-here
   ```

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

This smoke test exercises the offline path `synthetic_data -> wrangled markdown -> consolidated judge labels -> VIF training` using tiny local fixtures and a mock text encoder, so it does not require live LLM calls.

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
