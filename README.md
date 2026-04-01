# Twinkl

Twinkl is an "inner compass" that helps users align their daily behavior with their long-term values. Unlike traditional journaling apps that summarize moods and topics, Twinkl maintains a dynamic self-model of the user's declared priorities and surfaces tensions when behavior drifts from intent. It answers the question: *"Am I living in line with what I said I value?"*

> **Project status:** Twinkl is under active development as an academic capstone. Status markers in each section indicate capability maturity:
> ✅ Complete · 🧪 Experimental · ⚠️ Partial · 📋 Specified · ❌ Not Started
>
> See [Known Gaps](#known-gaps) for the summary and [Implementation Status](docs/prd.md#implementation-status) for the full breakdown.

## Value Identity Function (VIF) — 🧪 Experimental

The VIF is Twinkl's core evaluative engine. It compares what users *do* (daily journal entries) against what they *value* (their declared priorities) across multiple dimensions like Health, Relationships, and Growth.

Key properties:
- **Vector-valued**: Tracks multiple life dimensions simultaneously, preserving trade-offs (e.g., "work goals crushed, but sleep suffered")
- **Uncertainty-aware**: Holds back judgment when the situation is complex or data is sparse
- **Trajectory-aware**: Detects patterns over time rather than reacting to single entries

These are target design properties. The Critic model infrastructure is functional (ordinal MLP heads + MC Dropout, config-driven sentence encoders with `nomic-embed-text-v1.5` as the active default, CLI training scripts), but **QWK and structural value-consistency optimization are still in progress**. The active corrected-split default remains the `run_019`-`run_021` BalancedSoftmax family; recent post-lift reruns are logged in [`logs/experiments/index.md`](logs/experiments/index.md).

**Target behavior:** When the VIF detects significant misalignment with high confidence, it triggers the Coach layer to surface evidence-based feedback. Currently the Coach is ⚠️ Partial — entry processing is ready but digest generation is not yet implemented. See `docs/vif/` for architecture details.

### Automated Experiment Logging & Review

An automated logging system tracks VIF training experiments. Each time the critic v2 notebooks are run, metadata, configurations, model capacity, and evaluation metrics are written to `logs/experiments/runs/`.

An AI **experiment-review skill** acts as an autonomous data science partner to process these runs. Rather than mechanically tuning hyperparameters, it synthesizes results to provide research-backed insights and hypotheses.

**To trigger it:** Point any capable LLM at `/.claude/skills/experiment-review/SKILL.md` and ask it to read the skill and run it via the instructions.

**What it does:** 
- **Intelligent Backfilling**: Reads `git` logs and configuration diffs to reconstruct the rationale for past runs, automatically backfilling missing provenance and observations.
- **Data Science Partner**: Synthesizes interacting variables (e.g., encoder choice vs model capacity) to form hypotheses about the model's fundamental understanding of the task.
- **Research Colleague**: Actively browses the web for state-of-the-art literature to validate its recommendations for next-step experiments.
- **Reporting**: Produces a structured analysis of metric trade-offs (e.g., hedging vs minority recall), logs compact circumplex summaries, and maintains a leaderboard of the best models.

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

## Human Annotation Tool — ✅ Complete

Validates LLM Judge labels via blind human annotation across 10 Schwartz value dimensions. Annotators provide independent scores without seeing Judge labels first; the system then computes agreement metrics (Cohen's κ, Fleiss' κ).

**Run the tool:**
```sh
shiny run src/annotation_tool/app.py
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
shiny run src/demo_tool/app.py
```

Open `http://127.0.0.1:8000` in your browser when running via `shiny run`, or `http://127.0.0.1:8001` when launching the file directly with Python.

**Key files:**
- `src/demo_tool/app.py` — Main Shiny demo application
- `src/demo_tool/data_loader.py` — Persona catalog and chronological timeline loading
- `src/demo_tool/runtime_bridge.py` — Checkpoint discovery and weekly Coach runtime wrapper
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
python -m src.vif.extract_embeddings \
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

**Key files:**
- `src/vif/extract_embeddings.py` — Extraction script and HTML template
- `viz/embedding_explorer.html` — Generated visualization (gitignored)

## Known Gaps

| Capability | Status | Note |
|---|---|---|
| Onboarding (BWS Values Assessment) | 📋 Specified | Flow designed; not yet implemented |
| Coach digest generation | ⚠️ Partial | Entry processing ready; weekly digest not built |
| Nudge signal quality validation | 🧪 Experimental | Annotation study in progress |
| Embedding Explorer | ✅ Complete | Interactive 3D visualization of critic embedding space |
| Journaling anomaly radar | ❌ Not Started | Cadence/gap detection |
| Goal-aligned inspiration feed | ❌ Not Started | External API integration |

For the full breakdown, see the [Implementation Status](docs/prd.md#implementation-status) table in prd.md.

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
3. Activate it (Fish shell):
   ```sh
   source .venv/bin/activate.fish
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
- If/when a dev group is added later:
  ```sh
  uv sync --dev
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
