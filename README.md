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

These are target design properties. The Critic model infrastructure is functional (MLP ordinal base + MC Dropout, nomic-embed-text-v1.5 encoder, CLI training scripts) but **QWK metric optimization is in progress** — see the [Implementation Status](docs/prd.md#implementation-status) for details.

**Target behavior:** When the VIF detects significant misalignment with high confidence, it triggers the Coach layer to surface evidence-based feedback. Currently the Coach is ⚠️ Partial — entry processing is ready but digest generation is not yet implemented. See `docs/vif/` for architecture details.

### Automated Experiment Logging & Review

An automated logging system tracks VIF training experiments. Each time the critic v2 notebooks are run, metadata, configurations, model capacity, and evaluation metrics are written to `logs/experiments/runs/`.

An AI **experiment-review skill** acts as an autonomous data science partner to process these runs. Rather than mechanically tuning hyperparameters, it synthesizes results to provide research-backed insights and hypotheses.

**To trigger it:** Point any capable LLM at `/.claude/skills/experiment-review/SKILL.md` and ask it to read the skill and run it via the instructions.

**What it does:** 
- **Intelligent Backfilling**: Reads `git` logs and configuration diffs to reconstruct the rationale for past runs, automatically backfilling missing provenance and observations.
- **Data Science Partner**: Synthesizes interacting variables (e.g., encoder choice vs model capacity) to form hypotheses about the model's fundamental understanding of the task.
- **Research Colleague**: Actively browses the web for state-of-the-art literature to validate its recommendations for next-step experiments.
- **Reporting**: Produces a structured analysis of metric trade-offs (e.g., hedging vs minority recall) and maintains a leaderboard of the best models.

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
| Personas | 192 |
| Journal entries | 1,555 |
| Avg entries/persona | 8.1 |
| Entries with generated nudges | 957 (61.5%) |

**Demographics:** 6 cultures, 9 professions, and 5 standard age brackets.

**Schwartz Value Distribution** (personas can have 1-2 values):
| Value | Personas | % |
|-------|----------|---|
| Power | 37 | 19% |
| Universalism | 32 | 17% |
| Security | 30 | 16% |
| Conformity | 28 | 15% |
| Tradition | 28 | 15% |
| Hedonism | 27 | 14% |
| Achievement | 25 | 13% |
| Benevolence | 25 | 13% |
| Self-Direction | 24 | 13% |
| Stimulation | 24 | 13% |

**Judge Label Distribution** (15,550 per-dimension labels across 1,555 entries):
| Label | Count | % |
|-------|-------|---|
| -1 | 1,122 | 7.2% |
| 0 | 11,720 | 75.4% |
| +1 | 2,708 | 17.4% |

**Standard nudge types** (953 taxonomy-tagged nudges): Tension-surfacing (42.2%), Elaboration (40.4%), Clarification (17.4%). Four older one-off nudge labels remain in legacy artifacts.

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
- `src/nudge/decision.py` + `src/nudge/generation.py` — Two-way conversational nudging logic
- `scripts/journalling/generation_sanity_check.py` — Quick local sanity checks

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

## Evaluation Pipeline — ⚠️ Partial

Sequential validation pipeline for the VIF with four stages:
1. **Judge Validation** — Training data quality (Cohen's κ > 0.60)
2. **Value Modeling** — Critic learns value hierarchies correctly
3. **Drift Detection** — Triggers fire accurately on misalignment
4. **Explanation Quality** — Explanations are grounded and useful

See [`docs/evals/overview.md`](docs/evals/overview.md) for the full pipeline overview and current status.

## Known Gaps

| Capability | Status | Note |
|---|---|---|
| Onboarding (BWS Values Assessment) | 📋 Specified | Flow designed; not yet implemented |
| Coach digest generation | ⚠️ Partial | Entry processing ready; weekly digest not built |
| Nudge signal quality validation | 🧪 Experimental | Annotation study in progress |
| "Map of Me" visualization | ❌ Not Started | Embedding trajectories |
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
