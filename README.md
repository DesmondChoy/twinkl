# Twinkl

Twinkl is an "inner compass" that helps users align their daily behavior with their long-term values. Unlike traditional journaling apps that summarize moods and topics, Twinkl maintains a dynamic self-model of the user's declared priorities and surfaces tensions when behavior drifts from intent. It answers the question: *"Am I living in line with what I said I value?"*

## Value Identity Function (VIF)

The VIF is Twinkl's core evaluative engine. It compares what users *do* (daily journal entries) against what they *value* (their declared priorities) across multiple dimensions like Health, Relationships, and Growth.

Key properties:
- **Vector-valued**: Tracks multiple life dimensions simultaneously, preserving trade-offs (e.g., "work goals crushed, but sleep suffered")
- **Uncertainty-aware**: Holds back judgment when the situation is complex or data is sparse
- **Trajectory-aware**: Detects patterns over time rather than reacting to single entries

When the VIF detects significant misalignment with high confidence, it triggers the Coach layer to surface evidence-based feedback. See `docs/vif/` for architecture details.

## Synthetic Data Generation

Bootstraps training data for value tagging and reward modeling. Generates realistic, longitudinal journal entries from synthetic personas with known Schwartz value profiles.

Key features:
- Personas with 1-2 assigned values expressed through concrete life details (not labels)
- Longitudinal entries that exhibit value drift, conflicts, and ambiguity
- Parallel async pipeline for efficient generation
- Configurable tone, verbosity, and reflection mode per entry
- Two-way conversational journaling with nudge system (LLM-based classification determines when/how to nudge, LLM generates contextual follow-up questions)

**Validation in progress**: Assessing whether nudging helps improve VIF data signal quality. See `docs/pipeline/annotation_guidelines.md` for the study methodology.

See `docs/pipeline/pipeline_specs.md` for implementation details.

### Current Dataset

| Metric | Value |
|--------|-------|
| Personas | 100 |
| Journal entries | 729 |
| Avg entries/persona | 7.1 |
| Entries with nudges | 65.8% |

**Demographics:** 6 cultures, 9 professions, 5 age brackets.

**Schwartz Value Distribution** (personas can have 1-2 values):
| Value | Personas | % |
|-------|----------|---|
| Hedonism | 19 | 19% |
| Conformity | 18 | 18% |
| Stimulation | 17 | 17% |
| Security | 16 | 16% |
| Tradition | 15 | 15% |
| Universalism | 15 | 15% |
| Benevolence | 14 | 14% |
| Self-Direction | 14 | 14% |
| Power | 14 | 14% |
| Achievement | 13 | 13% |

**Nudge types:** Elaboration (44%), Tension Surfacing (41%), Clarification (14%)

See [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md) for parquet schemas and query examples.

## Judge Labeling Pipeline

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

**Experimentation Notebooks** (for prompt iteration and testing):
- `notebooks/journal_gen.ipynb` — One-way journal generation
- `notebooks/journal_nudge.ipynb` — Two-way conversational journaling with nudges

## Human Annotation Tool

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

## Evaluation Pipeline

Sequential validation pipeline for the VIF with four stages:
1. **Judge Validation** — Training data quality (Cohen's κ > 0.60)
2. **Value Modeling** — Critic learns value hierarchies correctly
3. **Drift Detection** — Triggers fire accurately on misalignment
4. **Explanation Quality** — Explanations are grounded and useful

See [`docs/evals/overview.md`](docs/evals/overview.md) for the full pipeline overview and current status.

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
