# Twinkl

Twinkl is a voice-first "inner compass" that helps users align their daily behavior with their long-term values. Unlike traditional journaling apps that summarize moods and topics, Twinkl maintains a dynamic self-model of the user's declared priorities and surfaces tensions when behavior drifts from intent. It answers the question: *"Am I living in line with what I said I value?"*

## Value Identity Function (VIF)

The VIF is Twinkl's core evaluative engine. It compares what users *do* (daily journal entries) against what they *value* (their declared priorities) across multiple dimensions like Health, Relationships, and Growth.

Key properties:
- **Vector-valued**: Tracks multiple life dimensions simultaneously, preserving trade-offs (e.g., "work goals crushed, but sleep suffered")
- **Uncertainty-aware**: Holds back judgment when the situation is complex or data is sparse
- **Trajectory-aware**: Detects patterns over time rather than reacting to single entries

When the VIF detects significant misalignment with high confidence, it triggers the Coach layer to surface evidence-based feedback. See `docs/VIF/` for architecture details.

## Synthetic Data Generation

Bootstraps training data for value tagging and reward modeling. Generates realistic, longitudinal journal entries from synthetic personas with known Schwartz value profiles.

Key features:
- Personas with 1-2 assigned values expressed through concrete life details (not labels)
- Longitudinal entries that exhibit value drift, conflicts, and ambiguity
- Parallel async pipeline for efficient generation
- Configurable tone, verbosity, and reflection mode per entry
- Two-way conversational journaling with nudge system (LLM-based classification determines when/how to nudge, LLM generates contextual follow-up questions)

**Validation in progress**: Assessing whether nudging helps improve VIF data signal quality. See `docs/synthetic_data/annotation_guidelines.md` for the study methodology.

See `docs/synthetic_data/pipeline_specs.md` for implementation details.

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

**Scoring:** Each entry receives a 10-dimensional vector with values `{-1, 0, +1}` indicating misalignment, neutrality, or alignment with each Schwartz value. Most entries have 1-3 non-zero scores.

**Data outputs:** See [`docs/data_schema.md`](docs/data_schema.md) for parquet file schemas, example Polars queries, and analytics guidance.

**Key files:**
- `.claude/commands/judge.md` — Skill entry point
- `.claude/skills/judge/orchestration.md` — Detailed workflow
- `.claude/skills/judge/annotation_guide.md` — Scorability heuristics and calibration examples
- `.claude/skills/judge/rubric.md` — Schwartz value reference for scoring

**Primary Generation Method:**
- `docs/synthetic_data/claude_gen_instructions.md` — Instructions for Claude Code to generate synthetic data using parallel subagents

**Experimentation Notebooks** (for prompt iteration and testing):
- `notebooks/journal_gen.ipynb` — One-way journal generation
- `notebooks/journal_nudge.ipynb` — Two-way conversational journaling with nudges

# Setup

This repo uses `uv` and `pyproject.toml` for dependency management.

1. Install `uv` (see https://docs.astral.sh/uv/).
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
