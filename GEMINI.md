# GEMINI.md

## Project Overview

Twinkl is an "inner compass" that helps users align daily behavior with
long-term values. Unlike traditional journaling apps that summarize
moods, Twinkl maintains a dynamic self-model of the user's declared
priorities and surfaces tensions when behavior drifts from intent.

The core component is the **Value Identity Function (VIF)**: an
evaluative engine that compares what users *do* (journal entries)
against what they *value* (declared priorities) across Schwartz value
dimensions. The intended behavior is vector-valued, uncertainty-aware,
and trajectory-aware.

This is an academic capstone project for the NUS Master of Technology
in Intelligent Systems program, so favor clear, practical
implementations over heavy architecture.

## Operational Defaults

1. Read `docs/prd.md` first for product intent. It is the source of truth.
2. Treat other `docs/` files as supporting context unless they contradict
   `docs/prd.md`.
3. Keep solutions scoped to a time-boxed POC. Avoid over-engineering.
4. Prefer small, testable increments over broad rewrites.
5. Preserve existing project conventions unless there is a concrete reason
   to change them.

## Environment and Commands

Do NOT use git worktrees. Work only in the main working directory.

Activate the virtual environment before Python commands:

```sh
source .venv/bin/activate.fish   # Preferred in this repo
source .venv/bin/activate        # Bash/Zsh fallback
```

Use `uv` for package/dependency actions:

```sh
uv sync
uv add <package>
uv pip install <package>
```

Run tests and linting with `uv` before commits:

```sh
uv run pytest
uv run ruff check .
```

Script-based generation/judging helpers live in `src/synthetic/`,
`src/judge/`, and `scripts/journalling/`.

## Architecture Snapshot

### Source Code (`src/`)

- `src/vif/` — VIF critic models (MLP ordinal, BNN), text/state encoders, dataset loading, training loops, evaluation metrics, and experiment logging
- `src/nudge/` — Nudge decision logic, generation and schemas
- `src/evals/` — Standalone evaluation scripts (e.g., nudge ranking)
- `src/synthetic/` — Synthetic generation primitives
- `src/registry/` — Persona registry with pipeline stages (`stage_synthetic`, `stage_wrangled`, `stage_labeled`)
- `src/judge/` — Judge labeling consolidation
- `src/wrangling/` — Parsers for synthetic and wrangled persona data
- `src/models/` — Pydantic models (judge label schema)
- `src/annotation_tool/` — Shiny for Python app for human annotation with inter-rater agreement metrics

### Configuration and Prompts

- `config/` — `synthetic_data.yaml`, `schwartz_values.yaml`, `vif.yaml`
- `prompts/` — Prompt templates (`*.yaml`) exposed via `prompts/__init__.py`

### Data and Artifacts (`logs/`)

- `logs/registry/` — `personas.parquet` (central persona registry)
- `logs/synthetic_data/` — Raw LLM-generated persona markdown files
- `logs/wrangled/` — Parsed/cleaned persona markdown files
- `logs/judge_labels/` — Per-persona JSON labels + consolidated `judge_labels.parquet`
- `logs/annotations/` — Human annotator parquet files (per-annotator)
- `logs/experiments/` — VIF training run logs (`runs/*.yaml`) and `index.md`
- `logs/exports/` — Agreement reports and other exports

### Synthetic Data Pipeline

The project utilizes a highly flexible, parallelized LLM-based synthetic data generation pipeline (documented in `docs/pipeline/claude_gen_instructions.md` and `docs/pipeline/pipeline_specs.md`). 

A critical feature of this pipeline is its native capability to resolve **class imbalances** in the training dataset:
- **Targeted Value Generation**: Use `TARGET_VALUES` to force the generation of personas with specific, underrepresented Schwartz values.
- **Tension-Targeted Generation**: Use `TARGET_TENSIONS` to oversample "Unsettled" journal entries tied to specific values by leveraging a scenario bank (via the `tension-selection` Claude skill). This specifically repairs label imbalances (e.g., establishing clear `-1` ground truth labels for underrepresented misalignments).
- **Emergent vs Prescriptive Logging**: Data is generated longitudinally over random gaps and variables, creating a robust, cold-start-ready dataset for the VIF.

### Scripted Workflows

- `src/vif/train.py` and `src/vif/train_bnn.py` — Critic training CLIs
- `src/synthetic/generation.py` — Synthetic generation primitives and safeguards
- `src/judge/labeling.py` — Judge rubric + scoring helpers
- `scripts/journalling/` — Lightweight sanity-check scripts for generation/judge flows

### Tests (`tests/`)

- `tests/vif/` — Eval metrics, loss functions, ordinal base tests
- `tests/wrangling/` — Wrangled data parser tests
- `tests/judge/` — Judge labeling and consolidation tests
- `tests/nudge/` — Nudge functionality and routing tests
- `tests/registry/` — Persona registry parsing tests
- `tests/synthetic/` — Synthetic generation test suites
- `tests/evals/` — Standalone evaluations tests

### Documentation (`docs/`)

- `docs/prd.md` — Product requirements (authoritative)
- `docs/vif/` — VIF concepts, architecture, training, uncertainty, state pipeline
- `docs/pipeline/` — Pipeline specs, annotation guidelines, judge instructions, data schema
- `docs/evals/` — Evaluation specs (drift detection, explanation quality, judge validation, value modeling)
- `docs/onboarding/` — Onboarding flow spec
- `docs/capstone_report/` — Report sections
- `docs/archive/` — Historical only
- `docs/future_work/` — Non-committed ideas

## Implementation Principles

- Async persona generation is parallel per persona and sequential within a
  persona for continuity.
- Journal content should stay emergent from persona context, not rigid value
  labels.
- Keep banned-term/value leakage protections intact when touching prompts or
  generation logic.
- Avoid data leakage in any logic intended to mirror inference behavior (e.g. VIF training or prompt generation). Do not include ground-truth features that will not be available in a production inference context.

## Code Style

- Imports: standard library first, then third-party, then local.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Keep comments concise and only where they reduce cognitive load.

## Quality Gate Before Commit

Before creating a commit:

1. Run `/quality` (repo alias for `.claude/skills/quality/SKILL.md`) before committing to review changes with "fresh eyes."
2. Review complete changed files, not only diffs.
3. Run targeted tests/linting (`uv run pytest` / `uv run ruff`) for touched modules.
4. Remove obvious dead code and debug remnants.
5. Confirm no behavior regressions in critical paths.

If there is ambiguity and no blocking risk, proceed with explicit
assumptions and note them. If ambiguity affects correctness or design
direction, ask one concise clarifying question.

## Issue Tracking with Beads (`bd`)

Use `bd` (beads) for all issue tracking. This is mandatory, not optional.

### Before starting work
- Run `bd list` to see open issues and find relevant ones.
- If the work maps to an existing issue, note its ID (e.g., `twinkl-abc`).
- If no issue exists, create one before starting:
  ```sh
  bd create "Short descriptive title" -d "Description of what needs to be done"
  ```

### During implementation
- Reference the issue ID in commit messages when relevant.

### After completing work
- Close the issue with a reason:
  ```sh
  bd close <issue-id> -r "Implemented in <commit or PR ref>"
  ```
- Use `--suggest-next` to see newly unblocked issues:
  ```sh
  bd close <issue-id> -r "Done" --suggest-next
  ```

### Key commands
| Action | Command |
|---|---|
| List open issues | `bd list` |
| Show issue details | `bd show <id>` |
| Create issue | `bd create "title" -d "description"` |
| Close issue | `bd close <id> -r "reason"` |
| Search issues | `bd search "query"` |
