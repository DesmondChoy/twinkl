# CLAUDE.md

This file started as Claude-specific guidance and is now adapted for
Codex-style execution in this repository.

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

Notebooks live in `notebooks/` and should be run from the project root
after activation.

## Architecture Snapshot

### Synthetic Data Pipeline

The project generates synthetic VIF training data through:

- Pipeline instructions: `docs/pipeline/claude_gen_instructions.md`
- Persona registry: `logs/registry/personas.parquet`
- Raw outputs: `logs/synthetic_data/persona_<uuid>.md`
- Wrangled outputs: `logs/wrangled/persona_<uuid>.md`
- Judge labels: `logs/judge_labels/persona_<uuid>_labels.json`
- Consolidated labels: `logs/judge_labels/judge_labels.parquet`

Registry stages in `src/registry/personas.py`:

- `stage_synthetic`
- `stage_wrangled`
- `stage_labeled`

### Configuration

- `config/synthetic_data.yaml`
- `config/schwartz_values.yaml`
- `config/vif.yaml`

### Prompt Templates

Prompt templates live in `prompts/*.yaml` and are exposed through
`prompts/__init__.py`.

### VIF Module

Core training/eval code lives in `src/vif/`:

- `encoders.py`
- `state_encoder.py`
- `critic.py`
- `critic_ordinal.py`
- `dataset.py`
- `eval.py`
- `train.py`

## Implementation Principles

- Async persona generation is parallel per persona and sequential within a
  persona for continuity.
- Journal content should stay emergent from persona context, not rigid value
  labels.
- Keep banned-term/value leakage protections intact when touching prompts or
  generation logic.
- Avoid metadata leakage in any logic intended to mirror production behavior.

## Documentation Hierarchy

- `docs/prd.md` (authoritative)
- `docs/onboarding/`
- `docs/vif/`
- `docs/pipeline/pipeline_specs.md`
- `docs/archive/` (historical only)
- `docs/future_work/` (non-committed ideas)

## Code Style

- Imports: standard library first, then third-party, then local.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Keep comments concise and only where they reduce cognitive load.

## Quality Gate Before Commit

Before creating a commit:

1. Before running `git commit`, run `.claude/skills/quality/SKILL.md`.
2. Inspect full changed files, not only diffs.
3. Run targeted tests/linting for touched modules.
4. Remove obvious dead code and debug remnants.
5. Confirm no behavior regressions in critical paths.

If there is ambiguity and no blocking risk, proceed with explicit
assumptions and note them. If ambiguity affects correctness or design
direction, ask one concise clarifying question.
Use 'bd' for task tracking
