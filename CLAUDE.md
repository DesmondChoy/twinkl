# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Twinkl is an "inner compass" that helps users align daily behavior with long-term values. Unlike traditional journaling apps that summarize moods, Twinkl maintains a dynamic self-model of the user's declared priorities and surfaces tensions when behavior drifts from intent.

The core component is the **Value Identity Function (VIF)** — an evaluative engine that compares what users *do* (journal entries) against what they *value* (declared priorities) across Schwartz value dimensions. Key properties: vector-valued (tracks multiple life dimensions), uncertainty-aware (holds back when data is sparse), and trajectory-aware (detects patterns over time).

This is an academic capstone project for the NUS Master of Technology in Intelligent Systems program.

## Commands

**Virtual Environment** (required before running Python):
```sh
source .venv/bin/activate.fish   # Fish shell
source .venv/bin/activate        # Bash/Zsh
```

**Package Management** (uses `uv`, not pip):
```sh
uv sync                    # Install dependencies from lockfile
uv add <package>           # Add new dependency
uv pip install <package>   # Direct install (prefer uv add)
```

**Running Notebooks**:
Notebooks are in `notebooks/` and should be run from the project root after activating the venv.

## Architecture

### Current Implementation: Synthetic Data Generation

The project generates synthetic training data for the VIF using two approaches:

**Primary: Claude Code Subagents**
- `docs/synthetic_data/claude_gen_instructions.md` — Instructions for Claude Code to generate synthetic data at scale using parallel subagents
- Outputs to `logs/synthetic_data/persona_<uuid>.md` (one file per persona)
- Registry tracks pipeline progress at `logs/registry/personas.parquet`

**Experimentation: Jupyter Notebooks** (for prompt iteration and testing)
- `notebooks/journal_gen.ipynb` — One-way journal generation
- `notebooks/journal_nudge.ipynb` — Two-way conversational journaling with nudges

### Configuration Files

- **`config/synthetic_data.yaml`** — Persona attributes (age, culture, profession), journal entry parameters (tone, verbosity, reflection_mode), and nudge settings
- **`config/schwartz_values.yaml`** — Rich psychological elaborations for each of the 10 Schwartz values (core motivation, behavioral manifestations, life domain expressions, typical stressors/goals)

### Prompt Templates

LLM prompts are stored in `prompts/` as YAML files with embedded Jinja2 templates:

```
prompts/
├── __init__.py              # Loader utility + exports
├── persona_generation.yaml  # Generate synthetic personas
├── journal_entry.yaml       # Generate journal entries
├── nudge_generation.yaml    # Generate follow-up nudges
└── nudge_response.yaml      # Generate responses to nudges
```

**Usage in notebooks:**
```python
from prompts import persona_generation_prompt, journal_entry_prompt

# Render a template
prompt_text = persona_generation_prompt.render(age="25-34", profession="Engineer", ...)
```

**YAML format:**
```yaml
name: prompt_name
description: Brief description
version: "1.0.0"
input_variables:
  - var1
  - var2
template: |
  Your Jinja2 template with {{ var1 }} placeholders...
```

**Adding new prompts:** Create a new YAML file in `prompts/`, then add it to `prompts/__init__.py` exports.

### Output Structure

Pipeline outputs are organized by persona UUID across multiple directories:

- `logs/synthetic_data/persona_<uuid>.md` — Raw synthetic journal entries per persona
- `logs/wrangled/persona_<uuid>.md` — Cleaned/parsed version for judge labeling
- `logs/judge_labels/persona_<uuid>_labels.json` — Labels per persona from judge pipeline
- `logs/judge_labels/judge_labels.parquet` — Consolidated training data (all personas)
- `logs/registry/personas.parquet` — Central tracking of pipeline stages

### Registry System

The central persona registry (`logs/registry/personas.parquet`) tracks each persona's progress through the data pipeline:

**Pipeline Stages:**
- `stage_synthetic` — Synthetic journal generation complete
- `stage_wrangled` — Data cleaning/parsing complete
- `stage_labeled` — Judge labeling complete

**Key Features:**
- File locking for safe concurrent writes from parallel subagents
- Enables incremental runs (only process personas at needed stage)
- Implementation: `src/registry/personas.py`

### VIF Critic Training Module

The `src/vif/` module implements the VIF Critic — an MLP that predicts per-dimension alignment scores from journal entries with MC Dropout uncertainty estimation.

```
src/vif/
├── __init__.py          # Module exports
├── encoders.py          # TextEncoder protocol + SBERTEncoder
├── state_encoder.py     # StateEncoder (builds state vectors)
├── critic.py            # CriticMLP with MC Dropout
├── dataset.py           # VIFDataset + data loading
├── eval.py              # Evaluation metrics (MSE, Spearman, calibration)
└── train.py             # CLI training script
```

**Usage:**
```python
from src.vif import CriticMLP, StateEncoder, SBERTEncoder

# Create encoder pipeline
encoder = SBERTEncoder("all-MiniLM-L6-v2")  # 384-dim embeddings
state_encoder = StateEncoder(encoder)        # 1,174-dim state vectors

# Train via CLI
# python -m src.vif.train --epochs 100
```

**Configuration:** `config/vif.yaml` (encoder model, hyperparameters, ablation presets)
**Checkpoints:** `models/vif/` (gitignored)
**Notebook:** `notebooks/critic_training.ipynb`

### Key Design Patterns

**Async Generation Pipeline**: Personas run in parallel via `asyncio.gather()`, but entries within each persona are sequential (for continuity). Results return in order regardless of completion time.

**Emergent Content**: Journal entries emerge from persona context (bio, reflection_mode) rather than prescriptive categories. The prompt tells the LLM *what kind of moment* to write about, not *what values* to express.

**Banned Terms Validation**: Generated content is validated against a list of Schwartz value labels and derivative adjectives to prevent "label leakage" — values must be shown through concrete life details, not named explicitly.

**Metadata Leakage Awareness**: Nudge decision logic only uses data available in production (entry content, previous entries), not synthetic generation metadata like `tone`.

## Documentation Hierarchy

- **`docs/PRD.md`** — Definitive specification (source of truth)
- **`docs/onboarding/`** — Onboarding flow specification and BWS values assessment
- **`docs/VIF/`** — VIF architecture, training, uncertainty logic
- **`docs/synthetic_data/pipeline_specs.md`** — Synthetic data pipeline design
- **`docs/ideas/`** — Brainstorming (not authoritative)

## Code Style

- **Imports**: Standard library first, then third-party
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Models**: Pydantic `BaseModel` for data structures; explicit JSON schemas for OpenAI structured output

## Quality Review

**Before every `git commit`**, run the `/quality` skill to review changes with "fresh eyes". This catches bugs that accumulate during implementation when focus is on making things work.

The quality review process:
1. Identifies all changed files via `git status` and `git diff`
2. Reads **entire files** (not just diffs) to understand full context
3. Checks for logic errors, missing error handling, type mismatches, dead code
4. **Fixes issues immediately** rather than just flagging them
5. Produces a summary of files reviewed and issues fixed

This is mandatory before committing to catch issues like:
- Logic errors that compile but behave incorrectly
- Missing edge case handling
- Integration issues between components
- Debug statements left in code

## Planning Mode Behavior

When in planning mode, be proactive and curious by using the `AskUserQuestion` tool to:

1. **Gather requirements upfront** — Before designing a solution, ask structured questions to understand the user's needs, constraints, and preferences
2. **Clarify ambiguities** — When requirements are unclear or could be interpreted multiple ways, ask rather than assume
3. **Surface better alternatives** — If you identify a potentially better implementation approach, present options to the user with trade-offs explained
4. **Validate key decisions** — At architectural decision points (library choice, data structures, API design), confirm direction before proceeding

The goal is collaborative planning — treat the user as a partner in design decisions, not just a recipient of your plan.