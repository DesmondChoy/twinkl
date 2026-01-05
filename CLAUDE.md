# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Twinkl is a voice-first "inner compass" that helps users align daily behavior with long-term values. Unlike traditional journaling apps that summarize moods, Twinkl maintains a dynamic self-model of the user's declared priorities and surfaces tensions when behavior drifts from intent.

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

The project is currently focused on generating synthetic training data for the VIF. Two Jupyter notebooks drive this:

1. **`notebooks/journal_gen.ipynb`** — One-way journal generation
   - Generates synthetic personas with Schwartz value profiles
   - Creates longitudinal journal entries that exhibit value drift/conflicts
   - Parallel async pipeline using `AsyncOpenAI`

2. **`notebooks/journal_nudge.ipynb`** — Two-way conversational journaling
   - Extends journal_gen with a nudging system
   - Rule-based decision logic determines when/how to nudge
   - LLM generates natural language nudges; rules validate output

### Configuration Files

- **`config/synthetic_data.yaml`** — Persona attributes (age, culture, profession), journal entry parameters (tone, verbosity, reflection_mode), and nudge settings
- **`config/schwartz_values.yaml`** — Rich psychological elaborations for each of the 10 Schwartz values (core motivation, behavioral manifestations, life domain expressions, typical stressors/goals)

### Output Logging

Synthetic data runs are logged to `logs/synthetic_data/<timestamp>/` with:
- `config.md` — Run parameters
- `persona_XXX.md` — Each persona's entries/nudges/responses
- `prompts.md` — All LLM prompts for debugging

### Key Design Patterns

**Async Generation Pipeline**: Personas run in parallel via `asyncio.gather()`, but entries within each persona are sequential (for continuity). Results return in order regardless of completion time.

**Emergent Content**: Journal entries emerge from persona context (bio, reflection_mode) rather than prescriptive categories. The prompt tells the LLM *what kind of moment* to write about, not *what values* to express.

**Banned Terms Validation**: Generated content is validated against a list of Schwartz value labels and derivative adjectives to prevent "label leakage" — values must be shown through concrete life details, not named explicitly.

**Metadata Leakage Awareness**: Nudge decision logic only uses data available in production (entry content, previous entries), not synthetic generation metadata like `tone`.

## Documentation Hierarchy

- **`docs/PRD.md`** — Definitive specification (source of truth)
- **`docs/VIF/`** — VIF architecture, training, uncertainty logic
- **`docs/synthetic_data/pipeline_specs.md`** — Synthetic data pipeline design
- **`docs/ideas/`** — Brainstorming (not authoritative)

## Code Style

- **Imports**: Standard library first, then third-party
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Models**: Pydantic `BaseModel` for data structures; explicit JSON schemas for OpenAI structured output

## One-Word Commands

- `$craft`: Generate high-quality conventional commit messages for this session’s changes (do not commit; user reviews first).
  - Behavior:
    - Inspect staged/unstaged changes and summarize what changed and why.
    - Always propose a single commit message combining all changes.
  - Output format (no extra prose; emit only commit message text in code fences):
    - Single commit:
      ```
      <type>(<scope>): <summary>
      
      <body>
      
      - <bullet describing change>
      - <bullet describing change>
      
      Affected: <file1>, <file2>, ...
      Test Plan:
      - <how you verified>
      Revert plan:
      - <how to undo safely>
      ```

  - Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
  - Conventions:
    - Subject ≤ 50 chars, imperative mood; wrap body at ~72 chars.
    - Use BREAKING CHANGE: in body when applicable.
    - Add Refs:/Closes: lines for issues/PRs when available.
  - If context is missing, ask one concise question; otherwise proceed with best assumption and note it in the body.