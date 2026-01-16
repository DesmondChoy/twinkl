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

The project generates synthetic training data for the VIF using two approaches:

**Primary: Claude Code Subagents**
- `docs/synthetic_data/claude_gen_instructions.md` — Instructions for Claude Code to generate synthetic data at scale using parallel subagents
- Outputs to `logs/synthetic_data/<timestamp>/`

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

## Planning Mode Behavior

When in planning mode, be proactive and curious by using the `AskUserQuestion` tool to:

1. **Gather requirements upfront** — Before designing a solution, ask structured questions to understand the user's needs, constraints, and preferences
2. **Clarify ambiguities** — When requirements are unclear or could be interpreted multiple ways, ask rather than assume
3. **Surface better alternatives** — If you identify a potentially better implementation approach, present options to the user with trade-offs explained
4. **Validate key decisions** — At architectural decision points (library choice, data structures, API design), confirm direction before proceeding

The goal is collaborative planning — treat the user as a partner in design decisions, not just a recipient of your plan.