# AGENTS.md

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
Stay on the current branch by default. Do NOT create or switch to a new
branch unless the user explicitly tells you to. If branch isolation seems
safer, ask first instead of deciding unilaterally.

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

Script-based generation/judging helpers live in `src/synthetic/`,
`src/judge/`, and `scripts/journalling/`.

## Architecture Snapshot

### Source Code (`src/`)

- `src/vif/` — VIF critic models (MLP ordinal, BNN), text/state encoders, dataset loading, training loops, evaluation metrics, and experiment logging
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

### Scripted Workflows

- `src/vif/train.py` and `src/vif/train_bnn.py` — Critic training CLIs
- `scripts/experiments/critic_training_v4_review.py` — Canonical frontier experiment driver for multi-model VIF review runs
- `src/synthetic/generation.py` — Synthetic generation primitives and safeguards
- `src/judge/labeling.py` — Judge rubric + scoring helpers
- `scripts/journalling/` — Lightweight sanity-check scripts for generation/judge flows

### Tests (`tests/`)

- `tests/vif/` — Eval metrics, loss functions, ordinal base tests
- `tests/wrangling/` — Wrangled data parser tests

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
- Avoid metadata leakage in any logic intended to mirror production behavior.

## Code Style

- Imports: standard library first, then third-party, then local.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Keep comments concise and only where they reduce cognitive load.

## Quality Gate Before Commit

Before creating a commit:

1. Run `/quality` (repo alias for `.claude/skills/quality/SKILL.md`) before committing to review changes with fresh eyes.
2. Review complete changed files, not only diffs.
3. Run targeted tests/linting for touched modules.
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

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write a plan with checkable items
2. **Verify Plan**: Use the `AskUserQuestion` tool to check in with the user
   before starting implementation. Present structured, multiple-choice questions
   to resolve ambiguity quickly and keep the workflow moving. (Unavailable in
   non-interactive mode / `codex exec`.)
3. **Explain Changes**: High-level summary at each step
4. **Capture Lessons**: Update `tasks/lessons.md` after corrections
5. **Update Documentation**: Run parallel sub-agents to scan potentially
   affected documentation and update where needed

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
