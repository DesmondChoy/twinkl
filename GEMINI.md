# GEMINI.md

This file mirrors `AGENTS.md` for Gemini. Keep their repository policy aligned;
only Gemini-specific interface wording should differ.

## Project Intent

Twinkl is an academic capstone and time-boxed proof of concept. It helps users
compare daily behavior with their declared priorities through an evolving
self-model, the VIF Critic, the Weekly Drift Reviewer, and explainable Weekly
Coach reflections.

For product, architecture, or evaluation decisions:

1. Read the relevant section of `docs/prd.md`; it is authoritative for product
   intent.
2. Follow any detailed specification that the PRD explicitly delegates to.
3. Use current code, tests, experiment reports, and Beads for implementation
   status. Do not infer current status from old planning prose.

Keep work practical: prefer the smallest complete, testable change over a broad
rewrite or speculative architecture.

## Product and Data Invariants

- Async persona generation is parallel between personas and sequential within
  each persona so Journal Entries remain coherent.
- Journal content should emerge from persona context, not rigid value labels.
- Preserve banned-term and value-leakage protections when changing prompts or
  generation logic.
- Do not expose generation or labeling metadata to logic intended to represent
  production behavior.
- Do not treat AI review as human validation. State the label or review source
  whenever it affects the conclusion.

## Repository Map

- `docs/prd.md` — product intent and current scope
- `docs/canonical_nouns.md` — required product terminology for maintained prose
- `src/vif/` — VIF Critic models, encoders, training, metrics, and run logging
- `src/{synthetic,judge,wrangling,registry}/` — data generation and labeling
- `src/{coach,nudge,evals}/` — downstream reasoning and evaluation components
- `scripts/{experiments,journalling,drift}/` — executable research workflows
- `tests/` — tests organized by the corresponding source area
- `logs/experiments/` — run records, reports, and the experiment index
- `docs/archive/` is historical; `docs/future_work/` is non-committed scope.

Inspect the live tree instead of relying on this map for exhaustive inventory.

## Environment and Commands

Do not use Git worktrees. Work in the main working directory and stay on the
current branch unless the user explicitly asks for another branch.

Activate the virtual environment before Python commands:

```sh
source .venv/bin/activate.fish   # Fish
source .venv/bin/activate        # Bash/Zsh
```

Use `uv` for dependency actions:

```sh
uv sync
uv add <package>
uv pip install <package>
```

Common checks:

```sh
uv run pytest <target>
uv run ruff check <target>
uv run mypy <target>             # When type behavior changed
```

Follow `pyproject.toml` for machine-enforced style. Keep comments concise and
only where they reduce cognitive load.

## Scope and Authorization

- A request to explain, inspect, diagnose, or review is read-only unless the
  user also asks for changes.
- A request to implement or fix authorizes the smallest necessary repository
  edits and proportionate verification, not unrelated cleanup.
- Do not create or switch branches, commit, pull/rebase, sync Beads remotely,
  push, open a PR, or change external state unless the user explicitly requests
  it or it is already part of the agreed task scope.
- Preserve unrelated working-tree changes. If they overlap the requested work
  and cannot be handled safely, stop and ask one concise question.

If ambiguity does not affect correctness or design direction, proceed with an
explicit assumption. Ask only when the answer would materially change the
result or authorize a meaningful external action.

## Planning and Delegation

- For non-trivial implementation or architecture choices, write a proportional
  plan in the active Beads issue before editing.
- Re-plan when evidence invalidates the approach. Skip formal planning for
  small edits and read-only questions.
- Use subagents only for bounded, independent work where parallelism materially
  improves speed or confidence. Give each one clear ownership and avoid
  concurrent edits to the same files.
- For architecture or capstone-scope decisions, present options and trade-offs.
  Do not silently commit the project to a major modeling or architecture choice.

## Issue Tracking with Beads

Use `bd` for implementation, durable research, and documentation work. Run
`bd prime` for current workflow instructions.

- Read-only analysis may inspect Beads when relevant but must not claim, create,
  update, or close issues unless tracker changes are requested.
- Before material repository edits, search for a matching issue. Claim it if it
  exists; otherwise create a focused issue.
- Capture a non-trivial plan, important decisions, and validation evidence in
  the active issue at meaningful milestones.
- Close an issue only after its acceptance criteria and relevant checks pass.
  Use a commit or PR reference only when one actually exists.
- Use `bd remember` for durable project knowledge. Do not create ad hoc memory
  files or automatically log every conversational correction.

## Verification

Verify in proportion to risk before describing work as complete:

1. Read the complete changed files, not only the diff.
2. Run targeted tests and Ruff checks for touched code; run broader checks when
   shared contracts or critical paths changed.
3. Run MyPy when type behavior or typed interfaces changed.
4. For documentation, verify links, canonical nouns, and claims against current
   behavior or source reports.
5. For data or experiments, preserve provenance and record the exact inputs,
   commands, seeds, outputs, and limitations needed to reproduce the result.
6. Remove dead code, debug remnants, and accidental generated files.
7. Inspect `git diff` and `git status`, then report what was and was not tested.

Before an authorized commit, follow `.claude/skills/quality/SKILL.md` manually
as the fresh-eyes review checklist; do not assume the `/quality` alias is
available.

When asked to fix a bug or failing CI, reproduce it, identify the root cause,
implement the smallest complete fix, and verify it without unnecessary
questions. A diagnostic or status request alone does not authorize edits.

Update affected documentation when behavior, contracts, commands, or adopted
decisions change. Do not perform a generic documentation sweep by default.

## Communication

- All Gemini prose on every surface, including interactive responses, plans,
  Beads issue text, reviews, handoffs, and maintained documentation, must
  strictly follow `docs/canonical_nouns.md`.
- Write in plain English for immediate understanding. Use technical jargon only
  when a real distinction requires it, and define it on first use.
- Do not invent synonyms for canonical product terms. This prose rule does not
  require renaming code identifiers, data fields, file paths, or historical
  records.
- Name the exact component, data, experiment setup, or output instead of vague
  words such as “system,” “surface,” “condition,” or “artifact.”
- In non-trivial handoffs, summarize material assumptions, the chosen approach,
  verification performed, and remaining risks.

## Git and Completion

Commit messages use `<type>: <summary>`. Prefer `feat`, `fix`, `refactor`,
`chore`, `docs`, and `test`. Reference the Beads issue in parentheses when
useful, for example:

```text
docs: refresh eval scope (twinkl-3cb)
```

An issue ID may be the prefix when the commit concerns only that issue. Use
`chore:` for tracker-only or maintenance-only commits.

Commit, pull/rebase, `bd dolt push`, and `git push` only when publishing was
explicitly requested or agreed. If publishing is in scope, verify both Git and
Beads remote state before claiming completion. Otherwise leave changes
uncommitted and report the working-tree state.
