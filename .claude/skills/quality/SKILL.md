---
name: quality
description: Review scoped code or documentation changes for correctness, integration, and relevant verification. Use when a quality review is requested or before an authorized commit; fix findings only within an authorized implementation task.
---

# Quality Review

Review the requested changes with a fresh examination of their assumptions,
affected behavior, and evidence. Follow the user's scope and the repository's
[AGENTS.md](../../../AGENTS.md).

## Choose the Mode from Existing Authorization

- **Review-only:** For a review or inspection request, report actionable
  findings without editing files or changing issue state. An explicit
  read-only instruction takes precedence over earlier implementation authority.
- **Review and fix:** During authorized implementation, or when the user asks
  for fixes, correct relevant findings within that scope. Explain findings
  that need a new design decision or changes outside the authorized scope.

A standalone `/quality` request defaults to review-only unless fixes are
already authorized by the task. Do not ask again for permission already given.
Neither mode authorizes staging, committing, or publishing by itself.

## Establish the Changes Under Review

Use the requested files, commit, or comparison as the scope. For current local
work, inspect `git status --short`, `git diff`, and `git diff --cached` to
understand staged, unstaged, and untracked files. Review relevant untracked
files explicitly; they do not appear in ordinary diffs.

Keep unrelated working-tree changes outside the repair scope. When reviewing
for an authorized commit, distinguish the intended changes from other local
work. Read changed logic, relevant surrounding context, affected callers, and
tests. Read entire files when their behavior or structure requires it.

## Review Relevant Risks

Use the checks that apply to the changes:

- **Correctness:** Does the behavior match the request and current contracts?
  Check boundaries, empty inputs, error handling, and failure recovery.
- **Types and integration:** Trace changed interfaces through their callers.
  Check request and response shapes, state transitions, and persistence.
- **Project invariants:** Check the relevant product scope, input visibility,
  value-leakage protections, label provenance, and AI-review evidence limits.
- **Maintainability:** Identify dead code, unused imports, accidental debug
  remnants, or misleading names introduced or made obsolete by the change.
  Preserve intentional logging and avoid unrelated cleanup.
- **Documentation:** Verify affected claims, commands, links, and canonical
  product names against current behavior and source reports.

Report reproducible defects or specific contract risks. Separate them from
optional design preferences, and distinguish pre-existing issues from
regressions in the reviewed changes.

## Verify and Report

In review-and-fix mode, make the smallest complete correction and recheck the
affected behavior. Use targeted tests and Ruff for touched code, MyPy when type
behavior changes, and broader checks when shared contracts or critical paths
warrant them. For documentation-only changes, check references, consistency,
and claims. Existing relevant verification can be reused if the reviewed
changes have not invalidated it.

Once relevant checks pass, broaden or repeat them only when a new change,
failure, or unresolved risk warrants it. State unavailable checks explicitly.
Inspect the final scoped diff and Git status after fixes.

Report the scope and mode, actionable findings or fixes, verification results,
and remaining limitations. Link findings to the relevant file and location and
explain the trigger and consequence. If no actionable findings remain, say so
without claiming broader correctness than the review established.
