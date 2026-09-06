# North Star Moment implementation validation

The first POC integration covers fresh onboarding and all five saved Personas.
It uses full eligible history with Luna low; xhigh source and quotation
assessment is evaluation-only. An absent Drift is not evidence of alignment.
One optional card uses pre-onset reflection, current-week encouragement, or an
older reminder, with exact quotation and source/owner/Profile/time checks.
Insufficient Evidence and unsuitable writing omit the card.

## Implementation checks

- Shared runtime, provider accounting, source rules, scenario integration and
  live service: 339 targeted tests passed before the final trace reconciliation.
- After the final reconciliation, all 16 NSM service tests and the broader 50
  Experience service/deployment tests passed. Tests cover missing asynchronous
  events, pending/completed invalidation, explicit record identity checks,
  coalescing, deletion, disconnects, response availability and restore.
- Broad Python regression run: 1661 passed, 2 failed, 10 warnings in 56.69s.
  The only failures are pre-existing July frozen-manifest comparisons in
  `test_resolve_twinkl_752_5_null_cases.py::test_all_frozen_and_result_hashes_match_manifests`
  and `test_weekly_drift_reviewer_prompt_alignment.py::test_saved_result_is_complete_and_reproducible`.
  Both compare an older expected `config/schwartz_values.yaml` hash with the
  current unchanged file. No baseline artifact was rewritten to make them pass.
- Scoped Ruff and MyPy 2.3.0 with `--follow-imports=silent` passed for the
  changed source files. The existing full-import typing baseline is not a
  passing claim. TypeScript and production build pass.
- Final frontend verification against the final exported bundles: all 201
  tests in 15 files passed, including selected-week Inspect navigation and
  five complete persistence round trips; TypeScript and production build
  passed. The existing Vite chunk-size advisory remains.
- [Browser checks](browser-qc/README.md) cover both real frontend paths,
  all 36 replay weeks, 1440px/390px layouts, source links, Inspect, keyboard
  expansion, retry, reload and in-flight source removal/session deletion.

## Self-review and Opus review

The repository quality checklist guided review of the changed logic, callers,
contracts, provenance, lifecycle and verification. Independent Opus review
used `claude -p --model opus` with read-only Read/Grep/Glob access. Its initial
review and follow-up were adjudicated against current implementation and
actual browser evidence; AI review is not human validation. The [final Opus
review](reviews/opus-final.md) found no reproducible safety or correctness
regression in the final trace reconciliation and closed the remaining
report/export and budget-runbook findings. The [initial review](reviews/opus-initial.md)
and [follow-up](reviews/opus-followup.md) retain the full findings and their
evidence-based dispositions.

Corrections include separate live storage, cumulative spending preservation,
worker-thread isolation for file locks/fsync, serialized cross-process token
count receipts, recoverable not-ready requests, stale frontend response
recovery, deletion guards, per-week Inspect focus, and browser/server NSM trace
reconciliation during source mutation. Live execution has one worker and a
16000-token complete-input ceiling. It writes only under the ignored
`logs/exports/demo_tool_runs/north_star/` directory. Finalized experiment spend
seeds its private ledger; later source changes fail closed.

Opus withdrew the proposed nudge-generation-time substitution because it does
not prove user response availability. It also withdrew its immediate browser
storage concern after five complete persistence round trips and measurement:
the largest saved Experience is 780192 UTF-8 bytes (about 1.56MB with conservative
UTF-16 accounting); storage quota failures already show a warning. Full
prompts/source/raw responses intentionally remain in browser Inspect data and
server session memory; the documentation now states this explicitly.

The temporary in-progress report prevented live budget initialization and
would have invalidated old bundle hashes on completion. Final report creation
and re-export are complete. An offline live-ledger seed inherited all 134
charged attempts with raw responses redacted, and left every source file
unchanged. Thus this temporary artifact state is resolved. Provider/budget failures retain their visible bounded
recovery behavior rather than being silently relabeled as semantic omissions.
The strict internal-label quotation filter remains unchanged: ordinary words
that also name internal values can still cause a failed review after bounded
retry. This residual and unbounded live history growth remain POC limitations.

## Evidence and delivery boundary

The pre-publication quality review found no new actionable issues. A fresh run
of the saved-scenario and NSM service tests passed all 37 tests. All 18 frozen
input hashes, nine execution hashes, and four audited output hashes still
match. Existing frontend, type, build, and browser checks remain applicable;
publication preparation changed only documentation and added the final
[workflow infographic](../../../../docs/north_star/assets/north-star-moment-workflow.png).

The stronger evaluator accepted 26 of 30 selected quotations, with four
unresolved judgments; one additional omitted week missed a supportive example.
The [saved-Persona report](README.md) records all selections, omissions,
independent AI disagreements, invalid attempts, costs and frozen inputs.
The original experiments and all reserved source hashes remain unchanged.
There is no human benefit study, reserved final benchmark result, production
multi-user storage guarantee, or new capstone report/PDF incorporation.
This report records implementation-stage validation. Subsequent publication
is recorded in Git history and the parent issue, `twinkl-fz34`.
