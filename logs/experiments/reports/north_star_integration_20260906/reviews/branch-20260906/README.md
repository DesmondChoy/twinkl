# Branch review and corroborated fixes, 6 September 2026

Beads: `twinkl-fz34.12`. Both reviewers received the same [prompt](prompt.txt)
for the 11 commits between `493bc086a4205bc94ff64d2c5df677faa3255d17` and
`76fecbdcca508d45bbdf1a0fd8e25454ad6635c6`. The working tree was clean until
both reviews completed. The reviews are AI assessments, not human validation.

- [Opus 5 review](opus.txt): `claude -p --model claude-opus-5 --effort high
  --permission-mode plan`, with Read/Glob/Grep/Bash tools, JSON output and no
  session persistence. Exit 0, no permission denials, model metadata confirmed
  `claude-opus-5`.
- [Gemini review](gemini.txt): `agy -p --model gemini-3.8-flash-high --mode plan`,
  with JSON output. The first attempt returned an empty response after a
  `read_file` denial and was excluded. The completed retry used CLI terminal
  sandboxing and headless tool approval; the user also granted folder access.
  It returned substantive findings with status `SUCCESS` and no denied actions.

The reviewer responses are retained with trailing whitespace normalized,
including their unverified claims, locations, and test summaries. The dispositions and checks below are based on
separate inspection and local verification, rather than those summaries.

## Findings and decisions

| Finding | Disposition and evidence |
|---|---|
| Gemini: remove full prompts and raw responses from browser storage | Not adopted. The frontend README explicitly retains them for Inspect and session restoration. Confirmed Delete session clears server state and then browser state; storage failures display a warning. Replacing these inputs with hashes would change the adopted Inspect/resume contract. |
| Both: reject only the quotation containing an internal value label, rather than its whole batch | Preserve the current guard and clarify its limitation in the specification. The existing validator and parameterized tests deliberately reject words such as “security” and “achievement,” and the prior implementation report discloses whole-batch failure after bounded retry. Candidate salvage or distinguishing natural usage would change frozen validation semantics. The specification now states the actual behavior explicitly. |
| Gemini: saved replays exclude nudge responses | Preserve. These responses lack independent availability evidence. The specification and integration report already disclose their exclusion and the corresponding evidence limit. |
| Gemini: make stale-session conflicts retryable | Not adopted. Repeating an obsolete revision cannot repair it, and the same branch covers deleted sessions. The hook discards responses while a mutation is busy or identity changes, then requests current state. Existing mutation/deletion regression tests pass. |
| Opus: capstone paper says integration and browser checks remain outstanding | Corrected the source status throughout, with links to the separate integration evidence. Numerical results and source-level walkthrough incorporation remain paused; the PDF remains explicitly stale and was not regenerated. |
| Opus: experiment log/index omit integration | Added links to the 36-week integration report and its implementation/browser validation. |
| Opus: earlier saved checks read as current | Added a historical scope note linking to the expanded integration. The original report body and results remain unchanged. |
| Opus: missing canonical North Star Moment term | Added the definition and its source and display boundaries to the glossary. |
| Independent corroboration: transient input-token-count failure cannot recover | Fixed. A timeout returned `failed`, `retryable=False`, with zero generation attempts; the service reused that failure even when explicitly retried. Counting now applies the same transient exception classification as generation, allowing the existing explicit retry path to recover. Invalid/mismatched/oversized inputs and permanent HTTP failures remain terminal. |

Opus also raised deployment availability/durability, dormant historical paid
scripts, unavailable-provider retries, pre-existing Experience terminology,
payload size, and QC/fixture coverage questions. No new supported-flow defect
was established for those items in this review. Production storage/deployment
and new paid historical runs are outside this correction. The documented live
ledger must still be preserved; an ephemeral deployment is not a global spend
ledger. No paid evaluation was run and no budget was reset.

## Runtime correction

The runtime now applies generation's timeout, connection and
HTTP 408/429/500/502/503/504 classification to counting failures. There is
no new automatic retry loop. The service reuses failed records until explicit
retry, and the frontend keeps its existing two-attempt interaction bound.
Completed Core Value requests are reused; generation attempt and spending
limits are unchanged. Exception messages from provider counting failures are
not copied into validation evidence.

The regression tests cover zero-generation failure and recovery, invalid and
oversized receipts, permanent HTTP failures, reuse of the first Core Value's
completed receipt when counting the second fails, and service recovery without
repeating Weekly Drift Detection or Coach Digest events. Before the fix,
12 regression cases failed at the expected retryability assertion.

## Frozen evidence and reproduction

Historical manifests, JSON results, receipts, and all five saved Persona
bundles remain unchanged. All 27 original integration input/execution hashes
match commit `76fecbdcca508d45bbdf1a0fd8e25454ad6635c6`. Use that exact revision
to reproduce the frozen integration run. The corrected `runtime.py`
intentionally differs from its recorded execution hash, so
the historical runner's `verify`/`run`/`export` checks must reject the newer
source. Do not replace those hashes with current ones or overwrite the
original results. Saved replay contract validation remains applicable to the
unchanged bundles, and live budget initialization still uses the finalized
original spending receipts.

## Verification

Commands run from the repository root activate `.venv` first and use
`UV_CACHE_DIR=/tmp/twinkl-review-uv-cache` for `uv`.

- Before edits: focused NSM/demo/evaluation Python tests passed; full Python
  suite: 1,670 passed and two existing frozen-manifest failures. Frontend:
  201 passed; TypeScript passed. Scoped Ruff passed.
- Final `uv run pytest`: 1,689 passed, two existing failures, 10 warnings in
  58.34 seconds. All 19 new regression cases passed; no new suite failures.
- `uv run ruff check src/north_star/runtime.py
  tests/north_star/test_runtime.py tests/demo/test_north_star_service.py` passed.
- `uv run --with 'mypy==2.3.0' mypy --follow-imports=silent
  src/north_star/runtime.py` passed. This is a scoped type check, not a claim
  that full-import typing is clean.
- Independent review of the final runtime-only fix found no actionable issues.
- All 19 added/changed local documentation links resolve, and `git diff --check`
  passed. Frozen input/execution comparison found only the intended current
  `runtime.py` mismatch; every hash matches the recorded reproduction commit.
- Frontend tests and TypeScript checks from before edits remain applicable:
  no frontend code changed. Browser QC was not rerun, and the already-stale
  paper PDF was not regenerated.

The initial review handoff left the changes uncommitted. The user subsequently
authorized temporary-file cleanup, commit, and push. Temporary runner outputs
were removed after checking the retained review copies. Publication is recorded
in Git and `twinkl-fz34.12`; no new paid experiment or change to reserved histories
was performed.

The two baseline failures are
`test_resolve_twinkl_752_5_null_cases.py::test_all_frozen_and_result_hashes_match_manifests`
and `test_weekly_drift_reviewer_prompt_alignment.py::test_saved_result_is_complete_and_reproducible`.
Both compare the unchanged current `config/schwartz_values.yaml` with an older
frozen hash. They were not hidden by updating historical evidence.
