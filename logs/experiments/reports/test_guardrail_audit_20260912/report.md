# Capstone test and guardrail assessment — 12 September 2026

The stale-suite failures are resolved, and the added tests cover the identified
application and research-data defects. The full Python suite passes 2,111 tests;
the frontend passes 385 tests across 24 files. Two Chromium journeys also pass
against the built frontend and a controlled Python API. These results support
the academic capstone's implementation claims. They do not establish model
quality or human validation.

## Changes and verification

Historical experiment tests now replay pinned Git source archives while keeping
the original evidence hashes. The obsolete North Star Moment saved-check runner
and its tests were removed because they depended on a deleted experiment and
retired behavior. Preview scripts and retained VIF compatibility tests now state
what they actually verify.

Current Coach Digest generation checks every quoted passage and requires a
question in question form; the displayed nudge shares the structural question
check. Provider tests use the actual SDK with a local HTTP transport to verify
attempt counts, refusals, and incomplete responses. Parquet evaluation restores
prior-week evidence and distinguishes recorded validation from current rules.
Dataset merging rejects duplicate identity keys before they can multiply rows
or conceal missing entries. Regression tests exercise each of these defects.

| Check | Command or scope | Result |
|---|---|---|
| Python | `uv run --no-sync python -m pytest -q` | 2,111 passed; exit 0 |
| Frontend | `npm test -- --run` in `frontend/onboarding` | 385 passed, 24 files |
| Browser | `npm run test:e2e` in `frontend/onboarding` | 2 passed; Chromium at 390px width |
| Frontend compilation | Browser startup runs TypeScript and Vite build | Passed; bundle-size warning remains |
| Ruff | All changed and new Python files | Passed |
| MyPy | Eight changed source/helper files, with `--follow-imports=silent` | Passed |
| Additional MyPy checks | `src/vif/dataset.py` and `src/coach/weekly_digest.py` | Existing errors reproduced against unchanged Git source; not a clean repository-wide type check |
| Diff | `git diff --check` | Passed |

Python commands used the activated repository virtual environment and a
temporary uv cache. The repository already configures pytest's quiet option,
so the final extra-quiet log omits its numerical footer; its 2,111 passing
progress markers and exit status establish the reported total. Deprecation
warnings remain. Browser execution required local server binding and Chromium
process access. No paid model calls were made.

## Existing saved Coach Digests

The [per-response assessment](./saved_response_checks.json) applies current
mechanical rules to the catalog-hash-verified weekly inputs and existing saved
responses. It records input and response hashes without changing saved
validation records. The assessment uses `collect_cases`,
`validate_weekly_digest_narrative` with voice checks enabled, and
`validate_demo_comparison_narrative` with current validation policy.

All 44 responses in the 22 paired Coach Digest comparisons pass. Of the 27
older base responses, 21 pass. The default replay uses the 22 comparison
responses without a North Star Moment plus five base responses, so 25 of its
27 displayed digests pass current rules.

The displayed Lukas response for 9 June 2025 fails quotation grounding and a
voice check. The displayed Wei Jun response for 26 May 2025 fails quotation
grounding. Each unmatched quotation matches its source after removing the
terminal comma: these are punctuation differences, not evidence of invented
passages. Historical records remain intact; the new generation checks apply
to future responses. Question-form validation remains an English structural
heuristic and does not prove that a question is reflective or non-prescriptive.

## Remaining evaluation work

The new behavioral corpus supplies eight pairs of Codex-authored synthetic
examples. Its 16 responses and source boundaries are covered by deterministic
tests, but the semantic expectations have not been measured through a live
judge or validated by people. Combined Coach Digest and selected North Star
Moment review also remains outside this corpus. The remaining useful capstone
work is bounded semantic review and review of the two flagged saved responses;
these test results do not justify broader claims about model reliability.
