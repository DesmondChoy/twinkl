# Onboarding QA fixes and verification

**16 September 2026 · Beads `twinkl-rklc.53`**

The implementation addresses the five corroborated onboarding QA findings,
with desktop and laptop browsers as the primary target. Verification was
performed locally before publication; these results do not verify a Railway
deployment. The original QA used fictional Journal Entries written by AI
reviewers; the checks below do not establish human usefulness, customer
satisfaction, or general model accuracy.

## Changes

| Finding | Implemented behavior | Evidence |
| --- | --- | --- |
| Historical Coach recovery left its Moment unavailable | A valid historical Coach Digest offers **Review moment** when its review is missing and **Resume moment review** for a stored pending review. Local failures retain the retry action. Ordinary history browsing and reload make no new Experience API requests. | Hook/component regressions and the complete Chromium recovery flow. |
| A historical Moment could attach to newer activity | New Moment events reference the authoritative digest for the requested week. Later Journal Entries or completed weeks cannot redirect its Inspect ancestry. Existing completed records retain their identities. | Backend lineage regression and a browser reproduction that failed before the fix and passed after it. |
| An exact quotation containing “improvement” was falsely rejected | Current validation checks generated narration for prohibited transition wording, excluding quotations that match supplied source text exactly. Altered or invented quotations and unsupported generated transition claims still fail. | Original captured response now passes; negative controls and frozen-policy tests pass. |
| Coach retry repeated a known quotation error | Each explicit retry makes at most one provider attempt and receives the latest matching invalid response's validation failures. Failed attempts, malformed raw responses, and the actual repair prompt remain inspectable. No Weekly Drift re-review is added. | Retry, idempotency, provider-failure, malformed-output, and input-identity tests. |
| Accepted reflections added details and extra questions | Prompt **4.7** preserves actor, action, occasion, means of interaction, and explicit uncertainty about motives. Validation requires one generated question across all three fields; exact source questions remain allowed. | Captured two-question response is rejected; three bounded live samples and source review below. |
| Source navigation lost the weekly reading context | Weekly evidence, Coach citations, and Moment links use the existing centered Journal Entry dialog. Closing or pressing Escape restores focus to the source link and retains the selected week. | Desktop/narrow browser tests and visual inspection of the dialog. |

The mobile change is limited to showing the existing Journal section links in
a compact wrapping row. No separate mobile layout was introduced. Inspect now
distinguishes source-availability timestamps from Journal Entry chronology and
puts dates before source identifiers. The PRD and interaction specifications
record desktop/laptop priority.

Saved Persona responses were not regenerated. The original prompt **4.6** is
archived byte-for-byte, and replay, generation scripts, the earlier context
experiment, and batch validation retain the appropriate historical policy.
Comparison against the original validator produced identical complete
validation outputs for **192 synthetic fixture/settings combinations**,
including their check details and length rules. These are eight constructed
narratives under 24 configurations each, not 192 saved model responses. See
[legacy-validation-equivalence.json](legacy-validation-equivalence.json) and
the [exact fixtures and reproduction command](legacy-validation-command.md).

## Three approved live Coach requests

The user explicitly approved three requests to the configured OpenAI API.
Each case received one `gpt-5.6-luna` request with reasoning effort `none`;
there were no repair attempts or evaluator calls. No seed was set. The
requests used the original frozen Weekly Drift Detection outputs, without
the review concerns or expected answers. These are selected development
cases, not a held-out evaluation.

| Captured case | Current response and independent AI source review |
| --- | --- |
| Exact task-improvement quotation | The original response passes current validation. The fresh response chose a different exact quotation and described the map, course, and exhibition consistently with the supplied writing. The live sample alone does not test acceptance of the original quotation. |
| Visit/action attribution and two questions | The fresh response distinguishes visiting the brother from sending comments on the teammate's slides. It asks one question. The unsupported teammate visit is absent. |
| Uncertain wish for acknowledgment | The fresh response preserves uncertainty about wanting acknowledgment versus being tired. The definite claim of expecting nothing in return is absent. A second AI reviewer noted that “your own needs were present” remains a broader interpretation than the source explicitly establishes. |

All three fresh responses pass the final deterministic checks. Exact
quotation matching and question counts do **not** establish the truth of
surrounding interpretation. The original unsupported-motive response still
passes mechanical validation; this work does not claim to solve semantic
accuracy with word matching. The broader evaluation work in `twinkl-z5nr`
and the historical saved-output cases in `twinkl-rklc.39` remain open.

[inputs.json](inputs.json) preserves the exact inputs, original responses,
source hashes, and original session/event identifiers.
[receipts.json](receipts.json) retains the actual requests, responses, model
metrics, and generation-time checks. The final validator adds a distinct
`single_generated_question` receipt marker for version routing; the retained
responses were rechecked without further provider calls in
[verification.json](verification.json). Provider generation is stochastic,
so repeating the same request may produce different writing.

## Engineering verification

- **1,083 Python tests passed** across Coach, demo, North Star Moment,
  evaluations, the pinned context experiment, and model-output guardrails.
  Six warnings concern existing deprecated runtime helpers.
- **394 frontend tests passed** across all 24 Vitest files.
- **Six Chromium E2E tests passed:** three flows at 1280×900 and the same
  flows at 390×844. They use the built frontend and real local Python HTTP
  boundary with controlled provider doubles. Coverage includes onboarding,
  writing, historical Coach/Moment recovery after newer activity, source-dialog
  focus, zero API work while browsing/reloading history, and deletion.
- TypeScript checking and the Vite production build passed as part of E2E
  startup. Vite emits a large-chunk warning.
- Ruff passed for all **17 changed Python files**; `git diff --check` passed.
- Scoped MyPy 2.3.0 found **six existing Polars scalar-conversion errors** in
  unchanged sections of `src/coach/weekly_digest.py`, at lines 288, 289, 563,
  670, 673, and 676. A shadow check of the original file produced the same
  errors. The other changed implementation files pass scoped typing.
  [Exact baseline comparison](mypy-baseline-comparison.json).

The entire repository's VIF/training suite was not rerun. This work does not
establish production restart durability, Safari behavior, physical-device
behavior, or human acceptance. The browser fixes were checked locally; a
hosted deployment was not part of this verification.

## Reproduction

From the repository root, activate `.venv`, then run:

```sh
uv run --no-sync pytest tests/coach tests/demo tests/north_star tests/evals tests/experiments/test_run_coach_context_eval.py tests/test_model_guardrails.py
PYTHONPATH=. uv run --no-sync python logs/experiments/reports/onboarding_qa_fixes_20260916/run_smoke.py
```

The second command is offline: it revalidates the original and captured
responses. `run_smoke.py --execute --output <new-file>` would make three new
provider requests; existing live receipts cannot be overwritten. Use a
writable `UV_CACHE_DIR` if the default cache is
restricted.

From `frontend/onboarding`:

```sh
npm test
npm run test:e2e
```

The browser command builds the app and starts its controlled local server on
port 8765. Desktop is the first Playwright project; the narrow project retains
existing compatibility coverage.
