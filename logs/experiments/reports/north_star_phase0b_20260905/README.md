# North Star Moment: failed development AI-review gate

**Date:** 5 September 2026. **Issue:** `twinkl-fz34`.
**Branch:** `codex/north-star-moment`. Frozen code and raw evidence snapshot:
[`f3030c8d`](https://github.com/DesmondChoy/twinkl/tree/f3030c8deb9400685185e7c620bda53a6f58e8aa).

Phase 0A passed, but **Phase 0B failed** the adopted acceptance criteria. The
[specification](../../../../docs/north_star/north_star_moment.md) requires
stopping dependent work when a phase fails. The application therefore retains
its existing Coach Digest behavior. No prompt adjustment, final evaluation,
or further paid work followed this result.

## Method and source boundaries

The user approved a small separate benchmark, US$20 total, US$0.25 per attempt,
and one retry before assignment, retrieval tuning, or paid review. Eight entire
non-demo Persona histories were reserved using seed 20260905 and SHA-256
identifier ordering. The development group contains 27 Personas and 33 Drift
episodes; the reserved group contains eight Personas and nine episodes. These
are NSM-specific reservations within an already studied synthetic corpus, not
a claim that the histories are untouched by earlier Twinkl research.

The [Phase 0A report](../north_star_phase0_20260905/README.md) records the exact
baseline reproduction, split, source hashes, fixed Nomic encoder revisions,
and local resource measurements. The query uses only the Core Value phrase
and approved definition. At k=1, 3, and 5, persisted-label proxy recall was
13/22, 21/22, and 22/22. The smallest passing setting, k=3, was frozen.
Positive LLM-Judge VIF Labels are a retrieval proxy, not NSM reference decisions.

All 33 development episodes entered the frozen [manifest](manifest.json).
Only original Journal Entries preceding onset in stored order and no later
than onset by date were eligible. Legacy nudge responses lack independent
availability evidence and were excluded; AI-written nudges, Persona biographies,
generation metadata, labels, and current Conflict text were absent from model
input. Sources were synthetic writing, not fresh user onboarding data.

For each of the 28 nonempty histories, `gpt-5.6-luna` at reasoning `none`
reviewed the first three retrieved sources, or all sources when fewer existed.
Independently, `gemini-3.5-flash` at thinking `low` reviewed **every eligible
source** to establish reference support and histories with no AI-reference-
accepted example.
The runtime selected the first code-valid supportive result in frozen retrieval
order. A selected quotation differing from a supportive reference quotation
received a predefined exact-candidate reference check: four such calls occurred.
Primary reference rejection or abstention counted as incorrect; an unresolved
reference could not approve a selection. This is independent-provider AI review,
not human validation. Prompts, schemas, decisions, raw JSON, model identifiers,
usage, and failures are saved; hidden model reasoning is not collected.

## Results and gate decision

| Measure | Result | Interpretation |
|---|---:|---|
| Selected quotations accepted by reference | 12/19 (63.2%) | Fails zero-incorrect-selection criterion |
| Correct omission where exhaustive reference found no valid source | 5/9 (55.6%) | Fails correct-omission criterion; excludes empty histories |
| Structurally empty histories | 5/5 omitted without calls | Deterministic exclusion, separate from semantic omission |
| Selections across development episodes | 19/33 (57.6%) | Experimental per-episode coverage; no minimum imposed |
| Task-reference retrieval recall at k=3 | 17/19 (89.5%) | Lower than persisted-label proxy recall |
| Failed cases after bounded retry | 1/33 | No result published by the offline runner |
| Unexpected invalid OpenAI attempts | 2/29 (6.9%) | Exceeds the 5% ceiling |
| Unexpected failed Gemini attempts | 0/32 (0.0%) | Within ceiling |
| Accepted saved Persona examples | Wei Jun and Marc | Meets example criterion, insufficient to pass other gates |
| Actual attempts / unmetered attempts | 61 / 0 | Includes unsuccessful attempts |
| Calculated cost | US$0.20617785 | Below both approved ceilings |

The coverage denominator is **episodes in this experiment**. It does not
implement the application's closed-week priority rule across multiple Core
Values and cannot be reported as live Active Drift card coverage. No synthetic
failure injections enter the paid failure denominator. The two invalid OpenAI
attempts were successful transport responses with inconsistent schema fields,
not provider outages. Quotation, identity, and source-order checks found no
failures among the 19 retained selections in the [audit](audit.md); those checks
did not establish semantic correctness.

The frozen [report.json](report.json) also contains `retrieval_only_precision`
of 7/28 (25.0%) and `verification_lift` of 0.3816. The first measure grades the
top-ranked **source**, whereas NSM precision grades a selected **quotation**.
Their subtraction is not a matched quotation-level verification effect and is
not used as evidence of one. The audit records the descriptive source-level
comparison, its changed coverage, and the absence of a causal inference.

Seven rejected selections comprise three `wrong_value`, two `ambiguous`, and
two `same_value_conflict` reference decisions. Universalism accounts for three;
Tradition, Security, Hedonism, and Conformity account for one each. The failed
case, `dbe2c53d:universalism:episode_01`, returned valid JSON twice but paired
`decision=abstain` with `reason_code=other_actor`, which is allowed only with
`not_supportive`. The strict contract rejected both batches and stopped after
the one permitted retry. Exact passages and all five saved Persona dispositions
are recorded in the [offline evidence walkthrough](audit.md).

## Implemented experimental architecture

The Phase 0A harness owns source eligibility, the history split, pinned CPU
encoding, deterministic ranking, and label-proxy measurement. The Phase 0B
runner consumes its frozen rankings and creates the immutable development
manifest before paid calls. `src/north_star/review.py` provides a strict
Pydantic review contract, source-grounded prompts, full-batch validation, exact
quotation and source-membership checks, and deterministic selection. Semantic
actor attribution remains the AI review's responsibility.

`src/north_star/provider.py` wraps OpenAI and Gemini with explicit attempt
limits and timeouts, disabled implicit SDK retries, conservative per-attempt
reservations, refusal handling, usage receipts, and request reuse/coalescing.
Its file ledger uses a lock and atomic replacement; interrupted or unmetered
attempts retain their reservation. This adapter was used for the experiment.
It has not been connected to the Experience service or exposed through Inspect.

The chosen future hosting arrangement is one lazily loaded, pinned CPU encoder
in the existing backend, outside the asyncio event loop with bounded concurrency
and one worker. A separate service adds deployment and network responsibilities
without removing encoder memory cost. The local Phase 0A run peaked at
1,584.1 MiB RSS and took 5.482 seconds including 2.174 seconds of encoding.
Container packaging, Railway capacity, representative live concurrency, and
full end-to-end latency were not measured or implemented.

## Paid operation measurements

| Provider and role | Attempts | Input tokens | Output tokens | Median attempt latency | Attempt latency range | Calculated cost |
|---|---:|---:|---:|---:|---:|---:|
| OpenAI experimental review | 29 | 37,664 | 4,398 | 2.073 s | 1.367–3.898 s | US$0.01425435 |
| Gemini reference and exact-quotation checks | 32 | 45,773 | 13,696 | 2.271 s | 1.261–7.939 s | US$0.19192350 |

Input totals include 1,285 cached OpenAI tokens and 33,505 OpenAI cache-write
tokens. The calculation applies the recorded cache-write uplift and cached
rates; Gemini output includes billed thinking tokens. The frozen
[policy](../../../../config/evals/north_star_moment_v1.json) records prices,
provider documentation, 8,192 output-token limit, 60-second timeout, and budget.
Costs derive from returned usage and those rates, not a provider invoice.
Latencies describe these development calls from this machine; they exclude
encoding, browser work, and the rest of the live application workflow.

## Reproduction, validation, and remaining work

Use [execution_freeze.json](execution_freeze.json) to check the exact inputs
and code used by this run. The [budget ledger](budget.json) contains every
actual attempt; [cases/](cases/) contains 28 nonempty-case receipts, while the
five no-call cases are retained in `report.json`. Completed requests were reused
when the complete run followed the first-case provider check, without duplicate
paid calls. The original JSON outputs remain unchanged after the audit.

The recorded execution sequence was:

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/twinkl-uv-cache
uv run python scripts/experiments/north_star_phase0b.py --prepare
# Freeze the execution hashes before the first authorized paid call.
uv run python scripts/experiments/north_star_phase0b.py --run
```

`--prepare` refuses to overwrite the manifest; the run exited with code 2
because the gate failed. Do not execute the paid runner merely to read these
results. Read `report.json`, the ledger, and the audit instead. Verification
commands make no provider calls:

```sh
uv run pytest tests/north_star tests/evals/test_north_star_phase0.py \
  tests/evals/test_north_star_phase0b.py tests/demo tests/coach \
  tests/test_drift_detector.py tests/test_weekly_drift_reviewer.py
uv run ruff check src/north_star scripts/experiments/north_star_*.py \
  tests/north_star tests/evals/test_north_star_phase0*.py
uv run --with 'mypy==2.3.0' mypy --explicit-package-bases --follow-imports=silent \
  src/north_star scripts/experiments/north_star_*.py
```

The focused NSM suite has 110 passing tests; the combined regression target
above has 249 passing tests. Ruff and isolated MyPy pass. Full-import MyPy
retains five pre-existing wrangling/registry errors described in the Phase 0A
report. The source, frozen receipts, and metric denominators received a separate
AI code/evidence audit. No full-repository suite or frontend suite was run for
this change; application code is unchanged.

[validation.json](validation.json) records the exact verification scope and
report hashes. The generated 37-page capstone PDF was visually inspected on
every page; the chart-label overlap and two separated table captions were
fixed. [Saved PDF QC captures](report_qc/) show the method, results, and offline
walkthrough pages. They are report renders, not browser screenshots.

The pre-publication quality review identified two limitations of the frozen
experimental runner, tracked in **`twinkl-fz34.9`** before any resumed run.
Preparation reloads source text without rejecting changes against its recorded
retrieval hash; retry replay can request another attempt after a saved failure
has exhausted its allowance, raising `BudgetError` instead of reconstructing
the report. Both were reproduced using temporary synthetic inputs without
provider calls. Every source and execution hash in this historical run was
independently verified, so neither changes its measured result. The frozen code
is retained for provenance; a future runner must address both limitations.

The remaining work is tracked as dependency-linked children `twinkl-fz34.1`
through `twinkl-fz34.9`: runner hardening precedes the Phase 0B decision, which
gates shared integration, Persona preparation and live lifecycle, React views,
browser QC, reserved evaluation, and the final implemented-feature report.

The measured gate failure is the current blocker. Shared application lifecycle,
source-change invalidation, saved Persona NSM bundles, React cards, quotation
expansion, source navigation, NSM Inspect, narrow/wide browser QC and screenshots,
and fresh onboarding through closed-week NSM review are **not implemented or
claimed as validated**. The eight reserved histories remain unevaluated.
Reopening this work requires a revised development approach and an explicit
decision to resume feasibility work under the existing acceptance criteria;
the remaining budget alone is not evidence of feasibility. The updated
[Technical Paper](../../../../docs/capstone_report/capstone_project_report.md)
reports this negative result separately from the existing application walkthrough.
