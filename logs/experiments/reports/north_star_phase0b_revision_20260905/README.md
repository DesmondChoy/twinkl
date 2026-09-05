# North Star Moment development revision: gate failed

**Issue:** `twinkl-fz34.1`. **Date:** 5 September 2026.

The revised reviewer and runner completed all 33 development cases. **The
semantic gate still failed:** independent AI reference review accepted 11 of
19 selected quotations (57.9%), and correct omission was 7 of 9 (77.8%). All
40 new provider attempts passed their contracts, with no retries or unmetered
calls. They cost US$0.09887802; cumulative spending is US$0.30505587. The
implementation passes **300 NSM tests**, Ruff, and scoped MyPy. These code
checks do not override the failed semantic criteria; dependent integration
remains blocked.

The initial `--run` request was rejected before process creation by automatic
approval review. The user then explicitly approved the frozen external run,
and execution proceeded through the normal approval path. OpenAI performed
28 runtime reviews. Gemini performed 12 missing exact-quotation reference
checks; its 28 original exhaustive references were reused. Gemini evaluates
the experiment independently and is not part of the proposed runtime selection
path. Its decisions do not enter the OpenAI review inputs.

## Implemented revision

The [offline diagnostics](diagnostics.md) and [derived JSON](diagnostics.json)
separate seven false selections against the frozen AI references from two
retrieval misses. Top-5 would increase source decisions from 74 to 104 while
fixing only one miss. This supports testing the review revision first while
retaining Nomic top-3 and the existing provider models and settings.

The new [review contract](../../../../src/north_star/review_v2.py) requires brief
factual assessments of the writer's action, its connection to the approved
Core Value definition, and whole-entry Conflict against that same Core Value.
The model returns one reason code; code derives the decision. This removes the
representational contradiction that invalidated both attempts in the original
failed case. Original source, quotation, complete-batch, and value-label checks
still apply. No new response had the original decision/reason contradiction,
but the revision did not satisfy the selection or omission criteria.

The [v3 runner](../../../../scripts/experiments/north_star_phase0b_v3.py) retains
the verified development sources, original exhaustive Gemini references, and
v2 recovery/shared-budget behavior. Both runtime selections and the entire
rank-one Journal Entry baseline are graded as exact quotations. A different
shorter reference quotation cannot substitute for the proposed display.
Known contradictory source/value references cannot approve a selection.
Unresolved references and the original nine-case omission denominator remain
visible; reference decisions never enter runtime inputs.

## Measured result and frozen protocol

The [executed report](run/executed_report.json) preserves the original paid-run
output. The [replayed report](run/report.json) reproduces every case, selection,
grade, attempt, and cost; only its execution mode differs. The replay requested
zero provider calls and left the cumulative ledger unchanged.

| Measure | Revised result | Gate or interpretation |
|---|---:|---|
| Accepted selected quotations | 11/19 (57.9%) | Fails zero incorrect selections. |
| Correct omission in nonempty histories with no reference-accepted source | 7/9 (77.8%) | Fails 100% omission. |
| Reviewer coverage | 19/33 (57.6%) | Development episodes, including empty histories. |
| Retrieval-only exact-quotation precision | 7/28 (25.0%) | Exact proposed whole-entry quotations are graded. |
| Retrieval-only coverage | 28/33 (84.8%) | Higher coverage accompanies lower precision. |
| Quotation-precision difference | +32.9 percentage points | Descriptive comparison; different selections and coverage. |
| Task-reference retrieval recall at k=3 | 17/19 (89.5%) | Unchanged frozen ranking and references. |
| Runtime abstentions | 10/74 (13.5%) | Valid reviewed-source decisions. |
| OpenAI unexpected failures | 0/28 | All new attempts; no retries. |
| Gemini unexpected failures | 0/12 | All new exact-quotation attempts; reused references excluded. |
| Structurally empty histories | 5/5 omitted without calls | Separate from the nine semantic-omission cases. |
| Failed case executions | 0/33 | Every required request or reused reference completed. |
| Accepted saved Persona examples | Wei Jun and Marc | Meets the bounded development demonstration criterion. |

The original reported precision was 12/19. Applying this revision's conservative
contradictory-reference rule to the original retained selections yields 11/19
there too. Thus the raw 12/19-to-11/19 change is not clean evidence of model
regression. Both runs fail the unchanged zero-incorrect-selection criterion.
Correct omission rose from 5/9 to 7/9 on the same frozen reference denominator.
Five of those nine histories contain reference abstentions; the denominator
means no AI-reference-accepted example, not proved absence of supportive action.

The eight nonaccepted selections comprise five primary-reference disagreements,
two exact-quotation rejections, and one unresolved contradictory reference:

| Case and selected entry | Nonacceptance |
|---|---|
| `2541429a:tradition:episode_01`, entry 5 | Primary reference: `wrong_value`. |
| `66ced716:universalism:episode_01`, entry 2 | Primary reference: `ambiguous`. |
| `7c712a0a:conformity:episode_01`, entry 7 | Primary source supportive; exact quotation rejected as `wrong_value`. |
| `87e92805:security:episode_02`, entry 4 | Identical source/value has conflicting primary references across episodes; unresolved. |
| `9d126412:power:episode_01`, entry 1 | Primary reference: `wrong_value`. |
| `bf44e50f:hedonism:episode_01`, entry 1 | Primary reference: `same_value_conflict`. |
| `dbe2c53d:conformity:episode_01`, entry 4 | Primary reference: `same_value_conflict`. |
| `e6838e16:security:episode_01`, entry 0 | Primary source supportive; exact quotation rejected as `wrong_value`. |

Three original disagreements were resolved: `152df7a4` and `7ff1d0fb` now omit,
and `5fa8b540` selects an accepted earlier entry. Four original disagreements
persist. The newly nonaccepted selections are `7c712a0a`, `9d126412`,
`e6838e16`, and the reference-rule-only `87e92805`. These are AI-reference
comparisons, not human adjudications or proof of each underlying behavior.

The [protocol](../../../../docs/north_star/phase0b_revision_20260905.md),
[manifest](run/manifest.json), and [execution freeze](run/execution_freeze.json)
bind all 33 development episodes, code, source hashes, original evidence, and
budget seed. The eight reserved Persona histories remain unevaluated. Coverage
uses the 33 development episodes; closed-week coverage under application
Core Value priority awaits integration.

| Accounting item | US dollars |
|---|---:|
| Original 61 attempts, already incurred | 0.20617785 |
| Conservative maximum additional spending or reservations | 10.45116090 |
| Conservative cumulative maximum | 10.65733875 |
| Largest planned single-attempt bound | 0.08900850 |
| Existing cumulative ceiling | 20.00 |
| Existing per-attempt ceiling | 0.25 |
| Actual new spending, 40 attempts | 0.09887802 |
| Actual cumulative spending, 101 attempts | 0.30505587 |

The bound includes maximum output allowances and one retry for each possible
runtime, selected-quotation, and baseline-quotation request, without relying
on cache discounts or reuse. Unneeded requests will be omitted and completed
identical requests reused. Missing usage retains its reservation. The existing
locked cumulative ledger and origin record prevent budget resets across runs.

The explicitly approved run used:

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/private/tmp/twinkl-nsm-v3-uv
uv run python scripts/experiments/north_star_phase0b_v3.py --run \
  --directory logs/experiments/reports/north_star_phase0b_revision_20260905/run
```

The command verifies the freeze before provider work. Do not edit frozen code,
protocol, inputs, or seed and then reuse this directory. A failed evaluation
must retain its outcome; the remaining budget does not establish feasibility.
Use `--replay` to reconstruct these saved outcomes without provider transport.
There are no pending attempts to resume in this run.

| New provider work | Calls | Calculated cost | Median attempt latency |
|---|---:|---:|---:|
| OpenAI, `gpt-5.6-luna`, reasoning `none` | 28 | US$0.01617552 | 3.870 seconds |
| Gemini, `gemini-3.5-flash`, thinking `low` | 12 | US$0.08270250 | 3.385 seconds |

These are offline provider-attempt timings, not live user wait times. The
[budget snapshot](run/budget.json) contains all 101 cumulative receipts. This
revision evaluates 69 unique underlying attempts: 40 new attempts, 28 reused
exhaustive references, and one reused exact-quotation check. Reuse is not
counted as a new provider call or charged twice.

## Validation and limitations

[Validation evidence](validation.json) and the [mechanical audit](audit.json)
record checks and hashes. All 19 selected quotations and 28 retrieval-only
quotations pass independent exact-source, identity, and chronology checks.
All 39 original report files and eight reserved Persona hashes remain unchanged.
The full NSM
suite passes **300 tests: 193 existing and 107 new**. New tests cover the revised
reason mapping, exact quotation and batch contracts, source boundaries, budget
bounds, unchanged historical evidence, frozen inputs, missing receipts,
interrupted completed responses, exhausted retries, pending reservations,
coalescing, reference contradictions, and metric denominators.

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/private/tmp/twinkl-nsm-v3-uv
uv run pytest tests/north_star tests/evals/test_north_star*.py
uv run ruff check src/north_star/review_v2.py \
  scripts/experiments/north_star_phase0b_v3.py \
  scripts/experiments/north_star_phase0b_diagnostics.py \
  tests/north_star/test_review_v2.py \
  tests/evals/test_north_star_phase0b_v3.py \
  tests/evals/test_north_star_phase0b_diagnostics.py
uv run --with 'mypy==2.3.0' mypy --explicit-package-bases --follow-imports=silent \
  src/north_star/review_v2.py \
  scripts/experiments/north_star_phase0b_v3.py \
  scripts/experiments/north_star_phase0b_diagnostics.py \
  tests/north_star/test_review_v2.py \
  tests/evals/test_north_star_phase0b_v3.py \
  tests/evals/test_north_star_phase0b_diagnostics.py
```

An independent review found and resolved inherited invalidation compatibility,
candidate-quotation wrong-value reporting, historical receipt provenance, and
candidate budget source bounds. A separate 33-case offline rehearsal, with
provider completion explicitly forbidden, made zero calls and correctly kept
missing revised runtime work failed. That rehearsal is not a paid development
result and cannot pass the semantic gate.

These are implementation checks and a synthetic development evaluation with
independent AI references. They do not establish human validation,
application/browser behavior, or final benchmark performance. The full
repository suite and application/browser tests were not run because this
change affects the offline NSM experiment only. `twinkl-fz34.1` is not closed:
its semantic criteria remain unmet. Resolve the remaining quotation/value
judgments and reference ambiguity in a separately documented development
decision before another experiment; changing retrieval alone does not address
these nonaccepted selections. No further paid run or architecture change is
adopted by this report.
