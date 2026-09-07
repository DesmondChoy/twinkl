# Core Value Definitions in the Weekly Drift Reviewer (`twinkl-j3k7`)

**Date:** 2026-09-07

**Scope:** detected Drift totals, Core Values, and individual Drift changes

**Status:** complete; runtime definitions implemented, comparison independently reconciled

## Results

Adding the selected Core Values' Schwartz-based definitions plus
project-specific elaborations changed detected Drift totals from
**22, 22, and 21** to **19, 21, and 18** across the three Runs.
The median total fell from **22 to 19**, while the composition changed across
Core Values and individual Personas. This establishes a change in detections;
it does not establish an improvement or regression in accuracy.

Each cell below shows **baseline v3 → definitions v4 (difference)**. The
definitions variant includes both existing `definition` and `core_motivation`
fields; neither was evaluated separately. Every
Run column sums to its total. Personas may contribute more than one Core Value.

| Core Value | Personas | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|---:|
| Self-Direction | 24 | 2 → 2 (+0) | 2 → 2 (+0) | 2 → 2 (+0) |
| Stimulation | 24 | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) |
| Hedonism | 33 | 3 → 3 (+0) | 3 → 2 (-1) | 3 → 2 (-1) |
| Achievement | 25 | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) |
| Power | 37 | 1 → 1 (+0) | 2 → 2 (+0) | 1 → 3 (+2) |
| Security | 36 | 3 → 2 (-1) | 2 → 3 (+1) | 3 → 2 (-1) |
| Conformity | 28 | 4 → 2 (-2) | 2 → 2 (+0) | 3 → 1 (-2) |
| Tradition | 28 | 1 → 0 (-1) | 0 → 0 (+0) | 0 → 0 (+0) |
| Benevolence | 25 | 1 → 2 (+1) | 2 → 1 (-1) | 1 → 2 (+1) |
| Universalism | 32 | 7 → 7 (+0) | 9 → 9 (+0) | 8 → 6 (-2) |
| **Total** | **204 unique** | **22 → 19 (−3)** | **22 → 21 (−1)** | **21 → 18 (−3)** |

Three Conformity identity changes repeated across all three Runs:
`110fcd4f:conformity:1` and `11de77e8:conformity:3` disappeared in all three
Runs, while `663cb334:conformity:1` appeared in all three. These changes did
not overlap invalid reviews. Conformity counts fell from 4 to 2 and from 3 to
1 in Runs 1 and 3, respectively, but remained **2 to 2 in Run 2** because
`a532dde6:conformity:6` also appeared. Run 2 therefore replaced both original
Conformity detections with different Personas despite an unchanged total for
that Core Value. All relevant Run 2 reviews were valid.

| Run 2 Conformity change | Persona | Supporting Journal Entry indices |
|---|---|---|
| Removed | `110fcd4f` | 1, 2 |
| Removed | `11de77e8` | 3, 4 |
| Added | `663cb334` | 1, 2 |
| Added | `a532dde6` | 6, 7 |

Other values moved in different directions. For example, Run 3 gained two
Power Drifts and one Benevolence Drift while losing Drifts in Hedonism,
Security, Conformity, and Universalism. These are observed count changes;
the Runs do not imply that an individual Drift literally transfers from one
Core Value to another.

### Individual Drift changes

The difference in totals is smaller than the number of changed detections.
For example, Run 2 loses only one Drift overall but contains seven additions,
eight removals, and four overlapping changes. The separate boundary and
termination columns below count one-to-one overlap groups. No splits, merges,
or broader resegmentation occurred in the comparisons between variants.

| Run | Exactly unchanged | Added | Removed | Boundary changes | Termination-only changes | Changed entry decisions / 2,377 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 4 | 7 | 2 | 1 | 305 |
| 2 | 10 | 7 | 8 | 3 | 1 | 308 |
| 3 | 8 | 5 | 8 | 3 | 2 | 319 |

Self-Direction illustrates a boundary detail that aggregate counts also miss:
both variants detect two Drifts in every Run, but the ending decision for
`11de77e8:self_direction:9` changes from Abstain to Not Conflict at Journal
Entry index 11 in all three Runs. Its supporting Conflict indices remain 9
and 10; this is a termination-only change, not an added or removed Drift.

The complete [Core Value count table](../artifacts/twinkl_j3k7_core_value_definitions_20260907/core_value_counts.csv)
includes every value, including zero rows. The
[Drift change table](../artifacts/twinkl_j3k7_core_value_definitions_20260907/drift_identity_changes.csv)
records every added, removed, and overlapping change with its before-and-after
indices and termination fields. Indices are zero-based. Full Drift Records,
evidence quotes, and changed entry decisions are retained in
[results.json](../artifacts/twinkl_j3k7_core_value_definitions_20260907/results.json).

### Variation between Runs

Identities also vary when the prompt is held constant, so each changed pair
cannot automatically be attributed to definitions. The baseline's first two
Runs both detect 22 Drifts, yet only 13 match exactly: six disappear, six
appear, and three have overlapping changes. The definitions variant also
shows repeat variation. These descriptive comparisons are not a significance
test and do not remove sampling uncertainty.

| Variant | Compared Runs | Totals | Exactly unchanged | Added | Removed | Overlapping changes |
|---|---|---|---:|---:|---:|---:|
| Baseline | 1 vs 2 | 22 → 22 | 13 | 6 | 6 | 3 |
| Baseline | 1 vs 3 | 22 → 21 | 15 | 2 | 3 | 4 |
| Baseline | 2 vs 3 | 22 → 21 | 15 | 3 | 4 | 3 |
| Definitions | 1 vs 2 | 19 → 21 | 12 | 6 | 4 | 3 |
| Definitions | 1 vs 3 | 19 → 18 | 11 | 5 | 6 | 2 |
| Definitions | 2 vs 3 | 21 → 18 | 11 | 3 | 6 | 4 |

## What changed

The Weekly Drift Reviewer previously received selected Core Value identifiers
without their explanatory context. Runtime prompt version `4.0` adds only those
Core Values' existing `definition` and `core_motivation` fields from
[`config/schwartz_values.yaml`](../../../config/schwartz_values.yaml).
These are Schwartz-based definitions plus project-specific elaborations,
including motivation prose from the project's Persona-generation
configuration. They should not be described as canonical definitions alone.
The application supplies this context in the trusted instructions, while
Journal Entry text remains separate untrusted input. The change excludes other
generation guidance, behavioral examples, labels, and unselected values.

The comparison uses fresh calls for both the immediately preceding runtime
prompt, version `3.0`, and the definitions-added version `4.0`. Preparation
checks that removing the definitions block recovers the exact baseline
instructions and that both variants have identical Journal Entry input and
response schemas. Model, reasoning effort, Conflict rules, validation, and the
deterministic Drift rule are held constant.

This distinction matters because the published Luna-low development results
used prompt version `2.0`. Its historical detected totals were 29, 27, and 27
across three Runs; the often-cited 42 is the number of reference Drifts in the
AI-reviewed development analysis. Those historical receipts provide context,
but they cannot isolate the effect of adding definitions to the current
runtime. Version `3.0` also introduced stricter confidence, reason, and
exact-quote rules after that earlier study. See the
[Luna-low report](experiment_review_2026-07-14_twinkl_52zz_luna_low.md) and
[evaluation contract](../../../docs/evals/drift_detection_eval.md).

## Protocol and population

The experiment reuses the complete synthetic Weekly Drift development
population: 204 Personas, 951 observed weeks, 1,651 unique Journal Entries,
and 292 Persona/Core Value combinations. Each complete Run produces 2,377
Journal Entry/Core Value decisions. The exact historical student-visible
text was recovered from the existing study's frozen prompts and verified
against the wrangled histories, which also supply dates. Original generator
files are not experimental inputs. The
[baseline snapshot](../artifacts/twinkl_j3k7_core_value_definitions_20260907/baseline_snapshot.json)
preserves all baseline requests, source hashes, response schema, and the
original runtime and prompt source text.

Both variants use `gpt-5.6-luna` with reasoning effort `low`, three independent
Runs each, the Responses API, service tier `default`, `store: false`, a
2,000-output-token cap, and a 60-second timeout. The 5,706 requests are
interleaved in a shuffled schedule with local seed `20260907` and concurrency
eight. This seed determines call ordering; it is not a provider sampling seed.
Run numbers pair the same inputs across variants, without implying shared
randomness between calls. The 16-request execution pilot is retained in the
complete results rather than discarded or counted twice.

Only transient errors may receive a second attempt; SDK retries are disabled.
Invalid or refused output is terminal and becomes an Abstain decision for
every requested coordinate. The executor records each attempt before sending
it and appends the response afterward. On resumption, it skips completed
requests and treats an interrupted attempt with an unknown outcome as an
unavailable review rather than silently paying to repeat it. One process owns
the output directory. The complete request bodies, settings, source hashes,
and comparison rules were frozen before paid execution and were not tuned
against observed responses.

## Counting and comparing Drifts

For each variant and Run, the analysis combines current-week decisions into
one complete chronological history per Persona and calls the maintained
Drift Detector once on that history. Every evaluated Journal Entry appears
exactly once in its current-week requests. Two consecutive Conflicts for the
same Core Value create one Drift; a longer uninterrupted run remains one
Drift. Counts include all confirmed Drifts through each Persona's final
cutoff, including Historical Drift Records that have ended. They are not a
sum of repeated weekly snapshots or a count of only currently Active Drift.

Exact agreement requires the same Persona, Core Value, onset, confirmation,
supporting Journal Entry indices, and termination reason, index, and verdict.
Quotation wording is excluded because it does not change Drift identity.
After removing exact agreements, overlapping spans for the same Persona and
Core Value are grouped into boundary changes, termination changes, splits,
merges, or broader resegmentation. Unmatched spans without a corresponding
overlap are reported as added or removed. This prevents a shifted boundary
from being presented as both an entirely new Drift and a lost Drift.

The results retain every before-and-after Drift, each change group, and all
changed entry decisions. This allows unchanged totals within a Core Value to
be inspected for changes in Personas or evidence, as well as exposing
redistribution between Core Values. Pairwise comparisons between Runs of the
same variant provide a descriptive check of ordinary repeat variation.
The median for each Core Value is calculated separately; these medians need
not add up to the median total, whereas every individual Run's rows must sum
to its total.

## Execution and validation

All **5,706 terminal requests** completed in **5,706 attempts**. The provider
returned `gpt-5.6-luna` for every attempt. There were 5,684 valid reviews and
22 invalid reviews, with no refusals, transport errors, retries, interrupted
attempts, or missing token receipts. The invalid reviews comprise 20
exact-quote failures and two coordinate mismatches; all were retained as
fail-closed Abstain decisions. Invalid counts by Run were 6/2/0 for baseline
and 6/4/4 for definitions.

| Variant | Requests | Valid | Invalid | Input tokens | Output tokens | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2,853 | 2,845 | 8 | 4,406,379 | 648,737 | 3.04 s |
| Definitions | 2,853 | 2,839 | 14 | 4,888,647 | 631,148 | 3.00 s |

The combined receipts record 9,295,026 input tokens and 1,279,885 output
tokens. The interval from the pilot's first attempt to the final completion
was approximately 41 minutes, from `2026-09-07T01:50:22Z` to
`2026-09-07T02:31:22Z`. Token counts are provider usage records, not an invoice
or a monetary cost estimate; detailed cache fields remain in the receipts.

Invalid output affects some observed changes. Definitions-side invalid
reviews overlap the removed `988d1a65:power:5` Drift in Runs 1 and 2, while a
baseline invalid review overlaps the added `e6838e16:security:5` Drift in
Run 2. Those examples cannot be read purely as different semantic judgments.
The Conformity example above is unaffected by invalid reviews.

An independent reconstruction using only the Python standard library grouped
consecutive Conflict indices without importing the runtime detector or the
experiment scorer. It matched all reported counts, full Drift Records,
added/removed identities, overlap groups, and changed entry decisions. It
also verified all 204 Personas, 1,651 Journal Entries, 951 weekly requests per
variant/Run, and 14,262 effective decisions across the six complete Runs.
Incorrectly summing repeated weekly snapshots would have produced baseline
58/61/62 and definitions 51/51/49; the reported totals correctly count each
confirmed Drift once per complete Persona history.

Verification covered 195 unique targeted tests across the Weekly Drift
Reviewer, Coach runtime, Experience, Drift Detector, historical experiment
compatibility, and the new comparison. Ruff passed for all touched Python
files. MyPy passed for the runtime and runner with imported modules excluded
from diagnostics; the ordinary imported check still reports the pre-existing
`prompts/__init__.py:44` `no-any-return` error. The full repository test suite
was not run. A separate implementation review found no concrete correctness
or scope findings. Final checks verified the frozen experiment hashes, all
211 baseline sources outside the two intentionally changed runtime files,
and all added local documentation links. The publication scope comprises
the task's code, tests, documentation, and frozen experiment artifacts.

## Interpretation and limits

This study measures how the combined definition and motivation context changes
detected Drifts. A
higher count does not establish higher recall, and a lower count does not
establish fewer false alerts. Reference labels were not adjudicated or used
to score accuracy in this experiment. The data are synthetic development
fixtures, so these observations do not constitute human validation, real-user
prevalence estimates, a fresh final test, or deployment approval.

Three independent Runs expose some model variation but do not provide a
provider seed or identify a causal effect for each individual changed
decision. Invalid outputs remain part of the observed behavior because the
runtime also fails closed. The experiment does not rerun North Star Moment,
replace its inputs, revise historical evaluation receipts, or amend other
prompt findings. Runtime version `4.0` retains both evaluated fields; the
existing model and deterministic Drift rule remain in place. The study does
not isolate the effect of the motivation prose or establish that it harms
detection.

The adopted follow-up uses the existing v4 **Run 1** outputs for a separate,
targeted North Star Moment evaluation update, matching the original NSM
repeat-1 convention. It does not change the Weekly Drift Reviewer prompt or
rerun Weekly Drift Detection. Both NSM methods must receive the updated
inputs on affected cases; previous evidence is reusable only where its
relevant inputs and evaluation rules are unchanged. That follow-up is not
part of the completed comparison reported here.

## Reproducibility

The frozen artifacts can be checked and scored without model calls:

```sh
source .venv/bin/activate
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.weekly_drift_definitions verify
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.weekly_drift_definitions report
```

Execution used these commands, with the pilot retained by the second command:

```sh
source .venv/bin/activate
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.weekly_drift_definitions run --limit 16
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.weekly_drift_definitions run
```

`run` makes paid calls for incomplete requests and requires provider network
access. It resumes the frozen directory; it does not replace completed
responses. Do not run two executors against that directory simultaneously.
The verifier checks current code against the frozen hashes, so future code
changes require the recorded source versions for an exact reproduction.

- Runner: [`weekly_drift_definitions.py`](../../../scripts/experiments/weekly_drift_definitions.py)
- Manifest: [`manifest.json`](../artifacts/twinkl_j3k7_core_value_definitions_20260907/manifest.json)
- Frozen requests: [`requests.jsonl`](../artifacts/twinkl_j3k7_core_value_definitions_20260907/requests.jsonl)
- Attempt receipts: [`attempts.jsonl`](../artifacts/twinkl_j3k7_core_value_definitions_20260907/attempts.jsonl)
- Effective decisions: [`responses.jsonl`](../artifacts/twinkl_j3k7_core_value_definitions_20260907/responses.jsonl)
