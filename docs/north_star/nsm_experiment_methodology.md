# North Star Moment experiment methodology

**Latest NSM results to use:** the [targeted Weekly Drift v4 Run 1 update](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md), completed on 7 September 2026, is the current basis for NSM reporting and method comparisons. It supersedes the original v2 repeat-1 results for current reporting; the original record remains historical evidence. These results retain prior observations and remain AI assessments of synthetic histories.

**Status, 7 September 2026:** the original paired AI comparison and its targeted
Weekly Drift v4 Run 1 update are complete across 501 weeks from 105 Personas,
with the unchanged 81/24 development/final Persona partition. Full eligible
history achieved higher Card precision and Opportunity recall than Nomic
top-three retrieval in both partitions. Human review remains deferred.

The [targeted-update report](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md)
and [versioned record](../../logs/experiments/reports/north_star_v4_run1_20260907/nsm_experiment.json)
contain the current results with retained observations. The
[original record](../../logs/experiments/reports/north_star_20260906/nsm_experiment.json)
and [original completed results](#completed-results) remain intact as historical
evidence using Weekly Drift v2 repeat 1.

## Targeted update contract

The follow-up under `twinkl-fz34.14` retains the original 105 Personas, 501
weekly cases, 81/24 Persona partition, and 20-case evaluator consistency sample.
It uses completed Weekly Drift **v4 Run 1** outputs without changing the
Weekly Drift Reviewer's existing `definition` and `core_motivation` fields or
rerunning detection. These fields supply Schwartz-based definitions plus
project-specific elaborations; neither their separate effects nor harm from
motivation prose is established by this NSM comparison.

The persisted impact audit confirms 38 affected weeks across 22 Personas:
29 development weeks and nine final weeks. It compares all effective request,
eligibility, and priority inputs, including state, selected Core Values,
definitions, source windows, ordered Journal Entries, eligible Persona responses,
Profile, and chronology. The 463 unchanged effective contexts retain every
observation and final grade, including failures and unresolved judgments.

Both NSM methods were executed on affected contexts. Shared Luna-xhigh
evaluation, selection priorities, contradiction handling, paired exclusions,
and whole-Persona bootstrap settings remain unchanged. Individual request reuse
requires an identical complete ordered input, method/role/repeat identity,
model settings, prompts, schema, validation and evaluation contracts, and
attempt limit. Source reviews are indivisible batches. Quotation reuse also
requires the identical quotation, complete source, eligible response, and Core
Value context. Retained receipts include their original hash and justification.

The original consolidated record, generation hashes, methodology snapshot,
receipts, and correction history remain unchanged. A separate manifest binds
the update's inputs and current corrected evaluation code. Historical
unnecessary rechecks remain historical protocol deviations; they were not
scheduled as new work. The original 20-case consistency sample contains no
affected week, so all repeat evidence was retained with its distinct identity.

Preparation re-encoded 138 unique eligible documents locally with the original
pinned Nomic configuration and response-inclusive serialization, then recomputed
affected rankings. Ten changed cases had no eligible source batch. The remaining
28 affected cases required 43 new Nomic source requests, one exact-compatible
retained Nomic source request, 44 new full-history source requests, and 44 new
primary reference requests. Quotation and recheck evidence followed the selected
outputs. These batch counts differ from the number of cases and from generation
attempts, which include the permitted retries.

## Targeted update results

The update regraded all 501 cases, recomputing reference opportunities,
priorities, deterministic grades, paired exclusions, and 10,000 whole-Persona
bootstrap resamples per split. It is a **targeted update with retained
observations**, not a wholly fresh independent experiment.

| Partition | Method | Card precision | Opportunity recall |
| --- | --- | ---: | ---: |
| Development | Nomic top three | 181/309 = 58.58% | 181/327 = 55.35% |
| Development | Full eligible history | 244/323 = 75.54% | 244/327 = 74.62% |
| Qualified final | Nomic top three | 49/85 = 57.65% | 49/86 = 56.98% |
| Qualified final | Full eligible history | 70/85 = 82.35% | 70/86 = 81.40% |

Paired differences are Nomic minus full history, in percentage points, with
95% percentile intervals from the original seed `20260906`.

| Partition | Card precision difference [95% interval] | Opportunity recall difference [95% interval] |
| --- | ---: | ---: |
| Development | -16.97 [-23.14, -10.72] | -19.27 [-25.45, -12.72] |
| Qualified final | -24.71 [-36.18, -10.75] | -24.42 [-36.47, -9.88] |

| Partition | Confirmed opportunities | Unresolved opportunities | Paired precision exclusions | Paired recall exclusions |
| --- | ---: | ---: | ---: | ---: |
| Development | 353 | 9 | 31 | 35 |
| Qualified final | 96 | 1 | 10 | 11 |

The updated record uses 2952 retained request receipts and
190 new requests, comprising 202 new generation attempts.
Incremental known API cost is **$0.90766631**, with
5 new attempts lacking usage receipts. This is separate from the
original historical known cost of $10.84758089 and its five unknown-cost
attempts. The report itemizes incremental Nomic, full-history, and shared
evaluation costs, along with retained costs and summed request durations.
All-case runtime cost and latency comparisons mix retained and newly observed
requests; they are not a fresh independent latency benchmark.

Two new Nomic runtime requests exhausted validation attempts and remain
failures; the original three Nomic and one full-history runtime failures remain
retained. Runtime failure does not itself remove a known opportunity from
recall. The report and linked record list reference and quotation failures,
every affected metric exclusion, diagnostics, and undefined bootstrap resamples. Source-decision
consistency remains 341/384 (88.80%), and quotation-acceptance consistency remains
107/109 (98.17%) on the unchanged 20-case sample.

These findings favour full history for the tested synthetic configuration,
without establishing human validity, real-user benefit, Weekly Drift accuracy,
or deployment approval. The original qualified-final exposure limitation
remains: the final histories had prior upstream research use, including the
broader Weekly Drift development comparison. Saved-card export and the final
application walkthrough remain separate work.

Validation passed 441 relevant tests, Ruff, scoped MyPy, and independent
quality review. See the report for exact commands, frozen hashes, failure
provenance, and verification details. The original study below retains its
own design, results, timing observations, and correction history.

## Reset scope

The 6 September study was the first experiment after the NSM reset. It did
not reuse pre-reset results, judgments, or conclusions. The targeted update
above retains eligible observations from that completed post-reset study.

- Removed old NSM reports, saved replay records, documentation findings,
  the dated protocol, stale capstone PDF, and obsolete experiment illustration.
- Preserved source histories, upstream Drift data, reusable code, existing
  configuration inputs, and unrelated work. Git and closed Beads history remain.
  The user has now authorized this experiment's API calls without a spending
  cap; the [execution settings](#execution-settings)
  supersede the old experiment's monetary limits.
- Saved NSM cards await fresh export. Live NSM fails closed until a fresh
  budget is configured; the rest of saved Persona replay remains available.
- Amended `twinkl-fz34`, `twinkl-fz34.7`, and `twinkl-fz34.8`; closed
  housekeeping task `twinkl-fz34.13`.
  Reset checks passed: 21 Python tests, 201 frontend tests, TypeScript,
  210 local links, and unchanged hashes for 901 existing code/source files.

## Dataset curation

### Selected Persona index

Use **35 Personas with known Drift plus 70 without known Drift**, a **2:1
non-drift:drift ratio**. Their histories contain **881 Journal Entries**.
The source corpus has 204 Personas and 1,651 entries; upstream AI review
identifies 42 Drifts across the 35 retained Personas.

- [Retained Personas with known Drift: 35](cohorts/known_drift_personas.md)
- [Selected Personas without known Drift: 70](cohorts/non_drift_personas.md)

The files preserve IDs, names, Core Value order, entry counts, and source
provenance. Use the unique eight-character `persona_id` as the dataset key.
Both original and wrangled histories exist for every selected ID. Cohort
membership does not determine weekly state or prove supportive behavior.

### Why use a 2:1 ratio?

Both 1:1 and 2:1 retain every known-Drift Persona and cover all ten Core Values.
**2:1 doubles sampled memberships per Core Value in the added sample, from five
to ten, for 50% more total histories**, leaving 99 non-drift histories outside
the initial cohort.

| Non-drift:drift | Added Personas | Total Personas | Balanced memberships per value in added sample |
| --- | ---: | ---: | ---: |
| 1:1 | 35 | 70 | 5 |
| **2:1, selected** | **70** | **105** | **10** |
| 3:1 | 105 | 140 | 15 |
| 5:1 | 175 | 210 | Unavailable: only 169 non-drift Personas exist |

The first three allocations were feasible in metadata-only calculations.
This is a practical coverage and workload choice, not a power calculation
or population-prevalence estimate. Per-value findings remain exploratory;
actual cost depends on weekly cases, eligible writing, and experiment variants.

### Stratification calculation and verification

The added sample has **40 single-value and 30 two-value Personas**:
`40 + 30 = 70` unique IDs and `40 + 2 × 30 = 100` memberships. Every Core
Value has exactly ten members. The two-value share is 42.86%, close to the
pool's 43.20%; stored Core Value order is preserved.

| Core Value | Available non-drift | Selected non-drift | Retained Drift | Combined |
| --- | ---: | ---: | ---: | ---: |
| Achievement | 25 | 10 | 0 | 10 |
| Benevolence | 22 | 10 | 3 | 13 |
| Conformity | 21 | 10 | 7 | 17 |
| Hedonism | 28 | 10 | 5 | 15 |
| Power | 32 | 10 | 5 | 15 |
| Security | 31 | 10 | 5 | 15 |
| Self-Direction | 17 | 10 | 7 | 17 |
| Stimulation | 22 | 10 | 2 | 12 |
| Tradition | 23 | 10 | 5 | 15 |
| Universalism | 21 | 10 | 11 | 21 |
| **Memberships** | **242** | **100** | **50** | **150** |

This is balanced sampling over overlapping Core Values. Each Persona appears
once, even when contributing to two quotas. Inclusion probabilities have not
been derived; [classical stratified sampling](https://www150.statcan.gc.ca/n1/edu/power-pouvoir/ch13/prob/5214899-eng.htm)
uses mutually exclusive strata.

Balance applies to the added sample's declared Core Values. The retained
Drift cohort remains uneven and has no Achievement Persona. These counts
are not Drift-by-value counts and do not guarantee balanced weekly states,
supportive actions, or displayed cards.

### Reproduce the selection

The [selection script](../../scripts/experiments/north_star_cohort_selection.py)
uses seed `20260906` and the original metadata to reproduce the 70 added
IDs, verify all ten quotas and the frozen 105-ID cohort, and print the result.
It reads no Journal Entry text, writes no files, and makes no model calls.

From the repository root, with `.venv` activated:

```sh
uv run --no-sync python scripts/experiments/north_star_cohort_selection.py
```

### Development and final evaluation histories

**Frozen split, with the qualified final partition accepted after the audit:**

| Partition | Drift | Non-drift | Total |
| --- | ---: | ---: | ---: |
| Development | 27 | 54 | 81 |
| Final evaluation | 8 | 16 | 24 |

Development contains 391 observed weeks, 696 Journal Entries, and 309 Persona
responses; final evaluation contains 110 weeks, 185 entries, and 91 responses.

Both partitions retain 2:1. The added 16 final IDs were selected from the
frozen 70 using seed `20260906`, binary constrained selection, and the cost
`int(SHA256("20260906:final:" + persona_id)[:12], 16) / 16**12`.
Require 16 unique IDs and 2–3 memberships per Core Value; the remaining 54
give 7–8 development memberships. The consolidated record stores the IDs,
solver settings, hashes, and achieved counts. Value-specific final findings
remain exploratory.

The exposure audit checked 317 distinct text versions across 17 NSM commits,
including deleted reports. Historical retrieval and model-review cases used
the original 27 development Personas or the five saved Personas. No such use
was found for the eight reserved or 70 added histories. However, the earlier
Phase 0 baseline automatically parsed all 35 known-Drift histories and used
their availability and label statistics. Its reserved contribution covered
66 Journal Entries and nine Drift records before the original split was frozen.
The source is `3315325a:logs/experiments/reports/north_star_phase0_20260905/baseline.json`;
the consolidated audit records the other inspected Git sources.

The user accepted retaining these eight in final evaluation with that
limitation disclosed. **Final means held apart from recorded NSM retrieval
and semantic development; it does not mean untouched data.** The audit cannot
rule out unlogged manual use, and the corpus has prior upstream research use.
Keep whole histories together, use development for prompt/configuration choices,
freeze the setup before final evaluation, and report the partitions separately.

Preserve the registry's Core Value order as source provenance. For NSM
priority, use the current confirmed Profile contract's canonical
[`CORE_VALUE_ORDER`](../../src/demo/contracts.py), restricted to the Persona's
declared values. Registry order differs for 28 Personas; historical reviewer
prompt order also differs and must not determine NSM selection priority.
Construct a declared-value benchmark Profile reference without inventing
questionnaire responses; the five saved-demo Profile projections do not cover
every selected Core Value combination.

### Closed-week cases and source eligibility

For each selected Persona, assess North Star Moment separately for every
closed week. A Persona with three closed weeks contributes three evaluation
cases. Each assessment uses that week's per-Core Value Drift states and only
writing available at its cutoff, including eligible earlier writing.

Evaluate each week's outcome: a selected quotation or no card. Three cases
do not necessarily mean three cards or three model calls: Insufficient
Evidence or no eligible writing can produce no-card outcomes without a call.

Use all 501 observed Monday–Sunday weeks, including 87 sparse final weeks;
do not sample weeks to reduce spending. Three additional calendar weeks within
history spans have no Journal Entries and no upstream review. Record them as
unreviewed empty weeks outside the comparison, without inventing Drift states
or reference opportunities. Cohort balance does not imply weekly balance.

Freeze **repeat 1** from the existing Luna-low Weekly Drift Reviewer run for
both variants, matching the saved replay convention. The source prompt records
are `twinkl_52zz_model_comparison_20260714/prompts.jsonl`; responses are
`twinkl_52zz_luna_low_20260714/responses_gpt_5_6_luna_low.jsonl`, both under
`logs/experiments/artifacts/`. All 501 prompt/receipt pairs and 1,255
Journal Entry–Core Value coordinates are present. Applying the current Drift
Detector gives **24 Active Drift, 443 No Active Drift, and 34 Insufficient
Evidence** weeks. Cohort reference labels do not replace these inputs.

| Split | Active Drift | No Active Drift | Insufficient Evidence | Total |
| --- | ---: | ---: | ---: | ---: |
| Development | 13 | 353 | 25 | 391 |
| Final evaluation | 11 | 90 | 9 | 110 |

Keep the two invalid repeat-1 reviews (`742c98d6`, week `2025-10-06`;
`ed67c9cc`, week `2025-02-10`) as failed reviews with Abstain decisions.
Do not replace them with another repeat. These records used prompt **v2.0**,
consistent with the [accepted no-rerun decision](../evals/drift_detection_eval.md).
Current v3.0 validation would reject one additional historical valid receipt
(`621be543`, week `2025-09-29`); retain and disclose its original v2.0 status.
This benchmark therefore uses frozen v2.0 upstream inputs, not newly validated
v3.0 outputs.

The historical receipts record Sunday as `review_at_date`. Preserve that field
as provenance; construct fresh replay cutoffs at the following Monday after
the full week has closed. Record this as a synthetic replay convention, not
an observed historical review timestamp. Freeze each derived Drift Detector
output and cutoff once for both variants.

Follow the [product priority and source rules](north_star_moment.md#2-which-writing-qualifies):

| Weekly state | Eligible writing |
| --- | --- |
| Active Drift | Same-Persona writing available before onset; its Journal Entry index must be earlier and its date no later than onset. |
| No Active Drift | Same-Persona writing available through week end; prefer current-week encouragement, then a historical reminder. |
| Insufficient Evidence | Retain as a no-card control with eligibility metadata. |

Keep empty-source and no-support cases, and record unavailable or unfinalized
inputs. [Evaluation](#evaluation) defines the metrics and reporting rules.

### Journal Entries and immediate synthetic nudge responses

Treat existing synthetic responses as available immediately after their
parent Journal Entry, in the order **entry → nudge → response**. This follows
the [prompt](../../prompts/nudge_response.yaml),
[generator](../../src/nudge/generation.py), and
[journalling workflow](../../notebooks/journalling/journal_nudge.ipynb).
Record the parent date and sequence as synthetic provenance; live responses
still require their actual availability evidence.

Keep entry, nudge, and response text separate. Only the entry or Persona
response can supply an exact quotation; the AI-written nudge is not the
user's evidence. Preserve missing responses and source boundaries. Both
[comparison variants](#controlled-comparisons) use Journal Entries and eligible
Persona nudge responses for selection and review.

The audit found **881 Journal Entries, 542 nudges, and 400 nonempty Persona
responses**. Original and wrangled text, dates, entry order, and Core Values
match, with no parse warnings or missing histories. Preserve the 142 nudges
without a response. Sixty-four Personas have same-day entries, so entry index
and interaction order remain necessary alongside dates.

### Provenance, labels, and validation before freezing

Consolidate each experiment run into one **`nsm_experiment.json`** containing
shared provenance and configuration, all variants' case outputs and reference
judgments, retries/errors, validation results, metrics, and usage/costs.
Keep variants and attempts within the same file.

Reference existing datasets and cohort files by path and hash. Use separate
files only when their format or size makes JSON unsuitable, such as screenshots,
and record their paths and hashes in `nsm_experiment.json`. Derive any reports
from this consolidated record.

- Record source paths/hashes, code revision, seed, selected IDs, split,
  Profile, week/cutoff, upstream decisions, source order, and availability.
- Check unique IDs, disjoint partitions, complete histories, parse warnings,
  unchanged text, matching Core Values, and chronology. Report counts and
  exclusions by cohort, split, weekly state, and source type. Account for
  correlated weeks within each Persona.
- Keep biographies, generation instructions, response modes, offline labels,
  VIF Critic Predictions, and cohort/split metadata out of semantic inputs.
  Use approved Core Value definitions and eligible writing. Generate fresh
  NSM reference judgments later and disclose the AI evaluator and settings;
  these are not human validation.

The preparation audit enumerated weekly metadata and recomputed upstream states
before any fresh NSM semantic review. The runner subsequently froze all 501
executable cases and requests, then generated the judgments and quotations
reported below. The original methodology snapshot and source hashes remain in
the consolidated record alongside the completed analysis.

### Execution settings

Histories contain 2–12 entries, with a median of nine. Local Nomic tokenization
of complete entry-and-response candidates measured a median of 187 tokens,
95th percentile of 358, and maximum of 596 against the pinned tokenizer's
8,192-token limit. The runner froze the response-inclusive serialization before
retrieval and embedded 839 distinct eligible documents without truncation.

The authorized OpenAI count-only audit successfully measured 150 complete-history
Persona–Core Value requests: **1,682–4,095 input tokens**, with a median of
3,063.5 and 95th percentile of 3,794.7. All fit the 16,000-token limit. Two
additional byte-size diagnostic probes also fit. These 152 calls produced
token receipts, not NSM judgments. Their full-history contexts and placeholder
context hashes are sizing probes. Execution separately counted all 2,011 fixed
source-review requests after freezing: 2,010 succeeded at 1,304–4,067 tokens;
one exhausted its two network attempts and remained a failed Nomic case.
Quotation requests were counted before their own calls. All receipts and
prompt/schema hashes are in the consolidated record.

These settings govern the fresh experiment. They preserve the existing
reliability controls while removing its superseded spending limits:

| Setting | Frozen choice |
| --- | --- |
| Spending | No total or per-attempt monetary cap, as authorized by the user. Record usage, costs, latency, and failed attempts. |
| Luna input | At most 16,000 measured tokens per complete request, including instructions and schema; no truncation. Count each final request before generation. |
| Luna output | At most 32,768 tokens per attempt, for both runtime and evaluation. |
| Attempts | At most two total attempts per ordinary request; SDK retries disabled. Retry only transient failures, incomplete responses, or invalid structured output. Refusals are terminal. The contradictory-judgment recheck remains one fresh attempt. |
| Timeout | 180 seconds per Luna-low attempt; 300 seconds per Luna-xhigh attempt. |
| Bootstrap | 10,000 whole-Persona resamples per split, seed `20260906`, 95% percentile intervals. Report undefined resamples. |
| Evaluator consistency | Twenty development cases from distinct Personas; two additional blinded Luna-xhigh assessments per case, in addition to the primary assessment. |
| Human review | Deferred by user decision. This experiment supplies AI evaluation only; evaluator agreement does not establish human validity. |

Select the consistency sample before inspecting NSM outputs: order development
Personas by `SHA256("20260906:repeat:" + persona_id)`, retain the first 20
with a non-Insufficient-Evidence week containing mechanically eligible writing,
and choose one such week per Persona by
`SHA256("20260906:repeat:" + persona_id + ":" + week_start)`.
Repeat the complete reference review and the exact-quotation reviews of any
displayed cards; do not rerun runtime selection. Use distinct repeat IDs to
prevent receipt reuse, and retain failures and disagreements rather than
selecting the most favourable pass. The primary judgments and the stated
adjudication rule determine headline metrics. This sample is a practical
consistency check, not a power calculation.

The [controlled runner](../../scripts/experiments/nsm_experiment.py) constructs
all 501 cases, includes eligible responses in Nomic serialization, and implements
the paired runtime, blinded evaluation, rechecks, repeated assessments, and
consolidated reporting. Its experiment-specific provider adapter applies the
uncapped policy while preserving the reliability controls above. Preparation
freezes serialization before embedding, then records code, source, prompt,
configuration, and request hashes. Execution counts all fixed source-review
requests before generation and each exact-quotation request before its call.
No further spending approval is required within this experiment.

The generation commands used were `prepare`, followed by `run --concurrency 8`.
After completion, two audited corrections to deterministic adjudication were
applied as described below. The completed record accepts verification and
regrading, while `prepare` and `run` reject the changed generation-code hashes.
The original generation code is archived inside the same record.

From the repository root, with `.venv` activated, reproduce validation and scores
without provider calls:

```sh
uv run --no-sync python scripts/experiments/nsm_experiment.py verify
uv run --no-sync python scripts/experiments/nsm_experiment.py report
```

The runner checkpoints into the consolidated record and reuses completed
receipts on resume. It records uncertain interrupted attempts without replaying
them. Runtime latency sums token-count and generation attempt durations, local
selection, and Nomic ranking plus its equally allocated embedding preparation;
batch queueing and experiment-checkpoint writes are excluded. Separate stage
timestamps record experiment wall time. Human review remains deferred.

## Evaluation

A **correct card** is an exact quotation of the Persona's supportive action
that satisfies the Core Value, source eligibility, cutoff, selection priority,
and framing rules.
Showing a card is not automatically a success.

A **reference opportunity** exists when independent review of the full eligible
history identifies a quotation the product could display under its frozen
[eligibility and selection rules](north_star_moment.md#2-which-writing-qualifies).
Active Drift permits support only for the selected Core Value, with no fallback
to another value. No Active Drift prefers current-week support over historical
support, applying Profile and source order within each. Insufficient Evidence
controls have no reference opportunity
and do not enter the recall denominator; their no-card outcomes are correct
omissions.

Judge both variants against **full-history reference priorities**. For example,
in No Active Drift, selecting a historical quotation when suitable current-week
writing exists is an incorrect selection (FP) and a missed opportunity (FN),
even if the quotation is exact and supportive. Retrieval hit rate can still
be positive: it checks for supportive writing in the shortlist, not whether
the candidate preferred under product rules was retained. Sorting the shortlist cannot
recover a preferred candidate that retrieval omitted.

**Hypothetical ten-week example—not experiment results.** This confusion
matrix distinguishes quotation quality from whether a card appeared:

| Reference outcome | Correct card | Incorrect card | No card | Total weeks |
| --- | ---: | ---: | ---: | ---: |
| Eligible supportive quotation exists | 3 | 1 | 2 | 6 |
| Eligible supportive quotation does not exist | 0 | 1 | 3 | 4 |
| **Total** | **3** | **2** | **5** | **10** |

- **TP = 3:** correct cards.
- **FP = 2:** incorrect cards (`1 + 1`).
- **FN = 3:** missed opportunities (`1` incorrect-card week + `2` no-card weeks).

The incorrect card in a week with a suitable quotation counts as both an
incorrect selection (FP) and a missed opportunity (FN): a valid quotation
was available, but the system did not select it. These are selection-error
counts, not mutually exclusive binary classifications of weeks. A card's
appearance alone does not make it a true positive.

Use two headline metrics:

| Metric | Plain-English question | Definition | Example |
| --- | --- | --- | --- |
| **Card precision** | Can we trust the cards we show? | Correct cards ÷ all displayed cards | `3 / 5 = 60%` |
| **Opportunity recall** | When a suitable moment exists, do we find it? | Weeks receiving a correct card ÷ weeks with a reference opportunity under the product rules | `3 / 6 = 50%` |

Keep three secondary diagnostics to explain failures; they are not additional
headline scores:

| Diagnostic | What it checks |
| --- | --- |
| Correct omission | Among confidently assessed weeks without an eligible supportive quotation, the fraction receiving no card. |
| Retrieval hit rate@k | Among opportunity weeks, the fraction whose top-k shortlist contains at least one eligible supportive candidate. Applies only to retrieval variants. |
| Selection-rule correctness | Whether displayed cards follow the Core Value, source-window, framing, and full-history reference priority rules. |

Score the final runtime outcome after the predeclared allowed attempts. A
confirmed reference opportunity ending with no card because of a timeout,
exhausted retries, input-limit rejection, or another runtime failure stays in
the recall denominator and counts as a missed opportunity (FN). It adds no
incorrect-card count (FP). Record the operational failure reason separately in
`nsm_experiment.json`; do not exclude either variant's case from the paired
comparison solely because runtime failed. These failures do not make a known
reference opportunity uncertain.

Reference judgments are **AI assessments, not human validation**. Report
unresolved cases separately, show each metric's numerator and denominator,
and state the counts and reasons for denominator exclusions. Use **N/A** for
zero denominators. Account for repeated weeks within Personas when reporting
uncertainty; do not treat those weeks as independent people.

Store case judgments, selection-error counts, headline results, diagnostics,
and uncertainty in the consolidated **`nsm_experiment.json`**.

## Controlled comparisons

**Core research question:** Can Nomic embedding retrieval provide a shortlist
of supportive actions that preserves **Card precision** and **Opportunity
recall** while reducing runtime cost, or does reviewing the full eligible
history provide quality gains that justify its token use?

This is an exploratory benchmark for method selection. Both approaches use
Luna low for quotation selection; the comparison tests the effect of
embedding-based shortlisting. Full-history review remains subject to the
declared input limits. Neither approach is preferred in advance.

Compare **two variants only**:

1. **Nomic shortlist:** retrieve the top three eligible candidates per Core Value,
   then use Luna low for quotation selection.
2. **Full eligible history:** use Luna low for quotation selection across all
   eligible candidates.

One **retrieval candidate** is a complete Journal Entry together with its
eligible Persona nudge response, embedded and ranked as one unit. Top three
means three distinct entry IDs. Both Luna low and Luna xhigh receive the
complete entry and eligible response, retaining their boundaries and checking
both for same-Core Value Conflict. The exact quotation must come from just one
component. An entry without an eligible response remains one candidate.

**Run Nomic first.** Both variants receive the same Journal Entries and eligible
Persona nudge responses. Use **Luna low** (`gpt-5.6-luna`, reasoning `low`)
with the same runtime settings, semantic rubric, weekly inputs, and product
selection rules; apply the same source order after retrieval. If fewer than
three candidates are eligible, retain all of them. **Luna xhigh** (`gpt-5.6-luna`,
reasoning `xhigh`) evaluates both variants.

Compare both variants on the same cases and report **Nomic minus full-history**
differences in the two headline metrics, runtime cost, and latency. Include
embedding preparation/retrieval overhead in runtime measurements and report
shared Luna xhigh evaluation costs separately. For
[Persona-level bootstrap intervals](https://rsample.tidymodels.org/reference/group_bootstraps.html),
resample whole Personas with replacement within each split, keeping all their
included weeks and both variants together. Recompute the aggregate metrics
and their paired differences in each resample using the frozen execution
settings; report undefined resamples separately.
Store the comparison and intervals in `nsm_experiment.json`.

Use precision and recall as the primary selection criteria. Matching or better
scores for Nomic can support adopting the shortlist, with measured cost and
latency informing the choice. Full-history quality gains can support that
approach. If the metrics favour different variants or uncertainty prevents a
clear comparison, report that trade-off and explain any practical choice.
These results provide evidence for the tested configuration and synthetic
cohort; matching point estimates alone do not establish equivalence.

Luna xhigh first assesses all eligible sources independently of the Nomic
shortlist. Reuse the same reference judgments and opportunity counts for both
variants. Then judge each variant's **exact selected quotation** in its complete
source context, without exposing runtime reasoning, prior judgments, or variant
names.

If source and quotation judgments contradict each other, allow **one fresh
Luna xhigh recheck** of the complete entry, eligible response, and exact
quotation under the same rubric, without earlier judgments or variant names.
A valid, unambiguous recheck supplies the final ruling; a failed or ambiguous
recheck leaves the case unresolved. Apply revised source judgments to the
shared reference and recompute opportunity and priority labels for both
variants, grading their quotations separately. If final rechecks disagree on
a shared source, keep it unresolved. Exclude unresolved cases from the affected
paired metric for both variants using the same case list. Preserve original
judgments, rechecks, final rulings, and exclusion reasons/counts in
`nsm_experiment.json`.

Check evaluator consistency through the repeated assessments specified above
and report agreement and unresolved disagreements. Human review is deferred.
Using one evaluator improves comparability, but does not establish correctness.
AI judges can show systematic
biases, including self-preference ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685));
this motivates checking Luna, rather than establishing a Luna-specific finding.

### Prompts to use

| Purpose | Prompt source |
| --- | --- |
| Nomic query and document text | [Embedding templates](../../scripts/experiments/north_star_phase0.py): `retrieve()` query and `DOCUMENT_TEMPLATE` |
| Luna-low source assessment and independent Luna-xhigh reference review | [Source prompt](../../src/north_star/assessment.py): `SOURCE_SYSTEM_PROMPT` and `build_source_prompt()` |
| Luna-xhigh assessment of the selected quotation | [Quotation prompt](../../src/north_star/assessment.py): `CANDIDATE_SYSTEM_PROMPT` and `build_candidate_prompt()` |

Both Luna prompts include `SHARED_SEMANTIC_RUBRIC`. Freeze adapted prompt text
and settings in **`nsm_experiment.json`**, alongside both variants, repeated
assessments, and the human-review deferral. The controlled runner composes these
helpers without changing their semantic rubric.

## Completed results

Both variants completed all 501 weekly cases with `gpt-5.6-luna` at `low`
reasoning; fresh shared references, exact-quotation reviews, contradiction
rechecks, and consistency repeats used the same model at `xhigh`. The following
results use corrected deterministic adjudication and identical paired exclusion
lists for each affected metric. They are AI assessments of synthetic histories.

| Partition | Variant | Card precision | Opportunity recall |
| --- | --- | ---: | ---: |
| Development: 391 weeks, 81 Personas | Nomic top three | 180/302 = 59.60% | 180/319 = 56.43% |
| Development | Full eligible history | 240/314 = 76.43% | 240/319 = 75.24% |
| Qualified final: 110 weeks, 24 Personas | Nomic top three | 52/86 = 60.47% | 52/87 = 59.77% |
| Qualified final | Full eligible history | 72/86 = 83.72% | 72/87 = 82.76% |

The paired differences below are **Nomic minus full history**, with 95%
percentile intervals from 10,000 whole-Persona resamples within each partition.
Negative precision and recall differences favour full history. None of these
headline, cost, or latency intervals had an undefined resample.

| Paired difference | Development: estimate [95% interval] | Qualified final: estimate [95% interval] |
| --- | ---: | ---: |
| Card precision, percentage points | −16.83 [−22.67, −10.80] | −23.26 [−35.26, −9.30] |
| Opportunity recall, percentage points | −18.81 [−24.92, −12.46] | −22.99 [−35.62, −8.57] |
| Total runtime API cost, USD | −0.33436 [−0.39536, −0.27730] | −0.06575 [−0.08897, −0.04369] |
| Mean runtime latency per week, seconds | −5.37 [−6.27, −4.50] | −3.04 [−3.99, −2.06] |

Full history therefore provides the stronger quality result for this tested
configuration. Nomic's lower cost and latency came with lower precision and
recall in both partitions; the benchmark does not support claiming that its
top-three shortlist preserved quality. It supplies method-selection evidence
without establishing human validity or deployment approval.

### Denominators and diagnostics

Development had 339 confirmed reference opportunities and nine unresolved
opportunities. Final evaluation had 95 confirmed and one unresolved. After
shared-reference and quotation adjudication, the paired exclusions were 25
development cases for precision and 29 for recall; final exclusions were eight
and nine respectively. Precision excludes unresolved displayed-card judgments;
recall excludes unresolved opportunities or card judgments. The remaining
non-opportunity and no-card cases contribute only where the metric definition
applies. Exact case IDs, original judgments, and exclusion reasons are stored.

Nomic displayed 322 development and 94 final cards; full history displayed 338
and 94. Three Nomic runtime failures and one full-history runtime failure all
occurred in development. Each retained a confirmed opportunity and counted as
a missed opportunity, with no incorrect displayed card. Six primary reference
requests failed in development; none failed in final evaluation. The final
uncertainty exclusions therefore also reflect semantic disagreement and
adjudication, rather than only failed requests.

Nomic retrieval hit rate@3 was 329/336 (97.92%) in development and 95/95 (100%)
in final evaluation. This checks whether any supportive source was retained.
Among its scored cards with resolved reference priority, the shortlist omitted
the source preferred under product rules in 90/299 development cases and
27/86 final cases. Selection-rule correctness was 184/302 for Nomic versus
251/313 for full history in development, and 54/86 versus 75/86 in final
evaluation. This distinction explains why high supportive-source retrieval
coverage did not preserve card quality.

Both variants correctly omitted all 14 confidently assessed final cases
without an opportunity. The final correct-omission diagnostic and its paired
difference had 9,999 defined bootstrap resamples and one undefined resample;
all other stored intervals had 10,000 defined resamples. The consolidated
record contains every diagnostic's denominator and interval, plus descriptive
counts and exclusions by cohort, partition, weekly state, and eligible source
components. Of 501 cases, 418 had eligible Journal Entries and responses, 48
had eligible Journal Entries only, and 35 had no semantically eligible writing
(34 Insufficient Evidence controls and one empty-source Active Drift case).
Selected quotation-source counts are reported separately for each variant.

### Cost, consistency, and scoring corrections

Measured runtime API costs were **$0.55685 for Nomic** and **$0.95696 for full
history**, a 41.81% reduction for Nomic across all 501 cases. Mean per-week
latency was 10.84 versus 16.21 seconds in development, and 9.90 versus 12.94
seconds in final evaluation. These timings include Nomic embedding preparation
and retrieval overhead under the allocation defined above. Local CPU compute
was not assigned a dollar cost. Nomic ran first, so provider load and caching
may affect the observed cost and latency differences.

Shared AI evaluation had **$9.33377** in recorded usage costs; combined known
runtime and evaluation costs were **$10.84758** across 3,079 generation attempts.
Five timed-out evaluation attempts returned no usage receipt, so their costs
are unknown and the combined figure is incomplete. All 3,074 received responses
reported `gpt-5.6-luna`; no model substitution occurred.

The 20-Persona consistency sample had raw source-decision pairwise agreement
of 341/384 (88.80%); 21 of 128 source coordinates did not agree across all
three assessments. Exact-quotation acceptance agreement was 107/109 (98.17%),
with one disagreement and one incomplete coordinate among 37 quotations.
One repeat quotation request exhausted both validation attempts. These raw
label comparisons measure evaluator consistency. Derived opportunity, priority,
and card-grade agreement is also stored, but compares the adjudicated primary
pass with unadjudicated repeats and should be read only as a diagnostic.

Independent review identified two deterministic adjudication defects during
primary source review, before quotation review or headline scoring. The
correction plan was recorded before those stages; the frozen generation run
then completed unchanged. The corrected analysis requires an actual primary
source judgment before a contradiction recheck can affect grading, and leaves
a quotation unresolved when its source-based rejection was superseded without
an independent assessment of that quotation. It does not transfer a recheck's
quotation ruling between variants.

Ten unnecessary rechecks across six development cases were preserved as
protocol deviations and their rulings ignored; their $0.01214 cost remains in
the evaluation total. The corrections added two development cases to each
headline metric's paired exclusion list. Final metrics and intervals were
unchanged. The record preserves the original grades, metrics, evaluator and
runner source, immutable generation hashes, correction reasons, and corrected
code hashes, so both analyses remain auditable. No additional recovery calls
or runtime tuning followed the corrections.

Validation passed 394 targeted experiment, North Star Moment, and affected
demo tests, Ruff for the new modules and tests, and scoped MyPy for all four
experiment modules. Independent audits reproduced all original and corrected
grades, all 50 stored intervals, provider attempts and costs, and source hashes.
The full repository suite was not rerun. The accepted qualified-final exposure
limitation above still applies, and human review remains deferred. Saved-card
export and the final capstone walkthrough remain separate work.
