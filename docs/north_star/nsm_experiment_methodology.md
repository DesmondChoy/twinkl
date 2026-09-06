# North Star Moment experiment methodology

**Status, 6 September 2026:** Housekeeping complete. The 105-Persona cohort
is selected and verified; weekly cases, NSM judgments, and experiments remain
pending.

## Reset scope

This is the first experiment in the fresh NSM record. Previous results,
judgments, and conclusions will not be reused.

- Removed old NSM reports, saved replay records, documentation findings,
  the dated protocol, stale capstone PDF, and obsolete experiment illustration.
- Preserved source histories, upstream Drift data, reusable code, existing
  configuration inputs, and unrelated work. Old configurations do not authorize
  new experiment settings or spending. Git and closed Beads history remain.
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

**Proposed split, subject to the NSM exposure audit; added IDs remain unassigned:**

| Partition | Drift | Non-drift | Total |
| --- | ---: | ---: | ---: |
| Development | 27 | 54 | 81 |
| Final evaluation | 8 | 16 | 24 |

This targets 2:1 in both partitions. The existing eight reserved histories
remain in the cohort, but enter final evaluation only if they pass the prior
NSM exposure audit. A feasible split gives 7–8 non-drift memberships per value
in development and 2–3 in final evaluation; the latter supports only
exploratory value-specific findings.

Keep whole Persona histories together. Exposed histories belong in development;
if any reserved history fails the audit, revisit the proposed split rather
than force the eight-history allocation. Use development for
prompt/configuration choices and freeze the setup before final evaluation.
Report the partitions separately. “Unseen” refers to NSM development, not
prior upstream research or a new human sample. **Decision pending:** settle
the split after the audit and assign the 70 additional IDs.

### Closed-week cases and source eligibility

For each selected Persona, assess North Star Moment separately for every
closed week. A Persona with three closed weeks contributes three evaluation
cases. Each assessment uses that week's per-Core Value Drift states and only
writing available at its cutoff, including eligible earlier writing.

Evaluate each week's outcome: a selected quotation or no card. Three cases
do not necessarily mean three cards or three model calls: Insufficient
Evidence or no eligible writing can produce no-card outcomes without a call.

Include finalized Monday–Sunday weeks, including sparse final weeks, and
record each cutoff. Any budget-driven week sampling must be specified before
selection. Cohort balance does not imply weekly balance.

Freeze the same upstream Weekly Drift Reviewer Decisions and Drift Detector
outputs across NSM variants. Cohort reference labels cannot replace them.
**Preparation pending:** identify their source and any missing coverage.

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

Before constructing weekly cases, settle the split and complete the remaining
experiment choices and budget. No weekly cases or NSM judgments
have been generated.

## Evaluation

A **correct card** is an exact quotation of the Persona's supportive action
that satisfies the Core Value, source eligibility, cutoff, and framing rules.
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
| Selection-rule correctness | Whether displayed cards follow the Core Value, source-window, and framing rules. |

Reference judgments are **AI assessments, not human validation**. Report
unresolved cases separately, show each metric's numerator and denominator,
and state the counts and reasons for denominator exclusions. Use **N/A** for
zero denominators. Account for repeated weeks within Personas when reporting
uncertainty; do not treat those weeks as independent people.

Store case judgments, selection-error counts, headline results, diagnostics,
and uncertainty in the consolidated **`nsm_experiment.json`**.

## Controlled comparisons

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

Luna xhigh first assesses all eligible sources independently of the Nomic
shortlist. Reuse the same reference judgments and opportunity counts for both
variants. Then judge each variant's **exact selected quotation** in its complete
source context, without exposing runtime reasoning, prior judgments, or variant
names.

Check evaluator consistency through repeated assessments and a small
independently human-reviewed sample; specify their sizes before execution and
report agreement and unresolved disagreements. Using one evaluator improves
comparability, but does not establish correctness. AI judges can show systematic
biases, including self-preference ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685));
this motivates checking Luna, rather than establishing a Luna-specific finding.

### Prompts to use

| Purpose | Prompt source |
| --- | --- |
| Nomic query and document text | [Embedding templates](../../scripts/experiments/north_star_phase0.py): `retrieve()` query and `DOCUMENT_TEMPLATE` |
| Luna-low source assessment and independent Luna-xhigh reference review | [Source prompt](../../src/north_star/assessment.py): `SOURCE_SYSTEM_PROMPT` and `build_source_prompt()` |
| Luna-xhigh assessment of the selected quotation | [Quotation prompt](../../src/north_star/assessment.py): `CANDIDATE_SYSTEM_PROMPT` and `build_candidate_prompt()` |

Both Luna prompts include `SHARED_SEMANTIC_RUBRIC`. Freeze prompt text and
settings in **`nsm_experiment.json`**, alongside both variants, repeat
assessments, and human-review comparisons. **Preparation pending:** adapt the
Nomic template to include eligible Persona nudge responses while preserving
entry/response boundaries, and prepare the controlled runner; the linked code
does not yet implement this complete comparison.
