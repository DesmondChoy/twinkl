# Full Legacy-Discoverable Drift Review (`twinkl-752.4`)

**Date:** 2026-07-13
**Disposition:** Complete cohort review; no architecture or deployment decision

> **2026-07-14 correction:** The full reviewed cohort contains 31 episodes
> across 26 Drift trajectories. Its 27/4 internal split records historical
> evaluation provenance, not label validity or current evaluation status. All
> 31 are development evidence. Three of the earlier five overlap this cohort;
> adding the omitted `3a3b15e4:tradition` and `7adc5866:benevolence` episodes
> produces the 33-episode / 28-Drift-trajectory known-development union for
> `twinkl-752.5`. The frozen `twinkl-752.4` artifacts remain unchanged.

## Result

The review found **31 episodes across 26 resolved Drift trajectories**. Of
these, 28 are net-new relative to the earlier five. Adding the two omitted
earlier episodes produces a known-development union of 33 episodes across 28
Drift trajectories. Four reviewed episodes came from the former final-test
split; they remain in the primary development analysis, with provenance
reported as a subgroup. This is enough to run the bounded comparison in
`twinkl-752.5`; it is still not a fresh final test.

| Stratum | Legacy candidates | Matched controls | Resolved | Drift trajectories | Drift episodes |
|---|---:|---:|---:|---:|---:|
| Development reference | 44 | 44 | 85 / 88 | 23 | 27 |
| Retired audit only | 8 | 8 | 15 / 16 | 3 | 4 |
| **All reviewed** | **52** | **52** | **100 / 104** | **26** | **31** |

Among resolved development-reference cases, 22/43 legacy candidates (51.2%)
and 1/42 matched legacy-negative controls (2.4%) contain Drift. The control is
a legacy-miner miss, not a reviewer false positive.

## What was reviewed

The candidate cohort is the union of every persona/Core-Value trajectory with
an adjacent `-1/-1` pair under either legacy label source. Both entries had to
be negative under the same source; mixed-source pairs did not qualify.

- 52 candidate trajectories across 51 personas and 448 entry/Core-Value cells
- 39 candidates found by both sources
- 11 found only by persisted single-pass labels
- 2 found only by five-pass consensus labels
- 52 controls with no adjacent legacy `-1/-1` pair under either source
- controls hard-matched on Core Value, with no candidate persona, control
  trajectory, or control persona reused
- 104 trajectories and 874 entry/Core-Value review decisions in total

The deterministic assignment produced 46 same-historical-split pairs, minimum
total absolute trajectory-length gap 52, and 27 exact-length matches. The
artifacts name eight former-final-test candidates and their controls
`retired_audit_only`. That name records historical provenance only: all
reviewed cases are now development-only and enter `twinkl-752.5`'s primary
analysis.

The former 24-person `twinkl-v8pb` final-test cohort is now development-only.
The immutable old artifacts remain historical, and `twinkl-pv6s` still owns a
fresh final test.

## Review protocol

Four hash-bound packets omitted persona IDs, dates, cohort roles, historical
splits, legacy labels, VIF Critic outputs, expected outcomes, and author notes.
Two separate packet-only Codex lanes reviewed all 874 entries from the full
displayed runtime text.

- 1,748 paired entry judgments
- 849/874 initial entry agreements (97.1%)
- 25 disagreements across 22 trajectories
- a third packet-only adjudicator saw only full displayed text and anonymized
  prior judgments for those positions
- 21 disagreements resolved; four remain explicitly uncertain
- final resolution: 870/874 entries and 100/104 trajectories

Agreed entries were immutable during adjudication. Maximal consecutive Conflict
runs were derived mechanically: a length-three run is one episode, not two.

## Findings

1. **The five-case reference was too small.** This review contributes 28
   net-new episodes. The known-development union now has 33 episodes, 6.6
   times the earlier five-episode surface.
2. **Legacy labels were useful candidate miners, not current Drift truth.** Only
   half of resolved legacy candidates confirmed as Drift under the displayed-
   behavior rule.
3. **The candidate miner is selective but not exhaustive.** Only one resolved
   matched control contains Drift. Controls sample only part of the legacy-
   negative pool, so other Drifts missed by both sources may remain outside the
   cohort.
4. **Training exposure does not invalidate these labels.** The reviewed Drift
   target was not the MLP training target. However, any scheduler result using
   VIF Critic scores on training-seen Journal Entries is in-sample and must be
   reported as such.
5. **No MLP conclusion follows yet.** This task created reference episodes; it
   did not score an MLP scheduler, select a threshold, or compare review cost.

## Remaining uncertainty

Four trajectories retain one uncertain entry each and therefore have no case-
level Drift outcome:

- development candidate `5943c186:hedonism`
- former-final-test-provenance candidate `799f3751:hedonism`
- development controls `3cfa2ebf:universalism` and `65ed1278:benevolence`

They should remain null unless human review supplies genuinely new evidence.
Excluding them from rate denominators is fail-closed, not selective cleanup.

## Decision and next task

Proceed to `twinkl-752.5`. Compare:

1. end-of-week review only;
2. end-of-week review with raw `run_020` `P(-1)` and uncertainty inputs;
3. MLP-triggered early review plus the end-of-week review, with numeric MLP
   scores hidden from the LLM; and
4. a model-free early-review schedule at the same review-call budget.

Report raw hits and denominators, Drift recall and precision, additional Drift
alerts, detection delay, cross-week recovery, trigger rate, coverage,
abstention, paired uncertainty, and review cost. Use the 33-episode
known-development union for primary analysis, report the former-final-test
provenance subgroup separately as a sensitivity check, and make no
out-of-sample or deployment claim.

The architecture decision remains in `twinkl-752.2` after that comparison. The
user still has to decide whether any measured latency/recall benefit justifies
keeping the MLP, and what false-alert and review-cost tolerance is acceptable.

## Artifacts

- Config: [`config/evals/twinkl_752_4_legacy_drift_review_v1.yaml`](../../../config/evals/twinkl_752_4_legacy_drift_review_v1.yaml)
- Cohort receipt: [`parent_control/cohort_manifest.json`](../artifacts/twinkl_752_4_legacy_drift_review_20260713/parent_control/cohort_manifest.json)
- Pair selection: [`parent_control/selection_pairs.parquet`](../artifacts/twinkl_752_4_legacy_drift_review_20260713/parent_control/selection_pairs.parquet)
- Final entry labels: [`results/entry_target_final.parquet`](../artifacts/twinkl_752_4_legacy_drift_review_20260713/results/entry_target_final.parquet)
- Final case outcomes: [`results/case_outcomes_final.parquet`](../artifacts/twinkl_752_4_legacy_drift_review_20260713/results/case_outcomes_final.parquet)
- Final episodes: [`results/drift_episodes_final.parquet`](../artifacts/twinkl_752_4_legacy_drift_review_20260713/results/drift_episodes_final.parquet)
- Final summary: [`results/summary_final.json`](../artifacts/twinkl_752_4_legacy_drift_review_20260713/results/summary_final.json)
- Final audit: [`results/audit_manifest_final.json`](../artifacts/twinkl_752_4_legacy_drift_review_20260713/results/audit_manifest_final.json)

All rates are selection-biased AI-reviewed development evidence, not human
ground truth or population prevalence.
