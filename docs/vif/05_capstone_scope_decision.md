# VIF Capstone Scope and Evaluation Decision

**Status:** Scope adopted on 2026-07-12 under `twinkl-752`; staged architecture
adopted on 2026-07-14 under `twinkl-752.2`.

This document records the detailed Value Identity Function (VIF) scope decision
for the remaining capstone period. The [PRD](../prd.md) remains authoritative for
product intent. The metrics for individual Journal Entries live in
[Value & Identity Modeling Evaluation](../evals/value_modeling_eval.md), and the
Drift definition and deployment-approval protocol live in
[Alignment and Drift Detection Evaluation](../evals/drift_detection_eval.md).

## Decision

> Twinkl's VIF Critic is primarily a Conflict-screening component. Its current
> job is to recover `-1` evidence for offline review, candidate mining, and
> retraining. Per-Journal-Entry `recall_-1` is the main model-development
> metric; QWK is retained only as an ordinal-health diagnostic. The approved
> user-facing Drift path does not consume VIF Critic predictions.

The ten-value Schwartz ontology and ternary `{-1, 0, +1}` output remain. The
scope decision changed what the capstone optimizes and claims without converting
the VIF Critic into a binary classifier. `twinkl-752.2` later adopted the staged
architecture recorded below.

## Metric Hierarchy

### Model development

- Primary metric: per-Journal-Entry `recall_-1`, macro-averaged across value
  dimensions for continuity with the existing experiment archive.
- Every VIF Critic checkpoint must also report `-1` precision, its
  precision-recall curve, predicted-negative rate, calibration, per-dimension
  results, and seed spread.
- QWK, `+1` recall, minority recall, and circumplex metrics remain diagnostics.
  They show whether a recall gain destroyed useful ordinal structure, but they
  do not outrank `recall_-1`.
- No fixed precision floor is adopted yet. Recall-first development cannot by
  itself support a deployment-approval claim.

The existing code still selects mainline checkpoints QWK-first. Historical run
rankings therefore remain valid records of the policy used at the time, not the
forward selection policy. Implementing recall-first selection needs a separate,
tested change before another training run is treated as decision evidence.

### Product and deployment evaluation

- Primary future deployment metric: Drift recall.
- A deployment operating point must also satisfy a conservative Drift-level
  precision or user-facing false Drift alert constraint.
- `twinkl-7vam` must define the acceptable false Drift alert tolerance, minimum
  Drift recall, coverage, abstention, stability, and any efficiency requirement
  before a fresh final test is scored.
- Abstention produces no Drift claim. Coverage, abstention count, and true
  Drifts suppressed by abstention or uncertainty must be reported.

Because there is no active fresh final test set, this is a development contract
only. The staged architecture is selected, but no VIF Critic,
candidate-confirmation path, or Drift Detector has deployment approval.

## Canonical Drift

A v1 Drift occurs when two consecutive Journal Entries each clearly show the
writer making a behavior or choice against the same Core Value.

- Only the user's Core Values, stored in the discrete `top_values` set, are
  eligible to trigger Drift.
  Existing synthetic personas temporarily use `core_values` as the equivalent
  Core Values until `twinkl-1m8` wires the onboarding profile contract.
- Values are evaluated independently; aligned evidence on another value cannot
  cancel a Conflict or Drift.
- A longer uninterrupted Conflict run is one Drift, not repeated alerts.
- A non-Conflict or uncertain Journal Entry breaks the run.
- Later recovery changes Weekly Coach wording to recovered or mixed; it does
  not erase the earlier Drift.

The current user-facing path uses Weekly Drift Reviewer decisions without VIF
Critic input. The deterministic Drift Detector then requires two consecutive
Conflicts for the same Core Value. Soft `P(-1)` evidence and uncertainty apply
only to the conditional VIF Critic candidate-selection path. Its exact rule must
preserve adequate evidence from each Journal Entry and remains a downstream
decision under `twinkl-7vam`.

## Role of `+1` and QWK

Positive alignment remains useful, but it is non-gating:

- `+1` cannot trigger or cancel Drift.
- It may support occasional evidence-based acknowledgment by the Weekly Coach.
- It remains part of the ternary output and QWK diagnostic so recall-focused
  development cannot silently collapse the rest of the model.

The capstone no longer treats an aggregate QWK threshold as the product bar.
QWK is retained for historical comparison and ordinal-health monitoring.

## Architecture Study Result and Boundary

The first development-only comparison is complete. The Weekly Drift Reviewer was run
without VIF Critic input and with fixed `run_020` VIF Critic predictions. Adding
those predictions cut median Drift recall from 0.40 to 0.20, removed the median
false Drift alert from 1 to 0, and reduced coverage from 0.756 to 0.732. However,
the recall comparison contained only five episodes, so the difference was one
detected episode. That provisional recommendation is superseded by the larger
`twinkl-752.5` result below.

`twinkl-752.3` then tested whether that `0.40` result was limited by prompt
differences. The aligned Weekly Drift Reviewer repeated complete adjacent Journal
Entry pairs, including week-boundary pairs, added a versioned Core Value rubric,
and returned explicit Drift decisions. Median Drift recall fell to `0.20`, median
false Drift alerts rose to `5`, and neither cross-week reference Drift was
recovered. Journal Entry `recall_-1` improved slightly, but Conflict precision
fell and the extra Conflict decisions formed false Drifts. The tested prompt
differences therefore do not explain the weak Drift result.

`twinkl-752.4` then added a much larger reviewed cohort for future architecture
work. Two separate packet-only Codex lanes and a disagreement-only adjudicator
reviewed 52 legacy-discoverable candidate trajectories plus 52 matched controls
and found 31 Drift episodes across 26 resolved trajectories. Three overlap the
earlier five; adding the two prior episodes missed by candidate mining produces
the 33-episode / 28-Drift-trajectory known-development union. Four reviewed
episodes came from the former final-test split; include them in the primary
development analysis and report provenance subgroups separately. A blind Opus
follow-up resolved the four remaining trajectories without adding Drift, so all
106 union trajectories are resolved. The reviewed Drift labels remain valid
development references even when the MLP saw the Journal Entries during
training, but any VIF Critic scheduler score on those entries is in-sample.
Candidate mining may also miss Drifts absent from both legacy label sources.

`twinkl-752.5` completed the bounded reassessment on the 33-Drift union. Weekly
review without VIF Critic input found a median 9/33 Drifts (`0.273` recall),
versus 7/33 (`0.212`) with raw VIF Critic input. The paired recall delta was
`-0.061`, but its 95% trajectory-bootstrap interval crossed zero
(`[-0.158, 0.033]`), so the earlier raw-input rejection is inconclusive rather
than reversed. Raw input also lowered median coverage by `0.094` and added
three median false Drift alerts.

VIF-Critic-triggered early-plus-weekly review also found 9/33 Drifts. It moved
median delay from 5 to 1 day but added one median false Drift alert and 57
reviewer calls. The recall delta and interval were both zero. The frozen
VIF Critic placements hit 7/19 Drift-relevant opportunities versus a random
median of 1/19, but the scheduling timing benefit disappeared on the
non-training subgroup. This is evidence that the in-sample scores target
relevant development opportunities, not evidence that early review improves
Drift detection.

After this architecture decision, `twinkl-qtwz` reviewed the 186 cases outside
the earlier union and found nine additional Drifts across eight Drift
trajectories. The complete development analysis is now 292/292 resolved cases
with 42 Drifts across 36 Drift trajectories. All nine new Drifts have
historical training provenance. This expands the input for future development
studies; it does not alter the `twinkl-752.5` results or reopen the approved
architecture.

These results led to the explicit architecture decision below. The full study
is recorded in the
[`twinkl-752.1` report](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_752_1_weekly_verifier_ablation.md).
The prompt-alignment result is recorded in the
[`twinkl-752.3` report](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_3_weekly_drift_reviewer_prompt_alignment.md).
The reviewed cohort and union correction are recorded in the
[`twinkl-752.4` report](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md).
The fourth-review labels are recorded in the
[`twinkl-752.5` resolution report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md).
The raw-input and scheduling reassessment is recorded in the
[`twinkl-752.5` reassessment report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md).
The post-decision complete review is recorded in the
[`twinkl-qtwz` report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md).
[`twinkl-1r3d`](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_1r3d_shortcut_audit.md)
completed the prerequisite Conformity and Self-Direction audit: 3,406
single-word removals plus 20 repeated-word or phrase removals across 35
confident-correct active validation cases caused no class flips. This does not
support the tested brittle lexical-shortcut explanations, but it does not prove
construct understanding. The study replayed the same cells against the
`twinkl-754` consensus target. The existing three-annotator human anchor had no
strict overlap with the development set, so it was explicitly unavailable
rather than substituted.

## Evidence Behind the Scope Change

- The historical corrected-split reference remains `run_019`-`run_021`; extensive
  formulation, selector, label, encoder, and context experiments did not produce
  a broadly superior family.
- Repairing the student-visible Security target materially improved Security,
  showing that target reachability matters more than another generic model
  sweep.
- Soft labels and compact history changed behavior but did not receive
  deployment approval.
- The Hedonism matched hard-set found very low `-1` recall, including confident
  errors, so confidence-only escalation is unsafe.
- The Weekly Drift Reviewer and VIF Critic recover different Conflict cases,
  which justifies a bounded comparison but not an architecture decision from
  historical test data.
- The aligned Weekly Drift Reviewer raised Journal Entry Conflict coverage but
  worsened Drift recall, false Drift alerts, and repeat stability. Prompt
  alignment at reasoning effort `none` did not reveal a stronger Weekly Drift
  Reviewer setup.
- The expanded student-visible review found 31 episodes across 26 resolved
  trajectories, including three of the earlier five. The known-development
  union contains 33 episodes across 28 Drift trajectories and is 106/106
  resolved after the blind Opus follow-up. Four reviewed
  episodes retain former-final-test provenance for subgroup reporting, but
  remain in the primary development analysis. This is selection-biased
  AI-reviewed development evidence, not a fresh final test.
- The later complete review adds nine Drifts across eight Drift trajectories,
  producing a 292-case development analysis with 42 Drifts across 36 Drift
  trajectories. It supplied the frozen input for `twinkl-52zz`, not a reason to
  reopen this architecture decision.
- On that complete data, `twinkl-52zz` found median Drift recall of `0.167` for
  `gpt-5.4-mini` and `0.476` for `gpt-5.6-luna`, while median false Drift alerts
  rose from 5 to 13. The user accepted that trade-off and selected Luna at
  reasoning effort `none` as the current development Weekly Drift Reviewer.
  The exact setup remains the frozen baseline for a bounded reasoning-effort
  follow-up. This choice does not change the approved component boundaries or
  grant deployment approval.

The experiment history and numeric evidence remain in
[`logs/experiments/index.md`](../../logs/experiments/index.md).

## Adopted Staged Architecture

The user approved the following architecture under `twinkl-752.2`:

1. **Current user-facing path:** Journal Entries and Core Values go to the
   Weekly Drift Reviewer without VIF Critic input. The deterministic Drift
   Detector declares Drift only after two consecutive Weekly Drift Reviewer
   Conflicts for the same Core Value. Confirmed Drift then flows into the Weekly
   Digest and Weekly Coach.
2. **Required VIF Critic path:** the VIF Critic produces versioned predictions
   and uncertainty for offline comparison, disagreement review, candidate
   mining, error analysis, and retraining. Weekly Drift Reviewer outputs do not
   automatically become LLM-Judge reference labels, and retraining cases cannot
   double as final-test evidence.
3. **Conditional user-facing path:** after predefined development criteria are
   fixed and met, the VIF Critic may propose candidate adjacent Conflict pairs.
   The Weekly Drift Reviewer must confirm them from Journal Entry text without
   seeing VIF Critic predictions. The candidate-selection rule, checkpoint,
   prompt, and criteria must be frozen before a fresh final test. Only a passing
   final test can support deployment approval.

Raw VIF Critic prompt input, direct VIF Critic Drift decisions,
confidence-only fallback, and early-plus-weekly scheduling are not selected.
See [VIF Critic Role in Drift Detection](../architecture/drift_detection.md).

## Explicitly Deferred

- the acceptable Drift precision or false Drift alert tolerance;
- the exact Core Value-gated VIF Critic candidate-selection rule;
- recall-first checkpoint-selection implementation;
- a fresh independently resolved final test set;
- production Drift Detector and Weekly Coach wiring; and
- any conversion from the ternary VIF Critic to a binary Conflict model.
