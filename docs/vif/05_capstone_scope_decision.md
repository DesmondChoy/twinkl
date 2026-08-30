# VIF Capstone Scope and Evaluation Decision

**Status:** Scope adopted on 2026-07-12 under `twinkl-752`; staged architecture
adopted on 2026-07-14 under `twinkl-752.2`; Weekly Drift Reviewer configuration
and metric hierarchy updated on 2026-07-15 under `twinkl-52zz`; the Weekly Drift
Reviewer model contract was fixed at `gpt-5.6-luna` with reasoning effort `low`
on 2026-07-17; and the optional Weekly Drift Reviewer confirmation path for
cases nominated by VIF Critic Predictions was
removed from the remaining capstone scope on 2026-07-17. The Weekly Drift
Reviewer evaluation and deployment policy was fixed under `twinkl-7vam` on
2026-07-19. On 2026-07-19, the user clarified that the VIF Critic (Offline) is optional
research rather than an essential architecture dependency. The user later
closed the fresh final test as not planned and chose to end the time-boxed
capstone without deployment approval. The user then marked VIF Critic (Offline) training
complete for the time-boxed capstone and closed further VIF Critic (Offline) research as
not planned. On 2026-08-11, Twinkl reviewed `twinkl-ck3w` and retained Luna-low
after Luna-`xhigh` raised both Drift recall and false Drift alerts.

This document records the detailed Value Identity Function (VIF) scope decision
for the remaining capstone period. The [PRD](../prd.md) remains authoritative for
product intent. The metrics for individual Journal Entries live in
[Value & Identity Modeling Evaluation](../evals/value_modeling_eval.md), and the
Drift definition and deployment-approval protocol live in
[Alignment and Drift Detection Evaluation](../evals/drift_detection_eval.md).

## Decision

> Twinkl's VIF Critic (Offline) is a completed capstone research component for
> Conflict screening. Per-Journal-Entry `recall_-1` was the main
> model-development metric, and QWK remains an ordinal-health diagnostic. The
> approved user-facing Drift path does not run the VIF Critic (Offline) or consume VIF
> Critic Predictions.

The ten-value Schwartz ontology and ternary `{-1, 0, +1}` output remain. The
scope decision changed what the capstone optimizes and claims without converting
the VIF Critic (Offline) into a binary classifier. `twinkl-752.2` later adopted the staged
architecture recorded below.

## Metric Hierarchy

### Model development

- Primary metric: per-Journal-Entry `recall_-1`, macro-averaged across value
  dimensions for continuity with the existing experiment archive.
- Every VIF Critic (Offline) checkpoint must also report `-1` precision, its
  precision-recall curve, predicted-negative rate, calibration, per-dimension
  results, and seed spread.
- QWK, `+1` recall, minority recall, and circumplex metrics remain diagnostics.
  They show whether a recall gain destroyed useful ordinal structure, but they
  do not outrank `recall_-1`.
- No fixed precision floor was adopted. Recall-first development cannot by
  itself support a deployment-approval claim.

Mainline training uses the versioned `recall_first_qwk_guarded_v1` policy:
validation QWK must be at least `0.3712`, after which `recall_-1` ranks eligible
checkpoints first. The floor is the repaired-Security family median selected
validation QWK (`0.3912`) minus `0.02`. Historical Run rankings remain valid
records of their original QWK-first policy, which remains available as the
named `qwk_then_recall_guarded` option.

### Weekly Drift Reviewer contract and deployment evaluation

- The model contract is fixed at `gpt-5.6-luna` with reasoning effort `low`.
- Its development selection prioritized Drift recall first and false Drift
  alerts second.
- Coverage and abstention are diagnostic metrics, not selection gates. They
  must still be reported because Abstain produces no Drift claim.
- The three complete Luna-low development Runs provide sufficient evidence to
  freeze the contract. Drift recall was `0.571`, `0.548`, and `0.548`; false
  Drift alerts were 5, 4, and 4 across 256 non-Drift Core Value trajectories.
- Any future fresh final test should reuse the Luna-low response schema,
  fail-closed request handling, one-to-one Drift scoring with a two-Entry
  confirmation allowance, three-Run protocol, and reported metrics. No
  separate efficiency gate is adopted.
- The frozen policy would withhold deployment approval if final-test references
  were unresolved, a Run were incomplete, any Run had Drift recall below
  `0.50`, or any Run exceeded a `2%` false-alert burden on resolved non-Drift
  Core Value trajectories.

The model choice and development evidence are settled. No development rerun is
required for the cleaned prompt. The capstone ends without a fresh final test,
so the cleaned prompt has no final-test score and the Drift Detector has no
deployment approval. Weekly Drift Reviewer confirmation of cases nominated by
VIF Critic Predictions is outside the remaining capstone scope.

## Canonical Drift

A v1 Drift occurs when two consecutive Journal Entries each clearly show the
writer making a behavior or choice against the same Core Value.

- Only the user's Core Values, stored in the discrete `top_values` set, are
  eligible to trigger Drift.
  The runtime imports them from a confirmed onboarding Profile when supplied;
  existing synthetic personas retain `core_values` as a compatibility path.
- Values are evaluated independently; aligned evidence on another value cannot
  cancel a Conflict or Drift.
- A longer uninterrupted Conflict run is one Drift, not repeated alerts.
- A Not Conflict decision ends the active Conflict run. An Abstain decision can
  produce Insufficient Evidence when it blocks a current claim.
- A later Not Conflict decision does not erase the earlier Drift. The Drift
  Detector keeps that Drift in a Historical Drift Record and can set the current
  state to No Active Drift. This state does not prove improvement.

The current user-facing path uses decisions from `gpt-5.6-luna` at reasoning
effort `low`, without VIF Critic input. The deterministic Drift Detector then
requires two consecutive Conflicts for the same Core Value. VIF Critic (Offline)
probabilities and uncertainty remain offline research outputs, not inputs to
the user-facing Drift path.

## Role of `+1` and QWK

`+1` remains useful in the offline ternary classification task, but it is
non-gating:

- `+1` cannot trigger or cancel Drift.
- It does not enter Weekly Drift Detection or the Coach Digest.
- It remains part of the ternary output and QWK diagnostic so recall-focused
  development cannot silently collapse the rest of the model.

The capstone no longer treats an aggregate QWK threshold as the product bar.
QWK is retained for historical comparison and ordinal-health monitoring.

## Architecture Study Result and Boundary

The first development-only comparison is complete. The Weekly Drift Reviewer was run
without VIF Critic input and with fixed `run_020` VIF Critic Predictions. Adding
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

`twinkl-752.4` reviewed a much larger cohort for future architecture work. Two
separate packet-only Codex lanes and a disagreement-only adjudicator reviewed
52 Persona/Core Value trajectories selected because the persisted single-pass
LLM-Judge VIF Labels or five-pass consensus LLM-Judge VIF Labels contained an
adjacent `-1/-1` pair. They also reviewed 52 controls hard-matched on Core Value
and trajectory length for which neither LLM-Judge VIF Label source contained an
adjacent `-1/-1` pair.
They found 31 Drifts across 26 resolved Drift trajectories. Three overlap the
earlier five; adding the two prior Drifts that neither LLM-Judge VIF Label
selection found produces the 33-Drift / 28-Drift-trajectory known-development
union. Four
reviewed Drifts came from the former final-test split; include them in the primary
development analysis and report provenance subgroups separately. A blind Opus
follow-up resolved the four remaining trajectories without adding Drift, so all
106 union trajectories are resolved. The reviewed Drifts remain valid
development references even when the MLP saw the Journal Entries during
training, but any VIF Critic Prediction used to schedule review on those
entries is in-sample.
Selection through adjacent pairs in the persisted single-pass LLM-Judge VIF
Labels and five-pass consensus LLM-Judge VIF Labels may also miss Drifts absent
from both sources.

`twinkl-752.5` completed the bounded reassessment on the 33-Drift union. The
Weekly Drift Reviewer without VIF Critic Predictions found a median 9/33 Drifts
(`0.273` recall), versus 7/33 (`0.212`) with raw VIF Critic Predictions. The
paired recall delta was
`-0.061`, but its 95% trajectory-bootstrap interval crossed zero
(`[-0.158, 0.033]`), so the earlier rejection of raw VIF Critic Predictions is
inconclusive rather than reversed. Raw VIF Critic Predictions also lowered
median coverage by `0.094` and added three median false Drift alerts.

Early-plus-weekly review triggered by VIF Critic Predictions also found 9/33
Drifts. It moved median delay from 5 to 1 day but added one median false Drift
alert and 57 Weekly Drift Reviewer calls. The recall delta and interval were
both zero. The frozen VIF Critic Prediction triggers occurred at 7/19
Drift-relevant opportunities versus a random
median of 1/19, but the scheduling timing benefit disappeared on the
non-training subgroup. This is evidence that the in-sample VIF Critic
Predictions identify relevant development opportunities, not evidence that
early review improves Drift detection.

After this architecture decision, `twinkl-qtwz` reviewed the 186 cases outside
the earlier union and found nine additional Drifts across eight Drift
trajectories. The complete development analysis contains 292/292 resolved cases
with 42 Drifts across 36 Drift trajectories. All nine Drifts from the expanded cohort have
historical training provenance. This completed the historical development
record; it does not alter the `twinkl-752.5` results or reopen the approved
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
The raw VIF Critic Prediction and scheduling reassessment is recorded in the
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
- The Weekly Drift Reviewer and VIF Critic (Offline) recover different Conflict cases,
  which justified the completed bounded comparison but not an architecture
  decision from historical test data.
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
  trajectories. It supplied the frozen input for `twinkl-52zz` and completed
  the historical development record.
- On that complete data, `twinkl-52zz` found median Drift recall of `0.167` for
  `gpt-5.4-mini` and `0.476` for `gpt-5.6-luna`, while median false Drift alerts
  rose from 5 to 13. The user accepted that trade-off and selected Luna at
  reasoning effort `none` for the next development comparison. The
  reasoning-effort-`low` follow-up raised median Drift recall to `0.548` and cut
  false Drift alerts to 4, while coverage fell from `0.777` to `0.637`. `low`
  mechanically failed the preregistered coverage gate, but the approved metric
  hierarchy treats coverage as diagnostic. The comparison selected Luna at
  reasoning effort `low`, and the model contract is fixed on that setup.
  `twinkl-ck3w` later compared Luna `medium`, `high`, and `xhigh` on
  the same development data. Luna-`xhigh` raised median Drift recall to `0.667`
  and raised median false Drift alerts to 9. Luna-low remains the fixed contract
  because `xhigh` is a more aggressive operating point, not a clean improvement.
  This no-change decision does not change the approved component boundaries,
  validate the fixed setup on a fresh final test, or grant deployment approval.

The experiment history and numeric evidence remain in
[`logs/experiments/index.md`](../../logs/experiments/index.md).

## Adopted Staged Architecture

The user approved the following architecture under `twinkl-752.2`:

1. **Approved user-facing path:** Journal Entries and Core Values go to the
   fixed `gpt-5.6-luna` reasoning-effort-`low` Weekly Drift Reviewer without VIF
   Critic input. The deterministic Drift Detector declares Drift only after two
   consecutive Weekly Drift Reviewer Conflicts for the same Core Value.
   Weekly Drift Detection stores confirmed Drift in its structured output. The
   Coach Digest consumes that output.
2. **Completed VIF Critic (Offline) research:** training, evaluation, raw output export,
   and timeline inference remain available for offline reproduction. A
   generalized review-and-retrain loop is not implemented or planned for the
   time-boxed capstone. Weekly Drift Reviewer Decisions must not automatically
   become LLM-Judge VIF Labels.
3. **Out-of-scope idea:** Weekly Drift Reviewer confirmation of cases nominated
   by VIF Critic Predictions is not part of the remaining capstone work.
   Revisiting it requires a new scope decision and a fresh evaluation that
   keeps VIF Critic Predictions hidden from the Weekly Drift Reviewer.

Raw VIF Critic prompt input, direct VIF Critic (Offline) Drift decisions,
confidence-only fallback, and early-plus-weekly scheduling are not selected.
See [VIF Critic (Offline) Role in Drift Detection](../architecture/drift_detection.md).

## Remaining Work and Out-of-Scope Ideas

The shared React app synchronizes the confirmed Profile and browser-held
Experience state with the in-memory Python boundary. The batch runtime can
import Core Values from a confirmed Profile JSON. Durable multi-user storage,
a fresh independently resolved final test, deployment approval, and automatic
Profile evolution are outside the time-boxed capstone scope.

The VIF Critic (Offline) review-and-retrain demonstration, data-scaling curve,
Weekly Drift Reviewer confirmation of cases nominated by VIF Critic
Predictions, and conversion to a binary Conflict model are not planned for
the time-boxed capstone.
