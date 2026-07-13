# VIF Capstone Scope and Evaluation Decision

**Status:** Adopted on 2026-07-12 under `twinkl-752`.

This document records the detailed Value Identity Function (VIF) scope decision
for the remaining capstone period. The [PRD](../prd.md) remains authoritative for
product intent. The metrics for individual Journal Entries live in
[Value & Identity Modeling Evaluation](../evals/value_modeling_eval.md), and the
Drift definition and deployment-approval protocol live in
[Alignment and Drift Detection Evaluation](../evals/drift_detection_eval.md).

## Decision

> Twinkl's VIF Critic is primarily a Conflict-screening component. Its
> product-critical job is to recover `-1` evidence that supports correctly
> detecting Drift. We maximize Drift recall subject to a conservative
> precision/false Drift alert constraint. Per-Journal-Entry
> `recall_-1` is the main model-development metric; QWK is retained only as an
> ordinal-health diagnostic.

The ten-value Schwartz ontology and ternary `{-1, 0, +1}` output remain. This
decision changes what the capstone optimizes and claims; it does not adopt a
new model architecture or convert the VIF Critic into a binary classifier.

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
- The acceptable false Drift alert tolerance is deliberately deferred until
  after recall-focused VIF Critic development. Precision and false Drift alert
  behavior must still be measured throughout; they are not ignored.
- Abstention produces no Drift claim. Coverage, abstention count, and true
  Drifts suppressed by uncertainty must be reported.

Because the current final test set is unresolved, this is a development
contract only. No VIF Critic, cascade, or Drift Detector has deployment approval.

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

The runtime is intended to use soft `P(-1)` evidence and uncertainty rather than
requiring two hard `-1` predictions. The exact soft-evidence rule remains an
implementation decision: current experimental scoring thresholds the mean
pair probability, while the product definition requires adequate evidence from
each Journal Entry. That mismatch must be resolved before runtime adoption.

## Role of `+1` and QWK

Positive alignment remains useful, but it is non-gating:

- `+1` cannot trigger or cancel Drift.
- It may support occasional evidence-based acknowledgment by the Weekly Coach.
- It remains part of the ternary output and QWK diagnostic so recall-focused
  development cannot silently collapse the rest of the model.

The capstone no longer treats an aggregate QWK threshold as the product bar.
QWK is retained for historical comparison and ordinal-health monitoring.

## Architecture Study Result and Boundary

The development-only comparison is complete. The Weekly Drift Reviewer was run
without VIF Critic input and with fixed `run_020` VIF Critic predictions. Adding
those predictions cut median Drift recall from 0.40 to 0.20, removed the median
false Drift alert from 1 to 0, and reduced coverage from 0.756 to 0.732. Under
the registered recall-first rule, the result is negative. The conditional
recommendation for `twinkl-752.2` is the Weekly Drift Reviewer without VIF Critic
input rather than the tested raw-prediction setup.

This decision does not adopt a VIF Critic-only, Weekly Drift Reviewer-only,
ensemble, or cascade architecture. Architecture adoption still requires
explicit user approval. The full study is recorded in the
[`twinkl-752.1` report](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_752_1_weekly_verifier_ablation.md).
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

- The historical corrected-split default remains `run_019`-`run_021`; extensive
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
- The student-visible Drift review found only one of five development-set
  Drifts and withheld the final test score after an unresolved case.

The experiment history and numeric evidence remain in
[`logs/experiments/index.md`](../../logs/experiments/index.md).

## Explicitly Deferred

- the acceptable Drift precision or false Drift alert tolerance;
- the exact Core Value-gated `P(-1)` optimization and aggregation rule;
- recall-first checkpoint-selection implementation;
- the final VIF Critic-only, Weekly Drift Reviewer-only, or hybrid runtime
  architecture;
- a fresh independently resolved final test set;
- production Drift Detector and Weekly Coach wiring; and
- any conversion from the ternary VIF Critic to a binary Conflict model.
