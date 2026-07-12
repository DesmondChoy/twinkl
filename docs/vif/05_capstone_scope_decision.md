# VIF Capstone Scope and Evaluation Decision

**Status:** Adopted on 2026-07-12 under `twinkl-752`.

This document records the detailed Value Identity Function (VIF) scope decision
for the remaining capstone period. The [PRD](../prd.md) remains authoritative for
product intent. The entry-level metric definitions live in
[Value & Identity Modeling Evaluation](../evals/value_modeling_eval.md), and the
event definition and promotion protocol live in
[Alignment and Drift Detection Evaluation](../evals/drift_detection_eval.md).

## Decision

> Twinkl's Critic is primarily a conflict-screening component. Its
> product-critical job is to recover `-1` evidence that supports correctly
> detecting sustained two-entry drift episodes. We maximize episode recall
> subject to a conservative precision/false-alert constraint. Entry-level
> `recall_-1` is the main model-development metric; QWK is retained only as an
> ordinal-health diagnostic.

The ten-value Schwartz ontology and ternary `{-1, 0, +1}` output remain. This
decision changes what the capstone optimizes and claims; it does not adopt a
new model architecture or convert the Critic into a binary classifier.

## Metric Hierarchy

### Model development

- Primary metric: entry-level `recall_-1`, macro-averaged across value
  dimensions for continuity with the existing experiment archive.
- Every candidate must also report `-1` precision, its precision-recall curve,
  predicted-negative rate, calibration, per-dimension results, and seed spread.
- QWK, `+1` recall, minority recall, and circumplex metrics remain diagnostics.
  They show whether a recall gain destroyed useful ordinal structure, but they
  do not outrank `recall_-1`.
- No fixed precision floor is adopted yet. Recall-first development cannot by
  itself support a deployment or promotion claim.

The existing code still selects mainline checkpoints QWK-first. Historical run
rankings therefore remain valid records of the policy used at the time, not the
forward selection policy. Implementing recall-first selection needs a separate,
tested change before another training run is treated as decision evidence.

### Product and deployment evaluation

- Primary future deployment metric: recall of sustained-conflict episodes.
- A deployment operating point must also satisfy a conservative episode-level
  precision or user-facing false-alert constraint.
- The acceptable false-alert tolerance is deliberately deferred until after
  recall-focused candidate development. Precision and false-alert behavior must
  still be measured throughout; they are not ignored.
- Abstention produces no drift claim. Coverage, abstention count, and true
  episodes suppressed by uncertainty must be reported.

Because the current locked promotion surface is unresolved, this is a
development contract only. No scorer, cascade, or detector is promotion-ready.

## Canonical Episode

A v1 sustained-conflict episode occurs when two adjacent journal entries each
clearly show the writer making a behavior or choice against the same declared
core value.

- Only the user's discrete `top_values` set is eligible to trigger drift.
  Existing synthetic personas temporarily use `core_values` as the equivalent
  declared set until `twinkl-1m8` wires the onboarding profile contract.
- Values are evaluated independently; aligned evidence on another value cannot
  cancel a conflict episode.
- A longer uninterrupted negative run is one episode, not repeated alerts.
- A non-negative or uncertain entry breaks the run.
- Later recovery changes Coach wording to recovered or mixed; it does not erase
  the earlier episode.

The runtime is intended to use soft `P(-1)` evidence and uncertainty rather than
requiring two hard `-1` predictions. The exact soft-evidence rule remains an
implementation decision: current experimental scoring thresholds the mean
pair probability, while the product definition requires adequate evidence from
each entry. That mismatch must be resolved before runtime adoption.

## Role of `+1` and QWK

Positive alignment remains useful, but it is non-gating:

- `+1` cannot trigger or cancel a drift episode.
- It may support occasional evidence-based acknowledgment by the Coach.
- It remains part of the ternary output and QWK diagnostic so recall-focused
  development cannot silently collapse the rest of the model.

The capstone no longer treats an aggregate QWK threshold as the product bar.
QWK is retained for historical comparison and ordinal-health monitoring.

## Architecture Study Boundary

The approved follow-up is development-only. Its primary architecture comparison
is the same weekly verifier **with** versus **without** Critic inputs. The MLP
family and entry-level LLM remain baselines. Any uncertain verifier result must
abstain rather than emit a drift claim.

This decision does not adopt an MLP-only, LLM-only, ensemble, cascade, or
verifier architecture. Architecture adoption still requires explicit user
approval after the bounded study. Before crediting the MLP with unique value,
`twinkl-1r3d` must audit the suspected Conformity and Self-direction shortcuts.

## Evidence Behind the Scope Change

- The historical corrected-split default remains `run_019`-`run_021`; extensive
  formulation, selector, label, encoder, and context experiments did not produce
  a broadly superior family.
- Repairing the student-visible Security target materially improved Security,
  showing that target reachability matters more than another generic model
  sweep.
- Soft labels and compact history changed behavior but were not promoted.
- The Hedonism matched hard-set found very low `-1` recall, including confident
  errors, so confidence-only escalation is unsafe.
- The LLM and MLP recover different negative cases, which justifies a bounded
  verifier ablation but not an architecture decision from historical test data.
- The student-visible drift review found only one of five development episodes
  and withheld the promotion score after an unresolved case.

The experiment history and numeric evidence remain in
[`logs/experiments/index.md`](../../logs/experiments/index.md).

## Explicitly Deferred

- the acceptable episode precision or false-alert tolerance;
- the exact core-gated `P(-1)` optimization and aggregation rule;
- recall-first checkpoint-selection implementation;
- the final MLP, LLM, verifier, or hybrid runtime architecture;
- a fresh independently resolved promotion surface;
- production detector and Coach wiring; and
- any conversion from the ternary Critic to a binary conflict model.
