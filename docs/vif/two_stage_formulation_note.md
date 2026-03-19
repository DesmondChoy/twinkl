# Two-Stage VIF Formulation Note

This note records the exact v1 design for `twinkl-746` before the experiment
run. The goal is to test task reformulation, not to stack several other
training interventions on top of it.

## Motivation

The current 3-class formulation asks one head to solve two different questions
at once:

1. Is this value dimension active in the entry at all?
2. If it is active, is the behavior aligned or misaligned?

In the current data, `0` often behaves like "inactive / insufficient evidence"
more than a true midpoint. That makes the head fight both a dominant inactive
class and an active polarity decision at the same time.

## Chosen Formulation

We use one shared MLP backbone and two binary heads per Schwartz dimension:

- Stage A: `inactive` vs `active`
- Stage B: `misaligned` vs `aligned`, trained only on active labels

The model still exports the same final `{-1, 0, +1}` contract by reconstructing
class probabilities as:

- `p(-1) = p(active) * p(misaligned | active)`
- `p(0) = p(inactive)`
- `p(+1) = p(active) * p(aligned | active)`

This keeps the external evaluation and artifact format compatible with the
current frontier.

## Loss and Train-Split Statistics

The v1 loss keeps the same long-tail philosophy as the current frontier:

- Stage A uses binary Balanced Softmax with train-split priors for
  `[inactive, active]`
- Stage B uses binary Balanced Softmax with active-only priors for
  `[misaligned, aligned]`
- Both stages are weighted equally when both are present
- If a batch has no active examples, the activation loss is used on its own

Deliberately out of scope for this first pass:

- per-dimension weighting
- circumplex regularization
- separate activity/polarity models
- alternative checkpoint or scheduler policies

## Added Metrics

In addition to the existing frontier metrics, the eval path now records:

- activation precision / recall / F1
- active-subset sign accuracy

These are derived from final 3-class probabilities for every ordinal family, so
the comparison stays fair between the new two-stage model and the older
single-head families.

## Training-Signal Follow-Up Rule

The main 3-seed comparison keeps the current scheduler and early-stopping logic
unchanged so the primary intervention is the formulation itself.

After the main runs, compare the selected operational epoch against the
minimum-loss epoch using `selection_trace.parquet`. Run one extra seed-22
follow-up only if any seed shows a non-selected epoch with either:

- at least `+0.02` `qwk_mean`, or
- at least `+0.03` `recall_-1`

relative to the selected epoch.

If that condition is met, rerun seed 22 with scheduler and early-stop patience
stretched beyond the training horizon and report whether the conclusion changes.
