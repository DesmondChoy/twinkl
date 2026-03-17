# Experiment Review — 2026-03-11 — `twinkl-691.5` circumplex-aware batch sampler de-scope

## 1. Overview

`twinkl-691.5` was the last open step in the post-`twinkl-681` circumplex
rollout. Its brief was intentionally narrow: either run one controlled
circumplex-aware batch sampler ablation, or explicitly de-scope the idea if the
earlier rollout steps showed it was not worth the added complexity.

The relevant prerequisites are now complete:

- `twinkl-691.3` reran the frontier after the frozen-holdout
  `Hedonism`/`Security` lift and kept `run_019`-`run_021` as the incumbent
  default.
- `twinkl-691.4` showed that a soft circumplex prior can improve structural
  diagnostics, but at a meaningful cost to `recall_-1`, minority recall, and
  hedging.
- `twinkl-715` fixed checkpoint-selection hygiene and confirmed that even the
  guardrailed regularized family still did not beat the incumbent frontier on
  the tail-sensitive metrics that matter operationally.

The question for this issue is therefore not "can a sampler be implemented?"
but "should it be the next experiment now that the lower-risk circumplex steps
have reported out?"

## 2. Decision

De-scope the circumplex-aware batch sampler for the current rollout.

This is an explicit keep/drop decision, not a punt. The sampler remains a
theoretically plausible future lever, but the March 9-10 evidence says it is
not the best next use of effort on the corrected-split frontier.

## 3. Why The Sampler Is Not The Next Lever

### 3.1 The current evidence points to hard-dimension polarity, not missing batch structure

The active corrected-split default is still `run_019`-`run_021`
`BalancedSoftmax`, with family-median holdout metrics:

- `qwk_mean = 0.362`
- `recall_-1 = 0.313`
- `minority_recall_mean = 0.448`
- `hedging_mean = 0.621`
- `calibration_global = 0.713`

`twinkl-691.3` did not displace that baseline. The regenerated
`Hedonism`/`Security` lift improved `hedonism` slightly, but the new family
still regressed on `security` and circumplex structure relative to the
incumbent. That already suggested the main bottleneck was not simply a lack of
more conflict-pair exposure.

### 3.2 The regularizer answered the structural question without rescuing the tail

`twinkl-691.4` gave the sampler idea a useful proxy test: if a soft structural
prior improves circumplex behavior cleanly, a more explicit structure-aware
training intervention might be justified next.

That is not what happened. The regularized `BalancedSoftmax` family improved the
aggregate structure metrics, but the operational trade-off was wrong:

- `opposite_violation_mean` improved from `0.082` to `0.039`
- `adjacent_support_mean` recovered from `0.072` to `0.077`
- but `recall_-1` fell from `0.328` to `0.265`
- and `minority_recall_mean` fell from `0.442` to `0.411`

So the structural intervention helped the prior more than it helped the actual
hard-dimension behavior we needed to rescue.

### 3.3 The selection-policy fix did not change the frontier recommendation

`twinkl-715` fixed the checkpoint-selection bug that had allowed low-recall
epochs to be promoted. That was the right cleanup to do before entertaining any
further structure-aware training changes.

The guarded rerun did improve family-level holdout QWK to `0.366`, but it still
landed materially below the incumbent on the tail-sensitive metrics:

- `recall_-1 = 0.267` vs incumbent `0.313`
- `minority_recall_mean = 0.409` vs incumbent `0.448`
- `hedging_mean = 0.641` vs incumbent `0.621`

That means the regularized branch was not merely being held back by selection
hygiene. The family itself still was not a better frontier recommendation.

### 3.4 A sampler would be higher-risk than the interventions now recommended

In the current stack, a sampler would not be a free toggle:

- the active ordinal notebook path builds train loaders with plain
  `shuffle=True`
- there is no existing sampler abstraction or experiment logger metadata for
  sampler behavior
- `BalancedSoftmax` already uses corrected train-split class priors during loss
  computation

So a batch sampler would change the effective minibatch distribution on top of
an already prior-aware loss. That makes attribution harder and can muddy
frontier comparability unless exposure statistics are logged carefully.

By contrast, the current frontier review already points to lower-risk next
experiments that target the actual failure mode more directly:

- per-dimension weighting on `BalancedSoftmax`
- validation-only logit retargeting from `run_020`
- head-only retraining only if those still miss

## 4. Recommendation

Close `twinkl-691.5` as an explicit de-scope.

The circumplex rollout still delivered useful value:

- diagnostics are now first-class and reproducible
- the targeted data lift clarified where the hard dimensions still fail
- the regularizer established the structure-vs-tail trade-off
- the guardrail fixed a real checkpoint-selection problem

What it did **not** establish is that stronger structure-aware training
interventions should come next. The better handoff is to continue with the
lower-risk frontier work already recommended in the latest full review:
per-dimension weighting first, then validation-only `BalancedSoftmax`
logit retargeting, then head-only retraining only if those still fail.

## 5. Reopen Conditions

Revisit a circumplex-aware sampler only if all of the following become true:

1. Per-dimension weighting and post-hoc retargeting fail to produce enough
   `recall_-1` / hard-dimension lift.
2. Circumplex diagnostics in the newer frontier still show unresolved
   theory-violating structure errors after those lower-risk interventions.
3. The sampler ablation is instrumented to log effective exposure/prior
   distortion so it can be compared fairly against the current frontier.

Until then, the sampler should stay off the active roadmap.
