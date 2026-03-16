# Experiment Review — 2026-03-16 — `twinkl-733` semantic counterexample batch de-scope

## 1. Overview

`twinkl-733` was opened to test one narrow question: after the representation
diagnostics reported out, was it still worth generating one more small
`Hedonism` / `Security` counterexample batch targeted at the remaining semantic
polarity failures?

The issue was intentionally scoped as a lightweight data-centric follow-up, not
as a new subsystem. The proposed batch would have paired topical overlap with
flipped polarity so the frontier could be tested on defended rest, quiet
pleasure, and anxious-surface stability language more directly than in the
earlier targeted lifts.

The relevant prerequisites are now complete:

- `twinkl-730` quantified frontier uncertainty and showed the earlier
  `Hedonism` / `Security` lift did not produce a clean family-level promotion
  case.
- `twinkl-731`, `twinkl-732`, and `twinkl-742` tested the remaining
  representation-swap hypotheses and did not overturn the existing diagnosis.
- The March 8-11 review chain repeatedly replayed the same failure mode:
  defended rest or quiet pleasure being read as guilt or misalignment, and
  stability-seeking language being read as fragility, insecurity, or threat.

The question for this issue is therefore not "can the repo generate and judge a
counterexample batch?" but "does the current frontier still need a new batch to
establish the finding?"

## 2. Decision

De-scope `twinkl-733` for the current frontier and close it with documentation
only.

This is not a "no finding" outcome. It is an explicit decision that the finding
`twinkl-733` was meant to test is already established strongly enough by the
existing experiment record, so another generate -> judge -> retrain cycle is
not worth the cost right now.

## 3. Why A New Counterexample Batch Is Not Needed Now

### 3.1 The semantic-polarity failure has already been shown repeatedly

The earlier review chain is already unusually consistent for a small frontier:

- The March 8 replay showed `Hedonism` positives predicted as `-1` when the
  text described protecting rest, decompressing, or quietly enjoying relief,
  while `Security` flips depended on surface risk language rather than the
  underlying stability meaning.
- The March 10 full reviews kept the same read: the model was not failing
  randomly on rare labels, it was repeatedly misreading semantic register.
- The March 11 weighted review again identified the largest misses as defended
  rest and stability-seeking language being read through the wrong polarity.

At this point the repo no longer needs another experiment to prove that the
failure mode exists. The failure mode is already the dominant explanatory thread
across multiple reviews, checkpoints, and intervention types.

### 3.2 The earlier targeted data lifts already tested the data-support direction

The corrected-split frontier has not ignored targeted data. The repo already
ran:

- the targeted hard-dimension batch `run_022`-`run_024`
- the regenerated `Hedonism` / `Security` lift `run_025`-`run_027`

Those runs were informative. They showed that targeted augmentation can move the
tail-sensitive package somewhat, but they did not produce a clean frontier
change on the metrics that decide promotion. In other words, the broad "add
more targeted data" direction has already had a fair test. What remains open is
not whether semantic support matters, but whether one more narrower matched-pair
batch would change the recommendation enough to justify the full synthetic-data
workflow again.

### 3.3 The uncertainty review lowered the value of one more small batch

`twinkl-730` is the key gating result here. It showed that the prior
`Hedonism` / `Security` lift still had:

- unresolved family-level `qwk_mean` delta versus the incumbent
- unresolved family-level `recall_-1` delta versus the incumbent
- hard-dimension QWK that was useful diagnostically but not strong enough by
  itself to drive promotion

That means a new small batch would enter a review regime where point gains alone
are no longer enough. Even if a matched-pair batch nudged `Hedonism` or
`Security`, it would still need to separate from noise cleanly enough to change
the frontier recommendation. The expected information gain from one more small
data-only cycle is therefore lower than it looked before the bootstrap review.

### 3.4 The representation diagnostics did not create a new blocker that this batch must answer

The negative 768d, v2-MoE, and Qwen diagnostics matter here for a different
reason: they remove the argument that `twinkl-733` is needed to break a
representation deadlock.

Across `twinkl-731`, `twinkl-732`, and `twinkl-742`, the repo already learned:

- wider or swapped encoders do not cleanly rescue the hard-dimension problem
- `Security` can improve in isolated branches without producing a better
  frontier overall
- `Hedonism` remains weak even when broader representation quality recovers

So the frontier is not blocked on "one more data point before we know whether
representation is the issue." That question has already been answered well
enough for current roadmap purposes.

### 3.5 A new batch would cost a full synthetic-data cycle, not just a note in the margin

Even a "small" targeted batch in this repo is not free. To keep the comparison
clean it would still require:

- frozen baseline handling
- raw-batch verification
- wrangling and judge labeling
- manual QA with keep / ambiguous / bad outcomes
- at least one frozen-holdout `BalancedSoftmax` retrain
- a new review against the incumbent under the current uncertainty-aware gates

That is a reasonable cost when the intervention could plausibly change the
frontier recommendation. It is not a reasonable cost when the main likely
outcome is merely reconfirming a diagnosis that the repo already documents well.

## 4. Recommendation

Close `twinkl-733` as an explicit de-scope and keep the current frontier
unchanged.

That means:

- `run_019`-`run_021` stay the active corrected-split default.
- Closing `twinkl-733` does **not** mean the model is fixed on `Hedonism` or
  `Security`.
- Closing `twinkl-733` means a new semantic counterexample batch is not worth
  another generation / labeling / retraining cycle for the current frontier.

For roadmap hygiene, `twinkl-734` becomes tracker-unblocked after this closeout,
but it should still be treated as a reserve branch rather than an automatically
recommended next experiment.

## 5. Reopen Conditions

Revisit a semantic counterexample batch only if at least one of the following
becomes true:

1. A future branch needs a tightly matched causal falsification test and the
   team wants one last low-scope data probe before changing model design again.
2. New evidence contradicts the current diagnosis and suggests the historical
   replay pattern was overstated or misattributed.
3. A later intervention still fails on the same hard dimensions and the team
   decides a final matched-pair data test is worth the full synthetic-data
   workflow cost.

Until then, `twinkl-733` should stay closed.
