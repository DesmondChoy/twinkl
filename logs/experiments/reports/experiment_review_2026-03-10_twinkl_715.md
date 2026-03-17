# Experiment Review — 2026-03-10 — `twinkl-715` hard `recall_-1` checkpoint guardrail

## 1. Overview

`twinkl-715` added a hard validation `recall_-1` floor to ordinal checkpoint
selection in the active corrected-split v4 path and reran the motivating
`BalancedSoftmax` soft-circumplex-regularizer branch on the frozen holdout.

- Guardrail rule: validation `recall_-1 >= 0.4032`
- Derivation: incumbent `run_019`-`run_021` selected validation
  `recall_-1 = 0.4226 / 0.4527 / 0.4332`; median `0.4332` minus `0.03`
- Fallback mode: `debug_best_finite_qwk_only`
- Rerun family: `run_031`-`run_033` (`BalancedSoftmax`, same frozen holdout,
  same LR overrides, same regularizer weights as `run_028`-`run_030`)

The issue question was narrow: does the guardrail stop low-recall validation
epochs from being silently promoted, and what does that do to downstream
holdout outcomes?

## 2. Selection Behavior

The guardrail did exactly what it was supposed to do. In all three seeds, the
highest-QWK validation checkpoint matched the previously promoted
`run_028`-`run_030` selection, and in all three seeds that checkpoint was now
marked ineligible because it missed the `0.4032` `recall_-1` floor.

| Seed | Previous run | Previously promoted val epoch | Previous val `qwk_mean` | Previous val `recall_-1` | New promoted val epoch | New val `qwk_mean` | New val `recall_-1` |
|---|---|---:|---:|---:|---:|---:|---:|
| `11` | `run_028` | 28 | 0.4280 | 0.3953 | 29 | 0.3948 | 0.4235 |
| `22` | `run_029` | 27 | 0.3906 | 0.3868 | 23 | 0.3796 | 0.4055 |
| `33` | `run_030` | 24 | 0.4113 | 0.3380 | 28 | 0.4024 | 0.4268 |

Per-seed artifact traces confirm the blocked candidate reason was always
`recall_minus1_below_floor`.

- Seed `11`: best-QWK candidate was epoch 28 (`qwk_mean=0.4280`,
  `recall_-1=0.3953`) and was rejected; epoch 29 was promoted instead.
- Seed `22`: epoch 27 reproduced the old winner and was rejected; the promoted
  checkpoint moved back to epoch 23.
- Seed `33`: the old winner was the clearest failure case, with
  `recall_-1=0.3380`; the new selected epoch moved later to 28.

No seed fell into the debug-only no-eligible branch. All three had at least one
validation epoch that cleared the floor, so the new debug fallback path remains
covered by unit tests rather than this rerun.

## 3. Family Comparison

Median holdout metrics:

| Family | Runs | Val `qwk_mean` | Val `recall_-1` | Test `qwk_mean` | Test `recall_-1` | Test minority recall | Test hedging | Test calibration |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Previous regularized family | `run_028`-`run_030` | 0.4113 | 0.3868 | 0.3473 | 0.2650 | 0.4113 | 0.6412 | 0.7086 |
| Guardrailed regularized family | `run_031`-`run_033` | 0.3948 | 0.4235 | 0.3664 | 0.2668 | 0.4093 | 0.6407 | 0.7128 |
| Current default frontier | `run_019`-`run_021` | 0.4025 | 0.4332 | 0.3620 | 0.3130 | 0.4480 | 0.6210 | 0.7130 |

What changed versus the previous regularized family:

- Validation selection is now operationally sane: the median selected
  validation `recall_-1` rose from `0.3868` to `0.4235`.
- Median selected validation `qwk_mean` fell from `0.4113` to `0.3948`, which
  is expected because the old winners were the exact below-floor epochs.
- Holdout median `qwk_mean` improved from `0.3473` to `0.3664`.
- Holdout median `recall_-1` moved only slightly, from `0.2650` to `0.2668`.
- Holdout median minority recall was effectively flat to slightly worse
  (`0.4113` to `0.4093`).
- Holdout hedging and calibration were nearly unchanged overall.

So the guardrail improved checkpoint selection discipline and modestly improved
family-level holdout QWK, but it did not materially rescue the family’s
misalignment sensitivity on the frozen holdout.

## 4. Recommendation

Keep the hard validation `recall_-1` guardrail in the ordinal selection path.

This issue is a positive result for **selection policy**, not for the
regularized family itself:

- The old failure mode is now explicit and auditable in `selection_trace` and
  `selection_summary` artifacts.
- The rerun demonstrates that the previous `run_028`-`run_030` winners were not
  just “a little better on QWK”; they were below the minimum acceptable
  `recall_-1` threshold.
- The new guardrail should remain in place for future corrected-split ordinal
  experiments.

But this does **not** justify promoting the regularized family over the current
default `run_019`-`run_021`:

- median test `recall_-1` is still materially worse (`0.2668` vs `0.3130`)
- median minority recall is still worse (`0.4093` vs `0.4480`)
- median hedging is still worse (`0.6407` vs `0.6210`)
- calibration is only roughly equal

Bottom line: `twinkl-715` should be treated as complete because it fixes the
checkpoint-selection policy bug and documents its downstream effect, but the
frontier recommendation does not change. Keep `run_019`-`run_021` as the active
corrected-split default and carry the new guardrail forward into subsequent
ordinal experiments.
