# Experiment Review — 2026-03-11 — `twinkl-719.5` BalancedSoftmax post-hoc retargeting

## 1. Overview

`twinkl-719.5` tested the lowest-risk frontier follow-up after the weighted
`BalancedSoftmax` rerun from `twinkl-719.3`: validation-only Menon-style logit
retargeting on saved artifacts, with **no retraining**.

- Primary checkpoint target: incumbent `run_020` `BalancedSoftmax`
- Weighted reference checkpoint: `run_036` from `twinkl-719.3`
- Selection discipline: validation-only sweep, one untouched final test
  evaluation per checkpoint
- Metric basis: this review scores post-hoc artifacts on the same thresholded
  continuous prediction basis used by the frontier run YAMLs, so the baseline
  post-hoc metrics line up with the original run logs

The core decision question was simple: can post-hoc retargeting extract enough
extra `recall_-1` from the incumbent checkpoint to close the frontier gap, or
does the project still need a stronger intervention?

## 2. Run-Level Results

| Run | Selected tau | Test QWK | Delta QWK | Test recall_-1 | Delta recall_-1 | Test MinR | Delta MinR | Test hedging | Delta hedging | Test calibration | Delta calibration | OppV | Delta OppV | AdjS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_020` | `0.30` | 0.338 | -0.039 | 0.350 | +0.008 | 0.397 | -0.052 | 0.562 | -0.060 | 0.583 | -0.131 | 0.107 | +0.037 | 0.089 |
| `run_036` | `0.00` | 0.381 | +0.000 | 0.387 | +0.000 | 0.492 | +0.000 | 0.599 | +0.000 | 0.726 | +0.000 | 0.068 | +0.000 | 0.084 |

Only `run_020` moved. The weighted reference `run_036` selected `tau=0.00`,
which means the validation sweep found no retargeted policy that beat its saved
baseline under the guarded selector.

For `run_020`, `tau=0.30` did recover a small amount of extra `recall_-1`
(`0.342 -> 0.350`) and noticeably reduced hedging (`0.621 -> 0.562`), but the
cost was too high on the rest of the package:

- `qwk_mean` fell from `0.378` to `0.338`
- minority recall fell from `0.449` to `0.397`
- calibration fell from `0.713` to `0.583`
- opposite-pair circumplex violations worsened from `0.070` to `0.107`

So the retargeted incumbent became slightly more willing to emit minority
signals, but it did so by becoming materially less stable and less well-behaved.

## 3. Frontier Comparison

The post-hoc result does **not** close the frontier gap.

Relative to the active default family `run_019`-`run_021`:

- retargeted `run_020` does beat the family median on `recall_-1`
  (`0.350` vs `0.313`)
- but it falls below the family median on `qwk_mean` (`0.338` vs `0.362`)
  and minority recall (`0.397` vs `0.448`)
- it also gives back too much calibration (`0.583` vs `0.713`) and worsens
  compact circumplex structure

Relative to the weighted reference checkpoint `run_036`:

- `run_036` remains stronger on every operational metric that matters here:
  `qwk_mean 0.381`, `recall_-1 0.387`, minority recall `0.492`,
  hedging `0.599`, calibration `0.726`, and `OppV 0.068`
- the fact that `run_036` selected `tau=0.00` is itself useful evidence:
  the weighted branch was already near its best operating point under this
  style of post-hoc prior retargeting

This means the validation-only boundary shift is not enough to turn the
incumbent checkpoint into a better frontier candidate than either the active
default family or the strongest weighted checkpoint.

## 4. Recommendation

No frontier change.

- Keep `run_019`-`run_021` as the active corrected-split default family.
- Keep `run_034`-`run_036` as the best tail-sensitive reference branch.
- Treat `twinkl-719.5` as a negative result for the incumbent-centered
  post-hoc path: it produced a small recall lift, but not a clean enough
  trade-off to promote.

If the project still wants a stronger incumbent-centered follow-up, then
`twinkl-719.6` remains warranted as the next fallback: freeze `run_020` and
retrain the classifier head only. If not, this post-hoc line is effectively
exhausted and the frontier should stay where it is.

## 5. Bottom Line

`twinkl-719.5` confirms that the current frontier is not blocked by an obvious
cheap boundary tweak.

The incumbent `run_020` can be nudged to emit slightly more `-1` decisions, but
the trade comes with too much damage to ranking quality, minority balance,
calibration, and circumplex cleanliness. The weighted checkpoint `run_036`
needs no such retargeting, and the overall recommendation remains unchanged.
