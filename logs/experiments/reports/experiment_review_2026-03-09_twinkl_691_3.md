# Experiment Review — 2026-03-09 — `twinkl-691.3` post-lift rebaseline

## 1. Overview

`twinkl-691.3` reran the frozen-holdout corrected-split frontier after the regenerated `twinkl-691.2` `Hedonism`/`Security` batch was verified, wrangled, labeled, and consolidated. The current workspace now contains `204` personas and `1651` judged entries. The frozen holdout persona IDs stayed fixed; with the extra train-only personas, the realized row split is now `1213 / 217 / 221` train / val / test.

The rebaseline used the frontier experiment driver at `scripts/experiments/critic_training_v4_review.py` and trained paired `BalancedSoftmax` and `SoftOrdinal` runs on the same holdout with model seeds `11 / 22 / 33`. The new experiments are:

- `run_025_BalancedSoftmax`, `run_026_BalancedSoftmax`, `run_027_BalancedSoftmax`
- `run_025_SoftOrdinal`, `run_026_SoftOrdinal`, `run_027_SoftOrdinal`

The decision question is narrow: does the regenerated `Hedonism`/`Security` lift displace the active corrected-split default `run_019`-`run_021`, or at least beat the earlier targeted branch `run_022`-`run_024` cleanly enough to replace it as the best post-lift follow-up?

## 2. Family Comparison

All values below are family medians with IQR in parentheses.

| Family | Runs | `qwk_mean` | `recall_-1` | `minority_recall_mean` | `hedging_mean` | `calibration_global` |
|---|---|---:|---:|---:|---:|---:|
| Current default BalancedSoftmax | `run_019`-`run_021` | **0.362** (0.010) | 0.313 (0.033) | **0.448** (0.025) | 0.621 (0.038) | 0.713 (0.036) |
| Prior targeted BalancedSoftmax | `run_022`-`run_024` | 0.349 (0.016) | **0.342** (0.029) | 0.434 (0.008) | 0.619 (0.015) | 0.687 (0.022) |
| New BalancedSoftmax | `run_025`-`run_027` | 0.346 (0.009) | 0.328 (0.039) | 0.442 (0.023) | **0.598** (0.021) | 0.693 (0.026) |
| New SoftOrdinal | `run_025`-`run_027` | 0.340 (0.010) | 0.082 (0.021) | 0.260 (0.017) | 0.823 (0.013) | **0.738** (0.022) |

The new-family winner is still `BalancedSoftmax`: it beats the rerun `SoftOrdinal` family on `qwk_mean`, `recall_-1`, minority recall, and hedging. But it does not beat the incumbent corrected-split default. Relative to `run_019`-`run_021`, the new `BalancedSoftmax` family gives back `0.016` median QWK, `0.006` median `recall_-1`, and `0.006` median minority recall. Relative to the earlier targeted branch `run_022`-`run_024`, it recovers calibration (`0.693` vs `0.687`) and hedging (`0.598` vs `0.619`) but still falls slightly on QWK (`0.346` vs `0.349`) and `recall_-1` (`0.328` vs `0.342`).

`SoftOrdinal` remains the calibration-conscious comparator rather than a deployment candidate. Its median calibration is the best of the new families, but the cost is unchanged neutral collapse: `recall_-1 = 0.082` and `hedging_mean = 0.823`.

## 3. Target-Dimension Readout

Median per-dimension holdout summaries:

| Family | `hedonism qwk` | `hedonism cal` | `hedonism hedge` | `security qwk` | `security cal` | `security hedge` |
|---|---:|---:|---:|---:|---:|---:|
| Current default BalancedSoftmax | 0.247 | 0.856 | 0.720 | **0.297** | 0.496 | 0.706 |
| Prior targeted BalancedSoftmax | 0.147 | 0.795 | 0.729 | **0.300** | 0.521 | 0.710 |
| New BalancedSoftmax | **0.256** | **0.877** | 0.765 | 0.199 | 0.528 | 0.738 |
| New SoftOrdinal | -0.035 | 0.685 | 0.964 | 0.205 | **0.551** | 0.901 |

The regenerated batch produced a mixed hard-dimension result rather than a clean win:

- `Hedonism` improved modestly for the new `BalancedSoftmax` family. Its median `hedonism qwk` rose to `0.256`, which is slightly better than the incumbent `0.247` and clearly better than the earlier targeted branch `0.147`. Calibration also improved.
- `Security` still broke the rebaseline. The new `BalancedSoftmax` family dropped to median `security qwk = 0.199`, materially below both the incumbent `0.297` and the earlier targeted branch `0.300`. Calibration moved slightly up, but that came with more hedging and no usable accuracy gain.
- `SoftOrdinal` did not rescue the target dimensions. It stayed highly neutral on both `Hedonism` and `Security`, with especially severe `hedonism` hedging (`0.964`) and a negative median `hedonism qwk` (`-0.035`).

So the regenerated batch helped one of the two intended dimensions a little, but not enough to outweigh the `Security` regression.

## 4. Circumplex Diagnostics

Median family summaries:

| Family | `opposite_violation_mean` | `adjacent_support_mean` |
|---|---:|---:|
| Current default BalancedSoftmax | 0.070 (0.016) | **0.077** (0.013) |
| Prior targeted BalancedSoftmax | **0.067** (0.008) | 0.072 (0.010) |
| New BalancedSoftmax | 0.082 (0.010) | 0.072 (0.008) |
| New SoftOrdinal | 0.069 (0.006) | 0.056 (0.002) |

Circumplex behavior is another reason not to switch the default:

- The new `BalancedSoftmax` family shows **worse opposite-pair collapse** than the incumbent (`0.082` vs `0.070`) and no adjacent-support gain (`0.072` vs `0.077`).
- The new `SoftOrdinal` family is slightly cleaner on opposite-pair violation than the new `BalancedSoftmax` rerun, but it pays for that with the weakest adjacent support of any compared family (`0.056`).

Because the active recommendation stays with the current default `run_019`-`run_021`, its worst family-median circumplex pairs remain the operative diagnostic reference:

- Highest opposite-pair scores:
  - `achievement <> benevolence`: `0.148`
  - `self_direction <> tradition`: `0.102`
  - `conformity <> self_direction`: `0.090`
  - `security <> self_direction`: `0.082`
  - `benevolence <> power`: `0.075`
- Weakest adjacent-pair scores:
  - `hedonism <> stimulation`: `0.020`
  - `achievement <> hedonism`: `0.025`
  - `power <> security`: `0.028`
  - `self_direction <> stimulation`: `0.045`
  - `self_direction <> universalism`: `0.046`

## 5. Recommendation

Keep `run_019`-`run_021` as the active corrected-split default.

The new `BalancedSoftmax` family is within the `0.03` QWK guard of the incumbent, but it does **not** offer a clear enough hard-dimension or structure-aware advantage to justify a switch:

- median `qwk_mean` is lower: `0.346` vs `0.362`
- median `security qwk` regresses materially: `0.199` vs `0.297`
- median `opposite_violation_mean` worsens: `0.082` vs `0.070`
- median `adjacent_support_mean` does not improve: `0.072` vs `0.077`

The rerun `SoftOrdinal` family remains useful as a calibration reference, but its `recall_-1` and hedging profile still disqualify it as the mainline model family.

No post-hoc follow-up was run. The provisional new-family winner (`BalancedSoftmax`) was not merely blocked by calibration or neutral-bias tax; it lost on the more important combination of aggregate QWK, `Security`, and circumplex structure relative to the incumbent.

## 6. Bottom Line

`twinkl-691.3` is a negative result in the narrow sense that the regenerated `Hedonism`/`Security` lift did not move the active frontier. It did produce a cleaner post-lift `BalancedSoftmax` rerun than the earlier targeted branch on hedging and calibration, and it modestly improved `Hedonism`, but the unresolved `Security` regression means the corrected-split default should stay where it is.
