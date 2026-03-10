# Experiment Review — 2026-03-10 — `twinkl-691.4` soft circumplex-regularizer ablation

## 1. Overview

`twinkl-691.4` ran one narrow `BalancedSoftmax` ablation on top of the post-lift
`twinkl-691.3` setup to test whether a soft circumplex prior could improve
structural diagnostics without giving back the corrected-split frontier metrics.

- Ablation family: `run_028`-`run_030` (`BalancedSoftmax` with
  `circumplex_regularizer_opposite_weight=0.5` and
  `circumplex_regularizer_adjacent_weight=0.1`)
- Direct control family: `run_025`-`run_027` (same post-lift data, same frozen
  holdout, no regularizer)
- Promotion gate family: `run_019`-`run_021` (current active corrected-split
  default)
- Constants held fixed: `nomic-ai/nomic-embed-text-v1.5` at 256d,
  `window_size=1`, `hidden_dim=64`, `dropout=0.3`, `split_seed=2025`,
  model seeds `11/22/33`, and frozen holdout
  `config/experiments/vif/twinkl_681_5_holdout.yaml`

The decision question was strict: does this regularizer improve circumplex
structure **and** avoid regressions on `qwk_mean`, `recall_-1`,
`minority_recall_mean`, `hedging_mean`, and `calibration_global` versus the
post-lift control? If not, drop it and move on.

## 2. Family Comparison

All values below are family medians with IQR in parentheses.

| Family | Runs | `qwk_mean` | `recall_-1` | `minority_recall_mean` | `hedging_mean` | `calibration_global` |
|---|---|---:|---:|---:|---:|---:|
| Current default BalancedSoftmax | `run_019`-`run_021` | **0.362** (0.010) | 0.313 (0.033) | **0.448** (0.025) | 0.621 (0.038) | **0.713** (0.036) |
| Post-lift control BalancedSoftmax | `run_025`-`run_027` | 0.346 (0.009) | **0.328** (0.039) | 0.442 (0.023) | **0.598** (0.021) | 0.693 (0.026) |
| Circumplex-regularized BalancedSoftmax | `run_028`-`run_030` | 0.347 (0.033) | 0.265 (0.020) | 0.411 (0.012) | 0.641 (0.009) | 0.709 (0.030) |

The regularizer did **not** clear the direct keep gate:

- Median `qwk_mean` improved only marginally over the post-lift control
  (`0.347` vs `0.346`).
- Median `calibration_global` also improved (`0.709` vs `0.693`).
- But the family gave back too much decisive boundary behavior:
  `recall_-1` fell from `0.328` to `0.265`, `minority_recall_mean` fell from
  `0.442` to `0.411`, and `hedging_mean` worsened from `0.598` to `0.641`.

So this was not a clean structure-without-cost win. It traded away too much
misalignment sensitivity to stay on the frontier.

## 3. Target-Dimension Readout

Median per-dimension holdout summaries:

| Family | `hedonism qwk` | `security qwk` | `opposite_violation_mean` | `adjacent_support_mean` |
|---|---:|---:|---:|---:|
| Current default BalancedSoftmax | 0.247 | **0.297** | 0.070 | 0.077 |
| Post-lift control BalancedSoftmax | **0.256** | 0.199 | 0.082 | 0.072 |
| Circumplex-regularized BalancedSoftmax | 0.111 | 0.199 | **0.039** | **0.077** |

Two things were true at once:

- The regularizer **did** improve the structure metrics sharply. Median
  `opposite_violation_mean` dropped from `0.082` to `0.039`, and median
  `adjacent_support_mean` recovered from `0.072` to `0.077`.
- The target dimensions did not justify promoting it. `security qwk` stayed
  stuck at `0.199`, still far below the incumbent `0.297`, and `hedonism qwk`
  collapsed from `0.256` in the post-lift control to `0.111`.

That means the ablation helped the global structural prior more than the actual
hard-dimension behavior we needed to rescue.

## 4. Circumplex Diagnostics

For the incumbent default `run_019`-`run_021`, the circumplex summaries below
were recomputed from the saved selected-test artifacts because those runs
predated direct circumplex payload logging in the YAMLs.

Largest opposite-pair median improvements for `run_028`-`run_030` relative to
`run_025`-`run_027`:

- `achievement <> benevolence`: `0.094` vs `0.186` (`-0.092`)
- `benevolence <> power`: `0.032` vs `0.099` (`-0.067`)
- `self_direction <> tradition`: `0.085` vs `0.140` (`-0.054`)
- `conformity <> stimulation`: `0.010` vs `0.061` (`-0.051`)
- `hedonism <> tradition`: `0.025` vs `0.073` (`-0.048`)

Strongest adjacent-pair gains for `run_028`-`run_030` relative to
`run_025`-`run_027`:

- `conformity <> security`: `0.098` vs `0.055` (`+0.043`)
- `security <> tradition`: `0.105` vs `0.068` (`+0.037`)
- `benevolence <> conformity`: `0.092` vs `0.075` (`+0.017`)
- `benevolence <> universalism`: `0.110` vs `0.094` (`+0.016`)
- `conformity <> tradition`: `0.092` vs `0.082` (`+0.011`)

Weakest adjacent pairs in the regularized family:

- `power <> security`: `0.026` (still below control `0.027`)
- `achievement <> hedonism`: `0.033` (below control `0.042`)
- `hedonism <> stimulation`: `0.043` (essentially flat vs control `0.042`)
- `achievement <> power`: `0.064` (below control `0.068`)
- `self_direction <> universalism`: `0.077` (flat vs control `0.077`)

So the regularizer did the job it was supposed to do structurally, but the
remaining weak adjacency story is still concentrated around the same hard
behavioral zones: `hedonism`, `power`, and `security`.

## 5. Keep/Drop Gate

Direct ablation success criteria versus `run_025`-`run_027`:

- Better `opposite_violation_mean`: **pass**
- No worse `adjacent_support_mean`: **pass**
- No regression on `qwk_mean`: **pass**, but only marginally
- No regression on `recall_-1`: **fail**
- No regression on `minority_recall_mean`: **fail**
- No regression on `hedging_mean`: **fail**
- No regression on `calibration_global`: **pass**

Promotion criteria versus `run_019`-`run_021`:

- Median `qwk_mean` no worse than incumbent: **fail** (`0.347` vs `0.362`)
- Median `security qwk` no worse than incumbent: **fail** (`0.199` vs `0.297`)

## 6. Recommendation

Drop the soft circumplex-regularizer ablation.

This issue still delivered useful evidence: the structural prior can reduce
opposite-pair collapse substantially, and it can recover adjacent support to the
incumbent default range. But on this fixed-weight setting, that came with too
much cost in `recall_-1`, minority recall, and hedging. The model became more
structurally polite and less operationally decisive, which is the wrong trade
for the current frontier.

The next follow-up should not be another regularizer-weight sweep inside this
thread. The better handoff is the newer March 10 review direction:
per-dimension uncertainty weighting on `BalancedSoftmax`, which directly targets
the unresolved `hedonism`/`security` noise-vs-signal problem without forcing a
global structure penalty.
