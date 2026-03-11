# VIF Experiment Review v9 - Full Frontier Audit (2026-03-11)

**Scope:** All `run_001`-`run_036` manifests. Leaderboard claims below are made only inside the active corrected-split regime (`run_016+`, post-`d937094`). A fresh metadata scan found `0` empty or placeholder `provenance.rationale` fields and `0` empty or placeholder `observations`, so no run YAML backfill was required.

## 1. Experiment Overview

Within the active regime, the backbone stayed fixed: `nomic-embed-text-v1.5`
at 256d, `window_size=1`, `hidden_dim=64`, dropout `0.3`, batch size `16`,
`epochs=100`, and `split_seed=2025`. What changed was the loss family, then
the training data (`n_train 1022 -> 1117 -> 1213`), then the circumplex
regularizer, then checkpoint guardrails, and finally inverse-loss
dimension-weighting.

Historical runs (`run_001`-`run_015`) remain archival because commit
`d937094` changed validation/test splitting to persona-level multi-label
stratification. The active frozen-holdout frontier still uses the realized
`1213 / 217 / 221` train / val / test split from `204` personas and `1651`
judged entries.

## 2. Head-to-Head Comparison

### Active corrected-split families (median across seeds)

| Family | Runs | `n_train` | MAE | Acc | QWK | Spearman | Cal | `recall_-1` | MinR | Hedging | OppV | AdjS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BalancedSoftmax | `run_019`-`run_021` | 1022 | **0.304** | 0.753 | **0.362** | **0.365** | 0.713 | 0.313 | 0.448 | 0.621 | 0.070 | **0.077** |
| BalancedSoftmax + dimweight | `run_034`-`run_036` | 1213 | 0.309 | 0.752 | 0.342 | 0.342 | 0.726 | **0.378** | **0.449** | 0.599 | 0.068 | 0.076 |
| BalancedSoftmax + circreg + recall floor | `run_031`-`run_033` | 1213 | 0.306 | **0.761** | 0.366 | 0.343 | 0.713 | 0.267 | 0.409 | 0.641 | **0.035** | **0.077** |
| BalancedSoftmax + targeted batch | `run_022`-`run_024` | 1117 | 0.312 | 0.748 | 0.349 | 0.344 | 0.687 | 0.342 | 0.434 | 0.619 | — | — |
| BalancedSoftmax + hedonism/security lift | `run_025`-`run_027` | 1213 | 0.319 | 0.737 | 0.346 | 0.345 | 0.693 | 0.328 | 0.442 | **0.598** | 0.082 | 0.072 |
| CDWCE_a3 | `run_016`-`run_018` | 1022 | 0.229 | 0.799 | 0.353 | **0.365** | 0.762 | 0.104 | 0.276 | 0.804 | — | — |
| SoftOrdinal | `run_016`-`run_018` | 1022 | **0.220** | 0.807 | 0.346 | 0.353 | 0.781 | 0.077 | 0.283 | 0.796 | — | — |
| CORN | `run_016`-`run_018` | 1022 | **0.218** | **0.811** | 0.315 | 0.356 | **0.818** | 0.089 | 0.273 | 0.801 | — | — |

`run_019`-`run_021` remains the best family-level default: QWK `0.362` (fair),
`recall_-1 0.313` (reasonable), MinR `0.448` (reasonable), and hedging `0.621`
(moderate). The weighted family is the strongest tail-sensitive reference
branch, but its QWK `0.342` and seed IQR `0.030` are too unstable to replace
the incumbent family median.

### Historical archive

| Historical leader | MAE | Acc | QWK | Spearman | Cal | `recall_-1` | MinR | Hedging |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_010` CORN | 0.206 | 0.821 | **0.434** | 0.407 | **0.835** | 0.089 | **0.285** | 0.820 |
| `run_015` CDWCE_a3 | **0.203** | **0.822** | 0.402 | 0.384 | 0.755 | 0.056 | 0.259 | 0.852 |

## 3. Per-Dimension Analysis

| Dimension | Mean QWK | Variance | Read |
|---|---:|---:|---|
| hedonism | 0.086 | 0.0146 | Hardest, still poor |
| security | 0.232 | 0.0013 | Hard, still weak |
| stimulation | 0.240 | 0.0070 | Hard and loss-sensitive |
| power | 0.305 | 0.0049 | Borderline hard, volatile |
| benevolence | 0.334 | 0.0014 | Moderate |
| achievement | 0.373 | 0.0017 | Moderate |
| universalism | 0.407 | 0.0055 | Easy but volatile |
| tradition | 0.481 | 0.0011 | Easy |
| self_direction | 0.514 | 0.0006 | Easy |
| conformity | 0.530 | 0.0004 | Easy |

Easy dimensions remain `conformity`, `self_direction`, and `tradition`. The
hardest pair is still `hedonism` and `security`.

**Weighting audit:** the selected weighted checkpoints did not focus on the
hardest dimensions. Median selected weights were `universalism 1.500`,
`stimulation 1.321`, `tradition 1.151`, `power 1.064`, `hedonism 0.972`, and
`security 0.818`. `Universalism` hit the max clamp `49` times; no dimension hit
the min clamp. Because the recipe is inverse-loss, it mostly upweighted the
already low-CE heads instead of the hard dimensions that motivated the branch.

**Error analysis:** I reran frozen-holdout validation inference from
`run_036`'s `selected_checkpoint.pt`, the strongest single corrected-split
checkpoint with both QWK `0.381` and `recall_-1 0.387`.

- `Hedonism`: the biggest misses are still polarity flips on defended rest or
  leisure. Entries about protecting Saturdays, ignoring summer curriculum email,
  and declining a promotion to keep weekends were all labeled `+1` but predicted
  `-1`.
- `Security`: the model still reads stability-seeking through the surface tone
  of worry. Entries about schedule stability, staying near family/support
  systems, and preferring predictable work over travel were labeled `+1` but
  predicted `-1`.

## 4. Calibration Deep-Dive

Every corrected-split family still has `10/10` dimensions with positive
calibration, so there is no deployment-risk dimension below `-0.4`. `CORN`
remains the calibration anchor at `0.818` (good), while the BalancedSoftmax
families sit in the moderate band at `0.687-0.726`.

The failure mode is therefore weak positive calibration, not catastrophic
negative calibration. `Security` remains the incumbent family’s weakest
dimension (`0.496` median calibration), and even the weighted branch only raises
it to `0.558` while still leaving QWK below the incumbent.

## 5. Hedging vs Minority Recall Trade-off

| Family | Hedging | MinR | Verdict |
|---|---:|---:|---|
| BalancedSoftmax + dimweight | 0.599 | 0.449 | Decisive + balanced |
| BalancedSoftmax + hedonism/security lift | 0.598 | 0.442 | Decisive + balanced |
| BalancedSoftmax | 0.621 | **0.448** | Moderate hedging, strong minority recall |
| BalancedSoftmax + targeted batch | 0.619 | 0.434 | Moderate hedging, strong minority recall |
| BalancedSoftmax + circreg + recall floor | 0.641 | 0.409 | Moderate hedging, weaker tail |
| SoftOrdinal | 0.796 | 0.283 | Excessive hedging |
| CORN | 0.801 | 0.273 | Excessive hedging |
| CDWCE_a3 | 0.804 | 0.276 | Excessive hedging |

The core lesson is unchanged: only BalancedSoftmax-style families are willing
to spend decision mass on the minority classes. Conservative ordinal losses stay
stuck near the same hedging ceiling even when QWK or calibration look cleaner.

## 6. Capacity & Overfitting

Capacity is no longer the main differentiator. All active families sit in the
same high param/sample regime (`19.3-22.9`, high but not severe). The cleaner
separation comes from training gap and selection behavior:

- `SoftOrdinal`: gap `0.027` (good) but hedging `0.796`
- BalancedSoftmax default: gap `0.123` (some overfitting)
- BalancedSoftmax + dimweight: gap `0.193` (some overfitting)
- BalancedSoftmax + circreg + recall floor: gap `0.251` (some overfitting)

Checkpoint selection worked as intended on the modern guardrailed families.
`run_031`-`run_033` rejected `20-22` epochs per seed for
`recall_minus1_below_floor`, but every seed still promoted via
`eligible_policy`; no debug fallback was needed.

## 7. Systemic Insights & Hypotheses

The broad story is now clearer. On the same `1022` rows, switching from
`CDWCE_a3` to BalancedSoftmax raised `recall_-1` from `0.104` to `0.313`, so
training-time prior correction truly mattered. After that, later improvements
cannot be attributed to loss changes alone because `n_train` also moved from
`1022` to `1117` and then `1213`.

Two hypotheses remain strongest. First, `hedonism` and `security` are semantic
polarity problems, not just class-count problems: the model keeps reading calm
pleasure or stability-seeking as guilt, fragility, or threat. Second, the
weighted branch improved the tail package mostly through broader boundary
movement, not by fixing the hardest heads. The weight trace shows exactly why:
inverse-loss weighting amplified easy low-CE dimensions instead of the ones with
the worst QWK.

## 8. Actionable Recommendations

Recent literature still supports lightweight follow-ups over heavier retraining:
[LORT 2024](https://arxiv.org/abs/2403.00250) argues simple logits retargeting
is strong enough to revisit decoupled classifier adaptation;
[LIFT 2024](https://openreview.net/forum?id=ccSSKTz9LX) finds heavy fine-tuning
hurts tail classes while lightweight adaptation helps; and
[Focal Temperature Scaling 2024](https://arxiv.org/abs/2408.11598) reports
better post-hoc calibration than standard temperature scaling.

1. Run the planned validation-only BalancedSoftmax logit-retargeting follow-up
   on both `run_020` and `run_036`. Evidence: the weighted branch improved
   `recall_-1` to `0.378` without becoming a stable default. Watch family
   median QWK, `recall_-1`, MinR, and calibration together.
2. If a training-time rerun is still needed, replace inverse-loss weighting
   with an explicit hard-dimension schedule keyed to validation difficulty or a
   whitelist (`hedonism`, `security`, `stimulation`). Evidence: the current
   recipe max-clamps `universalism` and downweights `security`.
3. Add one small semantic batch targeted at the observed polarity failures.
   Evidence: the frozen-holdout validation split still has only `14`
   `hedonism=-1` labels and `23` `security=-1` labels, and the largest replay
   errors were all semantically consistent flips. Watch `hedonism` family-median
   QWK above `0.20` and `security` back above `0.30`.
4. Run a post-hoc calibration pass on saved artifacts, starting with focal
   temperature scaling on the incumbent and weighted branches. Watch whether
   calibration rises above `0.75` without sacrificing `recall_-1`.

## 9. Summary Verdict

**Best config:** `run_019`-`run_021` BalancedSoftmax remains the best
corrected-split family-level default. The strongest exploratory single
checkpoint is `run_036`, but its family is still too QWK-unstable to promote.

**Key weakness:** `hedonism` and `security` remain polarity-confused, and the
new inverse-loss weighting recipe does not actually target those heads.

**Highest-leverage next experiment:** validation-only logit retargeting on the
incumbent and weighted BalancedSoftmax checkpoints. It is the lowest-risk way
to test whether the current tail gains can be converted into a better frontier
without another full training sweep.
