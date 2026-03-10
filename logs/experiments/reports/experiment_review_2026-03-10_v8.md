# VIF Experiment Review v8 - Full Frontier Refresh (2026-03-10)

**Scope:** All `run_001`-`run_033` manifests (97 YAML configs), split into historical pre-`d937094` (`run_001`-`run_015`) and active corrected-split (`run_016+`). Leaderboard claims below are made only inside the corrected-split regime.

## 1. Experiment Overview

The active regime keeps the same backbone and training scaffold across all corrected-split runs: `nomic-embed-text-v1.5` at 256d, `window_size=1`, `hidden_dim=64`, dropout `0.3`, batch size `16`, MC dropout `50`, and `split_seed=2025`. What changed was the loss family, then the training data (`n_train` `1022 -> 1117 -> 1213`), then the circumplex regularizer, and finally the checkpoint-selection policy.

Historical runs (`run_001`-`run_015`) remain useful only as archive because commit `d937094` changed validation/test splitting to persona-level multi-label stratification. The active corrected-split frontier is still the right board for recommendations, and the current dataset now contains `1651` judged entries from `204` personas with realized active splits of `1213 / 217 / 221`.

## 2. Head-to-Head Comparison

### Active corrected-split families (median across seeds unless noted)

| Family | n_train | MAE | Acc | QWK | Spearman | Cal | Recall -1 | MinR | Hedging |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BalancedSoftmax (`run_019`-`run_021`) | 1022 | 0.304 | 0.753 | 0.362 | 0.365 | 0.713 | 0.313 | 0.448 | 0.621 |
| BalancedSoftmax + circreg + recall floor (`run_031`-`run_033`) | 1213 | 0.306 | 0.761 | 0.366 | 0.343 | 0.713 | 0.267 | 0.409 | 0.641 |
| BalancedSoftmax + targeted batch (`run_022`-`run_024`) | 1117 | 0.313 | 0.748 | 0.349 | 0.344 | 0.687 | 0.342 | 0.434 | 0.619 |
| BalancedSoftmax + hedonism/security lift (`run_025`-`run_027`) | 1213 | 0.319 | 0.737 | 0.346 | 0.345 | 0.693 | 0.328 | 0.442 | 0.598 |
| CDWCE_a3 (`run_016`-`run_018`) | 1022 | 0.229 | 0.799 | 0.353 | 0.365 | 0.762 | 0.104 | 0.276 | 0.804 |
| SoftOrdinal (`run_016`-`run_018`) | 1022 | 0.220 | 0.807 | 0.346 | 0.354 | 0.781 | 0.077 | 0.283 | 0.796 |
| CORN (`run_016`-`run_018`) | 1022 | 0.218 | 0.811 | 0.315 | 0.356 | 0.818 | 0.089 | 0.273 | 0.801 |

`run_031`-`run_033` are comparable to the incumbent on QWK (`0.366` vs `0.362`, <5% delta) and calibration (`0.713` vs `0.713`), but they are materially worse on `recall_-1`, minority recall, and hedging. That keeps `run_019`-`run_021` as the best balanced corrected-split family instead of promoting the newer regularized rerun.

### Historical archive

| Historical leader | MAE | Acc | QWK | Spearman | Cal | Recall -1 | MinR | Hedging |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_010` CORN | 0.206 | 0.821 | 0.434 | 0.407 | 0.835 | 0.089 | 0.285 | 0.820 |
| `run_015` CDWCE_a3 | 0.203 | 0.822 | 0.402 | 0.384 | 0.755 | 0.056 | 0.259 | 0.852 |
| `run_007` CORN | 0.205 | 0.821 | 0.413 | 0.402 | 0.838 | 0.103 | 0.285 | 0.817 |

These numbers stayed stronger only because the pre-`d937094` evaluation regime was easier. They are not active SOTA.

## 3. Per-Dimension Analysis

| Dimension | Mean QWK | Variance | Read |
|---|---:|---:|---|
| hedonism | 0.081 | 0.0160 | Hardest; still poor |
| security | 0.234 | 0.0014 | Hard; persistently weak |
| stimulation | 0.250 | 0.0068 | Hard and loss-sensitive |
| power | 0.298 | 0.0049 | Borderline hard and volatile |
| benevolence | 0.333 | 0.0015 | Moderate |
| achievement | 0.381 | 0.0012 | Moderate |
| universalism | 0.409 | 0.0061 | Easy but volatile |
| tradition | 0.479 | 0.0011 | Easy |
| self_direction | 0.512 | 0.0007 | Easy |
| conformity | 0.530 | 0.0004 | Easy |

Easy dimensions are `conformity`, `self_direction`, and `tradition`, all with mean QWK above `0.47` (moderate). `hedonism` is the only dimension that stays genuinely poor at mean QWK `0.081`, while `security`, `stimulation`, and `power` remain operationally hard. The most volatile dimensions are `hedonism`, `stimulation`, `universalism`, and `power`.

For qualitative replay, I loaded `run_020_BalancedSoftmax`, the strongest single checkpoint inside the active frontier family. Re-running the saved validation sample IDs produced replayed validation QWK `0.417` versus the saved selected-validation QWK `0.408`, close enough for qualitative inspection.

- `Hedonism`: opposite-end errors are mostly polarity mistakes. One +1 entry about imagining future free evenings and enjoying pottery/wine after a parent's recovery (`e5cea325`, `t=7`) was predicted -1 with high uncertainty `0.970`; another +1 entry defending protected Saturdays (`e5cea325`, `t=1`) was predicted -1 with probability `0.863` on the negative class. The mirror failure also appears: a clearly frustrated shift-schedule entry (`5fcf93f5`, `t=2`) was predicted +1.
- `Security`: the model confuses stability-seeking with anxiety language. Losing a condo that would place a family near relatives (`7664d969`, `t=7`) was labeled +1 but predicted -1, while a risky startup transition (`68d0e778`, `t=1`) was labeled -1 but predicted +1. A scheduling-stability complaint (`5fcf93f5`, `t=2`) was also flipped from +1 to -1.

## 4. Calibration Deep-Dive

Every corrected-split family still has `10/10` dimensions with positive calibration, so there is no deployment-risk dimension below `-0.4`. Global calibration stays good for `CORN` at `0.818`, moderate-to-good for `CDWCE_a3` at `0.762`, and moderate for the BalancedSoftmax branches at `0.687-0.713`.

The weak point is systematic rather than catastrophic: `security` is the lowest-calibrated active dimension almost everywhere, often landing in the `0.41-0.63` band while the same runs keep global calibration positive. The circreg families did not solve that; they only moved the weakness around slightly.

## 5. Hedging vs Minority Recall Trade-off

| Family | Hedging % | Minority Recall | Verdict |
|---|---:|---:|---|
| BalancedSoftmax + hedonism/security lift | 59.8 | 0.442 | Decisive + balanced |
| BalancedSoftmax + targeted batch | 61.9 | 0.434 | Moderate hedging, strong minority recall |
| BalancedSoftmax | 62.1 | 0.448 | Moderate hedging, best minority recall |
| BalancedSoftmax + circreg + recall floor | 64.1 | 0.409 | Moderate hedging, weaker minority recall |
| SoftOrdinal | 79.6 | 0.283 | Moderate hedging ceiling |
| CORN | 80.1 | 0.273 | Excessive hedging |
| CDWCE_a3 | 80.4 | 0.276 | Excessive hedging |

Only `run_025`-`run_027` crosses both thresholds simultaneously. The core lesson is unchanged: BalancedSoftmax-like families are the only ones willing to spend decision mass on minorities at all.

## 6. Capacity & Overfitting

Active corrected-split runs all sit in the high param/sample regime at `19.3-22.9`, while the earliest MiniLM experiments were severe at `>500`. That means capacity is no longer the main differentiator inside the frontier.

Training gaps tell the cleaner story. `SoftOrdinal` is tight at gap `0.027` (good) but pays for it with heavy hedging. `BalancedSoftmax` is moderate at `0.123` (some overfitting), `CDWCE_a3` and both guardrailed circreg families are higher at `0.251-0.255` (some), and `LDAM_DRW` is unusable at `0.816` (overfitting). Early stopping still triggered well before the `100`-epoch budget in every active family.

## 7. Systemic Insights & Hypotheses

The main story is not "which loss wins?" anymore. On the same `1022` rows, switching from `CDWCE_a3` to `BalancedSoftmax` raised `recall_-1` from `0.104` to `0.313`, so prior-corrected loss design clearly matters. After that, adding data changed the frontier only incrementally and inconsistently: the targeted batch improved `recall_-1` to `0.342`, the regenerated hedonism/security lift lowered hedging to `0.598`, and the regularizer/guardrail combo only made QWK comparable while weakening the tail.

Two hypotheses look strongest. First, hedonism/security are semantic polarity problems, not just class-count problems: the model often treats relief, stability-seeking, or defended pleasure as if they were threat or guilt. Second, the circumplex regularizer is acting as a ranking smoother, not a tail-signal generator; it can polish aggregate QWK when data are fixed (`run_025 -> run_028`) but it does not create the minority evidence needed for reliable `-1` and `+1` calls on the hardest dimensions.

## 8. Actionable Recommendations

Web corroboration before finalizing: [LORT 2024](https://arxiv.org/abs/2403.00250) revisits decoupled classifier retraining for long-tail recognition and reports state-of-the-art results with simple logits retargeting; [LIFT at ICML 2024](https://proceedings.mlr.press/v235/shi24g.html) finds heavy fine-tuning hurts tail classes and lightweight adaptation works better; [UW-SO in IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02625-x) argues classical uncertainty weighting overfits and proposes temperature-controlled inverse-loss weighting; [Focal Temperature Scaling at ECAI 2024](https://arxiv.org/abs/2408.11598) reports better calibration than standard temperature scaling.

1. Add per-dimension loss weighting on top of frontier BalancedSoftmax, using a tempered inverse-loss or uncertainty-style schedule rather than fixed equal weights. Evidence: hedonism mean QWK is only `0.081` while conformity is `0.530`; equal weighting is letting noisy heads drag the average. Watch `qwk_mean`, hedonism/security QWK, and `recall_-1`.
2. Freeze the `run_020` frontier checkpoint and retrain only the classifier head with class-balanced sampling or logits retargeting. Evidence: `run_020` already has the best active single-seed tail profile (`qwk_mean 0.378`, `recall_-1 0.342`, `MinR 0.449`), and recent long-tail literature still favors lightweight decoupled adaptation over heavier end-to-end changes. Watch `recall_-1`, minority recall, and calibration drift.
3. Generate a small, explicit hedonism/security batch targeted at the observed polarity errors: quiet pleasure without guilt, stability gain described through prior risk, and security loss framed through agency. Evidence: current label support is still thin (`hedonism` `141/-1`, `173/+1`; `security` `151/-1`, `288/+1`), and replay errors show semantic confusion rather than random noise. Watch hedonism QWK above `0.15` and security QWK back above `0.30`.
4. Run focal temperature scaling as a post-hoc calibration pass on saved corrected-split artifacts, especially `BalancedSoftmax` and `CORN`. Evidence: BalancedSoftmax pays a real calibration tax (`0.713` vs CORN `0.818`), while ECAI 2024 shows focal temperature scaling outperforming standard temperature scaling. Watch calibration, hedging, and whether `recall_-1` stays flat instead of collapsing.

## 9. Summary Verdict

**Best config:** `run_019`-`run_021` BalancedSoftmax remains the best corrected-split family because it combines QWK `0.362` (fair), `recall_-1` `0.313` (reasonable), minority recall `0.448` (reasonable), and hedging `0.621` (moderate) better than any alternative family. The newer `run_031`-`run_033` branch is comparable on QWK but not on the tail metrics that matter operationally.

**Key weakness:** `hedonism` is still effectively under-modeled, and `security` is the next clear blocker. The model has not learned those dimensions' polarity cleanly.

**Highest-leverage next experiment:** BalancedSoftmax plus per-dimension uncertainty/inverse-loss weighting. It is the smallest change that directly targets the current failure mode - noisy hard dimensions consuming too much training signal - without abandoning the only loss family that already moves the minority classes.
