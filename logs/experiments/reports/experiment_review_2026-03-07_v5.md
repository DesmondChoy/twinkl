# Experiment Review — 2026-03-07 — corrected-split long-tail ablations

## 1. Experiment Overview

This review covers all `run_001`-`run_020` logs, but leaderboard claims are made only inside the active corrected-split regime introduced after commit `d937094`. The historical pre-`d937094` board remains useful context for how the search evolved, not for direct SOTA ranking.

What changed historically: encoder family (`MiniLM-384d` to `nomic-256d`), window size (`3` to `1`/`2`), hidden size (`32/64/128/256`), and loss heads. What changed in the active regime: only the loss family. `run_016`-`run_018` form a complete 3-seed corrected-split matrix for `CDWCE_a3`, `CORN`, and `SoftOrdinal`; `run_019`-`run_020` add provisional 2-seed families for `BalancedSoftmax` and `LDAM_DRW`.

Active-regime constants: `nomic-ai/nomic-embed-text-v1.5` truncated to 256d, `window_size=1`, `hidden_dim=64`, `dropout=0.3`, `split_seed=2025`, `state_dim=266`, no truncation, and split sizes `1022/217/221`. That means the newest signal is mostly about decision-boundary design, not representation capacity. Capacity is also effectively fixed: param/sample ratio is 22.3-22.95 (high) for every active run.

## 2. Head-to-Head Comparison

### Current Frontier (Post-`d937094`, corrected split)

| Metric | BalancedSoftmax (`run_019-020`, seeds 11/22) | CDWCE_a3 (`run_016-018`, seeds 11/22/33) | SoftOrdinal (`run_016-018`, seeds 11/22/33) | LDAM_DRW (`run_019-020`, seeds 11/22) | CORN (`run_016-018`, seeds 11/22/33) |
|---|---:|---:|---:|---:|---:|
| MAE | 0.301 | 0.229 | 0.220 | 0.223 | **0.218** |
| Accuracy | 0.754 | 0.799 | 0.807 | 0.807 | **0.811** |
| QWK | **0.370** | 0.353 | 0.346 | 0.344 | 0.315 |
| Spearman | 0.362 | **0.365** | 0.353 | 0.342 | 0.356 |
| Calibration | 0.720 | 0.762 | 0.781 | 0.757 | **0.818** |
| Minority Recall | **0.424** | 0.276 | 0.283 | 0.285 | 0.273 |
| recall_-1 | **0.309** | 0.104 | 0.077 | 0.069 | 0.089 |
| Hedging | **0.631** | 0.804 | 0.796 | 0.794 | 0.801 |

BalancedSoftmax is the new provisional leader: median QWK 0.370 (fair), recall_-1 0.309 (reasonable by project history, though still below a robust production target), minority recall 0.424 (reasonable), and hedging 0.631 (moderate). The catch is family completeness: it has only two seeds so far, while `CDWCE_a3`, `SoftOrdinal`, and `CORN` already have full 3-seed summaries.

### Historical Frontier (Pre-`d937094`, archival only)

| Metric | `run_010 CORN` | `run_015 CDWCE_a3` | `run_014 SoftOrdinal` |
|---|---:|---:|---:|
| MAE | 0.206 | **0.203** | 0.204 |
| Accuracy | 0.821 | 0.822 | **0.825** |
| QWK | **0.434** | 0.402 | 0.388 |
| Spearman | **0.407** | 0.384 | 0.394 |
| Calibration | **0.835** | 0.755 | 0.801 |
| Minority Recall | 0.285 | 0.259 | **0.288** |
| recall_-1 | **0.089** | 0.056 | 0.075 |
| Hedging | **0.820** | 0.852 | **0.820** |

The historical board mainly tells us how much the split fix changed the evaluation regime: the old QWK leader no longer wins once the validation/test partitions preserve per-dimension sign support.

## 3. Per-Dimension Analysis

Corrected-split mean QWKs sort into three buckets:

| Dimension | Mean QWK | Std | Read |
|---|---:|---:|---|
| conformity | 0.525 | 0.048 | easy |
| self_direction | 0.525 | 0.045 | easy |
| universalism | 0.474 | 0.086 | easy but somewhat volatile |
| tradition | 0.449 | 0.035 | easy |
| achievement | 0.401 | 0.041 | borderline easy |
| stimulation | 0.311 | 0.068 | middling |
| benevolence | 0.293 | 0.056 | hard |
| power | 0.277 | 0.129 | hard and volatile |
| security | 0.234 | 0.040 | hard |
| hedonism | -0.010 | 0.137 | hardest and volatile |

The hardest two dimensions are `hedonism` and `security`. Checkpoint-level review on the top single corrected-split run (`run_016_SoftOrdinal`) shows the failure mode is mostly semantic under-reading, not random noise. For Hedonism, the model predicted neutral on explicitly enjoyable episodes such as a three-hour phone-free woodworking session (`target=+1`, `pred=0`) and sitting with the emotional payoff of a promotion (`target=+1`, `pred=0`). For Security, the worst miss was a full polarity flip: a promotion-threat entry about workplace instability was labeled `-1` but predicted `+1`; other misses defaulted to neutral on obvious precarity signals such as gig-payment instability.

These patterns fit the data support. Under the corrected split, Hedonism has only 14 non-zero personas in validation and 13 in test; Power has only 11 in validation and 9 in test, explaining why it remains the most volatile dimension.

## 4. Calibration Deep-Dive

Calibration is the healthiest part of the active frontier. Every corrected-split run has 10/10 positively calibrated dimensions, and no dimension in any active run fell below the deployment-risk threshold of `-0.4`. CORN remains the calibration anchor at median calibration 0.818 (good), followed by SoftOrdinal 0.781, CDWCE_a3 0.762, LDAM_DRW 0.757, and BalancedSoftmax 0.720. That means BalancedSoftmax pays a calibration tax relative to CORN, but not a catastrophic one; its calibration is still in the project's "good" range.

## 5. Hedging vs Minority Recall Trade-off

| Run + Loss | Hedging % | Minority Recall | Verdict |
|---|---:|---:|---|
| run_016 CDWCE_a3 | 80.6 | 0.266 | over-hedged |
| run_017 CDWCE_a3 | 80.3 | 0.294 | over-hedged |
| run_018 CDWCE_a3 | 80.4 | 0.276 | over-hedged |
| run_016 CORN | 78.4 | 0.274 | moderate hedge, weak minority |
| run_017 CORN | 81.3 | 0.266 | over-hedged |
| run_018 CORN | 80.1 | 0.273 | over-hedged |
| run_016 SoftOrdinal | 79.6 | 0.292 | moderate hedge, weak minority |
| run_017 SoftOrdinal | 84.3 | 0.229 | over-hedged |
| run_018 SoftOrdinal | 79.5 | 0.283 | moderate hedge, weak minority |
| run_019 BalancedSoftmax | 64.2 | 0.399 | balanced but still hedged |
| run_020 BalancedSoftmax | 62.1 | 0.449 | balanced but still hedged |
| run_019 LDAM_DRW | 79.1 | 0.274 | moderate hedge, weak minority |
| run_020 LDAM_DRW | 79.6 | 0.296 | moderate hedge, weak minority |

No active run achieved the "decisive + balanced" target of hedging < 60% and minority recall > 30%. BalancedSoftmax came closest by a wide margin.

## 6. Capacity & Overfitting

Because active runs all share the same backbone, capacity is not the differentiator; param/sample ratio is high (22.3-22.95) across the board. Overfitting behavior does differ materially. SoftOrdinal has the healthiest median gap at 0.027 (good) and CORN stays good at 0.070. BalancedSoftmax sits in the "some overfitting" range with median gap 0.146, while CDWCE_a3 is higher at 0.255. LDAM_DRW is the outlier: median gap 0.816, with one seed at 1.101, which is severe overfitting. Early stopping also tells the story: BalancedSoftmax peaks around epoch 19-22/28-29, CORN around 18-30/33-37, while one CDWCE_a3 seed runs all the way to 35/35 and both LDAM seeds still overfit badly despite stopping.

## 7. Systemic Insights & Hypotheses

The overarching story is that the corrected split turned this from a "find the best aggregate QWK run" exercise into a long-tail boundary problem. Both the earlier post-hoc logit adjustment win and the new BalancedSoftmax jump point in the same direction: the backbone already has enough information to help on minority cases, but the decision rule is too biased toward neutral predictions.

Two hypotheses look strongest. First, class-prior correction matters more than margin shaping on this dataset size; that explains why BalancedSoftmax moved both recall_-1 and hedging while LDAM_DRW mostly increased train/val gap. Second, Hedonism and Security rely on implicit, situational cues rather than explicit value words, so the model under-calls them unless the text makes enjoyment or safety conflict unusually obvious.

## 8. Actionable Recommendations

1. Finish the corrected-split BalancedSoftmax family with seed 33. The current two-seed median already leads on QWK 0.370, recall_-1 0.309, and minority recall 0.424, but the family is still provisional. Watch whether the 3-seed median holds above CDWCE_a3 without calibration dropping below about 0.70.
2. Run a calibration-only follow-up on BalancedSoftmax, not another loss sweep first. Its calibration 0.720 is good but well below CORN's 0.818. A validation-only temperature or ordinal-aware calibration pass is the cheapest way to test whether we can keep the tail gains while reducing confidence mismatch.
3. Prioritize data-centric hard-negative lift for Hedonism and Security, with Power as the volatility sidecar. The corrected split has only 14/13 non-zero Hedonism personas in val/test and 11/9 for Power, which is too thin for stable boundary learning. Watch per-dimension QWK and recall_-1, not just global QWK.
4. De-prioritize LDAM_DRW unless a single rescue run can cut `gap_at_best` below 0.5. Right now it is the only active family in severe overfitting territory and still trails the frontier on both QWK and recall_-1.

Research check: I cross-checked these recommendations against primary sources on class-prior correction and calibration. Menon et al.'s [Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314) and Ren et al.'s [Balanced Meta-Softmax](https://arxiv.org/abs/2007.10740) both support the idea that prior-aware logits should be the next lever when minority classes are suppressed by dominant-label bias. Cao et al.'s [LDAM-DRW](https://arxiv.org/abs/1906.07413) makes LDAM's logic plausible in principle, but the current runs suggest it is a poor small-data fit here. Kim et al.'s [Calibration of Ordinal Regression Networks](https://arxiv.org/abs/2410.15658) supports doing a calibration-focused follow-up before inventing another new loss.

## 9. Summary Verdict

Best config today: `BalancedSoftmax` is the provisional corrected-split leader, but `CDWCE_a3` remains the strongest completed 3-seed family. The key weakness is still value-specific: Hedonism and Security are under-modeled, and the overall system remains too willing to hedge toward neutral. The highest-leverage next experiment is to complete BalancedSoftmax with seed 33 on the corrected split; that single run will tell us whether the active frontier has genuinely moved or whether the current jump is an incomplete-family mirage.
