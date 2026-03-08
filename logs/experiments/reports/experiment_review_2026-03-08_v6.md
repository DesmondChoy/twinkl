# Experiment Review — 2026-03-08 — full frontier refresh

## 1. Experiment Overview

This review covers `run_001`-`run_024`, but leaderboard claims remain split-aware. `run_001`-`run_015` are the historical pre-`d937094` regime; `run_016`-`run_024` are the active corrected-split regime. Historical work changed encoder family (`MiniLM-384d` → `nomic-256d`), window size (`3` → `1`/`2`), hidden size (`32/64/128/256`), LR policy, and loss head. The active regime keeps the backbone fixed and varies only loss family plus one targeted-data branch.

Corrected-split constants are stable: `nomic-ai/nomic-embed-text-v1.5` truncated to 256d, `window_size=1`, `hidden_dim=64`, `dropout=0.3`, `split_seed=2025`, `state_dim=266`, and no truncation. The original corrected-split frontier (`run_016`-`run_021`) uses `1022/217/221` train/val/test rows. The `twinkl-681.5` targeted branch (`run_022`-`run_024`) keeps the same frozen holdout but increases train rows to `1117`. Active-board claims below are therefore made only within the corrected-split regime.

## 2. Head-to-Head Comparison

### Current Frontier (Post-`d937094`, corrected split; family medians)

| Metric | BalancedSoftmax `run_019-021` | CDWCE_a3 `run_016-018` | BalancedSoftmax + targeted batch `run_022-024` | SoftOrdinal `run_016-018` | CORN `run_016-018` |
|---|---:|---:|---:|---:|---:|
| MAE | 0.304 | 0.229 | 0.313 | 0.220 | **0.219** |
| Accuracy | 0.753 | 0.800 | 0.748 | 0.807 | **0.811** |
| QWK | **0.362** | 0.353 | 0.349 | 0.346 | 0.315 |
| Spearman | 0.365 | **0.365** | 0.344 | 0.354 | 0.356 |
| Calibration | 0.713 | 0.762 | 0.687 | 0.781 | **0.818** |
| Minority Recall | **0.448** | 0.276 | 0.434 | 0.283 | 0.273 |
| `recall_-1` | 0.313 | 0.104 | **0.342** | 0.077 | 0.089 |
| Hedging | 0.621 | 0.804 | **0.619** | 0.796 | 0.801 |

`BalancedSoftmax` remains the active default because it keeps the best median QWK 0.362 (fair) while already recovering minority labels much better than the conservative baselines. The targeted-data branch is now a real secondary candidate rather than just a side note: its median `recall_-1` 0.342 is better, but its QWK 0.349 and calibration 0.687 are both worse. `LDAM_DRW` stayed off the board because its provisional two-seed family only reached median QWK 0.344 with severe overfitting.

### Historical Frontier (Pre-`d937094`, archival only)

| Metric | `run_010 CORN` | `run_015 CDWCE_a3` | `run_014 SoftOrdinal` |
|---|---:|---:|---:|
| MAE | 0.206 | **0.203** | 0.204 |
| Accuracy | 0.821 | 0.822 | **0.825** |
| QWK | **0.434** | 0.402 | 0.388 |
| Spearman | **0.407** | 0.384 | 0.394 |
| Calibration | **0.835** | 0.755 | 0.801 |
| Minority Recall | 0.285 | 0.259 | **0.288** |
| `recall_-1` | **0.089** | 0.056 | 0.075 |
| Hedging | **0.820** | 0.852 | **0.820** |

The historical board is still useful regression context, but it is not comparable to the corrected-split frontier.

## 3. Per-Dimension Analysis

| Dimension | Mean QWK | Std | Read |
|---|---:|---:|---|
| hedonism | 0.021 | 0.136 | hardest + volatile |
| security | 0.249 | 0.047 | hard |
| stimulation | 0.273 | 0.090 | hard + volatile |
| power | 0.297 | 0.121 | hard + volatile |
| benevolence | 0.314 | 0.063 | middling |
| achievement | 0.386 | 0.052 | middling |
| universalism | 0.445 | 0.093 | easy but volatile |
| tradition | 0.463 | 0.041 | easy |
| self_direction | 0.519 | 0.041 | easy |
| conformity | 0.527 | 0.043 | easy |

Checkpoint-level error analysis used `run_023_BalancedSoftmax`, not `run_020`, because it is the highest-QWK current-data run whose validation split can still be replayed exactly from today’s workspace via the frozen `twinkl-681.5` holdout manifest. Its replayed validation QWK was 0.415 versus logged selection QWK 0.417, so the examples are faithful.

`Hedonism` misses were mostly `target=+1` entries predicted as `-1`: protecting a Saturday after caregiving obligations, decompressing alone after an exhausting risk meeting, and sitting with the emotional payoff of a promotion. The model appears to overweight sacrifice, duty, or achievement cues and under-read quieter pleasure or relief signals. `Security` misses were polarity-confused: reflective stability-seeking entries were predicted as `-1`, while one supplier-risk entry labeled `-1` was predicted `+1`, suggesting the model latches onto explicit safety words and misses whether the broader narrative is about stability or fear-driven constriction.

Current split support explains part of this. Under the frozen holdout, validation/test have only `14/13` non-zero Hedonism personas and `16/18` non-zero Security personas; Power is even thinner at `11/9`, which explains its volatility even after the targeted batch.

## 4. Calibration Deep-Dive

Calibration is still the healthiest part of the corrected-split stack. Every active run has `10/10` positively calibrated dimensions, and no active dimension falls below the deployment-risk threshold of `-0.4`. `CORN` remains the calibration anchor at 0.818 (good), followed by `SoftOrdinal` 0.781, `CDWCE_a3` 0.762, `BalancedSoftmax` 0.713, and the targeted branch 0.687. That means the targeted batch paid a real calibration tax, but not a dangerous one.

## 5. Hedging vs Minority Recall Trade-off

| Run + Loss | Hedging % | Minority Recall | Verdict |
|---|---:|---:|---|
| BalancedSoftmax `run_019-021` | 62.1 | 0.448 | closest to balanced |
| BalancedSoftmax + targeted `run_022-024` | 61.9 | 0.434 | closest to balanced |
| SoftOrdinal `run_016-018` | 79.6 | 0.283 | over-hedged |
| LDAM_DRW `run_019-020` | 79.4 | 0.285 | over-hedged |
| CORN `run_016-018` | 80.1 | 0.273 | over-hedged |
| CDWCE_a3 `run_016-018` | 80.4 | 0.276 | over-hedged |

No family achieved the `hedging < 60%` and `minority recall > 30%` target, so there is still no truly decisive + balanced configuration.

## 6. Capacity & Overfitting

Active capacity is effectively fixed: the corrected-split families all sit in a high param/sample regime around `21.0-23.0`. The differentiator is overfitting. `SoftOrdinal` has the healthiest median gap at 0.027 (good), `CORN` stays good at 0.070, `BalancedSoftmax` shows some overfitting at 0.123, and the targeted branch worsens to 0.216. `CDWCE_a3` is higher again at 0.255. `LDAM_DRW` is the red flag: median gap 0.816 (overfitting), so larger decision margins are not helping on this dataset. Early stopping fired appropriately for the safer families, but the targeted branch often peaked later, which fits its higher gap.

## 7. Systemic Insights & Hypotheses

The main story has not changed: this is now a boundary-shape and data-support problem more than a backbone-capacity problem. The best gains since `d937094` came from decision-rule changes (`BalancedSoftmax`, prior-aware post-hoc tuning), not from larger models or longer windows.

Two hypotheses look strongest. First, the targeted batch helped most where the label signal is explicit and sparse (`Power`, global `recall_-1`), but it did not fix dimensions whose positives are semantically quiet (`Hedonism`) or polarity-ambiguous (`Security`). Second, the model still over-indexes on salient surface cues such as obligation, achievement, and risk language; when a value is expressed as relief, permission, or uneasy safety, the classifier defaults toward the wrong pole or back to neutral.

## 8. Actionable Recommendations

Keep `run_019-021 BalancedSoftmax` as the default corrected-split base for now. It still has the best median QWK 0.362, while the targeted branch should be treated as a specialized recall-recovery variant, not a blanket replacement.

The next-step rollout is now tracked explicitly in beads under epic `twinkl-691`, with `P0` reserved for the highest-leverage post-merge work and `P1` reserved for contingent structure-aware ablations:

1. `[P0] twinkl-691.1` — Add circumplex diagnostics to the active VIF evaluation stack before changing training behavior. Use the current `src/vif/eval.py` + logger path, operate on expected scores or class probabilities rather than only hard labels, and export per-pair summaries for opposing and adjacent Schwartz dimensions. This makes theory violations measurable instead of inferred.
2. `[P0] twinkl-691.2` — Run the next frozen-holdout targeted data lift on `Hedonism` and `Security`, but shape it explicitly around circumplex conflict/adjacency scenarios. Focus on quiet restorative Hedonism positives, polarity-ambiguous Security positives, and tensions such as `Self-Direction <-> Security` and `Stimulation <-> Security`.
3. `[P0] twinkl-691.3` — Rebaseline the frontier after that lift by retraining the current BalancedSoftmax branch and running one `SoftOrdinal` comparator on the same frozen holdout. Treat this step as the calibration-conscious decision point: compare QWK, `recall_-1`, minority recall, hedging, calibration, and the new circumplex diagnostics, then keep a single active default.
4. `[P1] twinkl-691.4` — Only if the P0 steps still show meaningful opposite-pair or adjacency-structure errors, run one soft circumplex-regularizer ablation. Keep it config-gated and off by default; this should be a narrow test of whether a weak structural prior improves diagnostics without regressing QWK, recall, or calibration.
5. `[P1] twinkl-691.5` — Leave the circumplex-aware batch sampler for last, and treat explicit de-scope as a legitimate outcome. In the current stack it is the highest-risk idea because it changes minibatch priors while the frontier is already relying on prior-aware losses and calibration checks.

This ordering intentionally moves diagnostics and data before stronger training-time structure. The current evidence says the immediate problem is still boundary shape plus data support, especially on `Hedonism` and `Security`, rather than the lack of a circumplex-aware architecture.

Research check: recent primary sources support this sequencing. [Balanced Meta-Softmax](https://arxiv.org/abs/2007.10740) and [logit adjustment](https://arxiv.org/abs/2007.07314) still support prior-aware boundary correction as the first lever when minority classes are suppressed by dominant neutral labels. [An Overview of the Schwartz Theory of Basic Values](https://doi.org/10.9707/2307-0919.1116) and [Behavioral Signatures of Values in Everyday Behavior](https://doi.org/10.3389/fpsyg.2019.00281) justify adding circumplex-aware diagnostics and conflict-pair data because adjacent/opposing relationships are psychologically meaningful, but everyday behavior can still be value-ambivalent. The recent value-detection result [Do Schwartz Higher-Order Values Help Sentence-Level Human Value Detection? When Hard Gating Hurts](https://doi.org/10.2139/ssrn.6262579) argues against turning that structure into a hard constraint too early. For later ablations, [Calibration of Ordinal Regression Networks](https://arxiv.org/abs/2410.15658) supports treating ordinal calibration as a separate optimization target rather than assuming a new loss or regularizer will fix it automatically.

## 9. Summary Verdict

Best config: `BalancedSoftmax` (`run_019-021`) remains the active corrected-split leader because it keeps the best median QWK 0.362 while materially reducing hedging and lifting minority recovery. Key weakness: `Hedonism` and `Security` are still under-modeled, and the system remains slightly too neutral-biased to be called decisive. Highest-leverage rollout, now tracked in `twinkl-691`, is `P0` circumplex diagnostics → frozen-holdout Hedonism/Security conflict-pair data lift → BalancedSoftmax/SoftOrdinal rebaseline with calibration comparison, followed by `P1` regularizer and sampler ablations only if the earlier steps still leave meaningful structure errors.
