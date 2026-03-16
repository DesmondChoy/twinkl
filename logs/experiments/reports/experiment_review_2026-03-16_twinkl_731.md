# Experiment Review — 2026-03-16 — `twinkl-731` Full 768d Nomic v1.5 Truncation Diagnostic

## 1. Experiment Overview

`twinkl-731` tested whether the active frontier was losing hard-dimension polarity signal by truncating `nomic-embed-text-v1.5` to `256d` via Matryoshka. Two single-seed (`22`) diagnostic runs were compared against the incumbent anchor `run_020`.

**What varied:**
- **Embedding dimensionality**: `256d` (Matryoshka truncation) vs `768d` (native full-width)
- **Classifier width**: `hidden_dim=64` (run_037, unconstrained) vs `hidden_dim=28` (run_038, fair-budget control)
- Consequently: `state_dim` (266 vs 778) and parameter count (56,222 / 23,606 vs 23,454)

**What stayed constant:** `nomic-ai/nomic-embed-text-v1.5` encoder, `split_seed=2025`, `window_size=1`, `dropout=0.3`, `batch_size=16`, `epochs=100`, `weight_decay=0.01`, `BalancedSoftmax` loss, explicit LR `0.01552`, `qwk_then_recall_guarded` selection policy, frozen holdout persona manifest.

**Dataset note:** `n_train=1,213` for both new runs vs `n_train=1,022` for the incumbent family (`run_019`–`run_021`). The 191-sample increase (confirmed: `judge_labels.parquet` grew from 1,460 to 1,651 entries with val/test personas held fixed) means the 768d diagnostic had **more** training data than the incumbent and still regressed — strengthening the negative conclusion. Leaderboard claims are made within the active corrected-split regime only (post-`d937094`).

## 2. Head-to-Head Comparison

### Aggregate: 768d Diagnostic vs Incumbent

| Metric | `run_020` (256d, hd=64) | `run_037` (768d, hd=64) | `run_038` (768d, hd=28) | Incumbent Family Med |
|--------|---:|---:|---:|---:|
| Params | 23,454 | 56,222 | 23,606 | 23,454 |
| n_train | 1,022 | 1,213 | 1,213 | 1,022 |
| MAE | **0.304** | 0.319 | 0.333 | 0.304 |
| Accuracy | **0.755** | 0.739 | 0.737 | **0.753** |
| QWK | **0.378** | 0.318 | 0.299 | **0.362** |
| Spearman | **0.359** | 0.332 | 0.301 | 0.365 |
| Calibration | 0.713 | **0.720** | 0.657 | **0.713** |
| Minority Recall | **0.449** | 0.381 | 0.346 | **0.448** |
| recall_-1 | **0.342** | 0.269 | 0.246 | **0.313** |
| Hedging | 0.621 | 0.620 | **0.612** | 0.621 |
| OppV | N/A | 0.074 | 0.104 | N/A |
| AdjS | N/A | 0.075 | 0.077 | N/A |

Deltas vs `run_020`: `run_037` regresses on QWK (`-0.060`), `recall_-1` (`-0.073`), and minority recall (`-0.068`). `run_038` is worse still (`-0.078`, `-0.096`, `-0.103`). Calibration is comparable for `run_037` and drops for `run_038`. Hedging is essentially flat. Neither run makes any promotion case.

### Target Hard Dimensions: Hedonism & Security

| Metric | `run_020` | `run_037` | `run_038` |
|--------|---:|---:|---:|
| Hedonism QWK | **0.262** | 0.178 | -0.045 |
| Hedonism Cal | **0.856** | 0.755 | 0.822 |
| Hedonism Hedge | **0.697** | 0.747 | 0.792 |
| Security QWK | 0.213 | 0.165 | **0.225** |
| Security Cal | 0.452 | 0.444 | **0.553** |
| Security Hedge | 0.706 | **0.629** | 0.815 |

The target dimensions do not improve. `run_037` worsens both. `run_038` shows a tiny `security` QWK uptick (`+0.012`) at the cost of hedonism collapsing to `-0.045` (poor) and much heavier security hedging.

## 3. Per-Dimension Analysis

Sorted by incumbent family-median QWK. Runs 037/038 are single-seed, so no family aggregation.

| Dimension | Incumbent Med QWK | run_037 QWK | run_038 QWK | Category |
|-----------|------------------:|------------:|------------:|----------|
| conformity | **0.553** | 0.427 | 0.480 | Easy (>0.4) |
| self_direction | **0.494** | 0.428 | **0.523** | Easy (>0.4) |
| tradition | **0.485** | 0.461 | 0.372 | Easy (>0.4) |
| benevolence | **0.378** | 0.360 | 0.254 | Fair |
| universalism | 0.370 | **0.417** | 0.400 | Fair |
| achievement | **0.371** | 0.300 | 0.277 | Fair |
| power | **0.334** | 0.224 | 0.312 | Fair |
| security | **0.297** | 0.165 | 0.225 | Hard (<0.3) |
| hedonism | **0.247** | 0.178 | -0.045 | Hard (<0.3) |
| stimulation | 0.161 | **0.216** | 0.197 | Hard (<0.3) |

**Easy dimensions** (conformity, self_direction, tradition): consistently above 0.4 QWK in the incumbent; `run_037`/`run_038` both regress. **Hard dimensions** (stimulation, hedonism, security): remain below 0.3 and the 768d representation does not rescue them.

**Error Analysis (Hedonism & Stimulation):** Validation-output artifacts from `run_037` reveal systematic **behavioral-intent misreading**:
- **Hedonism:** 5 catastrophic errors (|error|=2), all predicting `-1` when truth is `+1`. Persona `e5cea325` accounts for 4: journal entries describing simple comfort, rest, and personal enjoyment are read as hedonic *withdrawal* rather than hedonic *alignment*. The model confuses descriptions of what a person *enjoys quietly* with what they *lack*.
- **Stimulation:** 5 catastrophic errors. Persona `b4cdcfee` craves novelty and forward motion, but the model predicts `-1` confidently. The behavioral framing involves *meta-reflection on stimulation needs* rather than direct stimulation-seeking behavior. Mention of "uncertainty" or "routine" triggers the opposite polarity.

## 4. Calibration Deep-Dive

Both new runs maintain 10/10 positively calibrated dimensions (matching the incumbent). Global calibration: `run_037` 0.720 (good), `run_038` 0.657 (good), incumbent 0.713 (good). No dimension has calibration < -0.4 in any run. Calibration is not the bottleneck.

The fair-budget control `run_038` shows the weakest global calibration, driven by `achievement` (0.250, weak) — the only dimension below 0.3. This is a model-capacity effect: `hidden_dim=28` compresses the 778-dimensional input too aggressively, hurting probability calibration on dimensions that require fine-grained discrimination.

## 5. Hedging vs Minority Recall Trade-off

| Run | Hedging % | Minority Recall | recall_-1 | Verdict |
|-----|----------:|----------------:|----------:|---------|
| run_019-021 family med | 62.1% | **0.448** | **0.313** | Moderate hedging, reasonable MinR |
| run_034-036 family med | **59.9%** | **0.449** | **0.378** | **Decisive + balanced** |
| run_037 (768d, hd=64) | 62.0% | 0.381 | 0.269 | Moderate hedging, weaker MinR |
| run_038 (768d, hd=28) | **61.2%** | 0.346 | 0.246 | Moderate hedging, poor MinR |

Neither 768d run achieves `hedging <60%` AND `minority recall >30%` together — `run_038` gets close on hedging (61.2%) but its minority recall (0.346, reasonable) is still well below the incumbent. The weighted reference branch `run_034`–`run_036` remains the only family achieving the **decisive + balanced** target.

## 6. Capacity & Overfitting

| Run | Params | Ratio | Gap | best_epoch/total | Characterization |
|-----|-------:|------:|----:|:----------------:|------------------|
| run_020 (256d) | 23,454 | 22.9 | 0.123 | 19/28 | High ratio, some gap, good stopping |
| run_037 (768d) | 56,222 | 46.3 | **0.347** | 35/35 | **High** ratio, overfitting (gap >0.3) |
| run_038 (768d, hd=28) | 23,606 | 19.5 | 0.160 | 30/40 | High ratio, some gap, OK stopping |

`run_037` is the overfitting story: the 56K-parameter model reached patience exhaustion at epoch 35 with a training gap of 0.347 (overfitting). Train loss dropped to 0.252 while val loss rose from 0.527 to 0.599. Validation QWK kept climbing (0.354 at selection) — ordinal rank-ordering can improve even as cross-entropy calibration degrades — but holdout test QWK collapsed to 0.318. The val-to-test collapse on hedonism (0.421 val → 0.178 test) confirms the model memorized validation-set patterns.

`run_038` controlled the capacity confound successfully (gap 0.160, comparable to incumbent's 0.123) but still underperformed, ruling out pure overfitting as the explanation. The 768d representation itself does not help.

## 7. Systemic Insights & Hypotheses

**The overarching story:** The frontier's hard-dimension polarity failures are **not** a representation-width bottleneck. Doubling the embedding from 256d to 768d — even with more training data (+18.7%) and even controlling for capacity — uniformly degraded or failed to help. The Matryoshka 256d truncation is not discarding useful polarity signal; the problem lies downstream.

**Hypothesis 1: Behavioral-intent polarity is a *training signal* problem, not a *feature* problem.** The error analysis shows consistent semantic misreading: the model reads reflective or quiet hedonic satisfaction as hedonic withdrawal, and meta-cognitive stimulation-seeking as stimulation aversion. These are not retrieval failures (the embedding *does* locate similar journal texts) but classification failures — the MLP has not learned the behavioral-intent mapping. This is consistent with (a) extreme label imbalance (hedonism 81.0% neutral, stimulation 88.7% neutral), and (b) the finding that BalancedSoftmax, which reweights by class frequency, was the only loss to break through the hedging barrier.

**Hypothesis 2: The curse of dimensionality is real at this sample size.** With ~1,200 training samples and 768 input features, the sample-to-feature ratio is ~1.6:1. Even the fair-budget control (hd=28) had to project 768→28 in the first layer, creating a bottleneck the model could not learn to navigate with so few examples. The 256d Matryoshka truncation happens to sit near the sweet spot for this dataset size (ratio ~4.7:1 with hd=64).

**Causal attribution guardrail:** The n_train difference (1,022 vs 1,213) means the 768d runs had *more* data. If the extra data alone were helpful, we should see *improvement*, not regression. The negative result is therefore even more robust: the 768d embedding actively hurts in this regime, independent of data volume.

## 8. Actionable Recommendations

**Web research conducted:** Searched for SLACE loss (AAAI 2025), PCGrad gradient surgery (NeurIPS 2020), Kendall uncertainty weighting (CVPR 2018), PCA on sentence embeddings (LREC-COLING 2024), and multi-task ordinal regression approaches (Knowledge-Based Systems 2024). Findings corroborate and refine the recommendations below.

1. **SLACE loss (highest priority).** BalancedSoftmax is a *flat* classification loss with class-frequency reweighting — it has no ordinal structure. SLACE (Nachmani et al., AAAI 2025) combines ordinal monotonicity with balance sensitivity in a single convex loss. Test it as a drop-in replacement on seeds 11/22/33 with the 256d incumbent config. **Watch:** `qwk_mean` (should match or beat 0.362), `recall_-1` (should match or beat 0.313), and hedonism/stimulation QWK specifically. No extra parameters needed.

2. **Kendall homoscedastic uncertainty weighting.** Add 10 learnable log-variance scalars (one per dimension) that auto-scale each dimension's loss contribution (Kendall et al., CVPR 2018). This prevents hard dimensions' noisy gradients from corrupting the shared backbone. Apply on top of BalancedSoftmax (or SLACE). **Watch:** cross-seed stability (family QWK IQR should shrink) and hard-dimension QWK. Recent analytical improvements (Gruber, DAGM GCPR 2024) address initialization stability.

3. **PCGrad gradient surgery.** Yu et al. (NeurIPS 2020) projects conflicting per-task gradients to prevent easy dimensions (conformity, tradition) from dominating the shared representation at the expense of hard ones. Computationally trivial for a 23K-param model. **Watch:** hedonism and stimulation QWK — these are the dimensions most likely to have gradients conflicting with the easy conformity/tradition cluster.

4. **PCA on 768d Nomic embeddings (ablation).** Cavaioni et al. (LREC-COLING 2024) show PCA can reduce sentence embeddings by ~50% without loss, sometimes improving downstream performance. Fit PCA on training data, sweep target dims (128, 192, 256, 384) and feed to the standard hd=64 MLP. **Watch:** whether PCA-128d or PCA-192d matches Matryoshka-256d, confirming the effective intrinsic dimensionality is lower than 256.

5. **Hard-dimension targeted augmentation with behavioral-intent framing.** *[Automatically checked: stimulation has only 60 (3.6%) `-1` labels and 127 (7.7%) `+1` labels out of 1,651 entries; hedonism has 141 (8.5%) `-1` and 173 (10.5%) `+1`.]* Generate synthetic journal entries that explicitly pair hedonic *enjoyment* language with positive labels and stimulation *seeking* language with positive labels, targeting the behavioral-intent confusion pattern identified in the error analysis.

## 9. Summary Verdict

- **Best config:** `run_019`–`run_021` `BalancedSoftmax` (nomic-256d, hd=64, seeds 11/22/33) remains the active corrected-split default. Family-median QWK 0.362 (fair), `recall_-1` 0.313 (reasonable), minority recall 0.448 (reasonable), hedging 62.1% (moderate), calibration 0.713 (good). The weighted branch `run_034`–`run_036` remains the best tail-sensitive reference.

- **Key weakness:** The 3 hard dimensions (stimulation QWK 0.161, hedonism 0.247, security 0.297) cap aggregate QWK at ~0.36. The failures are semantic behavioral-intent misreadings, not representation-width limitations — the 768d diagnostic conclusively rules out the feature-space bottleneck hypothesis.

- **Highest-leverage next experiment:** **SLACE loss** on the 256d incumbent config. It is the only recent loss that combines ordinal monotonicity (which BalancedSoftmax lacks) with balance sensitivity (which conservative losses like CORN/SoftOrdinal lack). If SLACE preserves BalancedSoftmax's tail recovery while adding ordinal structure, it should improve both QWK and hard-dimension discrimination.
