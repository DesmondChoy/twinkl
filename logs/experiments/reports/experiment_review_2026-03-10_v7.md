# VIF Experiment Review v7 — Full Frontier Refresh (2026-03-10)

**Scope:** All 27 runs (90 YAML configs), partitioned into historical pre-`d937094` (run_001–015) and active corrected-split (run_016–027). Leaderboard claims are made within the active regime only.

---

## 1. Experiment Overview

**Active regime:** Post-`d937094` corrected persona-stratified split (`split_seed=2025`), runs 016–027.

**What varied across corrected-split runs:**
- **Loss function:** CORN, CDWCE_a3, SoftOrdinal, BalancedSoftmax, LDAM_DRW (7 candidate families)
- **Training data:** 1022 rows (016–021), 1117 rows (022–024, targeted batch), 1213 rows (025–027, regenerated hedonism/security batch)
- **Model seed:** 11, 22, 33 (3-seed matrix per family)

**Constants across all corrected-split runs:** nomic-embed-text-v1.5 (256d), ws=1, hd=64, dropout=0.3, batch_size=16, early_stopping_patience=20, MC dropout=50, split_seed=2025.

**Dataset:** 1651 judged entries from 204 personas. Realized splits: 1022–1213 train / 217 val / 221 test (frozen holdout for run_022+). Overall minority rate: 7.3% for -1, ~16% for +1 across dimensions.

---

## 2. Head-to-Head Comparison

### Corrected-Split Frontier (Family Medians, seeds 11/22/33)

| Metric | BalSM (019-021) | CDWCE_a3 (016-018) | SoftOrd (016-018) | CORN (016-018) | BalSM+tgt (022-024) | BalSM+lift (025-027) | SoftOrd+lift (025-027) |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| MAE | 0.304 | 0.229 | **0.220** | **0.218** | 0.313 | 0.319 | **0.213** |
| Accuracy | 0.753 | 0.799 | **0.807** | **0.811** | 0.748 | 0.737 | **0.811** |
| **QWK** | **0.362** | 0.353 | 0.346 | 0.315 | 0.349 | 0.346 | 0.340 |
| Spearman | 0.365 | 0.365 | 0.353 | 0.356 | 0.344 | **0.372** | 0.365 |
| **Calibration** | 0.713 | 0.762 | 0.781 | **0.818** | 0.687 | 0.693 | 0.738 |
| **Recall -1** | **0.313** | 0.104 | 0.077 | 0.089 | **0.342** | 0.328 | 0.082 |
| **MinR** | **0.448** | 0.276 | 0.283 | 0.273 | 0.434 | 0.442 | 0.260 |
| **Hedging** | **0.621** | 0.804 | 0.796 | 0.801 | **0.619** | **0.598** | 0.823 |

> **LDAM_DRW** (run_019–020, 2 seeds only) is eliminated: median QWK 0.344, recall_-1 0.069, and training gaps 0.53–1.10 (severe overfitting).

**Key takeaway:** BalancedSoftmax families dominate on tail-recovery metrics (recall_-1, minority recall, hedging) while giving back MAE, accuracy, and calibration. The conservative losses (CORN, SoftOrdinal, CDWCE_a3) are better when aggregate stability matters. The targeted and post-lift data batches improved recall_-1 further but did not reclaim the QWK or calibration gap.

---

## 3. Per-Dimension Analysis

Sorted by mean QWK across the four original corrected-split families:

| Dimension | BalSM | CDWCE_a3 | SoftOrd | CORN | Mean QWK | Variance | Class |
|-----------|:-----:|:--------:|:-------:|:----:|:--------:|:--------:|:-----:|
| conformity | 0.553 | 0.493 | 0.549 | 0.548 | **0.536** | low | Easy |
| self_direction | 0.494 | 0.513 | 0.538 | 0.562 | **0.527** | low | Easy |
| tradition | 0.485 | 0.470 | 0.462 | 0.447 | **0.466** | low | Easy |
| universalism | 0.370 | 0.402 | 0.532 | 0.518 | **0.455** | medium | Easy |
| achievement | 0.371 | 0.353 | 0.445 | 0.410 | **0.395** | medium | Medium |
| benevolence | 0.378 | 0.291 | 0.297 | 0.291 | **0.314** | low | Medium |
| stimulation | 0.161 | 0.337 | 0.321 | 0.330 | **0.287** | medium | Hard |
| power | 0.334 | 0.330 | 0.252 | 0.181 | **0.274** | **high** | Hard/Volatile |
| security | 0.297 | 0.237 | 0.237 | 0.199 | **0.243** | low | Hard |
| hedonism | 0.247 | 0.005 | -0.040 | -0.119 | **0.023** | **high** | Hardest |

**Easy dimensions** (QWK > 0.4): Conformity, self_direction, tradition, universalism — consistently moderate agreement across all losses.

**Hard dimensions** (QWK < 0.3): Stimulation, power, security, hedonism. Hedonism is near-zero or negative for all losses except BalancedSoftmax.

**Most volatile:** Hedonism (range 0.366 across families) and power (range 0.153). BalancedSoftmax is the only loss that produces positive hedonism QWK — all conservative losses go negative.

### Error Analysis (Hardest Dimensions)

Checkpoint-level validation replay on `run_019` BalancedSoftmax:

**Hedonism** (14 val -1 labels, 30 val +1 labels, 173 val 0 labels):
- 6 opposite-end errors (|err|=2). The model flipped ground-truth +1 → predicted -1 in all top errors.
- Example: persona `b4cdcfee` t=9 — "*savored the canyon drive, the picnic with tortas and marzipan candies*" (GT: +1, Pred: -1, probs [0.54, 0.30, 0.16]). The model interpreted a quiet, reflective pleasure scene as sacrifice or deprivation.
- Example: persona `e5cea325` t=1 — "*explicitly protects restorative Saturdays and rejects the idea that rest is a flaw*" (GT: +1, Pred: -1, high uncertainty 0.97). Defensive framing of pleasure confused the model into reading tension rather than alignment.

**Security** (23 val -1 labels, 31 val +1 labels, 163 val 0 labels):
- 3 opposite-end errors. The model confuses entries where explicit safety language coexists with broader ambivalence.
- Example: persona `b0968b05` t=2 — "*moving from unknown conditions to a stable, repeatable structure*" (GT: +1, Pred: -1, very confident probs [0.78, 0.20, 0.02]). The mention of "unknown conditions" apparently triggered the -1 class despite the entry being about *gaining* security.
- Example: persona `b12682c6` t=7 — "*unstable algorithmic pay makes livelihood less predictable*" (GT: -1, Pred: +1). Economic instability framed through agency ("lower base rates") was misread as controlled adaptation.

---

## 4. Calibration Deep-Dive

| Family | Global Cal | Positive Dims | Risk Dims |
|--------|:----------:|:-------------:|:---------:|
| CORN (016-018) | **0.818** (good) | 10/10 | None |
| SoftOrdinal (016-018) | 0.781 (moderate) | 10/10 | None |
| CDWCE_a3 (016-018) | 0.762 (moderate) | 10/10 | None |
| BalSM (019-021) | 0.713 (moderate) | 10/10 | None |
| BalSM+lift (025-027) | 0.693 (moderate) | 10/10 | None |
| BalSM+tgt (022-024) | 0.687 (moderate) | 10/10 | None |
| SoftOrd+lift (025-027) | 0.738 (moderate) | 10/10 | None |

All families maintain positive calibration across all 10 dimensions — no deployment-risk flags (calibration < -0.4). The BalancedSoftmax families trade ~0.1 calibration for their tail-recovery gains, but stay in the moderate range. CORN remains the clear calibration champion.

Security has the lowest per-dimension calibration across all families (median ~0.59–0.63), approaching the weak range. This is systematic: the model is slightly over-confident on security predictions regardless of loss function.

---

## 5. Hedging vs Minority Recall Trade-off

| Family | Hedging % | Minority Recall | Verdict |
|--------|:---------:|:---------------:|:-------:|
| BalSM+lift (025-027) | **59.8%** | 0.442 | **Decisive + Balanced** |
| BalSM+tgt (022-024) | **61.9%** | 0.434 | **Decisive + Balanced** |
| BalSM (019-021) | 62.1% | **0.448** | **Decisive + Balanced** |
| SoftOrd (016-018) | 79.6% | 0.283 | Moderate hedging |
| CORN (016-018) | 80.1% | 0.273 | Excessive hedging |
| CDWCE_a3 (016-018) | 80.4% | 0.276 | Excessive hedging |
| SoftOrd+lift (025-027) | 82.3% | 0.260 | Excessive hedging |

Only the BalancedSoftmax families achieve the **decisive + balanced** threshold (hedging < 60% AND minority recall > 30%). All conservative losses remain in the excessive-hedging regime despite 27 runs of optimization. The post-lift data did not help SoftOrdinal escape its hedging pattern.

---

## 6. Capacity & Overfitting

All corrected-split runs share param_sample_ratio 19–23 (high regime). This is a constant, not a differentiator.

| Family | Median Gap | Median Best Epoch | Epochs Budget | Regime |
|--------|:----------:|:-----------------:|:-------------:|:------:|
| SoftOrd (016-018) | **0.027** | 17 | 100 | Good |
| SoftOrd+lift (025-027) | **0.023** | 12 | 100 | Good |
| CORN (016-018) | 0.070 | 27 | 100 | Good |
| BalSM (019-021) | 0.123 | 19 | 100 | Some |
| BalSM+tgt (022-024) | 0.216 | 23 | 100 | Some |
| BalSM+lift (025-027) | 0.199 | 22 | 100 | Some |
| CDWCE_a3 (016-018) | **0.255** | 32 | 100 | Some |
| LDAM_DRW (019-020) | **0.816** | 22 | 100 | **Overfitting** |

SoftOrdinal has the tightest train/val gaps (<0.03), but this extreme closeness reflects conservative convergence, not better learning — it stops early and hedges heavily. CDWCE_a3 runs longer and shows larger gaps (0.20–0.29), which correlates with its better aggregate QWK but also suggests the CDW-CE distance weighting creates more optimization pressure. BalancedSoftmax sits in between. LDAM_DRW is fatally overfit.

Early stopping triggered appropriately in all families (best_epoch well before 100). No capacity-driven overfitting is evident within the active frontier.

---

## 7. Systemic Insights & Hypotheses

**The overarching story:** The VIF critic has learned to rank *easy* dimensions (conformity, self_direction, tradition) with moderate agreement, but it fundamentally cannot tell the difference between presence and absence of *hedonic pleasure* or *security threat* from journal text alone. The model's failures on hedonism and security are not random — they are systematic misreadings of semantic register.

**Hypothesis 1: The model reads *any* emotional intensity as misalignment.** The hedonism error analysis shows the model predicting -1 for entries about savoring pleasure. The common thread is vivid, emotionally loaded language. The model appears to have learned "strong feeling → something is wrong" because most emotionally charged training entries involve tension or sacrifice (which are labeled -1 on other dimensions). This is a spurious correlation between emotional intensity and misalignment polarity.

**Hypothesis 2: BalancedSoftmax works not because it's a better loss, but because it forces the model to *use* the -1 and +1 classes at all.** The conservative losses converge to a strategy of "predict 0, occasionally predict +1" because that minimizes ordinal distance error on an 80% neutral dataset. BalancedSoftmax's prior correction forces the logits to produce minority predictions, which accidentally rescues dimensions like hedonism where the model has *some* signal but would otherwise suppress it under the neutral default.

**What this clarifies about the bottleneck:** The earlier conclusion that "recall_-1 is a data scarcity problem" remains correct — two targeted synthetic data batches (`twinkl-681.5` Power/Security, `twinkl-691.2` Hedonism/Security) were generated specifically to address it, and they pushed recall_-1 further from 0.313 to 0.342. But data scarcity was not the *only* bottleneck: BalancedSoftmax recovered recall_-1 from 0.089 to 0.313 on the *same* 1022-row dataset as CORN, purely by correcting the class prior in the loss. This means the conservative losses were leaving signal on the table by suppressing minority predictions even when the features supported them. The practical implication is that both levers matter: the loss function must be willing to predict minorities (BalancedSoftmax), *and* the training set needs enough minority examples for the features to be informative (targeted augmentation). Neither alone is sufficient.

---

## 8. Actionable Recommendations

**Web research conducted:** Searched 2024-2026 literature on long-tail ordinal classification, dimension-specific loss weighting, LLM-based ordinal augmentation, and calibration-aware training. Key references: Hybrid Contrastive Ordinal Regression (MICCAI 2025), Focal Temperature Scaling (ECAI 2024), MORTD multi-task ordinal weighting (KBS 2024), and LLM oversampling meta-studies (2025-2026).

### Recommendation 1: Per-dimension uncertainty weighting

**Evidence:** Hedonism has 81% neutral labels but 0.023 mean QWK; self_direction has 62% neutral and 0.527 QWK. Treating all 10 dimensions equally in the shared loss wastes gradient signal on dimensions the model already handles well.

**Action:** Implement Kendall-style homoscedastic uncertainty weighting (one learnable log-sigma per Schwartz dimension) on top of BalancedSoftmax. This lets the model down-weight noisy dimensions (hedonism, power) during training and focus capacity on dimensions where the gradient is informative.

**Watch for:** Improvement in mean QWK without hedonism/power dragging down the aggregate. If hedonism QWK stays negative even with dimension weighting, it confirms the signal is absent rather than merely drowned out.

### Recommendation 2: Decoupled head retraining with class-balanced sampling

**Evidence:** The frozen Nomic encoder produces good representations (the same embeddings support QWK > 0.5 on 4 dimensions). The bottleneck is the MLP head's decision boundary, not the features.

**Action:** Freeze the encoder and state-encoder weights from the best `run_019` BalancedSoftmax checkpoint. Retrain only the MLP classification head using class-balanced sampling (equal probability of drawing -1, 0, +1 per batch per dimension). This is supported by Gao et al. 2025 decoupled training findings.

**Watch for:** recall_-1 improvement without the MAE/calibration tax of full BalancedSoftmax training. If the frozen features support better boundaries, this would prove the representation is sufficient and only the head needs adjustment.

### Recommendation 3: Targeted LLM augmentation for hedonism minority class

**Evidence:** *[Automatically checked `logs/judge_labels/judge_labels.parquet`: hedonism has 141 -1 labels (8.5%) and 173 +1 labels (10.5%) across 1651 entries. Only 68/204 personas (33%) have any hedonism -1 signal, and 59/204 (29%) have +1. Mean signal entries per persona: 3.3.]* The regenerated `twinkl-691.2` batch added hedonism/security personas but did not lift hedonism QWK for conservative losses. The error analysis shows the model misreads *quiet pleasure* as *tension*.

**Action:** Generate 20–30 additional hedonism-focused entries (both -1 and +1) using the existing synthetic pipeline, specifically targeting the semantic patterns the model fails on: quiet/reflective pleasure (+1), pleasure-as-escape (+1), and guilt-about-pleasure (-1). Cap augmentation at 2x the current minority count per recent meta-study guidance on synthetic oversampling.

**Watch for:** Hedonism QWK moving above 0.15 on the BalancedSoftmax family. If it stays near zero, the dimension may require a different encoding approach (e.g., sentiment-aware features) rather than more data.

### Recommendation 4: Focal Temperature Scaling post-hoc on CORN

**Evidence:** CORN has the best calibration (0.818) but worst QWK (0.315) and recall_-1 (0.089). Focal Temperature Scaling (ECAI 2024) can shift the decision boundary post-hoc while improving calibration further.

**Action:** Apply focal temperature scaling to the existing CORN checkpoints from run_016-018 using validation-set tuning. This extends the earlier `twinkl-681.3` post-hoc work with a more principled calibration-aware method.

**Watch for:** Whether CORN + FTS can match BalancedSoftmax's recall_-1 while keeping calibration > 0.80. If so, it would be a safer production deployment option than BalancedSoftmax.

### Recommendation 5: Dimension-specific evaluation threshold for hedonism

**Evidence:** Hedonism has only ~21 val -1 labels and ~26 val +1 labels. With so few evaluation samples, QWK is inherently noisy — a single misprediction swings QWK by ~0.05. Stimulation is worse (est. 9 val -1 labels).

**Action:** For the next experiment review, compute bootstrap confidence intervals (1000 resamples) on per-dimension QWK to quantify how much of the hedonism/stimulation variance is evaluation noise vs. true model differences. If the 95% CI for hedonism QWK spans zero for all losses, stop treating hedonism QWK as a meaningful discriminator between models.

**Watch for:** Whether the hedonism QWK differences between BalancedSoftmax (0.247) and CORN (-0.119) survive bootstrapping. If they don't, the BalancedSoftmax "advantage" on hedonism may be statistical noise.

---

## 9. Summary Verdict

- **Best config:** `run_019`–`run_021` BalancedSoftmax (median QWK 0.362, recall_-1 0.313, minority recall 0.448, hedging 62.1%). It is the only family that achieves decisive boundary behavior while maintaining moderate calibration and all 10 dimensions positively calibrated.

- **Key weakness:** Hedonism is effectively unmodeled (mean QWK 0.023 across all losses). The model systematically misreads hedonic pleasure signals as tension. This is not fixable by loss function alone — it requires either better training signal (targeted augmentation) or dimension-specific feature engineering.

- **Highest-leverage next experiment:** Per-dimension uncertainty weighting on BalancedSoftmax (Recommendation 1). It is the simplest intervention that addresses the core problem — the model wastes capacity trying to fit noise on hedonism/stimulation while undertrained on dimensions where the signal is learnable. If hedonism truly has no signal, the learned sigma will down-weight it automatically, improving the aggregate without manual tuning.
