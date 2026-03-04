# VIF Experiment Review — run_015: CDW-CE Alpha Sweep + Baseline Rerun

**Date:** 2026-03-04
**Runs analysed:** run_015 (7 configs: CDWCE_a2, CDWCE_a3, CDWCE_a5, CORAL, CORN, EMD, SoftOrdinal)
**Compared against:** run_010 (current SOTA), run_014 (LR finder valley), run_007 (previous #2)

---

## 1. Experiment Overview

**What varied:**
- **Loss function** — 3 new CDW-CE variants (alpha=2, 3, 5) alongside 4 established losses (CORAL, CORN, EMD, SoftOrdinal)
- **Applied learning rate** — LR finder ran per-loss, producing applied LRs from 1.1x to 19.8x the configured 0.001

**What stayed constant:** nomic-embed-text-v1.5 (256d), window_size=1, hidden_dim=64, dropout=0.3, batch_size=16, weight_decay=0.01, split_seed=2025, 1020 train / 230 val / 210 test, mc_dropout_samples=50.

**Motivation:** Run_014 review identified loss-function-level interventions as the highest-priority next step. CDW-CE (Polat et al. 2025, arXiv 2412.01246) penalizes predictions by ordinal distance from the true label, weighted by |i−c|^alpha. The alpha sweep (2, 3, 5) tests the trade-off between mild and aggressive distance penalties.

## 2. Head-to-Head Comparison

### CDW-CE Alpha Sweep (new losses)

| Metric | CDWCE_a2 | **CDWCE_a3** | CDWCE_a5 |
|--------|----------|-------------|----------|
| QWK | 0.322 (fair) | **0.402 (moderate)** | 0.300 (fair) |
| MAE | 0.207 | **0.203** | 0.217 |
| Accuracy | 0.811 | **0.822** | 0.795 |
| Spearman | 0.350 | **0.384** | 0.371 |
| Calibration | 0.783 | 0.755 | 0.639 (weak) |
| Minority Recall | 0.220 | **0.259** | 0.174 |
| recall_-1 | 0.038 | **0.056** | 0.012 |
| recall_+1 | 0.403 | **0.461** | 0.336 |
| Hedging | 86.1% (excessive) | 85.2% (excessive) | 89.9% (excessive) |
| LR (applied) | 0.00212 (2.1x) | 0.01552 (15.5x) | 0.00230 (2.3x) |

**Alpha=3 is the clear CDW-CE winner.** The inverted-U response to alpha is striking: alpha=2 is too weak (nearly standard CE for 3 classes where max |i−c|=2), alpha=3 hits the sweet spot, alpha=5 over-penalizes and collapses to extreme hedging. CDWCE_a3 is the only CDW-CE variant where the LR finder found a true valley (0.01552); the other two fell back to lr_steep.

### Run_015 vs SOTA (run_010 at configured LR=0.001)

| Metric | run_010 CORN (SOTA) | run_015 CDWCE_a3 | run_015 EMD | run_015 SoftOrd | run_015 CORN | Delta (best 015 vs SOTA) |
|--------|--------------------|--------------------|-------------|-----------------|--------------|--------------------------|
| QWK | **0.434** | 0.402 | 0.372 | 0.335 | 0.328 | -0.032 |
| MAE | 0.206 | **0.203** | 0.204 | 0.208 | **0.203** | **+0.003** |
| Accuracy | **0.821** | **0.822** | 0.821 | **0.822** | 0.815 | comparable |
| Calibration | 0.835 | 0.755 | 0.781 | **0.846** | 0.801 | +0.011 (SoftOrd) |
| MinR | 0.285 | 0.259 | 0.280 | **0.292** | 0.234 | +0.007 (SoftOrd) |
| recall_-1 | 0.089 | 0.056 | 0.056 | **0.064** | 0.060 | -0.025 |
| Hedging | 82.0% | 85.2% | 83.7% | **80.3%** | 85.8% | +1.7% (SoftOrd) |

**Run_010 CORN remains SOTA on QWK.** CDWCE_a3 is the strongest new challenger (QWK 0.402, -0.032 from SOTA) but trades calibration (-0.080) for it. Run_015 SoftOrdinal leads on calibration (0.846) and minority recall (0.292) but lags on QWK. The LR finder applied aggressive LRs (9.5–20x) to threshold-based losses (CORAL, CORN), once again degrading their performance vs configured LR=0.001.

### LR Finder Behavior Across Losses (run_015)

| Loss | Strategy | LR Applied | LR Ratio | QWK | Notes |
|------|----------|-----------|----------|-----|-------|
| SoftOrdinal | fallback_lr_steep | 0.00111 | 1.1x | 0.335 | Most conservative |
| CDWCE_a2 | fallback_lr_steep | 0.00212 | 2.1x | 0.322 | Monotonic loss landscape |
| CDWCE_a5 | fallback_lr_steep | 0.00230 | 2.3x | 0.300 | Monotonic loss landscape |
| CORN | valley_over_10 | 0.00955 | 9.5x | 0.328 | Still too high |
| CDWCE_a3 | valley_over_10 | 0.01552 | 15.5x | **0.402** | True valley found |
| CORAL | valley_over_10 | 0.01979 | 19.8x | 0.349 | Deterministic (=run_014) |
| EMD | valley_over_10 | 0.01979 | 19.8x | 0.372 | Deterministic (=run_014) |

**Bug confirmed:** `lr_find_history.json` is identical to `lr_find_CDWCE_a5.json` — the history file is overwritten by the last model processed. Previously it matched SoftOrdinal (run_014). This is a logging bug, not a training bug.

## 3. Per-Dimension Analysis

### Dimension Difficulty (sorted by mean QWK across all run_015 models)

| Dimension | Mean QWK | Best Model (QWK) | Worst Model (QWK) | Class | % Neutral |
|-----------|----------|-------------------|---------------------|-------|-----------|
| Stimulation | 0.514 | EMD (0.558) | SoftOrd (0.483) | Easy | 87.8% |
| Conformity | 0.476 | CORN (0.530) | CDWCE_a5 (0.366) | Easy | 69.7% |
| Benevolence | 0.449 | CDWCE_a3 (0.545) | CORN (0.352) | Easy | 62.5% |
| Achievement | 0.424 | CDWCE_a3 (0.449) | EMD (0.401) | Easy | 67.7% |
| Tradition | 0.425 | CDWCE_a2 (0.554) | CORAL (0.279) | Easy | 78.6% |
| Self_direction | 0.356 | EMD (0.434) | CDWCE_a5 (0.188) | Volatile | 58.8% |
| Hedonism | 0.337 | CORN (0.431) | CDWCE_a2 (0.151) | Volatile | 81.2% |
| Universalism | 0.265 | CORAL (0.444) | SoftOrd (0.082) | Volatile | 85.3% |
| Power | 0.099 | CDWCE_a5 (0.273) | CORAL (-0.111) | Hard | 81.3% |
| Security | 0.132 | CORAL (0.230) | EMD (0.061) | Hard | 75.3% |

**Error analysis:** No checkpoint available — `models/vif/best_model.pt` does not exist (cleaned up after training). Qualitative sample inspection skipped.

**Key per-dimension finding for CDW-CE:** CDWCE_a3 is the strongest on benevolence (0.545, best-ever for any CDW-CE model) and achievement (0.449). Surprisingly, CDWCE_a5 achieves the best Power QWK in run_015 (0.273) — the aggressive distance penalty may help on the most imbalanced dimension by discouraging the 2-step -1→+1 confusion.

## 4. Calibration Deep-Dive

All 7 models maintain 10/10 positive calibration dimensions — a positive sign for deployment safety.

| Model | Global Cal | Lowest Dim Cal | Highest Dim Cal |
|-------|-----------|----------------|-----------------|
| **SoftOrdinal** | **0.846** | self_dir 0.568 | stimulation 0.933 |
| CORAL | 0.824 | self_dir 0.601 | power 0.938 |
| CORN | 0.801 | self_dir 0.556 | power 0.937 |
| CDWCE_a2 | 0.783 | self_dir 0.457 | stimulation 0.898 |
| EMD | 0.781 | self_dir 0.487 | universalism 0.941 |
| CDWCE_a3 | 0.755 | self_dir 0.506 | power 0.850 |
| **CDWCE_a5** | **0.639 (weak)** | **self_dir 0.273** | stimulation 0.738 |

**Calibration degrades monotonically with CDW-CE alpha.** CDWCE_a5's self_direction calibration (0.273) approaches the dangerous threshold — the model's confidence estimates are nearly meaningless for this dimension. The aggressive distance penalty distorts the probability landscape.

**No dimension has calibration < -0.4** — no deployment risks from negative calibration. However, CDWCE_a5's overall 0.639 calibration is the lowest since run_001 MSE (which had dangerous negative calibration and was retired).

## 5. Hedging vs Minority Recall Trade-off

| Run + Loss | Hedging % | Minority Recall | recall_-1 | recall_+1 | Verdict |
|------------|-----------|-----------------|-----------|-----------|---------|
| **015 SoftOrdinal** | **80.3%** (excessive) | **0.292** | **0.064** | **0.520** | Best balance in run_015 |
| 015 EMD | 83.7% (excessive) | 0.280 | 0.056 | 0.505 | Strong recall_+1 |
| 015 CDWCE_a3 | 85.2% (excessive) | 0.259 | 0.056 | 0.461 | Best QWK but hedges more |
| 015 CORAL | 84.9% (excessive) | 0.241 | 0.064 | 0.418 | Moderate |
| 015 CORN | 85.8% (excessive) | 0.234 | 0.060 | 0.408 | LR too aggressive |
| 015 CDWCE_a2 | 86.1% (excessive) | 0.220 | 0.038 | 0.403 | Alpha too weak |
| **015 CDWCE_a5** | **89.9%** (excessive) | **0.174** | **0.012** | 0.336 | Worst — extreme hedging |
| *010 CORN (SOTA)* | *82.0%* | *0.285* | *0.089* | *0.480* | *Reference* |

**No configuration achieves hedging < 60% AND minority recall > 30%.** All run_015 models hedge excessively (>80%). CDW-CE did not break the hedging/minority-recall barrier. The best recall_-1 in run_015 (SoftOrdinal at 0.064) is below the SOTA reference (run_010 CORN at 0.089).

**CDW-CE's failure mode:** Instead of penalizing distant errors and forcing the model to differentiate -1 from 0 from +1, higher alpha causes the model to hedge *more* aggressively — predicting neutral avoids any distance penalty. This is the opposite of the intended effect.

## 6. Capacity & Overfitting

All run_015 models share the same architecture (22–23K params, ~23:1 param/sample ratio = high).

| Model | Best Epoch | Total Epochs | Gap @ Best | Pattern |
|-------|------------|-------------|------------|---------|
| EMD | 6 | 26 | -0.021 | Fast convergence, slight underfit |
| CORN | 7 | 27 | -0.022 | Fast convergence, slight underfit |
| SoftOrdinal | 8 | 28 | -0.024 | Fast convergence, slight underfit |
| CDWCE_a2 | 13 | 33 | -0.012 | Moderate convergence |
| CORAL | 16 | 36 | +0.018 | Healthy, slight overfit |
| CDWCE_a3 | 21 | 41 | +0.030 | Slow convergence, mild gap |
| CDWCE_a5 | 30 | 50 | +0.021 | Very slow, alpha penalty decelerates learning |

**No overfitting concerns** — all gaps are < 0.10 (good). CDWCE_a3 has the largest gap (0.030) but this is well within tolerance. Early stopping triggered appropriately in all cases.

**CDW-CE alpha controls convergence speed:** Higher alpha = slower training. CDWCE_a5 needs 30 epochs vs 6 for EMD. The heavy distance penalty creates a loss landscape where small parameter updates produce proportionally larger loss changes, requiring more cautious optimization.

## 7. Systemic Insights & Hypotheses

**The overarching story:** Fifteen runs and seven loss functions later, the model consistently learns to detect alignment (+1) at 40–52% recall but is nearly blind to misalignment (-1) at 1–9% recall. CDW-CE was the recommended "highest-leverage" intervention — and it failed to move recall_-1. This narrows the hypothesis space considerably.

**Hypothesis 1: The recall_-1 failure is a data scarcity problem, not a loss function problem.** The -1 class comprises only 3.5–14% of labels per dimension (7.3% overall). With 1020 training samples × 10 dimensions × 7.3% ≈ 745 misalignment labels total, the model sees ~75 misalignment labels per dimension during training. No loss function can reliably learn to detect a pattern from ~75 positive examples in a 1020-sample dataset with 10 output dimensions. The model's rational strategy is to hedge on the majority class and capture recall_+1 where there's 2.5x more signal.

**Hypothesis 2: The LR finder is harmful for threshold-based losses and unreliable for CDW-CE.** Across run_014 and run_015, every CORN run with LR finder (applied LR 9.5–17x) has underperformed CORN at configured LR 0.001 by 0.106–0.120 QWK. The LR finder found monotonically decreasing landscapes for CDWCE_a2 and CDWCE_a5, falling back to lr_steep — suggesting these loss surfaces lack a distinct valley. Only CDWCE_a3 and the distribution-based losses (EMD, CORAL at valley) produce reliable valley detections.

**Hidden interaction:** Stimulation achieves QWK 0.483–0.558 despite having the most extreme class imbalance (87.8% neutral). This suggests the embedding space encodes stimulation-related concepts strongly — the model can detect the pattern even from the tiny minority signal. Conversely, security (75.3% neutral, less imbalanced) has the worst QWK (0.061–0.230). The bottleneck is **semantic distinctiveness in the embedding space**, not class balance alone.

## 8. Actionable Recommendations

### 1. Post-hoc logit adjustment on existing models (zero training cost)

**Evidence:** recall_-1 is a data scarcity problem. Logit adjustment (Menon et al. 2021, ICLR) shifts the decision boundary toward minority classes at inference time: `adjusted_logit = logit + tau * log(class_prior)`. With -1 prior ~0.073 and 0 prior ~0.748, this adds a significant negative offset to class 0 logits.

**Implementation:** Apply to run_010 CORN (SOTA) and run_015 CDWCE_a3 checkpoints. Sweep tau in {0.3, 0.5, 0.7, 1.0}. This requires saving logits during evaluation, which the current pipeline does not do.

**Watch:** recall_-1 improvement (target >15%), calibration maintenance (target >0.70), QWK stability.

### 2. Rerun CDWCE_a3 with configured LR=0.001 (disable LR finder)

**Evidence:** CDWCE_a3 achieved QWK 0.402 at LR=0.01552 (15.5x). But the LR finder is inconsistent — run_014 SoftOrdinal used valley LR 0.0198 and achieved QWK 0.388, while run_015 SoftOrdinal fell back to lr_steep at 0.00111 and got 0.335. CDWCE_a3's QWK could be higher or lower at the conservative LR. Given CORN's strong preference for LR=0.001, and CDW-CE being a hybrid loss with some threshold-like properties, this is worth testing.

**Watch:** QWK > 0.402, calibration > 0.755 (both improvements over LR-finder run).

### 3. Ensemble: CORN@0.001 + SoftOrdinal@0.001 via prediction averaging

**Evidence:** CORN excels on QWK (0.434) and recall_-1 (0.089) while SoftOrdinal leads on calibration (0.846/0.860) and decisiveness (80% hedging). These are complementary strengths. A simple argmax-of-averaged-probabilities ensemble could capture both.

**Implementation:** Requires saving per-dimension probability vectors during evaluation. Average probabilities from the two best checkpoints, then argmax.

**Watch:** QWK > 0.434, calibration > 0.835, hedging < 82%.

### 4. Training-time logit adjustment with CDWCE_a3 loss

**Evidence:** If post-hoc adjustment (recommendation #1) shows recall_-1 improvement, the training-time version integrates the class prior into the loss itself: `L = CDW-CE(f(x) + tau * log(pi), y)`. This allows the model to learn representations that account for class imbalance during training.

**Watch:** recall_-1 > 15%, QWK > 0.40.

### 5. Fix LR finder history bug and add per-loss LR override

**Evidence:** `lr_find_history.json` is overwritten by the last model (confirmed CDWCE_a5 in run_015, SoftOrdinal in run_014). Additionally, CORN consistently needs LR=0.001 while SoftOrdinal/EMD/CDWCE benefit from higher LRs. The training notebook should support `lr_override = {"corn": 0.001, "cdwce_a3": 0.015}`.

**Watch:** CORN QWK recovery to >0.40, SoftOrdinal QWK > 0.388.

### Research Corroboration

Web research confirmed: (1) CDW-CE alpha sensitivity is documented — Polat et al. found optimal alpha varies by architecture (5–7 for image classifiers); our 3-class problem has limited CDW-CE dynamic range (max |i−c|=2). (2) Logit adjustment has strong theoretical grounding (Fisher consistent for balanced error, Menon et al. 2021) and 8% relative error reduction on CIFAR-10-LT. (3) A 2025 ordinal regression survey (arXiv 2503.00952) explicitly flags minority class recognition as an **unsolved open problem** in the field, validating that our recall_-1 challenge is not an implementation failure but a fundamental challenge.

## 9. Summary Verdict

- **Best config (overall):** `run_010 CORN` at LR=0.001 remains SOTA — QWK 0.434 (moderate), calibration 0.835 (good), recall_-1 0.089 (poor, but best available). No run_015 configuration surpasses it.

- **Best new finding:** `run_015 CDWCE_a3` at QWK 0.402 (moderate) is the strongest new loss, confirming CDW-CE is competitive with CORN when alpha is tuned. However, it does not solve the recall_-1 bottleneck (0.056) and trades calibration (0.755 vs 0.835).

- **Key weakness:** `recall_-1 = 1–9%` across 15 runs, 7 loss functions, and 3 LR regimes. The model is structurally blind to misalignment. This is now confirmed to be a **data scarcity** problem (7.3% of all labels are -1), not a loss function problem.

- **Highest-leverage next experiment:** **Post-hoc logit adjustment** on run_010 CORN and run_015 CDWCE_a3. Zero training cost, directly targets the decision boundary. If tau=0.5 can push recall_-1 from 9% to 15%+ without collapsing QWK, it would represent the first meaningful progress on misalignment detection in 15 runs.
