# VIF Experiment Cross-Run Review — 2026-03-04 (v2)

**Scope**: 57 YAML configs across 13 runs (run_001–run_013). This review focuses on the frontier nomic-embed runs (run_007–run_013), with emphasis on the new run_013 which introduced LR-finder-selected learning rates.

## 1. Experiment Overview

**What varied across runs:**
- **Encoder**: MiniLM-384d (runs 001, 003, 005, 009 — retired) vs nomic-embed-256d (all others)
- **Window size**: ws=1 (runs 002, 004, 006, 007, 008, 010) vs ws=2 (runs 011, 012, 013)
- **Capacity**: hd=32 (runs 002, 004, 006), hd=64 (runs 007, 010–013), hd=128 (run 008), hd=256 (runs 001, 003, 005)
- **Loss function**: CORAL, CORN, EMD, SoftOrdinal (all runs); CORAL_IW (010, 011, 012 — dropped); MSE (001, 002 — dropped)
- **Learning rate**: 0.001 fixed (all prior runs) vs **LR-finder-selected** (run_013 only)

**What stayed constant:** nomic-embed-text-v1.5 @ 256d (for frontier), split_seed=2025, dropout=0.3, batch_size=16, AdamW w/ ReduceLROnPlateau, train/val/test = 70/15/15 by persona, n_train=1020.

**Key methodological note:** run_013 introduced a hidden confound. The `config_delta` shows no changes vs run_012, but `critic_training_v4` now runs an LR finder before training and uses its suggestion as the starting LR. Per-loss LRs: CORAL/CORN ~0.00477 (4.8× configured), EMD ~0.00166 (1.7×), SoftOrdinal ~0.000944 (~1×). The experiment logger records configured LR, not applied LR — a provenance gap.

## 2. Head-to-Head Comparison

### A. ws=2 Stochastic Variance (run_011 vs 012 vs 013, hd=64, nomic)

| Metric | run_011 CORN | run_012 CORN | run_013 CORN | Range |
|--------|:---:|:---:|:---:|:---:|
| QWK | 0.335 | 0.346 | **0.382** | 0.047 |
| Calibration | **0.811** | 0.804 | 0.828 | 0.024 |
| MAE | 0.209 | **0.193** | 0.208 | 0.016 |
| MinR | 0.232 | **0.296** | 0.269 | 0.064 |
| recall_minus1 | 0.056 | **0.096** | 0.075 | 0.040 |
| Hedging | 84.0% | 84.1% | **82.3%** | 1.8pp |

Stochastic variance across three identical-config reruns is **substantial**: QWK ranges 0.335–0.382 (0.047 spread), MinR ranges 0.232–0.296. run_013's QWK uplift may partly reflect LR-finder benefit and partly stochastic luck.

### B. ws=1 vs ws=2 (hd=64, nomic, averaged across reruns)

| Loss | ws=1 avg QWK | ws=2 avg QWK | Delta |
|------|:---:|:---:|:---:|
| CORAL | 0.365 | 0.361 | -0.005 (comparable) |
| **CORN** | **0.424** | 0.354 | **-0.070** |
| EMD | 0.360 | 0.381 | +0.021 (comparable) |
| SoftOrdinal | 0.311 | 0.334 | +0.022 (comparable) |

CORN is the only loss with a meaningful ws=1 advantage (-0.070). For EMD and SoftOrdinal, ws=2 is comparable or marginally better, but within stochastic variance.

### C. Overall Frontier (best per configuration)

| Metric | run_010 CORN (ws=1) | run_013 EMD (ws=2) | run_013 CORAL (ws=2) | run_012 SoftOrdinal (ws=2) |
|--------|:---:|:---:|:---:|:---:|
| QWK | **0.434** | 0.391 | 0.384 | 0.334 |
| Calibration | 0.835 | 0.840 | 0.820 | **0.850** |
| MinR | 0.285 | 0.293 | 0.259 | **0.391** |
| recall_minus1 | **0.089** | 0.056 | 0.061 | 0.083 |
| Hedging | 82.0% | 80.4% | 81.2% | **77.9%** |

run_010 CORN remains the QWK leader. run_012 SoftOrdinal is the minority-sensitivity leader.

## 3. Per-Dimension Analysis

Mean QWK across all nomic runs (excl. MSE/CORAL_IW), sorted by difficulty:

| Dimension | Mean QWK | Std | Min | Max | Category |
|-----------|:---:|:---:|:---:|:---:|----------|
| Power | 0.084 | 0.201 | -0.347 | 0.406 | **Hard** (poor) |
| Security | 0.180 | 0.089 | 0.029 | 0.419 | **Hard** (poor) |
| Universalism | 0.317 | 0.151 | 0.056 | 0.610 | Volatile |
| Self-direction | 0.380 | 0.050 | 0.258 | 0.447 | Moderate (stable) |
| Achievement | 0.379 | 0.070 | 0.228 | 0.469 | Moderate |
| Tradition | 0.422 | 0.118 | 0.036 | 0.560 | Moderate-Good |
| Conformity | 0.426 | 0.134 | 0.170 | 0.586 | Moderate-Good |
| Benevolence | 0.470 | 0.079 | 0.285 | 0.569 | **Easy** (moderate+) |
| Stimulation | 0.487 | 0.085 | 0.314 | 0.667 | **Easy** (moderate+) |

**Error Analysis**: No checkpoint available (cleaned up). Skipped.

**Hardest dimensions deep-dive (Power & Security):**
- **Power**: Only 78/180 personas (43%) ever produce non-zero labels. Val split has just 12 misalignment (-1) labels across 9 personas — model selection for Power is effectively random. The extreme QWK variance (-0.347 to +0.406) confirms this.
- **Security**: Better coverage (119/180 personas, 66%), but val split has an inverted -1/+1 ratio (14.2% vs 10.6%) compared to the full dataset (9.3% vs 15.3%). This distribution artifact may confound checkpoint selection.

## 4. Calibration Deep-Dive

All nomic frontier runs achieve **10/10 positive-calibration dimensions** — no dangerous negative calibration. Global calibration ranges 0.785–0.862 (good). Lowest per-dimension calibration is consistently self_direction (~0.53–0.59, weak), reflecting the dimension with the most balanced label distribution (41.2% non-zero).

No dimension has calibration < -0.4 in any nomic run. MSE was the only loss to produce negative global calibration (-0.218 in run_001, -0.073 in run_002) — correctly retired.

## 5. Hedging vs Minority Recall Trade-off

| Run + Loss | Hedging % | MinR | recall_minus1 | Verdict |
|------------|:---------:|:----:|:-------------:|---------|
| run_010 CORN | 82.0% | 0.285 | 0.089 | EXCESSIVE hedging |
| run_012 SoftOrdinal | **77.9%** | **0.391** | 0.083 | **moderate + balanced** |
| run_012 EMD | 79.2% | 0.364 | 0.065 | moderate + balanced |
| run_011 SoftOrdinal | 77.8% | 0.312 | 0.069 | moderate + balanced |
| run_011 EMD | 78.6% | 0.308 | 0.053 | moderate + balanced |
| run_013 EMD | 80.4% | 0.293 | 0.056 | EXCESSIVE hedging |
| run_013 CORN | 82.3% | 0.269 | 0.075 | EXCESSIVE hedging |

**No configuration achieves hedging < 60% and MinR > 30%.** The best trade-off is run_012 SoftOrdinal (77.9% hedging, MinR 0.391), but this is a stochastic peak — run_013 SoftOrdinal with identical config dropped to MinR 0.269. The LR finder's higher starting LR appears to have shifted the QWK/MinR trade-off toward agreement at the expense of minority sensitivity.

## 6. Capacity & Overfitting

| Run | Params | Ratio | Gap | Best Epoch | Status |
|-----|:------:|:-----:|:---:|:----------:|--------|
| run_010 (ws=1, hd=64) | 22,804 | 22.4 (high) | 0.007 | 16 | Good fit |
| run_013 (ws=2, hd=64) | 39,252 | 38.5 (high) | 0.006 | 9–11 | Good fit |
| run_008 (ws=1, hd=128) | 53,780 | 52.7 (high) | 0.005 | 4–8 | Early stops too fast |

All frontier runs show **minimal overfitting** (gap < 0.01). The ws=2 models are ~1.7× larger than ws=1 but train comparably. run_008's hd=128 early-stopped at epoch 4–8, never reaching the QWK levels of hd=64 — confirming hd=64 is the capacity sweet spot for this dataset.

run_013's LR-finder runs reached best_epoch 9–11 (vs 7–12 for run_011/012), suggesting the higher LR neither over-nor under-trained.

## 7. Systemic Insights & Hypotheses

**The overarching story:** The VIF critic can learn meaningful ordinal structure for ~7 of 10 dimensions (QWK 0.3–0.6), but **systematically fails on Power and Security** — dimensions where the label signal is structurally thin (Power: 43% persona coverage, 5.3% val -1 labels) or distribution-inverted in the val split (Security). This is not a model architecture or loss function problem — it's a **data sufficiency** problem masked by high-variance metrics.

**Hypothesis 1: The LR finder helps QWK but hurts minority recall.** run_013's higher starting LRs (especially 4.8× for CORAL/CORN) produced the best ws=2 QWK results but consistently lower MinR than run_012. The likely mechanism: a higher LR finds wider optima that generalize better on the majority class but are less sensitive to rare-label gradients. This suggests post-hoc calibration adjustments (logit adjustment at inference time) rather than training-time LR changes are the right lever for minority recall.

**Hypothesis 2: CORN's ws=1 advantage is real, not stochastic.** CORN is the only loss where ws=1 consistently outperforms ws=2 (0.424 vs 0.354 mean, 0.070 gap — larger than the stochastic variance of 0.047 observed across ws=2 reruns). The conditional rank structure of CORN may be particularly sensitive to the added noise of historical state features in the 523-dim ws=2 input.

## 8. Actionable Recommendations

### Rec 1: CDW-CE Loss (Highest Priority)
**Evidence**: All current losses produce hedging > 77% and treat ordinal distance equally. CDW-CE (Polat et al. 2025, [arXiv:2412.01246](https://arxiv.org/abs/2412.01246)) penalizes predictions proportionally to |predicted - true|^α, which directly addresses the hedging problem by making neutral predictions for non-neutral labels increasingly costly. CDW-CE outperformed CORN on QWK across all architectures tested in the paper.

**Implementation**: `CDW-CE = -Σ log(1-ŷᵢ) × |i-c|^α` with α ∈ {2, 3, 5}, starting with α=2 for this small-data regime. For 3 classes, a 2-away error gets `2^α` penalty vs 1 for adjacent — at α=5 that's 32× (may be too aggressive with ~1000 samples). **Important**: CDW-CE operates on softmax outputs (K=3 logits per dimension = 30 total), not CORAL/CORN binary thresholds. Implement alongside existing EMD/SoftOrdinal architecture in `critic_ordinal.py`. Public PyTorch code: [carlomarxdk/cdw-cross-entropy-loss](https://github.com/carlomarxdk/cdw-cross-entropy-loss).

**What to watch**: QWK improvement over 0.434, hedging reduction toward <75%, per-dimension Power/Security QWK stability.

**Web research corroboration**: CDW-CE outperformed CORN (QWK 0.857 vs 0.841) on a 4-class imbalanced ordinal task (LIMUC). Additive margin variant available for further tuning. SLACE (Nachmani & Genossar, [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34158)) offers an alternative with provable monotonicity and balance-sensitivity but no public implementation. OLL (Ordinal Log Loss, [COLING 2022](https://aclanthology.org/2022.coling-1.407/)) is another option, but SLACE outperforms OLL by 4.7% on AMAE.

**Note**: CDW-CE alone does NOT address class imbalance — it only addresses ordinal distance. For full effect, combine with logit adjustment (Rec 2).

### Rec 2: Post-hoc Logit Adjustment at Inference
**Evidence**: run_012 SoftOrdinal achieved MinR 0.391 but this was stochastic (run_013: 0.269). Training-time class weighting (CORAL_IW) was already tried and failed. Post-hoc logit adjustment (Menon et al. 2020, [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)) applies `logit_adj_k = logit_k - τ·log(π_k)` at inference using known class priors — no retraining needed.

**Implementation**: For softmax heads (EMD/SoftOrdinal/CDW-CE), add `τ·log(π_y)` to the 3 logits before computing loss. Use **per-dimension** class priors (distributions vary from 58.8% neutral for Self-direction to 87.8% for Stimulation). Sweep τ ∈ {0.3, 0.5, 0.7, 1.0} — start low since τ=1 may be too aggressive with ~1000 samples. Post-hoc variant is a free lunch (no retraining): `ŷ = argmax_y [f_y(x) - τ·log(π_y)]`. For CORAL/CORN binary thresholds, logit adjustment does not directly apply — use the softmax architecture instead.

**What to watch**: recall_minus1 improvement (currently 0.089), hedging reduction, with acceptable QWK drop < 0.03.

### Rec 3: Per-Dimension Stratified Validation Splits
**Evidence**: Power val has only 9 personas with non-zero labels (12 misalignment labels total). Security val has an inverted -1/+1 ratio. Model selection on these dimensions is effectively unreliable.

**Implementation**: Use per-dimension stratified sampling ensuring minimum 5% representation of each class in val. If insufficient labels exist (Power), consider leaving Power out of early-stopping criteria.

**What to watch**: Cross-rerun QWK variance for Power and Security. Currently Power ranges -0.347 to 0.406 — a well-stratified split should narrow this.

### Rec 4: Fix Experiment Logger to Record Applied LR
**Evidence**: run_013's YAML files show `learning_rate: 0.001` despite actual training LRs of 0.00477 (CORAL/CORN). This breaks the provenance chain.

**Implementation**: Log `learning_rate_applied` alongside `learning_rate_configured` in the experiment YAML.

### Rec 5: Run LR-Finder on ws=1 Baseline
**Evidence**: The LR finder improved ws=2 CORN QWK from 0.335–0.346 to 0.382. The ws=1 CORN leader (run_010, QWK 0.434) used the hardcoded 0.001 LR. A higher LR may push ws=1 CORN even higher.

**What to watch**: QWK above 0.434, gap_at_best (should stay < 0.01).

## 9. Summary Verdict

- **Best config**: `run_010 CORN` (ws=1, nomic-256d, hd=64) — QWK 0.434 (moderate), calibration 0.835 (good). Still the state of the art after 3 additional runs.

- **Key weakness**: Misalignment detection remains broken — `recall_minus1` = 8.9% and hedging 82% mean the critic effectively ignores the -1 class. This is the single biggest deployment risk.

- **Highest-leverage next experiment**: Implement CDW-CE loss on the ws=1/hd=64 config with LR-finder-selected starting LR. The distance-weighted penalty directly targets hedging (the primary bottleneck) while preserving ordinal structure. If CDW-CE lifts QWK above 0.45 with hedging < 75%, it would be the first configuration to achieve "moderate agreement with actionable minority detection."
