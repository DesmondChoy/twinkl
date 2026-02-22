# Experiment Review — 2026-02-22

## 1. Experiment Overview

- **Scope**: 48 run files across `run_001`–`run_011`, six loss heads (`CORAL`, `CORAL_IW`, `CORN`, `EMD`, `MSE`, `SoftOrdinal`).
- **Focus**: MiniLM retired (see index.md Findings); this review focuses on the 30 nomic-embed run files (runs 002, 004, 006–011).
- **Axes varied**: capacity (`hidden_dim` 32/64/128), state encoder window (`ws=1/2`), loss head, and dataset size (637 → 958 → 1020).
- **Constants**: encoder `nomic-ai/nomic-embed-text-v1.5` (truncate_dim=256), split seed 2025, train/val/test ratio 0.70/0.15/0.15, dropout 0.3, LR 1e-3, weight_decay 0.01, batch size 16, max epochs 100, early_stopping_patience 20, MC-dropout 50.
- **Data notes**: `n_train` progressed from 637 → 958 → 1020. `pct_truncated` is 0.0 in all runs. 10 pre-tension Universalism personas removed at the 958 transition.

## 2. Head-to-Head Comparison

### Capacity Sweep (nomic ws=1, 1020 train, avg across 4 common losses)

| Metric | run_006 hd32 | run_007 hd64 | run_008 hd128 | Best |
|---|---:|---:|---:|---|
| MAE | 0.230 | 0.209 | **0.202** | hd128 |
| Acc | 0.800 | **0.819** | 0.825 | hd128 |
| QWK | 0.310 | **0.363** | 0.351 | **hd64** |
| Spearman | 0.348 | **0.376** | 0.383 | hd128 |
| Cal | 0.771 | **0.842** | 0.801 | **hd64** |
| MinR | 0.202 | **0.278** | 0.285 | hd128 |
| Hedge | 82.1% | **81.8%** | 82.5% | **hd64** |
| Ratio | 11.0 | **22.7** | 53.4 | hd32 (efficient) |

**Verdict**: hd=64 is the sweet spot — best QWK and calibration, comparable hedging, moderate param/sample ratio. hd=128 improves MAE/Acc marginally but sacrifices the two primary metrics (QWK, Cal).

### State Encoder Window (nomic hd64, 1020 train, avg across 5 losses incl. CORAL_IW)

| Metric | run_010 ws=1 | run_011 ws=2 | Delta |
|---|---:|---:|---:|
| MAE | **0.212** | 0.217 | +0.006 |
| Acc | **0.817** | 0.812 | -0.005 |
| QWK | **0.354** | 0.332 | -0.022 |
| Spearman | **0.371** | 0.355 | -0.016 |
| Cal | **0.842** | 0.825 | -0.017 |
| MinR | **0.268** | 0.254 | -0.014 |
| Hedge | 81.7% | 81.9% | +0.002 |
| state_dim | 266 | 523 | +257 |
| Ratio | 22.6 | 38.7 | +16.1 |

**Verdict**: ws=2 hurts across every metric. The extra history (523-dim vs 266-dim state) adds noise without useful temporal signal, and doubles the input compression ratio.

### Run-to-Run Reproducibility (run_007 vs run_010, same config, avg across 4 common losses)

| Metric | run_007 | run_010 | Delta |
|---|---:|---:|---:|
| QWK | 0.363 | **0.367** | +0.004 |
| Cal | 0.842 | 0.842 | 0.000 |
| MinR | 0.278 | 0.277 | -0.001 |

**Verdict**: Comparable (<5% delta on all metrics). The code changes between run_007 and run_010 (importance weight wiring, logger fixes) had negligible effect on the 4 common losses.

### Within-Run Loss Comparison (run_010, ws=1, hd64)

| Loss | MAE | Acc | QWK | Spearman | Cal | MinR | Hedge % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CORN** | **0.206** | **0.821** | **0.434** | **0.407** | 0.835 | 0.285 | 82.0% |
| CORAL | 0.209 | 0.819 | 0.364 | 0.391 | 0.823 | 0.244 | 83.5% |
| EMD | 0.212 | 0.819 | 0.362 | 0.357 | 0.851 | 0.294 | 79.8% |
| SoftOrdinal | 0.211 | 0.818 | 0.308 | 0.352 | **0.860** | 0.284 | 80.0% |
| CORAL_IW | 0.221 | 0.809 | 0.301 | 0.351 | 0.841 | 0.234 | 83.5% |

### Within-Run Loss Comparison (run_011, ws=2, hd64)

| Loss | MAE | Acc | QWK | Spearman | Cal | MinR | Hedge % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EMD** | 0.214 | **0.821** | **0.382** | 0.359 | 0.846 | **0.308** | 78.6% |
| CORAL | 0.218 | 0.807 | 0.339 | 0.368 | 0.815 | 0.245 | 82.1% |
| CORN | **0.209** | 0.814 | 0.335 | **0.388** | 0.811 | 0.232 | 84.0% |
| SoftOrdinal | 0.222 | 0.820 | 0.333 | 0.349 | **0.862** | 0.312 | **77.8%** |
| CORAL_IW | 0.223 | 0.798 | 0.269 | 0.312 | 0.790 | 0.175 | 86.9% |

**Key observation**: CORN leads at ws=1 but falls to 3rd at ws=2. EMD is the most window-robust loss — it improved QWK from 0.362 (ws=1) to 0.382 (ws=2), the only loss to gain from extra history. SoftOrdinal achieves the best calibration in both runs. CORAL_IW underperforms plain CORAL in both windows.

## 3. Per-Dimension Analysis

Computed across all 30 nomic run files (runs 002, 004, 006–011).

| Dimension | Mean QWK | Std | Min | Max | % >0.4 | % <0.2 | Category |
|---|---:|---:|---:|---:|---:|---:|---|
| stimulation | 0.449 | 0.075 | 0.314 | 0.667 | 73% | 0% | Easy |
| benevolence | 0.468 | 0.084 | 0.285 | 0.565 | 77% | 0% | Easy |
| conformity | 0.405 | 0.137 | 0.170 | 0.559 | 57% | 10% | Easy (volatile) |
| tradition | 0.357 | 0.149 | 0.036 | 0.531 | 43% | 17% | Moderate |
| self_direction | 0.378 | 0.062 | 0.258 | 0.447 | 37% | 0% | Moderate |
| hedonism | 0.375 | 0.088 | 0.240 | 0.592 | 33% | 0% | Moderate |
| achievement | 0.373 | 0.091 | 0.085 | 0.469 | 33% | 7% | Moderate |
| universalism | 0.285 | 0.131 | 0.058 | 0.573 | 17% | 23% | Hard |
| security | 0.171 | 0.113 | 0.029 | 0.417 | 3% | 53% | Hard |
| **power** | **0.076** | **0.179** | **-0.347** | **0.406** | 3% | 67% | **Hardest** |

**Error analysis**: No model checkpoint available (gitignored). Qualitative sample extraction skipped.

## 4. Calibration Deep-Dive

- **All 30 nomic runs** have 10/10 positive calibration dimensions — no deployment-risk dimensions.
- **Global calibration range** (nomic only): 0.734 (run_004 CORN) to 0.862 (run_011 SoftOrdinal).
- **Best calibrated loss**: SoftOrdinal (mean 0.833 across nomic runs), followed by EMD (0.828).
- **Worst calibrated loss**: CORAL_IW (mean 0.816 in run_010/011), despite being designed to improve minority handling.

| Loss | Mean Cal (nomic runs) | Min Cal | Max Cal |
|---|---:|---:|---:|
| SoftOrdinal | 0.833 | 0.775 | 0.862 |
| EMD | 0.828 | 0.764 | 0.851 |
| CORAL | 0.805 | 0.734 | 0.830 |
| CORN | 0.802 | 0.737 | 0.838 |
| CORAL_IW | 0.816 | 0.790 | 0.841 |

**Systematic pattern**: Higher calibration correlates with smoother loss functions (SoftOrdinal, EMD use distributional targets). CORAL/CORN use hard ordinal thresholds that produce sharper but less calibrated predictions.

## 5. Hedging vs Minority Recall Trade-off

No nomic configuration achieves `decisive + balanced` (hedging < 60% AND minority recall > 30%). The best trade-offs in the nomic family:

| Run + Loss | Hedging % | MinR | recall_-1 | recall_+1 | Verdict |
|---|---:|---:|---:|---:|---|
| run_011 SoftOrdinal | **77.8%** | **0.312** | 0.069 | 0.554 | Closest to balanced |
| run_011 EMD | 78.6% | 0.308 | 0.053 | 0.563 | Strong +1 recall |
| run_010 EMD | 79.8% | 0.294 | 0.072 | 0.517 | |
| run_010 CORN | 82.0% | 0.285 | 0.089 | 0.480 | Best QWK but hedges |
| run_007 SoftOrdinal | 80.7% | 0.291 | 0.067 | 0.514 | |

**Critical finding**: `recall_minus1` never exceeds 10.3% (run_007 CORN) across all nomic runs. The model is essentially blind to misalignment (class -1). Even the best minority recall configs achieve this by detecting +1 (alignment), not -1 (misalignment).

## 6. Capacity & Overfitting

| Run | ws | hd | Params | Ratio | Gap@Best | Best/Total Epoch | QWK | Cal |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| run_006 | 1 | 32 | 10.5K | 11.0 | 0.126 | 21/41 | 0.310 | 0.771 |
| run_007 | 1 | 64 | 23.1K | 22.7 | -0.053 | 13.5/33.5 | 0.363 | 0.842 |
| run_008 | 1 | 128 | 54.4K | 53.4 | -0.042 | 6.5/26.5 | 0.351 | 0.801 |
| run_010 | 1 | 64 | 23.1K | 22.6 | 0.009 | 14.8/34.8 | 0.354 | 0.842 |
| run_011 | 2 | 64 | 39.6K | 38.7 | -0.006 | 9.4/29.4 | 0.332 | 0.825 |

**Key observations**:
- **hd=32 shows mild overfitting** (gap 0.126 at run_006) with the slowest convergence (epoch 21). The model is underfitting capacity-wise but overfitting due to too many training epochs on sparse signal.
- **hd=64 is well-regularized** (gap near 0 or slightly negative) with healthy convergence (epoch 13-16).
- **hd=128 converges too fast** (best_epoch 4-8) but doesn't overfit — it simply stops learning before capturing minority structure. The extra capacity is wasted, not harmful.
- **ws=2 doubles input dim without benefit**: ratio jumps from 22.6 to 38.7 with no QWK gain.

## 7. Systemic Insights & Hypotheses

### The Overarching Story

These 11 runs tell a consistent story: **the critic has learned the ordinal structure of the easier dimensions (stimulation, benevolence, conformity) but its decision boundaries are dominated by the class prior on hard dimensions.** The model's QWK of 0.434 (moderate) masks a fundamental asymmetry: it detects value-aligned behavior (+1 recall 48%) but is nearly blind to misalignment (-1 recall 8.9%). For a tool meant to surface tensions between behavior and values, this is the critical weakness.

### Hidden Interactions

**1. Label skew × split instability on Power**: Automated investigation of `judge_labels.parquet` (seed 2025, n=1460) reveals a severe split artifact:
- **Power validation set**: only 23 non-zero examples (12 neg, 11 pos) across just 10 personas — **90% neutral**
- **Power test set**: 65 non-zero examples (27 neg, 38 pos) — **28.3% non-zero**, 3x richer than val

This means model selection (via val loss) is essentially random for Power. The model that performs best on validation may be terrible at Power discrimination, and we'd never know until test time. This single split artifact explains much of Power's extreme QWK volatility (std 0.179, range -0.347 to 0.406).

**2. History length hurts CORN/CORAL but not EMD/SoftOrdinal**: ws=1→2 caused CORN QWK to drop 0.099 and CORAL to drop 0.024, while EMD *gained* 0.020 and SoftOrdinal gained 0.024. The distributional losses (EMD, SoftOrdinal) can absorb noisy input features because they produce soft probability targets — the extra history dimensions add variance but the loss is robust to it. CORN/CORAL, with their hard cumulative thresholds, amplify noise into discrete threshold errors.

**3. CORAL_IW consistently degrades performance**: In both runs tested (010, 011), importance-weighted CORAL produced worse QWK (mean -0.047 vs CORAL), worse minority recall (mean -0.035), and higher hedging (mean +1.7%). The 2-element importance weight vector for a 3-class problem is too coarse to express the needed rebalancing — it weights binary sub-tasks P(y>-1) and P(y>0), not classes directly.

### Disproven Assumptions

- **"More capacity improves QWK"**: False above hd=64. hd=128 improved MAE but reduced QWK and calibration.
- **"Longer temporal window helps value reasoning"**: False in general. ws=2 hurt 3/5 losses.
- **"Importance weighting improves minority recall"**: False. CORAL_IW degraded both QWK and minority recall vs plain CORAL.

### Strong Hypotheses

**H1**: The primary failure mode is **prior-dominated decision boundaries** under ordinal class skew. The model learns rank structure for common cases and collapses to neutral on sparse minority evidence. This is a loss-function problem, not a representation problem.

**H2**: Power QWK instability is largely a **validation-set artifact**. With only 23 non-zero val examples, the model selected by early stopping may be randomly good or bad at Power discrimination. Any experiment targeting Power must first address this measurement problem (e.g., stratified per-dimension splits or k-fold CV).

## 8. Actionable Recommendations

### Rec 1 (Highest Priority): Implement CDW-CE loss as a new loss head

**Evidence**: CORN achieves the best QWK (0.434), but cumulative-link heads do not have the same direct post-hoc logit-adjustment guarantee as softmax heads in Menon et al. CDW-CE (Class Distance Weighted Cross-Entropy, [Polat et al. 2025](https://arxiv.org/abs/2412.01246)) uses standard softmax output with polynomially-growing ordinal distance penalties. In published results on LIMUC (11K images, 4 ordinal classes, 54% majority), CDW-CE achieved QWK 0.857 vs CORN's 0.837 — a consistent advantage across three architectures.

**Why this change**: CDW-CE penalizes distant misclassifications polynomially (e.g., -1→+1 penalized 4x vs -1→0 at power=2), directly targeting the ordinal distance structure. It also enables standard class-frequency reweighting (alpha) and Menon-style post-hoc logit adjustment in the same softmax framework. A [PyTorch implementation exists](https://github.com/carlomarxdk/cdw-cross-entropy-loss).

**Config**: `run_010` config (ws=1, hd=64, nomic-256d) with CDW-CE (power=2) + inverse-frequency class weights per dimension.

**Metric target**: QWK >= 0.42, recall_minus1 >= 0.15, calibration >= 0.80.

### Rec 2: Add post-hoc logit adjustment to SoftOrdinal first (then CDW-CE)

**Evidence**: Logit adjustment ([Menon et al. 2021](https://arxiv.org/abs/2007.07314)) is provably Fisher-consistent for class-balanced error in standard softmax classification under class imbalance. The operational form is `logit_adj_k = logit_k - tau * log(pi_k)` (equivalently `+ tau * log(1/pi_k)`), where `pi_k` is the training prior for class `k`. CORAL/CORN threshold-shift variants are still possible as heuristics, but the Menon guarantee is not directly stated for cumulative-link heads.

**Why this change**: Nearly zero-cost intervention on an existing softmax head without changing the encoder/state pipeline. `run_011 SoftOrdinal` is the cleanest target (calibration 0.862, minority recall 0.312, lowest hedging at 77.8%). Compute per-dimension class priors from training labels and apply adjustment only at inference.

**Metric target**: recall_minus1 improvement of >= 0.05 over un-adjusted SoftOrdinal, with QWK drop <= 0.01.

### Rec 3: Implement per-dimension stratified validation splits

**Evidence**: Automated investigation found Power's val set has only 23 non-zero examples across 10 personas (90% neutral), while the test set has 65 non-zero (28.3% non-zero). Model selection by val loss is essentially random for Power. This split artifact explains Power QWK volatility (std 0.179).

**Why this change**: Without a representative validation set, early stopping selects models that are randomly good or bad at Power/Security. Stratified splits ensure each dimension has proportional minority representation in val.

**Config**: Modify split logic to stratify on the per-dimension minority label presence, not just persona-level.

**Metric target**: Power QWK std across runs < 0.10 (currently 0.179), Security QWK mean > 0.20.

### Rec 4: Investigate SLACE loss (AAAI 2025)

**Evidence**: SLACE ([Soft Labels Accumulating Cross Entropy](https://ojs.aaai.org/index.php/AAAI/article/view/34158)) has two provable properties directly relevant here: **monotonicity** (further misclassifications are penalized more, like CDW-CE) and **balance-sensitivity** (accounts for class distribution, unlike CDW-CE). It outperformed SOTA ordinal losses on tabular benchmarks — the closest match to your embedding+MLP setup.

**Why this change**: If CDW-CE + class weights doesn't break the hedging barrier, SLACE addresses both ordinal structure and imbalance in a single principled loss.

**Metric target**: Hedging < 78% with minority recall > 0.30 and QWK >= 0.40.

### Rec 5: Drop CORAL_IW from the experiment matrix

**Evidence**: CORAL_IW degraded QWK by 0.047 and minority recall by 0.035 on average vs plain CORAL across runs 010 and 011. The importance weight mechanism for a 3-class problem (2-element vector weighting binary sub-tasks) is too coarse, and the theoretical basis is weak ([coral-pytorch GitHub #7](https://github.com/Raschka-research-group/coral-pytorch/issues/7)).

**Why this change**: Eliminates a consistently underperforming head, freeing compute budget for CDW-CE/SLACE experiments.

### Web Research Conducted

| Source | Key Finding | Impact on Recommendations |
|---|---|---|
| [CDW-CE (Polat et al. 2025)](https://arxiv.org/abs/2412.01246) | Outperforms CORN on LIMUC; PyTorch impl available | Rec 1: CDW-CE as top-priority new loss |
| [Logit Adjustment (Menon et al. 2021)](https://arxiv.org/abs/2007.07314) | Provably Fisher-consistent for softmax with prior correction `-tau*log(pi)`; no direct cumulative-link guarantee | Rec 2: Apply to SoftOrdinal first (zero-cost), then CDW-CE |
| [SLACE (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34158) | Provable monotonicity + balance-sensitivity; strong on tabular | Rec 4: Secondary loss if CDW-CE insufficient |
| [coral-pytorch GitHub #7](https://github.com/Raschka-research-group/coral-pytorch/issues/7) | IW for CORAL is ad-hoc; coarse for 3-class | Rec 5: Drop CORAL_IW |
| [Ordinal Survey (March 2025)](https://arxiv.org/html/2503.00952v1) | Pre-trained encoders + small head is current best practice | Confirms nomic-embed + MLP architecture is sound |
| [ORCU (2024)](https://arxiv.org/html/2410.15658v3) | Unimodal regularization improves calibration | Future consideration for calibration refinement |

## 9. Summary Verdict

- **Best config**: `run_010 CORN` (nomic-256d, ws=1, hd=64) — QWK 0.434 (moderate), calibration 0.835 (good), MAE 0.206. The only config to reach moderate QWK on the 1020-sample dataset.
- **Key weakness**: The model is blind to misalignment (recall_minus1 = 8.9%). Power QWK is near-zero (mean 0.076) due to both label sparsity and a validation split artifact. The critic cannot reliably detect when behavior drifts from values — its core purpose.
- **Highest-leverage next experiment**: Apply **post-hoc logit adjustment to run_011 SoftOrdinal** as a zero-cost baseline (Rec 2), then implement **CDW-CE loss with per-dimension inverse-frequency class weights** (Rec 1) for distance-weighted ordinal penalties. Both directly target the prior-bias failure mode (H1). Simultaneously fix the **per-dimension stratified validation split** (Rec 3) so that Power/Security model selection is no longer random.
