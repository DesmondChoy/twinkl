# VIF Experiment Review — 2026-03-04 v3

**Scope:** 14 runs (61 YAML configs), incorporating new **run_014** (ws=1 + LR finder valley).
**Analyst mindset:** Senior AI/Data Scientist, reading between metrics for mechanistic understanding.

---

## 1. Experiment Overview

**What varied across runs:**
- **Encoder:** MiniLM-384d (runs 001, 003, 005, 009 — retired) vs nomic-256d (all others)
- **Capacity:** hd=32/64/128/256 sweep; frontier settled at hd=64
- **Window size:** ws=1 (runs 002, 004, 006–008, 010, 014), ws=2 (011–013), ws=3 (MiniLM only)
- **Loss function:** CORAL, CORN, EMD, SoftOrdinal (all runs); MSE (001–002, retired); CORAL_IW (010–012, dropped)
- **Learning rate:** 0.001 fixed (001–012), LR finder lr_steep (013, ws=2), LR finder lr_valley (014, ws=1)
- **Dataset:** 637 → 958 → 1020 train samples across generations

**Constants:** nomic-256d frozen encoder, dropout=0.3, wd=0.01, batch=16, patience=20, seed=2025, 70/15/15 split.

**Dataset:** 1020 train / 230 val / 210 test entries from 180 personas. Zero truncation (nomic). Class distribution heavily neutral (75–88% per dimension).

---

## 2. Head-to-Head Comparison

### 2a. run_014 vs run_010: LR Finder Valley Impact on ws=1

Both use identical architecture (nomic-256d, ws=1, hd=64). Only LR differs: run_010 uses 0.001; run_014 uses LR finder valley (~17–20x higher).

| Metric | run_010 CORN | run_014 CORN | run_010 SoftOrd | run_014 SoftOrd |
|--------|-------------|-------------|-----------------|-----------------|
| **QWK** | **0.434** | 0.314 (-0.120) | 0.308 | **0.388 (+0.080)** |
| Spearman | **0.407** | 0.368 | 0.352 | **0.394** |
| Calibration | **0.835** | 0.753 (-0.082) | **0.860** | 0.801 (-0.059) |
| MinR | **0.285** | 0.246 | 0.284 | **0.288** |
| recall_-1 | **0.089** | 0.060 | 0.062 | **0.075** |
| Hedging | **0.820** | 0.848 | **0.800** | 0.820 |
| MAE | **0.206** | 0.207 | 0.211 | **0.204** |
| best_epoch | 16 | 7 | 8 | 16 |

**Verdict:** The LR finder valley recommendation is **loss-function-dependent**. It catastrophically hurts CORN (-0.120 QWK) and CORAL (-0.035) but uniquely benefits SoftOrdinal (+0.080, best-ever). EMD is roughly flat (+0.011 QWK but -0.066 calibration). Threshold-based losses (CORAL/CORN) are more LR-sensitive than distribution-based losses (EMD/SoftOrdinal).

### 2b. Cross-Loss Comparison on ws=1 Frontier (runs 010 + 014 best per loss)

| Metric | CORN (r010) | SoftOrd (r014) | EMD (r014) | CORAL (r010) |
|--------|------------|----------------|------------|-------------|
| **QWK** | **0.434** | 0.388 | 0.373 | 0.364 |
| Calibration | 0.835 | 0.801 | 0.785 | 0.823 |
| MinR | 0.285 | **0.288** | 0.274 | 0.244 |
| recall_-1 | **0.089** | 0.075 | 0.058 | 0.063 |
| Hedging | **0.820** | **0.820** | 0.834 | 0.836 |

### 2c. ws=1 vs ws=2 (LR-finder-tuned)

| Metric | ws=1 best (r010 CORN) | ws=2 best (r013 EMD) | Delta |
|--------|----------------------|---------------------|-------|
| **QWK** | **0.434** | 0.391 | +0.043 |
| Calibration | 0.835 | **0.840** | comparable |
| MinR | 0.285 | **0.294** | comparable |
| Hedging | **0.820** | 0.804 | comparable |

ws=1 maintains a clear QWK advantage over ws=2 even after LR-finder tuning on both.

---

## 3. Per-Dimension Analysis

Mean QWK across nomic frontier runs (007, 010, 011–014, CORN+SoftOrdinal+EMD pooled):

| Dimension | Mean QWK | Std | Category |
|-----------|---------|-----|----------|
| Conformity | 0.510 | 0.049 | **Easy** (> 0.4) |
| Stimulation | 0.507 | 0.052 | **Easy** |
| Benevolence | 0.497 | 0.049 | **Easy** |
| Tradition | 0.461 | 0.065 | **Easy** |
| Hedonism | 0.393 | 0.060 | Moderate |
| Achievement | 0.393 | 0.057 | Moderate |
| Self_direction | 0.405 | 0.036 | Moderate |
| Universalism | 0.243 | 0.130 | **Volatile** |
| Security | 0.140 | 0.054 | **Hard** (< 0.3) |
| Power | 0.031 | 0.196 | **Hard / Broken** |

**Easy dimensions** (QWK > 0.4): Conformity, Stimulation, Benevolence, Tradition — consistently moderate agreement. These leverage clear behavioral signals in journal text.

**Hard dimensions** (QWK < 0.3): Security (0.140, fair-to-poor) and Power (0.031, essentially random). Power has negative QWK in 11 of 61 configs. Data analysis confirms: Power has only ~10 misalignment (-1) labels in validation; model selection is random noise.

**Volatile:** Universalism QWK ranges 0.031–0.466 across runs (std=0.130). The 85.3% neutral rate and only ~10.5 val -1 labels make this highly split-dependent.

**Error analysis:** No model checkpoint available (best_model.pt is gitignored and not present on disk). Skipped.

---

## 4. Calibration Deep-Dive

| Run + Loss | Global Cal | Positive Dims | Worst Per-Dim |
|-----------|-----------|---------------|---------------|
| run_010 SoftOrd | **0.860** | 10/10 | Self_dir 0.568 |
| run_010 EMD | 0.851 | 10/10 | Self_dir 0.447 |
| run_010 CORN | 0.835 | 10/10 | Security 0.567 |
| run_013 EMD | 0.840 | 10/10 | Security 0.527 |
| run_014 SoftOrd | 0.801 | 10/10 | Self_dir 0.549 |
| run_014 EMD | 0.785 | 10/10 | Self_dir 0.478 |
| **run_014 CORN** | **0.753** | 10/10 | **Security 0.335** |

**Key finding:** Calibration degraded across ALL losses in run_014 vs run_010. The aggressive valley LR widens confidence intervals. **run_014 CORN's Security calibration at 0.335 is a deployment risk** (weak by threshold table, approaching dangerous). No dimension has negative calibration in any nomic run — a clear improvement over MiniLM/MSE days.

**Systematic pattern:** Self_direction and Security consistently have the lowest per-dim calibration (0.33–0.67), likely because these dimensions have the most complex behavioral signals that the model is least confident about.

---

## 5. Hedging vs Minority Recall Trade-off

| Run + Loss | Hedging % | MinR | recall_-1 | recall_+1 | Verdict |
|-----------|----------|------|-----------|-----------|---------|
| run_012 SoftOrd | 77.9% | **0.391** | 0.083 | **0.699** | **Best MinR, moderate hedging** |
| run_012 EMD | 79.2% | **0.364** | 0.065 | **0.663** | **Decisive + balanced** |
| run_011 SoftOrd | 77.8% | 0.312 | 0.069 | 0.554 | Moderate hedging |
| run_011 EMD | 78.6% | 0.308 | 0.053 | 0.563 | Moderate hedging |
| run_014 SoftOrd | 82.0% | 0.288 | 0.075 | 0.502 | Excessive hedging |
| run_010 CORN | 82.0% | 0.285 | 0.089 | 0.480 | Excessive hedging |
| run_014 CORN | 84.8% | 0.246 | 0.060 | 0.431 | Excessive hedging |

**No configuration achieves both hedging < 60% and MinR > 30%.** The closest is run_012 SoftOrd (77.9% hedging, 0.391 MinR) — still excessive hedging by threshold definition.

**Critical asymmetry:** recall_plus1 (0.43–0.70) vastly exceeds recall_minus1 (0.05–0.09) across ALL configs. The model detects alignment (+1) reasonably but is nearly blind to misalignment (-1). This is the core deployment risk — Twinkl's value proposition depends on surfacing tensions (misalignment), which the model systematically misses.

**LR finder effect on minority recall:** run_014's aggressive LR did NOT improve recall_minus1 (0.060–0.075 vs run_010's 0.063–0.089). The hedging problem is structural, not LR-dependent.

---

## 6. Capacity & Overfitting

| Run | hd | Params | Ratio | Gap | best_epoch | Characterization |
|-----|---:|-------:|------:|----:|----------:|------------------|
| 006 (nomic, ws=1, hd=32) | 32 | 10.7K | 10.8 | 0.03–0.11 | 14–29 | Moderate ratio, some overfit |
| 007 (nomic, ws=1, hd=64) | 64 | 22.8K | 22.4 | -0.01–0.06 | 8–22 | High ratio, well-controlled gap |
| 008 (nomic, ws=1, hd=128) | 128 | 53.8K | 52.7 | -0.02–0.01 | 4–8 | High ratio, fast convergence |
| 010 (nomic, ws=1, hd=64) | 64 | 22.8K | 22.4 | -0.03–0.06 | 8–22 | Same as 007, SOTA |
| 014 (nomic, ws=1, hd=64, LR valley) | 64 | 22.8K | 22.4 | -0.02–0.02 | 6–16 | Faster convergence, tighter gap |

**Capacity sweet spot confirmed at hd=64.** hd=128 (run_008) did not improve QWK (0.344 avg) vs hd=64 (0.413 best) while doubling params. hd=32 (run_006) has best ratio (10.8) but lower QWK (0.358 best).

**LR finder tightened the train-val gap** (run_014 gaps: -0.023 to +0.018 vs run_010: -0.027 to +0.059) but didn't improve test metrics for most losses. The tighter gap suggests better regularization but the test performance drop for CORN/CORAL indicates the model found a different (worse) basin.

**Early stopping behaved correctly** across all runs. No run hit the 100-epoch ceiling.

---

## 7. Systemic Insights & Hypotheses

### The Overarching Story

These 14 runs tell a consistent story: **the VIF critic has learned to detect the _presence_ of value-relevant behavior (alignment, +1) but not _misalignment_ (-1)**. Every metric improvement — better QWK, higher calibration — has come from better majority-class and alignment-class predictions. The misalignment detection problem (recall_minus1 = 5–9%) has not budged across any architectural or LR intervention.

### Hypothesis 1: The Loss Landscape Has Two Regimes

The run_014 results reveal that CORAL/CORN and EMD/SoftOrdinal live in fundamentally different loss landscapes:
- **Threshold-based losses (CORAL/CORN)** have sharp optima sensitive to LR. The 17–20x valley LR overshoots these optima, causing large regressions (-0.035 to -0.120 QWK).
- **Distribution-based losses (EMD/SoftOrdinal)** have smoother landscapes. SoftOrdinal *improved* with the 20x LR (+0.080 QWK), suggesting its original LR of 0.001 was actually too conservative.

**Implication:** Loss-specific LR tuning (not a single LR finder pass) is needed. The `lr_steep` heuristic (used in run_013) was less aggressive and produced more consistent results across losses.

### Hypothesis 2: The Model Cannot Learn -1 From Text Alone

With recall_minus1 stuck at 5–9% across ALL 14 runs and ALL loss functions, this is not a capacity, LR, or loss-function problem. The frozen nomic-256d encoder may not produce sufficiently discriminative embeddings to distinguish "misaligned behavior" from "neutral behavior" in journal text. The -1 signal may require:
- Explicit value-behavior contradiction features (not available from frozen embeddings)
- Historical trajectory context that ws=1 cannot capture (but ws=2 didn't help either)
- Loss-level intervention that directly penalizes -1 misclassification (CDW-CE, cost-sensitive weighting)

---

## 8. Actionable Recommendations

### 8.1. CDW-CE Loss with Alpha Tuning (HIGH PRIORITY)

**Evidence:** All current losses treat class-distance errors uniformly. CDW-CE (Polat et al. 2025, arxiv 2412.01246) penalizes distant misclassifications by `|i-c|^alpha`, directly penalizing the 0-when-truth-is-(-1) error that causes our hedging problem. On the LIMUC benchmark (4-class ordinal, similar imbalance), CDW-CE outperformed CORAL, CORN, and CO2 on QWK.

**Action:** Implement CDW-CE loss operating on the existing 30-logit softmax architecture. Sweep alpha in {2, 3, 5}. Use ws=1, hd=64, LR=0.001.

**Watch for:** recall_minus1 > 0.15 (doubling current best) AND QWK > 0.40 (maintaining frontier).

### 8.2. Post-hoc Logit Adjustment on Existing Models (FREE)

**Evidence:** Menon et al. (ICLR 2021) showed that adding `tau * log(pi_y)` to logits at inference shifts the decision boundary toward minority classes without retraining. Our class priors (pi_0 ~ 0.82, pi_{-1} ~ 0.09, pi_{+1} ~ 0.09) have extreme skew amenable to adjustment.

**Action:** Apply logit adjustment with tau in {0.3, 0.5, 0.7, 1.0} to EMD and SoftOrdinal outputs from existing run_010/014 checkpoints (if saved). For CORAL/CORN, adjust cumulative probability thresholds instead.

**Watch for:** recall_minus1 improvement without calibration dropping below 0.70.

### 8.3. Loss-Specific LR Scheduling (MEDIUM PRIORITY)

**Evidence:** run_014 proved SoftOrdinal benefits from ~20x LR while CORN is destroyed by it. The lr_steep heuristic (3.7x) may be appropriate for CORAL/CORN while the valley LR works for SoftOrdinal.

**Action:** Run ws=1/hd=64 with per-loss LR: CORN at 0.001 (proven), SoftOrdinal at lr_valley (~0.020), EMD at lr_steep (~0.002), CORAL at 0.001.

**Watch for:** SoftOrdinal QWK > 0.40 with calibration > 0.83.

### 8.4. Soft Ordinal Labels (SORD) for Calibration Recovery

**Evidence:** run_014 showed calibration degradation with aggressive LR. SORD (Diaz & Marathe, CVPR 2019) spreads probability mass to adjacent classes during training, acting as ordinal-aware label smoothing that improves calibration without reducing discriminative power. ORCU (2024) extends this with unimodal regularization.

**Action:** Replace one-hot training labels with soft ordinal encoding: `soft_label_i = exp(-|i-c|) / Z`. Combine with CDW-CE.

**Watch for:** Calibration > 0.85 with QWK maintained.

### 8.5. Investigate LR Finder Bug (QUICK FIX)

**Evidence:** `lr_find_history.json` is identical to `lr_find_SoftOrdinal.json` in the ws=1 run. The `config_delta` for CORN shows 0.01979 but actual applied LR was 0.01683. These suggest a bookkeeping bug in the LR finder or experiment logger.

**Action:** Audit the LR finder code path to confirm each loss function gets an independent finder run. Fix the config_delta to log per-loss LR correctly.

---

## 9. Summary Verdict

**Best config:** `run_010 CORN` (nomic-256d, ws=1, hd=64, LR=0.001) — QWK 0.434 (moderate), Cal 0.835, MAE 0.206. Remains SOTA after run_014 failed to improve it.

**Notable:** `run_014 SoftOrdinal` (QWK 0.388) is the best-ever SoftOrdinal and suggests this loss benefits from higher LR — but still trails CORN by 0.046 QWK and has worse calibration (0.801 vs 0.835).

**Key weakness:** recall_minus1 = 5–9% across ALL 14 runs and ALL losses. The model is nearly blind to misalignment (-1), which is the core value proposition of Twinkl. No architectural, capacity, or LR intervention has moved this metric. This is a loss-function and class-imbalance problem requiring targeted intervention.

**Highest-leverage next experiment:** **CDW-CE loss** (alpha=2–5) on ws=1/hd=64/LR=0.001. This directly attacks the hedging problem by penalizing the 0-when-truth-is-(-1) error with ordinal-distance-weighted penalties. It operates on the existing softmax architecture, requires no architectural changes, and has demonstrated superiority over CORAL/CORN on comparable ordinal benchmarks. If CDW-CE lifts recall_minus1 above 0.15 while maintaining QWK > 0.40, it would represent the first meaningful progress on the misalignment detection bottleneck.

---

*Research corroboration: CDW-CE recommendations validated against Polat et al. 2025 (arxiv 2412.01246, ESWA); logit adjustment against Menon et al. (ICLR 2021); SORD against Diaz & Marathe (CVPR 2019) and ORCU (arxiv 2410.15658). All techniques confirmed compatible with small-dataset (n~1000) ordinal classification with extreme class imbalance.*
