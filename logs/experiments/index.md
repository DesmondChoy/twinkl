# VIF Experiment Index

## Current State of the Art

| Rank | Run + Loss | Encoder | hd | n_train | QWK | Cal | MAE | Acc | MinR | Rationale |
|------|-----------|---------|---:|--------:|----:|----:|----:|----:|-----:|-----------|
| 1 | run_007 CORN | nomic-256d | 64 | 1020 | **0.413** | **0.838** | 0.205 | 0.821 | 0.285 | Best QWK on expanded data with good calibration. Conformity breakthrough (0.535). |
| 2 | run_007 CORAL | nomic-256d | 64 | 1020 | 0.367 | 0.830 | 0.208 | 0.819 | 0.247 | Strong calibration, benevolence QWK 0.532. Stable across capacity changes. |
| 3 | run_008 EMD | nomic-256d | 128 | 1020 | 0.365 | 0.802 | **0.200** | 0.821 | 0.300 | Best MAE ever. Stimulation QWK 0.667. Most capacity-robust loss function. |

> **Key insight**: nomic-embed at hd=64 is the sweet spot. All top-3 are nomic-based on the 1020-sample dataset. MiniLM is no longer competitive on the expanded data.
>
> **Primary bottleneck**: QWK (0.413) is fair but below the moderate threshold (>0.4). More critically, -1 recall is just 10.3% — the model almost completely fails to detect value misalignment, which is the signal Twinkl exists to surface. Hedging exceeds 80% across all runs. Next experiments should target class-imbalance interventions (loss reweighting, focal loss, oversampling) to boost minority recall alongside QWK. See [`docs/evals/value_modeling_eval.md`](../../docs/evals/value_modeling_eval.md) for metric definitions and targets.

## Run Log

<!-- AUTO-TABLE:START -->
| run | model | encoder | ws | hd | do | loss | params | ratio | MAE | Acc | QWK | Spear | Cal | MinR | file |
|-----|-------|---------|---:|---:|---:|------|-------:|------:|----:|----:|----:|------:|----:|-----:|------|
| 001 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 372756 | 585.2 | 0.232 | 0.782 | 0.398 | 0.459 | 0.644 | 0.298 | runs/run_001_CORAL.yaml |
| 001 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 372756 | 585.2 | 0.236 | 0.782 | 0.384 | 0.452 | 0.633 | 0.306 | runs/run_001_CORN.yaml |
| 001 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 375326 | 589.2 | 0.243 | 0.773 | 0.395 | 0.459 | 0.648 | 0.340 | runs/run_001_EMD.yaml |
| 001 | MSE | MiniLM-384d | 3 | 256 | 0.2 | weighted_mse_s5.0 | 370186 | 581.1 | 0.450 | 0.641 | 0.338 | 0.379 | -0.218 | 0.428 | runs/run_001_MSE.yaml |
| 001 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 375326 | 589.2 | 0.248 | 0.777 | 0.417 | 0.455 | 0.724 | 0.372 | runs/run_001_SoftOrdinal.yaml |
| 002 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10708 | 16.8 | 0.263 | 0.770 | 0.335 | 0.349 | 0.734 | 0.234 | runs/run_002_CORAL.yaml |
| 002 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10708 | 16.8 | 0.260 | 0.766 | 0.355 | 0.371 | 0.737 | 0.204 | runs/run_002_CORN.yaml |
| 002 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 11038 | 17.3 | 0.270 | 0.764 | 0.365 | 0.365 | 0.772 | 0.309 | runs/run_002_EMD.yaml |
| 002 | MSE | nomic-256d | 1 | 32 | 0.3 | weighted_mse_s5.0 | 10378 | 16.3 | 0.418 | 0.683 | 0.348 | 0.378 | -0.073 | 0.390 | runs/run_002_MSE.yaml |
| 002 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 11038 | 17.3 | 0.267 | 0.780 | 0.385 | 0.356 | 0.774 | 0.310 | runs/run_002_SoftOrdinal.yaml |
| 003 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 370196 | 581.2 | 0.241 | 0.776 | 0.369 | 0.394 | 0.640 | 0.265 | runs/run_003_CORAL.yaml |
| 003 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 370196 | 581.2 | 0.249 | 0.764 | 0.326 | 0.395 | 0.648 | 0.260 | runs/run_003_CORN.yaml |
| 003 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 372766 | 585.2 | 0.240 | 0.782 | 0.410 | 0.416 | 0.697 | 0.332 | runs/run_003_EMD.yaml |
| 003 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 372766 | 585.2 | 0.253 | 0.772 | 0.383 | 0.396 | 0.743 | 0.333 | runs/run_003_SoftOrdinal.yaml |
| 004 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10388 | 16.3 | 0.273 | 0.768 | 0.331 | 0.346 | 0.755 | 0.269 | runs/run_004_CORAL.yaml |
| 004 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10388 | 16.3 | 0.274 | 0.760 | 0.291 | 0.293 | 0.734 | 0.217 | runs/run_004_CORN.yaml |
| 004 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 10718 | 16.8 | 0.266 | 0.780 | 0.391 | 0.361 | 0.766 | 0.343 | runs/run_004_EMD.yaml |
| 004 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 10718 | 16.8 | 0.269 | 0.779 | 0.385 | 0.351 | 0.786 | 0.332 | runs/run_004_SoftOrdinal.yaml |
| 005 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 370196 | 386.4 | 0.215 | 0.798 | 0.282 | 0.411 | 0.629 | 0.186 | runs/run_005_CORAL.yaml |
| 005 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 370196 | 386.4 | 0.216 | 0.797 | 0.254 | 0.365 | 0.640 | 0.181 | runs/run_005_CORN.yaml |
| 005 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 372766 | 389.1 | 0.226 | 0.791 | 0.272 | 0.312 | 0.651 | 0.217 | runs/run_005_EMD.yaml |
| 005 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 372766 | 389.1 | 0.215 | 0.798 | 0.304 | 0.368 | 0.650 | 0.165 | runs/run_005_SoftOrdinal.yaml |
| 006 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10388 | 10.8 | 0.233 | 0.793 | 0.278 | 0.354 | 0.768 | 0.166 | runs/run_006_CORAL.yaml |
| 006 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10388 | 10.8 | 0.236 | 0.791 | 0.280 | 0.350 | 0.777 | 0.183 | runs/run_006_CORN.yaml |
| 006 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 10718 | 11.2 | 0.227 | 0.803 | 0.324 | 0.334 | 0.764 | 0.225 | runs/run_006_EMD.yaml |
| 006 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 10718 | 11.2 | 0.223 | 0.811 | 0.358 | 0.354 | 0.775 | 0.235 | runs/run_006_SoftOrdinal.yaml |
| 007 | CORAL | nomic-256d | 1 | 64 | 0.3 | coral | 22804 | 22.4 | 0.208 | 0.819 | 0.367 | 0.398 | 0.830 | 0.247 | runs/run_007_CORAL.yaml |
| 007 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.4 | 0.205 | 0.821 | 0.413 | 0.402 | 0.838 | 0.285 | runs/run_007_CORN.yaml |
| 007 | EMD | nomic-256d | 1 | 64 | 0.3 | emd | 23454 | 23.0 | 0.211 | 0.817 | 0.357 | 0.363 | 0.849 | 0.288 | runs/run_007_EMD.yaml |
| 007 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 23.0 | 0.212 | 0.821 | 0.314 | 0.343 | 0.852 | 0.291 | runs/run_007_SoftOrdinal.yaml |
| 008 | CORAL | nomic-256d | 1 | 128 | 0.3 | coral | 53780 | 52.7 | 0.204 | 0.824 | 0.341 | 0.397 | 0.805 | 0.273 | runs/run_008_CORAL.yaml |
| 008 | CORN | nomic-256d | 1 | 128 | 0.3 | corn | 53780 | 52.7 | 0.203 | 0.828 | 0.344 | 0.359 | 0.785 | 0.276 | runs/run_008_CORN.yaml |
| 008 | EMD | nomic-256d | 1 | 128 | 0.3 | emd | 55070 | 54.0 | 0.200 | 0.821 | 0.365 | 0.390 | 0.802 | 0.300 | runs/run_008_EMD.yaml |
| 008 | SoftOrdinal | nomic-256d | 1 | 128 | 0.3 | soft_ordinal | 55070 | 54.0 | 0.201 | 0.826 | 0.354 | 0.387 | 0.811 | 0.291 | runs/run_008_SoftOrdinal.yaml |
| 009 | CORAL | MiniLM-384d | 3 | 64 | 0.2 | coral | 80276 | 78.7 | 0.228 | 0.785 | 0.176 | 0.284 | 0.695 | 0.137 | runs/run_009_CORAL.yaml |
| 009 | CORN | MiniLM-384d | 3 | 64 | 0.2 | corn | 80276 | 78.7 | 0.227 | 0.792 | 0.227 | 0.323 | 0.711 | 0.166 | runs/run_009_CORN.yaml |
| 009 | EMD | MiniLM-384d | 3 | 64 | 0.2 | emd | 80926 | 79.3 | 0.225 | 0.799 | 0.259 | 0.303 | 0.776 | 0.223 | runs/run_009_EMD.yaml |
| 009 | SoftOrdinal | MiniLM-384d | 3 | 64 | 0.2 | soft_ordinal | 80926 | 79.3 | 0.239 | 0.787 | 0.236 | 0.300 | 0.775 | 0.234 | runs/run_009_SoftOrdinal.yaml |
<!-- AUTO-TABLE:END -->

## Findings

### 2026-02-22 — Resolved: Universalism QWK collapse (run_003 → run_009)

Universalism QWK dropped from 0.732 (run_003 EMD, 637 train) to 0.042 (run_009 EMD, 1020 train). This looks alarming but is explained by three compounding factors — **no further data intervention is needed**.

**1. The 0.732 was inflated by a skewed test distribution.** run_003 included 10 pre-tension Universalism personas whose labels were 87% +1, 13% 0, and **0% −1** (98 entries total). The model achieved high QWK by predicting +1 for the dominant class without ever needing to detect misalignment.

**2. Removing those personas (commit a036004) was correct.** Batch 1B replacement personas (generated with tension-selection) have 54.8% −1 labels, giving the critic a balanced signal. The drop to QWK 0.290 in run_005 reflects honest evaluation against a harder, more realistic distribution — not a regression.

**3. MiniLM's collapse to 0.042 (run_009) is an encoder bottleneck, not a data problem.** MiniLM at window_size=3 produces a 1164-dim state vector; at hd=64, this is an 18:1 compression bottleneck. On the **same 1020-sample dataset**, nomic CORN at hd=64 (266-dim state, 4:1 compression) achieves Universalism QWK **0.466**.

| Run | Encoder | hd | n_train | Uni QWK | Uni Hedge | State dim |
|-----|---------|---:|--------:|--------:|----------:|----------:|
| run_003 EMD | MiniLM | 256 | 637 | 0.732 | 74.8% | 1164 |
| run_005 EMD | MiniLM | 256 | 958 | 0.290 | 84.7% | 1164 |
| run_007 CORN | nomic | 64 | 1020 | 0.466 | 81.9% | 266 |
| run_009 EMD | MiniLM | 64 | 1020 | 0.042 | 82.4% | 1164 |

**Conclusion**: Universalism performance is recovered by the encoder switch to nomic (run_007). No additional Universalism persona generation or dimension-specific loss weighting is warranted.

### 2026-02-22 — Resolved: MiniLM retired from future experiments

MiniLM (all-MiniLM-L6-v2) is **no longer a candidate encoder** for future VIF runs. All future experiments, analysis, and recommendations should focus exclusively on nomic-embed-text-v1.5.

**1. MiniLM's state pipeline is fundamentally over-parameterized for this dataset.** The window_size=3 state encoder produces a 1164-dim state vector. Even at hd=64, this yields an 80K-parameter model with a 79:1 param/sample ratio — still firmly in the "high" regime. The model consistently early-stops at epoch 2-5, never learning beyond majority-class hedging.

**2. MiniLM degraded on every metric when the dataset expanded.** From 637 → 1020 train samples, MiniLM QWK dropped across all losses (e.g., EMD: 0.410 → 0.259; SoftOrdinal: 0.383 → 0.236). More data amplified the class-imbalance signal rather than improving generalization.

**3. nomic dominates on the expanded dataset at every capacity level tested.** On the same 1020-train data, nomic at hd=64 (22:1 ratio) achieves QWK 0.413 vs MiniLM at hd=64 (79:1 ratio) achieving QWK 0.227. The gap is too large to close with hyperparameter tuning.

| Metric (avg across losses) | MiniLM hd=256 (run_005) | MiniLM hd=64 (run_009) | nomic hd=64 (run_007) |
|-----------------------------|------------------------:|------------------------:|----------------------:|
| QWK | 0.278 | 0.224 | **0.363** |
| Calibration | 0.643 | 0.739 | **0.839** |
| Minority Recall | 0.187 | 0.190 | **0.278** |
| Hedging | 86.6% | 85.4% | **81.7%** |
| Param/sample ratio | 388 (severe) | 79 (high) | **22 (high)** |

**Conclusion**: MiniLM runs (001, 003, 005, 009) are retained as historical baselines. Future `/experiment-review` reports should treat MiniLM as a closed investigation and focus all insights, comparisons, and recommendations on nomic-embed configurations only.
