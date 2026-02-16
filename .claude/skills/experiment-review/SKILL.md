---
name: experiment-review
description: Cross-run comparison of VIF experiment logs with qualitative assessment and recommendations. Use when LLM needs to synthesize results across runs in logs/experiments and produce a structured report with metric trade-offs, per-dimension behavior, calibration risks, capacity/overfitting analysis, and next-step experiments.
---

Analyze all VIF experiment logs and produce a structured cross-run comparison report. This skill is read-only — it interprets existing data and outputs findings to the conversation.

## Data Collection

### Step 1: Read the index

Read `logs/experiments/index.md` for the high-level summary table.

### Step 2: Read all run files

Use Glob to find `logs/experiments/runs/*.yaml`, then read every file. Extract:
- `metadata` (run_id, model_name)
- `config` (encoder, state_encoder, model, training)
- `data` (n_train, pct_truncated, state_dim)
- `capacity` (n_parameters, param_sample_ratio)
- `training_dynamics` (best_epoch, gap_at_best)
- `evaluation` (all aggregate metrics)
- `per_dimension` (all 10 Schwartz dimensions)

### Step 3: Identify axes of variation

Group runs by what changed between them:
- **Encoder**: model name, embedding dimension, truncation
- **Capacity**: hidden_dim, param count, param/sample ratio
- **Loss function**: within a run, compare across loss heads
- **State encoder**: window_size, state_dim

Anything identical across all runs is a **constant** — mention once, don't repeat.

## Metric Interpretation Thresholds

Use these thresholds when characterizing results. Always cite the actual number alongside the qualitative label.

| Metric | Poor | Fair | Moderate | Good/Substantial |
|--------|------|------|----------|------------------|
| QWK | < 0.2 | 0.2 – 0.4 | 0.4 – 0.6 | > 0.6 |
| Calibration | < 0 (dangerous) | 0 – 0.1 (useless) | 0.1 – 0.3 (weak) | 0.3 – 0.6 (moderate), > 0.6 (good) |
| Hedging % | — | — | 60 – 80% (moderate) | < 60% (decisive) |
| Hedging % (excessive) | > 80% | — | — | — |
| Minority Recall | < 10% (ignores rare) | 10 – 30% (poor) | > 30% (reasonable) | — |
| Param/Sample Ratio | > 100 (severe) | 10 – 100 (high) | 1 – 10 (moderate) | < 1 (efficient) |
| Training Gap | > 0.5 (overfitting) | 0.1 – 0.5 (some) | < 0.1 (good) | — |
| Spearman | < 0.3 (weak) | 0.3 – 0.5 (moderate) | > 0.5 (strong) | — |

## Report Structure

Produce the report in exactly these 8 sections. Cap the report at ~800 words excluding tables. Cite specific numbers and use run IDs (e.g., run_001, run_002). When two runs differ by < 5% on a metric, say "comparable" rather than declaring a winner.

### 1. Experiment Overview

- What varied across runs (encoder, capacity, loss, etc.)
- What stayed constant (training hyperparameters, data splits, seed)
- Dataset size and any data notes (e.g., truncation %)

### 2. Head-to-Head Comparison

For each axis of variation, produce a compact comparison table. Flag the winner per metric. Use bold for the better value. Example:

| Metric | run_001 (MiniLM-384d) | run_002 (nomic-256d) | Delta |
|--------|-----------------------|----------------------|-------|

Cover: MAE, Accuracy, QWK, Spearman, Calibration, Minority Recall, Hedging.

If there are multiple loss functions, also compare within each run across losses.

### 3. Per-Dimension Analysis

Identify:
- **Easy dimensions**: consistently high QWK (> 0.4) across runs
- **Hard dimensions**: consistently low QWK (< 0.3) across runs
- **Volatile dimensions**: large QWK variance across runs

Present as a compact table sorted by mean QWK across all runs.

### 4. Calibration Deep-Dive

- How many dimensions have positive calibration per run?
- Global calibration comparison
- Flag any dimension with calibration < -0.4 as a deployment risk
- Note if negative calibration is systematic (model always over/under-confident)

### 5. Hedging vs Minority Recall Trade-off

- For each loss function, plot the hedging % against minority recall
- Identify which losses achieve low hedging AND reasonable minority recall
- Flag any loss where hedging > 60% (over-predicting the majority class)

### 6. Capacity & Overfitting

- Compare param/sample ratios and characterize using thresholds
- Compare training gaps (train_loss - val_loss at best epoch)
- Note best_epoch vs total_epochs (did early stopping trigger appropriately?)
- Flag if a larger model overfits more than a smaller one

### 7. Actionable Recommendations

Provide 3–5 concrete, motivated, testable next steps. Each should:
- Reference the specific evidence from the analysis
- Be scoped to a single experiment or change
- Include what metric improvement to watch for

Examples of good recommendations:
- "Try nomic-embed with hidden_dim=128 (between 32 and 256) to find the capacity sweet spot — watch param/sample ratio and training gap"
- "Investigate why power dimension has near-zero QWK across all runs — check label distribution"

### 8. Summary Verdict

- **Best config**: which run_id + loss combination looks most promising and why
- **Key weakness**: the single biggest limitation across all experiments
- **Highest-leverage next experiment**: one thing to try that would most improve results

## Style Constraints

- Use the threshold table above for all qualitative characterizations
- Always cite the actual number: "QWK 0.42 (moderate)" not just "moderate QWK"
- Context: this is a capstone POC with ~637 training samples — focus on relative comparisons, not absolute benchmarks
- Do not editorialize about the project or its goals — stick to the data
- If a metric is missing or an observations field says `<fill in>`, note it but don't speculate
