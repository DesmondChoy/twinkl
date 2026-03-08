---
name: experiment-review
description: Cross-run comparison of VIF experiment logs with qualitative assessment and recommendations. Use when LLM needs to synthesize results across runs in logs/experiments and produce a structured report with metric trade-offs, per-dimension behavior, calibration risks, capacity/overfitting analysis, and next-step experiments.
---

Analyze all VIF experiment logs, backfill any empty provenance/observations, and then produce a structured cross-run comparison report that synthesizes findings across runs.

**Role & Mindset**: Act as a Senior AI/Data Scientist. Do not just mechanically report numbers. Look for interacting variables (e.g., does increasing capacity only help when using a specific state encoder?), synthesize what the metrics imply about the model's fundamental understanding of the task, and formulate hypotheses about *why* certain interventions succeeded or failed.

## Split-Aware Frontier Guardrails

Before making any leaderboard claim, partition runs by **evaluation regime**. This project has a known split boundary:

- Commit `d937094` changed validation/test splitting to persona-level multi-label stratification.
- Runs `run_001`-`run_015` are the **historical pre-`d937094` frontier** and must stay on a historical board only.
- Runs `run_016+` are the **active corrected-split frontier** unless a later split change is explicitly documented.

Rules:

1. Never place pre- and post-`d937094` runs on the same SOTA leaderboard.
2. Cross-regime comparisons are allowed only as context for why the frontier changed, not as direct performance ranking.
3. When multiple corrected-split runs share the same config but differ only by model seed, treat them as one **candidate family** and summarize with median and IQR across seeds rather than cherry-picking the best single seed.
4. Base recommendations and observations on the active corrected-split frontier by default. Use the historical frontier only for background, hypotheses, or regression context.
5. If split metadata is incomplete, infer the regime from `config.data.split_seed`, `metadata.run_id`, `provenance.git_log`, and the experiment index. If still ambiguous, state the ambiguity explicitly and do not mix regimes in the same leaderboard.

Current project context to preserve unless superseded by newer corrected-split evidence:

- `CDWCE_a3` is the active corrected-split leader on median `qwk_mean` and `recall_minus1`.
- `SoftOrdinal` is the best minority-sensitive corrected-split option on median `minority_recall_mean` and hedging.
- `CORN` is the best-calibrated corrected-split baseline.

## Data Collection

### Step 1: Read the index

Read `logs/experiments/index.md` for the high-level summary tables and the latest findings. Check whether it already contains both:

- `## Current Frontier (Post-d937094 Split)`
- `## Historical Frontier (Pre-d937094 Split)`

### Step 2: Read all run files

Use Glob to find `logs/experiments/runs/*.yaml`, then read every file. Extract:
- `metadata` (run_id, model_name)
- `provenance` (prev_run_id, git_log, config_delta, rationale)
- `config` (encoder, state_encoder, model, training)
- `data` (n_train, pct_truncated, state_dim)
- `capacity` (n_parameters, param_sample_ratio)
- `training_dynamics` (best_epoch, gap_at_best)
- `evaluation` (all aggregate metrics)
- `per_dimension` (all 10 Schwartz dimensions)

### Step 3: Identify axes of variation

First, group runs by **split regime / evaluation regime**. Only then group within each regime by what changed:
- **Encoder**: model name, embedding dimension, truncation
- **Capacity**: hidden_dim, param count, param/sample ratio
- **Loss function**: within a run, compare across loss heads
- **State encoder**: window_size, state_dim

Anything identical across all runs is a **constant** — mention once, don't repeat.

## Step 4: Provenance & Observation Backfill

Before generating your analytical report, check all run YAML files for empty `provenance.rationale` and `observations` fields. Writing these out explicitly will help solidify your understanding of the run trajectory. For each file in `logs/experiments/runs/`:

1. **Empty `provenance.rationale`**: Generate 1–3 sentences explaining *why* this run was created, based on:
   - `provenance.git_log` (what code changes preceded this run)
   - `provenance.config_delta` (what config changed vs the previous run)
   - If both are empty (first run), write a brief note like "Baseline run establishing initial metrics for [encoder family]."

2. **Empty or placeholder `observations`**: Generate 2–4 sentences summarizing *what was learned*, based on:
   - `evaluation` metrics (MAE, QWK, calibration, hedging)
   - `per_dimension` results (which dimensions improved/degraded)
   - Comparison with the previous run (if `provenance.prev_run_id` exists)
   - A placeholder is any value containing `<fill in` or that is empty.

3. **Write back**: Use the Edit tool to update only fields that are empty or contain a placeholder (any value matching `<fill in`). **Never overwrite** fields that already have substantive content — i.e., anything that is neither empty nor a placeholder.

Once all historical context and observations are properly recorded, proceed to generate the cross-run comparison report.

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

After completing data collection and backfilling, produce the report in exactly these 9 sections. Cap the report at ~1000 words excluding tables. Cite specific numbers and use run IDs (e.g., run_001, run_002). When two runs differ by < 5% on a metric, say "comparable" rather than declaring a winner.

### 1. Experiment Overview

- What varied across runs (encoder, capacity, loss, etc.)
- What stayed constant (training hyperparameters, data splits, seed)
- Dataset size and any data notes (e.g., truncation %)
- If both historical and corrected-split runs are present, explicitly name the active regime and state that leaderboard claims are made within that regime only.

### 2. Head-to-Head Comparison

For each axis of variation, produce a compact comparison table. Flag the winner per metric. Use bold for the better value. Example:

| Metric | run_001 (MiniLM-384d) | run_002 (nomic-256d) | Delta |
|--------|-----------------------|----------------------|-------|

Cover: MAE, Accuracy, QWK, Spearman, Calibration, Minority Recall, Hedging.

If multiple split regimes are present, use separate comparison tables per regime. Do not produce a single winner table that mixes pre- and post-`d937094` runs.

If there are multiple loss functions, also compare within each run across losses.

### 3. Per-Dimension Analysis

Identify:
- **Easy dimensions**: consistently high QWK (> 0.4) across runs
- **Hard dimensions**: consistently low QWK (< 0.3) across runs
- **Volatile dimensions**: large QWK variance across runs

Present as a compact table sorted by mean QWK across all runs.

**Error Analysis (Hardest Dimensions):**
For the 2 hardest dimensions (lowest mean QWK), write and execute a temporary read-only Python script that:
1. Loads the best checkpoint for the top-performing run from its output directory
2. Runs inference on the validation split
3. Extracts 2–3 samples with the highest absolute error on each hard dimension
4. Displays the journal excerpt (truncated to ~100 words), ground-truth label, and model prediction

This provides qualitative context on *why* the model struggles. Do not save artifacts or modify any repository files.

If no checkpoint is available (e.g., it was cleaned up), skip this analysis and note it in the report.

### 4. Calibration Deep-Dive

- How many dimensions have positive calibration per run?
- Global calibration comparison
- Flag any dimension with calibration < -0.4 as a deployment risk
- Note if negative calibration is systematic (model always over/under-confident)

### 5. Hedging vs Minority Recall Trade-off

- For each run × loss combination, present a comparison table with hedging % and minority recall side by side. Example format:

| Run + Loss | Hedging % | Minority Recall | Verdict |
|------------|-----------|-----------------|---------|

- Mark configurations that achieve both hedging < 60% **and** minority recall > 30% as **decisive + balanced**
- Flag any loss where hedging > 60% (over-predicting the majority class)

### 6. Capacity & Overfitting

- Compare param/sample ratios and characterize using thresholds
- Compare training gaps (train_loss - val_loss at best epoch)
- Note best_epoch vs total_epochs (did early stopping trigger appropriately?)
- Flag if a larger model overfits more than a smaller one

### 7. Systemic Insights & Hypotheses

Take a step back from the individual metrics and read between the lines. Synthesize what the experiments reveal about the model's fundamental understanding of the task:
- What is the overarching story these experiments are telling us? 
- Are there hidden interactions? (e.g., "The model isn't just failing on Power; it's failing on any dimension that relies heavily on historical state rather than immediate journal text.")
- What assumptions did we make in the previous runs that this data proves wrong?
- Formulate 1-2 strong hypotheses about *why* the model is behaving the way it is.

### 8. Actionable Recommendations

Provide 3–5 concrete, motivated, testable next steps. Each should:
- Reference the specific evidence from the analysis
- Be scoped to a single experiment or change
- Include what metric improvement to watch for
- Default to the active corrected-split frontier. If you recommend reviving a historical configuration, explain why it is still plausible under the corrected split rather than assuming the old board still applies.

**Automated Investigations:**
Do not ask the user to manually investigate data distributions or anomalies. If a recommendation involves investigating data (e.g., checking label distributions for a dimension where QWK is near-zero, auditing dataset changes), **you must automatically perform this analysis**. Write and execute the necessary read-only Python scripts (e.g., using `pandas` on `logs/judge_labels/judge_labels.parquet` or `logs/registry/personas.parquet`) to find the answer immediately. Include the specific findings directly in your report. Do not make any code changes to the repository; only use temporary scripts to gather the information needed to validate your hypotheses.

**Corroborate with Web Research:**
Before finalizing your recommendations, spawn parallel sub-agents or use tools like `search_web` to research current state-of-the-art approaches relevant to your findings. Use this research to ensure your recommendations (e.g., loss functions, model capacities, or data sampling techniques) are based on up-to-date best practices and not outdated knowledge. Mention in the report what research was conducted and how it corroborated or modified your recommendations.

Examples of good recommendations:
- "Try nomic-embed with hidden_dim=128 (between 32 and 256) to find the capacity sweet spot — watch param/sample ratio and training gap"
- "Power dimension has near-zero QWK across all runs. *[Automatically checked `logs/judge_labels/judge_labels.parquet`: found 92% 0-labels.]* Recommendation: implement binary collapsing for Power or use dimension-specific focal loss."

### 9. Summary Verdict

- **Best config**: which run_id + loss combination looks most promising and why
- **Key weakness**: the single biggest limitation across all experiments
- **Highest-leverage next experiment**: one thing to try that would most improve results

## Style Constraints

- Use the threshold table above for all qualitative characterizations
- Always cite the actual number: "QWK 0.42 (moderate)" not just "moderate QWK"
- Context: this is a capstone POC with ~637 training samples — focus on relative comparisons, not absolute benchmarks
## Leaderboard Updates

After generating the report, update `logs/experiments/index.md` with **split-aware** frontier sections.

1. Ensure the file has both of these sections near the top:
   - `## Current Frontier (Post-d937094 Split)`
   - `## Historical Frontier (Pre-d937094 Split)`
2. Update the **current frontier** using only corrected-split runs. When the frontier is based on a controlled multi-seed matrix, use one row per candidate family with:
   - candidate name
   - supporting run IDs
   - split seed
   - model seeds
   - median `qwk_mean`
   - median `recall_minus1`
   - median `minority_recall_mean`
   - median hedging
   - median `calibration_global`
   - a one-sentence rationale
3. Prefer median/IQR summaries over best-seed cherry-picking. A single corrected-split run can appear on the board only when no seed-matched family summary exists yet, and it should be labeled provisional.
4. Keep the historical frontier archival. Never promote a pre-`d937094` run onto the active board, even if its raw QWK is higher.
5. If the active frontier changes, add a newest-first entry to the `## Findings` section describing what changed and how recommendations shift.
6. Make observations and future recommendations consistent with the active board. Historical findings may inform hypotheses, but they should not drive next-step recommendations when corrected-split evidence disagrees.
