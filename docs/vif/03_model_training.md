# VIF – Reward Modeling & Training Strategy

This document describes the current training story for the Value Identity
Function (VIF): how Judge labels become student targets, what the live student
families are, and how frontier evaluation is managed.

---

## 1. Teacher-Student Training Stack

### 1.1 Why Reward Modeling Exists

For real users, Twinkl observes journal text and profile context, not an oracle
alignment score. The project therefore uses an explicit teacher-student setup:

1. **Generator** creates synthetic personas and journal trajectories.
2. **Judge** labels each entry against the Schwartz dimensions.
3. **Critic** distills those labels into a fast supervised student.

This lets the live scoring path stay fast and uncertainty-aware without calling
an LLM on every runtime inference.

### 1.2 Judge Output Contract

The Judge emits a per-dimension categorical label:

- `-1`: misaligned
- `0`: neutral / not enough evidence
- `+1`: aligned

These labels are the current student target surface.

---

## 2. Training Target: What Is Current vs Exploratory

### 2.1 Current Mainline Target

The live stack uses **Option A: immediate alignment**.

$$
\vec{V}_\theta(s_{u,t}) \approx \hat{\vec{a}}_{u,t}
$$

That means the student predicts the Judge's current-entry alignment labels, and
longer-horizon drift logic is derived downstream from aggregated outputs rather
than learned as a discounted-return target.

### 2.2 Exploratory Alternatives

The older design space still matters conceptually, but it is not the active
training path:

- **Option B**: short-horizon forecast targets
- **Option C**: discounted return targets

These remain future research directions, not the current implementation.

### 2.3 Scalar Aggregation

When a single summary score is needed, the vector output can be aggregated with
profile weights:

$$
V^{\text{scalar}}_{u,t} = w_u^\top \vec{V}_\theta(s_{u,t})
$$

This scalar is for summaries and downstream trigger logic. The student itself is
trained to preserve the vector of trade-offs.

---

## 3. Student Model Families

### 3.1 Shared Input Setup

All student variants consume the same state vector described in
[System Architecture, State, and Runtime Flow](02_system_architecture.md):

- frozen sentence embeddings
- optional recent-history window
- time-gap features
- 10-dim value-profile weights

The encoder is frozen in the current POC for simplicity, reproducibility, and
data efficiency.

### 3.2 Active Student Families

The mainline student is no longer a single regression head. The active training
stack now centers on a shared MLP backbone with multiple comparison families:

- **Ordinal heads**: CORAL, CORN, EMD, CDW-CE, SoftOrdinal
- **Long-tail baselines**: BalancedSoftmax, LDAM-DRW
- **Experimental reformulation**: two-stage BalancedSoftmax
- **Baselines retained for comparison**: legacy MSE MLP and Bayesian neural net

In practice, the corrected-split experiment board currently treats the
BalancedSoftmax family as the default frontier reference, while newer branches
such as the two-stage formulation remain diagnostic challengers rather than the
mainline default. See `logs/experiments/index.md` for the live ranking.

### 3.3 Uncertainty Path

For the MLP path, uncertainty is estimated with MC Dropout over repeated forward
passes. The BNN path provides a separate Bayesian baseline.

---

## 4. Data, Splits, and Evaluation

### 4.1 Data Path

Training rows are built by joining:

- wrangled journal entries from `logs/wrangled`
- consolidated Judge labels from `logs/judge_labels/judge_labels.parquet`

Each labeled entry becomes one `(state_vector, target_vector)` pair.

### 4.2 Split Policy

The current evaluation regime is persona-level and holdout-aware:

- train/val/test splits are by persona, not by entry
- validation/test are sign-stratified at the persona level
- optional fixed holdout manifests are used for augmentation and audit rounds

This is the only fair basis for comparing current frontier runs.

### 4.3 Metrics

The training stack tracks more than loss:

- QWK
- MAE
- accuracy / recall
- calibration
- raw probability/logit exports when needed
- circumplex diagnostics and related experiment summaries

### 4.4 Current Caveat: Reachability Audit

The training pipeline is operational, but the completed `twinkl-747`
reachability audit changed how the current board should be interpreted. The hard
dimensions, especially `security`, are not yet a clean long-term distillation
target for the current student. That means the current frontier should be read
as a useful experimental baseline, not the final target definition.

The latest consensus-label diagnostic branch (`run_048`-`run_050`) reinforces
that framing. It improved within-regime QWK and calibration on the
consensus-relabeled holdout, but it changed labels on the frozen test split and
did not replace the persisted-label frontier cleanly. The active corrected-split
default remains `run_019`-`run_021`.

---

## 5. Implementation Reference

The VIF training stack lives in `src/vif/`.

| Module | Description |
|--------|-------------|
| `src/vif/encoders.py` | Sentence-encoder wrapper and encoder creation |
| `src/vif/state_encoder.py` | State-vector construction |
| `src/vif/critic.py` | Legacy regression-style Critic MLP |
| `src/vif/critic_ordinal.py` | Active ordinal and long-tail head families |
| `src/vif/critic_bnn.py` | Bayesian neural baseline |
| `src/vif/dataset.py` | Data loading, joins, and persona-level splits |
| `src/vif/eval.py` | Evaluation metrics and uncertainty-aware evaluation |
| `src/vif/posthoc.py` | Validation-only post-hoc boundary tuning |
| `src/vif/experiment_logger.py` | Persisted run YAMLs and experiment index support |
| `src/vif/train.py` | General single-model training entrypoint |
| `src/vif/train_bnn.py` | BNN training entrypoint |
| `scripts/experiments/critic_training_v4_review.py` | Canonical frontier review driver |

---

## 6. Operational Notes

### 6.1 Current Runtime Choices

The current config defaults matter:

- encoder family is configured in `config/vif.yaml`
- the live default state currently uses `window_size: 1`
- MC Dropout uses 50 samples by default

These are implementation choices, not permanent design constraints.

### 6.2 Training Instrumentation

The general training entrypoint now includes:

- default LR-finder pass before training
- gradient clipping and gradient telemetry
- immediate non-finite loss termination with preserved artifacts

These used to live as design notes; they are now part of the implemented
training workflow.

### 6.3 CLI Override Surface

The mainline training CLI exposes the following commonly used overrides:

- `--config`
  - alternate config YAML
- `--epochs`, `--batch-size`, `--learning-rate`
  - basic optimization overrides
- `--grad-clip`
  - clip total gradient norm; `<= 0` disables clipping
- `--no-log-gradients`
  - disable gradient telemetry in `training_log.json`
- `--grad-log-every`
  - control gradient telemetry sampling frequency
- `--encoder-model`, `--hidden-dim`, `--seed`
  - quick architecture and reproducibility overrides
- `--quiet`
  - reduce CLI output
- `--lr-find-output-path`
  - save the LR-finder plot plus matching history JSON

The BNN entrypoint keeps the shared config/optimization overrides:

- `--config`
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--encoder-model`
- `--hidden-dim`
- `--seed`
- `--quiet`

### 6.4 Recommended Entrypoints

```bash
# Single-model training with config defaults
uv run python -m src.vif.train

# Quick smoke run
uv run python -m src.vif.train --epochs 5 --batch-size 8

# Encoder ablation
uv run python -m src.vif.train --encoder-model all-mpnet-base-v2

# Export LR-finder artifacts for review
uv run python -m src.vif.train --lr-find-output-path logs/exports/lr_find.png

# BNN baseline
uv run python -m src.vif.train_bnn

# Frontier experiment workflow
uv run python scripts/experiments/critic_training_v4_review.py
```
