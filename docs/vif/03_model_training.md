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
- optional raw recent-history window or config-gated compact prior summary
- time-gap features
- 10-dim value-profile weights

The encoder is frozen in the current POC for simplicity, reproducibility, and
data efficiency.

### 3.2 Active Student Families

The mainline student uses a shared MLP backbone with multiple comparison
families:

- **Ordinal heads**: CORAL, CORN, EMD, CDW-CE, SoftOrdinal
- **Long-tail baselines**: BalancedSoftmax, LDAM-DRW
- **Experimental reformulation**: two-stage BalancedSoftmax
- **Baselines retained for comparison**: legacy MSE MLP and Bayesian neural net

The experiment board treats the
BalancedSoftmax `run_019`-`run_021` family as the default frontier reference.
The two-stage, consensus-label, recall-aware candidate-retention, soft-label,
and compact-history branches remain diagnostic challengers rather than the
mainline default. Compact-history `run_069` stayed under its 5,000-weight
increment but regressed on QWK, minority recall, Security, hedging, and
overfitting versus seed-matched repaired-target baseline `run_058`. See
`logs/experiments/index.md` for the live ranking.

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
reachability audit changed how the historical board should be interpreted. The
full-corpus `twinkl-a30f` review has now produced a separate, training-ready
Security target under the exact active `window_size: 1` state contract.

The historical audit established the target-contract risk, but its three prompt
arms did not exactly match the active `window_size: 1` state. The 14-case
Security artifact formerly derived from its legacy `student_visible` arm has
been retired. It is not a retraining source, evaluation lens, or repaired-target
result. The replacement artifact was written only after a receipt-bound
`active_critic_state_v1` review; the selected frozen-test subset remains
diagnostic-only. The completed full-corpus review changed 678 of
1,651 Security labels. Paired BalancedSoftmax runs `run_057`-`run_062` improved
median test Security QWK from `0.156` to `0.328` under the repaired lens and
from `0.205` to `0.372` under the historical lens. Use the repaired regime for
future comparable training, but do not merge its scores into the historical
leaderboard. See the [Security target contract](security_target_contract.md)
and [experiment review](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_a30f_security_target.md).

The consensus-label diagnostic branch (`run_048`-`run_050`) reinforces
that framing. It improved within-regime QWK and calibration on the
consensus-relabeled holdout, but it changed labels on the frozen test split and
did not replace the persisted-label frontier cleanly. The active corrected-split
default remains `run_019`-`run_021`.

The recall-aware reruns (`run_051`-`run_056`) persist alternate candidate
checkpoints and their validation/test outputs. The wider `0.02` QWK window helps
the consensus diagnostic regime but does not improve the persisted-label
frontier. Candidate retention is therefore reproducibility and analysis
hygiene, not the default checkpoint selector.

The hybrid soft vote-distribution experiment (`run_063`-`run_068`) preserves
the five `twinkl-754` vote fractions for nine dimensions and uses the repaired
active-state Security reviews from `twinkl-a30f`. Against a matching hard
hybrid control, soft BalancedSoftmax lowers median hedging and improves
minority recall and Hedonism QWK, but it slightly lowers `recall_-1` and
materially weakens Security and Stimulation. Its prior-adjusted training-space
NLL improves slightly, while the raw probabilities exported for runtime have
worse NLL and Brier scores. Keep the soft path config-gated and diagnostic; do
not make it the default target regime. See the
[experiment review](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_j0ck_soft_vote_labels.md).

### 4.5 LLM Critic Context Baseline

`scripts/experiments/llm_critic_baseline.py` measures the frozen-holdout ceiling
under three context contracts:

- `student_visible`: current journal session plus the normalized ten-dimensional
  value profile
- `human_context`: student-visible input plus earlier entries from the same
  persona
- `full_judge_context`: human context plus persona biography and demographics;
  upper-bound diagnostic only

Future entries, target labels, rationales, and generation metadata are excluded
from every arm.

This script's `student_visible` name refers to its own session-plus-profile
contract. It must not be confused with the differently scoped legacy
`twinkl-747` condition of the same name.

On the 221-row test split:

| Critic | QWK | `recall_-1` | Minority recall | Hedging |
|---|---:|---:|---:|---:|
| `gpt-5.4-mini`, `student_visible` | 0.434 | 0.188 | 0.428 | 0.789 |
| `gpt-5.4-mini`, `human_context` | 0.450 | 0.302 | 0.534 | 0.707 |
| `run_020` BalancedSoftmax MLP | 0.378 | 0.342 | 0.449 | 0.621 |

History improves the LLM's misalignment recall and broad minority-class
performance. The MLP still retains higher `recall_-1`, lower hedging, local
execution, and a fixed cost profile. The retired consensus-derived drift
benchmark exposed a target-validity mismatch; it does not select among these
architectures and must not be rerun or used for a promotion decision.
[`twinkl-v8pb`](../evals/drift_v1_student_visible_target.md) completed a
student-visible full-runtime-text review before an LLM, MLP, or cascade
decision. Its weak development recall and unresolved locked promotion case mean
no scorer can be promoted. The earlier AI audit is diagnostic evidence, not
human ground truth.

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
| `src/vif/drift_target.py` | Student-visible review packets, reconciliation, and target materialization |
| `src/vif/security_target.py` | Fail-closed exact-state Security target validation and diagnostic materialization |
| `scripts/experiments/prepare_a30f_security_target_audit.py` | Receipt-bound active-Critic-state Security review bundle |
| `scripts/experiments/build_a30f_security_target.py` | Diagnostic Security target materialization after exact-state review |
| `scripts/experiments/run_a30f_security_target_reviews.py` | Full-corpus repeated exact-state review runner with receipts and resume support |
| `scripts/experiments/materialize_a30f_full_security_target.py` | Training-ready full-corpus Security target materialization |
| `scripts/experiments/evaluate_a30f_security_comparison.py` | Historical/repaired model × label-lens comparison |
| `scripts/experiments/build_v8pb_student_visible_target.py` | Development or locked-promotion review packet generation |
| `scripts/experiments/materialize_v8pb_student_visible_target.py` | Reviewed student-visible target materialization |
| `src/vif/eval.py` | Evaluation metrics and uncertainty-aware evaluation |
| `src/vif/posthoc.py` | Validation-only post-hoc boundary tuning |
| `src/vif/experiment_logger.py` | Persisted run YAMLs and experiment index support |
| `src/vif/train.py` | General single-model training entrypoint |
| `src/vif/train_bnn.py` | BNN training entrypoint |
| `scripts/experiments/critic_training_v4_review.py` | Canonical frontier review driver |
| `scripts/experiments/llm_critic_baseline.py` | LLM context-arm estimate/run/score/report workflow |
| `scripts/experiments/replay_recall_aware_checkpoint_selection.py` | Artifact-only replay of candidate checkpoint policies |
| `scripts/experiments/no_new_data_vif_policy_search.py` | Validation-selected no-new-data ensemble/routing diagnostic |

---

## 6. Operational Notes

### 6.1 Current Runtime Choices

The current config defaults matter:

- encoder family is configured in `config/vif.yaml`
- the live default state currently uses `window_size: 1`
- MC Dropout uses 50 samples by default

These are implementation choices, not permanent design constraints.

### 6.2 Training Instrumentation

The general training entrypoint includes:

- default LR-finder pass before training
- gradient clipping and gradient telemetry
- immediate non-finite loss termination with preserved artifacts

The frontier driver also accepts optional candidate-checkpoint policies in its
YAML/JSON override surface:

```yaml
candidate_checkpoint_policies:
  - name: recall_qwk_window_0.01
    type: recall_qwk_window
    qwk_window: 0.01
  - name: recall_qwk_window_0.02
    type: recall_qwk_window
    qwk_window: 0.02
```

Each policy selects the strongest `recall_-1` checkpoint within the configured
QWK window, then persists its checkpoint, validation/test outputs, selection
summary, and compact metric comparison. These candidates supplement the
mainline selected checkpoint; they do not change the default promotion policy.

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

Both training entrypoints use the checkpoint directory from the selected config
and write `best_model.pt` plus `training_log.json`. Use an alternate config with
a separate `output.checkpoint_dir` for BNN runs that must not overwrite an MLP
run; neither CLI exposes a direct `--output-dir` flag.

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

The frontier driver has no `argparse` interface; passing `--help` starts the
experiment. Its supported overrides are environment variables:

- `TWINKL_VIF_NOTEBOOK_CONFIG`: YAML override path, resolved from the repo root
- `TWINKL_VIF_NOTEBOOK_OVERRIDES`: JSON object deep-merged after the YAML file

### 6.5 LLM Baseline Commands

```bash
# Estimate token use and cost without API calls
uv run python scripts/experiments/llm_critic_baseline.py estimate \
  --split test \
  --context-arms student_visible human_context

# Write dry-run records; still no API calls
uv run python scripts/experiments/llm_critic_baseline.py run \
  --limit 10 \
  --context-arms student_visible human_context

# Execute API calls; requires OPENAI_API_KEY in the environment or root .env
uv run python scripts/experiments/llm_critic_baseline.py run \
  --execute \
  --context-arms student_visible human_context \
  --reasoning-efforts none

# Score result JSONL and build a comparison report
uv run python scripts/experiments/llm_critic_baseline.py score \
  logs/experiments/artifacts/llm_critic_baseline/<run>/<results>.jsonl
uv run python scripts/experiments/llm_critic_baseline.py report \
  logs/experiments/artifacts/llm_critic_baseline/<run>/*.metrics.json
```

`estimate` and `run` accept `--labels-path`, `--wrangled-dir`,
`--holdout-manifest`, `--split`, `--limit`, `--seed`, `--shots`, `--models`,
and `--context-arms`. `run` also accepts `--reasoning-efforts`, `--output-dir`,
`--timeout`, `--max-attempts`, `--max-output-tokens`, and `--execute`. Without
`--execute`, `run` only writes dry-run records.

### 6.6 Experiment Utilities

| Purpose | Command | Important options |
|---|---|---|
| Replay recall-aware selection without retraining | `uv run python scripts/experiments/replay_recall_aware_checkpoint_selection.py` | repeatable `--run-file`, `--repo-root`, `--output-dir` |
| Search persisted no-new-data policies | `uv run python scripts/experiments/no_new_data_vif_policy_search.py` | `--weight-step`, `--temperatures`, `--output-dir` |
| Run frontier uncertainty review | `uv run python scripts/experiments/frontier_uncertainty_review.py` | `--config` |
| Run validation-only boundary tuning | `uv run python -m src.vif.posthoc` | `--config` |
