# VIF – Reward Modeling & Training Strategy

This document describes the current training story for the Value Identity
Function (VIF): how LLM-Judge labels become VIF Critic targets, which VIF Critic
families are live, and how frontier evaluation is managed.

The forward capstone policy is defined in
[VIF Capstone Scope and Evaluation Decision](05_capstone_scope_decision.md):
the VIF Critic is primarily a Conflict screener, per-Journal-Entry
`recall_-1` is the primary development metric, and QWK is an ordinal-health
diagnostic.

---

## 1. LLM-Judge and VIF Critic Training Stack

### 1.1 Why Reward Modeling Exists

For real users, Twinkl observes Journal Entry text and profile context, not a
ground-truth alignment score. The project therefore uses this training setup:

1. **Generator** creates synthetic personas and Journal Entry sequences.
2. **LLM-Judge** labels each Journal Entry against the Schwartz dimensions.
3. **VIF Critic** learns those labels as a fast supervised model.

This lets the live scoring path stay fast and uncertainty-aware without calling
an LLM on every runtime inference.

### 1.2 LLM-Judge Output Contract

The LLM-Judge emits a categorical label for each value:

- `-1`: Conflict
- `0`: neutral / not enough evidence
- `+1`: aligned

These labels are the current VIF Critic targets.

---

## 2. Training Target: What Is Current vs Exploratory

### 2.1 Current Mainline Target

The live stack uses **Option A: immediate alignment**.

$$
\vec{V}_\theta(s_{u,t}) \approx \hat{\vec{a}}_{u,t}
$$

That means the VIF Critic predicts the LLM-Judge's labels for the current
Journal Entry. Its outputs support offline review and retraining rather than a
learned discounted-return target.
The approved user-facing Drift path uses decisions from the fixed
`gpt-5.6-luna` reasoning-effort-`low` Weekly Drift Reviewer.

### 2.2 Exploratory Alternatives

The older design space still matters conceptually, but it is not the active
training path:

- **Option B**: short-horizon forecast targets
- **Option C**: discounted return targets

These remain future research directions, not the current implementation.

### 2.3 Scalar Aggregation

When a single summary score is needed, the VIF Critic output can be aggregated
with profile weights:

$$
V^{\text{scalar}}_{u,t} = w_u^\top \vec{V}_\theta(s_{u,t})
$$

This scalar is for offline summaries and monitoring. The approved Drift
Detector consumes Weekly Drift Reviewer decisions instead. The VIF Critic is
trained to preserve the vector of trade-offs.

---

## 3. VIF Critic Families

### 3.1 Shared Input Setup

All VIF Critic variants consume the same state vector described in
[System Architecture, State, and Runtime Flow](02_system_architecture.md):

- frozen sentence embeddings
- optional raw recent-history window or config-gated compact prior summary
- time-gap features
- 10-dim value-profile weights

The encoder is frozen in the current POC for simplicity, reproducibility, and
data efficiency.

### 3.2 Active VIF Critic Families

The mainline VIF Critic uses a shared MLP backbone with multiple comparison
families:

- **Ordinal heads**: CORAL, CORN, EMD, CDW-CE, SoftOrdinal
- **Long-tail baselines**: BalancedSoftmax, LDAM-DRW
- **Experimental reformulation**: two-stage BalancedSoftmax
- **Baselines retained for comparison**: legacy MSE MLP and Bayesian neural net

The experiment board treats the
BalancedSoftmax `run_019`-`run_021` family as the historical corrected-split
reference.
The two-stage, consensus-label, recall-aware checkpoint-retention, soft-label,
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

- wrangled Journal Entries from `logs/wrangled`
- consolidated LLM-Judge labels from `logs/judge_labels/judge_labels.parquet`

Each labeled Journal Entry becomes one `(state_vector, target_vector)` pair.

### 4.2 Split Policy

The current evaluation regime is persona-level and holdout-aware:

- train/val/test splits are by persona, not by Journal Entry
- validation/test are sign-stratified at the persona level
- optional fixed holdout manifests are used for augmentation and audit rounds

This is the only fair basis for comparing current frontier runs.

### 4.3 Metrics

The training stack tracks more than loss. Under the adopted `twinkl-752`
policy, the metric roles are:

- primary development metric: macro per-dimension `recall_-1`;
- mandatory companion reports: `-1` precision, precision-recall behavior,
  predicted-negative rate, per-dimension results, and seed spread;
- ordinal-health diagnostics: QWK, MAE, accuracy, `+1` recall, minority recall,
  calibration, and circumplex behavior;
- raw probability/logit exports when needed

No fixed precision floor is active yet. Recall-first development can identify
promising VIF Critic checkpoints for offline review and retraining, but it does
not grant user-facing Drift authority.

Implementation caveat: `src/vif/eval.py` still selects mainline checkpoints
QWK-first. Historical runs and the current board therefore reflect their
original policy. Recall-first selection needs a tested implementation before a
future training run is decision evidence.

### 4.4 Current Caveat: Reachability Audit

The training workflow is operational, but the completed `twinkl-747`
reachability audit changed how the historical board should be interpreted. The
full-corpus `twinkl-a30f` review has now produced a separate, training-ready
Security target under the exact active `window_size: 1` state contract.

The historical audit established the target-contract risk, but its three prompt
setups did not exactly match the active `window_size: 1` state. The 14-case
Security labels formerly derived from its legacy `student_visible` setup have
been retired. They are not a retraining source, evaluation lens, or
repaired-target result. The replacement labels were written only after a receipt-bound
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
did not replace the persisted-label frontier cleanly. The historical
corrected-split reference remains `run_019`-`run_021`.

The recall-aware reruns (`run_051`-`run_056`) persist alternate
checkpoints and their validation/test outputs. The wider `0.02` QWK window helps
the consensus diagnostic regime but does not improve the persisted-label
frontier. Checkpoint retention is therefore reproducibility and analysis
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

### 4.5 LLM Alignment Context Baseline

`scripts/experiments/llm_critic_baseline.py` measures the frozen-holdout ceiling
under three context contracts:

- `student_visible`: current Journal Entry, including its displayed nudge and
  response when present, plus the normalized ten-dimensional value profile
- `human_context`: student-visible input plus earlier Journal Entries from the
  same persona
- `full_judge_context`: human context plus persona biography and demographics;
  upper-bound diagnostic only

Future Journal Entries, target labels, rationales, and generation metadata are
excluded from every experiment setup.

This script's `student_visible` name refers to its own session-plus-profile
contract. It must not be confused with the differently scoped legacy
`twinkl-747` experiment setup of the same name.

On the 221-row test split:

| Scoring setup | QWK | `recall_-1` | Minority recall | Hedging |
|---|---:|---:|---:|---:|
| `gpt-5.4-mini`, `student_visible` | 0.434 | 0.188 | 0.428 | 0.789 |
| `gpt-5.4-mini`, `human_context` | 0.450 | 0.302 | 0.534 | 0.707 |
| `run_020` BalancedSoftmax MLP | 0.378 | 0.342 | 0.449 | 0.621 |

History improves the LLM's Conflict recall and broad minority-class
performance. The MLP still retains higher `recall_-1`, lower hedging, local
execution, and a fixed cost profile. The retired consensus-derived Drift
benchmark exposed a target-validity mismatch; it does not select among these
architectures and must not be rerun or used for deployment approval.
[`twinkl-v8pb`](../evals/drift_v1_student_visible_target.md) completed a
student-visible full-runtime-text review and correctly withheld its former
final-test score. That population and the expanded 106-trajectory union are
now development-only. No evaluated Journal Entry predictor or Drift Detector
has deployment approval without a fresh final test. The earlier AI audit is
diagnostic evidence, not human ground truth.

---

## 5. Implementation Reference

The VIF training stack lives in `src/vif/`.

| Module | Description |
|--------|-------------|
| `src/vif/encoders.py` | Sentence-encoder wrapper and encoder creation |
| `src/vif/state_encoder.py` | State-vector construction |
| `src/vif/critic.py` | Legacy regression-style VIF Critic MLP |
| `src/vif/critic_ordinal.py` | Active ordinal and long-tail head families |
| `src/vif/critic_bnn.py` | Bayesian neural baseline |
| `src/vif/dataset.py` | Data loading, joins, and persona-level splits |
| `src/vif/drift_target.py` | Student-visible review bundles, reconciliation, and target materialization |
| `src/vif/security_target.py` | Fail-closed exact-state Security target validation and diagnostic materialization |
| `scripts/experiments/prepare_a30f_security_target_audit.py` | Receipt-bound exact-state Security review bundle |
| `scripts/experiments/build_a30f_security_target.py` | Diagnostic Security target materialization after exact-state review |
| `scripts/experiments/run_a30f_security_target_reviews.py` | Full-corpus repeated exact-state review runner with receipts and resume support |
| `scripts/experiments/materialize_a30f_full_security_target.py` | Training-ready full-corpus Security target materialization |
| `scripts/experiments/evaluate_a30f_security_comparison.py` | Historical/repaired model × label-lens comparison |
| `scripts/experiments/build_v8pb_student_visible_target.py` | Development-set or final-test-set review bundle generation |
| `scripts/experiments/materialize_v8pb_student_visible_target.py` | Reviewed student-visible target materialization |
| `src/vif/eval.py` | Evaluation metrics and uncertainty-aware evaluation |
| `src/vif/posthoc.py` | Validation-only post-hoc boundary tuning |
| `src/vif/experiment_logger.py` | Persisted run YAMLs and experiment index support |
| `src/vif/train.py` | General single-model training entrypoint |
| `src/vif/train_bnn.py` | BNN training entrypoint |
| `scripts/experiments/critic_training_v4_review.py` | Canonical frontier review driver |
| `scripts/experiments/llm_critic_baseline.py` | LLM context estimate/run/score/report workflow |
| `scripts/experiments/replay_recall_aware_checkpoint_selection.py` | Checkpoint-only replay of alternate checkpoint-selection policies |
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
- immediate non-finite loss termination with preserved training logs

The frontier driver also accepts optional `candidate_checkpoint_policies` in
its YAML/JSON overrides:

```yaml
candidate_checkpoint_policies:
  - name: recall_qwk_window_0.01
    type: recall_qwk_window
    qwk_window: 0.01
  - name: recall_qwk_window_0.02
    type: recall_qwk_window
    qwk_window: 0.02
```

Each historical policy selected the strongest `recall_-1` checkpoint within the
configured QWK window, then persisted its checkpoint, validation/test outputs,
selection summary, and compact metric comparison. These checkpoints supplemented
the QWK-first mainline checkpoint. They remain reproducibility evidence, not the
implementation of the adopted recall-first policy.

### 6.3 CLI Overrides

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

# Export the LR-finder plot and history JSON for review
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
