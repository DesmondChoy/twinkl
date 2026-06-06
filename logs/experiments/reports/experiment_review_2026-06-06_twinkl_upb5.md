# Experiment Review - 2026-06-06 - `twinkl-upb5` Recall-Aware Checkpoint Rerun

## 1. Question

`twinkl-t2r0` replayed saved validation traces and found earlier epochs with
better validation `recall_-1`, but those epochs had not been serialized as
checkpoints. `twinkl-upb5` reran the same configs with candidate-checkpoint
retention so the alternate checkpoints could be scored on the fixed corrected
test split.

This rerun did not add data, change labels, reshuffle splits, tune encoders, or
change the optimizer. The only intentional code change was output-side
retention and evaluation of recall-aware candidate checkpoints.

## 2. What Changed

The training driver now accepts `candidate_checkpoint_policies` and can retain
additional candidate checkpoints beside the normal selected checkpoint:

- `recall_qwk_window_0.01`: maximize validation `recall_-1` among eligible
  epochs within `0.01` validation QWK of the best eligible validation QWK.
- `recall_qwk_window_0.02`: same policy with a `0.02` validation QWK window.

Each retained candidate is saved as a `.pt` checkpoint and evaluated on the
same validation and fixed test loaders as the selected checkpoint. Evaluation
RNG is reset before each selected/candidate MC-dropout pass so same-epoch
candidates compare cleanly instead of picking up dropout noise.

## 3. Rerun Scope

| Family | Runs | Source configs | Label regime | Notes |
|---|---|---|---|---|
| Consensus diagnostic | `run_051`-`run_053` | `twinkl_754_seed11/22/33.yaml` | consensus | Same branch as `run_048`-`run_050`; useful diagnostic, not a persisted-label promotion. |
| Weighted persisted | `run_054`-`run_056` | `twinkl_719_3_seed11/22/33.yaml` | persisted | Clean promotion regime for the existing corrected-split frontier comparison. |

All six reruns used split seed `2025`, model seeds `11/22/33`, the frozen
holdout manifest `config/experiments/vif/twinkl_681_5_holdout.yaml`, the same
explicit learning rate override `0.015522253574270487`, and the same
`nomic-ai/nomic-embed-text-v1.5` encoder with `truncate_dim=256`.

## 4. Artifacts

Run YAMLs:

- `logs/experiments/runs/run_051_BalancedSoftmax.yaml`
- `logs/experiments/runs/run_052_BalancedSoftmax.yaml`
- `logs/experiments/runs/run_053_BalancedSoftmax.yaml`
- `logs/experiments/runs/run_054_BalancedSoftmax.yaml`
- `logs/experiments/runs/run_055_BalancedSoftmax.yaml`
- `logs/experiments/runs/run_056_BalancedSoftmax.yaml`

Summary artifacts:

- `logs/experiments/artifacts/recall_checkpoint_retention_twinkl_upb5_20260606/checkpoint_comparison.csv`
- `logs/experiments/artifacts/recall_checkpoint_retention_twinkl_upb5_20260606/checkpoint_comparison.parquet`
- `logs/experiments/artifacts/recall_checkpoint_retention_twinkl_upb5_20260606/policy_family_summary.csv`
- `logs/experiments/artifacts/recall_checkpoint_retention_twinkl_upb5_20260606/policy_family_summary.parquet`

Each run artifact directory also contains:

- `candidate_recall_qwk_window_0_01_checkpoint.pt`
- `candidate_recall_qwk_window_0_02_checkpoint.pt`
- candidate validation/test output parquets
- `candidate_checkpoint_metrics.yaml`

Training logs:

- `logs/experiments/artifacts/twinkl_upb5_training_logs/`

## 5. Family Median Results

Values below are deterministic fixed-test metrics. Deltas are versus the normal
selected checkpoint inside the same run family.

| Family | Policy | Median QWK | Median `recall_-1` | Median MinR | Median hedge | Median cal | dQWK | d`recall_-1` | dMinR | dHedge | dCal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Consensus diagnostic | selected | 0.374 | 0.257 | 0.397 | 0.656 | 0.767 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Consensus diagnostic | `0.01` window | 0.374 | 0.257 | 0.397 | 0.656 | 0.767 | +0.000 | -0.000 | -0.000 | +0.000 | -0.000 |
| Consensus diagnostic | `0.02` window | 0.393 | 0.323 | 0.445 | 0.650 | 0.738 | +0.000 | +0.055 | +0.032 | -0.006 | -0.014 |
| Weighted persisted | selected | 0.345 | 0.376 | 0.448 | 0.597 | 0.713 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Weighted persisted | `0.01` window | 0.332 | 0.360 | 0.448 | 0.611 | 0.695 | -0.013 | -0.000 | +0.000 | -0.000 | -0.000 |
| Weighted persisted | `0.02` window | 0.329 | 0.350 | 0.448 | 0.636 | 0.739 | -0.021 | -0.006 | -0.026 | +0.014 | +0.008 |

Read this carefully: the consensus branch likes the `0.02` rule; the persisted
weighted branch does not.

## 6. Per-Run Recall-Aware Results

### Consensus Diagnostic Branch

| Run | Seed | Policy | Epoch | Test QWK | Test `recall_-1` | dQWK | d`recall_-1` | Hedo QWK | Sec QWK | Stim QWK |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_051` | 11 | selected | 27 | 0.365 | 0.256 | 0.000 | 0.000 | 0.051 | 0.264 | 0.423 |
| `run_051` | 11 | `0.02` | 14 | 0.393 | 0.311 | +0.028 | +0.055 | 0.137 | 0.352 | 0.376 |
| `run_052` | 22 | selected | 22 | 0.374 | 0.257 | 0.000 | 0.000 | 0.128 | 0.187 | 0.352 |
| `run_052` | 22 | `0.02` | 14 | 0.352 | 0.380 | -0.022 | +0.123 | 0.127 | 0.264 | 0.369 |
| `run_053` | 33 | selected | 14 | 0.403 | 0.323 | 0.000 | 0.000 | 0.148 | 0.231 | 0.321 |
| `run_053` | 33 | `0.02` | 14 | 0.403 | 0.323 | +0.000 | -0.000 | 0.148 | 0.231 | 0.321 |

The `0.02` rule is genuinely useful in the consensus-label branch. It improves
test `recall_-1` on two seeds, leaves the third unchanged, lowers median
hedging, and improves median QWK. Calibration drops modestly.

That still does not make it a frontier promotion because the label target is
different from the persisted-label board.

### Weighted Persisted Branch

| Run | Seed | Policy | Epoch | Test QWK | Test `recall_-1` | dQWK | d`recall_-1` | Hedo QWK | Sec QWK | Stim QWK |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_054` | 11 | selected | 27 | 0.345 | 0.281 | 0.000 | 0.000 | 0.129 | 0.258 | 0.256 |
| `run_054` | 11 | `0.01` | 21 | 0.332 | 0.360 | -0.013 | +0.079 | 0.126 | 0.261 | 0.213 |
| `run_054` | 11 | `0.02` | 32 | 0.310 | 0.275 | -0.034 | -0.006 | 0.038 | 0.201 | 0.194 |
| `run_055` | 22 | selected | 20 | 0.329 | 0.376 | 0.000 | 0.000 | 0.203 | 0.185 | 0.150 |
| `run_055` | 22 | `0.01` | 20 | 0.329 | 0.376 | +0.000 | -0.000 | 0.203 | 0.185 | 0.150 |
| `run_055` | 22 | `0.02` | 20 | 0.329 | 0.376 | +0.000 | -0.000 | 0.203 | 0.185 | 0.150 |
| `run_056` | 33 | selected | 26 | 0.367 | 0.391 | 0.000 | 0.000 | 0.230 | 0.222 | 0.140 |
| `run_056` | 33 | `0.01` | 30 | 0.346 | 0.350 | -0.021 | -0.041 | 0.192 | 0.202 | 0.149 |
| `run_056` | 33 | `0.02` | 30 | 0.346 | 0.350 | -0.021 | -0.041 | 0.192 | 0.202 | 0.149 |

The persisted branch is the one that matters for promotion against the current
frontier. It fails the case. Seed 11 has a useful `0.01` trade-off, but seed 33
goes the other way, and the family median does not improve recall.

## 7. Verdict

Do not promote recall-aware checkpoint selection as the default persisted-label
VIF checkpoint policy.

The reason is no longer missing checkpoints. We reran with the checkpoints
saved, scored them, and the clean persisted-label result is not strong enough:

- `0.01` window: median QWK drops from `0.345` to `0.332`, while median
  `recall_-1` is effectively flat to slightly worse.
- `0.02` window: median QWK drops to `0.329`, median `recall_-1` drops to
  `0.350`, minority recall drops, and hedging worsens.

The consensus-label branch is different. There, the `0.02` policy is promising:
median `recall_-1` rises from `0.257` to `0.323`, median MinR rises from
`0.397` to `0.445`, and median QWK rises from `0.374` to `0.393`. But consensus
labels are a diagnostic target, not the active persisted-label frontier target.

Recommendation:

1. Keep the active persisted-label frontier unchanged.
2. Keep candidate checkpoint retention in the driver. It is useful experiment
   hygiene and prevents another "interesting epoch but no checkpoint" problem.
3. Do not switch the default persisted-label selection policy to recall-aware
   based on this result.
4. If the project adopts consensus or soft vote-distribution labels as the next
   target, include the `0.02` recall-window candidate as a standard retained
   checkpoint and evaluate it directly.

## 8. Reproduction

Example command shape:

```bash
TWINKL_VIF_NOTEBOOK_CONFIG=config/experiments/vif/twinkl_754_seed11.yaml \
TWINKL_VIF_NOTEBOOK_OVERRIDES='{"experiment_group":"twinkl-upb5_recall_checkpoint_retention_consensus","candidate_checkpoint_policies":[{"name":"recall_qwk_window_0.01","type":"recall_qwk_window","qwk_window":0.01},{"name":"recall_qwk_window_0.02","type":"recall_qwk_window","qwk_window":0.02}]}' \
.venv/bin/python scripts/experiments/critic_training_v4_review.py
```

The same override was applied to:

- `config/experiments/vif/twinkl_754_seed11.yaml`
- `config/experiments/vif/twinkl_754_seed22.yaml`
- `config/experiments/vif/twinkl_754_seed33.yaml`
- `config/experiments/vif/twinkl_719_3_seed11.yaml`
- `config/experiments/vif/twinkl_719_3_seed22.yaml`
- `config/experiments/vif/twinkl_719_3_seed33.yaml`

Targeted tests:

```bash
.venv/bin/python -m pytest \
  tests/vif/test_eval_metrics.py::TestOrdinalSelectionHelpers \
  tests/vif/test_recall_checkpoint_replay.py \
  tests/vif/test_experiment_logger.py

.venv/bin/python -m py_compile \
  src/vif/eval.py \
  scripts/experiments/critic_training_v4_review.py
```
