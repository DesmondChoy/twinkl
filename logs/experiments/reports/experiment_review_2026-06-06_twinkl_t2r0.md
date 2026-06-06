# Experiment Review - 2026-06-06 - `twinkl-t2r0` Recall-Aware Checkpoint Replay

## 1. Question

`twinkl-t2r0` asked whether the VIF training path is throwing away useful
`recall_-1` operating points by selecting checkpoints primarily on validation
QWK. The product reason is straightforward: Twinkl is supposed to catch real
value conflicts. Missing true `-1` cases is worse than winning a tidy aggregate
score by hedging toward neutral.

This review replayed checkpoint-selection traces only. It did not add data,
change labels, reshuffle splits, or tune encoders.

## 2. Artifacts

Replay script:

- `scripts/experiments/replay_recall_aware_checkpoint_selection.py`

Replay output:

- `logs/experiments/artifacts/recall_checkpoint_replay_twinkl_t2r0_20260606/selection_replay.csv`
- `logs/experiments/artifacts/recall_checkpoint_replay_twinkl_t2r0_20260606/selection_replay.parquet`
- `logs/experiments/artifacts/recall_checkpoint_replay_twinkl_t2r0_20260606/inventory.yaml`

Run scope:

- Incumbent persisted-label frontier: `run_019`-`run_021`
- Weighted persisted-label reference: `run_034`-`run_036`
- Consensus-label diagnostic branch: `run_048`-`run_050`

Artifact inventory:

| Family | Runs | Selection trace | Dimension trace | Selected checkpoint/test outputs | Alternate epoch checkpoints |
|---|---|---:|---:|---:|---:|
| Incumbent persisted `BalancedSoftmax` | `run_019`-`run_021` | no | no | yes | no |
| Weighted persisted `BalancedSoftmax` | `run_034`-`run_036` | yes | yes | yes | no |
| Consensus diagnostic `BalancedSoftmax` | `run_048`-`run_050` | yes | yes | yes | no |

The uncomfortable fact: the traces are enough to replay policy choice, but not
enough to test alternate epochs. The training driver saved the selected model
state only. Earlier attractive epochs are metrics rows, not recoverable
checkpoints.

## 3. Policies

| Policy | Rule |
|---|---|
| Current | Select the best eligible validation `qwk_mean`, then break ties by `recall_-1`, calibration, lower hedging, and lower validation loss. |
| Recall A | Maximize validation `recall_-1` among epochs within `0.01` validation QWK of the best validation QWK. Tie-break lower hedging, then higher calibration, then earlier epoch. |
| Recall B | Same as Recall A, but allow a `0.02` validation QWK window. |

Epochs in the tables below are 1-based for readability. The parquet trace keeps
zero-based epochs.

## 4. Validation Replay

### Weighted Persisted Branch

| Run | Policy | Epoch | Val QWK | Val `recall_-1` | dQWK | d`recall_-1` | Hedging | Cal | Hedo QWK | Sec QWK | Stim QWK | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `run_034` | Current | 27 | 0.401 | 0.462 | 0.000 | 0.000 | 0.599 | 0.720 | 0.371 | 0.460 | 0.554 | selected exists |
| `run_034` | Recall A | 21 | 0.400 | 0.491 | -0.002 | +0.030 | 0.595 | 0.647 | 0.330 | 0.454 | 0.596 | missing |
| `run_034` | Recall B | 32 | 0.385 | 0.536 | -0.016 | +0.074 | 0.606 | 0.706 | 0.484 | 0.387 | 0.537 | missing |
| `run_035` | Current | 20 | 0.391 | 0.555 | 0.000 | 0.000 | 0.550 | 0.622 | 0.356 | 0.435 | 0.567 | selected exists |
| `run_035` | Recall A | 20 | 0.391 | 0.555 | 0.000 | 0.000 | 0.550 | 0.622 | 0.356 | 0.435 | 0.567 | selected exists |
| `run_035` | Recall B | 20 | 0.391 | 0.555 | 0.000 | 0.000 | 0.550 | 0.622 | 0.356 | 0.435 | 0.567 | selected exists |
| `run_036` | Current | 26 | 0.399 | 0.446 | 0.000 | 0.000 | 0.600 | 0.731 | 0.459 | 0.507 | 0.504 | selected exists |
| `run_036` | Recall A | 30 | 0.394 | 0.494 | -0.005 | +0.047 | 0.616 | 0.730 | 0.458 | 0.495 | 0.533 | missing |
| `run_036` | Recall B | 30 | 0.394 | 0.494 | -0.005 | +0.047 | 0.616 | 0.730 | 0.458 | 0.495 | 0.533 | missing |

Read: the weighted branch already behaves more tail-sensitively than the
incumbent, but even there the replay finds validation recall on the table for
`run_034` and `run_036`. The hard-dimension validation profile is mixed rather
than obviously better.

### Consensus Diagnostic Branch

| Run | Policy | Epoch | Val QWK | Val `recall_-1` | dQWK | d`recall_-1` | Hedging | Cal | Hedo QWK | Sec QWK | Stim QWK | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `run_048` | Current | 27 | 0.441 | 0.453 | 0.000 | 0.000 | 0.656 | 0.765 | 0.397 | 0.456 | 0.684 | selected exists |
| `run_048` | Recall A | 27 | 0.441 | 0.453 | 0.000 | 0.000 | 0.656 | 0.765 | 0.397 | 0.456 | 0.684 | selected exists |
| `run_048` | Recall B | 14 | 0.427 | 0.513 | -0.014 | +0.060 | 0.627 | 0.745 | 0.353 | 0.391 | 0.726 | missing |
| `run_049` | Current | 22 | 0.424 | 0.427 | 0.000 | 0.000 | 0.669 | 0.780 | 0.404 | 0.402 | 0.760 | selected exists |
| `run_049` | Recall A | 22 | 0.424 | 0.427 | 0.000 | 0.000 | 0.669 | 0.780 | 0.404 | 0.402 | 0.760 | selected exists |
| `run_049` | Recall B | 14 | 0.413 | 0.534 | -0.011 | +0.107 | 0.682 | 0.780 | 0.443 | 0.394 | 0.742 | missing |
| `run_050` | Current | 14 | 0.441 | 0.464 | 0.000 | 0.000 | 0.615 | 0.711 | 0.447 | 0.410 | 0.643 | selected exists |
| `run_050` | Recall A | 14 | 0.441 | 0.464 | 0.000 | 0.000 | 0.615 | 0.711 | 0.447 | 0.410 | 0.643 | selected exists |
| `run_050` | Recall B | 14 | 0.441 | 0.464 | 0.000 | 0.000 | 0.615 | 0.711 | 0.447 | 0.410 | 0.643 | selected exists |

Read: the suspicion from `twinkl-754.6` is real, but only under the `0.02`
window. The `0.01` rule is too tight for the two interesting consensus seeds.
`run_048` and `run_049` both have epoch 14 candidates that materially lift
validation `recall_-1`; `run_050` already selected that operating point.

## 5. Fixed-Test Evidence

Only the already selected checkpoints can be scored on the fixed corrected split
from current artifacts. That table is below for context, but it is not evidence
for the alternate epochs.

| Run | Regime | Test QWK | Test `recall_-1` | Test MinR | Hedging | Cal | Hedo QWK | Sec QWK | Stim QWK |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_034` | persisted weighted | 0.342 | 0.299 | 0.412 | 0.627 | 0.740 | 0.067 | 0.222 | 0.228 |
| `run_035` | persisted weighted | 0.321 | 0.378 | 0.449 | 0.578 | 0.686 | 0.129 | 0.121 | 0.152 |
| `run_036` | persisted weighted | 0.381 | 0.387 | 0.492 | 0.599 | 0.726 | 0.217 | 0.258 | 0.133 |
| `run_048` | consensus diagnostic | 0.372 | 0.259 | 0.389 | 0.655 | 0.778 | 0.104 | 0.278 | 0.406 |
| `run_049` | consensus diagnostic | 0.369 | 0.270 | 0.408 | 0.688 | 0.770 | 0.059 | 0.247 | 0.340 |
| `run_050` | consensus diagnostic | 0.393 | 0.338 | 0.480 | 0.623 | 0.724 | 0.109 | 0.204 | 0.332 |

The incumbent `run_019`-`run_021` selected checkpoints still have test outputs,
but they do not have selection traces, so this issue cannot replay alternate
incumbent epochs from current artifacts.

## 6. Verdict

Do not promote recall-aware checkpoint selection from this evidence alone.

The validation case is promising, especially for the consensus branch under a
`0.02` QWK window:

- `run_048`: validation `recall_-1` +0.060 for QWK -0.014.
- `run_049`: validation `recall_-1` +0.107 for QWK -0.011.
- `run_050`: current selection already matches the recall-aware choice.

But the alternate checkpoints are not serialized. Without fixed-test scoring,
promoting the policy would be hand-wavy, and this project has had enough
hand-wavy experiment lore. Validation traces are a lead, not a win.

Recommendation:

1. Keep the current frontier unchanged.
2. If this line continues, use the `0.02` QWK-window policy, not `0.01`.
3. Run exact same-config reruns only after adding candidate-checkpoint retention
   to the training driver, or otherwise guaranteeing that the selected
   recall-aware epoch is actually test-evaluable.
4. Do not mix label regimes in the promotion decision. Consensus-label runs
   remain diagnostic until a persisted-label sibling is evaluated.

## 7. Reproduction

```bash
.venv/bin/python scripts/experiments/replay_recall_aware_checkpoint_selection.py \
  --output-dir logs/experiments/artifacts/recall_checkpoint_replay_twinkl_t2r0_20260606
```

Targeted test:

```bash
.venv/bin/python -m pytest tests/vif/test_recall_checkpoint_replay.py
```
