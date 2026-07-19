# Recall-First VIF Critic Checkpoint Selection

**Beads issue:** `twinkl-6mrt`
**Date:** 2026-07-19
**Data role:** existing repaired-Security development traces and saved
validation outputs; no fresh final test was opened

## Decision

Mainline VIF Critic training uses `recall_first_qwk_guarded_v1`:

1. Reject a validation checkpoint when mean QWK is below `0.3712`, any
   per-value QWK is undefined, or calibration is negative.
2. Rank eligible checkpoints by higher `recall_-1`, higher QWK, higher
   calibration, lower hedging, lower validation loss, then earlier epoch.
3. If no checkpoint is eligible, retain the best finite-QWK checkpoint only for
   debugging; it is not eligible for promotion.

The QWK floor is configurable and recorded in checkpoint, selection-summary,
and Run metadata. The historical `qwk_then_recall_guarded` policy remains
available for reproduction.

## Why the QWK floor is `0.3712`

The current training-ready `window_size: 1` family is the repaired-Security
BalancedSoftmax family from `twinkl-a30f`:

| Run | Seed | Saved validation QWK | Saved validation `recall_-1` |
|---|---:|---:|---:|
| `run_058` | 11 | 0.3912 | 0.3740 |
| `run_060` | 22 | 0.3943 | 0.4984 |
| `run_062` | 33 | 0.3837 | 0.4356 |

The median selected validation QWK is `0.3912`. The approved tolerance is
`0.02`, so the fixed floor is `0.3912 - 0.02 = 0.3712`.

Pure recall-first replay would select early checkpoints with QWK as low as
`0.1556`. Their saved traces do not include Conflict precision, so the cause of
the recall increase cannot be resolved retrospectively. The QWK collapse alone
is enough to reject them. The floor keeps recall primary without accepting that
loss of ordinal structure.

No fixed Conflict precision floor is adopted. Conflict precision,
predicted-negative rate, calibration, and per-value results remain mandatory
companion reports rather than selection gates.

## Retrospective development-trace replay

The replay used each Run's existing `selection_trace.parquet`, filtered to the
new QWK floor, and applied the deterministic ranking above.

| Run | Recall-first epoch | Validation `recall_-1` | Validation QWK | Saved state available? |
|---|---:|---:|---:|---|
| `run_058` | 22 | 0.4793 | 0.3763 | No |
| `run_060` | 21 | 0.5224 | 0.3714 | No |
| `run_062` | 14 | 0.4356 | 0.3837 | Yes; this is the saved selection |

The traces can identify preferred epochs but cannot recreate unsaved model
states. Historical Runs and their original QWK-first selections therefore stay
unchanged.

## Comparable saved-checkpoint nomination

All three saved repaired-Security checkpoints clear the new floor. Re-ranking
only those available checkpoints nominates `run_060` for the offline VIF Critic
path because its stored selection `recall_-1` is highest.

| Run | Selection `recall_-1` | Selection QWK | Calibration | Hedging | Validation Conflict precision | Predicted-negative rate |
|---|---:|---:|---:|---:|---:|---:|
| `run_058` | 0.3740 | 0.3912 | 0.6908 | 0.5926 | 0.296 | 0.107 |
| **`run_060`** | **0.4984** | **0.3943** | 0.6514 | 0.5839 | 0.304 | 0.129 |
| `run_062` | 0.4356 | 0.3837 | 0.7103 | 0.5857 | 0.313 | 0.122 |

Conflict precision and predicted-negative rate were recomputed from the saved
validation-output files using the training metric's `mean_prediction < -0.5`
decision rule. They are companion measurements from the saved checkpoint
evaluation pass; MC Dropout means they need not exactly reproduce the earlier
selection pass.

The nominated checkpoint is:

`logs/experiments/artifacts/ordinal_v4_s2025_m22_20260711_162502/BalancedSoftmax/selected_checkpoint.pt`

SHA-256:
`ffa71dab2c92bb7938472032a0a11ac9efa74699cf533db71ea1a15647ad196c`

This nomination is for stored prediction, offline comparison, independent
review, and retraining only. It does not enter the Weekly Drift Reviewer prompt,
does not make a user-facing Drift decision, and is not deployment approval.

## Inputs and limitations

- Run records: `run_058_BalancedSoftmax.yaml`, `run_060_BalancedSoftmax.yaml`,
  and `run_062_BalancedSoftmax.yaml`.
- Selection traces and validation outputs: the paths recorded in those Run
  records under `artifacts`.
- Source target: `security_active_critic_state_v1`; the three Runs hold the
  encoder, architecture, split, optimizer settings, and repaired label regime
  fixed while changing model seed.
- No historical checkpoint or Run record was overwritten.
- No test output selected the policy or the nominated checkpoint.
- A future comparable retrain should use `recall_first_qwk_guarded_v1` from the
  start so its selected model state is persisted.
