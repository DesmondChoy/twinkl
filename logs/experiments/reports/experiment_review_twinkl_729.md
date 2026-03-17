# Experiment Review — twinkl-729 run_020 vs run_036 BalancedSoftmax effective-prior post-hoc retargeting

## Scope

Validation-only tuning on the corrected-split frontier `run_020, run_036` using existing selected-checkpoint artifacts only. No retraining was performed.

Primary checkpoint target: incumbent `run_020` `BalancedSoftmax`. Best current tail-sensitive reference checkpoint: `run_036` from `twinkl-719.3`.

## Test Summary

| Run | Model | Selected policy | Test QWK | Test recall_-1 | Test MinR | Test hedging | Test neutral rate | Test calibration | OppV | AdjS |
|-----|-------|-----------------|---------:|---------------:|----------:|-------------:|------------------:|-----------------:|-----:|-----:|
| run_020 | BalancedSoftmax | logit_adjustment_tau_0.30 | 0.338 | 0.350 | 0.397 | 0.562 | 0.740 | 0.583 | 0.107 | 0.089 |
| run_036 | BalancedSoftmax | effective_prior_per_dimension_tau | 0.360 | 0.339 | 0.440 | 0.581 | 0.724 | 0.592 | 0.089 | 0.102 |

## Branch Comparison

| Run | Variant | Validation policy | Test QWK | Test recall_-1 | Test MinR | Test hedging | Test calibration | OppV | AdjS |
|-----|---------|-------------------|---------:|---------------:|----------:|-------------:|-----------------:|-----:|-----:|
| run_020 | Untouched baseline | logit_adjustment_tau_0.00 | 0.378 | 0.342 | 0.449 | 0.621 | 0.713 | 0.070 | 0.077 |
| run_020 | Best standard Menon branch | logit_adjustment_tau_0.30 | 0.338 | 0.350 | 0.397 | 0.562 | 0.583 | 0.107 | 0.089 |
| run_020 | Effective-prior + per-dimension tau | effective_prior_per_dimension_tau | 0.343 | 0.275 | 0.364 | 0.625 | 0.576 | 0.088 | 0.084 |
| run_036 | Untouched baseline | logit_adjustment_tau_0.00 | 0.381 | 0.387 | 0.492 | 0.599 | 0.726 | 0.068 | 0.084 |
| run_036 | Best standard Menon branch | logit_adjustment_tau_0.00 | 0.381 | 0.387 | 0.492 | 0.599 | 0.726 | 0.068 | 0.084 |
| run_036 | Effective-prior + per-dimension tau | effective_prior_per_dimension_tau | 0.360 | 0.339 | 0.440 | 0.581 | 0.592 | 0.089 | 0.102 |

## Median / IQR by Family

| Model | Median QWK | IQR QWK | Median recall_-1 | IQR recall_-1 | Median MinR | Median hedging | IQR hedging | Median neutral rate | Median calibration | Median OppV | IQR OppV | Median AdjS | IQR AdjS |
|-------|-----------:|--------:|-----------------:|--------------:|------------:|---------------:|------------:|--------------------:|-------------------:|------------:|---------:|------------:|---------:|
| BalancedSoftmax | 0.349 | 0.011 | 0.344 | 0.005 | 0.419 | 0.571 | 0.010 | 0.732 | 0.588 | 0.098 | 0.009 | 0.096 | 0.007 |

## Policy-Level Takeaways

- Softmax logit adjustment median delta: recall_-1 -0.020, QWK -0.030, hedging -0.039, OppV 0.029, AdjS 0.015.
- CORN threshold tuning median delta: recall_-1 nan, QWK nan, hedging nan, OppV nan, AdjS nan.
- Standard Menon median delta vs baseline: recall_-1 0.004, QWK -0.020, hedging -0.030, OppV 0.018, AdjS 0.006.
- Effective-prior + per-dimension tau median delta vs baseline: recall_-1 -0.057, QWK -0.028, hedging -0.007, OppV 0.019, AdjS 0.013.
- Recommended post-hoc BalancedSoftmax family anchor for `twinkl-729`: `BalancedSoftmax`.

## Conclusion

The effective-prior + per-dimension tau branch did not beat the standard Menon control on the guarded median test comparison, and neither branch produced a clean enough package to justify a frontier change. Treat this as evidence that the current post-hoc line is likely exhausted for the active frontier.
