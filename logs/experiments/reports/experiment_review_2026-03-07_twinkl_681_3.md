# Experiment Review — 2026-03-07 — twinkl-681.3 post-hoc boundary optimization

## Scope

Validation-only tuning on the corrected-split frontier `run_016-run_018` using existing selected-checkpoint artifacts only. No retraining was performed.

## Test Summary

| Run | Model | Selected policy | Test QWK | Test recall_-1 | Test MinR | Test neutral rate | Test calibration |
|-----|-------|-----------------|---------:|---------------:|----------:|------------------:|-----------------:|
| run_016 | CDWCE_a3 | logit_adjustment_tau_0.70 | 0.361 | 0.276 | 0.433 | 0.731 | 0.509 |
| run_016 | SoftOrdinal | logit_adjustment_tau_0.00 | 0.372 | 0.151 | 0.335 | 0.823 | 0.607 |
| run_016 | CORN | corn_per_dimension_margin | 0.312 | 0.148 | 0.331 | 0.813 | 0.821 |
| run_017 | CDWCE_a3 | logit_adjustment_tau_0.70 | 0.334 | 0.244 | 0.417 | 0.736 | 0.555 |
| run_017 | SoftOrdinal | logit_adjustment_tau_0.30 | 0.354 | 0.164 | 0.350 | 0.803 | 0.500 |
| run_017 | CORN | corn_per_dimension_margin | 0.336 | 0.161 | 0.335 | 0.825 | 0.818 |
| run_018 | CDWCE_a3 | logit_adjustment_tau_0.70 | 0.369 | 0.288 | 0.440 | 0.740 | 0.545 |
| run_018 | SoftOrdinal | logit_adjustment_tau_0.30 | 0.343 | 0.179 | 0.403 | 0.746 | 0.517 |
| run_018 | CORN | corn_per_dimension_margin | 0.355 | 0.182 | 0.351 | 0.809 | 0.815 |

## Median / IQR by Family

| Model | Median QWK | IQR QWK | Median recall_-1 | IQR recall_-1 | Median MinR | Median neutral rate | Median calibration |
|-------|-----------:|--------:|-----------------:|--------------:|------------:|--------------------:|-------------------:|
| CDWCE_a3 | 0.361 | 0.018 | 0.276 | 0.022 | 0.433 | 0.736 | 0.545 |
| SoftOrdinal | 0.354 | 0.015 | 0.164 | 0.014 | 0.350 | 0.803 | 0.517 |
| CORN | 0.336 | 0.021 | 0.161 | 0.017 | 0.335 | 0.813 | 0.818 |

## Policy-Level Takeaways

- Softmax logit adjustment median delta: recall_-1 0.095, QWK 0.006.
- CORN threshold tuning median delta: recall_-1 0.009, QWK 0.009.
- Recommended softmax base for `twinkl-681.4`: `CDWCE_a3`.

## Conclusion

`CDWCE_a3` is the strongest softmax-family handoff from `681.3` under the recall-first guarded selector. The comparison above also shows whether softmax logit adjustment or CORN boundary tuning extracted more validation-disciplined recall gains from the corrected-split frontier.
