# twinkl-a30f Security target comparison

Split: `val`. Samples: 217. Changed Security targets: 82.

| model | target lens | QWK | Security QWK | Security recall -1 | hedging | calibration |
|---|---|---:|---:|---:|---:|---:|
| historical_model | historical_labels | 0.386 | 0.564 | 0.609 | 0.602 | 0.687 |
| historical_model | repaired_labels | 0.371 | 0.408 | 0.400 | 0.602 | 0.641 |
| repaired_model | historical_labels | 0.381 | 0.383 | 0.565 | 0.583 | 0.684 |
| repaired_model | repaired_labels | 0.386 | 0.430 | 0.525 | 0.583 | 0.695 |

Each model is scored against both label regimes. Values across target lenses are diagnostic and must not be treated as one leaderboard.
