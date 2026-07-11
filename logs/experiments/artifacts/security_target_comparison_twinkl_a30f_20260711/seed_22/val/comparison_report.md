# twinkl-a30f Security target comparison

Split: `val`. Samples: 217. Changed Security targets: 82.

| model | target lens | QWK | Security QWK | Security recall -1 | hedging | calibration |
|---|---|---:|---:|---:|---:|---:|
| historical_model | historical_labels | 0.397 | 0.366 | 0.652 | 0.591 | 0.635 |
| historical_model | repaired_labels | 0.394 | 0.335 | 0.525 | 0.591 | 0.605 |
| repaired_model | historical_labels | 0.387 | 0.330 | 0.478 | 0.577 | 0.646 |
| repaired_model | repaired_labels | 0.395 | 0.405 | 0.500 | 0.577 | 0.657 |

Each model is scored against both label regimes. Values across target lenses are diagnostic and must not be treated as one leaderboard.
