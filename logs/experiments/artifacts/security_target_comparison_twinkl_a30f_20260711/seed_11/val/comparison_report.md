# twinkl-a30f Security target comparison

Split: `val`. Samples: 217. Changed Security targets: 82.

| model | target lens | QWK | Security QWK | Security recall -1 | hedging | calibration |
|---|---|---:|---:|---:|---:|---:|
| historical_model | historical_labels | 0.413 | 0.452 | 0.652 | 0.618 | 0.698 |
| historical_model | repaired_labels | 0.398 | 0.303 | 0.425 | 0.618 | 0.653 |
| repaired_model | historical_labels | 0.376 | 0.360 | 0.609 | 0.590 | 0.677 |
| repaired_model | repaired_labels | 0.385 | 0.448 | 0.425 | 0.590 | 0.701 |

Each model is scored against both label regimes. Values across target lenses are diagnostic and must not be treated as one leaderboard.
