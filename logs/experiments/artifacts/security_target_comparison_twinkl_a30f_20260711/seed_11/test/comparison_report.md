# twinkl-a30f Security target comparison

Split: `test`. Samples: 221. Changed Security targets: 86.

| model | target lens | QWK | Security QWK | Security recall -1 | hedging | calibration |
|---|---|---:|---:|---:|---:|---:|
| historical_model | historical_labels | 0.343 | 0.195 | 0.429 | 0.628 | 0.713 |
| historical_model | repaired_labels | 0.333 | 0.095 | 0.286 | 0.628 | 0.685 |
| repaired_model | historical_labels | 0.371 | 0.414 | 0.500 | 0.597 | 0.674 |
| repaired_model | repaired_labels | 0.363 | 0.339 | 0.500 | 0.597 | 0.688 |

Each model is scored against both label regimes. Values across target lenses are diagnostic and must not be treated as one leaderboard.
