# twinkl-a30f Security target comparison

Split: `test`. Samples: 221. Changed Security targets: 86.

| model | target lens | QWK | Security QWK | Security recall -1 | hedging | calibration |
|---|---|---:|---:|---:|---:|---:|
| historical_model | historical_labels | 0.353 | 0.205 | 0.429 | 0.592 | 0.698 |
| historical_model | repaired_labels | 0.353 | 0.201 | 0.321 | 0.592 | 0.674 |
| repaired_model | historical_labels | 0.373 | 0.372 | 0.500 | 0.587 | 0.691 |
| repaired_model | repaired_labels | 0.363 | 0.277 | 0.536 | 0.587 | 0.703 |

Each model is scored against both label regimes. Values across target lenses are diagnostic and must not be treated as one leaderboard.
