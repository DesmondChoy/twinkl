# twinkl-a30f Security target comparison

Split: `test`. Samples: 221. Changed Security targets: 86.

| model | target lens | QWK | Security QWK | Security recall -1 | hedging | calibration |
|---|---|---:|---:|---:|---:|---:|
| historical_model | historical_labels | 0.320 | 0.214 | 0.643 | 0.587 | 0.676 |
| historical_model | repaired_labels | 0.314 | 0.156 | 0.464 | 0.587 | 0.658 |
| repaired_model | historical_labels | 0.336 | 0.332 | 0.500 | 0.586 | 0.681 |
| repaired_model | repaired_labels | 0.335 | 0.328 | 0.500 | 0.586 | 0.693 |

Each model is scored against both label regimes. Values across target lenses are diagnostic and must not be treated as one leaderboard.
