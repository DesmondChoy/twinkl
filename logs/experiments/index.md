# VIF Experiment Index

| run | model | encoder | ws | hd | do | loss | params | ratio | MAE | Acc | QWK | Spear | Cal | MinR | file |
|-----|-------|---------|---:|---:|---:|------|-------:|------:|----:|----:|----:|------:|----:|-----:|------|
| 001 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 372756 | 585.2 | 0.232 | 0.782 | 0.398 | 0.459 | 0.644 | 0.298 | runs/run_001_CORAL.yaml |
| 001 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 372756 | 585.2 | 0.236 | 0.782 | 0.384 | 0.452 | 0.633 | 0.306 | runs/run_001_CORN.yaml |
| 001 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 375326 | 589.2 | 0.243 | 0.773 | 0.395 | 0.459 | 0.648 | 0.340 | runs/run_001_EMD.yaml |
| 001 | MSE | MiniLM-384d | 3 | 256 | 0.2 | weighted_mse_s5.0 | 370186 | 581.1 | 0.450 | 0.641 | 0.338 | 0.379 | -0.218 | 0.428 | runs/run_001_MSE.yaml |
| 001 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 375326 | 589.2 | 0.248 | 0.777 | 0.417 | 0.455 | 0.724 | 0.372 | runs/run_001_SoftOrdinal.yaml |
| 002 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10708 | 16.8 | 0.263 | 0.770 | 0.335 | 0.349 | 0.734 | 0.234 | runs/run_002_CORAL.yaml |
| 002 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10708 | 16.8 | 0.260 | 0.766 | 0.355 | 0.371 | 0.737 | 0.204 | runs/run_002_CORN.yaml |
| 002 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 11038 | 17.3 | 0.270 | 0.764 | 0.365 | 0.365 | 0.772 | 0.309 | runs/run_002_EMD.yaml |
| 002 | MSE | nomic-256d | 1 | 32 | 0.3 | weighted_mse_s5.0 | 10378 | 16.3 | 0.418 | 0.683 | 0.348 | 0.378 | -0.073 | 0.390 | runs/run_002_MSE.yaml |
| 002 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 11038 | 17.3 | 0.267 | 0.780 | 0.385 | 0.356 | 0.774 | 0.310 | runs/run_002_SoftOrdinal.yaml |
| 003 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 370196 | 581.2 | 0.241 | 0.776 | 0.369 | 0.394 | 0.640 | 0.265 | runs/run_003_CORAL.yaml |
| 003 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 370196 | 581.2 | 0.249 | 0.764 | 0.326 | 0.395 | 0.648 | 0.260 | runs/run_003_CORN.yaml |
| 003 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 372766 | 585.2 | 0.240 | 0.782 | 0.410 | 0.416 | 0.697 | 0.332 | runs/run_003_EMD.yaml |
| 003 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 372766 | 585.2 | 0.253 | 0.772 | 0.383 | 0.396 | 0.743 | 0.333 | runs/run_003_SoftOrdinal.yaml |
| 004 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10388 | 16.3 | 0.273 | 0.768 | 0.331 | 0.346 | 0.755 | 0.269 | runs/run_004_CORAL.yaml |
| 004 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10388 | 16.3 | 0.274 | 0.760 | 0.291 | 0.293 | 0.734 | 0.217 | runs/run_004_CORN.yaml |
| 004 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 10718 | 16.8 | 0.266 | 0.780 | 0.391 | 0.361 | 0.766 | 0.343 | runs/run_004_EMD.yaml |
| 004 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 10718 | 16.8 | 0.269 | 0.779 | 0.385 | 0.351 | 0.786 | 0.332 | runs/run_004_SoftOrdinal.yaml |
| 005 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 370196 | 386.4 | 0.215 | 0.798 | 0.282 | 0.411 | 0.629 | 0.186 | runs/run_005_CORAL.yaml |
| 005 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 370196 | 386.4 | 0.216 | 0.797 | 0.254 | 0.365 | 0.640 | 0.181 | runs/run_005_CORN.yaml |
| 005 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 372766 | 389.1 | 0.226 | 0.791 | 0.272 | 0.312 | 0.651 | 0.217 | runs/run_005_EMD.yaml |
| 005 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 372766 | 389.1 | 0.215 | 0.798 | 0.304 | 0.368 | 0.650 | 0.165 | runs/run_005_SoftOrdinal.yaml |
| 006 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10388 | 10.8 | 0.233 | 0.793 | 0.278 | 0.354 | 0.768 | 0.166 | runs/run_006_CORAL.yaml |
| 006 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10388 | 10.8 | 0.236 | 0.791 | 0.280 | 0.350 | 0.777 | 0.183 | runs/run_006_CORN.yaml |
| 006 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 10718 | 11.2 | 0.227 | 0.803 | 0.324 | 0.334 | 0.764 | 0.225 | runs/run_006_EMD.yaml |
| 006 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 10718 | 11.2 | 0.223 | 0.811 | 0.358 | 0.354 | 0.775 | 0.235 | runs/run_006_SoftOrdinal.yaml |
