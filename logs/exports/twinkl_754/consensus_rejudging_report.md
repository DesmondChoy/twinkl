# twinkl-754 Consensus Re-judging Report

## Scope Summary

- Prompt condition: `profile_only`
- Bundle mode: `full`
- Entries: `1651`
- Passes: `5`
- Personas: `204`
- Worker model: `gpt-5.4`
- Stability bootstrap: `2000` persona-cluster resamples, seed `42`

## 1. Judge Repeated-Call Self-Consistency

These kappas measure repeated-call consistency of the same judge workflow, not agreement among independent raters.

| Dimension | Fleiss kappa | Human baseline |
| --- | --- | --- |
| self_direction | 0.765 | N/A |
| stimulation | 0.809 | 0.580 |
| hedonism | 0.797 | 0.640 |
| achievement | 0.789 | N/A |
| power | 0.793 | N/A |
| security | 0.775 | 0.480 |
| conformity | 0.781 | N/A |
| tradition | 0.809 | N/A |
| benevolence | 0.779 | N/A |
| universalism | 0.890 | N/A |

## 2. Consensus vs Persisted

| Dimension | Consensus vs persisted Cohen kappa |
| --- | --- |
| self_direction | 0.714 |
| stimulation | 0.804 |
| hedonism | 0.776 |
| achievement | 0.750 |
| power | 0.753 |
| security | 0.775 |
| conformity | 0.785 |
| tradition | 0.778 |
| benevolence | 0.763 |
| universalism | 0.898 |
| aggregate | 0.778 |

Confusion counts:

| Dimension | Persisted | Consensus | Count |
| --- | --- | --- | --- |
| achievement | -1 | -1 | 39 |
| achievement | -1 | 0 | 38 |
| achievement | -1 | 1 | 8 |
| achievement | 0 | -1 | 10 |
| achievement | 0 | 0 | 1055 |
| achievement | 0 | 1 | 65 |
| achievement | 1 | -1 | 1 |
| achievement | 1 | 0 | 63 |
| achievement | 1 | 1 | 372 |
| benevolence | -1 | -1 | 75 |
| benevolence | -1 | 0 | 42 |
| benevolence | -1 | 1 | 6 |
| benevolence | 0 | -1 | 6 |
| benevolence | 0 | 0 | 1003 |
| benevolence | 0 | 1 | 64 |
| benevolence | 1 | -1 | 3 |
| benevolence | 1 | 0 | 68 |
| benevolence | 1 | 1 | 384 |
| conformity | -1 | -1 | 93 |
| conformity | -1 | 0 | 36 |
| conformity | 0 | -1 | 15 |
| conformity | 0 | 0 | 1156 |
| conformity | 0 | 1 | 18 |
| conformity | 1 | -1 | 7 |
| conformity | 1 | 0 | 70 |
| conformity | 1 | 1 | 256 |
| hedonism | -1 | -1 | 105 |
| hedonism | -1 | 0 | 35 |
| hedonism | -1 | 1 | 1 |
| hedonism | 0 | -1 | 34 |
| hedonism | 0 | 0 | 1277 |
| hedonism | 0 | 1 | 26 |
| hedonism | 1 | -1 | 2 |
| hedonism | 1 | 0 | 23 |
| hedonism | 1 | 1 | 148 |
| power | -1 | -1 | 80 |
| power | -1 | 0 | 48 |
| power | -1 | 1 | 18 |
| power | 0 | -1 | 4 |
| power | 0 | 0 | 1314 |
| power | 0 | 1 | 20 |
| power | 1 | -1 | 2 |
| power | 1 | 0 | 31 |
| power | 1 | 1 | 134 |
| security | -1 | -1 | 117 |
| security | -1 | 0 | 26 |
| security | -1 | 1 | 8 |
| security | 0 | -1 | 16 |
| security | 0 | 0 | 1146 |
| security | 0 | 1 | 50 |
| security | 1 | -1 | 9 |
| security | 1 | 0 | 47 |
| security | 1 | 1 | 232 |
| self_direction | -1 | -1 | 118 |
| self_direction | -1 | 0 | 83 |
| self_direction | -1 | 1 | 15 |
| self_direction | 0 | -1 | 14 |
| self_direction | 0 | 0 | 965 |
| self_direction | 0 | 1 | 40 |
| self_direction | 1 | 0 | 87 |
| self_direction | 1 | 1 | 329 |
| stimulation | -1 | -1 | 36 |
| stimulation | -1 | 0 | 19 |
| stimulation | -1 | 1 | 5 |
| stimulation | 0 | -1 | 7 |
| stimulation | 0 | 0 | 1447 |
| stimulation | 0 | 1 | 10 |
| stimulation | 1 | 0 | 22 |
| stimulation | 1 | 1 | 105 |
| tradition | -1 | -1 | 34 |
| tradition | -1 | 0 | 19 |
| tradition | -1 | 1 | 5 |
| tradition | 0 | -1 | 2 |
| tradition | 0 | 0 | 1315 |
| tradition | 0 | 1 | 20 |
| tradition | 1 | -1 | 1 |
| tradition | 1 | 0 | 61 |
| tradition | 1 | 1 | 194 |
| universalism | -1 | -1 | 45 |
| universalism | -1 | 0 | 6 |
| universalism | -1 | 1 | 5 |
| universalism | 0 | -1 | 1 |
| universalism | 0 | 0 | 1427 |
| universalism | 0 | 1 | 8 |
| universalism | 1 | 0 | 18 |
| universalism | 1 | 1 | 141 |

## 3. Human-Overlap Benchmark (Advisory)

These kappas are a limited non-expert human-overlap benchmark. They are advisory only and do not act as the hard retrain gate.

- Annotator files loaded: `3`
- Union coverage: `150` unique annotated entries across `24` personas
- Strict 3-way overlap used for comparison: `115` entries across `19` personas
- Singly annotated entries excluded from majority aggregation: `35`
- Full-corpus entries outside the overlap excluded from the human benchmark: `1536`

| Dimension | Consensus vs human overlap (advisory) | Persisted vs human overlap (advisory) |
| --- | --- | --- |
| self_direction | 0.483 | 0.699 |
| stimulation | 0.714 | 0.763 |
| hedonism | 0.554 | 0.638 |
| achievement | 0.696 | 0.731 |
| power | 0.662 | 0.661 |
| security | 0.501 | 0.495 |
| conformity | 0.643 | 0.654 |
| tradition | 0.784 | 0.806 |
| benevolence | 0.737 | 0.748 |
| universalism | 0.883 | 0.924 |
| aggregate | 0.674 | 0.723 |

## 4. Confidence Tier Distribution

| Dimension | Tier | Entries | Non-neutral entries |
| --- | --- | --- | --- |
| achievement | bare_majority | 143 | 65 |
| achievement | strong | 167 | 73 |
| achievement | unanimous | 1341 | 357 |
| benevolence | bare_majority | 170 | 87 |
| benevolence | no_majority | 1 | 0 |
| benevolence | strong | 171 | 65 |
| benevolence | unanimous | 1309 | 386 |
| conformity | bare_majority | 118 | 66 |
| conformity | no_majority | 2 | 0 |
| conformity | strong | 165 | 66 |
| conformity | unanimous | 1366 | 257 |
| hedonism | bare_majority | 100 | 46 |
| hedonism | strong | 132 | 49 |
| hedonism | unanimous | 1419 | 221 |
| power | bare_majority | 100 | 56 |
| power | strong | 90 | 32 |
| power | unanimous | 1461 | 170 |
| security | bare_majority | 128 | 75 |
| security | no_majority | 1 | 0 |
| security | strong | 197 | 81 |
| security | unanimous | 1325 | 276 |
| self_direction | bare_majority | 171 | 82 |
| self_direction | no_majority | 2 | 0 |
| self_direction | strong | 202 | 64 |
| self_direction | unanimous | 1276 | 370 |
| stimulation | bare_majority | 46 | 29 |
| stimulation | strong | 75 | 23 |
| stimulation | unanimous | 1530 | 111 |
| tradition | bare_majority | 71 | 40 |
| tradition | no_majority | 1 | 0 |
| tradition | strong | 105 | 41 |
| tradition | unanimous | 1474 | 175 |
| universalism | bare_majority | 40 | 26 |
| universalism | strong | 38 | 17 |
| universalism | unanimous | 1573 | 157 |

## 5. Pass Diagnostics

| Pass | Attempt | Worker model | Raw hash | Score hash | Rationale coverage | Completed |
| --- | --- | --- | --- | --- | --- | --- |
| pass_1 | 3 | gpt-5.4 | 1053dde9b730 | 6addd04601da | 1.000 | 2026-03-20T11:43:20.871496+00:00 |
| pass_2 | 1 | gpt-5.4 | 09c747ca998f | 4233aa814640 | 1.000 | 2026-03-20T12:03:35.101592+00:00 |
| pass_3 | 3 | gpt-5.4 | 36539617f090 | 8aa482cd5159 | 1.000 | 2026-03-20T12:28:35.540445+00:00 |
| pass_4 | 1 | gpt-5.4 | 469fa3426d63 | 5c99e7091b8c | 1.000 | 2026-03-21T00:12:46.335614+00:00 |
| pass_5 | 4 | gpt-5.4 | 0dcfd26d647a | 9542cfff71d8 | 1.000 | 2026-03-26T15:01:38.367259+00:00 |

Pairwise similarity:

| Pass pair | Raw hash match | Score hash match | Identical entry vectors | Differing entry vectors |
| --- | --- | --- | --- | --- |
| pass_1 vs pass_2 | False | False | 834 | 817 |
| pass_1 vs pass_3 | False | False | 868 | 783 |
| pass_1 vs pass_4 | False | False | 891 | 760 |
| pass_1 vs pass_5 | False | False | 817 | 834 |
| pass_2 vs pass_3 | False | False | 833 | 818 |
| pass_2 vs pass_4 | False | False | 846 | 805 |
| pass_2 vs pass_5 | False | False | 773 | 878 |
| pass_3 vs pass_4 | False | False | 865 | 786 |
| pass_3 vs pass_5 | False | False | 823 | 828 |
| pass_4 vs pass_5 | False | False | 832 | 819 |

## 6. Full-Corpus Stability Gate

This is the hard retrain gate. It uses full-corpus stability only: the upper 95% CI of `low_confidence_non_neutral_ratio` must stay below `0.5` for `security`, `hedonism`, and `stimulation`.

The human-overlap benchmark above remains advisory and limited-sample; it is not used in this go/no-go decision.

- Full-corpus stability gate passed: `True`
- Retrain readiness summary: `Eligible for retrain comparison under full-corpus stability criteria.`

| Dimension | Non-neutral labels | Low-confidence ratio (95% CI) | Mean vote entropy (95% CI) | Passes |
| --- | --- | --- | --- | --- |
| security | 432 | 0.174 [0.131, 0.218] | 0.310 [0.255, 0.363] | True |
| hedonism | 316 | 0.146 [0.104, 0.193] | 0.256 [0.206, 0.317] | True |
| stimulation | 163 | 0.178 [0.112, 0.255] | 0.280 [0.207, 0.364] | True |

## 7. Per-Dimension Stability

Dimensions are ranked by `mean_vote_entropy_non_neutral` point estimate from highest to lowest. The 95% CIs below show uncertainty around the estimate; they are not the difficulty ranking itself.

| Rank | Dimension | Non-neutral labels | Mean vote entropy (95% CI) | Non-unanimous rate (95% CI) | Polarity-flip rate (95% CI) | Low-confidence ratio (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | power | 258 | 0.312 [0.249, 0.382] | 0.341 [0.274, 0.413] | 0.116 [0.074, 0.161] | 0.217 [0.157, 0.285] |
| 2 | security | 432 | 0.310 [0.255, 0.363] | 0.361 [0.300, 0.421] | 0.081 [0.052, 0.113] | 0.174 [0.131, 0.218] |
| 3 | conformity | 389 | 0.298 [0.247, 0.353] | 0.339 [0.281, 0.401] | 0.028 [0.011, 0.048] | 0.170 [0.131, 0.212] |
| 4 | stimulation | 163 | 0.280 [0.207, 0.364] | 0.319 [0.240, 0.407] | 0.037 [0.006, 0.077] | 0.178 [0.112, 0.255] |
| 5 | tradition | 256 | 0.272 [0.212, 0.343] | 0.316 [0.246, 0.401] | 0.051 [0.026, 0.082] | 0.156 [0.110, 0.212] |
| 6 | hedonism | 316 | 0.256 [0.206, 0.317] | 0.301 [0.241, 0.370] | 0.028 [0.011, 0.048] | 0.146 [0.104, 0.193] |
| 7 | self_direction | 516 | 0.251 [0.208, 0.298] | 0.283 [0.236, 0.334] | 0.037 [0.021, 0.055] | 0.159 [0.123, 0.199] |
| 8 | benevolence | 538 | 0.249 [0.210, 0.292] | 0.283 [0.242, 0.330] | 0.033 [0.019, 0.050] | 0.162 [0.124, 0.203] |
| 9 | achievement | 495 | 0.243 [0.201, 0.288] | 0.279 [0.232, 0.328] | 0.030 [0.016, 0.046] | 0.131 [0.100, 0.169] |
| 10 | universalism | 200 | 0.192 [0.120, 0.278] | 0.215 [0.135, 0.310] | 0.050 [0.011, 0.094] | 0.130 [0.069, 0.208] |

Hard-dimension callouts:

- `security`: low-confidence ratio `0.174 [0.131, 0.218]`; mean vote entropy `0.310 [0.255, 0.363]`; gate pass=`True`
- `hedonism`: low-confidence ratio `0.146 [0.104, 0.193]`; mean vote entropy `0.256 [0.206, 0.317]`; gate pass=`True`
- `stimulation`: low-confidence ratio `0.178 [0.112, 0.255]`; mean vote entropy `0.280 [0.207, 0.364]`; gate pass=`True`

## 8. Hard-Dimension Deep Dive

### security

Full-corpus stability first: `432` non-neutral labels, mean vote entropy `0.310 [0.255, 0.363]`, non-unanimous rate `0.361 [0.300, 0.421]`, polarity-flip rate `0.081 [0.052, 0.113]`, and low-confidence ratio `0.174 [0.131, 0.218]` (gate pass=`True`).

Secondary advisory benchmark: persisted-vs-consensus kappa `0.775`; consensus-vs-human overlap kappa `0.501` on the strict `115`-entry advisory subset.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 117 |
| -1 | 0 | 26 |
| -1 | 1 | 8 |
| 0 | -1 | 16 |
| 0 | 0 | 1146 |
| 0 | 1 | 50 |
| 1 | -1 | 9 |
| 1 | 0 | 47 |
| 1 | 1 | 232 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 128 | 75 |
| no_majority | 1 | 0 |
| strong | 197 | 81 |
| unanimous | 1325 | 276 |

### hedonism

Full-corpus stability first: `316` non-neutral labels, mean vote entropy `0.256 [0.206, 0.317]`, non-unanimous rate `0.301 [0.241, 0.370]`, polarity-flip rate `0.028 [0.011, 0.048]`, and low-confidence ratio `0.146 [0.104, 0.193]` (gate pass=`True`).

Secondary advisory benchmark: persisted-vs-consensus kappa `0.776`; consensus-vs-human overlap kappa `0.554` on the strict `115`-entry advisory subset.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 105 |
| -1 | 0 | 35 |
| -1 | 1 | 1 |
| 0 | -1 | 34 |
| 0 | 0 | 1277 |
| 0 | 1 | 26 |
| 1 | -1 | 2 |
| 1 | 0 | 23 |
| 1 | 1 | 148 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 100 | 46 |
| strong | 132 | 49 |
| unanimous | 1419 | 221 |

### stimulation

Full-corpus stability first: `163` non-neutral labels, mean vote entropy `0.280 [0.207, 0.364]`, non-unanimous rate `0.319 [0.240, 0.407]`, polarity-flip rate `0.037 [0.006, 0.077]`, and low-confidence ratio `0.178 [0.112, 0.255]` (gate pass=`True`).

Secondary advisory benchmark: persisted-vs-consensus kappa `0.804`; consensus-vs-human overlap kappa `0.714` on the strict `115`-entry advisory subset.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 36 |
| -1 | 0 | 19 |
| -1 | 1 | 5 |
| 0 | -1 | 7 |
| 0 | 0 | 1447 |
| 0 | 1 | 10 |
| 1 | 0 | 22 |
| 1 | 1 | 105 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 46 | 29 |
| strong | 75 | 23 |
| unanimous | 1530 | 111 |


## 9. Rationale Source Summary

- Entries with a perfect 10/10 rationale-source match: `1622`
- Entries using fallback rationale selection: `29`
- Maximum label mismatches on a chosen rationale source: `1`

| Source pass | Mismatch count | Entries |
| --- | --- | --- |
| 1 | 0 | 1143 |
| 1 | 1 | 24 |
| 2 | 0 | 281 |
| 2 | 1 | 4 |
| 3 | 0 | 128 |
| 3 | 1 | 1 |
| 4 | 0 | 55 |
| 5 | 0 | 15 |

## 10. Label Migration Summary

| Dimension | Persisted | Consensus | Changed entries |
| --- | --- | --- | --- |
| achievement | -1 | 0 | 38 |
| achievement | -1 | 1 | 8 |
| achievement | 0 | -1 | 10 |
| achievement | 0 | 1 | 65 |
| achievement | 1 | -1 | 1 |
| achievement | 1 | 0 | 63 |
| benevolence | -1 | 0 | 42 |
| benevolence | -1 | 1 | 6 |
| benevolence | 0 | -1 | 6 |
| benevolence | 0 | 1 | 64 |
| benevolence | 1 | -1 | 3 |
| benevolence | 1 | 0 | 68 |
| conformity | -1 | 0 | 36 |
| conformity | 0 | -1 | 15 |
| conformity | 0 | 1 | 18 |
| conformity | 1 | -1 | 7 |
| conformity | 1 | 0 | 70 |
| hedonism | -1 | 0 | 35 |
| hedonism | -1 | 1 | 1 |
| hedonism | 0 | -1 | 34 |
| hedonism | 0 | 1 | 26 |
| hedonism | 1 | -1 | 2 |
| hedonism | 1 | 0 | 23 |
| power | -1 | 0 | 48 |
| power | -1 | 1 | 18 |
| power | 0 | -1 | 4 |
| power | 0 | 1 | 20 |
| power | 1 | -1 | 2 |
| power | 1 | 0 | 31 |
| security | -1 | 0 | 26 |
| security | -1 | 1 | 8 |
| security | 0 | -1 | 16 |
| security | 0 | 1 | 50 |
| security | 1 | -1 | 9 |
| security | 1 | 0 | 47 |
| self_direction | -1 | 0 | 83 |
| self_direction | -1 | 1 | 15 |
| self_direction | 0 | -1 | 14 |
| self_direction | 0 | 1 | 40 |
| self_direction | 1 | 0 | 87 |
| stimulation | -1 | 0 | 19 |
| stimulation | -1 | 1 | 5 |
| stimulation | 0 | -1 | 7 |
| stimulation | 0 | 1 | 10 |
| stimulation | 1 | 0 | 22 |
| tradition | -1 | 0 | 19 |
| tradition | -1 | 1 | 5 |
| tradition | 0 | -1 | 2 |
| tradition | 0 | 1 | 20 |
| tradition | 1 | -1 | 1 |
| tradition | 1 | 0 | 61 |
| universalism | -1 | 0 | 6 |
| universalism | -1 | 1 | 5 |
| universalism | 0 | -1 | 1 |
| universalism | 0 | 1 | 8 |
| universalism | 1 | 0 | 18 |

## 11. Retrain Comparison

The full-corpus stability gate passed, so the retrain comparison was run with
the run-specific configs `config/experiments/vif/twinkl_754_seed11.yaml`,
`config/experiments/vif/twinkl_754_seed22.yaml`, and
`config/experiments/vif/twinkl_754_seed33.yaml`. Each config overrides
`data.labels_path` to `logs/judge_labels/consensus_labels.parquet` without
editing the repo default `config/vif.yaml`. The three seeded retrains produced
`run_048`-`run_050` `BalancedSoftmax` on the fixed holdout with seeds
`11 / 22 / 33`.

The required primary comparison is against incumbent `run_019`-`run_021`.
Because that incumbent family predates the later `1,651`-row dataset refresh,
its training split is `1,022` rows rather than the `1,213` rows used by both
the refreshed persisted-label family `run_025`-`run_027` and the consensus
family `run_048`-`run_050`. Treat the `run_019`-`run_021` deltas below as the
requested baseline comparison, not a pure label-only ablation. The refreshed
persisted-label family is included as advisory same-size context. One further
warning matters: this retrain path also changed the evaluation labels on the
same fixed holdout, so the aggregate metric deltas below are diagnostic rather
than a strict leaderboard replacement for the persisted-label board.

| Family | Train rows | Labels | qwk_mean | recall_-1 | minority recall | hedging | calibration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `run_019`-`run_021` baseline | 1022 | persisted | 0.362 | 0.313 | 0.448 | 0.621 | 0.713 |
| `run_025`-`run_027` refreshed persisted (advisory) | 1213 | persisted | 0.346 | 0.328 | 0.442 | 0.598 | 0.693 |
| `run_048`-`run_050` consensus | 1213 | consensus | 0.372 | 0.270 | 0.408 | 0.655 | 0.770 |

Seed-aligned comparison versus the requested incumbent baseline:

| Seed | Baseline run | Consensus run | Δ qwk_mean | Δ recall_-1 | Δ minority recall | Δ hedging | Δ calibration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | `run_019` | `run_048` | +0.010 | -0.017 | -0.010 | +0.014 | +0.051 |
| 22 | `run_020` | `run_049` | -0.009 | -0.072 | -0.042 | +0.067 | +0.057 |
| 33 | `run_021` | `run_050` | +0.034 | +0.025 | +0.032 | +0.058 | +0.070 |

Hard-dimension QWK outcome versus the incumbent family:

| Dimension | `run_019`-`run_021` median | `run_048`-`run_050` median | Δ median | `run_019`-`run_021` ceiling | `run_048`-`run_050` ceiling | Δ ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| security | 0.297 | 0.247 | -0.051 | 0.338 | 0.278 | -0.059 |
| hedonism | 0.247 | 0.104 | -0.143 | 0.262 | 0.108 | -0.153 |
| stimulation | 0.161 | 0.340 | +0.179 | 0.303 | 0.406 | +0.104 |

Advisory same-size read versus `run_025`-`run_027`: consensus labels recover
aggregate `qwk_mean` (`0.372` vs `0.346`) and calibration (`0.770` vs `0.693`)
and strongly lift `stimulation qwk` (`0.340` vs `0.186` median) while modestly
helping `security qwk` (`0.247` vs `0.199` median). But they materially worsen
`hedonism qwk` (`0.104` vs `0.256` median), `recall_-1` (`0.270` vs `0.328`),
minority recall (`0.408` vs `0.442`), and hedging (`0.655` vs `0.598`).

## 12. Recommendation

The stability-first retrain gate passed and the retrain comparison is now
complete, but the consensus-label family does not deliver a clean
hard-dimension ceiling lift. Relative to the requested incumbent baseline
`run_019`-`run_021`, it improves aggregate `qwk_mean` slightly (`0.372` vs
`0.362`) and materially improves `stimulation`, yet it weakens `security` and
especially `hedonism`, while also giving back `recall_-1`, minority recall, and
decisiveness.

Treat this as a mixed or negative diagnostic result rather than a promotion
case. The evidence is good enough to say the full-corpus stability gate was not
too loose, because the retrain could be run safely and the resulting branch can
be audited. But it is **not** a clean frontier replacement for the persisted
label regime, and it is not strong enough to claim that hard consensus labels
improve the hard-dimension ceiling overall. If this line is revisited,
`hedonism` remains the clearest blocker, and a same-code persisted-label sibling
rerun should be established first.
