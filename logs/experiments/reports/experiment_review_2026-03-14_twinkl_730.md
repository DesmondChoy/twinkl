# Experiment Review — 2026-03-15 — `twinkl-730` frontier uncertainty

This review quantifies evaluation uncertainty for the active corrected-split
BalancedSoftmax frontier using saved test-output artifacts only. The core
question is whether the apparent family-level differences driving recent
promotion decisions are distinguishable from holdout noise once the fixed
27-persona frozen test set is treated as clustered data.

## Method

- Families reviewed: Current default BalancedSoftmax, BalancedSoftmax + dimweight, BalancedSoftmax + circreg + recall floor, BalancedSoftmax + targeted batch, BalancedSoftmax + hedonism/security lift.
- Bootstrap: 1000 persona-cluster resamples with 95% BCa confidence intervals.
- Cluster definition: resample entire persona trajectories so all within-persona time steps stay together.
- Hard-dimension significance: 1000 stratified persona-cluster permutations, preserving trajectory length while breaking prediction/target pairing.
- Hard dimensions: hedonism, security, stimulation.

## Findings

1. The weighted reference branch still has the strongest tail package, but its QWK change versus the incumbent remains `likely_noise` at -0.020 [-0.062, 0.018]. That means the point-estimate drop should not be treated as a certain regression on its own.
2. The same weighted branch shows a `distinguishable_gain` `recall_-1` delta of 0.065 [0.021, 0.128] and a `likely_noise` minority-recall delta of 0.001 [-0.037, 0.036]. Tail-sensitive gains are real only when those delta intervals stay cleanly above zero.
3. Hard-dimension significance is uneven. `stimulation` is above chance in 1/5 reviewed families under the stratified permutation test, so it remains the weakest discriminator. `hedonism` and `security` should only influence promotion calls when both their own QWK is above chance and the challenger-vs-incumbent family delta is not just bootstrap noise.

## Checkpoint Intervals

| Run | Family | QWK | recall_-1 | MinR |
| --- | --- | --- | --- | --- |
| run_019 | Current default BalancedSoftmax | 0.362 [0.299, 0.444] | 0.277 [0.176, 0.388] | 0.399 [0.336, 0.492] |
| run_020 | Current default BalancedSoftmax | 0.378 [0.313, 0.473] | 0.342 [0.263, 0.445] | 0.449 [0.387, 0.534] |
| run_021 | Current default BalancedSoftmax | 0.358 [0.294, 0.438] | 0.313 [0.220, 0.444] | 0.448 [0.389, 0.530] |
| run_034 | BalancedSoftmax + dimweight | 0.342 [0.262, 0.443] | 0.298 [0.190, 0.424] | 0.412 [0.341, 0.510] |
| run_035 | BalancedSoftmax + dimweight | 0.321 [0.246, 0.415] | 0.378 [0.262, 0.502] | 0.449 [0.379, 0.532] |
| run_036 | BalancedSoftmax + dimweight | 0.381 [0.311, 0.471] | 0.387 [0.273, 0.502] | 0.492 [0.421, 0.576] |
| run_031 | BalancedSoftmax + circreg + recall floor | 0.353 [0.259, 0.456] | 0.267 [0.157, 0.379] | 0.409 [0.348, 0.491] |
| run_032 | BalancedSoftmax + circreg + recall floor | 0.366 [0.281, 0.465] | 0.355 [0.238, 0.477] | 0.435 [0.367, 0.535] |
| run_033 | BalancedSoftmax + circreg + recall floor | 0.372 [0.301, 0.464] | 0.260 [0.187, 0.374] | 0.409 [0.358, 0.493] |
| run_022 | BalancedSoftmax + targeted batch | 0.349 [0.272, 0.445] | 0.366 [0.261, 0.502] | 0.434 [0.360, 0.528] |
| run_023 | BalancedSoftmax + targeted batch | 0.372 [0.289, 0.466] | 0.308 [0.191, 0.426] | 0.450 [0.386, 0.545] |
| run_024 | BalancedSoftmax + targeted batch | 0.339 [0.250, 0.432] | 0.342 [0.215, 0.476] | 0.433 [0.356, 0.532] |
| run_025 | BalancedSoftmax + hedonism/security lift | 0.346 [0.277, 0.441] | 0.305 [0.202, 0.454] | 0.411 [0.346, 0.497] |
| run_026 | BalancedSoftmax + hedonism/security lift | 0.334 [0.254, 0.424] | 0.383 [0.263, 0.510] | 0.457 [0.389, 0.545] |
| run_027 | BalancedSoftmax + hedonism/security lift | 0.351 [0.288, 0.438] | 0.328 [0.221, 0.438] | 0.442 [0.374, 0.525] |

## Family Intervals

| Family | Family-median QWK | Family-median recall_-1 | Family-median MinR |
| --- | --- | --- | --- |
| Current default BalancedSoftmax | 0.362 [0.300, 0.444] | 0.313 [0.226, 0.421] | 0.448 [0.393, 0.541] |
| BalancedSoftmax + dimweight | 0.342 [0.265, 0.442] | 0.378 [0.266, 0.498] | 0.449 [0.379, 0.532] |
| BalancedSoftmax + circreg + recall floor | 0.366 [0.289, 0.464] | 0.267 [0.172, 0.368] | 0.409 [0.350, 0.487] |
| BalancedSoftmax + targeted batch | 0.349 [0.267, 0.441] | 0.342 [0.222, 0.461] | 0.434 [0.367, 0.527] |
| BalancedSoftmax + hedonism/security lift | 0.346 [0.279, 0.435] | 0.328 [0.222, 0.432] | 0.442 [0.378, 0.526] |

## Family Deltas vs Incumbent

| Challenger family | Delta QWK | Verdict | Delta recall_-1 | Verdict | Delta MinR | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| BalancedSoftmax + dimweight | -0.020 [-0.062, 0.018] | likely_noise | 0.065 [0.021, 0.128] | distinguishable_gain | 0.001 [-0.037, 0.036] | likely_noise |
| BalancedSoftmax + circreg + recall floor | 0.004 [-0.036, 0.046] | likely_noise | -0.046 [-0.098, -0.008] | distinguishable_regression | -0.039 [-0.082, -0.020] | distinguishable_regression |
| BalancedSoftmax + targeted batch | -0.013 [-0.053, 0.019] | likely_noise | 0.029 [-0.029, 0.088] | likely_noise | -0.014 [-0.055, 0.006] | likely_noise |
| BalancedSoftmax + hedonism/security lift | -0.016 [-0.059, 0.014] | likely_noise | 0.015 [-0.036, 0.064] | likely_noise | -0.006 [-0.041, 0.015] | likely_noise |

## Hard-Dimension QWK and Chance Tests

| Family | Dimension | Family-median QWK | Permutation p | Above chance? |
| --- | --- | --- | --- | --- |
| Current default BalancedSoftmax | hedonism | 0.247 [0.120, 0.481] | 0.001 | yes |
| Current default BalancedSoftmax | security | 0.297 [0.162, 0.457] | 0.002 | yes |
| Current default BalancedSoftmax | stimulation | 0.161 [-0.040, 0.526] | 0.047 | yes |
| BalancedSoftmax + dimweight | hedonism | 0.129 [0.000, 0.367] | 0.091 | no |
| BalancedSoftmax + dimweight | security | 0.222 [0.086, 0.410] | 0.003 | yes |
| BalancedSoftmax + dimweight | stimulation | 0.152 [-0.094, 0.549] | 0.126 | no |
| BalancedSoftmax + circreg + recall floor | hedonism | 0.160 [-0.005, 0.378] | 0.029 | yes |
| BalancedSoftmax + circreg + recall floor | security | 0.229 [0.094, 0.426] | 0.001 | yes |
| BalancedSoftmax + circreg + recall floor | stimulation | 0.196 [-0.099, 0.692] | 0.119 | no |
| BalancedSoftmax + targeted batch | hedonism | 0.147 [-0.002, 0.312] | 0.088 | no |
| BalancedSoftmax + targeted batch | security | 0.300 [0.162, 0.460] | 0.001 | yes |
| BalancedSoftmax + targeted batch | stimulation | 0.159 [-0.120, 0.567] | 0.101 | no |
| BalancedSoftmax + hedonism/security lift | hedonism | 0.256 [0.181, 0.525] | 0.001 | yes |
| BalancedSoftmax + hedonism/security lift | security | 0.199 [0.051, 0.397] | 0.003 | yes |
| BalancedSoftmax + hedonism/security lift | stimulation | 0.186 [-0.056, 0.592] | 0.081 | no |

## Promotion Gate Recommendation

1. Default replacement decisions should require family-median delta intervals, not point medians alone. Treat any 95% BCa delta interval spanning zero as unresolved noise.
2. For a tail-first promotion, require the challenger's 95% BCa lower bound to stay above zero on both `recall_-1` and minority recall, while `qwk_mean` must not show a materially negative interval. As a practical review guardrail, treat `qwk_mean` lower bounds below `-0.010` as a meaningful regression risk.
3. Treat `hedonism`, `security`, and other hard-dimension QWK numbers as tie-breakers only after two conditions hold: the family-level hard-dimension QWK is above chance under permutation (`p < 0.05`), and the challenger's family-level delta is not classified as bootstrap noise.

## Artifacts

- Artifact root: `/Users/desmondchoy/Projects/twinkl/logs/experiments/artifacts/frontier_uncertainty_twinkl_730_20260315_195919`
- Family delta plot: `/Users/desmondchoy/Projects/twinkl/logs/experiments/artifacts/frontier_uncertainty_twinkl_730_20260315_195919/plots/family_delta_intervals.png`
- Hard-dimension QWK plot: `/Users/desmondchoy/Projects/twinkl/logs/experiments/artifacts/frontier_uncertainty_twinkl_730_20260315_195919/plots/hard_dimension_qwk.png`
