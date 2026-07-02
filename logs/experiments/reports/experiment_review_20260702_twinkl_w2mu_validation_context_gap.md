# twinkl-w2mu LLM Critic Baseline

Generated: 2026-07-02T15:03:46.367632+00:00

## Contract

- `student_visible`: current journal session plus normalized 10-dim Core Values profile.
- `human_context`: `student_visible` plus previous entries for the same persona where previous.t_index < current.t_index.
- `full_judge_context`: `human_context` plus persona bio and demographics; upper-bound diagnostic only.
- Always excluded: future entries, target labels, rationales, and generation metadata.
- Output: structured JSON scores in {-1, 0, +1} for all 10 values.

## Summary

| Arm | Model | Effort | Shots | Rows | QWK | recall_-1 | MinR | Hedging | p95 latency | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| student_visible | gpt-5.4-nano | none | 0 | 40 | 0.430 | 0.175 | 0.385 | 0.823 | 2.075s | $0.0090 |
| student_visible | gpt-5.4-mini | none | 0 | 40 | 0.536 | 0.250 | 0.489 | 0.782 | 1.577s | $0.0333 |
| human_context | gpt-5.4-nano | none | 0 | 40 | 0.400 | 0.225 | 0.423 | 0.787 | 1.933s | $0.0148 |
| human_context | gpt-5.4-mini | none | 0 | 40 | 0.435 | 0.325 | 0.531 | 0.700 | 1.886s | $0.0552 |
| full_judge_context | gpt-5.4-nano | none | 0 | 40 | 0.375 | 0.175 | 0.403 | 0.800 | 3.180s | $0.0159 |
| full_judge_context | gpt-5.4-mini | none | 0 | 40 | 0.398 | 0.275 | 0.493 | 0.703 | 1.740s | $0.0593 |
| current_vif | run_020_BalancedSoftmax | n/a | n/a | 221 | 0.378 | 0.342 | 0.449 | 0.621 | n/a | n/a |

## Context Gap

| Model | Effort | Shots | Metric | human_context - student_visible | Interpretation |
|---|---:|---:|---|---:|---|
| gpt-5.4-nano | none | 0 | qwk_mean | -0.031 | small context gap |
| gpt-5.4-nano | none | 0 | recall_minus1 | 0.050 | history likely helps |
| gpt-5.4-nano | none | 0 | minority_recall_mean | 0.038 | small context gap |
| gpt-5.4-mini | none | 0 | qwk_mean | -0.101 | history did not help this arm |
| gpt-5.4-mini | none | 0 | recall_minus1 | 0.075 | history likely helps |
| gpt-5.4-mini | none | 0 | minority_recall_mean | 0.041 | small context gap |

## Per-Dimension QWK

| Dimension | Human Fleiss k | run_020 | student_visible:gpt-5.4-nano | student_visible:gpt-5.4-mini | human_context:gpt-5.4-nano | human_context:gpt-5.4-mini | full_judge_context:gpt-5.4-nano | full_judge_context:gpt-5.4-mini | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| self_direction | 0.440 | 0.541 | 0.008 | 0.241 | 0.130 | 0.194 | 0.000 | 0.321 | run_020 stronger; small history gap; best full_judge_context:gpt-5.4-mini |
| stimulation | 0.580 | 0.303 | 0.583 | 0.649 | 0.211 | 0.593 | 0.545 | 0.545 | LLM stronger; history hurts; best student_visible:gpt-5.4-mini |
| hedonism | 0.640 | 0.262 | 0.394 | 0.408 | 0.417 | 0.408 | 0.338 | 0.295 | LLM stronger; small history gap; best human_context:gpt-5.4-nano |
| achievement | 0.470 | 0.334 | 0.581 | 0.600 | 0.640 | 0.448 | 0.581 | 0.469 | LLM stronger; small history gap; best human_context:gpt-5.4-nano |
| power | 0.610 | 0.342 | 0.313 | 0.660 | 0.313 | 0.606 | 0.319 | 0.154 | LLM stronger; history hurts; best student_visible:gpt-5.4-mini |
| security | 0.480 | 0.213 | 0.584 | 0.657 | 0.502 | 0.303 | 0.427 | 0.484 | LLM stronger; history hurts; best student_visible:gpt-5.4-mini |
| conformity | 0.430 | 0.577 | 0.375 | 0.250 | 0.444 | 0.083 | 0.235 | 0.000 | run_020 stronger; history helps; best human_context:gpt-5.4-nano |
| tradition | 0.500 | 0.483 | 0.525 | 0.494 | 0.690 | 0.645 | 0.639 | 0.603 | LLM stronger; history helps; best human_context:gpt-5.4-nano |
| benevolence | 0.610 | 0.367 | 0.286 | 0.500 | 0.368 | 0.419 | 0.368 | 0.508 | LLM stronger; history hurts; best full_judge_context:gpt-5.4-mini |
| universalism | 0.720 | 0.355 | 0.655 | 0.907 | 0.280 | 0.655 | 0.298 | 0.600 | LLM stronger; history hurts; best student_visible:gpt-5.4-mini |

## Notes

- Verdicts are heuristic: compare per-dimension QWK against run_020 and check whether history improves the best LLM arm.
- Treat large-model comparisons, if any, as oracle diagnostics only.
