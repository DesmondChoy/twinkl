# twinkl-w2mu LLM Critic Baseline

Generated: 2026-07-02T15:03:48.001214+00:00

## Contract

- `student_visible`: current journal session plus normalized 10-dim Core Values profile.
- `human_context`: `student_visible` plus previous entries for the same persona where previous.t_index < current.t_index.
- `full_judge_context`: `human_context` plus persona bio and demographics; upper-bound diagnostic only.
- Always excluded: future entries, target labels, rationales, and generation metadata.
- Output: structured JSON scores in {-1, 0, +1} for all 10 values.

## Summary

| Arm | Model | Effort | Shots | Rows | QWK | recall_-1 | MinR | Hedging | p95 latency | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| student_visible | gpt-5.4-nano | none | 0 | 221 | 0.350 | 0.134 | 0.327 | 0.861 | 3.215s | $0.0489 |
| student_visible | gpt-5.4-mini | none | 0 | 221 | 0.434 | 0.188 | 0.428 | 0.789 | 1.884s | $0.1808 |
| human_context | gpt-5.4-nano | none | 0 | 221 | 0.399 | 0.182 | 0.386 | 0.833 | 2.855s | $0.0830 |
| human_context | gpt-5.4-mini | none | 0 | 221 | 0.450 | 0.302 | 0.534 | 0.707 | 3.077s | $0.3089 |
| current_vif | run_020_BalancedSoftmax | n/a | n/a | 221 | 0.378 | 0.342 | 0.449 | 0.621 | n/a | n/a |

## Context Gap

| Model | Effort | Shots | Metric | human_context - student_visible | Interpretation |
|---|---:|---:|---|---:|---|
| gpt-5.4-nano | none | 0 | qwk_mean | 0.049 | small context gap |
| gpt-5.4-nano | none | 0 | recall_minus1 | 0.048 | small context gap |
| gpt-5.4-nano | none | 0 | minority_recall_mean | 0.058 | history likely helps |
| gpt-5.4-mini | none | 0 | qwk_mean | 0.017 | small context gap |
| gpt-5.4-mini | none | 0 | recall_minus1 | 0.115 | history likely helps |
| gpt-5.4-mini | none | 0 | minority_recall_mean | 0.106 | history likely helps |

## Per-Dimension QWK

| Dimension | Human Fleiss k | run_020 | student_visible:gpt-5.4-nano | student_visible:gpt-5.4-mini | human_context:gpt-5.4-nano | human_context:gpt-5.4-mini | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| self_direction | 0.440 | 0.541 | 0.375 | 0.363 | 0.359 | 0.401 | run_020 stronger; small history gap; best human_context:gpt-5.4-mini |
| stimulation | 0.580 | 0.303 | 0.169 | 0.314 | 0.234 | 0.336 | near tie; small history gap; best human_context:gpt-5.4-mini |
| hedonism | 0.640 | 0.262 | 0.225 | 0.182 | 0.295 | 0.209 | near tie; history helps; best human_context:gpt-5.4-nano |
| achievement | 0.470 | 0.334 | 0.360 | 0.554 | 0.500 | 0.567 | LLM stronger; small history gap; best human_context:gpt-5.4-mini |
| power | 0.610 | 0.342 | 0.489 | 0.524 | 0.329 | 0.441 | LLM stronger; history hurts; best student_visible:gpt-5.4-mini |
| security | 0.480 | 0.213 | 0.196 | 0.389 | 0.337 | 0.449 | LLM stronger; history helps; best human_context:gpt-5.4-mini |
| conformity | 0.430 | 0.577 | 0.363 | 0.460 | 0.471 | 0.489 | run_020 stronger; small history gap; best human_context:gpt-5.4-mini |
| tradition | 0.500 | 0.483 | 0.522 | 0.494 | 0.470 | 0.513 | near tie; small history gap; best student_visible:gpt-5.4-nano |
| benevolence | 0.610 | 0.367 | 0.203 | 0.373 | 0.239 | 0.429 | LLM stronger; history helps; best human_context:gpt-5.4-mini |
| universalism | 0.720 | 0.355 | 0.597 | 0.684 | 0.752 | 0.669 | LLM stronger; history helps; best human_context:gpt-5.4-nano |

## Notes

- Verdicts are heuristic: compare per-dimension QWK against run_020 and check whether history improves the best LLM arm.
- Treat large-model comparisons, if any, as oracle diagnostics only.
