# Luna Higher-Reasoning Comparison (`twinkl-ck3w`)

**Date:** 2026-08-09
**Decision status:** pending user decision; Luna `low` remains the fixed
development Weekly Drift Reviewer

## Result

The study compared `gpt-5.6-luna` reasoning effort `medium`, `high`, and
`xhigh` with the frozen Luna-`low` and Luna-`none` responses. All setups used
the same 951 prompts, 204-persona development data, three repeats, structured
response, and Weekly Drift Reviewer without VIF Critic input.

Latency is the median end-to-end API time for one terminal persona-week call.
Each persona can contribute multiple weeks, and each persona-week has three
repeat calls. This unit measures the latency that one Weekly Drift Reviewer
request adds. It does not mix a persona's complete journal history into one
timing value.

| Reasoning effort | Median Drift hits | Median Drift recall | Median false Drift alerts | Median Drift precision | Median coverage | Median abstention | Median delay | Median call latency | Invalid responses | Current-rate calculation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 20/42 | 0.476 | 13 | 0.606 | 0.777 | 0.223 | 2.5 days | 1.38 seconds | 16/2,853 | $1.20 |
| `low` | 23/42 | 0.548 | 4 | 0.852 | 0.637 | 0.363 | 5.0 days | 2.81 seconds | 8/2,853 | $1.67 |
| `medium` | 24/42 | 0.571 | 4 | 0.857 | 0.682 | 0.318 | 4.5 days | 2.96 seconds | 14/2,853 | $1.92 |
| `high` | 26/42 | 0.619 | 8 | 0.758 | 0.716 | 0.284 | 4.0 days | 3.66 seconds | 9/2,853 | $2.71 |
| `xhigh` | 28/42 | 0.667 | 9 | 0.750 | 0.702 | 0.298 | 4.0 days | 4.91 seconds | 8/2,853 | $4.77 |

The paired trajectory bootstrap reports each setup minus Luna `low`:

| Reasoning effort | Drift-recall delta, 95% interval | False-alert delta, 95% interval | Coverage delta, 95% interval | Delay delta, 95% interval |
|---|---:|---:|---:|---:|
| `medium` | 0.000 [-0.085, +0.077] | -1 [-2, +3] | +0.045 [+0.007, +0.082] | -0.5 days [-2.0, +0.5] |
| `high` | +0.048 [-0.023, +0.128] | +4 [0, +7] | +0.079 [+0.027, +0.116] | -1.0 day [-2.0, 0.0] |
| `xhigh` | +0.095 [+0.023, +0.186] | +5 [+1, +9] | +0.065 [+0.027, +0.106] | -1.0 day [-3.0, 0.0] |

The paired trajectory bootstrap also reports each reasoning setup minus Luna
`none`:

| Reasoning effort | Drift-recall delta, 95% interval | False-alert delta, 95% interval | Coverage delta, 95% interval | Delay delta, 95% interval |
|---|---:|---:|---:|---:|
| `low` | +0.071 [-0.071, +0.205] | -9 [-16, -3] | -0.140 [-0.188, -0.092] | +2.5 days [0.0, +4.0] |
| `medium` | +0.048 [-0.050, +0.194] | -10 [-16, -3] | -0.096 [-0.147, -0.048] | +2.0 days [-0.5, +4.0] |
| `high` | +0.143 [+0.027, +0.257] | -5 [-13, +1] | -0.075 [-0.123, -0.027] | +1.5 days [-1.0, +3.5] |
| `xhigh` | +0.190 [+0.053, +0.318] | -4 [-12, +2] | -0.075 [-0.123, -0.024] | +0.5 days [-1.5, +3.0] |

`medium` does not establish a gain over `low`. `high` raises the median result,
but its Drift-recall interval includes zero and it adds false Drift alerts.
`xhigh` establishes both a Drift-recall gain and an increase in false Drift
alerts. It is a more aggressive operating point, not a clean improvement.

Against `none`, `high` and `xhigh` establish Drift-recall gains. `Low` and
`medium` establish fewer false Drift alerts, but their Drift-recall intervals
include zero. All reasoning efforts reduce coverage against `none`.

The approved metric hierarchy ranks Drift recall first, false Drift alerts
second, and coverage as a diagnostic. Under that hierarchy, `xhigh` ranks
first. Selecting it would accept a median rise from 4 to 9 false Drift alerts
and a current-rate full-run calculation that is about 2.9 times the `low`
calculation. This report does not change the fixed Luna-`low` contract without
an explicit user decision.

## Repeat stability

- `medium` Drift hits were 24, 21, and 24. False Drift alerts were 4, 3, and 6.
- `high` Drift hits were 26, 26, and 25. False Drift alerts were 6, 9, and 8.
- `xhigh` Drift hits were 28, 28, and 27. False Drift alerts were 8, 10, and 9.

The repeat-level median call latencies were stable within each run: 1.377 to
1.381 seconds for `none`, 2.794 to 2.840 for `low`, 2.943 to 2.977 for
`medium`, 3.659 to 3.670 for `high`, and 4.857 to 4.921 for `xhigh`.

The latency values are diagnostic only. `None` and `low` ran on earlier dates.
The higher-reasoning study also changed client concurrency and output ceilings
during its documented continuation. Network and service load were not held
constant, so the values do not establish that reasoning effort alone caused
the latency differences.

## Cost

[OpenAI Docs](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
listed Luna at $0.20 input, $0.02 cached input, $0.25 cache write, and $1.20
output per million tokens on 2026-08-09.

The three final response sets produce a $9.4009938 full-input standard-rate
token calculation and an $8.48606217 cache-aware calculation. Smoke tests and
discarded incomplete attempts add $0.3040494 and $0.29688253 respectively.
The complete study therefore produces a $9.7050432 standard-rate calculation
or an $8.7829447 cache-aware calculation. These values are token calculations,
not an OpenAI billing export.

At the same current rates, the frozen `none` responses produce a $1.2028278
full-input calculation. This historical reference cost is not added to the
$9.7050432 cost of the new higher-reasoning study.

Reasoning-output use was 588,962 tokens for `medium`, 1,248,121 for `high`, and
2,962,637 for `xhigh`. The frozen `low` run used 387,885 reasoning-output
tokens.

## Protocol and amendments

- The study used the committed July prompt JSONL with SHA-256
  `f0c7e68b5906c3ceeaf27dfc5d5b305252ee2298d688193363d79f6ac370c539`.
  The current prompt template does not rebuild byte-for-byte to that file.
- The initial 2,000-output-token smoke tests passed. The partial full runs then
  produced 14 incomplete `high` responses and 52 incomplete `xhigh` responses.
  The runs stopped before further calls.
- Every initial receipt remains stored. Only completed receipts continued into
  the final response sets. Incomplete and non-terminal error receipts were
  retried.
- The ceiling increased to 8,000 for all three setups. Two `xhigh` responses
  then reached that ceiling, so only `xhigh` increased to 32,000. No final
  response reached the final ceiling.
- Client concurrency moved from 4 to 12 and then 24. The final executor used a
  continuous 24-worker pool. These changes affect scheduling, so recorded
  latency is diagnostic and is not directly comparable with Luna `low`.
- One `xhigh` call returned a non-terminal error and then completed on the
  recorded retry. The final data contain 2,853 terminal coordinates for every
  setup.

## Limitations

- The references are AI-reviewed LLM-Judge Conflict Labels on synthetic
  development data. They are not human validation or real-user prevalence.
- This is the same development data used to select Luna `low`, not a fresh
  final test.
- The prompt, schema, reference data, model, and scoring stayed fixed. Output
  ceilings and client scheduling changed to prevent truncation and excessive
  idle time.
- Latency was not measured under a controlled common schedule. It cannot guide
  the reasoning-effort selection without a new latency-controlled run.
- No setup receives deployment approval from this study.

## Reproducibility

Re-score the stored responses without API calls:

```sh
uv run python -m scripts.experiments.compare_twinkl_ck3w_luna_higher_reasoning prepare
uv run python -m scripts.experiments.compare_twinkl_ck3w_luna_higher_reasoning score
```

- Config: [`twinkl_ck3w_luna_higher_reasoning_v1.yaml`](../../../config/evals/twinkl_ck3w_luna_higher_reasoning_v1.yaml)
- Runner: [`compare_twinkl_ck3w_luna_higher_reasoning.py`](../../../scripts/experiments/compare_twinkl_ck3w_luna_higher_reasoning.py)
- Manifest: [`manifest.json`](../artifacts/twinkl_ck3w_luna_higher_reasoning_20260809/manifest.json)
- Metrics: [`metrics.json`](../artifacts/twinkl_ck3w_luna_higher_reasoning_20260809/metrics.json)
- Medium responses: [`responses_gpt_5_6_luna_medium.jsonl`](../artifacts/twinkl_ck3w_luna_higher_reasoning_20260809/responses_gpt_5_6_luna_medium.jsonl)
- High responses: [`responses_gpt_5_6_luna_high.jsonl`](../artifacts/twinkl_ck3w_luna_higher_reasoning_20260809/responses_gpt_5_6_luna_high.jsonl)
- Xhigh responses: [`responses_gpt_5_6_luna_xhigh.jsonl`](../artifacts/twinkl_ck3w_luna_higher_reasoning_20260809/responses_gpt_5_6_luna_xhigh.jsonl)
- Frozen Luna-`none` report: [`experiment_review_2026-07-14_twinkl_52zz_model_comparison.md`](experiment_review_2026-07-14_twinkl_52zz_model_comparison.md)
- Frozen Luna-`low` report: [`experiment_review_2026-07-14_twinkl_52zz_luna_low.md`](experiment_review_2026-07-14_twinkl_52zz_luna_low.md)
