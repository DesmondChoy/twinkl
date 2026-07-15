# Luna Reasoning-Effort Comparison (`twinkl-52zz`)

**Date:** 2026-07-14
**Decision updated:** 2026-07-15
**Disposition:** select `gpt-5.6-luna` at reasoning effort `low` as the current
development Weekly Drift Reviewer; stop before `medium`; no deployment approval

## Result

Reasoning effort `low` did not pass the preregistered selection rule against
the frozen reasoning-effort-`none` baseline. It nevertheless becomes the
development selection under the approved metric hierarchy: Drift recall first,
false Drift alerts second, and coverage as a diagnostic. It found three more
known Drifts at the median and cut false Drift alerts from 13 to 4, while median
coverage fell from `0.777` to `0.637` and median delay rose from 2.5 to 5.0 days.

| Luna reasoning effort | Median Drift hits | Median Drift recall | Median false Drift alerts | Median Drift precision | Median coverage | Median abstention | Median delay | Invalid responses | Median latency | Full-input token calculation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 20/42 | 0.476 | 13 | 0.606 | 0.777 | 0.223 | 2.5 days | 16/2,853 | 1.38 s | $6.01 |
| `low` | 23/42 | 0.548 | 4 | 0.852 | 0.637 | 0.363 | 5.0 days | 8/2,853 | 2.81 s | $8.37 |

Reasoning-effort-`low` Drift recall was `0.571`, `0.548`, and `0.548` across
the three repeats, versus `0.429`, `0.476`, and `0.524` for `none`. Its false
Drift alert counts were 5, 4, and 4, versus 15, 13, and 13.

The paired trajectory bootstrap reports `low` minus `none`:

| Metric | Median paired delta | 95% interval |
|---|---:|---:|
| Drift recall | +0.071 | [-0.071, +0.205] |
| False Drift alerts | -9 | [-16, -3] |
| Drift precision | +0.246 | [+0.097, +0.400] |
| Coverage | -0.140 | [-0.188, -0.092] |
| Detection delay | +2.5 days | [0.0, +4.0] |

## Interpretation

`low` is more abstention-heavy, not a clean improvement on every metric. The
statistically clear reduction in false Drift alerts comes with a statistically
clear 14-point loss of coverage. The observed Drift-recall gain is not
statistically clear. A Weekly Drift Reviewer can reduce false Drift alerts by
abstaining more often, so the result does not establish better discrimination
across the full development data.

The preregistered rule allowed at most a `0.05` coverage loss and therefore
mechanically retained `none`. The 2026-07-15 decision does not rewrite that
historical result. It replaces the development-selection rule with the approved
hierarchy: Drift recall first, false Drift alerts second, and coverage as a
diagnostic rather than a veto. On those priorities, `low` is preferred because
its observed Drift recall is higher and its false Drift alert count is much
lower. The user chose to stop at `low`, so `medium` will not be tested.

The configuration and `development_selection` field in `metrics.json` preserve
the preregistered rule and its mechanical `keep_luna_none` result. This report,
the PRD, and the evaluation specifications record the later approved decision.

For the academic proof of concept, the trade-off is useful. It shows that more
reasoning changes the Weekly Drift Reviewer's operating point, latency, and
cost rather than improving every metric together. The selected proof of concept
remains an explainable hybrid: the Weekly Drift Reviewer makes cited Conflict
decisions, then the deterministic Drift Detector applies the
two-consecutive-Conflict rule. This development comparison strengthens the
selection record; it does not provide human validation, a fresh final test, or
deployment approval.

## Protocol

- Same complete synthetic development data: 204 personas, 1,651 Journal
  Entries, 292 persona/Core Value cases, 951 persona-weeks, and 42 known Drifts
  across 36 Drift trajectories.
- Exact frozen `weekly_vif_verifier` prompts without VIF Critic input; prompt
  file SHA-256
  `f0c7e68b5906c3ceeaf27dfc5d5b305252ee2298d688193363d79f6ac370c539`.
- Same `gpt-5.6-luna` model identifier, Responses API, structured response,
  `store: false`, service tier `default`, 2,000-output-token cap, fail-closed
  validation, and three repeats. Only reasoning effort changed from `none` to
  `low`.
- A deterministic 24-prompt smoke test checked execution, the token cap, and
  projected cost. It was not used as selection evidence. All 24 responses were
  valid; the largest used 361 output tokens.
- The full run produced terminal receipts for all 2,853 calls: 2,845 valid and
  eight fail-closed invalid responses. The API returned `gpt-5.6-luna`.

## Cost provenance

The preregistered estimate was `$6.843660`. The smoke test projected `$7.345851`
at full-input standard rates and `$8.815021` after the 20% budget contingency,
below the `$15` cap. The completed run recorded 4,077,741 input tokens and
715,619 output tokens, producing an `$8.371455` calculation if every input
token is charged at the full `$1.00` rate.

The detailed receipts also record 1,970,063 cached-input tokens and 1,447,618
cache-write tokens, leaving 660,060 ordinary input tokens. Applying the
published Standard short-context rates for `gpt-5.6-luna`—`$1.00` input,
`$0.10` cached input, `$1.25` cache writes, and `$6.00` output—produces a
cache-aware token calculation of `$6.9603028`. This explains why actual API
cost can be lower than the full-input estimate. It is a call-specific token
calculation, not an OpenAI billing export; the project report should cite an
updated billing export separately if exact invoiced cost is required. See
[OpenAI API pricing](https://developers.openai.com/api/docs/pricing).

Reasoning effort `low` used 387,885 reasoning-output tokens. The frozen `none`
receipts predate detailed cache accounting, so only their conservative
full-input calculation is available for the matched comparison.

## Limitations

- The references are AI-reviewed LLM-Judge Conflict Labels on synthetic data,
  not human validation or real-user prevalence.
- This is development data, not a fresh final test.
- The coverage limit is a preregistered historical choice, not the approved
  development metric hierarchy or a deployment-approved product threshold.
- No prompt change, reasoning effort above `low`, VIF Critic input, runtime
  architecture change, or deployment decision was tested.

## Reproducibility

- Config: [`twinkl_52zz_luna_low_v1.yaml`](../../../config/evals/twinkl_52zz_luna_low_v1.yaml)
- Runner: [`compare_twinkl_52zz_luna_reasoning.py`](../../../scripts/experiments/compare_twinkl_52zz_luna_reasoning.py)
- Manifest: [`manifest.json`](../artifacts/twinkl_52zz_luna_low_20260714/manifest.json)
- Smoke responses: [`smoke_responses.jsonl`](../artifacts/twinkl_52zz_luna_low_20260714/smoke_responses.jsonl)
- Luna `low` responses: [`responses_gpt_5_6_luna_low.jsonl`](../artifacts/twinkl_52zz_luna_low_20260714/responses_gpt_5_6_luna_low.jsonl)
- Metrics: [`metrics.json`](../artifacts/twinkl_52zz_luna_low_20260714/metrics.json)
- Frozen Luna `none` report: [`experiment_review_2026-07-14_twinkl_52zz_model_comparison.md`](experiment_review_2026-07-14_twinkl_52zz_model_comparison.md)
