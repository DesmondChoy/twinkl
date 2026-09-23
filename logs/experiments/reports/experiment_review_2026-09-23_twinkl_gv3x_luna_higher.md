# Weekly Drift Reviewer Luna higher-reasoning comparison (`twinkl-gv3x`)

**Date:** 2026-09-23
**Scope:** extend the matched prompt-version-4.0 GPT-5.6 versus GPT-6 Luna study from `none` and `low` to `medium`, `high`, and `xhigh`. The application still uses `gpt-5.6-luna` at `low`.

## Result

On the complete synthetic development set, GPT-6 Luna had higher trajectory coverage and lower calculated token cost than GPT-5.6 Luna at each of the three new reasoning efforts. None of the paired Drift-recall intervals establishes an increase or decrease. GPT-6's median Drift recall across the full prompt-v4 series was `0.357`, `0.405`, `0.452`, `0.405`, and `0.452` from `none` through `xhigh`. The nonmonotonic series does not establish that more GPT-6 reasoning improves Drift recall. The earlier `none` and `low` results are in [`twinkl-q6pt`](experiment_review_2026-09-23_twinkl_q6pt_luna_model.md).

| Effort | Model | Drift hits, Runs 1 / 2 / 3 (of 42) | Median Drift recall | False Drift alerts, Runs 1 / 2 / 3 | Median precision | Median coverage | Invalid responses / 2,853 | Median attempt latency | Calculated cost, all Runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `medium` | `gpt-5.6-luna` | 19 / 17 / 17 | 0.405 | 2 / 2 / 3 | 0.895 | 0.873 | 11 | 3.50 s | $1.5289 |
| `medium` | `gpt-6-luna` | 18 / 19 / 19 | 0.452 | 2 / 3 / 2 | 0.900 | 0.986 | 2 | 3.53 s | $0.6736 |
| `high` | `gpt-5.6-luna` | 18 / 18 / 19 | 0.429 | 3 / 4 / 1 | 0.857 | 0.853 | 15 | 3.88 s | $2.1731 |
| `high` | `gpt-6-luna` | 18 / 15 / 17 | 0.405 | 1 / 3 / 2 | 0.895 | 0.976 | 1 | 4.06 s | $0.8381 |
| `xhigh` | `gpt-5.6-luna` | 21 / 21 / 18 | 0.500 | 3 / 5 / 5 | 0.808 | 0.849 | 19 | 5.60 s | $3.9247 |
| `xhigh` | `gpt-6-luna` | 16 / 20 / 19 | 0.452 | 1 / 2 / 4 | 0.909 | 0.966 | 4 | 4.86 s | $1.1121 |

Each setup has 951 Persona-week requests in each of three Runs. Coverage is the share of 292 Persona/Core Value trajectories with usable decisions. An invalid Weekly Drift Reviewer response becomes Abstain through the existing fail-closed rule. The same Drift Detector then applies the two-consecutive-Conflict rule. There were no terminal provider errors or provider responses marked incomplete.

The paired trajectory bootstrap compares GPT-6 minus GPT-5.6 within each effort. It resamples the 292 matched trajectories 10,000 times and recomputes the median of the three paired Run differences. These paired differences can differ from subtracting the two row medians.

| Metric | `medium`: paired median [95% interval] | `high`: paired median [95% interval] | `xhigh`: paired median [95% interval] |
|---|---:|---:|---:|
| Drift recall | +0.048 [−0.065, +0.130] | −0.048 [−0.128, +0.045] | −0.024 [−0.125, +0.050] |
| False Drift alerts | 0 [−3, +2] | −1 [−4, +2] | −2 [−4, 0] |
| Drift precision | −0.005 [−0.066, +0.129] | +0.015 [−0.132, +0.159] | +0.066 [−0.005, +0.162] |
| Coverage | +0.116 [+0.086, +0.147] | +0.120 [+0.089, +0.154] | +0.120 [+0.086, +0.147] |
| Median detection delay | −1 day [−2, +0.5] | 0 days [−2, +1] | +0.5 days [−0.5, +2] |

At `medium`, the observed GPT-6 recall is higher, while the paired interval includes zero. At `high`, its median recall is lower, with an interval that also includes zero. At `xhigh`, GPT-6 has a lower raw row median and two fewer false alerts in the paired result; the recall interval includes zero and the false-alert interval reaches zero. The reliable directional result across these three matched comparisons is higher coverage. The calculated token-cost reductions are 56% at `medium`, 61% at `high`, and 72% at `xhigh`. Median attempt latency was similar at `medium`, 5% higher for GPT-6 at `high`, and 13% lower at `xhigh`; provider load and the concurrency amendment make latency diagnostic rather than a service guarantee.

The completed prompt-v4 curve is now in the [interactive research-path chart](../../../docs/capstone_report/vif-to-weekly-drift-research-path.html). Historical prompt-v2 reasoning results used a different prompt and should not serve as the matched model-swap baseline. No result here changes the current Weekly Drift Reviewer contract or grants deployment approval.

## Protocol and provenance

The new study uses exactly the 951 frozen September prompt-v4 requests, 204 synthetic Personas, 292 Persona/Core Value trajectories, and 42 AI-reviewed reference Drifts from the `none`/`low` comparison. Source-request, prompt, reference-source, reference-case, and request hashes match its manifest. All six arms share the same response schema, source-quote validation, fail-closed decisions, and Drift Detector. Within each reasoning effort, only the model ID differs. The output-token cap is 8,000 for `medium` and `high` and 32,000 for `xhigh`, matched between models. All arms use the Responses API, default service tier, `store: false`, a 120-second timeout, up to two attempts on transient errors, schedule seed `20260923`, and three Runs.

The 24-week length-stratified smoke sample produced 144 terminal responses. Two completed responses failed exact-source evidence-quote validation; none reached the output cap. Smoke responses were retained as part of Run 1. The full ledger has exactly 17,118 unique terminal responses, 2,853 per arm. Every provider response returned the requested model ID. The 17,118 provider responses all reported `completed`. Two additional GPT-5.6 `xhigh` attempts timed out at 120 seconds, and both succeeded on retry.

After 1,087 terminal responses, request concurrency was raised from 16 to 48 to complete the study in a practical time. The original configuration, manifest, and attempt journal are preserved in the artifact directory. Sixteen requests were in flight when that process stopped; their unpaired reservations were removed from the active journal and their keys were rerun. The manifest records the amendment, hashes, and affected keys. Request content, model assignments, output caps, validation, scoring, and spend ceiling did not change. The final active journal has 17,120 finished attempts and no unpaired reservations.

The cache-aware cost calculation uses provider token receipts and the published rates for [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna). It totals **$10.2505** across the six arms, versus the preregistered $30 calculated-spend ceiling. This is a receipt-based calculation, not a billing export. Billing for the sixteen interrupted calls and two timed-out attempts is unknown, so the figure may be lower than the invoice total. GPT-5.6 and GPT-6 recorded respectively 0.84 / 0.83 million output tokens at `medium`, 1.38 / 1.15 million at `high`, and 2.83 / 1.70 million at `xhigh`.

## Evidence limits

Reference Drifts come from AI-reviewed LLM-Judge Conflict Labels on synthetic Persona histories. The study is a development comparison without human validation, real-user prevalence, a fresh final test, or deployment approval. Three Runs and shared synthetic trajectories limit how broadly the bootstrap intervals generalize. The two prompt-v4 studies used different output caps and ran at different times; cross-effort latency and cost comparisons are descriptive. Model comparisons within each new effort use the same cap and schedule.

## Reproduction and artifacts

With the repository environment active, these commands verify and re-score the saved data without API calls:

```sh
uv run python -m scripts.experiments.replay_luna_comparison gv3x verify
uv run python -m scripts.experiments.replay_luna_comparison gv3x score
```

The replay command checks the Reviewer's source against the experiment revision
and allows the subsequent live model-ID change. The original runner and manifest
remain frozen. `score` rewrites `metrics.json` with a new scoring timestamp.

`smoke --execute` and `run --execute` make paid API calls. The runner resumes from terminal responses, and the configuration sets the calculated-spend ceiling.

- [Configuration](../../../config/evals/twinkl_gv3x_luna_higher_v1.yaml)
- [Runner](../../../scripts/experiments/compare_twinkl_gv3x_luna_higher.py)
- [Manifest](../artifacts/twinkl_gv3x_luna_higher_20260923/manifest.json)
- [Attempt receipts](../artifacts/twinkl_gv3x_luna_higher_20260923/attempts.jsonl)
- [Terminal responses](../artifacts/twinkl_gv3x_luna_higher_20260923/responses.jsonl)
- [Full metrics](../artifacts/twinkl_gv3x_luna_higher_20260923/metrics.json)

Validation: 59 focused experiment tests passed; Ruff check and format check passed for the new runner. The scorer required every terminal request exactly once, and independent arithmetic checks confirmed per-arm counts, reference denominators, predicted Drift totals, and status totals. The chart was checked statically against metrics; browser visual inspection remains unavailable under the local-file browser policy.
