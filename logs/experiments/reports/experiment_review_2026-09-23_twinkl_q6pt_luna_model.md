# Weekly Drift Reviewer Luna model comparison (`twinkl-q6pt`)

**Date:** 2026-09-23
**Scope:** replace `gpt-5.6-luna` with `gpt-6-luna` in the same Weekly Drift Reviewer requests, separately at reasoning effort `none` and `low`. The application model setting was not changed.

## Result

GPT-6 Luna did not establish higher Drift recall at either reasoning effort. At `none`, it produced substantially fewer false Drift alerts and covered more Core Value trajectories. At `low`, coverage also rose, while the observed false alert reduction was too small to distinguish from variation in these three Runs. GPT-6 Luna used fewer output tokens and had a lower calculated token cost in both comparisons.

| Reasoning effort | Model | Drift hits across Runs (of 42) | Median Drift recall | False Drift alerts across Runs | Median Drift precision | Median coverage | Median abstention | Invalid / error responses | Median attempt latency | Calculated token cost, all Runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | `gpt-5.6-luna` | 15 / 13 / 15 | 0.357 | 10 / 7 / 9 | 0.625 | 0.918 | 0.082 | 12 / 0 | 1.75 s | $0.9112 |
| `none` | `gpt-6-luna` | 13 / 15 / 15 | 0.357 | 2 / 3 / 2 | 0.867 | 0.976 | 0.024 | 7 / 5 | 1.79 s | $0.4177 |
| `low` | `gpt-5.6-luna` | 17 / 16 / 16 | 0.381 | 4 / 3 / 2 | 0.842 | 0.860 | 0.140 | 12 / 0 | 2.82 s | $1.2730 |
| `low` | `gpt-6-luna` | 17 / 15 / 17 | 0.405 | 2 / 2 / 1 | 0.895 | 0.990 | 0.010 | 2 / 0 | 2.35 s | $0.5083 |

| Reasoning effort | Model | Drift recall, Runs 1 / 2 / 3 | Coverage, Runs 1 / 2 / 3 |
|---|---|---:|---:|
| `none` | `gpt-5.6-luna` | 0.357 / 0.310 / 0.357 | 0.918 / 0.918 / 0.945 |
| `none` | `gpt-6-luna` | 0.310 / 0.357 / 0.357 | 0.976 / 0.983 / 0.973 |
| `low` | `gpt-5.6-luna` | 0.405 / 0.381 / 0.381 | 0.877 / 0.846 / 0.860 |
| `low` | `gpt-6-luna` | 0.405 / 0.357 / 0.405 | 0.990 / 0.990 / 0.986 |

Each model and effort combination made 2,853 terminal requests: 951 observed weeks × three Runs. Coverage is the fraction of the 292 Persona/Core Value trajectories with usable decisions. An invalid or failed Weekly Drift Reviewer response becomes Abstain under the existing fail-closed rule. The Drift Detector then applies the same two-consecutive-Conflict rule to every setup.

Across 2,853 requests per setup, invalid-response rates were 0.42% for GPT-5.6 `none`, 0.25% for GPT-6 `none`, 0.42% for GPT-5.6 `low`, and 0.07% for GPT-6 `low`. GPT-6 `none` also had a 0.18% terminal error rate. The reported abstention is measured on whole Persona/Core Value trajectories after applying the Drift Detector, so it has a different denominator.

The paired trajectory bootstrap compares GPT-6 Luna minus GPT-5.6 Luna within each reasoning effort. It resamples the 292 matched trajectories 10,000 times and recomputes the median of the three paired Run differences.

| Metric | `none`: paired median difference [95% interval] | `low`: paired median difference [95% interval] |
|---|---:|---:|
| Drift recall | 0.000 [−0.073, +0.065] | 0.000 [−0.095, +0.089] |
| False Drift alerts | −7 [−12, −2] | −1 [−4, +1] |
| Drift precision | +0.257 [+0.093, +0.438] | +0.056 [−0.065, +0.208] |
| Coverage | +0.058 [+0.027, +0.079] | +0.127 [+0.092, +0.164] |
| Median detection delay | 0 days [−2, +2] | 0 days [−1, +2] |

The `low` model rows have different unpaired median Drift recall values (0.381 and 0.405), but their paired median difference is zero: the three Run differences are 0, −1/42, and +1/42. The paired result is the relevant comparison.

## Interpretation

For `none`, GPT-6 Luna has evidence of fewer false Drift alerts and greater coverage without an established change in Drift recall. It produced a median of two false Drift alerts versus nine, with a paired interval entirely below zero. Its median request latency was similar. The calculated token cost was 54% lower.

For `low`, GPT-6 Luna covered more trajectories and its median request latency was 17% lower. Its calculated token cost was 60% lower. The observed changes in Drift recall and false Drift alerts are uncertain. This is the relevant comparison for the current development Weekly Drift Reviewer, which uses `gpt-5.6-luna` at `low`; the experiment does not itself adopt a replacement model.

The `gpt-5.6-luna` results here are fresh requests using the current prompt version `4.0`. Historical Luna measurements with prompt version `2.0` answer a different question and should not be used as the comparator for this model swap.

## Protocol and provenance

- The complete synthetic development set contains 204 Personas, 951 observed weeks, 292 Persona/Core Value trajectories, and 42 AI-reviewed reference Drifts. All four setups use the same 951 frozen September `twinkl-j3k7` prompt version `4.0` requests, response schema, 2,000-output-token cap, Responses API, `store: false`, default service tier, validation, and Drift Detector. Only model ID and the explicit reasoning effort setting vary across the four setups.
- The manifest freezes the Git HEAD, configuration, source request hash, reference hashes, code and prompt hashes, 951 request hashes, Run count, API settings, and pricing. Requests were shuffled with schedule seed `20260923` and executed with 16 concurrent workers. The runner allows two attempts on transient errors; one GPT-5.6 `low` transient failure retried successfully.
- The 24-week length-stratified smoke set was run for all four setups: 96 valid responses, all within the output cap. These responses became part of Run 1, so the total remains 11,412 terminal requests. The complete run has 2,853 unique terminal responses per setup, with 951 in each Run. Returned model IDs match the requested IDs.
- Five GPT-6 `none` calls ended in SDK `ValidationError` without provider usage receipts. One was on a trajectory containing a reference Drift. All five count as fail-closed Abstain. The calculated cost can omit billed tokens for these calls. There were no provider responses marked `incomplete`.
- Independent reconstruction from the raw effective decisions matched every predicted Drift count: GPT-5.6 `none` 25/20/24, GPT-6 `none` 15/18/17, GPT-5.6 `low` 21/19/18, and GPT-6 `low` 19/17/18. The current scorer also reproduced the historical prompt-v4 GPT-5.6 `low` Drift counts before this experiment.
- After provider execution, a type annotation and Ruff formatting were applied to the runner. Both changes are recorded in the manifest with the runner hash used during provider calls. They did not alter request, validation, or scoring behavior. The final metrics hashes match the amended manifest and both receipt files.

The cache-aware cost calculation uses provider token receipts and the published model rates at the time of the run: [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna). Total calculated cost across all four setups was $3.1102. This is a token calculation, not an invoice or billing export. GPT-5.6 `none` / GPT-6 `none` recorded 329,612 / 320,615 output tokens; GPT-5.6 `low` / GPT-6 `low` recorded 631,131 / 500,945. The `low` setups recorded 299,577 / 177,480 reasoning-output tokens, respectively.

## Evidence limits

The reference Drifts come from AI-reviewed LLM-Judge Conflict Labels on synthetic Persona histories. These Runs measure a current development comparison, not human validation, real-user prevalence, a fresh final test, or deployment approval. There are only three Runs per setup, and their shared synthetic trajectories limit how broadly the bootstrap intervals generalize. Cost and latency were observed under this request mix, concurrency, and provider conditions; they are not service guarantees.

## Reproduction and artifacts

With the repository virtual environment active, inspect and re-score the recorded responses without API calls:

```sh
uv run python -m scripts.experiments.replay_luna_comparison q6pt verify
uv run python -m scripts.experiments.replay_luna_comparison q6pt score
```

The replay command checks the Reviewer's source against the experiment revision
and allows the subsequent live model-ID change. The original runner and manifest
remain frozen. `score` rewrites `metrics.json` with a new scoring timestamp.

`prepare` creates the frozen manifest. `smoke --execute` and `run --execute` make paid API calls and resume from the recorded terminal responses. The configuration records their request cap, retry policy, schedule seed, and $20 calculated-spend ceiling.

- [Configuration](../../../config/evals/twinkl_q6pt_luna_model_v1.yaml)
- [Runner](../../../scripts/experiments/compare_twinkl_q6pt_luna.py)
- [Manifest](../artifacts/twinkl_q6pt_luna_model_20260923/manifest.json)
- [Attempt receipts](../artifacts/twinkl_q6pt_luna_model_20260923/attempts.jsonl)
- [Terminal responses](../artifacts/twinkl_q6pt_luna_model_20260923/responses.jsonl)
- [Full metrics](../artifacts/twinkl_q6pt_luna_model_20260923/metrics.json)

Validation: 60 focused tests passed; Ruff check, Ruff format check, and scoped MyPy passed for the runner.
