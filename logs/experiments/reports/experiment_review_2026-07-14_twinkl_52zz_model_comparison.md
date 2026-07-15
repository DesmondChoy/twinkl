# Weekly Drift Reviewer Model Comparison (`twinkl-52zz`)

**Date:** 2026-07-14
**Disposition:** `gpt-5.6-luna` at reasoning effort `none` is the frozen
baseline for the reasoning-effort comparison; `low` is the current development
Weekly Drift Reviewer; no deployment approval

## Result

On the complete 204-persona development data, `gpt-5.6-luna` at reasoning
effort `none` found substantially more known Drifts than
`gpt-5.4-mini-2026-03-17` at the same reasoning effort and with the same prompt.
It also produced substantially more false Drift alerts.

| Weekly Drift Reviewer | Median Drift hits | Median Drift recall | Median false Drift alerts | Median Drift precision | Median coverage | Median delay | Invalid responses | Median latency | Standard-rate token calculation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-mini-2026-03-17`, reasoning `none` | 7/42 | 0.167 | 5 | 0.583 | 0.740 | 5.0 days | 38/2,853 | 1.61 s | $4.85 |
| `gpt-5.6-luna`, reasoning `none` | 20/42 | 0.476 | 13 | 0.606 | 0.777 | 2.5 days | 16/2,853 | 1.38 s | $6.01 |

Luna's Drift recall was higher in every repeat: `0.429`, `0.476`, and `0.524`,
versus mini's `0.167`, `0.167`, and `0.238`. It found a median 12/28 known
cross-week Drifts versus mini's 5/28.

The paired trajectory bootstrap reports Luna minus mini:

| Metric | Median paired delta | 95% interval |
|---|---:|---:|
| Drift recall | +0.286 | [+0.158, +0.425] |
| False Drift alerts | +9 | [+2, +17] |
| Drift precision | +0.007 | [-0.263, +0.190] |
| Coverage | +0.034 | [-0.007, +0.092] |
| Detection delay | -2.5 days | [-5.0, +1.0] |

The recall improvement is not confined to cases with historical VIF Critic
training provenance. On the 16 non-training Drifts, Luna's median Drift recall
was `0.688`, versus `0.188` for mini. Luna had a median five false Drift alerts
on that subgroup; mini had zero.

## Interpretation

Luna is the stronger Conflict reviewer in this comparison. Its median
Journal Entry Conflict recall rose from `0.245` to `0.513` and macro Conflict
recall from `0.232` to `0.433`. Journal Entry Conflict precision fell from
`0.661` to `0.603`, which explains the additional false Drift alerts. The
deterministic two-Conflict Drift Detector filtered enough isolated errors that
median Drift precision remained similar.

The preregistered selection rule required higher Drift recall without more
false Drift alerts. Luna fails that rule, so the mechanical disposition is
`inconclusive_keep_gpt_5_4_mini_baseline`. This is not evidence that mini is the
better reviewer. It records that Luna did not pass the preregistered
no-added-false-alert rule; the later development decision below explicitly
accepts that trade-off rather than rewriting the rule after seeing the result.

`gpt-5.6-luna` at reasoning effort `none` is the frozen baseline for the
reasoning-effort comparison, with explicit acceptance of the median increase
from five to thirteen false Drift alerts. The exact prompt, model, and
reasoning-effort-`none` setup remain fixed. The
[`low` comparison](experiment_review_2026-07-14_twinkl_52zz_luna_low.md) records
the current development Weekly Drift Reviewer. Neither result grants deployment
approval.

## Protocol

- Complete synthetic development data: 204 personas, 1,651 Journal Entries,
  292 persona/Core Value cases, 2,377 Journal Entry/Core Value combinations,
  951 persona-weeks, and 42 known Drifts across 36 Drift trajectories.
- Exact same `weekly_vif_verifier` prompt without VIF Critic input for both
  models.
- Responses API, reasoning effort `none`, structured `WeeklyVerifierResponse`,
  three repeats, `store: false`, service tier `default`, two attempts, and the
  same fail-closed validation.
- 5,706 calls total. Both runs used the same frozen 951 prompt hashes and ran
  concurrently into separate resumable response files.
- The two requested model identifiers were returned unchanged by the API.
- The preregistered token estimate was `$11.976405`, under the `$15` cap.
  Recorded response-token totals produce a `$10.86520875` standard-rate
  calculation when every input token is charged at the full uncached rate.

## Cost provenance

The downloaded OpenAI Platform Costs export reports `$12.01916515 USD` for the
whole `Default project` from `2026-07-14T00:00:00` to
`2026-07-15T00:00:00` UTC. This is the official project-day billing record and
is below the `$15` experiment cap. It is `$0.04276015` above the preregistered
token estimate, but the two amounts are not directly comparable: the CSV covers
all API activity in that project-day, not only the `twinkl-52zz` calls.

The export has no model, API-key, or line-item breakdown. The experiment
response receipts record aggregate input and output tokens but omit the
cached-input token split. Consequently, the `$10.86520875` value is a
study-specific standard-rate token calculation, while `$12.01916515` is a
broader official billing receipt; neither is an exact billed cost for this
comparison. [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
lists separate full-input and cached-input rates. The receipt and its citation
guidance are preserved in
[`receipts/`](../artifacts/twinkl_52zz_model_comparison_20260714/receipts/README.md).

## Limitations

- The references are AI-reviewed LLM-Judge Conflict Labels on synthetic data,
  not human validation or real-user prevalence.
- This is development data, not a fresh final test. Results can select a
  development configuration but cannot support deployment approval.
- The false Drift alert tolerance is deliberately unresolved under
  `twinkl-7vam`; this report does not invent one after seeing the results.
- This comparison holds reasoning effort at `none`. The separate
  [`low` comparison](experiment_review_2026-07-14_twinkl_52zz_luna_low.md)
  covers the current development selection; no prompt variant or reasoning
  effort above `low` is part of the decision record.

## Reproducibility

Prepare the frozen prompt contract, verify its estimate, or re-score the
committed responses without API calls:

```sh
uv run python -m scripts.experiments.compare_twinkl_52zz_models prepare
uv run python -m scripts.experiments.compare_twinkl_52zz_models estimate
uv run python -m scripts.experiments.compare_twinkl_52zz_models score
```

The paid execution command requires an explicit guard and can limit execution
to one registered model:

```sh
uv run python -m scripts.experiments.compare_twinkl_52zz_models run \
  --model-key all \
  --execute
```

`--model-key` accepts `all`, `gpt_5_4_mini`, or `gpt_5_6_luna`. Global
`--root` and `--config` options override the repository root and registered
configuration path and must precede the subcommand. Paid execution is
unnecessary for re-scoring the committed responses.

- Config: [`twinkl_52zz_model_comparison_v1.yaml`](../../../config/evals/twinkl_52zz_model_comparison_v1.yaml)
- Runner: [`compare_twinkl_52zz_models.py`](../../../scripts/experiments/compare_twinkl_52zz_models.py)
- Frozen manifest: [`manifest.json`](../artifacts/twinkl_52zz_model_comparison_20260714/manifest.json)
- Frozen prompts: [`prompts.jsonl`](../artifacts/twinkl_52zz_model_comparison_20260714/prompts.jsonl)
- Mini responses: [`responses_gpt_5_4_mini.jsonl`](../artifacts/twinkl_52zz_model_comparison_20260714/responses_gpt_5_4_mini.jsonl)
- Luna responses: [`responses_gpt_5_6_luna.jsonl`](../artifacts/twinkl_52zz_model_comparison_20260714/responses_gpt_5_6_luna.jsonl)
- Metrics: [`metrics.json`](../artifacts/twinkl_52zz_model_comparison_20260714/metrics.json)
- Cost receipt: [`openai_cost_2026-07-14_utc.csv`](../artifacts/twinkl_52zz_model_comparison_20260714/receipts/openai_cost_2026-07-14_utc.csv)
- Cost receipt note: [`receipts/README.md`](../artifacts/twinkl_52zz_model_comparison_20260714/receipts/README.md)
