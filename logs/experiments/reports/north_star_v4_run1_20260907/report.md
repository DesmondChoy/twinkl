# North Star Moment: targeted v4 Run 1 update

**Latest NSM results to use:** the [targeted Weekly Drift v4 Run 1 update](nsm_experiment.json), completed on 7 September 2026, is the current basis for NSM reporting and method comparisons. It supersedes the original v2 repeat-1 results for current reporting; the original record remains historical evidence. These results retain prior observations and remain AI assessments of synthetic histories.

**Status, 7 September 2026:** complete under `twinkl-fz34.14`. Both methods were executed where effective inputs changed, and all 501 cases were regraded. Full eligible history achieved higher Card precision and Opportunity recall than Nomic top-three retrieval in both partitions. This is a targeted update with retained observations, not a wholly fresh independent experiment.

The update preserves the original 105 synthetic Personas, 501 weekly cases, and 81-development/24-final Persona partition. It uses the completed Weekly Drift prompt v4 **Run 1** outputs. The existing `definition` and `core_motivation` fields remain unchanged; Weekly Drift Detection was not rerun. Runtime reviews use `gpt-5.6-luna` at `low`, and shared AI evaluation uses the same model at `xhigh`.

## Results across all 501 cases

| Partition | Method | Card precision | Opportunity recall |
| --- | --- | ---: | ---: |
| Development (391 weeks, 81 Personas) | Nomic top three | 181/309 = 58.58% | 181/327 = 55.35% |
| Development (391 weeks, 81 Personas) | Full eligible history | 244/323 = 75.54% | 244/327 = 74.62% |
| Qualified final (110 weeks, 24 Personas) | Nomic top three | 49/85 = 57.65% | 49/86 = 56.98% |
| Qualified final (110 weeks, 24 Personas) | Full eligible history | 70/85 = 82.35% | 70/86 = 81.40% |

The paired differences below are **Nomic minus full history**, in percentage points. Intervals use 10,000 paired whole-Persona resamples within each partition, seed `20260906`, and the unchanged 95% percentile procedure. Negative differences favour full history.

| Partition | Card precision difference [95% interval] | Opportunity recall difference [95% interval] |
| --- | ---: | ---: |
| Development | -16.97 [-23.14, -10.72] | -19.27 [-25.45, -12.72] |
| Qualified final | -24.71 [-36.18, -10.75] | -24.42 [-36.47, -9.88] |

These results continue to favour full history for quality on this tested synthetic cohort. The update preserves the original evaluation contract, rather than tuning either method to the changed Weekly Drift inputs. Full-history reference opportunities and priorities, contradiction rulings, exact-quotation grades, and paired exclusions were recomputed for every case.

## Denominators, exclusions, and failures

| Partition | Confirmed reference opportunities | Unresolved opportunities | Paired precision exclusions | Paired recall exclusions |
| --- | ---: | ---: | ---: | ---: |
| Development | 353 | 9 | 31 | 35 |
| Qualified final | 96 | 1 | 10 | 11 |

Each affected metric excludes the same case IDs for both methods. Unresolved source opportunities or quotation judgments are retained with their exclusion reasons; runtime failures alone do not remove known opportunities from recall. The consolidated record includes every numerator, denominator, case ID, and exclusion reason.

| Partition | Method | Displayed cards | Runtime failures | Correct omission | Selection-rule correctness | Retrieval hit rate@3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development | Nomic top three | 333 | 5 | 26/29 = 89.66% | 185/309 = 59.87% | 342/348 = 98.28% |
| Development | Full eligible history | 353 | 1 | 24/29 = 82.76% | 255/322 = 79.19% | N/A |
| Qualified final | Nomic top three | 95 | 0 | 13/13 = 100.00% | 51/85 = 60.00% | 96/96 = 100.00% |
| Qualified final | Full eligible history | 95 | 0 | 13/13 = 100.00% | 73/85 = 85.88% | N/A |

Failed requests remain terminal under the original limits. Their identities and provenance are listed below; a request failure can affect a runtime omission or reference/quotation uncertainty depending on its experimental role.

| Request purpose | Evidence | Status |
| --- | --- | --- |
| `runtime:nomic:013d8101:week:2025-12-01:security` | retained | `count_failed` |
| `runtime:nomic:0e9a45e9:week:2026-01-05:hedonism` | retained | `invalid` |
| `runtime:nomic:0e9a45e9:week:2026-01-26:self_direction` | retained | `invalid` |
| `reference:primary:2541429a:week:2025-11-17:tradition` | retained | `invalid` |
| `reference:primary:3a3b15e4:week:2025-06-16:hedonism` | retained | `invalid` |
| `reference:primary:621be543:week:2025-09-29:power` | retained | `failed` |
| `reference:primary:991917dc:week:2025-06-16:security` | retained | `invalid` |
| `reference:primary:d911e384:week:2025-04-14:stimulation` | retained | `invalid` |
| `reference:primary:d911e384:week:2025-05-12:stimulation` | retained | `invalid` |
| `runtime:full_history:f79f99b6:week:2025-07-07:stimulation` | retained | `invalid` |
| `recheck:primary:nomic:3f984932:week:2025-04-14` | retained | `invalid` |
| `quotation:repeat_2:full_history:ed67c9cc:week:2025-02-17` | retained | `invalid` |
| `runtime:nomic:0e9a45e9:week:2026-01-19:self_direction` | new | `invalid` |
| `reference:primary:11de77e8:week:2025-10-13:conformity` | new | `failed` |
| `reference:primary:22b117be:week:2025-05-19:power` | new | `invalid` |
| `reference:primary:3a3b15e4:week:2025-06-09:hedonism` | new | `invalid` |
| `runtime:nomic:87e92805:week:2025-03-31:hedonism` | new | `invalid` |
| `reference:primary:87e92805:week:2025-04-07:hedonism` | new | `invalid` |

Undefined bootstrap resamples: Qualified final / nomic / correct_omission: 2; Qualified final / full_history / correct_omission: 2; Qualified final / nomic_minus_full_history / correct_omission: 2

## Persisted impact audit

The [impact audit](impact_audit.json) confirms **38 affected cases across 22 Personas: 29 development weeks and nine final weeks**. It compares state, selected Core Values, definitions, ordered Journal Entries and eligible Persona responses, source windows, chronology, Profile, and every other effective selection/grading field. Before/after differences and hashes are stored for all 501 cases. Upstream receipt and historical Drift evidence changes are separately retained as provenance. Thirty-seven affected cases change weekly state; `ed67c9cc:week:2025-02-03` remains Active Drift but its eligible source context changes.

| Partition | No Active Drift | Active Drift | Insufficient Evidence | Cases |
| --- | ---: | ---: | ---: | ---: |
| Development | 374 | 6 | 11 | 391 |
| Qualified final | 93 | 11 | 6 | 110 |

All 463 unchanged effective contexts retain the original runtime outputs for both methods, reference judgments, quotation reviews, rechecks, applicable repeats, and final deterministic grades. This includes failures and unresolved judgments. No unchanged observation was opportunistically retried.

## Retained and new evidence

The [separate manifest](manifest.json) binds code, upstream files, the impact audit, cases, retrieval, policy, settings, partition, consistency sample, and complete static requests. The [consolidated record](nsm_experiment.json) links the original record, generation freeze, methodology snapshot, and correction history by hash. The original record remains byte-for-byte unchanged, with SHA-256 `5eb9a0ef4b1c82447ab06ad0bd80f339c5acaccbd3a7de5d7ae3f808c06e2020`.

Reuse requires an identical complete request, purpose, method, case, role, repeat/recheck identity, model settings, prompt, schema, validation/evaluation contract, and attempt limit. Changed ordered source batches were reviewed as complete batches. Quotation reuse also requires the identical quotation, complete source, eligible response, and Core Value context. Each retained receipt records its original receipt hash and justification.

| Evidence purpose | Retained requests | New requests | New generation attempts |
| --- | ---: | ---: | ---: |
| `quotation:primary` | 825 | 51 | 52 |
| `quotation:repeat_2` | 37 | 0 | 0 |
| `quotation:repeat_3` | 37 | 0 | 0 |
| `recheck:primary` | 83 | 8 | 8 |
| `reference:primary` | 639 | 44 | 51 |
| `reference:repeat_2` | 26 | 0 | 0 |
| `reference:repeat_3` | 26 | 0 | 0 |
| `runtime:full_history` | 639 | 44 | 46 |
| `runtime:nomic` | 640 | 43 | 45 |

The update uses 2952 retained requests and 190 new requests, with 202 new generation attempts and 190 new token-count attempts. The original record preserves 64 historical requests that were not reused in the updated contexts. The 131 new static source-review requests comprised 43 Nomic runtime reviews, 44 full-history runtime reviews, and 44 shared primary reference reviews; additional quotation and recheck evidence followed the resulting selections.

Ten affected cases have no eligible source batch under the new contexts and require no semantic source call. The other 28 affected cases produce the new or reusable source-review requests. The largest measured new complete request contains 3660 input tokens, below the unchanged 16,000-token ceiling; inputs were not truncated.

The fixed 20-case consistency sample contains no affected week. Its primary assessments and two additional blinded assessments were retained, with no new repeat calls. Primary judgments, repeats, and rechecks remain distinct. Historical unnecessary rechecks remain preserved as protocol deviations and do not create new recheck authority.

Original Nomic vectors were not persisted. The update re-encoded 138 unique eligible documents from affected cases using the identical pinned encoder configuration and document serialization, then recomputed affected rankings. Maximum input length was 460 tokens, without truncation. New preparation took 6.85 seconds, allocated over the 38 affected cases. Unchanged rankings and timing observations remain historical. The `update_preparation` object supplies new embedding hashes and measurements; inherited top-level embedding metadata describes original preparation.

## Incremental cost and timing

| Component | New generation attempts | Incremental API cost (USD) | New summed count/generation latency (seconds) | Unknown-cost attempts |
| --- | ---: | ---: | ---: | ---: |
| Nomic top three | 45 | $0.04129774 | 355.74 | 0 |
| Full eligible history | 46 | $0.08363613 | 581.08 | 0 |
| Shared AI evaluation | 111 | $0.78273244 | 7997.78 | 5 |
| All | 202 | $0.90766631 | 8934.61 | 5 |

| Evidence accounting | Generation attempts | Known API cost (USD) | Summed count/generation latency (seconds) | Unknown-cost attempts |
| --- | ---: | ---: | ---: | ---: |
| Original historical experiment | 3079 | $10.84758089 | 97522.35 | 5 |
| Historical evidence retained in update | 3014 | $10.66421853 | 95973.73 | 5 |
| New incremental evidence | 202 | $0.90766631 | 8934.61 | 5 |
| Retained plus new evidence | 3216 | $11.57188484 | 104908.34 | 10 |

Historical costs are not incremental spending. Costs use the unchanged frozen pricing schedule and recorded provider usage. Missing usage is unknown, not zero, so totals with unknown-cost attempts remain incomplete. Summed request durations are not wall-clock time because requests run concurrently; they exclude queueing and checkpoint writes. Local CPU compute has no assigned dollar cost.

| Partition | Method | Mixed-observation runtime API cost (USD) | Mean runtime latency per week (seconds) |
| --- | --- | ---: | ---: |
| Development | Nomic top three | $0.47699220 | 11.49 |
| Development | Full eligible history | $0.84474262 | 17.26 |
| Qualified final | Nomic top three | $0.11165786 | 10.04 |
| Qualified final | Full eligible history | $0.17811469 | 13.14 |

These all-case runtime measurements combine retained historical requests with new execution, local selection, and the allocated Nomic preparation/ranking overhead. Their paired cost and latency intervals remain in the consolidated record, but this update is not a wholly fresh latency benchmark. Nomic runs before full history, and provider load and cache conditions may differ between observations.

## Evaluator consistency and evidence limits

The original 20-Persona consistency sample is unchanged. Raw source-decision pairwise agreement is 341/384 (88.80%); 21 of 128 source coordinates do not agree across all three assessments. Exact-quotation acceptance agreement is 107/109 (98.17%), with 1 disagreement and 1 incomplete coordinate among 37 quotations. Derived opportunity, priority, and card-grade consistency remains a diagnostic because the primary pass is adjudicated and repeats are not.

This is AI assessment of synthetic writing. It does not establish human validity, real-user benefit, an untouched final test, or deployment approval. The original qualified-final exposure limitation still applies: final histories were held apart from recorded NSM semantic development, but had prior upstream research exposure. The v4 Weekly Drift comparison also used the broader synthetic development corpus; its selected Run 1 is an adopted input update, not new holdout isolation. The update measures NSM outcomes under those inputs and does not establish Weekly Drift accuracy or isolate the effects of motivation prose. Saved-card export and the final application walkthrough remain separate work.

## Reproduction and verification

From the repository root, activate the environment first. Audit, verification, and regrading make no provider calls. Preparation reuses the frozen prepared record; the original encoding used cached local model files with offline flags. Execution was explicitly authorized for these synthetic inputs at `api.openai.com`, with concurrency eight. The earlier approval-review blocks occurred before any provider request and are retained as resolved history in the new record.

```sh
source .venv/bin/activate
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.nsm_targeted_update audit
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.nsm_targeted_update prepare
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.nsm_targeted_update verify
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.nsm_targeted_update run --concurrency 8
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.experiments.nsm_targeted_update report
```

Validation passed 441 relevant experiment, North Star Moment, and demo tests, including 47 targeted-update tests, plus Ruff, scoped MyPy, and independent quality review. Historical case-building tests reconstruct a temporary hash-verified source tree using the original reviewer file from its frozen Git revision; they do not alter production compatibility checks or frozen hashes. Unrestricted MyPy import traversal encountered pre-existing errors outside the new module; the scoped check passed. The full repository suite and browser checks were not run. This follow-up changes no application behavior.

Verification commands (after environment activation):

```sh
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync pytest tests/north_star tests/demo/test_north_star_service.py tests/demo/test_north_star_scenarios.py tests/evals/test_nsm_targeted_update.py tests/evals/test_nsm_experiment.py tests/evals/test_nsm_cases.py tests/evals/test_nsm_evaluation.py tests/evals/test_nsm_provider.py tests/evals/test_nsm_reporting_corrections.py
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync ruff check scripts/experiments/nsm_targeted_update.py tests/evals/test_nsm_targeted_update.py tests/evals/test_nsm_cases.py
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --with 'mypy==2.3.0' mypy --follow-imports=silent scripts/experiments/nsm_targeted_update.py
git diff --check
```

Independent final review reproduced all primary and repeat grades, consistency results, aggregate metrics, and bootstrap intervals exactly; it reconciled retained receipts, paired exclusions, complete request inputs, and per-attempt frozen-price calculations. All 212 local file links across the seven updated prose documents resolve.

## Affected cases

| Case | Partition | Previous state | v4 Run 1 state |
| --- | --- | --- | --- |
| `02fb94f3:week:2025-04-14` | development | insufficient_evidence | no_active_drift |
| `02fb94f3:week:2025-04-28` | development | active_drift | no_active_drift |
| `0e9a45e9:week:2026-01-12` | development | insufficient_evidence | no_active_drift |
| `0e9a45e9:week:2026-01-19` | development | insufficient_evidence | no_active_drift |
| `110fcd4f:week:2025-06-23` | development | active_drift | no_active_drift |
| `11de77e8:week:2025-09-15` | development | active_drift | no_active_drift |
| `11de77e8:week:2025-10-13` | development | insufficient_evidence | no_active_drift |
| `22b117be:week:2025-05-19` | development | insufficient_evidence | no_active_drift |
| `22b117be:week:2025-05-26` | development | insufficient_evidence | no_active_drift |
| `3a3b15e4:week:2025-06-09` | development | active_drift | no_active_drift |
| `43f4507b:week:2025-01-13` | development | insufficient_evidence | no_active_drift |
| `43f4507b:week:2025-01-20` | development | insufficient_evidence | no_active_drift |
| `43f4507b:week:2025-01-27` | development | insufficient_evidence | no_active_drift |
| `5c8ad0db:week:2025-05-26` | development | insufficient_evidence | no_active_drift |
| `5c8ad0db:week:2025-06-02` | development | insufficient_evidence | no_active_drift |
| `621be543:week:2025-09-22` | development | insufficient_evidence | no_active_drift |
| `742c98d6:week:2025-10-06` | development | insufficient_evidence | no_active_drift |
| `7c712a0a:week:2025-12-15` | development | insufficient_evidence | no_active_drift |
| `7cc5cf92:week:2025-07-14` | development | insufficient_evidence | no_active_drift |
| `87e92805:week:2025-03-17` | development | no_active_drift | insufficient_evidence |
| `87e92805:week:2025-03-31` | development | insufficient_evidence | no_active_drift |
| `87e92805:week:2025-04-07` | development | insufficient_evidence | no_active_drift |
| `87e92805:week:2025-04-14` | development | active_drift | insufficient_evidence |
| `8f83c818:week:2025-06-23` | development | no_active_drift | insufficient_evidence |
| `8f83c818:week:2025-06-30` | development | active_drift | insufficient_evidence |
| `988d1a65:week:2025-03-10` | development | active_drift | insufficient_evidence |
| `a24b8d8f:week:2025-07-07` | final | insufficient_evidence | no_active_drift |
| `abf1ce49:week:2025-09-08` | final | active_drift | insufficient_evidence |
| `abf1ce49:week:2025-09-15` | final | active_drift | insufficient_evidence |
| `ad378991:week:2025-04-21` | final | insufficient_evidence | no_active_drift |
| `ad378991:week:2025-04-28` | final | insufficient_evidence | no_active_drift |
| `d7a0683f:week:2025-12-29` | final | insufficient_evidence | active_drift |
| `d7a0683f:week:2026-01-05` | final | insufficient_evidence | active_drift |
| `d7a0683f:week:2026-02-02` | final | active_drift | insufficient_evidence |
| `dbe2c53d:week:2025-03-03` | development | insufficient_evidence | no_active_drift |
| `ed67c9cc:week:2025-02-03` | development | active_drift | active_drift |
| `ed67c9cc:week:2025-02-10` | development | insufficient_evidence | no_active_drift |
| `f6180c27:week:2025-07-21` | final | insufficient_evidence | active_drift |
