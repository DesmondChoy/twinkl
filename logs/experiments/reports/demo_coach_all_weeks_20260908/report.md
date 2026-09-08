# Coach Digest completion for every saved replay week

**8 September 2026; `twinkl-rklc.38`.** All 27 saved replay weeks now have a source-compatible Coach Digest. The prior export contained five responses, one curated key week per Persona. The other 22 weeks had complete Weekly Drift Detection inputs but no saved Coach response; this was missing generated content rather than a frontend projection or stale-hash problem.

| Persona | Existing responses retained | Missing responses generated | Completed weeks |
| --- | ---: | ---: | ---: |
| Noor | 1 | 5 | 6 |
| Nisha | 1 | 4 | 5 |
| Sook Yin | 1 | 3 | 4 |
| Wei Jun | 1 | 5 | 6 |
| Henrik | 1 | 5 | 6 |
| Total | 5 | 22 | 27 |

## Generation and provenance

Generation reused the exact `weekly_digest_built` inputs from catalog-hash-verified public scenario bundles. It made no Weekly Drift Reviewer or North Star Moment provider calls. Each missing response used OpenAI `gpt-5.6-luna`, reasoning `none`, prompt `weekly_digest_coach` version `4.2`, service tier `default`, and the existing Coach Digest diagnostic/validation implementation. Responses were merged by scenario and week; all five existing narratives and generation receipts remain exactly unchanged.

There were **26 provider calls**: 22 accepted responses and four rejected drafts followed by the single permitted validation-guided retry. Retries occurred for Noor weeks 1 and 5, Nisha week 2, and Sook Yin week 4. The rejected drafts failed groundedness checks. SDK automatic retries were disabled (`max_retries=0`); the maximum was two attempts per case, and no extra retries were made. All accepted responses passed Coach Digest Validations. This generation did not run Coach Digest Evals or human review.

Recorded usage totals 39,221 input tokens and 4,939 output tokens. All 26 attempts have usage receipts. The configured published-rate calculation is **US$0.01558455**, not a billing export. Summed request latency is 96.15 seconds. These are incremental Coach-generation figures; they exclude the five retained responses and all prior NSM/Weekly Drift experiments.

The [frozen plan](plan.json) preserves all 27 digest inputs, original bundle/event provenance, the five retained receipts, and generation settings. `inputs/`, `diagnostics/`, `responses/`, and `cases/` contain the 22 prepared inputs, all 26 exact prompts/raw outputs/metrics/validation results, accepted enriched digests, and durable per-case checkpoints. [Summary](summary.json) records usage and completion. [Verification](verification.json) records final implementation hashes, per-Persona counts, and preservation checks. [Composite review](composite_review.md) renders all 27 Coach narratives alongside their unchanged NSM quotations and original reflective questions for editorial review.

A post-run review added the complete prompt-template hash to the resume policy; the initial policy already recorded prompt metadata and implementation hashes. The tracked template was unchanged during generation, and every exact assembled prompt was recorded at execution. `plan.json` discloses this correction. Final implementation hashes describe the reviewed runner, including this added resume check and a type annotation correction, rather than claiming its final source bytes were frozen before generation.

## Implementation and reproduction

The scenario validator now checks the matching response and event for each week instead of enforcing one key-week Coach event per Persona. New non-key-week Coach events use deterministic `scenario:coach:week-start` IDs without advancing the original numbered event sequence. The original key-week Coach slot remains reserved. This preserves every original trace ID and ensures all 27 current digest event IDs still match their frozen generation-source event IDs; adding a response does not rewrite provider provenance. It still rejects incompatible source inputs. The completion runner refuses to replace existing responses, checks frozen inputs and policy on resume, saves an attempt checkpoint before transport, and recovers accepted saved diagnostics without another call. An interrupted attempt without a receipt stops for inspection rather than silently repeating an unknown paid call. Terminal failures retain their attempt history and do not gain retries on resume.

From the repository root:

```sh
source .venv/bin/activate
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.coach.complete_scenario_coach
# Paid generation was explicitly authorized for this run:
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.coach.complete_scenario_coach --execute
# Offline export makes no provider calls:
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python -m scripts.export_demo_experiments
```

The completed runner reused all saved case receipts in an offline replay with an injected completion function that rejects any provider invocation. Future paid execution requires a separate authorization when outside an already authorized task. The generator reads `OPENAI_API_KEY` through the existing environment setup and never stores credentials in these artifacts.

## Verification

All **48 focused tests** passed: 23 saved-scenario tests, 10 NSM scenario tests, 10 existing Coach sample generator tests, and five new retry/resume/preservation tests. Ruff and scoped MyPy passed. The offline exporter produced 27 Coach events across the five scenarios; every response matches its current Weekly Drift input hash. All 27 current digest IDs match their frozen generation-source IDs, every original trace ID is retained, all five original Coach receipt objects are unchanged, and `src/demo/north_star_replay_records.json` is byte-identical to its pre-task Git version. No NSM evidence was regenerated or rewritten. After the stable-event-ID correction, all 33 scenario and NSM scenario tests were rerun successfully, together with Ruff and scoped MyPy.

```sh
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync pytest tests/demo/test_scenarios.py tests/demo/test_north_star_scenarios.py tests/coach/test_generate_approved_judge_sample.py tests/coach/test_complete_scenario_coach.py -q
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync ruff check scripts/coach/complete_scenario_coach.py src/demo/scenarios.py tests/demo/test_scenarios.py tests/coach/test_complete_scenario_coach.py
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --with 'mypy==2.3.0' mypy --follow-imports=silent scripts/coach/complete_scenario_coach.py src/demo/scenarios.py
```

This report covers saved Coach completion. Wider application tests, browser verification, and integrated frontend prose review are recorded by the surrounding integration task. The synthetic outputs and deterministic validations do not establish human validity or user benefit.
