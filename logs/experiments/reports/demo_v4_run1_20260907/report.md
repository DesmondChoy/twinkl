# Deployed Persona Coach Digest Sample

- Source: each public scenario bundle's stored `weekly_digest_built` output
- Weekly Drift Reviewer calls: 0
- Coach Digest prompt: `weekly_digest_coach` v4.2
- Model: `gpt-5.6-luna`
- Reasoning effort: `none`
- Accepted responses: 5
- Paid generation calls: 7
- Validation-guided retries: 2
- Input tokens: 10980
- Cached input tokens: 0
- Output tokens: 1335
- Calculated published-rate cost: `$0.00434595`
- Total request latency: 25.383s
- Cost basis: response token usage and published standard-tier Luna rates; not a billing export
- Prompt tuning after final scores: none

## Command

`TWINKL_COACH_PROVIDER=openai TWINKL_COACH_MODEL=gpt-5.6-luna uv run python scripts/coach/generate_approved_judge_sample.py --personas 02fb94f3 5fa8b540 ed67c9cc 8f83c818 2d928d8a --reuse-scenario-key-weeks --manifest-out logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json --parquet-path logs/experiments/reports/demo_v4_run1_20260907/weekly_digests.parquet --execute (executed in a Python process wrapper setting AsyncOpenAI max_retries=0; see execution.json)`

## Responses

| Scenario | Persona | Week | Attempts | Response hash | Generated response | Public bundle |
| --- | --- | --- | ---: | --- | --- | --- |
| stable-noor | 02fb94f3 | 2025-05-19 to 2025-05-25 | 2 | `befe5b3c9ba08769f00cbd068850956657f349d8a98cda3d341c8a03cefa5ac9` | `logs/experiments/reports/demo_v4_run1_20260907/generated_responses/02fb94f3_2025-05-25.json` | `frontend/onboarding/public/scenarios/stable-noor.json` |
| active-nisha | 5fa8b540 | 2025-03-03 to 2025-03-09 | 1 | `b179fb0754fe74d9342c33f9263637abd07f79a42f8d09ce40d0b5f5b7225572` | `logs/experiments/reports/demo_v4_run1_20260907/generated_responses/5fa8b540_2025-03-09.json` | `frontend/onboarding/public/scenarios/active-nisha.json` |
| ended-sook-yin | ed67c9cc | 2025-02-10 to 2025-02-16 | 1 | `2101f4f7f2742c49579041a540b13bafd41dd7a85e1b2a518d82b7400a39f928` | `logs/experiments/reports/demo_v4_run1_20260907/generated_responses/ed67c9cc_2025-02-16.json` | `frontend/onboarding/public/scenarios/ended-sook-yin.json` |
| uncertain-wei-jun | 8f83c818 | 2025-06-30 to 2025-07-06 | 2 | `342bea4f2464d7a8bf8d042a782cbe83e5dc56640f9c09338d1eeb0d4629b504` | `logs/experiments/reports/demo_v4_run1_20260907/generated_responses/8f83c818_2025-07-06.json` | `frontend/onboarding/public/scenarios/uncertain-wei-jun.json` |
| two-values-henrik | 2d928d8a | 2025-02-17 to 2025-02-23 | 1 | `50b75e84764b1475b379adf7591b0b0e95bed300fe911a10b83c6e3798d46346` | `logs/experiments/reports/demo_v4_run1_20260907/generated_responses/2d928d8a_2025-02-23.json` | `frontend/onboarding/public/scenarios/two-values-henrik.json` |

## Failed Attempts and Review
- `02fb94f3`: coach_validation; raw output preserved at `logs/experiments/reports/demo_v4_run1_20260907/generation_diagnostics/02fb94f3_2025-05-25.20260907T123902218331Z.coach_diagnostic.json`.
- `8f83c818`: coach_validation; raw output preserved at `logs/experiments/reports/demo_v4_run1_20260907/generation_diagnostics/8f83c818_2025-07-06.20260907T123916365076Z.coach_diagnostic.json`.
