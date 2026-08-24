# Deployed Persona Coach Digest Sample

- Source: each public scenario bundle's stored `weekly_digest_built` output
- Weekly Drift Reviewer calls: 0
- Coach Digest prompt: `weekly_digest_coach` v4.1
- Model: `gpt-5.6-luna`
- Reasoning effort: `none`
- Accepted responses: 5
- Paid generation calls: 7
- Validation-guided retries: 2
- Input tokens: 9626
- Cached input tokens: 0
- Output tokens: 1198
- Calculated published-rate cost: `$0.00379690`
- Total request latency: 22.665s
- Cost basis: response token usage and published standard-tier Luna rates; not a billing export
- Prompt tuning after final scores: none

## Command

`.venv/bin/python scripts/coach/generate_approved_judge_sample.py --personas 11de77e8 23d101f8 8f83c818 988d1a65 02fb94f3 --reuse-scenario-key-weeks --execute`

## Responses

| Scenario | Persona | Week | Attempts | Response hash | Generated response | Public bundle |
| --- | --- | --- | ---: | --- | --- | --- |
| two-values-lukas | 11de77e8 | 2025-10-13 to 2025-10-19 | 1 | `e1ae4538af5bd7096c2fd339925b0e50ffcf965aceef6c387a5360bbc2e85586` | `logs/experiments/reports/coach_digest_sample_20260824/generated_responses/11de77e8_2025-10-19.json` | `frontend/onboarding/public/scenarios/two-values-lukas.json` |
| stable-meera | 23d101f8 | 2025-09-15 to 2025-09-21 | 2 | `ffdb7c31376af4a51b439206d2b1217ad71ce46d4186dc03aff22d84ed961046` | `logs/experiments/reports/coach_digest_sample_20260824/generated_responses/23d101f8_2025-09-21.json` | `frontend/onboarding/public/scenarios/stable-meera.json` |
| active-wei-jun | 8f83c818 | 2025-06-30 to 2025-07-06 | 1 | `0b7bfcb04e530026944f443f7cb375ea807f58ca5f9e4e3212530096c07e0bec` | `logs/experiments/reports/coach_digest_sample_20260824/generated_responses/8f83c818_2025-07-06.json` | `frontend/onboarding/public/scenarios/active-wei-jun.json` |
| recovered-marc | 988d1a65 | 2025-03-17 to 2025-03-23 | 1 | `793424db0f1a16075b83861e0f78842f57564662264b7a22dcc979862d3168f3` | `logs/experiments/reports/coach_digest_sample_20260824/generated_responses/988d1a65_2025-03-23.json` | `frontend/onboarding/public/scenarios/recovered-marc.json` |
| uncertain-noor | 02fb94f3 | 2025-04-14 to 2025-04-20 | 2 | `668686b6542c7479695e4cd99cadfd0f6e7a87ad2ccfc3e7e05a69a3a0a6fe5b` | `logs/experiments/reports/coach_digest_sample_20260824/generated_responses/02fb94f3_2025-04-20.json` | `frontend/onboarding/public/scenarios/uncertain-noor.json` |

## Failed Attempts and Review
- `23d101f8`: coach_validation; raw output preserved at `logs/experiments/reports/coach_digest_sample_20260824/generation_diagnostics/23d101f8_2025-09-21.20260824T142754878766Z.coach_diagnostic.json`.
- `02fb94f3`: coach_validation; raw output preserved at `logs/experiments/reports/coach_digest_sample_20260824/generation_diagnostics/02fb94f3_2025-04-20.20260824T142807932310Z.coach_diagnostic.json`.
