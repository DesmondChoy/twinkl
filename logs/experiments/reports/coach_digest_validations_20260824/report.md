# Coach Digest Validations — Batch Report

**Source:** mechanical code checks (not human validation). Surface properties only.

- Input: `logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json`
- Input kind: `public_scenario_manifest`
- Signal source filter: `weekly_drift_reviewer` (5 of 5 rows)
- With narrative: 5
- Evaluated: 5
- API calls: 0
- Provider cost: $0.00
- Provider request latency: not applicable

| Check | Passed | Total | Pass rate | Target | Meets target |
| --- | --- | --- | --- | --- | --- |
| groundedness | 5 | 5 | 100% | > 70% | ✅ |
| non_circularity | 5 | 5 | 100% | > 95% | ✅ |
| value_leakage | 5 | 5 | 100% | — | — |
| state_claims | 5 | 5 | 100% | — | — |
| length | 5 | 5 | 100% | > 90% | ✅ |

## Per-Response Results

| Response | Scenario | Result | Failed checks |
| --- | --- | --- | --- |
| 11de77e8:2025-10-19 | two-values-lukas | evaluated | none |
| 23d101f8:2025-09-21 | stable-meera | evaluated | none |
| 8f83c818:2025-07-06 | active-wei-jun | evaluated | none |
| 988d1a65:2025-03-23 | recovered-marc | evaluated | none |
| 02fb94f3:2025-04-20 | uncertain-noor | evaluated | none |
