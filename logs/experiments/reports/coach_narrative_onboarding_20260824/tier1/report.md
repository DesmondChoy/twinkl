# Coach Digest Validations — Batch Report

**Source:** mechanical code checks (not human validation). Surface properties only.

- Parquet: `logs/exports/weekly_digests/coach_onboarding_eval_20260824.parquet`
- Signal source filter: `weekly_drift_reviewer` (5 of 5 rows)
- With narrative: 5
- Evaluated: 5

| Check | Passed | Total | Pass rate | Target | Meets target |
| --- | --- | --- | --- | --- | --- |
| groundedness | 4 | 5 | 80% | > 70% | ✅ |
| non_circularity | 5 | 5 | 100% | > 95% | ✅ |
| value_leakage | 5 | 5 | 100% | — | — |
| length | 4 | 5 | 80% | > 90% | ❌ |
