# Weekly Digest Coach Flow

## Plan

- [x] Inspect the existing weekly digest, Coach prompt, and evaluation docs.
- [x] Expand the weekly digest schema to carry response mode, full journal context, coach output, and validation results.
- [x] Clean up focus-dimension and evidence selection so the digest is more coach-usable.
- [x] Add structured Coach narrative generation helpers and Tier 1 validation checks.
- [x] Persist weekly digests to a consolidated parquet artifact in addition to per-run exports.
- [x] Add or update targeted tests for the weekly Coach slice.
- [x] Update `docs/weekly/weekly_digest_generation.md` with current implementation and remaining gaps.

## Review

- Implemented a fuller weekly Coach vertical slice in `src/coach/`.
- Added prompt/schema support for structured reflective Coach narratives.
- Added Tier 1 validation and consolidated parquet persistence.
- Verified with `pytest tests/coach/test_weekly_digest.py`.
