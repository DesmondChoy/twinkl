# Weekly Digest Coach Flow

## Plan

- [x] Inspect the existing weekly digest, Coach prompt, and evaluation docs.
- [x] Expand the weekly digest schema to carry response mode, full journal context, coach output, and validation results.
- [x] Clean up focus-dimension and evidence selection so the digest is more coach-usable.
- [x] Add structured Coach narrative generation helpers and Tier 1 validation checks.
- [x] Persist weekly digests to a consolidated parquet artifact in addition to per-run exports.
- [x] Add or update targeted tests for the weekly Coach slice.
- [x] Update `docs/weekly/weekly_digest_generation.md` with current implementation and remaining gaps.
- [x] Stop overlapping tension/strength dimensions in the same digest by default.
- [x] Truncate journal history to entries on or before the digest end date.
- [x] Add a conservative acute-distress heuristic so grief-heavy weeks route to `high_uncertainty` instead of surfacing misleading tensions like `Hedonism`.
- [x] Add `mixed_state` and `background_strain` fallback modes for weeks that are neither cleanly stable nor clearly in rut.
- [x] Move fallback mode inference into a dedicated helper module to keep weekly digest assembly readable.
- [x] Refactor the digest contract so upstream drift output is the primary mode source and local mode inference is clearly fallback-only.

## Review

- Implemented a fuller weekly Coach vertical slice in `src/coach/`.
- Added prompt/schema support for structured reflective Coach narratives.
- Added Tier 1 validation and consolidated parquet persistence.
- Verified with `pytest tests/coach/test_weekly_digest.py`.
- Tightened digest semantics so backfilled digests are time-safe, focus dimensions do not overlap by default, and acute grief/distress weeks fall back to a presence-oriented mode.
- Added a dedicated mode-logic helper module and new fallback states so nuanced weeks can be framed as mixed or quietly strained instead of being flattened to either `stable` or `None clear this week`.
- Regenerated the sample markdowns and confirmed that Chen Meiling now routes to `background_strain`, while Fatima Al-Hassan remains on the safer `high_uncertainty` path.
- Updated the weekly digest contract so it can take a structured upstream drift result directly; in-module mode logic now exists only as development scaffolding until the Drift Detection Engine is built.
