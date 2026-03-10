# twinkl-691.4

## Checklist

- [x] Implement config-gated circumplex regularizer for `BalancedSoftmax`
- [x] Wire notebook config overrides and experiment logger metadata
- [x] Add `twinkl_691_4` experiment configs
- [x] Extend targeted tests for loss and logger behavior
- [x] Run targeted tests
- [x] Run seed 11 smoke ablation
- [x] Run seeds 22 and 33 ablations
- [x] Write `twinkl-691.4` experiment report with keep/drop recommendation
- [x] Close beads issue with results

## Review

- Added a probability-space circumplex regularizer to `BalancedSoftmax`, wired it
  only through the ordinal notebook driver, and logged the ablation distinctly as
  `balanced_softmax_circreg`.
- Targeted tests passed for the loss and logger changes.
- Three ablation runs completed successfully as `run_028`-`run_030`.
- Result: drop the regularizer. Circumplex structure improved, but `recall_-1`,
  minority recall, and hedging regressed relative to the post-lift control.

# twinkl-715

## Checklist

- [x] Add config-driven `recall_-1` guardrail to ordinal checkpoint selection
- [x] Wire the v4 notebook path and script mirror to the shared guardrail helper
- [x] Persist selection summary and per-epoch selection trace artifacts
- [x] Add `twinkl_715` frozen-holdout experiment configs
- [x] Extend targeted tests for selection eligibility and logging
- [x] Run targeted tests
- [x] Run seed 11 smoke rerun
- [x] Run seeds 22 and 33 reruns
- [x] Write `twinkl-715` experiment report with guarded-selection outcome
- [x] Close beads issue with results

## Review

- Added a config-driven hard validation `recall_-1` floor to ordinal checkpoint
  eligibility, with explicit `recall_minus1_below_floor` reasons and a
  testable finalization helper for promotable vs debug-only outcomes.
- Wired the guardrail through the v4 notebook and its checked-in script mirror,
  and persisted `selection_summary.yaml` plus `selection_trace.parquet` in each
  model artifact directory.
- Added the `twinkl_715` frozen-holdout config family and reran the motivating
  regularized `BalancedSoftmax` branch as `run_031`-`run_033`.
- Result: keep the new guardrail, but do not promote the regularized family.
  The guardrail blocked the previously selected low-recall epochs cleanly, yet
  the rerun still did not beat the incumbent default on holdout
  `recall_-1`/minority recall/hedging.

# twinkl-716

## Checklist

- [x] Read and execute `.claude/skills/experiment-review/SKILL.md`
- [x] Audit `logs/experiments/index.md` and all `logs/experiments/runs/*.yaml`
- [x] Backfill missing `provenance.rationale` / `observations` fields only where blank or placeholder
- [x] Run supporting analyses for frontier comparison, hard-dimension error slices, and data-distribution checks
- [x] Refresh split-aware frontier sections in `logs/experiments/index.md`
- [x] Verify changed files and close beads issue with results

## Review

- Backfilled the only missing run metadata fields in `run_028`-`run_033`,
  leaving all previously substantive YAML narrative untouched.
- Added a fresh full frontier report at
  `logs/experiments/reports/experiment_review_2026-03-10_v8.md` covering all
  runs through `run_033`, including checkpoint replay on `run_020` for
  hedonism/security error analysis and updated recommendations.
- Refreshed `logs/experiments/index.md` so the current frontier links to the
  v8 report, includes the regularized follow-up families on the corrected-split
  board, and records the new v8 finding while keeping the incumbent
  `run_019`-`run_021` default unchanged.
- Verification: reran the missing-field scan (`missing_count 0`) and confirmed
  the index contains the required split-aware sections plus the new v8 report
  references.
