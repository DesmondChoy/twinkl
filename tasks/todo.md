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

# twinkl-691.5

## Checklist

- [x] Re-read the post-lift, regularizer, and guardrail reports
- [x] Confirm whether a sampler hook already exists in the active ordinal path
- [x] Write a `twinkl-691.5` de-scope report grounded in March 9-10 results
- [x] Record the closeout recommendation in the experiment index
- [x] Close beads issue with de-scope reason

## Review

- Confirmed the active ordinal frontier still uses plain shuffled train
  loaders, with no existing sampler abstraction in the notebook review path or
  shared dataloader helper.
- Wrote
  `logs/experiments/reports/experiment_review_2026-03-11_twinkl_691_5.md`
  documenting why the circumplex-aware batch sampler is being explicitly
  de-scoped rather than implemented now.
- Decision: drop the sampler for this rollout. The regularizer improved
  circumplex structure, but the guarded rerun still lagged the incumbent
  `run_019`-`run_021` frontier on `recall_-1`, minority recall, and hedging,
  so the next better levers are per-dimension weighting and post-hoc
  `BalancedSoftmax` retargeting instead.

# twinkl-719.2

## Checklist

- [x] Add frontier-default dimension-weighting defaults to `config/vif.yaml`
- [x] Implement BalancedSoftmax per-dimension CE/EMA weighting helpers
- [x] Wire notebook config, training-loop weighting, and trace artifacts
- [x] Extend experiment logger metadata and loss shorthands
- [x] Add targeted tests for loss, logging, and trace helpers
- [x] Run focused pytest suites
- [x] Review changed files and summarize results

## Review

- Added explicit BalancedSoftmax helpers for per-dimension CE extraction,
  inverse-loss EMA weighting, and weighting-config validation, while keeping
  the low-level loss unweighted unless a dimension-weight vector is passed in.
- Wired the frontier notebook/script mirror to read default-on weighting
  settings from `config/vif.yaml`, apply uniform weights for epoch 0, update
  EMA-smoothed weights after each epoch, and persist both an expanded
  `selection_trace.parquet` and a new `dimension_weight_trace.parquet`.
- Persisted weighting metadata and distinct loss shorthands in experiment run
  YAMLs so weighted BalancedSoftmax runs can be recognized and reproduced from
  artifacts alone.
- Verification: `python -m py_compile` on the touched modules passed, and
  focused pytest passed for `tests/vif/test_losses.py`,
  `tests/vif/test_experiment_logger.py`, and
  `tests/vif/test_training_traces.py` (`100 passed`).

# twinkl-719.3

## Checklist

- [x] Add weighted frontier rerun configs for seeds `11` / `22` / `33`
- [x] Run seed `11` smoke rerun and verify weighted artifacts/logging
- [x] Run seeds `22` and `33` reruns
- [x] Compare weighted family against incumbent and circumplex branches
- [x] Write `twinkl-719.3` experiment report with keep/drop recommendation
- [x] Update `logs/experiments/index.md` current-frontier narrative
- [x] Close beads issue with results

## Review

- Added the `twinkl_719_3` family manifest plus three flat notebook override
  YAMLs for the weighted `BalancedSoftmax` rerun, pinning the frozen holdout,
  guarded selection policy, incumbent LR override, and explicit
  `dimension_weighting_*` settings.
- Ran the full 3-seed family through the notebook driver as
  `run_034`-`run_036`, and verified that each run logged
  `balanced_softmax_dimweight` plus the new `dimension_weight_trace.parquet`
  artifact.
- Recomputed the incumbent `run_019`-`run_021` circumplex summaries from saved
  selected-test artifacts, then compared the weighted family against the
  incumbent, post-lift, and circumplex branches in
  `logs/experiments/reports/experiment_review_2026-03-11_twinkl_719_3.md`.
- Result: keep weighted `BalancedSoftmax` as the strongest tail-sensitive
  reference branch, but do not promote it over `run_019`-`run_021` as the
  default. The weighted family materially improved `recall_-1`, minority
  recall, hedging, and calibration, but its median QWK regressed and the hard
  `hedonism` / `security` story still did not beat the incumbent.
- Updated `logs/experiments/index.md`, backfilled rationale/observations in the
  new run YAMLs, closed `twinkl-719.3`, and unblocked `twinkl-719.4`.

# twinkl-721

## Checklist

- [x] Read and execute `.claude/skills/experiment-review/SKILL.md`
- [x] Audit `logs/experiments/index.md`, all `logs/experiments/runs/*.yaml`,
  and frontier artifacts for the current split-aware board
- [x] Backfill any empty `provenance.rationale` / `observations` fields only
  where blank or placeholder
- [x] Run supporting analyses for checkpoint selection, dimension weighting,
  hard-dimension error slices, dataset label distributions, and corroborating
  literature
- [x] Write a refreshed experiment review report and update
  `logs/experiments/index.md` if the audit changes the frontier narrative
- [x] Verify changed files and close beads issue with results

## Review

- Confirmed the run manifests and split-aware experiment index were already
  internally consistent: no run YAMLs needed `provenance.rationale` or
  `observations` backfill, and the frontier medians in
  `logs/experiments/index.md` matched the run YAML metrics for `run_016`-`run_036`.
- Wrote `logs/experiments/reports/experiment_review_2026-03-11_twinkl_721.md`,
  a fresh full-frontier audit covering all runs through `run_036` with
  artifact-backed circumplex summaries, checkpoint-selection review,
  dimension-weighting analysis, validation error slices from `run_036`, label
  distribution checks, and literature-backed recommendations.
- Refreshed `logs/experiments/index.md` so the latest full-frontier link points
  at the new v9 review and the findings section records the sharper conclusion:
  weighted `BalancedSoftmax` remains the best tail-sensitive reference branch,
  but inverse-loss weighting mostly amplified easy low-CE heads instead of the
  hard `hedonism` / `security` dimensions.
- Verification: reran the missing-field scan (`0` missing), rebuilt the frozen
  holdout from `run_036`'s checkpoint for qualitative error analysis,
  recomputed circumplex summaries from saved test artifacts for incumbent vs
  weighted branches, and inspected the updated report/index content before
  closeout.
