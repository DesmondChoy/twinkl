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

# twinkl-722

## Checklist

- [x] Re-read the peer critique against `twinkl-721` and confirm which points
  are supported by the repo and current literature
- [x] Revise `logs/experiments/reports/experiment_review_2026-03-11_twinkl_721.md`
  to fix literature fit, weighting diagnosis, and recommendation framing
- [x] Update `logs/experiments/index.md` if the latest frontier note should
  reflect the revised guidance
- [x] Update `tasks/lessons.md` with the correction pattern and verify the
  edited docs
- [x] Close beads issue with results

# twinkl-723

## Checklist

- [x] Read `docs/prd.md` and current project lessons for context
- [x] Audit the experiment logging and notebook training paths for loss/history persistence
- [x] Inspect representative `logs/experiments` run YAMLs, artifacts, reports, and notebooks
- [x] Summarize whether training-loss curves are used in current diagnosis and how they are surfaced

## Review

- Confirmed that notebook review code still plots training-loss and LR-finder
  curves interactively, but those visuals are not promoted into
  `logs/experiments/index.md` or the run YAML summaries.
- Current experiment logging records only compact training dynamics in run YAMLs
  (`best_epoch`, `train_loss_at_best`, `val_loss_at_best`, `gap_at_best`,
  `final_lr`) plus artifact links when available.
- Newer corrected-split `BalancedSoftmax` runs add per-epoch
  `selection_trace.parquet`, and the most recent weighted runs also add
  `dimension_weight_trace.parquet`, which makes curve reconstruction possible
  for those runs without persisting notebook figures.
- Older runs mostly lack per-epoch trace artifacts, so historical diagnosis in
  reports relies on scalar summaries and output artifacts rather than full
  training-loss curves.

# twinkl-723

## Checklist

- [x] Read `docs/prd.md` and current lessons for project context
- [x] Create a beads issue for the training-diagnostics audit
- [x] Inspect the current VIF training/logger code paths for per-epoch history capture
- [x] Inspect representative `logs/experiments` run manifests and artifact directories
- [x] Check whether experiment reviews/index currently surface or consume loss-curve diagnostics
- [x] Summarize whether prior notebook loss curves are used today and whether they would still be helpful

## Review

- Confirmed the shared training code still captures per-epoch loss histories, and
  the notebook review path now also persists per-epoch checkpoint-selection and
  dimension-weight traces as parquet artifacts.
- Confirmed the current experiment index and run YAMLs surface only compressed
  training-dynamics summaries such as `best_epoch`, `train_loss_at_best`,
  `val_loss_at_best`, and `gap_at_best`, not rendered training-loss figures.
- Confirmed training/loss plots still exist in notebook-only analysis cells, but
  they are not persisted into `logs/experiments` as PNG/SVG assets and are not a
  first-class part of the current review loop.
- Confirmed LR-finder history remains partially surfaced through JSON snapshots
  and past reviews, while notebook PNG LR plots are explicitly deleted in the v4
  review script.

## Review

- Revised `logs/experiments/reports/experiment_review_2026-03-11_twinkl_721.md`
  so the weighting diagnosis now explicitly says inverse-loss weighting was
  optimizing the wrong proxy, not merely the wrong dimensions, and the main
  recommendation now points at the repo's existing Menon-style validation-only
  logit-adjustment path rather than weak-fit literature analogies.
- Updated the latest `twinkl-721` findings note in `logs/experiments/index.md`
  to keep the frontier narrative consistent with the revised report: same
  frontier conclusion, sharper explanation, and cleaner separation between
  boundary-shift follow-ups and calibration-only follow-ups.
- Captured the correction pattern in `tasks/lessons.md`: in experiment reviews,
  keep intervention recommendations anchored to the repo's real tooling/open
  issues and only cite external work that genuinely matches the class count,
  training regime, and architecture.
- Verification: re-read the edited report and index sections, confirmed the old
  `LORT` / `LIFT` framing is gone from the `twinkl-721` report, and checked that
  the revised recommendation ordering is internally consistent across both docs.

# twinkl-719.4

## Checklist

- [x] Extend post-hoc policy metrics to compute and persist circumplex diagnostics
- [x] Add compact circumplex fields to validation sweep exports and Markdown report tables
- [x] Persist full baseline/selected circumplex payloads in post-hoc policy YAMLs
- [x] Add ready-to-run `twinkl_719_5` BalancedSoftmax post-hoc config
- [x] Extend targeted `tests/vif/test_posthoc.py` coverage for circumplex-aware outputs
- [x] Run focused pytest verification for post-hoc tuning
- [x] Review changed files and summarize results

## Review

- Extended the generic post-hoc tuning path in `src/vif/posthoc.py` so every
  candidate now computes circumplex diagnostics from probability-aware outputs,
  writes compact `opposite_violation_mean` / `adjacent_support_mean` fields into
  validation sweeps and compact metric blocks, and persists full baseline vs
  selected circumplex payloads in each `selected_policy.yaml`.
- Updated the Markdown report renderer to keep the existing recall/QWK/calibration
  framing while adding `OppV` and `AdjS` to the per-run and family tables,
  surfacing circumplex deltas in the policy takeaways, and supporting an
  optional scope note plus non-contiguous run-scope formatting for `run_020`
  versus `run_036` comparisons.
- Added `config/experiments/vif/twinkl_719_5.yaml` as the ready-to-run
  BalancedSoftmax post-hoc config for the incumbent `run_020` checkpoint versus
  the weighted `run_036` reference checkpoint.
- Verification: `source .venv/bin/activate && pytest tests/vif/test_posthoc.py -q`
  passed, `python -m py_compile src/vif/posthoc.py` passed, and
  `load_config('config/experiments/vif/twinkl_719_5.yaml')` successfully loaded
  the new config.

# twinkl-719.5

## Checklist

- [x] Align post-hoc metric semantics with frontier run metrics
- [x] Extend targeted `tests/vif/test_posthoc.py` coverage for baseline parity
  and `tau=0` no-drift behavior
- [x] Run focused pytest verification for the post-hoc metric update
- [x] Execute `twinkl_719_5` validation-only post-hoc sweep
- [x] Review generated artifacts and finalize the report conclusion
- [x] Update `logs/experiments/index.md` if the post-hoc result changes the
  manual frontier narrative or needs an explicit post-hoc note
- [x] Review changed files and summarize results

## Review

- Updated `src/vif/posthoc.py` so softmax post-hoc scoring now uses the same
  thresholded continuous prediction basis as the frontier run YAMLs. This keeps
  `tau=0` baseline metrics aligned with the original saved run metrics and
  avoids comparing argmax-based post-hoc numbers against MC-dropout mean-based
  frontier results.
- Extended `tests/vif/test_posthoc.py` with focused parity coverage for
  score-based softmax metrics, threshold-vs-argmax disagreement, and `tau=0`
  no-drift behavior. Verification passed with
  `source .venv/bin/activate && pytest tests/vif/test_posthoc.py -q` and
  `python -m py_compile src/vif/posthoc.py tests/vif/test_posthoc.py`.
- Ran `python -m src.vif.posthoc --config config/experiments/vif/twinkl_719_5.yaml`,
  which wrote artifacts under
  `logs/experiments/artifacts/posthoc_twinkl_719_5_20260311_163957/` and updated
  `logs/experiments/reports/experiment_review_twinkl_719_5.md`.
- Final result: validation-only retargeting did **not** close the frontier gap.
  `run_020` selected `tau=0.30`, gaining a small `recall_-1` lift while losing
  too much QWK, minority recall, calibration, and circumplex cleanliness;
  `run_036` selected `tau=0.00`. `logs/experiments/index.md` now records that
  the frontier remains unchanged and that `twinkl-719.6` stays as the stronger
  fallback only if an incumbent-centered follow-up is still desired.

# twinkl-724

## Checklist

- [x] Create a beads issue for the repo Git-workflow instruction change
- [x] Add a repo-specific branch-creation rule to `AGENTS.md`
- [x] Re-read the edited `AGENTS.md` section to verify the wording is adjacent
  to the existing worktree rule and covers both creating and switching branches
- [x] Capture the correction pattern in `tasks/lessons.md`
- [x] Close the beads issue with the result

## Review

- Added an explicit repo-level Git workflow rule to `AGENTS.md` directly under
  the existing “Do NOT use git worktrees” instruction: stay on the current
  branch by default, do not create or switch branches unless the user
  explicitly asks, and ask first if branch isolation seems safer.
- Verified the new wording is adjacent to the existing worktree rule and closes
  the loophole around proactively switching branches without permission.
- Recorded the correction pattern in `tasks/lessons.md` so future workflow
  choices default to the current branch unless the user explicitly requests a
  branch.
