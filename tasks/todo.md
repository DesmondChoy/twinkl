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
