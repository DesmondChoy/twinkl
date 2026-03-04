# Experiment Review Recommendations (2026-03-04)

Scope:
- This report summarizes recommendations from a full cross-run review of `logs/experiments/runs/*.yaml` (53 runs).
- It complements the running narrative in `logs/experiments/index.md` with concrete, testable next experiments.

## Current Snapshot

- Best overall run remains: `run_010 CORN` (`QWK 0.434`, calibration `0.835`).
- Primary weakness remains: high hedging and weak rare-negative recall (`recall_minus1` still low in top runs).
- Hardest dimensions remain: `power` (worst mean QWK) and `security`.

## Recommendation 1: Logit-Adjusted Training on a Strong Ordinal Baseline

Action:
- Start with `run_012 SoftOrdinal` (already stronger minority recall profile) and apply logit adjustment using training priors.

Why:
- Class imbalance is still strong in labels (global `-1` is underrepresented), which encourages neutral-class hedging.

What to watch:
- `recall_minus1`
- `minority_recall_mean`
- `qwk_mean`
- `calibration_global`
- `hedging_mean`

Success condition:
- Improve rare-negative recall without collapsing calibration or QWK.

## Recommendation 2: Distance-Aware Ordinal Reweighting (CDW-CE style)

Action:
- Implement a class-distance-aware ordinal loss variant for `run_010 CORN` and `run_012 EMD`-like settings.

Why:
- Current failures concentrate in dimensions where ordinal distance matters (`power`, `security`), and plain objectives still over-hedge.

What to watch:
- Per-dimension QWK for `power` and `security`
- Cross-run variance reduction in those dimensions
- `hedging_mean`

Success condition:
- Better worst-dimension QWK with stable or improved global QWK.

## Recommendation 3: Dimension-Aware Class-Balanced Weighting

Action:
- Add per-dimension class-balanced weighting (effective-number/long-tail style) rather than only global weights.

Why:
- Dimension-specific skew differs materially; hard dimensions should not inherit a one-size-fits-all class prior.

What to watch:
- `power` and `security` QWK
- `recall_minus1`
- Gap between aggregate calibration and actionable minority detection

Success condition:
- Lift minority sensitivity while keeping calibration in moderate-to-good range.

## Recommendation 4: Keep ws=1 and hd=64 as the Main Tuning Lane

Action:
- Keep optimization centered on `nomic`, `ws=1` (plus selective `ws=2` checks), `hd=64`.

Why:
- This lane currently gives the best QWK/calibration compromise; larger capacity improved MAE/Acc at times but did not improve agreement reliably.

What to watch:
- QWK first, then calibration
- Avoid selecting by MAE alone

Success condition:
- New runs that exceed `run_010 CORN` on QWK with comparable calibration.

## Recommendation 5: De-Scope Weighted MSE from Frontier Candidate Set

Action:
- Keep MSE for historical diagnostics only, not as a candidate for the main optimization frontier.

Why:
- Weighted MSE can reduce hedging but produced dangerous/negative calibration behavior in prior runs.

What to watch:
- No new frontier decisions should depend on MSE outputs.

Success condition:
- Cleaner experiment bandwidth focused on ordinal heads with safer calibration behavior.

## Suggested Experiment Order

1. `run_012 SoftOrdinal` + logit adjustment
2. `run_010 CORN` + distance-aware ordinal reweighting
3. Per-dimension class-balanced weighting on best config from 1/2
4. Re-run a fixed seed pair (`ws=1` vs `ws=2`) only if frontier improves

## References (for methodological grounding)

- Menon et al., Long-tail learning via logit adjustment: https://arxiv.org/abs/2007.07314
- Polat et al., Class-distance-weighted objective (ordinal long-tail): https://arxiv.org/abs/2412.01246
- Class-balanced loss (effective number of samples): https://arxiv.org/abs/1901.05555
- LDAM (label-distribution-aware margins): https://arxiv.org/abs/1906.07413
- CORAL ordinal regression: https://arxiv.org/abs/1901.07884
- CORN ordinal regression: https://arxiv.org/abs/2111.08851
- Calibration baseline (temperature scaling): https://proceedings.mlr.press/v70/guo17a.html
