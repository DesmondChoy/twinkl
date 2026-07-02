# Experiment Review - 2026-05-28 - twinkl-lct3

## Verdict

No promotable no-new-data VIF Critic candidate was reached.

The selected artifact-only candidate is a **dead end**, not a new default and
not a reference branch. It is useful only as a diagnostic branch showing that
simple cross-family ensembling/routing over existing persisted-label checkpoint
outputs does not unlock the promotion floor.

No retraining was performed.

## Ideas Considered

The preflight memo is
`logs/experiments/reports/experiment_selection_memo_2026-05-28_twinkl_lct3.md`.
It considered four paths:

1. Conservative: two-stage active-recall reweighting.
2. Creative/high-upside: seed-aligned ensemble or decision policy over existing
   persisted-label checkpoint outputs.
3. Artifact-only diagnostic: validation complementarity ceiling.
4. Soft-label persisted-board sibling using existing consensus vote signal.

The selected path was Idea 2 with Idea 3 as the falsification gate. This had the
highest expected value because the repo evidence showed different families
winning different axes: incumbent hard-dimension stability, dimweight tail
recall/calibration, Qwen QWK/hedging, and two-stage calibration/stimulation.

## What Changed

Added:

- `scripts/experiments/no_new_data_vif_policy_search.py`
- `logs/experiments/reports/experiment_selection_memo_2026-05-28_twinkl_lct3.md`
- `logs/experiments/artifacts/no_new_data_vif_policy_twinkl_lct3_20260528/`

The search used only existing persisted-label selected output artifacts:

- incumbent `run_019`-`run_021`
- dimweighted `run_034`-`run_036`
- Qwen `run_042`-`run_044`
- two-stage `run_045`-`run_047`

All selected families shared the same fixed validation/test manifests:

- validation: 217 entries, 2,170 dimension rows
- test: 221 entries, 2,210 dimension rows

Consensus-label outputs were not used for promotion.

## Selected Candidate

Validation selected this dimension router:

| Dimension | Family |
|---|---|
| self_direction | Qwen |
| stimulation | Dimweight |
| hedonism | Two-stage |
| achievement | Two-stage |
| power | Qwen |
| security | Two-stage |
| conformity | Dimweight |
| tradition | Two-stage |
| benevolence | Two-stage |
| universalism | Qwen |

It looked tempting on validation:

| Split | QWK | recall_-1 | MinR | Hedging | Calibration | Floors |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.464 | 0.425 | 0.494 | 0.671 | 0.762 | 4/5 |
| Test | 0.365 | 0.244 | 0.393 | 0.676 | 0.718 | 0/5 |

The important catch: **even validation did not meet the hedging floor**. Across
all 143 searched validation policies, zero met `hedging <= 0.580`. The selected
candidate was therefore never honestly promotable; the test result simply made
the failure undeniable.

## Promotion Metrics

| Metric | Baseline `run_019`-`run_021` | Selected test | Delta | Floor | Pass? |
|---|---:|---:|---:|---:|---|
| QWK | 0.362 | 0.365 | +0.003 | >= 0.400 | No |
| recall_-1 | 0.313 | 0.244 | -0.070 | >= 0.400 | No |
| minority recall | 0.448 | 0.393 | -0.055 | >= 0.480 | No |
| hedging | 0.621 | 0.676 | +0.055 worse | <= 0.580 | No |
| calibration | 0.713 | 0.718 | +0.005 | >= 0.720 | No |

It did not beat the incumbent package. It barely nudged QWK and calibration, but
lost the actual operational reason BalancedSoftmax exists: rare-class recovery.

## Hard Dimensions

| Dimension | Baseline QWK | Selected QWK | QWK delta | Baseline active recall | Selected active recall | Active recall delta |
|---|---:|---:|---:|---:|---:|---:|
| hedonism | 0.247 | 0.242 | -0.005 | 0.284 | 0.274 | -0.010 |
| security | 0.297 | 0.276 | -0.021 | 0.342 | 0.267 | -0.075 |
| stimulation | 0.161 | 0.152 | -0.009 | 0.363 | 0.440 | +0.077 |

The candidate gets the required `stimulation` active-recall lift, but it misses
the hard-dimension floor because `security` regresses by slightly more than the
allowed 0.02 QWK and active recall drops hard.

## Three-Seed and Uncertainty Read

The result fails on the 3-seed family median before uncertainty intervals are
needed:

- 0/5 promotion floors met on test.
- Hard-dimension no-material-regression guardrail failed.
- The selected validation policy itself missed hedging by 9.1 points.

Because there is no point-estimate win, a persona-cluster bootstrap would not
change the decision. The existing `twinkl-730` uncertainty rule says intervals
matter for promotion claims; this branch has no promotion claim to defend.

## Diagnostic Read

The validation oracle upper bound was high on QWK/recall:

| Diagnostic | QWK | recall_-1 | MinR | Hedging | Calibration |
|---|---:|---:|---:|---:|---:|
| Per-cell validation oracle | 0.769 | 0.685 | 0.730 | 0.788 | 0.085 |

That oracle is label-aware and not deployable. Its real message is subtler:
there is complementary signal somewhere in the checkpoint family, but not in a
simple stable policy that can generalize without looking at labels. The usable
policies either keep hedging too high or overfit validation dimensions and
collapse on test.

## Path Ruled Out

Ruled out:

- Simple validation-selected probability ensembles over the existing
  persisted-label checkpoint families.
- Simple per-dimension family routing over the same artifacts.
- Temperature-sharpened coarse ensembles in the searched range.
- Treating dimweight/Qwen/two-stage complementarity as enough, by itself, to
  reach the promotable floor.

Not ruled out:

- A real training-time change to the two-stage active-recall loss.
- A carefully separated persisted-board soft-label experiment using existing
  consensus vote distributions.
- New data or target repair. Those were outside this session's constraints.

## Status

Classification: **dead end / diagnostic branch**.

Recommendation: keep `run_019`-`run_021` as the active persisted-label default.
Do not promote the selected policy. Do not spend a retrain just to create new
run IDs from this hypothesis.
