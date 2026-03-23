# Experiment Review — 2026-03-18 — `twinkl-734` SLACE Reserve-Branch Diagnostic

## 1. Experiment Overview

`twinkl-734` tested the reserve loss-branch hypothesis left open after the
representation diagnostics and the semantic-counterexample de-scope: can
`SLACE` add ordinal structure without giving back the tail-sensitive behavior
that made `BalancedSoftmax` the active frontier?

**What varied:**
- **Loss family:** `BalancedSoftmax` / prior long-tail losses -> `SLACE`
- **Loss hyperparameter:** `slace_alpha=1.0`

**What stayed constant:** `nomic-ai/nomic-embed-text-v1.5` at `256d`,
`window_size=1`, `hidden_dim=64`, `dropout=0.3`, `batch_size=16`, `epochs=100`,
fixed learning rate `0.015522253574270487`, `split_seed=2025`, `model_seed=22`,
the frozen holdout manifest, and the guarded checkpoint-selection policy.

**Dataset note:** `run_041` used the same refreshed corrected split as the late
March diagnostics: `n_train=1,213`, `n_val=217`, `n_test=221`, with `0/1,651`
entries truncated.

## 2. Head-to-Head Comparison

### Aggregate: `run_041` vs Active Incumbent and Weighted Reference

| Metric | Incumbent family med (`run_019`-`run_021`) | Weighted family med (`run_034`-`run_036`) | `run_041` SLACE |
|--------|---:|---:|---:|
| Params | 23,454 | 23,454 | 23,454 |
| `state_dim` | 266 | 266 | 266 |
| `n_train` | 1,022 | 1,213 | 1,213 |
| MAE | 0.304 | 0.315 | **0.222** |
| Accuracy | 0.753 | 0.746 | **0.811** |
| QWK | **0.362** | 0.342 | 0.338 |
| `recall_-1` | **0.313** | **0.378** | 0.134 |
| Minority Recall | **0.448** | **0.449** | 0.293 |
| Hedging | 0.621 | **0.599** | 0.810 |
| Calibration | 0.713 | 0.726 | **0.772** |

**Main read:** `run_041` is not a frontier candidate. It misses the incumbent
family-median floor on `qwk_mean` and collapses the tail package
(`recall_-1 0.134`, minority recall `0.293`) while reverting to extreme neutral
hedging (`81.0%`).

## 3. What Helped and What Failed

**1. SLACE keeps strong global calibration, but that is not the bottleneck.**
`run_041` posts `calibration_global 0.772` with `10/10` positively calibrated
dimensions, outperforming both the incumbent and the weighted reference on that
axis. But the frontier is not currently blocked on calibration. The question is
whether a new loss can preserve BalancedSoftmax-style tail recovery while
adding ordinal discipline, and this run does not.

**2. The tail-sensitive package regresses sharply.** Relative to the incumbent
family median, `run_041` gives back `0.179` on `recall_-1` (`0.313 -> 0.134`)
and `0.155` on minority recall (`0.448 -> 0.293`). Relative to the weighted
reference, the drop is even larger (`0.378 -> 0.134`). This is the opposite of
what `twinkl-734` needed to show.

**3. Hedging returns to the conservative-family regime.** `run_041` predicts
neutral on `80.95%` of outputs overall, worse than the active frontier
(`62.1%`) and the weighted branch (`59.9%`). That places it back in the same
qualitative zone as the conservative ordinal families the repo already ruled
out (`CDWCE_a3 80.4%`, `SoftOrdinal 79.6%`, `CORN 80.1%`).

**4. Better MAE / accuracy do not rescue the frontier case.** The run looks
strong on surface classification metrics (`MAE 0.222`, `accuracy 0.811`), but
those gains are driven by neutral-biased behavior rather than better
minority-class discrimination. This is exactly the failure mode the frontier
process is trying to avoid.

## 4. Dimension-Level Trade-offs

### Hard Dimensions and Key Regressions

| Metric | `run_020` incumbent anchor | Incumbent family med | `run_041` SLACE |
|--------|---:|---:|---:|
| Stimulation QWK | 0.303 | 0.161 | **0.337** |
| Hedonism QWK | **0.262** | **0.247** | 0.055 |
| Security QWK | 0.213 | **0.297** | 0.196 |
| Power QWK | **0.342** | 0.334 | -0.000 |

**Stimulation is the one apparent bright spot, but not a clean one.**
`stimulation qwk` rises to `0.337`, above both the incumbent anchor and the
family median. But that comes with `91.9%` stimulation hedging, so the metric
is not supported by healthy class commitment. It is not a clean sign that SLACE
solved the hard-dimension problem.

**Hedonism and power are clear failures.** `hedonism qwk` collapses to `0.055`,
far below the incumbent anchor (`0.262`) and the incumbent family median
(`0.247`). `power qwk` falls all the way to chance (`-0.000`), which helps
explain why the aggregate `qwk_mean` never reaches the incumbent floor despite
the superficially strong MAE and accuracy.

## 5. Selection and Overfitting Read

The selected checkpoint came from epoch `14/32`, with validation metrics that
looked stronger than the final holdout read:

- Validation selection package: `qwk_mean 0.433`, `recall_-1 0.198`,
  `hedging 0.791`, `calibration 0.803`
- Holdout package: `qwk_mean 0.338`, `recall_-1 0.134`, `hedging 0.810`,
  `calibration 0.772`

This is not a catastrophic collapse, but it is still a meaningful downgrade on
the metrics that matter. The run therefore fails both as a frontier challenger
and as a stable reserve branch worth expanding to three seeds immediately.

## 6. Recommendation

**Close `twinkl-734` as completed and drop SLACE from the active frontier
roadmap.**

- Do **not** run a 3-seed SLACE family rerun.
- Keep `run_019`-`run_021` as the active corrected-split default.
- Keep `run_034`-`run_036` as the best tail-sensitive reference branch.
- Treat this as another negative loss-family diagnostic: SLACE did not thread
  the QWK / tail-recall / hedging needle better than the current frontier.

If loss-design work is revisited later, it should be under a materially
different hypothesis than "try another plain ordinal loss." This run suggests
the remaining bottleneck is still not solved by swapping the loss alone.

## 7. Summary Verdict

`run_041` answered the reserve-branch question clearly. SLACE was easy to
integrate, calibrated well, and kept the incumbent parameter budget fixed, but
it failed the only gates that mattered: it did not meet the incumbent
`qwk_mean` floor, it badly regressed `recall_-1` and minority recall, and it
returned the model to extreme neutral hedging. The active frontier does not
change.

## 8. Artifacts

- Run config: `logs/experiments/runs/run_041_SLACE.yaml`
- Artifact root: `logs/experiments/artifacts/ordinal_v4_s2025_m22_20260318_213805/SLACE/`
