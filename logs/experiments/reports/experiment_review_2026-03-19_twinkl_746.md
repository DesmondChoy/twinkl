# Experiment Review — 2026-03-19 — `twinkl-746` Two-Stage VIF Reformulation

## 1. Experiment Overview

`twinkl-746` tested a new `TwoStageBalancedSoftmax` head: predict
`inactive` vs `active` first, then predict `misaligned` vs `aligned` inside the
active slice, with final probabilities reconstructed back to `{-1, 0, +1}`.
What varied was the model family and seed (`run_045`-`run_047`, seeds
`11/22/33`). What stayed constant was the active corrected-split frontier
budget: `nomic-256d`, `window_size=1`, `hidden_dim=64`, `dropout=0.3`,
`batch_size=16`, fixed LR `0.015522253574270487`, frozen holdout manifest, and
guardrailed checkpoint selection.

Leaderboard claims below are made only inside the active post-`d937094`
corrected-split regime. One causal caveat matters: the incumbent family
`run_019`-`run_021` used `n_train=1,022`, while the weighted reference
`run_034`-`run_036` and the new two-stage family both used `n_train=1,213`.
So the cleanest formulation comparison is **two-stage vs weighted**, not
two-stage vs incumbent alone.

## 2. Head-to-Head Comparison

| Family | Runs | `n_train` | Median QWK | Median `recall_-1` | Median MinR | Median Hedging | Median Cal | OppV / AdjS |
|--------|------|----------:|-----------:|-------------------:|------------:|---------------:|-----------:|------------:|
| Incumbent `BalancedSoftmax` | `run_019`-`run_021` | 1022 | **0.362** | 0.313 | 0.448 | 0.621 | 0.713 | 0.070 / 0.077 |
| Weighted reference | `run_034`-`run_036` | 1213 | 0.342 | **0.378** | **0.449** | **0.599** | 0.726 | 0.068 / 0.076 |
| `TwoStageBalancedSoftmax` | `run_045`-`run_047` | 1213 | 0.360 | 0.266 | 0.382 | 0.708 | **0.743** | **0.063 / 0.071** |

Main read: the reformulation is viable. Family-median QWK reached `0.360`
(fair), comparable to the incumbent `0.362`. But it did **not** beat the
frontier package because `recall_-1` fell to `0.266` and hedging rose to
`70.8%`, worse than both the incumbent and weighted branch.

## 3. Per-Dimension Analysis

| Dimension | Mean QWK Across Compared Runs | Read |
|-----------|------------------------------:|------|
| hedonism | 0.175 | Hard + volatile |
| stimulation | 0.217 | Hard + volatile |
| security | 0.246 | Hard |
| power | 0.316 | Fair but most volatile |
| universalism | 0.350 | Fair |
| achievement | 0.358 | Fair |
| benevolence | 0.379 | Fair |
| tradition | 0.499 | Easy |
| self_direction | 0.509 | Easy |
| conformity | 0.544 | Easy |

The two hardest dimensions stayed `hedonism` and `stimulation`. The new family
helped there more than the aggregate table suggests: median `stimulation qwk`
rose to `0.309` versus `0.161` incumbent and `0.152` weighted, while median
`hedonism qwk` reached `0.242`, comparable to incumbent `0.247` and well above
weighted `0.129`. `Power` stayed the blocker: median `power qwk` was `0.311`,
below incumbent `0.334` and weighted `0.368`.

**Weighting audit:** the weighted branch consistently upweighted
`universalism` to the max clamp `1.5`, `stimulation` to `1.27–1.34`, and
`power` to about `1.06`, while leaving `hedonism` near neutral (`0.97–0.99`)
and downweighting `security` (`0.82–0.85`). That schedule coherently targeted
`stimulation` and `power`, but it did **not** target `hedonism`, which helps
explain why the weighted family never solved that dimension.

**Hard-dimension qualitative check on `run_045` validation:** the largest
`hedonism` errors were Rachel Munoz entries where taking a Saturday for herself
or imagining free evenings after her mother's PT was labeled `+1`, but the
model predicted `-1` with scores `-0.759` and `-0.608`. The largest
`stimulation` errors showed the mirror image: Marcus Delgado's nostalgia for a
less routine life was labeled `-1` but predicted `+1` (`0.921`), while Lars
Henriksen and Tariq Haddad expressed novelty-seeking and were predicted `-1`.
The failure mode is not random. The model still confuses **guilt-laden leisure
language** and **ruminative change-seeking language**.

## 4. Calibration Deep-Dive

All three families remained globally well-calibrated: incumbent
`0.713` (good), weighted `0.726` (good), two-stage `0.743` (good). Every run
in the compared families kept `10/10` dimensions positively calibrated, and no
dimension crossed the deployment-risk threshold of calibration `< -0.4`. The
two-stage family therefore improved calibration cleanly; calibration is not the
reason it failed promotion.

## 5. Hedging vs Minority Recall Trade-off

| Family | Hedging % | Minority Recall | `recall_-1` | Verdict |
|--------|----------:|----------------:|------------:|---------|
| Incumbent `BalancedSoftmax` | 62.1% | 0.448 | 0.313 | Balanced, but still moderately hedged |
| Weighted reference | **59.9%** | **0.449** | **0.378** | **Decisive + balanced** |
| `TwoStageBalancedSoftmax` | 70.8% | 0.382 | 0.266 | Reasonable tail recall, but too hedged |

This is the promotion blocker. The two-stage head improved structure and some
hard dimensions, but it moved the model toward a more conservative operating
point instead of a more decisive one.

## 6. Capacity & Overfitting

| Family | Params | Param / Sample | Median Gap | Typical Selection Epoch |
|--------|-------:|---------------:|-----------:|-------------------------|
| Incumbent | 23,454 | 22.9 (high) | 0.123 | `19/29` |
| Weighted | 23,454 | 19.3 (high) | 0.193 | `26/32` |
| Two-stage | 24,104 | 19.9 (high) | 0.031 | `10/30` |

All three families are still in the high-ratio regime. The two-stage branch did
**not** look more overfit than the alternatives; if anything, it selected much
earlier. The `selection_trace` follow-up check also stayed below threshold:
`run_046` and `run_047` selected the minimum-loss epoch directly, and `run_045`
had no non-selected minimum-loss epoch that improved QWK by `+0.02` or
`recall_-1` by `+0.03`. No stretched-patience replay was justified.

## 7. Systemic Insights & Hypotheses

The new evidence changes the diagnosis slightly. The main bottleneck is **not**
that the two-stage model cannot tell `-1` from `+1` once a dimension is active.
On artifact-recomputed metrics, family-median active-sign accuracy was `0.778`,
comparable to incumbent `0.778` and slightly above weighted `0.755`. The real
problem is earlier in the pipeline: activation precision rose from `0.478` to
`0.565`, but activation recall fell from `0.745` / `0.779` to `0.676`. In
plain language, the new head is better at saying "active" only when it is
confident, but it still misses too many active cases.

The data audit supports that read. In the full labeled set, `stimulation` has
only `187` active examples total (`60` misaligned, `127` aligned), and
`hedonism` has `314` (`141` misaligned, `173` aligned). Those are exactly the
dimensions where the qualitative errors hinged on subtle wording rather than
explicit value labels. So the formulation helps, but data scarcity and subtle
cueing still dominate.

## 8. Actionable Recommendations

Primary-source check before recommending next steps: I reviewed current
long-tail literature touchpoints including **Decoupling Representation and
Classifier for Long-Tailed Recognition**, **Improving Calibration for
Long-Tailed Recognition**, and **Balanced Product of Calibrated Experts for
Long-Tailed Recognition**. The common lesson is to push the next intervention
toward **classifier-head or calibration-level changes**, not another broad
encoder swap.

1. Run a **Stage-A active-positive reweighting** ablation on top of the current
   two-stage head. Evidence: activation precision improved, but activation
   recall and global `recall_-1` stayed too low. Watch activation recall,
   `recall_-1`, and hedging.
2. Run a **two-stage + dimension-weighting** hybrid against the same
   `1,213`-row split. Evidence: the weighted branch already targets
   `stimulation` / `power`, while the two-stage head helps `stimulation` /
   `hedonism`. Watch `power qwk`, `hedonism qwk`, and family-median hedging.
3. Try a **decoupled head-only fine-tune** from `run_045` with active-balanced
   batches. Evidence: the two-stage family had the lowest median train/val gap,
   which makes classifier-side adaptation plausible without another end-to-end
   retrain. Watch `recall_-1` and calibration drift together.
4. Generate a **small targeted batch for guilt-coded leisure and
   rumination-coded novelty**. Evidence: the hardest validation misses on
   `hedonism` and `stimulation` were exactly those semantics. Watch only
   `hedonism` / `stimulation` QWK and activation recall; do not claim a new
   family win unless aggregate hedging also improves.

## 9. Summary Verdict

**Best config:** the active default remains `run_020 BalancedSoftmax`, because
it still offers the strongest overall operational package. The best new
checkpoint is `run_045 TwoStageBalancedSoftmax`, but it is a structural
diagnostic, not a promotion candidate.

**Key weakness:** the model still misses too many active hard-dimension cases
before polarity recovery even begins.

**Highest-leverage next experiment:** keep the two-stage head, but explicitly
rebalance **Stage A** toward active cases and judge it against the weighted
reference on the same `1,213`-row corrected split.
