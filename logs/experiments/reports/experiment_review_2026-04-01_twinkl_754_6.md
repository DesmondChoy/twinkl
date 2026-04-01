# Experiment Review — 2026-04-01 — `twinkl-754.6` Consensus-Label Retrain Diagnostic

## 1. Experiment Overview

`twinkl-754.6` reran the active `BalancedSoftmax` recipe with
`consensus_labels.parquet` on the frozen corrected-split holdout. What varied
inside the new family was only the model seed (`run_048`-`run_050`, seeds
`11/22/33`). What stayed constant was the active frontier budget:
`nomic-256d`, `window_size=1`, `hidden_dim=64`, `dropout=0.3`,
`batch_size=16`, fixed LR `0.015522253574270487`, and the same fixed holdout
manifest used by the post-lift corrected-split runs.

Two comparison baselines matter. The active persisted-label frontier is still
`run_019`-`run_021` (`n_train=1022`). The cleaner same-budget persisted-label
reference is `run_025`-`run_027` (`n_train=1213`), with `run_034`-`run_036`
as the tail-sensitive weighted reference. Leaderboard claims below stay inside
the active post-`d937094` corrected-split regime, but one caveat is decisive:
the consensus rerun changed the **evaluation labels** as well as the training
labels. On the fixed 221-row test split, consensus relabeled 17 `hedonism`,
27 `security`, and 13 `stimulation` entries. So `run_048`-`run_050` are a
diagnostic branch, not a directly leaderboard-eligible replacement for the
persisted-label board.

## 2. Head-to-Head Comparison

**Active persisted-label context**

| Family | Runs | Labels | `n_train` | MAE | Acc | QWK | Spearman | Cal | MinR | Hedging | `recall_-1` | OppV / AdjS |
|--------|------|--------|----------:|----:|----:|----:|---------:|----:|-----:|--------:|------------:|------------:|
| Incumbent `BalancedSoftmax` | `run_019`-`run_021` | persisted | 1022 | **0.304** | **0.753** | **0.362** | **0.365** | 0.713 | 0.448 | 62.1% | 0.313 | 0.070 / 0.077 |
| Post-lift `BalancedSoftmax` | `run_025`-`run_027` | persisted | 1213 | 0.319 | 0.737 | 0.346 | 0.345 | 0.693 | 0.442 | **59.8%** | 0.328 | 0.082 / 0.072 |
| Weighted reference | `run_034`-`run_036` | persisted | 1213 | 0.309 | 0.752 | 0.342 | 0.342 | **0.726** | **0.449** | 59.9% | **0.378** | **0.068 / 0.076** |

The active board still tells the same story: the incumbent is the best
all-around persisted-label package, while the weighted branch remains the best
tail-sensitive reference.

**Same-budget advisory label-source comparison**

| Metric | `run_025`-`run_027` persisted | `run_048`-`run_050` consensus | Delta |
|--------|-------------------------------:|-------------------------------:|------:|
| MAE | 0.319 | **0.275** | -0.044 |
| Accuracy | 0.737 | **0.781** | +0.044 |
| QWK | 0.346 | **0.372** | +0.026 |
| Spearman | 0.345 | **0.363** | +0.019 |
| Calibration | 0.693 | **0.770** | +0.077 |
| Minority Recall | **0.442** | 0.408 | -0.034 |
| Hedging | **59.8%** | 65.5% | +5.7 pts |
| `recall_-1` | **0.328** | 0.270 | -0.058 |
| OppV / AdjS | 0.082 / 0.072 | **0.047 / 0.062** | cleaner |

Main read: the consensus family improved surface metrics and circumplex
cleanliness, but it regressed on the tail-sensitive package that justified the
frontier in the first place. Against the incumbent `run_019`-`run_021`, QWK is
comparable (`0.372` vs `0.362`) and calibration is higher, but `recall_-1`,
minority recall, and hedging are worse, and the label regime is no longer the
same.

## 3. Per-Dimension Analysis

| Dimension | Mean family-median QWK | Read |
|-----------|-----------------------:|------|
| hedonism | 0.184 | Hard + volatile |
| stimulation | 0.210 | Hard + most volatile |
| security | 0.241 | Hard |
| power | 0.332 | Fair |
| benevolence | 0.339 | Fair |
| universalism | 0.368 | Fair |
| achievement | 0.383 | Fair + volatile |
| tradition | 0.500 | Easy |
| self_direction | 0.511 | Easy |
| conformity | 0.532 | Easy |

The consensus branch did **not** solve the hard dimensions evenly. Family-median
`stimulation qwk` rose to `0.340`, well above incumbent `0.161`, but
`hedonism qwk` fell to `0.104`, and `security qwk` only recovered to `0.247`,
still below incumbent `0.297`. The `dimension_weight_trace` is inert here:
selected-epoch `applied_weight` stayed `1.0` for every dimension in all three
runs because dimension weighting was disabled.

Temporary validation error analysis on the best new checkpoint (`run_050`)
confirmed that `hedonism` is semantic, not just marginal. The largest misses
were entries where keeping a Saturday free, refusing a charge-nurse promotion,
or pushing for a faster prototype were labeled `+1`, but the model predicted
neutral or `-1`. In plain language, it still confuses **self-protective leisure
or room-to-breathe language** with **duty avoidance**. On `stimulation`, the
largest misses flipped between novelty and stability: declining travel nursing
was labeled `-1` but predicted near-neutral, while “everything is visible at a
new company” was pushed to `+1` when the consensus label was `0`.

## 4. Calibration Deep-Dive

All compared families stayed globally well calibrated. Median global
calibration was `0.713` for the incumbent, `0.693` post-lift, `0.726` for the
weighted reference, and `0.770` for the consensus family, all in the good
range for this stack. Every family kept `10/10` dimensions positively
calibrated, and no dimension crossed the deployment-risk threshold of
calibration `< -0.4`.

So calibration is not the blocker. If anything, the consensus branch is the
best-calibrated `BalancedSoftmax` family so far. But because the evaluation
labels changed, that gain is diagnostic evidence, not promotion evidence.

## 5. Hedging vs Minority Recall Trade-off

| Family | Hedging % | Minority Recall | `recall_-1` | Verdict |
|--------|----------:|----------------:|------------:|---------|
| Incumbent `BalancedSoftmax` | 62.1% | 0.448 | 0.313 | Balanced, but still moderately hedged |
| Post-lift `BalancedSoftmax` | **59.8%** | 0.442 | 0.328 | **Decisive + balanced** |
| Weighted reference | 59.9% | **0.449** | **0.378** | **Decisive + balanced** |
| Consensus `BalancedSoftmax` | 65.5% | 0.408 | 0.270 | Too hedged and weaker on the rare classes |

This is the operational blocker. The consensus branch looks cleaner on MAE,
accuracy, QWK, and calibration, but it does so while predicting the neutral
class more often and recovering fewer rare active cases.

## 6. Capacity & Overfitting

| Family | Params | Param / Sample | Median Gap | Typical Selection Epoch |
|--------|-------:|---------------:|-----------:|-------------------------|
| Incumbent | 23,454 | 22.9 (high) | 0.123 | `19 / 29` |
| Post-lift | 23,454 | 19.3 (high) | 0.199 | `22 / 30` |
| Weighted reference | 23,454 | 19.3 (high) | 0.193 | `26 / 32` |
| Consensus | 23,454 | 19.3 (high) | 0.238 | `22 / 28` |

Capacity did not change, so this review is not about a larger model
overfitting. But the consensus family did show the loosest median train/val
gap (`0.238`, some overfitting). The selection-trace audit also matters:
`run_050` selected the minimum-loss epoch directly, but `run_048` and
`run_049` chose later QWK-first checkpoints over lower-loss, much higher-recall
alternatives. On `run_048`, epoch `13` offered `+0.060` validation
`recall_-1` at only `-0.014` QWK. On `run_049`, epoch `13` offered `+0.107`
`recall_-1` at `-0.011` QWK. So part of the conservative operating point is a
selection-policy choice, not just a data effect.

## 7. Systemic Insights & Hypotheses

The new evidence does **not** support the simple story “consensus labels are
better.” The more precise story is: hard consensus relabeling makes the branch
look cleaner on global metrics and circumplex structure, but it weakens the
rare active signals on exactly the ambiguous dimensions we care about.

The causal guardrail matters here. Relative to the incumbent, two confounds are
present: `n_train` rose from `1022` to `1213`, and the labels changed. Relative
to the same-size persisted baseline `run_025`-`run_027`, the data-size confound
disappears, but the evaluation-label confound remains. That is why the current
result should be treated as a diagnostic branch, not a new frontier family.

Two hypotheses stand out. First, hard majority replacement is neutralizing
disagreement-heavy active cases. On the fixed test split, `security` active
counts moved from `14/-1` and `38/+1` under persisted labels to `8/-1` and
`31/+1` under consensus, and `stimulation` negatives fell from `12` to `7`.
Second, `hedonism` is failing on semantics rather than gross class counts: its
overall marginals barely changed, yet family-median `hedonism qwk` collapsed to
`0.104`, and the worst misses still center on guilt-coded leisure and boundary
setting.

## 8. Actionable Recommendations

Primary-source check before finalizing these steps: I reviewed recent
multi-annotator and subjective-label work including
[Annot-Mix](https://arxiv.org/abs/2405.03386),
[Annotator-Centric Active Learning for Subjective NLP Tasks](https://arxiv.org/abs/2404.15720),
[Meta-learning Representations for Learning from Multiple Annotators](https://arxiv.org/abs/2506.10259),
[Perspectives in Play](https://arxiv.org/abs/2506.20209), and
[Transfer Knowledge from Head to Tail](https://arxiv.org/abs/2304.06537).
That literature reinforced two points: preserve disagreement signal instead of
collapsing it too early, and do not spend the next cycle on generic calibration
alone.

1. Run a **same-code sibling family** with `judge_labels.parquet` on the exact
   `twinkl-754` configs and dual-score both families on both label sets. The
   evidence is the holdout relabeling itself: `17` `hedonism`, `27` `security`,
   and `13` `stimulation` test labels changed. Watch whether the QWK/calibration
   lift survives once evaluation labels match.
2. Train a **confidence-tiered or soft-label BalancedSoftmax variant** from the
   existing 5-pass vote distributions instead of full hard replacement. Recent
   multi-annotator work above consistently preserves disagreement structure
   rather than collapsing it to one label. Watch `hedonism qwk`,
   `stimulation qwk`, `recall_-1`, and calibration together.
3. Replay **selection-policy sensitivity** on the existing consensus traces with
   a light recall-aware tie-break or floor before another full retrain. The
   evidence is the `run_048` / `run_049` high-recall near-miss epochs. Watch
   holdout `recall_-1` and minority recall while capping QWK loss to `<= 0.02`.
4. If soft labels still leave `hedonism` weak, generate a **small targeted
   batch for guilt-coded leisure and boundary-setting language**, not another
   broad relabeling sweep. The evidence is the hard-dimension error analysis.
   Watch `hedonism qwk` and active-sign accuracy only.

## 9. Summary Verdict

**Best config:** the active default remains the persisted-label incumbent
`run_019`-`run_021`. The best new checkpoint is `run_050`, but it is a
diagnostic best case, not a promotion candidate.

**Key weakness:** hard consensus relabeling improves the clean surface metrics
while erasing or muting the disagreement-heavy active signals needed for
tail-sensitive `hedonism` / `security` behavior.

**Highest-leverage next experiment:** run a two-arm same-code ablation:
persisted hard labels versus confidence-tiered or soft consensus labels, with
both arms scored on both label sets. That is the cleanest way to separate true
learning gains from relabeling gains.
