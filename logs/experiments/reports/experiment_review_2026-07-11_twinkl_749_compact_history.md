# Experiment Review: Compact Prior-Entry History (`twinkl-749`)

## 1. Experiment Overview

`run_069` tests one bounded context intervention against repaired-Security
seed-11 baseline `run_058`. Both use split seed 2025, model seed 11, 1,213 / 217
/ 221 train/validation/test entries, Nomic-256, hidden width 64, dropout 0.3,
BalancedSoftmax, the same learning rate and checkpoint policy, and
`security_active_critic_state_v1`. Leaderboard claims remain inside the
corrected post-`d937094` regime.

The only representation change is a separate 64-dimensional mean summary of
up to three strictly prior embeddings plus one history-count feature. State
width grows 266 to 331 and parameters grow 23,454 to 27,614 (+4,160).

## 2. Head-to-Head Comparison

| Test metric | `run_058` current only | `run_069` compact history | Delta |
|---|---:|---:|---:|
| MAE | **0.3239** | 0.3312 | +0.0073 |
| Accuracy | **0.7367** | 0.7290 | -0.0077 |
| QWK | **0.3631** | 0.3424 | -0.0207 |
| Spearman | **0.3645** | 0.3301 | -0.0344 |
| Calibration | 0.6883 | **0.7241** | +0.0358 |
| `recall_-1` | 0.2966 | **0.3033** | +0.0067 |
| Minority recall | **0.4456** | 0.3997 | -0.0459 |
| Hedging | **0.5968** | 0.6109 | +0.0141 |
| Opposite violation | 0.0893 | **0.0780** | -0.0113 |
| Adjacent support | **0.0900** | 0.0752 | -0.0148 |

The `recall_-1` difference is comparable under the 5% rule. QWK falls 5.7%
relative to baseline, while the minority-recall loss is larger still; the
broader package is worse.

## 3. Per-Dimension Analysis

| Dimension | `run_058` QWK | `run_069` QWK | Delta |
|---|---:|---:|---:|
| Self-direction | **0.557** | 0.538 | -0.018 |
| Conformity | **0.609** | 0.463 | -0.146 |
| Tradition | **0.444** | 0.435 | -0.009 |
| Benevolence | **0.386** | 0.367 | -0.019 |
| Achievement | 0.324 | **0.333** | +0.009 |
| Power | **0.356** | 0.254 | -0.103 |
| Security | **0.339** | 0.267 | -0.072 |
| Universalism | **0.302** | 0.276 | -0.026 |
| Stimulation | 0.163 | **0.356** | +0.193 |
| Hedonism | **0.152** | 0.135 | -0.017 |

The two lowest two-run mean-QWK dimensions are Hedonism and Stimulation.
Validation error inspection found extreme sign reversals. Three Hedonism `+1`
entries from persona `e5cea325` were predicted between `-0.64` and `-0.88`;
they discuss protecting free time, anticipated leisure, and guilt about taking
a Saturday. For Stimulation, one restless former Rockies guide labeled `-1`
was predicted `+0.82`, while two novelty/career-change entries labeled `+1`
were predicted near neutral (`-0.14` and `-0.02`). The summary can amplify a
persona-level theme in the wrong direction rather than resolve the current
entry.

## 4. Calibration Deep-Dive

Both runs have positive calibration in 10/10 dimensions and no deployment-risk
negative correlations. Global calibration improves from 0.6883 to 0.7241
(good), but that means uncertainty tracks the candidate's larger errors more
faithfully; it does not make those predictions better. Security calibration
improves 0.454 to 0.567 while Security QWK falls.

## 5. Hedging vs Minority Recall Trade-off

| Run | Hedging | Minority recall | Verdict |
|---|---:|---:|---|
| `run_058` | 59.7% | 44.6% | Decisive + balanced |
| `run_069` | 61.1% | 40.0% | Reasonable recall, but still neutral-biased |

Compact history crosses back above the 60% hedging boundary and loses 4.6
percentage points of minority recall. The small `recall_-1` gain comes with a
larger `+1` recall fall (0.595 to 0.496).

## 6. Capacity & Overfitting

The parameter/sample ratio rises from 19.3x to 22.8x; both are high. The
selected train-validation gap widens from 0.1081 to 0.1474, and training lasts
34 rather than 25 epochs. Selection traces confirm that `run_069` chose the
best eligible QWK epoch (23), not the lowest-loss epoch, so checkpoint policy
does not explain the holdout regression.

## 7. Systemic Insights & Hypotheses

The LLM context gain does not transfer automatically to deterministic pooling.
An LLM can interpret which prior event disambiguates the current entry; a mean
embedding erases order and relevance. Hypothesis: the summary blends unrelated
prior themes and creates persona-level shortcuts, explaining the sharp
Stimulation gain alongside Security, Power, and Conformity regressions.

The Security conclusion is narrower: its repaired target was judged from the
current session only. `run_069` is a fixed-target representation ablation, not
evidence for or against a separately history-visible Security target.

Current primary-source research supports mean pooling as the simplest
permutation-invariant control ([Deep Sets](https://arxiv.org/abs/1703.06114)),
while GRU and standard attention introduce more trainable structure
([Cho et al.](https://aclanthology.org/D14-1179/),
[Vaswani et al.](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need.pdf)).
Given this POC's 1,213 training rows and already-high parameter/sample ratio,
we infer greater estimation risk. The negative control therefore argues against
escalating architecture now.

## 8. Actionable Recommendations

1. Keep compact mean history config-gated and diagnostic; do not run seeds 22
   and 33 or change the default.
2. Carry the negative result into `twinkl-752`: the POC should not claim that
   the local MLP is trajectory-aware merely because history is available.
3. If context work is reopened, require a separately justified relevance-aware
   mechanism and target contract. Watch Security/Power/Conformity QWK and the
   train-validation gap, not aggregate QWK alone.
4. Prefer the existing LLM/Coach context layer for longitudinal interpretation
   in the current POC rather than widening the student again.

## 9. Summary Verdict

- **Best config:** `run_058` for the repaired-Security seed-11 comparison;
  `run_019`-`run_021` remains the historical corrected-split default.
- **Key weakness:** mean pooling cannot select which prior event is relevant and
  appears to amplify wrong persona-level signals.
- **Highest-leverage next experiment:** none inside `twinkl-749`; stop this line
  and use the result in the final capstone-scope decision.
