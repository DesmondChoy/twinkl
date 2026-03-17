# Experiment Review — 2026-03-11 — `twinkl-719.3` weighted BalancedSoftmax frontier rerun

## 1. Overview

`twinkl-719.3` reran the frozen-holdout corrected-split `BalancedSoftmax`
frontier with the new EMA-smoothed per-dimension weighting path added in
`twinkl-719.2`.

- Weighted family: `run_034`-`run_036` (`BalancedSoftmax`,
  `balanced_softmax_dimweight`)
- Incumbent default: `run_019`-`run_021`
- Post-lift control: `run_025`-`run_027`
- Circumplex branches: `run_028`-`run_030` and `run_031`-`run_033`
- Constants held fixed: frozen holdout
  `config/experiments/vif/twinkl_681_5_holdout.yaml`, `split_seed=2025`,
  model seeds `11/22/33`, `nomic-ai/nomic-embed-text-v1.5` at 256d,
  `window_size=1`, `hidden_dim=64`, `dropout=0.3`, `batch_size=16`,
  `epochs=100`, `weight_decay=0.01`, no circumplex regularizer, and the
  `twinkl-715` validation `recall_-1` guardrail
  (`recall_minus1_floor=0.4032`)
- Weighting recipe: `inverse_loss`, `temperature=0.5`, `ema_alpha=0.3`,
  `warmup_epochs=1`, `eps=1e-6`, clamps `[0.5, 1.5]`

The current workspace still contains `204` personas and `1651` judged entries,
so the realized frozen-holdout split remains `1213 / 217 / 221`
train / val / test.

The decision question was narrow: does weighted `BalancedSoftmax` beat the
active default cleanly enough to become the new frontier, or is it better
treated as a strong reference branch for the next post-hoc step?

## 2. Family Comparison

All values below are family medians with IQR in parentheses.

| Family | Runs | `qwk_mean` | `recall_-1` | `minority_recall_mean` | `hedging_mean` | `calibration_global` |
|---|---|---:|---:|---:|---:|---:|
| Current default BalancedSoftmax | `run_019`-`run_021` | **0.362** (0.010) | 0.313 (0.033) | 0.448 (0.025) | 0.621 (0.038) | 0.713 (0.036) |
| Weighted BalancedSoftmax | `run_034`-`run_036` | 0.342 (0.030) | **0.378** (0.044) | **0.449** (0.040) | 0.599 (0.024) | **0.726** (0.027) |
| Post-lift control BalancedSoftmax | `run_025`-`run_027` | 0.346 (0.009) | 0.328 (0.039) | 0.442 (0.023) | **0.598** (0.021) | 0.693 (0.026) |
| Circumplex-regularized BalancedSoftmax | `run_028`-`run_030` | 0.347 (0.033) | 0.265 (0.020) | 0.411 (0.012) | 0.641 (0.009) | 0.709 (0.030) |
| Guardrailed circreg BalancedSoftmax | `run_031`-`run_033` | **0.366** (0.010) | 0.267 (0.047) | 0.409 (0.013) | 0.641 (0.013) | 0.713 (0.020) |

The weighted family is the clearest **tail-sensitive reference branch** so far,
but it does not clear the bar for a default change.

- Relative to the incumbent `run_019`-`run_021`, weighted `BalancedSoftmax`
  improves median `recall_-1` from `0.313` to `0.378`, keeps minority recall
  effectively flat to slightly better (`0.448` to `0.449`), reduces hedging
  from `0.621` to `0.599`, and improves calibration from `0.713` to `0.726`.
- The blocking cost is aggregate ranking quality: median `qwk_mean` falls from
  `0.362` to `0.342`, and the family is much less stable across seeds
  (`IQR 0.030` vs `0.010`).
- Relative to the recent circumplex branches, weighted `BalancedSoftmax`
  clearly wins the operational package that matters here: it is materially
  better on `recall_-1`, minority recall, hedging, and calibration than both
  `run_028`-`run_030` and `run_031`-`run_033`.
- Relative to the post-lift control `run_025`-`run_027`, the weighted family
  is the better reference branch overall. It gives back only `0.004` median
  QWK while improving `recall_-1` by `0.050`, minority recall by `0.007`, and
  calibration by `0.033`, with essentially flat hedging.

The best single weighted checkpoint is `run_036`: QWK `0.381`,
`recall_-1 0.387`, minority recall `0.492`, hedging `0.599`, and calibration
`0.726`. That is a strong single-seed result, but the family median is still
too QWK-volatile to replace the incumbent default.

## 3. Target-Dimension Readout

Median per-dimension holdout summaries:

| Family | `hedonism qwk` | `hedonism cal` | `hedonism hedge` | `security qwk` | `security cal` | `security hedge` | `power qwk` | `power cal` | `power hedge` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current default BalancedSoftmax | 0.247 | 0.856 | 0.719 | **0.297** | 0.496 | 0.706 | 0.334 | 0.759 | 0.769 |
| Weighted BalancedSoftmax | 0.129 | 0.786 | 0.733 | 0.222 | **0.558** | **0.674** | 0.368 | 0.827 | 0.765 |
| Post-lift control BalancedSoftmax | **0.256** | **0.877** | 0.765 | 0.199 | 0.528 | 0.738 | 0.307 | 0.762 | **0.751** |
| Circumplex-regularized BalancedSoftmax | 0.111 | 0.850 | 0.778 | 0.199 | 0.529 | 0.679 | **0.375** | **0.838** | 0.792 |
| Guardrailed circreg BalancedSoftmax | 0.160 | 0.829 | 0.747 | 0.229 | 0.459 | **0.674** | 0.370 | 0.828 | 0.819 |

This is the clearest reason not to promote the weighted family to the default.

- The branch does **not** rescue `hedonism`. Median `hedonism qwk` drops to
  `0.129`, well below the incumbent `0.247` and the post-lift control `0.256`.
- `Security` is better than the post-lift and regularized branches on both
  calibration and hedging, but it still trails the incumbent on the core
  ranking metric (`0.222` vs `0.297`).
- The weighted family’s extra tail recovery appears to come more from broader
  boundary movement, including stronger `power` behavior (`power qwk 0.368`
  vs incumbent `0.334`), than from a clean `hedonism` / `security` semantic
  fix.

So the family-level tail gains are real, but they are not being driven by the
two dimensions that motivated this intervention.

## 4. Circumplex Diagnostics

For the incumbent `run_019`-`run_021`, the circumplex summaries below were
recomputed from the saved selected-test artifacts because those runs predated
direct circumplex payload logging in the YAMLs. The same artifact path was used
for the new weighted family to keep the comparison basis consistent.

| Family | `opposite_violation_mean` | `adjacent_support_mean` |
|---|---:|---:|
| Current default BalancedSoftmax | 0.070 (0.016) | 0.077 (0.013) |
| Weighted BalancedSoftmax | 0.068 (0.014) | 0.076 (0.008) |
| Post-lift control BalancedSoftmax | 0.082 (0.010) | 0.072 (0.008) |
| Circumplex-regularized BalancedSoftmax | 0.039 (0.006) | **0.077** (0.010) |
| Guardrailed circreg BalancedSoftmax | **0.035** (0.002) | 0.077 (0.002) |

Weighted `BalancedSoftmax` is structurally acceptable:

- It is essentially **incumbent-equivalent** on the compact circumplex
  summaries: `0.068 / 0.076` vs `0.070 / 0.077`.
- It is clearly better than the post-lift control on opposite-pair collapse.
- It does **not** match the circumplex-regularized branches if pure structure
  cleanliness is the objective, but those branches still lose badly on the
  tail-sensitive operating metrics that matter more here.

So the weighted family is not blocked by circumplex regressions. Its failure to
become the default is about QWK and hard-dimension semantics, not structure.

## 5. Weighting Audit

Selected-epoch weighting was stable across all three seeds.

| Dimension | Median selected weight | Seed range |
|---|---:|---:|
| `universalism` | 1.500 | 1.500-1.500 |
| `stimulation` | 1.321 | 1.270-1.338 |
| `tradition` | 1.151 | 1.147-1.175 |
| `power` | 1.064 | 1.061-1.065 |
| `hedonism` | 0.972 | 0.970-0.993 |
| `security` | 0.818 | 0.817-0.846 |
| `self_direction` | 0.703 | 0.700-0.724 |

The schedule behaved coherently for the chosen inverse-loss recipe:

- No dimension ever hit the minimum clamp.
- `Universalism` hit the maximum clamp in `49 / 97` trace rows across the three
  runs; no other dimension hit either clamp.
- `Security` was consistently downweighted at the selected epoch
  (`0.817 / 0.846 / 0.818`), which is at least directionally consistent with
  the idea of softening one noisy hard dimension.
- `Hedonism` stayed almost neutral (`0.970 / 0.993 / 0.970`), so this branch
  was **not** aggressively suppressing it.

In other words, the weighting path is not behaving erratically. It is doing
what inverse-loss weighting would suggest: upweighting low-loss stable
dimensions and softening noisier ones. The negative result is therefore about
the intervention’s empirical effect, not a broken schedule.

## 6. Recommendation

Keep weighted `BalancedSoftmax` as a **reference branch**, not the active
default.

Why it stays alive:

- It is the best current BalancedSoftmax-family result on the **tail package**:
  `recall_-1`, minority recall, hedging, and calibration.
- It clearly beats both circumplex branches on the operational metrics that
  justified those follow-ups.
- Its circumplex summaries are effectively incumbent-equivalent, so there is no
  hidden structure tax blocking future use.

Why it does not replace `run_019`-`run_021`:

- median `qwk_mean` regresses from `0.362` to `0.342`
- seed stability is materially worse (`IQR 0.030` vs `0.010`)
- `hedonism` and `security` still do not beat the incumbent where the frontier
  is most fragile

Operationally, this means:

- keep `run_019`-`run_021` as the active corrected-split default
- keep `run_034`-`run_036` as the strongest **training-time reference branch**
  for the next frontier comparison
- treat `run_036` as the strongest single weighted checkpoint for qualitative
  inspection or future checkpoint-level follow-up

## 7. Bottom Line

`twinkl-719.3` is a partial positive result.

Per-dimension weighting improves the part of the frontier story that most
concerns misalignment sensitivity, but it does not yet produce a default-worthy
family because the QWK regression and hard-dimension semantics are still too
real. The branch should stay alive as the best tail-sensitive reference for the
next post-hoc comparison, not as the new mainline model family.
