# twinkl-681.5 Hard-Negative Data Lift Review

## Scope

`twinkl-681.5` tested whether a small leakage-safe targeted data lift could
improve hard-negative recovery on the frozen corrected holdout without changing
the current `BalancedSoftmax` frontier setup.

- Baseline family: `run_019`-`run_021` (`BalancedSoftmax`, seeds `11/22/33`)
- Augmented family: `run_022`-`run_024` (`BalancedSoftmax`, seeds `11/22/33`)
- Holdout: frozen `config/experiments/vif/twinkl_681_5_holdout.yaml`
- New synthetic batch: 12 targeted personas, 95 new entries, all routed to train
- Target dimensions: `Power`, `Security`
- QA gate: passed (`7 keep`, `1 ambiguous`, `0 bad label`) in `logs/exports/twinkl_681_5_label_qa.md`

## Family-Level Delta

Median metrics, augmented family minus baseline family:

| Metric | Baseline (`run_019`-`run_021`) | Augmented (`run_022`-`run_024`) | Delta |
|---|---:|---:|---:|
| `qwk_mean` | 0.3619 | 0.3488 | -0.0131 |
| `recall_-1` | 0.3132 | 0.3420 | +0.0288 |
| `minority_recall_mean` | 0.4480 | 0.4344 | -0.0136 |
| `hedging_mean` | 0.6213 | 0.6190 | -0.0023 |
| `calibration_global` | 0.7134 | 0.6866 | -0.0268 |
| `mae_mean` | 0.3042 | 0.3125 | +0.0083 |
| `accuracy_mean` | 0.7534 | 0.7484 | -0.0050 |

## Target-Dimension Readout

| Dimension | Baseline QWK | Augmented QWK | Delta QWK | Baseline `recall_-1` | Augmented `recall_-1` | Delta `recall_-1` | Delta Hedging | Delta Calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `power` | 0.3337 | 0.3452 | +0.0115 | 0.1250 | 0.3125 | +0.1875 | +0.0227 | +0.0419 |
| `security` | 0.2973 | 0.2995 | +0.0022 | 0.5714 | 0.5714 | +0.0000 | +0.0045 | +0.0255 |
| `hedonism` | 0.2471 | 0.1466 | -0.1005 | 0.6522 | 0.4348 | -0.2174 | +0.0090 | -0.0606 |

## Full Per-Dimension Delta Table

| dimension | qwk_delta | recall_minus1_delta | hedging_delta | calibration_delta |
|---|---:|---:|---:|---:|
| `self_direction` | -0.0057 | -0.0222 | -0.0633 | +0.0142 |
| `stimulation` | -0.0019 | -0.0833 | +0.0769 | -0.0316 |
| `hedonism` | -0.1005 | -0.2174 | +0.0090 | -0.0606 |
| `achievement` | -0.0406 | +0.1429 | +0.0045 | -0.0158 |
| `power` | +0.0115 | +0.1875 | +0.0227 | +0.0419 |
| `security` | +0.0022 | +0.0000 | +0.0045 | +0.0255 |
| `conformity` | -0.0367 | +0.0556 | -0.0046 | +0.0864 |
| `tradition` | +0.0313 | +0.0000 | +0.1040 | +0.0197 |
| `benevolence` | +0.0193 | +0.0000 | +0.0453 | -0.0672 |
| `universalism` | -0.0448 | -0.3333 | +0.0136 | +0.0087 |

## Interpretation

**1. The added data helped `Power` in the way we wanted.** The strongest
issue-scoped result is `power recall_-1`: median test recall rose from `0.125`
 to `0.3125`, while `power qwk` and calibration also improved modestly. This is
 consistent with the new batch adding useful mild-misalignment examples rather
 than only easy negatives.

**2. `Security` did not meaningfully move on the hard-negative metric.** The
 median `security recall_-1` stayed exactly `0.5714`. QWK and calibration rose a
 little, but the targeted batch did not produce the same boundary shift here as
 it did for `Power`.

**3. The lift does not generalize cleanly across the whole model family.**
 Family-level `recall_-1` improved by `+0.0288`, and hedging improved
 marginally, but median `qwk_mean`, calibration, MAE, and accuracy all worsened.
 On this evidence, `run_022`-`run_024` are not a clean frontier replacement for
 `run_019`-`run_021`.

**4. The absence of `Hedonism` targeting mattered.** `hedonism` regressed on
 both `qwk` and `recall_-1`, which is useful negative evidence: the 681.5 batch
 helped where it was targeted, but it did not create broader collateral gains on
 another difficult dimension.

## Verdict

The 681.5 data lift improved boundary clarity for `Power`, left `Security`
 mostly flat, and did not produce a family-wide win. The cleanest summary is:

- good targeted win on `Power`
- near-null result on `Security`
- mixed global result with a small `recall_-1` gain but weaker aggregate quality

`run_019`-`run_021` should remain the default general frontier family. Treat
`run_022`-`run_024` as targeted evidence that mild-misalignment data can help a
 hard dimension, but not yet as the new default `BalancedSoftmax` baseline.

## Suggested Handoff

If there is a follow-up issue, it should not be framed as “more of the same
batch.” The next step should either:

1. isolate `Security` with a sharper hypothesis about why its hard-negative
   boundary did not move, or
2. revisit `Hedonism` explicitly rather than expecting collateral transfer from
   `Power`/`Security` augmentation.
