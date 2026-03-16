# Experiment Review — 2026-03-16 — `twinkl-731` full-768d Nomic v1.5 truncation diagnostic

## 1. Overview

`twinkl-731` tested whether the active `nomic-ai/nomic-embed-text-v1.5`
frontier was losing hard-dimension polarity signal by truncating to `256d`.
The diagnostic stayed intentionally narrow:

- Incumbent anchor: `run_020` `BalancedSoftmax` (`seed=22`)
- Holdout: frozen `config/experiments/vif/twinkl_681_5_holdout.yaml`
- Shared constants: `split_seed=2025`, `window_size=1`, `dropout=0.3`,
  `batch_size=16`, `epochs=100`, `weight_decay=0.01`, `BalancedSoftmax`,
  explicit LR `0.015522253574270487`, no LR finder, incumbent
  `qwk_then_recall_guarded` selection policy
- New runs:
  - `run_037_BalancedSoftmax`: full native `768d`, `hidden_dim=64`
  - `run_038_BalancedSoftmax`: full native `768d`, fair-budget
    `hidden_dim=28`

Both new runs logged `state_dim=778` and `truncate_dim=null`, confirming the
diagnostic actually exercised the full Nomic representation. The only intended
difference between the two new runs was classifier width and therefore critic
capacity.

## 2. Aggregate Comparison Against `run_020`

| Run | Encoder | `hidden_dim` | `state_dim` | Params | `qwk_mean` | `recall_-1` | `minority_recall_mean` | `calibration_global` | `hedging_mean` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_020` incumbent | `nomic-256d` | 64 | 266 | 23,454 | **0.378** | **0.342** | **0.449** | 0.713 | 0.621 |
| `run_037` full 768d | `nomic-768d` | 64 | 778 | 56,222 | 0.318 | 0.269 | 0.381 | **0.720** | 0.620 |
| `run_038` fair-budget 768d | `nomic-768d` | 28 | 778 | 23,606 | 0.299 | 0.246 | 0.346 | 0.657 | **0.612** |

Holdout deltas versus `run_020`:

| Run | Δ `qwk_mean` | Δ `recall_-1` | Δ `minority_recall_mean` | Δ `calibration_global` | Δ `hedging_mean` |
|---|---:|---:|---:|---:|---:|
| `run_037` | `-0.060` | `-0.073` | `-0.068` | `+0.006` | `-0.001` |
| `run_038` | `-0.078` | `-0.096` | `-0.103` | `-0.057` | `-0.009` |

The headline result is negative:

- The unconstrained full-768d run `run_037` does **not** beat the incumbent on
  any of the core frontier metrics except a tiny calibration uptick.
- The fair-budget companion `run_038` is worse still on aggregate ranking,
  tail recovery, minority recall, and calibration.
- Neither run makes a promotion case even as a single-checkpoint curiosity; both
  trail the active family median (`qwk_mean 0.362`, `recall_-1 0.313`) and the
  weighted reference family on the tail-sensitive package.

## 3. Hedonism / Security Readout

| Run | `hedonism qwk` | `hedonism cal` | `hedonism hedge` | `security qwk` | `security cal` | `security hedge` |
|---|---:|---:|---:|---:|---:|---:|
| `run_020` incumbent | **0.262** | **0.856** | 0.697 | 0.213 | 0.452 | 0.706 |
| `run_037` full 768d | 0.178 | 0.755 | 0.747 | 0.165 | 0.444 | **0.629** |
| `run_038` fair-budget 768d | -0.045 | 0.822 | 0.792 | **0.225** | **0.553** | 0.815 |

Target-dimension interpretation:

- `run_037` makes both hard dimensions worse on QWK. `Security` hedging
  improves, but not enough to compensate for the loss in ranking quality.
- `run_038` shows a tiny `security qwk` uptick (`+0.012`) and stronger
  `security` calibration, but it pays for that with much heavier security
  hedging (`+0.109`) and a catastrophic `hedonism qwk` collapse to `-0.045`.
- Because the issue was specifically about recovering polarity signal for
  `hedonism` / `security`, these results argue against a useful
  representation-level win.

## 4. Capacity Confound Readout

The primary diagnostic carried the expected capacity confound:

- `run_020`: `23,454` parameters
- `run_037`: `56,222` parameters
- `run_038`: `23,606` parameters

The fair-budget companion matters because it keeps parameter count almost flat
to the incumbent while preserving the full `768d` representation. If the 768d
embedding were genuinely rescuing polarity signal, the matched-budget control
should at least preserve or selectively improve the hard dimensions. It did not.

This means the evidence is not merely “the bigger model overfit.” The stronger
negative conclusion is:

- the extra 768d representation did not unlock a cleaner `hedonism`/`security`
  signal in the current frontier setup
- the larger-width run did not produce a useful upside that the fair-budget run
  merely failed to hold onto

## 5. Recommendation

Do **not** run a 3-seed full-768d follow-up. Close `twinkl-731` as a negative
diagnostic and proceed to `twinkl-732`.

Why:

- both 768d runs regress against the incumbent `run_020` on aggregate QWK,
  `recall_-1`, and minority recall
- the target dimensions do not improve in a way that supports the original
  polarity-loss hypothesis
- the fair-budget control rules out the strongest “maybe this is just a
  capacity confound” excuse for continuing the line

Operationally, the active corrected-split frontier stays unchanged:

- default family remains `run_019`-`run_021` `BalancedSoftmax`
- best tail-sensitive reference branch remains `run_034`-`run_036`
- next issue should be `twinkl-732`, not a full-768d seed expansion
