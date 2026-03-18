# Experiment Review — 2026-03-18 — `twinkl-744` Controlled Qwen Frontier Rerun

## 1. Experiment Overview

`twinkl-744` reran the `Qwen/Qwen3-Embedding-0.6B` encoder under the active
corrected-split frontier budget to answer the question left open by the earlier
single-seed near-miss: is Qwen actually frontier-credible as a family, or was
`run_040` just a lucky seed?

**What varied:**
- **Model seed:** `11`, `22`, `33`

**What stayed constant:** `BalancedSoftmax`, `256d`, `window_size=1`,
`hidden_dim=64`, `dropout=0.3`, `batch_size=16`, fixed learning rate
`0.015522253574270487`, frozen holdout manifest, corrected split `split_seed=2025`,
no dimension weighting, and no circumplex regularizer.

## 2. Family Comparison

| Candidate | Runs | Median QWK | Median `recall_-1` | Median MinR | Median Hedging | Median Cal |
|----------|------|-----------:|-------------------:|------------:|---------------:|-----------:|
| Incumbent `BalancedSoftmax` | `run_019`-`run_021` | 0.362 | 0.313 | **0.448** | 0.621 | **0.713** |
| Weighted reference | `run_034`-`run_036` | 0.342 | **0.378** | **0.449** | 0.599 | **0.726** |
| Qwen rerun | `run_042`-`run_044` | **0.370** | 0.318 | 0.436 | **0.591** | 0.691 |

**Main read:** the Qwen family is real. It no longer looks like a one-off
single-seed curiosity. Across three seeds, it reaches incumbent-level median
QWK and `recall_-1` while lowering hedging. But the edge is too small and too
selective to justify an immediate default swap.

## 3. What Improved

**1. The encoder branch survived multi-seed rerun.** `run_042` reached
`qwk_mean 0.378`, `run_043` reproduced the original near-miss at `0.356`, and
`run_044` closed at `0.370`. That yields a family median `qwk_mean 0.370`,
slightly above the incumbent median `0.362` and well above the weighted
reference median `0.342`.

**2. Hedging stayed lower than the incumbent.** The Qwen family-median hedging
was `0.591` versus the incumbent `0.621`, with two seeds (`run_042`,
`run_043`) comfortably inside the lower-hedging range that made the original
Qwen diagnostic interesting.

**3. Tail recall remained viable instead of collapsing.** Median
`recall_-1 = 0.318` is essentially incumbent-level (`0.313`) and far stronger
than the conservative-family baselines. `run_044` also posted the best single
Qwen tail package with `recall_-1 0.370` and minority recall `0.446`.

## 4. Why It Still Falls Short of Promotion

**1. The gains are mostly comparable, not decisive.** The family-median
improvements over the incumbent are small enough to treat as comparable rather
than a clean win: `qwk_mean 0.370 vs 0.362`, `recall_-1 0.318 vs 0.313`,
hedging `0.591 vs 0.621`, minority recall `0.436 vs 0.448`, calibration
`0.691 vs 0.713`.

**2. Hard dimensions are still the blocker.** Qwen helps some dimensions but
not the ones that matter most for promotion. Family-median QWKs are:
- `stimulation 0.206` vs incumbent median `0.161` (better)
- `security 0.261` vs incumbent median `0.297` (worse)
- `hedonism 0.154` vs incumbent median `0.247` (materially worse)
- `power 0.149` vs incumbent median `0.334` (materially worse)

That profile is still too uneven to replace the incumbent as the active
default, even though the aggregate table now looks competitive.

**3. Structure and calibration do not clearly beat the incumbent.** Qwen keeps
good calibration, but family-median `calibration_global 0.691` is still below
the incumbent `0.713` and weighted `0.726`. Circumplex cleanliness is also not
strictly better: `run_042` and `run_043` both showed higher opposite-pair
violation than the incumbent range, even though `run_044` improved that side.

## 5. Selection and Stability Read

All three seeds selected eligible checkpoints cleanly without falling into the
SLACE-style conservative trap:

- `run_042`: selected epoch `27/40`, val `qwk 0.451`, `recall_-1 0.530`,
  holdout `qwk 0.378`
- `run_043`: selected epoch `24/29`, val `qwk 0.436`, `recall_-1 0.496`,
  holdout `qwk 0.356`
- `run_044`: selected epoch `28/33`, val `qwk 0.456`, `recall_-1 0.526`,
  holdout `qwk 0.370`

The gaps stayed in a familiar moderate range (`+0.163` to `+0.173`), so this
does not look like a fragile overfit branch. The branch is simply not dominant
enough on the hard dimensions to displace the incumbent.

## 6. Recommendation

**Do not promote Qwen to the active default yet, but move it onto the current
frontier board as the strongest encoder-swap challenger.**

- Keep `run_019`-`run_021` as the active corrected-split default.
- Keep `run_034`-`run_036` as the best tail-sensitive reference branch.
- Treat `run_042`-`run_044` as the strongest representation challenger and the
  first encoder family worth re-opening if representation work resumes.

If future work revisits encoder choice, Qwen should now be the first branch to
try, not a speculative revisit.

## 7. Summary Verdict

`twinkl-744` turned Qwen from a single-seed near-miss into a legitimate family.
That is a meaningful result. But the family still does not deliver a decisive
frontier change because the hard-dimension package remains too weak on
`hedonism` and `power`, and the family-median aggregate gains are only
comparable to the incumbent rather than clearly superior.

## 8. Artifacts

- Run configs: `config/experiments/vif/twinkl_744_qwen_seed11.yaml`,
  `config/experiments/vif/twinkl_744_qwen_seed22.yaml`,
  `config/experiments/vif/twinkl_744_qwen_seed33.yaml`
- Run logs: `logs/experiments/runs/run_042_BalancedSoftmax.yaml`,
  `logs/experiments/runs/run_043_BalancedSoftmax.yaml`,
  `logs/experiments/runs/run_044_BalancedSoftmax.yaml`
- Artifact roots:
  - `logs/experiments/artifacts/ordinal_v4_s2025_m11_20260318_223948/BalancedSoftmax/`
  - `logs/experiments/artifacts/ordinal_v4_s2025_m22_20260318_224404/BalancedSoftmax/`
  - `logs/experiments/artifacts/ordinal_v4_s2025_m33_20260318_224807/BalancedSoftmax/`
