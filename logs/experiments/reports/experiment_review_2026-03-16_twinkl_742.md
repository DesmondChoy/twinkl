# Experiment Review — 2026-03-16 — `twinkl-742` Qwen3-Embedding-0.6B Final Encoder Diagnostic

## 1. Experiment Overview

`twinkl-742` tested the last encoder-swap hypothesis left open after the Nomic
diagnostics: keep the active frontier critic budget fixed, but replace the
incumbent `nomic-embed-text-v1.5` encoder with `Qwen/Qwen3-Embedding-0.6B`
using the model's native prompt path and native `256d` truncation.

**What varied:**
- **Encoder family:** `nomic-ai/nomic-embed-text-v1.5` / `nomic-ai/nomic-embed-text-v2-moe` → `Qwen/Qwen3-Embedding-0.6B`
- **Input contract:** `classification: ` / `search_document: ` prefixing → custom native Qwen prompt
- **Context window:** `8192` / `512` → `32768`, eliminating truncation on the full corpus

**What stayed constant:** `BalancedSoftmax`, `window_size=1`, `hidden_dim=64`,
`dropout=0.3`, `batch_size=16`, `epochs=100`, explicit LR
`0.015522253574270487`, `split_seed=2025`, `model_seed=22`, frozen holdout
persona manifest, and the guarded single-seed checkpoint selection policy.

**Dataset note:** `run_040` uses the same refreshed `n_train=1,213` corrected
split as the post-`twinkl-731` / `twinkl-732` diagnostics. The incumbent
`run_020` still used the older `1,022`-sample training split, so direct
comparisons should focus on holdout behavior, not raw training-size parity.
Unlike `run_039`, the Qwen run truncates `0/1,651` entries.

## 2. Head-to-Head Comparison

### Aggregate: Qwen Diagnostic vs Incumbent and Prior Representation Controls

| Metric | `run_020` v1.5-256d | `run_037` v1.5-768d | `run_038` v1.5-768d hd28 | `run_039` v2-moe-256d | `run_040` Qwen-256d |
|--------|---:|---:|---:|---:|---:|
| Params | **23,454** | 56,222 | 23,606 | **23,454** | **23,454** |
| `state_dim` | **266** | 778 | 778 | **266** | **266** |
| `pct_truncated` | 0.0% | 0.0% | 0.0% | 0.8% | **0.0%** |
| MAE | **0.304** | 0.319 | 0.333 | 0.325 | 0.324 |
| Accuracy | **0.755** | 0.739 | 0.737 | 0.737 | 0.740 |
| QWK | **0.378** | 0.318 | 0.299 | 0.305 | **0.356** |
| Spearman | **0.359** | 0.332 | 0.301 | 0.336 | 0.343 |
| Calibration | **0.713** | **0.720** | 0.657 | 0.691 | 0.691 |
| Minority Recall | **0.449** | 0.381 | 0.346 | 0.433 | 0.436 |
| `recall_-1` | **0.342** | 0.269 | 0.246 | **0.336** | 0.296 |
| Hedging | 0.621 | 0.620 | 0.612 | 0.598 | **0.585** |

**Main read:** `run_040` is the strongest non-incumbent representation
diagnostic so far, but it still does not replace the frontier. It recovers much
of the broad-structure loss seen in `run_039` and removes truncation entirely,
yet it still trails incumbent `run_020` on holdout `qwk_mean`, accuracy,
calibration, minority recall, and especially `recall_-1`.

## 3. What Improved and What Did Not

**1. Qwen clearly beats the earlier encoder-swap controls on aggregate QWK.**
Relative to `run_039`, `run_040` improves holdout `qwk_mean` from `0.305` to
`0.356` while keeping the exact same parameter budget (`23,454`) and state
width (`266`). It also beats both 768d v1.5 controls by `+0.038` to `+0.056`
QWK while preserving the fair-budget setup.

**2. The zero-truncation result confirms that context length was only part of
the v2-moe story.** `run_040` uses a `32768`-token context window and truncates
`0/1,651` entries, so the remaining gap to `run_020` cannot be blamed on
context loss alone. Fixing truncation helped, but it did not solve the
frontier's hard polarity boundary problem by itself.

**3. Qwen recovers broad structure better than v2-moe, but the tail package is
still weaker than the incumbent.** The Qwen run recovers `power qwk`
(`0.117 -> 0.187`), `self_direction qwk` (`0.441 -> 0.555`), `conformity qwk`
(`0.454 -> 0.524`), and `benevolence qwk` (`0.307 -> 0.399`) relative to
`run_039`. That broad recovery is why overall QWK rebounds so strongly.
However, `recall_-1` still falls short of both incumbent `run_020` and the
v2-moe diagnostic (`0.296` vs `0.342` / `0.336`), so the tail-sensitive case is
not strong enough for promotion.

**4. The original hard dimensions still do not come back cleanly.** `security`
stays better than incumbent (`0.261` vs `0.213`), but `stimulation` remains far
below the incumbent (`0.165` vs `0.303`) and `hedonism` degrades further
(`0.095` vs incumbent `0.262` and v2-moe `0.188`). This is the clearest reason
not to treat `run_040` as a frontier replacement despite its stronger aggregate
QWK.

## 4. Dimension-Level Trade-offs

### Target Hard Dimensions

| Metric | `run_020` | `run_039` | `run_040` |
|--------|---:|---:|---:|
| Stimulation QWK | **0.303** | 0.168 | 0.165 |
| Hedonism QWK | **0.262** | **0.188** | 0.095 |
| Security QWK | 0.213 | **0.284** | 0.261 |

**Security remains the one clear bright spot across the representation swaps.**
Qwen keeps `security qwk` above the incumbent and close to the v2-moe peak,
while also reducing overall hedging to the lowest level in the five-run
comparison (`58.5%`).

**But the polarity-sensitive package is still not promotable.** `stimulation`
does not recover at all, and `hedonism` becomes the weakest of the three
single-seed encoder diagnostics. That means the encoder is helping some broad
structure while still missing the specific semantic separations that matter for
the frontier's hardest dimensions.

### Broader Recoveries vs `run_039`

The broad-structure win is real:

- `self_direction qwk` rises from `0.441` to `0.555`
- `conformity qwk` rises from `0.454` to `0.524`
- `benevolence qwk` rises from `0.307` to `0.399`
- `power qwk` rises from `0.117` to `0.187`

That explains why `qwk_mean` rebounds so much even though the target hard
dimensions do not.

## 5. Recommendation

**Do not promote `run_040` and do not delay `twinkl-733` for an immediate Qwen
rerun.**

- Keep `run_019`-`run_021` as the active corrected-split default.
- Keep `run_034`-`run_036` as the best tail-sensitive reference branch.
- Treat `run_040` as the best alternative representation diagnostic so far, but
  still short of the frontier because it gives back too much `recall_-1` and
  hard-dimension polarity.

**What changes from the earlier representation conclusion is narrower than a
frontier swap.** Qwen is the only encoder-swap branch that meaningfully
improved on `run_039` and materially narrowed the QWK gap to `run_020`
(`0.305 -> 0.356`, now only `-0.022` behind the incumbent). So if
representation work is revisited later, Qwen is the branch worth reopening, not
v2-moe or 768d v1.5. But it is still not strong enough to justify another
encoder-focused detour ahead of `twinkl-733`.

## 6. Summary Verdict

`run_040` is a meaningful positive diagnostic but not a new frontier. It
removes truncation, keeps the incumbent parameter budget fixed, and recovers
most of the broad-structure loss introduced by the earlier Nomic swaps. Even
so, it still trails the incumbent on the metrics that decide promotion, and the
target hard dimensions remain too weak. The active frontier does not change.
