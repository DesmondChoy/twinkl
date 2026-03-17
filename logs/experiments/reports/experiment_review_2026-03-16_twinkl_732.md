# Experiment Review — 2026-03-16 — `twinkl-732` Nomic v2-MoE 256d Frontier Diagnostic

## 1. Experiment Overview

`twinkl-732` tested the cleaner within-family encoder-swap hypothesis left open
after `twinkl-731`: keep the active frontier critic budget fixed, but replace
`nomic-embed-text-v1.5` with `nomic-embed-text-v2-moe` at `256d`.

**What varied:**
- **Encoder family member:** `nomic-ai/nomic-embed-text-v1.5` → `nomic-ai/nomic-embed-text-v2-moe`
- **Task prefix:** `classification: ` → `search_document: `
- **Effective max sequence length:** `8192` → `512`, producing light real-data truncation

**What stayed constant:** `BalancedSoftmax`, `window_size=1`, `hidden_dim=64`,
`dropout=0.3`, `batch_size=16`, `epochs=100`, explicit LR
`0.015522253574270487`, `split_seed=2025`, `model_seed=22`, frozen holdout
persona manifest, and the guarded single-seed checkpoint selection policy.

**Dataset note:** `run_039` uses the same refreshed corpus size as the `twinkl-731`
controls (`n_train=1,213`) rather than the older incumbent training size
(`1,022`). The v2-moe context limit truncates only `13/1,651` entries (`0.8%`),
so this is a measurable caveat but not large enough to explain the full
performance regression on its own.

## 2. Head-to-Head Comparison

### Aggregate: v2-MoE Diagnostic vs Incumbent and 768d Controls

| Metric | `run_020` v1.5-256d | `run_037` v1.5-768d | `run_038` v1.5-768d hd28 | `run_039` v2-moe-256d |
|--------|---:|---:|---:|---:|
| Params | **23,454** | 56,222 | 23,606 | **23,454** |
| `n_train` | 1,022 | 1,213 | 1,213 | 1,213 |
| `state_dim` | **266** | 778 | 778 | **266** |
| `pct_truncated` | 0.0% | 0.0% | 0.0% | 0.8% |
| MAE | **0.304** | 0.319 | 0.333 | 0.325 |
| Accuracy | **0.755** | 0.739 | 0.737 | 0.737 |
| QWK | **0.378** | 0.318 | 0.299 | 0.305 |
| Spearman | **0.359** | 0.332 | 0.301 | 0.336 |
| Calibration | **0.713** | **0.720** | 0.657 | 0.691 |
| Minority Recall | **0.449** | 0.381 | 0.346 | 0.433 |
| `recall_-1` | **0.342** | 0.269 | 0.246 | 0.336 |
| Hedging | 0.621 | 0.620 | 0.612 | **0.598** |

**Main read:** `run_039` is a better representation diagnostic than the 768d
controls, but still a negative frontier result. It nearly matches incumbent
`recall_-1` (`0.342 -> 0.336`) and lowers hedging (`0.621 -> 0.598`), yet it
still loses badly on holdout `qwk_mean` (`0.378 -> 0.305`), accuracy
(`0.755 -> 0.737`), calibration (`0.713 -> 0.691`), and minority recall
(`0.449 -> 0.433`).

## 3. What Improved and What Did Not

**1. v2-MoE clearly beats the two 768d controls on the tail-sensitive package.**
Relative to `run_037` and `run_038`, `run_039` improves `recall_-1`
(`0.336` vs `0.269` / `0.246`), minority recall (`0.433` vs `0.381` / `0.346`),
and hedging (`0.598` vs `0.620` / `0.612`). This means the v2-moe swap is not
just “another bad representation” in exactly the same way as the full-width
v1.5 line.

**2. That improvement still does not rescue the aggregate frontier case.**
`run_039` only clears the fair-budget `run_038` by `+0.005` QWK and still trails
the unconstrained `run_037` by `-0.013`, while remaining far behind incumbent
`run_020` (`-0.073`). A run that improves tail behavior but cannot get back into
the `~0.36+` QWK band is not strong enough to justify a 3-seed family rerun.

**3. The regression is not just a single hard-dimension issue.**
The gains are concentrated in `security` and overall hedging, but several other
dimensions degrade enough to erase that benefit.

## 4. Dimension-Level Trade-offs

### Target Hard Dimensions

| Metric | `run_020` | `run_037` | `run_038` | `run_039` |
|--------|---:|---:|---:|---:|
| Stimulation QWK | **0.303** | 0.216 | 0.197 | 0.168 |
| Hedonism QWK | **0.262** | 0.178 | -0.045 | 0.188 |
| Security QWK | 0.213 | 0.165 | 0.225 | **0.284** |

**Security is the one real bright spot.** `run_039` produces the strongest
`security` QWK of the whole representation-diagnostic set (`0.284`), along with
better `security` calibration (`0.530`) and lower `security` hedging (`0.679`)
than the incumbent (`0.706` hedging).

**But the hard-dimension package is still not clean enough.** `hedonism` only
partially recovers from the 768d failures and still misses the incumbent
(`0.262 -> 0.188`), while `stimulation` becomes the weakest of the three
representation variants (`0.303 -> 0.168`).

### Broader Regressions

The bigger problem is that the losses spread beyond the target dimensions:

- `power qwk` collapses from `0.342` to `0.117`
- `self_direction qwk` falls from `0.541` to `0.441`
- `conformity qwk` falls from `0.577` to `0.454`
- `benevolence qwk` falls from `0.367` to `0.307`

That pattern is why the aggregate QWK does not recover even though `security`
looks better.

## 5. Truncation, Capacity, and Causal Read

`run_039` keeps the exact incumbent parameter budget (`23,454`) and state width
(`266`), so this is the cleanest representation-only comparison we have run so
far. It also benefits from the expanded `1,213`-sample training set used by the
post-`twinkl-731` diagnostics. Despite those advantages, it still regresses on
the metrics that decide the frontier.

The shorter context window is real, but the observed truncation rate is small:
`13/1,651` entries (`0.8%`). That is enough to note in the review, but not
enough to explain a broad drop across `power`, `stimulation`,
`self_direction`, and `conformity` all at once. The cleaner conclusion is that
v2-moe changes the representation in a way that helps `security` and reduces
hedging, but does not preserve the overall decision boundary quality that made
`run_020` promotable.

## 6. Calibration and Hedging

Calibration remains usable: global error-uncertainty correlation is `0.691`
with `10/10` positively calibrated dimensions. Hedging is actually the best of
the four compared runs at `59.8%`, edging below the weighted family median
threshold of `~60%`.

That matters, but it is not enough. This run is another reminder that lower
hedging by itself is not a frontier win if it comes with weaker ordinal
discrimination and broad QWK erosion.

## 7. Recommendation

**Treat `twinkl-732` as a negative representation diagnostic.**

- Do **not** run a full 3-seed v2-moe family rerun.
- Keep `run_019`-`run_021` as the active corrected-split default.
- Keep `run_034`-`run_036` as the best tail-sensitive reference branch.
- Move the roadmap forward to `twinkl-733` rather than spending more time on
  encoder-family swaps without a sharper hypothesis.

If representation work is revisited later, it should be driven by a targeted
explanation for the `security` gain without the simultaneous collapse in
`power` and `stimulation`, not by a generic “newer encoder might help” premise.

## 8. Summary Verdict

`run_039` is the cleanest apples-to-apples encoder-swap test in the Nomic
family, and it still does not move the frontier. It beats the 768d controls on
tail recovery, minority recall, and hedging, but it remains materially worse
than incumbent `run_020` on aggregate QWK and several key dimensions. The active
frontier does not change.
