# VIF Experiment Index

## Current Frontier (Post-d937094 Split)

| Rank | Candidate | Runs | Split Seed | Model Seeds | Median QWK | Median recall_-1 | Median MinR | Median Hedging | Median Cal | Positioning |
|------|-----------|------|-----------:|------------|-----------:|-----------------:|------------:|---------------:|-----------:|-------------|
| 1 | BalancedSoftmax | run_019-run_021 | 2025 | 11, 22, 33 | 0.362 | 0.313 | **0.448** | 0.621 | **0.713** | Active corrected-split default. Still the best all-around package once hard-dimension stability, minority recall, and calibration are considered together. |
| 2 | Qwen3-0.6B + BalancedSoftmax | run_042-run_044 | 2025 | 11, 22, 33 | **0.370** | 0.318 | 0.436 | **0.591** | 0.691 | Strongest encoder-swap challenger so far. Family medians are comparable to the incumbent on QWK and `recall_-1` while reducing hedging, but `hedonism` / `power` stay too weak for a clean default swap. |
| 3 | BalancedSoftmax + dimweight | run_034-run_036 | 2025 | 11, 22, 33 | 0.342 | **0.378** | **0.449** | 0.599 | **0.726** | Best tail-sensitive reference branch so far. Strongest `recall_-1` / minority-recall package with lower hedging and better calibration than the incumbent, but median QWK is too volatile and too low to replace the default. |
| 4 | BalancedSoftmax + circreg + recall floor | run_031-run_033 | 2025 | 11, 22, 33 | 0.366 | 0.267 | 0.409 | 0.641 | 0.713 | QWK/calibration are comparable to the incumbent, but the guardrailed rerun still loses on `recall_-1`, minority recall, and hedging. Keep as a reference branch, not the default. |
| 5 | BalancedSoftmax + targeted batch | run_022-run_024 | 2025 | 11, 22, 33 | 0.349 | 0.342 | 0.434 | 0.619 | 0.687 | Best targeted hard-dimension follow-up before weighting. Improves `recall_-1`, but gives back QWK and calibration relative to the default family. |
| 6 | BalancedSoftmax + hedonism/security lift | run_025-run_027 | 2025 | 11, 22, 33 | 0.346 | 0.328 | 0.442 | 0.598 | 0.693 | Lowest-hedging training-time branch, but still not a clean frontier change on QWK or hard-dimension stability. |
| 7 | TwoStageBalancedSoftmax | run_045-run_047 | 2025 | 11, 22, 33 | 0.360 | 0.266 | 0.382 | 0.708 | 0.743 | Structurally promising reformulation. It keeps comparable QWK and the best calibration, and it materially helps `stimulation`, but it still hedges too much and gives back too much `recall_-1` to replace the frontier. |
| 8 | CDWCE_a3 | run_016-run_018 | 2025 | 11, 22, 33 | 0.353 | 0.104 | 0.276 | 0.804 | 0.762 | Best conservative 3-seed baseline when MAE, accuracy, and calibration matter more than strong tail recovery. |
| 9 | BalancedSoftmax + circumplex regularizer | run_028-run_030 | 2025 | 11, 22, 33 | 0.347 | 0.265 | 0.411 | 0.641 | 0.709 | Soft circumplex regularization improved some aggregate structure but weakened the tail-sensitive behavior that justified BalancedSoftmax in the first place. |
| 10 | SoftOrdinal | run_016-run_018 | 2025 | 11, 22, 33 | 0.346 | 0.077 | 0.283 | 0.796 | 0.781 | Best low-gap comparator. Competitive on QWK, but it remains much more neutral-biased than the BalancedSoftmax branches. |
| 11 | CORN | run_016-run_018 | 2025 | 11, 22, 33 | 0.315 | 0.089 | 0.273 | 0.801 | 0.818 | Best-calibrated corrected-split baseline. Keep it as the calibration anchor and sanity check for post-hoc calibration follow-ups. |
| 12 | SoftOrdinal + hedonism/security lift | run_025-run_027 | 2025 | 11, 22, 33 | 0.340 | 0.082 | 0.260 | 0.823 | 0.738 | Post-lift SoftOrdinal comparator. The extra data did not help it escape excessive hedging. |

> **Active recommendation (2026-03-19):** `run_019`-`run_021` remain the default corrected-split frontier family. The new two-stage reformulation family `run_045`-`run_047` reached comparable median holdout `qwk_mean` (`0.360` vs `0.362`) and the best calibration on the current board (`0.743`), but it gave back too much `recall_-1` (`0.266`) and hedging (`0.708`). `run_034`-`run_036` remain the best tail-sensitive reference branch, and `run_042`-`run_044` remain the strongest encoder-swap challenger.
>
> **Latest consensus-label retrain diagnostic review:** [`reports/experiment_review_2026-04-01_twinkl_754_6.md`](reports/experiment_review_2026-04-01_twinkl_754_6.md) did **not** change the frontier. `run_048`-`run_050` improved median consensus-label holdout QWK to `0.372` and calibration to `0.770`, but the branch changed the evaluation labels on the same frozen holdout (`17` `hedonism`, `27` `security`, and `13` `stimulation` test entries were relabeled), so it is not a like-for-like replacement for the persisted-label board. Inside that diagnostic regime it still regressed on `recall_-1` (`0.270`), minority recall (`0.408`), and `hedonism qwk` (`0.104` median). Keep `run_019`-`run_021` active, and treat consensus labels as a diagnostic branch until a same-code persisted-label sibling rerun or soft-label variant is evaluated.
>
> **Latest two-stage reformulation review:** [`reports/experiment_review_2026-03-19_twinkl_746.md`](reports/experiment_review_2026-03-19_twinkl_746.md) did **not** change the frontier. `TwoStageBalancedSoftmax` improved activation precision, kept active-sign accuracy competitive, and materially lifted `stimulation`, but the branch still missed too many active hard-dimension cases: family-median `recall_-1` fell from incumbent `0.313` to `0.266`, minority recall fell to `0.382`, and hedging rose to `70.8%`. Treat it as a useful structural diagnostic branch rather than a new default or reserve replacement.
>
> **Latest controlled Qwen frontier rerun review:** [`reports/experiment_review_2026-03-18_twinkl_744.md`](reports/experiment_review_2026-03-18_twinkl_744.md) did **not** change the active default, but it did materially upgrade the status of the encoder branch. `run_042`-`run_044` confirmed that Qwen is a legitimate family rather than a one-off near-miss: median holdout `qwk_mean` reached `0.370`, median `recall_-1` stayed at `0.318`, and hedging stayed lower than the incumbent. But the family still gives back too much on `hedonism` and `power`, so keep it as the strongest representation challenger rather than the new default.
>
> **Latest SLACE reserve-branch diagnostic review:** [`reports/experiment_review_2026-03-18_twinkl_734.md`](reports/experiment_review_2026-03-18_twinkl_734.md) did **not** change the frontier. `run_041` matched the incumbent parameter budget and improved surface metrics like MAE / accuracy, but it failed the actual frontier gates: holdout `qwk_mean` stayed below the incumbent family median (`0.338` vs `0.362`), `recall_-1` collapsed (`0.134` vs `0.313` incumbent, `0.378` weighted reference), and hedging rebounded to `0.810`, back in the conservative-family regime. Treat `twinkl-734` as completed and drop SLACE from the active roadmap.
>
> **Latest semantic counterexample closeout:** [`reports/experiment_review_2026-03-16_twinkl_733.md`](reports/experiment_review_2026-03-16_twinkl_733.md) explicitly de-scopes the proposed `Hedonism` / `Security` semantic counterexample batch. The repeated replay errors, earlier targeted lifts, uncertainty review, and later representation diagnostics already answer the core question strongly enough, so another generate -> judge -> retrain cycle is not justified for the current frontier. Keep `run_019`-`run_021` active, treat `twinkl-734` as tracker-unblocked but still reserve-only, and reopen `twinkl-733` only if a future branch needs a tightly matched causal falsification test.
>
> **Latest Qwen final diagnostic review:** [`reports/experiment_review_2026-03-16_twinkl_742.md`](reports/experiment_review_2026-03-16_twinkl_742.md) also did **not** change the frontier, but it is the first encoder swap that materially narrowed the gap. `run_040` kept the incumbent parameter budget and state width fixed, removed truncation entirely, and recovered holdout `qwk_mean` from `0.305` (`run_039`) to `0.356`, bringing it within `0.022` of incumbent `run_020` while lowering hedging to `0.585`. It still trails the incumbent on `recall_-1` (`0.296` vs `0.342`) and the target hard dimensions remain weak (`stimulation qwk 0.165`, `hedonism qwk 0.095`), so there is no immediate promotion or rerun recommendation. Keep `run_019`-`run_021` active, and treat Qwen as the only encoder branch worth reconsidering later if representation work is reopened.
>
> **Latest Nomic v2-MoE 256d diagnostic review:** [`reports/experiment_review_2026-03-16_twinkl_732.md`](reports/experiment_review_2026-03-16_twinkl_732.md) did **not** change the frontier. The cleaner within-family encoder swap `run_039` matched the incumbent parameter budget and recovered much of the tail-sensitive package relative to the `twinkl-731` 768d controls (`recall_-1 0.336`, minority recall `0.433`, hedging `0.598`, `security qwk 0.284`), but it still regressed sharply against incumbent `run_020` on holdout `qwk_mean` (`0.378 -> 0.305`), accuracy (`0.755 -> 0.737`), and several dimensions, especially `power` (`0.342 -> 0.117`) and `stimulation` (`0.303 -> 0.168`). Treat this as another negative representation diagnostic: no 3-seed v2-moe rerun, and no need to reopen `twinkl-733` just to reconfirm the already-established semantic-polarity diagnosis.
>
> **Latest weighted frontier review:** [`reports/experiment_review_2026-03-11_twinkl_719_3.md`](reports/experiment_review_2026-03-11_twinkl_719_3.md)
>
> **Latest post-hoc retargeting review:** [`reports/experiment_review_twinkl_729.md`](reports/experiment_review_twinkl_729.md) did **not** change the frontier. The new effective-prior + per-dimension tau branch did not beat the standard Menon control on the guarded median test comparison: standard Menon still delivered the better median `recall_-1` delta (`+0.004` vs baseline) while both branches gave back QWK, calibration, and circumplex cleanliness. On the strongest weighted checkpoint `run_036`, the new branch was validation-selected but regressed on holdout (`qwk_mean 0.381 -> 0.360`, `recall_-1 0.387 -> 0.339`, `calibration 0.726 -> 0.592`). Treat this as evidence that the current post-hoc line is likely exhausted for the active frontier; `run_034`-`run_036` remain the best tail-sensitive reference branch, and any further incumbent-centered follow-up should be a materially different intervention such as `twinkl-719.6`.
>
> **Latest frontier uncertainty review:** [`reports/experiment_review_2026-03-14_twinkl_730.md`](reports/experiment_review_2026-03-14_twinkl_730.md) adds persona-cluster bootstrap BCa intervals and hard-dimension chance tests for the active BalancedSoftmax frontier families. The weighted reference branch still shows a statistically distinguishable family-median `recall_-1` gain over the incumbent (`+0.065`, 95% BCa CI `[+0.021, +0.128]`), but its QWK delta remains unresolved noise (`-0.020`, CI `[-0.062, +0.018]`) and minority recall is effectively flat. Future promotion decisions should therefore gate on family-delta intervals, not point medians, and treat hard-dimension QWK as secondary evidence only when it is itself above chance.
>
> **Latest 768d truncation diagnostic review:** [`reports/experiment_review_2026-03-16_twinkl_731.md`](reports/experiment_review_2026-03-16_twinkl_731.md) did **not** change the frontier. The unconstrained full-768d diagnostic `run_037` regressed against incumbent `run_020` on holdout `qwk_mean` (`0.378 -> 0.318`), `recall_-1` (`0.342 -> 0.269`), minority recall (`0.449 -> 0.381`), and both target hard-dimension QWKs (`hedonism 0.262 -> 0.178`, `security 0.213 -> 0.165`). The fair-budget control `run_038` matched the incumbent parameter budget more closely but degraded further (`qwk_mean 0.299`, `recall_-1 0.246`) while collapsing `hedonism qwk` to `-0.045`. Treat this as the first negative representation diagnostic in the pair completed by `twinkl-732`.
>
> **Latest full frontier review:** [`reports/experiment_review_2026-03-11_twinkl_721.md`](reports/experiment_review_2026-03-11_twinkl_721.md)
>
> **Latest checkpoint-selection guardrail review:** [`reports/experiment_review_2026-03-10_twinkl_715.md`](reports/experiment_review_2026-03-10_twinkl_715.md)
>
> **Circumplex rollout closeout:** [`reports/experiment_review_2026-03-11_twinkl_691_5.md`](reports/experiment_review_2026-03-11_twinkl_691_5.md) explicitly de-scopes the circumplex-aware batch sampler. The diagnostics remain useful, and the weighted rerun now supplies the training-time comparison branch for the next validation-only logit-retargeting follow-up from `run_020`.
>
> **Previous full frontier review:** [`reports/experiment_review_2026-03-10_v8.md`](reports/experiment_review_2026-03-10_v8.md)
>
> **Latest post-lift rebaseline review:** [`reports/experiment_review_2026-03-09_twinkl_691_3.md`](reports/experiment_review_2026-03-09_twinkl_691_3.md)
>
> **Latest targeted data-lift review:** [`reports/experiment_review_2026-03-08_twinkl_681_5.md`](reports/experiment_review_2026-03-08_twinkl_681_5.md)
>
> **Post-hoc reporting:** these `681.3` results are documented in [`reports/experiment_review_2026-03-07_twinkl_681_3.md`](reports/experiment_review_2026-03-07_twinkl_681_3.md) with persisted artifacts under [`artifacts/posthoc_twinkl_681_3_20260307_142717/`](artifacts/posthoc_twinkl_681_3_20260307_142717/). They are not added as new rows in the auto-generated run log because no retraining was performed.
>
> **Evaluation hygiene:** the board above is based on the corrected persona-stratified split introduced after commit `d937094`. Compare runs within this regime first. Use the historical board below for context only, not for direct SOTA claims.

## Historical Frontier (Pre-d937094 Split)

| Rank | Run + Loss | Encoder | hd | n_train | QWK | Cal | MAE | Acc | MinR | Rationale |
|------|-----------|---------|---:|--------:|----:|----:|----:|----:|-----:|-----------|
| 1 | run_010 CORN | nomic-256d | 64 | 1020 | **0.434** | 0.835 | **0.206** | 0.821 | 0.285 | Pre-split SOTA since run_010. LR=0.001 (configured) is optimal for CORN; LR finder consistently overshoots. |
| 2 | run_015 CDWCE_a3 | nomic-256d | 64 | 1020 | 0.402 | 0.755 | **0.203** | **0.822** | 0.259 | Best new loss from CDW-CE alpha sweep. Alpha=3 is the sweet spot; alpha=2 too weak, alpha=5 collapses. QWK approaches SOTA but trades calibration. |
| 3 | run_007 CORN | nomic-256d | 64 | 1020 | 0.413 | 0.838 | 0.205 | 0.821 | 0.285 | Previous leader; identical config to run_010 CORN (stochastic variance explains QWK gap). |

> **Why this frontier is deprecated:** `run_001`-`run_015` were evaluated before commit `d937094` (`twinkl-675: stratify persona val/test splits`). They are retained as an archival record of the pre-fix search path, but they should not be used for active SOTA claims or direct comparison against `run_016+`.
>
> **What `d937094` fixed:** before the fix, `split_by_persona()` kept entries from the same persona together, but it still formed validation/test by randomly shuffling persona IDs and slicing them. That preserved persona isolation, but it did **not** preserve the distribution of per-dimension `+1` and `-1` signals across val/test. Commit `d937094` changed the split logic to build persona-level sign features for every Schwartz dimension, search deterministic candidate partitions, and score them against global prevalence while strongly penalizing val/test splits that drop expected minority signals.
>
> **Why pre- and post-fix runs are not comparable:** the split seed stayed the same (`2025`), but the partitioning algorithm changed, so the actual personas and label marginals in validation/test changed. That means pre-`d937094` runs were optimized and measured on a different evaluation regime. In practice, historical results could look better or worse simply because rare misalignment signals were under- or over-represented in the holdout sets, especially on volatile dimensions like `Power` and `Security`. The corrected-split frontier is therefore the only fair basis for current model ranking and future recommendations.

> **Key insight (run_015):** CDW-CE loss (Polat et al. 2025) was tested as an alpha sweep (2, 3, 5) — the #1 recommended intervention from run_014. **CDW-CE alpha=3 is competitive** (QWK 0.402, moderate) but **did not solve the recall_-1 bottleneck** (0.056 vs SOTA 0.089). The inverted-U alpha response confirms alpha=3 as the sweet spot: alpha=2 is too weak for 3-class ordinal (max |i−c|=2), alpha=5 over-penalizes causing extreme hedging (89.9%). **run_010 CORN remained the pre-split SOTA.**
>
> **Primary bottleneck in the pre-split regime (confirmed structural at the time):** `recall_minus1 = 1–9%` across the first 15 runs, 7 loss functions, and 3 LR regimes. The -1 class is only 7.3% of all labels (~75 examples per dimension). This supported the hypothesis that the bottleneck was largely **data scarcity**, not just loss function choice. At the time, the next recommendations were: (1) apply **post-hoc logit adjustment** (Menon et al. 2021, tau=0.3–1.0) on run_010 CORN and run_015 CDWCE_a3 for zero-cost recall_-1 gains; (2) rerun **CDWCE_a3 at configured LR=0.001** (disable LR finder); (3) try **CORN+SoftOrdinal ensemble** via probability averaging; (4) fix the **LR finder history bug**. The corrected-split frontier above supersedes these as the active baseline for future work. See full analysis in [`logs/experiments/reports/experiment_review_2026-03-04_v4.md`](reports/experiment_review_2026-03-04_v4.md).

## Run Log

<!-- AUTO-TABLE:START -->
| run | model | encoder | ws | hd | do | loss | params | ratio | MAE | Acc | QWK | Spear | Cal | MinR | OppV | AdjS | file |
|-----|-------|---------|---:|---:|---:|------|-------:|------:|----:|----:|----:|------:|----:|-----:|-----:|-----:|------|
| 001 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 372756 | 585.2 | 0.232 | 0.782 | 0.398 | 0.459 | 0.644 | 0.298 | N/A | N/A | runs/run_001_CORAL.yaml |
| 001 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 372756 | 585.2 | 0.236 | 0.782 | 0.384 | 0.452 | 0.633 | 0.306 | N/A | N/A | runs/run_001_CORN.yaml |
| 001 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 375326 | 589.2 | 0.243 | 0.773 | 0.395 | 0.459 | 0.648 | 0.340 | N/A | N/A | runs/run_001_EMD.yaml |
| 001 | MSE | MiniLM-384d | 3 | 256 | 0.2 | weighted_mse_s5.0 | 370186 | 581.1 | 0.450 | 0.641 | 0.338 | 0.379 | -0.218 | 0.428 | N/A | N/A | runs/run_001_MSE.yaml |
| 001 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 375326 | 589.2 | 0.248 | 0.777 | 0.417 | 0.455 | 0.724 | 0.372 | N/A | N/A | runs/run_001_SoftOrdinal.yaml |
| 002 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10708 | 16.8 | 0.263 | 0.770 | 0.335 | 0.349 | 0.734 | 0.234 | N/A | N/A | runs/run_002_CORAL.yaml |
| 002 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10708 | 16.8 | 0.260 | 0.766 | 0.355 | 0.371 | 0.737 | 0.204 | N/A | N/A | runs/run_002_CORN.yaml |
| 002 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 11038 | 17.3 | 0.270 | 0.764 | 0.365 | 0.365 | 0.772 | 0.309 | N/A | N/A | runs/run_002_EMD.yaml |
| 002 | MSE | nomic-256d | 1 | 32 | 0.3 | weighted_mse_s5.0 | 10378 | 16.3 | 0.418 | 0.683 | 0.348 | 0.378 | -0.073 | 0.390 | N/A | N/A | runs/run_002_MSE.yaml |
| 002 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 11038 | 17.3 | 0.267 | 0.780 | 0.385 | 0.356 | 0.774 | 0.310 | N/A | N/A | runs/run_002_SoftOrdinal.yaml |
| 003 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 370196 | 581.2 | 0.241 | 0.776 | 0.369 | 0.394 | 0.640 | 0.265 | N/A | N/A | runs/run_003_CORAL.yaml |
| 003 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 370196 | 581.2 | 0.249 | 0.764 | 0.326 | 0.395 | 0.648 | 0.260 | N/A | N/A | runs/run_003_CORN.yaml |
| 003 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 372766 | 585.2 | 0.240 | 0.782 | 0.410 | 0.416 | 0.697 | 0.332 | N/A | N/A | runs/run_003_EMD.yaml |
| 003 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 372766 | 585.2 | 0.253 | 0.772 | 0.383 | 0.396 | 0.743 | 0.333 | N/A | N/A | runs/run_003_SoftOrdinal.yaml |
| 004 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10388 | 16.3 | 0.273 | 0.768 | 0.331 | 0.346 | 0.755 | 0.269 | N/A | N/A | runs/run_004_CORAL.yaml |
| 004 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10388 | 16.3 | 0.274 | 0.760 | 0.291 | 0.293 | 0.734 | 0.217 | N/A | N/A | runs/run_004_CORN.yaml |
| 004 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 10718 | 16.8 | 0.266 | 0.780 | 0.391 | 0.361 | 0.766 | 0.343 | N/A | N/A | runs/run_004_EMD.yaml |
| 004 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 10718 | 16.8 | 0.269 | 0.779 | 0.385 | 0.351 | 0.786 | 0.332 | N/A | N/A | runs/run_004_SoftOrdinal.yaml |
| 005 | CORAL | MiniLM-384d | 3 | 256 | 0.2 | coral | 370196 | 386.4 | 0.215 | 0.798 | 0.282 | 0.411 | 0.629 | 0.186 | N/A | N/A | runs/run_005_CORAL.yaml |
| 005 | CORN | MiniLM-384d | 3 | 256 | 0.2 | corn | 370196 | 386.4 | 0.216 | 0.797 | 0.254 | 0.365 | 0.640 | 0.181 | N/A | N/A | runs/run_005_CORN.yaml |
| 005 | EMD | MiniLM-384d | 3 | 256 | 0.2 | emd | 372766 | 389.1 | 0.226 | 0.791 | 0.272 | 0.312 | 0.651 | 0.217 | N/A | N/A | runs/run_005_EMD.yaml |
| 005 | SoftOrdinal | MiniLM-384d | 3 | 256 | 0.2 | soft_ordinal | 372766 | 389.1 | 0.215 | 0.798 | 0.304 | 0.368 | 0.650 | 0.165 | N/A | N/A | runs/run_005_SoftOrdinal.yaml |
| 006 | CORAL | nomic-256d | 1 | 32 | 0.3 | coral | 10388 | 10.8 | 0.233 | 0.793 | 0.278 | 0.354 | 0.768 | 0.166 | N/A | N/A | runs/run_006_CORAL.yaml |
| 006 | CORN | nomic-256d | 1 | 32 | 0.3 | corn | 10388 | 10.8 | 0.236 | 0.791 | 0.280 | 0.350 | 0.777 | 0.183 | N/A | N/A | runs/run_006_CORN.yaml |
| 006 | EMD | nomic-256d | 1 | 32 | 0.3 | emd | 10718 | 11.2 | 0.227 | 0.803 | 0.324 | 0.334 | 0.764 | 0.225 | N/A | N/A | runs/run_006_EMD.yaml |
| 006 | SoftOrdinal | nomic-256d | 1 | 32 | 0.3 | soft_ordinal | 10718 | 11.2 | 0.223 | 0.811 | 0.358 | 0.354 | 0.775 | 0.235 | N/A | N/A | runs/run_006_SoftOrdinal.yaml |
| 007 | CORAL | nomic-256d | 1 | 64 | 0.3 | coral | 22804 | 22.4 | 0.208 | 0.819 | 0.367 | 0.398 | 0.830 | 0.247 | N/A | N/A | runs/run_007_CORAL.yaml |
| 007 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.4 | 0.205 | 0.821 | 0.413 | 0.402 | 0.838 | 0.285 | N/A | N/A | runs/run_007_CORN.yaml |
| 007 | EMD | nomic-256d | 1 | 64 | 0.3 | emd | 23454 | 23.0 | 0.211 | 0.817 | 0.357 | 0.363 | 0.849 | 0.288 | N/A | N/A | runs/run_007_EMD.yaml |
| 007 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 23.0 | 0.212 | 0.821 | 0.314 | 0.343 | 0.852 | 0.291 | N/A | N/A | runs/run_007_SoftOrdinal.yaml |
| 008 | CORAL | nomic-256d | 1 | 128 | 0.3 | coral | 53780 | 52.7 | 0.204 | 0.824 | 0.341 | 0.397 | 0.805 | 0.273 | N/A | N/A | runs/run_008_CORAL.yaml |
| 008 | CORN | nomic-256d | 1 | 128 | 0.3 | corn | 53780 | 52.7 | 0.203 | 0.828 | 0.344 | 0.359 | 0.785 | 0.276 | N/A | N/A | runs/run_008_CORN.yaml |
| 008 | EMD | nomic-256d | 1 | 128 | 0.3 | emd | 55070 | 54.0 | 0.200 | 0.821 | 0.365 | 0.390 | 0.802 | 0.300 | N/A | N/A | runs/run_008_EMD.yaml |
| 008 | SoftOrdinal | nomic-256d | 1 | 128 | 0.3 | soft_ordinal | 55070 | 54.0 | 0.201 | 0.826 | 0.354 | 0.387 | 0.811 | 0.291 | N/A | N/A | runs/run_008_SoftOrdinal.yaml |
| 009 | CORAL | MiniLM-384d | 3 | 64 | 0.2 | coral | 80276 | 78.7 | 0.228 | 0.785 | 0.176 | 0.284 | 0.695 | 0.137 | N/A | N/A | runs/run_009_CORAL.yaml |
| 009 | CORN | MiniLM-384d | 3 | 64 | 0.2 | corn | 80276 | 78.7 | 0.227 | 0.792 | 0.227 | 0.323 | 0.711 | 0.166 | N/A | N/A | runs/run_009_CORN.yaml |
| 009 | EMD | MiniLM-384d | 3 | 64 | 0.2 | emd | 80926 | 79.3 | 0.225 | 0.799 | 0.259 | 0.303 | 0.776 | 0.223 | N/A | N/A | runs/run_009_EMD.yaml |
| 009 | SoftOrdinal | MiniLM-384d | 3 | 64 | 0.2 | soft_ordinal | 80926 | 79.3 | 0.239 | 0.787 | 0.236 | 0.300 | 0.775 | 0.234 | N/A | N/A | runs/run_009_SoftOrdinal.yaml |
| 010 | CORAL | nomic-256d | 1 | 64 | 0.3 | coral | 22804 | 22.4 | 0.209 | 0.819 | 0.364 | 0.391 | 0.823 | 0.244 | N/A | N/A | runs/run_010_CORAL.yaml |
| 010 | CORAL_IW | nomic-256d | 1 | 64 | 0.3 | coral_iw | 22804 | 22.4 | 0.221 | 0.809 | 0.301 | 0.351 | 0.841 | 0.234 | N/A | N/A | runs/run_010_CORAL_IW.yaml |
| 010 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.4 | 0.206 | 0.821 | 0.434 | 0.407 | 0.835 | 0.285 | N/A | N/A | runs/run_010_CORN.yaml |
| 010 | EMD | nomic-256d | 1 | 64 | 0.3 | emd | 23454 | 23.0 | 0.212 | 0.819 | 0.362 | 0.357 | 0.851 | 0.294 | N/A | N/A | runs/run_010_EMD.yaml |
| 010 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 23.0 | 0.211 | 0.818 | 0.308 | 0.352 | 0.860 | 0.284 | N/A | N/A | runs/run_010_SoftOrdinal.yaml |
| 011 | CORAL | nomic-256d | 2 | 64 | 0.3 | coral | 39252 | 38.5 | 0.218 | 0.807 | 0.339 | 0.368 | 0.815 | 0.245 | N/A | N/A | runs/run_011_CORAL.yaml |
| 011 | CORAL_IW | nomic-256d | 2 | 64 | 0.3 | coral_iw | 39252 | 38.5 | 0.223 | 0.798 | 0.269 | 0.312 | 0.790 | 0.175 | N/A | N/A | runs/run_011_CORAL_IW.yaml |
| 011 | CORN | nomic-256d | 2 | 64 | 0.3 | corn | 39252 | 38.5 | 0.209 | 0.814 | 0.335 | 0.388 | 0.811 | 0.232 | N/A | N/A | runs/run_011_CORN.yaml |
| 011 | EMD | nomic-256d | 2 | 64 | 0.3 | emd | 39902 | 39.1 | 0.214 | 0.821 | 0.382 | 0.359 | 0.846 | 0.308 | N/A | N/A | runs/run_011_EMD.yaml |
| 011 | SoftOrdinal | nomic-256d | 2 | 64 | 0.3 | soft_ordinal | 39902 | 39.1 | 0.222 | 0.820 | 0.333 | 0.349 | 0.862 | 0.312 | N/A | N/A | runs/run_011_SoftOrdinal.yaml |
| 012 | CORAL | nomic-256d | 2 | 64 | 0.3 | coral | 39252 | 38.5 | 0.209 | 0.806 | 0.359 | 0.381 | 0.812 | 0.342 | N/A | N/A | runs/run_012_CORAL.yaml |
| 012 | CORAL_IW | nomic-256d | 2 | 64 | 0.3 | coral_iw | 39252 | 38.5 | 0.214 | 0.801 | 0.271 | 0.304 | 0.802 | 0.265 | N/A | N/A | runs/run_012_CORAL_IW.yaml |
| 012 | CORN | nomic-256d | 2 | 64 | 0.3 | corn | 39252 | 38.5 | 0.193 | 0.820 | 0.346 | 0.396 | 0.804 | 0.296 | N/A | N/A | runs/run_012_CORN.yaml |
| 012 | EMD | nomic-256d | 2 | 64 | 0.3 | emd | 39902 | 39.1 | 0.213 | 0.808 | 0.369 | 0.357 | 0.857 | 0.364 | N/A | N/A | runs/run_012_EMD.yaml |
| 012 | SoftOrdinal | nomic-256d | 2 | 64 | 0.3 | soft_ordinal | 39902 | 39.1 | 0.224 | 0.801 | 0.334 | 0.356 | 0.850 | 0.391 | N/A | N/A | runs/run_012_SoftOrdinal.yaml |
| 013 | CORAL | nomic-256d | 2 | 64 | 0.3 | coral | 39252 | 38.5 | 0.215 | 0.814 | 0.384 | 0.353 | 0.820 | 0.259 | N/A | N/A | runs/run_013_CORAL.yaml |
| 013 | CORN | nomic-256d | 2 | 64 | 0.3 | corn | 39252 | 38.5 | 0.208 | 0.821 | 0.382 | 0.388 | 0.828 | 0.269 | N/A | N/A | runs/run_013_CORN.yaml |
| 013 | EMD | nomic-256d | 2 | 64 | 0.3 | emd | 39902 | 39.1 | 0.209 | 0.823 | 0.391 | 0.371 | 0.840 | 0.293 | N/A | N/A | runs/run_013_EMD.yaml |
| 013 | SoftOrdinal | nomic-256d | 2 | 64 | 0.3 | soft_ordinal | 39902 | 39.1 | 0.212 | 0.820 | 0.334 | 0.367 | 0.830 | 0.269 | N/A | N/A | runs/run_013_SoftOrdinal.yaml |
| 014 | CORAL | nomic-256d | 1 | 64 | 0.3 | coral | 22804 | 22.4 | 0.206 | 0.821 | 0.329 | 0.391 | 0.813 | 0.237 | N/A | N/A | runs/run_014_CORAL.yaml |
| 014 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.4 | 0.207 | 0.819 | 0.314 | 0.367 | 0.753 | 0.246 | N/A | N/A | runs/run_014_CORN.yaml |
| 014 | EMD | nomic-256d | 1 | 64 | 0.3 | emd | 23454 | 23.0 | 0.203 | 0.818 | 0.373 | 0.406 | 0.785 | 0.274 | N/A | N/A | runs/run_014_EMD.yaml |
| 014 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 23.0 | 0.204 | 0.825 | 0.388 | 0.394 | 0.801 | 0.288 | N/A | N/A | runs/run_014_SoftOrdinal.yaml |
| 015 | CDWCE_a2 | nomic-256d | 1 | 64 | 0.3 | cdwce_a2 | 23454 | 23.0 | 0.207 | 0.811 | 0.322 | 0.350 | 0.783 | 0.220 | N/A | N/A | runs/run_015_CDWCE_a2.yaml |
| 015 | CDWCE_a3 | nomic-256d | 1 | 64 | 0.3 | cdwce_a3 | 23454 | 23.0 | 0.203 | 0.822 | 0.402 | 0.384 | 0.755 | 0.259 | N/A | N/A | runs/run_015_CDWCE_a3.yaml |
| 015 | CDWCE_a5 | nomic-256d | 1 | 64 | 0.3 | cdwce_a5 | 23454 | 23.0 | 0.217 | 0.795 | 0.300 | 0.371 | 0.639 | 0.174 | N/A | N/A | runs/run_015_CDWCE_a5.yaml |
| 015 | CORAL | nomic-256d | 1 | 64 | 0.3 | coral | 22804 | 22.4 | 0.205 | 0.822 | 0.349 | 0.388 | 0.824 | 0.241 | N/A | N/A | runs/run_015_CORAL.yaml |
| 015 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.4 | 0.203 | 0.815 | 0.328 | 0.397 | 0.801 | 0.234 | N/A | N/A | runs/run_015_CORN.yaml |
| 015 | EMD | nomic-256d | 1 | 64 | 0.3 | emd | 23454 | 23.0 | 0.204 | 0.821 | 0.372 | 0.405 | 0.781 | 0.280 | N/A | N/A | runs/run_015_EMD.yaml |
| 015 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 23.0 | 0.208 | 0.822 | 0.335 | 0.381 | 0.846 | 0.292 | N/A | N/A | runs/run_015_SoftOrdinal.yaml |
| 016 | CDWCE_a3 | nomic-256d | 1 | 64 | 0.3 | cdwce_a3 | 23454 | 22.9 | 0.224 | 0.802 | 0.355 | 0.373 | 0.760 | 0.266 | N/A | N/A | runs/run_016_CDWCE_a3.yaml |
| 016 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.3 | 0.230 | 0.810 | 0.315 | 0.338 | 0.821 | 0.274 | N/A | N/A | runs/run_016_CORN.yaml |
| 016 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 22.9 | 0.221 | 0.805 | 0.388 | 0.363 | 0.781 | 0.292 | N/A | N/A | runs/run_016_SoftOrdinal.yaml |
| 017 | CDWCE_a3 | nomic-256d | 1 | 64 | 0.3 | cdwce_a3 | 23454 | 22.9 | 0.231 | 0.799 | 0.353 | 0.334 | 0.779 | 0.294 | N/A | N/A | runs/run_017_CDWCE_a3.yaml |
| 017 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.3 | 0.218 | 0.815 | 0.315 | 0.356 | 0.818 | 0.266 | N/A | N/A | runs/run_017_CORN.yaml |
| 017 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 22.9 | 0.219 | 0.807 | 0.330 | 0.352 | 0.746 | 0.229 | N/A | N/A | runs/run_017_SoftOrdinal.yaml |
| 018 | CDWCE_a3 | nomic-256d | 1 | 64 | 0.3 | cdwce_a3 | 23454 | 22.9 | 0.229 | 0.796 | 0.338 | 0.365 | 0.762 | 0.276 | N/A | N/A | runs/run_018_CDWCE_a3.yaml |
| 018 | CORN | nomic-256d | 1 | 64 | 0.3 | corn | 22804 | 22.3 | 0.218 | 0.811 | 0.355 | 0.382 | 0.815 | 0.273 | N/A | N/A | runs/run_018_CORN.yaml |
| 018 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 22.9 | 0.220 | 0.811 | 0.346 | 0.353 | 0.798 | 0.283 | N/A | N/A | runs/run_018_SoftOrdinal.yaml |
| 019 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 22.9 | 0.299 | 0.753 | 0.362 | 0.365 | 0.727 | 0.399 | N/A | N/A | runs/run_019_BalancedSoftmax.yaml |
| 019 | LDAM_DRW | nomic-256d | 1 | 64 | 0.3 | ldam_drw | 23454 | 22.9 | 0.229 | 0.803 | 0.329 | 0.336 | 0.753 | 0.274 | N/A | N/A | runs/run_019_LDAM_DRW.yaml |
| 020 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 22.9 | 0.304 | 0.755 | 0.378 | 0.359 | 0.713 | 0.449 | N/A | N/A | runs/run_020_BalancedSoftmax.yaml |
| 020 | LDAM_DRW | nomic-256d | 1 | 64 | 0.3 | ldam_drw | 23454 | 22.9 | 0.216 | 0.812 | 0.358 | 0.348 | 0.762 | 0.296 | N/A | N/A | runs/run_020_LDAM_DRW.yaml |
| 021 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 22.9 | 0.324 | 0.734 | 0.358 | 0.371 | 0.654 | 0.448 | N/A | N/A | runs/run_021_BalancedSoftmax.yaml |
| 022 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 21.0 | 0.306 | 0.750 | 0.349 | 0.354 | 0.728 | 0.434 | N/A | N/A | runs/run_022_BalancedSoftmax.yaml |
| 023 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 21.0 | 0.312 | 0.748 | 0.372 | 0.344 | 0.685 | 0.450 | N/A | N/A | runs/run_023_BalancedSoftmax.yaml |
| 024 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 21.0 | 0.314 | 0.748 | 0.339 | 0.340 | 0.687 | 0.433 | N/A | N/A | runs/run_024_BalancedSoftmax.yaml |
| 025 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.311 | 0.756 | 0.346 | 0.345 | 0.711 | 0.411 | 0.092 | 0.070 | runs/run_025_BalancedSoftmax.yaml |
| 025 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 19.3 | 0.213 | 0.811 | 0.342 | 0.354 | 0.734 | 0.260 | 0.061 | 0.056 | runs/run_025_SoftOrdinal.yaml |
| 026 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.333 | 0.728 | 0.334 | 0.342 | 0.659 | 0.457 | 0.082 | 0.072 | runs/run_026_BalancedSoftmax.yaml |
| 026 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 19.3 | 0.213 | 0.807 | 0.322 | 0.382 | 0.738 | 0.233 | 0.069 | 0.055 | runs/run_026_SoftOrdinal.yaml |
| 027 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.319 | 0.737 | 0.351 | 0.372 | 0.693 | 0.442 | 0.072 | 0.085 | runs/run_027_BalancedSoftmax.yaml |
| 027 | SoftOrdinal | nomic-256d | 1 | 64 | 0.3 | soft_ordinal | 23454 | 19.3 | 0.216 | 0.812 | 0.340 | 0.365 | 0.777 | 0.267 | 0.072 | 0.058 | runs/run_027_SoftOrdinal.yaml |
| 028 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_circreg | 23454 | 19.3 | 0.293 | 0.761 | 0.384 | 0.362 | 0.761 | 0.422 | 0.031 | 0.075 | runs/run_028_BalancedSoftmax.yaml |
| 029 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_circreg | 23454 | 19.3 | 0.301 | 0.760 | 0.347 | 0.346 | 0.709 | 0.411 | 0.039 | 0.077 | runs/run_029_BalancedSoftmax.yaml |
| 030 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_circreg | 23454 | 19.3 | 0.308 | 0.751 | 0.318 | 0.362 | 0.702 | 0.398 | 0.043 | 0.095 | runs/run_030_BalancedSoftmax.yaml |
| 031 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_circreg | 23454 | 19.3 | 0.306 | 0.761 | 0.353 | 0.343 | 0.707 | 0.409 | 0.035 | 0.079 | runs/run_031_BalancedSoftmax.yaml |
| 032 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_circreg | 23454 | 19.3 | 0.307 | 0.752 | 0.366 | 0.342 | 0.713 | 0.435 | 0.037 | 0.075 | runs/run_032_BalancedSoftmax.yaml |
| 033 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_circreg | 23454 | 19.3 | 0.286 | 0.779 | 0.372 | 0.359 | 0.747 | 0.409 | 0.033 | 0.077 | runs/run_033_BalancedSoftmax.yaml |
| 034 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_dimweight | 23454 | 19.3 | 0.309 | 0.752 | 0.342 | 0.323 | 0.740 | 0.412 | 0.063 | 0.068 | runs/run_034_BalancedSoftmax.yaml |
| 035 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_dimweight | 23454 | 19.3 | 0.333 | 0.728 | 0.321 | 0.342 | 0.686 | 0.449 | 0.092 | 0.076 | runs/run_035_BalancedSoftmax.yaml |
| 036 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax_dimweight | 23454 | 19.3 | 0.303 | 0.758 | 0.381 | 0.388 | 0.726 | 0.492 | 0.068 | 0.084 | runs/run_036_BalancedSoftmax.yaml |
| 037 | BalancedSoftmax | nomic-768d | 1 | 64 | 0.3 | balanced_softmax | 56222 | 46.3 | 0.319 | 0.739 | 0.318 | 0.332 | 0.720 | 0.381 | 0.074 | 0.075 | runs/run_037_BalancedSoftmax.yaml |
| 038 | BalancedSoftmax | nomic-768d | 1 | 28 | 0.3 | balanced_softmax | 23606 | 19.5 | 0.333 | 0.737 | 0.299 | 0.301 | 0.657 | 0.346 | 0.104 | 0.077 | runs/run_038_BalancedSoftmax.yaml |
| 039 | BalancedSoftmax | nomic-v2-moe-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.325 | 0.737 | 0.305 | 0.336 | 0.691 | 0.433 | 0.073 | 0.083 | runs/run_039_BalancedSoftmax.yaml |
| 040 | BalancedSoftmax | Qwen3-0.6B-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.324 | 0.740 | 0.355 | 0.343 | 0.691 | 0.436 | 0.083 | 0.085 | runs/run_040_BalancedSoftmax.yaml |
| 041 | SLACE | nomic-256d | 1 | 64 | 0.3 | slace | 23454 | 19.3 | 0.222 | 0.811 | 0.338 | 0.316 | 0.772 | 0.293 | 0.029 | 0.034 | runs/run_041_SLACE.yaml |
| 042 | BalancedSoftmax | Qwen3-0.6B-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.325 | 0.739 | 0.378 | 0.345 | 0.691 | 0.436 | 0.081 | 0.090 | runs/run_042_BalancedSoftmax.yaml |
| 043 | BalancedSoftmax | Qwen3-0.6B-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.324 | 0.740 | 0.355 | 0.343 | 0.691 | 0.436 | 0.083 | 0.085 | runs/run_043_BalancedSoftmax.yaml |
| 044 | BalancedSoftmax | Qwen3-0.6B-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.302 | 0.754 | 0.370 | 0.366 | 0.711 | 0.446 | 0.064 | 0.067 | runs/run_044_BalancedSoftmax.yaml |
| 045 | TwoStageBalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | two_stage_balanced_softmax | 24104 | 19.9 | 0.268 | 0.787 | 0.393 | 0.371 | 0.758 | 0.404 | 0.051 | 0.077 | runs/run_045_TwoStageBalancedSoftmax.yaml |
| 046 | TwoStageBalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | two_stage_balanced_softmax | 24104 | 19.9 | 0.267 | 0.788 | 0.360 | 0.351 | 0.743 | 0.382 | 0.063 | 0.071 | runs/run_046_TwoStageBalancedSoftmax.yaml |
| 047 | TwoStageBalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | two_stage_balanced_softmax | 24104 | 19.9 | 0.270 | 0.779 | 0.339 | 0.353 | 0.737 | 0.373 | 0.073 | 0.070 | runs/run_047_TwoStageBalancedSoftmax.yaml |
| 048 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.275 | 0.781 | 0.372 | 0.363 | 0.778 | 0.389 | 0.044 | 0.062 | runs/run_048_BalancedSoftmax.yaml |
| 049 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.253 | 0.803 | 0.369 | 0.362 | 0.770 | 0.408 | 0.047 | 0.058 | runs/run_049_BalancedSoftmax.yaml |
| 050 | BalancedSoftmax | nomic-256d | 1 | 64 | 0.3 | balanced_softmax | 23454 | 19.3 | 0.285 | 0.768 | 0.393 | 0.382 | 0.724 | 0.480 | 0.069 | 0.086 | runs/run_050_BalancedSoftmax.yaml |
<!-- AUTO-TABLE:END -->

> **Contributor note:** Keep this section in **newest-first** chronological order (most recent date at top).

## Findings

### 2026-04-01 — Consensus-label retrains are informative diagnostics, not a clean frontier board (`twinkl-754.6`)

`twinkl-754.6` reran the active `BalancedSoftmax` recipe with
`consensus_labels.parquet` across seeds `11/22/33` (`run_048`-`run_050`) on the
same corrected-split holdout IDs. The family improved consensus-label holdout
surface metrics such as median `qwk_mean` (`0.372`), accuracy (`0.781`),
calibration (`0.770`), and circumplex cleanliness, and it materially lifted
`stimulation`.

That still does **not** make it the new frontier. The branch changed the
evaluation labels on the same holdout, so its results are not directly
comparable to the persisted-label board. On the fixed 221-row test split,
consensus relabeled `17` `hedonism`, `27` `security`, and `13` `stimulation`
entries. Inside that diagnostic regime, the family also stayed too conservative:
median `recall_-1` fell to `0.270`, minority recall fell to `0.408`, hedging
rose to `65.5%`, and `hedonism qwk` collapsed to `0.104`.

The updated recommendation is therefore narrow and explicit: keep
`run_019`-`run_021` as the active corrected-split default, do **not** add the
consensus family to the active frontier board, and use the new review to drive
a cleaner next ablation. The highest-leverage follow-up is a same-code sibling
rerun with persisted labels plus a confidence-tiered or soft-label variant so
we can separate true learning gains from relabeling gains. Full details:
[`reports/experiment_review_2026-04-01_twinkl_754_6.md`](reports/experiment_review_2026-04-01_twinkl_754_6.md).

### 2026-03-18 — Qwen graduates from single-seed near-miss to real frontier challenger (`twinkl-744`)

`twinkl-744` reran `Qwen/Qwen3-Embedding-0.6B` across the full 3-seed corrected-split
matrix to answer the question left open by the original single-seed diagnostic:
is Qwen actually frontier-credible as a family, or was `run_040` just a near-miss?

**1. The Qwen branch is real.** `run_042`-`run_044` finished with family-median
`qwk_mean 0.370`, `recall_-1 0.318`, minority recall `0.436`, hedging `0.591`,
and calibration `0.691`. That makes it broadly comparable to the incumbent
family (`0.362`, `0.313`, `0.448`, `0.621`, `0.713`) rather than a speculative
encoder revisit.

**2. It still does not clear promotion cleanly.** The family improves the
headline QWK/hedging package slightly, but the gains are small enough to treat
as comparable, not decisive, and the hard-dimension profile is still weaker on
`hedonism` and `power`. The new family-median `hedonism qwk` is only `0.154`,
and `power qwk` is only `0.149`, both materially below the incumbent family.

**3. Qwen should move onto the active board, but not replace the default.**
`run_019`-`run_021` remain the active corrected-split default because they still
offer the cleaner all-around package once hard-dimension stability, minority
recall, and calibration are considered together. But `run_042`-`run_044` are
now the strongest encoder-swap challenger and the first representation branch
worth reopening if encoder work resumes.

**4. Recommendation: keep the incumbent default, keep the weighted branch as
the tail-sensitive reference, and log Qwen as the strongest representation
challenger.** Full details:
[`reports/experiment_review_2026-03-18_twinkl_744.md`](reports/experiment_review_2026-03-18_twinkl_744.md).

### 2026-03-18 — SLACE fails the reserve-branch gate and returns to conservative-family hedging (`twinkl-734`)

`twinkl-734` tested whether `SLACE` could add ordinal structure to the active
`nomic-256d`, `ws=1`, `hd=64` frontier budget without sacrificing the
tail-sensitive behavior that justified `BalancedSoftmax`.

**1. The run did not clear the incumbent or weighted-reference floor.**
`run_041` finished at `qwk_mean 0.338`, below the incumbent family median
(`0.362`) and even slightly below the weighted reference median (`0.342`).
More importantly, `recall_-1` collapsed to `0.134`, well below both the
incumbent (`0.313`) and weighted (`0.378`) branches.

**2. Hedging snapped back to the conservative-loss regime.** Overall hedging
rose to `81.0%`, much worse than the active frontier (`62.1%`) and weighted
reference (`59.9%`). That puts SLACE back in the same qualitative zone as the
repo's earlier conservative ordinal families, which were already ruled out for
production-facing drift detection.

**3. Calibration stayed strong, but the hard-dimension package did not.**
`run_041` kept `10/10` positively calibrated dimensions and strong global
calibration (`0.772`), but `hedonism qwk` fell to `0.055`, `security qwk`
remained below the incumbent family median (`0.196` vs `0.297`), and `power`
collapsed to chance. `stimulation qwk` improved, but only alongside `91.9%`
hedging on that dimension, so it does not rescue the run.

**4. Recommendation: close `twinkl-734` and drop SLACE from the active
roadmap.** `run_019`-`run_021` remain the active corrected-split default, and
`run_034`-`run_036` remain the best tail-sensitive reference branch. Full
details:
[`reports/experiment_review_2026-03-18_twinkl_734.md`](reports/experiment_review_2026-03-18_twinkl_734.md).

### 2026-03-16 — `twinkl-733` semantic counterexample batch is explicitly de-scoped (`twinkl-733`)

`twinkl-733` was opened as a narrow follow-up after the representation
diagnostics: generate one more small `Hedonism` / `Security` counterexample
batch and see whether matched-pair semantic support could cleanly move the hard
dimensions. That experiment is now explicitly de-scoped.

**1. The main finding was already established before this issue would have run.**
By the March 8, March 10, and March 11 reviews, the same replay pattern had
already appeared repeatedly: quiet pleasure and defended rest were being read as
guilt or misalignment, while anxious-surface stability language was being read
as insecurity or threat. The repo does not need one more experiment to show that
this semantic-polarity failure exists.

**2. The targeted-data direction already had a real test.** The earlier
targeted batch `run_022`-`run_024` and the regenerated `Hedonism` / `Security`
lift `run_025`-`run_027` already exercised the "add targeted semantic support"
lever. Those lifts were informative, but they did not produce a clean frontier
change on the family-level metrics that now gate promotion decisions.

**3. The uncertainty and representation reviews reduced the value of one more
small batch.** `twinkl-730` showed the prior lift still lacked clean
family-level separation from the incumbent, while `twinkl-731`, `twinkl-732`,
and `twinkl-742` failed to overturn the underlying diagnosis from the
representation side. That means a new batch would most likely pay the full
synthetic-data workflow cost just to reconfirm what the current record already
supports.

**4. Recommendation: close `twinkl-733` and keep the frontier unchanged.**
`run_019`-`run_021` remain the active corrected-split default. This closeout
does **not** mean `Hedonism` and `Security` are solved; it means another
counterexample batch is not worth a fresh generation / labeling / retraining
cycle right now. `twinkl-734` becomes tracker-unblocked after this closeout, but
it should still be treated as a reserve branch rather than the automatically
recommended next step. Full details:
[`reports/experiment_review_2026-03-16_twinkl_733.md`](reports/experiment_review_2026-03-16_twinkl_733.md).

### 2026-03-16 — Qwen is the first credible encoder-swap near-miss, but still not a frontier win (`twinkl-742`)

`twinkl-742` ran the last planned encoder diagnostic using
`Qwen/Qwen3-Embedding-0.6B` with a native custom classification prompt and
native `256d` truncation, while keeping the active frontier critic budget fixed
at `ws=1`, `hd=64`, and `23,454` parameters.

**1. This is the strongest non-incumbent representation test so far.**
`run_040` improves holdout `qwk_mean` to `0.356`, a large recovery over the
earlier Nomic swaps (`run_039: 0.305`, `run_037: 0.318`, `run_038: 0.299`) at
the same fair-budget `266`-dimensional state width as incumbent `run_020`.

**2. Zero truncation helps, but does not close the frontier by itself.**
The Qwen run truncates `0/1,651` entries, confirming that the shorter
`v2-moe` context window was a real constraint. Even so, `run_040` still trails
incumbent `run_020` on holdout `qwk_mean` (`0.378 -> 0.356`), accuracy
(`0.755 -> 0.740`), calibration (`0.713 -> 0.691`), minority recall
(`0.449 -> 0.436`), and `recall_-1` (`0.342 -> 0.296`).

**3. The win is broad-structure recovery, not hard-dimension polarity.**
Relative to `run_039`, Qwen recovers `self_direction qwk` (`0.441 -> 0.555`),
`conformity qwk` (`0.454 -> 0.524`), `benevolence qwk` (`0.307 -> 0.399`), and
`power qwk` (`0.117 -> 0.187`). But the original hard dimensions still lag:
`stimulation qwk` stays flat (`0.165`) and `hedonism qwk` worsens further to
`0.095`, even though `security qwk` remains better than the incumbent
(`0.261` vs `0.213`).

**4. Recommendation: no immediate rerun, but keep Qwen as the only encoder
branch worth reopening later.** The active frontier remains `run_019`-
`run_021`, with `run_034`-`run_036` as the best tail-sensitive reference
branch. The main change is that Qwen is now the only encoder-swap line with a
plausible future case if representation work is revisited. Full details:
[`reports/experiment_review_2026-03-16_twinkl_742.md`](reports/experiment_review_2026-03-16_twinkl_742.md).

### 2026-03-16 — Nomic v2-MoE 256d improves security and tail recovery, but still loses the frontier (`twinkl-732`)

`twinkl-732` ran the cleaner within-family encoder swap suggested by the
negative `twinkl-731` result: replace `nomic-embed-text-v1.5` with
`nomic-embed-text-v2-moe` at the same `256d`, `ws=1`, `hd=64`,
BalancedSoftmax frontier budget.

**1. The cleaner encoder swap is still a negative frontier result.**
`run_039` keeps the incumbent parameter budget (`23,454`) and state width
(`266`), then still regresses against `run_020` on holdout `qwk_mean`
(`0.378 -> 0.305`), accuracy (`0.755 -> 0.737`), minority recall
(`0.449 -> 0.433`), and calibration (`0.713 -> 0.691`). `recall_-1` is almost
flat (`0.342 -> 0.336`), so there is no convincing “trade a little QWK for a
lot more tail recovery” case here.

**2. It does recover much of what the 768d controls lost.** Relative to
`run_037` and `run_038`, `run_039` improves `recall_-1` (`0.336` vs `0.269` /
`0.246`), minority recall (`0.433` vs `0.381` / `0.346`), and hedging
(`0.598` vs `0.620` / `0.612`). `security qwk` also rises to `0.284`, the best
of the three representation diagnostics. That is useful signal, but it still
does not get the run back into the incumbent QWK range.

**3. The regressions spread beyond the original hard dimensions.** `hedonism`
only partially recovers (`0.188` vs incumbent `0.262`), `stimulation` drops
further to `0.168`, and `power` collapses from `0.342` to `0.117`. Easy/fair
dimensions such as `self_direction` (`0.541 -> 0.441`) and `conformity`
(`0.577 -> 0.454`) also slide. That broader erosion is why the aggregate
frontier case stays negative.

**4. The shorter context window is a caveat, not the main explanation.** The
v2-moe run truncates only `13/1,651` entries (`0.8%`). That matters enough to
document, but not enough to plausibly explain the broad QWK drop on its own.

**5. Recommendation: stop the representation-swap line here.** `run_019`-
`run_021` remain the active corrected-split default, and `run_034`-`run_036`
remain the best tail-sensitive reference branch. This result did not justify a
3-seed v2-moe rerun, and it also did not create a new reason to run `twinkl-733`
once the later de-scope review reported out. Full details:
[`reports/experiment_review_2026-03-16_twinkl_732.md`](reports/experiment_review_2026-03-16_twinkl_732.md).

### 2026-03-16 — Full 768d Nomic v1.5 does not rescue hard-dimension polarity on the frontier (`twinkl-731`)

`twinkl-731` reran the incumbent single-seed frontier checkpoint (`run_020`,
seed `22`) at full native `768d` and added a same-seed fair-budget control to
separate embedding signal from added MLP capacity.

**1. The full-width diagnostic was negative on the metrics that matter.**
`run_037` raised `state_dim` from `266` to `778` and parameter count from
`23,454` to `56,222`, but holdout performance regressed versus `run_020`:
`qwk_mean 0.378 -> 0.318`, `recall_-1 0.342 -> 0.269`, and minority recall
`0.449 -> 0.381`. Calibration stayed roughly flat and hedging barely moved, so
there is no “pay a little QWK for much better tail recovery” argument here.

**2. The matched-budget control rules out a hidden representation win.**
`run_038` kept the same `768d` encoder and `state_dim=778` but reduced
`hidden_dim` to `28`, yielding `23,606` parameters, essentially flat to the
incumbent. It still fell further on aggregate performance
(`qwk_mean 0.299`, `recall_-1 0.246`, minority recall `0.346`), so the 768d
embedding does not appear to unlock useful polarity information once capacity is
normalized.

**3. The target hard dimensions do not improve cleanly.** `run_037` makes both
`hedonism qwk` (`0.262 -> 0.178`) and `security qwk` (`0.213 -> 0.165`) worse.
`run_038` nudges `security qwk` up slightly (`0.225`) and improves security
calibration, but only alongside much heavier `security` hedging (`0.706 ->
0.815`) and a collapse in `hedonism qwk` to `-0.045`. That is not evidence of a
representation-level fix for the frontier’s remaining polarity failures.

**4. Recommendation: stop this line and move to `twinkl-732`.** The active
corrected-split frontier remains `run_019`-`run_021` with
`run_034`-`run_036` as the best tail-sensitive reference branch. Full details:
[`reports/experiment_review_2026-03-16_twinkl_731.md`](reports/experiment_review_2026-03-16_twinkl_731.md).

### 2026-03-14 — Persona-cluster bootstrap shows the weighted branch has real recall lift but unresolved QWK cost (`twinkl-730`)

`twinkl-730` quantified holdout uncertainty on the active corrected-split
BalancedSoftmax families using saved test-output artifacts only: 1,000 persona-
cluster bootstrap resamples for 95% BCa intervals plus 1,000 stratified
persona-cluster permutations for hard-dimension above-chance checks.

**1. The weighted reference branch does have a real tail-recovery lift.**
Relative to the incumbent `run_019`-`run_021`, family-median `recall_-1`
improves by `+0.065` with a 95% BCa CI of `[+0.021, +0.128]`, so that gain is
not just holdout noise.

**2. The weighted branch still does not have a clean promotion case.** Its
family-median QWK delta versus the incumbent is `-0.020` with CI
`[-0.062, +0.018]`, and minority recall is effectively flat at `+0.001` with
CI `[-0.037, +0.036]`. That means the apparent QWK cost is unresolved rather
than proven, but the branch still lacks the all-metric separation needed to
replace the default.

**3. Hard-dimension QWK remains too fragile to drive frontier calls by
itself.** `stimulation` clears the above-chance permutation test in only `1/5`
reviewed families, while weighted `hedonism` also fails the chance check
despite a positive point estimate. `security` is more stable than `hedonism`,
but neither should be treated as decisive without a non-noisy family delta.

**4. Recommendation: use uncertainty-aware promotion gates going forward.**
Future reviews should gate on 95% BCa family-median deltas first: treat any
delta interval spanning zero as unresolved noise, require tail-first promotions
to keep both `recall_-1` and minority-recall lower bounds above zero, and use
hard-dimension QWK only as secondary evidence after it clears the above-chance
test. Full details:
[`reports/experiment_review_2026-03-14_twinkl_730.md`](reports/experiment_review_2026-03-14_twinkl_730.md).

### 2026-03-12 — Effective-prior + per-dimension tau still fails to move the frontier (`twinkl-729`)

`twinkl-729` extended the validation-only softmax post-hoc path with
validation-posterior effective-prior estimation and greedy per-dimension tau
selection, then compared that new branch directly against the untouched
baseline and the existing shared-tau Menon path on `run_020` and `run_036`.

**1. The new branch did not beat the old post-hoc control.** Median test deltas
versus baseline were worse for the new branch than for standard Menon:
effective-prior + per-dim tau reached `recall_-1 -0.057`, `qwk_mean -0.028`,
and `calibration -0.136`, while standard Menon still had the better recall
package at `recall_-1 +0.004` with `qwk_mean -0.020`.

**2. `run_036` was the critical negative result.** Validation selected the new
branch for the strongest weighted reference checkpoint, but holdout metrics
regressed from baseline: `qwk_mean 0.381 -> 0.360`, `recall_-1 0.387 -> 0.339`,
minority recall `0.492 -> 0.440`, and calibration `0.726 -> 0.592`.

**3. `run_020` still behaves like `twinkl-719.5`.** The incumbent checkpoint
again preferred the old shared-tau Menon branch (`tau=0.30`), which recovered a
small `recall_-1` lift (`0.342 -> 0.350`) but still paid too much on QWK,
minority recall, calibration, and opposition structure.

**4. Recommendation: close the current post-hoc line.** Keep `run_019`-`run_021`
as the default corrected-split family and `run_034`-`run_036` as the best
tail-sensitive reference branch. On current evidence, further work should move
to materially different interventions rather than more softmax retargeting.
Full details: [`reports/experiment_review_twinkl_729.md`](reports/experiment_review_twinkl_729.md).

### 2026-03-11 — Full frontier audit v9 keeps the default and shows why inverse-loss weighting stalls (`twinkl-721`)

The fresh full audit across `run_001`-`run_036` found no missing run metadata
to backfill and no leaderboard change on the corrected split: `run_019`-`run_021`
BalancedSoftmax remains the active family-level default.

**1. The weighted branch is still the best tail-sensitive reference family.**
Relative to the incumbent, `run_034`-`run_036` improves median `recall_-1`
from `0.313` to `0.378`, keeps minority recall effectively flat to slightly
better (`0.448` to `0.449`), lowers hedging from `0.621` to `0.599`, and
improves calibration from `0.713` to `0.726`.

**2. The reason it still fails promotion is sharper now.** The new artifact
audit shows inverse-loss weighting was optimizing the wrong proxy, not merely
the wrong heads. `Universalism` hits the max clamp `49` times because it has
very low CE, while `self_direction` stays high-CE despite being easy by QWK;
`hedonism` stays near-neutral (`0.972` selected median weight) and `security`
is consistently downweighted (`0.818`).

**3. Hard-dimension replay still points to semantic polarity errors.** Rebuilt
validation inference from `run_036` shows the largest `hedonism` / `security`
misses are not random; they are consistent flips where defended rest or
stability-seeking language is read as guilt, fragility, or threat.

**4. Recommendation shifts toward lighter post-hoc follow-ups.** The next best
step is the repo's existing Menon-style validation-only logit-adjustment path
on `run_020` and `run_036`, with calibration work treated as a separate follow-
up and any future training-time weighting rerun using a better difficulty proxy
than CE. Full details:
[`reports/experiment_review_2026-03-11_twinkl_721.md`](reports/experiment_review_2026-03-11_twinkl_721.md).

### 2026-03-11 — Weighted BalancedSoftmax improves the tail package, but not enough to replace the default (`twinkl-719.3`)

`twinkl-719.3` reran `BalancedSoftmax` on the frozen holdout with the new
inverse-loss EMA dimension weighting path as `run_034`-`run_036`. The result is
the strongest current **reference branch** for misalignment sensitivity, but it
still falls short of a default promotion.

**1. The family wins the tail-sensitive package.** Relative to the incumbent
`run_019`-`run_021`, the weighted family improves median `recall_-1` from
`0.313` to `0.378`, keeps minority recall effectively flat to slightly better
(`0.448` to `0.449`), reduces hedging from `0.621` to `0.599`, and improves
calibration from `0.713` to `0.726`.

**2. QWK and seed stability still block promotion.** Median `qwk_mean` drops
from `0.362` to `0.342`, and the new family is much less stable across seeds
(`IQR 0.030` vs `0.010`). The strongest single weighted seed (`run_036`) is
excellent, but the family does not yet behave predictably enough to replace the
incumbent default.

**3. The branch is not a clean `hedonism` / `security` rescue.** Median
`hedonism qwk` falls to `0.129`, and median `security qwk` reaches only
`0.222` versus the incumbent `0.297`. The weight traces show a coherent but
limited schedule: `security` is consistently downweighted, `hedonism` stays
near neutral, and `universalism` repeatedly hits the max clamp.

**4. Recommendation: keep it as the active reference branch, not the default.**
Weighted `BalancedSoftmax` now replaces the circumplex branches as the best
training-time comparator for the next frontier step, but the mainline default
stays with `run_019`-`run_021`. Full details:
[`reports/experiment_review_2026-03-11_twinkl_719_3.md`](reports/experiment_review_2026-03-11_twinkl_719_3.md).

### 2026-03-11 — Circumplex-aware batch sampler is explicitly de-scoped for the current rollout (`twinkl-691.5`)

`twinkl-691.5` closed the circumplex rollout by making an explicit keep/drop
decision on the sampler idea. This was not a feasibility punt: the March 9-10
corrected-split evidence already answered the sequencing question strongly
enough that a sampler ablation is not the right next experiment.

**1. The current bottleneck is still hard-dimension polarity, not missing batch
structure.** The incumbent `run_019`-`run_021` `BalancedSoftmax` family remains
the active default at median `qwk_mean 0.362`, `recall_-1 0.313`,
`minority_recall_mean 0.448`, `hedging_mean 0.621`, and
`calibration_global 0.713`. The regenerated `Hedonism` / `Security` lift from
`twinkl-691.3` nudged `hedonism` upward, but it still regressed `security` and
opposition structure relative to the incumbent.

**2. The regularizer already tested the structural hypothesis closely enough.**
`twinkl-691.4` improved aggregate circumplex structure versus the post-lift
control, moving `opposite_violation_mean` from `0.082` to `0.039` and
`adjacent_support_mean` from `0.072` to `0.077`, but it gave back too much
tail-sensitive behavior: `recall_-1` fell from `0.328` to `0.265`,
`minority_recall_mean` from `0.442` to `0.411`, and `hedging_mean` worsened
from `0.598` to `0.641`.

**3. The checkpoint-selection fix did not change that conclusion.**
`twinkl-715` showed the guardrailed rerun family `run_031`-`run_033` could
recover family-level QWK to `0.366`, but it still trailed the incumbent on the
metrics that matter operationally: `recall_-1 0.267` vs `0.313`,
`minority_recall_mean 0.409` vs `0.448`, and `hedging_mean 0.641` vs `0.621`.
That means the circumplex branch was not merely being held back by bad
checkpoint promotion.

**4. Recommendation: keep the sampler off the active roadmap for now.** The
better handoff is the lower-risk sequence already supported by the frontier
evidence: per-dimension weighting on `BalancedSoftmax`, then validation-only
logit retargeting from `run_020`, then head-only retraining only if those still
miss. Reopen the sampler only if those levers fail and newer circumplex
diagnostics still show unresolved theory-violating structure errors. Full
details:
[`reports/experiment_review_2026-03-11_twinkl_691_5.md`](reports/experiment_review_2026-03-11_twinkl_691_5.md).

### 2026-03-10 — Full frontier refresh v8 adds the regularized branches, but the default stays put

The v8 full review now covers all `run_001`-`run_033` manifests, including the circumplex-regularized family `run_028`-`run_030` and the guardrailed rerun family `run_031`-`run_033`. The corrected-split recommendation is still unchanged: `run_019`-`run_021` BalancedSoftmax remains the best default family.

**1. The guarded rerun is only a partial recovery.** `run_031`-`run_033` reach median QWK `0.366` (fair), comparable to the incumbent `0.362`, and median calibration `0.713`, but they still trail on `recall_-1` (`0.267` vs `0.313`), minority recall (`0.409` vs `0.448`), and hedging (`0.641` vs `0.621`).

**2. The regularizer smooths ranking more than it fixes tail errors.** The unguarded regularized family `run_028`-`run_030` kept moderate calibration (`0.709`) and fair median QWK (`0.347`), yet its median `recall_-1` fell to `0.265` and the hard dimensions stayed weak, especially `hedonism`, `security`, and `stimulation`.

**3. Hedonism and security are still the structural blockers.** Across corrected-split family medians, `hedonism` remains the hardest dimension at mean QWK `0.081`, followed by `security` at `0.234`. Replay on the strongest active single checkpoint (`run_020`) again showed polarity errors on quiet pleasure and stability-seeking language rather than random misses.

**4. Recommendation stays structural, not loss-sweep driven.** The next experiment should focus on per-dimension weighting and targeted semantic data support, not another generic loss swap. Full details: [`reports/experiment_review_2026-03-10_v8.md`](reports/experiment_review_2026-03-10_v8.md).

### 2026-03-10 — Hard validation `recall_-1` guardrail fixes checkpoint selection, but not the frontier (`twinkl-715`)

`twinkl-715` added a hard validation `recall_-1 >= 0.4032` guardrail to ordinal checkpoint selection, persisted per-epoch selection traces, and reran the circumplex-regularized `BalancedSoftmax` family on the corrected split as `run_031`-`run_033`. The new policy blocked the same high-QWK but low-tail checkpoints that had previously been promoted in `run_028`-`run_030`.

**1. The selection pathology is real and now fixed.** For all three seeds, the validation-best finite-QWK checkpoint failed the new floor and was marked ineligible with `recall_minus1_below_floor`. The promoted checkpoints moved to later eligible epochs with lower validation QWK but materially higher validation `recall_-1`, which is the intended behavior.

**2. The guarded family improves aggregate holdout QWK without becoming the new default.** Relative to `run_028`-`run_030`, the guarded rerun family median `qwk_mean` improved from `0.347` to `0.366`, while median holdout `recall_-1` only nudged from `0.265` to `0.267`. Minority recall stayed slightly worse than the incumbent default (`0.409` vs `0.448`), and hedging remained higher (`0.641` vs `0.621`).

**3. No seeds became debug-only.** Every rerun still had at least one promotion-eligible checkpoint above the floor, so the new debug-only fallback path was exercised only in tests and remains a safety mechanism rather than an active training outcome.

**4. Recommendation stays unchanged.** Keep the guardrail as evaluation hygiene, but do not promote the circumplex-regularized family over `run_019`-`run_021`. The fix closes a selection bug; it does not yet establish a better frontier family. Full details: [`reports/experiment_review_2026-03-10_twinkl_715.md`](reports/experiment_review_2026-03-10_twinkl_715.md).

### 2026-03-10 — Full frontier refresh v7: hedonism confirmed as the structural bottleneck, per-dimension weighting recommended

The v7 full review covers all 27 runs (90 configs) across both split regimes. The active corrected-split frontier is unchanged: `run_019`-`run_021` BalancedSoftmax remains the default at median QWK 0.362 (fair), recall_-1 0.313, minority recall 0.448, and hedging 62.1%. Post-lift families `run_025`-`run_027` are added to the board as reference rows but do not displace the incumbent.

**1. Hedonism is effectively unmodeled.** Mean QWK across all 4 corrected-split family medians is 0.023 (poor). Only BalancedSoftmax produces positive hedonism QWK (0.247); all conservative losses go negative. Error analysis reveals the model systematically misreads quiet pleasure as tension — it has learned "strong feeling → misalignment" as a spurious correlation. With only ~21 val -1 labels for hedonism, even QWK measurement is inherently noisy.

**2. Both loss correction and data augmentation matter for recall_-1.** BalancedSoftmax recovered recall_-1 from 0.089 (CORN) to 0.313 on the same 1022-row dataset by correcting the class prior in the loss — the conservative losses were suppressing minority predictions even when features supported them. Two subsequent targeted synthetic batches (`twinkl-681.5`, `twinkl-691.2`) then pushed recall_-1 further to 0.342, confirming that data scarcity remains the deeper bottleneck. Neither lever alone is sufficient.

**3. All conservative losses share the same hedging ceiling.** CORN, CDWCE_a3, and SoftOrdinal are all locked at 79-80% hedging despite different ordinal loss formulations. No conservative loss achieved hedging < 60% or minority recall > 30% in 27 runs.

**4. Top recommendations shift from loss sweeps to structural interventions.** (a) Per-dimension uncertainty weighting to down-weight noisy dimensions; (b) decoupled head retraining with class-balanced sampling; (c) targeted LLM augmentation for hedonism; (d) focal temperature scaling post-hoc on CORN; (e) bootstrap CI on per-dimension QWK to quantify evaluation noise. Full details: [`reports/experiment_review_2026-03-10_v7.md`](reports/experiment_review_2026-03-10_v7.md).

### 2026-03-09 — Regenerated `Hedonism`/`Security` lift did not displace the corrected-split default (`twinkl-691.3`)

`twinkl-691.3` reran paired `BalancedSoftmax` and `SoftOrdinal` families on the frozen `twinkl-681.5` holdout after the regenerated `twinkl-691.2` batch was verified, wrangled, labeled, and consolidated. The current workspace now has `204` personas and `1651` judged entries, so the realized fixed-holdout row split is `1213 / 217 / 221` train / val / test.

**1. The new `BalancedSoftmax` family is the better new-family candidate, but it still loses to the incumbent.** Relative to the rerun `SoftOrdinal` family, `run_025`-`run_027` keep higher median `qwk_mean` (`0.346` vs `0.340`), much higher median `recall_-1` (`0.328` vs `0.082`), much higher minority recall (`0.442` vs `0.260`), and far less hedging (`0.598` vs `0.823`). But relative to the current default `run_019`-`run_021`, the new `BalancedSoftmax` family gives back median `qwk_mean` (`0.346` vs `0.362`) and minority recall (`0.442` vs `0.448`) while only partially recovering calibration and hedging versus the earlier targeted branch `run_022`-`run_024`.

**2. `Hedonism` improved a little, but `Security` remains the blocking regression.** The new `BalancedSoftmax` family nudged median `hedonism qwk` up to `0.256` from `0.247` in the incumbent and `0.147` in the earlier targeted branch. That gain did not transfer to `Security`: median `security qwk` fell to `0.199`, below both the incumbent `0.297` and the earlier targeted branch `0.300`, while `security` hedging also rose.

**3. Circumplex diagnostics do not justify a switch either.** Recomputed family summaries from the saved selected-test artifacts show the incumbent default at median `opposite_violation_mean = 0.070` and `adjacent_support_mean = 0.077`. The new `BalancedSoftmax` family worsens opposition structure (`0.082`) without an adjacent-support gain (`0.072`), while the rerun `SoftOrdinal` family preserves opposition slightly better (`0.069`) but collapses compatible co-activation harder (`0.056`).

**4. Recommendation stays unchanged.** Keep `run_019`-`run_021` as the active corrected-split default. The new post-lift `BalancedSoftmax` rerun is a useful data point, but it is not a frontier change. No post-hoc follow-up was run because the new-family winner was not blocked merely by calibration or neutral bias; the deeper issue is the unresolved `Security` and circumplex trade-off. Full details: [`reports/experiment_review_2026-03-09_twinkl_691_3.md`](reports/experiment_review_2026-03-09_twinkl_691_3.md).

### 2026-03-08 — Full frontier refresh keeps original BalancedSoftmax as the default and promotes the targeted batch to a secondary board row

The refreshed full review across `run_001`-`run_024` keeps the split-aware recommendation unchanged: `run_019`-`run_021` remain the default corrected-split family because they still have the best median `qwk_mean` at `0.362` while maintaining strong minority recovery. The frozen-holdout targeted branch `run_022`-`run_024` is now promoted onto the current board as a **secondary** candidate because its median `recall_-1` rises further to `0.342` and median hedging edges down to `0.619`, but its median `qwk_mean` (`0.349`) and calibration (`0.687`) stay worse than the original BalancedSoftmax family.

**1. The hard-dimension story is now sharper.** Across corrected-split runs, `hedonism` remains the hardest dimension with mean QWK `0.021`, followed by `security` at `0.249`; `power` remains the most operationally volatile dimension even after the targeted batch. Checkpoint-level replay on the reproducible frozen-holdout `run_023` shows the model still misreads quiet pleasure/relief signals as duty or sacrifice, and it still flips `security` when entries mix explicit safety language with broader ambivalence or fear.

**2. The targeted batch helped the tail, but mostly where the signal is explicit.** The added training data improved `power recall_-1` and the family-level `recall_-1` median, which is real progress for the immediate bottleneck. It did **not** produce the same kind of lift on `hedonism` or `security`, which argues that the next data-centric step should target those semantics directly rather than assuming more generic hard negatives will transfer.

**3. The next highest-leverage follow-up is still data + calibration, not another global loss sweep.** The corrected-split families all operate in the same high param/sample regime, and the main differences keep coming from boundary behavior and label support rather than backbone size. That makes the most promising next package: (a) a frozen-holdout `hedonism`/`security` targeted batch, (b) one `SoftOrdinal` rerun on the same batch as a low-gap comparator, and (c) a calibration-only follow-up on the winning BalancedSoftmax branch. Full details: [`reports/experiment_review_2026-03-08_v6.md`](reports/experiment_review_2026-03-08_v6.md).

### 2026-03-08 — Targeted `Power`/`Security` batch produced a narrow `Power` win, not a new frontier (`twinkl-681.5`)

`twinkl-681.5` froze the corrected `2025` holdout, added 12 leakage-safe targeted synthetic personas (95 new entries), verified the batch against the baseline snapshot, passed a small label QA gate (`7 keep`, `1 ambiguous`, `0 bad label`), and retrained the existing `BalancedSoftmax` family on seeds `11/22/33`.

**1. `Power` improved in the intended direction.** Median `power recall_-1` rose from `0.1250` in `run_019`-`run_021` to `0.3125` in `run_022`-`run_024`, while median `power qwk` also improved from `0.3337` to `0.3452`. This is the clearest positive signal from the targeted batch.

**2. `Security` was mostly flat on the hard-negative metric.** Median `security recall_-1` stayed at `0.5714`, with only a negligible `security qwk` change (`0.2973 -> 0.2995`). The targeted batch therefore did not shift both issue-scoped dimensions equally.

**3. The family-level trade-off is still unfavorable for a frontier change.** Median `recall_-1` improved from `0.3132` to `0.3420`, but median `qwk_mean` fell from `0.3619` to `0.3488`, calibration fell from `0.7134` to `0.6866`, accuracy dipped from `0.7534` to `0.7484`, and MAE worsened from `0.3042` to `0.3125`.

**4. The result looks targeted rather than broadly general.** `hedonism` regressed despite not being part of the augmentation batch, which argues against claiming broad hard-dimension transfer from this data lift alone.

**Conclusion**: keep `run_019`-`run_021` as the default frontier, and treat `run_022`-`run_024` as evidence that mild-misalignment augmentation can help `Power` without yet establishing a family-wide win. Full details: [`reports/experiment_review_2026-03-08_twinkl_681_5.md`](reports/experiment_review_2026-03-08_twinkl_681_5.md).

### 2026-03-07 — BalancedSoftmax confirms the corrected-split frontier shift (run_019-run_021, `twinkl-681.4`)

`run_019`-`run_021` complete the training-time long-tail softmax ablation on the corrected persona-stratified split while holding the active ws=1 / hd=64 / nomic-256d frontier fixed. The completed family confirms that `BalancedSoftmax` changed the active board, while `LDAM_DRW` remains a negative result.

**1. BalancedSoftmax is now the confirmed 3-seed leader.** Across seeds 11, 22, and 33, the family median is QWK 0.362 (IQR 0.010), `recall_-1` 0.313 (IQR 0.033), minority recall 0.448 (IQR 0.025), hedging 0.621 (IQR 0.038), and calibration 0.713 (IQR 0.036). That keeps it ahead of the completed `CDWCE_a3`, `SoftOrdinal`, and `CORN` families on every tail-recovery metric while still edging `CDWCE_a3` on aggregate QWK.

**2. Seed 33 confirmed the pattern, but not without cost.** `run_021` posted QWK 0.358, `recall_-1` 0.313, minority recall 0.448, and hedging 0.565, so the tail-heavy decision shift held on the third seed. The trade-off also became clearer: MAE rose to 0.324, accuracy fell to 0.734, and calibration slipped to 0.654, leaving the family clearly useful but not universally safer than the conservative alternatives.

**3. `CDWCE_a3` remains the best conservative fallback, and `LDAM_DRW` is eliminated.** `CDWCE_a3` still has better MAE, accuracy, and calibration than the completed BalancedSoftmax family, so it stays relevant when aggregate stability matters more than tail recovery. `LDAM_DRW` still trails the frontier on both QWK and `recall_-1` while showing severe overfitting, so it should not be promoted further in the current regime.

**4. The remaining problem is now more clearly data-centric than loss-centric.** Even after the frontier shift, corrected-split mean QWK is still worst on `hedonism` and `security`, with `power` remaining the most volatile dimension. That keeps `twinkl-681.5` appropriately focused on hard-dimension data lift rather than another fresh loss sweep.

**Conclusion**: the corrected-split board has genuinely moved. Carry `BalancedSoftmax` forward as the primary base for `twinkl-681.5`, then test whether a calibration-only follow-up can recover some of the MAE/accuracy/calibration tax without giving back the tail gains. Full details: [`reports/experiment_review_2026-03-07_v5.md`](reports/experiment_review_2026-03-07_v5.md).

### 2026-03-07 — Validation-only post-hoc tuning favors softmax logit adjustment (`twinkl-681.3`)

`twinkl-681.3` ran a validation-only policy sweep on the corrected-split frontier `run_016`-`run_018` using the existing selected-checkpoint artifacts only. Softmax families (`CDWCE_a3`, `SoftOrdinal`) were evaluated with train-prior logit adjustment over `tau in {0.0, 0.3, 0.5, 0.7, 1.0}`; `CORN` was evaluated with guarded probability-margin threshold policies. Selection was recall-first on validation with a hard `qwk_mean` drop guard of `0.03`, followed by one untouched final test evaluation per tuned model.

**1. Softmax logit adjustment delivered the larger guarded recall lift.** Across the tuned frontier, softmax adjustment produced a median test delta of `+0.095 recall_-1` with `+0.006 qwk_mean`, compared with `CORN` threshold tuning at `+0.009 recall_-1` and `+0.009 qwk_mean`. This makes post-hoc boundary movement a stronger short-term lever for the softmax families than for `CORN`.

**2. `CDWCE_a3` is the clearest `twinkl-681.4` handoff.** The selected `tau=0.70` policy was chosen for all three corrected-split `CDWCE_a3` seeds and lifts test `recall_-1` to 0.276 / 0.244 / 0.288 while keeping test `qwk_mean` at 0.361 / 0.334 / 0.369. Its median tuned profile is now QWK 0.361, `recall_-1` 0.276, minority recall 0.433, and neutral rate 0.736.

**3. `SoftOrdinal` remained a useful comparator, but gains were smaller and less consistent.** One seed kept the reversible baseline (`tau=0.00`) and two selected `tau=0.30`. The family median improved to `recall_-1 = 0.164`, but it still trails tuned `CDWCE_a3` on both recall recovery and balanced test performance.

**4. `CORN` stayed valuable as a calibration anchor, not as the best recovery path.** Per-dimension guarded margin policies produced small recall gains while preserving strong calibration (`median calibration_global = 0.818`), but the family median `recall_-1` only reached 0.161. That makes `CORN` a useful comparator for calibration-sensitive follow-up, not the main optimization base.

**Conclusion**: `twinkl-681.3` should be treated as complete once its report and artifacts are linked from the index. For the next training step, carry tuned `CDWCE_a3` forward as the primary softmax base, keep `SoftOrdinal` as the minority-sensitive comparator, and use `CORN` mainly as a calibration reference. Full details: [`reports/experiment_review_2026-03-07_twinkl_681_3.md`](reports/experiment_review_2026-03-07_twinkl_681_3.md).

### 2026-03-06 — Corrected-split rebaseline resets the active frontier (run_016-run_018)

`run_016`-`run_018` reran the ws=1 nomic hd=64 frontier on the corrected post-`d937094` persona-stratified split using a fixed `split_seed=2025` and model seeds `11/22/33`. These results should be read as a 3-seed family comparison, not as isolated single-run winners.

**1. CDWCE_a3 is the active corrected-split SOTA.** Its median `qwk_mean` is 0.353 (fair) and median `recall_minus1` is 0.104 (poor, but best of the three). It is also the most stable candidate on the hardest `Power` dimension, with seed QWKs 0.291 / 0.406 / 0.330 rather than the near-collapse seen in SoftOrdinal seed 22.

**2. SoftOrdinal remains the best minority-sensitive option.** It has the highest median `minority_recall_mean` at 0.283 and the lowest median hedging at 79.6%, but its seed spread is much wider than CDWCE_a3 (`qwk_mean` 0.388 / 0.330 / 0.346). This makes it a strong comparator for boundary tuning, not the default carry-forward checkpoint family.

**3. CORN is now a calibration anchor, not the leader.** It still has the best median `calibration_global` at 0.818, but its median `qwk_mean` falls to 0.315 on the corrected split. The split fix therefore changed the frontier rather than merely adding noise to the previous ordering.

**4. The split fix narrowed some dimension noise without eliminating the hard cases.** `Security` is materially tighter across seeds than the pre-split picture suggested, while `Power` remains the most volatile dimension. This makes future recommendations more trustworthy if they are based on corrected-split multi-seed summaries instead of single-run pre-split wins.

**Conclusion**: `run_016`-`run_018` are now the active frontier inputs for `twinkl-681.3`. Future experiment reviews should update the post-split board first, keep the pre-split board as historical only, and anchor recommendations on corrected-split evidence.

### 2026-03-04 — CDW-CE alpha sweep and loss-function intervention (run_015)

`run_015` is the first evaluation of **CDW-CE loss** (Class Distance Weighted Cross-Entropy, Polat et al. 2025) with an alpha sweep (2, 3, 5), alongside standard loss reruns (CORAL, CORN, EMD, SoftOrdinal) on the ws=1 frontier. All 7 models used LR finder with valley/steep fallback.

**1. CDWCE alpha=3 is the best new loss.** QWK 0.402 (moderate), approaching run_010 CORN SOTA (0.434, -0.032). The LR finder found a true valley at 0.01552 (15.5x). Strong on benevolence (0.545), conformity (0.484), achievement (0.449). Calibration 0.755 — below SOTA but all 10 dims positive.

**2. CDW-CE shows an inverted-U alpha response.** Alpha=2 (QWK 0.322) is too weak — with only 3 ordinal classes, max |i−c|=2, so 2^2=4 vs 1^2=1 provides minimal differentiation from standard CE. Alpha=3 hits the sweet spot (0.402). Alpha=5 over-penalizes (QWK 0.300, hedging 89.9%, calibration collapsed to 0.639). This matches Polat et al.'s finding that excessive alpha harms performance.

**3. CDW-CE did NOT solve recall_-1.** recall_-1 ranges from 0.012 (CDWCE_a5) to 0.064 (SoftOrdinal) — no improvement over the SOTA reference of 0.089 (run_010 CORN). The ordinal distance penalty causes models to hedge *more* (predicting neutral avoids any penalty), not less. This confirms the bottleneck is **data scarcity** (7.3% of labels are -1, ~75 examples per dimension), not loss function design.

**4. LR finder patterns confirmed.** CORAL and EMD received identical applied LR (0.01979) — deterministic and matching run_014. CORN at 0.00955 (9.5x) again underperformed vs configured 0.001. SoftOrdinal fell back to lr_steep at 0.00111 (1.1x), unlike run_014's valley at 0.0198, producing QWK 0.335 vs 0.388. LR finder inconsistency between runs adds noise.

**5. LR finder history bug confirmed.** `lr_find_history.json` is identical to `lr_find_CDWCE_a5.json` (the last model processed). Previously matched SoftOrdinal in run_014. This is a logging bug, not a training bug.

**Conclusion**: CDW-CE alpha=3 is a viable alternative to CORN but does not advance the frontier on the primary bottleneck (recall_-1). The recall_-1 problem is now confirmed as structural — 15 runs, 7 loss functions, 3 LR regimes have all failed to move it. **Post-hoc logit adjustment** (zero training cost, directly shifts decision boundary) is the highest-leverage next step. See full analysis: [`reports/experiment_review_2026-03-04_v4.md`](reports/experiment_review_2026-03-04_v4.md).

### 2026-03-04 — LR-finder valley on ws=1 baseline (run_014, loss-specific LR sensitivity)

`run_014` applies the LR finder's **valley** LR recommendation to the ws=1 frontier (matching run_010's architecture). The valley LRs are much more aggressive than run_013's lr_steep: CORAL/EMD/SoftOrdinal at ~0.0198 (20× configured), CORN at ~0.0168 (17×). The ws=1 architecture produced true valleys rather than the fallback lr_steep used for ws=2.

**1. CORN regressed catastrophically.** QWK dropped from 0.434 (run_010, SOTA) to 0.314 (-0.120). Calibration fell to 0.753 — worst across all runs 010–014. Security per-dim calibration collapsed to 0.335 (deployment risk). The 17× LR overshoots CORN's sharp optimum at ws=1.

**2. SoftOrdinal achieved best-ever QWK.** QWK 0.388 (+0.080 vs run_010's 0.308) — the largest single-loss improvement in the project. The 20× LR uniquely benefits SoftOrdinal's smooth probability-based loss landscape. Training gap near-zero (0.0002), best_epoch=16 (healthy).

**3. Loss functions have different LR regimes.** This is the key mechanistic finding: threshold-based losses (CORAL/CORN) have sharp optima sensitive to LR magnitude, while distribution-based losses (EMD/SoftOrdinal) have smoother landscapes that tolerate or benefit from higher LR. A single LR finder pass cannot serve all losses.

**4. recall_minus1 still stuck at 5–9%.** The aggressive LR did not improve misalignment detection. CORN recall_-1 fell from 0.089 to 0.060. This confirms the hedging problem is structural, not LR-dependent.

**5. LR finder bug suspected.** `lr_find_history.json` is identical to `lr_find_SoftOrdinal.json`. Config delta for CORN shows generic 0.01979 instead of actual 0.01683. Needs code audit.

**Conclusion**: run_014 is a **negative result for automated LR selection** on threshold-based losses but a **positive result for SoftOrdinal**. The manually configured LR of 0.001 remains superior for CORN at ws=1. Next steps should focus on loss-function-level interventions (CDW-CE, logit adjustment) rather than further LR tuning. See full analysis: [`reports/experiment_review_2026-03-04_v3.md`](reports/experiment_review_2026-03-04_v3.md).

### 2026-03-04 — LR-finder impact on ws=2 baseline (run_013, critic_training_v4)

`run_013` is the third rerun of the ws=2 nomic hd=64 baseline, but produced under `critic_training_v4` which runs an LR finder before training and uses its suggestion as the starting LR. The explicit config is identical to run_011/012, but the applied LRs differ: CORAL/CORN started at ~0.00477 (4.8× configured), EMD at ~0.00166, SoftOrdinal at ~0.00094.

**1. Modest QWK uplift, still below ws=1 leader.** Best run_013 QWK is EMD 0.391 (fair) and CORAL 0.384 — improvements over run_011/012 but still below `run_010 CORN` at 0.434 (moderate). The higher starting LR appears to help escape early plateaus.

**2. Minority recall regressed vs run_012.** SoftOrdinal MinR dropped from 0.391 (run_012) to 0.269; EMD from 0.364 to 0.293; CORAL from 0.342 to 0.259. The higher LR likely finds wider optima that generalize on the majority class but are less sensitive to rare-label gradients.

**3. Stochastic variance confirmed as substantial.** Across run_011/012/013 (three identical-config CORN runs), QWK ranges 0.335–0.382 (0.047 spread) and MinR ranges 0.232–0.296 (0.064 spread). Any single rerun can appear meaningfully different from another.

**4. Power still broken.** Power QWK in run_013: CORAL 0.147, CORN 0.136, EMD 0.136, SoftOrdinal -0.091. Automated data analysis confirms only 9 val personas contribute non-zero Power labels (12 misalignment labels total) — model selection is random.

**5. Experiment logger provenance gap identified.** The YAML files record `learning_rate: 0.001` (configured) despite actual training LRs up to 0.00477. This should be fixed by logging both `learning_rate_configured` and `learning_rate_applied`.

**Conclusion**: The LR finder improves QWK modestly (+0.02–0.05 vs run_011/012) but does not change the frontier. The MinR regression suggests loss-function-level interventions (CDW-CE, logit adjustment) are more promising than LR tuning for addressing the hedging bottleneck. See full analysis: [`reports/experiment_review_2026-03-04_v2.md`](reports/experiment_review_2026-03-04_v2.md).

### 2026-03-03 — critic_training_v3 rerun review (run_012, ws=2, nomic hd=64)

`run_012` is a like-for-like rerun of `run_011` (same encoder, window size, hidden dim, split seed, and losses) after `critic_training_v3` code changes. This isolates stochastic training variance and confirms there is no architectural uplift from the code cleanup itself.

**1. No new QWK state of the art.** Best `run_012` QWK is EMD 0.369 (fair), below `run_010 CORN` at 0.434 (moderate) and slightly below `run_011 EMD` at 0.382 (comparable by <5% rule).

**2. Minority recall improved across all losses, but hedging stayed high.**  
- CORAL: MinR 0.342 (+0.097 vs run_011), hedging 82.7% (excessive)  
- CORN: MinR 0.296 (+0.064), hedging 84.1% (excessive)  
- EMD: MinR 0.364 (+0.056), hedging 79.2% (moderate)  
- SoftOrdinal: MinR 0.391 (+0.079), hedging 77.9% (moderate)  
- CORAL_IW: MinR 0.265 (+0.090), hedging 87.0% (excessive)

**3. Power remains the systemic failure mode.** In `run_012`, Power QWK is 0.021 (CORAL), 0.015 (CORAL_IW), -0.186 (CORN), -0.083 (EMD), -0.225 (SoftOrdinal), indicating persistent instability despite global calibration >0.80.

**Conclusion**: `run_012` is a useful reproducibility checkpoint, not a frontier shift. Keep `run_010 CORN` as QWK leader; use `run_012 EMD`/`run_012 SoftOrdinal` as minority-sensitive baselines for logit-adjustment and class-imbalance interventions.

### 2026-02-22 — Universalism QWK collapse (run_003 → run_009)

Universalism QWK dropped from 0.732 (run_003 EMD, 637 train) to 0.042 (run_009 EMD, 1020 train). This looks alarming but is largely explained by three compounding factors; follow-up monitoring is still warranted before declaring this fully closed.

**1. The 0.732 was inflated by a skewed test distribution.** run_003 included 10 pre-tension Universalism personas whose labels were 87% +1, 13% 0, and **0% −1** (98 entries total). The model achieved high QWK by predicting +1 for the dominant class without ever needing to detect misalignment.

**2. Removing those personas (commit a036004) was correct.** Batch 1B replacement personas (generated with tension-selection) have 54.8% −1 labels, giving the critic a balanced signal. The drop to QWK 0.290 in run_005 reflects honest evaluation against a harder, more realistic distribution — not a regression.

**3. MiniLM's collapse to 0.042 (run_009) is an encoder bottleneck, not a data problem.** MiniLM at window_size=3 produces a 1164-dim state vector; at hd=64, this is an 18:1 compression bottleneck. On the **same 1020-sample dataset**, nomic CORN at hd=64 (266-dim state, 4:1 compression) achieves Universalism QWK **0.466**.

| Run | Encoder | hd | n_train | Uni QWK | Uni Hedge | State dim |
|-----|---------|---:|--------:|--------:|----------:|----------:|
| run_003 EMD | MiniLM | 256 | 637 | 0.732 | 74.8% | 1164 |
| run_005 EMD | MiniLM | 256 | 958 | 0.290 | 84.7% | 1164 |
| run_007 CORN | nomic | 64 | 1020 | 0.466 | 81.9% | 266 |
| run_009 EMD | MiniLM | 64 | 1020 | 0.042 | 82.4% | 1164 |

**Conclusion**: Universalism performance improved materially after the encoder switch to nomic (run_007), but remains variance-sensitive across runs. Keep this dimension under active monitoring; defer additional Universalism data generation or dimension-specific weighting unless instability persists.

### 2026-02-22 — MiniLM retired from frontier experiments

MiniLM (all-MiniLM-L6-v2) is **no longer a primary frontier candidate** for future VIF runs. Frontier experiments, analysis, and recommendations should prioritize nomic-embed-text-v1.5, while keeping MiniLM as an occasional sentinel baseline for regression checks.

**1. MiniLM's state pipeline is fundamentally over-parameterized for this dataset.** The window_size=3 state encoder produces a 1164-dim state vector. Even at hd=64, this yields an 80K-parameter model with a 79:1 param/sample ratio — still firmly in the "high" regime. The model consistently early-stops at epoch 2-5, never learning beyond majority-class hedging.

**2. MiniLM degraded on every metric when the dataset expanded.** From 637 → 1020 train samples, MiniLM QWK dropped across all losses (e.g., EMD: 0.410 → 0.259; SoftOrdinal: 0.383 → 0.236). More data amplified the class-imbalance signal rather than improving generalization.

**3. nomic dominates on the expanded dataset at every capacity level tested.** On the same 1020-train data, nomic at hd=64 (22:1 ratio) achieves QWK 0.413 vs MiniLM at hd=64 (79:1 ratio) achieving QWK 0.227. The gap is too large to close with hyperparameter tuning.

| Metric (avg across losses) | MiniLM hd=256 (run_005) | MiniLM hd=64 (run_009) | nomic hd=64 (run_007) |
|-----------------------------|------------------------:|------------------------:|----------------------:|
| QWK | 0.278 | 0.224 | **0.363** |
| Calibration | 0.643 | 0.739 | **0.839** |
| Minority Recall | 0.187 | 0.190 | **0.278** |
| Hedging | 86.6% | 85.4% | **81.7%** |
| Param/sample ratio | 388 (severe) | 79 (high) | **22 (high)** |

**Conclusion**: MiniLM runs (001, 003, 005, 009) are retained as historical baselines. Future `/experiment-review` reports should focus frontier insights and recommendations on nomic-embed configurations, with infrequent MiniLM sentinel reruns only for regression detection.
