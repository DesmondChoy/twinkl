# Experiment Review: Hybrid Soft Vote-Distribution Labels (`twinkl-j0ck`)

## Decision

Do **not** make soft vote-distribution BalancedSoftmax the default VIF target
regime. The soft family is less neutral-biased and improves median minority
recall and Hedonism QWK, but it does not improve median `recall_-1`, weakens
Security and Stimulation, and produces worse probability-distribution scores
on the raw outputs consumed at inference.

The experiment does establish one useful mechanism-level result: empirical
vote distributions can be learned in the prior-adjusted BalancedSoftmax loss
space. The gap appears when those logits are decoded back to the raw class
probabilities used by the product. If this line is revisited, the next bounded
test should be plain soft cross-entropy or an explicitly calibrated decoding
rule, not another identical soft BalancedSoftmax rerun.

## Question

Does preserving Judge disagreement as a three-class target distribution help
the current `window_size: 1` MLP relative to collapsing the same evidence into
one hard label?

## Target contract

The experiment uses one current-contract hybrid artifact:

- nine non-Security values: the five validated `twinkl-754` profile-only votes;
- Security: the receipt-bound `security_active_critic_state_v1` reviews from
  `twinkl-a30f`, using three votes normally and the fourth vote for the 25
  three-way tie cases;
- hard target: the existing two-stage consensus resolver for nine values plus
  the repaired Security decision;
- soft target: empirical class fractions in explicit `[-1, 0, +1]` order.

The target covers all 1,651 entries. It retains a ten-value hard
`alignment_vector` and a value-major 30-probability `soft_alignment_vector`.
The original persisted labels, consensus labels, and repaired Security labels
remain unchanged. Of 16,510 entry-dimension targets, 2,599 (15.7%) are
non-unanimous under the hybrid contract.

Artifact:
`logs/exports/twinkl_j0ck_soft_vote_target_v1/hybrid_soft_vote_labels.parquet`.
The composite source hashes and output receipt are recorded in
`logs/exports/twinkl_j0ck_soft_vote_target_v1/target_provenance.json` and
backfilled explicitly into `run_063`-`run_068` without changing model weights.

## Controlled design

The hard and soft arms use the same target artifact, split, state, model,
optimizer, and checkpoint policy. Only the target representation, class-prior
counting, and loss change.

| Setting | Value |
|---|---|
| Hard runs | `run_063`, `run_065`, `run_067` |
| Soft runs | `run_064`, `run_066`, `run_068` |
| Model seeds | 11, 22, 33 |
| Split seed | 2025 |
| Holdout | `twinkl_681_5_holdout.yaml` |
| Train / validation / test | 1,213 / 217 / 221 entries |
| Encoder | Nomic v1.5, 256 dimensions |
| State | `window_size: 1`, 266 dimensions |
| MLP | hidden 64, dropout 0.3, 23,454 parameters |
| Learning rate | 0.015522253574270487 |
| Selection | guarded QWK-first; retained `0.02` recall-window candidate |

Checkpoint selection and standard metrics use the matching hard hybrid labels
for both arms. This isolates the effect of preserving disagreement and avoids
confounding the result with the Security repair.

## Selected-checkpoint results

| Seed | Arm | QWK | `recall_-1` | Minority recall | Hedging | Calibration |
|---:|---|---:|---:|---:|---:|---:|
| 11 | Hard | 0.3894 | 0.3200 | 0.4159 | 0.6575 | 0.7710 |
| 11 | Soft | 0.4050 | 0.3075 | 0.4355 | 0.6045 | 0.7232 |
| 22 | Hard | 0.3890 | 0.2412 | 0.4081 | 0.6407 | 0.7547 |
| 22 | Soft | 0.3769 | 0.3467 | 0.4498 | 0.5982 | 0.7223 |
| 33 | Hard | 0.3998 | 0.3267 | 0.4529 | 0.6357 | 0.7344 |
| 33 | Soft | 0.3966 | 0.3008 | 0.4138 | 0.6262 | 0.7584 |
| **Median** | **Hard** | **0.3894** | **0.3200** | **0.4159** | **0.6407** | **0.7547** |
| **Median** | **Soft** | **0.3966** | **0.3075** | **0.4355** | **0.6045** | **0.7232** |

The soft arm changes behavior, but not in the way required for a default:

- median QWK rises only 0.0072 and is not directionally consistent by seed;
- median minority recall rises 0.0196;
- median hedging falls 3.62 percentage points, so the model predicts active
  classes more often;
- median `recall_-1` falls 0.0125 and improves in only one of three seeds;
- the repository's uncertainty-error correlation falls from 0.7547 to 0.7232.

## Hard dimensions

| Dimension | Hard median QWK | Soft median QWK | Hard hedging | Soft hedging |
|---|---:|---:|---:|---:|
| Hedonism | 0.1528 | **0.2160** | 0.7692 | **0.6787** |
| Security | **0.3578** | 0.2811 | **0.4253** | 0.4389 |
| Stimulation | **0.4106** | 0.3185 | 0.8190 | **0.7466** |

Hedonism is the clearest positive result: soft targets reduce neutral collapse
and lift QWK. That gain is not general. Security loses 0.0767 median QWK even
under the repaired target contract, while Stimulation loses 0.0921.

## Distribution-aware evaluation

Lower NLL, Brier, and entropy MAE are better. The raw-output scores use the
probabilities exported for runtime consumption. The loss-space scores reapply
the per-dimension class priors used inside BalancedSoftmax.

| Probability space | Arm | Soft NLL | Brier | Entropy MAE (bits) | Entropy correlation |
|---|---|---:|---:|---:|---:|
| Raw/runtime | Hard | **0.6239** | **0.2844** | **0.7167** | **0.2305** |
| Raw/runtime | Soft | 0.6705 | 0.3097 | 0.7676 | 0.2249 |
| Prior-adjusted loss | Hard | 0.5216 | 0.2131 | **0.4117** | 0.3114 |
| Prior-adjusted loss | Soft | **0.5076** | **0.2105** | 0.4476 | **0.3194** |

This is the main diagnosis. The soft objective is functioning: in the exact
prior-adjusted space it optimizes, median NLL improves by 0.0140 and Brier by
0.0026. But the raw decoded probabilities become worse on every seed: median
NLL increases by 0.0466 and Brier by 0.0252. BalancedSoftmax's class-prior
correction and empirical soft-target calibration are pulling on different
probability contracts.

The retained `0.02` recall-window checkpoints do not rescue the soft family.
Their median QWK is 0.3769 and median `recall_-1` is 0.3467; they remain useful
diagnostic artifacts, not a default selection policy.

Machine-readable comparison:
`logs/exports/twinkl_j0ck_soft_vote_target_v1/paired_comparison_summary.json`.

## Implications

1. **Do not promote the soft BalancedSoftmax regime.** It is a useful
   diagnostic, not a better all-around VIF target.
2. **Judge disagreement contains dimension-specific signal.** Hedonism benefits
   while Security and Stimulation do not, so one global soft-target policy is
   too blunt.
3. **The remaining bottleneck is not simply hard-label collapse.** Preserving
   vote fractions changes model behavior but does not consistently lift the
   misalignment class or the repaired hard dimensions.
4. **If revisited, change the objective/decoding contract.** Test unadjusted
   soft cross-entropy or a validation-fitted probability decoder before more
   data generation or architecture work.
5. **No drift-promotion claim follows.** The retired WQ9P benchmark was not used,
   and the unresolved `twinkl-v8pb` promotion population remains the production
   gate.

The active historical frontier remains `run_019`-`run_021`. For future
`window_size: 1` work, retain `security_active_critic_state_v1`; use this hybrid
artifact only when an experiment explicitly needs the vote-distribution
regime and keeps the label lenses separate.
