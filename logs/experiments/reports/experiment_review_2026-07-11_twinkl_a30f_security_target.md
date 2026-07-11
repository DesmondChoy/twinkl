# Security target repair review (`twinkl-a30f`)

## Decision

Use `security_active_critic_state_v1` as the Security supervision regime for
future `window_size: 1` Critic experiments. Keep the historical labels intact
for provenance and historical-lens comparisons.

The old Security target was materially mismatched to the active Critic state.
This is not just an easier-label effect: the repaired-target model family
improved median test Security QWK by about `0.17` under both the historical and
repaired lenses. The repair does not finish Security modeling. Absolute median
Security QWK is still only `0.328` against the repaired target, and performance
is weaker on cases where the repeated reviews disagreed. Richer legal history
remains a separate representation experiment.

## Target construction

The review covered all 1,651 entries, so the train, validation, and test
populations stayed fixed. Each entry received three isolated, blinded
`gpt-5.4-mini` reviews of the exact active state: current runtime-formatted
session plus normalized declared-value profile. Twenty-five three-way ties
received a fourth review. Persisted labels, split membership, biography,
demographics, earlier entries, and model predictions were excluded.

| Receipt | Result |
|---|---:|
| Review calls | 4,978 |
| Recorded cost | USD 11.7325 |
| Unanimous decisions | 1,162 |
| Two-of-three majority decisions | 464 |
| Fourth-review tie breaks | 25 |
| Historical `-1 / 0 / +1` | 151 / 1,212 / 288 |
| Repaired `-1 / 0 / +1` | 265 / 697 / 689 |
| Changed Security labels | 678 (41.1%) |

This is repeated model review, not independent human ground truth. Agreement
and rationale receipts make the target auditable, but the large marginal shift
must remain explicit in every comparison.

## Legacy 14-case audit

The historical `twinkl-747` sample was selected from known frontier misses, so
it remains diagnostic rather than an evaluation set. Under the conservative
observed-delta buckets in the target contract:

| Bucket | Cases | Meaning |
|---|---:|---|
| `matches_full_context` | 6 | Exact active-state and legacy rich-context labels agree. |
| `changes_with_bio_or_history` | 1 | Active state matches the legacy profile-only arm but not the combined biography/history arm. The added factor is not isolated. |
| `changes_between_active_state_and_legacy_profile_prompt` | 7 | The two legacy arms agree but the exact active-state review differs. This is an input/prompt-contract delta, not proof that biography is required. |
| `unresolved_context_sensitivity` | 0 | No case had three different labels. |

The full-corpus target changed 7 of these 14 historical labels. One required a
fourth-review tie break. This confirms why the old selected subset could not be
used as the repair itself.

## Controlled comparison

Six BalancedSoftmax runs held architecture, encoder, corrected holdout,
optimizer settings, and seeds fixed. Only the Security target regime changed:

| Seed | Historical target | Repaired target |
|---:|---|---|
| 11 | `run_057` | `run_058` |
| 22 | `run_059` | `run_060` |
| 33 | `run_061` | `run_062` |

The frozen test split contains 221 entries. Its Security marginals changed
from `14 / 169 / 38` to `28 / 110 / 83` for `-1 / 0 / +1`; 86 test labels
changed. The validation split contains 217 entries and 82 changed labels.

### Family medians on the frozen test split

| Model target | Scoring lens | Overall QWK | Security QWK | Security recall `-1 / 0 / +1` | Global `recall_-1` | Hedging | Calibration |
|---|---|---:|---:|---:|---:|---:|---:|
| Historical | Historical | 0.343 | 0.205 | 0.429 / 0.864 / 0.105 | 0.337 | 0.592 | 0.698 |
| Historical | Repaired | 0.333 | 0.156 | 0.321 / 0.945 / 0.096 | 0.323 | 0.592 | 0.674 |
| Repaired | Historical | 0.371 | 0.372 | 0.500 / 0.621 / 0.711 | 0.355 | 0.587 | 0.681 |
| Repaired | Repaired | 0.363 | 0.328 | 0.500 / 0.700 / 0.446 | 0.355 | 0.587 | 0.693 |

The paired interpretation is:

- Against repaired labels, repaired-target training raises median Security QWK
  from `0.156` to `0.328` (`+0.172`).
- Against historical labels, the same training raises median Security QWK from
  `0.205` to `0.372` (`+0.168`).
- Repaired-target training replaces the historical model's neutral collapse
  with materially better `+1` recall. The cost is lower neutral recall, which
  is expected after the target moves 515 full-corpus entries out of neutral.
- Aggregate QWK, global `recall_-1`, and hedging do not regress. On the same
  historical lens, median Conformity QWK moves from `0.549` to `0.572`, while
  Tradition moves from `0.485` to `0.446`; the shared representation still has
  some seed-sensitive spillover.

### Per-seed Security QWK

| Seed | Historical model / historical lens | Historical model / repaired lens | Repaired model / historical lens | Repaired model / repaired lens |
|---:|---:|---:|---:|---:|
| 11 | 0.195 | 0.095 | 0.414 | 0.339 |
| 22 | 0.214 | 0.156 | 0.332 | 0.328 |
| 33 | 0.205 | 0.201 | 0.372 | 0.277 |

Every seed improves under both lenses. That consistency is stronger evidence
than a single favorable checkpoint.

## Remaining error boundary

On repaired-label test cases, repaired-model median exact-class accuracy is
`0.628` for the 145 unanimous cases, `0.514` for the 70 majority cases, and
`0.333` for the six fourth-review tie-break cases. The final stratum is too
small for a stable estimate, but the ordering is consistent with residual
target ambiguity.

The evidence therefore supports a mixed conclusion:

1. **Target mismatch was material.** The old target encouraged neutral
   predictions and obscured student-visible positive Security evidence.
2. **The repaired target is useful but provisional.** It is complete,
   non-destructive, receipt-bound, and learnable enough to replace historical
   Security supervision for comparable `window_size: 1` work.
3. **Representation and semantics still limit performance.** A median Security
   QWK of `0.328` is not a solved dimension, disagreement cases remain harder,
   and this experiment does not test whether legal trajectory history would
   improve further.

Do not merge these scores into the historical leaderboard as if the target
regime were unchanged. Use paired historical/repaired lenses when comparing
across the boundary, and carry review disagreement forward into the planned
soft-label work in `twinkl-j0ck`.

## Artifacts

- Review receipts: `logs/exports/twinkl_a30f_active_critic_state_full_v1/`
- Repaired target and training labels:
  `logs/exports/twinkl_a30f_security_target_full_v1/`
- Six run records: `logs/experiments/runs/run_057_BalancedSoftmax.yaml` through
  `run_062_BalancedSoftmax.yaml`
- Paired evaluation cells and reports:
  `logs/experiments/artifacts/security_target_comparison_twinkl_a30f_20260711/`
