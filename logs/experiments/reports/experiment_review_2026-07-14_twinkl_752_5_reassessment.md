# `twinkl-752.5`: Raw-input and scheduling reassessment

## Decision

The old conditional rejection of raw `run_020` input is now **inconclusive**,
not reversed. Across three repeats, weekly review without the VIF Critic found a
median 9/33 Drifts (`0.273` recall), while weekly review with raw VIF Critic
input found 7/33 (`0.212`). The paired recall delta was `-0.061`, but its 95%
trajectory-bootstrap interval crossed zero (`[-0.158, 0.033]`). Raw input also
reduced median coverage by `0.094` and added three median false Drift alerts.

VIF-Critic-triggered early-plus-weekly review did **not** improve Drift recall:
it also found a median 9/33 Drifts. It moved median detection delay from 5 to 1
day, added one median false Drift alert, and cost 57 extra reviewer calls. The
timing delta was `-4` days with a 95% interval of `[-5.5, 0]`; the recall delta
was exactly zero with interval `[0, 0]`.

The machine-readable metrics call this branch `earlier_detection_only`, where
“only” means no Drift-hit gain. It does not erase the observed false-alert and
coverage changes reported here.

The offline diagnostic found that the VIF Critic placed 7/19 triggers after a
known Drift confirmation in the same persona-week (`36.8%`), versus a random
median of `5.3%` and random 95% interval `[0%, 15.8%]` (`p=0.0001`, one-sided
empirical test). This says the frozen scores target Drift-relevant review
opportunities better than chance on this development union. It does not show
that early review improves Drift detection.

No threshold, false-alert tolerance, deployment approval, or runtime
architecture is adopted. This study tested **review-again**; it did not test
replacing weekly review with **review-early**.

## Frozen study

- Development union: 106 trajectories, 105 personas, 882 unique Journal
  Entries, 894 entry/Core-Value cells, and 33 known Drifts across 28 Drift
  trajectories.
- All 106 trajectory-level Drift outcomes are resolved. Two old entry-level
  reviewer disagreements remain in the two omitted prior-Drift trajectories,
  but they do not change either resolved Drift.
- The four Opus-resolved trajectories contain no Drift. Excluding them leaves
  102 trajectories and all 33 Drifts.
- Historical provenance: 64 training trajectories / 17 Drifts; 10 development
  / 3; 16 former-promotion / 7; 14 retired / 4; and 2 validation / 2.
- Model: `gpt-5.4-mini-2026-03-17`, Responses API structured parsing,
  reasoning effort `none`, `store: false`, three repeats.
- Comparable weekly setups used identical Journal Entry text, Core Values,
  prompt, validation, and failure handling. Only the raw VIF Critic block
  differed. Early calls hid the numeric VIF Critic scores.
- Scheduler rule: first qualifying trigger per persona-week when two
  consecutive Journal Entries for a reviewed Core Value had mean `P(-1) >=
  0.8` and maximum uncertainty `<= 1.010153`. The fixed-seed MC Dropout receipt
  used 50 samples.
- The protocol, prompts, eligible opportunities, 19 triggers, hashes, decision
  rules, and cost cap were committed in `dd49b5c` before outcomes were called or
  the zero-call diagnostic was scored.

This is selection-biased, AI-reviewed development evidence. It is not a
prevalence sample or final test set. Scores on the 64 training-seen trajectories
are in-sample.

## Primary results

Values are medians across the three registered repeats.

| Setup | Drift hits | Recall | Precision | Drift alerts | False Drift alerts | Median delay | Cross-week hits | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Weekly, no VIF Critic input | 9/33 | 0.273 | 1.000 | 9 | 0 | 5 days | 4/21 | 0.670 | 0.330 |
| Weekly, raw VIF Critic input | 7/33 | 0.212 | 0.700 | 10 | 3 | 3 days | 3/21 | 0.594 | 0.406 |
| VIF-Critic-triggered early + weekly | 9/33 | 0.273 | 0.900 | 10 | 1 | 1 day | 5/21 | 0.670 | 0.330 |

Raw repeat-level numerators were:

| Repeat | Weekly, no input | Weekly, raw input | Scheduled early + weekly |
|---:|---:|---:|---:|
| 1 | 9/33 hits, 2 false | 6/33 hits, 2 false | 9/33 hits, 3 false |
| 2 | 8/33 hits, 0 false | 7/33 hits, 3 false | 9/33 hits, 0 false |
| 3 | 9/33 hits, 0 false | 7/33 hits, 4 false | 9/33 hits, 1 false |

## Paired uncertainty

Each interval used 10,000 trajectory bootstrap resamples. Deltas are second
setup minus weekly review without VIF Critic input.

| Comparison | Metric | Median delta | 95% interval |
|---|---|---:|---:|
| Raw VIF Critic input | Drift recall | -0.061 | [-0.158, 0.033] |
| Raw VIF Critic input | Drift precision | -0.300 | [-0.600, 0.000] |
| Raw VIF Critic input | False Drift alerts | +3 | [0, 7] |
| Raw VIF Critic input | Coverage | -0.094 | [-0.170, -0.019] |
| Raw VIF Critic input | Median delay | -2 days | [-7.5, 1.0] |
| Scheduled early + weekly | Drift recall | 0.000 | [0.000, 0.000] |
| Scheduled early + weekly | Drift precision | -0.068 | [-0.222, 0.000] |
| Scheduled early + weekly | False Drift alerts | +1 | [0, 3] |
| Scheduled early + weekly | Coverage | +0.009 | [0.000, 0.028] |
| Scheduled early + weekly | Median delay | -4 days | [-5.5, 0.0] |

## Sensitivity and provenance

Excluding the four Opus-resolved trajectories does not change any median Drift
hit, recall, precision, false-alert, delay, or cross-week result. It changes
only coverage denominators because all four trajectories contain no Drift.

The non-training subgroup contains 42 trajectories and 16 Drifts. All three
setups found a median 3/16 Drifts (`0.188` recall), with precision `1.0` and no
false Drift alerts. Raw input lowered median coverage from `0.643` to `0.571`.
Scheduled early-plus-weekly matched weekly-only on hits, delay, cross-week
recovery, coverage, and abstention. The observed scheduling timing benefit in
the primary result therefore comes from training-seen development evidence.

Median Drift hits by historical provenance were:

| Historical split | Known Drifts | Weekly, no input | Weekly, raw input | Scheduled early + weekly |
|---|---:|---:|---:|---:|
| Training | 17 | 6 | 4 | 6 |
| Development | 3 | 2 | 1 | 2 |
| Former promotion | 7 | 1 | 2 | 1 |
| Retired | 4 | 0 | 0 | 0 |
| Validation | 2 | 0 | 0 | 0 |

These small, heterogeneous slices explain why former-test provenance is
reported rather than promoted into a new final test.

## Trigger placement, calls, and cost

- Eligible early-review opportunities: 671 across 424 persona-weeks.
- Realized triggers: 19, or `19/424 = 4.48%` of eligible persona-weeks and
  `19/510 = 3.73%` of all union persona-weeks.
- Trigger hits: 7/19 (`36.8%`). Random placement at the same count had mean
  `5.86%`, median `5.26%`, and 95% interval `[0%, 15.79%]` across 10,000 frozen
  seeds. No random draw matched the observed rate; the corrected one-sided
  empirical `p` is `1/10,001 = 0.0001`.
- Weekly-only: 1,530 calls, `$2.3711`.
- Weekly with raw VIF Critic input: 1,530 calls, `$2.4373`.
- Scheduled early-plus-weekly: the same 1,530 weekly-only receipts plus 57
  early calls, or 1,587 logical calls and `$2.4507`.
- Total unique API execution: 3,117 calls and `$4.8880`.
- Valid receipts: 1,506/1,530 weekly-only, 1,514/1,530 raw-input, and 57/57
  early. The 40 invalid weekly receipts failed closed: 35 coordinate mismatches
  and 5 evidence-quote mismatches. There were no remaining API errors.

## Handoff to `twinkl-752.2`

1. Raw VIF Critic input has not earned inclusion: its recall effect is
   uncertain, while coverage and false Drift alerts worsened on the primary
   development union.
2. The frozen VIF Critic scores identify Drift-relevant timing opportunities,
   but the tested review-again setup did not add Drift hits and its apparent
   timing benefit disappeared on the non-training subgroup.
3. Architecture approval remains a user decision. A fresh final test is still
   required before deployment, regardless of that choice.

## Artifacts

- Config: `config/evals/twinkl_752_5_reassessment_v1.yaml`
- Runner: `scripts/experiments/reassess_twinkl_752_5.py`
- Frozen and scored artifacts:
  `logs/experiments/artifacts/twinkl_752_5_reassessment_20260714/`
- Response SHA-256:
  `8c6a5127f14fb32a3ab1c3f465c44cce72cd6fc3051fb9e0778e0096d8111659`
- Metrics SHA-256:
  `b368c1ddd47e711700c6224f02bf1580c10e2d5a799f34e6e03d0bb990ed8f6b`
