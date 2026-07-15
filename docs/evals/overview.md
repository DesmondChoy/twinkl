# VIF Evaluation Overview

This folder contains evaluation specifications for the **Value Identity Function (VIF)**. The evaluations check each component before work moves to the next stage.

---

## Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VIF EVALUATION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │    Stage 1   │    │    Stage 2   │    │    Stage 3   │    │    Stage 4   │  │
│   │  Data Prep   │───▶│   Training   │───▶│  Inference   │───▶│ User Output  │  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│          │                   │                   │                   │          │
│          ▼                   ▼                   ▼                   ▼          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │    LLM-Judge     │    │    Value     │    │    Drift     │    │ Explanation  │  │
│   │  Validation  │    │   Modeling   │    │  Detection   │    │   Quality    │  │
│   │     Eval     │    │     Eval     │    │     Eval     │    │     Eval     │  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Order | Eval File | Stage | What It Validates | Key Metrics |
|:-----:|-----------|----------------|-------------------|-------------|
| 1 | [`judge_validation_eval.md`](./judge_validation_eval.md) | Data Preparation | LLM-Judge labels are consistent & agree with humans | Cohen's κ > 0.60 |
| 2 | [`value_modeling_eval.md`](./value_modeling_eval.md) | Model Training | VIF Critic recovers Conflict | Primary: entry-level `recall_-1`; mandatory precision-recall reporting; QWK and `+1` diagnostic |
| 3 | [`drift_detection_eval.md`](./drift_detection_eval.md) | Inference | Weekly Drift Reviewer and Drift Detector find Drift without unacceptable false Drift alerts | Drift recall first; false Drift alerts second; coverage and abstention diagnostic |
| 4 | [`explanation_quality_eval.md`](./explanation_quality_eval.md) | User Output | Explanations are grounded and useful | Likert ≥ 3.5/5 |

---

## Dependencies

The adopted architecture has a user-facing evaluation path and a VIF Critic
review-and-retrain path:

```
Journal Entries + Core Values ──▶ drift_detection_eval ──▶ explanation_quality_eval
                                  (Weekly Drift Reviewer     (Weekly Digest and
                                   + Drift Detector)          Weekly Coach)

judge_validation_eval ──▶ value_modeling_eval ──▶ stored VIF Critic predictions
       ▲                       │                              │
       └──── independent review and retraining ◀────────────┘
```

**Implications:**
- You cannot train the VIF Critic without validated LLM-Judge labels
- The current user-facing Drift evaluation does not require VIF Critic input
- A future VIF Critic candidate-confirmation path must pass predefined
  development criteria and a fresh final test before deployment approval
- Explanation quality can be partially tested at any stage (rationales work independently)

---

## Current Status Summary

| Eval | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| LLM-Judge Validation | 🟢 Operational | 1 651 labels across 204 personas; the shared 115-Journal-Entry / 19-persona benchmark yields Fleiss' κ **0.56** and avg LLM-Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up audits include the `twinkl-747` reachability report and the `twinkl-754` 5-pass consensus/self-consistency rerun. | Add automated post-label QA (all-zero rate, sparsity, distribution) and continue hard-dimension target refinement, especially after the `Security` caveat from `twinkl-747` |
| Value Modeling | 🟡 In Progress | `run_019`-`run_021` remains the historical corrected-split reference. The paired `run_057`-`run_062` experiment shows that active-state Security repair raises median test Security QWK by about **0.17**. `twinkl-j0ck` did not promote soft targets, compact-history `run_069` failed its seed-11 expansion gate, and the Codex-reviewed `twinkl-748` Hedonism hard-set found only 0.05 median `-1` recall and 0.05 strict-pair accuracy for the incumbent. The [`twinkl-752.5` reassessment](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) found no reliable benefit from exposing raw `run_020` scores to the Weekly Drift Reviewer and no scheduling recall gain. | Keep the VIF Critic in the required stored-prediction, independent-review, and retraining path. Persist full probabilities and provenance, build reviewed disagreement data, and evaluate any candidate-confirmation path only after predefined criteria are fixed. |
| Drift Detection | 🟡 Partial | The complete development review contains 42 Drifts across 36 Drift trajectories in 292 resolved cases. The earlier 106-case `twinkl-752.5` study found median Drift recall of `0.273` weekly-only, `0.212` with raw VIF Critic input, and `0.273` with VIF-Critic-triggered early-plus-weekly review. On the complete data, `gpt-5.6-luna` at reasoning effort `low` had median Drift recall of `0.548`, 4 false Drift alerts, and `0.637` coverage, versus `0.476`, 13, and `0.777` at reasoning effort `none`. `low` mechanically failed the preregistered coverage gate, but the approved metric hierarchy ranks Drift recall first, false Drift alerts second, and coverage as diagnostic. `low` is the selected development Weekly Drift Reviewer, and the study stops before `medium`. This is AI-reviewed synthetic development evidence, not human validation or a final test. | Keep the staged architecture: Weekly Drift Reviewer decisions without VIF Critic input feed the deterministic Drift Detector. `twinkl-7vam` must define deployment-approval criteria; `twinkl-pv6s` must provide a fresh final test. Runtime wiring remains pending before `twinkl-olen`. |
| Explanation Quality | 🟡 Partial | 1 594 / 1 651 persisted labels have rationales, the annotation tool can display/compare them, and Weekly Coach Tier 1 narrative checks are implemented in [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py) | Batch pass-rate reporting for Weekly Coach narratives, rationale-specific Tier 1 checks in `src/judge/`, and deeper rationale-review LLM / human-calibration work |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`drift_v1_student_visible_target.md`](./drift_v1_student_visible_target.md) — historical five-episode development result and withheld former final-test score
- [`../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) — reviewed cohort and 33-episode union correction
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md) — four-label Opus follow-up and revised 106/106-resolved union
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) — raw-input and scheduling reassessment
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md) — complete 292-case development review and expanded contract
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md) — Luna reasoning-effort comparison and approved development selection
- [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) — retired benchmark record
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation workflow
- [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) — Concept note for a possible future value-evolution filter
