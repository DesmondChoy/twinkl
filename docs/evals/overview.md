# Twinkl Evaluation Overview

This folder contains Twinkl evaluation specifications. The evaluations check
each component before work moves to the next stage. The folder also records
structural verification for controls that apply across live model calls.

---

## Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TWINKL EVALUATION FLOW                              │
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
| 3 | [`drift_detection_eval.md`](./drift_detection_eval.md) | Inference | Weekly Drift Detection finds Drift without unacceptable false Drift alerts and stores valid structured output | Drift recall first; false Drift alerts second; coverage and abstention diagnostic |
| 4 | [`explanation_quality_eval.md`](./explanation_quality_eval.md) | User Output | Coach Digest responses are grounded and useful | Likert ≥ 3.5/5 |
| — | [`live_prompt_boundary_verification.md`](./live_prompt_boundary_verification.md) | Live Model Boundary | Stable Twinkl instructions stay separate from user-controlled data | Structural message separation and fail-closed provenance; no attack-success score |

---

## Dependencies

The adopted architecture has a required user-facing evaluation path and a
separate completed VIF Critic research path:

```
Journal Entries + Core Values ──▶ drift_detection_eval ──▶ explanation_quality_eval
                                  (Weekly Drift Detection)   (Coach Digest)

judge_validation_eval ──▶ completed value_modeling_eval ──▶ VIF Critic research archive
```

**Implications:**
- The completed VIF Critic training used validated LLM-Judge labels
- The current user-facing Drift evaluation does not require VIF Critic input
- VIF Critic candidate confirmation is outside the remaining capstone scope
- Explanation quality can be partially tested at any stage (rationales work independently)

---

## Current Status Summary

| Eval | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| LLM-Judge Validation | 🟢 Operational | 1 651 labels across 204 personas; the shared 115-Journal-Entry / 19-persona benchmark yields Fleiss' κ **0.56** and avg LLM-Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up work includes the `twinkl-747` reachability report, the `twinkl-754` 5-pass consensus rerun, the completed `twinkl-a30f` Security target repair, and the completed `twinkl-748` Hedonism hard-set review. | Add automated post-label QA. Optional human re-annotation remains future work. No further VIF Critic target work is planned for the time-boxed capstone. |
| Value Modeling | ✅ Complete for capstone POC | The VIF Critic training and evaluation stack is complete. `run_019`-`run_021` remains the historical corrected-split reference. The paired `run_057`-`run_062` experiment shows that active-state Security repair raises median test Security QWK by about **0.17**. `twinkl-j0ck` did not promote soft targets, compact-history `run_069` failed its seed-11 expansion gate, and the Codex-reviewed `twinkl-748` Hedonism hard-set found only 0.05 median `-1` recall and 0.05 strict-pair accuracy for the incumbent. `twinkl-6mrt` implemented recall-first checkpoint selection and nominated `run_060` for offline use. The [`twinkl-752.5` reassessment](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) found no reliable benefit from exposing raw VIF Critic Predictions to the Weekly Drift Reviewer and no scheduling recall gain. | No further VIF Critic work is planned for the time-boxed capstone. The VIF Critic remains outside the user-facing Drift path. |
| Weekly Drift Detection | 🟡 Development-only | The workflow is complete and wired for the capstone POC. Its internal Weekly Drift Reviewer and Drift Detector use the fixed Luna-low contract and store structured output. The complete development review contains 42 Drifts across 36 Drift trajectories in 292 resolved cases. Luna-low had median Drift recall of `0.548`, 4 false Drift alerts, and `0.637` coverage. The later `twinkl-ck3w` study found `0.667` recall and 9 false Drift alerts for Luna-`xhigh`; Twinkl retains Luna-low because `xhigh` is a more aggressive operating point. The evidence is AI-reviewed synthetic development evidence, not human validation or a final test. | No further capstone model comparison is planned; no fresh final test or deployment approval is claimed. |
| Coach Digest Explanation Quality | 🟡 Partial | Coach Digest Validations, batch reporting, and Coach Digest Evals are implemented. The eval scores are AI review, not human validation. | Run and record the current Coach Digest Validations and Coach Digest Evals results. Future human calibration of the AI review remains separate work. Rationale-specific evaluation also remains separate work. |
| Live Prompt Boundary | ✅ Structurally verified | `live-prompt-boundary-v1` separates stable instructions from user-controlled JSON for the live Nudge, Weekly Drift Reviewer, and Coach Digest calls. Tests cover instruction-like and boundary-like text, provider field mapping, Nudge and Weekly Drift Reviewer prompt provenance, and existing validation paths. | A live provider evaluation can measure prompt injection attack success. It is not a capstone acceptance requirement. |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`live_prompt_boundary_verification.md`](./live_prompt_boundary_verification.md) — live prompt trust boundary, structural evidence, and claim limit
- [`drift_v1_student_visible_target.md`](./drift_v1_student_visible_target.md) — historical five-episode development result and withheld former final-test score
- [`../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) — reviewed cohort and 33-episode union correction
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md) — four-label Opus follow-up and revised 106/106-resolved union
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) — raw-input and scheduling reassessment
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md) — complete 292-case development review and expanded contract
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md) — evidence behind the fixed Luna-low model contract
- [`../../logs/experiments/reports/experiment_review_2026-08-09_twinkl_ck3w_luna_higher_reasoning.md`](../../logs/experiments/reports/experiment_review_2026-08-09_twinkl_ck3w_luna_higher_reasoning.md) — higher-reasoning results and no-change Luna-low decision
- [`../demo/weekly_drift_review_app.md`](../demo/weekly_drift_review_app.md) — read-only inspection of the frozen Weekly Drift Reviewer development Runs
- [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) — retired benchmark record
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation workflow
- [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) — Concept note for a possible future value-evolution filter
