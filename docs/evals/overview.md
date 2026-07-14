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
| 3 | [`drift_detection_eval.md`](./drift_detection_eval.md) | Inference | Drift Detector finds Drift without unacceptable false Drift alerts | Primary: Drift recall; precision / false-alert tolerance deferred until VIF Critic development |
| 4 | [`explanation_quality_eval.md`](./explanation_quality_eval.md) | User Output | Explanations are grounded and useful | Likert ≥ 3.5/5 |

---

## Dependencies

Each eval builds on the previous stage:

```
judge_validation_eval  ─┐
(training data quality) │
                        ▼
            value_modeling_eval  ─┐
            (trained VIF Critic)      │
                                  ▼
                      drift_detection_eval  ─┐
                      (uncertainty-aware     │
                       triggers)             │
                                             ▼
                              explanation_quality_eval
                              (end-to-end user value)
```

**Implications:**
- You cannot evaluate VIF Critic-based Drift detection without a trained VIF Critic
- You cannot train the VIF Critic without validated LLM-Judge labels
- Explanation quality can be partially tested at any stage (rationales work independently)

---

## Current Status Summary

| Eval | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| LLM-Judge Validation | 🟢 Operational | 1 651 labels across 204 personas; the shared 115-Journal-Entry / 19-persona benchmark yields Fleiss' κ **0.56** and avg LLM-Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up audits include the `twinkl-747` reachability report and the `twinkl-754` 5-pass consensus/self-consistency rerun. | Add automated post-label QA (all-zero rate, sparsity, distribution) and continue hard-dimension target refinement, especially after the `Security` caveat from `twinkl-747` |
| Value Modeling | 🟡 In Progress | `run_019`-`run_021` remains the historical corrected-split reference. The paired `run_057`-`run_062` experiment shows that active-state Security repair raises median test Security QWK by about **0.17**. `twinkl-j0ck` did not promote soft targets, compact-history `run_069` failed its seed-11 expansion gate, and the Codex-reviewed `twinkl-748` Hedonism hard-set found only 0.05 median `-1` recall and 0.05 strict-pair accuracy for the incumbent. [`twinkl-752.1`](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_752_1_weekly_verifier_ablation.md) found that adding `run_020` signals changed median Drift recall from 0.40 to 0.20, but only five episodes supported that comparison. [`twinkl-752.3`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_3_weekly_drift_reviewer_prompt_alignment.md) found that prompt alignment lowered median Drift recall to 0.20 and raised median false Drift alerts to 5. | Run `twinkl-752.5` on the 29-episode known-development union to reassess raw MLP input and test whether the MLP has a scheduling role; keep QWK and `+1` as diagnostics. Any MLP result on training-seen Journal Entries is in-sample. |
| Drift Detection | 🟡 Partial | [`twinkl-752.4`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) reviewed all 52 legacy-discoverable candidate trajectories plus 52 matched legacy-negative controls and found 27 episodes across 23 Drift trajectories. Three overlap the earlier five; adding the two omitted prior episodes produces the 29-episode / 25-Drift-trajectory known-development union. The retired-test audit adds four separate episodes, and four `twinkl-752.4` trajectories remain uncertain. This is selection-biased AI-reviewed evidence, and candidate mining can miss Drifts absent from both legacy sources. | Use the 29-episode union in `twinkl-752.5` to compare weekly-only, weekly-with-raw-MLP-input, MLP-triggered early-plus-weekly, and budget-matched model-free schedules. Keep the retired audit separate and build a fresh final test before any deployment claim. Production `P(-1)` persistence, calibrated Drift detection, and Weekly Coach delivery remain pending. |
| Explanation Quality | 🟡 Partial | 1 594 / 1 651 persisted labels have rationales, the annotation tool can display/compare them, and Weekly Coach Tier 1 narrative checks are implemented in [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py) | Batch pass-rate reporting for Weekly Coach narratives, rationale-specific Tier 1 checks in `src/judge/`, and deeper rationale-review LLM / human-calibration work |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`drift_v1_student_visible_target.md`](./drift_v1_student_visible_target.md) — historical five-episode development result and withheld former final-test score
- [`../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) — reviewed cohort, 29-episode union correction, and four-setup handoff
- [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) — retired benchmark record
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation workflow
- [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) — Concept note for a possible future value-evolution filter
