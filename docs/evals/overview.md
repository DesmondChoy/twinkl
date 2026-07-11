# VIF Evaluation Pipeline

This folder contains evaluation specifications for the **Value Identity Function (VIF)** — a sequential validation pipeline that ensures each component works correctly before moving to the next stage.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VIF EVALUATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │    Stage 1   │    │    Stage 2   │    │    Stage 3   │    │    Stage 4   │  │
│   │  Data Prep   │───▶│   Training   │───▶│  Inference   │───▶│ User Output  │  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│          │                   │                   │                   │          │
│          ▼                   ▼                   ▼                   ▼          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │    Judge     │    │    Value     │    │    Drift     │    │ Explanation  │  │
│   │  Validation  │    │   Modeling   │    │  Detection   │    │   Quality    │  │
│   │     Eval     │    │     Eval     │    │     Eval     │    │     Eval     │  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Order | Eval File | Pipeline Stage | What It Validates | Key Metrics |
|:-----:|-----------|----------------|-------------------|-------------|
| 1 | [`judge_validation_eval.md`](./judge_validation_eval.md) | Data Preparation | Judge labels are consistent & agree with humans | Cohen's κ > 0.60 |
| 2 | [`value_modeling_eval.md`](./value_modeling_eval.md) | Model Training | VIF learns value hierarchies correctly | Entry-level: QWK > 0.4, Minority Recall (-1) > 20%; Persona-level: Spearman ρ > 0.7 |
| 3 | [`drift_detection_eval.md`](./drift_detection_eval.md) | Inference | Drift triggers fire accurately on misalignment | Hit Rate ≥ 80% |
| 4 | [`explanation_quality_eval.md`](./explanation_quality_eval.md) | User Output | Explanations are grounded and useful | Likert ≥ 3.5/5 |

---

## Dependencies

Each eval builds on the previous stage:

```
judge_validation_eval  ─┐
(training data quality) │
                        ▼
            value_modeling_eval  ─┐
            (trained Critic)      │
                                  ▼
                      drift_detection_eval  ─┐
                      (uncertainty-aware     │
                       triggers)             │
                                             ▼
                              explanation_quality_eval
                              (end-to-end user value)
```

**Implications:**
- You cannot evaluate drift detection without a trained Critic model
- You cannot train the Critic without validated Judge labels
- Explanation quality can be partially tested at any stage (rationales work independently)

---

## Current Status Summary

| Eval | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| Judge Validation | 🟢 Operational | 1 651 labels across 204 personas; the shared 115-entry / 19-persona benchmark yields Fleiss' κ **0.56** and avg Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up audits include the `twinkl-747` reachability report and the `twinkl-754` 5-pass consensus/self-consistency rerun. | Add automated post-label QA (all-zero rate, sparsity, distribution) and continue hard-dimension target refinement, especially after the `Security` caveat from `twinkl-747` |
| Value Modeling | 🟡 In Progress | `run_019`-`run_021` remains the historical corrected-split reference. The paired `run_057`-`run_062` experiment shows that the completed active-state Security repair raises median test Security QWK by about **0.17** under both historical and repaired scoring lenses; absolute repaired-lens QWK remains **0.328**. The LLM context baseline separately shows potential upside from legal history ([experiment index](../../logs/experiments/index.md)). | Preserve review disagreement through `twinkl-j0ck`, build the matched hard-set (`twinkl-748`), prototype compact student context (`twinkl-749`), and compare candidates at the sustained-conflict decision layer before gated PEFT work (`twinkl-750`) |
| Drift Detection | 🟡 Partial | The former consensus-derived frozen benchmark is [retired historical evidence](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) after its audit showed it was not a fair student-visible promotion surface. [`twinkl-v8pb` completed a full-runtime-text review](./drift_v1_student_visible_target.md): `run_020` found 1/5 development episodes at its fixed threshold, while promotion case_023 remained unresolved across 19 entries. The promotion score was deliberately not performed. | Build a fresh, independently resolved promotion surface before any future promotion claim; do not fall back to the retired benchmark or score only agreed promotion cases. `twinkl-a2w` remains blocked. Production `P(-1)` persistence, the calibrated runtime detector, and active/recovered/mixed/uncertain Coach delivery remain pending. Evolution/fade/rise taxonomies remain parked. |
| Explanation Quality | 🟡 Partial | 1 594 / 1 651 persisted labels have rationales, the annotation tool can display/compare them, and Coach Tier 1 narrative checks are implemented in [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py) | Batch pass-rate reporting for Coach narratives, rationale-specific Tier 1 checks in `src/judge/`, and deeper meta-judge / human-calibration work |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`drift_v1_student_visible_target.md`](./drift_v1_student_visible_target.md) — completed development target and blocked locked-promotion result
- [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) — retired benchmark record
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation pipeline
- [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) — Concept note for a possible future value-evolution filter
