# VIF Evaluation Pipeline

This folder contains evaluation specifications for the **Value Identity Function (VIF)** — a sequential validation pipeline that ensures each component works correctly before moving to the next stage.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VIF EVALUATION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
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
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Order | Eval File | Pipeline Stage | What It Validates | Key Metric |
|:-----:|-----------|----------------|-------------------|------------|
| 1 | [`judge_validation_eval.md`](./judge_validation_eval.md) | Data Preparation | Judge labels are consistent & agree with humans | Cohen's κ > 0.60 |
| 2 | [`value_modeling_eval.md`](./value_modeling_eval.md) | Model Training | VIF learns value hierarchies correctly | Spearman ρ > 0.7 |
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

| Eval | Status | Blockers |
|------|--------|----------|
| Judge Validation | 🟡 Partial | Needs human annotations for κ comparison |
| Value Modeling | 🔴 Blocked | Requires Critic model implementation |
| Drift Detection | 🔴 Blocked | Requires trained Critic + MC Dropout |
| Explanation Quality | 🟡 Partial | None — Tier 1 checks can start now |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/VIF/`](../VIF/) — VIF architecture documentation
- [`docs/PRD.md`](../PRD.md) — Product requirements (Evaluation Strategy section)
- [`docs/synthetic_data/pipeline_specs.md`](../synthetic_data/pipeline_specs.md) — Data generation pipeline
