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
| Judge Validation | 🟢 Operational | 1 555 labels across 192 personas; 3 annotators × 46 entries; aggregate Cohen's κ 0.57–0.76 ([report](../../logs/exports/agreement_report_20260128_133444.md)) | Automated quality checks (all-zero rate, sparsity) |
| Value Modeling | 🟡 In Progress | 9 runs × 4+ losses; best QWK **0.413** (run_007 CORN) — fair but below moderate target; -1 recall **10.3%** — model nearly blind to misalignment; hedging >80% ([experiment index](../../logs/experiments/index.md)) | Boost minority recall via class-imbalance interventions; boost QWK; persona-level aggregation for Top-K accuracy |
| Drift Detection | 🟡 Partial | Critic trained + MC Dropout implemented ([`src/vif/critic.py`](../../src/vif/critic.py)) | Critic QWK too low for reliable triggers; crash/rut trigger code; crisis-injection test data |
| Explanation Quality | 🟡 Partial | 133/134 rationales stored in parquet; display UI operational | Tier 1 automated checks (groundedness, circularity, length) |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation pipeline
