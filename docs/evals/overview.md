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
| Judge Validation | 🟢 Operational | 1 651 labels across 204 personas; the shared 115-entry / 19-persona benchmark yields Fleiss' κ **0.56** and avg Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up audits now include the `twinkl-747` reachability report and the `twinkl-754` 5-pass consensus/self-consistency rerun. | Add automated post-label QA (all-zero rate, sparsity, distribution) and continue hard-dimension target refinement, especially after the `Security` caveat from `twinkl-747` |
| Value Modeling | 🟡 In Progress | Experiment archive now spans 50 run IDs / 114 persisted configs; active corrected-split frontier is `run_019`-`run_021` BalancedSoftmax with median QWK **0.362**, `recall_-1` **31.3%**, minority recall **44.8%**, and logged circumplex summaries. Qwen (`run_042`-`run_044`), two-stage (`run_045`-`run_047`), and consensus-label (`run_048`-`run_050`) diagnostics did not replace it ([experiment index](../../logs/experiments/index.md)). | Build the matched hard-set (`twinkl-748`), prototype compact student context (`twinkl-749`), quantify training-signal divergence (`twinkl-751`), then revisit gated PEFT work (`twinkl-750`); persona-level aggregation for Top-K accuracy |
| Drift Detection | 🟡 Partial | Critic uncertainty is implemented, and the runtime path now includes weekly aggregation, crash/rut-style detection experiments, and Coach-facing orchestration in [`src/vif/runtime.py`](../../src/vif/runtime.py), [`src/vif/drift.py`](../../src/vif/drift.py), and [`src/coach/runtime.py`](../../src/coach/runtime.py), with tests in `tests/vif/` | Critic QWK is still too low for trustworthy triggers; thresholds are uncalibrated; no synthetic crisis-injection benchmark or end-to-end hit-rate/precision/recall report yet. Evolution gating remains an undecided idea, not part of the active evaluation contract |
| Explanation Quality | 🟡 Partial | 1 594 / 1 651 persisted labels have rationales, the annotation tool can display/compare them, and Coach Tier 1 narrative checks are implemented in [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py) | Batch pass-rate reporting for Coach narratives, rationale-specific Tier 1 checks in `src/judge/`, and deeper meta-judge / human-calibration work |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation pipeline
- [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) — Concept note for a possible future value-evolution filter
