# Experiment Review — 2026-07-10 — `twinkl-wq9p` Drift-Trigger Benchmark

## Decision

No scorer is promotion-ready. At least one arm clears the deliberately explicit designed holdout but not the frozen consensus evaluation. Human review of the cross-set disagreement must precede production wiring or a cascade decision.

## Scope and evidence

- Strict reference: two adjacent stored consensus `-1` labels on the same declared core value.
- Soft detector: the mean `P(-1)` over an adjacent pair plus a maximum uncertainty gate.
- Thresholds were selected on frozen validation personas only.
- Frozen test results are diagnostic because that split has only 5 strict episodes.
- Designed holdout: `twinkl_wq9p_drift_v1_designed_holdout`, SHA-256 `2c6f9c163b753bbc70a672a3d400d522490c16f609bee44f77983c9eb00aa1e3`.
- Designed holdout review status: `designed_not_human_reviewed`; it is not human ground truth.

## Frozen test comparison

| Arm | Evidence | Ref | Pred | Precision | Recall | F1 | Window FPR | Max latency | Recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run_020_selected | soft_probability | 5 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | N/A | N/A |
| run_025_same_budget_persisted | soft_probability | 5 | 3 | 0.333 | 0.200 | 0.250 | 0.014 | 0 | N/A |
| run_052_consensus_recall_0.02 | soft_probability | 5 | 1 | 0.000 | 0.000 | 0.000 | 0.007 | N/A | N/A |
| run_053_consensus_selected | soft_probability | 5 | 13 | 0.231 | 0.600 | 0.333 | 0.056 | 0 | 1.000 |
| llm_gpt-5.4-mini_student_visible | hard_class | 5 | 1 | 0.000 | 0.000 | 0.000 | 0.003 | N/A | N/A |
| llm_gpt-5.4-mini_human_context | hard_class | 5 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | N/A | N/A |

## Isolated designed holdout

POC targets: recall `>= 0.80`, precision `> 0.60`, F1 `> 0.50`, window false-positive rate `< 0.20`, and maximum confirmation-anchored latency `<= 2` entries.

| Arm | Evidence | Ref | Pred | Precision | Recall | F1 | Window FPR | Max latency | Recovery | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| run_020_selected | soft_probability | 10 | 1 | 1.000 | 0.100 | 0.182 | 0.040 | 0 | 0.000 | no |
| run_052_consensus_recall_0.02 | soft_probability | 10 | 3 | 0.667 | 0.200 | 0.308 | 0.120 | 1 | 0.000 | no |
| run_053_consensus_selected | soft_probability | 10 | 3 | 0.667 | 0.200 | 0.308 | 0.080 | 0 | 0.000 | no |
| llm_gpt-5.4-mini_student_visible | hard_class | 10 | 10 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 1.000 | yes |
| llm_gpt-5.4-mini_human_context | hard_class | 10 | 10 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 1.000 | yes |

## Cross-set interpretation

The designed holdout and frozen consensus split disagree sharply. This is not evidence that an LLM is generally perfect; it shows that observable, explicit sustained conflict is within the evaluated scorers' capability while the consensus-derived cases may depend on subtler context, disputed labels, or a different target contract.

Designed LLM episode hits: `llm_gpt-5.4-mini_student_visible` 10/10; `llm_gpt-5.4-mini_human_context` 10/10. Frozen LLM episode hits: `llm_gpt-5.4-mini_student_visible` 0/5; `llm_gpt-5.4-mini_human_context` 0/5.

Designed incumbent MLP episode hits: `run_020_selected` 1/10. Designed consensus-trained MLP episode hits: `run_052_consensus_recall_0.02` 2/10; `run_053_consensus_selected` 2/10. No consensus-trained arm clears both evaluation surfaces, so the current consensus retraining has not closed the decision-level recall gap.

Architecture consequence: keep the production trigger blocked unless one scorer passes both evaluation surfaces. Human-review the frozen reference episodes alongside the designed cases before deciding whether to repair labels/context, test an LLM verifier, or narrow the capstone claim to explicit conflict detection.

## Promotion regime

1. Primary: episode precision/recall/F1, window false-positive rate, and confirmation-anchored latency on an isolated holdout.
2. Supporting: entry-level `recall_-1` at a declared precision floor.
3. Diagnostic: QWK, hedging, calibration, and per-value slices.
4. Production trigger schema and Coach delivery integration remain in `twinkl-a2w`.

## Limitations

- The designed holdout is intentionally small and author-designed; human review is the next validity upgrade.
- LLM arms emit hard classes, not calibrated conflict probabilities, so their detector uses the strict two-label rule without uncertainty gating.
- Per-value results are descriptive only; the benchmark does not fit per-value thresholds.
- The consensus reference is more stable than one-pass labels but remains an AI-Judge reference rather than human ground truth.
