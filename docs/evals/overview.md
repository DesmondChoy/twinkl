# Twinkl Evaluation Overview

This folder contains Twinkl evaluation specifications for the required
user-facing path and the separate VIF Critic (Offline) research path. It also
records structural verification for controls that apply across live model
calls.

---

## Evaluation Flow

```text
User-facing path
Journal Entries + Core Values
        └──▶ Weekly Drift Detection evaluation
                  ├──▶ Coach Digest explanation-quality evaluation
                  └──▶ North Star Moment quotation and omission evaluation

Offline research path
Synthetic Journal Entries
        └──▶ LLM-Judge VIF Label validation
                  └──▶ VIF Critic (Offline) value-modeling evaluation
```

---

## Quick Reference

| Path | Eval File | Stage | What It Validates | Key Metrics |
|:-----|-----------|----------------|-------------------|-------------|
| Offline research | [`judge_validation_eval.md`](./judge_validation_eval.md) | Data Preparation | LLM-Judge VIF Labels have bounded repeated-call consistency and overlap with project-team annotations | Per-dimension repeated-call Fleiss' κ; human-human Fleiss' κ; LLM-Judge-human Cohen's κ; prevalence and sample limits |
| Offline research | [`value_modeling_eval.md`](./value_modeling_eval.md) | Model Training | VIF Critic (Offline) recovers Conflict | Primary: entry-level `recall_-1`; mandatory precision-recall reporting; QWK and `+1` diagnostic |
| User-facing | [`drift_detection_eval.md`](./drift_detection_eval.md) | Weekly Drift Detection | Weekly Drift Detection finds Drift without unacceptable false Drift alerts and stores valid structured output | Drift recall first; false Drift alerts second; coverage and abstention diagnostic |
| User-facing | [`explanation_quality_eval.md`](./explanation_quality_eval.md) | Coach Digest | Coach Digest responses use the cited evidence and follow the response contract under code checks and AI review | Pass rates from Coach Digest Validations; four means from Coach Digest Evals; reflective-question result |
| User-facing | [`nsm_experiment_methodology.md`](../north_star/nsm_experiment_methodology.md) | North Star Moment | Selected exact quotations and no-card outcomes satisfy source eligibility, supportive-action, and selection rules under shared AI assessment | Card precision; Opportunity recall; correct omission; selection-rule correctness; evaluator consistency; cost and latency diagnostics |
| Shared control | [`live_prompt_boundary_verification.md`](./live_prompt_boundary_verification.md) | Live model boundary | Stable Twinkl instructions stay separate from user-controlled data | Structural message separation and fail-closed provenance; no attack-success score |

---

## Dependencies

The adopted architecture has a required user-facing evaluation path and a
separate completed VIF Critic (Offline) research path:

```
Journal Entries + Core Values ──▶ drift_detection_eval
                                  (Weekly Drift Detection)
                                    ├──▶ explanation_quality_eval (Coach Digest)
                                    └──▶ nsm_experiment_methodology (North Star Moment)

judge_validation_eval ──▶ completed value_modeling_eval ──▶ VIF Critic (Offline) research archive
```

**Implications:**
- The completed VIF Critic (Offline) training used persisted LLM-Judge VIF
  Labels with bounded project-team agreement and repeated-call evidence
- The current user-facing Drift evaluation does not require VIF Critic
  Predictions
- Weekly Drift Reviewer confirmation of cases nominated by VIF Critic
  Predictions is outside the remaining capstone scope
- Explanation quality can be partially tested at any stage (rationales work independently)
- North Star Moment evaluation consumes fixed Weekly Drift Detection results
  and eligible writing. Its quotation results do not establish Weekly Drift
  accuracy, and No Active Drift alone does not establish a supportive action.

---

## Current Status Summary

Completion refers to the accepted capstone scope. Human calibration and the
user pilot are not done and closed outside that scope. A completed evaluation
workflow does not imply that every supported study has a paid result.

| Eval | Status | Evidence | Limits and unfinished scope |
|------|--------|----------|----------------|
| LLM-Judge Validation | Done | 1,651 Journal Entries across 204 personas contain 16,510 per-dimension LLM-Judge VIF Labels; the shared 115-Journal-Entry / 19-persona benchmark yields Fleiss' κ **0.56** and mean LLM-Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up work includes the `twinkl-747` reachability report, the `twinkl-754` 5-pass consensus rerun, the completed `twinkl-a30f` Security target repair, and the completed `twinkl-748` Hedonism hard-set review. | Add automated post-label QA. Optional human re-annotation remains future work. No further VIF Critic (Offline) target work is planned for the time-boxed capstone. |
| Value Modeling | Done | The VIF Critic (Offline) training and evaluation stack is complete. `run_019`-`run_021` remains the historical corrected-split reference. The paired `run_057`-`run_062` experiment shows that active-state Security repair raises median test Security QWK by about **0.17**. `twinkl-j0ck` did not promote soft targets, compact-history `run_069` failed its seed-11 expansion gate, and the Codex-reviewed `twinkl-748` Hedonism hard-set found only 0.05 median `-1` recall and 0.05 strict-pair accuracy for the incumbent. `twinkl-6mrt` implemented recall-first checkpoint selection and nominated `run_060` for offline use. The [`twinkl-752.5 reassessment`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) found no reliable benefit from exposing raw VIF Critic Predictions to the Weekly Drift Reviewer and no scheduling recall gain. | No further VIF Critic (Offline) work is planned for the time-boxed capstone. It remains outside the user-facing Drift path. |
| Weekly Drift Detection | Done | The live workflow uses `gpt-6-luna` at `low`; saved Runs retain their recorded models. The complete development review contains 42 reference Drifts across 36 Drift trajectories in 292 resolved cases. Historical prompt-v2 Luna-low results had median Drift recall `0.548`, 4 false Drift alerts, and coverage `0.637`; Luna-`xhigh` had recall `0.667` and 9 false alerts. Runtime prompt `4.0` includes selected Core Value definitions and core motivations. The [v3/v4 comparison](../../logs/experiments/reports/experiment_review_2026-09-07_twinkl_j3k7_core_value_definitions.md) covers 204 Personas and 951 weeks, with detected Drift totals of 22/22/21 for v3 and 19/21/18 for v4 across three Runs each. It records Core Value counts, individual Drift changes, and repeat variation without establishing accuracy. Matched prompt-v4 comparisons of `gpt-5.6-luna` and `gpt-6-luna` from `none` through `xhigh` ([`twinkl-q6pt`](../../logs/experiments/reports/experiment_review_2026-09-23_twinkl_q6pt_luna_model.md), [`twinkl-gv3x`](../../logs/experiments/reports/experiment_review_2026-09-23_twinkl_gv3x_luna_higher.md)) find no paired Drift-recall difference; GPT-6 Luna has higher coverage at every effort and lower calculated token cost. At `low`, median Drift recall is `0.381` for GPT-5.6 Luna and `0.405` for GPT-6 Luna. | The prompt-v2 accuracy metrics do not measure prompt v4; the prompt-v4 comparisons support the current GPT-6 Luna low contract on cost and coverage, with no established recall gain. Evidence remains AI-reviewed synthetic development evidence without a fresh final test or deployment approval. |
| Coach Digest Explanation Quality | Done for the accepted workflow | The [voice refresh and repairs](../../logs/experiments/reports/coach_voice_refresh_20260909/report.md) cover 27 saved baseline responses with Luna-none and prompt `4.4`; 22 eligible weeks have [saved comparisons](../north_star/demo_coach_comparison.md). Ordinary generation uses prompt `4.7` and complete selected Journal Entry text. The [source-context experiment](../../logs/experiments/reports/coach_context_eval_20260913/report.md) records six generations and nine same-model evaluator calls with archived prompt `4.6` and evaluator `3.1`; it identifies source-attribution limits and a missed known error. The separate [August sample](../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json) contains five earlier responses, with correctness `4.80`, specificity `5.00`, non-prescriptive tone `5.00`, and tension honesty `4.60`. Evaluator options support independent OpenAI or Gemini selection. The Drift/control runner selects 42 known development Drifts and 42 matched controls. | The August scores do not describe current replay responses. Mechanical checks and AI editorial review do not establish factual accuracy or human validity. Independent-provider evaluation and the 42-Drift/42-control workflow are done within the accepted scope. No paid independent study result is committed or required for closeout. Factual-attribution and saved-output semantic follow-ups remain not done under `twinkl-z5nr` and `twinkl-rklc.39`. |
| North Star Moment | Done | The [v4 Run 1 comparison](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md) covers 501 weeks from 105 synthetic Personas, retaining the 81/24 development/final Persona partition. Both methods were reassessed on 38 affected cases; 463 retain earlier observations. Full eligible history achieved Card precision / Opportunity recall of `75.54% / 74.62%` in development and `82.35% / 81.40%` in qualified final evaluation; Nomic top-three retrieval achieved `58.58% / 55.35%` and `57.65% / 56.98%`. The shared evaluator is Luna-xhigh. Paired exclusions, failures, evaluator consistency, and whole-Persona confidence intervals are recorded. Saved replay preserves the full-history outcomes for five Personas and 27 weeks. | Human calibration and a user pilot are not done and closed outside the capstone scope. Qualified final histories have prior upstream research exposure; this is a targeted update with retained observations, not an untouched final test. Real-user benefit and deployment approval remain unestablished. A selected passage appears within a valid Coach Digest after its narrative and reflective question. Live execution uses a pinned US$1 allowance shared across sessions and restarts; invalid or exhausted budgets fail closed. |
| Live Prompt Boundary | Done | `live-prompt-boundary-v1` separates stable instructions from user-controlled JSON for the live Nudge, Weekly Drift Reviewer, and Coach Digest calls. Tests cover instruction-like and boundary-like text, provider field mapping, Nudge and Weekly Drift Reviewer prompt provenance, and existing validation paths. | A live provider evaluation can measure prompt injection attack success. It is not a capstone acceptance requirement. |
| Independent-provider Coach Digest evaluation | Done | Evaluator provider/model selection and generator/evaluator identity reporting are implemented. | No committed paid independent-provider result; none is required for capstone closeout. |
| 42-Drift/42-control study | Done | Deterministic target selection, generation, safe resume, and grouped comparison reporting are implemented. | No committed paid study result; none is required for capstone closeout. |
| Human calibration | Not done — closed | No human calibration result for Coach Digest explanations or North Star Moment assessments. | Outside the capstone scope; AI assessments remain identified as AI evidence. |
| User pilot | Not done — closed | No real-user pilot result. | Outside the capstone scope; real-user usefulness remains unestablished. |

See each evaluation specification for its implementation status and evidence limits.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`coach_narrative_test_and_eval_guide.md`](./coach_narrative_test_and_eval_guide.md) — exact Coach Digest Validations, Coach Digest Evals, and Drift/control study commands
- [`../north_star/north_star_moment.md`](../north_star/north_star_moment.md) — North Star Moment eligibility, quotation, and display contract
- [`../north_star/nsm_experiment_methodology.md`](../north_star/nsm_experiment_methodology.md) — NSM cohort, comparison, paired metrics, evidence limits, and frozen-code reproduction
- [`live_prompt_boundary_verification.md`](./live_prompt_boundary_verification.md) — live prompt trust boundary, structural evidence, and claim limit
- [`drift_v1_student_visible_target.md`](./drift_v1_student_visible_target.md) — historical five-Drift development result and withheld former final-test score
- [`../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) — reviewed cohort and 33-episode union correction
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md) — four-label Opus follow-up and revised 106/106-resolved union
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) — raw-input and scheduling reassessment
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md) — complete 292-case development review and expanded contract
- [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md) — evidence behind the fixed Luna-low model contract
- [`../../logs/experiments/reports/experiment_review_2026-08-09_twinkl_ck3w_luna_higher_reasoning.md`](../../logs/experiments/reports/experiment_review_2026-08-09_twinkl_ck3w_luna_higher_reasoning.md) — higher-reasoning results and no-change Luna-low decision
- [`../../logs/experiments/reports/experiment_review_2026-09-23_twinkl_q6pt_luna_model.md`](../../logs/experiments/reports/experiment_review_2026-09-23_twinkl_q6pt_luna_model.md) — prompt-v4 GPT-5.6/GPT-6 Luna comparison at `none` and `low`
- [`../../logs/experiments/reports/experiment_review_2026-09-23_twinkl_gv3x_luna_higher.md`](../../logs/experiments/reports/experiment_review_2026-09-23_twinkl_gv3x_luna_higher.md) — prompt-v4 GPT-5.6/GPT-6 Luna comparison at `medium`, `high`, and `xhigh`
- [`../demo/weekly_drift_review_app.md`](../demo/weekly_drift_review_app.md) — read-only inspection of the frozen Weekly Drift Reviewer development Runs
- [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) — retired benchmark record
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation workflow
- [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) — Concept note for a possible future value-evolution filter
