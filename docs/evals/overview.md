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
| User-facing | [`explanation_quality_eval.md`](./explanation_quality_eval.md) | Coach Digest | Coach Digest responses use the cited evidence, follow the response contract, and support future user review | Pass rates from Coach Digest Validations; four means from Coach Digest Evals; reflective-question result; future perceived-accuracy Likert rating |
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

| Eval | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| LLM-Judge Validation | 🟢 Operational | 1,651 Journal Entries across 204 personas contain 16,510 per-dimension LLM-Judge VIF Labels; the shared 115-Journal-Entry / 19-persona benchmark yields Fleiss' κ **0.56** and mean LLM-Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up work includes the `twinkl-747` reachability report, the `twinkl-754` 5-pass consensus rerun, the completed `twinkl-a30f` Security target repair, and the completed `twinkl-748` Hedonism hard-set review. | Add automated post-label QA. Optional human re-annotation remains future work. No further VIF Critic (Offline) target work is planned for the time-boxed capstone. |
| Value Modeling | ✅ Complete for capstone POC | The VIF Critic (Offline) training and evaluation stack is complete. `run_019`-`run_021` remains the historical corrected-split reference. The paired `run_057`-`run_062` experiment shows that active-state Security repair raises median test Security QWK by about **0.17**. `twinkl-j0ck` did not promote soft targets, compact-history `run_069` failed its seed-11 expansion gate, and the Codex-reviewed `twinkl-748` Hedonism hard-set found only 0.05 median `-1` recall and 0.05 strict-pair accuracy for the incumbent. `twinkl-6mrt` implemented recall-first checkpoint selection and nominated `run_060` for offline use. The [`twinkl-752.5 reassessment`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) found no reliable benefit from exposing raw VIF Critic Predictions to the Weekly Drift Reviewer and no scheduling recall gain. | No further VIF Critic (Offline) work is planned for the time-boxed capstone. It remains outside the user-facing Drift path. |
| Weekly Drift Detection | 🟡 Development-only | The workflow uses the fixed Luna-low contract. The complete development review contains 42 reference Drifts across 36 Drift trajectories in 292 resolved cases. Historical prompt-v2 Luna-low results had median Drift recall `0.548`, 4 false Drift alerts, and coverage `0.637`; Luna-`xhigh` had recall `0.667` and 9 false alerts. Runtime prompt `4.0` includes selected Core Value definitions and core motivations. The [v3/v4 comparison](../../logs/experiments/reports/experiment_review_2026-09-07_twinkl_j3k7_core_value_definitions.md) covers 204 Personas and 951 weeks, with detected Drift totals of 22/22/21 for v3 and 19/21/18 for v4 across three Runs each. It records Core Value counts, individual Drift changes, and repeat variation without establishing accuracy. | The historical accuracy metrics do not measure prompt v4. Evidence remains AI-reviewed synthetic development evidence without a fresh final test or deployment approval. |
| Coach Digest Explanation Quality | 🟡 Partial | The [current five-Persona sample](../../logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json) matches the public React key-week responses and their current input hashes. All five Luna-none, prompt-`4.2` responses pass Coach Digest Validations; they have no Coach Digest Evals result. The separate [August sample](../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json), for the previous Persona roster and inputs, passed all Coach Digest Validations and scored correctness `4.80`, specificity `5.00`, non-prescriptive tone `5.00`, and tension honesty `4.60`, with all questions passing and no failed verdicts or review flags. Evaluator options support independent OpenAI or Gemini selection and record generator/evaluator identities. The Drift/control runner selects 42 known development Drifts and 42 matched controls. | The August scores are same-model AI review and do not describe the current saved responses. Paid cross-provider Drift/control evaluation, the comparison report, and future human calibration remain incomplete. |
| North Star Moment | 🟡 Targeted AI comparison complete | The [v4 Run 1 comparison](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md) covers 501 weeks from 105 synthetic Personas, retaining the 81/24 development/final Persona partition. Both methods were reassessed on 38 affected cases; 463 retain earlier observations. Full eligible history achieved Card precision / Opportunity recall of `75.54% / 74.62%` in development and `82.35% / 81.40%` in qualified final evaluation; Nomic top-three retrieval achieved `58.58% / 55.35%` and `57.65% / 56.98%`. The shared evaluator is Luna-xhigh. Paired exclusions, failures, evaluator consistency, and whole-Persona confidence intervals are recorded. Saved replay preserves the full-history outcomes for five Personas and 27 weeks. | Human review is deferred. Qualified final histories have prior upstream research exposure; this is a targeted update with retained observations, not an untouched final test. Real-user benefit and deployment approval remain unestablished. Live execution requires a finalized integration budget. |
| Live Prompt Boundary | ✅ Structurally verified | `live-prompt-boundary-v1` separates stable instructions from user-controlled JSON for the live Nudge, Weekly Drift Reviewer, and Coach Digest calls. Tests cover instruction-like and boundary-like text, provider field mapping, Nudge and Weekly Drift Reviewer prompt provenance, and existing validation paths. | A live provider evaluation can measure prompt injection attack success. It is not a capstone acceptance requirement. |

See each eval file's **Implementation Status** section for detailed breakdowns.

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
- [`../demo/weekly_drift_review_app.md`](../demo/weekly_drift_review_app.md) — read-only inspection of the frozen Weekly Drift Reviewer development Runs
- [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) — retired benchmark record
- [`docs/pipeline/pipeline_specs.md`](../pipeline/pipeline_specs.md) — Data generation workflow
- [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) — Concept note for a possible future value-evolution filter
