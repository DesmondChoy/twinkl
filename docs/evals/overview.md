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
                  └──▶ Coach Digest explanation-quality evaluation

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
| Shared control | [`live_prompt_boundary_verification.md`](./live_prompt_boundary_verification.md) | Live model boundary | Stable Twinkl instructions stay separate from user-controlled data | Structural message separation and fail-closed provenance; no attack-success score |

---

## Dependencies

The adopted architecture has a required user-facing evaluation path and a
separate completed VIF Critic (Offline) research path:

```
Journal Entries + Core Values ──▶ drift_detection_eval ──▶ explanation_quality_eval
                                  (Weekly Drift Detection)   (Coach Digest)

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

---

## Current Status Summary

| Eval | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| LLM-Judge Validation | 🟢 Operational | 1,651 Journal Entries across 204 personas contain 16,510 per-dimension LLM-Judge VIF Labels; the shared 115-Journal-Entry / 19-persona benchmark yields Fleiss' κ **0.56** and mean LLM-Judge-human Cohen's κ **0.66** ([report](../../logs/exports/agreement_report_20260318_130642.md)). Follow-up work includes the `twinkl-747` reachability report, the `twinkl-754` 5-pass consensus rerun, the completed `twinkl-a30f` Security target repair, and the completed `twinkl-748` Hedonism hard-set review. | Add automated post-label QA. Optional human re-annotation remains future work. No further VIF Critic (Offline) target work is planned for the time-boxed capstone. |
| Value Modeling | ✅ Complete for capstone POC | The VIF Critic (Offline) training and evaluation stack is complete. `run_019`-`run_021` remains the historical corrected-split reference. The paired `run_057`-`run_062` experiment shows that active-state Security repair raises median test Security QWK by about **0.17**. `twinkl-j0ck` did not promote soft targets, compact-history `run_069` failed its seed-11 expansion gate, and the Codex-reviewed `twinkl-748` Hedonism hard-set found only 0.05 median `-1` recall and 0.05 strict-pair accuracy for the incumbent. `twinkl-6mrt` implemented recall-first checkpoint selection and nominated `run_060` for offline use. The [`twinkl-752.5 reassessment`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) found no reliable benefit from exposing raw VIF Critic Predictions to the Weekly Drift Reviewer and no scheduling recall gain. | No further VIF Critic (Offline) work is planned for the time-boxed capstone. It remains outside the user-facing Drift path. |
| Weekly Drift Detection | 🟡 Development-only | The workflow is complete and wired for the capstone POC. Its internal Weekly Drift Reviewer and Drift Detector use the fixed Luna-low contract and store structured output. The complete development review contains 42 Drifts across 36 Drift trajectories in 292 resolved cases. Luna-low had median Drift recall of `0.548`, 4 false Drift alerts, and `0.637` coverage. The later `twinkl-ck3w` study found `0.667` recall and 9 false Drift alerts for Luna-`xhigh`; Twinkl retains Luna-low because `xhigh` is a more aggressive operating point. The evidence is AI-reviewed synthetic development evidence, not human validation or a final test. | No further capstone model comparison is planned; no fresh final test or deployment approval is claimed. |
| Coach Digest Explanation Quality | 🟡 Partial | The key-week responses for the five deployed Personas are identical in the public React fixtures and [evaluation manifest](../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json). All five passed all Coach Digest Validations. Coach Digest Evals scored mean correctness `4.80`, specificity `5.00`, non-prescriptive tone `5.00`, and tension honesty `4.60`; all questions passed, with no failed verdicts or review flags. Coach Digest Evals support independent OpenAI or Gemini evaluator selection and record generator/evaluator identities. The deterministic Drift/control runner selects 42 known development Drifts and 42 matched controls from the committed inputs. | The committed result remains Luna-none same-model AI review, not human validation or a fresh final test. The paid cross-provider Drift/control run, comparison report, and future human calibration remain separate work. |
| Live Prompt Boundary | ✅ Structurally verified | `live-prompt-boundary-v1` separates stable instructions from user-controlled JSON for the live Nudge, Weekly Drift Reviewer, and Coach Digest calls. Tests cover instruction-like and boundary-like text, provider field mapping, Nudge and Weekly Drift Reviewer prompt provenance, and existing validation paths. | A live provider evaluation can measure prompt injection attack success. It is not a capstone acceptance requirement. |

See each eval file's **Implementation Status** section for detailed breakdowns.

---

## References

- [`docs/vif/`](../vif/) — VIF architecture documentation
- [`docs/prd.md`](../prd.md) — Product requirements (Evaluation Strategy section)
- [`coach_narrative_test_and_eval_guide.md`](./coach_narrative_test_and_eval_guide.md) — exact Coach Digest Validations, Coach Digest Evals, and Drift/control study commands
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
