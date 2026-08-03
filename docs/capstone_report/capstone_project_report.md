# Twinkl Capstone Project Report

- **Working title:** Twinkl: An Inner Compass for Alignment Between Behavior and Core Values
- Author names and student numbers
- NUS programme, module, and submission date
- Phase 2 Technical Paper requirement: [IS Capstone Requirements](../is_capstone_slides.pdf)

## Abstract

- Problem, objective, and target users
- Longitudinal synthetic Journal Entry data and label workflow
- Small-model-first motivation for the VIF Critic
- Main VIF Critic findings and limits
- Evidence for the architecture change
- Weekly Drift Reviewer and deterministic Drift Detector
- Main quantitative results
- Weekly Drift Detection, Coach Digest, and capstone POC limits
- Live prompt trust boundary and structural verification

## Keywords

- Core Values
- Journal Entries
- Value alignment
- Drift detection
- LLM evaluation

## 1. Introduction

### 1.1 Problem and Objective

- Gap between declared Core Values and daily behavior
- Limits of mood summaries and isolated Journal Entry analysis
- Need for evidence-based weekly reflection
- Longitudinal comparison of Journal Entries with Core Values
- Explainable identification of Conflict and Drift
- Evidence-grounded Coach Digest response and reflective question
- Time-boxed academic POC
- Product intent in the [Product Requirements Document](../prd.md)

### 1.2 Research Questions and Contributions

- Human agreement with LLM-Judge VIF Labels
- VIF Critic Conflict and Drift performance against user-facing requirements
- Evidence that led from the VIF Critic to the Weekly Drift Reviewer
- Weekly Drift Reviewer and Drift Detector recovery of known Drifts
- Coach Digest evidence grounding and validation
- Longitudinal synthetic data and provenance-aware label workflow
- 69 Runs and 133 VIF Critic experiment configurations
- Small-model-first evaluation and evidence-led architecture decision
- Fixed Weekly Drift Reviewer and deterministic Drift Detector
- Versioned Weekly Drift Detection output and Experience and Inspect demo
- Provider instruction fields that separate stable Twinkl rules from
  user-controlled JSON data

### 1.3 Scope and Implementation Status

- **Complete offline research:** synthetic data, LLM-Judge VIF Labels, VIF Critic, and experiment record
- **Complete capstone POC:** Weekly Drift Detection with the fixed Weekly Drift
  Reviewer contract and Drift Detector
- **Limited benchmark:** human annotation
- **Development evidence only:** Weekly Drift Reviewer and Drift Detector evaluation
- **Experimental:** Coach Digest
- **Supporting POCs:** Onboarding and Conversational Nudge
- **Assessment-only deployment:** public Experience and Inspect demo on Railway
- **Implemented structural control:** live model instructions are separate from
  user-controlled JSON data
- **In progress:** professor walkthrough evidence
- No fresh final test or deployment approval
- User-facing Drift path separated from offline VIF Critic research

## 2. Related Work

### 2.1 Human Values and Reflective Tools

- Schwartz theory of basic human values
- Ten-value ontology used by Twinkl
- Best-Worst Scaling and the Schwartz Values Best-Worst Survey (SVBWS)
- Declared-value models versus post-hoc theme discovery
- Long-horizon reflection and accountability
- Evidence-grounded versus generic reflection
- Non-prescriptive coaching language
- Sources in the [April 2026 Project Proposal](../archive/capstone/2026-04-proposal-submission/April_Project_Proposal.md)

### 2.2 Value Modeling, Synthetic Data, and LLM Evaluation

- Ternary value alignment prediction
- Ordinal classification and uncertainty
- Rare-class and class-imbalance methods
- Persona-controlled longitudinal data generation
- LLM-created training and reference labels
- Human validation of LLM-created labels
- Label uncertainty, leakage, and synthetic bias
- Consecutive Conflict as Drift
- Explanation evaluation in [Explanation Quality Evaluation](../evals/explanation_quality_eval.md)

## 3. System Design and Methods

### 3.1 End-to-End Architecture

- SVBWS onboarding creates the confirmed Profile and Core Values
- Journal Entries grouped by Monday-to-Sunday week
- Weekly review starts only after the Monday-to-Sunday week closes
- Weekly Drift Reviewer without VIF Critic input
- Drift Detector across week boundaries
- Weekly Drift Detection output with cited Journal Entries
- Optional Coach Digest response and reflective question
- Core Values as the only values eligible for Drift
- No future-Journal-Entry or generation-metadata leakage
- Versioned receipts, prompts, schemas, and request hashes
- OpenAI `instructions` and Gemini `system_instruction` fields for stable
  Nudge, Weekly Drift Reviewer, and Coach Digest rules
- Separate JSON data for Journal Entries, nudge responses, preferred names,
  and current focus text
- `live-prompt-boundary-v1` Inspect receipt, with receipt hashes for the live
  Nudge and Weekly Drift Reviewer calls
- Design in [End-to-End Architecture](../architecture/e2e_architecture.md)

### 3.2 Data, Labels, and Human Validation

- 204 personas and 1,651 Journal Entries
- Demographic and Core Value variation
- Parallel generation between personas and sequential generation within each persona
- Banned-term and value-leakage controls
- Production-like Journal Entry representation
- Ten ternary LLM-Judge VIF Labels with rationale and provenance
- LLM-Judge Conflict Labels for the student-visible Drift development target
- Independent review lanes and disagreement adjudication
- Label consolidation, schema checks, and Security target repair
- Consensus labels used as diagnostic evidence only
- Three human annotators and 380 saved annotations
- Shared 115-Journal-Entry benchmark across 19 personas
- Cohen's kappa, Fleiss' kappa, and ambiguous-case review
- Workflow in [Synthetic Data Pipeline](../pipeline/pipeline_specs.md)

### 3.3 Weekly Drift Review and Detection

- Fixed `gpt-5.6-luna` contract with reasoning effort `low`
- Cumulative student-visible Journal Entry history
- Conflict, Not Conflict, and Abstain decisions
- Evidence citation, rationale, retry, and fail-closed behavior
- [Trajectory-level analysis](../drift/trajectory_eda.md) used to define the two-Conflict Drift rule
- Two consecutive Conflicts for one Core Value as Drift
- Independent Core Value sequences
- Cross-week detection and longer Conflict runs
- Recovery, uncertainty, and deduplication rules
- Active, recovered, uncertain, and mixed delivery states
- Detailed design in [Drift Detection Architecture](../architecture/drift_detection.md)

### 3.4 Coach Digest and Demo

- Weekly Drift Detection output as the structured input to the Coach Digest
- Core Values, Drift state, cited evidence, and date-window metadata
- Coach Digest response and one reflective question
- Groundedness, jargon, and length checks
- Complete Weekly Drift Detection integration for the capstone POC
- Experimental Coach Digest evaluation
- Supporting Onboarding and Conversational Nudge POCs
- Shared Experience and Inspect React session
- Five deterministic persona replays
- Live-provider and offline-fixture behavior
- Public Railway assessment deployment with anonymous access and server-side provider credentials
- Current design in [Experience and Inspect Specification](../demo/experience_inspect_app.md)

## 4. VIF Critic Experiments and Architecture Decision

### 4.1 Why Twinkl Tested a Small Model First

- Deliberate test of a simple and low-cost model before a runtime LLM
- 23,454-parameter MLP head
- Frozen text embedding and normalized ten-value Profile
- Ten ternary ordinal prediction heads
- Low marginal inference cost after text encoding
- Fast per-Journal-Entry prediction
- Local or offline inference option
- Reduced provider dependence and Journal Entry exposure
- Repeatable output and direct uncertainty measurement
- LLM-Judge VIF Labels used to train a cheaper model
- Accuracy, latency, cost, and privacy evidence required before replacement
- Initial design in [VIF Concepts and Roadmap](../vif/01_concepts_and_roadmap.md)

### 4.2 Experiment Evolution

- Table columns: Run range, intervention, outcome, and decision
- Runs 1-18: baseline design, hyperparameters, and split correction
- Runs 19-36: class imbalance, targeted data, regularization, and weighting
- Runs 37-56: encoder and model reformulation, consensus labels, and checkpoint diagnostics
- Runs 57-69: Security target repair, soft labels, and compact history
- Pivotal moment: persona-level split correction
- Pivotal moment: BalancedSoftmax reduction of neutral hedging
- Pivotal moment: active-state Security target repair
- Recall-first checkpoint policy and `run_060` offline nomination
- Complete chronology in the [VIF Experiment Index](../../logs/experiments/index.md)

### 4.3 Main Technical Advances

- Persona-level split correction
- Preservation of rare `-1` and `+1` labels across data splits
- Replacement of misleading pre-split model rankings
- BalancedSoftmax reduction of neutral hedging
- Targeted Power Conflict-recall improvement
- Weighted-loss Conflict-recall improvement
- Persona-cluster bootstrap check
- Qwen embedding comparison
- Active-state Security target repair
- Shortcut audit without support for the tested single-word explanation
- Recall-first checkpoint selection with a QWK safeguard
- Reproducible run logs, raw predictions, and selection traces
- Diagnostic tooling: interactive 3D Embedding Explorer for prediction-error and uncertainty inspection
- Metric synthesis in the [July 2026 Strategy Review](../../logs/experiments/reports/experiment_review_2026-07-02_strategy.md)

### 4.4 Challenges, Negative Results, and LLM Comparison

- Rare Conflict labels and strong neutral-class imbalance
- Neutral hedging and random-seed sensitivity
- Hedonism, Security, Stimulation, and Power performance gaps
- Errors on quiet pleasure, defended rest, and stability language
- LLM-Judge VIF Label ambiguity and target reachability
- Current-Journal-Entry context limit
- Compact-history and soft-target failures
- Improved Conflict recall with weak Conflict precision
- Wrong adjacent Conflict pairs after Drift aggregation
- Diminishing returns from later interventions
- Same-split comparison with a human-context LLM
- MLP advantage in Conflict recall
- LLM advantage in ordinal agreement and minority recall
- Context, representation, and label-ceiling limits
- Comparison in the [Context-Gap Report](../../logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md)

### 4.5 Evidence-Led Architecture Decision

- VIF Critic research retained as a completed technical contribution
- Explicit end point for further MLP intervention work
- No reliable Drift-recall gain from raw VIF Critic input
- No Drift-recall gain from VIF-Critic-triggered review
- Additional Weekly Drift Reviewer calls and false Drift alerts
- In-sample limits and no fresh final test
- Weekly Drift Reviewer selected after bounded MLP and LLM comparisons
- Deterministic Drift Detector after Weekly Drift Reviewer Decisions
- VIF Critic Predictions excluded from the user-facing Drift path
- Qualitative MLP advantages in local execution and provider independence
- Cost, latency, provider-use, and data-exposure trade-offs
- Development-study token cost and latency measurements
- No public-demo, per-user, or production cost benchmark
- Conditions for a future cheaper model
- Final boundary in the [VIF Capstone Decision](../vif/05_capstone_scope_decision.md)

## 5. Evaluation and Results

### 5.1 Evaluation Design and Label Validation

- Development set separated from any future final test
- AI-reviewed evidence separated from human validation
- Primary metrics separated from diagnostic metrics
- Persona-level data splits, repeated Runs, and fixed seeds
- Versioned configurations and receipts
- AI-reviewed student-visible Drift target audit and repair
- Fleiss' kappa across human annotators
- Mean human agreement with LLM-Judge VIF Labels
- Per-value disagreement and ambiguous examples
- Security input-contract repair and consensus diagnostics
- Human-sample limits
- Design in [Evaluation Overview](../evals/overview.md)

### 5.2 VIF Critic Results

- Corrected-split median `recall_-1` increase from `0.104` to `0.313`
- Targeted Power `recall_-1` increase from `0.125` to `0.313`
- Weighted branch median `recall_-1` of `0.378`
- Active-state Security repair with about `+0.17` median test QWK
- Same-split MLP QWK of `0.378` and `recall_-1` of `0.342`
- Human-context `gpt-5.4-mini` QWK of `0.450` and `recall_-1` of `0.302`
- Aggregate and per-value precision-recall trade-offs
- Calibration and seed spread
- Failed and inconclusive experiment families
- `run_060` offline checkpoint nomination
- Offline-only conclusion
- Detailed metrics in [Value Modeling Evaluation](../evals/value_modeling_eval.md)

### 5.3 Drift Detection Results

- 292 resolved development cases
- 42 known Drifts across 36 Drift trajectories
- Comparison of three Weekly Drift Reviewer model and reasoning-effort setups
- Three complete Runs per experiment setup
- Drift recall as the first selection metric
- False Drift alerts as the second selection metric
- Coverage and Abstain rate as diagnostic metrics
- Luna-low median Drift recall of `0.548`
- Luna-low median of four false Drift alerts
- Luna-low median coverage of `0.637`
- Luna-low median request latency of `2.81` seconds
- Cache-aware token cost calculation of `$6.96` for 2,853 development calls, not a billing export
- Cost and latency provenance in the [Luna Reasoning-Effort Review](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md)
- Missed Drift and false Drift alert categories
- Fixed Luna-low Weekly Drift Reviewer contract
- Development-only conclusion
- Reproducibility tooling: read-only [Drift Inspection App](../demo/weekly_drift_review_app.md) for development and persona-level result review without provider calls
- Full results in the [Complete Development Review](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md)

### 5.4 Coach Digest and Demo Results

- Weekly Drift Detection output schema and cited Journal Entry evidence
- Active, recovered, uncertain, and mixed examples
- Coach Digest Validations for groundedness, jargon, and length
- No current Coach Digest Validations or Coach Digest Evals results
- No Coach Digest usefulness claim without batch results and human validation
- Brief Experience and Inspect example of end-to-end POC behavior
- Evaluation boundary in [Explanation Quality Evaluation](../evals/explanation_quality_eval.md)

### 5.5 Live Prompt Boundary Verification

- Prompt injection risk from instruction-like text in user-controlled fields
- OpenAI `instructions` and Gemini `system_instruction` fields for stable
  Twinkl rules
- Separate JSON data for live Nudge, Weekly Drift Reviewer, and Coach Digest
  calls
- Direct rule that JSON text is evidence and is not an instruction
- Boundary-like text preserved as JSON rather than parsed as a prompt boundary
- Exact two-message receipt for Inspect
- Receipt hashes and fail-closed provenance checks for live Nudge and Weekly
  Drift Reviewer calls
- Structural tests for Journal Entries, nudge responses, preferred names,
  current focus text, schemas, evidence validation, retry, and fail-closed paths
- 166 relevant tests passed on 2026-08-03
- No claim that structural verification proves provider model resistance to
  prompt injection
- Detailed evidence in [Live Prompt Boundary Verification](../evals/live_prompt_boundary_verification.md)

## 6. Discussion

### 6.1 Main Findings and Trade-offs

- Feasibility of longitudinal value-alignment analysis
- Value of explicit Core Values
- Value of testing a small and cheap model first
- Contribution from successful and negative VIF Critic experiments
- Separation of Weekly Drift Reviewer Decisions and deterministic Drift logic
- VIF Critic research value without user-facing authority
- Current-Journal-Entry MLP input versus longitudinal LLM context
- Conflict recall versus false Drift alerts
- Abstain behavior versus coverage
- LLM cost and latency versus local inference
- Explainability benefit of cited Journal Entries

### 6.2 Failure Cases and Validity Limits

- Ambiguous or low-detail Journal Entries
- Core Value interpretation differences
- Human disagreement with LLM-Judge VIF Labels
- VIF Critic hard dimensions and weak Conflict precision
- Incorrect adjacent Conflict pairs
- Weekly Drift Reviewer missed Drifts and false Drift alerts
- Coach Digest generic or weakly grounded text
- No live provider measurement of prompt injection attack success
- Synthetic development data
- Historical training provenance in part of the Drift development set
- Limited human annotation and no completed user study
- No fresh final test or deployment approval; the public Railway deployment is for assessment only

### 6.3 Safety, Privacy, and Ethics

- Sensitive Journal Entry data
- Consent, retention, access, export, and deletion
- Non-therapy boundary
- Non-prescriptive and non-judgmental Coach Digest language
- Fail-closed Abstain behavior
- Higher-priority Twinkl instructions separated from user-controlled JSON data
- The live message boundary reduces but does not eliminate prompt injection
- Synthetic bias and demographic stereotype risks
- Provider data exposure
- No automatic use of real Journal Entries for training

## 7. Conclusion and Future Work

### 7.1 Conclusion

- Problem and objective restatement
- Small-model-first research approach
- Main VIF Critic findings and limits
- Evidence for the Weekly Drift Reviewer architecture
- Deterministic Drift Detector contribution
- Weekly Drift Detection and Coach Digest scope
- Main evaluation results and capstone POC limits

### 7.2 Focused Future Work

- Fresh final test excluded from model and prompt development
- Coach Digest batch evaluation and human validation
- Real-user pilot with explicit consent and privacy controls
- Final runtime cost, latency, and provider-use measurement
- Optional live provider prompt injection evaluation with a fixed list of
  attack text
- Conditions for reconsideration of a cheaper local model

## References

- Schwartz value theory
- Best-Worst Scaling and SVBWS
- Synthetic data and LLM evaluation
- Ordinal classification, uncertainty, and class imbalance
- Explainable AI and reflective technology
- Model, encoder, and provider documentation
- Citation-to-claim audit

## Appendix A. Reproduction and Evidence Map

### A.1 Core Specifications and Source Code

- [Repository README](../../README.md)
- [Product Requirements Document](../prd.md)
- [Canonical Nouns and Communication Rules](../canonical_nouns.md)
- [Synthetic Data Pipeline](../pipeline/pipeline_specs.md)
- [`src/synthetic/generation.py`](../../src/synthetic/generation.py)
- [`src/judge/labeling.py`](../../src/judge/labeling.py)
- [`src/weekly_drift_reviewer.py`](../../src/weekly_drift_reviewer.py)
- [`src/prompt_boundary.py`](../../src/prompt_boundary.py)
- [`src/drift_detector.py`](../../src/drift_detector.py)
- [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py)
- [`src/coach/llm_client.py`](../../src/coach/llm_client.py)
- [`src/demo/experience_service.py`](../../src/demo/experience_service.py)

### A.2 VIF Critic Experiments

- [VIF Experiment Index](../../logs/experiments/index.md)
- [VIF Training](../vif/03_model_training.md)
- [VIF Uncertainty](../vif/04_uncertainty_logic.md)
- [`src/vif/critic_ordinal.py`](../../src/vif/critic_ordinal.py)
- [`src/vif/train.py`](../../src/vif/train.py)
- [`src/vif/extract_embeddings.py`](../../src/vif/extract_embeddings.py)
- [`scripts/experiments/critic_training_v4_review.py`](../../scripts/experiments/critic_training_v4_review.py)
- [`scripts/experiments/llm_critic_baseline.py`](../../scripts/experiments/llm_critic_baseline.py)
- [`scripts/experiments/reassess_twinkl_752_5.py`](../../scripts/experiments/reassess_twinkl_752_5.py)
- [BalancedSoftmax Review](../../logs/experiments/reports/experiment_review_2026-03-07_v5.md)
- [Weighted Branch Review](../../logs/experiments/reports/experiment_review_2026-03-11_twinkl_719_3.md)
- [Security Target Review](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_a30f_security_target.md)
- [Recall-First Checkpoint Review](../../logs/experiments/reports/experiment_review_2026-07-19_twinkl_6mrt_recall_first_checkpoint_selection.md)
- [Weekly Drift Reviewer Input Ablation](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_752_1_weekly_verifier_ablation.md)
- [Raw-Input and Scheduling Reassessment](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md)

### A.3 Evaluation and Supplementary Results

- [Evaluation Overview](../evals/overview.md)
- [Judge Validation Evaluation](../evals/judge_validation_eval.md)
- [Value Modeling Evaluation](../evals/value_modeling_eval.md)
- [Drift Detection Evaluation](../evals/drift_detection_eval.md)
- [Explanation Quality Evaluation](../evals/explanation_quality_eval.md)
- [Live Prompt Boundary Verification](../evals/live_prompt_boundary_verification.md)
- [Agreement Report](../../logs/exports/agreement_report_20260318_130642.md)
- [Luna Model Comparison](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_model_comparison.md)
- [Luna Reasoning-Effort Review](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md)
- [`scripts/experiments/compare_twinkl_52zz_models.py`](../../scripts/experiments/compare_twinkl_52zz_models.py)
- [`tests/test_drift_detector.py`](../../tests/test_drift_detector.py)
- [`tests/coach/test_weekly_digest.py`](../../tests/coach/test_weekly_digest.py)
- [`tests/coach/test_runtime.py`](../../tests/coach/test_runtime.py)
- [`tests/nudge/test_runtime.py`](../../tests/nudge/test_runtime.py)
- [`tests/test_weekly_drift_reviewer.py`](../../tests/test_weekly_drift_reviewer.py)
- [`tests/demo/test_experience_service.py`](../../tests/demo/test_experience_service.py)
