# Twinkl Capstone Project Report

- **Working title:** Twinkl: An Inner Compass for Alignment Between Behavior and Core Values
- Author names and student numbers
- NUS programme, module, and submission date
- Phase 2 Technical Paper requirement: [IS Capstone Requirements](../is_capstone_slides.pdf)

## Draft Controls

- **Document status:** Capstone Project Report scaffold
- **NUS deliverable:** Phase 2 Technical Paper formatted as a publishable paper
- **Product source:** [Product Requirements Document](../prd.md)
- **Required terms:** [Canonical Nouns and Communication Rules](../canonical_nouns.md)
- **Prior submission:** [April 2026 Project Proposal](../archive/capstone/2026-04-proposal-submission/April_Project_Proposal.md)
- **Status date:** 2026-07-31
- **Status key:** ✅ complete for the capstone POC; 🟡 partial or development-only; 🧪 experimental; 🚧 in progress; ⏸ outside the time-boxed capstone
- AI-reviewed synthetic evidence identified as AI-reviewed synthetic evidence
- Human annotation identified as human annotation
- No fresh final test or deployment approval claim
- Clear separation between the user-facing Drift path and offline VIF Critic research

## Abstract

- Problem and target users
- Twinkl objective
- Longitudinal synthetic data and LLM-Judge VIF Label workflow
- Cost, latency, and privacy motivation for testing a small MLP first
- VIF Critic experiment progression and main technical findings
- Offline VIF Critic research contribution
- Evidence-led change from the VIF Critic to the Weekly Drift Reviewer for Drift
- Weekly Drift Reviewer and deterministic Drift Detector contribution
- Main quantitative results
- Weekly Digest and Weekly Coach scope
- Main limitations and capstone POC boundary

## Keywords

- Core Values
- Journal Entries
- Value alignment
- Drift detection
- LLM evaluation
- Synthetic data
- Explainable AI

## 1. Introduction

### 1.1 Problem

- Gap between declared Core Values and daily behavior
- Limits of mood summaries and isolated Journal Entry analysis
- Need for evidence-based weekly reflection
- Target users and capstone use case
- Product intent from the [Product Requirements Document](../prd.md)

### 1.2 Project Objective

- Longitudinal comparison of Journal Entries with declared Core Values
- Explainable identification of Conflict and Drift
- Evidence-grounded Weekly Digest and Weekly Coach question
- Time-boxed academic POC scope

### 1.3 Research Questions

- Human agreement with LLM-Judge VIF Labels
- Performance and operating-cost limits of a small MLP
- VIF Critic recovery of Conflict in offline research
- Evidence required before using a larger LLM in the user-facing path
- Weekly Drift Reviewer and Drift Detector recovery of known Drifts
- False Drift alerts, coverage, and Abstain behavior
- Weekly Digest evidence grounding
- Weekly Coach accuracy and usefulness

### 1.4 Contributions

- Longitudinal synthetic Journal Entry data workflow
- Provenance-aware LLM-Judge VIF Label workflow
- Human annotation tool and agreement benchmark
- 69-Run, 133-configuration VIF Critic experiment record
- Completed offline VIF Critic training and evaluation stack
- Evidence-led architecture decision after the VIF Critic intervention wave
- Fixed Weekly Drift Reviewer contract
- Deterministic two-Conflict Drift Detector
- Versioned Weekly Digest contract and cited evidence
- Experience and Inspect capstone demo

## 2. Related Work

### 2.1 Human Values and Value Elicitation

- Schwartz theory of basic human values
- Ten-value ontology used by Twinkl
- Best-Worst Scaling and SVBWS value elicitation
- Core Value selection and profile transformation
- Research basis in the [Onboarding Specification](../onboarding/onboarding_spec.md)

### 2.2 AI Journaling and Reflective Tools

- AI journal summaries and mood analysis
- Declared-value models versus post-hoc theme discovery
- Long-horizon reflection and accountability
- Product comparison source in the [April 2026 Project Proposal](../archive/capstone/2026-04-proposal-submission/April_Project_Proposal.md)

### 2.3 Synthetic Data and LLM-Judge Labels

- Need for longitudinal Journal Entry data
- Persona-controlled generation
- LLM-Judge VIF Labels as training data
- LLM-Judge Conflict Labels as Drift reference data
- Label uncertainty, leakage, and validation risks
- Workflow design in the [Synthetic Data Pipeline](../pipeline/pipeline_specs.md)

### 2.4 Value Modeling and Drift Detection

- Ternary value alignment prediction
- Ordinal classification and uncertainty
- Consecutive Conflict as Drift
- Weekly Drift Reviewer Decisions combined with a deterministic rule
- Architecture boundary in [Drift Detection](../architecture/drift_detection.md)

### 2.5 Explainable Weekly Reflection

- Evidence selection from Journal Entries
- Grounded reflection versus generic summary
- Non-prescriptive Weekly Coach tone
- Explanation evaluation tiers in [Explanation Quality Evaluation](../evals/explanation_quality_eval.md)

## 3. Twinkl Design

### 3.1 Product Scope

- Onboarding Profile and Core Values
- Journal Entry capture and optional nudge response
- Closed-week review cadence
- Weekly Drift Reviewer Decisions
- Drift Detector state
- Weekly Digest
- Weekly Coach reflection and question
- POC boundary and excluded production claims

### 3.2 User-Facing Path

- Confirmed onboarding Profile
- Journal Entries grouped by Monday-to-Sunday week
- Weekly Drift Reviewer without VIF Critic input
- Drift Detector across week boundaries
- Active, recovered, uncertain, and mixed delivery states
- Weekly Digest with cited Journal Entries
- Optional Weekly Coach reflection
- Current architecture in [End-to-End Architecture](../architecture/e2e_architecture.md)

### 3.3 Offline Research Path

- Synthetic persona generation
- LLM-Judge VIF Label creation
- Human annotation benchmark
- VIF Critic training and evaluation
- Experiment archive and checkpoint selection
- No user-facing VIF Critic authority
- Adopted scope in the [VIF Capstone Decision](../vif/05_capstone_scope_decision.md)

### 3.4 Data and Decision Boundaries

- Production-like inputs versus generation metadata
- Core Values eligible for Drift
- Weekly Drift Reviewer input cutoff
- No future-Journal-Entry leakage
- No LLM-Judge VIF Label summaries or VIF Critic Predictions in the approved Weekly Digest
- Versioned receipts, schemas, prompts, and request hashes

## 4. Methodology

### 4.1 Synthetic Data Workflow — ✅ Complete

- 204 personas and 1,651 Journal Entries
- Demographic and Core Value variation
- Parallel generation between personas
- Sequential generation within each persona
- Journal Entry, nudge, and response structure
- Banned-term and value-leakage controls
- YAML prompt templates and Jinja2 rendering
- [Pipeline Specification](../pipeline/pipeline_specs.md)
- [`src/synthetic/generation.py`](../../src/synthetic/generation.py)
- [`src/synthetic/batch_preparation.py`](../../src/synthetic/batch_preparation.py)
- [`src/synthetic/batch_verification.py`](../../src/synthetic/batch_verification.py)
- [`prompts/persona_generation.yaml`](../../prompts/persona_generation.yaml)

### 4.2 Wrangling and Registry — ✅ Complete

- Generation metadata removal
- Production-like Journal Entry representation
- Persona and stage registry
- Input validation and warnings
- [`src/wrangling/parse_synthetic_data.py`](../../src/wrangling/parse_synthetic_data.py)
- [`src/wrangling/parse_wrangled_data.py`](../../src/wrangling/parse_wrangled_data.py)
- [`src/registry/personas.py`](../../src/registry/personas.py)

### 4.3 LLM-Judge VIF Label Creation — ✅ Complete

- LLM-Judge VIF Label definition
- Persona context and previous Journal Entries
- Ten value dimensions and ternary labels
- Rationale generation and storage
- Label consolidation and schema checks
- Security target repair and immutable historical labels
- Consensus LLM-Judge VIF Labels as diagnostic evidence only
- [Historical Labeling Instructions](../pipeline/claude_judge_instructions.md)
- [Consensus Rejudging Instructions](../pipeline/consensus_rejudging_instructions.md)
- [`src/judge/labeling.py`](../../src/judge/labeling.py)
- [`src/judge/consolidate.py`](../../src/judge/consolidate.py)
- [`src/judge/consensus_utils.py`](../../src/judge/consensus_utils.py)

### 4.4 Human Annotation — ✅ Tool Complete; 🟡 Benchmark Scope Limited

- Annotation task and scoring guide
- Three annotators
- 380 saved annotations across 24 personas
- Shared 115-Journal-Entry benchmark across 19 personas
- Cohen's kappa and Fleiss' kappa
- Human disagreement and ambiguous cases
- [Annotation Tool Plan](../pipeline/annotation_tool_plan.md)
- [Judge Validation Evaluation](../evals/judge_validation_eval.md)
- [`src/annotation_tool/app.py`](../../src/annotation_tool/app.py)
- [`src/annotation_tool/agreement_metrics.py`](../../src/annotation_tool/agreement_metrics.py)

### 4.5 VIF Critic — ✅ Complete Offline Research

- Current-Journal-Entry and normalized value-profile input
- Ten ternary value predictions
- Ordinal MLP heads and BNN baseline
- Frozen text encoders
- MC Dropout uncertainty
- Corrected persona-level splits
- Class imbalance methods
- Recall-first checkpoint selection
- Nominated offline checkpoint and experiment provenance
- No user-facing Drift role
- [VIF Concepts](../vif/01_concepts_and_roadmap.md)
- [VIF Training](../vif/03_model_training.md)
- [VIF Uncertainty](../vif/04_uncertainty_logic.md)
- [`src/vif/critic_ordinal.py`](../../src/vif/critic_ordinal.py)
- [`src/vif/critic_bnn.py`](../../src/vif/critic_bnn.py)
- [`src/vif/train.py`](../../src/vif/train.py)
- [`scripts/experiments/critic_training_v4_review.py`](../../scripts/experiments/critic_training_v4_review.py)

### 4.6 Weekly Drift Reviewer — ✅ POC Complete; 🟡 Development Evidence Only

- Fixed `gpt-5.6-luna` model contract
- Fixed reasoning effort `low`
- Cumulative student-visible Journal Entry history
- Core Values and current-week coordinates
- Conflict, Not Conflict, and Abstain decisions
- Evidence citation and decision rationale
- Retry and fail-closed behavior
- Versioned request and response receipts
- No VIF Critic input
- [`src/weekly_drift_reviewer.py`](../../src/weekly_drift_reviewer.py)
- [`prompts/weekly_drift_reviewer_aligned.yaml`](../../prompts/weekly_drift_reviewer_aligned.yaml)
- [`config/evals/twinkl_52zz_luna_low_v1.yaml`](../../config/evals/twinkl_52zz_luna_low_v1.yaml)

### 4.7 Drift Detector — ✅ Complete for Capstone POC

- Two consecutive Conflicts for the same Core Value
- Independent Core Value sequences
- Cross-week detection
- Longer Conflict-run extension
- Recovery handling
- Abstain and uncertainty handling
- Deduplication and deterministic Drift Detector result
- Active, recovered, uncertain, and mixed delivery
- Development-only evaluation and no deployment approval
- [Drift Detection Architecture](../architecture/drift_detection.md)
- [Drift Detection Evaluation](../evals/drift_detection_eval.md)
- [`src/drift_detector.py`](../../src/drift_detector.py)
- [`src/drift_rules.py`](../../src/drift_rules.py)
- [`tests/test_drift_detector.py`](../../tests/test_drift_detector.py)

### 4.8 Weekly Digest — 🟡 Implemented Contract; Partial Product Integration

- Structured bridge from Drift Detector to Weekly Coach
- Core Values and per-value Drift state
- Cited Journal Entry evidence
- Decision rationale and date-window metadata
- Weekly Digest JSON, Markdown, prompt, and parquet files
- Optional Weekly Coach narrative and validation results
- Approved path versus deprecated compatibility paths
- No VIF Critic Predictions or LLM-Judge VIF Label summaries in the approved path
- [Weekly Digest Generation](../weekly/weekly_digest_generation.md)
- [`src/coach/schemas.py`](../../src/coach/schemas.py)
- [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py)
- [`src/coach/weekly_drift_runtime.py`](../../src/coach/weekly_drift_runtime.py)

### 4.9 Weekly Coach — 🧪 Experimental

- Weekly Digest input
- Evidence-grounded reflection
- One reflective question
- Active, recovered, uncertain, and mixed tone
- Provider-backed generation adapters
- Programmatic groundedness, class-code jargon, and length checks
- Implemented prompt rendering and optional narrative generation
- Missing committed batch pass-rate report
- Missing Tier 2 rationale-review LLM evaluation
- Missing Tier 3 human calibration
- Incomplete product-facing orchestration
- [Explanation Quality Evaluation](../evals/explanation_quality_eval.md)
- [`prompts/weekly_digest_coach.yaml`](../../prompts/weekly_digest_coach.yaml)
- [`src/coach/llm_client.py`](../../src/coach/llm_client.py)
- [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py)

### 4.10 Onboarding — 🧪 Experimental

- Published 11-group, six-object SVBWS design
- Group and card randomization
- Raw BWS result storage
- Ten-value Profile transformation
- User confirmation and Profile Core Values
- Runtime import of a confirmed Profile
- Standalone React POC
- No validated Twinkl-specific instrument claim
- No automated browser-to-service storage in capstone scope
- [Onboarding Specification](../onboarding/onboarding_spec.md)
- [`frontend/onboarding/src/App.tsx`](../../frontend/onboarding/src/App.tsx)
- [`src/demo/profile_projection.py`](../../src/demo/profile_projection.py)

### 4.11 Conversational Nudge — 🧪 Experimental

- Observable content signals
- Decision, generation, reply, skip, retry, and safe failure
- Journal Entry and nudge-response linkage
- Implemented runtime and Experience integration
- Missing evidence that nudging improves VIF Critic signal quality
- [Nudge Design Rationale](../pipeline/nudge_design_rationale.md)
- [`src/nudge/decision.py`](../../src/nudge/decision.py)
- [`src/nudge/generation.py`](../../src/nudge/generation.py)
- [`src/nudge/runtime.py`](../../src/nudge/runtime.py)

## 5. VIF Critic Experiment Program and Architecture Decision — ✅ Complete Capstone Research

- Deliberate test of a simple and low-cost model before a runtime LLM
- Full experiment chronology in the [VIF Experiment Index](../../logs/experiments/index.md)
- Metric and scope synthesis in the [July 2026 Strategy Review](../../logs/experiments/reports/experiment_review_2026-07-02_strategy.md)
- Final boundary in the [VIF Capstone Decision](../vif/05_capstone_scope_decision.md)

### 5.1 Starting Hypothesis: Use the Smallest Model That Works

- Avoidance of an automatic jump to the latest and largest LLM
- 23,454-parameter MLP head over frozen text embeddings and a ten-value profile
- Low marginal inference cost after text encoding
- Fast per-Journal-Entry prediction
- Local or offline inference option
- Reduced provider dependence and data exposure
- Repeatable output and direct uncertainty measurement
- Offline LLM-Judge VIF Labels used to distill a cheaper runtime model
- Required accuracy, latency, cost, and privacy evidence before model replacement
- Beads design question from `twinkl-w2mu`: distillation cost versus runtime value

### 5.2 Initial VIF Critic Design

- Frozen `nomic-embed-text-v1.5` representation
- 256-dimensional Journal Entry embedding
- Normalized ten-value profile
- 266-dimensional `window_size: 1` state
- Ten ternary ordinal prediction heads
- MLP hidden layer and dropout
- MC Dropout uncertainty
- Current Journal Entry without prior Journal Entry history
- Parameter count, model size, memory use, and supported hardware
- [`src/vif/state_encoder.py`](../../src/vif/state_encoder.py)
- [`src/vif/critic_ordinal.py`](../../src/vif/critic_ordinal.py)
- [`config/vif.yaml`](../../config/vif.yaml)

### 5.3 Experiment Evolution

- Runs 1-15: encoder, context window, hidden size, learning rate, and ordinal-loss exploration
- Runs 16-18: corrected persona-stratified split and three-seed rebaseline
- Runs 19-21: BalancedSoftmax long-tail breakthrough
- Runs 22-27: targeted Power, Security, and Hedonism data interventions
- Runs 28-36: circumplex regularization, checkpoint guardrails, and dimension weighting
- Runs 37-47: embedding-width, encoder-swap, SLACE, and two-stage reformulation tests
- Runs 48-56: consensus-label and recall-aware checkpoint diagnostics
- Runs 57-62: active-state Security target repair
- Runs 63-68: hard-target versus soft-vote training
- Run 69: compact-history experiment
- Recall-first checkpoint policy and `run_060` offline nomination
- Fixed seeds, configs, checkpoints, and experiment reports
- [`scripts/experiments/critic_training_v4_review.py`](../../scripts/experiments/critic_training_v4_review.py)
- [VIF Experiment Index](../../logs/experiments/index.md)

### 5.4 Main Technical Advances and Findings

- Split correction that preserved rare `-1` and `+1` signals across personas
- Replacement of misleading pre-split model rankings
- BalancedSoftmax reduction of neutral hedging
- Corrected-split median `recall_-1` increase from `0.104` to `0.313`
- Targeted Power `recall_-1` increase from `0.125` to `0.313`
- Weighted branch median `recall_-1` of `0.378`
- Persona-cluster bootstrap confirmation of the weighted recall gain
- Qwen embedding branch as a credible but non-winning encoder challenger
- Active-state Security target repair with about `+0.17` median test QWK
- Shortcut audit that did not support the tested single-word explanation
- Recall-first checkpoint selection with an explicit QWK safeguard
- `run_060` nomination for the offline VIF Critic path
- Reproducible experiment logging, selection traces, raw predictions, and provenance
- [BalancedSoftmax Review](../../logs/experiments/reports/experiment_review_2026-03-07_v5.md)
- [Weighted Branch Review](../../logs/experiments/reports/experiment_review_2026-03-11_twinkl_719_3.md)
- [Uncertainty Review](../../logs/experiments/reports/experiment_review_2026-03-14_twinkl_730.md)
- [Security Target Review](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_a30f_security_target.md)
- [Recall-First Checkpoint Review](../../logs/experiments/reports/experiment_review_2026-07-19_twinkl_6mrt_recall_first_checkpoint_selection.md)

### 5.5 Main Challenges and Negative Results

- Rare Conflict labels and strong neutral-class imbalance
- Neutral hedging across conservative ordinal losses
- High sensitivity to persona split and random seed
- Hedonism, Security, Stimulation, and Power performance gaps
- Semantic-polarity errors on quiet pleasure, defended rest, and stability language
- LLM-Judge VIF Label ambiguity and target reachability
- Security input-contract mismatch before target repair
- Current-Journal-Entry context limit
- Compact-history failure to transfer the LLM context gain
- Soft-target objective and runtime-decoding mismatch
- QWK pathologies under class imbalance
- Entry-level Conflict recovery that did not form the correct consecutive Drifts
- Improved recall with weak Conflict precision
- Diminishing returns from loss, encoder, weighting, and post-hoc interventions
- Synthetic-data and human-agreement limits
- [Hedonism Hard-Set Review](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_748_hedonism_hard_set.md)
- [Soft-Target Review](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_j0ck_soft_vote_labels.md)
- [Compact-History Review](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_749_compact_history.md)
- [Shortcut Audit](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_1r3d_shortcut_audit.md)

### 5.6 Bounding the MLP Against an LLM

- Same frozen 221-Journal-Entry test split
- Same student-visible input contract
- `run_020` MLP QWK of `0.378` and `recall_-1` of `0.342`
- Human-context `gpt-5.4-mini` QWK of `0.450` and `recall_-1` of `0.302`
- MLP advantage on Conflict recall
- LLM advantage on ordinal agreement and minority recall
- Complementary MLP-only and LLM-only Conflict recovery
- Quantified benefit from prior Journal Entry context
- Separation of label ceiling, representation limit, and context limit
- [`scripts/experiments/llm_critic_baseline.py`](../../scripts/experiments/llm_critic_baseline.py)
- [Frozen Context-Gap Report](../../logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md)

### 5.7 Why the VIF Critic Did Not Receive User-Facing Drift Authority

- Entry-level VIF Critic Prediction versus two-Conflict Drift decision
- Low Conflict precision despite useful Conflict recovery
- Wrong adjacent Conflict pairs after aggregation
- No reliable Drift recall gain from raw VIF Critic input
- No Drift recall gain from VIF-Critic-triggered early-plus-weekly review
- Additional Weekly Drift Reviewer calls and false Drift alerts from scheduling
- In-sample limits for development cases with training provenance
- No fresh final test or deployment approval
- Useful offline role without user-facing decision authority
- [`scripts/experiments/reassess_twinkl_752_5.py`](../../scripts/experiments/reassess_twinkl_752_5.py)
- [Weekly Drift Reviewer Input Ablation](../../logs/experiments/reports/experiment_review_2026-07-12_twinkl_752_1_weekly_verifier_ablation.md)
- [Raw-Input and Scheduling Reassessment](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md)

### 5.8 Evidence-Led Architecture Change

- No framing of the VIF Critic research as discarded work
- VIF Critic as proof that the cheaper option received serious evaluation
- Explicit end point for further MLP intervention work
- Weekly Drift Reviewer selected only after bounded MLP and LLM comparisons
- Cumulative student-visible Journal Entry context
- Fixed Luna-low Weekly Drift Reviewer contract
- Deterministic Drift Detector after Weekly Drift Reviewer Decisions
- VIF Critic retained as complete offline research
- VIF Critic Predictions excluded from the user-facing Drift path
- Technical contribution from successful and negative experiments
- [Drift Detection Architecture](../architecture/drift_detection.md)
- [Luna Model Comparison](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_model_comparison.md)
- [Luna Reasoning-Effort Review](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md)
- [`scripts/experiments/compare_twinkl_52zz_models.py`](../../scripts/experiments/compare_twinkl_52zz_models.py)

### 5.9 Cost, Latency, and Privacy Comparison

- MLP head cost separated from frozen-encoder cost
- MLP parameter count and serialized checkpoint size
- MLP per-Journal-Entry latency on stated hardware
- Local embedding and prediction feasibility
- Weekly Drift Reviewer call frequency
- Weekly Drift Reviewer token use, latency, and provider cost
- Per-user weekly cost estimate
- Accuracy gained per additional runtime cost
- Network and provider dependency
- Journal Entry data exposure boundary
- Conditions that could justify a future cheaper model

## 6. Implementation

### 6.1 Runtime Architecture

- Python services and typed data contracts
- Weekly Drift Reviewer client
- Versioned decision receipts
- Deterministic Drift Detector
- Weekly Digest persistence
- Weekly Coach provider adapters
- Error handling and retry boundaries
- [`src/coach/weekly_drift_runtime.py`](../../src/coach/weekly_drift_runtime.py)
- [`src/demo/api.py`](../../src/demo/api.py)
- [`src/demo/experience_service.py`](../../src/demo/experience_service.py)

### 6.2 Experience and Inspect Demo — 🚧 In Progress

- Shared React session and contracts
- Five deterministic persona replays
- Manual Journal Entries and nudge actions
- Closed-week review cadence
- Live Weekly Drift Reviewer integration
- Drift and cited Weekly Digests
- Linked Inspect events and no-future-data projection
- Completed release quality gate
- Remaining professor walkthrough and capstone evidence
- [Experience and Inspect Specification](../demo/experience_inspect_app.md)
- [`frontend/onboarding/src/JournalExperience.tsx`](../../frontend/onboarding/src/JournalExperience.tsx)
- [`frontend/onboarding/src/InspectView.tsx`](../../frontend/onboarding/src/InspectView.tsx)
- [`frontend/onboarding/src/PersonaReplay.tsx`](../../frontend/onboarding/src/PersonaReplay.tsx)

### 6.3 Drift Inspection App — ✅ Complete

- Read-only comparison of three frozen Weekly Drift Reviewer setups
- Complete and persona-level development results
- Journal Entries and input cutoffs
- LLM-Judge Conflict Labels and Weekly Drift Reviewer Decisions
- Evidence citations and Drift metrics
- No provider API calls during inspection
- [Drift Inspection App Guide](../demo/weekly_drift_review_app.md)
- [`src/drift_review_app/app.py`](../../src/drift_review_app/app.py)
- [`src/drift_review_app/data.py`](../../src/drift_review_app/data.py)

### 6.4 Deprecated Compatibility Paths

- Former VIF Critic crash, rut, and evolution runtime
- Standalone Weekly Digest input from LLM-Judge VIF Labels or VIF Critic Predictions
- Historical reproduction purpose
- Exclusion from the approved user-facing architecture
- [Deprecated Demo Review App](../demo/review_app.md)
- [`src/coach/runtime.py`](../../src/coach/runtime.py)
- [`src/vif/drift.py`](../../src/vif/drift.py)

## 7. Evaluation

### 7.1 Evaluation Design

- Data preparation, VIF Critic training, Drift detection, and Weekly Coach stages
- Development set versus any future final test
- AI-reviewed evidence versus human validation
- Primary and diagnostic metrics
- Reproducible Runs, seeds, configs, and receipts
- [Evaluation Overview](../evals/overview.md)

### 7.2 LLM-Judge Validation — 🟡 Operational

- Shared human benchmark design
- Fleiss' kappa across human annotators
- Mean human agreement with LLM-Judge VIF Labels
- Hard-dimension disagreements
- Security target repair
- Consensus stability diagnostics
- [Agreement Report](../../logs/exports/agreement_report_20260318_130642.md)
- [Consensus Rejudging Report](../../logs/exports/twinkl_754/consensus_rejudging_report.md)

### 7.3 VIF Critic Evaluation — ✅ Complete for Capstone Research

- Entry-level `recall_-1` as primary model-development metric
- `-1` precision and precision-recall behavior
- QWK and `+1` recall as diagnostics
- Calibration and per-dimension results
- Seed spread and checkpoint selection
- Repaired-Security comparison
- Compact-history and hard-Hedonism limits
- [Value Modeling Evaluation](../evals/value_modeling_eval.md)
- [Experiment Index](../../logs/experiments/index.md)
- [Security Target Report](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_a30f_security_target.md)
- [Recall-First Checkpoint Report](../../logs/experiments/reports/experiment_review_2026-07-19_twinkl_6mrt_recall_first_checkpoint_selection.md)

### 7.4 Drift Detection Evaluation — 🟡 Development Only

- 292 resolved cases
- 42 known Drifts across 36 Drift trajectories
- Three complete Runs per experiment setup
- Drift recall as first selection metric
- False Drift alerts as second selection metric
- Coverage and Abstain rate as diagnostics
- Luna-low median Drift recall of `0.548`
- Luna-low median of 4 false Drift alerts
- Luna-low median coverage of `0.637`
- Development provenance and synthetic-data limits
- No fresh final test or deployment approval
- [Complete Development Review](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md)
- [Luna Reasoning-Effort Report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md)
- [`scripts/experiments/compare_twinkl_52zz_models.py`](../../scripts/experiments/compare_twinkl_52zz_models.py)

### 7.5 Weekly Digest and Weekly Coach Evaluation — 🟡 Partial

- Cited-evidence substring checks
- VIF Critic metric and class-code jargon exclusion
- Narrative length bounds
- Existing Tier 1 unit tests
- Missing batch pass-rate report
- Missing human accuracy and usefulness study
- Missing Tier 2 and Tier 3 results
- [Explanation Quality Evaluation](../evals/explanation_quality_eval.md)
- [`tests/coach/test_weekly_digest.py`](../../tests/coach/test_weekly_digest.py)

### 7.6 Demo Verification — 🚧 In Progress

- Contract tests
- Experience service tests
- Persona replay tests
- Narrow-screen behavior
- Reduced-motion behavior
- Safe failure and retry behavior
- Professor walkthrough evidence
- [`tests/demo/test_contracts.py`](../../tests/demo/test_contracts.py)
- [`tests/demo/test_experience_service.py`](../../tests/demo/test_experience_service.py)
- [`frontend/onboarding/src/JournalExperience.test.tsx`](../../frontend/onboarding/src/JournalExperience.test.tsx)
- [`frontend/onboarding/src/InspectView.test.tsx`](../../frontend/onboarding/src/InspectView.test.tsx)

## 8. Results

### 8.1 Data and Label Results

- Persona and Journal Entry counts
- LLM-Judge VIF Label distribution by value and class
- Rationale coverage
- Human annotation counts
- Human-human agreement and human agreement with LLM-Judge VIF Labels
- Security repair and consensus findings

### 8.2 VIF Critic Results

- Historical corrected-split reference Runs
- Repaired-Security paired result
- Nominated offline checkpoint
- Aggregate and per-value results
- Conflict recall and precision trade-off
- Failed or inconclusive experiment families
- Offline-only conclusion

### 8.3 Weekly Drift Reviewer and Drift Detector Results

- Weekly Drift Reviewer model and reasoning-effort comparison
- Drift recall, false Drift alerts, coverage, and Abstain
- Cross-week and longer-run behavior
- Error categories and missed Drifts
- Fixed Luna-low contract
- Development-only conclusion

### 8.4 Weekly Digest and Weekly Coach Results

- Example Weekly Digest structures
- Active, recovered, uncertain, and mixed examples
- Cited Journal Entry evidence
- Tier 1 validation examples
- Missing batch and human-study results
- Experimental conclusion

### 8.5 Demo Results

- Implemented Experience actions
- Implemented Inspect evidence
- Deterministic persona coverage
- Live-provider and offline-fixture behavior
- Remaining professor walkthrough evidence

## 9. Discussion

### 9.1 Main Findings

- Feasibility of longitudinal value-alignment analysis
- Value of explicit Core Values
- Separation of Weekly Drift Reviewer Decisions and deterministic Drift logic
- VIF Critic research value without user-facing authority
- Explainability benefits of cited Journal Entries

### 9.2 Technical Trade-offs

- Current-Journal-Entry VIF Critic input versus longitudinal LLM context
- Recall versus false Drift alerts
- Abstain behavior versus coverage
- LLM cost and latency versus local inference
- Synthetic control versus real-user validity
- Full-context prompting versus future retrieval

### 9.3 Failure Cases

- Ambiguous or low-detail Journal Entries
- Core Value interpretation differences
- Human disagreement with LLM-Judge VIF Labels
- VIF Critic hard dimensions
- Weekly Drift Reviewer false Drift alerts and missed Drifts
- Weekly Coach generic or weakly grounded text

### 9.4 Validity Limits

- Synthetic development data
- Historical training provenance in part of the Drift development set
- AI review versus human validation
- Limited human annotation sample
- No fresh final test
- No deployment approval
- No completed user study

### 9.5 Safety, Privacy, and Ethics

- Sensitive Journal Entry data
- Consent, retention, access, export, and deletion
- Non-therapy boundary
- Avoidance of prescriptive or judgmental Weekly Coach language
- Fail-closed Abstain behavior
- Risks from synthetic bias and demographic stereotypes
- No automatic use of real Journal Entries for training

### 9.6 Capstone Scope

- Complete contributions
- Partial contributions
- Experimental contributions
- Deprecated paths
- Evidence that supports each claim
- Claims explicitly withheld

## 10. Conclusion and Future Work

### 10.1 Conclusion

- Problem restatement
- Research and implementation contributions
- Main findings
- Capstone POC outcome
- Evidence limits

### 10.2 Near-Term Completion

- Professor walkthrough and demo evidence
- Weekly Coach Tier 1 batch report
- Final figures and tables
- Final reference audit
- Final reproduction commands

### 10.3 Future Work — ⏸ Outside the Time-Boxed Capstone

- Fresh final test and deployment evaluation
- Weekly Coach Tier 2 and Tier 3 evaluation
- Real-user pilot with explicit consent
- Automated Profile transfer from browser to service
- Long-history retrieval for the Weekly Coach
- Value Evolution research
- Journaling Anomaly Radar
- Goal-aligned inspiration feed
- Multimodal sensing
- [Value Evolution Concept](../evolution/01_value_evolution.md)
- [Future Work Notes](../future_work/)

## References

- Peer-reviewed sources from the April proposal
- Schwartz value theory sources
- Best-Worst Scaling and SVBWS sources
- Synthetic data and LLM evaluation sources
- Ordinal classification and uncertainty sources
- Explainable AI and reflective technology sources
- Dependency, Weekly Drift Reviewer, and Weekly Coach documentation
- Complete citation-to-claim audit

## Appendices

### Appendix A. Reproduction

- Environment and dependency versions
- Exact commands
- Config files
- Random seeds
- Input data versions
- Output paths
- Weekly Drift Reviewer and Weekly Coach provider contracts
- [Repository README](../../README.md)

### Appendix B. Data Schemas

- Persona schema
- Journal Entry schema
- LLM-Judge VIF Label schema
- VIF Critic input and prediction schema
- Weekly Drift Reviewer request, response, and receipt schema
- Drift Detector result schema
- Weekly Digest schema

### Appendix C. Prompts and Rubrics

- Persona generation prompt
- LLM-Judge VIF Label prompt
- LLM-Judge Conflict Label rubric
- Weekly Drift Reviewer prompt
- Conflict rubric
- Weekly Coach prompt
- Version and change history

### Appendix D. Additional Results

- Per-value VIF Critic metrics
- Precision-recall curves
- Calibration plots
- Drift result breakdowns
- Error examples
- Additional demo screenshots

### Appendix E. Team Contributions

- Work item by team member
- Design contributions
- Implementation contributions
- Evaluation contributions
- Report contributions
- Individual Accomplishment Report cross-reference
