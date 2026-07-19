# About This Project

Twinkl is an academic capstone project for the **NUS Master of Technology in Intelligent Systems (AI Systems)** program, with an expected duration of 6–9 months. The project spans multiple submodules including Intelligent Reasoning Systems, Pattern Recognition Systems, Intelligent Sensing Systems, and Architecting AI Systems. For additional context and presentation materials, see [is_capstone_slides.pdf](is_capstone_slides.pdf).

---

## Implementation Status

*Last updated: 2026-07-19*

| Feature | Status | Details |
|---------|--------|---------|
| **Synthetic Data Workflow** | ✅ Complete | 204 personas (1,651 Journal Entries) generated via Claude Code parallel subagents; YAML prompt templates with Jinja2; targeted value generation supports family-specific tension banks, frozen-holdout experiments, and judged acceptance gates for hard-dimension batches |
| **LLM-Judge Labeling** | ✅ Complete | 1,651 Journal Entries labeled across 204 personas; two-phase workflow (Python wrangling + parallel subagents); consolidated to `judge_labels.parquet` with rationales. A separate receipt-bound full-corpus Security review now provides a non-destructive `security_active_critic_state_v1` target; persisted labels remain immutable. |
| **VIF Critic Training** | 🧪 Experimental | Training stack complete with ordinal MLP heads, BNN baseline, configurable sentence encoders, uncertainty estimates, raw output export, experiment logging, and recall-first checkpoint selection. `run_019`-`run_021` remains the historical corrected-split reference, while repaired-Security `run_060` is nominated for optional offline research. Repaired Security supervision raises median test Security QWK by about 0.17 without regressing aggregate QWK. Compact-history and matched Hedonism diagnostics did not establish a stronger product role. This is AI diagnostic evidence, not human validation. The VIF Critic is not part of the user-facing Drift path and is not required for Twinkl to function; further review, retraining, and data-scaling work is optional P3 research. |
| **Human Annotation Tool** | ✅ Complete | ~4,200 LOC Shiny app; 380 saved annotations across 24 personas, with a 115-entry shared subset across 19 personas used for the current inter-rater agreement benchmark; Cohen's κ / Fleiss' κ metrics; modular components with analysis view; annotation ordering for persona prioritization |
| **Drift Inspection App** | ✅ Complete | Read-only desktop Shiny app for comparing Runs 1–3 across three frozen Weekly Drift Reviewer setups: `gpt-5.4-mini` at reasoning effort `none`, `gpt-5.6-luna` at reasoning effort `none`, and `gpt-5.6-luna` at reasoning effort `low`. It shows complete development and persona-level results, Journal Entries, AI-reviewed LLM-Judge Conflict Labels, Weekly Drift Reviewer Decisions, cited evidence, and verified input cutoffs without model or provider API calls. Local and Railway launch paths are documented in the [app guide](demo/weekly_drift_review_app.md). |
| **Conversational Nudging** | 🧪 Experimental | 3-category LLM classification (clarification/elaboration/tension-surfacing); pending validation that nudging improves VIF signal quality |
| **Drift Detector** | 🧪 Experimental | Drift is two consecutive Conflicts for the same Core Value. The implemented capstone POC path persists versioned Weekly Drift Reviewer Decisions without VIF Critic input and applies the deterministic Drift Detector across week boundaries. It handles extension, recovery, abstention, deduplication, and active, recovered, uncertain, or mixed delivery. The fixed Luna-low model contract retains its frozen AI-reviewed synthetic development evidence. The time-boxed capstone does not run a fresh final test or claim deployment approval. The former VIF Critic crash/rut/evolution runtime is explicitly deprecated and retained only for historical compatibility. |
| **Weekly Coach** | 🧪 Experimental | The approved runtime sends deterministic Drift Detector output based on Weekly Drift Reviewer Decisions into the Weekly Digest and optional Weekly Coach reflection. The Weekly Digest cites supporting Journal Entries without VIF Critic or LLM-Judge numeric summaries. Explanation validation depth and product-facing orchestration remain incomplete; no fresh final test or deployment approval is claimed. |
| **Onboarding (BWS Values Assessment)** | 📋 Specified | 6-set BWS flow over 10 Schwartz dimensions; PVQ21-adapted card phrases; mid-flow + end-of-flow reflective mirrors; a graded 10-value weight vector for VIF Critic conditioning plus Core Values stored in `top_values` for Drift gating; 6 structured goal categories mapping to Weekly Coach monitoring priorities; scoring with confidence estimation and user refinement support; [full spec](onboarding/onboarding_spec.md) |
| **Embedding Explorer** | ✅ Complete | Interactive 3D visualization of VIF hidden-layer and SBERT embedding spaces; self-contained HTML with Three.js |
| **Journaling Anomaly Radar** | ❌ Not Started | Cadence/gap detection |
| **Goal-aligned Inspiration Feed** | ❌ Not Started | External API integration |

**Data Pipeline Progress:**
```
logs/
├── synthetic_data/     # 204 persona markdown files
├── wrangled/           # 204 cleaned files (generation metadata stripped)
├── judge_labels/       # 204 JSON label files + consolidated parquet
├── annotations/        # 3 annotator parquet files (380 saved annotations; 115-entry shared subset)
└── registry/           # personas.parquet (tracks pipeline stages)

models/
└── vif/                # Trained critic checkpoints (gitignored)
```

> **References:**
> - [Synthetic Data Pipeline](pipeline/pipeline_specs.md)
> - [Claude Code Generation Instructions](pipeline/claude_gen_instructions.md)
> - [Historical Claude LLM-Judge Labeling Instructions](pipeline/claude_judge_instructions.md)
> - [Historical LLM-Judge Reachability Audit Instructions](pipeline/judge_reachability_audit_instructions.md)
> - [Human Annotation Tool](pipeline/annotation_tool_plan.md)
> - [Drift Inspection App](demo/weekly_drift_review_app.md)
> - [VIF Critic Training](vif/03_model_training.md) — Training strategy and implementation
> - [CLAUDE.md](../CLAUDE.md) — Project architecture overview

---

# Elevator Pitch

* **Working name:** Twinkl — a long-horizon "inner compass."
* **What:** Journal reflections feed a living user model (values, identity themes, north star) that mirrors back where behaviour diverges from intent; it is not another "feel-better" journal.
* **Promise:** Honest, explainable alignment check-ins that combine deep introspection with accountability so users stop drifting from their declared priorities.
* **Capstone hook:** Pattern recognition + hybrid reasoning + explainable UX → direct throughline to all submodules.
* **Key properties:** Dynamic self-model that updates gradually, identity treated as slowly evolving, value-alignment questions over dopamine loops.

# Pain Point(s) it solves & Target Users

* **Pain points**
    * Ambitious people articulate values (health, family, creativity) yet their weeks quietly fill with conflicting work, doomscrolling, or obligation; very few tools hold up a mirror to that behavioral divergence.
    * Traditional journaling is high-friction and dies off; light prompts and low-barrier entry match how people naturally reflect, but current apps stay at mood-tracking or streak mechanics.
    * Users crave kind accountability—context-aware reflections that cite evidence—while commercial products optimise for dopamine loops, not truth.
* **Target users / addressable market**
    * Knowledge workers in transition (grad students, new managers, founders) and high-agency professionals managing career-family-growth trade-offs—large cohorts already paying for journaling + coaching, yet underserved by static apps.
    * Pick 1–2 personas for the capstone run; each provides rich, recurring scenarios to evaluate alignment/misalignment feedback.

# Difference vs commercial peers

AI journaling apps (Reflection, Mindsera, Insight Journal, Day One, Pixel Journal, Rosebud) summarise moods and trends yet treat every entry as an isolated blob; none maintain a dynamic, explainable self-model that challenges users when their actions contradict their stated direction—leaving a white space for people already paying for coaching or multiple journaling subscriptions.

| Feature                | Scenario A: Current AI Journals (The "Summarizer")                                                                                                                                                                | Scenario B: Twinkl (The "Alignment Engine")                                                                                                                                                                       |
| :--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Premise**       | **Starts with a "Blank Slate."** Knowledge is built *only* from the Journal Entries as they come in.                                                                                                              | **Starts with a "Self-Model."** The user first defines their Core Values, goals, and priorities during onboarding.                                                                                                |
| **Example Self-Model** | *None exists.*                                                                                                                                                                                                    | **Value 1:** "My health is my foundation." **Value 2:** "My relationship is my anchor." **Priority 1:** "The 'Project X' at work is my focus this month."                                                         |
| **User Entries**       | *(Constant for both scenarios)*  1\. "So stressed, the big project at work is derailing everything." 2\. "Skipped the gym again... feel guilty." 3\. "Had a nice dinner with my partner, which was a good break." | *(Constant for both scenarios)*  1\. "So stressed, the big project at work is derailing everything." 2\. "Skipped the gym again... feel guilty." 3\. "Had a nice dinner with my partner, which was a good break." |

* **Alignment engine:** Weekly reasoning compares lived behaviour vs. declared priorities, surfaces tensions, and cites evidence snippets—turning “you said X but did Y” into actionable prompts.
* **Explainable accountability:** Every nudge shows why (phrases, time windows, rules), plus contextual quotes/interventions tuned to the conflict at hand.
* **Capstone-ready architecture:** synthetic-data generation, LLM-Judge
  labeling, an uncertainty-aware MLP, independent weekly review, and a
  deterministic two-Conflict rule provide concrete work across Intelligent
  Sensing, Pattern Recognition, Reasoning, and Architecting AI Systems.

# How it works

## **System loop**

1. **Perception:** Typed Journal Entries flow through an LLM that tags values, identity claims, sentiment, intent, and direction-of-travel.
2. **Memory:** Tags incrementally update a decay-aware user profile/knowledge base (value weights, goals, tensions, evidence snippets) instead of resetting each week.
3. **Reasoning + action:** The required user-facing path uses independent weekly review; the **[VIF Critic](vif/01_concepts_and_roadmap.md)** remains optional experimental research:
   * **VIF Critic:** A numeric, uncertainty-aware model that predicts `-1`, `0`, or `+1` for each value from the current Journal Entry plus the normalized 10-dimensional value profile. It uses [LLM-Judge labels for reward modeling](vif/03_model_training.md) and [MC Dropout for epistemic uncertainty](vif/04_uncertainty_logic.md). At `window_size: 1`, it has no date/time-gap feature, prior Journal Entries, demographics, or biography; larger legal-history windows remain an experiment. Existing code supports training, evaluation, raw output export, and timeline inference. A generalized review-and-retrain loop is not implemented and is not required for the current user-facing path. The exact relabeling invariant is defined in the [Security target contract](vif/security_target_contract.md).
   * **Drift Detector:** Receives Weekly Drift Reviewer Decisions made without VIF Critic input. Drift is two consecutive Conflicts for the same Core Value; other values do not cancel it. The Weekly Drift Reviewer model contract is fixed at `gpt-5.6-luna` with reasoning effort `low`. The implemented capstone POC persists versioned decisions, fails closed to Abstain on invalid or refused responses, and applies the deterministic rule across week boundaries. The complete development analysis contains 42 Drifts across 36 Drift trajectories in 292 resolved cases. The fixed Luna-low setup had median Drift recall of `0.548`, 4 false Drift alerts, and `0.637` coverage. These are AI-reviewed synthetic development results. The capstone stops without a fresh final test and therefore makes no deployment-approval claim. The optional VIF Critic candidate-confirmation path is outside the remaining capstone scope. The crash/rut/evolution router is explicitly deprecated and retained only for historical compatibility.
   * **Weekly Coach:** Receives the Weekly Digest and reads the user's full Journal Entry history via **full-context prompting** (at POC scale, all Journal Entries fit in the LLM context window) to surface thematic evidence, explain *why* Conflict occurred, and offer reflective prompts. For positive patterns, it provides occasional evidence-based acknowledgment without gamification. At production scale with longer histories, this would transition to retrieval-augmented generation (RAG). (See [System Architecture](vif/02_system_architecture.md)). For a concrete scenario, see [Worked Example: Sarah's Journey](vif/example.md). Drift Detector calibration and evaluation remain experimental — see [Implementation Status](#implementation-status).
   * A **possible future idea** is a **[Value Evolution Detection](evolution/01_value_evolution.md)** layer between VIF Critic predictions and the Drift Detector. If revisited later, it would aim to distinguish genuine value shifts from behavioral Drift. It is not part of the current committed system scope.

### Canonical VIF scope and evaluation contract

> Twinkl's VIF Critic is an optional experimental Conflict-screening component.
> If research resumes, entry-level `recall_-1` is the main model-development
> metric and QWK remains an ordinal-health diagnostic. The current user-facing
> Drift path does not run the VIF Critic or consume VIF Critic Predictions.

For the remaining capstone scope, Drift means two consecutive Conflicts for the
same Core Value.

- Entry-level `recall_-1` is the primary model-development metric.
- The Weekly Drift Reviewer model contract is fixed at `gpt-5.6-luna` with
  reasoning effort `low`. Its development evaluation prioritized Drift recall
  first and false Drift alerts second. Coverage and abstention are diagnostic
  metrics, not selection gates.
- The three complete Luna-low development Runs satisfy the evidence needed to
  freeze this capstone POC contract: each achieved Drift recall above `0.50`,
  with 5, 4, and 4 false Drift alerts across 256 non-Drift Core Value
  trajectories. No development rerun is required after the prompt cleanup.
  The capstone does not proceed to a fresh final test or deployment approval.
- QWK, `+1` recall, calibration, and circumplex metrics remain diagnostics.
- Only Core Values, stored in `top_values`, can produce Drift. `+1` evidence is
  non-gating and may support occasional positive Weekly Coach acknowledgment.
- An uncertain or abstaining Weekly Drift Reviewer produces no Drift claim;
  coverage, abstention, and suppressed known Drifts must still be reported even
  though they do not outrank Drift recall or false Drift alerts.
- Any future fresh final test should reuse the Luna-low response schema,
  fail-closed request handling, scoring, three-Run protocol, and reported
  metrics. It should add no separate efficiency gate or reporting-only
  acceptance criteria.
- The ternary ten-value VIF Critic output remains available for optional
  research. Weekly Drift Reviewer Decisions drive the current user-facing path.
  VIF-Critic-proposed candidates and a review-and-retrain loop are not part of
  the remaining critical capstone scope.

The detailed adopted decision and its implementation gaps are recorded in
[VIF Capstone Scope and Evaluation Decision](vif/05_capstone_scope_decision.md).

### Prompt Templates

LLM prompts are stored as YAML files with Jinja2 templating in `prompts/`:
- `persona_generation.yaml` — Generate synthetic personas with value context
- `journal_entry.yaml` — Generate entries from persona perspective
- `nudge_decision.yaml` — Classify entries for nudge appropriateness
- `nudge_generation.yaml` — Generate contextual follow-up nudges
- `nudge_response.yaml` — Generate persona responses to nudges
- `judge_alignment.yaml` — Score entries against Schwartz value dimensions

Value context is injected from `config/schwartz_values.yaml`, which contains rich psychological elaborations (core motivation, behavioral manifestations, life domain expressions) for each Schwartz dimension.

## **Product principles**

* Identity-first mini-assessment ("build your inner compass" via quick BWS screens of big, tappable cards and tradeoffs) before daily journaling. See the [Onboarding Spec](onboarding/onboarding_spec.md) for the canonical flow definition.
* Longitudinal honesty engine that gently says "you keep claiming X but living Y."
* Quotes/prompts are precision interventions tied to observed conflicts.
* Low-friction journaling: prompts reduce blank-page paralysis and encourage regular reflection.
* Evidence-based reinforcement, not gamification: when users sustain alignment with their values, the system acknowledges it by citing specific behaviors and connecting them to the user's own words — never through streaks, points, leaderboards, or generic praise. Positive feedback is infrequent (only when patterns emerge) and grounded in what the user actually wrote.

## **Onboarding (BWS Values Assessment)** 📋

The onboarding flow uses **Best-Worst Scaling (BWS)** — a forced-choice psychometric technique — to elicit value priorities across 10 Schwartz dimensions while minimizing social desirability bias. The **[Onboarding Spec](onboarding/onboarding_spec.md)** is the source of truth for the canonical flow (number of screens, interaction model, scoring logic, and data output schema). That spec is still subject to change as the design evolves, but this PRD defers to it for onboarding details rather than duplicating them here.

In summary: users complete BWS forced-choice screens, see reflective mirrors for correction, and select a structured goal category. The output includes a graded 10-dimensional value weight vector for VIF Critic conditioning, Core Values stored in `top_values` for Drift gating, and an initial Weekly Coach monitoring focus. Persisting and consuming both value fields remains implementation work under `twinkl-1m8`.

This onboarding directly anchors the capstone submodules: the latent dimensions form named slots in the knowledge base and rule layer (**Intelligent Reasoning Systems**), the mapping from user responses to those dimensions plus later corrections is a compact supervised modelling task (**Pattern Recognition Systems**), entry content analysis and temporal patterns feed the sensing layer (**Intelligent Sensing Systems**), and treating the quiz as just one input stream into a shared user-state vector `z` illustrates end-to-end orchestration and state management across Perception → Memory → Reasoning → Action (**Architecting AI Systems**).

## **Core Feature Modules**

* **Weekly Coach** ⚠️: Batch Journal Entries, run the reasoning components, and produce a one-page Weekly Digest (Pattern Recognition + Reasoning).
* **Conversational introspection agent** 🧪: Live mirroring via agent loop (Perception → Cognition → Action) to highlight contradictions mid-conversation. The system uses a three-category **nudge taxonomy**:
  - **Clarification** — for vague entries lacking concrete details
  - **Elaboration** — for surface-level entries with unexplored depth
  - **Tension-surfacing** — for hedging language or conflicted statements

  Nudge decisions use **LLM-based semantic classification** (not regex/heuristics) to detect when deeper reflection would yield VIF signal. Anti-annoyance logic caps nudges at 2 per 3-entry window. See [pipeline_specs.md](pipeline/pipeline_specs.md) for implementation details.
* **”Map of Me”** ❌: Embed each entry, visualise trajectories, overlay alignment scores (Pattern Recognition + Intelligent Sensing).
* **Journaling anomaly radar** ❌: After 2–3 weeks of entries establish cadence baselines, a lightweight time-series/anomaly detector tracks check-in gaps, flags “silent weeks,” cites evidence windows, and triggers empathetic nudges (Pattern Recognition + Architecting).
* **Goal-aligned inspiration feed** ❌: When the profile shows intent (e.g., “pick up Japanese”) but no supporting activities, call a real-time search API (SerpAPI/Tavily) constrained by what the user enjoys (e.g., highly rated anime) and reason over the results before surfacing next-step suggestions (Intelligent Reasoning + Intelligent Sensing). Each curated option is presented as an explicit choice; the user’s accept/decline actions feed back into the values/identity graph so future nudges learn which media or effort types actually motivate them.

**Implementation path**

1. Frame the research question (“How do we sustain a dynamic model of values/identity and reflect alignment?”) and map subsystems to submodules.
2. Define the MVP loop: onboarding (BWS-based values assessment — see [spec](onboarding/onboarding_spec.md))
3. **Scoping Strategy:** Adopt a **Hybrid Approach** (simple journaling loop + Weekly Digest + lightweight trajectory visualization). Build small slices of each feature to demonstrate breadth without over-building.
4. Specify the profile schema:
   * **Value dimensions** anchored in [Schwartz's theory of basic human values](https://en.wikipedia.org/wiki/Theory_of_basic_human_values) (e.g., Self-Direction, Benevolence, Achievement, Security) with definitions, rubrics, and examples.
   * **User value profile:** vector of value weights `w_u ∈ ℝ^K` (normalized, sum to 1), Core Values stored in `top_values`, plus narrative descriptions and constraints. The full vector conditions the VIF Critic; `top_values` gates Drift v1.
   * **State representation:** the current Journal Entry embedding plus a
     10-dimensional value-weight vector. Configurable legal-history windows
     remain experimental; no label-derived history features are allowed at
     inference time.
5. Implement **[LLM-Judge labeling and VIF Critic training](vif/03_model_training.md):** For each Journal Entry, the LLM-Judge outputs per-dimension categorical alignment labels in `{-1, 0, +1}` with rationales. Use synthetic personas for initial training and validation.

   > **Status:** Steps 1-5 complete (204 personas, 1,651 labeled Journal Entries). Human annotation tool is operational with 380 saved annotations, including the current 115-entry shared subset used for inter-rater agreement. Multiple VIF Critic architectures have been evaluated (ordinal MLP heads, BNN, TCN). See [Implementation Status](#implementation-status) for current progress. Step 6 (additional lightweight classifiers) is optional and deferred until a user-facing need justifies it.

6. Tooling: start with API LLM for tagging + reflection, add lightweight classifiers later if needed; keep reasoning layer explainable for XRAI.
7. Evaluation plan: combine Likert feedback on "felt accurate?" with inter-rater agreement on value tags and stability metrics for the profile.
8. Instrument the inspiration feed so each recommendation decision (accept/reject/ignore) is stored as structured evidence linked to values, identities, and interest embeddings, enabling closed-loop personalization.

| Component | Traditional Journaling (Summarizer) | Twinkl (Alignment Engine) |
| :--- | :--- | :--- |
| **Process** | **1. Tagging:** Identifies sentiment and topics.<br>• Journal Entry 1: Negative, Work<br>• Journal Entry 2: Guilt, Health<br>• Journal Entry 3: Positive, Partner<br>**2. Aggregation:** Groups these tags together. | **1. Reasoning:** Compares Journal Entries *against* the Self-Model.<br>• Journal Entry 1 → **Matches** Priority 1 = **Expected Friction**<br>• Journal Entry 2 → **Conflicts with** Value 1 = **Conflict (`-1`)**<br>• Journal Entry 3 → **Matches** Value 2 = **Alignment** |
| **Question it Answers** | **"What have I been feeling/talking about?"** | **"Am I living in line with what I *said* I value?"** |
| **Final Output (Insight)** | A high-level summary: "This week, your mood was primarily **stressed** and **guilty**. Your main topics were **'Work'** and the **'Gym'**. A dinner with your **'Partner'** was a positive moment." | An evidence-based alignment report:<br>**1. Alignment (Partner):** You honored your 'Partnership' value. (Evidence: *'nice dinner...'*)<br>**2. Misalignment (Health):** You broke your 'Health' value. (Evidence: *'Skipped the gym...'*)<br>**Prompt:** Your 'Work' priority is creating high stress, just as you expected, but it is now in conflict with your 'Health' value. Is this an acceptable trade-off for this week?" |
| **Core Concept** | **Retrospective Summarization** | **Prospective Accountability** |

**Twinkl’s edge**

* **Structured self-model:** Onboarding + ongoing journaling build a knowledge base of values, goals, tensions, and identity themes that evolves with decay-aware updates.
* **Alignment engine:** Weekly reasoning compares lived behaviour vs. declared priorities, surfaces tensions, and cites evidence snippets—turning “you said X but did Y” into actionable prompts.
* **Explainable accountability:** Every nudge shows why (phrases, time windows, rules), plus contextual quotes/interventions tuned to the conflict at hand.
* **Capstone-ready architecture:** Synthetic-data generation, LLM-Judge
  labeling, an uncertainty-aware MLP, independent weekly review, and a
  deterministic two-Conflict rule provide concrete work across Intelligent
  Sensing, Pattern Recognition, Reasoning, and Architecting AI Systems.

## Design Lessons Learned

### Metadata Leakage in Synthetic Data

A critical anti-pattern discovered during development: using synthetic generation instructions (e.g., `tone: Exhausted`, `reflection_mode: Neutral`) in decision logic creates train/serve skew. These labels exist only during data generation — they won't be available in production.

**Resolution:** All nudge decision logic now uses only **observable content signals**:
- Entry word count
- Presence of concrete details (nouns/verbs)
- Hedging language patterns ("sort of", "I guess", "maybe")
- Previous nudge history

Generation instructions remain useful for creating diverse training data, but must never influence runtime decisions.

### LLM vs. Rule-Based Classification

Early nudge logic used regex patterns for hedging detection. This was replaced with LLM-based semantic classification for:
- Better handling of context-dependent language
- Reduced false positives on quoted speech or hypotheticals
- Simpler maintenance (prompt updates vs. regex engineering)

The tradeoff is latency (additional LLM call), acceptable for conversational journaling but may need distillation for real-time use cases.

### LLM-Judge vs VIF Critic Context Windows

A key architectural decision: the LLM-Judge and VIF Critic use different context windows.

| Component | Context | Rationale |
|-----------|---------|-----------|
| **LLM-Judge** | Persona context plus previous Journal Entries | Better labeling: trajectory context helps disambiguate vague Journal Entries like "feeling better" |
| **VIF Critic** | Current Journal Entry plus normalized value profile (`window_size: 1`) | Fixed student-visible contract for fast local inference + MC Dropout |

**Why decouple?** The LLM-Judge runs offline during experimental training-data creation, while the user-facing path relies on the Weekly Drift Reviewer and Drift Detector. The frozen-holdout LLM baseline shows that adding previous Journal Entries improves the `human_context` setup's `recall_-1`, while the local MLP retains higher `recall_-1` and lower hedging. That result remains useful research evidence; it does not make the VIF Critic a runtime dependency.

This avoids the trap of matching windows "for consistency" when the constraints are fundamentally different.

# Potential Stretch goals

| Goal | Why it matters |
| :--- | :--- |
| **Neuro-symbolic reasoning** | Add a tiny knowledge graph + rule layer on top of LLM outputs to show which logical checks fired (great for XRAI storytelling). |
| **Multimodal fusion** | *Future work (out of scope for capstone):* Blend text + prosodic audio cues to extend Intelligent Sensing value beyond text-only analysis. |
| **Personalised quote recommender** | Build embeddings of quotes + user resonance to deliver “micro-anchors” tuned to each identity conflict. |
| **Distilled VIF Critic** | Train a smaller supervised model from LLM-Judge labels, reducing latency and cost while enabling offline inference. (See [Model Training](vif/03_model_training.md)) |
| **Ordinal regression models** | Treat alignment as ordinal classification {-1, 0, +1} instead of regression; architectures under investigation include CORAL, CORN, EMD, and soft ordinal ranking losses. |
| **Advanced uncertainty modeling** | Extend MC Dropout with ensembles or density models; add explicit OOD detectors on the text embedding space. (See [Uncertainty Logic](vif/04_uncertainty_logic.md)) |
| **Tiered VIF implementation** | Progress from Tier 1 (immediate alignment) → Tier 2 (short-horizon forecast) → Tier 3 (time-aware discounted returns). See [VIF design](vif/01_concepts_and_roadmap.md). |

# Features that tie back to Masters' submodules

| Submodule                         | Features in Twinkl                                                                                                                                                                                                                  |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intelligent Reasoning Systems** | Formal value/goal knowledge base + decay rules cover knowledge representation; a hybrid reasoning layer mixes LLM inference with symbolic “if value X high but mentions drop Y weeks → flag misalignment” rules, and the inspiration feed performs decision-theoretic ranking (with [uncertainty-aware scoring](vif/04_uncertainty_logic.md)) of real-time search hits plus logged user accept/reject choices. |
| **Pattern Recognition Systems**   | Transformer tagging for sentiment/topics, sequential models for cadence baselines, clustering/trajectory viz (“Map of Me”) to detect seasons, and anomaly detection that spots journal absences while continuously re-learning from the recommendation-choice dataset. |
| **Intelligent Sensing Systems**   | Text-based sensing: entry content analysis (value mentions, sentiment, hedging), temporal patterns (entry cadence, time-of-day), and journal gap detection. The real-time search layer acts as an external "sensor" that ingests up-to-date cultural/learning stimuli, and choice telemetry becomes another sensed signal that is fused with identity/value embeddings. *(Multimodal audio sensing deferred to future work.)* |
| **Architecting AI Systems**       | Agentic loop (Perception → Memory → Reasoning → Action), explainable feedback via XRAI, privacy-first storage of sensitive logs, and orchestration of background workers that run anomaly checks, call external APIs, and write preference updates while following MLSecOps guardrails. |



# Evaluation Strategy

**Purpose:** Validate that Twinkl's alignment engine produces outputs that are (1) technically correct and (2) subjectively useful to users—without over-investing in elaborate benchmarks that won't change design decisions for a time-boxed POC.

| # | Component | Purpose | Method | Example |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Value-mention tagging** | Verify LLM correctly identifies which Schwartz values a Journal Entry touches | Hand-label 50 Journal Entries with Schwartz value dimensions. Measure **Cohen's κ** between LLM and human labels. | Journal Entry: *"Dropped everything to help my sister move."* Human tags: `Benevolence`. LLM tags: `Benevolence`. → Agreement ✓ |
| 2 | **VIF Critic Conflict screening** | Evaluate optional local-model research without collapsing into neutral predictions | If research resumes, prioritize entry-level `recall_-1`; report `-1` precision and precision-recall behavior alongside QWK, calibration, `+1` recall, and per-dimension diagnostics. | Better Conflict recovery may justify later offline research; it does not receive user-facing Drift authority or block the current capstone path. |
| 3 | **Drift detection** | Confirm the Drift Detector finds Drift for a Core Value | Evaluate the fixed `gpt-5.6-luna` reasoning-effort-`low` Weekly Drift Reviewer with the displayed-behavior target: each of two consecutive Journal Entries must clearly show Conflict against the same Core Value. The complete development analysis contains 42 Drifts across 36 Drift trajectories in 292 resolved cases; report historical provenance subgroups rather than treating them as separate evaluation sets. The earlier 106-case `twinkl-752.5` result leaves the raw-input comparison inconclusive and finds no scheduling recall gain. Prioritize Drift recall first and false Drift alerts second; report coverage as a diagnostic. No fresh final test or deployment approval is claimed. | Drift: two consecutive Journal Entries both visibly show Conflict against Benevolence. An uncertain Weekly Drift Reviewer decision produces no Drift. `+1` on another value cannot cancel the Drift. |
| 4 | **Explanation quality** | Ensure explanations feel accurate and actionable | Show 5–10 users their Weekly Digest and ask "Did this feel accurate?" on a **5-point Likert scale**. | User sees: *"You wrote twice about wanting to make room for people close to you, then cancelled on a friend."* Rates it 4/5 for accuracy. |
| 5 | **Nudge relevance** | Verify the top prompt is contextually appropriate | A/B test: random prompt vs. model-selected prompt. Measure **engagement rate** (did user respond?). | Model picks *"What held you back from helping?"* after detecting a Benevolence Conflict. User responds → engagement ✓ |
| 6 | **Nudge signal quality** | Validate that nudging improves VIF Critic training data | Compare LLM-Judge alignment scores for nudged vs. non-nudged Journal Entries from the same personas. Measure **mean alignment confidence** and **value dimension coverage**. | Hypothesis: Nudged Journal Entries yield higher-confidence scores and more explicit value signals due to increased expressiveness. |

## Operational & User Success Metrics

| Category | Metrics |
| :--- | :--- |
| **User impact** | Likert ratings on "helps me act in line with values," % of suggested weekly experiments attempted, retention over a 1–2 week pilot. |
| **System & safety** | Latency from entry → feedback, LLM failure rates, privacy posture (encryption, export/delete), and qualitative review of guardrails for "it's not therapy" messaging. |

**Validation approach:** Mini user study (5–10 people over 1–2 weeks) focusing on "felt accuracy" plus synthetic stress tests for technical correctness.

# Related Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](../CLAUDE.md) | Project architecture, commands, code style |
| **Pipeline** | |
| [pipeline_specs.md](pipeline/pipeline_specs.md) | Synthetic data workflow design and rationale |
| [claude_gen_instructions.md](pipeline/claude_gen_instructions.md) | Parallel subagent generation workflow |
| [claude_judge_instructions.md](pipeline/claude_judge_instructions.md) | Historical LLM-Judge labeling workflow (wrangling + scoring) |
| [judge_reachability_audit_instructions.md](pipeline/judge_reachability_audit_instructions.md) | LLM-agnostic workflow for the twinkl-747 reachability audit |
| [annotation_guidelines.md](pipeline/annotation_guidelines.md) | Human annotation for nudge effectiveness study |
| [annotation_tool_plan.md](pipeline/annotation_tool_plan.md) | Shiny annotation tool implementation plan |
| [nudge_design_rationale.md](pipeline/nudge_design_rationale.md) | Nudge validation plan and design rationale |
| **VIF** | |
| [01_concepts_and_roadmap.md](vif/01_concepts_and_roadmap.md) | Value Identity Function theory |
| [02_system_architecture.md](vif/02_system_architecture.md) | System architecture, state, and runtime flow |
| [03_model_training.md](vif/03_model_training.md) | LLM-Judge labeling and VIF Critic training |
| [04_uncertainty_logic.md](vif/04_uncertainty_logic.md) | VIF Critic uncertainty and Drift review logic |
| [05_capstone_scope_decision.md](vif/05_capstone_scope_decision.md) | Adopted VIF capstone scope, metric hierarchy, and deferred decisions |
| [example.md](vif/example.md) | Worked end-to-end VIF behavior example |
| **Evals** | |
| [evals/overview.md](evals/overview.md) | Evaluation workflow overview |
| [evals/judge_validation_summary.md](evals/judge_validation_summary.md) | LLM-Judge validation results |
| [drift/trajectory_eda.md](drift/trajectory_eda.md) | Historical empirical basis for the Drift definition |
| [evals/drift_detection_eval.md](evals/drift_detection_eval.md) | Drift Detector target and evaluation protocol |
| [evals/drift_v1_student_visible_target.md](evals/drift_v1_student_visible_target.md) | Historical five-episode development review and withheld former final-test score |
| [twinkl-752.4 full review](../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) | Reviewed cohort and 33-episode union correction |
| [twinkl-752.5 Opus label resolution](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md) | Four resolved Conflict labels and the 106/106-resolved development union |
| [twinkl-752.5 reassessment](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) | Raw VIF Critic input, scheduling, trigger placement, and subgroup results |
| [twinkl-qtwz complete development review](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md) | Complete 292-case development labels and 42-Drift contract |
| [twinkl-52zz Luna reasoning-effort comparison](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md) | Evidence behind the fixed Luna-low model contract, metric hierarchy, cost, and limitations |
| **Other** | |
| [architecture/e2e_architecture.md](architecture/e2e_architecture.md) | High-level product and system map |
| [weekly/weekly_digest_generation.md](weekly/weekly_digest_generation.md) | Weekly Digest contract, runtime commands, and generated files |
| [demo/weekly_drift_review_app.md](demo/weekly_drift_review_app.md) | Read-only comparison of frozen Weekly Drift Reviewer development Runs, including the fixed Luna-low setup |
| [demo/review_app.md](demo/review_app.md) | Runtime Demo Review App for the local VIF Critic-to-Weekly-Digest path |
| [01_value_evolution.md](evolution/01_value_evolution.md) | Concept note for a possible future filter distinguishing value evolution from Drift |
| [onboarding_spec.md](onboarding/onboarding_spec.md) | BWS-based onboarding flow, item design, and data output schema |
| [capstone_report/](capstone_report/) | Current capstone report work and guidance for new reports |
| [April 2026 proposal submission](archive/capstone/2026-04-proposal-submission/) | Immutable snapshot of the already-submitted proposal, slides, figures, and sources |
