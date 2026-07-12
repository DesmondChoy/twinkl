# About This Project

Twinkl is an academic capstone project for the **NUS Master of Technology in Intelligent Systems (AI Systems)** program, with an expected duration of 6–9 months. The project spans multiple submodules including Intelligent Reasoning Systems, Pattern Recognition Systems, Intelligent Sensing Systems, and Architecting AI Systems. For additional context and presentation materials, see [is_capstone_slides.pdf](is_capstone_slides.pdf).

---

## Implementation Status

*Last updated: 2026-07-12*

| Feature | Status | Details |
|---------|--------|---------|
| **Synthetic Data Pipeline** | ✅ Complete | 204 personas (1,651 entries) generated via Claude Code parallel subagents; YAML prompt templates with Jinja2; targeted value generation supports family-specific tension banks, frozen-holdout experiments, and judged acceptance gates for hard-dimension batches |
| **Judge Labeling (VIF)** | ✅ Complete | 1,651 entries labeled across 204 personas; two-phase pipeline (Python wrangling + parallel subagents); consolidated to `judge_labels.parquet` with rationales. A separate receipt-bound full-corpus Security review now provides a non-destructive `security_active_critic_state_v1` target; persisted labels remain immutable. |
| **VIF Critic Training** | 🧪 Experimental | Training stack complete with ordinal MLP heads, BNN baseline, and configurable sentence encoders (`nomic` active default; MiniLM/mpnet ablations). `run_019`-`run_021` remains the historical corrected-split reference. Repaired Security supervision raises median test Security QWK by about 0.17 without regressing aggregate QWK. Soft vote-distribution training changed behavior but was not promoted. Compact-history `run_069` stayed within its added-weight budget but regressed on QWK, minority recall, Security, hedging, and overfitting versus its seed-matched repaired-target baseline. A 20-pair Codex-reviewed Hedonism diagnostic found median `-1` recall of only 0.05 for the incumbent and 0.20 for the tail-sensitive reference, with strict both-members-correct pair rates of 0.05 and 0.15. This is AI diagnostic evidence, not human validation. The local MLP default remains `window_size: 1` and Hedonism remains an unresolved hard dimension. |
| **Human Annotation Tool** | ✅ Complete | ~4,200 LOC Shiny app; 380 saved annotations across 24 personas, with a 115-entry shared subset across 19 personas used for the current inter-rater agreement benchmark; Cohen's κ / Fleiss' κ metrics; modular components with analysis view; annotation ordering for persona prioritization |
| **Conversational Nudging** | 🧪 Experimental | 3-category LLM classification (clarification/elaboration/tension-surfacing); pending validation that nudging improves VIF signal quality |
| **Drift Detection Engine** | 🧪 Experimental | Drift v1 is a per-value sustained conflict episode: two adjacent entries must each clearly show the writer making a behavior or choice against the same declared core value. The former consensus-derived frozen benchmark is [retired historical evidence](archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md), not a target, threshold-selection input, or promotion surface. [`twinkl-v8pb` completed the separate student-visible review](evals/drift_v1_student_visible_target.md) using the full runtime text; `run_020` found 1 of 5 development episodes, and one 19-entry promotion case remained unresolved. No promotion score was run, so no scorer is promotion-ready and trigger wiring remains blocked. The weekly crash/rut/evolution prototype is still not the selected v1 detector. |
| **Weekly Alignment Coach** | 🧪 Experimental | Weekly digest generation consumes Judge labels or live VIF signal artifacts plus optional upstream drift output. The offline runtime exports validated weekly frames, structured drift payloads, prompts, JSON, markdown, and consolidated parquet records. Narrative generation, validation depth, product-facing orchestration, and trigger calibration remain incomplete. |
| **Onboarding (BWS Values Assessment)** | 📋 Specified | 6-set BWS flow over 10 Schwartz dimensions; PVQ21-adapted card phrases; mid-flow + end-of-flow reflective mirrors; a graded 10-value weight vector for Critic conditioning plus a discrete `top_values` declared-core set for drift gating; 6 structured goal categories mapping to Coach monitoring priorities; scoring with confidence estimation and user refinement support; [full spec](onboarding/onboarding_spec.md) |
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
> - [Claude Judge Labeling Instructions](pipeline/claude_judge_instructions.md)
> - [Judge Reachability Audit Instructions](pipeline/judge_reachability_audit_instructions.md)
> - [Human Annotation Tool](pipeline/annotation_tool_plan.md)
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
    * Ambitious people articulate values (health, family, creativity) yet their weeks quietly fill with conflicting work, doomscrolling, or obligation; very few tools hold up a mirror to that drift.
    * Traditional journaling is high-friction and dies off; light prompts and low-barrier entry match how people naturally reflect, but current apps stay at mood-tracking or streak mechanics.
    * Users crave kind accountability—context-aware reflections that cite evidence—while commercial products optimise for dopamine loops, not truth.
* **Target users / addressable market**
    * Knowledge workers in transition (grad students, new managers, founders) and high-agency professionals managing career-family-growth trade-offs—large cohorts already paying for journaling + coaching, yet underserved by static apps.
    * Pick 1–2 personas for the capstone run; each provides rich, recurring scenarios to evaluate alignment/misalignment feedback.

# Difference vs commercial peers

AI journaling apps (Reflection, Mindsera, Insight Journal, Day One, Pixel Journal, Rosebud) summarise moods and trends yet treat every entry as an isolated blob; none maintain a dynamic, explainable self-model that challenges users when their actions contradict their stated direction—leaving a white space for people already paying for coaching or multiple journaling subscriptions.

| Feature                | Scenario A: Current AI Journals (The "Summarizer")                                                                                                                                                                | Scenario B: Twinkl (The "Alignment Engine")                                                                                                                                                                       |
| :--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Premise**       | **Starts with a "Blank Slate."** Knowledge is built *only* from the entries as they come in.                                                                                                                      | **Starts with a "Self-Model."** The user first defines their core values, goals, and priorities during onboarding.                                                                                                |
| **Example Self-Model** | *None exists.*                                                                                                                                                                                                    | **Value 1:** "My health is my foundation." **Value 2:** "My relationship is my anchor." **Priority 1:** "The 'Project X' at work is my focus this month."                                                         |
| **User Entries**       | *(Constant for both scenarios)*  1\. "So stressed, the big project at work is derailing everything." 2\. "Skipped the gym again... feel guilty." 3\. "Had a nice dinner with my partner, which was a good break." | *(Constant for both scenarios)*  1\. "So stressed, the big project at work is derailing everything." 2\. "Skipped the gym again... feel guilty." 3\. "Had a nice dinner with my partner, which was a good break." |

* **Alignment engine:** Weekly reasoning compares lived behaviour vs. declared priorities, surfaces tensions, and cites evidence snippets—turning “you said X but did Y” into actionable prompts.
* **Explainable accountability:** Every nudge shows why (phrases, time windows, rules), plus contextual quotes/interventions tuned to the conflict at hand.
* **Capstone-ready architecture:** LLM tagging + time-series smoothing + symbolic rules = rich ground across Intelligent Sensing, Pattern Recognition, Reasoning, and Architecting AI Systems.

# How it works

## **System loop**

1. **Perception:** Typed journal entries flow through an LLM that tags values, identity claims, sentiment, intent, and direction-of-travel.
2. **Memory:** Tags incrementally update a decay-aware user profile/knowledge base (value weights, goals, tensions, evidence snippets) instead of resetting each week.
3. **Reasoning + action:** A two-stage evaluative layer powered by the **[Value Identity Function (VIF)](vif/01_concepts_and_roadmap.md)**:
   * **Critic (VIF):** A numeric, uncertainty-aware engine that computes per-value-dimension alignment estimates from the current student-visible state. It uses [LLM-as-Judge for reward modeling](vif/03_model_training.md) and [MC Dropout for epistemic uncertainty](vif/04_uncertainty_logic.md). The current config uses the runtime-formatted journal session plus the normalized 10-dimensional value profile. At `window_size: 1`, it has no date/time-gap feature, prior entries, demographics, or biography; larger legal-history windows remain an experiment. The exact relabeling invariant is defined in the [Security target contract](vif/security_target_contract.md).
   * **Drift detector:** Reads Critic outputs over time. The v1 product target is two adjacent entries that each visibly show a behavior or choice against the same declared core value; other values do not cancel that per-value episode. `twinkl-v8pb` completed the full-runtime-text development review and a locked promotion review. The development threshold found only 1 of 5 reference episodes, while one 19-entry promotion case was unresolved; the promotion score was not run. The old consensus-derived frozen benchmark is retired historical evidence, so no production scorer may be promoted from it. The intended runtime still estimates sustained conflict from rolling soft `P(-1)` evidence under uncertainty gating. The existing crash/rut/evolution router is an experimental implementation surface, not the selected v1 contract.
   * **Coach:** Receives a weekly structured artifact and reads the user's full journal history via **full-context prompting** (at POC scale, all entries fit in the LLM context window) to surface thematic evidence, explain *why* misalignment occurred, and offer reflective prompts. For positive patterns, it provides occasional evidence-based acknowledgment without gamification. At production scale with longer histories, this would transition to retrieval-augmented generation (RAG). (See [System Architecture](vif/02_system_architecture.md)). For a concrete scenario, see [Worked Example: Sarah's Journey](vif/example.md). Trigger calibration and evaluation remain experimental — see [Implementation Status](#implementation-status).
   * A **possible future idea** is a **[Value Evolution Detection](evolution/01_value_evolution.md)** layer between Critic outputs and drift triggers. If revisited later, it would aim to distinguish genuine value shifts from behavioral drift. It is not part of the current committed system scope.

### Canonical VIF scope and evaluation contract

> Twinkl's Critic is primarily a conflict-screening component. Its
> product-critical job is to recover `-1` evidence that supports correctly
> detecting sustained two-entry drift episodes. We maximize episode recall
> subject to a conservative precision/false-alert constraint. Entry-level
> `recall_-1` is the main model-development metric; QWK is retained only as an
> ordinal-health diagnostic.

For the remaining capstone scope, a sustained-conflict episode means two
adjacent entries that each visibly show a choice against the same declared
core value.

- Entry-level `recall_-1` is the primary model-development metric.
- Product evaluation prioritizes episode recall. A conservative precision or
  user-facing false-alert constraint must be chosen before deployment, but no
  numerical tolerance is adopted yet.
- QWK, `+1` recall, calibration, and circumplex metrics remain diagnostics.
- Only the discrete `top_values` set can trigger drift. `+1` evidence is
  non-gating and may support occasional positive Coach acknowledgment.
- An uncertain or abstaining scorer produces no drift claim; coverage and
  suppressed true episodes must be reported.
- The ternary ten-value output remains. No MLP, LLM, verifier, ensemble, or
  cascade architecture is adopted by this scope decision.

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

In summary: users complete BWS forced-choice screens, see reflective mirrors for correction, and select a structured goal category. The output includes a graded 10-dimensional value weight vector for Critic conditioning, a discrete `top_values` set representing the user's declared core values for drift gating, and an initial Coach monitoring focus. Persisting and consuming both value fields remains implementation work under `twinkl-1m8`.

This onboarding directly anchors the capstone submodules: the latent dimensions form named slots in the knowledge base and rule layer (**Intelligent Reasoning Systems**), the mapping from user responses to those dimensions plus later corrections is a compact supervised modelling task (**Pattern Recognition Systems**), entry content analysis and temporal patterns feed the sensing layer (**Intelligent Sensing Systems**), and treating the quiz as just one input stream into a shared user-state vector `z` illustrates end-to-end orchestration and state management across Perception → Memory → Reasoning → Action (**Architecting AI Systems**).

## **Core Feature Modules**

* **Weekly alignment coach** ⚠️: Batch entries, run the reasoning engine, ship a 1-page digest (Pattern Recognition + Reasoning).
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
3. **Scoping Strategy:** Adopt a **Hybrid Approach** (Simple journaling loop + weekly digest + lightweight trajectory viz). Build small slices of each feature to demonstrate breadth without over-building.
4. Specify the profile schema:
   * **Value dimensions** anchored in [Schwartz's theory of basic human values](https://en.wikipedia.org/wiki/Theory_of_basic_human_values) (e.g., Self-Direction, Benevolence, Achievement, Security) with definitions, rubrics, and examples.
   * **User value profile:** vector of value weights `w_u ∈ ℝ^K` (normalized, sum to 1), a discrete declared-core `top_values` set, plus narrative descriptions and constraints. The full vector conditions the Critic; `top_values` gates drift v1.
   * **State representation:** sliding window of N recent entry embeddings + time deltas + a 10-dim value-weight vector, with zero-padding for early timesteps and no label-derived history features at inference time.
5. Implement **[Reward Modeling (LLM-as-Judge)](vif/03_model_training.md):** For each entry, the Judge outputs per-dimension categorical alignment labels in `{-1, 0, +1}` with rationales. Use synthetic personas for initial training/validation.

   > **Status:** Steps 1-5 complete (204 personas, 1,651 labeled entries). Human annotation tool is operational with 380 saved annotations, including the current 115-entry shared subset used for inter-rater agreement. Multiple Critic architectures have been evaluated (ordinal MLP heads, BNN, TCN). See [Implementation Status](#implementation-status) for current progress. Step 6 (lightweight classifiers) remains deferred pending Critic training results.

6. Tooling: start with API LLM for tagging + reflection, add lightweight classifiers later if needed; keep reasoning layer explainable for XRAI.
7. Evaluation plan: combine Likert feedback on "felt accurate?" with inter-rater agreement on value tags and stability metrics for the profile.
8. Instrument the inspiration feed so each recommendation decision (accept/reject/ignore) is stored as structured evidence linked to values, identities, and interest embeddings, enabling closed-loop personalization.

| Component | Traditional Journaling (Summarizer) | Twinkl (Alignment Engine) |
| :--- | :--- | :--- |
| **Process** | **1. Tagging:** Identifies sentiment and topics.<br>• Entry 1: Negative, Work<br>• Entry 2: Guilt, Health<br>• Entry 3: Positive, Partner<br>**2. Aggregation:** Groups these tags together. | **1. Reasoning:** Compares entries *against* the Self-Model.<br>• Entry 1 → **Matches** Priority 1 = **Expected Friction**<br>• Entry 2 → **Conflicts with** Value 1 = **Misalignment**<br>• Entry 3 → **Matches** Value 2 = **Alignment** |
| **Question it Answers** | **"What have I been feeling/talking about?"** | **"Am I living in line with what I *said* I value?"** |
| **Final Output (Insight)** | A high-level summary: "This week, your mood was primarily **stressed** and **guilty**. Your main topics were **'Work'** and the **'Gym'**. A dinner with your **'Partner'** was a positive moment." | An evidence-based alignment report:<br>**1. Alignment (Partner):** You honored your 'Partnership' value. (Evidence: *'nice dinner...'*)<br>**2. Misalignment (Health):** You broke your 'Health' value. (Evidence: *'Skipped the gym...'*)<br>**Prompt:** Your 'Work' priority is creating high stress, just as you expected, but it is now in conflict with your 'Health' value. Is this an acceptable trade-off for this week?" |
| **Core Concept** | **Retrospective Summarization** | **Prospective Accountability** |

**Twinkl’s edge**

* **Structured self-model:** Onboarding + ongoing journaling build a knowledge base of values, goals, tensions, and identity themes that evolves with decay-aware updates.
* **Alignment engine:** Weekly reasoning compares lived behaviour vs. declared priorities, surfaces tensions, and cites evidence snippets—turning “you said X but did Y” into actionable prompts.
* **Explainable accountability:** Every nudge shows why (phrases, time windows, rules), plus contextual quotes/interventions tuned to the conflict at hand.
* **Capstone-ready architecture:** LLM tagging + time-series smoothing + symbolic rules = rich ground across Intelligent Sensing, Pattern Recognition, Reasoning, and Architecting AI Systems.

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

### Judge vs Critic Context Windows

A key architectural decision: the Judge (LLM-as-Judge for labeling) and Critic (VIF model) use different context windows.

| Component | Context | Rationale |
|-----------|---------|-----------|
| **Judge** | Persona context plus previous entries | Better labeling: trajectory context helps disambiguate vague entries like "feeling better" |
| **Critic** | Current journal session plus normalized value profile (`window_size: 1`) | Fixed student-visible contract for fast local inference + MC Dropout |

**Why decouple?** The Judge runs offline during training data creation, while the Critic runs locally at inference time. The frozen-holdout LLM baseline measures the consequence directly: adding previous entries improves the strongest LLM arm's `recall_-1`, but the local MLP still retains higher `recall_-1` and lower hedging. The result supports a target/context repair or teacher/fallback role for the LLM rather than an automatic replacement of the MLP.

This avoids the trap of matching windows "for consistency" when the constraints are fundamentally different.

# Potential Stretch goals

| Goal | Why it matters |
| :--- | :--- |
| **Neuro-symbolic reasoning** | Add a tiny knowledge graph + rule layer on top of LLM outputs to show which logical checks fired (great for XRAI storytelling). |
| **Multimodal fusion** | *Future work (out of scope for capstone):* Blend text + prosodic audio cues to extend Intelligent Sensing value beyond text-only analysis. |
| **Personalised quote recommender** | Build embeddings of quotes + user resonance to deliver “micro-anchors” tuned to each identity conflict. |
| **Distilled Reward Model** | Train a smaller supervised model to mimic LLM-as-Judge, reducing latency and cost while enabling offline VIF training. (See [Model Training](vif/03_model_training.md)) |
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
| 1 | **Value-mention tagging** | Verify LLM correctly identifies which Schwartz values an entry touches | Hand-label 50 journal entries with Schwartz value dimensions. Measure **Cohen's κ** between LLM and human labels. | Entry: *"Dropped everything to help my sister move."* Human tags: `Benevolence`. LLM tags: `Benevolence`. → Agreement ✓ |
| 2 | **Critic conflict screening** | Recover visible misalignment evidence without collapsing into neutral predictions | Prioritize entry-level `recall_-1`; report `-1` precision and precision-recall behavior alongside QWK, calibration, `+1` recall, and per-dimension diagnostics. No fixed precision floor is adopted during recall-focused development. | A candidate that recovers more true `-1` entries is useful development evidence, but cannot be deployed if the resulting false-alert burden is unacceptable. |
| 3 | **Drift detection** | Confirm the runtime detects sustained conflict with a declared core value | Use a student-visible target: each of two adjacent entries must clearly show a behavior or choice against the same declared core value. Optimize future product evaluation for episode recall, then choose a conservative precision or false-alert operating constraint before deployment. `twinkl-v8pb` found 1 of 5 development episodes and withheld the promotion score after one unresolved case, so no production claim exists. | Reference event: two adjacent entries both visibly show a clear choice against Benevolence. An uncertain scorer abstains. `+1` on another value cannot cancel the episode. |
| 4 | **Explanation quality** | Ensure explanations feel accurate and actionable | Show 5–10 users their weekly digest and ask "Did this feel accurate?" on a **5-point Likert scale**. | User sees: *"Your Benevolence score dropped—you mentioned helping others twice but cancelled on a friend."* Rates it 4/5 for accuracy. |
| 5 | **Nudge relevance** | Verify the top prompt is contextually appropriate | A/B test: random prompt vs. model-selected prompt. Measure **engagement rate** (did user respond?). | Model picks *"What held you back from helping?"* after detecting Benevolence drift. User responds → engagement ✓ |
| 6 | **Nudge signal quality** | Validate that nudging improves VIF training data | Compare Judge alignment scores for nudged vs. non-nudged entries from same personas. Measure **mean alignment confidence** and **value dimension coverage**. | Hypothesis: Nudged entries yield higher-confidence scores and more explicit value signals due to increased expressiveness. |

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
| [pipeline_specs.md](pipeline/pipeline_specs.md) | Synthetic data pipeline design and rationale |
| [claude_gen_instructions.md](pipeline/claude_gen_instructions.md) | Parallel subagent generation workflow |
| [claude_judge_instructions.md](pipeline/claude_judge_instructions.md) | Judge labeling workflow (wrangling + scoring) |
| [judge_reachability_audit_instructions.md](pipeline/judge_reachability_audit_instructions.md) | LLM-agnostic workflow for the twinkl-747 reachability audit |
| [annotation_guidelines.md](pipeline/annotation_guidelines.md) | Human annotation for nudge effectiveness study |
| [annotation_tool_plan.md](pipeline/annotation_tool_plan.md) | Shiny annotation tool implementation plan |
| [nudge_design_rationale.md](pipeline/nudge_design_rationale.md) | Nudge validation plan and design rationale |
| **VIF** | |
| [01_concepts_and_roadmap.md](vif/01_concepts_and_roadmap.md) | Value Identity Function theory |
| [02_system_architecture.md](vif/02_system_architecture.md) | System architecture, state, and runtime flow |
| [03_model_training.md](vif/03_model_training.md) | LLM-as-Judge and Critic training |
| [04_uncertainty_logic.md](vif/04_uncertainty_logic.md) | Uncertainty, drift, and trigger logic |
| [05_capstone_scope_decision.md](vif/05_capstone_scope_decision.md) | Adopted VIF capstone scope, metric hierarchy, and deferred decisions |
| [example.md](vif/example.md) | Worked end-to-end VIF behavior example |
| **Evals** | |
| [evals/overview.md](evals/overview.md) | Evaluation pipeline overview |
| [evals/judge_validation_summary.md](evals/judge_validation_summary.md) | Judge validation results |
| [drift/trajectory_eda.md](drift/trajectory_eda.md) | Empirical basis for the sustained-conflict drift definition and benchmark candidates |
| [evals/drift_detection_eval.md](evals/drift_detection_eval.md) | Sustained-conflict runtime target and evaluation protocol |
| [evals/drift_v1_student_visible_target.md](evals/drift_v1_student_visible_target.md) | Completed student-visible development review and blocked locked-promotion result |
| **Other** | |
| [architecture/e2e_architecture.md](architecture/e2e_architecture.md) | High-level product and system map |
| [weekly/weekly_digest_generation.md](weekly/weekly_digest_generation.md) | Weekly digest contract, runtime commands, and artifacts |
| [demo/review_app.md](demo/review_app.md) | Local Shiny review and detector-comparison UI |
| [01_value_evolution.md](evolution/01_value_evolution.md) | Concept note for a possible future filter distinguishing value evolution from behavioral drift |
| [onboarding_spec.md](onboarding/onboarding_spec.md) | BWS-based onboarding flow, item design, and data output schema |
| [capstone_report/](capstone_report/) | Current capstone report work and guidance for new artifacts |
| [April 2026 proposal submission](archive/capstone/2026-04-proposal-submission/) | Immutable snapshot of the already-submitted proposal, slides, figures, and sources |
