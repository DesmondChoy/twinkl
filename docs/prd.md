# About This Project

Twinkl is an academic capstone project for the **NUS Master of Technology in Intelligent Systems (AI Systems)** program, with an expected duration of 6–9 months. The project spans multiple submodules including Intelligent Reasoning Systems, Pattern Recognition Systems, Intelligent Sensing Systems, and Architecting AI Systems. For additional context and presentation materials, see [IS_Capstone_Slides.pdf](IS_Capstone_Slides.pdf).

---

## Implementation Status

*Last updated: 2025-01-19*

| Feature | Status | Details |
|---------|--------|---------|
| **Synthetic Data Pipeline** | ✅ Complete | 31 personas generated via Claude Code parallel subagents; YAML prompt templates with Jinja2 |
| **Judge Labeling (VIF)** | ✅ Complete | 213 entries labeled across 31 personas; two-phase pipeline (Python wrangling + parallel subagents); consolidated to `judge_labels.parquet` with rationales |
| **VIF Critic Training** | ✅ Complete | MLP critic (370K params) with MC Dropout; SBERT text encoder (384-dim); persona-level train/val/test splits; `src/vif/` module with CLI training script |
| **Human Annotation Tool** | ✅ Complete | ~4,200 LOC Shiny app; 56 annotations across 3 annotators; Cohen's κ / Fleiss' κ metrics; modular components with analysis view |
| **Conversational Nudging** | 🧪 Experimental | 3-category LLM classification (clarification/elaboration/tension-surfacing); pending validation that nudging improves VIF signal quality |
| **Weekly Alignment Coach** | ⚠️ Partial | Entry processing ready; digest generation not implemented |
| **Onboarding (BWS Values Assessment)** | 📋 Specified | [Spec](onboarding/onboarding_spec.md) |
| **"Map of Me" Visualization** | ❌ Not Started | Embedding trajectories |
| **Journaling Anomaly Radar** | ❌ Not Started | Cadence/gap detection |
| **Goal-aligned Inspiration Feed** | ❌ Not Started | External API integration |

**Data Pipeline Progress:**
```
logs/
├── synthetic_data/     # 31 persona markdown files
├── wrangled/           # 31 cleaned files (generation metadata stripped)
├── judge_labels/       # 31 JSON label files + consolidated parquet
├── annotations/        # 3 annotator parquet files (56 entries)
└── registry/           # personas.parquet (tracks pipeline stages)

models/
└── vif/                # Trained critic checkpoints (gitignored)
```

> **References:**
> - [Synthetic Data Pipeline](synthetic_data/pipeline_specs.md)
> - [Claude Code Generation Instructions](synthetic_data/claude_gen_instructions.md)
> - [Claude Judge Labeling Instructions](synthetic_data/claude_judge_instructions.md)
> - [Human Annotation Tool](data_loader/human_annotator_tool.md)
> - [VIF Critic Training](VIF/VIF_03_Model_Training.md) — Training strategy and implementation
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
3. **Reasoning + action:** A two-stage evaluative layer powered by the **[Value Identity Function (VIF)](VIF/VIF_01_Concepts_and_Roadmap.md)**:
   * **Critic (VIF):** A numeric, uncertainty-aware engine that computes per-value-dimension alignment scores from a sliding window of recent entries. Uses [LLM-as-Judge for reward modeling](VIF/VIF_03_Model_Training.md) and [MC Dropout for epistemic uncertainty](VIF/VIF_04_Uncertainty_Logic.md). Triggers critiques only when confident and detecting significant patterns (sudden crashes or chronic ruts).
   * **Coach:** Activated when the Critic identifies significant patterns — whether problematic (crashes, ruts) or positive (sustained alignment). Uses retrieval-augmented generation (RAG) over the user's full journal history to surface thematic evidence, explain *why* misalignment occurred, and offer reflective prompts or "micro-anchors." For positive patterns, provides occasional evidence-based acknowledgment without gamification. (See [System Architecture](VIF/VIF_02_System_Architecture.md)). For a concrete scenario, see [Worked Example: Sarah's Journey](VIF/VIF_Example.md).

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

* Identity-first mini-assessment ("build your inner compass" via 3–5 quick screens of big, tappable cards and tradeoffs) before daily journaling.
* Longitudinal honesty engine that gently says "you keep claiming X but living Y."
* Quotes/prompts are precision interventions tied to observed conflicts.
* Low-friction journaling: prompts reduce blank-page paralysis and encourage regular reflection.
* Evidence-based reinforcement, not gamification: when users sustain alignment with their values, the system acknowledges it by citing specific behaviors and connecting them to the user's own words — never through streaks, points, leaderboards, or generic praise. Positive feedback is infrequent (only when patterns emerge) and grounded in what the user actually wrote.

## **Onboarding (BWS Values Assessment)**

The onboarding flow uses **Best-Worst Scaling (BWS)** — a forced-choice psychometric technique — to elicit value priorities across 10 Schwartz dimensions while minimizing social desirability bias. Users complete 6 quick screens (4 items each, tap "most like me" / "least like me"), see mid-flow and end-of-flow reflective mirrors for correction, and select a structured goal category. The output is a graded 10-dimensional value weight vector plus initial Coach monitoring focus.

For the full specification including BWS item design, scoring logic, user flow, and data output schema, see **[Onboarding Spec](onboarding/onboarding_spec.md)**.

This onboarding directly anchors the capstone submodules: the latent dimensions form named slots in the knowledge base and rule layer (**Intelligent Reasoning Systems**), the mapping from user responses to those dimensions plus later corrections is a compact supervised modelling task (**Pattern Recognition Systems**), entry content analysis and temporal patterns feed the sensing layer (**Intelligent Sensing Systems**), and treating the quiz as just one input stream into a shared user-state vector `z` illustrates end-to-end orchestration and state management across Perception → Memory → Reasoning → Action (**Architecting AI Systems**).

## **Core Feature Modules**

* **Weekly alignment coach:** Batch entries, run the reasoning engine, ship a 1-page digest (Pattern Recognition + Reasoning).
* **Conversational introspection agent:** Live mirroring via agent loop (Perception → Cognition → Action) to highlight contradictions mid-conversation. The system uses a three-category **nudge taxonomy**:
  - **Clarification** — for vague entries lacking concrete details
  - **Elaboration** — for surface-level entries with unexplored depth
  - **Tension-surfacing** — for hedging language or conflicted statements

  Nudge decisions use **LLM-based semantic classification** (not regex/heuristics) to detect when deeper reflection would yield VIF signal. Anti-annoyance logic caps nudges at 2 per 3-entry window. See [pipeline_specs.md](synthetic_data/pipeline_specs.md) for implementation details.
* **“Map of Me”:** Embed each entry, visualise trajectories, overlay alignment scores (Pattern Recognition + Intelligent Sensing).
* **Journaling anomaly radar:** After 2–3 weeks of entries establish cadence baselines, a lightweight time-series/anomaly detector tracks check-in gaps, flags “silent weeks,” cites evidence windows, and triggers empathetic nudges (Pattern Recognition + Architecting).
* **Goal-aligned inspiration feed:** When the profile shows intent (e.g., “pick up Japanese”) but no supporting activities, call a real-time search API (SerpAPI/Tavily) constrained by what the user enjoys (e.g., highly rated anime) and reason over the results before surfacing next-step suggestions (Intelligent Reasoning + Intelligent Sensing). Each curated option is presented as an explicit choice; the user’s accept/decline actions feed back into the values/identity graph so future nudges learn which media or effort types actually motivate them.

**Implementation path**

1. Frame the research question (“How do we sustain a dynamic model of values/identity and reflect alignment?”) and map subsystems to submodules.
2. Define the MVP loop: onboarding (BWS-based values assessment — see [spec](onboarding/onboarding_spec.md))
3. **Scoping Strategy:** Adopt a **Hybrid Approach** (Simple journaling loop + weekly digest + lightweight trajectory viz). Build small slices of each feature to demonstrate breadth without over-building.
4. Specify the profile schema:
   * **Value dimensions** anchored in [Schwartz's theory of basic human values](https://en.wikipedia.org/wiki/Theory_of_basic_human_values) (e.g., Self-Direction, Benevolence, Achievement, Security) with definitions, rubrics, and examples.
   * **User value profile:** vector of value weights `w_u ∈ ℝ^K` (normalized, sum to 1), plus narrative descriptions and constraints.
   * **State representation:** sliding window of N recent entry embeddings + time deltas + history stats (EMA of per-dimension alignment, rolling std dev, entry counts).
5. Implement **[Reward Modeling (LLM-as-Judge)](VIF/VIF_03_Model_Training.md):** For each entry, LLM outputs per-dimension alignment scores (Likert scale normalized to [-1,1]) with rationales and optional confidence scores. Use synthetic personas for initial training/validation.

   > **Status:** Steps 1-5 complete (31 personas, 213 labeled entries). Human annotation tool operational with 56 annotations for inter-rater agreement. See [Implementation Status](#implementation-status) for current progress. Step 6 (lightweight classifiers) deferred pending Critic training results.

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
| **Judge** | All previous entries | Better labeling: trajectory context helps disambiguate vague entries like "feeling better" |
| **Critic** | Sliding window (N=3) | Inference efficiency: fixed window enables fast MLP inference + MC Dropout |

**Why decouple?** The Judge runs offline during training data creation — cost/latency is acceptable for better labels. The Critic runs at inference time — fixed window keeps the model efficient and the input dimension bounded.

This avoids the trap of matching windows "for consistency" when the constraints are fundamentally different.

# Potential Stretch goals

| Goal | Why it matters |
| :--- | :--- |
| **Neuro-symbolic reasoning** | Add a tiny knowledge graph + rule layer on top of LLM outputs to show which logical checks fired (great for XRAI storytelling). |
| **Multimodal fusion** | *Future work (out of scope for capstone):* Blend text + prosodic audio cues to extend Intelligent Sensing value beyond text-only analysis. |
| **Personalised quote recommender** | Build embeddings of quotes + user resonance to deliver “micro-anchors” tuned to each identity conflict. |
| **Distilled Reward Model** | Train a smaller supervised model to mimic LLM-as-Judge, reducing latency and cost while enabling offline VIF training. (See [Model Training](VIF/VIF_03_Model_Training.md)) |
| **Advanced uncertainty modeling** | Extend MC Dropout with ensembles or density models; add explicit OOD detectors on the text embedding space. (See [Uncertainty Logic](VIF/VIF_04_Uncertainty_Logic.md)) |
| **Tiered VIF implementation** | Progress from Tier 1 (immediate alignment) → Tier 2 (short-horizon forecast) → Tier 3 (time-aware discounted returns). See [VIF design](VIF/VIF_01_Concepts_and_Roadmap.md). |

# Features that tie back to Masters' submodules

| Submodule                         | Features in Twinkl                                                                                                                                                                                                                  |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intelligent Reasoning Systems** | Formal value/goal knowledge base + decay rules cover knowledge representation; a hybrid reasoning layer mixes LLM inference with symbolic “if value X high but mentions drop Y weeks → flag misalignment” rules, and the inspiration feed performs decision-theoretic ranking (with [uncertainty-aware scoring](VIF/VIF_04_Uncertainty_Logic.md)) of real-time search hits plus logged user accept/reject choices. |
| **Pattern Recognition Systems**   | Transformer tagging for sentiment/topics, sequential models for cadence baselines, clustering/trajectory viz (“Map of Me”) to detect seasons, and anomaly detection that spots journal absences while continuously re-learning from the recommendation-choice dataset. |
| **Intelligent Sensing Systems**   | Text-based sensing: entry content analysis (value mentions, sentiment, hedging), temporal patterns (entry cadence, time-of-day), and journal gap detection. The real-time search layer acts as an external "sensor" that ingests up-to-date cultural/learning stimuli, and choice telemetry becomes another sensed signal that is fused with identity/value embeddings. *(Multimodal audio sensing deferred to future work.)* |
| **Architecting AI Systems**       | Agentic loop (Perception → Memory → Reasoning → Action), explainable feedback via XRAI, privacy-first storage of sensitive logs, and orchestration of background workers that run anomaly checks, call external APIs, and write preference updates while following MLSecOps guardrails. |



# Evaluation Strategy

**Purpose:** Validate that Twinkl's alignment engine produces outputs that are (1) technically correct and (2) subjectively useful to users—without over-investing in elaborate benchmarks that won't change design decisions for a time-boxed POC.

| # | Component | Purpose | Method | Example |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Value-mention tagging** | Verify LLM correctly identifies which Schwartz values an entry touches | Hand-label 50 journal entries with Schwartz value dimensions. Measure **Cohen's κ** between LLM and human labels. | Entry: *"Dropped everything to help my sister move."* Human tags: `Benevolence`. LLM tags: `Benevolence`. → Agreement ✓ |
| 2 | **Value profile modeling** | Check if predicted Schwartz value rankings match ground truth | Create 3–5 synthetic personas with known value orderings. Feed their simulated entries and compare predicted vs. true rankings using **Spearman correlation**. | Persona "Mia" values Benevolence > Achievement > Self-Direction. After 10 entries, model predicts same ordering. ρ = 1.0 ✓ |
| 3 | **Drift detection** | Confirm system flags obvious misalignment | Generate 10 synthetic "crisis weeks" (e.g., Benevolence-first persona neglects family for work). Measure **hit rate**: did the system flag it? | Ground truth: Week 3 is a crisis. System flags Week 3? Yes → Hit. Target: ≥8/10 hits. |
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
| [pipeline_specs.md](synthetic_data/pipeline_specs.md) | Synthetic data pipeline design and rationale |
| [claude_gen_instructions.md](synthetic_data/claude_gen_instructions.md) | Parallel subagent generation workflow |
| [claude_judge_instructions.md](synthetic_data/claude_judge_instructions.md) | Judge labeling workflow (wrangling + scoring) |
| [annotation_guidelines.md](synthetic_data/annotation_guidelines.md) | Human annotation for nudge effectiveness study |
| [human_annotator_tool.md](data_loader/human_annotator_tool.md) | Shiny annotation tool implementation plan |
| [onboarding_spec.md](onboarding/onboarding_spec.md) | BWS-based onboarding flow, item design, and data output schema |
| [VIF_01_Concepts_and_Roadmap.md](VIF/VIF_01_Concepts_and_Roadmap.md) | Value Identity Function theory |
| [VIF_03_Model_Training.md](VIF/VIF_03_Model_Training.md) | LLM-as-Judge and Critic training |