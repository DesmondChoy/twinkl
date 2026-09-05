# About This Project

Twinkl is an academic capstone project for the **NUS Master of Technology in Intelligent Systems (AI Systems)** program, with an expected duration of 6–9 months. The project spans multiple submodules including Intelligent Reasoning Systems, Pattern Recognition Systems, Intelligent Sensing Systems, and Architecting AI Systems. For additional context and presentation materials, see [capstone_requirements.pdf](capstone_report/capstone_requirements.pdf).

---

## Implementation Status

*Last verified: 2026-08-31*

| Feature | Status | Details |
|---------|--------|---------|
| **Synthetic Data Workflow** | ✅ Complete | 204 personas (1,651 Journal Entries) generated via Claude Code parallel subagents; YAML prompt templates with Jinja2; targeted value generation supports family-specific tension banks, frozen-holdout experiments, and judged acceptance gates for hard-dimension batches |
| **LLM-Judge Labeling** | ✅ Complete | 1,651 Journal Entries labeled across 204 personas; two-phase workflow (Python wrangling + parallel subagents); consolidated to `judge_labels.parquet` with rationales. A separate receipt-bound full-corpus Security review provides a non-destructive `security_active_critic_state_v1` target; persisted labels remain immutable. |
| **VIF Critic (Offline) Training** | ✅ Complete | The capstone training and evaluation stack is complete, with ordinal MLP heads, a BNN baseline, configurable sentence encoders, uncertainty estimates, raw output export, experiment logging, and recall-first checkpoint selection. `run_019`-`run_021` remains the historical corrected-split reference, while repaired-Security `run_060` is the nominated offline checkpoint. Repaired Security supervision raises median test Security QWK by about 0.17 without regressing aggregate QWK. Compact-history and matched Hedonism diagnostics did not establish a stronger product role. This is AI diagnostic evidence, not human validation. The VIF Critic (Offline) is not part of the user-facing Drift path, and no further work on it is planned for the time-boxed capstone. |
| **Human Annotation Tool** | ✅ Complete | ~4,200 LOC Shiny app; 380 saved annotations across 24 personas, with a 115-entry shared subset across 19 personas used for the current inter-rater agreement benchmark; Cohen's κ / Fleiss' κ metrics; modular components with analysis view; annotation ordering for persona prioritization |
| **Drift Inspection App** | ✅ Complete | Read-only desktop Shiny app for comparing Runs 1–3 across three frozen Weekly Drift Reviewer setups: `gpt-5.4-mini` at reasoning effort `none`, `gpt-5.6-luna` at reasoning effort `none`, and `gpt-5.6-luna` at reasoning effort `low`. It shows complete development and persona-level results, Journal Entries, AI-reviewed LLM-Judge Conflict Labels, Weekly Drift Reviewer Decisions, cited evidence, and verified input cutoffs without model or provider API calls. Local and Railway launch paths are documented in the [app guide](demo/weekly_drift_review_app.md). |
| **Conversational Nudging** | ✅ Complete | The runtime and manual Experience integration are implemented and tested, including safe failure, retry, reply, skip, and linked Inspect events. A displayed nudge gives the user an immediate, contextual interaction between Journal Entries and the Coach Digest. It is a product design choice. It does not depend on measurable gains for the VIF Critic (Offline) or Weekly Drift Detection. A future external pilot can measure response rate, continued journaling, and perceived relevance. |
| **Drift Detector** | ✅ Complete | The capstone POC implementation is complete and wired. It persists versioned Weekly Drift Reviewer Decisions without VIF Critic Predictions and recomputes the full history at each cutoff. It applies the deterministic two-consecutive-Conflict rule across week boundaries. Each Core Value has one current state: Active Drift, No Active Drift, or Insufficient Evidence. The output also keeps Historical Drift Records, current run length, the latest decision, and the end reason. A failed review always gives Insufficient Evidence. A valid Abstain or Journal Entry gap gives Insufficient Evidence only when it blocks a current Drift claim after recent Conflict evidence. The fixed Luna-low model contract retains AI-reviewed synthetic development evidence. A later development-only comparison found median Drift recall of `0.667` and 9 false Drift alerts for Luna-`xhigh`, compared with `0.548` recall and 4 false Drift alerts for Luna-low. Twinkl retains Luna-low because `xhigh` is a more aggressive operating point, not a clean improvement. No fresh final test or deployment approval is claimed. The former VIF Critic (Offline) crash/rut/evolution runtime is explicitly deprecated and retained only for historical compatibility. |
| **Coach Digest** | 🧪 Experimental | The Coach Digest runs after each Weekly Drift Detection result, including No Active Drift. It supplies the current state, a prior closed-week comparison, separate prior and current evidence, and cited Journal Entries to a prompt. It can describe that an active pattern did not continue only when a Not Conflict decision supports that deterministic change. It cannot treat Not Conflict as proof of improvement. If it cannot return a valid response, the Weekly Drift Detection result remains available. OpenAI response generation uses `gpt-5.6-luna` with reasoning effort `none`. The same five accepted key-week responses appear in the public Persona replay fixtures and the [evaluation manifest](../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json). All five passed all Coach Digest Validations. Coach Digest Evals scored mean correctness `4.80`, specificity `5.00`, non-prescriptive tone `5.00`, and tension honesty `4.60`; all reflective questions passed, with no failed verdicts or review flags. These scores are same-model AI review, not human validation. Coach Digest Evals can select OpenAI or Gemini independently from the generator and record both model identities. The deterministic Drift/control study selects the 42 known development Drifts and 42 matched controls, supports safe resume, and reports pass rates from Coach Digest Validations and means from Coach Digest Evals by group. No paid cross-provider or Drift/control result is committed. Future human calibration of the AI review, a fresh final test, and deployment approval remain incomplete. |
| **Onboarding (SVBWS Values Assessment)** | 🧪 Experimental | The onboarding phase of the shared React app implements the published 11-group, six-object balanced SVBWS design, then presents a label-free Core Value summary and first Journal Entry handoff. A confirmed Profile has at most two Core Values. If more than two values share the highest score, the user selects exactly two while the Profile retains every tied value. The app randomizes group and card order, stores raw 11-object BWS results separately from the ten-value Profile transformation, and omits midpoint feedback and unsupported confidence claims. The manual Experience synchronizes the confirmed Profile and browser-held interaction state with the in-memory Python boundary. Production multi-user storage remains outside the capstone. It is a research-grounded pilot instrument, not a validated Twinkl instrument. [Full spec](onboarding/onboarding_spec.md) |
| **Experience and Inspect React App** | 🚧 In Progress | Shared contracts, five deterministic week-by-week Persona replays, manual Journal Entries, displayed nudges with reply and skip actions, Journal Entry removal, Simulated time, explicit closed-week review, affected-week recomputation, safe retry, live Weekly Drift Reviewer integration, Drift, required Coach Digest runs, and linked Inspect events are implemented. Manual Journal Entry cards show the newest entry first, while Weekly Drift Detection and Inspect keep chronological order. Monday-through-Sunday weeks become eligible only after Sunday closes; saving a Journal Entry never reviews its open week. Persona selection, manual next-step replay, previous-week navigation, optional automatic replay and pause, restart, named jumps to key weeks, reduced-motion behavior, and no-future-data projection are wired into the shared React session. Scenario requests bypass the browser cache and retain catalogued SHA-256 verification. The frontend build imports the exact Coach Digest evaluation manifest for the saved Persona replay checks before Vite produces the deployable assets, and Inspect treats complete and reused Coach Digest events as available responses. A first-use notice gates manual writing, and Delete session removes matching browser and Python state only after confirmed Python deletion. The core release quality gate is complete. Coach Digest feedback, longitudinal Core Value history, and final professor walkthrough evidence remain open. Optional live rerun remains separate work. Narrow-screen phones remain the primary verification target. [Design](demo/experience_inspect_app.md) |
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
> - [VIF Critic (Offline) Training](vif/03_model_training.md) — Training strategy and implementation
> - [CLAUDE.md](../CLAUDE.md) — Project architecture overview

---

# Elevator Pitch

* **Working name:** Twinkl — a long-horizon "inner compass."
* **What:** A confirmed Profile anchors longitudinal Journal Entry review. Twinkl
  shows where behavior conflicts with Core Values; it is not another
  "feel-better" journal. Automatic Profile evolution is future work.
* **Promise:** Honest, explainable alignment check-ins that combine deep introspection with accountability so users stop drifting from their declared priorities.
* **Capstone hook:** Pattern recognition + hybrid reasoning + explainable UX → direct throughline to all submodules.
* **Key properties:** A confirmed Profile anchors the current capstone POC.
  Journal Entries add longitudinal evidence without automatically changing that
  Profile. An evolving Profile is future work.

# Pain Point(s) it solves & Target Users

* **Pain points**
    * Ambitious people articulate values (health, family, creativity) yet their weeks quietly fill with conflicting work, doomscrolling, or obligation; very few tools hold up a mirror to that behavioral divergence.
    * Traditional journaling is high-friction and dies off; light prompts and low-barrier entry match how people naturally reflect, but current apps stay at mood-tracking or streak mechanics.
    * Users crave kind accountability—context-aware reflections that cite evidence—while commercial products optimise for dopamine loops, not truth.
* **Target users / addressable market**
    * Knowledge workers in transition (grad students, new managers, founders) and high-agency professionals managing career-family-growth trade-offs—large cohorts already paying for journaling + coaching, yet underserved by static apps.
    * Use five curated personas for the capstone menu, with one recommended walkthrough. Together they cover seven Schwartz Core Values, all three current Drift states, and Historical Drift Records that end.

# Difference vs commercial peers

AI journaling apps (Reflection, Mindsera, Insight Journal, Day One, Pixel Journal, Rosebud) summarise moods and trends yet often treat each entry as an isolated item. Twinkl starts with a declared Profile, reviews Journal Entries over time, and cites evidence when behavior conflicts with Core Values. Automatic Profile evolution remains future work.

| Feature                | Scenario A: Current AI Journals (The "Summarizer")                                                                                                                                                                | Scenario B: Twinkl (Evidence-grounded Drift reflection)                                                                                                                                                                       |
| :--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Premise**       | **Starts with a "Blank Slate."** Knowledge is built *only* from the Journal Entries as they come in.                                                                                                              | **Starts with a confirmed Profile.** The user first defines their Core Values, goals, and priorities during onboarding.                                                                                                |
| **Example Profile** | *None exists.*                                                                                                                                                                                                    | **Value 1:** "My health is my foundation." **Value 2:** "My relationship is my anchor." **Priority 1:** "The 'Project X' at work is my focus this month."                                                         |
| **User Entries**       | *(Constant for both scenarios)*  1\. "So stressed, the big project at work is derailing everything." 2\. "Skipped the gym again... feel guilty." 3\. "Had a nice dinner with my partner, which was a good break." | *(Constant for both scenarios)*  1\. "So stressed, the big project at work is derailing everything." 2\. "Skipped the gym again... feel guilty." 3\. "Had a nice dinner with my partner, which was a good break." |

* **Weekly Drift Detection:** The Weekly Drift Reviewer compares closed-week
  Journal Entries with Core Values. The Drift Detector creates Drift only from
  two consecutive Conflicts for the same Core Value.
* **Inspectable accountability:** Weekly Drift Detection cites the relevant
  Journal Entries, and Inspect exposes Weekly Drift Reviewer Decisions, Drift
  Detector transitions, Coach Digest responses, and provider receipts.
* **Capstone-ready architecture:** synthetic-data generation, LLM-Judge
  labeling, an uncertainty-aware MLP, Weekly Drift Detection, and a
  deterministic two-Conflict rule provide concrete work across Intelligent
  Sensing, Pattern Recognition, Reasoning, and Architecting AI Systems.

# How it works

## **Current product loop**

1. **Perception:** Typed Journal Entries and Core Values enter the displayed
   nudge path and Weekly Drift Detection.
2. **Memory:** The Experience service stores the confirmed Profile, Journal
   Entries, displayed nudges and responses, Weekly Drift Detection outputs, and
   Coach Digest responses. The current POC does not update the Profile from
   Journal Entries. An evolving or decay-aware Profile is future work.
3. **Reasoning + action:** The required user-facing path reviews closed Monday-through-Sunday calendar weeks. The first partial week becomes eligible after its first Sunday; saving a Journal Entry does not review the open week. The completed **[VIF Critic (Offline)](vif/01_concepts_and_roadmap.md)** remains an offline research component:
   * **VIF Critic (Offline):** A numeric, uncertainty-aware model that predicts `-1`, `0`, or `+1` for each value from the current Journal Entry plus the normalized 10-dimensional value profile. It uses [LLM-Judge VIF Labels for reward modeling](vif/03_model_training.md) and [MC Dropout for epistemic uncertainty](vif/04_uncertainty_logic.md). At `window_size: 1`, it has no date/time-gap feature, prior Journal Entries, demographics, or biography; larger legal-history windows remain diagnostic experiments. Existing code supports training, evaluation, raw output export, and timeline inference. A generalized review-and-retrain loop is not implemented or planned for the time-boxed capstone. The exact relabeling invariant is defined in the [Security target contract](vif/security_target_contract.md).
   * **Weekly Drift Detection:** At the end of each week, the internal Weekly Drift Reviewer reviews Journal Entries without VIF Critic Predictions. Its fixed model contract is `gpt-5.6-luna` with reasoning effort `low`. The internal Drift Detector then applies the rule that two consecutive Conflicts for the same Core Value form Drift. The Drift Detector recomputes all stored decisions. It stores Active Drift, No Active Drift, or Insufficient Evidence per Core Value and keeps Historical Drift Records separately. It fails closed to Abstain on invalid or refused responses. The complete development analysis contains 42 Drifts across 36 Drift trajectories in 292 resolved cases. The fixed Luna-low setup had median Drift recall of `0.548`, 4 false Drift alerts, and `0.637` coverage. These are AI-reviewed synthetic development results. The capstone has no fresh final test or deployment approval. The crash/rut/evolution router remains only for historical compatibility.
   * **Coach Digest:** Runs after each Weekly Drift Detection result, including No Active Drift. It supplies the current state, a prior closed-week comparison, and separate evidence for both weeks. The prompt asks for an evidence-based user response and a reflective question. It does not treat a Not Conflict decision as proof of improvement. If no valid response is available, the Weekly Drift Detection result remains available. The Coach Digest does not decide whether Drift exists. See [System Architecture](vif/02_system_architecture.md) and [Worked Example: Sarah's Journey](vif/example.md).
   * A **possible future idea** is **[Value Evolution Detection](evolution/01_value_evolution.md)** inside Weekly Drift Detection. It would try to separate a real value change from behavioral Drift. It is not in the current scope.

### Canonical VIF scope and evaluation contract

> Twinkl's VIF Critic (Offline) is a completed capstone research component for
> Conflict screening.
> In the completed research, entry-level `recall_-1` was the main
> model-development metric, and QWK remains an ordinal-health diagnostic. The
> current user-facing Drift path does not run the VIF Critic (Offline) or consume VIF
> Critic Predictions.

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
- `twinkl-ck3w` later compared Luna `medium`, `high`, and `xhigh` on the same
  development data. Luna-`xhigh` raised median Drift recall from `0.548` to
  `0.667`, but raised median false Drift alerts from 4 to 9 and increased the
  current-rate full-run calculation. Twinkl therefore retains Luna-low as the
  fixed capstone contract. This no-change decision uses development evidence
  and grants no deployment approval.
- QWK, `+1` recall, calibration, and circumplex metrics remain diagnostics.
- Only Core Values, stored in `top_values`, can produce Drift. `+1` evidence is
  non-gating and may support occasional positive Coach Digest acknowledgment.
- A failed review always produces Insufficient Evidence because it prevents a
  current claim. A valid Abstain or Journal Entry gap produces Insufficient
  Evidence only when it blocks a current Drift claim after recent Conflict
  evidence; without recent Conflict evidence, it leaves No Active Drift. None
  of these cases produces a Drift claim. Coverage, abstention, and suppressed
  known Drifts must still be reported even though they do not outrank Drift
  recall or false Drift alerts.
- Any future fresh final test should reuse the Luna-low response schema,
  fail-closed request handling, scoring, three-Run protocol, and reported
  metrics. It should add no separate efficiency gate or reporting-only
  acceptance criteria.
- The ternary ten-value VIF Critic Predictions remain available for offline
  reproduction. Weekly Drift Reviewer Decisions drive the current user-facing
  path. No review-and-retrain loop based on VIF Critic Predictions is planned
  for the time-boxed capstone.

The detailed adopted decision and its implementation gaps are recorded in
[VIF Capstone Scope and Evaluation Decision](vif/05_capstone_scope_decision.md).

### Prompt Templates

LLM prompts are stored as YAML files with Jinja2 templating in `prompts/`:
- `persona_generation.yaml` — Generate synthetic personas with value context
- `journal_entry.yaml` — Generate entries from persona perspective
- `nudge_decision.yaml` — Classify entries for nudge appropriateness
- `nudge_generation.yaml` — Generate contextual follow-up nudges
- `nudge_decision_and_generation.yaml` — Make the Experience nudge decision
  and optionally generate its question in one call
- `nudge_response.yaml` — Generate persona responses to nudges
- `judge_alignment.yaml` — Score entries against Schwartz value dimensions

Value context is injected from `config/schwartz_values.yaml`, which contains rich psychological elaborations (core motivation, behavioral manifestations, life domain expressions) for each Schwartz dimension.

## **Product principles**

* Mobile-first interaction: design and verify the React Experience and Inspect
  demo for narrow-screen phones first, then progressively enhance it for wider
  screens. Native mobile packaging is not required for the capstone.
* Identity-first mini-assessment ("build your inner compass" via quick BWS screens of illustrated cards and tap-and-drag trade-offs) before daily journaling. See the [Onboarding Spec](onboarding/onboarding_spec.md) for the canonical flow definition.
* Evidence-grounded longitudinal comparison between Journal Entries and Core
  Values, without judging the user's character.
* The Coach Digest cites relevant Journal Entries and asks an open reflective
  question without prescribing action.
* Low-friction journaling: prompts reduce blank-page paralysis and encourage regular reflection.
* Evidence-based reflection, not gamification: No Active Drift does not prove
  alignment or improvement. The Coach Digest can offer warm encouragement only
  from cited Journal Entries, without streaks, points, leaderboards, generic
  praise, or unsupported positive claims.

## **Onboarding (SVBWS Values Assessment)** 🧪

The onboarding flow uses the published **Schwartz Values Best-Worst Survey (SVBWS)** design to elicit relative priorities. The mobile-first onboarding phase of the shared React app presents 11 randomized groups of six neutral cards, followed directly by a label-free Core Value summary and first Journal Entry handoff. Universalism–Nature and Universalism–Social remain distinct in the raw BWS result and are merged only in the separately named ten-value Profile transformation. The flow has no midpoint result or confidence proxy. It is a research-grounded pilot instrument, not a psychometrically validated Twinkl instrument. The **[Onboarding Spec](onboarding/onboarding_spec.md)** is authoritative for the interaction, scoring, and Profile contracts.

The internal Profile includes a graded 10-dimensional product weight vector,
with Core Values stored in `top_values` for Drift gating. The shared React app
keeps that Profile in the browser without exposing technical JSON to the user
and synchronizes the confirmed Profile and browser-held Experience state with
the in-memory Python boundary. A separate host can persist the Profile exposed
by the callback or browser event. The batch runtime imports saved Profile JSON
with `--profile-path` and uses `top_values` as Core Values. Durable multi-user
storage is outside the capstone.

This onboarding directly anchors the capstone submodules: the latent dimensions form named slots in the knowledge base and rule layer (**Intelligent Reasoning Systems**), the mapping from card responses to those dimensions is a compact modelling task (**Pattern Recognition Systems**), entry content analysis and temporal patterns feed the sensing layer (**Intelligent Sensing Systems**), and treating the assessment as one input stream into a shared user-state vector `z` illustrates end-to-end orchestration and state management across Perception → Memory → Reasoning → Action (**Architecting AI Systems**).

## **Core Feature Modules**

* **Weekly Drift Detection** ✅: Review end-of-week Journal Entries, apply the Drift rule, and store structured output with cited evidence (Pattern Recognition + Reasoning).
* **Coach Digest** ⚠️: Supply Weekly Drift Detection output to a prompt, then produce the user response (Reasoning).
* **Conversational introspection agent** ✅: The displayed nudge interaction is
  complete for the capstone POC. It uses a three-category **nudge taxonomy**:
  - **Clarification** — for vague entries lacking concrete details
  - **Elaboration** — for surface-level entries with unexplored depth
  - **Tension-surfacing** — for hedging language or conflicted statements

  After the deterministic anti-annoyance check, the Experience runtime uses
  one `gpt-5.6-luna` reasoning-effort-`none` structured call to return
  `no_nudge`, clarification, elaboration, or tension-surfacing plus one
  contextual question when applicable. The historical synthetic workflow
  retains its separate decision and generation calls for reproducibility.
  Anti-annoyance logic caps nudges at 2 per 3-entry window. See
  [pipeline_specs.md](pipeline/pipeline_specs.md) for implementation details.

  A displayed nudge prevents the journaling loop from becoming write-only
  until the next Coach Digest. It gives the user an immediate response and a
  reason to continue the Journal Entry. The capstone does not require a
  displayed nudge to improve VIF Critic (Offline) training data or Weekly Drift Reviewer
  Decisions. A future external pilot can measure whether users respond,
  continue journaling, and find the question relevant.
* **”Map of Me”** ❌: Embed each Journal Entry, visualise trajectories, and overlay VIF Critic Predictions (Pattern Recognition + Intelligent Sensing).
* **Journaling anomaly radar** ❌: After 2–3 weeks of entries establish cadence baselines, a lightweight time-series/anomaly detector tracks check-in gaps, flags “silent weeks,” cites evidence windows, and triggers empathetic nudges (Pattern Recognition + Architecting).
* **Goal-aligned inspiration feed** ❌: When the profile shows intent (e.g., “pick up Japanese”) but no supporting activities, call a real-time search API (SerpAPI/Tavily) constrained by what the user enjoys (e.g., highly rated anime) and reason over the results before surfacing next-step suggestions (Intelligent Reasoning + Intelligent Sensing). Each curated option is presented as an explicit choice; the user’s accept/decline actions feed back into the values/identity graph so future nudges learn which media or effort types actually motivate them.

**Implementation path**

1. Frame the original research question (“How do we sustain a dynamic model of
   values and identity and reflect alignment?”) and map components to
   submodules. The capstone POC narrows this question to a confirmed Profile and
   longitudinal evidence. Profile evolution is future work.
2. Define the MVP loop: onboarding (SVBWS values assessment — see [spec](onboarding/onboarding_spec.md))
3. **Scoping Strategy:** Adopt a **Hybrid Approach** (simple journaling loop + Weekly Drift Detection + Coach Digest + lightweight trajectory visualization). Build small slices of each feature to demonstrate breadth without over-building.
4. Specify the profile schema:
   * **Value dimensions** anchored in [Schwartz's theory of basic human values](https://en.wikipedia.org/wiki/Theory_of_basic_human_values) (e.g., Self-Direction, Benevolence, Achievement, Security) with definitions, rubrics, and examples.
   * **User value profile:** vector of value weights `w_u ∈ ℝ^K` (normalized, sum to 1), Core Values stored in `top_values`, plus narrative descriptions and constraints. The full vector conditions the VIF Critic (Offline); `top_values` gates Drift v1.
   * **State representation:** the current Journal Entry embedding plus a
     10-dimensional value-weight vector. Configurable legal-history windows
     remain experimental; no label-derived history features are allowed at
     inference time.
5. Implement **[LLM-Judge labeling and VIF Critic (Offline) training](vif/03_model_training.md):** For each Journal Entry, the LLM-Judge creates per-dimension categorical LLM-Judge VIF Labels in `{-1, 0, +1}` with rationales. Use synthetic personas for initial training and validation.

   > **Status:** Steps 1-5 complete (204 personas, 1,651 labeled Journal Entries). Human annotation tool is operational with 380 saved annotations, including the current 115-entry shared subset used for inter-rater agreement. Multiple VIF Critic (Offline) architectures have been evaluated (ordinal MLP heads, BNN, TCN). See [Implementation Status](#implementation-status) for current progress. Step 6 (additional lightweight classifiers) is not planned for the time-boxed capstone.

6. Tooling: the time-boxed capstone does not add another lightweight
   classifier. The completed VIF Critic (Offline) remains offline research.
7. Evaluation plan: collect optional Coach Digest perceived-accuracy feedback
   without changing the fixed Profile. A future external pilot can add broader
   user measures.
8. The inspiration feed and its recommendation feedback loop are future work.

| Component | Traditional Journaling (Summarizer) | Twinkl (Evidence-grounded Drift reflection) |
| :--- | :--- | :--- |
| **Process** | **1. Tagging:** Identifies sentiment and topics.<br>• Journal Entry 1: Negative, Work<br>• Journal Entry 2: Guilt, Health<br>• Journal Entry 3: Positive, Partner<br>**2. Aggregation:** Groups these tags together. | **1. Review:** The Weekly Drift Reviewer decides Conflict, Not Conflict, or Abstain for each Journal Entry and Core Value.<br>**2. Rule:** The Drift Detector creates Drift only from two consecutive Conflicts for the same Core Value.<br>**3. Response:** The Coach Digest cites the stored evidence and asks a non-prescriptive reflective question. |
| **Question it Answers** | **"What have I been feeling or talking about?"** | **"Does the available evidence show a repeated Conflict with one of my Core Values?"** |
| **Final Output** | A high-level summary of moods and topics. | Active Drift, No Active Drift, or Insufficient Evidence for each Core Value, with cited Journal Entries and a Coach Digest response when one is valid. |
| **Core Concept** | **Retrospective summarization** | **Evidence-grounded longitudinal accountability** |

**Twinkl’s edge**

* **Structured Profile:** Onboarding creates the confirmed Profile and Core
  Values. Ongoing journaling adds longitudinal evidence. Automatic Profile
  evolution and decay-aware updates are future work.
* **Weekly Drift Detection:** The Weekly Drift Reviewer and Drift Detector
  compare Journal Entries with Core Values and cite the evidence behind each
  current state.
* **Inspectable accountability:** Inspect exposes the exact Profile, Journal
  Entries, Weekly Drift Reviewer Decisions, Drift Detector transitions, Coach
  Digest responses, and provider receipts behind the displayed result.
* **Capstone-ready architecture:** Synthetic-data generation, LLM-Judge
  labeling, an uncertainty-aware MLP, Weekly Drift Detection, and a
  deterministic two-Conflict rule provide concrete work across Intelligent
  Sensing, Pattern Recognition, Reasoning, and Architecting AI Systems.

## Design Lessons Learned

### Metadata Leakage in Synthetic Data

A critical anti-pattern discovered during development: using synthetic generation instructions (e.g., `tone: Exhausted`, `reflection_mode: Neutral`) in decision logic creates train/serve skew. These labels exist only during data generation — they won't be available in production.

**Resolution:** All nudge decision logic uses only **observable content signals**:
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

### LLM-Judge vs VIF Critic (Offline) Context Windows

A key architectural decision: the LLM-Judge and VIF Critic (Offline) use different context windows.

| Component | Context | Rationale |
|-----------|---------|-----------|
| **LLM-Judge** | Persona context plus previous Journal Entries | Better labeling: trajectory context helps disambiguate vague Journal Entries like "feeling better" |
| **VIF Critic (Offline)** | Current Journal Entry plus normalized value profile (`window_size: 1`) | Fixed student-visible contract for fast local inference + MC Dropout |

**Why decouple?** The LLM-Judge runs offline during experimental training-data creation, while the user-facing path relies on the Weekly Drift Reviewer and Drift Detector. The frozen-holdout LLM baseline shows that adding previous Journal Entries improves the `human_context` setup's `recall_-1`, while the local MLP retains higher `recall_-1` and lower hedging. That result remains useful research evidence; it does not make the VIF Critic (Offline) a runtime dependency.

This avoids the trap of matching windows "for consistency" when the constraints are fundamentally different.

# Unplanned Stretch Goals

These ideas are outside the time-boxed capstone. No further VIF Critic (Offline) work
listed here is planned.

| Goal | Why it matters |
| :--- | :--- |
| **Neuro-symbolic reasoning** | Add a tiny knowledge graph + rule layer on top of LLM outputs to show which logical checks fired (great for XRAI storytelling). |
| **Multimodal fusion** | *Future work (out of scope for capstone):* Blend text + prosodic audio cues to extend Intelligent Sensing value beyond text-only analysis. |
| **Personalised quote recommender** | Build embeddings of quotes + user resonance to deliver “micro-anchors” tuned to each identity conflict. |
| **Advanced uncertainty modeling** | Extend MC Dropout with ensembles or density models; add explicit OOD detectors on the text embedding space. (See [Uncertainty Logic](vif/04_uncertainty_logic.md)) |
| **VIF Critic (Offline) extensions** | Keep the VIF Critic (Offline) as the current-Journal-Entry POC. Short-history inputs, calibrated prediction, and time-aware multimodal research remain outside the time-boxed capstone. See [VIF design](vif/01_concepts_and_roadmap.md). |

# Features that tie back to Masters' submodules

| Submodule                         | Features in Twinkl                                                                                                                                                                                                                  |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intelligent Reasoning Systems** | The confirmed Profile supplies Core Values. The Weekly Drift Reviewer decides Conflict, Not Conflict, or Abstain. The deterministic Drift Detector applies the two-consecutive-Conflict rule. The Coach Digest uses the stored result. |
| **Pattern Recognition Systems**   | The VIF Critic (Offline) compares Journal Entries with a ten-dimensional Profile. The completed research covers ordinal prediction, uncertainty, embeddings, hard-dimension analysis, and recall-first checkpoint selection. |
| **Intelligent Sensing Systems**   | The capstone uses text from Journal Entries, displayed nudges, and responses. Journal gap detection, real-time search input, and multimodal sensing are future work. |
| **Architecting AI Systems**       | The React app and Python service use versioned session, scenario, and trace contracts. Saved Persona replay, fail-closed provider handling, prompt boundaries, and Inspect show the current orchestration. Production multi-user storage and background schedules are outside scope. |



# Evaluation Strategy

**Purpose:** Evaluate the technical correctness of Weekly Drift Detection and
the Coach Digest, then keep future user-perceived usefulness separate from
AI-reviewed synthetic development evidence.

| # | Component | Purpose | Method | Example |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **LLM-Judge agreement** | Describe how consistently the LLM-Judge VIF Labels overlap with project-team annotation without treating that subset as independent human ground truth | Use the shared 115-Journal-Entry subset across 19 personas. Report human-human Fleiss' κ, each annotator's LLM-Judge-human Cohen's κ, the mean Cohen value, per-value results, prevalence sensitivity, and sample limitations. The coefficients use different rater structures and do not define a human-consistency ceiling. | A Journal Entry receives project-team and LLM-Judge VIF Labels for the same Schwartz values; the report shows agreement and exact disagreements rather than declaring either source correct by construction. |
| 2 | **VIF Critic (Offline) Conflict screening** | Record the completed local-model research without collapsing into neutral predictions | Report historical entry-level `recall_-1`, `-1` precision and precision-recall behavior alongside QWK, calibration, `+1` recall, and per-dimension diagnostics. | The completed research documents Conflict-recovery limits; it does not receive user-facing Drift authority, and no further VIF Critic (Offline) work is planned. |
| 3 | **Weekly Drift Detection** | Confirm that Weekly Drift Detection finds Drift for a Core Value | Evaluate the fixed `gpt-5.6-luna` reasoning-effort-`low` Weekly Drift Reviewer with the displayed-behavior target. Each of two consecutive Journal Entries must clearly show Conflict against the same Core Value. Apply the internal Drift Detector and assess the stored structured output. Prioritize Drift recall first and false Drift alerts second. Report coverage as a diagnostic. No fresh final test or deployment approval is claimed. | Drift: two consecutive Journal Entries both visibly show Conflict against Benevolence. An Abstain Weekly Drift Reviewer Decision produces no Drift. `+1` on another value cannot cancel the Drift. |
| 4 | **Coach Digest explanation quality** | Check evidence use, response tone, and perceived accuracy without treating AI review as human validation | Run Coach Digest Validations over saved responses. Run Coach Digest Evals for correctness, specificity, non-prescriptive tone, tension honesty, and the reflective question, with an independent evaluator provider when required. The deterministic Drift/control study compares known development Drifts with matched no-known-Drift controls. A future pilot asks 5–10 users whether each response felt accurate on a **5-point Likert scale**. | A saved response passes its evidence and language checks, receives source-labelled AI review scores, and remains separate from a future user's perceived-accuracy rating. |
| 5 | **Displayed nudge experience** | Check whether immediate interaction supports continued journaling | During a future external pilot, measure response rate, the rate of a later Journal Entry, and perceived relevance. Keep these user measures separate from AI-reviewed synthetic evidence. | The user receives a contextual question, responds, and writes another Journal Entry before the Coach Digest. |

## Operational & User Success Metrics

| Category | Metrics |
| :--- | :--- |
| **User impact** | Likert ratings on "helps me act in line with values," displayed nudge response rate, continued journaling, % of suggested weekly experiments attempted, and retention over a 1–2 week pilot. |
| **Application and safety** | Latency from entry → feedback, LLM failure rates, data-notice acknowledgement, confirmed session deletion, provider boundaries, and qualitative review of the non-therapy message. |

**Validation approach:** Mini user study (5–10 people over 1–2 weeks) focusing on "felt accuracy" plus synthetic stress tests for technical correctness.

# Related Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](../CLAUDE.md) | Claude-specific repository policy that mirrors `AGENTS.md` |
| **Pipeline** | |
| [pipeline_specs.md](pipeline/pipeline_specs.md) | Synthetic data workflow design and rationale |
| [claude_gen_instructions.md](pipeline/claude_gen_instructions.md) | Parallel subagent generation workflow |
| [claude_judge_instructions.md](pipeline/claude_judge_instructions.md) | Historical LLM-Judge labeling workflow (wrangling + scoring) |
| [judge_reachability_audit_instructions.md](pipeline/judge_reachability_audit_instructions.md) | LLM-agnostic workflow for the twinkl-747 reachability audit |
| [annotation_guidelines.md](pipeline/annotation_guidelines.md) | Historical human annotation guide for displayed nudge scorability research |
| [annotation_tool_plan.md](pipeline/annotation_tool_plan.md) | Shiny annotation tool implementation plan |
| [nudge_design_rationale.md](pipeline/nudge_design_rationale.md) | Displayed nudge product design rationale and future pilot measures |
| **VIF** | |
| [01_concepts_and_roadmap.md](vif/01_concepts_and_roadmap.md) | Value Identity Function theory |
| [02_system_architecture.md](vif/02_system_architecture.md) | System architecture, state, and runtime flow |
| [03_model_training.md](vif/03_model_training.md) | LLM-Judge labeling and VIF Critic (Offline) training |
| [04_uncertainty_logic.md](vif/04_uncertainty_logic.md) | VIF Critic (Offline) uncertainty and Drift review logic |
| [05_capstone_scope_decision.md](vif/05_capstone_scope_decision.md) | Adopted VIF capstone scope, metric hierarchy, and deferred decisions |
| [example.md](vif/example.md) | Worked end-to-end VIF behavior example |
| **Evals** | |
| [evals/overview.md](evals/overview.md) | Evaluation workflow overview |
| [evals/coach_narrative_test_and_eval_guide.md](evals/coach_narrative_test_and_eval_guide.md) | Coach Digest Validations, Coach Digest Evals, and Drift/control study runbook |
| [evals/judge_validation_summary.md](evals/judge_validation_summary.md) | LLM-Judge validation results |
| [drift/trajectory_eda.md](drift/trajectory_eda.md) | Historical empirical basis for the Drift definition |
| [evals/drift_detection_eval.md](evals/drift_detection_eval.md) | Drift Detector target and evaluation protocol |
| [evals/drift_v1_student_visible_target.md](evals/drift_v1_student_visible_target.md) | Historical five-episode development review and withheld former final-test score |
| [twinkl-752.4 full review](../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) | Reviewed cohort and 33-episode union correction |
| [twinkl-752.5 Opus label resolution](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md) | Four resolved Conflict labels and the 106/106-resolved development union |
| [twinkl-752.5 reassessment](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md) | Raw VIF Critic Predictions, scheduling, trigger placement, and subgroup results |
| [twinkl-qtwz complete development review](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md) | Complete 292-case development labels and 42-Drift contract |
| [twinkl-52zz Luna reasoning-effort comparison](../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md) | Evidence behind the fixed Luna-low model contract, metric hierarchy, cost, and limitations |
| [twinkl-ck3w Luna higher-reasoning comparison](../logs/experiments/reports/experiment_review_2026-08-09_twinkl_ck3w_luna_higher_reasoning.md) | Medium, high, and xhigh development results and the no-change Luna-low decision |
| **Other** | |
| [architecture/e2e_architecture.md](architecture/e2e_architecture.md) | High-level product and system map |
| [weekly/weekly_drift_detection.md](weekly/weekly_drift_detection.md) | Weekly Drift Detection and Coach Digest contracts, runtime commands, and generated files |
| [demo/weekly_drift_review_app.md](demo/weekly_drift_review_app.md) | Read-only comparison of frozen Weekly Drift Reviewer development Runs, including the fixed Luna-low setup |
| [demo/review_app.md](demo/review_app.md) | Historical Runtime Demo Review App for the local VIF Critic (Offline) compatibility path |
| [demo/experience_inspect_app.md](demo/experience_inspect_app.md) | Specified React capstone demo with synchronized Experience and Inspect views |
| [01_value_evolution.md](evolution/01_value_evolution.md) | Concept note for a possible future filter distinguishing value evolution from Drift |
| [onboarding_spec.md](onboarding/onboarding_spec.md) | BWS-based onboarding flow, item design, and data output schema |
| [capstone_report/capstone_project_report.md](capstone_report/capstone_project_report.md) | Maintained Phase 2 Technical Paper source |
| [capstone_report/capstone_project_report.pdf](capstone_report/capstone_project_report.pdf) | Rendered Phase 2 Technical Paper |
| [April 2026 proposal submission](archive/capstone/2026-04-proposal-submission/) | Immutable snapshot of the already-submitted proposal, slides, figures, and sources |
