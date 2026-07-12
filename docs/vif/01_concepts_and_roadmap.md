# Value Identity Function (VIF) – Concepts & Roadmap

This document outlines the high-level objectives, core concepts, and implementation roadmap for Twinkl’s **Value Identity Function (VIF)**. The VIF is the core evaluative engine that estimates alignment between a user's behavior and their long-term values.

For technical details, see:
*   [System Architecture, State, and Runtime Flow](02_system_architecture.md)
*   [Reward Modeling & Training](03_model_training.md)
*   [Uncertainty, Drift, and Trigger Logic](04_uncertainty_logic.md)
*   [Capstone Scope and Evaluation Decision](05_capstone_scope_decision.md)

---

## What is the VIF?

Think of the **Value Identity Function (VIF)** as Twinkl's internal "compass."

While the user journals and interacts with the app, the VIF quietly observes their behavior over time. It compares what the user *does* (their daily actions and struggles) against what they *value* (their long-term identity and goals).

Instead of giving a generic sentiment score, the VIF tracks the ten Schwartz value dimensions: Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, and Universalism. It answers the question: *"Is the user moving towards the person they want to be, or drifting away?"*

Crucially, the VIF is designed to be:
*   **Nuanced:** It acknowledges trade-offs (e.g., "You crushed your work goals this week, but your sleep suffered").
*   **Cautious:** It knows when it's unsure. If the user's situation is complex or new, the VIF holds back its judgment rather than giving bad advice.
*   **Time-Aware:** The downstream timeline looks for repeated evidence rather than reacting to a single entry. Drift v1 requires sustained conflict on a declared core value.

This engine powers Twinkl's feedback system: flagging drifts so the Coach (the conversational AI) can gently surface tensions, and recognizing sustained alignment so the Coach can offer occasional evidence-based acknowledgment.

---

## 1. Objectives and Design Principles

### 1.1 Objectives

The VIF is designed to:

* Estimate **per-value-dimension alignment** for a user at a given point in time.
* Detect **emerging misalignment trajectories** early (before long-term regret).
* Support **explainable feedback**: not just “good/bad,” but “which parts of your life are drifting and why.”
* Operate safely on **real user data**, where no ground-truth reward labels exist.

### 1.2 Key Design Principles

1. **Reward modeling, not oracle reward**
   The system does not assume access to ground-truth alignment labels. Instead, it uses an explicit **Reward Model (RM)** to infer alignment scores from text and user profiles.

2. **Vector-valued evaluation**
   Alignment is evaluated across all ten Schwartz value dimensions. The value function remains **vector-valued** to preserve tensions and trade-offs, and is only aggregated when needed.

   For the remaining capstone scope, the Critic's primary operational role is
   conflict screening. Entry-level `recall_-1` drives model development; QWK
   and positive alignment remain diagnostics and non-gating Coach context.

3. **Uncertainty-aware feedback**
   The system estimates **epistemic uncertainty** in its predictions and only issues feedback when it is both:
   * Confident in its judgment, and
   * Detecting a significant pattern (negative or positive).

4. **Trajectory-aware downstream evaluation**
   The live Critic default uses `window_size: 1`, so each prediction sees the current journal session and normalized value profile. Runtime timeline reconstruction and drift detection provide the temporal layer. `twinkl-749` tested a small prior-entry mean summary, but its seed-11 package regressed and was not promoted. History support therefore remains diagnostic rather than an assumed property of the default Critic.

5. **Separation of concerns: Critic vs Coach**
   The VIF Critic produces numeric per-entry alignment evidence and uncertainty
   from the configured student-visible state. The downstream timeline supplies
   temporal evaluation. A separate **Coach / Explanation layer** reads the
   user's full journal history via **full-context prompting** (at POC scale, all
   entries fit in the LLM context window) to surface thematic evidence after
   the Critic and drift layer produce structured signals. At production scale
   with longer histories, this would transition to retrieval-augmented
   generation (RAG) — see [Section 4](#4-extensions-and-future-work).

---

## 2. Value Dimensions and User Profile

### 2.1 Value Dimensions

We define a set of $K$ value dimensions:

* The canonical dimensions are Self-Direction, Stimulation, Hedonism,
  Achievement, Power, Security, Conformity, Tradition, Benevolence, and
  Universalism.
* Each value dimension has:
  * A **definition** in natural language.
  * Positive and negative **examples**.
  * A **rating rubric** (e.g. from “misaligned” to “aligned”).

These dimensions form an ontology for both the reward model and the critic.

### 2.2 User Value Profile

Each user ($u$) has a **value profile**:

* A vector of value weights:
  * $w_{u,t} \in \mathbb{R}^K$, with $w_{u,t} \ge 0$ and $\sum_k w_{u,t,k} = 1$.
  * The synthetic runtime assigns equal mass to declared core values and falls
    back to a uniform vector if none match. The graded BWS output is specified
    but is not wired into runtime state construction.
  * The profile is piecewise constant in the current POC. [Value Evolution
    Detection](../evolution/01_value_evolution.md) remains a future product
    decision even though an experimental classifier exists in the prototype
    router.
* Additional profile information:
  * Narrative descriptions of what each value means to them.
  * Known constraints and long-term goals.

This profile is used both in the **Reward Model prompts** and in aggregating vector-valued outputs for summaries.

---

## 3. Implementation Roadmap

To make this design capstone-friendly, we summarise a recommended tiered approach. The team can choose which tier to implement while keeping a coherent long-term architecture.

* **Tier 1 (Current POC)**
  * State: current journal-session embedding + normalized profile.
  * Target: immediate alignment (Option A).
  * Critic: ordinal MLP with a BNN comparison baseline.
  * Adopted evaluation scope: recover `-1` evidence for sustained conflict;
    preserve the ternary ten-value output without claiming equal reliability
    across dimensions.
  * Uncertainty: MC Dropout.
  * Drift target: two adjacent entries that each visibly show a behavior or
    choice against the same declared core value.
  * Student-visible drift target: [`twinkl-v8pb`](../evals/drift_v1_student_visible_target.md)
    completed a full-runtime-text development target and locked promotion
    review. `run_020` found 1/5 development episodes, and one 19-entry
    promotion case remained unresolved, so no promotion score was run. The old
    consensus-derived frozen benchmark is retired historical evidence, not a
    runnable benchmark or promotion surface.

* **Tier 2 (Optional capstone extension)**
  * Evaluated diagnostics: compact mean history (`twinkl-749`) and soft
    vote-distribution labels (`twinkl-j0ck`) both completed without promotion.
    Neither changes the Tier 1 state or target default.
  * State or target extensions should reopen only with a materially different,
    evidence-backed mechanism and a matching student-visible target contract.
  * Critic: calibrated local MLP, LLM teacher/fallback, or a measured cascade.
  * Drift rule: the same sustained-conflict construct with calibrated
    thresholds for declared core values and active, recovered, mixed, or
    uncertain weekly wording.

* **Tier 3 (Out of Scope for Capstone)**
  * State: multimodal, sliding-window state with audio/physio.
  * Target: time-aware discounted returns (Option C).
  * Additional: offline RL for suggestion policies, more advanced uncertainty and personalization.
  * *Note: Audio/prosodic modalities deferred to future work.*

---

## 4. Extensions and Future Work

Potential extensions beyond the POC:

* **Distilled Reward Model**:
  * Train a smaller supervised model to mimic the LLM-as-Judge, reducing latency and cost.
* **Policy Learning**:
  * After VIF is stable and safety mechanisms are validated, explore offline RL to learn and evaluate **action suggestions** (micro-experiments, habits). This remains out of scope for the core capstone POC.
* **Richer Uncertainty Modeling**:
  * Incorporate ensembles, density models, or explicit OOD detectors on the text embedding space.
* **More Modalities** *(Out of scope for capstone)*:
  * Incorporate prosodic and physiological features robustly, especially for early warning signals of stress or overload.
* **Value Evolution Detection**: Possible statistical filter for user-confirmed profile updates after the sustained-conflict path is validated. See [Value Evolution Detection](../evolution/01_value_evolution.md).
* **Personalisation Layers**:
  * Explore global VIF plus lightweight per-user adapters for users whose trajectories systematically diverge from the population.
* **Retrieval-Augmented Coach (scaling)**:
  * At POC scale (8–12 entries per persona), all journal entries fit in the LLM context window, so the Coach uses full-context prompting — passing all entries directly in the prompt. For production deployment with longer user histories (50+ entries), the Coach would transition to RAG with a vector store for semantic similarity retrieval over the journal corpus. This is a scaling concern, not a capability gap at current data volumes.
