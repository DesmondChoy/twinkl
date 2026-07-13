# Value Identity Function (VIF) – Concepts & Roadmap

This document outlines the high-level objectives, core concepts, and implementation roadmap for Twinkl’s **Value Identity Function (VIF)**. The VIF is the core evaluative engine that estimates alignment between a user's behavior and their long-term values.

For technical details, see:
*   [System Architecture, State, and Runtime Flow](02_system_architecture.md)
*   [Reward Modeling & Training](03_model_training.md)
*   [Uncertainty and Drift Detector Logic](04_uncertainty_logic.md)
*   [Capstone Scope and Evaluation Decision](05_capstone_scope_decision.md)

---

## What is the VIF?

Think of the **Value Identity Function (VIF)** as Twinkl's internal "compass."

While the user journals and interacts with the app, the VIF quietly observes their behavior over time. It compares what the user *does* (their daily actions and struggles) against what they *value* (their long-term identity and goals).

Instead of giving a generic sentiment score, the VIF tracks the ten Schwartz value dimensions: Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, and Universalism. It answers the question: *"Is the user moving towards the person they want to be, or drifting away?"*

Crucially, the VIF is designed to be:
*   **Nuanced:** It acknowledges trade-offs (e.g., "You crushed your work goals this week, but your sleep suffered").
*   **Cautious:** It knows when it's unsure. If the user's situation is complex or new, the VIF holds back its judgment rather than giving bad advice.
*   **Time-Aware:** The downstream timeline looks for repeated evidence rather than reacting to one Journal Entry. Drift v1 requires two consecutive Conflicts for the same Core Value.

This engine supports Twinkl's feedback: the Drift Detector flags Drift so the Weekly Coach can gently surface tensions, while sustained alignment can support occasional evidence-based acknowledgment.

---

## 1. Objectives and Design Principles

### 1.1 Objectives

The VIF is designed to:

* Estimate **per-value-dimension alignment** for a user at a given point in time.
* Detect **emerging misalignment trajectories** early (before long-term regret).
* Support **explainable feedback**: not just “good/bad,” but “which parts of your life are drifting and why.”
* Operate safely on **real user data**, where no ground-truth reward labels exist.

### 1.2 Key Design Principles

1. **Reward modeling, not ground-truth reward**
   Twinkl does not assume access to ground-truth alignment labels. Instead, the
   **LLM-Judge** creates reference labels from Journal Entry text and user
   profiles.

2. **Vector-valued evaluation**
   Alignment is evaluated across all ten Schwartz value dimensions. The value function remains **vector-valued** to preserve tensions and trade-offs, and is only aggregated when needed.

   For the remaining capstone scope, the VIF Critic's primary operational role
   is Conflict screening. Per-Journal-Entry `recall_-1` drives model development;
   QWK and positive alignment remain diagnostics and non-gating Weekly Coach
   context.

3. **Uncertainty-aware feedback**
   The VIF Critic estimates **epistemic uncertainty** in its predictions, and Twinkl only issues feedback when the prediction is both:
   * Confident in its judgment, and
   * Detecting a significant pattern (negative or positive).

4. **Trajectory-aware downstream evaluation**
   The live VIF Critic default uses `window_size: 1`, so each prediction sees the current Journal Entry, including its displayed nudge and response when present, plus the normalized value profile. Runtime timeline reconstruction and the Drift Detector provide the temporal layer. `twinkl-749` tested a small prior-Journal-Entry mean summary, but its seed-11 results regressed and did not receive deployment approval. History support therefore remains diagnostic rather than an assumed property of the default VIF Critic.

5. **Separation of concerns: VIF Critic vs Weekly Coach**
   The VIF Critic produces numeric per-Journal-Entry alignment evidence and
   uncertainty
   from the configured student-visible state. The downstream timeline supplies
   temporal evaluation. A separate **Weekly Coach** reads the
   user's full journal history via **full-context prompting** (at POC scale, all
   Journal Entries fit in the LLM context window) to surface thematic evidence
   after the VIF Critic and Drift Detector produce structured signals. At
   production scale
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

These dimensions form an ontology for both the LLM-Judge and the VIF Critic.

### 2.2 User Value Profile

Each user ($u$) has a **value profile**:

* A vector of value weights:
  * $w_{u,t} \in \mathbb{R}^K$, with $w_{u,t} \ge 0$ and $\sum_k w_{u,t,k} = 1$.
  * The synthetic runtime assigns equal mass to Core Values and falls
    back to a uniform vector if none match. The graded BWS output is specified
    but is not wired into runtime state construction.
  * The profile is piecewise constant in the current POC. [Value Evolution
    Detection](../evolution/01_value_evolution.md) remains a future product
    decision even though an experimental classifier exists in the prototype
    router.
* Additional profile information:
  * Narrative descriptions of what each value means to them.
  * Known constraints and long-term goals.

This profile is used both in the **LLM-Judge prompts** and in aggregating vector-valued outputs for summaries.

---

## 3. Implementation Roadmap

To make this design capstone-friendly, we summarise a recommended tiered approach. The team can choose which tier to implement while keeping a coherent long-term architecture.

* **Tier 1 (Current POC)**
  * State: current Journal Entry embedding, including any displayed nudge and
    response, plus the normalized profile.
  * Target: immediate alignment (Option A).
  * VIF Critic: ordinal MLP with a BNN comparison baseline.
  * Adopted evaluation scope: recover LLM-Judge Conflict (`-1`) labels that
    support Drift detection;
    preserve the ternary ten-value output without claiming equal reliability
    across dimensions.
  * Uncertainty: MC Dropout.
  * Drift target: two consecutive Journal Entries that each show a Conflict
    for the same Core Value.
  * Student-visible Drift target: [`twinkl-v8pb`](../evals/drift_v1_student_visible_target.md)
    completed a full-runtime-text development-set review and final-test-set
    review. `run_020` found 1/5 development-set Drifts, and one final-test-set
    case containing 19 Journal Entries remained unresolved, so no final test
    score was run.
    The old
    consensus-derived frozen benchmark is retired historical evidence, not a
    runnable benchmark or final test set.

* **Tier 2 (Optional capstone extension)**
  * Evaluated diagnostics: compact mean history (`twinkl-749`) and soft
    vote-distribution labels (`twinkl-j0ck`) both completed without deployment
    approval.
    Neither changes the Tier 1 state or target default.
  * State or target extensions should reopen only with a materially different,
    evidence-backed mechanism and a matching student-visible target contract.
  * VIF Critic: calibrated local MLP, `gpt-5.4-mini` per-Journal-Entry
    fallback, or an MLP-to-LLM cascade.
  * Drift rule: the same Drift definition with calibrated
    thresholds for Core Values and active, recovered, mixed, or
    uncertain weekly wording.

* **Tier 3 (Out of Scope for Capstone)**
  * State: multimodal, sliding-window state with audio/physio.
  * Target: time-aware discounted returns (Option C).
  * Additional: offline RL for suggestion policies, more advanced uncertainty and personalization.
  * *Note: Audio/prosodic modalities deferred to future work.*

---

## 4. Extensions and Future Work

Potential extensions beyond the POC:

* **LLM-Judge Distillation**:
  * Train a smaller supervised model to mimic the LLM-Judge, reducing latency and cost.
* **Policy Learning**:
  * After VIF is stable and safety mechanisms are validated, explore offline RL to learn and evaluate **action suggestions** (micro-experiments, habits). This remains out of scope for the core capstone POC.
* **Richer Uncertainty Modeling**:
  * Incorporate ensembles, density models, or explicit OOD detectors on the text embedding space.
* **More Modalities** *(Out of scope for capstone)*:
  * Incorporate prosodic and physiological features robustly, especially for early warning signals of stress or overload.
* **Value Evolution Detection**: Possible statistical filter for user-confirmed profile updates after the Drift path is validated. See [Value Evolution Detection](../evolution/01_value_evolution.md).
* **Personalisation Layers**:
  * Explore global VIF plus lightweight per-user adapters for users whose trajectories systematically diverge from other users.
* **Retrieval-Augmented Weekly Coach (scaling)**:
  * At POC scale (8–12 Journal Entries per persona), all Journal Entries fit in the LLM context window, so the Weekly Coach uses full-context prompting. For production deployment with longer user histories (50+ Journal Entries), the Weekly Coach would transition to RAG with a vector store for semantic similarity retrieval over the journal corpus. This is a scaling concern, not a capability gap at current data volumes.
