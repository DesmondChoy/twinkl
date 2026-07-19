# Value Identity Function (VIF) – Concepts & Roadmap

This document preserves the high-level objectives, concepts, and research
roadmap for Twinkl's **Value Identity Function (VIF)**. The trained VIF Critic
is optional experimental research. The current user-facing decision path uses
the Weekly Drift Reviewer and Drift Detector without VIF Critic Predictions.

For technical details, see:
*   [System Architecture, State, and Runtime Flow](02_system_architecture.md)
*   [Reward Modeling & Training](03_model_training.md)
*   [Uncertainty and Drift Review Logic](04_uncertainty_logic.md)
*   [Capstone Scope and Evaluation Decision](05_capstone_scope_decision.md)

---

## What is the VIF?

Think of the **Value Identity Function (VIF)** as Twinkl's original internal
"compass" concept, not the current runtime contract.

The concept compares what a user *does* in Journal Entries against their Core
Values. The implemented user-facing path now performs that comparison through
the Weekly Drift Reviewer and Drift Detector.

Instead of giving a generic sentiment score, the VIF tracks the ten Schwartz value dimensions: Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, and Universalism. It answers the question: *"Is the user moving towards the person they want to be, or drifting away?"*

Crucially, the VIF is designed to be:
*   **Nuanced:** It acknowledges trade-offs (e.g., "You crushed your work goals this week, but your sleep suffered").
*   **Cautious:** It knows when it's unsure. If the user's situation is complex or new, the VIF holds back its judgment rather than giving bad advice.
*   **Time-Aware:** The downstream timeline looks for repeated evidence rather than reacting to one Journal Entry. Drift v1 requires two consecutive Conflicts for the same Core Value.

In the approved user-facing path, the Weekly Drift Reviewer is fixed at
`gpt-5.6-luna` with reasoning effort `low`. It and the Drift Detector decide
Drift so the Weekly Coach can surface tensions. The trained VIF Critic is an
optional experimental model, not a dependency of that path.

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

   If optional VIF Critic research resumes, its primary evaluation role is
   Conflict screening. Per-Journal-Entry `recall_-1` drives model development;
   QWK and positive alignment remain diagnostics. VIF Critic results are not
   Weekly Coach inputs.

3. **Uncertainty-aware review**
   The VIF Critic estimates **epistemic uncertainty** in its predictions.
   Existing experiment outputs store uncertainty for analysis. A generalized
   independent-review and retraining loop is optional P3 work. Uncertainty does
   not authorize user-facing feedback.

4. **Trajectory-aware downstream evaluation**
   The live VIF Critic default uses `window_size: 1`, so each prediction sees the current Journal Entry, including its displayed nudge and response when present, plus the normalized value profile. The fixed Luna-low Weekly Drift Reviewer and deterministic Drift Detector provide the approved temporal decision layer. `twinkl-749` tested a small prior-Journal-Entry mean summary, but its seed-11 results regressed and did not receive deployment approval. History support therefore remains diagnostic rather than an assumed property of the default VIF Critic.

5. **Separation of concerns: VIF Critic vs Weekly Coach**
   The VIF Critic produces numeric per-Journal-Entry alignment evidence and
   uncertainty
   from the configured student-visible state. The downstream timeline supplies
   temporal evaluation. A separate **Weekly Coach** reads the
   user's full journal history via **full-context prompting** (at POC scale, all
   Journal Entries fit in the LLM context window) to surface thematic evidence
   after the Weekly Drift Reviewer and Drift Detector produce structured
   signals. The VIF Critic is not required in this path. At production scale
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
    completed a full-runtime-text development review and correctly withheld its
    former final-test score. That population is now development-only. The
    known-development union contains 33 Drifts across 28 Drift trajectories,
    but no fresh final test exists. The old consensus-derived frozen benchmark
    is retired historical evidence, not a runnable benchmark or final test set.

* **Tier 2 (Optional capstone extension)**
  * Evaluated diagnostics: compact mean history (`twinkl-749`) and soft
    vote-distribution labels (`twinkl-j0ck`) both completed without deployment
    approval.
    Neither changes the Tier 1 state or target default.
  * State or target extensions should reopen only with a materially different,
    evidence-backed mechanism and a matching student-visible target contract.
  * VIF Critic: calibrated local MLP with versioned prediction, uncertainty,
    independent review, and retraining.
  * VIF Critic candidate confirmation is outside the remaining capstone scope.
    Raw prompt input, confidence-only fallback, and direct VIF Critic Drift
    decisions remain unapproved.
  * Drift rule: the same deterministic definition for Core Values, with active,
    recovered, mixed, or uncertain weekly wording.

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
