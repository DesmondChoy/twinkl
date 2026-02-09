# Value Identity Function (VIF) – Concepts & Roadmap

This document outlines the high-level objectives, core concepts, and implementation roadmap for Twinkl’s **Value Identity Function (VIF)**. The VIF is the core evaluative engine that estimates alignment between a user's behavior and their long-term values.

For technical details, see:
*   [System Architecture & Inference](02_system_architecture.md)
*   [Reward Modeling & Training](03_model_training.md)
*   [Uncertainty & Critique Logic](04_uncertainty_logic.md)

---

## What is the VIF?

Think of the **Value Identity Function (VIF)** as Twinkl's internal "compass."

While the user journals and interacts with the app, the VIF quietly observes their behavior over time. It compares what the user *does* (their daily actions and struggles) against what they *value* (their long-term identity and goals).

Instead of just giving a generic "sentiment score," the VIF tracks multiple dimensions of life simultaneously—like Health, Career, and Relationships. It answers the question: *"Is the user moving towards the person they want to be, or drifting away?"*

Crucially, the VIF is designed to be:
*   **Nuanced:** It acknowledges trade-offs (e.g., "You crushed your work goals this week, but your sleep suffered").
*   **Cautious:** It knows when it's unsure. If the user's situation is complex or new, the VIF holds back its judgment rather than giving bad advice.
*   **Time-Aware:** It looks for patterns, not just snapshots. One bad day isn't a crisis, but a three-week slide is.

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
   Alignment is evaluated in **multiple value dimensions** (e.g. Health, Relationships, Growth, Contribution). The value function remains **vector-valued** to preserve tensions and trade-offs, and is only aggregated when needed.

3. **Uncertainty-aware feedback**
   The system estimates **epistemic uncertainty** in its predictions and only issues feedback when it is both:
   * Confident in its judgment, and
   * Detecting a significant pattern (negative or positive).

4. **Trajectory-aware, not purely Markovian**
   The VIF explicitly incorporates **recent temporal history** (sliding windows and simple statistics) instead of assuming that a single entry and static profile fully determine future trajectories.

5. **Separation of concerns: Critic vs Coach**
   The VIF (critic) focuses on **temporal, numeric evaluation** using sequential history. A separate **Coach / Explanation layer** uses retrieval-augmented generation (RAG) to surface thematic evidence and narratives *after* the critic has produced its scores.

---

## 2. Value Dimensions and User Profile

### 2.1 Value Dimensions

We define a set of $K$ value dimensions:

* Example: Health, Relationships, Growth, Contribution, Autonomy, etc.
* Each value dimension has:
  * A **definition** in natural language.
  * Positive and negative **examples**.
  * A **rating rubric** (e.g. from “misaligned” to “aligned”).

These dimensions form an ontology for both the reward model and the critic.

### 2.2 User Value Profile

Each user ($u$) has a **value profile**:

* A vector of value weights:
  * $w_{u,t} \in \mathbb{R}^K$, with $w_{u,t} \ge 0$ and $\sum_k w_{u,t,k} = 1$.
  * For the POC, $w_{u,t}$ may be treated as **piecewise constant** over time (updated infrequently), but conceptually it can evolve slowly as the user refines their priorities.
* Additional profile information:
  * Narrative descriptions of what each value means to them.
  * Known constraints and long-term goals.

This profile is used both in the **Reward Model prompts** and in aggregating vector-valued outputs for summaries.

---

## 3. Implementation Roadmap

To make this design capstone-friendly, we summarise a recommended tiered approach. The team can choose which tier to implement while keeping a coherent long-term architecture.

* **Tier 1 (Minimum POC)**
  * State: current text embedding + profile (optional small history window).
  * Target: immediate alignment (Option A).
  * Critic: MLP regressor.
  * Uncertainty: MC Dropout.
  * Critique rule: dual-trigger (crash + rut) on simple rolling averages.

* **Tier 2 (Enhanced POC)**
  * State: sliding window over last $N$ entries with history stats.
  * Target: short-horizon average (Option B).
  * Critic: MLP with tuned hyperparameters and calibrated uncertainty.
  * Critique rule: weekly crash/rut logic with user-specific thresholds.

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
* **Personalisation Layers**:
  * Explore global VIF plus lightweight per-user adapters for users whose trajectories systematically diverge from the population.
