# VIF – System Architecture & Inference

This document details the inputs, state representation, and inference flow for the **Value Identity Function (VIF)**. It covers how raw data is processed into state vectors and how the system separates the "Critic" (numeric evaluation) from the "Coach" (explanation).

---

## 1. Inputs and State Representation

### 1.1 Raw Inputs

For each user $u$ at time step $t$, we assume:

* Text journal or transcript: $T_{u,t}$.
* Optional audio: $A_{u,t}$ (voice recordings, prosody).
* Optional physiological signals: $H_{u,t}$ (e.g. heart rate, HRV).
* Time features:
  * $\Delta t_{u,t} = t_{u,t} - t_{u,t-1}$: time since previous entry.
* User profile:
  * $z_u$: embedding or feature vector representing the user’s values and identity.

### 1.2 Embedding Functions

We define embedding functions:

* Text encoder (e.g. SBERT or similar):
  $\\phi_{\text{text}}(T_{u,t}) \in \mathbb{R}^{d_e}$
* Optional audio/prosodic encoder:
  $\\phi_{\text{audio}}(A_{u,t}) \in \mathbb{R}^{d_p}$
* Optional physiological encoder (e.g. via time-series feature extraction):
  $\\phi_{\text{physio}}(H_{u,t}) \in \mathbb{R}^{d_h}$
* User profile embedding:
  $z_u \in \mathbb{R}^{d_z}$

### 1.3 State Vector – Sliding Window Design

#### 1.3.1 Motivation: Beyond the Markov Assumption

A design that uses only the current entry $T_{u,t}$ and static profile $z_u$ implicitly assumes a **Markov state**: all relevant information for future alignment is contained in the present features. In real life, this is false. A single entry of —I’m exhausted— may be noise; three consecutive entries with the same theme form a **trajectory**.

To capture trajectories without complex recurrent architectures in the critic, we adopt a **sliding window** of recent entries.

#### 1.3.2 Sliding Window State Definition (Core POC Option)

For a window size $N$ (e.g. $N = 3$ for the POC):

$$
 s_{u,t} = \text{Concat}\Big[
 \\phi_{\text{text}}(T_{u,t}),
 \\phi_{\text{text}}(T_{u,t-1}),\ \dots,\
 \\phi_{\text{text}}(T_{u,t-N+1}),
 $\Delta t_{u,t},\dots,\Delta t_{u,t-N+2}$,
 history stats,
 $z_u$
\Big]
$$

Where **history stats** may include simple per-dimension statistics derived from past Reward Model outputs (for a chosen lookback horizon):

* Exponential moving averages (EMA) of per-dimension alignment.
* Rolling standard deviations.
* Counts of entries over recent time windows (e.g. last 7 days, last 30 days).

For early time steps, where fewer than $N$ entries exist, we can:

* Pad missing embeddings with zeros, or
* Use special —no-history— tokens/flags.

#### 1.3.3 Minimal MVP State (Simplified Option)

For the simplest POC, the state can initially be:

$$
 s_{u,t} = \text{Concat}\big[\\phi_{\text{text}}(T_{u,t}),\ z_u\big]
$$

with the sliding window and history stats added in a later iteration.

#### 1.3.4 Multimodal Extension (Future Option)

Audio and physiological channels can be appended once available:

$$
 s_{u,t} = \text{Concat}\Big[\text{(text window)},\ \\phi_{\text{audio}}(A_{u,t}),\ \\phi_{\text{physio}}(H_{u,t}),\ z_u,\ \text{history stats}\Big]
$$

---

## 2. Inference Flow and Memory Architecture

### 2.1 Inference Flow for VIF (Critic)

For a real user session:

1. **Collect input**:
   * User submits a new journal entry (and optional voice/physio signals).
2. **Build sequential state**:
   * Construct sliding window state $s_{u,t}$ from the current and $N-1$ previous entries, plus profile and history stats.
3. **Reward Model (optional at inference)**:
   * Option A: call LLM-as-Judge online for immediate alignment scores and explanations.
   * Option B: use an offline-trained distilled reward model to approximate judge scores.
4. **Critic evaluation with uncertainty**:
   * Run VIF with MC Dropout to get $\\mu_{u,t}^{(j)}$ and $\\sigma_{u,t}^{(j)}$ per dimension. (See [Uncertainty Logic](VIF_04_Uncertainty_Logic.md))
5. **Aggregate over time**:
   * Compute updated weekly aggregates (e.g. weekly means of $\\mu_{u,t}^{(j)}$).
6. **Apply trigger rules**:
   * If significant weekly drop (crash) and low uncertainty → generate a targeted critique.
   * If sustained low weekly values for $\ge C_{\text{min}}$ weeks (rut) and low uncertainty → generate a targeted critique.
   * If sustained high values over multiple weeks and low uncertainty → generate occasional evidence-based acknowledgment.
   * If high uncertainty → generate a clarifying prompt instead of a critique or acknowledgment.

### 2.2 Separation of Concerns: Critic vs Coach

We explicitly separate two memory and reasoning systems:

1. **VIF (Critic)** – Sequential, time-aware evaluator
   * Uses **strict sequential history** (sliding window of recent entries and derived stats).
   * Does **not** use semantic retrieval of arbitrary past entries for its numeric prediction.
   * Focus: estimate current/short-horizon alignment per dimension and uncertainty.

2. **Coach / Explanation Layer** – Thematic, retrieval-augmented guide
   * Activated after the VIF identifies significant patterns — whether problematic (crash or rut) or positive (sustained alignment).
   * Uses **retrieval-augmented generation (RAG)** over the user's longer-term journal history to:
     * Retrieve thematically relevant entries (e.g. similar conflicts, repeated patterns, or evidence of consistent alignment).
     * Provide context-rich explanations, reflective prompts, or evidence-based acknowledgment.
   * For positive patterns, acknowledgment is infrequent and grounded in specific behaviors — never gamified (no streaks, points, or generic praise).
   * Retrieval is based on **semantic similarity**, not recency, to support pattern recognition and narrative insight.

This separation ensures that:

* The **numeric value scores** remain grounded in recent temporal dynamics and are not contaminated by arbitrarily distant or thematically similar but contextually irrelevant entries.
* The **explanations and reflections** can draw on the full richness of the user’s history.
