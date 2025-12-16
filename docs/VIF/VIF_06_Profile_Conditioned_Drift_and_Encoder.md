## Frozen Text Encoder & Profile‑Conditioned Drift (POC Spec)

This document extends:

- `VIF_02_System_Architecture.md` (state & inference)
- `VIF_03_Model_Training.md` (critics & targets)
- `VIF_05_State_and_Data_Pipeline.md` (state construction)

It pins down:

- How the **frozen text encoder** is used in the POC.
- How the **state definition** supports **profile‑conditioned drift** signals.
- How to compute **drift metrics** on top of the existing Critic without adding a new model family.

---

## 1. Frozen Text Encoder – Role and Usage

### 1.1 Encoder Definition

For each entry text $T_{u,t}$:

- We use a **pretrained sentence encoder** (e.g. SBERT, such as `all-MiniLM-L6-v2`) as

  $$
  \phi_{\text{text}}(T_{u,t}) \in \mathbb{R}^{d_e}
  $$

- The encoder is treated as a **frozen feature extractor** for the capstone POC:
  - Its weights are **not updated** during Critic training.
  - The same input text always maps to the same embedding.

### 1.2 Offline Embedding Pipeline

As part of the pipeline from `VIF_05_State_and_Data_Pipeline.md`:

1. For each `Entry` row `(persona_id, t_index)`:
   - Compute:
     $$
     \mathbf{e}_{u,t} = \phi_{\text{text}}(T_{u,t})
     $$
   - Store these embeddings in an intermediate structure (e.g. an `EntryEmbedding` table).
2. During state construction, we only read $\mathbf{e}_{u,t}$ from storage; we do **not** call the encoder again.

This keeps the Critic training loop simple and makes experiments reproducible (fixed encoder, fixed embeddings).

### 1.3 Rationale for Freezing in the POC

- **Sample‑efficient**: Leverages strong pretrained representations with limited synthetic data.
- **Stable geometry**: Embedding space is fixed over time, which makes history statistics (e.g. EMAs) and drift metrics easier to interpret.
- **Simpler optimisation**: Only the MLP Critic is trained; no gradient flow through the encoder.

Future work (beyond the capstone) may optionally explore **light fine‑tuning** of the encoder, but this is explicitly out of scope for the first POC.

---

## 2. State Vector and Profile Interaction

### 2.1 POC State Definition (Recap)

From `VIF_05_State_and_Data_Pipeline.md`, for window size $N=3$, the state is:

$$
 s_{u,t} = \text{Concat}\Big[
   \underbrace{\phi_{\text{text}}(T_{u,t}), \phi_{\text{text}}(T_{u,t-1}), \phi_{\text{text}}(T_{u,t-2})}_{\text{text window}},
   \underbrace{\Delta t_{u,t}, \Delta t_{u,t-1}}_{\text{time gaps}},
   \underbrace{\text{history\_stats}_{u,t}}_{\text{per-dimension EMA of past alignment}},
   \underbrace{w_u}_{\text{user value profile}}
 \Big]
$$

Where:

- `text_window` are frozen embeddings $\mathbf{e}_{u,t-k}$.
- `history_stats_{u,t}` is the vector of EMAs per value dimension (defined in `VIF_05`).
- $w_u \in \mathbb{R}^K$ is the **value weight vector** for the user/persona, with $w_{u,j} \ge 0$ and $\sum_j w_{u,j} = 1$.

### 2.2 How the Profile Affects the Critic

The user profile is **concatenated** as one block in the state; it is not used by the text encoder itself.

The Critic is a multi‑layer perceptron:

- Input: $s_{u,t}$ (full state, including $w_u$).
- Output: $\hat{\vec{a}}_{u,t} \in [-1,1]^K$, approximating the Judge alignment $\hat{\vec{a}}_{u,t}$ (Option A in `VIF_03`).

Because the Critic sees both:

- The **recent behaviour** (text window + history stats), and
- The **value profile** $w_u$,

it can learn **profile‑conditioned mappings**, e.g. the *same* text trajectory can be treated differently depending on which dimensions the user cares most about.

---

## 3. Drift as “Profile‑Conditioned Misalignment”

### 3.1 Conceptual Definition

Informally, we say a user is **drifting** when:

- Their **recent behaviour**, as seen through the Critic, is repeatedly misaligned with
- The **value dimensions they say matter most** (encoded in $w_u$),
- Over a meaningful time horizon (weeks), not just a single bad day.

The POC implements drift as a set of **deterministic metrics and thresholds** computed on top of:

- The Critic’s alignment estimates \(\hat{\vec{a}}_{u,t}\), and
- The user’s value weights \(w_u\).

No additional “drift model” is required for the POC.

### 3.2 Per‑Dimension Drift Signal

Let:

- $\hat{a}^{(j)}_{u,t}$ be the Critic’s predicted alignment on value $j$ at time $t$,
  $\hat{a}^{(j)}_{u,t} \in [-1,1]$.
- $w_{u,j}$ be the user’s weight for value $j$.

Define a **profile‑weighted misalignment** per dimension:

$$
d^{(j)}_{u,t} = w_{u,j} \cdot \max\big(0,\ -\hat{a}^{(j)}_{u,t}\big)
$$

Interpretation:

- If alignment is positive or neutral ($\hat{a}^{(j)}_{u,t} \ge 0$), we set $d^{(j)}_{u,t} = 0$.
- If alignment is negative ($\hat{a}^{(j)}_{u,t} < 0$), the drift signal scales with **how much the user cares** about this dimension ($w_{u,j}$).

We can track EMAs of $d^{(j)}_{u,t}$ over time to obtain **smoothed drift scores** per value dimension:

$$
\text{ema\_drift}^{(j)}_{u,t} =
\alpha_{\text{drift}} d^{(j)}_{u,t} +
(1-\alpha_{\text{drift}})\, \text{ema\_drift}^{(j)}_{u,t-1}
$$

with $\text{ema\_drift}^{(j)}_{u,0} = 0$.

### 3.3 Profile‑Weighted Scalar Alignment

`VIF_03_Model_Training.md` defines an optional scalar aggregation:

$$
V^{\text{scalar}}_{u,t} = w_u^\top \hat{\vec{a}}_{u,t}
$$

- This is a **profile‑weighted overall alignment score**.
- We can apply the same weekly averaging / EMA logic used for crash/rut detection to \(V^{\text{scalar}}_{u,t}\).

Example usage:

- A **sustained drop** in the weekly EMA of \(V^{\text{scalar}}_{u,t}\) with low uncertainty
  indicates global drift away from what the user values.

### 3.4 Directional Drift vs. Identity Vector

We can also compare:

- The **behavioural alignment vector**, aggregated over a recent horizon (e.g. week):

  $$
  \bar{\vec{a}}_{u,\text{week}} =
  \frac{1}{H'} \sum_{k=0}^{H'-1} \hat{\vec{a}}_{u,t-k}
  $$

- To the **identity / importance vector** $w_u$.

Define:

- **Cosine similarity** between profile and recent alignment:

  $$
  \text{cos\_sim}_u =
  \frac{w_u^\top \bar{\vec{a}}_{u,\text{week}}}
       {\|w_u\| \,\|\bar{\vec{a}}_{u,\text{week}}\|}
  $$

  - Values close to 1: behaviour points broadly in the direction of what the user says they care about.
  - Sustained decreases over weeks: behaviour drifting away from identity.

These scalar metrics are used for **summaries and triggers**, not as training targets.

---

## 4. Drift Detection Rules (on Top of VIF)

### 4.1 Inputs from the Critic and State

For drift detection, we assume access to:

- $\hat{\vec{a}}_{u,t}$: Critic mean predictions per dimension.
- $\hat{\vec{\sigma}}_{u,t}$: Critic uncertainty estimates per dimension (from MC Dropout, see `VIF_04_Uncertainty_Logic.md`).
- $w_u$: user value weight vector.
- History statistics (`history_stats_{u,t}`) including EMAs of alignment, as defined in `VIF_05`.

### 4.2 Example Rule Templates

**Per‑dimension drift rule** (profile‑weighted rut):

- For any value dimension $j$, raise a drift flag if:
  - $w_{u,j} \ge w_{\text{min}}$ (e.g. 0.15),
  - The weekly EMA of $\hat{a}^{(j)}_{u,t}$ is below a negative threshold
    (e.g. $< -0.4$) for at least $C_{\text{min}}$ consecutive weeks,
  - And the corresponding uncertainty is low
    (e.g. weekly average \(\sigma^{(j)}_{u,t} < \sigma_{\text{max}}\)).

**Global drift rule** (profile‑conditioned crash):

- Compute a weekly EMA of the scalar:
  $$
  V^{\text{scalar}}_{u,t} = w_u^\top \hat{\vec{a}}_{u,t}
  $$
- Trigger a **crash** event if:
  - The weekly EMA drops by more than \(\Delta_{\text{crash}}\) compared to the previous week.
  - And uncertainty remains below a chosen threshold.

**Directional identity drift rule**:

- If cosine similarity $\text{cos\_sim}_u$ between $w_u$ and $\bar{\vec{a}}_{u,\text{week}}$:
  - Falls below a fixed threshold (e.g. $< 0.4$), and
  - Has decreased by more than $\Delta_{\text{cos}}$ compared to a baseline period,
  - Then mark this period as a potential **identity drift** episode.

All thresholds $(w_{\text{min}}, \sigma_{\text{max}}, C_{\text{min}}, \Delta_{\text{crash}}, \Delta_{\text{cos}})$ can be tuned on synthetic personas to match intuitive behaviour.

### 4.3 Separation from the Coach

As with crash/rut logic in `VIF_02` and `VIF_04`:

- The **Critic** produces numeric signals ($\hat{\vec{a}}_{u,t}$, uncertainties).
- The **Drift detection layer** applies deterministic rules over time.
- The **Coach** uses these drift events to:
  - Retrieve evidence from the user’s history.
  - Generate gentle, value‑grounded reflections (not prescriptive or gamified).

---

## 5. Model Choices and Scope for the Capstone POC

### 5.1 Models to Use

- **Frozen Text Encoder**
  - Pretrained sentence encoder (e.g. SBERT‑style).
  - Used as \(\phi_{\text{text}}(T_{u,t})\); no fine‑tuning in the POC.
- **Critic (VIF)**
  - Input: full state $s_{u,t}$ including text window, time gaps, history stats, and $w_u$.
  - Architecture: 2–3 layer MLP with ReLU or GELU, dropout on hidden layers.
  - Output: $\hat{\vec{a}}_{u,t} \in [-1,1]^K$ (Option A: immediate alignment).
  - Uncertainty: MC Dropout at inference time.
- **Drift Metrics**
  - Implemented as **deterministic computations** (per‑dimension drift, scalar alignment, cosine similarity, EMAs) on top of Critic outputs and $w_u$.

This keeps the POC within a **supervised learning + rules** regime, avoiding additional model families such as separate drift classifiers.

### 5.2 Optional Future Extensions (Non‑POC)

For future work (not required for the capstone), you could explore:

- A small supervised **drift classifier**:
  - Input features: aggregated statistics (e.g. weekly means and EMAs of $\hat{\vec{a}}_{u,t}$, $V^{\text{scalar}}_{u,t}$, $\text{cos\_sim}_u$, and $w_u$).
  - Model: logistic regression or a 1–2 layer MLP.
  - Target: synthetic labels such as “in drift episode / not in drift episode”.
- **Personalisation layers**:
  - Global Critic + tiny per‑user adapters for users whose trajectories systematically diverge from population‑level patterns.

These are explicitly **out of scope** for the academic, time‑boxed POC but are compatible with the architecture in this document.

---

## 6. Summary

- The **frozen text encoder** provides stable, reusable embeddings for each entry; it is not trained in the POC.
- The **state vector** combines text window, time gaps, history EMAs, and the user profile $w_u$ so that the Critic can learn profile‑conditioned alignment estimates.
- **Drift** is defined as sustained, profile‑weighted misalignment over time and is implemented via **transparent metrics and rules** (per‑dimension drift, scalar alignment, cosine similarity) built on top of the Critic outputs.
- No new model family is required for drift in the POC; the existing Critic plus deterministic logic suffices, keeping the implementation feasible while remaining faithful to the VIF design goals.


