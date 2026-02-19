# VIF – Reward Modeling & Training Strategy

This document specifies the **Reward Model**, the **Vector-Valued Value Function** targets, and the **Training Strategy** for the VIF Critic.

---

## 1. Reward Modeling (LLM-as-Judge)

### 1.1 Problem: No Oracle Reward in Real Data

For real users, we do not observe:

* True per-dimension alignment $a_{u,t}^{(j)}$ (e.g. “Health = −0.8”).
* True cumulative returns.

We only observe **text and signals**. To bridge this, we explicitly introduce a **Reward Model**.

### 1.2 The Generator-Judge-Critic Workflow (Teacher-Student Architecture)

To handle the complexity of real-life trade-offs (e.g., sacrificing sleep for work) while enabling efficient uncertainty estimation, we adopt a **Knowledge Distillation** approach.

This workflow transforms rich but slow LLM reasoning into a fast, uncertainty-aware numerical signal.

#### 1.2.1 Phase 1: Synthesis (The Generator)
We use an LLM to generate a diverse dataset of synthetic journal entries.
*   **Goal:** Create "messy," realistic data that reflects genuine human conflict, not just clean archetypes.
*   **Prompt Strategy:** Instead of asking for "A bad health entry," we ask for scenarios involving specific tensions, e.g., *"Write an entry from a user who is proud of their work achievement but is neglecting their physical needs to get there."*

#### 1.2.2 Phase 2: Labeling (The Judge / Teacher)
We feed the generated text into a second LLM pass (The Judge) to assign ground-truth scores.
*   **Why this is needed:** A user entry often impacts multiple values simultaneously (Side Effects). The Generator prompt might target "Work," but the resulting text might inadvertently reveal neglect of "Family."
    *   **Mechanism (Categorical Protocol):** To avoid subjective noise (e.g., arbitrary "4.5" vs "4.2"), the Judge classifies behavior using a strict 3-point rubric:
    *   **Misaligned (-1):** The entry actively conflicts with the value.
    *   **Neutral (0):** The entry is irrelevant to the value or maintains status quo.
    *   **Aligned (+1):** The entry actively supports the value.
    *   **Output:** These classifications are mapped to a precise integer vector, e.g., `[Health: -1, Career: +1, Family: 0]`, which serves as the regression target for the Student.
#### 1.2.3 Phase 3: Distillation (The Critic / Student)
We train the **MLP Critic** (the VIF) on this labeled dataset.
*   **Input:** Text Embedding (SBERT) + User Profile Embedding.
*   **Target:** The Judge's vector scores.
*   **Benefit:**
    1.  **Speed:** The MLP runs in milliseconds, enabling the 50+ forward passes required for MC Dropout.
    2.  **Privacy:** The MLP can potentially run locally or with lower privacy overhead than a full LLM call.

### 1.3 Uncertainty Gating
By using the MLP as the live Critic, we unlock **Epistemic Uncertainty** estimation via MC Dropout.
*   If the MLP's 50 predictions cluster tightly (Low Variance), we trust the score.
*   If the MLP's predictions scatter widely (High Variance), it means the input is **Out-of-Distribution** (unlike anything the Judge taught it).
    *   *Action:* The system suppresses the critique and instead triggers the Coach to ask a clarifying question ("I'm sensing some complexity here...").

---

## 2. Vector-Valued Value Function – Multiple Target Options

### 2.1 Motivation

Scalarizing rewards too early (e.g. $r = w^	op a$) hides important conflicts:

* A week with (+10) Career and (−10) Relationships is not “neutral” in human terms.
* Twinkl’s purpose is to **surface such tensions**, not to flatten them away.

Therefore, we keep the value function **vector-valued**.

We define several **target options** for the VIF, from simplest to most advanced. The capstone team can choose one for implementation.

### 2.2 Option A: Immediate Alignment (Simplest)

The VIF directly predicts the Reward Model’s immediate alignment scores:

$$ 
\vec{V}_\theta(s_{u,t}) \approx \hat{\vec{a}}_{u,t}
$$ 

This is a **multi-output regression** problem, where the critic learns to emulate the judge, potentially with:

* Lower latency.
* Better generalisation from richer state features (e.g. using history window and profile).

Longer-term trends are then derived via **deterministic smoothing** (e.g. EMA over time), not via discounted returns.

### 2.3 Option B: Short-Horizon Forecast (Intermediate)

The VIF predicts short-horizon average alignment, e.g. over the next $H$ days or entries (e.g. $H = 7$ days):

$$ 
G_{u,t}^{(j, H)} = \frac{1}{H'} \sum_{k=0}^{H'-1} \hat{a}_{u,t+k}^{(j)}
$$ 

where $H'$ is the number of future entries within the horizon. The critic is trained to predict:

$$ 
\vec{V}_\theta(s_{u,t}) \approx \vec{G}_{u,t}^{(H)} = \big(G_{u,t}^{(1,H)}, \dots, G_{u,t}^{(K,H)}\big)
$$ 

This gives genuinely **prospective** signals but limits the horizon to a more realistic, data-supported window.

### 2.4 Option C: Discounted Returns (Advanced)

For each user $u$, time $t$, and value dimension $j$:

1. The Reward Model provides immediate alignment: $\hat{a}_{u,t}^{(j)} \in [-1,1]$.
2. We define a **discounted cumulative alignment** (return):

$$ 
G_{u,t}^{(j)} = \sum_{k=0}^{T_u - t} \gamma^{g(\Delta t_{u,t+k})} \hat{a}_{u,t+k}^{(j)}
$$ 

where:

* $\gamma \in (0,1]$ is the base discount factor.
* $g(\Delta t)$ is a function that maps irregular time gaps to effective discount exponents (e.g. $g(\Delta t) = \Delta t / \tau$ for some time constant $\tau$).
* $T_u$ is the final time step for user $u$.

The critic approximates:

$$ 
\vec{V}_\theta(s_{u,t}) \approx \mathbb{E}[\vec{G}_{u,t} \mid s_{u,t}]
$$ 

This option is closest to an RL-style value function but also the most demanding.

### 2.5 Scalar Aggregation (for Summaries Only)

When needed (e.g. for an overall weekly alignment score), we can aggregate:

$$ V^{\text{scalar}}_{u,t} = w_{u,t}^\top \vec{V}_\theta(s_{u,t})
$$ 

This scalar is used only for **summary views**; critiques and explanations are primarily based on the vector $\vec{V}_\theta$.

---

## 3. Training Objective and Strategy

### 3.1 Training Type

For the capstone POC, training the critic is:

* **Supervised learning**, not reinforcement learning.
* The model learns a mapping:
  * From state $s_{u,t}$
  * To a chosen target vector (depending on Option A/B/C): $\hat{\vec{a}}_{u,t}$, $\vec{G}_{u,t}^{(H)}$, or $\vec{G}_{u,t}$.

We use:

* **Pretrained models** (e.g. SBERT) for text embeddings.
* A **from-scratch multi-output regressor** for the critic itself.

### 3.2 Loss Function

For a generic target vector $\vec{Y}_{u,t}$ (immediate, short-horizon, or discounted), we define the loss over all users and time steps:

$$ 
\mathcal{L}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{(u,t) \in \mathcal{D}} \sum_{j=1}^K w_{u,t}^{\text{label}} \Big( V_\theta^{(j)}(s_{u,t}) - Y_{u,t}^{(j)} \Big)^2
$$ 

where:

* $\mathcal{D}$ is the dataset of all (state, target) pairs.
* $w_{u,t}^{\text{label}}$ is an optional **label-weighting term** (e.g. based on judge confidence or label variance estimates).

In the simplest version (Option A), $\vec{Y}_{u,t} = \hat{\vec{a}}_{u,t}$ and $w_{u,t}^{\text{label}} = 1$.

### 3.3 Model Architecture

A minimal architecture:

* Input: state vector $s_{u,t}$.
* Hidden layers: 2–3 dense layers (MLP) with non-linearities (e.g. ReLU or GELU), with dropout for regularisation and uncertainty estimation.
* Output layer: linear layer with $K$ outputs (one per value dimension).

The text encoder (e.g. SBERT) is:

* Either frozen in the first iteration (for simplicity and data efficiency).
* Or lightly fine-tuned jointly with the critic if data allows.

### 3.4 Training Pipeline Overview

1. **Ontology creation**: define value dimensions, rubrics, and examples.
2. **Reward modeling (offline)**:
   * Run LLM-as-Judge on existing (synthetic and/or real) entries to obtain $\hat{\vec{a}}_{u,t}$ and optional confidence scores.
3. **Target construction**:
   * Depending on chosen option:
     * Option A: set $\vec{Y}_{u,t} = \hat{\vec{a}}_{u,t}$.
     * Option B: compute short-horizon averages $\vec{G}_{u,t}^{(H)}$.
     * Option C: compute discounted returns $\vec{G}_{u,t}$ with time-aware discounting.
4. **State construction**:
   * Encode text (and other signals if available) into windowed state vectors $s_{u,t}$.
5. **Critic training**:
   * Train $\vec{V}_\theta$ via supervised regression on the chosen targets.
6. **Evaluation**:
   * Use held-out trajectories/personas to evaluate prediction quality (e.g. MSE, correlation) and ranking consistency (e.g. ordering weeks by misalignment).

---

## 4. Implementation Reference

The VIF Critic training pipeline is implemented in `src/vif/`. Key modules:

| Module | Description |
|--------|-------------|
| `src/vif/encoders.py` | `TextEncoder` protocol + `SBERTEncoder` (supports ablation studies) |
| `src/vif/state_encoder.py` | `StateEncoder` class (builds state vectors per Section 2 above) |
| `src/vif/critic.py` | `CriticMLP` with MC Dropout for uncertainty estimation |
| `src/vif/dataset.py` | `VIFDataset` + data loading with persona-level splits |
| `src/vif/eval.py` | Evaluation metrics (MSE, Spearman, accuracy, calibration) |
| `src/vif/train.py` | CLI training script with config overrides |

> **Note:** Specific model names, embedding dimensions, window sizes, and
> hyperparameters referenced below are illustrative. See `config/vif.yaml`
> for current runtime values.

### 4.1 Implemented Architecture

The POC implements **Option A (Immediate Alignment)** with these concrete choices:

- **Text encoder**: Frozen SBERT encoder (model and $d_e$ configured in `config/vif.yaml`)
- **State dimension**: $N \times d_e + (N{-}1) + 10 + 10$ (see config for current $N$ and $d_e$)
- **Critic architecture**: 2-layer MLP with LayerNorm + GELU + Dropout (param count depends on `hidden_dim`)
- **MC Dropout**: 50 forward passes for uncertainty estimation
- **Loss**: Ordinal classification losses (CORAL, CORN, EMD, SoftOrdinal)

### 4.2 Usage

```bash
# Full training with default config
python -m src.vif.train

# Quick test run
python -m src.vif.train --epochs 5 --batch-size 8

# Ablation with different encoder
python -m src.vif.train --encoder-model all-mpnet-base-v2
```

Configuration: `config/vif.yaml`
Scripts: `src/vif/train.py` and `src/vif/train_bnn.py`
