# VIF – Uncertainty & Critique Logic

This document details the mathematical logic for **uncertainty estimation** and the **dual-trigger critique rule** (Crash & Rut detection).

---

## 1. Uncertainty-Aware Critiques

### 1.1 Problem: Overconfident Critic on OOD Inputs

A critic trained offline (especially on synthetic or limited datasets) can produce **over-confident, incorrect** predictions when faced with out-of-distribution (OOD) inputs (e.g. novel situations or unusual life events).

This is especially problematic in a mental health or values-alignment context.

### 1.2 Monte Carlo Dropout

To estimate **epistemic uncertainty**, we use **Monte Carlo Dropout** at inference time:

1. Keep dropout layers **active** during testing.
2. For a given state $s_{u,t}$, run the critic $N$ times:
   * $\vec{V}^{(i)} = V_\theta^{(i)}(s_{u,t})$, for $i = 1, \dots, N$.
3. Compute, for each value dimension $j$:
   * Mean (estimate):

$$ 
\mu_{u,t}^{(j)} = \frac{1}{N} \sum_{i=1}^N V^{(i)}_j
$$ 

* Variance (uncertainty):

$$ 
\sigma_{u,t}^{2(j)} = \text{Var}_i[V^{(i)}_j]
$$ 

The mean $\mu_{u,t}^{(j)}$ serves as the value estimate; the variance $\sigma_{u,t}^{2(j)}$ is a proxy for epistemic uncertainty.

We can monitor calibration by checking, on held-out data, whether higher $\sigma_{u,t}^{2(j)}$ correlates with larger prediction errors.

### 1.3 Dual-Trigger Critique Rule (Crashes and Ruts)

We define two types of problematic patterns for each value dimension $j$:

* **Sudden crash:** recent sharp negative change.
* **Chronic rut:** sustained low values over time.

Let:

* $V_{t-1}^{(j)} = \mu_{u,t-1}^{(j)}$: previous estimate.
* $V_t^{(j)} = \mu_{u,t}^{(j)}$: current estimate.
* $\sigma_{V_t}^{(j)} = \sqrt{\sigma_{u,t}^{2(j)}}$: current uncertainty.
* $\delta_j$: crash threshold for dimension $j$.
* $\tau_{\text{low}}^{(j)}$: low-value threshold for dimension $j$.
* $C_t^{(j)}$: count of **consecutive time windows** (e.g. weeks) where $V_t^{(j)} < \tau_{\text{low}}^{(j)}$.

We trigger a **critique** for dimension $j$ if **uncertainty is low enough** and either a crash or rut condition is met.

**Condition A – Sudden Crash:**

$$ 
V_{t-1}^{(j)} - V_t^{(j)} > \delta_j
$$ 

**Condition B – Chronic Rut:**

$$ 
V_t^{(j)} < \tau_{\text{low}}^{(j)} \quad \text{and} \quad C_t^{(j)} \ge C_{\text{min}}
$$ 

where $C_{\text{min}}$ is a minimum number of consecutive periods (e.g. 3 weeks).

**Uncertainty Constraint:**

$$ 
\sigma_{V_t}^{(j)} < \epsilon_j
$$ 

If the uncertainty is **high** (i.e. $\sigma_{V_t}^{(j)} \ge \epsilon_j$), the system should:

* Avoid issuing a strong critique.
* Prefer asking clarifying questions or deferring judgment.

### 1.4 Time and Aggregation

To implement crash and rut logic robustly:

* VIF outputs can be aggregated to **weekly averages** per dimension.
* Crash/rut detection operates on these weekly aggregates, not raw per-entry values.
* $C_t^{(j)}$ is then defined over weeks, making –three weeks in a rut– a natural threshold.

### 1.5 Alternative Uncertainty Methods (Comparison)

While MC Dropout is chosen for the Tier 1 POC due to its implementation simplicity, other methods exist and may be appropriate for production systems.

| Method | Pros | Cons | Best For |
| :--- | :--- | :--- | :--- |
| **MC Dropout** (Chosen for POC) | • Zero extra code complexity.<br>• No storage overhead.<br>• "Good enough" epistemic uncertainty. | • Requires multiple forward passes (inference latency).<br>• Calibration can be sensitive to dropout rate. | **Academic POCs & MVP** where dev speed > perfect calibration. |
| **Deep Ensembles** | • Gold standard for accuracy & calibration.<br>• Handles both aleatoric & epistemic well. | • High compute/storage cost (train & store 5+ models). | **High-budget Production** where reliability is paramount. |
| **Deep Evidential Regression** | • Fast (single forward pass).<br>• Disentangles uncertainty types explicitly. | • Custom loss functions can be unstable to train.<br>• Harder to implement. | **Low-latency Production** apps on mobile/edge. |
| **Conformal Prediction** | • Provides statistical **guarantees** (e.g., "90% confidence interval").<br>• Model-agnostic. | • Requires separate calibration dataset.<br>• Focuses on intervals, not just scalar uncertainty. | **Safety-Critical Systems** (Medical/Finance) needing strict bounds. |
