# VIF – Uncertainty, Drift, and Trigger Logic

This document describes how the VIF turns raw model outputs into Coach-facing
signals. It combines the uncertainty rules, profile-conditioned drift framing,
and the weekly trigger logic used by the experimental runtime bridge.

---

## 1. Why Uncertainty Gating Matters

A critic trained on synthetic or otherwise limited data can be confidently wrong
on unfamiliar inputs. In a values-alignment product, that is more harmful than a
well-calibrated refusal to judge.

The system therefore separates two questions:

1. What alignment signal does the student predict?
2. How much should we trust that signal?

Only the combination of a meaningful pattern and low enough uncertainty should
reach the Coach as a confident critique.

---

## 2. Monte Carlo Dropout

For the MLP path, epistemic uncertainty is estimated with MC Dropout:

1. keep dropout active at inference time
2. run the same state through the model `N` times
3. compute a mean and spread over those predictions

For a dimension `j`:

$$
\mu_{u,t}^{(j)} = \frac{1}{N}\sum_{i=1}^{N} V_j^{(i)}(s_{u,t})
$$

$$
\sigma_{u,t}^{2(j)} = \text{Var}_i[V_j^{(i)}(s_{u,t})]
$$

The mean is the usable alignment estimate. The spread is the uncertainty proxy.

### 2.1 Ambiguous Inputs

Variance is especially useful when the entry contains mixed evidence. An
ambiguous input may average toward neutral while still producing high spread
across dropout samples. In practice, that is a good reason not to treat the
output as a confident neutral judgment.

---

## 3. Weekly Signal Surface

The runtime bridge does not feed raw per-entry predictions directly into the
Coach. It first aggregates them into weekly signals.

Each weekly row includes:

- per-dimension mean alignment
- per-dimension mean uncertainty
- profile weights
- profile-weighted overall mean alignment
- profile-weighted overall uncertainty

These weekly summaries are the input to crash/rut-style detection.

---

## 4. Profile-Conditioned Drift Framing

The VIF is not only asking "is this behavior negative?" It is asking whether the
behavior conflicts with what this user says matters.

### 4.1 Per-Dimension Weighted Misalignment

For a profile weight `w_{u,j}` and predicted alignment `\hat{a}_{u,t}^{(j)}`:

$$
d_{u,t}^{(j)} = w_{u,j} \cdot \max(0, -\hat{a}_{u,t}^{(j)})
$$

This is a useful conceptual drift signal:

- if alignment is positive or neutral, the contribution is zero
- if alignment is negative, the contribution scales with user importance

### 4.2 Profile-Weighted Scalar Alignment

The scalar summary used downstream is:

$$
V_{u,t}^{\text{scalar}} = w_u^\top \hat{\vec{a}}_{u,t}
$$

This is not a replacement for the vector output. It is a compact summary used
for weekly monitoring and trigger decisions.

---

## 5. Crash and Rut Logic

### 5.1 High-Uncertainty Gate

At the weekly level, the first gate is uncertainty:

- if overall uncertainty is above threshold, do not emit a confident critique
- instead route to a high-uncertainty / clarifying Coach mode

This matches the current runtime detector more closely than a purely
dimension-local rule.

### 5.2 Rut

A rut is a sustained low-alignment pattern on an important value dimension.

In the current runtime detector, a dimension is a rut candidate when:

- its profile weight is above a minimum importance threshold
- its weekly alignment stays below a low threshold
- this persists for at least `C_min` weeks
- weekly uncertainty stays below threshold during that span

### 5.3 Crash

A crash is a sharp week-over-week drop in overall profile-weighted alignment.

In the current runtime detector:

- compute the drop in overall weekly scalar alignment
- if that drop exceeds the crash threshold, inspect important dimensions
- dimensions that declined become the triggered crash dimensions

This makes the crash rule profile-aware without requiring a separate learned
drift model.

### 5.4 Stable and Positive Weeks

When neither crash nor rut fires and uncertainty stays acceptable, the system
can classify the week as stable. Positive acknowledgment remains a Coach-layer
behavior built on top of these signals rather than a separate student target.

---

## 6. Experimental Evolution Routing

There is now experimental code that classifies recent weekly behavior as:

- `stable`
- `evolution`
- `drift`

The idea is to distinguish:

- **behavioral struggle**: noisy or volatile divergence from stated values
- **genuine value change**: sustained, lower-volatility directional shift

When enabled, dimensions classified as `evolution` can bypass crash/rut
messaging and instead route to profile-update-style Coach language.

Important scope note:

- the runtime experiment path supports this
- the PRD still treats value-evolution filtering as experimental rather than
  part of the committed product scope

So this logic should be read as an active experiment, not settled product
behavior.

---

## 7. Future Drift Signals

Some useful drift summaries remain future-facing:

- EMA-based smoothed drift curves
- cosine similarity between recent behavioral alignment and declared profile
- explicit embedding-space OOD detectors layered on top of MC Dropout

These can strengthen monitoring and calibration later, but they are not required
for the current crash/rut runtime bridge.

---

## 8. Implementation Reference

| Module | Role |
|--------|------|
| `src/vif/eval.py` | Uncertainty-aware evaluation utilities |
| `src/vif/runtime.py` | Per-entry and weekly VIF artifact generation |
| `src/vif/drift.py` | Weekly crash/rut/high-uncertainty detection |
| `src/vif/evolution.py` | Experimental stable/evolution/drift classification |

---

## 9. Alternative Uncertainty Methods

MC Dropout remains the practical POC default, but future alternatives include:

- deep ensembles
- evidential methods
- conformal wrappers

Those may improve calibration later, but they are not needed to understand the
current runtime trigger stack.
