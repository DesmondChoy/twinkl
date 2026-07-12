# VIF – Uncertainty, Drift, and Trigger Logic

This document describes how VIF outputs become Coach-facing signals. It keeps
the selected sustained-conflict contract separate from the weekly
crash/rut/evolution prototype that is still wired into the offline runtime.

---

## 1. Why Uncertainty Gating Matters

A Critic trained on synthetic or otherwise limited data can be confidently
wrong on unfamiliar inputs. In a values-alignment product, a conservative
deferral is safer than a confident but brittle interpretation.

The system therefore separates three questions:

1. What class probabilities or alignment estimate does the Critic produce?
2. How much should the system trust that estimate?
3. Does the recent evidence form the sustained pattern required by the drift
   contract?

Only low-enough uncertainty plus meaningful repeated evidence should reach the
Coach as a confident conflict reflection.

Under the adopted [capstone scope decision](05_capstone_scope_decision.md), an
uncertain or abstaining scorer emits no drift claim. Evaluation must report
coverage, abstention count, and any true episodes suppressed by uncertainty.
The conservative precision or false-alert tolerance is not fixed yet.

---

## 2. Monte Carlo Dropout

For the MLP path, epistemic uncertainty is estimated with MC Dropout:

1. keep dropout active at inference time;
2. run the same state through the model `N` times; and
3. summarize the predictive distribution.

For a dimension `j`:

$$
\mu_{u,t}^{(j)} = \frac{1}{N}\sum_{i=1}^{N} V_j^{(i)}(s_{u,t})
$$

$$
\sigma_{u,t}^{2(j)} = \operatorname{Var}_i[V_j^{(i)}(s_{u,t})]
$$

The mean is the current runtime alignment estimate. The spread is the
uncertainty proxy.

Ambiguous inputs can average toward neutral while still producing high spread
across dropout samples. That combination should not be interpreted as a
confident neutral judgment.

Global calibration is not sufficient for the drift use case. Uncertainty must
also be checked on the `-1` class because selective prediction can otherwise
suppress the exact minority cases Twinkl needs to detect.

---

## 3. Runtime Signal Surfaces

### 3.1 Per-Entry Timeline

`src/vif/runtime.py` reconstructs student-visible states and writes one row per
journal session with:

- per-dimension alignment means;
- per-dimension uncertainties;
- the persona profile weights; and
- entry metadata.

The current timeline artifact does not persist ordinal class probabilities.
The selected v1 detector needs `P(-1)`, so the runtime still requires either a
probability artifact surface or a deterministic reconstruction path from the
checkpoint output.

### 3.2 Weekly Frame

The existing prototype also aggregates the timeline into weekly rows with:

- per-dimension mean alignment;
- per-dimension mean uncertainty;
- profile weights;
- profile-weighted overall mean alignment; and
- profile-weighted overall uncertainty.

`src/vif/weekly_schema.py` is the source of truth for this producer/consumer
contract. `aggregate_timeline_by_week()` emits the ordered columns and
`detect_weekly_drift()` validates all required fields before routing. Missing
columns fail at the boundary with a `ValueError` that names them.

---

## 4. Profile-Conditioned Evidence

Twinkl asks whether behavior conflicts with what this person says matters, not
whether the behavior is generically negative.

For profile weight `w_{u,j}` and predicted alignment
`\hat{a}_{u,t}^{(j)}`:

$$
d_{u,t}^{(j)} = w_{u,j} \cdot \max(0, -\hat{a}_{u,t}^{(j)})
$$

This conceptual signal is zero for positive or neutral alignment and scales
negative alignment by declared importance.

The profile-weighted scalar summary is:

$$
V_{u,t}^{\text{scalar}} = w_u^\top \hat{\vec{a}}_{u,t}
$$

The scalar is useful for monitoring. It does not replace the vector output or
the named value dimension required for an explainable Coach reflection.

---

## 5. Selected v1 Drift Contract

Drift v1 is a sustained conflict episode:

> Two adjacent journal entries clearly show the writer making a behavior or
> choice against the same declared core value.

The runtime target accumulates recent soft `P(-1)` evidence for that value while
uncertainty remains below a calibrated ceiling. Hard argmax sequences are not
the runtime target because the current Critic frequently hedges true conflict
toward neutral.

| Layer | v1 behavior |
|---|---|
| Student-visible target | Two adjacent entries visibly show a behavior or choice against the same declared core value; `twinkl-v8pb` completed its full-runtime-text review, but no promotion score was allowed after one case remained unresolved |
| Historical consensus table | Retired diagnostic provenance only; not a drift target, threshold-selection input, or promotion surface |
| Runtime | Rolling `P(-1)` evidence with declared-core and uncertainty gates; production integration remains blocked |
| Delivery | Weekly digest with cited journal evidence and active, recovered, mixed, or uncertain wording; abstention emits no drift claim; exact schema pending |

The EDA supports this definition because most single-entry dips recover within
two entries, while three-step and multi-week definitions are too sparse for the
short observed trajectories. See
[`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md).

### 5.1 Delivery-Time Recovery

Target detection and Coach wording answer different questions. The
student-visible target records whether a sustained-conflict episode occurred.
The weekly digest should describe the state at delivery time.

For example, `-1, -1, +1, +1, +1` remains a true benchmark episode, but the
Coach should describe it as **recovered** rather than **active**. **Mixed** is a
digest-level summary used only when relevant value-specific episodes have
different delivery states; it is not another state for a single episode.
**Uncertain** applies when evidence reliability is too low to call an episode
active or recovered. Exact schema values and transition rules still require
implementation and scenario tests.

Abstention is not a correct negative. Coverage and suppressed reference
episodes must remain visible in evaluation reports so uncertainty gating cannot
improve apparent precision merely by hiding hard cases.

---

## 6. Existing Weekly Runtime Prototype

`src/vif/drift.py` consumes the weekly mean/uncertainty frame and can emit:

- `stable`;
- `crash`;
- `rut`;
- `evolution`; or
- `high_uncertainty`.

Its first gate routes high overall uncertainty away from a confident critique.
It then checks a week-over-week profile-weighted drop, consecutive low weekly
means on important dimensions, and experimental evolution classifications.

This path is wired into `src/coach/runtime.py` and is useful for end-to-end
schema, artifact, and UI testing. It is not the selected v1 detector because it
does not consume rolling `P(-1)` evidence or evaluate the strict
sustained-conflict construct.

---

## 7. Experimental Evolution Routing

The current prototype automatically calls `classify_weekly_evolution()` when
no precomputed evolution result is supplied. Eligible dimensions can therefore
produce the literal `evolution` response mode and an optional profile-update
suggestion.

This is implementation truth, not a product-scope decision. The PRD parks value
evolution outside the committed v1 contract. The prototype branch remains an
experimental compatibility surface until it is either adopted explicitly or
removed from the active router.

---

## 8. Exploratory Detector Comparison

The demo review app compares six rule-based detector families:

- Baseline;
- EMA;
- CUSUM;
- Cosine;
- Control Chart; and
- KL Divergence.

These detectors operate on persisted single-pass Judge labels or Critic mean
predictions. Their vote count is detector agreement, not the five-pass Judge
reference and not v1 benchmark ground truth.
They remain useful for diagnosis and visualization but do not define the
promoted runtime rule.

---

## 9. Coach-Facing Safety Behavior

The standalone weekly digest also has conservative fallback modes for offline
prompt testing when no upstream drift result is supplied. Acute grief or
distress markers can route to `high_uncertainty`, while mixed or burdened weeks
can use `mixed_state` or `background_strain`.

These lexical/aggregate fallbacks are not substitutes for calibrated Critic
uncertainty. They are local safety scaffolding around the artifact-generation
path.

---

## 10. Implementation Reference

| Module | Role |
|---|---|
| `src/vif/eval.py` | Entry-level metrics and uncertainty-aware evaluation |
| `src/vif/runtime.py` | Per-entry and weekly VIF artifact generation |
| `src/vif/weekly_schema.py` | Shared weekly frame names and required-column validation |
| `src/vif/drift.py` | Existing weekly crash/rut/evolution/high-uncertainty router |
| `src/vif/evolution.py` | Experimental stable/evolution/drift classifier |
| `src/coach/runtime.py` | Offline checkpoint-to-digest orchestration |
| `src/demo_tool/multi_drift.py` | Six-detector exploratory comparison |
| `scripts/drift/trajectory_eda.py` | Sustained-conflict definition analysis |

---

## 11. Later Uncertainty Extensions

MC Dropout remains the practical POC default. Later candidates include deep
ensembles, evidential methods, conformal wrappers, and explicit embedding-space
out-of-distribution detection. The retired `twinkl-wq9p` diagnostic showed a
target-validity problem before it could establish whether uncertainty calibration
is the binding constraint. `twinkl-v8pb` completed that review and withheld a
promotion score because one locked case was unresolved; the next step is a
fresh, independently resolved promotion surface, not a heavier uncertainty
method.
