# VIF – Uncertainty and Drift Review Logic

This document describes how VIF Critic uncertainty supports offline review,
retraining, and a conditional candidate-generation path. The approved
user-facing Drift path does not consume VIF Critic outputs. It remains separate
from the weekly crash/rut/evolution prototype that is still wired into the
offline runtime.

---

## 1. Why Uncertainty Gating Matters

A VIF Critic trained on synthetic or otherwise limited data can be confidently
wrong on unfamiliar inputs. In a values-alignment product, a conservative
deferral is safer than a confident but brittle interpretation.

The architecture therefore separates four questions:

1. What class probabilities or alignment estimate does the VIF Critic produce?
2. Which predictions warrant offline review or candidate generation?
3. Does the Weekly Drift Reviewer confirm Conflict from Journal Entry text?
4. Do two consecutive confirmed Conflicts meet the Drift definition?

Uncertainty can prioritize review, but it cannot create or confirm a
user-facing Drift claim by itself.

Under the adopted [capstone scope decision](05_capstone_scope_decision.md), an
uncertain VIF Critic prediction remains useful for error analysis and candidate
review. The Weekly Drift Reviewer owns abstention in the approved user-facing
path. Evaluation must report coverage, abstention count, and suppressed true
Drifts. The conservative precision or false Drift alert tolerance is not fixed
yet.

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

Global calibration is not sufficient for Conflict screening. Any conditional
candidate-selection rule must check uncertainty on the `-1` class because
selective prediction can otherwise suppress the exact minority cases the VIF
Critic is meant to recover.

---

## 3. Runtime Outputs

### 3.1 Journal Entry Timeline

`src/vif/runtime.py` reconstructs student-visible states and writes one row per
Journal Entry, including its displayed nudge and response when present, with:

- per-dimension alignment means;
- per-dimension uncertainties;
- the persona profile weights; and
- Journal Entry metadata.

The current timeline parquet does not persist ordinal class probabilities.
The approved Drift Detector does not need them, but the VIF Critic
review-and-retrain path requires persisted probabilities, uncertainty,
checkpoint provenance, and input-contract version.

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
negative alignment by declared importance. It is useful for offline monitoring
and candidate-selection research, not as direct input to the approved Drift
Detector.

The profile-weighted scalar summary is:

$$
V_{u,t}^{\text{scalar}} = w_u^\top \hat{\vec{a}}_{u,t}
$$

The scalar is useful for offline monitoring. It does not replace the vector
output or the named Core Value needed for candidate-selection research.

---

## 5. Approved v1 Drift Contract

Drift v1 is:

> Two consecutive Journal Entries each clearly show the writer making a
> behavior or choice against the same Core Value.

The Weekly Drift Reviewer decides whether each relevant Journal Entry shows
Conflict, non-Conflict, or insufficient evidence without seeing VIF Critic
predictions. The Drift Detector then applies the deterministic rule. The VIF
Critic may later propose candidate adjacent pairs, but only after predefined
criteria and a fresh final test support deployment approval.

| Layer | v1 behavior |
|---|---|
| Student-visible target | Two consecutive Journal Entries each visibly show a Conflict for the same Core Value |
| Historical consensus table | Retired diagnostic provenance only; not a Drift target, threshold-selection input, or final test set |
| Approved user-facing path | Weekly Drift Reviewer decisions without VIF Critic input, followed by the deterministic Drift Detector; production integration remains pending |
| VIF Critic path | Store predictions and uncertainty for offline comparison, independent review, retraining, and conditional candidate generation |
| Delivery | Weekly Digest with cited Journal Entry evidence and active, recovered, mixed, or uncertain wording; Weekly Drift Reviewer abstention emits no Drift claim; exact schema pending |

The EDA supports this definition because most dips spanning one Journal Entry
recover within two Journal Entries, while three-step and multi-week definitions are too
sparse for the short observed trajectories. See
[`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md).

### 5.1 Delivery-Time Recovery

Drift detection and Weekly Coach wording answer different questions. The
student-visible target records whether Drift occurred. The Weekly Digest should
describe the state at delivery time.

For example, `-1, -1, +1, +1, +1` remains a true benchmark Drift, but the
Weekly Coach should describe it as **recovered** rather than **active**.
**Mixed** is a Weekly Digest summary used only when relevant value-specific
Drifts have different delivery states; it is not another state for one Drift.
**Uncertain** applies when evidence reliability is too low to call a Drift
active or recovered. Exact schema values and transition rules still require
implementation and scenario tests.

Abstention is not a correct negative. Coverage and suppressed reference Drifts
must remain visible for both Weekly Drift Reviewer abstention and any
uncertainty-gated candidate rule, so apparent precision cannot improve merely
by hiding hard cases.

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
schema, output-file, and UI testing. It is not the approved v1 Drift Detector
because it does not consume Weekly Drift Reviewer decisions or apply the
two-consecutive-Conflict rule.

---

## 7. Experimental Evolution Routing

The current prototype automatically calls `classify_weekly_evolution()` when
no precomputed evolution result is supplied. Eligible dimensions can therefore
produce the literal `evolution` response mode and an optional profile-update
suggestion.

This is implementation truth, not a product-scope decision. The PRD parks value
evolution outside the committed v1 contract. The prototype branch remains an
experimental compatibility path until it is either adopted explicitly or
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

These detectors operate on persisted single-pass LLM-Judge labels or VIF Critic
mean predictions. Their vote count is agreement among the six exploratory
detectors, not the five-pass LLM-Judge reference and not v1 benchmark ground
truth.
They remain useful for diagnosis and visualization but do not define the
selected runtime rule.

---

## 9. Weekly Coach Safety Behavior

The standalone Weekly Digest also has conservative fallback modes for offline
prompt testing when no upstream Drift result is supplied. Acute grief or
distress markers can route to `high_uncertainty`, while mixed or burdened weeks
can use `mixed_state` or `background_strain`.

These lexical/aggregate fallbacks are not substitutes for explicit Weekly
Drift Reviewer abstention. VIF Critic uncertainty remains offline evidence.
The fallbacks are local safety scaffolding around Weekly Digest file generation.

---

## 10. Implementation Reference

| Module | Role |
|---|---|
| `src/vif/eval.py` | Metrics for individual Journal Entries and uncertainty-aware evaluation |
| `src/vif/runtime.py` | Per-Journal-Entry and weekly VIF parquet generation |
| `src/vif/weekly_schema.py` | Shared weekly frame names and required-column validation |
| `src/vif/drift.py` | Existing weekly crash/rut/evolution/high-uncertainty router |
| `src/vif/evolution.py` | Experimental `stable`/`evolution`/`drift` classifier |
| `src/coach/runtime.py` | Offline checkpoint-to-Weekly-Digest orchestration |
| `src/demo_tool/multi_drift.py` | Six-detector exploratory comparison |
| `scripts/drift/trajectory_eda.py` | Drift-definition analysis |

---

## 11. Later Uncertainty Extensions

MC Dropout remains the practical POC default. Later options include deep
ensembles, evidential methods, conformal wrappers, and explicit embedding-space
out-of-distribution detection. The retired `twinkl-wq9p` diagnostic showed a
target-validity problem before it could establish whether uncertainty calibration
is the binding constraint. `twinkl-v8pb` correctly withheld its former
final-test score, and that population is now development-only. The next step is
the bounded candidate-confirmation study followed by a fresh final test, not a
heavier uncertainty method.
