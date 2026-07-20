# Value Evolution Detection

> **Scope:** Value evolution is a future product concept, not part of the
> selected Drift v1 contract. An experimental classifier is
> invoked automatically by the deprecated weekly compatibility router, but onboarding
> profile updates, user confirmation, and production Weekly Coach messaging
> are not implemented. Current Drift scope is defined in
> [`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md).

## 1. Baseline Architecture

### 1.1 Onboarding: Declaring Values

The implemented onboarding flow uses the published 11-group **Schwartz Values
Best-Worst Survey (SVBWS)** balanced design. Each group presents six neutral
descriptor cards; the user selects the Most and Least important guiding
principles. The raw Profile keeps 11 object scores, including separate
Universalism–Nature and Universalism–Social scores. A separately named product
transformation averages those facets and creates the ten-dimensional weight
vector `w_u`.

The flow has no midpoint result or value-specific card art. The end summary
uses friendly descriptions instead of Schwartz labels; `Set my compass`
confirms those descriptions as Core Values. The Profile is still not wired
into the runtime.

The synthetic runtime assigns equal mass to Core Values, with a
uniform fallback. The graded BWS profile remains an integration gap. Once a
profile exists, `w_u` is treated as fixed. The VIF notation uses
`w_{u,t}` to suggest time-dependence, and the architecture describes it as
"piecewise constant, updated infrequently" — but no update mechanism exists
in the current implementation.

### 1.2 Journaling and the VIF Critic

Once the user starts journaling:

1. Each Journal Entry is embedded via a frozen sentence encoder (SBERT).
2. The **state encoder** constructs a state vector from recent text embeddings,
   time gaps between entries, and `w_u`.
3. The **VIF Critic** — an MLP with 2-3 ReLU layers — processes this state
   and outputs per-dimension alignment estimates:
   `a_hat_{u,t} in {-1, 0, +1}^10`.

The VIF Critic is **profile-conditioned**: the same Journal Entry is evaluated
differently depending on which dimensions the user cares about.

### 1.3 Drift Detector

The repository contains a deprecated weekly crash/rut/evolution router and a
six-detector comparison interface. The selected v1 product target is different:
Drift is two consecutive Conflicts for the same Core Value. Other value
dimensions do not offset the Drift. The implemented capstone POC runtime uses
the fixed Luna-low Weekly Drift Reviewer without VIF Critic input followed by
the deterministic Drift Detector. The time-boxed capstone stops without a
fresh final test or deployment approval. The former
consensus-derived frozen benchmark is retired historical evidence, and the
six-detector comparison's vote count is not the LLM-Judge reference.

The older prototype research includes these deterministic metrics:

| Metric | Formula | What It Captures |
|--------|---------|------------------|
| Per-dimension divergence | `d_j = w_{u,j} * max(0, -a_hat_j)` | Profile-weighted Conflict on dimension j |
| Scalar alignment | `V = w_u^T * a_hat` | Overall alignment score |
| Directional divergence | `cos_sim(w_u, mean(a_hat_week))` | Whether behavioral direction matches declared identity |

The exploratory detectors smooth or threshold these metrics in different ways:

- **Per-dimension rut:** Dimension j has weight >= 0.15, weekly EMA < -0.4
  for >= C_min consecutive weeks, with low uncertainty.
- **Global crash:** Weekly EMA of scalar alignment drops by > delta_crash
  compared to the previous week.
- **Identity divergence**: Cosine similarity falls below a threshold and decreases
  from baseline; this remains an exploratory comparison, not v1 scope.

### 1.4 The Weekly Coach

The intended Weekly Coach response:

- At POC scale, reads the user's full Journal Entry history via full-context prompting to surface thematic evidence. (At production scale, this would transition to RAG — see [`docs/vif/01_concepts_and_roadmap.md` Section 4](../vif/01_concepts_and_roadmap.md#4-extensions-and-future-work).)
- Explains *why* Conflict occurred, citing specific Journal Entry snippets.
- Offers reflective prompts and micro-anchors (personalized nudges), never
  prescriptive advice.
- For positive patterns, provides occasional evidence-based acknowledgment
  without gamification.

---

## 2. What Is Potentially Lacking

The current prototype workflow — divergence metrics, triggers, and Weekly
Coach messaging — treats `w_u` as ground truth. Every comparison asks: *"Is
the user's behavior consistent with what they declared at onboarding?"*

This creates a fundamental problem: **when values genuinely evolve, the prototype
misinterprets growth as failure.**

### 2.1 The False-Positive Problem

Consider a user who declared Achievement as their top value during onboarding.
Six months later, after having a child, they consistently prioritize
Benevolence over career milestones. Their Journal Entries reflect genuine,
sustained reorientation — not a lapse in discipline.

Under the older crash/rut research path, without an adopted evolution layer:

- The VIF Critic correctly scores low alignment on Achievement, high on
  Benevolence.
- The prototype divergence classifier sees sustained negative alignment on a
  Core Value.
- The prototype per-dimension rut trigger fires: weekly EMA < -0.4 for multiple
  consecutive weeks.
- The Weekly Coach surfaces: *"You seem to be losing track of Achievement."*

This is the wrong message. The user hasn't lost track of anything — their
priorities changed. Calling this pattern Drift without checking the canonical
two-Conflict rule would undermine trust in Twinkl and may discourage the very
growth it should support.

### 2.2 The Core Question

This is exactly what
[Issue #21](https://github.com/DesmondChoy/twinkl/issues/21) asks:

> *"Is +1 to -1 a drift, or is +1 to 0 a drift?"* (historical wording)

The answer depends on *how* the divergence manifests:

- A sustained, directional shift (consistently -1 on a formerly +1 dimension)
  suggests the user's relationship with that value has changed.
- A volatile, oscillating pattern (some weeks +1, some weeks -1) suggests
  the user is struggling, not evolving.
- A small shift (+1 to 0) may not be meaningful at all.

That prototype cannot make this distinction and treats all divergence from
`w_u` as its literal `drift` mode. The selected v1 contract is narrower: it
records Drift only after two consecutive Conflicts on a Core Value, without
deciding whether that Conflict means temporary struggle or genuine value
evolution.

### 2.3 What the Architecture Already Anticipated

The design documents acknowledge this gap without solving it:

- `w_{u,t}` notation implies time-dependence, but no update mechanism exists.
- A "60-70% weighting schedule" was proposed (gradually blending behavioral
  evidence into the profile over weeks), but was scoped out of the capstone.
- Value-profile divergence for >4 weeks was noted as a future re-assessment
  trigger.
- Major life event detection was listed as future work.

The missing piece is a **classification step** that distinguishes genuine value
evolution from volatile divergence before the Drift Detector runs.

---

## 3. What Value Evolution Is

Value evolution detection is a **statistical filter** that sits between the
VIF Critic's predictions and the Drift Detector. It classifies
each dimension's divergence pattern into one of three categories:

### 3.1 Three-Way Classification

| Classification | Pattern | Volatility | What It Means |
|---------------|---------|------------|---------------|
| **Stable** | Behavior matches declared values (small residual) | Any | No action needed |
| **Evolution** | Sustained, directional divergence from declared values | Low | User's values may have genuinely shifted |
| **Volatile divergence** | Volatile, inconsistent divergence from declared values | High | User may be struggling to live their values |

These are **mutually exclusive per dimension per analysis window**. Different
dimensions can have different classifications simultaneously — a user might be
evolving on Benevolence while showing volatile divergence on Self-Direction
and remaining stable on everything else.

### 3.2 How Classification Works

For each of the 10 Schwartz dimensions, the algorithm computes three
statistics over a window of recent alignment scores:

1. **Mean alignment** (`mu_j`): The average alignment score over the window.
2. **Volatility** (`sigma_j`): The standard deviation of alignment scores.
3. **Residual** (`residual_j = mu_j - expected_j`): How much observed behavior
   deviates from what the declared profile predicts.

The **expected alignment** is derived from the profile: high-weight dimensions
expect positive alignment; low-weight dimensions expect neutral.

Classification follows a simple decision tree:

```
For each dimension j:
    if |residual_j| < residual_threshold:
        → STABLE (behavior matches declared values)
    elif volatility_j < volatility_threshold:
        → EVOLUTION (sustained, directional — low noise)
    else:
        → VOLATILE_DIVERGENCE (volatile, inconsistent — high noise)
```

The key insight: **evolution and volatile divergence share the same residual magnitude but
differ in volatility.** A user who consistently scores -1 on Achievement
(low volatility) is evolving. A user who oscillates between +1 and -1 on
Achievement (high volatility) shows volatile divergence. This prototype class
does not itself claim canonical Drift.

### 3.3 Profile Update Suggestion

When evolution is detected on one or more dimensions, the module computes a
**suggested profile update**:

1. For evolved dimensions, derive implied weight from the sustained alignment
   direction (consistently positive → increase weight; consistently negative →
   decrease weight).
2. Blend the declared profile with this behavioral evidence at a conservative
   rate (default 30% behavioral, 70% declared).
3. Re-normalize so the profile sums to 1.

The suggested profile is a recommendation, not an automatic update. In
production, the user would confirm: *"It seems your priorities around
Benevolence have grown. Would you like to update your profile?"*

---

## 4. How Evolution Fits as a Future Extension

### 4.1 Workflow Integration

```
                          Current Workflow
                          ================

Journal Entry → Text Encoder → State Encoder → VIF Critic → Predictions
                                                              |
                                                              v
                                                      Drift Detector
                                                              |
                                                              v
                                                      Weekly Coach


                    Workflow with Evolution Detection
                    =================================

Journal Entry → Text Encoder → State Encoder → VIF Critic → Predictions
                                                              |
                                                              v
                                                    Evolution Detection
                                                     /        |        \
                                                    /         |         \
                                              evolution    stable   volatile
                                                                  divergence
                                                 |           |           |
                                                 v           v           v
                                       Weekly Coach:    Drift         Drift
                                       "priorities      Detector      Detector
                                       shifting?"       (normal)      (normal)
                                                 |                       |
                                                 v                       v
                                           Suggest            Weekly Coach:
                                           profile           "losing track?"
                                           update
```

In this concept, **evolution detection runs before the Drift Detector.**
Dimensions classified as "evolution" are excluded from the Drift rule
entirely. Stable and volatile-divergence classifications continue to the
Drift Detector, which still requires two consecutive Conflicts for a Core
Value before claiming Drift.

### 4.2 What Changes for Each Component

| Component | v1 responsibility | Evolution-extension responsibility |
|-----------|-------------------|------------------------------------|
| **VIF Critic** | Produces alignment predictions and uncertainty | Unchanged |
| **Evolution Detection** | Outside committed scope | Classifies per-dimension patterns |
| **Drift Detector** | Detects Drift | Skips user-confirmed evolution classifications |
| **Weekly Coach** | Reflects active, recovered, mixed, or uncertain states | Asks whether priorities are shifting |
| **Profile `w_u`** | Fixed runtime input | Updates only after explicit user confirmation |

### 4.3 Weekly Coach Messaging Differences

**For volatile divergence:**

> *"Over the past few weeks, your Journal Entries show inconsistent alignment with
> Achievement — some weeks strongly aligned, others not. You may be losing
> track of what matters to you here. Would you like to reflect on what's
> getting in the way?"*

**For evolution (sustained directional shift):**

> *"I've noticed a sustained shift in how your Journal Entries relate to Benevolence
> — it's been consistently important in your recent reflections, more than
> your profile suggests. Your priorities may be evolving. Would you like to
> update your value profile to reflect this?"*

### 4.4 Relationship to Synthetic Data

An important clarification: **evolution detection does not require the VIF
Critic to be trained on evolution examples.** The VIF Critic's job is to score
whether a given Journal Entry aligns with a given value dimension — which is
exactly what it is trained to do on synthetic data.

Evolution detection operates entirely on the **statistical pattern** of those
scores over time (mean, standard deviation, direction). It is a post-hoc
analysis layer, not a learned model. The synthetic data does not need to
include life-changing events because:

- The VIF Critic scores individual Journal Entries — it does not need to understand
  multi-week trends.
- Evolution detection is a deterministic statistical test on the VIF Critic's
  outputs, not a model that needs training data.
- Validation can be done by simulating evolution patterns from existing
  synthetic scores (e.g., concatenating aligned Journal Entries from one
  persona phase with Conflict-labeled Journal Entries from another).

---

## 5. Design Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_entries` | 6 | Minimum entries in the analysis window (~3 weeks at 2 entries/week) |
| `residual_threshold` | 0.4 | Minimum \|residual\| to consider non-stable |
| `volatility_threshold` | 0.5 | Boundary between evolution (below) and volatile divergence (above) |
| `blend_rate` | 0.3 | How much behavioral evidence influences the suggested profile update (0 = keep declared, 1 = fully adopt behavioral) |

These defaults are conservative — they favor classifying dimensions as
"stable" over premature evolution or volatile-divergence classification.
Thresholds can be tuned using synthetic persona experiments.

---

## 6. Limitations

- **Cold start:** Requires a sustained window of Journal Entries (minimum 6). Not
  useful in the first weeks of journaling.
- **Ordinal discreteness:** Alignment scores are in {-1, 0, +1}, not
  continuous. Standard deviation on small windows of discrete values can be
  noisy.
- **Cannot distinguish evolution from confusion:** A user who is genuinely
  confused about their values may produce a sustained pattern that looks like
  evolution. User confirmation is required before updating the profile.
- **Zero-sum profile:** Re-normalizing after a blend means increasing one
  dimension's weight necessarily decreases others. This is intentional
  (attention is finite) but should be communicated to the user.
- **No automatic updates:** The suggested profile is always a recommendation
  that requires user confirmation, not an autonomous rewrite.
