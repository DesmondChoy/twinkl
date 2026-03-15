# Value Evolution Detection

## 1. How the System Works Today

### 1.1 Onboarding: Declaring Values

During onboarding, the user completes a 6-set **Best-Worst Scaling (BWS)**
assessment. Each set presents 4 cards; the user selects "Most like me" and
"Least like me." Raw scores are computed as
`raw_score(v) = best_count(v) - worst_count(v)` per value, then normalized to
produce a **value profile** `w_u` — a 10-dimensional weight vector covering
the Schwartz value dimensions (Security, Self-Direction, Achievement,
Benevolence, etc.) that sums to 1.0.

A mid-flow mirror (after set 3) and end-of-flow mirror allow users to refine
the profile before it is finalized.

**After onboarding, `w_u` is treated as fixed.** The VIF notation uses
`w_{u,t}` to suggest time-dependence, and the architecture describes it as
"piecewise constant, updated infrequently" — but no update mechanism exists
in the current implementation.

### 1.2 Journaling and the Critic

Once the user starts journaling:

1. Each journal entry is embedded via a frozen sentence encoder (SBERT).
2. The **state encoder** constructs a state vector from recent text embeddings,
   time gaps between entries, and `w_u`.
3. The **Critic (VIF)** — an MLP with 2-3 ReLU layers — processes this state
   and outputs per-dimension alignment estimates:
   `a_hat_{u,t} in {-1, 0, +1}^10`.

The Critic is **profile-conditioned**: the same journal text is evaluated
differently depending on which dimensions the user cares about.

### 1.3 Drift Detection

Drift detection operates on the Critic's alignment outputs using three
deterministic metrics (no separate model required):

| Metric | Formula | What It Captures |
|--------|---------|------------------|
| Per-dimension drift | `d_j = w_{u,j} * max(0, -a_hat_j)` | Profile-weighted misalignment on dimension j |
| Scalar alignment | `V = w_u^T * a_hat` | Overall alignment score |
| Directional drift | `cos_sim(w_u, mean(a_hat_week))` | Whether behavioral direction matches declared identity |

These metrics are smoothed via exponential moving averages (EMA) and tested
against threshold-based triggers:

- **Per-dimension rut:** Dimension j has weight >= 0.15, weekly EMA < -0.4
  for >= C_min consecutive weeks, with low uncertainty.
- **Global crash:** Weekly EMA of scalar alignment drops by > delta_crash
  compared to the previous week.
- **Identity drift:** Cosine similarity falls below threshold (e.g., < 0.4)
  and has decreased by > delta_cos from baseline.

### 1.4 The Coach

When drift triggers fire, the Coach responds:

- Uses RAG over full journal history to surface thematic evidence.
- Explains *why* misalignment occurred, citing specific entry snippets.
- Offers reflective prompts and micro-anchors (personalized nudges), never
  prescriptive advice.
- For positive patterns, provides occasional evidence-based acknowledgment
  without gamification.

---

## 2. What Is Potentially Lacking

The entire pipeline — drift metrics, triggers, coach messaging — treats `w_u`
as ground truth. Every comparison asks: *"Is the user's behavior consistent
with what they declared at onboarding?"*

This creates a fundamental problem: **when values genuinely evolve, the system
misinterprets growth as failure.**

### 2.1 The False-Positive Problem

Consider a user who declared Achievement as their top value during onboarding.
Six months later, after having a child, they consistently prioritize
Benevolence over career milestones. Their journal entries reflect genuine,
sustained reorientation — not a lapse in discipline.

Under the current design:

- The Critic correctly scores low alignment on Achievement, high on
  Benevolence.
- Drift detection sees sustained negative alignment on a high-weight
  dimension.
- The per-dimension rut trigger fires: weekly EMA < -0.4 for multiple
  consecutive weeks.
- The Coach surfaces: *"You seem to be losing track of Achievement."*

This is the wrong message. The user hasn't lost track of anything — their
priorities changed. Flagging this as drift undermines trust in the system and
may discourage the very growth it should support.

### 2.2 The Core Question

This is exactly what
[Issue #21](https://github.com/DesmondChoy/twinkl/issues/21) asks:

> *"Is +1 to -1 a drift, or is +1 to 0 a drift?"*

The answer depends on *how* the divergence manifests:

- A sustained, directional shift (consistently -1 on a formerly +1 dimension)
  suggests the user's relationship with that value has changed.
- A volatile, oscillating pattern (some weeks +1, some weeks -1) suggests
  the user is struggling, not evolving.
- A small shift (+1 to 0) may not be meaningful at all.

The current system cannot make this distinction. It treats all divergence from
`w_u` as drift.

### 2.3 What the Architecture Already Anticipated

The design documents acknowledge this gap without solving it:

- `w_{u,t}` notation implies time-dependence, but no update mechanism exists.
- A "60-70% weighting schedule" was proposed (gradually blending behavioral
  evidence into the profile over weeks), but was scoped out of the capstone.
- Profile drift detection for >4 weeks was noted as a future re-assessment
  trigger.
- Major life event detection was listed as future work.

The missing piece is a **classification step** that distinguishes genuine value
evolution from behavioral drift before the drift triggers fire.

---

## 3. What Value Evolution Is

Value evolution detection is a **statistical filter** that sits between the
Critic's raw alignment outputs and the drift detection triggers. It classifies
each dimension's divergence pattern into one of three categories:

### 3.1 Three-Way Classification

| Classification | Pattern | Volatility | What It Means |
|---------------|---------|------------|---------------|
| **Stable** | Behavior matches declared values (small residual) | Any | No action needed |
| **Evolution** | Sustained, directional divergence from declared values | Low | User's values may have genuinely shifted |
| **Drift** | Volatile, inconsistent divergence from declared values | High | User is struggling to live their values |

These are **mutually exclusive per dimension per analysis window**. Different
dimensions can have different classifications simultaneously — a user might be
evolving on Benevolence while drifting on Self-Direction and stable on
everything else.

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
        → DRIFT (volatile, inconsistent — high noise)
```

The key insight: **evolution and drift share the same residual magnitude but
differ in volatility.** A user who consistently scores -1 on Achievement
(low volatility) is evolving. A user who oscillates between +1 and -1 on
Achievement (high volatility) is drifting.

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

## 4. How Evolution Fits Into the Current Workflow

### 4.1 Pipeline Integration

```
                          Current Pipeline
                          ================

Journal Entry → Text Encoder → State Encoder → Critic → Alignment Scores
                                                              |
                                                              v
                                                      Drift Detection
                                                              |
                                                              v
                                                           Coach


                    Pipeline with Evolution Detection
                    =================================

Journal Entry → Text Encoder → State Encoder → Critic → Alignment Scores
                                                              |
                                                              v
                                                    Evolution Detection
                                                     /        |        \
                                                    /         |         \
                                              evolution    stable      drift
                                                 |           |           |
                                                 v           v           v
                                          Coach:        Drift         Drift
                                          "priorities   Detection     Detection
                                          shifting?"    (normal)      (normal)
                                                 |                       |
                                                 v                       v
                                           Suggest               Coach:
                                           profile              "losing
                                           update               track?"
```

The critical change: **evolution detection runs BEFORE drift triggers.**
Dimensions classified as "evolution" are excluded from the drift rules
entirely. Only "drift" and "stable" dimensions feed into the existing
crash/rut/identity-drift triggers.

### 4.2 What Changes for Each Component

| Component | Before | After |
|-----------|--------|-------|
| **Critic** | No change | No change — still produces alignment scores |
| **Evolution Detection** | Does not exist | New module: classifies per-dimension patterns |
| **Drift Detection** | Runs on all dimensions | Skips dimensions classified as "evolution" |
| **Coach** | One type of message for divergence | Two types: evolution ("priorities shifting") vs drift ("losing track") |
| **Profile `w_u`** | Fixed after onboarding | Can be updated when user confirms evolution |

### 4.3 Coach Messaging Differences

**For drift (volatile divergence):**

> *"Over the past few weeks, your entries show inconsistent alignment with
> Achievement — some weeks strongly aligned, others not. You may be losing
> track of what matters to you here. Would you like to reflect on what's
> getting in the way?"*

**For evolution (sustained directional shift):**

> *"I've noticed a sustained shift in how your entries relate to Benevolence
> — it's been consistently important in your recent reflections, more than
> your profile suggests. Your priorities may be evolving. Would you like to
> update your value profile to reflect this?"*

### 4.4 Relationship to Synthetic Data

An important clarification: **evolution detection does not require the Critic
to be trained on evolution examples.** The Critic's job is to score whether a
given journal entry aligns with a given value dimension — which is exactly
what it is trained to do on synthetic data.

Evolution detection operates entirely on the **statistical pattern** of those
scores over time (mean, standard deviation, direction). It is a post-hoc
analysis layer, not a learned model. The synthetic data does not need to
include life-changing events because:

- The Critic scores individual entries — it does not need to understand
  multi-week trends.
- Evolution detection is a deterministic statistical test on the Critic's
  outputs, not a model that needs training data.
- Validation can be done by simulating evolution patterns from existing
  synthetic scores (e.g., concatenating aligned entries from one persona phase
  with misaligned entries from another).

---

## 5. Design Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_entries` | 6 | Minimum entries in the analysis window (~3 weeks at 2 entries/week) |
| `residual_threshold` | 0.4 | Minimum \|residual\| to consider non-stable |
| `volatility_threshold` | 0.5 | Boundary between evolution (below) and drift (above) |
| `blend_rate` | 0.3 | How much behavioral evidence influences the suggested profile update (0 = keep declared, 1 = fully adopt behavioral) |

These defaults are conservative — they favor classifying dimensions as
"stable" over premature evolution or drift detection. Thresholds can be tuned
using synthetic persona experiments.

---

## 6. Limitations

- **Cold start:** Requires a sustained window of entries (minimum 6). Not
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
