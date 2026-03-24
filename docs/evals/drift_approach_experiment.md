# Drift Detection Approach Experiment

## Status: 📋 Specified

This document answers five design questions about how Twinkl should detect, define, and validate value drift. It bridges the conceptual framework (what is drift?) with the empirical plan (how do we pick an approach?).

### Related Documents

- [`06_profile_conditioned_drift_and_encoder.md`](../vif/06_profile_conditioned_drift_and_encoder.md) — drift formulas and profile interaction
- [`04_uncertainty_logic.md`](../vif/04_uncertainty_logic.md) — dual-trigger rules and MC Dropout
- [`drift_detection_eval.md`](drift_detection_eval.md) — evaluation protocol and success criteria
- [`value_modeling_eval.md`](value_modeling_eval.md) — Critic accuracy (upstream dependency)
- [`notebooks/annotations/drift_detection_comparison.ipynb`](../../notebooks/annotations/drift_detection_comparison.ipynb) — 5-approach comparison on annotation data

---

## 1. How do we determine drift from initial persona values?

### The conceptual frame

Values are central schemas resistant to change — people remember value-congruent events vividly and rationalize away incongruent ones (Schwartz, 2012; Rokeach, 1973). This means:

- **Default assumption:** the declared value profile holds. Any sustained behavioral departure is drift until the user says otherwise.
- **Drift vs. evolution:** The distinction is not temporal (short-term vs. long-term) but **awareness + endorsement**. Drift = the user would not endorse the change if shown evidence. Evolution = the user explicitly acknowledges and accepts the shift.
- **The system detects; the user classifies.** The Critic + drift detectors surface misalignment. The Coach asks: "Is this something you want to address, or has your thinking shifted?" The user's answer determines whether the profile updates.

### What "drift from initial values" means operationally

The onboarding BWS flow produces a 10-dimensional weight vector `w_u` (summing to 1.0) and a goal category. Drift is measured relative to this baseline:

1. **Profile-weighted misalignment:** For each journal entry, the Critic produces `â_t ∈ [-1, 1]^10`. Drift signal per dimension: `d_j = w_j × max(0, -â_j)`. Only dimensions the user cares about (w_j ≥ w_min) are monitored.
2. **Temporal accumulation:** A single misaligned entry is noise. Sustained misalignment is signal. The five candidate approaches (EMA, CUSUM, Cosine, Control Charts, KL Divergence) each define "sustained" differently.
3. **Absence as signal:** Because users cognitively filter value-incongruent experiences from their writing, a core value going dormant (sustained 0 on a high-weight dimension) may indicate suppressed drift, not true neutrality. This motivates a shortfall-from-expected variant alongside the standard negative-only formula.

### Duration as a separate axis

Duration does not distinguish drift from evolution — but it does distinguish noise from real signals, and it converts legitimate tradeoffs into self-deception:

| | Short-term | Long-term |
|---|---|---|
| **Unaware** | Noise (one bad week) | Drift |
| **Aware, not endorsing** | Legitimate tradeoff ("sacrificing gym for this deadline") | Rationalized drift ("I'll get back to it... eventually") |
| **Aware, endorsing** | Experiment | Evolution (update profile) |

The detection layer handles the duration question. The Coach conversation handles the awareness/endorsement question. Together, they classify the signal.

---

## 2. Is +1 → -1 a drift, or is +1 → 0? What magnitude of weighted change constitutes drift?

### Every transition can be meaningful — the question is severity and context

| Transition | What it means | Drift signal strength |
|---|---|---|
| **+1 → -1** | Full reversal: behavior now contradicts the declared value | **Strong.** This is the Sudden Crash trigger from the eval spec (δ = 0.5 starting value, and a +1 → -1 drop = 2.0, far exceeding it). |
| **+1 → 0** | Value goes dormant: no longer expressed but not contradicted | **Moderate.** Could be a real fade (dilution) or the user simply didn't write about that value this week. Context-dependent. |
| **0 → -1** | Value now actively contradicted where it was previously absent | **Moderate.** New misalignment emerging, but no prior positive baseline to "fall from." |
| **+1 → +1 → +1 → 0 → 0 → 0** | Slow fade over many steps | **Moderate-to-strong depending on duration.** This is the Chronic Rut scenario: not a crash, but a pattern. |
| **0 → 0 → 0 → 0** | Sustained dormancy on a core value | **Weak-to-moderate.** Depends on whether the value was ever active. If the user declared it in onboarding but it never shows up in journal behavior, that's itself a signal. |

### Profile weighting changes the calculus

The same raw transition has different severity depending on the user's profile:

```
drift_signal = w_j × max(0, τ_expect - â_j)
```

Where `τ_expect` is the expected alignment floor:
- **Standard formula (τ_expect = 0):** Only negative scores produce signal. A +1 → 0 transition generates zero drift signal. This is the current spec.
- **Absence-aware formula (τ_expect > 0, e.g., 0.3):** A +1 → 0 transition on a core value (w_j = 0.5) produces `0.5 × 0.3 = 0.15` — a soft warning. A +1 → -1 produces `0.5 × 1.3 = 0.65` — a strong signal.

**Which formula to use is an empirical question** — one of the things the experiment should test.

### Proposed threshold ranges for grid search

| Parameter | Symbol | Range | Rationale |
|---|---|---|---|
| Profile weight floor | w_min | {0.10, 0.15, 0.20} | Below this, the dimension is not monitored. 0.15 = a 2-value persona's secondary value barely qualifies. |
| Expected alignment floor | τ_expect | {0.0, 0.2, 0.3} | 0.0 = standard (negatives only); 0.2–0.3 = absence-aware. |
| EMA blending factor | α | {0.2, 0.3, 0.4} | Higher = more reactive, lower = more forgiving of isolated bad weeks. |
| EMA alert threshold | ema_thresh | {0.08, 0.10, 0.15} | The worry level that fires an alert. |
| CUSUM allowance | k | {0.2, 0.3, 0.4} | How much a neutral score drains the jar. |
| CUSUM alarm level | h | {1.0, 1.5, 2.0} | Total evidence needed to fire. |
| Crash threshold | δ | {0.5, 1.0, 1.5} | Minimum single-step drop. +1 → 0 = 1.0, +1 → -1 = 2.0. |
| Rut duration | C_min | {2, 3, 4} | Consecutive steps below τ_low needed for rut. |

---

## 3. How do we use existing synthetic data to determine the right approach?

### What we have

- **204 personas** (1,651 entries) with declared core values and judge-labeled alignment scores in `logs/judge_labels/judge_labels.parquet`
- **24 personas with human annotations** (380 annotations from 3 annotators) in `logs/annotations/`
- **Qualitative drift narratives** for each annotated persona in `notebooks/annotations/persona_drift.ipynb`
- **Trained Critic** (median QWK 0.362, recall_-1 0.313) producing noisy but signal-bearing predictions

### Step 1: Create ground truth crisis labels from existing annotations

The persona drift notebook already contains narrative analysis identifying drift patterns. Codify these into structured labels:

```
For each (persona_id, t_index, dimension):
    is_crisis: bool        # Is this a drift/crisis point?
    crisis_type: str       # "crash" | "rut" | "dormancy" | null
    confidence: str        # "clear" | "ambiguous"
```

Source these from:
- **Narrative readings** (e.g., "Lukas Bergmann: conformity collapses steps 0→2" → conformity is crisis at t=1, t=2, type=crash)
- **Inter-annotator agreement** — high-agreement misalignment is a confident crisis; high-disagreement is ambiguous
- **Profile context** — only label as crisis if the dimension is a declared core value (w_j ≥ w_min)

This gives us ~50-100 labeled (persona, step, dimension) tuples — small but sufficient for approach selection.

### Step 2: Score all 5 approaches against ground truth

Run each detector on the 24 annotated personas (using human annotation means as "perfect Critic" input). For each approach × parameter combination, compute:

| Metric | Definition | Target |
|---|---|---|
| **Hit Rate** | Fraction of crisis labels correctly flagged | ≥ 80% |
| **Precision** | Fraction of alerts that correspond to real crises | > 60% |
| **F1** | Harmonic mean of hit rate and precision | > 0.5 |
| **FPR** | Fraction of non-crisis steps incorrectly flagged | < 20% |
| **First-alert latency** | How many steps after the crisis begins before first alert | Lower is better |

### Step 3: Re-run with Critic predictions

Replace human annotations with `critic.predict_with_uncertainty()` output. This measures degradation from Critic noise. The gap between Step 2 (human annotations) and Step 3 (Critic predictions) quantifies how much Critic accuracy improvement would improve drift detection.

### Step 4: Select approach combination

The dual-trigger design requires one crash detector + one rut detector. Use Step 2-3 results to select:

- **Crash trigger:** The approach with highest precision on `crisis_type=crash` labels (we want few false alarms on sudden events). Likely candidates: Control Charts (LCL breach) or Cosine Similarity (instant direction reversal).
- **Rut trigger:** The approach with highest hit rate on `crisis_type=rut` labels (we want to catch slow decay even if some false positives). Likely candidates: EMA or CUSUM (both designed for sustained accumulation).

---

## 4. How do onboarding and Critic responses update user state and define drift?

### State lifecycle

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Onboarding │     │  Journal     │     │  Drift        │     │  Coach       │
│  (BWS)      │────▶│  Entry       │────▶│  Detection    │────▶│  Conversation│
│             │     │  + Critic    │     │  Layer        │     │              │
└─────────────┘     └──────────────┘     └───────────────┘     └──────┬───────┘
      │                    │                     │                     │
      ▼                    ▼                     ▼                     ▼
  w_u initialized    â_t produced         Alert fired?          User classifies:
  (10-dim weights)   (10-dim alignment    (crash/rut/none)      drift / tradeoff /
                      + uncertainty)                            evolution
                                                                      │
                                                    ┌─────────────────┤
                                                    ▼                 ▼
                                              Keep monitoring    Update w_u
                                              (maybe schedule    (profile evolves)
                                               follow-up)
```

### When each component writes to user state

| Component | Writes | When |
|---|---|---|
| **Onboarding BWS** | `w_u` (initial weight vector), goal category, confidence estimate | Once, at registration |
| **Critic** | `â_t` (alignment vector), `σ_t` (uncertainty vector) per entry | Every journal entry |
| **Drift detectors** | Internal state (EMA accumulators, CUSUM jars, baseline stats) | Every entry, updated incrementally |
| **Coach** | Profile update flag + optional `w_u` revision | Only when user endorses a value shift in conversation |

### Profile update rules

The profile `w_u` is **not** automatically updated by the Critic or drift detectors. It changes only through user-endorsed actions:

1. **Explicit profile edit:** User directly adjusts their value weights (settings UI).
2. **Coach-mediated evolution:** When the Coach surfaces a drift and the user responds with "yes, my thinking has changed," the Coach proposes a specific weight adjustment. The user confirms before it takes effect.
3. **Periodic re-assessment:** Every N weeks (e.g., 12), offer an optional BWS re-assessment. Compare new weights to old. Significant changes are flagged as potential evolution for the user to reflect on.

**What does NOT update the profile:**
- Critic scores alone (noisy, one-directional)
- Drift alerts alone (these are questions to the user, not answers)
- Sustained misalignment without user acknowledgment (this is the definition of drift — the system keeps flagging it)

### How this defines drift

Drift is operationally defined as: **the gap between `w_u` (what the user says they value) and the temporal trend of `â_t` (what the Critic observes in their behavior), sustained beyond noise thresholds, and not yet endorsed by the user as intentional.**

The moment the user endorses the change, `w_u` updates and the gap closes — not because behavior changed, but because the declared values caught up to reality. This is evolution, and the drift signal resets.

---

## 5. Next steps: empirical experiment plan and success metrics

### Phase 1: Ground truth labeling (1-2 days)

**Goal:** Create crisis labels from existing annotation narratives.

1. Review the 24 annotated personas in `persona_drift.ipynb`
2. For each persona with ≥5 steps, label each (t_index, dimension) as crisis/non-crisis
3. Tag crisis type (crash, rut, dormancy) and confidence (clear, ambiguous)
4. Store as `logs/annotations/crisis_labels.parquet`

**Output:** ~50-100 labeled crisis points across ~15 personas.

### Phase 2: Approach selection on human annotations (1-2 days)

**Goal:** Find the best crash trigger + rut trigger combination.

1. Run all 5 approaches × parameter grid on human annotation means
2. Score each against crisis labels (hit rate, precision, F1, FPR, latency)
3. Select best crash approach and best rut approach
4. Test the combined dual-trigger system

**Output:** Selected approach pair with tuned parameters. Documented in experiment log.

### Phase 3: Critic-in-the-loop evaluation (1 day)

**Goal:** Measure degradation when using Critic predictions instead of human annotations.

1. Run the selected dual-trigger system on Critic predictions for the same 24 personas
2. Compare hit rate / precision / FPR against Phase 2 results
3. Quantify the "Critic noise penalty" — how much does detection degrade?

**Output:** Gap analysis showing which Critic improvements would most benefit drift detection.

### Phase 4: Absence-aware variant (0.5 day)

**Goal:** Test whether the τ_expect > 0 formula improves rut/dormancy detection.

1. Re-run Phase 2 with `τ_expect ∈ {0.0, 0.2, 0.3}` as an additional grid parameter
2. Compare specifically on `crisis_type=dormancy` labels
3. If it improves dormancy detection without increasing FPR, adopt it

**Output:** Decision on whether to include the absence-aware formula.

### Success metrics

| Metric | Target | Measured on |
|---|---|---|
| **Hit Rate** (crisis weeks correctly flagged) | ≥ 80% | Phase 2 (human annotations) |
| **Precision** (alerts that are real crises) | > 60% | Phase 2 |
| **F1 per value dimension** | > 0.5 | Phase 2 |
| **FPR** (false alarm rate) | < 20% | Phase 2 |
| **Critic noise penalty** (hit rate drop from Phase 2 → Phase 3) | < 15pp | Phase 3 |
| **First-alert latency** | ≤ 2 steps after crisis onset | Phase 2 |

### Blocking dependency

The drift detection eval spec notes that the Critic frontier (median QWK 0.362) is "not yet strong enough for reliable automated drift triggers." Phase 3 will quantify exactly how much this matters. If the Critic noise penalty exceeds 15 percentage points on hit rate, further Critic improvement should be prioritized before productionizing drift detection.

---

## References

- Schwartz, S. H. (2012). An Overview of the Schwartz Theory of Basic Values. *Online Readings in Psychology and Culture*, 2(1).
- Rokeach, M. (1973). *The Nature of Human Values*. Free Press.
- Bardi, A., & Goodwin, R. (2011). The dual route to value change. *Journal of Cross-Cultural Psychology*, 42(2), 271-287.
