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

## 1. What is drift? A unified framework

### Two independent axes

Drift detection involves two questions that must not be conflated:

**Question A — What happened in the signal?** This is observable. The Critic measures it.

| Signal pattern | Description | Example |
|---|---|---|
| **Crash** | Sharp reversal in one step — behavior contradicts the value | +1 → -1, or 0 → -1 |
| **Rut** | Sustained low over many steps | -1, -1, -1 or prolonged 0s on a core value |
| **Spike** | Single bad step that recovers | -1 then back to 0 or +1 |
| **Fade** | Gradual decline from alignment to dormancy or misalignment | +1 → +1 → 0 → 0 → -1 |

**Question B — What does it mean?** This requires user input. The Coach resolves it.

| Interpretation | Description | Example |
|---|---|---|
| **Noise** | Signal blip, no real behavioral change | One rough week, scores recover |
| **Tradeoff** | Intentional short-term sacrifice | "Gym can wait this month — deadline" |
| **Drift** | Unintended or unacknowledged departure from values | User didn't realize they stopped prioritizing family |
| **Evolution** | Intentional, endorsed change in what the user values | "I've decided career matters more than fitness now" |

These axes are **independent**. A crash (signal) could be noise, a tradeoff, or drift. A rut (signal) could be drift or evolution. There is no 1:1 mapping between signal patterns and interpretations.

### Running example: Maya

Maya completes onboarding. Her BWS results: **Benevolence (w=0.45)** and **Self-Direction (w=0.40)** are her top two values. Achievement is low (w=0.05). Her profile `w_u` is set.

Over 12 weeks of journaling, four different things happen:

**Crash — Benevolence collapses in one step (week 7)**

Maya writes about blowing up at her sister who asked for help during a stressful move. Critic scores Benevolence as -1, down from +1 the previous week.

```
Benevolence:  +1  +1  +1  +1  +1  +1  [-1]  ...
                                        ^^^
                                      crash: +1 → -1 in one step
```

Signal is strong (w=0.45 × 2.0 magnitude = 0.90). But is it drift? Depends on what happens next.

**Spike — Benevolence recovers (weeks 8-9)**

Maya writes about calling her sister to apologize and helping with the move after all. Benevolence bounces back to +1.

```
Benevolence:  +1  +1  +1  +1  +1  +1  -1  [+1  +1]  ...
                                            ^^^^^^^
                                          recovery: spike, not drift
```

The crash was real but the pattern wasn't sustained → **noise**. EMA worry rises briefly, then decays. CUSUM jar fills then drains. No alert fires.

**Fade — Self-Direction gradually goes dormant (weeks 5-12)**

Maya's journal entries slowly stop mentioning independent choices. She's not acting *against* Self-Direction — it just stops appearing.

```
Self-Direction:  +1  +1  +1  +1  [0   0   0   0   0   0   0   0]
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  fade: value goes dormant over 8 weeks
```

With the standard formula (τ_expect=0), this produces zero drift signal — scores never go negative. With the absence-aware formula (τ_expect=0.3), each 0 on a w=0.40 dimension produces `0.40 × 0.3 = 0.12` per step, accumulating in EMA/CUSUM over time.

This is the case where **absence is the signal**. Maya's cognitive filtering may be at work — she doesn't write about Self-Direction because she's avoiding the tension of not exercising it.

**Rise — Achievement emerges as a new dominant value (weeks 4-12)**

Maya starts writing about career goals, promotions, and outperforming peers. Achievement was w=0.05 at onboarding — the system isn't monitoring it.

```
Achievement:  0   0   0  [+1  +1  +1  +1  +1  +1  +1  +1  +1]
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          sustained +1 on a non-core value
```

**This is the gap in the current framework.** The detectors only watch for decline on declared core values. They don't notice a new value rising. But Maya's behavioral profile has shifted: she now acts like someone who values Achievement highly, even though her onboarding said otherwise.

This matters because:
- The Self-Direction fade and the Achievement rise may be connected — she's trading autonomy for career advancement
- The Coach should surface this: "Achievement has been a consistent theme in your recent entries, but you ranked it low during onboarding. Has something changed?"
- If Maya endorses it → evolution (update `w_u`: Achievement goes up, possibly Self-Direction goes down)
- If Maya is surprised → drift she wasn't aware of

**How to detect rising values:** Periodically compare the behavioral profile (aggregated recent `â_t` scores) against `w_u`. If a non-core dimension (w_j < w_min) shows sustained positive alignment (+1 for ≥ C_min steps), flag it as a **profile divergence** — not a crisis, but a prompt for the Coach to ask whether the user's priorities have shifted. This is separate from the crash/rut detectors and uses the same Critic output.

### Three factors connect the axes

| Factor | What it determines | Who measures it |
|---|---|---|
| **Size** | Whether the signal is worth noticing at all | Critic (magnitude of change × profile weight) |
| **Duration** | Whether it's a pattern or a blip | Drift detectors (EMA, CUSUM, thresholds) |
| **Awareness** | Whether it's drift, tradeoff, or evolution | The user, via the Coach conversation |

### The decision tree

```
Journal entry → Critic scores â_t (all 10 dimensions)
        │
        ├─── Core values (w_j ≥ w_min): watch for DECLINE
        │           │
        │           ▼
        │      Is the change big enough to notice?
        │      (size: magnitude × profile weight)
        │           │
        │       no ──┘              yes
        │       (ignore)             │
        │                            ▼
        │                      Is it sustained?
        │                      (duration: one step or a pattern?)
        │                            │
        │                    no ──┘              yes
        │                    (spike → noise)      │
        │                                         ▼
        │                                   Does the user know?
        │                                   (Coach asks)
        │                                         │
        │                                ┌────────┴────────┐
        │                                no                yes
        │                                = DRIFT           │
        │                                                  ▼
        │                                            Does the user endorse it?
        │                                                  │
        │                                          ┌───────┴───────┐
        │                                          no              yes
        │                                          = DRIFT         │
        │                                          (with denial)   ▼
        │                                                     Is it short-term?
        │                                                          │
        │                                                  ┌───────┴───────┐
        │                                                  yes             no
        │                                                  = TRADEOFF      = EVOLUTION
        │                                                  (schedule        (update w_u)
        │                                                   follow-up)
        │
        └─── Non-core values (w_j < w_min): watch for RISE
                    │
                    ▼
              Is a non-core dimension sustained positive?
              (+1 for ≥ C_min steps)
                    │
                no ──┘         yes
                (ignore)        │
                                ▼
                          PROFILE DIVERGENCE
                          Coach asks: "Achievement has been a consistent
                          theme — has something changed?"
                                │
                        ┌───────┴───────┐
                        "No, just        "Yes, it matters
                         a phase"         to me now"
                        (keep w_u)       = EVOLUTION
                                          (update w_u)
```

Duration plays two distinct roles here:

1. **Filtering noise from signal** (the size + duration gates at the top). This is the Critic + detector's job.
2. **Converting tradeoffs into self-deception** (the follow-up check). "Work is crazy this month" is reasonable at week 2. At week 16, it's a story the user is telling themselves. The system should literally schedule a follow-up and let time be the test.

### Psychological grounding: past, present, future

Values are central schemas resistant to change — people remember value-congruent events vividly and rationalize away incongruent ones (Schwartz, 2012; Rokeach, 1973). People engage with their values through three temporal orientations (Schwartz & Bilsky, 1987; Hitlin & Piliavin, 2004):

| Orientation | Question the user asks | What Twinkl provides |
|---|---|---|
| **Past** | "Did I really used to care about that?" | **Evidence-based recall.** The journal history + alignment scores over time are more reliable than the user's own memory, which the literature says is biased toward value-congruent events. The system can say: "In weeks 1-6, your Benevolence scores were consistently +1 — here are the entries." This corrects the very cognitive filtering that makes drift invisible to the person experiencing it. |
| **Present** | "How can I get what matters to me now?" | **Tension surfacing.** The Coach uses drift alerts to show the gap between declared values and current behavior, then helps the user act on it. The 5 drift detectors feed this by identifying *which* values need attention right now. |
| **Future** | "Where do I see my life heading?" | **The missing piece that distinguishes drift from evolution.** When the Coach surfaces a drift, it asks: "Is this something you want to address, or has your thinking shifted?" The user's answer *is* the future question — it reveals whether their aspirational trajectory still matches the profile. If not, the profile updates (evolution). If so, the drift is confirmed and monitoring continues. |

The past orientation is where Twinkl is strongest: it provides objective behavioral evidence that counters the user's natural tendency to rationalize away value-incongruent behavior. The present orientation is the Coach's core function. The future orientation is handled through Coach conversation rather than a formal model input — when drift is detected, the user's response about whether they want to change reveals their aspirational direction.

This maps onto the decision tree: the **past** question feeds the "is the change big enough?" gate (historical evidence), the **present** question feeds the "is it sustained?" gate (current pattern), and the **future** question feeds the "does the user endorse it?" gate (aspirational direction).

### Absence as signal

Because users cognitively filter value-incongruent experiences from their writing, a core value going dormant (sustained 0 on a high-weight dimension) may indicate suppressed drift, not true neutrality. Journal text is already filtered through the user's value-protection schema — a user who values Benevolence but neglected a friend might write "I was so busy with the project" (reframing as Achievement) rather than "I ignored my friend."

This means: if the Critic scores a -1 on a core value, that's a *strong* signal — the misalignment survived the user's own cognitive filtering. A sustained 0 on a core value that was previously active may be soft drift that the user has rationalized away.

---

## 2. Which transitions matter, and how much?

### Signal strength depends on three things

Every alignment score transition has a signal strength determined by: what changed, how much the user cares, and how expected alignment is defined.

```
signal_strength = w_j × |change from expected|
```

#### Which transitions produce signal

| Transition | On core value (w=0.5) | On peripheral (w=0.05) | Signal type |
|---|---|---|---|
| **+1 → -1** | 1.0 (strong) | 0.1 (ignore) | Crash |
| **0 → -1** | 0.5 (moderate) | 0.05 (ignore) | Crash (emerging) |
| **+1 → 0** | 0.5 (moderate) | 0.05 (ignore) | Fade / dormancy onset |
| **0 → 0 → 0 → 0** (was +1) | Depends on formula | Ignore | Sustained dormancy |
| **-1 → +1** or **0 → +1** | Recovery (reduces signal) | n/a | Drains EMA/CUSUM |

The key design choice is **what counts as the expected baseline**:

- **Standard formula (τ_expect = 0):** `d_j = w_j × max(0, -â_j)`. Only negative scores produce signal. +1 → 0 generates zero drift signal. This is the current spec.
- **Absence-aware formula (τ_expect > 0):** `d_j = w_j × max(0, τ_expect - â_j)`. A core value *should* show up positively; its absence is a soft warning. +1 → 0 on a core value (w=0.5, τ=0.3) produces 0.15. +1 → -1 produces 0.65.

Which formula to use is an empirical question — one of the things the experiment tests.

#### Summary: what matters

- **+1 → -1 on a core value:** Always matters. Full reversal. Strong crash signal regardless of approach.
- **+1 → 0 on a core value:** Matters if sustained (rut/fade) or if using absence-aware formula. A single step of dormancy is ambiguous.
- **0 → -1 on a core value:** Moderate. New misalignment with no positive baseline to fall from.
- **Sustained 0s on a previously active core value:** Matters. The value has gone dormant — possibly suppressed by cognitive filtering.
- **Any transition on a low-weight dimension:** Ignore. The user didn't declare it as important.

### Proposed threshold ranges for grid search

| Parameter | Symbol | Range | Rationale |
|---|---|---|---|
| Profile weight floor | w_min | {0.10, 0.15, 0.20} | Below this, the dimension is not monitored. 0.15 = a 2-value persona's secondary value barely qualifies. |
| Expected alignment floor | τ_expect | {0.0, 0.2, 0.3} | 0.0 = standard (negatives only); 0.2-0.3 = absence-aware. |
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

### Step 2: Score all 6 approaches against ground truth (5 candidates + baseline)

The existing dual-trigger rules from [`drift_detection_eval.md`](drift_detection_eval.md) and [`04_uncertainty_logic.md`](../vif/04_uncertainty_logic.md) are the simplest possible implementation and serve as **Approach 0 (Baseline)**:

| Trigger | Rule | Parameters |
|---|---|---|
| **Sudden Crash** | `V_{t-1} - V_t > δ_j` (single-step drop exceeds threshold) | δ_j = 0.5 |
| **Chronic Rut** | `V_t < τ_low` for ≥ C_min consecutive steps | τ_low = -0.4, C_min = 3 |

Both gated by uncertainty: only fire when `σ_t < ε_j` (ε_j = 0.3).

The 5 candidate approaches (EMA, CUSUM, Cosine, Control Charts, KL Divergence) are compared against this baseline. The experiment must answer: **does the added complexity of a stateful detector improve hit rate or reduce FPR enough to justify it over simple thresholds?**

If the baseline already meets targets, the simpler approach wins — no EMA/CUSUM needed. If it doesn't, we know specifically where it falls short (e.g., misses slow ruts, too many crash false alarms) and which candidate addresses that gap.

Run each detector on the 24 annotated personas (using human annotation means as "perfect Critic" input). For each approach × parameter combination, compute:

| Metric | Definition | Target |
|---|---|---|
| **Hit Rate** | Fraction of crisis labels correctly flagged | ≥ 80% |
| **Precision** | Fraction of alerts that correspond to real crises | > 60% |
| **F1** | Harmonic mean of hit rate and precision | > 0.5 |
| **FPR** | Fraction of non-crisis steps incorrectly flagged | < 20% |
| **First-alert latency** | Steps after crisis onset before first alert | Lower is better |

### Step 3: Re-run with Critic predictions

Replace human annotations with `critic.predict_with_uncertainty()` output. This measures degradation from Critic noise. The gap between Step 2 (human annotations) and Step 3 (Critic predictions) quantifies how much Critic accuracy improvement would improve drift detection.

Importantly, Step 3 enables uncertainty gating — the baseline's `σ_t < ε_j` constraint can only be tested with MC Dropout predictions, not human annotations (which have no uncertainty estimate). This may change the relative ranking: an approach that performs well on clean data but is sensitive to noise may lose to one that is more robust.

### Step 4: Select approach combination

The dual-trigger design requires one crash detector + one rut detector. Use Step 2-3 results to select:

- **Crash trigger:** The approach with highest precision on `crisis_type=crash` labels (few false alarms on sudden events). Candidates: Baseline simple threshold, Control Charts (LCL breach), or Cosine Similarity (instant direction reversal).
- **Rut trigger:** The approach with highest hit rate on `crisis_type=rut` labels (catch slow decay even if some false positives). Candidates: Baseline consecutive-low counter, EMA, or CUSUM.

If the baseline wins both categories, the answer is: simple thresholds with uncertainty gating are sufficient for the POC. The more complex approaches are documented as future-work options for when longer user histories make them more valuable.

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

The past/present/future temporal orientations map onto this lifecycle:

| Orientation | System component | What it provides |
|---|---|---|
| **Past** ("Did I really used to care about that?") | Journal history + stored alignment scores | Evidence-based behavioral record that is more reliable than user recall |
| **Present** ("How can I get what matters to me now?") | Drift detection + Coach conversation | Identifies the current gap and helps the user act on it |
| **Future** ("Where do I see my life heading?") | Coach follow-up question when drift is detected | User's response reveals whether aspirational trajectory still matches `w_u` |

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
2. **Coach-mediated evolution:** When the Coach surfaces a drift and the user responds with "yes, my thinking has changed," the Coach proposes a specific weight adjustment. The user confirms before it takes effect. This is the **future** question in action — the user is declaring where they see their values heading.
3. **Periodic re-assessment:** Every N weeks (e.g., 12), offer an optional BWS re-assessment. Compare new weights to old. Significant changes are flagged as potential evolution for the user to reflect on. This directly addresses the **past** question — "here's what you said 12 weeks ago, here's what you say now."

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

1. Run baseline (simple crash/rut thresholds) + 5 candidate approaches × parameter grid on human annotation means
2. Score each against crisis labels (hit rate, precision, F1, FPR, latency)
3. If baseline meets targets (≥80% hit rate, <20% FPR): stop here — simple thresholds win
4. If not: identify where baseline fails, select best crash approach and best rut approach from candidates
5. Test the combined dual-trigger system

**Output:** Selected approach pair with tuned parameters, or confirmation that baseline suffices. Documented in experiment log.

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
- Schwartz, S. H., & Bilsky, W. (1987). Toward a universal psychological structure of human values. *Journal of Personality and Social Psychology*, 53(3), 550-562.
- Hitlin, S., & Piliavin, J. A. (2004). Values: Reviving a dormant concept. *Annual Review of Sociology*, 30, 359-393.
- Rokeach, M. (1973). *The Nature of Human Values*. Free Press.
- Bardi, A., & Goodwin, R. (2011). The dual route to value change. *Journal of Cross-Cultural Psychology*, 42(2), 271-287.
