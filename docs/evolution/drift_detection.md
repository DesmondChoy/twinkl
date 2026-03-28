# Drift Detection

## 1. What is drift?

### Definition

Drift is operationally defined as: **the gap between `w_u` (what the user says they value) and the temporal trend of `â_t` (what the Critic observes in their behavior), sustained beyond noise thresholds, and not yet endorsed by the user as intentional.**

The moment the user endorses the change, `w_u` updates and the gap closes — not because behavior changed, but because the declared values caught up to reality. This is evolution, and the drift signal resets.

Drift detection operates on **weekly aggregates**, not individual entries. If a user writes 5 entries in a week with Benevolence scores [+1, +1, -1, +1, +1], the weekly average is +0.6 — no crash. This smooths within-week noise. Skipped weeks produce no data point; the system waits for the next entry without imputing.

### Two independent axes

Drift detection involves two questions that must not be conflated:

**Question A — What happened in the signal?** This is observable. The Critic measures it. Drift detection cares about **transitions**, not states.

**Question B — What does it mean?** This requires user input. The Coach resolves it.

| Interpretation | Description | Example |
|---|---|---|
| **Noise** | Signal blip, no real behavioral change | One rough week, scores recover (spike) |
| **Tradeoff** | Intentional short-term sacrifice, user is aware | "Gym can wait this month — deadline" |
| **Drift** | Unintended or unacknowledged departure from values | User didn't realize they stopped prioritizing family |
| **Evolution** | Intentional, endorsed change in what the user values | "I've decided career matters more than fitness now" |

These axes are **independent**. A crash (signal) could be noise, a tradeoff, or drift. A no-recovery (signal) could be drift or evolution. There is no 1:1 mapping between signal patterns and interpretations.

### Core vs peripheral values

**Core values** (`w_j ≥ w_min`) are monitored for decline. **Peripheral values** (`w_j < w_min`) are only monitored for rise (a new value emerging). If Power is `w=0.05` and scores -1, that's not a drift alert — the user didn't declare it as important.

### Interpretations

Four factors connect the two axes:

| Factor | What it determines | Who measures it |
|---|---|---|
| **Confidence** | Whether the Critic's score is trustworthy at all | MC Dropout uncertainty (σ < ε_j) |
| **Size** | Whether the signal is worth noticing | Critic (magnitude of change × profile weight) |
| **Duration** | Whether it's a pattern or a blip | Drift detectors (EMA, CUSUM, thresholds) |
| **Awareness** | Whether it's drift, tradeoff, or evolution | The user, via the Coach conversation |

Confidence comes first. An uncertain score must not enter the size or duration gates — it pollutes detector state (EMA accumulators, CUSUM jars) with noise.

### Psychological grounding

Values are central schemas resistant to change — people remember value-congruent events vividly and rationalize away incongruent ones (Schwartz, 2012; Rokeach, 1973). Users engage with values through three temporal orientations:

| Orientation | Question the user asks | What Twinkl provides |
|---|---|---|
| **Past** | "Did I really used to care about that?" | Evidence-based recall. Alignment scores over time are more reliable than user memory, which is biased toward value-congruent events. |
| **Present** | "How can I get what matters to me now?" | Tension surfacing. The Coach uses drift alerts to show the gap between declared values and current behavior. |
| **Future** | "Where do I see my life heading?" | The Coach asks: "Is this something you want to address, or has your thinking shifted?" — the user's answer reveals whether aspirational trajectory still matches the profile. |

This maps onto the decision tree: the **past** question feeds the size gate (historical evidence), the **present** question feeds the duration gate (current pattern), and the **future** question feeds the awareness gate (aspirational direction).

### Absence as signal

Because users cognitively filter value-incongruent experiences from their writing, a core value going dormant (sustained 0 on a high-weight dimension) may indicate suppressed drift, not true neutrality. Journal text is already filtered through the user's value-protection schema — a user who values Benevolence but neglected a friend might write "I was so busy with the project" (reframing as Achievement) rather than "I ignored my friend."

This means: if the Critic scores a -1 on a core value, that's a *strong* signal — the misalignment survived the user's own cognitive filtering. A sustained 0 on a core value that was previously active may be soft drift the user has rationalized away.

---

## 2. Signal patterns

### Core value patterns

| Signal pattern | What happened | Numbers | Coach framing |
|---|---|---|---|
| **Crash** | Sharp transition to misalignment in one step | `+1 → -1` or `0 → -1` | "Something shifted this week" |
| **Fade** | Gradual decline to dormancy over multiple steps | `+1 → +1 → 0 → 0 → 0` | "This value has been quietly fading" |
| **Spike** | Temporary dip that recovers within 1-2 steps | `+1 → -1 → +1` | No alert (or retract) |
| **No recovery** | Score stays negative/dormant after crash or fade for ≥ C_min steps | `+1 → -1 → -1 → -1 → -1` | "This has been going on for N weeks — it's not temporary" |
| **Onboarding gap** | Core value was never aligned from day one | `-1 → -1 → -1` from week 1 | "You ranked this as important, but it hasn't appeared in your journal yet" |

### Peripheral value patterns

| Signal pattern | What happened | Numbers | Coach framing |
|---|---|---|---|
| **Rise** | Sustained positive alignment on a non-core dimension | `0 → 0 → +1 → +1 → +1 → +1` | "Achievement has been a consistent theme — has something changed?" |

### Transition permutation table

Every pair of consecutive weekly scores produces one of 9 transitions.

**Core values (`w_j ≥ w_min`):** monitored for decline and dormancy.

| From → To | Δ | Signal name | Action | Notes |
|---|---|---|---|---|
| **+1 → +1** | 0 | Consistency | None | Aligned and stable. Ideal state. |
| **+1 → 0** | -1 | Fade onset | Soft signal | One step is ambiguous. If sustained (≥ C_min 0s), becomes fade. With absence-aware formula (τ_expect > 0), produces `w_j × τ_expect` per step. |
| **+1 → -1** | -2 | Crash | Alert | Full reversal. Strongest single-step signal: `w_j × (1 + τ_expect)` or `w_j × 2` depending on formula. |
| **0 → +1** | +1 | Recovery / activation | Drains detectors | If recovering from fade: EMA decays, CUSUM jar drains. If first-ever +1: late activation (not drift-relevant). |
| **0 → 0** | 0 | Dormancy | Soft signal (if sustained) | Single step: nothing. Sustained after prior +1: fade continues. Sustained from day one: onboarding gap. With τ_expect > 0, each step adds `w_j × τ_expect`. |
| **0 → -1** | -1 | Emerging crash | Alert (moderate) | Transition from dormancy to active misalignment. Weaker than +1→-1 but still actionable. |
| **-1 → +1** | +2 | Sharp recovery | Drains detectors | Full reversal upward. Resets no-recovery counter. May follow a spike pattern if crash was recent. |
| **-1 → 0** | +1 | Partial recovery | Drains detectors (partial) | Moving toward neutral. Not yet aligned but improving. |
| **-1 → -1** | 0 | No recovery | Accumulates | Each step adds to no-recovery counter. After ≥ C_min steps: escalation alert. |

**Peripheral values (`w_j < w_min`):** monitored only for rise.

| From → To | Signal name | Action | Notes |
|---|---|---|---|
| **0 → +1** | Rise onset | Watch | Start counting. Not flagged until sustained ≥ C_min steps. |
| **+1 → +1** | Rise continues | Watch / flag | If count ≥ C_min: profile divergence alert. Coach asks if priorities shifted. |
| **Any → 0 or -1** | — | None | Not monitored. User didn't declare this value as important. |
| **-1 → -1** | — | None | Sustained negative on a peripheral value is not a concern. |

### Multi-step signal rules

Single transitions are raw observations, not signals. No single transition triggers a final interpretation by itself. **Signals are multi-step patterns** built from sequences of transitions.

**Core values:**

| Signal | Required sequence | Min steps | Trigger condition |
|---|---|---|---|
| **Crash** | Contains +1→-1 or 0→-1 | 2 | Single-step drop passes size gate: `w_j × |Δ| > δ_j`. Alert fires immediately but interpretation waits for next steps. |
| **Spike** | Crash → recovery within 1-2 steps | 3-4 | Score returns to +1 or 0 within 2 steps after crash. Retracts crash alert → noise. |
| **Fade** | +1→0 → sustained 0s | ≥ 1 + C_min | C_min consecutive confident 0s on a previously active dimension. |
| **No recovery** | Crash → sustained -1s | ≥ 1 + C_min | After crash, score stays < τ_low for ≥ C_min consecutive confident steps. Escalation alert. |
| **Onboarding gap** | -1 or 0 from day one, never +1 | ≥ 3 | Core value with no positive confident score in first N entries. |

**Peripheral values:**

| Signal | Required sequence | Min steps | Trigger condition |
|---|---|---|---|
| **Rise** | 0→+1 or -1→+1 → sustained +1s | ≥ C_min | C_min consecutive confident +1 scores on a dimension with `w_j < w_min`. Profile divergence alert. |

### Maya running example

Maya completes onboarding. Her BWS results produce:

- **Core values:** Benevolence (w=0.45), Self-Direction (w=0.40)
- **Peripheral values:** Achievement (w=0.05) and 7 others

**Crash — Benevolence drops (week 7):**

Maya blows up at her sister. The Critic runs 50 forward passes:

- **Scenario A (confident):** mean = -1, σ = 0.08. All 50 sub-networks agree. Passes Gate 1 (σ < ε=0.3). Enters detectors.
- **Scenario B (uncertain):** mean = -0.3, σ = 0.55. Sub-networks disagreed. Gate 1 blocks this score. The Coach asks a clarifying question instead.

```
Benevolence (core, w=0.45):

Week:        1    2    3    4    5    6    7
Score:      +1   +1   +1   +1   +1   +1   -1
                                          ^^^
                                        crash: drop of 2.0, confident
```

**Spike — Benevolence recovers (weeks 8-9):**

Maya apologizes and helps her sister. EMA worry rises then decays. CUSUM jar fills then drains. No alert fires.

```
Week:        1    2    3    4    5    6    7    8    9
Score:      +1   +1   +1   +1   +1   +1   -1   +1   +1
                                               ^^^^^^^
                                             recovery → spike (noise)
```

**Fade — Self-Direction goes dormant (weeks 5-12):**

Maya's entries slowly stop mentioning independent choices. The value doesn't score -1 — it stops appearing.

```
Self-Direction (core, w=0.40):

Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:      +1   +1   +1   +1    0    0    0    0    0    0    0    0
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 confident 0s → genuine dormancy
```

With the absence-aware formula (τ_expect=0.3), each confident 0 produces `0.40 × 0.3 = 0.12` drift signal, accumulating over 8 weeks.

**No recovery — Benevolence never comes back:**

```
Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:      +1   +1   +1   +1   +1   +1   -1   -1   -1   -1   -1   -1
                                          ^^^  ─────────────────────────
                                        crash  no recovery for 5 weeks
                                               → escalate at week 7 + C_min
```

**Onboarding gap — Benevolence was never aligned:**

```
Week:        1    2    3    4    5    6    7    8    9   10
Score:      -1   -1   -1   -1   -1   -1   -1   -1   -1   -1
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            no transition ever happened. this is not drift.
```

After ~3 entries: "You ranked Benevolence as your top value, but it hasn't appeared positively in your journal entries yet."

**Rise — Achievement emerges (weeks 4-12):**

Maya starts writing about career goals and promotions. Achievement was w=0.05 at onboarding — not monitored for crash/fade.

```
Achievement (peripheral, w=0.05):

Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:       0    0    0   +1   +1   +1   +1   +1   +1   +1   +1   +1
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              sustained +1 on a non-core value → rise
```

The Self-Direction fade and Achievement rise may be connected. The Coach surfaces both: "Self-Direction has been absent from your entries for 8 weeks, while Achievement has been a consistent theme. Has something changed?" If Maya endorses it → evolution (update `w_u`). If Maya is surprised → drift she wasn't aware of.

---

## 3. Detection pipeline

### Four gates

```
Journal entry → Critic runs 50 forward passes (MC Dropout)
      │         produces â_t (mean) and σ_t (uncertainty) per dimension
      │
      ▼
 GATE 1: Is the Critic confident? (σ_t < ε_j per dimension)
      │
  no ──┘                              yes
  │                                    │
  ▼                                    ▼
DEFER                            GATE 2–4 below
- Do NOT update detector         (proceed with confident scores)
  state (EMA, CUSUM, jars)
- Do NOT fire alerts
- Coach asks clarifying
  question if high uncertainty persists
      │
      ├─── Core values (w_j ≥ w_min): watch for DECLINE
      │           │
      │           ▼
      │      GATE 2: Is the change big enough to notice?
      │      (size: magnitude × profile weight)
      │           │
      │       no ──┘              yes
      │       (ignore)             │
      │                            ▼
      │                      GATE 3: Is it sustained?
      │                      (duration: one step or a pattern?)
      │                            │
      │                    no ──┘              yes
      │                    (spike → noise)      │
      │                                         ▼
      │                                   GATE 4: Does the user know?
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
            (+1 for ≥ C_min steps, only counting confident scores)
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

### Sequential ordering: where does evolution check go?

After Gate 1 (confidence), four checks remain: signal type, evolution, size, and duration. The recommended order is **Order B: Signal-type first**.

| Order | Early-week behavior | Crash handling | Fade handling | Complexity | False alert risk |
|---|---|---|---|---|---|
| **A: Evolution-first** | Blind (can't assess volatility) | Unnecessary check | Good — excludes evolution early | Moderate | Low (after ramp-up) |
| **B: Signal-type-first** ✓ | Works immediately | Fast-pathed, clean | Good — evolution check on gradual only | Moderate | Low |
| **C: Size-then-duration** | Works immediately | Passes (crashes are big) | Late — detector state polluted during wait | Simple | Medium |
| **D: Duration-first** | Works immediately | Fires when threshold met | Evolution pollutes detectors | Simplest | Highest |

**Order B recommended:**

```
Confident score → Classify signal pattern (crash/fade/rise/spike) →
  If crash or spike: skip evolution check → Size → Duration → Awareness
  If fade or sustained shift: check evolution (volatility) → Size → Duration → Awareness
```

Crashes are fast-pathed — no unnecessary evolution check on single-step events. Evolution check only runs where it's meaningful (gradual patterns). Works from week 1 without a ramp-up period.

### Uncertainty gating

Uncertain entries are **invisible to drift detectors.** They do not update EMA accumulators, do not add marbles to CUSUM jars, do not count toward consecutive-step thresholds, and do not contribute to KL divergence windows. Only confident scores (`σ < ε_j`) modify detector state.

| Signal | Low uncertainty (σ < ε) | High uncertainty (σ ≥ ε) |
|---|---|---|
| **Crash** | Trust it. Real misalignment. Update detector state. | Suppress. Entry has conflicting signals. Coach asks a clarifying question. |
| **Rut** (sustained 0s or -1s) | Trust it. Value is genuinely dormant. Each step accumulates. | Do NOT accumulate. Scores may be measurement noise. Wait for confident data. |
| **Spike** (-1 then recovery) | Trust both dip and recovery. Confirmed noise. | The dip may not have been real. Ignore the entire episode. |
| **Fade** (+1 → 0 → 0) | Trust the trend. Real decline entering detectors. | The decline may be an artifact. Exclude uncertain entries. |
| **Rise** (non-core at +1) | Trust it. New behavioral pattern worth surfacing. | Do NOT flag as profile divergence yet. May be noise. |

A Critic that is frequently uncertain effectively slows down drift detection — which is the correct behavior.

---

## 4. Detection approaches

### Five categories

| Category | Online? | Data needed | Interpretability | Signal taxonomy | Profile weights `w_u` |
|---|---|---|---|---|---|
| **1. Rule-based** | Yes | Low (24 personas) | High | Hand-coded | Native (weighted formulas) |
| **2. BOCPD** | Yes | Moderate (~50 personas) | Medium | Emergent (changepoint type) | Requires wrapper |
| **3. GP regression** | Yes | Moderate per user (~8+ entries) | Medium | Not applicable (anomaly-based) | Requires wrapper |
| **4. Regime-switching / HMM** | Yes | High (~30+ per user, or pooled) | Medium | Emergent (learned regimes) | Can encode in emissions |
| **5. Autoencoders** | No (batch) | Very high (~500+ windows) | Low | Not applicable (anomaly-based) | Encoded in training data |

**For the POC: Category 1 (rule-based).** The dataset (24 annotated personas, ~380 annotations) is too small for learned models.

**First upgrade path: BOCPD.** When the dataset grows to ~50+ personas with confirmed changepoints, BOCPD replaces the hand-coded signal taxonomy with a unified probabilistic framework.

**Second upgrade path: Regime-switching / HMM.** When enough per-user data exists (~30+ entries per user), a hierarchical HMM can learn regime structure the hand-coded taxonomy might miss.

### Mathematical primitive: rolling standard deviation

Four of the five candidate rule-based approaches share a common foundation: they detect when the variability or level of recent scores deviates from a baseline.

$$
\bar{a}^{(j)}_W = \frac{1}{W} \sum_{k=0}^{W-1} \hat{a}^{(j)}_{t-k}, \qquad
\sigma^{(j)}_W = \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} \big(\hat{a}^{(j)}_{t-k} - \bar{a}^{(j)}_W\big)^2}
$$

| Approach | What it computes | Relationship to rolling σ |
|---|---|---|
| **EMA** | Exponentially weighted mean of drift signal | Tracks level shift. Equivalent to rolling mean with exponential decay. |
| **CUSUM** | Cumulative sum of deviations from allowance k | Detects sustained level shift. Sensitive to mean shifts that move rolling mean from baseline. |
| **Control Charts** | Mean ± n×σ from a baseline period | Directly uses σ. Breach means current scores are > n standard deviations from baseline. |
| **KL Divergence** | Distribution distance between baseline and recent window | Detects both level and shape changes. High KL ↔ different means or spreads. |
| **Cosine Similarity** | Directional agreement between `w_u` and `â_t` | Different primitive: angle, not variance. Complementary. |

### Sub-approach comparison and selection

**Rule-based sub-approaches being evaluated:**

| Sub-approach | Mechanism | What it detects well |
|---|---|---|
| **Baseline (dual-trigger)** | Crash: single-step drop > δ_j. No-recovery: score < τ_low for ≥ C_min steps. | Sharp crashes, sustained misalignment |
| **EMA** | Exponentially weighted mean of drift signal `d_j`. Alert when EMA > threshold. | Gradual trends (fade), with forgetting of old data |
| **CUSUM** | Cumulative sum of deviations from allowance k. Alert when sum > h. | Sustained small shifts that individually look harmless |
| **Cosine Similarity** | Angle between profile weights `w_u` and alignment vector `â_t`. Alert when cosine < threshold. | Holistic misalignment across multiple dimensions simultaneously |
| **Control Charts** | Mean ± nσ computed from a baseline period. Alert when score breaches lower control limit. | Deviations from the user's own established baseline |
| **KL Divergence** | Distribution distance between a baseline window and a recent window. | Both level shifts and shape changes |

**Selection logic:** Use Step 2-3 experiment results to select approaches per signal type:

- **Crash detection:** Approach with highest precision on consensus-crash points. Candidates: Baseline simple threshold, Control Charts (LCL breach), Cosine Similarity.
- **Fade / no-recovery detection:** Approach with highest hit rate on sustained patterns. Candidates: Baseline consecutive-low counter, EMA, CUSUM.

If the baseline meets targets, simpler thresholds win — no EMA/CUSUM needed.

---

## 5. System integration

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

### Onboarding → Critic → detectors → Coach

During onboarding, the user completes a 6-set **Best-Worst Scaling (BWS)** assessment. Each set presents 4 cards; the user selects "Most like me" and "Least like me." Raw scores are computed as `raw_score(v) = best_count(v) - worst_count(v)` per value, then normalized to produce a **value profile** `w_u` — a 10-dimensional weight vector summing to 1.0.

After onboarding, `w_u` is treated as fixed until a user-endorsed update.

Once journaling begins:

1. Each journal entry is embedded via a frozen sentence encoder (SBERT).
2. The **state encoder** constructs a state vector from recent text embeddings, time gaps between entries, and `w_u`.
3. The **Critic (VIF)** — an MLP with 2-3 ReLU layers — processes this state and outputs per-dimension alignment estimates: `â_{u,t} ∈ {-1, 0, +1}^10`.
4. The **evolution detection** layer classifies each dimension's divergence pattern as STABLE, EVOLUTION, or DRIFT before drift triggers evaluate it.
5. Dimensions classified as EVOLUTION are excluded from crash/rut evaluation and routed to Coach profile-update messaging.
6. When drift triggers fire, the **Coach** responds: reads the user's journal history to surface thematic evidence, explains *why* misalignment occurred with specific entry snippets, and offers reflective prompts — never prescriptive advice.

**When each component writes to user state:**

| Component | Writes | When |
|---|---|---|
| **Onboarding BWS** | `w_u` (initial weight vector) | Once, at registration |
| **Critic** | `â_t` (alignment vector), `σ_t` (uncertainty vector) | Every journal entry |
| **Drift detectors** | Internal state (EMA accumulators, CUSUM jars, baseline stats) | Every entry, updated incrementally |
| **Coach** | Profile update flag + optional `w_u` revision | Only when user endorses a value shift |

### Profile updates

The profile `w_u` is **not** automatically updated by the Critic or drift detectors. It changes only through user-endorsed actions:

1. **Explicit profile edit:** User directly adjusts value weights in settings.
2. **Coach-mediated evolution:** When the Coach surfaces a drift and the user responds "yes, my thinking has changed," the Coach proposes a specific weight adjustment. The user confirms before it takes effect.
3. **Periodic re-assessment:** Every N weeks (e.g., 12), offer an optional BWS re-assessment. Compare new weights to old; significant changes are flagged as potential evolution.

**What does NOT update the profile:**
- Critic scores alone (noisy, one-directional)
- Drift alerts alone (these are questions to the user, not answers)
- Sustained misalignment without user acknowledgment

**Evolution detection three-way classification:**

| Classification | Pattern | Volatility | What It Means |
|---|---|---|---|
| **Stable** | Behavior matches declared values | Any | No action needed |
| **Evolution** | Sustained, directional divergence | Low | User's values may have genuinely shifted |
| **Drift** | Volatile, inconsistent divergence | High | User is struggling to live their values |

For each dimension, the algorithm computes over a window of recent alignment scores:
- **Mean alignment** (`mu_j`): average over the window
- **Volatility** (`sigma_j`): standard deviation
- **Residual** (`residual_j = mu_j - expected_j`): deviation from what the declared profile predicts

```
For each dimension j:
    if |residual_j| < residual_threshold:
        → STABLE
    elif volatility_j < volatility_threshold:
        → EVOLUTION (sustained, directional — low noise)
    else:
        → DRIFT (volatile, inconsistent — high noise)
```

Evolution and drift share the same residual magnitude but differ in volatility. A user who consistently scores -1 on Achievement (low volatility) is evolving. A user who oscillates between +1 and -1 (high volatility) is drifting.

---

## 6. Experiment plan

### Data

- **204 personas** (1,651 entries) with declared core values and judge-labeled alignment scores in `logs/judge_labels/judge_labels.parquet`
- **24 personas with human annotations** (380 annotations from 3 annotators) in `logs/annotations/`
- **Qualitative drift narratives** for each annotated persona in `notebooks/annotations/persona_drift.ipynb`
- **Trained Critic** (median QWK 0.362, recall_-1 0.313) producing noisy but signal-bearing predictions

### Consensus ground truth

Rather than manually labeling crisis points, use **cross-approach agreement** as a proxy for ground truth. If most approaches independently flag the same (persona, step, dimension), it's likely a real signal.

```
consensus_score = number of approaches that flag this (t, dim)  # 0–6

consensus_crisis:
  strong  = consensus_score ≥ 4   (majority agreement → high-confidence crisis)
  weak    = consensus_score ∈ {2, 3} (split opinion → ambiguous)
  none    = consensus_score ≤ 1   (at most one approach → not a crisis)
```

**Why consensus works:** The 6 approaches have genuinely different philosophies (memoryless vs. stateful, per-dimension vs. holistic, threshold vs. distributional). Agreement across philosophically different methods is stronger evidence than agreement across similar methods.

**Limitation:** If all approaches share a blind spot (e.g., none detect fades because all use τ_expect=0), consensus will miss it too. Phase 3 (absence-aware variant) addresses this.

### Phases

**Phase 1: Consensus ground truth + approach selection (1-2 days)**

1. Add baseline approach (simple crash/no-recovery thresholds) to the comparison notebook
2. Run all 6 approaches × parameter grid on human annotation means for personas with ≥5 steps
3. Compute consensus labels: strong (≥4 agree), weak (2-3 agree), none (≤1 agrees)
4. Score each approach against consensus (hit rate, precision, F1, FPR, latency)
5. If baseline meets targets: stop — simple thresholds win
6. If not: select best crash approach + best fade/no-recovery approach from candidates

Output: Selected approach pair with tuned parameters. Consensus labels stored as `logs/annotations/consensus_crisis_labels.parquet`.

**Phase 2: Critic-in-the-loop evaluation (1 day)**

1. Run selected approach pair on Critic predictions (with MC Dropout uncertainty) for the same personas
2. Apply uncertainty gating: exclude entries where σ ≥ ε_j from detector state
3. Compare hit rate / precision / FPR against Phase 1 results
4. Quantify the "Critic noise penalty"

Output: Gap analysis showing which Critic improvements would most benefit drift detection.

**Phase 3: Absence-aware variant (0.5 day)**

1. Re-run Phase 1 with `τ_expect ∈ {0.0, 0.2, 0.3}` as an additional grid parameter
2. Compare specifically on consensus points involving fade patterns
3. If it improves fade detection without increasing FPR, adopt it

Output: Decision on whether to include the absence-aware formula.

### Metrics

| Metric | Target | Measured on |
|---|---|---|
| **Hit Rate** (consensus crises correctly flagged) | ≥ 80% | Phase 1 (human annotations) |
| **Precision** (alerts that are consensus crises) | > 60% | Phase 1 |
| **F1 per value dimension** | > 0.5 | Phase 1 |
| **FPR** (false alarm rate) | < 20% | Phase 1 |
| **Critic noise penalty** (hit rate drop Phase 1 → Phase 2) | < 15pp | Phase 2 |
| **First-alert latency** | ≤ 2 steps after crisis onset | Phase 1 |

**Threshold ranges for grid search:**

| Parameter | Symbol | Range | Rationale |
|---|---|---|---|
| Uncertainty ceiling | ε_j | {0.2, 0.3, 0.4} | Gate 1: scores with σ ≥ ε excluded from detectors. 0.3 is the starting value. |
| Profile weight floor | w_min | {0.10, 0.15, 0.20} | Below this, the dimension is not monitored. |
| Expected alignment floor | τ_expect | {0.0, 0.2, 0.3} | 0.0 = standard (negatives only); 0.2-0.3 = absence-aware. |
| EMA blending factor | α | {0.2, 0.3, 0.4} | Higher = more reactive, lower = more forgiving. |
| EMA alert threshold | ema_thresh | {0.08, 0.10, 0.15} | The worry level that fires an alert. |
| CUSUM allowance | k | {0.2, 0.3, 0.4} | How much a neutral score drains the jar. |
| CUSUM alarm level | h | {1.0, 1.5, 2.0} | Total evidence needed to fire. |
| Crash threshold | δ | {0.5, 1.0, 1.5} | Minimum single-step drop. +1 → 0 = 1.0, +1 → -1 = 2.0. |
| Rut duration | C_min | {2, 3, 4} | Consecutive steps below τ_low needed for rut. |

**Blocking dependency:** The Critic frontier (median QWK 0.362) may not be strong enough for reliable automated drift triggers. Phase 2 will quantify exactly how much this matters. If the Critic noise penalty exceeds 15 percentage points on hit rate, further Critic improvement should be prioritized before productionizing drift detection.

---

## References

- Schwartz, S. H. (2012). An Overview of the Schwartz Theory of Basic Values. *Online Readings in Psychology and Culture*, 2(1).
- Schwartz, S. H., & Bilsky, W. (1987). Toward a universal psychological structure of human values. *Journal of Personality and Social Psychology*, 53(3), 550–562.
- Hitlin, S., & Piliavin, J. A. (2004). Values: Reviving a dormant concept. *Annual Review of Sociology*, 30, 359–393.
- Rokeach, M. (1973). *The Nature of Human Values*. Free Press.
- Bardi, A., & Goodwin, R. (2011). The dual route to value change. *Journal of Cross-Cultural Psychology*, 42(2), 271–287.

---

## Related documents

- [`docs/vif/06_profile_conditioned_drift_and_encoder.md`](../vif/06_profile_conditioned_drift_and_encoder.md) — drift formulas and profile interaction
- [`docs/vif/04_uncertainty_logic.md`](../vif/04_uncertainty_logic.md) — dual-trigger rules and MC Dropout
- [`docs/evals/drift_detection_eval.md`](../evals/drift_detection_eval.md) — evaluation protocol and success criteria
- [`docs/evals/drift_approach_experiment.md`](../evals/drift_approach_experiment.md) — source material (original, unstructured)
- [`docs/evolution/01_value_evolution.md`](01_value_evolution.md) — value evolution detection design
- [`notebooks/annotations/drift_detection_comparison.ipynb`](../../notebooks/annotations/drift_detection_comparison.ipynb) — 5-approach comparison on annotation data
