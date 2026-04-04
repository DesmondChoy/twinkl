# Drift Detection

## 0. Critic performance and its implications for drift detection

The trained Critic (median QWK 0.362, recall_-1 = 0.313) produces ordinal alignment scores `â_t ∈ {-1, 0, +1}^10`. 62.1% of predictions are neutral (0), meaning genuine misalignment events are often predicted as 0 rather than -1. Because a 0 is indistinguishable from noise to any drift detector, crashes that the Critic hedges on become invisible to the detection layer.

**Example.** Maya's Benevolence scores (true vs. Critic predicted):

```
Week:   1    2    3    4    5    6    7    8
True:  +1   +1    0   -1   -1   -1    0   +1    ← crash weeks 4–6
Pred:   0    0    0    0   -1    0    0    0    ← only week 5 caught
```

A detector looking for two consecutive −1s (or a rolling mean below −0.5 for three weeks) fires on the true scores but stays silent on the Critic's output.

**Why the notebooks decouple the two.** The detection notebooks (Section 4) currently run on Judge labels — clean, directly scored ground truth — rather than Critic predictions. This deliberate separation means drift detector thresholds can be designed and tuned independently of Critic quality. Once thresholds are locked, Phase 2 re-runs detectors on Critic-predicted score sequences to quantify the gap. At the current Critic frontier, meaningful drift results will not be achievable for all ten value dimensions — dimensions with sparse −1 signal (e.g. Power, Security) are most at risk of being undetectable in practice.

---

## 1. Where drift detection sits in the pipeline

### The Critic and the drift detector are not the same thing

The **Critic** (also called the VIF — Value Identity Function) and the **drift detector** are two sequential stages. A common point of confusion is treating them as one component.

```
Journal Entry
      ↓
Configured frozen sentence encoder
      ↓
State Encoder (assembles text window + time gaps + w_u)
      ↓
Critic / VIF  ← produces â_t and σ_t per entry
      ↓
Drift Detector  ← reads the sequence of â_t scores over time
      ↓
Coach / Weekly Digest
```

| Stage | Question it answers | Temporal scope |
|---|---|---|
| **Critic / VIF** | "For this entry, is Maya aligned with her values?" | Single entry |
| **Drift detector** | "Across entries over time, is there a pattern worth surfacing?" | All entries so far |

The Critic produces scores. The drift detector reads the history of those scores.

### Why the Critic has a temporal window

The Critic predicts alignment **per entry** — one score vector `â_t` per journal entry. But the state it receives as input includes a window of recent entries (`N` embeddings + time gaps), not just the current one.

This window is the **input** to the Critic, not the output. It exists to help the Critic score the current entry more accurately — some entries only make sense in context:

> *"I finally called her back"* — aligned with Benevolence? Only interpretable if the Critic can see that two weeks ago Maya wrote about avoiding her sister.

The window goes *in*. One score vector comes *out*.

```
Entry 6  +  [Entry 5, Entry 4, Entry 3]  +  time gaps  +  w_u
                        ↓
                    Critic / VIF
                        ↓
              â_6 = [+1, 0, -1, ...]    ← single score for entry 6 only
```

The drift detector never sees the window. It only sees the sequence of output scores the Critic has produced over time: `â_1, â_2, â_3, â_4, â_5, â_6, ...`

### This design assumes Option A (immediate alignment)

The Critic has three possible target options (from `docs/vif/03_model_training.md`):

| Option | What the Critic outputs | Temporal smoothing |
|---|---|---|
| **A — Immediate alignment** | Raw score for this entry: `â_t ∈ {-1, 0, +1}` | None — done downstream by drift detector |
| **B — Short-horizon forecast** | Average alignment over next H days | Pre-smoothed by construction |
| **C — Discounted returns** | Long-horizon cumulative alignment with time-aware discounting | Heavily pre-smoothed |

**The entire drift detection spec in this document is designed for Option A outputs.** All temporal reasoning — EMA, CUSUM, weekly aggregation, the evolution classifier's volatility measure, the crash fast-path — lives in the drift layer because the Critic hands off raw, unsmoothed scores.

If the POC ever switched to Option B or C, the drift layer would need to be redesigned from scratch:

- **Crash fast-path breaks:** A sharp behavioral `+1 → -1` produces a gradual slope in a forecast output, not a step. The single-step crash detector never fires.
- **Spike/retract logic breaks:** There are no sharp transitions to retract — the forecast has already smoothed the dip and recovery into a shallow curve.
- **Evolution classifier breaks:** `sigma_j` (the volatility measure used to distinguish DRIFT from EVOLUTION) would reflect VIF smoothing artifacts, not behavioral volatility. A volatile Maya whose forecast hovers near 0 would be classified as STABLE — the wrong answer.

Options B and C move temporal reasoning *into* the Critic. The drift layer can no longer do it independently. The division of responsibility collapses.

---

## 2. What is drift?

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
| **Noise** | Signal blip, no real behavioral change | One rough week, scores recover |
| **Tradeoff** | Intentional short-term sacrifice, user is aware | "Gym can wait this month — deadline" |
| **Drift** | Unintended or unacknowledged departure from values | User didn't realize they stopped prioritizing family |
| **Evolution** | Intentional, endorsed change in what the user values | "I've decided career matters more than fitness now" |

These axes are **independent**. The same signal pattern could be noise, a tradeoff, or drift depending on user intent. There is no 1:1 mapping between signal patterns and interpretations.

### Psychological grounding

Values are central schemas resistant to change — people remember value-congruent events vividly and rationalize away incongruent ones (Schwartz, 2012; Rokeach, 1973). Users engage with values through three temporal orientations:

| Orientation | Question the user asks | What Twinkl provides |
|---|---|---|
| **Past** | "Did I really used to care about that?" | Evidence-based recall. Alignment scores over time are more reliable than user memory, which is biased toward value-congruent events. |
| **Present** | "How can I get what matters to me now?" | Tension surfacing. The Coach uses drift alerts to show the gap between declared values and current behavior. |
| **Future** | "Where do I see my life heading?" | The Coach asks: "Is this something you want to address, or has your thinking shifted?" — the user's answer reveals whether aspirational trajectory still matches the profile. |

### Absence as signal

Because users cognitively filter value-incongruent experiences from their writing, a core value going dormant (sustained 0 on a high-weight dimension) may indicate suppressed drift, not true neutrality. A user who values Benevolence but neglected a friend might write "I was so busy with the project" (reframing as Achievement) rather than "I ignored my friend."

If the Critic scores a -1 on a core value, that's a *strong* signal — the misalignment survived the user's own cognitive filtering. A sustained 0 on a core value that was previously active may be soft drift the user has rationalized away.

---

## 3. Applicable to all approaches

### Core vs peripheral values

All detection approaches must distinguish between dimensions the user declared as important and those they didn't:

- **Core values** (`w_j ≥ w_min`): monitored for **decline** — the system watches for the value going misaligned or dormant.
- **Peripheral values** (`w_j < w_min`): monitored only for **rise** — a previously unimportant dimension becoming consistently positive may signal emerging priorities.

If Power is `w=0.05` and scores -1, that's not a drift alert regardless of detection approach — the user didn't declare it as important.

### Four factors connecting signal to meaning

Regardless of detection approach, every signal passes through the same four conceptual filters:

| Factor | What it determines | Who measures it |
|---|---|---|
| **Confidence** | Whether the Critic's score is trustworthy at all | MC Dropout uncertainty (σ < ε_j) |
| **Size** | Whether the signal is worth noticing | Magnitude of change × profile weight |
| **Duration** | Whether it's a pattern or a blip | Detector state (varies by approach) |
| **Awareness** | Whether it's drift, tradeoff, or evolution | The user, via the Coach conversation |

Confidence comes first. An uncertain score must not enter the size or duration evaluation — it pollutes detector state with noise.

### Uncertainty gating (Gate 1 — applies to all approaches)

Uncertain entries are **invisible to drift detectors** regardless of which approach is used. Only confident scores (`σ < ε_j`) modify detector state.

A Critic that is frequently uncertain effectively slows down drift detection — which is the correct behavior. If the model can't confidently score an entry, the system should wait rather than act on a guess.

If a pattern of high uncertainty persists, the Coach asks a clarifying question rather than claiming misalignment.

### The awareness gate (applies to all approaches)

Once any approach fires an alert, the resolution pathway is the same. The Coach:

1. Reads the user's journal history to surface thematic evidence
2. Explains *why* misalignment occurred, citing specific entry snippets
3. Asks whether the gap is drift, a tradeoff, or evolution — never prescribes

The profile `w_u` is **not** automatically updated by any detection approach. It changes only when the user explicitly endorses a value shift in the Coach conversation, edits it directly in settings, or confirms a re-assessment result.

### Evolution gating (applies to all approaches)

Before drift triggers evaluate a dimension, evolution detection classifies its recent divergence pattern:

| Classification | Pattern | Volatility | Routing |
|---|---|---|---|
| **Stable** | Behavior matches declared values | Any | No action |
| **Evolution** | Sustained, directional divergence | Low | Coach: "priorities shifting?" → suggest profile update |
| **Drift** | Volatile, inconsistent divergence | High | Drift triggers evaluate normally |

Dimensions classified as EVOLUTION are excluded from drift trigger evaluation entirely. This prevents genuine value shifts from being misread as behavioral failure.

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
  (10-dim weights)   (10-dim alignment    (approach-specific)   drift / tradeoff /
                      + uncertainty)                            evolution
                                                                      │
                                                    ┌─────────────────┤
                                                    ▼                 ▼
                                              Keep monitoring    Update w_u
                                              (maybe schedule    (profile evolves)
                                               follow-up)
```

| Component | Writes | When |
|---|---|---|
| **Onboarding BWS** | `w_u` (initial weight vector) | Once, at registration |
| **Critic** | `â_t` (alignment vector), `σ_t` (uncertainty vector) | Every journal entry |
| **Drift detectors** | Internal state (approach-specific) | Every confident entry, updated incrementally |
| **Coach** | Profile update flag + optional `w_u` revision | Only when user endorses a value shift |

---

## 4. Approach-specific

### Comparison

| Category | Online? | Data needed | Interpretability | Signal taxonomy | Profile weights `w_u` |
|---|---|---|---|---|---|
| **1. Rule-based** | Yes | Low (24 personas) | High | Hand-coded | Native (weighted formulas) |
| **2. BOCPD** | Yes | Moderate (~50 personas) | Medium | Emergent from posterior | Requires wrapper |
| **3. GP regression** | Yes | Moderate (~8+ entries/user) | Medium | Not applicable (anomaly-based) | Requires wrapper |
| **4. HMM** | Yes | High (~30+ entries/user, or pooled) | Medium | Emergent (learned regimes) | Can encode in emissions |
| **5. Autoencoders** | No (batch) | Very high (~500+ windows) | Low | Not applicable (anomaly-based) | Encoded in training data |

**Selection rationale:**

- **POC: Rule-based.** Dataset (24 annotated personas, ~380 annotations) is too small for learned models. Every alert is explainable and thresholds are tractable to tune.
- **First upgrade: BOCPD.** At ~50+ personas with confirmed changepoints, BOCPD replaces the hand-coded taxonomy with a unified probabilistic framework.
- **Second upgrade: HMM.** At ~30+ entries/user, a hierarchical HMM can learn regime structure the hand-coded taxonomy might miss.
- **GP regression:** Lateral option — useful if irregular journaling (skipped weeks, bursts) proves problematic for weekly aggregation.
- **Autoencoders:** Long-term only. Viable at hundreds of users with months of history.

---

### 3.1 Rule-based (selected for POC)

Hand-coded thresholds define what counts as a crash, fade, or rise. No learning — the signal taxonomy is specified upfront and detectors check whether each pattern's conditions are met.

#### Signal taxonomy

The signal taxonomy is **specific to the rule-based approach.** BOCPD and other ML approaches subsume these patterns into a unified framework (changepoints, regime transitions, anomaly scores) without pre-specifying names.

**Core value patterns (`w_j ≥ w_min`):**

| Signal | What happened | Numbers | Coach framing |
|---|---|---|---|
| **Crash** | Sharp transition to misalignment in one step | `+1 → -1` or `0 → -1` | "Something shifted this week" |
| **Fade** | Gradual decline to dormancy over multiple steps | `+1 → +1 → 0 → 0 → 0` | "This value has been quietly fading" |
| **Spike** | Temporary dip that recovers within 1-2 steps | `+1 → -1 → +1` | No alert (or retract) |
| **No recovery** | Score stays negative/dormant for ≥ C_min steps after crash or fade | `+1 → -1 → -1 → -1 → -1` | "This has been going on for N weeks — it's not temporary" |
| **Onboarding gap** | Core value was never aligned from day one | `-1 → -1 → -1` from week 1 | "You ranked this as important, but it hasn't appeared in your journal yet" |

**Peripheral value patterns (`w_j < w_min`):**

| Signal | What happened | Numbers | Coach framing |
|---|---|---|---|
| **Rise** | Sustained positive alignment on a non-core dimension | `0 → 0 → +1 → +1 → +1 → +1` | "Achievement has been a consistent theme — has something changed?" |

#### Transition permutation table

Every pair of consecutive weekly scores produces one of 9 transitions.

**Core values:**

| From → To | Δ | Signal name | Action | Notes |
|---|---|---|---|---|
| **+1 → +1** | 0 | Consistency | None | Aligned and stable. Ideal state. |
| **+1 → 0** | -1 | Fade onset | Soft signal | One step is ambiguous. If sustained (≥ C_min 0s), becomes fade. With absence-aware formula (τ_expect > 0), produces `w_j × τ_expect` per step. |
| **+1 → -1** | -2 | Crash | Alert | Full reversal. Strongest single-step signal: `w_j × (1 + τ_expect)` or `w_j × 2`. |
| **0 → +1** | +1 | Recovery / activation | Drains detectors | If recovering from fade: EMA decays, CUSUM jar drains. If first-ever +1: late activation. |
| **0 → 0** | 0 | Dormancy | Soft signal (if sustained) | Single step: nothing. Sustained after prior +1: fade continues. Sustained from day one: onboarding gap. |
| **0 → -1** | -1 | Emerging crash | Alert (moderate) | Transition from dormancy to active misalignment. |
| **-1 → +1** | +2 | Sharp recovery | Drains detectors | Full reversal upward. Resets no-recovery counter. |
| **-1 → 0** | +1 | Partial recovery | Drains detectors (partial) | Moving toward neutral. Not yet aligned but improving. |
| **-1 → -1** | 0 | No recovery | Accumulates | Each step adds to no-recovery counter. After ≥ C_min steps: escalation alert. |

**Peripheral values:**

| From → To | Signal name | Action | Notes |
|---|---|---|---|
| **0 → +1** | Rise onset | Watch | Start counting. Not flagged until sustained ≥ C_min steps. |
| **+1 → +1** | Rise continues | Watch / flag | If count ≥ C_min: profile divergence alert. |
| **Any → 0 or -1** | — | None | Not monitored. |
| **-1 → -1** | — | None | Sustained negative on a peripheral value is not a concern. |

#### Multi-step signal rules

Single transitions are raw observations, not signals. No single transition triggers a final interpretation by itself.

**Core values:**

| Signal | Required sequence | Min steps | Trigger condition |
|---|---|---|---|
| **Crash** | Contains +1→-1 or 0→-1 | 2 | Single-step drop passes size gate: `w_j × |Δ| > δ_j`. Alert fires immediately but interpretation waits for next steps. |
| **Spike** | Crash → recovery within 1-2 steps | 3-4 | Score returns to +1 or 0 within 2 steps. Retracts crash alert → noise. |
| **Fade** | +1→0 → sustained 0s | ≥ 1 + C_min | C_min consecutive confident 0s on a previously active dimension. |
| **No recovery** | Crash → sustained -1s | ≥ 1 + C_min | After crash, score stays < τ_low for ≥ C_min consecutive confident steps. |
| **Onboarding gap** | -1 or 0 from day one, never +1 | ≥ 3 | Core value with no positive confident score in first N entries. |

**Peripheral values:**

| Signal | Required sequence | Min steps | Trigger condition |
|---|---|---|---|
| **Rise** | 0→+1 → sustained +1s | ≥ C_min | C_min consecutive confident +1 scores on a dimension with `w_j < w_min`. |

#### Detection pipeline

The rule-based pipeline implements all four universal factors as explicit gates:

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
- Do NOT update EMA/CUSUM/jars
- Do NOT fire alerts
- Coach asks clarifying question
  if high uncertainty persists
      │
      ├─── Core values (w_j ≥ w_min): watch for DECLINE
      │           │
      │           ▼
      │      GATE 2: Is the change big enough?
      │      Classify signal type (crash vs gradual)
      │      If gradual: run evolution check (volatility gate)
      │           │
      │       no / EVOLUTION ──┘      yes (DRIFT or crash)
      │       (ignore / route to          │
      │        Coach profile msg)         ▼
      │                            GATE 3: Is it sustained?
      │                            (EMA / CUSUM / consecutive counter)
      │                                   │
      │                           no ──┘              yes
      │                           (spike → noise)      │
      │                                                ▼
      │                                          GATE 4: Does the user know?
      │                                          (Coach asks)
      │                                                │
      │                                       ┌────────┴────────┐
      │                                       no                yes
      │                                       = DRIFT           │
      │                                                         ▼
      │                                                   Does the user endorse it?
      │                                                         │
      │                                                 ┌───────┴───────┐
      │                                                 no              yes
      │                                                 = DRIFT         │
      │                                                 (with denial)   ▼
      │                                                            Is it short-term?
      │                                                                  │
      │                                                          ┌───────┴───────┐
      │                                                          yes             no
      │                                                          = TRADEOFF      = EVOLUTION
      │                                                                           (update w_u)
      │
      └─── Peripheral values (w_j < w_min): watch for RISE
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
```

#### Sequential ordering of evolution check

After Gate 1 (confidence), four checks remain.

| Order | Early-week behavior | Crash handling | Fade handling | False alert risk |
|---|---|---|---|---|
| **A: Evolution-first** | Blind (can't assess volatility) | Unnecessary check | Good | Low (after ramp-up) |
| **B: Signal-type-first** | Works immediately | Fast-pathed, clean | Good | Low |
| **C: Size-then-duration** | Works immediately | Passes | Late — detector state polluted | Medium |
| **D: Duration-first** | Works immediately | Fires when threshold met | Evolution pollutes detectors | Highest |

#### Open design conflict: should crashes bypass the evolution check?

This is the key fork between two defensible positions. The choice affects whether a sudden value reversal can ever be routed as evolution rather than drift.

---

**Option 1 — Universal evolution pre-filter (from `01_value_evolution.md`)**

Evolution detection runs before all signal types, including crashes. Every confident score first passes through the three-way classifier (STABLE / EVOLUTION / DRIFT) before any drift trigger fires.

```
Confident score → Evolution detection (all signal types) →
  If EVOLUTION: route to Coach ("priorities shifting?") — skip drift triggers
  If STABLE or DRIFT: → Size → Duration → Awareness
```

*Consequence for crashes:* A `+1 → -1` crash on a dimension that has been trending negative for several weeks might already satisfy the evolution classifier (low volatility, high residual). Under this option, that crash is classified as EVOLUTION and never reaches the drift alert. The Coach asks whether priorities shifted, not whether the user is struggling.

*Consequence for early entries:* The evolution classifier requires `min_entries ≥ 6` to compute a stable residual and volatility estimate. Before that window is full, the classifier defaults to STABLE — meaning crashes in the first ~3 weeks are silently suppressed rather than alerting. This is the "blind early-week" problem in the table above.

*When to prefer this:* If the system should be conservative about firing drift alerts until it has enough history to distinguish a crash from a trend reversal. Prioritizes false-negative drift over false-positive drift.

---

**Option 2 — Signal-type-conditional evolution check (Order B)**

Crashes and spikes bypass the evolution check. Evolution check only runs on gradual patterns (fade, sustained shift). The intuition: a single-step reversal is too short to have stable volatility statistics — running the evolution classifier on it produces noise.

```
Confident score → Classify signal pattern (crash/fade/rise/spike) →
  If crash or spike: skip evolution check → Size → Duration → Awareness
  If fade or sustained shift: check evolution (volatility) → Size → Duration → Awareness
```

*Consequence for crashes:* Every crash reaches the size gate immediately, regardless of prior trend. A sudden `+1 → -1` always produces a potential alert — even if the preceding weeks suggested a directional shift was underway. The evolution classifier cannot intercept it.

*Consequence for fades:* Evolution check runs where it is most meaningful — on gradual, multi-step patterns where volatility statistics are stable and the classifier has signal to work with.

*When to prefer this:* If crashes should always be surfaced to the user, letting the Coach conversation (Gate 4) resolve whether it is drift, tradeoff, or evolution. Prioritizes false-positive drift over silent suppression of sudden events.

---

**The design question this forces:**

> *Can a crash ever be classified as evolution — and if so, what is lost by not alerting?*

If the answer is "yes, a crash can be evolution" → Option 1. The evolution pre-filter intercepts it before the drift trigger fires.

If the answer is "a crash should always alert, and the user resolves the meaning" → Option 2. The Coach conversation handles the ambiguity at Gate 4.

**Current recommendation: Option 2 (Order B).** Crashes are fast-pathed — no unnecessary evolution check on single-step events. Works from week 1 without a ramp-up period. The Coach's Gate 4 conversation is the correct place to distinguish a genuine crash from a trend reversal the user was already aware of.

#### Uncertainty gating per signal type

| Signal | Low uncertainty (σ < ε) | High uncertainty (σ ≥ ε) |
|---|---|---|
| **Crash** | Trust it. Update detector state. | Suppress. Coach asks a clarifying question. |
| **Rut** (sustained 0s or -1s) | Trust it. Each step accumulates. | Do NOT accumulate. Wait for confident data. |
| **Spike** (-1 then recovery) | Trust both dip and recovery. Confirmed noise. | Ignore the entire episode. |
| **Fade** (+1 → 0 → 0) | Trust the trend. Real decline entering detectors. | Exclude uncertain entries — only count confident 0s. |
| **Rise** (non-core at +1) | Trust it. New behavioral pattern worth surfacing. | Do NOT flag as profile divergence yet. |

#### Sub-approaches

| Sub-approach | Mechanism | What it detects well |
|---|---|---|
| **Baseline (dual-trigger)** | Crash: single-step drop > δ_j. No-recovery: score < τ_low for ≥ C_min steps. | Sharp crashes, sustained misalignment |
| **EMA** | Exponentially weighted mean of drift signal `d_j`. Alert when EMA > threshold. | Gradual trends (fade), with forgetting of old data |
| **CUSUM** | Cumulative sum of deviations from allowance k. Alert when sum > h. | Sustained small shifts that individually look harmless |
| **Cosine Similarity** | Angle between profile weights `w_u` and alignment vector `â_t`. Alert when cosine < threshold. | Holistic misalignment across multiple dimensions |
| **Control Charts** | Mean ± nσ from a baseline period. Alert when score breaches lower control limit. | Deviations from the user's own established baseline |
| **KL Divergence** | Distribution distance between baseline and recent window. | Both level shifts and shape changes |

Four of the six sub-approaches share a common mathematical foundation — they detect when the level or variability of recent scores deviates from a baseline:

$$
\bar{a}^{(j)}_W = \frac{1}{W} \sum_{k=0}^{W-1} \hat{a}^{(j)}_{t-k}, \qquad
\sigma^{(j)}_W = \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} \big(\hat{a}^{(j)}_{t-k} - \bar{a}^{(j)}_W\big)^2}
$$

Cosine Similarity uses a different primitive (angle, not variance) and is complementary.

#### Maya running example (rule-based)

Maya completes onboarding. BWS results produce:
- **Core values:** Benevolence (w=0.45), Self-Direction (w=0.40)
- **Peripheral values:** Achievement (w=0.05) and 7 others

**Crash — Benevolence drops (week 7):**

```
Week:        1    2    3    4    5    6    7
Score:      +1   +1   +1   +1   +1   +1   -1
                                          ^^^
                                        crash: drop of 2.0, confident (σ=0.08)
```

Baseline fires immediately (drop of 2.0 > δ=0.5). EMA jumps to 0.45×1.0=0.45. CUSUM jar fills by 0.45. All three agree: crash.

If σ=0.55 instead: Gate 1 blocks the score. Coach asks: "This week's entry sent mixed signals about how you're relating to people close to you. Could you tell me more?"

**Spike — Benevolence recovers (weeks 8-9):**

```
Week:        1    2    3    4    5    6    7    8    9
Score:      +1   +1   +1   +1   +1   +1   -1   +1   +1
                                               ^^^^^^^
                                             recovery → spike (noise)
```

EMA decays. CUSUM jar drains. No alert fires.

**Fade — Self-Direction goes dormant (weeks 5-12):**

```
Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:      +1   +1   +1   +1    0    0    0    0    0    0    0    0
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 confident 0s → genuine dormancy
```

With absence-aware formula (τ_expect=0.3): each confident 0 produces `0.40 × 0.3 = 0.12` drift signal, accumulating over 8 weeks.

**No recovery — Benevolence never comes back:**

```
Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:      +1   +1   +1   +1   +1   +1   -1   -1   -1   -1   -1   -1
                                          ^^^  ─────────────────────────
                                        crash  no recovery → escalate at week 10 (C_min=3)
```

**Onboarding gap — Benevolence was never aligned:**

```
Week:        1    2    3    4    5    6    7    8    9   10
Score:      -1   -1   -1   -1   -1   -1   -1   -1   -1   -1
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            no transition ever happened. not drift.
```

After ~3 entries: "You ranked Benevolence as your top value, but it hasn't appeared positively in your journal entries yet."

**Rise — Achievement emerges (weeks 4-12):**

```
Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:       0    0    0   +1   +1   +1   +1   +1   +1   +1   +1   +1
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              sustained +1 on peripheral value → rise
```

Crash/fade detectors ignore this (w=0.05 < w_min). Rise detector catches it at C_min steps. Coach surfaces it alongside the Self-Direction fade: "Self-Direction has been absent from your entries for 8 weeks, while Achievement has been a consistent theme. Has something changed?"

#### Threshold grid for experiment

| Parameter | Symbol | Range | Rationale |
|---|---|---|---|
| Uncertainty ceiling | ε_j | {0.2, 0.3, 0.4} | Gate 1: scores with σ ≥ ε excluded. 0.3 is the starting value. |
| Profile weight floor | w_min | {0.10, 0.15, 0.20} | Below this, the dimension is not monitored. |
| Expected alignment floor | τ_expect | {0.0, 0.2, 0.3} | 0.0 = standard (negatives only); 0.2-0.3 = absence-aware. |
| EMA blending factor | α | {0.2, 0.3, 0.4} | Higher = more reactive, lower = more forgiving. |
| EMA alert threshold | ema_thresh | {0.08, 0.10, 0.15} | The worry level that fires an alert. |
| CUSUM allowance | k | {0.2, 0.3, 0.4} | How much a neutral score drains the jar. |
| CUSUM alarm level | h | {1.0, 1.5, 2.0} | Total evidence needed to fire. |
| Crash threshold | δ | {0.5, 1.0, 1.5} | Minimum single-step drop. +1→0 = 1.0, +1→-1 = 2.0. |
| Rut duration | C_min | {2, 3, 4} | Consecutive steps below τ_low needed for rut. |

---

### 3.2 Bayesian Online Changepoint Detection (BOCPD)

Instead of hand-coded signal patterns, BOCPD models the **posterior probability that a changepoint occurred at each time step**. It maintains a "run length" distribution — how long since the last change in the generative process.

The signal taxonomy (crash, fade, no-recovery) is **not an input**. Crash, fade, and no-recovery all emerge from the posterior shape — a crash produces a sudden spike in P(changepoint), a fade produces a gradual rise over multiple steps.

**How it works:**
- Assume scores within a regime are drawn from a distribution (e.g., Dirichlet-Categorical for {-1, 0, +1} scores).
- At each step t, compute P(changepoint at t | data so far) using Bayesian updating.
- When P(changepoint) exceeds a threshold, flag it — no need to specify whether it's a crash or fade.

**Maya example:** For weeks 1-6 on Benevolence (all +1), BOCPD learns a regime with mean ≈ +1. At week 7 (score = -1), P(changepoint at week 7) ≈ 0.95 — the observation is extremely unlikely under the current regime. For the Self-Direction fade (+1 → 0 → 0 → 0), the changepoint probability rises gradually as 0s accumulate.

**Pro:**
- Unified framework — crash, fade, and no-recovery are all changepoints with different posterior shapes.
- Naturally handles "how many steps before it's real?" — the posterior probability *is* the confidence.
- Adapts to each user's baseline (regime parameters are learned, not hand-coded).

**Con:**
- With 10-15 data points per user per dimension, the posterior will be wide.
- Less interpretable for the Coach: "P(changepoint) = 0.87" needs translation to user-facing language.
- Incorporating profile weights `w_u` requires a wrapper layer (BOCPD operates per-dimension independently).

**Data requirement:** Moderate (~50 personas). Works with conjugate priors so can start with informative priors and update per-user.

---

### 3.3 Gaussian Process (GP) regression

Models each dimension's alignment trajectory as a Gaussian Process — a distribution over smooth functions. Drift = observations falling outside the GP's predictive interval.

The signal taxonomy is **not an input** — anomalies are flagged purely by deviation from the learned trajectory, without naming the pattern.

**How it works:**
- Fit a GP to (week, score) pairs for each dimension per user.
- GP predicts expected score at week t+1 with a confidence interval.
- If the observed score falls outside the interval, flag it as anomalous.
- Uncertainty naturally widens during gaps (skipped weeks) and narrows where data is dense.

**Maya example:** For weeks 1-6 on Benevolence (all +1), the GP fits a flat function at mean ≈ +1 with tight bands (±0.2). At week 7 (score = -1), the observation is ~10σ below the GP's prediction → strong anomaly. The fade on Self-Direction shows the GP's predicted mean gradually shifting downward as 0s accumulate.

**Pro:**
- Handles irregular time gaps naturally (works with arbitrary input spacing).
- Uncertainty-aware natively at the detector level.
- Predictive interval adapts to each user's trajectory shape, not a fixed baseline.

**Con:**
- With only 10-15 data points per dimension, uncertainty bands may be too generous — may rarely flag anything.
- Requires choosing a kernel (RBF? Matérn?). Wrong kernel imposes smoothness assumptions that may not hold for discrete {-1, 0, +1} scores.
- Doesn't naturally incorporate profile weights `w_u`.

**Data requirement:** Moderate per user (~8+ entries). Best option if irregular journaling (skipped weeks, bursts) proves problematic for weekly aggregation.

---

### 3.4 Regime-switching / Hidden Markov Models (HMM)

Explicitly models latent regimes the user transitions between. The model learns: (a) what each regime looks like (emission distribution), and (b) the probability of switching between regimes (transition matrix).

The signal taxonomy **emerges from the learned transition patterns** rather than being hand-coded.

**How it works:**
- Define K latent states, e.g., K=3: "aligned" (emissions centered on +1), "struggling" (centered on 0, high variance), "drifting" (centered on -1).
- Learn emission parameters and transition probabilities using EM (Baum-Welch) or MCMC.
- At each time step, infer the most likely regime via Viterbi decoding or forward-backward.
- A crash = transition from aligned → drifting. A fade = aligned → struggling → drifting. A spike = aligned → drifting → aligned.

**Maya example:** The model infers Maya is in the "aligned" regime for weeks 1-6 (emissions: +1, +1, +1, +1, +1, +1). At week 7 (-1), the Viterbi path switches to "drifting." If weeks 8-9 return to +1, the path switches back to "aligned" — a spike. If weeks 8-10 stay at -1, weeks 7+ are confidently classified as the "drifting" regime.

**Pro:**
- The signal taxonomy emerges from data — if real users show patterns the hand-coded taxonomy doesn't cover, the HMM can discover them.
- Transition probabilities encode base rates: "aligned → drifting transitions are rare (P=0.05), so evidence must be strong."
- Can be extended to a hierarchical HMM, sharing regime parameters while allowing per-user transition rates.

**Con:**
- With 10 entries per user, per-user EM will overfit. Requires a hierarchical Bayesian HMM across all users.
- K must be specified. K=3 mirrors the rule-based taxonomy but the right K is not obvious.
- "The Viterbi path switched to state 2" needs translation for the Coach.

**Data requirement:** High (~30+ entries/user for per-user fitting, or pooled across ~100+ users for a hierarchical model).

---

### 3.5 Autoencoders (anomaly detection)

Train a neural network to reconstruct "normal" alignment trajectories. High reconstruction error = anomaly = potential drift.

The signal taxonomy is **not an input** — any deviation from learned normal behavior is flagged without naming the pattern.

**How it works:**
- Encode a window of W recent scores (across all 10 dimensions) into a latent vector, then decode back.
- Train on windows from the "aligned" portion of trajectories.
- At inference, if reconstruction error exceeds a threshold, the current window is anomalous.

**Maya example:** Train on windows where scores are [+1, +1, +1, ...] across core dimensions. When the window includes the week-7 crash (Benevolence = -1 while others remain +1), reconstruction error spikes — this pattern can't be reconstructed from the latent space learned on aligned windows.

**Pro:**
- Captures cross-dimension patterns — e.g., Self-Direction fading while Achievement rises may be a recognizable "career-shift" pattern in the latent space.
- No signal taxonomy needed.

**Con:**
- **Data is the fundamental bottleneck.** With 24 personas × ~10 entries = ~240 windows, an autoencoder will memorize rather than generalize.
- The {-1, 0, +1} discrete score space has only 3^10 = 59,049 possible alignment vectors — simpler distance metrics (like cosine similarity) capture the same information.
- Least interpretable of all options: "reconstruction error = 0.73" tells the Coach nothing about *what* changed.

**Data requirement:** Very high (~500+ diverse trajectory windows). Viable only when Twinkl scales to hundreds of real users with months of journaling history.

---

## 5. Experiment plan

### Data

- **204 personas** (1,651 entries) with declared core values and judge-labeled alignment scores in `logs/judge_labels/judge_labels.parquet`
- **24 personas with human annotations** (380 annotations from 3 annotators) in `logs/annotations/`
- **Qualitative drift narratives** for each annotated persona in `notebooks/annotations/persona_drift.ipynb`
- **Trained Critic** (median QWK 0.362, recall_-1 0.313)

### Consensus ground truth

Rather than manually labeling crisis points, use **cross-approach agreement** as a proxy for ground truth:

```
consensus_score = number of approaches that flag this (t, dim)  # 0–6

strong  = consensus_score ≥ 4   (majority agreement → high-confidence crisis)
weak    = consensus_score ∈ {2, 3} (split opinion → ambiguous)
none    = consensus_score ≤ 1   (at most one approach → not a crisis)
```

**Why it works:** The 6 rule-based sub-approaches have genuinely different philosophies (memoryless vs. stateful, per-dimension vs. holistic, threshold vs. distributional). Agreement across philosophically different methods is stronger evidence than agreement across similar methods.

**Limitation:** If all approaches share a blind spot (e.g., none detect fades because all use τ_expect=0), consensus will miss it too. Phase 3 addresses this.

### Phases

**Phase 1: Consensus ground truth + approach selection (1-2 days)**

1. Add baseline approach (simple crash/no-recovery thresholds) to the comparison notebook
2. Run all 6 sub-approaches × parameter grid on human annotation means for personas with ≥5 steps
3. Compute consensus labels
4. Score each sub-approach against consensus (hit rate, precision, F1, FPR, latency)
5. If baseline meets targets: stop — simple thresholds win
6. If not: select best crash sub-approach + best fade/no-recovery sub-approach

Output: Selected sub-approach pair with tuned parameters. Labels stored as `logs/annotations/consensus_crisis_labels.parquet`.

**Phase 2: Critic-in-the-loop evaluation (1 day)**

1. Run selected sub-approach pair on Critic predictions (with MC Dropout uncertainty)
2. Apply uncertainty gating: exclude entries where σ ≥ ε_j from detector state
3. Compare hit rate / precision / FPR against Phase 1 results
4. Quantify the "Critic noise penalty"

Output: Gap analysis showing which Critic improvements would most benefit drift detection.

**Phase 3: Absence-aware variant (0.5 day)**

1. Re-run Phase 1 with `τ_expect ∈ {0.0, 0.2, 0.3}` as an additional grid parameter
2. Compare on consensus points involving fade patterns
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

**Blocking dependency:** The Critic frontier (median QWK 0.362) may not be strong enough for reliable automated drift triggers. Phase 2 will quantify exactly how much this matters. If the Critic noise penalty exceeds 15pp on hit rate, further Critic improvement should be prioritized before productionizing drift detection.

---

## References

- Schwartz, S. H. (2012). An Overview of the Schwartz Theory of Basic Values. *Online Readings in Psychology and Culture*, 2(1).
- Schwartz, S. H., & Bilsky, W. (1987). Toward a universal psychological structure of human values. *Journal of Personality and Social Psychology*, 53(3), 550–562.
- Hitlin, S., & Piliavin, J. A. (2004). Values: Reviving a dormant concept. *Annual Review of Sociology*, 30, 359–393.
- Rokeach, M. (1973). *The Nature of Human Values*. Free Press.
- Bardi, A., & Goodwin, R. (2011). The dual route to value change. *Journal of Cross-Cultural Psychology*, 42(2), 271–287.

---

## Related documents

- [`docs/vif/04_uncertainty_logic.md`](../vif/04_uncertainty_logic.md) — uncertainty, drift formulas, and trigger logic
- [`docs/vif/02_system_architecture.md`](../vif/02_system_architecture.md) — state and runtime artifact flow
- [`docs/evals/drift_detection_eval.md`](../evals/drift_detection_eval.md) — evaluation protocol and success criteria
- [`docs/evolution/01_value_evolution.md`](01_value_evolution.md) — value evolution detection design
- [`notebooks/annotations/drift_detection_comparison.ipynb`](../../notebooks/annotations/drift_detection_comparison.ipynb) — 5-approach comparison on annotation data
