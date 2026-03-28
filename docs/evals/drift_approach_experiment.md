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

### Time unit: weekly aggregates

The Critic scores each journal entry, but drift detection operates on **weekly aggregates**. If a user writes 5 entries in one week with Benevolence scores [+1, +1, -1, +1, +1], the weekly average is +0.6 — no crash. This smooths within-week noise. If the user writes 1 entry per week, entry = week.

The annotation data uses `t_index` (entry order) without timestamps. For the experiment, each `t_index` is treated as one time step. In production, entries would be aggregated to weekly averages before running detectors.

Skipped weeks (no journal entry) produce no data point. The system does not impute or interpolate — it simply waits for the next entry.

### Two independent axes

Drift detection involves two questions that must not be conflated:

**Question A — What happened in the signal?** This is observable. The Critic measures it. Drift detection cares about **transitions**, not states. Sustained +1 is consistency (no issue). Sustained -1 after a crash is a failure to recover (escalation). Sustained -1 from day one is an onboarding gap (not drift).

Signal patterns on **core values** (w_j ≥ w_min — dimensions the user declared as important):

| Signal pattern | What happened | Numbers | Coach framing |
|---|---|---|---|
| **Crash** | Sharp transition to misalignment in one step | `+1 → -1` or `0 → -1` | "Something shifted this week" |
| **Fade** | Gradual decline to dormancy over multiple steps | `+1 → +1 → 0 → 0 → 0` | "This value has been quietly fading" |
| **Spike** | Temporary dip that recovers within 1-2 steps | `+1 → -1 → +1` | No alert (or retract) |
| **No recovery** | Score stays negative/dormant after crash or fade for ≥ C_min steps | `+1 → -1 → -1 → -1 → -1` | "This has been going on for N weeks — it's not temporary" |
| **Onboarding gap** | Core value was never aligned from day one | `-1 → -1 → -1` from week 1 | "You ranked this as important, but it hasn't appeared in your journal yet" |

Signal patterns on **peripheral values** (w_j < w_min — dimensions the user ranked low):

| Signal pattern | What happened | Numbers | Coach framing |
|---|---|---|---|
| **Rise** | Sustained positive alignment on a non-core dimension | `0 → 0 → +1 → +1 → +1 → +1` | "Achievement has been a consistent theme — has something changed?" |

Peripheral dimensions are NOT monitored for crash/fade/no-recovery. If Power is w=0.05 and scores -1, that's not a drift alert — the user didn't declare it as important. The system only watches peripheral values for the rise pattern (a new value emerging).

**Question B — What does it mean?** This requires user input. The Coach resolves it.

| Interpretation | Description | Example |
|---|---|---|
| **Noise** | Signal blip, no real behavioral change | One rough week, scores recover (spike) |
| **Tradeoff** | Intentional short-term sacrifice, user is aware | "Gym can wait this month — deadline" |
| **Drift** | Unintended or unacknowledged departure from values | User didn't realize they stopped prioritizing family |
| **Evolution** | Intentional, endorsed change in what the user values | "I've decided career matters more than fitness now" |

These axes are **independent**. A crash (signal) could be noise, a tradeoff, or drift. A no-recovery (signal) could be drift or evolution. There is no 1:1 mapping between signal patterns and interpretations.

**NOTES: Get an LLM to link signal pattern + interpretation to a question. Link to Weekly Digest & Coach 

### Running example: Maya

Maya completes onboarding. Her BWS results produce profile weights:

- **Core values:** Benevolence (w=0.45), Self-Direction (w=0.40) — both above w_min=0.15, monitored for decline
- **Peripheral values:** Achievement (w=0.05), and 7 others — below w_min, only monitored for rise

Over 12 weeks of journaling, five different things happen on her core and peripheral values:

**Crash — Benevolence drops in one step (week 7) [core value]**

Maya writes about blowing up at her sister who asked for help during a stressful move. The Critic runs 50 forward passes:

- **Scenario A (confident):** mean = -1, σ = 0.08. All 50 sub-networks agree: this entry is misaligned on Benevolence. The score passes Gate 1 (σ < ε=0.3). It enters the drift detectors. Signal is strong (w=0.45 × 2.0 magnitude = 0.90).
- **Scenario B (uncertain):** mean = -0.3, σ = 0.55. The sub-networks disagreed — some saw -1 (the blowup), others saw +1 (she mentions feeling guilty and wanting to help). Gate 1 blocks this score. It does NOT enter the EMA or CUSUM. The Coach doesn't claim misalignment — instead it asks: "This week's entry sent mixed signals about how you're relating to people close to you. Could you tell me more?"

Assuming Scenario A (confident crash):

```
Benevolence (core, w=0.45):

Week:        1    2    3    4    5    6    7
Score:      +1   +1   +1   +1   +1   +1   -1
Uncertain?:  n    n    n    n    n    n    n
                                          ^^^
                                        crash: drop of 2.0, confident
```

But is it drift? Depends on what happens next.

**Spike — Benevolence recovers (weeks 8-9) [core value]**

Maya apologizes to her sister and helps with the move. Benevolence bounces back.

```
Benevolence (core, w=0.45):

Week:        1    2    3    4    5    6    7    8    9
Score:      +1   +1   +1   +1   +1   +1   -1   +1   +1
                                               ^^^^^^^
                                             recovery within 2 weeks
```

The crash was real and confident, but the pattern wasn't sustained → **spike (noise)**. EMA worry rises briefly then decays. CUSUM jar fills then drains. No alert fires.

**Fade — Self-Direction goes dormant (weeks 5-12) [core value]**

Maya's entries slowly stop mentioning independent choices. She's not acting *against* Self-Direction — it just stops appearing.

```
Self-Direction (core, w=0.40):

Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:      +1   +1   +1   +1    0    0    0    0    0    0    0    0
Uncertain?:  n    n    n    n    n    n    n    n    n    n    n    n
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 all 0s are confident → genuine dormancy
```

Uncertainty matters here:

- **If the 0s are confident (σ < ε):** The Critic is sure there's no Self-Direction signal. Each 0 passes Gate 1 and enters the detectors. This is genuine dormancy — with the absence-aware formula (τ_expect=0.3), each confident 0 produces `0.40 × 0.3 = 0.12` drift signal, accumulating over 8 weeks.
- **If the 0s are uncertain (σ ≥ ε):** The entries might contain Self-Direction content that the sub-networks interpreted differently. These scores are excluded from detector state. The apparent dormancy may be a measurement problem. The system waits for confident data.

This is where **absence is the signal**. Maya's cognitive filtering may be at work — she doesn't write about Self-Direction because she's avoiding the tension of not exercising it.

**No recovery — what if Benevolence never came back? [core value]**

Alternative to the spike scenario. Maya crashes at week 7 and stays misaligned:

```
Benevolence (core, w=0.45):

Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:      +1   +1   +1   +1   +1   +1   -1   -1   -1   -1   -1   -1
                                          ^^^  ─────────────────────────
                                        crash  no recovery for 5 weeks
                                               → escalate at week 7 + C_min
```

The crash alert fires at week 7. The no-recovery escalation fires at week 7 + C_min (e.g., week 10 if C_min=3). The Coach reframes: "Your Benevolence scores have been negative for 4 consecutive weeks now. This isn't a bad week — it's a pattern."

**Onboarding gap — what if Benevolence was never aligned? [core value]**

Different scenario entirely. Maya declared Benevolence as core (w=0.45), but her behavior never matched from day one:

```
Benevolence (core, w=0.45):

Week:        1    2    3    4    5    6    7    8    9   10
Score:      -1   -1   -1   -1   -1   -1   -1   -1   -1   -1
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            no transition ever happened. this is not drift.
```

No crash detector fires (no transition). No fade detector fires (no decline). This is a gap between declared values and observed behavior from the start. After ~3 entries, the system flags it: "You ranked Benevolence as your top value, but it hasn't appeared positively in your journal entries yet. Would you like to explore what's getting in the way?"

This could mean: the user over-claimed during onboarding (social desirability bias), the Critic is miscalibrating, or the user's life circumstances prevent acting on this value right now.

**Rise — Achievement emerges (weeks 4-12) [peripheral value]**

Maya starts writing about career goals, promotions, and outperforming peers. Achievement was w=0.05 at onboarding — below w_min, so the system doesn't monitor it for crash/fade.

```
Achievement (peripheral, w=0.05):

Week:        1    2    3    4    5    6    7    8    9   10   11   12
Score:       0    0    0   +1   +1   +1   +1   +1   +1   +1   +1   +1
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              sustained +1 on a non-core value
```

The crash/fade detectors ignore this because w=0.05 < w_min. But the rise detector catches it: a peripheral dimension showing sustained +1 for ≥ C_min steps.

This matters because:
- The Self-Direction fade and the Achievement rise may be connected — she's trading autonomy for career advancement
- The Coach surfaces both together: "Self-Direction has been absent from your entries for 8 weeks, while Achievement has been a consistent theme. Has something changed?"
- If Maya endorses it → evolution (update `w_u`: Achievement goes up, Self-Direction goes down)
- If Maya is surprised → drift she wasn't aware of

**How to detect rising values:** Periodically compare the behavioral profile (aggregated recent `â_t` scores) against `w_u`. If a peripheral dimension (w_j < w_min) shows sustained positive alignment (+1 for ≥ C_min confident steps), flag it as a **profile divergence** — not a crisis, but a prompt for the Coach to ask whether the user's priorities have shifted.

### Four factors connect the axes

| Factor | What it determines | Who measures it |
|---|---|---|
| **Confidence** | Whether the Critic's score is trustworthy at all | MC Dropout uncertainty (σ < ε_j) |
| **Size** | Whether the signal is worth noticing | Critic (magnitude of change × profile weight) |
| **Duration** | Whether it's a pattern or a blip | Drift detectors (EMA, CUSUM, thresholds) |
| **Awareness** | Whether it's drift, tradeoff, or evolution | The user, via the Coach conversation |

Confidence comes first. An uncertain score should not enter the size or duration gates at all — it pollutes detector state (EMA accumulators, CUSUM jars) with noise. Only confident predictions update drift detector state.

### The decision tree

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
    question if pattern of
    high uncertainty persists
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

### How uncertainty interacts with each signal pattern

| Signal | Low uncertainty (σ < ε) | High uncertainty (σ ≥ ε) |
|---|---|---|
| **Crash** (score = -1) | Trust it. Real misalignment. Update detector state. Fire alert if sustained. | Suppress. Entry has conflicting signals. Do NOT update EMA/CUSUM. Coach asks a clarifying question instead of claiming misalignment. |
| **Rut** (sustained 0s or -1s) | Trust it. Value is genuinely dormant or misaligned. Each step accumulates in detectors. | Do NOT accumulate. The scores may be measurement noise. Wait for confident data — a rut of uncertain scores is not a real rut. |
| **Spike** (-1 then recovery) | Trust both the dip and the recovery. Confirmed noise — detector state rises then falls. | The dip may not have been real. Ignore the entire episode. |
| **Fade** (+1 → 0 → 0) | Trust the trend. Real decline entering detectors. | The decline may be an artifact. Exclude uncertain entries — only count confident 0s toward dormancy. |
| **Rise** (non-core at +1) | Trust it. New behavioral pattern worth surfacing. | Do NOT flag as profile divergence yet. May be noise. |

The key design rule: **uncertain entries are invisible to drift detectors.** They do not update EMA accumulators, do not add marbles to CUSUM jars, do not count toward consecutive-step thresholds, and do not contribute to KL divergence windows. Only confident scores (σ < ε_j) modify detector state.

This means a Critic that is frequently uncertain effectively slows down drift detection — which is the correct behavior. If the model can't confidently score an entry, the system should wait rather than act on a guess.

Duration plays two distinct roles:

1. **Filtering noise from signal** (the confidence + size + duration gates). This is the Critic + detector's job.
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

### Signal strength depends on four things

Every alignment score transition has a signal strength determined by: whether the Critic is confident, what changed, how much the user cares, and how expected alignment is defined. An uncertain score has zero signal strength regardless of its magnitude — it doesn't enter the detectors.

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

### Complete transition permutation table

Every pair of consecutive weekly scores produces one of 9 transitions. The table below maps all of them for core and peripheral values, including transitions that are *not* actionable — so nothing is left implicit.

**Core values (w_j ≥ w_min):** monitored for decline and dormancy.

| From → To | Δ | Signal name | Action | Notes |
|---|---|---|---|---|
| **+1 → +1** | 0 | Consistency | None | Aligned and stable. Ideal state. |
| **+1 → 0** | -1 | Fade onset | Soft signal | One step is ambiguous. If sustained (≥ C_min 0s), becomes fade. With absence-aware formula (τ_expect > 0), produces `w_j × τ_expect` per step. |
| **+1 → -1** | -2 | Crash | Alert | Full reversal. Strongest single-step signal: `w_j × (1 + τ_expect)` or `w_j × 2` depending on formula. |
| **0 → +1** | +1 | Recovery / activation | Drains detectors | If recovering from fade: EMA decays, CUSUM jar drains. If first-ever +1: late activation (not drift-relevant). |
| **0 → 0** | 0 | Dormancy | Soft signal (if sustained) | Single step: nothing. Sustained after prior +1: fade continues. Sustained from day one: onboarding gap. With τ_expect > 0, each step adds `w_j × τ_expect`. |
| **0 → -1** | -1 | Emerging crash | Alert (moderate) | Transition from dormancy to active misalignment. Weaker than +1→-1 but still actionable: `w_j × (0 + τ_expect)` or `w_j × 1`. |
| **-1 → +1** | +2 | Sharp recovery | Drains detectors | Full reversal upward. Resets no-recovery counter. May follow a spike pattern if crash was recent. |
| **-1 → 0** | +1 | Partial recovery | Drains detectors (partial) | Moving toward neutral. Not yet aligned but improving. With τ_expect > 0, still produces small signal. |
| **-1 → -1** | 0 | No recovery | Accumulates | Each step adds to no-recovery counter. After ≥ C_min steps: escalation alert. Signal per step: `w_j × (1 + τ_expect)`. |

**Peripheral values (w_j < w_min):** monitored only for rise (sustained positive alignment).

| From → To | Signal name | Action | Notes |
|---|---|---|---|
| **0 → +1** | Rise onset | Watch | Start counting. Not flagged until sustained ≥ C_min steps. |
| **+1 → +1** | Rise continues | Watch / flag | If count ≥ C_min: profile divergence alert. Coach asks if priorities shifted. |
| **Any → 0 or -1** | — | None | Not monitored. User didn't declare this value as important. |
| **-1 → -1** | — | None | Sustained negative on a peripheral value is not a concern. |

All 9 transitions are covered. The key asymmetry: core values trigger on *decline* (negative transitions and dormancy), peripheral values trigger only on *rise* (sustained positive emergence).

### Single transitions are not signals

A single-step transition (e.g., +1 → -1 this week) is a raw observation, not a signal. **No single transition triggers a final interpretation by itself.** Signals are multi-step patterns built from sequences of transitions. Even the sharpest crash (+1 → -1) fires an immediate alert but waits 1-2 steps before the interpretation is locked in — if the next week is +1 (spike), it was noise.

**Multi-step signal patterns (core values):**

| Signal | Required sequence | Min steps | Trigger condition |
|---|---|---|---|
| **Crash** | Contains +1→-1 or 0→-1 | 2 | Single-step drop passes size gate: `w_j × \|Δ\| > δ_j`. Alert fires immediately but interpretation waits for next steps. |
| **Spike** | Crash → recovery within 1-2 steps | 3-4 | Score returns to +1 or 0 within 2 steps after crash. Retracts the crash alert → noise. |
| **Fade** | +1→0 → sustained 0s | ≥ 1 + C_min | C_min consecutive confident 0s on a previously active dimension. With τ_expect > 0, each 0 accumulates soft drift signal. |
| **No recovery** | Crash → sustained -1s | ≥ 1 + C_min | After crash, score stays < τ_low for ≥ C_min consecutive confident steps. Escalation alert. |
| **Onboarding gap** | -1 or 0 from day one, never +1 | ≥ 3 | Core value with no positive confident score in first N entries. Not a transition — absence of expected activation. |

**Multi-step signal patterns (peripheral values):**

| Signal | Required sequence | Min steps | Trigger condition |
|---|---|---|---|
| **Rise** | 0→+1 or -1→+1 → sustained +1s | ≥ C_min | C_min consecutive confident +1 scores on a dimension with w_j < w_min. Profile divergence alert. |

The detectors (EMA, CUSUM, etc.) are the mechanism that tracks these multi-step patterns. Each single-step transition updates detector state; the detector fires when accumulated state crosses a threshold. This is why duration is a gate, not a filter — the system must observe multiple transitions before classifying.

### Drift detection approach taxonomy

There are five categories of drift detection, ranging from hand-coded rules to fully learned models. The rule-based category contains the 6 sub-approaches being evaluated in this experiment. The other four are ML alternatives that trade interpretability and low data requirements for flexibility and unified detection.

#### Category 1: Rule-based (selected for POC)

Hand-coded thresholds define what counts as a crash, fade, or rise. No learning — the signal taxonomy (Section 1) is specified upfront and detectors check whether each pattern's conditions are met.

**Sub-approaches within rule-based:**

| Sub-approach | Mechanism | What it detects well |
|---|---|---|
| **Baseline (dual-trigger)** | Crash: single-step drop > δ_j. No-recovery: score < τ_low for ≥ C_min steps. | Sharp crashes, sustained misalignment |
| **EMA** | Exponentially weighted mean of drift signal `d_j`. Alert when EMA > threshold. | Gradual trends (fade), with forgetting of old data |
| **CUSUM** | Cumulative sum of deviations from allowance k. Alert when sum > h. | Sustained small shifts that individually look harmless |
| **Cosine Similarity** | Angle between profile weights `w_u` and alignment vector `â_t`. Alert when cosine < threshold. | Holistic misalignment across multiple dimensions simultaneously |
| **Control Charts** | Mean ± nσ computed from a baseline period. Alert when score breaches lower control limit. | Deviations from the user's own established baseline |
| **KL Divergence** | Distribution distance between a baseline window and a recent window. | Both level shifts and shape changes (e.g., stable +1 → volatile mix of -1 and +1) |

These are all rule-based because they use fixed formulas with tunable parameters — none of them learn from data. They differ in **aggregation strategy** (exponential weighting vs. cumulative sum vs. distribution comparison) and **memory** (EMA forgets old data exponentially; CUSUM accumulates indefinitely until reset; Control Charts compare against a fixed baseline).

**Why rule-based is right for the POC:**
- **Transparency:** Every alert is explainable — "Benevolence dropped below -0.4 for 3 consecutive weeks."
- **Low data requirement:** Works with 24 annotated personas (~380 annotations).
- **Tunable:** Grid search over threshold combinations is tractable with 6 sub-approaches × ~10 parameters.

**Maya example (rule-based):** Maya's Benevolence crash at week 7 (+1 → -1). The baseline fires immediately (drop of 2.0 > δ=0.5). EMA jumps from ~0 to 0.45 × 1.0 = 0.45 (above threshold). CUSUM jar fills by 0.45 in one step. All three agree: crash. If weeks 8-9 stay at -1, the baseline's no-recovery counter hits C_min=3 at week 10. EMA stays elevated. CUSUM keeps accumulating. Consensus: crisis.

#### Category 2: Bayesian Online Changepoint Detection (BOCPD)

Instead of hand-coded signal patterns, BOCPD models the **posterior probability that a changepoint occurred at each time step**. It maintains a "run length" distribution — how long since the last change in the generative process. When a new score arrives, BOCPD updates the probability that the current run continues vs. a new regime started.

**How it works on alignment scores:**
- Assume scores within a regime are drawn from a distribution (e.g., Normal with unknown mean and variance, or a categorical over {-1, 0, +1}).
- At each step t, compute P(changepoint at t | data so far) using Bayesian updating.
- When P(changepoint) exceeds a threshold, flag it — no need to specify whether it's a "crash" or "fade." The model infers the change type from the posterior.

**Maya example (BOCPD):** For weeks 1-6, BOCPD learns a regime with mean ≈ +1, low variance. At week 7 (score = -1), the posterior probability of a changepoint spikes — the new observation is extremely unlikely under the current regime. P(changepoint at week 7) ≈ 0.95. The system flags it without needing a crash threshold δ. For a fade (+1 → 0 → 0 → 0), the changepoint probability rises more gradually — each 0 is unlikely under a mean=+1 regime, and by week 3-4 of dormancy the posterior confidently identifies a regime shift.

**Pro:**
- Unified framework — crash, fade, and no-recovery are all "changepoint with different posterior shapes." No signal taxonomy needed.
- Naturally handles the "how many steps before it's real?" question — the posterior probability *is* the confidence that something changed.
- Adapts to each user's baseline (the regime parameters are learned, not hand-coded).

**Con:**
- Requires choosing a prior and likelihood model. For discrete {-1, 0, +1} scores, the standard Gaussian assumption is a poor fit — need a categorical or ordinal likelihood.
- With 10-15 data points per user per dimension, the posterior will be wide. Early regime estimates are uncertain.
- Less interpretable than "Benevolence dropped below -0.4 for 3 weeks." The Coach would need to translate P(changepoint) into a user-facing message.
- Harder to incorporate profile weights `w_u` — BOCPD operates per-dimension independently. Weighting by importance requires a wrapper layer.

**Data requirement:** Moderate. BOCPD works with conjugate priors (e.g., Dirichlet-Categorical for {-1,0,+1} scores), so it can start with informative priors and update per-user. ~50+ personas would help set reasonable hyperpriors.

#### Category 3: Gaussian Process (GP) regression

Model each dimension's alignment trajectory as a Gaussian Process — a distribution over smooth functions. The GP learns a trend with uncertainty bands from the data. Drift = observations falling outside the GP's predictive interval.

**How it works on alignment scores:**
- Fit a GP to the (week, score) pairs for each dimension per user.
- The GP predicts the expected score at week t+1 with a confidence interval.
- If the observed score falls outside the interval, flag it as anomalous.
- The GP's uncertainty naturally widens during gaps (skipped weeks) and narrows where data is dense.

**Maya example (GP):** For weeks 1-6 on Benevolence (all +1), the GP fits a flat function at mean ≈ +1 with tight bands (±0.2). At week 7 (score = -1), the observation is ~10σ below the GP's prediction → strong anomaly. For the fade on Self-Direction (+1 → 0 → 0 → 0), the GP's predicted mean gradually shifts downward as 0s accumulate, but the first +1 → 0 transition already falls outside the band from the +1 regime.

**Pro:**
- Uncertainty-aware natively — no separate MC Dropout layer needed for the detector (though the Critic still uses it for scoring confidence).
- Handles irregular time gaps naturally (GPs work with arbitrary input spacing — a user who journals sporadically doesn't break the model).
- The predictive interval adapts to each user's trajectory shape, not a fixed baseline period.

**Con:**
- With only 10-15 data points per dimension, the GP posterior will be wide — may rarely flag anything as anomalous because the uncertainty bands are too generous.
- Requires choosing a kernel (RBF? Matérn? Periodic?). The wrong kernel imposes smoothness assumptions that may not hold for {-1, 0, +1} discrete scores.
- Computationally heavier than rule-based (O(n³) for exact GP, though n is small here).
- Like BOCPD, doesn't naturally incorporate profile weights `w_u`.

**Data requirement:** Moderate per user. The GP needs enough points to learn the kernel hyperparameters. With < 8 entries, the posterior is prior-dominated (which may be acceptable if the prior encodes "expect alignment on core values").

#### Category 4: Regime-switching / Hidden Markov Models (HMM)

Explicitly model latent regimes that the user transitions between. The model learns: (a) what each regime looks like (emission distribution), and (b) the probability of switching between regimes (transition matrix).

**How it works on alignment scores:**
- Define K latent states, e.g., K=3: "aligned" (emissions centered on +1), "struggling" (emissions centered on 0, high variance), "drifting" (emissions centered on -1).
- Learn emission parameters and transition probabilities from data using EM (Baum-Welch) or MCMC.
- At each time step, infer the most likely regime via Viterbi decoding or forward-backward algorithm.
- A crash = transition from aligned → drifting. A fade = aligned → struggling → drifting. A spike = aligned → drifting → aligned.

**Maya example (HMM):** The model infers Maya is in the "aligned" regime for Benevolence weeks 1-6 (emissions: +1, +1, +1, +1, +1, +1). At week 7 (-1), the Viterbi path switches to "drifting." If weeks 8-9 return to +1, the path switches back to "aligned" — a spike. If weeks 8-10 stay at -1, the model confidently classifies weeks 7+ as the "drifting" regime. The signal taxonomy (crash, fade, spike) **emerges from the learned transition patterns** rather than being hand-coded.

**Pro:**
- The signal taxonomy emerges from data. If real user behavior has patterns the hand-coded taxonomy doesn't cover, the HMM can discover them.
- Transition probabilities encode base rates — e.g., "aligned → drifting transitions are rare (P=0.05), so evidence must be strong." This replaces the threshold tuning problem with a learning problem.
- Can be extended to a hierarchical HMM across users, sharing regime parameters while allowing per-user transition rates.

**Con:**
- Needs per-user fitting or a hierarchical model. With 10 entries per user, per-user EM will overfit. A hierarchical Bayesian HMM across all users is more robust but significantly more complex.
- The number of regimes K must be specified. K=3 (aligned/struggling/drifting) mirrors the rule-based taxonomy, but the right K is not obvious.
- Emission parameters for discrete {-1, 0, +1} observations work well (categorical emissions), but the small state space means regimes may not be clearly separable.
- Less interpretable than rule-based for the Coach: "the Viterbi path switched to state 2" needs translation to user-facing language.

**Data requirement:** High. Per-user fitting needs ~30+ observations per dimension for stable EM convergence. A pooled model across all 204 personas is feasible if users share regime characteristics, but individual variation in transition rates reduces pooling benefit.

#### Category 5: Autoencoders (anomaly detection)

Train a neural network to reconstruct "normal" alignment trajectories. At inference, high reconstruction error = anomaly = potential drift.

**How it works on alignment scores:**
- Encode a window of W recent scores (across all 10 dimensions) into a latent vector, then decode back to the original scores.
- Train on windows from the "aligned" portion of trajectories (e.g., first N entries before any drift).
- At inference, if reconstruction error exceeds a threshold, the current window is anomalous.

**Maya example (autoencoder):** Train on windows where Maya's scores are [+1, +1, +1, ...] across core dimensions. When the window includes the week-7 crash (Benevolence = -1 while others remain +1), reconstruction error spikes — the network can't reconstruct this pattern from the latent space learned on aligned windows.

**Pro:**
- Captures cross-dimension patterns — e.g., Self-Direction fading while Achievement rises might be a recognizable "career-shift" pattern in the latent space.
- No signal taxonomy needed. Any deviation from learned "normal" is flagged.

**Con:**
- **Data requirement is the fundamental bottleneck.** Needs hundreds of trajectory windows to learn meaningful representations. With 24 personas × ~10 entries each = ~240 windows, an autoencoder will memorize rather than generalize.
- The {-1, 0, +1} discrete score space has only 3^10 = 59,049 possible alignment vectors. An autoencoder is overkill for this — simpler distance metrics (like cosine similarity, already a rule-based sub-approach) capture the same information.
- Requires a clear definition of "normal" training data. If the training set includes drift windows, the autoencoder learns to reconstruct drift as normal.
- Least interpretable of all options. "Reconstruction error = 0.73" tells the Coach nothing about *what* changed.

**Data requirement:** High. Would need ~500+ diverse trajectory windows across many users to learn generalizable representations. Not feasible with current dataset. Becomes viable if Twinkl scales to hundreds of real users with months of journaling history.

#### Comparison

| Category | Online? | Data needed | Interpretability | Signal taxonomy | Profile weights `w_u` |
|---|---|---|---|---|---|
| **1. Rule-based** | Yes | Low (24 personas) | High | Hand-coded | Native (weighted formulas) |
| **2. BOCPD** | Yes | Moderate (~50 personas) | Medium | Emergent (changepoint type) | Requires wrapper |
| **3. GP regression** | Yes | Moderate per user (~8+ entries) | Medium | Not applicable (anomaly-based) | Requires wrapper |
| **4. Regime-switching / HMM** | Yes | High (~30+ per user, or pooled) | Medium | Emergent (learned regimes) | Can encode in emissions |
| **5. Autoencoders** | No (batch) | Very high (~500+ windows) | Low | Not applicable (anomaly-based) | Encoded in training data |

#### Selection rationale

**For the POC: Category 1 (rule-based).** The dataset (24 annotated personas, ~380 annotations) is too small for learned models. The rule-based taxonomy (crash, fade, spike, no recovery, onboarding gap, rise) is comprehensive for the known signal patterns, and the consensus-based evaluation framework tests whether the 6 sub-approaches can reliably detect them.

**First upgrade path: BOCPD.** When the dataset grows to ~50+ personas with confirmed changepoints, BOCPD replaces the hand-coded signal taxonomy with a unified probabilistic framework. This is the most natural next step because it's online (works per-entry like the current pipeline), requires moderate data, and subsumes crash/fade/no-recovery into a single model.

**Second upgrade path: Regime-switching / HMM.** When enough per-user data exists (~30+ entries per user), a hierarchical HMM can learn regime structure that the hand-coded taxonomy might miss. This is especially valuable if real users show patterns not covered by the current taxonomy.

**GP regression** is a lateral option — useful if irregular journaling (skipped weeks, bursts of entries) proves to be a problem that the rule-based weekly aggregation doesn't handle well.

**Autoencoders** are a long-term option only viable at scale. They become interesting when Twinkl has hundreds of users and the cross-dimension interaction patterns (e.g., "career shift" = Self-Direction down + Achievement up) are the primary detection target.

### Mathematical primitive: rolling standard deviation

Four of the five candidate approaches (EMA, CUSUM, Control Charts, KL Divergence) share a common mathematical foundation: they detect when the **variability or level of recent scores deviates from a baseline**. Rolling standard deviation captures this directly.

For a window of the last W weekly scores on dimension j:

$$
\bar{a}^{(j)}_W = \frac{1}{W} \sum_{k=0}^{W-1} \hat{a}^{(j)}_{t-k}, \qquad
\sigma^{(j)}_W = \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} \big(\hat{a}^{(j)}_{t-k} - \bar{a}^{(j)}_W\big)^2}
$$

How each approach relates to this primitive:

| Approach | What it computes | Relationship to rolling σ |
|---|---|---|
| **EMA** | Exponentially weighted mean of drift signal | Tracks level shift. Equivalent to rolling mean with exponential decay. High σ_W indicates instability that EMA smooths. |
| **CUSUM** | Cumulative sum of deviations from allowance k | Detects sustained level shift. Sensitive to the same mean shifts that move rolling mean away from baseline. |
| **Control Charts** | Mean ± n×σ from a baseline period | Directly uses σ. A breach means current scores are > n standard deviations from baseline behavior. |
| **KL Divergence** | Distribution distance between baseline and recent window | Detects both level and shape changes. High KL ↔ baseline and recent distributions have different means or spreads. |
| **Cosine Similarity** | Directional agreement between w_u and â_t | Different primitive: angle, not variance. Complementary to σ-based approaches. |

This means: if rolling σ_W on a core dimension spikes (scores become volatile) or the rolling mean drops (level shift), most approaches will fire — which is why consensus works as a ground truth proxy. The approaches disagree mainly on **sensitivity** (how much shift is enough) and **memory** (how far back they look).

### Sequential detection order: where does evolution check go?

After Gate 1 (confidence — must come first since nothing works on uncertain data), four checks remain: **signal type** (crash/fade/rise), **evolution** (low-volatility directional shift), **size** (magnitude × weight), and **duration** (sustained or blip). The question is what order to apply them.

The [value evolution detection](../evolution/01_value_evolution.md) spec uses a volatility layer to distinguish evolution from drift: low volatility + sustained directional change = evolution candidate, high volatility + oscillating = behavioral struggle, sudden level shift = crash.

#### Order A: Evolution → Drift

```
Confident score → Is it evolution? (volatility check) → If not → Size → Duration → Awareness
```

- **Pro:** Prevents false drift alerts on genuinely evolving values. Evolution dimensions excluded early, drift detectors stay clean.
- **Con:** Requires enough history to assess volatility *before* any drift detection runs. For a new user with 3 weeks of data, you can't distinguish evolution from fade — both look like gradual decline. **Blind during early weeks.**
- **Con:** A crash (+1 → -1 in one step) is clearly not evolution, but this order still runs the volatility check on it unnecessarily.

#### Order B: Signal-type → Evolution/Drift (recommended)

```
Confident score → Classify signal pattern (crash/fade/rise/spike) →
  If crash or spike: skip evolution check → Size → Duration → Awareness
  If fade or sustained shift: check evolution (volatility) → Size → Duration → Awareness
```

- **Pro:** Crashes are fast-pathed — no unnecessary evolution check on single-step events. Evolution check only runs where it's meaningful (gradual patterns).
- **Pro:** Works from week 1. No ramp-up period needed for crash detection.
- **Pro:** Avoids polluting drift detector state with evolution dimensions — they're checked before entering fade/no-recovery detectors.
- **Con:** Requires defining signal type before checking size, which means small signals get classified too. A 0→0 step on a w=0.03 peripheral value gets classified as "fade" before being filtered by size. Minor cost — size filter discards it immediately after.

#### Order C: Size → Duration → Evolution/Drift

```
Confident score → Is it big enough? → Is it sustained? → Is it evolution or drift?
```

- **Pro:** Filters noise early. Small signals and single-step blips never reach the evolution/drift distinction.
- **Pro:** By the time you ask "evolution or drift?", you know it's a real, sustained pattern — the question is meaningful.
- **Con:** Evolution is only detected *after* duration gate fires. The system accumulates drift detector state (EMA/CUSUM) during the waiting period. If it turns out to be evolution, those weeks of accumulation were wasted and need to be unwound or ignored.

#### Order D: Duration → Signal-type → Evolution

```
Confident score → Update all detectors blindly → When any fires → Classify → Check evolution
```

- **Pro:** Simplest implementation. All scores feed all detectors. Classification only at alert time.
- **Con:** Evolution dimensions pollute drift detector state for weeks before being identified and excluded. A smooth value shift accumulates in CUSUM/EMA as if it were drift, potentially triggering false alerts before the evolution check runs.

#### Comparison

| Order | Early-week behavior | Crash handling | Fade handling | Complexity | False alert risk |
|---|---|---|---|---|---|
| **A: Evolution-first** | Blind (can't assess volatility) | Unnecessary check | Good — excludes evolution early | Moderate | Low (after ramp-up) |
| **B: Signal-type-first** | Works immediately | Fast-pathed, clean | Good — evolution check on gradual only | Moderate | Low |
| **C: Size-then-duration** | Works immediately | Passes (crashes are big) | Late — detector state polluted during wait | Simple | Medium |
| **D: Duration-first** | Works immediately | Fires when threshold met | Evolution pollutes detectors | Simplest | Highest |

#### Recommendation: Order B

Order B handles the two main failure modes well: crashes are fast-pathed (no waiting for volatility data that doesn't apply to single-step events), and fades/sustained shifts get the evolution check where it matters. It works from week 1 and avoids polluting detector state with evolution dimensions.

The full decision tree from Section 1 uses this ordering: Gate 1 (confidence) → signal-type classification (crash vs. gradual) → evolution check (gradual patterns only) → Gates 2–4 (size, duration, awareness). Dimensions classified as EVOLUTION are excluded from drift trigger evaluation and do not count toward consensus crisis labels.

### Proposed threshold ranges for grid search

| Parameter | Symbol | Range | Rationale |
|---|---|---|---|
| Uncertainty ceiling | ε_j | {0.2, 0.3, 0.4} | Gate 1: scores with σ ≥ ε are excluded from detectors entirely. 0.3 is the starting value from the eval spec. |
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

### Step 1: Establish ground truth via approach consensus

Rather than manually labeling crisis points, use **cross-approach agreement** as a proxy for ground truth. The logic: if most approaches independently flag the same (persona, step, dimension), it's likely a real signal regardless of which individual approach is "correct."

For each (persona_id, t_index, dimension), run all 6 approaches (baseline + 5 candidates) and count how many fire an alert:

```
consensus_score = number of approaches that flag this (t, dim)  # 0–6

consensus_crisis:
  strong  = consensus_score ≥ 4   (majority agreement → high-confidence crisis)
  weak    = consensus_score ∈ {2, 3} (split opinion → ambiguous)
  none    = consensus_score ≤ 1   (at most one approach → not a crisis)
```

This produces labels without manual work and covers all personas with ≥5 steps. The labels are then used to evaluate each individual approach: which one best predicts what the consensus identifies?

**Why consensus works here:** The 6 approaches have genuinely different philosophies (memoryless vs. stateful, per-dimension vs. holistic, threshold vs. distributional). Agreement across philosophically different methods is stronger evidence than agreement across similar methods.

**Limitation:** If all approaches share a blind spot (e.g., none detect fades because they all use τ_expect=0), consensus will miss it too. Phase 3 (absence-aware variant) addresses this.

### Step 2: Score all 6 approaches against consensus (5 candidates + baseline)

The existing dual-trigger rules from [`drift_detection_eval.md`](drift_detection_eval.md) and [`04_uncertainty_logic.md`](../vif/04_uncertainty_logic.md) are the simplest possible implementation and serve as **Approach 0 (Baseline)**:

| Trigger | Rule | Parameters |
|---|---|---|
| **Crash** | `V_{t-1} - V_t > δ_j` (single-step drop exceeds threshold) | δ_j = 0.5 |
| **No recovery** | `V_t < τ_low` for ≥ C_min consecutive steps | τ_low = -0.4, C_min = 3 |

Both gated by uncertainty: only fire when `σ_t < ε_j` (ε_j = 0.3).

The 5 candidate approaches (EMA, CUSUM, Cosine, Control Charts, KL Divergence) are compared against this baseline. The experiment must answer: **does the added complexity of a stateful detector improve hit rate or reduce FPR enough to justify it over simple thresholds?**

If the baseline already meets targets, the simpler approach wins — no EMA/CUSUM needed. If it doesn't, we know specifically where it falls short (e.g., misses slow fades, too many crash false alarms) and which candidate addresses that gap.

Run each detector on the annotated personas (using human annotation means as "perfect Critic" input). For each approach × parameter combination, score against consensus labels:

| Metric | Definition | Target |
|---|---|---|
| **Hit Rate** | Fraction of consensus-crisis points this approach flagged | ≥ 80% |
| **Precision** | Fraction of this approach's alerts that are consensus crises | > 60% |
| **F1** | Harmonic mean of hit rate and precision | > 0.5 |
| **FPR** | Fraction of consensus-none points this approach incorrectly flagged | < 20% |
| **First-alert latency** | Steps after crisis onset before this approach's first alert | Lower is better |

An approach that agrees with consensus on most points but adds unique early detections (flagging crises 1-2 steps before consensus) is particularly valuable — it catches real problems faster.

### Step 3: Re-run with Critic predictions

Replace human annotations with `critic.predict_with_uncertainty()` output. This measures degradation from Critic noise. The gap between Step 2 (human annotations) and Step 3 (Critic predictions) quantifies how much Critic accuracy improvement would improve drift detection.

Importantly, Step 3 enables uncertainty gating — the baseline's `σ_t < ε_j` constraint can only be tested with MC Dropout predictions, not human annotations (which have no uncertainty estimate). This may change the relative ranking: an approach that performs well on clean data but is sensitive to noise may lose to one that is more robust.

### Step 4: Select approach combination

Use Step 2-3 results to select approaches for each signal type:

- **Crash detection:** The approach with highest precision on consensus-crash points (few false alarms on sudden events). Candidates: Baseline simple threshold, Control Charts (LCL breach), or Cosine Similarity (instant direction reversal).
- **Fade / no-recovery detection:** The approach with highest hit rate on consensus points involving sustained patterns (catch slow decay even if some false positives). Candidates: Baseline consecutive-low counter, EMA, or CUSUM.

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

### Phase 1: Consensus ground truth + approach selection (1-2 days)

**Goal:** Run all 6 approaches on annotated personas, compute consensus labels, then score each approach against consensus.

1. Add baseline approach (simple crash/no-recovery thresholds) to the comparison notebook
2. Run all 6 approaches × parameter grid on human annotation means for personas with ≥5 steps
3. Compute consensus labels: strong (≥4 agree), weak (2-3 agree), none (≤1 agrees)
4. Score each approach against consensus (hit rate, precision, F1, FPR, latency)
5. If baseline meets targets (≥80% hit rate, <20% FPR): stop — simple thresholds win
6. If not: select best crash approach + best fade/no-recovery approach from candidates

**Output:** Selected approach pair with tuned parameters, or confirmation that baseline suffices. Consensus labels stored as `logs/annotations/consensus_crisis_labels.parquet`.

### Phase 2: Critic-in-the-loop evaluation (1 day)

**Goal:** Measure degradation when using Critic predictions instead of human annotations.

1. Run the selected approach pair on Critic predictions (with MC Dropout uncertainty) for the same personas
2. Apply uncertainty gating: exclude entries where σ ≥ ε_j from detector state
3. Compare hit rate / precision / FPR against Phase 1 results
4. Quantify the "Critic noise penalty" — how much does detection degrade?

**Output:** Gap analysis showing which Critic improvements would most benefit drift detection.

### Phase 3: Absence-aware variant (0.5 day)

**Goal:** Test whether the τ_expect > 0 formula improves fade detection.

1. Re-run Phase 1 with `τ_expect ∈ {0.0, 0.2, 0.3}` as an additional grid parameter
2. Compare specifically on consensus points involving fade patterns (core value transitioning from +1 to sustained 0)
3. If it improves fade detection without increasing FPR, adopt it

**Output:** Decision on whether to include the absence-aware formula.

### Success metrics

| Metric | Target | Measured on |
|---|---|---|
| **Hit Rate** (consensus crises correctly flagged) | ≥ 80% | Phase 1 (human annotations) |
| **Precision** (alerts that are consensus crises) | > 60% | Phase 1 |
| **F1 per value dimension** | > 0.5 | Phase 1 |
| **FPR** (false alarm rate) | < 20% | Phase 1 |
| **Critic noise penalty** (hit rate drop Phase 1 → Phase 2) | < 15pp | Phase 2 |
| **First-alert latency** | ≤ 2 steps after crisis onset | Phase 1 |

### Blocking dependency

The drift detection eval spec notes that the Critic frontier (median QWK 0.362) is "not yet strong enough for reliable automated drift triggers." Phase 2 will quantify exactly how much this matters. If the Critic noise penalty exceeds 15 percentage points on hit rate, further Critic improvement should be prioritized before productionizing drift detection.

---

## References

- Schwartz, S. H. (2012). An Overview of the Schwartz Theory of Basic Values. *Online Readings in Psychology and Culture*, 2(1).
- Schwartz, S. H., & Bilsky, W. (1987). Toward a universal psychological structure of human values. *Journal of Personality and Social Psychology*, 53(3), 550-562.
- Hitlin, S., & Piliavin, J. A. (2004). Values: Reviving a dormant concept. *Annual Review of Sociology*, 30, 359-393.
- Rokeach, M. (1973). *The Nature of Human Values*. Free Press.
- Bardi, A., & Goodwin, R. (2011). The dual route to value change. *Journal of Cross-Cultural Psychology*, 42(2), 271-287.
