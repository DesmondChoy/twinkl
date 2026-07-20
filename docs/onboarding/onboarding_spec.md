# Onboarding Specification: BWS Values Assessment

## 1. Purpose & Scope

### What Onboarding Achieves

The onboarding flow solves Twinkl's **cold-start problem**: before a user has written any Journal Entries, Twinkl needs an initial Profile containing declared Core Values and value weights. Without it, later Journal Entries have no declared priorities to compare against.

The onboarding uses **Best-Worst Scaling (BWS)** — a forced-choice psychometric technique — to elicit a user's value priorities across the 10 Schwartz value dimensions. Combined with a structured goal selection, this produces:

1. A **value weight vector** (`w_u ∈ ℝ^10`) that initializes the user's VIF profile
2. A discrete set of **Core Values** (`top_values`) containing every value tied for the highest exposure-normalized BWS score
3. A **primary goal/tension** that focuses the Weekly Coach's initial monitoring
4. A **confidence baseline** that signals how much to trust explicit vs. behavioral data

These two value outputs have different jobs. The VIF Critic keeps the full
graded 10-dimensional weight vector for conditioning. Drift v1 uses the Core
Values stored in `top_values` as the eligibility gate: only two consecutive
Conflicts on the same Core Value can form Drift.

### What's Out of Scope

- **Contextual story / narrative input** — The user's first personal narrative is captured via a guided journal prompt *after* onboarding, not during it. Onboarding is structured choice only.
- **VIF integration mechanics** — This spec defines the semantic contract for `weights` and `top_values`; wiring both fields into the runtime remains future work documented in [Section 8: Future Considerations](#8-future-considerations).
- **Adaptive item selection** — All users see the same 6 BWS sets. Computerized adaptive testing (CAT) is a future optimization.

### Cross-References

- [PRD](../prd.md) — Product requirements and implementation status
- [VIF Concepts & Roadmap](../vif/01_concepts_and_roadmap.md) — Value Identity Function theory
- [VIF System Architecture](../vif/02_system_architecture.md) — State representation and inference flow
- [VIF Worked Example](../vif/example.md) — Sarah's journey through Twinkl (includes BWS onboarding walkthrough)

---

## 2. Design Rationale

### Why BWS Over Alternatives

| Method | Problem | BWS Advantage |
|--------|---------|---------------|
| **Likert scales** ("Rate how important X is, 1–7") | Social desirability bias — users rate everything as important | Forced trade-offs make "all high" impossible |
| **Select all that apply** | Same problem — no cost to selecting everything | Each "most" choice implicitly deprioritizes 3 others |
| **Full ranking** (drag 10 items into order) | Cognitively demanding; poor mobile UX | Only 2 decisions per screen (most/least) |
| **Simple pick-2** | Loses information about non-selected values; binary rather than graded | BWS produces a full ranking with interval-scale properties |

### Social Desirability Bias: Two Mechanisms

Values assessment is uniquely vulnerable to social desirability bias, which operates through two distinct mechanisms (Paulhus, 1984):

1. **Impression management** — Consciously presenting oneself favorably ("I *should* say I value fairness")
2. **Self-deceptive enhancement** — Genuinely believing an inflated self-image ("I *do* value fairness... I just never act on it")

BWS addresses both:
- **Impression management**: When all 4 options in a set are roughly equally "good," there's no obviously desirable answer to fake toward. The trade-off forces honest prioritization.
- **Self-deceptive enhancement**: Concrete, behaviorally-anchored phrases (see [Section 4](#4-bws-item-bank)) ground choices in recognizable life patterns rather than abstract ideals.

### Why PVQ21 Rewritten

The Portrait Values Questionnaire (PVQ21; Schwartz et al., 2001) is the standard instrument for measuring Schwartz values. However, its items are designed for survey research, not mobile UX:

- **Original PVQ21 format**: "He/she thinks it is important to be rich. He/she wants to have a lot of money and expensive things." (Third-person, long, two sentences)
- **BWS card format needed**: First-person, single phrase, selectable by tap or drag on mobile

We preserve the **conceptual content** of PVQ21 items while rewriting for:
- First-person voice ("Having the freedom to choose my own path")
- Concrete behavioral anchoring (actions and feelings, not abstract ideals)
- Similar social desirability across items within each set
- No Schwartz value labels or jargon

### Why 6 Sets of 4

The Balanced Incomplete Block Design (BIBD) for BWS requires each item to appear multiple times across sets to produce reliable scores. With 10 values:

- **6 sets × 4 items = 24 item slots**
- Each value appears **2–3 times** across all sets
- Each set mixes values from **different higher-order quadrants** (Openness vs Conservation, Self-Enhancement vs Self-Transcendence) to force meaningful trade-offs
- **~2 minutes total** (approximately 20 seconds per set) — well within mobile attention spans

### Research Backing

- **Vignette methodology** (Alexander & Becker, 1978): Short behavioral descriptions outperform abstract value labels for eliciting authentic responses
- **Behavioral anchoring** (Schwartz, 2003): PVQ's portrait approach works because it grounds values in concrete person-descriptions; our BWS phrases preserve this property
- **BWS psychometric properties** (Louviere et al., 2015): BWS produces ratio-scale measurement from simple max-diff choices, with better discrimination than rating scales

---

## 3. User Flow

### 3.1 Direct Entry

The user lands directly on Set 1. There is no welcome screen, implementation
status, storage claim, or developer-facing preamble. One short sentence explains
the immediate action and why the choices matter. The single progress label changes
with the phase: `Values · n of 6`, `Your focus`, then `Your compass`. The mid-flow
mirror is an interstitial after Set 3 and keeps the label `Values · 3 of 6`.

### 3.2 BWS Sets 1–3

Each set is a literal deck of four animated cards. The cards have physical
proportions, subtle fan angles, and a distinct code-native illustration for each
Schwartz value. They show only the first-person phrase, never a numbered card
header. Desktop keeps all four choices in one horizontal row. Mobile uses a
readable 2×2 grid without horizontal page scrolling.

The four cards begin in the selection area. The user moves one card into a
centered `Most` box above the choices, then a different card into a centered
`Least` box below them. The remaining two cards stay in the selection area.

**Interaction model:**
- The `Most` and `Least` boxes are visible before any card is moved
- Pointer and touch dragging move cards between the selection area and either box
- Tapping a card selects it and activates explicit `Most` and `Least` placement targets
- Tapping either placement target moves the selected card; tapping a placed card returns it to the selection area
- A placed card can be dragged back to the selection area or directly to the other box
- Moving a card into an occupied box returns the previous card to the selection area
- Cards keep the same portrait dimensions, illustration scale, and caption treatment after placement
- Placement triggers a short directional settle before a clear, color-matched pulse makes each active choice apparent
- `Continue` activates only after both selections exist
- Card order is randomized once per user session to reduce position bias
- `M` and `L` move a focused card into either box; Backspace, Delete, or Arrow Down returns it
- Keyboard focus is visible, and reduced-motion preferences disable decorative movement

When a phrase first repeats in Set 3, the instruction briefly explains that some
cards return and asks the user to choose what feels true in the current group.
Later repeated phrases need no additional explanation.

### 3.3 Mid-flow Mirror (After Set 3)

After Set 3, Twinkl computes a preliminary result and shows the current
highest- and lowest-scoring phrases. The mirror is informational: it says a
pattern is beginning to appear and offers one `Keep going` action.

Twinkl does **not** ask whether it placed a value too high or low. There is no
promote/demote correction screen, approval gate, or user-authored score override.
The six forced-choice card responses are the sole value-ranking input.

### 3.4 BWS Sets 4–6

Same interaction model as Sets 1–3. The user continues through the remaining 3 sets.

### 3.5 Goal Selection

After all BWS sets, the user selects their primary tension/goal from structured categories:

```
┌─────────────────────────────────┐
│                                 │
│   What brought you here?        │
│                                 │
│   Pick the one that resonates   │
│   most right now.               │
│                                 │
│   ┌───────────────────────┐     │
│   │ I'm stretched too thin│     │
│   │ between work and      │     │
│   │ everything else       │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ I'm going through a   │     │
│   │ career or life        │     │
│   │ transition            │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ I want to be more     │     │
│   │ present for people    │     │
│   │ I care about          │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ I'm neglecting my     │     │
│   │ health or wellbeing   │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ I feel stuck or       │     │
│   │ unclear about my      │     │
│   │ direction             │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ I want to make more   │     │
│   │ room for what matters │     │
│   │ to me                 │     │
│   └───────────────────────┘     │
│                                 │
│        [ Next → ]               │
│   ○ ○ ○ ○ ○ ○ ● ○  (progress)  │
└─────────────────────────────────┘
```

Single selection. See [Section 6: Goal Categories](#6-goal-categories) for how each goal maps to Weekly Coach monitoring priorities.

### 3.6 End Summary

```
┌─────────────────────────────────┐
│                                 │
│   What sits at the center.      │
│                                 │
│   Core Values:                  │
│   [Every value tied for the     │
│    highest score]               │
│                                 │
│   Your focus:                   │
│   "[Goal display text]"         │
│                                 │
│   This is just a starting       │
│   point — your compass will     │
│   keep learning as you journal. │
│                                 │
│   [ Set my compass ]            │
│                                 │
│   ○ ○ ○ ○ ○ ○ ○ ●  (progress)  │
└─────────────────────────────────┘
```

The summary presents Twinkl's inferred Core Values and the selected goal under
the progress label `Your compass`. Every tied Core Value has equal visual weight;
there are no ordinal numbers that imply a ranking among ties. It has one
confirmation action and does not ask the user to rank, promote, or demote values
directly.

### 3.7 Transition to First Guided Journal Prompt

After confirming the summary, onboarding presents a clear handoff to the first
guided Journal Entry. Activating it opens the first writing prompt, not another
onboarding explanation or a restart control. The standalone React POC also
exposes the confirmed Profile through an `onStartJournal` callback and a
`twinkl:start-first-journal` browser event so the host application can perform
the transition. The POC currently opens one generic prompt. A future Journaling
UI may tailor that prompt using the user's Core Values and selected goal.

Example transition:

```
┌─────────────────────────────────┐
│                                 │
│   Let's start your first        │
│   Journal Entry.                │
│                                 │
│   Think about the past week.    │
│   When was a moment where you   │
│   felt most like yourself?      │
│                                 │
│   ┌───────────────────────┐     │
│   │                       │     │
│   │   [text input area]   │     │
│   │                       │     │
│   └───────────────────────┘     │
│                                 │
│     [ Save Journal Entry → ]    │
└─────────────────────────────────┘
```

The Journal Entry editor is **not** part of the onboarding spec; it belongs to the
journaling module. Onboarding owns the visible handoff and the confirmed Profile
output described in [Section 7](#7-data-output-schema).

### 3.8 Privacy Language

The onboarding UI makes no `on-device` or `private on this device` claim. The
standalone POC does not call a model provider, but deployed Twinkl is expected to
send user data to LLM-backed services. A future product privacy notice must
describe the deployed data path accurately rather than inherit the POC's local
storage behavior.

---

## 4. BWS Item Bank

### Set Design

Each set mixes values from different Schwartz higher-order dimensions (Openness to Change vs Conservation, Self-Enhancement vs Self-Transcendence) so that every choice forces a meaningful trade-off between fundamentally different motivations.

| Set | Item 1 | Item 2 | Item 3 | Item 4 | Design Rationale |
|-----|--------|--------|--------|--------|------------------|
| 1 | Security | Self-Direction | Achievement | Benevolence | One value from each higher-order quadrant |
| 2 | Stimulation | Power | Conformity | Universalism | One value from each higher-order quadrant |
| 3 | Hedonism | Tradition | Self-Direction | Power | Cross-quadrant contrasts |
| 4 | Achievement | Benevolence | Stimulation | Conformity | Cross-quadrant contrasts |
| 5 | Security | Universalism | Hedonism | Tradition | Cross-quadrant contrasts |
| 6 | Self-Direction | Stimulation | Universalism | Security | Openness vs Conservation focus |

### Balance Matrix

| Schwartz Value | Higher-Order Dimension | Appearances | Sets |
|----------------|----------------------|-------------|------|
| Self-Direction | Openness to Change | 3 | 1, 3, 6 |
| Stimulation | Openness to Change | 3 | 2, 4, 6 |
| Hedonism | Openness / Self-Enhancement | 2 | 3, 5 |
| Achievement | Self-Enhancement | 2 | 1, 4 |
| Power | Self-Enhancement | 2 | 2, 3 |
| Security | Conservation | 3 | 1, 5, 6 |
| Conformity | Conservation | 2 | 2, 4 |
| Tradition | Conservation | 2 | 3, 5 |
| Benevolence | Self-Transcendence | 2 | 1, 4 |
| Universalism | Self-Transcendence | 3 | 2, 5, 6 |
| **Total** | | **24** | |

Each value appears 2–3 times. Values with 3 appearances (Self-Direction, Stimulation, Security, Universalism) receive slightly more measurement precision — these were chosen because they span all four higher-order quadrants, ensuring balanced coverage of the circumplex.

### Card-Friendly BWS Phrases

Each phrase is adapted from PVQ21 concepts, rewritten for first-person voice and behavioral concreteness.

| Schwartz Value | BWS Card Phrase | PVQ21 Source Concept |
|----------------|----------------|---------------------|
| Self-Direction | Having the freedom to choose my own path | Independent thought and action; choosing, creating, exploring |
| Stimulation | Seeking new experiences and challenges | Excitement, novelty, and challenge in life |
| Hedonism | Enjoying life and having fun | Pleasure and sensuous gratification for oneself |
| Achievement | Making progress toward something meaningful | Personal success through demonstrating competence |
| Power | Having influence over how things go | Social status and control over people and resources |
| Security | Feeling calm and secure in my life | Safety, harmony, and stability of self and relationships |
| Conformity | Being someone others can count on to do the right thing | Restraint of actions likely to upset others or violate norms |
| Tradition | Honoring the customs and practices I was raised with | Respect and acceptance of cultural/religious customs and ideas |
| Benevolence | Being there for the people closest to me | Preserving and enhancing welfare of close others |
| Universalism | Making the world a fairer, better place | Understanding, tolerance, and protection for all people and nature |

### Item Wording Principles

1. **Concrete and behavioral** — Phrases describe recognizable life orientations, not abstract ideals
2. **First-person** — "Having the freedom..." not "It is important to have freedom..."
3. **No value labels** — Never uses the words "self-direction," "benevolence," etc.
4. **No jargon** — Accessible to any adult regardless of psychology background
5. **Similar social desirability** — All phrases are framed positively; none sounds obviously "better" than others within a set. Power is framed as "influence" (neutral) not "dominance" (negative). Conformity is framed as reliability, not compliance.

---

## 5. Scoring Logic

### BWS Counting

For each value *v*, across all sets where it appears:

```
net_count(v) = best_count(v) − worst_count(v)
score(v) = net_count(v) / appearances(v)
```

Where:
- `best_count(v)` = number of times the user selected *v* as "Most like me"
- `worst_count(v)` = number of times the user selected *v* as "Least like me"

The exposure-normalized `score` has range `[-1, +1]` for every value.
This keeps the six fixed sets while making values shown twice comparable with
values shown three times. `net_count` and `appearances` remain in the output for
reproducibility.

### Normalization to Weight Vector

Exposure-normalized scores are converted to a weight vector
`w_u ∈ ℝ^10` where all weights are positive:

```
# Step 1: Shift to non-negative
shifted(v) = score(v) − min(scores) + 1

# Step 2: Normalize to sum to 1
w_u(v) = shifted(v) / Σ shifted(all values)
```

The +1 in the shift ensures no value has zero weight (even the least-preferred value has a small positive weight, reflecting that all Schwartz values are present in all people to some degree).

Weights are stored to 8 decimal places. The final canonical value receives the
rounding residual so the serialized weights sum to `1.0` within normal
floating-point tolerance.

### Tie Handling

If multiple values have the same exposure-normalized score:
- They receive the same weight after normalization (no arbitrary tie-breaking)
- If the highest score is tied, every tied value is shown in the summary
- The Weekly Coach treats tied values as genuinely co-important

The final `top_values` field is the complete set of values tied for the highest
score. It is not a fixed top-two or top-three list. A
single highest-scoring value produces a one-value Core Value set; a tie
produces a larger set with every tied value retained.

Core Values are emitted in the canonical Schwartz order used by
`src/models/judge.py`, regardless of the order in which the user saw or chose
the cards. A flat result intentionally makes all ten values Core Values; the
flow records low differentiation in confidence metadata rather than silently
truncating the set.

### Confidence Estimation

The onboarding scoring logic estimates confidence in the BWS-derived profile based on:

1. **Response consistency**: If a user selects a value as "Most" in one set and "Least" in another, this signals low confidence in that value's placement
2. **Score spread**: Population standard deviation across the ten exposure-normalized scores. A flat profile suggests the BWS did not differentiate priorities well.

```
confidence = {
  "consistent": true/false,     # No value selected as both Most and Least
  "spread": float,              # Population std dev of scores
  "method": "response_consistency_population_spread_v1"
}
```

Confidence is descriptive metadata, not a calibrated probability or a runtime
gate. No user correction is applied to the scoring result.

---

## 6. Goal Categories

### Enumerated Tension Categories

| Category Key | Display Text | Primary Value Tensions to Monitor | Weekly Coach Monitoring Priority |
|-------------|-------------|----------------------------------|--------------------------|
| `work_life_balance` | I'm stretched too thin between work and everything else | Achievement vs Benevolence, Achievement vs Hedonism | Work-related Journal Entries; time allocation patterns |
| `life_transition` | I'm going through a career or life transition | Security vs Self-Direction, Tradition vs Stimulation | Decision-making Journal Entries; uncertainty/ambiguity language |
| `relationships` | I want to be more present for people I care about | Benevolence vs Achievement, Benevolence vs Self-Direction | Mentions of close others; presence/absence patterns |
| `health_wellbeing` | I'm neglecting my health or wellbeing | Hedonism vs Achievement, Security vs Stimulation | Health-related Journal Entries; self-care language |
| `direction` | I feel stuck or unclear about my direction | Self-Direction vs Security, Stimulation vs Conformity | Purpose/meaning language; expressions of stagnation |
| `meaningful_work` | I want to make more room for what matters to me | Self-Direction vs Conformity, Universalism vs Power | Fulfillment language; value-action gaps |

### How Goals Map to Weekly Coach Behavior

The selected goal does **not** override BWS-derived values. Once the Profile is
wired into the product runtime, the goal is intended to:

1. **Focus initial attention** — The Weekly Coach can prioritize Journal Entries related to the goal tension in its first 2–3 weeks of monitoring
2. **Choose starter prompts** — The Journaling UI can tailor its first prompt to the goal category
3. **Set expectation** — The user understands *why* they're journaling (not just "reflect on your day" but "let's explore this tension")

Over time, behavioral data from Journal Entries should supersede the initial goal selection as the primary driver of Weekly Coach focus.

---

## 7. Data Output Schema

### Onboarding Output JSON

```json
{
  "schema_version": 1,
  "user_id": "uuid",
  "session_id": "uuid",
  "onboarding_version": "1.0.0",
  "scoring_method": "exposure_normalized_best_worst_v1",
  "started_at": "2025-01-15T10:28:00Z",
  "timestamp": "2025-01-15T10:30:00Z",
  "bws_responses": [
    {
      "set_number": 1,
      "items": ["security", "self_direction", "achievement", "benevolence"],
      "item_order_shown": ["self_direction", "benevolence", "security", "achievement"],
      "selected_best": "benevolence",
      "selected_worst": "security",
      "response_time_ms": 4200
    }
  ],
  "value_scores": {
    "appearances": {
      "self_direction": 3, "stimulation": 3, "hedonism": 2,
      "achievement": 2, "power": 2, "security": 3,
      "conformity": 2, "tradition": 2, "benevolence": 2,
      "universalism": 3
    },
    "best_counts": {
      "self_direction": 2, "stimulation": 0, "hedonism": 1,
      "achievement": 0, "power": 0, "security": 0,
      "conformity": 0, "tradition": 0, "benevolence": 2,
      "universalism": 1
    },
    "worst_counts": {
      "self_direction": 0, "stimulation": 1, "hedonism": 1,
      "achievement": 0, "power": 2, "security": 1,
      "conformity": 0, "tradition": 1, "benevolence": 0,
      "universalism": 0
    },
    "net_counts": {
      "self_direction": 2,
      "stimulation": -1,
      "hedonism": 0,
      "achievement": 0,
      "power": -2,
      "security": -1,
      "conformity": 0,
      "tradition": -1,
      "benevolence": 2,
      "universalism": 1
    },
    "scores": {
      "self_direction": 0.66666667,
      "stimulation": -0.33333333,
      "hedonism": 0.0,
      "achievement": 0.0,
      "power": -1.0,
      "security": -0.33333333,
      "conformity": 0.0,
      "tradition": -0.5,
      "benevolence": 1.0,
      "universalism": 0.33333333
    },
    "weights": {
      "self_direction": 0.13445378,
      "stimulation": 0.08403361,
      "hedonism": 0.10084034,
      "achievement": 0.10084034,
      "power": 0.05042017,
      "security": 0.08403361,
      "conformity": 0.10084034,
      "tradition": 0.07563025,
      "benevolence": 0.1512605,
      "universalism": 0.11764706
    }
  },
  "confidence": {
    "consistent": true,
    "spread": 0.55,
    "method": "response_consistency_population_spread_v1"
  },
  "top_values": ["benevolence"],
  "goal_category": "work_life_balance",
  "user_confirmed": true,
  "provenance": {
    "source": "react_onboarding_poc",
    "card_order_randomized": true
  }
}
```

### Schema Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Version of the persisted profile schema |
| `user_id` | string (UUID) | Unique user identifier |
| `session_id` | string (UUID) | Identifier for one resumable onboarding session |
| `onboarding_version` | string (semver) | Version of the onboarding flow for A/B testing and data lineage |
| `scoring_method` | string | Versioned scoring contract |
| `started_at` | string (ISO 8601) | When the session began |
| `timestamp` | string (ISO 8601) | When onboarding was completed |
| `bws_responses` | array[6] | Raw response data for each BWS set |
| `bws_responses[].set_number` | int | Set identifier (1–6) |
| `bws_responses[].items` | array[4] | Schwartz values in this set (canonical order) |
| `bws_responses[].item_order_shown` | array[4] | Actual display order (randomized) |
| `bws_responses[].selected_best` | string | Value chosen as "Most like me" |
| `bws_responses[].selected_worst` | string | Value chosen as "Least like me" |
| `bws_responses[].response_time_ms` | int | Time from set display to `Continue` |
| `value_scores.appearances` | object | Number of times each value appeared |
| `value_scores.best_counts` | object | Most selections per value |
| `value_scores.worst_counts` | object | Least selections per value |
| `value_scores.net_counts` | object | Best count minus worst count per value |
| `value_scores.scores` | object | Exposure-normalized BWS scores |
| `value_scores.weights` | object | Normalized weight vector (sums to 1.0) |
| `confidence` | object | Confidence metadata (consistency and score spread) |
| `top_values` | array | Core Values: every value tied for the highest exposure-normalized BWS score; never truncated to a fixed count |
| `goal_category` | string | Selected tension category key |
| `user_confirmed` | boolean | Whether the user confirmed the end summary |
| `provenance` | object | Source and randomized-card-order metadata |

### How This Initializes the VIF User Profile

The onboarding output maps to the existing VIF profile structure:

```
VIF User Profile
├── values: top_values (from BWS)
├── weights: value_scores.weights (from BWS normalization)
├── descriptions: (empty — populated from first Journal Entries)
└── goal: goal_category (from structured selection)
```

The two value fields are intentionally complementary:

- `value_scores.weights` is the full graded 10-dimensional vector used to
  condition the VIF Critic. It replaces a simple "pick 2 values" profile and
  reflects that all Schwartz values matter to some degree.
- `top_values` stores the Core Values used to gate Drift v1. Drift is only
  eligible when two consecutive Conflicts occur on one of these values.

The current synthetic personas already carry an explicit `core_values` list, so
their Core Value gate does not need to be inferred from graded weights.
Persisting BWS `weights` and `top_values` for real users, and threading both
through the VIF runtime, remains pending under `twinkl-1m8`.

---

## 8. Future Considerations

### VIF Integration Points

- **VIF Critic conditioning**: BWS `weights` should feed into the user profile vector `z_u` that the StateEncoder uses (see [VIF System Architecture](../vif/02_system_architecture.md) §1.1). The VIF Critic continues to receive all 10 graded weights.
- **Drift eligibility**: BWS `top_values` should populate the user's Core Values. Drift v1 evaluates consecutive Conflicts only on these values; it must not infer eligibility from an undocumented weight threshold.
- **Synthetic-persona compatibility**: Existing synthetic personas already provide `core_values`. That field remains their Core Value gate until onboarding output is integrated.
- **Implementation status**: Persisting and consuming both onboarding fields is pending under `twinkl-1m8`; this spec defines the contract rather than claiming the runtime wiring exists.
- **VIF Critic cold-start**: Until enough Journal Entries exist for the VIF Critic to produce reliable predictions, the BWS weights serve as the *only* value signal. The Weekly Coach should acknowledge this explicitly ("Based on what you told me during setup...").

### Explicit vs. Behavioral Weighting (60–70% Strategy)

A key open question: how much should Twinkl trust what users *say* they value (BWS) vs. what they *do* (Journal Entries)?

**Proposed approach (not yet implemented):**
- **Weeks 1–2**: 100% explicit (BWS weights only — no behavioral data yet)
- **Weeks 3–6**: Gradual blend (70% explicit, 30% behavioral) as Journal Entry history builds
- **Weeks 7+**: Shift to 40% explicit, 60% behavioral as behavioral signal strengthens
- **Never 0% explicit**: Users' declared values always retain some weight — Twinkl shouldn't completely override what someone says they care about

This weighting schedule requires calibration against real user data and is marked as a future implementation concern. The [value evolution detection](../evolution/01_value_evolution.md) mechanism provides the statistical basis for when behavioral evidence should influence the profile — classifying sustained directional shifts as EVOLUTION triggers profile update suggestions.

### Progressive Profiling

The initial 6-set BWS provides a coarse profile. Future iterations could:
- **Add sets over time**: Introduce additional BWS sets (e.g., 2 more after 1 month) that focus on values with high uncertainty
- **Narrow items**: Use Journal Entry history to generate personalized BWS items that probe specific tensions (e.g., "Working late to hit a deadline" vs. "Leaving on time to cook dinner" for a user showing Achievement-Benevolence tension)

### Re-Assessment Triggers

Twinkl should prompt re-assessment when:
- **Major life event detected**: Journal Entries mention significant changes (new job, relationship change, relocation)
- **Profile divergence**: Behavioral data diverges significantly from BWS-derived weights for >4 weeks. The mechanism for classifying whether this divergence is genuine value evolution or a behavior-value conflict pattern is specified in the [value evolution detection](../evolution/01_value_evolution.md) design.
- **User request**: Users can always re-take the assessment from settings

---

## 9. References

### Academic

- Schwartz, S. H. (1992). Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries. *Advances in Experimental Social Psychology*, 25, 1–65.
- Schwartz, S. H. (2012). An overview of the Schwartz theory of basic values. *Online Readings in Psychology and Culture*, 2(1).
- Schwartz, S. H., et al. (2001). Extending the cross-cultural validity of the theory of basic human values with a different method of measurement. *Journal of Cross-Cultural Psychology*, 32(5), 519–542.
- Louviere, J. J., Flynn, T. N., & Marley, A. A. J. (2015). *Best-Worst Scaling: Theory, Methods and Applications*. Cambridge University Press.
- Paulhus, D. L. (1984). Two-component models of socially desirable responding. *Journal of Personality and Social Psychology*, 46(3), 598–609.
- Alexander, C. S., & Becker, H. J. (1978). The use of vignettes in survey research. *Public Opinion Quarterly*, 42(1), 93–104.

### Internal Documentation

- [PRD](../prd.md) — Product requirements document
- [Schwartz Values Configuration](../../config/schwartz_values.yaml) — Value elaborations used in synthetic data generation
- [VIF Concepts & Roadmap](../vif/01_concepts_and_roadmap.md) — Value Identity Function theory
- [VIF System Architecture](../vif/02_system_architecture.md) — State representation
- [VIF Model Training](../vif/03_model_training.md) — VIF Critic training workflow
- [VIF Worked Example](../vif/example.md) — End-to-end scenario walkthrough
