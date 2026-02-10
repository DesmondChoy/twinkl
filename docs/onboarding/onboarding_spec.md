# Onboarding Specification: BWS Values Assessment

## 1. Purpose & Scope

### What Onboarding Achieves

The onboarding flow solves Twinkl's **cold-start problem**: before a user has written any journal entries, the system needs an initial value profile to power the VIF Critic and Coach. Without it, the alignment engine has nothing to align *against*.

The onboarding uses **Best-Worst Scaling (BWS)** — a forced-choice psychometric technique — to elicit a user's value priorities across the 10 Schwartz value dimensions. Combined with a structured goal selection, this produces:

1. A **value weight vector** (`w_u ∈ ℝ^10`) that initializes the user's VIF profile
2. A **primary goal/tension** that focuses the Coach's initial monitoring
3. A **confidence baseline** that signals how much to trust explicit vs. behavioral data

### What's Out of Scope

- **Contextual story / narrative input** — The user's first personal narrative is captured via a guided journal prompt *after* onboarding, not during it. Onboarding is structured choice only.
- **VIF integration mechanics** — How BWS weights feed into the StateEncoder is a future concern. This spec defines the UX and data output; integration is documented in [Section 8: Future Considerations](#8-future-considerations).
- **Adaptive item selection** — All users see the same 6 BWS sets. Computerized adaptive testing (CAT) is a future optimization.

### Cross-References

- [PRD](../prd.md) — Product requirements and implementation status
- [VIF Concepts & Roadmap](../vif/01_concepts_and_roadmap.md) — Value Identity Function theory
- [VIF System Architecture](../vif/02_system_architecture.md) — State representation and inference flow
- [VIF Worked Example](../vif/example.md) — Sarah's journey through the system (includes BWS onboarding walkthrough)

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
- **BWS card format needed**: First-person, single phrase, tappable on mobile

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

### 3.1 Welcome Screen

```
┌─────────────────────────────────┐
│                                 │
│   Build Your Inner Compass      │
│                                 │
│   We'll show you 6 quick        │
│   screens. On each one, pick    │
│   what feels MOST like you      │
│   and LEAST like you.           │
│                                 │
│   There are no right answers    │
│   — just honest ones.           │
│                                 │
│        [ Let's go → ]           │
│                                 │
│   ○ ○ ○ ○ ○ ○ ○ ○  (progress)  │
└─────────────────────────────────┘
```

**Key UX notes:**
- Progress indicator shows 8 total steps (6 BWS sets + mid-flow mirror + goal/summary)
- Framing as "inner compass" connects to the product metaphor
- "No right answers" reduces performance anxiety

### 3.2 BWS Sets 1–3

Each set displays 4 cards. The user taps one as "Most like me" and one as "Least like me." The remaining 2 are implicitly neutral.

```
┌─────────────────────────────────┐
│   Which feels MOST like you?    │
│   Which feels LEAST like you?   │
│                                 │
│   ┌───────────────────────┐     │
│   │ Feeling calm and      │ ◀── tap for Most/Least
│   │ secure in my life     │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ Having the freedom to │     │
│   │ choose my own path    │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ Making progress       │     │
│   │ toward something      │     │
│   │ meaningful            │     │
│   └───────────────────────┘     │
│   ┌───────────────────────┐     │
│   │ Being there for the   │     │
│   │ people closest to me  │     │
│   └───────────────────────┘     │
│                                 │
│        [ Next → ]               │
│   ● ● ● ○ ○ ○ ○ ○  (progress)  │
└─────────────────────────────────┘
```

**Interaction model:**
- First tap on a card marks it as "Most like me" (highlighted green/primary)
- Second tap on a different card marks it as "Least like me" (highlighted red/muted)
- Tapping a selected card deselects it (allows correction)
- "Next" button activates only when both selections are made
- Card order within each set is randomized per user to prevent position bias

### 3.3 Mid-flow Mirror (After Set 3)

After the first 3 sets, the system computes a preliminary profile and reflects it back:

```
┌─────────────────────────────────┐
│                                 │
│   Here's what I'm hearing       │
│   so far...                     │
│                                 │
│   It sounds like you care a     │
│   lot about [top value phrase]  │
│   and [second value phrase],    │
│   and less about                │
│   [bottom value phrase].        │
│                                 │
│   Does this feel roughly right? │
│                                 │
│   [ Yes, that's me ]            │
│   [ Not quite — let me adjust ] │
│                                 │
│   ○ ○ ○ ● ○ ○ ○ ○  (progress)  │
└─────────────────────────────────┘
```

**Purpose:**
- **Validation**: Lets users catch obvious misreadings early
- **Agency**: Users feel heard, not categorized
- **Data quality**: "Not quite" responses are logged and used to adjust final weights (see [Section 5: Scoring Logic](#5-scoring-logic))

**"Not quite" flow:** If the user taps "Not quite," they see a simplified correction screen showing their top 3 and bottom 3 values with the ability to promote/demote one value. This correction is recorded as a `refinement` in the output schema.

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

Single selection. See [Section 6: Goal Categories](#6-goal-categories) for how each goal maps to Coach monitoring priorities.

### 3.6 End Summary + Refinement

```
┌─────────────────────────────────┐
│                                 │
│   Your Inner Compass            │
│                                 │
│   Your top values:              │
│   1. [Top value phrase]         │
│   2. [Second value phrase]      │
│   3. [Third value phrase]       │
│                                 │
│   Your focus:                   │
│   "[Goal display text]"         │
│                                 │
│   This is just a starting       │
│   point — your compass will     │
│   keep learning as you journal. │
│                                 │
│   [ Looks right → Start ]       │
│   [ Let me adjust something ]   │
│                                 │
│   ○ ○ ○ ○ ○ ○ ○ ●  (progress)  │
└─────────────────────────────────┘
```

**"Let me adjust" flow:** Same promote/demote interface as the mid-flow mirror. Refinements are recorded in the output schema with `stage: "end_summary"`.

### 3.7 Transition to First Guided Journal Prompt

After confirming the summary, the user transitions directly to their first guided journal prompt. The prompt is tailored based on:
- Their top value(s) from BWS
- Their selected goal category

Example transition:

```
┌─────────────────────────────────┐
│                                 │
│   Let's start your first        │
│   journal entry.                │
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
│        [ Save entry → ]         │
└─────────────────────────────────┘
```

This is **not** part of the onboarding spec — it belongs to the journaling module. The onboarding spec's responsibility ends at producing the data output described in [Section 7](#7-data-output-schema).

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
raw_score(v) = best_count(v) − worst_count(v)
```

Where:
- `best_count(v)` = number of times the user selected *v* as "Most like me"
- `worst_count(v)` = number of times the user selected *v* as "Least like me"

**Range:** For a value appearing in *n* sets: score ∈ [-n, +n]
- Self-Direction, Stimulation, Security, Universalism (3 appearances): score ∈ [-3, +3]
- Remaining 6 values (2 appearances): score ∈ [-2, +2]

### Normalization to Weight Vector

Raw scores are normalized to produce a weight vector `w_u ∈ ℝ^10` where all weights are non-negative:

```
# Step 1: Shift to non-negative
shifted(v) = raw_score(v) − min(raw_scores) + 1

# Step 2: Normalize to sum to 1
w_u(v) = shifted(v) / Σ shifted(all values)
```

The +1 in the shift ensures no value has zero weight (even the least-preferred value has a small positive weight, reflecting that all Schwartz values are present in all people to some degree).

### Tie Handling

If multiple values have the same raw score:
- They receive the same weight after normalization (no arbitrary tie-breaking)
- If the user's top 2 values are tied, both are shown as "top values" in the summary
- The Coach treats tied values as genuinely co-important

### Confidence Estimation

The system estimates confidence in the BWS-derived profile based on:

1. **Response consistency**: If a user selects a value as "Most" in one set and "Least" in another, this signals low confidence in that value's placement
2. **Score spread**: A flat profile (all scores near 0) suggests the BWS didn't differentiate well — the system should weight behavioral data more heavily once available
3. **Refinement count**: If the user corrected the profile at mid-flow mirror and/or end summary, this indicates the BWS alone didn't capture their self-model — log this for future analysis

```
confidence = {
  "consistent": true/false,     # No contradictions across sets
  "spread": float,              # Std dev of raw scores (higher = more differentiated)
  "refinements": int            # Number of user corrections
}
```

### Applying User Refinements

When a user adjusts the profile at mid-flow mirror or end summary:
1. The promoted value receives a bonus of +1 to its raw score
2. The demoted value receives a penalty of -1 to its raw score
3. Normalization is recomputed
4. The refinement is logged in the output schema for analysis

---

## 6. Goal Categories

### Enumerated Tension Categories

| Category Key | Display Text | Primary Value Tensions to Monitor | Coach Monitoring Priority |
|-------------|-------------|----------------------------------|--------------------------|
| `work_life_balance` | I'm stretched too thin between work and everything else | Achievement vs Benevolence, Achievement vs Hedonism | Work-related entries; time allocation patterns |
| `life_transition` | I'm going through a career or life transition | Security vs Self-Direction, Tradition vs Stimulation | Decision-making entries; uncertainty/ambiguity language |
| `relationships` | I want to be more present for people I care about | Benevolence vs Achievement, Benevolence vs Self-Direction | Mentions of close others; presence/absence patterns |
| `health_wellbeing` | I'm neglecting my health or wellbeing | Hedonism vs Achievement, Security vs Stimulation | Health-related entries; self-care language |
| `direction` | I feel stuck or unclear about my direction | Self-Direction vs Security, Stimulation vs Conformity | Purpose/meaning language; expressions of stagnation |
| `meaningful_work` | I want to make more room for what matters to me | Self-Direction vs Conformity, Universalism vs Power | Fulfillment language; value-action gaps |

### How Goals Map to Coach Behavior

The selected goal does **not** override BWS-derived values. Instead, it:

1. **Focuses initial attention** — The Coach prioritizes entries related to the goal tension in its first 2–3 weeks of monitoring
2. **Chooses starter prompts** — The first guided journal prompt is tailored to the goal category
3. **Sets expectation** — The user understands *why* they're journaling (not just "reflect on your day" but "let's explore this tension")

Over time, behavioral data from journal entries should supersede the initial goal selection as the primary driver of Coach focus.

---

## 7. Data Output Schema

### Onboarding Output JSON

```json
{
  "user_id": "uuid",
  "onboarding_version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "bws_responses": [
    {
      "set_number": 1,
      "items": ["Security", "Self-Direction", "Achievement", "Benevolence"],
      "item_order_shown": ["Self-Direction", "Benevolence", "Security", "Achievement"],
      "selected_best": "Benevolence",
      "selected_worst": "Security",
      "response_time_ms": 4200
    }
  ],
  "value_scores": {
    "raw": {
      "Self-Direction": 2,
      "Stimulation": -1,
      "Hedonism": 0,
      "Achievement": 0,
      "Power": -2,
      "Security": -1,
      "Conformity": 0,
      "Tradition": -1,
      "Benevolence": 2,
      "Universalism": 1
    },
    "weights": {
      "Self-Direction": 0.167,
      "Stimulation": 0.067,
      "Hedonism": 0.100,
      "Achievement": 0.100,
      "Power": 0.033,
      "Security": 0.067,
      "Conformity": 0.100,
      "Tradition": 0.067,
      "Benevolence": 0.167,
      "Universalism": 0.132
    }
  },
  "confidence": {
    "consistent": true,
    "spread": 1.35,
    "refinements": 0
  },
  "top_values": ["Self-Direction", "Benevolence"],
  "goal_category": "work_life_balance",
  "user_confirmed": true,
  "refinements": [],
  "mirror_responses": {
    "mid_flow": {
      "accepted": true,
      "adjustment": null
    },
    "end_summary": {
      "accepted": true,
      "adjustment": null
    }
  }
}
```

### Schema Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string (UUID) | Unique user identifier |
| `onboarding_version` | string (semver) | Version of the onboarding flow for A/B testing and data lineage |
| `timestamp` | string (ISO 8601) | When onboarding was completed |
| `bws_responses` | array[6] | Raw response data for each BWS set |
| `bws_responses[].set_number` | int | Set identifier (1–6) |
| `bws_responses[].items` | array[4] | Schwartz values in this set (canonical order) |
| `bws_responses[].item_order_shown` | array[4] | Actual display order (randomized) |
| `bws_responses[].selected_best` | string | Value chosen as "Most like me" |
| `bws_responses[].selected_worst` | string | Value chosen as "Least like me" |
| `bws_responses[].response_time_ms` | int | Time from set display to "Next" tap |
| `value_scores.raw` | object | Raw BWS scores (best_count − worst_count) per value |
| `value_scores.weights` | object | Normalized weight vector (sums to 1.0) |
| `confidence` | object | Confidence metadata (consistency, spread, refinement count) |
| `top_values` | array | Values with highest weights (may be >2 if tied) |
| `goal_category` | string | Selected tension category key |
| `user_confirmed` | boolean | Whether user accepted the end summary |
| `refinements` | array | User corrections at mirror/summary stages |
| `mirror_responses` | object | User response at mid-flow and end-summary mirrors |

### How This Initializes the VIF User Profile

The onboarding output maps to the existing VIF profile structure:

```
VIF User Profile
├── values: top_values (from BWS)
├── weights: value_scores.weights (from BWS normalization)
├── descriptions: (empty — populated from first journal entries)
└── goal: goal_category (from structured selection)
```

The `weights` vector replaces a simple "pick 2 values" approach (which would produce equal weights like `[0.5, 0.5]` for selected values and zero for everything else). BWS produces a graded 10-dimensional vector that better reflects the reality that all values matter to some degree.

---

## 8. Future Considerations

### VIF Integration Points

- **StateEncoder initialization**: BWS `weights` should feed into the user profile vector `z_u` that the StateEncoder uses (see [VIF System Architecture](../vif/02_system_architecture.md) §1.1). The exact mechanism (direct concatenation, learned mapping, or simple lookup) is TBD.
- **Critic cold-start**: Until enough journal entries exist for the Critic to produce reliable scores, the BWS weights serve as the *only* value signal. The Coach should acknowledge this explicitly ("Based on what you told me during setup...").

### Explicit vs. Behavioral Weighting (60–70% Strategy)

A key open question: how much should the system trust what users *say* they value (BWS) vs. what they *do* (journal entries)?

**Proposed approach (not yet implemented):**
- **Weeks 1–2**: 100% explicit (BWS weights only — no behavioral data yet)
- **Weeks 3–6**: Gradual blend (70% explicit, 30% behavioral) as entry history builds
- **Weeks 7+**: Shift to 40% explicit, 60% behavioral as behavioral signal strengthens
- **Never 0% explicit**: Users' declared values always retain some weight — the system shouldn't completely override what someone says they care about

This weighting schedule requires calibration against real user data and is marked as a future implementation concern.

### Progressive Profiling

The initial 6-set BWS provides a coarse profile. Future iterations could:
- **Add sets over time**: Introduce additional BWS sets (e.g., 2 more after 1 month) that focus on values with high uncertainty
- **Narrow items**: Use entry history to generate personalized BWS items that probe specific tensions (e.g., "Working late to hit a deadline" vs. "Leaving on time to cook dinner" for a user showing Achievement-Benevolence tension)

### Re-Assessment Triggers

The system should prompt re-assessment when:
- **Major life event detected**: Journal entries mention significant changes (new job, relationship change, relocation)
- **Profile drift**: Behavioral data diverges significantly from BWS-derived weights for >4 weeks
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
- [VIF Model Training](../vif/03_model_training.md) — Critic training pipeline
- [VIF Worked Example](../vif/example.md) — End-to-end scenario walkthrough
