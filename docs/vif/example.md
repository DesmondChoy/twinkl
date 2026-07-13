# Worked Example: Sarah's Journey Through Twinkl

> **This is a design specification** illustrating Twinkl's target behavior across stages. Not all stages are fully implemented yet:
>
> | Stage | Status |
> |---|---|
> | Stage 0 (Offline Training) | 🧪 Experimental — synthetic generation, LLM-Judge labeling, and VIF Critic training are all implemented; frontier quality is still under active review |
> | Stage 1 (Onboarding) | 📋 Specified |
> | Stages 2–4 (Journaling + Weekly Coach) | ⚠️ Partial — runtime bridge, weekly signals, Weekly Digest outputs, and prompt rendering exist; the selected rolling-soft-evidence Drift Detector and recovery wording remain to be implemented |
> | Stage 5 (High-uncertainty) | ⚠️ Partial — MC Dropout and high-uncertainty routing exist experimentally; sensitive-case behavior should still be treated as target behavior under evaluation |
>
> See the [Implementation Status](../prd.md#implementation-status) table in prd.md for the full breakdown.

This example follows a single user through onboarding and four scenarios, showing which components are active at each stage.

## Component Reference

| Component | Role | When Active |
|-----------|------|-------------|
| **Generator** | Creates synthetic training data | Offline only (before any user exists) |
| **LLM-Judge** | Offline LLM that labels Journal Entries against values | Training time |
| **VIF Critic** | Fast neural net that predicts `-1`, `0`, or `+1` for each Journal Entry and value, plus uncertainty | Every Journal Entry |
| **Drift Detector** | Reads VIF Critic outputs over time and decides whether Drift occurred | After each scored Journal Entry; delivered through the Weekly Digest |
| **Weekly Coach** | Turns the Weekly Digest into an evidence-based reflection | When the Weekly Digest records a tension, uncertainty, or occasional acknowledgment |

---

## Stage 0: Offline Training (Before Sarah Exists)

Before any user signs up, the VIF Critic must be trained.

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | **ACTIVE** | Creates synthetic personas and Journal Entry sequences with diverse value tensions |
| LLM-Judge | **ACTIVE** | Labels each synthetic Journal Entry across all Schwartz value dimensions |
| VIF Critic | **ACTIVE** | Trains on state vectors and LLM-Judge labels |
| Weekly Coach | N/A | No users exist yet |

**Output:** A trained VIF Critic checkpoint ready to score real Journal Entries.

---

## Stage 1: Onboarding — BWS Values Assessment

> For the full onboarding specification, see [Onboarding Spec](../onboarding/onboarding_spec.md).

Sarah downloads Twinkl and completes the BWS-based values assessment. Rather than simply picking her top 2 values, she works through 6 forced-choice screens that reveal her value priorities through trade-offs.

### What Sarah Sees

**BWS Sets (showing 2 of 6):**

> **Set 1:** Security · Self-Direction · Achievement · Benevolence
>
> Sarah taps **"Being there for the people closest to me"** as Most like me (Benevolence) and **"Feeling calm and secure in my life"** as Least like me (Security). She cares about safety, but it's not what *drives* her.

> **Set 3:** Hedonism · Tradition · Self-Direction · Power
>
> Sarah taps **"Having the freedom to choose my own path"** as Most like me (Self-Direction) and **"Having influence over how things go"** as Least like me (Power). Creative freedom over control.

**Mid-flow mirror (after Set 3):**

> "It sounds like you care a lot about **being there for the people closest to you** and **having the freedom to choose your own path**, and less about **having influence over how things go**. Does this feel roughly right?"

Sarah taps **"Yes, that's me"** — the onboarding flow is reading her well so far.

**Goal selection:**

Sarah picks **"I'm stretched too thin between work and everything else"** — the tension that brought her to Twinkl.

**End summary:**

> Your Core Values: **Benevolence**, **Self-Direction**
> Your focus: "I'm stretched too thin between work and everything else"

Sarah confirms. Twinkl now has a graded value profile — not just her top 2, but a full 10-dimensional weight vector showing *how much* each value matters relative to others.

### Component Involvement

| Component | Status | Reason |
|-----------|--------|--------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | No Journal Entry to label yet |
| VIF Critic | N/A | No Journal Entry to score yet |
| Onboarding flow | **ACTIVE** | Guides Sarah through the BWS assessment and stores her profile |
| Weekly Coach | N/A | No Weekly Digest exists yet |

**Output:** Sarah's value profile is saved:

```json
{
  "user_id": "sarah",
  "onboarding_version": "1.0.0",
  "value_scores": {
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
  "top_values": ["Benevolence", "Self-Direction"],
  "goal_category": "work_life_balance",
  "user_confirmed": true,
  "confidence": {
    "consistent": true,
    "spread": 1.35,
    "refinements": 0
  }
}
```

Note the difference from a simple "pick 2" approach: Sarah's profile now captures that Achievement and Universalism have moderate weight, Hedonism and Conformity are middling, and Power is her lowest priority. The VIF Critic uses the full weight vector, not just the two Core Values.

---

## Stage 2: Week 4 — Stable Alignment

Sarah has been journaling for a month. Here is this week's Journal Entry:

### Sarah's Journal Entry

> "Took Friday afternoon off to pick Emma up from school. We made cookies together — flour everywhere, total disaster, but she was so happy. Got back to my design work Saturday morning feeling refreshed."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | Offline labeling is complete |
| VIF Critic | **ACTIVE** | Scores the Journal Entry across all 10 value dimensions |
| Weekly Coach | **ACTIVE** (occasional) | Offers evidence-based acknowledgment |

### VIF Critic Processing

**Input:** Configured sentence embedding of the Journal Entry + Sarah's profile

**Output:** Alignment scores across all 10 Schwartz dimensions (showing Sarah's Core Values below)

**This Week's Scores:**

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | +0.9 | 0.2 (low) | Aligned — prioritized Emma, fully present |
| Self-Direction | +0.6 | 0.3 (low) | Aligned — creative work with energy and ownership |

> _The VIF Critic always outputs all 10 Schwartz dimension scores. Tables in this walkthrough show only Benevolence and Self-Direction — Sarah's Core Values — because only Core Values can produce Drift. The remaining eight dimensions are still scored._

**Trajectory check:**

- no Drift is active for either Core Value;
- uncertainty is low enough to retain the evidence; and
- the Weekly Digest can support occasional, evidence-based acknowledgment.

**Weekly Coach Decision:** Offer occasional reinforcement because Sarah has shown sustained alignment.

### Weekly Coach Response

> "Something I noticed: you wrote about taking Friday off for Emma, and then returning to your design work feeling refreshed. That's the third time this month you've found a way to be present for her without sacrificing your creative energy.
>
> You're living what you told me matters."

**What the Weekly Coach does NOT say:**
- "You're on a 4-week streak!" (gamification)
- "Great job! Keep it up!" (generic praise)
- "You earned 50 alignment points!" (rewards/scores)
- "You're doing better than 80% of users!" (comparison)

The reinforcement is *evidence-based* (cites specific behavior), *tied to the user's own words* (references what Sarah said matters), and *infrequent* (not after every Journal Entry — only when sustained patterns emerge).

---

## Stage 3: Weeks 7–8 — Drift Detected

Sarah's situation changes. A major client pitch consumes her, and two
consecutive Journal Entries are Conflicts for Benevolence.

### Sarah's Journal Entries

> **Week 7:** "The pitch is tomorrow. I've been at the office until midnight
> every night. Emma asked why I missed her recital. I said I'd make it up to
> her. I always say that."

> **Week 8:** "The client wants another revision. I cancelled Sunday with Emma
> and told myself it was just one more time. She didn't argue. That felt worse."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | Stored five-pass consensus labels are diagnostic provenance, not an active Drift benchmark or live runtime input |
| VIF Critic | **ACTIVE** | Produces per-Journal-Entry class evidence and uncertainty |
| Drift Detector | **EXPERIMENTAL, NOT WIRED** | Accumulates rolling soft `P(-1)` evidence on Benevolence; no Drift Detector has deployment approval |
| Weekly Coach | **ACTIVE AT DELIVERY** | Uses the Weekly Digest to surface the repeated Conflict |

### Illustrative Historical Label and Runtime Views

| Journal Entry | Illustrative historical Benevolence label | Runtime target |
|---|---:|---|
| Week 7 | `-1` | elevated `P(-1)`, low enough uncertainty to retain |
| Week 8 | `-1` | repeated `P(-1)` evidence crosses the calibrated persistence gate |

This illustration shows how two consecutive Conflicts for the same Core Value
could form one Drift. It is not an active benchmark target: the
five-pass consensus table is diagnostic provenance only. The runtime does not
require two hard argmax `-1` predictions; it estimates Drift from soft
probability mass under uncertainty gating.

This walkthrough illustrates intended behavior for an explicit Conflict case;
it is not production-validation evidence. The `gpt-5.4-mini` setup with the
current Journal Entry and profile, and the setup that also includes prior
Journal Entries, detect all 10 designed Drifts but miss all five frozen
consensus Drifts. No Drift Detector has deployment approval until that
disagreement is reviewed.

**Weekly Digest content:** Record the Drift and cite both Journal Entries as
evidence.

### Weekly Coach Response

The Weekly Coach reads Sarah's full journal history and speaks:

> "You wrote that missing Emma's recital was something you would 'make up,' and
> then the following week you cancelled Sunday with her for another revision.
> You also told me that being present for her is central to who you want to be.
>
> The pitch can matter and this pattern can still be worth noticing. What feels
> hardest to admit about the trade-off you are making?"

**What the Weekly Coach does NOT say:**

- "You're failing as a parent" (judgment)
- "You should skip the pitch" (advice)
- "Your Benevolence score is down" (score jargon)

---

## Stage 4: Weeks 9–12 — Recovery Changes the Wording

The benchmark Drift remains part of Sarah's history, but later evidence
changes what the Weekly Coach should say:

- Week 9: "I moved the Monday review and took Emma to dinner. We actually talked."
- Week 10: "Left the laptop at the office Friday. Emma picked the movie."
- Week 11: "The client work is still intense, but I stopped treating every request like an emergency."
- Week 12: "I made a full weekend free before anyone asked me to."

### VIF Critic and Delivery Views

| Week | Benevolence reference | Delivery interpretation |
|---|---:|---|
| 7 | `-1` | Conflict evidence begins |
| 8 | `-1` | Drift confirmed |
| 9 | `0` | Conflict is no longer accumulating |
| 10–12 | `+1` | recovery is sustained |

The v1 benchmark still counts the Weeks 7–8 Drift. At Week 12, the Weekly Coach
should not say Sarah is in ongoing Drift. It can acknowledge the
recovery with evidence and remain alert to recurrence.

The intended delivery-time vocabulary is **active**, **recovered**, **mixed**,
or **uncertain**. The exact response-mode schema and transition rules remain
implementation work. See
[Uncertainty and Drift Detector Logic](04_uncertainty_logic.md).

### Weekly Coach Response

> "A few weeks ago, work repeatedly displaced time with Emma. Your more recent
> what you wrote more recently shows a different pattern: dinner together, a laptop-free Friday, and
> a weekend you protected before anyone asked.
>
> What helped you stop treating every request as an emergency?"

---

## Stage 5: Week 14 — Uncertain Delivery (Novel Situation)

Something happens outside the VIF Critic's training distribution.

### Sarah's Journal Entry

> "Mom's diagnosis came back. It's bad. Stage 4. I don't know what to think. Took the week off work. Just sitting here."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | Offline labels do not decide the response to acute grief |
| VIF Critic | **ACTIVE** | Runs inference but uncertainty is high |
| Weekly Coach | **ACTIVE** | Responds with presence, not analysis |

### VIF Critic Processing

**This Week's Scores:**

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | ??? | 0.7 (HIGH) | Predictions scatter from -0.5 to +1.0 |
| Self-Direction | ??? | 0.6 (HIGH) | Predictions scatter from -1.0 to +0.3 |

**Why High Uncertainty?**
- The VIF Critic is trained on synthetic data from the Generator (see [Model Training](03_model_training.md))
- The Generator produces diverse value tensions, but likely doesn't cover acute grief or trauma scenarios — these are difficult to synthesize authentically and ethically fraught to "score"
- Parental terminal illness is therefore out-of-distribution
- MC Dropout reveals the VIF Critic does not know how to score this — predictions scatter widely across forward passes

**Weekly Digest content:** Record high uncertainty and no Conflict claim. The
Weekly Coach responds with presence.

### Weekly Coach Response

> "I'm not going to try to analyze what you wrote — that doesn't feel right for something like this.
>
> I'm just here. If you want to write more, I'll read it. If you want to talk, I'll listen. And if you just want to sit with it, that's okay too."

**What this response deliberately avoids:**
- Scoring grief as "misalignment" with any value
- Suggesting Sarah is neglecting Benevolence or Self-Direction
- Offering productivity tips or reframing

---

## Summary: Component Activity by Stage

| Stage | Generator | LLM-Judge | VIF Critic | Weekly Coach |
|-------|-----------|-------|--------|-------|
| Offline Training | ✅ Creates data | ✅ Labels data | ✅ Trains | — |
| Onboarding | — | — | — | — |
| Stable Alignment | — | — | ✅ Scores | ✅ Occasional acknowledgment |
| Drift | — | Reference only | ✅ Produces soft evidence | ✅ Surfaces repeated Conflict |
| Recovery at Delivery | — | Reference only | ✅ Continues scoring | ✅ Describes recovery rather than ongoing Drift |
| Uncertain Delivery | — | — | ✅ Admits uncertainty | ✅ Offers presence |

Key insight: The Generator and LLM-Judge do their work *before* any user arrives. The VIF Critic handles real-time evaluation. The Weekly Coach speaks when the Weekly Digest contains something worth saying, including occasional evidence-based acknowledgment when users sustain alignment, but never through gamification or generic praise.
