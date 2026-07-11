# Worked Example: Sarah's Journey Through Twinkl

> **This is a design specification** illustrating Twinkl's target behavior across stages. Not all stages are fully implemented yet:
>
> | Stage | Status |
> |---|---|
> | Stage 0 (Offline Training) | 🧪 Experimental — synthetic generation, judge labeling, and critic training are all implemented; frontier quality is still under active review |
> | Stage 1 (Onboarding) | 📋 Specified |
> | Stages 2–4 (Journaling + Coach) | ⚠️ Partial — runtime bridge, weekly signals, digest artifacts, and prompt rendering exist; the selected rolling-soft-evidence sustained-conflict detector and recovery wording remain to be implemented |
> | Stage 5 (High-uncertainty) | ⚠️ Partial — MC Dropout and high-uncertainty routing exist experimentally; sensitive-case behavior should still be treated as target behavior under evaluation |
>
> See the [Implementation Status](../prd.md#implementation-status) table in prd.md for the full breakdown.

This example follows a single user through onboarding and four scenarios, showing which system components are active at each stage.

## System Components Reference

| Component | Role | When Active |
|-----------|------|-------------|
| **Generator** | Creates synthetic training data | Offline only (before any user exists) |
| **Judge** | LLM that scores entries against values | Training time; optionally live for novel inputs |
| **Critic** | Fast neural net that predicts value alignment | Every journal entry |
| **Drift detector** | Reads Critic outputs over time and decides whether a pattern meets the trigger contract | After each scored entry; delivered through the weekly digest |
| **Coach** | Conversational AI that turns structured signals into evidence-based reflection | When the weekly artifact calls for a tension, uncertainty, or occasional acknowledgment |

---

## Stage 0: Offline Training (Before Sarah Exists)

Before any user signs up, the system must be trained.

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | **ACTIVE** | Creates synthetic personas and journal trajectories with diverse value tensions |
| Judge | **ACTIVE** | Scores each synthetic entry across all Schwartz value dimensions |
| Critic | **ACTIVE** | Trains on state vectors and Judge labels as a supervised student |
| Coach | N/A | No users exist yet |

**Output:** A trained Critic model ready to score real user entries.

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

Sarah taps **"Yes, that's me"** — the system is reading her well so far.

**Goal selection:**

Sarah picks **"I'm stretched too thin between work and everything else"** — the tension that brought her to Twinkl.

**End summary:**

> Your top values: **Benevolence**, **Self-Direction**
> Your focus: "I'm stretched too thin between work and everything else"

Sarah confirms. The system now has a graded value profile — not just her top 2, but a full 10-dimensional weight vector showing *how much* each value matters relative to others.

### Component Involvement

| Component | Status | Reason |
|-----------|--------|--------|
| Generator | N/A | Only used during offline training |
| Judge | N/A | No journal entry to score yet |
| Critic | N/A | No journal entry to score yet |
| Coach | **ACTIVE** | Guides Sarah through BWS assessment; stores her profile |

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

Note the difference from a simple "pick 2" approach: Sarah's profile now captures that Achievement and Universalism have moderate weight, Hedonism and Conformity are middling, and Power is her lowest priority. The Critic uses the full weight vector, not just the top 2.

---

## Stage 2: Week 4 — Stable Alignment

Sarah has been journaling for a month. Here's this week's entry:

### Sarah's Journal Entry

> "Took Friday afternoon off to pick Emma up from school. We made cookies together — flour everywhere, total disaster, but she was so happy. Got back to my design work Saturday morning feeling refreshed."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| Judge | N/A | Critic is confident; no need to escalate |
| Critic | **ACTIVE** | Scores entry across all 10 value dimensions |
| Coach | **ACTIVE** (occasional) | Offers evidence-based acknowledgment |

### Critic's Processing

**Input:** Configured sentence embedding of the journal entry + Sarah's profile
**Output:** Alignment scores across all 10 Schwartz dimensions (showing Sarah's declared values below)

**This Week's Scores:**

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | +0.9 | 0.2 (low) | Aligned — prioritized Emma, fully present |
| Self-Direction | +0.6 | 0.3 (low) | Aligned — creative work with energy and ownership |

> _The Critic always outputs all 10 Schwartz dimension scores. Tables in this walkthrough show only Benevolence and Self-Direction — Sarah's declared values — because v1 drift detection is gated to declared core values. The remaining eight dimensions are still scored._

**Trajectory check:**

- no two-entry sustained-conflict episode is active on either declared value;
- uncertainty is low enough to retain the evidence; and
- the weekly artifact can support occasional, evidence-based acknowledgment.

**System Decision:** Occasional reinforcement. Sarah has shown sustained alignment — worth acknowledging.

### Coach's Response

> "Something I noticed: you wrote about taking Friday off for Emma, and then returning to your design work feeling refreshed. That's the third time this month you've found a way to be present for her without sacrificing your creative energy.
>
> You're living what you told me matters."

**What the Coach does NOT say:**
- "You're on a 4-week streak!" (gamification)
- "Great job! Keep it up!" (generic praise)
- "You earned 50 alignment points!" (rewards/scores)
- "You're doing better than 80% of users!" (comparison)

The reinforcement is *evidence-based* (cites specific behavior), *tied to the user's own words* (references what Sarah said matters), and *infrequent* (not every entry — only when sustained patterns emerge).

---

## Stage 3: Weeks 7–8 — Sustained Conflict Detected

Sarah's situation changes. A major client pitch consumes her, and two
consecutive journal sessions conflict with Benevolence.

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
| Judge | N/A | Stored five-pass consensus labels are diagnostic provenance, not an active drift benchmark or live runtime input |
| Critic | **ACTIVE** | Produces per-entry class evidence and uncertainty |
| Drift detector | **EXPERIMENTAL, NOT WIRED** | Accumulates rolling soft `P(-1)` evidence on Benevolence; no scorer is promotion-ready |
| Coach | **ACTIVE AT DELIVERY** | Uses the weekly artifact to surface the repeated conflict |

### Illustrative Historical Label and Runtime Views

| Entry | Illustrative historical Benevolence label | Runtime target |
|---|---:|---|
| Week 7 | `-1` | elevated `P(-1)`, low enough uncertainty to retain |
| Week 8 | `-1` | repeated `P(-1)` evidence crosses the calibrated persistence gate |

This illustration shows how two adjacent value-conflicting entries could form
one sustained-conflict episode. It is not an active benchmark target: the
five-pass consensus table is diagnostic provenance only. The runtime does not
require two hard argmax `-1` predictions; it estimates the episode from soft
probability mass under uncertainty gating.

This walkthrough illustrates intended behavior for an explicit conflict case;
it is not production-validation evidence. Both LLM arms detect the similarly
explicit locked designed cases, but those same arms miss all five frozen
consensus episodes. No scorer is approved until that disagreement is reviewed.

**System decision:** Include the conflict in the weekly digest with both entries
as evidence.

### Coach's Response

The Coach reads Sarah's full journal history and speaks:

> "You wrote that missing Emma's recital was something you would 'make up,' and
> then the following week you cancelled Sunday with her for another revision.
> You also told me that being present for her is central to who you want to be.
>
> The pitch can matter and this pattern can still be worth noticing. What feels
> hardest to admit about the trade-off you are making?"

**What the Coach does NOT say:**

- "You're failing as a parent" (judgment)
- "You should skip the pitch" (advice)
- "Your Benevolence score is down" (score jargon)

---

## Stage 4: Weeks 9–12 — Recovery Changes the Wording

The benchmark episode remains part of Sarah's history, but later evidence
changes what the weekly Coach should say:

- Week 9: "I moved the Monday review and took Emma to dinner. We actually talked."
- Week 10: "Left the laptop at the office Friday. Emma picked the movie."
- Week 11: "The client work is still intense, but I stopped treating every request like an emergency."
- Week 12: "I made a full weekend free before anyone asked me to."

### Critic and Delivery Views

| Week | Benevolence reference | Delivery interpretation |
|---|---:|---|
| 7 | `-1` | conflict evidence begins |
| 8 | `-1` | sustained-conflict episode confirmed |
| 9 | `0` | conflict is no longer accumulating |
| 10–12 | `+1` | recovery is sustained |

The v1 benchmark still counts the Weeks 7–8 episode as a true event. At Week 12,
the Coach should not say Sarah is in ongoing drift. It can acknowledge the
recovery with evidence and remain alert to recurrence.

The intended delivery-time vocabulary is **active**, **recovered**, **mixed**,
or **uncertain**. The exact response-mode schema and transition rules remain
implementation work. See
[Uncertainty, Drift, and Trigger Logic](04_uncertainty_logic.md).

### Coach's Response

> "A few weeks ago, work repeatedly displaced time with Emma. Your more recent
> entries show a different pattern: dinner together, a laptop-free Friday, and
> a weekend you protected before anyone asked.
>
> What helped you stop treating every request as an emergency?"

---

## Stage 5: Week 14 — Uncertain Delivery (Novel Situation)

Something happens outside the Critic's training distribution.

### Sarah's Journal Entry

> "Mom's diagnosis came back. It's bad. Stage 4. I don't know what to think. Took the week off work. Just sitting here."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| Judge | **OPTIONAL** | Could be called to help, but even Judge may struggle with grief |
| Critic | **ACTIVE** | Runs inference but uncertainty is high |
| Coach | **ACTIVE** | Responds with presence, not analysis |

### Critic's Processing

**This Week's Scores:**

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | ??? | 0.7 (HIGH) | Predictions scatter from -0.5 to +1.0 |
| Self-Direction | ??? | 0.6 (HIGH) | Predictions scatter from -1.0 to +0.3 |

**Why High Uncertainty?**
- The Critic is trained on synthetic data from the Generator (see [Model Training](03_model_training.md))
- The Generator produces diverse value tensions, but likely doesn't cover acute grief or trauma scenarios — these are difficult to synthesize authentically and ethically fraught to "score"
- Parental terminal illness is therefore out-of-distribution
- MC Dropout reveals the model doesn't know how to score this — predictions scatter widely across forward passes

**System Decision:** Suppress confident value-level interpretation. Do not add
the entry to sustained-conflict evidence. The Coach responds with presence.

### Coach's Response

> "I'm not going to try to analyze what you wrote — that doesn't feel right for something like this.
>
> I'm just here. If you want to write more, I'll read it. If you want to talk, I'll listen. And if you just want to sit with it, that's okay too."

**What the system deliberately avoids:**
- Scoring grief as "misalignment" with any value
- Suggesting Sarah is neglecting Benevolence or Self-Direction
- Offering productivity tips or reframing

---

## Summary: Component Activity by Stage

| Stage | Generator | Judge | Critic | Coach |
|-------|-----------|-------|--------|-------|
| Offline Training | ✅ Creates data | ✅ Labels data | ✅ Trains | — |
| Onboarding | — | — | — | ✅ Captures values |
| Stable Alignment | — | — | ✅ Scores | ✅ Occasional acknowledgment |
| Sustained Conflict | — | Reference only | ✅ Produces soft evidence | ✅ Surfaces repeated conflict |
| Recovery at Delivery | — | Reference only | ✅ Continues scoring | ✅ Describes recovery rather than ongoing drift |
| Uncertain Delivery | — | (optional) | ✅ Admits uncertainty | ✅ Offers presence |

Key insight: The Generator and Judge do their work *before* any user arrives. The Critic handles the real-time evaluation. The Coach speaks when there's something worth saying — including occasional evidence-based acknowledgment when users sustain alignment, but never through gamification or generic praise.
