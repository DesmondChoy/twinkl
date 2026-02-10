# Worked Example: Sarah's Journey Through Twinkl

This example follows a single user through onboarding and four scenarios, showing which system components are active at each stage.

## System Components Reference

| Component | Role | When Active |
|-----------|------|-------------|
| **Generator** | Creates synthetic training data | Offline only (before any user exists) |
| **Judge** | LLM that scores entries against values | Training time; optionally live for novel inputs |
| **Critic** | Fast neural net that predicts value alignment | Every journal entry |
| **Coach** | Conversational AI that surfaces insights | When Critic flags a tension or uncertainty |

---

## Stage 0: Offline Training (Before Sarah Exists)

Before any user signs up, the system must be trained.

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | **ACTIVE** | Creates 200 synthetic journal entries with diverse value tensions |
| Judge | **ACTIVE** | Scores each synthetic entry across all Schwartz value dimensions |
| Critic | **ACTIVE** | Trains on (embedding, Judge scores) pairs via linear probing |
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
| Critic | **ACTIVE** | Scores entry against Sarah's two values |
| Coach | **ACTIVE** (occasional) | Offers evidence-based acknowledgment |

### Critic's Processing

**Input:** SBERT embedding of journal entry + Sarah's profile
**Output:** Scores only for Sarah's declared values

**This Week's Scores:**

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | +0.9 | 0.2 (low) | Aligned — prioritized Emma, fully present |
| Self-Direction | +0.6 | 0.3 (low) | Aligned — creative work with energy and ownership |

**Trajectory Check (4-week rolling averages):**
- Benevolence: +0.8 → stable
- Self-Direction: +0.7 → stable
- No crash (no sudden drop)
- No rut (not below threshold)

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

## Stage 3: Week 7 — Crash Detected

Sarah's situation changes. A major client pitch consumes her.

### Sarah's Journal Entry

> "The pitch is tomorrow. I've been at the office until midnight every night. Emma asked why I missed her recital. I said I'd make it up to her. I always say that."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| Judge | N/A | Critic is confident; pattern matches training data |
| Critic | **ACTIVE** | Detects sharp drop in Benevolence |
| Coach | **ACTIVE** | Surfaces the tension to Sarah |

### Critic's Processing

**This Week's Scores:**

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | -0.9 | 0.25 (low) | Misaligned — missed key moment, acknowledged pattern |
| Self-Direction | +0.2 | 0.4 (low) | Neutral — work is obligation-driven, not creative expression |

**Crash Detection (comparing to rolling average):**
- Benevolence previous average (weeks 5-6): +0.7
- Benevolence current: -0.9
- **Drop: 1.6 points** → exceeds crash threshold (δ = 1.0)
- Uncertainty: Low → confident enough to act

**System Decision:** Trigger Coach.

### Coach's Response

The Coach retrieves relevant context from Sarah's history and speaks:

> "I noticed something in what you wrote. You mentioned Emma's recital — and that you 'always say' you'll make it up. Looking back at your entries, this is the third time this month you've written something similar.
>
> I'm not here to tell you the pitch isn't important. But you told me when we started that being present for Emma is core to who you want to be.
>
> When the pitch is done, what would 'making it up' actually look like this time?"

**What the Coach does NOT say:**
- "You're failing as a parent" (judgment)
- "You should skip the pitch" (advice)
- "Your Achievement score is up!" (Sarah never said she cared about Achievement)

---

## Stage 4: Weeks 9-12 — Rut Detected

The pitch succeeded, but Sarah doesn't bounce back. Her entries become flat:

- Week 9: "Work's fine. Nothing exciting. Emma's at her dad's."
- Week 10: "Same routine. Did some client revisions. Whatever."
- Week 11: "Stayed in all weekend. Watched TV."
- Week 12: "I used to have ideas. Now I just execute tasks."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| Judge | N/A | Critic is confident; low-energy entries are well-represented in training |
| Critic | **ACTIVE** | Tracks sustained low scores over time |
| Coach | **ACTIVE** | Surfaces the pattern after rut threshold is crossed |

### Critic's Processing (Weekly Scores)

| Week | Benevolence | Self-Direction |
|------|-------------|----------------|
| 9 | 0.0 | -0.4 |
| 10 | +0.2 | -0.6 |
| 11 | -0.2 | -0.8 |
| 12 | +0.1 | -0.7 |

**Rut Detection (Self-Direction):**
- Threshold for "low": τ = -0.3
- Weeks below threshold: 4 consecutive
- Minimum for rut: 3 weeks
- Uncertainty: Low across all weeks
- **Rut confirmed**

**About These Thresholds:**

| Parameter | Who Sets It | Rationale |
|-----------|-------------|-----------|
| **τ (low threshold)** | Developer default, potentially user-adjustable | Set per value dimension. Default τ = -0.3 means "mildly misaligned." Users with high self-compassion might prefer τ = -0.5 (only flag severe dips); stricter users might want τ = -0.2. |
| **C_min (minimum weeks)** | Developer default | System-wide. 3 weeks balances sensitivity (catching real ruts) vs. specificity (ignoring normal fluctuations). Shorter risks false alarms; longer delays intervention. |
| **Uncertainty threshold (ε)** | Algorithm (calibrated) | Derived from model calibration on held-out data. Not user-facing — ensures the system only speaks when confident. |

For the full mathematical formulation of crash/rut detection and uncertainty gating, see [Uncertainty & Critique Logic](04_uncertainty_logic.md).

**System Decision:** Trigger Coach.

### Coach's Response

> "I've been reading your entries over the past month. There's a phrase from this week that stood out: 'I used to have ideas. Now I just execute tasks.'
>
> When you first told me about yourself, you said creative freedom is how you feel alive. That voice has been quieter lately — not just this week, but for about a month now.
>
> I'm curious: when was the last time you made something just because you wanted to?"

---

## Stage 5: Week 14 — High Uncertainty (Novel Situation)

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

**System Decision:** Suppress scores. Do not trigger crash/rut logic. Coach responds differently.

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
| Crash Detected | — | — | ✅ Flags drop | ✅ Surfaces tension |
| Rut Detected | — | — | ✅ Flags pattern | ✅ Invites reflection |
| High Uncertainty | — | (optional) | ✅ Admits uncertainty | ✅ Offers presence |

Key insight: The Generator and Judge do their work *before* any user arrives. The Critic handles the real-time evaluation. The Coach speaks when there's something worth saying — including occasional evidence-based acknowledgment when users sustain alignment, but never through gamification or generic praise.
