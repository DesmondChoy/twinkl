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

## Stage 1: Onboarding — Sarah Picks Her Values

Sarah downloads Twinkl and completes the values screening.

### What Sarah Sees

> **Welcome to Twinkl.**
>
> We'd like to understand what matters most to you. Below are descriptions of different values. Pick the **two** that feel most central to who you want to be.

Sarah reads through the options (derived from Schwartz's Values) and selects:

| Value | Why Sarah Chose It |
|-------|-------------------|
| **Benevolence** | "Being present for my daughter Emma is everything. I don't want to be the parent who's always distracted." |
| **Self-Direction** | "I need creative freedom in my work. When I can't create, I feel like I'm dying inside." |

She skips Achievement, Security, Power, etc. — not because they don't matter, but because they're not *core* to her identity right now.

### Component Involvement

| Component | Status | Reason |
|-----------|--------|--------|
| Generator | N/A | Only used during offline training |
| Judge | N/A | No journal entry to score yet |
| Critic | N/A | No journal entry to score yet |
| Coach | **ACTIVE** | Guides Sarah through value selection; stores her profile |

**Output:** Sarah's value profile is saved:

```json
{
  "user": "sarah",
  "values": ["Benevolence", "Self-Direction"],
  "weights": [0.5, 0.5],
  "descriptions": {
    "Benevolence": "Being present for Emma",
    "Self-Direction": "Creative freedom in work"
  }
}
```

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
| Coach | N/A | No tension detected; nothing to surface |

### Critic's Processing

**Input:** SBERT embedding of journal entry + Sarah's profile
**Output:** Scores only for Sarah's declared values

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | +1.8 | 0.2 (low) | Strong alignment — prioritized Emma, fully present |
| Self-Direction | +1.1 | 0.3 (low) | Alignment — creative work with energy and ownership |

**Trajectory Check:**
- Benevolence: 4-week average = +1.5 → stable
- Self-Direction: 4-week average = +1.2 → stable
- No crash (no sudden drop)
- No rut (not below threshold)

**System Decision:** Silent. Sarah is living according to her values.

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

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | -1.7 | 0.25 (low) | Strong misalignment — missed key moment, acknowledged pattern |
| Self-Direction | +0.3 | 0.4 (low) | Neutral — work is obligation-driven, not creative expression |

**Crash Detection:**
- Benevolence previous average (weeks 5-6): +1.3
- Benevolence current: -1.7
- **Drop: 3.0 points** → exceeds crash threshold (δ = 1.5)
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
| 9 | 0.0 | -0.8 |
| 10 | +0.2 | -1.2 |
| 11 | -0.3 | -1.5 |
| 12 | +0.1 | -1.4 |

**Rut Detection (Self-Direction):**
- Threshold for "low": τ = -0.5
- Weeks below threshold: 4 consecutive
- Minimum for rut: 3 weeks
- Uncertainty: Low across all weeks
- **Rut confirmed**

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

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | ??? | 1.4 (HIGH) | Predictions scatter from -1 to +2 |
| Self-Direction | ??? | 1.1 (HIGH) | Predictions scatter from -2 to +1 |

**Why High Uncertainty?**
- The Critic was trained on work-life tensions, daily routines, small wins and setbacks
- Parental terminal illness is out-of-distribution
- MC Dropout reveals the model doesn't know how to score this

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
| Stable Alignment | — | — | ✅ Scores | — |
| Crash Detected | — | — | ✅ Flags drop | ✅ Surfaces tension |
| Rut Detected | — | — | ✅ Flags pattern | ✅ Invites reflection |
| High Uncertainty | — | (optional) | ✅ Admits uncertainty | ✅ Offers presence |

Key insight: The Generator and Judge do their work *before* any user arrives. The Critic handles the real-time evaluation. The Coach only speaks when there's something worth saying.
