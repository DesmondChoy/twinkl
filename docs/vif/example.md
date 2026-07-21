# Worked Example: Sarah's Journey Through Twinkl

> **This is a design specification** illustrating Twinkl's target behavior across stages. Not all stages are fully implemented yet:
>
> | Stage | Status |
> |---|---|
> | Stage 0 (Offline Training) | ✅ Complete for the capstone POC — synthetic generation, LLM-Judge labeling, and VIF Critic training are implemented; known model limits remain documented |
> | Stage 1 (Onboarding) | 🧪 Experimental — the standalone React POC implements the complete local, user-facing flow and internal Profile; automatic browser-to-service storage remains outside the capstone |
> | Stages 2–4 (Journaling + Weekly Coach) | ⚠️ Partial — the Weekly Drift Reviewer and deterministic Drift Detector runtime, onboarding Profile import, Weekly Digest, and prompt rendering are implemented; the Journaling UI and product-facing orchestration remain incomplete |
> | Stage 5 (Uncertain delivery) | ✅ Complete for the capstone POC — Weekly Drift Reviewer Abstain fails closed, and the Weekly Digest handles uncertain delivery; no deployment approval is claimed |
>
> See the [Implementation Status](../prd.md#implementation-status) table in prd.md for the full breakdown.

This example follows a single user through onboarding and four scenarios, showing which components are active at each stage.

## Component Reference

| Component | Role | When Active |
|-----------|------|-------------|
| **Generator** | Creates synthetic training data | Offline only (before any user exists) |
| **LLM-Judge** | Offline LLM that labels Journal Entries against values | Training time |
| **VIF Critic** | Completed capstone research model that predicts `-1`, `0`, or `+1` for each Journal Entry and value, plus uncertainty | Offline reproduction only |
| **Weekly Drift Reviewer** | Fixed `gpt-5.6-luna` reasoning-effort-`low` LLM that decides Conflict, Not Conflict, or Abstain from Journal Entry text without VIF Critic input | Weekly review in the approved user-facing path |
| **Drift Detector** | Applies the deterministic two-consecutive-Conflict rule | After Weekly Drift Reviewer Decisions; delivered through the Weekly Digest |
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

Sarah downloads Twinkl and completes the research-grounded SVBWS onboarding
assessment. Rather than picking two labels, she works through 11 randomized
groups from the published balanced design. Each group contains six descriptor
cards and requires one Most and one Least choice.

### What Sarah Sees

**BWS groups (showing 2 of 11):**

> **One group:** Successful, capable, ambitious · Protecting the environment,
> a world of beauty, unity with nature · Helpful, honest, forgiving · Devout,
> accepting portion in life, humble · Clean, national & family security,
> social order · Equality, world at peace, social justice
>
> Sarah selects **"Helpful, honest, forgiving"** as Most important and
> **"Devout, accepting portion in life, humble"** as Least important.

> **Another group:** Social power, authority, wealth · Successful, capable,
> ambitious · Pleasure, enjoying life, self-indulgent · Creativity, curious,
> freedom · Protecting the environment, a world of beauty, unity with nature ·
> Devout, accepting portion in life, humble
>
> Sarah selects **"Creativity, curious, freedom"** as Most important and
> **"Social power, authority, wealth"** as Least important.

Twinkl shows no preliminary result between groups.

**Goal selection:**

Sarah picks **"I'm stretched too thin between work and everything else"** — the tension that brought her to Twinkl.

**End summary:**

> What sits at the center: **Being there for the people closest to me** and
> **Having the freedom to choose my own path**
> Your focus: "I'm stretched too thin between work and everything else"

The summary never reveals the Schwartz labels. Sarah selects `Set my compass`,
which confirms the displayed descriptions as her Core Values. Twinkl retains
the raw 11-object BWS result and a separately named ten-value product
transformation.

### Component Involvement

| Component | Status | Reason |
|-----------|--------|--------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | No Journal Entry to label yet |
| VIF Critic | N/A | No Journal Entry to score yet |
| Onboarding flow | **ACTIVE** | Guides Sarah through the BWS assessment and creates her local Profile |
| Weekly Coach | N/A | No Weekly Digest exists yet |

**Abridged internal output:** Sarah's Profile is generated in the browser and
is not exposed as technical output to Sarah. A host can persist it, and the
approved runtime can import the saved JSON:

```json
{
  "schema_version": 2,
  "user_id": "sarah",
  "session_id": "example-session",
  "onboarding_version": "2.1.0",
  "instrument": "svbws_lee_soutar_louviere_2008_ui_adaptation_v2",
  "scoring_method": "best_minus_worst_divided_by_appearances_v1",
  "bws_results": {
    "scores": {
      "universalism_nature": 0.167,
      "universalism_social": 0.0
    }
  },
  "value_profile": {
    "method": "mean_universalism_facets_then_shift_normalize_v1",
    "weights": {
      "self_direction": 0.167,
      "stimulation": 0.067,
      "hedonism": 0.100,
      "achievement": 0.100,
      "power": 0.033,
      "security": 0.067,
      "conformity": 0.100,
      "tradition": 0.067,
      "benevolence": 0.167,
      "universalism": 0.132
    },
    "top_values": ["self_direction", "benevolence"]
  },
  "top_values": ["self_direction", "benevolence"],
  "goal_category": "work_life_balance",
  "user_confirmed": true
}
```

The weights preserve the order of the ten-value scores but are product features,
not psychometric preference shares. The full vector remains available for
offline VIF Critic analysis; the approved user-facing Drift path uses Core
Values imported from a confirmed onboarding Profile. Synthetic personas retain
their explicit `core_values` compatibility path.

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
| VIF Critic | **OFFLINE REPRODUCTION** | Can reproduce saved research behavior; it does not affect the user-facing path |
| Weekly Drift Reviewer | **IMPLEMENTED POC** | Finds no Conflict in the Journal Entry from its text |
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

**Approved user-facing check:**

- the Weekly Drift Reviewer finds no Conflict for either Core Value;
- the deterministic Drift Detector finds no two-Conflict sequence; and
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
| VIF Critic | **OFFLINE REPRODUCTION** | Can reproduce VIF Critic Predictions for historical analysis |
| Weekly Drift Reviewer | **IMPLEMENTED POC** | Confirms each Benevolence Conflict from Journal Entry text without VIF Critic input |
| Drift Detector | **IMPLEMENTED POC** | Applies the deterministic two-consecutive-Conflict rule; no setup has deployment approval |
| Weekly Coach | **ACTIVE AT DELIVERY** | Uses the Weekly Digest to surface the repeated Conflict |

### Illustrative Historical Label and Decision Views

| Journal Entry | Illustrative historical Benevolence label | Approved decision path |
|---|---:|---|
| Week 7 | `-1` | Weekly Drift Reviewer confirms Conflict from text |
| Week 8 | `-1` | second confirmed Conflict completes the deterministic Drift rule |

This illustration shows how two consecutive Conflicts for the same Core Value
could form one Drift. It is not an active benchmark target: the
five-pass consensus table is diagnostic provenance only. Offline VIF Critic
Predictions do not produce the user-facing Drift.

This walkthrough illustrates intended behavior for an explicit Conflict case;
it is not deployment evidence. On the larger known-development union, weekly
review without VIF Critic input found a median 9/33 Drifts, while raw VIF Critic
input found 7/33 and early-plus-weekly scheduling found 9/33. No Drift Detector
has deployment approval without predefined criteria and a fresh final test.

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

### Review and Delivery Views

| Week | Weekly Drift Reviewer decision | Delivery interpretation |
|---|---:|---|
| 7 | Conflict | Conflict evidence begins |
| 8 | Conflict | Drift confirmed |
| 9 | non-Conflict | Conflict is no longer accumulating |
| 10–12 | non-Conflict | recovery is sustained |

The v1 benchmark still counts the Weeks 7–8 Drift. At Week 12, the Weekly Coach
should not say Sarah is in ongoing Drift. It can acknowledge the
recovery with evidence and remain alert to recurrence.

The implemented delivery-time vocabulary is **active**, **recovered**,
**mixed**, or **uncertain**. See
[Uncertainty and Drift Review Logic](04_uncertainty_logic.md).

### Weekly Coach Response

> "A few weeks ago, work repeatedly displaced time with Emma. What you wrote
> more recently shows a different pattern: dinner together, a laptop-free Friday, and
> a weekend you protected before anyone asked.
>
> What helped you stop treating every request as an emergency?"

---

## Stage 5: Week 14 — Uncertain Delivery (Novel Situation)

Something happens outside the VIF Critic's training distribution and does not
support an ordinary value judgment from text.

### Sarah's Journal Entry

> "Mom's diagnosis came back. It's bad. Stage 4. I don't know what to think. Took the week off work. Just sitting here."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | Offline labels do not decide the response to acute grief |
| VIF Critic | **OFFLINE REPRODUCTION** | Can reproduce a high-uncertainty prediction for historical analysis |
| Weekly Drift Reviewer | **IMPLEMENTED POC** | Abstains because the text does not support a responsible Conflict decision |
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

**Weekly Digest content:** Record the Weekly Drift Reviewer abstention and no
Conflict claim. Offline VIF Critic uncertainty remains diagnostic and does not
decide the user-facing response. The Weekly Coach responds with presence.

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

| Stage | Generator | LLM-Judge | VIF Critic | Weekly Drift Reviewer | Weekly Coach |
|-------|-----------|-----------|------------|-----------------------|--------------|
| Offline Training | ✅ Creates data | ✅ Labels data | ✅ Trains | — | — |
| Onboarding | — | — | — | — | — |
| Stable Alignment | — | — | ✅ Stores offline evidence | ✅ Finds no Conflict | ✅ Occasional acknowledgment |
| Drift | — | Reference only | ✅ Stores offline evidence | ✅ Confirms Conflicts | ✅ Surfaces confirmed Drift |
| Recovery at Delivery | — | Reference only | ✅ Continues offline scoring | ✅ Finds later non-Conflict | ✅ Describes recovery rather than ongoing Drift |
| Uncertain Delivery | — | — | ✅ Stores uncertainty | ✅ Abstains | ✅ Offers presence |

Key insight: The Generator and LLM-Judge create and label training data before
any user arrives. The completed VIF Critic remains available for offline
reproduction. The Weekly Drift Reviewer and deterministic Drift Detector own
the user-facing Drift decision; the Weekly Coach speaks from the resulting
Weekly Digest.
