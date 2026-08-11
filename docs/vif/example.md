# Worked Example: Sarah's Journey Through Twinkl

> **This is a design specification** illustrating Twinkl's target behavior across stages. Not all stages are fully implemented yet:
>
> | Stage | Status |
> |---|---|
> | Stage 0 (Offline Training) | ✅ Complete for the capstone POC — synthetic generation, LLM-Judge labeling, and VIF Critic training are implemented; known model limits remain documented |
> | Stage 1 (Onboarding) | 🧪 Experimental — the standalone React POC implements the complete local, user-facing flow and internal Profile; automatic browser-to-service storage remains outside the capstone |
> | Stages 2–4 (Journaling + weekly workflows) | ✅ Complete for the core capstone POC — the React Experience, displayed nudges, Weekly Drift Detection, Coach Digest generation, saved Persona replay, and linked Inspect events are implemented; current Coach Digest evaluation results and final capstone evidence remain open |
> | Stage 5 (Insufficient Evidence) | ✅ Complete for the capstone POC — Weekly Drift Detection fails closed to Abstain and can store Insufficient Evidence; no deployment approval is claimed |
>
> See the [Implementation Status](../prd.md#implementation-status) table in prd.md for the full breakdown.

This example follows a single user through onboarding and four scenarios, showing which components are active at each stage.

## Component Reference

| Component | Role | When Active |
|-----------|------|-------------|
| **Generator** | Creates synthetic training data | Offline only (before any user exists) |
| **LLM-Judge** | Offline LLM that labels Journal Entries against values | Training time |
| **VIF Critic** | Completed capstone research model that predicts `-1`, `0`, or `+1` for each Journal Entry and value, plus uncertainty | Offline reproduction only |
| **Weekly Drift Detection** | Reviews Journal Entries, applies the Drift rule, and stores structured output | At the end of each week |
| **Weekly Drift Reviewer** | Internal LLM that decides Conflict, Not Conflict, or Abstain without VIF Critic input | Inside Weekly Drift Detection |
| **Drift Detector** | Internal deterministic two-consecutive-Conflict rule | Inside Weekly Drift Detection |
| **Coach Digest** | Uses Weekly Drift Detection output to produce an evidence-based response | After Weekly Drift Detection stores its output |

---

## Stage 0: Offline Training (Before Sarah Exists)

Before any user signs up, the VIF Critic must be trained.

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | **ACTIVE** | Creates synthetic personas and Journal Entry sequences with diverse value tensions |
| LLM-Judge | **ACTIVE** | Labels each synthetic Journal Entry across all Schwartz value dimensions |
| VIF Critic | **ACTIVE** | Trains on state vectors and LLM-Judge labels |
| Coach Digest | N/A | No users exist yet |

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

Twinkl shows no preliminary result between groups. After the 11th group, it
opens the Core Value summary directly.

**End summary:**

> What sits at the center: **Being there for the people closest to me** and
> **Having the freedom to choose my own path**

The summary never reveals the Schwartz labels. Sarah selects `Confirm my compass`,
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
| Coach Digest | N/A | No Weekly Drift Detection output exists yet |

**Abridged internal output:** Sarah's Profile is generated in the browser and
is not exposed as technical output to Sarah. A host can persist it, and the
approved runtime can import the saved JSON:

```json
{
  "schema_version": 4,
  "user_id": "sarah",
  "session_id": "example-session",
  "onboarding_version": "2.3.0",
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
  "user_confirmed": true
}
```

The weights preserve the order of the ten-value scores but are product features,
not psychometric preference shares. The full vector remains available for
offline VIF Critic analysis; the approved user-facing Drift path uses Core
Values imported from a confirmed onboarding Profile. Synthetic personas retain
their explicit `core_values` compatibility path.

---

## Stage 2: Week 4 — No Active Drift with Aligned Evidence

Sarah has been journaling for a month. Here is this week's Journal Entry:

### Sarah's Journal Entry

> "Took Friday afternoon off to pick Emma up from school. We made cookies together — flour everywhere, total disaster, but she was so happy. Got back to my design work Saturday morning feeling refreshed."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | Offline labeling is complete |
| VIF Critic | **OFFLINE REPRODUCTION** | Can reproduce saved research behavior; it does not affect the user-facing path |
| Weekly Drift Detection | **IMPLEMENTED POC** | Stores No Active Drift with supporting evidence |
| Coach Digest | **ACTIVE** (occasional) | Offers evidence-based acknowledgment |

### VIF Critic Processing

**Input:** Configured sentence embedding of the Journal Entry + Sarah's Profile

**Output:** Alignment scores across all 10 Schwartz dimensions (showing Sarah's Core Values below)

**This Week's Scores:**

| Value | Score | Uncertainty (σ) | Interpretation |
|-------|-------|-----------------|----------------|
| Benevolence | +0.9 | 0.2 (low) | Aligned — prioritized Emma, fully present |
| Self-Direction | +0.6 | 0.3 (low) | Aligned — creative work with energy and ownership |

> _The VIF Critic always outputs all 10 Schwartz dimension scores. Tables in this walkthrough show only Benevolence and Self-Direction — Sarah's Core Values — because only Core Values can produce Drift. The remaining eight dimensions are still scored._

**Approved user-facing check:**

- Weekly Drift Detection finds no two-Conflict sequence; and
- the Coach Digest can provide occasional, evidence-based acknowledgment.

**Coach Digest Decision:** Offer occasional reinforcement because Sarah has shown sustained alignment.

### Coach Digest Response

> "Something I noticed: you wrote about taking Friday off for Emma, and then returning to your design work feeling refreshed. That's the third time this month you've found a way to be present for her without sacrificing your creative energy.
>
> You're living what you told me matters."

**What the Coach Digest does NOT say:**
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
| Weekly Drift Detection | **IMPLEMENTED POC** | The internal reviewer confirms each Conflict and the internal rule finds Drift; no setup has deployment approval |
| Coach Digest | **ACTIVE AT DELIVERY** | Uses the Weekly Drift Detection output to surface the repeated Conflict |

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

**Weekly Drift Detection output content:** Record the Drift and cite both Journal Entries as
evidence.

### Coach Digest Response

The Coach Digest uses the stored output, which cites both Journal Entries:

> "You wrote that missing Emma's recital was something you would 'make up,' and
> then the following week you cancelled Sunday with her for another revision.
> You also told me that being present for her is central to who you want to be.
>
> The pitch can matter and this pattern can still be worth noticing. What feels
> hardest to admit about the trade-off you are making?"

**What the Coach Digest does NOT say:**

- "You're failing as a parent" (judgment)
- "You should skip the pitch" (advice)
- "Your Benevolence score is down" (score jargon)

---

## Stage 4: Weeks 9–12 — Historical Drift and No Active Drift

The benchmark Drift remains part of Sarah's history, but later evidence
changes what the Coach Digest should say:

- Week 9: "I moved the Monday review and took Emma to dinner. We actually talked."
- Week 10: "Left the laptop at the office Friday. Emma picked the movie."
- Week 11: "The client work is still intense, but I stopped treating every request like an emergency."
- Week 12: "I made a full weekend free before anyone asked me to."

### Review and Delivery Views

| Week | Weekly Drift Reviewer decision | Delivery interpretation |
|---|---:|---|
| 7 | Conflict | Conflict evidence begins |
| 8 | Conflict | Drift confirmed |
| 9 | Not Conflict | The earlier Conflict run ends |
| 10–12 | Not Conflict | No new Conflict run forms |

The v1 benchmark still counts the Weeks 7–8 Drift. At Week 12, Weekly Drift
Detection stores No Active Drift and keeps the earlier pair in a Historical
Drift Record. The Coach Digest can state that the earlier pattern did not
continue because the Not Conflict decisions support that change. It cannot call
this proof of improvement.

The implemented current states are Active Drift, No Active Drift, and
Insufficient Evidence. See [Uncertainty and Drift Review
Logic](04_uncertainty_logic.md).

### Coach Digest Response

> "A few weeks ago, work repeatedly displaced time with Emma. What you wrote
> more recently shows a different pattern: dinner together, a laptop-free Friday, and
> a weekend you protected before anyone asked.
>
> What helped you stop treating every request as an emergency?"

---

## Stage 5: Week 14 — Insufficient Evidence (Novel Situation)

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
| Weekly Drift Detection | **IMPLEMENTED POC** | Records Abstain because the text does not support a responsible Conflict decision |
| Coach Digest | **ACTIVE** | Responds with presence, not analysis |

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

**Weekly Drift Detection output content:** Record the Weekly Drift Reviewer abstention and no
Conflict claim. Offline VIF Critic uncertainty remains diagnostic and does not
decide the user-facing response. The Coach Digest responds with presence.

### Coach Digest Response

> "I'm not going to try to analyze what you wrote — that doesn't feel right for something like this.
>
> I'm just here. If you want to write more, I'll read it. If you want to talk, I'll listen. And if you just want to sit with it, that's okay too."

**What this response deliberately avoids:**
- Scoring grief as "misalignment" with any value
- Suggesting Sarah is neglecting Benevolence or Self-Direction
- Offering productivity tips or reframing

---

## Summary: Component Activity by Stage

| Stage | Generator | LLM-Judge | VIF Critic | Weekly Drift Detection | Coach Digest |
|-------|-----------|-----------|------------|------------------------|--------------|
| Offline Training | ✅ Creates data | ✅ Labels data | ✅ Trains | — | — |
| Onboarding | — | — | — | — | — |
| No Active Drift | — | — | ✅ Stores offline evidence | ✅ Stores No Active Drift | ✅ Evidence-based response |
| Drift | — | Reference only | ✅ Stores offline evidence | ✅ Confirms Conflicts | ✅ Surfaces confirmed Drift |
| Historical Drift, No Active Drift | — | Reference only | ✅ Continues offline scoring | ✅ Keeps the Historical Drift Record | ✅ Can state that the pattern did not continue |
| Insufficient Evidence | — | — | ✅ Stores uncertainty | ✅ Abstains | ✅ Asks for more reflection |

Key insight: The Generator and LLM-Judge create and label training data before
any user arrives. The completed VIF Critic remains available for offline
reproduction. Weekly Drift Detection owns the Drift decision and stores
structured output. The Coach Digest uses that output to produce the user
response.
