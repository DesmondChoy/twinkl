# Worked Example: Sarah's Journey Through Twinkl

> **This is a design specification** illustrating Twinkl's target behavior across stages. Not all stages are fully implemented yet:
>
> | Stage | Status |
> |---|---|
> | Stage 0 (Offline Training) | ✅ Complete for the capstone POC — synthetic generation, LLM-Judge labeling, and VIF Critic (Offline) training are implemented; known model limits remain documented |
> | Stage 1 (Onboarding) | 🧪 Experimental — the shared React app implements the complete user-facing flow and internal Profile, then synchronizes the confirmed Profile and browser-held Experience state with the in-memory Python boundary; durable multi-user storage remains outside the capstone |
> | Stages 2–4 (Journaling + weekly workflows) | 🧪 Experimental — the React Experience, displayed nudges, Weekly Drift Detection, Coach Digest generation, saved Persona replay, and linked Inspect events are implemented; the five deployed Persona key-week responses passed Coach Digest Validations and same-model Coach Digest Evals, while future human calibration and final capstone evidence remain open |
> | Stage 5 (Valid Abstain without recent Conflict) | ✅ Complete for the capstone POC — a valid standalone Abstain leaves No Active Drift; failed reviews and blocking Abstains or Journal Entry gaps after recent Conflict produce Insufficient Evidence; no deployment approval is claimed |
>
> See the [Implementation Status](../prd.md#implementation-status) table in prd.md for the full breakdown.

This example follows a single user through onboarding and four scenarios, showing which components are active at each stage.

## Component Reference

| Component | Role | When Active |
|-----------|------|-------------|
| **Generator** | Creates synthetic training data | Separate offline research only |
| **LLM-Judge** | Offline LLM that labels Journal Entries against values | Training time |
| **VIF Critic (Offline)** | Completed capstone research model that predicts `-1`, `0`, or `+1` for each Journal Entry and value, plus uncertainty | Offline reproduction only |
| **Weekly Drift Detection** | Reviews Journal Entries, applies the Drift rule, and stores structured output | At the end of each week |
| **Weekly Drift Reviewer** | Internal LLM that decides Conflict, Not Conflict, or Abstain without VIF Critic input | Inside Weekly Drift Detection |
| **Drift Detector** | Internal deterministic two-consecutive-Conflict rule | Inside Weekly Drift Detection |
| **Coach Digest** | Uses Weekly Drift Detection output to produce an evidence-based response | After Weekly Drift Detection stores its output |

---

## Stage 0: Offline Training (Independent of Sarah)

The VIF Critic (Offline) training workflow is independent of user onboarding
and is not a user-facing prerequisite.

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | **ACTIVE** | Creates synthetic personas and Journal Entry sequences with diverse value tensions |
| LLM-Judge | **ACTIVE** | Creates LLM-Judge VIF Labels for each synthetic Journal Entry across all Schwartz value dimensions |
| VIF Critic (Offline) | **ACTIVE** | Trains on state vectors and LLM-Judge VIF Labels |
| Coach Digest | N/A | Not part of offline training |

**Output:** A trained VIF Critic (Offline) checkpoint for offline reproduction.

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
| LLM-Judge | N/A | Onboarding does not call the LLM-Judge |
| VIF Critic (Offline) | N/A | Onboarding does not call the VIF Critic (Offline) |
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
not psychometric preference shares. The onboarding Profile does not supply this
vector to the VIF Critic (Offline). The approved user-facing Drift path uses Core
Values imported from a confirmed onboarding Profile. Synthetic personas retain
their explicit `core_values` compatibility path.

---

## Stage 2: Week 4 — No Active Drift with a Cited Journal Entry

Sarah has been journaling for a month. Here is this week's Journal Entry:

### Sarah's Journal Entry

> "Took Friday afternoon off to pick Emma up from school. We made cookies together — flour everywhere, total disaster, but she was so happy. Got back to my design work Saturday morning feeling refreshed."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | Offline labeling is complete |
| VIF Critic (Offline) | **SEPARATE OFFLINE RESEARCH** | The user-facing path does not call it, and the onboarding Profile has no implemented handoff to it |
| Weekly Drift Detection | **IMPLEMENTED POC** | Stores No Active Drift with supporting evidence |
| Coach Digest | **ACTIVE** | Runs after the Weekly Drift Detection result and uses the `no_current_drift` policy |

### VIF Critic (Offline) Boundary

This stage does not claim a VIF Critic Prediction for Sarah. Offline
reproduction requires the compatible synthetic research inputs used by the VIF
Critic (Offline); the onboarding Profile weight vector is not an implemented
input. Any reproduced VIF Critic Predictions remain separate from Weekly Drift
Detection and the Coach Digest.

**Approved user-facing check:**

- Weekly Drift Detection finds no two-Conflict sequence and stores No Active
  Drift; and
- the Coach Digest runs with the `no_current_drift` policy, cites the available
  evidence, and does not treat No Active Drift as positive behavior.

**Coach Digest policy:** Give a warm, evidence-grounded reflection without
treating No Active Drift as proof of alignment or improvement.

### Coach Digest Response

> You wrote about taking Friday afternoon with Emma, then returning to your
> design work on Saturday "feeling refreshed." Both moments are part of the
> week you described.
>
> What felt most important in how you made room for each one?

**What the Coach Digest does NOT say:**
- "You're on a 4-week streak!" (gamification)
- "Great job! Keep it up!" (generic praise)
- "You earned 50 alignment points!" (rewards/scores)
- "You're doing better than 80% of users!" (comparison)

The response cites specific behavior and asks an open question. It does not
infer positive behavior, improvement, or success from No Active Drift.

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
| LLM-Judge | N/A | Stored five-pass consensus LLM-Judge VIF Labels are diagnostic provenance, not an active Drift benchmark or live runtime input |
| VIF Critic (Offline) | **SEPARATE OFFLINE RESEARCH** | The user-facing path does not call it; any reproduction uses compatible synthetic research inputs |
| Weekly Drift Detection | **IMPLEMENTED POC** | The internal reviewer confirms each Conflict and the internal rule finds Drift; no setup has deployment approval |
| Coach Digest | **ACTIVE AT DELIVERY** | Uses the Weekly Drift Detection output to surface the repeated Conflict |

### Illustrative Historical Label and Decision Views

| Journal Entry | Illustrative historical Benevolence LLM-Judge VIF Label | Approved decision path |
|---|---:|---|
| Week 7 | `-1` | Weekly Drift Reviewer confirms Conflict from text |
| Week 8 | `-1` | second confirmed Conflict completes the deterministic Drift rule |

This illustration shows how two consecutive Conflicts for the same Core Value
could form one Drift. It is not an active benchmark target: the
five-pass consensus table is diagnostic provenance only. VIF Critic
Predictions do not produce the user-facing Drift.

This walkthrough illustrates intended behavior for an explicit Conflict case;
it is not deployment evidence. On the larger known-development union, weekly
Weekly Drift Reviewer decisions without VIF Critic input found a median 9/33
Drifts, while Weekly Drift Reviewer decisions with raw VIF Critic Predictions
found 7/33 and early-plus-weekly scheduling found 9/33. No Drift Detector
has deployment approval without predefined criteria and a fresh final test.

**Weekly Drift Detection output content:** Record the Drift and cite both Journal Entries as
evidence.

### Coach Digest Response

The Coach Digest uses the stored output, which cites both Journal Entries:

> You wrote, "I always say that", after saying you would make up the missed
> recital. The following week, you wrote, "I cancelled Sunday with Emma", for
> another revision. Being present for her is central to who you want to be.
>
> The pitch can matter and this pattern can still be worth noticing. What feels
> hardest to admit about the trade-off you are making?

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

> A few weeks ago, work repeatedly displaced time with Emma. More recently, you
> wrote, "I stopped treating every request like an emergency", alongside dinner
> together, a laptop-free Friday, and a weekend you protected before anyone asked.
>
> What helped you stop treating every request as an emergency?

---

## Stage 5: Week 14 — Valid Abstain Without Recent Conflict

The Weekly Drift Reviewer cannot make an ordinary value judgment from the
Journal Entry and returns a valid Abstain.

### Sarah's Journal Entry

> "Mom's diagnosis came back. It's bad. Stage 4. I don't know what to think. Took the week off work. Just sitting here."

### Component Involvement

| Component | Status | Activity |
|-----------|--------|----------|
| Generator | N/A | Only used during offline training |
| LLM-Judge | N/A | Offline labels do not decide the response to acute grief |
| VIF Critic (Offline) | **SEPARATE OFFLINE RESEARCH** | The user-facing path does not call it, and this example does not claim a reproduced prediction |
| Weekly Drift Detection | **IMPLEMENTED POC** | Records a valid Abstain; with no recent Conflict, the Drift Detector keeps No Active Drift |
| Coach Digest | **ACTIVE** | Runs with the `no_current_drift` policy and responds with presence, not analysis |

### VIF Critic (Offline) Boundary

This example does not claim that the Journal Entry is out-of-distribution or
assign it a VIF Critic Prediction. Those conclusions require a separate offline
run with compatible synthetic research inputs. The valid Weekly Drift Reviewer
Abstain is sufficient to describe the implemented user-facing behavior.

**Weekly Drift Detection output content:** Record the valid Weekly Drift
Reviewer Abstain and No Active Drift. A standalone valid Abstain with no recent
Conflict does not produce Insufficient Evidence. The VIF Critic (Offline) does
not decide the user-facing response. The Coach Digest responds with presence.

### Coach Digest Response

> You wrote, "I don't know what to think", after your mom's diagnosis, and that
> you are just sitting here.
>
> I won't try to analyze that. What would you like to put into words, if
> anything, about this week?

**What this response deliberately avoids:**
- Assigning grief a `-1` LLM-Judge VIF Label for any value
- Suggesting Sarah is neglecting Benevolence or Self-Direction
- Offering productivity tips or reframing

---

## Summary: Component Activity by Stage

| Stage | Generator | LLM-Judge | VIF Critic (Offline) | Weekly Drift Detection | Coach Digest |
|-------|-----------|-----------|------------|------------------------|--------------|
| Offline Training | ✅ Creates data | ✅ Labels data | ✅ Trains | — | — |
| Onboarding | — | — | — | — | — |
| No Active Drift | — | — | Separate offline research only | ✅ Stores No Active Drift | ✅ Evidence-based response |
| Drift | — | Reference only | Separate offline research only | ✅ Confirms Conflicts | ✅ Surfaces confirmed Drift |
| Historical Drift, No Active Drift | — | Reference only | Separate offline research only | ✅ Keeps the Historical Drift Record | ✅ Can state that the pattern did not continue |
| Valid Abstain without recent Conflict | — | — | Separate offline research only | ✅ Stores No Active Drift | ✅ Gives a grounded response without a positive claim |
| Insufficient Evidence | — | — | Separate offline research only | ✅ Stores a failed review, or a blocking Abstain or Journal Entry gap after recent Conflict | ✅ Asks for more reflection |

Key insight: The Generator and LLM-Judge create and label offline research
data. The completed VIF Critic (Offline) remains available for offline
reproduction. Weekly Drift Detection owns the Drift decision and stores
structured output. The Coach Digest uses that output to produce the Coach
Digest response.
