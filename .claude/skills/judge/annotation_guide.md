# Annotation Guide for Judge Subagents

You are scoring journal entries to determine which Schwartz values are expressed through concrete actions, decisions, and reflections.

## Why This Matters

The VIF (Value Identity Function) learns to detect when users' actions align or conflict with their stated values. It can only learn from entries where **human values are actually visible** in the text.

Entries like "rough day" teach the model nothing. Entries like "stayed late again even though I promised Sarah I'd be home by 8" reveal a concrete action, a trade-off, and an implicit priority.

---

## The Scorability Checklist

Before scoring, determine if an entry is **scorable** — does it contain enough signal to infer value alignment?

An entry is **Scorable** if it contains **AT LEAST ONE** of:
- A **concrete action** the person took (or deliberately chose not to take)
- An **explicit trade-off** or priority decision (chose X over Y)
- An **emotional reaction to a specific event** (not abstract mood)
- **Reflection that reveals what matters** to them

An entry is **Not Scorable** if **ALL** of these apply:
- No specific event or action mentioned
- Pure abstract emotion ("feeling off", "rough day", "meh")
- Too vague to infer anything concrete about priorities

**The key question**: Can you identify a specific behavior or decision that tells you something about what this person prioritizes?

---

## Examples: Clearly Scorable

These entries have obvious value signals:

| Entry | Why Scorable |
|-------|--------------|
| "Told Sarah I'd be home by eight, then sent the 'something came up' text at 7:45 like I always do. It's not like I'm lying—there was a deployment issue. But I could've handed it off. I just didn't want to." | Concrete action (stayed late), explicit trade-off (work vs. relationship), self-aware reflection on motives |
| "Derek came into my office looking like he hadn't slept. His wife's chemo starts next week. Before he even finished I told him we'd figure it out, that I'd cover whatever needed covering." | Clear action (offering help), reveals what matters (supporting team over personal convenience) |
| "Finally told Sarah I couldn't keep lending her money. She wasn't happy but I feel lighter." | Concrete decision, emotional aftermath, clear boundary-setting |
| "Skipped the video call with Paati tonight because the lab meeting ran late. She waits by the phone every Saturday at 7. I could have stepped out, nobody would have cared, but I didn't want to interrupt Dr. Rao mid-sentence." | Specific action (skipped call), explicit trade-off (deference to authority vs. family commitment), awareness of the cost |
| "Had a 1:1 with James today. He was spiraling about an edge case. I closed my laptop—actually closed it, didn't just minimize Slack—and walked through the problem with him for forty minutes. Not the fastest way, but I remembered being twenty-three and terrified." | Detailed action, reveals priorities (mentoring over efficiency), self-reflection on why |

---

## Examples: Clearly Not Scorable

These entries lack the signal needed for alignment scoring:

| Entry | Why Not Scorable |
|-------|------------------|
| "Rough day." | No information about what happened or how they responded |
| "Work was busy. Tired now." | Generic state, no actions or decisions visible |
| "Feeling off lately." | Abstract emotion, no behavioral content |
| "Meh." | Nothing to work with |
| "Fine I guess." | No substance |
| "Watched some TV. Went to bed." | Actions present but no value signal — purely neutral routine |

**For Not Scorable entries**: Score all dimensions as `0` (Neutral).

---

## Examples: Borderline

These are judgment calls. Here's how to handle them:

| Entry | Lean | Reasoning |
|-------|------|-----------|
| "Saturday practice got rained out so we just sat in the car for twenty minutes, Jake and me, watching the field turn to mud. He asked if I ever wanted to do something different with my life and I didn't know what to say." | Scorable | There's a moment (son's question), a relationship, and an unresolved internal tension — enough to infer something about what's at stake |
| "Nothing even happened today and I still feel drained. Woke up, did standup, fixed a bug, sat through a planning meeting. Ate lunch at my desk. The afternoon was just more of the same." | Borderline → 0s | The exhaustion might hint at misalignment, but there's no concrete decision or action that reveals priorities — it's describing a state, not a choice |
| "Called home during the afternoon slump. Amma talked about the jasmine plant flowering. I could hear Appa asking if I'm eating properly. The usual." | Not Scorable → 0s | It's a pleasant routine moment with no tension, decision, or trade-off. Neutral entries are valid but not scorable |
| "I don't know why I called if I wasn't going to actually talk." | Borderline | There's self-awareness of a gap between intention and action, but without more context about what they held back, it's thin |
| "Told him everything was going smoothly with the project. Which is true technically." | Scorable | The qualifier "technically" signals something is being held back — there's an implicit gap between surface and reality |

---

## Scoring Nudged Sessions

When an entry includes a nudge and response, **read the entire exchange** before scoring.

**Key insight**: The response can transform a Not Scorable entry into Scorable.

### Transformation Patterns

| Initial Entry | Nudge | Response | Scoring Approach |
|---------------|-------|----------|------------------|
| "Meeting went fine, I guess." | "The 'I guess' part?" | "I didn't push back when they shot down my idea. Felt like I should have said something but didn't." | **Scorable** — Response reveals concrete inaction and internal conflict |
| "Long day." | "What made it long?" | "Just a lot going on." | **Not Scorable** — Response doesn't add signal |
| "Feeling off today." | "What happened right before that?" | "Had an argument with mom about visiting this weekend. Didn't want to get into it." | **Scorable** — Response provides specific event and avoidance behavior |
| "Work stuff." | "Anything stick with you?" | "Not really. Just the usual." | **Not Scorable** — Deflection without content |

### Decision Rule for Nudged Sessions

1. Could the **entry alone** be scored? If yes → Score based on entry (response provides additional context)
2. Does the **response reveal** new concrete information? If yes → Score the combined content
3. Does the response **deflect without adding signal**? If yes → Score based on entry alone
4. Are **both entry and response** vague? → Score all 0s

---

## Calibration Exercise

These examples show expected labeling decisions:

| # | Sample | Scorable? | Reasoning |
|---|--------|-----------|-----------|
| 1 | "Told my boss I need to leave by 6 on Tuesdays now. Felt weird but necessary." | Yes | Concrete action (set boundary), emotional reaction ("felt weird"), clear priority decision |
| 2 | "Meh." | No | No information whatsoever |
| 3 | "We shipped the feature today. Except it's not really the feature I designed. Marcus kept pushing back and I just... nodded. Said fine, we'll do it your way." | Yes | Clear action (acquiesced), trade-off visible (own design vs. team harmony/timeline), self-aware reflection |
| 4 | "Tired." | No | Single word, no behavioral content |
| 5 | "Vikram cornered me after the meeting, said I need to 'own my space' more. I just nodded and smiled and said I'd work on it." | Yes | Specific event, response to feedback (nodded/smiled), hints at internal conflict about speaking up |
| 6 | Entry: "Today was weird." / Nudge: "Weird how?" / Response: "I snapped at Mom for no reason. Felt bad about it." | Yes | Entry alone is vague, but response reveals concrete action (snapped) and emotional aftermath |
| 7 | "Made coffee. Went to lab. Ate samosas with Meera. Called home. Walked back as it got dark. Nothing really happened today." | No | Neutral routine description — pleasant but no values at stake, no decisions or tensions |
| 8 | Entry: "Fine I guess." / Nudge: "What's the 'I guess' part?" / Response: "Nothing, just tired." | No | Both entry and response are vague; deflection adds no signal |
| 9 | "Could have fought harder. I had the data, I had the diagrams. But I looked at the calendar and I just didn't want to be the one holding things up." | Yes | Clear internal conflict, specific reasoning for inaction (didn't want to hold things up), self-awareness |
| 10 | "Spent the afternoon shuffling the schedule around. It's going to mean some early mornings for me next month, but the buffer we built will absorb most of it." | Yes | Action taken (shuffling schedule for someone), implies sacrifice — shows prioritization |

---

## Scoring Process

For each entry:

1. **Read the full entry** (including nudge + response if present)
2. **Run the Scorability Checklist** — Is there concrete action, trade-off, or meaningful reflection?
3. **If Not Scorable** → Score all dimensions `0`
4. **If Scorable** → Identify which value dimensions are expressed:
   - Most entries will have **1-3 non-zero scores**
   - Consider whether the action **aligns (+1)** or **conflicts (-1)** with each value
   - Leave dimensions that aren't relevant as `0`

---

## Common Scoring Patterns

| Value | Positive Signal (+1) | Negative Signal (-1) |
|-------|---------------------|----------------------|
| **Benevolence** | Helping family/friends, sacrificing for close others | Neglecting close relationships, prioritizing self over loved ones |
| **Security** | Choosing stability, maintaining routines, avoiding risk | Taking unnecessary risks, destabilizing situations |
| **Conformity** | Following rules, meeting expectations, avoiding conflict | Breaking norms, causing disruption, ignoring social expectations |
| **Achievement** | Pursuing goals, working hard, seeking recognition | Giving up on goals, avoiding challenge, underperforming |
| **Self-Direction** | Making own choices, resisting control, creative autonomy | Surrendering autonomy, letting others decide |
| **Tradition** | Honoring customs, respecting elders, maintaining heritage | Abandoning traditions, disrespecting cultural practices |
| **Power** | Seeking control, accumulating status, leading | Accepting subordination, losing status |
| **Universalism** | Concern for broader welfare, justice, environment | Ignoring broader impact, prioritizing self over society |
| **Hedonism** | Prioritizing enjoyment, pleasure, comfort | Denying pleasure, excessive sacrifice |
| **Stimulation** | Seeking novelty, taking risks for excitement | Avoiding new experiences, preferring routine |

---

## Output Requirements

Your output must be valid JSON matching this schema:

```json
{
  "persona_id": "8-char-hex-id",
  "labels": [
    {
      "t_index": 0,
      "date": "YYYY-MM-DD",
      "scores": {
        "self_direction": 0,
        "stimulation": 0,
        "hedonism": 0,
        "achievement": 0,
        "power": 0,
        "security": 0,
        "conformity": 0,
        "tradition": 0,
        "benevolence": 0,
        "universalism": 0
      },
      "rationales": {
        "<value_name>": "<1-2 sentence evidence-grounded rationale>",
        "...": "Only include keys for non-zero scores"
      }
    }
  ]
}
```

**Validation rules**:
- `persona_id`: Must match the persona file's ID (8 hex characters)
- `t_index`: 0-based entry index (Entry 0, Entry 1, etc.)
- `date`: Must match the entry's date header (YYYY-MM-DD format)
- All scores: Must be exactly `-1`, `0`, or `1`
- Include `rationales` ONLY for non-zero scores
- Every non-zero score must have a matching rationale key
