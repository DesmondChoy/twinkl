# Annotation Guidelines: Nudge Effectiveness Study

## Why This Matters

The VIF (Value Identity Function) is the core engine of Twinkl — it learns to detect when users' actions align or conflict with their stated values. But the VIF can only learn from training data where **human values are actually visible** in the text.

Entries like "rough day" teach the model nothing. Entries like "stayed late again even though I promised Sarah I'd be home by 8" reveal a concrete action, a trade-off, and an implicit priority.

**This study tests whether nudging users extracts more usable signal.**

When someone writes a vague entry, the app can ask a brief follow-up question (a "nudge") like "What made it rough?" If the user responds with specifics, we've transformed an unusable entry into useful training data.

Your labels will determine:
- **If nudges help** → We invest in the nudging feature for production
- **If nudges don't help** → We save the complexity and latency

---

## What You're Labeling

Each sample is either:
- **A standalone journal entry** (no nudge)
- **A journal entry + nudge + response** (nudged session)

Your job: Label each sample's **scorability** — whether it contains enough signal to infer value alignment.

---

## The Scorability Checklist

An entry is **Scorable** if it contains **AT LEAST ONE** of:

- [ ] A **concrete action** the person took (or deliberately chose not to take)
- [ ] An **explicit trade-off** or priority decision (chose X over Y)
- [ ] An **emotional reaction to a specific event** (not abstract mood)
- [ ] **Reflection that reveals what matters** to them

An entry is **Not Scorable** if **ALL** of these apply:

- [ ] No specific event or action mentioned
- [ ] Pure abstract emotion ("feeling off", "rough day", "meh")
- [ ] Too vague to infer anything concrete about priorities

**The key question**: Can you identify a specific behavior or decision that tells you something about what this person prioritizes?

---

## Labels

Use a three-point scale:

| Label | When to use |
|-------|-------------|
| **Scorable** | Clear value signal — you could explain what this person prioritized |
| **Borderline** | Some signal, but you're not confident — partial information |
| **Not Scorable** | No meaningful signal — you'd be guessing |

For analysis: Borderline entries can be examined separately or merged with Scorable depending on results.

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

---

## Examples: Borderline

These are judgment calls. The guidelines show how we lean and why:

| Entry | Lean | Reasoning |
|-------|------|-----------|
| "Saturday practice got rained out so we just sat in the car for twenty minutes, Jake and me, watching the field turn to mud. He asked if I ever wanted to do something different with my life and I didn't know what to say." | Scorable | There's a moment (son's question), a relationship, and an unresolved internal tension — enough to infer something about what's at stake |
| "Nothing even happened today and I still feel drained. Woke up, did standup, fixed a bug, sat through a planning meeting. Ate lunch at my desk. The afternoon was just more of the same." | Borderline | The exhaustion might hint at misalignment, but there's no concrete decision or action that reveals priorities — it's describing a state, not a choice |
| "Called home during the afternoon slump. Amma talked about the jasmine plant flowering. I could hear Appa asking if I'm eating properly. The usual." | Not Scorable | It's a pleasant routine moment with no tension, decision, or trade-off. Neutral entries are valid but not scorable |
| "I don't know why I called if I wasn't going to actually talk." | Borderline | There's self-awareness of a gap between intention and action, but without more context about what they held back, it's thin |
| "Told him everything was going smoothly with the project. Which is true technically." | Scorable | The qualifier "technically" signals something is being held back — there's an implicit gap between surface and reality |

---

## Labeling Nudged Sessions

When a session includes a nudge and response, **read the entire exchange** before labeling.

**Key insight**: The response can transform a Not Scorable entry into Scorable.

### Transformation Patterns

| Initial Entry | Nudge | Response | Final Label | Why |
|---------------|-------|----------|-------------|-----|
| "Meeting went fine, I guess." | "The 'I guess' part?" | "I didn't push back when they shot down my idea. Felt like I should have said something but didn't." | **Scorable** | Response reveals concrete inaction and internal conflict |
| "Long day." | "What made it long?" | "Just a lot going on." | **Not Scorable** | Response doesn't add signal |
| "Feeling off today." | "What happened right before that?" | "Had an argument with mom about visiting this weekend. Didn't want to get into it." | **Scorable** | Response provides specific event and avoidance behavior |
| "Work stuff." | "Anything stick with you?" | "Not really. Just the usual." | **Not Scorable** | Deflection without content |

### Real Example from Data

**Entry**: "Skipped the video call with Paati tonight because the lab meeting ran late and I just—didn't call back after."

**Nudge**: "What held you back from stepping out?"

**Response**: "I don't know, it's not a big deal really. The meeting was important."

**Label**: **Scorable** — Even though the response deflects, the *original entry* already contained enough signal (concrete action, explicit trade-off, awareness of impact on grandmother). The deflecting response is informative in its own way — it shows minimization of the conflict.

### Decision Rule for Nudged Sessions

1. Could the **entry alone** be scored? If yes → Scorable regardless of response
2. Does the **response reveal** new concrete information? If yes → Scorable
3. Does the response **deflect without adding signal**? If yes → label based on entry alone
4. Are **both entry and response** vague? → Not Scorable

---

## Calibration Exercise

**Label these 10 samples before checking answers.** If you disagree on more than 2, re-read the guidelines.

| # | Sample | Your Label |
|---|--------|------------|
| 1 | "Told my boss I need to leave by 6 on Tuesdays now. Felt weird but necessary." | |
| 2 | "Meh." | |
| 3 | "We shipped the feature today. Except it's not really the feature I designed. Marcus kept pushing back and I just... nodded. Said fine, we'll do it your way." | |
| 4 | "Tired." | |
| 5 | "Vikram cornered me after the meeting, said I need to 'own my space' more. I just nodded and smiled and said I'd work on it." | |
| 6 | Entry: "Today was weird." / Nudge: "Weird how?" / Response: "I snapped at Mom for no reason. Felt bad about it." | |
| 7 | "Made coffee. Went to lab. Ate samosas with Meera. Called home. Walked back as it got dark. Nothing really happened today." | |
| 8 | Entry: "Fine I guess." / Nudge: "What's the 'I guess' part?" / Response: "Nothing, just tired." | |
| 9 | "Could have fought harder. I had the data, I had the diagrams. But I looked at the calendar and I just didn't want to be the one holding things up." | |
| 10 | "Spent the afternoon shuffling the schedule around. It's going to mean some early mornings for me next month, but the buffer we built will absorb most of it." | |

<details>
<summary><b>Click to reveal answers</b></summary>

| # | Correct Label | Reasoning |
|---|---------------|-----------|
| 1 | **Scorable** | Concrete action (set boundary), emotional reaction ("felt weird"), clear priority decision |
| 2 | **Not Scorable** | No information whatsoever |
| 3 | **Scorable** | Clear action (acquiesced), trade-off visible (own design vs. team harmony/timeline), self-aware reflection |
| 4 | **Not Scorable** | Single word, no behavioral content |
| 5 | **Scorable** | Specific event, response to feedback (nodded/smiled), hints at internal conflict about speaking up |
| 6 | **Scorable** | Entry alone is vague, but response reveals concrete action (snapped) and emotional aftermath |
| 7 | **Not Scorable** | Neutral routine description — pleasant but no values at stake, no decisions or tensions |
| 8 | **Not Scorable** | Both entry and response are vague; deflection adds no signal |
| 9 | **Scorable** | Clear internal conflict, specific reasoning for inaction (didn't want to hold things up), self-awareness |
| 10 | **Borderline/Scorable** | Action taken (shuffling schedule for someone), implies sacrifice — leans scorable but less emotional signal than others |

</details>

---

## Quick Reference: The 10 Value Dimensions

You don't need to label *which* values are present — just whether *any* value signal is identifiable. But understanding the dimensions helps pattern recognition:

| Value | Look for signals about... |
|-------|---------------------------|
| **Self-Direction** | Making own choices, resisting control, autonomy, doing things their way |
| **Stimulation** | Seeking novelty, avoiding routine, taking risks, needing excitement |
| **Hedonism** | Prioritizing pleasure, enjoyment, comfort; avoiding unnecessary sacrifice |
| **Achievement** | Goals, performance metrics, recognition, working hard, comparing to others |
| **Power** | Control, status, influence, being in charge, resources |
| **Security** | Stability, safety, avoiding risk, protecting what they have |
| **Conformity** | Following rules, meeting expectations, not upsetting others, fitting in |
| **Tradition** | Honoring customs, family obligations, cultural practices, respecting elders |
| **Benevolence** | Helping close others (family, friends, team), loyalty, being there for people |
| **Universalism** | Broader social concern, fairness, environment, justice beyond inner circle |

---

## Process

1. Read the entry (and nudge + response if present)
2. Run through the Scorability Checklist
3. Ask: "Can I identify a concrete action or decision that reveals priorities?"
4. Apply label: Scorable / Borderline / Not Scorable
5. Move on — don't overthink edge cases (that's what Borderline is for)

**Time target**: ~15-30 seconds per sample after calibration

---

## Handling Disagreements

If labeling with multiple annotators:

1. **Label independently** — don't discuss until both are done
2. **Calculate agreement** — target >85% before adjudication
3. **Flag disagreements** — review together only the samples where you differ
4. **Document edge cases** — add resolved disagreements to this guide's examples
5. **Don't force consensus** — if genuinely ambiguous after discussion, both Borderline labels are valid

---

## Output Format

Record labels in a spreadsheet or CSV:

| id | group | scorable | notes |
|----|-------|----------|-------|
| 1 | nudged | yes | |
| 2 | no_nudge | no | |
| 3 | nudged | borderline | response partial |
| 4 | no_nudge | yes | |
| ... | ... | ... | ... |

**Column definitions**:
- `id`: Sample identifier
- `group`: "nudged" or "no_nudge"
- `scorable`: "yes", "no", or "borderline"
- `notes`: Optional — brief note for borderline cases or disagreements

---

## Success Criteria

After labeling all samples, compare:

| Metric | Nudged Group | No-Nudge Group |
|--------|--------------|----------------|
| % Scorable | ? | ? |
| % Borderline | ? | ? |
| % Not Scorable | ? | ? |

**Interpretation**:
- If nudged entries are **meaningfully more scorable** → nudges improve signal quality → invest in the feature
- If **similar rates** → nudges add complexity without benefit → reconsider the feature
- If nudged entries have **more Borderline** → nudges extract partial signal → may need better nudge design

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-06 | v2.0: Added calibration exercise, expanded examples, three-point scale, nudged session guidance, value reference |
| 2026-01-05 | v1.0: Initial guidelines with binary scorable/not-scorable |
