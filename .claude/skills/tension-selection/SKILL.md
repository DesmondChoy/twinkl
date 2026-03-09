---
name: tension-selection
description: Value-specific tension scenarios for synthetic data generation. Provides scenario banks that replace the generic Unsettled prompt when TARGET_TENSIONS is active.
---

# Tension-Selection Skill

## Purpose

This skill addresses label imbalance or boundary weakness for specific
Schwartz value dimensions (e.g., Universalism -1 at 0.6% vs 5.4%+ for all
other values, or later batches that need more nuanced Hedonism/Security
conflict cases). The root cause: the generic Unsettled prompt in
`journal_entry.yaml` produces personal dilemmas that the judge often scores
as neutral (0) rather than the sharper or more theory-relevant boundary the
batch is trying to repair.

For some values, a single bank is enough because the negative pole is
relatively easy to surface (`Universalism`, `Power`). For values like
`Security`, the same surface vocabulary can map to `+1`, `0`, or `-1`
depending on whether the persona is protecting stability, merely noticing
routine, or actively choosing a less settled path. Those values need
family-specific banks and judged-label acceptance checks, not just more
generic "unsettled" writing.

This file is read by the orchestrator (following the pattern in
`.claude/skills/judge/orchestration.md` lines 49-73) and embedded inline
in each subagent's prompt when `TARGET_TENSIONS` is non-empty.

## When It Applies

- **Active**: `TARGET_TENSIONS` is non-empty (e.g., `["Universalism"]`, `["Power"]`)
- **Inactive**: `TARGET_TENSIONS` is empty (`[]`) — pipeline is completely
  unchanged, this file is never read

## Scenario Banks

### Universalism

Scenarios that surface failures to act on broader-good values — advocacy,
ethical choices, environmental/social justice — rather than personal
compromises. These produce entries the judge can score as Universalism -1.

```yaml
universalism_tension_scenarios:
  - "Something happened where you could have spoken up about something unfair — but you didn't. Maybe it was easier to stay quiet. Just describe what happened."
  - "You made a choice today that was convenient but doesn't sit right with your sense of what's right for the world. Describe the moment."
  - "Someone around you did or said something that goes against what you believe in, and you let it pass. Describe the situation."
  - "You chose the easier, less ethical option today — the one that benefits you but maybe not others. What happened?"
```

These already tend to generate reliable `-1` signals, so they can stay as a
single default bank unless a future batch needs a narrower subtype.

### Power

Scenarios that surface moments where the persona's sense of agency,
voice, or standing was diminished — without naming the value directly.
These produce entries the judge can score as Power -1.

```yaml
power_tension_scenarios:
  - "Someone spoke over you today in a way that made you feel smaller — maybe they cut you off, presented your idea as theirs, or made a decision that should have been yours. You didn't push back. Just describe what happened."
  - "You needed something today — an answer, a resource, a green light — and the only way to get it was to wait for someone else to decide. You asked, and then you just had to sit there. Describe how that played out."
  - "You had a clear view on how something should go — and someone pushed back, hard. You ended up going along with their way instead. Maybe it was easier, maybe the fight wasn't worth it. Describe the moment you gave in."
  - "Someone treated you today like your experience or track record didn't count for much — maybe they overlooked you for something, explained something you already knew, or assumed you needed to prove yourself all over again. What happened?"
```

These also already aim directly at the negative pole, so a single bank is
usually sufficient unless a future batch wants separate status-loss vs
agency-loss subfamilies.

### Hedonism

Scenarios that surface quiet pleasure/rest tensions rather than loud
indulgence. These are designed for the 691.2 batch, where the main issue is
that restorative or permission-giving moments can be drowned out by duty,
achievement, or caregiving cues.

```yaml
hedonism_tension_scenarios:
  - "You made room for something small that felt good today — rest, food, craft, music, being left alone, whatever it was — and then immediately questioned whether you had earned it. Describe the moment."
  - "You were tired enough that rest or comfort would have helped, but part of you kept translating that into laziness or selfishness. What happened?"
  - "Something genuinely pleasant happened today, but another obligation stayed in the room with you the whole time. Describe what it was like to try to enjoy it anyway."
  - "You chose the useful or disciplined option over the enjoyable one today, and the trade-off made sense, but it also felt a little bleak. What happened?"
```

This bank is intentionally tuned for quiet `Hedonism +1` and mixed `0/+1`
boundary cases. If a future batch needs reliable `Hedonism -1`, define a
dedicated family-specific negative bank rather than reusing these prompts.

### Security

Security needs family-specific banks because "safety language" is not enough
to determine polarity. The same vocabulary can describe buffer-building
(`+1`), diffuse responsibility/routine (`0`), or choosing autonomy/novelty
over a steadier path (`-1`).

```yaml
security_negative_tradeoff_scenarios:
  - "The practical option today would have protected something real — money, benefits, routine, childcare, keeping the peace at home — but taking it would have meant letting someone else set the terms. You still chose the less settled path. What happened?"
  - "You knew exactly what the safer move was today, and you could name the costs of not taking it. You still couldn't make yourself choose it. Describe the moment."
  - "Something stable was on offer today — steady pay, predictable hours, less risk, fewer family worries — and you turned away from it because the trade felt wrong in your body. What happened?"
  - "You did the math today and the math pointed one way, but some other part of you chose uncertainty anyway. Describe what the choice was and what you gave up."

security_positive_ambiguous_scenarios:
  - "You did something today that might look overly cautious from the outside, but to you it felt like keeping life from getting harder later. Describe it."
  - "You protected a buffer today — time, money, routine, coverage, a relationship, your home — and part of you wondered whether you were being wise or just scared. What happened?"
  - "A decision today was about staying settled more than chasing upside. From the outside it might read as playing small. Describe why it did or didn't feel that way to you."
  - "You kept something steady today in a way that doesn't make for a dramatic story — it just made tomorrow feel more manageable. Describe the moment."

self_direction_vs_security_conflict_scenarios:
  - "A choice today was between the arrangement that keeps life steadier and the one that lets you work or live on your own terms. Describe the moment and which pull was stronger."
  - "Someone offered a version of help, structure, or certainty today that would have made life easier, but it also would have meant doing things their way. What happened?"
  - "You defended a plan today because it keeps life workable, but you also needed it to still feel like your choice. Describe where that tension showed up."
  - "The safer option today came bundled with someone else's expectations about how you should live or work. Describe what you did with that."

stimulation_vs_security_conflict_scenarios:
  - "Something new or alive pulled at you today, but saying yes would have made the rest of life less settled. Describe how close you came to taking it."
  - "You could have kept things predictable today, or followed the option that felt more vivid, risky, or open-ended. What happened?"
  - "A change showed up today that sounded exciting for exactly the reasons it also felt hard to justify. Describe the trade-off."
  - "You caught yourself wanting the version of today that would have disrupted the routine, the budget, or the plan. Describe what that pull felt like."

security_tension_scenarios:
  - "A decision today changed how settled you felt, but it wasn't obvious whether that was because you were being careful, boxed in, or both. Describe the moment."
  - "Something in your life felt safer on paper than it felt in your chest today. What happened?"
  - "You talked yourself through a practical choice today and still couldn't decide whether it counted as stability, avoidance, or just adulthood. Describe it."
  - "You kept noticing security-language today — risk, buffer, coverage, routine, stability — but the real issue underneath it was harder to name. What happened?"
```

## Application Rule

When `TARGET_TENSIONS` is active and the persona has a matching core value:

1. **ALL Unsettled entries** use a value-specific scenario bank instead of
   the generic Unsettled text from `journal_entry.yaml` (lines 43-44).
2. If the batch spec provides **family-specific targets** for that persona
   or value, select the scenario bank by family name first. For example:
   - `security_negative_tradeoff`
   - `security_positive_ambiguous`
   - `self_direction_vs_security_conflict`
   - `stimulation_vs_security_conflict`
3. **Cycle through scenarios sequentially within the chosen bank**:
   1st use scenario 1, 2nd scenario 2, 3rd scenario 3, 4th scenario 4,
   5th wraps back to scenario 1, etc.
4. If no family-specific bank is requested, fall back to the generic
   `<value>_tension_scenarios` bank when one exists.
5. **Grounded and Neutral entries are unaffected** — they continue to use
   the standard prompts from `journal_entry.yaml`.

When `TARGET_TENSIONS` is empty:

- Pipeline is **completely unchanged** — this file is never consulted

## Integration with journal_entry.yaml

The subagent still uses `journal_entry.yaml` for everything:
- Persona context (name, age, profession, culture, bio)
- Tone and verbosity settings
- Style rules (no "Dear Diary", no therapy speak, etc.)
- Cultural context instructions
- Previous entry continuity

Only the **"What to write about"** section (lines 43-44 of
`journal_entry.yaml`) is replaced when reflection_mode is Unsettled and
`TARGET_TENSIONS` applies to this persona's core values. The replacement
text comes from the chosen scenario bank above.

## Guardrails

- `TARGET_TENSIONS` is **generation-time only** — it is never written to
  persona markdown files, the registry, or judge context
- Family selections such as `security_negative_tradeoff` are also
  **generation-time only** — they are steering hints for coverage, not part
  of the runtime text or judge context
- The **judge pipeline is completely untouched** — judges score entries
  without any knowledge of whether tension scenarios were used
- Entries produced with tension scenarios are **indistinguishable** in
  output format from regular entries
- The scenario bank text follows the same style as the existing Unsettled
  prompt: second-person, present-tense, no value labels, no Schwartz
  terminology
- Scenario banks for hard-negative lift should prefer **mild misalignment**
  or **boundary ambiguity** over obviously extreme failures so the batch
  helps the hard `0/-1` edge rather than only adding easy negatives
- For polarity-repair batches, acceptance should be based on **judged labels**
  after generation, not on `Reflection Mode`, family name, or prompt intent

## Adding New Value Scenarios

To address imbalance in other values, add a new section under
"Scenario Banks" following the same pattern:

```yaml
<value_name>_tension_scenarios:
  - "Scenario that surfaces failure to act on <value> ..."
  - "..."
```

If a value needs more controlled polarity or conflict coverage, define
family-specific banks as well:

```yaml
<value_name>_negative_tradeoff_scenarios:
  - "..."
<value_name>_positive_ambiguous_scenarios:
  - "..."
```

Then add the value to `TARGET_TENSIONS` in the generation config and map the
family names intentionally in the batch spec instead of relying on a single
generic bank.
