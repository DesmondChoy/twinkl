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

### Security

Scenarios that surface small but meaningful tensions around stability,
preparedness, or peace of mind — not catastrophic danger. These are broader
than the 681.5 version because later batches need not only mild negatives,
but also polarity-ambiguous cases where steadiness competes with autonomy or
novelty.

```yaml
security_tension_scenarios:
  - "A decision today came down to steadiness versus doing it your own way. You chose one, but the other kept tugging. Describe the moment."
  - "Something tempting or new opened up today, but taking it would have made the rest of life feel less settled. What happened?"
  - "You did something practical today to protect a buffer — time, money, routine, coverage, a relationship, your home — but part of you wondered if you were being too cautious. Describe it."
  - "A small risk showed up today. You could absorb it, plan around it, or ignore it. Whatever you did, it changed how settled you felt. Describe the moment."
```

## Application Rule

When `TARGET_TENSIONS` is active and the persona has a matching core value:

1. **ALL Unsettled entries** use the scenario bank instead of the generic
   Unsettled text from `journal_entry.yaml` (lines 43-44)
2. **Cycle through scenarios sequentially**: 1st Unsettled entry uses
   scenario 1, 2nd uses scenario 2, 3rd uses scenario 3, 4th uses
   scenario 4, 5th wraps back to scenario 1, etc.
3. **Grounded and Neutral entries are unaffected** — they continue to use
   the standard prompts from `journal_entry.yaml`

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
TARGET_TENSIONS applies to this persona's core values. The replacement
text comes from the scenario bank above.

## Guardrails

- `TARGET_TENSIONS` is **generation-time only** — it is never written to
  persona markdown files, the registry, or judge context
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

## Adding New Value Scenarios

To address imbalance in other values, add a new section under
"Scenario Banks" following the same pattern:

```yaml
<value_name>_tension_scenarios:
  - "Scenario that surfaces failure to act on <value> ..."
  - "..."
```

Then add the value to `TARGET_TENSIONS` in the generation config.
