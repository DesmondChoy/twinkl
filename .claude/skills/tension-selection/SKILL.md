---
name: tension-selection
description: Value-specific tension scenarios for synthetic data generation. Provides scenario banks that replace the generic Unsettled prompt when TARGET_TENSIONS is active.
---

# Tension-Selection Skill

## Purpose

This skill addresses label imbalance for specific Schwartz value dimensions
(e.g., Universalism -1 at 0.6% vs 5.4%+ for all other values). The root
cause: the generic Unsettled prompt in `journal_entry.yaml` produces
personal dilemmas that the judge scores as neutral (0) rather than
misaligned (-1) for broader-good values like Universalism.

This file is read by the orchestrator (following the pattern in
`.claude/skills/judge/orchestration.md` lines 49-73) and embedded inline
in each subagent's prompt when `TARGET_TENSIONS` is non-empty.

## When It Applies

- **Active**: `TARGET_TENSIONS` is non-empty (e.g., `["Universalism"]`)
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

## Adding New Value Scenarios

To address imbalance in other values, add a new section under
"Scenario Banks" following the same pattern:

```yaml
<value_name>_tension_scenarios:
  - "Scenario that surfaces failure to act on <value> ..."
  - "..."
```

Then add the value to `TARGET_TENSIONS` in the generation config.
