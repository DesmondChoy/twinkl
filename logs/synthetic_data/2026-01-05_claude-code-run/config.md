# Run Configuration

**Timestamp**: 2026-01-05
**Method**: Claude Code subagents (parallel with run_in_background)

## Parameters
- Personas: 5
- Entries per persona: 5
- Start date: 2025-10-25
- Min days between entries: 2
- Max days between entries: 10

## Nudge Settings
- Base probability: 0.4
- Response probability: 0.7
- Category weights:
  - clarification: 0.25
  - elaboration: 0.35
  - tension_surfacing: 0.25
  - grounding: 0.15

## Nudge Decision Rules Applied
1. Session cap: 2+ nudges in last 3 entries → No nudge
2. Entry too vague: <15 words, no concrete details → "clarification"
3. Hedging language + Unsettled mode → "tension_surfacing"
4. Grounded mode + brief (<50 words) → "grounding"
5. Random gate (40% chance) → "elaboration"
6. Otherwise → No nudge

## Model
- Orchestrator: Claude Opus 4.5 (claude-opus-4-5-20251101)
- Subagents: general-purpose (Claude Code Task tool)

## Execution Statistics
- Total subagent calls: ~50 (entries + nudges + responses)
- Parallel entry launches: 5 (Entry 1 for all personas simultaneously)
- Sequential processing: Entries 2-5 with nudge decisions between
