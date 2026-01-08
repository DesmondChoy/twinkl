# Run Configuration

**Timestamp**: 2026-01-08 22:31:11
**Method**: Claude Code subagents (parallel with run_in_background)

## Parameters
| Parameter | Value |
|-----------|-------|
| Personas | 5 |
| Entries per persona | 3-10 (random) |
| Start date | 2025-12-01 |
| Days between entries | 2-10 (random) |
| Nudge decision | LLM-based classification |
| Response probability | 0.7 |

## Persona Summary
| ID | Name | Age | Profession | Culture | Values | Entries |
|----|------|-----|------------|---------|--------|---------|
| 1 | Priya Sharma | 25-34 | Gig Worker | South Asian | Self-Direction, Conformity | 10 |
| 2 | Mateo Reyes | 35-44 | Gig Worker | Latin American | Universalism | 5 |
| 3 | Marcus Reyes | 35-44 | Artist | North American | Stimulation | 5 |
| 4 | Rodrigo Mendes | 35-44 | Entrepreneur | Latin American | Achievement | 9 |
| 5 | Tariq Hassan | 25-34 | Gig Worker | Middle Eastern | Security, Benevolence | 4 |

## Nudge Settings
- Categories: clarification, elaboration, tension_surfacing
- Session cap: 2 nudges in last 3 entries → no_nudge
- Nudge length: 2-12 words
- Response modes:
  - Answering directly (50%)
  - Deflecting/redirecting (30%)
  - Revealing deeper thought (20%)

## Source Files
- Config: `config/synthetic_data.yaml`
- Values: `config/schwartz_values.yaml`
- Prompts: `prompts/*.yaml`
- Instructions: `docs/synthetic_data/claude_gen_instructions.md`
