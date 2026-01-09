# Run Configuration

**Timestamp**: 2026-01-09 09:37:09
**Method**: Claude Code subagents (parallel with run_in_background)

## Parameters
- Personas: 5
- Entries per persona: 3-10 (variable)
- Start date: 2025-12-01
- Days between entries: 2-10 (random)

## Nudge Settings
- Decision method: LLM-based classification (prompts/nudge_decision.yaml)
- Session cap: 2+ nudges in last 3 entries → no nudge
- Response probability: 0.7

## Response Modes
- Answering directly: 50%
- Deflecting/redirecting: 30%
- Revealing deeper thought: 20%

## Personas Generated

| ID | Name | Age | Profession | Culture | Values | Entries |
|----|------|-----|------------|---------|--------|---------|
| 001 | Gabriela Mendoza | 25-34 | Parent (Stay-at-home) | Latin American | Power | 7 |
| 002 | Wei-Lin Chen | 45-54 | Grad Student | East Asian | Power, Self-Direction | 3 |
| 003 | Tarek Al-Rashid | 45-54 | Manager | Middle Eastern | Benevolence | 9 |
| 004 | Marcus Chen | 35-44 | Entrepreneur | North American | Conformity | 9 |
| 005 | Karen Mitchell | 45-54 | Nurse | North American | Conformity | 9 |

## Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total Entries | 37 |
| Total Nudges | 22 |
| Total Responses | 17 |
| Response Rate | 77.3% |
| Entries per Persona | min=3, max=9, avg=7.4 |
