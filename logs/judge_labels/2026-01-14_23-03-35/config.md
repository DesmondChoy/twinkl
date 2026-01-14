# Judge Labeling Run Configuration

**Timestamp**: 2026-01-14_23-03-35
**Source**: logs/wrangled/2026-01-09_09-37-09/
**Method**: Claude Code subagents (parallel)

## Parameters
- Personas: 5
- Total entries: 37

## Persona Summary
| ID | Name | Entries | Core Values |
|----|------|---------|-------------|
| 1 | Gabriela Mendoza | 7 | Power |
| 2 | Wei-Lin Chen | 3 | Power, Self-Direction |
| 3 | Tarek Al-Rashid | 9 | Benevolence |
| 4 | Marcus Chen | 9 | Conformity |
| 5 | Karen Mitchell | 9 | Conformity |

## Value Rubric Source
- `config/schwartz_values.yaml` (core_motivation + first 3 behavioral_manifestations)

## Output Files
- `persona_*_labels.json` - Per-persona raw scores
- `judge_labels.parquet` - Consolidated DataFrame
- `validation_report.md` - Quality metrics
