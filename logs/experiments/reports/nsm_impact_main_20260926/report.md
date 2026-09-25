# North Star Moment impact main run

Judge: `claude-opus-5-5` at effort `medium`, run as the `nsm-impact-judge` Claude Code subagent (prompt `nsm-impact-judge-1.0`).
Pairs: 72 generated pairs from `logs/experiments/reports/nsm_impact_generation_20260926/pairs.json`, each judged in both orders.
These are AI review results on synthetic Personas, not human validation.

## Judge reliability

- Order consistency: 61/72 pairs gave the same verdict in both orders.
- Position choice among decisive judgments: Response 1 52, Response 2 56 (of 108).

## Result

- Wins 36, losses 12, ties 24 (inconsistent orders count as ties).
- Win difference: 33.3 points; 95% Persona-resampled range: [19.4, 45.5].
  With only 23 Personas, this range is a rough indication.

| North Star Moment mode | Pairs | Wins | Losses | Ties |
| --- | ---: | ---: | ---: | ---: |
| encouragement | 58 | 27 | 11 | 20 |
| reflection | 7 | 2 | 1 | 4 |
| reminder | 7 | 7 | 0 | 0 |

## Honesty flags

Flagged judgments: with the North Star Moment 2/144, without 0/144.
