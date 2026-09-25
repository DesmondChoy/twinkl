# North Star Moment impact pilot

Judge: `claude-opus-5-5` at effort `medium`, run as the `nsm-impact-judge` Claude Code subagent (prompt `nsm-impact-judge-1.0`).
Pairs: the 22 saved demo comparison pairs, each judged in both orders.
These are AI review results on synthetic Personas, not human validation.

## Judge reliability

- Order consistency: 18/22 pairs gave the same verdict in both orders.
- Position choice among decisive judgments: Response 1 19, Response 2 21 (of 40).

## Result

- Wins 16, losses 2, ties 4 (inconsistent orders count as ties).
- Win difference: 63.6 points; 95% Persona-resampled range: [48.0, 82.4].
  With only 5 Personas, this range is a rough indication.

| North Star Moment mode | Pairs | Wins | Losses | Ties |
| --- | ---: | ---: | ---: | ---: |
| encouragement | 17 | 14 | 1 | 2 |
| reflection | 4 | 2 | 1 | 1 |
| reminder | 1 | 0 | 0 | 1 |

## Honesty flags

Flagged judgments: with the North Star Moment 3/44, without 7/44.
