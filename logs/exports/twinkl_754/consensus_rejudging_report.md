# twinkl-754 Consensus Re-judging Report

## Scope Summary

- Prompt condition: `profile_only`
- Bundle mode: `full`
- Entries: `1651`
- Passes: `5`
- Personas: `204`
- Worker model: `gpt-5.4`

## 1. Judge Repeated-Call Self-Consistency

These kappas measure repeated-call consistency of the same judge workflow, not agreement among independent raters.

| Dimension | Fleiss kappa | Human baseline |
| --- | --- | --- |
| self_direction | 0.765 | N/A |
| stimulation | 0.809 | 0.580 |
| hedonism | 0.797 | 0.640 |
| achievement | 0.789 | N/A |
| power | 0.793 | N/A |
| security | 0.775 | 0.480 |
| conformity | 0.781 | N/A |
| tradition | 0.809 | N/A |
| benevolence | 0.779 | N/A |
| universalism | 0.890 | N/A |

## 2. Consensus vs Persisted

| Dimension | Consensus vs persisted Cohen kappa |
| --- | --- |
| self_direction | 0.714 |
| stimulation | 0.804 |
| hedonism | 0.776 |
| achievement | 0.750 |
| power | 0.753 |
| security | 0.775 |
| conformity | 0.785 |
| tradition | 0.778 |
| benevolence | 0.763 |
| universalism | 0.898 |
| aggregate | 0.778 |

Confusion counts:

| Dimension | Persisted | Consensus | Count |
| --- | --- | --- | --- |
| achievement | -1 | -1 | 39 |
| achievement | -1 | 0 | 38 |
| achievement | -1 | 1 | 8 |
| achievement | 0 | -1 | 10 |
| achievement | 0 | 0 | 1055 |
| achievement | 0 | 1 | 65 |
| achievement | 1 | -1 | 1 |
| achievement | 1 | 0 | 63 |
| achievement | 1 | 1 | 372 |
| benevolence | -1 | -1 | 75 |
| benevolence | -1 | 0 | 42 |
| benevolence | -1 | 1 | 6 |
| benevolence | 0 | -1 | 6 |
| benevolence | 0 | 0 | 1003 |
| benevolence | 0 | 1 | 64 |
| benevolence | 1 | -1 | 3 |
| benevolence | 1 | 0 | 68 |
| benevolence | 1 | 1 | 384 |
| conformity | -1 | -1 | 93 |
| conformity | -1 | 0 | 36 |
| conformity | 0 | -1 | 15 |
| conformity | 0 | 0 | 1156 |
| conformity | 0 | 1 | 18 |
| conformity | 1 | -1 | 7 |
| conformity | 1 | 0 | 70 |
| conformity | 1 | 1 | 256 |
| hedonism | -1 | -1 | 105 |
| hedonism | -1 | 0 | 35 |
| hedonism | -1 | 1 | 1 |
| hedonism | 0 | -1 | 34 |
| hedonism | 0 | 0 | 1277 |
| hedonism | 0 | 1 | 26 |
| hedonism | 1 | -1 | 2 |
| hedonism | 1 | 0 | 23 |
| hedonism | 1 | 1 | 148 |
| power | -1 | -1 | 80 |
| power | -1 | 0 | 48 |
| power | -1 | 1 | 18 |
| power | 0 | -1 | 4 |
| power | 0 | 0 | 1314 |
| power | 0 | 1 | 20 |
| power | 1 | -1 | 2 |
| power | 1 | 0 | 31 |
| power | 1 | 1 | 134 |
| security | -1 | -1 | 117 |
| security | -1 | 0 | 26 |
| security | -1 | 1 | 8 |
| security | 0 | -1 | 16 |
| security | 0 | 0 | 1146 |
| security | 0 | 1 | 50 |
| security | 1 | -1 | 9 |
| security | 1 | 0 | 47 |
| security | 1 | 1 | 232 |
| self_direction | -1 | -1 | 118 |
| self_direction | -1 | 0 | 83 |
| self_direction | -1 | 1 | 15 |
| self_direction | 0 | -1 | 14 |
| self_direction | 0 | 0 | 965 |
| self_direction | 0 | 1 | 40 |
| self_direction | 1 | 0 | 87 |
| self_direction | 1 | 1 | 329 |
| stimulation | -1 | -1 | 36 |
| stimulation | -1 | 0 | 19 |
| stimulation | -1 | 1 | 5 |
| stimulation | 0 | -1 | 7 |
| stimulation | 0 | 0 | 1447 |
| stimulation | 0 | 1 | 10 |
| stimulation | 1 | 0 | 22 |
| stimulation | 1 | 1 | 105 |
| tradition | -1 | -1 | 34 |
| tradition | -1 | 0 | 19 |
| tradition | -1 | 1 | 5 |
| tradition | 0 | -1 | 2 |
| tradition | 0 | 0 | 1315 |
| tradition | 0 | 1 | 20 |
| tradition | 1 | -1 | 1 |
| tradition | 1 | 0 | 61 |
| tradition | 1 | 1 | 194 |
| universalism | -1 | -1 | 45 |
| universalism | -1 | 0 | 6 |
| universalism | -1 | 1 | 5 |
| universalism | 0 | -1 | 1 |
| universalism | 0 | 0 | 1427 |
| universalism | 0 | 1 | 8 |
| universalism | 1 | 0 | 18 |
| universalism | 1 | 1 | 141 |

## 3. Consensus vs Human

| Dimension | Consensus vs human | Persisted vs human |
| --- | --- | --- |
| self_direction | 0.483 | 0.699 |
| stimulation | 0.714 | 0.763 |
| hedonism | 0.554 | 0.638 |
| achievement | 0.696 | 0.731 |
| power | 0.662 | 0.661 |
| security | 0.501 | 0.495 |
| conformity | 0.643 | 0.654 |
| tradition | 0.784 | 0.806 |
| benevolence | 0.737 | 0.748 |
| universalism | 0.883 | 0.924 |
| aggregate | 0.674 | 0.723 |

## 4. Confidence Tier Distribution

| Dimension | Tier | Entries | Non-neutral entries |
| --- | --- | --- | --- |
| achievement | bare_majority | 143 | 65 |
| achievement | strong | 167 | 73 |
| achievement | unanimous | 1341 | 357 |
| benevolence | bare_majority | 170 | 87 |
| benevolence | no_majority | 1 | 0 |
| benevolence | strong | 171 | 65 |
| benevolence | unanimous | 1309 | 386 |
| conformity | bare_majority | 118 | 66 |
| conformity | no_majority | 2 | 0 |
| conformity | strong | 165 | 66 |
| conformity | unanimous | 1366 | 257 |
| hedonism | bare_majority | 100 | 46 |
| hedonism | strong | 132 | 49 |
| hedonism | unanimous | 1419 | 221 |
| power | bare_majority | 100 | 56 |
| power | strong | 90 | 32 |
| power | unanimous | 1461 | 170 |
| security | bare_majority | 128 | 75 |
| security | no_majority | 1 | 0 |
| security | strong | 197 | 81 |
| security | unanimous | 1325 | 276 |
| self_direction | bare_majority | 171 | 82 |
| self_direction | no_majority | 2 | 0 |
| self_direction | strong | 202 | 64 |
| self_direction | unanimous | 1276 | 370 |
| stimulation | bare_majority | 46 | 29 |
| stimulation | strong | 75 | 23 |
| stimulation | unanimous | 1530 | 111 |
| tradition | bare_majority | 71 | 40 |
| tradition | no_majority | 1 | 0 |
| tradition | strong | 105 | 41 |
| tradition | unanimous | 1474 | 175 |
| universalism | bare_majority | 40 | 26 |
| universalism | strong | 38 | 17 |
| universalism | unanimous | 1573 | 157 |

## 5. Pass Diagnostics

| Pass | Attempt | Worker model | Raw hash | Score hash | Rationale coverage | Completed |
| --- | --- | --- | --- | --- | --- | --- |
| pass_1 | 3 | gpt-5.4 | 1053dde9b730 | 6addd04601da | 1.000 | 2026-03-20T11:43:20.871496+00:00 |
| pass_2 | 1 | gpt-5.4 | 09c747ca998f | 4233aa814640 | 1.000 | 2026-03-20T12:03:35.101592+00:00 |
| pass_3 | 3 | gpt-5.4 | 36539617f090 | 8aa482cd5159 | 1.000 | 2026-03-20T12:28:35.540445+00:00 |
| pass_4 | 1 | gpt-5.4 | 469fa3426d63 | 5c99e7091b8c | 1.000 | 2026-03-21T00:12:46.335614+00:00 |
| pass_5 | 4 | gpt-5.4 | 0dcfd26d647a | 9542cfff71d8 | 1.000 | 2026-03-26T15:01:38.367259+00:00 |

Pairwise similarity:

| Pass pair | Raw hash match | Score hash match | Identical entry vectors | Differing entry vectors |
| --- | --- | --- | --- | --- |
| pass_1 vs pass_2 | False | False | 834 | 817 |
| pass_1 vs pass_3 | False | False | 868 | 783 |
| pass_1 vs pass_4 | False | False | 891 | 760 |
| pass_1 vs pass_5 | False | False | 817 | 834 |
| pass_2 vs pass_3 | False | False | 833 | 818 |
| pass_2 vs pass_4 | False | False | 846 | 805 |
| pass_2 vs pass_5 | False | False | 773 | 878 |
| pass_3 vs pass_4 | False | False | 865 | 786 |
| pass_3 vs pass_5 | False | False | 823 | 828 |
| pass_4 vs pass_5 | False | False | 832 | 819 |

## 6. Hard-Dimension Gate

- Aggregate consensus-vs-human kappa: `0.674`
- Aggregate persisted-vs-human kappa: `0.723`
- Agreement gate passed: `False`
- Confidence gate passed: `True`
- Overall retrain gate passed: `False`

| Dimension | Non-neutral labels | Bare/no-majority labels | Low-confidence ratio | Passes |
| --- | --- | --- | --- | --- |
| security | 432 | 75 | 0.174 | True |
| hedonism | 316 | 46 | 0.146 | True |
| stimulation | 163 | 29 | 0.178 | True |

## 7. Hard-Dimension Deep Dive

### security

Persisted-vs-consensus kappa: `0.775`; consensus-vs-human kappa: `0.501`; low-confidence non-neutral labels: `75/432`.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 117 |
| -1 | 0 | 26 |
| -1 | 1 | 8 |
| 0 | -1 | 16 |
| 0 | 0 | 1146 |
| 0 | 1 | 50 |
| 1 | -1 | 9 |
| 1 | 0 | 47 |
| 1 | 1 | 232 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 128 | 75 |
| no_majority | 1 | 0 |
| strong | 197 | 81 |
| unanimous | 1325 | 276 |

### hedonism

Persisted-vs-consensus kappa: `0.776`; consensus-vs-human kappa: `0.554`; low-confidence non-neutral labels: `46/316`.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 105 |
| -1 | 0 | 35 |
| -1 | 1 | 1 |
| 0 | -1 | 34 |
| 0 | 0 | 1277 |
| 0 | 1 | 26 |
| 1 | -1 | 2 |
| 1 | 0 | 23 |
| 1 | 1 | 148 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 100 | 46 |
| strong | 132 | 49 |
| unanimous | 1419 | 221 |

### stimulation

Persisted-vs-consensus kappa: `0.804`; consensus-vs-human kappa: `0.714`; low-confidence non-neutral labels: `29/163`.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 36 |
| -1 | 0 | 19 |
| -1 | 1 | 5 |
| 0 | -1 | 7 |
| 0 | 0 | 1447 |
| 0 | 1 | 10 |
| 1 | 0 | 22 |
| 1 | 1 | 105 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 46 | 29 |
| strong | 75 | 23 |
| unanimous | 1530 | 111 |


## 8. Rationale Source Summary

- Entries with a perfect 10/10 rationale-source match: `1622`
- Entries using fallback rationale selection: `29`
- Maximum label mismatches on a chosen rationale source: `1`

| Source pass | Mismatch count | Entries |
| --- | --- | --- |
| 1 | 0 | 1143 |
| 1 | 1 | 24 |
| 2 | 0 | 281 |
| 2 | 1 | 4 |
| 3 | 0 | 128 |
| 3 | 1 | 1 |
| 4 | 0 | 55 |
| 5 | 0 | 15 |

## 9. Label Migration Summary

| Dimension | Persisted | Consensus | Changed entries |
| --- | --- | --- | --- |
| achievement | -1 | 0 | 38 |
| achievement | -1 | 1 | 8 |
| achievement | 0 | -1 | 10 |
| achievement | 0 | 1 | 65 |
| achievement | 1 | -1 | 1 |
| achievement | 1 | 0 | 63 |
| benevolence | -1 | 0 | 42 |
| benevolence | -1 | 1 | 6 |
| benevolence | 0 | -1 | 6 |
| benevolence | 0 | 1 | 64 |
| benevolence | 1 | -1 | 3 |
| benevolence | 1 | 0 | 68 |
| conformity | -1 | 0 | 36 |
| conformity | 0 | -1 | 15 |
| conformity | 0 | 1 | 18 |
| conformity | 1 | -1 | 7 |
| conformity | 1 | 0 | 70 |
| hedonism | -1 | 0 | 35 |
| hedonism | -1 | 1 | 1 |
| hedonism | 0 | -1 | 34 |
| hedonism | 0 | 1 | 26 |
| hedonism | 1 | -1 | 2 |
| hedonism | 1 | 0 | 23 |
| power | -1 | 0 | 48 |
| power | -1 | 1 | 18 |
| power | 0 | -1 | 4 |
| power | 0 | 1 | 20 |
| power | 1 | -1 | 2 |
| power | 1 | 0 | 31 |
| security | -1 | 0 | 26 |
| security | -1 | 1 | 8 |
| security | 0 | -1 | 16 |
| security | 0 | 1 | 50 |
| security | 1 | -1 | 9 |
| security | 1 | 0 | 47 |
| self_direction | -1 | 0 | 83 |
| self_direction | -1 | 1 | 15 |
| self_direction | 0 | -1 | 14 |
| self_direction | 0 | 1 | 40 |
| self_direction | 1 | 0 | 87 |
| stimulation | -1 | 0 | 19 |
| stimulation | -1 | 1 | 5 |
| stimulation | 0 | -1 | 7 |
| stimulation | 0 | 1 | 10 |
| stimulation | 1 | 0 | 22 |
| tradition | -1 | 0 | 19 |
| tradition | -1 | 1 | 5 |
| tradition | 0 | -1 | 2 |
| tradition | 0 | 1 | 20 |
| tradition | 1 | -1 | 1 |
| tradition | 1 | 0 | 61 |
| universalism | -1 | 0 | 6 |
| universalism | -1 | 1 | 5 |
| universalism | 0 | -1 | 1 |
| universalism | 0 | 1 | 8 |
| universalism | 1 | 0 | 18 |

## 10. Recommendation

Stop after repeated-call diagnostics review; do not retrain until the gate is addressed.
