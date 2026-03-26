# twinkl-754 Consensus Re-judging Report

## Scope Summary

- Prompt condition: `profile_only`
- Bundle mode: `pilot`
- Entries: `50`
- Passes: `5`
- Personas: `34`
- Worker model: `gpt-5.4`

## 1. Judge Repeated-Call Self-Consistency

These kappas measure repeated-call consistency of the same judge workflow, not agreement among independent raters.

Pilot selection:
- Requested entries: `50`
- Selected entries: `50`
- Selected non-zero `security` entries: `30`
- Selected non-zero `hedonism` entries: `18`
- Selected non-zero `stimulation` entries: `18`

| Dimension | Fleiss kappa | Human baseline |
| --- | --- | --- |
| self_direction | 0.774 | N/A |
| stimulation | 0.897 | 0.580 |
| hedonism | 0.829 | 0.640 |
| achievement | 0.765 | N/A |
| power | 0.870 | N/A |
| security | 0.855 | 0.480 |
| conformity | 0.849 | N/A |
| tradition | 0.807 | N/A |
| benevolence | 0.829 | N/A |
| universalism | 1.000 | N/A |

## 2. Consensus vs Persisted

| Dimension | Consensus vs persisted Cohen kappa |
| --- | --- |
| self_direction | 0.766 |
| stimulation | 0.790 |
| hedonism | 0.839 |
| achievement | 0.700 |
| power | 0.793 |
| security | 0.877 |
| conformity | 0.914 |
| tradition | 0.570 |
| benevolence | 0.715 |
| universalism | 1.000 |
| aggregate | 0.813 |

Confusion counts:

| Dimension | Persisted | Consensus | Count |
| --- | --- | --- | --- |
| achievement | 0 | 0 | 34 |
| achievement | 0 | 1 | 2 |
| achievement | 1 | -1 | 1 |
| achievement | 1 | 0 | 3 |
| achievement | 1 | 1 | 10 |
| benevolence | -1 | -1 | 2 |
| benevolence | -1 | 0 | 2 |
| benevolence | 0 | 0 | 29 |
| benevolence | 0 | 1 | 3 |
| benevolence | 1 | 0 | 2 |
| benevolence | 1 | 1 | 12 |
| conformity | -1 | -1 | 6 |
| conformity | -1 | 0 | 2 |
| conformity | 0 | 0 | 34 |
| conformity | 1 | 1 | 8 |
| hedonism | -1 | -1 | 8 |
| hedonism | -1 | 0 | 1 |
| hedonism | 0 | 0 | 32 |
| hedonism | 1 | -1 | 1 |
| hedonism | 1 | 0 | 2 |
| hedonism | 1 | 1 | 6 |
| power | -1 | -1 | 1 |
| power | -1 | 0 | 1 |
| power | 0 | 0 | 47 |
| power | 1 | 1 | 1 |
| security | -1 | -1 | 11 |
| security | -1 | 0 | 2 |
| security | 0 | 0 | 20 |
| security | 1 | 0 | 2 |
| security | 1 | 1 | 15 |
| self_direction | -1 | -1 | 3 |
| self_direction | -1 | 1 | 1 |
| self_direction | 0 | -1 | 1 |
| self_direction | 0 | 0 | 29 |
| self_direction | 0 | 1 | 2 |
| self_direction | 1 | 0 | 2 |
| self_direction | 1 | 1 | 12 |
| stimulation | -1 | -1 | 5 |
| stimulation | -1 | 0 | 2 |
| stimulation | 0 | 0 | 32 |
| stimulation | 1 | 0 | 3 |
| stimulation | 1 | 1 | 8 |
| tradition | -1 | 0 | 1 |
| tradition | 0 | 0 | 41 |
| tradition | 1 | 0 | 4 |
| tradition | 1 | 1 | 4 |
| universalism | 0 | 0 | 48 |
| universalism | 1 | 1 | 2 |

## 3. Consensus vs Human

| Dimension | Consensus vs human | Persisted vs human |
| --- | --- | --- |
| self_direction | 0.412 | 0.438 |
| stimulation | 0.747 | 0.893 |
| hedonism | 0.522 | 0.522 |
| achievement | 0.759 | 0.774 |
| power | 1.000 | 1.000 |
| security | 0.534 | 0.513 |
| conformity | 0.356 | 0.293 |
| tradition | 0.352 | 0.581 |
| benevolence | 1.000 | 0.883 |
| universalism | 1.000 | 1.000 |
| aggregate | 0.636 | 0.651 |

## 4. Confidence Tier Distribution

| Dimension | Tier | Entries | Non-neutral entries |
| --- | --- | --- | --- |
| achievement | bare_majority | 5 | 4 |
| achievement | strong | 3 | 0 |
| achievement | unanimous | 42 | 9 |
| benevolence | bare_majority | 4 | 2 |
| benevolence | strong | 4 | 3 |
| benevolence | unanimous | 42 | 12 |
| conformity | bare_majority | 3 | 1 |
| conformity | strong | 4 | 2 |
| conformity | unanimous | 43 | 11 |
| hedonism | bare_majority | 4 | 2 |
| hedonism | strong | 4 | 2 |
| hedonism | unanimous | 42 | 11 |
| power | bare_majority | 1 | 0 |
| power | unanimous | 49 | 2 |
| security | bare_majority | 5 | 2 |
| security | strong | 4 | 2 |
| security | unanimous | 41 | 22 |
| self_direction | bare_majority | 5 | 3 |
| self_direction | strong | 7 | 5 |
| self_direction | unanimous | 38 | 11 |
| stimulation | bare_majority | 1 | 0 |
| stimulation | strong | 4 | 2 |
| stimulation | unanimous | 45 | 11 |
| tradition | bare_majority | 3 | 0 |
| tradition | unanimous | 47 | 4 |
| universalism | unanimous | 50 | 2 |

## 5. Pass Diagnostics

| Pass | Attempt | Worker model | Raw hash | Score hash | Rationale coverage | Completed |
| --- | --- | --- | --- | --- | --- | --- |
| pass_1 | 1 | gpt-5.4 | 267eb6154669 | a75740c792b6 | 1.000 | 2026-03-20T06:29:21.706506+00:00 |
| pass_2 | 1 | gpt-5.4 | 123cb17954ac | 590aff7ba6f1 | 1.000 | 2026-03-20T06:39:26.980257+00:00 |
| pass_3 | 1 | gpt-5.4 | 4e6f9ceef7d4 | 14f365b6c7a3 | 1.000 | 2026-03-20T06:44:17.765067+00:00 |
| pass_4 | 1 | gpt-5.4 | 2033173ed700 | 745d4a5f338a | 1.000 | 2026-03-20T06:48:07.099730+00:00 |
| pass_5 | 1 | gpt-5.4 | 6c4bb874aa72 | 22d25f0ed474 | 1.000 | 2026-03-20T06:54:21.984819+00:00 |

Pairwise similarity:

| Pass pair | Raw hash match | Score hash match | Identical entry vectors | Differing entry vectors |
| --- | --- | --- | --- | --- |
| pass_1 vs pass_2 | False | False | 22 | 28 |
| pass_1 vs pass_3 | False | False | 25 | 25 |
| pass_1 vs pass_4 | False | False | 20 | 30 |
| pass_1 vs pass_5 | False | False | 23 | 27 |
| pass_2 vs pass_3 | False | False | 28 | 22 |
| pass_2 vs pass_4 | False | False | 32 | 18 |
| pass_2 vs pass_5 | False | False | 27 | 23 |
| pass_3 vs pass_4 | False | False | 29 | 21 |
| pass_3 vs pass_5 | False | False | 36 | 14 |
| pass_4 vs pass_5 | False | False | 29 | 21 |

## 6. Hard-Dimension Gate

- Aggregate consensus-vs-human kappa: `0.636`
- Aggregate persisted-vs-human kappa: `0.651`
- Agreement gate passed: `False`
- Confidence gate passed: `True`
- Overall retrain gate passed: `False`

| Dimension | Non-neutral labels | Bare/no-majority labels | Low-confidence ratio | Passes |
| --- | --- | --- | --- | --- |
| security | 26 | 2 | 0.077 | True |
| hedonism | 15 | 2 | 0.133 | True |
| stimulation | 13 | 0 | 0.000 | True |

## 7. Hard-Dimension Deep Dive

### security

Persisted-vs-consensus kappa: `0.877`; consensus-vs-human kappa: `0.534`; low-confidence non-neutral labels: `2/26`.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 11 |
| -1 | 0 | 2 |
| 0 | 0 | 20 |
| 1 | 0 | 2 |
| 1 | 1 | 15 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 5 | 2 |
| strong | 4 | 2 |
| unanimous | 41 | 22 |

### hedonism

Persisted-vs-consensus kappa: `0.839`; consensus-vs-human kappa: `0.522`; low-confidence non-neutral labels: `2/15`.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 8 |
| -1 | 0 | 1 |
| 0 | 0 | 32 |
| 1 | -1 | 1 |
| 1 | 0 | 2 |
| 1 | 1 | 6 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 4 | 2 |
| strong | 4 | 2 |
| unanimous | 42 | 11 |

### stimulation

Persisted-vs-consensus kappa: `0.790`; consensus-vs-human kappa: `0.747`; low-confidence non-neutral labels: `0/13`.

| Persisted | Consensus | Count |
| --- | --- | --- |
| -1 | -1 | 5 |
| -1 | 0 | 2 |
| 0 | 0 | 32 |
| 1 | 0 | 3 |
| 1 | 1 | 8 |

Confidence breakdown:

| Tier | Entries | Non-neutral entries |
| --- | --- | --- |
| bare_majority | 1 | 0 |
| strong | 4 | 2 |
| unanimous | 45 | 11 |


## 8. Rationale Source Summary

- Entries with a perfect 10/10 rationale-source match: `49`
- Entries using fallback rationale selection: `1`
- Maximum label mismatches on a chosen rationale source: `1`

| Source pass | Mismatch count | Entries |
| --- | --- | --- |
| 1 | 0 | 26 |
| 2 | 0 | 17 |
| 3 | 0 | 4 |
| 3 | 1 | 1 |
| 4 | 0 | 2 |

## 9. Label Migration Summary

| Dimension | Persisted | Consensus | Changed entries |
| --- | --- | --- | --- |
| achievement | 0 | 1 | 2 |
| achievement | 1 | -1 | 1 |
| achievement | 1 | 0 | 3 |
| benevolence | -1 | 0 | 2 |
| benevolence | 0 | 1 | 3 |
| benevolence | 1 | 0 | 2 |
| conformity | -1 | 0 | 2 |
| hedonism | -1 | 0 | 1 |
| hedonism | 1 | -1 | 1 |
| hedonism | 1 | 0 | 2 |
| power | -1 | 0 | 1 |
| security | -1 | 0 | 2 |
| security | 1 | 0 | 2 |
| self_direction | -1 | 1 | 1 |
| self_direction | 0 | -1 | 1 |
| self_direction | 0 | 1 | 2 |
| self_direction | 1 | 0 | 2 |
| stimulation | -1 | 0 | 2 |
| stimulation | 1 | 0 | 3 |
| tradition | -1 | 0 | 1 |
| tradition | 1 | 0 | 4 |

## 10. Recommendation

Stop after repeated-call diagnostics review; do not retrain until the gate is addressed.
