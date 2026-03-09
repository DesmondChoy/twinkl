# twinkl-691.2 Label QA

Reviewed after wrangling + judge-label generation on the regenerated 12-persona targeted batch.

## Verdict

- Reviewed samples: 12
- Keep: 11
- Ambiguous: 1
- Bad label: 0
- Gate result: pass
- Security label counts in new batch: `-1 = 8`, `+1 = 24`

## Reviewed Samples

| sample_family | persona_id | t_index | date | reviewed_signal | outcome | note |
|---|---|---:|---|---|---|---|
| quiet_hedonism_positive | 344ea8f7 | 1 | 2025-11-11 | `hedonism=+1` | keep | Quiet restorative pleasure is explicit: he uses the free hour for a stroopwafel and people-watching and says he feels better than he has all week. |
| quiet_hedonism_positive | b8a23604 | 0 | 2025-10-27 | `hedonism=+1` | keep | Strong intended signal: phone off, slow cooking, reading, bodily need for rest, and explicit rejection of the idea that this day was “wasted.” |
| quiet_hedonism_positive | e209b4cd | 5 | 2025-11-05 | `hedonism=+1` | keep | The response makes the boundary clear: she stops work and goes to bed because the day was already full and she “didn’t owe it anything else.” |
| security_positive_ambiguous | 5b836f2b | 0 | 2025-07-03 | `security=+1` | keep | Positive Security is correctly carried by lease structure, insurance, rent reliability, and the fear of an arrangement collapsing. |
| security_positive_ambiguous | 61d7d490 | 4 | 2025-06-05 | `security=+1` | ambiguous | Defensible as stability/operational order (“everyone got paid, tomorrow is prepared”), but it sits near the edge of a neutral end-of-week routine entry. |
| security_negative_tradeoff | 43f4507b | 0 | 2025-01-02 | `security=-1` | keep | Clear hard-negative Security: she rejects salary, housing, and school-fee stability to keep the uncertain PhD path. |
| security_negative_tradeoff | 43f4507b | 3 | 2025-01-09 | `security=-1` | keep | Another clean negative-tradeoff case: permanent government role and pension are explicitly refused in favor of uncertainty. |
| security_negative_tradeoff | abf1ce49 | 2 | 2025-09-14 | `security=-1` | keep | Thin but solid hard-negative example: she names salary and loan relief as the safer option and still cannot make herself take it. |
| self_direction_security_conflict | 61d7d490 | 1 | 2025-05-26 | `self_direction=+1` | keep | Good family fit: the recruiter offer is rejected because leaving the business would mean surrendering a path he chose and built himself. |
| self_direction_security_conflict | 61d7d490 | 5 | 2025-06-09 | `self_direction=+1`, `security=+1` | keep | Strong intended mixed signal: he wants a safety net, but only if it remains his rather than a corporation’s. |
| stimulation_security_conflict | 3a517a79 | 0 | 2025-12-22 | `stimulation=-1`, `security=+1` | keep | The California travel-nursing fantasy is legible, but she closes the tab because giving up the familiar life she already knows feels too threatening. |
| stimulation_security_conflict | 849a9d72 | 3 | 2025-06-26 | `stimulation=-1`, `security=+1` | keep | Clear conflict-pair case: the promotion-track move is treated as an unnecessary gamble against what already works. |

## Summary

The regenerated `twinkl-691.2` batch passes the manual label QA gate. The
reviewed samples cover the intended family mix, and the target `Security`
polarity gate is satisfied with materially better negative coverage than the
rolled-back batch. One `Security=+1` sample was borderline because it could be
read as orderly routine rather than strong stability-seeking, but no reviewed
sample rose to the level of a bad label or regeneration trigger.
