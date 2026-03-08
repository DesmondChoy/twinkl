# twinkl-681.5 Label QA

Reviewed after wrangling + judge consolidation on the accepted 12-persona targeted batch.

## Verdict

- Reviewed samples: 8
- Keep: 7
- Ambiguous: 1
- Bad label: 0
- Gate result: pass

## Reviewed Samples

| target_dimension | persona_id | t_index | date | target_label | outcome | note |
|---|---|---:|---|---:|---|---|
| Power | 236d4507 | 0 | 2025-02-14 | -1 | keep | Clear public credit loss plus deliberate silence; `power=-1` is the primary read. |
| Power | 9510e81a | 0 | 2025-11-11 | -1 | keep | Mild but genuine status/voice diminishment; the target label matches the entry. |
| Power | b604addd | 1 | 2025-10-20 | -1 | keep | Dependency on Hassan and felt loss of control make the `power=-1` label defensible. |
| Power | bcfa309d | 1 | 2025-09-18 | -1 | ambiguous | Borderline case: the response rejects status positioning, but a stricter sparse read could leave `power=0`. |
| Security | c876219a | 1 | 2025-04-08 | -1 | keep | Strong mild-misalignment example: he destabilizes a working routine for speculative extra pay. |
| Security | e6838e16 | 3 | 2025-03-02 | -1 | keep | Small-budget lapse framed as erosion of order; good hard-boundary `security=-1` sample. |
| Security | eb969aef | 3 | 2026-01-03 | -1 | keep | Postponing the insurance comparison clearly weakens a buffer he values. |
| Security | 3b8b0795 | 2 | 2025-04-19 | 0 | keep | Routine fence repair / social time is appropriately neutral for the target dimension. |

## Summary

The reviewed `Power` and `Security` entries are mostly the intended mild `0/-1`
boundary cases rather than obvious melodramatic negatives. One `Power` sample
was borderline but still defensible; the batch does not trigger regeneration.
