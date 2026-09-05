# North Star Moment: offline development diagnostics

These diagnostics preserve the failed Phase 0B gate and all frozen AI reference decisions. They make no provider calls.

## Reproduction and provenance

```sh
source .venv/bin/activate
uv run python scripts/experiments/north_star_phase0b_diagnostics.py --output-dir logs/experiments/reports/north_star_phase0b_revision_20260905
```

Execution-freeze and manifest input hashes, the historical report hash in validation.json, development episode membership, source text hashes, retrieval order, review contracts, and saved selections are checked before analysis. diagnostics.json records every input hash.

## Mechanically derived evidence

The 33 development episodes include 28 nonempty histories and 5 structurally empty histories. Seven selections disagree with the primary reference: three wrong_value, two ambiguous, and two same_value_conflict.

| k | Reference-positive histories reached | Source decisions | Source characters |
|---|---:|---:|---:|
| 1 | 7/19 | 28 | 16436 |
| 3 | 17/19 | 74 | 46363 |
| 5 | 18/19 | 104 | 65163 |
| 7 | 19/19 | 124 | 77937 |

At k=3, Noor's second Tradition episode misses entry 7 at rank 7. Lukas's second Self-Direction episode misses entries 4 and 5 at ranks 5 and 7. Top-5 would add 30 source decisions (40.5%) while reaching only one additional reference-positive history. These are reference/ranking calculations, not rerun selection outcomes.

| No-reference-supportive stratum | Histories | Correct omission |
|---|---:|---:|
| all_not_supportive | 4 | 1/4 |
| includes_abstain | 3 | 2/3 |
| all_abstain | 2 | 2/2 |

8 identical source/requested-value groups change reference decision or reason across episodes. One changes supportive status: `87e92805:entry:4` is supportive in Security episode 02 and same_value_conflict in episode 03. Its text and requested phrase/definition are identical; it describes authorizing necessary motorcycle brake repairs despite financial anxiety.

Both invalid OpenAI attempts belong to `dbe2c53d:universalism:episode_01`. Replaying their saved JSON reproduces malformed_decision:results.0:value_error; both combine abstain with other_actor. diagnostics.json retains the contradictory fields. Invalid historical attempts remain invalid.

## Separate analyst interpretations

These explanations are a new AI analysis of development evidence; they do not replace the frozen reference labels.

| Case | Interpretation |
|---|---|
| `152df7a4:universalism:episode_01` | Explicit paperwork assistance; the Universalism relationship is unclear because the writing emphasizes operational efficiency. |
| `2541429a:tradition:episode_01` | The quotation reflects on a daughter's learning; a completed supportive action by the writer is inferred rather than clearly described. |
| `5fa8b540:universalism:episode_01` | The quotation describes an outcome. The full source explicitly mentions a water-pollution lesson the writer delivered, so source-level abstention remains debatable even though the selected quotation is weak. |
| `66ced716:universalism:episode_01` | Praising a painting and helping place it on a rack establishes action; the broader welfare relationship depends on missing workshop context. |
| `7ff1d0fb:security:episode_01` | A claim that rent is covered supplies no action that secured it. |
| `bf44e50f:hedonism:episode_01` | The full source explicitly interrupts the enjoyment in the selected passage; a whole-source Conflict check is needed. |
| `dbe2c53d:conformity:episode_01` | Deleting confrontational replies and closing a family chat demonstrates restraint. Regret about restraint does not necessarily establish Conflict against Conformity; the frozen reference remains disputed. |

The first revision should retain Nomic top-3 and the model settings, require concise action/value/context assessment, and derive decision from one reason enum. An already-made commitment can be a completed choice even if the event is future; mere intentions remain insufficient. Compare any later retrieval revision at matched k and candidate workload. Grade reviewed and retrieval-only quotations with the same exact-candidate protocol, reporting precision and coverage separately.

## Limitations

- All semantic judgments are AI-derived development references, not human validation; the frozen gate and labels are unchanged.
- No accepted reference example is not proof of absence. Report all-rejected, mixed-abstention, all-abstention, structurally empty and unresolved cases separately.
- Repeated episodes may share identical writing. Episode counts are not independent participants; repeated-source inconsistencies expose reference limits.
- Source-character workloads exclude prompts and provider tokenization; they are not token counts, cost estimates, or observed results at larger k.
- Historical retrieval-only precision grades a source identifier whereas reviewed precision grades a quotation. Their subtraction is not matched quotation-level verification lift or a causal estimate.
- No reserved writing, paid review, embeddings, application integration or browser behavior is evaluated by these diagnostics.
