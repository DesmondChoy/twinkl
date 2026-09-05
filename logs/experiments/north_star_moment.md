# North Star Moment Results Log

North Star Moment (NSM) remains blocked at the Phase 0B feasibility gate under
`twinkl-fz34`. Runner hardening and a separately frozen review revision are
implemented. The revised paid run also failed the semantic gate, despite all
provider attempts completing validly. Experience and Coach Digest retain their
existing behavior. The [NSM specification](../../docs/north_star/north_star_moment.md)
governs implementation; this log records measured results and their limits.

The NSM development group contains 27 synthetic Personas and 33 Drift episodes.
Eight reserved Personas, containing nine Drift episodes, remain unevaluated.
AI review of these synthetic histories is not human validation, and repeated
episodes can share Journal Entries.

## Results at a glance

| Date | Work | Result | Disposition |
|---|---|---|---|
| 2026-09-05 | Phase 0A retrieval | Proxy recall at k=3: 21/22 (95.5%) | Passed; froze k=3 for Phase 0B. |
| 2026-09-05 | Phase 0B AI review | Quotation precision: 12/19 (63.2%); correct omission: 5/9 (55.6%) | Failed; dependent integration stopped. |
| 2026-09-05 | Runner hardening, `twinkl-fz34.9` | 193 tests passed; offline replay reproduced the failed result with zero new provider calls | Issue closed; Phase 0B remains blocked. |
| 2026-09-05 | Development revision, `twinkl-fz34.1` | 11/19 accepted quotations; 7/9 correct omissions; 0/40 new provider failures; 300 NSM tests pass | Semantic gate failed; dependent integration remains blocked. |

## 2026-09-05 — Development revision failed (`twinkl-fz34.1`)

Offline diagnostics distinguish seven selection disagreements from two
task-reference retrieval misses. The revision keeps Nomic top-3 and the provider
settings fixed, adds factual action/value/context assessments, and derives the
decision from one reason code. Exact quotation grading now applies to both
runtime selections and the rank-one Journal Entry baseline. Frozen exhaustive
AI references are reused, with abstentions and contradictory identical-source
judgments disclosed; unresolved references cannot approve selections.

All 33 development episodes and the original evidence are bound in a new
execution freeze. The conservative maximum is US$10.45116090 additional and
US$10.65733875 cumulative, including every allowed retry. The original US$20
total and US$0.25 per-attempt ceilings remained enforced. After an initial
automatic approval-review rejection, the user explicitly approved the external
run. It completed all 33 cases with 28 new OpenAI runtime calls and 12 new
Gemini exact-quotation checks. Original exhaustive references were reused.
Actual new cost was US$0.09887802, and cumulative cost is US$0.30505587 across
101 attempts. Every new attempt passed its contract without a retry.

The result still fails: 11/19 quotations accepted and 7/9 correct omissions.
Retrieval-only exact-quotation precision is 7/28. Coverage is 19/33 for the
reviewer and 28/33 for retrieval alone; the +32.9 percentage-point precision
difference is descriptive rather than causal. Task-reference retrieval recall
remains 17/19 at k=3. Three original selection disagreements were resolved,
four persisted, and four other selections were nonaccepted. One of those four
is the previously accepted contradictory Security reference: applying the
same conservative rule to the original selections also yields 11/19. The raw
12/19-to-11/19 comparison therefore does not establish model regression.

The NSM suite passes 300 tests; Ruff, scoped MyPy, independent review, and
offline no-transport replay pass. The replay reproduces every selection, grade,
attempt, and cost without new calls. Independent checks found zero quotation,
identity, or chronology failures across all 47 displayed quotations from the
two comparison paths; the original 39 report files and eight reserved Persona
hashes are unchanged. See the
[revision report](reports/north_star_phase0b_revision_20260905/README.md),
[diagnostics](reports/north_star_phase0b_revision_20260905/diagnostics.md), and
[validation evidence](reports/north_star_phase0b_revision_20260905/validation.json).

## 2026-09-05 — Runner recovery repaired (`twinkl-fz34.9`)

The v2 runner verifies frozen source and execution records before use and
reconstructs saved outcomes. Exhausted failures remain terminal; pending
reservations with unknown transport outcomes are retained.
Concurrent requests share an active attempt, and paid run directories share
one cumulative budget. These are preparation and recovery repairs; the prompt,
encoder, retrieval settings, and application behavior are unchanged.

Offline replay reproduced all 33 original case statuses and selections, all
61 attempts, the failed gate, and US$0.20617785 in calculated cost. It made
**zero new provider calls**. The original receipts and frozen execution
hashes remained unchanged. The v2 report also distinguishes retrieval-only
source precision from quotation precision and omits the unmatched
verification-lift subtraction.

The combined NSM suite passed **193 tests**. Ruff and scoped MyPy passed;
full-import MyPy retains five existing wrangling/registry errors. No full
repository or application/browser suite was run for these runner changes.
See the [hardening report](reports/north_star_runner_hardening_20260905/README.md)
and [validation record](reports/north_star_runner_hardening_20260905/validation.json).

## 2026-09-05 — Phase 0B failed the development AI-review gate

Retrieval stayed fixed at `nomic-ai/nomic-embed-text-v1.5`, 256 dimensions,
k=3. Runtime review used `gpt-5.6-luna` with reasoning `none`; independent
reference review used `gemini-3.5-flash` with thinking `low` and examined every
eligible earlier Journal Entry. Sources excluded AI-written nudges, responses
without independent availability evidence, biographies, and labeling metadata.

| Measure | Observed | Interpretation |
|---|---:|---|
| Selected quotations accepted by AI reference | 12/19 (63.2%) | Failed zero-incorrect-selection criterion. |
| Correct omission in nonempty histories with no reference-accepted example | 5/9 (55.6%) | Failed 100% omission criterion. |
| Invalid runtime attempts | 2/29 (6.9%) | Exceeded 5% ceiling; inconsistent decision/reason fields. |
| Task-specific retrieval recall at k=3 | 17/19 (89.5%) | Two histories with accepted earlier examples were missed. |
| Selections across development episodes | 19/33 (57.6%) | Experimental coverage, not live Active Drift card coverage. |
| Structurally empty histories omitted | 5/5 | No review calls; separate from semantic omission. |
| Attempts and calculated cost | 61; US$0.20617785 | Includes unsuccessful attempts; none unmetered. |

The seven rejected selections comprised three `wrong_value`, two `ambiguous`,
and two `same_value_conflict` AI-reference decisions. All retained quotations
passed the deterministic text and identity checks. One case exhausted its retry
allowance after two invalid responses. Budget availability did not overcome
the independent precision and omission failures.

The [Phase 0B report](reports/north_star_phase0b_20260905/README.md),
[saved results](reports/north_star_phase0b_20260905/report.json), and
[derived audit](reports/north_star_phase0b_20260905/audit.md) preserve the protocol,
denominators, failure examples, and evidence boundaries. No NSM cards, live
integration, browser QC, or reserved evaluation followed this failed gate.

## 2026-09-05 — Phase 0A passed the local retrieval proxy gate

The query used only the user-facing Core Value phrase and approved definition.
The pinned Nomic encoder used retrieval prefixes and a normalized
256-dimensional representation. No paid calls were made.

| Retrieved Journal Entries | Histories with a persisted positive LLM-Judge VIF Label found | Proxy recall |
|---|---:|---:|
| k=1 | 13/22 | 59.1% |
| k=3 | 21/22 | 95.5% |
| k=5 | 22/22 | 100.0% |

The smallest setting above the 90% proxy threshold was k=3. Positive LLM-Judge
VIF Labels are a retrieval proxy; the later task-specific AI reference accepted
a different set of examples. The two recall measures must remain separate.
See the [Phase 0A report](reports/north_star_phase0_20260905/README.md) and
[retrieval results](reports/north_star_phase0_20260905/retrieval.json).

## Pending development work (`twinkl-fz34.1`)

The revised run is complete and its semantic gate remains failed. A further
development decision must address quotation/value judgments and disputed AI
references before another experiment. Retrieval changes alone do not address
the observed nonaccepted selections; several cases already retrieved a
reference-accepted alternative. If later evidence shows retrieval remains
limiting, the proposed sequence is another embedding
model, then BM25 or hybrid retrieval, with bounded agentic search considered
if simpler methods still miss valid examples. These approaches have no NSM
results yet.
Any retrieval comparison must preserve separate checks for semantic selection
and omission, record its development protocol, and leave the reserved histories
unused until the final implementation is frozen for evaluation.
