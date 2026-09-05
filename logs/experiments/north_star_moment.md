# North Star Moment Results Log

North Star Moment (NSM) remains blocked at the Phase 0B feasibility gate under
`twinkl-fz34`. Runner hardening is complete; semantic selection quality has not
been retested. Experience and Coach Digest retain their
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

First separate retrieval misses from false selections using the frozen
development references. If retrieval remains limiting, the proposed sequence
is another embedding model, then BM25 or hybrid retrieval, with bounded agentic
search considered if simpler methods still miss valid examples. These
approaches have no NSM results yet.
Any retrieval comparison must preserve separate checks for semantic selection
and omission, record its development protocol, and leave the reserved histories
unused until the final implementation is frozen for evaluation.
