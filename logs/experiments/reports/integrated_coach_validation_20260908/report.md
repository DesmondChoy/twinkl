# Integrated Coach Digest validation

The 8 September 2026 implementation for `twinkl-rklc.38` presents one Coach Digest reading experience in both frontend routes. Its original weekly mirror and tension explanation precede an optional independently validated North Star Moment passage; the original reflective question remains last. The application inserts fixed framing and the exact selected quotation without another model rewrite. Onboarding uses plain headings, while Persona replay identifies the added passage and links its exact backend event in Inspect.

All 27 saved weeks now contain a source-compatible Coach Digest. The [completion report](../demo_coach_all_weeks_20260908/report.md) records the 22 new responses, four bounded validation retries, and preservation of the five existing responses. Every original numbered trace event retains its identifier. Additional Coach events use named week identifiers, preserving the source event IDs in immutable generation receipts. All existing North Star Moment records remain unchanged.

The engineering checks passed. AI editorial review of the assembled text found five content concerns, including one direct contradiction between the two components. Those findings remain open in `twinkl-rklc.39`; passing code checks does not establish that every combined reflection is coherent or semantically suitable.

## Verification performed

The frontend suite passed **255 tests across 19 files**, plus TypeScript checking and the production build. Relevant regressions cover the shared narrative/quotation/question order, one final question, route-specific attribution, missing or invalid Coach suppression, pending/failed/ineligible NSM omission, source/Profile/week changes, rebuilt weekly results, stale-result fallback, quotation expansion, exact Inspect navigation, and nudge-response source drawer focus.

The Python verification covered **417 distinct passing tests**: 369 North Star Moment and demo tests excluding the two scenario files, plus 48 scenario and Coach-generation tests. After the event-ID correction, the 33 affected scenario tests passed again. Ruff passed for changed implementation and tests. Scoped MyPy passed for the changed runtime, live wrapper, scenario exporter and completion runner. Ordinary MyPy also traversed an existing imported error at `prompts/__init__.py:44` (`no-any-return`); this report does not claim a clean repository-wide type check. The production build retains its existing large-chunk warning. A whole-repository pytest run was not performed.

Browser checks used Chrome against local Vite and the explicitly controlled `scripts.demo_north_star_qc` backend. They covered the saved Persona and from-scratch onboarding routes, a new completed replay week, default desktop layout, a 390 px manual Experience, and a 320 px replay. The narrow replay had document/body width 305 px within a 320 px viewport, with no horizontal page overflow. Both routes retain normal page scrolling.

In Persona replay, the quotation opened its original Journal Entry; Escape returned focus to the exact source link. **Inspect this moment** expanded the matching `north_star_reviewed` event and showed the deterministic introduction, exact quotation, eligibility outcome and source/AI checks. Returning to Experience preserved the week and reopened Journal Entries as specified. In a separate synthetic onboarding session, an 11-group assessment produced a confirmed Profile, a Journal Entry was saved and its week closed, and the unbranded passage appeared before the original question. A long quotation expanded fully. A second controlled week verified that the base Coach Digest remained visible while NSM was held pending and after a controlled failure. These browser provider responses were test doubles and incurred no paid calls. The controlled backend was stopped after verification; the normal `src.demo.api:app` backend was started on port 8000 and returned a successful health check. Temporary viewport overrides were reset.

## Real provider smoke and cost

The [smoke script](live_smoke.py) invoked the actual `LiveNorthStarRuntime` through the in-memory Experience service on synthetic writing about helping a sister with groceries and dinner. Nudge, Weekly Drift Reviewer, and Coach providers were deterministic test doubles; only North Star Moment token counting and generation used the real OpenAI provider. This is a real NSM service-boundary smoke, not an all-provider end-to-end run or human validation.

The live run completed with a supportive action selected, an exact quotation present in the submitted source, a live event, actual provider response ID and token usage, and an unchanged weekly Coach Digest. Repeating the same service request reused the existing event and added zero attempts. [Request](live_request.json), [event and raw review](live_event.json), and [result](live_smoke_result.json) preserve the evidence.

| Work | New generation calls | Calculated cost, USD |
| --- | ---: | ---: |
| 22 missing Coach Digests, including validation retries | 26 | 0.01558455 |
| Live North Star Moment smoke | 1 | 0.00049740 |
| Total this task | 27 | 0.01608195 |

These calculations use the repository's configured rates and recorded usage, not a billing export or a new pricing verification. NSM input counting is recorded separately from generation. No unmetered generation attempt remains in this run.

The new pinned live policy permits a conservative US$1 total allowance with zero prior spend, two attempts per exact request, no SDK retries, a 16,000-token input limit and US$0.25 per-attempt ceiling. One fixed private ledger retains accounting across sessions and restarts. This is a separately authorized allowance; the removed integration budget and report were not recreated. Validation accepts only the two pinned integration/live policy hashes, consistently across a record's value reviews. Existing terminal `budget_unavailable` session records remain terminal: changing budget configuration alone does not retry them. The smoke used a fresh synthetic session.

Reproduce the real-provider smoke only within authorized paid scope:

```sh
source .venv/bin/activate
PYTHONPATH=. UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run --no-sync python logs/experiments/reports/integrated_coach_validation_20260908/live_smoke.py
```

## AI editorial findings

All 27 original narratives, selected quotations or omissions, and final questions were read in the [composite review artifact](../demo_coach_all_weeks_20260908/composite_review.md), using the actual mode-specific introduction in `frontend/onboarding/src/northStar.ts`. A second AI reviewer independently corroborated the flagged cases. These are qualitative editorial observations, not fresh evaluation scores, an exhaustive semantic audit, or human validation.

| Persona and week | Observation | Evidence boundary |
| --- | --- | --- |
| Sook Yin, week 3 | The selected quotation describes eating a pastry over the sink while rinsing a thermos. Encouragement sits awkwardly beside the Coach's account of enjoyment interrupted by chores; the full entry describes the missed opportunity to savour it. | Existing saved NSM selection; exact source checks pass, but semantic suitability is questionable. |
| Noor, week 1 | “I was up every two hours” describes caregiving and exhaustion. On its own it offers weak support for an introduction about exercising freedom of choice. | Existing saved NSM selection; weak interpretation of the selected quotation. |
| Noor, week 5 | Accepting the logo project before considering capacity is a real choice, but the quotation foregrounds overcommitment while the Coach questions how freely chosen work feels. Generic encouragement therefore sounds awkward. | Existing selection; this is an editorial tension, not proof of Conflict against that Core Value. |
| Henrik, week 1 | Telling Ingrid he would slow down is a statement about intended future conduct. The quotation does not show that he followed through. | Existing saved NSM selection; promise versus completed supportive action needs review. |
| Sook Yin, week 4 | The new Coach says reflexology remained undecided, while NSM quotes that she went. | Direct cross-component contradiction. The Coach input contains earlier hesitation and a later selected excerpt about worrying after a nice afternoon, but omits the later entry's completed reflexology action. NSM sees the full entry. Both input hashes are correct. |

The current implementation preserves all raw outputs as agreed. `twinkl-rklc.39` records the required source-context and semantic-review follow-up before these cases can be presented as resolved. Fixing them may require changing the Coach's evidence context and reassessing affected selections; changing layout or rewriting quoted words would not establish the missing evidence.

No commit, push, deployment, remote Beads synchronization, or new semantic NSM experiment was performed.
