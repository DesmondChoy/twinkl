## Continued review — new live runtime and fixes

Two actionable items remain, both from the same root cause. Everything else in this round checks out.

---

### F1 — `report.json` carries three roles with conflicting lifecycles (Material: current + scheduled)

The shipped artifact has only `cases`, `evaluation_status: "in_progress"`, `manifest_hash`, `schema_version`, `scope` (`logs/experiments/reports/north_star_integration_20260906/report.json:14988-14991`). It does **not** contain `generation_attempts`, `new_spent_or_reserved_usd`, or `cumulative_spent_or_reserved_usd`.

`_SeededLiveLedger._seed` requires all three (`live_runtime.py:97-107`), so it raises `BudgetError("Integration budget report is not finalized")` on every seed. `live_runtime.py:190-197` converts that to `status="failed", reason="budget_unavailable"` and leaves `retryable` at its `False` default. `_north_star_event` maps that to event status `failed`; `useNorthStarReview.ts:122-125` yields `failed=true, retryable=false`; `WeeklyExperience.tsx` then renders "Your weekly response is ready. A North Star Moment could not be prepared." with no retry control.

Current consequence: in manual mode, every closed week shows a permanent failure banner. No spend occurs — `live_runtime.py:168` seeds before counting or generating, which I verified. Saved replay is unaffected.

Scheduled consequence: `north_star_integration.run()` (`scripts/experiments/north_star_integration.py:513-525`) always writes those three fields and never writes `evaluation_status`, so finalizing will satisfy `_seed`. But it rewrites the whole file, changing `_sha256_file(report.json)` (`scenarios.py:278-283`), which is bound into every bundle's `manifest.input_hash` by both `attach_saved_north_star` and `_validate_fixture_semantics`. The five committed `public/scenarios/*.json` will then fail `load_scenario_file` (`scenarios.py:1449`) with "Scenario input hash differs from frozen sources" until re-exported. The xhigh evaluation labels have nothing to do with the saved records, yet they invalidate the bundles.

Minimal fix, one change for both halves: bind bundle provenance and live-budget finalization to a records-only artifact rather than the evaluation report. `records.json` already holds the 36 runtime records (`north_star_integration.py:365-366`). Point `NORTH_STAR_REPORT_PATH` at it and have `_seed` read finalization from a small sibling written when the runtime records complete, so an in-progress evaluation cannot invalidate either.

If you keep the current coupling instead: add the three finalization fields to the interim report now so live NSM is not dead, and schedule a bundle re-export as part of finalizing.

### F2 — Environment-level unavailability renders as a user-facing failure rather than an omission (Medium)

`runtime.py:552-554` (`provider_unavailable`, retryable `True`) and `live_runtime.py:190-197` (`budget_unavailable`, retryable `False`) both produce `status="failed"`, which the weekly panel surfaces as a banner. On a deployment without `OPENAI_API_KEY`, or in the current unfinalized state, that is the steady state, not a transient error: the retry button is bounded at two attempts and then disappears while the banner remains.

The card is optional and the approved rule is that Insufficient Evidence omits. Environment unavailability is closer to omission than to telling the user their moment failed. Minimal fix: treat `reason in {"provider_unavailable", "budget_unavailable"}` as a silent omission in the `failed` predicate (`useNorthStarReview.ts:122`), keeping the failed record in Inspect for diagnosis. One predicate, no contract change.

### F3 — Note only

`_SeededLiveLedger.transact:126-127` fails closed permanently if the source ledger ever changes after seeding, and the only recovery is deleting `logs/exports/demo_tool_runs/north_star/budget.json` by hand. Correct behaviour; worth a runbook line. Not code-actionable while F1 blocks seeding.

---

### Verified in the new code (not findings)

- The no-I/O fallback claim at `live_runtime.py:180-182` holds. Empty `source_review_requests` → `_context` reason ≠ `eligible` → `not_eligible` before provider construction; missing key → `runtime.py:552` returns before `BudgetedProvider` is built. Neither touches ledger or counts; `test_live_runtime.py:89-109` asserts `list(tmp_path.iterdir()) == []`.
- One worker with `asyncio.run` per call is safe: `BudgetedProvider.run()`'s `finally` drains `_inflight` before `asyncio.run` returns, so no closed-loop task survives between calls.
- `_measure_locked` holds the exclusive flock across the awaited count call on the worker only; `test_live_runtime.py:170-183` confirms the app loop stays responsive under an externally held source lock.
- The new hook response binding is safe: `_append_session` defaults `increment_revision=False` and `review_north_star` does not pass it (`experience_service.py:977, 2142`), while line 1013 mutates `self._sessions` — so `response.session.revision === expectedRevision` holds and `useNorthStarReview.ts:79-82` will not spuriously conflict.
- The `settled` flag (`useNorthStarReview.ts:54, 95, 105, 108`) refunds the attempt only on discarded-result paths, and server-side coalescing keyed on `(session_id, input_hash)` prevents a duplicate paid call on the re-run.
- Deletion race: `northStarReviewEnabled={!deletingSession && deleteError === null}` (`App.tsx:1421`) reaches `enabled`, which `stillCurrent()` reads through `latest.current.enabled`, so an in-flight result is discarded during delete.

### Disposition of prior findings

| | Disposition |
|---|---|
| D1 | Resolved as isolation. Live artifacts moved to `logs/exports/demo_tool_runs/north_star/`, seeded from integration spend, fail-closed on source change; `test_live_files_are_separate_and_seeded_spend_is_preserved` asserts the experiment files are byte-identical after a live run, and `test_separate_live_directory_does_not_reset_authorized_budget` asserts the US$20 envelope is not reset. Residual is F1, which concerns *which file proves finalization*, not isolation. |
| D2 | Withdrawn. Intentional for exact Inspect and resume; 780,192 UTF-8 bytes worst case with five passing round-trip tests and a visible `persistenceError` alert answers the quota concern, and the docs now state browser raw text, which answers the accuracy concern. |
| D3 | Withdrawn, and I agree with the reasoning: `nudge_generated.started_at` is simulated generation evidence, not an independent record that the reply existed. Using it would fabricate availability. `prepare()`'s hard guard correctly enforces the exclusion. |
| D4 | Closed, not actionable. Live path is locked by `_measure_locked`; the experiment path pre-counts all `planned` requests serially before the concurrent `evaluate()` gather, so the happy path has no concurrent write. |
| D5 | Resolved; blocking flock/fsync now run only on the dedicated worker, with a regression test. |
| D6 | Resolved at `experience_service.py:2072` (pop inside the publication lock). The `finally` at 2075 remains for the not-current path, where returning a conflict is the correct outcome. |
| D7 | Resolved; `retryable=True` at `experience_service.py:2105`. |
| B1 | Accepted as recorded residual. Weakening the internal-label protection is a policy change, not a code fix, and per-result demotion would silently narrow an approved guarantee. One consequence to keep in view: the residual now sits on the live user path, so a real quote that trips it fails the week after two attempts — F2's omission treatment would keep that from surfacing as a user-facing error. |
