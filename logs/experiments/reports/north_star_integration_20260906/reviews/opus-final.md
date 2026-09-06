## Conclusion

No reproducible safety or correctness regression in this fix. The gate is strictly narrower than the check it enables, and the browser's NSM data is discarded rather than trusted.

**Why it holds** (`experience_service.py:431-450`, `725-730`):

- Condition 1 pins the entire non-NSM history by full Pydantic model equality in order, so the swap cannot launder any writing, nudge, review, digest, coach, or clock event.
- Condition 2 requires every browser-held NSM event to carry a server-known `event_id` *and* a matching `input_hash`, so a fabricated or re-pointed moment is rejected before the swap.
- After the swap the browser's NSM objects are discarded entirely (`update={"trace_events": list(server_events)}`). Their only influence is on whether the swap is permitted. A tampered `details.record` that preserves `event_id`/`input_hash` therefore has no effect — and cannot even reach the helper, since `NorthStarReviewedEvent.validate_record_owner` forces `input_hash == details.record.input_hash` at the API boundary.
- Ordering is correct: the swap precedes `_resume_update_date` (line 731), which reads only `revision`, `journal_entries`, `nudges`, and `assessment_clock`. Removal and nudge classification are unaffected.
- The subsequent strict `resume_state.trace_events != self._events[...]` checks (736-739, 768-769) become trivially true *only* when both conditions already passed. That is the intended reduction, not a bypass.
- `list(server_events)` is a fresh list; `_apply_resume_update` rebuilds via `model_copy` and never mutates the server's event objects, so the concurrent `_execute_north_star` path (which indexes into `self._events[...]`) is not aliased.

**In-flight interaction checks out.** With a pending NSM event removed by `_apply_resume_update`, the still-running task recomputes `input_hash` under the lock, finds `still_current` false and `index` None, calls `forget(record)` to release raw text, and returns without publishing. The pending record it discarded has `reviews == []`, so the earlier `forget` in `_apply_resume_update` is a harmless no-op. `test_browser_can_remove_before_receiving_independent_nsm_trace` exercises both orderings and asserts no NSM event survives.

**One residual, safe direction.** If the browser holds an NSM event the server has already dropped, condition 2 fails and the request still returns 409 rather than resurrecting a server-deleted record. That is the correct conservative outcome; the `session_not_found` restore path in `useNorthStarReview` does not reach this branch, and other resume callers re-read the trace after each mutation. Not actionable.

**Test coverage note, not a defect.** The unit test covers injection and forgery but not "browser holds an event the server dropped" or "non-NSM mismatch while NSM events are present." Both fail closed by construction, so the gap is in assertion coverage, not behaviour.

## Disposition of F1–F3

- **F1 — closed.** Resolved by the planned finalization rather than a code change: `report.json` now carries `generation_attempts` and both cost fields, so `_SeededLiveLedger._seed` succeeds, and the five bundles plus catalog were re-exported against the final report hash. Both halves of the finding (live NSM dead, bundles scheduled to go stale) are discharged. The lifecycle coupling itself remains as accepted design; I am not re-raising it.
- **F2 — withdrawn, declined by you.** Recorded as a deliberate contract decision: optional failed records keep a visible bounded recovery path, and provider/budget errors are not silently recategorized as semantic omissions. I accept the reasoning — recategorizing them would make an environment failure indistinguishable from an evidence-based omission, which is the weaker of the two behaviours for a system whose central claim is source-bound honesty.
- **F3 — closed as documented.** Docs now preserve both ledgers, prohibit deleting budget history, and require reconciliation before a successor budget. That is the right resolution for what was only ever a runbook gap.
