## North Star Moment — independent review (read-only)

Read: `/tmp/twinkl-nsm-opus-review/diff.patch` (full), `src/north_star/{runtime,provider,assessment,review,input_budget}.py`, `scripts/experiments/north_star_integration.py`, `src/demo/{experience_service,scenarios,contracts}.py`, `frontend/onboarding/src/{NorthStarMoment,northStar,useNorthStarReview,WeeklyExperience,ReplayTimeline,JournalExperience,session,domain}.*`, all five new/changed test files, `config/evals/north_star_integration_v1.json`. No edits, no tracker writes, no model calls, no `.env` reads.

Seven defects and one newly load-bearing baseline problem. Nothing found that breaks the approved selection semantics.

---

### D1 — Live demo traffic writes into the frozen experiment report directory (Medium)

`src/north_star/runtime.py:41,517-518` defaults `ledger_path` / `counts_path` to `logs/experiments/reports/north_star_integration_20260906/{budget.json,input-counts.json}`, and `src/demo/experience_service.py:206` constructs `OpenAINorthStarRuntime()` with no overrides. Every live NSM review in the deployed Experience therefore appends attempts to the same ledger and counts file the experiment run uses.

Consequence: the run's `new_spent_or_reserved_usd` / `cumulative_spent_or_reserved_usd` (`north_star_integration.py:505-522`) and `input-counts.json` become a mix of experiment and demo spend, so the recorded evidence is no longer reproducible from the manifest. It also means demo users draw down the same US$20 cumulative envelope as the reserved evaluation.

Minimal fix: pass distinct live paths at the service construction site (e.g. `OpenAINorthStarRuntime(ledger_path=Path("logs/runtime/north_star/budget.json"), counts_path=.../input-counts.json)`), leaving `DEFAULT_DIRECTORY` for the experiment only.

### D2 — Full prompts and raw provider text are published to the browser and localStorage (Medium)

`runtime.py:617` appends the live `ProviderAttempt` (with `raw_text`) into `record.reviews[].provider_attempts`, and `NorthStarValueReview` also carries `provider_request` (system prompt + full prompt + JSON schema) and a second full copy of `sources`. The whole `NorthStarRecord` becomes `NorthStarReviewedDetails.record`, `read_trace` returns events unredacted (`experience_service.py:2197-2203`), and `session.ts:546` persists `session.experience.trace_events` to localStorage.

Two consequences:
- The stated privacy boundary is narrower than described. `_PrivateResponseLedger` (`runtime.py:480-499`) keeps raw text off the *disk ledger*, but the same text is sent to the client and written to localStorage, where `forget()` cannot reach it. `delete_session` clears server memory only.
- Size. Per week per Core Value the record carries the source text three times (`record.sources`, `reviews[].sources`, `provider_request.prompt`) plus the ~5 KB system prompt and the schema. For the saved replays this is 36 weeks × up to 2 values baked into `frontend/onboarding/public/scenarios/*.json` (static assets), and for live sessions it accumulates in a single localStorage key whose write fails soft (`session.ts:544-557`) — a quota failure silently disables the restore path that `useNorthStarReview.ts:60-75` depends on.

Minimal fix: `reviews[].sources` is always identical to `record.sources` (set together at `runtime.py:571` and `runtime.py:664`), and `validate_north_star_record` only ever compares `provider_request` for equality (`runtime.py:748`). Replace both with `provider_request_hash: str` and drop `reviews[].sources`, re-deriving the full request from `source_review_requests(request)` at validation time. That removes most of the payload without weakening any check.

### D3 — Saved replays discard every nudge response as evidence (Medium)

`src/demo/scenarios.py:1190` hard-codes `response_available_at=None` in `build_saved_north_star_request`, so `build_north_star_request` nulls `nudge_response` for all sources. `north_star_integration.py:131-134` then *hard-fails* preparation if any response survives.

Consequence: for all five Personas across all 36 weeks, roughly the writing produced by a 0.7 nudge-response rate is excluded from the evidence pool, and the `quote_source == "nudge_response"` path — which the live path does support and `tests/north_star/test_runtime.py:262-277` covers — is never exercised by the saved evaluation. The saved-replay evidence therefore does not represent the live selection surface, and this is asserted as intended by `tests/demo/test_north_star_scenarios.py:56-57` rather than flagged.

The availability timestamp does exist: `nudge_generated` events are emitted with `started_at = _simulated_at(date, hour=12, sequence=index+300)` (`scenarios.py:835-841`), strictly after the entry's `hour=12, sequence=index`.

Minimal fix: build a `{journal_entry_id: nudge_generated.started_at}` map alongside the existing `availability` map (`scenarios.py:1173-1174`) and use it for `response_available_at` when the entry has a response; drop the `prepare()` guard. If you prefer not to change the frozen evaluation now, state the exclusion explicitly in the report scope rather than leaving it implicit.

### D4 — `measure_requests` loses receipts under concurrency (Low-Medium)

`src/north_star/input_budget.py:104` reads the counts file, then `:131` awaits the provider between read and `:147` write, with no lock. `north_star_integration.py:372` gathers 36 `prepare_record` tasks (semaphore 3), each calling `runtime()` → `count_requests`, and concurrent live sessions do the same.

Consequence: last-writer-wins drops earlier receipts. The in-flight call still succeeds (the returned in-memory `state` holds the receipt), so no incorrect NSM result — but `input-counts.json` is a provenance artifact and becomes non-deterministic, and later runs re-issue count calls.

Minimal fix: wrap the read/write span in an `asyncio.Lock` held by the caller, or reuse `BudgetLedger.transact`'s flock pattern for the counts file.

### D5 — Blocking file locks and fsync on the live request path (Low-Medium)

`BudgetLedger.transact` (`provider.py:199-254`) does `fcntl.flock(LOCK_EX)`, `os.fsync`, and a directory fsync synchronously; it is reached from `runtime.py:604 provider.complete(...)` inside the asyncio event loop, as is `_write_counts`. Within one process this cannot deadlock (`transact` never awaits), but a second process holding the lock, or a slow/network filesystem, blocks the entire demo server, not just the NSM task.

Minimal fix: `await asyncio.to_thread(...)` around `ledger.reserve` / `ledger.finish` in `BudgetedProvider._complete`, and around `_write_counts`.

### D6 — A retry click can be silently consumed by an inflight-map race (Low)

`experience_service.py` `_execute_north_star` publishes the result inside `async with self._lock` and pops `_north_star_inflight` in its `finally`, which runs *after* the lock is released. In that window a `retry=True` request whose `matching` record is failed-and-retryable skips the early return, finds the already-finished task via `_north_star_inflight.get(key)`, and awaits it — returning the same failed record. The frontend still counted the attempt (`useNorthStarReview.ts:49`), so `retryable` drops to false and the user gets no second chance.

Minimal fix: move `self._north_star_inflight.pop(key, None)` inside the same `async with self._lock` block that writes `events[index]`.

### D7 — Recoverable server errors render as a permanent failure (Low)

All four `_error` calls in `review_north_star` use the default `retryable=False`. `north_star_not_ready` ("Complete this week's review first.") and the post-await `session_conflict` therefore surface through `useNorthStarReview.ts:92-95` as `failed` with `retryable: false`, and `WeeklyExperience.tsx` shows "A North Star Moment could not be prepared." with no retry button — for what is a transient ordering condition, not a terminal outcome.

Minimal fix: pass `retryable=True` for `north_star_not_ready`.

---

### B1 — Baseline debt now on the live user path: whole-batch rejection on ordinary English words (Medium impact)

`src/north_star/review.py:55-60` builds `_INTERNAL_LABEL_PATTERN` from `SCHWARTZ_VALUE_ORDER`, which includes `power`, `security`, `tradition`, `achievement`, `conformity`, `stimulation`. `validate_review:309-312` accumulates `internal_value_label_in_quote` into `errors` and then rejects the **entire batch** (`:311`), not just the offending result.

Trigger: a user writes "I finally set aside money for some job security" and the model quotes it. Consequence in the integration: `runtime.py:629` invalidates the attempt, retries once, the model very likely returns the same quote, and the week ends at `status="failed"` with both paid attempts spent. This code is pre-existing and outside the diff, but this change is the first thing to run it against real user writing, so the risk is newly material.

Minimal fix: in `validate_review`, treat `internal_value_label_in_quote` as a per-result disqualification (demote that entry to non-selectable) instead of adding it to the batch-fatal `errors` list; keep the other checks batch-fatal.

---

### Verified correct (checked, not findings)

- `profile_reference` (`runtime.py:47-62`) and `northStarProfileRef` (`northStar.ts:55-63`) agree: recursive key sort, integral-float normalisation, and the `preferred_name ?? null` shim line up with `OnboardingProfile.validate_preferred_name`, which `domain.ts:436-437 normalizePreferredName` already applies client-side. (Only a literal U+FEFF in a name would diverge.)
- `_pick` (`runtime.py:443-459`) matches the approved rules: Active Drift → longest `current_run_length`, ties by Profile order via `max`'s first-maximum, pre-onset sources only, no other-value fallback; no Active Drift → current-week across values first, then newest by Profile order with `reminder` framing. `_context` excludes post-onset sources by t_index, date **and** `available_at`.
- `results[source.entry_id]` in `_pick` cannot `KeyError`: `validate_review:292-295` proves exact request membership.
- `insufficient_evidence` returns `not_eligible` before any provider call (`runtime.py:290-291`), and the frontend independently refuses to render it (`northStar.ts:152,154`).
- Invalidation reparenting in `_apply_resume_update:563-571` resolves transitively and preserves the trace chain.
- `attach_saved_north_star` reconstruction from `week["event_ids"]` is lossless — `profile_event_id` is folded into week 1 (`scenarios.py:744`), so no event is dropped.
- Replay projection leaks no future events; `scenarioReplay.test.ts` asserts this per week per Persona.
- The `demoContracts.ts` `exactKeys` list for the record matches all 24 `NorthStarRecord` fields.

### Residual validation limits

- I did not execute anything. The "scoped tests/Ruff/MyPy pass" and "2 known preexisting July manifest/config-hash failures" claims are taken as reported.
- `report.json` and `records.json` do not exist yet, so `attach_saved_north_star` / `_validate_fixture_semantics` were only reviewed against the pending-record fixtures built by `tests/demo/test_north_star_scenarios.py:79-96`, not against real Luna output. Selection quality, `metrics_by_mode`, and the actual `cumulative_spent_or_reserved_usd` are unverified by construction.
- No test covers the D6 race window, the D2 payload size, or a real `not_eligible → inputs_changed` publication reaching the browser; the lifecycle tests use `ControlledRuntime`, which never returns a `complete` record with a selection, so the end-to-end live path from a real provider response to a rendered card is untested.
- Live behaviour under a real `OPENAI_API_KEY` (ledger contention, count-call latency, D1 contamination) is inferred from code, not observed.
