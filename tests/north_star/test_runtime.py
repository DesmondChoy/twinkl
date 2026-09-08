"""Source boundaries, deterministic selection, and interrupted receipt recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.drift_detector import detect_drift
from src.north_star import assessment, input_budget
from src.north_star.provider import BudgetedProvider, BudgetLedger, stable_hash
from src.north_star.runtime import (
    INTEGRATION_POLICY_PATH,
    LIVE_POLICY_PATH,
    NorthStarRequest,
    OpenAINorthStarRuntime,
    SourceWriting,
    _PrivateResponseLedger,
    build_north_star_request,
    profile_reference,
    source_review_requests,
    validate_north_star_record,
)
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision


def source(index: int, **updates: Any) -> SourceWriting:
    dates = ["2026-07-01", "2026-07-06", "2026-07-07", "2026-07-08"]
    return SourceWriting.model_validate(
        {
            "owner_id": "user",
            "entry_id": f"entry:{index}",
            "t_index": index,
            "date": dates[index],
            "journal_entry": f"I helped my neighbour on day {index}.",
            "available_at": f"2026-09-0{index + 1}T12:00:00Z",
            **updates,
        }
    )


def request(
    *,
    writing: list[SourceWriting] | None = None,
    values: list[str] | None = None,
    active: dict[str, int] | None = None,
    unavailable: bool = False,
) -> NorthStarRequest:
    values = values or ["benevolence"]
    active = active or {}
    decisions = [
        WeeklyDriftReviewerDecision(
            persona_id="user",
            week_start="2026-07-06",
            week_end="2026-07-12",
            t_index=index,
            date=source(index).date,
            core_value=value,
            verdict="abstain"
            if unavailable
            else ("conflict" if index >= active.get(value, 4) else "not_conflict"),
            review_status="error" if unavailable else "ok",
        )
        for value in values
        for index in range(4)
    ]
    return build_north_star_request(
        session_id="session",
        owner_id="user",
        profile_ref="profile-hash",
        core_values=values,
        week_start="2026-07-06",
        week_end="2026-07-12",
        cutoff_at="2026-09-06T20:00:00Z",
        drift_result=detect_drift(decisions, persona_id="user"),
        writing=writing if writing is not None else [source(i) for i in range(4)],
    )


async def count_requests(requests: list[dict], policy: dict, output: Path) -> dict:
    return {
        "counts": {
            stable_hash(item): {
                "request_hash": stable_hash(item),
                "payload_hash": stable_hash(input_budget.count_payload(item, policy)),
                "model": policy[item["role"]]["model"],
                "input_tokens": 1200,
            }
            for item in requests
        }
    }


class FakeProvider(BudgetedProvider):
    def __init__(
        self,
        tmp_path: Path,
        supportive: set[tuple[str, str]] | None = None,
        invalid: bool = False,
        response_quote: bool = False,
        ledger: BudgetLedger | None = None,
    ):
        super().__init__(
            ledger or BudgetLedger(tmp_path / "budget.json", INTEGRATION_POLICY_PATH)
        )
        self.supportive = supportive or set()
        self.invalid = invalid
        self.response_quote = response_quote
        self.generated = 0

    async def _complete(self, request: dict, *, retry: bool):
        attempt = self.ledger.reserve(request, retry=retry)
        if attempt.reused:
            return attempt
        self.generated += 1
        payload = json.loads(request["prompt"])
        value = payload["core_value"]
        results = []
        for entry in payload["sources"]:
            supportive = (value, entry["entry_id"]) in self.supportive
            quote_source = "nudge_response" if self.response_quote else "journal_entry"
            results.append(
                {
                    "entry_id": entry["entry_id"],
                    "reason_code": "observable_choice" if supportive else "wrong_value",
                    "quote_source": quote_source if supportive else None,
                    "evidence_quote": entry[quote_source] if supportive else "",
                    "action_assessment": "An action is explicitly reported.",
                    "value_assessment": "The action supports the requested definition.",
                    "conflict_assessment": "No opposing action is reported.",
                }
            )
        attempt.raw_text = (
            "invalid"
            if self.invalid
            else json.dumps(
                {
                    "schema_version": assessment.SOURCE_SCHEMA_VERSION,
                    "core_value": value,
                    "results": results,
                }
            )
        )
        attempt.status = "completed"
        attempt.actual_model = attempt.requested_model
        attempt.calculated_cost_usd = 0.001
        return self.ledger.finish(attempt)


def runtime(tmp_path: Path, **kwargs: Any) -> OpenAINorthStarRuntime:
    return OpenAINorthStarRuntime(
        provider=FakeProvider(tmp_path, **kwargs),
        count_requests=count_requests,
        counts_path=tmp_path / "counts.json",
    )


@pytest.mark.asyncio
async def test_current_week_support_has_priority_across_profile_values(tmp_path):
    value = request(values=["benevolence", "security"])
    result = await runtime(
        tmp_path,
        supportive={
            ("benevolence", "entry:0"),
            ("security", "entry:1"),
        },
    )(value)
    assert result.status == "complete"
    assert result.mode == "encouragement"
    assert result.core_value == "security"
    assert result.selected.entry_id == "entry:1"
    assert len(result.reviews) == 2
    assert all(
        review.provider_attempts[-1].reasoning_effort == "low"
        for review in result.reviews
    )


@pytest.mark.asyncio
async def test_profile_order_then_newest_source_resolves_encouragement(tmp_path):
    result = await runtime(
        tmp_path,
        supportive={
            ("benevolence", "entry:1"),
            ("benevolence", "entry:2"),
            ("security", "entry:3"),
        },
    )(request(values=["benevolence", "security"]))
    assert result.core_value == "benevolence"
    assert result.selected.entry_id == "entry:2"


@pytest.mark.asyncio
async def test_old_support_is_historical_reminder(tmp_path):
    result = await runtime(tmp_path, supportive={("benevolence", "entry:0")})(request())
    assert result.mode == "reminder"
    assert result.selected.date == "2026-07-01"


@pytest.mark.asyncio
async def test_not_conflict_is_not_assumed_supportive(tmp_path):
    result = await runtime(tmp_path)(request())
    assert result.status == "complete"
    assert result.reason == "no_supportive_source"
    assert result.selected is None and result.mode is None


@pytest.mark.asyncio
async def test_insufficient_evidence_never_calls_provider(tmp_path):
    runner = runtime(tmp_path)
    result = await runner(request(unavailable=True))
    assert result.status == "not_eligible"
    assert result.reason == "insufficient_evidence"
    assert runner.provider.generated == 0


@pytest.mark.asyncio
async def test_active_drift_uses_longest_run_without_other_value_fallback(tmp_path):
    value = request(
        values=["benevolence", "security"], active={"benevolence": 2, "security": 1}
    )
    result = await runtime(tmp_path, supportive={("benevolence", "entry:0")})(value)
    assert result.selected is None
    assert [review.core_value for review in result.reviews] == ["security"]
    assert result.source_ids == ["entry:0"]
    assert result.onset_t_index == 1


@pytest.mark.asyncio
async def test_active_drift_tie_uses_profile_order_and_exact_earlier_source(tmp_path):
    value = request(
        values=["security", "benevolence"], active={"benevolence": 2, "security": 2}
    )
    result = await runtime(
        tmp_path,
        supportive={
            ("security", "entry:0"),
            ("security", "entry:1"),
        },
    )(value)
    assert result.mode == "reflection"
    assert result.core_value == "security"
    assert result.selected.entry_id == "entry:1"
    assert result.source_ids == ["entry:1", "entry:0"]
    assert result.onset_available_at == source(2).available_at


def test_post_onset_and_unknown_response_availability_are_excluded():
    sources = [source(i) for i in range(4)]
    sources[0] = source(
        0,
        nudge_response="I checked in after the argument.",
        response_available_at="2026-09-03T12:00:00Z",
    )
    sources[1] = source(1, nudge_response="I called my neighbour.")
    value = request(writing=sources, active={"benevolence": 2})
    payload = json.loads(source_review_requests(value)[0]["prompt"])
    assert all(entry["nudge_response"] is None for entry in payload["sources"])


@pytest.mark.asyncio
async def test_pre_onset_response_uses_actual_availability_not_journal_midnight(
    tmp_path,
):
    sources = [source(i) for i in range(4)]
    sources[1] = source(
        1,
        nudge_response="I called my neighbour.",
        response_available_at="2026-09-02T13:00:00Z",
    )
    result = await runtime(
        tmp_path, supportive={("benevolence", "entry:1")}, response_quote=True
    )(request(writing=sources, active={"benevolence": 2}))
    assert result.selected.quote_source == "nudge_response"
    assert result.selected.evidence_quote == "I called my neighbour."


def test_future_writing_and_unavailable_response_do_not_change_request_hash():
    baseline = request()
    sources = [source(i) for i in range(4)]
    sources[0] = source(
        0,
        nudge_response="Future response.",
        response_available_at="2026-09-09T12:00:00Z",
    )
    sources.append(
        SourceWriting(
            owner_id="user",
            entry_id="future",
            t_index=4,
            date="2026-07-09",
            journal_entry="Future writing.",
            available_at="2026-09-09T12:00:00Z",
        )
    )
    assert request(writing=sources).input_hash == baseline.input_hash
    assert "Future" not in request(writing=sources).model_dump_json()


def test_owner_isolation_and_source_state_bind_receipts():
    baseline = request()
    sources = [source(i) for i in range(4)]
    sources.append(source(0, owner_id="other", entry_id="other:0"))
    assert request(writing=sources).input_hash == baseline.input_hash
    sources[0] = source(0, journal_entry="Changed writing.")
    assert request(writing=sources).input_hash != baseline.input_hash
    assert "drift_result" not in json.loads(
        source_review_requests(baseline)[0]["prompt"]
    )


@pytest.mark.asyncio
async def test_over_limit_complete_input_omits_without_truncation_or_generation(
    tmp_path,
):
    async def oversized(requests, policy, output):
        counts = await count_requests(requests, policy, output)
        next(iter(counts["counts"].values()))["input_tokens"] = 16001
        return counts

    runner = runtime(tmp_path)
    runner.count_requests = oversized
    result = await runner(request())
    assert result.status == "failed"
    assert not result.retryable
    assert result.validation_evidence == ["complete_input_exceeds_16000"]
    assert runner.provider.generated == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("private request content"),
        ConnectionError("private request content"),
        APITimeoutError(request=httpx.Request("POST", "https://example.test")),
        APIConnectionError(request=httpx.Request("POST", "https://example.test")),
        *[
            APIStatusError(
                "private request content",
                response=httpx.Response(
                    status, request=httpx.Request("POST", "https://example.test")
                ),
                body=None,
            )
            for status in (408, 429, 500, 502, 503, 504)
        ],
    ],
)
async def test_transient_count_failure_allows_explicit_retry(tmp_path, error):
    async def unavailable(*args):
        raise error

    runner = runtime(tmp_path)
    runner.count_requests = unavailable
    value = request()
    failed = await runner(value)
    assert failed.status == "failed"
    assert failed.retryable
    assert failed.reason == f"input_budget:{type(error).__name__}"
    assert failed.validation_evidence == []
    assert failed.attempts == 0
    assert runner.provider.generated == 0
    runner.count_requests = count_requests
    recovered = await runner(value, retry=True)
    assert recovered.status == "complete"
    assert runner.provider.generated == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["request_hash", "input_tokens"])
async def test_invalid_count_receipt_remains_terminal(tmp_path, invalid):
    async def malformed(requests, policy, output):
        counts = await count_requests(requests, policy, output)
        next(iter(counts["counts"].values()))[invalid] = "invalid"
        return counts

    runner = runtime(tmp_path)
    runner.count_requests = malformed
    result = await runner(request())
    assert result.status == "failed"
    assert not result.retryable
    assert runner.provider.generated == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_permanent_count_service_failure_remains_terminal(tmp_path, status):
    async def rejected(*args):
        raise APIStatusError(
            "private request content",
            response=httpx.Response(
                status, request=httpx.Request("POST", "https://example.test")
            ),
            body=None,
        )

    runner = runtime(tmp_path)
    runner.count_requests = rejected
    result = await runner(request())
    assert result.status == "failed"
    assert not result.retryable
    assert result.validation_evidence == []
    assert runner.provider.generated == 0


@pytest.mark.asyncio
async def test_count_retry_reuses_successful_earlier_value_receipt(tmp_path):
    calls = 0

    async def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("temporary count failure for second Core Value")
        return await count_requests(*args)

    runner = runtime(tmp_path)
    runner.count_requests = interrupted
    value = request(values=["benevolence", "security"])
    failed = await runner(value)
    assert failed.retryable
    assert runner.provider.generated == 1
    recovered = await runner(value, retry=True)
    assert recovered.status == "complete"
    assert recovered.reviews[0].provider_attempts[-1].reused
    assert recovered.attempts == 1
    assert runner.provider.generated == 2
    attempts = runner.provider.ledger.snapshot()["attempts"]
    assert len(attempts) == 2
    assert sum(row["calculated_cost_usd"] for row in attempts) == 0.002


@pytest.mark.asyncio
async def test_cached_completed_invalid_response_is_invalidated_then_retried(tmp_path):
    runner = runtime(tmp_path, invalid=True)
    value = request()
    item = source_review_requests(value)[0]
    # Simulate interruption after the provider wrote 'completed', before validation.
    await runner.provider.complete(**item)
    runner.provider.invalid = False
    result = await runner(value)
    assert result.status == "complete"
    assert runner.provider.generated == 2
    assert [a.status for a in result.reviews[0].provider_attempts] == [
        "invalid",
        "completed",
    ]
    assert result.reviews[0].provider_attempts[-1].attempt_number == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_path", [INTEGRATION_POLICY_PATH, LIVE_POLICY_PATH])
async def test_invalid_response_stops_at_two_attempts_and_reuses_terminal_failure(
    tmp_path, policy_path,
):
    runner = OpenAINorthStarRuntime(
        provider=FakeProvider(
            tmp_path, invalid=True,
            ledger=BudgetLedger(tmp_path / "budget.json", policy_path),
        ),
        count_requests=count_requests,
        policy_path=policy_path,
    )
    first = await runner(request())
    assert first.status == "failed" and not first.retryable
    assert runner.provider.generated == 2
    repeated = await runner(request(), retry=True)
    assert repeated.status == "failed"
    assert runner.provider.generated == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_path", [INTEGRATION_POLICY_PATH, LIVE_POLICY_PATH])
async def test_completed_record_accepts_only_its_pinned_budget_policy(
    tmp_path, policy_path,
):
    value = request(values=["benevolence", "security"])
    provider = FakeProvider(
        tmp_path, supportive={("benevolence", "entry:1")},
        ledger=BudgetLedger(tmp_path / "budget.json", policy_path),
    )
    runner = OpenAINorthStarRuntime(
        provider=provider, count_requests=count_requests, policy_path=policy_path,
    )
    record = await runner(value)
    assert record.status == "complete"
    assert record.selected.entry_id == "entry:1"
    assert validate_north_star_record(record, value) == record
    for review, item in zip(record.reviews, source_review_requests(value), strict=True):
        assert review.provider_attempts[-1].request_hash == stable_hash({
            **item, "policy_hash": stable_hash(provider.ledger.policy),
        })


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_change", ["unknown", "mixed"])
async def test_completed_record_rejects_unapproved_or_mixed_policies(
    tmp_path, policy_change,
):
    value = request(values=["benevolence", "security"])
    record = await runtime(tmp_path, supportive={("benevolence", "entry:1")})(value)
    changed = record.model_copy(deep=True)
    policy = json.loads(LIVE_POLICY_PATH.read_text())
    if policy_change == "unknown":
        policy["budget_usd"] = 0.75
    items = source_review_requests(value)
    changed.reviews[0].provider_attempts[-1].request_hash = stable_hash({
        **items[0], "policy_hash": stable_hash(policy),
    })
    with pytest.raises(ValueError, match="provider receipt policy changed"):
        validate_north_star_record(changed, value)


@pytest.mark.asyncio
async def test_no_api_key_creates_no_ledger_or_counts(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = OpenAINorthStarRuntime(
        ledger_path=tmp_path / "ledger.json", counts_path=tmp_path / "counts.json"
    )
    result = await runner(request())
    assert result.reason == "provider_unavailable" and result.retryable
    assert list(tmp_path.iterdir()) == []
    assert runner.provider is None


@pytest.mark.asyncio
async def test_redacted_ledger_keeps_spend_and_forget_removes_memory(tmp_path):
    ledger = _PrivateResponseLedger(tmp_path / "budget.json", INTEGRATION_POLICY_PATH)
    provider = FakeProvider(
        tmp_path, supportive={("benevolence", "entry:1")}, ledger=ledger
    )
    runner = OpenAINorthStarRuntime(provider=provider, count_requests=count_requests)
    result = await runner(request())
    state = json.loads(ledger.path.read_text())
    assert state["attempts"][0]["raw_text"] is None
    assert state["attempts"][0]["calculated_cost_usd"] == 0.001
    assert ledger.raw_responses
    assert result.reviews[0].provider_attempts[-1].raw_text
    runner.forget(result)
    assert not ledger.raw_responses


@pytest.mark.asyncio
async def test_saved_record_rejects_stale_profile_quote_and_provider_receipt(tmp_path):
    value = request()
    result = await runtime(tmp_path, supportive={("benevolence", "entry:1")})(value)
    assert validate_north_star_record(result, value) == result
    with pytest.raises(ValueError, match="profile_ref"):
        validate_north_star_record(
            result.model_copy(update={"profile_ref": "old"}), value
        )
    selected = result.selected.model_copy(update={"evidence_quote": "Invented words"})
    with pytest.raises(ValueError, match="selection changed"):
        validate_north_star_record(
            result.model_copy(update={"selected": selected}), value
        )
    bad = result.model_copy(deep=True)
    bad.reviews[0].provider_attempts[-1].reasoning_effort = "xhigh"
    with pytest.raises(ValueError, match="provider receipt changed"):
        validate_north_star_record(bad, value)


def test_profile_reference_handles_integral_numbers_and_unicode():
    assert profile_reference({"name": "Méera", "score": 1.0}) == profile_reference(
        {"score": 1, "name": "Méera"}
    )
