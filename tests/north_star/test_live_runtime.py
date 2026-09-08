"""Live accounting isolation, receipt concurrency, and main-loop responsiveness."""

from __future__ import annotations

import asyncio
import fcntl
import json
from pathlib import Path

import pytest

from src.north_star.live_runtime import LiveNorthStarRuntime, _SeededLiveLedger
from src.north_star.provider import BudgetError, BudgetLedger, stable_hash
from src.north_star.runtime import (
    INTEGRATION_POLICY_PATH,
    LIVE_POLICY_PATH,
    build_north_star_request,
    source_review_requests,
)
from tests.north_star.test_runtime import FakeProvider, count_requests, request


def finalized_source(path: Path, *, spent: float = 0) -> tuple[Path, BudgetLedger]:
    path.mkdir()
    ledger = BudgetLedger(path / "budget.json", INTEGRATION_POLICY_PATH)
    if spent:
        envelope = {
            **source_review_requests(request())[0],
            "policy_hash": stable_hash(ledger.policy),
        }
        attempt = ledger.reserve(envelope, retry=False)
        attempt.status = "completed"
        attempt.raw_text = "Synthetic source text must not be copied."
        attempt.calculated_cost_usd = spent
        ledger.finish(attempt)
    state = ledger.snapshot()
    (path / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "north-star-integration-results-v1",
                "generation_attempts": len(state["attempts"]),
                "new_spent_or_reserved_usd": spent,
                "cumulative_spent_or_reserved_usd": spent
                + ledger.policy["prior_spend_usd"],
            }
        )
    )
    return path, ledger


def fake_live(path, source_path, *, measure=count_requests):
    return LiveNorthStarRuntime(
        directory=path,
        source_directory=source_path,
        measure=measure,
        provider_factory=lambda ledger: FakeProvider(
            path, supportive={("benevolence", "entry:1")}, ledger=ledger
        ),
    )


def fresh_live(path, *, measure=count_requests):
    return LiveNorthStarRuntime(
        directory=path,
        measure=measure,
        provider_factory=lambda ledger: FakeProvider(
            path, supportive={("benevolence", "entry:1")}, ledger=ledger
        ),
    )


@pytest.mark.asyncio
async def test_fresh_live_authorization_needs_no_experiment_and_redacts_receipts(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    runner = fresh_live(tmp_path / "live")
    try:
        result = await runner(request())
        assert result.status == "complete"
        assert result.selected.entry_id == "entry:1"
        ledger = runner._runtime.provider.ledger
        assert ledger.policy == json.loads(LIVE_POLICY_PATH.read_text())
        assert ledger.policy["budget_usd"] == 1
        assert ledger.policy["prior_spend_usd"] == 0
        state = json.loads(ledger.path.read_text())
        assert state["policy_hash"] == stable_hash(ledger.policy)
        assert "integration_budget_sha256" not in state
        assert len(state["attempts"]) == 1
        assert state["attempts"][0]["raw_text"] is None
        assert state["attempts"][0]["calculated_cost_usd"] == 0.001
        assert ledger.raw_responses
        runner.forget(result)
    finally:
        runner.close()
    assert not ledger.raw_responses
    assert sorted(path.name for path in tmp_path.iterdir()) == ["live"]


@pytest.mark.asyncio
async def test_fresh_live_restart_keeps_previously_charged_attempts(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    path = tmp_path / "live"
    first = fresh_live(path)
    try:
        assert (await first(request())).status == "complete"
        prior = first._runtime.provider.ledger.snapshot()["attempts"]
    finally:
        first.close()
    second = fresh_live(path)
    try:
        next_request = request().model_copy(update={"session_id": "next-live-session"})
        next_request = build_north_star_request(
            **next_request.model_dump(include={
                "session_id", "owner_id", "profile_ref", "core_values",
                "week_start", "week_end", "cutoff_at",
            }),
            drift_result=next_request.drift_result,
            writing=next_request.writing,
        )
        assert (await second(next_request)).status == "complete"
        attempts = second._runtime.provider.ledger.snapshot()["attempts"]
        assert attempts[: len(prior)] == prior
        assert len(attempts) == 2
        assert sum(row["calculated_cost_usd"] for row in attempts) == 0.002
    finally:
        second.close()


@pytest.mark.asyncio
async def test_fresh_live_pending_reservations_exhaust_the_shared_allowance(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    path = tmp_path / "live"
    ledger = BudgetLedger(path / "budget.json", LIVE_POLICY_PATH)
    template = source_review_requests(request())[0]
    for number in range(100):
        try:
            ledger.reserve(
                {
                    **template,
                    "purpose": f"interrupted-live-request:{number}",
                    "policy_hash": stable_hash(ledger.policy),
                },
                retry=False,
            )
        except BudgetError as exc:
            assert "Total authorized budget exhausted" in str(exc)
            break
    else:
        pytest.fail("The fresh live ledger did not enforce its total allowance")
    original = ledger.path.read_bytes()
    runner = fresh_live(path)
    try:
        result = await runner(request())
        assert result.reason == "budget_unavailable"
        assert runner._runtime.provider.generated == 0
        assert ledger.path.read_bytes() == original
        state = ledger.snapshot()
        assert all(row["status"] == "pending" for row in state["attempts"])
        assert sum(row["reserved_cost_usd"] for row in state["attempts"]) <= 1
    finally:
        runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["corrupt", "changed_policy"])
async def test_fresh_live_invalid_existing_ledger_fails_before_counting(
    tmp_path, monkeypatch, invalid
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    path = tmp_path / "live"
    path.mkdir()
    ledger_path = path / "budget.json"
    ledger_path.write_text(
        "{broken" if invalid == "corrupt" else json.dumps({
            "schema_version": "north-star-budget-v1",
            "policy_hash": "unapproved-policy",
            "attempts": [],
        })
    )
    original = ledger_path.read_bytes()

    async def forbidden_measure(*args):
        pytest.fail("Invalid budget must fail before input counting")

    runner = fresh_live(path, measure=forbidden_measure)
    try:
        result = await runner(request())
        assert result.reason == "budget_unavailable"
        assert runner._runtime is None
        assert ledger_path.read_bytes() == original
    finally:
        runner.close()


@pytest.mark.asyncio
async def test_live_files_are_separate_and_seeded_spend_is_preserved(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    source_path, _ = finalized_source(tmp_path / "experiment", spent=0.02)
    original_files = {file.name: file.read_bytes() for file in source_path.iterdir()}
    live_path = tmp_path / "live"
    runner = fake_live(live_path, source_path)
    try:
        result = await runner(request())
        assert result.status == "complete"
        state = json.loads((live_path / "budget.json").read_text())
        assert len(state["attempts"]) == 2
        assert all(attempt["raw_text"] is None for attempt in state["attempts"])
        assert sum(row["calculated_cost_usd"] for row in state["attempts"]) == 0.021
        assert {
            file.name: file.read_bytes() for file in source_path.iterdir()
        } == original_files
        live_ledger = runner._runtime.provider.ledger
        assert live_ledger.raw_responses
        runner.forget(result)
    finally:
        runner.close()
    assert not live_ledger.raw_responses


@pytest.mark.asyncio
async def test_missing_key_initializes_no_threads_or_files(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = LiveNorthStarRuntime(
        directory=tmp_path / "live", source_directory=tmp_path / "missing"
    )
    result = await runner(request())
    assert result.reason == "provider_unavailable"
    assert runner._executor is None and runner._runtime is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_insufficient_evidence_needs_no_seed_or_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    runner = LiveNorthStarRuntime(
        directory=tmp_path / "live", source_directory=tmp_path / "missing"
    )
    result = await runner(request(unavailable=True))
    assert result.reason == "insufficient_evidence"
    assert runner._executor is None and runner._runtime is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_missing_seed_returns_explicit_budget_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    runner = LiveNorthStarRuntime(
        directory=tmp_path / "live", source_directory=tmp_path / "missing"
    )
    try:
        result = await runner(request())
        assert result.reason == "budget_unavailable"
        assert result.validation_evidence == [
            "Finalized integration budget is unavailable"
        ]
        assert not (tmp_path / "live" / "budget.json").exists()
    finally:
        runner.close()


def test_unfinalized_or_changed_source_budget_fails_closed(tmp_path):
    source_path, source_ledger = finalized_source(tmp_path / "experiment")
    ledger = _SeededLiveLedger(
        tmp_path / "live" / "budget.json", source_directory=source_path
    )
    ledger.snapshot()
    item = source_review_requests(request())[0]
    source_ledger.reserve(
        {**item, "policy_hash": stable_hash(ledger.policy)}, retry=False
    )
    with pytest.raises(BudgetError, match="not finalized"):
        ledger.snapshot()
    attempt = source_ledger.snapshot()["attempts"][0]
    attempt["status"] = "failed"
    source_ledger.transact(lambda state: state.update(attempts=[attempt]))
    report = json.loads((source_path / "report.json").read_text())
    report.update(
        {
            "generation_attempts": 1,
            "new_spent_or_reserved_usd": attempt["reserved_cost_usd"],
            "cumulative_spent_or_reserved_usd": attempt["reserved_cost_usd"]
            + ledger.policy["prior_spend_usd"],
        }
    )
    (source_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(BudgetError, match="accounting is stale"):
        ledger.snapshot()


def test_separate_live_directory_does_not_reset_authorized_budget(tmp_path):
    source_path, _ = finalized_source(tmp_path / "experiment", spent=19.36)
    ledger = _SeededLiveLedger(
        tmp_path / "live" / "budget.json", source_directory=source_path
    )
    item = source_review_requests(request())[0]
    item["purpose"] = "new-live-request"
    with pytest.raises(BudgetError, match="Total authorized budget exhausted"):
        ledger.reserve({**item, "policy_hash": stable_hash(ledger.policy)}, retry=False)


@pytest.mark.asyncio
async def test_external_file_lock_does_not_block_app_event_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    source_path, source_ledger = finalized_source(tmp_path / "experiment")
    runner = fake_live(tmp_path / "live", source_path)
    try:
        with source_ledger.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            task = asyncio.create_task(runner(request()))
            await asyncio.wait_for(asyncio.sleep(0.03), timeout=0.2)
            assert not task.done()
        result = await asyncio.wait_for(task, timeout=3)
        assert result.status == "complete"
    finally:
        runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
async def test_concurrent_live_instances_keep_every_input_count_receipt(
    tmp_path, monkeypatch, fresh
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-no-network")
    source_path, _ = finalized_source(tmp_path / "experiment")
    live_path = tmp_path / "live"

    async def slow_measure(requests, policy, output):
        state = json.loads(output.read_text()) if output.exists() else {"counts": {}}
        measured = await count_requests(requests, policy, output)
        await asyncio.sleep(0.03)
        state["counts"].update(measured["counts"])
        output.write_text(json.dumps(state))
        return state

    one = (
        fresh_live(live_path, measure=slow_measure)
        if fresh else fake_live(live_path, source_path, measure=slow_measure)
    )
    two = (
        fresh_live(live_path, measure=slow_measure)
        if fresh else fake_live(live_path, source_path, measure=slow_measure)
    )
    first = request()
    second = build_north_star_request(
        **first.model_dump(
            include={
                "owner_id",
                "profile_ref",
                "core_values",
                "week_start",
                "week_end",
                "cutoff_at",
            }
        ),
        session_id="other-session",
        drift_result=first.drift_result,
        writing=first.writing,
    )
    try:
        results = await asyncio.gather(one(first), two(second))
        assert all(result.status == "complete" for result in results)
        counts = json.loads((live_path / "input-counts.json").read_text())["counts"]
        expected = {
            stable_hash(item)
            for value in (first, second)
            for item in source_review_requests(value)
        }
        assert set(counts) == expected
        attempts = one._runtime.provider.ledger.snapshot()["attempts"]
        assert len(attempts) == 2
        assert sum(row["calculated_cost_usd"] for row in attempts) == 0.002
    finally:
        one.close()
        two.close()
