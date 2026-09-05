"""Offline replay tests use real temporary ledgers and synthetic provider output."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from scripts.experiments import north_star_phase0b as legacy
from scripts.experiments import north_star_phase0b_v2 as runner
from src.north_star.provider import (
    POLICY_PATH,
    BudgetedProvider,
    BudgetError,
    BudgetLedger,
    ProviderAttempt,
)
from src.north_star.review import SourceEntry

ACTION = "I helped neighbors file their claims."
OTHER_ACTION = "I taught visitors how to recycle."


def synthetic_case(name="test", *, empty=False):
    sources = (
        []
        if empty
        else [
            {
                "entry_id": f"{name}:entry:0",
                "journal_entry": f"{ACTION} {OTHER_ACTION}",
                "nudge_response": None,
            }
        ]
    )
    return {
        "case_id": f"{name}:universalism:episode_01",
        "core_value": "universalism",
        "value": {
            "user_phrase": "Care for people and the world",
            "definition": "Protect the welfare of all people and nature.",
        },
        "all_eligible_sources_in_retrieval_order": sources,
        "runtime_entry_ids": [source["entry_id"] for source in sources],
    }


def sources_for(case):
    return [
        SourceEntry(**source)
        for source in case["all_eligible_sources_in_retrieval_order"]
    ]


def response(case, *, decision="supportive", quote=ACTION, reason=None):
    supportive = decision == "supportive"
    return {
        "schema_version": "north-star-moment-review-v1",
        "core_value": case["core_value"],
        "results": [
            {
                "entry_id": source["entry_id"],
                "decision": decision,
                "quote_source": "journal_entry" if supportive else None,
                "evidence_quote": quote if supportive else "",
                "reason_code": reason
                or ("observable_choice" if supportive else "wrong_value"),
            }
            for source in case["all_eligible_sources_in_retrieval_order"]
        ],
    }


def completed(payload):
    return {
        "status": "completed",
        "raw_text": json.dumps(payload),
        "input_tokens": 100,
        "output_tokens": 20,
        "calculated_cost_usd": 0.001,
    }


class ScriptedProvider(BudgetedProvider):
    """Replace transport only; exercise the real request hash and ledger limits."""

    def __init__(self, budget, outputs):
        super().__init__(budget)
        self.outputs = list(outputs)
        self.transport_calls = 0
        self.interrupt_after_finish = False

    async def _complete(self, request, *, retry):
        attempt = self.ledger.reserve(request, retry=retry)
        if attempt.reused:
            return attempt
        self.transport_calls += 1
        assert self.outputs, "Unexpected provider transport"
        attempt = attempt.model_copy(update=self.outputs.pop(0))
        self.ledger.finish(attempt)
        if self.interrupt_after_finish:
            raise RuntimeError("Injected interruption after provider receipt")
        return attempt


@pytest.fixture
def budget(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_bytes(POLICY_PATH.read_bytes())
    return BudgetLedger(tmp_path / "ledger.json", policy)


def seed_role(provider, case, role="runtime", *, candidate=None):
    """Use v1 itself to seed exact historical request hashes and receipts."""
    return asyncio.run(
        legacy.review(
            provider,
            case=case,
            sources=sources_for(case),
            role=role,
            candidate=candidate,
        )
    )


def forbid_complete(monkeypatch, provider):
    complete = AsyncMock(side_effect=AssertionError("Replay called provider.complete"))
    monkeypatch.setattr(provider, "complete", complete)
    return complete


def replay_case(case, provider, directory, *, allow_paid=False):
    directory.mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        runner.run_case(
            case,
            provider,
            asyncio.Semaphore(1),
            directory=directory,
            allow_paid=allow_paid,
        )
    )


def receipt_keys(result):
    return [
        (row["attempt"]["request_hash"], row["attempt"]["attempt_number"])
        for row in result["attempts"]
    ]


def test_legacy_replay_of_exhausted_invalid_attempts_raises_without_transport(budget):
    case = synthetic_case()
    invalid = completed(response(case, decision="abstain", reason="other_actor"))
    provider = ScriptedProvider(budget, [invalid, invalid])
    batch, attempts = seed_role(provider, case)
    assert batch is None
    assert len(attempts) == provider.transport_calls == 2
    saved = budget.snapshot()

    with pytest.raises(BudgetError, match="Retry limit"):
        seed_role(provider, case)

    assert provider.transport_calls == 2
    assert budget.snapshot() == saved


@pytest.mark.parametrize("outcome", ["selected", "no_example", "exhausted"])
@pytest.mark.parametrize("allow_paid", [False, True])
def test_terminal_case_replay_preserves_all_receipts_and_budget(
    tmp_path, budget, monkeypatch, outcome, allow_paid
):
    case = synthetic_case()
    omitted = completed(response(case, decision="not_supportive"))
    invalid = completed(response(case, decision="abstain", reason="other_actor"))
    runtime_outputs = (
        [invalid, invalid]
        if outcome == "exhausted"
        else [omitted if outcome == "no_example" else completed(response(case))]
    )
    provider = ScriptedProvider(
        budget,
        [
            *runtime_outputs,
            omitted if outcome != "selected" else completed(response(case)),
        ],
    )
    seed_role(provider, case)
    seed_role(provider, case, "reference")
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    result = replay_case(case, provider, tmp_path / "results", allow_paid=allow_paid)

    assert result["status"] == ("failed" if outcome == "exhausted" else "completed")
    assert bool(result["selected"]) == (outcome == "selected")
    assert result["reference_no_example"] == (outcome != "selected")
    expected = [
        (attempt["request_hash"], attempt["attempt_number"])
        for attempt in saved["attempts"]
    ]
    assert receipt_keys(result) == expected
    assert len(receipt_keys(result)) == len(set(receipt_keys(result)))
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


def test_empty_history_omits_without_provider_or_attempts(
    tmp_path, budget, monkeypatch
):
    provider = ScriptedProvider(budget, [])
    complete = forbid_complete(monkeypatch, provider)

    result = replay_case(synthetic_case(empty=True), provider, tmp_path / "results")

    assert result["status"] == "no_earlier_writing"
    assert result["selected"] is None
    assert result["attempts"] == []
    assert budget.snapshot()["attempts"] == []
    complete.assert_not_awaited()


@pytest.mark.parametrize("saved_role", [None, "runtime", "reference"])
def test_offline_replay_of_missing_role_fails_without_provider(
    tmp_path, budget, monkeypatch, saved_role
):
    case = synthetic_case()
    provider = ScriptedProvider(budget, [completed(response(case))])
    if saved_role is not None:
        seed_role(provider, case, saved_role)
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    result = replay_case(case, provider, tmp_path / "results")

    assert result["status"] == "failed"
    summary = runner.summarize([result], budget.snapshot()["attempts"])
    assert summary["failed_cases"] == 1
    assert not summary["gate_passed"]
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


@pytest.mark.parametrize("allow_paid", [False, True])
def test_pending_attempt_keeps_reservation_and_does_not_block_other_case(
    tmp_path, budget, monkeypatch, allow_paid
):
    interrupted, unrelated = synthetic_case("pending"), synthetic_case("unrelated")
    provider = ScriptedProvider(
        budget,
        [
            {"status": "pending"},
            completed(response(interrupted)),
            completed(response(unrelated)),
            completed(response(unrelated)),
        ],
    )
    seed_role(provider, interrupted)
    seed_role(provider, interrupted, "reference")
    seed_role(provider, unrelated)
    seed_role(provider, unrelated, "reference")
    saved = budget.snapshot()
    assert saved["attempts"][0]["status"] == "pending"
    assert saved["attempts"][0]["reserved_cost_usd"] > 0
    assert saved["attempts"][0]["calculated_cost_usd"] is None
    complete = forbid_complete(monkeypatch, provider)
    (tmp_path / "results").mkdir()

    async def run_both():
        semaphore = asyncio.Semaphore(2)
        return await asyncio.gather(
            *[
                runner.run_case(
                    case,
                    provider,
                    semaphore,
                    directory=tmp_path / "results",
                    allow_paid=allow_paid,
                )
                for case in [interrupted, unrelated]
            ]
        )

    failed, completed_case = asyncio.run(run_both())

    assert failed["status"] == "failed"
    assert failed["selected"] is None
    assert failed["attempts"][0]["attempt"]["status"] == "pending"
    assert completed_case["status"] == "completed"
    assert completed_case["selected"]["evidence_quote"] == ACTION
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


def test_completed_receipt_survives_interruption_before_case_write(
    tmp_path, budget, monkeypatch
):
    case = synthetic_case()
    provider = ScriptedProvider(budget, [completed(response(case))] * 2)
    provider.interrupt_after_finish = True
    with pytest.raises(RuntimeError, match="Injected interruption"):
        seed_role(provider, case)
    provider.interrupt_after_finish = False
    seed_role(provider, case, "reference")
    saved = budget.snapshot()
    assert all(attempt["status"] == "completed" for attempt in saved["attempts"])
    directory = tmp_path / "results"
    assert not directory.exists()
    complete = forbid_complete(monkeypatch, provider)

    result = replay_case(case, provider, directory)

    assert result["status"] == "completed"
    assert result["selected"]["evidence_quote"] == ACTION
    saved_case = directory / "cases" / f"{case['case_id'].replace(':', '_')}.json"
    assert json.loads(saved_case.read_text()) == result
    assert len(result["attempts"]) == 2
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


def test_one_saved_invalid_attempt_permits_only_remaining_mocked_retry(
    budget, monkeypatch
):
    case = synthetic_case()
    invalid = completed(response(case, decision="abstain", reason="other_actor"))
    provider = ScriptedProvider(budget, [invalid, completed(response(case))])
    provider.interrupt_after_finish = True
    with pytest.raises(RuntimeError, match="Injected interruption"):
        seed_role(provider, case)
    first = budget.snapshot()["attempts"][0]
    # Reconstruct the validation step lost at interruption, without a new attempt.
    provider.invalidate(ProviderAttempt(**first), "review_contract_invalid")
    provider.interrupt_after_finish = False
    saved_failure = budget.snapshot()
    with monkeypatch.context() as patch:
        offline_complete = forbid_complete(patch, provider)
        offline_batch, offline_receipts = asyncio.run(
            runner.review(
                provider,
                case=case,
                sources=sources_for(case),
                role="runtime",
            )
        )
        assert offline_batch is None
        assert len(offline_receipts) == 1
        assert budget.snapshot() == saved_failure
        offline_complete.assert_not_awaited()

    complete = AsyncMock(wraps=provider.complete)
    monkeypatch.setattr(provider, "complete", complete)

    batch, attempts = asyncio.run(
        runner.review(
            provider,
            case=case,
            sources=sources_for(case),
            role="runtime",
            allow_paid=True,
        )
    )

    assert batch is not None
    assert [row["attempt"]["attempt_number"] for row in attempts] == [1, 2]
    assert [row["attempt"]["status"] for row in attempts] == ["invalid", "completed"]
    assert complete.await_count == 1
    assert complete.call_args.kwargs["retry"] is True
    assert provider.transport_calls == 2
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    replayed, receipts = asyncio.run(
        runner.review(
            provider,
            case=case,
            sources=sources_for(case),
            role="runtime",
            allow_paid=True,
        )
    )

    assert replayed == batch
    assert len(receipts) == 2
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


@pytest.mark.parametrize("replacement", [OTHER_ACTION, "I invented a new action."])
def test_saved_completed_quote_check_still_rejects_replacement_or_fabrication(
    budget, monkeypatch, replacement
):
    case = synthetic_case()
    candidate = response(case)["results"][0]
    provider = ScriptedProvider(budget, [completed(response(case, quote=replacement))])
    provider.interrupt_after_finish = True
    with pytest.raises(RuntimeError, match="Injected interruption"):
        seed_role(provider, case, "quote_reference", candidate=candidate)
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    batch, receipts = asyncio.run(
        runner.review(
            provider,
            case=case,
            sources=sources_for(case),
            role="quote_reference",
            candidate=candidate,
        )
    )

    assert batch is None
    assert len(receipts) == 1
    assert receipts[0]["attempt"]["status"] == "invalid"
    assert len(budget.snapshot()["attempts"]) == len(saved["attempts"]) == 1
    assert budget.snapshot()["attempts"][0]["calculated_cost_usd"] == 0.001
    complete.assert_not_awaited()


def test_exact_candidate_quotation_replays_after_different_primary_quote(
    tmp_path, budget, monkeypatch
):
    case = synthetic_case()
    provider = ScriptedProvider(
        budget,
        [
            completed(response(case)),
            completed(response(case, quote=OTHER_ACTION)),
            completed(response(case)),
        ],
    )
    seed_role(provider, case)
    seed_role(provider, case, "reference")
    seed_role(provider, case, "quote_reference", candidate=response(case)["results"][0])
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    result = replay_case(case, provider, tmp_path / "results")

    assert result["status"] == "completed"
    assert result["selected"]["evidence_quote"] == ACTION
    assert result["quote_reference"]["results"][0]["evidence_quote"] == ACTION
    assert not result["incorrect_displayed"]
    assert len(result["attempts"]) == 3
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


def test_missing_exact_candidate_verification_is_failed_offline_case(
    tmp_path, budget, monkeypatch
):
    case = synthetic_case()
    provider = ScriptedProvider(
        budget,
        [completed(response(case)), completed(response(case, quote=OTHER_ACTION))],
    )
    seed_role(provider, case)
    seed_role(provider, case, "reference")
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    result = replay_case(case, provider, tmp_path / "results")

    assert result["status"] == "failed"
    assert result["quote_reference"] is None
    assert result["incorrect_displayed"]
    assert len(result["attempts"]) == 2
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


@pytest.mark.parametrize("allow_paid", [False, True])
def test_nonretryable_saved_refusal_never_calls_provider_again(
    budget, monkeypatch, allow_paid
):
    case = synthetic_case()
    provider = ScriptedProvider(
        budget,
        [{"status": "refused", "error_type": "provider_refusal", "retryable": False}],
    )
    seed_role(provider, case)
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    batch, receipts = asyncio.run(
        runner.review(
            provider,
            case=case,
            sources=sources_for(case),
            role="runtime",
            allow_paid=allow_paid,
        )
    )

    assert batch is None
    assert len(receipts) == 1
    assert receipts[0]["attempt"]["status"] == "refused"
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


def test_concurrent_reviews_await_one_active_provider_attempt(budget):
    case = synthetic_case()

    async def run_concurrently():
        started, release = asyncio.Event(), asyncio.Event()

        class DelayedProvider(BudgetedProvider):
            transport_calls = 0

            async def _complete(self, request, *, retry):
                attempt = self.ledger.reserve(request, retry=retry)
                if attempt.reused:
                    return attempt
                self.transport_calls += 1
                started.set()
                await release.wait()
                return self.ledger.finish(
                    attempt.model_copy(update=completed(response(case)))
                )

        provider = DelayedProvider(budget)

        async def review():
            return await runner.review(
                provider,
                case=case,
                sources=sources_for(case),
                role="runtime",
                allow_paid=True,
            )

        first = asyncio.create_task(review())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert budget.snapshot()["attempts"][0]["status"] == "pending"
        second = asyncio.create_task(review())
        # Give the second review a turn while the first transport is still active.
        await asyncio.sleep(0)
        finished_before_transport = second.done()
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
        assert not finished_before_transport
        assert provider.transport_calls == 1
        return results

    results = asyncio.run(run_concurrently())

    for batch, receipts in results:
        assert batch is not None
        assert batch.results[0].evidence_quote == ACTION
        assert len(receipts) == 1
        assert receipts[0]["attempt"]["status"] == "completed"
    assert results[0] == results[1]
    assert len(budget.snapshot()["attempts"]) == 1
    assert budget.snapshot()["attempts"][0]["calculated_cost_usd"] == 0.001
