"""One real NSM service call on synthetic writing; other providers are test doubles."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.demo.canonical_fixture import build_canonical_fixture
from src.demo.contracts import (
    AssessmentTimeAdvanceRequest,
    NorthStarReviewedEvent,
    NorthStarReviewRequest,
    SessionCreateRequest,
)
from src.north_star.live_runtime import DEFAULT_LIVE_DIRECTORY, LiveNorthStarRuntime
from tests.demo.test_experience_service import _receipt, _service, _submit_request

OUTPUT = Path(__file__).resolve().parent
QUOTE = (
    "I helped my sister carry the groceries upstairs "
    "and stayed to cook dinner with her."
)


async def main():
    load_dotenv(".env")
    coach = json.dumps(
        {
            "weekly_mirror": (
                f'You wrote, "{QUOTE}" '
                "That gives us one concrete moment from this week."
            ),
            "tension_explanation": (
                "This moment describes time spent caring for someone close to you; "
                "it does not establish a wider pattern."
            ),
            "reflective_question": "What did making that time mean to you?",
        }
    )
    service, _, reviewer = _service(
        [_receipt(decision=None, nudge_text=None)], coach_response=coach
    )
    runtime = LiveNorthStarRuntime()
    service._north_star_runtime = runtime
    profile = build_canonical_fixture().session.profile.model_copy(
        update={"session_id": "integrated-coach-live-smoke-20260908"}
    )
    create = SessionCreateRequest(
        operation="create_session",
        request_id="create-smoke",
        idempotency_key="9" * 64,
        profile=profile,
        assessment_timezone="Asia/Singapore",
    )
    created = await service.create_session(create)
    assert created.operation == "create_session"
    submit = _submit_request(create, index=0, expected_revision=0)
    submit = submit.model_copy(
        update={
            "journal_entry": submit.journal_entry.model_copy(update={"content": QUOTE})
        }
    )
    submitted = await service.submit_journal_entry(submit)
    assert submitted.operation == "submit_journal_entry"
    closed = await service.advance_assessment_time(
        AssessmentTimeAdvanceRequest(
            operation="advance_assessment_time",
            request_id="close-smoke",
            idempotency_key="e" * 64,
            session_id=profile.session_id,
            expected_revision=1,
            action="close_week",
        )
    )
    assert closed.operation == "advance_assessment_time"
    assert closed.session.weekly_digest is not None
    assert closed.session.weekly_digest.coach_narrative is not None
    request = NorthStarReviewRequest(
        operation="review_north_star",
        request_id="moment-smoke",
        session_id=profile.session_id,
        expected_revision=closed.session.revision,
        week_start="2026-07-20",
    )
    snapshot = service._north_star_request(closed.session, request.week_start)
    (OUTPUT / "live_request.json").write_text(snapshot.model_dump_json(indent=2) + "\n")
    before_path = DEFAULT_LIVE_DIRECTORY / "budget.json"
    before = (
        json.loads(before_path.read_text())
        if before_path.exists()
        else {"attempts": []}
    )
    try:
        first = await service.review_north_star(request)
        assert first.operation == "north_star_reviewed", first.model_dump()
        event = next(
            e
            for e in reversed(service._events[profile.session_id])
            if isinstance(e, NorthStarReviewedEvent)
        )
        record = event.details.record
        (OUTPUT / "live_event.json").write_text(event.model_dump_json(indent=2) + "\n")
        assert record.status == "complete", (record.reason, record.validation_evidence)
        assert record.selected is not None
        assert record.selected.evidence_quote in QUOTE
        assert event.source == "live_run"
        assert all(
            a.provider_response_id
            and a.input_tokens is not None
            and a.output_tokens is not None
            for r in record.reviews
            for a in r.provider_attempts
        )
        charged = json.loads(before_path.read_text())
        second = await service.review_north_star(request)
        assert second.operation == "north_star_reviewed"
        assert second.event_ids == first.event_ids
        after = json.loads(before_path.read_text())
        assert after["attempts"] == charged["attempts"]
        assert first.session.weekly_digest == closed.session.weekly_digest
        assert len(reviewer.requests) == 1
        attempts = charged["attempts"][len(before["attempts"]) :]
        result = {
            "source": (
                "real OpenAI NSM on synthetic input; nudge, Weekly Drift Reviewer "
                "and Coach are deterministic test doubles"
            ),
            "status": record.status,
            "reason": record.reason,
            "new_generation_attempts": len(attempts),
            "calculated_cost_usd": sum(a["calculated_cost_usd"] or 0 for a in attempts),
            "unmetered_attempts": sum(
                a["calculated_cost_usd"] is None for a in attempts
            ),
            "repeat_added_attempts": 0,
            "weekly_digest_unchanged": True,
            "live_budget_policy": "config/evals/north_star_live_v1.json",
            "ledger": str(before_path.relative_to(Path.cwd())),
            "pricing": "repository configured rates, not a billing statement",
            "not_human_validation": True,
        }
        (OUTPUT / "live_smoke_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result))
    finally:
        runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
