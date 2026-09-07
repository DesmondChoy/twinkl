"""Shared NSM source boundaries and deterministic saved replay attachment."""

import asyncio
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import north_star_integration as integration
from src.demo.scenarios import (
    NORTH_STAR_REPORT_PATH,
    SELECTIONS,
    attach_saved_north_star,
    build_saved_north_star_request,
    build_scenario_fixture,
    project_scenario_week,
)
from src.north_star.provider import BudgetedProvider
from src.north_star.runtime import (
    pending_north_star_record,
    source_review_requests,
    validate_north_star_record,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def base_fixtures():
    return [
        build_scenario_fixture(ROOT, selection, include_north_star=False)
        for selection in SELECTIONS
    ]


def test_all_saved_week_inputs_preserve_available_original_writing(base_fixtures):
    total = 0
    for fixture in base_fixtures:
        for week in fixture.scenario.weeks:
            total += 1
            request = build_saved_north_star_request(fixture, week.week_id)
            session, events = project_scenario_week(fixture, week.week_id)
            expected = {
                entry.journal_entry_id: (
                    datetime.combine(
                        date.fromisoformat(entry.date), time.min, tzinfo=UTC
                    )
                    + timedelta(microseconds=3 * entry.t_index)
                ).isoformat()
                for entry in session.journal_entries
            }
            assert request.profile_ref
            assert request.owner_id == fixture.scenario.persona_id
            assert {source.entry_id for source in request.writing} == set(expected)
            assert all(
                source.available_at == expected[source.entry_id]
                and source.available_at <= request.cutoff_at
                and source.date <= week.week_end
                and (source.nudge_response is None)
                == (source.response_available_at is None)
                for source in request.writing
            )
            for provider_request in source_review_requests(request):
                payload = json.loads(provider_request["prompt"])
                assert set(payload) == {
                    "core_value",
                    "user_phrase",
                    "approved_definition",
                    "sources",
                    "context_hash",
                }
                for source in payload["sources"]:
                    assert source["nudge_response"] == next(
                        entry.nudge_response
                        for entry in session.journal_entries
                        if entry.journal_entry_id == source["entry_id"]
                    )
                    assert source["journal_entry"] == next(
                        entry.content
                        for entry in session.journal_entries
                        if entry.journal_entry_id == source["entry_id"]
                    )
    assert total == 27


def _save_pending_records(directory, fixtures):
    cases = []
    for fixture in fixtures:
        for week in fixture.scenario.weeks:
            request = build_saved_north_star_request(fixture, week.week_id)
            cases.append(
                {
                    "scenario_id": fixture.scenario.scenario_id,
                    "week_id": week.week_id,
                    "record": pending_north_star_record(request).model_dump(
                        mode="json"
                    ),
                }
            )
    path = directory / NORTH_STAR_REPORT_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"cases": cases}))
    return path


def test_saved_attachment_and_projection_make_no_provider_calls(
    base_fixtures, tmp_path, monkeypatch
):
    async def forbidden(*args, **kwargs):
        pytest.fail("Saved replay must never call a model")

    monkeypatch.setattr(BudgetedProvider, "complete", forbidden)
    _save_pending_records(tmp_path, base_fixtures)
    for original in base_fixtures:
        attached = attach_saved_north_star(original, root=tmp_path)
        assert len(attached.trace_events) == (
            len(original.trace_events) + len(original.scenario.weeks)
        )
        original_events = {event.event_id: event for event in original.trace_events}
        for event in attached.trace_events:
            if event.event_type != "north_star_reviewed":
                assert event == original_events[event.event_id]
        for index, week in enumerate(attached.scenario.weeks):
            session, events = project_scenario_week(attached, week.week_id)
            visible_records = [
                event for event in events if event.event_type == "north_star_reviewed"
            ]
            assert len(visible_records) == index + 1
            current = next(
                event for event in visible_records if event.event_id in week.event_ids
            )
            request = build_saved_north_star_request(attached, week.week_id)
            validate_north_star_record(current.details.record, request)
            assert set(session.trace_event_ids) == {event.event_id for event in events}
            assert all(
                record.details.record.week_end <= week.week_end
                for record in visible_records
            )


def test_saved_attachment_rejects_missing_week(base_fixtures, tmp_path):
    path = _save_pending_records(tmp_path, base_fixtures)
    payload = json.loads(path.read_text())
    payload["cases"].pop(0)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cover every week"):
        attach_saved_north_star(base_fixtures[0], root=tmp_path)


def test_saved_attachment_rejects_stale_profile(base_fixtures, tmp_path):
    path = _save_pending_records(tmp_path, base_fixtures)
    payload = json.loads(path.read_text())
    payload["cases"][0]["record"]["profile_ref"] = "changed-profile"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="profile_ref"):
        attach_saved_north_star(base_fixtures[0], root=tmp_path)


def test_legacy_paid_preparation_rejects_experiment_response_availability(tmp_path):
    # This helper used the pre-reset Journal-Entry-only policy. The current
    # experiment export must never silently become a fresh live budget setup.
    with pytest.raises(
        ValueError, match="Legacy responses lack independent availability"
    ):
        integration.prepare(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_offline_assessment_records_invalid_cached_response_without_mutation(
    base_fixtures, tmp_path
):
    fixture = base_fixtures[0]
    request = integration.reference_request(
        source_review_requests(
            build_saved_north_star_request(fixture, fixture.scenario.weeks[0].week_id)
        )[0]
    )
    policy = json.loads(integration.INTEGRATION_POLICY_PATH.read_text())
    attempt = {
        "request_hash": integration.stable_hash(
            {**request, "policy_hash": integration.stable_hash(policy)}
        ),
        "status": "completed",
        "raw_text": "{}",
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Offline assessment must not call or mutate the provider")

    provider = SimpleNamespace(
        ledger=SimpleNamespace(policy=policy, snapshot=lambda: {"attempts": [attempt]}),
        complete=forbidden,
        invalidate=forbidden,
    )
    assert (
        asyncio.run(
            integration.execute_assessment(
                request, provider, tmp_path, allow_paid=False
            )
        )
        is None
    )


def test_reference_execution_rejects_oversized_complete_input_before_generation(
    base_fixtures, tmp_path, monkeypatch
):
    fixture = base_fixtures[0]
    request = integration.reference_request(
        source_review_requests(
            build_saved_north_star_request(fixture, fixture.scenario.weeks[0].week_id)
        )[0]
    )
    policy = json.loads(integration.INTEGRATION_POLICY_PATH.read_text())

    async def count(*args):
        return {
            "counts": {
                integration.stable_hash(request): {
                    "request_hash": integration.stable_hash(request),
                    "payload_hash": integration.stable_hash(
                        integration.input_budget.count_payload(request, policy)
                    ),
                    "model": policy["reference"]["model"],
                    "input_tokens": 16001,
                }
            }
        }

    async def forbidden(*args, **kwargs):
        pytest.fail("An oversized input must not generate a response")

    monkeypatch.setattr(integration.input_budget, "measure_requests", count)
    provider = SimpleNamespace(
        ledger=SimpleNamespace(policy=policy, snapshot=lambda: {"attempts": []}),
        complete=forbidden,
    )
    with pytest.raises(ValueError, match="exceeds 16,000"):
        asyncio.run(
            integration.execute_assessment(request, provider, tmp_path, allow_paid=True)
        )


@pytest.mark.parametrize("exhausted", [False, True])
def test_interrupted_reference_resume_preserves_cost_and_retry_limit(
    base_fixtures, tmp_path, monkeypatch, exhausted
):
    from tests.north_star.test_runtime import FakeProvider, count_requests

    fixture = base_fixtures[0]
    request = integration.reference_request(
        source_review_requests(
            build_saved_north_star_request(fixture, fixture.scenario.weeks[0].week_id)
        )[0]
    )
    provider = FakeProvider(tmp_path, invalid=True)
    monkeypatch.setattr(integration.input_budget, "measure_requests", count_requests)

    async def resume():
        first = await provider.complete(**request)
        if exhausted:
            provider.invalidate(first, "Previously validated as malformed")
            await provider.complete(**request, retry=True)
        # The latest receipt is completed but has not yet passed schema validation.
        before = provider.ledger.snapshot()["attempts"]
        provider.invalid = False
        result = await integration.execute_assessment(
            request, provider, tmp_path, allow_paid=True
        )
        after = provider.ledger.snapshot()["attempts"]
        assert provider.generated == 2
        assert len(after) == 2
        for original, resumed in zip(before, after[: len(before)], strict=True):
            assert resumed["status"] == "invalid"
            assert resumed["raw_text"] == original["raw_text"]
            assert resumed["calculated_cost_usd"] == original["calculated_cost_usd"]
            assert resumed["calculated_cost_usd"] == 0.001
        if exhausted:
            assert result is None
        else:
            assert result is not None
            assert after[-1]["status"] == "completed"
        assert after[-1]["attempt_number"] == 2

    asyncio.run(resume())


def test_mode_metrics_do_not_credit_failed_or_unassessed_omissions():
    rows = [
        {
            "evaluation_context": "reflection",
            "record": {"status": status, "selected": None, "mode": None},
            "reference_state": reference_state,
            "selection_grade": None,
        }
        for status, reference_state in (
            ("complete", "no_supportive_example"),
            ("failed", "no_supportive_example"),
            ("not_eligible", "not_assessed"),
        )
    ]
    metrics = integration.summarize_by_mode(rows)["reflection"]
    assert metrics["weeks"] == 3
    assert metrics["no_supportive_histories"] == 2
    assert metrics["correct_omissions"] == 1
    assert metrics["failed_weeks"] == 1
    assert metrics["not_eligible_weeks"] == 1
