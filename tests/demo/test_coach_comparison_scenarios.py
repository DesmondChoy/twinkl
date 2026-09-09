"""Saved paired Coach responses remain source-bound through export and replay."""

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from src.coach.demo_comparison import validate_saved_comparison
from src.demo.contracts import TraceEvent
from src.demo.scenarios import (
    COACH_COMPARISONS_PATH,
    _validate_fixture_semantics,
    attach_saved_coach_comparisons,
    load_saved_coach_comparisons,
    load_scenario_catalog,
    project_scenario_week,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compared_scenarios():
    return load_scenario_catalog(ROOT)[1]


def test_all_eligible_pairs_replay_the_exact_default_and_inspect_receipts(
    compared_scenarios,
) -> None:
    saved = load_saved_coach_comparisons(ROOT).comparisons
    assert len(saved) == 22
    seen = set()
    source_types = Counter()
    no_selection = 0
    for fixture in compared_scenarios.values():
        assert COACH_COMPARISONS_PATH.as_posix() in (
            fixture.scenario.manifest.source_files
        )
        for week in fixture.scenario.weeks:
            session, visible = project_scenario_week(fixture, week.week_id)
            events = [event for event in visible if event.event_id in week.event_ids]
            coach = next(e for e in events if e.event_type == "weekly_coach_generated")
            moment = next(e for e in events if e.event_type == "north_star_reviewed")
            pair = coach.details.comparison
            record = moment.details.record
            if record.status != "complete" or record.selected is None:
                no_selection += 1
                assert pair is None
                continue
            assert pair is not None
            key = f"{fixture.scenario.scenario_id}::{week.week_start}"
            seen.add(key)
            assert pair == saved[key]
            assert session.weekly_digest is not None
            validate_saved_comparison(
                pair, session.weekly_digest, record, fixture.scenario.scenario_id
            )
            baseline = pair.without_north_star
            # Independent generations can arrive at the same reflective question.
            assert baseline.narrative != pair.with_north_star.narrative
            assert session.weekly_digest.coach_narrative == baseline.narrative
            assert session.weekly_digest.validation == baseline.validation
            assert coach.details.narrative == baseline.narrative
            assert coach.prompt == baseline.prompt
            assert coach.raw_response == baseline.raw_output
            source_types[pair.north_star_context.source_type] += 1
            inputs = []
            instructions = []
            for arm in (pair.without_north_star, pair.with_north_star):
                trusted, data = arm.base_prompt.split("\nUNTRUSTED INPUT DATA\n")
                instructions.append(trusted)
                inputs.append(json.loads(data))
                assert json.loads(arm.raw_output) == arm.narrative.model_dump()
                diagnostics = [
                    json.loads((ROOT / path).read_text())
                    for path in arm.diagnostic_paths
                ]
                accepted = diagnostics[-1]
                assert accepted["accepted"] is True
                assert accepted["prompt"] == arm.prompt
                assert accepted["raw_output"] == arm.raw_output
                assert accepted["validation"] == arm.validation.model_dump()
                assert [row["llm_call"] for row in diagnostics] == [
                    metric.model_dump() for metric in arm.call_metrics
                ]
            assert instructions[0] == instructions[1]
            assert inputs[0].pop("north_star_context") is None
            assert inputs[1].pop("north_star_context") == (
                pair.north_star_context.model_dump(exclude_none=True)
            )
            assert inputs[0] == inputs[1]
    assert seen == set(saved)
    assert no_selection == 5
    assert source_types == {"journal_entry": 21, "nudge_response": 1}


def test_comparison_is_never_a_live_session_event(compared_scenarios) -> None:
    fixture = compared_scenarios["active-nisha"]
    event = next(
        e for e in fixture.trace_events
        if e.event_type == "weekly_coach_generated" and e.details.comparison
    )
    payload = event.model_dump(mode="json")
    payload["source"] = "live_run"
    with pytest.raises(ValidationError, match="only in saved replay"):
        TypeAdapter(TraceEvent).validate_python(payload)


def test_default_event_cannot_point_at_the_other_response(compared_scenarios) -> None:
    fixture = compared_scenarios["active-nisha"]
    event = next(
        e for e in fixture.trace_events
        if e.event_type == "weekly_coach_generated" and e.details.comparison
    )
    payload = event.model_dump(mode="json")
    payload["prompt"] = payload["details"]["comparison"]["with_north_star"]["prompt"]
    with pytest.raises(ValidationError, match="without-context response"):
        TypeAdapter(TraceEvent).validate_python(payload)


def test_scenario_rejects_missing_pair_from_its_saved_receipt(
    compared_scenarios,
) -> None:
    fixture = compared_scenarios["active-nisha"].model_copy(deep=True)
    event = next(
        e for e in fixture.trace_events
        if e.event_type == "weekly_coach_generated" and e.details.comparison
    )
    event.details.comparison = None
    with pytest.raises(ValueError, match="differs from saved pair"):
        _validate_fixture_semantics(fixture, root=ROOT)


def test_scenario_rejects_changed_selection_context(compared_scenarios) -> None:
    fixture = compared_scenarios["active-nisha"].model_copy(deep=True)
    event = next(
        e for e in fixture.trace_events
        if e.event_type == "weekly_coach_generated" and e.details.comparison
    )
    assert event.details.comparison is not None
    event.details.comparison.north_star_context.source_text = "Another source."
    with pytest.raises(ValueError, match="differs from saved pair"):
        _validate_fixture_semantics(fixture, root=ROOT)


def test_missing_comparison_file_keeps_existing_replay(
    compared_scenarios, tmp_path,
) -> None:
    fixture = compared_scenarios["active-nisha"]
    assert load_saved_coach_comparisons(tmp_path).comparisons == {}
    assert attach_saved_coach_comparisons(fixture, root=tmp_path) is fixture
