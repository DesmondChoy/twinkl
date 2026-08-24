"""Saved persona scenario export and replay checks."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.demo.scenarios import (
    CATALOG_PATH,
    PROMPTS_PATH,
    RESPONSES_PATH,
    SCENARIO_DIRECTORY,
    SELECTIONS,
    _read_jsonl,
    build_scenario_fixture,
    load_saved_coach_responses,
    load_scenario_catalog,
    load_scenario_file,
    project_scenario_week,
)
from src.drift_review_app.data import load_review_data

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loaded_scenarios():
    return load_scenario_catalog(ROOT)


def test_catalog_covers_five_diverse_personas(loaded_scenarios) -> None:
    catalog, fixtures = loaded_scenarios

    assert len(catalog.scenarios) == 5
    assert len(fixtures) == 5
    assert sum(item.recommended for item in catalog.scenarios) == 1
    assert next(item for item in catalog.scenarios if item.recommended).scenario_id == (
        "two-values-lukas"
    )
    assert {value for item in catalog.scenarios for value in item.core_values} == {
        "achievement",
        "conformity",
        "power",
        "security",
        "self_direction",
        "tradition",
        "universalism",
    }
    assert {item.culture for item in catalog.scenarios} == {
        "East Asian",
        "Middle Eastern",
        "South Asian",
        "Western European",
    }
    assert all(
        fixture.scenario.profile.provenance.source == "synthetic_persona_projection"
        for fixture in fixtures.values()
    )
    reviewer_inputs = [
        event.details.request.model_dump_json()
        for fixture in fixtures.values()
        for event in fixture.trace_events
        if event.event_type == "weekly_review_requested"
    ]
    assert all(
        marker not in reviewer_input
        for reviewer_input in reviewer_inputs
        for marker in (
            "**Tone**",
            "Reflection Mode",
            "final_conflict",
            "resolution_method",
            "LLM-Judge",
        )
    )


def test_required_drift_progressions_are_preserved(loaded_scenarios) -> None:
    _, fixtures = loaded_scenarios

    stable = fixtures["stable-meera"].scenario.drift_result
    assert stable.delivery_state == "no_active_drift"
    assert stable.drifts == []

    active = fixtures["active-wei-jun"].scenario.drift_result
    assert active.delivery_state == "active_drift"
    assert [
        (drift.core_value, drift.onset_t_index, drift.confirmation_t_index)
        for drift in active.drifts
    ] == [("universalism", 8, 9)]

    ended = fixtures["recovered-marc"].scenario.drift_result
    assert ended.delivery_state == "no_active_drift"
    assert ended.drifts[0].termination_verdict == "not_conflict"

    uncertain_fixture = fixtures["uncertain-noor"]
    assert "insufficient_evidence" in {
        week.expected_delivery_state for week in uncertain_fixture.scenario.weeks
    }
    assert any(
        decision.verdict == "abstain"
        for decision in uncertain_fixture.scenario.weekly_reviewer_decisions
    )

    two_values = fixtures["two-values-lukas"].scenario.drift_result
    assert two_values.delivery_state == "insufficient_evidence"
    assert two_values.core_value_states == {
        "conformity": "no_active_drift",
        "self_direction": "insufficient_evidence",
    }


def test_deployed_persona_roster_and_key_week_rules(loaded_scenarios) -> None:
    _, fixtures = loaded_scenarios
    expected = {
        "two-values-lukas": ("11de77e8", "2025-10-13", "insufficient_evidence"),
        "stable-meera": ("23d101f8", "2025-09-15", "no_active_drift"),
        "active-wei-jun": ("8f83c818", "2025-06-30", "active_drift"),
        "recovered-marc": ("988d1a65", "2025-03-17", "no_active_drift"),
        "uncertain-noor": ("02fb94f3", "2025-04-14", "insufficient_evidence"),
    }

    for scenario_id, (persona_id, week_start, state) in expected.items():
        fixture = fixtures[scenario_id]
        key_index = next(
            index
            for index, week in enumerate(fixture.scenario.weeks)
            if week.week_start == week_start
        )
        assert fixture.scenario.persona_id == persona_id
        assert fixture.scenario.weeks[key_index].expected_delivery_state == state

    assert fixtures["stable-meera"].scenario.weeks[-1].week_start == "2025-09-15"
    assert all(
        week.expected_delivery_state != "active_drift"
        for week in fixtures["active-wei-jun"].scenario.weeks[:5]
    )
    marc_weeks = fixtures["recovered-marc"].scenario.weeks
    assert marc_weeks[4].expected_delivery_state == "active_drift"
    assert marc_weeks[5].expected_delivery_state == "no_active_drift"
    assert fixtures["uncertain-noor"].scenario.weeks[0].week_start == "2025-04-14"
    assert fixtures["two-values-lukas"].scenario.weeks[-1].week_start == (
        "2025-10-13"
    )


def test_five_key_weeks_have_the_exact_evaluated_coach_digests(
    loaded_scenarios,
) -> None:
    _, fixtures = loaded_scenarios
    saved_responses = load_saved_coach_responses(ROOT)
    manifest = json.loads(
        (
            ROOT
            / "logs/experiments/reports/coach_digest_sample_20260824/"
            "judge_sample_manifest.json"
        ).read_text(encoding="utf-8")
    )
    coach_events = [
        event
        for fixture in fixtures.values()
        for event in fixture.trace_events
        if event.event_type == "weekly_coach_generated"
    ]

    assert len(saved_responses.responses) == 5
    assert len(coach_events) == 5
    assert len(manifest) == 5

    for selection in SELECTIONS:
        fixture = fixtures[selection.scenario_id]
        key_week_index = next(
            index
            for index, week in enumerate(fixture.scenario.weeks)
            if week.week_start == selection.coach_week_start
        )
        key_week = fixture.scenario.weeks[key_week_index]
        event = next(
            event
            for event in fixture.trace_events
            if event.event_type == "weekly_coach_generated"
        )
        digest_event = next(
            event
            for event in fixture.trace_events
            if event.event_type == "weekly_digest_built"
            and event.event_id in key_week.event_ids
        )
        narrative = digest_event.details.digest.coach_narrative
        validation = digest_event.details.digest.validation
        manifest_entry = next(
            item
            for item in manifest
            if item["provenance"]["scenario_id"] == selection.scenario_id
        )

        assert event.event_id in key_week.event_ids
        assert event.source == "saved_replay"
        assert event.model_contract is not None
        assert event.model_contract.model == "gpt-5.6-luna"
        assert event.model_contract.reasoning_effort == "none"
        assert event.prompt is not None
        assert event.raw_response is not None
        assert narrative == event.details.narrative
        assert narrative is not None
        assert narrative.model_dump(mode="json") == manifest_entry["narrative"]
        assert validation == event.details.validation
        assert validation is not None
        assert all(check.passed for check in validation.checks)
        assert manifest_entry["provenance"]["scenario_bundle_content_sha256"]

        for earlier_index in range(key_week_index):
            earlier, earlier_events = project_scenario_week(
                fixture,
                fixture.scenario.weeks[earlier_index].week_id,
            )
            assert earlier.weekly_digest is not None
            assert earlier.weekly_digest.coach_narrative is None
            assert all(
                prior.event_type != "weekly_coach_generated"
                for prior in earlier_events
            )


def test_checked_in_scenarios_match_deterministic_builder(
    loaded_scenarios,
) -> None:
    _, fixtures = loaded_scenarios
    data = load_review_data(ROOT)
    prompts = _read_jsonl(ROOT / PROMPTS_PATH)
    responses = _read_jsonl(ROOT / RESPONSES_PATH)

    for selection in SELECTIONS:
        rebuilt = build_scenario_fixture(
            ROOT,
            selection,
            data=data,
            prompt_rows=prompts,
            response_rows=responses,
        )
        assert rebuilt == fixtures[selection.scenario_id]


def test_week_projection_never_reveals_future_entries_or_results(
    loaded_scenarios,
) -> None:
    _, fixtures = loaded_scenarios

    for fixture in fixtures.values():
        for week_index, week in enumerate(fixture.scenario.weeks):
            session, events = project_scenario_week(fixture, week.week_id)
            visible_week_ids = {
                prior.week_id for prior in fixture.scenario.weeks[: week_index + 1]
            }
            future_entries = [
                entry
                for later in fixture.scenario.weeks[week_index + 1 :]
                for entry in fixture.scenario.journal_entries
                if entry.journal_entry_id in later.journal_entry_ids
            ]
            visible_json = json.dumps(
                {
                    "session": session.model_dump(mode="json"),
                    "events": [event.model_dump(mode="json") for event in events],
                },
                ensure_ascii=False,
            )

            assert all(
                decision.week_end <= week.week_end
                for decision in session.weekly_reviewer_decisions
            )
            assert all(later.content not in visible_json for later in future_entries)
            assert {
                event.details.request.week_start
                for event in events
                if event.event_type == "weekly_review_requested"
            } == {
                prior.week_start
                for prior in fixture.scenario.weeks
                if prior.week_id in visible_week_ids
            }


def _scenario_payload(scenario_id: str) -> tuple[Path, dict]:
    catalog = json.loads((ROOT / CATALOG_PATH).read_text(encoding="utf-8"))
    item = next(
        item for item in catalog["scenarios"] if item["scenario_id"] == scenario_id
    )
    path = ROOT / SCENARIO_DIRECTORY / item["file"]
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_loader_rejects_changed_scenario_content(tmp_path: Path) -> None:
    source, payload = _scenario_payload("stable-meera")
    expected_hash = next(
        item["content_sha256"]
        for item in json.loads((ROOT / CATALOG_PATH).read_text())["scenarios"]
        if item["scenario_id"] == "stable-meera"
    )
    payload["scenario"]["title"] += " changed"
    changed = tmp_path / source.name
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_scenario_file(
            changed,
            root=ROOT,
            expected_content_sha256=expected_hash,
        )


@pytest.mark.parametrize(
    ("name", "mutation", "error_type", "match"),
    [
        (
            "missing-provenance",
            lambda payload: payload["scenario"]["profile"].pop("provenance"),
            ValidationError,
            "provenance",
        ),
        (
            "missing-source",
            lambda payload: payload["scenario"]["manifest"]["source_files"].pop(),
            ValueError,
            "source provenance",
        ),
        (
            "wrong-model",
            lambda payload: payload["scenario"]["manifest"]["model_contract"].update(
                reasoning_effort="medium"
            ),
            ValidationError,
            "Luna-low",
        ),
        (
            "invalid-time",
            lambda payload: payload["scenario"]["journal_entries"].reverse(),
            ValueError,
            "temporal order",
        ),
    ],
)
def test_loader_rejects_invalid_manifest_or_time(
    tmp_path: Path,
    name: str,
    mutation,
    error_type: type[Exception],
    match: str,
) -> None:
    _, payload = _scenario_payload("stable-meera")
    mutation(payload)
    changed = tmp_path / f"{name}.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error_type, match=match):
        load_scenario_file(changed, root=ROOT)
