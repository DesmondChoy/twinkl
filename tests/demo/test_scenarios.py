"""Saved persona scenario export and replay checks."""

import hashlib
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
    SavedCoachResponseFixture,
    _coach_response_sha256,
    _read_jsonl,
    _weekly_drift_input_sha256,
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
        "active-nisha"
    )
    assert {value for item in catalog.scenarios for value in item.core_values} == {
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


def test_catalog_weekly_states_match_each_replay_value(loaded_scenarios) -> None:
    catalog, fixtures = loaded_scenarios
    assert [item.scenario_id for item in catalog.scenarios] == [
        "active-nisha",
        "stable-noor",
        "persistent-lukas",
        "uncertain-wei-jun",
        "two-values-meera",
    ]
    for item in catalog.scenarios:
        for index, week in enumerate(fixtures[item.scenario_id].scenario.weeks):
            session, _ = project_scenario_week(fixtures[item.scenario_id], week.week_id)
            assert {
                value: states[index]
                for value, states in item.core_value_progression.items()
            } == session.drift_result.core_value_states
    meera = next(
        item for item in catalog.scenarios if item.scenario_id == "two-values-meera"
    )
    assert meera.core_value_progression == {
        "self_direction": ["active_drift", *["no_active_drift"] * 4],
        "tradition": ["no_active_drift"] * 5,
    }


def test_catalog_rejects_forged_core_value_states(tmp_path, monkeypatch) -> None:
    from src.demo import scenarios

    payload = json.loads((ROOT / CATALOG_PATH).read_bytes())
    meera = next(
        item for item in payload["scenarios"]
        if item["scenario_id"] == "two-values-meera"
    )
    meera["core_value_progression"]["tradition"][0] = "active_drift"
    catalog_path = tmp_path / "index.json"
    catalog_path.write_text(json.dumps(payload))
    monkeypatch.setattr(scenarios, "CATALOG_PATH", catalog_path)
    with pytest.raises(ValueError, match="catalog identity mismatch"):
        load_scenario_catalog(ROOT)


def test_retired_personas_have_no_bundled_fallbacks(loaded_scenarios) -> None:
    catalog, _ = loaded_scenarios
    assert {path.name for path in (ROOT / SCENARIO_DIRECTORY).glob("*.json")} == {
        "index.json", *(item.file for item in catalog.scenarios)
    }
    current_ids = {item.scenario_id for item in catalog.scenarios}
    assert {
        response.scenario_id
        for response in load_saved_coach_responses(ROOT).responses.values()
    } == current_ids
    nsm = json.loads((ROOT / "src/demo/north_star_replay_records.json").read_bytes())
    assert {case["scenario_id"] for case in nsm["cases"]} == current_ids


def test_saved_nudges_preserve_sources_without_claiming_live_policy(
    loaded_scenarios,
) -> None:
    _, fixtures = loaded_scenarios
    meera = fixtures["two-values-meera"]
    checks = [
        event for event in meera.trace_events
        if event.event_type == "nudge_suppression_checked"
    ]
    assert all(event.details.policy_applied is False for event in checks)
    recorded_check = next(
        event for event in checks if event.input_refs[0].id == "961a4e3f:entry:3"
    )
    assert recorded_check.details.suppressed is True
    recorded_nudge = next(
        event.details.nudge for event in meera.trace_events
        if event.event_type == "nudge_generated"
        and event.details.nudge.journal_entry_id == "961a4e3f:entry:3"
    )
    assert recorded_nudge.text == "Which part of the day stuck with you more?"


def test_required_drift_progressions_are_preserved(loaded_scenarios) -> None:
    _, fixtures = loaded_scenarios
    assert all(
        week.expected_delivery_state == "no_active_drift"
        for week in fixtures["stable-noor"].scenario.weeks
    )
    nisha = fixtures["active-nisha"]
    assert [week.expected_delivery_state for week in nisha.scenario.weeks] == [
        "no_active_drift",
        "no_active_drift",
        "no_active_drift",
        "active_drift",
        "no_active_drift",
    ]
    assert [
        (d.core_value, d.onset_t_index, d.confirmation_t_index)
        for d in nisha.scenario.drift_result.drifts
    ] == [("universalism", 5, 6)]
    lukas = fixtures["persistent-lukas"]
    assert [week.expected_delivery_state for week in lukas.scenario.weeks] == [
        "active_drift",
        "active_drift",
        "active_drift",
        "active_drift",
        "no_active_drift",
    ]
    assert len(lukas.scenario.drift_result.drifts) == 1
    assert lukas.scenario.drift_result.drifts[0].supporting_t_indices == [1, 2, 3, 4, 5]
    assert lukas.scenario.drift_result.drifts[0].termination_verdict == "not_conflict"
    uncertain = fixtures["uncertain-wei-jun"].scenario.drift_result
    assert uncertain.delivery_state == "insufficient_evidence"
    assert uncertain.drifts == []
    meera = fixtures["two-values-meera"]
    key = next(w for w in meera.scenario.weeks if w.week_start == "2025-11-10")
    session, _ = project_scenario_week(meera, key.week_id)
    assert session.drift_result.core_value_states == {
        "self_direction": "active_drift",
        "tradition": "no_active_drift",
    }


def test_deployed_persona_roster_and_key_week_rules(loaded_scenarios) -> None:
    _, fixtures = loaded_scenarios
    expected = {
        "stable-noor": ("02fb94f3", "2025-05-19", "no_active_drift"),
        "active-nisha": ("5fa8b540", "2025-03-03", "active_drift"),
        "persistent-lukas": ("a24b8d8f", "2025-06-30", "active_drift"),
        "uncertain-wei-jun": ("8f83c818", "2025-06-30", "insufficient_evidence"),
        "two-values-meera": ("961a4e3f", "2025-11-10", "active_drift"),
    }
    assert set(fixtures) == set(expected)
    assert sum(len(f.scenario.weeks) for f in fixtures.values()) == 27
    for scenario_id, (persona_id, week_start, state) in expected.items():
        fixture = fixtures[scenario_id]
        key = next(w for w in fixture.scenario.weeks if w.week_start == week_start)
        assert fixture.scenario.persona_id == persona_id
        assert key.expected_delivery_state == state


def test_all_weeks_reuse_exact_source_bound_coach_digests(loaded_scenarios) -> None:
    from src.coach.schemas import WeeklyDigest

    _, fixtures = loaded_scenarios
    saved_responses = load_saved_coach_responses(ROOT)
    expected_keys = {
        f"{fixture.scenario.scenario_id}::{week.week_start}"
        for fixture in fixtures.values()
        for week in fixture.scenario.weeks
    }
    assert len(expected_keys) == 27
    assert set(saved_responses.responses) == expected_keys
    for fixture in fixtures.values():
        for week in fixture.scenario.weeks:
            saved = saved_responses.responses[
                f"{fixture.scenario.scenario_id}::{week.week_start}"
            ]
            generation = saved.generation
            assert generation is not None
            assert generation.prompt_version == "4.4"
            assert generation.model_contract.provider == "openai"
            assert generation.model_contract.model == "gpt-5.6-luna"
            assert generation.model_contract.reasoning_effort == "none"
            assert (
                generation.prompt_sha256
                == hashlib.sha256(generation.prompt.encode("utf-8")).hexdigest()
            )
            assert json.loads(generation.raw_output) == saved.narrative.model_dump(
                mode="json"
            )
            assert generation.response_sha256 == _coach_response_sha256(saved.narrative)
            assert 1 <= generation.attempt_count <= 4
            assert len(generation.call_metrics) == len(generation.diagnostic_paths)
            assert len(generation.call_metrics) == generation.attempt_count
            for metric, path in zip(
                generation.call_metrics, generation.diagnostic_paths, strict=True
            ):
                diagnostic = json.loads((ROOT / path).read_text())
                assert diagnostic["llm_call"] == metric.model_dump(mode="json")
                assert metric.model == "gpt-5.6-luna"
                assert metric.reasoning_effort == "none"
                assert metric.status == "completed"
                assert metric.response_id
                assert metric.input_tokens > 0
                assert metric.output_tokens > 0
            accepted = json.loads((ROOT / generation.diagnostic_paths[-1]).read_text())
            assert accepted["accepted"] is True
            assert accepted["raw_output"] == generation.raw_output
            assert all(check["passed"] for check in accepted["validation"]["checks"])
            coach_events = [
                event
                for event in fixture.trace_events
                if event.event_type == "weekly_coach_generated"
                and event.event_id in week.event_ids
            ]
            assert len(coach_events) == 1
            event = coach_events[0]
            comparison = event.details.comparison
            baseline = comparison.without_north_star if comparison else None
            assert event.prompt == (baseline.prompt if baseline else generation.prompt)
            assert event.raw_response == (
                baseline.raw_output if baseline else generation.raw_output
            )
            assert event.details.narrative == (
                baseline.narrative if baseline else saved.narrative
            )
            assert event.model_contract == generation.model_contract
            generated_digest = WeeklyDigest.model_validate_json(
                (ROOT / generation.generated_response_path).read_bytes()
            )
            digest_event = next(
                event
                for event in fixture.trace_events
                if event.event_type == "weekly_digest_built"
                and event.event_id in week.event_ids
            )
            expected_digest = generated_digest.model_copy(update={
                "coach_narrative": baseline.narrative,
                "validation": baseline.validation,
            }) if baseline else generated_digest
            assert digest_event.details.digest == expected_digest
            assert digest_event.event_id == generation.weekly_digest_event_id
            assert _weekly_drift_input_sha256(generated_digest) == (
                generation.weekly_drift_input_sha256
            )
            assert digest_event.details.coach_unavailable_reason is None


def test_pre_refresh_coach_responses_remain_exactly_preserved() -> None:
    plan = json.loads(
        (
            ROOT
            / "logs/experiments/reports/demo_persona_replacement_20260908/plan.json"
        ).read_text()
    )
    refresh_root = ROOT / "logs/experiments/reports/coach_voice_refresh_20260909"
    original_bytes = (
        refresh_root / "original_coach_digest_responses.json"
    ).read_bytes()
    refresh_plan = json.loads((refresh_root / "plan.json").read_text())
    assert hashlib.sha256(original_bytes).hexdigest() == refresh_plan["original_sha256"]
    responses = SavedCoachResponseFixture.model_validate_json(original_bytes).responses
    assert len(plan["retained_responses"]) == 17
    for key, original in plan["retained_responses"].items():
        assert responses[key].model_dump(mode="json") == original


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
    source, payload = _scenario_payload("stable-noor")
    expected_hash = next(
        item["content_sha256"]
        for item in json.loads((ROOT / CATALOG_PATH).read_text())["scenarios"]
        if item["scenario_id"] == "stable-noor"
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
            "temporal order|chronological t_index",
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
    _, payload = _scenario_payload("stable-noor")
    mutation(payload)
    changed = tmp_path / f"{name}.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error_type, match=match):
        load_scenario_file(changed, root=ROOT)


@pytest.fixture(scope="module")
def weekly_sources():
    from src.demo.scenarios import ATTEMPTS_PATH

    return {
        "prompt_rows": _read_jsonl(ROOT / PROMPTS_PATH),
        "response_rows": _read_jsonl(ROOT / RESPONSES_PATH),
        "attempt_rows": _read_jsonl(ROOT / ATTEMPTS_PATH),
    }


def _historical_selection(persona_id: str, scenario_id: str, week: str):
    from src.demo.scenarios import ScenarioSelection

    return ScenarioSelection(
        persona_id=persona_id,
        scenario_id=scenario_id,
        role="no_active_drift",
        title="Source compatibility check",
        description="Source compatibility check",
        summary="Source compatibility check",
        coach_week_start=week,
    )


def test_v4_failed_receipt_preserves_effective_abstain(weekly_sources) -> None:
    selection = next(s for s in SELECTIONS if s.persona_id == "8f83c818")
    fixture = build_scenario_fixture(
        ROOT, selection, include_north_star=False, **weekly_sources
    )
    event = next(
        event
        for event in fixture.trace_events
        if event.event_type == "weekly_review_completed"
        and event.details.receipt.week_start == "2025-06-23"
    )
    receipt = event.details.receipt
    assert receipt.status == "invalid"
    assert receipt.prompt_version == "4.0"
    assert receipt.validation_error == "Evidence quote is not present in t_index=8"
    assert receipt.assessments[0].verdict == "conflict"
    assert receipt.decisions[0].verdict == "abstain"
    assert receipt.decisions[0].review_status == "invalid"
    assert event.raw_response["attempt"]["response_id"] == receipt.response_id
    assert fixture.scenario.drift_result.delivery_state == "insufficient_evidence"


@pytest.mark.parametrize("missing_provenance", [False, True])
def test_incompatible_coach_source_is_omitted(
    weekly_sources, missing_provenance
) -> None:
    selection = SELECTIONS[0]
    payload = load_saved_coach_responses(ROOT).model_dump(mode="json")
    key = f"{selection.scenario_id}::{selection.coach_week_start}"
    if missing_provenance:
        payload["responses"][key]["generation"] = None
    else:
        payload["responses"][key]["generation"]["weekly_drift_input_sha256"] = "0" * 64
    fixture = build_scenario_fixture(
        ROOT,
        selection,
        include_north_star=False,
        **weekly_sources,
        coach_responses=SavedCoachResponseFixture.model_validate(payload),
    )
    key_week = next(
        week
        for week in fixture.scenario.weeks
        if week.week_start == selection.coach_week_start
    )
    assert not any(
        event.event_type == "weekly_coach_generated"
        and event.event_id in key_week.event_ids
        for event in fixture.trace_events
    )
    digest_event = next(
        event
        for event in fixture.trace_events
        if event.event_type == "weekly_digest_built"
        and event.details.digest.week_start == selection.coach_week_start
    )
    assert digest_event.details.digest.coach_narrative is None
    reason = digest_event.details.coach_unavailable_reason
    if missing_provenance:
        assert "input provenance is unavailable" in reason
    else:
        assert "Weekly Drift input differs" in reason
        assert "0" * 64 in reason
        assert _weekly_drift_input_sha256(digest_event.details.digest) in reason


@pytest.mark.parametrize(
    "mutation",
    [
        "variant_hash",
        "request_input",
        "response_hash",
        "duplicate_response",
        "attempt_hash",
        "raw_response",
        "decision",
        "empty_responses",
    ],
)
def test_v4_builder_rejects_mismatched_source_binding(weekly_sources, mutation) -> None:
    import copy

    from src.demo.scenarios import _sha256_json

    sources = copy.deepcopy(weekly_sources)
    selection = _historical_selection("23d101f8", "stable-meera", "2025-09-15")
    request = next(
        row
        for row in sources["prompt_rows"]
        if row["request"]["persona_id"] == selection.persona_id
    )
    key = f"definitions:1:{request['case_id']}"
    response = next(
        row for row in sources["response_rows"] if row["request_key"] == key
    )
    attempt = next(
        row
        for row in sources["attempt_rows"]
        if row["request_key"] == key and row["event"] == "finished"
    )
    if mutation == "variant_hash":
        request["variants"]["definitions"]["request_sha256"] = "0" * 64
    elif mutation == "request_input":
        variant = request["variants"]["definitions"]
        variant["input_data"] = "{}"
        variant["request_sha256"] = _sha256_json(
            {k: v for k, v in variant.items() if k != "request_sha256"}
        )
    elif mutation == "response_hash":
        response["request_sha256"] = "0" * 64
    elif mutation == "duplicate_response":
        sources["response_rows"].append(copy.deepcopy(response))
    elif mutation == "attempt_hash":
        attempt["request_sha256"] = "0" * 64
    elif mutation == "raw_response":
        attempt["raw_text"] = '{"assessments": []}'
    elif mutation == "decision":
        response["decisions"][0]["verdict"] = "abstain"
    else:
        sources["response_rows"] = []
    with pytest.raises(ValueError):
        build_scenario_fixture(ROOT, selection, include_north_star=False, **sources)
