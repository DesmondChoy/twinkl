"""Cross-language contract checks for Experience and Inspect."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.coach.weekly_drift_runtime import load_onboarding_core_values
from src.demo.canonical_fixture import build_canonical_fixture
from src.demo.contracts import (
    CONTRACT_VERSION,
    SCHEMA_ID,
    ContractFixtureSet,
)

FIXTURE_PATH = Path(
    "frontend/onboarding/src/contracts/experience_inspect_v1.fixture.json"
)
SCHEMA_PATH = Path(
    "frontend/onboarding/src/contracts/experience_inspect_v1.schema.json"
)


def _fixture_payload() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def test_canonical_fixture_matches_python_models_and_generator() -> None:
    payload = _fixture_payload()

    parsed = ContractFixtureSet.model_validate(payload)
    generated = build_canonical_fixture().model_dump(mode="json")

    assert parsed.schema_version == CONTRACT_VERSION
    assert payload == generated
    assert {event.event_type for event in parsed.trace_events} == {
        "profile_confirmed",
        "journal_entry_submitted",
        "nudge_suppression_checked",
        "nudge_decided",
        "nudge_generated",
        "weekly_review_requested",
        "weekly_review_completed",
        "drift_detected",
        "weekly_digest_built",
        "weekly_coach_generated",
    }
    assert {event.status for event in parsed.trace_events} == {
        "complete",
        "reused",
        "refused",
        "invalid",
        "failed",
    }
    main_events = [
        event
        for event in parsed.trace_events
        if event.event_id in parsed.session.trace_event_ids
    ]
    assert all(event.input_refs or event.result_refs for event in main_events)


def test_checked_in_json_schema_matches_python_models() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())

    assert schema == ContractFixtureSet.model_json_schema()
    assert schema["$id"] == SCHEMA_ID


def test_profile_fixture_matches_runtime_import_contract(tmp_path: Path) -> None:
    profile = _fixture_payload()["session"]["profile"]
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))

    assert load_onboarding_core_values(
        profile_path,
        persona_id="persona-demo",
    ) == ["benevolence"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(schema_version="experience-inspect-v2"),
        lambda payload: payload["trace_events"][0].pop("input_hash"),
        lambda payload: payload["trace_events"][0].update(
            parent_event_id="missing-event"
        ),
        lambda payload: payload["session"]["weekly_reviewer_decisions"][0].update(
            prediction=-1,
            uncertainty=0.08,
        ),
    ],
)
def test_python_contract_rejects_incompatible_payloads(mutation) -> None:
    payload = deepcopy(_fixture_payload())
    mutation(payload)

    with pytest.raises(ValidationError):
        ContractFixtureSet.model_validate(payload)


def test_weekly_review_rejects_changed_model_contract() -> None:
    payload = _fixture_payload()
    weekly_event = next(
        event
        for event in payload["trace_events"]
        if event["event_type"] == "weekly_review_completed"
    )
    weekly_event["model_contract"]["reasoning_effort"] = "medium"

    with pytest.raises(ValidationError, match="gpt-5.6-luna at low"):
        ContractFixtureSet.model_validate(payload)


def test_safe_errors_reject_provider_secrets() -> None:
    payload = _fixture_payload()
    failed_event = next(
        event for event in payload["trace_events"] if event["status"] == "failed"
    )
    failed_event["error"]["message"] = "authorization: Bearer sk-secret"

    with pytest.raises(ValidationError, match="provider secrets"):
        ContractFixtureSet.model_validate(payload)
