"""Contract checks with toy writing; these do not measure model semantics."""

import json
from copy import deepcopy
from typing import Any

import pytest

from src.north_star import assessment, review


@pytest.fixture
def source() -> review.SourceEntry:
    return review.SourceEntry(
        entry_id="writer:1",
        journal_entry="I carried her shopping home.",
        nudge_response="We cooked dinner together.\nIt was José’s recipe.",
    )


@pytest.fixture
def source_payload(source: review.SourceEntry) -> dict[str, Any]:
    return {
        "schema_version": assessment.SOURCE_SCHEMA_VERSION,
        "core_value": "benevolence",
        "results": [
            {
                "entry_id": source.entry_id,
                "action_assessment": "The writer carried shopping.",
                "value_assessment": "The action helped someone close.",
                "conflict_assessment": "No opposing behavior is described.",
                "reason_code": "observable_choice",
                "quote_source": "journal_entry",
                "evidence_quote": source.journal_entry,
            }
        ],
    }


@pytest.fixture
def candidate_payload(
    source: review.SourceEntry,
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    result = source_payload["results"][0]
    return {
        "schema_version": assessment.CANDIDATE_SCHEMA_VERSION,
        "core_value": "benevolence",
        "entry_id": source.entry_id,
        "quote_source": "journal_entry",
        "source_reason": "observable_choice",
        "quote_reason": "supported_action",
        "evaluated_quote": source.journal_entry,
        "quote_assessment": "The quotation describes carrying shopping.",
        **{
            key: result[key]
            for key in ("action_assessment", "value_assessment", "conflict_assessment")
        },
    }


def validate_candidate(
    payload: str | dict[str, Any],
    source: review.SourceEntry,
) -> assessment.CandidateAssessment:
    return assessment.validate_candidate_review(
        payload,
        core_value="benevolence",
        source=source,
        quote_source="journal_entry",
        evidence_quote=source.journal_entry,
    )


@pytest.mark.parametrize("reason", list(assessment.DECISION_BY_REASON))
def test_source_normalization_preserves_reason_and_original(
    source: review.SourceEntry,
    source_payload: dict[str, Any],
    reason: str,
) -> None:
    source_payload["results"][0]["reason_code"] = reason
    if reason != "observable_choice":
        source_payload["results"][0].update(quote_source=None, evidence_quote="")
    original = deepcopy(source_payload)
    result = assessment.validate_source_review(
        json.dumps(source_payload), core_value="benevolence", sources=[source]
    ).results[0]
    assert result.decision == assessment.DECISION_BY_REASON[reason]
    assert result.reason_code == reason
    assert source_payload == original


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing", "missing_result_entry_id"),
        ("duplicate", "duplicate_result_entry_id"),
        ("extra", "unexpected_result_entry_id"),
        ("value", "wrong_core_value"),
        ("quote", "quote_not_exact_substring"),
        ("quote_source", "quote_not_exact_substring"),
        ("contradiction", "malformed_decision"),
        ("blank_assessment", "malformed_assessment"),
        ("decision", "malformed_assessment"),
    ],
)
def test_source_batch_fails_closed(
    source: review.SourceEntry,
    source_payload: dict[str, Any],
    mutation: str,
    error: str,
) -> None:
    result = source_payload["results"][0]
    if mutation == "missing":
        source_payload["results"] = []
    elif mutation == "duplicate":
        source_payload["results"].append(deepcopy(result))
    elif mutation == "extra":
        result["entry_id"] = "someone-else:1"
    elif mutation == "value":
        source_payload["core_value"] = "universalism"
    elif mutation == "quote":
        result["evidence_quote"] = "I helped her."
    elif mutation == "quote_source":
        result["quote_source"] = "nudge_response"
    elif mutation == "contradiction":
        result["reason_code"] = "ambiguous"
    elif mutation == "blank_assessment":
        result["action_assessment"] = " "
    elif mutation == "decision":
        result["decision"] = "supportive"
    with pytest.raises(review.ReviewValidationError, match=error):
        assessment.validate_source_review(
            source_payload, core_value="benevolence", sources=[source]
        )


@pytest.mark.parametrize(
    "source_reason,quote_reason,accepted",
    [
        ("observable_choice", "supported_action", True),
        ("observable_choice", "missing_action", False),
        ("observable_choice", "not_writer_action", False),
        ("observable_choice", "wrong_value", False),
        ("observable_choice", "insufficient_context", False),
        ("same_value_conflict", "same_value_conflict", False),
        ("ambiguous", "source_not_supportive", False),
        ("wrong_value", "source_not_supportive", False),
    ],
)
def test_candidate_acceptance_is_derived(
    source: review.SourceEntry,
    candidate_payload: dict[str, Any],
    source_reason: str,
    quote_reason: str,
    accepted: bool,
) -> None:
    candidate_payload.update(source_reason=source_reason, quote_reason=quote_reason)
    assert (
        validate_candidate(json.dumps(candidate_payload), source).accepted is accepted
    )


@pytest.mark.parametrize(
    "source_reason,quote_reason",
    [
        ("same_value_conflict", "supported_action"),
        ("same_value_conflict", "source_not_supportive"),
        ("ambiguous", "supported_action"),
        ("wrong_value", "wrong_value"),
        ("observable_choice", "same_value_conflict"),
        ("observable_choice", "source_not_supportive"),
    ],
)
def test_candidate_contradictions_reject(
    source: review.SourceEntry,
    candidate_payload: dict[str, Any],
    source_reason: str,
    quote_reason: str,
) -> None:
    candidate_payload.update(source_reason=source_reason, quote_reason=quote_reason)
    with pytest.raises(review.ReviewValidationError, match="malformed_assessment"):
        validate_candidate(candidate_payload, source)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("entry_id", "someone-else:1", "wrong_entry_id"),
        ("core_value", "universalism", "wrong_core_value"),
        ("quote_source", "nudge_response", "wrong_quote_source"),
        ("evaluated_quote", "I carried her shopping home", "evaluated_quote_changed"),
        ("evaluated_quote", " I carried her shopping home.", "evaluated_quote_changed"),
        ("quote_assessment", " ", "malformed_assessment"),
        ("accepted", True, "malformed_assessment"),
    ],
)
def test_candidate_request_identity_and_quote_cannot_change(
    source: review.SourceEntry,
    candidate_payload: dict[str, Any],
    field: str,
    value: Any,
    error: str,
) -> None:
    candidate_payload[field] = value
    with pytest.raises(review.ReviewValidationError, match=error):
        validate_candidate(candidate_payload, source)


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "null",
        "[]",
        '{"results":[],"results":[]}',
        '{"results":[{"entry_id":"x","entry_id":"y"}]}',
    ],
)
def test_malformed_json_rejected_for_both_contracts(
    source: review.SourceEntry,
    raw: str,
) -> None:
    with pytest.raises(review.ReviewValidationError):
        validate_candidate(raw, source)
    with pytest.raises(review.ReviewValidationError):
        assessment.validate_source_review(
            raw, core_value="benevolence", sources=[source]
        )


def test_candidate_preserves_exact_unicode_nudge(
    source: review.SourceEntry,
    candidate_payload: dict[str, Any],
) -> None:
    assert source.nudge_response is not None
    candidate_payload.update(
        quote_source="nudge_response", evaluated_quote=source.nudge_response
    )
    result = assessment.validate_candidate_review(
        candidate_payload,
        core_value="benevolence",
        source=source,
        quote_source="nudge_response",
        evidence_quote=source.nudge_response,
    )
    assert result.evaluated_quote == source.nudge_response


@pytest.mark.parametrize("kind", ["internal_label", "missing_source", "altered_quote"])
def test_candidate_input_fidelity_checked_before_evaluation(kind: str) -> None:
    source = review.SourceEntry(entry_id="toy:1", journal_entry="I value benevolence.")
    kwargs: dict[str, Any] = {
        "core_value": "benevolence",
        "source": source,
        "quote_source": "journal_entry",
        "evidence_quote": source.journal_entry,
    }
    if kind == "missing_source":
        kwargs.update(quote_source="nudge_response", evidence_quote="I cooked.")
    elif kind == "altered_quote":
        kwargs["evidence_quote"] = "I value helping."
    with pytest.raises(review.ReviewValidationError):
        assessment.build_candidate_prompt(
            user_phrase="Care for close others",
            approved_definition="Care for others.",
            **kwargs,
        )
    with pytest.raises(review.ReviewValidationError):
        assessment.validate_candidate_review({}, **kwargs)


def test_prompts_share_rubric_and_only_expose_bounded_writing(
    source: review.SourceEntry,
) -> None:
    args = {
        "core_value": "benevolence",
        "user_phrase": "Care for close others",
        "approved_definition": "Preserve the welfare of close contacts.",
    }
    system, message = assessment.build_source_prompt(**args, sources=[source])
    candidate_system, candidate_message = assessment.build_candidate_prompt(
        **args,
        source=source,
        quote_source="journal_entry",
        evidence_quote=source.journal_entry,
    )
    assert system.startswith(assessment.SHARED_SEMANTIC_RUBRIC)
    assert candidate_system.startswith(assessment.SHARED_SEMANTIC_RUBRIC)
    payload = json.loads(message)
    assert set(payload) == {
        "core_value",
        "user_phrase",
        "approved_definition",
        "sources",
    }
    candidate = json.loads(candidate_message)
    proposed = candidate.pop("proposed_quote")
    assert candidate == payload
    assert set(proposed) == {"entry_id", "quote_source", "evidence_quote"}
    assert source.journal_entry not in system + candidate_system


@pytest.mark.parametrize(
    "schema",
    [
        assessment.source_json_schema(),
        assessment.candidate_json_schema(),
    ],
)
def test_schemas_are_strict_and_every_field_required(schema: dict[str, Any]) -> None:
    objects = [schema, *schema.get("$defs", {}).values()]
    for obj in objects:
        if obj.get("type") == "object":
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])
            assert "accepted" not in obj["properties"]
            assert "decision" not in obj["properties"]


def test_errors_never_echo_sensitive_text(
    source: review.SourceEntry,
    candidate_payload: dict[str, Any],
) -> None:
    marker = "PRIVATE_SENTINEL"
    candidate_payload["source_reason"] = marker
    with pytest.raises(review.ReviewValidationError) as caught:
        validate_candidate(candidate_payload, source)
    assert marker not in str(caught.value)
    assert source.journal_entry not in str(caught.value)
