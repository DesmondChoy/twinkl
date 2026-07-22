"""Deterministic canonical fixture for both contract implementations."""

from __future__ import annotations

from typing import Any

from src.coach.schemas import (
    CoachNarrative,
    DigestValidation,
    EvidenceSnippet,
    ValidationCheck,
    WeeklyDigest,
)
from src.demo.contracts import (
    BWS_OBJECT_ORDER,
    BWS_SETS,
    CONTRACT_VERSION,
    CORE_VALUE_ORDER,
    ApiErrorResponse,
    ContractFixtureSet,
    ExperienceSession,
    JournalEntry,
    JournalEntrySubmitRequest,
    ModelContract,
    NudgeInteraction,
    OnboardingProfile,
    SafeError,
    ScenarioBundle,
    ScenarioLoadRequest,
    ScenarioManifest,
    ScenarioWeek,
    SessionCreatedResponse,
    SessionCreateRequest,
    SessionSelection,
    TraceReadRequest,
    WeeklyDriftReviewerDecisionContract,
)
from src.drift_detector import detect_drift
from src.weekly_drift_reviewer import (
    VerifierAssessment,
    WeeklyDriftReviewerDecision,
    WeeklyDriftReviewerEntry,
    WeeklyDriftReviewerReceipt,
    WeeklyDriftReviewerRequest,
)

SESSION_ID = "session-demo"
PERSONA_ID = "persona-demo"
LUNA_LOW = ModelContract(
    provider="openai",
    model="gpt-5.6-luna",
    reasoning_effort="low",
)


def _round(value: float, digits: int = 8) -> float:
    """Match the Profile's eight-decimal rounding for these fixture values."""
    return round(value + 2.220446049250313e-16, digits)


def _profile() -> dict[str, Any]:
    selected_pairs = (
        ("benevolence", "tradition"),
        ("benevolence", "power"),
        ("achievement", "power"),
        ("self_direction", "conformity"),
        ("universalism_nature", "hedonism"),
        ("benevolence", "universalism_social"),
        ("tradition", "achievement"),
        ("benevolence", "stimulation"),
        ("benevolence", "security"),
        ("universalism_social", "self_direction"),
        ("benevolence", "power"),
    )
    appearances = {value: 0 for value in BWS_OBJECT_ORDER}
    best_counts = {value: 0 for value in BWS_OBJECT_ORDER}
    worst_counts = {value: 0 for value in BWS_OBJECT_ORDER}
    responses = []
    for set_number, (items, choices) in enumerate(
        zip(BWS_SETS, selected_pairs, strict=True),
        start=1,
    ):
        selected_best, selected_worst = choices
        for item in items:
            appearances[item] += 1
        best_counts[selected_best] += 1
        worst_counts[selected_worst] += 1
        responses.append(
            {
                "set_number": set_number,
                "items": list(items),
                "item_order_shown": list(reversed(items)),
                "selected_best": selected_best,
                "selected_worst": selected_worst,
                "response_time_ms": 999 + set_number,
            }
        )

    net_counts = {
        value: best_counts[value] - worst_counts[value] for value in BWS_OBJECT_ORDER
    }
    scores = {
        value: _round(net_counts[value] / appearances[value])
        for value in BWS_OBJECT_ORDER
    }
    profile_scores = {
        value: (
            _round((scores["universalism_nature"] + scores["universalism_social"]) / 2)
            if value == "universalism"
            else scores[value]
        )
        for value in CORE_VALUE_ORDER
    }
    minimum = min(profile_scores.values())
    shifted = [profile_scores[value] - minimum + 1 for value in CORE_VALUE_ORDER]
    total = sum(shifted)
    weights: dict[str, float] = {}
    assigned = 0.0
    for index, value in enumerate(CORE_VALUE_ORDER):
        if index == len(CORE_VALUE_ORDER) - 1:
            weights[value] = _round(1 - assigned)
        else:
            weights[value] = _round(shifted[index] / total)
            assigned += weights[value]

    highest = max(profile_scores.values())
    lowest = min(profile_scores.values())
    top_values = [
        value
        for value in CORE_VALUE_ORDER
        if abs(profile_scores[value] - highest) <= 1e-8
    ]
    bottom_values = [
        value
        for value in CORE_VALUE_ORDER
        if abs(profile_scores[value] - lowest) <= 1e-8
    ]
    return {
        "schema_version": 2,
        "onboarding_version": "2.1.0",
        "instrument": "svbws_lee_soutar_louviere_2008_ui_adaptation_v2",
        "scoring_method": "best_minus_worst_divided_by_appearances_v1",
        "user_id": PERSONA_ID,
        "session_id": SESSION_ID,
        "started_at": "2026-07-22T10:00:00.000Z",
        "timestamp": "2026-07-22T10:05:00.000Z",
        "bws_responses": responses,
        "bws_results": {
            "appearances": appearances,
            "best_counts": best_counts,
            "worst_counts": worst_counts,
            "net_counts": net_counts,
            "scores": scores,
        },
        "value_profile": {
            "method": "mean_universalism_facets_then_shift_normalize_v1",
            "scores": profile_scores,
            "weights": weights,
            "top_values": top_values,
            "bottom_values": bottom_values,
        },
        "top_values": top_values,
        "goal_category": "relationships",
        "user_confirmed": True,
        "provenance": {
            "source": "react_onboarding_poc",
            "set_order_randomized": True,
            "card_order_randomized": True,
        },
    }


def _review_decisions() -> list[WeeklyDriftReviewerDecisionContract]:
    rows = (
        (
            0,
            "2026-07-06",
            "Cancelled dinner with my sister to stay at work.",
        ),
        (
            1,
            "2026-07-13",
            "Ignored my sister's call to finish another deadline.",
        ),
    )
    return [
        WeeklyDriftReviewerDecisionContract(
            persona_id=PERSONA_ID,
            week_start=date,
            week_end=date,
            t_index=t_index,
            date=date,
            core_value="benevolence",
            verdict="conflict",
            confidence="high",
            reason_code="direct_behavior_or_choice",
            evidence_quote=quote,
            review_status="ok",
        )
        for t_index, date, quote in rows
    ]


def _digest() -> WeeklyDigest:
    narrative = CoachNarrative(
        weekly_mirror=(
            "You wrote twice about choosing work over time with your sister."
        ),
        tension_explanation=(
            "Those choices conflict with the importance you placed on being "
            "present for people close to you."
        ),
        reflective_question="What boundary could protect one evening next week?",
    )
    validation = DigestValidation(
        grounded_quotes=["Cancelled dinner with my sister", "Ignored my sister's call"],
        word_count=38,
        checks=[
            ValidationCheck(name="groundedness", passed=True, details="Two citations"),
            ValidationCheck(name="non_circularity", passed=True, details="Specific"),
            ValidationCheck(name="length", passed=True, details="Within limit"),
        ],
    )
    return WeeklyDigest(
        persona_id=PERSONA_ID,
        persona_name="Casey",
        week_start="2026-07-06",
        week_end="2026-07-13",
        response_mode="active",
        mode_source="drift_detector",
        mode_rationale="Two consecutive Conflicts for Benevolence formed Drift.",
        signal_source="weekly_drift_reviewer",
        n_entries=2,
        overall_mean=None,
        overall_uncertainty=None,
        core_values=["benevolence"],
        drift_states={"benevolence": "active"},
        drift_reasons=["Benevolence: Journal Entries 0 and 1 were Conflicts."],
        top_tensions=["Work repeatedly displaced time with a close relationship."],
        top_strengths=[],
        dimensions=[],
        evidence=[
            EvidenceSnippet(
                date="2026-07-06",
                t_index=0,
                direction="misaligned",
                dimensions=["benevolence"],
                excerpt="Cancelled dinner with my sister to stay at work.",
            ),
            EvidenceSnippet(
                date="2026-07-13",
                t_index=1,
                direction="misaligned",
                dimensions=["benevolence"],
                excerpt="Ignored my sister's call to finish another deadline.",
            ),
        ],
        coach_narrative=narrative,
        validation=validation,
    )


def _event(
    *,
    event_id: str,
    event_type: str,
    parent_event_id: str | None,
    details: dict[str, Any],
    status: str = "complete",
    source: str = "live_run",
    model_contract: dict[str, Any] | None = None,
    prompt: str | None = None,
    raw_response: Any | None = None,
    validation: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    input_refs: list[dict[str, str]] | None = None,
    result_refs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    number = int(event_id.split("-")[-1])
    return {
        "schema_version": CONTRACT_VERSION,
        "event_id": event_id,
        "session_id": SESSION_ID,
        "parent_event_id": parent_event_id,
        "event_type": event_type,
        "status": status,
        "source": source,
        "started_at": f"2026-07-22T10:{number:02d}:00.000Z",
        "completed_at": f"2026-07-22T10:{number:02d}:00.120Z",
        "duration_ms": 120,
        "input_refs": input_refs or [],
        "model_contract": model_contract,
        "prompt": prompt,
        "raw_response": raw_response,
        "validation": validation
        or {
            "valid": status in {"complete", "reused"},
            "schema_name": f"{event_type}-v1",
            "errors": [],
        },
        "result_refs": result_refs or [],
        "input_hash": f"{number:064x}",
        "error": error,
        "details": details,
    }


def build_canonical_fixture() -> ContractFixtureSet:
    """Build the cross-language fixture and validate every reference."""
    profile = OnboardingProfile.model_validate(_profile())
    entries = [
        JournalEntry(
            journal_entry_id="entry-0",
            t_index=0,
            date="2026-07-06",
            content="Cancelled dinner with my sister to stay at work.",
        ),
        JournalEntry(
            journal_entry_id="entry-1",
            t_index=1,
            date="2026-07-13",
            content="Ignored my sister's call to finish another deadline.",
            nudge_response="I keep assuming work cannot wait.",
        ),
    ]
    nudge = NudgeInteraction(
        nudge_id="nudge-1",
        journal_entry_id="entry-1",
        outcome="answered",
        category="tension_surfacing",
        reason="The Journal Entry describes a repeated choice against a Core Value.",
        text="What made work feel less movable than time with your sister?",
        response="I keep assuming work cannot wait.",
    )
    decisions = _review_decisions()
    drift_result = detect_drift(decisions, persona_id=PERSONA_ID)
    digest = _digest()
    history = [
        WeeklyDriftReviewerEntry(
            t_index=entry.t_index,
            date=entry.date,
            text=entry.content,
        )
        for entry in entries
    ]
    review_request = WeeklyDriftReviewerRequest(
        persona_id=PERSONA_ID,
        week_start="2026-07-06",
        week_end="2026-07-19",
        core_values=["benevolence"],
        history=history,
        current_t_indices=[0, 1],
        prompt="Review the two displayed Journal Entries for Benevolence Conflict.",
        prompt_sha256="a" * 64,
        runtime_text_sha256="b" * 64,
    )
    assessments = [
        VerifierAssessment(
            t_index=decision.t_index,
            dimension=decision.core_value,
            verdict=decision.verdict,
            confidence="high",
            reason_code="direct_behavior_or_choice",
            evidence_quote=decision.evidence_quote,
        )
        for decision in decisions
    ]
    receipt = WeeklyDriftReviewerReceipt(
        created_at="2026-07-22T10:07:00.000Z",
        persona_id=PERSONA_ID,
        week_start="2026-07-06",
        week_end="2026-07-19",
        core_values=["benevolence"],
        current_t_indices=[0, 1],
        prompt_name="weekly_vif_verifier",
        prompt_version="2.0",
        prompt_sha256="a" * 64,
        runtime_text_sha256="b" * 64,
        requested_model="gpt-5.6-luna",
        reasoning_effort="low",
        status="ok",
        attempts=1,
        latency_seconds=1.25,
        resolved_model="gpt-5.6-luna",
        response_id="response-demo",
        usage={"input_tokens": 420, "output_tokens": 90},
        assessments=assessments,
        decisions=[
            WeeklyDriftReviewerDecision.model_validate(decision.model_dump())
            for decision in decisions
        ],
    )
    invalid_decisions = [
        decision.model_copy(
            update={
                "verdict": "abstain",
                "confidence": None,
                "reason_code": None,
                "evidence_quote": "",
                "review_status": "invalid",
            }
        )
        for decision in decisions
    ]
    invalid_receipt = receipt.model_copy(
        update={
            "status": "invalid",
            "validation_error": "Provider response did not match the schema.",
            "assessments": [],
            "decisions": invalid_decisions,
        }
    )

    profile_event = _event(
        event_id="event-01",
        event_type="profile_confirmed",
        parent_event_id=None,
        details={"profile": profile.model_dump()},
        result_refs=[{"kind": "profile", "id": SESSION_ID}],
    )
    entry_zero_event = _event(
        event_id="event-02",
        event_type="journal_entry_submitted",
        parent_event_id="event-01",
        details={"journal_entry": entries[0].model_dump(), "ordering_valid": True},
        input_refs=[{"kind": "profile", "id": SESSION_ID}],
        result_refs=[{"kind": "journal_entry", "id": "entry-0"}],
    )
    entry_one_event = _event(
        event_id="event-03",
        event_type="journal_entry_submitted",
        parent_event_id="event-02",
        details={"journal_entry": entries[1].model_dump(), "ordering_valid": True},
        input_refs=[{"kind": "profile", "id": SESSION_ID}],
        result_refs=[{"kind": "journal_entry", "id": "entry-1"}],
    )
    suppression_event = _event(
        event_id="event-04",
        event_type="nudge_suppression_checked",
        parent_event_id="event-03",
        details={
            "previous_entry_ids": ["entry-0"],
            "window_size": 3,
            "max_nudges": 2,
            "suppressed": False,
        },
        input_refs=[
            {"kind": "journal_entry", "id": "entry-0"},
            {"kind": "journal_entry", "id": "entry-1"},
        ],
    )
    nudge_decision_event = _event(
        event_id="event-05",
        event_type="nudge_decided",
        parent_event_id="event-04",
        details={
            "should_nudge": True,
            "category": "tension_surfacing",
            "reason": nudge.reason,
        },
        model_contract={
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "reasoning_effort": "none",
        },
        prompt="Decide whether this Journal Entry needs a nudge.",
        raw_response={
            "decision": "tension_surfacing",
            "reason": nudge.reason,
        },
        input_refs=[{"kind": "journal_entry", "id": "entry-1"}],
        result_refs=[{"kind": "event", "id": "event-05"}],
    )
    nudge_generated_event = _event(
        event_id="event-06",
        event_type="nudge_generated",
        parent_event_id="event-05",
        details={"nudge": nudge.model_dump(), "word_count": 10, "attempts": 1},
        model_contract={
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "reasoning_effort": "none",
        },
        prompt="Write one brief reflective question.",
        raw_response={"nudge_text": nudge.text},
        input_refs=[
            {"kind": "journal_entry", "id": "entry-1"},
            {"kind": "event", "id": "event-05"},
        ],
        result_refs=[{"kind": "nudge", "id": "nudge-1"}],
    )
    review_requested_event = _event(
        event_id="event-07",
        event_type="weekly_review_requested",
        parent_event_id="event-06",
        details={"request": review_request.model_dump()},
        source="saved_replay",
        model_contract=LUNA_LOW.model_dump(),
        prompt=review_request.prompt,
        input_refs=[
            {"kind": "profile", "id": SESSION_ID},
            {"kind": "journal_entry", "id": "entry-0"},
            {"kind": "journal_entry", "id": "entry-1"},
            {"kind": "week", "id": "2026-07-13"},
        ],
        result_refs=[{"kind": "event", "id": "event-07"}],
    )
    review_completed_event = _event(
        event_id="event-08",
        event_type="weekly_review_completed",
        parent_event_id="event-07",
        details={"receipt": receipt.model_dump()},
        status="reused",
        source="saved_replay",
        model_contract=LUNA_LOW.model_dump(),
        prompt=review_request.prompt,
        raw_response={
            "assessments": [assessment.model_dump() for assessment in assessments]
        },
        input_refs=[{"kind": "event", "id": "event-07"}],
        result_refs=[{"kind": "weekly_review", "id": "response-demo"}],
    )
    drift_event = _event(
        event_id="event-09",
        event_type="drift_detected",
        parent_event_id="event-08",
        source="saved_replay",
        details={
            "decisions": [decision.model_dump() for decision in decisions],
            "steps": [
                {
                    "t_index": 0,
                    "core_value": "benevolence",
                    "verdict": "conflict",
                    "pending_run_length": 1,
                    "effect": "start",
                },
                {
                    "t_index": 1,
                    "core_value": "benevolence",
                    "verdict": "conflict",
                    "pending_run_length": 2,
                    "effect": "confirm",
                },
            ],
            "result": drift_result.model_dump(),
        },
        input_refs=[{"kind": "weekly_review", "id": "response-demo"}],
        result_refs=[{"kind": "drift", "id": "persona-demo:benevolence:0"}],
    )
    digest_event = _event(
        event_id="event-10",
        event_type="weekly_digest_built",
        parent_event_id="event-09",
        source="saved_replay",
        details={
            "digest": digest.model_dump(),
            "cited_journal_entry_ids": ["entry-0", "entry-1"],
        },
        input_refs=[
            {"kind": "drift", "id": "persona-demo:benevolence:0"},
            {"kind": "journal_entry", "id": "entry-0"},
            {"kind": "journal_entry", "id": "entry-1"},
        ],
        result_refs=[{"kind": "weekly_digest", "id": "digest-demo"}],
    )
    narrative = digest.coach_narrative
    digest_validation = digest.validation
    assert narrative is not None
    assert digest_validation is not None
    coach_event = _event(
        event_id="event-11",
        event_type="weekly_coach_generated",
        parent_event_id="event-10",
        source="saved_replay",
        details={
            "narrative": narrative.model_dump(),
            "validation": digest_validation.model_dump(),
        },
        model_contract={
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "reasoning_effort": "none",
        },
        prompt="Turn the Weekly Digest into a reflection and one question.",
        raw_response=narrative.model_dump(),
        input_refs=[{"kind": "weekly_digest", "id": "digest-demo"}],
        result_refs=[{"kind": "weekly_coach", "id": "coach-demo"}],
    )
    refused_event = _event(
        event_id="event-12",
        event_type="nudge_decided",
        parent_event_id="event-04",
        details={"should_nudge": False, "category": None, "reason": None},
        status="refused",
        model_contract={
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "reasoning_effort": "none",
        },
        prompt="Decide whether this Journal Entry needs a nudge.",
        error={
            "code": "provider_refusal",
            "message": "The provider refused the request.",
            "retryable": False,
        },
        input_refs=[{"kind": "journal_entry", "id": "entry-1"}],
    )
    invalid_event = _event(
        event_id="event-13",
        event_type="weekly_review_completed",
        parent_event_id="event-07",
        details={"receipt": invalid_receipt.model_dump()},
        status="invalid",
        model_contract=LUNA_LOW.model_dump(),
        prompt=review_request.prompt,
        raw_response={"unexpected": True},
        validation={
            "valid": False,
            "schema_name": "WeeklyVerifierResponse",
            "errors": ["assessments is required"],
        },
        error={
            "code": "invalid_response",
            "message": "The provider response did not match the response schema.",
            "retryable": False,
        },
        input_refs=[{"kind": "event", "id": "event-07"}],
    )
    failed_event = _event(
        event_id="event-14",
        event_type="nudge_generated",
        parent_event_id="event-05",
        details={"nudge": None, "word_count": None, "attempts": 2},
        status="failed",
        model_contract={
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "reasoning_effort": "none",
        },
        prompt="Write one brief reflective question.",
        validation={
            "valid": False,
            "schema_name": "Nudge",
            "errors": ["No valid response after two attempts"],
        },
        error={
            "code": "provider_timeout",
            "message": "The nudge request timed out after two attempts.",
            "retryable": True,
        },
        input_refs=[
            {"kind": "journal_entry", "id": "entry-1"},
            {"kind": "event", "id": "event-05"},
        ],
    )
    event_payloads = [
        profile_event,
        entry_zero_event,
        entry_one_event,
        suppression_event,
        nudge_decision_event,
        nudge_generated_event,
        review_requested_event,
        review_completed_event,
        drift_event,
        digest_event,
        coach_event,
        refused_event,
        invalid_event,
        failed_event,
    ]
    main_event_ids = [f"event-{index:02d}" for index in range(1, 12)]
    session = ExperienceSession(
        session_id=SESSION_ID,
        revision=3,
        profile=profile,
        journal_entries=entries,
        nudges=[nudge],
        weekly_reviewer_decisions=decisions,
        drift_result=drift_result,
        weekly_digest=digest,
        trace_event_ids=main_event_ids,
        selection=SessionSelection(
            view="inspect",
            selected_week="2026-07-13",
            selected_journal_entry_id="entry-1",
            selected_event_id="event-09",
        ),
        updated_at="2026-07-22T10:11:00.120Z",
    )
    scenario = ScenarioBundle(
        scenario_id="active-drift-demo-v1",
        title="Work repeatedly displaces a close relationship",
        description="Two saved Conflicts for Benevolence form active Drift.",
        source="saved_replay",
        persona_id=PERSONA_ID,
        profile=profile,
        journal_entries=entries,
        weekly_reviewer_decisions=decisions,
        drift_result=drift_result,
        weekly_digest=digest,
        weeks=[
            ScenarioWeek(
                week_id="2026-07-06",
                week_start="2026-07-06",
                week_end="2026-07-12",
                journal_entry_ids=["entry-0"],
                event_ids=["event-02", "event-07", "event-08"],
                expected_delivery_state="stable",
            ),
            ScenarioWeek(
                week_id="2026-07-13",
                week_start="2026-07-13",
                week_end="2026-07-19",
                journal_entry_ids=["entry-1"],
                event_ids=["event-03", "event-09", "event-10", "event-11"],
                expected_delivery_state="active",
            ),
        ],
        trace_event_ids=main_event_ids,
        manifest=ScenarioManifest(
            bundle_version="scenario-bundle-v1",
            created_at="2026-07-22T10:12:00.000Z",
            input_hash="c" * 64,
            source_files=[
                "logs/wrangled/persona_demo.md",
                "logs/experiments/artifacts/demo/weekly_review.json",
            ],
            model_contract=LUNA_LOW,
            prompt_sha256="a" * 64,
        ),
    )
    requests = [
        SessionCreateRequest(
            operation="create_session",
            request_id="request-create",
            idempotency_key="d" * 64,
            profile=profile,
        ),
        JournalEntrySubmitRequest(
            operation="submit_journal_entry",
            request_id="request-entry",
            idempotency_key="e" * 64,
            session_id=SESSION_ID,
            expected_revision=2,
            journal_entry=entries[1],
        ),
        ScenarioLoadRequest(
            operation="load_scenario",
            request_id="request-scenario",
            scenario_id=scenario.scenario_id,
        ),
        TraceReadRequest(
            operation="read_trace",
            request_id="request-trace",
            session_id=SESSION_ID,
            after_event_id="event-06",
        ),
    ]
    responses = [
        SessionCreatedResponse(
            operation="create_session",
            request_id="request-create",
            status="ok",
            session=session,
        ),
        ApiErrorResponse(
            operation="error",
            requested_operation="submit_journal_entry",
            request_id="request-conflict",
            status="error",
            error=SafeError(
                code="revision_conflict",
                message="The session changed after this request was prepared.",
                retryable=True,
            ),
        ),
    ]
    return ContractFixtureSet.model_validate(
        {
            "session": session,
            "scenario": scenario,
            "requests": requests,
            "responses": responses,
            "trace_events": event_payloads,
        }
    )
