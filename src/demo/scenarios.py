"""Build, export, load, and replay the five saved persona scenarios."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.coach.schemas import CoachNarrative, LLMCallMetrics
from src.coach.weekly_digest import (
    attach_coach_artifacts,
    build_weekly_drift_reviewer_digest,
    validate_weekly_digest_narrative,
)
from src.demo.contracts import (
    ContractFixtureSet,
    ExperienceSession,
    ModelContract,
    TraceEvent,
    WeeklyDriftReviewerDecisionContract,
    build_drift_rule_steps,
)
from src.demo.profile_projection import build_projected_profile
from src.drift_detector import detect_drift
from src.drift_review_app.data import ReviewData, load_review_data
from src.nudge.decision import should_suppress_nudge
from src.prompt_boundary import render_live_prompt_receipt
from src.weekly_drift_reviewer import (
    WeeklyDriftReviewerDecision,
    WeeklyDriftReviewerReceipt,
    WeeklyDriftReviewerRequest,
    WeeklyVerifierResponse,
    _effective_decisions,
    validate_weekly_drift_reviewer_response,
)
from src.wrangling.parse_wrangled_data import parse_wrangled_file

if TYPE_CHECKING:
    from src.north_star.runtime import NorthStarRequest

SCENARIO_DIRECTORY = Path("frontend/onboarding/public/scenarios")
CATALOG_PATH = SCENARIO_DIRECTORY / "index.json"
WEEKLY_EXPERIMENT_DIRECTORY = Path(
    "logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907"
)
PROMPTS_PATH = WEEKLY_EXPERIMENT_DIRECTORY / "requests.jsonl"
RESPONSES_PATH = WEEKLY_EXPERIMENT_DIRECTORY / "responses.jsonl"
ATTEMPTS_PATH = WEEKLY_EXPERIMENT_DIRECTORY / "attempts.jsonl"
WEEKLY_MANIFEST_PATH = WEEKLY_EXPERIMENT_DIRECTORY / "manifest.json"
WEEKLY_VARIANT = "definitions"
COACH_RESPONSES_PATH = Path("src/demo/coach_digest_responses.json")
NORTH_STAR_REPORT_PATH = Path("src/demo/north_star_replay_records.json")

LUNA_LOW = ModelContract(
    provider="openai",
    model="gpt-5.6-luna",
    reasoning_effort="low",
)

ScenarioRole = Literal[
    "no_active_drift",
    "active_drift",
    "drift_ended",
    "insufficient_evidence",
    "two_core_values",
]


@dataclass(frozen=True)
class ScenarioSelection:
    """One manually reviewed frozen persona replay."""

    scenario_id: str
    persona_id: str
    role: ScenarioRole
    title: str
    description: str
    summary: str
    coach_week_start: str
    recommended: bool = False
    run: int = 1


SELECTIONS = (
    ScenarioSelection(
        scenario_id="stable-noor",
        persona_id="02fb94f3",
        role="no_active_drift",
        title="Noor — autonomy, family, and steady priorities",
        description=(
            "A young parent navigates Self-Direction and Tradition with "
            "No Active Drift across six reviewed weeks."
        ),
        summary="A baseline with no detected Drift across the saved history.",
        coach_week_start="2025-05-19",
    ),
    ScenarioSelection(
        scenario_id="active-nisha",
        persona_id="5fa8b540",
        role="active_drift",
        title="Nisha — everyday choices and fairness",
        description=(
            "A teacher's Universalism Drift becomes active, then a later "
            "Not Conflict decision ends the pattern."
        ),
        summary="Recommended: a complete progression into and out of Active Drift.",
        coach_week_start="2025-03-03",
        recommended=True,
    ),
    ScenarioSelection(
        scenario_id="ended-sook-yin",
        persona_id="ed67c9cc",
        role="drift_ended",
        title="Sook Yin — making room for enjoyment",
        description=(
            "A stay-at-home mother's Hedonism Drift ends after a later "
            "Not Conflict decision; the Historical Drift Record remains visible."
        ),
        summary="A closed Hedonism Drift episode with its supporting Journal Entries.",
        coach_week_start="2025-02-10",
    ),
    ScenarioSelection(
        scenario_id="uncertain-wei-jun",
        persona_id="8f83c818",
        role="insufficient_evidence",
        title="Wei Jun — fairness and uncertain evidence",
        description=(
            "A fintech engineer considers remittance failures. A rejected "
            "evidence quote leaves Insufficient Evidence for Universalism Drift."
        ),
        summary="A failed Weekly Drift review prevents an unsupported Drift claim.",
        coach_week_start="2025-06-30",
    ),
    ScenarioSelection(
        scenario_id="two-values-henrik",
        persona_id="2d928d8a",
        role="two_core_values",
        title="Henrik — security alongside new experiences",
        description=(
            "Security has No Active Drift while Stimulation has Insufficient "
            "Evidence in the key week, with independent Core Value histories."
        ),
        summary="Two Core Values with different current states at the same cutoff.",
        coach_week_start="2025-02-17",
    ),
)


class CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SavedCoachGeneration(CatalogModel):
    """Accepted Coach Digest call and source provenance for one saved response."""

    model_contract: ModelContract
    service_tier: str
    prompt_name: str
    prompt_version: Literal["4.1", "4.2"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt: str = Field(min_length=1)
    raw_output: str = Field(min_length=1)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_count: int = Field(ge=1)
    diagnostic_paths: list[str] = Field(min_length=1)
    call_metrics: list[LLMCallMetrics] = Field(min_length=1)
    weekly_drift_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_response_path: str = Field(min_length=1)
    source_bundle_path: str = Field(min_length=1)
    source_bundle_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weekly_digest_event_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_call_contract(self) -> SavedCoachGeneration:
        if (
            self.model_contract.provider != "openai"
            or self.model_contract.model != "gpt-5.6-luna"
            or self.model_contract.reasoning_effort != "none"
        ):
            raise ValueError("Saved Coach Digest must use OpenAI Luna-none")
        if len(self.call_metrics) != self.attempt_count:
            raise ValueError("Saved Coach Digest attempt metrics are incomplete")
        return self


class SavedCoachResponse(CatalogModel):
    """One accepted response for an exact saved scenario week."""

    scenario_id: str
    persona_id: str
    week_start: str
    week_end: str
    narrative: CoachNarrative
    generation: SavedCoachGeneration | None = None


class SavedCoachResponseFixture(CatalogModel):
    """Five Coach Digest responses keyed by Scenario ID and week start."""

    schema_version: Literal["coach-digest-scenario-fixture-v1"] = (
        "coach-digest-scenario-fixture-v1"
    )
    responses: dict[str, SavedCoachResponse]

    @model_validator(mode="after")
    def validate_response_keys(self) -> SavedCoachResponseFixture:
        for key, response in self.responses.items():
            if key != f"{response.scenario_id}::{response.week_start}":
                raise ValueError("Coach Digest response key does not match its week")
        return self


class ScenarioCatalogItem(CatalogModel):
    scenario_id: str
    file: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    persona_id: str
    persona_name: str
    age: str
    profession: str
    culture: str
    core_values: list[str] = Field(min_length=1)
    role: ScenarioRole
    progression: list[str] = Field(min_length=1)
    summary: str
    recommended: bool
    key_week_start: str | None = None


class ScenarioCatalog(CatalogModel):
    schema_version: Literal["scenario-catalog-v1"] = "scenario-catalog-v1"
    evidence_source: Literal["ai_reviewed_synthetic_development"] = (
        "ai_reviewed_synthetic_development"
    )
    scenarios: list[ScenarioCatalogItem] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_menu(self) -> ScenarioCatalog:
        if len({item.scenario_id for item in self.scenarios}) != len(self.scenarios):
            raise ValueError("Scenario IDs must be unique")
        if sum(item.recommended for item in self.scenarios) != 1:
            raise ValueError("Exactly one scenario must be recommended")
        if {item.role for item in self.scenarios} != {
            "no_active_drift",
            "active_drift",
            "drift_ended",
            "insufficient_evidence",
            "two_core_values",
        }:
            raise ValueError("The menu must cover all five scenario roles")
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coach_response_sha256(narrative: CoachNarrative) -> str:
    return _sha256_json(narrative.model_dump(mode="json"))


def _weekly_drift_input_sha256(digest: Any) -> str:
    payload = digest.model_dump(
        mode="json",
        exclude={"coach_narrative", "validation"},
    )
    return _sha256_json(payload)


def _coach_unavailable_reason(
    response: SavedCoachResponse | None, digest: Any
) -> str | None:
    if response is None:
        return "No saved Coach Digest response exists for this replay week."
    if response.generation is None:
        return (
            "Historical Coach Digest omitted: "
            "its saved input provenance is unavailable."
        )
    actual = _weekly_drift_input_sha256(digest)
    expected = response.generation.weekly_drift_input_sha256
    if actual != expected:
        return (
            "Historical Coach Digest omitted: its Weekly Drift input differs from "
            f"v4 Run 1 (saved SHA-256 {expected}; current SHA-256 {actual})."
        )
    return None


def load_saved_coach_responses(root: Path) -> SavedCoachResponseFixture:
    """Load and validate the checked-in scenario Coach Digest responses."""
    fixture: SavedCoachResponseFixture = SavedCoachResponseFixture.model_validate_json(
        (root.resolve() / COACH_RESPONSES_PATH).read_bytes()
    )
    for response in fixture.responses.values():
        if response.generation is not None:
            if response.generation.response_sha256 != _coach_response_sha256(
                response.narrative
            ):
                raise ValueError("Coach Digest response hash differs from its text")
    return fixture


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _source_files(selection: ScenarioSelection) -> list[Path]:
    return [
        Path(f"logs/wrangled/persona_{selection.persona_id}.md"),
        Path(f"logs/synthetic_data/persona_{selection.persona_id}.md"),
        PROMPTS_PATH,
        RESPONSES_PATH,
        ATTEMPTS_PATH,
        WEEKLY_MANIFEST_PATH,
        COACH_RESPONSES_PATH,
    ]


def _input_hash(root: Path, selection: ScenarioSelection) -> str:
    source_hashes = {
        path.as_posix(): _sha256_file(root / path) for path in _source_files(selection)
    }
    return _sha256_json(
        {
            "persona_id": selection.persona_id,
            "run": selection.run,
            "source_files": source_hashes,
        }
    )


def _normalized_core_values(values: list[str]) -> list[str]:
    return [
        value.strip().lower().replace("-", "_").replace(" ", "_") for value in values
    ]


def _simulated_at(day: str, *, hour: int, sequence: int = 0) -> str:
    value = datetime.combine(date.fromisoformat(day), time(hour=hour), tzinfo=UTC)
    return (value + timedelta(milliseconds=sequence)).isoformat().replace("+00:00", "Z")


def _raw_nudge_metadata(path: Path) -> dict[int, dict[str, str]]:
    content = path.read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r"^## Entry (?P<number>\d+) - \d{4}-\d{2}-\d{2}\n"
            r"(?P<body>.*?)(?=^## Entry \d+ - |\Z)",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    metadata: dict[int, dict[str, str]] = {}
    for match in matches:
        body = match.group("body")
        category = re.search(r"^### Nudge \(([^)]+)\)$", body, re.MULTILINE)
        if category is None:
            continue
        trigger = re.search(r"^\*\*Trigger\*\*: (.+)$", body, re.MULTILINE)
        nudge_text = re.search(
            r"^### Nudge \([^)]+\)\n"
            r"\*\*Trigger\*\*: .+\n"
            r'"(.+)"$',
            body,
            re.MULTILINE,
        )
        if trigger is None or nudge_text is None:
            raise ValueError(f"Cannot parse saved nudge in {path}")
        metadata[int(match.group("number")) - 1] = {
            "category": category.group(1),
            "reason": trigger.group(1).strip(),
            "text": nudge_text.group(1).strip(),
        }
    return metadata


def _saved_weekly_sources(
    *,
    selection: ScenarioSelection,
    prompts: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    entries_by_index: dict[int, dict[str, Any]],
) -> tuple[dict[str, WeeklyDriftReviewerRequest], dict[str, dict[str, Any]]]:
    """Bind v4 Run 1 decisions to the exact frozen messages and provider output."""
    requests: dict[str, WeeklyDriftReviewerRequest] = {}
    receipts: dict[str, dict[str, Any]] = {}
    selected = [
        row for row in prompts if row["request"]["persona_id"] == selection.persona_id
    ]
    for row in selected:
        original = WeeklyDriftReviewerRequest.model_validate(row["request"])
        week_start = original.week_start
        if week_start in requests:
            raise ValueError("Duplicate selected Weekly Drift request")
        case_id = f"{selection.persona_id}:week:{week_start}"
        if row["case_id"] != case_id:
            raise ValueError("Weekly Drift request case identity mismatch")
        variant = row["variants"][WEEKLY_VARIANT]
        variant_hash = _sha256_json(
            {key: value for key, value in variant.items() if key != "request_sha256"}
        )
        if (
            variant_hash != variant["request_sha256"]
            or variant["prompt_version"] != "4.0"
        ):
            raise ValueError("Weekly Drift variant hash or version mismatch")
        history = [entry.model_dump(mode="json") for entry in original.history]
        if _sha256_json(history) != original.runtime_text_sha256:
            raise ValueError("Weekly Drift runtime source hash mismatch")
        for entry in original.history:
            source = entries_by_index[entry.t_index]
            parts = [str(source["initial_entry"])]
            if source.get("nudge_text"):
                parts.append(f'Nudge: "{source["nudge_text"]}"')
            if source.get("response_text"):
                parts.append(f"Response: {source['response_text']}")
            if entry.date != str(source["date"]) or entry.text != "\n\n".join(parts):
                raise ValueError(
                    "Weekly Drift history differs from saved Journal Entries"
                )
        expected_input = {
            "current_week_entry_t_indices": original.current_t_indices,
            "declared_core_values": original.core_values,
            "journal_entries": [
                {"t_index": entry.t_index, "text": entry.text}
                for entry in original.history
            ],
        }
        if json.loads(variant["input_data"]) != expected_input:
            raise ValueError("Weekly Drift variant input differs from saved history")
        prompt = render_live_prompt_receipt(
            instructions=variant["instructions"], input_data=variant["input_data"]
        )
        request = original.model_copy(
            update={
                "prompt": prompt,
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            }
        )
        request_key = f"{WEEKLY_VARIANT}:{selection.run}:{case_id}"
        matching = [
            response
            for response in responses
            if response.get("request_key") == request_key
        ]
        if len(matching) != 1:
            raise ValueError("Selected Weekly Drift response is missing or duplicated")
        response = matching[0]
        if (
            response.get("case_id") != case_id
            or response.get("variant") != WEEKLY_VARIANT
            or response.get("repeat") != selection.run
            or response.get("status") not in {"ok", "invalid"}
            or response.get("request_sha256") != variant_hash
        ):
            raise ValueError("Weekly Drift response source binding mismatch")
        matching_attempts = [
            attempt
            for attempt in attempts
            if attempt.get("request_key") == request_key
            and attempt.get("event") == "finished"
            and attempt.get("attempt_number") == response["attempts"]
        ]
        if len(matching_attempts) != 1:
            raise ValueError(
                "Selected Weekly Drift provider attempt is missing or duplicated"
            )
        attempt = matching_attempts[0]
        if (
            attempt.get("request_sha256") != variant_hash
            or attempt.get("status") != response["status"]
        ):
            raise ValueError("Weekly Drift provider attempt source binding mismatch")
        parsed = WeeklyVerifierResponse.model_validate(attempt["parsed"])
        if WeeklyVerifierResponse.model_validate_json(attempt["raw_text"]) != parsed:
            raise ValueError(
                "Weekly Drift raw response differs from parsed assessments"
            )
        if response["status"] == "ok":
            validate_weekly_drift_reviewer_response(parsed, request)
        else:
            try:
                validate_weekly_drift_reviewer_response(parsed, request)
            except ValueError as error:
                if str(error) != attempt.get("validation_error"):
                    raise ValueError(
                        "Weekly Drift validation diagnostic differs"
                    ) from error
            else:
                raise ValueError("Weekly Drift invalid receipt has valid assessments")
        decisions = _effective_decisions(
            request, status=response["status"], response=parsed
        )
        if [decision.model_dump(mode="json") for decision in decisions] != response[
            "decisions"
        ]:
            raise ValueError(
                "Weekly Drift saved decisions differ from provider assessments"
            )
        requests[week_start] = request
        receipts[week_start] = {"response": response, "attempt": attempt}
    return requests, receipts


def _event(
    *,
    event_id: str,
    session_id: str,
    parent_event_id: str | None,
    event_type: str,
    started_at: str,
    duration_ms: int,
    input_refs: list[dict[str, str]],
    result_refs: list[dict[str, str]],
    details: dict[str, Any],
    model_contract: dict[str, Any] | None = None,
    prompt: str | None = None,
    raw_response: Any | None = None,
) -> dict[str, Any]:
    completed = (
        (
            datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            + timedelta(milliseconds=duration_ms)
        )
        .isoformat()
        .replace("+00:00", "Z")
    )
    event_input = {
        "event_type": event_type,
        "input_refs": input_refs,
        "details": details,
        "prompt": prompt,
    }
    return {
        "schema_version": "experience-inspect-v1",
        "event_id": event_id,
        "session_id": session_id,
        "parent_event_id": parent_event_id,
        "event_type": event_type,
        "status": "reused",
        "source": "saved_replay",
        "started_at": started_at,
        "completed_at": completed,
        "duration_ms": duration_ms,
        "input_refs": input_refs,
        "model_contract": model_contract,
        "prompt": prompt,
        "raw_response": raw_response,
        "validation": {
            "valid": True,
            "schema_name": f"{event_type}-saved-replay-v1",
            "errors": [],
        },
        "result_refs": result_refs,
        "input_hash": _sha256_json(event_input),
        "error": None,
        "details": details,
    }


def _make_event_id(scenario_id: str, number: int) -> str:
    return f"{scenario_id}:event:{number:03d}"


def _receipt(
    *,
    request: WeeklyDriftReviewerRequest,
    source: dict[str, Any],
) -> WeeklyDriftReviewerReceipt:
    response, attempt = source["response"], source["attempt"]
    return WeeklyDriftReviewerReceipt(
        created_at=attempt["completed_at"],
        persona_id=request.persona_id,
        week_start=request.week_start,
        week_end=request.week_end,
        core_values=request.core_values,
        current_t_indices=request.current_t_indices,
        prompt_name="weekly_vif_verifier",
        prompt_version="4.0",
        prompt_sha256=request.prompt_sha256,
        runtime_text_sha256=request.runtime_text_sha256,
        requested_model=LUNA_LOW.model,
        reasoning_effort="low",
        status=response["status"],
        validation_error=attempt.get("validation_error"),
        attempts=response["attempts"],
        latency_seconds=attempt["latency_seconds"],
        resolved_model=attempt["resolved_model"],
        response_id=attempt["response_id"],
        usage={
            key: value
            for key, value in attempt["usage"].items()
            if isinstance(value, int) and not isinstance(value, bool)
        },
        assessments=WeeklyVerifierResponse.model_validate(
            attempt["parsed"]
        ).assessments,
        decisions=[
            WeeklyDriftReviewerDecision.model_validate(row)
            for row in response["decisions"]
        ],
    )


def _initial_session_payload(
    *,
    scenario: dict[str, Any],
    trace_events: list[dict[str, Any]],
    week_index: int,
) -> dict[str, Any]:
    weeks = scenario["weeks"]
    selected_week = weeks[week_index]
    allowed_event_ids = {
        event_id for week in weeks[: week_index + 1] for event_id in week["event_ids"]
    }
    journal_ids = {
        journal_id
        for week in weeks[: week_index + 1]
        for journal_id in week["journal_entry_ids"]
    }
    journal_entries = [
        entry
        for entry in scenario["journal_entries"]
        if entry["journal_entry_id"] in journal_ids
    ]
    visible_decisions = [
        decision
        for decision in scenario["weekly_reviewer_decisions"]
        if decision["week_end"] <= selected_week["week_end"]
    ]
    selected_events = [
        event for event in trace_events if event["event_id"] in allowed_event_ids
    ]
    nudges = [
        event["details"]["nudge"]
        for event in selected_events
        if event["event_type"] == "nudge_generated"
        and event["details"]["nudge"] is not None
    ]
    drift_event = next(
        event
        for event in selected_events
        if event["event_type"] == "drift_detected"
        and event["event_id"] in selected_week["event_ids"]
    )
    digest_event = next(
        event
        for event in selected_events
        if event["event_type"] == "weekly_digest_built"
        and event["event_id"] in selected_week["event_ids"]
    )
    return {
        "schema_version": "experience-inspect-v1",
        "session_id": scenario["profile"]["session_id"],
        "revision": len(journal_entries),
        "profile": scenario["profile"],
        "journal_entries": journal_entries,
        "nudges": nudges,
        "weekly_reviewer_decisions": visible_decisions,
        "drift_result": drift_event["details"]["result"],
        "weekly_digest": digest_event["details"]["digest"],
        "trace_event_ids": [
            event["event_id"]
            for event in trace_events
            if event["event_id"] in allowed_event_ids
        ],
        "selection": {
            "view": "experience",
            "selected_week": selected_week["week_id"],
            "selected_journal_entry_id": None,
            "selected_event_id": None,
        },
        "updated_at": digest_event["completed_at"],
    }


def build_scenario_fixture(
    root: Path,
    selection: ScenarioSelection,
    *,
    data: ReviewData | None = None,
    prompt_rows: list[dict[str, Any]] | None = None,
    response_rows: list[dict[str, Any]] | None = None,
    attempt_rows: list[dict[str, Any]] | None = None,
    coach_responses: SavedCoachResponseFixture | None = None,
    include_north_star: bool = True,
) -> ContractFixtureSet:
    """Build one saved replay without provider calls."""
    root = root.resolve()
    review_data = data or load_review_data(root)
    prompts = (
        prompt_rows if prompt_rows is not None else _read_jsonl(root / PROMPTS_PATH)
    )
    responses = (
        response_rows
        if response_rows is not None
        else _read_jsonl(root / RESPONSES_PATH)
    )
    attempts = (
        attempt_rows if attempt_rows is not None else _read_jsonl(root / ATTEMPTS_PATH)
    )
    weekly_manifest = json.loads((root / WEEKLY_MANIFEST_PATH).read_text())
    if (
        weekly_manifest["requests_sha256"] != _sha256_file(root / PROMPTS_PATH)
        or weekly_manifest["treatment_prompt_version"] != "4.0"
        or weekly_manifest["settings"]["model"] != LUNA_LOW.model
        or weekly_manifest["settings"]["reasoning_effort"] != "low"
    ):
        raise ValueError("Weekly Drift experiment manifest differs from saved sources")
    saved_coach_responses = coach_responses or load_saved_coach_responses(root)

    wrangled_path = root / f"logs/wrangled/persona_{selection.persona_id}.md"
    raw_path = root / f"logs/synthetic_data/persona_{selection.persona_id}.md"
    profile_row, parsed_entries, warnings = parse_wrangled_file(wrangled_path)
    if warnings:
        raise ValueError(f"Selected persona has parse warnings: {selection.persona_id}")
    entries_by_index = {int(entry["t_index"]): entry for entry in parsed_entries}
    core_values = _normalized_core_values(list(profile_row["core_values"]))
    boundaries = review_data.boundaries_for_persona(selection.persona_id)
    if not boundaries:
        raise ValueError(
            f"Selected persona has no weekly boundaries: {selection.persona_id}"
        )

    session_id = f"scenario-session:{selection.scenario_id}"
    first_date = min(str(entry["date"]) for entry in parsed_entries)
    profile_day = (date.fromisoformat(first_date) - timedelta(days=1)).isoformat()
    profile = build_projected_profile(
        persona_id=selection.persona_id,
        session_id=session_id,
        core_values=core_values,
        started_at=_simulated_at(profile_day, hour=9),
        completed_at=_simulated_at(profile_day, hour=9, sequence=300_000),
    )
    raw_nudges = _raw_nudge_metadata(raw_path)
    journal_entries = [
        {
            "journal_entry_id": f"{selection.persona_id}:entry:{entry['t_index']}",
            "t_index": int(entry["t_index"]),
            "date": str(entry["date"]),
            "content": str(entry["initial_entry"]),
            "nudge_response": (
                str(entry["response_text"]) if entry.get("response_text") else None
            ),
        }
        for entry in sorted(parsed_entries, key=lambda row: int(row["t_index"]))
    ]
    journal_id_by_index = {
        entry["t_index"]: entry["journal_entry_id"] for entry in journal_entries
    }
    request_by_week, response_by_week = _saved_weekly_sources(
        selection=selection,
        prompts=prompts,
        responses=responses,
        attempts=attempts,
        entries_by_index=entries_by_index,
    )
    if set(request_by_week) != {boundary.week_start for boundary in boundaries}:
        raise ValueError("Selected prompt weeks are incomplete")
    for boundary in boundaries:
        request = request_by_week[boundary.week_start]
        if (
            request.week_end != boundary.week_end
            or request.current_t_indices != list(boundary.current_t_indices)
            or [entry.t_index for entry in request.history]
            != list(boundary.visible_t_indices)
            or set(request.core_values) != set(core_values)
        ):
            raise ValueError("Weekly Drift request cutoff differs from saved replay")
    decisions = sorted(
        [
            WeeklyDriftReviewerDecisionContract.model_validate(row)
            for source in response_by_week.values()
            for row in source["response"]["decisions"]
        ],
        key=lambda decision: (decision.t_index, decision.core_value),
    )

    event_rows: list[dict[str, Any]] = []
    week_rows: list[dict[str, Any]] = []
    sequence = 0
    parent_event_id: str | None = None

    def append_event(**kwargs: Any) -> str:
        nonlocal sequence, parent_event_id
        sequence += 1
        event_id = _make_event_id(selection.scenario_id, sequence)
        event_rows.append(
            _event(
                event_id=event_id,
                session_id=session_id,
                parent_event_id=parent_event_id,
                **kwargs,
            )
        )
        parent_event_id = event_id
        return event_id

    profile_event_id = append_event(
        event_type="profile_confirmed",
        started_at=profile.timestamp,
        duration_ms=0,
        input_refs=[],
        result_refs=[{"kind": "profile", "id": selection.persona_id}],
        details={"profile": profile.model_dump(mode="json")},
    )
    displayed_history: list[bool] = []
    final_drift: dict[str, Any] | None = None
    final_digest: dict[str, Any] | None = None

    for week_number, boundary in enumerate(boundaries, start=1):
        week_event_ids = [profile_event_id] if week_number == 1 else []
        current_indices = list(boundary.current_t_indices)
        for index in current_indices:
            entry = entries_by_index[index]
            journal_id = journal_id_by_index[index]
            started_at = _simulated_at(str(entry["date"]), hour=12, sequence=index)
            week_event_ids.append(
                append_event(
                    event_type="journal_entry_submitted",
                    started_at=started_at,
                    duration_ms=0,
                    input_refs=[{"kind": "profile", "id": selection.persona_id}],
                    result_refs=[{"kind": "journal_entry", "id": journal_id}],
                    details={
                        "journal_entry": next(
                            row
                            for row in journal_entries
                            if row["journal_entry_id"] == journal_id
                        ),
                        "ordering_valid": True,
                    },
                )
            )
            suppressed = should_suppress_nudge(displayed_history)
            previous_ids = [
                journal_id_by_index[previous]
                for previous in sorted(entries_by_index)
                if previous < index
            ][-3:]
            week_event_ids.append(
                append_event(
                    event_type="nudge_suppression_checked",
                    started_at=_simulated_at(
                        str(entry["date"]), hour=12, sequence=index + 100
                    ),
                    duration_ms=0,
                    input_refs=[{"kind": "journal_entry", "id": journal_id}],
                    result_refs=[],
                    details={
                        "previous_entry_ids": previous_ids,
                        "window_size": 3,
                        "max_nudges": 2,
                        "suppressed": suppressed,
                    },
                )
            )
            saved_nudge = raw_nudges.get(index)
            if suppressed and saved_nudge is not None:
                raise ValueError(
                    f"Saved nudge violates anti-annoyance rule: "
                    f"{selection.persona_id} t_index={index}"
                )
            decision_category = (
                saved_nudge["category"]
                if saved_nudge is not None and not suppressed
                else None
            )
            decision_reason = (
                saved_nudge["reason"]
                if saved_nudge is not None and not suppressed
                else None
            )
            should_nudge = decision_category is not None
            week_event_ids.append(
                append_event(
                    event_type="nudge_decided",
                    started_at=_simulated_at(
                        str(entry["date"]), hour=12, sequence=index + 200
                    ),
                    duration_ms=0,
                    input_refs=[{"kind": "journal_entry", "id": journal_id}],
                    result_refs=[],
                    details={
                        "should_nudge": should_nudge,
                        "category": decision_category,
                        "reason": decision_reason,
                    },
                )
            )
            displayed_history.append(should_nudge)
            if should_nudge and saved_nudge is not None:
                response_text = entry.get("response_text")
                nudge = {
                    "nudge_id": f"{selection.persona_id}:nudge:{index}",
                    "journal_entry_id": journal_id,
                    "outcome": "answered" if response_text else "displayed",
                    "category": saved_nudge["category"],
                    "reason": saved_nudge["reason"],
                    "text": saved_nudge["text"],
                    "response": str(response_text) if response_text else None,
                }
                week_event_ids.append(
                    append_event(
                        event_type="nudge_generated",
                        started_at=_simulated_at(
                            str(entry["date"]), hour=12, sequence=index + 300
                        ),
                        duration_ms=0,
                        input_refs=[{"kind": "journal_entry", "id": journal_id}],
                        result_refs=[{"kind": "nudge", "id": nudge["nudge_id"]}],
                        details={
                            "nudge": nudge,
                            "word_count": len(saved_nudge["text"].split()),
                            "attempts": 1,
                        },
                    )
                )

        request = request_by_week[boundary.week_start]
        response_row = response_by_week[boundary.week_start]
        current_decisions = [
            row for row in decisions if row.week_start == boundary.week_start
        ]
        cumulative_decisions = [
            row for row in decisions if row.week_end <= boundary.week_end
        ]
        base_cumulative_decisions = [
            WeeklyDriftReviewerDecision.model_validate(row.model_dump(mode="json"))
            for row in cumulative_decisions
        ]
        review_started_at = _simulated_at(boundary.week_end, hour=20)
        week_id = f"{selection.scenario_id}:week:{week_number}"
        week_event_ids.append(
            append_event(
                event_type="weekly_review_requested",
                started_at=review_started_at,
                duration_ms=0,
                input_refs=[
                    {"kind": "week", "id": week_id},
                    *[
                        {
                            "kind": "journal_entry",
                            "id": journal_id_by_index[index],
                        }
                        for index in boundary.visible_t_indices
                    ],
                ],
                result_refs=[{"kind": "weekly_review", "id": week_id}],
                details={"request": request.model_dump(mode="json")},
                model_contract=LUNA_LOW.model_dump(mode="json"),
                prompt=request.prompt,
            )
        )
        receipt = _receipt(
            request=request,
            source=response_row,
        )
        week_event_ids.append(
            append_event(
                event_type="weekly_review_completed",
                started_at=review_started_at,
                duration_ms=round(receipt.latency_seconds * 1000),
                input_refs=[{"kind": "weekly_review", "id": week_id}],
                result_refs=[
                    {
                        "kind": "journal_entry",
                        "id": journal_id_by_index[row.t_index],
                    }
                    for row in current_decisions
                ],
                details={"receipt": receipt.model_dump(mode="json")},
                model_contract=LUNA_LOW.model_dump(mode="json"),
                prompt=request.prompt,
                raw_response=response_row,
            )
        )
        drift_result = detect_drift(
            base_cumulative_decisions,
            persona_id=selection.persona_id,
        )
        final_drift = drift_result.model_dump(mode="json")
        week_event_ids.append(
            append_event(
                event_type="drift_detected",
                started_at=_simulated_at(boundary.week_end, hour=21),
                duration_ms=0,
                input_refs=[{"kind": "weekly_review", "id": week_id}],
                result_refs=[{"kind": "drift", "id": week_id}],
                details={
                    "decisions": [
                        row.model_dump(mode="json") for row in cumulative_decisions
                    ],
                    "steps": [
                        step.model_dump(mode="json")
                        for step in build_drift_rule_steps(cumulative_decisions)
                    ],
                    "result": final_drift,
                },
            )
        )
        digest = build_weekly_drift_reviewer_digest(
            persona_id=selection.persona_id,
            wrangled_dir=root / "logs/wrangled",
            week_start=boundary.week_start,
            week_end=boundary.week_end,
            core_values=core_values,
            decisions=base_cumulative_decisions,
            drift_result=drift_result,
        )
        coach_key = f"{selection.scenario_id}::{boundary.week_start}"
        saved_coach_response = saved_coach_responses.responses.get(coach_key)
        coach_unavailable_reason = _coach_unavailable_reason(
            saved_coach_response, digest
        )
        coach_narrative = (
            saved_coach_response.narrative
            if saved_coach_response is not None and coach_unavailable_reason is None
            else None
        )
        coach_validation = None
        if coach_narrative is not None:
            if (
                saved_coach_response is None
                or saved_coach_response.persona_id != selection.persona_id
                or saved_coach_response.week_end != boundary.week_end
            ):
                raise ValueError("Saved Coach Digest response has the wrong identity")
            coach_validation = validate_weekly_digest_narrative(
                digest,
                coach_narrative,
            )
            failed_checks = [
                check.name for check in coach_validation.checks if not check.passed
            ]
            if failed_checks:
                raise ValueError(
                    "Saved Coach Digest response failed checks: "
                    f"{', '.join(failed_checks)}"
                )
            digest = attach_coach_artifacts(
                digest,
                coach_narrative,
                coach_validation,
            )
        final_digest = digest.model_dump(mode="json")
        cited_ids = [
            journal_id_by_index[evidence.t_index] for evidence in digest.evidence
        ]
        week_event_ids.append(
            append_event(
                event_type="weekly_digest_built",
                started_at=_simulated_at(boundary.week_end, hour=21, sequence=100),
                duration_ms=0,
                input_refs=[
                    {"kind": "week", "id": week_id},
                    {"kind": "drift", "id": week_id},
                ],
                result_refs=[{"kind": "weekly_digest", "id": week_id}],
                details={
                    "digest": final_digest,
                    "cited_journal_entry_ids": cited_ids,
                    "coach_unavailable_reason": coach_unavailable_reason,
                },
            )
        )
        if coach_narrative is not None and coach_validation is not None:
            coach_generation = (
                saved_coach_response.generation
                if saved_coach_response is not None
                else None
            )
            accepted_call = (
                coach_generation.call_metrics[-1]
                if coach_generation is not None
                else None
            )
            week_event_ids.append(
                append_event(
                    event_type="weekly_coach_generated",
                    started_at=_simulated_at(boundary.week_end, hour=21, sequence=200),
                    duration_ms=(
                        round(accepted_call.latency_seconds * 1000)
                        if accepted_call is not None
                        else 0
                    ),
                    input_refs=[{"kind": "weekly_digest", "id": week_id}],
                    result_refs=[{"kind": "weekly_coach", "id": week_id}],
                    details={
                        "narrative": coach_narrative.model_dump(mode="json"),
                        "validation": coach_validation.model_dump(mode="json"),
                    },
                    model_contract=(
                        coach_generation.model_contract.model_dump(mode="json")
                        if coach_generation is not None
                        else None
                    ),
                    prompt=(
                        coach_generation.prompt
                        if coach_generation is not None
                        else None
                    ),
                    raw_response=(
                        coach_generation.raw_output
                        if coach_generation is not None
                        else None
                    ),
                )
            )
        week_rows.append(
            {
                "week_id": week_id,
                "week_start": boundary.week_start,
                "week_end": boundary.week_end,
                "journal_entry_ids": [
                    journal_id_by_index[index] for index in current_indices
                ],
                "event_ids": week_event_ids,
                "expected_delivery_state": drift_result.delivery_state,
            }
        )

    if final_drift is None or final_digest is None:
        raise ValueError("Scenario has no weekly result")
    prompt_set_hash = _sha256_json(
        [request_by_week[boundary.week_start].prompt_sha256 for boundary in boundaries]
    )
    weekly_manifest = json.loads(
        (root / WEEKLY_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    scenario = {
        "schema_version": "experience-inspect-v1",
        "scenario_id": selection.scenario_id,
        "title": selection.title,
        "description": selection.description,
        "source": "saved_replay",
        "persona_id": selection.persona_id,
        "profile": profile.model_dump(mode="json"),
        "journal_entries": journal_entries,
        "weekly_reviewer_decisions": [row.model_dump(mode="json") for row in decisions],
        "drift_result": final_drift,
        "weekly_digest": final_digest,
        "weeks": week_rows,
        "trace_event_ids": [event["event_id"] for event in event_rows],
        "manifest": {
            "bundle_version": "scenario-bundle-v1",
            "created_at": str(weekly_manifest["frozen_at"]),
            "input_hash": _input_hash(root, selection),
            "source_files": [path.as_posix() for path in _source_files(selection)],
            "model_contract": LUNA_LOW.model_dump(mode="json"),
            "prompt_sha256": prompt_set_hash,
        },
    }
    session = _initial_session_payload(
        scenario=scenario,
        trace_events=event_rows,
        week_index=0,
    )
    request_id = f"request:{selection.scenario_id}:load"
    fixture = cast(
        ContractFixtureSet,
        ContractFixtureSet.model_validate(
            {
                "schema_version": "experience-inspect-v1",
                "session": session,
                "scenario": scenario,
                "requests": [
                    {
                        "schema_version": "experience-inspect-v1",
                        "operation": "load_scenario",
                        "request_id": request_id,
                        "scenario_id": selection.scenario_id,
                    }
                ],
                "responses": [
                    {
                        "schema_version": "experience-inspect-v1",
                        "operation": "load_scenario",
                        "request_id": request_id,
                        "status": "ok",
                        "session": session,
                        "scenario": scenario,
                        "event_ids": session["trace_event_ids"],
                    }
                ],
                "trace_events": event_rows,
            }
        ),
    )
    if include_north_star and (root / NORTH_STAR_REPORT_PATH).exists():
        fixture = attach_saved_north_star(fixture, root=root)
    return fixture


def build_saved_north_star_request(
    fixture: ContractFixtureSet, week_id: str
) -> NorthStarRequest:
    """Build the shared request from only the selected replay cutoff."""
    from src.north_star.runtime import (
        SourceWriting,
        build_north_star_request,
        profile_reference,
    )

    session, events = project_scenario_week(fixture, week_id)
    week = next(w for w in fixture.scenario.weeks if w.week_id == week_id)
    # The completed synthetic experiment uses immediate parent ordering. This
    # is declared synthetic availability, not a human response timestamp.
    availability = {
        entry.journal_entry_id: datetime.combine(
            date.fromisoformat(entry.date), time.min, tzinfo=UTC
        )
        + timedelta(microseconds=3 * entry.t_index)
        for entry in session.journal_entries
    }
    digest_event = next(
        event
        for event in events
        if event.event_type == "weekly_digest_built"
        and event.event_id in week.event_ids
    )
    writing = [
        SourceWriting(
            owner_id=fixture.scenario.persona_id,
            entry_id=entry.journal_entry_id,
            t_index=entry.t_index,
            date=entry.date,
            journal_entry=entry.content,
            nudge_response=entry.nudge_response,
            available_at=availability[entry.journal_entry_id].isoformat(),
            response_available_at=(
                (
                    availability[entry.journal_entry_id] + timedelta(microseconds=2)
                ).isoformat()
                if entry.nudge_response
                else None
            ),
        )
        for entry in session.journal_entries
    ]
    if session.drift_result is None:
        raise ValueError("Saved North Star Moment requires a completed weekly review")
    return build_north_star_request(
        session_id=session.session_id,
        owner_id=fixture.scenario.persona_id,
        profile_ref=profile_reference(session.profile.model_dump(mode="json")),
        core_values=[str(value) for value in session.profile.top_values],
        week_start=week.week_start,
        week_end=week.week_end,
        cutoff_at=digest_event.completed_at or digest_event.started_at,
        drift_result=session.drift_result,
        writing=writing,
    )


def attach_saved_north_star(
    fixture: ContractFixtureSet, *, root: Path
) -> ContractFixtureSet:
    """Attach frozen offline records; exporting and replaying never call a model."""
    from pydantic import TypeAdapter

    from src.demo.north_star_replay import (
        SavedExperimentRecord,
        validate_saved_record,
        verify_experiment_source,
    )
    from src.north_star.runtime import NorthStarRecord

    verify_experiment_source(root)
    adapter: TypeAdapter[SavedExperimentRecord | NorthStarRecord] = TypeAdapter(
        SavedExperimentRecord | NorthStarRecord
    )
    report = json.loads((root / NORTH_STAR_REPORT_PATH).read_text())
    matching = [
        row
        for row in report["cases"]
        if row["scenario_id"] == fixture.scenario.scenario_id
    ]
    records = {row["week_id"]: row["record"] for row in matching}
    if len(records) != len(matching):
        raise ValueError("Saved North Star Moment repeats a week")
    if set(records) != {week.week_id for week in fixture.scenario.weeks}:
        raise ValueError("Saved North Star Moment records do not cover every week")
    payload = fixture.model_dump(mode="json")
    events_by_id = {event["event_id"]: event for event in payload["trace_events"]}
    ordered_events: list[dict[str, Any]] = []
    for week in payload["scenario"]["weeks"]:
        request = build_saved_north_star_request(fixture, week["week_id"])
        record = adapter.validate_python(records[week["week_id"]])
        validate_saved_record(record, request)
        ordered_events.extend(events_by_id[event_id] for event_id in week["event_ids"])
        event_id = f"{fixture.scenario.scenario_id}:north-star:{week['week_start']}"
        event = _event(
            event_id=event_id,
            session_id=fixture.session.session_id,
            parent_event_id=week["event_ids"][-1],
            event_type="north_star_reviewed",
            started_at=ordered_events[-1]["completed_at"],
            duration_ms=0,
            input_refs=[{"kind": "week", "id": week["week_id"]}],
            result_refs=[],
            details={"record": record.model_dump(mode="json")},
            model_contract=LUNA_LOW.model_dump(mode="json"),
        )
        event["input_hash"] = record.input_hash
        ordered_events.append(event)
        week["event_ids"].append(event_id)
    payload["trace_events"] = ordered_events
    scenario = payload["scenario"]
    scenario["trace_event_ids"] = [event["event_id"] for event in ordered_events]
    scenario["manifest"]["source_files"].append(NORTH_STAR_REPORT_PATH.as_posix())
    scenario["manifest"]["input_hash"] = _sha256_json(
        {
            "base_input_hash": scenario["manifest"]["input_hash"],
            "north_star_report_sha256": _sha256_file(root / NORTH_STAR_REPORT_PATH),
        }
    )
    session = _initial_session_payload(
        scenario=scenario, trace_events=ordered_events, week_index=0
    )
    payload["session"] = session
    for response in payload["responses"]:
        if response["operation"] == "load_scenario":
            response["session"] = session
            response["scenario"] = scenario
            response["event_ids"] = session["trace_event_ids"]
    return ContractFixtureSet.model_validate(payload)


def project_scenario_week(
    fixture: ContractFixtureSet,
    week_id: str,
) -> tuple[ExperienceSession, list[TraceEvent]]:
    """Return only the session state and trace events visible through one week."""
    scenario = fixture.scenario.model_dump(mode="json")
    trace_events = [event.model_dump(mode="json") for event in fixture.trace_events]
    try:
        week_index = next(
            index
            for index, week in enumerate(fixture.scenario.weeks)
            if week.week_id == week_id
        )
    except StopIteration as error:
        raise ValueError(f"Unknown scenario week: {week_id}") from error
    session = ExperienceSession.model_validate(
        _initial_session_payload(
            scenario=scenario,
            trace_events=trace_events,
            week_index=week_index,
        )
    )
    allowed = set(session.trace_event_ids)
    visible_events = [
        event for event in fixture.trace_events if event.event_id in allowed
    ]
    return session, visible_events


def _selection_by_scenario_id(scenario_id: str) -> ScenarioSelection:
    try:
        return next(
            selection
            for selection in SELECTIONS
            if selection.scenario_id == scenario_id
        )
    except StopIteration as error:
        raise ValueError(f"Unknown curated scenario: {scenario_id}") from error


def _validate_fixture_semantics(
    fixture: ContractFixtureSet,
    *,
    root: Path,
) -> None:
    scenario = fixture.scenario
    selection = _selection_by_scenario_id(scenario.scenario_id)
    if scenario.persona_id != selection.persona_id:
        raise ValueError("Scenario persona does not match the curated selection")
    if scenario.profile.provenance.source != "synthetic_persona_projection":
        raise ValueError("Saved persona replay lacks synthetic Profile provenance")
    expected_sources = [path.as_posix() for path in _source_files(selection)]
    expected_input_hash = _input_hash(root, selection)
    has_north_star = any(
        event.event_type == "north_star_reviewed" for event in fixture.trace_events
    )
    if has_north_star:
        expected_sources.append(NORTH_STAR_REPORT_PATH.as_posix())
        expected_input_hash = _sha256_json(
            {
                "base_input_hash": expected_input_hash,
                "north_star_report_sha256": _sha256_file(root / NORTH_STAR_REPORT_PATH),
            }
        )
    if scenario.manifest.source_files != expected_sources:
        raise ValueError("Scenario source provenance is incomplete")
    if scenario.manifest.input_hash != expected_input_hash:
        raise ValueError("Scenario input hash differs from frozen sources")
    if has_north_star:
        from src.demo.north_star_replay import (
            validate_saved_record,
            verify_experiment_source,
        )

        verify_experiment_source(root)
        report = json.loads((root / NORTH_STAR_REPORT_PATH).read_text())
        expected_records = {
            row["week_id"]: row["record"]
            for row in report["cases"]
            if row["scenario_id"] == scenario.scenario_id
        }
        for week in scenario.weeks:
            records = [
                event.details.record
                for event in fixture.trace_events
                if event.event_type == "north_star_reviewed"
                and event.event_id in week.event_ids
            ]
            if len(records) != 1:
                raise ValueError("Saved North Star Moment must cover every week once")
            if records[0].model_dump(mode="json") != expected_records.get(week.week_id):
                raise ValueError(
                    "Saved NSM record differs from the exported experiment"
                )
            validate_saved_record(
                records[0], build_saved_north_star_request(fixture, week.week_id)
            )

    coach_responses = load_saved_coach_responses(root)
    expected_coach_key = f"{selection.scenario_id}::{selection.coach_week_start}"
    expected_coach = coach_responses.responses.get(expected_coach_key)
    coach_events = [
        event
        for event in fixture.trace_events
        if event.event_type == "weekly_coach_generated"
    ]
    key_week = next(
        week for week in scenario.weeks if week.week_start == selection.coach_week_start
    )
    key_digest_event = next(
        event
        for event in fixture.trace_events
        if event.event_type == "weekly_digest_built"
        and event.event_id in key_week.event_ids
    )
    unavailable_reason = _coach_unavailable_reason(
        expected_coach, key_digest_event.details.digest
    )
    if unavailable_reason is not None:
        if coach_events or key_digest_event.details.digest.coach_narrative is not None:
            raise ValueError(
                "Scenario has an incompatible historical Coach Digest response"
            )
        if key_digest_event.details.coach_unavailable_reason != unavailable_reason:
            raise ValueError("Scenario Coach Digest source diagnostic differs")
    else:
        if len(coach_events) != 1:
            raise ValueError(
                "Scenario must have one compatible key-week Coach Digest response"
            )
        coach_event = coach_events[0]
        if coach_event.event_id not in key_week.event_ids:
            raise ValueError("Coach Digest response is attached to the wrong week")
        if (
            expected_coach is None
            or coach_event.details.narrative != expected_coach.narrative
        ):
            raise ValueError("Coach Digest event differs from the saved response")

    entries = scenario.journal_entries
    coordinates = [(entry.date, entry.t_index) for entry in entries]
    if coordinates != sorted(coordinates):
        raise ValueError("Scenario Journal Entries are not in temporal order")
    journal_ids = {entry.journal_entry_id for entry in entries}
    if len(journal_ids) != len(entries):
        raise ValueError("Scenario Journal Entry IDs repeat")
    weeks = scenario.weeks
    if [(week.week_start, week.week_end) for week in weeks] != sorted(
        (week.week_start, week.week_end) for week in weeks
    ):
        raise ValueError("Scenario weeks are not in temporal order")
    assigned_journal_ids = [
        journal_id for week in weeks for journal_id in week.journal_entry_ids
    ]
    if (
        len(assigned_journal_ids) != len(set(assigned_journal_ids))
        or set(assigned_journal_ids) != journal_ids
    ):
        raise ValueError("Scenario weeks must partition all Journal Entries")
    entry_by_id = {entry.journal_entry_id: entry for entry in entries}
    for week in weeks:
        if week.week_start > week.week_end:
            raise ValueError("Scenario week starts after it ends")
        if any(
            not (week.week_start <= entry_by_id[journal_id].date <= week.week_end)
            for journal_id in week.journal_entry_ids
        ):
            raise ValueError("Scenario week contains an out-of-window Journal Entry")

    event_ids = [event.event_id for event in fixture.trace_events]
    if event_ids != scenario.trace_event_ids:
        raise ValueError("Scenario trace order differs from its trace events")
    position = {event_id: index for index, event_id in enumerate(event_ids)}
    for event in fixture.trace_events:
        if event.parent_event_id is not None and (
            position[event.parent_event_id] >= position[event.event_id]
        ):
            raise ValueError("Scenario trace parent is not earlier than its child")
    assigned_event_ids = [event_id for week in weeks for event_id in week.event_ids]
    if len(assigned_event_ids) != len(set(assigned_event_ids)) or set(
        assigned_event_ids
    ) != set(event_ids):
        raise ValueError("Scenario weeks must partition all trace events")

    prompt_hashes = [
        event.details.request.prompt_sha256
        for event in fixture.trace_events
        if event.event_type == "weekly_review_requested"
    ]
    if scenario.manifest.prompt_sha256 != _sha256_json(prompt_hashes):
        raise ValueError("Scenario prompt-set hash differs from trace requests")
    initial, visible_events = project_scenario_week(fixture, weeks[0].week_id)
    if initial != fixture.session:
        raise ValueError("Loaded session is not the first replay week")
    if {event.event_id for event in visible_events} != set(initial.trace_event_ids):
        raise ValueError("Loaded session trace visibility differs")


def load_scenario_file(
    path: Path,
    *,
    root: Path,
    expected_content_sha256: str | None = None,
) -> ContractFixtureSet:
    """Load one saved replay and verify content, provenance, and time boundaries."""
    raw = path.read_bytes()
    actual_hash = _sha256_bytes(raw)
    if expected_content_sha256 is not None and actual_hash != expected_content_sha256:
        raise ValueError(f"Scenario content hash mismatch: {path}")
    fixture = cast(ContractFixtureSet, ContractFixtureSet.model_validate_json(raw))
    _validate_fixture_semantics(fixture, root=root.resolve())
    return fixture


def load_scenario_catalog(
    root: Path,
) -> tuple[ScenarioCatalog, dict[str, ContractFixtureSet]]:
    """Load and verify the full five-persona menu."""
    root = root.resolve()
    catalog = ScenarioCatalog.model_validate_json((root / CATALOG_PATH).read_bytes())
    fixtures: dict[str, ContractFixtureSet] = {}
    for item in catalog.scenarios:
        fixture = load_scenario_file(
            root / SCENARIO_DIRECTORY / item.file,
            root=root,
            expected_content_sha256=item.content_sha256,
        )
        if (
            fixture.scenario.scenario_id != item.scenario_id
            or fixture.scenario.persona_id != item.persona_id
            or fixture.scenario.profile.top_values != item.core_values
            or item.key_week_start
            != _selection_by_scenario_id(item.scenario_id).coach_week_start
        ):
            raise ValueError(f"Scenario catalog identity mismatch: {item.scenario_id}")
        fixtures[item.scenario_id] = fixture
    return catalog, fixtures


def export_scenarios(root: Path) -> ScenarioCatalog:
    """Regenerate all five saved replays and their content-hashed catalog."""
    root = root.resolve()
    output_directory = root / SCENARIO_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    data = load_review_data(root)
    prompt_rows = _read_jsonl(root / PROMPTS_PATH)
    response_rows = _read_jsonl(root / RESPONSES_PATH)
    coach_responses = load_saved_coach_responses(root)
    catalog_items: list[ScenarioCatalogItem] = []
    for selection in SELECTIONS:
        fixture = build_scenario_fixture(
            root,
            selection,
            data=data,
            prompt_rows=prompt_rows,
            response_rows=response_rows,
            coach_responses=coach_responses,
        )
        filename = f"{selection.scenario_id}.json"
        payload = (
            json.dumps(
                fixture.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        (output_directory / filename).write_bytes(payload)
        profile = data.profiles[selection.persona_id]
        catalog_items.append(
            ScenarioCatalogItem(
                scenario_id=selection.scenario_id,
                file=filename,
                content_sha256=_sha256_bytes(payload),
                persona_id=selection.persona_id,
                persona_name=str(profile["name"]),
                age=str(profile["age"]),
                profession=str(profile["profession"]),
                culture=str(profile["culture"]),
                core_values=[
                    str(value) for value in fixture.scenario.profile.top_values
                ],
                role=selection.role,
                progression=[
                    week.expected_delivery_state for week in fixture.scenario.weeks
                ],
                summary=selection.summary,
                recommended=selection.recommended,
                key_week_start=selection.coach_week_start,
            )
        )
    catalog = ScenarioCatalog(scenarios=catalog_items)
    (root / CATALOG_PATH).write_text(
        json.dumps(catalog.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    load_scenario_catalog(root)
    return catalog


if __name__ == "__main__":
    export_scenarios(Path.cwd())
