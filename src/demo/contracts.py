"""Versioned wire contracts for the React Experience and Inspect demo."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.coach.schemas import CoachNarrative, DigestValidation, WeeklyDigest
from src.drift_detector import DriftDetectorResult
from src.models.nudge import NudgeCategory
from src.weekly_drift_reviewer import (
    WeeklyDriftReviewerDecision,
    WeeklyDriftReviewerReceipt,
    WeeklyDriftReviewerRequest,
)

CONTRACT_VERSION: Literal["experience-inspect-v1"] = "experience-inspect-v1"
SCHEMA_ID = "https://twinkl.local/contracts/experience-inspect-v1.schema.json"

CoreValue = Literal[
    "self_direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power",
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
]
BwsObjectKey = Literal[
    "power",
    "achievement",
    "hedonism",
    "stimulation",
    "self_direction",
    "universalism_nature",
    "benevolence",
    "tradition",
    "conformity",
    "security",
    "universalism_social",
]
GoalCategory = Literal[
    "work_life_balance",
    "life_transition",
    "relationships",
    "health_wellbeing",
    "direction",
    "meaningful_work",
]
EventStatus = Literal[
    "queued",
    "running",
    "complete",
    "reused",
    "refused",
    "invalid",
    "failed",
]
EventSource = Literal["saved_replay", "live_run"]
Operation = Literal[
    "create_session",
    "submit_journal_entry",
    "load_scenario",
    "read_trace",
]

CORE_VALUE_ORDER = (
    "self_direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power",
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
)
BWS_OBJECT_ORDER = (
    "power",
    "achievement",
    "hedonism",
    "stimulation",
    "self_direction",
    "universalism_nature",
    "benevolence",
    "tradition",
    "conformity",
    "security",
    "universalism_social",
)
BWS_SETS = (
    (
        "achievement",
        "universalism_nature",
        "benevolence",
        "tradition",
        "security",
        "universalism_social",
    ),
    (
        "power",
        "hedonism",
        "benevolence",
        "tradition",
        "conformity",
        "universalism_social",
    ),
    (
        "power",
        "achievement",
        "stimulation",
        "tradition",
        "conformity",
        "security",
    ),
    (
        "achievement",
        "hedonism",
        "self_direction",
        "conformity",
        "security",
        "universalism_social",
    ),
    (
        "power",
        "hedonism",
        "stimulation",
        "universalism_nature",
        "security",
        "universalism_social",
    ),
    (
        "power",
        "achievement",
        "stimulation",
        "self_direction",
        "benevolence",
        "universalism_social",
    ),
    (
        "power",
        "achievement",
        "hedonism",
        "self_direction",
        "universalism_nature",
        "tradition",
    ),
    (
        "achievement",
        "hedonism",
        "stimulation",
        "universalism_nature",
        "benevolence",
        "conformity",
    ),
    (
        "hedonism",
        "stimulation",
        "self_direction",
        "benevolence",
        "tradition",
        "security",
    ),
    (
        "stimulation",
        "self_direction",
        "universalism_nature",
        "tradition",
        "conformity",
        "universalism_social",
    ),
    (
        "power",
        "self_direction",
        "universalism_nature",
        "benevolence",
        "conformity",
        "security",
    ),
)


class ContractModel(BaseModel):
    """Strict base for data crossing the React-Python boundary."""

    model_config = ConfigDict(extra="forbid")


class WeeklyDriftReviewerDecisionContract(WeeklyDriftReviewerDecision):
    """Strict wire form of the existing Weekly Drift Reviewer Decision."""

    model_config = ConfigDict(extra="forbid")


class WeeklyDriftReviewerRequestContract(WeeklyDriftReviewerRequest):
    """Strict wire form of the existing Weekly Drift Reviewer request."""

    model_config = ConfigDict(extra="forbid")


class WeeklyDriftReviewerReceiptContract(WeeklyDriftReviewerReceipt):
    """Strict wire form of the existing persisted receipt."""

    model_config = ConfigDict(extra="forbid")
    decisions: list[WeeklyDriftReviewerDecisionContract]  # type: ignore[assignment]


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class BwsResponse(ContractModel):
    """One response in the canonical 11-group SVBWS design."""

    set_number: int = Field(ge=1, le=11)
    items: list[BwsObjectKey] = Field(min_length=6, max_length=6)
    item_order_shown: list[BwsObjectKey] = Field(min_length=6, max_length=6)
    selected_best: BwsObjectKey
    selected_worst: BwsObjectKey
    response_time_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_group(self) -> BwsResponse:
        expected = set(BWS_SETS[self.set_number - 1])
        if set(self.items) != expected or set(self.item_order_shown) != expected:
            raise ValueError("SVBWS response must contain its canonical group")
        if self.selected_best not in expected or self.selected_worst not in expected:
            raise ValueError("SVBWS choices must belong to the group")
        if self.selected_best == self.selected_worst:
            raise ValueError("Most and Least choices must differ")
        return self


class BwsResults(ContractModel):
    """Raw 11-object SVBWS scores kept separate from the Profile transform."""

    appearances: dict[BwsObjectKey, int]
    best_counts: dict[BwsObjectKey, int]
    worst_counts: dict[BwsObjectKey, int]
    net_counts: dict[BwsObjectKey, int]
    scores: dict[BwsObjectKey, float]

    @model_validator(mode="after")
    def validate_vectors(self) -> BwsResults:
        expected = set(BWS_OBJECT_ORDER)
        for vector in (
            self.appearances,
            self.best_counts,
            self.worst_counts,
            self.net_counts,
            self.scores,
        ):
            if set(vector) != expected:
                raise ValueError("SVBWS vectors must contain all 11 objects")
        return self


class ProfileTransform(ContractModel):
    """Ten-value product transform derived from the raw SVBWS result."""

    method: Literal["mean_universalism_facets_then_shift_normalize_v1"]
    scores: dict[CoreValue, float]
    weights: dict[CoreValue, float]
    top_values: list[CoreValue] = Field(min_length=1)
    bottom_values: list[CoreValue] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_vectors(self) -> ProfileTransform:
        expected = set(CORE_VALUE_ORDER)
        if set(self.scores) != expected or set(self.weights) != expected:
            raise ValueError("Profile vectors must contain all ten values")
        return self


class ProfileProvenance(ContractModel):
    source: Literal["react_onboarding_poc"]
    set_order_randomized: Literal[True]
    card_order_randomized: Literal[True]


class OnboardingProfile(ContractModel):
    """Authoritative confirmed Profile shape emitted by React onboarding."""

    schema_version: Literal[2]
    onboarding_version: Literal["2.1.0"]
    instrument: Literal["svbws_lee_soutar_louviere_2008_ui_adaptation_v2"]
    scoring_method: Literal["best_minus_worst_divided_by_appearances_v1"]
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    started_at: str
    timestamp: str
    bws_responses: list[BwsResponse] = Field(min_length=11, max_length=11)
    bws_results: BwsResults
    value_profile: ProfileTransform
    top_values: list[CoreValue] = Field(min_length=1)
    goal_category: GoalCategory
    user_confirmed: Literal[True]
    provenance: ProfileProvenance

    @field_validator("started_at", "timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        _parse_timestamp(value)
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> OnboardingProfile:
        if {row.set_number for row in self.bws_responses} != set(range(1, 12)):
            raise ValueError("A confirmed Profile requires all 11 SVBWS groups")
        if self.top_values != self.value_profile.top_values:
            raise ValueError("Profile top_values must match value_profile.top_values")
        canonical = [value for value in CORE_VALUE_ORDER if value in self.top_values]
        if self.top_values != canonical:
            raise ValueError("Profile top_values must use canonical order")
        return self


class JournalEntry(ContractModel):
    journal_entry_id: str = Field(min_length=1)
    t_index: int = Field(ge=0)
    date: str
    content: str = Field(min_length=1)
    nudge_response: str | None = None


class NudgeInteraction(ContractModel):
    nudge_id: str = Field(min_length=1)
    journal_entry_id: str = Field(min_length=1)
    outcome: Literal["suppressed", "no_nudge", "displayed", "skipped", "answered"]
    category: NudgeCategory | None = None
    reason: str | None = None
    text: str | None = None
    response: str | None = None


class ModelContract(ContractModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: str | None = None


class ResourceRef(ContractModel):
    kind: Literal[
        "profile",
        "journal_entry",
        "week",
        "nudge",
        "weekly_review",
        "drift",
        "weekly_digest",
        "weekly_coach",
        "event",
    ]
    id: str = Field(min_length=1)


class EventValidation(ContractModel):
    valid: bool
    schema_name: str = Field(min_length=1)
    errors: list[str] = Field(default_factory=list)


class SafeError(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool

    @field_validator("message")
    @classmethod
    def reject_secret_material(cls, value: str) -> str:
        lowered = value.lower()
        if "sk-" in lowered or "api_key=" in lowered or "authorization:" in lowered:
            raise ValueError("Safe errors must not contain provider secrets")
        return value


class TraceEventBase(ContractModel):
    """Fields shared by every Inspect trace event."""

    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    parent_event_id: str | None = None
    event_type: str
    status: EventStatus
    source: EventSource
    started_at: str
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    input_refs: list[ResourceRef] = Field(default_factory=list)
    model_contract: ModelContract | None = None
    prompt: str | None = None
    raw_response: Any | None = None
    validation: EventValidation | None = None
    result_refs: list[ResourceRef] = Field(default_factory=list)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    error: SafeError | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> TraceEventBase:
        _parse_timestamp(self.started_at)
        terminal = {"complete", "reused", "refused", "invalid", "failed"}
        failures = {"refused", "invalid", "failed"}
        if self.status in terminal:
            if self.completed_at is None or self.duration_ms is None:
                raise ValueError("Terminal trace events require completion timing")
            if _parse_timestamp(self.completed_at) < _parse_timestamp(self.started_at):
                raise ValueError("Trace completion cannot precede its start")
        elif self.completed_at is not None or self.duration_ms is not None:
            raise ValueError("Queued or running events cannot be complete")
        if self.status in failures and self.error is None:
            raise ValueError("Refused, invalid, and failed events require a safe error")
        if self.status in {"complete", "reused"} and self.error is not None:
            raise ValueError("Successful events cannot contain an error")
        if self.status == "invalid" and (
            self.validation is None or self.validation.valid
        ):
            raise ValueError("Invalid events require a failed validation result")
        return self


class ProfileConfirmedDetails(ContractModel):
    profile: OnboardingProfile


class JournalEntrySubmittedDetails(ContractModel):
    journal_entry: JournalEntry
    ordering_valid: bool


class NudgeSuppressionDetails(ContractModel):
    previous_entry_ids: list[str]
    window_size: Literal[3] = 3
    max_nudges: Literal[2] = 2
    suppressed: bool


class NudgeDecisionDetails(ContractModel):
    should_nudge: bool
    category: NudgeCategory | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> NudgeDecisionDetails:
        if self.should_nudge != (self.category is not None):
            raise ValueError("Nudge category must agree with should_nudge")
        return self


class NudgeGeneratedDetails(ContractModel):
    nudge: NudgeInteraction | None = None
    word_count: int | None = Field(default=None, ge=0)
    attempts: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_result(self) -> NudgeGeneratedDetails:
        if (self.nudge is None) != (self.word_count is None):
            raise ValueError("Nudge and word count must be present together")
        return self


class WeeklyReviewRequestedDetails(ContractModel):
    request: WeeklyDriftReviewerRequestContract


class WeeklyReviewCompletedDetails(ContractModel):
    receipt: WeeklyDriftReviewerReceiptContract


class DriftRuleStep(ContractModel):
    t_index: int = Field(ge=0)
    core_value: CoreValue
    verdict: Literal["conflict", "not_conflict", "abstain"]
    pending_run_length: int = Field(ge=0)
    effect: Literal["start", "confirm", "extend", "recover", "uncertain", "break"]


class DriftDetectedDetails(ContractModel):
    decisions: list[WeeklyDriftReviewerDecisionContract] = Field(min_length=1)
    steps: list[DriftRuleStep] = Field(min_length=1)
    result: DriftDetectorResult


class WeeklyDigestBuiltDetails(ContractModel):
    digest: WeeklyDigest
    cited_journal_entry_ids: list[str]


class WeeklyCoachGeneratedDetails(ContractModel):
    narrative: CoachNarrative
    validation: DigestValidation


class ProfileConfirmedEvent(TraceEventBase):
    event_type: Literal["profile_confirmed"]
    details: ProfileConfirmedDetails


class JournalEntrySubmittedEvent(TraceEventBase):
    event_type: Literal["journal_entry_submitted"]
    details: JournalEntrySubmittedDetails


class NudgeSuppressionCheckedEvent(TraceEventBase):
    event_type: Literal["nudge_suppression_checked"]
    details: NudgeSuppressionDetails


class NudgeDecidedEvent(TraceEventBase):
    event_type: Literal["nudge_decided"]
    details: NudgeDecisionDetails


class NudgeGeneratedEvent(TraceEventBase):
    event_type: Literal["nudge_generated"]
    details: NudgeGeneratedDetails


class WeeklyReviewRequestedEvent(TraceEventBase):
    event_type: Literal["weekly_review_requested"]
    details: WeeklyReviewRequestedDetails

    @model_validator(mode="after")
    def validate_reviewer_contract(self) -> WeeklyReviewRequestedEvent:
        contract = self.model_contract
        if contract is None or (
            contract.model != "gpt-5.6-luna" or contract.reasoning_effort != "low"
        ):
            raise ValueError("Weekly Drift Reviewer must use gpt-5.6-luna at low")
        return self


class WeeklyReviewCompletedEvent(TraceEventBase):
    event_type: Literal["weekly_review_completed"]
    details: WeeklyReviewCompletedDetails

    @model_validator(mode="after")
    def validate_reviewer_contract(self) -> WeeklyReviewCompletedEvent:
        contract = self.model_contract
        if contract is None or (
            contract.model != "gpt-5.6-luna" or contract.reasoning_effort != "low"
        ):
            raise ValueError("Weekly Drift Reviewer must use gpt-5.6-luna at low")
        return self


class DriftDetectedEvent(TraceEventBase):
    event_type: Literal["drift_detected"]
    details: DriftDetectedDetails


class WeeklyDigestBuiltEvent(TraceEventBase):
    event_type: Literal["weekly_digest_built"]
    details: WeeklyDigestBuiltDetails


class WeeklyCoachGeneratedEvent(TraceEventBase):
    event_type: Literal["weekly_coach_generated"]
    details: WeeklyCoachGeneratedDetails


TraceEvent = Annotated[
    ProfileConfirmedEvent
    | JournalEntrySubmittedEvent
    | NudgeSuppressionCheckedEvent
    | NudgeDecidedEvent
    | NudgeGeneratedEvent
    | WeeklyReviewRequestedEvent
    | WeeklyReviewCompletedEvent
    | DriftDetectedEvent
    | WeeklyDigestBuiltEvent
    | WeeklyCoachGeneratedEvent,
    Field(discriminator="event_type"),
]


class SessionSelection(ContractModel):
    view: Literal["experience", "inspect"]
    selected_week: str | None = None
    selected_journal_entry_id: str | None = None
    selected_event_id: str | None = None


class ExperienceSession(ContractModel):
    """Serializable state shared by Experience and Inspect."""

    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    session_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    profile: OnboardingProfile
    journal_entries: list[JournalEntry] = Field(default_factory=list)
    nudges: list[NudgeInteraction] = Field(default_factory=list)
    weekly_reviewer_decisions: list[WeeklyDriftReviewerDecisionContract] = Field(
        default_factory=list
    )
    drift_result: DriftDetectorResult | None = None
    weekly_digest: WeeklyDigest | None = None
    trace_event_ids: list[str] = Field(default_factory=list)
    selection: SessionSelection
    updated_at: str

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: str) -> str:
        _parse_timestamp(value)
        return value

    @model_validator(mode="after")
    def validate_identity_and_order(self) -> ExperienceSession:
        if self.profile.session_id != self.session_id:
            raise ValueError("Session and Profile session_id must match")
        indices = [entry.t_index for entry in self.journal_entries]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("Journal Entries must have unique chronological t_index")
        if len(self.trace_event_ids) != len(set(self.trace_event_ids)):
            raise ValueError("Session trace_event_ids must be unique")
        return self


class ScenarioWeek(ContractModel):
    week_id: str
    week_start: str
    week_end: str
    journal_entry_ids: list[str]
    event_ids: list[str]
    expected_delivery_state: Literal[
        "stable", "active", "recovered", "uncertain", "mixed"
    ]


class ScenarioManifest(ContractModel):
    bundle_version: Literal["scenario-bundle-v1"]
    created_at: str
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_files: list[str] = Field(min_length=1)
    model_contract: ModelContract
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioBundle(ContractModel):
    """Deterministic saved replay consumed by Experience and Inspect."""

    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    scenario_id: str
    title: str
    description: str
    source: Literal["saved_replay"]
    persona_id: str
    profile: OnboardingProfile
    journal_entries: list[JournalEntry] = Field(min_length=1)
    weekly_reviewer_decisions: list[WeeklyDriftReviewerDecisionContract] = Field(
        min_length=1
    )
    drift_result: DriftDetectorResult
    weekly_digest: WeeklyDigest
    weeks: list[ScenarioWeek] = Field(min_length=1)
    trace_event_ids: list[str] = Field(min_length=1)
    manifest: ScenarioManifest

    @model_validator(mode="after")
    def validate_model_contract(self) -> ScenarioBundle:
        contract = self.manifest.model_contract
        if contract.model != "gpt-5.6-luna" or contract.reasoning_effort != "low":
            raise ValueError("Saved scenarios must preserve the Luna-low contract")
        return self


class SessionCreateRequest(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["create_session"]
    request_id: str
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: OnboardingProfile


class JournalEntrySubmitRequest(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["submit_journal_entry"]
    request_id: str
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_id: str
    expected_revision: int = Field(ge=0)
    journal_entry: JournalEntry


class ScenarioLoadRequest(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["load_scenario"]
    request_id: str
    scenario_id: str


class TraceReadRequest(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["read_trace"]
    request_id: str
    session_id: str
    after_event_id: str | None = None


ApiRequest = Annotated[
    SessionCreateRequest
    | JournalEntrySubmitRequest
    | ScenarioLoadRequest
    | TraceReadRequest,
    Field(discriminator="operation"),
]


class SessionCreatedResponse(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["create_session"]
    request_id: str
    status: Literal["ok"]
    session: ExperienceSession


class JournalEntrySubmittedResponse(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["submit_journal_entry"]
    request_id: str
    status: Literal["ok"]
    session: ExperienceSession
    event_ids: list[str] = Field(min_length=1)


class ScenarioLoadedResponse(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["load_scenario"]
    request_id: str
    status: Literal["ok"]
    session: ExperienceSession
    scenario: ScenarioBundle
    event_ids: list[str] = Field(min_length=1)


class TraceReadResponse(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["read_trace"]
    request_id: str
    status: Literal["ok"]
    session_id: str
    events: list[TraceEvent]


class ApiErrorResponse(ContractModel):
    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    operation: Literal["error"]
    requested_operation: Operation
    request_id: str
    status: Literal["error"]
    error: SafeError


ApiResponse = Annotated[
    SessionCreatedResponse
    | JournalEntrySubmittedResponse
    | ScenarioLoadedResponse
    | TraceReadResponse
    | ApiErrorResponse,
    Field(discriminator="operation"),
]


class ContractFixtureSet(ContractModel):
    """Canonical examples consumed by Python and React contract tests."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"$id": SCHEMA_ID},
    )

    schema_version: Literal["experience-inspect-v1"] = CONTRACT_VERSION
    session: ExperienceSession
    scenario: ScenarioBundle
    requests: list[ApiRequest]
    responses: list[ApiResponse]
    trace_events: list[TraceEvent]

    @model_validator(mode="after")
    def validate_links(self) -> ContractFixtureSet:
        event_ids = [event.event_id for event in self.trace_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Trace event IDs must be unique")
        known = set(event_ids)
        for event in self.trace_events:
            if event.session_id != self.session.session_id:
                raise ValueError("Fixture trace events must belong to the session")
            if event.parent_event_id is not None and event.parent_event_id not in known:
                raise ValueError("Trace parent_event_id must reference a known event")
        if not set(self.session.trace_event_ids).issubset(known):
            raise ValueError("Session trace_event_ids must reference fixture events")
        if not set(self.scenario.trace_event_ids).issubset(known):
            raise ValueError("Scenario trace_event_ids must reference fixture events")
        return self
