"""Pydantic schemas for Weekly Drift Detection output records."""

from typing import Literal

from pydantic import BaseModel, Field

CoachResponseMode = Literal[
    "active_drift",
    "no_active_drift",
    "insufficient_evidence",
    "active",
    "recovered",
    "uncertain",
    "mixed",
    "stable",
    "rut",
    "crash",
    "evolution",
    "high_uncertainty",
    "mixed_state",
    "background_strain",
]
CoachDigestPolicy = Literal[
    "drift_detected",
    "no_current_drift",
    "more_reflection_needed",
]
CoreValueDriftState = Literal[
    "active_drift",
    "no_active_drift",
    "insufficient_evidence",
]
CoreValueStateChange = Literal[
    "unchanged",
    "active_drift_started",
    "active_drift_restarted",
    "active_drift_ended",
    "evidence_became_insufficient",
    "evidence_resolved",
]

DriftDimensionClassification = Literal["stable", "evolution", "drift"]
DriftTriggerType = Literal[
    "stable",
    "rut",
    "crash",
    "evolution",
    "high_uncertainty",
    "mixed_state",
    "background_strain",
    "acknowledgement",
]


class DimensionDigest(BaseModel):
    """Per-dimension summary over one Weekly Drift Detection output window."""

    dimension: str = Field(description="Schwartz value dimension in snake_case")
    mean_score: float = Field(description="Mean alignment score in [-1, 1]")
    pct_neg: float = Field(description="Percentage of -1 labels in [0, 1]")
    pct_neutral: float = Field(description="Percentage of 0 labels in [0, 1]")
    pct_pos: float = Field(description="Percentage of +1 labels in [0, 1]")


class EvidenceSnippet(BaseModel):
    """Representative evidence excerpt for a Weekly Drift Detection output."""

    date: str
    t_index: int = Field(ge=0)
    direction: Literal[
        "misaligned",
        "aligned",
        "strain",
        "recovery",
        "context",
    ]
    dimensions: list[str]
    score_mean: float | None = None
    excerpt: str


class CoreValueDigestDetail(BaseModel):
    """Compact current state metadata supplied to the Coach Digest."""

    state: CoreValueDriftState
    current_run_length: int = Field(ge=0)
    last_decision: Literal["conflict", "not_conflict", "abstain"]
    last_review_status: Literal["ok", "refusal", "invalid", "error"]
    last_t_index: int = Field(ge=0)
    last_date: str


class CoreValueWeekComparison(BaseModel):
    """Deterministic prior-week comparison for one Core Value."""

    core_value: str
    previous_week_start: str
    previous_week_end: str
    current_week_start: str
    current_week_end: str
    previous_state: CoreValueDriftState
    current_state: CoreValueDriftState
    change: CoreValueStateChange
    end_reason: Literal["not_conflict", "abstain", "gap"] | None = None
    previous_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    current_evidence: list[EvidenceSnippet] = Field(default_factory=list)


class JournalHistoryEntry(BaseModel):
    """Sanitized Journal Entry used to build Weekly Drift Detection output."""

    date: str
    t_index: int = Field(ge=0)
    content: str
    has_response: bool = False


class CoachNarrative(BaseModel):
    """Structured Coach Digest response."""

    weekly_mirror: str
    tension_explanation: str
    reflective_question: str


class ValidationCheck(BaseModel):
    """One Coach Digest Validation."""

    name: str
    passed: bool
    details: str


class DigestValidation(BaseModel):
    """Coach Digest Validations for one Coach Digest response."""

    grounded_quotes: list[str] = Field(default_factory=list)
    word_count: int = Field(ge=0)
    checks: list[ValidationCheck] = Field(default_factory=list)

    @property
    def groundedness_passed(self) -> bool:
        return any(
            check.name == "groundedness" and check.passed for check in self.checks
        )

    @property
    def non_circularity_passed(self) -> bool:
        return any(
            check.name == "non_circularity" and check.passed for check in self.checks
        )

    @property
    def value_leakage_passed(self) -> bool:
        return any(
            check.name == "value_leakage" and check.passed for check in self.checks
        )

    @property
    def length_passed(self) -> bool:
        return any(check.name == "length" and check.passed for check in self.checks)

    @property
    def all_passed(self) -> bool:
        """Return whether every required Coach Digest Validation passed."""
        required = {
            "groundedness",
            "non_circularity",
            "value_leakage",
            "state_claims",
            "length",
        }
        observed = {check.name for check in self.checks}
        return required <= observed and all(check.passed for check in self.checks)


class DriftDetectionResult(BaseModel):
    """Deprecated compatibility result used to build structured output."""

    class DimensionSignal(BaseModel):
        """Per-dimension drift/evolution summary for one weekly decision."""

        dimension: str
        classification: DriftDimensionClassification
        mean_alignment: float
        mean_uncertainty: float
        trigger: DriftTriggerType | None = None
        residual: float | None = None
        volatility: float | None = None

    response_mode: CoachResponseMode
    rationale: str = Field(
        description=(
            "Human-readable explanation for why compatibility routing selected "
            "this mode."
        )
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Optional machine-readable or audit-friendly supporting reasons.",
    )
    source: str = Field(
        default="drift_detector",
        description=(
            "Upstream component that produced the mode, typically drift_detector."
        ),
    )
    trigger_type: DriftTriggerType | None = Field(
        default=None,
        description="Primary trigger category that produced the response mode.",
    )
    week_start: str | None = None
    week_end: str | None = None
    overall_mean: float | None = None
    overall_uncertainty: float | None = None
    triggered_dimensions: list[str] = Field(default_factory=list)
    dimension_signals: list[DimensionSignal] = Field(default_factory=list)
    profile_update: dict[str, float] | None = Field(
        default=None,
        description="Optional suggested profile weights when evolution is detected.",
    )


class WeeklyDigest(BaseModel):
    """Structured Weekly Drift Detection output used by the Coach Digest."""

    persona_id: str
    persona_name: str | None = None
    week_start: str
    week_end: str
    response_mode: CoachResponseMode
    mode_source: str = Field(
        description=(
            "How the response mode was assigned, such as fallback_heuristic "
            "or drift_detector."
        )
    )
    mode_rationale: str = Field(
        description="Short explanation for why this response mode was selected."
    )
    signal_source: str = Field(
        default="judge_labels",
        description="Origin of the numeric signals used to build this digest.",
    )
    n_entries: int = Field(ge=1)
    overall_mean: float | None
    overall_uncertainty: float | None = None
    core_values: list[str] = Field(default_factory=list)
    goal_context: str | None = None
    drift_states: dict[str, CoreValueDriftState] = Field(default_factory=dict)
    drift_details: dict[str, CoreValueDigestDetail] = Field(default_factory=dict)
    state_comparisons: list[CoreValueWeekComparison] = Field(default_factory=list)
    drift_reasons: list[str] = Field(default_factory=list)
    top_tensions: list[str]
    top_strengths: list[str]
    dimensions: list[DimensionDigest]
    evidence: list[EvidenceSnippet]
    coach_narrative: CoachNarrative | None = None
    validation: DigestValidation | None = None


WEEKLY_DIGEST_COACH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "weekly_mirror": {"type": "string"},
        "tension_explanation": {"type": "string"},
        "reflective_question": {"type": "string"},
    },
    "required": ["weekly_mirror", "tension_explanation", "reflective_question"],
}

WEEKLY_DIGEST_COACH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "WeeklyDigestCoachNarrative",
    "schema": WEEKLY_DIGEST_COACH_SCHEMA,
    "strict": True,
}
