"""Source-bound paired Coach Digest receipts for the five saved demo Personas."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from prompts import get_prompt_metadata, load_prompt
from src.coach.schemas import (
    WEEKLY_DIGEST_COACH_RESPONSE_FORMAT,
    CoachDigestDiagnostic,
    CoachNarrative,
    DigestValidation,
    LLMCallMetrics,
    ValidationCheck,
    WeeklyDigest,
)
from src.coach.weekly_digest import (
    LLMCompleteFn,
    _extract_quoted_phrases,
    render_digest_messages,
    validate_weekly_digest_narrative,
)
from src.north_star.runtime import MomentMode, NorthStarRecord, _timestamp
from src.prompt_boundary import render_live_prompt_receipt, serialize_untrusted_data

PROMPT_NAME: Literal["demo_coach_nsm_comparison"] = "demo_coach_nsm_comparison"
PROMPT_VERSION: Literal["1.0"] = "1.0"
COMPARISONS_PATH = "src/demo/coach_digest_comparisons.json"


def hash_json(value: Any) -> str:
    return hash_text(serialize_untrusted_data(value))


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParentJournalEntry(_Model):
    source_id: str
    date: str
    source_text: str


class NorthStarCoachContext(_Model):
    source_id: str
    source_type: Literal["journal_entry", "nudge_response"]
    date: str
    exact_quote: str
    source_text: str
    mode: MomentMode
    core_value_phrase: str
    parent_journal_entry: ParentJournalEntry | None = None


class CoachComparisonArm(_Model):
    narrative: CoachNarrative
    validation: DigestValidation
    base_prompt: str
    prompt: str
    repair_requirements: list[str] = Field(default_factory=list)
    raw_output: str
    provider: Literal["openai"] = "openai"
    model: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"
    reasoning_effort: Literal["none"] = "none"
    service_tier: Literal["default"] = "default"
    prompt_name: Literal["demo_coach_nsm_comparison"] = PROMPT_NAME
    prompt_version: Literal["1.0"] = PROMPT_VERSION
    call_metrics: list[LLMCallMetrics]
    diagnostic_paths: list[str]
    base_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SavedCoachComparison(_Model):
    schema_version: Literal["coach-digest-nsm-comparison-v1"] = (
        "coach-digest-nsm-comparison-v1"
    )
    scenario_id: str
    persona_id: str
    week_start: str
    week_end: str
    weekly_drift_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    north_star_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    north_star_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    north_star_context: NorthStarCoachContext
    without_north_star: CoachComparisonArm
    with_north_star: CoachComparisonArm


class SavedCoachComparisonFixture(_Model):
    schema_version: Literal["coach-digest-nsm-comparison-fixture-v1"] = (
        "coach-digest-nsm-comparison-fixture-v1"
    )
    comparisons: dict[str, SavedCoachComparison] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_keys(self) -> Self:
        if any(
            key != f"{pair.scenario_id}::{pair.week_start}"
            for key, pair in self.comparisons.items()
        ):
            raise ValueError("Coach comparison fixture key differs from its identity")
        return self


def build_north_star_context(record: NorthStarRecord) -> NorthStarCoachContext:
    """Project a previously validated selection without its assessment metadata."""
    selected = record.selected
    if record.status != "complete" or selected is None or record.mode is None:
        raise ValueError("Comparison requires an accepted North Star Moment")
    sources = [s for s in record.sources if s.entry_id == selected.entry_id]
    if len(sources) != 1 or selected.entry_id not in record.source_ids:
        raise ValueError("Selected North Star Moment source is missing or ambiguous")
    source = sources[0]
    if (source.owner_id, source.date, source.t_index) != (
        record.owner_id,
        selected.date,
        selected.t_index,
    ) or not record.value_phrase:
        raise ValueError("Selected North Star Moment identity differs")
    source_text = getattr(source, selected.quote_source)
    available_at = (
        source.response_available_at
        if selected.quote_source == "nudge_response"
        else source.available_at
    )
    if (
        not source_text
        or not selected.evidence_quote
        or (selected.evidence_quote not in source_text)
    ):
        raise ValueError("Selected quotation does not match its exact source")
    if available_at is None or _timestamp(available_at) > _timestamp(record.cutoff_at):
        raise ValueError("Selected source was unavailable at the reviewed cutoff")
    selected_date = date.fromisoformat(source.date)
    start, end = (
        date.fromisoformat(record.week_start),
        date.fromisoformat(record.week_end),
    )
    if (
        selected_date > end
        or (record.mode == "encouragement" and not start <= selected_date <= end)
        or (record.mode == "reminder" and selected_date >= start)
    ):
        raise ValueError("Selected source date differs from its temporal mode")
    if record.mode == "reflection" and (
        record.onset_t_index is None
        or source.t_index >= record.onset_t_index
        or record.onset_date is None
        or selected_date > date.fromisoformat(record.onset_date)
        or record.onset_available_at is None
        or _timestamp(available_at) >= _timestamp(record.onset_available_at)
    ):
        raise ValueError("Reflection source must precede the active Drift onset")
    return NorthStarCoachContext(
        source_id=source.entry_id,
        source_type=selected.quote_source,
        date=source.date,
        exact_quote=selected.evidence_quote,
        source_text=source_text,
        mode=record.mode,
        core_value_phrase=record.value_phrase,
        parent_journal_entry=ParentJournalEntry(
            source_id=source.entry_id,
            date=source.date,
            source_text=source.journal_entry,
        )
        if selected.quote_source == "nudge_response"
        else None,
    )


def render_demo_comparison_messages(
    digest: WeeklyDigest,
    context: NorthStarCoachContext | None,
    *,
    repair_requirements: Sequence[str] = (),
) -> tuple[str, str]:
    """Both initial arms share instructions; only the optional JSON field varies."""
    if get_prompt_metadata("weekly_digest_coach")["version"] != "4.4":
        raise ValueError("The saved comparison requires base Coach prompt 4.4")
    instructions, input_data = render_digest_messages(digest)
    marker = "\n\nReturn JSON with exactly these keys:"
    if instructions.count(marker) != 1:
        raise ValueError("Coach comparison output boundary changed")
    extension = str(load_prompt(PROMPT_NAME).render()).strip()
    lead, _, output_rules = instructions.partition(marker)
    instructions = f"{lead.rstrip()}\n\n{extension}{marker}{output_rules}"
    if repair_requirements:
        requirements = "\n".join(f"- {item}" for item in repair_requirements)
        instructions += (
            "\n\nA prior response needs revision. Generate a new response. "
            "Follow each requirement below. Do not discuss the revision in the "
            f"response.\n{requirements}"
        )
    data = json.loads(input_data)
    data["north_star_context"] = (
        context.model_dump(mode="json", exclude_none=True) if context else None
    )
    return instructions, serialize_untrusted_data(data)


def render_demo_comparison_prompt(
    digest: WeeklyDigest,
    context: NorthStarCoachContext | None,
    *,
    repair_requirements: Sequence[str] = (),
) -> str:
    instructions, input_data = render_demo_comparison_messages(
        digest, context, repair_requirements=repair_requirements
    )
    return render_live_prompt_receipt(instructions=instructions, input_data=input_data)


def validate_demo_comparison_narrative(
    digest: WeeklyDigest,
    narrative: CoachNarrative,
    context: NorthStarCoachContext | None,
) -> DigestValidation:
    """Retain all base checks and require every quotation to have a supplied source."""
    validation = validate_weekly_digest_narrative(
        digest, narrative, validate_voice=True
    )
    weekly_sources = [item.excerpt for item in digest.evidence]
    sources = [*weekly_sources]
    for comparison in digest.state_comparisons:
        sources.extend(item.excerpt for item in comparison.previous_evidence)
        sources.extend(item.excerpt for item in comparison.current_evidence)
    if context:
        sources.append(context.source_text)
        if context.parent_journal_entry:
            sources.append(context.parent_journal_entry.source_text)
    fields = narrative.model_dump().values()
    quotations = [q for text in fields for q in _extract_quoted_phrases(text)]
    weekly_quotes = [
        q
        for q in _extract_quoted_phrases(narrative.weekly_mirror)
        if any(q in source for source in weekly_sources)
    ]
    ungrounded = [q for q in quotations if not any(q in source for source in sources)]
    checks = [
        ValidationCheck(
            name="weekly_mirror_verbatim",
            passed=bool(weekly_quotes),
            details="Include an exact, case-preserving phrase from evidence_lines "
            "in double quotation marks in weekly_mirror."
            if not weekly_quotes
            else "Weekly Mirror preserves a quoted phrase from weekly evidence.",
        ),
        ValidationCheck(
            name="all_quotes_grounded",
            passed=not ungrounded,
            details="Every quotation matches a supplied source exactly."
            if not ungrounded
            else "Copy every quotation exactly from supplied source text; "
            "remove quotation marks around paraphrases or invented phrases.",
        ),
        ValidationCheck(
            name="one_reflective_question",
            passed=narrative.reflective_question.count("?") == 1,
            details="Return one reflective question, with one question mark.",
        ),
        ValidationCheck(
            name="comparison_metadata_hidden",
            passed=re.search(
                r"north[ -]?star|north_star|without.context|with.context|"
                r"source_id|source_type|comparison arm",
                " ".join(fields),
                re.IGNORECASE,
            )
            is None,
            details="Keep feature names and comparison metadata out of the reflection.",
        ),
    ]
    return validation.model_copy(update={"checks": [*validation.checks, *checks]})


async def generate_demo_comparison_diagnostic(
    digest: WeeklyDigest,
    context: NorthStarCoachContext | None,
    llm_complete: LLMCompleteFn,
    *,
    repair_requirements: Sequence[str] = (),
) -> tuple[CoachDigestDiagnostic, str]:
    """Make exactly one provider call and return its complete validation receipt."""
    instructions, input_data = render_demo_comparison_messages(
        digest, context, repair_requirements=repair_requirements
    )
    prompt = render_live_prompt_receipt(
        instructions=instructions, input_data=input_data
    )
    raw = await llm_complete(
        input_data, WEEKLY_DIGEST_COACH_RESPONSE_FORMAT, instructions
    )
    diagnostic = CoachDigestDiagnostic(
        persona_id=digest.persona_id,
        week_start=digest.week_start,
        week_end=digest.week_end,
        accepted=False,
        raw_output=raw,
    )
    if not raw:
        diagnostic.failure_stage = "no_response"
        diagnostic.failure_details = ["The provider returned no response text."]
        return diagnostic, prompt
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        diagnostic.failure_stage = "json_parse"
        diagnostic.failure_details = [
            "Return valid JSON with the three required fields."
        ]
        return diagnostic, prompt
    try:
        narrative = CoachNarrative.model_validate(payload)
    except ValidationError:
        diagnostic.failure_stage = "schema_validation"
        diagnostic.failure_details = [
            "Return exactly the three required string fields."
        ]
        return diagnostic, prompt
    validation = validate_demo_comparison_narrative(digest, narrative, context)
    diagnostic.narrative = narrative
    diagnostic.validation = validation
    diagnostic.accepted = validation.all_passed
    diagnostic.failure_stage = None if diagnostic.accepted else "coach_validation"
    diagnostic.failure_details = [
        f"{check.name}: {check.details}"
        for check in validation.checks
        if not check.passed
    ]
    return diagnostic, prompt


def validate_saved_comparison(
    comparison: SavedCoachComparison,
    digest: WeeklyDigest,
    record: NorthStarRecord,
    scenario_id: str,
) -> SavedCoachComparison:
    """Fail closed when source, identity, prompt, settings, or response has changed."""
    comparison = SavedCoachComparison.model_validate(comparison.model_dump(mode="json"))
    context = build_north_star_context(record)
    expected = {
        "scenario_id": scenario_id,
        "persona_id": digest.persona_id,
        "week_start": digest.week_start,
        "week_end": digest.week_end,
        "weekly_drift_input_sha256": hash_json(
            digest.model_dump(mode="json", exclude={"coach_narrative", "validation"})
        ),
        "north_star_input_hash": record.input_hash,
        "north_star_context_sha256": hash_json(
            context.model_dump(mode="json", exclude_none=True)
        ),
        "north_star_context": context,
    }
    if (record.owner_id, record.week_start, record.week_end) != (
        digest.persona_id,
        digest.week_start,
        digest.week_end,
    ) or any(getattr(comparison, key) != value for key, value in expected.items()):
        raise ValueError("Coach comparison identity or selected input changed")
    for arm, arm_context in (
        (comparison.without_north_star, None),
        (comparison.with_north_star, context),
    ):
        base_prompt = render_demo_comparison_prompt(digest, arm_context)
        prompt = render_demo_comparison_prompt(
            digest, arm_context, repair_requirements=arm.repair_requirements
        )
        validation = validate_demo_comparison_narrative(
            digest, arm.narrative, arm_context
        )
        if (
            arm.base_prompt != base_prompt
            or arm.prompt != prompt
            or arm.base_prompt_sha256 != hash_text(base_prompt)
            or arm.prompt_sha256 != hash_text(prompt)
            or arm.response_sha256 != hash_json(arm.narrative.model_dump(mode="json"))
            or arm.raw_output_sha256 != hash_text(arm.raw_output)
            or CoachNarrative.model_validate_json(arm.raw_output) != arm.narrative
            or arm.validation != validation
            or not validation.all_passed
            or not arm.call_metrics
            or len(arm.call_metrics) != len(arm.diagnostic_paths)
            or len(set(arm.diagnostic_paths)) != len(arm.diagnostic_paths)
            or any(
                (
                    metric.provider,
                    metric.model,
                    metric.reasoning_effort,
                    metric.service_tier,
                )
                != (arm.provider, arm.model, arm.reasoning_effort, arm.service_tier)
                or metric.status != "completed"
                or metric.calculated_cost_usd is None
                for metric in arm.call_metrics
            )
        ):
            raise ValueError("Coach comparison prompt, response, or receipt changed")
    return comparison
