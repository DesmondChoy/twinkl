"""JSON response schemas for OpenAI structured output in the nudge pipeline.

These schemas are passed as `response_format` to the LLM completion function
to enforce structured JSON responses.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.nudge import NudgeCategory

NudgeRuntimeDecision = Literal[
    "no_nudge",
    "clarification",
    "elaboration",
    "tension_surfacing",
]


class NudgeDecisionAndGenerationResponse(BaseModel):
    """One merged runtime decision and optional generated nudge."""

    model_config = ConfigDict(extra="forbid")

    decision: NudgeRuntimeDecision
    reason: str = Field(min_length=1)
    nudge_text: str | None

    @model_validator(mode="after")
    def validate_nudge_text(self) -> "NudgeDecisionAndGenerationResponse":
        if self.decision == "no_nudge":
            if self.nudge_text is not None:
                raise ValueError("no_nudge requires null nudge_text")
            return self

        if self.nudge_text is None:
            raise ValueError("A nudge decision requires nudge_text")
        stripped = self.nudge_text.strip()
        word_count = len(stripped.split())
        if not 2 <= word_count <= 12:
            raise ValueError("nudge_text must contain 2-12 words")
        self.nudge_text = stripped
        return self

    @property
    def category(self) -> NudgeCategory | None:
        """Return the historical category when a nudge is requested."""
        if self.decision == "no_nudge":
            return None
        return self.decision


NUDGE_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["no_nudge", "clarification", "elaboration", "tension_surfacing"],
        },
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
}

NUDGE_DECISION_AND_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "no_nudge",
                "clarification",
                "elaboration",
                "tension_surfacing",
            ],
        },
        "reason": {"type": "string"},
        "nudge_text": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
    },
    "required": ["decision", "reason", "nudge_text"],
}

NUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "nudge_text": {"type": "string"},
    },
    "required": ["nudge_text"],
}

NUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "content": {"type": "string"},
    },
    "required": ["content"],
}

NUDGE_DECISION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "NudgeDecision",
    "schema": NUDGE_DECISION_SCHEMA,
    "strict": True,
}

NUDGE_DECISION_AND_GENERATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "NudgeDecisionAndGeneration",
    "schema": NUDGE_DECISION_AND_GENERATION_SCHEMA,
    "strict": True,
}

NUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "Nudge",
    "schema": NUDGE_SCHEMA,
    "strict": True,
}

NUDGE_RESPONSE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "NudgeResponse",
    "schema": NUDGE_RESPONSE_SCHEMA,
    "strict": True,
}
