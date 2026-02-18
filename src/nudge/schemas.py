"""JSON response schemas for OpenAI structured output in the nudge pipeline.

These schemas are passed as `response_format` to the LLM completion function
to enforce structured JSON responses.
"""

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
