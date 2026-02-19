"""Judge labeling helpers extracted from notebook prototypes.

This module centralizes rubric construction and per-session judging helpers
so docs and scripts can reference stable Python paths instead of notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable, Sequence

import yaml

from prompts import judge_alignment_prompt
from src.models.judge import AlignmentScores, SCHWARTZ_VALUE_ORDER

LLMCompleteFn = Callable[[str, dict | None], Awaitable[str | None]]

# Map model keys to display names used in schwartz_values.yaml.
SCHWARTZ_VALUE_DISPLAY = {
    "self_direction": "Self-Direction",
    "stimulation": "Stimulation",
    "hedonism": "Hedonism",
    "achievement": "Achievement",
    "power": "Power",
    "security": "Security",
    "conformity": "Conformity",
    "tradition": "Tradition",
    "benevolence": "Benevolence",
    "universalism": "Universalism",
}

JUDGE_LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {key: {"type": "integer", "minimum": -1, "maximum": 1} for key in SCHWARTZ_VALUE_ORDER},
            "required": SCHWARTZ_VALUE_ORDER,
        },
        "rationales": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["scores", "rationales"],
}

JUDGE_LABEL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "JudgeLabel",
    "schema": JUDGE_LABEL_SCHEMA,
    "strict": True,
}


def load_schwartz_values(path: str | Path = "config/schwartz_values.yaml") -> dict:
    """Load Schwartz value elaborations YAML."""
    config_path = Path(path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_value_rubric_context(schwartz_config: dict) -> str:
    """Build a concise rubric section for each Schwartz dimension."""
    context_parts: list[str] = []
    value_map = schwartz_config.get("values", {})

    for key in SCHWARTZ_VALUE_ORDER:
        display_name = SCHWARTZ_VALUE_DISPLAY[key]
        value_data = value_map.get(display_name)
        if not value_data:
            continue

        context_parts.append(
            f"""
### {display_name}
**Core Motivation:** {value_data["core_motivation"].strip()}

**Key Behaviors (Aligned):**
{chr(10).join(f"- {behavior}" for behavior in value_data["behavioral_manifestations"][:3])}

**Key Behaviors (Misaligned):**
- Acting against the core motivation
- Neglecting or undermining this value
- Making choices that conflict with this value's principles
""".strip()
        )

    return "\n\n".join(context_parts)


def _safe_load_json_object(raw_json: str) -> dict | None:
    try:
        data = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _normalize_rationales(raw_rationales: object) -> dict[str, str] | None:
    """Keep only valid value keys with non-empty string rationales."""
    if not isinstance(raw_rationales, dict):
        return None

    cleaned: dict[str, str] = {}
    valid_keys = set(SCHWARTZ_VALUE_ORDER)
    for key, value in raw_rationales.items():
        if key in valid_keys and isinstance(value, str) and value.strip():
            cleaned[key] = value.strip()

    return cleaned or None


async def judge_session(
    session_content: str,
    entry_date: str,
    persona_name: str,
    persona_age: str,
    persona_profession: str,
    persona_culture: str,
    persona_core_values: Sequence[str],
    persona_bio: str,
    schwartz_config: dict,
    llm_complete: LLMCompleteFn,
    previous_entries: list[dict] | None = None,
) -> tuple[AlignmentScores | None, dict[str, str] | None, str]:
    """Judge one journal session and return validated scores + rationales.

    Returns:
        Tuple of (scores_or_none, rationales_or_none, prompt_used)
    """
    value_rubric = build_value_rubric_context(schwartz_config)

    prompt = judge_alignment_prompt.render(
        persona_name=persona_name,
        persona_age=persona_age,
        persona_profession=persona_profession,
        persona_culture=persona_culture,
        persona_core_values=list(persona_core_values),
        persona_bio=persona_bio,
        entry_date=entry_date,
        session_content=session_content,
        value_rubric=value_rubric,
        previous_entries=previous_entries,
    )

    raw_json = await llm_complete(prompt, JUDGE_LABEL_RESPONSE_FORMAT)
    if not raw_json:
        return None, None, prompt

    payload = _safe_load_json_object(raw_json)
    if payload is None:
        return None, None, prompt

    scores_raw = payload.get("scores")
    if not isinstance(scores_raw, dict):
        return None, None, prompt

    try:
        scores = AlignmentScores.model_validate(scores_raw)
    except Exception:
        return None, None, prompt

    rationales = _normalize_rationales(payload.get("rationales"))
    return scores, rationales, prompt
