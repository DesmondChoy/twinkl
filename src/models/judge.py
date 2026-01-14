"""Pydantic models for Judge labeling pipeline output validation.

These models validate the JSON output from Claude Code judge subagents,
ensuring scores are within valid ranges and required fields are present.

Usage (validation):
    from src.models.judge import PersonaLabels
    import json

    with open("persona_1_labels.json") as f:
        data = json.load(f)
    validated = PersonaLabels.model_validate(data)

Usage (construct programmatically):
    from src.models.judge import AlignmentScores, EntryLabel, PersonaLabels

    scores = AlignmentScores(
        self_direction=1, stimulation=0, hedonism=-1,
        achievement=0, power=0, security=1,
        conformity=0, tradition=0, benevolence=1, universalism=0
    )
    entry = EntryLabel(t_index=0, date="2024-01-15", scores=scores)
    persona = PersonaLabels(persona_id=1, labels=[entry])
"""

from pydantic import BaseModel, Field


# Canonical order of Schwartz values (must match across all components)
SCHWARTZ_VALUE_ORDER = [
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


class AlignmentScores(BaseModel):
    """Per-dimension alignment scores from the Judge.

    Each value dimension gets a score in {-1, 0, +1}:
    - -1 (Misaligned): Entry actively conflicts with this value
    - 0 (Neutral): Entry is irrelevant to the value
    - +1 (Aligned): Entry actively supports this value
    """

    self_direction: int = Field(ge=-1, le=1)
    stimulation: int = Field(ge=-1, le=1)
    hedonism: int = Field(ge=-1, le=1)
    achievement: int = Field(ge=-1, le=1)
    power: int = Field(ge=-1, le=1)
    security: int = Field(ge=-1, le=1)
    conformity: int = Field(ge=-1, le=1)
    tradition: int = Field(ge=-1, le=1)
    benevolence: int = Field(ge=-1, le=1)
    universalism: int = Field(ge=-1, le=1)

    def to_vector(self) -> list[int]:
        """Convert to ordered vector matching SCHWARTZ_VALUE_ORDER."""
        return [
            self.self_direction,
            self.stimulation,
            self.hedonism,
            self.achievement,
            self.power,
            self.security,
            self.conformity,
            self.tradition,
            self.benevolence,
            self.universalism,
        ]


class EntryLabel(BaseModel):
    """Single journal entry's alignment label."""

    t_index: int = Field(ge=0, description="0-based entry index within persona")
    date: str = Field(description="Entry date in YYYY-MM-DD format")
    scores: AlignmentScores


class PersonaLabels(BaseModel):
    """All labels for one persona (output of a judge subagent).

    This is the top-level model that validates the entire JSON output
    from a judge subagent for a single persona.
    """

    persona_id: int = Field(ge=0, description="Unique persona identifier")
    labels: list[EntryLabel] = Field(description="List of entry labels in order")
