"""Data loading helpers for the demo review app."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

from src.annotation_tool.data_loader import get_ordered_entries, load_entries_with_warnings
from src.wrangling.parse_wrangled_data import ParseWarning


@dataclass
class DemoLoadResult:
    """Loaded persona catalog plus any non-fatal parsing warnings."""

    personas: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[ParseWarning] = field(default_factory=list)

    @property
    def total_personas(self) -> int:
        return len(self.personas)

    @property
    def total_entries(self) -> int:
        return sum(len(persona["entries"]) for persona in self.personas)


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep demo entry rows compact and consistently typed."""
    return {
        "persona_id": entry["persona_id"],
        "persona_name": entry.get("persona_name"),
        "t_index": int(entry["t_index"]),
        "date": entry["date"],
        "initial_entry": entry.get("initial_entry") or "",
        "nudge_text": entry.get("nudge_text"),
        "response_text": entry.get("response_text"),
        "has_nudge": bool(entry.get("has_nudge")),
        "has_response": bool(entry.get("has_response")),
    }


def _build_persona_record(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one persona-centric record from ordered entry rows."""
    first = entries[0]
    normalized_entries = sorted(
        (_normalize_entry(entry) for entry in entries),
        key=lambda row: (row["date"], row["t_index"]),
    )
    core_values = first.get("persona_core_values") or []
    return {
        "persona_id": first["persona_id"],
        "persona_name": first.get("persona_name") or first["persona_id"],
        "persona_age": first.get("persona_age"),
        "persona_profession": first.get("persona_profession"),
        "persona_culture": first.get("persona_culture"),
        "persona_core_values": core_values,
        "persona_bio": first.get("persona_bio"),
        "annotation_order": first.get("annotation_order"),
        "entries": normalized_entries,
        "n_entries": len(normalized_entries),
        "first_entry_date": normalized_entries[0]["date"],
        "last_entry_date": normalized_entries[-1]["date"],
    }


def load_demo_personas(
    wrangled_dir: str | Path = "logs/wrangled",
    registry_path: str | Path = "logs/registry/personas.parquet",
) -> DemoLoadResult:
    """Load demo personas from wrangled files, grouped by persona."""
    result = load_entries_with_warnings(
        wrangled_dir=wrangled_dir,
        registry_path=registry_path,
    )
    ordered_entries = get_ordered_entries(result.df)

    personas: list[dict[str, Any]] = []
    for _persona_id, group in groupby(ordered_entries, key=lambda row: row["persona_id"]):
        grouped_entries = list(group)
        if grouped_entries:
            personas.append(_build_persona_record(grouped_entries))

    personas.sort(
        key=lambda persona: (
            persona["annotation_order"] if persona["annotation_order"] is not None else 999999,
            persona["persona_id"],
        )
    )
    return DemoLoadResult(personas=personas, warnings=result.warnings)


def build_persona_choices(personas: list[dict[str, Any]]) -> dict[str, str]:
    """Build Shiny select choices for persona selection."""
    choices: dict[str, str] = {}
    for persona in personas:
        label = (
            f"{persona['persona_name']} · {persona['n_entries']} entries · "
            f"{persona['first_entry_date']} to {persona['last_entry_date']}"
        )
        choices[persona["persona_id"]] = label
    return choices


def get_persona(personas: list[dict[str, Any]], persona_id: str | None) -> dict[str, Any] | None:
    """Look up one persona record by ID."""
    if not persona_id:
        return None
    for persona in personas:
        if persona["persona_id"] == persona_id:
            return persona
    return None


def summarize_warnings(warnings: list[ParseWarning], limit: int = 3) -> str | None:
    """Condense parsing warnings into a short demo-friendly banner."""
    if not warnings:
        return None

    snippets = []
    for warning in warnings[:limit]:
        location = (
            f"{warning.file} entry {warning.entry_index}"
            if warning.entry_index is not None
            else warning.file
        )
        snippets.append(f"{location}: {warning.message}")

    summary = f"Loaded with {len(warnings)} parsing warning(s). " + " | ".join(snippets)
    if len(warnings) > limit:
        summary += " | ..."
    return summary
