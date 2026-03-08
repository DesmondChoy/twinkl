"""Verification helpers for manually generated targeted synthetic batches."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import polars as pl
import yaml

from src.wrangling.parse_synthetic_data import (
    extract_persona_id,
    parse_entries,
    parse_persona_profile,
)

REFLECTION_MODE_PATTERN = re.compile(
    r"(?:\*\*(?:Reflection Mode|Mode)(?:\*\*:|:\*\*)|(?:Reflection Mode|Mode):)\s*([^\n|]+)",
    re.IGNORECASE,
)
ENTRY_SPLIT_PATTERN = re.compile(r"## Entry (\d+) - (\d{4}-\d{2}-\d{2})")


def load_yaml_file(path: str | Path) -> dict:
    """Load a YAML file and validate that it is a mapping."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def load_baseline_persona_ids(path: str | Path) -> set[str]:
    """Load the pre-generation persona snapshot."""
    payload = load_yaml_file(path)
    persona_ids = payload.get("registry_persona_ids")
    if not isinstance(persona_ids, list):
        raise ValueError("Baseline snapshot missing registry_persona_ids")

    normalized: set[str] = set()
    for persona_id in persona_ids:
        if not isinstance(persona_id, str) or not persona_id.strip():
            raise ValueError(f"Invalid persona ID in baseline snapshot: {persona_id!r}")
        normalized.add(persona_id.strip())
    return normalized


def summarize_raw_persona_file(filepath: str | Path) -> dict:
    """Summarize one raw synthetic persona markdown file."""
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")
    persona_id = extract_persona_id(path.name)
    profile = parse_persona_profile(content)
    entries = parse_entries(content)
    reflection_modes = _extract_reflection_modes(content)

    first_unsettled_entry = None
    for entry, reflection_mode in zip(entries, reflection_modes, strict=False):
        if reflection_mode.lower() != "unsettled":
            continue
        first_unsettled_entry = {
            "t_index": int(entry["t_index"]),
            "date": entry["date"],
            "initial_entry_excerpt": _truncate(entry["initial_entry"] or "", limit=320),
        }
        break

    return {
        "persona_id": persona_id,
        "core_values": profile.get("core_values", []),
        "entry_count": len(entries),
        "reflection_mode_counts": dict(Counter(reflection_modes)),
        "unsettled_entry_count": sum(mode.lower() == "unsettled" for mode in reflection_modes),
        "first_unsettled_entry": first_unsettled_entry,
        "raw_path": str(path),
    }


def verify_targeted_batch(
    *,
    baseline_persona_ids: set[str],
    registry_path: str | Path,
    synthetic_dir: str | Path,
    required_targets: list[str],
    expected_new_persona_count: int,
    expected_min_personas_per_target: int,
    min_entries: int,
    max_entries: int,
    require_unsettled_entries: bool = True,
) -> dict:
    """Verify the manually generated targeted batch against expected constraints."""
    registry = pl.read_parquet(registry_path)
    current_persona_ids = set(registry.get_column("persona_id").to_list())
    new_persona_ids = sorted(current_persona_ids - baseline_persona_ids)

    failures: list[str] = []
    if len(new_persona_ids) != expected_new_persona_count:
        failures.append(
            "Expected "
            f"{expected_new_persona_count} new personas but found {len(new_persona_ids)}"
        )

    records: list[dict] = []
    synthetic_path = Path(synthetic_dir)
    for persona_id in new_persona_ids:
        raw_file = synthetic_path / f"persona_{persona_id}.md"
        if not raw_file.exists():
            failures.append(f"Missing raw synthetic file for persona {persona_id}")
            continue

        record = summarize_raw_persona_file(raw_file)
        registry_row = registry.filter(pl.col("persona_id") == persona_id)
        record["registered"] = bool(registry_row.height)
        record["stage_synthetic"] = bool(registry_row["stage_synthetic"][0]) if registry_row.height else False
        records.append(record)

    required_target_set = set(required_targets)
    target_counts = {target: 0 for target in required_targets}

    for record in records:
        core_values = {
            value.strip()
            for value in record["core_values"]
            if isinstance(value, str) and value.strip()
        }

        if not core_values & required_target_set:
            failures.append(
                f"Persona {record['persona_id']} is missing all required target values"
            )

        for target in required_targets:
            if target in core_values:
                target_counts[target] += 1

        if not min_entries <= record["entry_count"] <= max_entries:
            failures.append(
                f"Persona {record['persona_id']} has {record['entry_count']} entries "
                f"(expected {min_entries}-{max_entries})"
            )

        if require_unsettled_entries and record["unsettled_entry_count"] < 1:
            failures.append(
                f"Persona {record['persona_id']} has no Unsettled entries in the raw batch"
            )

        if not record["registered"]:
            failures.append(f"Persona {record['persona_id']} is missing from the registry")
        elif not record["stage_synthetic"]:
            failures.append(
                f"Persona {record['persona_id']} is not marked stage_synthetic=true"
            )

    for target, count in target_counts.items():
        if count < expected_min_personas_per_target:
            failures.append(
                f"Target value {target} appears in {count} new personas "
                f"(expected at least {expected_min_personas_per_target})"
            )

    return {
        "accepted": not failures,
        "new_persona_ids": new_persona_ids,
        "required_targets": required_targets,
        "target_counts": target_counts,
        "failures": failures,
        "records": records,
    }


def render_spot_check_report(summary: dict) -> str:
    """Render a compact markdown report for manual review."""
    lines = [
        "# twinkl-681.5 Generation Spot Check",
        "",
        f"- Accepted: {'yes' if summary['accepted'] else 'no'}",
        f"- New personas: {len(summary['new_persona_ids'])}",
        f"- Target counts: {summary['target_counts']}",
        "",
    ]

    if summary["failures"]:
        lines.extend(["## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- {failure}")
        lines.append("")

    lines.extend(["## First Unsettled Entry Per Persona", ""])
    for record in summary["records"]:
        lines.append(
            f"### {record['persona_id']} ({', '.join(record['core_values']) or 'no core values'})"
        )
        lines.append(f"- Entries: {record['entry_count']}")
        lines.append(f"- Reflection modes: {record['reflection_mode_counts']}")

        first_unsettled = record["first_unsettled_entry"]
        if first_unsettled is None:
            lines.append("- First Unsettled entry: none found")
        else:
            lines.append(
                f"- First Unsettled entry: t_index={first_unsettled['t_index']}, "
                f"date={first_unsettled['date']}"
            )
            lines.append("")
            lines.append(first_unsettled["initial_entry_excerpt"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _extract_reflection_modes(content: str) -> list[str]:
    """Extract one reflection-mode label per raw entry block."""
    modes: list[str] = []
    parts = ENTRY_SPLIT_PATTERN.split(content)
    for index in range(1, len(parts), 3):
        if index + 2 >= len(parts):
            break
        entry_content = parts[index + 2]
        match = REFLECTION_MODE_PATTERN.search(entry_content)
        if match:
            modes.append(match.group(1).strip())
        else:
            modes.append("unknown")
    return modes


def _truncate(text: str, *, limit: int) -> str:
    """Trim long excerpts while keeping them readable in markdown exports."""
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."
