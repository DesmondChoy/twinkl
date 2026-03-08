"""Helpers for freezing validation/test holdouts across augmentation rounds."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_holdout_manifest(path: str | Path) -> dict:
    """Load a YAML holdout manifest.

    Args:
        path: Path to the holdout manifest YAML file.

    Returns:
        Parsed manifest payload.

    Raises:
        FileNotFoundError: If the manifest path does not exist.
        ValueError: If the payload is empty or not a mapping.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Holdout manifest not found: {manifest_path}")

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Holdout manifest is empty or invalid: {manifest_path}")

    return payload


def load_fixed_holdout_ids(path: str | Path) -> tuple[set[str], set[str]]:
    """Load fixed validation and test persona IDs from a holdout manifest."""
    payload = load_holdout_manifest(path)

    val_ids = _normalize_persona_ids(payload.get("val_persona_ids"), field_name="val_persona_ids")
    test_ids = _normalize_persona_ids(
        payload.get("test_persona_ids"),
        field_name="test_persona_ids",
    )

    overlap = val_ids & test_ids
    if overlap:
        raise ValueError(
            "Holdout manifest has overlapping validation/test persona IDs: "
            f"{sorted(overlap)}"
        )

    return val_ids, test_ids


def _normalize_persona_ids(raw_ids: object, *, field_name: str) -> set[str]:
    if raw_ids is None:
        raise ValueError(f"Holdout manifest missing required field: {field_name}")
    if not isinstance(raw_ids, list):
        raise ValueError(f"{field_name} must be a list of persona IDs")

    normalized: set[str] = set()
    for persona_id in raw_ids:
        if not isinstance(persona_id, str) or not persona_id.strip():
            raise ValueError(f"{field_name} contains an invalid persona ID: {persona_id!r}")
        normalized.add(persona_id.strip())

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    return normalized
