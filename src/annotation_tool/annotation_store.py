"""Persist annotations with file locking (one parquet per annotator).

This module handles saving and loading human annotations for journal entries.
Each annotator gets their own parquet file to avoid conflicts.

Uses file locking for safe concurrent writes (pattern from src/registry/personas.py).

Schema for logs/annotations/<annotator_id>.parquet:
    persona_id: Utf8
    t_index: Int64
    annotator_id: Utf8
    timestamp: Datetime("us", "UTC")
    alignment_self_direction: Int8
    alignment_stimulation: Int8
    alignment_hedonism: Int8
    alignment_achievement: Int8
    alignment_power: Int8
    alignment_security: Int8
    alignment_conformity: Int8
    alignment_tradition: Int8
    alignment_benevolence: Int8
    alignment_universalism: Int8
    notes: Utf8 (nullable)
    confidence: Int8 (nullable, 1-5)

Usage:
    from src.annotation_tool.annotation_store import save_annotation, load_annotations

    save_annotation(
        annotator_id="alice",
        persona_id="a3f8b2c1",
        t_index=0,
        scores={"self_direction": 1, "stimulation": 0, ...},
        notes="Clear security-seeking behavior",
        confidence=4
    )

    df = load_annotations("alice")
"""

import fcntl
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.models.judge import SCHWARTZ_VALUE_ORDER

# Annotations directory
ANNOTATIONS_DIR = Path("logs/annotations")

# Schema for annotation parquet files
ANNOTATION_SCHEMA = {
    "persona_id": pl.Utf8,
    "t_index": pl.Int64,
    "annotator_id": pl.Utf8,
    "timestamp": pl.Datetime("us", "UTC"),
    **{f"alignment_{value}": pl.Int8 for value in SCHWARTZ_VALUE_ORDER},
    "notes": pl.Utf8,
    "confidence": pl.Int8,
}


def _get_annotator_path(annotator_id: str) -> Path:
    """Get the parquet file path for an annotator."""
    # Sanitize annotator_id for filesystem safety
    safe_id = "".join(c for c in annotator_id if c.isalnum() or c in "-_").lower()
    if not safe_id:
        safe_id = "anonymous"
    return ANNOTATIONS_DIR / f"{safe_id}.parquet"


def _ensure_annotations_dir() -> None:
    """Create annotations directory if needed."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _create_empty_annotation_df() -> pl.DataFrame:
    """Create an empty DataFrame with the annotation schema."""
    return pl.DataFrame(schema=ANNOTATION_SCHEMA)


def save_annotation(
    annotator_id: str,
    persona_id: str,
    t_index: int,
    scores: dict[str, int],
    notes: str | None = None,
    confidence: int | None = None,
) -> None:
    """Save or update an annotation with file locking.

    If an annotation for this (persona_id, t_index) already exists for this
    annotator, it will be updated (upsert behavior).

    Args:
        annotator_id: Free-form annotator name
        persona_id: Persona UUID (e.g., "a3f8b2c1")
        t_index: Entry index within persona (0-based)
        scores: Dict mapping value names to scores {-1, 0, +1}
                Keys should match SCHWARTZ_VALUE_ORDER
        notes: Optional annotation notes
        confidence: Optional confidence level (1-5)

    Raises:
        ValueError: If scores contain invalid values or missing keys
    """
    # Validate scores
    for value in SCHWARTZ_VALUE_ORDER:
        if value not in scores:
            raise ValueError(f"Missing score for value: {value}")
        if scores[value] not in (-1, 0, 1):
            raise ValueError(f"Invalid score {scores[value]} for {value}. Must be -1, 0, or 1")

    if confidence is not None and confidence not in (1, 2, 3, 4, 5):
        raise ValueError(f"Invalid confidence {confidence}. Must be 1-5 or None")

    _ensure_annotations_dir()

    parquet_path = _get_annotator_path(annotator_id)
    lock_path = parquet_path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)

    with open(lock_path) as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Load existing annotations or create empty DataFrame
            if parquet_path.exists():
                df = pl.read_parquet(parquet_path)
            else:
                df = _create_empty_annotation_df()

            # Create new row
            new_row_data = {
                "persona_id": [persona_id],
                "t_index": [t_index],
                "annotator_id": [annotator_id],
                "timestamp": [datetime.now(timezone.utc)],
                **{f"alignment_{value}": [scores[value]] for value in SCHWARTZ_VALUE_ORDER},
                "notes": [notes],
                "confidence": [confidence],
            }
            new_row = pl.DataFrame(new_row_data, schema=ANNOTATION_SCHEMA)

            # Upsert: remove existing row for this (persona_id, t_index) if present
            df = df.filter(
                ~((pl.col("persona_id") == persona_id) & (pl.col("t_index") == t_index))
            )

            # Append new row
            df = pl.concat([df, new_row])

            # Write back
            df.write_parquet(parquet_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_annotations(annotator_id: str) -> pl.DataFrame:
    """Load all annotations for an annotator.

    Args:
        annotator_id: Free-form annotator name

    Returns:
        DataFrame with all annotations for this annotator, or empty DataFrame
        if no annotations exist yet.
    """
    parquet_path = _get_annotator_path(annotator_id)

    if not parquet_path.exists():
        return _create_empty_annotation_df()

    return pl.read_parquet(parquet_path)


def get_annotated_keys(annotator_id: str) -> set[tuple[str, int]]:
    """Get the set of (persona_id, t_index) pairs already annotated.

    Args:
        annotator_id: Free-form annotator name

    Returns:
        Set of (persona_id, t_index) tuples that have been annotated
    """
    df = load_annotations(annotator_id)

    if len(df) == 0:
        return set()

    return {(row["persona_id"], row["t_index"]) for row in df.to_dicts()}


def get_annotation(
    annotator_id: str, persona_id: str, t_index: int
) -> dict | None:
    """Get a specific annotation if it exists.

    Args:
        annotator_id: Free-form annotator name
        persona_id: Persona UUID
        t_index: Entry index within persona

    Returns:
        Dict with annotation data, or None if not annotated
    """
    df = load_annotations(annotator_id)

    if len(df) == 0:
        return None

    filtered = df.filter(
        (pl.col("persona_id") == persona_id) & (pl.col("t_index") == t_index)
    )

    if len(filtered) == 0:
        return None

    return filtered.to_dicts()[0]


def get_annotation_count(annotator_id: str) -> int:
    """Get the number of annotations for an annotator.

    Args:
        annotator_id: Free-form annotator name

    Returns:
        Count of annotations
    """
    df = load_annotations(annotator_id)
    return len(df)
