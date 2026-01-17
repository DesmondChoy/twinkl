"""Persona registry for tracking pipeline stages.

Provides CRUD operations for a central parquet registry that tracks all
personas and their progress through the data pipeline (synthetic → wrangled → labeled).

Uses file locking for safe concurrent writes from parallel subagents.

Registry Schema:
    persona_id: str         - 8-character UUID (e.g., "a3f8b2c1")
    name: str               - Persona's full name
    age: str                - Age bracket (e.g., "25-34")
    profession: str         - Occupation
    culture: str            - Cultural background
    core_values: list[str]  - Schwartz values (e.g., ["Power", "Achievement"])
    entry_count: int        - Number of journal entries
    created_at: datetime    - When generated
    stage_synthetic: bool   - True after generation
    stage_wrangled: bool    - True after wrangling
    stage_labeled: bool     - True after judge labeling
    nudge_enabled: bool     - Whether nudges were enabled during generation
"""

import fcntl
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Central registry location
REGISTRY_PATH = Path("logs/registry/personas.parquet")

# Registry schema for new registries
REGISTRY_SCHEMA = {
    "persona_id": pl.Utf8,
    "name": pl.Utf8,
    "age": pl.Utf8,
    "profession": pl.Utf8,
    "culture": pl.Utf8,
    "core_values": pl.List(pl.Utf8),
    "entry_count": pl.Int64,
    "created_at": pl.Datetime("us", "UTC"),
    "stage_synthetic": pl.Boolean,
    "stage_wrangled": pl.Boolean,
    "stage_labeled": pl.Boolean,
    "nudge_enabled": pl.Boolean,
}


def _ensure_registry_exists() -> None:
    """Create registry directory and empty parquet file if needed."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_PATH.exists():
        # Create empty DataFrame with schema
        empty_df = pl.DataFrame(
            schema=REGISTRY_SCHEMA,
        )
        empty_df.write_parquet(REGISTRY_PATH)


def _read_registry() -> pl.DataFrame:
    """Read the registry, creating it if needed."""
    _ensure_registry_exists()
    return pl.read_parquet(REGISTRY_PATH)


def _write_registry_locked(df: pl.DataFrame) -> None:
    """Write registry with file locking for concurrent safety.

    Uses fcntl.flock for advisory locking. Multiple subagents can safely
    append to the registry without corruption.
    """
    _ensure_registry_exists()

    # Use a separate lock file to avoid issues with parquet binary format
    lock_path = REGISTRY_PATH.with_suffix(".lock")
    lock_path.touch(exist_ok=True)

    with open(lock_path) as lock_file:
        # Acquire exclusive lock (blocks until available)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            df.write_parquet(REGISTRY_PATH)
        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def register_persona(
    persona_id: str,
    name: str,
    age: str,
    profession: str,
    culture: str,
    core_values: list[str],
    entry_count: int,
    nudge_enabled: bool = True,
) -> None:
    """Register a newly generated persona in the registry.

    Should be called by the generation subagent after writing persona_*.md.
    Uses file locking for safe concurrent registration from parallel subagents.

    Args:
        persona_id: 8-character UUID (e.g., "a3f8b2c1")
        name: Persona's full name
        age: Age bracket (e.g., "25-34")
        profession: Occupation
        culture: Cultural background
        core_values: List of Schwartz values
        entry_count: Number of journal entries generated
        nudge_enabled: Whether nudges were enabled during generation

    Raises:
        ValueError: If persona_id already exists in registry
    """
    _ensure_registry_exists()

    # Use lock for read-modify-write
    lock_path = REGISTRY_PATH.with_suffix(".lock")
    lock_path.touch(exist_ok=True)

    with open(lock_path) as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            df = pl.read_parquet(REGISTRY_PATH)

            # Check for duplicate ID
            if len(df.filter(pl.col("persona_id") == persona_id)) > 0:
                raise ValueError(f"Persona {persona_id} already exists in registry")

            # Create new row
            new_row = pl.DataFrame(
                {
                    "persona_id": [persona_id],
                    "name": [name],
                    "age": [age],
                    "profession": [profession],
                    "culture": [culture],
                    "core_values": [core_values],
                    "entry_count": [entry_count],
                    "created_at": [datetime.now(timezone.utc)],
                    "stage_synthetic": [True],
                    "stage_wrangled": [False],
                    "stage_labeled": [False],
                    "nudge_enabled": [nudge_enabled],
                },
                schema=REGISTRY_SCHEMA,
            )

            # Append and write
            df = pl.concat([df, new_row])
            df.write_parquet(REGISTRY_PATH)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def update_stage(persona_id: str, stage: str) -> None:
    """Mark a persona as having completed a pipeline stage.

    Args:
        persona_id: 8-character UUID
        stage: One of "wrangled" or "labeled"

    Raises:
        ValueError: If persona_id not found or invalid stage
    """
    valid_stages = {"wrangled", "labeled"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

    column_name = f"stage_{stage}"

    lock_path = REGISTRY_PATH.with_suffix(".lock")
    lock_path.touch(exist_ok=True)

    with open(lock_path) as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            df = pl.read_parquet(REGISTRY_PATH)

            # Check persona exists
            if len(df.filter(pl.col("persona_id") == persona_id)) == 0:
                raise ValueError(f"Persona {persona_id} not found in registry")

            # Update the stage column
            df = df.with_columns(
                pl.when(pl.col("persona_id") == persona_id)
                .then(True)
                .otherwise(pl.col(column_name))
                .alias(column_name)
            )

            df.write_parquet(REGISTRY_PATH)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_pending(stage: str) -> pl.DataFrame:
    """Get personas that haven't completed the specified stage.

    Args:
        stage: One of "synthetic", "wrangled", or "labeled"

    Returns:
        DataFrame of personas where stage_{stage} is False
        (or where the previous stage is True but this stage is False)

    Examples:
        # Get personas ready to wrangle (synthetic=True, wrangled=False)
        get_pending("wrangled")

        # Get personas ready to label (wrangled=True, labeled=False)
        get_pending("labeled")
    """
    valid_stages = {"synthetic", "wrangled", "labeled"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

    df = _read_registry()

    if stage == "synthetic":
        # Shouldn't really happen, but return any not generated
        return df.filter(pl.col("stage_synthetic") == False)
    elif stage == "wrangled":
        # Ready to wrangle: synthetic=True, wrangled=False
        return df.filter(
            (pl.col("stage_synthetic") == True) & (pl.col("stage_wrangled") == False)
        )
    else:  # labeled
        # Ready to label: wrangled=True, labeled=False
        return df.filter(
            (pl.col("stage_wrangled") == True) & (pl.col("stage_labeled") == False)
        )


def get_status() -> dict:
    """Get pipeline status summary.

    Returns:
        Dict with counts by stage:
        {
            "total": int,
            "synthetic": int,
            "wrangled": int,
            "labeled": int,
            "pending_wrangling": int,
            "pending_labeling": int,
        }
    """
    df = _read_registry()

    return {
        "total": len(df),
        "synthetic": df["stage_synthetic"].sum(),
        "wrangled": df["stage_wrangled"].sum(),
        "labeled": df["stage_labeled"].sum(),
        "pending_wrangling": len(
            df.filter(
                (pl.col("stage_synthetic") == True)
                & (pl.col("stage_wrangled") == False)
            )
        ),
        "pending_labeling": len(
            df.filter(
                (pl.col("stage_wrangled") == True) & (pl.col("stage_labeled") == False)
            )
        ),
    }


def get_registry() -> pl.DataFrame:
    """Get the full registry DataFrame.

    Returns:
        Complete registry as a Polars DataFrame
    """
    return _read_registry()
