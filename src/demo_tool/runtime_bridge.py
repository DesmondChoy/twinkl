"""Checkpoint discovery and demo-friendly runtime wrappers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.coach.runtime import run_weekly_coach_cycle

DEFAULT_CHECKPOINT_ROOTS = (
    Path("logs/experiments/artifacts"),
    Path("models/vif"),
    Path("logs/experiments"),
)
EXACT_CHECKPOINT_FILENAMES = ("selected_checkpoint.pt", "best_model.pt")


@dataclass(frozen=True)
class CheckpointOption:
    """One selectable runtime checkpoint for the demo UI."""

    path: str
    label: str
    source: str
    run_id: str | None = None
    model_name: str | None = None
    metrics_summary: dict[str, float] | None = None


def _relative_label(path: Path) -> str:
    """Render a readable repo-relative path when possible."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _extract_run_id(path: Path) -> str | None:
    """Extract run_id from a checkpoint path if present."""
    for part in reversed(path.parts):
        match = re.search(r"(run_\d+)", part)
        if match:
            return match.group(1)
    return None


def _extract_model_name(path: Path) -> str | None:
    """Guess the model family from nearby directories."""
    for part in reversed(path.parts[:-1]):
        if part in {"artifacts", "models", "logs", "experiments"}:
            continue
        if part.startswith("run_"):
            continue
        if any(char.isupper() for char in part):
            return part
    return None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    """Read a YAML sidecar file if it exists and is well-formed."""
    if not path.exists():
        return None
    with path.open("r") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, dict):
        return payload
    return None


def _load_metrics_summary(path: Path) -> dict[str, float] | None:
    """Extract a compact metric summary from nearby experiment sidecars."""
    candidates = (
        path.with_suffix(".yaml"),
        path.parent / "metrics_summary.yaml",
        path.parent / "selected_policy.yaml",
        path.parent / "selection_summary.yaml",
    )
    for candidate in candidates:
        payload = _load_yaml(candidate)
        if not payload:
            continue

        selected_metrics = payload.get("selected_validation_metrics") or payload.get(
            "selected_test_metrics"
        )
        if isinstance(selected_metrics, dict):
            summary: dict[str, float] = {}
            for key in ("qwk_mean", "recall_minus1", "calibration_global"):
                value = selected_metrics.get(key)
                if isinstance(value, (int, float)):
                    summary[key] = float(value)
            if summary:
                return summary

        candidate_block = payload.get("selected_candidate")
        if isinstance(candidate_block, dict):
            summary = {}
            for key in ("qwk_mean", "recall_minus1", "calibration_global"):
                value = candidate_block.get(key)
                if isinstance(value, (int, float)):
                    summary[key] = float(value)
            if summary:
                return summary

    return None


def _format_metrics(metrics_summary: dict[str, float] | None) -> str:
    """Render a short metric tail for a checkpoint label."""
    if not metrics_summary:
        return ""

    pieces = []
    if "qwk_mean" in metrics_summary:
        pieces.append(f"QWK {metrics_summary['qwk_mean']:.3f}")
    if "recall_minus1" in metrics_summary:
        pieces.append(f"R-1 {metrics_summary['recall_minus1']:.3f}")
    return " · " + " · ".join(pieces) if pieces else ""


def _build_checkpoint_option(path: Path) -> CheckpointOption:
    """Create a UI-friendly checkpoint option from a path."""
    run_id = _extract_run_id(path)
    model_name = _extract_model_name(path)
    metrics_summary = _load_metrics_summary(path)
    source = path.parts[0] if path.parts else "local"

    label_parts = []
    if run_id:
        label_parts.append(run_id)
    if model_name and model_name not in label_parts:
        label_parts.append(model_name)
    if not label_parts:
        label_parts.append(path.stem)
    label = " · ".join(label_parts)
    label += _format_metrics(metrics_summary)
    label += f" · {_relative_label(path)}"

    return CheckpointOption(
        path=str(path.resolve()),
        label=label,
        source=source,
        run_id=run_id,
        model_name=model_name,
        metrics_summary=metrics_summary,
    )


def discover_checkpoints(
    search_roots: tuple[Path, ...] = DEFAULT_CHECKPOINT_ROOTS,
) -> list[CheckpointOption]:
    """Discover locally available checkpoints for the demo selector."""
    discovered: dict[str, Path] = {}

    for root in search_roots:
        if not root.exists():
            continue

        for filename in EXACT_CHECKPOINT_FILENAMES:
            for path in root.rglob(filename):
                discovered[str(path.resolve())] = path

        for path in root.rglob("*.pt"):
            normalized = str(path.resolve())
            if normalized in discovered:
                continue
            lower_name = path.name.lower()
            if "checkpoint" in lower_name or "model" in lower_name:
                discovered[normalized] = path

    options = [_build_checkpoint_option(path) for path in discovered.values()]
    options.sort(key=lambda option: option.label.lower())
    return options


def build_checkpoint_choices(options: list[CheckpointOption]) -> dict[str, str]:
    """Build select input choices for the checkpoint dropdown."""
    return {option.path: option.label for option in options}


def get_checkpoint_option(
    options: list[CheckpointOption], checkpoint_path: str | None
) -> CheckpointOption | None:
    """Look up a discovered checkpoint option by its absolute path."""
    if not checkpoint_path:
        return None
    for option in options:
        if option.path == checkpoint_path:
            return option
    return None


def _slugify(value: str) -> str:
    """Convert a string into a filesystem-safe slug."""
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    return cleaned.strip("-") or "artifact"


def build_output_dir(
    persona_id: str,
    checkpoint_path: str | Path,
    output_root: str | Path = "logs/exports/demo_tool_runs",
) -> Path:
    """Create a stable output directory for a persona/checkpoint pair."""
    checkpoint = Path(checkpoint_path)
    digest = hashlib.sha1(str(checkpoint.resolve()).encode("utf-8")).hexdigest()[:10]
    checkpoint_slug = _slugify(checkpoint.stem)
    return Path(output_root) / persona_id / f"{checkpoint_slug}-{digest}"


def discover_cached_artifacts(
    persona_id: str,
    checkpoint_path: str | Path,
    output_root: str | Path = "logs/exports/demo_tool_runs",
    parquet_path: str | Path = "logs/exports/weekly_digests/weekly_digests.parquet",
) -> dict[str, str] | None:
    """Locate the latest persisted artifact bundle for a persona/checkpoint pair."""
    output_dir = build_output_dir(
        persona_id=persona_id,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
    )
    if not output_dir.exists():
        return None

    timeline_path = output_dir / f"{persona_id}_vif_timeline.parquet"
    weekly_path = output_dir / f"{persona_id}_vif_weekly.parquet"
    drift_files = sorted(output_dir.glob("*.drift.json"), key=lambda path: path.stat().st_mtime)
    digest_json_files = sorted(
        (
            path
            for path in output_dir.glob("*.json")
            if not path.name.endswith(".drift.json")
        ),
        key=lambda path: path.stat().st_mtime,
    )
    digest_md_files = sorted(output_dir.glob("*.md"), key=lambda path: path.stat().st_mtime)
    prompt_files = sorted(output_dir.glob("*.prompt.txt"), key=lambda path: path.stat().st_mtime)

    required = [timeline_path, weekly_path]
    if not all(path.exists() for path in required):
        return None
    if not drift_files or not digest_json_files or not digest_md_files:
        return None

    bundle = {
        "timeline_path": str(timeline_path),
        "weekly_path": str(weekly_path),
        "drift_json_path": str(drift_files[-1]),
        "digest_json_path": str(digest_json_files[-1]),
        "digest_md_path": str(digest_md_files[-1]),
        "parquet_path": str(parquet_path),
    }
    if prompt_files:
        bundle["prompt_path"] = str(prompt_files[-1])
    return bundle


def load_artifact_bundle(artifact_paths: dict[str, str]) -> dict[str, Any]:
    """Read persisted runtime artifacts into memory for UI rendering."""
    timeline_df = pl.read_parquet(artifact_paths["timeline_path"])
    weekly_df = pl.read_parquet(artifact_paths["weekly_path"])
    drift_payload = json.loads(Path(artifact_paths["drift_json_path"]).read_text())
    digest_payload = json.loads(Path(artifact_paths["digest_json_path"]).read_text())
    digest_markdown = Path(artifact_paths["digest_md_path"]).read_text()
    prompt_text = None
    prompt_path = artifact_paths.get("prompt_path")
    if prompt_path and Path(prompt_path).exists():
        prompt_text = Path(prompt_path).read_text()

    return {
        "timeline_df": timeline_df,
        "weekly_df": weekly_df,
        "drift_payload": drift_payload,
        "digest_payload": digest_payload,
        "digest_markdown": digest_markdown,
        "prompt_text": prompt_text,
    }


def load_cached_run(
    persona_id: str,
    checkpoint_path: str | Path,
    *,
    output_root: str | Path = "logs/exports/demo_tool_runs",
    parquet_path: str | Path = "logs/exports/weekly_digests/weekly_digests.parquet",
) -> dict[str, Any] | None:
    """Load the most recent cached run for a persona/checkpoint pair, if present."""
    artifact_paths = discover_cached_artifacts(
        persona_id=persona_id,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        parquet_path=parquet_path,
    )
    if artifact_paths is None:
        return None

    return {
        "persona_id": persona_id,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "output_dir": str(
            build_output_dir(
                persona_id=persona_id,
                checkpoint_path=checkpoint_path,
                output_root=output_root,
            )
        ),
        "artifact_paths": artifact_paths,
        "artifacts": load_artifact_bundle(artifact_paths),
    }


def load_multi_drift_bundle(
    persona_id: str,
    source: str = "judge",
    timeline_df: Any | None = None,
    judge_labels_path: str | Path = "logs/judge_labels/judge_labels.parquet",
    registry_path: str | Path = "logs/registry/personas.parquet",
) -> Any | None:
    """Run all 6 drift detectors for a persona and return a MultiDriftBundle.

    source: "judge" uses judge_labels.parquet; "critic" uses the provided timeline_df.
    Returns None if insufficient data or source files are missing.
    """
    from src.demo_tool.multi_drift import (
        run_multi_drift_from_critic,
        run_multi_drift_from_judge,
    )

    if source == "critic":
        if timeline_df is None:
            return None
        return run_multi_drift_from_critic(
            persona_id=persona_id,
            timeline_df=timeline_df,
            registry_path=Path(registry_path),
        )

    return run_multi_drift_from_judge(
        persona_id=persona_id,
        judge_labels_path=Path(judge_labels_path),
        registry_path=Path(registry_path),
    )


def run_demo_pipeline(
    *,
    persona_id: str,
    checkpoint_path: str | Path,
    wrangled_dir: str | Path = "logs/wrangled",
    output_root: str | Path = "logs/exports/demo_tool_runs",
    parquet_path: str | Path = "logs/exports/weekly_digests/weekly_digests.parquet",
) -> dict[str, Any]:
    """Run the full weekly demo pipeline and return UI-ready outputs."""
    resolved_checkpoint = Path(checkpoint_path).resolve()
    output_dir = build_output_dir(
        persona_id=persona_id,
        checkpoint_path=resolved_checkpoint,
        output_root=output_root,
    )
    digest, artifact_paths = run_weekly_coach_cycle(
        persona_id=persona_id,
        checkpoint_path=resolved_checkpoint,
        wrangled_dir=wrangled_dir,
        output_dir=output_dir,
        parquet_path=parquet_path,
    )
    return {
        "persona_id": persona_id,
        "checkpoint_path": str(resolved_checkpoint),
        "output_dir": str(output_dir),
        "digest": digest.model_dump(),
        "artifact_paths": artifact_paths,
        "artifacts": load_artifact_bundle(artifact_paths),
    }
