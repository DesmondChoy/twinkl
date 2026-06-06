"""Replay recall-aware VIF checkpoint selection from saved per-epoch traces.

This utility is intentionally artifact-driven: it reads existing run YAML files,
selection traces, and dimension-weight traces, then writes a compact comparison
table. It does not retrain models and it does not pretend an alternate epoch is
test-evaluable unless the corresponding checkpoint artifact exists.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml


DEFAULT_RUN_FILES = (
    "logs/experiments/runs/run_019_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_020_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_021_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_034_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_035_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_036_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_048_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_049_BalancedSoftmax.yaml",
    "logs/experiments/runs/run_050_BalancedSoftmax.yaml",
)

HARD_DIMENSIONS = ("hedonism", "security", "stimulation")
BASELINE_POLICY = "current_qwk_then_recall"
RECALL_POLICIES = {
    "recall_qwk_window_0.01": 0.01,
    "recall_qwk_window_0.02": 0.02,
}
TRACE_REQUIRED_COLUMNS = {
    "epoch",
    "qwk_mean",
    "recall_minus1",
    "calibration_global",
    "hedging_mean",
    "val_loss",
    "eligible",
}
DELTA_METRICS = (
    "qwk_mean",
    "recall_minus1",
    "hedging_mean",
    "calibration_global",
    "val_loss",
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML document, returning an empty dict for empty files."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_repo_path(repo_root: Path, path_value: str | None) -> Path | None:
    """Resolve a run-artifact path relative to the repo root."""
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def label_regime_for_run(run_data: dict[str, Any]) -> str:
    """Classify the run label source for comparison guardrails."""
    run_id = str(run_data.get("metadata", {}).get("run_id", ""))
    if run_id.startswith(("run_048", "run_049", "run_050")):
        return "consensus"
    labels_path = str(run_data.get("config", {}).get("data", {}).get("labels_path", ""))
    if "consensus_labels" in labels_path:
        return "consensus"
    rationale = str(run_data.get("provenance", {}).get("rationale", "")).lower()
    if "consensus-label" in rationale or "consensus label" in rationale:
        return "consensus"
    return "persisted"


def model_family_for_run(run_id: str) -> str:
    """Group the specific run IDs used in this replay."""
    run_number = int(run_id.split("_")[1])
    if 19 <= run_number <= 21:
        return "incumbent_balanced_softmax"
    if 34 <= run_number <= 36:
        return "weighted_balanced_softmax"
    if 48 <= run_number <= 50:
        return "consensus_balanced_softmax"
    return "other"


def validate_trace_schema(trace: pl.DataFrame, *, source: Path) -> None:
    """Fail fast when a selection trace is too old or malformed."""
    missing = sorted(TRACE_REQUIRED_COLUMNS.difference(trace.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def eligible_trace_rows(trace: pl.DataFrame) -> pl.DataFrame:
    """Return finite-QWK eligible rows from a selection trace."""
    return trace.filter(pl.col("eligible") & pl.col("qwk_mean").is_finite())


def row_for_epoch(trace: pl.DataFrame, epoch: int) -> dict[str, Any] | None:
    """Return one trace row for an epoch, or None when absent."""
    rows = trace.filter(pl.col("epoch") == int(epoch))
    if rows.is_empty():
        return None
    return rows.row(0, named=True)


def select_current_policy(
    trace: pl.DataFrame,
    *,
    selected_epoch: int | None = None,
) -> dict[str, Any] | None:
    """Select the current qwk-first checkpoint from a trace."""
    eligible = eligible_trace_rows(trace)
    if eligible.is_empty():
        return None
    if selected_epoch is not None:
        selected = row_for_epoch(eligible, selected_epoch)
        if selected is not None:
            return selected
    return eligible.sort(
        ["qwk_mean", "recall_minus1", "calibration_global", "hedging_mean", "val_loss"],
        descending=[True, True, True, False, False],
    ).row(0, named=True)


def select_recall_window_policy(
    trace: pl.DataFrame,
    *,
    qwk_window: float,
) -> dict[str, Any] | None:
    """Maximize recall_-1 while staying within a QWK window of the best epoch."""
    eligible = eligible_trace_rows(trace)
    if eligible.is_empty():
        return None
    best_qwk = float(eligible.select(pl.max("qwk_mean")).item())
    if not math.isfinite(best_qwk):
        return None
    candidates = eligible.filter(pl.col("qwk_mean") >= best_qwk - float(qwk_window))
    if candidates.is_empty():
        return None
    return candidates.sort(
        ["recall_minus1", "hedging_mean", "calibration_global", "epoch"],
        descending=[True, False, True, False],
    ).row(0, named=True)


def selected_epoch_from_summary(
    repo_root: Path,
    artifacts: dict[str, Any],
) -> int | None:
    """Read the zero-based selected epoch from selection_summary.yaml."""
    summary_path = resolve_repo_path(repo_root, artifacts.get("selection_summary"))
    if summary_path is None or not summary_path.is_file():
        return None
    summary = load_yaml(summary_path)
    candidate = summary.get("selected_candidate") or {}
    epoch = candidate.get("epoch")
    return int(epoch) if epoch is not None else None


def hard_dimension_values(
    repo_root: Path,
    artifacts: dict[str, Any],
    *,
    epoch: int | None,
) -> dict[str, float | None]:
    """Read validation hard-dimension QWK values for a selected epoch."""
    values = {f"val_{dim}_qwk": None for dim in HARD_DIMENSIONS}
    if epoch is None:
        return values
    trace_path = resolve_repo_path(repo_root, artifacts.get("dimension_weight_trace"))
    if trace_path is None or not trace_path.is_file():
        return values
    trace = pl.read_parquet(trace_path)
    if not {"epoch", "dimension", "val_qwk"}.issubset(trace.columns):
        return values
    rows = trace.filter(pl.col("epoch") == int(epoch))
    for dim in HARD_DIMENSIONS:
        dim_rows = rows.filter(pl.col("dimension") == dim)
        if not dim_rows.is_empty():
            values[f"val_{dim}_qwk"] = float(dim_rows.select("val_qwk").item())
    return values


def baseline_test_metrics(
    run_data: dict[str, Any],
    *,
    include: bool,
) -> dict[str, float | None]:
    """Return fixed-test metrics for the original selected checkpoint."""
    fields = {
        "test_qwk_mean": ("evaluation", "qwk_mean"),
        "test_recall_minus1": ("evaluation", "recall_minus1"),
        "test_minority_recall_mean": ("evaluation", "minority_recall_mean"),
        "test_hedging_mean": ("evaluation", "hedging_mean"),
        "test_calibration_global": ("evaluation", "calibration_global"),
    }
    metrics: dict[str, float | None] = {name: None for name in fields}
    for dim in HARD_DIMENSIONS:
        metrics[f"test_{dim}_qwk"] = None
    if not include:
        return metrics

    for name, (section, key) in fields.items():
        value = run_data.get(section, {}).get(key)
        metrics[name] = float(value) if value is not None else None
    per_dimension = run_data.get("per_dimension", {})
    for dim in HARD_DIMENSIONS:
        value = per_dimension.get(dim, {}).get("qwk")
        metrics[f"test_{dim}_qwk"] = float(value) if value is not None else None
    return metrics


def artifact_status(
    repo_root: Path,
    artifacts: dict[str, Any],
    *,
    candidate_epoch: int | None,
    current_epoch: int | None,
) -> tuple[str, str | None]:
    """State whether a policy candidate has a test-evaluable checkpoint artifact."""
    checkpoint_path = resolve_repo_path(repo_root, artifacts.get("checkpoint"))
    test_outputs = resolve_repo_path(repo_root, artifacts.get("test_outputs"))
    if candidate_epoch is None:
        return "no_candidate", None
    if current_epoch is not None and candidate_epoch == current_epoch:
        checkpoint_available = checkpoint_path is not None and checkpoint_path.is_file()
        test_outputs_available = test_outputs is not None and test_outputs.is_file()
        if checkpoint_available and test_outputs_available:
            return "selected_checkpoint_test_outputs_available", str(checkpoint_path)
        path_text = str(checkpoint_path) if checkpoint_path else None
        return "selected_checkpoint_missing_outputs", path_text
    return "alternate_checkpoint_not_serialized", None


def build_policy_row(
    *,
    repo_root: Path,
    run_path: Path,
    run_data: dict[str, Any],
    trace: pl.DataFrame | None,
    policy_name: str,
    qwk_window: float | None,
    current_epoch: int | None,
    baseline_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one output row for one run/policy pair."""
    metadata = run_data.get("metadata", {})
    run_id = metadata.get("run_id", run_path.stem.split("_BalancedSoftmax")[0])
    model_name = metadata.get("model_name", "unknown")
    artifacts = run_data.get("artifacts", {})

    if trace is None:
        row = {
            "run_id": run_id,
            "model_name": model_name,
            "run_file": str(run_path),
            "family": model_family_for_run(run_id),
            "label_regime": label_regime_for_run(run_data),
            "policy": policy_name,
            "qwk_window": qwk_window,
            "trace_status": "missing",
            "candidate_epoch_zero_based": None,
            "candidate_epoch_display": None,
            "is_current_selected_epoch": None,
            "checkpoint_status": "trace_missing",
            "checkpoint_path": None,
        }
        row.update(
            {name: None for name in TRACE_REQUIRED_COLUMNS if name != "eligible"}
        )
        row.update({"eligible": None, "ineligible_reasons": None})
        row.update({f"delta_{name}": None for name in DELTA_METRICS})
        row.update(hard_dimension_values(repo_root, artifacts, epoch=None))
        row.update(baseline_test_metrics(run_data, include=False))
        return row

    if policy_name == BASELINE_POLICY:
        candidate = select_current_policy(trace, selected_epoch=current_epoch)
    elif qwk_window is not None:
        candidate = select_recall_window_policy(trace, qwk_window=qwk_window)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    candidate_epoch = int(candidate["epoch"]) if candidate is not None else None
    same_as_current = current_epoch is not None and candidate_epoch == current_epoch
    checkpoint_state, checkpoint_path = artifact_status(
        repo_root,
        artifacts,
        candidate_epoch=candidate_epoch,
        current_epoch=current_epoch,
    )
    row = {
        "run_id": run_id,
        "model_name": model_name,
        "run_file": str(run_path),
        "family": model_family_for_run(run_id),
        "label_regime": label_regime_for_run(run_data),
        "policy": policy_name,
        "qwk_window": qwk_window,
        "trace_status": "available",
        "candidate_epoch_zero_based": candidate_epoch,
        "candidate_epoch_display": (
            candidate_epoch + 1 if candidate_epoch is not None else None
        ),
        "is_current_selected_epoch": same_as_current,
        "checkpoint_status": checkpoint_state,
        "checkpoint_path": checkpoint_path,
    }
    for name in TRACE_REQUIRED_COLUMNS:
        if name == "epoch":
            continue
        value = candidate.get(name) if candidate is not None else None
        row[name] = value
    row["ineligible_reasons"] = (
        candidate.get("ineligible_reasons") if candidate is not None else None
    )

    for metric in DELTA_METRICS:
        if candidate is None or baseline_row is None:
            row[f"delta_{metric}"] = None
            continue
        row[f"delta_{metric}"] = float(candidate[metric]) - float(baseline_row[metric])

    row.update(hard_dimension_values(repo_root, artifacts, epoch=candidate_epoch))
    row.update(baseline_test_metrics(run_data, include=bool(same_as_current)))
    return row


def replay_run_file(repo_root: Path, run_file: Path) -> list[dict[str, Any]]:
    """Replay all policies for one run YAML file."""
    run_data = load_yaml(run_file)
    artifacts = run_data.get("artifacts", {})
    trace_path = resolve_repo_path(repo_root, artifacts.get("selection_trace"))
    trace = None
    if trace_path is not None and trace_path.is_file():
        trace = pl.read_parquet(trace_path)
        validate_trace_schema(trace, source=trace_path)

    current_epoch = selected_epoch_from_summary(repo_root, artifacts)
    if current_epoch is None:
        best_epoch = run_data.get("training_dynamics", {}).get("best_epoch")
        current_epoch = int(best_epoch) - 1 if best_epoch is not None else None

    baseline_row = (
        select_current_policy(trace, selected_epoch=current_epoch)
        if trace is not None
        else None
    )
    rows = [
        build_policy_row(
            repo_root=repo_root,
            run_path=run_file,
            run_data=run_data,
            trace=trace,
            policy_name=BASELINE_POLICY,
            qwk_window=None,
            current_epoch=current_epoch,
            baseline_row=baseline_row,
        )
    ]
    for policy_name, qwk_window in RECALL_POLICIES.items():
        rows.append(
            build_policy_row(
                repo_root=repo_root,
                run_path=run_file,
                run_data=run_data,
                trace=trace,
                policy_name=policy_name,
                qwk_window=qwk_window,
                current_epoch=current_epoch,
                baseline_row=baseline_row,
            )
        )
    return rows


def build_replay_frame(repo_root: Path, run_files: list[Path]) -> pl.DataFrame:
    """Build the full replay result frame."""
    rows: list[dict[str, Any]] = []
    for run_file in run_files:
        rows.extend(replay_run_file(repo_root, run_file))
    return pl.DataFrame(rows)


def write_outputs(
    frame: pl.DataFrame,
    output_dir: Path,
    *,
    run_files: list[Path],
) -> None:
    """Persist replay outputs in stable machine-readable and readable formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_dir / "selection_replay.parquet")
    frame.write_csv(output_dir / "selection_replay.csv")
    inventory = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policies": {
            BASELINE_POLICY: {
                "description": "Existing qwk_mean-first selected checkpoint"
            },
            **{
                name: {
                    "description": (
                        "Maximize recall_minus1 subject to qwk_mean within "
                        "window of best validation qwk_mean"
                    ),
                    "qwk_window": window,
                    "tie_breakers": [
                        "lower hedging_mean",
                        "higher calibration_global",
                        "earlier epoch",
                    ],
                }
                for name, window in RECALL_POLICIES.items()
            },
        },
        "run_files": [str(path) for path in run_files],
    }
    (output_dir / "inventory.yaml").write_text(
        yaml.safe_dump(inventory, sort_keys=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve run and artifact paths.",
    )
    parser.add_argument(
        "--run-file",
        action="append",
        dest="run_files",
        help=(
            "Run YAML file to replay. May be passed multiple times. "
            "Defaults to the t2r0 run set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Defaults to logs/experiments/artifacts/"
            "recall_checkpoint_replay_twinkl_t2r0_<timestamp>."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_files = [
        resolve_repo_path(repo_root, path)
        for path in (args.run_files if args.run_files else DEFAULT_RUN_FILES)
    ]
    missing = [str(path) for path in run_files if path is None or not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run files not found: {missing}")
    resolved_run_files = [path for path in run_files if path is not None]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            repo_root
            / "logs"
            / "experiments"
            / "artifacts"
            / f"recall_checkpoint_replay_twinkl_t2r0_{timestamp}"
        )
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    frame = build_replay_frame(repo_root, resolved_run_files)
    write_outputs(frame, output_dir, run_files=resolved_run_files)
    print(f"Wrote replay artifacts to {output_dir}")
    print(
        frame.select(
            [
                "run_id",
                "policy",
                "candidate_epoch_display",
                "qwk_mean",
                "recall_minus1",
                "delta_qwk_mean",
                "delta_recall_minus1",
                "checkpoint_status",
            ]
        )
    )


if __name__ == "__main__":
    main()
