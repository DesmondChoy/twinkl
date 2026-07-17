#!/usr/bin/env python3
"""Evaluate frozen twinkl-748 Hedonism pairs on incumbent/reference VIF families."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.judge import SCHWARTZ_VALUE_ORDER  # noqa: E402
from src.vif.critic_ordinal import OrdinalCriticBase  # noqa: E402
from src.vif.runtime import load_runtime_bundle  # noqa: E402

DEFAULT_BUNDLE = Path(
    "logs/experiments/artifacts/hedonism_hard_set_twinkl_748_20260712"
)
DEFAULT_FROZEN = DEFAULT_BUNDLE / "parent_control/frozen_hedonism_hard_set.parquet"
DEFAULT_OUTPUT = DEFAULT_BUNDLE / "parent_control/evaluation"

FAMILIES = {
    "incumbent": (19, 20, 21),
    "tail_sensitive_reference": (34, 35, 36),
}
TARGET_VERSION = "twinkl-748-hedonism-hard-set-v1"


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_for_record(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_frozen_hard_set(path: str | Path) -> pl.DataFrame:
    """Load the reconciled set and reject incomplete or malformed pairs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen hard-set not found: {path}. Complete reconciliation first."
        )
    frame = pl.read_parquet(path)
    required = {
        "review_pair_id",
        "review_entry_id",
        "source_pair_id",
        "source_variant_id",
        "family",
        "core_values",
        "journal_entry",
        "hedonism_target",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frozen hard-set is missing columns: {sorted(missing)}")
    if frame.is_empty():
        raise ValueError("Frozen hard-set contains no accepted entries")
    if frame["review_entry_id"].n_unique() != frame.height:
        raise ValueError("Frozen hard-set contains duplicate review_entry_id values")
    if frame["source_variant_id"].n_unique() != frame.height:
        raise ValueError("Frozen hard-set contains duplicate source_variant_id values")
    if frame["journal_entry"].is_null().any() or (
        frame["journal_entry"].str.strip_chars().str.len_chars() == 0
    ).any():
        raise ValueError("Frozen hard-set contains an empty journal entry")
    if set(frame["hedonism_target"].unique()) - {-1, 1}:
        raise ValueError("Frozen hard-set targets must be -1 or +1")
    if any(values != ["Hedonism"] for values in frame["core_values"].to_list()):
        raise ValueError("Every frozen entry must use core_values ['Hedonism']")

    for pair_id, pair in frame.group_by("review_pair_id"):
        pair_name = pair_id[0]
        if pair.height != 2:
            raise ValueError(f"Pair {pair_name} must contain exactly two entries")
        if set(pair["hedonism_target"].to_list()) != {-1, 1}:
            raise ValueError(f"Pair {pair_name} must contain one -1 and one +1 target")
    return frame.sort(["review_pair_id", "review_entry_id"])


def validate_review_receipt(frozen_path: str | Path) -> dict[str, str]:
    """Bind evaluation to the completed paired-review materialization receipt."""
    frozen_path = Path(frozen_path)
    parent_dir = frozen_path.parent
    summary_path = parent_dir / "review_summary.json"
    manifest_path = parent_dir / "audit_manifest.json"
    for path in (summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Review receipt file not found: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary.get("target_version") != TARGET_VERSION:
        raise ValueError("Review summary target version mismatch")
    if manifest.get("target_version") != TARGET_VERSION:
        raise ValueError("Audit manifest target version mismatch")
    if not manifest.get("materialization_complete"):
        raise ValueError("Audit manifest is not materialization-complete")
    frozen_sha256 = sha256_file(frozen_path)
    if summary.get("frozen_hard_set_sha256") != frozen_sha256:
        raise ValueError("Frozen hard-set SHA-256 does not match review summary")
    review_summary_sha256 = sha256_file(summary_path)
    if manifest.get("review_summary_sha256") != review_summary_sha256:
        raise ValueError("Review summary SHA-256 does not match audit manifest")
    return {
        "frozen_hard_set_sha256": frozen_sha256,
        "review_summary_sha256": review_summary_sha256,
        "audit_manifest_sha256": sha256_file(manifest_path),
    }


def resolve_checkpoints(
    *, root: str | Path = ROOT, run_ids: dict[str, tuple[int, ...]] = FAMILIES
) -> list[dict[str, Any]]:
    """Resolve checkpoint paths from the canonical run manifests."""
    root = Path(root)
    resolved = []
    for family, family_runs in run_ids.items():
        for run_id in family_runs:
            manifest_path = (
                root
                / "logs/experiments/runs"
                / f"run_{run_id:03d}_BalancedSoftmax.yaml"
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            checkpoint_value = manifest.get("artifacts", {}).get("checkpoint")
            if not checkpoint_value:
                raise ValueError(f"Run {run_id:03d} has no checkpoint artifact")
            checkpoint_path = Path(checkpoint_value)
            if not checkpoint_path.is_absolute():
                checkpoint_path = root / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Run {run_id:03d} checkpoint not found: {checkpoint_path}"
                )
            resolved.append(
                {
                    "family": family,
                    "run_id": run_id,
                    "run_name": f"run_{run_id:03d}",
                    "manifest_path": manifest_path,
                    "manifest_relative_path": str(manifest_path.relative_to(root)),
                    "manifest_sha256": sha256_file(manifest_path),
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_relative_path": _path_for_record(
                        checkpoint_path, root
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            )
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _runtime_signature_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("training_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing training_config")
    return {
        key: config.get(key)
        for key in (
            "encoder_model",
            "trust_remote_code",
            "truncate_dim",
            "text_prefix",
            "prompt_name",
            "prompt",
            "window_size",
            "history_pooling",
            "history_window_size",
            "history_summary_dim",
        )
    }


def load_checkpoint_model(
    run: dict[str, Any], *, device: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a later model without reloading the shared text encoder."""
    checkpoint = torch.load(
        run["checkpoint_path"], map_location=device, weights_only=True
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict) or "variant" not in model_config:
        raise ValueError(f"{run['run_name']} is not an ordinal VIF checkpoint")
    model = OrdinalCriticBase.from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _recall(targets: np.ndarray, predictions: np.ndarray, label: int) -> float | None:
    mask = targets == label
    if not mask.any():
        return None
    return float(np.mean(predictions[mask] == label))


def score_checkpoint(
    *,
    frame: pl.DataFrame,
    run: dict[str, Any],
    base_seed: int,
    mc_samples: int,
    confidence_threshold: float,
    model: torch.nn.Module,
    state_tensor: torch.Tensor,
    runtime_config: dict[str, Any],
    checkpoint: dict[str, Any],
    resolved_device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score one checkpoint on Hedonism only and return summary/output rows."""
    run_seed = base_seed + int(run["run_id"])
    _seed_everything(run_seed)
    model.eval()
    with torch.no_grad():
        _logits, probabilities_tensor = model.predict_logits_and_probabilities(
            state_tensor
        )
    probabilities = probabilities_tensor.detach().cpu().numpy()
    _seed_everything(run_seed)
    mc_mean_tensor, mc_std_tensor = model.predict_with_uncertainty(
        state_tensor, n_samples=mc_samples
    )
    mc_mean = mc_mean_tensor.detach().cpu().numpy()
    mc_std = mc_std_tensor.detach().cpu().numpy()

    dim_index = SCHWARTZ_VALUE_ORDER.index("hedonism")
    hedonism_probabilities = probabilities[:, dim_index, :]
    predictions = hedonism_probabilities.argmax(axis=1).astype(int) - 1
    scores = hedonism_probabilities[:, 2] - hedonism_probabilities[:, 0]
    targets = frame["hedonism_target"].to_numpy().astype(int)
    confidences = hedonism_probabilities.max(axis=1)
    wrong = predictions != targets
    high_confidence_wrong = wrong & (confidences >= confidence_threshold)

    output_rows = []
    for index, row in enumerate(frame.iter_rows(named=True)):
        output_rows.append(
            {
                "model_family": run["family"],
                "run_name": run["run_name"],
                "run_id": int(run["run_id"]),
                "review_pair_id": row["review_pair_id"],
                "review_entry_id": row["review_entry_id"],
                "source_pair_id": row["source_pair_id"],
                "source_variant_id": row["source_variant_id"],
                "scenario_family": row["family"],
                "hedonism_target": int(targets[index]),
                "predicted_class": int(predictions[index]),
                "probability_minus1": float(hedonism_probabilities[index, 0]),
                "probability_zero": float(hedonism_probabilities[index, 1]),
                "probability_plus1": float(hedonism_probabilities[index, 2]),
                "direction_score": float(scores[index]),
                "mc_class_mean": float(mc_mean[index, dim_index]),
                "mc_class_std": float(mc_std[index, dim_index]),
                "exact_correct": bool(not wrong[index]),
                "high_confidence_error": bool(high_confidence_wrong[index]),
            }
        )

    strict_pair_results = []
    directional_results = []
    pair_margins = []
    pair_ids = frame["review_pair_id"].unique(maintain_order=True).to_list()
    for pair_id in pair_ids:
        indices = np.flatnonzero(frame["review_pair_id"].to_numpy() == pair_id)
        negative_index = indices[np.flatnonzero(targets[indices] == -1)[0]]
        positive_index = indices[np.flatnonzero(targets[indices] == 1)[0]]
        margin = float(scores[positive_index] - scores[negative_index])
        strict_pair_results.append(bool((~wrong[indices]).all()))
        directional_results.append(margin > 0.0)
        pair_margins.append(margin)

    if len(np.unique(predictions)) < 2:
        qwk = float("nan")
    else:
        qwk = float(
            cohen_kappa_score(
                targets, predictions, labels=[-1, 0, 1], weights="quadratic"
            )
        )
    pair_margin_array = np.asarray(pair_margins, dtype=float)
    summary = {
        "model_family": run["family"],
        "run_name": run["run_name"],
        "run_id": int(run["run_id"]),
        "seed": run_seed,
        "entry_count": int(frame.height),
        "pair_count": len(pair_ids),
        "exact_accuracy": float(np.mean(~wrong)),
        "class_recall": {
            "minus1": _recall(targets, predictions, -1),
            "zero": _recall(targets, predictions, 0),
            "plus1": _recall(targets, predictions, 1),
        },
        "strict_both_members_correct_pair_rate": float(np.mean(strict_pair_results)),
        "pair_directional_accuracy": float(np.mean(directional_results)),
        "pair_margin": {
            "mean": float(np.mean(pair_margin_array)),
            "median": float(np.median(pair_margin_array)),
            "minimum": float(np.min(pair_margin_array)),
            "maximum": float(np.max(pair_margin_array)),
        },
        "qwk_secondary": _finite_or_none(qwk),
        "mc_class_mean": {
            "mean": float(mc_mean[:, dim_index].mean()),
            "minimum": float(mc_mean[:, dim_index].min()),
            "maximum": float(mc_mean[:, dim_index].max()),
        },
        "mc_class_std": {
            "mean": float(mc_std[:, dim_index].mean()),
            "median": float(np.median(mc_std[:, dim_index])),
            "maximum": float(mc_std[:, dim_index].max()),
        },
        "high_confidence_error_threshold": confidence_threshold,
        "high_confidence_error_count": int(high_confidence_wrong.sum()),
        "high_confidence_error_rate": float(high_confidence_wrong.mean()),
        "high_confidence_error_entry_ids": frame.filter(
            pl.Series(high_confidence_wrong)
        )["review_entry_id"].to_list(),
        "runtime": {
            "device": resolved_device,
            "mc_samples": mc_samples,
            "encoder_model": runtime_config["encoder"]["model_name"],
            "truncate_dim": runtime_config["encoder"].get("truncate_dim"),
            "text_prefix": runtime_config["encoder"].get("text_prefix"),
            "window_size": runtime_config["state_encoder"]["window_size"],
            "checkpoint_epoch": checkpoint.get("epoch"),
        },
        "manifest_path": run["manifest_relative_path"],
        "manifest_sha256": run["manifest_sha256"],
        "checkpoint_path": run["checkpoint_relative_path"],
        "checkpoint_sha256": run["checkpoint_sha256"],
    }
    del probabilities_tensor
    del mc_mean_tensor, mc_std_tensor, probabilities, mc_mean, mc_std
    return summary, output_rows


def _range(values: list[float | None]) -> dict[str, float | None]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(value)
    ]
    if not finite:
        return {"median": None, "minimum": None, "maximum": None}
    return {
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def summarize_families(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize checkpoint-seed variation without pooling runs as samples."""
    output = []
    for family in FAMILIES:
        rows = [row for row in run_summaries if row["model_family"] == family]
        output.append(
            {
                "model_family": family,
                "runs": [row["run_name"] for row in rows],
                "exact_accuracy": _range([row["exact_accuracy"] for row in rows]),
                "recall_minus1": _range(
                    [row["class_recall"]["minus1"] for row in rows]
                ),
                "recall_plus1": _range(
                    [row["class_recall"]["plus1"] for row in rows]
                ),
                "strict_both_members_correct_pair_rate": _range(
                    [row["strict_both_members_correct_pair_rate"] for row in rows]
                ),
                "pair_directional_accuracy": _range(
                    [row["pair_directional_accuracy"] for row in rows]
                ),
                "pair_margin_mean": _range(
                    [row["pair_margin"]["mean"] for row in rows]
                ),
                "qwk_secondary": _range([row["qwk_secondary"] for row in rows]),
                "mc_class_std_mean": _range(
                    [row["mc_class_std"]["mean"] for row in rows]
                ),
                "high_confidence_error_rate": _range(
                    [row["high_confidence_error_rate"] for row in rows]
                ),
            }
        )
    return output


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        "# twinkl-748 Hedonism hard-set evaluation",
        "",
        (
            "> Codex-reviewed diagnostic evidence; not human validation or a "
            "promotion surface."
        ),
        "",
        (
            f"Frozen entries: {summary['entry_count']} across "
            f"{summary['pair_count']} pairs."
        ),
        f"MC samples per checkpoint: {summary['mc_samples']}.",
        "",
        "## Family summary",
        "",
        (
            "| Family | Exact accuracy | Recall -1 | Recall +1 | Strict pair | "
            "Directional | Mean margin | QWK (secondary) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["family_summaries"]:
        def cell(key: str, family_row: dict[str, Any] = row) -> str:
            metric = family_row[key]
            if metric["median"] is None:
                return "n/a"
            return (
                f"{metric['median']:.3f} "
                f"[{metric['minimum']:.3f}, {metric['maximum']:.3f}]"
            )

        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_family"],
                    cell("exact_accuracy"),
                    cell("recall_minus1"),
                    cell("recall_plus1"),
                    cell("strict_both_members_correct_pair_rate"),
                    cell("pair_directional_accuracy"),
                    cell("pair_margin_mean"),
                    cell("qwk_secondary"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Directional accuracy uses `P(+1) - P(-1)` and succeeds only when the",
            (
                "positive member scores above its matched negative member. QWK is "
                "secondary"
            ),
            "because this is a small, deliberately balanced diagnostic set.",
            "",
            "## Per-run summary",
            "",
            (
                "| Run | Family | Exact | Recall -1 | Recall +1 | Strict pair | "
                "Directional | High-confidence errors |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["run_summaries"]:
        lines.append(
            f"| {row['run_name']} | {row['model_family']} | "
            f"{row['exact_accuracy']:.3f} | {row['class_recall']['minus1']:.3f} | "
            f"{row['class_recall']['plus1']:.3f} | "
            f"{row['strict_both_members_correct_pair_rate']:.3f} | "
            f"{row['pair_directional_accuracy']:.3f} | "
            f"{row['high_confidence_error_count']} |"
        )
    return "\n".join(lines) + "\n"


def evaluate(
    *,
    frozen_path: str | Path,
    output_dir: str | Path,
    base_seed: int = 748,
    mc_samples: int = 50,
    confidence_threshold: float = 0.8,
    device: str = "cpu",
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Run the six-checkpoint evaluation and write immutable result artifacts."""
    if mc_samples < 2:
        raise ValueError("mc_samples must be at least 2")
    if not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in (0, 1]")
    frozen_path = Path(frozen_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_dir}")
    frame = load_frozen_hard_set(frozen_path)
    review_receipt = validate_review_receipt(frozen_path)
    runs = resolve_checkpoints(root=root)
    (
        first_model,
        state_encoder,
        runtime_config,
        first_checkpoint,
        resolved_device,
    ) = load_runtime_bundle(
        runs[0]["checkpoint_path"],
        config_path=Path(root) / "config/vif.yaml",
        device=device,
    )
    if state_encoder.input_entry_count != 1 or state_encoder.state_dim != 266:
        raise ValueError(
            "Checkpoint family is outside the frozen current-entry contract: "
            f"input_entry_count={state_encoder.input_entry_count}, "
            f"state_dim={state_encoder.state_dim}"
        )
    expected_runtime_signature = _runtime_signature_from_checkpoint(first_checkpoint)
    states = np.stack(
        [
            state_encoder.build_state_vector(
                texts=[str(text)],
                dates=["2026-01-01"],
                core_values=["Hedonism"],
            )
            for text in frame["journal_entry"].to_list()
        ]
    ).astype(np.float32)
    state_tensor = torch.from_numpy(states).to(resolved_device)
    run_summaries = []
    output_rows = []
    for run_index, run in enumerate(runs):
        if run_index == 0:
            model = first_model
            checkpoint = first_checkpoint
        else:
            model, checkpoint = load_checkpoint_model(run, device=resolved_device)
        if _runtime_signature_from_checkpoint(checkpoint) != expected_runtime_signature:
            raise ValueError(
                f"{run['run_name']} does not share the frozen runtime contract"
            )
        summary, rows = score_checkpoint(
            frame=frame,
            run=run,
            base_seed=base_seed,
            mc_samples=mc_samples,
            confidence_threshold=confidence_threshold,
            model=model,
            state_tensor=state_tensor,
            runtime_config=runtime_config,
            checkpoint=checkpoint,
            resolved_device=resolved_device,
        )
        run_summaries.append(summary)
        output_rows.extend(rows)
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del state_encoder, state_tensor, first_model, first_checkpoint

    result = {
        "schema_version": 1,
        "target": TARGET_VERSION,
        "evidence_class": "Codex-reviewed diagnostic; not human validation",
        "frozen_hard_set_path": _path_for_record(frozen_path, Path(root)),
        **review_receipt,
        "entry_count": frame.height,
        "pair_count": frame["review_pair_id"].n_unique(),
        "base_seed": base_seed,
        "mc_samples": mc_samples,
        "confidence_threshold": confidence_threshold,
        "run_summaries": run_summaries,
        "family_summaries": summarize_families(run_summaries),
        "limitations": [
            "Labels are paired Codex agreement, not human ground truth.",
            (
                "The small balanced diagnostic makes QWK unstable; pair metrics are "
                "primary."
            ),
            (
                "Family summaries report seed medians and ranges; checkpoints are not "
                "pooled as independent cases."
            ),
        ],
    }
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "hedonism_predictions.parquet"
    pl.DataFrame(output_rows).write_parquet(predictions_path)
    result["predictions_path"] = _path_for_record(predictions_path, Path(root))
    result["predictions_sha256"] = sha256_file(predictions_path)
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(_format_report(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-seed", type=int, default=748)
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    return parser


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(
        frozen_path=_rooted(args.frozen_path),
        output_dir=_rooted(args.output_dir),
        base_seed=args.base_seed,
        mc_samples=args.mc_samples,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
