#!/usr/bin/env python3
"""Tune once on v8pb development labels, then score the locked promotion set."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.dataset import load_entries  # noqa: E402
from src.vif.drift_benchmark import (  # noqa: E402
    build_detection_decisions,
    build_eligible_trajectories,
    build_reference_episodes,
    detect_sustained_conflict_episodes,
    episode_metrics,
    evidence_from_ordinal_artifact,
    tune_detector_thresholds,
)
from src.vif.drift_scoring import score_mlp_cases  # noqa: E402
from src.vif.drift_target import (  # noqa: E402
    TARGET_VERSION,
    build_full_trajectory_cases,
    parse_aware_timestamp,
    record_promotion_threshold_receipt,
    sha256_file,
    sha256_json,
    target_split_sha256,
    validate_promotion_threshold_receipt,
    validate_target_manifest,
)

DEFAULT_MANIFEST = Path("config/evals/drift_v1_student_visible_v1.yaml")
DEFAULT_OUTPUT_ROOT = Path(
    "logs/experiments/artifacts/drift_target_twinkl_v8pb_20260711"
)
PROBABILITY_GRID = tuple(round(value, 2) for value in np.arange(0.30, 0.81, 0.05))


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _uncertainty_grid(evidence_df: pl.DataFrame) -> tuple[float, ...]:
    values = evidence_df["uncertainty"].drop_nulls()
    if values.is_empty():
        return (1.0,)
    quantiles = [
        float(values.quantile(quantile, interpolation="nearest"))
        for quantile in (0.25, 0.50, 0.75, 0.90, 1.0)
    ]
    return tuple(sorted(set(round(value, 6) for value in quantiles)))


def _load_run_020(manifest: dict[str, Any]) -> tuple[dict[str, Any], Path, Path, Path]:
    comparison = manifest.get("comparison") or {}
    if comparison.get("allowed_mlp_run_id") != "run_020":
        raise ValueError("v8pb only permits run_020 for the one-checkpoint comparison")
    run_yaml_path = _rooted(Path(comparison["allowed_mlp_run_yaml"]))
    if comparison.get("approved_run_yaml_sha256") != sha256_file(run_yaml_path):
        raise ValueError("Configured run_020 YAML does not match its approved digest")
    run = _read_yaml(run_yaml_path)
    if run.get("metadata", {}).get("run_id") != "run_020":
        raise ValueError("Configured v8pb run file is not run_020")
    row_count = sum(int(run["data"][field]) for field in ("n_train", "n_val", "n_test"))
    if row_count != 1460:
        raise ValueError(
            "run_020 provenance must cover the original 1,460 rows; "
            f"configured run covers {row_count}"
        )
    artifacts = run.get("artifacts") or {}
    checkpoint = _rooted(Path(artifacts["checkpoint"]))
    validation_outputs = _rooted(Path(artifacts["validation_outputs"]))
    if not checkpoint.is_file() or not validation_outputs.is_file():
        raise FileNotFoundError("run_020 checkpoint or validation outputs are missing")
    if comparison.get("approved_checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError(
            "Configured run_020 checkpoint does not match its approved digest"
        )
    if comparison.get("approved_validation_outputs_sha256") != sha256_file(
        validation_outputs
    ):
        raise ValueError(
            "Configured run_020 validation outputs do not match their approved digest"
        )
    return run, run_yaml_path, checkpoint, validation_outputs


def _model_cases(
    entries_df: pl.DataFrame,
    registry_df: pl.DataFrame,
    persona_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected_registry_rows = registry_df.filter(
        pl.col("persona_id").is_in(list(persona_ids))
    )
    registry_rows = {
        str(row["persona_id"]): row for row in selected_registry_rows.to_dicts()
    }
    cases = []
    for persona_id in persona_ids:
        profile = registry_rows.get(persona_id)
        if profile is None:
            raise ValueError(f"Promotion persona {persona_id} is missing from registry")
        entries = (
            entries_df.filter(pl.col("persona_id") == persona_id)
            .sort("t_index", "date")
            .select(
                "t_index",
                "date",
                "initial_entry",
                "nudge_text",
                "response_text",
            )
            .to_dicts()
        )
        if not entries:
            raise ValueError(f"Promotion persona {persona_id} has no journal entries")
        cases.append(
            {
                "persona_id": persona_id,
                "core_values": list(profile["core_values"]),
                "entries": entries,
            }
        )
    return cases


def _target_frames(
    target_path: Path,
    registry: pl.DataFrame,
    entries: pl.DataFrame,
    persona_ids: tuple[str, ...],
    *,
    source: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    target = pl.read_parquet(target_path).filter(
        pl.col("persona_id").is_in(list(persona_ids))
    )
    profiles = registry.filter(pl.col("persona_id").is_in(list(persona_ids)))
    required_columns = {"persona_id", "t_index", "date"}
    if not required_columns.issubset(target.columns):
        raise ValueError("Target variant lacks its required coordinate columns")
    expected_coordinates = {
        (str(row["persona_id"]), int(row["t_index"]), str(row["date"]))
        for row in entries.filter(pl.col("persona_id").is_in(list(persona_ids)))
        .select("persona_id", "t_index", "date")
        .to_dicts()
    }
    observed_coordinates = {
        (str(row["persona_id"]), int(row["t_index"]), str(row["date"]))
        for row in target.select("persona_id", "t_index", "date").to_dicts()
    }
    if target.height != len(observed_coordinates):
        raise ValueError("Target variant has duplicate entry coordinates")
    if observed_coordinates != expected_coordinates:
        raise ValueError("Target variant does not cover exactly the locked entries")
    reference = build_reference_episodes(target, profiles, source=source)
    eligible = build_eligible_trajectories(target, profiles)
    return target, reference, eligible


def _validate_materialized_target(
    summary_path: Path,
    target_path: Path,
    *,
    split_name: str,
    split_fingerprint: str,
    review_cases_fingerprint: str,
    require_promotable: bool,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    if summary.get("target_version") != TARGET_VERSION:
        raise ValueError("Target summary is not for the v8pb target")
    if summary.get("split") != split_name:
        raise ValueError("Target summary does not match the requested split")
    if not summary.get("materialization_complete"):
        raise ValueError("Target summary does not confirm complete materialization")
    if summary.get("target_variant_sha256") != sha256_file(target_path):
        raise ValueError("Target summary does not match the supplied target variant")
    if summary.get("target_split_sha256") != split_fingerprint:
        raise ValueError("Target summary does not match the locked target population")
    if summary.get("review_cases_sha256") != review_cases_fingerprint:
        raise ValueError("Target summary does not match the reviewed source cases")
    delta_path = summary_path.parent / "student_visible_target_delta.parquet"
    if not delta_path.is_file() or summary.get("target_delta_sha256") != sha256_file(
        delta_path
    ):
        raise ValueError("Target summary does not match its reviewed target delta")
    audit_manifest_path = summary_path.parent / "audit_manifest.json"
    audit_manifest = _read_json(audit_manifest_path)
    if summary.get("audit_manifest_sha256") != sha256_file(audit_manifest_path):
        raise ValueError("Target summary does not match the finalized audit manifest")
    if (
        audit_manifest.get("target_version") != TARGET_VERSION
        or audit_manifest.get("split") != split_name
    ):
        raise ValueError("Target audit manifest does not match the requested split")
    submissions = audit_manifest.get("assessment_submissions")
    if not isinstance(submissions, dict) or set(submissions) != {
        "reviewer_a",
        "reviewer_b",
    }:
        raise ValueError("Target audit manifest lacks both review submissions")
    submission_ids = {
        str(submission.get("reviewer_id")) for submission in submissions.values()
    }
    if len(submission_ids) != 2 or "None" in submission_ids:
        raise ValueError("Target audit manifest does not show distinct reviewers")
    if {submission.get("sha256") for submission in submissions.values()} != {
        summary.get("reviewer_a_sha256"),
        summary.get("reviewer_b_sha256"),
    }:
        raise ValueError("Target summary does not match reviewed response hashes")
    if audit_manifest.get("review_cases_sha256") != review_cases_fingerprint:
        raise ValueError(
            "Target audit manifest does not match the reviewed source cases"
        )
    if audit_manifest.get("outputs", {}).get(
        "reviewer_packet/blind_packet.json"
    ) != summary.get("packet_sha256"):
        raise ValueError("Target summary does not match the reviewed packet hash")
    if audit_manifest.get("outputs", {}).get(
        "parent_control/reconciliation_key.json"
    ) != summary.get("reconciliation_key_sha256"):
        raise ValueError("Target summary does not match the reconciliation-key hash")
    if require_promotable and not summary.get("promotable"):
        raise ValueError(
            "Promotion target has unresolved paired-review cases; do not score "
            "for a promotion claim"
        )
    return summary


def _review_cases_fingerprint(
    entries: pl.DataFrame,
    registry: pl.DataFrame,
    persona_ids: tuple[str, ...],
    *,
    split: str,
) -> str:
    cases = build_full_trajectory_cases(entries, registry, persona_ids, split=split)
    return sha256_json(cases)


def _validate_threshold_timing(
    threshold_payload: dict[str, Any],
    promotion_summary: dict[str, Any],
) -> None:
    submission_timestamps = promotion_summary.get("reviewer_submission_timestamps")
    if not isinstance(submission_timestamps, list) or len(submission_timestamps) != 2:
        raise ValueError("Promotion summary lacks reviewer submission timestamps")
    promotion_submission_times = [
        parse_aware_timestamp(value, field_name="promotion reviewer submission")
        for value in submission_timestamps
    ]
    threshold_created_at = parse_aware_timestamp(
        threshold_payload.get("created_at"),
        field_name="threshold manifest created_at",
    )
    if threshold_created_at >= min(promotion_submission_times):
        raise ValueError(
            "Threshold manifest must be created before the first promotion-review "
            "submission"
        )


def tune(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _rooted(args.manifest)
    manifest = _read_yaml(manifest_path)
    source = manifest["source"]
    registry = pl.read_parquet(_rooted(Path(source["registry_path"])))
    entries = load_entries(_rooted(Path(source["wrangled_dir"])))
    _validated_manifest, split = validate_target_manifest(
        manifest_path, registry, entries
    )
    _run, run_yaml, checkpoint, validation_outputs = _load_run_020(manifest)
    development_target_path = _rooted(args.development_target)
    development_summary_path = _rooted(args.development_summary)
    split_fingerprint = target_split_sha256(split)
    development_review_cases = _review_cases_fingerprint(
        entries,
        registry,
        split.development_persona_ids,
        split="development",
    )
    _validate_materialized_target(
        development_summary_path,
        development_target_path,
        split_name="development",
        split_fingerprint=split_fingerprint,
        review_cases_fingerprint=development_review_cases,
        require_promotable=False,
    )
    _target, reference, eligible = _target_frames(
        development_target_path,
        registry,
        entries,
        split.development_persona_ids,
        source=TARGET_VERSION,
    )
    if reference.is_empty():
        raise ValueError("Development student-visible target has no reference episodes")
    evidence = evidence_from_ordinal_artifact(
        pl.read_parquet(validation_outputs), registry, source="run_020"
    ).filter(pl.col("persona_id").is_in(list(split.development_persona_ids)))
    thresholds, grid = tune_detector_thresholds(
        evidence,
        reference,
        eligible,
        probability_thresholds=PROBABILITY_GRID,
        uncertainty_thresholds=_uncertainty_grid(evidence),
    )
    predictions = detect_sustained_conflict_episodes(evidence, **thresholds)
    decisions = build_detection_decisions(evidence, **thresholds)
    metrics = episode_metrics(reference, predictions, eligible, decisions_df=decisions)

    output_dir = _rooted(args.output_dir) / "development"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.write_parquet(output_dir / "threshold_grid.parquet")
    reference.write_parquet(output_dir / "reference_episodes.parquet")
    evidence.write_parquet(output_dir / "run_020_evidence.parquet")
    decisions.write_parquet(output_dir / "run_020_decisions.parquet")
    payload = {
        "target_version": TARGET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "development_threshold_selection",
        "run_id": "run_020",
        "run_yaml_sha256": sha256_file(run_yaml),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_outputs_sha256": sha256_file(validation_outputs),
        "development_target_path": str(development_target_path.relative_to(ROOT)),
        "development_target_sha256": sha256_file(development_target_path),
        "development_summary_sha256": sha256_file(development_summary_path),
        "target_split_sha256": split_fingerprint,
        "development_evidence_sha256": sha256_file(
            output_dir / "run_020_evidence.parquet"
        ),
        "development_reference_sha256": sha256_file(
            output_dir / "reference_episodes.parquet"
        ),
        "development_decisions_sha256": sha256_file(
            output_dir / "run_020_decisions.parquet"
        ),
        "threshold_grid_sha256": sha256_file(output_dir / "threshold_grid.parquet"),
        "thresholds": thresholds,
        "metrics": metrics,
        "promotion_population_locked": True,
    }
    threshold_path = output_dir / "thresholds.json"
    _write_json(threshold_path, payload)
    promotion_manifest_path = (
        development_summary_path.parents[2]
        / "promotion/parent_control/audit_manifest.json"
    )
    record_promotion_threshold_receipt(
        promotion_manifest_path,
        root=ROOT,
        threshold_path=threshold_path,
    )
    return payload


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _rooted(args.manifest)
    manifest = _read_yaml(manifest_path)
    source = manifest["source"]
    registry = pl.read_parquet(_rooted(Path(source["registry_path"])))
    entries = load_entries(_rooted(Path(source["wrangled_dir"])))
    _validated_manifest, split = validate_target_manifest(
        manifest_path, registry, entries
    )
    _run, run_yaml, checkpoint, validation_outputs = _load_run_020(manifest)
    split_fingerprint = target_split_sha256(split)

    threshold_payload = _read_json(_rooted(args.thresholds))
    if threshold_payload.get("target_version") != TARGET_VERSION:
        raise ValueError("Threshold manifest is not for the v8pb target")
    if threshold_payload.get("stage") != "development_threshold_selection":
        raise ValueError("Threshold manifest is not a development threshold selection")
    if threshold_payload.get("run_id") != "run_020":
        raise ValueError("Threshold manifest is not for run_020")
    if threshold_payload.get("run_yaml_sha256") != sha256_file(run_yaml):
        raise ValueError("Threshold manifest does not match the approved run YAML")
    development_target_path = _rooted(args.development_target)
    development_summary_path = _rooted(args.development_summary)
    development_review_cases = _review_cases_fingerprint(
        entries,
        registry,
        split.development_persona_ids,
        split="development",
    )
    _validate_materialized_target(
        development_summary_path,
        development_target_path,
        split_name="development",
        split_fingerprint=split_fingerprint,
        review_cases_fingerprint=development_review_cases,
        require_promotable=False,
    )
    if threshold_payload.get("development_target_sha256") != sha256_file(
        development_target_path
    ):
        raise ValueError(
            "Threshold manifest does not match the supplied development target"
        )
    if threshold_payload.get("development_summary_sha256") != sha256_file(
        development_summary_path
    ):
        raise ValueError(
            "Threshold manifest does not match the supplied development summary"
        )
    if threshold_payload.get("target_split_sha256") != split_fingerprint:
        raise ValueError(
            "Threshold manifest does not match the locked target population"
        )
    if threshold_payload.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("Threshold manifest does not match the run_020 checkpoint")
    if threshold_payload.get("validation_outputs_sha256") != sha256_file(
        validation_outputs
    ):
        raise ValueError("Threshold manifest does not match validation outputs")
    thresholds = threshold_payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Threshold manifest has no thresholds")

    promotion_target_path = _rooted(args.promotion_target)
    promotion_summary_path = _rooted(args.promotion_summary)
    promotion_review_cases = _review_cases_fingerprint(
        entries,
        registry,
        split.promotion_persona_ids,
        split="promotion",
    )
    promotion_summary = _validate_materialized_target(
        promotion_summary_path,
        promotion_target_path,
        split_name="promotion",
        split_fingerprint=split_fingerprint,
        review_cases_fingerprint=promotion_review_cases,
        require_promotable=False,
    )
    promotion_audit_manifest = _read_json(
        promotion_summary_path.parent / "audit_manifest.json"
    )
    threshold_receipt = validate_promotion_threshold_receipt(
        promotion_audit_manifest,
        root=ROOT,
    )
    threshold_path = _rooted(args.thresholds)
    if threshold_receipt.get("threshold_sha256") != sha256_file(threshold_path):
        raise ValueError(
            "Promotion threshold receipt does not match the supplied threshold file"
        )
    receipt_time = parse_aware_timestamp(
        threshold_receipt.get("recorded_at"),
        field_name="threshold receipt",
    )
    first_promotion_review = min(
        parse_aware_timestamp(
            value,
            field_name="promotion reviewer submission",
        )
        for value in promotion_summary["reviewer_submission_timestamps"]
    )
    if receipt_time >= first_promotion_review:
        raise ValueError(
            "Promotion threshold receipt was not recorded before promotion review"
        )
    _validate_threshold_timing(threshold_payload, promotion_summary)

    output_dir = _rooted(args.output_dir) / "promotion"
    if not promotion_summary.get("promotable"):
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_version": TARGET_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "stage": "promotion_review_blocked",
            "run_id": "run_020",
            "run_yaml_sha256": sha256_file(run_yaml),
            "promotion_target_path": str(promotion_target_path.relative_to(ROOT)),
            "promotion_target_sha256": sha256_file(promotion_target_path),
            "promotion_summary_sha256": sha256_file(promotion_summary_path),
            "threshold_manifest_sha256": sha256_file(_rooted(args.thresholds)),
            "thresholds": thresholds,
            "scoring_performed": False,
            "metrics": None,
            "unresolved_case_ids": promotion_summary["unresolved_case_ids"],
            "unresolved_entry_count": promotion_summary["unresolved_entry_count"],
            "disposition": (
                "promotion score not run because the locked target is unresolved; "
                "production wiring remains blocked"
            ),
        }
        _write_json(output_dir / "promotion_no_score.json", payload)
        report = "\n".join(
            [
                "# v8pb locked promotion review",
                "",
                "## Decision",
                "",
                "`run_020` was not scored on the locked promotion population. "
                "At least one paired-review case or entry remains unresolved, so "
                "scoring only the agreed subset would cherry-pick the evidence.",
                "",
                "## Audit result",
                "",
                f"- Unresolved cases: {promotion_summary['unresolved_case_ids']}",
                f"- Unresolved entries: {promotion_summary['unresolved_entry_count']}",
                f"- Frozen thresholds: `{json.dumps(thresholds, sort_keys=True)}`",
                "",
                "Production wiring remains blocked. The retired benchmark is not "
                "reproduced or used as a fallback.",
            ]
        )
        (output_dir / "experiment_review.md").write_text(
            report + "\n", encoding="utf-8"
        )
        return payload

    _target, reference, eligible = _target_frames(
        promotion_target_path,
        registry,
        entries,
        split.promotion_persona_ids,
        source=TARGET_VERSION,
    )
    cases = _model_cases(entries, registry, split.promotion_persona_ids)
    evidence_path = output_dir / "run_020_evidence.parquet"
    evidence = score_mlp_cases(
        cases=cases,
        checkpoint_path=checkpoint,
        arm_id="run_020",
        output_path=evidence_path,
    )
    predictions = detect_sustained_conflict_episodes(evidence, **thresholds)
    decisions = build_detection_decisions(evidence, **thresholds)
    metrics = episode_metrics(reference, predictions, eligible, decisions_df=decisions)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference.write_parquet(output_dir / "reference_episodes.parquet")
    decisions.write_parquet(output_dir / "run_020_decisions.parquet")
    predictions.write_parquet(output_dir / "run_020_predicted_episodes.parquet")
    payload = {
        "target_version": TARGET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "locked_promotion_evaluation",
        "run_id": "run_020",
        "run_yaml_sha256": sha256_file(run_yaml),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_outputs_sha256": sha256_file(validation_outputs),
        "promotion_target_path": str(promotion_target_path.relative_to(ROOT)),
        "promotion_target_sha256": sha256_file(promotion_target_path),
        "promotion_summary_sha256": sha256_file(promotion_summary_path),
        "target_split_sha256": split_fingerprint,
        "scoring_evidence_sha256": sha256_file(evidence_path),
        "scoring_provenance_sha256": sha256_file(
            evidence_path.with_suffix(".provenance.json")
        ),
        "threshold_manifest_sha256": sha256_file(_rooted(args.thresholds)),
        "thresholds": thresholds,
        "scoring_performed": True,
        "metrics": metrics,
        "coverage_limit": (
            "The fresh population covers targeted Security, Power, and Hedonism "
            "batches only. It cannot positively promote a general trigger."
        ),
        "disposition": (
            "production wiring remains blocked pending a broader diverse "
            "promotion surface"
        ),
    }
    _write_json(output_dir / "promotion_summary.json", payload)
    report = "\n".join(
        [
            "# v8pb student-visible drift target evaluation",
            "",
            "## Decision",
            "",
            "Production wiring remains blocked. This target repair gives a fairer "
            "evidence-only check for the existing `run_020` model, but the locked "
            "promotion population covers only Security, Power, and Hedonism. A pass "
            "there cannot approve a general ten-value trigger.",
            "",
            "## Locked promotion metrics",
            "",
            f"- Reference episodes: {metrics['reference_episodes']}",
            f"- Predicted episodes: {metrics['predicted_episodes']}",
            f"- Precision: {metrics['precision']:.3f}",
            f"- Recall: {metrics['recall']:.3f}",
            f"- F1: {metrics['f1']:.3f}",
            f"- Window false-positive rate: {metrics['false_positive_rate']:.3f}",
            f"- Thresholds: `{json.dumps(thresholds, sort_keys=True)}`",
            "",
            "## Interpretation boundary",
            "",
            "The retired frozen benchmark is not reproduced or compared here. A "
            "weak result means the target mismatch was not the whole problem; this "
            "evaluation alone cannot distinguish representation, data, and capacity "
            "causes. A strong result is still insufficient for production promotion "
            "without a broader, diverse, independently reviewed surface.",
        ]
    )
    (output_dir / "experiment_review.md").write_text(report + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--development-target", type=Path, required=True)
    tune_parser.add_argument("--development-summary", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--development-target", type=Path, required=True)
    evaluate_parser.add_argument("--development-summary", type=Path, required=True)
    evaluate_parser.add_argument("--promotion-target", type=Path, required=True)
    evaluate_parser.add_argument("--promotion-summary", type=Path, required=True)
    evaluate_parser.add_argument("--thresholds", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "tune":
        payload = tune(args)
        print(f"Wrote development thresholds: {payload['thresholds']}")
    else:
        payload = evaluate(args)
        print(f"Wrote promotion evaluation: {payload['metrics']}")


if __name__ == "__main__":
    main()
