"""Build an auditable Security target variant from reachability evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = {
    "case_id",
    "dimension",
    "persona_id",
    "t_index",
    "date",
    "persisted_label",
    "student_visible_label",
    "profile_only_label",
    "full_context_label",
    "student_visible_rationales_json",
}


def classify_reachability_bucket(
    *,
    student_visible_label: int,
    profile_only_label: int,
    full_context_label: int,
) -> str:
    """Classify which additional context changes the Security judgment."""
    if student_visible_label == profile_only_label == full_context_label:
        return "visible_from_student_input"
    if student_visible_label == profile_only_label:
        return "requires_prior_trajectory"
    if profile_only_label == full_context_label:
        return "requires_profile_or_bio_context"
    return "ambiguous_security_vs_conformity_or_tradition"


def build_security_target_variant(joined_results: pl.DataFrame) -> pl.DataFrame:
    """Return the sampled, student-visible Security target with provenance."""
    missing = REQUIRED_COLUMNS - set(joined_results.columns)
    if missing:
        raise ValueError(f"Missing required reachability columns: {sorted(missing)}")

    security = joined_results.filter(pl.col("dimension") == "security")
    if security.is_empty():
        raise ValueError("Reachability evidence contains no Security cases")
    if security.select("case_id").n_unique() != security.height:
        raise ValueError("Security reachability evidence has duplicate case IDs")

    rows = []
    for row in security.sort("case_id").iter_rows(named=True):
        student_label = int(row["student_visible_label"])
        profile_label = int(row["profile_only_label"])
        full_label = int(row["full_context_label"])
        condition_labels = [student_label, profile_label, full_label]
        agreement_count = Counter(condition_labels).most_common(1)[0][1]
        bucket = classify_reachability_bucket(
            student_visible_label=student_label,
            profile_only_label=profile_label,
            full_context_label=full_label,
        )
        old_label = int(row["persisted_label"])
        rows.append(
            {
                "case_id": row["case_id"],
                "example_id": f"{row['persona_id']}::{int(row['t_index'])}",
                "persona_id": row["persona_id"],
                "t_index": int(row["t_index"]),
                "date": row["date"],
                "dimension": "security",
                "old_label": old_label,
                "new_label": student_label,
                "label_changed": old_label != student_label,
                "target_policy": "student_visible_current_session_v1",
                "reachability_bucket": bucket,
                "likely_label_error_or_overreach": (
                    old_label not in condition_labels
                    and agreement_count == len(condition_labels)
                ),
                "review_source": "twinkl-747_condition_reruns",
                "reviewer": "existing_audit_bundle",
                "rationale": _security_rationale(
                    row["student_visible_rationales_json"]
                ),
                "confidence": (
                    "high"
                    if agreement_count == 3
                    else "medium"
                    if agreement_count == 2
                    else "low"
                ),
                "artifact_scope": "selected_audit_subset_only",
                "training_ready": False,
            }
        )
    return pl.DataFrame(rows)


def write_security_target_artifacts(
    *,
    joined_results_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the target table and a machine-readable audit summary."""
    source = Path(joined_results_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    target = build_security_target_variant(pl.read_csv(source))
    target_path = output / "security_target_variant.parquet"
    summary_path = output / "audit_summary.json"
    target.write_parquet(target_path)

    bucket_counts = {
        row["reachability_bucket"]: int(row["len"])
        for row in target.group_by("reachability_bucket").len().to_dicts()
    }
    summary = {
        "source": str(source),
        "target_policy": "student_visible_current_session_v1",
        "artifact_scope": "selected_audit_subset_only",
        "training_ready": False,
        "training_blocker": (
            "The 14 cases were selected from known frontier errors and do not form "
            "an unbiased full-corpus Security target."
        ),
        "case_count": target.height,
        "changed_label_count": target.filter(pl.col("label_changed")).height,
        "likely_label_error_or_overreach_count": target.filter(
            pl.col("likely_label_error_or_overreach")
        ).height,
        "reachability_bucket_counts": dict(sorted(bucket_counts.items())),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target_path, summary_path


def _security_rationale(raw_rationales: str | None) -> str:
    if not raw_rationales:
        return "No student-visible Security rationale; target is neutral."
    payload = json.loads(raw_rationales)
    return str(
        payload.get(
            "security", "No student-visible Security rationale; target is neutral."
        )
    )
