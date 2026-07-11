import json

import polars as pl
import pytest

from src.vif.security_target import (
    build_security_target_variant,
    classify_reachability_bucket,
    write_security_target_artifacts,
)


@pytest.mark.parametrize(
    ("student", "profile", "full", "expected"),
    [
        (0, 0, 0, "visible_from_student_input"),
        (0, 0, 1, "requires_prior_trajectory"),
        (0, 1, 1, "requires_profile_or_bio_context"),
        (-1, 0, 1, "ambiguous_security_vs_conformity_or_tradition"),
    ],
)
def test_classify_reachability_bucket(student, profile, full, expected):
    assert (
        classify_reachability_bucket(
            student_visible_label=student,
            profile_only_label=profile,
            full_context_label=full,
        )
        == expected
    )


def test_build_security_target_variant_marks_subset_and_label_overreach():
    source = _source_frame()

    result = build_security_target_variant(source)

    assert result.height == 1
    row = result.to_dicts()[0]
    assert row["new_label"] == 0
    assert row["label_changed"] is True
    assert row["likely_label_error_or_overreach"] is True
    assert row["training_ready"] is False
    assert row["rationale"] == (
        "No student-visible Security rationale; target is neutral."
    )


def test_write_security_target_artifacts_records_training_blocker(tmp_path):
    source_path = tmp_path / "joined.csv"
    output_dir = tmp_path / "output"
    _source_frame().write_csv(source_path)

    target_path, summary_path = write_security_target_artifacts(
        joined_results_path=source_path,
        output_dir=output_dir,
    )

    assert pl.read_parquet(target_path).height == 1
    summary = json.loads(summary_path.read_text())
    assert summary["case_count"] == 1
    assert summary["changed_label_count"] == 1
    assert summary["training_ready"] is False
    assert "known frontier errors" in summary["training_blocker"]


def _source_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": ["security__example__1"],
            "dimension": ["security"],
            "persona_id": ["example"],
            "t_index": [1],
            "date": ["2026-01-01"],
            "persisted_label": [1],
            "student_visible_label": [0],
            "profile_only_label": [0],
            "full_context_label": [0],
            "student_visible_rationales_json": ["{}"],
        }
    )
