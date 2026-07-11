from __future__ import annotations

import json

import polars as pl
import pytest

from src.judge.labeling import load_schwartz_values, render_active_critic_state_prompt
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.security_target import (
    ACTIVE_CRITIC_STATE_CONTRACT_VERSION,
    EXPECTED_CONTEXT_FLAGS,
    TARGET_POLICY,
    build_full_corpus_security_target,
    build_security_target_variant,
    classify_reachability_bucket,
    sha256_canonical_json,
    sha256_file,
    write_security_target_artifacts,
)


@pytest.mark.parametrize(
    ("active", "profile", "full", "expected"),
    [
        (0, 0, 0, "matches_full_context"),
        (0, 0, 1, "changes_with_bio_or_history"),
        (0, 1, 1, "changes_between_active_state_and_legacy_profile_prompt"),
        (-1, 0, 1, "unresolved_context_sensitivity"),
    ],
)
def test_classify_reachability_bucket_reports_only_observed_deltas(
    active, profile, full, expected
):
    assert (
        classify_reachability_bucket(
            active_critic_state_label=active,
            profile_only_label=profile,
            full_context_label=full,
        )
        == expected
    )


def test_build_uses_exact_active_state_label_not_legacy_student_visible_label():
    source = _source_frame(
        legacy_student_visible=-1,
        profile_only=0,
        full_context=0,
    )
    manifest = _manifest()
    result = _result(manifest, security_label=0)

    target = build_security_target_variant(
        source,
        active_state_manifest=[manifest],
        active_state_results=[result],
    )

    row = target.to_dicts()[0]
    assert row["new_label"] == 0
    assert row["legacy_student_visible_label"] == -1
    assert row["target_policy"] == TARGET_POLICY
    assert row["audit_arm"] == "active_critic_state"
    assert row["training_ready"] is False
    assert row["evaluation_ready"] is False
    assert row["rationale_status"] == "not_applicable_neutral"


def test_build_fails_closed_without_exact_active_state_evidence():
    with pytest.raises(ValueError, match="case set does not match"):
        build_security_target_variant(
            _source_frame(),
            active_state_manifest=[],
            active_state_results=[],
        )


def test_build_rejects_result_with_stale_prompt_receipt():
    manifest = _manifest()
    result = _result(manifest, security_label=0)
    result["prompt_sha256"] = "stale"

    with pytest.raises(ValueError, match="prompt_sha256"):
        build_security_target_variant(
            _source_frame(),
            active_state_manifest=[manifest],
            active_state_results=[result],
        )


def test_build_rejects_manifest_whose_prompt_changed_after_hashing():
    manifest = _manifest()
    manifest["prompt"] = "changed prompt"

    with pytest.raises(ValueError, match="manifest prompt_sha256 mismatch"):
        build_security_target_variant(
            _source_frame(),
            active_state_manifest=[manifest],
            active_state_results=[_result(_manifest(), security_label=0)],
        )


def test_build_rejects_noncanonical_prompt_with_matching_hash():
    manifest = _manifest()
    manifest["prompt"] = f"{manifest['prompt']}\nHidden biography: secret context."
    manifest["prompt_sha256"] = sha256_canonical_json({"prompt": manifest["prompt"]})

    with pytest.raises(ValueError, match="canonical rendering"):
        build_security_target_variant(
            _source_frame(),
            active_state_manifest=[manifest],
            active_state_results=[_result(manifest, security_label=1)],
        )


def test_build_rejects_malformed_or_naive_review_timestamp():
    manifest = _manifest()
    for reviewed_at, message in (
        ("nonsense", "invalid reviewed_at"),
        ("2026-07-11T00:00:00", "must include a timezone"),
    ):
        result = _result(manifest, security_label=0)
        result["reviewed_at"] = reviewed_at
        with pytest.raises(ValueError, match=message):
            build_security_target_variant(
                _source_frame(),
                active_state_manifest=[manifest],
                active_state_results=[result],
            )


def test_build_rejects_non_ordinal_legacy_labels():
    source = _source_frame()
    source = source.with_columns(pl.lit(0.5).alias("profile_only_label"))

    with pytest.raises(ValueError, match="profile_only_label"):
        build_security_target_variant(
            source,
            active_state_manifest=[_manifest()],
            active_state_results=[_result(_manifest(), security_label=0)],
        )


def test_writer_records_exact_evidence_hashes_and_leaves_no_output_on_invalid_input(
    tmp_path,
):
    source_path = tmp_path / "joined.csv"
    manifest_path = tmp_path / "active_manifest.jsonl"
    results_path = tmp_path / "active_results.jsonl"
    output_dir = tmp_path / "target"
    _source_frame().write_csv(source_path)
    manifest = _manifest()
    result = _result(manifest, security_label=0)
    _write_jsonl(manifest_path, [manifest])
    _write_jsonl(results_path, [result])

    target_path, summary_path = write_security_target_artifacts(
        joined_results_path=source_path,
        active_state_manifest_path=manifest_path,
        active_state_results_path=results_path,
        output_dir=output_dir,
    )

    assert pl.read_parquet(target_path).height == 1
    summary = json.loads(summary_path.read_text())
    assert summary["training_ready"] is False
    assert summary["evaluation_ready"] is False
    assert summary["exact_state_manifest_sha256"] == sha256_file(manifest_path)
    assert summary["exact_state_results_sha256"] == sha256_file(results_path)
    assert "frozen-test" in summary["training_blocker"]

    invalid_output_dir = tmp_path / "invalid-target"
    _write_jsonl(results_path, [])
    with pytest.raises(ValueError, match="case set does not match"):
        write_security_target_artifacts(
            joined_results_path=source_path,
            active_state_manifest_path=manifest_path,
            active_state_results_path=results_path,
            output_dir=invalid_output_dir,
        )
    assert not invalid_output_dir.exists()


def test_full_corpus_target_majority_updates_only_security_label_vector_and_rationale():
    manifest = _manifest()
    base = pl.DataFrame(
        {
            "persona_id": ["example"],
            "t_index": [1],
            "date": ["2026-01-01"],
            "alignment_vector": [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
            "alignment_self_direction": [1],
            "alignment_stimulation": [0],
            "alignment_hedonism": [0],
            "alignment_achievement": [0],
            "alignment_power": [0],
            "alignment_security": [1],
            "alignment_conformity": [0],
            "alignment_tradition": [0],
            "alignment_benevolence": [0],
            "alignment_universalism": [0],
            "rationales_json": ['{"security":"old","self_direction":"keep"}'],
        }
    )
    passes = {}
    for index, label in enumerate((-1, -1, 0), start=1):
        result = _result(manifest, security_label=label)
        result.update(
            {
                "pass_index": index,
                "status": "ok",
                "response_id": f"r{index}",
                "response_model": "gpt-5.4-mini",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                "estimated_cost_usd": 0.01,
            }
        )
        passes[index] = [result]
    target, repaired = build_full_corpus_security_target(
        base, active_state_manifest=[manifest], review_passes=passes
    )
    assert target["new_label"].to_list() == [-1]
    assert target["decision_method"].to_list() == ["majority"]
    row = repaired.to_dicts()[0]
    assert row["alignment_security"] == -1
    assert row["alignment_vector"] == [1, 0, 0, 0, 0, -1, 0, 0, 0, 0]
    rationales = json.loads(row["rationales_json"])
    assert rationales["self_direction"] == "keep"
    assert rationales["security"] == "Direct session evidence."


def test_full_corpus_target_fails_closed_on_unresolved_three_way_tie():
    manifest = _manifest()
    base = pl.DataFrame(
        {
            "persona_id": ["example"],
            "t_index": [1],
            "date": ["2026-01-01"],
            "alignment_vector": [[0] * 10],
            "alignment_security": [0],
            "rationales_json": ["{}"],
        }
    )
    passes = {}
    for index, label in enumerate((-1, 0, 1), start=1):
        result = _result(manifest, security_label=label)
        result.update(
            {
                "pass_index": index,
                "status": "ok",
                "response_id": str(index),
                "response_model": "gpt-5.4-mini",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                "estimated_cost_usd": 0.0,
            }
        )
        passes[index] = [result]
    with pytest.raises(ValueError, match="Missing required tiebreak"):
        build_full_corpus_security_target(
            base, active_state_manifest=[manifest], review_passes=passes
        )


def _source_frame(
    *,
    legacy_student_visible: int = 0,
    profile_only: int = 0,
    full_context: int = 0,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": ["security__example__1"],
            "dimension": ["security"],
            "persona_id": ["example"],
            "t_index": [1],
            "date": ["2026-01-01"],
            "persisted_label": [1],
            "student_visible_label": [legacy_student_visible],
            "profile_only_label": [profile_only],
            "full_context_label": [full_context],
        }
    )


def _manifest() -> dict:
    state_input = {
        "window_size": 1,
        "session_content": 'Current session.\n\nNudge: "What happened?"',
        "profile_weights": {
            dimension: 1.0 if dimension == "security" else 0.0
            for dimension in SCHWARTZ_VALUE_ORDER
        },
    }
    prompt = render_active_critic_state_prompt(
        session_content=state_input["session_content"],
        profile_weights=[
            state_input["profile_weights"][dimension]
            for dimension in SCHWARTZ_VALUE_ORDER
        ],
        schwartz_config=load_schwartz_values("config/schwartz_values.yaml"),
    )
    return {
        "case_id": "security__example__1",
        "dimension": "security",
        "persona_id": "example",
        "t_index": 1,
        "date": "2026-01-01",
        "state_contract_version": ACTIVE_CRITIC_STATE_CONTRACT_VERSION,
        "context_flags": EXPECTED_CONTEXT_FLAGS,
        "state_input": state_input,
        "prompt": prompt,
        "state_input_sha256": sha256_canonical_json(state_input),
        "prompt_sha256": sha256_canonical_json({"prompt": prompt}),
    }


def _result(manifest: dict, *, security_label: int) -> dict:
    scores = {dimension: 0 for dimension in SCHWARTZ_VALUE_ORDER}
    scores["security"] = security_label
    return {
        "case_id": manifest["case_id"],
        "state_contract_version": manifest["state_contract_version"],
        "state_input_sha256": manifest["state_input_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "reviewer": "test-reviewer",
        "reviewed_at": "2026-07-11T00:00:00+00:00",
        "confidence": "high",
        "rationale_status": (
            "not_applicable_neutral" if security_label == 0 else "provided"
        ),
        "scores": scores,
        "rationales": (
            {} if security_label == 0 else {"security": "Direct session evidence."}
        ),
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
