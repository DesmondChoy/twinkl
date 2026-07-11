"""Build a fail-closed Security target variant from exact-state review evidence."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.judge.labeling import load_schwartz_values, render_active_critic_state_prompt
from src.models.judge import SCHWARTZ_VALUE_ORDER, AlignmentScores

ACTIVE_CRITIC_STATE_CONTRACT_VERSION = "active_critic_state_v1"
TARGET_POLICY = "security_active_critic_state_v1"
ARTIFACT_SCOPE = "selected_frozen_test_audit_subset_only"
FULL_CORPUS_ARTIFACT_SCOPE = "full_corpus_security_target"
EXPECTED_FULL_CORPUS_ROWS = 1651

REQUIRED_LEGACY_COLUMNS = {
    "case_id",
    "dimension",
    "persona_id",
    "t_index",
    "date",
    "persisted_label",
    "student_visible_label",
    "profile_only_label",
    "full_context_label",
}

REQUIRED_MANIFEST_FIELDS = {
    "case_id",
    "dimension",
    "persona_id",
    "t_index",
    "date",
    "state_contract_version",
    "state_input_sha256",
    "prompt_sha256",
    "context_flags",
    "state_input",
    "prompt",
}

REQUIRED_RESULT_FIELDS = {
    "case_id",
    "state_contract_version",
    "state_input_sha256",
    "prompt_sha256",
    "reviewer",
    "reviewed_at",
    "confidence",
    "rationale_status",
    "scores",
    "rationales",
}

EXPECTED_CONTEXT_FLAGS = {
    "runtime_session_text_included": True,
    "profile_weights_included": True,
    "date_included": False,
    "demographics_included": False,
    "bio_included": False,
    "previous_entries_included": False,
    "raw_core_value_names_included": False,
    "labels_or_rationales_included": False,
    "generation_metadata_included": False,
}

VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_RATIONALE_STATUS = {"provided", "not_applicable_neutral"}
SCHWARTZ_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "schwartz_values.yaml"
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without interpreting its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_json(payload: Mapping[str, Any]) -> str:
    """Hash a stable JSON representation used to bind review inputs to results."""
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and reject malformed or non-object records."""
    records: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number} is not valid JSON.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number} must contain a JSON object.")
            records.append(record)
    return records


def classify_reachability_bucket(
    *,
    active_critic_state_label: int,
    profile_only_label: int,
    full_context_label: int,
) -> str:
    """Describe observed context deltas without claiming an unmeasured cause.

    The legacy full-context arm adds biography and prior entries together, so it
    cannot establish whether a disagreement is caused by biography, trajectory,
    prompt framing, or their interaction.
    """
    if active_critic_state_label == full_context_label:
        return "matches_full_context"
    if active_critic_state_label == profile_only_label:
        return "changes_with_bio_or_history"
    if profile_only_label == full_context_label:
        return "changes_between_active_state_and_legacy_profile_prompt"
    return "unresolved_context_sensitivity"


def build_security_target_variant(
    joined_results: pl.DataFrame,
    *,
    active_state_manifest: Iterable[Mapping[str, Any]],
    active_state_results: Iterable[Mapping[str, Any]],
) -> pl.DataFrame:
    """Return a diagnostic-only target from receipt-bound exact-state reviews.

    The legacy three-arm ``twinkl-747`` values are retained only as diagnostic
    provenance. ``new_label`` always comes from the separately reviewed
    ``active_critic_state_v1`` arm.
    """
    missing = REQUIRED_LEGACY_COLUMNS - set(joined_results.columns)
    if missing:
        raise ValueError(
            f"Missing required legacy reachability columns: {sorted(missing)}."
        )

    security = joined_results.filter(pl.col("dimension") == "security")
    if security.is_empty():
        raise ValueError("Reachability evidence contains no Security cases.")
    if security.select("case_id").n_unique() != security.height:
        raise ValueError("Security reachability evidence has duplicate case IDs.")

    source_rows = {
        str(row["case_id"]): row
        for row in security.sort("case_id").iter_rows(named=True)
    }
    manifest_by_case = _validate_manifest(active_state_manifest, source_rows)
    result_by_case = _validate_results(active_state_results, manifest_by_case)

    rows = []
    for case_id, source_row in source_rows.items():
        manifest = manifest_by_case[case_id]
        result = result_by_case[case_id]
        active_label = result["security_label"]
        profile_label = _as_alignment_label(
            source_row["profile_only_label"], field="profile_only_label"
        )
        full_label = _as_alignment_label(
            source_row["full_context_label"], field="full_context_label"
        )
        old_label = _as_alignment_label(
            source_row["persisted_label"], field="persisted_label"
        )
        legacy_student_visible_label = _as_alignment_label(
            source_row["student_visible_label"], field="student_visible_label"
        )

        rows.append(
            {
                "case_id": case_id,
                "example_id": (
                    f"{source_row['persona_id']}::{int(source_row['t_index'])}"
                ),
                "persona_id": str(source_row["persona_id"]),
                "t_index": int(source_row["t_index"]),
                "date": str(source_row["date"]),
                "dimension": "security",
                "old_label": old_label,
                "new_label": active_label,
                "label_changed": old_label != active_label,
                "target_policy": TARGET_POLICY,
                "audit_arm": "active_critic_state",
                "state_contract_version": ACTIVE_CRITIC_STATE_CONTRACT_VERSION,
                "state_input_sha256": str(manifest["state_input_sha256"]),
                "prompt_sha256": str(manifest["prompt_sha256"]),
                "reachability_bucket": classify_reachability_bucket(
                    active_critic_state_label=active_label,
                    profile_only_label=profile_label,
                    full_context_label=full_label,
                ),
                "likely_label_error_or_overreach": (
                    old_label != active_label and active_label == full_label
                ),
                "review_source": "active_critic_state_review",
                "reviewer": result["reviewer"],
                "reviewed_at": result["reviewed_at"],
                "rationale": result["rationale"],
                "rationale_status": result["rationale_status"],
                "confidence": result["confidence"],
                "legacy_student_visible_label": legacy_student_visible_label,
                "legacy_profile_only_label": profile_label,
                "legacy_full_context_label": full_label,
                "legacy_condition_pattern": (
                    f"student_visible={legacy_student_visible_label};"
                    f"profile_only={profile_label};full_context={full_label}"
                ),
                "artifact_scope": ARTIFACT_SCOPE,
                "training_ready": False,
                "evaluation_ready": False,
            }
        )

    return pl.DataFrame(rows).sort("case_id")


def write_security_target_artifacts(
    *,
    joined_results_path: str | Path,
    active_state_manifest_path: str | Path,
    active_state_results_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Validate exact-state evidence, then atomically write a diagnostic artifact.

    All evidence is validated before the target output directory is created. A
    selected frozen-test subset remains unsuitable for retraining or model
    evaluation even after an exact-state review is complete.
    """
    source_path = Path(joined_results_path)
    manifest_path = Path(active_state_manifest_path)
    results_path = Path(active_state_results_path)
    output = Path(output_dir)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Security target artifact: {output}"
        )

    target = build_security_target_variant(
        pl.read_csv(source_path),
        active_state_manifest=read_jsonl(manifest_path),
        active_state_results=read_jsonl(results_path),
    )

    bucket_counts = {
        row["reachability_bucket"]: int(row["len"])
        for row in target.group_by("reachability_bucket").len().to_dicts()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        target_path = temporary_dir / "security_target_variant.parquet"
        summary_path = temporary_dir / "audit_summary.json"
        target.write_parquet(target_path)
        summary = {
            "artifact_scope": ARTIFACT_SCOPE,
            "audit_arm": "active_critic_state",
            "case_count": target.height,
            "changed_label_count": target.filter(pl.col("label_changed")).height,
            "evaluation_ready": False,
            "exact_state_manifest_sha256": sha256_file(manifest_path),
            "exact_state_results_sha256": sha256_file(results_path),
            "likely_label_error_or_overreach_count": target.filter(
                pl.col("likely_label_error_or_overreach")
            ).height,
            "reachability_bucket_counts": dict(sorted(bucket_counts.items())),
            "source_labels_sha256": sha256_file(source_path),
            "target_policy": TARGET_POLICY,
            "target_variant_sha256": sha256_file(target_path),
            "training_blocker": (
                "This selected frozen-test diagnostic subset is not a full-corpus "
                "training target and must not be used for retraining or evaluation."
            ),
            "training_ready": False,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_dir.rename(output)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return (
        output / "security_target_variant.parquet",
        output / "audit_summary.json",
    )


def build_full_corpus_security_target(
    base_labels: pl.DataFrame,
    *,
    active_state_manifest: Iterable[Mapping[str, Any]],
    review_passes: Mapping[int, Iterable[Mapping[str, Any]]],
    tiebreak_results: Iterable[Mapping[str, Any]] = (),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a full-corpus majority target and loader-compatible label variant."""
    required = {
        "persona_id",
        "t_index",
        "date",
        "alignment_vector",
        "alignment_security",
        "rationales_json",
    }
    missing = required - set(base_labels.columns)
    if missing:
        raise ValueError(f"Base labels are missing required fields: {sorted(missing)}.")
    if (
        base_labels.select(["persona_id", "t_index", "date"]).n_unique()
        != base_labels.height
    ):
        raise ValueError("Base labels have duplicate entry coordinates.")
    if set(review_passes) != {1, 2, 3}:
        raise ValueError(
            "Full-corpus target requires exactly review passes 1, 2, and 3."
        )

    source_rows: dict[str, dict[str, Any]] = {}
    for row in base_labels.iter_rows(named=True):
        case_id = f"security__{row['persona_id']}__{int(row['t_index'])}"
        source_rows[case_id] = {**row, "case_id": case_id, "dimension": "security"}
    manifest_by_case = _validate_manifest(active_state_manifest, source_rows)
    validated_passes: dict[int, dict[str, dict[str, Any]]] = {}
    for pass_index in (1, 2, 3):
        raw_rows = list(review_passes[pass_index])
        for row in raw_rows:
            _validate_full_review_provenance(row, pass_index=pass_index)
        validated_passes[pass_index] = _validate_results(raw_rows, manifest_by_case)

    tie_rows = list(tiebreak_results)
    tie_by_case: dict[str, dict[str, Any]] = {}
    if tie_rows:
        tie_manifest = {
            case_id: manifest_by_case[case_id]
            for case_id in {str(row["case_id"]) for row in tie_rows}
            if case_id in manifest_by_case
        }
        for row in tie_rows:
            _validate_full_review_provenance(row, pass_index=4)
        tie_by_case = _validate_results(tie_rows, tie_manifest)

    target_rows: list[dict[str, Any]] = []
    replacement: dict[str, tuple[int, str]] = {}
    expected_ties: set[str] = set()
    for case_id in sorted(source_rows):
        votes = [
            validated_passes[index][case_id]["security_label"] for index in (1, 2, 3)
        ]
        counts = Counter(votes)
        if len(counts) == 3:
            expected_ties.add(case_id)
            tie = tie_by_case.get(case_id)
            if tie is None:
                raise ValueError(f"Missing required tiebreak review for {case_id}.")
            votes.append(tie["security_label"])
            counts = Counter(votes)
            decision_method = "tie_break_review"
        else:
            decision_method = "unanimous" if max(counts.values()) == 3 else "majority"
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]
        if len(winners) != 1:
            raise ValueError(f"Tiebreak review did not resolve {case_id}.")
        new_label = winners[0]
        selected_results = [
            validated_passes[index][case_id]
            for index in (1, 2, 3)
            if validated_passes[index][case_id]["security_label"] == new_label
        ]
        if (
            case_id in tie_by_case
            and tie_by_case[case_id]["security_label"] == new_label
        ):
            selected_results.append(tie_by_case[case_id])
        rationale = next(
            (row["rationale"] for row in selected_results if row["rationale"]), ""
        )
        total = len(votes)
        entropy = -sum(
            (count / total) * math.log2(count / total) for count in counts.values()
        )
        old_label = _as_alignment_label(
            source_rows[case_id]["alignment_security"], field="alignment_security"
        )
        example_id = (
            f"{source_rows[case_id]['persona_id']}::"
            f"{int(source_rows[case_id]['t_index'])}"
        )
        target_rows.append(
            {
                "case_id": case_id,
                "example_id": example_id,
                "persona_id": source_rows[case_id]["persona_id"],
                "t_index": int(source_rows[case_id]["t_index"]),
                "date": str(source_rows[case_id]["date"]),
                "dimension": "security",
                "old_label": old_label,
                "new_label": new_label,
                "label_changed": old_label != new_label,
                "vote_minus1": counts.get(-1, 0),
                "vote_neutral": counts.get(0, 0),
                "vote_plus1": counts.get(1, 0),
                "vote_count": total,
                "agreement_count": max_count,
                "vote_entropy": entropy,
                "decision_method": decision_method,
                "review_confidences": json.dumps(
                    [validated_passes[i][case_id]["confidence"] for i in (1, 2, 3)]
                    + (
                        [tie_by_case[case_id]["confidence"]]
                        if case_id in tie_by_case
                        else []
                    )
                ),
                "rationale": rationale,
                "target_policy": TARGET_POLICY,
                "state_contract_version": ACTIVE_CRITIC_STATE_CONTRACT_VERSION,
                "state_input_sha256": manifest_by_case[case_id]["state_input_sha256"],
                "prompt_sha256": manifest_by_case[case_id]["prompt_sha256"],
                "artifact_scope": FULL_CORPUS_ARTIFACT_SCOPE,
                "training_ready": True,
                "evaluation_ready": True,
            }
        )
        replacement[case_id] = (new_label, rationale)
    if set(tie_by_case) != expected_ties:
        raise ValueError(
            "Tiebreak result set does not exactly match the three-way ties."
        )

    label_rows: list[dict[str, Any]] = []
    security_index = SCHWARTZ_VALUE_ORDER.index("security")
    for row in base_labels.iter_rows(named=True):
        case_id = f"security__{row['persona_id']}__{int(row['t_index'])}"
        new_label, rationale = replacement[case_id]
        updated = dict(row)
        vector = list(row["alignment_vector"])
        if len(vector) != len(SCHWARTZ_VALUE_ORDER):
            raise ValueError(f"Invalid alignment_vector length for {case_id}.")
        vector[security_index] = new_label
        updated["alignment_security"] = new_label
        updated["alignment_vector"] = vector
        try:
            rationales = json.loads(row["rationales_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid rationales_json for {case_id}.") from exc
        if new_label == 0:
            rationales.pop("security", None)
        else:
            rationales["security"] = rationale
        updated["rationales_json"] = json.dumps(
            rationales, ensure_ascii=False, sort_keys=True
        )
        label_rows.append(updated)
    return pl.DataFrame(target_rows).sort("case_id"), pl.DataFrame(
        label_rows, schema=base_labels.schema
    )


def _validate_full_review_provenance(
    row: Mapping[str, Any], *, pass_index: int
) -> None:
    if int(row.get("pass_index", -1)) != pass_index:
        raise ValueError(f"Review pass {pass_index} contains a mismatched pass_index.")
    if row.get("status") != "ok":
        raise ValueError(f"Review pass {pass_index} contains a non-ok result.")
    for field in ("response_id", "response_model"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Review pass {pass_index} result has invalid {field}.")
    usage = row.get("usage")
    if not isinstance(usage, dict) or not {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }.issubset(usage):
        raise ValueError(
            f"Review pass {pass_index} result has invalid usage provenance."
        )
    cost = row.get("estimated_cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        raise ValueError(
            f"Review pass {pass_index} result has invalid cost provenance."
        )
    if not math.isfinite(float(cost)) or float(cost) < 0:
        raise ValueError(
            f"Review pass {pass_index} result has invalid cost provenance."
        )


def write_full_corpus_security_target_artifacts(
    *,
    base_labels_path: str | Path,
    active_state_manifest_path: str | Path,
    review_pass_paths: Mapping[int, str | Path],
    tiebreak_results_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Validate all evidence before atomically publishing a full target."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite full-corpus Security target: {output}"
        )
    base_path = Path(base_labels_path)
    manifest_path = Path(active_state_manifest_path)
    tie_path = Path(tiebreak_results_path)
    base_labels = pl.read_parquet(base_path)
    if base_labels.height != EXPECTED_FULL_CORPUS_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_FULL_CORPUS_ROWS} base-label rows, "
            f"found {base_labels.height}."
        )
    target, labels = build_full_corpus_security_target(
        base_labels,
        active_state_manifest=read_jsonl(manifest_path),
        review_passes={
            index: read_jsonl(path) for index, path in review_pass_paths.items()
        },
        tiebreak_results=read_jsonl(tie_path) if tie_path.exists() else [],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        target_path = temporary / "security_target_variant.parquet"
        labels_path = temporary / "security_repaired_labels.parquet"
        summary_path = temporary / "target_summary.json"
        target.write_parquet(target_path)
        labels.write_parquet(labels_path)
        summary = {
            "artifact_scope": FULL_CORPUS_ARTIFACT_SCOPE,
            "case_count": target.height,
            "changed_label_count": target.filter(pl.col("label_changed")).height,
            "training_ready": True,
            "evaluation_ready": True,
            "base_labels_sha256": sha256_file(base_path),
            "manifest_sha256": sha256_file(manifest_path),
            "review_pass_sha256": {
                str(index): sha256_file(path)
                for index, path in review_pass_paths.items()
            },
            "tiebreak_sha256": sha256_file(tie_path) if tie_path.exists() else None,
            "target_variant_sha256": sha256_file(target_path),
            "repaired_labels_sha256": sha256_file(labels_path),
            "decision_method_counts": {
                row["decision_method"]: int(row["len"])
                for row in target.group_by("decision_method").len().to_dicts()
            },
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return (
        output / target_path.name,
        output / labels_path.name,
        output / summary_path.name,
    )


def _validate_manifest(
    active_state_manifest: Iterable[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    manifest_by_case: dict[str, Mapping[str, Any]] = {}
    for record in active_state_manifest:
        missing = REQUIRED_MANIFEST_FIELDS - set(record)
        if missing:
            raise ValueError(
                "Active-Critic-state manifest record is missing required fields: "
                f"{sorted(missing)}."
            )
        case_id = str(record["case_id"])
        if case_id in manifest_by_case:
            raise ValueError(
                f"Active-Critic-state manifest has duplicate case_id: {case_id}"
            )
        source_row = source_rows.get(case_id)
        if source_row is None:
            raise ValueError(
                "Active-Critic-state manifest includes unknown Security case: "
                f"{case_id}"
            )
        _validate_coordinate(record, source_row, source_name="manifest")
        _validate_contract(record)
        manifest_by_case[case_id] = record

    _validate_case_set(
        observed=set(manifest_by_case),
        expected=set(source_rows),
        source_name="Active-Critic-state manifest",
    )
    return manifest_by_case


def _validate_results(
    active_state_results: Iterable[Mapping[str, Any]],
    manifest_by_case: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    results_by_case: dict[str, dict[str, Any]] = {}
    for record in active_state_results:
        missing = REQUIRED_RESULT_FIELDS - set(record)
        if missing:
            raise ValueError(
                "Active-Critic-state result is missing required fields: "
                f"{sorted(missing)}."
            )
        case_id = str(record["case_id"])
        if case_id in results_by_case:
            raise ValueError(
                f"Active-Critic-state results have duplicate case_id: {case_id}"
            )
        manifest = manifest_by_case.get(case_id)
        if manifest is None:
            raise ValueError(
                f"Active-Critic-state results include unknown Security case: {case_id}"
            )
        for field in (
            "state_contract_version",
            "state_input_sha256",
            "prompt_sha256",
        ):
            if record[field] != manifest[field]:
                raise ValueError(
                    f"Active-Critic-state result {case_id} does not match "
                    f"manifest {field}."
                )
        reviewer = record["reviewer"]
        reviewed_at = record["reviewed_at"]
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"Active-Critic-state result {case_id} has no reviewer.")
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise ValueError(
                f"Active-Critic-state result {case_id} has no reviewed_at timestamp."
            )
        try:
            reviewed_at_value = datetime.fromisoformat(reviewed_at.strip())
        except ValueError as exc:
            raise ValueError(
                f"Active-Critic-state result {case_id} has invalid reviewed_at."
            ) from exc
        if reviewed_at_value.tzinfo is None or reviewed_at_value.utcoffset() is None:
            raise ValueError(
                f"Active-Critic-state result {case_id} reviewed_at must include "
                "a timezone."
            )
        confidence = record["confidence"]
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(
                f"Active-Critic-state result {case_id} has invalid confidence: "
                f"{confidence!r}"
            )
        rationale_status = record["rationale_status"]
        if rationale_status not in VALID_RATIONALE_STATUS:
            raise ValueError(
                "Active-Critic-state result "
                f"{case_id} has invalid rationale_status: {rationale_status!r}"
            )
        try:
            scores = AlignmentScores.model_validate(record["scores"])
        except Exception as exc:
            raise ValueError(
                f"Active-Critic-state result {case_id} has invalid scores."
            ) from exc
        rationales = record["rationales"]
        if not isinstance(rationales, dict):
            raise ValueError(
                f"Active-Critic-state result {case_id} has invalid rationales."
            )
        security_label = int(scores.security)
        security_rationale = rationales.get("security")
        if security_label == 0:
            if rationale_status != "not_applicable_neutral":
                raise ValueError(
                    f"Active-Critic-state neutral result {case_id} must use "
                    "rationale_status='not_applicable_neutral'."
                )
            rationale = "No non-neutral Security evidence in the reviewed active state."
        else:
            if rationale_status != "provided":
                raise ValueError(
                    f"Active-Critic-state non-neutral result {case_id} must "
                    "provide a rationale."
                )
            if (
                not isinstance(security_rationale, str)
                or not security_rationale.strip()
            ):
                raise ValueError(
                    f"Active-Critic-state non-neutral result {case_id} has no "
                    "Security rationale."
                )
            rationale = security_rationale.strip()
        results_by_case[case_id] = {
            "security_label": security_label,
            "reviewer": reviewer.strip(),
            "reviewed_at": reviewed_at.strip(),
            "confidence": confidence,
            "rationale_status": rationale_status,
            "rationale": rationale,
        }

    _validate_case_set(
        observed=set(results_by_case),
        expected=set(manifest_by_case),
        source_name="Active-Critic-state results",
    )
    return results_by_case


def _validate_coordinate(
    record: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    if record["dimension"] != "security":
        raise ValueError(
            f"Active-Critic-state {source_name} has non-Security dimension."
        )
    for field in ("persona_id", "t_index", "date"):
        if str(record[field]) != str(source_row[field]):
            raise ValueError(
                f"Active-Critic-state {source_name} coordinate mismatch for "
                f"{record['case_id']}: {field}."
            )


def _validate_contract(record: Mapping[str, Any]) -> None:
    if record["state_contract_version"] != ACTIVE_CRITIC_STATE_CONTRACT_VERSION:
        raise ValueError(
            "Active-Critic-state manifest has unsupported contract version: "
            f"{record['state_contract_version']!r}."
        )
    if (
        not isinstance(record["state_input_sha256"], str)
        or not record["state_input_sha256"]
    ):
        raise ValueError("Active-Critic-state manifest is missing state_input_sha256.")
    if not isinstance(record["prompt_sha256"], str) or not record["prompt_sha256"]:
        raise ValueError("Active-Critic-state manifest is missing prompt_sha256.")
    flags = record["context_flags"]
    if flags != EXPECTED_CONTEXT_FLAGS:
        raise ValueError(
            "Active-Critic-state manifest context flags do not match the exact "
            "window_size=1 contract."
        )
    state_input = record["state_input"]
    if not isinstance(state_input, dict) or set(state_input) != {
        "window_size",
        "session_content",
        "profile_weights",
    }:
        raise ValueError(
            "Active-Critic-state manifest state_input must contain only "
            "window_size, session_content, and profile_weights."
        )
    if state_input["window_size"] != 1:
        raise ValueError("Active-Critic-state manifest must use window_size=1.")
    if (
        not isinstance(state_input["session_content"], str)
        or not state_input["session_content"]
    ):
        raise ValueError("Active-Critic-state manifest has no runtime session text.")
    profile_weights = state_input["profile_weights"]
    if not isinstance(profile_weights, dict) or set(profile_weights) != set(
        AlignmentScores.model_fields
    ):
        raise ValueError(
            "Active-Critic-state manifest must contain the canonical 10-value profile."
        )
    numeric_weights = []
    for dimension, value in profile_weights.items():
        if isinstance(value, bool):
            raise ValueError(
                f"Active-Critic-state profile weight {dimension} must be numeric."
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Active-Critic-state profile weight {dimension} must be numeric."
            ) from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(
                f"Active-Critic-state profile weight {dimension} is out of range."
            )
        numeric_weights.append(numeric)
    if not math.isclose(sum(numeric_weights), 1.0, abs_tol=1e-6):
        raise ValueError("Active-Critic-state profile weights must sum to 1.0.")
    prompt = record["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Active-Critic-state manifest has no rendered prompt.")
    if sha256_canonical_json(state_input) != record["state_input_sha256"]:
        raise ValueError("Active-Critic-state manifest state_input_sha256 mismatch.")
    if sha256_canonical_json({"prompt": prompt}) != record["prompt_sha256"]:
        raise ValueError("Active-Critic-state manifest prompt_sha256 mismatch.")
    expected_prompt = render_active_critic_state_prompt(
        session_content=state_input["session_content"],
        profile_weights=[
            profile_weights[dimension] for dimension in SCHWARTZ_VALUE_ORDER
        ],
        schwartz_config=load_schwartz_values(SCHWARTZ_CONFIG_PATH),
    )
    if prompt != expected_prompt:
        raise ValueError(
            "Active-Critic-state manifest prompt is not the canonical rendering "
            "of state_input."
        )


def _validate_case_set(
    *,
    observed: set[str],
    expected: set[str],
    source_name: str,
) -> None:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{source_name} case set does not match the selected Security subset. "
            f"Missing={missing[:5]}, extra={extra[:5]}"
        )


def _as_alignment_label(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be one of -1, 0, or 1, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be one of -1, 0, or 1.") from exc
    if numeric not in {-1.0, 0.0, 1.0}:
        raise ValueError(f"{field} must be one of -1, 0, or 1, got {value!r}.")
    return int(numeric)
