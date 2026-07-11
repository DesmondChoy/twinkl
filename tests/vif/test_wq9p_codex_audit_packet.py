"""Integration checks for the published twinkl-16ar audit packet."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / (
    "logs/experiments/artifacts/drift_trigger_benchmark_twinkl_wq9p_20260710/"
    "codex_audit"
)
PACKET_BUILDER = REPO_ROOT / "scripts/experiments/build_wq9p_codex_audit_packet.py"
ASSESSMENT_FILENAMES = (
    "blind_assessment.json",
    "blind_assessment_check.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_manifest_hashes(manifest: dict[str, object], artifact_root: Path) -> None:
    generator = manifest["generator"]
    assert isinstance(generator, dict)
    assert generator["path"] == "scripts/experiments/build_wq9p_codex_audit_packet.py"
    assert generator["sha256"] == _sha256(PACKET_BUILDER)

    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    assert outputs["blind_packet.json"] == _sha256(
        artifact_root / "blind_packet.json"
    )
    assert outputs["reconciliation_key.json"] == _sha256(
        artifact_root / "reconciliation_key.json"
    )
    assessment_hashes = outputs["assessment_sha256"]
    assert isinstance(assessment_hashes, dict)
    assert assessment_hashes == {
        filename: _sha256(artifact_root / filename)
        for filename in ASSESSMENT_FILENAMES
    }

    source_inputs = manifest["source_inputs"]
    assert isinstance(source_inputs, dict)
    for source_name in ("reference_episodes", "designed_holdout"):
        source = source_inputs[source_name]
        assert isinstance(source, dict)
        assert source["sha256"] == _sha256(REPO_ROOT / source["path"])
    frozen_files = source_inputs["frozen_wrangled_files"]
    assert isinstance(frozen_files, dict)
    source_files = frozen_files["files"]
    assert isinstance(source_files, list)
    assert len(source_files) == 4
    assert all(
        source["sha256"] == _sha256(REPO_ROOT / source["path"])
        for source in source_files
    )


def test_packet_builder_reproduces_published_packet_and_manifest(
    tmp_path: Path,
) -> None:
    """The committed packet/key remain reproducible and provenance is complete."""
    for filename in ASSESSMENT_FILENAMES:
        shutil.copy2(AUDIT_ROOT / filename, tmp_path / filename)

    subprocess.run(
        [sys.executable, str(PACKET_BUILDER), "--output-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for filename in ("blind_packet.json", "reconciliation_key.json"):
        assert (tmp_path / filename).read_bytes() == (
            AUDIT_ROOT / filename
        ).read_bytes()

    packet = json.loads((tmp_path / "blind_packet.json").read_text())
    key_rows = json.loads((tmp_path / "reconciliation_key.json").read_text())["cases"]
    manifest = json.loads((tmp_path / "audit_manifest.json").read_text())
    published_manifest = json.loads((AUDIT_ROOT / "audit_manifest.json").read_text())

    assert len(packet["cases"]) == 25
    assert sum(len(case["entries"]) for case in packet["cases"]) == 74
    assert all(
        set(case) == {"review_case_id", "declared_core_value", "entries"}
        for case in packet["cases"]
    )
    assert all(
        set(entry) == {"journal_entry"}
        for case in packet["cases"]
        for entry in case["entries"]
    )
    assert {row["case_category"] for row in key_rows} == {
        "frozen_consensus",
        "designed_positive",
        "designed_control",
    }

    _assert_manifest_hashes(manifest, tmp_path)
    _assert_manifest_hashes(published_manifest, AUDIT_ROOT)
    assert manifest["review_protocol"]["technical_isolation"] is False
    assert published_manifest["review_protocol"]["technical_isolation"] is False
