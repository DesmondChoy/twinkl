"""Regression checks that keep the invalid frozen benchmark out of active use."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = (
    ROOT
    / "logs/experiments/artifacts"
    / ("drift_trigger_benchmark_twinkl_wq9p_20260710")
)


def _active_text_files() -> list[Path]:
    roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "config",
        ROOT / "docs",
        ROOT / "logs/experiments/index.md",
        ROOT / "README.md",
    ]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "docs/archive" not in str(path)
            and path.suffix in {".md", ".py", ".yaml", ".yml", ".json"}
        )
    return files


def test_legacy_frozen_benchmark_has_no_runnable_surface():
    assert not LEGACY_ROOT.exists()
    assert not (ROOT / "scripts/experiments/drift_trigger_benchmark.py").exists()
    assert not (ROOT / "scripts/experiments/build_wq9p_codex_audit_packet.py").exists()
    assert not (
        ROOT / "logs/experiments/reports/experiment_review_2026-07-10_twinkl_wq9p.md"
    ).exists()


def test_active_paths_do_not_name_the_retired_frozen_benchmark():
    retired_markers = {
        "reference_episodes_test.parquet",
        "drift_trigger_benchmark_twinkl_wq9p_20260710",
        "build_wq9p_codex_audit_packet.py",
    }
    matches = []
    for path in _active_text_files():
        content = path.read_text(encoding="utf-8")
        for marker in retired_markers:
            if marker in content:
                matches.append(f"{path.relative_to(ROOT)}: {marker}")
    assert not matches, "Retired frozen benchmark remains referenced:\n" + "\n".join(
        matches
    )
