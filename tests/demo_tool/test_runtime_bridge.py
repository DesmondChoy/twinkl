import json
from pathlib import Path

import polars as pl

from src.demo_tool.runtime_bridge import (
    build_output_dir,
    discover_checkpoints,
    load_cached_run,
    run_demo_pipeline,
)


def test_discover_checkpoints_reads_local_candidates(tmp_path: Path):
    artifacts_root = tmp_path / "logs" / "experiments" / "artifacts"
    checkpoint_dir = artifacts_root / "run_020_BalancedSoftmax" / "BalancedSoftmax"
    checkpoint_dir.mkdir(parents=True)

    checkpoint_path = checkpoint_dir / "selected_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    (checkpoint_dir / "selection_summary.yaml").write_text(
        """selected_candidate:
  qwk_mean: 0.402
  recall_minus1: 0.427
"""
    )

    options = discover_checkpoints(search_roots=(artifacts_root,))

    assert len(options) == 1
    assert options[0].path == str(checkpoint_path.resolve())
    assert "run_020" in options[0].label
    assert options[0].metrics_summary is not None
    assert options[0].metrics_summary["qwk_mean"] == 0.402


def test_run_demo_pipeline_returns_loaded_artifacts(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "exports"
    checkpoint_path = tmp_path / "selected_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    timeline_df = pl.DataFrame(
        [
            {
                "persona_id": "deadbeef",
                "date": "2025-01-06",
                "t_index": 0,
                "initial_entry": "Worked late.",
                "overall_mean": 0.15,
                "overall_uncertainty": 0.1,
                **{f"alignment_{dim}": 0.0 for dim in [
                    "self_direction",
                    "stimulation",
                    "hedonism",
                    "achievement",
                    "power",
                    "security",
                    "conformity",
                    "tradition",
                    "benevolence",
                    "universalism",
                ]},
            }
        ]
    )
    weekly_df = pl.DataFrame(
        [
            {
                "persona_id": "deadbeef",
                "week_start": "2025-01-06",
                "week_end": "2025-01-12",
                "n_entries": 1,
                "overall_mean": 0.15,
                "overall_uncertainty": 0.1,
                **{f"alignment_{dim}": 0.0 for dim in [
                    "self_direction",
                    "stimulation",
                    "hedonism",
                    "achievement",
                    "power",
                    "security",
                    "conformity",
                    "tradition",
                    "benevolence",
                    "universalism",
                ]},
            }
        ]
    )

    def _fake_run_weekly_coach_cycle(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = output_dir / "deadbeef_vif_timeline.parquet"
        weekly_path = output_dir / "deadbeef_vif_weekly.parquet"
        drift_json_path = output_dir / "deadbeef_2025-01-06_2025-01-12.drift.json"
        digest_json_path = output_dir / "deadbeef_2025-01-06_2025-01-12.json"
        digest_md_path = output_dir / "deadbeef_2025-01-06_2025-01-12.md"
        prompt_path = output_dir / "deadbeef_2025-01-06_2025-01-12.prompt.txt"

        timeline_df.write_parquet(timeline_path)
        weekly_df.write_parquet(weekly_path)
        drift_json_path.write_text(
            json.dumps(
                {
                    "response_mode": "stable",
                    "trigger_type": None,
                    "rationale": "No drift.",
                    "dimension_signals": [],
                    "triggered_dimensions": [],
                }
            )
        )
        digest_json_path.write_text(
            json.dumps(
                {
                    "persona_id": "deadbeef",
                    "response_mode": "stable",
                    "overall_mean": 0.15,
                    "overall_uncertainty": 0.1,
                    "top_tensions": [],
                    "top_strengths": ["benevolence"],
                }
            )
        )
        digest_md_path.write_text("# Digest")
        prompt_path.write_text("Prompt")

        class _Digest:
            def model_dump(self):
                return {"persona_id": "deadbeef", "response_mode": "stable"}

        return _Digest(), {
            "timeline_path": str(timeline_path),
            "weekly_path": str(weekly_path),
            "drift_json_path": str(drift_json_path),
            "digest_json_path": str(digest_json_path),
            "digest_md_path": str(digest_md_path),
            "prompt_path": str(prompt_path),
            "parquet_path": str(tmp_path / "weekly_digests.parquet"),
        }

    monkeypatch.setattr(
        "src.demo_tool.runtime_bridge.run_weekly_coach_cycle",
        _fake_run_weekly_coach_cycle,
    )

    bundle = run_demo_pipeline(
        persona_id="deadbeef",
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        parquet_path=tmp_path / "weekly_digests.parquet",
    )

    assert bundle["persona_id"] == "deadbeef"
    assert bundle["checkpoint_path"] == str(checkpoint_path.resolve())
    assert bundle["artifacts"]["digest_payload"]["response_mode"] == "stable"
    assert bundle["artifacts"]["timeline_df"].height == 1

    cached = load_cached_run(
        "deadbeef",
        checkpoint_path,
        output_root=output_root,
        parquet_path=tmp_path / "weekly_digests.parquet",
    )
    assert cached is not None
    assert Path(cached["artifact_paths"]["timeline_path"]).exists()
    assert cached["artifacts"]["weekly_df"].height == 1

    assert build_output_dir("deadbeef", checkpoint_path, output_root=output_root).exists()
