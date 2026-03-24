"""Runtime orchestration for the weekly VIF -> drift -> Coach flow."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.coach.weekly_digest import (
    _default_output_stem,
    attach_coach_artifacts,
    build_weekly_digest,
    generate_weekly_digest_coach,
    persist_weekly_digest_record,
    render_digest_markdown,
    render_digest_prompt,
    validate_weekly_digest_narrative,
)
from src.vif.drift import detect_weekly_drift
from src.vif.runtime import (
    aggregate_timeline_by_week,
    persist_runtime_artifacts,
    predict_persona_timeline,
)


def run_weekly_coach_cycle(
    *,
    persona_id: str,
    checkpoint_path: str | Path,
    wrangled_dir: str | Path = "logs/wrangled",
    config_path: str | Path | None = "config/vif.yaml",
    output_dir: str | Path = "logs/exports/weekly_coach",
    parquet_path: str | Path = "logs/exports/weekly_digests/weekly_digests.parquet",
    start_date: str | None = None,
    end_date: str | None = None,
    n_mc_samples: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
    llm_complete=None,
) -> tuple[object, dict[str, str]]:
    """Run the full weekly Coach path from VIF checkpoint to digest artifacts."""
    timeline_df, _metadata = predict_persona_timeline(
        persona_id=persona_id,
        checkpoint_path=checkpoint_path,
        wrangled_dir=wrangled_dir,
        config_path=config_path,
        n_mc_samples=n_mc_samples,
        batch_size=batch_size,
        device=device,
    )
    weekly_df = aggregate_timeline_by_week(timeline_df)
    drift_result = detect_weekly_drift(
        weekly_df,
        target_week_end=end_date,
    )

    digest = build_weekly_digest(
        persona_id=persona_id,
        labels_path=None,
        wrangled_dir=Path(wrangled_dir),
        signals_df=timeline_df,
        start_date=start_date,
        end_date=end_date,
        drift_result=drift_result,
    )

    if llm_complete is not None:
        narrative, _prompt = asyncio.run(generate_weekly_digest_coach(digest, llm_complete))
        validation = (
            validate_weekly_digest_narrative(digest, narrative)
            if narrative is not None
            else None
        )
        digest = attach_coach_artifacts(digest, narrative, validation)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = _default_output_stem(digest)

    signal_artifacts = persist_runtime_artifacts(
        timeline_df=timeline_df,
        weekly_df=weekly_df,
        output_dir=output_path,
        timeline_filename=f"{persona_id}_vif_timeline.parquet",
        weekly_filename=f"{persona_id}_vif_weekly.parquet",
    )

    drift_json_path = output_path / f"{stem}.drift.json"
    digest_json_path = output_path / f"{stem}.json"
    digest_md_path = output_path / f"{stem}.md"
    prompt_path = output_path / f"{stem}.prompt.txt"

    drift_json_path.write_text(json.dumps(drift_result.model_dump(), indent=2) + "\n")
    digest_json_path.write_text(json.dumps(digest.model_dump(), indent=2) + "\n")
    digest_md_path.write_text(render_digest_markdown(digest))
    prompt_path.write_text(render_digest_prompt(digest))
    persist_weekly_digest_record(digest, Path(parquet_path))

    return digest, {
        **signal_artifacts,
        "drift_json_path": str(drift_json_path),
        "digest_json_path": str(digest_json_path),
        "digest_md_path": str(digest_md_path),
        "prompt_path": str(prompt_path),
        "parquet_path": str(parquet_path),
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the weekly VIF-to-Coach pipeline.")
    parser.add_argument("--persona-id", required=True, help="Persona ID (without prefix).")
    parser.add_argument("--checkpoint-path", required=True, help="Path to trained Critic checkpoint.")
    parser.add_argument("--wrangled-dir", default="logs/wrangled")
    parser.add_argument("--config-path", default="config/vif.yaml")
    parser.add_argument("--output-dir", default="logs/exports/weekly_coach")
    parser.add_argument(
        "--parquet-path",
        default="logs/exports/weekly_digests/weekly_digests.parquet",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--n-mc-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    digest, artifact_paths = run_weekly_coach_cycle(
        persona_id=args.persona_id,
        checkpoint_path=args.checkpoint_path,
        wrangled_dir=args.wrangled_dir,
        config_path=args.config_path,
        output_dir=args.output_dir,
        parquet_path=args.parquet_path,
        start_date=args.start_date,
        end_date=args.end_date,
        n_mc_samples=args.n_mc_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        f"Built weekly coach digest for {digest.persona_id} "
        f"({digest.week_start} -> {digest.week_end})"
    )
    for name, path in artifact_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
