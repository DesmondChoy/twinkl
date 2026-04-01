#!/usr/bin/env python3
"""Prepare the twinkl-754 consensus re-judging bundle.

This script renders the `profile_only` judge prompt for every labeled entry,
writes five identical pass prompt files, and deterministically shards each pass
for short-lived worker sub-agents.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation_tool.data_loader import load_entries
from src.judge.consensus_utils import write_bundle_status, write_jsonl
from src.judge.labeling import (
    build_session_content,
    load_schwartz_values,
    render_judge_prompt,
)
from src.models.judge import SCHWARTZ_VALUE_ORDER

N_PASSES = 5
EXPECTED_TOTAL_ENTRIES = 1651
MAX_SHARD_PERSONAS = 5
MAX_SHARD_ENTRIES = 24
DEFAULT_PILOT_HARD_DIMENSIONS = ("security", "hedonism", "stimulation")


def _entry_id(persona_id: str, t_index: int) -> str:
    return f"{persona_id}__{t_index}"


def _reset_bundle_output(output_dir: Path) -> None:
    """Remove previously generated bundle artifacts before rewriting them."""
    for directory_name in ("prompts", "shards", "results", "provenance"):
        target = output_dir / directory_name
        if target.exists():
            shutil.rmtree(target)

    for filename in (
        "manifest.csv",
        "shard_manifest.csv",
        "README.md",
        "comparison_rows.csv",
        "confidence_summary.csv",
        "consensus_rejudging_report.md",
        "flip_summary.csv",
        "irr_summary.csv",
        "joined_results.csv",
        "bundle_status.json",
    ):
        target = output_dir / filename
        if target.exists():
            target.unlink()


def _persona_round_robin_rows(rows: list[dict]) -> list[dict]:
    persona_rows: OrderedDict[str, list[dict]] = OrderedDict()
    for row in sorted(rows, key=lambda item: (item["persona_id"], item["t_index"])):
        persona_rows.setdefault(row["persona_id"], []).append(row)

    ordered_rows: list[dict] = []
    while True:
        progressed = False
        for persona_id in list(persona_rows.keys()):
            if not persona_rows[persona_id]:
                continue
            ordered_rows.append(persona_rows[persona_id].pop(0))
            progressed = True
        if not progressed:
            break
    return ordered_rows


def _select_pilot_entries(
    joined: pl.DataFrame,
    *,
    pilot_size: int,
    hard_dimensions: tuple[str, ...],
) -> tuple[pl.DataFrame, dict]:
    if pilot_size <= 0:
        raise ValueError("Pilot size must be greater than zero.")

    ordered_rows = joined.sort(["persona_id", "t_index"]).to_dicts()
    if pilot_size >= len(ordered_rows):
        selected_rows = ordered_rows
    else:
        candidate_lists = {
            value_name: _persona_round_robin_rows(
                [
                    row
                    for row in ordered_rows
                    if int(row[f"alignment_{value_name}"]) != 0
                ]
            )
            for value_name in hard_dimensions
        }
        candidate_indices = {value_name: 0 for value_name in hard_dimensions}
        selected_lookup: OrderedDict[str, dict] = OrderedDict()

        while len(selected_lookup) < pilot_size:
            progressed = False
            for value_name in hard_dimensions:
                candidates = candidate_lists[value_name]
                while candidate_indices[value_name] < len(candidates):
                    candidate = candidates[candidate_indices[value_name]]
                    candidate_indices[value_name] += 1
                    if candidate["entry_id"] in selected_lookup:
                        continue
                    selected_lookup[candidate["entry_id"]] = candidate
                    progressed = True
                    break
                if len(selected_lookup) >= pilot_size:
                    break
            if not progressed:
                break

        if len(selected_lookup) < pilot_size:
            for row in _persona_round_robin_rows(ordered_rows):
                if row["entry_id"] in selected_lookup:
                    continue
                selected_lookup[row["entry_id"]] = row
                if len(selected_lookup) >= pilot_size:
                    break

        selected_rows = sorted(
            selected_lookup.values(),
            key=lambda row: (row["persona_id"], int(row["t_index"])),
        )

    if len(selected_rows) != min(pilot_size, len(ordered_rows)):
        raise ValueError(
            f"Pilot selection expected {min(pilot_size, len(ordered_rows))} entries "
            f"but selected {len(selected_rows)}."
        )

    selection_counts = {
        value_name: sum(
            int(row[f"alignment_{value_name}"]) != 0 for row in selected_rows
        )
        for value_name in hard_dimensions
    }
    candidate_counts = {
        value_name: int(
            joined.filter(pl.col(f"alignment_{value_name}") != 0).height
        )
        for value_name in hard_dimensions
    }
    selected = pl.DataFrame(selected_rows).sort(["persona_id", "t_index"])
    return selected, {
        "requested_size": pilot_size,
        "selected_size": selected.height,
        "hard_dimensions": list(hard_dimensions),
        "candidate_non_zero_counts": candidate_counts,
        "selected_non_zero_counts": selection_counts,
    }


def _build_entry_maps(
    entries: pl.DataFrame,
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], int]]:
    entry_lookup: dict[tuple[str, int], dict] = {}
    previous_entry_counts: dict[tuple[str, int], int] = {}
    seen_counts_by_persona: dict[str, int] = {}
    for row in entries.sort(["persona_id", "t_index"]).to_dicts():
        persona_id = row["persona_id"]
        t_index = int(row["t_index"])
        key = (persona_id, t_index)
        entry_lookup[key] = row
        previous_entry_counts[key] = seen_counts_by_persona.get(persona_id, 0)
        seen_counts_by_persona[persona_id] = previous_entry_counts[key] + 1
    return entry_lookup, previous_entry_counts


def _persisted_scores_from_row(row: dict) -> dict[str, int]:
    return {
        value_name: int(row[f"alignment_{value_name}"])
        for value_name in SCHWARTZ_VALUE_ORDER
    }


def _load_entry_universe(
    *,
    labels_path: Path,
    wrangled_dir: Path,
    expected_total_entries: int = EXPECTED_TOTAL_ENTRIES,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    labels = pl.read_parquet(labels_path).sort(["persona_id", "t_index"])
    entries = load_entries(wrangled_dir).sort(["persona_id", "t_index"])

    joined = labels.join(entries, on=["persona_id", "t_index", "date"], how="inner")
    if joined.height != expected_total_entries:
        raise ValueError(
            f"Expected exactly {expected_total_entries} joined entries, "
            f"found {joined.height}."
        )
    if joined.height != labels.height or joined.height != entries.height:
        raise ValueError(
            "Consensus prep expects labels and wrangled entries to align 1:1."
        )

    joined = joined.with_columns(
        pl.struct(["persona_id", "t_index"])
        .map_elements(
            lambda row: _entry_id(row["persona_id"], int(row["t_index"])),
            return_dtype=pl.Utf8,
        )
        .alias("entry_id")
    )

    duplicates = joined.group_by("entry_id").len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError("Detected duplicate entry_id values in the consensus bundle.")

    return joined, entries


def _render_prompt_record(
    row: dict,
    *,
    entry_lookup: dict[tuple[str, int], dict],
    previous_entry_counts: dict[tuple[str, int], int],
    schwartz_config: dict,
) -> dict:
    entry_key = (row["persona_id"], int(row["t_index"]))
    entry = entry_lookup[entry_key]
    session_content = build_session_content(
        entry["initial_entry"],
        entry["nudge_text"],
        entry["response_text"],
    )
    prompt = render_judge_prompt(
        session_content=session_content,
        entry_date=entry["date"],
        persona_name=entry.get("persona_name") or "",
        persona_age=entry.get("persona_age") or "",
        persona_profession=entry.get("persona_profession") or "",
        persona_culture=entry.get("persona_culture") or "",
        persona_core_values=list(entry.get("persona_core_values") or []),
        persona_bio="",
        schwartz_config=schwartz_config,
        previous_entries=None,
    )

    return {
        "entry_id": row["entry_id"],
        "persona_id": row["persona_id"],
        "t_index": int(row["t_index"]),
        "date": row["date"],
        "persisted_scores": _persisted_scores_from_row(row),
        "prompt": prompt,
        "context_flags": {
            "bio_included": False,
            "previous_entries_included": False,
            "core_values_included": True,
        },
        "context_stats": {
            "previous_entries_count": previous_entry_counts[entry_key],
            "current_session_chars": len(session_content),
            "prompt_chars": len(prompt),
        },
    }


def build_manifest_frame(
    joined: pl.DataFrame,
    *,
    schwartz_config: dict,
    entry_lookup: dict[tuple[str, int], dict],
    previous_entry_counts: dict[tuple[str, int], int],
) -> tuple[pl.DataFrame, list[dict]]:
    prompt_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for row in joined.sort(["persona_id", "t_index"]).to_dicts():
        prompt_row = _render_prompt_record(
            row,
            entry_lookup=entry_lookup,
            previous_entry_counts=previous_entry_counts,
            schwartz_config=schwartz_config,
        )
        prompt_rows.append(prompt_row)
        manifest_row = {
            "entry_id": row["entry_id"],
            "persona_id": row["persona_id"],
            "t_index": int(row["t_index"]),
            "date": row["date"],
            "persona_name": row.get("persona_name") or "",
            "persona_age": row.get("persona_age") or "",
            "persona_profession": row.get("persona_profession") or "",
            "persona_culture": row.get("persona_culture") or "",
            "session_chars": prompt_row["context_stats"]["current_session_chars"],
            "prompt_chars": prompt_row["context_stats"]["prompt_chars"],
        }
        for value_name in SCHWARTZ_VALUE_ORDER:
            manifest_row[f"alignment_{value_name}"] = int(
                row[f"alignment_{value_name}"]
            )
        manifest_rows.append(manifest_row)

    return pl.DataFrame(manifest_rows), prompt_rows


def build_shards(
    prompt_rows: list[dict],
    *,
    pass_index: int,
    max_personas: int = MAX_SHARD_PERSONAS,
    max_entries: int = MAX_SHARD_ENTRIES,
) -> list[dict]:
    persona_rows: OrderedDict[str, list[dict]] = OrderedDict()
    for row in sorted(prompt_rows, key=lambda item: (item["persona_id"], item["t_index"])):
        persona_rows.setdefault(row["persona_id"], []).append(row)

    shards: list[dict] = []
    current_rows: list[dict] = []
    current_personas: list[str] = []

    def flush_current() -> None:
        if not current_rows:
            return
        shard_index = len(shards) + 1
        shard_id = f"pass_{pass_index}_shard_{shard_index:03d}"
        rows = []
        for row in current_rows:
            shard_row = dict(row)
            shard_row["pass_index"] = pass_index
            shard_row["shard_id"] = shard_id
            rows.append(shard_row)
        shards.append(
            {
                "shard_id": shard_id,
                "pass_index": pass_index,
                "persona_ids": list(current_personas),
                "rows": rows,
            }
        )

    for persona_id, rows in persona_rows.items():
        persona_count = len(rows)
        would_exceed_personas = len(current_personas) >= max_personas
        would_exceed_entries = current_rows and len(current_rows) + persona_count > max_entries
        if would_exceed_personas or would_exceed_entries:
            flush_current()
            current_rows = []
            current_personas = []

        current_rows.extend(rows)
        current_personas.append(persona_id)

    flush_current()
    return shards


def _write_readme(
    output_dir: Path,
    shard_manifest: pl.DataFrame,
    *,
    bundle_status: dict,
) -> None:
    n_shards = shard_manifest.select("shard_id").n_unique()
    bundle_mode = str(bundle_status.get("bundle_mode", "full"))
    selected_entry_count = int(bundle_status.get("selected_entry_count", 0))
    selection_summary = bundle_status.get("selection_summary") or {}
    lines = [
        "# twinkl-754 Consensus Re-judging Bundle",
        "",
        "This bundle was prepared by `scripts/journalling/twinkl_754_prepare_consensus.py`.",
        "",
        "## Scope",
        "",
        f"- Bundle mode: `{bundle_mode}`",
        f"- Selected entries: `{selected_entry_count}`",
        f"- Prompt condition: `{bundle_status.get('prompt_condition', 'profile_only')}`",
        "",
        "## Files",
        "",
        "- `manifest.csv`: the selected entries with deterministic `entry_id` keys and persisted labels.",
        "- `shard_manifest.csv`: shard-level execution plan for all 5 passes.",
        "- `bundle_status.json`: bundle lifecycle state, selection mode, and operator warnings.",
        "- `prompts/pass_<n>.jsonl`: full pass prompt files.",
        "- `shards/pass_<n>/shard_*.jsonl`: persona-preserving worker shards.",
        "- `results/pass_<n>_results.jsonl`: merged per-pass result placeholders.",
        "- `results/pass_<n>/shards/shard_*_results.jsonl`: shard result placeholders.",
        "- `provenance/shard_provenance.csv`: accepted shard validation and hash records.",
        "- `provenance/pass_provenance.csv`: merged pass fingerprints and rationale coverage.",
        "- `provenance/pass_similarity.csv`: pairwise pass similarity diagnostics.",
        "",
        "## Worker Model",
        "",
        "- Main agent: orchestrator only",
        "- Worker sub-agents: `gpt-5.4`, `fork_context=false`, one worker per shard",
        "- Suggested concurrency: up to 10 workers per wave",
        "",
        "## Shard Policy",
        "",
        f"- Max personas per shard: `{MAX_SHARD_PERSONAS}`",
        f"- Max entries per shard: `{MAX_SHARD_ENTRIES}`",
        f"- Total shards across all passes: `{n_shards}`",
        "",
        "## Validation",
        "",
        "Validate each shard before merging it into a pass file:",
        "",
        "```bash",
        "source .venv/bin/activate",
        "python scripts/journalling/twinkl_754_validate_results.py \\",
        "  --manifest logs/exports/twinkl_754/manifest.csv \\",
        "  --expected-jsonl logs/exports/twinkl_754/shards/pass_1/pass_1_shard_001.jsonl \\",
        "  --results logs/exports/twinkl_754/results/pass_1/shards/pass_1_shard_001_results.jsonl",
        "```",
        "",
        "After all validated shard files are ready, merge them and persist provenance:",
        "",
        "```bash",
        "source .venv/bin/activate",
        "python scripts/journalling/twinkl_754_merge_pass_results.py \\",
        "  --bundle-dir logs/exports/twinkl_754 \\",
        "  --worker-model gpt-5.4",
        "```",
        "",
        "Only after the merge step succeeds should you run `twinkl_754_summarize_consensus.py`.",
        "",
    ]
    if bundle_mode == "pilot":
        selected_counts = selection_summary.get("selected_non_zero_counts") or {}
        lines.extend(
            [
                "## Pilot Selection",
                "",
                f"- Requested entries: `{selection_summary.get('requested_size', selected_entry_count)}`",
                f"- Selected entries: `{selection_summary.get('selected_size', selected_entry_count)}`",
            ]
        )
        for value_name, count in selected_counts.items():
            lines.append(
                f"- Selected non-zero `{value_name}` entries: `{int(count)}`"
            )
        lines.append("")
    (output_dir / "README.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def prepare_consensus_bundle(
    *,
    output_dir: Path,
    labels_path: Path,
    wrangled_dir: Path,
    schwartz_path: Path,
    expected_total_entries: int = EXPECTED_TOTAL_ENTRIES,
    max_shard_personas: int = MAX_SHARD_PERSONAS,
    max_shard_entries: int = MAX_SHARD_ENTRIES,
    pilot_size: int | None = None,
    pilot_hard_dimensions: tuple[str, ...] = DEFAULT_PILOT_HARD_DIMENSIONS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    joined, entries = _load_entry_universe(
        labels_path=labels_path,
        wrangled_dir=wrangled_dir,
        expected_total_entries=expected_total_entries,
    )
    selection_summary: dict | None = None
    selected_joined = joined
    if pilot_size is not None:
        if not pilot_hard_dimensions:
            raise ValueError("Pilot selection requires at least one hard dimension.")
        selected_joined, selection_summary = _select_pilot_entries(
            joined,
            pilot_size=pilot_size,
            hard_dimensions=pilot_hard_dimensions,
        )

    schwartz_config = load_schwartz_values(schwartz_path)
    entry_lookup, previous_entry_counts = _build_entry_maps(entries)
    manifest, prompt_rows = build_manifest_frame(
        selected_joined,
        schwartz_config=schwartz_config,
        entry_lookup=entry_lookup,
        previous_entry_counts=previous_entry_counts,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_bundle_output(output_dir)
    manifest.write_csv(output_dir / "manifest.csv")

    prompts_dir = output_dir / "prompts"
    shards_root = output_dir / "shards"
    results_root = output_dir / "results"
    shard_manifest_rows: list[dict] = []

    for pass_index in range(1, N_PASSES + 1):
        pass_name = f"pass_{pass_index}"
        pass_prompt_path = prompts_dir / f"{pass_name}.jsonl"
        write_jsonl(pass_prompt_path, prompt_rows)

        pass_results_dir = results_root / pass_name / "shards"
        pass_results_dir.mkdir(parents=True, exist_ok=True)
        (results_root / f"{pass_name}_results.jsonl").write_text("", encoding="utf-8")

        shards = build_shards(
            prompt_rows,
            pass_index=pass_index,
            max_personas=max_shard_personas,
            max_entries=max_shard_entries,
        )
        pass_shards_dir = shards_root / pass_name
        for shard in shards:
            shard_prompt_path = pass_shards_dir / f"{shard['shard_id']}.jsonl"
            shard_result_path = pass_results_dir / f"{shard['shard_id']}_results.jsonl"
            write_jsonl(shard_prompt_path, shard["rows"])
            shard_result_path.write_text("", encoding="utf-8")
            shard_manifest_rows.append(
                {
                    "pass_index": pass_index,
                    "pass_name": pass_name,
                    "shard_id": shard["shard_id"],
                    "prompt_path": str(shard_prompt_path),
                    "result_path": str(shard_result_path),
                    "n_entries": len(shard["rows"]),
                    "n_personas": len(shard["persona_ids"]),
                    "first_entry_id": shard["rows"][0]["entry_id"],
                    "last_entry_id": shard["rows"][-1]["entry_id"],
                    "persona_ids_json": json.dumps(shard["persona_ids"], ensure_ascii=True),
                }
            )

    shard_manifest = pl.DataFrame(shard_manifest_rows).sort(
        ["pass_index", "shard_id"]
    )
    shard_manifest.write_csv(output_dir / "shard_manifest.csv")
    bundle_status = {
        "status": "prepared",
        "invalidated": False,
        "warning": "",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "prompt_condition": "profile_only",
        "bundle_mode": "pilot" if pilot_size is not None else "full",
        "selected_entry_count": manifest.height,
        "expected_total_entries": expected_total_entries,
        "selection_summary": selection_summary,
    }
    write_bundle_status(output_dir, bundle_status)
    _write_readme(output_dir, shard_manifest, bundle_status=bundle_status)
    return manifest, shard_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the twinkl-754 consensus re-judging bundle."
    )
    parser.add_argument(
        "--output-dir",
        default="logs/exports/twinkl_754",
        help="Directory to write the consensus bundle.",
    )
    parser.add_argument(
        "--labels-path",
        default="logs/judge_labels/judge_labels.parquet",
        help="Path to the persisted judge labels parquet.",
    )
    parser.add_argument(
        "--wrangled-dir",
        default="logs/wrangled",
        help="Directory containing wrangled persona markdown files.",
    )
    parser.add_argument(
        "--schwartz-path",
        default="config/schwartz_values.yaml",
        help="Path to the Schwartz value config used to render prompts.",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=None,
        help=(
            "Optional deterministic pilot size. When provided, the bundle is "
            "restricted to a persona-stratified subset that oversamples the "
            "requested hard dimensions."
        ),
    )
    parser.add_argument(
        "--pilot-hard-dimensions",
        default=",".join(DEFAULT_PILOT_HARD_DIMENSIONS),
        help=(
            "Comma-separated Schwartz dimensions to oversample when "
            "`--pilot-size` is set."
        ),
    )
    args = parser.parse_args()

    pilot_hard_dimensions = tuple(
        value.strip()
        for value in args.pilot_hard_dimensions.split(",")
        if value.strip()
    )
    invalid_dimensions = sorted(
        set(pilot_hard_dimensions) - set(SCHWARTZ_VALUE_ORDER)
    )
    if invalid_dimensions:
        raise ValueError(
            "Invalid pilot hard dimensions: " + ", ".join(invalid_dimensions)
        )

    manifest, shard_manifest = prepare_consensus_bundle(
        output_dir=Path(args.output_dir),
        labels_path=Path(args.labels_path),
        wrangled_dir=Path(args.wrangled_dir),
        schwartz_path=Path(args.schwartz_path),
        pilot_size=args.pilot_size,
        pilot_hard_dimensions=pilot_hard_dimensions,
    )

    print(f"Wrote consensus bundle: {Path(args.output_dir)}")
    print(f"Entries: {manifest.height}")
    print(f"Shards per pass: {shard_manifest.filter(pl.col('pass_index') == 1).height}")


if __name__ == "__main__":
    main()
