#!/usr/bin/env python3
"""Prepare receipt-bound exact-state reviews for the selected Security subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judge.labeling import (  # noqa: E402
    load_schwartz_values,
    render_active_critic_state_prompt,
)
from src.models.judge import SCHWARTZ_VALUE_ORDER  # noqa: E402
from src.vif.dataset import load_entries  # noqa: E402
from src.vif.security_target import (  # noqa: E402
    ACTIVE_CRITIC_STATE_CONTRACT_VERSION,
    EXPECTED_CONTEXT_FLAGS,
    REQUIRED_LEGACY_COLUMNS,
    sha256_canonical_json,
)
from src.vif.state_encoder import (  # noqa: E402
    concatenate_entry_text,
    core_values_to_profile_weights,
)


def build_active_state_manifest(
    *,
    joined_results: pl.DataFrame,
    entries: pl.DataFrame,
    schwartz_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render an exact active-Critic-state prompt for each selected Security case."""
    missing = REQUIRED_LEGACY_COLUMNS - set(joined_results.columns)
    if missing:
        raise ValueError(
            "Missing required legacy reachability columns: "
            f"{sorted(missing)}."
        )
    required_entries = {
        "persona_id",
        "t_index",
        "date",
        "initial_entry",
        "nudge_text",
        "response_text",
        "core_values",
    }
    missing_entries = required_entries - set(entries.columns)
    if missing_entries:
        raise ValueError(
            "Wrangled entries are missing active-state fields: "
            f"{sorted(missing_entries)}."
        )

    security = joined_results.filter(pl.col("dimension") == "security").select(
        ["case_id", "dimension", "persona_id", "t_index", "date"]
    )
    if security.is_empty():
        raise ValueError("Reachability evidence contains no Security cases.")
    if security.select("case_id").n_unique() != security.height:
        raise ValueError("Security reachability evidence has duplicate case IDs.")
    coordinate_columns = ["persona_id", "t_index", "date"]
    if security.select(coordinate_columns).n_unique() != security.height:
        raise ValueError(
            "Security reachability evidence has duplicate entry coordinates."
        )

    entry_subset = entries.select(sorted(required_entries))
    duplicate_entries = (
        entry_subset.group_by(coordinate_columns)
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_entries.is_empty():
        raise ValueError("Wrangled entries have duplicate entry coordinates.")
    joined = security.join(
        entry_subset,
        on=coordinate_columns,
        how="inner",
    )
    matched_counts = joined.group_by("case_id").len()
    if (
        joined.height != security.height
        or matched_counts.height != security.height
        or matched_counts.filter(pl.col("len") != 1).height
    ):
        matched = set(joined["case_id"].to_list())
        expected = set(security["case_id"].to_list())
        raise ValueError(
            "Could not reconstruct the active state for every selected Security case. "
            f"Missing={sorted(expected - matched)[:5]}"
        )

    records = []
    for row in joined.sort("case_id").iter_rows(named=True):
        session_content = concatenate_entry_text(
            row["initial_entry"],
            row["nudge_text"],
            row["response_text"],
        )
        profile_values = core_values_to_profile_weights(row["core_values"] or [])
        profile_weights = {
            dimension: float(profile_values[index])
            for index, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
        }
        state_input = {
            "window_size": 1,
            "session_content": session_content,
            "profile_weights": profile_weights,
        }
        prompt = render_active_critic_state_prompt(
            session_content=session_content,
            profile_weights=profile_values.tolist(),
            schwartz_config=schwartz_config,
        )
        records.append(
            {
                "case_id": row["case_id"],
                "dimension": "security",
                "persona_id": row["persona_id"],
                "t_index": int(row["t_index"]),
                "date": row["date"],
                "state_contract_version": ACTIVE_CRITIC_STATE_CONTRACT_VERSION,
                "context_flags": EXPECTED_CONTEXT_FLAGS,
                "state_input": state_input,
                "state_input_sha256": sha256_canonical_json(state_input),
                "prompt_sha256": sha256_canonical_json({"prompt": prompt}),
                "prompt": prompt,
            }
        )
    return records


def write_active_state_bundle(
    *,
    output_dir: Path,
    manifest: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write immutable reviewer inputs and an empty result template."""
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite active-state bundle: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "active_critic_state_manifest.jsonl"
    results_path = output_dir / "active_critic_state_results.jsonl"
    _write_jsonl(manifest_path, manifest)
    results_path.write_text("", encoding="utf-8")
    (output_dir / "README.md").write_text(
        _bundle_readme(manifest_path.name, results_path.name), encoding="utf-8"
    )
    return manifest_path, results_path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _bundle_readme(manifest_name: str, results_name: str) -> str:
    return f"""# twinkl-a30f Active-Critic-State Review Bundle

This bundle is the only review input that may supply `new_label` for the
selected Security diagnostic subset. It is not a training or evaluation target.

## Contract

Every prompt contains only the active `window_size: 1` Critic state:

- runtime-formatted current session text; and
- the normalized 10-dimensional profile vector.

It excludes date, demographics, biography, prior entries, raw core-value names,
labels, rationales, and generation metadata.

## Files

- `{manifest_name}`: immutable reviewer prompts with state and prompt hashes.
- `{results_name}`: fill one JSON object per manifest case.

The review runner must send only each record's `prompt` value to the judging
model or human reviewer. The surrounding case coordinates and hashes are
reconciliation metadata; they are not part of the review input.

## Result format

Each result must bind to the supplied contract and hashes:

```json
{{
  "case_id": "security__example__1",
  "state_contract_version": "active_critic_state_v1",
  "state_input_sha256": "<copied from manifest>",
  "prompt_sha256": "<copied from manifest>",
  "reviewer": "reviewer-or-runtime-id",
  "reviewed_at": "2026-07-11T00:00:00+00:00",
  "confidence": "high",
  "rationale_status": "provided",
  "scores": {{
    "self_direction": 0,
    "stimulation": 0,
    "hedonism": 0,
    "achievement": 0,
    "power": 0,
    "security": 1,
    "conformity": 0,
    "tradition": 0,
    "benevolence": 0,
    "universalism": 0
  }},
  "rationales": {{"security": "Cites behavior from the current journal session."}}
}}
```

For a neutral Security score, use `"rationale_status": "not_applicable_neutral"`
and an empty `rationales` object. The target builder rejects missing cases,
duplicate cases, hash mismatches, unavailable-context flags, and any attempt to
use the legacy `twinkl-747` reduced-context labels as a fallback.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare exact active-Critic-state Security review prompts."
    )
    parser.add_argument(
        "--joined-results",
        type=Path,
        default=Path("logs/exports/twinkl_747/joined_results.csv"),
        help="Legacy reachability summary used only to identify the selected cases.",
    )
    parser.add_argument(
        "--wrangled-dir",
        type=Path,
        default=Path("logs/wrangled"),
        help="Wrangled persona directory used to reconstruct runtime state.",
    )
    parser.add_argument(
        "--schwartz-path",
        type=Path,
        default=Path("config/schwartz_values.yaml"),
        help="Schwartz rubric configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/exports/twinkl_a30f_active_critic_state_v1"),
        help="New directory for the immutable prompt manifest and empty results file.",
    )
    args = parser.parse_args()

    manifest = build_active_state_manifest(
        joined_results=pl.read_csv(args.joined_results),
        entries=load_entries(args.wrangled_dir),
        schwartz_config=load_schwartz_values(args.schwartz_path),
    )
    manifest_path, results_path = write_active_state_bundle(
        output_dir=args.output_dir,
        manifest=manifest,
    )
    print(f"Wrote {len(manifest)} active-state prompts: {manifest_path}")
    print(f"Fill exact-state review results: {results_path}")


if __name__ == "__main__":
    main()
