"""Tests for the twinkl-754 consensus re-judging helpers."""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.judge.consolidate import consolidate_consensus_labels
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.dataset import load_labels

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar divide:RuntimeWarning"
    ),
]


ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = _load_script_module(
    "twinkl_754_prepare_consensus",
    "scripts/journalling/twinkl_754_prepare_consensus.py",
)
VALIDATE = _load_script_module(
    "twinkl_754_validate_results",
    "scripts/journalling/twinkl_754_validate_results.py",
)
MERGE = _load_script_module(
    "twinkl_754_merge_pass_results",
    "scripts/journalling/twinkl_754_merge_pass_results.py",
)
SUMMARIZE = _load_script_module(
    "twinkl_754_summarize_consensus",
    "scripts/journalling/twinkl_754_summarize_consensus.py",
)


def _scores(**overrides: int) -> dict[str, int]:
    scores = {value_name: 0 for value_name in SCHWARTZ_VALUE_ORDER}
    scores.update(overrides)
    return scores


def _alignment_vector(scores: dict[str, int]) -> list[int]:
    return [scores[value_name] for value_name in SCHWARTZ_VALUE_ORDER]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_wrangled_persona(
    path: Path,
    *,
    persona_id: str,
    name: str,
    age: str,
    profession: str,
    culture: str,
    core_values: list[str],
    bio: str,
    entries: list[dict],
) -> None:
    lines = [
        f"# Persona {persona_id}: {name}",
        "",
        "## Profile",
        f"- **Persona ID:** {persona_id}",
        f"- **Name:** {name}",
        f"- **Age:** {age}",
        f"- **Profession:** {profession}",
        f"- **Culture:** {culture}",
        f"- **Core Values:** {', '.join(core_values)}",
        f"- **Bio:** {bio}",
        "",
        "---",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"## Entry {entry['t_index']} - {entry['date']}",
                "",
                entry["initial_entry"],
                "",
                f'**Nudge:** "{entry["nudge_text"]}"',
                "",
                f"**Response:** {entry['response_text']}",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_prepare_fixture(tmp_path: Path) -> tuple[Path, Path]:
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir()

    persona_entries = {
        "aaa11111": {
            "name": "Alicia Tan",
            "age": "25-34",
            "profession": "Teacher",
            "culture": "Singaporean",
            "core_values": ["Security", "Benevolence"],
            "bio": "Alicia is careful with routines and deeply committed to her students.",
            "entries": [
                {
                    "t_index": 0,
                    "date": "2025-01-01",
                    "initial_entry": "I stayed late to reorganize the classroom before school reopened.",
                    "nudge_text": "Why did that feel important?",
                    "response_text": "I wanted the children to feel settled on the first day back.",
                },
                {
                    "t_index": 1,
                    "date": "2025-01-08",
                    "initial_entry": "I skipped dinner out so I could help a parent prepare learning materials.",
                    "nudge_text": "Did you resent missing the evening?",
                    "response_text": "Not really. It felt useful, even if I was tired.",
                },
                {
                    "t_index": 2,
                    "date": "2025-01-15",
                    "initial_entry": "I finally booked a short holiday after weeks of overworking.",
                    "nudge_text": "What made you choose now?",
                    "response_text": "I could feel myself running out of patience and needed a reset.",
                },
            ],
        },
        "bbb22222": {
            "name": "Ben Yeo",
            "age": "35-44",
            "profession": "Operations Manager",
            "culture": "Singaporean",
            "core_values": ["Achievement", "Security"],
            "bio": "Ben likes dependable systems and prefers being over-prepared.",
            "entries": [
                {
                    "t_index": 0,
                    "date": "2025-02-01",
                    "initial_entry": "I rewrote tomorrow's presentation three times to avoid any loose ends.",
                    "nudge_text": "What worried you most?",
                    "response_text": "Looking unprepared in front of the senior team.",
                },
                {
                    "t_index": 1,
                    "date": "2025-02-10",
                    "initial_entry": "I handed part of the project to my deputy so I could leave on time for once.",
                    "nudge_text": "How did that decision sit with you?",
                    "response_text": "Uncomfortable at first, but I was glad to keep my promise at home.",
                },
                {
                    "t_index": 2,
                    "date": "2025-02-18",
                    "initial_entry": "I turned down a flashy role because the new company felt unstable.",
                    "nudge_text": "Do you think you played it too safe?",
                    "response_text": "Maybe, but I sleep better with predictability.",
                },
            ],
        },
    }

    label_rows: list[dict] = []
    for persona_id, persona in persona_entries.items():
        _write_wrangled_persona(
            wrangled_dir / f"persona_{persona_id}.md",
            persona_id=persona_id,
            name=persona["name"],
            age=persona["age"],
            profession=persona["profession"],
            culture=persona["culture"],
            core_values=persona["core_values"],
            bio=persona["bio"],
            entries=persona["entries"],
        )
        for entry in persona["entries"]:
            if persona_id == "aaa11111":
                scores = _scores(
                    security=1 if entry["t_index"] == 0 else 0,
                    benevolence=1 if entry["t_index"] in {0, 1} else 0,
                    hedonism=1 if entry["t_index"] == 2 else 0,
                )
            else:
                scores = _scores(
                    achievement=1 if entry["t_index"] == 0 else 0,
                    security=1 if entry["t_index"] == 2 else 0,
                    hedonism=-1 if entry["t_index"] == 0 else 0,
                )
            label_rows.append(
                {
                    "persona_id": persona_id,
                    "t_index": entry["t_index"],
                    "date": entry["date"],
                    "alignment_vector": _alignment_vector(scores),
                    **{
                        f"alignment_{value_name}": scores[value_name]
                        for value_name in SCHWARTZ_VALUE_ORDER
                    },
                    "rationales_json": None,
                }
            )

    labels_path = tmp_path / "judge_labels.parquet"
    pl.DataFrame(label_rows).write_parquet(labels_path)
    return labels_path, wrangled_dir


def _make_result_row(
    entry_id: str,
    *,
    security: int = 0,
    power: int = 0,
    rationale_prefix: str = "pass",
) -> dict:
    scores = _scores(security=security, power=power)
    rationales = {}
    if security != 0:
        rationales["security"] = f"{rationale_prefix} security rationale for {entry_id}"
    if power != 0:
        rationales["power"] = f"{rationale_prefix} power rationale for {entry_id}"
    return {
        "entry_id": entry_id,
        "scores": scores,
        "rationales": rationales,
    }


def _build_summary_fixture(
    tmp_path: Path,
    *,
    merge_results: bool = True,
) -> tuple[Path, Path]:
    bundle_dir = tmp_path / "bundle"
    prompts_dir = bundle_dir / "prompts"
    results_dir = bundle_dir / "results"
    shards_dir = bundle_dir / "shards"
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    entry_specs = [
        {
            "entry_id": "aaa11111__0",
            "persona_id": "aaa11111",
            "t_index": 0,
            "date": "2025-03-01",
            "persisted": _scores(security=0),
            "passes": [(0, 0), (0, 1), (0, -1), (0, 0), (0, 0)],
        },
        {
            "entry_id": "aaa11111__1",
            "persona_id": "aaa11111",
            "t_index": 1,
            "date": "2025-03-02",
            "persisted": _scores(security=1),
            "passes": [(0, 0), (0, 0), (0, 0), (0, 0), (1, 0)],
        },
        {
            "entry_id": "aaa11111__2",
            "persona_id": "aaa11111",
            "t_index": 2,
            "date": "2025-03-03",
            "persisted": _scores(security=-1),
            "passes": [(1, 0), (1, 0), (1, 0), (1, 0), (-1, 0)],
        },
        {
            "entry_id": "bbb22222__0",
            "persona_id": "bbb22222",
            "t_index": 0,
            "date": "2025-03-04",
            "persisted": _scores(security=1),
            "passes": [(1, 0), (1, 0), (-1, 0), (0, 0), (0, 0)],
        },
        {
            "entry_id": "bbb22222__1",
            "persona_id": "bbb22222",
            "t_index": 1,
            "date": "2025-03-05",
            "persisted": _scores(security=-1, power=1),
            "passes": [(1, -1), (1, -1), (1, 1), (1, 1), (-1, 0)],
        },
        {
            "entry_id": "bbb22222__2",
            "persona_id": "bbb22222",
            "t_index": 2,
            "date": "2025-03-06",
            "persisted": _scores(security=-1),
            "passes": [(-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0)],
        },
    ]

    manifest_rows = []
    prompt_rows = []
    pass_rows = {index: [] for index in range(1, 6)}
    shard_manifest_rows = []
    human_rows = []

    for spec in entry_specs:
        manifest_rows.append(
            {
                "entry_id": spec["entry_id"],
                "persona_id": spec["persona_id"],
                "t_index": spec["t_index"],
                "date": spec["date"],
                **{
                    f"alignment_{value_name}": spec["persisted"][value_name]
                    for value_name in SCHWARTZ_VALUE_ORDER
                },
            }
        )
        prompt_rows.append({"entry_id": spec["entry_id"]})

        consensus_security, _, _ = SUMMARIZE._resolve_votes(
            [vote[0] for vote in spec["passes"]]
        )
        consensus_power, _, _ = SUMMARIZE._resolve_votes(
            [vote[1] for vote in spec["passes"]]
        )
        human_rows.append(
            {
                "persona_id": spec["persona_id"],
                "t_index": spec["t_index"],
                **{
                    f"alignment_{value_name}": 0
                    for value_name in SCHWARTZ_VALUE_ORDER
                },
                "alignment_security": consensus_security,
                "alignment_power": consensus_power,
            }
        )

        for pass_index, (security_vote, power_vote) in enumerate(spec["passes"], start=1):
            pass_rows[pass_index].append(
                _make_result_row(
                    spec["entry_id"],
                    security=security_vote,
                    power=power_vote,
                    rationale_prefix=f"pass_{pass_index}",
                )
            )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(manifest_rows).write_csv(bundle_dir / "manifest.csv")
    for pass_index in range(1, 6):
        _write_jsonl(prompts_dir / f"pass_{pass_index}.jsonl", prompt_rows)
        shard_id = f"pass_{pass_index}_shard_001"
        shard_prompt_path = shards_dir / f"pass_{pass_index}" / f"{shard_id}.jsonl"
        shard_result_path = (
            results_dir / f"pass_{pass_index}" / "shards" / f"{shard_id}_results.jsonl"
        )
        _write_jsonl(shard_prompt_path, prompt_rows)
        _write_jsonl(shard_result_path, pass_rows[pass_index])
        (results_dir / f"pass_{pass_index}_results.jsonl").write_text(
            "",
            encoding="utf-8",
        )
        shard_manifest_rows.append(
            {
                "pass_index": pass_index,
                "pass_name": f"pass_{pass_index}",
                "shard_id": shard_id,
                "prompt_path": str(shard_prompt_path),
                "result_path": str(shard_result_path),
                "n_entries": len(prompt_rows),
                "n_personas": 2,
                "first_entry_id": prompt_rows[0]["entry_id"],
                "last_entry_id": prompt_rows[-1]["entry_id"],
                "persona_ids_json": json.dumps(["aaa11111", "bbb22222"]),
            }
        )
    pl.DataFrame(shard_manifest_rows).write_csv(bundle_dir / "shard_manifest.csv")

    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for annotator_id in ("des", "jl", "km"):
        rows = []
        for row in human_rows:
            rows.append(
                {
                    "persona_id": row["persona_id"],
                    "t_index": row["t_index"],
                    "annotator_id": annotator_id,
                    "timestamp": timestamp,
                    **{
                        f"alignment_{value_name}": row[f"alignment_{value_name}"]
                        for value_name in SCHWARTZ_VALUE_ORDER
                    },
                }
            )
        pl.DataFrame(rows).write_parquet(annotations_dir / f"{annotator_id}.parquet")

    if merge_results:
        MERGE.merge_pass_results(bundle_dir)

    return bundle_dir, annotations_dir


def _make_comparison_row(
    *,
    dimension: str,
    persona_id: str,
    t_index: int,
    votes: list[int],
    consensus_label: int,
    confidence_tier: str,
) -> dict:
    return {
        "entry_id": f"{persona_id}__{t_index}",
        "persona_id": persona_id,
        "t_index": t_index,
        "date": f"2025-04-{t_index + 1:02d}",
        "dimension": dimension,
        "persisted_label": 0,
        "consensus_label": consensus_label,
        "confidence_tier": confidence_tier,
        "consensus_agreement": 0,
        "label_changed": False,
        **{
            f"pass_{pass_index}_label": vote
            for pass_index, vote in enumerate(votes, start=1)
        },
    }


def test_prepare_consensus_bundle_is_deterministic_and_strips_profile_history(tmp_path):
    labels_path, wrangled_dir = _build_prepare_fixture(tmp_path)
    output_dir = tmp_path / "export"

    manifest_a, shard_manifest_a = PREP.prepare_consensus_bundle(
        output_dir=output_dir,
        labels_path=labels_path,
        wrangled_dir=wrangled_dir,
        schwartz_path=ROOT / "config" / "schwartz_values.yaml",
        expected_total_entries=6,
        max_shard_personas=1,
        max_shard_entries=4,
    )

    stale_file = output_dir / "prompts" / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    manifest_b, shard_manifest_b = PREP.prepare_consensus_bundle(
        output_dir=output_dir,
        labels_path=labels_path,
        wrangled_dir=wrangled_dir,
        schwartz_path=ROOT / "config" / "schwartz_values.yaml",
        expected_total_entries=6,
        max_shard_personas=1,
        max_shard_entries=4,
    )

    assert manifest_a.equals(manifest_b)
    assert shard_manifest_a.equals(shard_manifest_b)
    assert not stale_file.exists()

    prompt_rows = _read_jsonl(output_dir / "prompts" / "pass_1.jsonl")
    assert len(prompt_rows) == 6
    assert (output_dir / "README.md").exists()

    second_entry = next(row for row in prompt_rows if row["entry_id"] == "aaa11111__1")
    assert second_entry["context_flags"] == {
        "bio_included": False,
        "previous_entries_included": False,
        "core_values_included": True,
    }
    assert second_entry["context_stats"]["previous_entries_count"] == 1
    assert "Alicia is careful with routines" not in second_entry["prompt"]
    assert "## Recent Entries" not in second_entry["prompt"]
    assert "Core Values (from profile):** Security, Benevolence" in second_entry["prompt"]
    assert 'Nudge: "Did you resent missing the evening?"' in second_entry["prompt"]
    assert "Response: Not really. It felt useful, even if I was tired." in second_entry["prompt"]

    bundle_status = json.loads((output_dir / "bundle_status.json").read_text(encoding="utf-8"))
    assert bundle_status["status"] == "prepared"
    assert bundle_status["bundle_mode"] == "full"
    assert bundle_status["invalidated"] is False


def test_prepare_consensus_bundle_supports_deterministic_pilot_selection(tmp_path):
    labels_path, wrangled_dir = _build_prepare_fixture(tmp_path)
    output_dir = tmp_path / "pilot_export"

    manifest, shard_manifest = PREP.prepare_consensus_bundle(
        output_dir=output_dir,
        labels_path=labels_path,
        wrangled_dir=wrangled_dir,
        schwartz_path=ROOT / "config" / "schwartz_values.yaml",
        expected_total_entries=6,
        max_shard_personas=1,
        max_shard_entries=4,
        pilot_size=4,
        pilot_hard_dimensions=("security", "hedonism"),
    )

    assert manifest.height == 4
    assert shard_manifest.filter(pl.col("pass_index") == 1).height == 2
    assert manifest["entry_id"].to_list() == [
        "aaa11111__0",
        "aaa11111__2",
        "bbb22222__0",
        "bbb22222__2",
    ]

    bundle_status = json.loads((output_dir / "bundle_status.json").read_text(encoding="utf-8"))
    assert bundle_status["bundle_mode"] == "pilot"
    assert bundle_status["selected_entry_count"] == 4
    assert bundle_status["selection_summary"]["selected_non_zero_counts"] == {
        "security": 2,
        "hedonism": 2,
    }
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Bundle mode: `pilot`" in readme
    assert "Selected non-zero `security` entries: `2`" in readme


def test_build_shards_respects_default_persona_and_entry_caps():
    prompt_rows = []
    for persona_index in range(6):
        persona_id = f"{persona_index + 1:08x}"
        for t_index in range(5):
            prompt_rows.append(
                {
                    "entry_id": f"{persona_id}__{t_index}",
                    "persona_id": persona_id,
                    "t_index": t_index,
                    "prompt": f"prompt {persona_id} {t_index}",
                    "persisted_scores": _scores(),
                }
            )

    shards = PREP.build_shards(prompt_rows, pass_index=1)

    assert len(shards) == 2
    assert all(len(shard["persona_ids"]) <= PREP.MAX_SHARD_PERSONAS for shard in shards)
    assert all(len(shard["rows"]) <= PREP.MAX_SHARD_ENTRIES for shard in shards)
    assert shards[0]["persona_ids"] == ["00000001", "00000002", "00000003", "00000004"]
    assert shards[1]["persona_ids"] == ["00000005", "00000006"]


def test_validate_results_rejects_missing_duplicate_and_unknown_ids(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    expected_jsonl = tmp_path / "expected.jsonl"

    manifest = pl.DataFrame(
        [
            {"entry_id": "aaa11111__0"},
            {"entry_id": "aaa11111__1"},
        ]
    )
    manifest.write_csv(manifest_path)
    _write_jsonl(
        expected_jsonl,
        [
            {"entry_id": "aaa11111__0"},
            {"entry_id": "aaa11111__1"},
        ],
    )

    valid_rows = [
        {
            "entry_id": "aaa11111__0",
            "scores": _scores(security=1),
            "rationales": {"security": "Kept the routine stable."},
        },
        {
            "entry_id": "aaa11111__1",
            "scores": _scores(),
            "rationales": {},
        },
    ]
    normalized = VALIDATE.validate_result_rows(
        valid_rows,
        valid_manifest_ids={"aaa11111__0", "aaa11111__1"},
        expected_entry_ids=["aaa11111__0", "aaa11111__1"],
    )
    assert [row["entry_id"] for row in normalized] == ["aaa11111__0", "aaa11111__1"]

    with pytest.raises(ValueError, match="Missing="):
        VALIDATE.validate_result_rows(
            valid_rows[:1],
            valid_manifest_ids={"aaa11111__0", "aaa11111__1"},
            expected_entry_ids=["aaa11111__0", "aaa11111__1"],
        )

    with pytest.raises(ValueError, match="Duplicate entry_ids"):
        VALIDATE.validate_result_rows(
            [valid_rows[0], valid_rows[0]],
            valid_manifest_ids={"aaa11111__0", "aaa11111__1"},
            expected_entry_ids=["aaa11111__0", "aaa11111__1"],
        )

    with pytest.raises(ValueError, match="Unknown entry_id"):
        VALIDATE.validate_result_rows(
            [
                valid_rows[0],
                {
                    "entry_id": "ccc33333__9",
                    "scores": _scores(),
                    "rationales": {},
                },
            ],
            valid_manifest_ids={"aaa11111__0", "aaa11111__1"},
            expected_entry_ids=["aaa11111__0", "aaa11111__1"],
        )

    with pytest.raises(ValueError, match="missing rationales for non-zero scores"):
        VALIDATE.validate_result_rows(
            [
                {
                    "entry_id": "aaa11111__0",
                    "scores": _scores(security=1),
                    "rationales": {},
                },
                valid_rows[1],
            ],
            valid_manifest_ids={"aaa11111__0", "aaa11111__1"},
            expected_entry_ids=["aaa11111__0", "aaa11111__1"],
        )


def test_resolve_votes_tier_boundaries():
    assert SUMMARIZE._resolve_votes([1, 1, 1, 1, 1]) == (1, "unanimous", 5)
    assert SUMMARIZE._resolve_votes([1, 1, 1, 1, -1]) == (1, "strong", 4)
    assert SUMMARIZE._resolve_votes([1, 1, 1, -1, -1]) == (1, "bare_majority", 3)
    assert SUMMARIZE._resolve_votes([1, 1, -1, -1, 0]) == (0, "no_majority", 4)
    assert SUMMARIZE._resolve_votes([0, 0, 0, 1, -1]) == (0, "bare_majority", 3)


def test_load_human_benchmark_counts_strict_three_way_overlap(tmp_path):
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)

    annotator_rows = {
        "des": [("aaa11111", 0), ("aaa11111", 1), ("bbb22222", 0)],
        "jl": [("aaa11111", 0), ("aaa11111", 1)],
        "km": [("aaa11111", 0), ("ccc33333", 0)],
    }

    for annotator_id, keys in annotator_rows.items():
        rows = []
        for persona_id, t_index in keys:
            rows.append(
                {
                    "persona_id": persona_id,
                    "t_index": t_index,
                    "annotator_id": annotator_id,
                    "timestamp": timestamp,
                    **{f"alignment_{value_name}": 0 for value_name in SCHWARTZ_VALUE_ORDER},
                }
            )
        pl.DataFrame(rows).write_parquet(annotations_dir / f"{annotator_id}.parquet")

    majority_wide, summary = SUMMARIZE._load_human_benchmark(
        annotations_dir,
        full_corpus_entry_count=10,
    )

    assert summary == {
        "annotator_file_count": 3,
        "union_entry_count": 4,
        "union_persona_count": 3,
        "strict_overlap_entry_count": 1,
        "strict_overlap_persona_count": 1,
        "single_annotated_entry_count": 2,
        "excluded_full_corpus_entry_count": 9,
    }
    assert majority_wide.height == 1
    assert majority_wide.select(["persona_id", "t_index"]).to_dicts() == [
        {"persona_id": "aaa11111", "t_index": 0}
    ]


def test_build_stability_summary_computes_metrics_and_is_deterministic():
    comparison_rows = pl.DataFrame(
        [
            _make_comparison_row(
                dimension="security",
                persona_id="aaa11111",
                t_index=0,
                votes=[1, 1, 1, 1, 1],
                consensus_label=1,
                confidence_tier="unanimous",
            ),
            _make_comparison_row(
                dimension="security",
                persona_id="aaa11111",
                t_index=1,
                votes=[1, 1, 1, -1, -1],
                consensus_label=1,
                confidence_tier="bare_majority",
            ),
            _make_comparison_row(
                dimension="security",
                persona_id="bbb22222",
                t_index=0,
                votes=[0, 0, 0, 0, 0],
                consensus_label=0,
                confidence_tier="unanimous",
            ),
            _make_comparison_row(
                dimension="security",
                persona_id="bbb22222",
                t_index=1,
                votes=[1, -1, 0, 0, 0],
                consensus_label=0,
                confidence_tier="bare_majority",
            ),
        ]
    )

    first = SUMMARIZE._build_stability_summary(comparison_rows)
    second = SUMMARIZE._build_stability_summary(comparison_rows)

    assert first.equals(second)

    security_row = first.filter(pl.col("dimension") == "security").to_dicts()[0]
    entropy_three_two = -(0.6 * math.log2(0.6) + 0.4 * math.log2(0.4))
    entropy_three_one_one = (
        -(0.6 * math.log2(0.6) + 0.2 * math.log2(0.2) + 0.2 * math.log2(0.2))
    )

    assert security_row["difficulty_rank"] == 1
    assert security_row["n_entries_point"] == pytest.approx(4.0)
    assert security_row["n_non_neutral_entries_point"] == pytest.approx(2.0)
    assert security_row["non_unanimous_rate_all_point"] == pytest.approx(0.5)
    assert security_row["polarity_flip_rate_all_point"] == pytest.approx(0.5)
    assert security_row["low_confidence_non_neutral_ratio_point"] == pytest.approx(0.5)
    assert security_row["non_unanimous_rate_non_neutral_point"] == pytest.approx(0.5)
    assert security_row["polarity_flip_rate_non_neutral_point"] == pytest.approx(0.5)
    assert security_row["mean_vote_entropy_all_point"] == pytest.approx(
        (entropy_three_two + entropy_three_one_one) / 4.0
    )
    assert security_row["mean_vote_entropy_non_neutral_point"] == pytest.approx(
        entropy_three_two / 2.0
    )
    assert math.isfinite(security_row["low_confidence_non_neutral_ratio_ci_lo"])
    assert math.isfinite(security_row["low_confidence_non_neutral_ratio_ci_hi"])


def test_evaluate_gate_uses_upper_ci_on_low_confidence_ratio():
    stability_summary = pl.DataFrame(
        [
            {
                "dimension": "security",
                "n_non_neutral_entries_point": 40.0,
                "low_confidence_non_neutral_ratio_point": 0.31,
                "low_confidence_non_neutral_ratio_ci_lo": 0.20,
                "low_confidence_non_neutral_ratio_ci_hi": 0.49,
                "non_unanimous_rate_non_neutral_point": 0.42,
                "non_unanimous_rate_non_neutral_ci_lo": 0.30,
                "non_unanimous_rate_non_neutral_ci_hi": 0.54,
                "mean_vote_entropy_non_neutral_point": 0.81,
                "mean_vote_entropy_non_neutral_ci_lo": 0.70,
                "mean_vote_entropy_non_neutral_ci_hi": 0.92,
                "polarity_flip_rate_non_neutral_point": 0.11,
                "polarity_flip_rate_non_neutral_ci_lo": 0.06,
                "polarity_flip_rate_non_neutral_ci_hi": 0.17,
            },
            {
                "dimension": "hedonism",
                "n_non_neutral_entries_point": 32.0,
                "low_confidence_non_neutral_ratio_point": 0.28,
                "low_confidence_non_neutral_ratio_ci_lo": 0.17,
                "low_confidence_non_neutral_ratio_ci_hi": 0.44,
                "non_unanimous_rate_non_neutral_point": 0.39,
                "non_unanimous_rate_non_neutral_ci_lo": 0.25,
                "non_unanimous_rate_non_neutral_ci_hi": 0.52,
                "mean_vote_entropy_non_neutral_point": 0.76,
                "mean_vote_entropy_non_neutral_ci_lo": 0.64,
                "mean_vote_entropy_non_neutral_ci_hi": 0.89,
                "polarity_flip_rate_non_neutral_point": 0.08,
                "polarity_flip_rate_non_neutral_ci_lo": 0.03,
                "polarity_flip_rate_non_neutral_ci_hi": 0.15,
            },
            {
                "dimension": "stimulation",
                "n_non_neutral_entries_point": 18.0,
                "low_confidence_non_neutral_ratio_point": 0.24,
                "low_confidence_non_neutral_ratio_ci_lo": 0.13,
                "low_confidence_non_neutral_ratio_ci_hi": 0.48,
                "non_unanimous_rate_non_neutral_point": 0.33,
                "non_unanimous_rate_non_neutral_ci_lo": 0.20,
                "non_unanimous_rate_non_neutral_ci_hi": 0.47,
                "mean_vote_entropy_non_neutral_point": 0.58,
                "mean_vote_entropy_non_neutral_ci_lo": 0.45,
                "mean_vote_entropy_non_neutral_ci_hi": 0.71,
                "polarity_flip_rate_non_neutral_point": 0.05,
                "polarity_flip_rate_non_neutral_ci_lo": 0.01,
                "polarity_flip_rate_non_neutral_ci_hi": 0.11,
            },
        ]
    )

    passing_gate = SUMMARIZE._evaluate_gate(stability_summary)
    assert passing_gate["stability_gate_passed"] is True
    assert passing_gate["overall_passed"] is True
    assert passing_gate["recommendation"] == (
        "Eligible for retrain comparison under full-corpus stability criteria."
    )

    failing_gate = SUMMARIZE._evaluate_gate(
        stability_summary.with_columns(
            pl.when(pl.col("dimension") == "security")
            .then(0.51)
            .otherwise(pl.col("low_confidence_non_neutral_ratio_ci_hi"))
            .alias("low_confidence_non_neutral_ratio_ci_hi")
        )
    )
    assert failing_gate["stability_gate_passed"] is False
    assert failing_gate["overall_passed"] is False
    assert failing_gate["recommendation"] == "Hold retrain until full-corpus stability improves."


def test_summarize_bundle_applies_consensus_voting_and_gate_rules(tmp_path):
    bundle_dir, annotations_dir = _build_summary_fixture(tmp_path)

    (
        report,
        consensus_frame,
        joined_results,
        comparison_rows,
        _flip_summary,
        _confidence_summary,
        irr_summary,
        stability_summary,
        gate_summary,
    ) = SUMMARIZE.summarize_bundle(bundle_dir, annotations_dir=annotations_dir)

    by_entry = {
        row["entry_id"]: row
        for row in consensus_frame.sort(["persona_id", "t_index"]).to_dicts()
    }

    assert by_entry["aaa11111__0"]["alignment_security"] == 0
    assert by_entry["aaa11111__0"]["confidence_security"] == "unanimous"
    assert by_entry["aaa11111__0"]["consensus_agreement_security"] == 5

    assert by_entry["aaa11111__1"]["alignment_security"] == 0
    assert by_entry["aaa11111__1"]["confidence_security"] == "strong"
    assert by_entry["aaa11111__1"]["consensus_agreement_security"] == 4

    assert by_entry["aaa11111__2"]["alignment_security"] == 1
    assert by_entry["aaa11111__2"]["confidence_security"] == "strong"
    assert by_entry["aaa11111__2"]["consensus_agreement_security"] == 4

    assert by_entry["bbb22222__0"]["alignment_security"] == 1
    assert by_entry["bbb22222__0"]["confidence_security"] == "bare_majority"
    assert by_entry["bbb22222__0"]["consensus_agreement_security"] == 2

    assert by_entry["bbb22222__1"]["alignment_power"] == 0
    assert by_entry["bbb22222__1"]["confidence_power"] == "no_majority"
    assert by_entry["bbb22222__1"]["consensus_agreement_power"] == 4
    assert by_entry["bbb22222__1"]["rationale_source_pass"] == 1
    assert by_entry["bbb22222__1"]["rationale_mismatch_count"] == 1

    assert by_entry["bbb22222__2"]["alignment_security"] == -1
    assert by_entry["bbb22222__2"]["confidence_security"] == "unanimous"
    assert by_entry["bbb22222__2"]["consensus_agreement_security"] == 5

    assert gate_summary["overall_passed"] == gate_summary["stability_gate_passed"]
    assert "agreement_passed" not in gate_summary
    assert "Entries using fallback rationale selection: `1`" in report
    assert "Judge Repeated-Call Self-Consistency" in report
    assert "Consensus vs persisted Cohen kappa" in report
    assert "Pairwise similarity:" in report
    assert "Human-Overlap Benchmark (Advisory)" in report
    assert "non-expert human-overlap benchmark" in report
    assert "- Annotator files loaded: `3`" in report
    assert "- Strict 3-way overlap used for comparison: `6` entries across `2` personas" in report
    assert "Full-Corpus Stability Gate" in report
    assert "Hard-Dimension Gate" not in report
    assert "Full-corpus stability first:" in report
    assert "Hard-Dimension Deep Dive" in report
    assert "Stop after repeated-call diagnostics review" not in report
    assert (
        "Eligible for retrain comparison under full-corpus stability criteria." in report
        or "Hold retrain until full-corpus stability improves." in report
    )
    assert stability_summary.filter(pl.col("dimension") == "security").height == 1
    advisory_roles = irr_summary.filter(
        pl.col("metric") == "consensus_vs_human_cohen_kappa"
    )["decision_role"].to_list()
    assert advisory_roles == ["advisory_only"] * (len(SCHWARTZ_VALUE_ORDER) + 1)

    power_row = comparison_rows.filter(
        (pl.col("entry_id") == "bbb22222__1") & (pl.col("dimension") == "power")
    ).to_dicts()[0]
    assert power_row["consensus_label"] == 0
    assert power_row["confidence_tier"] == "no_majority"
    assert power_row["label_changed"] is True

    joined_entry = joined_results.filter(pl.col("entry_id") == "bbb22222__1").to_dicts()[0]
    assert joined_entry["pass_1_rationales_json"] is not None
    assert joined_entry["selected_rationales_json"] is not None


def test_summarize_bundle_surfaces_invalidated_bundle_warning(tmp_path):
    bundle_dir, annotations_dir = _build_summary_fixture(tmp_path)
    status_path = bundle_dir / "bundle_status.json"
    bundle_status = json.loads(status_path.read_text(encoding="utf-8"))
    bundle_status["invalidated"] = True
    bundle_status["warning"] = (
        "This bundle was invalidated after duplicate-pass QC; treat it as forensic-only."
    )
    status_path.write_text(json.dumps(bundle_status, indent=2) + "\n", encoding="utf-8")

    report, *_ = SUMMARIZE.summarize_bundle(bundle_dir, annotations_dir=annotations_dir)

    assert "> Warning: this bundle is marked invalidated." in report
    assert "forensic-only" in report


def test_consensus_parquet_is_compatible_with_dataset_loader(tmp_path):
    bundle_dir, annotations_dir = _build_summary_fixture(tmp_path)
    (
        _report,
        consensus_frame,
        _joined_results,
        _comparison_rows,
        _flip_summary,
        _confidence_summary,
        _irr_summary,
        _stability_summary,
        _gate_summary,
    ) = SUMMARIZE.summarize_bundle(bundle_dir, annotations_dir=annotations_dir)

    output_path = tmp_path / "consensus_labels.parquet"
    written = consolidate_consensus_labels(consensus_frame, output_path=output_path)
    reloaded = load_labels(output_path)

    assert output_path.exists()
    assert reloaded.height == written.height
    assert "confidence_security" in reloaded.columns
    assert "consensus_agreement_security" in reloaded.columns
    assert "label_changed_security" in reloaded.columns


def test_merge_results_rejects_duplicate_passes_before_summarization(tmp_path):
    bundle_dir, _annotations_dir = _build_summary_fixture(tmp_path, merge_results=False)
    source_path = bundle_dir / "results" / "pass_1" / "shards" / "pass_1_shard_001_results.jsonl"
    duplicate_path = bundle_dir / "results" / "pass_4" / "shards" / "pass_4_shard_001_results.jsonl"
    duplicate_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate pass outputs detected"):
        MERGE.merge_pass_results(bundle_dir)


def test_merge_results_rejects_incomplete_rationale_coverage(tmp_path):
    bundle_dir, _annotations_dir = _build_summary_fixture(tmp_path, merge_results=False)
    failing_path = bundle_dir / "results" / "pass_3" / "shards" / "pass_3_shard_001_results.jsonl"
    rows = _read_jsonl(failing_path)
    for row in rows:
        if row["scores"]["security"] != 0:
            row["rationales"].pop("security", None)
            break
    _write_jsonl(failing_path, rows)

    with pytest.raises(ValueError, match="missing rationales for non-zero scores"):
        MERGE.merge_pass_results(bundle_dir)
