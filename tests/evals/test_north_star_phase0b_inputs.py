"""Synthetic input adversaries for future NSM development preparation."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from scripts.experiments import north_star_phase0b as legacy
from scripts.experiments import north_star_phase0b_inputs as inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def episode(persona, number, onset, confirmation, end):
    return {
        "episode_id": f"{persona}:benevolence:episode_{number:02d}",
        "canonical_case_id": f"{persona}:benevolence",
        "persona_id": persona,
        "dimension": "benevolence",
        **{
            f"{point}_{field}": value
            for point, index in (
                ("onset", onset),
                ("confirmation", confirmation),
                ("end", end),
            )
            for field, value in (
                ("t_index", index),
                ("position", index + 1),
                ("date", f"2026-01-{index + 1:02d}"),
            )
        },
    }


@pytest.fixture
def frozen(tmp_path):
    development = "aaaa"
    reserved = "bbbb"
    entries = [
        {
            "t_index": i,
            "date": f"2026-01-{i + 1:02d}",
            "initial_entry": f"I helped my friend with a difficult task on day {i}.",
        }
        for i in range(5)
    ]
    development_path = tmp_path / f"logs/wrangled/persona_{development}.md"
    development_path.parent.mkdir(parents=True)
    development_path.write_text(
        f"# Persona {development}: Synthetic development\n\n"
        + "\n\n".join(
            f"## Entry {row['t_index']} - {row['date']}\n\n{row['initial_entry']}"
            for row in entries
        )
    )
    reserved_path = tmp_path / f"logs/wrangled/persona_{reserved}.md"
    reserved_path.write_text("Reserved bytes only; this file must never be parsed.")
    episodes = [
        episode(development, 1, 2, 3, 4),
        episode(development, 2, 0, 1, 1),
        episode(reserved, 1, 0, 1, 1),
    ]
    episode_path = tmp_path / inputs.EPISODES
    episode_path.parent.mkdir(parents=True)
    pl.DataFrame(episodes).write_parquet(episode_path)
    for relative in (inputs.LABELS, inputs.CONSENSUS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Opaque hashed labels; preparation must not load labels")
    values_path = tmp_path / inputs.VALUES
    values_path.parent.mkdir(parents=True)
    values_path.write_text(
        "values:\n  Benevolence:\n    user_phrase: Helping friends\n"
        "    definition: Caring for close others\n"
    )
    phase0_path = tmp_path / "scripts/experiments/north_star_phase0.py"
    phase0_path.parent.mkdir(parents=True)
    phase0_path.write_text("# Synthetic source identity\n")
    review_path = tmp_path / "src/north_star/review.py"
    review_path.parent.mkdir(parents=True)
    review_path.write_text("# Synthetic legacy preparation source hash\n")
    policy_path = tmp_path / "config/north_star_policy.json"
    write_json(policy_path, {"test": "policy bytes"})
    retrieval_path = tmp_path / "report/retrieval.json"
    cohort_path = retrieval_path.with_name("cohort.json")
    cohort = {
        "development_persona_ids": [development],
        "reserved_persona_ids": [reserved],
        "excluded_personas": {},
        "evaluation_scope": "small_separate",
        "decision_source": "Synthetic regression fixture",
        "frozen_at": "2026-01-01T00:00:00+00:00",
        "seed": 7,
        "source_episode_sha256": digest(episode_path),
    }
    write_json(cohort_path, cohort)
    config_path = retrieval_path.with_name("retrieval_config.json")
    config = {
        "model": inputs.MODEL,
        "revision": inputs.REVISION,
        "code_revision": inputs.CODE_REVISION,
        "device": "cpu",
        "dimensions": 256,
        "normalization": "layer_norm -> truncate_256 -> L2",
        "document_template": inputs.DOCUMENT_TEMPLATE,
        "queries": {
            "benevolence": {
                "user_phrase": "Helping friends",
                "definition": "Caring for close others",
                "text": "search_query: Helping friends. Caring for close others",
            }
        },
        "batch_size": 8,
        "ks": list(inputs.KS),
        "tie_break": "cosine descending, t_index descending, entry_id ascending",
    }
    write_json(config_path, config)
    ranking = [
        {
            "entry_id": f"{development}:entry:{row['t_index']}",
            "t_index": row["t_index"],
            "date": row["date"],
            "source_sha256": hashlib.sha256(row["initial_entry"].encode()).hexdigest(),
            "similarity": 0.9 - row["t_index"] * 0.1,
        }
        for row in entries[:2]
    ]
    source_paths = [
        inputs.EPISODES,
        inputs.LABELS,
        inputs.CONSENSUS,
        inputs.VALUES,
        phase0_path.relative_to(tmp_path),
        development_path.relative_to(tmp_path),
        reserved_path.relative_to(tmp_path),
    ]
    result = {
        "schema_version": "north-star-phase0-v1",
        "sources": {str(path): digest(tmp_path / path) for path in source_paths},
        "retrieval": {
            "gate_passed": True,
            "selected_k": 3,
            "cohort": cohort,
            "cohort_sha256": digest(cohort_path),
            "config": config,
            "config_sha256": digest(config_path),
            "development_episodes": 2,
            "unique_documents": 2,
            "cases": [
                {"episode": row, "ranking": ranking if i == 0 else []}
                for i, row in enumerate(episodes[:2])
            ],
        },
    }
    write_json(retrieval_path, result)
    return {
        "root": tmp_path,
        "retrieval_path": retrieval_path,
        "policy_path": policy_path,
        "result": result,
        "entries": entries,
        "episodes": episodes,
        "development_path": development_path,
        "reserved_path": reserved_path,
    }


def build(frozen):
    return inputs.build_manifest(
        **{
            key: frozen[key]
            for key in (
                "root",
                "retrieval_path",
                "policy_path",
            )
        }
    )


def save(frozen, *, rehash=None):
    if rehash is not None:
        relative = str(rehash.relative_to(frozen["root"]))
        frozen["result"]["sources"][relative] = digest(rehash)
    write_json(frozen["retrieval_path"], frozen["result"])


def test_v1_reproduces_stale_text_preparation_defect(frozen, monkeypatch):
    changed = deepcopy(frozen["entries"])
    changed[0]["initial_entry"] = "A changed source silently replaces the frozen one."
    monkeypatch.setattr(legacy, "ROOT", frozen["root"])
    monkeypatch.setattr(legacy, "DIRECTORY", frozen["root"] / "legacy-output")
    monkeypatch.setattr(legacy, "RETRIEVAL", frozen["retrieval_path"])
    monkeypatch.setattr(legacy, "POLICY_PATH", frozen["policy_path"])
    monkeypatch.setattr(
        legacy, "load_inputs", lambda root: ([], {}, {}, {"aaaa": changed})
    )
    legacy.prepare()
    manifest = json.loads((legacy.DIRECTORY / "manifest.json").read_text())
    assert (
        manifest["cases"][0]["all_eligible_sources_in_retrieval_order"][0][
            "journal_entry"
        ]
        == changed[0]["initial_entry"]
    )


def test_valid_manifest_freezes_all_sources_without_reading_reserved_text(
    frozen,
    monkeypatch,
):
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path != frozen["reserved_path"], "Reserved text must remain unread"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    manifest = build(frozen)
    assert manifest["schema_version"] == "north-star-development-v2"
    assert manifest["case_count"] == 2
    assert manifest["cases"][0]["runtime_entry_ids"] == ["aaaa:entry:0", "aaaa:entry:1"]
    assert manifest["cases"][1]["case_categories"][0] == "no_earlier_writing"
    assert all(
        row["nudge_response"] is None
        for row in manifest["cases"][0]["all_eligible_sources_in_retrieval_order"]
    )
    assert set(frozen["result"]["sources"]) <= set(manifest["source_hashes"])
    assert manifest["source_hashes"]["report/retrieval.json"] == digest(
        frozen["retrieval_path"]
    )
    assert manifest["criteria"]["incorrect_displayed"] == 0
    assert not (frozen["root"] / "report/manifest.json").exists()


@pytest.mark.parametrize(
    "change", ["text", "missing_file", "missing_hash", "stale_hash"]
)
def test_source_integrity_fails_before_any_text_is_parsed(frozen, monkeypatch, change):
    path = frozen["development_path"]
    key = str(path.relative_to(frozen["root"]))
    if change == "text":
        path.write_text(path.read_text().replace("helped", "ignored"))
    elif change == "missing_file":
        path.unlink()
    elif change == "missing_hash":
        del frozen["result"]["sources"][key]
        save(frozen)
    else:
        frozen["result"]["sources"][key] = "0" * 64
        save(frozen)
    monkeypatch.setattr(
        inputs, "parse_wrangled_file", lambda path: pytest.fail("Parsed stale source")
    )
    with pytest.raises(ValueError, match="source|Source"):
        build(frozen)


@pytest.mark.parametrize("name", ["config", "cohort"])
@pytest.mark.parametrize("change", ["file", "embedded", "missing"])
def test_frozen_sidecars_are_bound_before_text_parsing(
    frozen, monkeypatch, name, change
):
    path = frozen["retrieval_path"].with_name(
        "retrieval_config.json" if name == "config" else "cohort.json"
    )
    if change == "file":
        path.write_text(path.read_text() + " ")
    elif change == "embedded":
        frozen["result"]["retrieval"][name]["unexpected"] = True
        save(frozen)
    else:
        path.unlink()
    monkeypatch.setattr(
        inputs, "parse_wrangled_file", lambda path: pytest.fail("Parsed stale source")
    )
    with pytest.raises(ValueError):
        build(frozen)


@pytest.mark.parametrize(
    "change", ["text", "date", "owner", "duplicate", "missing_text"]
)
def test_rehashed_source_still_must_match_frozen_rankings(frozen, change):
    path = frozen["development_path"]
    text = path.read_text()
    if change == "text":
        text = text.replace("helped", "ignored")
    elif change == "date":
        text = text.replace("2026-01-01", "2025-12-31")
    elif change == "owner":
        text = text.replace("# Persona aaaa:", "# Persona bbbb:")
    elif change == "duplicate":
        text = text.replace("## Entry 1", "## Entry 0")
    else:
        text = text.replace(frozen["entries"][0]["initial_entry"], "")
    path.write_text(text)
    save(frozen, rehash=path)
    with pytest.raises(ValueError):
        build(frozen)


def test_parser_metadata_must_match_source_owner(frozen, monkeypatch):
    original = inputs.parse_wrangled_file

    def wrong_owner(path):
        profile, entries, warnings = original(path)
        return {**profile, "persona_id": "bbbb"}, entries, warnings

    monkeypatch.setattr(inputs, "parse_wrangled_file", wrong_owner)
    with pytest.raises(ValueError, match="Parsed source owner"):
        build(frozen)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "owner",
        "t_index",
        "date",
        "hash",
        "order",
        "later",
    ],
)
def test_rejects_incomplete_or_stale_rankings(frozen, change):
    ranking = frozen["result"]["retrieval"]["cases"][0]["ranking"]
    if change == "missing":
        ranking.pop()
    elif change == "duplicate":
        ranking.append(deepcopy(ranking[0]))
    elif change == "owner":
        ranking[0]["entry_id"] = "bbbb:entry:0"
    elif change == "later":
        ranking[0]["entry_id"] = "aaaa:entry:4"
    elif change == "order":
        ranking.reverse()
    else:
        key, value = {
            "t_index": ("t_index", 1),
            "date": ("date", "2026-01-09"),
            "hash": ("source_sha256", "0" * 64),
        }[change]
        ranking[0][key] = value
    save(frozen)
    with pytest.raises(ValueError):
        build(frozen)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reserved", "episode"])
def test_rejects_missing_duplicate_or_stale_cases(frozen, change):
    cases = frozen["result"]["retrieval"]["cases"]
    if change == "missing":
        cases.pop()
    elif change == "duplicate":
        cases.append(deepcopy(cases[0]))
    elif change == "reserved":
        cases[0]["episode"] = frozen["episodes"][2]
    else:
        cases[0]["episode"]["onset_date"] = "2026-01-01"
    save(frozen)
    with pytest.raises(ValueError):
        build(frozen)


@pytest.mark.parametrize("change", ["onset_date", "onset_position", "cutoff"])
def test_actual_episode_chronology_must_match_writing(frozen, change):
    row = frozen["episodes"][0]
    if change == "cutoff":
        row.update(cutoff_t_index=1, cutoff_date="2026-01-02")
    else:
        row[change] = "2026-01-01" if change == "onset_date" else 1
    frozen["result"]["retrieval"]["cases"][0]["episode"] = row
    path = frozen["root"] / inputs.EPISODES
    pl.DataFrame(frozen["episodes"]).write_parquet(path)
    for case, source in zip(
        frozen["result"]["retrieval"]["cases"],
        pl.read_parquet(path).to_dicts()[:2],
        strict=True,
    ):
        case["episode"] = source
    cohort = frozen["result"]["retrieval"]["cohort"]
    cohort["source_episode_sha256"] = digest(path)
    cohort_path = frozen["retrieval_path"].with_name("cohort.json")
    write_json(cohort_path, cohort)
    frozen["result"]["retrieval"]["cohort_sha256"] = digest(cohort_path)
    save(frozen, rehash=path)
    with pytest.raises(ValueError, match="Episode"):
        build(frozen)


def test_duplicate_json_source_key_is_rejected(frozen):
    path = frozen["retrieval_path"]
    path.write_text(
        path.read_text().replace('"sources": {', '"sources": {}, "sources": {', 1)
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        build(frozen)


def test_matching_rehashed_config_cannot_change_the_frozen_model(frozen):
    retrieval = frozen["result"]["retrieval"]
    retrieval["config"]["model"] = "a-different-embedding-model"
    path = frozen["retrieval_path"].with_name("retrieval_config.json")
    write_json(path, retrieval["config"])
    retrieval["config_sha256"] = digest(path)
    save(frozen)
    with pytest.raises(ValueError, match="configuration differs"):
        build(frozen)


def test_missing_ranked_source_hash_rejects_with_validation_error(frozen):
    del frozen["result"]["retrieval"]["cases"][0]["ranking"][0]["source_sha256"]
    save(frozen)
    with pytest.raises(ValueError, match="Malformed frozen"):
        build(frozen)


def test_duplicate_actual_episode_is_rejected(frozen):
    path = frozen["root"] / inputs.EPISODES
    episodes = frozen["episodes"] + [deepcopy(frozen["episodes"][0])]
    pl.DataFrame(episodes).write_parquet(path)
    cohort = frozen["result"]["retrieval"]["cohort"]
    cohort["source_episode_sha256"] = digest(path)
    cohort_path = frozen["retrieval_path"].with_name("cohort.json")
    write_json(cohort_path, cohort)
    frozen["result"]["retrieval"]["cohort_sha256"] = digest(cohort_path)
    save(frozen, rehash=path)
    with pytest.raises(ValueError, match="Duplicate source development episode"):
        build(frozen)
