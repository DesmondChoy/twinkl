"""Focused tests for the deterministic twinkl-1r3d text perturbations."""

import importlib.util
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SCRIPT = REPO_ROOT / "scripts/experiments/audit_1r3d_shortcuts.py"
SPEC = importlib.util.spec_from_file_location("audit_1r3d_shortcuts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
content_word_occurrences = MODULE.content_word_occurrences
content_words = MODULE.content_words
remove_word_occurrence = MODULE.remove_word_occurrence
remove_all_word_occurrences = MODULE.remove_all_word_occurrences
remove_phrase = MODULE.remove_phrase
select_cases = MODULE._select_cases
display_path = MODULE._display_path


def test_content_words_normalizes_and_deduplicates_in_order() -> None:
    text = "Protocol, protocol! She followed the rules because rules mattered."

    assert content_words(text) == ["protocol", "followed", "rules", "mattered"]


def test_content_word_occurrences_preserves_repeated_words_and_spans() -> None:
    text = "I followed protocol. The Protocol mattered; protocol-like did too."

    assert content_word_occurrences(text) == [
        ("followed", 2, 10),
        ("protocol", 11, 19),
        ("protocol", 25, 33),
        ("mattered", 34, 42),
        ("protocol-like", 44, 57),
    ]


def test_remove_word_occurrence_removes_only_selected_span() -> None:
    text = "Protocol first, protocol second."

    assert remove_word_occurrence(text, 0, 8) == "first, protocol second."


def test_grouped_cue_removal_handles_repeats_and_phrases() -> None:
    assert (
        remove_all_word_occurrences("Protocol first, protocol second.", "protocol")
        == "first, second."
    )
    assert remove_phrase("I kept quiet, then KEPT QUIET again.", "kept quiet") == (
        "I, then again."
    )


def test_select_cases_stratifies_by_dimension_and_target() -> None:
    rows = []
    for dimension in ("conformity", "self_direction"):
        for target in (-1, 1):
            for rank, confidence in enumerate((0.9, 0.8, 0.7)):
                probabilities = [0.05, 0.05, 0.05]
                probabilities[target + 1] = confidence
                rows.append(
                    {
                        "persona_id": f"{dimension}-{target}-{rank}",
                        "t_index": 0,
                        "dimension": dimension,
                        "target": target,
                        "predicted_class": target,
                        "class_probabilities": probabilities,
                    }
                )

    selected = select_cases(pl.DataFrame(rows), top_k_per_target=2)

    assert len(selected) == 8
    assert {
        (row["dimension"], row["target"], row["saved_target_probability"])
        for row in selected
    } == {
        (dimension, target, confidence)
        for dimension in ("conformity", "self_direction")
        for target in (-1, 1)
        for confidence in (0.9, 0.8)
    }


def test_display_path_handles_repo_and_external_paths(tmp_path: Path) -> None:
    assert display_path(REPO_ROOT / "logs/example.json") == "logs/example.json"
    assert display_path(tmp_path / "example.json") == str(
        (tmp_path / "example.json").resolve()
    )
