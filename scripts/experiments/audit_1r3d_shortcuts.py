#!/usr/bin/env python3
"""Audit run_020 for single-word shortcut sensitivity on two strong dimensions.

This is a diagnostic, not a promotion evaluation. It selects the most confident
correct non-neutral validation predictions for Conformity and Self-Direction,
removes one content-word occurrence at a time, and measures the deterministic
change in the checkpoint's target-class probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import polars as pl
import torch

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.runtime import _load_persona_entries, load_runtime_bundle
from src.vif.state_encoder import concatenate_entry_text, core_values_to_profile_weights

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / (
    "logs/experiments/artifacts/ordinal_v4_s2025_m22_20260307_152002/"
    "BalancedSoftmax/selected_checkpoint.pt"
)
DEFAULT_OUTPUTS = DEFAULT_CHECKPOINT.with_name("selected_validation_outputs.parquet")
DEFAULT_LABELS = REPO_ROOT / "logs/judge_labels/judge_labels.parquet"
DEFAULT_WRANGLED = REPO_ROOT / "logs/wrangled"
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "logs/experiments/artifacts/twinkl_1r3d_shortcut_audit_20260712"
)
DIMENSIONS = ("conformity", "self_direction")

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "into",
    "its",
    "just",
    "like",
    "more",
    "most",
    "not",
    "now",
    "our",
    "ours",
    "out",
    "over",
    "really",
    "said",
    "she",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "too",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "response",
    "nudge",
}

# These are deliberately narrow, theory-motivated cue families. The exhaustive
# word-removal pass remains primary; this lexicon only distinguishes the
# suspected surface cues from other influential words.
CUE_LEXICONS = {
    "conformity": {
        "accept",
        "accepted",
        "approve",
        "approved",
        "authority",
        "book",
        "compliant",
        "duty",
        "expected",
        "expectation",
        "family",
        "follow",
        "followed",
        "following",
        "permission",
        "policy",
        "proper",
        "protocol",
        "must",
        "obey",
        "obeyed",
        "respect",
        "rule",
        "rules",
        "should",
        "supposed",
        "tradition",
        "traditional",
    },
    "self_direction": {
        "ask",
        "asked",
        "autonomy",
        "choice",
        "choose",
        "chose",
        "control",
        "decide",
        "decided",
        "decision",
        "freedom",
        "independence",
        "independent",
        "independently",
        "own",
        "path",
        "preference",
        "preferences",
        "say",
        "voice",
        "want",
        "wanted",
    },
}
PHRASE_CUES = {
    "conformity": (
        "by the book",
        "follow the rules",
        "followed protocol",
        "gave in",
        "go along",
        "kept quiet",
        "make waves",
        "supposed to",
        "went along",
    ),
    "self_direction": (
        "didn't ask",
        "keep quiet",
        "kept quiet",
        "my own",
        "never say",
        "own choice",
        "speak up",
        "waiting for permission",
        "what i wanted",
    ),
}


def normalize_word(word: str) -> str:
    """Normalize an extracted word for matching and stable artifact keys."""
    return word.lower().replace("’", "'").strip("'-")


def content_words(text: str) -> list[str]:
    """Return unique content words in first-occurrence order."""
    words: list[str] = []
    seen: set[str] = set()
    for match in WORD_PATTERN.finditer(text):
        word = normalize_word(match.group())
        if word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def content_word_occurrences(text: str) -> list[tuple[str, int, int]]:
    """Return content-word occurrences with source spans, preserving repeats."""
    return [
        (normalize_word(match.group()), match.start(), match.end())
        for match in WORD_PATTERN.finditer(text)
        if normalize_word(match.group()) not in STOPWORDS
    ]


def _clean_removed_text(text: str) -> str:
    """Normalize spacing after a bounded deletion without rewriting content."""
    perturbed = text
    perturbed = re.sub(r"[ \t]{2,}", " ", perturbed)
    perturbed = re.sub(r" +([,.;:!?])", r"\1", perturbed)
    return perturbed.strip()


def remove_word_occurrence(text: str, start: int, end: int) -> str:
    """Remove one word occurrence selected by its exact source span."""
    return _clean_removed_text(text[:start] + text[end:])


def remove_all_word_occurrences(text: str, word: str) -> str:
    """Remove every whole-word occurrence of one normalized candidate cue."""
    pattern = re.compile(
        rf"(?<![A-Za-z'’-]){re.escape(word)}(?![A-Za-z'’-])",
        flags=re.IGNORECASE,
    )
    return _clean_removed_text(pattern.sub("", text))


def remove_phrase(text: str, phrase: str) -> str:
    """Remove all case-insensitive occurrences of one literal cue phrase."""
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    pattern = re.compile(
        rf"(?<![A-Za-z'’-]){escaped}(?![A-Za-z'’-])",
        flags=re.IGNORECASE,
    )
    return _clean_removed_text(pattern.sub("", text))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a repo-relative artifact path, falling back to an absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _select_cases(outputs: pl.DataFrame, top_k_per_target: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for target in (-1, 1):
            rows = (
                outputs.filter(
                    (pl.col("dimension") == dimension)
                    & (pl.col("target") == target)
                    & (pl.col("target") == pl.col("predicted_class"))
                )
                .with_columns(
                    pl.struct(["class_probabilities", "target"])
                    .map_elements(
                        lambda row: float(
                            row["class_probabilities"][row["target"] + 1]
                        ),
                        return_dtype=pl.Float64,
                    )
                    .alias("saved_target_probability")
                )
                .sort(
                    ["saved_target_probability", "persona_id", "t_index"],
                    descending=[True, False, False],
                )
                .head(top_k_per_target)
            )
            if rows.is_empty():
                raise ValueError(
                    f"No correct active cases for {dimension} target {target}."
                )
            selected.extend(rows.to_dicts())
    return selected


def _predict_probabilities(
    model: torch.nn.Module,
    states: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            tensor = torch.from_numpy(states[start : start + batch_size]).to(device)
            batches.append(model.predict_probabilities(tensor).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    outputs = pl.read_parquet(args.outputs)
    labels = pl.read_parquet(args.labels).select(
        [
            "persona_id",
            "t_index",
            "rationales_json",
            "alignment_conformity",
            "alignment_self_direction",
        ]
    )
    label_lookup = {
        (str(row["persona_id"]), int(row["t_index"])): row
        for row in labels.iter_rows(named=True)
    }
    cases = _select_cases(outputs, args.top_k_per_target)

    model, state_encoder, _config, _checkpoint, device = load_runtime_bundle(
        args.checkpoint,
        device=args.device,
    )
    if state_encoder.window_size != 1 or state_encoder.history_pooling != "none":
        raise ValueError("twinkl-1r3d requires the exact run_020 window_size=1 state.")

    texts: list[str] = []
    profiles: list[np.ndarray] = []
    request_metadata: list[dict[str, Any]] = []
    persona_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}

    for case_index, case in enumerate(cases):
        persona_id = str(case["persona_id"])
        t_index = int(case["t_index"])
        if persona_id not in persona_cache:
            persona_cache[persona_id] = _load_persona_entries(persona_id, args.wrangled)
        profile, entries = persona_cache[persona_id]
        entry = next((row for row in entries if int(row["t_index"]) == t_index), None)
        if entry is None:
            raise ValueError(f"Missing wrangled entry {(persona_id, t_index)}")
        label_row = label_lookup.get((persona_id, t_index))
        if label_row is None:
            raise ValueError(f"Missing current label {(persona_id, t_index)}")
        dimension = str(case["dimension"])
        current_target = int(label_row[f"alignment_{dimension}"])
        if current_target != int(case["target"]):
            raise ValueError(
                f"Current label {current_target} != saved target {case['target']} "
                f"for {(persona_id, t_index, dimension)}"
            )
        text = concatenate_entry_text(
            entry.get("initial_entry"),
            entry.get("nudge_text"),
            entry.get("response_text"),
        )
        rationale = (
            json.loads(label_row["rationales_json"] or "{}").get(dimension, "") or ""
        )
        rationale_words = set(content_words(rationale))
        profile_weights = core_values_to_profile_weights(
            profile.get("core_values") or []
        )

        texts.append(text)
        profiles.append(profile_weights)
        request_metadata.append(
            {
                "case_index": case_index,
                "perturbation_type": "baseline",
                "word": None,
                "rationale_words": rationale_words,
                "text": text,
                "rationale": rationale,
            }
        )
        for occurrence_index, (word, start, end) in enumerate(
            content_word_occurrences(text)
        ):
            texts.append(remove_word_occurrence(text, start, end))
            profiles.append(profile_weights)
            request_metadata.append(
                {
                    "case_index": case_index,
                    "perturbation_type": "single_word_occurrence",
                    "word": word,
                    "occurrence_index": occurrence_index,
                    "context": text[max(0, start - 55) : min(len(text), end + 55)],
                    "rationale_words": rationale_words,
                    "text": text,
                    "rationale": rationale,
                }
            )

        occurrence_counts = Counter(
            word for word, _start, _end in content_word_occurrences(text)
        )
        repeated_candidate_words = sorted(
            word
            for word, count in occurrence_counts.items()
            if count > 1 and word in CUE_LEXICONS[dimension]
        )
        for word in repeated_candidate_words:
            texts.append(remove_all_word_occurrences(text, word))
            profiles.append(profile_weights)
            request_metadata.append(
                {
                    "case_index": case_index,
                    "perturbation_type": "all_candidate_word_occurrences",
                    "word": word,
                    "rationale_words": rationale_words,
                    "text": text,
                    "rationale": rationale,
                }
            )

        for phrase in PHRASE_CUES[dimension]:
            perturbed = remove_phrase(text, phrase)
            if perturbed == text:
                continue
            texts.append(perturbed)
            profiles.append(profile_weights)
            request_metadata.append(
                {
                    "case_index": case_index,
                    "perturbation_type": "candidate_phrase",
                    "word": phrase,
                    "rationale_words": rationale_words,
                    "text": text,
                    "rationale": rationale,
                }
            )

    embeddings = state_encoder.text_encoder.encode_batch(
        texts, batch_size=args.encode_batch_size
    )
    states = np.concatenate([embeddings, np.stack(profiles)], axis=1).astype(np.float32)
    if states.shape[1] != state_encoder.state_dim:
        raise ValueError(
            f"Rebuilt state width {states.shape[1]} != checkpoint width "
            f"{state_encoder.state_dim}."
        )
    probabilities = _predict_probabilities(
        model, states, batch_size=args.model_batch_size, device=device
    )

    baseline_by_case: dict[int, dict[str, Any]] = {}
    token_rows: list[dict[str, Any]] = []
    grouped_cue_rows: list[dict[str, Any]] = []
    for metadata, probability_vector in zip(
        request_metadata, probabilities, strict=True
    ):
        case_index = int(metadata["case_index"])
        case = cases[case_index]
        dimension = str(case["dimension"])
        dim_index = SCHWARTZ_VALUE_ORDER.index(dimension)
        target = int(case["target"])
        class_probs = probability_vector[dim_index]
        target_probability = float(class_probs[target + 1])
        predicted_class = int(class_probs.argmax()) - 1
        word = metadata["word"]
        perturbation_type = str(metadata["perturbation_type"])
        if perturbation_type == "baseline":
            saved_probability = float(case["saved_target_probability"])
            if abs(target_probability - saved_probability) > 1e-5:
                raise ValueError(
                    "Baseline probability mismatch for "
                    f"{(case['persona_id'], case['t_index'], dimension)}: "
                    f"rebuilt={target_probability:.8f}, saved={saved_probability:.8f}"
                )
            baseline_by_case[case_index] = {
                "probability": target_probability,
                "predicted_class": predicted_class,
            }
            continue

        baseline = baseline_by_case[case_index]
        if perturbation_type != "single_word_occurrence":
            grouped_cue_rows.append(
                {
                    "case_index": case_index,
                    "persona_id": str(case["persona_id"]),
                    "t_index": int(case["t_index"]),
                    "dimension": dimension,
                    "target": target,
                    "baseline_target_probability": baseline["probability"],
                    "perturbation_type": perturbation_type,
                    "removed_cue": str(word),
                    "cue_appears_in_judge_rationale": all(
                        cue_word in metadata["rationale_words"]
                        for cue_word in content_words(str(word))
                    ),
                    "perturbed_target_probability": target_probability,
                    "target_probability_drop": (
                        baseline["probability"] - target_probability
                    ),
                    "perturbed_prediction": predicted_class,
                    "prediction_flipped": predicted_class != target,
                }
            )
            continue

        token_rows.append(
            {
                "case_index": case_index,
                "persona_id": str(case["persona_id"]),
                "t_index": int(case["t_index"]),
                "dimension": dimension,
                "target": target,
                "baseline_target_probability": baseline["probability"],
                "removed_word": word,
                "occurrence_index": int(metadata["occurrence_index"]),
                "source_context": str(metadata["context"]),
                "is_candidate_cue": word in CUE_LEXICONS[dimension],
                "appears_in_judge_rationale": word in metadata["rationale_words"],
                "perturbed_target_probability": target_probability,
                "target_probability_drop": baseline["probability"] - target_probability,
                "perturbed_prediction": predicted_class,
                "prediction_flipped": predicted_class != target,
            }
        )

    token_frame = pl.DataFrame(token_rows).sort(
        ["dimension", "case_index", "target_probability_drop"],
        descending=[False, False, True],
    )
    grouped_cue_frame = pl.DataFrame(grouped_cue_rows).sort(
        ["dimension", "case_index", "target_probability_drop"],
        descending=[False, False, True],
    )
    case_summaries: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        rows = token_frame.filter(pl.col("case_index") == case_index)
        positive_rows = rows.filter(pl.col("target_probability_drop") > 0)
        cue_rows = rows.filter(pl.col("is_candidate_cue"))
        grouped_rows = grouped_cue_frame.filter(pl.col("case_index") == case_index)
        noncue_drops = (
            rows.filter(~pl.col("is_candidate_cue"))
            .get_column("target_probability_drop")
            .to_list()
        )
        top = rows.row(0, named=True)
        cue_top = cue_rows.row(0, named=True) if cue_rows.height else None
        rationale = request_metadata[
            next(
                index
                for index, metadata in enumerate(request_metadata)
                if metadata["case_index"] == case_index and metadata["word"] is None
            )
        ]["rationale"]
        case_summaries.append(
            {
                "case_index": case_index,
                "persona_id": str(case["persona_id"]),
                "t_index": int(case["t_index"]),
                "dimension": str(case["dimension"]),
                "target": int(case["target"]),
                "baseline_target_probability": baseline_by_case[case_index][
                    "probability"
                ],
                "words_tested": rows.height,
                "candidate_cues_present": cue_rows.height,
                "any_single_word_flip": bool(rows["prediction_flipped"].any()),
                "any_candidate_cue_flip": bool(
                    cue_rows["prediction_flipped"].any() if cue_rows.height else False
                ),
                "grouped_cue_perturbations": grouped_rows.height,
                "any_grouped_cue_flip": bool(
                    grouped_rows["prediction_flipped"].any()
                    if grouped_rows.height
                    else False
                ),
                "max_grouped_cue_drop": (
                    float(grouped_rows["target_probability_drop"].max())
                    if grouped_rows.height
                    else None
                ),
                "max_probability_drop": float(top["target_probability_drop"]),
                "most_influential_word": str(top["removed_word"]),
                "top_word_is_candidate_cue": bool(top["is_candidate_cue"]),
                "top_word_appears_in_rationale": bool(
                    top["appears_in_judge_rationale"]
                ),
                "max_candidate_cue_drop": (
                    float(cue_top["target_probability_drop"]) if cue_top else None
                ),
                "most_influential_candidate_cue": (
                    str(cue_top["removed_word"]) if cue_top else None
                ),
                "median_noncue_drop": _median_or_none(noncue_drops),
                "positive_drop_word_count": positive_rows.height,
                "judge_rationale": str(rationale),
                "runtime_text_sha256": _sha256_text(
                    str(
                        next(
                            metadata["text"]
                            for metadata in request_metadata
                            if metadata["case_index"] == case_index
                            and metadata["perturbation_type"] == "baseline"
                        )
                    )
                ),
                "judge_rationale_sha256": _sha256_text(str(rationale)),
            }
        )

    aggregate: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        rows = [row for row in case_summaries if row["dimension"] == dimension]
        with_cues = [row for row in rows if row["candidate_cues_present"] > 0]
        by_target: dict[str, Any] = {}
        for target in (-1, 1):
            target_rows = [row for row in rows if row["target"] == target]
            by_target[str(target)] = {
                "cases": len(target_rows),
                "median_baseline_target_probability": _median_or_none(
                    [row["baseline_target_probability"] for row in target_rows]
                ),
                "cases_with_any_single_word_flip": sum(
                    row["any_single_word_flip"] for row in target_rows
                ),
                "median_max_probability_drop": _median_or_none(
                    [row["max_probability_drop"] for row in target_rows]
                ),
            }
        aggregate[dimension] = {
            "cases": len(rows),
            "negative_targets": sum(row["target"] == -1 for row in rows),
            "positive_targets": sum(row["target"] == 1 for row in rows),
            "median_baseline_target_probability": _median_or_none(
                [row["baseline_target_probability"] for row in rows]
            ),
            "cases_with_any_single_word_flip": sum(
                row["any_single_word_flip"] for row in rows
            ),
            "cases_with_candidate_cues": len(with_cues),
            "cases_with_candidate_cue_flip": sum(
                row["any_candidate_cue_flip"] for row in with_cues
            ),
            "grouped_cue_perturbations": sum(
                row["grouped_cue_perturbations"] for row in rows
            ),
            "cases_with_grouped_cue_flip": sum(
                row["any_grouped_cue_flip"] for row in rows
            ),
            "top_word_is_candidate_cue": sum(
                row["top_word_is_candidate_cue"] for row in rows
            ),
            "top_word_appears_in_rationale": sum(
                row["top_word_appears_in_rationale"] for row in rows
            ),
            "median_max_probability_drop": _median_or_none(
                [row["max_probability_drop"] for row in rows]
            ),
            "by_target": by_target,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "token_perturbations.parquet"
    grouped_cue_path = output_dir / "grouped_cue_perturbations.parquet"
    summary_path = output_dir / "audit_summary.json"
    token_frame.write_parquet(token_path)
    grouped_cue_frame.write_parquet(grouped_cue_path)
    selected_case_manifest = [
        {
            key: row[key]
            for key in (
                "persona_id",
                "t_index",
                "dimension",
                "target",
                "runtime_text_sha256",
                "judge_rationale_sha256",
            )
        }
        for row in case_summaries
    ]
    payload = {
        "audit_id": "twinkl-1r3d",
        "scope": {
            "split": "validation",
            "dimensions": list(DIMENSIONS),
            "selection": (
                "top confident correct non-neutral predictions per dimension and target"
            ),
            "top_k_per_dimension_and_target": args.top_k_per_target,
            "perturbation": "remove one whole content-word occurrence at a time",
            "grouped_cue_perturbation": (
                "remove repeated candidate words and candidate phrases in full"
            ),
            "inference": "deterministic checkpoint probabilities; dropout disabled",
        },
        "provenance": {
            "checkpoint": _display_path(Path(args.checkpoint)),
            "checkpoint_sha256": _sha256(Path(args.checkpoint)),
            "outputs": _display_path(Path(args.outputs)),
            "outputs_sha256": _sha256(Path(args.outputs)),
            "labels": _display_path(Path(args.labels)),
            "labels_sha256": _sha256(Path(args.labels)),
        },
        "aggregate": aggregate,
        "selected_case_manifest_sha256": _sha256_text(
            json.dumps(selected_case_manifest, sort_keys=True, separators=(",", ":"))
        ),
        "cases": case_summaries,
        "artifacts": {
            "token_perturbations": _display_path(token_path),
            "grouped_cue_perturbations": _display_path(grouped_cue_path),
        },
        "limitations": [
            "Single-word deletion changes both meaning and syntax; sensitivity is not "
            "proof of a shortcut.",
            "Exact word overlap with a judge rationale is a conservative proxy for "
            "shared evidence.",
            "The deliberately narrow cue lexicons do not cover every possible "
            "surface shortcut.",
            "Repeated-word and phrase removal covers pre-registered cue families, "
            "not arbitrary multiword feature interactions.",
            "The selected confident-correct validation cases diagnose model behavior "
            "but do not estimate population performance.",
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--wrangled", type=Path, default=DEFAULT_WRANGLED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k-per-target", type=int, default=10)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    result = run_audit(parse_args())
    print(json.dumps(result["aggregate"], indent=2))
