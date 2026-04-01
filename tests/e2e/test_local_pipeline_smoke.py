"""Deterministic local smoke test for the offline data pipeline."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.vif.train as train_module
from src.judge.consolidate import consolidate_judge_labels
from src.registry.personas import get_registry, get_status, register_persona
from src.wrangling.parse_synthetic_data import write_wrangled_markdown

VALUE_ORDER = [
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
]

PERSONAS = [
    {
        "persona_id": "a1b2c3d4",
        "name": "Maya Tan",
        "age": "25-34",
        "profession": "Product Manager",
        "culture": "Singaporean",
        "core_values": ["Achievement", "Benevolence"],
        "entries": [
            {
                "date": "2025-01-10",
                "initial_entry": (
                    "I stayed late to finish the launch deck, but I kept thinking about "
                    "how I bailed on dinner with my brother again."
                ),
                "nudge_text": "What part of that trade-off still bothers you?",
                "response_text": (
                    "Missing him bothered me more than the presentation, even though I "
                    "keep telling myself the launch matters more right now."
                ),
                "scores": {
                    "achievement": 1,
                    "benevolence": -1,
                },
            },
            {
                "date": "2025-01-14",
                "initial_entry": (
                    "I blocked the evening for family and actually kept the laptop shut. "
                    "It felt calmer than I expected."
                ),
                "nudge_text": None,
                "response_text": None,
                "scores": {
                    "benevolence": 1,
                    "achievement": 0,
                },
            },
        ],
    },
    {
        "persona_id": "b1c2d3e4",
        "name": "Jonah Lim",
        "age": "35-44",
        "profession": "Teacher",
        "culture": "Malaysian",
        "core_values": ["Security", "Tradition"],
        "entries": [
            {
                "date": "2025-01-09",
                "initial_entry": (
                    "I stuck to the savings plan and finally sorted the paperwork for my "
                    "parents' insurance. Boring, but worth it."
                ),
                "nudge_text": None,
                "response_text": None,
                "scores": {
                    "security": 1,
                    "tradition": 1,
                },
            },
            {
                "date": "2025-01-15",
                "initial_entry": (
                    "I almost used the emergency fund for a weekend splurge because I was "
                    "so tired of being careful."
                ),
                "nudge_text": "What stopped you from going through with it?",
                "response_text": (
                    "Mostly fear. I knew I'd hate myself on Monday if I pretended the "
                    "money wasn't meant for something real."
                ),
                "scores": {
                    "security": -1,
                },
            },
        ],
    },
    {
        "persona_id": "c1d2e3f4",
        "name": "Aisha Noor",
        "age": "25-34",
        "profession": "Designer",
        "culture": "Indonesian",
        "core_values": ["Self-Direction", "Stimulation"],
        "entries": [
            {
                "date": "2025-01-11",
                "initial_entry": (
                    "I pitched the weird campaign idea anyway. It might flop, but at "
                    "least it sounds like me."
                ),
                "nudge_text": "What made it feel worth the risk?",
                "response_text": (
                    "Because I am tired of polishing safe ideas just because everyone "
                    "already agrees with them."
                ),
                "scores": {
                    "self_direction": 1,
                    "stimulation": 1,
                },
            },
            {
                "date": "2025-01-18",
                "initial_entry": (
                    "I watered down the concept before review because I didn't want the "
                    "room to go quiet again."
                ),
                "nudge_text": None,
                "response_text": None,
                "scores": {
                    "self_direction": -1,
                },
            },
        ],
    },
    {
        "persona_id": "d1e2f3a4",
        "name": "Leo Hartono",
        "age": "45-54",
        "profession": "Operations Lead",
        "culture": "Filipino",
        "core_values": ["Power", "Conformity"],
        "entries": [
            {
                "date": "2025-01-12",
                "initial_entry": (
                    "I pushed hard in the meeting and got my way, but the team looked "
                    "flat by the end of it."
                ),
                "nudge_text": "Did getting the decision feel clean to you?",
                "response_text": (
                    "Not really. I got control, but it felt like everyone else just gave "
                    "up instead of buying in."
                ),
                "scores": {
                    "power": 1,
                    "conformity": -1,
                },
            },
            {
                "date": "2025-01-19",
                "initial_entry": (
                    "This week I held back and let the process run, even though part of "
                    "me wanted to jump in and force the pace."
                ),
                "nudge_text": None,
                "response_text": None,
                "scores": {
                    "conformity": 1,
                    "power": 0,
                },
            },
        ],
    },
]


class _ContentAwareMockTextEncoder:
    """Deterministic test encoder that keeps text differences visible."""

    embedding_dim = 8
    model_name = "content-aware-mock"

    def _encode_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()[: self.embedding_dim]
        return [byte / 255.0 for byte in digest]

    def encode(self, texts: list[str]):
        return self.encode_batch(texts)

    def encode_batch(self, texts: list[str], batch_size: int = 32):  # noqa: ARG002
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.asarray([self._encode_text(text) for text in texts], dtype=np.float32)


def _full_scores(overrides: dict[str, int]) -> dict[str, int]:
    scores = {value_name: 0 for value_name in VALUE_ORDER}
    scores.update(overrides)
    return scores


def _build_rationales(scores: dict[str, int]) -> dict[str, str] | None:
    rationales = {
        value_name: f"Fixture rationale for {value_name}={score}"
        for value_name, score in scores.items()
        if score != 0
    }
    return rationales or None


def _render_raw_persona_markdown(persona: dict) -> str:
    lines = [
        f"# Persona {persona['persona_id']}: {persona['name']}",
        "",
        "## Profile",
        f"- Age: {persona['age']}",
        f"- Profession: {persona['profession']}",
        f"- Culture: {persona['culture']}",
        f"- Core Values: {', '.join(persona['core_values'])}",
        f"- Bio: {persona['name']} is a fixture persona for the offline smoke test.",
        "",
        "---",
        "",
    ]

    for index, entry in enumerate(persona["entries"], start=1):
        lines.extend(
            [
                f"## Entry {index} - {entry['date']}",
                "",
                "### Initial Entry",
                "**Tone:** reflective",
                "**Verbosity:** Medium",
                "**Reflection Mode:** Neutral",
                "",
                entry["initial_entry"],
                "",
            ]
        )

        if entry["nudge_text"] is None:
            lines.extend(
                [
                    "*(No nudge for this entry)*",
                    "",
                    "---",
                    "",
                ]
            )
            continue

        lines.extend(
            [
                "### Nudge",
                "**Trigger:** fixture trigger",
                "",
                f"\"{entry['nudge_text']}\"",
                "",
            ]
        )

        if entry["response_text"] is None:
            lines.extend(
                [
                    "*(No response - persona did not reply to nudge)*",
                    "",
                    "---",
                    "",
                ]
            )
            continue

        lines.extend(
            [
                "### Response",
                "**Mode:** thoughtful",
                "",
                entry["response_text"],
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines)


def _build_label_payload(persona: dict) -> dict:
    labels = []
    for t_index, entry in enumerate(persona["entries"]):
        scores = _full_scores(entry["scores"])
        labels.append(
            {
                "t_index": t_index,
                "date": entry["date"],
                "scores": scores,
                "rationales": _build_rationales(scores),
            }
        )

    return {
        "persona_id": persona["persona_id"],
        "labels": labels,
    }


def _write_holdout_manifest(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "val_persona_ids:",
                "  - c1d2e3f4",
                "test_persona_ids:",
                "  - d1e2f3a4",
                "",
            ]
        )
    )


def test_local_pipeline_smoke(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    raw_dir = logs_dir / "synthetic_data"
    wrangled_dir = logs_dir / "wrangled"
    labels_dir = logs_dir / "judge_labels"
    output_dir = tmp_path / "models" / "vif" / "e2e_smoke"

    raw_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "src.registry.personas.REGISTRY_PATH",
        logs_dir / "registry" / "personas.parquet",
    )

    for persona in PERSONAS:
        (raw_dir / f"persona_{persona['persona_id']}.md").write_text(
            _render_raw_persona_markdown(persona)
        )
        (labels_dir / f"persona_{persona['persona_id']}_labels.json").write_text(
            json.dumps(_build_label_payload(persona), indent=2)
        )
        register_persona(
            persona_id=persona["persona_id"],
            name=persona["name"],
            age=persona["age"],
            profession=persona["profession"],
            culture=persona["culture"],
            core_values=persona["core_values"],
            entry_count=len(persona["entries"]),
        )

    written_files = write_wrangled_markdown(raw_dir, wrangled_dir, update_registry=True)

    assert len(written_files) == len(PERSONAS)
    wrangled_sample = (wrangled_dir / "persona_a1b2c3d4.md").read_text()
    assert "### Initial Entry" not in wrangled_sample
    assert '**Nudge:** "What part of that trade-off still bothers you?"' in wrangled_sample

    status_after_wrangling = get_status()
    assert status_after_wrangling == {
        "total": 4,
        "synthetic": 4,
        "wrangled": 4,
        "labeled": 0,
        "pending_wrangling": 0,
        "pending_labeling": 4,
    }

    labels_path = labels_dir / "judge_labels.parquet"
    labels_df, errors = consolidate_judge_labels(
        labels_dir=labels_dir,
        output_path=labels_path,
        update_registry=True,
    )

    assert errors == []
    assert labels_df.height == 8
    assert labels_df["persona_id"].n_unique() == 4
    assert labels_path.is_file()

    registry_df = get_registry()
    assert registry_df["stage_wrangled"].all()
    assert registry_df["stage_labeled"].all()

    holdout_manifest = tmp_path / "holdout.yaml"
    _write_holdout_manifest(holdout_manifest)

    config = train_module.load_config(None)
    config["encoder"]["model_name"] = "content-aware-mock"
    config["state_encoder"]["window_size"] = 1
    config["model"]["hidden_dim"] = 8
    config["model"]["dropout"] = 0.0
    config["training"]["epochs"] = 2
    config["training"]["batch_size"] = 2
    config["training"]["learning_rate"] = 0.01
    config["training"]["weight_decay"] = 0.0
    config["training"]["early_stopping"]["patience"] = 2
    config["training"]["early_stopping"]["min_delta"] = 0.0
    config["training"]["scheduler"]["patience"] = 1
    config["training"]["lr_finder"]["enabled"] = False
    config["mc_dropout"]["n_samples"] = 2
    config["data"]["labels_path"] = str(labels_path)
    config["data"]["wrangled_dir"] = str(wrangled_dir)
    config["data"]["fixed_holdout_manifest_path"] = str(holdout_manifest)
    config["output"]["checkpoint_dir"] = str(output_dir)
    config["output"]["log_dir"] = str(output_dir)

    monkeypatch.setattr(
        train_module,
        "create_encoder",
        lambda *_args, **_kwargs: _ContentAwareMockTextEncoder(),
    )

    results = train_module.train(config, verbose=False)

    assert (output_dir / "best_model.pt").is_file()
    assert (output_dir / "best_model_config.json").is_file()
    assert (output_dir / "training_log.json").is_file()
    assert (output_dir / "training_curves.json").is_file()
    assert results["best_val_loss"] >= 0.0
    assert results["test_results"]["mse_mean"] >= 0.0

    training_log = json.loads((output_dir / "training_log.json").read_text())
    assert training_log["config"]["data"]["labels_path"] == str(labels_path)
    assert training_log["config"]["data"]["wrangled_dir"] == str(wrangled_dir)
    assert training_log["test_metrics"] is not None
