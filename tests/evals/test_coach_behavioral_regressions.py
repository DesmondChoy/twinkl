"""Author examples for deterministic checks and optional Coach Digest Evals.

Semantic counterexamples are review inputs, not measured model failures. These
tests do not require the mechanical validator to accept any semantic error.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from src.coach.weekly_digest import (
    render_digest_messages,
    validate_weekly_digest_narrative,
)
from src.evals import coach_narrative_judge as judge
from src.prompt_boundary import UNTRUSTED_DATA_RULE

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "config/evals/coach_behavioral_regressions_v1.json"
)
ITEMS = json.loads(MANIFEST.read_text())
PAIRS = judge._load_manifest(MANIFEST)
CASES = {
    (item["provenance"]["pair_id"], item["provenance"]["variant"]): pair
    for item, pair in zip(ITEMS, PAIRS, strict=True)
}


def test_corpus_has_paired_sources_and_explicit_author_provenance():
    by_pair = defaultdict(list)
    for item in ITEMS:
        assert set(item) == {"digest", "narrative", "provenance"}
        provenance = item["provenance"]
        assert provenance["source"] == "synthetic_author_example"
        assert "Not sampled model outputs" in provenance["measurement_status"]
        assert "human validation" in provenance["measurement_status"]
        by_pair[provenance["pair_id"]].append(item)
    assert len(by_pair) == 8
    for items in by_pair.values():
        assert {item["provenance"]["variant"] for item in items} == {"good", "bad"}
        # Distinct IDs keep the existing CLI's per-response labels unambiguous.
        digests = [
            {key: value for key, value in item["digest"].items() if key != "persona_id"}
            for item in items
        ]
        assert digests[0] == digests[1]
    assert len(set(judge._load_sample_labels(MANIFEST))) == 16
    assert judge._load_generator_model(MANIFEST) is None


@pytest.mark.parametrize("pair_id", sorted({key[0] for key in CASES}))
def test_good_controls_pass_current_mechanical_validations(pair_id):
    digest, narrative = CASES[pair_id, "good"]
    result = validate_weekly_digest_narrative(digest, narrative, validate_voice=True)
    assert all(check.passed for check in result.checks), result.model_dump()


@pytest.mark.parametrize(
    ("pair_id", "failed_check"),
    [
        ("mixed_fabricated_quote", "all_quotes_grounded"),
        ("blank_question", "reflective_question_form"),
        ("statement_question", "reflective_question_form"),
    ],
)
def test_mechanical_counterexamples_are_rejected(pair_id, failed_check):
    digest, narrative = CASES[pair_id, "bad"]
    result = validate_weekly_digest_narrative(digest, narrative, validate_voice=True)
    checks = {check.name: check.passed for check in result.checks}
    assert checks[failed_check] is False
    if pair_id == "mixed_fabricated_quote":
        assert checks["groundedness"] is True  # One real quote cannot excuse another.


@pytest.mark.parametrize("index", range(len(ITEMS)))
def test_judge_sees_complete_sources_without_author_expectations(index):
    item = ITEMS[index]
    digest, narrative = PAIRS[index]
    prompt = judge.render_judge_prompt(digest, narrative)
    for evidence in digest.evidence:
        assert json.dumps(evidence.excerpt, ensure_ascii=False) in prompt
    for field in narrative.model_dump().values():
        if field:
            assert field in prompt
    provenance = item["provenance"]
    for key in ("expectation_marker", "expected_behavior", "expected_quality"):
        assert provenance[key] not in prompt
    assert digest.persona_id not in prompt


def test_intention_case_keeps_late_negation_in_judge_context():
    digest, narrative = CASES["intention_not_completed_action", "bad"]
    source = digest.evidence[0].excerpt
    assert source.index("I did not make the call.") > 250
    assert source in judge.render_judge_prompt(digest, narrative)


def test_journal_instructions_remain_untrusted_generation_data():
    digest, _ = CASES["instruction_like_journal_entry", "good"]
    instructions, input_data = render_digest_messages(digest)
    sentinel = "INSTRUCTION_SENTINEL_COACH_2026"
    assert sentinel not in instructions
    assert UNTRUSTED_DATA_RULE in instructions
    assert sentinel in input_data
    assert json.loads(input_data)["evidence_lines"]
    ordinary_digest, _ = CASES["mixed_fabricated_quote", "good"]
    assert instructions == render_digest_messages(ordinary_digest)[0]


def test_existing_judge_cli_dry_run_does_not_build_a_provider(monkeypatch, capsys):
    def unexpected_provider(**_kwargs):
        raise AssertionError("A dry run must not construct a provider")

    monkeypatch.setattr(judge, "build_llm_complete", unexpected_provider)
    assert judge.main(["--manifest", str(MANIFEST)]) == 0
    output = capsys.readouterr().out
    assert "[dry run] Would evaluate 16 Coach Digest response(s)" in output
