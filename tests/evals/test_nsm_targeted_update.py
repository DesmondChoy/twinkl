"""Conservative evidence reuse for the separately versioned NSM update."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.experiments import nsm_experiment as experiment
from scripts.experiments import nsm_targeted_update as update
from scripts.experiments.nsm_provider import DEFAULT_POLICY
from src.north_star.provider import openai_input_payload, stable_hash
from tests.evals.test_nsm_evaluation import ACTION, batch
from tests.evals.test_nsm_experiment import small_case
from tests.evals.test_nsm_provider import Harness, counted, generated, request, validate
from tests.historical import assert_in_snapshot, source_snapshot


def source_request(item, purpose="runtime:nomic"):
    value = item["values"][0]
    ids = [source["entry_id"] for source in value["sources"]]
    return experiment.source_request(item, value, ids, purpose)


def terminal_receipt(frozen, *, policy=None, max_attempts=2):
    """A terminal provider failure with complete request/count identities."""
    policy = policy or DEFAULT_POLICY
    payload = openai_input_payload(frozen, policy)
    return {
        "request": deepcopy(frozen),
        "request_hash": stable_hash(frozen),
        "payload_hash": stable_hash(payload),
        "policy_hash": stable_hash(policy),
        "max_attempts": max_attempts,
        "status": "failed",
        "count_attempts": [],
        "count_receipt": {
            "request_hash": stable_hash(frozen),
            "payload_hash": stable_hash(payload),
            "model": payload["model"],
            "input_tokens": 125,
        },
        "attempts": [{"status": "failed", "retryable": False}],
        "result": None,
    }


def assert_not_reusable(frozen, receipt, policy=None, max_attempts=2):
    """Both incompatibility and corrupt matching evidence must fail closed."""
    try:
        reusable = update.compatible_receipt(
            frozen, receipt, policy or DEFAULT_POLICY, max_attempts
        )
    except ValueError:
        return
    assert reusable is False


@pytest.mark.parametrize("change", ["order", "membership", "entry", "response"])
def test_source_review_reuse_requires_the_complete_ordered_batch(change):
    original = small_case()
    frozen = source_request(original)
    receipt = terminal_receipt(frozen)
    assert update.compatible_receipt(frozen, receipt, DEFAULT_POLICY, 2)
    changed = deepcopy(original)
    sources = changed["values"][0]["sources"]
    if change == "order":
        sources.reverse()
    elif change == "membership":
        sources.pop()
    elif change == "entry":
        sources[0]["journal_entry"] += " Then I ignored her request for help."
    else:
        sources[0]["nudge_response"] = "I will call her again tomorrow."
    assert_not_reusable(source_request(changed), receipt)


@pytest.mark.parametrize(
    "left,right",
    [
        ("runtime:nomic", "runtime:full_history"),
        ("runtime:full_history", "reference:primary"),
        ("reference:primary", "reference:repeat_2"),
        ("reference:repeat_2", "reference:repeat_3"),
    ],
)
def test_equal_source_payloads_do_not_merge_methods_or_repeats(left, right):
    item = small_case()
    original = source_request(item, left)
    changed = source_request(item, right)
    assert original["prompt"] == changed["prompt"]
    assert original["schema"] == changed["schema"]
    assert_not_reusable(changed, terminal_receipt(original))


@pytest.mark.parametrize(
    "change", ["quotation", "entry", "response", "definition", "phrase", "value"]
)
def test_quote_reuse_requires_exact_quote_complete_source_and_value_context(change):
    item = small_case()
    output = experiment.nsm_evaluation.select_card(item, batch())
    frozen = experiment.quote_request(item, output, "quotation:primary:nomic")
    receipt = terminal_receipt(frozen)
    changed, selected = deepcopy(item), deepcopy(output)
    value = changed["values"][0]
    if change == "quotation":
        selected["selected"]["evidence_quote"] = "I felt relieved."
    elif change == "entry":
        value["sources"][0]["journal_entry"] += " Then I refused to listen."
    elif change == "response":
        value["sources"][0]["nudge_response"] = "Actually, I did not call her."
    elif change == "definition":
        value["approved_definition"] += " Taking responsibility for close people."
    elif change == "phrase":
        value["user_phrase"] = "Being dependable for close people"
    else:
        value["core_value"] = selected["core_value"] = "security"
    changed_request = experiment.quote_request(
        changed, selected, "quotation:primary:nomic"
    )
    assert_not_reusable(changed_request, receipt)


@pytest.mark.parametrize(
    "purpose",
    [
        "quotation:primary:full_history",
        "quotation:repeat_2:nomic",
        "quotation:repeat_3:nomic",
        "recheck:primary:nomic",
    ],
)
def test_quote_review_identity_keeps_method_repeat_and_recheck_independent(purpose):
    item = small_case()
    output = experiment.nsm_evaluation.select_card(item, batch())
    frozen = experiment.quote_request(item, output, "quotation:primary:nomic")
    changed = experiment.quote_request(item, output, purpose)
    assert frozen["prompt"] == changed["prompt"]
    assert_not_reusable(changed, terminal_receipt(frozen))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "different-model"),
        ("reasoning_effort", "xhigh"),
        ("service_tier", "priority"),
        ("timeout_seconds", 90),
    ],
)
def test_reuse_rejects_changed_runtime_settings(field, value):
    frozen = request()
    receipt = terminal_receipt(frozen)
    policy = deepcopy(DEFAULT_POLICY)
    policy["runtime"][field] = value
    assert_not_reusable(frozen, receipt, policy)


@pytest.mark.parametrize(
    "field,value",
    [("input_token_limit", 8000), ("max_output_tokens", 16000), ("sdk_retries", 1)],
)
def test_reuse_rejects_changed_policy_limits(field, value):
    frozen = request()
    receipt = terminal_receipt(frozen)
    policy = deepcopy(DEFAULT_POLICY)
    policy[field] = value
    assert_not_reusable(frozen, receipt, policy)


def test_recheck_attempt_limit_cannot_be_rebound_to_an_ordinary_request():
    frozen = request("recheck:primary:nomic:case", "reference")
    receipt = terminal_receipt(frozen, max_attempts=1)
    assert update.compatible_receipt(frozen, receipt, DEFAULT_POLICY, 1)
    assert_not_reusable(frozen, receipt, max_attempts=2)


@pytest.mark.parametrize("field", ["request_hash", "payload_hash", "policy_hash"])
def test_malformed_matching_receipt_fails_closed(field):
    frozen = request()
    receipt = terminal_receipt(frozen)
    receipt[field] = "corrupted"
    assert_not_reusable(frozen, receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_hash", "another-request"),
        ("payload_hash", "another-payload"),
        ("model", "another-model"),
        ("input_tokens", -1),
        ("input_tokens", True),
    ],
)
def test_matching_request_rejects_malformed_count_receipt(field, value):
    frozen = request()
    receipt = terminal_receipt(frozen)
    receipt["count_receipt"][field] = value
    with pytest.raises(ValueError):
        update.compatible_receipt(frozen, receipt, DEFAULT_POLICY, 2)


@pytest.mark.parametrize("field", ["count_attempts", "attempts"])
@pytest.mark.parametrize("limit", [1, 2])
def test_retention_rejects_attempts_beyond_the_frozen_limit(field, limit):
    frozen = request()
    receipt = terminal_receipt(frozen, max_attempts=limit)
    receipt[field] = [{"status": "failed", "retryable": False}] * (limit + 1)
    with pytest.raises(ValueError, match="attempt limit"):
        update.compatible_receipt(frozen, receipt, DEFAULT_POLICY, limit)


@pytest.mark.parametrize("field", ["system", "schema"])
def test_changed_prompt_or_schema_cannot_reuse_a_receipt(field):
    frozen = request()
    changed = deepcopy(frozen)
    if field == "system":
        changed[field] += " Require an independently supported action."
    else:
        changed[field]["properties"]["answer"]["enum"] = ["accepted"]
    assert_not_reusable(changed, terminal_receipt(frozen))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refused", "invalid", "count_failed", "failed"])
async def test_original_terminal_failures_are_retained_without_a_client(failure):
    replies = {
        "refused": [counted(), generated(refusal=True)],
        "invalid": [counted(), generated("bad JSON"), generated("bad JSON")],
        "count_failed": [(401, {"error": {"message": "Unauthorized"}})],
        "failed": [counted(), generated(status="failed")],
    }
    harness = Harness(replies[failure])
    frozen = request()
    original_receipt = await harness.provider().assess(frozen)
    assert original_receipt["status"] == failure
    original = deepcopy(harness.record)
    original_hash = stable_hash(original)
    record = {"execution": {"provider_policy": deepcopy(DEFAULT_POLICY)}}

    def no_client(**kwargs):
        raise AssertionError("Retained terminal evidence must not open a client")

    provider = update.RetainingProvider(
        record, lambda: None, original, validator=validate, client_factory=no_client
    )
    retained = await provider.assess(frozen)
    key = stable_hash(frozen)
    assert retained == original_receipt
    assert stable_hash(original) == original_hash
    assert record["execution"]["requests"][key] == original_receipt
    lineage = record["evidence_lineage"][key]
    assert lineage["origin"] == "retained"
    assert lineage["original_receipt_sha256"] == stable_hash(original_receipt)
    await provider.measure(frozen)
    await provider.assess(frozen)
    assert len(record["execution"]["requests"]) == 1


def test_provenance_only_changes_do_not_change_effective_context_or_semantic_input():
    item = {field: None for field in update.EFFECTIVE_FIELDS}
    item.update(small_case())
    original_context = stable_hash(update.effective(item))
    original_request = source_request(item)
    item["upstream"] = {"reason": "PRIVATE_UPSTREAM_LABEL"}
    item["drift_result"] = {"evidence": "PRIVATE_DRIFT_EVIDENCE"}
    item["input_hash"] = "new-provenance-only-hash"
    assert stable_hash(update.effective(item)) == original_context
    assert source_request(item) == original_request
    payload = str(openai_input_payload(source_request(item), DEFAULT_POLICY))
    assert "PRIVATE_" not in payload
    item["weekly_state"] = "active_drift"
    assert stable_hash(update.effective(item)) != original_context
    assert source_request(item) == original_request
    assert experiment.nsm_evaluation.select_card(item, batch())["mode"] == "reflection"


def test_order_and_availability_metadata_stay_outside_semantic_requests():
    item = small_case()
    output = experiment.nsm_evaluation.select_card(item, batch())
    source_before = source_request(item)
    quote_before = experiment.quote_request(item, output, "quotation:primary:nomic")
    item["split"] = "PRIVATE_FINAL_PARTITION"
    item["source_metadata"] = {"generation_label": "PRIVATE_GENERATION_LABEL"}
    item["values"][0]["source_metadata"]["new"]["generation_label"] = "PRIVATE_LABEL"
    assert source_request(item) == source_before
    quote_after = experiment.quote_request(item, output, "quotation:primary:nomic")
    assert quote_after == quote_before
    assert output["selected"]["evidence_quote"] == ACTION


def test_saved_impact_audit_reproduces_from_original_and_v4_run1(tmp_path):
    """Replay the published update with its frozen runtime, without model calls."""
    snapshot = source_snapshot(
        update.ROOT,
        "f7e14ebb09bdd1b5bf3e7628bbfcbf626f4b7b57",
        tmp_path / "historical",
    )
    assert_in_snapshot(snapshot, Path(__file__), "_assert_saved_impact_audit")


def test_current_update_still_rejects_changed_runtime_contract(monkeypatch):
    original = update.read(update.ORIGINAL)
    hashes = original["execution"]["freeze"]["code_sha256"]
    monkeypatch.setattr(experiment, "_reporting_corrections", lambda _: {})
    monkeypatch.setattr(
        experiment,
        "file_hash",
        lambda path: (
            "changed"
            if path == update.ROOT / "src/north_star/runtime.py"
            else hashes[path.relative_to(update.ROOT).as_posix()]
        ),
    )
    with pytest.raises(ValueError, match="Original NSM contract changed"):
        update.verify_original(original)


def _assert_saved_impact_audit():
    original = update.read(update.ORIGINAL)
    saved = update.read(update.OUTPUT / "impact_audit.json")
    update.verify_original(original)
    manifest, requests = update.upstream.verify(update.UPSTREAM)
    responses = update.upstream.read_rows(update.UPSTREAM / "responses.jsonl")
    update.upstream.score_rows(requests, responses, manifest["settings"]["repeats"])
    cases = update.rebuild_cases(original, responses, requests)
    assert cases == saved["cases"]
    old = {case["case_id"]: case for case in original["execution"]["cases"]}
    assert {case["case_id"] for case in cases} == set(old)
    assert len(cases) == 501
    assert len({case["persona_id"] for case in cases}) == 105
    assert saved["partition"] == original["partition"]
    sample = original["execution"]["consistency_case_ids"]
    assert saved["consistency_case_ids"] == sample
    assert len(sample) == len(set(sample)) == 20
    differences = []
    for case in cases:
        before = old[case["case_id"]]
        changed = {
            field: {"before": before[field], "after": case[field]}
            for field in update.EFFECTIVE_FIELDS
            if before[field] != case[field]
        }
        differences.append(
            {
                "case_id": case["case_id"],
                "persona_id": case["persona_id"],
                "split": case["split"],
                "changed": bool(changed),
                "before_effective_sha256": stable_hash(update.effective(before)),
                "after_effective_sha256": stable_hash(update.effective(case)),
                "differences": changed,
                "before_case_sha256": stable_hash(before),
                "after_case_sha256": stable_hash(case),
                "before_upstream_sha256": stable_hash(before["upstream"]),
                "after_upstream_sha256": stable_hash(case["upstream"]),
            }
        )
    assert differences == saved["case_differences"]
    affected = [row for row in differences if row["changed"]]
    assert len(affected) == saved["summary"]["changed_cases"]
    assert len({row["persona_id"] for row in affected}) == saved["summary"][
        "changed_personas"
    ]
    assert dict(Counter(row["split"] for row in affected)) == saved["summary"][
        "changed_by_split"
    ]
