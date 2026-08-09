"""Protocol checks for the Luna higher-reasoning comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments import compare_twinkl_ck3w_luna_higher_reasoning as study
from scripts.experiments import weekly_verifier_ablation as baseline

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/evals/twinkl_ck3w_luna_higher_reasoning_v1.yaml"


def _config() -> dict:
    return baseline._read_yaml(CONFIG_PATH)


def test_protocol_preserves_data_and_tests_requested_efforts() -> None:
    config = _config()
    frozen = study._load_frozen(config, ROOT)
    records = frozen[2]
    specs = study._model_specs(config)

    assert list(specs) == ["medium", "high", "xhigh"]
    assert len(records) == 951
    assert len(records) * config["study"]["repeats"] == 2853
    assert config["study"]["expected_total_calls"] == 8559
    assert all(spec["model"] == "gpt-5.6-luna" for spec in specs.values())


def test_protocol_preserves_the_luna_none_baseline() -> None:
    config = _config()
    _none_config, spec, responses, _metrics = study._load_no_reasoning_baseline(
        config, ROOT
    )
    records = study._load_frozen(config, ROOT)[2]

    assert spec["model"] == "gpt-5.6-luna"
    assert spec["reasoning_effort"] == "none"
    assert len(responses) == 2853
    study._validate_none_receipts(responses, records, repeats=3)


def test_protocol_uses_current_luna_prices() -> None:
    config = _config()

    assert config["api"]["pricing_usd_per_million_tokens"] == {
        "input": 0.2,
        "cached_input": 0.02,
        "cache_write": 0.25,
        "output": 1.2,
    }
    assert config["api"]["pricing_source"] == (
        "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
    )


def test_receipts_record_the_requested_reasoning_effort() -> None:
    study._validate_receipt_effort(
        [{"status": "ok", "reasoning_effort": "high"}], "high"
    )
    with pytest.raises(ValueError, match="do not match"):
        study._validate_receipt_effort(
            [{"status": "invalid", "reasoning_effort": "medium"}], "xhigh"
        )


def test_continuation_reuses_only_completed_initial_receipts() -> None:
    config = _config()
    study._validate_receipt_caps(
        [
            {
                "status": "ok",
                "response_status": "completed",
                "max_output_tokens": 2000,
            },
            {
                "status": "invalid",
                "response_status": "completed",
                "max_output_tokens": 8000,
            },
        ],
        config,
    )
    with pytest.raises(ValueError, match="incomplete initial"):
        study._validate_receipt_caps(
            [
                {
                    "status": "refusal",
                    "response_status": "incomplete",
                    "max_output_tokens": 2000,
                }
            ],
            config,
        )


def test_aggregate_budget_uses_each_smoke_projection() -> None:
    config = _config()
    projections = {
        effort: {
            "projected_standard_rate_cost_usd": 3.0,
            "projected_cost_with_contingency_usd": 3.6,
        }
        for effort in ("medium", "high", "xhigh")
    }

    summary = study._budget_summary(config, projections)

    assert summary["projected_standard_rate_cost_usd"] == 9.0
    assert summary["projected_cost_with_contingency_usd"] == pytest.approx(10.8)
    assert summary["within_budget"] is True


def test_latency_is_median_terminal_persona_week_call_time() -> None:
    summary = study._latency_summary(
        [
            {"status": "ok", "latency_seconds": 1.0, "repeat": 1},
            {"status": "invalid", "latency_seconds": 3.0, "repeat": 1},
            {"status": "ok", "latency_seconds": 8.0, "repeat": 2},
            {"status": "error", "latency_seconds": 99.0, "repeat": 2},
        ]
    )

    assert summary == {
        "unit": "terminal_persona_week_api_call",
        "count": 3,
        "median_seconds": 3.0,
        "repeat_median_seconds": {"1": 2.0, "2": 8.0},
        "diagnostic_only": True,
    }


def test_parser_uses_xhigh_as_the_api_name() -> None:
    args = study.build_parser().parse_args(["run", "--effort", "xhigh"])

    assert args.effort == "xhigh"
