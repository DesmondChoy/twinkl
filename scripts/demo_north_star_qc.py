"""Browser QC app with explicit controlled provider responses and zero paid calls.

Run: python -m uvicorn scripts.demo_north_star_qc:app --port 8000
POST /qc/mode/{success,failure,omission,pending,long} controls the NSM response.
This harness is never imported by the deployed Experience app.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.demo.api import create_app
from src.demo.experience_service import InMemoryExperienceService
from src.north_star import assessment
from src.north_star.provider import BudgetedProvider, BudgetLedger
from src.north_star.runtime import (
    INTEGRATION_POLICY_PATH,
    OpenAINorthStarRuntime,
    pending_north_star_record,
)
from tests.demo.test_experience_service import DeterministicWeeklyReviewer, _receipt
from tests.north_star.test_runtime import count_requests

QUOTE = (
    "I helped my sister carry the groceries upstairs "
    "and stayed to cook dinner with her."
)
LONG_QUOTE = QUOTE + (
    " She had been worrying about managing the stairs after a long shift, "
    "so I picked up the heavier bags and made another trip for the rest. "
    "We chopped vegetables together, and I washed up while she sat down. "
    "Before leaving, I packed a portion for her lunch and checked that she "
    "had everything she needed for the morning. I was tired too, but making "
    "the evening easier for her felt like a good use of the time."
)
_state = {"mode": "success"}
_gate = asyncio.Event()
_gate.set()
_directory = tempfile.TemporaryDirectory(prefix="twinkl-nsm-browser-qc-")


class ControlledProvider(BudgetedProvider):
    async def _complete(self, request: dict, *, retry: bool):
        attempt = self.ledger.reserve(request, retry=retry)
        if attempt.reused:
            return attempt
        prompt = json.loads(request["prompt"])
        rows = []
        for source in prompt["sources"]:
            accepted = (
                QUOTE in source["journal_entry"]
                and prompt["core_value"] == "benevolence"
                and _state["mode"] in {"success", "long"}
            )
            rows.append(
                {
                    "entry_id": source["entry_id"],
                    "reason_code": "observable_choice" if accepted else "ambiguous",
                    "quote_source": "journal_entry" if accepted else None,
                    "evidence_quote": (
                        LONG_QUOTE
                        if accepted and LONG_QUOTE in source["journal_entry"]
                        else QUOTE if accepted else ""
                    ),
                    "action_assessment": (
                        "Controlled fixture: helped sister carry groceries and cook."
                    ),
                    "value_assessment": (
                        "Controlled fixture: caring for a close family member."
                    ),
                    "conflict_assessment": (
                        "Controlled fixture: no opposing behavior supplied."
                    ),
                }
            )
        attempt.raw_text = json.dumps(
            {
                "schema_version": assessment.SOURCE_SCHEMA_VERSION,
                "core_value": prompt["core_value"],
                "results": rows,
            }
        )
        attempt.status = "completed"
        attempt.actual_model = "controlled-browser-test-double"
        attempt.calculated_cost_usd = 0
        return self.ledger.finish(attempt)


_provider = ControlledProvider(
    BudgetLedger(Path(_directory.name) / "budget.json", INTEGRATION_POLICY_PATH)
)
_runtime = OpenAINorthStarRuntime(
    provider=_provider,
    count_requests=count_requests,
    counts_path=Path(_directory.name) / "counts.json",
)


async def north_star(request, *, retry: bool = False):
    await _gate.wait()
    if _state["mode"] == "failure":
        return pending_north_star_record(request).model_copy(
            update={
                "status": "failed",
                "reason": "controlled_provider_unavailable",
                "retryable": True,
            }
        )
    return await _runtime(request, retry=retry)


async def nudge(request):
    return _receipt(decision=None, nudge_text=None)


async def coach(prompt, schema=None, instructions=None):
    return json.dumps(
        {
            "weekly_mirror": f'You wrote, "{QUOTE}"',
            "tension_explanation": (
                "That specific action gives us something concrete "
                "to reflect on this week."
            ),
            "reflective_question": "What did making that time mean to you?",
        }
    )


service = InMemoryExperienceService(
    nudge_runtime=nudge,
    weekly_reviewer=DeterministicWeeklyReviewer(),
    coach_llm_complete=coach,
    north_star_runtime=north_star,
)
app = create_app(service)


async def mode(request: Request):
    choice = request.path_params["mode"]
    if choice not in {"success", "failure", "omission", "pending", "long"}:
        return JSONResponse({"error": "Unknown controlled mode"}, status_code=400)
    _state["mode"] = choice
    if choice == "pending":
        _gate.clear()
    else:
        _gate.set()
    return JSONResponse(
        {"mode": choice, "provider": "controlled test double", "paid_calls": 0}
    )


app.routes.insert(0, Route("/qc/mode/{mode}", mode, methods=["POST"]))
