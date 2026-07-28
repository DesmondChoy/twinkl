"""Small HTTP adapter for the Experience and Inspect contracts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from src.demo.contracts import (
    ApiRequest,
    ApiResponse,
    SafeError,
    ScenarioLoadRequest,
)
from src.demo.experience_service import InMemoryExperienceService
from src.demo.scenarios import load_scenario_catalog

REQUEST_ADAPTER: TypeAdapter[ApiRequest] = TypeAdapter(ApiRequest)
RESPONSE_ADAPTER: TypeAdapter[ApiResponse] = TypeAdapter(ApiResponse)


class ExperienceStaticFiles(StaticFiles):
    """Serve the built Experience and preserve client-side route fallback."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404 or Path(path).suffix:
                raise
            return await super().get_response("index.html", scope)


def _status_code(response: ApiResponse) -> int:
    if response.operation != "error":
        return 200
    if response.error.code in {
        "idempotency_conflict",
        "journal_order_conflict",
        "session_conflict",
    }:
        return 409
    if response.error.code in {
        "scenario_not_found",
        "session_not_found",
        "trace_cursor_not_found",
    }:
        return 404
    if response.error.code == "scenario_integrity_error":
        return 500
    return 400


def _validation_error(payload: Any) -> JSONResponse:
    request_id = (
        str(payload.get("request_id"))
        if isinstance(payload, dict) and payload.get("request_id")
        else "unknown-request"
    )
    operation = (
        str(payload.get("operation")) if isinstance(payload, dict) else "create_session"
    )
    if operation not in {
        "create_session",
        "submit_journal_entry",
        "load_scenario",
        "read_trace",
    }:
        operation = "create_session"
    body = {
        "schema_version": "experience-inspect-v1",
        "operation": "error",
        "requested_operation": operation,
        "request_id": request_id,
        "status": "error",
        "error": {
            "code": "invalid_request",
            "message": "The Experience request does not match the current contract.",
            "retryable": False,
        },
    }
    RESPONSE_ADAPTER.validate_python(body)
    return JSONResponse(body, status_code=422)


def create_app(
    service: InMemoryExperienceService | None = None,
    *,
    scenario_root: Path | None = None,
    static_root: Path | None = None,
) -> Starlette:
    experience_service = service or InMemoryExperienceService()
    root = scenario_root or Path(__file__).resolve().parents[2]
    scenario_fixtures = None

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def experience(request: Request) -> JSONResponse:
        nonlocal scenario_fixtures
        try:
            payload = await request.json()
            parsed = REQUEST_ADAPTER.validate_python(payload)
        except (ValidationError, ValueError):
            return _validation_error(payload if "payload" in locals() else None)

        if isinstance(parsed, ScenarioLoadRequest):
            try:
                if scenario_fixtures is None:
                    _, scenario_fixtures = load_scenario_catalog(root)
            except (OSError, ValidationError, ValueError):
                response_payload = {
                    "schema_version": "experience-inspect-v1",
                    "operation": "error",
                    "requested_operation": "load_scenario",
                    "request_id": parsed.request_id,
                    "status": "error",
                    "error": SafeError(
                        code="scenario_integrity_error",
                        message=(
                            "The saved persona catalog failed its integrity check."
                        ),
                        retryable=False,
                    ).model_dump(mode="json"),
                }
                response = RESPONSE_ADAPTER.validate_python(response_payload)
            else:
                fixture = scenario_fixtures.get(parsed.scenario_id)
                if fixture is None:
                    response_payload = {
                        "schema_version": "experience-inspect-v1",
                        "operation": "error",
                        "requested_operation": "load_scenario",
                        "request_id": parsed.request_id,
                        "status": "error",
                        "error": SafeError(
                            code="scenario_not_found",
                            message="The selected saved persona is unavailable.",
                            retryable=False,
                        ).model_dump(mode="json"),
                    }
                    response = RESPONSE_ADAPTER.validate_python(response_payload)
                else:
                    response = await experience_service.load_saved_scenario(
                        parsed,
                        fixture,
                    )
        else:
            response = await experience_service.handle(parsed)

        return JSONResponse(
            response.model_dump(mode="json"),
            status_code=_status_code(response),
        )

    routes: list[BaseRoute] = [
        Route("/health", lambda _: PlainTextResponse("ok"), methods=["GET"]),
        Route("/api/health", health, methods=["GET"]),
        Route("/api/experience", experience, methods=["POST"]),
    ]
    if static_root is not None:
        routes.append(
            Mount(
                "/",
                app=ExperienceStaticFiles(directory=static_root, html=True),
                name="experience",
            )
        )
    return Starlette(routes=routes)


def create_deployment_app() -> Starlette:
    static_root = os.getenv("TWINKL_STATIC_ROOT")
    return create_app(
        static_root=Path(static_root) if static_root else None,
    )


app = create_deployment_app()
