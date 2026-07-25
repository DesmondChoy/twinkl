"""Small HTTP adapter for the Experience and Inspect contracts."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.demo.contracts import (
    ApiRequest,
    ApiResponse,
    SafeError,
    ScenarioLoadRequest,
)
from src.demo.experience_service import InMemoryExperienceService

REQUEST_ADAPTER: TypeAdapter[ApiRequest] = TypeAdapter(ApiRequest)
RESPONSE_ADAPTER: TypeAdapter[ApiResponse] = TypeAdapter(ApiResponse)


def _status_code(response: ApiResponse) -> int:
    if response.operation != "error":
        return 200
    if response.error.code in {
        "idempotency_conflict",
        "journal_order_conflict",
        "session_conflict",
    }:
        return 409
    if response.error.code in {"session_not_found", "trace_cursor_not_found"}:
        return 404
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
) -> Starlette:
    experience_service = service or InMemoryExperienceService()

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def experience(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            parsed = REQUEST_ADAPTER.validate_python(payload)
        except (ValidationError, ValueError):
            return _validation_error(payload if "payload" in locals() else None)

        if isinstance(parsed, ScenarioLoadRequest):
            response_payload = {
                "schema_version": "experience-inspect-v1",
                "operation": "error",
                "requested_operation": "load_scenario",
                "request_id": parsed.request_id,
                "status": "error",
                "error": SafeError(
                    code="operation_not_ready",
                    message=(
                        "Saved persona loading is implemented in the next "
                        "Experience slice."
                    ),
                    retryable=False,
                ).model_dump(mode="json"),
            }
            response = RESPONSE_ADAPTER.validate_python(response_payload)
        else:
            response = await experience_service.handle(parsed)

        return JSONResponse(
            response.model_dump(mode="json"),
            status_code=_status_code(response),
        )

    return Starlette(
        routes=[
            Route("/api/health", health, methods=["GET"]),
            Route("/api/experience", experience, methods=["POST"]),
        ]
    )


app = create_app()
