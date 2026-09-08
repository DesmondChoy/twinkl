"""Live NSM execution isolated from experiment artifacts and the app event loop.

The POC serializes live NSM work in one worker thread. The default private ledger
uses a separately authorized, fixed live budget across sessions and restarts.
Explicit legacy callers can still carry finalized integration spend forward.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from src.north_star import input_budget
from src.north_star.provider import (
    BudgetedProvider,
    BudgetError,
    BudgetLedger,
    ProviderAttempt,
    stable_hash,
)
from src.north_star.runtime import (
    INTEGRATION_POLICY_PATH,
    LIVE_POLICY_PATH,
    ROOT,
    CountRequests,
    NorthStarRecord,
    NorthStarRequest,
    OpenAINorthStarRuntime,
    RuntimeProvider,
    _PrivateResponseLedger,
    pending_north_star_record,
    source_review_requests,
)

DEFAULT_LIVE_DIRECTORY = ROOT / "logs/exports/demo_tool_runs/north_star"
T = TypeVar("T")


class _SeededLiveLedger(_PrivateResponseLedger):
    """Carry experiment spend forward without writing into its frozen artifacts."""

    def __init__(self, path: Path, *, source_directory: Path):
        super().__init__(path, INTEGRATION_POLICY_PATH)
        self.source_path = (source_directory / "budget.json").resolve()
        self.report_path = source_directory / "report.json"
        if self.source_path == self.path:
            raise BudgetError("Live and experiment ledgers must have distinct paths")
        key = hashlib.sha256(str(self.source_path).encode()).hexdigest()
        self.source_lock_path = Path(tempfile.gettempdir()) / f"twinkl-nsm-{key}.lock"

    def _seed(self) -> tuple[str, list[dict[str, Any]]]:
        try:
            # Use the experiment writer's existing lock without modifying its files.
            with self.source_lock_path.open("a+", encoding="utf-8") as lock:
                fcntl.flock(lock, fcntl.LOCK_SH)
                raw = self.source_path.read_bytes()
                source = json.loads(raw)
                report = json.loads(self.report_path.read_text())
        except (OSError, ValueError) as exc:
            raise BudgetError("Finalized integration budget is unavailable") from exc
        if (
            not isinstance(source, dict)
            or source.get("schema_version") != "north-star-budget-v1"
            or source.get("policy_hash") != stable_hash(self.policy)
            or not isinstance(source.get("attempts"), list)
            or not isinstance(report, dict)
            or report.get("schema_version") != "north-star-integration-results-v1"
        ):
            raise BudgetError("Integration budget provenance does not match")
        try:
            attempts = [
                ProviderAttempt.model_validate(row) for row in source["attempts"]
            ]
        except ValueError as exc:
            raise BudgetError("Integration budget receipts are malformed") from exc
        costs = [
            attempt.calculated_cost_usd
            if attempt.calculated_cost_usd is not None
            else attempt.reserved_cost_usd
            for attempt in attempts
        ]
        spent = sum(costs)
        reported = report.get("new_spent_or_reserved_usd")
        cumulative = report.get("cumulative_spent_or_reserved_usd")
        if (
            any(attempt.status == "pending" for attempt in attempts)
            or any(not math.isfinite(cost) or cost < 0 for cost in costs)
            or type(report.get("generation_attempts")) is not int
            or report.get("generation_attempts") != len(attempts)
            or not isinstance(reported, (int, float))
            or isinstance(reported, bool)
            or not isinstance(cumulative, (int, float))
            or isinstance(cumulative, bool)
            or not math.isfinite(spent)
            or not math.isclose(reported, spent, abs_tol=1e-9)
            or not math.isclose(
                cumulative, spent + self.policy["prior_spend_usd"], abs_tol=1e-9
            )
        ):
            raise BudgetError("Integration budget report is not finalized")
        return hashlib.sha256(raw).hexdigest(), [
            attempt.model_copy(update={"raw_text": None}).model_dump()
            for attempt in attempts
        ]

    def transact(self, operation: Callable[[dict], T]) -> T:
        def with_seed(state: dict) -> T:
            source_hash, attempts = self._seed()
            recorded_hash = state.get("integration_budget_sha256")
            if recorded_hash is None:
                if state["attempts"]:
                    raise BudgetError(
                        "Existing live budget lacks integration provenance"
                    )
                state["integration_budget_sha256"] = source_hash
                state["attempts"] = attempts
            elif recorded_hash != source_hash:
                raise BudgetError("Integration spend changed; live accounting is stale")
            return operation(state)

        return super().transact(with_seed)


class LiveNorthStarRuntime:
    """Run frozen selection away from the app loop with separate live artifacts."""

    def __init__(
        self,
        *,
        directory: Path = DEFAULT_LIVE_DIRECTORY,
        source_directory: Path | None = None,
        provider_factory: Callable[[BudgetLedger], RuntimeProvider] = BudgetedProvider,
        measure: CountRequests = input_budget.measure_requests,
    ):
        self.directory = directory.resolve()
        self.source_directory = (
            source_directory.resolve() if source_directory is not None else None
        )
        self._provider_factory = provider_factory
        self._measure = measure
        self._runtime: OpenAINorthStarRuntime | None = None
        self._executor: ThreadPoolExecutor | None = None

    async def _measure_locked(
        self, requests: list[dict], policy: dict, output: Path
    ) -> dict:
        key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
        lock_path = Path(tempfile.gettempdir()) / f"twinkl-live-nsm-counts-{key}.lock"
        # Each wrapper has one worker, and flock also covers other app processes.
        # The blocking lock and existing fsync calls execute only in that worker.
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return await self._measure(requests, policy, output)

    def _run(self, request: NorthStarRequest, retry: bool) -> NorthStarRecord:
        if self._runtime is None:
            policy_path = LIVE_POLICY_PATH
            ledger: _PrivateResponseLedger
            if self.source_directory is None:
                ledger = _PrivateResponseLedger(
                    self.directory / "budget.json", policy_path
                )
            else:
                policy_path = INTEGRATION_POLICY_PATH
                ledger = _SeededLiveLedger(
                    self.directory / "budget.json",
                    source_directory=self.source_directory,
                )
            # Verify accounting before counting or generating an input.
            ledger.snapshot()
            self._runtime = OpenAINorthStarRuntime(
                provider=self._provider_factory(ledger),
                count_requests=self._measure_locked,
                ledger_path=ledger.path,
                counts_path=self.directory / "input-counts.json",
                policy_path=policy_path,
            )
        return asyncio.run(self._runtime(request, retry=retry))

    async def __call__(
        self, request: NorthStarRequest, *, retry: bool = False
    ) -> NorthStarRecord:
        if not source_review_requests(request) or not os.environ.get("OPENAI_API_KEY"):
            # This path performs no ledger/count I/O and starts no worker thread.
            return await OpenAINorthStarRuntime()(request, retry=retry)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="twinkl-north-star"
            )
        future = self._executor.submit(self._run, request, retry)
        try:
            return await asyncio.shield(asyncio.wrap_future(future))
        except BudgetError as exc:
            return pending_north_star_record(request).model_copy(
                update={
                    "status": "failed",
                    "reason": "budget_unavailable",
                    "validation_evidence": [str(exc)],
                }
            )

    def forget(self, record: NorthStarRecord) -> None:
        """Queue raw-response cleanup on the same worker that owns that memory."""

        def release() -> None:
            if self._runtime is not None:
                self._runtime.forget(record)

        if self._executor is not None:
            self._executor.submit(release)

    def close(self) -> None:
        """Release the worker after callers finish outstanding requests."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._runtime = None


def create_live_north_star_runtime() -> LiveNorthStarRuntime:
    return LiveNorthStarRuntime()
