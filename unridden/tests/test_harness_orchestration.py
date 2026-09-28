"""Backend limits and brief contention are handled behind one application call."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from unridden.tests.test_harness import harness


def questions(count: int) -> dict[str, Any]:
    return {
        f"q{i}": {"type": "noul", "instructions": f"Does state concern ticket {i}?"}
        for i in range(count)
    }


async def until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_one_bundle_splits_at_backend_limit_and_aggregates_work(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        response = await h.client.post(
            "/v1/decisions",
            json={"state": "support tickets", "questions": questions(64)},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        batches = [
            body for _, path, body in h.transport.requests if path == "/v2/decisions"
        ]
        assert [len(body["questions"]) for body in batches] == [32, 32]
        assert len({body["snapshot"]["id"] for body in batches}) == 1
        assert result["execution"] == {"decision_batches": 2, "busy_retries": 0}
        assert result["context_usage"]["successful_decisions"] == 1
        assert list(result["answers"]) == list(questions(64))
        native = []
        for body in batches:
            direct = await h.service.client.post("/v2/decisions", json=body)
            assert direct.status_code == 200
            native.append(direct.json())
        assert result["answers"] == {
            key: answer for row in native for key, answer in row["answers"].items()
        }
        assert result["usage"]["input_tokens"] == sum(
            row["usage"]["input_tokens"] for row in native
        )
        assert result["usage"]["output_tokens"] == 0
        for depth in ("lower", "upper"):
            assert result["snapshot_usage"]["block_tokens"][depth] == sum(
                row["snapshot_usage"]["block_tokens"][depth] for row in native
            )
        assert len(result["snapshot_usage"]["suffix_tokens"]) == 64


async def test_batch_limit_is_discovered_and_manual_contexts_share_orchestration(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        model = (await h.service.client.get("/v2/models")).json()
        model["models"][0]["limits"]["questions"] = 2
        h.transport.fault = lambda request: (
            httpx.Response(200, json=model)
            if request.url.path == "/v2/models"
            else None
        )
        context = await h.context()
        result = await h.decide(context["context_id"], questions=questions(5))
        assert result.status_code == 200
        assert result.json()["execution"]["decision_batches"] == 3
        assert [
            len(body["questions"])
            for _, path, body in h.transport.requests
            if path == "/v2/decisions"
        ] == [2, 2, 1]


async def test_question_ceiling_rejects_before_capture(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        result = await h.client.post(
            "/v1/decisions", json={"state": "state", "questions": questions(65)}
        )
        assert result.status_code == 422
        assert not h.transport.requests


async def test_oversized_choice_is_not_transformed_into_a_tournament(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        criteria = {f"option{i}": f"description {i}" for i in range(27)}
        response = await h.client.post(
            "/v1/decisions",
            json={
                "state": "state",
                "questions": {"pick": {"type": "choice", "criteria": criteria}},
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "budget_error"
        requests = [
            body for _, path, body in h.transport.requests if path == "/v2/decisions"
        ]
        assert len(requests) == 1
        assert requests[0]["questions"]["pick"]["criteria"] == criteria


@pytest.mark.parametrize("lost", [False, True])
async def test_later_batch_failure_returns_no_partial_answers_or_replay(
    tmp_path: Path, lost: bool
) -> None:
    async with harness(tmp_path) as h:
        attempts = 0

        def fault(request: httpx.Request) -> httpx.Response | None:
            nonlocal attempts
            if request.url.path != "/v2/decisions":
                return None
            attempts += 1
            if attempts == 2:
                if lost:
                    return httpx.Response(
                        404, json={"error": {"code": "snapshot_not_found"}}
                    )
                raise httpx.ReadTimeout("private", request=request)
            return None

        h.transport.fault = fault
        result = await h.client.post(
            "/v1/decisions", json={"state": "state", "questions": questions(64)}
        )
        assert result.status_code == (410 if lost else 504)
        assert "answers" not in result.json()
        assert result.json()["error"]["retryable"] is False
        if lost:
            assert result.json()["error"]["code"] == "context_lost_after_progress"
        assert attempts == 2
        assert (
            len(
                [
                    1
                    for method, path, _ in h.transport.requests
                    if method == "POST" and path == "/v2/snapshots"
                ]
            )
            == 1
        )
        assert not h.service.entries  # first failed automatic scope is unclaimed


async def test_short_backend_busy_is_absorbed_and_wait_reported(tmp_path: Path) -> None:
    async with harness(tmp_path, backend_busy_timeout=1.0) as h:
        context = await h.context()
        attempts = 0

        def fault(request: httpx.Request) -> httpx.Response | None:
            nonlocal attempts
            if request.url.path == "/v2/decisions":
                attempts += 1
                if attempts < 3:
                    return httpx.Response(429, json={"error": {"code": "busy"}})
            return None

        h.transport.fault = fault
        response = await h.decide(context["context_id"])
        assert response.status_code == 200
        result = response.json()
        assert attempts == 3
        assert result["execution"] == {"decision_batches": 1, "busy_retries": 2}
        assert result["timing"]["backoff_ms"] >= 100
        assert result["timing"]["orchestration_ms"] >= result["timing"]["backoff_ms"]
        assert result["context_usage"]["successful_decisions"] == 1


@pytest.mark.parametrize(
    "status,code", [(429, "unknown_refusal"), (503, "unavailable")]
)
async def test_other_refusals_are_not_replayed_even_with_busy_budget(
    tmp_path: Path, status: int, code: str
) -> None:
    async with harness(tmp_path, backend_busy_timeout=1.0) as h:
        context = await h.context()
        h.transport.fault = lambda request: (
            httpx.Response(status, json={"error": {"code": code}})
            if request.url.path == "/v2/decisions"
            else None
        )
        before = len(h.transport.requests)
        result = await h.decide(context["context_id"])
        assert result.status_code == status
        assert len(h.transport.requests) == before + 1


async def test_backend_busy_budget_is_finite(tmp_path: Path) -> None:
    async with harness(tmp_path, backend_busy_timeout=0.02) as h:
        context = await h.context()
        h.transport.fault = lambda request: (
            httpx.Response(429, json={"error": {"code": "busy"}})
            if request.url.path == "/v2/decisions"
            else None
        )
        before = len(h.transport.requests)
        result = await asyncio.wait_for(h.decide(context["context_id"]), timeout=1)
        assert result.status_code == 429
        assert len(h.transport.requests) - before == 2
        assert not h.service._busy


async def test_queue_is_bounded_cancel_safe_and_successful_wait_is_timed(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path, queue_timeout=1.0, max_waiting=1) as h:
        context = await h.context()
        handle = context["context_id"]
        h.worker.gate = True
        active = asyncio.create_task(h.decide(handle))
        await h.worker.entered.wait()
        queued = asyncio.create_task(h.decide(handle))
        await until(lambda: h.service._waiting == 1)
        rejected = await h.decide(handle)
        assert rejected.status_code == 429
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert h.service._waiting == 0
        replacement = asyncio.create_task(h.decide(handle))
        await until(lambda: h.service._waiting == 1)
        h.worker.block.set()
        assert (await active).status_code == 200
        response = await replacement
        assert response.status_code == 200
        assert response.json()["timing"]["queue_ms"] > 0
        assert h.service.entries[handle].info.successful_decisions == 2
        assert not h.service._busy and h.service._waiting == 0


async def test_queue_timeout_does_not_start_backend_work(tmp_path: Path) -> None:
    async with harness(tmp_path, queue_timeout=0.01) as h:
        context = await h.context()
        h.worker.gate = True
        active = asyncio.create_task(h.decide(context["context_id"]))
        await h.worker.entered.wait()
        before = len(h.transport.requests)
        rejected = await h.decide(context["context_id"])
        assert rejected.status_code == 429
        assert len(h.transport.requests) == before
        assert h.service._waiting == 0
        h.worker.block.set()
        assert (await active).status_code == 200


async def test_operation_budget_releases_admission_without_replay(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path, request_timeout=0.05) as h:
        context = await h.context()
        h.worker.gate = True
        before = len(h.transport.requests)
        response = await asyncio.wait_for(h.decide(context["context_id"]), timeout=1)
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "request_timeout"
        assert response.json()["error"]["retryable"] is False
        assert len(h.transport.requests) == before + 1
        assert not h.service._busy
