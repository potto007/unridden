"""Exercise the front API against the real /v2 service and a fake native worker."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from unridden.api.app import ApiConfig
from unridden.api.app import create_app as backend_app
from unridden.harness.app import create_app
from unridden.harness.service import HarnessConfig, HarnessService
from unridden.tests.test_snapshots_app import FakeSnapshotBackend
from unridden.tests.test_snapshots_full_profile import FULL_HELLO

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Select the department responsible for this ticket.",
        "criteria": {"billing": "payments", "technical": "device support"},
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does this ticket describe an emergency?",
    },
    "severity": {"type": "score", "criteria": ["minor", "normal", "severe"]},
}


class MemoryBackend(FakeSnapshotBackend):
    # The shared fake retains metadata so its disk-load tests can reconstruct
    # dropped states. This harness uses only memory snapshots, like the native
    # worker's drop operation, which actually releases its owned snapshot.
    async def drop(self, snapshot_id: str) -> int:
        self._snaps.pop(snapshot_id, None)
        return await super().drop(snapshot_id)


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.requests: list[tuple[str, str, Any]] = []
        self.fault: Callable[[httpx.Request], httpx.Response | None] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raw = await request.aread()
        body = json.loads(raw) if raw else None
        self.requests.append((request.method, request.url.path, body))
        if self.fault is not None:
            result = self.fault(request)
            if result is not None:
                return result
        return await self.inner.handle_async_request(request)


@dataclass
class Fixture:
    client: httpx.AsyncClient
    worker: FakeSnapshotBackend
    service: HarnessService
    transport: RecordingTransport
    clock: list[float]

    async def context(self, **overrides: Any) -> dict[str, Any]:
        response = await self.client.post(
            "/v1/contexts",
            json={
                "state": {"ticket": "I was charged twice"},
                **overrides,
            },
        )
        assert response.status_code == 201, response.text
        return dict(response.json())

    async def decide(self, context_id: str, **overrides: Any) -> httpx.Response:
        return await self.client.post(
            "/v1/decisions",
            json={
                "context_id": context_id,
                "questions": QUESTIONS,
                **overrides,
            },
        )


@asynccontextmanager
async def harness(
    tmp_path: Path, *, full: bool = False, **overrides: Any
) -> AsyncIterator[Fixture]:
    # Most fault tests need immediate errors. Waiting/retry policy has dedicated
    # tests with positive budgets instead of slowing every native fault case.
    overrides.setdefault("queue_timeout", 0.0)
    overrides.setdefault("backend_busy_timeout", 0.0)
    worker = MemoryBackend()
    if full:
        worker.hello = FULL_HELLO
    upstream = backend_app(
        ApiConfig(
            v1_enabled=False,
            snapshots_enabled=True,
            snapshot_store_dir=tmp_path / "snapshots",
        ),
        snapshot_backend_factory=lambda _: worker,
    )
    transport = RecordingTransport(httpx.ASGITransport(app=upstream))
    clock = [10.0]
    front = create_app(
        HarnessConfig(backend_url="http://backend", cleanup_interval=3600, **overrides),
        transport=transport,
        monotonic=lambda: clock[0],
    )
    async with (
        upstream.router.lifespan_context(upstream),
        front.router.lifespan_context(front),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=front), base_url="http://front"
        ) as client,
    ):
        yield Fixture(client, worker, front.state.harness, transport, clock)


async def test_full_profile_discovers_depth_without_hardcoded_26b_blocks(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path, full=True) as h:
        created = await h.context()
        assert created["completed_blocks"] == 42
        assert created["profile"] == "full-v1"
        response = await h.decide(created["context_id"])
        assert response.status_code == 200
        assert response.json()["snapshot_usage"]["profile"] == "full-v1"
        assert h.worker.promote_calls == 0


async def test_gateway_has_no_raw_snapshot_rider_or_vector_routes(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        for path in ("/v2/snapshots", "/v2/rider", "/v2/state-evaluations"):
            assert (await h.client.post(path, json={})).status_code == 404
        assert not h.transport.requests


async def test_capture_before_question_then_aba_isolation(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        created = await h.context(instructions="Consider customer billing policy.")
        context_id = created["context_id"]
        captures = [
            body
            for method, path, body in h.transport.requests
            if method == "POST" and path == "/v2/snapshots"
        ]
        assert captures == [
            {
                "model": "local-gemma-unridden-v1",
                "input": {
                    "kind": "context",
                    "state": {
                        "invariant_instructions": "Consider customer billing policy.",
                        "state": {"ticket": "I was charged twice"},
                    },
                },
                "checkpoints": [30],
                "persistence": "memory",
                "ttl_seconds": 600,
            }
        ]
        assert all(path != "/v2/decisions" for _, path, _ in h.transport.requests)
        first = await h.decide(context_id)
        assert first.status_code == 200, first.text
        middle = await h.decide(
            context_id,
            questions={
                "different": {
                    "type": "noul",
                    "instructions": "Is this about the weather?",
                }
            },
        )
        assert middle.status_code == 200
        again = await h.decide(context_id)
        assert again.json()["answers"] == first.json()["answers"]
        assert first.json()["usage"]["output_tokens"] == 0
        assert h.worker.promote_calls == 0
        assert len(h.worker._snaps) == 1
        assert all(
            not body["save_result_snapshot"]
            for _, path, body in h.transport.requests
            if path == "/v2/decisions"
        )
        report = again.json()
        assert report["context_usage"]["successful_decisions"] == 3
        assert report["context_usage"]["capture_ms"] == created["capture_ms"]
        assert (
            report["context_usage"]["amortized_capture_ms"] == created["capture_ms"] / 3
        )
        timing = report["timing"]
        assert timing["orchestration_ms"] >= timing["backend_http_ms"] >= 0
        assert timing["local_ms"] >= 0
        assert again.headers["cache-control"] == "no-store"


async def test_identical_contexts_are_independent_and_hash_is_canonical(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        one = await h.context(state={"b": 2, "a": 1})
        two = await h.context(state={"a": 1, "b": 2})
        other = await h.context(state={"a": 1, "b": 3})
        assert one["context_id"] != two["context_id"]
        assert one["input_sha256"] == two["input_sha256"]
        assert one["input_sha256"] != other["input_sha256"]
        assert len(h.worker._snaps) == 3
        response = await h.client.delete("/v1/contexts/" + one["context_id"])
        assert response.status_code == 200
        assert len(h.worker._snaps) == 2
        assert (await h.decide(one["context_id"])).status_code == 404
        assert (await h.decide(two["context_id"])).status_code == 200
        assert (await h.client.get("/v1/contexts")).status_code == 405


async def test_expiry_is_fixed_and_sweep_frees_native_state(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        created = await h.context(ttl_seconds=10)
        context_id = created["context_id"]
        h.clock[0] += 9
        assert (await h.decide(context_id)).status_code == 200
        h.clock[0] += 1
        response = await h.decide(context_id)
        assert response.status_code == 410
        assert response.json()["error"]["code"] == "context_expired"
        assert len(h.worker._snaps) == 1
        await h.service.sweep()
        assert not h.worker._snaps
        assert not h.service.entries


async def test_lost_snapshot_is_not_silently_recreated_or_fallback(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()
        context_id = created["context_id"]
        snapshot_id = h.service.entries[context_id].snapshot_id
        await h.service.client.delete("/v2/snapshots/" + snapshot_id)
        count = len(h.transport.requests)
        response = await h.decide(context_id)
        assert response.status_code == 410
        assert response.json()["error"]["code"] == "context_lost"
        assert [(method, path) for method, path, _ in h.transport.requests[count:]] == [
            ("POST", "/v2/decisions")
        ]
        assert context_id not in h.service.entries


async def test_capacity_and_shutdown_cleanup(tmp_path: Path) -> None:
    async with harness(tmp_path, max_contexts=1) as h:
        first = await h.context()
        rejected = await h.client.post("/v1/contexts", json={"state": "another"})
        assert rejected.status_code == 507
        assert len(h.worker._snaps) == 1
        await h.client.delete("/v1/contexts/" + first["context_id"])
        await h.context(state="replacement")
        assert len(h.worker._snaps) == 1
    assert not h.worker._snaps
    assert not h.service.entries


async def test_cleanup_failure_retains_bounded_debt_and_revokes_handle(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path, max_contexts=1) as h:
        created = await h.context()
        context_id = created["context_id"]
        h.transport.fault = lambda request: (
            httpx.Response(429, json={"error": {"code": "busy"}})
            if request.method == "DELETE"
            else None
        )
        response = await h.client.delete("/v1/contexts/" + context_id)
        assert response.status_code == 429
        assert (await h.decide(context_id)).status_code == 410
        full = await h.client.post("/v1/contexts", json={"state": "new"})
        assert full.status_code == 507
        assert len(h.service.entries) == 1
        h.transport.fault = None
        await h.service.sweep()
        assert not h.service.entries
        assert not h.worker._snaps


async def test_concurrent_requests_fail_fast_without_duplicate_work(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()
        h.worker.gate = True
        active = asyncio.create_task(h.decide(created["context_id"]))
        await h.worker.entered.wait()
        other = await h.decide(created["context_id"])
        assert other.status_code == 429
        assert other.json()["error"]["code"] == "busy"
        await h.service.sweep()
        h.worker.block.set()
        assert (await active).status_code == 200
        assert h.service.entries[created["context_id"]].info.successful_decisions == 1


@pytest.mark.parametrize(
    "status,code,expected",
    [
        (429, "busy", 429),
        (529, "unavailable", 503),
        (408, "timeout", 504),
        (422, "budget_error", 422),
        (507, "snapshot_store_full", 507),
        (500, "internal_error", 502),
    ],
)
async def test_upstream_errors_are_sanitized_and_not_replayed(
    tmp_path: Path, status: int, code: str, expected: int
) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()
        h.transport.fault = lambda request: (
            httpx.Response(status, json={"error": {"code": code, "message": "private"}})
            if request.url.path == "/v2/decisions"
            else None
        )
        count = len(h.transport.requests)
        response = await h.decide(created["context_id"])
        assert response.status_code == expected
        if code == "budget_error":
            assert response.json()["error"]["code"] == "budget_error"
        assert "private" not in response.text
        assert len(h.transport.requests) == count + 1
        assert h.service.entries[created["context_id"]].info.successful_decisions == 0


async def test_transport_timeout_is_ambiguous_and_not_replayed(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()

        def timeout(request: httpx.Request) -> httpx.Response | None:
            if request.url.path == "/v2/decisions":
                raise httpx.ReadTimeout("private", request=request)
            return None

        h.transport.fault = timeout
        count = len(h.transport.requests)
        response = await h.decide(created["context_id"])
        assert response.status_code == 504
        assert response.json()["error"]["retryable"] is False
        assert len(h.transport.requests) == count + 1
        assert not h.service._busy
        h.transport.fault = None
        assert (await h.decide(created["context_id"])).status_code == 200


async def test_backend_restart_loses_memory_context_explicitly(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()
        h.worker.crash()
        response = await h.decide(created["context_id"])
        assert response.status_code == 410
        assert response.json()["error"]["code"] == "context_lost"
        assert h.worker.start_calls == 2
        assert not h.service.entries


async def test_cancellation_releases_admission_without_replaying(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()
        h.worker.gate = True
        active = asyncio.create_task(h.decide(created["context_id"]))
        await h.worker.entered.wait()
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert not h.service._busy
        assert h.service.entries[created["context_id"]].info.successful_decisions == 0
        h.worker.gate = False
        assert (await h.decide(created["context_id"])).status_code == 200


async def test_wrong_parent_response_is_rejected(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()
        good = await h.decide(created["context_id"])
        corrupted = good.json()
        del corrupted["context_usage"]
        del corrupted["timing"]
        del corrupted["execution"]
        corrupted["snapshot_usage"]["requested_parent"] = "unrelated"
        h.transport.fault = lambda request: (
            httpx.Response(200, json=corrupted)
            if request.url.path == "/v2/decisions"
            else None
        )
        response = await h.decide(created["context_id"])
        assert response.status_code == 502
        assert h.service.entries[created["context_id"]].info.successful_decisions == 1


async def test_delete_requires_confirmed_backend_deletion(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        created = await h.context()
        h.transport.fault = lambda request: (
            httpx.Response(200, json={"deleted": "wrong", "freed_bytes": 1})
            if request.method == "DELETE"
            else None
        )
        response = await h.client.delete("/v1/contexts/" + created["context_id"])
        assert response.status_code == 502
        assert created["context_id"] in h.service.entries
        h.transport.fault = None
        await h.service.sweep()
        assert not h.worker._snaps


@pytest.mark.parametrize(
    "body",
    [
        {"state": "context", "questions": QUESTIONS},
        {"state": "context", "ttl_seconds": 0},
        {"state": "context", "ttl_seconds": 86401},
        {"state": "context", "ttl_seconds": True},
        {"state": "context", "instructions": ""},
    ],
)
async def test_bad_context_never_reaches_backend(tmp_path: Path, body: Any) -> None:
    async with harness(tmp_path) as h:
        response = await h.client.post("/v1/contexts", json=body)
        assert response.status_code == 422
        assert not h.transport.requests


async def test_body_bounds_media_type_nonfinite_and_handle_validation(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path, max_request_bytes=128) as h:
        response = await h.client.post("/v1/contexts", json={"state": "x" * 200})
        assert response.status_code == 413

        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"state":"'
            yield b"x" * 200
            yield b'"}'

        response = await h.client.post(
            "/v1/contexts",
            content=chunks(),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        response = await h.client.post("/v1/contexts", content="{}")
        assert response.status_code == 415
        response = await h.client.post(
            "/v1/contexts",
            content='{"state":{"x":NaN}}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        response = await h.client.post(
            "/v1/contexts", content="{", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400
        assert (await h.client.get("/v1/contexts/not-a-handle")).status_code == 422
        assert not h.transport.requests


async def test_backend_response_is_bounded_and_validated(tmp_path: Path) -> None:
    async with harness(tmp_path, max_response_bytes=512) as h:
        h.transport.fault = lambda _: httpx.Response(200, content=b"x" * 513)
        response = await h.client.get("/v1/models")
        assert response.status_code == 502
        h.transport.fault = lambda _: httpx.Response(200, json={"models": []})
        response = await h.client.get("/v1/models")
        assert response.status_code == 502


async def test_discovery_and_health_verify_snapshot_backend(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        assert (await h.client.get("/health")).json() == {
            "status": "ok",
            "backend": "ok",
        }
        model = (await h.client.get("/v1/models")).json()
        assert model["models"][0]["capabilities"]["readout_blocks"] == [30]
        assert model["harness"]["automatic_recapture"] is True
        assert model["harness"]["explicit_context_recapture"] is False
        created = await h.context()
        metadata = await h.client.get("/v1/contexts/" + created["context_id"])
        assert metadata.status_code == 200
        assert "state" not in metadata.json()
        assert "snapshot_id" not in metadata.json()


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/backend",
        "http://a:secret@localhost",
        "http://localhost/v2",
        "http://localhost?token=secret",
        "http://localhost#fragment",
    ],
)
def test_backend_url_is_fixed_origin_without_credentials(url: str) -> None:
    with pytest.raises(ValueError):
        HarnessConfig(backend_url=url)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1.0])
def test_nonfinite_timeout_refused(timeout: float) -> None:
    with pytest.raises(ValueError):
        HarnessConfig(request_timeout=timeout)
