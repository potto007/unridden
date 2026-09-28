"""Ordinary callers get isolated automatic captures without managing handles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from unridden.tests.test_harness import QUESTIONS, harness


def body(**overrides: Any) -> dict[str, Any]:
    return {"state": {"ticket": "charged twice"}, "questions": QUESTIONS, **overrides}


async def test_cookie_reuse_and_question_isolation(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        first = await h.client.post("/v1/decisions", json=body())
        assert first.status_code == 200, first.text
        one = first.json()
        assert one["context_usage"]["cache"] == "miss"
        assert one["context_usage"]["capture_reason"] == "new_session"
        assert "context_id" not in one["context_usage"]
        assert "HttpOnly" in first.headers["set-cookie"]
        assert "SameSite=strict" in first.headers["set-cookie"]
        other = await h.client.post(
            "/v1/decisions",
            json=body(
                questions={
                    "different": {
                        "type": "noul",
                        "instructions": "Is this about astronomy?",
                    }
                }
            ),
        )
        assert other.status_code == 200
        last = (await h.client.post("/v1/decisions", json=body())).json()
        assert last["answers"] == one["answers"]
        assert last["context_usage"]["cache"] == "hit"
        assert last["context_usage"]["capture_ms_this_request"] == 0
        assert last["context_usage"]["successful_decisions"] == 3
        assert last["context_usage"]["expires_at"] == one["context_usage"]["expires_at"]
        captures = [
            data for _, path, data in h.transport.requests if path == "/v2/snapshots"
        ]
        assert len(captures) == 1
        assert captures[0]["input"] == {"kind": "context", "state": body()["state"]}
        assert len(h.worker._snaps) == 1


async def test_changed_state_or_instructions_replace_context(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        first = (await h.client.post("/v1/decisions", json=body())).json()
        old = first["snapshot_usage"]["requested_parent"]
        for change in (
            {"state": "new ticket"},
            {"state": "new ticket", "instructions": "Apply policy."},
        ):
            response = await h.client.post("/v1/decisions", json=body(**change))
            assert response.status_code == 200
            result = response.json()
            assert result["context_usage"]["cache"] == "miss"
            assert (
                result["context_usage"]["capture_reason"] == "state_or_profile_changed"
            )
            assert result["snapshot_usage"]["requested_parent"] != old
            assert len(h.worker._snaps) == 1
            assert old not in h.worker._snaps
            old = result["snapshot_usage"]["requested_parent"]


async def test_identical_state_never_shared_across_cookies(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        first = (await h.client.post("/v1/decisions", json=body())).json()
        h.client.cookies.clear()
        other = (await h.client.post("/v1/decisions", json=body())).json()
        assert other["context_usage"]["cache"] == "miss"
        assert (
            first["snapshot_usage"]["requested_parent"]
            != other["snapshot_usage"]["requested_parent"]
        )
        assert len(h.worker._snaps) == 2


async def test_expiry_recaptures_without_manual_work(tmp_path: Path) -> None:
    async with harness(tmp_path, automatic_ttl=10) as h:
        first = (await h.client.post("/v1/decisions", json=body())).json()
        h.clock[0] += 9
        assert (await h.client.post("/v1/decisions", json=body())).json()[
            "context_usage"
        ]["cache"] == "hit"
        h.clock[0] += 1
        result = (await h.client.post("/v1/decisions", json=body())).json()
        assert result["context_usage"]["capture_reason"] == "expired_or_evicted"
        assert result["context_usage"]["cache"] == "miss"
        assert (
            result["snapshot_usage"]["requested_parent"]
            != first["snapshot_usage"]["requested_parent"]
        )
        assert len(h.worker._snaps) == 1


async def test_definite_backend_loss_recaptures_once(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        first = (await h.client.post("/v1/decisions", json=body())).json()
        await h.service.client.delete(
            "/v2/snapshots/" + first["snapshot_usage"]["requested_parent"]
        )
        start = len(h.transport.requests)
        response = await h.client.post("/v1/decisions", json=body())
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["context_usage"]["capture_reason"] == "backend_lost"
        assert result["context_usage"]["cache"] == "miss"
        assert result["answers"] == first["answers"]
        assert (
            len(
                [
                    1
                    for _, path, _ in h.transport.requests[start:]
                    if path == "/v2/snapshots"
                ]
            )
            == 1
        )


async def test_timeout_never_replays_or_recaptures(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        await h.client.post("/v1/decisions", json=body())

        def fault(request: httpx.Request) -> httpx.Response | None:
            if request.url.path == "/v2/decisions":
                raise httpx.ReadTimeout("not exposed", request=request)
            return None

        h.transport.fault = fault
        start = len(h.transport.requests)
        response = await h.client.post("/v1/decisions", json=body())
        assert response.status_code == 504
        assert (
            len(
                [
                    1
                    for _, path, _ in h.transport.requests[start:]
                    if path == "/v2/decisions"
                ]
            )
            == 1
        )
        assert not any(
            path == "/v2/snapshots" for _, path, _ in h.transport.requests[start:]
        )


async def test_lru_capacity_protects_manual_contexts(tmp_path: Path) -> None:
    async with harness(tmp_path, max_contexts=2) as h:
        explicit = await h.context()
        for _ in range(5):
            h.client.cookies.clear()
            response = await h.client.post("/v1/decisions", json=body())
            assert response.status_code == 200, response.text
            assert response.json()["context_usage"]["cache"] == "miss"
            assert explicit["context_id"] in h.service.entries
            assert len(h.service.entries) == 2
            assert len(h.worker._snaps) == 2


async def test_profile_identity_change_rebuilds_scope(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        first = (await h.client.post("/v1/decisions", json=body())).json()
        model = (await h.service.client.get("/v2/models")).json()
        model["models"][0]["prompt_version"] += "-new"
        h.transport.fault = lambda request: (
            httpx.Response(200, json=model)
            if request.url.path == "/v2/models"
            else None
        )
        response = await h.client.post("/v1/decisions", json=body())
        assert response.status_code == 200
        result = response.json()
        assert result["context_usage"]["capture_reason"] == "state_or_profile_changed"
        assert (
            result["snapshot_usage"]["requested_parent"]
            != first["snapshot_usage"]["requested_parent"]
        )


async def test_failed_first_request_releases_unclaimed_capture(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        h.transport.fault = lambda request: (
            httpx.Response(422, json={"error": {"code": "budget_error"}})
            if request.url.path == "/v2/decisions"
            else None
        )
        response = await h.client.post("/v1/decisions", json=body())
        assert response.status_code == 422
        assert not h.service.entries
        assert not h.worker._snaps


async def test_manual_override_keeps_snapshots_but_disables_automatic_choices(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path, automatic_enabled=False) as h:
        rejected = await h.client.post("/v1/decisions", json=body())
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "manual_context_required"
        assert not h.transport.requests
        manual = await h.context()
        result = await h.decide(manual["context_id"])
        assert result.status_code == 200
        assert result.json()["context_usage"]["mode"] == "immutable_context"
        assert not h.client.cookies
        assert (
            await h.client.delete("/v1/contexts/" + manual["context_id"])
        ).status_code == 200
