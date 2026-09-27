"""/v2 routes under the full-v1 profile (ADR 0009) with the in-memory worker.

One whole-model checkpoint at the block count (42 here, like Gemma 4 E4B), no
promotion, and the final residual tagged as such rather than as block 30. The
split profile's own behavior is covered by test_snapshots_app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from unridden.api.snapshots.schema import WorkerHello
from unridden.tests.test_snapshots_app import (
    CONTEXT_BODY,
    HELLO,
    FakeSnapshotBackend,
    _decision,
    client_for,
)

FULL_HELLO = HELLO.model_copy(
    update={"profile": "full-v1", "n_layer": 42, "n_embd": 2560, "split_block": None}
)
FULL_BODY = CONTEXT_BODY | {"checkpoints": [42]}


def _full_backend() -> FakeSnapshotBackend:
    backend = FakeSnapshotBackend()
    backend.hello = FULL_HELLO
    return backend


def _hello(**overrides: object) -> dict[str, object]:
    return FULL_HELLO.model_dump() | overrides


def test_hello_shape_follows_the_profile() -> None:
    hello = WorkerHello.model_validate(_hello())
    assert hello.checkpoints == [42]
    assert hello.final_block == 42
    assert hello.layer_map == {"lower": [0, 42]}
    assert hello.execution_mode == "full"
    assert hello.final_representation == "raw_residual_after_final_block"
    assert HELLO.checkpoints == [18, 30]
    assert HELLO.layer_map == {"lower": [0, 18], "upper": [18, 30]}
    with pytest.raises(ValidationError, match="no split block"):
        WorkerHello.model_validate(_hello(split_block=18))
    with pytest.raises(ValidationError, match="no reference context"):
        WorkerHello.model_validate(_hello(reference_context=True))
    with pytest.raises(ValidationError, match="30 blocks split at 18"):
        WorkerHello.model_validate(_hello(profile="split18-30-v1", split_block=18))


@pytest.mark.asyncio
async def test_create_returns_one_whole_model_snapshot(tmp_path: Path) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        response = await client.post("/v2/snapshots", json=FULL_BODY)

    assert response.status_code == 200, response.text
    (row,) = response.json()["snapshots"]
    assert row["completed_blocks"] == 42
    assert row.get("parent") is None
    assert row["capabilities"] == ["continue", "inspect"]


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoints", [[18], [30], [18, 42]])
async def test_checkpoints_the_profile_lacks_are_refused(
    tmp_path: Path, checkpoints: list[int]
) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        response = await client.post(
            "/v2/snapshots", json=FULL_BODY | {"checkpoints": checkpoints}
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_unavailable"


@pytest.mark.asyncio
async def test_the_split_profile_refuses_a_full_depth_checkpoint(
    tmp_path: Path,
) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        response = await client.post("/v2/snapshots", json=FULL_BODY)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_unavailable"


@pytest.mark.asyncio
async def test_decide_reads_out_at_the_block_count(tmp_path: Path) -> None:
    backend = _full_backend()
    async with client_for(backend, tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=FULL_BODY)).json()
        snap = created["snapshots"][0]["id"]
        decided = await client.post(
            "/v2/decisions",
            json=_decision(snap, readout={"completed_blocks": 42}),
        )
        at_30 = await client.post("/v2/decisions", json=_decision(snap))

    assert decided.status_code == 200, decided.text
    usage = decided.json()["snapshot_usage"]
    assert usage["profile"] == "full-v1"
    assert usage["promotion"] == "none"
    assert usage["effective_parent"] == snap
    assert backend.promote_calls == 0
    assert at_30.status_code == 422
    assert at_30.json()["error"]["code"] == "capability_unavailable"


@pytest.mark.asyncio
async def test_saved_child_and_replace_question(tmp_path: Path) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=FULL_BODY)).json()
        snap = created["snapshots"][0]["id"]
        saved = await client.post(
            "/v2/decisions",
            json=_decision(
                snap, readout={"completed_blocks": 42}, save_result_snapshot=True
            ),
        )
        child = saved.json()["snapshot_usage"]["child"]["status"]
        meta = await client.get(f"/v2/snapshots/{child}")
        replaced = await client.post(
            "/v2/decisions",
            json=_decision(child, readout={"completed_blocks": 42})
            | {"snapshot": {"id": child, "relationship": "replace_question"}},
        )

    assert saved.status_code == 200, saved.text
    assert meta.json()["completed_blocks"] == 42
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["snapshot_usage"]["effective_parent"] == snap


@pytest.mark.asyncio
async def test_state_evaluation_tags_the_final_residual(tmp_path: Path) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=FULL_BODY)).json()
        snap = created["snapshots"][0]["id"]
        response = await client.post(
            "/v2/state-evaluations",
            json={
                "snapshot": {"id": snap, "relationship": "followup"},
                "prompt": "focus on the timeline",
                "readout": {"completed_blocks": 42, "export": ["last_residual"]},
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["vectors"]["last_residual"]["representation"] == (
        "raw_residual_after_final_block"
    )


@pytest.mark.asyncio
async def test_vectors_name_the_final_residual(tmp_path: Path) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=FULL_BODY)).json()
        snap = created["snapshots"][0]["id"]
        final = await client.get(
            f"/v2/snapshots/{snap}/vectors",
            params={"which": "final", "row_begin": 0, "row_end": 4},
        )
        h30 = await client.get(
            f"/v2/snapshots/{snap}/vectors",
            params={"which": "h30", "row_begin": 0, "row_end": 4},
        )

    assert final.status_code == 200, final.text
    assert final.json()["representation"] == "raw_residual_after_final_block"
    assert h30.status_code == 422


@pytest.mark.asyncio
async def test_rider_continues_a_full_depth_snapshot(tmp_path: Path) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=FULL_BODY)).json()
        snap = created["snapshots"][0]["id"]
        response = await client.post(
            "/v2/rider",
            json={
                "snapshot": {"id": snap, "relationship": "followup"},
                "prompt": "draft a reply",
                "max_tokens": 3,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["token_ids"] == [100, 101, 102]


@pytest.mark.asyncio
async def test_models_reports_the_profile_limits(tmp_path: Path) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        models = await client.get("/v2/models")

    profile = models.json()["models"][0]
    assert profile["profile"] == "full-v1"
    assert profile["limits"]["checkpoints"] == [42]
    assert profile["capabilities"]["operations"] == ["continue", "inspect"]
    assert profile["capabilities"]["readout_blocks"] == [42]
    assert profile["capabilities"]["vectors"] == ["final", "last_normalized"]


@pytest.mark.asyncio
async def test_disk_manifest_records_the_whole_model_range(tmp_path: Path) -> None:
    async with client_for(_full_backend(), tmp_path) as client:
        created = await client.post(
            "/v2/snapshots", json=FULL_BODY | {"persistence": "disk"}
        )
        snap = created.json()["snapshots"][0]["id"]

    manifest = json.loads((tmp_path / "snap" / snap / "manifest.json").read_text())
    assert manifest["profile"] == "full-v1"
    assert manifest["layer_map"] == {"lower": [0, 42]}
    assert manifest["completed_blocks"] == 42
