"""End-to-end /v2 route tests with an in-memory protocol-faithful backend.

The fake worker enforces the semantics the design promises: branches are pure
functions of their own rendered prompt (so A, B, A is identical), the parent is
never mutated, promotion happens once, and nothing generates tokens.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from unridden.api.app import ApiConfig, create_app
from unridden.api.native.build import TESTED_LLAMA_REVISION
from unridden.api.schema import BackendProfile, FullVocabularyArgmax
from unridden.api.snapshots.errors import SnapshotUnavailableError
from unridden.api.snapshots.schema import (
    BlockTokens,
    Boundary,
    ExportKind,
    RiderStep,
    SnapshotBytes,
    TopLogit,
    VectorArtifact,
    WorkerCreated,
    WorkerCreateTiming,
    WorkerHello,
    WorkerPromoted,
    WorkerPromoteTiming,
    WorkerReadout,
    WorkerResult,
    WorkerRide,
    WorkerRideTiming,
    WorkerSnapshotBranch,
    WorkerSnapshotQuestionResult,
    WorkerSnapshotRow,
    WorkerState,
)

HELLO = WorkerHello(
    type="hello",
    protocol="unridden-snapshot-v1",
    profile="split18-30-v1",
    model_id="local-gemma-unridden-v1",
    model_name="fixture",
    model_sha256="a" * 64,
    runtime_sha256="b" * 64,
    labels=["A", "B", "C", "D"],
    label_token_ids=[11, 12, 13, 14],
    context_size=2048,
    batch_size=256,
    ubatch_size=256,
    threads=8,
    n_layer=30,
    n_embd=2816,
    split_block=18,
    reference_context=False,
    context_prompt_version="unridden-gemma-context-v1",
    generated_tokens=0,
    rider_mode=True,
    callbacks_enabled=False,
)

V1_PROFILE = BackendProfile(
    model_id="local-gemma-unridden-v1",
    model_name="fixture",
    model_sha256="a" * 64,
    runtime_sha256="b" * 64,
    llama_revision=TESTED_LLAMA_REVISION,
    tested_revision=True,
    labels=["A", "B", "C", "D"],
    label_token_ids=[11, 12, 13, 14],
    context_size=2048,
    batch_size=256,
    ubatch_size=256,
    threads=8,
    max_questions=32,
    batched_mode=False,
    batched_context=0,
    generated_tokens=0,
    callbacks_enabled=False,
    execution_mode="full",
)


class MinimalV1Backend:
    def __init__(self) -> None:
        self.profile: BackendProfile | None = None
        self.ready = False

    async def start(self) -> BackendProfile:
        self.profile = V1_PROFILE
        self.ready = True
        return V1_PROFILE

    async def close(self) -> None:
        self.ready = False

    async def evaluate(self, batch: Any, *, timeout: float) -> Any:  # pragma: no cover
        raise AssertionError("v1 backend is not exercised by these tests")


def _tokens(content: str) -> int:
    return max(1, len(content.split()))


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _logits(content: str, count: int) -> list[float]:
    seed = int(hashlib.sha256(content.encode()).hexdigest()[:8], 16)
    return [((seed >> (index * 3)) & 0x7) * 0.5 for index in range(count)]


def _vector(values: list[float], representation: str) -> VectorArtifact:
    encoded = base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()
    return VectorArtifact(
        dtype="f32",
        shape=[len(values)],
        representation=representation,  # type: ignore[arg-type]
        base64=encoded,
    )


class FakeSnapshotBackend:
    """An in-memory worker that keeps parents immutable and branches pure."""

    def __init__(self) -> None:
        self.profile: WorkerHello | None = None
        self.ready = False
        self.promote_calls = 0
        self._snaps: dict[str, dict[str, Any]] = {}
        self._resident: set[str] = set()
        self.block = asyncio.Event()
        self.entered = asyncio.Event()
        self.gate = False
        self.start_calls = 0
        self.fail_start = False
        self.hello = HELLO

    async def start(self) -> WorkerHello:
        self.start_calls += 1
        if self.fail_start:
            raise SnapshotUnavailableError("snapshot backend failed to start")
        # A fresh worker process holds no snapshots, like the real one.
        self._snaps.clear()
        self._resident.clear()
        self.profile = self.hello
        self.ready = True
        return self.hello

    def crash(self) -> None:
        """What `_invalidate` leaves behind: the child is gone."""
        self.ready = False
        self.profile = None

    async def close(self) -> None:
        self.ready = False

    def _readout(self, content: str, count: int) -> WorkerReadout:
        logits = _logits(content, count)
        return WorkerReadout(
            label_logits=logits,
            label_token_ids=HELLO.label_token_ids[:count],
            allowed_label_mass=0.5,
            full_vocabulary_argmax=FullVocabularyArgmax(
                token_id=HELLO.label_token_ids[0], logit=max(logits) + 1.0
            ),
        )

    async def create(
        self,
        *,
        messages: list[dict[str, str]],
        answer_prefix: str,
        freeze: dict[str, Any],
        checkpoints: dict[str, str],
        labels: list[str] | None,
        top_logits: int,
        timeout: float | None = None,
    ) -> WorkerCreated:
        content = "".join(message["content"] for message in messages) + answer_prefix
        tokens = _tokens(content)
        kind: Boundary = "context" if freeze.get("kind") == "context" else "readout"
        sha = _sha(content)
        rows: list[WorkerSnapshotRow] = []
        # Split: 18 and/or 30. full-v1: only the block count, one whole range.
        split = self.hello.split_block
        final = self.hello.final_block
        id18 = checkpoints.get(str(split)) if split is not None else None
        id30 = checkpoints.get(str(final))
        paired = bool(id18 and id30)
        whole = split is None
        if id18:
            self._snaps[id18] = {"tokens": tokens, "content": content, "kind": kind}
            self._resident.add(id18)
            rows.append(
                WorkerSnapshotRow(
                    snapshot_id=id18,
                    completed_blocks=18,
                    parent=None,
                    tokens=tokens,
                    bytes=SnapshotBytes(lower_kv=tokens, upper_kv=0, h18=tokens, h30=0),
                    kind=kind,
                    prompt_sha256=sha,
                )
            )
        if id30:
            parent = id18 if paired else None
            self._snaps[id30] = {"tokens": tokens, "content": content, "kind": kind}
            self._resident.add(id30)
            rows.append(
                WorkerSnapshotRow(
                    snapshot_id=id30,
                    completed_blocks=final,
                    parent=parent,
                    tokens=tokens,
                    bytes=SnapshotBytes(
                        lower_kv=0 if paired else tokens,
                        upper_kv=0 if whole else tokens,
                        # Like the worker: an unpaired 30 keeps H18 itself.
                        h18=0 if (paired or whole) else tokens,
                        h30=tokens,
                    ),
                    kind=kind,
                    prompt_sha256=sha,
                )
            )
        readout = self._readout(content, len(labels)) if labels and id30 else None
        return WorkerCreated(
            type="created",
            id="c",
            prompt_sha256=_sha(content),
            tokens=tokens,
            snapshots=rows,
            readout=readout,
            block_tokens=BlockTokens(
                lower=tokens, upper=tokens if (id30 and not whole) else 0
            ),
            timing_ms=WorkerCreateTiming(lower=1.0, upper=1.0, total=3.0),
            generated_tokens=0,
        )

    async def promote(
        self, *, snapshot_id: str, new_id: str, timeout: float | None = None
    ) -> WorkerPromoted:
        self.promote_calls += 1
        parent = self._snaps[snapshot_id]
        tokens = parent["tokens"]
        kind = parent["kind"]
        self._snaps[new_id] = {
            "tokens": tokens,
            "content": parent["content"],
            "kind": kind,
        }
        self._resident.add(new_id)
        return WorkerPromoted(
            type="promoted",
            id="p",
            snapshot=WorkerSnapshotRow(
                snapshot_id=new_id,
                completed_blocks=30,
                parent=snapshot_id,
                tokens=tokens,
                bytes=SnapshotBytes(lower_kv=0, upper_kv=tokens, h18=0, h30=tokens),
                kind=kind,
                prompt_sha256=_sha(parent["content"]),
            ),
            block_tokens=BlockTokens(lower=0, upper=tokens),
            timing_ms=WorkerPromoteTiming(upper=1.0, total=2.0),
            generated_tokens=0,
        )

    def _branch(
        self, snapshot_id: str, question: dict[str, Any]
    ) -> WorkerSnapshotQuestionResult:
        content = (
            "".join(message["content"] for message in question["messages"])
            + question["answer_prefix"]
        )
        last_user = [
            message["content"]
            for message in question["messages"]
            if message["role"] == "user"
        ][-1]
        suffix = _tokens(last_user)
        reused = self._snaps[snapshot_id]["tokens"]
        labels = question["labels"]
        logits = _logits(content, len(labels))
        child = None
        if question["save_as"]:
            child_id = question["save_as"]
            self._snaps[child_id] = {
                "tokens": reused + suffix,
                "content": content,
                "kind": "readout",
            }
            self._resident.add(child_id)
            child = WorkerSnapshotRow(
                snapshot_id=child_id,
                completed_blocks=self.hello.final_block,
                parent=snapshot_id,
                tokens=reused + suffix,
                bytes=SnapshotBytes(
                    lower_kv=0, upper_kv=suffix, h18=0, h30=reused + suffix
                ),
                kind="readout",
                prompt_sha256=_sha(content),
            )
        return WorkerSnapshotQuestionResult(
            id=question["id"],
            label_logits=logits,
            label_token_ids=HELLO.label_token_ids[: len(labels)],
            allowed_label_mass=0.5,
            full_vocabulary_argmax=FullVocabularyArgmax(
                token_id=HELLO.label_token_ids[0], logit=max(logits) + 1.0
            ),
            prompt_sha256=_sha(content),
            prompt_tokens=reused + suffix,
            processed_tokens=suffix,
            reused_tokens=reused,
            cache_cleared=False,
            evaluation_mode="sequential",
            batch_sequences=1,
            timing_ms=1.0,
            snapshot=WorkerSnapshotBranch(
                parent=snapshot_id,
                suffix_tokens=suffix,
                block_tokens=BlockTokens(lower=suffix, upper=suffix),
                restore="resident",
                restored_bytes=0,
                restore_ms=0.0,
                inference_ms=1.0,
                child=child,
            ),
        )

    async def evaluate(
        self,
        *,
        snapshot_id: str,
        readout_blocks: int,
        questions: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> WorkerResult:
        if self.gate:
            self.entered.set()
            await self.block.wait()
        rows = [self._branch(snapshot_id, question) for question in questions]
        return WorkerResult(
            type="result",
            id="e",
            model_sha256=HELLO.model_sha256,
            runtime_sha256=HELLO.runtime_sha256,
            generated_tokens=0,
            callbacks_enabled=False,
            execution_mode=self.hello.execution_mode,
            questions=rows,
        )

    async def state_eval(
        self,
        *,
        snapshot_id: str,
        messages: list[dict[str, str]],
        answer_prefix: str,
        export: list[ExportKind],
        top_logits: int,
        save_as: str | None,
        timeout: float | None = None,
    ) -> WorkerState:
        content = "".join(message["content"] for message in messages) + answer_prefix
        suffix = _tokens([m["content"] for m in messages if m["role"] == "user"][-1])
        reused = self._snaps[snapshot_id]["tokens"]
        vectors: dict[str, VectorArtifact] = {}
        if "last_residual" in export:
            vectors["last_residual"] = _vector(
                [1.0, 2.0, 3.0], self.hello.final_representation
            )
        if "last_normalized" in export:
            vectors["last_normalized"] = _vector(
                [0.5, 0.5], "post_final_norm_head_input"
            )
        tops = (
            [
                TopLogit(token_id=index, logit=float(index))
                for index in range(top_logits)
            ]
            if "top_logits" in export
            else []
        )
        child = None
        if save_as:
            self._snaps[save_as] = {
                "tokens": reused + suffix,
                "content": content,
                "kind": "readout",
            }
            self._resident.add(save_as)
            child = WorkerSnapshotRow(
                snapshot_id=save_as,
                completed_blocks=self.hello.final_block,
                parent=snapshot_id,
                tokens=reused + suffix,
                bytes=SnapshotBytes(
                    lower_kv=0, upper_kv=suffix, h18=0, h30=reused + suffix
                ),
                kind="readout",
                prompt_sha256=_sha(content),
            )
        return WorkerState(
            type="state",
            id="s",
            parent=snapshot_id,
            suffix_tokens=suffix,
            prompt_sha256=_sha(content),
            prompt_tokens=reused + suffix,
            block_tokens=BlockTokens(lower=suffix, upper=suffix),
            restore="resident",
            restored_bytes=0,
            restore_ms=0.0,
            inference_ms=1.0,
            timing_ms=1.0,
            vectors=vectors,  # type: ignore[arg-type]
            top_logits=tops,
            child=child,
            generated_tokens=0,
        )

    async def ride(
        self,
        *,
        snapshot_id: str,
        messages: list[dict[str, str]],
        answer_prefix: str,
        max_tokens: int,
        top_logits: int,
        timeout: float | None = None,
    ) -> WorkerRide:
        content = "".join(message["content"] for message in messages) + answer_prefix
        suffix = _tokens([m["content"] for m in messages if m["role"] == "user"][-1])
        reused = self._snaps[snapshot_id]["tokens"]
        token_ids = list(range(100, 100 + max_tokens))
        steps = [
            RiderStep(
                token_id=token,
                top_logits=[TopLogit(token_id=token, logit=1.0)][:top_logits],
            )
            for token in token_ids
        ]
        return WorkerRide(
            type="ride_result",
            id="g",
            mode="snapshot",
            execution_mode=self.hello.execution_mode,
            text=" ".join(str(token) for token in token_ids),
            token_ids=token_ids,
            stop_reason="max_tokens",
            prompt_sha256=_sha(content),
            prompt_tokens=reused + suffix,
            reused_tokens=reused,
            prefilled_tokens=suffix,
            generated_tokens=max_tokens,
            block_tokens={
                "lower": suffix + max_tokens - 1,
                "upper": suffix + max_tokens - 1,
            },
            restore="resident",
            restored_bytes=0,
            timing_ms=WorkerRideTiming(
                restore=0.0, time_to_first_token=1.0, decode=2.0, total=3.0
            ),
            decode_tokens_per_second=100.0,
            steps=steps if top_logits else [],
        )

    async def inspect(self, *, snapshot_id: str, timeout: float | None = None) -> Any:
        raise AssertionError("inspect is served from the store")

    async def vectors(
        self,
        *,
        snapshot_id: str,
        which: str,
        row_begin: int,
        row_end: int,
        timeout: float | None = None,
    ) -> Any:
        from unridden.api.snapshots.schema import WorkerTensor, WorkerVectors

        representation = {
            "h18": "raw_residual_after_block_18",
            "h30": self.hello.final_representation,
            "last_normalized": "post_final_norm_head_input",
        }[which]
        returned = row_end - row_begin
        values = [float(index) for index in range(returned * 2)]
        encoded = base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()
        return WorkerVectors(
            type="vectors",
            id="v",
            snapshot_id=snapshot_id,
            which=which,  # type: ignore[arg-type]
            rows=row_end,  # a plausible total the slice fits within
            row_begin=row_begin,
            row_end=row_end,
            tensor=WorkerTensor(
                dtype="f32",
                byte_order="little",
                shape=[returned, 2],
                representation=representation,
                base64=encoded,
            ),
        )

    async def save(self, snapshot_id: str, directory: Path) -> Any:
        payload = snapshot_id.encode()
        (directory / "lower_kv.bin").write_bytes(payload)
        from unridden.api.snapshots.schema import WorkerFileEntry

        return {
            "lower_kv.bin": WorkerFileEntry(
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
            )
        }

    async def load(
        self, snapshot_id: str, directory: Path, files: dict[str, str]
    ) -> None:
        self._resident.add(snapshot_id)

    async def drop(self, snapshot_id: str) -> int:
        self._resident.discard(snapshot_id)
        return 10


class DiskBackedFakeSnapshotBackend(FakeSnapshotBackend):
    """Reload the fake's snapshot state from blobs in a fresh worker instance."""

    async def save(self, snapshot_id: str, directory: Path) -> Any:
        from unridden.api.snapshots.schema import WorkerFileEntry

        payload = json.dumps(self._snaps[snapshot_id], sort_keys=True).encode()
        (directory / "state.json").write_bytes(payload)
        return {
            "state.json": WorkerFileEntry(
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
            )
        }

    async def load(
        self, snapshot_id: str, directory: Path, files: dict[str, str]
    ) -> None:
        payload = (directory / "state.json").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == files["state.json"]
        self._snaps[snapshot_id] = json.loads(payload)
        self._resident.add(snapshot_id)


@asynccontextmanager
async def client_for(
    backend: FakeSnapshotBackend, tmp_path: Path, **overrides: Any
) -> AsyncIterator[httpx.AsyncClient]:
    config = ApiConfig(
        snapshots_enabled=True,
        snapshot_store_dir=tmp_path / "snap",
        snapshot_host_bytes=1 << 30,
        **overrides,
    )
    app = create_app(
        config,
        backend_factory=lambda _: MinimalV1Backend(),
        snapshot_backend_factory=lambda _: backend,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        yield client


CONTEXT_BODY = {
    "input": {"kind": "context", "state": "refund not arrived"},
    "checkpoints": [18, 30],
    "ttl_seconds": 600,
}


def _decision(
    snapshot_id: str, question_id: str = "status", **overrides: Any
) -> dict[str, Any]:
    body = {
        "snapshot": {"id": snapshot_id, "relationship": "followup"},
        "questions": {
            question_id: {
                "type": "choice",
                "instructions": "what to investigate?",
                "criteria": {"new_request": "no refund", "missing_refund": "lost"},
            }
        },
        "readout": {"completed_blocks": 30},
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_create_returns_two_context_snapshots(tmp_path: Path) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        response = await client.post("/v2/snapshots", json=CONTEXT_BODY)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["usage"] == {"generated_tokens": 0}
    snaps = {row["completed_blocks"]: row for row in body["snapshots"]}
    assert set(snaps) == {18, 30}
    assert snaps[18]["boundary"] == "context"
    assert snaps[18]["capabilities"] == ["continue", "promote", "inspect"]
    assert snaps[30]["parent"] == snaps[18]["id"]


@pytest.mark.asyncio
async def test_pair_creation_fails_cleanly_when_budget_cannot_hold_parent(
    tmp_path: Path,
) -> None:
    backend = FakeSnapshotBackend()
    app = create_app(
        ApiConfig(
            snapshots_enabled=True,
            v1_enabled=False,
            snapshot_store_dir=tmp_path / "snap",
            snapshot_host_bytes=80,
        ),
        snapshot_backend_factory=lambda _: backend,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        response = await client.post("/v2/snapshots", json=CONTEXT_BODY)

    assert response.status_code == 507
    assert app.state.snapshots.store._records == {}
    assert backend._resident == set()


@pytest.mark.asyncio
async def test_rejected_saved_children_do_not_accumulate_in_worker(
    tmp_path: Path,
) -> None:
    backend = FakeSnapshotBackend()
    app = create_app(
        ApiConfig(
            snapshots_enabled=True,
            v1_enabled=False,
            snapshot_store_dir=tmp_path / "snap",
            snapshot_host_bytes=100,
        ),
        snapshot_backend_factory=lambda _: backend,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        created = await client.post(
            "/v2/snapshots", json=CONTEXT_BODY | {"checkpoints": [30]}
        )
        assert created.status_code == 200
        snapshot_id = created.json()["snapshots"][0]["id"]
        for _ in range(2):
            rejected = await client.post(
                "/v2/decisions",
                json=_decision(snapshot_id, save_result_snapshot=True),
            )
            assert rejected.status_code == 507

    assert backend._resident == {snapshot_id}
    assert set(app.state.snapshots.store._records) == {snapshot_id}


@pytest.mark.asyncio
async def test_decide_promotes_once_and_reuses(tmp_path: Path) -> None:
    backend = FakeSnapshotBackend()
    async with client_for(backend, tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap18 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 18
        )

        first = await client.post("/v2/decisions", json=_decision(snap18))
        second = await client.post("/v2/decisions", json=_decision(snap18))

    assert first.status_code == 200, first.text
    body = first.json()
    assert body["usage"]["output_tokens"] == 0
    usage = body["snapshot_usage"]
    assert usage["requested_parent"] == snap18
    assert usage["promotion"] == "performed"
    assert usage["effective_parent"] != snap18
    assert usage["profile"] == "split18-30-v1"
    assert "status" in usage["suffix_tokens"]
    # The second decision reuses the memoized promotion.
    assert second.json()["snapshot_usage"]["promotion"] == "reused"
    assert backend.promote_calls == 1


@pytest.mark.asyncio
async def test_durable_promotion_reused_after_fresh_service_and_worker(
    tmp_path: Path,
) -> None:
    first_backend = DiskBackedFakeSnapshotBackend()
    body = CONTEXT_BODY | {"checkpoints": [18], "persistence": "disk"}
    async with client_for(first_backend, tmp_path) as client:
        created = await client.post("/v2/snapshots", json=body)
        assert created.status_code == 200, created.text
        snap18 = created.json()["snapshots"][0]["id"]
        first = await client.post("/v2/decisions", json=_decision(snap18))
        assert first.status_code == 200, first.text
        promoted = first.json()["snapshot_usage"]["effective_parent"]
        assert first.json()["snapshot_usage"]["promotion"] == "performed"
        assert first_backend.promote_calls == 1

    disk_ids = {path.name for path in (tmp_path / "snap").iterdir()}
    assert disk_ids == {snap18, promoted}

    restarted_backend = DiskBackedFakeSnapshotBackend()
    assert restarted_backend._snaps == {}
    async with client_for(restarted_backend, tmp_path) as client:
        reused = await client.post("/v2/decisions", json=_decision(snap18))
        assert reused.status_code == 200, reused.text
        usage = reused.json()["snapshot_usage"]
        assert usage["promotion"] == "reused"
        assert usage["effective_parent"] == promoted
        assert restarted_backend.promote_calls == 0
        assert {path.name for path in (tmp_path / "snap").iterdir()} == disk_ids

        deleted = await client.delete(f"/v2/snapshots/{promoted}")
        assert deleted.status_code == 200, deleted.text
        recreated = await client.post("/v2/decisions", json=_decision(snap18))
        assert recreated.status_code == 200, recreated.text
        fresh_usage = recreated.json()["snapshot_usage"]
        assert fresh_usage["promotion"] == "performed"
        assert fresh_usage["effective_parent"] != promoted
        assert restarted_backend.promote_calls == 1
        assert len(list((tmp_path / "snap").iterdir())) == 2


@pytest.mark.asyncio
async def test_branches_are_isolated_and_the_parent_is_immutable(
    tmp_path: Path,
) -> None:
    backend = FakeSnapshotBackend()
    async with client_for(backend, tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap30 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 30
        )

        a1 = await client.post("/v2/decisions", json=_decision(snap30, "a"))
        b1 = await client.post("/v2/decisions", json=_decision(snap30, "b"))
        a2 = await client.post("/v2/decisions", json=_decision(snap30, "a"))
        meta = await client.get(f"/v2/snapshots/{snap30}")

    answer_a1 = a1.json()["answers"]["a"]
    answer_a2 = a2.json()["answers"]["a"]
    # A, B, A: the two A branches are identical regardless of the B between them.
    assert answer_a1 == answer_a2
    # The parent is a 30 context snapshot and never mutates.
    assert meta.json()["completed_blocks"] == 30
    assert meta.json()["boundary"] == "context"
    assert b1.status_code == 200


@pytest.mark.asyncio
async def test_an_18_readout_is_refused(tmp_path: Path) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap30 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 30
        )
        response = await client.post(
            "/v2/decisions",
            json=_decision(snap30, readout={"completed_blocks": 18}),
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_unavailable"


@pytest.mark.asyncio
async def test_saved_child_supports_replace_question(tmp_path: Path) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap30 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 30
        )
        saved = await client.post(
            "/v2/decisions", json=_decision(snap30, save_result_snapshot=True)
        )
        child_id = saved.json()["snapshot_usage"]["child"]["status"]
        replaced = await client.post(
            "/v2/decisions",
            json=_decision(child_id)
            | {"snapshot": {"id": child_id, "relationship": "replace_question"}},
        )

    assert saved.status_code == 200, saved.text
    assert replaced.status_code == 200, replaced.text
    # replace_question branches from the context anchor, not the readout child.
    assert replaced.json()["snapshot_usage"]["effective_parent"] == snap30


@pytest.mark.asyncio
async def test_readout_create_has_no_context_parent_for_replace(
    tmp_path: Path,
) -> None:
    body = {
        "input": {
            "kind": "decision",
            "state": "s",
            "question": {"type": "noul", "instructions": "urgent?"},
        },
        "checkpoints": [30],
        "ttl_seconds": 600,
    }
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=body)).json()
        snap30 = created["snapshots"][0]
        assert snap30["boundary"] == "readout"
        response = await client.post(
            "/v2/decisions",
            json=_decision(snap30["id"])
            | {"snapshot": {"id": snap30["id"], "relationship": "replace_question"}},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "followup_unsupported"


@pytest.mark.asyncio
async def test_state_evaluation_returns_tagged_vectors(tmp_path: Path) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap30 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 30
        )
        response = await client.post(
            "/v2/state-evaluations",
            json={
                "snapshot": {"id": snap30, "relationship": "followup"},
                "prompt": "focus on the timeline",
                "readout": {
                    "completed_blocks": 30,
                    "export": ["last_residual", "last_normalized", "top_logits"],
                    "top_logits": 3,
                },
                "save_result_snapshot": True,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["usage"]["output_tokens"] == 0
    assert body["vectors"]["last_residual"]["representation"] == (
        "raw_residual_after_block_30"
    )
    assert body["vectors"]["last_normalized"]["representation"] == (
        "post_final_norm_head_input"
    )
    assert len(body["top_logits"]) == 3
    assert body["child"] is not None


@pytest.mark.asyncio
async def test_rider_generates_from_a_snapshot(tmp_path: Path) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap30 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 30
        )
        response = await client.post(
            "/v2/rider",
            json={
                "snapshot": {"id": snap30, "relationship": "followup"},
                "prompt": "draft a reply",
                "max_tokens": 4,
                "top_logits": 1,
            },
        )
        too_long = await client.post(
            "/v2/rider",
            json={
                "snapshot": {"id": snap30, "relationship": "followup"},
                "prompt": "draft a reply",
                "max_tokens": 5000,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_ids"] == [100, 101, 102, 103]
    assert body["stop_reason"] == "max_tokens"
    assert body["usage"]["output_tokens"] == 4
    assert body["reused_prefix_tokens"] > 0
    assert len(body["steps"]) == 4
    assert too_long.status_code == 422


@pytest.mark.asyncio
async def test_unknown_snapshot_and_delete_and_vectors(tmp_path: Path) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        missing = await client.get("/v2/snapshots/snap_" + "0" * 32)
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap30 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 30
        )
        vectors = await client.get(
            f"/v2/snapshots/{snap30}/vectors",
            params={"which": "h30", "row_begin": 0, "row_end": 4},
        )
        deleted = await client.delete(f"/v2/snapshots/{snap30}")
        gone = await client.get(f"/v2/snapshots/{snap30}")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "snapshot_not_found"
    assert vectors.status_code == 200
    assert vectors.json()["representation"] == "raw_residual_after_block_30"
    assert deleted.status_code == 200
    assert gone.status_code == 404


@pytest.mark.asyncio
async def test_models_reports_profile_and_no_early_head(tmp_path: Path) -> None:
    async with client_for(FakeSnapshotBackend(), tmp_path) as client:
        models = await client.get("/v2/models")

    profile = models.json()["models"][0]
    assert profile["profile"] == "split18-30-v1"
    assert profile["capabilities"]["early_head"] == "none"
    assert profile["capabilities"]["readout_blocks"] == [30]
    assert profile["capabilities"]["generation"] is False


@pytest.mark.asyncio
async def test_busy_is_rejected_without_partial_work(tmp_path: Path) -> None:
    backend = FakeSnapshotBackend()
    backend.gate = True
    async with client_for(backend, tmp_path) as client:
        created = (await client.post("/v2/snapshots", json=CONTEXT_BODY)).json()
        snap30 = next(
            r["id"] for r in created["snapshots"] if r["completed_blocks"] == 30
        )
        first = asyncio.create_task(
            client.post("/v2/decisions", json=_decision(snap30))
        )
        await backend.entered.wait()
        second = await client.post("/v2/decisions", json=_decision(snap30, "other"))
        backend.block.set()
        completed = await first

    assert second.status_code == 429
    assert second.json()["error"]["code"] == "busy"
    assert completed.status_code == 200


@pytest.mark.asyncio
async def test_v2_routes_are_absent_when_snapshots_are_disabled(
    tmp_path: Path,
) -> None:
    app = create_app(
        ApiConfig(),
        backend_factory=lambda _: MinimalV1Backend(),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        response = await client.post("/v2/snapshots", json=CONTEXT_BODY)

    assert response.status_code == 404


V1_REQUEST = {
    "state": "x",
    "questions": {"q": {"type": "noul", "instructions": "urgent?"}},
}


@pytest.mark.asyncio
async def test_snapshots_only_mode_disables_v1_but_serves_v2(tmp_path: Path) -> None:
    backend = FakeSnapshotBackend()
    async with client_for(backend, tmp_path, v1_enabled=False) as client:
        health = await client.get("/health")
        models = await client.get("/v1/models")
        decision = await client.post("/v1/decisions", json=V1_REQUEST)
        alias = await client.post("/v1/systemone", json=V1_REQUEST)
        created = await client.post("/v2/snapshots", json=CONTEXT_BODY)

    # /health is ok because the snapshot service is ready even with v1 off.
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert models.json() == {"models": []}
    assert decision.status_code == 529
    assert decision.json()["error"]["code"] == "unavailable"
    assert alias.status_code == 529
    assert created.status_code == 200


def test_v1_and_snapshots_both_off_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ApiConfig(v1_enabled=False)


def test_defaults_keep_v1_enabled() -> None:
    assert ApiConfig().v1_enabled is True
    assert ApiConfig().snapshots_enabled is False


@pytest.mark.asyncio
async def test_a_dead_worker_is_restarted_by_the_next_request(tmp_path: Path) -> None:
    # Disk-backed, so the restarted fake reloads real state as the worker does.
    backend = DiskBackedFakeSnapshotBackend()
    async with client_for(backend, tmp_path, v1_enabled=False) as client:
        memory = await client.post("/v2/snapshots", json=CONTEXT_BODY)
        disk = await client.post(
            "/v2/snapshots", json=CONTEXT_BODY | {"persistence": "disk"}
        )
        backend.crash()
        memory_30 = next(
            row["id"]
            for row in memory.json()["snapshots"]
            if row["completed_blocks"] == 30
        )
        disk_30 = next(
            row["id"]
            for row in disk.json()["snapshots"]
            if row["completed_blocks"] == 30
        )
        lost = await client.post("/v2/decisions", json=_decision(memory_30))
        kept = await client.post("/v2/decisions", json=_decision(disk_30))
        fresh = await client.post("/v2/snapshots", json=CONTEXT_BODY)
        health = await client.get("/health")

    assert backend.start_calls == 2
    # A memory snapshot died with its worker; a disk snapshot is reloaded.
    assert lost.status_code == 404
    assert lost.json()["error"]["code"] == "snapshot_not_found"
    assert kept.status_code == 200
    assert fresh.status_code == 200
    assert health.status_code == 200


@pytest.mark.asyncio
async def test_a_failed_restart_backs_off(tmp_path: Path) -> None:
    backend = FakeSnapshotBackend()
    async with client_for(backend, tmp_path, v1_enabled=False) as client:
        backend.crash()
        backend.fail_start = True
        first = await client.post("/v2/snapshots", json=CONTEXT_BODY)
        second = await client.post("/v2/snapshots", json=CONTEXT_BODY)

    assert first.status_code == 529
    assert second.status_code == 529
    # One attempt at startup, one recovery; the second request waits out the
    # backoff instead of respawning a worker that just failed.
    assert backend.start_calls == 2


@pytest.mark.asyncio
async def test_a_restarted_worker_must_match_the_first(tmp_path: Path) -> None:
    backend = FakeSnapshotBackend()
    async with client_for(backend, tmp_path, v1_enabled=False) as client:
        backend.crash()
        backend.hello = HELLO.model_copy(update={"runtime_sha256": "f" * 64})
        response = await client.post("/v2/snapshots", json=CONTEXT_BODY)
        health = await client.get("/health")

    assert response.status_code == 529
    assert health.status_code == 529


@pytest.mark.asyncio
async def test_health_starts_recovery_of_a_dead_worker(tmp_path: Path) -> None:
    backend = FakeSnapshotBackend()
    async with client_for(backend, tmp_path, v1_enabled=False) as client:
        backend.crash()
        down = await client.get("/health")
        for _ in range(50):
            if backend.ready:
                break
            await asyncio.sleep(0.01)
        up = await client.get("/health")

    assert down.status_code == 529
    assert backend.start_calls == 2
    assert up.status_code == 200
