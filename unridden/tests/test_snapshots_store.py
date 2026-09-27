"""Tests for the snapshot store: TTL, byte budget, leases, refcounts, disk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from unridden.api.snapshots.errors import (
    IntegrityError,
    SnapshotInUse,
    SnapshotNotFound,
    SnapshotStoreFull,
)
from unridden.api.snapshots.schema import WorkerFileEntry
from unridden.api.snapshots.store import (
    MANIFEST_NAME,
    SnapshotStore,
    new_snapshot_id,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeNativeIO:
    """A stand-in worker that writes real blob files and verifies them."""

    def __init__(self) -> None:
        self.dropped: list[str] = []
        self.loaded: list[str] = []
        self.bad_name = False

    async def save(
        self, snapshot_id: str, directory: Path
    ) -> dict[str, WorkerFileEntry]:
        name = "../escape" if self.bad_name else "lower_kv.bin"
        payload = f"{snapshot_id}-kv".encode()
        (directory / "lower_kv.bin").write_bytes(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        return {name: WorkerFileEntry(bytes=len(payload), sha256=checksum)}

    async def load(
        self, snapshot_id: str, directory: Path, files: dict[str, str]
    ) -> None:
        for name, checksum in files.items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != checksum:
                raise IntegrityError(f"blob {name} failed its checksum")
        self.loaded.append(snapshot_id)

    async def drop(self, snapshot_id: str) -> int:
        self.dropped.append(snapshot_id)
        return 42


def _store(
    tmp_path: Path, native: FakeNativeIO, *, host_bytes: int, clock: Clock
) -> SnapshotStore:
    return SnapshotStore(
        root=tmp_path / "snapshots",
        native=native,
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        profile="split18-30-v1",
        n_embd=2816,
        context_size=2048,
        host_bytes=host_bytes,
        clock=clock,
    )


async def _register(
    store: SnapshotStore,
    *,
    host_bytes: int,
    persistence: str = "memory",
    parent: str | None = None,
    completed_blocks: int = 30,
    ttl_seconds: int = 3600,
    promotion_of: str | None = None,
) -> str:
    snapshot_id = new_snapshot_id()
    await store.register(
        snapshot_id=snapshot_id,
        owner="local",
        completed_blocks=completed_blocks,
        boundary="context",
        parent=parent,
        context_parent=None,
        messages=[{"role": "user", "content": "x"}],
        answer_prefix="",
        tokens=10,
        host_bytes=host_bytes,
        persistence=persistence,  # type: ignore[arg-type]
        prompt_sha256="c" * 64,
        ttl_seconds=ttl_seconds,
        promotion_of=promotion_of,
    )
    return snapshot_id


@pytest.mark.asyncio
async def test_register_metadata_and_capabilities(tmp_path: Path) -> None:
    native = FakeNativeIO()
    clock = Clock()
    store = _store(tmp_path, native, host_bytes=1000, clock=clock)

    parent = await _register(store, host_bytes=10, completed_blocks=18)
    meta = store.metadata(parent)

    assert meta.completed_blocks == 18
    assert meta.capabilities == ["continue", "promote", "inspect"]
    assert meta.resident is True
    assert store.metadata(parent).parent is None


@pytest.mark.asyncio
async def test_ttl_expiry_forgets_an_idle_snapshot(tmp_path: Path) -> None:
    native = FakeNativeIO()
    clock = Clock()
    store = _store(tmp_path, native, host_bytes=1000, clock=clock)

    snapshot_id = await _register(store, host_bytes=10, ttl_seconds=60)
    clock.now += 61
    await store.expire()
    with pytest.raises(SnapshotNotFound):
        store.get(snapshot_id)
    assert snapshot_id in native.dropped
    assert store.resident_bytes == 0


@pytest.mark.asyncio
async def test_ttl_expiry_removes_a_durable_snapshot(tmp_path: Path) -> None:
    native = FakeNativeIO()
    clock = Clock()
    store = _store(tmp_path, native, host_bytes=100, clock=clock)
    snapshot_id = await _register(
        store, host_bytes=60, persistence="disk", ttl_seconds=1
    )
    directory = tmp_path / "snapshots" / snapshot_id
    assert directory.is_dir()

    clock.now += 2
    await store.expire()

    assert not directory.exists()
    assert snapshot_id in native.dropped
    with pytest.raises(SnapshotNotFound):
        store.get(snapshot_id)


@pytest.mark.asyncio
async def test_memory_budget_evicts_the_least_recently_used(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=150, clock=Clock())

    first = await _register(store, host_bytes=60)
    second = await _register(store, host_bytes=60)
    store.get(second)  # touch second so first is the LRU
    third = await _register(store, host_bytes=60)

    assert first in native.dropped
    with pytest.raises(SnapshotNotFound):
        store.get(first)
    assert store.get(second) is not None
    assert store.get(third) is not None


@pytest.mark.asyncio
async def test_disk_budget_persists_instead_of_dropping(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())

    first = await _register(store, host_bytes=60, persistence="disk")
    second = await _register(store, host_bytes=60, persistence="disk")

    # `first` was evicted to disk, not forgotten, and its manifest is durable.
    record = store.get(first)
    assert record.resident is False
    assert record.disk_dir is not None and (record.disk_dir / MANIFEST_NAME).is_file()
    assert first in native.dropped
    assert store.get(second).resident is True

    restored = await store.restore(first)
    assert restored.resident is True
    assert first in native.loaded


@pytest.mark.asyncio
async def test_disk_snapshot_is_available_after_store_restarts(tmp_path: Path) -> None:
    native = FakeNativeIO()
    clock = Clock()
    store = _store(tmp_path, native, host_bytes=100, clock=clock)
    snapshot_id = await _register(store, host_bytes=60, persistence="disk")
    assert (tmp_path / "snapshots" / snapshot_id / MANIFEST_NAME).is_file()

    replacement = _store(tmp_path, native, host_bytes=100, clock=clock)
    record = replacement.get(snapshot_id)
    assert record.resident is False
    assert record.messages == [{"role": "user", "content": "x"}]
    assert (await replacement.restore(snapshot_id)).resident is True


@pytest.mark.asyncio
async def test_only_marked_same_prefix_child_is_rebuilt_as_promotion(
    tmp_path: Path,
) -> None:
    native = FakeNativeIO()
    clock = Clock()
    store = _store(tmp_path, native, host_bytes=1000, clock=clock)
    parent = await _register(
        store, host_bytes=50, persistence="disk", completed_blocks=18
    )
    paired_child = await _register(
        store, host_bytes=20, persistence="disk", parent=parent
    )

    replacement = _store(tmp_path, native, host_bytes=1000, clock=clock)
    assert replacement.get(paired_child).parent == parent
    assert replacement.memoized_promotion(parent) is None

    promoted = await _register(
        replacement,
        host_bytes=20,
        persistence="disk",
        parent=parent,
        promotion_of=parent,
        ttl_seconds=1,
    )
    restarted = _store(tmp_path, native, host_bytes=1000, clock=clock)
    assert restarted.memoized_promotion(parent) == promoted

    clock.now += 2
    assert restarted.memoized_promotion(parent) is None
    await restarted.expire()
    assert not (tmp_path / "snapshots" / promoted).exists()
    assert restarted.get(paired_child).parent == parent


@pytest.mark.asyncio
async def test_durable_state_and_manifest_are_private(tmp_path: Path) -> None:
    store = _store(tmp_path, FakeNativeIO(), host_bytes=100, clock=Clock())
    snapshot_id = await _register(store, host_bytes=60, persistence="disk")
    directory = tmp_path / "snapshots" / snapshot_id
    manifest = directory / MANIFEST_NAME

    assert directory.stat().st_mode & 0o077 == 0
    assert manifest.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_restored_child_counts_its_new_lower_cache(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())
    parent = await _register(
        store, host_bytes=50, persistence="disk", completed_blocks=18
    )
    child = await _register(store, host_bytes=20, persistence="disk", parent=parent)
    await _register(store, host_bytes=40)
    assert not store.get(child).resident

    with store.lease(child):
        await store.restore(child)

    assert store.resident_bytes <= 100
    assert store.get(parent).resident
    assert store.get(child).host_bytes > 20


@pytest.mark.asyncio
async def test_disk_parent_can_spill_after_its_shared_child_spills(
    tmp_path: Path,
) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())
    parent = await _register(
        store, host_bytes=50, persistence="disk", completed_blocks=18
    )
    child = await _register(store, host_bytes=20, persistence="disk", parent=parent)
    third = await _register(store, host_bytes=40)
    assert not store.get(child).resident

    with store.lease(third):
        await _register(store, host_bytes=50)

    assert not store.get(parent).resident


@pytest.mark.asyncio
async def test_a_lease_pins_a_snapshot_against_eviction(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=150, clock=Clock())

    pinned = await _register(store, host_bytes=60)
    evictable = await _register(store, host_bytes=60)
    with store.lease(pinned):
        await _register(store, host_bytes=60)

    assert evictable in native.dropped
    assert store.get(pinned) is not None


@pytest.mark.asyncio
async def test_store_full_when_nothing_can_be_evicted(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())

    pinned = await _register(store, host_bytes=60)
    with store.lease(pinned):  # noqa: SIM117 - the lease must wrap the register
        with pytest.raises(SnapshotStoreFull):
            await _register(store, host_bytes=60)


@pytest.mark.asyncio
async def test_refcounts_block_delete_and_protect_memory_parents(
    tmp_path: Path,
) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=150, clock=Clock())

    parent = await _register(store, host_bytes=60, completed_blocks=18)
    child = await _register(store, host_bytes=60, parent=parent)

    with pytest.raises(SnapshotInUse):
        await store.delete(parent)

    # Registering a third snapshot forces an eviction: the referenced memory
    # parent is skipped and the unreferenced child is dropped instead.
    await _register(store, host_bytes=60)
    assert parent not in native.dropped
    assert child in native.dropped
    assert store.get(parent) is not None

    await store.delete(parent)
    with pytest.raises(SnapshotNotFound):
        store.get(parent)


@pytest.mark.asyncio
async def test_restore_rejects_a_tampered_manifest(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())

    first = await _register(store, host_bytes=60, persistence="disk")
    await _register(store, host_bytes=60, persistence="disk")  # evicts first

    record = store.get(first)
    assert record.disk_dir is not None
    manifest_path = record.disk_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["model_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(IntegrityError):
        await store.restore(first)


@pytest.mark.asyncio
async def test_restore_rejects_a_corrupted_blob(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())

    first = await _register(store, host_bytes=60, persistence="disk")
    await _register(store, host_bytes=60, persistence="disk")

    record = store.get(first)
    assert record.disk_dir is not None
    (record.disk_dir / "lower_kv.bin").write_bytes(b"tampered")

    with pytest.raises(IntegrityError):
        await store.restore(first)


@pytest.mark.asyncio
async def test_publish_refuses_an_unsafe_native_file_name(tmp_path: Path) -> None:
    native = FakeNativeIO()
    native.bad_name = True
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())

    with pytest.raises(IntegrityError):
        # Disk persistence publishes on registration and never leaves a
        # half-published directory when the worker supplies an unsafe name.
        await _register(store, host_bytes=60, persistence="disk")
    assert list((tmp_path / "snapshots").iterdir()) == []


@pytest.mark.asyncio
async def test_register_refuses_an_unsafe_snapshot_id(tmp_path: Path) -> None:
    native = FakeNativeIO()
    store = _store(tmp_path, native, host_bytes=100, clock=Clock())

    with pytest.raises(IntegrityError):
        await store.register(
            snapshot_id="../escape",
            owner="local",
            completed_blocks=30,
            boundary="context",
            parent=None,
            context_parent=None,
            messages=[],
            answer_prefix="",
            tokens=10,
            host_bytes=10,
            persistence="memory",
            prompt_sha256="c" * 64,
            ttl_seconds=60,
        )
