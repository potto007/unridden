"""In-process snapshot registry: leases, byte budget, TTL and disk publish.

The store owns host-side bookkeeping for snapshots the worker holds. It never
serializes tensors itself; a `NativeSnapshotIO` writes and reads the blobs and
the store owns the manifest, the temp-dir-then-rename publish, reopen
validation, TTL expiry, an explicit host byte budget with LRU eviction, leases
that pin a snapshot and its ancestors, and reference counting that keeps a
parent alive while a child needs it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from unridden.api.snapshots.compiler import Message
from unridden.api.snapshots.errors import (
    IntegrityError,
    SnapshotInUse,
    SnapshotNotFound,
    SnapshotStoreFull,
)
from unridden.api.snapshots.schema import (
    Boundary,
    Checkpoint,
    Persistence,
    SnapshotMetadata,
    WorkerFileEntry,
)

MANIFEST_SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"
# A snapshot id is opaque and must be safe to use as a single path component.
_ID_PATTERN = re.compile(r"^snap_[0-9a-f]{32}$")
# The lower/upper layer map of split18-30-v1 (design "runtime design"), the
# default. Recorded so a restore refuses a blob captured under another split.
LAYER_MAP = {"lower": [0, 18], "upper": [18, 30]}


def new_snapshot_id() -> str:
    return f"snap_{uuid.uuid4().hex}"


def _valid_id(snapshot_id: str) -> bool:
    return bool(_ID_PATTERN.fullmatch(snapshot_id))


class NativeSnapshotIO(Protocol):
    """What the store needs from the native worker to move blobs on and off host.

    `save` writes the snapshot's blobs into `directory` and returns each file's
    size and sha256. `load` re-registers a snapshot from a directory after the
    store has validated the manifest. `drop` frees the worker's host copy.
    """

    async def save(
        self, snapshot_id: str, directory: Path
    ) -> dict[str, WorkerFileEntry]: ...

    async def load(
        self, snapshot_id: str, directory: Path, files: dict[str, str]
    ) -> None: ...

    async def drop(self, snapshot_id: str) -> int: ...


@dataclass
class SnapshotRecord:
    id: str
    owner: str
    completed_blocks: Checkpoint
    boundary: Boundary
    parent: str | None
    context_parent: str | None
    messages: list[Message]
    answer_prefix: str
    tokens: int
    host_bytes: int
    persistence: Persistence
    prompt_sha256: str
    created_at: float
    expires_at: float
    resident: bool = True
    disk_dir: Path | None = None
    durable_files: dict[str, WorkerFileEntry] | None = None
    # The derived 30 snapshot promoted from this 18 snapshot, memoized so a
    # second full-depth question does not repeat promotion (design "S18").
    promotion_child: str | None = None
    promotion_of: str | None = None
    leases: int = 0
    refcount: int = 0
    last_used: int = 0
    persisted_host_bytes: int = 0

    def capabilities(self, split_block: int | None = 18) -> list[str]:
        """What a caller can do with it; only the split block promotes."""
        caps = ["continue", "inspect"]
        if split_block is not None and self.completed_blocks == split_block:
            caps.insert(1, "promote")
        return caps


class SnapshotStore:
    def __init__(
        self,
        *,
        root: Path,
        native: NativeSnapshotIO,
        model_sha256: str,
        runtime_sha256: str,
        profile: str,
        n_embd: int,
        context_size: int,
        host_bytes: int,
        layer_map: dict[str, list[int]] | None = None,
        split_block: int | None = 18,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if host_bytes <= 0:
            raise ValueError("host byte budget must be positive")
        # Absolute: the worker saves into and loads from these paths itself and
        # refuses a relative one (the units pass one relative to their cwd).
        self._root = root.resolve()
        self._native = native
        self._model_sha256 = model_sha256
        self._runtime_sha256 = runtime_sha256
        self._profile = profile
        self._n_embd = n_embd
        self._context_size = context_size
        self._host_bytes = host_bytes
        self._layer_map = LAYER_MAP if layer_map is None else layer_map
        # None for a profile without an early checkpoint (full-v1).
        self._split_block = split_block
        self._clock = clock
        self._records: dict[str, SnapshotRecord] = {}
        self._resident_bytes = 0
        self._tick = 0
        self._root.mkdir(parents=True, exist_ok=True)
        self._load_disk_index()

    # -- lifecycle ---------------------------------------------------------

    def _now(self) -> float:
        return self._clock()

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    async def expire(self) -> None:
        """Reclaim expired leaves, then parents whose last child was reclaimed."""
        now = self._now()
        while True:
            expired = next(
                (
                    record
                    for record in self._records.values()
                    if record.expires_at <= now
                    and record.leases == 0
                    and record.refcount == 0
                ),
                None,
            )
            if expired is None:
                return
            record = expired
            if record.resident:
                with contextlib.suppress(SnapshotNotFound):
                    await self._native.drop(record.id)
            if record.disk_dir is not None:
                _remove_tree(record.disk_dir)
            self._forget(record)

    def _forget(self, record: SnapshotRecord) -> None:
        if record.resident:
            self._resident_bytes -= record.host_bytes
        if record.parent is not None:
            self._release_ref(record.parent)
        if record.context_parent is not None and record.context_parent != record.parent:
            self._release_ref(record.context_parent)
        self._records.pop(record.id, None)

    def _release_ref(self, snapshot_id: str) -> None:
        parent = self._records.get(snapshot_id)
        if parent is not None and parent.refcount > 0:
            parent.refcount -= 1

    # -- registration ------------------------------------------------------

    async def register(
        self,
        *,
        snapshot_id: str,
        owner: str,
        completed_blocks: Checkpoint,
        boundary: Boundary,
        parent: str | None,
        context_parent: str | None,
        messages: list[Message],
        answer_prefix: str,
        tokens: int,
        host_bytes: int,
        persistence: Persistence,
        prompt_sha256: str,
        ttl_seconds: int,
        promotion_of: str | None = None,
    ) -> SnapshotRecord:
        await self.expire()
        if not _valid_id(snapshot_id):
            raise IntegrityError("snapshot id is not a safe opaque reference")
        if snapshot_id in self._records:
            raise IntegrityError("snapshot id already registered")
        for reference in (parent, context_parent):
            if reference is not None and reference not in self._records:
                raise SnapshotNotFound("referenced parent snapshot is unknown")
        if promotion_of is not None and promotion_of != parent:
            raise IntegrityError("promotion must reference its direct parent")
        # A new child has no refcount yet. Pin its parents while the budget
        # manager chooses victims, then publish the new reference.
        with self.lease(
            *(reference for reference in (parent, context_parent) if reference)
        ):
            await self._enforce_budget(host_bytes)
            now = self._now()
            record = SnapshotRecord(
                id=snapshot_id,
                owner=owner,
                completed_blocks=completed_blocks,
                boundary=boundary,
                parent=parent,
                context_parent=context_parent,
                messages=messages,
                answer_prefix=answer_prefix,
                tokens=tokens,
                host_bytes=host_bytes,
                persistence=persistence,
                prompt_sha256=prompt_sha256,
                created_at=now,
                expires_at=now + ttl_seconds,
                last_used=self._next_tick(),
                persisted_host_bytes=host_bytes,
                promotion_of=promotion_of,
            )
            if promotion_of is not None:
                self._validate_promotion(record, self._records[promotion_of])
            self._records[snapshot_id] = record
            self._resident_bytes += host_bytes
            if parent is not None:
                self._records[parent].refcount += 1
            if context_parent is not None and context_parent != parent:
                self._records[context_parent].refcount += 1
            try:
                if persistence == "disk":
                    await self._persist(record)
            except BaseException:
                self._forget(record)
                raise
            return record

    def memoize_promotion(self, parent_id: str, child_id: str) -> None:
        parent = self.get(parent_id)
        self._validate_promotion(self.get(child_id), parent)
        parent.promotion_child = child_id

    def memoized_promotion(self, parent_id: str) -> str | None:
        parent = self.get(parent_id)
        child_id = parent.promotion_child
        child = self._records.get(child_id) if child_id is not None else None
        if child is None or child.expires_at <= self._now():
            parent.promotion_child = None
            return None
        return child_id

    @staticmethod
    def _validate_promotion(child: SnapshotRecord, parent: SnapshotRecord) -> None:
        if (
            child.promotion_of != parent.id
            or child.parent != parent.id
            or parent.completed_blocks != 18
            or child.completed_blocks != 30
            or child.owner != parent.owner
            or child.boundary != parent.boundary
            or child.messages != parent.messages
            or child.answer_prefix != parent.answer_prefix
            or child.tokens != parent.tokens
            or child.prompt_sha256 != parent.prompt_sha256
            or child.context_parent
            != (parent.context_parent if parent.boundary == "readout" else None)
        ):
            raise IntegrityError("promoted snapshot does not match its 18-block parent")

    def capabilities(self, record: SnapshotRecord) -> list[str]:
        return record.capabilities(self._split_block)

    # -- access ------------------------------------------------------------

    def get(self, snapshot_id: str) -> SnapshotRecord:
        record = self._records.get(snapshot_id)
        if record is None or (
            record.expires_at <= self._now() and record.refcount == 0
        ):
            raise SnapshotNotFound(f"snapshot {snapshot_id!r} is unknown or expired")
        record.last_used = self._next_tick()
        return record

    def metadata(self, snapshot_id: str) -> SnapshotMetadata:
        record = self.get(snapshot_id)
        return SnapshotMetadata(
            id=record.id,
            completed_blocks=record.completed_blocks,
            boundary=record.boundary,
            parent=record.parent,
            context_parent=record.context_parent,
            persistence=record.persistence,
            resident=record.resident,
            tokens=record.tokens,
            capabilities=self.capabilities(record),  # type: ignore[arg-type]
            created_at=record.created_at,
            expires_at=record.expires_at,
        )

    def _ancestors(self, snapshot_id: str) -> set[str]:
        seen: set[str] = set()
        frontier = [snapshot_id]
        while frontier:
            current = frontier.pop()
            if current in seen or current not in self._records:
                continue
            seen.add(current)
            record = self._records[current]
            for reference in (record.parent, record.context_parent):
                if reference is not None:
                    frontier.append(reference)
        return seen

    @contextlib.contextmanager
    def lease(self, *snapshot_ids: str) -> Iterator[None]:
        """Pin the given snapshots and every ancestor against eviction."""
        pinned: set[str] = set()
        for snapshot_id in snapshot_ids:
            self.get(snapshot_id)
            pinned |= self._ancestors(snapshot_id)
        for snapshot_id in pinned:
            self._records[snapshot_id].leases += 1
        try:
            yield
        finally:
            for snapshot_id in pinned:
                record = self._records.get(snapshot_id)
                if record is not None and record.leases > 0:
                    record.leases -= 1

    # -- deletion ----------------------------------------------------------

    async def delete(self, snapshot_id: str) -> int:
        record = self.get(snapshot_id)
        if record.leases > 0:
            raise SnapshotInUse("snapshot is in use")
        if record.refcount > 0:
            raise SnapshotInUse("snapshot still has referencing children")
        freed = 0
        with contextlib.suppress(SnapshotNotFound):
            freed = await self._native.drop(snapshot_id)
        if record.disk_dir is not None:
            _remove_tree(record.disk_dir)
        self._forget(record)
        return freed

    # -- byte budget and eviction -----------------------------------------

    async def _enforce_budget(self, incoming_bytes: int) -> None:
        if incoming_bytes > self._host_bytes:
            raise SnapshotStoreFull("snapshot exceeds the whole host budget")
        await self.expire()
        while self._resident_bytes + incoming_bytes > self._host_bytes:
            if not await self._evict_one():
                raise SnapshotStoreFull("host byte budget is exhausted")

    async def _evict_one(self) -> bool:
        candidates = [
            record
            for record in self._records.values()
            if record.resident
            and record.leases == 0
            and (
                record.refcount == 0
                or (
                    record.persistence == "disk"
                    and not self._has_resident_shared_child(record)
                )
            )
        ]
        candidates.sort(key=lambda record: record.last_used)
        for record in candidates:
            if record.persistence == "disk":
                await self._persist(record)
                with contextlib.suppress(SnapshotNotFound):
                    await self._native.drop(record.id)
                self._resident_bytes -= record.host_bytes
                record.resident = False
                return True
            # A memory-only snapshot is simply dropped, but never while a child
            # still references its cache.
            if record.refcount == 0:
                await self._native.drop(record.id)
                self._forget(record)
                return True
        return False

    def _has_resident_shared_child(self, parent: SnapshotRecord) -> bool:
        return parent.completed_blocks == self._split_block and any(
            child.parent == parent.id
            and child.completed_blocks == 30
            and child.tokens == parent.tokens
            and child.resident
            for child in self._records.values()
        )

    # -- disk publish and restore -----------------------------------------

    def _disk_dir(self, snapshot_id: str) -> Path:
        if not _valid_id(snapshot_id):
            raise IntegrityError("snapshot id is not a safe opaque reference")
        return self._root / snapshot_id

    def _manifest(
        self, record: SnapshotRecord, files: dict[str, WorkerFileEntry]
    ) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "snapshot_id": record.id,
            "owner": record.owner,
            "created_at": record.created_at,
            "messages": record.messages,
            "answer_prefix": record.answer_prefix,
            "host_bytes": record.persisted_host_bytes,
            "persistence": record.persistence,
            "model_sha256": self._model_sha256,
            "runtime_sha256": self._runtime_sha256,
            "profile": self._profile,
            "completed_blocks": record.completed_blocks,
            "boundary": record.boundary,
            "layer_map": self._layer_map,
            "dtype": "f32",
            "n_embd": self._n_embd,
            "tokens": record.tokens,
            "prompt_version": "unridden-gemma-context-v1",
            "prompt_sha256": record.prompt_sha256,
            "parent": record.parent,
            "context_parent": record.context_parent,
            "promotion_of": record.promotion_of,
            "expires_at": record.expires_at,
            "files": {
                name: {"bytes": entry.bytes, "sha256": entry.sha256}
                for name, entry in files.items()
            },
        }

    async def _persist(self, record: SnapshotRecord) -> None:
        if record.durable_files is not None and record.disk_dir is not None:
            return
        final = self._disk_dir(record.id)
        os.makedirs(self._root, exist_ok=True)
        tmp = Path(f"{final}.tmp-{uuid.uuid4().hex}")
        tmp.mkdir(mode=0o700)
        published = False
        try:
            files = await self._native.save(record.id, tmp)
            _check_file_names(files)
            manifest = self._manifest(record, files)
            manifest_path = tmp / MANIFEST_NAME
            descriptor = os.open(
                manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, sort_keys=True) + "\n")
            _fsync_tree(tmp)
            os.replace(tmp, final)
            published = True
            _fsync_dir(self._root)
            reopened = self._read_manifest(final)
            self._validate_manifest(reopened, record)
        except BaseException:
            _remove_tree(tmp)
            if published:
                _remove_tree(final)
            raise
        record.disk_dir = final
        record.durable_files = files

    def _read_manifest(self, directory: Path) -> dict[str, Any]:
        path = directory / MANIFEST_NAME
        try:
            raw: Any = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise IntegrityError("snapshot manifest is unreadable") from error
        if not isinstance(raw, dict):
            raise IntegrityError("snapshot manifest is not an object")
        return raw

    def _validate_manifest(
        self, manifest: dict[str, Any], record: SnapshotRecord
    ) -> None:
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise IntegrityError("snapshot manifest schema is unsupported")
        for key, expected in (
            ("snapshot_id", record.id),
            ("owner", record.owner),
            ("boundary", record.boundary),
            ("parent", record.parent),
            ("context_parent", record.context_parent),
            ("promotion_of", record.promotion_of),
            ("messages", record.messages),
            ("answer_prefix", record.answer_prefix),
            ("host_bytes", record.persisted_host_bytes),
            ("persistence", record.persistence),
            ("created_at", record.created_at),
            ("expires_at", record.expires_at),
            ("prompt_sha256", record.prompt_sha256),
            ("tokens", record.tokens),
        ):
            if manifest.get(key) != expected:
                raise IntegrityError(f"snapshot {key} differs from its record")
        if manifest.get("model_sha256") != self._model_sha256:
            raise IntegrityError("snapshot was captured under another model")
        if manifest.get("runtime_sha256") != self._runtime_sha256:
            raise IntegrityError("snapshot was captured under another runtime")
        if manifest.get("profile") != self._profile:
            raise IntegrityError("snapshot was captured under another profile")
        if manifest.get("n_embd") != self._n_embd:
            raise IntegrityError("snapshot hidden width differs from the profile")
        if manifest.get("layer_map") != self._layer_map:
            raise IntegrityError("snapshot layer map differs from the profile")
        if manifest.get("completed_blocks") != record.completed_blocks:
            raise IntegrityError("snapshot depth differs from its record")
        tokens = manifest.get("tokens")
        if not isinstance(tokens, int) or not 1 <= tokens <= self._context_size:
            raise IntegrityError("snapshot token count is out of bounds")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise IntegrityError("snapshot manifest has no files")
        for name, entry in files.items():
            if not isinstance(name, str) or not _safe_name(name):
                raise IntegrityError("snapshot file name is unsafe")
            if not isinstance(entry, dict):
                raise IntegrityError("snapshot file entry is malformed")
            size = entry.get("bytes")
            checksum = entry.get("sha256")
            if not isinstance(size, int) or size < 0:
                raise IntegrityError("snapshot file size is out of bounds")
            if not isinstance(checksum, str) or not re.fullmatch(
                r"[0-9a-f]{64}", checksum
            ):
                raise IntegrityError("snapshot file checksum is malformed")

    def _load_disk_index(self) -> None:
        """Rebuild the process-local index from published disk snapshots."""
        for directory in sorted(self._root.iterdir()):
            if not directory.is_dir() or not _valid_id(directory.name):
                continue
            manifest = self._read_manifest(directory)
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                raise IntegrityError("snapshot manifest schema is unsupported")
            messages = manifest.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or any(
                    not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or message["role"] not in {"user", "assistant"}
                    or not isinstance(message["content"], str)
                    for message in messages
                )
            ):
                raise IntegrityError("snapshot messages are malformed")
            owner = manifest.get("owner")
            answer_prefix = manifest.get("answer_prefix")
            host_bytes = manifest.get("host_bytes")
            created_at = manifest.get("created_at")
            expires_at = manifest.get("expires_at")
            promotion_of = manifest.get("promotion_of")
            if (
                not isinstance(owner, str)
                or not owner
                or not isinstance(answer_prefix, str)
                or not isinstance(host_bytes, int)
                or host_bytes <= 0
                or not isinstance(created_at, (int, float))
                or not isinstance(expires_at, (int, float))
                or (
                    promotion_of is not None
                    and (
                        not isinstance(promotion_of, str) or not _valid_id(promotion_of)
                    )
                )
            ):
                raise IntegrityError("snapshot metadata is malformed")
            try:
                files = {
                    name: WorkerFileEntry.model_validate(entry)
                    for name, entry in manifest["files"].items()
                }
                record = SnapshotRecord(
                    id=directory.name,
                    owner=owner,
                    completed_blocks=manifest["completed_blocks"],
                    boundary=manifest["boundary"],
                    parent=manifest["parent"],
                    context_parent=manifest["context_parent"],
                    messages=messages,
                    answer_prefix=answer_prefix,
                    tokens=manifest["tokens"],
                    host_bytes=host_bytes,
                    persistence="disk",
                    prompt_sha256=manifest["prompt_sha256"],
                    created_at=float(created_at),
                    expires_at=float(expires_at),
                    resident=False,
                    disk_dir=directory,
                    durable_files=files,
                    last_used=self._next_tick(),
                    persisted_host_bytes=host_bytes,
                    promotion_of=promotion_of,
                )
            except (KeyError, AttributeError, TypeError, ValueError) as error:
                raise IntegrityError("snapshot manifest is malformed") from error
            self._validate_manifest(manifest, record)
            self._records[record.id] = record
        for record in self._records.values():
            for reference in {
                value
                for value in (record.parent, record.context_parent)
                if value is not None
            }:
                parent = self._records.get(reference)
                if parent is None:
                    raise IntegrityError("snapshot parent is missing from disk")
                parent.refcount += 1
            if record.promotion_of is not None:
                parent = self._records.get(record.promotion_of)
                if parent is None:
                    raise IntegrityError("promoted parent is missing from disk")
                self._validate_promotion(record, parent)
                previous = (
                    self._records[parent.promotion_child]
                    if parent.promotion_child is not None
                    else None
                )
                if record.expires_at > self._now() and (
                    previous is None or previous.created_at < record.created_at
                ):
                    parent.promotion_child = record.id

    async def restore(self, snapshot_id: str) -> SnapshotRecord:
        """Bring a snapshot back to resident, restoring from disk if needed."""
        record = self.get(snapshot_id)
        if record.resident:
            return record
        if record.disk_dir is None or record.durable_files is None:
            raise SnapshotNotFound("snapshot has no resident or durable copy")
        manifest = self._read_manifest(record.disk_dir)
        self._validate_manifest(manifest, record)
        # A paired S30 originally shares its parent's lower K/V and charges it
        # there. Native load reconstructs a separate blob. Reserve every saved
        # tensor byte before loading, including that newly owned lower K/V.
        loaded_bytes = max(
            record.host_bytes,
            sum(
                entry.bytes
                for name, entry in record.durable_files.items()
                if name != "native.json"
            ),
        )
        await self._enforce_budget(loaded_bytes)
        checksums = {name: entry.sha256 for name, entry in record.durable_files.items()}
        await self._native.load(snapshot_id, record.disk_dir, checksums)
        record.resident = True
        record.host_bytes = loaded_bytes
        self._resident_bytes += loaded_bytes
        record.last_used = self._next_tick()
        return record

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes


def _safe_name(name: str) -> bool:
    return (
        bool(name)
        and "/" not in name
        and "\\" not in name
        and name
        not in {
            ".",
            "..",
        }
    )


def _check_file_names(files: dict[str, WorkerFileEntry]) -> None:
    for name in files:
        if not _safe_name(name):
            raise IntegrityError("native save returned an unsafe file name")


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_file():
            fd = os.open(child, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    _fsync_dir(directory)


def _remove_tree(directory: Path) -> None:
    if not directory.exists():
        return
    for child in sorted(directory.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        else:
            child.rmdir()
    directory.rmdir()
