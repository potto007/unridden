"""Bounded async JSONL transport for one owned snapshot worker.

Mirrors `unridden/api/native_backend.py`: one resident child, a verified
install, correlation ids, a 4 MiB line bound, an stderr tail, and reaping the
child on any ambiguous transport failure. Typed worker errors are in-protocol
and leave the child running (the protocol table defines the state after each);
only malformed lines, timeouts, a closed stream or a failed validation reap it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from pydantic import BaseModel, ValidationError

from unridden.api.native.build import (
    bundle_digest,
    digest,
    is_tested_revision,
)
from unridden.api.native.bundle import read_manifest, resolve_manifest_paths
from unridden.api.snapshots.errors import (
    CapabilityUnavailable,
    FollowupUnsupported,
    IntegrityError,
    PrefixMismatch,
    SnapshotError,
    SnapshotExecutionError,
    SnapshotExists,
    SnapshotNotFound,
    SnapshotProtocolError,
    SnapshotRequestError,
    SnapshotUnavailableError,
)
from unridden.api.snapshots.schema import (
    SNAPSHOT_PROFILE,
    Checkpoint,
    ExportKind,
    ProfileName,
    WorkerCreated,
    WorkerDropped,
    WorkerErrorMessage,
    WorkerFileEntry,
    WorkerHello,
    WorkerInspect,
    WorkerLoaded,
    WorkerPromoted,
    WorkerResult,
    WorkerRide,
    WorkerSaved,
    WorkerState,
    WorkerVectors,
)

LOGGER = logging.getLogger("unridden.api.snapshots.native")
WORKER_PROTOCOL_BYTES = 4 * 1024 * 1024
WORKER_BINARY_NAME = "unridden-snapshot-worker"

# In-protocol worker error codes and the typed exception each raises. The
# worker stays up for all of them; the caller, not the transport, is at fault.
_ERROR_CODES: dict[str, type[SnapshotError]] = {
    "snapshot_not_found": SnapshotNotFound,
    "snapshot_exists": SnapshotExists,
    "snapshot_prefix_mismatch": PrefixMismatch,
    "followup_unsupported": FollowupUnsupported,
    "capability_unavailable": CapabilityUnavailable,
    "integrity_error": IntegrityError,
    "execution_error": SnapshotExecutionError,
}
# The execution mode a bundle manifest records for each profile it serves.
_PROFILE_MODES = {"split18-30-v1": "split18-30", "full-v1": "full"}


class VerifiedInstall(NamedTuple):
    model_sha256: str
    runtime_sha256: str
    runtime_dir: Path
    llama_revision: str


class SnapshotBackend(Protocol):
    """One resident snapshot model backend.

    Contract: any ambiguous transport failure (timeout, malformed line, closed
    stream, failed validation) stops the child and sets `ready` to False before
    the exception leaves. A typed worker error leaves the child running.
    """

    profile: WorkerHello | None
    ready: bool

    async def start(self) -> WorkerHello: ...

    async def close(self) -> None: ...

    async def create(
        self,
        *,
        messages: list[dict[str, str]],
        answer_prefix: str,
        freeze: dict[str, Any],
        checkpoints: dict[str, str],
        labels: list[str] | None,
        top_logits: int,
        timeout: float,
    ) -> WorkerCreated: ...

    async def promote(
        self, *, snapshot_id: str, new_id: str, timeout: float
    ) -> WorkerPromoted: ...

    async def evaluate(
        self,
        *,
        snapshot_id: str,
        readout_blocks: Checkpoint,
        questions: list[dict[str, Any]],
        timeout: float,
    ) -> WorkerResult: ...

    async def state_eval(
        self,
        *,
        snapshot_id: str,
        messages: list[dict[str, str]],
        answer_prefix: str,
        export: list[ExportKind],
        top_logits: int,
        save_as: str | None,
        timeout: float,
    ) -> WorkerState: ...

    async def ride(
        self,
        *,
        snapshot_id: str,
        messages: list[dict[str, str]],
        answer_prefix: str,
        max_tokens: int,
        top_logits: int,
        timeout: float,
    ) -> WorkerRide: ...

    async def inspect(self, *, snapshot_id: str, timeout: float) -> WorkerInspect: ...

    async def vectors(
        self,
        *,
        snapshot_id: str,
        which: str,
        row_begin: int,
        row_end: int,
        timeout: float,
    ) -> WorkerVectors: ...

    async def drop(self, snapshot_id: str) -> int: ...

    async def save(
        self, snapshot_id: str, directory: Path
    ) -> dict[str, WorkerFileEntry]: ...

    async def load(
        self, snapshot_id: str, directory: Path, files: dict[str, str]
    ) -> None: ...


class SnapshotNativeBackend:
    """One resident snapshot worker child; reaped on any ambiguous failure."""

    def __init__(
        self,
        *,
        worker_path: Path,
        model_path: Path,
        manifest_path: Path,
        gpu: bool = False,
        context_size: int = 2048,
        batch_size: int = 256,
        ubatch_size: int = 256,
        threads: int = 8,
        reference: bool = False,
        snapshot_profile: ProfileName = SNAPSHOT_PROFILE,
        max_response_bytes: int = WORKER_PROTOCOL_BYTES,
        default_timeout: float = 120.0,
        startup_timeout: float = 600.0,
        expected_model_sha256: str | None = None,
    ) -> None:
        self.worker_path = worker_path.resolve()
        self.model_path = model_path.resolve()
        self.manifest_path = manifest_path.resolve()
        self.gpu = gpu
        self.context_size = context_size
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.threads = threads
        self.reference_context = reference
        self.snapshot_profile = snapshot_profile
        self.max_response_bytes = max_response_bytes
        self.default_timeout = default_timeout
        self.startup_timeout = startup_timeout
        self.expected_model_sha256 = expected_model_sha256
        self.profile: WorkerHello | None = None
        self.ready = False
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._io_lock = asyncio.Lock()
        self._model_sha256: str | None = None
        self._runtime_sha256: str | None = None
        self._runtime_dir: Path | None = None
        self._stderr_tail: deque[str] = deque(maxlen=200)

    @property
    def pid(self) -> int | None:
        process = self._process
        if process is None or process.returncode is not None:
            return None
        return process.pid

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail)

    def _verify_files(self) -> VerifiedInstall:
        for path, name in (
            (self.worker_path, "worker"),
            (self.model_path, "model"),
            (self.manifest_path, "manifest"),
        ):
            if not path.is_file():
                raise ValueError(f"{name} file is missing")
        manifest = read_manifest(self.manifest_path)
        executable, runtime_dir = resolve_manifest_paths(manifest, self.manifest_path)
        revision = manifest.get("llama_revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("manifest does not name a llama revision")
        if executable != self.worker_path:
            raise ValueError("configured worker differs from manifest")
        if digest(self.worker_path) != manifest.get("executable_sha256"):
            raise ValueError("worker hash differs from manifest")
        if manifest.get("generated_tokens") != 0:
            raise ValueError("manifest permits generated tokens")
        if manifest.get("callbacks_enabled") is not False:
            raise ValueError("manifest enables callbacks")
        if manifest.get("protocol") != "unridden-snapshot-v1":
            raise ValueError("manifest is not the snapshot protocol")
        # A bundle from before ADR 0009 names only its one profile.
        profiles = manifest.get("profiles")
        if profiles is None:
            profiles = {manifest.get("profile"): manifest.get("execution_mode")}
        if not isinstance(profiles, dict) or self.snapshot_profile not in profiles:
            raise ValueError(
                f"manifest does not serve the {self.snapshot_profile} profile"
            )
        if profiles[self.snapshot_profile] != _PROFILE_MODES[self.snapshot_profile]:
            raise ValueError(f"manifest is not a {self.snapshot_profile} build")
        checksums = manifest.get("runtime_sha256")
        if not runtime_dir.is_dir() or not isinstance(checksums, dict) or not checksums:
            raise ValueError("manifest runtime inventory is incomplete")
        typed_checksums: dict[str, str] = {}
        for name, checksum in checksums.items():
            if not isinstance(name, str) or not isinstance(checksum, str):
                raise TypeError("manifest runtime checksum is invalid")
            if digest(runtime_dir / name) != checksum:
                raise ValueError(f"runtime hash differs for {name}")
            typed_checksums[name] = checksum
        runtime_sha256 = bundle_digest(typed_checksums)
        if runtime_sha256 != manifest.get("runtime_bundle_sha256"):
            raise ValueError("runtime bundle hash differs from manifest")
        model_sha256 = digest(self.model_path)
        expected = self.expected_model_sha256
        if expected is not None and model_sha256 != expected:
            raise ValueError("model hash differs from the configured model")
        is_tested_revision(revision)
        return VerifiedInstall(model_sha256, runtime_sha256, runtime_dir, revision)

    async def start(self) -> WorkerHello:
        if self.ready and self.profile is not None:
            return self.profile
        for name in (
            "UNRIDDEN_GEMMA4_EXIT_LAYER",
            "GGML_CUDA_DISABLE_FUSION",
            "GGML_CUDA_DISABLE_GRAPHS",
        ):
            if name in os.environ:
                raise SnapshotUnavailableError(
                    "snapshot backend has incompatible environment settings"
                )
        try:
            self._stderr_tail.clear()
            installed = await asyncio.to_thread(self._verify_files)
            command = [
                str(self.worker_path),
                "--model",
                str(self.model_path),
                "--model-sha256",
                installed.model_sha256,
                "--runtime-sha256",
                installed.runtime_sha256,
                "--runtime-dir",
                str(installed.runtime_dir),
                "--context",
                str(self.context_size),
                "--batch",
                str(self.batch_size),
                "--ubatch",
                str(self.ubatch_size),
                "--threads",
                str(self.threads),
            ]
            if self.gpu:
                command.append("--gpu")
            if self.reference_context:
                command.append("--reference")
            # Omitted for the default so a pre-0009 worker still starts.
            if self.snapshot_profile != SNAPSHOT_PROFILE:
                command.extend(["--profile", self.snapshot_profile])
            environment = os.environ.copy()
            library_path = environment.get("LD_LIBRARY_PATH")
            environment["LD_LIBRARY_PATH"] = (
                f"{installed.runtime_dir}:{library_path}"
                if library_path
                else str(installed.runtime_dir)
            )
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
                limit=self.max_response_bytes + 1,
            )
            self._stderr_task = asyncio.create_task(self._drain_stderr())
            raw = await asyncio.wait_for(
                self._read_json_line(), timeout=self.startup_timeout
            )
            profile = WorkerHello.model_validate(raw)
            if (
                profile.model_sha256 != installed.model_sha256
                or profile.runtime_sha256 != installed.runtime_sha256
                or profile.context_size != self.context_size
                or profile.batch_size != self.batch_size
                or profile.ubatch_size != self.ubatch_size
                or profile.threads != self.threads
                or profile.reference_context != self.reference_context
                or profile.profile != self.snapshot_profile
            ):
                raise SnapshotProtocolError(
                    "worker handshake differs from configuration"
                )
            self._model_sha256 = installed.model_sha256
            self._runtime_sha256 = installed.runtime_sha256
            self._runtime_dir = installed.runtime_dir
            self.profile = profile
            self.ready = True
            return profile
        except asyncio.CancelledError:
            await self._invalidate()
            raise
        except SnapshotError:
            await self._invalidate()
            raise
        except Exception as error:
            await self._invalidate()
            detail = f"snapshot backend failed to start: {error}"
            tail = [line for line in self._stderr_tail if line.strip()][-5:]
            if tail:
                detail += "; worker stderr: " + " | ".join(tail)
            raise SnapshotUnavailableError(detail) from error

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            try:
                line = await process.stderr.readline()
            except ValueError:
                line = b"[oversized stderr line dropped]\n"
            if not line:
                return
            message = line.decode("utf-8", errors="replace").rstrip()[-4096:]
            self._stderr_tail.append(message)
            LOGGER.debug("snapshot worker pid=%s: %s", process.pid, message)

    async def _stop_stderr_task(self) -> None:
        task = self._stderr_task
        self._stderr_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _read_json_line(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise SnapshotUnavailableError("snapshot child is not running")
        try:
            line = await process.stdout.readline()
        except ValueError as error:
            raise SnapshotProtocolError(
                "snapshot response exceeds size limit"
            ) from error
        if not line:
            raise SnapshotUnavailableError("snapshot child closed stdout")
        if len(line) > self.max_response_bytes:
            raise SnapshotProtocolError("snapshot response exceeds size limit")
        try:
            payload: Any = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SnapshotProtocolError("snapshot response is malformed") from error
        if not isinstance(payload, dict):
            raise SnapshotProtocolError("snapshot response is not an object")
        return payload

    def _raise_worker_error(self, raw: dict[str, Any]) -> None:
        try:
            message = WorkerErrorMessage.model_validate(raw)
        except ValidationError as error:
            raise SnapshotProtocolError("worker error is malformed") from error
        if message.code == "invalid_request":
            if message.reason in {"budget", "control_tokens"}:
                raise SnapshotRequestError(
                    "worker rejected the request in preflight",
                    reason=message.reason,
                )
            # The worker and compiler disagree; the child answered in protocol
            # and stays up, but the caller is not at fault.
            raise SnapshotExecutionError("worker preflight rejected a request")
        error_type = _ERROR_CODES.get(message.code)
        if error_type is None:
            raise SnapshotProtocolError(f"unknown worker error code {message.code!r}")
        raise error_type(message.message or message.code)

    async def _request(
        self, envelope: dict[str, Any], *, timeout: float | None
    ) -> dict[str, Any]:
        if not self.ready or self._process is None:
            raise SnapshotUnavailableError("snapshot backend is unavailable")
        deadline = self.default_timeout if timeout is None else timeout
        async with self._io_lock:
            correlation_id = uuid.uuid4().hex
            envelope = {**envelope, "id": correlation_id}
            encoded = (
                json.dumps(
                    envelope,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            if len(encoded) > min(self.max_response_bytes, WORKER_PROTOCOL_BYTES):
                raise SnapshotRequestError(
                    "compiled request exceeds protocol limit", reason="budget"
                )
            process = self._process
            if process.stdin is None:
                await self._invalidate()
                raise SnapshotUnavailableError("snapshot child stdin is closed")
            try:
                async with asyncio.timeout(deadline):
                    process.stdin.write(encoded)
                    await process.stdin.drain()
                    raw = await self._read_json_line()
                if raw.get("id") != correlation_id:
                    raise SnapshotProtocolError("snapshot correlation id differs")
                if raw.get("type") == "error":
                    self._raise_worker_error(raw)
                return raw
            except SnapshotError as error:
                # A typed, in-protocol failure leaves the child running; an
                # ambiguous protocol/transport failure reaps it.
                if isinstance(error, (SnapshotProtocolError, SnapshotUnavailableError)):
                    await self._invalidate()
                raise
            except TimeoutError:
                await self._invalidate()
                raise
            except asyncio.CancelledError:
                await self._invalidate()
                raise
            except ValidationError as error:
                await self._invalidate()
                raise SnapshotProtocolError(
                    "snapshot response failed validation"
                ) from error
            except (BrokenPipeError, ConnectionError) as error:
                # The child is gone, as with a closed stdout: retryable, and the
                # service restarts it on the next request.
                await self._invalidate()
                raise SnapshotUnavailableError(
                    "snapshot child transport failed"
                ) from error
            except BaseException:
                await self._invalidate()
                raise

    # -- typed commands ----------------------------------------------------

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
        raw = await self._request(
            {
                "type": "create",
                "messages": messages,
                "answer_prefix": answer_prefix,
                "freeze": freeze,
                "checkpoints": checkpoints,
                "labels": labels,
                "top_logits": top_logits,
            },
            timeout=timeout,
        )
        return await self._validated(WorkerCreated, raw)

    async def promote(
        self, *, snapshot_id: str, new_id: str, timeout: float | None = None
    ) -> WorkerPromoted:
        raw = await self._request(
            {"type": "promote", "snapshot_id": snapshot_id, "new_id": new_id},
            timeout=timeout,
        )
        return await self._validated(WorkerPromoted, raw)

    async def evaluate(
        self,
        *,
        snapshot_id: str,
        readout_blocks: Checkpoint,
        questions: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> WorkerResult:
        raw = await self._request(
            {
                "type": "evaluate",
                "snapshot_id": snapshot_id,
                "readout_blocks": readout_blocks,
                "questions": questions,
            },
            timeout=timeout,
        )
        return await self._validated(WorkerResult, raw)

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
        raw = await self._request(
            {
                "type": "state_eval",
                "snapshot_id": snapshot_id,
                "messages": messages,
                "answer_prefix": answer_prefix,
                "export": export,
                "top_logits": top_logits,
                "save_as": save_as,
            },
            timeout=timeout,
        )
        return await self._validated(WorkerState, raw)

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
        raw = await self._request(
            {
                "type": "ride",
                "mode": "snapshot",
                "snapshot_id": snapshot_id,
                "messages": messages,
                "answer_prefix": answer_prefix,
                "max_tokens": max_tokens,
                "top_logits": top_logits,
            },
            timeout=timeout,
        )
        return await self._validated(WorkerRide, raw)

    async def inspect(
        self, *, snapshot_id: str, timeout: float | None = None
    ) -> WorkerInspect:
        raw = await self._request(
            {"type": "inspect", "snapshot_id": snapshot_id}, timeout=timeout
        )
        return await self._validated(WorkerInspect, raw)

    async def vectors(
        self,
        *,
        snapshot_id: str,
        which: str,
        row_begin: int,
        row_end: int,
        timeout: float | None = None,
    ) -> WorkerVectors:
        raw = await self._request(
            {
                "type": "vectors",
                "snapshot_id": snapshot_id,
                "which": which,
                "row_begin": row_begin,
                "row_end": row_end,
            },
            timeout=timeout,
        )
        return await self._validated(WorkerVectors, raw)

    async def drop(self, snapshot_id: str) -> int:
        raw = await self._request(
            {"type": "drop", "snapshot_id": snapshot_id}, timeout=None
        )
        return (await self._validated(WorkerDropped, raw)).freed_bytes

    async def save(
        self, snapshot_id: str, directory: Path
    ) -> dict[str, WorkerFileEntry]:
        raw = await self._request(
            {"type": "save", "snapshot_id": snapshot_id, "directory": str(directory)},
            timeout=None,
        )
        return (await self._validated(WorkerSaved, raw)).files

    async def load(
        self, snapshot_id: str, directory: Path, files: dict[str, str]
    ) -> None:
        raw = await self._request(
            {
                "type": "load",
                "snapshot_id": snapshot_id,
                "directory": str(directory),
                "files": files,
            },
            timeout=None,
        )
        await self._validated(WorkerLoaded, raw)

    async def _validated[T: BaseModel](self, model: type[T], raw: dict[str, Any]) -> T:
        try:
            return model.model_validate(raw)
        except ValidationError as error:
            await self._invalidate()
            raise SnapshotProtocolError(
                "snapshot response failed validation"
            ) from error

    async def _invalidate(self) -> None:
        self.ready = False
        self.profile = None
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
        await self._stop_stderr_task()

    async def close(self) -> None:
        self.ready = False
        self.profile = None
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
                with contextlib.suppress(Exception):
                    await process.stdin.wait_closed()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        await self._stop_stderr_task()


__all__ = [
    "SnapshotBackend",
    "SnapshotNativeBackend",
    "WORKER_BINARY_NAME",
    "WORKER_PROTOCOL_BYTES",
]
