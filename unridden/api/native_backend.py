"""Bounded async JSONL transport for one owned native worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from unridden.api.backend import (
    BackendExecutionError,
    BackendProtocolError,
    BackendRequestError,
    BackendUnavailableError,
)
from unridden.api.compiler import CompiledBatch
from unridden.api.native.build import (
    TESTED_LLAMA_REVISION,
    bundle_digest,
    digest,
    is_tested_revision,
)
from unridden.api.native.bundle import read_manifest, resolve_manifest_paths
from unridden.api.schema import BackendProfile, WorkerBatchResult

LOGGER = logging.getLogger("unridden.api.native")
# Lines the workers write when a request or the worker itself fails. Everything
# else on their stderr is llama.cpp load chatter.
WORKER_FAILURE_PREFIXES = ("REQUEST_FAILED ", "PREFLIGHT_FAILED ", "WORKER_FAILED ")


def worker_line_level(message: str) -> int:
    """WARNING for a worker's own failure line, DEBUG for the rest."""
    if message.startswith(WORKER_FAILURE_PREFIXES):
        return logging.WARNING
    return logging.DEBUG


# Mirrors MAX_PROTOCOL_BYTES in native/worker.cpp; a longer line kills the worker.
WORKER_PROTOCOL_BYTES = 4 * 1024 * 1024


class VerifiedInstall(NamedTuple):
    """What the on-disk install proved about itself before the child started."""

    model_sha256: str
    runtime_sha256: str
    runtime_dir: Path
    llama_revision: str


class NativeBackend:
    """One resident child. Any ambiguous transport failure reaps that child."""

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
        batched: bool = False,
        batched_context: int = 8192,
        threads: int = 8,
        max_questions: int = 32,
        max_response_bytes: int = 4 * 1024 * 1024,
        startup_timeout: float = 600.0,
        expected_model_sha256: str | None = None,
    ) -> None:
        self.expected_model_sha256 = expected_model_sha256
        self.worker_path = worker_path.resolve()
        self.model_path = model_path.resolve()
        self.manifest_path = manifest_path.resolve()
        self.gpu = gpu
        self.context_size = context_size
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.batched = batched
        self.batched_context = batched_context if batched else 0
        self.threads = threads
        self.max_questions = max_questions
        self.max_response_bytes = max_response_bytes
        self.startup_timeout = startup_timeout
        self.profile: BackendProfile | None = None
        self.ready = False
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._io_lock = asyncio.Lock()
        self._runtime_dir: Path | None = None
        self._model_sha256: str | None = None
        self._runtime_sha256: str | None = None
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
        # A schema 2 manifest names its executable and runtime relative to
        # itself, so the bundle verifies wherever it was unpacked; a schema 1
        # manifest keeps its absolute paths and is read unchanged. Every hash
        # check below is the same either way.
        executable, runtime_dir = resolve_manifest_paths(manifest, self.manifest_path)
        # Which llama.cpp this was built against is provenance, not integrity:
        # every hash below is checked whatever the revision turns out to be.
        # A revision other than the tested one only earns a warning, because
        # the measurements this project publishes were taken on that one.
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
        if manifest.get("execution_mode") != "full":
            raise ValueError("manifest is not full-only")
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

    async def start(self) -> BackendProfile:
        if self.ready and self.profile is not None:
            return self.profile
        for name in (
            "UNRIDDEN_GEMMA4_EXIT_LAYER",
            "GGML_CUDA_DISABLE_FUSION",
            "GGML_CUDA_DISABLE_GRAPHS",
        ):
            if name in os.environ:
                raise BackendUnavailableError(
                    "native backend has incompatible environment settings"
                )
        try:
            self._stderr_tail.clear()
            installed = await asyncio.to_thread(self._verify_files)
            model_sha256 = installed.model_sha256
            runtime_sha256 = installed.runtime_sha256
            runtime_dir = installed.runtime_dir
            command = [
                str(self.worker_path),
                "--model",
                str(self.model_path),
                "--model-sha256",
                model_sha256,
                "--runtime-sha256",
                runtime_sha256,
                # The worker dlopens its ggml backends from the directory this
                # names. It is passed rather than compiled in, so the binary
                # stays valid wherever its bundle was unpacked.
                "--runtime-dir",
                str(runtime_dir),
                "--context",
                str(self.context_size),
                "--batch",
                str(self.batch_size),
                "--ubatch",
                str(self.ubatch_size),
                "--threads",
                str(self.threads),
                "--max-questions",
                str(self.max_questions),
            ]
            if self.gpu:
                command.append("--gpu")
            if self.batched:
                command += ["--batched", "--batched-context", str(self.batched_context)]
            environment = os.environ.copy()
            current_library_path = environment.get("LD_LIBRARY_PATH")
            environment["LD_LIBRARY_PATH"] = (
                f"{runtime_dir}:{current_library_path}"
                if current_library_path
                else str(runtime_dir)
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
            if raw.get("type") != "hello" or raw.get("protocol_version") != 3:
                raise BackendProtocolError("invalid worker handshake")
            profile_payload = {
                key: value
                for key, value in raw.items()
                if key not in {"type", "protocol_version"}
            }
            # The worker links the runtime but has no way to know which revision
            # produced it, so provenance is attached from the verified manifest.
            profile_payload["llama_revision"] = installed.llama_revision
            profile_payload["tested_revision"] = (
                installed.llama_revision == TESTED_LLAMA_REVISION
            )
            profile = BackendProfile.model_validate(profile_payload)
            if (
                profile.model_sha256 != model_sha256
                or profile.runtime_sha256 != runtime_sha256
                or profile.context_size != self.context_size
                or profile.batch_size != self.batch_size
                or profile.ubatch_size != self.ubatch_size
                or profile.threads != self.threads
                or profile.max_questions != self.max_questions
                or profile.batched_mode != self.batched
                or profile.batched_context != self.batched_context
            ):
                raise BackendProtocolError(
                    "worker handshake differs from configuration"
                )
            self._model_sha256 = model_sha256
            self._runtime_sha256 = runtime_sha256
            self._runtime_dir = runtime_dir
            self.profile = profile
            self.ready = True
            return profile
        except asyncio.CancelledError:
            await self._invalidate()
            raise
        except Exception as error:
            await self._invalidate()
            # A worker that dies before the handshake usually says why on
            # stderr (a missing shared library, a model it refuses); the
            # closed-stdout error alone hides that one line.
            detail = f"native backend failed to start: {error}"
            tail = [line for line in self._stderr_tail if line.strip()][-5:]
            if tail:
                detail += "; worker stderr: " + " | ".join(tail)
            raise BackendUnavailableError(detail) from error

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            try:
                line = await process.stderr.readline()
            except ValueError:
                # Line beyond the reader limit; readline already dropped it.
                line = b"[oversized stderr line dropped]\n"
            if not line:
                return
            message = line.decode("utf-8", errors="replace").rstrip()[-4096:]
            self._stderr_tail.append(message)
            LOGGER.log(
                worker_line_level(message),
                "native worker pid=%s: %s",
                process.pid,
                message,
            )

    async def _stop_stderr_task(self) -> None:
        task = self._stderr_task
        self._stderr_task = None
        if task is not None:
            task.cancel()
            # A drain failure must never replace the caller's real exception.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _read_json_line(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise BackendUnavailableError("native child is not running")
        try:
            line = await process.stdout.readline()
        except ValueError as error:
            # StreamReader reports a line beyond its limit as a bare ValueError.
            raise BackendProtocolError("native response exceeds size limit") from error
        if not line:
            raise BackendUnavailableError("native child closed stdout")
        if len(line) > self.max_response_bytes:
            raise BackendProtocolError("native response exceeds size limit")
        try:
            payload: Any = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendProtocolError("native response is malformed") from error
        if not isinstance(payload, dict):
            raise BackendProtocolError("native response is not an object")
        return payload

    async def evaluate(
        self, batch: CompiledBatch, *, timeout: float
    ) -> WorkerBatchResult:
        if not self.ready or self._process is None:
            raise BackendUnavailableError("native backend is unavailable")
        async with self._io_lock:
            correlation_id = uuid.uuid4().hex
            envelope = {
                "type": "evaluate",
                "id": correlation_id,
                "questions": [
                    question.worker_payload() for question in batch.questions
                ],
            }
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
                raise BackendRequestError(
                    "compiled request exceeds protocol limit", reason="budget"
                )
            process = self._process
            if process.stdin is None:
                await self._invalidate()
                raise BackendUnavailableError("native child stdin is closed")
            try:
                process.stdin.write(encoded)
                await process.stdin.drain()
                raw = await asyncio.wait_for(self._read_json_line(), timeout=timeout)
                if raw.get("id") != correlation_id:
                    raise BackendProtocolError("native correlation id differs")
                if raw.get("type") == "error":
                    if raw.get("code") == "invalid_request":
                        reason = raw.get("reason")
                        if reason in {"budget", "control_tokens"}:
                            raise BackendRequestError(
                                "native request preflight failed", reason=reason
                            )
                        # The worker and compiler disagree. The child answered in
                        # protocol, so it stays up; the caller is not at fault.
                        raise BackendExecutionError(
                            "native preflight rejected a compiled request"
                        )
                    raise BackendProtocolError(
                        "native worker returned an unknown error"
                    )
                result = WorkerBatchResult.model_validate(raw)
                return result
            except (BackendRequestError, BackendExecutionError):
                raise
            except TimeoutError:
                await self._invalidate()
                raise
            except asyncio.CancelledError:
                await self._invalidate()
                raise
            except (
                BackendProtocolError,
                BackendUnavailableError,
                ValidationError,
            ) as error:
                await self._invalidate()
                if isinstance(error, BackendUnavailableError):
                    raise BackendExecutionError(
                        "native worker failed during execution"
                    ) from error
                raise BackendProtocolError(
                    "native response failed validation"
                ) from error
            except (BrokenPipeError, ConnectionError) as error:
                await self._invalidate()
                raise BackendExecutionError("native child transport failed") from error
            except BaseException:
                # Unknown failure: the stream may be desynchronized, so reap.
                await self._invalidate()
                raise

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
