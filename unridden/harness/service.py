"""Single-process lifecycle policy over an existing /v2 HTTP backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from unridden.api.compiler import render
from unridden.api.schema import MODEL_ID, Question, State
from unridden.api.snapshots.schema import (
    SnapshotCreateResponse,
    V2DecisionResponse,
)
from unridden.harness.schema import (
    MAX_QUESTIONS,
    AutomaticDecision,
    ContextCreate,
    ContextCreated,
    ContextInfo,
    ContextUsage,
    Decision,
    DecisionResult,
    Execution,
    Timing,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessConfig:
    backend_url: str = "http://127.0.0.1:8091"
    backend_uds: str | None = None
    max_contexts: int = 64
    max_request_bytes: int = 1024 * 1024
    max_response_bytes: int = 4 * 1024 * 1024
    request_timeout: float = 180.0
    cleanup_interval: float = 5.0
    cleanup_timeout: float = 2.0
    shutdown_timeout: float = 5.0
    automatic_ttl: int = 600
    automatic_enabled: bool = True
    max_waiting: int = 16
    queue_timeout: float = 5.0
    backend_busy_timeout: float = 5.0

    def __post_init__(self) -> None:
        url = urlsplit(self.backend_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("backend_url must be an HTTP(S) origin")
        if (
            min(
                self.max_contexts,
                self.max_request_bytes,
                self.max_response_bytes,
                self.request_timeout,
                self.cleanup_interval,
                self.cleanup_timeout,
                self.shutdown_timeout,
            )
            <= 0
        ):
            raise ValueError("harness limits and timeouts must be positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.request_timeout,
                self.cleanup_interval,
                self.cleanup_timeout,
                self.shutdown_timeout,
            )
        ):
            raise ValueError("harness timeouts must be finite")
        _ = url.port
        if self.backend_uds is not None and not self.backend_uds.startswith("/"):
            raise ValueError("backend_uds must be an absolute Unix socket path")
        if not 1 <= self.automatic_ttl <= 86400:
            raise ValueError("automatic_ttl must be within one day")
        if self.max_waiting < 0 or any(
            not math.isfinite(value) or value < 0
            for value in (self.queue_timeout, self.backend_busy_timeout)
        ):
            raise ValueError(
                "queue limits and busy timeouts must be finite/nonnegative"
            )


class HarnessError(Exception):
    def __init__(
        self, status: int, code: str, message: str, *, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable


class Advertisement(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class Limits(Advertisement):
    checkpoints: list[int] = Field(min_length=1)
    context_tokens: int = Field(gt=0)
    questions: int = Field(gt=0)


class Capabilities(Advertisement):
    operations: list[str]
    readout_blocks: list[int] = Field(min_length=1, max_length=1)
    generation: Literal[False]


class BackendModel(Advertisement):
    id: Literal["local-gemma-unridden-v1"]
    profile: Literal["split18-30-v1", "full-v1"]
    prompt_version: str = Field(min_length=1)
    limits: Limits
    capabilities: Capabilities

    @property
    def checkpoint(self) -> int:
        return self.capabilities.readout_blocks[0]


class Catalog(Advertisement):
    models: list[BackendModel] = Field(min_length=1, max_length=1)


class Deleted(Advertisement):
    deleted: str
    freed_bytes: int = Field(ge=0)


@dataclass
class Entry:
    info: ContextInfo
    snapshot_id: str
    deadline: float
    # Retired records still count toward capacity until backend cleanup succeeds.
    # This bounds cleanup debt even if DELETE repeatedly fails.
    retired: bool = False
    automatic: bool = False
    last_used: int = 0
    question_limit: int = 32


@dataclass
class Measurement:
    started: float
    backend_seconds: float = 0.0
    queue_seconds: float = 0.0
    backoff_seconds: float = 0.0
    busy_retries: int = 0

    def finish(self) -> Timing:
        elapsed = time.perf_counter() - self.started
        return Timing(
            orchestration_ms=elapsed * 1000,
            backend_http_ms=self.backend_seconds * 1000,
            local_ms=max(
                0.0,
                elapsed
                - self.backend_seconds
                - self.queue_seconds
                - self.backoff_seconds,
            )
            * 1000,
            queue_ms=self.queue_seconds * 1000,
            backoff_ms=self.backoff_seconds * 1000,
        )


def protocol_error() -> HarnessError:
    return HarnessError(502, "backend_protocol_error", "backend response is invalid")


class HarnessService:
    def __init__(
        self,
        client: httpx.AsyncClient,
        config: HarnessConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self.config = config
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.entries: dict[str, Entry] = {}
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._use_sequence = 0

    @property
    def _busy(self) -> bool:
        return self._lock.locked()

    @asynccontextmanager
    async def admitted(self) -> AsyncIterator[Measurement]:
        measurement = Measurement(time.perf_counter())
        if self._busy or self._waiting:
            if (
                self._waiting >= self.config.max_waiting
                or self.config.queue_timeout == 0
            ):
                raise HarnessError(
                    429, "busy", "harness capacity is busy", retryable=True
                )
            self._waiting += 1
            try:
                async with asyncio.timeout(self.config.queue_timeout):
                    await self._lock.acquire()
            except TimeoutError as error:
                raise HarnessError(
                    429, "busy", "harness wait budget was exhausted", retryable=True
                ) from error
            finally:
                self._waiting -= 1
            measurement.queue_seconds = time.perf_counter() - measurement.started
        else:
            await self._lock.acquire()
        try:
            try:
                async with asyncio.timeout(self.config.request_timeout):
                    yield measurement
            except TimeoutError as error:
                raise HarnessError(
                    504,
                    "request_timeout",
                    "operation exceeded its time budget; its outcome may be incomplete",
                ) from error
        finally:
            self._lock.release()

    async def _call(
        self,
        method: str,
        path: str,
        measurement: Measurement,
        body: dict[str, Any] | None = None,
    ) -> Any:
        deadline = time.perf_counter() + self.config.backend_busy_timeout
        for attempt in range(20):
            try:
                return await self._call_once(method, path, measurement, body)
            except HarnessError as error:
                remaining = deadline - time.perf_counter()
                # Only a recognized pre-admission refusal is safe to replay.
                # Connection errors, timeouts and other failures remain visible.
                if error.code != "backend_busy" or remaining <= 0 or attempt == 19:
                    raise
                measurement.busy_retries += 1
                started = time.perf_counter()
                try:
                    await asyncio.sleep(
                        min(0.05 * 2 ** min(attempt, 3), 0.25, remaining)
                    )
                finally:
                    measurement.backoff_seconds += time.perf_counter() - started
        raise AssertionError("unreachable")

    async def _call_once(
        self,
        method: str,
        path: str,
        measurement: Measurement,
        body: dict[str, Any] | None = None,
    ) -> Any:
        started = time.perf_counter()
        try:
            # Streaming bounds the decoded response even without Content-Length.
            async with self.client.stream(method, path, json=body) as response:
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.config.max_response_bytes:
                        raise protocol_error()
                    chunks.append(chunk)
                try:
                    data = json.loads(b"".join(chunks))
                except (ValueError, UnicodeError) as error:
                    raise protocol_error() from error
                if response.status_code != 200:
                    self._upstream_error(response.status_code, data)
                return data
        except httpx.TimeoutException as error:
            raise HarnessError(
                504,
                "backend_timeout",
                "backend request timed out; its outcome is unknown",
                retryable=False,
            ) from error
        except httpx.HTTPError as error:
            raise HarnessError(
                503,
                "backend_unavailable",
                "backend connection failed; its outcome may be unknown",
                retryable=False,
            ) from error
        finally:
            measurement.backend_seconds += time.perf_counter() - started

    @staticmethod
    def _upstream_error(status: int, data: Any) -> None:
        error = data.get("error", {}) if isinstance(data, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        # Never echo an upstream body, prompt, URL or diagnostic to the caller.
        if status == 404 and code == "snapshot_not_found":
            raise HarnessError(404, "snapshot_not_found", "backend snapshot was lost")
        if status == 429 and code == "busy":
            raise HarnessError(429, "backend_busy", "backend is busy", retryable=True)
        if status == 429:
            raise HarnessError(429, "backend_refused", "backend refused the request")
        if status in {503, 529}:
            raise HarnessError(
                503, "backend_unavailable", "backend is unavailable", retryable=True
            )
        if status == 408:
            raise HarnessError(504, "backend_timeout", "backend request timed out")
        if status == 507:
            raise HarnessError(
                507,
                "backend_capacity",
                "backend snapshot capacity is exhausted",
                retryable=True,
            )
        if status == 422:
            if code == "budget_error":
                raise HarnessError(
                    422, "budget_error", "request exceeds the backend context budget"
                )
            raise HarnessError(
                422, "backend_rejected", "backend rejected the content or model limits"
            )
        raise HarnessError(502, "backend_error", "backend request failed")

    async def _model(self, measurement: Measurement) -> BackendModel:
        data = await self._call("GET", "/v2/models", measurement)
        try:
            model = Catalog.model_validate(data).models[0]
        except ValidationError as error:
            raise protocol_error() from error
        if (
            model.checkpoint not in model.limits.checkpoints
            or not 1 <= model.checkpoint <= 256
            or "continue" not in model.capabilities.operations
        ):
            raise protocol_error()
        return model

    async def health(self) -> dict[str, str]:
        measurement = Measurement(time.perf_counter())
        data = await self._call("GET", "/health", measurement)
        if not isinstance(data, dict) or data.get("status") != "ok":
            raise protocol_error()
        await self._model(measurement)
        return {"status": "ok", "backend": "ok"}

    async def models(self) -> dict[str, object]:
        model = await self._model(Measurement(time.perf_counter()))
        return {
            "models": [model.model_dump()],
            "harness": {
                "version": "decision-gateway-v1",
                "concurrency": 1,
                "max_waiting": self.config.max_waiting,
                "request_timeout_seconds": self.config.request_timeout,
                "queue_timeout_seconds": self.config.queue_timeout,
                "backend_busy_timeout_seconds": self.config.backend_busy_timeout,
                "max_questions": MAX_QUESTIONS,
                "backend_questions_per_batch": min(32, model.limits.questions),
                "max_contexts": self.config.max_contexts,
                "max_request_bytes": self.config.max_request_bytes,
                "max_ttl_seconds": 86_400,
                "persistent_handles": False,
                "automatic_recapture": self.config.automatic_enabled,
                "explicit_context_recapture": False,
                "snapshot_policy": "auto"
                if self.config.automatic_enabled
                else "manual",
                "automatic_sessions": "cookie_scoped_latest_context",
                "automatic_ttl_seconds": self.config.automatic_ttl,
                "automatic_lost_snapshot_recaptures": 1,
            },
        }

    async def create(self, body: ContextCreate) -> ContextCreated:
        async with self.admitted() as measurement:
            return await self._create(body, measurement)

    @staticmethod
    def _state(body: ContextCreate) -> State:
        if body.instructions is None:
            return body.state
        return {"invariant_instructions": body.instructions, "state": body.state}

    def _fingerprint(self, body: ContextCreate, model: BackendModel) -> str:
        return hashlib.sha256(
            render(
                {
                    "version": "explicit-context-v1",
                    "state": self._state(body),
                    "backend": self.config.backend_url.rstrip("/"),
                    "backend_uds": self.config.backend_uds,
                    "model": model.id,
                    "profile": model.profile,
                    "prompt_version": model.prompt_version,
                    "checkpoint": model.checkpoint,
                }
            ).encode()
        ).hexdigest()

    async def _create(
        self,
        body: ContextCreate,
        measurement: Measurement,
        *,
        model: BackendModel | None = None,
        automatic: bool = False,
    ) -> ContextCreated:
        await self._cleanup(measurement)
        if automatic and len(self.entries) >= self.config.max_contexts:
            candidates = [
                (key, entry)
                for key, entry in self.entries.items()
                if entry.automatic and not entry.retired
            ]
            if candidates:
                oldest, _ = min(candidates, key=lambda item: item[1].last_used)
                await self._retire(oldest, measurement)
        if len(self.entries) >= self.config.max_contexts:
            raise HarnessError(
                507,
                "context_capacity",
                "harness context capacity is exhausted",
                retryable=True,
            )
        model = model or await self._model(measurement)
        # Only invariant text can enter this request. A decision question
        # cannot be supplied to the strict ContextCreate schema.
        state = self._state(body)
        fingerprint = self._fingerprint(body, model)
        started = self.monotonic()
        expires_at = self.wall_time() + body.ttl_seconds
        before_capture = measurement.backend_seconds
        data = await self._call(
            "POST",
            "/v2/snapshots",
            measurement,
            {
                "model": MODEL_ID,
                "input": {"kind": "context", "state": state},
                "checkpoints": [model.checkpoint],
                "persistence": "memory",
                "ttl_seconds": body.ttl_seconds,
            },
        )
        try:
            created = SnapshotCreateResponse.model_validate(data)
        except ValidationError as error:
            raise protocol_error() from error
        rows = created.snapshots
        if len(rows) != 1:
            raise protocol_error()
        row = rows[0]
        if (
            row.completed_blocks != model.checkpoint
            or row.boundary != "context"
            or "continue" not in row.capabilities
            or row.parent is not None
            or row.context_parent is not None
        ):
            raise protocol_error()
        context_id = "ctx_" + secrets.token_hex(24)
        info = ContextInfo(
            context_id=context_id,
            input_sha256=fingerprint,
            model=model.id,
            profile=model.profile,
            prompt_version=model.prompt_version,
            completed_blocks=model.checkpoint,
            expires_at=expires_at,
            capture_ms=(measurement.backend_seconds - before_capture) * 1000,
            successful_decisions=0,
        )
        self.entries[context_id] = Entry(
            info=info,
            snapshot_id=row.id,
            deadline=started + body.ttl_seconds,
            automatic=automatic,
            question_limit=min(32, model.limits.questions),
        )
        self._get(context_id)  # A capture may itself exceed a very short TTL.
        return ContextCreated(**info.model_dump(), timing=measurement.finish())

    def _get(self, context_id: str) -> Entry:
        entry = self.entries.get(context_id)
        if entry is None:
            raise HarnessError(404, "context_not_found", "context is unknown")
        if self.monotonic() >= entry.deadline:
            entry.retired = True
        if entry.retired:
            raise HarnessError(
                410,
                "context_expired",
                "context is expired or deleted; create a new one",
            )
        return entry

    def inspect(self, context_id: str) -> ContextInfo:
        return self._get(context_id).info.model_copy()

    async def decide(self, body: Decision) -> DecisionResult:
        async with self.admitted() as measurement:
            return await self._decide(body, measurement)

    async def _decide(self, body: Decision, measurement: Measurement) -> DecisionResult:
        entry = self._get(body.context_id)
        responses: list[V2DecisionResponse] = []
        items = list(body.questions.items())
        for offset in range(0, len(items), entry.question_limit):
            questions = dict(items[offset : offset + entry.question_limit])
            try:
                responses.append(await self._ask_batch(entry, questions, measurement))
            except HarnessError as error:
                if error.code == "snapshot_not_found":
                    del self.entries[body.context_id]
                    raise HarnessError(
                        410,
                        "context_lost_after_progress" if responses else "context_lost",
                        "backend snapshot was lost; no complete answer bundle returned",
                    ) from error
                if responses:
                    # Work already completed, although no partial bundle is
                    # returned. Do not advertise replay of the entire request.
                    error.retryable = False
                raise
        # Every question branches independently from the same immutable parent.
        # Publish only a complete bundle; never mix parents or retry earlier work.
        result = responses[0]
        for additional in responses[1:]:
            result.answers.update(additional.answers)
            result.usage.input_tokens += additional.usage.input_tokens
            usage, extra = result.snapshot_usage, additional.snapshot_usage
            usage.suffix_tokens.update(extra.suffix_tokens)
            usage.block_tokens.lower += extra.block_tokens.lower
            usage.block_tokens.upper += extra.block_tokens.upper
            usage.reused_prefix_tokens += extra.reused_prefix_tokens
            usage.restored_bytes += extra.restored_bytes
            usage.timing.restore_ms += extra.timing.restore_ms
            usage.timing.promotion_ms += extra.timing.promotion_ms
            usage.timing.inference_ms += extra.timing.inference_ms
        entry.info.successful_decisions += 1
        self._use_sequence += 1
        entry.last_used = self._use_sequence
        return DecisionResult(
            **result.model_dump(),
            context_usage=ContextUsage(
                context_id=body.context_id,
                successful_decisions=entry.info.successful_decisions,
                capture_ms=entry.info.capture_ms,
                amortized_capture_ms=(
                    entry.info.capture_ms / entry.info.successful_decisions
                ),
            ),
            execution=Execution(
                decision_batches=len(responses), busy_retries=measurement.busy_retries
            ),
            timing=measurement.finish(),
        )

    async def _ask_batch(
        self,
        entry: Entry,
        questions: dict[str, Question],
        measurement: Measurement,
    ) -> V2DecisionResponse:
        data = await self._call(
            "POST",
            "/v2/decisions",
            measurement,
            {
                "model": entry.info.model,
                "snapshot": {"id": entry.snapshot_id, "relationship": "followup"},
                "questions": {
                    key: value.model_dump(exclude_none=True)
                    for key, value in questions.items()
                },
                "readout": {"completed_blocks": entry.info.completed_blocks},
                "save_result_snapshot": False,
            },
        )
        try:
            result = V2DecisionResponse.model_validate(data)
        except ValidationError as error:
            raise protocol_error() from error
        usage = result.snapshot_usage
        if (
            set(result.answers) != set(questions)
            or set(usage.suffix_tokens) != set(questions)
            or usage.requested_parent != entry.snapshot_id
            or usage.effective_parent != entry.snapshot_id
            or usage.profile != entry.info.profile
            or usage.promotion != "none"
            or usage.child is not None
            or any(
                result.answers[key].type != question.type
                for key, question in questions.items()
            )
        ):
            raise protocol_error()
        return result

    async def automatic_decide(
        self,
        body: AutomaticDecision,
        cookie: str | None,
    ) -> tuple[DecisionResult, str, int]:
        """One latest stable context per unguessable client cookie, never global."""
        if not self.config.automatic_enabled:
            raise HarnessError(
                422,
                "manual_context_required",
                "manual policy requires creating a context and sending context_id",
            )
        if body.model != MODEL_ID:
            raise HarnessError(404, "unknown_model", "requested model is unavailable")
        async with self.admitted() as measurement:
            model = await self._model(measurement)
            context = ContextCreate(
                state=body.state,
                instructions=body.instructions,
                ttl_seconds=self.config.automatic_ttl,
            )
            fingerprint = self._fingerprint(context, model)
            entry = self.entries.get(cookie or "")
            reason = "new_session" if cookie is None else "expired_or_evicted"
            if entry is not None and not entry.automatic:
                entry = None
            if entry is not None:
                if entry.retired or self.monotonic() >= entry.deadline:
                    await self._retire(cookie or "", measurement)
                    entry = None
                elif entry.info.input_sha256 != fingerprint:
                    reason = "state_or_profile_changed"
                    await self._retire(cookie or "", measurement)
                    entry = None
            hit = entry is not None
            if entry is None:
                created = await self._create(
                    context, measurement, model=model, automatic=True
                )
                handle = created.context_id
                entry = self.entries[handle]
            else:
                handle = entry.info.context_id
                reason = "same_stable_context"
                entry.question_limit = min(32, model.limits.questions)
            captured = 0.0 if hit else entry.info.capture_ms
            try:
                try:
                    result = await self._decide(
                        Decision(context_id=handle, questions=body.questions),
                        measurement,
                    )
                except HarnessError as error:
                    if error.code != "context_lost":
                        raise
                    # A definitive snapshot-not-found is pre-inference. Recapture
                    # once; never retry a timeout, transport error, or failed answer.
                    hit = False
                    reason = "backend_lost"
                    created = await self._create(context, measurement, automatic=True)
                    handle = created.context_id
                    entry = self.entries[handle]
                    captured += entry.info.capture_ms
                    result = await self._decide(
                        Decision(context_id=handle, questions=body.questions),
                        measurement,
                    )
            except BaseException:
                if not hit and handle in self.entries:
                    await self._retire(handle, measurement)
                raise
            usage = result.context_usage
            usage.mode = "automatic_context"
            usage.context_id = None
            usage.cache = "hit" if hit else "miss"
            usage.capture_reason = reason
            usage.capture_ms_this_request = captured
            usage.expires_at = entry.info.expires_at
            usage.session_transport = "cookie"
            result.timing = measurement.finish()
            return result, handle, max(1, int(entry.deadline - self.monotonic()))

    async def _retire(self, handle: str, measurement: Measurement) -> None:
        if handle not in self.entries:
            return
        self.entries[handle].retired = True
        try:
            async with asyncio.timeout(self.config.cleanup_timeout):
                await self._drop(handle, measurement)
        except (HarnessError, TimeoutError):
            LOGGER.warning("automatic context cleanup deferred")

    async def _drop(self, context_id: str, measurement: Measurement) -> None:
        entry = self.entries[context_id]
        entry.retired = True
        try:
            data = await self._call(
                "DELETE",
                "/v2/snapshots/" + quote(entry.snapshot_id, safe=""),
                measurement,
            )
            try:
                deleted = Deleted.model_validate(data)
            except ValidationError as error:
                raise protocol_error() from error
            if deleted.deleted != entry.snapshot_id:
                raise protocol_error()
        except HarnessError as error:
            if error.code != "snapshot_not_found":
                raise
        del self.entries[context_id]

    async def delete(self, context_id: str) -> dict[str, object]:
        async with self.admitted() as measurement:
            if context_id not in self.entries:
                raise HarnessError(404, "context_not_found", "context is unknown")
            await self._drop(context_id, measurement)
            return {"deleted": context_id}

    async def _cleanup(self, measurement: Measurement) -> None:
        try:
            async with asyncio.timeout(self.config.cleanup_timeout):
                for context_id, entry in list(self.entries.items()):
                    if not (entry.retired or self.monotonic() >= entry.deadline):
                        continue
                    await self._drop(context_id, measurement)
        except (HarnessError, TimeoutError):
            # Keep bounded cleanup debt and retry. A down backend must not hold
            # admission for the much longer inference request timeout.
            LOGGER.warning("context cleanup deferred")

    async def sweep(self) -> None:
        if self._busy or self._waiting:
            return
        async with self.admitted() as measurement:
            await self._cleanup(measurement)

    async def close(self) -> None:
        for entry in self.entries.values():
            entry.retired = True
        try:
            async with asyncio.timeout(self.config.shutdown_timeout):
                await self.sweep()
        except TimeoutError:
            LOGGER.warning("context cleanup timed out during shutdown")
