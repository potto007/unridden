"""A model-free ASGI front API. The lifespan owns only HTTP and metadata."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from unridden.api.schema import ErrorBody, ErrorResponse
from unridden.harness.schema import (
    AutomaticDecision,
    ContextCreate,
    ContextCreated,
    ContextId,
    ContextInfo,
    Decision,
    DecisionResult,
)
from unridden.harness.service import HarnessConfig, HarnessError, HarnessService


def _error(error: HarnessError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status,
        headers={"Cache-Control": "no-store"},
        content=ErrorResponse(
            error=ErrorBody(
                code=error.code, message=error.message, retryable=error.retryable
            )
        ).model_dump(exclude_none=True),
    )


def create_app(
    config: HarnessConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    monotonic: Callable[[], float] | None = None,
) -> FastAPI:
    resolved = config or HarnessConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=resolved.backend_url,
            timeout=httpx.Timeout(resolved.request_timeout, connect=5.0),
            transport=transport
            or (
                httpx.AsyncHTTPTransport(uds=resolved.backend_uds)
                if resolved.backend_uds is not None
                else None
            ),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            options = {} if monotonic is None else {"monotonic": monotonic}
            service = HarnessService(client, resolved, **options)
            app.state.harness = service

            async def cleanup() -> None:
                while True:
                    await asyncio.sleep(resolved.cleanup_interval)
                    await service.sweep()

            task = asyncio.create_task(cleanup())
            try:
                yield
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                await service.close()

    app = FastAPI(title="Unridden Context Harness", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(HarnessError)
    async def harness_error(request: Request, error: HarnessError) -> JSONResponse:
        del request
        return _error(error)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del request
        malformed = any(item.get("type") == "json_invalid" for item in error.errors())
        return _error(
            HarnessError(
                400 if malformed else 422,
                "malformed_json" if malformed else "invalid_request",
                "request body is not valid JSON"
                if malformed
                else "request schema is invalid",
            )
        )

    @app.middleware("http")
    async def bounds(request: Request, call_next: Any) -> Any:
        if request.method == "POST" and request.url.path in {
            "/v1/contexts",
            "/v1/decisions",
        }:
            media = request.headers.get("content-type", "").split(";", 1)[0]
            if media.strip().lower() != "application/json":
                return _error(
                    HarnessError(
                        415,
                        "unsupported_media_type",
                        "content-type must be application/json",
                    )
                )
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    count = int(length)
                    if count < 0:
                        raise ValueError
                except ValueError:
                    return _error(
                        HarnessError(
                            400, "invalid_content_length", "invalid content-length"
                        )
                    )
                if count > resolved.max_request_bytes:
                    return _error(
                        HarnessError(
                            413, "request_too_large", "request body is too large"
                        )
                    )
            chunks: list[bytes] = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > resolved.max_request_bytes:
                    return _error(
                        HarnessError(
                            413, "request_too_large", "request body is too large"
                        )
                    )
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        # Handles, answers, and context metadata are private and ephemeral.
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        service: HarnessService = request.app.state.harness
        return await service.health()

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, object]:
        service: HarnessService = request.app.state.harness
        return await service.models()

    @app.post("/v1/contexts", response_model=ContextCreated, status_code=201)
    async def create(body: ContextCreate, request: Request) -> ContextCreated:
        service: HarnessService = request.app.state.harness
        return await service.create(body)

    @app.get("/v1/contexts/{context_id}", response_model=ContextInfo)
    async def inspect(context_id: ContextId, request: Request) -> ContextInfo:
        service: HarnessService = request.app.state.harness
        return service.inspect(context_id)

    @app.delete("/v1/contexts/{context_id}")
    async def delete(context_id: ContextId, request: Request) -> dict[str, object]:
        service: HarnessService = request.app.state.harness
        return await service.delete(context_id)

    @app.post(
        "/v1/decisions", response_model=DecisionResult, response_model_exclude_none=True
    )
    async def decide(
        body: Decision | AutomaticDecision, request: Request, response: Response
    ) -> DecisionResult:
        service: HarnessService = request.app.state.harness
        if isinstance(body, AutomaticDecision):
            result, handle, max_age = await service.automatic_decide(
                body, request.cookies.get("unridden_context")
            )
            response.set_cookie(
                "unridden_context",
                handle,
                max_age=max_age,
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
                path="/v1",
            )
            return result
        return await service.decide(body)

    return app
