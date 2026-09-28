"""The front API deliberately accepts context state separately from questions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from unridden.api.schema import (
    DecisionRequest,
    Question,
    State,
    StrictModel,
    _reject_nonfinite,
)
from unridden.api.snapshots.schema import V2DecisionResponse

ContextId = Annotated[str, Field(pattern=r"^ctx_[0-9a-f]{48}$")]
MAX_QUESTIONS = 64


class AutomaticDecision(DecisionRequest):
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)
    instructions: str | None = Field(default=None, min_length=1)


class ContextCreate(StrictModel):
    state: State
    # These instructions must apply to every question asked of the context.
    instructions: str | None = Field(default=None, min_length=1)
    ttl_seconds: int = Field(default=600, ge=1, le=86_400)

    @model_validator(mode="before")
    @classmethod
    def finite(cls, value: object) -> object:
        _reject_nonfinite(value, "request")
        return value


class Decision(StrictModel):
    context_id: ContextId
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @model_validator(mode="before")
    @classmethod
    def finite(cls, value: object) -> object:
        _reject_nonfinite(value, "request")
        return value

    @field_validator("questions")
    @classmethod
    def nonempty_ids(cls, value: dict[str, Question]) -> dict[str, Question]:
        if any(not key for key in value):
            raise ValueError("question ids must be nonempty")
        return value


class Timing(StrictModel):
    # Service method entry to response construction. Excludes ASGI parsing,
    # response serialization/transmission, and the caller's network latency.
    orchestration_ms: float = Field(ge=0)
    backend_http_ms: float = Field(ge=0)
    local_ms: float = Field(ge=0)
    queue_ms: float = Field(default=0, ge=0)
    backoff_ms: float = Field(default=0, ge=0)


class Execution(StrictModel):
    decision_batches: int = Field(ge=1)
    busy_retries: int = Field(ge=0)


class ContextInfo(StrictModel):
    context_id: ContextId
    input_sha256: str
    model: str
    profile: str
    prompt_version: str
    completed_blocks: int
    expires_at: float
    capture_ms: float
    successful_decisions: int


class ContextCreated(ContextInfo):
    timing: Timing


class ContextUsage(StrictModel):
    context_id: ContextId | None
    successful_decisions: int
    # Initial capture cost for this parent; automatic misses include it in
    # this request, while manual creates pay it before the decision.
    capture_ms: float
    amortized_capture_ms: float
    mode: Literal["immutable_context", "automatic_context"] = "immutable_context"
    cache: Literal["hit", "miss"] | None = None
    capture_reason: str | None = None
    capture_ms_this_request: float = 0.0
    expires_at: float | None = None
    session_transport: Literal["cookie"] | None = None


class DecisionResult(V2DecisionResponse):
    context_usage: ContextUsage
    timing: Timing
    execution: Execution
