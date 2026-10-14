"""Strict wire and worker schemas for the isolated API."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODEL_ID = "local-gemma-unridden-v1"
type EvaluationMode = Literal["sequential", "batched"]
# The only reason the worker may decline a batched request (ADR 0004).
type BatchedFallback = Literal["context"]
type JsonValue = (
    str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
)
type Entry = str | dict[str, JsonValue] | list[JsonValue] | None
type ScoreEntry = str | dict[str, JsonValue] | list[JsonValue]
type State = str | dict[str, JsonValue] | list[JsonValue]


def _reject_nonfinite(value: object, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class NoulCriteria(StrictModel):
    true: Entry = None
    false: Entry = None


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Entry = None
    criteria: NoulCriteria | None = None

    @model_validator(mode="after")
    def _specified(self) -> Self:
        rubric = self.criteria
        if self.instructions is None and (
            rubric is None or (rubric.true is None and rubric.false is None)
        ):
            raise ValueError("noul question must specify instructions or a rubric")
        return self


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Entry = None
    criteria: dict[str, Entry] = Field(min_length=2, max_length=255)

    @field_validator("criteria")
    @classmethod
    def _option_ids_nonempty(cls, value: dict[str, Entry]) -> dict[str, Entry]:
        if any(not key for key in value):
            raise ValueError("choice option ids must be nonempty")
        return value


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Entry = None
    criteria: list[ScoreEntry] = Field(min_length=2, max_length=10)


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class DecisionRequest(StrictModel):
    model: str = MODEL_ID
    state: State
    questions: dict[str, Question] = Field(min_length=1, max_length=32)

    @model_validator(mode="before")
    @classmethod
    def _finite_json(cls, value: Any) -> Any:
        _reject_nonfinite(value, "request")
        return value

    @field_validator("questions")
    @classmethod
    def _question_ids_nonempty(cls, value: dict[str, Question]) -> dict[str, Question]:
        if any(not key for key in value):
            raise ValueError("question ids must be nonempty")
        return value


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: Literal[0] = 0


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, ScoreEntry]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class FullVocabularyArgmax(StrictModel):
    token_id: int = Field(ge=0)
    logit: float


GateAction = Literal["accept_primary", "run_permutation_probe", "escalate"]
GateReason = Literal[
    "high_calibrated_margin",
    "borderline_margin",
    "low_calibrated_margin",
    "low_coverage",
    "permutation_flip",
    "permutation_tvd",
    "low_post_probe_margin",
    "permutation_stable",
]
CalibrationStatus = Literal[
    "fitted_requires_held_out_validation", "validated_on_held_out"
]
TemperatureSource = Literal["bucket", "type_fallback", "default"]


class CalibrationDiagnostics(StrictModel):
    """Temperature-scaled readout of one question (ADR 0012).

    Additive to the pinned `confidence` (`max_probability_v1`): the public
    answer is unchanged and this block says what the same logits look like
    under the operator's fitted per-bucket temperature.
    """

    confidence_formula: Literal["temperature_scaled_max_probability_v1"] = (
        "temperature_scaled_max_probability_v1"
    )
    table_status: CalibrationStatus
    bucket: str
    temperature: float = Field(ge=0.5, le=5.0)
    temperature_source: TemperatureSource
    calibrated_probabilities: dict[str, float]
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    calibrated_probability_margin: float = Field(ge=0.0, le=1.0)
    scaled_logit_margin: float = Field(ge=0.0)
    calibrated_normalized_entropy: float = Field(ge=0.0, le=1.0)


class PermutationProbeDiagnostics(StrictModel):
    """The option-reversed re-ask of one question, realigned to its order."""

    flipped: bool
    permuted_top1: str
    max_probability_delta: float = Field(ge=0.0, le=1.0)
    total_variation_distance: float = Field(ge=0.0, le=1.0)
    post_probe_margin: float = Field(ge=0.0, le=1.0)
    post_probe_entropy: float = Field(ge=0.0, le=1.0)
    realigned_probabilities: dict[str, float]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processed_tokens: int = Field(ge=1)
    generated_tokens: Literal[0] = 0


class GateDiagnostics(StrictModel):
    """Escalation gate verdict for one question (`unridden.api.gate`)."""

    action: GateAction
    reason: GateReason
    probe: PermutationProbeDiagnostics | None = None


class QuestionDiagnostics(StrictModel):
    raw_label_logits: list[float]
    logit_margin: float = Field(ge=0.0)
    probability_margin: float = Field(ge=0.0, le=1.0)
    normalized_entropy: float = Field(ge=0.0, le=1.0)
    token_mapping: dict[str, int]
    coverage: float = Field(ge=0.0, le=1.0)
    full_vocabulary_argmax: FullVocabularyArgmax
    conditional_score_semantics: Literal[
        "softmax_over_allowed_single_token_labels_v1"
    ] = "softmax_over_allowed_single_token_labels_v1"
    confidence_formula: Literal["max_probability_v1"] = "max_probability_v1"
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str
    cache_cleared: bool
    prompt_tokens: int = Field(ge=1)
    processed_tokens: int = Field(ge=1)
    reused_tokens: int = Field(ge=0)
    # How this question was evaluated, and how many question sequences shared
    # its decodes. Sequential evaluation is always a batch of one.
    evaluation_mode: EvaluationMode
    batch_sequences: int = Field(ge=1, le=32)
    timing_ms: float = Field(ge=0.0)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_mode: Literal["full"]
    generated_tokens: Literal[0]
    callbacks_enabled: Literal[False]
    # Present only when the server loaded a calibration table bound to this
    # model, runtime and prompt version (ADR 0012). Never changes `confidence`.
    calibration: CalibrationDiagnostics | None = None
    gate: GateDiagnostics | None = None


class Evaluation(StrictModel):
    """How the request was evaluated. Present only on a batched worker.

    A batched worker can answer a request sequentially (the ADR 0004 fallback),
    and that is a different numeric regime, so it is reported in the public body
    and not only under `?diagnostics=true`.
    """

    mode: EvaluationMode
    fallback: BatchedFallback | None = None


class DecisionResponse(StrictModel):
    model: Literal["local-gemma-unridden-v1"]
    answers: dict[str, Answer]
    usage: Usage
    evaluation: Evaluation | None = None
    diagnostics: dict[str, QuestionDiagnostics] | None = None


class BackendProfile(StrictModel):
    model_id: Literal["local-gemma-unridden-v1"]
    model_name: str
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # The llama.cpp commit the linked runtime was built from, and whether that
    # is the revision this project's published measurements were taken on.
    llama_revision: str = Field(min_length=1)
    tested_revision: bool
    labels: list[str] = Field(min_length=2, max_length=255)
    label_token_ids: list[int] = Field(min_length=2, max_length=255)
    context_size: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    ubatch_size: int = Field(ge=1)
    threads: int = Field(ge=1)
    max_questions: int = Field(ge=1, le=32)
    batched_mode: bool
    batched_context: int = Field(ge=0)
    generated_tokens: Literal[0]
    callbacks_enabled: Literal[False]
    execution_mode: Literal["full"]

    @model_validator(mode="after")
    def _label_mapping(self) -> Self:
        if self.batched_mode:
            if self.batched_context < self.context_size:
                raise ValueError("batched context is below the context size")
        elif self.batched_context != 0:
            raise ValueError("a sequential worker reports a batched context")
        if len(self.labels) != len(self.label_token_ids):
            raise ValueError("label and token counts differ")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if len(set(self.label_token_ids)) != len(self.label_token_ids):
            raise ValueError("label token ids must be unique")
        return self


class WorkerQuestionResult(StrictModel):
    id: str = Field(min_length=1)
    label_logits: list[float] = Field(min_length=2, max_length=255)
    label_token_ids: list[int] = Field(min_length=2, max_length=255)
    allowed_label_mass: float = Field(ge=0.0, le=1.0)
    full_vocabulary_argmax: FullVocabularyArgmax
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_tokens: int = Field(ge=1)
    processed_tokens: int = Field(ge=1)
    reused_tokens: int = Field(ge=0)
    cache_cleared: bool
    evaluation_mode: EvaluationMode
    batch_sequences: int = Field(ge=1, le=32)
    timing_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _finite_and_aligned(self) -> Self:
        if self.processed_tokens + self.reused_tokens != self.prompt_tokens:
            raise ValueError("worker token accounting does not cover the prompt")
        if self.cache_cleared != (self.reused_tokens == 0):
            raise ValueError("worker cache state contradicts its reused tokens")
        if self.evaluation_mode == "sequential" and self.batch_sequences != 1:
            raise ValueError("a sequential question shared its decode with siblings")
        if len(self.label_logits) != len(self.label_token_ids):
            raise ValueError("worker label logit and token counts differ")
        if not all(math.isfinite(item) for item in self.label_logits):
            raise ValueError("worker returned nonfinite label logits")
        if not math.isfinite(self.full_vocabulary_argmax.logit):
            raise ValueError("worker returned nonfinite vocabulary argmax")
        return self


class WorkerBatchResult(StrictModel):
    type: Literal["result"]
    id: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_tokens: Literal[0]
    callbacks_enabled: Literal[False]
    execution_mode: Literal["full"]
    # Set only when a batched worker had to answer this request sequentially.
    batched_fallback: BatchedFallback | None
    questions: list[WorkerQuestionResult] = Field(max_length=32)

    @model_validator(mode="after")
    def _one_mode_per_request(self) -> Self:
        modes = {question.evaluation_mode for question in self.questions}
        if len(modes) > 1:
            raise ValueError("worker mixed evaluation modes within one request")
        if self.batched_fallback is not None and modes != {"sequential"}:
            raise ValueError("worker reported a fallback it did not take")
        if modes == {"batched"} and any(
            question.batch_sequences != len(self.questions)
            for question in self.questions
        ):
            raise ValueError("worker sibling count differs from the request")
        return self


class ErrorBody(StrictModel):
    code: str
    message: str
    field: str | None = None
    retryable: bool


class ErrorResponse(StrictModel):
    error: ErrorBody
