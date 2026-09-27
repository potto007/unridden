"""Strict wire and worker schemas for the /v2 snapshot extension.

The public `/v2` bodies live beside the worker-message models they map to.
Every native response is validated strictly here (`extra="forbid"`, finite
numbers, aligned accounting, echoed ids) so a worker that drifts from the
protocol is rejected at the transport boundary rather than believed.
"""

from __future__ import annotations

import base64
import binascii
import math
import struct
from typing import Annotated, Final, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from unridden.api.schema import (
    MODEL_ID,
    Answer,
    FullVocabularyArgmax,
    Question,
    State,
    StrictModel,
    Usage,
    WorkerQuestionResult,
    _reject_nonfinite,
)

CONTEXT_PROMPT_VERSION = "unridden-gemma-context-v1"
SNAPSHOT_PROFILE: Final = "split18-30-v1"
# One whole-model context, any Gemma 4 (ADR 0009).
FULL_PROFILE: Final = "full-v1"
SNAPSHOT_PROTOCOL = "unridden-snapshot-v1"
type ProfileName = Literal["split18-30-v1", "full-v1"]
# More blocks than any served model has; the running profile narrows it.
MAX_BLOCKS = 256
# A block boundary a snapshot is frozen at. The worker profile decides which
# are offered: 18 and 30 for split18-30-v1, the block count for full-v1.
Checkpoint = Annotated[int, Field(ge=1, le=MAX_BLOCKS)]
type Persistence = Literal["memory", "disk"]
type Relationship = Literal["followup", "replace_question"]
type Boundary = Literal["context", "readout"]
type Capability = Literal["continue", "promote", "inspect"]
type Promotion = Literal["performed", "reused", "none"]
type ExportKind = Literal["last_residual", "last_normalized", "top_logits"]
# How the exported vector was produced, so a caller never mistakes a raw
# residual for a post-norm head input (design "capture integrity"). The
# block-18 residual is only reachable through the `vectors` artifact endpoint.
type Representation = Literal[
    "raw_residual_after_block_18",
    "raw_residual_after_block_30",
    "raw_residual_after_final_block",
    "post_final_norm_head_input",
]
# One day, bounded so a runaway ttl cannot pin a snapshot on the host forever.
MAX_TTL_SECONDS = 86_400
# The worker returns at most this many rows per `vectors` call (protocol).
MAX_VECTOR_ROWS = 64
# The layer count and boundary split18-30-v1 requires (protocol handshake).
# A split worker that reports anything else is refused at startup.
PROFILE_N_EMBD = 2816
PROFILE_N_LAYER = 30
PROFILE_SPLIT_BLOCK = 18


class ContextInput(StrictModel):
    kind: Literal["context"]
    state: State


class DecisionInput(StrictModel):
    kind: Literal["decision"]
    state: State
    question: Question


class PromptInput(StrictModel):
    kind: Literal["prompt"]
    state: State
    prompt: str = Field(min_length=1)


SnapshotInput = Annotated[
    ContextInput | DecisionInput | PromptInput,
    Field(discriminator="kind"),
]


def _checkpoints(value: list[Checkpoint]) -> list[Checkpoint]:
    if len(set(value)) != len(value):
        raise ValueError("checkpoints must be unique")
    # A 30 checkpoint is derived from the same lower pass as its 18 (design
    # "creating the pair"); reporting them low-to-high keeps that order stable.
    return sorted(value)


class SnapshotCreateRequest(StrictModel):
    model: str = MODEL_ID
    input: SnapshotInput
    checkpoints: list[Checkpoint] = Field(min_length=1, max_length=2)
    persistence: Persistence = "memory"
    ttl_seconds: int = Field(ge=1, le=MAX_TTL_SECONDS)

    @model_validator(mode="before")
    @classmethod
    def _finite_json(cls, value: object) -> object:
        _reject_nonfinite(value, "request")
        return value

    @field_validator("checkpoints")
    @classmethod
    def _unique_sorted(cls, value: list[Checkpoint]) -> list[Checkpoint]:
        return _checkpoints(value)


class SnapshotRow(StrictModel):
    """One snapshot as it appears in a create/metadata response."""

    id: str = Field(min_length=1)
    completed_blocks: Checkpoint
    boundary: Boundary
    parent: str | None = None
    context_parent: str | None = None
    capabilities: list[Capability] = Field(min_length=1, max_length=3)

    @field_validator("capabilities")
    @classmethod
    def _distinct(cls, value: list[Capability]) -> list[Capability]:
        if len(set(value)) != len(value):
            raise ValueError("capabilities must be unique")
        return value


class CreateUsage(StrictModel):
    generated_tokens: Literal[0] = 0


class SnapshotCreateResponse(StrictModel):
    snapshots: list[SnapshotRow] = Field(min_length=1, max_length=2)
    usage: CreateUsage = Field(default_factory=CreateUsage)


class SnapshotMetadata(StrictModel):
    id: str = Field(min_length=1)
    completed_blocks: Checkpoint
    boundary: Boundary
    parent: str | None
    context_parent: str | None
    persistence: Persistence
    resident: bool
    tokens: int = Field(ge=1)
    capabilities: list[Capability] = Field(min_length=1, max_length=3)
    created_at: float = Field(ge=0.0)
    expires_at: float = Field(ge=0.0)


class SnapshotRef(StrictModel):
    id: str = Field(min_length=1)
    relationship: Relationship


class Readout(StrictModel):
    completed_blocks: Checkpoint


class V2DecisionRequest(StrictModel):
    model: str = MODEL_ID
    snapshot: SnapshotRef
    questions: dict[str, Question] = Field(min_length=1, max_length=32)
    readout: Readout
    save_result_snapshot: bool = False

    @model_validator(mode="before")
    @classmethod
    def _finite_json(cls, value: object) -> object:
        _reject_nonfinite(value, "request")
        return value

    @field_validator("questions")
    @classmethod
    def _question_ids_nonempty(cls, value: dict[str, Question]) -> dict[str, Question]:
        if any(not key for key in value):
            raise ValueError("question ids must be nonempty")
        return value


class BlockTokens(StrictModel):
    # Tokens pushed through blocks 1-18 and 19-30 respectively. Reported apart
    # from logical input tokens so finishing part of the depth on N tokens is
    # never misread as zero work or as a complete prefill (design "Proposed
    # API").
    lower: int = Field(ge=0)
    upper: int = Field(ge=0)


class SnapshotTiming(StrictModel):
    restore_ms: float = Field(ge=0.0)
    promotion_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)


class SnapshotUsage(StrictModel):
    requested_parent: str = Field(min_length=1)
    effective_parent: str = Field(min_length=1)
    promotion: Promotion
    profile: str = Field(min_length=1)
    # Per question, the suffix length that branched from the parent.
    suffix_tokens: dict[str, int]
    block_tokens: BlockTokens
    reused_prefix_tokens: int = Field(ge=0)
    restored_bytes: int = Field(ge=0)
    timing: SnapshotTiming
    # Per question, the child snapshot id when `save_result_snapshot` was set.
    child: dict[str, str] | None = None

    @field_validator("suffix_tokens")
    @classmethod
    def _positive_suffix(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 1 for count in value.values()):
            raise ValueError("each question must contribute at least one token")
        return value


class V2DecisionResponse(StrictModel):
    model: Literal["local-gemma-unridden-v1"]
    answers: dict[str, Answer]
    usage: Usage
    snapshot_usage: SnapshotUsage


def _decode_f32(shape: list[int], encoded: str) -> list[float]:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("vector is not valid base64") from error
    count = 1
    for dimension in shape:
        count *= dimension
    if len(raw) != count * 4:
        raise ValueError("vector byte length does not match its shape")
    values = list(struct.unpack(f"<{count}f", raw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vector contains nonfinite values")
    return values


class VectorArtifact(StrictModel):
    dtype: Literal["f32"]
    byte_order: Literal["little"] = "little"
    shape: list[int] = Field(min_length=1, max_length=2)
    representation: Representation
    base64: str

    @field_validator("shape")
    @classmethod
    def _bounded_shape(cls, value: list[int]) -> list[int]:
        if any(dimension < 1 for dimension in value):
            raise ValueError("vector shape dimensions must be positive")
        rows = value[0] if len(value) == 2 else 1
        if rows > MAX_VECTOR_ROWS:
            raise ValueError("vector exceeds the row bound")
        return value

    @model_validator(mode="after")
    def _decodes(self) -> Self:
        _decode_f32(self.shape, self.base64)
        return self


class TopLogit(StrictModel):
    token_id: int = Field(ge=0)
    logit: float

    @field_validator("logit")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("logit must be finite")
        return value


class StateReadout(StrictModel):
    completed_blocks: Checkpoint
    export: list[ExportKind] = Field(default_factory=list, max_length=3)
    top_logits: int = Field(ge=0, le=MAX_VECTOR_ROWS, default=0)

    @field_validator("export")
    @classmethod
    def _distinct(cls, value: list[ExportKind]) -> list[ExportKind]:
        if len(set(value)) != len(value):
            raise ValueError("export kinds must be unique")
        return value

    @model_validator(mode="after")
    def _top_logits_agree(self) -> Self:
        wants = "top_logits" in self.export
        if wants and self.top_logits == 0:
            raise ValueError("top_logits export requires a positive count")
        if not wants and self.top_logits != 0:
            raise ValueError("top_logits count set without requesting the export")
        return self


class StateEvaluationRequest(StrictModel):
    model: str = MODEL_ID
    snapshot: SnapshotRef
    prompt: str = Field(min_length=1)
    readout: StateReadout
    save_result_snapshot: bool = False


class StateEvaluationResponse(StrictModel):
    model: Literal["local-gemma-unridden-v1"]
    suffix_tokens: int = Field(ge=1)
    block_tokens: BlockTokens
    reused_prefix_tokens: int = Field(ge=0)
    restored_bytes: int = Field(ge=0)
    timing: SnapshotTiming
    vectors: dict[ExportKind, VectorArtifact] = Field(default_factory=dict)
    top_logits: list[TopLogit] = Field(default_factory=list, max_length=MAX_VECTOR_ROWS)
    child: str | None = None
    usage: Usage


# The most tokens one rider request may generate. The worker also refuses a
# prompt plus max_tokens that would not fit its context.
MAX_RIDER_TOKENS = 1024
type StopReason = Literal["eog", "max_tokens"]


class RiderRequest(StrictModel):
    model: str = MODEL_ID
    snapshot: SnapshotRef
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(ge=1, le=MAX_RIDER_TOKENS)
    # Per generated step, the top-k head logits (0 disables the trace).
    top_logits: int = Field(ge=0, le=MAX_VECTOR_ROWS, default=0)


class RiderStep(StrictModel):
    token_id: int = Field(ge=0)
    top_logits: list[TopLogit] = Field(max_length=MAX_VECTOR_ROWS)


class RiderTiming(StrictModel):
    restore_ms: float = Field(ge=0.0)
    promotion_ms: float = Field(ge=0.0)
    time_to_first_token_ms: float = Field(ge=0.0)
    decode_ms: float = Field(ge=0.0)


class RiderUsage(StrictModel):
    # Rider is the one /v2 route that generates, so unlike `Usage` its
    # output count is not pinned to zero.
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class RiderResponse(StrictModel):
    model: Literal["local-gemma-unridden-v1"]
    text: str
    token_ids: list[int]
    stop_reason: StopReason
    suffix_tokens: int = Field(ge=1)
    block_tokens: BlockTokens
    reused_prefix_tokens: int = Field(ge=0)
    restored_bytes: int = Field(ge=0)
    timing: RiderTiming
    decode_tokens_per_second: float = Field(ge=0.0)
    steps: list[RiderStep] = Field(default_factory=list)
    usage: RiderUsage


# ----------------------------------------------------------------------------
# Worker (native) messages. Each is validated strictly before it is believed.
# ----------------------------------------------------------------------------


class SnapshotBytes(StrictModel):
    lower_kv: int = Field(ge=0)
    upper_kv: int = Field(ge=0)
    h18: int = Field(ge=0)
    h30: int = Field(ge=0)


class WorkerSnapshotRow(StrictModel):
    snapshot_id: str = Field(min_length=1)
    completed_blocks: Checkpoint
    parent: str | None
    tokens: int = Field(ge=1)
    bytes: SnapshotBytes
    kind: Boundary
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _coverage_matches_depth(self) -> Self:
        blob = self.bytes
        # The upper range never ran on an 18, so its cache and H30 do not exist.
        if self.completed_blocks == 18 and (blob.upper_kv != 0 or blob.h30 != 0):
            raise ValueError("an 18 snapshot cannot hold upper-range state")
        # A 30 snapshot may hold H18: a 30 created without an 18 keeps it, and a
        # saved branch child carries its parent's. Only a 30 created beside its
        # 18 must reference it, which `WorkerCreated` checks.
        return self


class WorkerReadout(StrictModel):
    label_logits: list[float] = Field(min_length=2, max_length=255)
    label_token_ids: list[int] = Field(min_length=2, max_length=255)
    allowed_label_mass: float = Field(ge=0.0, le=1.0)
    full_vocabulary_argmax: FullVocabularyArgmax
    top_logits: list[TopLogit] = Field(default_factory=list, max_length=MAX_VECTOR_ROWS)

    @model_validator(mode="after")
    def _finite_and_aligned(self) -> Self:
        if len(self.label_logits) != len(self.label_token_ids):
            raise ValueError("worker label logit and token counts differ")
        if not all(math.isfinite(value) for value in self.label_logits):
            raise ValueError("worker returned nonfinite label logits")
        if not math.isfinite(self.full_vocabulary_argmax.logit):
            raise ValueError("worker returned nonfinite vocabulary argmax")
        return self


class WorkerCreateTiming(StrictModel):
    lower: float = Field(ge=0.0)
    upper: float = Field(ge=0.0)
    total: float = Field(ge=0.0)


class WorkerPromoteTiming(StrictModel):
    # Promotion runs only the upper range, so it has no lower timing.
    upper: float = Field(ge=0.0)
    total: float = Field(ge=0.0)


class WorkerCreated(StrictModel):
    type: Literal["created"]
    id: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokens: int = Field(ge=1)
    snapshots: list[WorkerSnapshotRow] = Field(min_length=1, max_length=2)
    readout: WorkerReadout | None = None
    block_tokens: BlockTokens
    timing_ms: WorkerCreateTiming
    generated_tokens: Literal[0]

    @model_validator(mode="after")
    def _consistent_set(self) -> Self:
        blocks = [row.completed_blocks for row in self.snapshots]
        if len(set(blocks)) != len(blocks):
            raise ValueError("worker returned duplicate checkpoints")
        by_depth = {row.completed_blocks: row for row in self.snapshots}
        thirty = by_depth.get(30)
        if thirty is not None and 18 in by_depth and thirty.parent is None:
            raise ValueError("a paired 30 snapshot must name its 18 parent")
        if thirty is not None and 18 in by_depth and thirty.bytes.h18 != 0:
            raise ValueError("a paired 30 snapshot must reference H18, not store it")
        if self.readout is not None and set(by_depth) == {PROFILE_SPLIT_BLOCK}:
            raise ValueError("a readout is only produced with a final checkpoint")
        for row in self.snapshots:
            if row.tokens != self.tokens:
                raise ValueError("snapshot token count differs from the prefix")
        return self


class WorkerPromoted(StrictModel):
    type: Literal["promoted"]
    id: str = Field(min_length=1)
    snapshot: WorkerSnapshotRow
    block_tokens: BlockTokens
    timing_ms: WorkerPromoteTiming
    generated_tokens: Literal[0]

    @model_validator(mode="after")
    def _promotion_ran_only_upper(self) -> Self:
        if self.snapshot.completed_blocks != 30:
            raise ValueError("promotion must yield a 30 snapshot")
        if self.snapshot.parent is None:
            raise ValueError("a promoted snapshot must name its 18 parent")
        if self.block_tokens.lower != 0:
            raise ValueError("promotion must not repeat the lower range")
        return self


class WorkerSnapshotBranch(StrictModel):
    parent: str = Field(min_length=1)
    suffix_tokens: int = Field(ge=1)
    block_tokens: BlockTokens
    restore: Literal["resident", "host"]
    restored_bytes: int = Field(ge=0)
    restore_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    child: WorkerSnapshotRow | None = None

    @model_validator(mode="after")
    def _restore_accounting(self) -> Self:
        if self.restore == "resident" and self.restored_bytes != 0:
            raise ValueError("a resident branch restored no bytes")
        if (
            self.child is not None
            and self.child.completed_blocks == PROFILE_SPLIT_BLOCK
        ):
            raise ValueError("a saved decision child must be a final snapshot")
        return self


class WorkerSnapshotQuestionResult(WorkerQuestionResult):
    snapshot: WorkerSnapshotBranch

    @model_validator(mode="after")
    def _branch_accounting(self) -> Self:
        # A branch processes exactly its suffix and reuses the whole parent
        # prefix; the inherited v1 validator already ties those to the prompt.
        if self.snapshot.suffix_tokens != self.processed_tokens:
            raise ValueError("branch suffix differs from processed tokens")
        return self


class WorkerResult(StrictModel):
    type: Literal["result"]
    id: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_tokens: Literal[0]
    callbacks_enabled: Literal[False]
    execution_mode: Literal["split18-30", "full"]
    questions: list[WorkerSnapshotQuestionResult] = Field(min_length=1, max_length=32)


class WorkerState(StrictModel):
    # The plain-prompt branch carries the same per-branch accounting as an
    # evaluate question, flattened to the top level, plus its artifacts.
    type: Literal["state"]
    id: str = Field(min_length=1)
    parent: str = Field(min_length=1)
    suffix_tokens: int = Field(ge=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_tokens: int = Field(ge=1)
    block_tokens: BlockTokens
    restore: Literal["resident", "host"]
    restored_bytes: int = Field(ge=0)
    restore_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    timing_ms: float = Field(ge=0.0)
    vectors: dict[ExportKind, VectorArtifact] = Field(default_factory=dict)
    top_logits: list[TopLogit] = Field(default_factory=list, max_length=MAX_VECTOR_ROWS)
    child: WorkerSnapshotRow | None = None
    generated_tokens: Literal[0]

    @model_validator(mode="after")
    def _restore_accounting(self) -> Self:
        if self.restore == "resident" and self.restored_bytes != 0:
            raise ValueError("a resident branch restored no bytes")
        if (
            self.child is not None
            and self.child.completed_blocks == PROFILE_SPLIT_BLOCK
        ):
            raise ValueError("a saved evaluation child must be a final snapshot")
        return self


class WorkerRideTiming(StrictModel):
    restore: float = Field(ge=0.0)
    time_to_first_token: float = Field(ge=0.0)
    decode: float = Field(ge=0.0)
    total: float = Field(ge=0.0)


class WorkerRide(StrictModel):
    # Greedy output continued from a 30 snapshot. `split_prefill` and
    # `reference` are qualification baselines a --reference worker also runs.
    type: Literal["ride_result"]
    id: str = Field(min_length=1)
    mode: Literal["snapshot", "split_prefill", "reference"]
    execution_mode: Literal["split18-30", "full", "stock30"]
    text: str
    token_ids: list[int]
    stop_reason: StopReason
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_tokens: int = Field(ge=1)
    reused_tokens: int = Field(ge=0)
    prefilled_tokens: int = Field(ge=1)
    generated_tokens: int = Field(ge=0)
    block_tokens: dict[Literal["lower", "upper", "stock"], int]
    restore: Literal["resident", "host"] | None
    restored_bytes: int = Field(ge=0)
    timing_ms: WorkerRideTiming
    decode_tokens_per_second: float = Field(ge=0.0)
    steps: list[RiderStep]

    @model_validator(mode="after")
    def _accounting(self) -> Self:
        if self.generated_tokens != len(self.token_ids):
            raise ValueError("generated token count and ids differ")
        if self.prompt_tokens != self.reused_tokens + self.prefilled_tokens:
            raise ValueError("prompt tokens are not reused plus prefilled")
        if (self.mode == "snapshot") != (self.restore is not None):
            raise ValueError("only a snapshot continuation reports a restore")
        if self.restore == "resident" and self.restored_bytes != 0:
            raise ValueError("a resident branch restored no bytes")
        if (self.mode == "reference") != (self.execution_mode == "stock30"):
            raise ValueError("execution mode does not match the ride mode")
        return self


class WorkerHolds(StrictModel):
    h18: bool
    h30: bool
    last_normalized: bool
    upper_kv: bool


class WorkerInspect(WorkerSnapshotRow):
    # `inspect` returns the snapshot row (type "snapshot") with its residency,
    # bounded token ids, and which tensors it holds.
    type: Literal["snapshot"]
    id: str = Field(min_length=1)
    token_ids: list[int] = Field(max_length=1 << 20)
    resident: bool
    holds: WorkerHolds


class WorkerTensor(StrictModel):
    dtype: Literal["f32"]
    byte_order: Literal["little"]
    shape: list[int] = Field(min_length=1, max_length=2)
    representation: Representation
    base64: str

    @model_validator(mode="after")
    def _decodes(self) -> Self:
        if len(self.shape) == 2 and self.shape[0] > MAX_VECTOR_ROWS:
            raise ValueError("vectors response exceeds the row bound")
        _decode_f32(self.shape, self.base64)
        return self


class WorkerVectors(StrictModel):
    type: Literal["vectors"]
    id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    which: Literal["h18", "h30", "last_normalized"]
    # `rows` is the tensor's total row count; the response returns the slice
    # [row_begin, row_end) of it.
    rows: int = Field(ge=1)
    row_begin: int = Field(ge=0)
    row_end: int = Field(ge=1)
    tensor: WorkerTensor

    @model_validator(mode="after")
    def _slice_within_bounds(self) -> Self:
        if not 0 <= self.row_begin < self.row_end <= self.rows:
            raise ValueError("vectors slice is out of bounds")
        returned = self.row_end - self.row_begin
        if returned > MAX_VECTOR_ROWS:
            raise ValueError("vectors slice exceeds the row bound")
        # A single row is a 1-D vector; otherwise the tensor is [rows, width].
        shape_rows = self.tensor.shape[0] if len(self.tensor.shape) == 2 else 1
        if shape_rows != returned:
            raise ValueError("vectors tensor rows differ from the slice")
        # `h30` is the residual after the final block: block 30 in the split
        # profile. The service checks the tag against the running profile.
        expected = {
            "h18": {"raw_residual_after_block_18"},
            "h30": {"raw_residual_after_block_30", "raw_residual_after_final_block"},
            "last_normalized": {"post_final_norm_head_input"},
        }[self.which]
        if self.tensor.representation not in expected:
            raise ValueError("vectors representation does not match the tensor")
        return self


class WorkerFileEntry(StrictModel):
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkerSaved(StrictModel):
    type: Literal["saved"]
    id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    files: dict[str, WorkerFileEntry] = Field(min_length=1)


class WorkerLoaded(WorkerSnapshotRow):
    # `load` re-registers a snapshot and echoes its row (type "loaded").
    type: Literal["loaded"]
    id: str = Field(min_length=1)


class WorkerDropped(StrictModel):
    type: Literal["dropped"]
    id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    freed_bytes: int = Field(ge=0)


class WorkerHello(StrictModel):
    type: Literal["hello"]
    protocol: Literal["unridden-snapshot-v1"]
    profile: ProfileName
    model_id: Literal["local-gemma-unridden-v1"]
    model_name: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels: list[str] = Field(min_length=2, max_length=255)
    label_token_ids: list[int] = Field(min_length=2, max_length=255)
    context_size: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    ubatch_size: int = Field(ge=1)
    threads: int = Field(ge=1)
    n_layer: int = Field(ge=1, le=MAX_BLOCKS)
    n_embd: int = Field(ge=1)
    split_block: Literal[18] | None
    reference_context: bool
    context_prompt_version: Literal["unridden-gemma-context-v1"]
    generated_tokens: Literal[0]
    # A worker built with the `ride` command; older workers omit it.
    rider_mode: bool = False
    callbacks_enabled: Literal[False]

    @model_validator(mode="after")
    def _label_mapping(self) -> Self:
        if len(self.labels) != len(self.label_token_ids):
            raise ValueError("label and token counts differ")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if len(set(self.label_token_ids)) != len(self.label_token_ids):
            raise ValueError("label token ids must be unique")
        return self

    @model_validator(mode="after")
    def _profile_shape(self) -> Self:
        if self.profile == SNAPSHOT_PROFILE:
            if (
                self.n_layer != PROFILE_N_LAYER
                or self.split_block != PROFILE_SPLIT_BLOCK
            ):
                raise ValueError("split18-30-v1 needs 30 blocks split at 18")
        elif self.split_block is not None:
            raise ValueError("full-v1 has no split block")
        if self.profile == FULL_PROFILE and self.reference_context:
            raise ValueError("full-v1 has no reference context")
        return self

    @property
    def final_block(self) -> int:
        return self.n_layer

    @property
    def checkpoints(self) -> list[int]:
        """The block boundaries this profile can freeze at, low to high."""
        if self.split_block is None:
            return [self.n_layer]
        return [self.split_block, self.n_layer]

    @property
    def execution_mode(self) -> Literal["split18-30", "full"]:
        return "full" if self.profile == FULL_PROFILE else "split18-30"

    @property
    def final_representation(self) -> Representation:
        if self.profile == FULL_PROFILE:
            return "raw_residual_after_final_block"
        return "raw_residual_after_block_30"

    @property
    def layer_map(self) -> dict[str, list[int]]:
        """The layer ranges a snapshot blob covers, recorded in its manifest."""
        if self.split_block is None:
            return {"lower": [0, self.n_layer]}
        return {
            "lower": [0, self.split_block],
            "upper": [self.split_block, self.n_layer],
        }


class WorkerErrorMessage(StrictModel):
    # Kept permissive on unknown codes so a new worker code is a typed protocol
    # error, not a validation crash: the backend maps known codes and treats
    # the rest as an ambiguous failure.
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["error"]
    id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    reason: str | None = None
    message: str | None = None


__all__ = [
    "CONTEXT_PROMPT_VERSION",
    "FULL_PROFILE",
    "MAX_BLOCKS",
    "MAX_TTL_SECONDS",
    "MAX_VECTOR_ROWS",
    "PROFILE_N_EMBD",
    "PROFILE_N_LAYER",
    "PROFILE_SPLIT_BLOCK",
    "SNAPSHOT_PROFILE",
    "SNAPSHOT_PROTOCOL",
    "BlockTokens",
    "Boundary",
    "Capability",
    "Checkpoint",
    "ContextInput",
    "CreateUsage",
    "DecisionInput",
    "ExportKind",
    "Persistence",
    "Promotion",
    "ProfileName",
    "PromptInput",
    "Readout",
    "Relationship",
    "Representation",
    "SnapshotBytes",
    "SnapshotCreateRequest",
    "SnapshotCreateResponse",
    "SnapshotInput",
    "SnapshotMetadata",
    "SnapshotRef",
    "SnapshotRow",
    "SnapshotTiming",
    "SnapshotUsage",
    "StateEvaluationRequest",
    "StateEvaluationResponse",
    "StateReadout",
    "TopLogit",
    "V2DecisionRequest",
    "V2DecisionResponse",
    "VectorArtifact",
    "WorkerCreateTiming",
    "WorkerCreated",
    "WorkerDropped",
    "WorkerErrorMessage",
    "WorkerFileEntry",
    "WorkerHello",
    "WorkerHolds",
    "WorkerInspect",
    "WorkerLoaded",
    "WorkerPromoteTiming",
    "WorkerPromoted",
    "WorkerReadout",
    "WorkerResult",
    "WorkerSaved",
    "WorkerSnapshotBranch",
    "WorkerSnapshotQuestionResult",
    "WorkerSnapshotRow",
    "WorkerState",
    "WorkerTensor",
    "WorkerVectors",
]
