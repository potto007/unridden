"""Strictness tests for the /v2 wire and worker schemas."""

from __future__ import annotations

import base64
import math
import struct

import pytest
from pydantic import ValidationError

from unridden.api.snapshots.schema import (
    SnapshotCreateRequest,
    StateReadout,
    V2DecisionRequest,
    VectorArtifact,
    WorkerCreated,
    WorkerHello,
    WorkerPromoted,
    WorkerReadout,
    WorkerSnapshotQuestionResult,
    WorkerSnapshotRow,
)


def _f32(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def test_create_request_accepts_each_input_kind() -> None:
    context = SnapshotCreateRequest.model_validate(
        {
            "input": {"kind": "context", "state": "s"},
            "checkpoints": [18, 30],
            "ttl_seconds": 60,
        }
    )
    assert context.input.kind == "context"
    assert context.checkpoints == [18, 30]
    decision = SnapshotCreateRequest.model_validate(
        {
            "input": {
                "kind": "decision",
                "state": "s",
                "question": {"type": "noul", "instructions": "urgent?"},
            },
            "checkpoints": [30],
            "ttl_seconds": 60,
        }
    )
    assert decision.input.kind == "decision"
    prompt = SnapshotCreateRequest.model_validate(
        {
            "input": {"kind": "prompt", "state": "s", "prompt": "why"},
            "checkpoints": [18],
            "ttl_seconds": 1,
        }
    )
    assert prompt.input.kind == "prompt"


def test_create_request_sorts_and_rejects_bad_checkpoints() -> None:
    request = SnapshotCreateRequest.model_validate(
        {
            "input": {"kind": "context", "state": "s"},
            "checkpoints": [30, 18],
            "ttl_seconds": 60,
        }
    )
    assert request.checkpoints == [18, 30]
    with pytest.raises(ValidationError, match="unique"):
        SnapshotCreateRequest.model_validate(
            {
                "input": {"kind": "context", "state": "s"},
                "checkpoints": [18, 18],
                "ttl_seconds": 60,
            }
        )
    # Which blocks are offered is the running profile's call (ADR 0009); the
    # schema bounds only the shape.
    for bad in ([], [0], [257], [18, 30, 30]):
        with pytest.raises(ValidationError):
            SnapshotCreateRequest.model_validate(
                {
                    "input": {"kind": "context", "state": "s"},
                    "checkpoints": bad,
                    "ttl_seconds": 60,
                }
            )


def test_create_request_bounds_ttl_and_forbids_extra() -> None:
    for ttl in (0, 86_401):
        with pytest.raises(ValidationError):
            SnapshotCreateRequest.model_validate(
                {
                    "input": {"kind": "context", "state": "s"},
                    "checkpoints": [18],
                    "ttl_seconds": ttl,
                }
            )
    with pytest.raises(ValidationError):
        SnapshotCreateRequest.model_validate(
            {
                "input": {"kind": "context", "state": "s"},
                "checkpoints": [18],
                "ttl_seconds": 5,
                "extra": 1,
            }
        )
    with pytest.raises(ValidationError, match="finite"):
        SnapshotCreateRequest.model_validate(
            {
                "input": {"kind": "context", "state": {"x": math.inf}},
                "checkpoints": [18],
                "ttl_seconds": 5,
            }
        )


def test_v2_decision_request_validates_questions() -> None:
    request = V2DecisionRequest.model_validate(
        {
            "snapshot": {"id": "snap_a", "relationship": "followup"},
            "questions": {"q": {"type": "noul", "instructions": "x"}},
            "readout": {"completed_blocks": 30},
        }
    )
    assert request.snapshot.relationship == "followup"
    with pytest.raises(ValidationError):
        V2DecisionRequest.model_validate(
            {
                "snapshot": {"id": "snap_a", "relationship": "sideways"},
                "questions": {"q": {"type": "noul", "instructions": "x"}},
                "readout": {"completed_blocks": 30},
            }
        )
    with pytest.raises(ValidationError, match="nonempty"):
        V2DecisionRequest.model_validate(
            {
                "snapshot": {"id": "snap_a", "relationship": "followup"},
                "questions": {"": {"type": "noul", "instructions": "x"}},
                "readout": {"completed_blocks": 30},
            }
        )


def test_state_readout_ties_top_logits_to_its_export() -> None:
    ok = StateReadout.model_validate(
        {"completed_blocks": 30, "export": ["top_logits"], "top_logits": 5}
    )
    assert ok.top_logits == 5
    with pytest.raises(ValidationError, match="positive count"):
        StateReadout.model_validate({"completed_blocks": 30, "export": ["top_logits"]})
    with pytest.raises(ValidationError, match="without requesting"):
        StateReadout.model_validate({"completed_blocks": 30, "top_logits": 5})
    with pytest.raises(ValidationError, match="unique"):
        StateReadout.model_validate(
            {"completed_blocks": 30, "export": ["last_residual", "last_residual"]}
        )


def test_vector_artifact_checks_bytes_shape_and_finiteness() -> None:
    good = VectorArtifact.model_validate(
        {
            "dtype": "f32",
            "shape": [4],
            "representation": "raw_residual_after_block_30",
            "base64": _f32([1.0, 2.0, 3.0, 4.0]),
        }
    )
    assert good.shape == [4]
    with pytest.raises(ValidationError, match="byte length"):
        VectorArtifact.model_validate(
            {
                "dtype": "f32",
                "shape": [5],
                "representation": "raw_residual_after_block_30",
                "base64": _f32([1.0, 2.0, 3.0, 4.0]),
            }
        )
    with pytest.raises(ValidationError, match="nonfinite"):
        VectorArtifact.model_validate(
            {
                "dtype": "f32",
                "shape": [1],
                "representation": "post_final_norm_head_input",
                "base64": _f32([math.inf]),
            }
        )
    with pytest.raises(ValidationError, match="row bound"):
        VectorArtifact.model_validate(
            {
                "dtype": "f32",
                "shape": [65, 1],
                "representation": "raw_residual_after_block_30",
                "base64": _f32([0.0] * 65),
            }
        )


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "snapshot_id": "snap_a",
        "completed_blocks": 18,
        "parent": None,
        "tokens": 10,
        "bytes": {"lower_kv": 10, "upper_kv": 0, "h18": 10, "h30": 0},
        "kind": "context",
        "prompt_sha256": "a" * 64,
    }
    row.update(overrides)
    return row


def test_worker_snapshot_row_rejects_impossible_coverage() -> None:
    WorkerSnapshotRow.model_validate(_row())
    with pytest.raises(ValidationError, match="upper-range"):
        WorkerSnapshotRow.model_validate(
            _row(bytes={"lower_kv": 10, "upper_kv": 5, "h18": 10, "h30": 0})
        )
    # A 30 created on its own keeps H18, so the row alone cannot forbid it.
    WorkerSnapshotRow.model_validate(
        _row(
            completed_blocks=30,
            bytes={"lower_kv": 10, "upper_kv": 10, "h18": 10, "h30": 10},
        )
    )


def _readout(**overrides: object) -> dict[str, object]:
    row = {
        "label_logits": [0.0, 1.0],
        "label_token_ids": [11, 12],
        "allowed_label_mass": 0.5,
        "full_vocabulary_argmax": {"token_id": 12, "logit": 3.0},
    }
    row.update(overrides)
    return row


def test_worker_readout_rejects_misaligned_or_nonfinite_logits() -> None:
    WorkerReadout.model_validate(_readout())
    with pytest.raises(ValidationError, match="counts differ"):
        WorkerReadout.model_validate(_readout(label_token_ids=[11, 12, 13]))
    with pytest.raises(ValidationError, match="finite"):
        WorkerReadout.model_validate(_readout(label_logits=[0.0, math.nan]))


def test_worker_created_enforces_a_consistent_snapshot_set() -> None:
    both = WorkerCreated.model_validate(
        {
            "type": "created",
            "id": "c1",
            "prompt_sha256": "a" * 64,
            "tokens": 10,
            "snapshots": [
                _row(),
                _row(
                    snapshot_id="snap_b",
                    completed_blocks=30,
                    parent="snap_a",
                    bytes={"lower_kv": 0, "upper_kv": 10, "h18": 0, "h30": 10},
                ),
            ],
            "readout": _readout(),
            "block_tokens": {"lower": 10, "upper": 10},
            "timing_ms": {"lower": 1.0, "upper": 1.0, "total": 3.0},
            "generated_tokens": 0,
        }
    )
    assert {row.completed_blocks for row in both.snapshots} == {18, 30}
    # A 30 created beside its 18 references H18 instead of storing a copy.
    with pytest.raises(ValidationError, match="reference H18"):
        WorkerCreated.model_validate(
            {
                "type": "created",
                "id": "c1",
                "prompt_sha256": "a" * 64,
                "tokens": 10,
                "snapshots": [
                    _row(),
                    _row(
                        snapshot_id="snap_b",
                        completed_blocks=30,
                        parent="snap_a",
                        bytes={"lower_kv": 0, "upper_kv": 10, "h18": 10, "h30": 10},
                    ),
                ],
                "block_tokens": {"lower": 10, "upper": 10},
                "timing_ms": {"lower": 1.0, "upper": 1.0, "total": 3.0},
                "generated_tokens": 0,
            }
        )
    # A 30 created alone keeps H18 and has no parent (the worker's behavior).
    alone = WorkerCreated.model_validate(
        {
            "type": "created",
            "id": "c2",
            "prompt_sha256": "a" * 64,
            "tokens": 10,
            "snapshots": [
                _row(
                    snapshot_id="snap_c",
                    completed_blocks=30,
                    bytes={"lower_kv": 10, "upper_kv": 10, "h18": 10, "h30": 10},
                )
            ],
            "block_tokens": {"lower": 10, "upper": 10},
            "timing_ms": {"lower": 1.0, "upper": 1.0, "total": 3.0},
            "generated_tokens": 0,
        }
    )
    assert alone.snapshots[0].bytes.h18 == 10
    # A readout without a 30 checkpoint is impossible.
    with pytest.raises(ValidationError, match="readout is only produced"):
        WorkerCreated.model_validate(
            {
                "type": "created",
                "id": "c1",
                "prompt_sha256": "a" * 64,
                "tokens": 10,
                "snapshots": [_row()],
                "readout": _readout(),
                "block_tokens": {"lower": 10, "upper": 0},
                "timing_ms": {"lower": 1.0, "upper": 0.0, "total": 2.0},
                "generated_tokens": 0,
            }
        )


def test_worker_promoted_ran_only_the_upper_range() -> None:
    good = WorkerPromoted.model_validate(
        {
            "type": "promoted",
            "id": "p1",
            "snapshot": _row(
                snapshot_id="snap_b",
                completed_blocks=30,
                parent="snap_a",
                bytes={"lower_kv": 0, "upper_kv": 10, "h18": 0, "h30": 10},
            ),
            "block_tokens": {"lower": 0, "upper": 10},
            "timing_ms": {"upper": 1.0, "total": 2.0},
            "generated_tokens": 0,
        }
    )
    assert good.snapshot.completed_blocks == 30
    with pytest.raises(ValidationError, match="repeat the lower"):
        WorkerPromoted.model_validate(
            {
                "type": "promoted",
                "id": "p1",
                "snapshot": _row(
                    snapshot_id="snap_b",
                    completed_blocks=30,
                    parent="snap_a",
                    bytes={"lower_kv": 0, "upper_kv": 10, "h18": 0, "h30": 10},
                ),
                "block_tokens": {"lower": 5, "upper": 10},
                "timing_ms": {"upper": 1.0, "total": 2.0},
                "generated_tokens": 0,
            }
        )


def _branch_question(**overrides: object) -> dict[str, object]:
    row = {
        "id": "q",
        "label_logits": [0.0, 1.0],
        "label_token_ids": [11, 12],
        "allowed_label_mass": 0.5,
        "full_vocabulary_argmax": {"token_id": 12, "logit": 3.0},
        "prompt_sha256": "a" * 64,
        "prompt_tokens": 30,
        "processed_tokens": 6,
        "reused_tokens": 24,
        "cache_cleared": False,
        "evaluation_mode": "sequential",
        "batch_sequences": 1,
        "timing_ms": 1.0,
        "snapshot": {
            "parent": "snap_a",
            "suffix_tokens": 6,
            "block_tokens": {"lower": 6, "upper": 6},
            "restore": "resident",
            "restored_bytes": 0,
            "restore_ms": 0.0,
            "inference_ms": 1.0,
            "child": None,
        },
    }
    row.update(overrides)
    return row


def test_branch_result_ties_suffix_to_processed_and_prompt() -> None:
    WorkerSnapshotQuestionResult.model_validate(_branch_question())
    branch: dict[str, object] = {
        "parent": "snap_a",
        "suffix_tokens": 5,
        "block_tokens": {"lower": 6, "upper": 6},
        "restore": "resident",
        "restored_bytes": 0,
        "restore_ms": 0.0,
        "inference_ms": 1.0,
        "child": None,
    }
    with pytest.raises(ValidationError, match="branch suffix"):
        WorkerSnapshotQuestionResult.model_validate(_branch_question(snapshot=branch))
    # The inherited v1 accounting still applies.
    with pytest.raises(ValidationError, match="cover the prompt"):
        WorkerSnapshotQuestionResult.model_validate(_branch_question(prompt_tokens=99))


def _hello(**overrides: object) -> dict[str, object]:
    row = {
        "type": "hello",
        "protocol": "unridden-snapshot-v1",
        "profile": "split18-30-v1",
        "model_id": "local-gemma-unridden-v1",
        "model_name": "fixture",
        "model_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "labels": ["A", "B"],
        "label_token_ids": [11, 12],
        "context_size": 2048,
        "batch_size": 256,
        "ubatch_size": 256,
        "threads": 8,
        "n_layer": 30,
        "n_embd": 2816,
        "split_block": 18,
        "reference_context": False,
        "context_prompt_version": "unridden-gemma-context-v1",
        "generated_tokens": 0,
        "callbacks_enabled": False,
    }
    row.update(overrides)
    return row


def test_hello_rejects_wrong_layout_or_label_mapping() -> None:
    WorkerHello.model_validate(_hello())
    with pytest.raises(ValidationError):
        WorkerHello.model_validate(_hello(n_layer=28))
    with pytest.raises(ValidationError):
        WorkerHello.model_validate(_hello(split_block=16))
    with pytest.raises(ValidationError, match="counts differ"):
        WorkerHello.model_validate(_hello(label_token_ids=[11, 12, 13]))
    with pytest.raises(ValidationError, match="labels must be unique"):
        WorkerHello.model_validate(_hello(labels=["A", "A"]))
