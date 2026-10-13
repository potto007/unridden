"""CPU tests for host-side H18 steering (priority matrix item 3).

The native worker applies `H18 += alpha * v` immediately before the upper
range. These tests pin the strict wire shape, the contrastive-direction
helper, and the pure-Python mirror of the worker's steering arithmetic; the
worker-utils C++ test pins the same arithmetic natively.
"""

from __future__ import annotations

import base64
import struct
from typing import Any

import pytest
from pydantic import ValidationError

from unridden.api.snapshots.schema import (
    H18Steering,
    StateEvaluationRequest,
    SteeringVector,
    V2DecisionRequest,
    VectorArtifact,
)
from unridden.api.snapshots.steering import (
    apply_steering,
    contrastive_direction,
    decode_rows,
    make_steering,
)


def _f32(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def _steering(
    values: list[float], *, alpha: float = 0.5, scope: str = "all"
) -> dict[str, Any]:
    return {
        "alpha": alpha,
        "scope": scope,
        "vector": {"dtype": "f32", "shape": [len(values)], "base64": _f32(values)},
    }


def test_steering_round_trips_values() -> None:
    steering = H18Steering.model_validate(_steering([1.0, -2.0, 3.5]))
    assert steering.alpha == 0.5
    assert steering.scope == "all"
    assert steering.vector.width == 3
    assert steering.vector.values() == pytest.approx([1.0, -2.0, 3.5])


def test_steering_vector_from_values() -> None:
    vector = SteeringVector.from_values([0.25, -0.75])
    assert vector.shape == [2]
    assert vector.values() == pytest.approx([0.25, -0.75])


@pytest.mark.parametrize(
    "payload",
    [
        _steering([1.0], scope="middle"),
        _steering([0.0, 0.0]),
        {**_steering([1.0]), "extra": 1},
        {
            "alpha": 1.0,
            "scope": "all",
            "vector": {"dtype": "f32", "shape": [2, 2], "base64": _f32([1, 2, 3, 4])},
        },
        {
            "alpha": 1.0,
            "scope": "all",
            "vector": {"dtype": "f32", "shape": [2], "base64": "not base64!"},
        },
        {
            "alpha": 1.0,
            "scope": "all",
            "vector": {"dtype": "f32", "shape": [3], "base64": _f32([1.0, 2.0])},
        },
    ],
)
def test_steering_rejects_bad_shapes(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        H18Steering.model_validate(payload)


def test_steering_rejects_nonfinite() -> None:
    with pytest.raises(ValidationError):
        H18Steering.model_validate(_steering([float("inf"), 1.0]))
    with pytest.raises(ValidationError):
        H18Steering.model_validate(_steering([1.0], alpha=float("nan")))


def test_decision_and_state_eval_requests_carry_optional_steering() -> None:
    plain = V2DecisionRequest.model_validate(
        {
            "snapshot": {"id": "s1", "relationship": "followup"},
            "questions": {"q": {"type": "noul", "instructions": "ok?"}},
            "readout": {"completed_blocks": 30},
        }
    )
    assert plain.steering is None
    steered = V2DecisionRequest.model_validate(
        {
            "snapshot": {"id": "s1", "relationship": "followup"},
            "questions": {"q": {"type": "noul", "instructions": "ok?"}},
            "readout": {"completed_blocks": 30},
            "steering": _steering([1.0, 2.0], scope="last"),
        }
    )
    assert steered.steering is not None
    assert steered.steering.scope == "last"
    state = StateEvaluationRequest.model_validate(
        {
            "snapshot": {"id": "s1", "relationship": "followup"},
            "prompt": "focus",
            "readout": {"completed_blocks": 30},
            "steering": _steering([1.0]),
        }
    )
    assert state.steering is not None


def test_contrastive_direction_is_mean_difference() -> None:
    direction = contrastive_direction(
        focused=[[3.0, 1.0], [1.0, 3.0]],
        neutral=[[0.0, 1.0], [2.0, -1.0]],
    )
    assert direction == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize(
    "focused,neutral",
    [
        ([], [[1.0]]),
        ([[1.0]], []),
        ([[1.0, 2.0]], [[1.0]]),
        ([[1.0], [1.0, 2.0]], [[1.0]]),
        ([[float("inf")]], [[0.0]]),
    ],
)
def test_contrastive_direction_rejects_bad_rows(
    focused: list[Any], neutral: list[Any]
) -> None:
    with pytest.raises(ValueError):
        contrastive_direction(focused=focused, neutral=neutral)


def test_decode_rows_from_vector_artifact() -> None:
    artifact = VectorArtifact(
        dtype="f32",
        shape=[2, 3],
        representation="raw_residual_after_block_18",
        base64=_f32([1, 2, 3, 4, 5, 6]),
    )
    assert decode_rows(artifact) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    single = VectorArtifact(
        dtype="f32",
        shape=[3],
        representation="raw_residual_after_block_18",
        base64=_f32([7, 8, 9]),
    )
    assert decode_rows(single) == [[7.0, 8.0, 9.0]]


def test_make_steering_builds_validated_spec() -> None:
    steering = make_steering([2.0, -1.0], alpha=-1.5, scope="last")
    assert steering.alpha == -1.5
    assert steering.vector.values() == pytest.approx([2.0, -1.0])


def test_apply_steering_all_positions() -> None:
    rows = [[1.0, 2.0], [3.0, 4.0]]
    steered = apply_steering(rows, [10.0, 100.0], alpha=0.5, scope="all")
    assert steered == [[6.0, 52.0], [8.0, 54.0]]
    # The input (mirror of the parent snapshot's H18) is never mutated.
    assert rows == [[1.0, 2.0], [3.0, 4.0]]


def test_apply_steering_last_position_only() -> None:
    rows = [[1.0, 2.0], [3.0, 4.0]]
    steered = apply_steering(rows, [10.0, 100.0], alpha=1.0, scope="last")
    assert steered == [[1.0, 2.0], [13.0, 104.0]]
    assert rows == [[1.0, 2.0], [3.0, 4.0]]


def test_apply_steering_zero_alpha_is_identity() -> None:
    rows = [[1.0, 2.0], [3.0, 4.0]]
    assert apply_steering(rows, [5.0, 6.0], alpha=0.0, scope="all") == rows


@pytest.mark.parametrize(
    "rows,direction,alpha",
    [
        ([], [1.0], 1.0),
        ([[1.0, 2.0]], [1.0], 1.0),
        ([[1.0], [1.0, 2.0]], [1.0], 1.0),
        ([[1.0]], [float("nan")], 1.0),
        ([[1.0]], [1.0], float("inf")),
        ([[1e38]], [1e38], 10.0),
    ],
)
def test_apply_steering_rejects_bad_inputs(
    rows: list[Any], direction: list[Any], alpha: float
) -> None:
    with pytest.raises(ValueError):
        apply_steering(rows, direction, alpha=alpha, scope="all")
