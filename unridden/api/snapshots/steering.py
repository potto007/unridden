"""Contrastive H18 steering directions and their pure-Python arithmetic.

The native worker owns the real intervention - `H18 += alpha * v` on the rows
it is about to feed the upper blocks (ADR 0011). This module is the caller's
side of that: turning `h18` rows exported through the snapshot `vectors`
endpoint into a validated `H18Steering`, plus a reference implementation of
the worker's arithmetic (`unridden::apply_steering` in `worker-utils.h`) so
the semantics are testable without a model.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from unridden.api.snapshots.schema import (
    H18Steering,
    SteeringScope,
    SteeringVector,
    VectorArtifact,
    WorkerTensor,
    _decode_f32,
)


def decode_rows(artifact: VectorArtifact | WorkerTensor) -> list[list[float]]:
    """The rows of an exported vector artifact, `[width]` read as one row."""
    values = _decode_f32(artifact.shape, artifact.base64)
    width = artifact.shape[-1]
    return [values[start : start + width] for start in range(0, len(values), width)]


def _mean(rows: Sequence[Sequence[float]], width: int) -> list[float]:
    totals = [0.0] * width
    for row in rows:
        for index, value in enumerate(row):
            totals[index] += value
    return [total / len(rows) for total in totals]


def _checked_width(rows: Sequence[Sequence[float]], name: str) -> int:
    if not rows:
        raise ValueError(f"{name} rows must not be empty")
    width = len(rows[0])
    if width < 1:
        raise ValueError(f"{name} rows must not be empty vectors")
    for row in rows:
        if len(row) != width:
            raise ValueError(f"{name} rows must all have the same width")
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{name} rows must be finite")
    return width


def contrastive_direction(
    *,
    focused: Sequence[Sequence[float]],
    neutral: Sequence[Sequence[float]],
) -> list[float]:
    """`mean(focused) - mean(neutral)` over H18 rows of one width.

    Rows are typically the last-position H18 row of snapshots rendered with
    and without a focusing instruction. The result is deliberately not
    normalized: `alpha` carries the experiment's scale, so directions from
    different extraction sets stay comparable in magnitude.
    """
    width = _checked_width(focused, "focused")
    if _checked_width(neutral, "neutral") != width:
        raise ValueError("focused and neutral rows must have the same width")
    focused_mean = _mean(focused, width)
    neutral_mean = _mean(neutral, width)
    direction = [a - b for a, b in zip(focused_mean, neutral_mean, strict=True)]
    if not all(math.isfinite(value) for value in direction):
        raise ValueError("contrastive direction must be finite")
    return direction


def make_steering(
    direction: list[float], *, alpha: float, scope: SteeringScope = "all"
) -> H18Steering:
    return H18Steering(
        alpha=alpha, scope=scope, vector=SteeringVector.from_values(direction)
    )


def apply_steering(
    rows: Sequence[Sequence[float]],
    direction: Sequence[float],
    *,
    alpha: float,
    scope: SteeringScope,
) -> list[list[float]]:
    """A steered copy of `rows`; the input rows are never mutated.

    Mirrors `unridden::apply_steering` exactly: `all` adds `alpha * direction`
    to every row, `last` to the final row only, and any nonfinite input or
    result (an alpha that overflows f32-range residuals) is refused rather
    than propagated into the upper blocks.
    """
    width = _checked_width(rows, "rows")
    if len(direction) != width:
        raise ValueError("steering direction width differs from the rows")
    if not all(math.isfinite(value) for value in direction):
        raise ValueError("steering direction must be finite")
    if not math.isfinite(alpha):
        raise ValueError("steering alpha must be finite")
    steered = [list(row) for row in rows]
    targets = steered if scope == "all" else steered[-1:]
    for row in targets:
        for index in range(width):
            # The worker's rows are f32, so the sum is rounded through f32
            # here too; an alpha that overflows it is refused, not widened.
            try:
                value = struct.unpack(
                    "<f", struct.pack("<f", row[index] + alpha * direction[index])
                )[0]
            except OverflowError as error:
                raise ValueError("steering produced a nonfinite residual") from error
            if not math.isfinite(value):
                raise ValueError("steering produced a nonfinite residual")
            row[index] = value
    return steered


__all__ = [
    "apply_steering",
    "contrastive_direction",
    "decode_rows",
    "make_steering",
]
