"""Task-general Block-18 Tuned Lens projection and zero-recompute suffix handoff.

Implements the affine residual translator:
    h_hat_30 = W @ h_18 + b,  with W = I + delta_W (ridge shrinkage toward identity)
followed by the model's exact final RMSNorm (`output_norm`), single-token label
unembedding rows (`output.weight[label_token_ids]`), and Gemma 4 final logit
softcapping (`f_final_logit_softcapping = 30.0`).

At inference time on `split18-30-v1`, a block-18 readout evaluates the Tuned Lens
at the final question suffix token and gates on `logit_margin` and
`entropy_concentration`:
- If the gate passes, the worker returns the block-18 Tuned Lens readout with
  `upper_tokens == 0` (`readout_source = "tuned_lens_18"`).
- If the gate defers, the worker feeds the already-captured `H18` suffix buffer
  directly into `upper` (blocks 18..29) without recomputing blocks 0..17
  (`readout_source = "cascade_fallback_30"`).
"""

from __future__ import annotations

import base64
import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, BinaryIO, Literal

from pydantic import Field, field_validator, model_validator

from unridden.api.mapping import entropy_concentration, logit_margin, stable_softmax
from unridden.api.schema import StrictModel, _reject_nonfinite

TUNED_LENS_SCHEMA_VERSION: Literal["tuned-lens-v1"] = "tuned-lens-v1"
SPLIT_PROFILE_ID: Literal["split18-30-v1"] = "split18-30-v1"
DEFAULT_RMS_NORM_EPS = 1e-6
DEFAULT_FINAL_LOGIT_SOFTCAPPING = 30.0
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ReadoutSource = Literal["block_30", "tuned_lens_18", "cascade_fallback_30"]
Matrix = list[list[float]]
Vector = list[float]


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cholesky_solve(gram: Matrix, rhs: Matrix) -> Matrix:
    """Solve `gram @ X = rhs` for SPD `gram` (n x n) and `rhs` (n x m) via Cholesky."""
    n = len(gram)
    if n == 0:
        raise ValueError("gram matrix must be non-empty")
    m = len(rhs[0]) if rhs else 0
    lower: Matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        row_i = gram[i]
        lower_i = lower[i]
        for j in range(i + 1):
            lower_j = lower[j]
            s = row_i[j]
            for k in range(j):
                s -= lower_i[k] * lower_j[k]
            if i == j:
                lower_i[j] = math.sqrt(max(s, 1e-12))
            else:
                lower_i[j] = s / lower_j[j]

    # Forward substitution: L @ Z = rhs
    z: Matrix = [[0.0] * m for _ in range(n)]
    for i in range(n):
        lower_i = lower[i]
        inv_diag = 1.0 / lower_i[i]
        z_i = list(rhs[i])
        for k in range(i):
            l_ik = lower_i[k]
            if l_ik != 0.0:
                z_k = z[k]
                for col in range(m):
                    z_i[col] -= l_ik * z_k[col]
        for col in range(m):
            z_i[col] *= inv_diag
        z[i] = z_i

    # Backward substitution: L^T @ X = Z
    out: Matrix = [[0.0] * m for _ in range(n)]
    for i in range(n - 1, -1, -1):
        inv_diag = 1.0 / lower[i][i]
        x_i = list(z[i])
        for k in range(i + 1, n):
            l_ki = lower[k][i]
            if l_ki != 0.0:
                x_k = out[k]
                for col in range(m):
                    x_i[col] -= l_ki * x_k[col]
        for col in range(m):
            x_i[col] *= inv_diag
        out[i] = x_i

    return out


class LensTensor(StrictModel):
    """Little-endian IEEE-754 float32 tensor serialized in base64."""

    dtype: Literal["f32"] = "f32"
    byte_order: Literal["little"] = "little"
    shape: list[int] = Field(min_length=1, max_length=2)
    base64: str = Field(min_length=1)

    @field_validator("shape")
    @classmethod
    def _positive_shape(cls, value: list[int]) -> list[int]:
        if any(dim <= 0 for dim in value):
            raise ValueError("tensor dimensions must be positive")
        return value

    @model_validator(mode="after")
    def _validate_payload(self) -> LensTensor:
        try:
            raw = base64.b64decode(self.base64, validate=True)
        except ValueError as error:
            raise ValueError("invalid base64 payload in LensTensor") from error
        expected_bytes = _prod(self.shape) * 4
        if len(raw) != expected_bytes:
            raise ValueError(
                f"LensTensor byte length {len(raw)} does not match shape {self.shape} "
                f"(expected {expected_bytes} bytes)"
            )
        count = _prod(self.shape)
        values = struct.unpack(f"<{count}f", raw)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("LensTensor contains non-finite float32 values")
        return self

    @classmethod
    def from_floats(cls, values: Sequence[float], shape: Sequence[int]) -> LensTensor:
        shape_list = [int(d) for d in shape]
        expected = _prod(shape_list)
        if len(values) != expected:
            raise ValueError(
                f"value count {len(values)} does not match shape {shape_list}"
            )
        floats = [float(x) for x in values]
        if not all(math.isfinite(x) for x in floats):
            raise ValueError("cannot encode non-finite values into LensTensor")
        raw = struct.pack(f"<{expected}f", *floats)
        return cls(
            dtype="f32",
            shape=shape_list,
            base64=base64.b64encode(raw).decode("ascii"),
        )

    @classmethod
    def from_matrix(cls, rows: Sequence[Sequence[float]]) -> LensTensor:
        if not rows or not rows[0]:
            raise ValueError("matrix must be non-empty")
        n_rows = len(rows)
        n_cols = len(rows[0])
        flat: list[float] = []
        for row in rows:
            if len(row) != n_cols:
                raise ValueError("ragged matrix rows")
            flat.extend(float(x) for x in row)
        return cls.from_floats(flat, [n_rows, n_cols])

    def to_floats(self) -> Vector:
        raw = base64.b64decode(self.base64, validate=True)
        count = _prod(self.shape)
        return [float(x) for x in struct.unpack(f"<{count}f", raw)]

    def to_matrix(self) -> Matrix:
        if len(self.shape) != 2:
            raise ValueError(f"expected 2D tensor, got shape {self.shape}")
        rows, cols = self.shape
        flat = self.to_floats()
        return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


class TunedLensGateConfig(StrictModel):
    """Confidence gate controlling early exit at block 18 vs suffix handoff to 30."""

    min_logit_margin: float = Field(default=1.25, ge=0.0, le=30.0)
    min_entropy_concentration: float = Field(default=0.35, ge=0.0, le=1.0)


class TunedLensGateEvaluation(StrictModel):
    """Per-question Tuned Lens gate diagnostics."""

    passed: bool
    logit_margin: float = Field(ge=0.0)
    entropy_concentration: float = Field(ge=0.0, le=1.0)
    top_probability: float = Field(ge=0.0, le=1.0)


class TunedLensReleaseGates(StrictModel):
    """Held-out validation summary required before exposing block-18 readout."""

    eval_samples: int = Field(ge=1)
    top1_agreement: float = Field(ge=0.0, le=1.0)
    mean_kl_divergence: float = Field(ge=0.0)
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    gated_coverage: float = Field(ge=0.0, le=1.0)
    gated_top1_agreement: float = Field(ge=0.0, le=1.0)
    passed: bool


class CascadeBranchOutcome(StrictModel):
    """Result of evaluating one question branch through the block-18 Tuned Lens."""

    readout_source: ReadoutSource
    label_logits: list[float] = Field(min_length=2)
    probabilities: list[float] = Field(min_length=2)
    gate: TunedLensGateEvaluation
    lower_tokens: int = Field(ge=1)
    upper_tokens: int = Field(ge=0)


class TunedLensArtifact(StrictModel):
    """Validated `tuned-lens-v1` artifact for block-18 readout on `split18-30-v1`."""

    schema_version: Literal["tuned-lens-v1"] = TUNED_LENS_SCHEMA_VERSION
    profile: Literal["split18-30-v1"] = SPLIT_PROFILE_ID
    model_sha256: Sha256Hex
    source_block: int = Field(default=18, ge=1)
    target_block: int = Field(default=30, ge=2)
    n_embd: int = Field(default=2816, ge=2)
    rms_norm_eps: float = Field(default=DEFAULT_RMS_NORM_EPS, gt=0.0, le=1e-2)
    final_logit_softcapping: float = Field(
        default=DEFAULT_FINAL_LOGIT_SOFTCAPPING, ge=0.0, le=1000.0
    )
    ridge_alpha: float = Field(default=100.0, gt=0.0)
    label_token_ids: list[int] = Field(min_length=2)
    gate: TunedLensGateConfig = Field(default_factory=TunedLensGateConfig)
    release_gates: TunedLensReleaseGates
    w: LensTensor
    b: LensTensor
    output_norm: LensTensor
    unembed: LensTensor

    @model_validator(mode="before")
    @classmethod
    def _finite(cls, value: object) -> object:
        _reject_nonfinite(value, "tuned_lens_artifact")
        return value

    @model_validator(mode="after")
    def _validate_shapes(self) -> TunedLensArtifact:
        if self.source_block >= self.target_block:
            raise ValueError("source_block must be strictly less than target_block")
        d = self.n_embd
        m = len(self.label_token_ids)
        if len(set(self.label_token_ids)) != m:
            raise ValueError("label_token_ids must be distinct")
        if self.w.shape != [d, d]:
            raise ValueError(f"w shape {self.w.shape} must be [{d}, {d}]")
        if self.b.shape != [d]:
            raise ValueError(f"b shape {self.b.shape} must be [{d}]")
        if self.output_norm.shape != [d]:
            raise ValueError(
                f"output_norm shape {self.output_norm.shape} must be [{d}]"
            )
        if self.unembed.shape != [m, d]:
            raise ValueError(f"unembed shape {self.unembed.shape} must be [{m}, {d}]")
        return self

    def require_releasable(self, *, expected_model_sha256: str | None = None) -> None:
        """Enforce AGENTS.md invariant: never expose unvalidated block-18 readout."""
        if not self.release_gates.passed:
            raise ValueError(
                "Tuned Lens artifact has not passed release gates; "
                "refusing to enable block-18 readout"
            )
        if (
            expected_model_sha256 is not None
            and self.model_sha256 != expected_model_sha256
        ):
            raise ValueError(
                f"Tuned Lens model_sha256 {self.model_sha256} does not match "
                f"expected model_sha256 {expected_model_sha256}"
            )

    def project_residual(self, h18: Sequence[float]) -> Vector:
        """Apply the affine residual translator `W @ h18 + b`."""
        if len(h18) != self.n_embd:
            raise ValueError(f"expected h18 length {self.n_embd}, got {len(h18)}")
        w_mat = self.w.to_matrix()
        b_vec = self.b.to_floats()
        return [
            _dot(w_row, h18) + b_val for w_row, b_val in zip(w_mat, b_vec, strict=True)
        ]

    def apply_rms_norm(self, h_pre_norm: Sequence[float]) -> Vector:
        """Apply final RMSNorm with `output_norm` weights."""
        if len(h_pre_norm) != self.n_embd:
            raise ValueError(
                f"expected residual length {self.n_embd}, got {len(h_pre_norm)}"
            )
        norm_w = self.output_norm.to_floats()
        mean_sq = sum(x * x for x in h_pre_norm) / len(h_pre_norm)
        scale = 1.0 / math.sqrt(mean_sq + self.rms_norm_eps)
        return [x * scale * w for x, w in zip(h_pre_norm, norm_w, strict=True)]

    def predict_label_logits(
        self,
        h18: Sequence[float],
        *,
        option_indices: Sequence[int] | None = None,
    ) -> Vector:
        """Project `H18` -> `H30_hat` -> `RMSNorm` -> `unembed` -> `softcap`."""
        h30_hat = self.project_residual(h18)
        normed = self.apply_rms_norm(h30_hat)
        unembed_mat = self.unembed.to_matrix()
        rows = (
            [unembed_mat[idx] for idx in option_indices]
            if option_indices is not None
            else unembed_mat
        )
        raw_logits = [_dot(row, normed) for row in rows]
        return apply_softcap(raw_logits, self.final_logit_softcapping)

    def evaluate_gate(self, label_logits: Sequence[float]) -> TunedLensGateEvaluation:
        """Evaluate whether a Tuned Lens readout is confident enough for early exit."""
        logits = [float(x) for x in label_logits]
        probs = stable_softmax(logits)
        margin = logit_margin(logits)
        concentration = entropy_concentration(probs)
        passed = (
            margin >= self.gate.min_logit_margin
            and concentration >= self.gate.min_entropy_concentration
        )
        return TunedLensGateEvaluation(
            passed=passed,
            logit_margin=margin,
            entropy_concentration=concentration,
            top_probability=max(probs),
        )

    def evaluate_branch(
        self,
        h18_last: Sequence[float],
        *,
        option_indices: Sequence[int],
        prefix_tokens: int,
        suffix_tokens: int,
        fallback_block30_logits: Sequence[float] | None = None,
        save_as: bool = False,
    ) -> CascadeBranchOutcome:
        """Evaluate a question suffix at block 18 with zero-recompute suffix handoff."""
        if prefix_tokens < 1 or suffix_tokens < 1:
            raise ValueError("prefix_tokens and suffix_tokens must be positive")
        lens_logits = self.predict_label_logits(h18_last, option_indices=option_indices)
        gate_eval = self.evaluate_gate(lens_logits)

        if gate_eval.passed and not save_as:
            probs = stable_softmax(lens_logits)
            return CascadeBranchOutcome(
                readout_source="tuned_lens_18",
                label_logits=lens_logits,
                probabilities=probs,
                gate=gate_eval,
                lower_tokens=suffix_tokens,
                upper_tokens=0,
            )

        if fallback_block30_logits is None:
            raise ValueError(
                "fallback_block30_logits is required when the Tuned Lens gate defers "
                "or when save_as is requested"
            )
        final_logits = [float(x) for x in fallback_block30_logits]
        probs = stable_softmax(final_logits)
        return CascadeBranchOutcome(
            readout_source="cascade_fallback_30",
            label_logits=final_logits,
            probabilities=probs,
            gate=gate_eval,
            lower_tokens=suffix_tokens,
            upper_tokens=prefix_tokens + suffix_tokens,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> TunedLensArtifact:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


def apply_softcap(logits: Sequence[float], softcap: float) -> Vector:
    """Apply Gemma 4 final logit softcapping `softcap * tanh(x / softcap)`."""
    if softcap <= 0.0:
        return [float(x) for x in logits]
    return [softcap * math.tanh(float(x) / softcap) for x in logits]


def invert_softcap(capped_logits: Sequence[float], softcap: float) -> Vector:
    """Invert `softcap * tanh(x / softcap)` on interior values."""
    if softcap <= 0.0:
        return [float(x) for x in capped_logits]
    limit = 1.0 - 1e-6
    out: Vector = []
    for x in capped_logits:
        scaled = max(-limit, min(limit, float(x) / softcap))
        out.append(softcap * math.atanh(scaled))
    return out


def _rms_normalize_row(row: Sequence[float], eps: float) -> Vector:
    mean_sq = sum(x * x for x in row) / len(row)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [x * scale for x in row]


def extract_gguf_head_weights(
    gguf_path: Path,
    label_token_ids: Sequence[int],
    *,
    n_embd: int = 2816,
) -> tuple[Vector, Matrix]:
    """Read exact `output_norm.weight` and label unembedding rows from GGUF.

    Reading the model's exact 2816-dimensional RMSNorm and tied token-embedding rows
    directly from the GGUF file avoids rank deficiency in the `(2816 - N)` orthogonal
    subspace when fitting a Tuned Lens with `N < n_embd` calibration samples.
    Supports `GGML_TYPE_F32` (0), `GGML_TYPE_F16` (1), and `GGML_TYPE_Q8_0` (8).
    """

    def _read_str(f: BinaryIO) -> str:
        length = int(struct.unpack("<Q", f.read(8))[0])
        return str(f.read(length).decode("utf-8"))

    def _skip_val(f: BinaryIO, vtype: int) -> None:
        scalar_sizes = {
            0: 1,
            1: 1,
            2: 2,
            3: 2,
            4: 4,
            5: 4,
            6: 4,
            7: 1,
            10: 8,
            11: 8,
            12: 8,
        }
        if vtype in scalar_sizes:
            f.read(scalar_sizes[vtype])
        elif vtype == 8:
            _read_str(f)
        elif vtype == 9:
            atype, alen = struct.unpack("<IQ", f.read(12))
            for _ in range(int(alen)):
                _skip_val(f, int(atype))
        else:
            raise ValueError(f"unsupported GGUF metadata value type {vtype}")

    def _decode_row(raw_bytes: bytes, ttype: int) -> Vector:
        if ttype == 0:  # GGML_TYPE_F32
            return [float(x) for x in struct.unpack(f"<{n_embd}f", raw_bytes)]
        if ttype == 1:  # GGML_TYPE_F16
            return [float(x) for x in struct.unpack(f"<{n_embd}e", raw_bytes)]
        if ttype == 8:  # GGML_TYPE_Q8_0 (34 bytes per 32-element block)
            if n_embd % 32 != 0:
                raise ValueError("Q8_0 row length must be divisible by 32")
            out: Vector = []
            for block_idx in range(n_embd // 32):
                block = raw_bytes[block_idx * 34 : (block_idx + 1) * 34]
                scale = float(struct.unpack("<e", block[:2])[0])
                quants = struct.unpack("<32b", block[2:])
                out.extend(scale * q for q in quants)
            return out
        raise ValueError(f"unsupported GGML tensor type {ttype}")

    def _row_byte_length(ttype: int) -> int:
        if ttype == 0:
            return n_embd * 4
        if ttype == 1:
            return n_embd * 2
        if ttype == 8:
            return (n_embd // 32) * 34
        raise ValueError(f"unsupported GGML tensor type {ttype}")

    with gguf_path.open("rb") as f:
        magic, _version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            raise ValueError(f"{gguf_path} is not a valid GGUF file")
        for _ in range(int(n_kv)):
            _ = _read_str(f)
            vtype = int(struct.unpack("<I", f.read(4))[0])
            _skip_val(f, vtype)

        tensors: dict[str, tuple[list[int], int, int]] = {}
        for _ in range(int(n_tensors)):
            name = _read_str(f)
            n_dims = int(struct.unpack("<I", f.read(4))[0])
            dims = [int(struct.unpack("<Q", f.read(8))[0]) for _ in range(n_dims)]
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors[name] = (dims, int(ttype), int(offset))

        pos = f.tell()
        data_start = (pos + 31) // 32 * 32

        if "output_norm.weight" not in tensors:
            raise ValueError("GGUF file is missing output_norm.weight tensor")
        norm_dims, norm_type, norm_offset = tensors["output_norm.weight"]
        if norm_dims[0] != n_embd:
            raise ValueError(
                f"output_norm.weight dim {norm_dims[0]} != n_embd {n_embd}"
            )
        f.seek(data_start + norm_offset)
        norm_w = _decode_row(f.read(_row_byte_length(norm_type)), norm_type)

        unembed_name = (
            "output.weight" if "output.weight" in tensors else "token_embd.weight"
        )
        if unembed_name not in tensors:
            raise ValueError(
                "GGUF file is missing output.weight and token_embd.weight tensors"
            )
        embd_dims, embd_type, embd_offset = tensors[unembed_name]
        if embd_dims[0] != n_embd:
            raise ValueError(f"{unembed_name} dim {embd_dims[0]} != n_embd {n_embd}")
        row_bytes = _row_byte_length(embd_type)
        unembed: Matrix = []
        for tok_id in label_token_ids:
            if tok_id < 0 or (len(embd_dims) > 1 and tok_id >= embd_dims[1]):
                raise ValueError(f"label token id {tok_id} out of range")
            f.seek(data_start + embd_offset + int(tok_id) * row_bytes)
            unembed.append(_decode_row(f.read(row_bytes), embd_type))

    return norm_w, unembed


def recover_output_norm_weight(
    h_pre_norm: Sequence[Sequence[float]],
    h_normalized: Sequence[Sequence[float]],
    *,
    rms_norm_eps: float = DEFAULT_RMS_NORM_EPS,
) -> Vector:
    """Recover exact `output_norm` scale from `(last_residual, last_normalized)`."""
    if not h_pre_norm or len(h_pre_norm) != len(h_normalized):
        raise ValueError(
            "h_pre_norm and h_normalized must have matching non-empty rows"
        )
    d = len(h_pre_norm[0])
    numer = [0.0] * d
    denom = [0.0] * d
    for pre_row, norm_row in zip(h_pre_norm, h_normalized, strict=True):
        if len(pre_row) != d or len(norm_row) != d:
            raise ValueError("inconsistent embedding dimension across rows")
        u_row = _rms_normalize_row(pre_row, rms_norm_eps)
        for j in range(d):
            u_j = u_row[j]
            numer[j] += u_j * norm_row[j]
            denom[j] += u_j * u_j
    return [numer[j] / denom[j] if denom[j] > 1e-12 else 1.0 for j in range(d)]


def fit_unembed_rows(
    h_pre_norm: Sequence[Sequence[float]],
    output_norm_weight: Sequence[float],
    target_logits: Sequence[Sequence[float]],
    *,
    rms_norm_eps: float = DEFAULT_RMS_NORM_EPS,
    final_logit_softcapping: float = DEFAULT_FINAL_LOGIT_SOFTCAPPING,
    ridge_alpha: float = 1e-4,
) -> Matrix:
    """Recover or fit `(n_labels, n_embd)` unembedding rows from normalized states."""
    n_samples = len(h_pre_norm)
    if n_samples == 0 or len(target_logits) != n_samples:
        raise ValueError("h_pre_norm and target_logits must have matching sample count")
    d = len(output_norm_weight)
    m = len(target_logits[0])
    if ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be positive")

    normed_rows: Matrix = []
    uncapped_targets: Matrix = []
    for pre_row, logit_row in zip(h_pre_norm, target_logits, strict=True):
        if len(pre_row) != d or len(logit_row) != m:
            raise ValueError("row dimension mismatch in fit_unembed_rows")
        u_row = _rms_normalize_row(pre_row, rms_norm_eps)
        normed_rows.append([u_row[j] * output_norm_weight[j] for j in range(d)])
        uncapped_targets.append(invert_softcap(logit_row, final_logit_softcapping))

    # Dual form: U^T = Z^T (Z Z^T + alpha I_N)^{-1} Y
    gram: Matrix = [[0.0] * n_samples for _ in range(n_samples)]
    for i in range(n_samples):
        row_i = normed_rows[i]
        for j in range(i, n_samples):
            val = _dot(row_i, normed_rows[j])
            gram[i][j] = val
            gram[j][i] = val
        gram[i][i] += ridge_alpha

    dual_weights = _cholesky_solve(gram, uncapped_targets)  # (n_samples, m)
    # unembed[label][feat] = sum_i dual_weights[i][label] * normed_rows[i][feat]
    unembed: Matrix = [[0.0] * d for _ in range(m)]
    for i in range(n_samples):
        z_i = normed_rows[i]
        dw_i = dual_weights[i]
        for label_idx in range(m):
            weight = dw_i[label_idx]
            row_out = unembed[label_idx]
            for feat_idx in range(d):
                row_out[feat_idx] += weight * z_i[feat_idx]
    return unembed


def fit_affine_ridge(
    h18: Sequence[Sequence[float]],
    h30: Sequence[Sequence[float]],
    *,
    ridge_alpha: float = 100.0,
    identity_prior: bool = True,
) -> tuple[Matrix, Vector]:
    """Fit `h30 ≈ h18 @ W^T + b` with L2 regularization toward identity (`W = I + ΔW`).

    Uses the exact dual push-through identity:
        ΔW^T = X_c^T (X_c X_c^T + alpha I_N)^{-1} R_c
    which holds for all `(n_samples, n_embd)` when `ridge_alpha > 0`.
    """
    n_samples = len(h18)
    if n_samples < 2 or len(h30) != n_samples:
        raise ValueError("at least 2 paired (h18, h30) rows are required")
    d = len(h18[0])
    if ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be positive")

    mean_x = [0.0] * d
    mean_r = [0.0] * d
    residuals: Matrix = []
    for row18, row30 in zip(h18, h30, strict=True):
        if len(row18) != d or len(row30) != d:
            raise ValueError("inconsistent embedding dimension in fit_affine_ridge")
        r_row = [
            (row30[j] - row18[j]) if identity_prior else row30[j] for j in range(d)
        ]
        residuals.append(r_row)
        for j in range(d):
            mean_x[j] += row18[j]
            mean_r[j] += r_row[j]

    inv_n = 1.0 / n_samples
    mean_x = [val * inv_n for val in mean_x]
    mean_r = [val * inv_n for val in mean_r]

    x_centered: Matrix = [[row18[j] - mean_x[j] for j in range(d)] for row18 in h18]
    r_centered: Matrix = [
        [r_row[j] - mean_r[j] for j in range(d)] for r_row in residuals
    ]

    gram: Matrix = [[0.0] * n_samples for _ in range(n_samples)]
    for i in range(n_samples):
        xi = x_centered[i]
        for j in range(i, n_samples):
            val = _dot(xi, x_centered[j])
            gram[i][j] = val
            gram[j][i] = val
        gram[i][i] += ridge_alpha

    dual_weights = _cholesky_solve(gram, r_centered)  # (n_samples, d)

    # delta_w[out_dim][in_dim] = sum_i dual_weights[i][out_dim] * x_centered[i][in_dim]
    # Collect x_centered by sample to accumulate efficiently:
    w_mat: Matrix = [[0.0] * d for _ in range(d)]
    for i in range(n_samples):
        xi = x_centered[i]
        dwi = dual_weights[i]
        for out_dim in range(d):
            coeff = dwi[out_dim]
            if coeff == 0.0:
                continue
            w_row = w_mat[out_dim]
            for in_dim in range(d):
                w_row[in_dim] += coeff * xi[in_dim]

    b_vec: Vector = [
        mean_r[out_dim] - _dot(w_mat[out_dim], mean_x) for out_dim in range(d)
    ]
    if identity_prior:
        for idx in range(d):
            w_mat[idx][idx] += 1.0

    return w_mat, b_vec


def evaluate_lens_quality(
    artifact: TunedLensArtifact,
    *,
    val_h18: Sequence[Sequence[float]],
    val_h30: Sequence[Sequence[float]],
    val_target_logits: Sequence[Sequence[float]],
    option_subsets: Sequence[Sequence[int]] | None = None,
    min_top1_agreement: float = 0.85,
    max_mean_kl: float = 0.15,
    min_gated_agreement: float = 0.92,
    min_gated_coverage: float = 0.30,
) -> TunedLensReleaseGates:
    """Evaluate a Tuned Lens against held-out block-30 residuals and label logits."""
    n_samples = len(val_h18)
    if (
        n_samples == 0
        or len(val_h30) != n_samples
        or len(val_target_logits) != n_samples
    ):
        raise ValueError("val_h18, val_h30, and val_target_logits must match in length")

    cosines: list[float] = []
    kls: list[float] = []
    agreements: list[bool] = []
    gated_agreements: list[bool] = []

    for idx in range(n_samples):
        h18_row = val_h18[idx]
        h30_row = val_h30[idx]
        h30_hat = artifact.project_residual(h18_row)
        norm_hat = math.sqrt(max(_dot(h30_hat, h30_hat), 1e-24))
        norm_true = math.sqrt(max(_dot(h30_row, h30_row), 1e-24))
        cosines.append(_dot(h30_hat, h30_row) / (norm_hat * norm_true))

        subset = option_subsets[idx] if option_subsets is not None else None
        pred_logits = artifact.predict_label_logits(h18_row, option_indices=subset)
        full_target = val_target_logits[idx]
        target_logits = (
            [float(full_target[i]) for i in subset]
            if subset is not None
            else [float(x) for x in full_target]
        )

        p_true = stable_softmax(target_logits)
        p_pred = stable_softmax(pred_logits)

        kl = sum(
            pt * math.log(max(pt, 1e-12) / max(pp, 1e-12))
            for pt, pp in zip(p_true, p_pred, strict=True)
            if pt > 0.0
        )
        kls.append(max(0.0, kl))

        top_true = max(range(len(p_true)), key=p_true.__getitem__)
        top_pred = max(range(len(p_pred)), key=p_pred.__getitem__)
        agree = top_true == top_pred
        agreements.append(agree)

        gate_eval = artifact.evaluate_gate(pred_logits)
        if gate_eval.passed:
            gated_agreements.append(agree)

    top1_agreement = sum(agreements) / n_samples
    mean_kl = sum(kls) / n_samples
    mean_cos = max(-1.0, min(1.0, sum(cosines) / n_samples))
    gated_cov = len(gated_agreements) / n_samples
    gated_acc = (
        sum(gated_agreements) / len(gated_agreements)
        if gated_agreements
        else top1_agreement
    )
    passed = (
        top1_agreement >= min_top1_agreement
        and mean_kl <= max_mean_kl
        and gated_cov >= min_gated_coverage
        and gated_acc >= min_gated_agreement
    )
    return TunedLensReleaseGates(
        eval_samples=n_samples,
        top1_agreement=top1_agreement,
        mean_kl_divergence=mean_kl,
        cosine_similarity=mean_cos,
        gated_coverage=gated_cov,
        gated_top1_agreement=gated_acc,
        passed=passed,
    )


def fit_tuned_lens(
    *,
    train_h18: Sequence[Sequence[float]],
    train_h30: Sequence[Sequence[float]],
    val_h18: Sequence[Sequence[float]],
    val_h30: Sequence[Sequence[float]],
    val_target_logits: Sequence[Sequence[float]],
    model_sha256: str,
    label_token_ids: Sequence[int],
    output_norm_weight: Sequence[float] | None = None,
    h30_normalized: Sequence[Sequence[float]] | None = None,
    unembed_weight: Sequence[Sequence[float]] | None = None,
    all_h30_for_head: Sequence[Sequence[float]] | None = None,
    all_target_logits_for_head: Sequence[Sequence[float]] | None = None,
    ridge_alpha: float = 100.0,
    rms_norm_eps: float = DEFAULT_RMS_NORM_EPS,
    final_logit_softcapping: float = DEFAULT_FINAL_LOGIT_SOFTCAPPING,
    gate: TunedLensGateConfig | None = None,
    option_subsets: Sequence[Sequence[int]] | None = None,
    min_top1_agreement: float = 0.85,
    max_mean_kl: float = 0.15,
    min_gated_agreement: float = 0.92,
    min_gated_coverage: float = 0.30,
) -> TunedLensArtifact:
    """Fit a complete `TunedLensArtifact` and evaluate its release gates."""
    w_mat, b_vec = fit_affine_ridge(
        train_h18, train_h30, ridge_alpha=ridge_alpha, identity_prior=True
    )
    d = len(b_vec)

    if output_norm_weight is None:
        if h30_normalized is None:
            raise ValueError(
                "either output_norm_weight or h30_normalized must be provided"
            )
        ref_h30 = all_h30_for_head if all_h30_for_head is not None else train_h30
        norm_w = recover_output_norm_weight(
            ref_h30, h30_normalized, rms_norm_eps=rms_norm_eps
        )
    else:
        norm_w = [float(x) for x in output_norm_weight]

    if unembed_weight is None:
        ref_h30 = all_h30_for_head if all_h30_for_head is not None else val_h30
        ref_logits = (
            all_target_logits_for_head
            if all_target_logits_for_head is not None
            else val_target_logits
        )
        unembed_mat = fit_unembed_rows(
            ref_h30,
            norm_w,
            ref_logits,
            rms_norm_eps=rms_norm_eps,
            final_logit_softcapping=final_logit_softcapping,
        )
    else:
        unembed_mat = [[float(x) for x in row] for row in unembed_weight]

    provisional_gates = TunedLensReleaseGates(
        eval_samples=max(1, len(val_h18)),
        top1_agreement=0.0,
        mean_kl_divergence=0.0,
        cosine_similarity=0.0,
        gated_coverage=0.0,
        gated_top1_agreement=0.0,
        passed=False,
    )
    provisional = TunedLensArtifact(
        model_sha256=model_sha256,
        n_embd=d,
        rms_norm_eps=rms_norm_eps,
        final_logit_softcapping=final_logit_softcapping,
        ridge_alpha=ridge_alpha,
        label_token_ids=list(label_token_ids),
        gate=gate or TunedLensGateConfig(),
        release_gates=provisional_gates,
        w=LensTensor.from_matrix(w_mat),
        b=LensTensor.from_floats(b_vec, [d]),
        output_norm=LensTensor.from_floats(norm_w, [d]),
        unembed=LensTensor.from_matrix(unembed_mat),
    )
    release_gates = evaluate_lens_quality(
        provisional,
        val_h18=val_h18,
        val_h30=val_h30,
        val_target_logits=val_target_logits,
        option_subsets=option_subsets,
        min_top1_agreement=min_top1_agreement,
        max_mean_kl=max_mean_kl,
        min_gated_agreement=min_gated_agreement,
        min_gated_coverage=min_gated_coverage,
    )
    return provisional.model_copy(update={"release_gates": release_gates})
