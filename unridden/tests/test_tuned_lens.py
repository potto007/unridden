"""Unit and integration tests for Tuned-Lens Block-18 Readout + Suffix Handoff."""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path
from typing import Any

import pytest

from scripts.unridden.train_tuned_lens import (
    load_hidden_state_samples,
    train_and_evaluate_from_samples,
)
from unridden.api.schema import FullVocabularyArgmax
from unridden.api.snapshots.backend import SnapshotNativeBackend
from unridden.api.snapshots.errors import SnapshotUnavailableError
from unridden.api.snapshots.lens import (
    LensTensor,
    TunedLensArtifact,
    TunedLensGateConfig,
    TunedLensReleaseGates,
    apply_softcap,
    evaluate_lens_quality,
    extract_gguf_head_weights,
    fit_affine_ridge,
    fit_tuned_lens,
    fit_unembed_rows,
    invert_softcap,
    recover_output_norm_weight,
)
from unridden.api.snapshots.schema import (
    BlockTokens,
    WorkerResult,
    WorkerSnapshotBranch,
    WorkerSnapshotQuestionResult,
)
from unridden.tests.test_snapshots_app import (
    CONTEXT_BODY,
    HELLO,
    FakeSnapshotBackend,
    _logits,
    _sha,
    _tokens,
    client_for,
)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _synthetic_lens_dataset(
    *,
    n_samples: int = 48,
    n_embd: int = 12,
    n_labels: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a synthetic `(H18, H30, last_normalized, label_logits)` dataset."""
    rng = random.Random(seed)
    true_w: list[list[float]] = [
        [(1.0 if r == c else 0.0) + rng.gauss(0.0, 0.08) for c in range(n_embd)]
        for r in range(n_embd)
    ]
    true_b = [rng.gauss(0.0, 0.05) for _ in range(n_embd)]
    true_norm = [rng.uniform(0.8, 1.4) for _ in range(n_embd)]
    true_unembed: list[list[float]] = [
        [rng.gauss(0.0, 2.2) for _ in range(n_embd)] for _ in range(n_labels)
    ]

    h18: list[list[float]] = []
    h30: list[list[float]] = []
    h30_norm: list[list[float]] = []
    logits: list[list[float]] = []

    for _ in range(n_samples):
        row18 = [rng.gauss(0.0, 1.0) for _ in range(n_embd)]
        row30 = [
            _dot(true_w[r], row18) + true_b[r] + rng.gauss(0.0, 0.005)
            for r in range(n_embd)
        ]
        rms = math.sqrt(sum(x * x for x in row30) / n_embd + 1e-6)
        row_norm = [(row30[j] / rms) * true_norm[j] for j in range(n_embd)]
        raw_logits = [_dot(true_unembed[k], row_norm) for k in range(n_labels)]
        capped_logits = apply_softcap(raw_logits, 30.0)

        h18.append(row18)
        h30.append(row30)
        h30_norm.append(row_norm)
        logits.append(capped_logits)

    return {
        "h18": h18,
        "h30": h30,
        "h30_norm": h30_norm,
        "logits": logits,
        "true_w": true_w,
        "true_b": true_b,
        "true_norm": true_norm,
        "true_unembed": true_unembed,
    }


def test_lens_tensor_roundtrip_and_validation() -> None:
    matrix = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
    tensor = LensTensor.from_matrix(matrix)
    decoded = tensor.to_matrix()
    for row_dec, row_exp in zip(decoded, matrix, strict=True):
        for val_dec, val_exp in zip(row_dec, row_exp, strict=True):
            assert val_dec == pytest.approx(val_exp)

    with pytest.raises(ValueError, match="non-finite"):
        LensTensor.from_floats([1.0, math.nan], [2])


def test_softcap_and_inverse_roundtrip() -> None:
    values = [-20.0, -3.5, 0.0, 4.2, 18.0]
    capped = apply_softcap(values, 30.0)
    recovered = invert_softcap(capped, 30.0)
    for rec, exp in zip(recovered, values, strict=True):
        assert rec == pytest.approx(exp, rel=1e-5, abs=1e-5)


def test_recover_output_norm_and_unembed_rows() -> None:
    data = _synthetic_lens_dataset(n_samples=40, n_embd=12, n_labels=4)
    norm_weight = recover_output_norm_weight(
        data["h30"], data["h30_norm"], rms_norm_eps=1e-6
    )
    for fit_val, true_val in zip(norm_weight, data["true_norm"], strict=True):
        assert fit_val == pytest.approx(true_val, rel=1e-4, abs=1e-4)

    unembed = fit_unembed_rows(
        data["h30"],
        norm_weight,
        data["logits"],
        rms_norm_eps=1e-6,
        final_logit_softcapping=30.0,
        ridge_alpha=1e-6,
    )
    for fit_row, true_row in zip(unembed, data["true_unembed"], strict=True):
        for fit_val, true_val in zip(fit_row, true_row, strict=True):
            assert fit_val == pytest.approx(true_val, rel=1e-3, abs=1e-3)


def test_fit_affine_ridge_identity_prior_and_reconstruction() -> None:
    rng = random.Random(7)
    h18_small = [[rng.gauss(0.0, 1.0) for _ in range(12)] for _ in range(8)]
    h30_small = [list(row) for row in h18_small]
    w_id, b_id = fit_affine_ridge(
        h18_small, h30_small, ridge_alpha=100.0, identity_prior=True
    )
    for r in range(12):
        assert b_id[r] == pytest.approx(0.0, abs=1e-6)
        for c in range(12):
            expected = 1.0 if r == c else 0.0
            assert w_id[r][c] == pytest.approx(expected, abs=1e-6)

    data = _synthetic_lens_dataset(n_samples=32, n_embd=12, n_labels=4)
    w_fit, b_fit = fit_affine_ridge(
        data["h18"], data["h30"], ridge_alpha=0.05, identity_prior=True
    )
    sq_err = 0.0
    count = 0
    for row18, row30 in zip(data["h18"], data["h30"], strict=True):
        for r in range(12):
            pred = _dot(w_fit[r], row18) + b_fit[r]
            sq_err += (pred - row30[r]) ** 2
            count += 1
    assert (sq_err / count) < 1e-3


def test_fit_tuned_lens_and_release_gates(tmp_path: Path) -> None:
    data = _synthetic_lens_dataset(n_samples=64, n_embd=12, n_labels=4)
    artifact = fit_tuned_lens(
        train_h18=data["h18"][:48],
        train_h30=data["h30"][:48],
        val_h18=data["h18"][48:],
        val_h30=data["h30"][48:],
        val_target_logits=data["logits"][48:],
        model_sha256="a" * 64,
        label_token_ids=[11, 12, 13, 14],
        h30_normalized=data["h30_norm"],
        all_h30_for_head=data["h30"],
        all_target_logits_for_head=data["logits"],
        ridge_alpha=0.5,
        gate=TunedLensGateConfig(min_logit_margin=0.5, min_entropy_concentration=0.1),
        min_top1_agreement=0.85,
        max_mean_kl=0.15,
        min_gated_agreement=0.90,
        min_gated_coverage=0.25,
    )
    assert artifact.release_gates.passed is True
    assert artifact.release_gates.top1_agreement >= 0.85
    assert artifact.release_gates.mean_kl_divergence <= 0.15
    assert artifact.release_gates.cosine_similarity > 0.95

    artifact_path = tmp_path / "lens.json"
    artifact.save(artifact_path)
    loaded = TunedLensArtifact.load(artifact_path)
    loaded.require_releasable(expected_model_sha256="a" * 64)

    with pytest.raises(ValueError, match="does not match"):
        loaded.require_releasable(expected_model_sha256="b" * 64)


def test_tuned_lens_evaluate_branch_early_exit_and_cascade_fallback() -> None:
    data = _synthetic_lens_dataset(n_samples=48, n_embd=8, n_labels=4)
    artifact = fit_tuned_lens(
        train_h18=data["h18"][:36],
        train_h30=data["h30"][:36],
        val_h18=data["h18"][36:],
        val_h30=data["h30"][36:],
        val_target_logits=data["logits"][36:],
        model_sha256="a" * 64,
        label_token_ids=[11, 12, 13, 14],
        output_norm_weight=data["true_norm"],
        unembed_weight=data["true_unembed"],
        ridge_alpha=0.5,
        gate=TunedLensGateConfig(min_logit_margin=0.5, min_entropy_concentration=0.15),
        min_top1_agreement=0.80,
        max_mean_kl=0.25,
        min_gated_agreement=0.80,
        min_gated_coverage=0.10,
    )

    confident_h18 = [x * 4.0 for x in data["h18"][0]]
    early = artifact.evaluate_branch(
        confident_h18,
        option_indices=[0, 1],
        prefix_tokens=120,
        suffix_tokens=15,
    )
    assert early.readout_source == "tuned_lens_18"
    assert early.gate.passed is True
    assert early.lower_tokens == 15
    assert early.upper_tokens == 0

    # Solve W @ h18 + b == 0 by constructing an input that maps near zero:
    # Since b is small and W is near I, [-b_j] with identity W gives 0 logits.
    zero_w = [[1.0 if r == c else 0.0 for c in range(8)] for r in range(8)]
    flat_artifact = artifact.model_copy(
        update={
            "w": LensTensor.from_matrix(zero_w),
            "b": LensTensor.from_floats([0.0] * 8, [8]),
        }
    )
    fallback_logits = [3.0, 0.5]
    cascaded = flat_artifact.evaluate_branch(
        [0.0] * 8,
        option_indices=[0, 1],
        prefix_tokens=120,
        suffix_tokens=15,
        fallback_block30_logits=fallback_logits,
    )
    assert cascaded.readout_source == "cascade_fallback_30"
    assert cascaded.gate.passed is False
    assert cascaded.label_logits == fallback_logits
    assert cascaded.lower_tokens == 15
    assert cascaded.upper_tokens == 135

    saved = artifact.evaluate_branch(
        confident_h18,
        option_indices=[0, 1],
        prefix_tokens=120,
        suffix_tokens=15,
        fallback_block30_logits=fallback_logits,
        save_as=True,
    )
    assert saved.readout_source == "cascade_fallback_30"
    assert saved.lower_tokens == 15
    assert saved.upper_tokens == 135


def test_train_tuned_lens_script_roundtrip(tmp_path: Path) -> None:
    data = _synthetic_lens_dataset(n_samples=32, n_embd=8, n_labels=4)
    samples_path = tmp_path / "samples.jsonl"
    rows: list[str] = []
    for idx in range(32):
        row = {
            "h18": [float(x) for x in data["h18"][idx]],
            "h30": [float(x) for x in data["h30"][idx]],
            "last_normalized": [float(x) for x in data["h30_norm"][idx]],
            "label_logits": [float(x) for x in data["logits"][idx]],
            "option_indices": [0, 1, 2, 3],
            "split": "train" if idx < 24 else "val",
        }
        rows.append(json.dumps(row))
    samples_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    train_samples, val_samples = load_hidden_state_samples(samples_path)
    assert len(train_samples) == 24
    assert len(val_samples) == 8

    artifact = train_and_evaluate_from_samples(
        train_samples,
        val_samples,
        model_sha256="a" * 64,
        label_token_ids=[11, 12, 13, 14],
        ridge_alpha=0.5,
        gate_config=TunedLensGateConfig(
            min_logit_margin=0.2, min_entropy_concentration=0.05
        ),
        min_top1_agreement=0.75,
        max_mean_kl=0.25,
        min_gated_agreement=0.75,
        min_gated_coverage=0.10,
    )
    assert artifact.release_gates.passed is True
    assert evaluate_lens_quality(
        artifact,
        val_h18=data["h18"][24:],
        val_h30=data["h30"][24:],
        val_target_logits=data["logits"][24:],
        min_top1_agreement=0.75,
        max_mean_kl=0.25,
        min_gated_agreement=0.75,
        min_gated_coverage=0.10,
    ).passed


@pytest.mark.asyncio
async def test_backend_rejects_unvalidated_tuned_lens_before_spawning_worker(
    tmp_path: Path,
) -> None:
    data = _synthetic_lens_dataset(n_samples=16, n_embd=8, n_labels=4)
    artifact = fit_tuned_lens(
        train_h18=data["h18"][:12],
        train_h30=data["h30"][:12],
        val_h18=data["h18"][12:],
        val_h30=data["h30"][12:],
        val_target_logits=data["logits"][12:],
        model_sha256="a" * 64,
        label_token_ids=[11, 12, 13, 14],
        output_norm_weight=data["true_norm"],
        unembed_weight=data["true_unembed"],
        min_top1_agreement=1.0,
        max_mean_kl=0.0,
    )
    failed_gates = TunedLensReleaseGates(
        eval_samples=4,
        top1_agreement=0.5,
        mean_kl_divergence=0.9,
        cosine_similarity=0.5,
        gated_coverage=0.2,
        gated_top1_agreement=0.5,
        passed=False,
    )
    failed_artifact = artifact.model_copy(update={"release_gates": failed_gates})
    lens_path = tmp_path / "failed_lens.json"
    failed_artifact.save(lens_path)

    model_file = tmp_path / "fake.gguf"
    model_file.write_bytes(b"fake-gguf")
    manifest_file = tmp_path / "build.json"
    manifest_file.write_text(json.dumps({"binary_sha256": "0" * 64}), encoding="utf-8")
    worker_file = tmp_path / "worker"
    worker_file.write_bytes(b"")

    backend = SnapshotNativeBackend(
        worker_path=worker_file,
        model_path=model_file,
        manifest_path=manifest_file,
        tuned_lens_path=lens_path,
    )
    with pytest.raises(SnapshotUnavailableError, match="has not passed release gates"):
        await backend.start()


class TunedLensFakeSnapshotBackend(FakeSnapshotBackend):
    """Fake snapshot worker with `early_head='tuned-lens-v1'` and suffix handoff."""

    def __init__(self) -> None:
        super().__init__()
        self.hello = HELLO.model_copy(
            update={
                "early_head": "tuned-lens-v1",
                "readout_blocks": [18, 30],
            }
        )

    async def evaluate(
        self,
        *,
        snapshot_id: str,
        readout_blocks: int,
        questions: list[dict[str, Any]],
        steering: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> WorkerResult:
        if readout_blocks == 30:
            return await super().evaluate(
                snapshot_id=snapshot_id,
                readout_blocks=readout_blocks,
                questions=questions,
                steering=steering,
                timeout=timeout,
            )
        self.steering_seen.append(steering)
        assert readout_blocks == 18
        reused = self._snaps[snapshot_id]["tokens"]
        rows: list[WorkerSnapshotQuestionResult] = []
        for question in questions:
            content = (
                "".join(message["content"] for message in question["messages"])
                + question["answer_prefix"]
            )
            last_user = [
                message["content"]
                for message in question["messages"]
                if message["role"] == "user"
            ][-1]
            suffix = _tokens(last_user)
            labels = question["labels"]
            logits = _logits(content, len(labels))
            # Simulate gate: questions with "borderline" in instructions or save_as
            # cascade to block 30 via zero-recompute suffix handoff; others exit at 18.
            cascaded = "borderline" in last_user or bool(question.get("save_as"))
            child = None
            if question.get("save_as"):
                child_id = question["save_as"]
                self._snaps[child_id] = {
                    "tokens": reused + suffix,
                    "content": content,
                    "kind": "readout",
                }
                self._resident.add(child_id)
            rows.append(
                WorkerSnapshotQuestionResult(
                    id=question["id"],
                    label_logits=logits,
                    label_token_ids=HELLO.label_token_ids[: len(labels)],
                    allowed_label_mass=1.0 if not cascaded else 0.5,
                    full_vocabulary_argmax=FullVocabularyArgmax(
                        token_id=HELLO.label_token_ids[0], logit=max(logits)
                    ),
                    prompt_sha256=_sha(content),
                    prompt_tokens=reused + suffix,
                    processed_tokens=suffix,
                    reused_tokens=reused,
                    cache_cleared=False,
                    evaluation_mode="sequential",
                    batch_sequences=1,
                    timing_ms=1.0,
                    snapshot=WorkerSnapshotBranch(
                        parent=snapshot_id,
                        suffix_tokens=suffix,
                        block_tokens=BlockTokens(
                            lower=suffix,
                            upper=(reused + suffix) if cascaded else 0,
                        ),
                        restore="resident",
                        restored_bytes=0,
                        restore_ms=0.0,
                        inference_ms=1.0,
                        readout_source=(
                            "cascade_fallback_30" if cascaded else "tuned_lens_18"
                        ),
                        child=child,
                    ),
                )
            )
        return WorkerResult(
            type="result",
            id="e",
            model_sha256=self.hello.model_sha256,
            runtime_sha256=self.hello.runtime_sha256,
            generated_tokens=0,
            callbacks_enabled=False,
            execution_mode=self.hello.execution_mode,
            questions=rows,
        )


@pytest.mark.asyncio
async def test_v2_decisions_at_block_18_with_tuned_lens_and_suffix_handoff(
    tmp_path: Path,
) -> None:
    backend = TunedLensFakeSnapshotBackend()
    async with client_for(backend, tmp_path) as client:
        models = (await client.get("/v2/models")).json()["models"][0]
        assert models["capabilities"]["early_head"] == "tuned-lens-v1"
        assert models["capabilities"]["readout_blocks"] == [18, 30]

        # Capture only a block-18 snapshot (no block 30)
        created = await client.post(
            "/v2/snapshots",
            json=CONTEXT_BODY | {"checkpoints": [18]},
        )
        assert created.status_code == 200
        snap18 = created.json()["snapshots"][0]
        assert snap18["completed_blocks"] == 18
        assert snap18["capabilities"] == ["continue", "promote", "inspect"]

        # Evaluate two sibling questions at readout.completed_blocks = 18:
        # - q_confident exits early at block 18 (upper = 0)
        # - q_borderline cascades via zero-recompute suffix handoff to block 30
        response = await client.post(
            "/v2/decisions",
            json={
                "model": "local-gemma-unridden-v1",
                "snapshot": {"id": snap18["id"], "relationship": "followup"},
                "readout": {"completed_blocks": 18},
                "questions": {
                    "q_confident": {
                        "type": "choice",
                        "instructions": "clear routing question",
                        "criteria": {"A": "alpha", "B": "beta"},
                    },
                    "q_borderline": {
                        "type": "choice",
                        "instructions": "borderline routing question",
                        "criteria": {"A": "alpha", "B": "beta"},
                    },
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        usage = body["snapshot_usage"]
        # Prefix was never promoted upfront; suffix handoff handled only the
        # cascaded question.
        assert backend.promote_calls == 0
        assert usage["promotion"] == "none"
        assert usage["effective_parent"] == snap18["id"]
        assert usage["readout_source"] == {
            "q_confident": "tuned_lens_18",
            "q_borderline": "cascade_fallback_30",
        }
        # lower ran both question suffixes; upper ran only the cascaded branch
        s_conf = usage["suffix_tokens"]["q_confident"]
        s_bord = usage["suffix_tokens"]["q_borderline"]
        assert usage["block_tokens"]["lower"] == s_conf + s_bord
        assert usage["block_tokens"]["upper"] == usage["reused_prefix_tokens"] + s_bord
        assert body["usage"]["output_tokens"] == 0


def test_extract_gguf_head_weights_f32_and_q8_0(tmp_path: Path) -> None:
    n_embd = 32
    norm_expected = [1.0 + 0.05 * i for i in range(n_embd)]
    scale0 = 0.5
    quants0 = [i - 16 for i in range(n_embd)]
    scale1 = 0.25
    quants1 = [16 - i for i in range(n_embd)]

    def _gguf_str(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    header = bytearray()
    header += struct.pack("<4sIQQ", b"GGUF", 3, 2, 1)
    # 1 metadata KV (uint32)
    header += _gguf_str("general.alignment")
    header += struct.pack("<II", 4, 32)
    # Tensor 0: output_norm.weight (1D F32, offset 0)
    header += _gguf_str("output_norm.weight")
    header += struct.pack("<IQ", 1, n_embd)
    header += struct.pack("<IQ", 0, 0)
    # Tensor 1: token_embd.weight (2D Q8_0 [32, 3], offset 128)
    header += _gguf_str("token_embd.weight")
    header += struct.pack("<IQQ", 2, n_embd, 3)
    header += struct.pack("<IQ", 8, n_embd * 4)

    pad = ((len(header) + 31) // 32 * 32) - len(header)
    header += b"\x00" * pad

    data = bytearray()
    data += struct.pack(f"<{n_embd}f", *norm_expected)
    # 3 Q8_0 rows (34 bytes each)
    block0 = struct.pack("<e", scale0) + struct.pack("<32b", *quants0)
    block1 = struct.pack("<e", scale1) + struct.pack("<32b", *quants1)
    block2 = struct.pack("<e", 1.0) + struct.pack("<32b", *([0] * n_embd))
    data += block0 + block1 + block2

    gguf_file = tmp_path / "tiny.gguf"
    gguf_file.write_bytes(bytes(header + data))

    norm_w, unembed = extract_gguf_head_weights(gguf_file, [2, 0], n_embd=n_embd)
    assert norm_w == pytest.approx(norm_expected, abs=1e-6)
    assert len(unembed) == 2
    assert unembed[0] == pytest.approx([0.0] * n_embd, abs=1e-6)
    assert unembed[1] == pytest.approx([scale0 * q for q in quants0], abs=1e-4)
