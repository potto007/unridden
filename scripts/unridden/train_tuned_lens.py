"""Offline fitting and release-gate evaluation for the block-18 Tuned Lens head.

Fits a task-general affine residual translator (W h_18 + b ≈ h_30^pre-norm)
shrunk toward the identity residual stream (W = I + ΔW), together with the
exact or least-squares-recovered final RMSNorm weights and validated single-token
label unembedding rows on `split18-30-v1`. Evaluates release gates on held-out
(H18, H30, label_logits) observations before writing a `tuned-lens-v1` artifact.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unridden.api.snapshots.lens import (
    TunedLensArtifact,
    TunedLensGateConfig,
    extract_gguf_head_weights,
    fit_tuned_lens,
)

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class HiddenStateSample:
    h18: tuple[float, ...]
    h30: tuple[float, ...]
    last_normalized: tuple[float, ...] | None
    label_logits: tuple[float, ...]
    option_indices: tuple[int, ...] | None
    split: str


def _load_jsonl(path: Path) -> Iterator[Row]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_hidden_state_samples(
    path: Path,
    *,
    val_fraction: float = 0.25,
) -> tuple[list[HiddenStateSample], list[HiddenStateSample]]:
    """Load `(H18, H30, label_logits)` rows and partition into train and val splits."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    raw_rows = list(_load_jsonl(path))
    if len(raw_rows) < 2:
        raise ValueError("at least 2 samples are required to fit and validate a lens")

    samples: list[HiddenStateSample] = []
    n_total = len(raw_rows)
    val_cutoff = max(1, int(round(n_total * (1.0 - val_fraction))))

    for index, row in enumerate(raw_rows):
        split = str(row.get("split", "train" if index < val_cutoff else "val"))
        if split not in ("train", "val"):
            raise ValueError(f"unsupported split {split!r} at row {index}")
        last_norm = (
            tuple(float(x) for x in row["last_normalized"])
            if "last_normalized" in row and row["last_normalized"] is not None
            else None
        )
        opt_indices = (
            tuple(int(i) for i in row["option_indices"])
            if "option_indices" in row and row["option_indices"] is not None
            else None
        )
        samples.append(
            HiddenStateSample(
                h18=tuple(float(x) for x in row["h18"]),
                h30=tuple(float(x) for x in row["h30"]),
                last_normalized=last_norm,
                label_logits=tuple(float(x) for x in row["label_logits"]),
                option_indices=opt_indices,
                split=split,
            )
        )

    train = [item for item in samples if item.split == "train"]
    val = [item for item in samples if item.split == "val"]
    if not train or not val:
        raise ValueError("both train and val splits must be non-empty")
    return train, val


def train_and_evaluate_from_samples(
    train_samples: Sequence[HiddenStateSample],
    val_samples: Sequence[HiddenStateSample],
    *,
    model_sha256: str,
    label_token_ids: Sequence[int],
    ridge_alpha: float = 100.0,
    rms_norm_eps: float = 1e-6,
    final_logit_softcapping: float = 30.0,
    gate_config: TunedLensGateConfig | None = None,
    output_norm_weight: Sequence[float] | None = None,
    unembed_weight: Sequence[Sequence[float]] | None = None,
    min_top1_agreement: float = 0.85,
    max_mean_kl: float = 0.15,
    min_gated_agreement: float = 0.92,
    min_gated_coverage: float = 0.30,
) -> TunedLensArtifact:
    """Fit a `TunedLensArtifact` from parsed train/val samples."""
    train_h18 = [s.h18 for s in train_samples]
    train_h30 = [s.h30 for s in train_samples]
    val_h18 = [s.h18 for s in val_samples]
    val_h30 = [s.h30 for s in val_samples]
    val_logits = [s.label_logits for s in val_samples]

    all_samples = list(train_samples) + list(val_samples)
    norm_rows: list[tuple[float, ...]] | None = None
    if output_norm_weight is None:
        if any(s.last_normalized is None for s in all_samples):
            raise ValueError(
                "last_normalized is required when output_norm_weight is not provided"
            )
        norm_rows = [
            s.last_normalized for s in all_samples if s.last_normalized is not None
        ]

    all_logits = [s.label_logits for s in all_samples]
    all_h30 = [s.h30 for s in all_samples]
    option_subsets = (
        [list(s.option_indices) for s in val_samples if s.option_indices is not None]
        if all(s.option_indices is not None for s in val_samples)
        else None
    )

    return fit_tuned_lens(
        train_h18=train_h18,
        train_h30=train_h30,
        val_h18=val_h18,
        val_h30=val_h30,
        val_target_logits=val_logits,
        model_sha256=model_sha256,
        label_token_ids=label_token_ids,
        output_norm_weight=output_norm_weight,
        h30_normalized=norm_rows,
        unembed_weight=unembed_weight,
        all_h30_for_head=all_h30,
        all_target_logits_for_head=all_logits,
        ridge_alpha=ridge_alpha,
        rms_norm_eps=rms_norm_eps,
        final_logit_softcapping=final_logit_softcapping,
        gate=gate_config,
        option_subsets=option_subsets,
        min_top1_agreement=min_top1_agreement,
        max_mean_kl=max_mean_kl,
        min_gated_agreement=min_gated_agreement,
        min_gated_coverage=min_gated_coverage,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="JSONL file containing h18, h30, last_normalized, and label_logits rows",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="optional GGUF path to extract exact output_norm and label unembed rows",
    )
    parser.add_argument(
        "--model-sha256",
        required=True,
        help="64-hex SHA256 of the target model GGUF",
    )
    parser.add_argument(
        "--label-token-ids",
        type=str,
        required=True,
        help="comma-separated validated label token IDs (A..Z, 0..9, true, false)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to write the tuned-lens-v1 JSON artifact",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="optional path to write compact release-gate summary JSON",
    )
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--min-logit-margin", type=float, default=1.25)
    parser.add_argument("--min-entropy-concentration", type=float, default=0.35)
    parser.add_argument("--min-top1-agreement", type=float, default=0.85)
    parser.add_argument("--max-mean-kl", type=float, default=0.15)
    parser.add_argument("--min-gated-agreement", type=float, default=0.92)
    parser.add_argument("--min-gated-coverage", type=float, default=0.30)
    args = parser.parse_args()

    token_ids = [
        int(tok.strip()) for tok in args.label_token_ids.split(",") if tok.strip()
    ]
    train_samples, val_samples = load_hidden_state_samples(
        args.samples, val_fraction=args.val_fraction
    )
    output_norm_w: list[float] | None = None
    unembed_w: list[list[float]] | None = None
    if args.model is not None:
        n_embd = len(train_samples[0].h18)
        output_norm_w, unembed_w = extract_gguf_head_weights(
            args.model, token_ids, n_embd=n_embd
        )
    gate_cfg = TunedLensGateConfig(
        min_logit_margin=args.min_logit_margin,
        min_entropy_concentration=args.min_entropy_concentration,
    )
    artifact = train_and_evaluate_from_samples(
        train_samples,
        val_samples,
        model_sha256=args.model_sha256,
        label_token_ids=token_ids,
        ridge_alpha=args.ridge_alpha,
        rms_norm_eps=args.rms_norm_eps,
        final_logit_softcapping=args.softcap,
        gate_config=gate_cfg,
        output_norm_weight=output_norm_w,
        unembed_weight=unembed_w,
        min_top1_agreement=args.min_top1_agreement,
        max_mean_kl=args.max_mean_kl,
        min_gated_agreement=args.min_gated_agreement,
        min_gated_coverage=args.min_gated_coverage,
    )
    artifact.save(args.output)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": artifact.schema_version,
            "profile": artifact.profile,
            "model_sha256": artifact.model_sha256,
            "source_block": artifact.source_block,
            "target_block": artifact.target_block,
            "n_embd": artifact.n_embd,
            "ridge_alpha": artifact.ridge_alpha,
            "gate": artifact.gate.model_dump(),
            "release_gates": artifact.release_gates.model_dump(),
        }
        args.summary_output.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
