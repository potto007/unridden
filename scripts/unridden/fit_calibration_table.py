"""Fit a servable temperature table from one model's suite observations.

Reads the `observations.jsonl` that `run_usecase_suites.py --diagnostics`
writes, fits one temperature per (question_type, option_count) bucket with
`fit_temperature_table`, binds the table to the model, runtime and prompt
version every row reported, and writes it as JSON for
`unridden-api serve --calibration-table` (ADR 0012).

The written table carries `status: fitted_requires_held_out_validation`. Use
`evaluate_escalation_gate.py` for the leave-one-suite-out held-out numbers
before trusting its calibrated confidence on new traffic. Fitting is CPU-only
and touches no model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scripts.unridden.evaluate_escalation_gate import (
    _gold_key,
    _load_jsonl,
    _option_keys,
)
from unridden.api.gate import (
    CalibrationSample,
    QuestionKind,
    TemperatureCalibrationTable,
    fit_temperature_table,
)


def load_calibration_samples(
    path: Path,
) -> tuple[list[CalibrationSample], dict[str, str]]:
    """Samples plus the single provenance every observation row reported."""
    samples: list[CalibrationSample] = []
    provenance: dict[str, str] | None = None
    for row in _load_jsonl(path):
        if row["status"] != "ok":
            continue
        outcomes = {item["qid"]: item for item in row["outcomes"]}
        diagnostics = row["response"].get("diagnostics")
        if diagnostics is None:
            raise ValueError(
                f"{path}: case {row['suite']}/{row['case']} has no diagnostics; "
                "rerun the suites with --diagnostics"
            )
        for qid, question in row["request"]["questions"].items():
            kind: QuestionKind = question["type"]
            keys = _option_keys(question)
            gold = _gold_key(kind, outcomes[qid]["expected"])
            if gold not in keys:
                raise ValueError(f"gold key {gold!r} not in option keys {keys!r}")
            diag = diagnostics[qid]
            seen = {
                "model_sha256": str(diag["model_sha256"]),
                "runtime_sha256": str(diag["runtime_sha256"]),
                "prompt_version": str(diag["prompt_version"]),
            }
            if provenance is None:
                provenance = seen
            elif seen != provenance:
                raise ValueError(
                    f"{path}: observations mix provenance {provenance} and {seen}"
                )
            samples.append(
                CalibrationSample(
                    suite=str(row["suite"]),
                    case=str(row["case"]),
                    question_id=str(qid),
                    question_type=kind,
                    raw_label_logits=[float(x) for x in diag["raw_label_logits"]],
                    gold_index=keys.index(gold),
                    coverage=float(diag["coverage"]),
                )
            )
    if provenance is None or not samples:
        raise ValueError(f"{path}: no successful observations to fit")
    return samples, provenance


def fit_from_observations(
    path: Path, *, min_bucket_samples: int = 10
) -> TemperatureCalibrationTable:
    samples, provenance = load_calibration_samples(path)
    return fit_temperature_table(
        samples,
        min_bucket_samples=min_bucket_samples,
        model_sha256=provenance["model_sha256"],
        runtime_sha256=provenance["runtime_sha256"],
        prompt_version=provenance["prompt_version"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="observations.jsonl from run_usecase_suites.py --diagnostics",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write the TemperatureCalibrationTable JSON (must not exist)",
    )
    parser.add_argument(
        "--min-bucket-samples",
        type=int,
        default=10,
        help="buckets below this fall back to the per-type temperature",
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; choose a new file")
    try:
        table = fit_from_observations(
            args.observations, min_bucket_samples=args.min_bucket_samples
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(table.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "out": str(args.out),
        "status": table.status,
        "samples": sum(bucket.sample_count for bucket in table.buckets.values()),
        "buckets": {
            key: {
                "temperature": bucket.temperature,
                "samples": bucket.sample_count,
                "clamped": bucket.clamped,
            }
            for key, bucket in table.buckets.items()
        },
        "warnings": table.warnings,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
