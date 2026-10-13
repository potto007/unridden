"""Offline evaluation of per-bucket temperature calibration and escalation gating.

Loads paired primary (e.g. Gemma 4 E4B) and secondary (e.g. Gemma 4 26B-A4B)
`observations.jsonl` files from `run_usecase_suites.py`, performs leave-one-suite-out
cross-fitted per-(question_type, option_count) temperature calibration, and
compares selective-prediction (ECE, Brier, NLL, AURC) and paired escalation
(rescues, reversals, net gain, routed accuracy) across raw and calibrated
margin/entropy signals.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unridden.api.gate import (
    CalibrationSample,
    EscalationPoint,
    QuestionKind,
    TemperatureCalibrationTable,
    area_under_risk_coverage,
    brier_toplabel,
    coverage_at_risk,
    cross_fit_calibrated_uncertainties,
    evaluate_escalation_budget,
    expected_calibration_error,
    fit_temperature_table,
    gold_nll,
    multiclass_brier,
)

Row = dict[str, Any]
DEFAULT_BUDGETS = (25, 50, 87, 100, 128, 150, 200, 250)


@dataclass(frozen=True, slots=True)
class PairedQuestion:
    suite: str
    case: str
    difficulty: str
    question_id: str
    question_type: QuestionKind
    option_keys: tuple[str, ...]
    gold_key: str
    gold_index: int
    primary_logits: tuple[float, ...]
    primary_coverage: float
    primary_correct: bool
    primary_within_tolerance: bool
    secondary_logits: tuple[float, ...]
    secondary_coverage: float
    secondary_correct: bool
    secondary_within_tolerance: bool
    model_sha256: str
    runtime_sha256: str
    prompt_version: str


def _load_jsonl(path: Path) -> Iterator[Row]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _option_keys(question: Row) -> tuple[str, ...]:
    kind = question["type"]
    if kind == "choice":
        return tuple(str(key) for key in question["criteria"])
    if kind == "score":
        return tuple(str(index) for index in range(len(question["criteria"])))
    if kind == "noul":
        return ("true", "false")
    raise ValueError(f"unknown question type {kind!r}")


def _gold_key(kind: QuestionKind, expected: Any) -> str:
    if kind == "noul":
        return "true" if bool(expected) else "false"
    return str(expected)


def load_paired_questions(
    primary_path: Path, secondary_path: Path
) -> list[PairedQuestion]:
    """Load and align paired primary and secondary observation rows."""
    primary_rows = {
        (row["suite"], row["case"]): row for row in _load_jsonl(primary_path)
    }
    secondary_rows = {
        (row["suite"], row["case"]): row for row in _load_jsonl(secondary_path)
    }
    if set(primary_rows) != set(secondary_rows):
        raise ValueError("primary and secondary runs do not cover the same cases")

    paired: list[PairedQuestion] = []
    for key in sorted(primary_rows):
        p_row = primary_rows[key]
        s_row = secondary_rows[key]
        if p_row["status"] != "ok" or s_row["status"] != "ok":
            raise ValueError(f"case {key} did not succeed in both runs")

        p_outcomes = {item["qid"]: item for item in p_row["outcomes"]}
        s_outcomes = {item["qid"]: item for item in s_row["outcomes"]}
        questions = p_row["request"]["questions"]

        for qid, question in questions.items():
            kind: QuestionKind = question["type"]
            keys = _option_keys(question)
            p_out = p_outcomes[qid]
            s_out = s_outcomes[qid]
            g_key = _gold_key(kind, p_out["expected"])
            if g_key not in keys:
                raise ValueError(f"gold key {g_key!r} not in option keys {keys!r}")
            g_idx = keys.index(g_key)

            p_diag = p_row["response"]["diagnostics"][qid]
            s_diag = s_row["response"]["diagnostics"][qid]
            paired.append(
                PairedQuestion(
                    suite=p_row["suite"],
                    case=p_row["case"],
                    difficulty=p_row["difficulty"],
                    question_id=qid,
                    question_type=kind,
                    option_keys=keys,
                    gold_key=g_key,
                    gold_index=g_idx,
                    primary_logits=tuple(float(x) for x in p_diag["raw_label_logits"]),
                    primary_coverage=float(p_diag["coverage"]),
                    primary_correct=bool(p_out["correct"]),
                    primary_within_tolerance=bool(
                        p_out.get("within_tolerance", p_out["correct"])
                    ),
                    secondary_logits=tuple(
                        float(x) for x in s_diag["raw_label_logits"]
                    ),
                    secondary_coverage=float(s_diag["coverage"]),
                    secondary_correct=bool(s_out["correct"]),
                    secondary_within_tolerance=bool(
                        s_out.get("within_tolerance", s_out["correct"])
                    ),
                    model_sha256=str(p_diag["model_sha256"]),
                    runtime_sha256=str(p_diag["runtime_sha256"]),
                    prompt_version=str(p_diag["prompt_version"]),
                )
            )
    return paired


def evaluate_paired_runs(
    primary_path: Path,
    secondary_path: Path,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    min_bucket_samples: int = 10,
) -> tuple[Row, TemperatureCalibrationTable]:
    """Run cross-fitted calibration, selective prediction, and escalation analysis."""
    paired = load_paired_questions(primary_path, secondary_path)
    if not paired:
        raise ValueError("no paired questions found")

    samples = [
        CalibrationSample(
            suite=item.suite,
            case=item.case,
            question_id=item.question_id,
            question_type=item.question_type,
            raw_label_logits=list(item.primary_logits),
            gold_index=item.gold_index,
            coverage=item.primary_coverage,
        )
        for item in paired
    ]

    full_table = fit_temperature_table(
        samples,
        min_bucket_samples=min_bucket_samples,
        model_sha256=paired[0].model_sha256,
        runtime_sha256=paired[0].runtime_sha256,
        prompt_version=paired[0].prompt_version,
    )
    cross_uncertainties, fold_tables = cross_fit_calibrated_uncertainties(
        samples,
        min_bucket_samples=min_bucket_samples,
    )

    primary_correct = [item.primary_correct for item in paired]
    secondary_correct = [item.secondary_correct for item in paired]
    gold_keys = [item.gold_key for item in paired]

    raw_dists = [
        dict(
            zip(
                item.option_keys,
                unc.calibrated_probabilities
                if unc.temperature == 1.0
                else [
                    # Reconstruct raw distribution from logits
                    p
                    for p in _raw_probs(item.primary_logits)
                ],
                strict=True,
            )
        )
        for item, unc in zip(paired, cross_uncertainties, strict=True)
    ]
    cal_dists = [
        dict(zip(item.option_keys, unc.calibrated_probabilities, strict=True))
        for item, unc in zip(paired, cross_uncertainties, strict=True)
    ]

    raw_conf = [unc.raw_confidence for unc in cross_uncertainties]
    cal_conf = [unc.calibrated_confidence for unc in cross_uncertainties]

    # Composite calibrated gate score: combines calibrated entropy concentration
    # and calibrated probability margin, and downweights ordinal `score`
    # escalations when top-1 and top-2 are adjacent levels (within tolerance=1),
    # where 26B rubric-framing shifts cause frequent exact-level reversals.
    composite_scores: list[float] = []
    for item, unc in zip(paired, cross_uncertainties, strict=True):
        base = 0.5 * (
            unc.calibrated_probability_margin + unc.calibrated_entropy_concentration
        )
        if item.question_type == "score" and len(item.primary_logits) >= 2:
            ordered_indices = sorted(
                range(len(item.primary_logits)),
                key=item.primary_logits.__getitem__,
                reverse=True,
            )
            if abs(ordered_indices[0] - ordered_indices[1]) == 1:
                # Adjacent score levels share neighboring ordinal semantics;
                # boost stability score so borderline adjacent-level scores are
                # not escalated ahead of choice/noul errors.
                base = min(1.0, base + 0.25)
        composite_scores.append(base)

    signals: dict[str, list[float]] = {
        "raw_confidence": raw_conf,
        "raw_probability_margin": [
            unc.raw_probability_margin for unc in cross_uncertainties
        ],
        "raw_logit_margin": [unc.raw_logit_margin for unc in cross_uncertainties],
        "raw_entropy_concentration": [
            unc.raw_entropy_concentration for unc in cross_uncertainties
        ],
        "calibrated_confidence": cal_conf,
        "calibrated_probability_margin": [
            unc.calibrated_probability_margin for unc in cross_uncertainties
        ],
        "scaled_logit_margin": [unc.scaled_logit_margin for unc in cross_uncertainties],
        "calibrated_entropy_concentration": [
            unc.calibrated_entropy_concentration for unc in cross_uncertainties
        ],
        "composite_calibrated_gate": composite_scores,
    }

    selective_metrics: dict[str, Row] = {}
    for name, values in signals.items():
        selective_metrics[name] = {
            "aurc": area_under_risk_coverage(values, primary_correct),
            "coverage_at_5pct_risk": coverage_at_risk(
                values, primary_correct, max_risk=0.05
            ),
            "coverage_at_10pct_risk": coverage_at_risk(
                values, primary_correct, max_risk=0.10
            ),
        }

    # Evaluate the exact retrospective `< 0.8` raw confidence gate from
    # docs/results/gemma4-e4b-vs-26b-usecases.md alongside matched-budget curves.
    raw_lt_08_indices = [idx for idx, conf in enumerate(raw_conf) if conf < 0.8]
    raw_lt_08_budget = len(raw_lt_08_indices)

    active_budgets = sorted({*budgets, raw_lt_08_budget})
    escalation_by_signal: dict[str, list[Row]] = {}
    for name, values in signals.items():
        points: list[EscalationPoint] = [
            evaluate_escalation_budget(
                name,
                values,
                primary_correct,
                secondary_correct,
                budget_questions=budget,
            )
            for budget in active_budgets
            if 0 <= budget <= len(paired)
        ]
        escalation_by_signal[name] = [point.model_dump() for point in points]

    total_rescues = sum(
        not p and s for p, s in zip(primary_correct, secondary_correct, strict=True)
    )
    total_reversals = sum(
        p and not s for p, s in zip(primary_correct, secondary_correct, strict=True)
    )

    report: Row = {
        "primary_path": str(primary_path),
        "secondary_path": str(secondary_path),
        "questions": len(paired),
        "suites": len({item.suite for item in paired}),
        "baseline": {
            "primary_correct": sum(primary_correct),
            "primary_accuracy": sum(primary_correct) / len(paired),
            "secondary_correct": sum(secondary_correct),
            "secondary_accuracy": sum(secondary_correct) / len(paired),
            "full_escalation_rescues": total_rescues,
            "full_escalation_reversals": total_reversals,
            "full_escalation_net_gain": total_rescues - total_reversals,
            "raw_confidence_ge_0_8_among_primary_errors": sum(
                1
                for is_corr, conf in zip(primary_correct, raw_conf, strict=True)
                if not is_corr and conf >= 0.8
            ),
            "primary_errors": sum(not is_corr for is_corr in primary_correct),
        },
        "calibration_held_out_leave_one_suite_out": {
            "folds": len(fold_tables),
            "raw": {
                "ece_10bin": expected_calibration_error(
                    raw_conf, primary_correct, bins=10
                ),
                "equal_mass_ece_10bin": expected_calibration_error(
                    raw_conf, primary_correct, bins=10, equal_mass=True
                ),
                "brier_toplabel": brier_toplabel(raw_conf, primary_correct),
                "multiclass_brier": multiclass_brier(raw_dists, gold_keys),
                "gold_nll": gold_nll(raw_dists, gold_keys),
            },
            "calibrated": {
                "ece_10bin": expected_calibration_error(
                    cal_conf, primary_correct, bins=10
                ),
                "equal_mass_ece_10bin": expected_calibration_error(
                    cal_conf, primary_correct, bins=10, equal_mass=True
                ),
                "brier_toplabel": brier_toplabel(cal_conf, primary_correct),
                "multiclass_brier": multiclass_brier(cal_dists, gold_keys),
                "gold_nll": gold_nll(cal_dists, gold_keys),
            },
        },
        "full_temperature_table": full_table.model_dump(),
        "selective_prediction": selective_metrics,
        "matched_budget_at_raw_lt_0_8": {
            "budget_questions": raw_lt_08_budget,
            "by_signal": {
                name: next(
                    row for row in rows if row["budget_questions"] == raw_lt_08_budget
                )
                for name, rows in escalation_by_signal.items()
            },
        },
        "escalation_curves": escalation_by_signal,
    }
    return report, full_table


def _raw_probs(logits: Sequence[float]) -> list[float]:
    from unridden.api.mapping import stable_softmax

    return stable_softmax(logits)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        required=True,
        help="Primary model observations.jsonl (e.g. Gemma 4 E4B)",
    )
    parser.add_argument(
        "--secondary",
        type=Path,
        required=True,
        help="Secondary model observations.jsonl (e.g. Gemma 4 26B-A4B)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write full JSON evaluation report",
    )
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=None,
        help="Optional path to write fitted TemperatureCalibrationTable JSON",
    )
    args = parser.parse_args(argv)
    report, table = evaluate_paired_runs(args.primary, args.secondary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.calibration_out is not None:
        args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_out.write_text(
            json.dumps(table.model_dump(), indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "questions": report["questions"],
        "baseline": report["baseline"],
        "calibration_held_out_leave_one_suite_out": report[
            "calibration_held_out_leave_one_suite_out"
        ],
        "selective_prediction": report["selective_prediction"],
        "matched_budget_at_raw_lt_0_8": report["matched_budget_at_raw_lt_0_8"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
