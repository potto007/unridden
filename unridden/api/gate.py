"""Calibrated margin, normalized entropy, and option-permutation escalation gate.

Implements Survey Items 1, 2, 3, and 4 (docs/research/open-model-survey.md) and
Proposal 5 (docs/research/intermediate-thinking-steps-review.md):

1. Per-(question_type, option_count) temperature scaling fit offline by NLL
   minimization on a log-spaced grid with [clamp_min, clamp_max] clamping and
   provenance binding.
2. Deterministic uncertainty signals: top-1/top-2 logit margin, probability
   margin, normalized Shannon entropy H(p)/ln(K), and entropy concentration
   1 - H(p)/ln(K).
3. Two-permutation option-order consistency gate: reverses/permutes option
   order on Choice and Score questions, realigns the returned probabilities to
   the canonical option order, and measures argmax flips, max probability move,
   and total variation distance (TVD).
4. Selective-prediction and paired-escalation evaluation metrics (ECE, equal-mass
   ECE, top-label Brier, multi-class Brier, gold NLL, tie-group-aware AURC, and
   paired primary-to-secondary rescue-vs-reversal curves).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from unridden.api.mapping import (
    entropy_concentration,
    logit_margin,
    normalized_entropy,
    probability_margin,
    stable_softmax,
)
from unridden.api.schema import (
    Answer,
    CalibrationStatus,
    ChoiceAnswer,
    ChoiceQuestion,
    GateAction,
    GateReason,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    StrictModel,
)

type QuestionKind = Literal["choice", "score", "noul"]


def bucket_key(question_type: QuestionKind, option_count: int) -> str:
    """Canonical bucket key for (question_type, option_count)."""
    if option_count < 2 or option_count > 255:
        raise ValueError("option_count must be between 2 and 255")
    return f"{question_type}:{option_count}"


class CalibrationSample(StrictModel):
    """One labeled readout observation used to fit or evaluate temperature scaling."""

    suite: str = Field(min_length=1)
    case: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_type: QuestionKind
    raw_label_logits: list[float] = Field(min_length=2, max_length=255)
    gold_index: int = Field(ge=0, le=254)
    coverage: float = Field(ge=0.0, le=1.0, default=1.0)

    @model_validator(mode="after")
    def _validate_sample(self) -> Self:
        if not all(math.isfinite(item) for item in self.raw_label_logits):
            raise ValueError("raw_label_logits must be finite")
        if self.gold_index >= len(self.raw_label_logits):
            raise ValueError("gold_index is out of range for raw_label_logits")
        if self.question_type == "noul" and len(self.raw_label_logits) != 2:
            raise ValueError("noul questions must have exactly two label logits")
        return self

    @property
    def option_count(self) -> int:
        return len(self.raw_label_logits)

    @property
    def bucket(self) -> str:
        return bucket_key(self.question_type, len(self.raw_label_logits))


class BucketCalibration(StrictModel):
    """Fitted temperature scaling parameters for one (type, count) bucket."""

    question_type: QuestionKind
    option_count: int = Field(ge=2, le=255)
    sample_count: int = Field(ge=1)
    raw_temperature: float = Field(gt=0.0)
    temperature: float = Field(ge=0.5, le=5.0)
    clamped: bool
    warning: str | None = None
    raw_nll: float = Field(ge=0.0)
    calibrated_nll: float = Field(ge=0.0)


class TemperatureCalibrationTable(StrictModel):
    """Per-bucket temperature table bound to model/runtime/prompt provenance."""

    status: CalibrationStatus = "fitted_requires_held_out_validation"
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prompt_version: str | None = None
    min_bucket_samples: int = Field(ge=1, default=10)
    default_temperature: float = Field(ge=0.5, le=5.0, default=1.0)
    buckets: dict[str, BucketCalibration] = Field(default_factory=dict)
    type_fallbacks: dict[QuestionKind, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def temperature_for(self, question_type: QuestionKind, option_count: int) -> float:
        key = bucket_key(question_type, option_count)
        if key in self.buckets:
            return self.buckets[key].temperature
        if question_type in self.type_fallbacks:
            return self.type_fallbacks[question_type]
        return self.default_temperature


class CalibratedUncertainty(StrictModel):
    """Raw and temperature-calibrated uncertainty metrics for one question readout."""

    question_type: QuestionKind
    option_count: int = Field(ge=2, le=255)
    temperature: float = Field(ge=0.5, le=5.0)
    coverage: float = Field(ge=0.0, le=1.0)
    raw_confidence: float = Field(ge=0.0, le=1.0)
    raw_probability_margin: float = Field(ge=0.0, le=1.0)
    raw_logit_margin: float = Field(ge=0.0)
    raw_normalized_entropy: float = Field(ge=0.0, le=1.0)
    raw_entropy_concentration: float = Field(ge=0.0, le=1.0)
    calibrated_probabilities: list[float] = Field(min_length=2, max_length=255)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    calibrated_probability_margin: float = Field(ge=0.0, le=1.0)
    scaled_logit_margin: float = Field(ge=0.0)
    calibrated_normalized_entropy: float = Field(ge=0.0, le=1.0)
    calibrated_entropy_concentration: float = Field(ge=0.0, le=1.0)


def nll_for_logits(
    samples: Sequence[tuple[Sequence[float], int]], temperature: float
) -> float:
    """Mean negative log-likelihood in nats at the given positive temperature."""
    if not samples:
        raise ValueError("cannot compute NLL on an empty sample sequence")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    total = 0.0
    for logits, gold_index in samples:
        if len(logits) < 2 or not (0 <= gold_index < len(logits)):
            raise ValueError("invalid logits or gold_index in sample")
        scaled = [float(item) / temperature for item in logits]
        top = max(scaled)
        log_denom = top + math.log(sum(math.exp(item - top) for item in scaled))
        total += max(0.0, log_denom - scaled[gold_index])
    return total / len(samples)


def fit_temperature(
    samples: Sequence[tuple[Sequence[float], int]],
    *,
    grid_min: float = 0.25,
    grid_max: float = 5.0,
    grid_points: int = 81,
    clamp_min: float = 1.0,
    clamp_max: float = 5.0,
) -> tuple[float, float, bool, str | None, float, float]:
    """Fit temperature by NLL minimization over a log-spaced grid + golden section.

    By default `clamp_min=1.0` and `clamp_max=5.0` per Survey Item 1: fitted
    temperatures below 1.0 sharpen an already overconfident distribution and are
    clamped with a warning.

    Returns `(raw_temp, clamped_temp, clamped, warning, raw_nll, calibrated_nll)`.
    """
    if not samples:
        raise ValueError("fit_temperature requires at least one sample")
    if grid_min <= 0.0 or grid_max <= grid_min or grid_points < 3:
        raise ValueError("invalid temperature grid bounds")
    if clamp_min < 0.5 or clamp_max > 5.0 or clamp_min > clamp_max:
        raise ValueError("clamp bounds must lie within [0.5, 5.0]")

    log_min = math.log(grid_min)
    log_max = math.log(grid_max)
    step = (log_max - log_min) / (grid_points - 1)
    grid = [math.exp(log_min + index * step) for index in range(grid_points)]
    losses = [nll_for_logits(samples, temp) for temp in grid]
    best_index = min(range(len(grid)), key=losses.__getitem__)

    # Refine in log-space around the best grid neighbor using golden-section search.
    left_log = math.log(grid[max(0, best_index - 1)])
    right_log = math.log(grid[min(len(grid) - 1, best_index + 1)])
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    x1 = left_log + resphi * (right_log - left_log)
    x2 = right_log - resphi * (right_log - left_log)
    f1 = nll_for_logits(samples, math.exp(x1))
    f2 = nll_for_logits(samples, math.exp(x2))
    for _ in range(24):
        if f1 < f2:
            right_log = x2
            x2, f2 = x1, f1
            x1 = left_log + resphi * (right_log - left_log)
            f1 = nll_for_logits(samples, math.exp(x1))
        else:
            left_log = x1
            x1, f1 = x2, f2
            x2 = right_log - resphi * (right_log - left_log)
            f2 = nll_for_logits(samples, math.exp(x2))

    raw_temp = math.exp(0.5 * (left_log + right_log))
    if losses[best_index] < nll_for_logits(samples, raw_temp):
        raw_temp = grid[best_index]

    raw_nll = nll_for_logits(samples, 1.0)
    clamped = False
    warning: str | None = None
    clamped_temp = raw_temp
    if raw_temp < clamp_min:
        clamped = True
        clamped_temp = clamp_min
        warning = (
            f"fitted temperature {raw_temp:.4f} < {clamp_min:.2f}; "
            f"clamped to {clamp_min:.2f}"
        )
    elif raw_temp > clamp_max:
        clamped = True
        clamped_temp = clamp_max
        warning = (
            f"fitted temperature {raw_temp:.4f} > {clamp_max:.2f}; "
            f"clamped to {clamp_max:.2f}"
        )
    calibrated_nll = nll_for_logits(samples, clamped_temp)
    return raw_temp, clamped_temp, clamped, warning, raw_nll, calibrated_nll


def fit_temperature_table(
    samples: Sequence[CalibrationSample],
    *,
    min_bucket_samples: int = 10,
    clamp_min: float = 1.0,
    clamp_max: float = 5.0,
    default_temperature: float = 1.0,
    model_sha256: str | None = None,
    runtime_sha256: str | None = None,
    prompt_version: str | None = None,
) -> TemperatureCalibrationTable:
    """Fit a per-(question_type, option_count) temperature calibration table."""
    if not samples:
        raise ValueError("fit_temperature_table requires at least one sample")
    by_bucket: dict[tuple[QuestionKind, int], list[CalibrationSample]] = {}
    by_type: dict[QuestionKind, list[CalibrationSample]] = {}
    for sample in samples:
        by_bucket.setdefault((sample.question_type, sample.option_count), []).append(
            sample
        )
        by_type.setdefault(sample.question_type, []).append(sample)

    warnings: list[str] = []
    type_fallbacks: dict[QuestionKind, float] = {}
    for kind, group in sorted(by_type.items()):
        pairs = [(item.raw_label_logits, item.gold_index) for item in group]
        _, temp, clamped, warn, _, _ = fit_temperature(
            pairs, clamp_min=clamp_min, clamp_max=clamp_max
        )
        type_fallbacks[kind] = temp
        if clamped and warn is not None:
            warnings.append(f"type {kind}: {warn}")

    buckets: dict[str, BucketCalibration] = {}
    for (kind, option_count), group in sorted(by_bucket.items()):
        key = bucket_key(kind, option_count)
        pairs = [(item.raw_label_logits, item.gold_index) for item in group]
        if len(group) < min_bucket_samples:
            fallback_temp = type_fallbacks.get(kind, default_temperature)
            raw_nll = nll_for_logits(pairs, 1.0)
            cal_nll = nll_for_logits(pairs, fallback_temp)
            msg = (
                f"bucket {key} has {len(group)} < {min_bucket_samples} samples; "
                f"using {kind} fallback temperature {fallback_temp:.4f}"
            )
            warnings.append(msg)
            buckets[key] = BucketCalibration(
                question_type=kind,
                option_count=option_count,
                sample_count=len(group),
                raw_temperature=fallback_temp,
                temperature=fallback_temp,
                clamped=True,
                warning=msg,
                raw_nll=raw_nll,
                calibrated_nll=cal_nll,
            )
            continue

        raw_t, clamped_t, clamped, warn, raw_nll, cal_nll = fit_temperature(
            pairs, clamp_min=clamp_min, clamp_max=clamp_max
        )
        bucket_warn: str | None = None
        if warn is not None:
            bucket_warn = f"bucket {key}: {warn}"
            warnings.append(bucket_warn)
        buckets[key] = BucketCalibration(
            question_type=kind,
            option_count=option_count,
            sample_count=len(group),
            raw_temperature=raw_t,
            temperature=clamped_t,
            clamped=clamped,
            warning=bucket_warn,
            raw_nll=raw_nll,
            calibrated_nll=cal_nll,
        )

    return TemperatureCalibrationTable(
        status="fitted_requires_held_out_validation",
        model_sha256=model_sha256,
        runtime_sha256=runtime_sha256,
        prompt_version=prompt_version,
        min_bucket_samples=min_bucket_samples,
        default_temperature=default_temperature,
        buckets=buckets,
        type_fallbacks=type_fallbacks,
        warnings=warnings,
    )


def calibrated_probabilities(
    logits: Sequence[float], temperature: float = 1.0
) -> list[float]:
    """Compute softmax(logits / temperature) with strict finiteness validation."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if len(logits) < 2 or not all(math.isfinite(item) for item in logits):
        raise ValueError("label logits must be finite and contain at least two values")
    return stable_softmax([float(item) / temperature for item in logits])


def calibrated_uncertainty(
    logits: Sequence[float],
    *,
    question_type: QuestionKind,
    coverage: float = 1.0,
    table: TemperatureCalibrationTable | None = None,
    temperature: float | None = None,
) -> CalibratedUncertainty:
    """Compute raw and temperature-calibrated margin and entropy metrics."""
    raw_probs = stable_softmax(logits)
    option_count = len(logits)
    resolved_temp = (
        temperature
        if temperature is not None
        else (
            table.temperature_for(question_type, option_count)
            if table is not None
            else 1.0
        )
    )
    cal_probs = calibrated_probabilities(logits, resolved_temp)
    raw_l_margin = logit_margin(logits)
    return CalibratedUncertainty(
        question_type=question_type,
        option_count=option_count,
        temperature=resolved_temp,
        coverage=coverage,
        raw_confidence=max(raw_probs),
        raw_probability_margin=probability_margin(raw_probs),
        raw_logit_margin=raw_l_margin,
        raw_normalized_entropy=normalized_entropy(raw_probs),
        raw_entropy_concentration=entropy_concentration(raw_probs),
        calibrated_probabilities=cal_probs,
        calibrated_confidence=max(cal_probs),
        calibrated_probability_margin=probability_margin(cal_probs),
        scaled_logit_margin=raw_l_margin / resolved_temp,
        calibrated_normalized_entropy=normalized_entropy(cal_probs),
        calibrated_entropy_concentration=entropy_concentration(cal_probs),
    )


def cross_fit_calibrated_uncertainties(
    samples: Sequence[CalibrationSample],
    *,
    min_bucket_samples: int = 10,
    clamp_min: float = 1.0,
    clamp_max: float = 5.0,
    default_temperature: float = 1.0,
) -> tuple[list[CalibratedUncertainty], dict[str, TemperatureCalibrationTable]]:
    """Leave-one-suite-out cross-fitting of per-bucket temperatures.

    Each suite's questions are calibrated exclusively using temperatures fitted
    on the remaining suites, preventing circular in-sample calibration claims.
    When only one suite is present, falls back to 5-fold cross-fitting by index.
    """
    if not samples:
        return [], {}
    suites = sorted({sample.suite for sample in samples})
    tables: dict[str, TemperatureCalibrationTable] = {}
    results: list[CalibratedUncertainty | None] = [None] * len(samples)

    if len(suites) >= 2:
        for held_out in suites:
            train = [sample for sample in samples if sample.suite != held_out]
            table = fit_temperature_table(
                train,
                min_bucket_samples=min_bucket_samples,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                default_temperature=default_temperature,
            ).model_copy(update={"status": "validated_on_held_out"})
            tables[held_out] = table
            for index, sample in enumerate(samples):
                if sample.suite == held_out:
                    results[index] = calibrated_uncertainty(
                        sample.raw_label_logits,
                        question_type=sample.question_type,
                        coverage=sample.coverage,
                        table=table,
                    )
    else:
        folds = min(5, len(samples))
        if folds < 2:
            table = fit_temperature_table(
                samples,
                min_bucket_samples=min_bucket_samples,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                default_temperature=default_temperature,
            )
            tables["all"] = table
            return [
                calibrated_uncertainty(
                    s.raw_label_logits,
                    question_type=s.question_type,
                    coverage=s.coverage,
                    table=table,
                )
                for s in samples
            ], tables
        for fold in range(folds):
            train = [
                sample for idx, sample in enumerate(samples) if idx % folds != fold
            ]
            table = fit_temperature_table(
                train,
                min_bucket_samples=min_bucket_samples,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                default_temperature=default_temperature,
            ).model_copy(update={"status": "validated_on_held_out"})
            tables[f"fold_{fold}"] = table
            for index, sample in enumerate(samples):
                if index % folds == fold:
                    results[index] = calibrated_uncertainty(
                        sample.raw_label_logits,
                        question_type=sample.question_type,
                        coverage=sample.coverage,
                        table=table,
                    )

    validated: list[CalibratedUncertainty] = []
    for item in results:
        if item is None:
            raise RuntimeError("cross-fit left an uncalibrated sample")
        validated.append(item)
    return validated, tables


# ----------------------------------------------------------------------------
# Two-Permutation Option-Order Consistency Gate
# ----------------------------------------------------------------------------


def is_permutable(question: Question) -> bool:
    """Return True if the question has caller-ordered options (Choice or Score)."""
    return isinstance(question, (ChoiceQuestion, ScoreQuestion))


def permute_choice_question(
    question: ChoiceQuestion, order: Sequence[str] | None = None
) -> ChoiceQuestion:
    """Permute (by default reverse) the criteria order of a ChoiceQuestion.

    Enforces Survey Item 3: the semantic option set must remain identical.
    """
    original_keys = list(question.criteria)
    target_order = list(reversed(original_keys)) if order is None else list(order)
    if len(target_order) != len(original_keys) or set(target_order) != set(
        original_keys
    ):
        raise ValueError(
            "permuted option order must contain the exact same option keys"
        )
    return ChoiceQuestion(
        type="choice",
        instructions=question.instructions,
        criteria={key: question.criteria[key] for key in target_order},
    )


def permute_score_question(question: ScoreQuestion) -> ScoreQuestion:
    """Reverse the criteria scale of a ScoreQuestion."""
    return ScoreQuestion(
        type="score",
        instructions=question.instructions,
        criteria=list(reversed(question.criteria)),
    )


def permute_question(
    question: Question, order: Sequence[str] | None = None
) -> Question:
    """Return a permuted copy of a Question (reversing options for Choice/Score)."""
    if isinstance(question, ChoiceQuestion):
        return permute_choice_question(question, order=order)
    if isinstance(question, ScoreQuestion):
        if order is not None:
            raise ValueError("custom key order applies only to ChoiceQuestion")
        return permute_score_question(question)
    if isinstance(question, NoulQuestion):
        if order is not None:
            raise ValueError("custom key order applies only to ChoiceQuestion")
        return question.model_copy(deep=True)
    raise TypeError(f"unsupported question type: {type(question)!r}")


def canonical_probabilities(question: Question, answer: Answer) -> dict[str, float]:
    """Extract canonical option-keyed probabilities in the question's option order."""
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        keys = list(question.criteria)
        if set(answer.probabilities) != set(keys):
            raise ValueError("ChoiceAnswer keys do not match ChoiceQuestion criteria")
        return {key: float(answer.probabilities[key]) for key in keys}
    if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        keys = [str(index) for index in range(len(question.criteria))]
        if set(answer.probabilities) != set(keys):
            raise ValueError("ScoreAnswer keys do not match ScoreQuestion levels")
        return {key: float(answer.probabilities[key]) for key in keys}
    if isinstance(question, NoulQuestion) and isinstance(answer, NoulAnswer):
        p_true = float(answer.noul)
        return {"true": p_true, "false": 1.0 - p_true}
    raise ValueError("question and answer types do not match")


def realign_probabilities(
    original_question: Question,
    permuted_question: Question,
    permuted_answer: Answer,
) -> dict[str, float]:
    """Realign permuted answer probabilities to the original option order."""
    if isinstance(original_question, ChoiceQuestion):
        if not isinstance(permuted_question, ChoiceQuestion) or not isinstance(
            permuted_answer, ChoiceAnswer
        ):
            raise ValueError("expected ChoiceQuestion and ChoiceAnswer for realignment")
        original_keys = list(original_question.criteria)
        if set(permuted_question.criteria) != set(original_keys):
            raise ValueError("permuted ChoiceQuestion changed the semantic option set")
        for key in original_keys:
            if permuted_question.criteria[key] != original_question.criteria[key]:
                raise ValueError(
                    f"permuted ChoiceQuestion altered description for option {key!r}"
                )
        if set(permuted_answer.probabilities) != set(original_keys):
            raise ValueError("permuted ChoiceAnswer keys do not match option set")
        return {key: float(permuted_answer.probabilities[key]) for key in original_keys}

    if isinstance(original_question, ScoreQuestion):
        if not isinstance(permuted_question, ScoreQuestion) or not isinstance(
            permuted_answer, ScoreAnswer
        ):
            raise ValueError("expected ScoreQuestion and ScoreAnswer for realignment")
        count = len(original_question.criteria)
        if len(permuted_question.criteria) != count:
            raise ValueError("permuted ScoreQuestion changed level count")
        is_reversed = list(reversed(original_question.criteria)) == list(
            permuted_question.criteria
        )
        is_identical = list(original_question.criteria) == list(
            permuted_question.criteria
        )
        if not (is_reversed or is_identical):
            raise ValueError(
                "permuted ScoreQuestion criteria must be either reversed or identical"
            )
        expected_keys = {str(index) for index in range(count)}
        if set(permuted_answer.probabilities) != expected_keys:
            raise ValueError("permuted ScoreAnswer keys do not match score levels")
        if is_reversed and not is_identical:
            return {
                str(index): float(permuted_answer.probabilities[str(count - 1 - index)])
                for index in range(count)
            }
        return {
            str(index): float(permuted_answer.probabilities[str(index)])
            for index in range(count)
        }

    if isinstance(original_question, NoulQuestion):
        if not isinstance(permuted_question, NoulQuestion) or not isinstance(
            permuted_answer, NoulAnswer
        ):
            raise ValueError("expected NoulQuestion and NoulAnswer for realignment")
        p_true = float(permuted_answer.noul)
        return {"true": p_true, "false": 1.0 - p_true}

    raise TypeError(f"unsupported question type: {type(original_question)!r}")


class PermutationStability(StrictModel):
    """Comparison between an original readout and its option-permuted probe."""

    original_top1: str | int | bool
    permuted_top1: str | int | bool
    flipped: bool
    max_probability_delta: float = Field(ge=0.0, le=1.0)
    total_variation_distance: float = Field(ge=0.0, le=1.0)
    realigned_probabilities: dict[str, float]
    mean_probabilities: dict[str, float]
    post_probe_margin: float = Field(ge=0.0, le=1.0)
    post_probe_entropy: float = Field(ge=0.0, le=1.0)


def _top1_of(
    question: Question, answer: Answer, realigned: Mapping[str, float]
) -> str | int | bool:
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        levels = range(len(realigned))
        return max(levels, key=lambda idx: realigned[str(idx)])
    if isinstance(question, NoulQuestion) and isinstance(answer, NoulAnswer):
        return answer.noul >= 0.5
    raise ValueError("mismatched question and answer types")


def compare_permutation_answers(
    original_question: Question,
    original_answer: Answer,
    permuted_question: Question,
    permuted_answer: Answer,
) -> PermutationStability:
    """Measure argmax flip and probability drift under option permutation."""
    orig_probs = canonical_probabilities(original_question, original_answer)
    perm_probs = realign_probabilities(
        original_question, permuted_question, permuted_answer
    )
    keys = list(orig_probs)
    orig_top1 = _top1_of(original_question, original_answer, orig_probs)
    if isinstance(original_question, ChoiceQuestion) and isinstance(
        permuted_answer, ChoiceAnswer
    ):
        perm_top1: str | int | bool = permuted_answer.choice
    else:
        perm_top1 = _top1_of(original_question, permuted_answer, perm_probs)

    deltas = [abs(orig_probs[key] - perm_probs[key]) for key in keys]
    max_delta = min(1.0, max(deltas, default=0.0))
    tvd = min(1.0, 0.5 * sum(deltas))
    mean_probs = {key: 0.5 * (orig_probs[key] + perm_probs[key]) for key in keys}
    mean_values = list(mean_probs.values())
    return PermutationStability(
        original_top1=orig_top1,
        permuted_top1=perm_top1,
        flipped=orig_top1 != perm_top1,
        max_probability_delta=max_delta,
        total_variation_distance=tvd,
        realigned_probabilities=perm_probs,
        mean_probabilities=mean_probs,
        post_probe_margin=probability_margin(mean_values),
        post_probe_entropy=normalized_entropy(mean_values),
    )


# ----------------------------------------------------------------------------
# Combined Calibrated Margin + Permutation Gate Decision Policy
# ----------------------------------------------------------------------------


class EscalationGateConfig(StrictModel):
    """Thresholds for the two-stage Calibrated Margin + Permutation Gate."""

    high_margin_threshold: float = Field(ge=0.0, le=1.0, default=0.60)
    low_margin_threshold: float = Field(ge=0.0, le=1.0, default=0.20)
    min_coverage: float = Field(ge=0.0, le=1.0, default=0.10)
    max_tvd_threshold: float = Field(ge=0.0, le=1.0, default=0.25)
    post_probe_margin_threshold: float = Field(ge=0.0, le=1.0, default=0.35)
    probe_before_escalate_on_low_margin: bool = False

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> Self:
        if self.low_margin_threshold > self.high_margin_threshold:
            raise ValueError(
                "low_margin_threshold must not exceed high_margin_threshold"
            )
        return self


class GateDecision(StrictModel):
    """Routing decision for a single question from the primary model."""

    question_id: str = Field(min_length=1)
    action: GateAction
    reason: GateReason
    uncertainty: CalibratedUncertainty
    stability: PermutationStability | None = None


def evaluate_question_gate(
    question_id: str,
    question: Question,
    logits: Sequence[float],
    *,
    coverage: float = 1.0,
    config: EscalationGateConfig | None = None,
    calibration: TemperatureCalibrationTable | None = None,
    original_answer: Answer | None = None,
    permuted_question: Question | None = None,
    permuted_answer: Answer | None = None,
) -> GateDecision:
    """Evaluate the two-stage Calibrated Margin + Permutation Gate for one question.

    Stage 1 (no permutation probe yet):
    - If `coverage < config.min_coverage`: escalate (`low_coverage`).
    - If `calibrated_probability_margin >= config.high_margin_threshold`:
      accept primary (`high_calibrated_margin`).
    - If `calibrated_probability_margin < config.low_margin_threshold`:
      if permutable and `probe_before_escalate_on_low_margin` is set, request
      a permutation probe (`borderline_margin`); otherwise escalate immediately
      (`low_calibrated_margin`).
    - Otherwise (`low_margin_threshold <= margin < high_margin_threshold`):
      if permutable, request `run_permutation_probe` (`borderline_margin`);
      if not permutable (e.g. `noul`), escalate (`low_calibrated_margin`).

    Stage 2 (when `original_answer`, `permuted_question`, and `permuted_answer`
    are provided):
    - Computes `PermutationStability`.
    - Escalates if `flipped` (`permutation_flip`), `total_variation_distance >
      config.max_tvd_threshold` (`permutation_tvd`), or `post_probe_margin <
      config.post_probe_margin_threshold` (`low_post_probe_margin`).
    - Otherwise accepts the primary model (`permutation_stable`).
    """
    cfg = config or EscalationGateConfig()
    uncertainty = calibrated_uncertainty(
        logits,
        question_type=question.type,
        coverage=coverage,
        table=calibration,
    )
    if coverage < cfg.min_coverage:
        return GateDecision(
            question_id=question_id,
            action="escalate",
            reason="low_coverage",
            uncertainty=uncertainty,
        )

    if (
        original_answer is not None
        and permuted_question is not None
        and permuted_answer is not None
    ):
        stability = compare_permutation_answers(
            question, original_answer, permuted_question, permuted_answer
        )
        if stability.flipped:
            return GateDecision(
                question_id=question_id,
                action="escalate",
                reason="permutation_flip",
                uncertainty=uncertainty,
                stability=stability,
            )
        if stability.total_variation_distance > cfg.max_tvd_threshold:
            return GateDecision(
                question_id=question_id,
                action="escalate",
                reason="permutation_tvd",
                uncertainty=uncertainty,
                stability=stability,
            )
        if stability.post_probe_margin < cfg.post_probe_margin_threshold:
            return GateDecision(
                question_id=question_id,
                action="escalate",
                reason="low_post_probe_margin",
                uncertainty=uncertainty,
                stability=stability,
            )
        return GateDecision(
            question_id=question_id,
            action="accept_primary",
            reason="permutation_stable",
            uncertainty=uncertainty,
            stability=stability,
        )

    margin = uncertainty.calibrated_probability_margin
    if margin >= cfg.high_margin_threshold:
        return GateDecision(
            question_id=question_id,
            action="accept_primary",
            reason="high_calibrated_margin",
            uncertainty=uncertainty,
        )
    if margin < cfg.low_margin_threshold:
        if cfg.probe_before_escalate_on_low_margin and is_permutable(question):
            return GateDecision(
                question_id=question_id,
                action="run_permutation_probe",
                reason="borderline_margin",
                uncertainty=uncertainty,
            )
        return GateDecision(
            question_id=question_id,
            action="escalate",
            reason="low_calibrated_margin",
            uncertainty=uncertainty,
        )

    if is_permutable(question):
        return GateDecision(
            question_id=question_id,
            action="run_permutation_probe",
            reason="borderline_margin",
            uncertainty=uncertainty,
        )
    return GateDecision(
        question_id=question_id,
        action="escalate",
        reason="low_calibrated_margin",
        uncertainty=uncertainty,
    )


def build_permutation_probe_questions(
    questions: Mapping[str, Question],
    stage1_decisions: Mapping[str, GateDecision],
) -> dict[str, Question]:
    """Build the subset of permuted questions needed for Stage 2 probing."""
    probe: dict[str, Question] = {}
    for qid, decision in stage1_decisions.items():
        if decision.action == "run_permutation_probe":
            question = questions[qid]
            if is_permutable(question):
                probe[qid] = permute_question(question)
    return probe


# ----------------------------------------------------------------------------
# Selective-Prediction & Paired-Escalation Metrics (Survey Items 2 & 4)
# ----------------------------------------------------------------------------


class RiskCoveragePoint(StrictModel):
    """One tie-group operating point on a selective-prediction risk-coverage curve."""

    threshold: float
    accepted_count: int = Field(ge=1)
    coverage: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)


class EscalationPoint(StrictModel):
    """One operating point on a paired primary-to-secondary escalation curve."""

    signal: str
    budget_questions: int = Field(ge=0)
    escalation_rate: float = Field(ge=0.0, le=1.0)
    threshold: float | None = None
    rescues: int = Field(ge=0)
    reversals: int = Field(ge=0)
    both_correct: int = Field(ge=0)
    both_wrong: int = Field(ge=0)
    net_gain: int
    rescue_to_reversal_ratio: float | None = None
    routed_correct: int = Field(ge=0)
    routed_accuracy: float = Field(ge=0.0, le=1.0)


def brier_toplabel(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error between top-label confidence and empirical correctness."""
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must be non-empty and equal length")
    return sum(
        (float(conf) - (1.0 if is_correct else 0.0)) ** 2
        for conf, is_correct in zip(confidences, correct, strict=True)
    ) / len(confidences)


def multiclass_brier(
    probabilities: Sequence[Mapping[str, float]], gold_keys: Sequence[str]
) -> float:
    """Mean multi-class Brier score sum_k (p_k - y_k)^2 across questions."""
    if len(probabilities) != len(gold_keys) or not probabilities:
        raise ValueError(
            "probabilities and gold_keys must be non-empty and equal length"
        )
    total = 0.0
    for dist, gold in zip(probabilities, gold_keys, strict=True):
        if gold not in dist:
            raise ValueError(f"gold key {gold!r} not present in distribution")
        for key, prob in dist.items():
            target = 1.0 if key == gold else 0.0
            total += (float(prob) - target) ** 2
    return total / len(probabilities)


def gold_nll(
    probabilities: Sequence[Mapping[str, float]], gold_keys: Sequence[str]
) -> float:
    """Mean negative log-likelihood of the gold option key."""
    if len(probabilities) != len(gold_keys) or not probabilities:
        raise ValueError(
            "probabilities and gold_keys must be non-empty and equal length"
        )
    total = 0.0
    for dist, gold in zip(probabilities, gold_keys, strict=True):
        if gold not in dist:
            raise ValueError(f"gold key {gold!r} not present in distribution")
        p_gold = max(float(dist[gold]), 1e-300)
        total += -math.log(p_gold)
    return total / len(probabilities)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
    equal_mass: bool = False,
) -> float:
    """Compute scalar Expected Calibration Error (fixed-width or equal-mass bins)."""
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must be non-empty and equal length")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    n = len(confidences)
    if not equal_mass:
        ece = 0.0
        for index in range(bins):
            members = [
                (float(conf), is_corr)
                for conf, is_corr in zip(confidences, correct, strict=True)
                if min(int(float(conf) * bins), bins - 1) == index
            ]
            if not members:
                continue
            acc = sum(1.0 for _, is_corr in members if is_corr) / len(members)
            mean_conf = sum(conf for conf, _ in members) / len(members)
            ece += (len(members) / n) * abs(acc - mean_conf)
        return ece

    # Equal-mass binning sorts by confidence and partitions into at most `bins`
    # slices while keeping exact confidence ties in the same bin.
    paired = sorted(
        zip((float(c) for c in confidences), correct, strict=True),
        key=lambda item: item[0],
    )
    target_size = max(1, n // bins)
    ece = 0.0
    index = 0
    while index < n:
        chunk: list[tuple[float, bool]] = []
        while index < n and (
            len(chunk) < target_size or paired[index][0] == chunk[-1][0]
        ):
            chunk.append(paired[index])
            index += 1
        acc = sum(1.0 for _, is_corr in chunk if is_corr) / len(chunk)
        mean_conf = sum(conf for conf, _ in chunk) / len(chunk)
        ece += (len(chunk) / n) * abs(acc - mean_conf)
    return ece


def risk_coverage_curve(
    scores: Sequence[float], correct: Sequence[bool]
) -> list[RiskCoveragePoint]:
    """Compute selective-prediction risk-coverage curve walking whole tie groups.

    Higher `scores` mean higher confidence / lower uncertainty. Walking whole
    tie groups (Survey Item 2) ensures saturated scores near 1.0 are never split
    by arbitrary row order.
    """
    if len(scores) != len(correct) or not scores:
        raise ValueError("scores and correct must be non-empty and equal length")
    n = len(scores)
    paired = sorted(
        zip((float(s) for s in scores), correct, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    curve: list[RiskCoveragePoint] = []
    accepted = 0
    correct_so_far = 0
    index = 0
    while index < n:
        threshold = paired[index][0]
        while index < n and paired[index][0] == threshold:
            accepted += 1
            if paired[index][1]:
                correct_so_far += 1
            index += 1
        accuracy = correct_so_far / accepted
        curve.append(
            RiskCoveragePoint(
                threshold=threshold,
                accepted_count=accepted,
                coverage=accepted / n,
                accuracy=accuracy,
                risk=1.0 - accuracy,
            )
        )
    return curve


def area_under_risk_coverage(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Compute Area Under the Risk-Coverage curve (AURC) via trapezoidal rule."""
    curve = risk_coverage_curve(scores, correct)
    if not curve:
        raise ValueError("empty risk-coverage curve")
    area = 0.0
    prev_cov = 0.0
    prev_risk = curve[0].risk
    for point in curve:
        delta_cov = point.coverage - prev_cov
        area += 0.5 * (prev_risk + point.risk) * delta_cov
        prev_cov = point.coverage
        prev_risk = point.risk
    return area


def coverage_at_risk(
    scores: Sequence[float],
    correct: Sequence[bool],
    *,
    max_risk: float = 0.05,
) -> float:
    """Maximum coverage achieved while keeping selective risk <= max_risk."""
    if not (0.0 <= max_risk <= 1.0):
        raise ValueError("max_risk must lie in [0.0, 1.0]")
    curve = risk_coverage_curve(scores, correct)
    eligible = [point.coverage for point in curve if point.risk <= max_risk]
    return max(eligible, default=0.0)


def evaluate_escalation_budget(
    signal_name: str,
    scores: Sequence[float],
    primary_correct: Sequence[bool],
    secondary_correct: Sequence[bool],
    *,
    budget_questions: int,
) -> EscalationPoint:
    """Evaluate routing the `budget_questions` lowest-scoring questions to secondary.

    When multiple questions tie at the boundary score, all tied questions are
    weighted by their exact expected share (or resolved deterministically by
    tie-group expectation; here we select the lowest-scoring items and break
    exact ties deterministically by index while reporting the threshold).
    """
    n = len(scores)
    if len(primary_correct) != n or len(secondary_correct) != n or n == 0:
        raise ValueError(
            "scores, primary_correct, and secondary_correct "
            "must have equal non-zero length"
        )
    if not (0 <= budget_questions <= n):
        raise ValueError("budget_questions must be between 0 and len(scores)")

    # Sort ascending by score: lowest confidence/margin is escalated first.
    order = sorted(range(n), key=lambda idx: (float(scores[idx]), idx))
    escalated_set = set(order[:budget_questions])
    threshold = (
        float(scores[order[budget_questions - 1]]) if budget_questions > 0 else None
    )

    rescues = 0
    reversals = 0
    both_correct = 0
    both_wrong = 0
    routed_correct = 0

    for idx in range(n):
        p_ok = bool(primary_correct[idx])
        s_ok = bool(secondary_correct[idx])
        if idx in escalated_set:
            if not p_ok and s_ok:
                rescues += 1
            elif p_ok and not s_ok:
                reversals += 1
            elif p_ok and s_ok:
                both_correct += 1
            else:
                both_wrong += 1
            if s_ok:
                routed_correct += 1
        else:
            if p_ok:
                routed_correct += 1

    return EscalationPoint(
        signal=signal_name,
        budget_questions=budget_questions,
        escalation_rate=budget_questions / n,
        threshold=threshold,
        rescues=rescues,
        reversals=reversals,
        both_correct=both_correct,
        both_wrong=both_wrong,
        net_gain=rescues - reversals,
        rescue_to_reversal_ratio=(rescues / reversals if reversals > 0 else None),
        routed_correct=routed_correct,
        routed_accuracy=routed_correct / n,
    )
