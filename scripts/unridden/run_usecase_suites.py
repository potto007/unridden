"""Run the use case suites against the non-generative API and score them.

This is an explicit GPU experiment driver, not a server. Every suite is loaded
and fully validated before the model is touched, each suite file hash is frozen
into run.json before any inference, and one backend serves all requests.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import statistics
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unridden.api.app import ApiConfig, ApiService, BackendFactory, _default_backend
from unridden.api.gate import (
    area_under_risk_coverage,
    brier_toplabel,
    expected_calibration_error,
)
from unridden.api.mapping import (
    entropy_concentration,
    normalized_entropy,
    probability_margin,
)
from unridden.api.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

Row = dict[str, Any]
Gold = str | bool | int
DIFFICULTIES = ("easy", "medium", "hard")
KINDS = ("choice", "score", "noul")
RELIABILITY_BINS = 5
NATIVE_LOGGER = "unridden.api.native"
LIMITATION = (
    "Suite labels are hand-authored gold answers frozen before inference. "
    "Probabilities are conditional label mass, not calibrated correctness."
)


class SuiteError(ValueError):
    """A suite file is unusable and the run must not start the model."""


@dataclass(frozen=True, slots=True)
class Target:
    qid: str
    kind: str
    expected: Gold
    tolerance: int = 0


@dataclass(frozen=True, slots=True)
class Ranking:
    name: str
    question_ids: tuple[str, ...]
    expected_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    difficulty: str
    rationale: str
    request: DecisionRequest
    targets: tuple[Target, ...]
    rankings: tuple[Ranking, ...]


@dataclass(frozen=True, slots=True)
class Suite:
    slug: str
    family: str
    source: tuple[str, ...]
    description: str
    path: Path
    sha256: str
    cases: tuple[Case, ...]


@dataclass(frozen=True, slots=True)
class Outcome:
    suite: str
    case: str
    difficulty: str
    qid: str
    kind: str
    expected: Gold
    got: Gold
    correct: bool
    confidence: float
    probabilities: dict[str, float]
    p_true: float | None = None
    score_value: float | None = None
    within_tolerance: bool | None = None

    def as_row(self) -> Row:
        row: Row = {
            "suite": self.suite,
            "case": self.case,
            "difficulty": self.difficulty,
            "qid": self.qid,
            "type": self.kind,
            "expected": self.expected,
            "got": self.got,
            "correct": self.correct,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }
        if self.kind == "score":
            row["score"] = self.score_value
            row["within_tolerance"] = self.within_tolerance
        return row


@dataclass(frozen=True, slots=True)
class RankOutcome:
    suite: str
    case: str
    difficulty: str
    name: str
    expected_order: tuple[str, ...]
    predicted_order: tuple[str, ...]
    top1_hit: bool
    concordant_pairs: int
    total_pairs: int
    probabilities: dict[str, float]

    def as_row(self) -> Row:
        return {
            "suite": self.suite,
            "case": self.case,
            "difficulty": self.difficulty,
            "ranking": self.name,
            "expected_order": list(self.expected_order),
            "predicted_order": list(self.predicted_order),
            "top1_hit": self.top1_hit,
            "concordant_pairs": self.concordant_pairs,
            "total_pairs": self.total_pairs,
            "probabilities": self.probabilities,
        }


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuiteError(message)


def _field(document: Row, key: str, kind: type, where: str) -> Any:
    _require(key in document, f"{where}: missing {key!r}")
    value = document[key]
    _require(
        isinstance(value, kind) and not isinstance(value, bool),
        f"{where}: {key!r} must be {kind.__name__}",
    )
    return value


def _target(qid: str, question: Question, raw: Any, where: str) -> Target:
    _require(isinstance(raw, dict), f"{where}: target for {qid!r} must be an object")
    kind = _field(raw, "type", str, where)
    _require(
        kind == question.type,
        f"{where}: target for {qid!r} is {kind!r} but the question is "
        f"{question.type!r}",
    )
    if kind == "choice":
        assert isinstance(question, ChoiceQuestion)
        expected = _field(raw, "expected", str, where)
        _require(
            expected in question.criteria,
            f"{where}: target for {qid!r} expects unknown option {expected!r}",
        )
        return Target(qid=qid, kind=kind, expected=expected)
    if kind == "noul":
        _require(
            "expected" in raw and isinstance(raw["expected"], bool),
            f"{where}: noul target for {qid!r} must expect true or false",
        )
        return Target(qid=qid, kind=kind, expected=bool(raw["expected"]))
    assert isinstance(question, ScoreQuestion)
    level = _field(raw, "expected_level", int, where)
    _require(
        0 <= level < len(question.criteria),
        f"{where}: score target for {qid!r} is level {level} outside "
        f"0..{len(question.criteria) - 1}",
    )
    tolerance = raw.get("tolerance", 0)
    _require(
        isinstance(tolerance, int)
        and not isinstance(tolerance, bool)
        and tolerance >= 0,
        f"{where}: score tolerance for {qid!r} must be a nonnegative integer",
    )
    return Target(qid=qid, kind=kind, expected=level, tolerance=tolerance)


def _ranking(raw: Any, questions: dict[str, Question], where: str) -> Ranking:
    _require(isinstance(raw, dict), f"{where}: ranking must be an object")
    name = _field(raw, "name", str, where)
    place = f"{where} ranking {name!r}"
    ids = _field(raw, "question_ids", list, place)
    order = _field(raw, "expected_order", list, place)
    _require(len(ids) >= 2, f"{place}: needs at least two question ids")
    _require(len(set(ids)) == len(ids), f"{place}: question ids must be unique")
    for qid in ids:
        _require(
            isinstance(qid, str) and qid in questions,
            f"{place}: unknown question id {qid!r}",
        )
        _require(
            questions[qid].type == "noul",
            f"{place}: question {qid!r} is not a noul question",
        )
    _require(bool(order), f"{place}: expected_order must not be empty")
    _require(len(set(order)) == len(order), f"{place}: expected_order must be unique")
    for qid in order:
        _require(qid in ids, f"{place}: expected_order entry {qid!r} is not ranked")
    return Ranking(name=name, question_ids=tuple(ids), expected_order=tuple(order))


def _case(raw: Any, where: str) -> Case:
    _require(isinstance(raw, dict), f"{where}: case must be an object")
    name = _field(raw, "name", str, where)
    place = f"{where} case {name!r}"
    difficulty = _field(raw, "difficulty", str, place)
    _require(
        difficulty in DIFFICULTIES,
        f"{place}: difficulty must be one of {', '.join(DIFFICULTIES)}",
    )
    rationale = _field(raw, "rationale", str, place)
    _require(bool(rationale.strip()), f"{place}: rationale must not be empty")
    payload = _field(raw, "request", dict, place)
    _require("model" not in payload, f"{place}: request must not set 'model'")
    try:
        request = DecisionRequest.model_validate(payload)
    except ValueError as error:
        raise SuiteError(f"{place}: request is invalid: {error}") from error
    targets = _field(raw, "targets", dict, place)
    _require(
        set(targets) == set(request.questions),
        f"{place}: targets {sorted(targets)} do not cover questions "
        f"{sorted(request.questions)}",
    )
    rankings = raw.get("rankings", [])
    _require(isinstance(rankings, list), f"{place}: rankings must be a list")
    parsed = tuple(_ranking(item, request.questions, place) for item in rankings)
    names = [ranking.name for ranking in parsed]
    _require(len(set(names)) == len(names), f"{place}: ranking names must be unique")
    return Case(
        name=name,
        difficulty=difficulty,
        rationale=rationale,
        request=request,
        targets=tuple(
            _target(qid, question, targets[qid], place)
            for qid, question in request.questions.items()
        ),
        rankings=parsed,
    )


def load_suite(path: Path) -> Suite:
    where = str(path)
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except json.JSONDecodeError as error:
        raise SuiteError(f"{where}: malformed JSON: {error}") from error
    _require(isinstance(document, dict), f"{where}: suite must be a JSON object")
    slug = _field(document, "suite", str, where)
    source = _field(document, "source", list, where)
    _require(
        bool(source) and all(isinstance(item, str) for item in source),
        f"{where}: source must be a nonempty list of strings",
    )
    cases = _field(document, "cases", list, where)
    _require(bool(cases), f"{where}: suite has no cases")
    parsed = tuple(_case(item, where) for item in cases)
    names = [case.name for case in parsed]
    _require(len(set(names)) == len(names), f"{where}: case names must be unique")
    return Suite(
        slug=slug,
        family=_field(document, "family", str, where),
        source=tuple(source),
        description=_field(document, "description", str, where),
        path=path,
        sha256=digest(path),
        cases=parsed,
    )


def collect_suite_paths(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            _require(path.is_file(), f"{path}: suite file does not exist")
            files.append(path)
    ordered = sorted(dict.fromkeys(files))
    _require(bool(ordered), "no suite files were found")
    return ordered


def load_suites(paths: Sequence[Path]) -> list[Suite]:
    suites = [load_suite(path) for path in collect_suite_paths(paths)]
    slugs = [suite.slug for suite in suites]
    _require(len(set(slugs)) == len(slugs), "suite slugs must be unique")
    return suites


def score_answer(suite: str, case: Case, target: Target, answer: Answer) -> Outcome:
    def outcome(**fields: Any) -> Outcome:
        return Outcome(
            suite=suite,
            case=case.name,
            difficulty=case.difficulty,
            qid=target.qid,
            kind=target.kind,
            expected=target.expected,
            **fields,
        )

    if isinstance(answer, ChoiceAnswer) and target.kind == "choice":
        return outcome(
            got=answer.choice,
            correct=answer.choice == target.expected,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
        )
    if isinstance(answer, NoulAnswer) and target.kind == "noul":
        predicted = answer.noul >= 0.5
        return outcome(
            got=predicted,
            correct=predicted == target.expected,
            confidence=max(answer.noul, 1.0 - answer.noul),
            probabilities={"true": answer.noul, "false": 1.0 - answer.noul},
            p_true=answer.noul,
        )
    if isinstance(answer, ScoreAnswer) and target.kind == "score":
        levels = range(len(answer.probabilities))
        # Ties resolve to the lowest level because max keeps the first maximum.
        level = max(levels, key=lambda index: answer.probabilities[str(index)])
        expected_level = int(target.expected)
        return outcome(
            got=level,
            correct=level == expected_level,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            score_value=answer.score,
            within_tolerance=abs(level - expected_level) <= target.tolerance,
        )
    raise ValueError(f"answer for {target.qid!r} does not match {target.kind!r}")


def score_ranking(
    suite: str, case: Case, ranking: Ranking, answers: Mapping[str, Answer]
) -> RankOutcome:
    probabilities: dict[str, float] = {}
    for qid in ranking.question_ids:
        answer = answers[qid]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"ranked question {qid!r} did not return a noul answer")
        probabilities[qid] = answer.noul
    order = tuple(
        sorted(
            ranking.question_ids,
            key=lambda qid: (-probabilities[qid], ranking.question_ids.index(qid)),
        )
    )
    rest = [qid for qid in ranking.question_ids if qid not in ranking.expected_order]
    pairs: list[tuple[str, str]] = []
    for index, better in enumerate(ranking.expected_order):
        pairs.extend((better, worse) for worse in ranking.expected_order[index + 1 :])
        pairs.extend((better, worse) for worse in rest)
    concordant = sum(
        probabilities[better] > probabilities[worse] for better, worse in pairs
    )
    return RankOutcome(
        suite=suite,
        case=case.name,
        difficulty=case.difficulty,
        name=ranking.name,
        expected_order=ranking.expected_order,
        predicted_order=order,
        top1_hit=order[0] == ranking.expected_order[0],
        concordant_pairs=concordant,
        total_pairs=len(pairs),
        probabilities=probabilities,
    )


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def latency_summary(values: Sequence[float]) -> Row:
    if not values:
        return {"n": 0, "median_ms": None, "p95_ms": None}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "n": len(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": p95,
    }


def reliability_table(outcomes: Sequence[Outcome]) -> list[Row]:
    table: list[Row] = []
    for index in range(RELIABILITY_BINS):
        low = index / RELIABILITY_BINS
        high = (index + 1) / RELIABILITY_BINS
        members = [
            outcome
            for outcome in outcomes
            if min(int(outcome.confidence * RELIABILITY_BINS), RELIABILITY_BINS - 1)
            == index
        ]
        table.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "count": len(members),
                "accuracy": ratio(sum(item.correct for item in members), len(members)),
                "mean_confidence": mean([item.confidence for item in members]),
            }
        )
    return table


def majority_baseline(outcomes: Sequence[Outcome]) -> Row:
    """Accuracy of always answering the most common gold label, per type."""
    result: Row = {}
    matched = 0
    for kind in KINDS:
        golds = [outcome.expected for outcome in outcomes if outcome.kind == kind]
        if not golds:
            result[kind] = None
            continue
        counts: dict[str, int] = {}
        for gold in golds:
            counts[json.dumps(gold)] = counts.get(json.dumps(gold), 0) + 1
        # Ties resolve to the lexicographically smallest encoded label.
        best = min(counts, key=lambda key: (-counts[key], key))
        matched += counts[best]
        result[kind] = {
            "label": json.loads(best),
            "n": len(golds),
            "accuracy": counts[best] / len(golds),
        }
    result["combined_accuracy"] = ratio(matched, len(outcomes))
    return result


def metric_block(outcomes: Sequence[Outcome], rankings: Sequence[RankOutcome]) -> Row:
    choice = [item for item in outcomes if item.kind == "choice"]
    noul = [item for item in outcomes if item.kind == "noul"]
    score = [item for item in outcomes if item.kind == "score"]
    true_targets = [item for item in noul if item.expected is True]
    false_targets = [item for item in noul if item.expected is False]
    pairs = sum(item.total_pairs for item in rankings)
    concordant = sum(item.concordant_pairs for item in rankings)
    return {
        "questions": len(outcomes),
        "correct": sum(item.correct for item in outcomes),
        "accuracy": ratio(sum(item.correct for item in outcomes), len(outcomes)),
        "choice": {
            "n": len(choice),
            "correct": sum(item.correct for item in choice),
            "accuracy": ratio(sum(item.correct for item in choice), len(choice)),
        },
        "noul": {
            "n": len(noul),
            "correct": sum(item.correct for item in noul),
            "accuracy": ratio(sum(item.correct for item in noul), len(noul)),
            "brier": mean(
                [
                    (float(item.p_true or 0.0) - float(bool(item.expected))) ** 2
                    for item in noul
                ]
            ),
            "mean_p_true_on_true_targets": mean(
                [float(item.p_true or 0.0) for item in true_targets]
            ),
            "mean_p_true_on_false_targets": mean(
                [float(item.p_true or 0.0) for item in false_targets]
            ),
        },
        "score": {
            "n": len(score),
            "exact": sum(item.correct for item in score),
            "exact_accuracy": ratio(sum(item.correct for item in score), len(score)),
            "within_tolerance": sum(bool(item.within_tolerance) for item in score),
            "within_tolerance_accuracy": ratio(
                sum(bool(item.within_tolerance) for item in score), len(score)
            ),
            "mean_absolute_error": mean(
                [
                    abs(float(item.score_value or 0.0) - float(int(item.expected)))
                    for item in score
                ]
            ),
        },
        "ranking": {
            "n": len(rankings),
            "top1_hits": sum(item.top1_hit for item in rankings),
            "top1_accuracy": ratio(
                sum(item.top1_hit for item in rankings), len(rankings)
            ),
            "pairs": pairs,
            "concordant_pairs": concordant,
            "pairwise_agreement": ratio(concordant, pairs),
        },
        "confidence": {
            "mean_on_correct": mean(
                [item.confidence for item in outcomes if item.correct]
            ),
            "mean_on_incorrect": mean(
                [item.confidence for item in outcomes if not item.correct]
            ),
        },
        "selective_prediction": _selective_prediction_block(outcomes),
        "reliability": reliability_table(outcomes),
        "majority_baseline": majority_baseline(outcomes),
    }


def _selective_prediction_block(outcomes: Sequence[Outcome]) -> Row:
    if not outcomes:
        return {
            "ece": None,
            "equal_mass_ece": None,
            "brier_toplabel": None,
            "aurc_confidence": None,
            "aurc_probability_margin": None,
            "aurc_entropy_concentration": None,
            "mean_margin_on_correct": None,
            "mean_margin_on_incorrect": None,
            "mean_normalized_entropy_on_correct": None,
            "mean_normalized_entropy_on_incorrect": None,
        }
    confidences = [item.confidence for item in outcomes]
    corrects = [item.correct for item in outcomes]
    margins = [
        probability_margin(list(item.probabilities.values())) for item in outcomes
    ]
    entropies = [
        normalized_entropy(list(item.probabilities.values())) for item in outcomes
    ]
    concentrations = [
        entropy_concentration(list(item.probabilities.values())) for item in outcomes
    ]
    return {
        "ece": expected_calibration_error(confidences, corrects, bins=RELIABILITY_BINS),
        "equal_mass_ece": expected_calibration_error(
            confidences, corrects, bins=RELIABILITY_BINS, equal_mass=True
        ),
        "brier_toplabel": brier_toplabel(confidences, corrects),
        "aurc_confidence": area_under_risk_coverage(confidences, corrects),
        "aurc_probability_margin": area_under_risk_coverage(margins, corrects),
        "aurc_entropy_concentration": area_under_risk_coverage(
            concentrations, corrects
        ),
        "mean_margin_on_correct": mean(
            [m for m, item in zip(margins, outcomes, strict=True) if item.correct]
        ),
        "mean_margin_on_incorrect": mean(
            [m for m, item in zip(margins, outcomes, strict=True) if not item.correct]
        ),
        "mean_normalized_entropy_on_correct": mean(
            [e for e, item in zip(entropies, outcomes, strict=True) if item.correct]
        ),
        "mean_normalized_entropy_on_incorrect": mean(
            [e for e, item in zip(entropies, outcomes, strict=True) if not item.correct]
        ),
    }


def build_summary(
    suites: Sequence[Suite],
    outcomes: Sequence[Outcome],
    rankings: Sequence[RankOutcome],
    errors: Sequence[Row],
    latencies: Sequence[float],
    usage: Row,
) -> Row:
    return {
        "suites": [
            {
                "suite": suite.slug,
                "family": suite.family,
                "source": list(suite.source),
                "description": suite.description,
                "path": str(suite.path),
                "sha256": suite.sha256,
                "cases": len(suite.cases),
                "questions": sum(len(case.targets) for case in suite.cases),
            }
            for suite in suites
        ],
        "totals": {
            "suites": len(suites),
            "cases": sum(len(suite.cases) for suite in suites),
            "errored_cases": len(errors),
            "scored_questions": len(outcomes),
            "rankings": len(rankings),
            **usage,
        },
        "request_latency_ms": latency_summary(latencies),
        "overall": metric_block(outcomes, rankings),
        "by_suite": {
            suite.slug: metric_block(
                [item for item in outcomes if item.suite == suite.slug],
                [item for item in rankings if item.suite == suite.slug],
            )
            for suite in suites
        },
        "by_difficulty": {
            difficulty: metric_block(
                [item for item in outcomes if item.difficulty == difficulty],
                [item for item in rankings if item.difficulty == difficulty],
            )
            for difficulty in DIFFICULTIES
        },
        "failed_questions": [
            outcome.as_row() for outcome in outcomes if not outcome.correct
        ],
        "failed_rankings": [
            ranking.as_row()
            for ranking in rankings
            if not ranking.top1_hit or ranking.concordant_pairs != ranking.total_pairs
        ],
        "errored_cases": list(errors),
        "limitation": LIMITATION,
    }


@contextlib.contextmanager
def native_logging(paths: Sequence[Path]) -> Iterator[None]:
    """Native startup, offload, and error lines arrive only at DEBUG."""
    logger = logging.getLogger(NATIVE_LOGGER)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    handlers: list[logging.FileHandler] = []
    try:
        for path in paths:
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
            logger.addHandler(handler)
            handlers.append(handler)
        yield
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous)


def _git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip()


def _config(args: argparse.Namespace) -> ApiConfig:
    return ApiConfig(
        model_path=args.model_path,
        worker_path=args.worker,
        manifest_path=args.manifest,
        gpu=True,
        batch_size=args.batch_size,
        ubatch_size=args.batch_size,
        share_prefix=not args.no_share_prefix,
        batched=args.batched,
        batched_context=args.batched_context,
    )


async def run(
    args: argparse.Namespace,
    *,
    backend_factory: BackendFactory | None = None,
) -> int:
    args.out.mkdir(parents=True, exist_ok=False)
    manifest: Row = {"state": "starting", "started_unix": time.time()}
    observations = args.out / "observations.jsonl"
    service: ApiService | None = None
    log_paths = [args.out / "native-worker.log"]
    if args.extra_log is not None:
        log_paths.append(args.extra_log)
    with native_logging(log_paths):
        try:
            # Labels are frozen before any model load: validate, then hash.
            suites = load_suites(args.suites)
            manifest["suites"] = [
                {
                    "suite": suite.slug,
                    "path": str(suite.path),
                    "sha256": suite.sha256,
                    "cases": len(suite.cases),
                }
                for suite in suites
            ]
            manifest["cases"] = sum(len(suite.cases) for suite in suites)
            manifest["questions"] = sum(
                len(case.targets) for suite in suites for case in suite.cases
            )
            config = _config(args)
            manifest["batched"] = config.batched
            # A sequential worker has no batched cache; recording the unused
            # default here would misattribute its VRAM figure.
            manifest["batched_context"] = (
                config.batched_context if config.batched else 0
            )
            manifest["batch_size"] = config.batch_size
            manifest["share_prefix"] = config.share_prefix
            for key, path in (
                ("worker_sha256", config.worker_path),
                ("build_manifest_sha256", config.manifest_path),
                ("model_sha256", config.model_path),
            ):
                manifest[key] = await asyncio.to_thread(digest, path)
            manifest["commit"] = await asyncio.to_thread(_git_commit)
            write_json(args.out / "run.json", manifest)

            factory = backend_factory or _default_backend
            service = ApiService(factory(config), config)
            started = time.perf_counter()
            await service.start()
            manifest["startup_seconds"] = time.perf_counter() - started
            manifest["state"] = "running"
            manifest["worker_pid"] = getattr(service.backend, "pid", None)
            write_json(args.out / "run.json", manifest)

            outcomes: list[Outcome] = []
            rankings: list[RankOutcome] = []
            errors: list[Row] = []
            latencies: list[float] = []
            generated_tokens = 0
            input_tokens = 0
            index = 0
            for suite in suites:
                for case in suite.cases:
                    request_started = time.perf_counter()
                    response: DecisionResponse | None = None
                    failure: str | None = None
                    try:
                        response = await service.evaluate(
                            case.request, diagnostics=True
                        )
                    except (ValueError, RuntimeError, OSError) as error:
                        # A rejected request is evidence, not a reason to stop.
                        # Every API failure mode is one of these three, and the
                        # backend errors and BusyError all derive from them.
                        failure = f"{type(error).__name__}: {error}"
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    record: Row = {
                        "index": index,
                        "suite": suite.slug,
                        "case": case.name,
                        "difficulty": case.difficulty,
                        "rationale": case.rationale,
                        "request": case.request.model_dump(mode="json"),
                        "latency_ms": elapsed_ms,
                    }
                    index += 1
                    if response is None:
                        record["status"] = "error"
                        record["error"] = failure
                        errors.append(
                            {
                                "suite": suite.slug,
                                "case": case.name,
                                "difficulty": case.difficulty,
                                "error": failure,
                            }
                        )
                    else:
                        latencies.append(elapsed_ms)
                        generated_tokens += response.usage.output_tokens
                        input_tokens += response.usage.input_tokens
                        case_outcomes = [
                            score_answer(
                                suite.slug,
                                case,
                                target,
                                response.answers[target.qid],
                            )
                            for target in case.targets
                        ]
                        case_rankings = [
                            score_ranking(suite.slug, case, ranking, response.answers)
                            for ranking in case.rankings
                        ]
                        outcomes.extend(case_outcomes)
                        rankings.extend(case_rankings)
                        record["status"] = "ok"
                        record["response"] = response.model_dump(
                            mode="json", exclude_none=True
                        )
                        record["outcomes"] = [item.as_row() for item in case_outcomes]
                        record["rankings"] = [item.as_row() for item in case_rankings]
                    with observations.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                print("SUITE_COMPLETE", suite.slug, flush=True)

            if generated_tokens != 0:
                raise RuntimeError("the API reported generated tokens")
            summary = build_summary(
                suites,
                outcomes,
                rankings,
                errors,
                latencies,
                {
                    "generated_tokens": generated_tokens,
                    "input_tokens": input_tokens,
                },
            )
            write_json(args.out / "summary.json", summary)
            manifest["state"] = "complete"
            manifest["errored_cases"] = len(errors)
            accuracy = summary["overall"]["accuracy"]
            print(
                "USECASE_RUN_COMPLETE",
                f"questions={len(outcomes)}",
                f"accuracy={accuracy if accuracy is None else round(accuracy, 4)}",
                f"errored_cases={len(errors)}",
                flush=True,
            )
            return 0
        except BaseException as error:
            manifest["state"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            tail = getattr(getattr(service, "backend", None), "stderr_tail", ())
            write_json(args.out / "native-stderr-tail.json", list(tail))
            print("USECASE_RUN_FAILED", type(error).__name__, error, flush=True)
            if isinstance(error, Exception):
                return 2
            raise
        finally:
            if service is not None:
                await service.close()
            manifest["finished_unix"] = time.time()
            write_json(args.out / "run.json", manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = ApiConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites",
        type=Path,
        nargs="+",
        required=True,
        help="suite directories or suite JSON files",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=defaults.worker_path)
    parser.add_argument("--manifest", type=Path, default=defaults.manifest_path)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--no-share-prefix", action="store_true")
    parser.add_argument(
        "--batched",
        action="store_true",
        help="evaluate each request's questions in one batched decode",
    )
    parser.add_argument(
        "--batched-context",
        type=int,
        default=defaults.batched_context,
        help="KV cells a batched worker reserves, about 0.21 MiB each",
    )
    parser.add_argument(
        "--extra-log",
        type=Path,
        default=None,
        help="also append native worker debug lines to this file",
    )
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_gpu:
        parser.error("Explicit --allow-gpu is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
