"""Unit and contract tests for calibrated margin, entropy, and permutation gating."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.unridden.evaluate_escalation_gate import evaluate_paired_runs
from unridden.api.compiler import compile_request
from unridden.api.gate import (
    CalibrationSample,
    EscalationGateConfig,
    area_under_risk_coverage,
    brier_toplabel,
    build_permutation_probe_questions,
    calibrated_probabilities,
    compare_permutation_answers,
    coverage_at_risk,
    cross_fit_calibrated_uncertainties,
    evaluate_escalation_budget,
    evaluate_question_gate,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_table,
    gold_nll,
    is_permutable,
    multiclass_brier,
    nll_for_logits,
    permute_choice_question,
    permute_question,
    permute_score_question,
    realign_probabilities,
    risk_coverage_curve,
)
from unridden.api.mapping import (
    entropy_concentration,
    logit_margin,
    normalized_entropy,
    probability_margin,
)
from unridden.api.native.build import TESTED_LLAMA_REVISION
from unridden.api.schema import (
    BackendProfile,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

PROFILE = BackendProfile(
    model_id="local-gemma-unridden-v1",
    model_name="fixture",
    model_sha256="a" * 64,
    runtime_sha256="b" * 64,
    llama_revision=TESTED_LLAMA_REVISION,
    tested_revision=True,
    labels=["A", "B", "C", "D"],
    label_token_ids=[101, 102, 103, 104],
    context_size=2048,
    batch_size=256,
    ubatch_size=256,
    threads=8,
    max_questions=32,
    batched_mode=False,
    batched_context=0,
    generated_tokens=0,
    callbacks_enabled=False,
    execution_mode="full",
)


def test_uncertainty_functions_distinguish_cardinality_and_concentration() -> None:
    # A 2-way [0.6, 0.4] and a 6-way [0.6, 0.08, 0.08, 0.08, 0.08, 0.08] share
    # identical max_probability (0.6), but very different margins and entropies.
    two_way = [0.6, 0.4]
    six_way = [0.6, 0.08, 0.08, 0.08, 0.08, 0.08]

    assert probability_margin(two_way) == pytest.approx(0.2)
    assert probability_margin(six_way) == pytest.approx(0.52)

    assert normalized_entropy(two_way) > normalized_entropy(six_way)
    assert entropy_concentration(six_way) > entropy_concentration(two_way)

    # Uniform distribution has 0 margin, 1.0 normalized entropy, 0.0 concentration.
    uniform = [0.25, 0.25, 0.25, 0.25]
    assert probability_margin(uniform) == pytest.approx(0.0)
    assert normalized_entropy(uniform) == pytest.approx(1.0)
    assert entropy_concentration(uniform) == pytest.approx(0.0)

    # Point mass has 1.0 margin, 0.0 normalized entropy, 1.0 concentration.
    point = [1.0, 0.0, 0.0]
    assert probability_margin(point) == pytest.approx(1.0)
    assert normalized_entropy(point) == pytest.approx(0.0)
    assert entropy_concentration(point) == pytest.approx(1.0)

    assert logit_margin([1.0, 4.5, 2.0]) == pytest.approx(2.5)
    assert logit_margin([3.0, 3.0, 1.0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bad_values",
    [
        [],
        [0.5],
        [math.nan, 0.5],
        [math.inf, 0.5],
    ],
)
def test_uncertainty_functions_reject_invalid_inputs(
    bad_values: list[float],
) -> None:
    with pytest.raises(ValueError):
        logit_margin(bad_values)
    with pytest.raises(ValueError):
        probability_margin(bad_values)
    with pytest.raises(ValueError):
        normalized_entropy(bad_values)


def test_fit_temperature_softens_overconfident_logits_and_clamps_under_one() -> None:
    # 80% accurate predictions with extreme raw logits [4.0, 0.0] (raw p ~ 0.982).
    # Optimal log-odds for 80% accuracy is ln(0.8 / 0.2) = ln(4) ~ 1.3863,
    # so optimal T = 4.0 / ln(4) ~ 2.8854.
    samples = [([4.0, 0.0], 0)] * 8 + [([4.0, 0.0], 1)] * 2
    raw_t, clamped_t, clamped, warning, raw_nll, cal_nll = fit_temperature(samples)

    expected_t = 4.0 / math.log(4.0)
    assert raw_t == pytest.approx(expected_t, rel=1e-2)
    assert clamped_t == pytest.approx(raw_t)
    assert clamped is False
    assert warning is None
    assert cal_nll < raw_nll

    # Underconfident logits [0.5, 0.0] that are 100% accurate want T < 1.0,
    # which must be clamped to clamp_min=1.0 with a warning.
    underconfident = [([0.5, 0.0], 0)] * 10
    raw_low, clamped_low, was_clamped, low_warn, _, _ = fit_temperature(
        underconfident, clamp_min=1.0, clamp_max=5.0
    )
    assert raw_low < 1.0
    assert clamped_low == pytest.approx(1.0)
    assert was_clamped is True
    assert low_warn is not None and "clamped to 1.00" in low_warn


def test_fit_temperature_table_buckets_by_type_and_option_count_with_fallback() -> None:
    samples = [
        *(
            CalibrationSample(
                suite="s1" if i < 6 else "s2",
                case=f"c{i}",
                question_id="q",
                question_type="choice",
                raw_label_logits=[4.0, 0.0, -1.0],
                gold_index=0 if i % 5 != 0 else 1,
            )
            for i in range(12)
        ),
        # Sparse bucket choice:4 with only 2 samples (< min_bucket_samples=10).
        CalibrationSample(
            suite="s1",
            case="sparse1",
            question_id="q4",
            question_type="choice",
            raw_label_logits=[3.0, 0.0, 0.0, 0.0],
            gold_index=0,
        ),
        CalibrationSample(
            suite="s2",
            case="sparse2",
            question_id="q4",
            question_type="choice",
            raw_label_logits=[3.0, 0.0, 0.0, 0.0],
            gold_index=1,
        ),
    ]
    table = fit_temperature_table(
        samples,
        min_bucket_samples=10,
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        prompt_version="unridden-gemma-choice-v1",
    )

    assert table.status == "fitted_requires_held_out_validation"
    assert "choice:3" in table.buckets
    assert table.buckets["choice:3"].clamped is False
    assert table.buckets["choice:3"].temperature > 1.5
    assert "choice:4" in table.buckets
    assert table.buckets["choice:4"].clamped is True
    assert table.buckets["choice:4"].temperature == pytest.approx(
        table.type_fallbacks["choice"]
    )
    assert any("choice:4" in msg for msg in table.warnings)
    # Unknown option count falls back to the question-type temperature.
    assert table.temperature_for("choice", 6) == pytest.approx(
        table.type_fallbacks["choice"]
    )
    # Unknown type falls back to default_temperature.
    assert table.temperature_for("noul", 2) == pytest.approx(1.0)


def test_cross_fit_calibrated_uncertainties_uses_out_of_fold_suites() -> None:
    # Suite A is very overconfident (needs high T); Suite B is well-calibrated (T ~ 1).
    suite_a = [
        CalibrationSample(
            suite="suite_a",
            case=f"a_{i}",
            question_id="q",
            question_type="noul",
            raw_label_logits=[5.0, 0.0],
            gold_index=0 if i < 6 else 1,
        )
        for i in range(10)
    ]
    suite_b = [
        CalibrationSample(
            suite="suite_b",
            case=f"b_{i}",
            question_id="q",
            question_type="noul",
            raw_label_logits=[1.5, 0.0],
            gold_index=0 if i < 8 else 1,
        )
        for i in range(10)
    ]
    uncertainties, fold_tables = cross_fit_calibrated_uncertainties(
        [*suite_a, *suite_b], min_bucket_samples=5
    )

    assert set(fold_tables) == {"suite_a", "suite_b"}
    assert fold_tables["suite_a"].status == "validated_on_held_out"
    # Suite B is calibrated using Suite A's high temperature, and vice versa.
    assert uncertainties[0].temperature == pytest.approx(
        fold_tables["suite_a"].temperature_for("noul", 2)
    )
    assert uncertainties[10].temperature == pytest.approx(
        fold_tables["suite_b"].temperature_for("noul", 2)
    )
    assert uncertainties[10].temperature > uncertainties[0].temperature


def test_permutation_helpers_rebind_compiler_labels_and_realign_answers() -> None:
    question = ChoiceQuestion(
        type="choice",
        instructions="Pick the department",
        criteria={
            "billing": "Invoices and charges",
            "support": "Technical troubleshooting",
            "sales": "New subscriptions",
        },
    )
    permuted = permute_choice_question(question)
    assert list(permuted.criteria) == ["sales", "support", "billing"]

    # Compiling original vs permuted re-binds options to different letters A, B, C.
    orig_req = DecisionRequest(state="x", questions={"q": question})
    perm_req = DecisionRequest(state="x", questions={"q": permuted})
    orig_compiled = compile_request(orig_req, PROFILE).questions[0]
    perm_compiled = compile_request(perm_req, PROFILE).questions[0]
    assert "A: billing - Invoices and charges" in orig_compiled.messages[0]["content"]
    assert "A: sales - New subscriptions" in perm_compiled.messages[0]["content"]

    # When the model produces consistent option probabilities in the permuted order,
    # realigning restores the exact canonical order and reports flipped=False.
    orig_answer = ChoiceAnswer(
        choice="support",
        probabilities={"billing": 0.1, "support": 0.7, "sales": 0.2},
        confidence=0.7,
    )
    perm_answer_stable = ChoiceAnswer(
        choice="support",
        probabilities={"sales": 0.22, "support": 0.68, "billing": 0.10},
        confidence=0.68,
    )
    stability = compare_permutation_answers(
        question, orig_answer, permuted, perm_answer_stable
    )
    assert stability.flipped is False
    assert list(stability.realigned_probabilities) == ["billing", "support", "sales"]
    assert stability.max_probability_delta == pytest.approx(0.02)
    assert stability.total_variation_distance == pytest.approx(0.02)
    assert stability.post_probe_margin == pytest.approx(0.69 - 0.21)

    # When the model suffers from letter-A bias and picks "sales" in the permuted run,
    # compare_permutation_answers detects the flip.
    perm_answer_flipped = ChoiceAnswer(
        choice="sales",
        probabilities={"sales": 0.65, "support": 0.25, "billing": 0.10},
        confidence=0.65,
    )
    flipped = compare_permutation_answers(
        question, orig_answer, permuted, perm_answer_flipped
    )
    assert flipped.flipped is True
    assert flipped.original_top1 == "support"
    assert flipped.permuted_top1 == "sales"
    assert flipped.total_variation_distance == pytest.approx(0.45)

    # Mutating the semantic option set is rejected.
    with pytest.raises(ValueError, match="exact same option keys"):
        permute_choice_question(question, order=["billing", "support", "other"])
    altered = ChoiceQuestion(
        type="choice",
        instructions="Pick the department",
        criteria={
            "sales": "Different text",
            "support": "Technical troubleshooting",
            "billing": "Invoices and charges",
        },
    )
    with pytest.raises(ValueError, match="altered description"):
        realign_probabilities(question, altered, perm_answer_stable)


def test_score_permutation_reverses_scale_and_realigns_levels() -> None:
    question = ScoreQuestion(
        type="score",
        instructions="Rate severity",
        criteria=["low", "medium", "high"],
    )
    permuted = permute_score_question(question)
    assert permuted.criteria == ["high", "medium", "low"]

    orig_answer = ScoreAnswer(
        score=1.6,
        legend={"0": "low", "1": "medium", "2": "high"},
        probabilities={"0": 0.1, "1": 0.2, "2": 0.7},
        confidence=0.7,
    )
    # In the reversed scale ("0": high, "1": medium, "2": low), level "0" corresponds
    # to canonical level "2" ("high").
    perm_answer = ScoreAnswer(
        score=0.4,
        legend={"0": "high", "1": "medium", "2": "low"},
        probabilities={"0": 0.7, "1": 0.2, "2": 0.1},
        confidence=0.7,
    )
    stability = compare_permutation_answers(
        question, orig_answer, permuted, perm_answer
    )
    assert stability.flipped is False
    assert stability.original_top1 == 2
    assert stability.permuted_top1 == 2
    assert stability.realigned_probabilities == pytest.approx(
        {"0": 0.1, "1": 0.2, "2": 0.7}
    )
    assert stability.total_variation_distance == pytest.approx(0.0)


def test_two_stage_escalation_gate_routes_borderline_and_catches_flips() -> None:
    choice_q = ChoiceQuestion(
        type="choice",
        instructions="Select route",
        criteria={"billing": "Invoices", "support": "Tech", "sales": "Orders"},
    )
    noul_q = NoulQuestion(type="noul", instructions="Is it urgent?")
    assert is_permutable(choice_q) is True
    assert is_permutable(noul_q) is False

    config = EscalationGateConfig(
        high_margin_threshold=0.60,
        low_margin_threshold=0.20,
        min_coverage=0.10,
        max_tvd_threshold=0.25,
        post_probe_margin_threshold=0.35,
    )

    # 1. High margin -> accept_primary immediately in Stage 1.
    high_decision = evaluate_question_gate(
        "q_high", choice_q, [5.0, 0.0, 0.0], coverage=0.95, config=config
    )
    assert high_decision.action == "accept_primary"
    assert high_decision.reason == "high_calibrated_margin"

    # 2. Borderline margin (probs [0.6, 0.25, 0.15] -> margin 0.35 in [0.20, 0.60])
    # -> Stage 1 requests run_permutation_probe.
    border_logits = [math.log(0.6), math.log(0.25), math.log(0.15)]
    stage1_border = evaluate_question_gate(
        "q_border", choice_q, border_logits, coverage=0.9, config=config
    )
    assert stage1_border.action == "run_permutation_probe"
    assert stage1_border.reason == "borderline_margin"

    probes = build_permutation_probe_questions(
        {"q_high": choice_q, "q_border": choice_q},
        {"q_high": high_decision, "q_border": stage1_border},
    )
    assert list(probes) == ["q_border"]
    permuted_q = probes["q_border"]

    # Stage 2a: Stable permutation probe -> accept_primary (permutation_stable).
    orig_ans = ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 0.60, "support": 0.25, "sales": 0.15},
        confidence=0.60,
    )
    stable_perm_ans = ChoiceAnswer(
        choice="billing",
        probabilities={"sales": 0.14, "support": 0.24, "billing": 0.62},
        confidence=0.62,
    )
    stage2_stable = evaluate_question_gate(
        "q_border",
        choice_q,
        border_logits,
        coverage=0.9,
        config=config,
        original_answer=orig_ans,
        permuted_question=permuted_q,
        permuted_answer=stable_perm_ans,
    )
    assert stage2_stable.action == "accept_primary"
    assert stage2_stable.reason == "permutation_stable"

    # Stage 2b: Flipping permutation probe -> escalate (permutation_flip).
    flip_perm_ans = ChoiceAnswer(
        choice="sales",
        probabilities={"sales": 0.55, "support": 0.25, "billing": 0.20},
        confidence=0.55,
    )
    stage2_flip = evaluate_question_gate(
        "q_border",
        choice_q,
        border_logits,
        coverage=0.9,
        config=config,
        original_answer=orig_ans,
        permuted_question=permuted_q,
        permuted_answer=flip_perm_ans,
    )
    assert stage2_flip.action == "escalate"
    assert stage2_flip.reason == "permutation_flip"

    # 3. Low margin (< 0.20) -> escalate immediately in Stage 1.
    low_logits = [math.log(0.40), math.log(0.35), math.log(0.25)]
    low_decision = evaluate_question_gate(
        "q_low", choice_q, low_logits, coverage=0.9, config=config
    )
    assert low_decision.action == "escalate"
    assert low_decision.reason == "low_calibrated_margin"

    # 4. Low coverage (< 0.10) -> escalate immediately with reason low_coverage.
    low_cov = evaluate_question_gate(
        "q_cov", choice_q, [5.0, 0.0, 0.0], coverage=0.02, config=config
    )
    assert low_cov.action == "escalate"
    assert low_cov.reason == "low_coverage"

    # 5. Borderline noul question (not permutable) -> escalates in Stage 1.
    noul_border = evaluate_question_gate(
        "q_noul", noul_q, [math.log(0.65), math.log(0.35)], coverage=0.9, config=config
    )
    assert noul_border.action == "escalate"
    assert noul_border.reason == "low_calibrated_margin"

    with pytest.raises(ValidationError, match="low_margin_threshold"):
        EscalationGateConfig(low_margin_threshold=0.7, high_margin_threshold=0.5)


def test_selective_prediction_and_escalation_metrics_handle_ties() -> None:
    # Saturated tie group at 1.0: 3 correct and 1 wrong all tied at 1.0,
    # followed by 2 wrong at 0.6.
    scores = [1.0, 1.0, 1.0, 1.0, 0.6, 0.6]
    correct = [True, True, True, False, False, False]

    curve = risk_coverage_curve(scores, correct)
    # Because tie groups are walked together, there are exactly 2 points on the curve.
    assert len(curve) == 2
    assert curve[0].accepted_count == 4
    assert curve[0].coverage == pytest.approx(4 / 6)
    assert curve[0].accuracy == pytest.approx(0.75)
    assert curve[0].risk == pytest.approx(0.25)
    assert curve[1].accepted_count == 6
    assert curve[1].risk == pytest.approx(0.5)

    aurc = area_under_risk_coverage(scores, correct)
    assert 0.0 < aurc < 0.5
    assert coverage_at_risk(scores, correct, max_risk=0.30) == pytest.approx(4 / 6)
    assert coverage_at_risk(scores, correct, max_risk=0.10) == pytest.approx(0.0)

    assert brier_toplabel([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert multiclass_brier(
        [{"a": 0.8, "b": 0.2}, {"a": 0.1, "b": 0.9}], ["a", "b"]
    ) == pytest.approx(0.05)
    assert gold_nll([{"a": 0.5, "b": 0.5}], ["a"]) == pytest.approx(math.log(2.0))
    assert expected_calibration_error([0.8] * 5, [True] * 4 + [False], bins=5) == (
        pytest.approx(0.0)
    )
    assert expected_calibration_error(
        [0.8] * 5, [True] * 4 + [False], bins=5, equal_mass=True
    ) == pytest.approx(0.0)

    # Escalation budget evaluation.
    primary_ok = [True, True, False, False]
    secondary_ok = [True, False, True, True]
    point = evaluate_escalation_budget(
        "margin",
        [0.9, 0.8, 0.2, 0.1],
        primary_ok,
        secondary_ok,
        budget_questions=2,
    )
    assert point.rescues == 2
    assert point.reversals == 0
    assert point.net_gain == 2
    assert point.routed_accuracy == pytest.approx(1.0)


def test_offline_evaluate_paired_runs_end_to_end(tmp_path: Path) -> None:
    def make_row(
        suite: str,
        case: str,
        logits: list[float],
        expected: str,
        got: str,
    ) -> dict[str, object]:
        probs = calibrated_probabilities(logits, 1.0)
        return {
            "suite": suite,
            "case": case,
            "difficulty": "medium",
            "status": "ok",
            "request": {
                "model": "local-gemma-unridden-v1",
                "state": "fixture state",
                "questions": {
                    "q1": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"a": "opt a", "b": "opt b", "c": "opt c"},
                    }
                },
            },
            "response": {
                "model": "local-gemma-unridden-v1",
                "answers": {
                    "q1": {
                        "type": "choice",
                        "choice": got,
                        "probabilities": {
                            "a": probs[0],
                            "b": probs[1],
                            "c": probs[2],
                        },
                        "confidence": max(probs),
                    }
                },
                "diagnostics": {
                    "q1": {
                        "raw_label_logits": logits,
                        "coverage": 0.95,
                        "model_sha256": "a" * 64,
                        "runtime_sha256": "b" * 64,
                        "prompt_version": "unridden-gemma-choice-v1",
                    }
                },
            },
            "outcomes": [
                {
                    "suite": suite,
                    "case": case,
                    "difficulty": "medium",
                    "qid": "q1",
                    "type": "choice",
                    "expected": expected,
                    "got": got,
                    "correct": got == expected,
                    "confidence": max(probs),
                    "probabilities": {
                        "a": probs[0],
                        "b": probs[1],
                        "c": probs[2],
                    },
                }
            ],
        }

    primary_path = tmp_path / "primary.jsonl"
    secondary_path = tmp_path / "secondary.jsonl"
    p_rows = [
        make_row("s1", "c1", [4.0, 0.0, 0.0], "a", "a"),
        make_row("s1", "c2", [0.4, 0.3, 0.0], "b", "a"),
        make_row("s2", "c3", [4.0, 0.0, 0.0], "a", "a"),
        make_row("s2", "c4", [0.5, 0.4, 0.0], "b", "a"),
    ]
    s_rows = [
        make_row("s1", "c1", [4.0, 0.0, 0.0], "a", "a"),
        make_row("s1", "c2", [0.0, 3.0, 0.0], "b", "b"),
        make_row("s2", "c3", [4.0, 0.0, 0.0], "a", "a"),
        make_row("s2", "c4", [0.0, 3.0, 0.0], "b", "b"),
    ]
    primary_path.write_text(
        "\n".join(json.dumps(r) for r in p_rows) + "\n", encoding="utf-8"
    )
    secondary_path.write_text(
        "\n".join(json.dumps(r) for r in s_rows) + "\n", encoding="utf-8"
    )

    report, table = evaluate_paired_runs(
        primary_path, secondary_path, budgets=(1, 2), min_bucket_samples=2
    )
    assert report["questions"] == 4
    assert report["baseline"]["primary_correct"] == 2
    assert report["baseline"]["secondary_correct"] == 4
    assert "choice:3" in table.buckets
    budget2 = next(
        row
        for row in report["escalation_curves"]["calibrated_probability_margin"]
        if row["budget_questions"] == 2
    )
    assert budget2["rescues"] == 2
    assert budget2["reversals"] == 0
    assert budget2["routed_accuracy"] == pytest.approx(1.0)


def test_misc_validation_guards() -> None:
    with pytest.raises(ValueError, match="exactly two label logits"):
        CalibrationSample(
            suite="s",
            case="c",
            question_id="q",
            question_type="noul",
            raw_label_logits=[1.0, 0.0, -1.0],
            gold_index=0,
        )
    with pytest.raises(ValueError, match="gold_index"):
        CalibrationSample(
            suite="s",
            case="c",
            question_id="q",
            question_type="noul",
            raw_label_logits=[1.0, 0.0],
            gold_index=2,
        )
    with pytest.raises(ValueError, match="empty"):
        nll_for_logits([], 1.0)
    with pytest.raises(ValueError, match="positive"):
        calibrated_probabilities([1.0, 0.0], 0.0)
    # Single-suite k-fold cross-fitting fallback.
    single_suite = [
        CalibrationSample(
            suite="only_one",
            case=f"c{i}",
            question_id="q",
            question_type="noul",
            raw_label_logits=[2.0, 0.0],
            gold_index=0 if i < 4 else 1,
        )
        for i in range(6)
    ]
    unc, tables = cross_fit_calibrated_uncertainties(single_suite, min_bucket_samples=2)
    assert len(unc) == 6
    assert len(tables) == 5
    # Noul permutation is a deep copy.
    noul = NoulQuestion(type="noul", instructions="check")
    assert permute_question(noul) == noul
    noul_ans = NoulAnswer(noul=0.75)
    noul_stab = compare_permutation_answers(noul, noul_ans, noul, noul_ans)
    assert noul_stab.flipped is False
    assert noul_stab.total_variation_distance == pytest.approx(0.0)
