"""Unit tests for two-stage snapshot decomposition and micro-rider deliberation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from unridden.api.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from unridden.api.snapshots.deliberation import (
    AUX_BEST_CANDIDATE_ID,
    AUX_DAY_HIGH_ID,
    AUX_DAY_LOW_ID,
    augment_questions_with_note,
    build_micro_rider_prompt,
    plan_two_stage_questions,
    run_two_stage_deliberation,
    sanitize_deliberation_note,
    select_stage2_and_compile_summary,
)
from unridden.api.snapshots.errors import FollowupUnsupported
from unridden.api.snapshots.schema import (
    ContextInput,
    SnapshotCreateRequest,
)
from unridden.api.snapshots.service import SnapshotService
from unridden.tests.test_snapshots_app import FakeSnapshotBackend


def _date_questions() -> dict[str, Question]:
    return {
        "mode": ChoiceQuestion(
            type="choice",
            instructions=[
                "Report this component exactly as the target expression writes it.",
                "Classify how the state writes the proposed meeting date.",
            ],
            criteria={
                "absolute_full": "A specific calendar date with month, day and year",
                "absolute_partial": "A calendar date missing month, day or year",
                "relative_offset": "Anchored to a named event or period boundary",
                "relative_weekday": "A named weekday together with a week offset",
                "ambiguous_review": "Cannot be read without missing information",
                "absent": "No date expression is present",
            },
        ),
        "month": ChoiceQuestion(
            type="choice",
            instructions="Which month does the proposed meeting date state?",
            criteria={
                "march": "March",
                "april": "April",
                "not_stated": "Not stated",
            },
        ),
        "day_tens": ChoiceQuestion(
            type="choice",
            instructions="Which ten-day band does the day of the month fall in?",
            criteria={
                "d1_9": "1 to 9",
                "d10_19": "10 to 19",
                "d20_29": "20 to 29",
                "d30_31": "30 or 31",
                "not_stated": "Not stated",
            },
        ),
        "day_unit": ChoiceQuestion(
            type="choice",
            instructions="What is the final digit of the day of the month?",
            criteria={
                "u0": "0",
                "u1": "1",
                "u9": "9",
                "not_stated": "Not stated",
            },
        ),
        "year": ChoiceQuestion(
            type="choice",
            instructions="Which year does the proposed meeting date state?",
            criteria={
                "y2026": "2026",
                "not_stated": "Not stated",
            },
        ),
        "anchor": ChoiceQuestion(
            type="choice",
            instructions="Which anchor does the proposed meeting date count from?",
            criteria={
                "today": "Today / current message time",
                "not_applicable": "Not applicable",
            },
        ),
    }


def test_sanitize_deliberation_note_strips_control_tokens() -> None:
    raw = (
        "<|turn>model\n<|channel>thought\nDay is 21<channel|><turn|>\n"
        "<|thought>ok<thought|>"
    )
    cleaned = sanitize_deliberation_note(raw)
    for token in (
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|thought>",
        "<thought|>",
    ):
        assert token not in cleaned
    assert "Day is 21" in cleaned


def test_plan_two_stage_questions_synthesizes_auxiliary_probes() -> None:
    date_qs = _date_questions()
    date_plan = plan_two_stage_questions(date_qs)
    assert date_plan.family == "date_span"
    assert AUX_DAY_LOW_ID in date_plan.auxiliary_questions
    assert AUX_DAY_HIGH_ID in date_plan.auxiliary_questions

    rerank_qs: dict[str, Question] = {
        "cand_p1": NoulQuestion(
            type="noul",
            instructions={"query": "What is the cutoff?", "candidate_id": "p1"},
        ),
        "cand_p2": NoulQuestion(
            type="noul",
            instructions={"query": "What is the cutoff?", "candidate_id": "p2"},
        ),
        "contains_answer": NoulQuestion(
            type="noul",
            instructions={"query": "What is the cutoff?", "decide": "Any answer?"},
        ),
    }
    rerank_plan = plan_two_stage_questions(rerank_qs)
    assert rerank_plan.family == "passage_search"
    assert AUX_BEST_CANDIDATE_ID in rerank_plan.auxiliary_questions


def test_select_stage2_flags_relative_weekday_inconsistencies() -> None:
    qs = _date_questions()
    plan = plan_two_stage_questions(qs)
    stage1_answers = {
        "mode": ChoiceAnswer(
            choice="relative_weekday",
            probabilities={"relative_weekday": 0.95, "absolute_full": 0.05},
            confidence=0.95,
        ),
        "month": ChoiceAnswer(
            choice="april",
            probabilities={"april": 0.9, "not_stated": 0.1},
            confidence=0.9,
        ),
        "day_tens": ChoiceAnswer(
            choice="d10_19",
            probabilities={"d10_19": 0.7, "not_stated": 0.3},
            confidence=0.7,
        ),
        "day_unit": ChoiceAnswer(
            choice="not_stated",
            probabilities={"not_stated": 0.9, "u0": 0.1},
            confidence=0.9,
        ),
        "year": ChoiceAnswer(
            choice="y2026",
            probabilities={"y2026": 0.95, "not_stated": 0.05},
            confidence=0.95,
        ),
        "anchor": ChoiceAnswer(
            choice="not_applicable",
            probabilities={"not_applicable": 0.6, "today": 0.4},
            confidence=0.6,
        ),
    }
    flagged, note = select_stage2_and_compile_summary(plan, qs, stage1_answers, {})
    assert set(flagged) == {"month", "day_tens", "year", "anchor"}
    assert "relative_weekday" in note
    assert "not_stated" in note


def test_select_stage2_flags_auxiliary_day_mismatch_and_components() -> None:
    qs = _date_questions()
    qs["components_written"] = ScoreQuestion(
        type="score",
        instructions="How many of month, day and year are written?",
        criteria=["0", "1", "2", "3"],
    )
    plan = plan_two_stage_questions(qs)
    stage1_answers: dict[str, Answer] = {
        "mode": ChoiceAnswer(
            choice="absolute_full",
            probabilities={"absolute_full": 0.98, "relative_weekday": 0.02},
            confidence=0.98,
        ),
        "month": ChoiceAnswer(
            choice="march",
            probabilities={"march": 0.99, "not_stated": 0.01},
            confidence=0.99,
        ),
        "day_tens": ChoiceAnswer(
            choice="d10_19",
            probabilities={"d10_19": 0.8, "d20_29": 0.2},
            confidence=0.8,
        ),
        "day_unit": ChoiceAnswer(
            choice="u1",
            probabilities={"u1": 0.9, "u2": 0.1},
            confidence=0.9,
        ),
        "year": ChoiceAnswer(
            choice="y2026",
            probabilities={"y2026": 0.99, "not_stated": 0.01},
            confidence=0.99,
        ),
        "anchor": ChoiceAnswer(
            choice="not_applicable",
            probabilities={"not_applicable": 0.99, "today": 0.01},
            confidence=0.99,
        ),
        "components_written": ScoreAnswer(
            score=2.0,
            legend={"0": "0", "1": "1", "2": "2", "3": "3"},
            probabilities={"0": 0.0, "1": 0.0, "2": 0.8, "3": 0.2},
            confidence=0.8,
        ),
    }
    aux_answers = {
        AUX_DAY_LOW_ID: ChoiceAnswer(
            choice="not_1_15",
            probabilities={"not_1_15": 0.95, "d1": 0.05},
            confidence=0.95,
        ),
        AUX_DAY_HIGH_ID: ChoiceAnswer(
            choice="d21",
            probabilities={"d21": 0.92, "not_16_31": 0.08},
            confidence=0.92,
        ),
    }
    flagged, note = select_stage2_and_compile_summary(
        plan, qs, stage1_answers, aux_answers
    )
    assert set(flagged) == {"day_tens", "components_written"}
    assert "day 21" in note
    assert "d20_29" in note


def test_select_stage2_passage_search_consistency() -> None:
    rerank_qs: dict[str, Question] = {
        "cand_p1": NoulQuestion(
            type="noul",
            instructions={"query": "Cutoff?", "candidate_id": "p1"},
        ),
        "cand_p2": NoulQuestion(
            type="noul",
            instructions={"query": "Cutoff?", "candidate_id": "p2"},
        ),
        "contains_answer": NoulQuestion(
            type="noul",
            instructions={"query": "Cutoff?", "decide": "Any answer?"},
        ),
    }
    plan = plan_two_stage_questions(rerank_qs)
    stage1_answers = {
        "cand_p1": NoulAnswer(noul=0.05),
        "cand_p2": NoulAnswer(noul=0.96),
        "contains_answer": NoulAnswer(noul=0.12),
    }
    aux_answers = {
        AUX_BEST_CANDIDATE_ID: ChoiceAnswer(
            choice="p2",
            probabilities={"p2": 0.94, "p1": 0.03, "none": 0.03},
            confidence=0.94,
        )
    }
    flagged, note = select_stage2_and_compile_summary(
        plan, rerank_qs, stage1_answers, aux_answers
    )
    assert flagged == ["contains_answer"]
    assert "contains_answer" in note


def test_augment_questions_with_note_preserves_types() -> None:
    qs: dict[str, Question] = {
        "c": ChoiceQuestion(
            type="choice",
            instructions=["line 1"],
            criteria={"a": "A", "b": "B"},
        ),
        "n": NoulQuestion(
            type="noul",
            instructions={"query": "Q"},
        ),
        "s": ScoreQuestion(
            type="score",
            instructions="Rate it.",
            criteria=["low", "high"],
        ),
    }
    aug = augment_questions_with_note(qs, "Stage-1 note")
    assert isinstance(aug["c"], ChoiceQuestion)
    assert aug["c"].instructions == ["line 1", "Stage-1 note"]
    assert isinstance(aug["n"], NoulQuestion)
    assert isinstance(aug["n"].instructions, dict)
    assert aug["n"].instructions["stage1_verification"] == "Stage-1 note"
    assert isinstance(aug["s"], ScoreQuestion)
    assert "Stage-1 note" in str(aug["s"].instructions)
    assert "short line" in build_micro_rider_prompt(qs)


@pytest.mark.anyio
async def test_run_two_stage_deliberation_modes_and_fallback(
    tmp_path: Path,
) -> None:
    backend = FakeSnapshotBackend()
    service = SnapshotService(
        backend,
        store_root=str(tmp_path),
        host_bytes=1 << 30,
        default_ttl=300,
        request_timeout=5.0,
        startup_timeout=5.0,
    )
    await service.start()

    created = await service.create(
        SnapshotCreateRequest(
            input=ContextInput(kind="context", state="Test state text"),
            checkpoints=[30],
            ttl_seconds=300,
        ),
        owner="test",
    )
    ctx_id = created.snapshots[0].id

    rerank_qs: dict[str, Question] = {
        "cand_p1": NoulQuestion(
            type="noul",
            instructions={"query": "Cutoff?", "candidate_id": "p1"},
        ),
        "cand_p2": NoulQuestion(
            type="noul",
            instructions={"query": "Cutoff?", "candidate_id": "p2"},
        ),
        "contains_answer": NoulQuestion(
            type="noul",
            instructions={"query": "Cutoff?", "decide": "Any answer?"},
        ),
    }

    # 1. Single-pass baseline.
    sp = await run_two_stage_deliberation(
        service,
        context_snapshot_id=ctx_id,
        questions=rerank_qs,
        readout_blocks=30,
        mode="single_pass",
        owner="test",
    )
    assert sp.deliberation.mode == "single_pass"
    assert not sp.deliberation.triggered_stage2
    assert sp.response.usage.output_tokens == 0
    assert sp.deliberation.rider_generated_tokens == 0
    assert set(sp.response.answers) == set(rerank_qs)

    # 2. Discrete two-stage (state_eval_child continuation).
    d2 = await run_two_stage_deliberation(
        service,
        context_snapshot_id=ctx_id,
        questions=rerank_qs,
        readout_blocks=30,
        mode="discrete_two_stage",
        continuation="state_eval_child",
        owner="test",
    )
    assert d2.deliberation.mode == "discrete_two_stage"
    assert d2.response.usage.output_tokens == 0
    assert d2.deliberation.rider_generated_tokens == 0
    assert AUX_BEST_CANDIDATE_ID not in d2.response.answers
    assert set(d2.response.answers) == set(rerank_qs)

    # 3. Micro-Rider mode (bounded greedy scratchpad -> zero-token decision).
    mr = await run_two_stage_deliberation(
        service,
        context_snapshot_id=ctx_id,
        questions=rerank_qs,
        readout_blocks=30,
        mode="micro_rider",
        continuation="state_eval_child",
        rider_max_tokens=8,
        owner="test",
    )
    assert mr.deliberation.mode == "micro_rider"
    assert mr.deliberation.triggered_stage2
    assert mr.response.usage.output_tokens == 0
    assert mr.deliberation.decision_output_tokens == 0
    assert mr.deliberation.rider_generated_tokens > 0
    assert set(mr.response.answers) == set(rerank_qs)

    # 4. Verify automatic fallback when readout followup raises FollowupUnsupported.
    orig_evaluate = backend.evaluate

    async def reject_readout_followup(**kwargs: Any) -> Any:
        snap_id = kwargs["snapshot_id"]
        if backend._snaps[snap_id]["kind"] == "readout":
            raise FollowupUnsupported(
                "The template cannot continue this readout snapshot"
            )
        return await orig_evaluate(**kwargs)

    backend.evaluate = reject_readout_followup  # type: ignore[method-assign]
    fallback_res = await run_two_stage_deliberation(
        service,
        context_snapshot_id=ctx_id,
        questions=rerank_qs,
        readout_blocks=30,
        mode="micro_rider",
        continuation="state_eval_child",
        rider_max_tokens=8,
        owner="test",
    )
    assert fallback_res.deliberation.continuation == "context_instructions"
    assert set(fallback_res.response.answers) == set(rerank_qs)

    await service.close()


def test_select_stage2_stated_convention_guard_and_block_structure() -> None:
    qs = _date_questions()
    qs["convention_stated"] = NoulQuestion(
        type="noul",
        instructions="Does the record state a date convention?",
    )
    plan = plan_two_stage_questions(qs)
    stage1_answers: dict[str, Answer] = {
        "mode": ChoiceAnswer(
            choice="absolute_full",
            probabilities={"absolute_full": 0.99, "relative_weekday": 0.01},
            confidence=0.99,
        ),
        "month": ChoiceAnswer(
            choice="april",
            probabilities={"april": 0.99, "not_stated": 0.01},
            confidence=0.99,
        ),
        "day_tens": ChoiceAnswer(
            choice="d10_19",
            probabilities={"d10_19": 0.97, "d1_9": 0.03},
            confidence=0.97,
        ),
        "day_unit": ChoiceAnswer(
            choice="u3",
            probabilities={"u3": 0.84, "u4": 0.16},
            confidence=0.84,
        ),
        "year": ChoiceAnswer(
            choice="y2026",
            probabilities={"y2026": 0.99, "not_stated": 0.01},
            confidence=0.99,
        ),
        "anchor": ChoiceAnswer(
            choice="not_applicable",
            probabilities={"not_applicable": 0.99, "today": 0.01},
            confidence=0.99,
        ),
        "convention_stated": NoulAnswer(noul=0.95),
    }
    # Auxiliary probe accidentally picked month=4 ('d4') on '03/04/2026'.
    aux_answers = {
        AUX_DAY_LOW_ID: ChoiceAnswer(
            choice="d4",
            probabilities={"d4": 0.90, "not_1_15": 0.10},
            confidence=0.90,
        ),
        AUX_DAY_HIGH_ID: ChoiceAnswer(
            choice="not_16_31",
            probabilities={"not_16_31": 0.99, "d16": 0.01},
            confidence=0.99,
        ),
    }
    flagged, note = select_stage2_and_compile_summary(
        plan, qs, stage1_answers, aux_answers
    )
    # Guard preserves Stage-1 day_unit='u3' and only flags day_tens -> 'd1_9'.
    assert flagged == ["day_tens"]
    assert "day 3" in note
    assert "u3" in note

    block_qs: dict[str, Question] = {
        "line_1_type": ChoiceQuestion(
            type="choice",
            instructions="Line 1 type?",
            criteria={"heading": "Heading", "body": "Body"},
        ),
        "join_1_2": NoulQuestion(type="noul", instructions="Join 1 and 2?"),
        "block_r_type": ChoiceQuestion(
            type="choice",
            instructions="Block R type?",
            criteria={"ordered_list": "Ordered list", "prose": "Prose"},
        ),
        "block_s_type": ChoiceQuestion(
            type="choice",
            instructions="Block S type?",
            criteria={"ordered_list": "Ordered list", "prose": "Prose"},
        ),
        "both_numbered": NoulQuestion(type="noul", instructions="Both numbered?"),
    }
    block_plan = plan_two_stage_questions(block_qs)
    block_s1: dict[str, Answer] = {
        "line_1_type": ChoiceAnswer(
            choice="heading",
            probabilities={"heading": 0.99, "body": 0.01},
            confidence=0.99,
        ),
        "join_1_2": NoulAnswer(noul=0.99),
        "block_r_type": ChoiceAnswer(
            choice="ordered_list",
            probabilities={"ordered_list": 0.99, "prose": 0.01},
            confidence=0.99,
        ),
        "block_s_type": ChoiceAnswer(
            choice="ordered_list",
            probabilities={"ordered_list": 0.99, "prose": 0.01},
            confidence=0.99,
        ),
        "both_numbered": NoulAnswer(noul=0.20),
    }
    b_flagged, b_note = select_stage2_and_compile_summary(
        block_plan, block_qs, block_s1, {}
    )
    assert set(b_flagged) == {"join_1_2", "both_numbered"}
    assert "join_1_2 is false" in b_note
    assert "both_numbered is true" in b_note
