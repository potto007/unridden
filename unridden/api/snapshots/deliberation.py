"""Two-stage structured micro-deliberation over /v2 state snapshots.

Implements Priority Matrix #4 (Proposal 2 of
`docs/research/intermediate-thinking-steps-review.md`):

1. **Zero-token discrete decomposition (`discrete_two_stage`):**
   Evaluates Stage-1 anchor/prerequisite questions plus bounded auxiliary
   `Choice` sub-questions (`output_tokens == 0`) against the warm context
   snapshot, checks cross-question structural consistency and margin, compiles a
   deterministic Stage-1 summary when needed, appends it via
   `SnapshotService.state_eval` (`save_result_snapshot=True`), and evaluates only
   the dependent/flagged Stage-2 questions from the resulting snapshot (with
   automatic fallback to context-branch instructions when a model template does
   not support readout follow-ups).
2. **Bounded Micro-Rider thought prefix (`micro_rider`):**
   Runs `SnapshotService.ride` (`POST /v2/rider`) for at most `K` (`8..16`)
   greedy tokens from the warm context snapshot to produce a compact intermediate
   extraction scratchpad, then evaluates `/v2/decisions` (`output_tokens == 0` on
   the decision readout) conditioned on that scratchpad.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from unridden.api.compiler import render
from unridden.api.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Entry,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    StrictModel,
    Usage,
)
from unridden.api.snapshots.errors import FollowupUnsupported
from unridden.api.snapshots.schema import (
    BlockTokens,
    Readout,
    RiderRequest,
    SnapshotRef,
    SnapshotTiming,
    SnapshotUsage,
    StateEvaluationRequest,
    StateReadout,
    V2DecisionRequest,
    V2DecisionResponse,
)
from unridden.api.snapshots.service import SnapshotService

type DeliberationMode = Literal["single_pass", "discrete_two_stage", "micro_rider"]
type ContinuationMode = Literal["state_eval_child", "context_instructions"]

CONTROL_TOKENS: tuple[str, ...] = (
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|thought>",
    "<thought|>",
)

AUX_DAY_LOW_ID = "_aux_day_1_15"
AUX_DAY_HIGH_ID = "_aux_day_16_31"
AUX_BEST_CANDIDATE_ID = "_aux_best_candidate"

_DATE_COMPONENT_IDS = frozenset(
    {
        "mode",
        "month",
        "day_tens",
        "day_unit",
        "year",
        "weekday",
        "week_offset",
        "anchor",
        "offset_unit",
        "offset_count",
        "components_written",
    }
)
_CALENDAR_ONLY_IDS = frozenset({"month", "day_tens", "day_unit", "year"})
_RELATIVE_ONLY_IDS = frozenset({"week_offset", "anchor", "offset_unit", "offset_count"})


class DeliberationStepUsage(StrictModel):
    """Accounting and audit record for a two-stage or micro-rider decision."""

    mode: DeliberationMode
    continuation: ContinuationMode | None = None
    triggered_stage2: bool
    stage1_question_ids: list[str]
    auxiliary_question_ids: list[str] = Field(default_factory=list)
    stage2_question_ids: list[str] = Field(default_factory=list)
    deliberation_note: str = ""
    state_eval_snapshot_id: str | None = None
    decision_output_tokens: Literal[0] = 0
    rider_generated_tokens: int = Field(ge=0, default=0)
    stage1_ms: float = Field(ge=0.0, default=0.0)
    state_eval_ms: float = Field(ge=0.0, default=0.0)
    stage2_ms: float = Field(ge=0.0, default=0.0)
    total_ms: float = Field(ge=0.0, default=0.0)


class DeliberationResult(StrictModel):
    """Combined `/v2/decisions` response and deliberation audit metadata."""

    response: V2DecisionResponse
    deliberation: DeliberationStepUsage


@dataclass(frozen=True, slots=True)
class TwoStagePlan:
    """Deterministic Stage-1 / Stage-2 question plan for a decision request."""

    family: Literal["date_span", "block_structure", "passage_search", "general"]
    stage1_questions: dict[str, Question]
    auxiliary_questions: dict[str, Question]


def sanitize_deliberation_note(text: str) -> str:
    """Strip chat-template control tokens and normalize whitespace."""
    cleaned = text
    for token in CONTROL_TOKENS:
        cleaned = cleaned.replace(token, " ")
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    compact = "\n".join(line for line in lines if line).strip()
    return compact or "No additional intermediate notes."


def _extract_target_description(questions: Mapping[str, Question]) -> str:
    """Extract the shared target description from date/span question instructions."""
    for qid in ("mode", "day_tens", "day_unit", "month"):
        question = questions.get(qid)
        if question is None:
            continue
        rendered = render(question.instructions)
        if rendered:
            lines = [line.strip() for line in rendered.splitlines() if line.strip()]
            if lines:
                return lines[-1]
    return "Identify the exact target date expression written in the state."


_MONTH_NUM: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _build_auxiliary_date_questions(
    questions: Mapping[str, Question],
) -> dict[str, Question]:
    """Synthesize two <=17-option Choice probes for literal day-of-month extraction."""
    if "day_tens" not in questions and "day_unit" not in questions:
        return {}
    target_hint = _extract_target_description(questions)
    base_rule = (
        "Report the literal day-of-month number only if the target date expression "
        "itself writes an explicit calendar day of the month (and its day/month "
        "order is not unrecoverably ambiguous). If the record states a date "
        "convention (such as DD/MM/YYYY or MM/DD/YYYY), follow that stated "
        "day/month/year order."
    )
    low_criteria: dict[str, Entry] = {
        f"d{day}": (
            f"The target expression explicitly writes day {day} (or {day:02d}) "
            "of the month"
        )
        for day in range(1, 16)
    }
    low_criteria["not_1_15"] = (
        "The day of the month is 16 to 31, not stated in the target expression, "
        "or ambiguous"
    )
    high_criteria: dict[str, Entry] = {
        f"d{day}": f"The target expression explicitly writes day {day} of the month"
        for day in range(16, 32)
    }
    high_criteria["not_16_31"] = (
        "The day of the month is 1 to 15, not stated in the target expression, "
        "or ambiguous"
    )
    return {
        AUX_DAY_LOW_ID: ChoiceQuestion(
            type="choice",
            instructions=[
                base_rule,
                target_hint,
                (
                    "Which day number between 1 and 15 is written in the target "
                    "date expression?"
                ),
            ],
            criteria=low_criteria,
        ),
        AUX_DAY_HIGH_ID: ChoiceQuestion(
            type="choice",
            instructions=[
                base_rule,
                target_hint,
                (
                    "Which day number between 16 and 31 is written in the target "
                    "date expression?"
                ),
            ],
            criteria=high_criteria,
        ),
    }


def _build_auxiliary_passage_questions(
    questions: Mapping[str, Question],
) -> dict[str, Question]:
    """Synthesize a single-choice passage selector for cand_p* Noul items."""
    if "best_candidate" in questions:
        return {}
    cand_ids = [qid for qid in questions if qid.startswith("cand_p")]
    if len(cand_ids) < 2:
        return {}
    query_text = ""
    first_cand = questions[cand_ids[0]]
    if isinstance(first_cand.instructions, dict):
        raw_query = first_cand.instructions.get("query")
        if isinstance(raw_query, str):
            query_text = raw_query
    criteria: dict[str, Entry] = {}
    for qid in cand_ids:
        pid = qid.removeprefix("cand_")
        criteria[pid] = (
            f"Candidate passage {pid} directly states the fact the query asks for"
        )
    criteria["none"] = "No candidate passage states the fact the query asks for"
    instructions: Entry = {
        "task": (
            "Select the single candidate passage that directly states the fact "
            "the query asks for."
        ),
        "query": query_text or "Answer the query using the candidate passages.",
        "rule": (
            "Select none when no candidate passage directly states the requested fact."
        ),
    }
    return {
        AUX_BEST_CANDIDATE_ID: ChoiceQuestion(
            type="choice",
            instructions=instructions,
            criteria=criteria,
        )
    }


def plan_two_stage_questions(questions: Mapping[str, Question]) -> TwoStagePlan:
    """Classify the request schema and build the Stage-1 + auxiliary question set."""
    qids = set(questions)
    if "mode" in qids or bool(qids & {"day_tens", "day_unit", "week_offset"}):
        return TwoStagePlan(
            family="date_span",
            stage1_questions=dict(questions),
            auxiliary_questions=_build_auxiliary_date_questions(questions),
        )
    if bool(
        qids
        & {
            "block_type",
            "line_1_type",
            "block_a_type",
            "block_r_type",
            "join_1_2",
        }
    ):
        return TwoStagePlan(
            family="block_structure",
            stage1_questions=dict(questions),
            auxiliary_questions={},
        )
    if (
        "contains_answer" in qids
        or "answer_line" in qids
        or any(qid.startswith("cand_p") for qid in qids)
    ):
        return TwoStagePlan(
            family="passage_search",
            stage1_questions=dict(questions),
            auxiliary_questions=_build_auxiliary_passage_questions(questions),
        )
    return TwoStagePlan(
        family="general",
        stage1_questions=dict(questions),
        auxiliary_questions={},
    )


def _choice_val(answers: Mapping[str, Answer], qid: str) -> str | None:
    ans = answers.get(qid)
    return ans.choice if isinstance(ans, ChoiceAnswer) else None


def _choice_conf(answers: Mapping[str, Answer], qid: str) -> float:
    ans = answers.get(qid)
    return ans.confidence if isinstance(ans, ChoiceAnswer) else 0.0


def _noul_prob(answers: Mapping[str, Answer], qid: str) -> float | None:
    ans = answers.get(qid)
    return ans.noul if isinstance(ans, NoulAnswer) else None


def _score_argmax(answers: Mapping[str, Answer], qid: str) -> str | None:
    ans = answers.get(qid)
    if not isinstance(ans, ScoreAnswer) or not ans.probabilities:
        return None
    return max(ans.probabilities.items(), key=lambda kv: kv[1])[0]


def _resolve_auxiliary_day(
    auxiliary_answers: Mapping[str, Answer],
) -> tuple[int, str, str, float] | None:
    """Resolve a concrete day (1..31) -> (day, day_tens_key, day_unit_key, conf)."""
    low_ans = auxiliary_answers.get(AUX_DAY_LOW_ID)
    high_ans = auxiliary_answers.get(AUX_DAY_HIGH_ID)
    low_choice = low_ans.choice if isinstance(low_ans, ChoiceAnswer) else "not_1_15"
    high_choice = high_ans.choice if isinstance(high_ans, ChoiceAnswer) else "not_16_31"
    low_conf = low_ans.confidence if isinstance(low_ans, ChoiceAnswer) else 0.0
    high_conf = high_ans.confidence if isinstance(high_ans, ChoiceAnswer) else 0.0

    day: int | None = None
    conf = 0.0
    if (
        low_choice != "not_1_15"
        and low_choice.startswith("d")
        and (high_choice == "not_16_31" or low_conf >= high_conf)
    ):
        day = int(low_choice[1:])
        conf = low_conf
    elif high_choice != "not_16_31" and high_choice.startswith("d"):
        day = int(high_choice[1:])
        conf = high_conf

    if day is None or not (1 <= day <= 31):
        return None
    if 1 <= day <= 9:
        tens = "d1_9"
    elif 10 <= day <= 19:
        tens = "d10_19"
    elif 20 <= day <= 29:
        tens = "d20_29"
    else:
        tens = "d30_31"
    unit = f"u{day % 10}"
    return day, tens, unit, conf


def select_stage2_and_compile_summary(
    plan: TwoStagePlan,
    questions: Mapping[str, Question],
    stage1_answers: Mapping[str, Answer],
    auxiliary_answers: Mapping[str, Answer],
) -> tuple[list[str], str]:
    """Identify Stage-2 questions needing re-evaluation and compile the Stage-1 note.

    Returns `(stage2_question_ids, deliberation_note)`. If Stage 1 is already
    mutually consistent, `stage2_question_ids` is empty and Stage 2 is skipped.
    """
    flagged: list[str] = []
    notes: list[str] = []

    def flag(qid: str) -> None:
        if qid in questions and qid not in flagged:
            flagged.append(qid)

    if plan.family == "date_span":
        mode = _choice_val(stage1_answers, "mode")
        month = _choice_val(stage1_answers, "month")
        day_tens = _choice_val(stage1_answers, "day_tens")
        day_unit = _choice_val(stage1_answers, "day_unit")
        year = _choice_val(stage1_answers, "year")
        weekday = _choice_val(stage1_answers, "weekday")
        week_offset = _choice_val(stage1_answers, "week_offset")
        anchor = _choice_val(stage1_answers, "anchor")

        conv_rec = _noul_prob(stage1_answers, "convention_recoverable")
        conv_stated = _noul_prob(stage1_answers, "convention_stated")
        rejects_nearest = _noul_prob(stage1_answers, "rejects_nearest_occurrence")
        weekday_named = _noul_prob(stage1_answers, "weekday_named")
        part_of_day = _noul_prob(stage1_answers, "part_of_day_named")
        review_urg = _score_argmax(stage1_answers, "review_urgency")

        # 1. Unrecoverable numeric date convention -> mode=ambiguous_review,
        #    month/day_tens/day_unit=not_stated.
        if conv_rec is not None and conv_rec < 0.5:
            notes.append(
                "Stage-1 verification: convention_recoverable is false"
                + (
                    f" and review_urgency is {review_urg}"
                    if review_urg is not None
                    else ""
                )
                + ". Because the record does not supply enough information to decide "
                "which leading number is the day vs the month, the date expression "
                "cannot be read without missing information: mode must be "
                "'ambiguous_review', and month, day_tens, and day_unit must be "
                "'not_stated' (only year is stated if unambiguous)."
            )
            if mode != "ambiguous_review":
                flag("mode")
            for qid in ("month", "day_tens", "day_unit"):
                if _choice_val(stage1_answers, qid) != "not_stated":
                    flag(qid)
            return flagged, sanitize_deliberation_note("\n".join(notes))

        # 2. Check if Stage-1 `mode` was misclassified as `ambiguous_review`
        #    when (a) month is stated and year is not_stated (partial date),
        #    (b) week_offset / rejects_nearest_occurrence indicates a relative
        #    weekday expression, or (c) a relative offset like "tomorrow morning"
        #    is present.
        effective_mode = mode
        if mode == "ambiguous_review":
            if month is not None and month != "not_stated" and year == "not_stated":
                effective_mode = "absolute_partial"
                notes.append(
                    f"Stage-1 verification: month is explicitly written as '{month}' "
                    "while year is 'not_stated' (missing year, not an unreadable "
                    "format). A specific calendar date with month and day written "
                    "but missing the year has mode='absolute_partial' and 2 written "
                    "components (month and day)."
                )
                flag("mode")
            elif (week_offset is not None and week_offset != "not_applicable") or (
                rejects_nearest is not None and rejects_nearest >= 0.5
            ):
                effective_mode = "relative_weekday"
                notes.append(
                    f"Stage-1 verification: week_offset is '{week_offset}' and "
                    "rejects_nearest_occurrence is true (the sender explicitly names "
                    "a weekday and specifies the occurrence after the nearest one, "
                    "such as 'not this Friday, the Friday after that one'). Therefore "
                    "mode is 'relative_weekday', weekday is the named weekday in the "
                    "expression, and anchor is 'today'."
                )
                flag("mode")
            elif (
                part_of_day is not None
                and part_of_day >= 0.5
                and weekday_named is not None
                and weekday_named < 0.5
                and month == "not_stated"
                and year == "not_stated"
            ):
                effective_mode = "relative_offset"
                notes.append(
                    "Stage-1 verification: the caller named a relative day offset "
                    "(such as 'tomorrow morning', which is 1 day from today) without "
                    "naming a weekday or calendar month/year. Therefore mode is "
                    "'relative_offset', anchor is 'today', offset_unit is 'days', and "
                    "offset_count is 'c1'."
                )
                for qid in ("mode", "anchor", "offset_unit", "offset_count"):
                    flag(qid)

        # 3. Relative weekday / relative offset consistency rules.
        if effective_mode == "relative_weekday":
            notes.append(
                "Stage-1 verification: mode is 'relative_weekday' (a relative weekday "
                "expression such as 'next Thursday' or 'the Friday after that one'). "
                "Do not copy calendar month, day, or year from message headers or "
                "sending timestamps: month, day_tens, day_unit, and year in the target "
                "expression are 'not_stated'. The relative weekday counts from "
                "anchor='today'."
            )
            for qid in _CALENDAR_ONLY_IDS:
                if (
                    qid in questions
                    and _choice_val(stage1_answers, qid) != "not_stated"
                ):
                    flag(qid)
            if anchor == "not_applicable":
                flag("anchor")
            if (
                rejects_nearest is not None
                and rejects_nearest >= 0.5
                and weekday == "not_stated"
            ):
                notes.append(
                    "Stage-1 verification: rejects_nearest_occurrence is true "
                    "(the sender explicitly names a weekday and rules out its nearest "
                    "occurrence); report that named weekday for 'weekday' rather than "
                    "'not_stated'."
                )
                flag("weekday")

        elif effective_mode == "relative_offset":
            notes.append(
                "Stage-1 verification: mode is 'relative_offset' (an expression "
                "anchored to a reference point or period boundary, such as 'tomorrow' "
                "or 'end of last quarter'). Do not resolve a relative offset into a "
                "calendar month, day, or year, and do not copy a memo/header date: "
                "month, day_tens, day_unit, and year are 'not_stated'."
            )
            for qid in _CALENDAR_ONLY_IDS:
                if (
                    qid in questions
                    and _choice_val(stage1_answers, qid) != "not_stated"
                ):
                    flag(qid)

        elif effective_mode in ("absolute_full", "absolute_partial"):
            # Explicit calendar date -> relative offset fields are not_applicable.
            for qid in _RELATIVE_ONLY_IDS:
                if (
                    qid in questions
                    and _choice_val(stage1_answers, qid) != "not_applicable"
                ):
                    flag(qid)
                    notes.append(
                        f"Stage-1 verification: mode is '{effective_mode}' (an "
                        "explicit calendar date, not a relative offset). Therefore "
                        "relative fields (week_offset, anchor, offset_unit, "
                        "offset_count) must be 'not_applicable'."
                    )
            if (
                effective_mode == "absolute_full"
                and weekday is not None
                and weekday != "not_stated"
                and "time_of_day_stated" in questions
                and "time_window_stated" not in questions
            ):
                # ISO date like 2026-03-09 does not name a weekday unless written.
                notes.append(
                    "Stage-1 verification: for weekday, report a weekday name ONLY if "
                    "the weekday word (Monday..Sunday) is literally written in the "
                    "target expression; do not compute a weekday from a numeric "
                    "YYYY-MM-DD date."
                )
                flag("weekday")

            # Check literal day-of-month auxiliary probes when calendar date is present.
            aux_day = _resolve_auxiliary_day(auxiliary_answers)
            if aux_day is not None:
                day_num, exp_tens, exp_unit, day_conf = aux_day
                # Guard against DD/MM vs MM/DD swap on numeric dates with stated
                # convention: if aux_day equals the stated month number while
                # Stage-1 day_unit already resolved to a different single digit,
                # keep Stage-1 day_unit and only repair day_tens to d1_9.
                if (
                    conv_stated is not None
                    and conv_stated >= 0.5
                    and month in _MONTH_NUM
                    and day_num == _MONTH_NUM[month]
                    and day_unit is not None
                    and day_unit.startswith("u")
                    and day_unit not in ("u0", exp_unit)
                ):
                    day_num = int(day_unit[1:])
                    exp_tens = "d1_9"
                    exp_unit = day_unit
                if day_conf >= 0.55:
                    if "day_tens" in questions and day_tens != exp_tens:
                        flag("day_tens")
                    if "day_unit" in questions and day_unit != exp_unit:
                        flag("day_unit")
                    if "day_tens" in flagged or "day_unit" in flagged:
                        band_desc = {
                            "d1_9": "1 to 9",
                            "d10_19": "10 to 19",
                            "d20_29": "20 to 29",
                            "d30_31": "30 or 31",
                        }[exp_tens]
                        notes.append(
                            f"Stage-1 day extraction: the target date expression "
                            f"writes day {day_num} of the month. Day {day_num} falls "
                            f"in ten-day band '{exp_tens}' ({band_desc}) and has "
                            f"final digit '{exp_unit}' ({day_num % 10})."
                        )

            # Check components_written consistency against stated month/day/year.
            if "components_written" in questions:
                has_month = month is not None and month != "not_stated"
                has_day = (day_tens is not None and day_tens != "not_stated") or (
                    aux_day is not None and aux_day[3] >= 0.55
                )
                has_year = year is not None and year != "not_stated"
                expected_count = int(has_month) + int(has_day) + int(has_year)
                if effective_mode == "absolute_full":
                    expected_count = 3
                elif effective_mode == "absolute_partial" and expected_count < 2:
                    expected_count = 2
                comp_argmax = _score_argmax(stage1_answers, "components_written")
                if comp_argmax != str(expected_count):
                    flag("components_written")
                    notes.append(
                        f"Stage-1 component count: the target date expression "
                        f"explicitly writes {expected_count} of the 3 calendar "
                        f"components (month, day, year), so components_written is "
                        f"{expected_count}."
                    )

        if (
            conv_stated is not None
            and conv_stated >= 0.5
            and ("day_tens" in flagged or "day_unit" in flagged)
        ):
            notes.append(
                "Stage-1 verification: convention_stated is true; use the ledger's "
                "stated day/month/year order when reading the day of the month."
            )

    elif plan.family == "block_structure":
        block_type = _choice_val(stage1_answers, "block_type")
        line_1_type = _choice_val(stage1_answers, "line_1_type")
        callout_kind = _choice_val(stage1_answers, "callout_kind")
        heading_level = _choice_val(stage1_answers, "heading_level")
        ordered_steps = _noul_prob(stage1_answers, "ordered_steps")
        block_r_type = _choice_val(stage1_answers, "block_r_type")
        block_s_type = _choice_val(stage1_answers, "block_s_type")

        if block_type == "table_row":
            if heading_level is not None and heading_level != "not_applicable":
                flag("heading_level")
            if callout_kind is not None and callout_kind != "not_applicable":
                flag("callout_kind")
            if flagged:
                notes.append(
                    "Stage-1 verification: block_type is 'table_row' (a markdown/pipe "
                    "table row, not a heading or callout). Therefore heading_level is "
                    "'not_applicable' and callout_kind is 'not_applicable'."
                )
        elif block_type == "callout_admonition":
            if ordered_steps is not None and ordered_steps >= 0.5:
                flag("ordered_steps")
                notes.append(
                    f"Stage-1 verification: block_type is 'callout_admonition' "
                    f"(callout_kind='{callout_kind}'). A prose callout paragraph "
                    "without numbered step items has ordered_steps=false."
                )
        elif block_type == "section_heading":
            if ordered_steps is not None and ordered_steps >= 0.5:
                flag("ordered_steps")
            if (
                heading_level is not None
                and _choice_conf(stage1_answers, "heading_level") < 0.75
            ):
                flag("heading_level")
            if flagged:
                notes.append(
                    "Stage-1 verification: block_type is 'section_heading' (a single "
                    "numbered section heading line such as '2.3.1 ...', not an ordered "
                    "list of steps). Count the dot-separated section numbers to "
                    "determine heading_level (e.g. X.Y.Z with three numbers is h3), "
                    "and ordered_steps is false."
                )

        if line_1_type in ("heading", "section_heading"):
            join_1_2 = _noul_prob(stage1_answers, "join_1_2")
            if join_1_2 is not None and join_1_2 >= 0.5:
                flag("join_1_2")
                notes.append(
                    f"Stage-1 verification: line_1_type is '{line_1_type}'. A "
                    "standalone section heading line must remain separate from the "
                    "body sentence on line 2, so join_1_2 is false."
                )

        if block_r_type == "ordered_list" and block_s_type == "ordered_list":
            both_num = _noul_prob(stage1_answers, "both_numbered")
            if both_num is not None and both_num < 0.5:
                flag("both_numbered")
                notes.append(
                    "Stage-1 verification: block_r_type is 'ordered_list' and "
                    "block_s_type is 'ordered_list'. Because both blocks use "
                    "numbered list items, both_numbered is true."
                )

    elif plan.family == "passage_search":
        answer_line = _choice_val(stage1_answers, "answer_line")
        best_cand = _choice_val(stage1_answers, "best_candidate")
        aux_best = _choice_val(auxiliary_answers, AUX_BEST_CANDIDATE_ID)
        aux_conf = _choice_conf(auxiliary_answers, AUX_BEST_CANDIDATE_ID)
        contains_prob = _noul_prob(stage1_answers, "contains_answer")

        cand_ids = [qid for qid in questions if qid.startswith("cand_p")]
        pos_cands = [
            qid for qid in cand_ids if (_noul_prob(stage1_answers, qid) or 0.0) >= 0.5
        ]

        # 1. Explicit `best_candidate` Choice question in request.
        if best_cand is not None:
            expected_contains = best_cand != "none"
            if (
                contains_prob is not None
                and (contains_prob >= 0.5) != expected_contains
            ):
                flag("contains_answer")
            if best_cand != "none":
                target_qid = f"cand_{best_cand}"
                if target_qid in questions and target_qid not in pos_cands:
                    flag(target_qid)
            if flagged:
                notes.append(
                    f"Stage-1 verification: best_candidate resolved to '{best_cand}'. "
                    + (
                        f"Because candidate '{best_cand}' states the fact the query "
                        f"asks for, cand_{best_cand} is true and contains_answer is "
                        "true."
                        if best_cand != "none"
                        else (
                            "Because no candidate states the fact, contains_answer "
                            "is false."
                        )
                    )
                )

        # 2. Line search (`answer_line` Choice + `contains_answer` Noul).
        elif answer_line is not None:
            expected_contains = answer_line != "none"
            if (
                contains_prob is not None
                and (contains_prob >= 0.5) != expected_contains
            ):
                flag("contains_answer")
                notes.append(
                    f"Stage-1 verification: answer_line resolved to '{answer_line}'. "
                    + (
                        f"Because line '{answer_line}' in the document directly "
                        "answers the query, contains_answer is true."
                        if expected_contains
                        else "Because answer_line is 'none', contains_answer is false."
                    )
                )

        # 3. Passage rerank (`cand_p1..cand_pK` Noul + `contains_answer` Noul)
        #    with auxiliary `_aux_best_candidate` Choice.
        elif cand_ids:
            chosen_pid = aux_best
            # Check if contains_answer disagrees with positive candidates.
            if len(pos_cands) == 1:
                only_pid = pos_cands[0].removeprefix("cand_")
                if contains_prob is not None and contains_prob < 0.5:
                    flag("contains_answer")
                    notes.append(
                        f"Stage-1 verification: candidate passage '{only_pid}' "
                        f"({pos_cands[0]}=true) directly states the fact the query "
                        "asks for; therefore contains_answer for the retrieved set "
                        "is true."
                    )
            elif (
                len(pos_cands) > 1
                and chosen_pid is not None
                and chosen_pid != "none"
                and aux_conf >= 0.60
            ):
                # Multiple Noul candidates fired in isolation; use mutual-competition
                # Choice winner to re-evaluate the decoy candidates.
                winner_qid = f"cand_{chosen_pid}"
                for qid in pos_cands:
                    if qid != winner_qid:
                        flag(qid)
                notes.append(
                    f"Stage-1 verification: when comparing all candidate passages "
                    f"jointly against the query, only candidate '{chosen_pid}' "
                    "directly states the exact fact requested; other candidates "
                    "mention related topics or different conditions/tiers and do not "
                    f"state the requested fact (so only {winner_qid} is true)."
                )
            elif (
                len(pos_cands) == 0
                and chosen_pid is not None
                and chosen_pid != "none"
                and aux_conf >= 0.75
            ):
                winner_qid = f"cand_{chosen_pid}"
                flag(winner_qid)
                if contains_prob is not None and contains_prob < 0.5:
                    flag("contains_answer")
                notes.append(
                    f"Stage-1 verification: joint candidate comparison identified "
                    f"candidate '{chosen_pid}' (confidence {aux_conf:.2f}) as directly "
                    f"stating the requested fact; verify {winner_qid} and "
                    "contains_answer."
                )

    return flagged, sanitize_deliberation_note("\n".join(notes)) if flagged else ""


def build_micro_rider_prompt(questions: Mapping[str, Question]) -> str:
    """Build a compact prompt for bounded `/v2/rider` (`max_tokens: 8..16`)."""
    plan = plan_two_stage_questions(questions)
    if plan.family == "date_span":
        target = _extract_target_description(questions)
        return (
            f"{target} Quote the exact date phrase from the state and write its "
            "mode, month, day number, and year in one short line."
        )
    if plan.family == "passage_search":
        for q in questions.values():
            if isinstance(q.instructions, dict) and "query" in q.instructions:
                return (
                    f"Query: {q.instructions['query']}\n"
                    "Name the exact candidate id or line id (or none) that states "
                    "this fact in one short line."
                )
    return (
        "State the key intermediate fact from the state needed to answer the "
        "questions in one short line."
    )


def augment_questions_with_note(
    questions: Mapping[str, Question], note: str
) -> dict[str, Question]:
    """Return a copy of `questions` with `note` appended to instructions."""
    clean_note = sanitize_deliberation_note(note)
    augmented: dict[str, Question] = {}
    for qid, question in questions.items():
        existing = question.instructions
        new_instructions: Entry
        if existing is None:
            new_instructions = clean_note
        elif isinstance(existing, str):
            new_instructions = f"{existing}\n\n{clean_note}"
        elif isinstance(existing, list):
            new_instructions = [*existing, clean_note]
        else:
            copied = dict(existing)
            copied["stage1_verification"] = clean_note
            new_instructions = copied

        if isinstance(question, ChoiceQuestion):
            augmented[qid] = ChoiceQuestion(
                type="choice",
                instructions=new_instructions,
                criteria=question.criteria,
            )
        elif isinstance(question, NoulQuestion):
            augmented[qid] = NoulQuestion(
                type="noul",
                instructions=new_instructions,
                criteria=question.criteria,
            )
        elif isinstance(question, ScoreQuestion):
            augmented[qid] = ScoreQuestion(
                type="score",
                instructions=new_instructions,
                criteria=question.criteria,
            )
    return augmented


async def run_two_stage_deliberation(
    service: SnapshotService,
    *,
    context_snapshot_id: str,
    questions: Mapping[str, Question],
    readout_blocks: int,
    mode: DeliberationMode = "discrete_two_stage",
    continuation: ContinuationMode = "state_eval_child",
    rider_max_tokens: int = 16,
    owner: str = "anonymous",
) -> DeliberationResult:
    """Execute single-pass, discrete two-stage, or bounded micro-rider deliberation."""
    if not 1 <= rider_max_tokens <= 32:
        raise ValueError("rider_max_tokens must be between 1 and 32")

    started = time.perf_counter()

    if mode == "single_pass":
        t0 = time.perf_counter()
        response = await service.decide(
            V2DecisionRequest(
                snapshot=SnapshotRef(id=context_snapshot_id, relationship="followup"),
                questions=dict(questions),
                readout=Readout(completed_blocks=readout_blocks),
            ),
            owner=owner,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DeliberationResult(
            response=response,
            deliberation=DeliberationStepUsage(
                mode="single_pass",
                continuation=None,
                triggered_stage2=False,
                stage1_question_ids=list(questions.keys()),
                stage1_ms=elapsed_ms,
                total_ms=elapsed_ms,
            ),
        )

    if mode == "micro_rider":
        t0 = time.perf_counter()
        rider_prompt = build_micro_rider_prompt(questions)
        rider_resp = await service.ride(
            RiderRequest(
                snapshot=SnapshotRef(id=context_snapshot_id, relationship="followup"),
                prompt=rider_prompt,
                max_tokens=rider_max_tokens,
            )
        )
        stage1_ms = (time.perf_counter() - t0) * 1000.0
        note = sanitize_deliberation_note(f"Intermediate scratchpad: {rider_resp.text}")
        state_eval_ms = 0.0
        child_id: str | None = None
        effective_cont: ContinuationMode = continuation

        if continuation == "state_eval_child":
            t_se = time.perf_counter()
            se_resp = await service.state_eval(
                StateEvaluationRequest(
                    snapshot=SnapshotRef(
                        id=context_snapshot_id, relationship="followup"
                    ),
                    prompt=note,
                    readout=StateReadout(completed_blocks=readout_blocks),
                    save_result_snapshot=True,
                ),
                owner=owner,
            )
            state_eval_ms = (time.perf_counter() - t_se) * 1000.0
            child_id = se_resp.child

        t_s2 = time.perf_counter()
        try:
            if effective_cont == "state_eval_child" and child_id is not None:
                try:
                    s2_resp = await service.decide(
                        V2DecisionRequest(
                            snapshot=SnapshotRef(id=child_id, relationship="followup"),
                            questions=dict(questions),
                            readout=Readout(completed_blocks=readout_blocks),
                        ),
                        owner=owner,
                    )
                except FollowupUnsupported:
                    effective_cont = "context_instructions"
                    s2_resp = await service.decide(
                        V2DecisionRequest(
                            snapshot=SnapshotRef(
                                id=context_snapshot_id, relationship="followup"
                            ),
                            questions=augment_questions_with_note(questions, note),
                            readout=Readout(completed_blocks=readout_blocks),
                        ),
                        owner=owner,
                    )
            else:
                effective_cont = "context_instructions"
                s2_resp = await service.decide(
                    V2DecisionRequest(
                        snapshot=SnapshotRef(
                            id=context_snapshot_id, relationship="followup"
                        ),
                        questions=augment_questions_with_note(questions, note),
                        readout=Readout(completed_blocks=readout_blocks),
                    ),
                    owner=owner,
                )
        finally:
            if child_id is not None:
                await service.delete(child_id)
        stage2_ms = (time.perf_counter() - t_s2) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return DeliberationResult(
            response=s2_resp,
            deliberation=DeliberationStepUsage(
                mode="micro_rider",
                continuation=effective_cont,
                triggered_stage2=True,
                stage1_question_ids=[],
                stage2_question_ids=list(questions.keys()),
                deliberation_note=note,
                state_eval_snapshot_id=child_id,
                decision_output_tokens=0,
                rider_generated_tokens=rider_resp.usage.output_tokens,
                stage1_ms=stage1_ms,
                state_eval_ms=state_eval_ms,
                stage2_ms=stage2_ms,
                total_ms=total_ms,
            ),
        )

    # mode == "discrete_two_stage"
    plan = plan_two_stage_questions(questions)
    combined_stage1: dict[str, Question] = {
        **plan.stage1_questions,
        **plan.auxiliary_questions,
    }
    t0 = time.perf_counter()
    s1_resp = await service.decide(
        V2DecisionRequest(
            snapshot=SnapshotRef(id=context_snapshot_id, relationship="followup"),
            questions=combined_stage1,
            readout=Readout(completed_blocks=readout_blocks),
        ),
        owner=owner,
    )
    stage1_ms = (time.perf_counter() - t0) * 1000.0

    caller_s1_answers = {
        qid: s1_resp.answers[qid] for qid in questions if qid in s1_resp.answers
    }
    aux_answers = {
        qid: s1_resp.answers[qid]
        for qid in plan.auxiliary_questions
        if qid in s1_resp.answers
    }

    stage2_ids, note = select_stage2_and_compile_summary(
        plan, questions, caller_s1_answers, aux_answers
    )

    if not stage2_ids:
        total_ms = (time.perf_counter() - started) * 1000.0
        trimmed_suffix = {
            qid: s1_resp.snapshot_usage.suffix_tokens[qid] for qid in questions
        }
        trimmed_sources = (
            {
                qid: s1_resp.snapshot_usage.readout_source[qid]
                for qid in questions
                if qid in s1_resp.snapshot_usage.readout_source
            }
            if s1_resp.snapshot_usage.readout_source is not None
            else None
        )
        clean_resp = V2DecisionResponse(
            model=s1_resp.model,
            answers=caller_s1_answers,
            usage=Usage(input_tokens=s1_resp.usage.input_tokens),
            snapshot_usage=SnapshotUsage(
                requested_parent=s1_resp.snapshot_usage.requested_parent,
                effective_parent=s1_resp.snapshot_usage.effective_parent,
                promotion=s1_resp.snapshot_usage.promotion,
                profile=s1_resp.snapshot_usage.profile,
                suffix_tokens=trimmed_suffix,
                block_tokens=s1_resp.snapshot_usage.block_tokens,
                reused_prefix_tokens=s1_resp.snapshot_usage.reused_prefix_tokens,
                restored_bytes=s1_resp.snapshot_usage.restored_bytes,
                timing=s1_resp.snapshot_usage.timing,
                readout_source=trimmed_sources,
                child=None,
            ),
        )
        return DeliberationResult(
            response=clean_resp,
            deliberation=DeliberationStepUsage(
                mode="discrete_two_stage",
                continuation=None,
                triggered_stage2=False,
                stage1_question_ids=list(questions.keys()),
                auxiliary_question_ids=list(plan.auxiliary_questions.keys()),
                stage2_question_ids=[],
                deliberation_note="",
                state_eval_snapshot_id=None,
                decision_output_tokens=0,
                rider_generated_tokens=0,
                stage1_ms=stage1_ms,
                state_eval_ms=0.0,
                stage2_ms=0.0,
                total_ms=total_ms,
            ),
        )

    stage2_questions = {qid: questions[qid] for qid in stage2_ids}
    state_eval_ms = 0.0
    child_id = None
    effective_cont = continuation

    if continuation == "state_eval_child":
        t_se = time.perf_counter()
        se_resp = await service.state_eval(
            StateEvaluationRequest(
                snapshot=SnapshotRef(id=context_snapshot_id, relationship="followup"),
                prompt=note,
                readout=StateReadout(completed_blocks=readout_blocks),
                save_result_snapshot=True,
            ),
            owner=owner,
        )
        state_eval_ms = (time.perf_counter() - t_se) * 1000.0
        child_id = se_resp.child

    t_s2 = time.perf_counter()
    try:
        if effective_cont == "state_eval_child" and child_id is not None:
            try:
                s2_resp = await service.decide(
                    V2DecisionRequest(
                        snapshot=SnapshotRef(id=child_id, relationship="followup"),
                        questions=stage2_questions,
                        readout=Readout(completed_blocks=readout_blocks),
                    ),
                    owner=owner,
                )
            except FollowupUnsupported:
                effective_cont = "context_instructions"
                s2_resp = await service.decide(
                    V2DecisionRequest(
                        snapshot=SnapshotRef(
                            id=context_snapshot_id, relationship="followup"
                        ),
                        questions=augment_questions_with_note(stage2_questions, note),
                        readout=Readout(completed_blocks=readout_blocks),
                    ),
                    owner=owner,
                )
        else:
            effective_cont = "context_instructions"
            s2_resp = await service.decide(
                V2DecisionRequest(
                    snapshot=SnapshotRef(
                        id=context_snapshot_id, relationship="followup"
                    ),
                    questions=augment_questions_with_note(stage2_questions, note),
                    readout=Readout(completed_blocks=readout_blocks),
                ),
                owner=owner,
            )
    finally:
        if child_id is not None:
            await service.delete(child_id)
    stage2_ms = (time.perf_counter() - t_s2) * 1000.0

    merged_answers: dict[str, Answer] = {}
    merged_suffix: dict[str, int] = {}
    merged_sources: dict[str, str] = {}
    for qid in questions:
        if qid in s2_resp.answers:
            merged_answers[qid] = s2_resp.answers[qid]
            merged_suffix[qid] = s2_resp.snapshot_usage.suffix_tokens[qid]
            if (
                s2_resp.snapshot_usage.readout_source
                and qid in s2_resp.snapshot_usage.readout_source
            ):
                merged_sources[qid] = s2_resp.snapshot_usage.readout_source[qid]
        else:
            merged_answers[qid] = caller_s1_answers[qid]
            merged_suffix[qid] = s1_resp.snapshot_usage.suffix_tokens[qid]
            if (
                s1_resp.snapshot_usage.readout_source
                and qid in s1_resp.snapshot_usage.readout_source
            ):
                merged_sources[qid] = s1_resp.snapshot_usage.readout_source[qid]

    combined_resp = V2DecisionResponse(
        model=s1_resp.model,
        answers=merged_answers,
        usage=Usage(
            input_tokens=s1_resp.usage.input_tokens + s2_resp.usage.input_tokens
        ),
        snapshot_usage=SnapshotUsage(
            requested_parent=s1_resp.snapshot_usage.requested_parent,
            effective_parent=s1_resp.snapshot_usage.effective_parent,
            promotion=s1_resp.snapshot_usage.promotion,
            profile=s1_resp.snapshot_usage.profile,
            suffix_tokens=merged_suffix,
            block_tokens=BlockTokens(
                lower=s1_resp.snapshot_usage.block_tokens.lower
                + s2_resp.snapshot_usage.block_tokens.lower,
                upper=s1_resp.snapshot_usage.block_tokens.upper
                + s2_resp.snapshot_usage.block_tokens.upper,
            ),
            reused_prefix_tokens=s1_resp.snapshot_usage.reused_prefix_tokens,
            restored_bytes=s1_resp.snapshot_usage.restored_bytes
            + s2_resp.snapshot_usage.restored_bytes,
            timing=SnapshotTiming(
                restore_ms=s1_resp.snapshot_usage.timing.restore_ms
                + s2_resp.snapshot_usage.timing.restore_ms,
                promotion_ms=s1_resp.snapshot_usage.timing.promotion_ms
                + s2_resp.snapshot_usage.timing.promotion_ms,
                inference_ms=s1_resp.snapshot_usage.timing.inference_ms
                + s2_resp.snapshot_usage.timing.inference_ms,
            ),
            readout_source=merged_sources or None,  # type: ignore[arg-type]
            child=None,
        ),
    )
    total_ms = (time.perf_counter() - started) * 1000.0
    return DeliberationResult(
        response=combined_resp,
        deliberation=DeliberationStepUsage(
            mode="discrete_two_stage",
            continuation=effective_cont,
            triggered_stage2=True,
            stage1_question_ids=list(questions.keys()),
            auxiliary_question_ids=list(plan.auxiliary_questions.keys()),
            stage2_question_ids=stage2_ids,
            deliberation_note=note,
            state_eval_snapshot_id=child_id,
            decision_output_tokens=0,
            rider_generated_tokens=0,
            stage1_ms=stage1_ms,
            state_eval_ms=state_eval_ms,
            stage2_ms=stage2_ms,
            total_ms=total_ms,
        ),
    )


__all__ = [
    "AUX_BEST_CANDIDATE_ID",
    "AUX_DAY_HIGH_ID",
    "AUX_DAY_LOW_ID",
    "ContinuationMode",
    "DeliberationMode",
    "DeliberationResult",
    "DeliberationStepUsage",
    "TwoStagePlan",
    "augment_questions_with_note",
    "build_micro_rider_prompt",
    "plan_two_stage_questions",
    "run_two_stage_deliberation",
    "sanitize_deliberation_note",
    "select_stage2_and_compile_summary",
]
