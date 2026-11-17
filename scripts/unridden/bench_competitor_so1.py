"""Score Open Alternative to Jev (`so1`) on the unridden use case suites.

so1 (github.com/ikermoel/open-alternative-jev) answers typed questions the same
way unridden does: it never generates, it reads the next-token distribution at
one readout position and restricts it to option-letter tokens. What differs is
everything around that readout. so1 writes each question as its own chat turn,
labels the options `A.`, `B.`, ... in the user turn, reads the distribution at
the end of the generation prompt, and offers two modes: `separate`, one
sequence per question with the full state, and `packed`, every question of an
item in one sequence with the state written once.

Running it on its own weights would compare two models, not two designs. So
this driver gives so1 unridden's exact backend instead: the same GGUF, the
same llama.cpp v0.4.1 runtime, the same llama_model_params and
llama_context_params worker.cpp builds (unridden/api/native/so1_probe.cpp).
Weights, engine and kernels are therefore held fixed and the only variables
left are so1's prompt format and its readout.

What this driver records, and does not paper over:

* so1's prompt is not unridden's prompt, and cannot be. unridden's content is
  carried over unchanged -- the same serialized state (`compiler.render`), the
  same question heading (`compiler._question_heading`), the same instruction
  text and the same per-option rubric text, in the same option order, so the
  option letters land on the same token ids -- but the frame around it is so1's:
  so1's `DEFAULT_INSTRUCTION`/`FOLLOW_UP_INSTRUCTION` replace unridden's
  `SYSTEM_INSTRUCTION`, options are rendered `A. <id> - <rubric>` instead of
  `A: <id> - <rubric>`, there is no `Answer:\\n` prefix, and the readout sits at
  the end of the chat template's generation prompt. Every one of those is a
  prompt difference on purpose; none of them is a scoring difference.
* Gemma 4 is not ChatML, so so1's default `ChatFormat` raises on this
  tokenizer. `GEMMA_FORMAT` below is the same structure expressed in Gemma 4's
  turn markers, verified against the tokenizer's own multi-turn rendering.
* so1's packed layout reuses the generation prompt for every turn, so each
  completed placeholder turn keeps the template's empty thought channel
  (`<|channel>thought\\n<channel|>_<turn|>`) where a real Gemma transcript would
  have just the answer. That is so1's construction, not a transcription error,
  and it is what the packed row measures.
* `raw_label_logits` in each observation row are real logits read from
  `llama_get_logits_ith` at so1's readout position -- the same quantity
  unridden reports, not a reconstruction from probabilities.
* `coverage` is the probability mass the option letters hold against the full
  262144-token vocabulary at the readout, the same statistic unridden reports.
  The helper returns it for the whole letter set the decider passed; the
  per-question value is recovered exactly from the label logits, because the
  two masses share a denominator.
* Latency is wall clock in-process around one helper call per case, the same
  boundary the unridden reference measures around its own owned child. In
  `separate` mode that one call carries one sequence per question; in `packed`
  mode one sequence for the whole case.
* so1 exposes an optional `label_scores_last` fast path for separate mode. It
  is deliberately not implemented, so both modes reach the model through the
  identical `label_scores` path and a separate-vs-packed difference cannot come
  from a different code path.

Usage (so1 and the suites both have to be importable):

    PYTHONPATH=.:scripts/unridden:/path/to/open-alternative-jev \\
    uv run --with transformers python -P \\
        scripts/unridden/bench_competitor_so1.py \\
        --suites unridden/examples/usecases \\
        --out outputs/competitor-so1-v1-separate \\
        --mode separate \\
        --helper build/so1-probe/unridden-so1-probe \\
        --runtime-dir build/llama-base/runtime/bin \\
        --model-path /path/to/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \\
        --tokenizer google/gemma-4-26B-A4B-it \\
        --allow-gpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import so1
from run_usecase_suites import (  # type: ignore[import-not-found]
    Case,
    Outcome,
    RankOutcome,
    Suite,
    build_summary,
    load_suites,
    score_answer,
    score_ranking,
    write_json,
)
from so1.decider import Decider
from so1.prompting import ChatFormat, PromptBuilder
from so1.schema import LETTERS, Choice, Decision
from transformers import AutoTokenizer

# `_options` and `_question_heading` decide the option ids, their order and the
# score legend. Importing them rather than restating them is what keeps so1's
# option letters on the same token ids, and the score legend on the same
# levels, as the unridden run this is compared against.
from unridden.api.compiler import (
    ANSWER_PREFIX,
    SYSTEM_INSTRUCTION,
    _options,
    _question_heading,
    compile_request,
    render,
)
from unridden.api.schema import (
    Answer,
    BackendProfile,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

Row = dict[str, Any]

# so1's ChatML `ChatFormat` in Gemma 4's turn markers. `user_turn_start` is
# what `PromptBuilder._strip_system` searches for to drop the BOS and any
# system preamble from a follow-up turn; `separator` closes the placeholder
# model turn so the next user turn starts where the template expects one.
# Verified against `apply_chat_template` on a real three-turn conversation.
GEMMA_FORMAT = ChatFormat(user_turn_start="<|turn>user", separator="_<turn|>\n")

PROBABILITY_SEMANTICS = "softmax_over_so1_option_letter_logits_v1"
LOGIT_SEMANTICS = "llama_cpp_logits_at_so1_readout_position_v1"
LOGIT_NOTE = (
    "Real logits from llama_get_logits_ith at so1's readout position, on the "
    "same GGUF and llama.cpp v0.4.1 runtime as the unridden reference. They "
    "are the same quantity as unridden's raw_label_logits, read after a "
    "different prompt at a different position."
)
CONFIDENCE_NOTE = (
    "answers[].confidence is unridden's max label probability, computed from "
    "so1's own softmax over the option-letter logits. so1's Decision.confidence "
    "is the same quantity (schema.py Decision.confidence), so nothing is "
    "recomputed here."
)
PROMPT_NOTE = (
    "so1's prompt frame around unridden's own state, heading, instruction and "
    "rubric text. See the module docstring for every difference."
)


class RequestFailed(RuntimeError):
    """so1 or the helper refused a case. Evidence, not a reason to stop."""


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class LlamaCppBackend:
    """so1's `Backend` protocol, served by unridden's own llama.cpp runtime.

    `label_scores` is the whole contract (so1/backends/base.py): token
    sequences and readout positions in, a logit per label per readout out. The
    helper clears the KV cache before every sequence, so a sequence's scores
    depend on nothing but that sequence.
    """

    def __init__(
        self,
        executable: Path,
        model_path: Path,
        tokenizer: Any,
        *,
        runtime_dir: Path,
        context: int,
        batch: int,
        ubatch: int,
        threads: int,
        gpu: bool,
        stderr_log: Path,
    ) -> None:
        self.tokenizer = tokenizer
        command = [
            str(executable),
            "--model",
            str(model_path),
            "--runtime-dir",
            str(runtime_dir),
            "--context",
            str(context),
            "--batch",
            str(batch),
            "--ubatch",
            str(ubatch),
            "--threads",
            str(threads),
        ]
        if gpu:
            command.append("--gpu")
        self.command = command
        self._log = stderr_log.open("wb")
        self._process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
        )
        self._counter = 0
        self.hello: Row = self._exchange({"type": "hello"}, handshake=True)
        self.calls = 0
        self.helper_seconds = 0.0
        # What the most recent `label_scores` call cost, so the caller can put
        # per-question token counts and decode times in its observation rows.
        self.last_call: Row = {}

    def _exchange(self, payload: Row, *, handshake: bool = False) -> Row:
        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:  # pragma: no cover - Popen contract
            raise RequestFailed("helper pipes are not open")
        if not handshake:
            self._counter += 1
            payload = {**payload, "id": str(self._counter)}
            stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            stdin.flush()
        line = stdout.readline()
        if not line:
            raise RequestFailed(
                f"helper exited with status {self._process.poll()}; "
                f"see {self._log.name}"
            )
        reply: Row = json.loads(line)
        if reply.get("type") == "error":
            raise RequestFailed(str(reply.get("error")))
        if not handshake and reply.get("id") != payload["id"]:
            raise RequestFailed("helper replied to the wrong request id")
        return reply

    def tokenize(self, text: str, *, add_special: bool = False) -> list[int]:
        reply = self._exchange(
            {"type": "tokenize", "text": text, "add_special": add_special}
        )
        return [int(value) for value in reply["ids"]]

    def label_scores(
        self,
        sequences: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        label_ids: Sequence[int],
    ) -> list[list[list[float]]]:
        started = time.perf_counter()
        reply = self._exchange(
            {
                "type": "scores",
                "label_ids": [int(value) for value in label_ids],
                "sequences": [
                    {
                        "ids": [int(value) for value in ids],
                        "positions": [int(value) for value in row],
                    }
                    for ids, row in zip(sequences, positions, strict=True)
                ],
            }
        )
        self.calls += 1
        self.helper_seconds += time.perf_counter() - started
        scores = [
            [[float(value) for value in row] for row in sequence]
            for sequence in reply["scores"]
        ]
        self.last_call = {
            "prompt_tokens": [int(value) for value in reply["prompt_tokens"]],
            "decode_ms": [float(value) for value in reply["decode_ms"]],
            "readouts": reply["readouts"],
            "positions": [list(row) for row in positions],
            "helper_total_ms": float(reply["total_ms"]),
            "label_ids": [int(value) for value in label_ids],
            # The full letter-set row per readout. `Decision.raw_scores` is
            # already truncated to a question's own options, and the coverage
            # arithmetic needs the untruncated row.
            "scores": scores,
        }
        return scores

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=30)
        self._log.close()

    @property
    def pid(self) -> int:
        return self._process.pid


def so1_question(question: Question) -> Choice:
    """One unridden question as the `Choice` so1 answers.

    unridden's own option ids, option order, heading, instruction text and
    per-option rubric text are carried over verbatim; only the frame is so1's.
    """
    option_ids, descriptions, _ = _options(question)
    options: list[str] = []
    for option_id, description in zip(option_ids, descriptions, strict=True):
        rendered = render(description)
        options.append(f"{option_id} - {rendered}" if rendered else option_id)
    heading = _question_heading(question)
    instructions = render(question.instructions)
    text = f"{heading}\n{instructions}" if instructions else heading
    return Choice(text, tuple(options))


def per_question_coverage(
    label_logits: Sequence[float], option_count: int, whole_set_mass: float
) -> float:
    """The option letters' share of the full vocabulary, for one question.

    The helper reports the mass of every letter the decider asked for, which for
    a case of mixed option counts is a superset of this question's letters. Both
    masses are softmaxes over the same vocabulary, so the ratio of their
    numerators converts one into the other exactly.
    """
    maximum = max(label_logits)
    whole = math.fsum(math.exp(value - maximum) for value in label_logits)
    part = math.fsum(math.exp(value - maximum) for value in label_logits[:option_count])
    return whole_set_mass * part / whole


def map_answers(
    case: Case, decisions: Sequence[Decision], call: Row, mode: str
) -> tuple[dict[str, Answer], Row, int]:
    """so1's decisions as unridden `Answer` objects, plus diagnostics."""
    answers: dict[str, Answer] = {}
    diagnostics: Row = {}
    questions = list(case.request.questions.items())
    if len(decisions) != len(questions):
        raise RequestFailed(
            f"so1 returned {len(decisions)} decisions for {len(questions)} questions"
        )
    label_ids = call["label_ids"]
    for index, ((qid, question), decision) in enumerate(
        zip(questions, decisions, strict=True)
    ):
        option_ids, _, legend = _options(question)
        probabilities = list(decision.probabilities)
        if len(probabilities) != len(option_ids):
            raise RequestFailed(f"{qid!r}: so1 returned {len(probabilities)} options")
        total = math.fsum(probabilities)
        if not math.isfinite(total) or not 0.99 <= total <= 1.01:
            raise RequestFailed(f"{qid!r}: probabilities sum to {total}")
        mapped = dict(zip(option_ids, probabilities, strict=True))
        confidence = max(probabilities)
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        if isinstance(question, ChoiceQuestion):
            answers[qid] = ChoiceAnswer(
                choice=option_ids[best], probabilities=mapped, confidence=confidence
            )
        elif isinstance(question, ScoreQuestion):
            if legend is None:  # pragma: no cover - _options always supplies one
                raise RequestFailed(f"{qid!r}: score question has no legend")
            answers[qid] = ScoreAnswer(
                # unridden's expected value over the level distribution
                # (unridden/api/mapping.py), computed the same way here.
                score=math.fsum(
                    level * value for level, value in enumerate(probabilities)
                ),
                legend=legend,
                probabilities=mapped,
                confidence=confidence,
            )
        else:
            assert isinstance(question, NoulQuestion)
            # unridden orders a noul's labels true, false (compiler.py), and
            # so1 sees the same order, so p[0] is p(true) on both sides.
            answers[qid] = NoulAnswer(noul=probabilities[0])
        # packed writes every question of the case into one sequence, one
        # readout each; separate gives every question its own sequence with a
        # single readout.
        sequence, position_index = (0, index) if mode == "packed" else (index, 0)
        summary = call["readouts"][sequence][position_index]
        diagnostics[qid] = {
            "raw_label_logits": list(decision.raw_scores),
            "logit_semantics": LOGIT_SEMANTICS,
            "logit_note": LOGIT_NOTE,
            "probability_semantics": PROBABILITY_SEMANTICS,
            "token_mapping": dict(
                zip(
                    LETTERS[: len(option_ids)],
                    label_ids[: len(option_ids)],
                    strict=True,
                )
            ),
            "coverage": per_question_coverage(
                call["scores"][sequence][position_index],
                len(option_ids),
                float(summary["allowed_label_mass"]),
            ),
            "full_vocabulary_argmax": summary["full_vocabulary_argmax"],
            "confidence_formula": "max_probability_v1",
            "confidence_note": CONFIDENCE_NOTE,
            "prompt_note": PROMPT_NOTE,
            "so1_mode": mode,
            "prompt_tokens": call["prompt_tokens"][sequence],
            "readout_position": call["positions"][sequence][position_index],
            "sequence_index": sequence,
            "decode_ms": call["decode_ms"][sequence],
            "cache_cleared": True,
            "generated_tokens": 0,
        }
    input_tokens = sum(call["prompt_tokens"])
    return answers, diagnostics, input_tokens


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Rank-based AUROC of `scores` for predicting label True, ties averaged."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and scores[order[stop + 1]] == scores[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    positive_rank_sum = math.fsum(
        rank for rank, label in zip(ranks, labels, strict=True) if label
    )
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 15
) -> float | None:
    """Binned |confidence - accuracy|, the same statistic the other drivers use."""
    if not confidences:
        return None
    edges = [index / bins for index in range(bins + 1)]
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        members = [
            (value, flag)
            for value, flag in zip(confidences, correct, strict=True)
            if (value > low or (low == 0.0 and value == 0.0)) and value <= high
        ]
        if not members:
            continue
        share = len(members) / len(confidences)
        error += share * abs(
            statistics.fmean([value for value, _ in members])
            - statistics.fmean([float(flag) for _, flag in members])
        )
    return error


def coverage_table(confidences: Sequence[float], correct: Sequence[bool]) -> list[Row]:
    """Accuracy over the most confident fraction of answers, at fixed coverages."""
    if not confidences:
        return []
    order = sorted(range(len(confidences)), key=lambda index: -confidences[index])
    table: list[Row] = []
    for coverage in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        take = max(1, round(coverage * len(order)))
        selected = order[:take]
        table.append(
            {
                "coverage": coverage,
                "n": take,
                "accuracy": statistics.fmean(
                    [float(correct[index]) for index in selected]
                ),
                "confidence_threshold": confidences[selected[-1]],
            }
        )
    return table


def calibration_block(confidences: Sequence[float], correct: Sequence[bool]) -> Row:
    return {
        "n": len(confidences),
        "accuracy": (
            statistics.fmean([float(flag) for flag in correct]) if correct else None
        ),
        "mean_confidence": (statistics.fmean(confidences) if confidences else None),
        "ece_15_bin": expected_calibration_error(confidences, correct),
        "auroc_for_correctness": auroc(confidences, correct),
        "accuracy_at_coverage": coverage_table(confidences, correct),
    }


def brier_block(outcomes: Sequence[Outcome]) -> Row:
    noul = [item for item in outcomes if item.kind == "noul"]
    choice = [item for item in outcomes if item.kind == "choice"]
    score = [item for item in outcomes if item.kind == "score"]

    def multiclass(items: Sequence[Outcome], gold: Any) -> float | None:
        if not items:
            return None
        totals = []
        for item in items:
            target = str(gold(item))
            totals.append(
                math.fsum(
                    (value - (1.0 if key == target else 0.0)) ** 2
                    for key, value in item.probabilities.items()
                )
            )
        return statistics.fmean(totals)

    return {
        "noul_brier": (
            statistics.fmean(
                [
                    (float(item.p_true or 0.0) - float(bool(item.expected))) ** 2
                    for item in noul
                ]
            )
            if noul
            else None
        ),
        "choice_multiclass_brier": multiclass(choice, lambda item: item.expected),
        "score_multiclass_brier": multiclass(score, lambda item: int(item.expected)),
        "note": (
            "noul_brier matches summary.json overall.noul.brier. The multiclass "
            "figures are sum over labels of (p - y)^2, one per question."
        ),
    }


def reference_calibration(path: Path | None) -> Row | None:
    """The same calibration block for a unridden observations.jsonl, if given."""
    if path is None or not path.is_file():
        return None
    confidences: list[float] = []
    correct: list[bool] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for outcome in row.get("outcomes", []):
                confidences.append(float(outcome["confidence"]))
                correct.append(bool(outcome["correct"]))
    block = calibration_block(confidences, correct)
    block["source"] = str(path)
    block["note"] = (
        "Computed from the unridden reference run's own observation rows, with "
        "the same confidence definition (max label probability) on both sides."
    )
    return block


def check_tokenizers(
    backend: LlamaCppBackend, builder: PromptBuilder, suites: Sequence[Suite], mode: str
) -> Row:
    """Do the HF tokenizer and the GGUF vocabulary agree on every prompt sent?

    so1 builds its sequences with the Hugging Face tokenizer; the model behind
    the helper carries llama.cpp's own copy of that vocabulary. If the two
    disagree, a so1 row would be measuring a tokenizer gap rather than a prompt
    design, so every prompt this run will send is decoded back to text, handed
    to the helper's `llama_tokenize`, and compared id for id.
    """
    tokenizer = backend.tokenizer
    prompts = 0
    mismatched = 0
    examples: list[Row] = []
    for suite in suites:
        for case in suite.cases:
            state = render(case.request.state)
            questions = [
                so1_question(question) for question in case.request.questions.values()
            ]
            built = (
                [builder.packed(state, questions)]
                if mode == "packed"
                else builder.separate(state, questions)
            )
            for prompt in built:
                prompts += 1
                text = tokenizer.decode(prompt.ids)
                native = backend.tokenize(text)
                if native != list(prompt.ids):
                    mismatched += 1
                    if len(examples) < 5:
                        examples.append(
                            {
                                "suite": suite.slug,
                                "case": case.name,
                                "hf_tokens": len(prompt.ids),
                                "llama_tokens": len(native),
                                "first_difference": next(
                                    (
                                        index
                                        for index, (left, right) in enumerate(
                                            zip(prompt.ids, native, strict=False)
                                        )
                                        if left != right
                                    ),
                                    min(len(prompt.ids), len(native)),
                                ),
                            }
                        )
    letter_ids = {
        letter: list(tokenizer.encode(letter, add_special_tokens=False))
        for letter in LETTERS
    }
    native_letters = {letter: backend.tokenize(letter) for letter in LETTERS}
    return {
        "prompts_checked": prompts,
        "prompts_mismatched": mismatched,
        "identical": mismatched == 0,
        "examples": examples,
        "letters_single_token_hf": all(len(ids) == 1 for ids in letter_ids.values()),
        "letters_agree": letter_ids == native_letters,
        "letter_token_ids": {
            letter: ids[0] for letter, ids in letter_ids.items() if len(ids) == 1
        },
        "method": (
            "every prompt this run sends is decoded with the Hugging Face "
            "tokenizer and re-tokenized by the helper's llama_tokenize with "
            "parse_special on and add_special off, then compared id for id"
        ),
    }


def check_reference_prompts(
    backend: LlamaCppBackend, reference: Path | None, limit: int
) -> Row | None:
    """Does the HF chat template reproduce the unridden worker's prompt length?

    The worker renders its prompt from the template inside the GGUF; this
    driver's tokenizer reads the template from the Hugging Face repository. The
    reference run recorded a `prompt_tokens` for every question it asked, so the
    two templates can be compared directly on the run being compared against.
    """
    if reference is None or not reference.is_file():
        return None
    tokenizer = backend.tokenizer
    checked = 0
    matched = 0
    differences: list[Row] = []
    with reference.open(encoding="utf-8") as stream:
        for line in stream:
            if checked >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            for qid, diagnostic in row["response"]["diagnostics"].items():
                if checked >= limit:
                    break
                message = reference_prompt_text(row["request"], qid)
                if message is None:
                    continue
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": message}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                ids = backend.tokenize(str(rendered) + ANSWER_PREFIX)
                checked += 1
                if len(ids) == int(diagnostic["prompt_tokens"]):
                    matched += 1
                elif len(differences) < 5:
                    differences.append(
                        {
                            "suite": row["suite"],
                            "case": row["case"],
                            "question": qid,
                            "reference_prompt_tokens": diagnostic["prompt_tokens"],
                            "rebuilt_prompt_tokens": len(ids),
                        }
                    )
    return {
        "reference": str(reference),
        "questions_checked": checked,
        "prompt_token_counts_matched": matched,
        "identical": checked > 0 and matched == checked,
        "differences": differences,
        "system_instruction_sha256": hashlib.sha256(
            SYSTEM_INSTRUCTION.encode()
        ).hexdigest(),
        "method": (
            "unridden's own compiled user message is re-rendered with the "
            "Hugging Face chat template, the worker's 'Answer:\\n' suffix is "
            "appended, the result is tokenized by the helper, and the token "
            "count is compared with the prompt_tokens the reference recorded"
        ),
    }


def reference_prompt_text(request: Row, qid: str) -> str | None:
    """Rebuild unridden's compiled user message for one question of a case."""
    profile = BackendProfile(
        model_id="local-gemma-unridden-v1",
        model_name="rebuild",
        model_sha256="0" * 64,
        runtime_sha256="0" * 64,
        llama_revision="rebuild",
        tested_revision=False,
        labels=list(LETTERS),
        label_token_ids=list(range(len(LETTERS))),
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
    batch = compile_request(DecisionRequest.model_validate(request), profile)
    for compiled in batch.questions:
        if compiled.id == qid:
            return str(compiled.messages[0]["content"])
    return None


def prompt_size_report(
    builder: PromptBuilder, suites: Sequence[Suite], mode: str
) -> Row:
    """How long the sequences of this mode get, before the model is touched."""
    lengths: list[int] = []
    longest: Row = {}
    for suite in suites:
        for case in suite.cases:
            state = render(case.request.state)
            questions = [
                so1_question(question) for question in case.request.questions.values()
            ]
            built = (
                [builder.packed(state, questions)]
                if mode == "packed"
                else builder.separate(state, questions)
            )
            for prompt in built:
                lengths.append(len(prompt.ids))
                if not longest or len(prompt.ids) > int(longest["tokens"]):
                    longest = {
                        "suite": suite.slug,
                        "case": case.name,
                        "questions": len(questions),
                        "tokens": len(prompt.ids),
                    }
    return {
        "mode": mode,
        "sequences": len(lengths),
        "tokens_total": sum(lengths),
        "tokens_median": statistics.median(lengths) if lengths else None,
        "tokens_max": max(lengths, default=0),
        "longest_sequence": longest,
    }


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=False)
    manifest: Row = {
        "state": "starting",
        "competitor": "so1",
        "competitor_name": "Open Alternative to Jev",
        "repository": "https://github.com/ikermoel/open-alternative-jev",
        "mode": args.mode,
        "transport": (
            "in-process so1 Decider over a subprocess llama.cpp helper, one "
            "helper call per case"
        ),
        "started_unix": time.time(),
    }
    # Labels are frozen before the model is touched: validate, then hash.
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    builder = PromptBuilder(tokenizer, GEMMA_FORMAT)
    sizes = prompt_size_report(builder, suites, args.mode)
    manifest["prompt_sizes"] = sizes
    if int(sizes["tokens_max"]) > args.context:
        manifest["state"] = "refused"
        manifest["error"] = (
            f"longest {args.mode} sequence is {sizes['tokens_max']} tokens but "
            f"--context is {args.context}"
        )
        write_json(args.out / "run.json", manifest)
        print("SO1_RUN_REFUSED", manifest["error"], flush=True)
        return 1

    started = time.perf_counter()
    backend = LlamaCppBackend(
        args.helper,
        args.model_path,
        tokenizer,
        runtime_dir=args.runtime_dir,
        context=args.context,
        batch=args.batch_size,
        ubatch=args.batch_size,
        threads=args.threads,
        gpu=args.allow_gpu,
        stderr_log=args.out / "helper.log",
    )
    manifest["load_seconds"] = time.perf_counter() - started
    manifest["helper"] = {
        "executable": str(args.helper),
        "sha256": digest(args.helper),
        "source": "unridden/api/native/so1_probe.cpp",
        "command": backend.command,
        "handshake": backend.hello,
        "pid": backend.pid,
    }
    manifest["model"] = {
        "path": str(args.model_path),
        "sha256": digest(args.model_path),
    }
    if args.build_manifest is not None:
        manifest["base_runtime"] = json.loads(
            args.build_manifest.read_text(encoding="utf-8")
        )
    manifest["so1"] = {
        "module": str(Path(so1.__file__).resolve().parent),
        "chat_format": {
            "user_turn_start": GEMMA_FORMAT.user_turn_start,
            "separator": GEMMA_FORMAT.separator,
            "chat_template_kwargs": dict(GEMMA_FORMAT.chat_template_kwargs),
            "note": (
                "so1's shipped ChatML ChatFormat raises on the Gemma 4 "
                "tokenizer; this is the same structure in Gemma 4's markers"
            ),
        },
        "temperature": args.temperature,
        "mode": args.mode,
        "label_scores_last_implemented": False,
        "tokenizer": args.tokenizer,
    }
    try:
        return benchmark(args, manifest, suites, builder, backend, sizes)
    finally:
        # The helper holds the whole model in VRAM on a shared GPU, so it
        # is shut down whether the benchmark finished, refused or raised.
        backend.close()


def benchmark(
    args: argparse.Namespace,
    manifest: Row,
    suites: Sequence[Suite],
    builder: PromptBuilder,
    backend: LlamaCppBackend,
    sizes: Row,
) -> int:
    """Check the tokenizers, answer every case, and write the reports."""
    observations = args.out / "observations.jsonl"
    try:
        manifest["tokenizer_equivalence"] = check_tokenizers(
            backend, builder, suites, args.mode
        )
        manifest["reference_prompt_equivalence"] = check_reference_prompts(
            backend, args.reference, args.reference_prompt_limit
        )
    except RequestFailed as error:
        manifest["state"] = "failed"
        manifest["error"] = f"tokenizer check failed: {error}"
        write_json(args.out / "run.json", manifest)
        print("SO1_RUN_FAILED", manifest["error"], flush=True)
        return 1
    equivalence = manifest["tokenizer_equivalence"]
    if not equivalence["identical"] and not args.allow_tokenizer_mismatch:
        manifest["state"] = "refused"
        manifest["error"] = (
            f"{equivalence['prompts_mismatched']} of "
            f"{equivalence['prompts_checked']} prompts tokenize differently "
            "under the Hugging Face tokenizer and the GGUF vocabulary"
        )
        write_json(args.out / "run.json", manifest)
        print("SO1_RUN_REFUSED", manifest["error"], flush=True)
        return 1
    manifest["state"] = "running"
    write_json(args.out / "run.json", manifest)

    decider = Decider(
        backend, prompt_builder=builder, temperature=args.temperature, mode=args.mode
    )
    outcomes: list[Outcome] = []
    rankings: list[RankOutcome] = []
    errors: list[Row] = []
    latencies: list[float] = []
    input_tokens = 0
    index = 0
    for suite in suites:
        for case in suite.cases:
            state = render(case.request.state)
            questions = [
                so1_question(question) for question in case.request.questions.values()
            ]
            failure: str | None = None
            answers: dict[str, Answer] = {}
            diagnostics: Row = {}
            case_tokens = 0
            request_started = time.perf_counter()
            try:
                decisions = decider.decide(state, questions)
                answers, diagnostics, case_tokens = map_answers(
                    case, decisions, backend.last_call, args.mode
                )
            except (RequestFailed, ValueError, RuntimeError, KeyError) as error:
                # A refused case is evidence, not a reason to stop.
                failure = f"{type(error).__name__}: {error}"
            elapsed_ms = (time.perf_counter() - request_started) * 1000
            record: Row = {
                "index": index,
                "suite": suite.slug,
                "case": case.name,
                "difficulty": case.difficulty,
                "rationale": case.rationale,
                "request": case.request.model_dump(mode="json"),
                "so1_request": {
                    "mode": args.mode,
                    "state": state,
                    "questions": [
                        {"question": item.question, "options": list(item.options)}
                        for item in questions
                    ],
                },
                "latency_ms": elapsed_ms,
            }
            index += 1
            if failure is not None:
                record["status"] = "error"
                record["error"] = failure
                errors.append(
                    {
                        "suite": suite.slug,
                        "case": case.name,
                        "difficulty": case.difficulty,
                        "questions": len(case.targets),
                        "error": failure,
                    }
                )
            else:
                latencies.append(elapsed_ms)
                input_tokens += case_tokens
                case_outcomes = [
                    score_answer(suite.slug, case, target, answers[target.qid])
                    for target in case.targets
                ]
                case_rankings = [
                    score_ranking(suite.slug, case, ranking, answers)
                    for ranking in case.rankings
                ]
                outcomes.extend(case_outcomes)
                rankings.extend(case_rankings)
                record["status"] = "ok"
                record["response"] = {
                    "model": "so1-on-unridden-llama-cpp",
                    "answers": {
                        qid: answer.model_dump(mode="json")
                        for qid, answer in answers.items()
                    },
                    "usage": {"input_tokens": case_tokens, "output_tokens": 0},
                    "diagnostics": diagnostics,
                }
                record["outcomes"] = [item.as_row() for item in case_outcomes]
                record["rankings"] = [item.as_row() for item in case_rankings]
            with observations.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("SUITE_COMPLETE", suite.slug, flush=True)

    summary = build_summary(
        suites,
        outcomes,
        rankings,
        errors,
        latencies,
        {"generated_tokens": 0, "input_tokens": input_tokens},
    )
    write_json(args.out / "summary.json", summary)

    posed = int(manifest["questions"])
    correct = sum(item.correct for item in outcomes)
    confidences = [item.confidence for item in outcomes]
    was_correct = [item.correct for item in outcomes]
    write_json(
        args.out / "so1-extras.json",
        {
            "mode": args.mode,
            "answered": {
                "questions_posed": posed,
                "questions_scored": len(outcomes),
                "questions_unanswered": posed - len(outcomes),
                "accuracy_scored_only": (correct / len(outcomes) if outcomes else None),
                "accuracy_unanswered_counted_wrong": correct / posed if posed else None,
                "note": (
                    "summary.json scores only the questions so1 answered, the "
                    "convention run_usecase_suites uses. "
                    "accuracy_unanswered_counted_wrong is the strict figure over "
                    "every question the suites pose."
                ),
            },
            "calibration_max_probability": calibration_block(confidences, was_correct),
            "calibration_unridden_reference": reference_calibration(args.reference),
            "brier": brier_block(outcomes),
            "prompt_sizes": sizes,
            "helper_calls": backend.calls,
            "helper_seconds": backend.helper_seconds,
            "probability_semantics": PROBABILITY_SEMANTICS,
            "confidence_note": CONFIDENCE_NOTE,
            "prompt_note": PROMPT_NOTE,
            "latency_note": (
                "summary.json request_latency_ms is wall clock around one "
                "so1 Decider.decide call per case, which is one helper call: "
                "one sequence per question in separate mode, one sequence for "
                "the whole case in packed mode. The unridden reference "
                "measures in-process calls to its own owned llama.cpp child, "
                "answering a case one question at a time, on this same GPU, "
                "GGUF and runtime."
            ),
        },
    )
    manifest["state"] = "complete"
    manifest["errored_cases"] = len(errors)
    manifest["helper_calls"] = backend.calls
    manifest["finished_unix"] = time.time()
    write_json(args.out / "run.json", manifest)
    accuracy = summary["overall"]["accuracy"]
    print(
        "SO1_RUN_COMPLETE",
        f"mode={args.mode}",
        f"questions={len(outcomes)}",
        f"accuracy={accuracy if accuracy is None else round(accuracy, 4)}",
        f"errored_cases={len(errors)}",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("separate", "packed"), required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        required=True,
        help="directory holding the ggml backend libraries the helper dlopens, "
        "normally the base runtime's runtime/bin",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        default="google/gemma-4-26B-A4B-it",
        help="Hugging Face repository so1's PromptBuilder tokenizes with",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        help=(
            "helper context. The longest packed sequence over these suites is "
            "2138 tokens, against unridden's own 2048, so both modes get one "
            "context that covers the longer of them. n_ctx sizes the KV buffer "
            "and does not enter the attention arithmetic, so it is not a "
            "numeric difference from the reference run"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="helper batch and ubatch, the worker's own default",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="so1 Decider temperature; 1.0 is raw, as unridden reports raw",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="a unridden observations.jsonl to cross-check and compare against",
    )
    parser.add_argument(
        "--reference-prompt-limit",
        type=int,
        default=200,
        help="how many reference questions to re-render for the template check",
    )
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=None,
        help="the base runtime build.json to copy into run.json",
    )
    parser.add_argument(
        "--allow-tokenizer-mismatch",
        action="store_true",
        help="record a tokenizer disagreement and run anyway instead of refusing",
    )
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_gpu:
        parser.error(
            "--allow-gpu is required: run `gpu-policy check` first "
            "(GO = pass --allow-gpu)"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
