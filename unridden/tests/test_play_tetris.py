"""CPU tests for the Tetris example's engine and its Unridden agents.

The model is replaced by a fake that answers over the same /v2 request shapes.
No test loads a model or touches the GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.unridden.play_tetris import (
    HEIGHT,
    WELL,
    WIDTH,
    Unridden,
    choose_choice,
    choose_score,
    dominates,
    empty_board,
    placements,
    prune,
    realtime_landing,
    run_game,
    shift_frames,
    strategy_filter,
)


class FakeUnridden(Unridden):
    """Answers every Choice with its last option and scores by question order."""

    def __init__(self) -> None:
        super().__init__("http://fake")
        self.decision_bodies: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def _req(self, method: str, path: str, body: Any = None) -> Any:
        out = self._answer(method, path, body)
        self.exchanges.append(
            {"method": method, "path": path, "request": body, "response": out}
        )
        return out

    def _answer(self, method: str, path: str, body: Any) -> Any:
        if method == "POST" and path == "/v2/snapshots":
            return {"snapshots": [{"id": f"snap_{len(self.decision_bodies)}"}]}
        if method == "DELETE":
            self.deleted.append(path)
            return None
        assert path == "/v2/decisions"
        self.decision_bodies.append(body)
        answers: dict[str, Any] = {}
        for index, (qid, question) in enumerate(body["questions"].items()):
            if question["type"] == "choice":
                keys = list(question["criteria"])
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[-1],
                    "probabilities": {k: 1 / len(keys) for k in keys},
                    "confidence": 1 / len(keys),
                }
            else:
                answers[qid] = {
                    "type": "score",
                    "score": float(index),
                    "probabilities": {},
                    "legend": {},
                    "confidence": 1.0,
                }
        return {"answers": answers, "usage": {"input_tokens": 10}}


def test_empty_board_placement_counts() -> None:
    counts = {p: len(placements(empty_board(), p)) for p in "IOTSZJL"}
    assert counts == {"I": 17, "O": 9, "T": 34, "S": 17, "Z": 17, "J": 34, "L": 34}


def test_horizontal_i_clears_the_bottom_row() -> None:
    board = empty_board()
    board[HEIGHT - 1] = ["."] * 4 + ["#"] * (WIDTH - 4)
    flat = next(p for p in placements(board, "I") if p.key == "r0c0")
    assert flat.features.lines == 1
    assert all(ch == "." for row in flat.board for ch in row)


def test_choice_runs_a_knockout_within_the_label_alphabet() -> None:
    client = FakeUnridden()
    cands = placements(empty_board(), "T")
    pick, note, detail = choose_choice(client, empty_board(), "T", cands)

    assert note.startswith("knockout")
    assert [r["name"] for r in detail["rounds"]] == ["g0", "g1", "final"]
    for body in client.decision_bodies:
        for question in body["questions"].values():
            assert 2 <= len(question["criteria"]) <= 26
    # The fake picks each group's last option, then the second finalist.
    assert pick.key == cands[-1].key
    assert client.deleted == []  # cleanup waits until the move is made
    client.flush()
    assert len(client.deleted) == 2


def test_score_asks_one_question_per_placement_in_batches_of_32() -> None:
    client = FakeUnridden()
    cands = placements(empty_board(), "L")
    pick, _, detail = choose_score(client, empty_board(), "L", cands, hints=False)

    assert [len(b["questions"]) for b in client.decision_bodies] == [32, 2]
    assert set(detail["scores"]) == {p.key for p in cands}
    assert pick.key == cands[31].key  # highest index within the first batch


@pytest.mark.parametrize("agent", ["choice", "score"])
def test_trace_records_every_move_and_exchange(tmp_path: Path, agent: str) -> None:
    trace = tmp_path / "trace.jsonl"
    args = argparse.Namespace(
        agent=agent,
        url="http://fake",
        seed=3,
        max_pieces=5,
        hints=True,
        watch=False,
        delay=0.0,
        verbose=False,
        trace=str(trace),
        realtime=False,
        start_level=0,
        rules=False,
        prune=False,
        compact=False,
        nes_delays=False,
        pipeline=False,
        static_state=False,
        dense_board=False,
        movement="das",
        tap_hz=15.0,
    )
    result = run_game(args, FakeUnridden())

    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [r["type"] for r in rows] == ["game", *["move"] * 5, "result"]
    assert rows[-1]["pieces"] == result["pieces"] == 5
    for move in rows[1:-1]:
        paths = [e["path"] for e in move["exchanges"]]
        assert paths[0] == "/v2/snapshots"
        assert move["pick"] in {c["key"] for c in move["candidates"]}


def test_realtime_decision_in_time_lands_on_target() -> None:
    board = empty_board()
    target = placements(board, "T")[-1]  # far right, needs a rotation and shifts
    landed, info = realtime_landing(board, target, latency_s=0.3, level=0)
    assert landed is target
    assert info["outcome"] == "on time"


def test_realtime_decision_too_slow_locks_at_spawn() -> None:
    board = empty_board()
    target = placements(board, "T")[-1]
    landed, info = realtime_landing(board, target, latency_s=1.5, level=18)
    assert landed is not target
    assert info["outcome"] == "landed before the decision"
    assert {c for _, c in landed.cells} == {3, 4, 5}  # spawn columns


def test_strategy_rules_keep_the_well_for_tetrises() -> None:
    board = empty_board()
    for r in range(HEIGHT - 3, HEIGHT):  # three rows ready, well column empty
        board[r] = ["#"] * (WIDTH - 1) + ["."]
    kept = strategy_filter(board, placements(board, "I"))
    assert kept
    assert all(WELL not in {c for _, c in p.cells} for p in kept)
    assert all(p.features.new_holes == 0 for p in kept)


def test_prune_keeps_only_non_dominated_placements() -> None:
    board = empty_board()
    cands = placements(board, "T")
    kept = prune(cands)
    assert 1 <= len(kept) < len(cands)
    for p in cands:
        if p not in kept:
            assert any(dominates(q.features, p.features) for q in kept)


@pytest.mark.parametrize("pipeline", [False, True])
def test_pipelining_gives_the_next_piece_a_head_start(
    tmp_path: Path, pipeline: bool
) -> None:
    trace = tmp_path / "t.jsonl"
    args = argparse.Namespace(
        agent="choice",
        url="http://fake",
        seed=0,
        max_pieces=12,
        hints=False,
        watch=False,
        delay=0.0,
        verbose=False,
        trace=str(trace),
        realtime=True,
        start_level=0,
        rules=False,
        prune=False,
        compact=True,
        nes_delays=True,
        pipeline=pipeline,
        static_state=False,
        dense_board=False,
        movement="das",
        tap_hz=15.0,
    )
    run_game(args, FakeUnridden())
    moves = [json.loads(x) for x in trace.read_text().splitlines()][1:-1]
    head = [m["realtime"]["head_start_ms"] for m in moves]
    spawn = [m["realtime"]["spawn_s"] for m in moves]
    assert head[0] == 0
    assert all(h > 0 for h in head[1:]) if pipeline else all(h == 0 for h in head)
    # A piece is visible only once the one before it spawns, so the head start
    # never exceeds the previous piece's time on the board.
    for i in range(1, len(moves)):
        assert head[i] <= 1000 * (spawn[i] - spawn[i - 1]) + 0.1


def test_static_state_keeps_one_snapshot_for_the_game(tmp_path: Path) -> None:
    client = FakeUnridden()
    args = argparse.Namespace(
        agent="choice",
        url="http://fake",
        seed=0,
        max_pieces=8,
        hints=False,
        watch=False,
        delay=0.0,
        verbose=False,
        trace=None,
        realtime=False,
        start_level=0,
        rules=False,
        prune=False,
        compact=True,
        nes_delays=False,
        pipeline=False,
        static_state=True,
        dense_board=True,
        movement="das",
        tap_hz=15.0,
    )
    run_game(args, client)
    used = {body["snapshot"]["id"] for body in client.decision_bodies}
    assert len(client.decision_bodies) > 1
    assert len(used) == 1  # every move branched from the same kept snapshot
    assert client.kept == {}  # closed at the end of the game
    assert len(client.deleted) == 1
    for body in client.decision_bodies:
        for question in body["questions"].values():
            assert "Current board:" in question["instructions"]


def test_shift_schedules_match_nes_timing() -> None:
    assert shift_frames(3, "das") == [0, 16, 22]
    assert shift_frames(3, "charged", charge=16) == [0, 6, 12]
    assert shift_frames(3, "charged", charge=10) == [6, 12, 18]
    assert shift_frames(3, "tap", tap_hz=15.0) == [0, 4, 8]
    assert shift_frames(2, "tap", tap_hz=60.0) == [0, 2]  # press plus release


def test_tapping_reaches_a_target_holding_cannot_at_level_29() -> None:
    board = empty_board()
    target = next(p for p in placements(board, "I") if p.key == "r0c0")
    held, held_info = realtime_landing(board, target, 0.0, 29, "das")
    tapped, _ = realtime_landing(board, target, 0.0, 29, "tap", tap_hz=15.0)
    assert held is not target and held_info["outcome"] != "on time"
    assert tapped is target
