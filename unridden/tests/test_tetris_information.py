"""One-preview information flow, using a deterministic controller clock."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.unridden import play_tetris as demo
from unridden.tests.test_play_tetris import FakeUnridden


def game_args(trace: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = dict(
        agent="choice",
        url="http://fake",
        seed=0,
        max_pieces=6,
        hints=False,
        watch=False,
        delay=0.0,
        verbose=False,
        trace=str(trace),
        realtime=True,
        start_level=0,
        rules=False,
        prune=True,
        compact=True,
        nes_delays=True,
        pipeline=True,
        static_state=True,
        dense_board=True,
        movement="tap",
        tap_hz=30.0,
        lookahead=False,
    )
    return argparse.Namespace(**(values | overrides))


def controlled_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [0.0]
    original = demo.choose

    def choose(*args: Any, **kwargs: Any) -> Any:
        answer = original(*args, **kwargs)
        clock[0] += 0.05
        return answer

    monkeypatch.setattr(demo, "choose", choose)
    monkeypatch.setattr(
        demo, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 0.0)
    )
    return clock


@pytest.mark.parametrize(
    "agent,lookahead,needs_preview",
    [
        ("choice", False, False),
        ("choice", True, True),
        ("pruned", False, False),
        ("pruned", True, True),
        ("strategy", False, True),
        ("score", False, False),
        ("heuristic", False, False),
        ("rules", False, False),
        ("random", False, False),
    ],
)
def test_every_policy_gets_only_information_available_at_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    lookahead: bool,
    needs_preview: bool,
) -> None:
    controlled_clock(monkeypatch)
    trace = tmp_path / "game.jsonl"
    seen: list[demo.Status] = []
    choose = demo.choose

    def observe(*args: Any, **kwargs: Any) -> Any:
        seen.append(args[-1])
        return choose(*args, **kwargs)

    monkeypatch.setattr(demo, "choose", observe)
    args = game_args(trace, agent=agent, lookahead=lookahead)
    demo.run_game(args, FakeUnridden())
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows[0]["information_rules"] == demo.INFORMATION_RULES
    moves = [row for row in rows if row["type"] == "move"]
    assert len(moves) == len(seen) == 6
    for move, status in zip(moves, seen, strict=True):
        timing = move["realtime"]
        start = timing["decision_start_s"]
        assert start >= timing["board_available_s"]
        assert start >= timing["piece_available_s"]
        visible = start >= timing["next_piece_available_s"]
        assert timing["next_piece_visible"] == visible
        assert (status.next_piece is not None) == visible
        assert move["next_piece"] == status.next_piece
        if needs_preview:
            assert visible
            assert timing["head_start_ms"] <= 0
    if not needs_preview:
        # Legal work still overlaps the entry delay, with no second future piece.
        assert any(m["realtime"]["head_start_ms"] > 0 for m in moves[1:])
        assert any(s.next_piece is None for s in seen[1:])


def test_unrevealed_piece_cannot_change_the_request_or_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controlled_clock(monkeypatch)
    records = []
    for hidden in ("J", "L"):
        monkeypatch.setattr(
            demo, "bag_stream", lambda seed, hidden=hidden: iter(("T", "I", hidden))
        )
        trace = tmp_path / f"{hidden}.jsonl"
        demo.run_game(game_args(trace, max_pieces=2), FakeUnridden())
        moves = [
            row
            for line in trace.read_text().splitlines()
            if (row := json.loads(line))["type"] == "move"
        ]
        assert moves[1]["next_piece"] is None
        assert moves[1]["realtime"]["head_start_ms"] > 0
        records.append(moves[1])
    assert records[0]["exchanges"] == records[1]["exchanges"]
    assert records[0]["pick"] == records[1]["pick"]
    assert records[0]["board_after"] == records[1]["board_after"]


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("needs_preview", [False, True])
def test_scheduling_waits_for_an_occupied_agent(
    pipeline: bool, needs_preview: bool
) -> None:
    assert (
        demo.decision_start(
            spawn_at=2.0,
            previewed_at=1.0,
            board_ready_at=1.8,
            agent_free_at=2.4,
            pipeline=pipeline,
            needs_preview=needs_preview,
        )
        == 2.4
    )


def test_no_entry_delay_means_no_confirmed_board_head_start() -> None:
    assert (
        demo.decision_start(
            spawn_at=2.0,
            previewed_at=1.0,
            board_ready_at=2.0,
            agent_free_at=1.1,
            pipeline=True,
            needs_preview=False,
        )
        == 2.0
    )


def test_native_cleanup_occupies_the_agent_before_the_next_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = controlled_clock(monkeypatch)
    client = FakeUnridden()

    def cleanup() -> None:
        clock[0] += 0.3

    monkeypatch.setattr(client, "flush", cleanup)
    trace = tmp_path / "cleanup.jsonl"
    demo.run_game(game_args(trace, max_pieces=3), client)
    moves = [
        row
        for line in trace.read_text().splitlines()
        if (row := json.loads(line))["type"] == "move"
    ]
    for previous, current in zip(moves, moves[1:], strict=False):
        assert previous["realtime"]["cleanup_ms"] == 300.0
        assert current["realtime"]["decision_start_s"] >= (
            previous["realtime"]["decision_ready_s"] + 0.3 - 0.000002
        )


def test_initial_candidate_work_counts_toward_the_decision_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = controlled_clock(monkeypatch)
    placements = demo.placements

    def enumerate_candidates(*args: Any) -> Any:
        clock[0] += 0.02
        return placements(*args)

    monkeypatch.setattr(demo, "placements", enumerate_candidates)
    trace = tmp_path / "candidate-work.jsonl"
    demo.run_game(game_args(trace, max_pieces=1), FakeUnridden())
    move = json.loads(trace.read_text().splitlines()[1])
    assert move["realtime"]["compute_ms"] == 70.0


def test_preview_dependent_policies_refuse_an_unseen_piece(tmp_path: Path) -> None:
    status = demo.Status(0, 0, 0, None, 0)
    with pytest.raises(ValueError, match="visible next-piece preview"):
        demo.strategy_state(demo.empty_board(), "T", status)
    with pytest.raises(ValueError, match="visible next-piece preview"):
        demo.choose_plan(
            game_args(tmp_path / "unused", lookahead=True),
            FakeUnridden(),
            random.Random(0),
            demo.empty_board(),
            "T",
            status,
        )


@pytest.mark.parametrize("pipeline", [False, True])
def test_turn_based_play_has_the_current_spawn_preview(
    tmp_path: Path, pipeline: bool
) -> None:
    trace = tmp_path / "turn-based.jsonl"
    demo.run_game(
        game_args(trace, realtime=False, pipeline=pipeline, lookahead=True),
        FakeUnridden(),
    )
    moves = [
        row
        for line in trace.read_text().splitlines()
        if (row := json.loads(line))["type"] == "move"
    ]
    assert all(m["next_piece"] in demo.SHAPES for m in moves)
    assert all(m["realtime"] is None for m in moves)
