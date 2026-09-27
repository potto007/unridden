"""Play Tetris with Unridden making every placement decision.

Each turn the engine enumerates every legal hard-drop placement (rotation x
column) of the current piece, then asks Unridden to pick one. The board is
prefilled once as a /v2 context snapshot and the questions branch from it.

Agents:
  choice     one Choice question per group of <=26 placements; each option is
             described by its outcome (lines, holes, height, bumpiness). More
             than 26 placements run a two-round knockout.
  score      one Score question per placement with the resulting board drawn;
             the placement with the highest expected score wins.
  heuristic  fixed linear evaluator (Yiyuan Lee weights), no model.
  random     uniform over placements, no model.

Stdlib only; talks HTTP to the /v2 snapshots service (`serve --snapshots`,
:8091 by default). `--trace` writes every move and HTTP exchange as JSONL for
tetris_replay.html. Results: docs/results/tetris-demo.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WIDTH, HEIGHT = 10, 20

# Rotation 0 of each tetromino as (row, col) cells; other rotations are derived.
SHAPES: dict[str, list[tuple[int, int]]] = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}

Cells = tuple[tuple[int, int], ...]
Board = list[list[str]]  # "." empty, piece letter filled; row 0 is the top


def normalize(cells: list[tuple[int, int]]) -> Cells:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def rotations(piece: str) -> list[Cells]:
    out: list[Cells] = []
    cells = SHAPES[piece]
    for _ in range(4):
        norm = normalize(cells)
        if norm not in out:
            out.append(norm)
        cells = [(c, -r) for r, c in cells]  # 90 degrees clockwise
    return out


ROTATIONS = {p: rotations(p) for p in SHAPES}


def empty_board() -> Board:
    return [["."] * WIDTH for _ in range(HEIGHT)]


def fits(board: Board, cells: Cells, top: int, left: int) -> bool:
    for r, c in cells:
        rr, cc = top + r, left + c
        if rr >= HEIGHT or cc < 0 or cc >= WIDTH or board[rr][cc] != ".":
            return False
    return True


@dataclass(frozen=True)
class Features:
    lines: int
    holes: int
    new_holes: int
    agg_height: int
    max_height: int
    bumpiness: int


@dataclass
class Placement:
    key: str
    piece: str
    rotation: int
    left: int
    cells: Cells
    board: Board  # after placement and line clears
    features: Features


def column_heights(board: Board) -> list[int]:
    heights = []
    for c in range(WIDTH):
        h = 0
        for r in range(HEIGHT):
            if board[r][c] != ".":
                h = HEIGHT - r
                break
        heights.append(h)
    return heights


def count_holes(board: Board) -> int:
    holes = 0
    for c in range(WIDTH):
        seen = False
        for r in range(HEIGHT):
            if board[r][c] != ".":
                seen = True
            elif seen:
                holes += 1
    return holes


def features_of(before: Board, after: Board, lines: int) -> Features:
    heights = column_heights(after)
    holes = count_holes(after)
    return Features(
        lines=lines,
        holes=holes,
        new_holes=max(0, holes - count_holes(before)),
        agg_height=sum(heights),
        max_height=max(heights),
        bumpiness=sum(abs(a - b) for a, b in zip(heights, heights[1:], strict=False)),
    )


def drop(board: Board, piece: str, cells: Cells) -> tuple[Board, int]:
    """Place cells (already positioned) and clear full rows."""
    new = [row[:] for row in board]
    for r, c in cells:
        new[r][c] = piece
    kept = [row for row in new if "." in row]
    lines = HEIGHT - len(kept)
    return [["."] * WIDTH for _ in range(lines)] + kept, lines


def placements(board: Board, piece: str) -> list[Placement]:
    out: list[Placement] = []
    seen: set[str] = set()
    for rot, cells in enumerate(ROTATIONS[piece]):
        width = max(c for _, c in cells) + 1
        for left in range(WIDTH - width + 1):
            if not fits(board, cells, 0, left):
                continue
            top = 0
            while fits(board, cells, top + 1, left):
                top += 1
            placed = tuple((top + r, left + c) for r, c in cells)
            after, lines = drop(board, piece, placed)
            fingerprint = render_board(after)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            out.append(
                Placement(
                    key=f"r{rot}c{left}",
                    piece=piece,
                    rotation=rot,
                    left=left,
                    cells=placed,
                    board=after,
                    features=features_of(board, after, lines),
                )
            )
    return out


def render_board(board: Board, highlight: Cells = ()) -> str:
    """Rows from the highest filled row down, plus a floor; empty sky elided."""
    marks = set(highlight)
    first = next(
        (r for r in range(HEIGHT) if any(ch != "." for ch in board[r])), HEIGHT
    )
    lines = []
    if first > 0:
        lines.append(f"({first} empty rows above)")
    for r in range(first, HEIGHT):
        row = "".join(
            "@" if (r, c) in marks else ("#" if board[r][c] != "." else ".")
            for c in range(WIDTH)
        )
        lines.append(f"|{' '.join(row)}|")
    lines.append("+" + "-" * (2 * WIDTH - 1) + "+")
    lines.append(" " + " ".join(str(c) for c in range(WIDTH)))
    return "\n".join(lines)


def describe(p: Placement) -> str:
    f = p.features
    cols = sorted({c for _, c in p.cells})
    span = f"column {cols[0]}" if len(cols) == 1 else f"columns {cols[0]}-{cols[-1]}"
    return (
        f"rotation {p.rotation}, {span}: clears {f.lines} line(s), "
        f"creates {f.new_holes} new hole(s), stack height {f.max_height}, "
        f"bumpiness {f.bumpiness}"
    )


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

RULES = (
    "You are playing Tetris on a 10-wide, 20-tall well. '#' is a filled cell, "
    "'.' is empty. Pieces drop straight down. Full rows clear. The game ends "
    "when the stack reaches the top. Good play keeps the stack low and flat, "
    "avoids covered holes, and clears lines."
)


class Unridden:
    def __init__(self, base: str, timeout: float = 60.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.calls = 0
        self.tokens = 0
        self.seconds = 0.0
        self.exchanges: list[dict[str, Any]] = []  # raw HTTP log for the trace

    def _req(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"content-type": "application/json"},
        )
        for attempt in range(20):
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    out = json.loads(raw) if raw else None
                    self.exchanges.append(
                        {
                            "method": method,
                            "path": path,
                            "status": resp.status,
                            "ms": round(1000 * (time.monotonic() - start), 1),
                            "request": body,
                            "response": out,
                        }
                    )
                    return out
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")
                if err.code == 429 and attempt < 19:
                    time.sleep(0.25)
                    continue
                raise RuntimeError(f"{method} {path} -> {err.code}: {detail}") from None
        raise RuntimeError("unreachable")

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        snap = self._req(
            "POST",
            "/v2/snapshots",
            {
                "input": {"kind": "context", "state": state},
                "checkpoints": [30],
                "persistence": "memory",
                "ttl_seconds": 300,
            },
        )
        snap_id = snap["snapshots"][0]["id"]
        answers: dict[str, Any] = {}
        try:
            items = list(questions.items())
            for i in range(0, len(items), 32):
                resp = self._req(
                    "POST",
                    "/v2/decisions",
                    {
                        "snapshot": {"id": snap_id, "relationship": "followup"},
                        "questions": dict(items[i : i + 32]),
                        "readout": {"completed_blocks": 30},
                    },
                )
                answers.update(resp["answers"])
                self.tokens += resp["usage"]["input_tokens"]
                self.calls += 1
        finally:
            with suppress(RuntimeError):
                self._req("DELETE", f"/v2/snapshots/{snap_id}")
        self.seconds += time.monotonic() - start
        return answers


def state_text(board: Board, piece: str) -> str:
    board_text = render_board(board)
    return f"{RULES}\n\nCurrent board:\n{board_text}\n\nPiece to place: {piece}"


def choose_choice(
    client: Unridden, board: Board, piece: str, cands: list[Placement]
) -> tuple[Placement, str, dict[str, Any]]:
    state = state_text(board, piece)
    by_key = {p.key: p for p in cands}
    instructions = (
        f"Pick where to drop the {piece} piece. Prefer placements that clear "
        "lines, create no new holes, and keep the stack low and flat."
    )

    def question(group: list[Placement]) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {p.key: describe(p) for p in group},
        }

    groups = [cands[i : i + 26] for i in range(0, len(cands), 26)]
    if len(groups) > 1:  # knockout: balance groups, then a final
        half = (len(cands) + 1) // 2
        groups = [cands[:half], cands[half:]]
    answers = client.decide(state, {f"g{i}": question(g) for i, g in enumerate(groups)})
    winners = [by_key[answers[f"g{i}"]["choice"]] for i in range(len(groups))]
    conf = [answers[f"g{i}"]["confidence"] for i in range(len(groups))]
    rounds = [
        {
            "name": f"g{i}",
            "options": [p.key for p in g],
            "probabilities": answers[f"g{i}"]["probabilities"],
            "choice": answers[f"g{i}"]["choice"],
        }
        for i, g in enumerate(groups)
    ]
    if len(winners) == 1:
        return winners[0], f"conf {conf[0]:.3f}", {"rounds": rounds}
    final = client.decide(state, {"final": question(winners)})["final"]
    rounds.append(
        {
            "name": "final",
            "options": [p.key for p in winners],
            "probabilities": final["probabilities"],
            "choice": final["choice"],
        }
    )
    return (
        by_key[final["choice"]],
        f"knockout, conf {final['confidence']:.3f}",
        {"rounds": rounds},
    )


SCORE_LEVELS = ["terrible", "bad", "mediocre", "good", "excellent"]


def choose_score(
    client: Unridden, board: Board, piece: str, cands: list[Placement], hints: bool
) -> tuple[Placement, str, dict[str, Any]]:
    state = state_text(board, piece)
    questions = {}
    for p in cands:
        text = (
            f"The {piece} piece is dropped as a candidate move. Its cells are "
            f"marked '@' in the board below (before any rows clear). Rate how "
            f"good this move is for the rest of the game.\n\n"
            f"{render_board(drop_only(board, p), highlight=p.cells)}"
        )
        if hints:
            text += f"\n\nOutcome: {describe(p)}"
        questions[p.key] = {
            "type": "score",
            "instructions": text,
            "criteria": SCORE_LEVELS,
        }
    answers = client.decide(state, questions)
    best = max(cands, key=lambda p: answers[p.key]["score"])
    detail = {
        "scores": {p.key: answers[p.key]["score"] for p in cands},
        "probabilities": {p.key: answers[p.key]["probabilities"] for p in cands},
    }
    return best, f"score {answers[best.key]['score']:.3f}", detail


def drop_only(board: Board, p: Placement) -> Board:
    new = [row[:] for row in board]
    for r, c in p.cells:
        new[r][c] = p.piece
    return new


def heuristic_value(p: Placement) -> float:
    f = p.features
    return (
        -0.510066 * f.agg_height
        + 0.760666 * f.lines
        - 0.35663 * f.holes
        - 0.184483 * f.bumpiness
    )


# --------------------------------------------------------------------------
# Game loop
# --------------------------------------------------------------------------


def bag_stream(seed: int) -> Iterator[str]:
    """Seeded 7-bag randomizer, so every agent sees the same piece sequence."""
    rng = random.Random(seed)
    while True:
        bag = list(SHAPES)
        rng.shuffle(bag)
        yield from bag


def choose(
    args: argparse.Namespace,
    client: Unridden | None,
    rng: random.Random,
    board: Board,
    piece: str,
    cands: list[Placement],
) -> tuple[Placement, str, dict[str, Any]]:
    if len(cands) == 1:  # forced; a Choice needs at least two options
        return cands[0], "forced", {}
    if args.agent == "random":
        return rng.choice(cands), "", {}
    if args.agent == "heuristic" or client is None:
        return max(cands, key=heuristic_value), "", {}
    if args.agent == "choice":
        return choose_choice(client, board, piece, cands)
    return choose_score(client, board, piece, cands, args.hints)


def move_record(
    number: int,
    piece: str,
    before: Board,
    cands: list[Placement],
    choice: tuple[Placement, str, dict[str, Any]],
    best: Placement,
    lines_total: int,
    ms: float,
    exchanges: list[dict[str, Any]],
) -> dict[str, Any]:
    pick, note, detail = choice
    return {
        "type": "move",
        "move": number,
        "piece": piece,
        "board_before": ["".join(r) for r in before],
        "candidates": [
            {
                "key": p.key,
                "rotation": p.rotation,
                "left": p.left,
                "cells": p.cells,
                "description": describe(p),
                "features": asdict(p.features),
                "heuristic": round(heuristic_value(p), 4),
            }
            for p in cands
        ],
        "pick": pick.key,
        "heuristic_pick": best.key,
        "note": note,
        "detail": detail,
        "lines_cleared": pick.features.lines,
        "lines_total": lines_total,
        "board_after": ["".join(r) for r in pick.board],
        "ms": round(ms, 1),
        "exchanges": exchanges,
    }


def show(
    args: argparse.Namespace,
    n: int,
    piece: str,
    pick: Placement,
    options: int,
    note: str,
    lines: int,
) -> None:
    f = pick.features
    if args.watch:
        sys.stdout.write("\x1b[2J\x1b[H")
        print(
            f"{args.agent}  piece #{n} {piece} -> {pick.key} ({options} options) {note}"
        )
        print(f"lines {lines}  holes {f.holes}  height {f.max_height}\n")
        print(render_board(pick.board), flush=True)
        if args.delay:
            time.sleep(args.delay)
    elif args.verbose:
        print(
            f"#{n:4d} {piece} -> {pick.key:6s} lines={lines:3d} holes={f.holes:2d} "
            f"h={f.max_height:2d} {note}",
            flush=True,
        )


def play(args: argparse.Namespace) -> dict[str, Any]:
    client = Unridden(args.url) if args.agent in ("choice", "score") else None
    return run_game(args, client)


def run_game(args: argparse.Namespace, client: Unridden | None) -> dict[str, Any]:
    rng = random.Random(args.seed + 1)
    pieces = bag_stream(args.seed)
    board = empty_board()
    lines = placed = agree = 0
    t0 = time.monotonic()
    agent = args.agent + ("+hints" if args.agent == "score" and args.hints else "")
    with ExitStack() as stack:
        trace = None
        if args.trace:
            Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
            trace = stack.enter_context(open(args.trace, "w"))
        if trace:
            header = {
                "type": "game",
                "agent": args.agent,
                "hints": args.hints,
                "seed": args.seed,
                "max_pieces": args.max_pieces,
                "width": WIDTH,
                "height": HEIGHT,
                "url": args.url if client else None,
                "rules": RULES,
                "started": time.time(),
            }
            trace.write(json.dumps(header) + "\n")
        while placed < args.max_pieces:
            piece = next(pieces)
            cands = placements(board, piece)
            if not cands:
                break
            if client:
                client.exchanges.clear()
            move_t0 = time.monotonic()
            choice = choose(args, client, rng, board, piece, cands)
            pick, note, _ = choice
            best = max(cands, key=heuristic_value)
            agree += pick is best
            lines += pick.features.lines
            placed += 1
            if trace:
                record = move_record(
                    placed,
                    piece,
                    board,
                    cands,
                    choice,
                    best,
                    lines,
                    1000 * (time.monotonic() - move_t0),
                    list(client.exchanges) if client else [],
                )
                trace.write(json.dumps(record) + "\n")
                trace.flush()
            board = pick.board
            show(args, placed, piece, pick, len(cands), note, lines)
        result: dict[str, Any] = {
            "agent": agent,
            "seed": args.seed,
            "pieces": placed,
            "lines": lines,
            "topped_out": placed < args.max_pieces,
            "heuristic_agreement": round(agree / max(1, placed), 3),
            "wall_s": round(time.monotonic() - t0, 1),
        }
        if client:
            result.update(
                requests=client.calls,
                input_tokens=client.tokens,
                model_s=round(client.seconds, 1),
            )
        if trace:
            trace.write(json.dumps({"type": "result", **result}) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--agent", choices=["choice", "score", "heuristic", "random"], default="choice"
    )
    ap.add_argument("--url", default="http://127.0.0.1:8091", help="/v2 service")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pieces", type=int, default=200)
    ap.add_argument(
        "--hints",
        action="store_true",
        help="score agent: append outcome stats to each drawn board",
    )
    ap.add_argument("--watch", action="store_true", help="redraw the board each move")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between moves")
    ap.add_argument("--verbose", action="store_true", help="one line per move")
    ap.add_argument(
        "--trace", help="write every move and HTTP exchange as JSONL for the replay"
    )
    print(json.dumps(play(ap.parse_args())))


if __name__ == "__main__":
    main()
