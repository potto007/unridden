"""Play Tetris with Unridden making every placement decision.

Each turn the engine enumerates every legal hard-drop placement (rotation x
column) of the current piece, then asks Unridden to pick one. The board is
prefilled once as a /v2 context snapshot and the questions branch from it.

Agents:
  choice     one Choice question per group of <=26 placements; each option is
             described by its outcome (lines, holes, height, bumpiness). More
             than 26 placements run as groups plus a final.
  strategy   like choice, with NES scoring and a high-score strategy (Tetris
             well, flat stack, burn only when high) in the state; --rules
             also enforces that strategy's hard rules before asking.
  score      one Score question per placement with the resulting board drawn;
             the placement with the highest expected score wins.
  heuristic  fixed linear evaluator (Yiyuan Lee weights), no model.
  rules      the strategy's hard rules, then a uniform pick; no model.
  random     uniform over placements, no model.

--realtime makes the piece fall at NES gravity (--start-level) while the agent
decides; a late decision locks the piece wherever it has fallen.

Stdlib only; talks HTTP to the /v2 snapshots service (`serve --snapshots`,
:8091 by default). `--trace` writes every move and HTTP exchange as JSONL for
tetris_replay.html. Results: docs/results/tetris-demo.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
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


WELL = WIDTH - 1  # the strategy agent keeps the rightmost column open

# NES scoring: base points per clear, multiplied by (level + 1); the level
# rises every 10 lines from level 0.
NES_POINTS = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
CLEAR_NAMES = {1: "single", 2: "double", 3: "triple", 4: "TETRIS"}


def nes_points(lines: int, level: int) -> int:
    return NES_POINTS[lines] * (level + 1)


@dataclass(frozen=True)
class Features:
    lines: int
    holes: int
    new_holes: int
    agg_height: int
    max_height: int
    bumpiness: int
    well_height: int  # height of the well column after the move
    well_height_before: int
    ready_rows: int  # rows open to the well with columns 0-8 full
    ready_rows_before: int
    spread: int  # highest minus lowest of columns 0-8


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


def ready_rows(board: Board) -> int:
    """Rows an I piece dropped in the well would reach that need only the well."""
    top = HEIGHT - column_heights(board)[WELL]
    return sum(1 for r in range(top) if all(board[r][c] != "." for c in range(WELL)))


def features_of(before: Board, after: Board, lines: int) -> Features:
    heights = column_heights(after)
    holes = count_holes(after)
    stack = heights[:WELL]
    return Features(
        lines=lines,
        holes=holes,
        new_holes=max(0, holes - count_holes(before)),
        agg_height=sum(heights),
        max_height=max(heights),
        bumpiness=sum(abs(a - b) for a, b in zip(heights, heights[1:], strict=False)),
        well_height=heights[WELL],
        well_height_before=column_heights(before)[WELL],
        ready_rows=ready_rows(after),
        ready_rows_before=ready_rows(before),
        spread=max(stack) - min(stack),
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


def dominates(a: Features, b: Features) -> bool:
    """a is at least as good as b on every measure and better on one."""
    va = (-a.lines, a.holes, a.agg_height, a.bumpiness)
    vb = (-b.lines, b.holes, b.agg_height, b.bumpiness)
    return va != vb and all(x <= y for x, y in zip(va, vb, strict=True))


def prune(cands: list[Placement]) -> list[Placement]:
    """Drop placements another placement beats on every measure; no weights."""
    return [
        p for p in cands if not any(dominates(q.features, p.features) for q in cands)
    ]


def describe_compact(p: Placement) -> str:
    f = p.features
    cols = sorted({c for _, c in p.cells})
    span = f"col {cols[0]}" if len(cols) == 1 else f"cols {cols[0]}-{cols[-1]}"
    return (
        f"{span}: clears {f.lines}, {f.new_holes} new holes, height {f.max_height}, "
        f"bumpiness {f.bumpiness}"
    )


def describe(p: Placement) -> str:
    f = p.features
    cols = sorted({c for _, c in p.cells})
    span = f"column {cols[0]}" if len(cols) == 1 else f"columns {cols[0]}-{cols[-1]}"
    return (
        f"rotation {p.rotation}, {span}: clears {f.lines} line(s), "
        f"creates {f.new_holes} new hole(s), stack height {f.max_height}, "
        f"bumpiness {f.bumpiness}"
    )


@dataclass(frozen=True)
class Status:
    """What a player sees beside the board: score, level and the preview."""

    score: int
    level: int
    lines: int
    next_piece: str
    since_i: int  # pieces placed since the last I


DANGER_HEIGHT = 12


def describe_strategic(p: Placement, status: Status) -> str:
    """The outcome of a placement in the terms the strategy is written in."""
    f = p.features
    cols = sorted({c for _, c in p.cells})
    span = f"column {cols[0]}" if len(cols) == 1 else f"columns {cols[0]}-{cols[-1]}"
    if f.lines:
        clear = f"{CLEAR_NAMES[f.lines]} (+{nes_points(f.lines, status.level)} points)"
        burned = f.ready_rows_before - f.ready_rows
        if f.lines < 4 and burned > 0:
            clear += f", burns {burned} Tetris-ready rows"
    else:
        clear = "no clear"
    if f.well_height == 0:
        well = "well open"
    elif f.well_height_before == 0:
        well = "BLOCKS the well"
    else:
        well = f"well still blocked ({f.well_height} high)"
    return (
        f"{span}: {clear}; {f.new_holes} new hole(s); {well}; "
        f"{f.ready_rows} Tetris-ready row(s); spread {f.spread}; "
        f"stack height {f.max_height}"
    )


def strategy_filter(board: Board, cands: list[Placement]) -> list[Placement]:
    """Apply the strategy's hard rules; a rule that would empty the set is skipped.

    Stack in danger (above DANGER_HEIGHT): keep the moves that leave the lowest
    stack, then the fewest new holes. Otherwise: fewest new holes, never block
    an open well, and an I goes into the well only for a Tetris.
    """

    def keep(
        rule: Callable[[Placement], bool], pool: list[Placement]
    ) -> list[Placement]:
        return [p for p in pool if rule(p)] or pool

    if max(column_heights(board)) > DANGER_HEIGHT:
        lowest = min(p.features.max_height for p in cands)
        kept = [p for p in cands if p.features.max_height == lowest]
        fewest = min(p.features.new_holes for p in kept)
        return [p for p in kept if p.features.new_holes == fewest]
    fewest = min(p.features.new_holes for p in cands)
    kept = [p for p in cands if p.features.new_holes == fewest]
    kept = keep(
        lambda p: not (p.features.well_height_before == 0 and p.features.well_height),
        kept,
    )
    return keep(
        lambda p: p.features.lines == 4 or WELL not in {c for _, c in p.cells},
        kept,
    )


# --------------------------------------------------------------------------
# Real time
# --------------------------------------------------------------------------

FPS = 60.0988  # NES NTSC frame rate
# NES gravity in frames per row for levels 0-28; level 29 and up is 1.
GRAVITY = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3]
GRAVITY += [2] * 10
DAS_DELAY, DAS_REPEAT = 16, 6  # NES delayed auto shift, in frames


LINE_CLEAR_FRAMES = 18  # NES line clear animation, about 17-20 frames


def entry_delay(lock_height: int) -> int:
    """NES entry delay: 10 frames low in the well, 2 more per 4 rows higher."""
    return 10 + 2 * ((lock_height + 1) // 4)


def frames_per_row(level: int) -> int:
    return GRAVITY[level] if level < len(GRAVITY) else 1


def spawn_left(piece: str) -> int:
    return 4 if piece == "O" else 3


def spawn_fits(board: Board, piece: str) -> bool:
    return fits(board, ROTATIONS[piece][0], 0, spawn_left(piece))


def realtime_landing(
    board: Board, target: Placement, latency_s: float, level: int
) -> tuple[Placement, dict[str, Any]]:
    """Where the piece really lands if the decision arrives after latency_s.

    The piece spawns in rotation 0 and falls at the level's NES gravity while
    the agent thinks. When the decision arrives it rotates once, shifts one
    column at a time on the NES auto-shift schedule while still falling, and
    hard-drops when it reaches the target column. A blocked rotation or shift,
    or landing first, locks the piece wherever it falls.
    """
    fpr = frames_per_row(level)
    piece = target.piece
    shape = ROTATIONS[piece][0]
    top, left = 0, spawn_left(piece)
    start = math.ceil(latency_s * FPS)
    step = 1 if target.left > left else -1
    shifts = abs(target.left - left)
    shift_at = {
        start + (0 if k == 0 else DAS_DELAY + DAS_REPEAT * (k - 1))
        for k in range(shifts)
    }
    done_at = max(shift_at, default=start)
    outcome = "on time"
    fallen = 0
    planning = True
    frame = 0
    while True:
        if frame == start:
            fallen = top
            if target.rotation:
                rotated = ROTATIONS[piece][target.rotation]
                if fits(board, rotated, top, left):
                    shape = rotated
                else:
                    planning, outcome = False, "rotation blocked"
        if planning and frame >= start and frame in shift_at:
            if fits(board, shape, top, left + step):
                left += step
            else:
                planning, outcome = False, "shift blocked"
        if planning and frame >= done_at:
            while fits(board, shape, top + 1, left):
                top += 1
            break
        if frame > 0 and frame % fpr == 0:
            if fits(board, shape, top + 1, left):
                top += 1
            else:
                if planning:
                    outcome = (
                        "landed before the decision"
                        if frame < start
                        else "landed before reaching the target"
                    )
                    fallen = top if frame < start else fallen
                break
        frame += 1
    cells = tuple(sorted((top + r, left + c) for r, c in shape))
    info = {
        "latency_ms": round(1000 * latency_s, 1),
        "frames_per_row": fpr,
        "rows_fallen_at_decision": fallen,
        "outcome": outcome,
        "lock_frame": frame,
        "lock_height": HEIGHT - max(r for r, _ in cells),
    }
    if cells == tuple(sorted(target.cells)):
        return target, info
    after, lines = drop(board, piece, cells)
    landed = Placement(
        key="missed",
        piece=piece,
        rotation=-1,
        left=left,
        cells=cells,
        board=after,
        features=features_of(board, after, lines),
    )
    return landed, info


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

RULES = (
    "You are playing Tetris on a 10-wide, 20-tall well. '#' is a filled cell, "
    "'.' is empty. Pieces drop straight down. Full rows clear. The game ends "
    "when the stack reaches the top. Good play keeps the stack low and flat, "
    "avoids covered holes, and clears lines."
)


class BudgetError(RuntimeError):
    """The compiled request does not fit the model's context."""


class Unridden:
    def __init__(self, base: str, timeout: float = 60.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.calls = 0
        self.tokens = 0
        self.seconds = 0.0
        self.exchanges: list[dict[str, Any]] = []  # raw HTTP log for the trace
        self.pending: list[str] = []  # snapshot ids awaiting deletion

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
                message = f"{method} {path} -> {err.code}: {detail}"
                if err.code == 422 and '"budget_error"' in detail:
                    raise BudgetError(message) from None
                raise RuntimeError(message) from None
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
        # Deleted by flush() once the move is made, so cleanup is off the clock.
        self.pending.append(snap_id)
        answers: dict[str, Any] = {}
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
        self.seconds += time.monotonic() - start
        return answers

    def flush(self) -> None:
        """Delete the snapshots made for the last move."""
        for snap_id in self.pending:
            with suppress(RuntimeError):
                self._req("DELETE", f"/v2/snapshots/{snap_id}")
        self.pending.clear()


def state_text(board: Board, piece: str) -> str:
    board_text = render_board(board)
    return f"{RULES}\n\nCurrent board:\n{board_text}\n\nPiece to place: {piece}"


STRATEGY = """GOAL: finish with the highest possible score, not just survive.

SCORING (NES): single 40, double 100, triple 300, Tetris (4 lines at once) 1200,
each multiplied by (level + 1). The level rises every 10 lines. One Tetris is
worth 30 singles, so a clear of fewer than 4 lines wastes rows that could have
become part of a Tetris.

STRATEGY used by high-scoring players:
1. Keep column 9 (the rightmost) empty as the well. Only an I piece goes in the
   well, and only when it clears a Tetris.
2. Stack columns 0-8 flat with no holes: keep the spread between their highest
   and lowest column at 2 or less and never cover an empty cell.
3. Build up to 4 or more Tetris-ready rows, then drop an I into the well.
4. Burn (take a single, double or triple) only when the stack is dangerously
   high (above 12 of 20 rows), when the well is blocked and must be dug out,
   or when the clear removes holes. Survival beats score.
5. Use the next piece: if it is an I and rows are Tetris-ready, keep the well
   open for it. After a long wait for an I, keep the stack low."""


def strategy_state(board: Board, piece: str, status: Status) -> str:
    return (
        f"{RULES}\n\n{STRATEGY}\n\n"
        f"Score {status.score}, level {status.level}, lines {status.lines}. "
        f"Pieces since the last I: {status.since_i}.\n\n"
        f"Current board:\n{render_board(board)}\n\n"
        f"Piece to place: {piece}. Next piece: {status.next_piece}."
    )


def choose_choice(
    client: Unridden,
    board: Board,
    piece: str,
    cands: list[Placement],
    compact: bool = False,
) -> tuple[Placement, str, dict[str, Any]]:
    instructions = (
        f"Pick where to drop the {piece} piece. Prefer placements that clear "
        "lines, create no new holes, and keep the stack low and flat."
    )
    describe_fn = describe_compact if compact else describe
    return knockout(client, state_text(board, piece), instructions, cands, describe_fn)


def choose_strategy(
    client: Unridden,
    board: Board,
    piece: str,
    cands: list[Placement],
    status: Status,
    rules: bool = False,
) -> tuple[Placement, str, dict[str, Any]]:
    instructions = (
        f"Pick where to drop the {piece} piece to maximize the final score, "
        "following the strategy in the state."
    )
    kept = strategy_filter(board, cands) if rules else cands
    if len(kept) == 1:
        return kept[0], "rules left one", {"kept": [kept[0].key]}
    pick, note, detail = knockout(
        client,
        strategy_state(board, piece, status),
        instructions,
        kept,
        lambda p: describe_strategic(p, status),
    )
    detail["kept"] = [p.key for p in kept]
    return pick, f"{len(kept)}/{len(cands)} kept, {note}", detail


def knockout(
    client: Unridden,
    state: str,
    instructions: str,
    cands: list[Placement],
    describe_fn: Callable[[Placement], str],
) -> tuple[Placement, str, dict[str, Any]]:
    """Choice over the placements in balanced groups, then a final of winners.

    Groups hold at most 26 options (the label alphabet). A group that does not
    fit the model's context is split further and the move is asked again.
    """
    by_key = {p.key: p for p in cands}

    def question(group: list[Placement]) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {p.key: describe_fn(p) for p in group},
        }

    size = 26
    while True:
        count = -(-len(cands) // size)
        groups = [cands[i::count] for i in range(count)] if count > 1 else [cands]
        try:
            answers = client.decide(
                state, {f"g{i}": question(g) for i, g in enumerate(groups)}
            )
            break
        except BudgetError:
            if size <= 4:
                raise
            size //= 2
    rounds = [
        {
            "name": f"g{i}",
            "options": [p.key for p in g],
            "probabilities": answers[f"g{i}"]["probabilities"],
            "choice": answers[f"g{i}"]["choice"],
        }
        for i, g in enumerate(groups)
    ]
    winners = [by_key[r["choice"]] for r in rounds]
    if len(winners) == 1:
        conf = answers["g0"]["confidence"]
        return winners[0], f"conf {conf:.3f}", {"rounds": rounds}
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
        f"knockout {len(groups)}, conf {final['confidence']:.3f}",
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
    status: Status,
) -> tuple[Placement, str, dict[str, Any]]:
    if args.prune and args.agent not in ("heuristic", "random", "rules"):
        kept = prune(cands)
        if len(kept) == 1:
            return kept[0], "pruned to one", {}
        cands = kept
    if len(cands) == 1:  # forced; a Choice needs at least two options
        return cands[0], "forced", {}
    if args.agent == "random":
        return rng.choice(cands), "", {}
    if args.agent == "rules":  # the strategy's hard rules, then a coin flip
        kept = strategy_filter(board, cands)
        return rng.choice(kept), f"{len(kept)}/{len(cands)} kept", {}
    if args.agent == "heuristic" or client is None:
        return max(cands, key=heuristic_value), "", {}
    if args.agent == "choice":
        return choose_choice(client, board, piece, cands, args.compact)
    if args.agent == "strategy":
        return choose_strategy(client, board, piece, cands, status, args.rules)
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
    status: Status,
    score_total: int,
    landed: Placement,
    timing: dict[str, Any] | None,
) -> dict[str, Any]:
    pick, note, detail = choice
    return {
        "landed": landed.key,
        "landed_cells": landed.cells,
        "landed_features": asdict(landed.features),
        "realtime": timing,
        "next_piece": status.next_piece,
        "level": status.level,
        "points": score_total - status.score,
        "score_total": score_total,
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
        "lines_cleared": landed.features.lines,
        "lines_total": lines_total,
        "board_after": ["".join(r) for r in landed.board],
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
    model_agents = ("choice", "strategy", "score")
    client = Unridden(args.url) if args.agent in model_agents else None
    return run_game(args, client)


def run_game(args: argparse.Namespace, client: Unridden | None) -> dict[str, Any]:
    rng = random.Random(args.seed + 1)
    pieces = bag_stream(args.seed)
    upcoming = next(pieces)
    board = empty_board()
    lines = placed = agree = score = tetrises = since_i = missed = 0
    # Simulated game clock in seconds (real-time mode). The agent may start on
    # a piece once the board it lands on is known and the agent is free.
    spawn_at = agent_free_at = previewed_at = 0.0
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
                "realtime": args.realtime,
                "start_level": args.start_level,
                "started": time.time(),
            }
            trace.write(json.dumps(header) + "\n")
        while placed < args.max_pieces:
            piece, upcoming = upcoming, next(pieces)
            level = max(args.start_level, lines // 10)
            status = Status(score, level, lines, upcoming, since_i)
            cands = placements(board, piece)
            if not cands or (args.realtime and not spawn_fits(board, piece)):
                break
            if client:
                client.exchanges.clear()
            move_t0 = time.monotonic()
            choice = choose(args, client, rng, board, piece, cands, status)
            latency = time.monotonic() - move_t0
            if client:
                client.flush()
            pick, note, _ = choice
            landed, timing = pick, None
            if args.realtime:
                # Pipelined, the agent starts once it is free and the piece is in
                # the one-piece preview, which happens when the previous piece
                # spawns. Otherwise it starts when the piece itself spawns.
                visible = previewed_at if args.pipeline else spawn_at
                start = max(visible, agent_free_at)
                decided = start + latency
                late_by = max(0.0, decided - spawn_at)
                landed, timing = realtime_landing(board, pick, late_by, level)
                missed += landed is not pick
                delay = 0
                if args.nes_delays:
                    delay = entry_delay(timing["lock_height"])
                    delay += LINE_CLEAR_FRAMES if landed.features.lines else 0
                timing.update(
                    compute_ms=round(1000 * latency, 1),
                    head_start_ms=round(1000 * (spawn_at - start), 1),
                    spawn_s=round(spawn_at, 4),
                )
                agent_free_at = decided
                previewed_at = spawn_at  # the next piece shows as this one spawns
                spawn_at += (timing["lock_frame"] + delay) / FPS
            best = max(cands, key=heuristic_value)
            agree += pick is best
            score += nes_points(landed.features.lines, level)
            tetrises += landed.features.lines == 4
            since_i = 0 if piece == "I" else since_i + 1
            lines += landed.features.lines
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
                    1000 * latency,
                    list(client.exchanges) if client else [],
                    status,
                    score,
                    landed,
                    timing,
                )
                trace.write(json.dumps(record) + "\n")
                trace.flush()
            board = landed.board
            if landed is not pick:
                note = f"MISSED {pick.key}: {timing and timing['outcome']}"
            show(args, placed, piece, landed, len(cands), note, lines)
        result: dict[str, Any] = {
            "agent": agent,
            "seed": args.seed,
            "pieces": placed,
            "lines": lines,
            "score": score,
            "tetrises": tetrises,
            "topped_out": placed < args.max_pieces,
            "realtime": args.realtime,
            "start_level": args.start_level,
            "missed_moves": missed,
            "prune": args.prune,
            "compact": args.compact,
            "nes_delays": args.nes_delays,
            "pipeline": args.pipeline,
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
        "--agent",
        choices=["choice", "strategy", "score", "heuristic", "rules", "random"],
        default="choice",
    )
    ap.add_argument("--url", default="http://127.0.0.1:8091", help="/v2 service")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pieces", type=int, default=200)
    ap.add_argument(
        "--hints",
        action="store_true",
        help="score agent: append outcome stats to each drawn board",
    )
    ap.add_argument(
        "--rules",
        action="store_true",
        help="strategy agent: apply the strategy's hard rules before asking",
    )
    ap.add_argument(
        "--realtime",
        action="store_true",
        help="the piece falls at NES gravity while the agent decides",
    )
    ap.add_argument(
        "--start-level", type=int, default=0, help="NES level (gravity, scoring)"
    )
    ap.add_argument(
        "--prune", action="store_true", help="ask only about non-dominated placements"
    )
    ap.add_argument("--compact", action="store_true", help="shorter option text")
    ap.add_argument(
        "--nes-delays",
        action="store_true",
        help="real time: NES entry delay and line-clear pause between pieces",
    )
    ap.add_argument(
        "--pipeline",
        action="store_true",
        help="real time: start on the next piece as soon as the last decision is in",
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
