# Tetris demo

Run on 2026-09-26, one RTX 5090, Gemma 4 26B-A4B UD-Q4_K_XL, through the `/v2`
snapshots service (`serve --snapshots`, profile `split18-30-v1`, readout at
block 30) on llama.cpp release `v0.4.1`, commit
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, with the layer-range patch.

This is a demonstration, not a benchmark: three seeds, one model, and a fixed
heuristic as the only non-trivial baseline.

## Setup

`scripts/unridden/play_tetris.py` is a stdlib-only Tetris engine: a 10x20 well,
a seeded 7-bag, hard drops only (no slides or spins). Each turn it enumerates
every distinct placement of the current piece, up to 34, and asks an agent for
one. Every agent sees the same piece sequence for a given seed.

- **choice**: the board is the state, prefilled once as a context snapshot. One
  Choice question lists the placements, each described by its outcome (lines
  cleared, new holes, stack height, bumpiness). More than 26 placements exceed
  the label alphabet, so they run as two independent Choice questions over
  halves of the list and the two winners meet in a final.
- **score**: one Score question per placement (five levels, terrible to
  excellent) with the candidate drawn into the board as `@`; the highest
  expected level wins. `--hints` appends the same outcome line the Choice agent
  gets.
- **heuristic**: a fixed linear evaluator over aggregate height, lines, holes
  and bumpiness, no model.
- **random**: uniform over placements.

A move with a single legal placement is taken without asking the model.

## Results

| Agent | Seed | Pieces | Lines | Outcome | Heuristic agreement | Model ms/move |
| --- | --- | --- | --- | --- | --- | --- |
| choice | 0 | 500 | 190 | survived | 83.2% | 304 |
| choice | 1 | 411 | 148 | topped out | 79.3% | 339 |
| choice | 2 | 323 | 112 | topped out | 84.8% | 314 |
| score + hints | 0 | 38 | 2 | topped out | 23.7% | 1,287 |
| score, board only | 0 | 27 | 0 | topped out | 33.3% | 1,115 |
| heuristic | 0 | 500 | 186 | survived | - | - |
| heuristic | 1 | 500 | 197 | survived | - | - |
| heuristic | 2 | 500 | 198 | survived | - | - |
| random | 0 | 20 | 0 | topped out | - | - |

Runs were capped at 500 pieces for choice and heuristic, 100 for score with
hints and 60 for board-only score.

- Comparing placements side by side in one Choice question plays competently:
  it matched or beat the heuristic on seed 0 and cleared 112 to 190 lines in
  every game, zero generated tokens throughout.
- Rating each placement in isolation does not work, even with the outcome line
  included. With hints the median placement's expected level is 0.00
  (terrible) and the 90th percentile 0.97; board only, 1.04 and 2.95. Nearly
  everything lands at the bottom of the scale, and the argmax agreed with the
  heuristic on only 24% and 33% of moves.
- Reading the drawn board alone is at the level of random play: 38 holes after
  15 pieces.

These games split more than 26 placements into contiguous halves. The engine
now splits them into interleaved groups, and a split that exceeds the context
is split again. That changes individual games: Unridden's plain Choice agent on
seed 0, which survived 500 pieces above, topped out at piece 286 with 98 lines
after the change.

## High-score goals and real time

The engine now scores NES points: 40, 100, 300 and 1200 for 1-4 lines, times
(level + 1). It also shows the next piece and has two further modes.

- `--agent strategy` puts the scoring goal and a standard high-score strategy
  in the state: keep column 9 as a Tetris well, stack flat, burn only when
  high, use the next piece. It describes each placement in those terms.
  `--rules` also enforces that strategy's hard rules in code before asking.
- `--realtime --start-level N`: the piece falls at NES gravity while the
  agent decides, then rotates, shifts on NES auto-shift timing and
  hard-drops. A late or blocked move locks wherever the piece is.

Seeds 0 and 1, 300-piece cap, same service as above:

| Agent | Mode | Seed 0 | Seed 1 | ms/move |
| --- | --- | --- | --- | --- |
| choice | turn-based | 286 pieces, 23,860 pts | 300, 30,240 | 300-336 |
| choice | real time from level 0 | 145, 5,600 | 271, 25,400 | 298-327 |
| choice | real time from level 18 | 17, 1,520 | 16, 0 | 312-318 |
| strategy | turn-based | 143, 6,980 | 48, 140 | 438-509 |
| strategy | real time from level 0 | 122, 6,180 | 41, 100 | 485-523 |
| heuristic | real time from level 0, seed 0 | 283, 22,060 | - | ~0 |
| heuristic | real time from level 18, seed 0 | 210, 57,000 | - | ~0 |

- The high-score strategy did not help. As text it scored below the plain
  prompt. Enforcing its rules in code was worse: a 200-piece trial topped
  out at piece 122, and random choices within the same rules lasted 89-141
  pieces. With hard drops only, holding a column open for Tetrises costs more
  than it earns. Both of this matrix's Tetrises came from the strategy agent
  on seed 0.
- Real time costs the model through latency: at level 18 a piece falls 20 rows
  a second, so 300 ms of reading is 6 rows before it moves. The instant
  heuristic lasts 210 pieces there; the model lasts 15-17.

## Latency optimizations

Four flags cut the decision time and use the game's own pauses:

- `--prune` offers only placements no other placement beats or ties on every
  measure (lines, holes, aggregate height, bumpiness), with no weights. On
  recorded boards the median set is 2 options (max 9), and a quarter of moves
  need no call.
- `--compact` shortens the option text.
- `--nes-delays` adds the NES entry delay and line-clear pause.
- `--pipeline` starts on the previewed piece as soon as the agent is free,
  capped by the one-piece preview.

Plain prompt, seeds 0 and 1, 300-piece cap:

| Setup | Mode | Seed 0 | Seed 1 | Decision p50 |
| --- | --- | --- | --- | --- |
| all four | turn-based | 300 pieces, 39,040 pts | 300, 32,540 | 86 ms |
| all four | real time from level 0 | 300, 39,040, 0 late | 300, 32,540, 0 late | 86 ms |
| all four | real time from level 18 | 179, 75,240 | 51, 7,980 | 86 ms |
| delays only | real time from level 18 | 17, 1,520 | 17, 0 | 240 ms |
| delays + pipeline | real time from level 18 | 27, 1,520 | 43, 7,220 | 234 ms |

- With all four, no move was late: 86 ms fits inside the median 0.5 s head
  start. At level 0 the real-time games were move-for-move identical to the
  turn-based ones.
- Pipelining without pruning is not enough: at 234 ms, 14 of 68 moves were
  still late at level 18.
- At level 18 the model still misses 4-6 moves a game. NES auto-shift cannot
  carry a piece across the board at 3 frames per row, and the zero-latency
  heuristic misses 8 there too.

## Reproduce

```bash
uv run python -m unridden.api.cli serve --gpu --snapshots --no-v1 --port 8091 &
uv run python scripts/unridden/play_tetris.py --agent choice --seed 0 \
  --max-pieces 500 --trace outputs/tetris/choice-seed0.jsonl
uv run python scripts/unridden/play_tetris.py --agent heuristic --seed 0 --max-pieces 500
```

`--watch` redraws the board in the terminal each move. Open a trace in
`scripts/unridden/tetris_replay.html` to step through a game: the board, the
probability read for every placement, the heuristic's pick, and the raw
request and response of every HTTP exchange.
