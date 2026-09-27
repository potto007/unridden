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
