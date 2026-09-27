# Tetris: Gemma 4 26B-A4B vs E4B

Run on 2026-09-27, one RTX 5090, Gemma 4 26B-A4B and E4B instruction-tuned,
both UD-Q4_K_XL from `unsloth/gemma-4-*-GGUF`, on llama.cpp release `v0.4.1`,
commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, with the layer-range patch.
Each model ran through both routes, one model on the GPU at a time:

- **v2**: the snapshots service (`serve --snapshots --no-v1`). The 26B-A4B
  uses the `split18-30-v1` profile (readout at block 30); the E4B uses
  `full-v1` (one context over all 42 blocks, readout from the final residual,
  ADR 0009).
- **v1**: the Choice API on the v1 worker (`serve`), one request per question,
  no snapshots.

The engine, agents and flags are the ones in [tetris-demo.md](tetris-demo.md).
This is a comparison on one machine, not a benchmark: five seeds per cell, and
the only non-model baseline is the fixed heuristic.

## Matrix

Every setup (model x route x agent) played the same 25 games: seeds 0-4 in
five modes, 300-piece cap, NES scoring. 225 games in all, including the
heuristic, and none failed.

- Agents: `--agent choice` (**1-move**: one Choice question over the pruned
  placements of the current piece) and `--agent choice --lookahead`
  (**lookahead**: plans for the current and previewed piece together).
- Shared flags: `--prune --compact --static-state --dense-board`.
- **Turn-based**: no clock, starting at level 0.
- **Real time**: `--realtime --nes-delays --pipeline`, starting at level 18
  (3 frames per row) or 29 (1 frame per row). The controller either holds the
  direction (**hold**, NES auto-shift: a shift at once, then after 16 frames,
  then every 6) or taps it 30 times a second (**tap**, `--movement tap
  --tap-hz 30`: a shift every 2 frames).
- **Heuristic**: `--agent heuristic`, the four-feature placement score with no
  model, in the same 25 games.

## Score

Mean score over seeds 0-4, with the number of games that reached piece 300:

| Setup | Turn-based L0 | L18 hold | L18 tap | L29 hold | L29 tap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 26B-A4B v2 1-move | 38,200 (5/5) | 29,792 (0/5) | 112,936 (5/5) | 0 (0/5) | 117,840 (3/5) |
| 26B-A4B v2 lookahead | 32,296 (5/5) | 39,444 (1/5) | 95,760 (5/5) | 0 (0/5) | 164,760 (4/5) |
| 26B-A4B v1 1-move | 37,896 (5/5) | 18,164 (0/5) | 117,192 (5/5) | 0 (0/5) | 101,520 (2/5) |
| 26B-A4B v1 lookahead | 33,820 (5/5) | 43,244 (0/5) | 100,928 (5/5) | 0 (0/5) | 159,360 (5/5) |
| E4B v2 1-move | 6,444 (0/5) | 10,412 (0/5) | 35,112 (0/5) | 0 (0/5) | 37,320 (0/5) |
| E4B v2 lookahead | 32,556 (5/5) | 17,328 (0/5) | 99,864 (5/5) | 0 (0/5) | 155,400 (4/5) |
| E4B v1 1-move | 25,184 (3/5) | 19,456 (0/5) | 74,328 (2/5) | 0 (0/5) | 71,520 (0/5) |
| E4B v1 lookahead | 33,136 (5/5) | 13,984 (0/5) | 98,572 (5/5) | 0 (0/5) | 125,520 (3/5) |
| Heuristic (no model) | 30,376 (5/5) | 31,996 (0/5) | 91,960 (5/5) | 0 (0/5) | 102,600 (1/5) |

The same setup, mode and seed, 26B-A4B against E4B (wins out of 5; the last
column is the range, over the four modes, of the E4B's total score divided by
the 26B-A4B's):

| Route | Agent | Turn-based | L18 hold | L18 tap | L29 tap | E4B / 26B-A4B |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 1-move | 26B 5-0 | 26B 4-1 | 26B 5-0 | 26B 5-0 | 0.17-0.35 |
| v2 | lookahead | E4B 3-2 | 26B 3-2 | E4B 3-2 | 26B 3-2 | 0.44-1.04 |
| v1 | 1-move | 26B 4-1 | 26B 3-2 | 26B 5-0 | 26B 3-2 | 0.63-1.07 |
| v1 | lookahead | 26B 3-2 | 26B 5-0 | 26B 3-2 | 26B 4-1 | 0.32-0.98 |

L29 hold is left out: every game there scored 0.

## Speed and tokens

Decision time is per move that reached the model (a placement pruned to one
needs no call), over all 25 games of the setup:

| Setup | ms p50 | ms p95 | Calls per turn-based game | Input tokens per turn-based game | Heuristic agreement, turn-based | Tetrises in 25 games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 26B-A4B v2 1-move | 52 | 102 | 232 | 52,023 | 0.85 | 10 |
| 26B-A4B v2 lookahead | 106 | 210 | 266 | 83,107 | 0.68 | 2 |
| 26B-A4B v1 1-move | 71 | 103 | 229 | 72,850 | 0.87 | 12 |
| 26B-A4B v1 lookahead | 107 | 174 | 268 | 111,104 | 0.68 | 3 |
| E4B v2 1-move | 31 | 56 | 113 | 27,950 | 0.70 | 3 |
| E4B v2 lookahead | 72 | 118 | 274 | 87,339 | 0.65 | 3 |
| E4B v1 1-move | 49 | 58 | 198 | 65,701 | 0.75 | 0 |
| E4B v1 lookahead | 80 | 124 | 274 | 114,413 | 0.64 | 0 |

The E4B v2 1-move games made fewer calls because they ended early (102-173
pieces).

GPU memory for the whole card, sampled once a minute during each phase. It
includes the bge-m3 embedding server (about 1 GiB) and the Windows desktop:

| Server | Whole card |
| --- | ---: |
| 26B-A4B v1 | 25.4 GiB |
| 26B-A4B v2 | 20.8 GiB before the run, 25.3 GiB after the end-of-run restart |
| E4B v2 | 12.2 GiB |
| E4B v1 | 12.2 GiB |

The difference between the two 26B-A4B v2 readings was not investigated.

## Real time

Share of moves that landed where the agent chose ("on time"):

| Setup | L18 hold | L18 tap | L29 hold | L29 tap |
| --- | ---: | ---: | ---: | ---: |
| 26B-A4B v2 1-move | 95% | 100% | 29% | 100% |
| 26B-A4B v2 lookahead | 97% | 100% | 23% | 99% |
| 26B-A4B v1 1-move | 94% | 100% | 25% | 99% |
| 26B-A4B v1 lookahead | 96% | 100% | 25% | 100% |
| E4B v2 1-move | 92% | 99% | 27% | 97% |
| E4B v2 lookahead | 92% | 100% | 24% | 100% |
| E4B v1 1-move | 94% | 100% | 30% | 98% |
| E4B v1 lookahead | 91% | 100% | 23% | 99% |
| Heuristic (no model) | 95% | 100% | 35% | 98% |

Almost every miss is "landed before reaching the target": the piece reaches the
stack while the controller is still shifting it. Decisions arrived before the
piece spawned in nearly every move, because pipelining starts each decision on
the previewed piece. The heuristic, which decides in no time, misses as often
as the models.

## Findings

- **The controller decides the real-time games, not the model.** Holding the
  direction at level 29 cannot reach the outer columns before a piece lands
  (the second shift comes 16 frames after the first, and the piece falls a row
  every frame), so about three moves in four miss and every setup, the
  heuristic included, scores 0. Tapping at 30 Hz lands 97-100% of moves at
  both levels. At level 18, holding still misses 3-9% of moves as the stack
  rises, and no setup survived there except one 26B-A4B lookahead game.
- **Tap games replay the turn-based games.** When every move is on time, the
  agent makes the same decisions as in the turn-based game on that seed. All
  30 of the 26B-A4B's and the E4B lookahead's L18 tap games pick exactly what
  the turn-based game on the same seed picked, all 300 moves, scored at level
  18's multiplier; the E4B 1-move games match until their last few moves. The
  tap columns are therefore not independent samples of the model's play.
- **The E4B needs lookahead.** With lookahead it matches the 26B-A4B: 32,556
  vs 32,296 turn-based on v2, and every lookahead game survived at turn-based
  and L18 tap on both routes, for both models. Choosing single moves, it
  survived 3 of 5 turn-based games on v1 and none on v2, while the 26B-A4B
  survived all 10.
- **The E4B's single-move choices are close calls.** On boards identical
  between its v1 and v2 games, the two routes picked the same placement in 32
  of 37 decisions. The largest probability difference per decision had a
  median of 0.03-0.13 per seed, about the 0.11 chunking noise measured for
  the E4B in [full-depth-snapshots.md](full-depth-snapshots.md). The gap
  between its top two options was often 0.1-0.5, so a small difference flips
  a pick and the games diverge within 7-23 moves. The 26B-A4B's margins were
  about 1.0 and, on four of five seeds, its routes agreed to within 0.001.
  With five seeds, 0 of 5 surviving on v2 against 3 of 5 on v1 is not a
  significant difference (Fisher exact, p = 0.17).
- **The 26B-A4B plays best with single moves, when it can.** Its 1-move games
  score 12-18% more than its lookahead games turn-based and take 4-5 times as
  many Tetrises, as in [tetris-demo.md](tetris-demo.md). Lookahead wins at
  L29 tap, where survival pays: 164,760 against 117,840 on v2.
- **The E4B is faster and much smaller.** Its v2 decisions take 31 ms (1-move)
  and 72 ms (lookahead) at the median, against 52 and 106 ms for the
  26B-A4B, and the card holds 8-13 GiB less. v2 is faster than v1 for both
  models on single moves (52 vs 71 ms for the 26B-A4B, 31 vs 49 ms for the
  E4B), and reports 24-29% fewer input tokens per full-length turn-based game,
  since the board is prefilled once per move instead of once per question.
- **Every model setup beat the heuristic turn-based** except the E4B's
  single-move ones.

## Reproduce

One game, with the server for the setup on port 8091 (v2) or 8090 (v1):

```bash
uv run python scripts/unridden/play_tetris.py --agent choice --seed 0 \
  --max-pieces 300 --prune --compact --static-state --dense-board \
  --realtime --nes-delays --pipeline --movement tap --tap-hz 30 --start-level 29 \
  --url http://127.0.0.1:8091 --api v2 --trace outputs/tetris/e4b-v2-L29-tap-s0.jsonl
```

Add `--lookahead` for the lookahead agent, drop the four real-time flags and
`--start-level` for turn-based, use `--movement das` for hold, and
`--agent heuristic` (no server) for the reference. The E4B v2 server is
`serve --gpu --snapshots --no-v1` with the E4B model path, which selects the
`full-v1` profile.
