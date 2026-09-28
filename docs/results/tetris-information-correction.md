# Tetris information-timing correction

September 27, 2026. The real-time simulator's pipeline gave the controller
information too early. This was a controller bug, independent of the model or
gateway. This report accompanies the source correction; the v0.4.0 release
predates it.

## What was wrong

For piece *n*, the old loop constructed the board after piece *n−1* landed and
passed piece *n+1* in `Status.next_piece`. It then credited the decision with a
start as early as piece *n−1*'s spawn. Two inputs could therefore arrive early:

- The confirmed board after the preceding landing, before that piece locked.
- The following piece used by lookahead and strategy, before it entered the
  one-piece preview at piece *n*'s spawn.

An eight-move CPU reproduction found premature preview access on moves 2–8
for both lookahead and strategy. Single-move Choice also received the confirmed
board before the preceding lock on four moves. Its existing pipeline test had
lookahead disabled, while its lookahead test was turn-based, so neither caught
the combination. Evidence is saved under `outputs/tetris-preview-fix/` in
`before-audit.json` and `before-*.jsonl`. These probes used fake model answers;
their elapsed times are not inference benchmarks.

## Corrected rules

The simulator owns the piece stream. A controller decision now starts only
after all required observations and the controller itself are available:

| Input | Earliest availability |
| --- | --- |
| Piece being planned | Previous piece's spawn, when it appears in the one-piece preview; initial piece at game start |
| Confirmed landing board, score and line count | Previous piece's lock; line clearing is deterministic at that event |
| Piece following the planned piece | Planned piece's own spawn |

Single-move policies may plan during the entry delay after the previous lock.
Their `Status.next_piece` is `None` until the following preview appears.
Lookahead, including the random/pruned-plan baseline, and strategy wait for
the planned piece to spawn. Their helpers reject an unavailable preview.
The controller does not receive a speculative future landing board.

The same game loop applies to gateway, native v1/v2, model and non-model modes.
Turn-based games already have the current spawn's preview and retain their
decision policy. There is no option to restore the premature-information path.

Traces identify `one-preview-confirmed-board-v1` and record decision start,
decision ready, board availability, current-piece availability, following-piece
availability and whether that following preview was visible. Trace `next_piece`
records the controller's observation, not the simulator's hidden stream.

Timing now includes initial placement enumeration as well as `choose()` in
`compute_ms`; previously initial enumeration was outside the timer. Native
snapshot cleanup is recorded separately and keeps the controller busy before
its next decision. Replay rendering and trace-file writes are still outside
the simulated controller clock.

## Limited corrected gameplay rerun

The existing healthy 26B-A4B UD-Q4_K_XL snapshot service and automatic gateway
were used without restart. RTX 5090, 32,607 MiB, driver 617.14, WSL Linux;
native profile `split18-30-v1`, context 2048, batch/ubatch 256, eight CPU
threads. Other resident GPU services were left running. No competing benchmark
was observed before the run.

Twelve games: two seeds, starting at levels 18 and 29, 300-piece cap, 30 Hz
simulated tapping, entry delays and pipeline enabled. Model runs used pruned,
compact, static-state, dense-board Choice with or without lookahead. The
heuristic used the same rules and piece streams. Each game was run once, in
level/seed order, heuristic then single-move then lookahead. This is a
correctness rerun, not an alternating latency comparison or a general model
ranking. All availability assertions passed over all recorded moves.

Each cell is **pieces / points / missed placements**, seeds 0 and 1:

| Policy | Level 18, seed 0 | Level 18, seed 1 | Level 29, seed 0 | Level 29, seed 1 |
| --- | ---: | ---: | ---: | ---: |
| Heuristic | 300 / 87,400 / 0 | 300 / 89,300 / 0 | 248 / 105,600 / 5 | 281 / 121,800 / 4 |
| Choice, single move | 300 / 136,040 / 0 | 300 / 119,700 / 0 | 171 / 70,200 / 3 | 300 / 189,000 / 0 |
| Choice, lookahead | 300 / 99,560 / 0 | 300 / 97,660 / 0 | **39 / 9,000 / 6** | **64 / 24,600 / 8** |

Both lookahead games still survived level 18, but neither survived level 29.
The earlier claim that lookahead survived both level-29 seeds is not reproduced
under legal preview timing. These are current gateway runs: historical direct
native results used different timing and possibly different runtime/build
conditions. Their numerical differences are not an isolated causal estimate
of either the clock correction or gateway overhead.

Median / nearest-rank p95 **controller compute milliseconds per move**,
including moves that pruning resolved without a model call:

| Policy | Level 18, seed 0 | Level 18, seed 1 | Level 29, seed 0 | Level 29, seed 1 |
| --- | ---: | ---: | ---: | ---: |
| Choice, single move | 49.2 / 79.9 | 46.2 / 71.8 | 47.7 / 76.5 | 46.1 / 70.6 |
| Choice, lookahead | 92.0 / 147.7 | 100.0 / 149.8 | 101.8 / 202.4 | 102.6 / 147.9 |

These are descriptive within-game timings on changing boards, not paired
request benchmarks. Setup/capture can contribute to a timed decision. They
are distinct from GPU-only inference, historical HTTP request averages, and
the separately measured [gateway overhead](context-harness.md).

## Verification and replay

- Deterministic tests exercise every current policy's observation boundary,
  a busy controller, no-entry-delay timing, hidden-preview rejection, and
  turn-based preview availability.
- A counterfactual test changes a still-hidden following piece and verifies
  that the earlier model request, selected placement and resulting board do
  not change.
- Candidate-enumeration time and native cleanup occupancy have separate tests.
- The replay shows hidden previews, marks old real-time traces as unverified,
  and uses actual landing cells and features after a missed target. New traces
  contain two-move option descriptions; older plan traces fall back to their
  original exchanges instead of failing to render.
- Browser checks covered current lookahead, a hidden-preview single move,
  a missed landing, and a legacy lookahead trace.
- Final verification: 310 tests passed; Ruff checks/formatting, mypy and
  `git diff --check` passed. Both model and gateway health checks remained
  healthy; neither service was restarted. The temporary replay server was
  stopped after inspection.

The tests are `unridden/tests/test_tetris_information.py` plus the existing
Tetris and gateway suites. Local evidence includes `corrected-results.json`,
`corrected-summary.json`, `corrected-metadata.json`, the twelve
`corrected-*.jsonl` traces and `run_corrected.py` under
`outputs/tetris-preview-fix/`. Metadata records the source hash and discovered
model profile. Raw local artifacts are not included in a public release.

Reproduce one corrected gateway game with both services healthy:

```bash
uv run python scripts/unridden/play_tetris.py --agent choice --lookahead \
  --seed 0 --max-pieces 300 --prune --compact --static-state --dense-board \
  --realtime --nes-delays --pipeline --movement tap --tap-hz 30 \
  --start-level 29 --trace outputs/tetris/corrected-lookahead-L29-s0.jsonl
```

Vary `--seed` and `--start-level`, omit `--lookahead` for single-move Choice,
or select `--agent heuristic`. Explicit native comparison modes still use the
corrected scheduler. They do not recreate the old timing bug.

## Historical claims and remaining limits

All historical **pipelined real-time** rows in the Tetris demo, expanded
26B/E4B comparison, and full-depth E4B/E2B snapshot report require rerun for
score, survival and on-time claims. Their tables remain intact and prominently
marked. Turn-based rows, native snapshot restoration/isolation qualification,
and ordinary seven-suite results are unaffected by this bug. Historical request
durations remain observations of those old workloads; they establish neither
corrected gameplay latency nor gateway performance.

The complete model/route/movement/seed matrix has not been rerun. E4B/E2B,
ordinary v1, direct native v2 and held-direction corrected gameplay remain
unmeasured here. This local simulator still uses a seven-bag randomizer, hard
drops, simplified rotation/movement and level progression; it is not a faithful
NES emulator, physical input system, wall-clock live game, or validated
continuous low-to-high-level demonstration. The replay animation is illustrative,
not a recording of every gravity frame. Frame rounding and normal measured
latency variation can change real-time outcomes.
