## Correction to the published Tetris gameplay results

September 27, 2026. The pipelined real-time simulator supplied information before it was available in the game. **The historical pipelined real-time score, survival and on-time results should not be used as evidence of valid full-speed play.** This applies across model and API routes, including the [initial demo](https://github.com/potto007/unridden/blob/main/docs/results/tetris-demo.md), [26B/E4B comparison](https://github.com/potto007/unridden/blob/main/docs/results/tetris-26b-vs-e4b.md), and real-time Tetris rows in the [full-depth snapshot report](https://github.com/potto007/unridden/blob/main/docs/results/full-depth-snapshots.md).

The controller received the confirmed post-landing board before the previous piece locked. Lookahead and strategy could also receive the following piece before it entered the one-piece preview. This was a controller scheduling bug, independent of the model and gateway.

A local fix gates decisions on observation availability: single-move planning can start after the previous lock, during entry delay; preview-dependent planning waits for the current spawn. Candidate enumeration is now included in controller timing, and native cleanup keeps the controller occupied. The fix, gateway and raw traces are **not yet pushed or released**. This issue publishes the correction and limited local results; it does not announce a software release.

## Limited corrected local rerun

Gemma 4 26B-A4B UD-Q4_K_XL, one RTX 5090, snapshot backend through the local gateway. Twelve games: starting levels 18/29, seeds 0/1, 300-piece cap, 30 Hz simulated taps, entry delays and pipeline enabled. Model policies used pruned, compact, static-state, dense-board Choice, with or without controller-generated two-piece lookahead plans. The model selects among supplied options; the controller performs candidate search, pruning and movement.

Each cell is **pieces / points / missed placements**:

| Policy | Level 18, seed 0 | Level 18, seed 1 | Level 29, seed 0 | Level 29, seed 1 |
| --- | ---: | ---: | ---: | ---: |
| Heuristic | 300 / 87,400 / 0 | 300 / 89,300 / 0 | 248 / 105,600 / 5 | 281 / 121,800 / 4 |
| Choice, single move | 300 / 136,040 / 0 | 300 / 119,700 / 0 | 171 / 70,200 / 3 | 300 / 189,000 / 0 |
| Choice, lookahead | 300 / 99,560 / 0 | 300 / 97,660 / 0 | 39 / 9,000 / 6 | 64 / 24,600 / 8 |

Both model policies reached the cap on both level-18 seeds with no missed placements. At level 29, single-move Choice reached it once; lookahead reached it neither time. Level 29 uses one gravity row per simulated frame (approximately 60 rows/second). These results do **not** establish reliable play at that speed or reliable high-score stacking. Lookahead made no four-line clears in these games.

Per-game median controller compute time was approximately **46–49 ms for single-move Choice** and **92–103 ms for lookahead**. This includes candidate enumeration, planning and request time, plus moves resolved by pruning without a model call. It is not GPU-only inference latency, gateway overhead, or a matched speedup comparison with the old runs. Each game was run once; these are descriptive within-game timings on changing boards.

The clock is a simulator driven by measured decision duration. It uses a seeded seven-bag randomizer, hard drops, simplified NES gravity/scoring and movement. This is not a wall-clock live NES recording or a validated continuous climb from low levels to level 29. Other resident GPU services were preserved.

## Verification and remaining work

All 2,903 recorded moves passed information-availability checks. The local verification run passed 310 tests, Ruff checks/formatting, mypy and diff whitespace checks. Deterministic tests cover hidden-preview counterfactuals, confirmed-board availability, controller occupancy, all current policies and turn-based behavior.

The local artifacts are `outputs/tetris-preview-fix/corrected-results.json`, `corrected-summary.json`, `corrected-metadata.json`, and twelve `corrected-*.jsonl` traces. The evaluated `scripts/unridden/play_tetris.py` SHA-256 is `9e12db069519af4c9e285c7e6bae5de6c74d15c815d40fd0f1c44ad037c56add`. These paths identify currently local evidence, not downloadable files in main.

Still open: publish the reviewed implementation and reproducibility artifacts, and rerun the full E4B/E2B/26B, native/gateway and movement matrix. The corrected gateway games are not a matched comparison against the historical direct-API runs. Turn-based results, ordinary seven-suite measurements, and snapshot restoration/isolation tests are unaffected by this scheduling bug. Historical request durations remain observations of their old workloads, not proof of valid real-time gameplay.

Project-author update, prepared with Codex assistance.
