# Full-depth snapshot profile: first GPU check, 2026-09-27

Profile `full-v1` ([ADR 0009](../decisions/0009-full-depth-snapshot-profile.md))
against Gemma 4 E4B UD-Q4_K_XL, plus a split18-30-v1 regression on 26B-A4B
UD-Q4_K_XL with the same new worker binary. llama.cpp v0.4.1 plus
`gemma4-layer-range.patch`, RTX 5090, context 2048, batch/ubatch 256, served
`--gpu --snapshots --no-v1` on an ad-hoc port.

This is a smoke check, not the eight-gate qualification of
[split18-30-v1](state-snapshots-qualification.md):
`scripts/unridden/qualify_snapshots.py` still assumes the split profile and
has not been run on full-v1. The profile stays experimental.

## Checks through the public /v2 routes

One context snapshot of a 97-token support ticket, then choice, noul and score
questions, state evaluations, Rider, disk persistence.

| Check | E4B full-v1 | 26B split18-30-v1 |
| --- | --- | --- |
| `/v2/models` checkpoints | `[42]` | `[18, 30]` |
| operations | continue, inspect | continue, promote, inspect |
| branch A, B, A on a resident parent | identical | identical |
| same branch after the parent was displaced (host restore) | identical, 5.6 MB restored | identical, 21.9 MB restored |
| state-evaluation vectors and top logits, resident vs restored | identical | identical |
| Rider tokens, resident vs restored | identical | identical |
| residual tag (`last_residual`, `vectors`) | `raw_residual_after_final_block` | `raw_residual_after_block_30` |
| checkpoint the profile lacks (30 / 42) | 422 `capability_unavailable` | 422 `capability_unavailable` |
| 18 promoted then asked vs paired 30 | n/a | identical |

"Identical" means equal JSON, i.e. bit-identical probabilities and vectors.

## Tetris through snapshots

The comparison game set from the Tetris demo (pruned, compact, static-state,
dense board; real-time rows at 30 Hz tapping), played through `/v2` snapshots
at checkpoint 42 on E4B, next to the same set over `/v1` on E4B.

| Game | v1 score | v2 score | v1 pieces | v2 pieces | v1 ms/request | v2 ms/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single, turn-based, seed 0 | 35,380 | 4,680 | 300 | 147 | 48.4 | 42.2 |
| lookahead, turn-based, seed 0 | 31,580 | 31,580 | 300 | 300 | 51.6 | 44.0 |
| single, level 18, seed 0 | 104,120 | 35,720 | 300 | 138 | 46.5 | 35.2 |
| lookahead, level 18, seed 0 | 94,620 | 94,240 | 300 | 300 | 51.3 | 43.6 |
| single, level 29, seed 0 | 111,600 | 30,000 | 231 | 86 | 46.5 | 36.4 |
| lookahead, level 29, seed 0 | 40,800 | 148,800 | 115 | 300 | 53.8 | 43.3 |
| single, turn-based, seed 1 | 2,720 | 10,980 | 120 | 171 | 47.8 | 35.5 |
| lookahead, turn-based, seed 1 | 32,900 | 29,700 | 300 | 300 | 51.4 | 44.9 |
| single, level 18, seed 1 | 22,800 | 55,100 | 105 | 142 | 47.6 | 33.3 |
| lookahead, level 18, seed 1 | 97,660 | 92,720 | 300 | 300 | 51.4 | 45.3 |
| single, level 29, seed 1 | 12,600 | 45,600 | 53 | 122 | 45.2 | 35.2 |
| lookahead, level 29, seed 1 | 154,200 | 135,000 | 300 | 284 | 51.4 | 46.2 |

- Requests are 13-25% faster through snapshots: the board prefix is computed
  once per move instead of once per question group.
- Lookahead survives 300 pieces in 5 of 6 games on both paths. Single-move
  play on E4B stays fragile on both; which games it loses differs, because the
  snapshot prompt splits state and question differently from `/v1`.
- 26B regression: the level-29 lookahead game on seed 0 through the new
  binary reproduced the pre-change run exactly (157,200 points, 119 lines,
  79,492 input tokens).
