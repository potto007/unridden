# Full-depth snapshot profile: Gemma 4 E4B and E2B, 2026-09-27

Profile `full-v1` ([ADR 0009](../decisions/0009-full-depth-snapshot-profile.md))
on Gemma 4 E4B and E2B UD-Q4_K_XL, plus a split18-30-v1 regression on 26B-A4B
UD-Q4_K_XL with the same new worker binary. llama.cpp v0.4.1 plus
`gemma4-layer-range.patch`, RTX 5090, context 2048, batch/ubatch 256.

| Model | Blocks (checkpoint) | Width | Shared-KV blocks | Sliding window |
| --- | ---: | ---: | ---: | ---: |
| E4B | 42 | 2560 | 18 | 512 |
| E2B | 35 | 1536 | 20 | 512 |

## Release gates

`scripts/unridden/qualify_snapshots.py --profile full-v1` over the frozen
`api-v1-cases.json` corpus (12 cases, 36 questions) plus the 1,415-token case
past the sliding window, driving the worker directly. Gates 4 (promotion) and
5 (stock vs split) do not apply: full-v1 has no split, and its snapshot path
is the stock graph.

| Gate | E2B pass / fail | E4B pass / fail |
| --- | --- | --- |
| 1 capture integrity | 13 / 0 | 13 / 0 |
| 2 execution audit | 51 / 0 | 51 / 0 |
| 3 restore identity (host restore, disk restore in a fresh process) | 51 / 0, all bit-exact | 51 / 0, all bit-exact |
| 6 branch isolation | 39 / 0 | 39 / 0 |
| 7 rejections | 28 / 0 | 28 / 0 |
| informational: branch vs same prompt run fresh | 30 / 8 | 32 / 6 |

### The informational row is chunking noise, not snapshot state

A branch (prefix in the snapshot, suffix appended) differed from the same
prompt run fresh by up to 0.38 in label probability on E2B and 0.11 on E4B,
where 26B stayed within 1e-3. To separate snapshot state from numerics, the
flagged E2B cases ran again at batch 16 with no snapshot involved:

| E2B question | branch vs fresh (batch 256) | fresh at 256 vs fresh at 16 | top probability |
| --- | ---: | ---: | ---: |
| message-policy-gates/requires_review | 3.8e-01 | 3.1e-01 | 0.82 |
| structured-expense-policy/review | 1.1e-01 | 1.1e-02 | 0.85 |
| citation-relations/claim_one | 3.7e-02 | 9.6e-02 | 0.78 |
| citation-relations/claim_three | 2.1e-02 | 1.4e-01 | 0.76 |
| passage-ranking/best_passage | 1.1e-08 | 3.2e-09 | 1.00 |

Changing only how the same tokens are chunked moves the answer by the same
order of magnitude, on the same questions, all of which the model is unsure
about. Confident questions agree to 1e-5 or better either way. Restores
reproduce the live cache bit for bit (gate 3), so the snapshot adds no error
of its own. Small quantized models are simply more sensitive to kernel and
chunk shape than 26B.

## Checks through the public /v2 routes

`scripts/unridden/validate_api_v2.py`, now reading the profile from
`/v2/models`, passed all 16 checks on E2B. A second harness took one context
snapshot of a 97-token ticket through choice, noul and score questions, state
evaluations, Rider and disk persistence:

| Check | E2B full-v1 | E4B full-v1 | 26B split18-30-v1 |
| --- | --- | --- | --- |
| `/v2/models` checkpoints | `[35]` | `[42]` | `[18, 30]` |
| operations | continue, inspect | continue, inspect | continue, promote, inspect |
| branch A, B, A on a resident parent | identical | identical | identical |
| same branch after a host restore | identical, 1.8 MB | identical, 5.6 MB | identical, 21.9 MB |
| state vectors and top logits, resident vs restored | identical | identical | identical |
| Rider tokens, resident vs restored | identical | identical | identical |
| residual tag | `raw_residual_after_final_block` | `raw_residual_after_final_block` | `raw_residual_after_block_30` |
| checkpoint the profile lacks | 422 `capability_unavailable` | 422 `capability_unavailable` | 422 `capability_unavailable` |
| 18 promoted then asked vs paired 30 | n/a | n/a | identical |

"Identical" means equal JSON: bit-identical probabilities and vectors.

## Tetris through snapshots

The comparison game set from the Tetris demo (pruned, compact, static-state,
dense board; real-time rows at 30 Hz tapping), through `/v2` snapshots and
over `/v1` on the same model. The client reads the readout block from
`/v2/models`.

### E4B (checkpoint 42)

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

### E2B (checkpoint 35)

| Game | v1 score | v2 score | v1 pieces | v2 pieces | v1 ms/request | v2 ms/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single, turn-based, seed 0 | 520 | 400 | 69 | 66 | 48.1 | 32.7 |
| lookahead, turn-based, seed 0 | 22,740 | 11,640 | 283 | 213 | 44.9 | 42.6 |
| single, level 18, seed 0 | 6,460 | 6,840 | 57 | 58 | 40.0 | 30.2 |
| lookahead, level 18, seed 0 | 31,160 | 42,180 | 130 | 169 | 44.5 | 40.0 |
| single, level 29, seed 0 | 10,200 | 8,400 | 56 | 51 | 40.5 | 32.5 |
| lookahead, level 29, seed 0 | 11,400 | 47,400 | 53 | 130 | 43.8 | 40.8 |
| single, turn-based, seed 1 | 720 | 500 | 76 | 70 | 40.0 | 31.2 |
| lookahead, turn-based, seed 1 | 3,240 | 28,920 | 130 | 300 | 44.9 | 38.5 |
| single, level 18, seed 1 | 9,880 | 8,740 | 69 | 65 | 35.8 | 31.9 |
| lookahead, level 18, seed 1 | 20,520 | 92,340 | 102 | 300 | 44.9 | 37.4 |
| single, level 29, seed 1 | 14,400 | 11,400 | 63 | 52 | 38.8 | 32.5 |
| lookahead, level 29, seed 1 | 13,200 | 145,800 | 51 | 300 | 50.0 | 37.7 |

- Snapshot requests are faster on both models (E4B 13-25%, E2B 5-30%): the
  board prefix is computed once per move instead of once per question group.
- E4B with lookahead survives 300 pieces in 5 of 6 games on both paths.
- E2B cannot play single-move: every game tops out within 51-76 pieces on
  either path. With lookahead it survives 300 pieces in 3 of 6 games through
  snapshots and in none over `/v1`, so it is usable only with lookahead, and
  even then not reliably. E2B is not adopted; E4B is the small model served
  (`localai-unridden-snapshots-e4b.service`).
- 26B regression: the level-29 lookahead game on seed 0 through the new
  binary reproduced the pre-change run exactly (157,200 points, 119 lines,
  79,492 input tokens).
