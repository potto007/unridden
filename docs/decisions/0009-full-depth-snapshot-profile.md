# 0009. A full-depth snapshot profile for any Gemma 4

- Status: accepted
- Date: 2026-09-26
- Extends [0006](0006-split-execution-for-state-snapshots.md). The
  split18-30-v1 profile, its patch and its wire contract are unchanged.

## Context

split18-30-v1 needs a model whose only state across the block-18 boundary is
the residual. Gemma 4 E4B (and E2B) break that: 18 of E4B's 42 blocks reuse
K/V from earlier blocks, and every block reads a per-layer input computed from
the tokens. The layer-range patch refuses a partial range on such models, and
the worker refused any model without exactly 30 blocks. `/v2` was therefore
26B-A4B only, although most of its value (prefix reuse across branches,
restore, save/load, Rider) does not need an intermediate checkpoint at all.

## Decision

- **A second profile, `full-v1`.** The worker takes `--profile full-v1`
  (default `split18-30-v1`); the API takes `--snapshot-profile`. One context
  runs the whole model with the default range `(0, -1)`, which is the stock
  graph, so every Gemma 4 qualifies, shared K/V and per-layer inputs included.
  No llama.cpp change.
- **One checkpoint, the model's block count.** Snapshots exist only at
  `n_layer` (42 for E4B). `checkpoints` and `readout.completed_blocks` are
  integers validated against the running profile, so a 26B client sending 30
  is unaffected and an E4B client sends 42. There is no promotion and no early
  checkpoint; a checkpoint the profile does not offer is
  `capability_unavailable`.
- **Same snapshot rules as 0006.** Fixed captures for the context's lifetime
  (residual after the final block, head input), SWA-masked cells kept,
  immutable host-side snapshots, resident fast path, branch then trim back.
- **Wire reuse.** The one context is the protocol's `lower` range and covers
  the whole model: its K/V is `lower_kv`, `block_tokens.upper` is 0, and the
  store's layer map is `{"lower": [0, n_layer]}`. The worker's `h30` slot holds
  the residual after the final block, tagged `raw_residual_after_final_block`;
  the public `vectors` name for it is `final`. `/v2/models` reports the
  profile's checkpoints, operations and readout blocks.
- **Rider continues snapshots; baselines stay split-only.** A full-v1 worker
  refuses `--reference`: its snapshot path is already the stock graph, so the
  split qualification baselines have nothing to compare.
- **One bundle, both profiles.** The worker bundle manifest lists
  `profiles`; the API refuses a profile the installed bundle does not list.

## Consequences

- A context snapshot always pays for the full depth at create time; the
  cheaper 18-block freeze and its promotion remain 26B-only.
- Restore identity (bitwise equal readouts from a resident and a restored
  parent) is the qualification gate for full-v1, as it was for split18-30-v1.
