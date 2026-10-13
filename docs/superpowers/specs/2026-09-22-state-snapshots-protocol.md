# Snapshot worker protocol (`unridden-snapshot-v1`)

Companion to `2026-09-22-state-snapshots-design.md`. This is the contract
between the Python snapshot service (`unridden/api/snapshots/`) and the native
`unridden-snapshot-worker` (`unridden/api/native/snapshots/`).

## Runtime profile `split18-30-v1`

- llama.cpp v0.4.1 plus `unridden/api/native/patches/gemma4-layer-range.patch`.
- One model load, two contexts sharing its weights:
  - lower: blocks 0-17 (the spec's "blocks 1-18"), token input, owns their K/V;
  - upper: blocks 18-29 ("19-30"), residual input through `llama_batch.embd`
    at the original positions, owns their K/V, final norm and head.
- Optional `--reference`: a third, stock full-depth context used only by the
  qualification harness (`reference` command).
- `H18[P]` is the raw residual entering block 19 (`layer_inp[18]`), `H30[P]`
  the raw residual after block 30 before the final norm (`layer_inp[30]`).
  `last_normalized` is the post-final-norm head input of the last position.
  All F32, row-major `[N, n_embd]`.
- Worker rejects: non-gemma4, `n_layer != 30`, shared KV layers, per-layer
  embedding inputs, any environment the v1 worker rejects.

## Framing

JSONL, one request per line, one response per line, `MAX_PROTOCOL_BYTES` =
4 MiB both ways. Every request carries `type` and a non-empty `id`; the
response echoes `id`. Tensors never travel in full over JSONL: only a
last-position vector (`base64` little-endian F32) or a bounded row range.

Errors: `{"type":"error","id":...,"code":C,"reason":R,"message":M}`.
`code` is one of

| code | meaning | worker state after |
| --- | --- | --- |
| `invalid_request` | preflight failed; `reason` in `budget`, `control_tokens`, `internal` | unchanged |
| `snapshot_not_found` | unknown `snapshot_id` | unchanged |
| `snapshot_exists` | id already registered | unchanged |
| `snapshot_prefix_mismatch` | the rendered branch does not start with the frozen tokens | unchanged |
| `followup_unsupported` | the template cannot continue this readout snapshot | unchanged |
| `capability_unavailable` | e.g. `readout_blocks: 18` with no registered early head, `promote` of a 30 snapshot | unchanged |
| `integrity_error` | blob checksum/size/shape mismatch on `load` | unchanged |
| `execution_error` | failure after inference began | contexts cleared, snapshots intact |

## Handshake

```json
{"type":"hello","protocol":"unridden-snapshot-v1","profile":"split18-30-v1",
 "model_id":"local-gemma-unridden-v1","model_name":"...","model_sha256":"<64hex>",
 "runtime_sha256":"<64hex>","labels":["A",...],"label_token_ids":[...],
 "context_size":2048,"batch_size":256,"ubatch_size":256,"threads":8,
 "n_layer":30,"n_embd":2816,"split_block":18,"reference_context":false,
 "context_prompt_version":"unridden-gemma-context-v1",
 "generated_tokens":0,"rider_mode":true,"steering_mode":true,
 "callbacks_enabled":false}
```

`steering_mode` is true only for the split profile; workers built before
ADR 0011 omit it. A caller must never send `steering` to a worker that did
not advertise it.

## Prompt profile `unridden-gemma-context-v1`

The native worker renders messages with the model's chat template (same call
as the v1 worker) and tokenizes; the Python side never handles tokens.

- A **context** freeze point: `freeze = {"kind":"context","content_bytes":B}`.
  The worker renders `messages`, finds the first `B` bytes of the LAST user
  message's content, trailing whitespace stripped (chat templates may trim a
  turn), inside the rendered prompt, tokenizes the prompt up to the
  end of that text, and drops the final token (it may merge with what
  follows). Those tokens are the frozen prefix `P`. Fewer than 1 token is
  `invalid_request/internal`.
- A **readout** freeze point: `freeze = {"kind":"readout"}`. `P` is the whole
  rendered prompt plus `answer_prefix`; the readout is at its last position.
- A **branch** renders its own complete `messages` + `answer_prefix` and must
  begin with the parent's exact tokens, else `snapshot_prefix_mismatch`.
  The suffix `Q` is the remainder; `len(Q) >= 1` or `invalid_request`.
- `N + len(Q) > context_size` is `invalid_request/budget`, checked before any
  inference. Positions are never shifted and prefix tokens never dropped.

The Python compiler builds context-profile user content as

```
<CONTEXT_INSTRUCTION>

STATE:
<rendered state>

<continuation block>
```

and sets `content_bytes` to the UTF-8 length of everything up to and including
`"STATE:\n<rendered state>\n\n"`. The continuation for a decision is the v1
question block (`QUESTION:` heading, instructions, `OPTIONS:`, `Reply with one
option label only.`) with `answer_prefix = "Answer:\n"`; for a plain prompt it
is `"PROMPT:\n<text>"` with `answer_prefix = ""`.

A follow-up to a **readout** snapshot is rendered by Python as the snapshot's
own messages, then `{"role":"assistant","content": <answer_prefix without
trailing newline>}`, then the new user turn. Whether that rendering starts with
the frozen tokens is decided by the worker's prefix check; a mismatch there is
reported as `followup_unsupported`.

## Commands

All token/work accounting is measured, not derived. `block_tokens.lower` is
tokens pushed through blocks 1-18, `block_tokens.upper` through 19-30.

### `create`

```json
{"type":"create","id":"c1","messages":[...],"answer_prefix":"",
 "freeze":{"kind":"context","content_bytes":312},
 "checkpoints":{"18":"snap_a18","30":"snap_a30"},
 "labels":null,"top_logits":0}
```

`checkpoints` maps a subset of `{"18","30"}` to fresh ids. `labels` (an
alphabet prefix, as in v1) is allowed only for a readout freeze with a 30
checkpoint and yields label logits at the readout position. `top_logits` in
`0..64` returns the top-k vocabulary logits at the last position for a 30
checkpoint. Only-18 runs no upper block. Both are published atomically: on any
failure neither id exists.

Response:

```json
{"type":"created","id":"c1","prompt_sha256":"<hex of rendered frozen text>",
 "tokens":312,
 "snapshots":[
   {"snapshot_id":"snap_a18","completed_blocks":18,"parent":null,"tokens":312,
    "bytes":{"lower_kv":N,"upper_kv":0,"h18":N,"h30":0}},
   {"snapshot_id":"snap_a30","completed_blocks":30,"parent":"snap_a18","tokens":312,
    "bytes":{"lower_kv":N,"upper_kv":N,"h18":0,"h30":N}}],
 "readout":{"label_logits":[..],"label_token_ids":[..],"allowed_label_mass":0.9,
            "full_vocabulary_argmax":{"token_id":1,"logit":2.0},
            "top_logits":[{"token_id":1,"logit":2.0}]} | null,
 "block_tokens":{"lower":312,"upper":312},
 "timing_ms":{"lower":1.0,"upper":1.0,"capture":1.0,"total":2.0},
 "generated_tokens":0}
```

A 30 snapshot created together with an 18 one shares its lower K/V blob by
reference (`bytes.lower_kv` counts it once, on the 18).

### `promote`

`{"type":"promote","id":"p1","snapshot_id":"snap_a18","new_id":"snap_a30d"}`
finishes `P` through blocks 19-30 from the stored `H18[P]` (same chunking as
`create`). Response `{"type":"promoted","id","snapshot":{...as above, parent =
the 18 id},"block_tokens":{"lower":0,"upper":N},"timing_ms":{...}}`.
`capability_unavailable` for a 30 snapshot. Memoization is the caller's job.

`promote`, `evaluate` and `state_eval` accept an optional `steering` object
(ADR 0011), absent or null for none:

```json
"steering":{"alpha":0.5,"scope":"all"|"last",
            "vector":{"dtype":"f32","byte_order":"little","shape":[2816],
                      "base64":"..."}}
```

Before the upper range runs, the worker adds `alpha * vector` to the H18 rows
it is about to continue: in `promote`, a copy of the parent's `H18[P]`; in a
branch, the branch's own suffix rows `H18[Q]`. `scope` selects every such row
or only the final one. The parent snapshot's stored H18 is never mutated. A
child saved by a steered branch stores the steered suffix rows, so its H18 is
what its upper K/V and H30 were computed from. The width must equal `n_embd`,
all values must be finite and the direction must not be all zeros, else
`invalid_request`. `capability_unavailable` in profile `full-v1`, which has
no host-side H18 boundary. A caller that memoizes promotion must promote
unsteered; a steered promotion is a different snapshot, not the canonical
continuation.

### `evaluate` (typed questions)

```json
{"type":"evaluate","id":"e1","snapshot_id":"snap_a30","readout_blocks":30,
 "questions":[{"id":"status","messages":[...],"answer_prefix":"Answer:\n",
               "labels":["A","B"],"prompt_version":"unridden-gemma-context-v1",
               "save_as":null}]}
```

The snapshot must be a 30 snapshot (`capability_unavailable` otherwise).
Every question is an independent branch from the immutable parent: the worker
restores (or reuses, if still resident and untouched) the parent state, runs
`Q` through both ranges, reads the label logits at the last position, and trims
back to `N`. `save_as` (a fresh id) keeps the branch as a child 30 snapshot
whose parent is `snapshot_id`.

Per question result: the v1 `WorkerQuestionResult` fields (`label_logits`,
`label_token_ids`, `allowed_label_mass`, `full_vocabulary_argmax`,
`prompt_sha256`, `prompt_tokens` = N+len(Q), `processed_tokens` = len(Q),
`reused_tokens` = N, `cache_cleared` = false, `evaluation_mode` =
`"sequential"`, `batch_sequences` = 1, `timing_ms`) plus

```json
"snapshot":{"parent":"snap_a30","suffix_tokens":24,
            "block_tokens":{"lower":24,"upper":24},
            "restore":"resident"|"host","restored_bytes":0,
            "restore_ms":0.0,"inference_ms":12.0,"child":null|{...snapshot row}}
```

Response `{"type":"result","id","model_sha256","runtime_sha256",
"generated_tokens":0,"callbacks_enabled":false,"execution_mode":"split18-30",
"questions":[...]}`.

### `state_eval` (plain prompt)

```json
{"type":"state_eval","id":"s1","snapshot_id":"snap_a30","messages":[...],
 "answer_prefix":"","export":["last_residual","last_normalized","top_logits"],
 "top_logits":20,"save_as":"snap_c30"}
```

Response `{"type":"state","id","suffix_tokens","block_tokens","restore",
"restored_bytes","timing_ms","vectors":{"last_residual":{"dtype":"f32",
"shape":[2816],"base64":"..."},"last_normalized":{...}},"top_logits":[...],
"child":null|{...},"generated_tokens":0}`.

### `inspect`, `vectors`, `drop`

- `inspect {snapshot_id}` returns the snapshot row plus `token_ids` (bounded by
  the context size), `completed_blocks`, `parent`, `resident` (bool).
- `vectors {snapshot_id, which:"h18"|"h30"|"last_normalized", row_begin,
  row_end}` returns at most 64 rows as base64 F32 with `shape`.
  `capability_unavailable` when the snapshot does not hold that tensor.
- `drop {snapshot_id}` frees host memory; the response lists freed bytes. A
  resident context holding it is cleared.

### `save` / `load`

- `save {snapshot_id, directory}` writes into an existing empty directory:
  `native.json` (tokens, prompt sha, completed blocks, n_embd, per-blob size
  and sha256, profile), and any of `lower_kv.bin`, `upper_kv.bin`,
  `h18.f32`, `h30.f32`, `last_normalized.f32`, each fsynced. Response lists
  `files: {name: {bytes, sha256}}`. The Python store owns the manifest, the
  temp-dir-then-rename publish, and re-open validation.
- `load {snapshot_id, directory, files:{name:sha256}}` re-hashes every file,
  checks sizes against `native.json` and the live profile (n_embd, context,
  model/runtime hash, profile id), and only then registers the snapshot in
  host memory. Any mismatch is `integrity_error` and registers nothing.
  Native K/V bytes are validated by `llama_state_seq_set_data` only when the
  snapshot is first restored; a failure there is `execution_error`.

### `reference` (qualification only, requires `--reference`)

`{"type":"reference","id","messages","answer_prefix","labels","capture":[18,30]}`
runs the stock uninterrupted 30-block graph from an empty cache (same chunking)
and returns the v1 question fields plus the last-position `H18` and `H30`
vectors (base64) and `tokens`. Used by the stock-versus-split gate.
