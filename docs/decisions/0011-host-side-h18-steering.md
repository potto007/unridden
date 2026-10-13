# 0011. Host-side residual steering at the H18 handoff

- Status: accepted
- Date: 2026-10-13
- Extends [0006](0006-split-execution-for-state-snapshots.md). Implements
  proposal 3 of
  [intermediate-thinking-steps-review.md](../research/intermediate-thinking-steps-review.md).
  The unsteered wire contract is unchanged.

## Context

Mid-depth activation steering (`h + alpha * v`) was assessed in `local-ai`
(`intermediate-state-control-assessment.md`) but never built: wiring
`llama_set_adapter_cvec` looked invasive because llama.cpp applies control
vectors inside the graph. That obstacle no longer exists. In
`split18-30-v1` the snapshot worker already externalizes the block-18
boundary: `worker::create`, `worker::promote` and `worker::run_branch` hold
H18 as an explicit host-side `matrix` before handing it to
`runtime_.run_upper`. Steering is now a host-side vector add on those rows,
with zero llama.cpp changes.

## Decision

- **Optional `steering` on the worker's `promote`, `evaluate` and
  `state_eval` commands, and on the public `/v2/decisions` and
  `/v2/state-evaluations` bodies.** The spec is `{alpha, scope, vector}`,
  where `vector` is a little-endian f32 base64 blob of width `n_embd` (the
  export artifact transport, without a representation tag: a steering
  direction is caller-supplied, not a captured residual) and `scope` is
  `all` or `last` (every row about to enter the upper blocks, or only the
  final row). Absent or null means no steering, and the unsteered path is
  bit-identical to before, sibling isolation included. `create` and `ride`
  are never steered, and the gateway does not gain a steering surface (it
  never proxies `/v2` internals).
- **Copy on steer; the parent snapshot is never mutated.** A branch steers
  its own suffix rows (`H18[Q]`); a steered promotion steers a copy of the
  parent's full `H18[P]`. The parent's shared H18 record keeps its exact
  bytes, so `A, B, A` branch isolation and snapshot immutability hold
  whether or not a sibling branch was steered.
- **A saved child stores the steered rows.** When a steered branch saves a
  child snapshot, its H18 suffix rows are the steered ones, so the child's
  H18 is exactly what its upper K/V and H30 were computed from. The child
  is self-consistent; only its lineage back through the parent is
  unsteered.
- **Service promotion is never steered.** `SnapshotService` memoizes
  promotion, and the memoized final snapshot must stay the canonical upper
  continuation of its 18 parent. The service therefore always promotes
  with `steering: null`; the worker-level steered `promote` exists for
  experiments that manage their own snapshot ids and never feed the
  memo. Public steering applies per branch instead.
- **Handshake-gated.** The worker hello gains `steering_mode` (true only
  for the split profile; older workers omit it and Python defaults it to
  false). The service refuses steering with `capability_unavailable` when
  the profile is `full-v1` or the worker lacks steering mode, so a spec is
  never silently ignored by a worker that does not understand it.
  `/v2/models` reports the capability as `h18_steering`.
- **Fail-closed validation on both sides.** Wrong width, nonfinite alpha
  or vector values, an unknown scope and an all-zero direction (an
  extraction bug, never a meaningful steer) are refused before inference
  in Python and again in the worker. An alpha that overflows f32 when
  added is refused at apply time (`unridden::apply_steering` computes on a
  copy and commits only finite results) rather than propagated into the
  upper blocks.
- **Extraction lives caller-side.** `unridden/api/snapshots/steering.py`
  decodes `h18` rows from the snapshot `vectors` endpoint and builds
  contrastive directions `mean(focused) - mean(neutral)`, unnormalized so
  `alpha` carries the experiment's scale. It also mirrors the worker's
  steering arithmetic (including f32 rounding) for offline checks.

## Consequences

- A caller can test whether a mid-depth direction beats prompt
  instructions on the guardrail and framing rubrics, at the cost of one
  vector add per branch. `scripts/unridden/steer_guardrail.py` runs that
  comparison against a live split-profile service; it needs explicit
  `--allow-gpu` confirmation and has not been run as part of this change.
  No accuracy claim is made here.
- Steered answers are ordinary non-generative readouts: zero generated
  tokens, same answer schemas. A steered request and an unsteered request
  to the same snapshot can legitimately disagree; that is the intervention
  working, and it is why steering is opt-in per request rather than a
  worker setting.
- `full-v1` cannot steer (no host-side boundary exists), which keeps
  steering a reason to run `split18-30-v1` on 26B-A4B rather than a
  property of snapshots in general.
