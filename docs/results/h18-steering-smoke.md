# H18 steering GPU smoke test

Date: 2026-10-13
Branch: `feat/h18-steering` (worktree `unridden-h18-steering`), ADR 0011.

## Setup

- Bundle built from the worktree against `llama-base-v0.4.1-split2`
  (`python -m unridden.api.native.snapshots.build`, output
  `build/snapshot-worker-steering`).
- Service: worktree API, snapshots-only, split18-30-v1, GPU, port 18091,
  model `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`. GPU before launch 1973 MiB
  used (bge-m3 embed only); with the worker loaded 19475 MiB, so roughly
  13 GiB stayed free on the 32 GiB card. After shutdown the card returned
  to 1961 MiB.
- Script: `/tmp/steering_smoke.py` (throwaway, not committed). Direction
  extracted live from one guardrail case through the `vectors` endpoint:
  focused vs neutral prompt, shared `Assessment:` tail, last H18 row.
  Direction width 2816, L2 norm 44.51. Questions: `hz_injection` (Noul)
  and `route` (Choice) from the same case.

## Results, 19/19 PASS

- `/v2/models` advertises `h18_steering: true` under `split18-30-v1`.
- Unsteered baseline: `hz_injection` noul 1.71e-07, `route` = pass.
- Steering moves probabilities monotonically with alpha (max probability
  delta vs unsteered): alpha 1 -> 5.07e-04, alpha 10 -> 6.88e-01,
  alpha 50 -> 9.66e-01. `scope: last` at alpha 10 -> 1.17e-01.
  `output_tokens` stayed 0 in every steered response.
- A/B/A: the unsteered repeat after all steered branches is bit-identical
  to the first unsteered answers, and the parent 18 snapshot's stored last
  H18 row is byte-identical before and after.
- Steered `state_eval` changes the exported final residual (L2 65.83 vs
  unsteered), saves a usable child (decisions via `replace_question`
  work), and a steered decision from an 18 parent promotes first
  (`promotion: performed`) then steers the suffix.
- Refusals hold live: wrong-width vector -> 422, all-zero vector -> 422,
  and the service answers bit-identically afterwards.
- Regression: `scripts/unridden/validate_api_v2.py` against the same
  service reports `{"failed": 0}`.

## What this does not show

This is a mechanism smoke test on one case with a one-pair direction. It
proves steering is causal (changing only the steering spec changes only
the steered branch's readout) and that the snapshot invariants survive it.
It does not show steering improves accuracy; at alpha 10 this direction
pushed `hz_injection` the wrong way (noul 0.37) on a case the unsteered
model already had right. The accuracy question is the
`scripts/unridden/steer_guardrail.py` run (multi-case extraction, held-out
evaluation), still to be run.
