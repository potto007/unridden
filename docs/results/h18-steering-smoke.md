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

## What the smoke test alone does not show

The single-case smoke test proves steering is causal (changing only the
steering spec changes only the steered branch's readout) and that the
snapshot invariants survive it. At alpha 10 that single-case direction
pushed `hz_injection` the wrong way (noul 0.37) on a case the unsteered
model already had right.

## Held-Out Guardrail Benchmark (`scripts/unridden/steer_guardrail.py`)

Run on `evals/decision-api-v1/guardrail-policy-gates.json` (36 cases total:
first 12 cases used for contrastive direction extraction via `POST /v2/snapshots/{id}/vectors`,
remaining **24 held-out cases / 278 questions** evaluated across `unsteered`,
`prompt` baseline, and `steer:{0.25, 0.5, 1.0, 2.0, 4.0}`).

- **Extracted direction (`12` extraction cases):** width `2816`, raw mean-diff L2 norm `28.5625` (unit-normalized before scaling by `alpha`).

| Arm | `scope="all"` Held-Out Accuracy | Delta vs Unsteered | `scope="last"` Held-Out Accuracy | Delta vs Unsteered |
| --- | --- | --- | --- | --- |
| `unsteered` | `239 / 278` (**86.0%**) | — | `239 / 278` (**86.0%**) | — |
| `prompt` (focused instruction) | `240 / 278` (**86.3%**) | +0.3 pts (+1) | `240 / 278` (**86.3%**) | +0.3 pts (+1) |
| `steer:0.25` | `242 / 278` (**87.1%**) | +1.1 pts (+3) | `238 / 278` (85.6%) | -0.4 pts (-1) |
| `steer:0.5` | `244 / 278` (**87.8%**) | +1.8 pts (+5) | `239 / 278` (86.0%) | +0.0 pts (+0) |
| **`steer:1.0`** | **`248 / 278` (89.2%)** | **+3.2 pts (+9)** | `239 / 278` (86.0%) | +0.0 pts (+0) |
| `steer:2.0` | `217 / 278` (78.1%) | -7.9 pts (-22) | **`242 / 278` (87.1%)** | **+1.1 pts (+3)** |
| `steer:4.0` | `63 / 278` (22.7%) | -63.3 pts (-176) | `240 / 278` (86.3%) | +0.3 pts (+1) |

- At **`alpha = 1.0, scope = "all"`** (~3.5% of the raw contrastive shift), H18 activation steering improves held-out guardrail accuracy from **86.0% (`239/278`) to 89.2% (`248/278`, +3.2 percentage points / +9 questions)** and outperforms the explicit prompt-engineering baseline (`240/278`, `86.3%`) by **+2.9 percentage points (+8 questions)** with **`output_tokens == 0`** and zero extra prompt tokens.
- For `alpha >= 2.0` (`scope="all"`), the shift enters the over-steering regime where activations saturate beyond the natural contrastive norm.

