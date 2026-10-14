# Unridden Agent Rules & Project Invariants

Unridden is a local, non-generative structured decision API backed by owned `llama.cpp` native workers (`unridden-api-worker` and `unridden-snapshot-worker`). Follow these architectural, wire, and operational invariants whenever modifying or evaluating this repository.

## 1. Non-Generative Decision Contract (`/v1` and `/v2`)

- **Zero Generated Tokens:** Standard decision routes (`/v1/decisions`, `/v1/systemone`, `/v2/decisions`, `/v2/state-evaluations`, and gateway `/v1/decisions`) must never sample, append, or decode autoregressive tokens (`output_tokens == 0`, `generated_tokens == 0`, `callbacks_enabled == False`). The only route permitted to generate tokens is the explicit `/v2/rider` diagnostic route when `rider_mode` is enabled on the snapshot worker.
- **Three Question Types Only:**
  - `choice`: 2 to 255 options at the wire boundary, bounded to the backend's validated single-token label capacity (`A..Z`, up to 26 options). Returns the argmax option key, full option probability map, and `confidence`.
  - `score`: 2 to 10 ordered rubric criteria. Returns the literal zero-based expected value `sum(i * p_i)` (never rescaled to `0..100`), a `legend` mapping `"0"..str(k-1)` to criteria, `probabilities`, and `confidence`.
  - `noul`: Binary `true`/`false` proposition (`A: true`, `B: false`). Returns `noul: P(true)` only. Do not add a `confidence` field to `NoulAnswer`.
- **Strict Wire Validation:** All Pydantic request, response, and native worker models in `unridden/api/schema.py`, `unridden/api/snapshots/schema.py`, and `unridden/harness/schema.py` must inherit from `StrictModel` (`ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)`) and reject `NaN`/`Inf` values and chat-template control tokens (`<|turn>`, `<turn|>`, `<|channel>`, `<channel|>`, `<|thought>`, `<thought|>`).
- **Sibling Isolation:** Question IDs are caller correlation metadata and must never appear in compiled prompts. In sequential mode, sibling questions share only the byte-identical state prefix (`shared_prefix_bytes`) and must produce bit-identical logits whether evaluated alone, reordered, or alongside siblings (ADR 0002). Batched mode (`--batched`, ADR 0004) must explicitly report `evaluation.mode` and any `batched_fallback` reason.

## 2. Intermediate State Snapshots (`/v2`) & Gateway (`unridden/harness`)

- **Snapshot Profiles:**
  - `split18-30-v1`: Supported on the 30-block Gemma 4 26B-A4B layout (`n_layer == 30`, `split_block == 18`, uniform KV across layers, no `inp_per_layer`). Checkpoints `[18]` (lower KV + `H18` residual) and `[30]` (upper KV + `H30` residual).
  - `full-v1` (ADR 0009): Whole-model context profile whose sole checkpoint is `n_layer` (e.g., `42` on Gemma 4 E4B, which uses shared KV layers and `per_layer_tok_embd`). Never hardcode `30` as the readout depth in `/v2` or gateway code; discover `final_block` from the worker handshake (`WorkerHello.final_block`).
- **No Unvalidated Early Readout:** Do not expose a block-18 decision readout unless a calibrated projection / Tuned Lens head passes frozen release gates. Block-18 snapshots have capabilities `["continue", "promote", "inspect"]` and must be promoted to `final_block` before readout.
- **Immutable Parent Snapshots:** Branch evaluations (`/v2/decisions`, `/v2/state-evaluations`, `/v2/rider`) must restore from the parent snapshot without mutating it, preserving exact `A, B, A` branch isolation.
- **H18 Steering (ADR 0011):** Optional per-request `steering` (`alpha`, `scope: all|last`, f32 direction of width `n_embd`) on `/v2/decisions` and `/v2/state-evaluations` adds `alpha * v` host-side to the branch's suffix H18 rows (worker `promote` accepts the same spec over a copy of the full `H18[P]`). Split profile plus worker `steering_mode` only - never send a spec to a worker that did not advertise it, and `full-v1` refuses with `capability_unavailable`. Steering copies, never mutates, the parent's H18; a saved child stores the steered rows it was computed from. Service promotion is always unsteered so the memoized promotion stays canonical; `create` and `/v2/rider` are never steered. Unsteered requests stay bit-identical.
- **Gateway Surface (`unridden/harness`):** The front gateway exposes only `/health`, `/v1/models`, `/v1/contexts`, `/v1/contexts/{id}`, and `/v1/decisions` (with optional inline `context` orchestration, ADR 0008). Never proxy raw `/v2/snapshots`, `/v2/rider`, or `/v2/state-evaluations` through the gateway.

## 3. Uncertainty, Calibration, and Escalation Gate (`unridden/api/gate.py`)

- **Preserve `confidence_formula`:** `ChoiceAnswer.confidence` and `ScoreAnswer.confidence` are pinned on the wire as `max_probability_v1` (`max(probabilities)`). Additive uncertainty signals (`logit_margin`, `probability_margin`, `normalized_entropy`) belong on `QuestionDiagnostics` and in `unridden/api/gate.py`. The served calibration (`QuestionDiagnostics.calibration` and `.gate`, ADR 0012) is diagnostics-only, present only when `--calibration-table` is loaded, and bound at startup to `model_sha256`, `runtime_sha256` and the compiler `PROMPT_VERSION`; a mismatched or provenance-less table must refuse to start. `?probe=true` may re-ask a question with reversed options but must never replace the primary answer, and its tokens count in `usage.input_tokens` with `output_tokens` still 0.
- **Temperature Calibration Discipline:**
  - Fit temperatures per `(question_type, option_count)` bucket (`fit_temperature_table`), never as a single global scalar across incompatible option cardinalities.
  - Clamp fitted temperatures into `[1.0, 5.0]` by default (with warning on clamped or sparse `< 10` sample buckets) and bind tables to `model_sha256`, `runtime_sha256`, and `prompt_version`.
  - Evaluate calibration (ECE, equal-mass ECE, Brier, gold NLL, AURC) and escalation curves using leave-one-suite-out cross-fitting (`cross_fit_calibrated_uncertainties`) so held-out claims are never circular.
- **Permutation Consistency Gate:** When probing borderline-margin `choice` or `score` questions (`permute_question`), always verify that the semantic option set and descriptions are unchanged before realigning probabilities (`realign_probabilities`) to the canonical option order.
- **Lookahead / Simulator Integrity:** Any benchmark or game controller (such as `scripts/unridden/play_tetris.py`) must only pass information to candidate-scoring prompts that is legally visible at the decision timestamp (ADR 0010).

## 4. GPU, Native Worker, and Systemd Safety

- **No Implicit GPU Allocation on Import or in Unit Tests:** Importing `unridden.*` or running `pytest` must never initialize CUDA or spawn a real model worker. All unit tests under `unridden/tests/` must run on CPU using deterministic fake backends.
- **Explicit `--allow-gpu` / `--gpu` Guard:** Benchmark and validation scripts under `scripts/unridden/` require explicit confirmation and `--allow-gpu` (or `--gpu`). Check running `unridden-*.service` user systemd units and GPU memory (`nvidia-smi`) before starting a worker process; E4B (~5.1 GiB) and 26B-A4B (~18+ GiB) share the local 32 GiB GPU and must not be launched blindly in parallel.
- **Native Worker Logs:** `llama.cpp` worker stderr is routed to `/tmp/llama-server.log` (or the configured `native-worker.log`). Do not write project code or persistent artifacts to `/tmp`.

## 5. Verification Commands

Run all checks from the repository root using `.venv` (or `uv run`) before completing any code change:

```bash
# 1. Unit and contract tests (CPU-only, fast)
.venv/bin/python -m pytest -q

# 2. Ruff lint and format checks (line-length = 88, py312)
.venv/bin/ruff check unridden scripts/unridden
.venv/bin/ruff format --check unridden scripts/unridden

# 3. Strict static type checking
.venv/bin/python -m mypy unridden scripts/unridden
```
