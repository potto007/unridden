# Review of Intermediate Thinking Steps & Proposals to Advance Prior Efforts

Date: 2026-10-13

## 1. Review of ADRs & Previous Attempts Around Intermediate Thinking Steps

Across `unridden` ([ADRs 0001–0010](../decisions/0001-full-depth-label-readout-in-an-owned-child.md)) and the precursor experiments in `local-ai`, work on intermediate thinking steps evolved through four distinct stages:

### A. Mid-Depth Early-Exit Cascade (Block 18 Linear Probe + Full Fallback)

- **Records:** [ADR 0001](../decisions/0001-full-depth-label-readout-in-an-owned-child.md), [whitepaper.md](../whitepaper.md), and the `local-ai` research reports (`docs/research/activation-probe-gemma4-results.md`, `docs/research/activation-probe-early-exit-results.md`, `docs/research/activation-probe-poc-results.md`).
- **What worked:** Stopping Gemma 4 26B-A4B after block 18 of 30 and reading a trained 3-class linear head cut median latency from **27.98 ms to 17.01 ms** (1.64x speedup) at 114/120 accuracy on BANKING77 with zero generated tokens.
- **Where it failed or stalled:**
  1. **Task-locked head:** The linear probe mapped `H18` (`d=2816`) to three fixed banking intents. [ADR 0001](../decisions/0001-full-depth-label-readout-in-an-owned-child.md) rejected applying it to arbitrary `Choice`/`Score`/`Noul` questions. Consequently, [`_require_readout`](../../unridden/api/snapshots/service.py) and [`worker::evaluate`](../../unridden/api/native/snapshots/snapshot-worker.cpp) still hard-reject `readout_blocks: 18` with `capability_unavailable`.
  2. **Zero accuracy gain from fallback + tail-latency penalty:** In the PoC cascade (`activation-probe-poc-results.md`), the 8/120 cases below the `0.71` confidence threshold triggered a full 30-layer re-prefill in a second context (~44.6 ms, pushing p95 from 32.4 ms to 43.6 ms), yet **0 of the 8 fallback cases changed their early class choice**.
  3. **Held-out distribution drift & CUDA fusion divergence:** The block-18 probe passed validation KL (`0.0146 <= 0.05`) but failed held-out test KL against the full teacher (`0.0860 > 0.05`), and truncating the graph at layer 17 altered CUDA kernel fusion (`max rel L2 = 0.0081 > 0.001`).

### B. Split-Depth Computation Snapshots (`split18-30-v1`)

- **Records:** [ADR 0006](../decisions/0006-split-execution-for-state-snapshots.md), [2026-09-22-state-snapshots-design.md](../superpowers/specs/2026-09-22-state-snapshots-design.md), [2026-09-22-state-snapshots-protocol.md](../superpowers/specs/2026-09-22-state-snapshots-protocol.md), [gemma4-layer-range.patch](../../unridden/api/native/patches/gemma4-layer-range.patch), [state-snapshots-qualification.md](../results/state-snapshots-qualification.md).
- **What worked:** Running two contexts (`lower` blocks 0–17, `upper` blocks 18–29) over one model load in [`snapshot-worker.cpp`](../../unridden/api/native/snapshots/snapshot-worker.cpp), capturing all-position `H18[P]` and `H30[P]` matrices (`worker::create`), and promoting `S18 -> S30` (`worker::promote`) without repeating blocks 1–18. Adding `LLAMA_STATE_SEQ_FLAGS_KEEP_SWA_MASKED` in [gemma4-layer-range.patch](../../unridden/api/native/patches/gemma4-layer-range.patch) solved sliding-window restore drift, making resident, host, disk, and `S18 -> S30` promoted readouts bit-identical.
- **Where it failed or stalled:**
  1. **Gate 5 (`stock vs split`) failed `0/38` on GPU:** Although CPU split vs. stock is bit-identical at every row of `H18` and `H30`, and GPU label-probability deltas are tiny (`max 1.8e-7`), GPU `H30` residuals differed by `1.2e-2` median relative L2 against the predeclared `1e-3` gate ([state-snapshots-qualification.md](../results/state-snapshots-qualification.md)) because splitting the graph changes GPU kernel selection.
  2. **Incompatibility with Gemma 4 E4B and E2B ([ADR 0009](../decisions/0009-full-depth-snapshot-profile.md)):** 18 of E4B's 42 blocks share KV from earlier blocks (`n_layer_kv_from_start`), and every block reads `per_layer_tok_embd` side inputs. [gemma4-layer-range.patch](../../unridden/api/native/patches/gemma4-layer-range.patch) refuses partial layer ranges on such models, forcing `full-v1` (checkpoint 42 only, no intermediate checkpoint or promotion).
  3. **Orphaned `S18` capability:** Neither the application gateway ([ADR 0010](../decisions/0010-explicit-context-harness.md)) nor [`play_tetris.py`](../../scripts/unridden/play_tetris.py) ever requests checkpoint `18`, and the spec's suffix-retention optimization (reusing `H18[Q]` when an early readout falls back to block 30, [2026-09-22-state-snapshots-design.md](../superpowers/specs/2026-09-22-state-snapshots-design.md)) was never built.

### C. Activation Steering & Latent Conditioning at `H18` / `H30`

- **Records:** `local-ai` `docs/research/intermediate-state-control-assessment.md`, [2026-09-22-state-snapshots-design.md](../superpowers/specs/2026-09-22-state-snapshots-design.md).
- **Where it stalled:**
  1. **Mid-depth activation steering (`h_modified = h + alpha * v`):** Assessed conceptually for focusing downstream layers (e.g., refund temporal status), noting `llama_set_adapter_cvec` in `llama.cpp`, but never wired into the worker or tested against prompt baselines.
  2. **Latent vector conditioning (`usage_mode: "latent_conditioning"`):** Specified as a learned resampler + projection over `H18`/`H30` without KV cache restoration, but remained unbuilt.

### D. Explicit Intermediate Deliberation Steps (`state-evaluations`, `Rider`, Tetris Lookahead, & Model Escalation)

- **Records:** [ADR 0007](../decisions/0007-generative-output-from-snapshots.md), [ADR 0008](../decisions/0008-name-the-generative-mode-rider.md), [tetris-demo.md](../results/tetris-demo.md), [tetris-information-correction.md](../results/tetris-information-correction.md), [gemma4-e4b-vs-26b-usecases.md](../results/gemma4-e4b-vs-26b-usecases.md).
- **Where it failed or stalled:**
  1. **Prompt-only intermediate turns (`POST /v2/state-evaluations`):** [`SnapshotService.state_eval`](../../unridden/api/snapshots/service.py) lets a caller append a "focus/deliberation" prompt and save a child snapshot, and [`compile_readout_followup`](../../unridden/api/snapshots/compiler.py) chains multi-turn readouts via an empty `Answer:` assistant stub. However, no experiment tested whether prefilling a zero-token deliberation prompt improves accuracy on the 7 suites, and in Tetris ([tetris-demo.md](../results/tetris-demo.md)) adding strategy instructions degraded play from 286–300 pieces to 48–143 pieces.
  2. **2-Move Lookahead under real-time constraints:** Once the premature preview clock bug was fixed ([tetris-information-correction.md](../results/tetris-information-correction.md)), 2-move lookahead at Level 29 collapsed from 300 pieces to **39 and 64 pieces** because planning could no longer start until spawn and took ~102 ms/move vs. ~47 ms/move for single-move choice.
  3. **Confidence-gated escalation (`E4B -> 26B`):** In [gemma4-e4b-vs-26b-usecases.md](../results/gemma4-e4b-vs-26b-usecases.md) and `docs/research/harness-model-escalation.md` (on branch `docs/research-social-notes`), 71.2% of E4B's 156 errors had `max_probability_v1 >= 0.8`, and naive confidence thresholds reversed 13–29 answers that E4B originally got right.

---

## 2. Proposed Areas to Advance Previously Failed Efforts

### Proposal 1: Question-Conditioned Block-18 Readout via a Tuned Projection onto the Existing Vocabulary Head (plus Zero-Recompute Cascade)

- **Fixes:** The "one early-exit head per task" dead-end ([ADR 0001](../decisions/0001-full-depth-label-readout-in-an-owned-child.md)) and the 45 ms double-prefill fallback penalty (`activation-probe-poc-results.md`).
- **Key insight:** When a question suffix `Q` runs through blocks 0–17 in [`snapshot-worker.cpp`](../../unridden/api/native/snapshots/snapshot-worker.cpp), the option definitions (`A: ...`, `B: ...`) have *already* been attended to by the final token position of `H18[Q]`. You do not need a fixed-class head; you only need to translate the block-18 residual space into the block-30 pre-head space (a **Tuned Lens** affine map $W \in \mathbb{R}^{2816 \times 2816} + b$, or low-rank $I + UV^T$, followed by the model's own final RMSNorm and `A..Z` vocabulary rows).
- **How it works in `split18-30-v1`:**
  1. Fit $W, b$ via KL distillation against the full model's `A..Z` logit distribution across diverse synthetic choice/score/noul prompts (strictly disjoint from the 7 frozen evaluation suites).
  2. In [`worker::evaluate`](../../unridden/api/native/snapshots/snapshot-worker.cpp), run suffix `Q` through `lower` (blocks 0–17) to get `H18[Q]`. Apply the tuned lens to read `A..Z` logits at block 18.
  3. **Zero-recompute fallback:** If the block-18 readout fails the margin/entropy gate, do **not** rerun blocks 0–17. Feed the already-captured `H18[Q]` directly into `runtime_.run_upper` (promoting an `S18` parent once if needed). Fallback then costs *only* blocks 18–29 on the suffix (~10–12 ms instead of 45 ms).

### Proposal 2: Two-Stage Structured "Micro-Deliberation" Over Snapshots (Zero-Generation Decomposition vs. Bounded Micro-Rider)

- **Fixes:** E4B's −12.4 pt drop on candidate/date extraction and −9.4 pt drop on passage search ([gemma4-e4b-vs-26b-usecases.md](../results/gemma4-e4b-vs-26b-usecases.md)), and the unused `state-evaluations` / `readout` follow-up machinery.
- **Key insight:** A single forward pass struggles when answering a question requires resolving an intermediate variable first (e.g., *which date is the anchor date?* -> *is candidate span X before or after it?*), whereas open-ended chain-of-thought violates Unridden's latency and auditability contract.
- **Two testable variants using existing `/v2` primitives:**
  1. **Zero-token discrete decomposition (`generated_tokens: 0`):** Branch 2–3 fast auxiliary `Choice`/`Noul` sub-questions off the context snapshot (or in one batched decode), format their discrete argmax/probabilities into a deterministic structured summary, append that summary via [`SnapshotService.state_eval`](../../unridden/api/snapshots/service.py) (`save_result_snapshot: true`), and evaluate the target question from that child snapshot.
  2. **Bounded "Micro-Rider" thought prefix (`max_tokens: 8..16`):** Add an opt-in `/v2` mode that runs [`worker::ride`](../../unridden/api/native/snapshots/snapshot-worker.cpp) for at most $K \in \{8, 16\}$ greedy tokens inside `<|channel>thought\n...<channel|>`, then closes the thought channel, appends `Answer:\n`, and reads the exact `A..Z` label distribution. At ~190 tok/s, 8 tokens cost ~42 ms—still much faster than full generative output, while testing whether a tiny latent/token scratchpad closes E4B's extraction gap.

### Proposal 3: Direct Residual Steering at the `H18` Handoff in `snapshot-worker.cpp`

- **Fixes:** The unimplemented activation-steering experiment (`intermediate-state-control-assessment.md`).
- **Key insight:** Wiring `llama_set_adapter_cvec` looked invasive when `llama.cpp` applied control vectors across all positions inside the graph. However, `split18-30-v1` subsequently externalized the split boundary: `worker::create`, `worker::promote`, and `worker::run_branch` in [`snapshot-worker.cpp`](../../unridden/api/native/snapshots/snapshot-worker.cpp) already hold `H18` as an explicit host-side `matrix` (`std::vector<float>`) before passing it to `runtime_.run_upper`.
- **How to advance:**
  1. Add an optional steering parameter to `promote` / `evaluate` / `state_eval` in [`snapshot-worker.cpp`](../../unridden/api/native/snapshots/snapshot-worker.cpp) that applies $H_{18}[i] \leftarrow H_{18}[i] + \alpha v$ (either all-position or last-position-only) right before `runtime_.run_upper`, with **zero changes to `llama.cpp`**.
  2. Use the existing `/v2/snapshots/{id}/vectors?which=h18` endpoint to extract contrastive directions $v = \mathbb{E}[H_{18}^{\text{focused}}] - \mathbb{E}[H_{18}^{\text{neutral}}]$ on the guardrail/framing rubrics (where [whitepaper.md](../whitepaper.md) noted the highest error concentration) and test whether mid-depth steering outperforms prompt instructions.

### Proposal 4: Unlock Split-Depth Checkpoints on Gemma 4 E4B by Forwarding Token IDs for `inp_per_layer` and Splitting After Block 24

- **Fixes:** [ADR 0009](../decisions/0009-full-depth-snapshot-profile.md)'s restriction of E4B (`full-v1`) to full-depth-only snapshots.
- **Key insight:** Look at why [gemma4-layer-range.patch](../../unridden/api/native/patches/gemma4-layer-range.patch) rejected E4B:
  1. **Shared-KV layers (`n_layer_kv_from_start`):** In Gemma 4 E4B (42 blocks total, 18 shared-KV blocks), the shared-KV layers are all in the *upper* stack (`il >= 24`), reusing K/V from the last 2 full/sliding layers below block 24. If E4B is split at **block 24** (`split24-42-v1`), `lower` (0–23) owns 100% of the KV buffers that `upper` (24–41) references.
  2. **Per-layer token inputs (`per_layer_tok_embd`):** In `src/models/gemma4.cpp` ([gemma4-layer-range.patch](../../unridden/api/native/patches/gemma4-layer-range.patch)), `inp_per_layer` is computed purely from the input token IDs (`tok_embd` + `per_layer_tok_embd`), *not* from lower-layer hidden states. Passing the token IDs alongside `H24` into `upper` lets `gemma4.cpp` compute `inp_per_layer` for layers `24..41` while reading the residual stream from `embd`.

### Proposal 5: Calibrated Margin/Entropy + Permutation-Stability Routing for Escalation (`E4B -> 26B` & `Block 18 -> Block 30`)

- **Fixes:** High-confidence errors and 26B reversals in `docs/research/harness-model-escalation.md` and [gemma4-e4b-vs-26b-usecases.md](../results/gemma4-e4b-vs-26b-usecases.md).
- **How to advance:**
  1. Implement Survey Items 1, 2, and 4 ([open-model-survey.md](open-model-survey.md)): expose **top-1/top-2 logit margin**, **normalized entropy** ($1 - H(p)/\log K$), and per-(question-type, option-count) temperature scaling in [`mapping.py`](../../unridden/api/mapping.py).
  2. **Two-permutation consistency gate on warm snapshots:** Because a warm branch on E4B takes only ~15–25 ms, when a question's logit margin falls in an intermediate band, run a second branch on the *same* warm E4B snapshot with the option order reversed (Survey Item 3). Confidently wrong answers on small models frequently exhibit high option-order/letter sensitivity; escalating to 26B *only* when E4B shows low margin **or** flips under option permutation will sharply cut false escalations and 26B reversals.
  3. **Speculative pipelining for real-time Tetris lookahead:** To fix the Level-29 lookahead collapse ([tetris-information-correction.md](../results/tetris-information-correction.md)), compute the **1-move choice during the entry delay** (when only the current piece is legal to know) and commit its initial shift direction at frame 0, then use the warm board snapshot at spawn to run a fast **top-2 plan refinement** once the next preview piece becomes visible.

---

## 3. Summary Priority Matrix

| Priority | Effort | Target Failure Addressed | Concrete First Step |
| :--- | :--- | :--- | :--- |
| **1. Calibrated Margin + Permutation Gate** | Small | E4B $\to$ 26B escalation reversals & saturated `max_probability_v1` | Add `logit_margin` & `normalized_entropy` to [`mapping.py`](../../unridden/api/mapping.py) and evaluate paired E4B/26B rescue-vs-reversal curves offline. |
| **2. Tuned-Lens Block-18 Readout + Suffix Handoff** | Medium | Task-locked early head ([ADR 0001](../decisions/0001-full-depth-label-readout-in-an-owned-child.md)) & 45 ms double-pass fallback | Train a $2816 \times 2816$ linear map from `H18` to pre-norm `H30` on `split18-30-v1` and wire `H18[Q]` reuse into [`worker::evaluate`](../../unridden/api/native/snapshots/snapshot-worker.cpp). |
| **3. Host-Side `H18` Steering in `snapshot-worker.cpp`** | Small–Med | Stalled activation-steering pilot (`intermediate-state-control-assessment.md`) | Add `H18 += alpha * v` before `runtime_.run_upper` in [`snapshot-worker.cpp`](../../unridden/api/native/snapshots/snapshot-worker.cpp) and test on the guardrail suite. |
| **4. Two-Stage Snapshot Decomposition / Micro-Rider** | Medium | E4B multi-step date/span extraction gap (−12.4 pts) | Benchmark 2-step `decisions -> state_eval -> decisions` vs. 8-token `Rider` scratchpad on `candidate-span-and-date-extraction.json`. |
| **5. Gate 5 Re-qualification Protocol** | Small | `split18-30-v1` stuck in experimental due to GPU `1e-3` L2 noise | Update [`qualify_snapshots.py`](../../scripts/unridden/qualify_snapshots.py) to gate on CPU bit-identity + GPU decision/probability parity + GPU residual drift relative to stock GPU-vs-CPU noise. |
