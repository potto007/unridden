# Two-Stage Snapshot Decomposition & Micro-Rider Benchmark (`full-v1`, Gemma 4 E4B)

Date: 2026-10-13
Branch: `feat/two-stage-snapshot-decomposition` (worktree `unridden-micro-deliberation`)
Priority Matrix Item: #4 (`docs/research/intermediate-thinking-steps-review.md`, Proposal 2)

## Setup

- **Model:** `gemma-4-E4B-it-UD-Q4_K_XL.gguf` (`sha256: 3cf61de12daa015ee0f7b68e7b7c541405bf220e1e942bad8b47cab827d7df80`)
- **Snapshot Worker:** `build/api-snapshot-worker/build/unridden-snapshot-worker` (`runtime_sha256: f1f3810e9ece5e0a62504696c1b5be1c23df840a603b3242a007036aac8f18e4`, `profile: full-v1`, `final_block: 42`)
- **Implementation:**
  - `unridden/api/snapshots/deliberation.py` (`plan_two_stage_questions`, `select_stage2_and_compile_summary`, `run_two_stage_deliberation`)
  - `unridden/tests/test_snapshots_deliberation.py`
  - `scripts/unridden/benchmark_micro_deliberation.py`
- **Target Suites (E4B vs 26B-A4B multi-step gap suites):**
  - `candidate-span-and-date-extraction.json` (33 cases, 194 questions)
  - `passage-rerank-and-line-search.json` (36 cases, 149 questions)
  - Combined: **69 cases, 343 questions**

## Evaluated Arms

1. `single_pass`: Standard single-pass `/v2/decisions` against the warm context snapshot (`output_tokens == 0`).
2. `discrete_state_eval` (`mode="discrete_two_stage"`, `continuation="state_eval_child"`):
   - **Stage 1:** Evaluates caller questions + bounded auxiliary `Choice` probes (`_aux_day_1_15` [16 options], `_aux_day_16_31` [17 options], `_aux_best_candidate` [$\le 6$ options]) in a single zero-token `/v2/decisions` call (`output_tokens == 0`).
   - **Consistency Gate:** Checks cross-question structural consistency (`mode` vs calendar/relative fields, `convention_recoverable`, literal day-of-month vs `day_tens`/`day_unit`/`components_written`, `block_type`/`line_1_type` vs `heading_level`/`callout_kind`/`ordered_steps`/`join_1_2`/`both_numbered`, and joint `Choice` passage selection vs `cand_p*`/`contains_answer`). Consistent cases short-circuit after Stage 1.
   - **Stage 2:** For flagged cases only (19 / 69 cases), appends the deterministic Stage-1 verification summary via `POST /v2/state-evaluations` (`save_result_snapshot=True`) and re-evaluates only the flagged dependent questions via `POST /v2/decisions` (`relationship="followup"`, `output_tokens == 0`).
3. `discrete_context_instr` (`mode="discrete_two_stage"`, `continuation="context_instructions"`): Same zero-token Stage-1 + consistency gate, passing the Stage-1 verification summary in the Stage-2 question instructions directly from the parent context snapshot (`output_tokens == 0`, compatible with templates whose readout turns cannot be extended).
4. `micro_rider_8` (`mode="micro_rider"`, `rider_max_tokens=8`): Bounded greedy scratchpad (`POST /v2/rider`, `max_tokens=8`) followed by `state_eval` and `POST /v2/decisions` (`decision_output_tokens == 0`).
5. `micro_rider_16` (`mode="micro_rider"`, `rider_max_tokens=16`): Bounded greedy scratchpad (`POST /v2/rider`, `max_tokens=16`) followed by `state_eval` and `POST /v2/decisions` (`decision_output_tokens == 0`).

## Overall Results (69 cases, 343 questions)

| Arm | Question Accuracy | Delta vs Single-Pass | Rescues / Reversals | All-Correct Cases | Stage-2 Triggered | Decision `output_tokens` | Rider Tokens | Median / Mean Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single_pass` | `294 / 343` (**85.71%**) | — | `+0 / -0` | `43 / 69` (62.3%) | `0 / 69` | `0` | `0` | `115.9 ms` / `130.1 ms` |
| `discrete_state_eval` | **`326 / 343` (95.04%)** | **+9.33 pts (+32)** | **`+32 / -0`** | **`55 / 69` (79.7%)** | `19 / 69` | **`0`** | **`0`** | `133.8 ms` / `195.7 ms` |
| `discrete_context_instr` | **`323 / 343` (94.17%)** | **+8.46 pts (+29)** | **`+29 / -0`** | **`55 / 69` (79.7%)** | `19 / 69` | **`0`** | **`0`** | `131.6 ms` / `184.6 ms` |
| `micro_rider_8` | `301 / 343` (87.76%) | +2.05 pts (+7) | `+14 / -7` | `42 / 69` (60.9%) | `69 / 69` | `0` | `351` | `168.5 ms` / `184.1 ms` |
| `micro_rider_16` | `304 / 343` (88.63%) | +2.92 pts (+10) | `+18 / -8` | `43 / 69` (62.3%) | `69 / 69` | `0` | `598` | `183.4 ms` / `202.4 ms` |

## Per-Suite Breakdown

### 1. `candidate-span-and-date-extraction` (33 cases, 194 questions)

| Arm | Question Accuracy | Delta vs Single-Pass | Rescues / Reversals | All-Correct Cases | Stage-2 Triggered | Median / Mean Latency |
| --- | --- | --- | --- | --- | --- | --- |
| `single_pass` | `160 / 194` (82.47%) | — | `+0 / -0` | `17 / 33` (51.5%) | `0 / 33` | `134.1 ms` / `158.0 ms` |
| `discrete_state_eval` | **`186 / 194` (95.88%)** | **+13.41 pts (+26)** | **`+26 / -0`** | **`26 / 33` (78.8%)** | `13 / 33` | `127.7 ms` / `267.7 ms` |
| `discrete_context_instr` | **`184 / 194` (94.85%)** | **+12.38 pts (+24)** | **`+24 / -0`** | **`26 / 33` (78.8%)** | `13 / 33` | `127.9 ms` / `251.8 ms` |
| `micro_rider_8` | `165 / 194` (85.05%) | +2.58 pts (+5) | `+8 / -3` | `16 / 33` (48.5%) | `33 / 33` | `204.0 ms` / `221.9 ms` |
| `micro_rider_16` | `168 / 194` (86.60%) | +4.13 pts (+8) | `+12 / -4` | `17 / 33` (51.5%) | `33 / 33` | `231.7 ms` / `256.0 ms` |

- Note that single-pass 26B-A4B scored `174 / 194` (`89.7%`) on this suite (`docs/results/gemma4-e4b-vs-26b-usecases.md`). Zero-token discrete two-stage deliberation on **E4B (`186 / 194`, `95.88%`)** not only closes the entire 14-question E4B vs 26B-A4B gap (`160 -> 174`), it surpasses single-pass 26B-A4B by **+12 questions (+6.2 percentage points)** with **zero generated tokens** and **zero reversals**.

### 2. `passage-rerank-and-line-search` (36 cases, 149 questions)

| Arm | Question Accuracy | Delta vs Single-Pass | Rescues / Reversals | All-Correct Cases | Stage-2 Triggered | Median / Mean Latency |
| --- | --- | --- | --- | --- | --- | --- |
| `single_pass` | `134 / 149` (89.93%) | — | `+0 / -0` | `26 / 36` (72.2%) | `0 / 36` | `111.0 ms` / `105.9 ms` |
| `discrete_state_eval` | **`140 / 149` (93.96%)** | **+4.03 pts (+6)** | **`+6 / -0`** | **`29 / 36` (80.6%)** | `6 / 36` | `134.9 ms` / `129.0 ms` |
| `discrete_context_instr` | **`139 / 149` (93.29%)** | **+3.36 pts (+5)** | **`+5 / -0`** | **`29 / 36` (80.6%)** | `6 / 36` | `133.3 ms` / `124.8 ms` |
| `micro_rider_8` | `136 / 149` (91.28%) | +1.35 pts (+2) | `+6 / -4` | `26 / 36` (72.2%) | `36 / 36` | `153.1 ms` / `151.4 ms` |
| `micro_rider_16` | `136 / 149` (91.28%) | +1.35 pts (+2) | `+6 / -4` | `26 / 36` (72.2%) | `36 / 36` | `156.4 ms` / `154.3 ms` |

## Key Technical Findings

1. **Zero-token discrete decomposition strictly dominates free-form Micro-Rider:**
   - `discrete_state_eval` achieves **+32 rescues and 0 reversals** (`95.04%`, `output_tokens == 0`, `rider_generated_tokens == 0`) while triggering Stage 2 on only `19 / 69` cases (`27.5%`).
   - Free-form `micro_rider_8` (`+14 / -7`, net `+7`) and `micro_rider_16` (`+18 / -8`, net `+10`) help on arithmetic/date extraction when the 8–16 token scratchpad quotes the right span, but suffer from 7–8 reversals when an unconstrained 8–16 token continuation truncates mid-thought or latches onto a distractor line.
2. **Why `state_eval` child continuation works natively on Gemma 4 E4B (`full-v1`) vs 26B-A4B (`split18-30-v1`):**
   - E4B's chat template emits `<|turn>model\n` at `add_generation_prompt`; closing the `state_eval` turn (`assistant: ""`) produces `<|turn>model\n<turn|>\n<|turn>user\n...`, which is an exact token-prefix extension of the `state_eval` snapshot (`200 OK`).
   - 26B-A4B's chat template appends `<|channel>thought\n<channel|>` on the active generation prompt (`enable_thinking=false`) but strips empty thought channels from prior turns before `last_user_idx`, causing `branch_suffix` to raise `409 followup_unsupported`.
   - `run_two_stage_deliberation` supports both `state_eval_child` (`95.04%`) and `context_instructions` (`94.17%`), and automatically falls back to `context_instructions` if a backend raises `FollowupUnsupported`.
