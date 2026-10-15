# Harness versus demo ownership audit

Follow-up: the [Tetris information-timing correction](../results/tetris-information-correction.md)
changes controller scheduling on all API modes. The preservation statements
below describe the gateway migration before that separate correctness fix.
Gameplay rules and observations remain the application's responsibility.

Date: September 27, 2026. Scope: actual source and runnable clients in this
checkout, its two local worktrees, and local output artifacts.

## Finding and integration

The Tetris controller existed in the clean `tetris-example` and
`full-depth-snapshots` worktrees. Their controller copies were identical.
The main checkout initially lacked the files because it was behind the already
merged [Tetris demo and comparison change](https://github.com/potto007/unridden/pull/36).

After fetching, the upstream difference was verified as six files: controller,
replay, tests, two result reports and changelog. The checkout was fast-forwarded
to that merge without overwriting existing harness or unrelated edits. A
recoverable tracked patch, untracked archive and integration state were saved
under `outputs/harness-check/demo-integration-before/` before the subsequent
demo edits. The original worktrees remain untouched.

The normal Tetris path now calls the gateway with state and questions and
retains a private cookie jar. Native v1/v2 modes remain explicit comparisons.
No game policy was moved into generic inference.

## Source inventory and ownership map

| Source / behavior | Evidence in code | Owner and action |
| --- | --- | --- |
| `scripts/unridden/play_tetris.py:Unridden.final_block/decide/flush/close` | Native v2 discovers readout depth, captures memory parents, keeps a state-to-ID map, branches, tracks pending deletion and releases parents. | **Gateway for normal use; migrated.** Those native mechanisms remain only for explicit v2 comparison. The default sends state/questions, never raw snapshot IDs. |
| `Unridden._req` | HTTP errors, up to twenty busy attempts spaced by 250 ms, per-exchange timing and trace data. | **Generic mechanics in the gateway; migrated.** The normal client makes one attempt; the gateway owns bounded admission and recognized-busy retry. Client cookie transport and application deadlines remain. Explicit native comparison modes retain their historical retry loop. |
| `knockout` | Groups at most 26 alternatives, selects winners, asks a final; halves group size on context budget error. | **Demo decision policy.** Tournament composition changes which alternatives the model compares and can change the winner. It is not a transparent gateway optimization. Gateway preserves a sanitized typed budget error so this existing behavior still works. |
| `Unridden.decide` question chunking | Splits Score or other question maps into batches of 32 and combines answers. | **Gateway; migrated.** One front call accepts up to 64 independent questions and discovers native batch size. It returns a complete keyed bundle or an error, sums work/timing and never publishes partial success or replays completed batches. Native comparison modes retain explicit batching. |
| `RULES/STRATEGY/state_text/strategy_state/choose_choice/choose_score` | Domain rules, board text, scoring rubric, compact descriptions and static rules versus changing board placement in the question. | **Application.** The developer defines stable state and the question. Gateway captures exactly that boundary and does not rewrite or infer Tetris semantics. All prompt text and selection behavior are preserved. |
| `prune/strategy_filter/two_move_plans/choose/choose_plan` | Legal-placement pruning, hand-coded rules, two-move lookahead, model versus heuristic/random agents. | **Game controller.** Keep out of reusable gateway policy; these materially contribute to game performance. |
| `placements/features_of/realtime_landing/shift_frames/bag_stream/run_game` | Game physics, deterministic pieces, movement schedules, score and simulated deadlines. | **Demo.** Unchanged. A faster/slower API can still alter real-time outcomes; turn-based parity is not real-time parity. |
| `run_game/move_record` | Measures the entire controller move, client decision calls and raw exchanges; previously names elapsed client work `model_s`. | **Both, distinct measurements.** Gateway reports capture/reuse/HTTP/inference diagnostics. Demo retains complete move timing, adds `decision_client_s` and API provenance, and preserves the old field for trace compatibility. |
| `scripts/unridden/tetris_replay.html` | Fetches local JSONL replay files; renders boards, candidate probabilities and HTTP exchanges. It never calls the model. | **Demo.** Updated to recognize automatic capture/reuse and display ordinary request state; existing native trace rendering remains. |
| `unridden/examples/hello.json`, `api-v1-cases.json`, seven use-case JSON files | Data payloads and frozen cases, not controller/service code. | **Examples/evaluation inputs.** Existing state/question shapes work with the gateway; no snapshot bookkeeping belongs in these files. Added the gateway integration path to README and guide. |
| `unridden/api/cli.py:run/evaluate_requests` | Offline ordinary evaluation owns a native backend and writes response JSONL. | **Native offline tool.** Do not silently replace its inference regime or model ownership. The independent gateway launcher is documented for applications. |
| `unridden/api/semif.py` | Pure JSONL format conversion, no inference or HTTP client. | **Interchange utility.** No lifecycle to migrate. |
| `scripts/unridden/run_usecase_suites.py:run` | Freezes case hashes, starts a pinned native ordinary worker, times `ApiService.evaluate`, scores gold labels and records native/GPU data. | **Evaluation driver.** Keep existing baseline intact. A separately named gateway mode could reuse frozen scoring, but must record HTTP/capture/hit rates and a different timing boundary. No seven-suite gateway run was implied or performed. |
| `validate_api_v1.py/validate_api_v2.py/validate_api_failures.py` | Assert native route semantics, failures, promotion, child snapshots, vectors and restart behavior. | **Backend qualification.** Direct access is intentional; routing these through an allowlisted gateway would erase what they test. New gateway validators cover the application contract separately. |
| `qualify_snapshots.py/bench_snapshots.py` | Direct native worker protocol, checkpoint/restore/reference gates and persistence/cost measurements. | **Native qualification.** Preserve direct IPC and explicit model control. The gateway should reuse this machinery, not contain benchmark/reference code. |
| Competitor runners and comparison scripts | External competitor adapters, frozen outcome scoring, calibration and aggregate reports. | **Research tooling.** Do not route competitor APIs through this Unridden gateway or move gold-label policy into live inference. |

No other runnable application controller or model-facing browser demo was found
in the inspected repository/worktree source and output artifacts. The replay
is the browser UI; benchmark and qualification scripts are distinct from
application demos.

## What changed in this migration

1. Tetris's default `gateway` mode selects the application listener and carries
   automatic cookies using the standard library. It ignores native-only
   snapshot retention flags and uses no snapshot/discovery/delete endpoint.
2. Explicit `--api v2` and `--api v1` keep their old paths and select their
   respective default ports. Explicit URLs continue to override defaults.
3. The gateway preserves the known `budget_error` code with a sanitized message.
   It still hides arbitrary upstream diagnostics and never chooses smaller
   Tetris groups itself.
4. Generic busy handling now runs in the gateway with queue/backoff diagnostics.
   The normal demo client has no retry loop; explicit native modes retain theirs. Cleanup now executes on exceptional exit
   from the game too; automatic parents are retired by gateway TTL/capacity.
5. Trace/result provenance includes API mode, client elapsed time is named
   accurately, and replay text supports gateway responses. Historical result
   tables remain explicitly native measurements.

Normal automatic reuse can retain a board across tournament rounds where the
old non-`keep` v2 client recaptured it. This changes resource/timing behavior as
intended without changing the state or question content. Automatic expiry and
definitive lost-parent recapture are gateway responsibilities. The two native comparison modes deliberately retain duplicate low-level
mechanics so the benchmark baseline is still available. Normal gateway clients
implement neither native batching nor a busy-retry loop.

## Verification

Existing engine/controller tests still pass. Added tests verify exact
state/question preservation, batching, absence of direct snapshot calls,
real HTTP cookie propagation, and default/explicit CLI routing. Harness tests
verify the retained budget error alongside sanitized failures.

Live 26B turn-based comparisons used six moves each, same seed, for both
static-rule and changing-board modes through direct snapshots and the gateway.
Answer JSON and selected moves matched exactly. In each game pruning left only
two model requests: the static gateway scope recorded one miss and one hit,
while changing state recorded two misses. The check is intentionally small and
does not establish equivalence for all seeds, real-time behavior or ordinary v1.

Artifacts: `outputs/harness-check/tetris-gateway-parity.json` and four
`tetris-{static,changing}-{v2,gateway}.jsonl` traces. Gateway/main backend health
was independently checked from WSL and Windows. No second model was loaded.

## Prioritized remaining work

| Priority | Work | Why it is separate |
| --- | --- | --- |
| P0, done | Default demo integration, automatic lifecycle, typed budget error, honest trace timing, native comparison preservation. | Directly satisfies this first gateway design. |
| P1 | Exact backend model/runtime identity and per-route provenance. | Current alias/profile cannot detect a replaced weights build; prerequisite for reproducible routing or future escalation. |
| P1 | Explicit gateway mode in the frozen suite runner, followed by held-out snapshot quality and full latency qualification. | Changes inference and timing regime; keep native baseline and label scopes separate. |
| P0, done | Bounded admission, cancellation, recognized busy retry and whole-operation deadline. | Generic orchestration is centralized; native comparison modes retain old mechanics. Production load qualification is still separate. |
| P2 | Optional SDK for cookie scope and structured errors; optional answer-streaming contract if useful. | Batching and short busy handling are already in the gateway. Streaming would add partial-result and cancellation semantics, so complete JSON remains the default. |
| P2 | Explicit generic ranking/tournament helper only if multiple apps need it. | Grouping affects semantics and quality; never silently transform a Choice into a tournament. |
| P2 | Longer seeded gateway/native demo and real-time comparisons. | Required before transferring historical gameplay/speed claims to the new API path. |

## Generic orchestration follow-up

The user's clarification moved independent-question batching and brief busy
handling into the gateway, rather than leaving those algorithms to clients.
The API now accepts up to 64 questions, uses the advertised native limit, keeps
one immutable parent and returns one complete answer bundle. Queue capacity,
wait budget, backoff and operation timeout are explicit and finite. Tests cover
overflow, cancellation, later-batch failure, no replay, and unchanged Choice
semantics. Live verification compared a 34-question call with two direct
native batches on the same parent and obtained identical answers.

The controller still decides what a Tetris move means and whether to compare
groups of candidate moves. That tournament originates in the demo and is not
generic snapshot/transport policy. It was deliberately not moved into the
harness. This distinction follows the user's confirmed boundary.

The [escalation investigation](harness-model-escalation.md) remains future work.
Its model switching, calibration and additional runtime ownership are not mixed
into this lifecycle migration.


Final live caller check: the actual Tetris Score client submitted all 34 legal
L-piece placement questions in one application call. The gateway used two
native batches, returned identical answer JSON to direct evaluation on the same
parent, and selected the same move. The result is saved in
`outputs/harness-check/tetris-score-gateway-parity.json`. The reduced 64-question
ceiling is bounded headroom for this existing caller, not a bulk feature.
