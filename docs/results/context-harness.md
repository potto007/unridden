# Gateway verification and latency, September 27, 2026

The automatic gateway is implemented and running as an on-demand local service.
Ordinary state-and-questions calls capture and reuse privately without manual
handles. The complex matched test measured about **4.3 ms added round-trip
latency**, about **0.8%** of the direct snapshot request. The cost is measurable;
proxying did not accelerate inference.

## Final automatic path: TCP

Each case used 100 alternating-order direct/front pairs after five warm-up
pairs, persistent HTTP connections, the same native parent and identical
questions within every pair. The gateway received the full ordinary state and
questions, discovered backend identity, checked its scope and resumed a warm
parent. Every warm front request was verified as a hit with zero capture cost
paid in that request. All answer JSON matched exactly and output tokens were
zero.

The simple case is one choice question from the hello example. The complex case
is the thirteen-question `injection-postmortem-quote` request from the frozen
guardrail policy suite. These are representative request shapes, not a claim of
representativeness across all workloads.

| Warm round-trip measure (ms) | Simple, 1 question | Complex, 13 questions |
| --- | ---: | ---: |
| Direct median | 26.938 | 520.898 |
| Gateway median | 30.224 | 526.362 |
| Direct p95 | 29.001 | 541.949 |
| Gateway p95 | 32.052 | 546.636 |
| Median paired difference, gateway minus direct | **3.354** | **4.274** |
| Paired median bootstrap 95% interval | 3.104–3.499 | 2.610–6.754 |
| Paired difference p05–p95 | 2.386–4.257 | -10.724–23.448 |
| Gateway local orchestration median | 0.211 | 0.309 |

Simple overhead is about 12.4% of the direct median; complex overhead is about
0.82%. The complex case's individual pairs are noisy, while its paired median
interval remains positive. Calling the cost negligible is an application
judgment; for this complex request it is small but detectable. The difference
of separate medians need not equal the median of paired differences.

The intervals resample paired differences 5,000 times with a fixed seed.
They describe these measurements, assume exchangeable pairs, and do not account
for every form of temporal correlation or generalize to other workloads.

This repeat includes the current batching, admission and busy-handling code.
No waiting or busy refusals were induced in the latency sample; contention
correctness is tested separately, without claiming production throughput.

## Capture is separate

Ten alternating capture pairs per case measured equivalent standalone native
capture versus front manual-context creation, followed by deletion. This
isolates setup from warm decisions; it is not the total first automatic call.

| Capture round-trip measure (ms) | Simple | Complex |
| --- | ---: | ---: |
| Direct median / p95 | 28.452 / 31.949 | 41.095 / 45.876 |
| Front create median / p95 | 31.483 / 37.624 | 45.109 / 50.760 |
| Median paired setup difference | 3.047 | 4.193 |
| Paired median bootstrap 95% interval | 2.151–4.010 | 3.129–4.790 |

A separate first automatic call took 65.8 ms for the simple case and 585.1 ms
for the complex case, including setup and decision; their backend capture
components were 28.9 and 42.7 ms. Each is one observation, not a first-call
distribution. Warm decisions pay no capture. Setup amortization is not itself
a measured saving over ordinary inference.

## Optional Unix socket versus TCP

Before automatic mode was completed, the same benchmark measured **manual**
context resumes over each transport. Those runs also used 100 warm pairs and
10 capture pairs per case. The front always accepted HTTP/TCP; the direct
privileged client and front-to-backend path used the selected transport.

| Manual warm path | Direct median (ms) | Front median (ms) | Paired overhead (ms), 95% interval |
| --- | ---: | ---: | ---: |
| TCP, simple | 27.131 | 29.175 | 2.023 [1.904, 2.178] |
| Socket, simple | 26.881 | 28.716 | 2.002 [1.769, 2.079] |
| TCP, complex | 530.677 | 534.307 | 3.911 [2.445, 6.543] |
| Socket, complex | 530.298 | 535.784 | 4.632 [2.346, 6.046] |

There is no material socket latency benefit in these observations. TCP/socket
runs were sequential and included a backend restart; they do not isolate a
sub-millisecond transport cost from temporal or model runtime variation.
Automatic mode adds identity discovery and state validation, so its simple
overhead should not be represented by the earlier manual result.

The user selected two local TCP services as the default. Socket transport remains
configurable for deployment isolation; remote HTTP(S) support is retained.

## Machine and scope

- NVIDIA RTX 5090, 32 GiB class (32,607 MiB reported), driver 617.14.
- WSL Linux 6.18.40.1; other GPU services remained present, but no competing
  model benchmark or test suite ran during the final timed sample.
- Gemma 4 26B-A4B UD-Q4_K_XL, native `split18-30-v1`, final checkpoint 30.
- Snapshot-only GPU worker; context 2048, batch/ubatch 256, eight threads.
- One front process, one admitted decision, persistent clients, no concurrent
  load. This does not qualify throughput or queueing under contention.
- No E4B native run was added by this harness task. Its full-profile discovery
  is tested with the actual API service and a fake native worker.
- No seven-suite snapshot quality benchmark or matched fresh-`/v1` break-even
  benchmark was performed. Existing ordinary model comparisons answer a
  different question.

Server `local_ms` excludes ASGI parsing/serialization and network work; it
must not be presented as full gateway overhead. Client measurements include
the actual extra gateway path.

## Lifecycle and code verification

The final complete Python suite passed **290 tests**, including the integrated
Tetris controller and its gateway migration. Focused harness coverage passed
59 tests using the actual backend ASGI service with a native fake. Coverage
includes split/full profiles, correct capture boundary, A/B/A branching,
cookie isolation, changed state/instructions/profile, fixed expiry, automatic
capacity eviction with manual protection, definitive loss recovery, manual
override, bounded cleanup, cancellation, body/response limits, sanitized errors,
and no ambiguous request replay. Generic orchestration coverage additionally
checks native batching, aggregated work, bounded admission/retry, cancellation,
operation deadlines, partial failure and unchanged Choice semantics.

Live 26B checks verified automatic first capture, warm reuse, independent clients,
A/B/A answer identity, instruction invalidation and recovery after deleting the
owned native parent. The manual check verified owned deletion and idle expiry. The updated live
validator also sent 34 independent questions in one call, observed two native
batches, and matched every answer against direct calls on the same parent.
Ruff formatting/lint, strict mypy and whitespace gates are recorded with the
final implementation checks.

## Reproduction and local artifacts

```bash
.venv/bin/python scripts/unridden/validate_harness_automatic.py
.venv/bin/python scripts/unridden/validate_harness.py \
  --output outputs/harness-check/manual-latest.json
.venv/bin/python scripts/unridden/bench_harness.py --mode automatic \
  --output outputs/harness-check/tcp-orchestration-matched.json
# Use --mode manual for the manual comparison; --backend-uds selects a socket.
```

Raw results retained under `outputs/harness-check/`:

- `tcp-orchestration-matched.json`: current automatic path with generic batching/admission, all paired samples.
- `tcp-automatic-matched.json`: earlier automatic snapshot-only orchestration measurement, retained for comparison.
- `tcp-matched.json`, `unix-matched.json`: earlier manual transport comparisons.
- `automatic-live-check.json`: automatic lifecycle verification.
- `2026-09-27.json`, `unix-live-check.json`: earlier manual native checks.
- `deployment-before/`: original model unit and service state.

The original model unit was temporarily switched to a socket for comparison,
then restored byte-for-byte to its original TCP configuration. Both native
backend and gateway are healthy, loopback-bound and running on demand, not
enabled at boot. The backend remains reachable by same-host processes.
See [deployment and rollback](../gateway-deployment.md).
