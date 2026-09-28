# 0010. An application gateway with automatic snapshot lifecycle

- Status: implemented first version, locally deployed, experimental
- Date: 2026-09-27

## Decision

Applications call a lightweight gateway with state and typed questions.
Automatic policy captures before any varying question, privately reuses the
latest stable parent for that client, and owns expiry, replacement and safe
recovery. Advanced callers can explicitly capture, resume and release contexts;
a deployment can require manual policy. Both modes use snapshots.

The native service remains responsible for model loading, snapshot storage,
immutable branching, restore, recovery and typed readouts. The gateway owns
application policy without loading a model or duplicating native machinery.

An explicit-only prototype did not satisfy the intended ordinary developer
experience. The final first version makes reuse automatic with an ordinary
persistent HTTP client's cookie jar. Manual handles are an advanced interface.

## Why a separate layer

A Python helper would leave other languages to reimplement lifecycle and would
not provide shared admission or capacity policy. Extending the native server
could remove an HTTP hop but would couple application policy and native-worker
deployment. A small independent gateway gives callers one stable application
contract and preserves the tested backend. Measurements now quantify its cost.

Direct worker IPC or in-process ownership would require restructuring ownership
of native recovery and storage. That complexity is not justified by the first
measurement: automatic gateway overhead was about 3.4 ms for one simple
question and 4.3 ms for a complex thirteen-question bundle on this machine.
These are measured costs, not speedups or universal limits.

## Automatic policy and isolation

Each random cookie capability owns one latest context. Exact canonical state,
optional invariant instructions and the advertised backend configuration define
reuse eligibility within that scope. Questions and answers never define or
contaminate the parent. No answer cache or global cross-caller matching exists.

First use captures at the final checkpoint advertised by the backend (30 for
the split 26B profile, 42 for the E4B full profile). Matching later requests
resume; changed state/instructions/profile replaces. Fixed TTL, bounded
capacity and eviction of automatic contexts bound retention. Manual contexts
are not selected by automatic eviction, although native memory eviction can
still remove them.

An unknown/expired scope is rebuilt from the current request. A definitive
snapshot-not-found response is known to precede inference and allows one
recapture. Ambiguous failures are not replayed. Manual requests never recapture
automatically. No warm-up question, answer child, model switching, generation,
ordinary-inference fallback or automatic model-service restart is introduced.

The gateway stores metadata, not prompts, after the request. Stateless clients
still work but pay capture each time. The cookie approach gives common browser
and persistent HTTP clients automatic behavior without a new SDK. It requires
a cookie jar, holds only one current state per scope, and is not an
authenticated tenant identity. These are explicit first-version tradeoffs.

The identity includes the backend origin/transport, alias, profile, prompt
version and checkpoint. It cannot pin exact weights/runtime hashes because
the backend does not advertise them. Exact identity is a production follow-up;
operators must currently pin their deployment version themselves.

## Benefit policy and observability

There is no evidence for a universal capture threshold. The gateway does not
know future reuse or a matched ordinary-decision baseline. Capture is required
for this snapshot contract; it is paid on a miss and amortized on hits. The
policy selects lifecycle actions from known input identity and resource state,
not a prediction of correctness or future performance.

Report setup separately, hit/miss and reason, completed request count, amortized
setup, backend HTTP time and local orchestration. Preserve native inference and
restore diagnostics. Client round trips must be measured separately because
server timing excludes parsing, serialization and client transport.

The first measurement covers simple and complex matched branches, not snapshot
versus ordinary inference or the seven-suite quality corpus. Snapshot and
ordinary paths can differ numerically on uncertain questions. Quality and
break-even must be qualified for the application's workload.

## Batching and streaming

Generic request orchestration belongs in the gateway. It accepts up to 64
independent questions, discovers native batch limits, executes bounded batches
against one immutable parent and publishes a complete keyed result. Tokens,
native timings and work totals aggregate across batches. A failure after partial
work yields an error, without presenting partial answers as a complete result
or replaying completed batches. The operation itself has a time budget.

Short busy periods are handled with bounded admission waiting and retries of
recognized pre-admission refusals. When cancellation reaches the operation or
its server deadline expires, queued work is removed and future batches stop.
A client disconnection alone does not guarantee that active server/native work
has stopped; the server budget still applies. Immediate preemption of an
already-running GPU operation is not promised. Both automatic and manual context modes
share these generic mechanics. Clients retain their own application deadlines.

**Keep complete JSON as the default.** It is simplest for ordinary clients and
preserves a clear all-answers-or-error response. The 64-question ceiling bounds
work and memory; it is not an empirically optimal batch size or a throughput
claim. The backend remains sequential, currently returning complete native
batches.

The concrete reason to cross the native 32-question boundary is Tetris's Score
caller: one question per legal placement can reach 34. The largest inspected
frozen-suite bundle has 13 questions and hello has three. There is no verified
256-question use case. The initially proposed ceiling was reduced to 64, modest
headroom covering the existing case and currently at most two native batches.
Larger retrieval or rubric workloads are speculative; they do not justify a
bulk API claim. One/few-question calls remain the ordinary path.

Optional answer streaming could reduce time to first useful answer, but would
not inherently improve total throughput. Immediate per-question delivery needs
smaller native calls or native streaming support, with measured transport,
restore and serialization costs. It would require question IDs/order, progress,
a final completion marker, and explicit partial-failure/cancellation semantics.
An interrupted stream must not look like a complete answer set. Backpressure
and reconnect/replay behavior also need a contract.

Do not add streamed question input now: questions arriving over time can use
ordinary calls and automatically reuse their context. A second bidirectional
protocol would add framing, idle admission and lifetime complexity without a
demonstrated need. No streaming contract is implemented by this decision.

Choice grouping, tournaments, ranking rubrics and Tetris strategy remain
application policy. Native Choice distributions retain their meaning; the
gateway never invents a distribution across separately compared groups.

## Deployment and limits

The chosen local deployment uses two loopback TCP listeners: the gateway is the
recommended application endpoint; the model service is the internal backend.
Same-host processes can still call the backend. This is not an exclusive
security boundary. The original backend unit and TCP functionality are retained.

HTTP over a Unix socket remains an optional private local transport. A measured
comparison found no material performance benefit here, so it is not the local
default. HTTP(S) to a compatible remote Unridden backend is also supported.
No generic cloud-provider adapter, remote authentication headers or external
ingress is part of this version.

One front process owns an ephemeral registry and runs one backend mutation or
decision at a time. A bounded queue and finite busy-retry budget absorb brief
contention; overflow, timeout and ambiguity remain explicit failures. Cleanup debt counts toward capacity, shutdown cleanup
is bounded, and native TTL is a further retention bound. Access logging is off
to avoid recording capability handles.

Local systemd installation is on demand, not enabled at boot. It is reversible
without altering the original model unit. Shared production use additionally
needs authentication, exact backend identity, monitoring/capacity qualification
and a separately reviewed shared or durable registry if required.

See [caller guide](../context-harness.md),
[deployment and rollback](../gateway-deployment.md), and
[verification and latency](../results/context-harness.md).
