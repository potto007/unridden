# Unridden application gateway

Send state and typed questions to one API. By default, the gateway captures the
state before the questions, reuses that context on later requests, and manages
expiry, invalidation and recovery. Applications do not create snapshot handles.

The gateway loads no model. It calls the existing native snapshot service,
which owns inference, immutable branches, restore and the memory budget.
[ADR 0010](decisions/0010-explicit-context-harness.md) explains the division.
[Deployment](gateway-deployment.md) covers local TCP, optional Unix sockets,
remote HTTP and rollback.

## Ordinary integration

Use a persistent HTTP client, as you normally would for connection reuse. Its
cookie jar carries the private context scope automatically:

```python
import httpx

state = "A customer was charged twice for one order."
questions = {"billing": {"type": "noul", "instructions": "Is this a billing issue?"}}
with httpx.Client(base_url="http://127.0.0.1:8092", timeout=180) as client:
    for question_set in (questions, questions):
        response = client.post(
            "/v1/decisions",
            json={"state": state, "questions": question_set},
        )
        response.raise_for_status()
        print(response.json()["answers"])
```

Changing the questions keeps the stable parent. Changing the state captures a
new parent and releases the previous one. Optional top-level `instructions`
are invariant instructions and are part of the parent identity. Per-question
instructions belong in `questions`. No warm-up question, answer, sibling
question, cookie, or gateway diagnostic enters the captured state.

The service is already installed locally and can be started on demand:

```bash
systemctl --user start localai-unridden-gateway.service
```

For development beside the running service:

```bash
.venv/bin/python -m unridden.harness \
  --backend-url http://127.0.0.1:8091 --port 8093
```

The front's `/docs` exposes its schema. `/health` checks upstream health and
the snapshot profile. `/v1/models` advertises the profile, limits and policy.

## What automatic policy decides

- Each independent cookie jar has its own unguessable capability scope and one
  latest stable context. Identical state is never matched across scopes.
- The first request captures; subsequent requests with the same canonical state,
  invariant instructions and advertised backend profile resume the same parent.
  JSON object key order is immaterial; array order and string contents matter.
- Changed input or advertised profile replaces the parent. Idle expiry, front
  restart, or capacity eviction causes a fresh capture on the next request.
- A definitive native `snapshot_not_found` response triggers one recapture only
  before any question batch completes. Loss partway through a larger bundle is
  an explicit incomplete-request error; earlier work is not replayed. Ambiguous
  transport errors, timeouts and failed inference are never automatically replayed.
- The default fixed TTL is ten minutes, measured from capture admission. Access
  does not extend it. `--automatic-ttl` accepts one second through one day.
- There are at most 64 context records by default, including cleanup debt.
  Automatic creation may evict the least recently used automatic context.
  Manual contexts are not selected by this eviction policy.
- Expiry cleanup runs independently of traffic, every five seconds, and during
  capture. Failed deletion remains bounded and is retried. Native TTL provides
  a further bound; the native store removes expired entries during store work.

Use one client/cookie jar per intended privacy and reuse scope. Sharing a client
across unrelated users also shares its latest-context scope. A client that does
not retain cookies still works, but captures on every request. Cookie loss and
interleaved changing states affect performance, not which state is evaluated.
For several simultaneously reusable contexts, use separate clients or manual
contexts. There is no global answer cache.

The cookie is HttpOnly, SameSite=Strict, limited to the API path, and Secure on
HTTPS. Its remaining lifetime follows the original fixed expiry. This is a
local capability mechanism, not user authentication. Treat cookies and manual
handles as private. Access logging is disabled by the supplied launcher.

Automatic policy does not infer which *part* of arbitrary state is invariant,
predict future requests, rewrite questions, switch models, or silently use
ordinary inference. Both automatic and manual modes use snapshots. There is no
no-snapshots mode.

## One call, backend limits handled internally

One or a few questions remain the ordinary integration. The ceiling is **64
independent questions**, within the 1 MiB body limit. This provides bounded
headroom for the verified Tetris Score caller's up to 34 placement questions;
the largest inspected use-case-suite bundle has 13 and hello has three. There
is no demonstrated need for 256 questions, and the initially chosen ceiling was
reduced. This is not a bulk-performance feature or parallel model evaluation.
The gateway discovers the native per-call limit (currently 32),
splits internally, and returns one complete keyed answer bundle. Every question
still sees only its own question and the same immutable parent. Choice, Noul and
Score semantics are unchanged; oversized Choice alternatives are not converted
into a tournament or given fabricated global probabilities.

Brief contention is handled behind the API. At most 16 requests wait for the
single active operation, for up to five seconds each. Recognized native busy
refusals are retried with bounded backoff, at most twenty attempts and five
seconds per upstream operation. Unknown refusals, timeouts and network failures
are not replayed. The admitted operation has a 180-second total budget by default,
covering all its batches, discovery, capture and retries. Queue time is additional;
bounded cleanup may finish after cancellation.

Advanced deployment settings are `--max-waiting`, `--queue-timeout`,
`--backend-busy-timeout` and `--request-timeout`. Zero wait/retry budgets request
immediate refusal. Queue overflow or exhausted wait returns 429. These are
generic resource limits, independent of automatic versus manual snapshot policy.

Clients do not implement native-sized batching or busy-retry loops. All answers
are returned together; if a later batch fails, the API returns an error rather
than a partial success. Earlier native work may have run, so this is response
atomicity, not rollback of computation. No completed batch is automatically
replayed after parent loss, and no success counter is incremented for an
incomplete bundle.

For interactive questions arriving over time, send ordinary calls using the
same persistent client. Automatic context reuse already applies. Answer streaming
is not implemented; the current default favors a simple complete response.
[Streaming assessment](decisions/0010-explicit-context-harness.md#batching-and-streaming)
explains the possible future tradeoff.

## Advanced manual control

Manual contexts are available even when automatic policy is the default. The
developer chooses exactly what and when to capture, which parent to resume,
and when to release it:

```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:8092", timeout=180) as client:
    created = client.post(
        "/v1/contexts",
        json={"state": "A customer was charged twice.", "ttl_seconds": 600},
    )
    created.raise_for_status()
    handle = created.json()["context_id"]
    try:
        result = client.post(
            "/v1/decisions",
            json={
                "context_id": handle,
                "questions": {
                    "billing": {
                        "type": "noul",
                        "instructions": "Is this a billing issue?",
                    }
                },
            },
        )
        result.raise_for_status()
        print(result.json()["answers"])
    finally:
        client.delete(f"/v1/contexts/{handle}").raise_for_status()
```

To require explicit control across a deployment, start it with
`--snapshot-policy manual`. Ordinary state-and-questions calls then return
`manual_context_required`; create/resume/release operations remain available.
This disables automatic policy decisions, not snapshot functionality.

| Operation | Meaning |
| --- | --- |
| `POST /v1/decisions` with `state`, `questions` | Automatic lifecycle, default policy. |
| `POST /v1/contexts` | Explicit capture of state, optional invariant instructions and TTL; returns 201 and a fresh handle. |
| `POST /v1/decisions` with `context_id`, `questions` | Branch from that immutable manual parent. |
| `GET /v1/contexts/{context_id}` | Metadata and successful request count, never the original state. |
| `DELETE /v1/contexts/{context_id}` | Revoke the capability and delete its backend snapshot. |

Manual handles have fixed TTLs and do not survive front restart. Lost or expired
manual contexts produce a distinct error; the developer recreates them.
No automatic recapture is performed for a manual decision. Each create is
independent even for identical state.

## Responses, timing and failure

Successful responses preserve native `model`, `answers`, `usage` and
`snapshot_usage`, with `context_usage` and `timing` for diagnostics. The happy
path needs only `answers`. Automatic responses omit the manual context handle.

- `context_usage.cache`: `hit` or `miss`.
- `capture_reason`: first scope, expired/evicted scope, changed input/profile,
  same stable context, or definitive backend loss.
- `capture_ms_this_request`: capture HTTP time paid now; zero on a warm hit.
- `capture_ms`: initial backend capture cost of the current parent.
- `successful_decisions`: completed request count, including multi-question
  requests. `amortized_capture_ms` divides setup cost by that count; it does
  not claim time saved.
- `execution`: successful native `decision_batches` and recognized `busy_retries`.
- `timing`: server orchestration, backend HTTP, local work, `queue_ms` and
  `backoff_ms`. Orchestration includes waiting; local work excludes it. These
  timings exclude ASGI parsing, response serialization/transmission and the
  client network. Native restore/inference and token/byte totals are summed
  across successful batches under `snapshot_usage`; suffix tokens remain keyed
  by the original question IDs.

One mutation or decision runs at a time with the bounded waiting policy above.
Request bodies are capped at 1 MiB and each decoded upstream response at 4 MiB. The native memory budget may evict snapshots before
front TTL or capacity is reached. The limits do not promise room for 64 maximal
contexts.

| Failure | Behavior |
| --- | --- |
| 404 `context_not_found`, 410 `context_expired` / `context_lost` | Manual context must be recreated explicitly. Automatic mode handles a missing scope and one definitive lost-snapshot recovery. |
| 429 `busy` / `backend_busy` | Capacity or the internal wait/retry budget was exhausted. Surface the failure or schedule a later application attempt; no client retry loop is required. |
| 503 `backend_unavailable`, 504 `backend_timeout` / `request_timeout` | No automatic replay; inspect health. An ambiguous connection/timeout is marked non-retryable automatically. |
| 410 `context_lost_after_progress` | A later batch lost its parent. No partial bundle is returned and earlier work is not replayed. |
| 507 capacity error | Release manual contexts, let cleanup finish, or reassess memory/capacity. |
| 422 validation or `backend_rejected` | Correct input or reduce it to fit model limits. |
| 502 backend/protocol error | Inspect service logs; raw upstream content is not returned. |

Error responses use `error.code`, `error.message`, `error.retryable`.
Responses carry `Cache-Control: no-store`. Failed first decisions release their
unclaimed capture where possible; cleanup debt still consumes bounded capacity.

## Honest boundaries

The frontend retains metadata after capture, not original state or answers.
Automatic requests send state again, allowing recapture without a retained
prompt store. Input identity covers backend origin/transport, model alias,
profile, prompt version and checkpoint. The backend does not yet advertise
exact weights/runtime hashes: changing weights behind an identical
advertisement cannot be detected. Pin deployment versions for reproducible use.

Snapshot inference can differ numerically from ordinary native `/v1` inference.
The current backend is snapshot-only to avoid loading another model instance.
The gateway accepts the familiar ordinary request shape, but is not a transparent
proxy for every native endpoint. Raw snapshots, vectors and Rider are excluded.

For capture cost C, warm cost R and fresh ordinary cost F, reuse would pay back
when F > R and N > C / (F - R). The gateway does not know future N or a matched
ordinary baseline. It therefore makes no invented break-even promise. Automatic
first capture is a lifecycle policy for this snapshot API; repeated stable
requests amortize setup. Changing state every request still pays setup.

This first version supports a single front process, ephemeral scopes and local
capabilities. Shared production use needs authenticated ingress, exact backend
identity, an operational capacity policy and a reviewed shared registry if
multiple front workers or seamless restarts are needed. E4B full-profile
checkpoint discovery is supported; the native source build requirement remains
in [the E4B guide](e4b-snapshots.md).

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/unridden/validate_harness_automatic.py
.venv/bin/python scripts/unridden/validate_harness.py \
  --front http://127.0.0.1:8092 --backend http://127.0.0.1:8091 \
  --output outputs/harness-check/manual-latest.json
.venv/bin/python scripts/unridden/bench_harness.py --mode automatic \
  --output outputs/harness-check/tcp-automatic-matched.json
```

See [measured results](results/context-harness.md). The automatic live validator
targets the default local ports and cleans up its owned contexts; the manual
validator and benchmark accept configurable backends.
