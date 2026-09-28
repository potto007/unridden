# Application gateway deployment

Applications use the gateway; it sends only the intended stable context and
typed question branches to the model service. The [caller guide](context-harness.md)
shows automatic integration and advanced manual control.

## Current local deployment

Two on-demand user services are installed and running:

| Service | Listener and purpose |
| --- | --- |
| `localai-unridden-gateway.service` | `127.0.0.1:8092`, recommended application API, automatic policy. |
| `localai-unridden-snapshots.service` | `127.0.0.1:8091`, native 26B snapshot-only backend. |

Both remain disabled for boot, preserving the original on-demand policy.
Starting the gateway starts its backend dependency. The gateway loads no model.
The existing 26B native unit is unchanged; the optional socket drop-in used
during comparison has been removed. The smaller-model service remains inactive.

Both TCP endpoints are reachable by same-host applications, including Windows
through WSL localhost forwarding. Loopback binding limits network exposure; it
does not make the backend inaccessible to local processes. The gateway is the
one recommended application interface, not an enforced exclusive boundary.

Before changing transport, active sockets, benchmark/model processes, user units
and configured forwarding were inspected. No unrelated live model workload was
found. The model was temporarily restarted for socket comparison and restored
to its original TCP configuration afterward. No unrelated service was stopped.
The original unit and state are saved under
`outputs/harness-check/deployment-before/`.

The managed front unit is provided in
`scripts/unridden/systemd/localai-unridden-gateway.service`. Its checkout path
and backend unit name match this machine; review them before installing elsewhere.
Importing the Python package does not install or start services.

```bash
systemctl --user start localai-unridden-gateway.service
systemctl --user status localai-unridden-gateway.service
curl --fail http://127.0.0.1:8092/health
curl --fail http://127.0.0.1:8091/health
```

The launcher binds only to loopback, disables access logs, and uses one process.
For access from other machines, place authenticated TLS ingress in front of the
gateway and expose only that ingress. No external ingress or authentication is
installed here.

## Select backend transport

The application API stays the same for all compatible backends:

```bash
# Default local TCP backend; separate development front port.
.venv/bin/python -m unridden.harness \
  --backend-url http://127.0.0.1:8091 --port 8093

# Compatible remote Unridden backend, with HTTPS.
.venv/bin/python -m unridden.harness \
  --backend-url https://model-service.example --port 8093

# Optional local Unix socket.
.venv/bin/python -m unridden.harness \
  --backend-uds /run/user/1000/unridden-backend/api.sock --port 8093
```

HTTP(S) means the existing Unridden snapshot contract, not arbitrary cloud
provider APIs. TLS verification is enabled; environment proxy settings,
redirects and URL-embedded credentials are not accepted as routing mechanisms.
Remote authentication headers are not implemented in this first pass.

| Internal path | Tradeoff |
| --- | --- |
| HTTP(S) over TCP | Local or remote, reuses the original service configuration. Chosen local default. |
| HTTP over a Unix socket | Can remove the backend TCP listener and restrict access via directory permissions. Still uses HTTP serialization and tested native snapshot machinery. Optional. |
| Direct worker IPC or gateway-owned model | Could avoid the extra ASGI/HTTP layer, but couples lifetimes and requires restructuring store/recovery ownership. Deferred. |

The [matched results](results/context-harness.md) show no material decision
latency advantage from switching TCP to a Unix socket on this machine. Socket
support is retained for deployment isolation rather than claimed speed.

## Optional Unix-socket configuration

The native CLI now accepts `--uds`; original host/port mode remains supported.
`scripts/unridden/systemd/90-private-backend.conf` is an **optional**, currently
uninstalled drop-in. It preserves this machine's GPU/snapshot-only flags and
selects a socket inside a runtime directory with mode 0700.

If selecting it later, save existing configuration, inspect consumers, install
only that drop-in, reload user systemd, and restart the backend. Verify socket
health before changing the gateway's backend argument and restarting it. Check
directory permissions and absence of the old model TCP listener if a private
socket is the objective. Same-user processes and root can still reach it.
The tested drop-in is specific to this checkout and model.

To undo that optional transition, remove only the added drop-in, reload systemd,
restart the backend, health-check the original TCP endpoint, and restore the
gateway's `--backend-url` argument. Do not delete the base unit or unrelated
drop-ins.

## Manual policy and rollback

Automatic policy is the default. Add `--snapshot-policy manual` to the front
command when developers must explicitly select capture/resume/release. Manual
context endpoints work under either setting. Neither setting disables snapshots.

For the **current TCP deployment**, rollback is simply stopping the gateway:

```bash
systemctl --user stop localai-unridden-gateway.service
curl --fail http://127.0.0.1:8091/health
```

The backend stays on its original endpoint. To remove the installed front unit,
remove only `localai-unridden-gateway.service` from the user's unit directory
and reload systemd. Its dependency does not require stopping the backend when
the front is stopped. No boot enablement needs undoing.

## Migration boundary

Point ordinary applications at the gateway with the same state-and-questions
request shape and a persistent cookie-capable client. Review actual quality
and latency: the gateway uses snapshot inference, which is not numerically
identical to ordinary native inference in every uncertain case. Existing
diagnostic scripts can continue to reach the raw backend on its original port.

The front deliberately excludes raw snapshots, arbitrary state evaluation,
vectors and Rider. It does not proxy the complete native API. Contexts are
ephemeral; manual users handle lost-context errors, automatic users get bounded
recapture from their current request. Multiple front workers cannot share the
current registry.

Before claiming an exclusive shared production gateway, inventory callers,
establish authentication/ingress, restrict backend network access, expose and
pin exact backend identity, and qualify concurrency, memory and lifecycle.
None of those claims follows merely from installing the local front service.
