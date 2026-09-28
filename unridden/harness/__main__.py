"""Serve the experimental context harness without loading a model."""

from __future__ import annotations

import argparse

import uvicorn

from unridden.harness.app import create_app
from unridden.harness.service import HarnessConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--backend-url", default="http://127.0.0.1:8091")
    transport.add_argument("--backend-uds", help="absolute path to a local Unix socket")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "::1"])
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--max-contexts", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--snapshot-policy", choices=["auto", "manual"], default="auto")
    parser.add_argument("--automatic-ttl", type=int, default=600)
    parser.add_argument("--max-waiting", type=int, default=16)
    parser.add_argument("--queue-timeout", type=float, default=5.0)
    parser.add_argument("--backend-busy-timeout", type=float, default=5.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        config = HarnessConfig(
            backend_url=args.backend_url,
            backend_uds=args.backend_uds,
            max_contexts=args.max_contexts,
            request_timeout=args.request_timeout,
            automatic_enabled=args.snapshot_policy == "auto",
            automatic_ttl=args.automatic_ttl,
            max_waiting=args.max_waiting,
            queue_timeout=args.queue_timeout,
            backend_busy_timeout=args.backend_busy_timeout,
        )
    except ValueError as error:
        parser.error(str(error))
    # One worker owns all handles. URL paths can contain capability handles.
    uvicorn.run(create_app(config), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
