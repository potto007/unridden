"""Check a running context harness and measure its HTTP hop on one warm parent.

Creates short-lived memory snapshots and deletes them in finally. Never starts
or stops either service. This is not a /v1 versus snapshots benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", default="http://127.0.0.1:8092")
    parser.add_argument("--backend", default="http://127.0.0.1:8091")
    parser.add_argument("--backend-uds", help="private local backend socket")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 100:
        parser.error("pairs must be between 1 and 100")
    hello = json.loads(Path("unridden/examples/hello.json").read_text())
    handles: list[str] = []
    result: dict[str, Any] = {}
    with (
        httpx.Client(base_url=args.front, timeout=180, trust_env=False) as front,
        httpx.Client(
            base_url=args.backend,
            timeout=180,
            trust_env=False,
            transport=httpx.HTTPTransport(uds=args.backend_uds),
        ) as backend,
    ):
        front.get("/health").raise_for_status()
        backend.get("/health").raise_for_status()
        try:
            start = time.perf_counter()
            created = front.post(
                "/v1/contexts",
                json={
                    "state": hello["state"],
                    "ttl_seconds": 180,
                },
            )
            created.raise_for_status()
            capture_rtt = (time.perf_counter() - start) * 1000
            context = created.json()
            context_id = context["context_id"]
            handles.append(context_id)
            body = {"context_id": context_id, "questions": hello["questions"]}
            first = front.post("/v1/decisions", json=body)
            first.raise_for_status()
            reference = first.json()
            assert reference["usage"]["output_tokens"] == 0
            snapshot_id = reference["snapshot_usage"]["requested_parent"]
            metadata = backend.get("/v2/snapshots/" + snapshot_id)
            metadata.raise_for_status()
            assert metadata.json()["boundary"] == "context"
            assert metadata.json()["completed_blocks"] == context["completed_blocks"]
            alternate = front.post(
                "/v1/decisions",
                json={
                    "context_id": context_id,
                    "questions": {
                        "other": {
                            "type": "noul",
                            "instructions": "Does this concern astronomy?",
                        }
                    },
                },
            )
            alternate.raise_for_status()
            repeated = front.post("/v1/decisions", json=body)
            repeated.raise_for_status()
            assert repeated.json()["answers"] == reference["answers"]
            native_body = {
                "snapshot": {"id": snapshot_id, "relationship": "followup"},
                "questions": hello["questions"],
                "readout": {"completed_blocks": context["completed_blocks"]},
                "save_result_snapshot": False,
            }
            rows: list[dict[str, float]] = []
            for index in range(args.pairs):
                row: dict[str, float] = {}
                # Alternate order, same parent and questions, persistent clients.
                order = ("front", "direct") if index % 2 == 0 else ("direct", "front")
                for path in order:
                    client, endpoint, payload = (
                        (front, "/v1/decisions", body)
                        if path == "front"
                        else (backend, "/v2/decisions", native_body)
                    )
                    start = time.perf_counter()
                    response = client.post(endpoint, json=payload)
                    response.raise_for_status()
                    row[path + "_rtt_ms"] = (time.perf_counter() - start) * 1000
                    answer = response.json()
                    assert answer["answers"] == reference["answers"]
                    assert answer["usage"]["output_tokens"] == 0
                    if path == "front":
                        row["front_local_ms"] = answer["timing"]["local_ms"]
                rows.append(row)
            deleted = front.delete("/v1/contexts/" + context_id)
            deleted.raise_for_status()
            handles.remove(context_id)
            assert backend.get("/v2/snapshots/" + snapshot_id).status_code == 404
            assert front.post("/v1/decisions", json=body).status_code == 404
            # A second context expires without a decision. Wait through one
            # default cleanup interval, then verify the handle was swept away.
            expiring = front.post(
                "/v1/contexts",
                json={
                    "state": "short-lived lifecycle check",
                    "ttl_seconds": 1,
                },
            )
            expiring.raise_for_status()
            expiring_id = expiring.json()["context_id"]
            handles.append(expiring_id)
            time.sleep(6)
            assert front.get("/v1/contexts/" + expiring_id).status_code == 404
            handles.remove(expiring_id)
            result = {
                "profile": context["profile"],
                "completed_blocks": context["completed_blocks"],
                "question_count": len(hello["questions"]),
                "paired_requests": args.pairs,
                "checks": {
                    "context_boundary": True,
                    "aba_identity": True,
                    "paired_answers_identical": True,
                    "zero_generated_tokens": True,
                    "delete_removed_backend_snapshot": True,
                    "idle_expiry_cleanup": True,
                },
                "capture_client_rtt_ms": capture_rtt,
                "capture_backend_http_ms": context["capture_ms"],
                "direct_median_rtt_ms": statistics.median(
                    row["direct_rtt_ms"] for row in rows
                ),
                "front_median_rtt_ms": statistics.median(
                    row["front_rtt_ms"] for row in rows
                ),
                "paired_median_delta_ms": statistics.median(
                    row["front_rtt_ms"] - row["direct_rtt_ms"] for row in rows
                ),
                "front_median_local_ms": statistics.median(
                    row["front_local_ms"] for row in rows
                ),
                "samples": rows,
                "scope": (
                    "One warm parent; HTTP overhead, not snapshot speedup or quality."
                ),
            }
        finally:
            for context_id in handles:
                response = front.delete("/v1/contexts/" + context_id)
                if response.status_code not in {200, 404}:
                    raise RuntimeError("could not clean up a validation context")
            backend.get("/health").raise_for_status()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
