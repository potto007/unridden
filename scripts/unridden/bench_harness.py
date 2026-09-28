"""Matched front/direct round trips on the same native parent and question set.

Never starts or stops a service. Run separately for HTTP and Unix socket backends.
Raw samples are saved; the model never receives benchmark timing or metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values: list[float]) -> dict[str, float]:
    return {"median_ms": statistics.median(values), "p95_ms": percentile(values, 0.95)}


def paired_stats(rows: list[dict[str, float]]) -> dict[str, Any]:
    direct = [row["direct_ms"] for row in rows]
    front = [row["front_ms"] for row in rows]
    delta = [f - d for f, d in zip(front, direct, strict=True)]
    rng = random.Random(20260927)
    bootstrap = [
        statistics.median(rng.choices(delta, k=len(delta))) for _ in range(5000)
    ]
    return {
        "direct": stats(direct),
        "front": stats(front),
        "paired_delta": stats(delta),
        "paired_delta_p05_ms": percentile(delta, 0.05),
        "paired_median_bootstrap_95_interval_ms": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "front_local": stats([row["local_ms"] for row in rows])
        if "local_ms" in rows[0]
        else None,
    }


def checked(
    client: httpx.Client, method: str, path: str, body: Any = None
) -> tuple[Any, float]:
    start = time.perf_counter()
    response = client.request(method, path, json=body)
    response.raise_for_status()
    data = response.json()
    return data, (time.perf_counter() - start) * 1000


def measure(
    front: httpx.Client,
    backend: httpx.Client,
    request: dict[str, Any],
    pairs: int,
    capture_pairs: int,
    automatic: bool,
) -> dict[str, Any]:
    model = checked(backend, "GET", "/v2/models")[0]["models"][0]
    block = model["capabilities"]["readout_blocks"][0]
    captures: list[dict[str, float]] = []
    for index in range(capture_pairs):
        row: dict[str, float] = {}
        for path in ("front", "direct") if index % 2 == 0 else ("direct", "front"):
            if path == "front":
                created, row["front_ms"] = checked(
                    front,
                    "POST",
                    "/v1/contexts",
                    {
                        "state": request["state"],
                        "ttl_seconds": 600,
                    },
                )
                checked(front, "DELETE", "/v1/contexts/" + created["context_id"])
            else:
                created, row["direct_ms"] = checked(
                    backend,
                    "POST",
                    "/v2/snapshots",
                    {
                        "input": {"kind": "context", "state": request["state"]},
                        "checkpoints": [block],
                        "ttl_seconds": 600,
                    },
                )
                checked(
                    backend, "DELETE", "/v2/snapshots/" + created["snapshots"][0]["id"]
                )
        captures.append(row)

    front.cookies.clear()
    if automatic:
        body = request
        first, first_ms = checked(front, "POST", "/v1/decisions", body)
        assert first["context_usage"]["cache"] == "miss"
        handle = front.cookies.get("unridden_context")
        assert handle is not None
    else:
        created = checked(
            front,
            "POST",
            "/v1/contexts",
            {
                "state": request["state"],
                "ttl_seconds": 3600,
            },
        )[0]
        handle = created["context_id"]
        body = {"context_id": handle, "questions": request["questions"]}
        first, first_ms = checked(front, "POST", "/v1/decisions", body)
    try:
        native = {
            "snapshot": {
                "id": first["snapshot_usage"]["requested_parent"],
                "relationship": "followup",
            },
            "questions": request["questions"],
            "readout": {"completed_blocks": block},
            "save_result_snapshot": False,
        }
        rows: list[dict[str, float]] = []
        for index in range(pairs + 5):
            row = {}
            for path in ("front", "direct") if index % 2 == 0 else ("direct", "front"):
                client, endpoint, payload = (
                    (front, "/v1/decisions", body)
                    if path == "front"
                    else (backend, "/v2/decisions", native)
                )
                answer, row[path + "_ms"] = checked(client, "POST", endpoint, payload)
                assert answer["answers"] == first["answers"], "answer parity failed"
                assert answer["usage"]["output_tokens"] == 0
                if path == "front":
                    row["local_ms"] = answer["timing"]["local_ms"]
                    if automatic:
                        assert answer["context_usage"]["cache"] == "hit"
                        assert answer["context_usage"]["capture_ms_this_request"] == 0
            if index >= 5:
                rows.append(row)
        return {
            "profile": model["profile"],
            "checkpoint": block,
            "question_count": len(request["questions"]),
            "request_sha256": hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest(),
            "answer_parity": True,
            "automatic": automatic,
            "first_front_request_ms": first_ms,
            "first_front_capture_ms": first["context_usage"]["capture_ms"],
            "warm_pairs": pairs,
            "capture_pairs": capture_pairs,
            "warm": paired_stats(rows),
            "capture": paired_stats(captures),
            "warm_samples": rows,
            "capture_samples": captures,
        }
    finally:
        checked(front, "DELETE", "/v1/contexts/" + handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", default="http://127.0.0.1:8092")
    parser.add_argument("--backend", default="http://127.0.0.1:8091")
    parser.add_argument("--backend-uds")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--capture-pairs", type=int, default=10)
    parser.add_argument("--mode", choices=["automatic", "manual"], default="automatic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 10 or args.capture_pairs < 1:
        parser.error("at least ten decision pairs and one capture pair are required")
    hello = json.loads(Path("unridden/examples/hello.json").read_text())
    simple = {
        "state": hello["state"],
        "questions": {"route": hello["questions"]["route"]},
    }
    suite = json.loads(
        Path("unridden/examples/usecases/guardrail-policy-gates.json").read_text()
    )
    complex_case = max(
        suite["cases"], key=lambda case: len(case["request"]["questions"])
    )
    results: dict[str, Any] = {
        "transport": "unix" if args.backend_uds else "tcp",
        "mode": args.mode,
        "complex_case": complex_case["name"],
        "cases": {},
        "method": (
            "Alternating order, five warm-up pairs, same parent per case, "
            "persistent clients."
        ),
    }
    with (
        httpx.Client(base_url=args.front, timeout=180, trust_env=False) as front,
        httpx.Client(
            base_url=args.backend,
            timeout=180,
            trust_env=False,
            transport=httpx.HTTPTransport(uds=args.backend_uds),
        ) as backend,
    ):
        checked(front, "GET", "/health")
        for name, request in (("simple", simple), ("complex", complex_case["request"])):
            result = measure(
                front,
                backend,
                request,
                args.pairs,
                args.capture_pairs,
                args.mode == "automatic",
            )
            results["cases"][name] = result
            print(
                json.dumps(
                    {
                        "case": name,
                        "transport": results["transport"],
                        "warm": result["warm"],
                        "capture": result["capture"],
                    }
                ),
                flush=True,
            )
        checked(backend, "GET", "/health")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
