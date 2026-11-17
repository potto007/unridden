"""Exercise the API's timeout and worker-death paths against the real worker.

This is an explicit GPU experiment driver, not a server. It never binds a port.
Two app lifespans run in sequence, each with one owned native child:

1. A request timeout far below real inference time: expects 408, the exact owned
   child reaped, then 529 on health and on the next request.
2. The exact owned child is killed by PID between requests: expects 500, then
   529 on health and on the next request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import httpx

Row = dict[str, Any]
MODEL_ID = "local-gemma-unridden-v1"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def payload(question_count: int) -> Row:
    question = {
        "type": "choice",
        "instructions": "Choose the explicitly stated flag color.",
        "criteria": {"green": "green", "red": "red"},
    }
    return {
        "model": MODEL_ID,
        "state": "The flag is green." + " note" * 1200 + " The flag is green.",
        "questions": {f"flag_{i}": question for i in range(question_count)},
    }


async def pid_gone(pid: int) -> bool:
    for _ in range(200):
        if not Path(f"/proc/{pid}").exists():
            return True
        await asyncio.sleep(0.05)
    return False


async def scenario(
    name: str,
    args: argparse.Namespace,
    *,
    request_timeout: float,
    kill_child: bool,
    expected_status: int,
    expected_code: str,
) -> Row:
    from unridden.api.app import ApiConfig, create_app

    app = create_app(
        ApiConfig(
            model_path=args.model,
            worker_path=args.worker,
            manifest_path=args.manifest,
            gpu=True,
            request_timeout=request_timeout,
        )
    )
    record: Row = {"scenario": name, "request_timeout": request_timeout}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://unridden-failures",
            timeout=None,
        ) as client,
    ):
        backend = app.state.service.backend
        pid = backend.pid
        assert isinstance(pid, int) and pid > 0
        assert Path(f"/proc/{pid}/exe").resolve() == args.worker.resolve()
        record["worker_pid"] = pid
        ready = await client.get("/health")
        assert ready.status_code == 200, ready.text
        if kill_child:
            warm = await client.post("/v1/decisions", json=payload(1))
            assert warm.status_code == 200, warm.text
            record["warm_request_status"] = warm.status_code
            # Exactly the owned child, by PID verified against /proc/<pid>/exe.
            os.kill(pid, signal.SIGKILL)
            assert await pid_gone(pid), "killed child still present"
        started = time.perf_counter()
        failed = await client.post("/v1/decisions", json=payload(8))
        record["failure_ms"] = (time.perf_counter() - started) * 1000
        record["failure_status"] = failed.status_code
        record["failure_body"] = failed.json()
        assert failed.status_code == expected_status, failed.text
        error = failed.json()["error"]
        assert error["code"] == expected_code, error
        assert "/home/" not in failed.text and "Traceback" not in failed.text
        record["child_reaped"] = await pid_gone(pid)
        assert record["child_reaped"], "owned child survived the failure"
        assert backend.pid is None and backend.ready is False
        health = await client.get("/health")
        record["health_after"] = [health.status_code, health.json()]
        assert health.status_code == 529, health.text
        after = await client.post("/v1/decisions", json=payload(1))
        record["next_request"] = [after.status_code, after.json()]
        assert after.status_code == 529, after.text
        assert after.json()["error"]["retryable"] is True
        record["stderr_tail_lines"] = len(backend.stderr_tail)
    print("SCENARIO_COMPLETE", name, flush=True)
    return record


async def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    native_logger = logging.getLogger("unridden.api.native")
    native_logger.setLevel(logging.DEBUG)
    log_paths = [args.out / "native-worker.log"]
    if args.extra_log is not None:
        log_paths.append(args.extra_log)
    for log_path in log_paths:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        native_logger.addHandler(handler)
    results: list[Row] = []
    state = "failed"
    try:
        results.append(
            await scenario(
                "timeout-408",
                args,
                request_timeout=0.05,
                kill_child=False,
                expected_status=408,
                expected_code="timeout",
            )
        )
        results.append(
            await scenario(
                "worker-killed-500",
                args,
                request_timeout=120.0,
                kill_child=True,
                expected_status=500,
                expected_code="internal_error",
            )
        )
        state = "complete"
        print("FAILURE_PATHS_COMPLETE", flush=True)
    except BaseException as error:
        print("FAILURE_PATHS_FAILED", type(error).__name__, flush=True)
        raise
    finally:
        write_json(args.out / "run.json", {"state": state, "scenarios": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--extra-log",
        type=Path,
        default=None,
        help="also append native worker debug lines to this file",
    )
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error(
            "--allow-gpu is required: run `gpu-policy check` first "
            "(GO = pass --allow-gpu)"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
