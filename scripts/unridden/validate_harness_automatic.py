"""Live automatic capture/reuse/isolation/recovery; uses only owned contexts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx


def main() -> None:
    hello = json.loads(Path("unridden/examples/hello.json").read_text())
    handles: set[str] = set()
    with (
        httpx.Client(base_url="http://127.0.0.1:8092", timeout=180) as one,
        httpx.Client(base_url="http://127.0.0.1:8092", timeout=180) as two,
        httpx.Client(base_url="http://127.0.0.1:8091", timeout=180) as backend,
    ):

        def ask(client: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
            response = client.post("/v1/decisions", json=body)
            response.raise_for_status()
            handle = client.cookies.get("unridden_context")
            assert handle is not None
            handles.add(handle)
            return dict(response.json())

        try:
            first = ask(one, hello)
            assert first["context_usage"]["cache"] == "miss"
            first_parent = first["snapshot_usage"]["requested_parent"]
            second = ask(one, hello)
            assert second["context_usage"]["cache"] == "hit"
            assert second["context_usage"]["capture_ms_this_request"] == 0
            assert second["snapshot_usage"]["requested_parent"] == first_parent
            assert second["answers"] == first["answers"]
            other = ask(two, hello)
            assert other["context_usage"]["cache"] == "miss"
            assert other["snapshot_usage"]["requested_parent"] != first_parent
            ask(
                one,
                {
                    "state": hello["state"],
                    "questions": {
                        "other": {
                            "type": "noul",
                            "instructions": "Does this concern astronomy?",
                        }
                    },
                },
            )
            assert ask(one, hello)["answers"] == first["answers"]
            backend.delete("/v2/snapshots/" + str(first_parent)).raise_for_status()
            restored = ask(one, hello)
            assert restored["context_usage"]["capture_reason"] == "backend_lost"
            assert restored["answers"] == first["answers"]
            changed = ask(
                one, hello | {"instructions": "Consider the supplied customer message."}
            )
            assert (
                changed["context_usage"]["capture_reason"] == "state_or_profile_changed"
            )
            two.cookies.clear()
            stateless = ask(two, hello)
            assert stateless["context_usage"]["cache"] == "miss"
            question_id, question = next(iter(hello["questions"].items()))
            many = {f"q{i}": question for i in range(34)}
            bundled = ask(one, {"state": hello["state"], "questions": many})
            assert bundled["execution"]["decision_batches"] == 2
            assert bundled["context_usage"]["successful_decisions"] == 1
            assert bundled["usage"]["output_tokens"] == 0
            assert len(bundled["answers"]) == 34
            assert all(
                answer == first["answers"][question_id]
                for answer in bundled["answers"].values()
            )
            direct_answers = {}
            metadata = backend.get(
                "/v2/snapshots/" + bundled["snapshot_usage"]["requested_parent"]
            )
            metadata.raise_for_status()
            items = list(many.items())
            for offset in range(0, len(items), 32):
                direct = backend.post(
                    "/v2/decisions",
                    json={
                        "snapshot": {
                            "id": bundled["snapshot_usage"]["requested_parent"],
                            "relationship": "followup",
                        },
                        "questions": dict(items[offset : offset + 32]),
                        "readout": {
                            "completed_blocks": metadata.json()["completed_blocks"]
                        },
                    },
                )
                direct.raise_for_status()
                direct_answers.update(direct.json()["answers"])
            assert bundled["answers"] == direct_answers
            result = {
                "automatic_first_capture": True,
                "automatic_resume": True,
                "client_isolation": True,
                "question_branch_identity": True,
                "lost_snapshot_recovery": True,
                "instruction_invalidation": True,
                "stateless_request_reports_miss": True,
                "independent_bundle_questions": 34,
                "independent_bundle_backend_batches": 2,
                "independent_bundle_matches_native": True,
            }
        finally:
            for handle in handles:
                response = one.delete("/v1/contexts/" + handle)
                assert response.status_code in {200, 404}, response.text
            backend.get("/health").raise_for_status()
    path = Path("outputs/harness-check/automatic-live-check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
