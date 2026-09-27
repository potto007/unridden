"""Qualify a snapshot worker profile against its release gates.

Drives the native `unridden-snapshot-worker` directly (no HTTP layer) over the
frozen `unridden/examples/api-v1-cases.json` corpus and checks the gates of
docs/superpowers/specs/2026-09-22-state-snapshots-design.md, "Verification and
release criteria":

1. capture integrity   - shapes, sizes, token coverage of H18/H30 and K/V
2. execution audit     - measured block-token work per operation, zero tokens
3. restore identity    - resident vs host restore vs disk restore in a fresh
                         process: same choices, label-prob delta <= 1e-5
4. promotion identity  - S18 promoted then asked == paired S30 asked (1e-5)
5. stock vs split      - stock uninterrupted 30-block graph vs the split graph
                         on identical tokens: prob delta and residual rel L2
                         <= 1e-3, equal decisions. Informational: a branch
                         (prefix + suffix chunking) against the same stock run
6. branch isolation    - A,B,A and reversed order, after failed requests;
                         parent blobs byte-identical before and after
7. rejections          - early readout, corrupt blob, budget overflow, prefix
                         mismatch

`--profile full-v1` (ADR 0009) runs one whole-model context whose only
checkpoint is the model's block count, so gates 4 and 5 do not apply. In their
place an informational row compares each branch with the same prompt run
fresh from an empty cache in one pass.

Tolerances are fixed here, before any run. A violated gate is a failure, never
a reason to relax it. The worker needs the GPU; run only with the user's
confirmation and with multi-GiB VRAM headroom.

    python scripts/unridden/qualify_snapshots.py \\
        --bundle build/snapshot-worker --model <gguf> --gpu --out outputs/snapshots \\
        [--profile full-v1]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from unridden.api.compiler import _options, _question_heading, render
from unridden.api.native.build import digest
from unridden.api.native.bundle import read_manifest, resolve_manifest_paths
from unridden.api.schema import DecisionRequest, Question

RESTORE_PROB_TOLERANCE = 1e-5
STOCK_PROB_TOLERANCE = 1e-3
STOCK_L2_TOLERANCE = 1e-3
PROMPT_VERSION = "unridden-gemma-context-v1"
SPLIT_PROFILE = "split18-30-v1"
FULL_PROFILE = "full-v1"
CONTEXT_INSTRUCTION = (
    "Use the supplied state to answer the request that follows it. "
    "Answer only what is asked."
)
ANSWER_PREFIX = "Answer:\n"
Row = dict[str, Any]
LONG_CASE: Row = {
    "id": "long-prefix-over-swa-window",
    "state": " ".join(
        f"Log line {index}: the customer asked about order {1000 + index} and was "
        "told a reply would follow within two business days."
        for index in range(50)
    ),
    "questions": {
        "topic": {
            "type": "choice",
            "instructions": "What do the log lines mostly concern?",
            "criteria": {"orders": "Questions about orders", "weather": None},
        },
        "promised_reply": {
            "type": "noul",
            "instructions": "The log says a reply was promised.",
        },
    },
}


class WorkerError(RuntimeError):
    def __init__(self, response: Row) -> None:
        super().__init__(f"{response.get('code')}: {response.get('message')}")
        self.code = str(response.get("code"))


class Worker:
    """Blocking JSONL client for one worker process."""

    def __init__(
        self,
        bundle: Path,
        model: Path,
        *,
        gpu: bool,
        reference: bool,
        profile: str = SPLIT_PROFILE,
    ) -> None:
        manifest_path = bundle / "build.json"
        manifest = read_manifest(manifest_path)
        executable, runtime_dir = resolve_manifest_paths(manifest, manifest_path)
        if digest(executable) != manifest["executable_sha256"]:
            raise ValueError("worker hash differs from manifest")
        self.runtime_sha256 = str(manifest["runtime_bundle_sha256"])
        command = [
            str(executable),
            "--model",
            str(model),
            "--model-sha256",
            model_digest(model),
            "--runtime-sha256",
            self.runtime_sha256,
            "--runtime-dir",
            str(runtime_dir),
            "--context",
            "2048",
            "--batch",
            "256",
            "--ubatch",
            "256",
            "--threads",
            "8",
        ]
        if gpu:
            command.append("--gpu")
        if reference:
            command.append("--reference")
        if profile != SPLIT_PROFILE:
            command.extend(["--profile", profile])
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(runtime_dir)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open("/tmp/unridden-snapshot-worker.stderr.log", "ab"),  # noqa: SIM115
            env=environment,
        )
        self.hello = self._read()
        if self.hello.get("protocol") != "unridden-snapshot-v1":
            raise RuntimeError(f"bad handshake {self.hello}")
        if self.hello.get("profile") != profile:
            raise RuntimeError(f"worker serves {self.hello.get('profile')}")
        self.labels: list[str] = list(self.hello["labels"])
        # The readout block: 30 in the split profile, the block count in full-v1.
        self.final = int(self.hello["n_layer"])
        self.counter = 0

    def _read(self) -> Row:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("worker closed stdout; see its stderr log")
        payload: Row = json.loads(line)
        return payload

    def call(self, request: Row) -> Row:
        assert self.process.stdin is not None
        self.counter += 1
        request = {**request, "id": f"q{self.counter}"}
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()
        response = self._read()
        if response.get("id") != request["id"]:
            raise RuntimeError("correlation id differs")
        if response.get("type") == "error":
            raise WorkerError(response)
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=60)


_MODEL_DIGESTS: dict[Path, str] = {}


def model_digest(model: Path) -> str:
    if model not in _MODEL_DIGESTS:
        _MODEL_DIGESTS[model] = digest(model)
    return _MODEL_DIGESTS[model]


# -- prompt construction (unridden-gemma-context-v1) --------------------------


def context_head(state: object) -> str:
    return f"{CONTEXT_INSTRUCTION}\n\nSTATE:\n{render(state)}\n\n"


def question_block(question: Question, labels: list[str]) -> tuple[str, list[str]]:
    option_ids, descriptions, _ = _options(question)
    used = labels[: len(option_ids)]
    lines = ["QUESTION:", _question_heading(question)]
    instructions = render(question.instructions)
    if instructions:
        lines.extend(["", "INSTRUCTIONS:", instructions])
    lines.extend(["", "OPTIONS:"])
    for label, option_id, description in zip(
        used, option_ids, descriptions, strict=True
    ):
        text = render(description)
        lines.append(
            f"{label}: {option_id} - {text}" if text else f"{label}: {option_id}"
        )
    lines.extend(["", "Reply with one option label only."])
    return "\n".join(lines), used


def context_messages(state: object) -> tuple[list[Row], int]:
    head = context_head(state)
    return [{"role": "user", "content": head}], len(head.encode("utf-8"))


def decision_messages(state: object, block: str) -> list[Row]:
    return [{"role": "user", "content": context_head(state) + block}]


# -- numerics ------------------------------------------------------------------


def softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    weights = [math.exp(item - top) for item in logits]
    total = sum(weights)
    return [item / total for item in weights]


def decode_vector(tensor: Row) -> list[float]:
    raw = base64.b64decode(tensor["base64"])
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def relative_l2(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector lengths differ")
    numerator = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
    denominator = math.sqrt(sum(y * y for y in b))
    return numerator / denominator if denominator else numerator


def prob_delta(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(softmax(a), softmax(b), strict=True))


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


class Gates:
    def __init__(self) -> None:
        self.rows: list[Row] = []

    def check(self, gate: str, case: str, passed: bool, **detail: Any) -> None:
        self.rows.append({"gate": gate, "case": case, "passed": passed, **detail})
        mark = "PASS" if passed else "FAIL"
        print(
            f"{mark} {gate} {case} {json.dumps(detail, default=str)[:240]}", flush=True
        )

    def summary(self) -> Row:
        by_gate: dict[str, Row] = {}
        for row in self.rows:
            entry = by_gate.setdefault(row["gate"], {"passed": 0, "failed": 0})
            entry["passed" if row["passed"] else "failed"] += 1
        return by_gate

    @property
    def ok(self) -> bool:
        return all(row["passed"] for row in self.rows if not row.get("informational"))


# -- the run -------------------------------------------------------------------


@contextmanager
def worker(args: argparse.Namespace, *, reference: bool = False) -> Iterator[Worker]:
    client = Worker(
        args.bundle,
        args.model,
        gpu=args.gpu,
        # The full-v1 snapshot path already is the stock graph.
        reference=reference and args.profile == SPLIT_PROFILE,
        profile=args.profile,
    )
    try:
        yield client
    finally:
        client.close()


def evaluate(client: Worker, snapshot: str, questions: list[Row]) -> list[Row]:
    payload = [
        {
            "id": item["id"],
            "messages": item["messages"],
            "answer_prefix": ANSWER_PREFIX,
            "labels": item["labels"],
            "prompt_version": PROMPT_VERSION,
            "save_as": None,
        }
        for item in questions
    ]
    response = client.call(
        {
            "type": "evaluate",
            "snapshot_id": snapshot,
            "readout_blocks": client.final,
            "questions": payload,
        }
    )
    assert response["generated_tokens"] == 0
    rows: list[Row] = response["questions"]
    return rows


def save_blobs(client: Worker, snapshot: str, root: Path) -> dict[str, str]:
    directory = Path(tempfile.mkdtemp(dir=root))
    files = client.call(
        {"type": "save", "snapshot_id": snapshot, "directory": str(directory)}
    )
    return {name: entry["sha256"] for name, entry in files["files"].items()} | {
        "_dir": str(directory)
    }


def promotion_and_stock(
    client: Worker,
    gates: Gates,
    name: str,
    index: int,
    messages: list[Row],
    content_bytes: int,
    n: int,
    questions: list[Row],
    resident: list[Row],
) -> None:
    """Gates 4 (promotion identity) and 5 (stock vs split): split profile only."""
    # 4. promotion identity
    only18 = f"c{index}_solo18"
    solo = client.call(
        {
            "type": "create",
            "messages": messages,
            "answer_prefix": "",
            "freeze": {"kind": "context", "content_bytes": content_bytes},
            "checkpoints": {"18": only18},
            "labels": None,
            "top_logits": 0,
        }
    )
    gates.check(
        "execution_audit",
        f"{name}/create18",
        solo["block_tokens"] == {"lower": n, "upper": 0},
    )
    promoted = client.call(
        {"type": "promote", "snapshot_id": only18, "new_id": f"c{index}_prom"}
    )
    gates.check(
        "execution_audit",
        f"{name}/promote",
        promoted["block_tokens"] == {"lower": 0, "upper": n},
    )
    via18 = evaluate(client, f"c{index}_prom", questions)
    for paired, promoted_row in zip(resident, via18, strict=True):
        delta = prob_delta(paired["label_logits"], promoted_row["label_logits"])
        gates.check(
            "promotion_identity",
            f"{name}/{paired['id']}",
            delta <= RESTORE_PROB_TOLERANCE
            and argmax(paired["label_logits"]) == argmax(promoted_row["label_logits"]),
            delta=delta,
        )

    # 5. stock vs split, identical tokens and chunking
    for item, branch in zip(questions, resident, strict=True):
        stock = client.call(
            {
                "type": "reference",
                "messages": item["messages"],
                "answer_prefix": ANSWER_PREFIX,
                "labels": item["labels"],
                "top_logits": 0,
            }
        )
        readout_id = f"c{index}_{item['id']}_ro"
        split = client.call(
            {
                "type": "create",
                "messages": item["messages"],
                "answer_prefix": ANSWER_PREFIX,
                "freeze": {"kind": "readout"},
                "checkpoints": {"30": readout_id},
                "labels": item["labels"],
                "top_logits": 0,
            }
        )
        tokens = int(split["tokens"])
        split_h18 = client.call(
            {
                "type": "vectors",
                "snapshot_id": readout_id,
                "which": "h18",
                "row_begin": tokens - 1,
                "row_end": tokens,
            }
        )
        split_h30 = client.call(
            {
                "type": "vectors",
                "snapshot_id": readout_id,
                "which": "h30",
                "row_begin": tokens - 1,
                "row_end": tokens,
            }
        )
        stock_logits = stock["label_logits"]
        split_logits = split["readout"]["label_logits"]
        delta = prob_delta(stock_logits, split_logits)
        l2_18 = relative_l2(
            decode_vector(split_h18["tensor"]),
            decode_vector(stock["vectors"]["h18_last"]),
        )
        l2_30 = relative_l2(
            decode_vector(split_h30["tensor"]),
            decode_vector(stock["vectors"]["h30_last"]),
        )
        gates.check(
            "stock_vs_split",
            f"{name}/{item['id']}",
            tokens == stock["tokens"]
            and delta <= STOCK_PROB_TOLERANCE
            and l2_18 <= STOCK_L2_TOLERANCE
            and l2_30 <= STOCK_L2_TOLERANCE
            and argmax(stock_logits) == argmax(split_logits),
            delta=delta,
            l2_h18=l2_18,
            l2_h30=l2_30,
        )
        branch_delta = prob_delta(stock_logits, branch["label_logits"])
        gates.rows.append(
            {
                "gate": "branch_vs_stock",
                "case": f"{name}/{item['id']}",
                "passed": branch_delta <= STOCK_PROB_TOLERANCE
                and argmax(stock_logits) == argmax(branch["label_logits"]),
                "informational": True,
                "delta": branch_delta,
            }
        )
        client.call({"type": "drop", "snapshot_id": readout_id})
    for snapshot in (only18, f"c{index}_prom"):
        client.call({"type": "drop", "snapshot_id": snapshot})


def branch_vs_fresh(
    client: Worker,
    gates: Gates,
    name: str,
    index: int,
    questions: list[Row],
    resident: list[Row],
) -> None:
    """full-v1, informational: each branch against its prompt run fresh."""
    for item, branch in zip(questions, resident, strict=True):
        readout_id = f"c{index}_{item['id']}_ro"
        fresh = client.call(
            {
                "type": "create",
                "messages": item["messages"],
                "answer_prefix": ANSWER_PREFIX,
                "freeze": {"kind": "readout"},
                "checkpoints": {str(client.final): readout_id},
                "labels": item["labels"],
                "top_logits": 0,
            }
        )
        fresh_logits = fresh["readout"]["label_logits"]
        delta = prob_delta(fresh_logits, branch["label_logits"])
        gates.rows.append(
            {
                "gate": "branch_vs_fresh",
                "case": f"{name}/{item['id']}",
                "passed": delta <= STOCK_PROB_TOLERANCE
                and argmax(fresh_logits) == argmax(branch["label_logits"]),
                "informational": True,
                "delta": delta,
            }
        )
        client.call({"type": "drop", "snapshot_id": readout_id})


def run(args: argparse.Namespace) -> Row:
    corpus = json.loads(Path(args.cases).read_text())
    cases = corpus["cases"][: args.limit] if args.limit else corpus["cases"]
    # A prefix beyond the 1,024-token sliding window, where SWA-masked cells
    # exist and a restore must still reproduce the live cache exactly.
    cases = [*cases, LONG_CASE]
    gates = Gates()
    scratch = Path(tempfile.mkdtemp(prefix="unridden-qualify-"))
    persisted: list[Row] = []
    width = 0
    started = time.monotonic()
    full = args.profile == FULL_PROFILE
    with worker(args, reference=True) as client:
        width = int(client.hello["n_embd"])
        final = client.final
        # The block an early (always refused) readout asks for.
        early = final - 1 if full else 18
        for index, case in enumerate(cases):
            name = str(case["id"])
            request = DecisionRequest.model_validate(
                {
                    "model": corpus["model"],
                    "state": case["state"],
                    "questions": case["questions"],
                }
            )
            messages, content_bytes = context_messages(request.state)
            questions: list[Row] = []
            for question_id, question in request.questions.items():
                block, used = question_block(question, client.labels)
                questions.append(
                    {
                        "id": question_id,
                        "labels": used,
                        "messages": decision_messages(request.state, block),
                    }
                )
            # s30 is the final-block snapshot in either profile.
            s18, s30 = f"c{index}_18", f"c{index}_30"
            created = client.call(
                {
                    "type": "create",
                    "messages": messages,
                    "answer_prefix": "",
                    "freeze": {"kind": "context", "content_bytes": content_bytes},
                    "checkpoints": (
                        {str(final): s30} if full else {"18": s18, "30": s30}
                    ),
                    "labels": None,
                    "top_logits": 5,
                }
            )
            n = int(created["tokens"])

            # 1. capture integrity
            rows = {row["snapshot_id"]: row for row in created["snapshots"]}
            info30 = client.call({"type": "inspect", "snapshot_id": s30})
            h30 = client.call(
                {
                    "type": "vectors",
                    "snapshot_id": s30,
                    "which": "h30",
                    "row_begin": n - 1,
                    "row_end": n,
                }
            )
            if full:
                # One range: whole-model K/V and the final residual, nothing else.
                gates.check(
                    "capture_integrity",
                    name,
                    rows[s30]["bytes"]["h30"] == n * width * 4
                    and rows[s30]["bytes"]["h18"] == 0
                    and rows[s30]["bytes"]["upper_kv"] == 0
                    and rows[s30]["bytes"]["lower_kv"] > 0
                    and rows[s30]["completed_blocks"] == final
                    and rows[s30]["parent"] is None
                    and len(info30["token_ids"]) == n
                    and h30["tensor"]["representation"]
                    == "raw_residual_after_final_block"
                    and len(decode_vector(h30["tensor"])) == width,
                    tokens=n,
                    bytes=rows[s30]["bytes"],
                )
            else:
                info18 = client.call({"type": "inspect", "snapshot_id": s18})
                h18 = client.call(
                    {
                        "type": "vectors",
                        "snapshot_id": s18,
                        "which": "h18",
                        "row_begin": max(0, n - 4),
                        "row_end": n,
                    }
                )
                gates.check(
                    "capture_integrity",
                    name,
                    rows[s18]["bytes"]["h18"] == n * width * 4
                    and rows[s30]["bytes"]["h30"] == n * width * 4
                    and rows[s18]["bytes"]["upper_kv"] == 0
                    and rows[s30]["bytes"]["upper_kv"] > 0
                    and rows[s30]["parent"] == s18
                    and len(info18["token_ids"]) == n == len(info30["token_ids"])
                    and info18["token_ids"] == info30["token_ids"]
                    and h18["tensor"]["shape"] == [min(4, n), width]
                    and len(decode_vector(h30["tensor"])) == width,
                    tokens=n,
                    bytes18=rows[s18]["bytes"],
                    bytes30=rows[s30]["bytes"],
                )

            # 2. execution audit (creation); full-v1 counts every block as lower
            gates.check(
                "execution_audit",
                f"{name}/create",
                created["block_tokens"] == {"lower": n, "upper": 0 if full else n}
                and created["generated_tokens"] == 0,
                block_tokens=created["block_tokens"],
            )

            # 3. restore identity: resident, then host restore
            resident = evaluate(client, s30, questions)
            # Each question is asked again right after another snapshot took
            # the contexts, so every one of them restores from host bytes.
            restored = []
            for number, item in enumerate(questions):
                evictor = f"c{index}_other{number}"
                client.call(
                    {
                        "type": "create",
                        "messages": messages,
                        "answer_prefix": "",
                        "freeze": {"kind": "context", "content_bytes": content_bytes},
                        "checkpoints": {str(final) if full else "18": evictor},
                        "labels": None,
                        "top_logits": 0,
                    }
                )
                restored.extend(evaluate(client, s30, [item]))
                client.call({"type": "drop", "snapshot_id": evictor})
            for first, again in zip(resident, restored, strict=True):
                audit = first["snapshot"]
                gates.check(
                    "execution_audit",
                    f"{name}/{first['id']}",
                    audit["block_tokens"]["lower"]
                    == audit["suffix_tokens"]
                    == first["processed_tokens"]
                    and audit["block_tokens"]["upper"]
                    == (0 if full else audit["suffix_tokens"])
                    and first["reused_tokens"] == n,
                    snapshot=audit,
                )
                delta = prob_delta(first["label_logits"], again["label_logits"])
                gates.check(
                    "restore_identity",
                    f"{name}/{first['id']}/host",
                    again["snapshot"]["restore"] == "host"
                    and delta <= RESTORE_PROB_TOLERANCE
                    and argmax(first["label_logits"]) == argmax(again["label_logits"]),
                    delta=delta,
                    exact=first["label_logits"] == again["label_logits"],
                )

            if full:
                branch_vs_fresh(client, gates, name, index, questions, resident)
            else:
                promotion_and_stock(
                    client,
                    gates,
                    name,
                    index,
                    messages,
                    content_bytes,
                    n,
                    questions,
                    resident,
                )

            # 6. branch isolation
            before = save_blobs(client, s30, scratch)
            if len(questions) >= 2:
                a, b = questions[0], questions[1]
                aba = [evaluate(client, s30, [q])[0] for q in (a, b, a)]
                ba = evaluate(client, s30, [b, a])
                gates.check(
                    "branch_isolation",
                    f"{name}/ABA",
                    aba[0]["label_logits"] == aba[2]["label_logits"]
                    and ba[1]["label_logits"] == aba[0]["label_logits"]
                    and ba[0]["label_logits"] == aba[1]["label_logits"],
                )
            for failing in (
                {
                    "type": "evaluate",
                    "snapshot_id": s30,
                    "readout_blocks": early,
                    "questions": [],
                },
                {
                    "type": "evaluate",
                    "snapshot_id": s30,
                    "readout_blocks": final,
                    "questions": [
                        {
                            "id": "x",
                            "messages": [{"role": "user", "content": "unrelated"}],
                            "answer_prefix": ANSWER_PREFIX,
                            "labels": client.labels[:2],
                            "prompt_version": PROMPT_VERSION,
                            "save_as": None,
                        }
                    ],
                },
            ):
                try:
                    client.call(failing)
                    gates.check(
                        "rejections", f"{name}/{failing['readout_blocks']}", False
                    )
                except WorkerError as error:
                    expected = (
                        "capability_unavailable"
                        if failing["readout_blocks"] == early
                        else "snapshot_prefix_mismatch"
                    )
                    gates.check(
                        "rejections",
                        f"{name}/{expected}",
                        error.code == expected,
                        code=error.code,
                    )
            after_failure = evaluate(client, s30, questions)
            gates.check(
                "branch_isolation",
                f"{name}/after_failures",
                [row["label_logits"] for row in after_failure]
                == [row["label_logits"] for row in restored],
            )
            after = save_blobs(client, s30, scratch)
            gates.check(
                "branch_isolation",
                f"{name}/parent_immutable",
                {k: v for k, v in before.items() if k != "_dir"}
                == {k: v for k, v in after.items() if k != "_dir"},
            )
            persisted.append(
                {
                    "case": name,
                    "dir": after["_dir"],
                    "files": after,
                    "questions": questions,
                    "expected": [row["label_logits"] for row in restored],
                }
            )
            for snapshot in (s30,) if full else (s18, s30):
                client.call({"type": "drop", "snapshot_id": snapshot})

        # 7. rejections that need no case
        huge = {"role": "user", "content": "word " * 5000}
        try:
            client.call(
                {
                    "type": "create",
                    "messages": [huge],
                    "answer_prefix": "",
                    "freeze": {"kind": "readout"},
                    "checkpoints": {str(final): "huge"},
                    "labels": None,
                    "top_logits": 0,
                }
            )
            gates.check("rejections", "budget", False)
        except WorkerError as error:
            gates.check("rejections", "budget", error.code == "invalid_request")

    # 3b. disk restore in a fresh process
    with worker(args) as fresh:
        for entry in persisted:
            files = {k: v for k, v in entry["files"].items() if k != "_dir"}
            snapshot = f"disk_{entry['case']}"
            fresh.call(
                {
                    "type": "load",
                    "snapshot_id": snapshot,
                    "directory": entry["dir"],
                    "files": files,
                }
            )
            reloaded = evaluate(fresh, snapshot, entry["questions"])
            deltas = [
                prob_delta(expected, row["label_logits"])
                for expected, row in zip(entry["expected"], reloaded, strict=True)
            ]
            gates.check(
                "restore_identity",
                f"{entry['case']}/disk_fresh_process",
                max(deltas) <= RESTORE_PROB_TOLERANCE,
                max_delta=max(deltas),
                exact=all(d == 0.0 for d in deltas),
            )
        if persisted:
            tampered = Path(persisted[0]["dir"]) / "h30.f32"
            data = bytearray(tampered.read_bytes())
            data[0] ^= 0xFF
            tampered.write_bytes(bytes(data))
            files = {k: v for k, v in persisted[0]["files"].items() if k != "_dir"}
            try:
                fresh.call(
                    {
                        "type": "load",
                        "snapshot_id": "tampered",
                        "directory": persisted[0]["dir"],
                        "files": files,
                    }
                )
                gates.check("rejections", "corrupt_blob", False)
            except WorkerError as error:
                gates.check(
                    "rejections", "corrupt_blob", error.code == "integrity_error"
                )

    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "profile": args.profile,
        "model": args.model.name,
        "tolerances": {
            "restore_prob": RESTORE_PROB_TOLERANCE,
            "stock_prob": STOCK_PROB_TOLERANCE,
            "stock_relative_l2": STOCK_L2_TOLERANCE,
        },
        "cases": len(cases),
        "elapsed_s": time.monotonic() - started,
        "summary": gates.summary(),
        "passed": gates.ok,
        "rows": gates.rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path("unridden/examples/api-v1-cases.json")
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--profile", choices=[SPLIT_PROFILE, FULL_PROFILE], default=SPLIT_PROFILE
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/snapshots"))
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.model = args.model.resolve()
    result = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"qualification-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "passed": result["passed"],
                "report": str(path),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
