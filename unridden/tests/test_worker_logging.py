"""Worker failures and 5xx causes reach the server log, not just DEBUG."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unridden.api.cli import configure_logging
from unridden.api.native_backend import worker_line_level
from unridden.api.snapshots.backend import SnapshotNativeBackend
from unridden.api.snapshots.errors import SnapshotExecutionError, SnapshotRequestError
from unridden.tests.test_snapshots_app import (
    CONTEXT_BODY,
    FakeSnapshotBackend,
    client_for,
)


def test_only_worker_failure_lines_are_warnings() -> None:
    for line in (
        "REQUEST_FAILED invalid_request save needs an existing empty absolute dir",
        "PREFLIGHT_FAILED prompt exceeds context",
        "WORKER_FAILED Model load failed",
    ):
        assert worker_line_level(line) == logging.WARNING
    for line in ("llama_model_loader: loaded meta data", "", "request_failed lower"):
        assert worker_line_level(line) == logging.DEBUG


@pytest.mark.asyncio
async def test_snapshot_worker_failures_are_logged_as_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backend = SnapshotNativeBackend(
        worker_path=tmp_path / "worker",
        model_path=tmp_path / "model.gguf",
        manifest_path=tmp_path / "build.json",
    )
    reader = asyncio.StreamReader()
    reader.feed_data(b"llama_model_loader: chatter\n")
    reader.feed_data(b"REQUEST_FAILED invalid_request save needs an absolute dir\n")
    reader.feed_eof()
    backend._process = SimpleNamespace(pid=7, stderr=reader)  # type: ignore[assignment]
    with caplog.at_level(logging.DEBUG, logger="unridden"):
        await backend._drain_stderr()

    levels = {record.getMessage(): record.levelno for record in caplog.records}
    assert levels["snapshot worker pid=7: llama_model_loader: chatter"] == (
        logging.DEBUG
    )
    assert levels[
        "snapshot worker pid=7: REQUEST_FAILED invalid_request save needs an "
        "absolute dir"
    ] == (logging.WARNING)


class FailingCreate(FakeSnapshotBackend):
    async def create(self, **kwargs: Any) -> Any:
        raise SnapshotExecutionError("worker preflight rejected a request")


@pytest.mark.asyncio
async def test_a_500_logs_its_cause(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="unridden"):
        async with client_for(FailingCreate(), tmp_path) as client:
            response = await client.post("/v2/snapshots", json=CONTEXT_BODY)

    assert response.status_code == 500
    assert "request failed internally" in response.text
    assert any(
        record.levelno == logging.ERROR
        and "SnapshotExecutionError: worker preflight rejected a request"
        in record.getMessage()
        for record in caplog.records
    )


class RefusingCreate(FakeSnapshotBackend):
    def __init__(self) -> None:
        super().__init__()
        self.drops: list[str] = []

    async def create(self, **kwargs: Any) -> Any:
        raise SnapshotRequestError("prompt exceeds context", reason="budget")

    async def drop(self, snapshot_id: str) -> int:
        self.drops.append(snapshot_id)
        return await super().drop(snapshot_id)


@pytest.mark.asyncio
async def test_a_refused_create_sends_no_cleanup_drop(tmp_path: Path) -> None:
    # The worker created nothing, so a drop would only log a spurious
    # snapshot_not_found warning next to the real refusal.
    backend = RefusingCreate()
    async with client_for(backend, tmp_path) as client:
        response = await client.post("/v2/snapshots", json=CONTEXT_BODY)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "budget_error"
    assert backend.drops == []


@pytest.fixture
def restore_unridden_logger() -> Iterator[None]:
    logger = logging.getLogger("unridden")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    yield
    logger.handlers, logger.level, logger.propagate = saved[0], saved[1], saved[2]


@pytest.mark.usefixtures("restore_unridden_logger")
def test_serve_logging_writes_unridden_warnings_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("info")
    logging.getLogger("unridden.api.snapshots.native").warning("worker said no")
    logging.getLogger("unridden.api.snapshots.native").debug("load chatter")

    err = capsys.readouterr().err
    assert "WARNING unridden.api.snapshots.native: worker said no" in err
    assert "load chatter" not in err
