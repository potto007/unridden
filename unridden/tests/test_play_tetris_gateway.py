"""The demo keeps its game policy and delegates ordinary snapshot work."""

from __future__ import annotations

import argparse
import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from scripts.unridden import play_tetris as demo


def test_gateway_payload_keeps_questions_and_state_without_snapshot_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = demo.Unridden("http://gateway")
    requests: list[tuple[str, str, Any]] = []

    def ask(method: str, path: str, body: Any = None) -> Any:
        requests.append((method, path, body))
        return {
            "answers": {
                key: {"type": "noul", "value": True} for key in body["questions"]
            },
            "usage": {"input_tokens": 2},
        }

    monkeypatch.setattr(client, "_req", ask)
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Question {i}"} for i in range(33)
    }
    assert set(client.decide("exact stable state", questions, keep=True)) == set(
        questions
    )
    client.flush()
    client.close()
    assert [path for _, path, _ in requests] == ["/v1/decisions"]
    assert all(body["state"] == "exact stable state" for _, _, body in requests)
    assert [len(body["questions"]) for _, _, body in requests] == [33]
    assert requests[0][2]["questions"]["q0"] == questions["q0"]
    assert not client.pending and not client.kept
    assert client.calls == 1


def test_default_client_preserves_automatic_cookie_between_http_calls() -> None:
    cookies: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            cookies.append(self.headers.get("Cookie"))
            self.rfile.read(int(self.headers["Content-Length"]))
            raw = b'{"answers": {}, "usage": {"input_tokens": 0}}'
            self.send_response(200)
            self.send_header("Set-Cookie", "unridden_context=test; Path=/v1; HttpOnly")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = demo.Unridden(f"http://127.0.0.1:{server.server_port}")
    try:
        for _ in range(2):
            client.decide("state", {"q": {"type": "noul", "instructions": "Q?"}})
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert cookies == [None, "unridden_context=test"]
    assert len(client.cookies) == 0


def test_gateway_client_leaves_busy_handling_to_the_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = demo.Unridden("http://gateway")
    attempts = 0

    def refuse(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "http://gateway/v1/decisions",
            429,
            "busy",
            None,  # type: ignore[arg-type]
            io.BytesIO(b'{"error":{"code":"busy"}}'),
        )

    monkeypatch.setattr(client.opener, "open", refuse)
    with pytest.raises(RuntimeError):
        client.decide("state", {"q": {"type": "noul", "instructions": "Q?"}})
    assert attempts == 1


@pytest.mark.parametrize(
    "arguments,api,port",
    [
        ([], "gateway", 8092),
        (["--api", "v2"], "v2", 8091),
        (["--api", "v1"], "v1", 8090),
    ],
)
def test_cli_selects_gateway_by_default_and_preserves_native_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    api: str,
    port: int,
) -> None:
    def play(args: argparse.Namespace) -> dict[str, str]:
        return {"api": args.api, "url": args.url}

    monkeypatch.setattr(demo, "play", play)
    monkeypatch.setattr("sys.argv", ["play_tetris.py", *arguments])
    demo.main()
    assert json.loads(capsys.readouterr().out) == {
        "api": api,
        "url": f"http://127.0.0.1:{port}",
    }
