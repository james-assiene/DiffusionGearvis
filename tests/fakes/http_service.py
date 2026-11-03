"""A loopback HTTP stand-in for the three services the doctor probes (T-0033 criterion 3).

Routes are a table: ``{path: (status, body)}``. Everything not in the table is 404, which is the
silent failure criterion 3 exists for — a live process with no ASR route. Binding is always
``127.0.0.1:0`` (the kernel picks the port), so nothing here can be reached off the box and no
test needs a fixed port.

The three ``*_routes`` helpers describe what the **real** services do; the contract test
``tests/contract/test_service_health_contract.py`` makes the same assertions against these and
against the real services under ``-m gpu`` (``.claude/rules/tests.md``: a behaviour added to a
fake needs a real-service test in the same PR).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

#: ``serve_forever``'s poll interval: how long ``close()`` can wait for the serving thread to notice shutdown.
POLL_INTERVAL_S = 0.01


class StubService:
    """One HTTP service on loopback with a fixed route table. Use as a context manager."""

    def __init__(self, routes: dict[str, tuple[int, str]], *, name: str = "stub") -> None:
        self.routes = dict(routes)
        self.name = name
        self.requests: list[tuple[str, str, int]] = []  # (method, path, content-length)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _respond(self) -> None:
                path = urlsplit(self.path).path
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                stub.requests.append((self.command, self.path, length))
                status, body = stub.routes.get(path, (404, "not found"))
                payload = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _respond
            do_POST = _respond
            do_PUT = _respond

            def log_message(self, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        # T-0110 part C: shutdown() waits for serve_forever's next poll; the default 0.5 s made every close() cost
        # up to half a second. A request is served the moment it arrives whatever the interval.
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": POLL_INTERVAL_S},
            daemon=True,
        )
        self._thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def paths(self) -> list[str]:
        return [urlsplit(path).path for _method, path, _length in self.requests]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> StubService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def vllm_routes(*, healthy: bool = True, model: str | None = None) -> dict[str, tuple[int, str]]:
    """vLLM: ``GET /health`` -> 200 with an empty body; ``GET /v1/models`` lists the served id."""
    served = model or "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
    listing = json.dumps({"object": "list", "data": [{"id": served, "object": "model"}]})
    return {
        "/health": (200 if healthy else 503, "" if healthy else "engine dead"),
        "/v1/models": (200, listing),
        "/models": (200, listing),
    }


def tts_routes(*, healthy: bool = True) -> dict[str, tuple[int, str]]:
    """Pocket TTS answers ``/health`` with a body that says ``healthy`` when it is."""
    return {"/health": (200, '{"status":"healthy"}' if healthy else '{"status":"starting"}')}


def stt_routes(*, mounted: bool = True) -> dict[str, tuple[int, str]]:
    """moshi-server has no health route.

    ``GET /api/asr-streaming?auth_id=public_token`` on a mounted route is a failed WebSocket
    upgrade — a 4xx — which is what proves the route exists. ``mounted=False`` is the silent
    failure: a live process answering 200 on ``/`` and 404 on the ASR path.
    """
    routes = {"/": (200, "moshi-server")}
    if mounted:
        routes["/api/asr-streaming"] = (400, "expected a WebSocket upgrade")
    return routes
