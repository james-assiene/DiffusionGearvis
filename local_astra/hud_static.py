"""Serving ``hud/`` on loopback, and the URL a human can paste (T-0033 criteria 16 and 18).

A static file server is a hole unless it is written as one that refuses: the path is
percent-decoded first and then **resolved** and checked against the root (an ``os.path.join`` of
a request path serves ``/../../local_astra/config.py``), write methods are refused, the socket is
bound to ``127.0.0.1`` so a non-loopback peer cannot reach it at all, and a taken port is an
error — never a move to another one. A page on a port nobody chose is a page talking to nothing
(T-0028 criterion 19, applied to the second listener).

This shares nothing with ``conductor/hud_server.py``, which is the WebSocket endpoint.
"""

from __future__ import annotations

import math
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from local_astra.config import Services

#: The directory served: the page T-0008 ships.
HUD_DIR = Path(__file__).resolve().parents[1] / "hud"

#: Loopback, always. Not a parameter: a static server that can be told to bind elsewhere is an
#: egress waiting for a typo (CLAUDE.md).
HOST = "127.0.0.1"

INDEX = "index.html"


def page_urls(services: Services) -> tuple[str, ...]:
    """The URLs the page needs, query string and all: the real share and the headless variant."""
    ws = f"ws://{HOST}:{services.conductor_port}/v1/realtime"
    base = f"http://{HOST}:{services.hud_port}/"
    return (f"{base}?ws={ws}", f"{base}?source=canvas&ws={ws}")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "astra-hud"
    # T-0165 C8: the headers and the body go out as two writes; with Nagle on, the second waits
    # for the client's delayed ACK (~40 ms on Linux) on every keep-alive request after the first.
    disable_nagle_algorithm = True

    @property
    def root(self) -> Path:
        server = self.server
        assert isinstance(server, _Server)
        return server.root

    def _resolved(self) -> Path | None:
        """The file this request asks for, or ``None`` if it points outside the root.

        ``..`` survives percent-encoding, so decode first and resolve after; then compare the
        resolved path to the resolved root. Symlinks are resolved too, so a link out of ``hud/``
        is refused for the same reason.
        """
        requested = unquote(urlsplit(self.path).path)
        root = self.root.resolve()
        candidate = (root / requested.lstrip("/")).resolve()
        if candidate != root and root not in candidate.parents:
            return None
        if candidate.is_dir():
            candidate = candidate / INDEX
            if candidate.resolve() != candidate:  # a symlinked index is still a way out
                return None
        return candidate if candidate.is_file() else None

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        target = self._resolved()
        if target is None:
            self._send(404, b"not found\n", "text/plain")
            return
        guessed, _encoding = mimetypes.guess_type(target.name)
        self._send(200, target.read_bytes(), guessed or "application/octet-stream")

    def do_HEAD(self) -> None:
        target = self._resolved()
        self._send(404 if target is None else 200, b"", "text/plain")

    def _refuse(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self._send(405, b"the HUD page is served read-only\n", "text/plain")

    do_POST = _refuse
    do_PUT = _refuse
    do_DELETE = _refuse
    do_PATCH = _refuse

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # the CLI prints the URL; request logging is noise on a static page


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # criterion 18: a taken port is an error, never a quiet move

    def __init__(self, address: tuple[str, int], root: Path) -> None:
        self.root = root
        super().__init__(address, _Handler)


class HudStaticServer:
    """``hud/`` on ``127.0.0.1:<port>``. Binds in ``__init__`` — a taken port raises ``OSError``.

    ``poll_interval`` is how often ``serve_forever`` checks for :meth:`close`'s stop request, so
    it bounds how long ``close()`` waits. The default is the stdlib's 0.5 s, which is what the CLI
    ships; tests pass a short one (T-0110c). A value that is not finite and positive raises
    ``ValueError`` before the socket binds.
    """

    def __init__(self, root: Path | str, port: int = 0, *, poll_interval: float = 0.5) -> None:
        if not (math.isfinite(poll_interval) and poll_interval > 0):
            raise ValueError(f"poll_interval must be finite and > 0; got {poll_interval!r}")
        self.root = Path(root)
        self._server = _Server((HOST, int(port)), self.root)
        bound = self._server.server_address
        self.address: tuple[str, int] = (str(bound[0]), int(bound[1]))
        self.port = self.address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": poll_interval},
            name="hud-static",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def serve_forever(self) -> None:
        """Block the caller until :meth:`close` (the serving itself is on the thread)."""
        self._thread.join()

    def close(self) -> None:
        """Stop serving and unbind — once, and even if an interrupt lands inside (T-0040).

        Ctrl-C reaches the whole foreground process group, so a session started through ``uv``
        sees two SIGINTs and the second used to land in ``shutdown()``'s ``Event.wait()``,
        leaving the serving thread alive and the port bound (logs/t0039-live-82d8462/run.log).
        The unbinding therefore runs from a ``finally``, and the stop is asked for again from a
        helper thread — no signal is delivered there, so that wait cannot be interrupted twice.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._stop_serving()
        finally:
            self._server.server_close()
            self._thread.join(timeout=5)

    def _stop_serving(self) -> None:
        """Ask ``serve_forever`` to return, absorbing an interrupt landing in its wait."""
        try:
            self._server.shutdown()
        except KeyboardInterrupt:
            # The interrupt *is* the stop request, and it arrived while we were already
            # stopping: ask again off the main thread, where no signal handler runs.
            stopper = threading.Thread(
                target=self._server.shutdown, name="hud-static-stop", daemon=True
            )
            stopper.start()
            stopper.join(timeout=5)

    def __enter__(self) -> HudStaticServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
