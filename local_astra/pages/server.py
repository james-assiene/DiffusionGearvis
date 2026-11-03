"""``StartPageServer``: the stdlib static server behind the start pages (T-0111g A4).

- Binds ``127.0.0.1`` on an ephemeral port in ``start()`` (constructing binds nothing) and serves on a daemon thread;
  ``origin`` is ``http://127.0.0.1:<port>``. ``stop()`` shuts down, closes the socket and joins the thread.
- GET and HEAD of regular files under ``static_dir`` only; ``/`` (and a directory with one) is ``index.html``. Every
  other method is 405. The request target is percent-decoded exactly once, then a path with a ``..`` or ``.``
  segment, an empty segment (``//``, an absolute path), a backslash or a NUL is 404, and so is anything whose
  real path (symlinks resolved) is not a file inside ``static_dir``. A directory is never listed. The server never
  redirects (the agent browser's allow-list aborts redirects anyway, ``computer/runtime.py``).
- It writes no file and logs nothing (no access log, so no query string anywhere). A target the filesystem rejects
  (``OSError`` from ``resolve`` or the read, e.g. a name too long) is a 404, and ``handle_error`` prints no
  traceback.
"""

from __future__ import annotations

import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

__all__ = ["STATIC_DIR", "StartPageServer"]

#: The package's own pages.
STATIC_DIR = Path(__file__).resolve().parent / "static"
_HOST = "127.0.0.1"
_METHODS = frozenset({"GET", "HEAD"})
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
_STOP_TIMEOUT_S = 5.0


def resolve(root: Path, target: str) -> Path | None:
    """The regular file under ``root`` that a raw request target names, or ``None`` (a 404). ``root`` is a real
    path. Pure apart from reading the filesystem's metadata."""
    raw = target.split("?", 1)[0].split("#", 1)[0]
    path = unquote(raw, errors="strict")
    if not path.startswith("/") or "\\" in path or "\x00" in path:
        return None
    segments = path[1:].split("/")
    directory = segments[-1] == ""
    if directory:
        segments = segments[:-1]
    if any(s in ("", ".", "..") for s in segments):
        return None
    candidate = root.joinpath(*segments)
    real = Path(os.path.realpath(candidate))
    if real != root and not real.is_relative_to(root):
        return None
    if real.is_dir():
        index = Path(os.path.realpath(real / "index.html"))
        if not index.is_relative_to(root) or not index.is_file():
            return None
        return index
    if directory or not real.is_file():
        return None
    return real


def _handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AstraStartPages"
        sys_version = ""

        def parse_request(self) -> bool:
            if not super().parse_request():
                return False
            if self.command not in _METHODS:
                self.close_connection = True
                self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
                self.send_header("Allow", "GET, HEAD")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False
            return True

        def do_GET(self) -> None:
            self._serve(body=True)

        def do_HEAD(self) -> None:
            self._serve(body=False)

        def _serve(self, *, body: bool) -> None:
            # UnicodeDecodeError: a target that is not UTF-8; OSError: a name the filesystem rejects (too long, a
            # loop) or a file that went away before the read. Each is a 404, never a 500 or a traceback.
            try:
                path = resolve(root, self.path)
                data = path.read_bytes() if path is not None else None
            except (UnicodeDecodeError, OSError):
                path, data = None, None
            if path is None or data is None:
                self._not_found(body)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", _TYPES.get(path.suffix.lower(), "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if body:
                self.wfile.write(data)

        def _not_found(self, body: bool) -> None:
            data = b"Not found\n"
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body:
                self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            """No access or error log: a request line carries the query string (A4)."""

    return Handler


class _QuietServer(ThreadingHTTPServer):
    """A ``ThreadingHTTPServer`` whose request errors are dropped: the default ``handle_error`` prints a traceback
    to stderr, and the server prints nothing (module docstring)."""

    def handle_error(self, request: object, client_address: object) -> None:
        """Nothing printed; the connection is closed by ``socketserver`` as usual."""


class StartPageServer:
    """The loopback static server of the start pages (module docstring)."""

    def __init__(self, static_dir: str | os.PathLike[str] | None = None) -> None:
        self._root = Path(os.path.realpath(static_dir if static_dir is not None else STATIC_DIR))
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def origin(self) -> str:
        if self._server is None:
            raise RuntimeError("the start-page server is not running; call start() first")
        return f"http://{_HOST}:{self._server.server_address[1]}"

    def start(self) -> None:
        """Bind ``127.0.0.1:0`` and serve on a daemon thread; returns once listening."""
        if self._server is not None:
            raise RuntimeError("the start-page server is already running")
        server = _QuietServer((_HOST, 0), _handler(self._root))
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="astra-start-pages",
            daemon=True,
        )
        thread.start()
        self._server, self._thread = server, thread

    def stop(self) -> None:
        """Stop serving, close the socket and join the thread; a server never started is a no-op."""
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(_STOP_TIMEOUT_S)
