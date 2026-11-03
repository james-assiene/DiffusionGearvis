"""T-0111g group A, criteria A4-A6: the loopback start-page server and its pages, without a browser
(docs/tasks/T-0111g.md; the browser half is ``tests/integration/test_computer_live_browser_e2e.py``).

**The API these rows pin** (``local_astra/pages/`` does not exist at BASE; the names are inferred from A4, the
task fixes behaviour only):

==================================================  ==================================================================
``pages.server.StartPageServer(static_dir=None)``   ``static_dir``: the tree to serve (``None``: the package's own
                                                    ``local_astra/pages/static/``); constructing binds nothing
``srv.start() -> None``                             binds ``127.0.0.1`` on an ephemeral port and serves on a daemon
                                                    thread (stdlib ``ThreadingHTTPServer``); returns once listening
``srv.origin -> str``                               ``"http://127.0.0.1:<port>"``, exactly (no trailing slash)
``srv.stop() -> None``                              shuts down, closes the socket, joins the thread
==================================================  ==================================================================

**Serving rules (A4)** as these rows read them: GET and HEAD of regular files under ``static_dir`` only; ``/`` is
``index.html`` (it is the start page, A5); a path that leaves the tree (``..`` raw, percent-encoded or double-encoded,
an absolute filesystem path, a symlink pointing out) is 404 and never shows the outside file; a directory without
``index.html`` is not listed (403 or 404, and its file names are not in the body); every other method is 405. The
server writes no file and logs no query string (stdout, stderr and ``logging`` are captured).

**The static scan (A5)**: every ``src``/``href``/``action``/``srcset``/``poster`` attribute, CSS ``url(...)`` and
``@import``, and JS ``fetch(``/``new WebSocket(``/``new EventSource(``/``sendBeacon(``/``import(`` target must be a
string literal that is relative or same-origin (no scheme, not ``//``-protocol-relative, no backslash start). The
whole text must name no ``http(s)://`` or ``ws(s)://`` URL other than the two XML namespace identifiers
``http://www.w3.org/2000/svg`` and ``http://www.w3.org/1999/xlink`` (never fetched), and contain no ``@font-face`` (no
web font) and no ``<base`` element. A JS ``fetch(variable)`` cannot be checked by a static scan, so it is refused:
the criterion asks the scan to check every target. ``data:`` URLs count as not relative. The scan reads text, so
prose can trip it (a comment saying ``fetch(`` or naming a URL): the failure line names the file and the match.

Every row was ``pending(task="T-0111g")`` before group A's implementation; at BASE each failed with a named
``AssertionError`` (the package was absent). The group A review rows (Rulings, "on the group A reviews of c4618da")
are B-F2 (a ~5000-character path, and an unreadable file: 404 and nothing printed) and B-F3 (a symlinked
``index.html`` out of the tree: 404, kills X5).
"""

from __future__ import annotations

import http.client
import importlib
import logging
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_computer_imports import imported_names

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "local_astra" / "pages"
STATIC = PAGES / "static"
PAGE_FILES = ("index.html", "search.html", "demo/shop.html", "demo/contact.html", "demo/deals.html")
SECRET = "SECRET-T0111G-OUTSIDE"
QUERY_SECRET = "Q7Z9"


def server_mod() -> Any:
    try:
        return importlib.import_module("local_astra.pages.server")
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name == "local_astra.pages" or exc.name.startswith("local_astra.pages.")
        ):
            raise AssertionError(f"T-0111g A4: {exc.name} does not exist yet") from exc
        raise


def new_server(static_dir: Path | None = None) -> Any:
    cls = getattr(server_mod(), "StartPageServer", None)
    assert isinstance(cls, type), "T-0111g A4: pages.server.StartPageServer does not exist"
    return cls(static_dir=static_dir) if static_dir is not None else cls()


def port_of(origin: str) -> int:
    m = re.fullmatch(r"http://127\.0\.0\.1:(\d+)", origin)
    assert m, f"the origin is not exactly http://127.0.0.1:<port>: {origin!r}"
    port = int(m.group(1))
    assert 0 < port < 65536, port
    return port


def request(port: int, method: str, target: str) -> tuple[int, bytes, dict[str, str]]:
    """One raw HTTP/1.1 request; ``http.client`` sends ``target`` byte for byte (no normalisation)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest(method, target, skip_host=False, skip_accept_encoding=True)
        conn.endheaders()
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, body, {k.lower(): v for k, v in resp.getheaders()}
    finally:
        conn.close()


def listeners(port: int) -> list[tuple[str, str]]:
    """``(table, local address hex)`` of every LISTEN socket on ``port`` in ``/proc/net/tcp`` and ``tcp6``."""
    out = []
    for table in ("tcp", "tcp6"):
        path = Path("/proc/net") / table
        if not path.exists():
            continue
        for line in path.read_text().splitlines()[1:]:
            parts = line.split()
            addr, hexport = parts[1].rsplit(":", 1)
            if int(hexport, 16) == port and parts[3] == "0A":
                out.append((table, addr))
    return out


# --------------------------------------------------------------------------------------------- A4


def test_binds_loopback_ephemeral() -> None:
    """A4: two servers bind two different ephemeral ports on 127.0.0.1 only (every LISTEN socket on the port is
    ``0100007F`` in ``/proc/net/tcp``, none in ``tcp6``); the origin names the port; ``/`` and each page answer 200
    as HTML, and ``/`` is ``index.html``."""
    a, b = new_server(), new_server()
    a.start()
    try:
        b.start()
        try:
            pa, pb = port_of(a.origin), port_of(b.origin)
            assert pa != pb, "two servers share a port: it is not ephemeral"
            for p in (pa, pb):
                assert listeners(p) == [("tcp", "0100007F")], (p, listeners(p))
            for target in ["/", *(f"/{f}" for f in PAGE_FILES)]:
                status, body, headers = request(pa, "GET", target)
                assert status == 200, (target, status)
                assert headers.get("content-type", "").startswith("text/html"), (target, headers)
                assert b"<" in body, target
            status, home, _ = request(pa, "GET", "/")
            assert home == request(pa, "GET", "/index.html")[1], (
                "/ is not the start page index.html"
            )
        finally:
            b.stop()
    finally:
        a.stop()


def _tree(tmp_path: Path) -> Path:
    static = tmp_path / "static"
    (static / "sub").mkdir(parents=True)
    (static / "nolist").mkdir()
    (static / "index.html").write_text("<p>INDEX-OK</p>")
    (static / "sub" / "page.html").write_text("<p>SUB-OK</p>")
    (static / "nolist" / "unlisted-name.html").write_text("<p>hidden</p>")
    (tmp_path / "secret.txt").write_text(SECRET)
    (static / "link.txt").symlink_to(tmp_path / "secret.txt")
    (static / "linkdir").symlink_to(tmp_path)
    return static


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
    }


def test_traversal_and_methods_refused(tmp_path: Path) -> None:
    """A4 adversarial: every way out of ``static/`` is 404 and never shows the outside file -- raw ``..``,
    ``%2e%2e`` in either case, an encoded slash, double encoding (``%252e%252e``), backslashes, an absolute path,
    a symlink to a file outside and through a symlinked directory. A directory is not listed. GET and HEAD of a
    file inside work; every other method is 405."""
    static = _tree(tmp_path)
    srv = new_server(static)
    srv.start()
    try:
        port = port_of(srv.origin)
        assert request(port, "GET", "/index.html")[:2] == (200, b"<p>INDEX-OK</p>")
        assert request(port, "GET", "/sub/page.html")[:2] == (200, b"<p>SUB-OK</p>")
        status, body, _headers = request(port, "HEAD", "/index.html")
        assert status == 200 and body == b"", (status, body)
        out = [
            "/../secret.txt",
            "/sub/../../secret.txt",
            "/%2e%2e/secret.txt",
            "/%2E%2E/secret.txt",
            "/sub/%2e%2e/%2e%2e/secret.txt",
            "/..%2fsecret.txt",
            "/%2e%2e%2fsecret.txt",
            "/sub%2f..%2f..%2fsecret.txt",
            "/%252e%252e/secret.txt",
            "/%252e%252e%252fsecret.txt",
            "/..\\secret.txt",
            "/%5c..%5csecret.txt",
            "/" + str(tmp_path / "secret.txt").lstrip("/"),
            "/%2f" + str(tmp_path / "secret.txt").lstrip("/").replace("/", "%2f"),
            "//" + str(tmp_path / "secret.txt").lstrip("/"),
            "/link.txt",
            "/linkdir/secret.txt",
        ]
        for target in out:
            status, body, _ = request(port, "GET", target)
            assert SECRET.encode() not in body, f"{target} served the file outside static/"
            assert status == 404, (target, status)
        status, body, _ = request(port, "GET", "/nolist/")
        assert status in (403, 404), ("a directory was listed", status)
        assert b"unlisted-name" not in body, "a directory listing names its files"
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "PROPFIND", "FOO"):
            status, _, _ = request(port, method, "/index.html")
            assert status == 405, (method, status)
    finally:
        srv.stop()
    assert (tmp_path / "secret.txt").read_text() == SECRET


def test_logs_no_query_and_writes_no_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A4: requests carrying ``?secret=Q7Z9`` (a 200, a 404, a 405 and a traversal) leave no trace of the query in
    stdout, stderr or any ``logging`` record, and the server writes no file: the served tree and the working
    directory are unchanged."""
    static = _tree(tmp_path)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    before_static, before_cwd = _snapshot(static), _snapshot(cwd)
    caplog.set_level(logging.DEBUG)
    srv = new_server(static)
    srv.start()
    try:
        port = port_of(srv.origin)
        q = f"?secret={QUERY_SECRET}&q=blue+mug"
        assert request(port, "GET", f"/index.html{q}")[0] == 200
        assert request(port, "GET", f"/missing.html{q}")[0] == 404
        assert request(port, "POST", f"/index.html{q}")[0] == 405
        assert request(port, "GET", f"/../secret.txt{q}")[0] == 404
    finally:
        srv.stop()
    captured = capfd.readouterr()
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for where, text in (("stdout", captured.out), ("stderr", captured.err), ("logging", logged)):
        assert QUERY_SECRET not in text, f"the query string reached {where}: {text[-500:]}"
    assert _snapshot(static) == before_static, "the server wrote into the served tree"
    assert _snapshot(cwd) == before_cwd, "the server wrote into the working directory"


def _status_or_none(port: int, target: str) -> tuple[int | None, bytes]:
    """``request`` that turns a connection the server dropped without answering (its handler raised) into
    ``None``, so a row fails on a named assertion instead of ``RemoteDisconnected``."""
    try:
        status, body, _ = request(port, "GET", target)
    except (http.client.RemoteDisconnected, ConnectionError, http.client.BadStatusLine):
        return None, b""
    return status, body


def test_long_path_is_404_and_quiet(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """Group A review ruling, row B-F2 (fix F2: an ``OSError`` from ``resolve`` answers 404, nothing printed). A path
    of about 5000 characters, as one segment (longer than NAME_MAX, 255) and as many short ones (longer than
    PATH_MAX, 4096), each answers 404 (not a dropped connection), and stdout and stderr stay empty: the stdlib
    server prints a handler's traceback to stderr, and that would be a path (and a query) in a log."""
    static = _tree(tmp_path)
    srv = new_server(static)
    srv.start()
    try:
        port = port_of(srv.origin)
        capfd.readouterr()
        for target in (
            "/" + "a" * 5000 + f"?secret={QUERY_SECRET}",
            "/" + "a/" * 2499 + "a.html",
            "/sub/" + "b" * 4995,
        ):
            assert 4990 <= len(target.split("?")[0]) <= 5010, len(target)
            status, body = _status_or_none(port, target)
            assert SECRET.encode() not in body
            assert status == 404, (
                f"a {len(target.split('?')[0])}-character path answered {status} (None: the handler raised and the "
                "connection was dropped)"
            )
        assert request(port, "GET", "/index.html")[0] == 200, "the server stopped serving"
    finally:
        srv.stop()
    captured = capfd.readouterr()
    assert captured.err == "", f"the server printed to stderr: {captured.err[-800:]}"
    assert captured.out == "", f"the server printed to stdout: {captured.out[-800:]}"


def test_unreadable_file_is_404_and_quiet(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Group A review ruling, fix F2's ``read_bytes`` half: a regular file under the tree that cannot be read
    (mode 000; ``PermissionError`` from ``read_bytes``) answers 404, and nothing is printed. Skipped as root, where
    mode 000 does not stop a read."""
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file; the premise cannot hold")
    static = _tree(tmp_path)
    locked = static / "locked.html"
    locked.write_text("<p>LOCKED</p>")
    locked.chmod(0)
    srv = new_server(static)
    srv.start()
    try:
        port = port_of(srv.origin)
        capfd.readouterr()
        status, body = _status_or_none(port, "/locked.html")
        assert b"LOCKED" not in body
        assert status == 404, f"an unreadable file answered {status} (None: the handler raised)"
        assert request(port, "GET", "/index.html")[0] == 200
    finally:
        srv.stop()
        locked.chmod(0o644)
    captured = capfd.readouterr()
    assert captured.err == "" and captured.out == "", (captured.out[-400:], captured.err[-800:])


def test_symlinked_index_out_of_tree_is_404(tmp_path: Path) -> None:
    """Group A review ruling, row B-F3 (kills mutant X5, the index-symlink guard removed). A directory under
    ``static/`` whose ``index.html`` is a symlink to a file outside the tree answers 404 for ``/dir/`` and ``/dir``,
    and never shows the outside file; the same directory's other file, and a directory whose ``index.html`` is a
    real file, still serve (so the 404 is the guard, not a directory rule)."""
    static = _tree(tmp_path)
    (static / "trap").mkdir()
    (static / "trap" / "index.html").symlink_to(tmp_path / "secret.txt")
    (static / "trap" / "ok.html").write_text("<p>TRAP-OK</p>")
    (static / "good").mkdir()
    (static / "good" / "index.html").write_text("<p>GOOD-INDEX</p>")
    srv = new_server(static)
    srv.start()
    try:
        port = port_of(srv.origin)
        assert request(port, "GET", "/trap/ok.html")[:2] == (200, b"<p>TRAP-OK</p>")
        assert request(port, "GET", "/good/")[:2] == (200, b"<p>GOOD-INDEX</p>"), (
            "the premise failed: a directory's real index.html is not served"
        )
        for target in ("/trap/", "/trap", "/trap/index.html"):
            status, body, _ = request(port, "GET", target)
            assert SECRET.encode() not in body, f"{target} served the file outside static/"
            assert status == 404, (target, status)
    finally:
        srv.stop()


def test_stop_leaves_nothing_bound() -> None:
    """A4: after ``stop()`` (bounded at 5 s) nothing listens on the port, a connection is refused, and the serving
    thread is gone (the live thread set is the one before ``start()``)."""
    threads_before = set(threading.enumerate())
    srv = new_server()
    srv.start()
    port = port_of(srv.origin)
    assert request(port, "GET", "/")[0] == 200
    assert listeners(port), "the premise failed: nothing listened"
    t0 = time.monotonic()
    srv.stop()
    assert time.monotonic() - t0 < 5.0, "stop() took longer than 5 s"
    assert listeners(port) == [], listeners(port)
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    deadline = time.monotonic() + 2.0
    while set(threading.enumerate()) - threads_before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert set(threading.enumerate()) - threads_before == set(), (
        f"threads outlived stop(): {set(threading.enumerate()) - threads_before}"
    )


# --------------------------------------------------------------------------------------------- A5 (static scan)

_ATTR = re.compile(
    r"""\b(src|href|action|formaction|poster|srcset|data|xlink:href)\s*=\s*(["'])(.*?)\2""",
    re.I | re.S,
)
_UNQUOTED_ATTR = re.compile(
    r"""\b(src|href|action|formaction|poster|srcset)\s*=\s*([^\s"'>]+)""", re.I
)
_CSS_URL = re.compile(r"""url\(\s*(["']?)(.*?)\1\s*\)""", re.I | re.S)
_CSS_IMPORT = re.compile(r"""@import\s+(["'])(.*?)\1""", re.I)
_JS_CALL = re.compile(
    r"""\b(fetch|WebSocket|EventSource|sendBeacon|import|Worker|SharedWorker)\s*\(\s*(.)"""
)
_JS_LITERAL = re.compile(
    r"""\b(fetch|WebSocket|EventSource|sendBeacon|import|Worker|SharedWorker)\s*\(\s*(["'`])(.*?)\2""",
    re.S,
)
_ABSOLUTE = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9+.\-]*:|//|\\\\|/\\)")
_URL_IN_TEXT = re.compile(r"\b(?:https?|wss?)://[^\s\"'<>)]*", re.I)
NAMESPACES = frozenset({"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"})


def scan(name: str, text: str) -> list[str]:
    """Every A5 violation in one file's text, as ``"<name>: <what>"``."""
    bad: list[str] = []

    def check(kind: str, target: str) -> None:
        for part in (
            [p.strip().split(" ")[0] for p in target.split(",")] if kind == "srcset" else [target]
        ):
            part = part.strip().lstrip("`")
            if kind.endswith("(") and part.startswith("${"):
                bad.append(
                    f"{name}: {kind} target starts computed, the scan cannot check it: {part!r}"
                )
            elif _ABSOLUTE.match(part):
                bad.append(f"{name}: {kind} target is not relative: {part!r}")

    for m in _ATTR.finditer(text):
        check(m.group(1).lower(), m.group(3))
    for m in _UNQUOTED_ATTR.finditer(text):
        check(m.group(1).lower(), m.group(2))
    for m in _CSS_URL.finditer(text):
        check("url(", m.group(2))
    for m in _CSS_IMPORT.finditer(text):
        check("@import", m.group(2))
    literal_at = {m.start() for m in _JS_LITERAL.finditer(text)}
    for m in _JS_CALL.finditer(text):
        if m.group(1) == "import" and text[max(0, m.start() - 1)] == "@":
            continue
        if m.start() not in literal_at:
            bad.append(
                f"{name}: {m.group(1)}( target is not a string literal, so the scan cannot check it"
            )
    for m in _JS_LITERAL.finditer(text):
        check(m.group(1) + "(", m.group(3))
    for m in _URL_IN_TEXT.finditer(text):
        if m.group(0).rstrip("/") not in NAMESPACES:
            bad.append(f"{name}: names an external URL {m.group(0)!r}")
    if re.search(r"@font-face", text, re.I):
        bad.append(f"{name}: @font-face (a web font)")
    if re.search(r"<base\b", text, re.I):
        bad.append(f"{name}: a <base> element")
    return bad


SCANNER_SELF_CHECK = {
    "img": ('<img src="https://cdn.example/x.png">', 1),
    "protocol-relative": ('<script src="//cdn.example/x.js"></script>', 1),
    "unquoted": ("<a href=http://example.com/>x</a>", 2),
    "css-url": ("body { background: url('https://example.com/a.png') }", 2),
    "css-url-bare": ("body { background: url(//example.com/a.png) }", 1),
    "font": ("@font-face { font-family: x; src: url(x.woff2) }", 1),
    "import": ('@import "https://fonts.example/css";', 2),
    "fetch": ('fetch("https://api.example/v1")', 2),
    "fetch-computed": ("fetch(base + '/x')", 1),
    "ws": ("new WebSocket('ws://127.0.0.1:8000/x')", 2),
    "beacon": ("navigator.sendBeacon(`http://127.0.0.1:8000`)", 2),
    "data": ('<img src="data:image/png;base64,AAAA">', 1),
    "base": ('<base href="/">', 1),
    "srcset": ('<img srcset="a.png 1x, https://cdn.example/b.png 2x">', 2),
    "meta-refresh": ('<meta http-equiv="refresh" content="0;url=https://example.com/">', 1),
}
SCANNER_CLEAN = (
    '<link rel="stylesheet" href="../style.css"><script src="app.js"></script>'
    '<a href="/demo/shop.html?x=1#cart">Shop</a><a href="search.html">s</a><form action="/search.html">'
    "<style>.m{background:url(img/mug.png)}</style><script>fetch('catalogue.json');"
    "document.createElementNS('http://www.w3.org/2000/svg', 'svg');</script>"
)


def _static_files() -> list[Path]:
    assert STATIC.is_dir(), "T-0111g A5: local_astra/pages/static/ does not exist yet"
    return sorted(p for p in STATIC.rglob("*") if p.is_file())


def test_no_external_references() -> None:
    """A5 (static): the scanner first proves it sees each kind of violation and passes a clean page, then every
    file under ``static/`` (HTML, CSS, JS, SVG, JSON) is scanned and the five pages exist. Each page also names
    ``__state`` (the runtime shape is the browser row's)."""
    for kind, (text, n) in SCANNER_SELF_CHECK.items():
        found = scan(kind, text)
        assert len(found) >= n and found, (kind, found)
    assert scan("clean", SCANNER_CLEAN) == []

    files = _static_files()
    rel = {p.relative_to(STATIC).as_posix() for p in files}
    assert set(PAGE_FILES) <= rel, f"missing pages: {sorted(set(PAGE_FILES) - rel)}"
    bad: list[str] = []
    for p in files:
        if p.suffix.lower() in (".html", ".htm", ".css", ".js", ".mjs", ".svg", ".json"):
            bad += scan(p.relative_to(STATIC).as_posix(), p.read_text(encoding="utf-8"))
        elif p.suffix.lower() in (".woff", ".woff2", ".ttf", ".otf", ".eot"):
            bad.append(f"{p.relative_to(STATIC)}: a font file")
    assert bad == [], "\n".join(bad)
    for page in PAGE_FILES:
        text = (STATIC / page).read_text(encoding="utf-8")
        scripts = [
            (STATIC / page).parent / s
            for s in re.findall(r"""<script[^>]*\bsrc\s*=\s*["']([^"']+)["']""", text, re.I)
            if not s.startswith("/")
        ] + [
            STATIC / s.lstrip("/")
            for s in re.findall(r"""<script[^>]*\bsrc\s*=\s*["'](/[^"']+)["']""", text, re.I)
        ]
        source = text + "".join(s.read_text(encoding="utf-8") for s in scripts if s.is_file())
        assert "__state" in source, f"{page} does not expose window.__state"


# --------------------------------------------------------------------------------------------- A6


def test_pages_import_nothing_from_computer_or_conductor() -> None:
    """A6: ``local_astra/pages/**`` imports nothing from ``local_astra.computer`` or ``local_astra.conductor`` (the
    AST scan of ``test_computer_imports.py``, relative imports resolved). The scan is not vacuous: ``server.py``
    imports something (the stdlib server)."""
    assert PAGES.is_dir(), "T-0111g A6: local_astra/pages/ does not exist yet"
    files = sorted(PAGES.rglob("*.py"))
    assert {"__init__.py", "server.py"} <= {p.name for p in files}, [p.name for p in files]
    seen: dict[str, list[str]] = {}
    bad = []
    for p in files:
        names = [n for _, n in imported_names(p, REPO)]
        seen[p.name] = names
        for lineno, n in imported_names(p, REPO):
            if any(
                n == m or n.startswith(m + ".")
                for m in ("local_astra.computer", "local_astra.conductor")
            ):
                bad.append(f"{p.relative_to(REPO)}:{lineno}: import {n}")
    assert any(n.startswith("http.server") for n in seen["server.py"]), (
        "the scan saw no stdlib server import in server.py",
        seen["server.py"],
    )
    assert bad == [], bad
