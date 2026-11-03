"""Serving the page: the URL a human can paste, and the hole a static server must not be.

Criteria 16 and 18. Everything binds ``127.0.0.1`` and the traversal probes are written as raw
HTTP requests, because ``http.client`` normalises ``..`` away before the server ever sees it —
a traversal test written through a client library proves nothing about the server.

Regime: no GPU, no network off the box, no measurement.
"""

from __future__ import annotations

import ast
import inspect
import socket
import socketserver
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from local_astra import cli
from local_astra.config import load
from tests.unit import _box_rig as rig

SENTINEL = "SECRET-OUTSIDE-THE-HUD-DIRECTORY"

#: T-0110c: the shutdown poll the rows of this module run at. The stdlib's 0.5 s made each
#: ``close()`` wait up to half a second for the serving thread (13 rows x 0.5 s of the module's
#: 6.6 s). Production keeps the stdlib default; only tests pass this.
FAST_POLL_S = 0.01

#: ``socketserver.BaseServer.serve_forever``'s documented default (Python docs, socketserver):
#: the production value T-0110c criterion 2 says must not move.
STDLIB_POLL_S = 0.5


def hud_static():
    return rig.module("hud_static")


def serve(root: Path, port: int = 0):
    return rig.attr(hud_static(), "HudStaticServer")(root, port, poll_interval=FAST_POLL_S)


def raw_request(port: int, request_line: str, body: str = "") -> str:
    """Send a request byte for byte — no client-side normalisation between us and the server."""
    payload = f"{request_line}\r\nHost: 127.0.0.1\r\nConnection: close\r\n"
    if body:
        payload += f"Content-Length: {len(body)}\r\n"
    payload += "\r\n" + body
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(payload.encode())
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks).decode(errors="replace")


def status_of(response: str) -> int:
    return int(response.split()[1])


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "hud"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html><title>astra</title>")
    (root / "styles.css").write_text("body{}")
    (tmp_path / "outside.txt").write_text(SENTINEL)
    return tmp_path


def test_the_printed_url_carries_the_ws_query_the_page_needs():
    """Criterion 16: ``hud_port`` (1420) is finally read, and the URL is complete."""
    services = load(None).profiles["lean"].services
    urls = rig.attr(hud_static(), "page_urls")(services)
    assert len(urls) >= 2, f"criterion 16: the plain URL and the ?source=canvas variant; got {urls}"

    parsed = [urlsplit(url) for url in urls]
    for url, parts in zip(urls, parsed, strict=True):
        assert parts.scheme == "http" and parts.hostname == "127.0.0.1", f"loopback only: {url}"
        assert parts.port == services.hud_port == 1420, (
            f"criterion 16: the page is served on config.Services.hud_port; {url} is not"
        )
        query = parse_qs(parts.query)
        assert query.get("ws") == [f"ws://127.0.0.1:{services.conductor_port}/v1/realtime"], (
            f"criterion 16: the URL includes the ws query string the page needs; {url} has "
            f"{parts.query!r}"
        )
    sources = [parse_qs(parts.query).get("source", [None])[0] for parts in parsed]
    assert "canvas" in sources, (
        f"criterion 16: …plus the ?source=canvas variant for headless runs; got {urls}"
    )
    assert None in sources, f"criterion 16: …and the plain one for a real share; got {urls}"


def test_the_hud_verb_prints_that_url(capsys, monkeypatch):
    """Criterion 16: one command gives the human a page URL — the whole point of group C."""
    services = load(None).profiles["lean"].services
    expected = rig.attr(hud_static(), "page_urls")(services)
    started = {}

    class Recorder:
        def __init__(self, root, port):
            started["root"], started["port"] = Path(root), int(port)
            self.port = int(port) or 1420
            self.address = ("127.0.0.1", self.port)

        def serve_forever(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(hud_static(), "HudStaticServer", Recorder)
    code = cli.main(["hud"])
    printed = capsys.readouterr()
    report = printed.out + printed.err
    assert code == 0, f"`astra hud` exits 0 when it served; it exited {code}. {report}"
    for url in expected:
        assert url in report, f"criterion 16: it prints {url}; it printed\n{report}"
    assert started["port"] == services.hud_port
    assert started["root"].resolve() == (rig.REPO / "hud").resolve(), (
        f"criterion 16: it serves hud/; it served {started['root']}"
    )


def test_the_server_binds_loopback_and_serves_the_directory(tree):
    server = serve(tree / "hud")
    try:
        assert server.address[0] == "127.0.0.1", (
            "criterion 18: a request from a non-loopback peer is impossible by binding; "
            f"getsockname() says {server.address}"
        )
        assert server.address[0] != "0.0.0.0"
        response = raw_request(server.port, "GET /index.html HTTP/1.1")
        assert status_of(response) == 200 and "astra" in response
    finally:
        server.close()


@pytest.mark.parametrize(
    "target",
    [
        "/../outside.txt",
        "/../../outside.txt",
        "/%2e%2e/outside.txt",
        "/..%2foutside.txt",
        "/%2e%2e%2f%2e%2e%2fetc/passwd",
        "/../local_astra/config.py",
        "//../outside.txt",
    ],
)
def test_path_traversal_is_a_404_that_reads_nothing(tree, target):
    """Criterion 18, adversarial: a naive ``os.path.join(root, path)`` serves every one of these."""
    server = serve(tree / "hud")
    try:
        response = raw_request(server.port, f"GET {target} HTTP/1.1")
        assert SENTINEL not in response, (
            f"criterion 18: {target} read a file outside hud/. The response was:\n{response[:400]}"
        )
        assert "LOOPBACK_PREFIXES" not in response, f"{target} served local_astra/config.py"
        assert status_of(response) == 404, (
            f"criterion 18: {target} must be a 404; the server answered "
            f"{response.splitlines()[0]!r}"
        )
    finally:
        server.close()


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_write_methods_are_refused(tree, method):
    server = serve(tree / "hud")
    try:
        response = raw_request(server.port, f"{method} /index.html HTTP/1.1", body="x=1")
        assert status_of(response) >= 400, (
            f"criterion 18: {method} is refused; the server answered {response.splitlines()[0]!r}"
        )
        assert not (tree / "hud" / "index.html").read_text().startswith("x=1")
    finally:
        server.close()


def test_a_taken_port_is_a_failure_and_never_a_move_to_another_one(tree, capsys, tmp_path):
    """Criterion 18: a page on a port nobody chose is a page talking to nothing (T-0028 c.19)."""
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]

        with pytest.raises(OSError):
            serve(tree / "hud", port)

        config = tmp_path / "astra.toml"
        config.write_text(
            "profile = 'lean'\n\n[profiles.lean]\nname = 'lean'\n\n"
            f"[profiles.lean.services]\nhud_port = {port}\n"
        )
        code = cli.main(["--config", str(config), "hud"])
        printed = capsys.readouterr()
        report = printed.out + printed.err
        assert code != 0, f"criterion 18: it exits non-zero; it exited {code}. {report}"
        assert str(port) in report, f"criterion 18: …naming the port; it said\n{report}"
        other_ports = {
            urlsplit(token).port for token in report.split() if token.startswith("http://127.0.0.1")
        } - {port, None}
        assert not other_ports, (
            f"criterion 18: it never falls back to another port; it advertised {other_ports}"
        )


def test_the_real_hud_directory_is_what_gets_served():
    hud_dir = Path(rig.attr(hud_static(), "HUD_DIR"))
    assert hud_dir.resolve() == (rig.REPO / "hud").resolve()
    assert (hud_dir / "index.html").exists(), "criterion 16: there is a page to serve"


def test_session_can_serve_the_page_and_prints_the_same_url(capsys):
    """Criterion 17: ``astra session --serve-hud`` is the one command, and it says the same URL.

    ``--dry-run`` is as far as a unit test can take this: the rest of criterion 17 (the thread
    beside the session, Ctrl-C stopping both, no ``turns.jsonl`` line at shutdown) needs a live
    session and is covered by T-0028 criterion 16 plus the box run.
    """
    import ast

    flags = {
        arg.value
        for node in ast.walk(ast.parse((rig.REPO / "local_astra" / "cli.py").read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert "--serve-hud" in flags, "criterion 17: `astra session --serve-hud`"

    code = cli.main(["session", "--serve-hud", "--dry-run"])
    printed = capsys.readouterr()
    assert code == 0, f"--dry-run still constructs nothing and exits 0; it exited {code}"
    services = load(None).profiles["lean"].services
    urls = rig.attr(hud_static(), "page_urls")(services)
    assert urls[0] in printed.out, (
        f"criterion 17: the printed line is the same URL as `astra hud`'s; it printed\n{printed.out}"
    )


# --- T-0040: closing is interrupt-safe and idempotent -------------------------------------------
#
# Ctrl-C in a terminal reaches the whole foreground process group, so a session started through
# ``uv`` sees two SIGINTs; the second lands inside ``close()`` -> ``HTTPServer.shutdown()`` ->
# ``Event.wait()`` (live log logs/t0039-live-82d8462/run.log). What T-0040 criterion 2 asks of
# this module is that an interrupt there still ends the serving thread and unbinds the socket,
# and that closing twice is a no-op.


def is_unbound(port: int) -> bool:
    """True if ``127.0.0.1:port`` can be bound again: the server let go of its socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def threads_started_by(call):
    """Run ``call`` and return the threads it started (so the check names no private attribute)."""
    import threading

    before = set(threading.enumerate())
    result = call()
    return result, [t for t in threading.enumerate() if t not in before]


def test_closing_twice_is_a_no_op(tree):
    """Criterion 2, second half: the second ``close()`` neither raises nor blocks nor rebinds.

    Not pending: a double close must keep working across T-0040's change, and the shutdown path
    the CLI takes calls ``close()`` from a ``finally`` that a retry may reach twice.
    """
    server, started = threads_started_by(lambda: serve(tree / "hud"))
    port = server.port
    assert started, "the server serves on a thread of its own"
    server.close()
    server.close()  # the one under test: a no-op, not an exception and not a second shutdown
    assert is_unbound(port), f"criterion 18/2: {port} is still bound after close()"
    assert not [t for t in started if t.is_alive()], (
        f"criterion 2: no serving thread survives close(); alive: {[t.name for t in started]}"
    )


def test_an_interrupt_inside_shutdown_still_ends_the_thread_and_unbinds(tree, monkeypatch):
    """Criterion 2: a ``KeyboardInterrupt`` raised inside ``close()`` leaves nothing behind.

    The interrupt is injected where the live one landed — the blocking ``shutdown()`` wait — and
    only on the first call, so ``close()`` is free to finish the job it was interrupted in. The
    assertions are about what is left on the box, not about whether the interrupt propagates: an
    implementation that absorbs it and one that re-raises it after tearing down both satisfy the
    criterion, an implementation that leaves a listener behind does not.
    """
    import contextlib
    import socketserver

    real_shutdown = socketserver.BaseServer.shutdown
    interrupted: list[int] = []

    def shutdown_interrupted_once(self):  # type: ignore[no-untyped-def]
        if not interrupted:
            interrupted.append(1)
            raise KeyboardInterrupt
        return real_shutdown(self)

    server, started = threads_started_by(lambda: serve(tree / "hud"))
    port = server.port
    monkeypatch.setattr(socketserver.BaseServer, "shutdown", shutdown_interrupted_once)
    try:
        with contextlib.suppress(KeyboardInterrupt):
            server.close()
        # The state is read *here*, while the test's own cleanup has not run: a check written
        # after the cleanup below passes on today's code and proves nothing.
        left_bound = not is_unbound(port)
        alive = [t.name for t in started if t.is_alive()]
    finally:
        monkeypatch.undo()
        with contextlib.suppress(Exception):
            server.close()  # cleanup only: no row after this one may inherit a listener
    assert interrupted, "the interrupt must have been raised inside the shutdown wait"
    assert not left_bound, (
        f"criterion 2: an interrupt inside close() must not leave the socket bound; {port} is "
        f"still taken"
    )
    assert not alive, f"criterion 2: …nor the serve_forever thread running; alive: {alive}"


# --- T-0110c: the shutdown poll is injectable; production keeps the stdlib's 0.5 s -------------
#
# ``close()`` -> ``shutdown()`` waits for ``serve_forever`` to notice the request, which it does
# once per ``poll_interval``. The seam is ``HudStaticServer(root, port=0, *, poll_interval=0.5)``
# (tests/unit/_box_rig.py's docstring); ``astra session`` and ``astra hud`` pass nothing.


def build(root: Path, port: int = 0, **kwargs):
    """Construct the server with keywords, turning "no such keyword" into the pending outcome.

    ``tests/conftest.py`` tolerates only ``AssertionError``/``NotImplementedError`` from a pending
    row; before T-0110c the ``poll_interval`` keyword is a ``TypeError``, which is the documented
    "not implemented yet" here and is reported as such. Any other ``TypeError`` propagates.
    """
    server_cls = rig.attr(hud_static(), "HudStaticServer")
    try:
        return server_cls(root, port, **kwargs)
    except TypeError as exc:
        if "poll_interval" not in str(exc):
            raise
        raise AssertionError(
            f"T-0110c criterion 1: HudStaticServer(root, port=0, *, poll_interval=0.5) is the "
            f"seam; constructing with {kwargs} said: {exc}"
        ) from None


@pytest.fixture
def serve_forever_calls(monkeypatch):
    """Record every call of ``hud_static._Server.serve_forever``, bound to the stdlib signature.

    Each entry is the ``inspect.BoundArguments`` of the call against
    ``socketserver.BaseServer.serve_forever`` (so "passed 0.5", "passed nothing" and "passed it
    positionally" all read the same), or the ``TypeError`` the binding raised. The recorder then
    really serves — at ``FAST_POLL_S``, so this fixture costs 10 ms rather than 500 ms — and so
    blocks until ``shutdown()``: a recorder that returned at once, or raised, would leave
    ``close()``'s ``shutdown()`` waiting for ever. The value under test is the recorded one.
    """
    server_cls = rig.attr(hud_static(), "_Server")
    real = server_cls.serve_forever
    signature = inspect.signature(socketserver.BaseServer.serve_forever)
    calls: list[object] = []
    entered = threading.Event()

    def recorder(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            calls.append(signature.bind(self, *args, **kwargs))
        except TypeError as exc:
            calls.append(exc)
        finally:
            entered.set()
        return real(self, poll_interval=FAST_POLL_S)

    monkeypatch.setattr(server_cls, "serve_forever", recorder)
    return calls, entered


def recorded_poll_interval(calls, entered) -> float:
    assert entered.wait(5), "the serving thread never entered _Server.serve_forever"
    assert len(calls) == 1, f"serve_forever is entered exactly once per server; calls: {calls}"
    (bound,) = calls
    assert isinstance(bound, inspect.BoundArguments), (
        f"serve_forever was called with arguments its stdlib signature refuses: {bound}"
    )
    bound.apply_defaults()
    return bound.arguments["poll_interval"]


def test_the_default_poll_interval_is_the_stdlib_half_second(tree, serve_forever_calls):
    """Criterion 2 (adversarial, negative): built the way the CLI builds it, the poll is 0.5 s.

    Not pending: it holds before T-0110c and must still hold after. The mutation it exists for
    is a default lowered to the test value (production would then wake 50 times a second).
    """
    calls, entered = serve_forever_calls
    server = rig.attr(hud_static(), "HudStaticServer")(tree / "hud", 0)
    try:
        poll = recorded_poll_interval(calls, entered)
    finally:
        server.close()
    assert poll == STDLIB_POLL_S, (
        f"criterion 2: a server built without poll_interval must serve at the stdlib's "
        f"{STDLIB_POLL_S} s; serve_forever got poll_interval={poll!r}"
    )


@pytest.mark.parametrize("value", [FAST_POLL_S, 0.25])
def test_the_poll_interval_reaches_serve_forever(tree, serve_forever_calls, value):
    """Criterion 1: the value given is the value ``serve_forever`` polls at (two values, so a
    hard-coded test constant cannot pass both)."""
    calls, entered = serve_forever_calls
    server = build(tree / "hud", poll_interval=value)
    try:
        poll = recorded_poll_interval(calls, entered)
    finally:
        server.close()
    assert poll == value, (
        f"criterion 1: HudStaticServer(..., poll_interval={value}) must reach serve_forever as "
        f"its poll_interval; it got {poll!r}"
    )


def refuses_connections(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return False
    except ConnectionRefusedError:
        return True


def test_twenty_closes_are_fast(tree):
    """Criterion 5: 20 construct-then-close cycles at ``poll_interval=0.01`` take < 1.0 s.

    At the stdlib's 0.5 s the same loop takes up to 10 s, so the bar separates "forwarded" from
    "accepted and dropped" by an order of magnitude. Only construct + close is timed; the
    refused-connection probe after each close (the server really let go) is not.
    """
    cycles, budget_s = 20, 1.0
    spent = 0.0
    for cycle in range(cycles):
        started = time.perf_counter()
        server = build(tree / "hud", poll_interval=FAST_POLL_S)
        try:
            port = server.port
        finally:
            server.close()
        spent += time.perf_counter() - started
        assert refuses_connections(port), (
            f"criterion 5: after close() number {cycle + 1}, 127.0.0.1:{port} still accepts"
        )
    assert spent < budget_s, (
        f"criterion 5: {cycles} construct+close cycles at poll_interval={FAST_POLL_S} took "
        f"{spent:.3f} s; the bar is < {budget_s} s"
    )


def test_the_poll_interval_is_keyword_only(tree):
    """Criterion 1: ``HudStaticServer(root, port=0, *, poll_interval=0.5)``; a third positional
    argument is a ``TypeError``, so ``HudStaticServer(HUD_DIR, port, 0.01)`` can never mean a
    poll interval by accident."""
    server_cls = rig.attr(hud_static(), "HudStaticServer")
    params = inspect.signature(server_cls).parameters
    assert "poll_interval" in params, (
        f"criterion 1: HudStaticServer takes poll_interval; its signature is "
        f"{inspect.signature(server_cls)}"
    )
    assert params["poll_interval"].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"criterion 1: poll_interval is keyword-only; it is {params['poll_interval'].kind}"
    )
    assert params["poll_interval"].default == STDLIB_POLL_S, (
        f"criterion 1: the default is {STDLIB_POLL_S}; it is {params['poll_interval'].default!r}"
    )
    assert list(params)[:2] == ["root", "port"] and params["port"].default == 0, (
        f"criterion 1: root and port keep their places; {inspect.signature(server_cls)}"
    )
    built = None
    try:
        with pytest.raises(TypeError):
            built = server_cls(tree / "hud", 0, FAST_POLL_S)
    finally:
        if built is not None:
            built.close()


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")], ids=["0", "-1", "nan", "inf"])
def test_a_bad_poll_interval_is_refused_before_binding(tree, bad):
    """Criterion 4 (H-T110c-1: refuse, not clamp): ``ValueError`` at construction, port free.

    ``refused`` holds the traceback, and with it the half-built server: a socket bound before
    the check would still be open when the port is probed, so "validated after binding" is red
    here even though the exception type is right.
    """
    port = rig.free_port()
    with pytest.raises(ValueError) as refused:
        build(tree / "hud", port, poll_interval=bad)
    assert is_unbound(port), (
        f"criterion 4: poll_interval={bad!r} must be refused before the socket binds; "
        f"127.0.0.1:{port} is taken after the refusal ({refused.value!r})"
    )


def server_constructions(path: Path) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and "HudStaticServer" in (getattr(node.func, "id", None), getattr(node.func, "attr", None))
    ]


def test_the_cli_builds_the_server_with_the_shipped_default():
    """Criterion 3 (negative): one construction site in ``local_astra/``, the CLI's, with the two
    positional arguments ``hud_static.HUD_DIR, services.hud_port`` and no keyword.

    Not pending: true before T-0110c and must stay true. ``cli.py`` being byte-identical is the
    verifier's diff check, not pinned here (a content hash would decay with the next CLI task).
    """
    package = rig.REPO / "local_astra"
    sites = {
        path.relative_to(rig.REPO).as_posix(): calls
        for path in sorted(package.rglob("*.py"))
        if (calls := server_constructions(path))
    }
    assert list(sites) == ["local_astra/cli.py"], (
        f"criterion 3: only local_astra/cli.py constructs HudStaticServer; sites: "
        f"{ {name: [ast.unparse(c) for c in calls] for name, calls in sites.items()} }"
    )
    calls = sites["local_astra/cli.py"]
    assert len(calls) == 1, f"criterion 3: one site; {[ast.unparse(c) for c in calls]}"
    (call,) = calls
    assert not call.keywords, (
        f"criterion 3: the CLI passes no keyword (so poll_interval is the default); "
        f"{ast.unparse(call)}"
    )
    assert [ast.unparse(arg) for arg in call.args] == ["hud_static.HUD_DIR", "services.hud_port"], (
        f"criterion 3: exactly two positional arguments; {ast.unparse(call)}"
    )


# --- T-0165 C8: one keep-alive connection, no per-request stall --------------------------------
#
# docs/tasks/T-0165.md C8 and C9(c). The page fetches ~30 modules over keep-alive connections; a
# response written as two small segments (headers, then body) waits for the client's delayed ACK
# under Nagle, ~40 ms per request on Linux. These rows talk raw HTTP/1.1 over *one* socket and read
# each response exactly by its ``Content-Length`` (no client library, no ``Connection: close``).

#: C8: the request the page makes most, against the real ``hud/`` at HEAD.
KEEP_ALIVE_PATH = "/src/panels.js"
KEEP_ALIVE_REQUESTS = 30
#: Just under the Linux delayed-ACK floor (40 ms): a stalled response takes at least this long.
STALL_MS = 35.0
#: C8: "at most 2 of the 30" may cross it (scheduler noise on a loaded box, not the stall).
MAX_STALLED = 2


class _Reply:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body


def _read_reply(stream, *, head: bool = False) -> _Reply:
    """One HTTP/1.1 response off a keep-alive stream: status line, headers, then exactly
    ``Content-Length`` bytes (none for a HEAD, RFC 9110 9.3.2)."""
    status_line = stream.readline()
    assert status_line, "the server closed the keep-alive connection instead of answering"
    parts = status_line.decode("latin-1").split()
    assert parts and parts[0] == "HTTP/1.1", f"not an HTTP/1.1 status line: {status_line!r}"
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        assert line, "the connection ended inside the headers"
        if line in (b"\r\n", b"\n"):
            break
        name, _, value = line.decode("latin-1").partition(":")
        headers[name.strip().lower()] = value.strip()
    length = 0 if head else int(headers.get("content-length", "0"))
    body = stream.read(length) if length else b""
    assert len(body) == length, f"short body: {len(body)} of {length} bytes"
    return _Reply(int(parts[1]), headers, body)


def _request(path: str, method: str = "GET", body: bytes = b"") -> bytes:
    lines = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1"]
    if body:
        lines.append(f"Content-Length: {len(body)}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


def _real_hud_dir() -> Path:
    return Path(rig.attr(hud_static(), "HUD_DIR"))


def test_keep_alive_requests_do_not_stall():
    """C8: 30 sequential keep-alive GETs of ``/src/panels.js``; at most 2 take >= 35 ms.

    Regime (docs/tasks/T-0165.md): CPython 3.11, loopback, one keep-alive connection, the real
    ``hud/src/panels.js`` at HEAD, n=1 run, threshold <= 2 of 30 at >= 35 ms. The message carries
    ``os.getloadavg()``, the file size and every per-request time. Each body must be the file,
    byte for byte, with the status, ``Content-Type`` and ``Content-Length`` it always had.
    """
    import mimetypes
    import os

    hud_dir = _real_hud_dir()
    expected = (hud_dir / KEEP_ALIVE_PATH.lstrip("/")).read_bytes()
    expected_type = mimetypes.guess_type(KEEP_ALIVE_PATH)[0]
    server = serve(hud_dir)
    times_ms: list[float] = []
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
            stream = sock.makefile("rb")
            for _ in range(KEEP_ALIVE_REQUESTS):
                started = time.perf_counter()
                sock.sendall(_request(KEEP_ALIVE_PATH))
                reply = _read_reply(stream)
                times_ms.append((time.perf_counter() - started) * 1000)
                assert reply.status == 200, f"C8: status {reply.status}, not 200"
                assert reply.body == expected, "C8: the body is not the file, byte for byte"
                assert reply.headers.get("content-type") == expected_type, reply.headers
                assert reply.headers.get("content-length") == str(len(expected)), reply.headers
    finally:
        server.close()
    stalled = [round(t, 2) for t in times_ms if t >= STALL_MS]
    assert len(stalled) <= MAX_STALLED, (
        f"C8: {len(stalled)} of {KEEP_ALIVE_REQUESTS} keep-alive requests took >= {STALL_MS} ms "
        f"(at most {MAX_STALLED} may): the response waits for the client's delayed ACK. "
        f"loadavg={os.getloadavg()}, {KEEP_ALIVE_PATH} is {len(expected)} bytes, "
        f"times_ms={[round(t, 2) for t in times_ms]}"
    )


def test_keep_alive_stream_stays_in_sync_across_404_head_and_405():
    """C8/C9(c), adversarial: a single-write or buffering change is what would desync framing.

    One connection: GET 200 -> GET traversal 404 -> HEAD 200 -> POST-with-body 405 -> GET 200.
    Every response parses with the status the refusals have always had, and the last body is the
    file. **Not pending**: it holds before T-0165 and must keep holding after (a pending row that
    passes is a strict XPASS); it is the guard on the C8 change, not its acceptance test.
    """
    hud_dir = _real_hud_dir()
    expected = (hud_dir / KEEP_ALIVE_PATH.lstrip("/")).read_bytes()
    posted = b"x=" + b"a" * 300  # a body the server must consume, or the next request desyncs
    sequence = [
        ("GET", KEEP_ALIVE_PATH, b"", 200),
        ("GET", "/../../local_astra/config.py", b"", 404),
        ("HEAD", KEEP_ALIVE_PATH, b"", 200),
        ("POST", KEEP_ALIVE_PATH, posted, 405),
        ("GET", KEEP_ALIVE_PATH, b"", 200),
    ]
    server = serve(hud_dir)
    seen: list[tuple[str, int]] = []
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
            stream = sock.makefile("rb")
            replies = []
            for method, path, body, _status in sequence:
                sock.sendall(_request(path, method, body))
                reply = _read_reply(stream, head=method == "HEAD")
                replies.append(reply)
                seen.append((f"{method} {path}", reply.status))
    finally:
        server.close()
    assert [status for _, status in seen] == [status for *_, status in sequence], (
        f"C9(c): one keep-alive stream answers each request with its own status; got {seen}"
    )
    assert replies[0].body == expected and replies[-1].body == expected, (
        "C8: the GETs around the refusals return the file, byte for byte (a desynced stream "
        "reads a refusal's bytes or a stale body here)"
    )
    assert b"Services" not in replies[1].body, (
        "C9(c): the traversal 404 carries none of local_astra/config.py"
    )
