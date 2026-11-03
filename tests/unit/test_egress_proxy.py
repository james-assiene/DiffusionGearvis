"""T-0160 group A, criteria A-5 and A-6: ``local_astra/computer/egress_proxy.py``'s ``EgressProxy``, an HTTP CONNECT
proxy on its own thread and loop, bound to ``127.0.0.1:0``.

No real network anywhere. Every row runs inside ``loopback_only()`` from ``tests/unit/test_session_no_egress.py``
(connect/bind/listen/sendto/getaddrinfo to anything but loopback raises and is recorded), and the fixture asserts the
trap recorded nothing. The proxy gets an injected fake resolver and connector:

- ``resolve(host, port, ...)`` answers from a hand table in ``getaddrinfo`` shape (``(family, type, proto,
  canonname, sockaddr)``), the shape of the default (``getaddrinfo``). Its return value is both a list and an
  awaitable, so the rows do not dictate whether the implementation awaits it (``loop.getaddrinfo``) or calls it in
  an executor. An IP literal resolves to itself, as ``getaddrinfo`` does, and ``2130706433`` to ``127.0.0.1``.
- ``connect(ip, port)`` is a coroutine function, the shape of the default (``asyncio.open_connection``). It records
  ``(ip, port)`` and opens a stream to a loopback upstream that sends ``HELLO\\n`` and then echoes upper-cased, so
  "pipes both ways" is checked from both ends. It never touches the address it was given.

Expected values are by hand: the egress record of a refused ``host`` is ``{"host": host, "scheme": "https"}``
(``fence.egress_record("https://host/")``, criterion A-5 (b)).

The rows are active (activated for /implement T-0160). The start-origin rows follow O-A2 (2):
``EgressProxy(..., tunnel_start_origin=True)`` pipes exactly the start host:port to the start server.
"""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import pytest

from tests.unit.test_session_no_egress import loopback_only

IO_TIMEOUT_S = 5.0
UNUSED_START_PORT = 9  # the start origin's port in these rows: not 443, and never connected to


def egress_proxy() -> Any:
    try:
        return importlib.import_module("local_astra.computer.egress_proxy")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("local_astra.computer"):
            raise AssertionError(f"T-0160 A-5: {exc.name} does not exist yet") from exc
        raise


def rec(host: str) -> dict[str, str]:
    return {"host": host, "scheme": "https"}


# --- the loopback upstream the fake connector reaches ----------------------------------------------------------


class Upstream:
    """A threaded loopback TCP server: on accept it sends ``HELLO\\n``, then echoes every byte upper-cased."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(256)
        self.sock.settimeout(0.05)
        self.port: int = self.sock.getsockname()[1]
        self.accepted = 0
        self.conns: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stopping = False
        self._thread = threading.Thread(target=self._accept, name="upstream-accept", daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._stopping:
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue  # wake up to see ``_stopping``: a blocked accept() does not wake on close()
            except OSError:
                return
            with self._lock:
                self.accepted += 1
                self.conns.append(conn)
            threading.Thread(
                target=self._serve, args=(conn,), name="upstream-conn", daemon=True
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        with suppress(OSError):
            conn.sendall(b"HELLO\n")
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                conn.sendall(data.upper())
        with suppress(OSError):
            conn.close()

    def close(self) -> None:
        self._stopping = True
        with suppress(OSError):
            self.sock.close()
        with self._lock:
            for c in self.conns:
                with suppress(OSError):
                    c.shutdown(socket.SHUT_RDWR)
                with suppress(OSError):
                    c.close()
        self._thread.join(2)


# --- the fake resolver and connector --------------------------------------------------------------------------


class _Answer(list):  # type: ignore[type-arg]
    """A ``getaddrinfo`` result that is also awaitable (sync or ``loop.getaddrinfo``-style callers both work)."""

    def __await__(self):  # type: ignore[no-untyped-def]
        async def value() -> list[Any]:
            return list(self)

        return value().__await__()


def _gai(ip: str, port: int) -> tuple[Any, ...]:
    addr = ipaddress.ip_address(ip)
    if addr.version == 4:
        return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
    return (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0))


@dataclass
class Fakes:
    table: dict[str, list[str]] = field(default_factory=dict)
    resolve_calls: list[str] = field(default_factory=list)
    connect_calls: list[tuple[str, int]] = field(default_factory=list)
    blocked: list[dict[str, str]] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    callback_threads: set[str] = field(default_factory=set)
    upstream_port: int = 0
    # a host whose answer changes after the first lookup (DNS rebinding)
    rebind: dict[str, list[str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def resolve(self, host: Any, port: Any = None, *a: Any, **k: Any) -> _Answer:
        h = host.decode() if isinstance(host, bytes) else str(host)
        with self.lock:
            first = h not in self.resolve_calls
            self.resolve_calls.append(h)
        p = int(port) if port not in (None, "") else 443
        bare = h.strip("[]")
        try:
            ipaddress.ip_address(bare)
            return _Answer([_gai(bare, p)])
        except ValueError:
            pass
        if not first and h in self.rebind:
            return _Answer([_gai(ip, p) for ip in self.rebind[h]])
        if h not in self.table:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return _Answer([_gai(ip, p) for ip in self.table[h]])

    async def connect(self, host: Any, port: Any, *a: Any, **k: Any) -> Any:
        with self.lock:
            self.connect_calls.append((str(host), int(port)))
        return await asyncio.open_connection("127.0.0.1", self.upstream_port)

    def on_blocked(self, record: Any) -> None:
        with self.lock:
            self.blocked.append(dict(record))
            self.callback_threads.add(threading.current_thread().name)

    def on_host(self, host: Any) -> None:
        with self.lock:
            self.hosts.append(host)
            self.callback_threads.add(threading.current_thread().name)


TABLE = {
    "example.com": ["93.184.215.14"],
    "two.example": ["93.184.215.14", "1.1.1.1"],
    "v6.example": ["2606:4700:4700::1111"],
    "private.example": ["10.0.0.5"],
    "meta.example": ["169.254.169.254"],
    "localtest.example": ["127.0.0.1"],
    "mixed.example": ["93.184.215.14", "127.0.0.1"],
    "mixed6.example": ["2606:4700:4700::1111", "fd00::1"],
    "mapped.example": ["::ffff:10.0.0.1"],
    "cgnat.example": ["100.64.0.1"],
    "2130706433": ["127.0.0.1"],  # what getaddrinfo (inet_aton) makes of the single-decimal form
    "empty.example": [],
    **{f"h{i}.example": [f"93.184.215.{i + 1}"] for i in range(16)},
    **{f"b{i}.example": [f"10.0.0.{i + 1}"] for i in range(16)},
}


@dataclass
class Env:
    proxy: Any
    fakes: Fakes
    upstream: Upstream
    host: str
    port: int


def _build(fakes: Fakes, start_origin: str, **extra: Any) -> Any:
    mod = egress_proxy()
    return mod.EgressProxy(
        start_origin=start_origin,
        on_blocked=fakes.on_blocked,
        on_host=fakes.on_host,
        resolve=fakes.resolve,
        connect=fakes.connect,
        **extra,
    )


@contextmanager
def proxy_env(start_origin: str, **extra: Any) -> Iterator[Env]:
    with loopback_only() as attempts:
        upstream = Upstream()
        fakes = Fakes(table=dict(TABLE), upstream_port=upstream.port)
        proxy = _build(fakes, start_origin, **extra)
        try:
            proxy.start()
            parts = urlsplit(proxy.origin)
            assert parts.hostname and parts.port, f"proxy.origin {proxy.origin!r} has no host:port"
            yield Env(proxy, fakes, upstream, parts.hostname, parts.port)
        finally:
            with suppress(Exception):
                proxy.stop()
            upstream.close()
    assert attempts == [], f"non-loopback socket attempts: {attempts}"


@pytest.fixture
def env() -> Iterator[Env]:
    with proxy_env(f"http://127.0.0.1:{UNUSED_START_PORT}") as e:
        yield e


@pytest.fixture
def start_server() -> Iterator[Upstream]:
    """A loopback stand-in for the ``StartPageServer``: a second ``Upstream``, distinct from the public one, so a
    row can tell which of the two a tunnel reached."""
    s = Upstream()
    try:
        yield s
    finally:
        s.close()


# --- client helpers -------------------------------------------------------------------------------------------


def open_client(e: Env) -> socket.socket:
    s = socket.create_connection((e.host, e.port), timeout=IO_TIMEOUT_S)
    s.settimeout(IO_TIMEOUT_S)
    return s


def read_head(s: socket.socket) -> tuple[int, bytes]:
    """The status code of the response head, and any bytes after the head."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        try:
            chunk = s.recv(4096)
        except ConnectionResetError:
            chunk = b""
        if not chunk:
            break
        buf += chunk
    line = buf.split(b"\r\n", 1)[0]
    parts = line.split()
    assert len(parts) >= 2 and parts[0].startswith(b"HTTP/"), (
        f"no HTTP response head: {buf[:200]!r}"
    )
    _head, _, rest = buf.partition(b"\r\n\r\n")
    return int(parts[1]), rest


def request(e: Env, raw: bytes) -> tuple[socket.socket, int, bytes]:
    s = open_client(e)
    s.sendall(raw)
    code, rest = read_head(s)
    return s, code, rest


def connect_line(target: str) -> bytes:
    return f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode()


def tunnel(e: Env, target: str) -> tuple[socket.socket, int, bytes]:
    return request(e, connect_line(target))


def recv_exact(s: socket.socket, n: int, already: bytes = b"") -> bytes:
    buf = already
    while len(buf) < n:
        chunk = s.recv(65536)
        assert chunk, f"tunnel closed after {len(buf)} of {n} bytes"
        buf += chunk
    return buf


def eventually(cond: Callable[[], bool], timeout: float = 2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def closed(s: socket.socket, timeout: float = 2.0) -> bool:
    """True when the peer closed (EOF or reset) within ``timeout``; data before the close is drained."""
    s.settimeout(timeout)
    end = time.monotonic() + timeout
    try:
        while time.monotonic() < end:
            if not s.recv(65536):
                return True
    except (ConnectionResetError, BrokenPipeError):
        return True
    except TimeoutError:
        return False
    return False


def assert_refused(e: Env, target: str, code: int, record_host: str | None) -> None:
    s, got, _ = tunnel(e, target)
    s.close()
    assert got == code, f"CONNECT {target}: {got}, want {code}"
    if record_host is not None:
        assert eventually(lambda: rec(record_host) in e.fakes.blocked), (
            f"CONNECT {target}: no egress record {rec(record_host)}; got {e.fakes.blocked}"
        )


# --- A-5 (a), (f), (g): allowed tunnels -----------------------------------------------------------------------


def test_connect_public_pipes_both_ways(env: Env) -> None:
    """A-5 (a): a public resolution is ``200``, upstream bytes reach the client and client bytes reach upstream;
    ``on_host`` gets the host only, once."""
    s, code, rest = tunnel(env, "example.com:443")
    try:
        assert code == 200, f"CONNECT example.com:443 -> {code}"
        assert recv_exact(s, 6, rest)[:6] == b"HELLO\n", "upstream-to-client bytes did not arrive"
        payload = (b"abcdefghij" * 20_000)[:200_000]
        s.sendall(payload)
        assert recv_exact(s, len(payload)) == payload.upper(), (
            "client-to-upstream bytes did not round-trip"
        )
    finally:
        s.close()
    assert env.fakes.connect_calls == [("93.184.215.14", 443)]
    assert env.fakes.hosts == ["example.com"]
    assert env.fakes.blocked == []


def test_resolver_called_once_and_connector_gets_the_ip(env: Env) -> None:
    """A-5 (f), (g), the TOCTOU row: the name resolves public once, then would rebind to loopback. The resolver is
    called exactly once per CONNECT and ``connect`` receives the checked IP literal, never the hostname."""
    env.fakes.table["rebind.example"] = ["93.184.215.14"]
    env.fakes.rebind["rebind.example"] = ["127.0.0.1"]
    s, code, _ = tunnel(env, "rebind.example:443")
    s.close()
    assert code == 200
    assert env.fakes.resolve_calls.count("rebind.example") == 1, env.fakes.resolve_calls
    assert env.fakes.connect_calls == [("93.184.215.14", 443)], env.fakes.connect_calls

    s, code, _ = tunnel(env, "v6.example:443")
    s.close()
    assert code == 200
    assert env.fakes.resolve_calls.count("v6.example") == 1
    assert env.fakes.connect_calls[-1] == ("2606:4700:4700::1111", 443)

    s, code, _ = tunnel(env, "two.example:443")
    s.close()
    assert code == 200
    assert env.fakes.resolve_calls.count("two.example") == 1
    assert env.fakes.connect_calls[-1] in {("93.184.215.14", 443), ("1.1.1.1", 443)}
    names = {h for h, _ in env.fakes.connect_calls}
    assert not names & {"rebind.example", "v6.example", "two.example"}, "connect got a hostname"
    assert len(env.fakes.connect_calls) == 3


# --- A-5 (b): refusals ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["private.example", "meta.example", "localtest.example", "mapped.example", "cgnat.example"],
)
def test_connect_private_resolution_refused_without_connect(env: Env, host: str) -> None:
    """A-5 (b): a name resolving to a non-public address is ``403``, logged as ``{host, scheme}``, never connected."""
    assert_refused(env, f"{host}:443", 403, host)
    assert env.fakes.connect_calls == []
    assert env.fakes.hosts == []
    assert env.upstream.accepted == 0


@pytest.mark.parametrize("host", ["mixed.example", "mixed6.example"])
def test_mixed_resolution_refused(env: Env, host: str) -> None:
    """A-5 (b) "any resolved address is not public": one public and one non-public answer is still ``403``, whatever
    the order (the public one is first in the table, so "connect to the first public" would pass a wrong proxy)."""
    assert_refused(env, f"{host}:443", 403, host)
    assert env.fakes.connect_calls == []
    assert env.fakes.hosts == []


@pytest.mark.parametrize("host", ["nx.example", "empty.example"])
def test_resolution_failure_refused_and_logged(env: Env, host: str) -> None:
    """A-5 (b): a resolution failure (``gaierror``, and an empty answer, where "every address is public" would be
    vacuously true) is ``403`` and logged the same way."""
    assert_refused(env, f"{host}:443", 403, host)
    assert env.fakes.connect_calls == []
    assert env.fakes.hosts == []


@pytest.mark.parametrize(
    "target", ["example.com:8443", "example.com:80", "example.com:22", "example.com:4430"]
)
def test_port_not_443_refused_without_resolving(env: Env, target: str) -> None:
    """A-5 (b): any port but 443 is ``403`` and logged, before resolving and with no connect attempted."""
    assert_refused(env, target, 403, "example.com")
    assert env.fakes.resolve_calls == []
    assert env.fakes.connect_calls == []
    assert env.fakes.hosts == []


@pytest.mark.parametrize(
    ("target", "record_host"),
    [
        ("10.0.0.1:443", "10.0.0.1"),
        ("127.0.0.1:443", "127.0.0.1"),
        ("[::1]:443", "::1"),
        ("[::ffff:127.0.0.1]:443", "::ffff:127.0.0.1"),
        ("169.254.169.254:443", "169.254.169.254"),
        ("100.64.0.1:443", "100.64.0.1"),
        ("2130706433:443", "2130706433"),
    ],
)
def test_ip_literal_private_refused(env: Env, target: str, record_host: str) -> None:
    """A-5 (b): a non-public IP literal target (and the single-decimal form of 127.0.0.1) is ``403``, logged, never
    connected. The start origin's own host:port is the O-A2 (2) rows below."""
    assert_refused(env, target, 403, record_host)
    assert env.fakes.connect_calls == []
    assert env.upstream.accepted == 0


# --- O-A2 (2): the start-origin tunnel ------------------------------------------------------------------------


def start_target(start: Upstream) -> str:
    return f"127.0.0.1:{start.port}"


@pytest.mark.parametrize("extra", [{"tunnel_start_origin": False}, {}], ids=["False", "default"])
def test_start_origin_refused_without_the_flag(
    start_server: Upstream, extra: dict[str, Any]
) -> None:
    """O-A2 (2), the re-pinned ``127.0.0.1:<start>`` row: with ``tunnel_start_origin=False`` (and by default),
    ``CONNECT`` to the start origin's own host:port is loopback like any other: ``403``, logged, never reached."""
    with proxy_env(f"http://{start_target(start_server)}", **extra) as e:
        assert_refused(e, start_target(start_server), 403, "127.0.0.1")
        assert e.fakes.connect_calls == []
        assert e.fakes.hosts == []
        time.sleep(0.05)
        assert start_server.accepted == 0
        assert e.upstream.accepted == 0


def test_start_origin_tunnelled_with_the_flag(start_server: Upstream) -> None:
    """O-A2 (2): with ``tunnel_start_origin=True``, ``CONNECT <start host>:<start port>`` is ``200`` and piped both
    ways to the start server (not the public upstream), with no resolve, no ``on_host`` and no egress record."""
    with proxy_env(f"http://{start_target(start_server)}", tunnel_start_origin=True) as e:
        s, code, rest = tunnel(e, start_target(start_server))
        try:
            assert code == 200, f"CONNECT {start_target(start_server)} -> {code}"
            assert recv_exact(s, 6, rest)[:6] == b"HELLO\n"
            s.sendall(b"get /web.html")
            assert recv_exact(s, 13) == b"GET /WEB.HTML"
        finally:
            s.close()
        assert start_server.accepted == 1
        assert e.upstream.accepted == 0, "the start-origin tunnel reached the public stand-in"
        assert e.fakes.resolve_calls == []
        assert e.fakes.hosts == []
        assert e.fakes.blocked == []


@pytest.mark.parametrize(
    ("target", "record_host"),
    [
        ("NEIGHBOUR", "127.0.0.1"),  # the start port + 1
        ("127.0.0.1:443", "127.0.0.1"),
        ("127.0.0.1:80", "127.0.0.1"),
        ("[::1]:START", "::1"),  # the start port on another spelling of loopback
        ("127.0.0.2:START", "127.0.0.2"),
    ],
)
def test_other_loopback_refused_with_the_flag(
    start_server: Upstream, target: str, record_host: str
) -> None:
    """O-A2 (2): only the exact start host:port qualifies. With ``tunnel_start_origin=True``, a neighbouring
    loopback port, ``127.0.0.1`` on other ports, and the start port on another loopback address are still ``403``,
    logged, never reached."""
    port = start_server.port
    target = (
        f"127.0.0.1:{port + 1}" if target == "NEIGHBOUR" else target.replace("START", str(port))
    )
    with proxy_env(f"http://{start_target(start_server)}", tunnel_start_origin=True) as e:
        assert_refused(e, target, 403, record_host)
        assert e.fakes.connect_calls == []
        assert e.fakes.hosts == []
        time.sleep(0.05)
        assert start_server.accepted == 0
        assert e.upstream.accepted == 0


def test_public_ip_literal_allowed(env: Env) -> None:
    """A-5 (a), the too-wide direction of the literal rows: a public IP literal on 443 tunnels, to that address."""
    s, code, _ = tunnel(env, "93.184.215.14:443")
    s.close()
    assert code == 200
    assert env.fakes.connect_calls == [("93.184.215.14", 443)]


# --- A-5 (c): other request forms -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n",
        b"GET http://93.184.215.14/x HTTP/1.1\r\nHost: 93.184.215.14\r\n\r\n",
        b"POST http://10.0.0.1:80/ HTTP/1.1\r\nHost: 10.0.0.1\r\nContent-Length: 0\r\n\r\n",
        b"GET http://127.0.0.1:1/ HTTP/1.1\r\nHost: 127.0.0.1:1\r\n\r\n",
    ],
)
def test_absolute_form_http_refused(env: Env, raw: bytes) -> None:
    """A-5 (c): an absolute-form plain-http request to anything but the start origin is ``403``, never forwarded."""
    s, code, _ = request(env, raw)
    s.close()
    assert code == 403, f"{raw.splitlines()[0]!r} -> {code}"
    time.sleep(0.05)
    assert env.fakes.connect_calls == []
    assert env.fakes.hosts == []
    assert env.upstream.accepted == 0


@pytest.mark.parametrize(
    "raw", [b"garbage\r\n\r\n", b"CONNECT\r\n\r\n", b"\x00\x01\x02\xff\r\n\r\n"]
)
def test_malformed_and_oversized_heads_400(env: Env, raw: bytes) -> None:
    """A-5 (c): a malformed request line is ``400``; a head over 8 KiB, with or without its terminator, is ``400``
    and the connection closes. None of them resolves or connects."""
    s, code, _ = request(env, raw)
    s.close()
    assert code == 400, f"{raw!r} -> {code}"

    big = b"CONNECT example.com:443 HTTP/1.1\r\nX-Pad: " + b"a" * 9000 + b"\r\n\r\n"
    s, code, _ = request(env, big)
    assert code == 400, f"9 KiB head -> {code}"
    assert closed(s), "connection not closed after the oversized head"
    s.close()

    s = open_client(env)
    with suppress(OSError):
        s.sendall(
            b"CONNECT example.com:443 HTTP/1.1\r\nX-Pad: " + b"a" * 9000
        )  # no terminator, ever
    code, _ = read_head(s)
    assert code == 400, f"unterminated 9 KiB head -> {code}"
    assert closed(s), "connection not closed after the unterminated oversized head"
    s.close()

    assert env.fakes.resolve_calls == []
    assert env.fakes.connect_calls == []


# --- A-5 (d): caps --------------------------------------------------------------------------------------------


def test_tunnel_cap_503(env: Env) -> None:
    """A-5 (d): 64 concurrent tunnels are allowed, the 65th is ``503``, and closing one frees a slot."""
    open_tunnels: list[socket.socket] = []
    try:
        for i in range(64):
            s, code, _ = tunnel(env, "example.com:443")
            open_tunnels.append(s)
            assert code == 200, f"tunnel {i + 1} of 64 -> {code}"
        s, code, _ = tunnel(env, "example.com:443")
        s.close()
        assert code == 503, f"tunnel 65 -> {code}"

        open_tunnels.pop().close()

        def slot_free() -> bool:
            t, c, _ = tunnel(env, "example.com:443")
            if c == 200:
                open_tunnels.append(t)
                return True
            t.close()
            return False

        assert eventually(slot_free, 3.0), "a closed tunnel did not free its slot"
    finally:
        for s in open_tunnels:
            s.close()


# --- loopback binding, shutdown -------------------------------------------------------------------------------


def test_bound_to_loopback_only(env: Env) -> None:
    """A-5: the listener is ``127.0.0.1`` on an OS-chosen port (the trap in the fixture would have recorded a
    wildcard or non-loopback bind/listen), and ``.origin`` names it."""
    assert env.host == "127.0.0.1", env.proxy.origin
    assert env.port > 0
    assert urlsplit(env.proxy.origin).scheme == "http"
    s = open_client(env)
    assert s.getpeername()[0] == "127.0.0.1"
    s.close()

    other = _build(env.fakes, f"http://127.0.0.1:{UNUSED_START_PORT}")
    other.start()
    try:
        assert urlsplit(other.origin).port != env.port, "port is not chosen by the OS (127.0.0.1:0)"
    finally:
        other.stop()


def test_stop_closes_tunnels_within_1s() -> None:
    """A-5 (e): with three open tunnels and one client mid-head, ``stop()`` returns within 1 s, every client sees
    the close, the listener is gone and the proxy's threads are joined."""
    with loopback_only() as attempts:
        upstream = Upstream()
        fakes = Fakes(table=dict(TABLE), upstream_port=upstream.port)
        before = set(threading.enumerate())
        proxy = _build(fakes, f"http://127.0.0.1:{UNUSED_START_PORT}")
        proxy.start()
        parts = urlsplit(proxy.origin)
        e = Env(proxy, fakes, upstream, parts.hostname or "", parts.port or 0)
        clients: list[socket.socket] = []
        try:
            for _ in range(3):
                s, code, _ = tunnel(e, "example.com:443")
                clients.append(s)
                assert code == 200
            half = open_client(e)
            half.sendall(b"CONNECT example.com:443 HTTP/1.1\r\n")  # never finished
            clients.append(half)
            mine = {
                t for t in set(threading.enumerate()) - before if not t.name.startswith("upstream")
            }
            assert mine, "the proxy started no thread of its own (A-5: its own thread and loop)"

            t0 = time.monotonic()
            proxy.stop()
            elapsed = time.monotonic() - t0
            assert elapsed <= 1.0, f"stop() took {elapsed:.2f} s"
            assert [closed(c, 1.0) for c in clients] == [True] * 4, (
                "a client was not closed by stop()"
            )
            alive = [t.name for t in mine if t.is_alive()]
            assert alive == [], f"proxy threads not joined: {alive}"
            with pytest.raises(OSError):
                socket.create_connection((e.host, e.port), timeout=1).close()
        finally:
            for c in clients:
                c.close()
            with suppress(Exception):
                proxy.stop()
            upstream.close()
    assert attempts == []


# --- A-5 (d) idle close and the head-read timeout (O-A1 (4): ``IDLE_TUNNEL_S``, patched small) ------------------

SHORT_IDLE_S = 0.4


@pytest.fixture
def short_idle(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(egress_proxy(), "IDLE_TUNNEL_S", SHORT_IDLE_S)
    return SHORT_IDLE_S


def seconds_until_closed(s: socket.socket, limit: float) -> float | None:
    """Seconds from now until the peer closes (EOF or reset), draining any reply; ``None`` if still open at
    ``limit``."""
    t0 = time.monotonic()
    while (left := limit - (time.monotonic() - t0)) > 0:
        s.settimeout(left)
        try:
            if not s.recv(65536):
                return time.monotonic() - t0
        except (ConnectionResetError, BrokenPipeError):
            return time.monotonic() - t0
        except TimeoutError:
            return None
    return None


def test_idle_tunnel_closes(short_idle: float, env: Env) -> None:
    """A-5 (d): an open tunnel that moves no byte either way is closed after ``IDLE_TUNNEL_S`` (0.4 s here): not
    before about 0.3 s (too wide: closing at once), not later than 1.2 s (absent: never closing)."""
    s, code, rest = tunnel(env, "example.com:443")
    try:
        assert code == 200
        assert recv_exact(s, 6, rest)[:6] == b"HELLO\n"  # the last byte that moves
        took = seconds_until_closed(s, 2.0)
        assert took is not None, "an idle tunnel stayed open for 2 s with IDLE_TUNNEL_S = 0.4"
        assert 0.3 <= took <= 1.2, (
            f"idle tunnel closed after {took:.2f} s, want about {short_idle} s"
        )
    finally:
        s.close()


def test_active_tunnel_stays_open(short_idle: float, env: Env) -> None:
    """A-5 (d): traffic resets the idle clock. One byte every 0.2 s for 1.6 s (four times ``IDLE_TUNNEL_S``) keeps
    the tunnel open, every byte round-trips; it then closes once the traffic stops."""
    s, code, rest = tunnel(env, "example.com:443")
    try:
        assert code == 200
        assert recv_exact(s, 6, rest)[:6] == b"HELLO\n"
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < 1.6:
            time.sleep(0.2)
            s.sendall(b"x")
            s.settimeout(IO_TIMEOUT_S)
            try:
                got = s.recv(1)
            except ConnectionResetError:
                got = b""
            assert got == b"X", (
                f"tunnel closed under traffic after {time.monotonic() - t0:.2f} s ({n} bytes)"
            )
            n += 1
        assert time.monotonic() - t0 >= 1.5
        assert seconds_until_closed(s, 2.0) is not None, (
            "the tunnel did not close once traffic stopped"
        )
    finally:
        s.close()


def test_half_sent_head_closes(short_idle: float, env: Env) -> None:
    """The head-read timeout: a request line with no terminating blank line, then silence, is closed within about
    1 s of ``IDLE_TUNNEL_S`` = 0.4 s, with no resolve and no connect."""
    s = open_client(env)
    try:
        s.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n")
        took = seconds_until_closed(s, 2.0)
        assert took is not None, "a half-sent head held the connection open for 2 s"
        assert took <= 1.2, f"half-sent head closed after {took:.2f} s"
    finally:
        s.close()
    assert env.fakes.resolve_calls == []
    assert env.fakes.connect_calls == []


def test_trickled_head_is_cut(short_idle: float, env: Env) -> None:
    """The slowloris guard: a head trickled one byte every 0.1 s, never reaching its terminator or 8 KiB, is cut
    within about 1 s. The timeout bounds the whole head read, so bytes arriving do not extend it."""
    s = open_client(env)
    t0 = time.monotonic()
    cut_at: float | None = None
    try:
        s.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nX-Slow: ")
        while time.monotonic() - t0 < 2.5:
            try:
                s.sendall(b"a")
            except OSError:
                cut_at = time.monotonic() - t0
                break
            s.settimeout(0.1)
            try:
                if not s.recv(65536):
                    cut_at = time.monotonic() - t0
                    break
            except (ConnectionResetError, BrokenPipeError):
                cut_at = time.monotonic() - t0
                break
            except TimeoutError:
                continue
        assert cut_at is not None, "a byte-at-a-time head was not cut in 2.5 s"
        assert cut_at <= 1.2, f"trickled head cut after {cut_at:.2f} s"
    finally:
        s.close()
    assert env.fakes.resolve_calls == []
    assert env.fakes.connect_calls == []


# --- A-6: callbacks under concurrency -------------------------------------------------------------------------


def test_callbacks_threadsafe_no_loss(env: Env) -> None:
    """A-6: 8 allowed and 8 refused CONNECTs at once, from 16 client threads released by one barrier, while the 8
    allowed tunnels stay open together: every ``on_host`` and ``on_blocked`` record arrives, none twice."""
    n = 8
    barrier = threading.Barrier(2 * n)
    codes: dict[str, int] = {}
    keep: list[socket.socket] = []
    lock = threading.Lock()
    errors: list[BaseException] = []

    def go(host: str) -> None:
        try:
            barrier.wait(IO_TIMEOUT_S)
            s, code, _ = tunnel(env, f"{host}:443")
            with lock:
                codes[host] = code
                keep.append(s)
        except BaseException as exc:  # surfaced below as an assertion
            errors.append(exc)

    allowed = [f"h{i}.example" for i in range(n)]
    refused = [f"b{i}.example" for i in range(n)]
    threads = [threading.Thread(target=go, args=(h,)) for h in allowed + refused]
    for t in threads:
        t.start()
    for t in threads:
        t.join(IO_TIMEOUT_S * 2)
    try:
        assert errors == [], errors
        assert {h: codes.get(h) for h in allowed} == dict.fromkeys(allowed, 200)
        assert {h: codes.get(h) for h in refused} == dict.fromkeys(refused, 403)
        assert eventually(lambda: len(env.fakes.hosts) >= n and len(env.fakes.blocked) >= n)
        time.sleep(0.05)
        assert sorted(env.fakes.hosts) == sorted(allowed), env.fakes.hosts
        assert sorted(r["host"] for r in env.fakes.blocked) == sorted(refused), env.fakes.blocked
        assert all(r == rec(r["host"]) for r in env.fakes.blocked), env.fakes.blocked
        assert sorted(ip for ip, _ in env.fakes.connect_calls) == sorted(
            f"93.184.215.{i + 1}" for i in range(n)
        )
    finally:
        for s in keep:
            s.close()
