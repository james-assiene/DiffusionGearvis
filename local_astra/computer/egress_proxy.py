"""The web-mode egress proxy (T-0160 A-5): an HTTP CONNECT proxy bound to ``127.0.0.1:0``, on its own thread and
its own asyncio loop (never the session loop, which carries the audio path).

Chromium sends all of its page traffic here. For ``CONNECT host:443`` the proxy resolves the host **once**, refuses
the tunnel unless every resolved address is public (``fence.is_public_address``), and connects to one of the exact
addresses it checked, never to the name (no TOCTOU, A-5 (f)/(g)). Everything else is refused before any resolution
or connection:

- a port other than 443, or a loopback target: ``403``;
- an absolute-form request (``GET http://…``, plain http): ``403``;
- a malformed request line, or a request head over 8 KiB: ``400`` and close;
- a tunnel beyond the 64th concurrent one: ``503``.

The one possible loopback exception, off by default, is ``tunnel_start_origin=True``: ``CONNECT <start host>:<start
port>`` is then tunnelled straight to the start-page server. P1 (2026-10-26) measured that Playwright's
``route.fetch``, which serves the start origin, runs in Playwright's own process, uses the launch proxy, ignores
Chromium's bypass list and sends ``CONNECT 127.0.0.1:<port>`` even for plain http. That tunnel resolves nothing,
never uses ``connect`` and reports no host.

Each refusal is reported once to ``on_blocked`` as ``fence.egress_record("https://host/")`` (``{host, scheme}``
only); each opened public tunnel reports its host to ``on_host``. Both callbacks run on the proxy thread. Idle
tunnels close after ``IDLE_TUNNEL_S``; ``stop()`` closes the listener and every tunnel and joins the thread within
1 s.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import ipaddress
import logging
import socket
import threading
from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

from local_astra.computer import fence

__all__ = [
    "CONNECT_TIMEOUT_S",
    "HEAD_LIMIT",
    "IDLE_TUNNEL_S",
    "MAX_TUNNELS",
    "EgressProxy",
]

_log = logging.getLogger(__name__)

_HOST = "127.0.0.1"
#: Largest request head accepted, CRLFCRLF included (A-5 (c)).
HEAD_LIMIT = 8 * 1024
#: Concurrent tunnels beyond this get ``503`` (A-5 (d)). Read at use time.
MAX_TUNNELS = 64
#: A tunnel with no byte in either direction for this long is closed, as is a client that sends no complete head
#: (A-5 (d), O-A1 (4)). Read at use time, so a row may patch it small.
IDLE_TUNNEL_S = 60.0
#: Bound of one connect attempt to one checked address; the next address is tried after it (P1 saw a stalled IPv6
#: connect hold a navigation past its 10 s timeout).
CONNECT_TIMEOUT_S = 4.0
_STOP_TIMEOUT_S = 1.0
_CHUNK = 64 * 1024

OnBlocked = Callable[[dict[str, str]], None]
OnHost = Callable[[str], None]
#: ``resolve(host, port)``: ``getaddrinfo``-shaped 5-tuples (or IP strings), returned directly or awaitable (O-A1).
Resolve = Callable[[str, int], Awaitable[Iterable[Any]] | Iterable[Any]]
Connect = Callable[[str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]]

_REASONS = {
    200: "Connection Established",
    400: "Bad Request",
    403: "Forbidden",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


async def _default_resolve(host: str, port: int) -> list[Any]:
    return await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)


async def _default_connect(ip: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection(ip, port)


def _address(item: Any) -> str:
    """One resolver result as an IP string: a string as is, a ``getaddrinfo`` tuple by its sockaddr."""
    if isinstance(item, tuple) and len(item) == 5 and isinstance(item[4], tuple):
        return str(item[4][0])
    return str(item)


def _is_loopback_host(host: str) -> bool:
    name = host.lower().rstrip(".")
    if name in fence.LOOPBACK_HOSTS or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _split_authority(target: str) -> tuple[str, int] | None:
    """``host:port`` (``[v6]:port`` bracketed) of a CONNECT target; ``None`` when malformed."""
    if target.startswith("["):
        end = target.find("]")
        if end < 0 or target[end + 1 : end + 2] != ":":
            return None
        host, port_s = target[1:end], target[end + 2 :]
    else:
        host, sep, port_s = target.rpartition(":")
        if not sep or ":" in host:
            return None
    if not host or not port_s.isascii() or not port_s.isdigit() or len(port_s) > 5:
        return None
    port = int(port_s)
    if not 0 < port < 65536:
        return None
    return host, port


def _start_target(start_origin: str | None) -> tuple[str, int] | None:
    """``(host, port)`` of a loopback start origin; ``None`` when there is none or it is not loopback."""
    if start_origin is None or not fence.is_loopback_url(start_origin):
        return None
    try:
        parts = urlsplit(start_origin)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if host is None:
        return None
    return host.lower(), port if port is not None else 80


def _record_url(host: str) -> str:
    return f"https://[{host}]/" if ":" in host else f"https://{host}/"


class EgressProxy:
    """The fenced CONNECT proxy (module docstring). ``start()`` binds and returns once listening; ``origin`` is
    ``http://127.0.0.1:<port>``; ``stop()`` is idempotent."""

    def __init__(
        self,
        start_origin: str | None = None,
        on_blocked: OnBlocked | None = None,
        on_host: OnHost | None = None,
        resolve: Resolve | None = None,
        connect: Connect | None = None,
        *,
        tunnel_start_origin: bool = False,
    ) -> None:
        self._start = _start_target(start_origin) if tunnel_start_origin else None
        self._on_blocked = on_blocked
        self._on_host = on_host
        self._resolve: Resolve = resolve if resolve is not None else _default_resolve
        self._connect: Connect = connect if connect is not None else _default_connect
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: asyncio.Server | None = None
        self._port: int | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._tunnels = 0

    # ------------------------------------------------------------------------------------------ lifecycle

    @property
    def origin(self) -> str:
        if self._port is None:
            raise RuntimeError("the egress proxy is not running; call start() first")
        return f"http://{_HOST}:{self._port}"

    def start(self) -> None:
        """Bind ``127.0.0.1:0`` on a new thread and loop; returns once listening, raises what the bind raised."""
        if self._thread is not None:
            raise RuntimeError("the egress proxy is already running")
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            asyncio.set_event_loop(loop)
            try:
                self._server = loop.run_until_complete(asyncio.start_server(self._handle, _HOST, 0))
                self._port = int(self._server.sockets[0].getsockname()[1])
            except BaseException as exc:  # re-raised by start() on the caller's thread
                failure.append(exc)
                ready.set()
                loop.close()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        thread = threading.Thread(target=run, name="astra-egress-proxy", daemon=True)
        self._loop, self._thread = loop, thread
        thread.start()
        ready.wait()
        if failure:
            thread.join(_STOP_TIMEOUT_S)
            self._loop, self._thread = None, None
            raise failure[0]

    def stop(self) -> None:
        """Close the listener and every tunnel, stop the loop and join the thread; a proxy never started is a
        no-op."""
        loop, thread = self._loop, self._thread
        self._loop, self._thread = None, None
        if loop is None or thread is None:
            return
        # suppressed: the loop already gone, or a tunnel slow to close (the thread is a daemon either way)
        with contextlib.suppress(RuntimeError, TimeoutError):
            asyncio.run_coroutine_threadsafe(self._shutdown(), loop).result(_STOP_TIMEOUT_S * 0.8)
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)
        thread.join(_STOP_TIMEOUT_S * 0.2)
        self._port = None

    async def _shutdown(self) -> None:
        """Stop listening, then cancel every other task on this loop until none is left: the connection handlers,
        and the accepts still in flight (their handler does not exist yet; cancelling the accept closes its
        transport). Every tunnel's sockets are aborted."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _STOP_TIMEOUT_S * 0.6
        server, self._server = self._server, None
        if server is not None:
            server.close()
        me = asyncio.current_task()
        while True:
            for writer in list(self._writers):
                writer.transport.abort()
            others = [t for t in asyncio.all_tasks() if t is not me]
            if not others or loop.time() >= deadline:
                break
            for task in others:
                task.cancel()
            await asyncio.wait(others, timeout=max(0.0, deadline - loop.time()))
        # let the aborted transports' connection_lost callbacks close their sockets
        await asyncio.sleep(0)

    # ------------------------------------------------------------------------------------------ callbacks

    def _blocked(self, url: str) -> None:
        if self._on_blocked is None:
            return
        try:
            self._on_blocked(fence.egress_record(url))
        except Exception:  # a consumer's bug never takes the proxy down; it is logged
            _log.exception("egress proxy: on_blocked raised")

    def _tunnelled(self, host: str) -> None:
        if self._on_host is None:
            return
        try:
            self._on_host(host)
        except Exception:  # as above
            _log.exception("egress proxy: on_host raised")

    # ------------------------------------------------------------------------------------------ connections

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            await self._serve(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError, TimeoutError):
            pass
        finally:
            self._writers.discard(writer)
            writer.close()

    async def _read_head(self, reader: asyncio.StreamReader) -> tuple[bytes, bytes] | None:
        """``(head, rest)``: the head before CRLFCRLF and the bytes after it; ``None`` when over ``HEAD_LIMIT``.
        Raises ``IncompleteReadError`` when the client closes first."""
        buf = b""
        while True:
            end = buf.find(b"\r\n\r\n")
            if end >= 0:
                return (buf[:end], buf[end + 4 :]) if end + 4 <= HEAD_LIMIT else None
            if len(buf) >= HEAD_LIMIT:
                return None
            chunk = await reader.read(HEAD_LIMIT + 4 - len(buf))
            if not chunk:
                raise asyncio.IncompleteReadError(buf, None)
            buf += chunk

    async def _reply(self, writer: asyncio.StreamWriter, status: int) -> None:
        if status == 200:
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        else:
            body = f"{status} {_REASONS[status]}\n".encode()
            writer.write(
                f"HTTP/1.1 {status} {_REASONS[status]}\r\nContent-Type: text/plain\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
        await writer.drain()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        got = await asyncio.wait_for(self._read_head(reader), IDLE_TUNNEL_S)
        if got is None:
            await self._reply(writer, 400)
            return
        head, rest = got
        try:
            line = head.split(b"\r\n", 1)[0].decode("ascii")
        except UnicodeDecodeError:
            await self._reply(writer, 400)
            return
        parts = line.split(" ")
        if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2].startswith("HTTP/1."):
            await self._reply(writer, 400)
            return
        method, target = parts[0], parts[1]
        if method.upper() == "CONNECT":
            await self._serve_connect(target, rest, reader, writer)
        elif "://" in target:
            self._blocked(target)  # absolute-form: plain http through the proxy is never forwarded
            await self._reply(writer, 403)
        else:
            await self._reply(writer, 400)

    async def _serve_connect(
        self, target: str, rest: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        authority = _split_authority(target)
        if authority is None:
            await self._reply(writer, 400)
            return
        host, port = authority
        if self._start is not None and (host.lower(), port) == self._start:
            await self._tunnel(None, [self._start[0]], port, rest, reader, writer)
            return
        if port != 443 or _is_loopback_host(host):
            self._blocked(_record_url(host))
            await self._reply(writer, 403)
            return
        try:
            literal: str | None = str(ipaddress.ip_address(host))
        except ValueError:
            literal = None
        if literal is not None:
            addresses = [literal]
        else:
            try:
                answer: Any = self._resolve(host, port)
                if inspect.isawaitable(answer):
                    answer = await answer
                addresses = [_address(a) for a in answer]
            except (
                Exception
            ):  # NXDOMAIN, timeouts and resolver bugs all refuse the tunnel the same way
                addresses = []
        if not addresses or not all(fence.is_public_address(a) for a in addresses):
            self._blocked(_record_url(host))
            await self._reply(writer, 403)
            return
        await self._tunnel(host, addresses, port, rest, reader, writer)

    async def _tunnel(
        self,
        host: str | None,
        addresses: list[str],
        port: int,
        rest: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Open a tunnel to the first reachable checked address; ``host is None`` is the start-page tunnel."""
        if self._tunnels >= MAX_TUNNELS:
            await self._reply(writer, 503)
            return
        self._tunnels += 1
        try:
            connect = self._connect if host is not None else _default_connect
            upstream = None
            for ip in addresses:
                try:
                    upstream = await asyncio.wait_for(connect(ip, port), CONNECT_TIMEOUT_S)
                    break
                except (OSError, TimeoutError):
                    continue
            if upstream is None:
                await self._reply(writer, 502)
                return
            up_reader, up_writer = upstream
            self._writers.add(up_writer)
            try:
                await self._reply(writer, 200)
                if host is not None:
                    self._tunnelled(host)
                if rest:
                    up_writer.write(rest)
                await self._pipe(reader, writer, up_reader, up_writer)
            finally:
                self._writers.discard(up_writer)
                up_writer.close()
        finally:
            self._tunnels -= 1

    async def _pipe(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        up_reader: asyncio.StreamReader,
        up_writer: asyncio.StreamWriter,
    ) -> None:
        """Copy both ways until either side closes or neither moves a byte for ``IDLE_TUNNEL_S``."""
        loop = asyncio.get_running_loop()
        last = [loop.time()]

        async def copy(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            while True:
                data = await src.read(_CHUNK)
                if not data:
                    return
                last[0] = loop.time()
                dst.write(data)
                await dst.drain()

        async def idle() -> None:
            while (remaining := last[0] + IDLE_TUNNEL_S - loop.time()) > 0:
                await asyncio.sleep(remaining)

        tasks = [
            asyncio.ensure_future(copy(reader, up_writer)),
            asyncio.ensure_future(copy(up_reader, writer)),
            asyncio.ensure_future(idle()),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
