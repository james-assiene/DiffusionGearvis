"""A scripted moshi-server 0.6.4 for T-0007a: an injectable connection *and* a loopback server.

Two faces, one script format (``docs/tasks/T-0007a.md``, "Tests to write"):

* :class:`SttConnectionDouble` — an object with ``send`` / ``recv`` / ``close``, handed to
  ``MoshiSttClient(services, connect=…)``. No socket, so ``tests/unit`` stays I/O-free. The knobs
  are exactly the failure modes the criteria name: never echo a ``Marker``, close after *n* audio
  blocks, stop reading, never finish the close handshake, emit an undecodable frame, raise on
  ``send``.
* :class:`SttLoopbackServer` — the same script over a real ``websockets`` server on 127.0.0.1,
  following the ``tests/fakes/hud_server.py`` precedent, for ``tests/integration``.

Both **record every client message decoded with the real ``msgpack.unpackb``** (:meth:`decoded`):
the client's framing is validated by the production codec, never by a second implementation of it.

The message shapes below are moshi-server 0.6.4's (`kyutai-labs/delayed-streams-modeling`,
`scripts/stt_from_mic_rust_server.py`), the same wire ``scripts/demo/_listen.py`` speaks. All five types
— ``Ready``, ``Step``, ``Word``, ``EndWord`` and ``Marker`` — were *observed* on this box by
T-0007a's group B census (2026-09-14, live 0.6.4 server on CPU), each with the key set its builder
below spells out. The first two runs saw no ``Marker`` echo because their harness stopped feeding
audio after ``mark()``; run 3 kept feeding and the echo arrived, ``{'type': 'Marker', 'id': 1}``,
3.0 s after the mark (the CPU server's step clock trailed the fed audio by ≈ 3 s) —
``logs/t0007a-group-b-census-38a7643.log``, ruling A14. The rule the doubles exist to honour: the
flush arrives only while audio keeps flowing. The census also settled the unit: ``start_time`` and
``stop_time`` are **seconds** (``logs/t0007a-group-b-census-f114adf.log``).
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Any

import msgpack
import websockets
from websockets.exceptions import ConnectionClosedOK

#: 0xC1 is the one byte msgpack never uses, so this frame cannot decode (criterion 5's
#: "undecodable msgpack frame"), whatever the packer on the other side does.
GARBAGE_FRAME = b"\xc1\xc1\xc1"

#: moshi-server steps the model every 80 ms (12.5 Hz), speaking or not.
STEP_MS = 80

#: Every attribute/method ``MoshiSttClient`` must expose for the group-A tests to mean anything.
#: ``feed``/``events``/``close`` come from ``contracts/protocols.py::SttClient``; ``mark``,
#: ``stats`` and ``max_queued_blocks`` are concrete-class surface (T-0007a keeps them off the
#: protocol — ``contracts/protocols.py`` is T-0007's file).
STT_CLIENT_SURFACE = ("feed", "events", "close", "mark", "stats", "url", "max_queued_blocks")

#: Counters ``MoshiSttClient.stats()`` must carry (criteria 3, 4, 7, 8). Extra keys are allowed.
STT_STATS_KEYS = (
    "estimated_word_times",
    "stale_markers",
    "unknown_messages",
    "dropped_blocks",
    "reconnects",
    "queued_blocks",
    # T-0038 criteria 1-3: the counters for the two types the census found on the wire.
    "words_without_end",
    "stray_end_words",
    "ready_seen",
    # T-0051 ruling B41, decided and recorded rather than left to the next reader: the counters
    # criteria 9-11 and ruling B38 add — ``held_word_timeouts``, ``word_delivery_lag_*`` and
    # ``late_end_words`` — are **not** listed here. This tuple is T-0007a's census list, asserted
    # as a subset by ``tests/unit/test_stt_client.py``, a module T-0051 does not own; adding a
    # name here would turn that row red for a reason the task file does not name (the activation
    # rule in `.claude/rules/tests.md`) and would read as a regression against a finished task.
    # A presence-only assertion is also the weaker pin — a key wired to 0 for ever satisfies it —
    # and it would give each name a second home to be renamed in. The counters are pinned by
    # meaning, with their fixtures, in ``tests/unit/test_stt_client_open_word_hold.py``.
)


# --- what the server sends ---------------------------------------------------------------------


def word(text: str, start_time: float | None = None) -> dict[str, Any]:
    """``Word``. ``start_time`` is in **seconds** from the start of *this* connection."""
    msg: dict[str, Any] = {"type": "Word", "text": text}
    if start_time is not None:
        msg["start_time"] = start_time
    return msg


def end_word(stop_time: float) -> dict[str, Any]:
    """``EndWord``: the **end** of the word that arrived last, in **seconds** (T-0038).

    Observed by T-0007a's group B census (2026-09-14, live 0.6.4 server on CPU,
    ``logs/t0007a-group-b-census-f114adf.log``): key set exactly ``{'type', 'stop_time'}``, one
    per ``Word``, e.g. ``{'type': 'EndWord', 'stop_time': 1.44}``. That run saw four of them for
    five words — the fifth was still inside the server's ~0.5 s lag when the client stopped
    reading, which is why T-0038 criterion 1 must also say what becomes of a word whose end
    never arrives.
    """
    return {"type": "EndWord", "stop_time": stop_time}


def step(prs: Sequence[float] = (0.0, 0.0, 0.0, 0.0), step_idx: int = 0) -> dict[str, Any]:
    """``Step``: the pause-probability heads for 0.5 / 1.0 / 2.0 / 3.0 s of silence."""
    return {"type": "Step", "step_idx": step_idx, "prs": list(prs)}


def marker(marker_id: int) -> dict[str, Any]:
    """``Marker``: what the client sends and what the server echoes back unchanged."""
    return {"type": "Marker", "id": marker_id}


def ready() -> dict[str, Any]:
    return {"type": "Ready"}


def unknown(type_name: str = "Frobnicate", **fields: Any) -> dict[str, Any]:
    """A message type the client has never heard of (criterion 3's "anything else")."""
    return {"type": type_name, **fields}


def pack(msg: Mapping[str, Any]) -> bytes:
    return bytes(msgpack.packb(dict(msg), use_bin_type=True))


# --- face (a): the injected connection ----------------------------------------------------------


class SttConnectionDouble:
    """One scripted connection. ``send``/``recv``/``close`` is the whole seam the client sees."""

    def __init__(
        self,
        *,
        echo_markers: bool = True,
        close_after_audio: int | None = None,
        stop_reading: bool = False,
        garbage_after_audio: int | None = None,
        hang_close: bool = False,
        send_error: BaseException | None = None,
        name: str = "conn",
    ) -> None:
        self.echo_markers = echo_markers
        self.close_after_audio = close_after_audio
        self.stop_reading = stop_reading
        self.garbage_after_audio = garbage_after_audio
        self.hang_close = hang_close
        self.send_error = send_error
        self.name = name

        self.raw_sent: list[bytes] = []
        self.audio_count = 0
        self.close_calls = 0
        self.closed = False
        self.send_calls = 0
        self._outbound: asyncio.Queue[bytes | BaseException] = asyncio.Queue()
        self._blocked = asyncio.Event()  # never set: "the peer stopped reading"

    # --- the connection face --------------------------------------------------------------

    async def send(self, data: bytes) -> None:
        self.send_calls += 1
        if self.closed:
            raise ConnectionClosedOK(None, None)
        raw = bytes(data)
        self.raw_sent.append(raw)
        msg = msgpack.unpackb(raw)  # the real codec, never a second implementation
        if self.send_error is not None:
            raise self.send_error
        if isinstance(msg, dict) and msg.get("type") == "Audio":
            self.audio_count += 1
            if self.garbage_after_audio == self.audio_count:
                self.push_raw(GARBAGE_FRAME)
            if self.close_after_audio == self.audio_count:
                self.server_close()
        if isinstance(msg, dict) and msg.get("type") == "Marker" and self.echo_markers:
            self.push(dict(msg))
        if self.stop_reading:
            await self._blocked.wait()  # this send never completes

    async def recv(self) -> bytes:
        item = await self._outbound.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        self.close_calls += 1
        if self.hang_close:
            # The measured hang: "closing down reliably blocks … cost 10+ seconds of dead air"
            # (scripts/demo/_listen.py::deliver). The client must not wait for this.
            await asyncio.Event().wait()
        self.closed = True
        self._outbound.put_nowait(ConnectionClosedOK(None, None))

    # --- the test face --------------------------------------------------------------------

    def push(self, msg: Mapping[str, Any]) -> None:
        """Queue one server message for the client to ``recv``."""
        self._outbound.put_nowait(pack(msg))

    def push_raw(self, raw: bytes) -> None:
        self._outbound.put_nowait(raw)

    def push_many(self, msgs: Sequence[Mapping[str, Any]]) -> None:
        for m in msgs:
            self.push(m)

    def release(self) -> None:
        """A peer that had stopped reading starts draining again (T-0007a ruling A7).

        Unparks the ``send`` that is currently blocked and lets every later one through, so a
        test can fill the outbound buffer deterministically and *then* look at the wire.
        """
        self.stop_reading = False
        self._blocked.set()

    def server_close(self) -> None:
        """The server drops the connection (criterion 5's "server-side close")."""
        self.closed = True
        self._outbound.put_nowait(ConnectionClosedOK(None, None))

    def decoded(self) -> list[Any]:
        return [msgpack.unpackb(raw) for raw in list(self.raw_sent)]

    def messages_of(self, type_name: str) -> list[dict[str, Any]]:
        return [m for m in self.decoded() if isinstance(m, dict) and m.get("type") == type_name]

    def audio_messages(self) -> list[dict[str, Any]]:
        return self.messages_of("Audio")

    def marker_messages(self) -> list[dict[str, Any]]:
        return self.messages_of("Marker")


class SttConnector:
    """The ``connect=`` seam: hands out doubles in order and records every URL it was given.

    Connection attempts past the script raise ``ConnectionRefusedError`` rather than inventing a
    connection, so "the client reconnects for ever" shows up as a long ``urls`` list (criterion 7)
    instead of as a hang.
    """

    def __init__(
        self,
        script: Sequence[SttConnectionDouble | BaseException] = (),
        *,
        factory: Callable[[int], SttConnectionDouble] | None = None,
    ) -> None:
        self.script = list(script)
        self.factory = factory
        self.urls: list[str] = []
        self.handed_out: list[SttConnectionDouble] = []

    @property
    def attempts(self) -> int:
        return len(self.urls)

    async def __call__(self, url: str) -> SttConnectionDouble:
        i = len(self.urls)
        self.urls.append(url)
        entry: SttConnectionDouble | BaseException | None = None
        if i < len(self.script):
            entry = self.script[i]
        elif self.factory is not None:
            entry = self.factory(i)
        if entry is None:
            raise ConnectionRefusedError(f"no connection left in the script (attempt {i + 1})")
        if isinstance(entry, BaseException):
            raise entry
        self.handed_out.append(entry)
        return entry


# --- face (b): a real loopback websockets server -------------------------------------------------


class SttLoopbackServer:
    """``ws://127.0.0.1:<port>/api/asr-streaming`` speaking the same script over a real socket."""

    def __init__(
        self,
        *,
        echo_markers: bool = True,
        path: str = "/api/asr-streaming",
        host: str = "127.0.0.1",
    ) -> None:
        self.echo_markers = echo_markers
        self.path = path
        self.host = host
        self.port = 0
        self.connections = 0
        self.request_paths: list[str] = []
        self.peers: list[tuple[str, int]] = []
        self.raw_received: list[bytes] = []
        self.client_gone = False
        self._lock = threading.Lock()
        self._outbox: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()

    # --- lifecycle ------------------------------------------------------------------------

    def start(self) -> SttLoopbackServer:
        self._thread = threading.Thread(target=self._run, name="stt-double", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("stt loopback double did not start")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> SttLoopbackServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.path}"

    # --- the test face --------------------------------------------------------------------

    def push(self, msg: Mapping[str, Any]) -> None:
        self._outbox.put(pack(msg))

    def push_raw(self, raw: bytes) -> None:
        self._outbox.put(raw)

    def snapshot(self) -> list[bytes]:
        with self._lock:
            return list(self.raw_received)

    def decoded(self) -> list[Any]:
        return [msgpack.unpackb(raw) for raw in self.snapshot()]

    def messages_of(self, type_name: str) -> list[dict[str, Any]]:
        return [m for m in self.decoded() if isinstance(m, dict) and m.get("type") == type_name]

    def wait_for(
        self, predicate: Callable[[SttLoopbackServer], bool], timeout_s: float = 5.0
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate(self):
                return True
            time.sleep(0.01)
        return predicate(self)

    # --- server side ----------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        async with websockets.serve(self._handler, self.host, 0) as server:  # type: ignore[attr-defined]
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.01)

    async def _handler(self, conn: Any) -> None:
        with self._lock:
            self.connections += 1
            request = getattr(conn, "request", None)
            self.request_paths.append(str(getattr(request, "path", getattr(conn, "path", ""))))
            self.peers.append(conn.remote_address)
        pump = asyncio.create_task(self._pump(conn))
        try:
            async for raw in conn:
                data = bytes(raw) if not isinstance(raw, str) else raw.encode()
                with self._lock:
                    self.raw_received.append(data)
                msg = msgpack.unpackb(data)
                if isinstance(msg, dict) and msg.get("type") == "Marker" and self.echo_markers:
                    await conn.send(pack(dict(msg)))
        except Exception:  # a client that goes away is data, not an error
            pass
        finally:
            self.client_gone = True
            pump.cancel()

    async def _pump(self, conn: Any) -> None:
        while True:
            try:
                raw = self._outbox.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue
            await conn.send(raw)


# --- the client under test: constructed through its seam, never imported blind -------------------


def build_stt_client(services: Any = None, **kwargs: Any) -> Any:
    """``MoshiSttClient(services=None, *, connect=…, auth_id=…)``, or an assertion naming T-0007a.

    The module is an importable stub today, so the test that calls this fails with an
    ``AssertionError`` (which ``pending`` tolerates) instead of a ``TypeError`` at construction.
    A ``ValueError`` — the loopback check — is deliberately *not* caught: criterion 1 asserts it.
    """
    from local_astra.conductor.stt_client import MoshiSttClient

    try:
        client = MoshiSttClient(services, **kwargs)
    except (TypeError, AttributeError) as exc:  # the stub still takes ``url: str``
        raise AssertionError(
            "T-0007a: MoshiSttClient(services=None, *, connect=…, auth_id='public_token') "
            f"is not implemented ({type(exc).__name__}: {exc})"
        ) from exc
    missing = [name for name in STT_CLIENT_SURFACE if not hasattr(client, name)]
    assert not missing, f"T-0007a: MoshiSttClient is missing {missing} (docs/tasks/T-0007a.md)"
    return client


class EventDrain:
    """One ``events()`` iterator per client, read with deadlines so a hang is an assertion.

    The pending ``__anext__`` is a **task that survives a deadline** (T-0038). Awaiting
    ``wait_for(self._it.__anext__(), t)`` directly cancels the generator when the window expires,
    which closes it: every later read then reports "the stream ended" and every assertion after a
    ``quiet()`` that timed out is vacuously true. Keeping the task and shielding it means a window
    that expires costs nothing but the wait, and ``quiet()`` may be called as often as a test
    likes — including before a ``take()``.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self._it = client.events().__aiter__()
        self._pending: asyncio.Task[Any] | None = None
        self.ended = False
        self.seen: list[Any] = []

    async def _next(self, timeout_s: float) -> Any:
        """The next event within ``timeout_s``; ``TimeoutError`` or ``StopAsyncIteration``."""
        if self._pending is None:
            self._pending = asyncio.ensure_future(self._it.__anext__())
        event = await asyncio.wait_for(asyncio.shield(self._pending), timeout_s)
        self._pending = None
        self.seen.append(event)
        return event

    async def take(self, n: int, timeout_s: float = 2.0) -> list[Any]:
        """The next ``n`` events, or an assertion saying how many actually arrived."""
        out: list[Any] = []
        for _ in range(n):
            try:
                out.append(await self._next(timeout_s))
            except TimeoutError as exc:
                raise AssertionError(
                    f"expected {n} events, got {len(out)} within {timeout_s}s: {out}"
                ) from exc
            except StopAsyncIteration as exc:
                self.ended = True
                raise AssertionError(
                    f"events() ended after {len(out)} of {n} expected events: {out}"
                ) from exc
        return out

    async def quiet(self, window_s: float = 0.2) -> list[Any]:
        """Everything the client offers until it stays silent for ``window_s``."""
        out: list[Any] = []
        while True:
            try:
                out.append(await self._next(window_s))
            except TimeoutError:
                return out
            except StopAsyncIteration:
                self.ended = True
                return out

    async def expect_end(self, timeout_s: float = 1.0) -> None:
        """Assert the stream terminates (only ``close()`` may end it — criterion 6)."""
        try:
            event = await self._next(timeout_s)
        except TimeoutError as exc:
            raise AssertionError("events() did not end after close()") from exc
        except StopAsyncIteration:
            self.ended = True
            return
        raise AssertionError(f"events() yielded {event!r} instead of ending")


def types_of(events: Sequence[Any]) -> list[str]:
    return [str(getattr(e, "type", type(e).__name__)) for e in events]


def f32_blocks(*counts: int, value: float = 0.25) -> Iterator[bytes]:
    """f32le blocks of the given sample counts, all samples ``value`` (exact in binary32)."""
    import struct

    for n in counts:
        yield struct.pack(f"<{n}f", *([value] * n))


async def wait_until(
    predicate: Callable[[], bool], *, timeout_s: float = 2.0, what: str = "condition"
) -> None:
    """Poll ``predicate`` until it holds. A timeout is an assertion, never a hang."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    assert predicate(), f"{what} did not hold within {timeout_s}s"


def live_tasks(baseline: set[asyncio.Task[Any]]) -> set[asyncio.Task[Any]]:
    """Tasks that exist beyond ``baseline`` and the caller's own task."""
    return {t for t in asyncio.all_tasks() if t not in baseline and t is not asyncio.current_task()}


async def settle(rounds: int = 20, step_s: float = 0.01) -> None:
    for _ in range(rounds):
        await asyncio.sleep(step_s)
