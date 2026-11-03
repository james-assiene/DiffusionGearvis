"""T-0128c group A: a fake vLLM tap, a delegating spy over pyzmq, and the relay's seams.

Not a test module. Three things live here because both ``tests/unit/test_canvas_relay.py`` and
``tests/contract/test_canvas_relay_wire.py`` need them:

* ``FakeTap`` -- a real pyzmq PUB that publishes with the plugin's **own** framing: it is a thin
  owner of ``astra_vllm_plugin.tap.StepPublisher`` (loaded off the source tree by
  ``load_plugin_module``), so the topic and the msgpack packing are the engine's, not a copy.
  Its real-service twin is the ``real`` side of the contract module (``gpu``).
* ``SpyZmq`` -- a stand-in for the ``zmq`` module that **delegates every name it does not fake**
  (``.claude/rules/tests.md``): ``Context`` and ``asyncio.Context`` are real pyzmq subclasses that
  record every context and socket the relay makes, refuse ``bind`` and ``Context.instance()`` with a
  named assertion (the relay is a SUB that connects; it owns its context), and record socket options.
* frame builders and ``relay_module()`` -- the module under test is imported lazily, and a missing
  module is an ``AssertionError`` naming T-0128c (the precedent is ``tests/fakes/diffusion_sampler.py``
  ``load_module``), so a pending row xfails instead of erroring at collection.

The recorded trajectory (``tests/fixtures/canvas/trajectory_806da12.msgpack``) is a real engine
recording; see that directory's MANIFEST for its regime. The hand-built frames below are
constructions and are never presented as a trajectory.
"""

from __future__ import annotations

import hashlib
import importlib
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack

from local_astra.contracts.protocol import AstraTurnState
from local_astra.contracts.turn import TurnState
from tests.fakes.diffusion_sampler import load_plugin_module

TASK = "T-0128c"
RELAY_MODULE = "local_astra.conductor.canvas_relay"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "canvas" / "trajectory_806da12.msgpack"
#: sha256 of the fixture, from its MANIFEST (computed on copy, 2026-10-26).
FIXTURE_SHA256 = "d837ef36755db7151af53d7b7830b96cefa598c85f686c2445ba9b50c424d914"
TOPIC = b"astra.canvas.step"


def relay_module() -> Any:
    """``local_astra.conductor.canvas_relay``, or an assertion naming what T-0128c has not written."""
    try:
        return importlib.import_module(RELAY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != RELAY_MODULE:
            raise
        raise AssertionError(
            f"{TASK} group A: {RELAY_MODULE} does not exist yet (docs/tasks/{TASK}.md, Owned files)"
        ) from None


def relay_attr(name: str) -> Any:
    module = relay_module()
    value = getattr(module, name, None)
    assert value is not None, (
        f"{TASK}: {RELAY_MODULE} must define {name!r} (the seam is fixed in "
        "tests/unit/test_canvas_relay.py's docstring)"
    )
    return value


# --------------------------------------------------------------------------- the trajectory


def load_trajectory() -> dict[str, Any]:
    """``{"regime": str, "request": dict, "frames": [dict, ...]}`` exactly as recorded."""
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, (
        f"{FIXTURE.name} is not the recording its MANIFEST describes (sha256 changed)"
    )
    data = msgpack.unpackb(raw, raw=False)
    assert isinstance(data, dict) and isinstance(data.get("frames"), list), "fixture shape"
    return data


# --------------------------------------------------------------------------- hand-built frames


def denoise(
    rid: str,
    canvas_idx: int,
    step: int,
    *,
    canvas: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    argmax: Sequence[int] | None = None,
    stable: str = "00",
    max_steps: int = 48,
) -> dict[str, Any]:
    """A valid R10 denoise frame, CL = len(canvas). ``argmax`` defaults to ``canvas`` shifted by the
    step, so two frames of one canvas at different steps are different dicts."""
    rows = list(canvas)
    arg = list(argmax) if argmax is not None else [t + 100 * step for t in rows]
    return {
        "type": "astra.canvas.step",
        "request_id": rid,
        "canvas_idx": canvas_idx,
        "step": step,
        "max_steps": max_steps,
        "canvas_length": len(rows),
        "phase": "denoise",
        "canvas": rows,
        "argmax": arg,
        "stable": stable,
    }


def commit(
    rid: str,
    canvas_idx: int,
    step: int,
    *,
    tokens: Sequence[int] = (11, 12, 13),
    canvas_length: int = 8,
    max_steps: int = 48,
) -> dict[str, Any]:
    """A valid R10 commit frame: committed ids, none of the rows."""
    return {
        "type": "astra.canvas.step",
        "request_id": rid,
        "canvas_idx": canvas_idx,
        "step": step,
        "max_steps": max_steps,
        "canvas_length": canvas_length,
        "phase": "commit",
        "committed_tokens": list(tokens),
    }


def pack(frame: Mapping[str, Any]) -> bytes:
    """The payload part exactly as ``StepPublisher.send`` packs it."""
    return bytes(msgpack.packb(dict(frame), use_bin_type=True))


#: Payloads the relay must count as invalid and never publish, each with why (by hand, from
#: ``AstraCanvasStep`` and R10).
INVALID_PAYLOADS: dict[str, bytes] = {
    "not msgpack": b"\xc1",
    "msgpack list": bytes(msgpack.packb([1, 2, 3])),
    "denoise missing stable": pack(
        {k: v for k, v in denoise("bad", 0, 3).items() if k != "stable"}
    ),
    "commit with rows": pack({**commit("bad", 0, 3), "canvas": [1] * 8}),
    "commit with no ids": pack({**commit("bad", 0, 3), "committed_tokens": []}),
    "stable too short": pack({**denoise("bad", 0, 3, canvas=list(range(16))), "stable": "ff"}),
    "denoise past the cap": pack(denoise("bad", 0, 49)),
    "a synthetic canvas": pack(
        {"type": "astra.canvas.synthetic", "request_id": "bad", "reason": "x"}
    ),
    "canvas length disagrees": pack({**denoise("bad", 0, 3), "canvas_length": 9}),
}


# --------------------------------------------------------------------------- state and publish


@dataclass
class StateBox:
    """The relay's injected ``state``: a zero-argument callable returning ``AstraTurnState``,
    as ``Session.turn_state_message`` does. Tests move it by assignment."""

    state: TurnState = TurnState.IDLE
    turn_id: str = "t0"

    def __call__(self) -> AstraTurnState:
        return AstraTurnState(state=self.state, turn_id=self.turn_id)

    def go(self, state: TurnState, turn_id: str | None = None) -> None:
        self.state = state
        if turn_id is not None:
            self.turn_id = turn_id


@dataclass
class Published:
    """The relay's injected ``publish``; stamps each message with the clock it was published at."""

    clock: Any = None
    messages: list[Any] = field(default_factory=list)
    at_ms: list[int] = field(default_factory=list)
    #: optional: called at each publish; its value is kept in ``marks`` (the unit rig passes the
    #: number of frames received so far, so "newest received before this publish" is exact even
    #: when a frame and a publish share one virtual millisecond)
    mark: Any = None
    marks: list[int] = field(default_factory=list)

    def __call__(self, message: Any) -> None:
        self.messages.append(message)
        self.at_ms.append(int(self.clock.now_ms) if self.clock is not None else -1)
        if self.mark is not None:
            self.marks.append(int(self.mark()))

    def canvas(self) -> list[Any]:
        return [m for m in self.messages if str(getattr(m, "type", "")).startswith("astra.canvas.")]

    def errors(self, code: str | None = None) -> list[Any]:
        return [
            m
            for m in self.messages
            if getattr(m, "type", "") == "astra.error" and (code is None or m.code == code)
        ]


@dataclass
class LogSpy:
    """A duck-typed ``SessionLog``: ``record(kind, data, *, turn_id=None, session_ms=None)``."""

    records: list[tuple[str, Any, dict[str, Any]]] = field(default_factory=list)

    def record(self, kind: str, data: Any, **kwargs: Any) -> None:
        self.records.append((kind, data, kwargs))

    def notes(self) -> list[Any]:
        return [data for kind, data, _ in self.records if kind == "note"]


# --------------------------------------------------------------------------- SpyZmq


@dataclass
class Ledger:
    contexts: list[Any] = field(default_factory=list)
    sockets: list[Any] = field(default_factory=list)
    connects: list[tuple[Any, str]] = field(default_factory=list)
    options: list[tuple[Any, int, Any]] = field(default_factory=list)
    #: (context, [socket.closed for every socket of that context]) at each term()
    terms: list[tuple[Any, list[bool]]] = field(default_factory=list)


class SpyZmq:
    """The ``zmq`` module, with the context classes spied. Every other name is the real one."""

    def __init__(self) -> None:
        import zmq
        import zmq.asyncio

        self._zmq = zmq
        self.ledger = Ledger()
        ledger = self.ledger

        def spy_socket(base: type) -> type:
            class SpySocket(base):  # type: ignore[misc, valid-type]
                def bind(self, addr: str) -> Any:
                    raise AssertionError(
                        f"{TASK} A1: the relay's SUB connects and never binds (bind({addr!r}))"
                    )

                def bind_to_random_port(self, *a: Any, **k: Any) -> Any:
                    raise AssertionError(f"{TASK} A1: the relay's SUB never binds")

                def connect(self, addr: str) -> Any:
                    ledger.connects.append((self, addr))
                    return super().connect(addr)

                def set(self, option: int, value: Any) -> Any:
                    ledger.options.append((self, int(option), value))
                    return super().set(option, value)

                def setsockopt(self, option: int, value: Any) -> Any:
                    ledger.options.append((self, int(option), value))
                    return super().setsockopt(option, value)

            return SpySocket

        def spy_context(base: type, socket_base: type) -> type:
            sock_cls = spy_socket(socket_base)

            class SpyContext(base):  # type: ignore[misc, valid-type]
                _socket_class = sock_cls

                @classmethod
                def instance(cls, *a: Any, **k: Any) -> Any:
                    raise AssertionError(
                        f"{TASK} implementation notes: the relay owns a zmq.Context; never "
                        "Context.instance() (the process-wide one is shared with anybody)"
                    )

                def __init__(self, *a: Any, **k: Any) -> None:
                    super().__init__(*a, **k)
                    ledger.contexts.append(self)

                def socket(self, *a: Any, **k: Any) -> Any:
                    sock = super().socket(*a, **k)
                    ledger.sockets.append(sock)
                    return sock

                def term(self) -> Any:
                    mine = [s for s in ledger.sockets if getattr(s, "context", None) is self]
                    ledger.terms.append((self, [bool(s.closed) for s in mine]))
                    still_open = [s for s in mine if not s.closed]
                    if still_open:
                        # the real term() would block forever on an open socket: close them so the
                        # test does not hang, and fail naming the defect (O-A2 M20)
                        for s in still_open:
                            s.close(linger=0)
                        super().term()
                        raise AssertionError(
                            f"{TASK}: term() called with {len(still_open)} socket(s) of its "
                            "context still open (close every socket before term())"
                        )
                    return super().term()

            return SpyContext

        self.Context = spy_context(zmq.Context, zmq.Socket)

        class _AsyncNamespace:
            def __init__(inner) -> None:
                inner.Context = spy_context(zmq.asyncio.Context, zmq.asyncio.Socket)

            def __getattr__(inner, name: str) -> Any:
                return getattr(zmq.asyncio, name)

        self.asyncio = _AsyncNamespace()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._zmq, name)

    # convenience
    def options_of(self, sock: Any) -> dict[int, Any]:
        return {opt: value for s, opt, value in self.ledger.options if s is sock}


# --------------------------------------------------------------------------- FakeTap


class FakeTap:
    """The engine's tap, with the plugin's own publisher: topic + msgpack, PUB, binds the endpoint.

    ``StepPublisher`` binds lazily on its first send (T-0069 c8 -- finding 6 of the task); the
    fake binds on ``bind()`` so a test can let the subscription propagate before the first frame.
    """

    def __init__(self, endpoint: str, *, hwm: int = 64) -> None:
        tapmod = load_plugin_module("tap", "8")
        self.endpoint = endpoint
        self.publisher = tapmod.StepPublisher(endpoint, hwm=hwm)
        self.sent: list[dict[str, Any]] = []

    def bind(self) -> FakeTap:
        self.publisher._socket()  # the plugin's own lazy bind, taken eagerly (see the docstring)
        return self

    def send(self, frame: Mapping[str, Any]) -> bool:
        ok = bool(self.publisher.send(frame))
        if ok:
            self.sent.append(dict(frame))
        return ok

    def send_raw(self, payload: bytes) -> None:
        """A payload the plugin would never send, under the plugin's topic (invalid-frame rows)."""
        self.publisher._socket().send_multipart([TOPIC, payload])

    def replay(self, frames: Iterable[Mapping[str, Any]], *, gap_s: float = 0.0) -> int:
        n = 0
        for frame in frames:
            n += int(self.send(frame))
            if gap_s:
                time.sleep(gap_s)
        return n

    def close(self) -> None:
        self.publisher.close()


# --------------------------------------------------------------------------- QueueZmq (O-A2)


class QueueSocket:
    """A SUB stand-in whose ``recv_multipart`` awaits a queue the test fills: the relay's own
    reader and timer then run on a virtual clock (``ScriptedClock``), with no socket and no wall
    time. Only what a SUB reader needs is faked; any other name is a named assertion (never a bare
    ``AttributeError`` inside the relay)."""

    def __init__(self, kind: int) -> None:
        import asyncio

        self.type = kind
        self.options: list[tuple[int, Any]] = []
        self.connects: list[str] = []
        self.closed = False
        self.recv_calls = 0
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    # options and wiring
    def setsockopt(self, option: int, value: Any) -> None:
        self.options.append((int(option), value))

    set = setsockopt

    def getsockopt(self, option: int) -> Any:
        values = [v for o, v in self.options if o == int(option)]
        assert values, f"{TASK}: QueueSocket option {option} was never set"
        return values[-1]

    get = getsockopt

    def connect(self, addr: str) -> None:
        self.connects.append(addr)

    def bind(self, addr: str) -> None:
        raise AssertionError(f"{TASK} A1: the relay's SUB connects and never binds ({addr!r})")

    def close(self, linger: Any = None) -> None:
        self.closed = True

    # the reader's one call
    async def recv_multipart(self, *args: Any, **kwargs: Any) -> list[bytes]:
        import asyncio

        self.recv_calls += 1
        item = await self._queue.get()
        if isinstance(item, BaseException):
            await asyncio.sleep(0)
            raise item
        return list(item)

    # the test's side
    def push(self, parts: Sequence[bytes]) -> None:
        self._queue.put_nowait(list(parts))

    def push_frame(self, frame: Mapping[str, Any]) -> None:
        self.push([TOPIC, pack(frame)])

    def push_error(self, exc: BaseException) -> None:
        self._queue.put_nowait(exc)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"{TASK}: QueueSocket does not fake {name!r} (extend tests/fakes)")


class QueueZmq:
    """The ``zmq`` module with ``asyncio.Context`` replaced by one that makes ``QueueSocket``s.
    Every other module-level name is the real pyzmq one (constants, errors)."""

    def __init__(self) -> None:
        import zmq

        self._zmq = zmq
        self.sockets: list[QueueSocket] = []
        self.contexts: list[Any] = []
        outer = self

        class Context:
            def __init__(inner, *a: Any, **k: Any) -> None:
                inner.termed = False
                inner.sockets: list[QueueSocket] = []
                outer.contexts.append(inner)

            @classmethod
            def instance(cls, *a: Any, **k: Any) -> Any:
                raise AssertionError(f"{TASK}: the relay owns its context; never instance()")

            def socket(inner, kind: int, *a: Any, **k: Any) -> QueueSocket:
                sock = QueueSocket(kind)
                inner.sockets.append(sock)
                outer.sockets.append(sock)
                return sock

            def term(inner) -> None:
                assert all(s.closed for s in inner.sockets), (
                    f"{TASK}: term() with a socket of its context still open"
                )
                inner.termed = True

        class _Asyncio:
            def __getattr__(inner, name: str) -> Any:
                if name == "Context":
                    return Context
                import zmq.asyncio

                return getattr(zmq.asyncio, name)

        self.asyncio = _Asyncio()

    def __getattr__(self, name: str) -> Any:
        if name == "Context":
            raise AssertionError(
                f"{TASK}: QueueZmq fakes zmq.asyncio.Context only (the relay reads on the loop)"
            )
        return getattr(self._zmq, name)

    def sub(self) -> QueueSocket:
        subs = [s for s in self.sockets if s.type == self._zmq.SUB]
        assert len(subs) == 1, f"{TASK}: the relay made {len(subs)} SUB sockets"
        return subs[0]


# --------------------------------------------------------------------------- T-0159 group A: pieces
#
# docs/tasks/T-0159.md A3/A4/A5: the relay adds ``argmax_pieces`` (denoise) / ``committed_pieces``
# (commit) to every frame it publishes, from a *piece source*. The fake tap itself sends no pieces.
#
# THE SEAM (fixed here; the A implementer builds behind it, anything wrong goes to the reviewer):
#
#     PieceSource: anything with ``pieces(ids: Sequence[int]) -> tuple[str, ...]`` (one string per
#         id; raises ``local_astra.toolcall.detok.UnknownTokenId`` for an id it cannot decode).
#         ``HostDetokenizer`` is one (A2).
#
#     CanvasRelay(..., piece_source: PieceSource | None = None,
#                 piece_loader: Callable[[], PieceSource] | None = None)
#         ``piece_source``: already loaded (tests); used from construction on, no load in ``start``.
#         ``piece_loader``: called once off the event loop (``asyncio.to_thread``) when
#         ``piece_source`` is ``None``; ``None`` means ``HostDetokenizer.load`` (production,
#         ``cli.py`` unchanged). Ruling O-A2: ``start()`` returns at once; the reader awaits the
#         load before its first receive, so no frame is read before the source is ready or has
#         failed; ``aclose()`` cancels a pending load (a cancellation is never a failed load). If
#         it raises an ``Exception``: A5 (one stderr line, one note, one
#         ``astra.error(code="denoise_no_pieces")``); frames then go out without pieces and are
#         not counted ``undecoded``.
#         A relay given neither and never started publishes no pieces.
#
#     counts["undecoded"]: frames published without pieces because ``pieces`` raised (A3); also in
#         the per-turn ``{"denoise": {...}}`` note.

TASK_PIECES = "T-0159"
#: The two T-0159 A1 fields. Every T-0128c "unchanged" row compares frames with these removed.
PIECE_FIELDS = ("argmax_pieces", "committed_pieces")
#: A5: the error code and the stderr prefix (by hand from the criterion).
NO_PIECES_CODE = "denoise_no_pieces"
PIECES_OFF_PREFIX = "astra session: denoising text off: "


def stub_piece(token_id: int) -> str:
    """The stub source's rule (the task's ``id -> f"#{id}"``), the oracle every stub row uses."""
    return f"#{token_id}"


class StubPieces:
    """A counting piece source: ``#<id>`` per id; ids in ``undecodable`` raise ``UnknownTokenId``
    (the real class, so the relay's ``except`` is the production one). ``calls`` keeps every id
    tuple handed to ``pieces``, in order -- the oracle of "a coalesced frame is never decoded"."""

    def __init__(self, undecodable: Iterable[int] = (), *, rule: Any = None) -> None:
        self.undecodable = frozenset(int(i) for i in undecodable)
        self.rule = rule if rule is not None else stub_piece
        self.calls: list[tuple[int, ...]] = []

    def pieces(self, ids: Iterable[int]) -> tuple[str, ...]:
        got = tuple(int(i) for i in ids)
        self.calls.append(got)
        bad = [i for i in got if i in self.undecodable]
        if bad:
            from local_astra.toolcall.detok import UnknownTokenId

            raise UnknownTokenId(f"t0159 stub: token id {bad[0]} is not in the stub vocabulary")
        return tuple(self.rule(i) for i in got)


class GatedLoader:
    """A ``piece_loader`` that blocks (in the thread ``start`` runs it in) until ``release()``.
    ``fail`` makes it raise that exception once released instead of returning ``source``."""

    def __init__(self, source: Any = None, *, fail: BaseException | None = None) -> None:
        import threading

        self.source = source if source is not None else StubPieces()
        self.fail = fail
        self.calls = 0
        self.thread_names: list[str] = []
        self._gate = threading.Event()
        self.done = threading.Event()

    def release(self) -> None:
        self._gate.set()

    def __call__(self) -> Any:
        import threading

        self.calls += 1
        self.thread_names.append(threading.current_thread().name)
        try:
            assert self._gate.wait(30.0), "t0159 GatedLoader: never released (a hung test)"
            if self.fail is not None:
                raise self.fail
            return self.source
        finally:
            self.done.set()


class SlowLoader:
    """A ``piece_loader`` that takes ``seconds`` of wall time in a pure-Python busy loop (it holds
    and drops the GIL like ordinary Python code, as a parser written in Python would)."""

    def __init__(self, seconds: float, source: Any = None) -> None:
        self.seconds = seconds
        self.source = source if source is not None else StubPieces()
        self.done = False

    def __call__(self) -> Any:
        end = time.perf_counter() + self.seconds
        n = 0
        while time.perf_counter() < end:
            n += 1
        self.done = True
        return self.source


def failing_loader(reason: str) -> Any:
    """A ``piece_loader`` that raises ``FileNotFoundError(reason)`` (the pinned file is absent)."""

    def load() -> Any:
        raise FileNotFoundError(reason)

    return load


def piece_kwargs(cls: Any, **kwargs: Any) -> dict[str, Any]:
    """``kwargs`` when ``cls`` (the relay) declares the T-0159 seam's keywords; else an assertion
    naming T-0159 -- so a pending row fails as an assertion, never a ``TypeError``."""
    import inspect

    params = inspect.signature(cls).parameters
    missing = sorted(k for k in kwargs if k not in params)
    assert not missing, (
        f"{TASK_PIECES} A4: CanvasRelay has no keyword(s) {missing} (the seam is fixed in "
        "tests/fakes/canvas_tap.py, T-0159 section)"
    )
    return kwargs


def optional_piece_kwargs(cls: Any, source: Any) -> dict[str, Any]:
    """For the T-0128c rigs that must stay green on the BASE relay: ``{"piece_source": source}``
    when the relay declares it, else nothing (BASE publishes no pieces, which the amended
    "unchanged except the T-0159 fields" rows accept). Keeps the real tokenizer out of the rigs."""
    import inspect

    return {"piece_source": source} if "piece_source" in inspect.signature(cls).parameters else {}


def without_pieces(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in PIECE_FIELDS}


def pieces_of(message: Any) -> tuple[Any, Any]:
    """``(argmax_pieces, committed_pieces)`` of a published message (``None`` when absent, so a
    BASE message without the attributes reads as "no pieces")."""
    if isinstance(message, Mapping):
        return message.get("argmax_pieces"), message.get("committed_pieces")
    return getattr(message, "argmax_pieces", None), getattr(message, "committed_pieces", None)


def assert_relayed(message: Any, frame: Mapping[str, Any], rule: Any = stub_piece) -> None:
    """T-0128c A2 as amended by T-0159 A3: ``message`` equals the validated ``frame`` field for
    field except the T-0159 fields, and those, when present, are ``rule`` over the frame's own ids
    (argmax on a denoise frame, committed ids on a commit frame), never the other phase's field."""
    from local_astra.contracts.protocol import AstraCanvasStep

    want = AstraCanvasStep.model_validate(dict(frame)).model_dump()
    got = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    assert without_pieces(got) == without_pieces(want), (
        f"A2 (amended by T-0159 A3): {frame.get('request_id')}/{frame.get('canvas_idx')}/"
        f"{frame.get('step')} changed on the way through outside the piece fields"
    )
    argmax_p, committed_p = pieces_of(message)
    if frame.get("phase", "denoise") == "commit":
        assert argmax_p is None, (
            f"T-0159 A1: a commit frame never carries argmax_pieces: {argmax_p}"
        )
        if committed_p is not None:
            assert tuple(committed_p) == tuple(rule(i) for i in frame["committed_tokens"]), (
                f"T-0159 A3: committed_pieces are not the pieces of the frame's own ids: {committed_p}"
            )
    else:
        assert committed_p is None, "T-0159 A1: a denoise frame never carries committed_pieces"
        if argmax_p is not None:
            assert tuple(argmax_p) == tuple(rule(i) for i in frame["argmax"]), (
                f"T-0159 A3: argmax_pieces are not the pieces of the frame's own argmax: {argmax_p}"
            )
