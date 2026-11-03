"""The conductor's relay of the vLLM plugin's per-step tap to the HUD (T-0128c group A).

The plugin (``plugin/astra_vllm_plugin/astra_vllm_plugin/tap.py``, T-0011) publishes every
denoising step on a ZeroMQ PUB as two parts, ``[b"astra.canvas.step", msgpack(dict)]``, at most
30 Hz per request. This module is the other end: a SUB that **connects** (never binds) to a
loopback endpoint, validates each payload as ``AstraCanvasStep`` and hands it, unchanged, to the
HUD publisher. It creates no frame of its own: no request id, no step, no synthetic canvas.

* **Admission by turn state (A4).** A request whose first frame arrives while the session is
  IDLE is the idle captioner and is refused for its lifetime; one first seen in any other state is
  admitted for its lifetime (its final commit can land after the turn went IDLE). Both tables are
  LRU, capacity ``TABLE_CAPACITY``.
* **Coalescing (A3).** Denoise frames are latest-wins per request, published at most
  ``PER_REQUEST_HZ`` per request and ``AGGREGATE_HZ`` in total (denoise only, ruling O-A) on the
  session clock. Commits always pass, at the next flush; a commit discards the request's pending
  denoise frame, so no denoise frame of a closed canvas is published after its commit.
* **Never blocks (A5).** ``receive`` and ``flush`` are synchronous and O(requests); the socket is
  read by a ``zmq.asyncio`` task on the loop, so ``publish`` is only ever called on the loop
  thread. The relay owns its ``zmq.Context`` (never ``Context.instance()``), sets ``LINGER 0`` and
  closes its socket before ``term()``.
* **Says what happened (A7, A8).** One session-log ``note`` per closed turn; one
  ``astra.error(code="denoise_no_tap")`` per session when a turn closes and no tap payload has
  arrived since the relay was built.
* **Pieces (T-0159 A3-A5).** Each published frame gains ``argmax_pieces`` (denoise) or
  ``committed_pieces`` (commit), decoded from the frame's own ids at publish time (``_emit``), so a
  coalesced frame is never decoded. The piece source is injected, or the reader task awaits
  ``HostDetokenizer.load()`` through ``asyncio.to_thread`` before its first ``recv`` (ruling O-A2:
  ``start`` returns at once; the load's GIL-bound stall at session start is accepted; the 20 ms
  bound is on frame handling). Ids that do not decode publish the frame without pieces, counted
  ``undecoded``. A source that cannot load is said once on stderr, once in the session log and
  once as ``astra.error(code="denoise_no_pieces")``; the relay runs on without pieces.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import msgpack
from pydantic import ValidationError

from local_astra.contracts.protocol import AstraCanvasStep, AstraError, AstraTurnState
from local_astra.contracts.turn import TurnState
from local_astra.toolcall.detok import UnknownTokenId

log = logging.getLogger(__name__)

#: = ``services/versions.lock`` ``[plugin] tap_endpoint``.
DEFAULT_ENDPOINT = "tcp://127.0.0.1:8002"
ENV_ENDPOINT = "ASTRA_TAP_ENDPOINT"
#: The plugin's topic frame (``tap.StepPublisher``).
TOPIC = b"astra.canvas.step"

#: A3: denoise frames per request in any 1000 ms of the session clock.
PER_REQUEST_HZ = 30
#: A3 (ruling O-A): denoise frames across all requests in any 1000 ms; commits are not counted.
AGGREGATE_HZ = 60
#: Minimum gaps that make the two caps hold in every half-open 1000 ms window: ``n`` publishes
#: ``gap`` apart span ``(n - 1) * gap`` ms, so ``ceil(1000 / gap) <= cap``.
PER_REQUEST_GAP_MS = -(-1000 // PER_REQUEST_HZ)  # 34 ms -> at most 30 in a window
AGGREGATE_GAP_MS = -(-1000 // AGGREGATE_HZ)  # 17 ms -> at most 59 in a window
#: A4: both admission tables (ruling O-A: exactly 64).
TABLE_CAPACITY = 64
#: A5: the SUB's receive high-water mark (<= 256).
RECEIVE_HWM = 256
#: A payload larger than this is refused by ZeroMQ itself; a CL 256 frame is about 4 KB.
MAX_MESSAGE_BYTES = 1 << 20
#: How often ``start``'s timer reads the turn state when nothing is pending.
STATE_POLL_MS = 50
#: O-A3 (2): at most one summary of survived failures per this many ms after the first.
FAULT_LOG_INTERVAL_MS = 5000

NO_TAP_CODE = "denoise_no_tap"
NO_TAP_MESSAGE = (
    "a turn closed and no denoising frame has arrived from the engine's tap; the Thinking pane "
    "has no real canvases to show (is the vLLM plugin loaded and its tap on?)"
)

_TCP = re.compile(r"tcp://127\.0\.0\.1:([0-9]{1,5})")
_IPC = re.compile(r"ipc://[^\s]+")
_COUNTERS = ("received", "published", "coalesced", "refused_idle", "invalid", "undecoded")

NO_PIECES_CODE = "denoise_no_pieces"
NO_PIECES_MESSAGE = (
    "the denoising view shows no token text: the conductor could not load the pinned tokenizer "
    "({reason}); the canvases are still the real ones"
)


class PieceSource(Protocol):
    """What the relay decodes ids with (``HostDetokenizer`` is one): one string per id."""

    def pieces(self, ids: Iterable[int]) -> tuple[str, ...]: ...


def _load_host_detokenizer() -> PieceSource:
    """The default ``piece_loader``: the pinned ``HostDetokenizer.load()`` (looked up at call time)."""
    from local_astra.toolcall.detok import HostDetokenizer

    return HostDetokenizer.load()


def check_endpoint(endpoint: str) -> str:
    """The endpoint itself when it is ``tcp://127.0.0.1:<port>`` or ``ipc://<path>`` (A1).

    Anything else raises ``ValueError``: a wildcard, a host name (``localhost`` included, ruling
    O-A), IPv6, a routable address, another transport, or trailing text. ``fullmatch``, so a
    trailing newline is refused too.
    """
    if not isinstance(endpoint, str):
        raise ValueError(f"tap endpoint must be a string, not {type(endpoint).__name__}")
    tcp = _TCP.fullmatch(endpoint)
    if tcp is not None and 1 <= int(tcp.group(1)) <= 65535:
        return endpoint
    if _IPC.fullmatch(endpoint) is not None:
        return endpoint
    raise ValueError(
        f"tap endpoint {endpoint!r} refused: only tcp://127.0.0.1:<port> or ipc://<path> "
        "(the relay is loopback only)"
    )


def endpoint_from_env(environ: Mapping[str, str] | None = None) -> str:
    """``environ[ENV_ENDPOINT]``, checked, or ``DEFAULT_ENDPOINT`` when unset."""
    env = os.environ if environ is None else environ
    value = env.get(ENV_ENDPOINT)
    return DEFAULT_ENDPOINT if value is None else check_endpoint(value)


class _Clock(Protocol):
    @property
    def now_ms(self) -> int: ...

    def call_at(self, at_ms: int, callback: Callable[[], None]) -> Any: ...


class _Log(Protocol):
    def record(self, kind: str, data: Any, *, turn_id: str | None = ...) -> None: ...


@dataclass
class _Request:
    """What the relay holds for one admitted request."""

    #: commits received and not yet published, oldest first
    commits: list[AstraCanvasStep]
    #: the newest denoise frame not yet published
    pending: AstraCanvasStep | None = None
    #: session ms of the last denoise publish (``None``: never)
    last_ms: int | None = None


class CanvasRelay:
    """The tap's subscriber: validate, admit, coalesce, publish (see the module docstring)."""

    def __init__(
        self,
        *,
        publish: Callable[[Any], None],
        clock: _Clock,
        state: Callable[[], AstraTurnState],
        session_log: _Log | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        zmq_module: Any = None,
        piece_source: PieceSource | None = None,
        piece_loader: Callable[[], PieceSource] | None = None,
    ) -> None:
        """``piece_source``: a ready piece source (T-0159, tests); otherwise ``start`` runs
        ``piece_loader`` (default: ``HostDetokenizer.load``) in a worker thread."""
        self._endpoint = check_endpoint(endpoint)  # A1: before anything else exists
        self._publish = publish
        self._pieces: PieceSource | None = piece_source
        self._load_pieces = _load_host_detokenizer if piece_loader is None else piece_loader
        #: T-0159 A5: why the piece source could not load (``None``: it did, or has not yet)
        self._pieces_off: str | None = None
        self._no_pieces_said = False
        self._clock = clock
        self._state = state
        self._log = session_log
        self._zmq = zmq_module
        self._counts: dict[str, int] = dict.fromkeys(_COUNTERS, 0)
        #: per-turn window (from the last close to the next), for the A7 note
        self._turn_counts: dict[str, int] = dict.fromkeys(_COUNTERS, 0)
        self._turn_requests: dict[str, None] = {}
        self._admitted: OrderedDict[str, _Request] = OrderedDict()
        self._refused: OrderedDict[str, None] = OrderedDict()
        self._last_denoise_ms: int | None = None
        self._was_open = False
        self._open_turn_id: str | None = None
        self._no_tap_said = False
        # sockets: made in ``start``, never here
        self._context: Any = None
        self._socket: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._timer: Any = None
        #: session ms the armed timer fires at (``None``: not armed)
        self._timer_at: int | None = None
        self._closed = False
        #: O-A2 (3): set once the relay stopped on its own, so it is said once
        self._stopped: str | None = None
        #: O-A3 (2): failures survived, and how many of them have been reported
        self._faults = 0
        self._faults_logged = 0
        self._fault_logged_ms = 0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def counts(self) -> Mapping[str, int]:
        return dict(self._counts)

    # --- the pure half --------------------------------------------------------------------

    def _count(self, name: str) -> None:
        self._counts[name] += 1
        self._turn_counts[name] += 1

    def receive(self, payload: bytes) -> None:
        """One payload part exactly as it came off the SUB. Never raises."""
        self._count("received")
        try:
            data = msgpack.unpackb(payload, raw=False)
            frame = AstraCanvasStep.model_validate(data)
        except Exception:  # a decode boundary: anything the wire carries is counted, not raised
            self._count("invalid")
            return
        rid = frame.request_id
        if rid in self._refused:
            self._refused.move_to_end(rid)
            self._count("refused_idle")
            return
        request = self._admitted.get(rid)
        if request is None:
            if self._state().state is TurnState.IDLE:
                self._refused[rid] = None
                if len(self._refused) > TABLE_CAPACITY:
                    self._refused.popitem(last=False)
                self._count("refused_idle")
                return
            request = _Request(commits=[])
            self._admitted[rid] = request
            if len(self._admitted) > TABLE_CAPACITY:
                self._admitted.popitem(last=False)
        else:
            self._admitted.move_to_end(rid)
        self._turn_requests[rid] = None
        if frame.phase == "commit":
            # A3: a commit closes its canvas; the pending denoise frame (received before it) is
            # never published after it.
            if request.pending is not None:
                request.pending = None
                self._count("coalesced")
            request.commits.append(frame)
            return
        if request.pending is not None:
            self._count("coalesced")  # latest-wins
        request.pending = frame

    def flush(self) -> None:
        """Publish what is due at ``clock.now_ms``, then read the turn state (turn close)."""
        now = int(self._clock.now_ms)
        for request in self._admitted.values():
            while request.commits:
                # O-A3 (1): popped before it is published, so a publish that raises mid-list
                # never publishes this commit a second time at the next flush
                self._emit(request.commits.pop(0))
        while True:
            if self._last_denoise_ms is not None and now - self._last_denoise_ms < AGGREGATE_GAP_MS:
                break
            due = [
                r
                for r in self._admitted.values()
                if r.pending is not None
                and (r.last_ms is None or now - r.last_ms >= PER_REQUEST_GAP_MS)
            ]
            if not due:
                break
            # fairness: the request shown least recently goes first
            request = min(due, key=lambda r: -1 if r.last_ms is None else r.last_ms)
            shown = request.pending
            assert shown is not None
            request.pending = None
            request.last_ms = now
            self._last_denoise_ms = now
            self._emit(shown)
        self._read_state()

    def _emit(self, frame: AstraCanvasStep) -> None:
        shown = self._with_pieces(frame)
        self._publish(shown)
        self._count("published")
        if shown is frame and self._pieces_off is not None and not self._no_pieces_said:
            # T-0159 A5: once per session, at the first frame published without pieces after the
            # piece source failed to load
            self._no_pieces_said = True
            self._publish(
                AstraError(
                    code=NO_PIECES_CODE, message=NO_PIECES_MESSAGE.format(reason=self._pieces_off)
                )
            )

    def _with_pieces(self, frame: AstraCanvasStep) -> AstraCanvasStep:
        """T-0159 A3: the frame with the pieces of its own ids; the frame itself when there is no
        piece source (yet) or its ids do not decode (counted ``undecoded``)."""
        source = self._pieces
        if source is None:
            return frame
        if frame.phase == "commit":
            key, ids = "committed_pieces", frame.committed_tokens
        elif frame.argmax is not None:
            key, ids = "argmax_pieces", frame.argmax
        else:
            return frame
        try:
            pieces = tuple(source.pieces(ids))
            # validated again (``model_copy`` would skip ``_lengths_agree``), so a source that
            # returns the wrong count or a non-string never reaches the page
            return AstraCanvasStep.model_validate({**frame.model_dump(), key: pieces})
        except (UnknownTokenId, ValidationError):  # ids that do not decode: never a dropped frame
            self._count("undecoded")
            return frame
        except Exception:  # O-A2: a broken source is a fault, said at a bounded rate
            self._count("undecoded")
            self._fault("a frame's pieces")
            return frame

    def _read_state(self) -> None:
        current = self._state()
        is_open = current.state is not TurnState.IDLE
        if is_open:
            if (
                self._was_open
                and self._open_turn_id is not None
                and current.turn_id != self._open_turn_id
            ):
                # O-A2 (2): a takeover or relisten opens the next turn without passing IDLE
                self._close_turn(self._open_turn_id)
            self._open_turn_id = current.turn_id
        elif self._was_open:
            self._close_turn(self._open_turn_id or current.turn_id)
        self._was_open = is_open

    def _close_turn(self, turn_id: str) -> None:
        if self._log is not None:
            note = {
                "denoise": {
                    "turn_id": turn_id,
                    "request_ids": list(self._turn_requests),
                    **{name: self._turn_counts[name] for name in _COUNTERS},
                }
            }
            self._log.record("note", note, turn_id=turn_id)
        self._turn_counts = dict.fromkeys(_COUNTERS, 0)
        self._turn_requests = {}
        if self._counts["received"] == 0 and not self._no_tap_said:
            self._no_tap_said = True
            self._publish(AstraError(code=NO_TAP_CODE, message=NO_TAP_MESSAGE))

    def _next_due_ms(self) -> int:
        """When the next ``flush`` has something to do: a pending frame's slot, or the state poll."""
        now = int(self._clock.now_ms)
        due = now + STATE_POLL_MS
        for request in self._admitted.values():
            if request.commits:
                return now + 1
            if request.pending is not None:
                at = now if request.last_ms is None else request.last_ms + PER_REQUEST_GAP_MS
                if self._last_denoise_ms is not None:
                    at = max(at, self._last_denoise_ms + AGGREGATE_GAP_MS)
                due = min(due, max(at, now + 1))
        return due

    # --- the socket half ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the SUB, connect it and start reading. Raises (``ImportError``, ``zmq`` errors)
        when the relay cannot start; nothing is left open then.

        T-0159 (A4 as amended by O-A2): returns as soon as the reader task is started, never
        awaiting the piece load (a cancel during ``start`` must not skip the caller's cleanup).
        The reader awaits the load before its first ``recv`` (``_read``)."""
        if self._socket is not None:
            return
        zmq = self._zmq
        if zmq is None:
            import zmq
            import zmq.asyncio
        context = zmq.asyncio.Context()
        try:
            sock = context.socket(zmq.SUB)
            try:
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVHWM, RECEIVE_HWM)
                sock.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
                sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
                sock.connect(self._endpoint)
            except BaseException:
                sock.close(linger=0)
                raise
        except BaseException:
            context.term()
            raise
        self._context = context
        self._socket = sock
        self._closed = False
        self._reader = asyncio.get_running_loop().create_task(
            self._read(sock), name="canvas-relay-reader"
        )
        self._reader.add_done_callback(self._reader_done)
        self._arm()

    async def _load(self) -> None:
        """T-0159 (O-A2): the piece load in a worker thread, awaited by the reader before its
        first ``recv``. ``HostDetokenizer.load`` holds the GIL in ``json.loads`` for ~300 ms: a
        one-off stall at session start, accepted. ``aclose`` cancels it with the reader; only an
        ``Exception`` takes the A5 path, a ``CancelledError`` propagates."""
        if self._pieces is not None or self._pieces_off is not None:
            return
        try:
            self._pieces = await asyncio.to_thread(self._load_pieces)
        except Exception as exc:  # A5: a missing or wrong file turns the text off, never the relay
            self._no_pieces(f"{type(exc).__name__}: {exc}")

    def _no_pieces(self, reason: str) -> None:
        """T-0159 A5: say once on stderr and once in the session log; the page is told at the
        next frame published without pieces (``_emit``)."""
        if self._pieces_off is not None:
            return
        self._pieces_off = reason
        print(f"astra session: denoising text off: {reason}", file=sys.stderr)
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log.record("note", {"denoise_relay": {"pieces_off": reason}})

    async def _read(self, sock: Any) -> None:
        # T-0159 (O-A2): no frame is read before the piece source is ready or has failed; frames
        # queue in the socket meanwhile, up to RCVHWM
        await self._load()
        while True:
            # A receive error (the socket or context went away under us) ends the reader; the
            # done-callback says so. Anything after the receive is per payload: logged, survived.
            parts = await sock.recv_multipart()
            try:
                if len(parts) == 2 and bytes(parts[0]) == TOPIC:
                    self.receive(bytes(parts[1]))
                else:
                    self._count("received")
                    self._count("invalid")
                self.flush()
                self._rearm_if_earlier()
            except Exception:  # O-A2 (3): one bad iteration never ends the relay
                self._fault("a payload's flush")

    def _reader_done(self, task: asyncio.Task[None]) -> None:
        """O-A2 (3): a reader that ended while the relay was open stopped the relay -- say so."""
        if self._closed or task.cancelled():
            return
        exc = task.exception()
        self._stop(f"the tap reader stopped: {type(exc).__name__}: {exc}")

    def _stop(self, reason: str) -> None:
        if self._stopped is not None:
            return
        self._stopped = reason
        print(f"astra session: denoising view off: {reason}", file=sys.stderr)
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log.record("note", {"denoise_relay": {"stopped": reason}})

    def _arm(self) -> None:
        if self._closed:
            return
        at = self._next_due_ms()
        self._timer_at = at
        self._timer = self._clock.call_at(at, self._tick)

    def _cancel_timer(self) -> None:
        timer, self._timer, self._timer_at = self._timer, None, None
        if timer is not None:
            cancel = getattr(timer, "cancel", None)
            if callable(cancel):
                cancel()

    def _rearm_if_earlier(self) -> None:
        """O-A2 (1): a frame read between ticks may be due before the armed tick (a 30 Hz stream
        against a 34 ms slot); move the tick earlier so the cap, not the timer, decides."""
        if self._closed or self._timer is None or self._timer_at is None:
            return
        if self._next_due_ms() < self._timer_at:
            self._cancel_timer()
            self._arm()

    def _tick(self) -> None:
        self._timer = None
        self._timer_at = None
        if self._closed:
            return
        try:
            self.flush()
        except Exception:  # O-A2 (3): the periodic flush survives a bad iteration
            self._fault("a periodic flush")
        self._arm()

    def _fault(self, where: str) -> None:
        """O-A3 (2): call from an ``except`` block. The first failure is logged with its
        traceback; later ones are counted and summarised at most once per
        ``FAULT_LOG_INTERVAL_MS`` of the session clock, so a persistent fault is not a traceback
        per frame on stderr (``astra session`` configures no logging)."""
        self._faults += 1
        now = int(self._clock.now_ms)
        if self._faults == 1:
            log.exception("canvas relay: %s failed; the relay carries on", where)
            self._fault_logged_ms, self._faults_logged = now, 1
            return
        if now - self._fault_logged_ms >= FAULT_LOG_INTERVAL_MS:
            log.error(
                "canvas relay: %d more failures (latest in %s) since the last report; "
                "the relay carries on",
                self._faults - self._faults_logged,
                where,
            )
            self._fault_logged_ms, self._faults_logged = now, self._faults

    async def aclose(self) -> None:
        """Stop reading, close the socket, then terminate the context. Idempotent.

        Only the reader's own cancellation is suppressed: an outer cancellation of the task
        running ``aclose`` is re-raised (O-A2 (4))."""
        self._closed = True
        self._cancel_timer()
        reader, self._reader = self._reader, None
        try:
            if reader is not None:
                if reader.done():
                    if not reader.cancelled():
                        reader.exception()  # already reported by ``_reader_done``
                else:
                    # Count the outer cancellations before the await: a caller that already
                    # swallowed its own cancel (``run_session``'s serve loop) is still
                    # ``cancelling()``, so only a cancel arriving *during* this await is outer.
                    current = asyncio.current_task()
                    before = current.cancelling() if current is not None else 0
                    reader.cancel()
                    try:
                        await reader
                    except asyncio.CancelledError:
                        if current is not None and current.cancelling() > before:
                            raise
        finally:
            self._close_sockets()

    def _close_sockets(self) -> None:
        sock, self._socket = self._socket, None
        if sock is not None:
            sock.close(linger=0)
        context, self._context = self._context, None
        if context is not None:
            context.term()
