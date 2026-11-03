"""Rig for T-0028: a ``Session`` over the fakes, a typed HUD publisher, and the endpoint's seams.

Not a test module (the leading underscore keeps pytest from collecting it). It is the T-0028
copy of ``tests/unit/test_session_loop.py``'s rig — copied, never imported, because
``tests/unit/test_session_*.py`` belongs to T-0007 and criterion 14 leans on those files being
green *unmodified*. What is new here is the publisher seam and the ``hud_server`` symbols.

==================================================================================================
THE SEAMS THIS TASK ADDS. The implementer owns the design *behind* them; these are the names the
tests call. Report anything wrong here to the reviewer; the implementer never edits ``tests/``.
==================================================================================================

``local_astra/conductor/session.py`` — narrow mandate (docs/tasks/T-0028.md, "Owned files")

    HudPublisher = Callable[[AstraMessage], None]
        The typed egress seam that replaces the scaffold's ``publish_hud``/``hud_out`` outbox
        (docs/02 §4 spec delta (b)). **Synchronous and non-blocking**: it is called from the pump
        and from ``CommandExecutor`` handlers, which may not block the loop for more than 20 ms
        (``.claude/rules/conductor.md``). It is never awaited.

    Session(machine, llm, stt, tts, audio, perception, memory, *, clock, config,
            publish: HudPublisher | None = None)
        ``publish=None`` is the default *no-I/O* publisher: it drops the message and touches
        nothing (criterion 14 — "the session's egress is one injected publisher with a default
        that touches no I/O").

    session.set_publisher(publish: HudPublisher | None) -> None
        Late binding for the server, which needs the session to exist first. ``None`` restores
        the no-I/O default. Idempotent.

    Everything else T-0007 fixed in ``test_session_loop.py`` is unchanged: ``start``/``aclose``,
    ``submit``, ``settle``, ``state``, ``assists``, ``executor``, ``turn_written``.

``local_astra/conductor/hud_server.py`` — new

    HudServer(session, *, frame_sink, services: config.Services | None = None,
              host: str = "127.0.0.1", port: int | None = None, path: str = "/v1/realtime",
              outbox_size: int = DEFAULT_OUTBOX)
        ``services`` defaults to ``config.Services()``; ``port=None`` means
        ``services.conductor_port``. Tests pass ``port=0`` to take a free loopback port.
        ``host`` outside {"127.0.0.1", "localhost", "::1"} raises ``ValueError`` **in the
        constructor**, before any socket exists (criterion 1). Construction subscribes the
        session: it calls ``session.set_publisher(server.publish)``.
        ``frame_sink`` is anything with ``ingest(AstraFrame) -> Frame | None`` — in production
        the very ``BrowserFrameSource`` that is also the session's ``PerceptionClient``
        (criterion 5); the server does no validation of its own.

    await server.start() -> None        bind and serve; sets ``server.port``
    await server.aclose() -> None       stop serving and drop every connection
    server.sockets -> tuple[socket.socket, ...]      the listening sockets (criterion 1)
    server.url -> str                   ``ws://<host>:<port><path>``
    server.publish(msg: AstraMessage) -> None
        The ``HudPublisher`` the session calls. Fans the message out to every connection's
        bounded outbox and returns immediately; it never awaits the wire (criterion 12).
    server.counts -> Mapping[str, int]  ingress bookkeeping, keys below (criterion 3, 4, 6)
    HudServer.origin_allowed(origin: str | None) -> bool
        The pure predicate behind criterion 2's 403: True for ``None``, ``"null"``, a ``file://``
        origin and any loopback origin; False for everything else.

    conn = server.open() -> HudConnection
        A connection with **no socket behind it**, so ingress and egress are unit-testable. The
        socket handler is this object plus a reader and a writer task and nothing else.
    conn.receive(raw: str | bytes) -> None
        One inbound frame, exactly as it came off the wire. Total: never raises on client input.
    conn.sent() -> list[AstraMessage]
        What this connection would put on the wire, in order, oldest first, not cleared. A
        message dropped by the outbox is not in it.
    conn.close() -> None
        The client went away. ``events.ClientDisconnected`` is submitted when the **last**
        connection closes, never before and never on a reconnect (criterion 13).
    conn.outbox -> Outbox
    conn.close_reason -> str | None     set when the server closed the connection itself

    Outbox(maxsize: int)                the bounded per-connection queue (criterion 12)
      .offer(msg: AstraMessage) -> bool
          Never blocks. True: queued — possibly after dropping a *superseded*
          ``astra.turn.state`` (latest-wins, idempotent). False: the outbox overflowed with a
          kind that may not be dropped, and the caller must close that connection.
      .drain() -> list[AstraMessage]    what the writer task would send next, in order
      .dropped -> int                   turn.state messages dropped
    DEFAULT_OUTBOX: int                 generous enough that an ordinary turn never drops

    ``server.counts`` keys (all present from the start, zero until they happen):
      ``routed``            client→server messages handed to the session/assists/frame sink
      ``wrong_direction``   a SERVER_TO_CLIENT type arriving from a client
      ``unhandled``         a known ``astra.*`` type with no ingress handler
      ``relay_ignored``     a non-``astra.`` ``RealtimeEvent`` (v1 relays nothing)
      ``malformed``         ``codec.Malformed`` — bad JSON, bad fields, over the byte limit
      ``no_hello``          a message before a valid hello
      ``protocol_version``  a hello with the wrong ``protocol_version``
      ``assist_dropped``    an ``astra.frame.text`` ``AssistState`` refused (fail-closed)
      ``frames_rejected``   an ``astra.frame`` the frame sink refused

``local_astra/cli.py`` — new verbs (criteria 15-19)

    cli.main(argv, *, runner=None, probe=None) -> int
        ``runner(cmd: Sequence[str]) -> int`` runs one ``services/*.sh`` invocation;
        ``probe(name: str, url: str) -> str | None`` health-probes one endpoint and returns
        ``None`` when healthy or the failing reason. Both default to the real implementations;
        both are injected by the fast suite, which starts nothing and probes nothing.
    cli.start_plan(profile: Profile) -> tuple[tuple[str, ...], ...]
        The ordered ``services/*.sh`` invocations that profile's backends need. Pure.
    cli.build_session(config: Config, args: argparse.Namespace) -> SessionGraph
        The production object graph. ``SessionGraph`` carries ``.session``, ``.server``,
        ``.frames`` (the ``BrowserFrameSource`` used as both ``PerceptionClient`` and frame sink)
        and ``.describe() -> str``. ``--dry-run`` prints ``describe()`` and never calls this.
    cli.run_session(graph: SessionGraph) -> int
        Start the session and the server, print the page URL and the bound port, serve until
        interrupted, then ``aclose`` both and return 0 (criterion 16).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.commands import Command
from local_astra.contracts.llm import Commit, LlmOptions, Prompt
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocol import AstraFrame
from local_astra.contracts.turn import MachineState, TurnState
from local_astra.memory.captioner import CAPTION_PROMPT
from local_astra.telemetry.turns_log import TurnRecord
from tests.conftest import VirtualClock
from tests.fakes import FakeAudioIO, FakeLlm, FakeMemory, FakePerception, FakeStt, FakeTts

REPO = Path(__file__).resolve().parents[2]

POLICY = TurnPolicy(thinking=True)

CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"

#: A valid A2UI v0.9.1 canvas (the shape ``tests/unit/test_ui_gate.py`` and T-0007's
#: ``test_session_record.py`` use): one surface, one root, one child. Valid on purpose — the
#: gate must produce a surface, so a missing ``astra.ui.surface`` is the session's fault.
UI_CANVAS = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"lesson_1","catalogId":"'
    + CATALOG_ID
    + '"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"lesson_1","components":['
    '{"id":"root","component":"Column","children":["a"]},'
    '{"id":"a","component":"Text","text":"one half"}]}}'
)

# --- guarded imports: a missing T-0028 symbol is an assertion, never a collection error ---------

_MISSING: dict[str, str] = {}

try:
    from local_astra.conductor.session import Session, SessionConfig
except ImportError as exc:  # pragma: no cover - T-0007 is merged, so this does not happen
    Session = SessionConfig = None  # type: ignore[assignment,misc]
    _MISSING["session"] = f"local_astra.conductor.session: {exc}"

try:
    from local_astra.conductor import hud_server as hud_server_module
except ImportError as exc:  # pragma: no cover - the pre-implementation path
    hud_server_module = None  # type: ignore[assignment]
    _MISSING["hud_server"] = f"local_astra.conductor.hud_server: {exc}"

T0028_SYMBOLS = (Session, SessionConfig, hud_server_module)


def require_hud_server() -> Any:
    """Return the ``hud_server`` module, or fail naming what T-0028 has not written yet."""
    assert not _MISSING, (
        "T-0028 is not implemented: "
        + "; ".join(sorted(_MISSING.values()))
        + " (docs/tasks/T-0028.md, Owned files)"
    )
    return hud_server_module


def require_publisher_seam() -> None:
    """The session's typed publisher seam, asserted *before* anything calls it.

    Without this guard the missing keyword would surface as a ``TypeError`` from
    ``Session(...)``, which is neither of the two outcomes ``tests/conftest.py`` tolerates for a
    pending test — the suite would go red instead of xfail. The seam itself is docs/02 §4 spec
    delta (b) and this module's docstring.
    """
    assert "session" not in _MISSING, _MISSING["session"]
    parameters = inspect.signature(Session.__init__).parameters
    assert "publish" in parameters, (
        "T-0028: Session(..., *, clock, config, publish: HudPublisher | None = None) — the typed "
        "HUD publisher replaces the scaffold's publish_hud/hud_out outbox (docs/02 §4 spec delta "
        f"(b)); Session.__init__ takes {sorted(parameters)}"
    )
    assert callable(getattr(Session, "set_publisher", None)), (
        "T-0028: Session.set_publisher(publish) is how HudServer subscribes after construction "
        "(tests/unit/_hud_rig.py's docstring)"
    )


def require_session_log_seam() -> None:
    """T-0043 criterion 4: the session takes its debug log by injection, like its publisher.

    Asserted *before* ``Session(...)`` is called, so a missing keyword is an ``AssertionError``
    (which ``tests/conftest.py`` tolerates for a pending test) rather than a ``TypeError`` from
    the constructor (which would turn the whole module red instead of xfail).
    """
    assert "session" not in _MISSING, _MISSING["session"]
    parameters = inspect.signature(Session.__init__).parameters
    assert "session_log" in parameters, (
        "T-0043: Session(..., *, clock, config, publish=None, session_log: SessionLog | None = "
        "None) — the log is injected and defaults to None (not logging); Session.__init__ takes "
        f"{sorted(parameters)} (docs/tasks/T-0043.md criteria 4 and 14)"
    )


def require_symbol(name: str) -> Any:
    """One public name of ``conductor/hud_server.py`` (see this module's docstring)."""
    module = require_hud_server()
    value = getattr(module, name, None)
    assert value is not None, (
        f"T-0028: local_astra/conductor/hud_server.py must define {name!r} "
        "(tests/unit/_hud_rig.py's docstring fixes the seam)"
    )
    return value


# --- the scripted clock ------------------------------------------------------------------------


@dataclass
class _Wake:
    at_ms: int
    callback: Callable[[], None]
    seq: int
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass
class ScriptedClock(VirtualClock):
    """``VirtualClock`` plus the one scheduling seam the session may use (T-0007 criterion 3)."""

    scheduled: list[_Wake] = field(default_factory=list)
    seq: int = 0

    def call_at(self, at_ms: int, callback: Callable[[], None]) -> _Wake:
        self.seq += 1
        wake = _Wake(at_ms=int(at_ms), callback=callback, seq=self.seq)
        self.scheduled.append(wake)
        return wake

    def pending(self) -> list[_Wake]:
        return [w for w in self.scheduled if not w.cancelled]

    def advance(self, ms: int) -> int:
        if ms < 0:
            raise ValueError("time does not go backwards")
        target = self.now_ms + ms
        for _ in range(10_000):
            due = [w for w in self.pending() if w.at_ms <= target]
            if not due:
                break
            wake = min(due, key=lambda w: (w.at_ms, w.seq))
            self.scheduled.remove(wake)
            self.now_ms = max(self.now_ms, wake.at_ms)
            self._log.append(self.now_ms)
            wake.callback()
        else:  # pragma: no cover - a runaway re-arm
            raise AssertionError("the session re-armed its tick 10000 times without advancing")
        self.now_ms = target
        self._log.append(target)
        return self.now_ms


# --- the publisher recorders --------------------------------------------------------------------


class Recorder:
    """A ``HudPublisher`` that keeps every message. Synchronous, as the seam requires."""

    def __init__(self) -> None:
        self.messages: list[Any] = []
        #: how deep the pump was when each message was published (see ``Reentrancy`` below)
        self.calls = 0

    def __call__(self, message: Any) -> None:
        self.calls += 1
        self.messages.append(message)

    def of(self, kind: type) -> list[Any]:
        return [m for m in self.messages if isinstance(m, kind)]

    def types(self) -> list[str]:
        return [getattr(m, "type", type(m).__name__) for m in self.messages]

    def wire(self) -> list[str]:
        """Every message as the real codec would put it on the wire."""
        return [encode(m) for m in self.messages]


#: T-0180 (B9, B-R5): the one message type whose ``text`` may carry committed thought text (R-1).
THOUGHT_TYPE = "astra.thought"


@dataclass(frozen=True)
class ThoughtSplit:
    """T-0180 B9/B-R5: what a narrowed leak row searches. ``thoughts`` holds ``(step, kind, end, text)`` of every
    message whose ``type`` is exactly ``astra.thought``, in publish order; ``elsewhere`` is the JSON of every other
    message plus every ``astra.thought`` with its ``text`` field removed (the sentinel may be in no other field)."""

    thoughts: tuple[tuple[Any, Any, Any, str], ...]
    elsewhere: str

    def texts(self, *, kind: str | None = None) -> list[str]:
        return [t for _, k, _, t in self.thoughts if kind is None or k == kind]


def split_thoughts(messages: Sequence[Any]) -> ThoughtSplit:
    """Split published messages for a B9-shaped leak row (see ``ThoughtSplit``)."""
    thoughts: list[tuple[Any, Any, Any, str]] = []
    rest: list[Any] = []
    for m in messages:
        d = m.model_dump(mode="json") if hasattr(m, "model_dump") else m
        if isinstance(d, dict) and d.get("type") == THOUGHT_TYPE:
            thoughts.append((d.get("step"), d.get("kind"), d.get("end"), str(d.get("text", ""))))
            rest.append({k: v for k, v in d.items() if k != "text"})
        else:
            rest.append(d)
    return ThoughtSplit(tuple(thoughts), json.dumps(rest))


class StalledReader:
    """A publisher standing in for a client that stopped reading (criterion 12, adversarial).

    Everything published is queued and **nothing is ever drained** — the writer task is stuck on
    a socket whose peer went quiet. The publisher itself still returns immediately, which is the
    contract; a session that instead waits for the wire cannot use this object at all.
    """

    def __init__(self) -> None:
        self.queued: list[Any] = []
        self.drained = 0

    def __call__(self, message: Any) -> None:
        self.queued.append(message)


# --- recording fakes ------------------------------------------------------------------------


class RecordingLlm(FakeLlm):
    """``FakeLlm`` with a call log and a hold on the commit stream (copied from T-0007's rig)."""

    def __init__(self, script: Sequence[Commit] = (), *, prefix_tokens: int = 1000) -> None:
        super().__init__(script, prefix_tokens=prefix_tokens)
        self.clock: ScriptedClock | None = None
        self.calls: list[tuple[str, str, int]] = []
        self.gen_prompts: list[tuple[str, Prompt]] = []
        self._flow = asyncio.Event()
        self._flow.set()

    def hold(self) -> None:
        self._flow.clear()

    def release(self) -> None:
        self._flow.set()

    def _note(self, what: str, request_id: str) -> None:
        self.calls.append((what, request_id, self.clock.now_ms if self.clock else 0))

    async def prefill(self, prefix: Prompt, request_id: str) -> Any:
        self._note("prefill", request_id)
        return await super().prefill(prefix, request_id)

    async def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
    ) -> Any:
        caption = CAPTION_PROMPT[:40] in " ".join(m.text() for m in prompt.messages)
        if not caption:
            self.gen_prompts.append((request_id, prompt))
            self._note("generate", request_id)
        async for commit in super().generate(prompt, request_id, options, xargs=xargs):
            await self._flow.wait()
            if request_id not in self.active:
                return
            yield commit

    async def abort(self, request_id: str) -> None:
        self._note("abort", request_id)
        await super().abort(request_id)

    def transcripts(self) -> list[str]:
        """The user transcript of every turn generate, in order (the ``[ui] …`` assertion)."""
        return [prompt.messages[-1].text() for _, prompt in self.gen_prompts]


class RecordingTts(FakeTts):
    def __init__(self, ms_per_word: int = 300) -> None:
        super().__init__(ms_per_word=ms_per_word)
        self.requests: list[tuple[str, str, str]] = []
        self._flow = asyncio.Event()
        self._flow.set()

    def hold(self) -> None:
        self._flow.clear()

    def release(self) -> None:
        self._flow.set()

    async def speak(self, utterance_id: str, text: str, voice: str = "default") -> Any:
        self.requests.append((utterance_id, text, voice))
        await self._flow.wait()
        return await super().speak(utterance_id, text, voice)

    def texts(self) -> list[str]:
        return [text for _, text, _ in self.requests]


class RecordingPerception(FakePerception):
    def __init__(self, frames: Sequence[Frame] = ()) -> None:
        super().__init__(frames)
        self.frame_calls: list[tuple[Source, int | None]] = []

    async def get_frame(self, source: Source, at_ms: int | None = None) -> Frame | None:
        self.frame_calls.append((source, at_ms))
        return await super().get_frame(source, at_ms)


class RecordingAudio(FakeAudioIO):
    def __init__(self, clock: VirtualClock) -> None:
        super().__init__(clock)
        self.plays: list[tuple[Any, str | None]] = []
        self.pauses: list[int] = []
        self.calls: list[str] = []

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        self.calls.append("play")
        self.plays.append((utterance, source))
        await super().play(utterance, source=source)

    async def pause(self) -> None:
        self.calls.append("pause")
        self.pauses.append(self.clock.now_ms)
        await super().pause()

    async def resume(self) -> None:
        self.calls.append("resume")
        await super().resume()

    async def flush(self) -> int:
        self.calls.append("flush")
        return await super().flush()

    def mark_drain(self) -> None:
        self.calls.append("mark_drain")
        super().mark_drain()


class RecordingMachine(TurnMachine):
    """The production machine, with a log of what the pump fed it and what came back."""

    def __init__(self, policy: TurnPolicy | None = None) -> None:
        super().__init__(policy or POLICY)
        self.log: list[tuple[E.Event, TurnState, tuple[Command, ...]]] = []

    def step(self, state: MachineState, event: E.Event) -> Any:
        new, cmds = super().step(state, event)
        self.log.append((event, new.state, tuple(cmds)))
        return new, cmds

    def events(self, kind: type | None = None) -> list[E.Event]:
        return [ev for ev, _, _ in self.log if kind is None or isinstance(ev, kind)]

    def commands(self, kind: type | None = None) -> list[Command]:
        out: list[Command] = []
        for _, _, cmds in self.log:
            out.extend(c for c in cmds if kind is None or isinstance(c, kind))
        return out


class PushStt(FakeStt):
    """``SttClient`` whose events are pushed by the test, one at a time, from the outside.

    ``FakeStt`` replays a script the moment ``events()`` is iterated, which races the reflex
    events a turn needs first. This one holds the stream open, so a test can deliver an
    ``SttEvent`` at a chosen moment through the *client* — the path
    ``Session._pump_stt`` adapts into ``contracts/events`` (ruling A15: that adapter had no
    coverage because every rig submitted the already-adapted event).
    """

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def push(self, event: Any) -> None:
        self._queue.put_nowait(event)

    async def events(self) -> typing.AsyncIterator[Any]:
        while not self.closed:
            yield await self._queue.get()


# --- script helpers ------------------------------------------------------------------------


def frame(ts_ms: int, tag: str = "a", source: Source = Source.SCREEN) -> Frame:
    return Frame(
        source=source,
        ts_ms=ts_ms,
        width=1280,
        height=720,
        jpeg=b"\xff\xd8" + tag.encode() * 8,
        phash=f"{tag * 4}{ts_ms:012d}",
        title=f"window {tag}",
        app="firefox",
    )


def answer(text: str, canvas_idx: int = 0) -> Commit:
    return Commit(request_id="?", canvas_idx=canvas_idx, segment="answer", text=text)


def thought(text: str, canvas_idx: int = 0, *, open_: bool = False) -> Commit:
    return Commit(
        request_id="?", canvas_idx=canvas_idx, segment="thought", text=text, thought_open=open_
    )


def ui(canvas_idx: int = 1, text: str = UI_CANVAS) -> Commit:
    return Commit(request_id="?", canvas_idx=canvas_idx, segment="ui", text=text)


def astra_frame(tag: str = "a", *, source: Source = Source.SCREEN, ts_ms: int = 0) -> AstraFrame:
    """An ``astra.frame`` the *real* ``BrowserFrameSource`` accepts (T-0008 criteria 14-18).

    ``jpeg_b64`` is base64 of a JPEG-magic payload and ``phash`` is 16 lowercase hex characters;
    anything else the source rejects, which would make criterion 5's assertion vacuous.
    """
    import base64

    jpeg = b"\xff\xd8\xff\xe0" + tag.encode() * 16
    return AstraFrame(
        source=source,
        ts_ms=ts_ms,
        width=1280,
        height=720,
        jpeg_b64=base64.b64encode(jpeg).decode(),
        phash=(tag.encode().hex() * 16)[:16],
        title=f"tab {tag}",
        app="chromium",
    )


# --- the rig -------------------------------------------------------------------------------


@dataclass
class Rig:
    clock: ScriptedClock
    machine: RecordingMachine
    llm: RecordingLlm
    stt: FakeStt
    tts: RecordingTts
    audio: RecordingAudio
    perception: Any
    memory: FakeMemory
    session: Any
    turns_path: Path
    published: Any = None
    #: T-0043: the ``SessionLog`` this rig injected, or ``None`` when the rig is not logging
    session_log: Any = None

    # -- driving ------------------------------------------------------------------------------
    def submit(self, event: E.Event) -> None:
        self.session.submit(event)

    async def pump(self, rounds: int = 4) -> None:
        for _ in range(rounds):
            self.clock.advance(0)
            await asyncio.sleep(0)
            await self.session.settle()

    async def advance(self, ms: int, step: int = 50) -> None:
        left = ms
        while left > 0:
            self.clock.advance(min(step, left))
            left -= min(step, left)
            await self.pump(rounds=2)
        await self.pump()

    async def until(
        self, predicate: Callable[[], bool], what: str, *, limit_ms: int = 30_000, step: int = 50
    ) -> None:
        waited = 0
        await self.pump()
        while not predicate():
            if waited >= limit_ms:
                raise AssertionError(
                    f"{what}: still not true after {limit_ms} ms of virtual time "
                    f"(state={self.state.state}, turn={self.state.turn_id})"
                )
            await self.advance(step, step=step)
            waited += step

    # -- state --------------------------------------------------------------------------------
    @property
    def state(self) -> MachineState:
        return self.session.state

    def records(self) -> list[TurnRecord]:
        return [TurnRecord.from_line(line) for line in self.lines()]

    def lines(self) -> list[str]:
        if not self.turns_path.exists():
            return []
        return [line for line in self.turns_path.read_text().splitlines() if line.strip()]

    # -- scripted turn pieces -------------------------------------------------------------------
    async def onset(self) -> int:
        at = self.clock.now_ms
        self.submit(E.UserSpeechOnset(t_ms=at, onset_ms=at))
        await self.pump()
        return at

    async def words(self, text: str, ms_per_word: int = 200) -> None:
        for word in text.split():
            await self.advance(ms_per_word, step=ms_per_word)
            now = self.clock.now_ms
            self.submit(E.SttWord(t_ms=now, text=word, start_ms=now - ms_per_word, end_ms=now))
        await self.pump()

    async def end_of_turn(self) -> None:
        await self.advance(100, step=100)
        now = self.clock.now_ms
        self.submit(E.SttEndOfTurn(t_ms=now, t_eot_ms=now - 20, p_end=0.9))
        self.submit(E.SttFlushed(t_ms=now, t_eot_ms=now - 20))
        await self.pump()
        await self.advance(POLICY.eot_settle_ms + 40, step=40)

    async def ask(self, text: str = "what is on this screen") -> None:
        await self.onset()
        await self.words(text)
        await self.end_of_turn()

    async def speak_out(self, limit_ms: int = 30_000) -> None:
        await self.until(
            lambda: self.state.state is TurnState.IDLE,
            "the turn never returned to IDLE",
            limit_ms=limit_ms,
        )


def make_rig(
    tmp_path: Path,
    *,
    script: Sequence[Commit] = (),
    frames: Sequence[Frame] = (),
    ms_per_word: int = 300,
    publish: Any = None,
    perception: Any = None,
    stt: Any = None,
    config_kwargs: dict[str, Any] | None = None,
    clock: ScriptedClock | None = None,
    session_log: Any = None,
) -> Rig:
    require_publisher_seam()
    # T-0043: the keyword is only passed when a test asked for a log, so every rig written before
    # this task builds exactly the ``Session`` it built before (criterion 24: logging is
    # observation only — including of the object graph).
    log_kwargs: dict[str, Any] = {}
    if session_log is not None:
        require_session_log_seam()
        log_kwargs["session_log"] = session_log
    clock = clock or ScriptedClock()
    machine = RecordingMachine(POLICY)
    llm = RecordingLlm(script)
    llm.clock = clock
    stt = stt if stt is not None else FakeStt()
    tts = RecordingTts(ms_per_word=ms_per_word)
    audio = RecordingAudio(clock)
    perception = (
        perception
        if perception is not None
        else RecordingPerception(list(frames) or [frame(0, "a")])
    )
    memory = FakeMemory()
    turns_path = tmp_path / "turns.jsonl"
    config = SessionConfig(data_dir=tmp_path, turns_path=turns_path, **(config_kwargs or {}))
    session = Session(
        machine,
        llm,
        stt,
        tts,
        audio,
        perception,
        memory,
        clock=clock,
        config=config,
        publish=publish,
        **log_kwargs,
    )
    return Rig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=stt,
        tts=tts,
        audio=audio,
        perception=perception,
        memory=memory,
        session=session,
        turns_path=turns_path,
        published=publish,
        session_log=session_log,
    )


@asynccontextmanager
async def running(tmp_path: Path, **kwargs: Any) -> typing.AsyncIterator[Rig]:
    rig = make_rig(tmp_path, **kwargs)
    await rig.session.start()
    try:
        yield rig
    finally:
        await rig.session.aclose()


# --- the contract-test half: a real loopback socket ----------------------------------------


@dataclass
class Served:
    """A rig whose session is published through a real ``HudServer`` on ``127.0.0.1:0``."""

    rig: Rig
    server: Any
    frames: Any

    @property
    def url(self) -> str:
        return str(self.server.url)


@asynccontextmanager
async def served(
    tmp_path: Path,
    *,
    browser_source: bool = True,
    outbox_size: int | None = None,
    **kwargs: Any,
) -> typing.AsyncIterator[Served]:
    """Rig + ``HudServer`` bound on a free loopback port, started, and torn down.

    ``outbox_size`` forces the per-connection bound small, which is how a test makes a real
    socket overflow without generating megabytes (criterion 12 / ruling A12).

    ``browser_source=True`` wires the production shape of criterion 15: one
    ``BrowserFrameSource`` is both the session's ``PerceptionClient`` and the server's frame
    sink, so a frame posted by the client is the frame the turn snapshots.
    """
    hud_server = require_hud_server()
    clock = kwargs.pop("clock", None) or ScriptedClock()
    sink: Any = None
    if browser_source:
        from local_astra.perception.browser_source import BrowserFrameSource

        sink = BrowserFrameSource(now_ms=lambda: clock.now_ms)
        kwargs.setdefault("perception", sink)
    rig = make_rig(tmp_path, clock=clock, **kwargs)
    if sink is None:
        sink = _NullSink()
    server_kwargs: dict[str, Any] = {}
    if outbox_size is not None:
        server_kwargs["outbox_size"] = outbox_size
    server = hud_server.HudServer(rig.session, frame_sink=sink, port=0, **server_kwargs)
    await rig.session.start()
    await server.start()
    try:
        yield Served(rig=rig, server=server, frames=sink)
    finally:
        await server.aclose()
        await rig.session.aclose()


class _NullSink:
    """A frame sink that accepts nothing; used when a test does not post frames."""

    def __init__(self) -> None:
        self.ingested: list[AstraFrame] = []

    def ingest(self, message: AstraFrame) -> None:
        self.ingested.append(message)
        return None


class HudClient:
    """A real WebSocket client for the contract tests: real socket, real codec, 2 s budget."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.received: list[Any] = []

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, url: str, *, origin: str | None = None, open_timeout: float = 2.0
    ) -> typing.AsyncIterator[HudClient]:
        from websockets.asyncio.client import connect as ws_connect

        async with ws_connect(url, origin=origin, open_timeout=open_timeout) as ws:  # type: ignore[arg-type]
            yield cls(ws)

    async def send(self, message: Any) -> None:
        """A ``Contract`` goes through the real codec; a ``str``/``bytes`` goes out unchanged."""
        await self.ws.send(message if isinstance(message, str | bytes) else encode(message))

    async def hello(self, protocol_version: int | None = None) -> None:
        from local_astra.contracts.protocol import PROTOCOL_VERSION, AstraHello

        version = PROTOCOL_VERSION if protocol_version is None else protocol_version
        await self.send(AstraHello(protocol_version=version))

    async def drain(
        self,
        *,
        until: Callable[[list[Any]], bool] | None = None,
        rig: Rig | None = None,
        budget_ms: int = 5_000,
        step_ms: int = 50,
        wall_ceiling_s: float = 20.0,
    ) -> list[Any]:
        """Read messages (decoded by the *production* codec) until ``until`` or the budget.

        **The budget is work, not wall time** (ruling A13). The session's clock is the
        ``ScriptedClock``: it only moves when this loop advances it, so a budget measured with
        ``loop.time()`` buys *less virtual progress on a loaded machine* — 20 ms of real time per
        50 ms of virtual time means a busy box runs out of seconds before the session has run out
        of ticks, and ``until`` is still false. That is a flake, not a failure. So:

        * ``budget_ms`` — with a ``rig``, the virtual milliseconds this call may advance the
          session's clock (``budget_ms // step_ms`` rounds of "read what is there, then tick");
          without one, the same number of rounds of pure reading. Deterministic under load.
        * ``wall_ceiling_s`` — a real-time backstop so a dead socket cannot wedge the suite. It is
          *not* the budget: a test that hits it has found a hang, and the message says so.

        Each round reads whatever has already arrived (20 ms is how long a loopback delivery gets
        to show up) and then lets the session advance, which is what produces the next messages.
        """
        from local_astra.contracts.codec import decode

        loop = asyncio.get_running_loop()
        rounds = max(1, budget_ms // max(1, step_ms))
        ceiling = loop.time() + wall_ceiling_s
        advanced_ms = 0
        for _ in range(rounds):
            if until is not None and until(self.received):
                return self.received
            if loop.time() > ceiling:
                raise AssertionError(
                    f"drain hit its {wall_ceiling_s} s real-time backstop after {advanced_ms} ms "
                    "of virtual time — something is hanging, which is not what the budget is for; "
                    f"got {[getattr(m, 'type', m) for m in self.received]}"
                )
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=0.02)
            except TimeoutError:
                if rig is not None:
                    await rig.advance(step_ms, step=step_ms)
                    advanced_ms += step_ms
                continue
            self.received.append(decode(raw))
        if until is not None and not until(self.received):
            raise AssertionError(
                f"the expected messages never arrived within {budget_ms} ms of virtual time "
                f"({rounds} rounds); got {[getattr(m, 'type', m) for m in self.received]}"
            )
        return self.received

    def of(self, kind: type) -> list[Any]:
        return [m for m in self.received if isinstance(m, kind)]
