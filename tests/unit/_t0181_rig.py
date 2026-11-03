"""Shared rig for T-0181 group A (docs/tasks/T-0181.md, "Tests to write -> Group A"; test-author, A).

Not a test module (the leading underscore keeps pytest from collecting it). Nothing here imports a T-0181 symbol at
module level: the rows reach the mic switch only through the wire (raw JSON into a socketless ``HudServer``
connection), the session's own keyword (checked against ``Session.__init__``'s signature first) and what the session
already exposes, so before group A lands every row is red on a named ``AssertionError``, never on a ``TypeError`` or
an import error at collection.

**The pieces.**

- ``RecordingMachine`` (the production machine, with a log of what the pump fed it and what came back);
- ``ToolLlm`` (T-0092c's tool-capable fake LLM: call log, holds, the ``cd`` keyword), so the same rig drives a tools-off,
  ``dry_run`` or live-kit session;
- ``RecordingStt`` around a *reactive* ``FakeStt`` (nothing comes back that the fed audio has not paid for; with
  ``hears_audio=True`` an all-zero frame is silence, ``tests/fakes/stt.py``): it records every ``feed`` (bytes and
  ``t_ms``) and every ``mark``;
- the audio: ``RecordingAudio`` (``FakeAudioIO``, which *is* a ``CaptureSource``: its ``stt_blocks`` yields 80 ms
  frames of 1920 samples as virtual time passes) for a mic session, or ``DeafAudio`` (exactly the ``AudioIO`` protocol,
  no ``stt_blocks``) for a session with no capture seam;
- a real ``SessionLog`` (read back after ``aclose`` into ``rig.log_records``, because its writer is asynchronous);
- a socketless ``HudServer`` (``port=0``, never started) with one greeted page, ``rig.page``; the session publishes to
  the server and to a ``Recorder`` (``rig.published``).

**The surface the rows assume** (the task's names; the implementer owns everything behind them):

- ``Session(..., mic_unavailable: str | None = <omitted>)`` (A-R1): omitted, availability is derived from whether
  ``audio`` is a ``CaptureSource`` (unavailable reason ``NO_SEAM``); a string means unavailable with that reason;
- ``session.apply_mic_mute(message)`` (A2), reached through ``hud_server.route``;
- ``astra.mic.mute {muted}`` in, ``astra.mic.state {muted, available, reason}`` out (A1);
- the records are ``note`` records with ``what: "mic.mute"`` (``{what, from, to}``) and ``what: "mic.gate"`` (A-R2).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, the ``FakeAudioIO`` capture (10 ms blocks at
48 kHz -> 1920-sample 80 ms STT frames at 24 kHz), n = 1, deterministic per row. No measurement.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import AstraHello, AstraSessionReset
from tests.fakes import FakeMemory
from tests.fakes.stt import FakeStt
from tests.unit import _tools_switch_support as T
from tests.unit._hud_rig import Recorder, _NullSink
from tests.unit._session_log_rig import make_log, read_records
from tests.unit._tool_turn_support import (
    MS_PER_WORD,
    ToolLlm,
    ToolRig,
    WatchingExecutor,
)
from tests.unit._tool_turn_support import (
    frame as tool_frame,
)
from tests.unit.test_session_loop import (
    POLICY,
    RecordingAudio,
    RecordingMachine,
    RecordingPerception,
    RecordingTts,
    ScriptedClock,
    Session,
    SessionConfig,
    require_session,
)

#: A1: the two wire types
MUTE = "astra.mic.mute"
STATE = "astra.mic.state"
#: A2: the error code a mute in a session with no mic answers its sender with
UNAVAILABLE = "mic_unavailable"
#: A-R1: the availability keyword
MIC_KWARG = "mic_unavailable"
#: A3: the reasons, verbatim
NO_SEAM = "this audio path has no capture seam"
REASON_NO_MIC = "this session has no microphone (--no-mic)"
REASON_MIC_NONE = "this session has no microphone (--mic none)"
REASON_SILENT = "this session is silent: no audio device is open (--silent)"
#: A-R2: the two note records
MUTE_NOTE = "mic.mute"
GATE_NOTE = "mic.gate"

#: the BASE hello reply of a fresh page in a tools-off session with no turn yet (hud_server.hello_reply at main
#: 74ae8b25, read off ``_hello_journal``: the assist state, the turn state, the log state; nothing replayed)
BASE_FRESH_HELLO = ["astra.assist.state", "astra.turn.state", "astra.log.state"]

_OMITTED: Any = object()


class RecordingStt:
    """Delegating wrapper over any ``SttClient``: records what was fed and what was marked (as
    ``test_session_listening.RecordingStt``, copied so this rig does not lean on that module)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.fed: list[tuple[bytes, int]] = []
        self.marks: list[int] = []

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        self.fed.append((pcm_24k_f32, t_ms))
        await self.inner.feed(pcm_24k_f32, t_ms)

    async def mark(self, t_eot_ms: int) -> int:
        self.marks.append(int(t_eot_ms))
        result: int = await self.inner.mark(t_eot_ms)
        return result

    def events(self) -> Any:
        return self.inner.events()

    def stats(self) -> Any:
        return self.inner.stats()

    async def close(self) -> None:
        await self.inner.close()

    @property
    def closed(self) -> bool:
        return bool(self.inner.closed)

    # -- what the rows read ----------------------------------------------------------------------
    def t_ms(self) -> list[int]:
        return [t for _, t in self.fed]

    def zero(self) -> list[bool]:
        """Per fed frame: every sample is exactly zero (little-endian f32 +0.0 is four zero bytes; -0.0 is not)."""
        return [not any(pcm) for pcm, _ in self.fed]


def reactive_stt(*, hears_audio: bool = False, **kwargs: Any) -> RecordingStt:
    """A reactive ``FakeStt`` (nothing comes back that the fed audio has not paid for), recorded."""
    return RecordingStt(FakeStt(reactive=True, hears_audio=hears_audio, **kwargs))


class DeafAudio:
    """An audio object with **no capture seam**: exactly the ``AudioIO`` protocol, delegating to a
    ``RecordingAudio`` (``inner``) so a row can still read what was played. No ``__getattr__``: a catch-all would
    give it ``stt_blocks`` and make it a ``CaptureSource``."""

    def __init__(self, clock: Any) -> None:
        self.inner = RecordingAudio(clock)
        self.clock = clock

    @property
    def plays(self) -> list[tuple[Any, str | None]]:
        return self.inner.plays

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        await self.inner.pause()

    async def resume(self) -> None:
        await self.inner.resume()

    async def flush(self) -> int:
        return await self.inner.flush()

    def played_ms(self) -> float:
        return float(self.inner.played_ms())

    def mark_drain(self) -> None:
        self.inner.mark_drain()

    def is_rendering(self, t_ms: float) -> bool:
        return bool(self.inner.is_rendering(t_ms))

    def playback_events(self) -> Any:
        return self.inner.playback_events()

    def capture(self) -> Any:
        return self.inner.capture()


@dataclass
class MicRig(ToolRig):
    """``ToolRig`` plus the HUD side and the recording STT."""

    server: Any = None
    page: Any = None
    rec_stt: Any = None
    pages: list[Any] = field(default_factory=list)

    # -- the wire --------------------------------------------------------------------------------
    def greet(self) -> Any:
        """A new page, greeted; its hello reply is ``page.sent()``."""
        page = self.server.open()
        page.receive(encode(AstraHello()))
        self.pages.append(page)
        return page

    def send_mute(self, muted: Any, page: Any = None) -> None:
        """The page's ``astra.mic.mute``, as raw JSON (so before A1 the codec answers ``Unknown``, never a
        ``ValidationError`` raised in the row)."""
        (page or self.page).receive(json.dumps({"type": MUTE, "muted": muted}))

    async def mute(self, muted: bool, page: Any = None) -> None:
        self.send_mute(muted, page)
        await self.pump()

    async def reset(self, page: Any = None) -> None:
        (page or self.page).receive(encode(AstraSessionReset()))
        await self.pump()
        await self.advance(500)

    async def type_question(self, text: str, page: Any = None) -> None:
        (page or self.page).receive(json.dumps({"type": "astra.text", "text": text}))
        await self.pump()

    # -- what the rows read ----------------------------------------------------------------------
    def notes(self, what: str) -> list[dict[str, Any]]:
        """Every ``note`` record's data with ``what`` (after ``aclose``)."""
        return [d for d in self.kind("note") if d.get("what") == what]

    # -- part 2: the gate, the abandon, the states ---------------------------------------------------
    async def engage(self, *, limit_ms: int = 480) -> int:
        """Mute over the wire and return the ``t_ms`` of the first all-zero frame fed after it (A5/A6: the span's
        first bound). A gate that never zeroes a frame is A5's named failure, not a hang."""
        assert self.rec_stt is not None, "rig premise: the STT is a RecordingStt"
        mark = len(self.rec_stt.fed)
        self.send_mute(True)
        await self.pump()
        waited = 0
        while not any(self.rec_stt.zero()[mark:]):
            assert waited < limit_ms, (
                f"T-0181 A5: no all-zero frame reached stt.feed within {limit_ms} ms of astra.mic.mute(true); "
                f"{len(self.rec_stt.fed) - mark} frames were fed live since (the gate is absent)"
            )
            await self.advance(40, step=40)
            waited += 40
        return self.rec_stt.t_ms()[mark + self.rec_stt.zero()[mark:].index(True)]

    async def release(self, *, limit_ms: int = 480) -> int:
        """Unmute over the wire and return the ``t_ms`` of the first captured (non-zero) frame fed after it."""
        assert self.rec_stt is not None, "rig premise: the STT is a RecordingStt"
        mark = len(self.rec_stt.fed)
        self.send_mute(False)
        await self.pump()
        waited = 0
        while all(self.rec_stt.zero()[mark:]):
            assert waited < limit_ms, (
                f"T-0181 A5: no captured frame reached stt.feed within {limit_ms} ms of astra.mic.mute(false)"
            )
            await self.advance(40, step=40)
            waited += 40
        return self.rec_stt.t_ms()[mark + self.rec_stt.zero()[mark:].index(False)]

    def say(self, text: str, start_ms: int, ms_per_word: int = 200) -> list[Any]:
        """Script words on the recogniser in *audio* time (``FakeStt.words``: honest durations); the reactive fake
        releases each once fed audio passes ``end_ms + lag_ms``."""
        assert self.rec_stt is not None, "rig premise: the STT is a RecordingStt"
        spoken = FakeStt.words(text, start_ms, ms_per_word)
        self.rec_stt.inner.script.extend(spoken)
        return spoken

    def stepped(self, kind: str, since: int = 0) -> list[Any]:
        """Events of wire type ``kind`` the machine was stepped with, from log index ``since``."""
        return [ev for ev, _, _ in self.machine.log[since:] if ev.type == kind]

    def mic_muted_events(self) -> list[Any]:
        return self.stepped("mic.muted")

    def commands_of(self, kind: str, since: int = 0) -> list[Any]:
        return [c for _, _, cmds in self.machine.log[since:] for c in cmds if c.type == kind]

    def transcripts(self, turn_id: str | None = None) -> list[Any]:
        return [
            m
            for m in self.published.messages
            if getattr(m, "type", None) == "astra.transcript"
            and (turn_id is None or m.turn_id == turn_id)
        ]

    def raw_lines(self) -> list[dict[str, Any]]:
        """``turns.jsonl`` as raw JSON (the ``extras`` keys are read as written)."""
        return [json.loads(line) for line in self.lines()]

    def log_text(self) -> str:
        """The whole session log file as text (after ``aclose``)."""
        return Path(self.log.path).read_text(encoding="utf-8")

    def published_text(self) -> str:
        """Every published message, JSON-encoded, one per line."""
        return "\n".join(encode(m) for m in self.published.messages)


class TapAudio(RecordingAudio):
    """``RecordingAudio`` that also keeps every STT frame its capture yields, packed exactly as the pump packs it
    (little-endian f32 bytes) with its integer ``t_ms``: the oracle for "one feed per captured frame" (A5)."""

    def __init__(self, clock: Any) -> None:
        super().__init__(clock)
        self.yielded: list[tuple[bytes, int]] = []

    async def stt_blocks(self) -> Any:
        async for frame in super().stt_blocks():
            pcm = np.ascontiguousarray(frame.samples, dtype="<f4").tobytes()
            self.yielded.append((pcm, int(frame.t_ms)))
            yield frame


def zero_spans(rec_stt: RecordingStt) -> list[tuple[int, int]]:
    """``(first, last)`` ``t_ms`` of every run of consecutive all-zero fed frames (A6's span, on the tap clock)."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    last = 0
    for (_, t), zero in zip(rec_stt.fed, rec_stt.zero(), strict=True):
        if zero and start is None:
            start = t
        if zero:
            last = t
        if not zero and start is not None:
            spans.append((start, last))
            start = None
    if start is not None:
        spans.append((start, last))
    return spans


#: part 2b: the barge-in scenario (A-R3, A-R12; A7's PAUSED/INTERRUPTED rows). A typed question whose answer is
#: about five seconds of speech at 200 ms a word, so a barge-in lands mid-answer
QUESTION = "tell me about the harbour"
LONG_ANSWER = (
    "The harbour was built long ago by people who fished there every single morning. "
    "Its stone walls still stand today and boats come and go all day long. "
)
#: the barge-in, 200 ms a word from its onset N (the last word starts at N + 800, past the takeover at N + 600, so a
#: mute in INTERRUPTED still has a barge-in word inside its span)
BARGE = "wait stop that please now"


async def barge_in(rig: MicRig, text: str = BARGE, *, question: str = QUESTION) -> list[Any]:
    """Type ``question``, wait until the answer has played 400 ms, then barge in: script ``text`` on the recogniser at
    audio time N (the clock now; the ``FakeAudioIO`` capture clock starts with the session clock) and submit the
    reflex lane's ``UserSpeechOnset(onset_ms=N)`` ourselves (A-R12), so the machine is PAUSED. Returns the scripted
    words. The takeover deadline is N + ``takeover_ms`` (600)."""
    from local_astra.contracts import events as E
    from local_astra.contracts.turn import TurnState

    await rig.advance(400, step=40)
    await rig.type_question(question)
    await rig.until(
        lambda: rig.state.state is TurnState.SPEAKING,
        "premise: the answer never started",
        limit_ms=6_000,
        step=40,
    )
    await rig.advance(400, step=40)
    assert rig.state.state is TurnState.SPEAKING, f"premise: SPEAKING; {rig.state.state}"
    now = rig.clock.now_ms
    spoken = rig.say(text, now)
    rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
    await rig.pump()
    assert rig.state.state is TurnState.PAUSED, (
        f"premise: the onset paused the answer; {rig.state.state}"
    )
    return spoken


def mic_states(messages: Sequence[Any]) -> list[Any]:
    return [m for m in messages if getattr(m, "type", None) == STATE]


def triple(message: Any) -> tuple[Any, Any, Any]:
    """``(muted, available, reason)`` of one ``astra.mic.state``."""
    return (
        getattr(message, "muted", None),
        getattr(message, "available", None),
        getattr(message, "reason", None),
    )


def told(page: Any, since: int = 0) -> list[tuple[str, Any]]:
    """``(type, code)`` of what a page was sent from index ``since``."""
    return [(m.type, getattr(m, "code", None)) for m in page.sent()[since:]]


def types(messages: Sequence[Any]) -> list[str]:
    return [getattr(m, "type", type(m).__name__) for m in messages]


def is_zero(samples: Any) -> bool:
    return not np.any(np.asarray(samples))


def require_session_keywords(names: Sequence[str]) -> None:
    """A ``Session`` keyword this task adds is a named ``AssertionError`` while it is missing, never a ``TypeError``."""
    params = inspect.signature(Session.__init__).parameters
    missing = [n for n in names if n not in params]
    assert not missing, (
        f"T-0181 A3 (A-R1): Session(..., {', '.join(f'{n}=...' for n in missing)}) does not exist yet; "
        f"`mic_unavailable: str | None` -- omitted: derived from whether audio is a CaptureSource; a string: "
        f"unavailable with that reason"
    )


@asynccontextmanager
async def mic_rig(
    tmp_path: Path,
    *,
    mic: bool = True,
    mic_unavailable: Any = _OMITTED,
    script: Sequence[Any] = (),
    continuation: Sequence[Any] | None = None,
    kit: Any = None,
    stt: Any = None,
    hears_audio: bool = False,
    stt_kwargs: dict[str, Any] | None = None,
    llm: Any = None,
    tts: Any = None,
    audio: Any = None,
    machine_cls: type = RecordingMachine,
    policy: Any = POLICY,
    frames: Sequence[Any] | None = None,
    ms_per_word: int = MS_PER_WORD,
    session_kwargs: dict[str, Any] | None = None,
    clock: ScriptedClock | None = None,
) -> AsyncIterator[MicRig]:
    """One T-0181 session over the fakes, started, with a socketless ``HudServer`` and one greeted page.

    ``mic``: ``True`` -- the audio is ``RecordingAudio`` (a ``CaptureSource``); ``False`` -- ``DeafAudio`` (none).
    ``audio`` overrides both (a ``callable(clock)``). ``mic_unavailable``: passed as the A-R1 keyword only when
    given (omitted: the session derives it). ``kit``: ``None`` tools off, ``"default"`` ``dry_run`` over a
    ``WatchingExecutor``, or a kit object (e.g. ``test_t0177_shared_frame.live_kit(...)``). ``stt``: an ``SttClient``
    (default ``reactive_stt(hears_audio=..., **stt_kwargs)``); when it is a ``RecordingStt`` it is ``rig.rec_stt``.
    ``session_kwargs``: further ``Session`` keywords, each checked against the signature first.
    """
    require_session()
    extra_kwargs = dict(session_kwargs or {})
    if mic_unavailable is not _OMITTED:
        extra_kwargs[MIC_KWARG] = mic_unavailable
    require_session_keywords(list(extra_kwargs))
    clock = clock or ScriptedClock()
    machine = machine_cls(policy)
    if llm is None:
        llm = ToolLlm(list(script))
    llm.clock = clock
    if continuation is not None:
        llm.script_continuation(*continuation)
    tts = tts if tts is not None else RecordingTts(ms_per_word=ms_per_word)
    if audio is not None:
        audio_obj = audio(clock)
    else:
        audio_obj = RecordingAudio(clock) if mic else DeafAudio(clock)
    if stt is None:
        stt = reactive_stt(hears_audio=hears_audio, **(stt_kwargs or {}))
    perception = RecordingPerception(list(frames) if frames is not None else [tool_frame(0, "a")])
    published = Recorder()
    log = make_log(tmp_path / "logs")
    executor: Any = None
    if kit == "default":
        executor = WatchingExecutor()
        executor.audio, executor.clock = audio_obj, clock
        kit = T.make_kit(executor=executor)
    turns_path = tmp_path / "turns.jsonl"
    kwargs: dict[str, Any] = {"publish": published, "session_log": log, **extra_kwargs}
    if kit is not None:
        kwargs[T.KIT_KWARG] = kit
    session = Session(
        machine,
        llm,
        stt,
        tts,
        audio_obj,
        perception,
        FakeMemory(),
        clock=clock,
        config=SessionConfig(data_dir=tmp_path, turns_path=turns_path),
        **kwargs,
    )
    rig = MicRig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=stt,
        tts=tts,
        audio=audio_obj,
        perception=perception,
        memory=session.memory,
        session=session,
        turns_path=turns_path,
        published=published,
        log=log,
        executor=executor,
        kit=kit,
        rec_stt=stt if isinstance(stt, RecordingStt) else None,
    )
    await session.start()
    try:
        from local_astra.conductor.hud_server import HudServer

        server = HudServer(session, frame_sink=_NullSink(), port=0)

        def tee(message: Any) -> None:
            server.publish(message)
            published(message)

        session.set_publisher(tee)
        rig.server = server
        rig.page = rig.greet()
        yield rig
    finally:
        await session.aclose()
        rig.log_records = read_records(log.path)
