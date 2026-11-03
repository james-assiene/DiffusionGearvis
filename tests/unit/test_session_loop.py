"""The Conductor session over the fakes: one pump, one clock, one record (T-0007 group A).

Criteria 1, 2, 3, 5 and the end-to-end shape of 12. Everything here runs on ``tests/fakes/`` and
the scripted clock: no GPU, no PipeWire, no network (docs/tasks/T-0007.md, "Group A").

Seams this module fixes
=======================
``docs/tasks/T-0007.md`` names the objects but not their signatures, and the other eight files of
this task are written against the shapes below. The implementer owns the design *behind* them;
these are the names the tests call. Report anything here that is wrong to the reviewer before
changing it (CLAUDE.md: the implementer never edits ``tests/``).

``local_astra/conductor/session.py``

    Session(machine, llm, stt, tts, audio, perception, memory, *, clock, config)
        The six ``contracts/protocols.py`` objects positionally, in the order the stub already
        has them, then the scripted clock and a ``SessionConfig``. Nothing else is injected:
        criterion 1 is that the module cannot reach a service by itself, so the *pure* components
        (``PromptBuilder``, ``SpeechGate``, ``UiGate``, ``AssistState``, ``IdleCaptioner``,
        ``filler_policy``) are constructed by the session from ``config``.

    SessionConfig(data_dir, *, turns_path=None, profile=Profile(name="lean"),
                  system_prompt=..., audio_path="null", source=Source.SCREEN)
        ``data_dir`` is where thumbnails and — unless ``turns_path`` overrides it — ``turns.jsonl``
        live (docs/02 §9; the task's Implementation notes ask for it to be configurable so tests
        use ``tmp_path``). ``profile`` and ``audio_path`` are regime fields (criterion 12).

    await session.start() / await session.aclose()
        Start and stop the pump and the producer adapters.

    session.submit(event) -> None
        Put one ``contracts/events.Event`` on the **single** pump queue. Every producer adapter
        (STT, playback, LLM, perception, timers, later the HUD) goes through this, and the pump
        consumes in arrival order (criterion 3). Tests use it for the producers T-0007a and
        T-0007b own (the reflex lane, ``stt.flushed``, HUD ingress).

    await session.settle(timeout_s=2.0) -> None
        Return once the pump has consumed every event queued so far and executed the commands
        they produced. Work the executor deliberately backgrounds (consuming a generate stream,
        synthesising, playing audio) is *not* awaited. Raises ``TimeoutError`` if the pump does not
        go quiet within ``timeout_s`` of real time.

    session.state -> MachineState        the machine state the pump is holding
    session.speech_gate -> SpeechGate | None    this turn's gate (criterion 4; None while IDLE)
    session.assists -> AssistState       the one assist authority (criterion 10; T-0007b's ingress
                                         toggles it)
    session.executor -> CommandExecutor  the object criterion 2 enumerates

``local_astra/conductor/executor.py``

    class CommandExecutor
        executor.handler_for(command) -> Callable[..., Awaitable[None]]
            The named executor for one ``contracts/commands.Command``. Raises ``UnknownCommand``
            for an object that is not a member of the union (criterion 2).
        await executor.execute(command) -> None
            Dispatch. ``llm.abort`` / ``tts.cancel`` / ``playback.flush`` are idempotent
            (``IDEMPOTENT_COMMANDS``, docs/02 §4).
    class UnknownCommand(TypeError)

``local_astra/telemetry/turns_writer.py``

    TurnsWriter(path) with ``.write(record: TurnRecord) -> None`` and ``.path`` — one JSON line per
    turn, appended on the turn-ending row (criterion 12).

The clock
    Anything with ``now_ms: int`` and ``call_at(at_ms: int, callback: Callable[[], None]) -> handle``
    where ``handle.cancel()`` unschedules. ``ArmTimer``/``CancelTimer`` are implemented **only**
    by scheduling the session's own tick on it (criterion 3), and the tick is stamped
    ``Tick(t_ms=clock.now_ms)`` when the callback fires. ``tests/conftest.VirtualClock`` plus that
    one method is ``ScriptedClock`` below; the production clock is a class in ``session.py``.

Two more shapes the criteria need and the task leaves open (reported as such):

* **clips.** ``PlayClip`` is rendered through ``TtsBackend.speak`` and played with
  ``audio.play(utterance, source="")`` — ``source=""`` because a clip is never history text
  (docs/02 §4: history holds played *answer* text). Criterion 11 counts clip plays this way.
* **the interim filler.** The session calls ``filler_policy.decide`` through the *module*
  (``from local_astra.reflex import filler_policy`` … ``filler_policy.decide(...)``), so the
  decision and its inputs are observable; ``test_session_fillers.py`` asserts on them.
"""

from __future__ import annotations

import ast
import asyncio
import typing
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.commands import IDEMPOTENT_COMMANDS, Command
from local_astra.contracts.llm import Commit, LlmOptions, Prompt
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.turn import MachineState, TurnState
from local_astra.memory.captioner import CAPTION_PROMPT
from local_astra.telemetry.turns_log import TurnRecord
from tests.conftest import VirtualClock
from tests.fakes import FakeAudioIO, FakeLlm, FakeMemory, FakePerception, FakeStt, FakeTts

REPO = Path(__file__).resolve().parents[2]

#: The files ``docs/tasks/T-0007.md`` lists under "Owned files".
OWNED_FILES = (
    "local_astra/conductor/session.py",
    "local_astra/conductor/executor.py",
    "local_astra/telemetry/turns_writer.py",
    "local_astra/contracts/protocols.py",
    "scripts/demo/session_rung.py",
)

#: sha256 of ``local_astra/conductor/turn_machine.py``. Criterion 3's negative: the machine is
#: *not* in T-0007's Owned files and must be byte-identical when the task ends. A later task that
#: legitimately edits the machine updates this constant in its own PR — T-0059 group A is the
#: first (ruling A7: ``turn_phashes``, ``max_frames_per_turn`` and the ``scene.changed`` row).
#: Re-pinned by T-0162 group A (A-2, the ``session.reset`` row): sha256 of turn_machine.py at the
#: implementation commit b946f207 (was be46b60b..., the d3bc6ce / T-0114 content).
#: Re-pinned by T-0178 B6a (R15/R16 rewritten: the honest ending, B2-B5): sha256 of turn_machine.py at the
#: implementation commit f8e26760 (56 985 bytes; was 33776b8b..., the b946f207 / T-0162 content).
#: Re-pinned again by T-0178 amendment B3a (``_SEARCH_CAP_REASONS``: a forced-search refusal voices a plain answer):
#: sha256 of turn_machine.py at the fix commit 52be36b0 (57 655 bytes; was a56d6a8a..., the f8e26760 content).
#: Re-pinned by T-0179 B (B-6a: the ``LlmError`` row plays ``lost_thread`` on ``DEGENERATE_RETRIED_MESSAGE``): sha256 of
#: turn_machine.py at the implementation commit ebaf4248 (was 9b01cdfe..., the 52be36b0 / T-0178 content).
#: Re-pinned by T-0182 B (B1 ``ack_clip``, B4 ``still_working_after_ms``): sha256 of turn_machine.py at the
#: implementation commit afc57335, unchanged at b8d26ae7 (was f462bff5..., the ebaf4248 / T-0179 content).
#: Re-pinned by T-0181 A0 (A0-7: the ``MicMuted`` abandon row): sha256 of turn_machine.py at the implementation commit
#: 9dbddf96 (was 507c51eb..., the afc57335 / T-0182 content, still the blob at T-0181's base 021c2ef6).
#: Re-pinned by T-0192 B (B2: the hand-back row in ``_on_tool_result``; B-RV6, both ends pinned): sha256 of
#: turn_machine.py at the implementation commit 97409910 (was 634c9260..., the 9dbddf96 / T-0181 A0 content, still the
#: blob at T-0192 B's literal base da29ddbe, ``test_t0192_hand_back_machine.T0192_BASE_SHA256``).
#: Re-pinned by T-0193 A0 (A0-9: the UserStop rows; A0-R5, both ends pinned): sha256 of turn_machine.py at the
#: implementation commit 373df531 (was 30506023..., the 97409910 / T-0192 B content, still the blob at T-0193 A0's
#: literal base fdb9fc01, ``test_t0193_stop_row.T0193_BASE_SHA256``).
TURN_MACHINE_SHA256 = "9944ee489b89bcb15720c5d130c735bfb1168145f5b730dc72b2a9433b41a266"

POLICY = TurnPolicy(thinking=True)

# --- guarded imports: absence of a T-0007 symbol is an assertion, never a collection error -------

_MISSING: dict[str, str] = {}

try:  # session.py exists as a stub; SessionConfig and the new surface do not
    from local_astra.conductor.session import Session, SessionConfig
except ImportError as exc:  # pragma: no cover - the pre-implementation path
    Session = SessionConfig = None  # type: ignore[assignment,misc]
    _MISSING["session"] = f"local_astra.conductor.session: {exc}"

try:
    from local_astra.conductor.executor import CommandExecutor, UnknownCommand
except ImportError as exc:  # pragma: no cover
    CommandExecutor = UnknownCommand = None  # type: ignore[assignment,misc]
    _MISSING["executor"] = f"local_astra.conductor.executor: {exc}"

try:
    from local_astra.telemetry.turns_writer import TurnsWriter
except ImportError as exc:  # pragma: no cover
    TurnsWriter = None  # type: ignore[assignment,misc]
    _MISSING["turns_writer"] = f"local_astra.telemetry.turns_writer: {exc}"


#: Every symbol T-0007 must add. Referenced here so the guarded imports above are not "unused".
T0007_SYMBOLS = (Session, SessionConfig, CommandExecutor, UnknownCommand, TurnsWriter)


def require_session() -> None:
    """Fail with the T-0007 symbol that is missing, instead of erroring at import time."""
    assert not _MISSING, (
        "T-0007 is not implemented: "
        + "; ".join(sorted(_MISSING.values()))
        + " (docs/tasks/T-0007.md)"
    )


# --- the scripted clock ---------------------------------------------------------------------


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
    """``VirtualClock`` plus the one scheduling seam the session may use (criterion 3).

    ``advance`` fires every wake-up whose deadline it passes, in deadline order, with ``now_ms``
    set to the deadline — so a ``Tick`` the session stamps inside the callback carries the time
    the timer was armed for, not the end of the advance.
    """

    scheduled: list[_Wake] = field(default_factory=list)
    seq: int = 0

    def call_at(self, at_ms: int, callback: Callable[[], None]) -> _Wake:
        self.seq += 1
        wake = _Wake(at_ms=int(at_ms), callback=callback, seq=self.seq)
        self.scheduled.append(wake)
        return wake

    def pending(self) -> list[_Wake]:
        return [w for w in self.scheduled if not w.cancelled]

    def next_wake_ms(self) -> int | None:
        live = [w.at_ms for w in self.pending()]
        return min(live) if live else None

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


# --- recording fakes (the six protocol objects, with call logs) -------------------------------


class RecordingLlm(FakeLlm):
    """``FakeLlm`` with a call log and an optional hold on the commit stream.

    The hold is T-0005 M5's window: ``generate`` is an async generator, so nothing happens until
    the first ``__anext__`` — which is exactly where a barge-in lands. ``hold()`` keeps the stream
    suspended there; ``release()`` lets it run. A commit produced while the request was aborted is
    dropped, as the engine drops it.
    """

    def __init__(self, script: typing.Sequence[Commit] = (), *, prefix_tokens: int = 1000) -> None:
        super().__init__(script, prefix_tokens=prefix_tokens)
        self.clock: ScriptedClock | None = None
        self.calls: list[tuple[str, str, int]] = []
        self.gen_prompts: list[tuple[str, Prompt]] = []
        self.live_prefills: set[str] = set()
        self.live_generates: set[str] = set()
        self.live_captions: set[str] = set()
        #: (prefills, turn generates, captions) after every call that changes what is live
        self.concurrency: list[tuple[int, int, int]] = []
        self.fail_generate = False
        self.fail_prefill = False
        self._flow = asyncio.Event()
        self._flow.set()
        self._hold_after: int | None = None

    def hold(self) -> None:
        self._flow.clear()

    def hold_after(self, commits: int) -> None:
        """Deliver ``commits`` commits, then keep the generate open (no ``llm.done``).

        ``hold()`` suspends the stream *before* its first commit, so a turn that must reach
        SPEAKING and only then be interrupted cannot use it (ruling A8: a single-commit script
        ends with ``llm.done``, which closes the gate and legitimately voices the held tail).
        """
        self._hold_after = commits

    def release(self) -> None:
        self._hold_after = None
        self._flow.set()

    def _note(self, what: str, request_id: str) -> None:
        self.calls.append((what, request_id, self.clock.now_ms if self.clock else 0))
        self.concurrency.append(
            (len(self.live_prefills), len(self.live_generates), len(self.live_captions))
        )

    def times(self, what: str) -> list[int]:
        return [t for kind, _, t in self.calls if kind == what]

    async def prefill(self, prefix: Prompt, request_id: str):
        if self.fail_prefill:
            self._note("prefill", request_id)
            raise RuntimeError("prefill exploded")
        self.live_prefills.add(request_id)
        self._note("prefill", request_id)
        return await super().prefill(prefix, request_id)

    async def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
    ):
        caption = CAPTION_PROMPT[:40] in " ".join(m.text() for m in prompt.messages)
        if not caption:
            self.gen_prompts.append((request_id, prompt))
        if self.fail_generate and not caption:
            self._note("generate", request_id)
            raise RuntimeError("vLLM returned HTTP 400: prompt too long")
        (self.live_captions if caption else self.live_generates).add(request_id)
        self._note("caption" if caption else "generate", request_id)
        delivered = 0
        try:
            async for commit in super().generate(prompt, request_id, options, xargs=xargs):
                if not caption and self._hold_after is not None and delivered >= self._hold_after:
                    self._flow.clear()
                await self._flow.wait()
                if request_id not in self.active:
                    return
                yield commit
                delivered += 1
        finally:
            self.live_generates.discard(request_id)
            self.live_captions.discard(request_id)
            self.concurrency.append(
                (len(self.live_prefills), len(self.live_generates), len(self.live_captions))
            )

    async def abort(self, request_id: str) -> None:
        self.live_prefills.discard(request_id)
        self.live_generates.discard(request_id)
        self.live_captions.discard(request_id)
        self._note("abort", request_id)
        await super().abort(request_id)


class RecordingTts(FakeTts):
    def __init__(self, ms_per_word: int = 300) -> None:
        super().__init__(ms_per_word=ms_per_word)
        self.requests: list[tuple[str, str, str]] = []
        self.fail = False
        self._flow = asyncio.Event()
        self._flow.set()

    def hold(self) -> None:
        self._flow.clear()

    def release(self) -> None:
        self._flow.set()

    async def speak(self, utterance_id: str, text: str, voice: str = "default"):
        self.requests.append((utterance_id, text, voice))
        await self._flow.wait()
        if self.fail:
            raise RuntimeError("pocket-tts is down")
        return await super().speak(utterance_id, text, voice)

    def texts(self) -> list[str]:
        return [text for _, text, _ in self.requests]


class RecordingPerception(FakePerception):
    def __init__(self, frames: typing.Sequence[Frame] = ()) -> None:
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
        self.fail_play = False

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        self.calls.append("play")
        if self.fail_play:
            raise RuntimeError("the playback device went away")
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

    def clip_plays(self) -> list[Any]:
        """Clips are played with ``source=""`` (module docstring), history text never is."""
        return [u for u, source in self.plays if source == ""]


class RecordingMachine(TurnMachine):
    """The production machine, with a log of what the pump fed it and what came back."""

    def __init__(self, policy: TurnPolicy | None = None) -> None:
        super().__init__(policy or POLICY)
        self.log: list[tuple[E.Event, TurnState, tuple[Command, ...]]] = []
        self.steps = 0

    def step(self, state: MachineState, event: E.Event):
        new, cmds = super().step(state, event)
        self.steps += 1
        self.log.append((event, new.state, tuple(cmds)))
        return new, cmds

    def events(self, kind: type[E._Event] | None = None) -> list[E.Event]:
        return [ev for ev, _, _ in self.log if kind is None or isinstance(ev, kind)]

    def commands(self, kind: type[Any] | None = None) -> list[Command]:
        out: list[Command] = []
        for _, _, cmds in self.log:
            out.extend(c for c in cmds if kind is None or isinstance(c, kind))
        return out

    def transitions(self) -> list[tuple[TurnState, E.Event, TurnState]]:
        """``(state before, event, state after)`` for every step, reconstructed from the log.

        The pump feeds the machine one event at a time, so the state before step *i* is the state
        after step *i-1* (and ``IDLE`` for the first). Ruling A8: assertions about *when* the
        machine moved belong here — the state at the end of a settle window is a race with how
        much of the script the fake LLM delivered, and says nothing about which event moved it.
        """
        out: list[tuple[TurnState, E.Event, TurnState]] = []
        before = TurnState.IDLE
        for event, after, _ in self.log:
            out.append((before, event, after))
            before = after
        return out

    def moved(self, frm: TurnState, to: TurnState) -> list[E.Event]:
        """The events that took the machine from ``frm`` to ``to``."""
        return [ev for before, ev, after in self.transitions() if before is frm and after is to]


# --- fixtures and the rig ----------------------------------------------------------------------


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


@dataclass
class Rig:
    clock: ScriptedClock
    machine: RecordingMachine
    llm: RecordingLlm
    stt: FakeStt
    tts: RecordingTts
    audio: RecordingAudio
    perception: RecordingPerception
    memory: FakeMemory
    session: Any
    turns_path: Path
    #: extra pump for an audio implementation that is driven by its own device task
    settle_audio: Any = None

    # -- driving ------------------------------------------------------------------------------
    def submit(self, event: E.Event) -> None:
        self.session.submit(event)

    async def pump(self, rounds: int = 4) -> None:
        """Let the pump, the producer tasks and the audio poller run to quiescence."""
        for _ in range(rounds):
            self.clock.advance(0)
            await asyncio.sleep(0)
            if self.settle_audio is not None:
                await self.settle_audio()
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
        if not self.turns_path.exists():
            return []
        return [
            TurnRecord.from_line(line)
            for line in self.turns_path.read_text().splitlines()
            if line.strip()
        ]

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

    async def ask(self, text: str = "what is on this screen", *, onset: bool = True) -> None:
        """Onset → words → end-of-turn → flushed → tick: the machine reaches THINKING.

        ``onset=False`` for a turn the machine opened itself (the INTERRUPTED → LISTENING row
        opens one), where a second ``user.speech_onset`` would be a no-op anyway.
        """
        if onset:
            await self.onset()
        await self.words(text)
        await self.end_of_turn()

    async def speak_out(self) -> None:
        """Let the queued audio play out so ``playback.drained`` can arrive."""
        await self.until(
            lambda: self.state.state is TurnState.IDLE, "the turn never returned to IDLE"
        )


def make_rig(
    tmp_path: Path,
    *,
    script: typing.Sequence[Commit] = (),
    frames: typing.Sequence[Frame] = (),
    prefix_tokens: int = 1000,
    ms_per_word: int = 300,
    policy: TurnPolicy | None = None,
    config_kwargs: dict[str, Any] | None = None,
    llm: RecordingLlm | None = None,
    stt: FakeStt | None = None,
    tts: RecordingTts | None = None,
    memory: FakeMemory | None = None,
    clock: ScriptedClock | None = None,
    audio: Any = None,
    settle_audio: Any = None,
) -> Rig:
    require_session()
    clock = clock or ScriptedClock()
    machine = RecordingMachine(policy or POLICY)
    llm = llm if llm is not None else RecordingLlm(script, prefix_tokens=prefix_tokens)
    llm.clock = clock
    stt = stt if stt is not None else FakeStt()
    tts = tts if tts is not None else RecordingTts(ms_per_word=ms_per_word)
    audio = audio if audio is not None else RecordingAudio(clock)
    perception = RecordingPerception(frames or [frame(0, "a")])
    memory = memory if memory is not None else FakeMemory()
    turns_path = tmp_path / "turns.jsonl"
    config = SessionConfig(data_dir=tmp_path, turns_path=turns_path, **(config_kwargs or {}))
    session = Session(machine, llm, stt, tts, audio, perception, memory, clock=clock, config=config)
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
        settle_audio=settle_audio,
    )


@asynccontextmanager
async def running(tmp_path: Path, **kwargs: Any):
    rig = make_rig(tmp_path, **kwargs)
    await rig.session.start()
    try:
        yield rig
    finally:
        await rig.session.aclose()


# --- criterion 1 and 3: one scripted turn, end to end ------------------------------------------


async def test_a_scripted_turn_runs_end_to_end_over_the_fakes(tmp_path):
    """Criteria 1, 3 and 12: onset → words → end-of-turn → flushed → tick → commits → done →
    drained gives one ``TurnRecord``, the state back to IDLE and the speak calls in order.

    Every expected value is computed by hand from the script: the answer is two sentences, so the
    speech gate emits two utterances whose *sources* rebuild the committed answer exactly, and the
    history holds what the (virtual) device reported playing — nothing else.
    """
    script = [
        thought("the screen shows a stack trace"),
        answer("It is a stack trace. "),
        answer("The error is on line four."),
    ]
    async with running(tmp_path, script=script, frames=[frame(500, "a")]) as rig:
        await rig.ask("what is on this screen")
        started = rig.machine.moved(TurnState.ENDPOINTING, TurnState.THINKING)
        assert len(started) == 1 and isinstance(started[0], E.Tick), (
            "criterion 3: the settle tick must reach the machine — ENDPOINTING is left on a Tick "
            f"{POLICY.eot_settle_ms} ms after stt.flushed, not on a drain timer and not on an "
            f"event the session synthesised; it was left on {started}"
        )
        assert rig.state.state in (TurnState.THINKING, TurnState.SPEAKING), (
            f"the turn is under way after the settle tick (state={rig.state.state})"
        )
        await rig.speak_out()

        assert rig.state.state is TurnState.IDLE
        assert [text for _, text, _ in rig.tts.requests if text] == [
            "It is a stack trace.",
            "The error is on line four.",
        ], f"criterion 1: the two committed sentences, in order; got {rig.tts.texts()}"

        history = rig.state.history
        assert len(history) == 1 and history[0].user_text == "what is on this screen"
        assert history[0].assistant_spoken == "It is a stack trace. The error is on line four.", (
            "criterion 6: history is the played answer text, byte for byte "
            f"(got {history[0].assistant_spoken!r})"
        )
        prompt_text = rig.llm.gen_prompts[0][1].messages[-1].text()
        assert "what is on this screen" in prompt_text, (
            f"the transcript rides in the query-dependent suffix (docs/01 D9); got {prompt_text!r}"
        )

        lines = rig.lines()
        assert len(lines) == 1, f"criterion 12: exactly one turns.jsonl line per turn, got {lines}"
        record = rig.records()[0]
        assert record.turn_id == "t1" and record.outcome == "spoken"
        assert record.unknown_spans() == [], record.spans


# --- criterion 2: the executor is total --------------------------------------------------------


def command_union_members() -> list[type[Any]]:
    """Every member of the ``Command`` union, the way T-0003 A18 enumerated ``EVENT_TYPES``."""
    annotated = typing.get_args(Command)[0]
    return list(typing.get_args(annotated))


def sample_command(kind: type[Any]) -> Any:
    """One instance of ``kind`` with every required field filled."""
    values: dict[str, Any] = {
        "at_ms": 100,
        "request_id": "t1-gen",
        "t_eot_ms": 100,
        "from_ms": 0,
        "to_ms": 100,
        "transcript": "hello",
        "kind": "ack",
        "name": "llm_timeout",
        "utterance_id": "t1-u1",
        "text": "hello",
        "source": "hello",
        "state": "THINKING",
        "turn_id": "t1",
        "message": "boom",
        "mode": "text_only",
        # T-0089 criterion 4: the three tool commands
        "draft_request_id": "t1-gen",
        "call_id": "c1",
        "arguments": {"x": 1, "y": 2},
        "reason": "interrupted",
        # T-0112: LlmContinue(request_id, step, call_id); request_id and call_id are above
        "step": 2,
    }
    # a field whose generic sample above is not a legal value for this kind
    overrides: dict[str, dict[str, Any]] = {
        "ToolExecute": {"source": "draft", "name": "click_at"},
        "ToolRepairStart": {"request_id": "t1-repair"},
    }
    values = {**values, **overrides.get(kind.__name__, {})}
    fields = {
        name: values[name]
        for name in kind.model_fields
        if name in values and kind.model_fields[name].is_required()
    }
    return kind(**fields)


async def test_every_command_type_has_an_executor(tmp_path):
    """Criterion 2: the same guard T-0003 A18 added for ``EVENT_TYPES``, on the command side.

    A command the machine can emit and the executor cannot dispatch is dropped silently today;
    this enumerates the union so the *absence* of a handler is a named failure.
    """
    async with running(tmp_path) as rig:
        executor = rig.session.executor
        members = command_union_members()
        assert len(members) >= 25, f"the Command union shrank: {len(members)} members"
        missing: list[str] = []
        for kind in members:
            try:
                handler = executor.handler_for(sample_command(kind))
            except Exception as exc:  # the point of the test is that this cannot happen
                missing.append(f"{kind.__name__}: {type(exc).__name__}: {exc}")
                continue
            if not callable(handler):
                missing.append(f"{kind.__name__}: handler_for returned {handler!r}")
        assert not missing, (
            "criterion 2: every member of the Command union dispatches to a named executor; "
            "these do not:\n  " + "\n  ".join(missing)
        )


TOOL_COMMANDS = ("ToolRepairStart", "ToolExecute", "ToolCancel")


async def test_t0089_tool_commands_have_named_inert_handlers(tmp_path):
    """T-0089 criterion 4: a named handler per new command, inert until T-0092.

    Inert means no effect on any service double: no LLM request or abort, no speech, no playback,
    no history, no published HUD state. (The one session-log ``note`` the criterion allows is not
    asserted here; with no session log configured it has nowhere to go.)
    """
    kinds_ = {k.__name__: k for k in command_union_members()}
    missing = [n for n in TOOL_COMMANDS if n not in kinds_]
    assert not missing, f"T-0089 criterion 1: the Command union lacks {missing}"
    async with running(tmp_path) as rig:
        executor = rig.session.executor
        handlers = [executor.handler_for(sample_command(kinds_[n])) for n in TOOL_COMMANDS]
        names = [getattr(h, "__name__", "") for h in handlers]
        assert len(set(names)) == 3 and all(names), f"one named handler each: {names}"
        before = (
            list(rig.llm.calls),
            list(rig.tts.requests),
            list(rig.audio.plays),
            list(rig.audio.calls),
            rig.session.summaries_scheduled,
        )
        for n in TOOL_COMMANDS:
            await executor.execute(sample_command(kinds_[n]))
        after = (
            list(rig.llm.calls),
            list(rig.tts.requests),
            list(rig.audio.plays),
            list(rig.audio.calls),
            rig.session.summaries_scheduled,
        )
        assert after == before, "T-0089 criterion 4: the tool handlers are inert until T-0092"


async def test_an_object_that_is_not_a_command_raises(tmp_path):
    """Criterion 2: an unknown object raises rather than being dropped."""
    async with running(tmp_path) as rig:
        with pytest.raises(UnknownCommand):
            rig.session.executor.handler_for(object())


async def test_idempotent_commands_execute_once_per_cause(tmp_path):
    """Criterion 2, negative: ``IDEMPOTENT_COMMANDS`` twice = the effect once, no second event.

    The three are ``llm.abort``, ``tts.cancel`` and ``playback.flush`` (docs/02 §4). The second
    execution must raise nothing and must not put a second ``playback.stopped`` on the queue — a
    duplicate terminal event would give the machine a second history commit for one interruption.
    """
    script = [answer("A sentence that plays for a while.")]
    async with running(tmp_path, script=script, frames=[frame(0, "a")]) as rig:
        assert set(IDEMPOTENT_COMMANDS) == {"llm.abort", "tts.cancel", "playback.flush"}
        await rig.ask("hello there")
        await rig.until(
            lambda: rig.audio.plays != [] and rig.state.state is TurnState.SPEAKING,
            "the answer never reached playback",
        )
        before = len(rig.machine.events(E.PlaybackStopped))
        rid = rig.state.active_generate
        executor = rig.session.executor
        for _ in range(2):
            await executor.execute(C.LlmAbort(request_id=rid or "t1-gen"))
            await executor.execute(C.TtsCancel())
            await executor.execute(C.PlaybackFlush())
        await rig.pump()
        stopped = len(rig.machine.events(E.PlaybackStopped)) - before
        assert stopped == 1, (
            f"two playback.flush executions produced {stopped} playback.stopped events; the "
            "executor must be idempotent per cause (docs/02 §4 IDEMPOTENT_COMMANDS)"
        )
        assert rig.machine.events(E.LlmError) == [], "an idempotent abort must not raise or error"
        assert rig.machine.events(E.TtsError) == [], "an idempotent cancel must not error"


# --- criterion 3: timers are the session tick --------------------------------------------------


async def test_timers_are_the_session_tick_and_nothing_else(tmp_path):
    """Criterion 3: ``ArmTimer`` schedules the session tick, ``CancelTimer`` unschedules it.

    The scripted clock is the test double: it records every ``call_at``. The machine arms
    ``llm_timeout`` 20 s ahead when it starts thinking, so a wake-up at or before that deadline
    must exist; cancelling it (``llm.done`` cancels ``llm_timeout``) must drop it again. The
    cadence half is criterion 3's "a tick reaches the machine at least every 50 ms while any timer
    is armed": the machine's shortest deadline that matters is ``eot_settle_ms`` = 120 ms.
    """
    script = [answer("Yes.")]
    async with running(tmp_path, script=script) as rig:
        # A8: hold the generate open, or the fake delivers the whole script inside the settle
        # window and the machine is legitimately past THINKING (with ``llm_timeout`` cancelled)
        # before the arming half of this test can look at it.
        rig.llm.hold()
        await rig.onset()
        await rig.words("hello")
        await rig.advance(100, step=100)
        now = rig.clock.now_ms
        rig.submit(E.SttEndOfTurn(t_ms=now, t_eot_ms=now - 20, p_end=0.9))
        rig.submit(E.SttFlushed(t_ms=now, t_eot_ms=now - 20))
        await rig.pump()

        assert rig.state.state is TurnState.ENDPOINTING
        nxt = rig.clock.next_wake_ms()
        assert nxt is not None and nxt - rig.clock.now_ms <= 50, (
            "criterion 3: while a deadline is pending the session keeps a tick scheduled no more "
            f"than 50 ms ahead; the next wake-up is {nxt} at now={rig.clock.now_ms}"
        )

        await rig.advance(POLICY.eot_settle_ms + 40, step=40)
        started = rig.machine.moved(TurnState.ENDPOINTING, TurnState.THINKING)
        assert len(started) == 1 and isinstance(started[0], E.Tick), (
            f"criterion 3: ENDPOINTING is left on a Tick, not on a timer of its own; got {started}"
        )
        assert rig.state.state is TurnState.THINKING, "the generate is held open"
        armed = [t for t in rig.state.timers if t.name == "llm_timeout"]
        assert armed, "the machine armed llm_timeout at llm.start"
        deadline = armed[0].at_ms
        assert any(w.at_ms <= deadline for w in rig.clock.pending()), (
            f"criterion 3: ArmTimer(llm_timeout, {deadline}) must schedule the session's own tick; "
            f"pending wake-ups: {[w.at_ms for w in rig.clock.pending()]}"
        )

        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never finished")
        assert not any(w.at_ms == deadline for w in rig.clock.pending()), (
            "criterion 3: CancelTimer(llm_timeout) must unschedule the wake-up it armed; a "
            f"wake-up for the cancelled deadline {deadline} is still pending"
        )


def owned_source(name: str) -> str:
    path = REPO / name
    assert path.is_file(), f"T-0007 adds {name} (docs/tasks/T-0007.md, Owned files)"
    return path.read_text()


def test_nothing_in_the_owned_files_steps_the_machine_twice_or_edits_it():
    """Criterion 3, negative: one ``machine.step`` call site, and ``turn_machine.py`` untouched.

    ``local_astra/conductor/turn_machine.py`` is not in T-0007's Owned files. A second call site
    is how a session grows a side channel that bypasses the pump's arrival order.
    """
    sites: list[str] = []
    for name in OWNED_FILES:
        tree = ast.parse(owned_source(name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "step":
                sites.append(f"{name}:{node.lineno}")
    assert len(sites) == 1, (
        f"criterion 3: exactly one place may call machine.step; found {sites or 'none'}"
    )

    # T-0192 B re-pin (B-RV6, both ends pinned): the working tree is the merged 97409910 machine
    # (TURN_MACHINE_SHA256), and at T-0192 B's literal base da29ddbe it was T-0181 A0's 9dbddf96 content
    from tests.unit.test_t0192_hand_back_machine import (
        T0192_BASE,
        T0192_BASE_SHA256,
        T0192_MACHINE,
        T0192_MOVED_SHA256,
        t0192_end_sha256,
    )

    # T-0193 A0 re-pin (A0-9; A0-R5, both ends pinned, the T-0192 B-RV6 way): T-0192 B's tree end is the machine at
    # T-0193 A0's literal base fdb9fc01 (the base end), and the working tree is exactly the merged 373df531 blob
    # (TURN_MACHINE_SHA256, the tree end)
    from tests.unit.test_t0193_stop_row import T0193_BASE, T0193_BASE_SHA256, T0193_MOVED_SHA256

    assert T0192_MOVED_SHA256[T0192_MACHINE] == T0193_BASE_SHA256[T0192_MACHINE], (
        "T-0193 A0: the base end of the machine pin is T-0192 B's tree end"
    )
    got = t0192_end_sha256(T0192_MACHINE, T0193_BASE)
    assert got == T0193_BASE_SHA256[T0192_MACHINE], (
        f"T-0193 A0: turn_machine.py at A0's base {T0193_BASE[:8]} is not the merged 97409910 blob ({got})"
    )
    assert T0193_MOVED_SHA256[T0192_MACHINE] == TURN_MACHINE_SHA256, (
        "T-0193 A0: the tree end of the machine pin is TURN_MACHINE_SHA256"
    )
    got = t0192_end_sha256(T0192_MACHINE)
    assert got == TURN_MACHINE_SHA256, (
        "criterion 3, negative: turn_machine.py is not in T-0007's Owned files; it is the re-pinned "
        f"TURN_MACHINE_SHA256 content (T-0193 A0's 373df531 blob) and nothing else ({got}). A later "
        "task that legitimately edits the machine updates the pin in its own PR."
    )
    assert t0192_end_sha256(T0192_MACHINE, T0192_BASE) == T0192_BASE_SHA256[T0192_MACHINE], (
        "T-0192 B: the machine at T-0192 B's base da29ddbe is not its earlier pinned content (the base end)"
    )


def test_the_owned_files_use_no_second_timer_mechanism():
    """Criterion 3: time reaches the machine as ``Tick`` and comes from the injected clock only."""
    banned = (
        "time.sleep",
        "time.monotonic",
        "time.time(",
        "time.perf_counter",
        "datetime.now",
        "threading.Timer",
        "call_later",
        "loop.call_at",
        "get_event_loop",
        "asyncio.Timer",
    )
    offenders: list[str] = []
    for name in OWNED_FILES:
        body = owned_source(name)
        for lineno, line in enumerate(body.splitlines(), start=1):
            code = line.split("#", 1)[0]
            for token in banned:
                if token in code:
                    offenders.append(f"{name}:{lineno}: {token}")
            if "asyncio.sleep(" in code and "asyncio.sleep(0)" not in code:
                offenders.append(f"{name}:{lineno}: asyncio.sleep with a delay")
    assert not offenders, (
        "criterion 3: ArmTimer/CancelTimer are implemented only by scheduling the session's own "
        "tick on the injected clock, and there is no second timer mechanism:\n  "
        + "\n  ".join(offenders)
    )


# --- criterion 5: the latched-empty answer -----------------------------------------------------


async def test_an_answer_that_is_entirely_latched_still_ends_the_turn(tmp_path):
    """Criterion 5, adversarial — the test this task most needs.

    Every sentence of this answer normalises to ``""`` (the model emitted ``<|think|>`` before any
    prose, T-0004 A2/A3: everything from that token on is trimmed out of the spoken text, while
    the *source* still carries the bytes). Nothing may be sent to TTS and nothing may be played —
    and the machine must still reach IDLE. With ``llm_done=True``, ``playback_drained=False`` and
    ``llm_timeout`` already cancelled, the only row out of SPEAKING is
    ``playback.drained ∧ llm.done`` (docs/02 §4), so the session owes the drain.

    The obvious implementation — skip the empty utterance, forget the drain — hangs in SPEAKING
    for ever, which is why this is the mutation the reviewer will try first.
    """
    script = [answer("<|think|>"), answer(" still thinking about it.")]
    async with running(tmp_path, script=script) as rig:
        await rig.ask("are you there")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "criterion 5: an answer whose every sentence latches to '' must still end the turn",
            limit_ms=5_000,
        )
        assert rig.tts.requests == [] or all(not t for _, t, _ in rig.tts.requests), (
            f"criterion 5: a latched TtsSpeak(text='') is never sent to TtsBackend.speak; "
            f"got {rig.tts.requests}"
        )
        assert rig.audio.plays == [] or all(source == "" for _, source in rig.audio.plays), (
            f"criterion 5: nothing but clips may be played; got {rig.audio.plays}"
        )

        history = rig.state.history
        assert len(history) == 1 and history[0].assistant_spoken == "", (
            "criterion 5, negative: the skipped utterance contributes nothing to played_text or "
            f"to history (got {history[0].assistant_spoken!r})"
        )
        record = rig.records()[0]
        assert record.outcome == "silent", (
            "a turn that played no answer audio is 'silent' in TurnRecord.outcome "
            f"(got {record.outcome!r})"
        )
        line = rig.lines()[0]
        assert "<|think|>" not in line and "still thinking about it" not in line, (
            "criterion 12, negative: no thinking text anywhere in the turns.jsonl line "
            "(CLAUDE.md); the latched source must not be logged as spoken text"
        )
