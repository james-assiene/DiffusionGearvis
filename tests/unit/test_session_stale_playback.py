"""T-0092a: a playback run never outlives its turn, and a stale terminal never reaches the machine.

docs/tasks/T-0092a.md, criteria 1-5 (criterion 6's golden is ``test_session_stale_playback_golden.py``).
Origin: the T-0089 ruling T15 follow-up — ``PlaybackDrained`` carries no run or turn identity, so a
drain left over from turn 1 can end turn 2 early and put turn 1's words in turn 2's history.

How a terminal is made *late* here. ``tests/fakes/audio.py`` grew one knob for this task,
``hold_events()``/``release_events()``: the run keeps going and keeps emitting (a drain still needs
``mark_drain()`` and the last sample, ``flush()`` still ends the run and stays idempotent), only the
hand-over to the session's ``playback_events()`` consumer waits. That is the real module's shape too —
``audio_io.PipeWireAudioIO`` appends terminals to a list the consumer drains when it next iterates —
and ``tests/contract/test_audio_late_consumer.py`` holds the two to it.

Oracles, none of them the session's own bookkeeping:

* *which terminal is which* -> object identity against ``StaleAudio.delivered`` (what the fake
  handed the session, in order). The session submits the event object unchanged (T-0007
  criterion 14), so "the machine saw run 2's drain" is ``event is run2_drain``;
* *what turn 2 said* -> the committed source the test scripted, played in full;
* *what was dropped, and said so* -> the session log, read back from the file after ``aclose``;
* *how many flushes* -> the fake device's call log.

Regime: CPU over fakes, ``ScriptedClock``, thinking on (``TurnPolicy(thinking=True)``, so every
voice turn plays the zero-length ``ack`` clip at THINKING), 200 ms per word, n = 1 and
deterministic per row. No measurement.

Every row except the guard-width "too wide" rows and the drain-gate row is red on ``main`` at
0304f3f; those three are regression guards for the over-wide direction and are green there by
construction (see each docstring).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.protocol import AstraSpokenPlayed
from local_astra.contracts.turn import TurnState
from tests.fakes import FakeMemory, FakeStt, FakeTts
from tests.fakes.llm import is_caption_prompt
from tests.unit._hud_rig import Recorder
from tests.unit._session_log_rig import make_log, of_kind, read_records
from tests.unit.test_session_loop import (
    POLICY,
    RecordingAudio,
    RecordingLlm,
    RecordingMachine,
    RecordingPerception,
    RecordingTts,
    Rig,
    ScriptedClock,
    Session,
    SessionConfig,
    answer,
    frame,
    require_session,
)

# unit layer: no marker (tests/unit is the default layer, pyproject declares no "unit" marker)

MS_PER_WORD = 200

#: Turn 1's answer: four words, 800 ms of audio at 200 ms/word.
S1 = "Turn one says this. "
#: Turn 2's answer: three words, 600 ms. Shares no word with ``S1``, so "no text from turn 1" is a
#: substring check on any word of it.
S2 = "Second answer here. "
#: A long answer for the rows that must cut a run mid-play: ten words, 2000 ms.
LONG = "This answer is long enough to still be playing at the cut. "

#: The ``what`` of criterion 3's note, verbatim from the task file.
STALE_WHAT = "stale playback terminal"
#: The note's fields, verbatim from criterion 3.
NOTE_FIELDS = {"what", "run_id", "run_turn", "current_turn", "type", "played_ms", "played_text"}

TERMINALS = (E.PlaybackStopped, E.PlaybackDrained)


# --- the rig ------------------------------------------------------------------------------------


class StaleAudio(RecordingAudio):
    """T-0007's recording fake device, plus a log of every terminal *delivered* to the session."""

    def __init__(self, clock: Any) -> None:
        super().__init__(clock)
        self.delivered: list[E.PlaybackStopped | E.PlaybackDrained] = []

    async def playback_events(self) -> AsyncIterator[E.PlaybackStopped | E.PlaybackDrained]:
        async for event in super().playback_events():
            self.delivered.append(event)
            yield event

    def answer_sources(self) -> list[str]:
        return [source for _, source in self.plays if source]


class DoneGatedLlm(RecordingLlm):
    """T-0007's recording fake backend, plus a gate *after* a turn's last commit.

    ``gate_done()`` keeps the next turn generate open once its script is exhausted, so
    ``llm.done`` waits for ``open_done()``. ``hold_after`` cannot do this: it holds *before* a
    commit, and a script whose last commit has been delivered ends the stream at once. With the
    gate open (the default) this is ``RecordingLlm`` unchanged; everything else is delegated.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._done_gate = asyncio.Event()
        self._done_gate.set()

    def gate_done(self) -> None:
        self._done_gate.clear()

    def open_done(self) -> None:
        self._done_gate.set()

    async def generate(self, prompt: Any, request_id: str, *args: Any, **kwargs: Any) -> Any:
        async for commit in super().generate(prompt, request_id, *args, **kwargs):
            yield commit
        if not is_caption_prompt(prompt):
            await self._done_gate.wait()


class SelectiveTts(RecordingTts):
    """T-0007's recording fake TTS, plus failure for chosen texts only (``fail_on``).

    ``RecordingTts.fail`` fails every call; R2 needs exactly one synthesis — a closed turn's — to
    raise while the current turn's speech succeeds. Empty ``fail_on`` is ``RecordingTts``
    unchanged; everything else is delegated.

    The choice is read *after* the hold, as ``RecordingTts.fail`` is: a call already waiting
    inside ``speak`` when ``fail_on`` is set fails when released (R2's utterance is exactly that).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_on: set[str] = set()

    async def speak(self, utterance_id: str, text: str, voice: str = "default") -> Any:
        self.requests.append((utterance_id, text, voice))
        await self._flow.wait()
        if self.fail or text in self.fail_on:
            raise RuntimeError(f"pocket-tts failed on {text!r}")
        return await FakeTts.speak(self, utterance_id, text, voice)


@dataclass
class StaleRig(Rig):
    published: Recorder | None = None
    log: Any = None

    def seen(self, event: Any) -> bool:
        """Did the machine get stepped with this very object?"""
        return any(ev is event for ev in self.machine.events())

    def terminals_seen(self) -> list[Any]:
        return [ev for ev in self.machine.events() if isinstance(ev, TERMINALS)]

    def played_messages(self) -> list[AstraSpokenPlayed]:
        assert self.published is not None
        return self.published.of(AstraSpokenPlayed)

    def flushes(self) -> int:
        return self.audio.calls.count("flush")

    async def held_terminal(self, kind: type, what: str) -> Any:
        """Wait until the device has emitted (and is holding) a terminal of ``kind``."""
        await self.until(
            lambda: any(isinstance(e, kind) for e in self.audio.pending_events()),
            what,
            limit_ms=10_000,
        )
        return next(e for e in self.audio.pending_events() if isinstance(e, kind))

    async def type_question(self, text: str) -> None:
        """A typed turn (T-0039): the page's ``user.text``, opened from IDLE."""
        assert self.state.state is TurnState.IDLE, (
            f"premise: a typed question opens a turn from IDLE, not from {self.state.state}"
        )
        self.submit(E.UserText(t_ms=self.clock.now_ms, text=text))
        await self.pump()

    async def speaking_with_llm_done(self, turn_id: str) -> None:
        await self.until(
            lambda: (
                self.state.turn_id == turn_id
                and self.state.state is TurnState.SPEAKING
                and self.state.llm_done
            ),
            f"{turn_id} never reached SPEAKING with llm.done",
            limit_ms=10_000,
        )


@asynccontextmanager
async def stale_rig(
    tmp_path: Path, *, script: list[Commit], hold_llm_after: int | None = None
) -> AsyncIterator[StaleRig]:
    """The T-0007 session over the fakes, with a HUD recorder and a real ``SessionLog``.

    The log is read back from its file only after ``aclose`` (``rig.log_records`` is filled then),
    because the writer is asynchronous and a mid-row read races it.
    """
    require_session()
    clock = ScriptedClock()
    machine = RecordingMachine(POLICY)
    llm = DoneGatedLlm(script)
    llm.clock = clock
    if hold_llm_after is not None:
        llm.hold_after(hold_llm_after)
    tts = SelectiveTts(ms_per_word=MS_PER_WORD)
    audio = StaleAudio(clock)
    perception = RecordingPerception([frame(0, "a")])
    published = Recorder()
    log = make_log(tmp_path / "logs")
    turns_path = tmp_path / "turns.jsonl"
    config = SessionConfig(data_dir=tmp_path, turns_path=turns_path)
    session = Session(
        machine,
        llm,
        FakeStt(),
        tts,
        audio,
        perception,
        FakeMemory(),
        clock=clock,
        config=config,
        publish=published,
        session_log=log,
    )
    rig = StaleRig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=session.stt,
        tts=tts,
        audio=audio,
        perception=perception,
        memory=session.memory,
        session=session,
        turns_path=turns_path,
        published=published,
        log=log,
    )
    await session.start()
    try:
        yield rig
    finally:
        await session.aclose()


def stale_notes(log: Any) -> list[dict[str, Any]]:
    """Criterion 3's notes, read back from the session-log file (after ``aclose``)."""
    records = read_records(log.path)
    return [r["data"] for r in of_kind(records, "note") if r["data"].get("what") == STALE_WHAT]


def check_note(
    note: dict[str, Any],
    event: Any,
    *,
    run_id: int,
    run_turn: str | None,
    current_turn: str | None,
) -> None:
    """One note against the event it reports.

    Ruling G2: ``current_turn`` is ``None`` when no turn is open. Ruling G3: ``run_id`` is an int,
    monotonic per session, starting at 1 — so the expected id is counted by hand from the row's own
    script (every ``play()`` with no run open mints one, clips included). Ruling G1: a run minted
    while no turn was open has ``run_turn = None``.
    """
    assert set(note) >= NOTE_FIELDS, (
        f"criterion 3: the note carries {sorted(NOTE_FIELDS)}; it has {sorted(note)}"
    )
    assert note["type"] == event.type, (
        f"criterion 3: the note names the terminal's type {event.type!r}; got {note['type']!r}"
    )
    assert (note["played_ms"], note["played_text"]) == (event.played_ms, event.played_text), (
        "criterion 3: the note carries the dropped event's own played_ms/played_text; "
        f"event {(event.played_ms, event.played_text)}, note {(note['played_ms'], note['played_text'])}"
    )
    assert note["run_turn"] == run_turn, (
        f"criterion 1: the run was started in {run_turn}; the note says {note['run_turn']!r}"
    )
    assert note["current_turn"] == current_turn, (
        f"criterion 3 / G2: the machine's turn when the terminal was handled was {current_turn!r} "
        f"(None: no turn open); the note says {note['current_turn']!r}"
    )
    assert type(note["run_id"]) is int and note["run_id"] == run_id, (
        f"criterion 1 / G3: run ids are ints, monotonic per session from 1; this run is the "
        f"session's run {run_id}, the note says {note['run_id']!r}"
    )


def record_of(rig: StaleRig, turn_id: str) -> Any:
    found = [r for r in rig.records() if r.turn_id == turn_id]
    assert len(found) == 1, (
        f"one turns.jsonl line for {turn_id}; got {[r.turn_id for r in rig.records()]}"
    )
    return found[0]


# --- criterion 4: the reproduction (T15, toolless form) ----------------------------------------


async def _reproduction(tmp_path: Path, first: str, second: str) -> dict[str, Any]:
    """Criterion 4's sequence, literally, with turn 1's drain held across the boundary.

    1. Turn 1 (voice) speaks ``first``, gets ``llm.done`` and has its drain marked; its audio plays
       out and the device emits ``playback.drained`` — which the session does not read yet.
    2. ``tts.error`` ends turn 1 (SPEAKING -> IDLE).
    3. Turn 2 opens (typed), commits ``second``, gets ``llm.done``; its audio is on the device.
    4. Turn 1's drain is delivered.
    """
    out: dict[str, Any] = {}
    async with stale_rig(tmp_path, script=[answer(first)]) as rig:
        await rig.ask("first question")
        await rig.speaking_with_llm_done("t1")
        await rig.until(lambda: "mark_drain" in rig.audio.calls, "turn 1's drain was never marked")
        rig.audio.hold_events()
        stale = await rig.held_terminal(E.PlaybackDrained, "turn 1's run never drained")
        assert stale.played_text == first, (
            f"premise: turn 1's drain reports all of {first!r}; got {stale.played_text!r}"
        )
        assert rig.state.state is TurnState.SPEAKING and rig.state.turn_id == "t1", (
            "premise: turn 1 is still SPEAKING, waiting for a drain it has not been handed"
        )

        rig.submit(E.TtsError(t_ms=rig.clock.now_ms, utterance_id="t1-late", message="tts down"))
        await rig.pump()
        assert rig.state.state is TurnState.IDLE, "premise: tts.error in SPEAKING ends turn 1"

        rig.llm.script = [answer(second)]
        await rig.type_question("second question")
        await rig.speaking_with_llm_done("t2")
        await rig.until(
            lambda: second in rig.audio.answer_sources(),
            "turn 2's answer never reached the device",
            limit_ms=5_000,
        )
        assert rig.audio.played_ms() < MS_PER_WORD * len(second.split()), (
            "premise: turn 2's answer has not finished playing when turn 1's drain is delivered"
        )

        rig.audio.release_events()
        await rig.pump()
        out["state_after_stale"] = rig.state.state
        out["turn_after_stale"] = rig.state.turn_id
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn 2 never ended")
        out["stale"] = stale
        out["delivered"] = list(rig.audio.delivered)
        out["moved"] = [
            ev
            for before, ev, after in rig.machine.transitions()
            if before is TurnState.SPEAKING and after is TurnState.IDLE
        ]
        out["history"] = list(rig.state.history)
        out["seen_stale"] = rig.seen(stale)
        out["played"] = rig.played_messages()
        out["records"] = {r.turn_id: r for r in rig.records()}
        out["flushes"] = rig.flushes()
    out["notes"] = stale_notes(rig.log)
    return out


async def test_c4_a_drain_left_over_from_a_tts_errored_turn_does_not_complete_the_next_turn(
    tmp_path,
):
    """Criterion 4, the reproduction, and the guard-width "too narrow" row.

    Baseline: on ``main`` at 0304f3f this row fails. Turn 1's drain reaches the machine while turn 2
    is SPEAKING with ``llm.done``, so turn 2 completes on it at once — before a word of its own
    answer has played — and turn 2's ``HistoryTurn.assistant_spoken`` is turn 1's
    ``"Turn one says this. "`` (both of criterion 4's requirements broken; measured on 0304f3f
    while writing this row: ``state_after_stale`` was IDLE and history held turn 1's text).
    """
    out = await _reproduction(tmp_path, S1, S2)

    assert out["state_after_stale"] is TurnState.SPEAKING and out["turn_after_stale"] == "t2", (
        "criterion 4: turn 2 does not complete on turn 1's drain; right after it was delivered "
        f"the machine was {out['state_after_stale']} in {out['turn_after_stale']}"
    )
    assert not out["seen_stale"], (
        "criterion 3: turn 1's drain belongs to a run of t1 and is handled while t2 is current, "
        "so it never reaches machine.step"
    )
    own = [
        e for e in out["delivered"] if e is not out["stale"] and isinstance(e, E.PlaybackDrained)
    ]
    assert len(own) == 1, f"premise: turn 2's run drained once; delivered {out['delivered']}"
    t2_completion = out["moved"][-1]
    assert t2_completion is own[0], (
        "criterion 4: turn 2 completes only on its own drain; it completed on "
        f"{t2_completion!r} (its own drain was {own[0]!r})"
    )
    spoken = out["history"][-1].assistant_spoken
    assert spoken == S2, (
        f"criterion 4: turn 2's history is what turn 2 played, {S2!r}; got {spoken!r}"
    )
    assert not any(word in spoken for word in ("Turn", "one", "says", "this")), (
        f"criterion 4: turn 2's history holds no text from turn 1; got {spoken!r}"
    )


async def test_c4_the_stale_drain_is_noted_and_publishes_no_played_report(tmp_path):
    """Criteria 1 and 3 on the reproduction: one note, no ``astra.spoken.played``, no span.

    On 0304f3f the stale drain is published as ``astra.spoken.played`` for **t2** — turn 2's
    ledger split by turn 1's ``played_text`` — and no note exists.
    """
    out = await _reproduction(tmp_path, S1, S2)

    played = out["played"]
    assert [m.turn_id for m in played] == ["t2"], (
        "criterion 3: the dropped drain publishes no astra.spoken.played; only turn 2's own run "
        f"reports, once. Published for turns {[m.turn_id for m in played]}"
    )
    assert "".join(u.played_text for u in played[0].utterances) == S2, (
        f"…and turn 2's report is turn 2's audio; got {[u.played_text for u in played[0].utterances]}"
    )
    notes = out["notes"]
    assert len(notes) == 1, f"criterion 3: one note per dropped terminal; got {notes}"
    check_note(notes[0], out["stale"], run_id=1, run_turn="t1", current_turn="t2")
    own = [e for e in out["delivered"] if e is not out["stale"]][-1]
    assert out["records"]["t2"].spans.get("playback.stopped") == (own.t_ms, own.t_ms), (
        "criterion 3: no span from the stale drain; turn 2's playback span is its own drain's "
        f"({own.t_ms}), got {out['records']['t2'].spans.get('playback.stopped')}"
    )
    assert out["flushes"] == 1, (
        "criterion 2: turn 1's run was still open (no terminal received, no flush issued) when "
        f"tts.error closed the turn, so the session flushes it once; flushes={out['flushes']}"
    )


async def test_c4_adversarial_a_stale_drain_with_the_same_text_is_still_stale(tmp_path):
    """Adversarial (T15's rejected mechanism): staleness by event order, never by ``played_text``.

    Both turns say the same words, so turn 1's leftover drain has exactly the ``played_text`` turn
    2 is about to play. A session that decides staleness (or rescues a stale terminal) by text
    equality accepts it and completes turn 2 before its answer has played. Red on 0304f3f too.
    """
    same = "The same words twice. "
    out = await _reproduction(tmp_path, same, same)

    assert out["state_after_stale"] is TurnState.SPEAKING, (
        "T15: a drain whose played_text happens to equal turn 2's answer is still turn 1's; turn 2 "
        f"must not complete on it (state right after it: {out['state_after_stale']})"
    )
    assert not out["seen_stale"], "T15: the look-alike drain never reaches machine.step"
    own = [
        e for e in out["delivered"] if e is not out["stale"] and isinstance(e, E.PlaybackDrained)
    ]
    assert own and out["moved"][-1] is own[0], (
        f"T15: turn 2 completes on its own drain only; it completed on {out['moved'][-1]!r}"
    )
    assert len(out["notes"]) == 1, f"the look-alike is noted as stale; notes={out['notes']}"


async def test_c4_b_a_run_still_playing_when_tts_error_closes_is_cut_and_not_joined(tmp_path):
    """Criterion 4's other form: turn 1's audio is still *playing* when ``tts.error`` ends it.

    Q-a (2026-10-12): the session cuts a closing turn's still-playing audio. On 0304f3f nothing
    flushes, turn 1's run keeps playing, and turn 2's speech is appended to the same device run —
    "joined there by turn 2's speech" (Goal) — so turn 2's drain reports turn 1's tail plus turn 2
    and that goes into turn 2's history.
    """
    async with stale_rig(tmp_path, script=[answer(LONG)]) as rig:
        await rig.ask("first question")
        await rig.speaking_with_llm_done("t1")
        await rig.until(
            lambda: LONG in rig.audio.answer_sources(), "turn 1 never reached the device"
        )
        await rig.advance(400, step=100)
        assert 0 < rig.audio.played_ms() < MS_PER_WORD * len(LONG.split()), (
            "premise: turn 1 is mid-play at the cut"
        )
        flushes_before = rig.flushes()
        rig.submit(E.TtsError(t_ms=rig.clock.now_ms, utterance_id="t1-late", message="tts down"))
        await rig.pump()
        assert rig.state.state is TurnState.IDLE, "premise: tts.error in SPEAKING ends turn 1"
        assert rig.flushes() == flushes_before + 1, (
            "criterion 2 / Q-a: the closing turn's still-playing run is flushed by the session"
        )

        rig.llm.script = [answer(S2)]
        await rig.type_question("second question")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn 2 never ended")
        spoken = rig.state.history[-1].assistant_spoken
        assert spoken == S2, (
            f"criterion 4: turn 2's history is turn 2's audio only, {S2!r}; got {spoken!r}"
        )
        assert rig.flushes() == flushes_before + 1, "the flush is issued once, not per tick"


# --- criterion 2: a run never outlives its turn, one row per close path ------------------------


async def _close_in_speaking(rig: StaleRig, close: str, *, hold: bool = False) -> None:
    """Reach SPEAKING with turn 1's answer mid-play, then close the turn by ``close``.

    ``hold=True`` holds terminal delivery *before* the close, so whatever the close produces is
    still undelivered afterwards.
    """
    await rig.ask("first question")
    await rig.until(
        lambda: LONG in rig.audio.answer_sources() and rig.state.state is TurnState.SPEAKING,
        "turn 1 never reached the device",
        limit_ms=5_000,
    )
    await rig.advance(400, step=100)
    assert rig.state.state is TurnState.SPEAKING
    assert 0 < rig.audio.played_ms() < MS_PER_WORD * len(LONG.split()), "premise: mid-play"
    if hold:
        rig.audio.hold_events()
    now = rig.clock.now_ms
    if close == "tts.error":
        rig.submit(E.TtsError(t_ms=now, utterance_id="t1-late", message="tts down"))
    elif close == "llm.error":
        rig.submit(E.LlmError(t_ms=now, request_id=rig.state.active_generate, message="boom"))
    elif close == "client.disconnected":
        rig.submit(E.ClientDisconnected(t_ms=now))
    else:  # pragma: no cover
        raise AssertionError(close)
    await rig.pump()


@pytest.mark.parametrize("close", ["tts.error", "llm.error", "client.disconnected"])
async def test_c2_a_turn_closed_in_speaking_ends_its_run_and_drops_the_late_stop(tmp_path, close):
    """Criterion 2, rows 1-3: tts.error / llm.error / client.disconnected in SPEAKING.

    None of the three machine rows emits ``playback.flush`` (``turn_machine.py``: ``_on_tts_error``,
    ``_fail_turn``, ``_on_client_disconnected``), so on 0304f3f the run outlives its turn: no flush,
    the audio plays on into IDLE. Required: the session flushes once, after the closing tuple; the
    ``playback.stopped`` that flush produces belongs to t1's run and arrives with no turn open, so it
    is dropped with a note and reaches neither the machine nor the HUD.

    ``llm.error`` is sent while the generate is still open (``hold_after(1)``), so SPEAKING has no
    ``llm.done`` and no drain is marked; the other two rows close after ``llm.done``.
    """
    hold = 1 if close == "llm.error" else None
    script = [answer(LONG), answer("And a tail. ")] if hold else [answer(LONG)]
    async with stale_rig(tmp_path, script=script, hold_llm_after=hold) as rig:
        await _close_in_speaking(rig, close)
        assert rig.state.state is TurnState.IDLE, f"premise: {close} in SPEAKING ends turn 1"
        await rig.advance(3_000, step=100)  # long past where the uncut audio would have ended

        assert rig.flushes() == 1, (
            f"criterion 2: a turn closed by {close} with its run open has that run flushed by the "
            f"session, exactly once; the device saw {rig.flushes()} flushes ({rig.audio.calls})"
        )
        stops = [e for e in rig.audio.delivered if isinstance(e, E.PlaybackStopped)]
        assert len(stops) == 1, f"premise: the flush produced one playback.stopped; {stops}"
        assert not rig.terminals_seen(), (
            "criterion 3: the run's terminal arrives after its turn closed; it never reaches "
            f"machine.step (the machine saw {rig.terminals_seen()})"
        )
        assert rig.played_messages() == [], (
            "criterion 3: a dropped terminal publishes no astra.spoken.played; got "
            f"{rig.played_messages()}"
        )
        record = record_of(rig, "t1")
        assert "playback.stopped" not in record.spans, (
            f"criterion 3: no span from the dropped terminal; t1's spans {sorted(record.spans)}"
        )
    notes = stale_notes(rig.log)
    assert len(notes) == 1, f"criterion 3: one note for the dropped terminal; got {notes}"
    check_note(notes[0], stops[0], run_id=1, run_turn="t1", current_turn=None)


async def test_c2_a_thinking_turn_that_ends_with_only_the_ack_clip_playing_ends_that_run(tmp_path):
    """Criterion 2, row 4: a run holding only the ack clip, still open when a THINKING turn ends.

    Thinking is on, so THINKING plays the ``ack`` clip (``play(…, source="")`` — clips included in
    criterion 1's runs). The engine never answers; ``llm.error`` in THINKING fails the turn to IDLE
    with no flush in its tuple. On 0304f3f the clip's run stays open into the next turn.
    """
    async with stale_rig(tmp_path, script=[answer(S2)]) as rig:
        rig.llm.hold()
        await rig.ask("first question")
        await rig.until(
            lambda: rig.state.state is TurnState.THINKING and rig.audio.clip_plays(),
            "THINKING never played its ack clip",
        )
        assert rig.flushes() == 0
        rig.submit(
            E.LlmError(t_ms=rig.clock.now_ms, request_id=rig.state.active_generate, message="boom")
        )
        await rig.pump()
        assert rig.state.state is TurnState.IDLE, "premise: llm.error in THINKING ends the turn"
        await rig.advance(500, step=100)

        assert rig.flushes() == 1, (
            "criterion 2: the clip-only run was open when its THINKING turn closed; the session "
            f"ends it with one flush (device calls {rig.audio.calls})"
        )
        stops = [e for e in rig.audio.delivered if isinstance(e, E.PlaybackStopped)]
        assert len(stops) == 1 and not rig.terminals_seen(), (
            "criterion 3: its playback.stopped is stale (no turn open) and never reaches the "
            f"machine; delivered {rig.audio.delivered}, machine saw {rig.terminals_seen()}"
        )
    notes = stale_notes(rig.log)
    assert len(notes) == 1, f"criterion 3: one note; got {notes}"
    check_note(notes[0], stops[0], run_id=1, run_turn="t1", current_turn=None)


async def test_c2_interrupt_timeout_then_a_late_stop_is_dropped_and_not_flushed_twice(tmp_path):
    """Criterion 2, row 5: the INTERRUPTED timeout path.

    The machine flushes on the barge-in; the device's ``playback.stopped`` is late, the 300 ms
    ``interrupt_timeout`` fires first and opens t2 (INTERRUPTED -> LISTENING on a Tick). The flush
    was already issued for t1's run, so the session issues no second one; when the stop finally
    arrives, t2 is current and it is dropped. On 0304f3f the late stop reaches the machine in t2's
    LISTENING and sets t2's ``played_text`` to t1's words.
    """
    async with stale_rig(tmp_path, script=[answer(LONG)]) as rig:
        await rig.ask("first question")
        await rig.until(
            lambda: LONG in rig.audio.answer_sources() and rig.state.state is TurnState.SPEAKING,
            "turn 1 never reached the device",
            limit_ms=5_000,
        )
        await rig.advance(600, step=100)
        rig.audio.hold_events()
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        assert rig.state.state is TurnState.INTERRUPTED, "premise: the takeover interrupts"
        stale = await rig.held_terminal(E.PlaybackStopped, "the machine's flush emitted no stop")
        await rig.advance(POLICY.interrupt_timeout_ms + 100, step=50)
        relisten = [
            ev
            for before, ev, after in rig.machine.transitions()
            if before is TurnState.INTERRUPTED and after is TurnState.LISTENING
        ]
        assert relisten and isinstance(relisten[0], E.Tick), (
            f"premise: the timeout, not the stop, left INTERRUPTED; {relisten}"
        )
        assert rig.state.turn_id == "t2"

        rig.audio.release_events()
        await rig.advance(200, step=50)
        assert rig.flushes() == 1, (
            "criterion 2: t1's run already had its PlaybackFlush; the session must not flush it "
            f"again when t1 closes (device calls {rig.audio.calls})"
        )
        assert not rig.seen(stale), (
            "criterion 3: the stop arrives while t2 is current; it is t1's run's and never "
            "reaches machine.step"
        )
        assert rig.state.played_text == "", (
            f"…so t2 carries none of t1's words; t2's played_text is {rig.state.played_text!r}"
        )
        assert rig.played_messages() == []
    notes = stale_notes(rig.log)
    assert len(notes) == 1, f"criterion 3: one note; got {notes}"
    check_note(notes[0], stale, run_id=1, run_turn="t1", current_turn="t2")


# --- criterion 3: no span, no played report; criterion 1: run ids -----------------------------


async def test_c3_stale_terminals_leave_no_span_on_the_turn_they_arrive_in(tmp_path):
    """Criterion 3's "no span on any turn record", and criterion 1's per-session run ids.

    Turn 1's drain is held across ``tts.error``; turn 2 (typed) is THINKING with only its ack clip
    playing when that drain is delivered; then ``llm.error`` ends turn 2 with the clip run open, so
    the session flushes it and *its* stop is stale too. Turn 2 has no terminal of its own reaching
    the machine, so any ``playback.stopped`` span on its record came from a stale terminal. On
    0304f3f turn 1's drain lands in turn 2's THINKING and writes that span.
    """
    async with stale_rig(tmp_path, script=[answer(S1)]) as rig:
        await rig.ask("first question")
        await rig.speaking_with_llm_done("t1")
        await rig.until(lambda: "mark_drain" in rig.audio.calls, "turn 1's drain was never marked")
        rig.audio.hold_events()
        drained = await rig.held_terminal(E.PlaybackDrained, "turn 1's run never drained")
        rig.submit(E.TtsError(t_ms=rig.clock.now_ms, utterance_id="t1-late", message="tts down"))
        await rig.pump()

        rig.llm.hold()
        await rig.type_question("second question")
        await rig.until(
            lambda: (
                rig.state.turn_id == "t2"
                and rig.state.state is TurnState.THINKING
                and len(rig.audio.clip_plays()) == 2
            ),
            "turn 2 never reached THINKING with its ack clip",
        )
        rig.audio.release_events()
        await rig.advance(100, step=50)
        assert rig.state.state is TurnState.THINKING and not rig.seen(drained), (
            "criterion 3: turn 1's drain, delivered during turn 2's THINKING, is dropped"
        )
        rig.submit(
            E.LlmError(t_ms=rig.clock.now_ms, request_id=rig.state.active_generate, message="boom")
        )
        await rig.advance(300, step=50)
        assert rig.state.state is TurnState.IDLE
        stops = [e for e in rig.audio.delivered if isinstance(e, E.PlaybackStopped)]
        assert len(stops) == 1, (
            "criterion 2: turn 2's clip run was open when llm.error closed turn 2, so it is "
            f"flushed; delivered {rig.audio.delivered}"
        )
        assert not rig.terminals_seen(), (
            f"neither terminal is the machine's: {rig.terminals_seen()}"
        )
        assert rig.played_messages() == []
        for turn_id in ("t1", "t2"):
            spans = record_of(rig, turn_id).spans
            assert "playback.stopped" not in spans, (
                f"criterion 3: no span from a dropped terminal on any turn record; {turn_id} has "
                f"playback.stopped={spans.get('playback.stopped')}"
            )
    notes = stale_notes(rig.log)
    assert len(notes) == 2, f"criterion 3: one note per dropped terminal; got {notes}"
    check_note(notes[0], drained, run_id=1, run_turn="t1", current_turn="t2")
    check_note(notes[1], stops[0], run_id=2, run_turn="t2", current_turn=None)
    first, second = notes[0]["run_id"], notes[1]["run_id"]
    assert first != second, (
        f"criterion 1: each run has its own id (clips included); both notes say {first!r}"
    )
    assert first < second, (
        f"criterion 1: run ids are monotonic per session; t1's run {first!r}, t2's {second!r}"
    )


# --- two runs in flight -----------------------------------------------------------------------


async def test_two_runs_in_flight_the_new_runs_terminal_still_reaches_the_machine(tmp_path):
    """Tests to write, "two runs in flight": an old run's terminal is outstanding when the new
    turn's run starts.

    Turn 1 is cut by ``tts.error`` mid-play; the session's close flush emits t1's stop, held. Turn
    2's answer starts a new run and is the turn's *last* play; turn 2's ``llm.done`` is held back
    (``DoneGatedLlm``) so the old stop is delivered while turn 2 is SPEAKING without ``llm.done``,
    with no later ``play()`` to reopen anything. If the stale stop closed the new run
    (``_playback_open = False`` on every terminal, the Implementation notes' "part of the gap"), the
    drain gate would then synthesise an empty ``playback.drained`` at ``llm.done`` and turn 2 would
    complete with nothing in its history. Required: ``_playback_open`` stays true until the new
    run's own terminal — observed as the device being asked to ``mark_drain()`` turn 2's run, and
    turn 2 completing on that run's own drain with what it played.
    """
    async with stale_rig(tmp_path, script=[answer(LONG)]) as rig:
        await _close_in_speaking(rig, "tts.error", hold=True)
        assert rig.state.state is TurnState.IDLE

        rig.llm.script = [answer(S2)]
        rig.llm.gate_done()
        await rig.type_question("second question")
        await rig.until(
            lambda: S2 in rig.audio.answer_sources() and rig.state.state is TurnState.SPEAKING,
            "turn 2's answer never reached the device",
            limit_ms=5_000,
        )
        assert not rig.state.llm_done, "premise: turn 2's llm.done is held back"
        pending = rig.audio.pending_events()
        assert [type(e) for e in pending] == [E.PlaybackStopped], (
            "criterion 2 (and this row's premise): t1's run was flushed at its close and that "
            f"stop is still outstanding while t2's run plays; undelivered terminals are {pending}"
        )
        marks_before = rig.audio.calls.count("mark_drain")
        plays_before = rig.audio.calls.count("play")
        rig.audio.release_events()
        await rig.pump()
        assert not rig.terminals_seen(), (
            f"criterion 3: the old run's stop is stale and dropped; machine saw {rig.terminals_seen()}"
        )
        rig.llm.open_done()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn 2 never ended")

        assert rig.audio.calls.count("play") == plays_before, (
            "premise: nothing was played after the stale stop, so nothing could reopen a run"
        )
        assert rig.audio.calls.count("mark_drain") == marks_before + 1, (
            "two runs in flight: turn 2's run is still open after the old run's terminal, so the "
            "drain gate asks the device to mark it (no mark means the stale terminal closed the "
            f"new run and the gate synthesised an empty drain); device calls {rig.audio.calls}"
        )
        own = [e for e in rig.audio.delivered if isinstance(e, E.PlaybackDrained)]
        assert len(own) == 1 and rig.seen(own[0]), (
            f"the new run's terminal reaches the machine; delivered {rig.audio.delivered}"
        )
        completion = rig.machine.moved(TurnState.SPEAKING, TurnState.IDLE)[-1]
        assert completion is own[0], (
            f"turn 2 completes on its own run's drain, not on {completion!r}"
        )
        assert rig.state.history[-1].assistant_spoken == S2, (
            f"…with what it played; history {rig.state.history[-1].assistant_spoken!r}"
        )


async def test_two_terminals_delivered_back_to_back_are_attributed_in_arrival_order(tmp_path):
    """Criterion 1's attribution: oldest outstanding run first, in arrival order.

    t1's stop (from the close flush) and t2's own drain are both held and delivered together.
    The first belongs to t1's run and is dropped; the second is t2's and completes t2.
    """
    async with stale_rig(tmp_path, script=[answer(LONG)]) as rig:
        await _close_in_speaking(rig, "tts.error", hold=True)
        rig.llm.script = [answer(S2)]
        await rig.type_question("second question")
        await rig.speaking_with_llm_done("t2")
        held = await rig.held_terminal(E.PlaybackDrained, "turn 2's run never drained")
        pending = rig.audio.pending_events()
        assert [type(e) for e in pending] == [E.PlaybackStopped, E.PlaybackDrained], (
            "criterion 2 (and this row's premise): t1's run was flushed at its close, so t1's stop "
            f"and then t2's own drain are both waiting, separately; undelivered: {pending}"
        )
        assert rig.state.state is TurnState.SPEAKING
        rig.audio.release_events()
        await rig.pump()

        assert not rig.seen(pending[0]) and rig.seen(held), (
            "criterion 1: the first terminal is t1's run's (dropped), the second t2's (delivered); "
            f"machine saw {rig.terminals_seen()}"
        )
        assert rig.state.state is TurnState.IDLE and rig.state.history[-1].assistant_spoken == S2
    notes = stale_notes(rig.log)
    assert len(notes) == 1
    check_note(notes[0], pending[0], run_id=1, run_turn="t1", current_turn="t2")


# --- guard width: too wide ---------------------------------------------------------------------


async def test_too_wide_a_same_turn_drain_after_a_barge_in_resume_reaches_the_machine(tmp_path):
    """Guard width, too wide (1): pause -> backchannel resume -> drain, all in one turn.

    Green on 0304f3f by construction; it fails for a guard that treats a pause, a resume or the
    time spent PAUSED as a turn boundary. The drain is the one the device emitted for t1's run,
    handled while t1 is current, so it reaches the machine byte for byte (criterion 6).
    """
    async with stale_rig(tmp_path, script=[answer(S1)]) as rig:
        await rig.ask("first question")
        await rig.speaking_with_llm_done("t1")
        await rig.advance(200, step=100)
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.pump()
        assert rig.state.state is TurnState.PAUSED, "premise: onset in SPEAKING pauses"
        await rig.advance(200, step=100)
        rig.submit(E.UserBackchannel(t_ms=rig.clock.now_ms, duration_ms=150))
        await rig.pump()
        assert rig.state.state is TurnState.SPEAKING, "premise: the backchannel resumes"
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the resumed turn never ended")

        drains = [e for e in rig.audio.delivered if isinstance(e, E.PlaybackDrained)]
        assert len(drains) == 1 and rig.seen(drains[0]), (
            f"the same-turn drain reaches the machine; delivered {drains}, saw {rig.terminals_seen()}"
        )
        assert rig.machine.moved(TurnState.SPEAKING, TurnState.IDLE)[-1] is drains[0]
        assert rig.state.history[-1].assistant_spoken == S1
        assert [m.turn_id for m in rig.played_messages()] == ["t1"]
        assert rig.flushes() == 0, "an ordinary drained turn is never flushed"
    assert stale_notes(rig.log) == [], "nothing was stale"


async def test_too_wide_an_interrupt_whose_stop_beats_the_timeout_reaches_the_machine(tmp_path):
    """Guard width, too wide (2): ``playback.stopped`` late, but inside the 300 ms timeout.

    Held for 200 ms after the barge-in and then delivered while t1 is still INTERRUPTED, it is t1's
    run's terminal handled in t1, so it reaches the machine and *it* — not the timeout — moves
    INTERRUPTED -> LISTENING with the played prefix. Green on 0304f3f by construction.
    """
    async with stale_rig(tmp_path, script=[answer(LONG)]) as rig:
        await rig.ask("first question")
        await rig.until(
            lambda: LONG in rig.audio.answer_sources() and rig.state.state is TurnState.SPEAKING,
            "turn 1 never reached the device",
            limit_ms=5_000,
        )
        await rig.advance(600, step=100)
        rig.audio.hold_events()
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        stop = await rig.held_terminal(E.PlaybackStopped, "the machine's flush emitted no stop")
        await rig.advance(200, step=50)
        assert rig.state.state is TurnState.INTERRUPTED, "premise: still inside the timeout"
        released_at = rig.clock.now_ms
        rig.audio.release_events()
        await rig.pump()

        # T-0173 R1: compared by fields, not identity -- the session may restamp a terminal on the session clock
        # (P10), so the stepped object can be a copy. Its t_ms is the device's stamp (before T-0173) or the receipt
        # time (after); the T-0173 rows pin which.
        def same_stop(ev: Any) -> bool:
            return (
                isinstance(ev, E.PlaybackStopped)
                and (ev.type, ev.played_ms, ev.played_text)
                == (stop.type, stop.played_ms, stop.played_text)
                and ev.t_ms in (stop.t_ms, released_at)
            )

        stepped = [ev for ev in rig.machine.events() if isinstance(ev, E.PlaybackStopped)]
        assert len(stepped) == 1 and same_stop(stepped[0]), (
            f"the in-time stop reaches the machine (released at {released_at}, held {stop}); stepped {stepped}"
        )
        relisten = rig.machine.moved(TurnState.INTERRUPTED, TurnState.LISTENING)
        assert len(relisten) == 1 and same_stop(relisten[0]), (
            f"…and it, not the timeout, leaves INTERRUPTED; {relisten}"
        )
        truncates = rig.machine.commands(C.TruncateHistoryToPlayed)
        assert truncates and truncates[-1].played_text == stop.played_text != "", (
            f"history keeps the device's played prefix {stop.played_text!r}; got {truncates}"
        )
        assert rig.flushes() == 1
    assert stale_notes(rig.log) == []


# --- criterion 5: the drain gate stays ----------------------------------------------------------


async def test_c5_the_drain_gate_still_waits_for_speaking_and_llm_done(tmp_path):
    """Criterion 5: ``_maybe_mark_drain`` marks only in SPEAKING with ``llm_done`` (T-0089 note).

    Regression guard, green on 0304f3f. Turn 1's first sentence plays out completely while the
    generate is held open: no ``mark_drain()``, no drain, still SPEAKING. Only ``llm.done`` lets
    the drain be marked and the turn end. A gate widened to "speech queue empty" fails here; one
    removed fails ``test_session_audio_seam.py`` (mark_drain after the last play).
    """
    async with stale_rig(tmp_path, script=[answer(S1), answer(S2)], hold_llm_after=1) as rig:
        await rig.ask("first question")
        await rig.until(lambda: S1 in rig.audio.answer_sources(), "S1 never reached the device")
        await rig.advance(MS_PER_WORD * len(S1.split()) + 400, step=100)
        assert rig.state.state is TurnState.SPEAKING and not rig.state.llm_done
        assert "mark_drain" not in rig.audio.calls, (
            "criterion 5: no drain is marked before llm.done, however long the queue is empty"
        )
        assert not rig.terminals_seen(), "…and none is synthesised"
        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        assert "mark_drain" in rig.audio.calls
        assert rig.state.history[-1].assistant_spoken == S1 + S2


# --- ruling G1: a run minted with no turn open ---------------------------------------------------


async def _idle_clip_with_held_stop(rig: StaleRig) -> Any:
    """In IDLE, play the ack clip (a run with ``run_turn = None``) and end it while still IDLE;
    its ``playback.stopped`` is emitted and held. The device's own flush stands in for whoever
    ends an idle clip, as ``test_spoken_played_egress.py``'s clip-only row does."""
    assert rig.state.state is TurnState.IDLE
    rig.audio.hold_events()
    await rig.session.play_clip("ack")
    await rig.pump()
    assert rig.audio.clip_plays(), "premise: the idle clip reached the device"
    await rig.audio.flush()
    return await rig.held_terminal(E.PlaybackStopped, "the idle clip's run emitted no stop")


async def test_g1_an_idle_clips_terminal_handled_after_a_turn_opened_is_stale(tmp_path):
    """Ruling G1, the in-between row: an IDLE clip, then a turn opens, then its terminal arrives.

    The run was minted with no turn open (``run_turn = None``); by the time its stop is handled, t1
    is LISTENING, so the stop is not t1's and is dropped, noted with ``current_turn = "t1"``. It is
    the session's first run (G3: ``run_id == 1``). On e2fa048 (0304f3f's session) the stop reaches
    the machine in t1's LISTENING.
    """
    async with stale_rig(tmp_path, script=[answer(S2)]) as rig:
        stop = await _idle_clip_with_held_stop(rig)
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.pump()
        assert rig.state.state is TurnState.LISTENING and rig.state.turn_id == "t1", (
            "premise: a turn opened before the idle clip's terminal was delivered"
        )
        steps = len(rig.machine.log)
        rig.audio.release_events()
        await rig.pump()
        assert not rig.seen(stop), (
            "G1: a run minted with no turn open whose terminal is handled after a turn opened is "
            "stale; it never reaches machine.step"
        )
        assert not any(isinstance(ev, TERMINALS) for ev, _, _ in rig.machine.log[steps:])
        assert rig.state.state is TurnState.LISTENING and rig.state.played_text == ""
        assert rig.played_messages() == []
    notes = stale_notes(rig.log)
    assert len(notes) == 1, f"criterion 3: one note for the dropped terminal; got {notes}"
    check_note(notes[0], stop, run_id=1, run_turn=None, current_turn="t1")


async def test_g1_an_idle_clips_terminal_handled_while_still_idle_is_same_turn(tmp_path):
    """Ruling G1, the other half (guard width, too wide): handled while still no turn is open, the
    idle clip's terminal is same-turn and reaches the machine as today. Green on e2fa048 by
    construction; it fails for a guard that calls every terminal handled in IDLE stale."""
    async with stale_rig(tmp_path, script=[answer(S2)]) as rig:
        stop = await _idle_clip_with_held_stop(rig)
        rig.audio.release_events()
        await rig.pump()
        assert rig.state.state is TurnState.IDLE
        assert rig.seen(stop), "G1: no turn opened in between, so the terminal is delivered"
    assert stale_notes(rig.log) == [], "nothing was stale"


# --- rulings G6-G8 (after review at 8ad8499) ----------------------------------------------------

#: Reviewer 1's F1/F2 fixture: two answer sentences, the second still in ``tts.speak`` at the close.
F_A = "First sentence here. "
F_B = "Queued tail of turn one. "
#: Words of ``F_B`` that are in neither ``F_A`` nor ``S2``: their presence anywhere is turn 1's tail.
F_B_ONLY = ("Queued", "tail", "turn", "one")
STALE_UTTERANCE = "stale utterance"
UTTERANCE_NOTE_FIELDS = {"what", "turn", "current_turn", "source"}
#: The virtual horizon a toolless typed turn must finish inside (its own answer is 600 ms).
HORIZON_MS = 20_000


def utterance_notes(log: Any) -> list[dict[str, Any]]:
    records = read_records(log.path)
    return [r["data"] for r in of_kind(records, "note") if r["data"].get("what") == STALE_UTTERANCE]


async def _close_with_tail_in_tts(rig: StaleRig, close: str) -> None:
    """Turn 1 speaks ``F_A``; ``F_B`` is committed and held inside ``tts.speak``; ``close`` ends t1."""
    await rig.ask("first question")
    await rig.until(
        lambda: F_A in rig.audio.answer_sources() and rig.state.state is TurnState.SPEAKING,
        "turn 1's first sentence never reached the device",
        limit_ms=5_000,
    )
    rig.tts.hold()
    rig.llm.release()
    await rig.advance(500)
    assert F_B.strip() in rig.tts.texts(), (
        f"premise: turn 1's second sentence is inside tts.speak at the close; {rig.tts.texts()}"
    )
    assert F_B not in rig.audio.answer_sources(), "premise: …and has not been played"
    now = rig.clock.now_ms
    if close == "llm.error":
        rig.submit(E.LlmError(t_ms=now, request_id=None, message="boom"))
    else:
        rig.submit(E.TtsError(t_ms=now, utterance_id="t1-late", message="tts down"))
    await rig.pump()
    assert rig.state.state is TurnState.IDLE, f"premise: {close} in SPEAKING ends turn 1"


#: Why a turn might not end, per ruling (the failure message of ``_turn_two_ends_on_its_own_drain``).
WHY_G6 = (
    "G6/F1: a closed turn's queued utterance played after the close and its run swallowed the "
    "turn's drain"
)
WHY_F3 = "G8/F3: a failed play left a minted run outstanding, and it swallowed the turn's drain"
WHY_G7 = "G7: the turn joined the idle clip's run (turn None), and its drain was dropped as stale"


async def _turn_two_ends_on_its_own_drain(rig: StaleRig, why: str = WHY_G6) -> None:
    waited = 0
    while rig.state.state is not TurnState.IDLE and waited < HORIZON_MS:
        await rig.advance(500, step=50)
        waited += 500
    assert rig.state.state is TurnState.IDLE, (
        f"the turn must complete within {HORIZON_MS} ms of virtual time; it is still "
        f"{rig.state.state} in {rig.state.turn_id} (llm_done={rig.state.llm_done}) — {why}"
    )
    own = [e for e in rig.audio.delivered if isinstance(e, E.PlaybackDrained)]
    completion = rig.machine.moved(TurnState.SPEAKING, TurnState.IDLE)[-1]
    assert own and completion is own[-1], (
        f"turn 2 completes on its own run's drain; it completed on {completion!r}"
    )


def check_utterance_note(note: dict[str, Any], *, current_turn: str | None) -> None:
    assert set(note) >= UTTERANCE_NOTE_FIELDS, (
        f"G6: the note carries {sorted(UTTERANCE_NOTE_FIELDS)}; it has {sorted(note)}"
    )
    assert (note["turn"], note["current_turn"], note["source"]) == ("t1", current_turn, F_B), (
        f"G6: the refused utterance is t1's {F_B!r}, refused while {current_turn!r} was current; "
        f"the note says {(note['turn'], note['current_turn'], note['source'])}"
    )


@pytest.mark.parametrize("close", ["llm.error", "tts.error"])
async def test_g8_f1_a_closed_turns_queued_utterance_never_hangs_the_next_turn(tmp_path, close):
    """G8 F1 (reviewer 1's hang), per error type.

    Turn 1's second sentence is inside ``tts.speak`` when the error closes turn 1 (no ``TtsCancel``
    in either row). TTS is released while IDLE, then typed turn 2 runs. At 8ad8499 the worker
    plays t1's tail after the close flush and mints a run for it; turn 2's audio joins that run,
    its drain is attributed to it and dropped as stale, and turn 2 sits in SPEAKING for good.
    Required (G6): the tail is not played, one ``stale utterance`` note says so, and turn 2
    completes on its own drain with its own words.
    """
    script = [answer(F_A), answer(F_B, canvas_idx=1)]
    async with stale_rig(tmp_path, script=script, hold_llm_after=1) as rig:
        await _close_with_tail_in_tts(rig, close)
        rig.tts.release()
        await rig.advance(3_000, step=100)
        rig.llm.script = [answer(S2)]
        await rig.type_question("second question")
        await _turn_two_ends_on_its_own_drain(rig)
        assert rig.state.history[-1].assistant_spoken == S2, rig.state.history[-1]
        assert F_B not in rig.audio.answer_sources(), (
            f"G6: a closed turn's speech is never played after it closes; the device played "
            f"{rig.audio.answer_sources()}"
        )
    notes = utterance_notes(rig.log)
    assert len(notes) == 1, f"G6: one note for the refused utterance; got {notes}"
    check_utterance_note(notes[0], current_turn=None)


async def test_g8_f2_a_closed_turns_tail_never_reaches_the_next_turns_history(tmp_path):
    """G8 F2 (reviewer 1's history leak): turn 2 is typed *before* turn 1's held tail is released.

    At 8ad8499 the tail plays during turn 2, is minted as a turn-2 run (or joins it) and turn 1's
    words go into turn 2's ``HistoryTurn.assistant_spoken``. Required (G6): the tail is refused
    while t2 is current, noted, and turn 2's history holds only its own answer.
    """
    script = [answer(F_A), answer(F_B, canvas_idx=1)]
    async with stale_rig(tmp_path, script=script, hold_llm_after=1) as rig:
        await _close_with_tail_in_tts(rig, "llm.error")
        rig.llm.script = [answer(S2)]
        await rig.type_question("second question")
        assert rig.state.turn_id == "t2" and rig.state.state is not TurnState.IDLE
        rig.tts.release()
        await _turn_two_ends_on_its_own_drain(rig)
        spoken = rig.state.history[-1].assistant_spoken
        assert not any(word in spoken for word in F_B_ONLY), (
            f"G6/F2: turn 2's history holds no turn 1 text; got {spoken!r}"
        )
        assert spoken == S2, f"…and holds turn 2's own answer, {S2!r}; got {spoken!r}"
        assert F_B not in rig.audio.answer_sources(), "G6: the closed turn's tail is never played"
    notes = utterance_notes(rig.log)
    assert len(notes) == 1, f"G6: one note for the refused utterance; got {notes}"
    check_utterance_note(notes[0], current_turn="t2")


async def test_g8_f3_a_failed_play_rolls_its_minted_run_back(tmp_path):
    """G8 F3: a play the device refused started no run, so no terminal will ever come for it.

    The ack clip of turn 1 fails (``fail_play``): the session reports ``client.disconnected`` and
    turn 1 ends. If the run minted for that play stayed outstanding, it would be the oldest run
    and turn 2's own drain would be attributed to it and dropped — turn 2 would never end.
    Green at 8ad8499; it kills the mutant that makes the rollback (``_forget_run``) a no-op.
    """
    async with stale_rig(tmp_path, script=[answer(S1)]) as rig:
        rig.audio.fail_play = True
        await rig.ask("first question")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE, "turn 1 never ended on the dead device"
        )
        assert rig.audio.clip_plays() == [] and rig.machine.events(E.ClientDisconnected), (
            "premise: the ack clip's play failed and became client.disconnected"
        )
        rig.audio.fail_play = False
        rig.llm.script = [answer(S2)]
        await rig.type_question("second question")
        await _turn_two_ends_on_its_own_drain(rig, WHY_F3)
        assert rig.state.history[-1].assistant_spoken == S2
    assert stale_notes(rig.log) == [], (
        f"no terminal is stale here: the failed play never started a run; {stale_notes(rig.log)}"
    )


async def test_g8_g7_a_turn_never_joins_an_idle_clips_open_run(tmp_path):
    """G8/G7: an IDLE clip's run (``run_turn = None``) is still open when a turn starts playing.

    The turn's first play must not join it: the session flushes the old run, then mints run 2 for
    the turn. The old run's stop is handled while t1 is current, so it is stale and noted
    (criterion 3, G1); the turn completes on its own drain with its own words. At 8ad8499 the
    turn's audio joins run 1, its drain is attributed to run 1 (turn None), dropped as stale, and
    the turn never ends.
    """
    async with stale_rig(tmp_path, script=[answer(S2)]) as rig:
        await rig.session.play_clip("ack")
        await rig.pump()
        assert rig.state.state is TurnState.IDLE and len(rig.audio.clip_plays()) == 1
        assert rig.flushes() == 0, "premise: the idle clip's run is open, not ended"
        await rig.type_question("second question")
        await _turn_two_ends_on_its_own_drain(rig, WHY_G7)
        assert rig.flushes() == 1, (
            f"G7: the open idle run is flushed once before the turn plays; calls {rig.audio.calls}"
        )
        first_turn_play = rig.audio.calls.index("play", rig.audio.calls.index("play") + 1)
        assert rig.audio.calls.index("flush") < first_turn_play, (
            f"G7: the flush comes before the turn's first play; calls {rig.audio.calls}"
        )
        stops = [e for e in rig.audio.delivered if isinstance(e, E.PlaybackStopped)]
        assert len(stops) == 1 and not rig.seen(stops[0]), (
            f"G7: the idle run's stop is stale and dropped; delivered {rig.audio.delivered}"
        )
        assert rig.state.history[-1].assistant_spoken == S2
    notes = stale_notes(rig.log)
    assert len(notes) == 1, f"criterion 3: one note for the idle run's stop; got {notes}"
    check_note(notes[0], stops[0], run_id=1, run_turn=None, current_turn="t1")


# --- R2 (review of cb0b41e): a closed turn's synthesis failing is not the current turn's error ---

F_C = "A third sentence also queued. "


async def test_r2_a_closed_turns_synthesis_failing_is_not_the_current_turns_error(tmp_path):
    """R2: t1's ``F_B`` is inside ``tts.speak`` when llm.error closes t1; typed t2 reaches SPEAKING
    with llm.done; only then does ``F_B``'s synthesis raise.

    That failure belongs to a closed turn (G6), so it must not become a ``tts.error`` on the
    machine: that would degrade the session and end t2 through the error row (with ``''`` in its
    history) instead of on its own drain. Required: no ``TtsError`` reaches ``machine.step``, t2
    completes on its own drain with ``S2`` and ``degraded`` False, and one ``stale utterance``
    note records the refusal.
    """
    script = [answer(F_A), answer(F_B, canvas_idx=1)]
    async with stale_rig(tmp_path, script=script, hold_llm_after=1) as rig:
        await _close_with_tail_in_tts(rig, "llm.error")
        rig.tts.fail_on = {F_B.strip()}
        rig.llm.script = [answer(S2)]
        await rig.type_question("second question")
        await rig.speaking_with_llm_done("t2")
        rig.tts.release()
        await _turn_two_ends_on_its_own_drain(
            rig, "R2: the closed turn's synthesis failure was taken as the current turn's error"
        )
        assert not rig.machine.events(E.TtsError), (
            "R2/G6: a closed turn's synthesis failure never reaches machine.step as tts.error; "
            f"the machine saw {rig.machine.events(E.TtsError)}"
        )
        assert rig.state.history[-1].assistant_spoken == S2, (
            f"R2: t2 keeps its own answer; history {rig.state.history[-1].assistant_spoken!r}"
        )
        assert rig.state.degraded is False, "R2: the session is not degraded by a stale failure"
        assert F_B.strip() in rig.tts.texts(), "premise: F_B was synthesised, and that raised"
    notes = utterance_notes(rig.log)
    assert len(notes) == 1, f"G6: one note for the refused utterance; got {notes}"
    check_utterance_note(notes[0], current_turn="t2")


async def test_g6_a_stale_utterance_still_queued_is_never_synthesised(tmp_path):
    """G6, pre-synthesis: t1's ``F_C`` is still *queued* (behind ``F_B``, held in ``tts.speak``)
    when llm.error closes t1. It is refused before ``tts.speak`` is ever called with it — a
    closed turn's speech costs no synthesis — and noted, after ``F_B``'s own note."""
    script = [answer(F_A), answer(F_B, canvas_idx=1), answer(F_C, canvas_idx=2)]
    async with stale_rig(tmp_path, script=script, hold_llm_after=1) as rig:
        await _close_with_tail_in_tts(rig, "llm.error")
        assert F_C.strip() not in rig.tts.texts(), "premise: F_C is queued, not yet synthesised"
        rig.tts.release()
        await rig.advance(3_000, step=100)
        assert rig.state.state is TurnState.IDLE
        assert F_C.strip() not in rig.tts.texts(), (
            f"G6: a stale utterance is never synthesised; TTS was asked for {rig.tts.texts()}"
        )
        assert not {F_B, F_C} & set(rig.audio.answer_sources()), "G6: and never played"
    notes = utterance_notes(rig.log)
    assert [n["source"] for n in notes] == [F_B, F_C], (
        f"G6: one note per refused utterance, in queue order; got {notes}"
    )
    assert all(n["turn"] == "t1" and n["current_turn"] is None for n in notes), notes
