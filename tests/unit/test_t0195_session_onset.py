"""T-0195 group A, the session half: the machine supplies the flag (A3), the pump re-checks it (A4), "Hello," is
answered (A5), the barge-in guard holds (A6), the race is lost by the estimated onset (A7), the mute still wins (A8),
every estimated word is recorded (A10), and the pinned files do not move (A9).

``docs/tasks/T-0195.md`` group A. **The surface these rows assume** (the task's names):

- ``Endpointer.observe(event, *, may_open_on_estimate: bool = False)`` (A1); the session calls it from
  ``_on_stt_event`` with ``True`` only when the machine is IDLE, or LISTENING with ``words == ()`` and
  ``t_eot_ms is None`` (A3);
- the session note ``{"what": "stt.estimated_word", "start_ms", "end_ms", "machine_state", "opened"}`` for every
  estimated word that reaches the endpointer (A10) and ``{"what": "stt.estimated_onset_dropped", "onset_ms",
  "machine_state"}`` when ``_handle`` drops an estimated onset (A4). ``machine_state`` is read as the ``TurnState``
  value (``"IDLE"``, ``"ENDPOINTING"``, ...), the convention of ``machine.step``'s ``state_before`` (the task does not
  spell the format; flagged in the test-author's report).

**The rig.** ``tests/unit/_t0181_rig.mic_rig`` (read-only): a started session over a reactive ``FakeStt`` (nothing is
released before the fed audio pays for it), ``FakeAudioIO`` capture, ``ToolLlm``, a real ``SessionLog`` read back after
``aclose``. An estimated word is ``FakeStt.open_word(text, start, end)`` (``end_estimated=True``), the shape
``MoshiSttClient._expire_open_word`` publishes after the hold; the fake does not model the 1 s hold per word, its
``lag_ms`` stands in for it (the desk's arrival is "after the hold", here "after the lag"; A5's arrival formula is read
off the steps the endpointer actually saw, so it holds for either). The endpointer is wrapped by ``Watch`` (a
delegating recorder installed on ``session._endpointer``): what the session passed (the flag), in which machine state,
and what came back. Busy states are reached by submitting the machine's events directly (``Rig.onset``/``words``/
``ask``), so the endpointer **has no open run** when the estimated word arrives: with the flag ``True`` it would open
one, and only the session's flag (A3) and the pump's re-check (A4) stand in the way. That is the adversarial premise.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), ``POLICY`` (thinking on), n = 1 deterministic per row.
Every row except ``test_pinned_files_unchanged`` and the two BASE-green guards named in their docstrings is
was ``pending(task="T-0195")`` until ``/implement T-0195`` group A started (activated 2026-10-29).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.speech import SttWordEvent
from local_astra.contracts.turn import TurnState
from local_astra.reflex.turn_predictor import Endpointer
from tests.fakes.stt import FakeStt
from tests.unit._t0181_rig import MicRig, mic_rig, reactive_stt
from tests.unit._tool_turn_support import answer

REPO = Path(__file__).resolve().parents[2]
FLAG = "may_open_on_estimate"
NOTE_WORD = "stt.estimated_word"
NOTE_DROPPED = "stt.estimated_onset_dropped"
WORD_KEYS = {"what", "start_ms", "end_ms", "machine_state", "opened"}
DROPPED_KEYS = {"what", "onset_ms", "machine_state"}
REPLY = "Hi there. "
LAG_MS = 500
#: A1's busy-state commands (A6): none may follow an estimated word
FORBIDDEN = (C.PlaybackPause, C.StopClip, C.Unfreeze, C.CancelTimer, C.ArmTimer)

#: A9: sha256 of each file at 341215b7 (``git show 341215b7:<path> | sha256sum``, 2026-10-29)
PINNED_SHA256 = {
    "local_astra/conductor/turn_machine.py": "9944ee489b89bcb15720c5d130c735bfb1168145f5b730dc72b2a9433b41a266",
    "local_astra/contracts/events.py": "1812fb3f191bff27d1461f92f7ecf294a1d7718a41003ccc9b333d128d789c34",
    "local_astra/contracts/commands.py": "307a016fa2e3b10fe47fa0d41e6e3b24cf93e1d0ca34f1da47a5a86e2dd2dfaf",
    "local_astra/contracts/speech.py": "da61f85786a30bf1733c8fe3015684daf0961cc40e53f69bf2abb75681565591",
    "local_astra/conductor/stt_client.py": "8c4bffa9e32135a7f887b3d992697a987b064b18751a784f2e1247a794cf2441",
    "local_astra/reflex/level.py": "a1b0aa0213c0bc4313ed85905dcc2ca55da1365bcc9d90e251d42de9894c41e5",
}


def require_flag() -> None:
    params = inspect.signature(Endpointer.observe).parameters
    assert FLAG in params, (
        f"T-0195 A1: Endpointer.observe(event, *, {FLAG}: bool = False) does not exist yet; observe takes "
        f"{list(params)}"
    )


class Watch:
    """Delegating recorder around the session's endpointer: one row per ``observe`` call."""

    def __init__(self, inner: Any, session: Any) -> None:
        self.inner = inner
        self.session = session
        #: ``(event, flag or None when omitted, machine state at the call, decision)``
        self.calls: list[tuple[Any, Any, MachineSnapshot, Any]] = []

    def observe(self, event: Any, **kw: Any) -> Any:
        state = self.session.state
        snap = MachineSnapshot(state.state, tuple(state.words), state.t_eot_ms)
        decision = self.inner.observe(event, **kw)
        self.calls.append((event, kw.get(FLAG), snap, decision))
        return decision

    def estimated(self) -> list[tuple[Any, Any, MachineSnapshot, Any]]:
        return [c for c in self.calls if isinstance(c[0], SttWordEvent) and c[0].end_estimated]

    def decisions(self) -> list[tuple[Any, ...]]:
        return [tuple(c[3]) for c in self.calls if c[3] is not None]


class MachineSnapshot(tuple):  # type: ignore[type-arg]
    def __new__(
        cls, state: TurnState, words: tuple[Any, ...], t_eot_ms: int | None
    ) -> MachineSnapshot:
        return super().__new__(cls, (state, words, t_eot_ms))

    @property
    def state(self) -> TurnState:
        return self[0]  # type: ignore[no-any-return]


def watch(rig: MicRig) -> Watch:
    w = Watch(rig.session._endpointer, rig.session)
    rig.session._endpointer = w
    return w


def notes(rig: MicRig, what: str) -> list[dict[str, Any]]:
    return [
        r["data"]
        for r in rig.log_records
        if r.get("kind") == "note" and r["data"].get("what") == what
    ]


def decision_rows(rig: MicRig) -> list[dict[str, Any]]:
    return [r["data"] for r in rig.log_records if r.get("kind") == "stt.decision"]


def fed_ms(rig: MicRig) -> int:
    return int(rig.rec_stt.inner.fed_ms)


async def deliver(rig: MicRig, w: Watch, text: str, *, estimated: bool = True) -> SttWordEvent:
    """Script a 0 ms word whose release is already paid for (``start = end = fed - lag``) and pump until the
    endpointer has seen it. No virtual time passes, so no timer of the machine fires meanwhile."""
    at = fed_ms(rig) - LAG_MS
    assert at > 0, f"rig premise: some audio has been fed ({fed_ms(rig)} ms)"
    ev = FakeStt.open_word(text, at, at) if estimated else FakeStt.words(text, at, 0)[0]
    rig.rec_stt.inner.script.append(ev)
    seen = len(w.calls)
    for _ in range(20):
        await rig.pump()
        if any(
            c[0] is ev or (isinstance(c[0], SttWordEvent) and c[0].text == text)
            for c in w.calls[seen:]
        ):
            break
    got = [c for c in w.calls[seen:] if isinstance(c[0], SttWordEvent) and c[0].text == text]
    assert got, f"rig premise: {text!r} never reached the endpointer (fed {fed_ms(rig)} ms)"
    return got[0][0]


def stt_kwargs(**extra: Any) -> dict[str, Any]:
    return {"lag_ms": LAG_MS, **extra}


# =====================================================================================================================
# A5: a one-word question is answered
# =====================================================================================================================


async def test_hello_is_answered(tmp_path: Path) -> None:
    """A5 (the desk session of 2026-10-29, seq 7045). IDLE; the recogniser publishes ``Hello,`` with
    ``start == end == S`` and its end estimated; steps every 80 ms with the fake's silence heads (0.9: the head path
    would cross within four steps if it were allowed to decide a 0 ms run); the marker echoes. The endpointer decides,
    in order, ``("onset", S)`` then ``("end_of_turn", S, 1.0)`` at the first step with ``t - arrival >= 800``, where
    arrival = max(the step clock when the word was delivered, S); the machine walks IDLE -> LISTENING -> ENDPOINTING ->
    THINKING with ``LlmStart.transcript == "Hello,"``, the reply is spoken, the turn ends IDLE with history
    ``("Hello,", <spoken>)``, and one ``stt.estimated_word`` note says the word opened the run (A10), without its text.
    BASE: rule (h) refuses the word, no onset, no turn, no reply (the desk defect)."""
    s = 1000
    async with mic_rig(tmp_path, stt_kwargs=stt_kwargs(steps=True), script=[answer(REPLY)]) as rig:
        w = watch(rig)
        rig.rec_stt.inner.script.append(FakeStt.open_word("Hello,", s, s))
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and bool(rig.state.history),
            "A5: 'Hello,' was never answered (BASE: rule (h) refuses the only word, no onset, no turn)",
            limit_ms=12_000,
            step=40,
        )
        calls = list(w.calls)
        starts = rig.commands_of("llm.start")
        history = rig.state.history
        speaks = rig.machine.commands(C.TtsSpeak)
        path = [after for _, _, after in rig.machine.transitions()]
    require_flag()

    # the endpointer: onset, then the backstop at the step A5 names
    idx = next(
        i for i, c in enumerate(calls) if isinstance(c[0], SttWordEvent) and c[0].text == "Hello,"
    )
    hello, flag, snap, _ = calls[idx]
    assert (hello.end_estimated, flag, snap.state) == (True, True, TurnState.IDLE), (
        f"A3: in IDLE the session passes {FLAG}=True for the estimated word; got flag={flag!r} in {snap.state}"
    )
    steps_before = [
        c[0].t_ms
        for c in calls[:idx]
        if not isinstance(c[0], SttWordEvent) and hasattr(c[0], "t_ms")
    ]
    arrival = max(steps_before[-1], s) if steps_before else s
    trigger = next(
        c[0].t_ms
        for c in calls[idx + 1 :]
        if hasattr(c[0], "pause_heads") and c[0].t_ms - arrival >= 800
    )
    fired = [(getattr(c[0], "t_ms", "word"), tuple(c[3])) for c in calls if c[3] is not None]
    assert fired[:2] == [("word", ("onset", s)), (trigger, ("end_of_turn", s, 1.0))], (
        f"A5: onset at {s}, then the backstop end of turn ({s}, {s}, 1.0) at step {trigger} (arrival {arrival} + 800), "
        f"never the head path on a 0 ms run. Got {fired}"
    )

    # the machine
    wanted = [TurnState.LISTENING, TurnState.ENDPOINTING, TurnState.THINKING]
    it = iter(path)
    assert all(any(st is t for st in it) for t in wanted), (
        f"A5: IDLE -> LISTENING -> ENDPOINTING -> THINKING; {path}"
    )
    assert [c.transcript for c in starts] == ["Hello,"], (
        f"A5: one LlmStart with transcript 'Hello,'; got {starts}"
    )
    assert speaks, "A5: the reply reached TtsSpeak"
    assert (
        len(history) == 1
        and history[0].user_text == "Hello,"
        and "Hi there" in history[0].assistant_spoken
    ), f"A5: history ('Hello,', <spoken>); got {history}"

    # A10, on the same run
    got = notes(rig, NOTE_WORD)
    assert got == [
        {
            "what": NOTE_WORD,
            "start_ms": s,
            "end_ms": s,
            "machine_state": TurnState.IDLE.value,
            "opened": True,
        }
    ], f"A10: one note for the one estimated word, opened=True, no text; got {got}"


async def test_listening_before_words_lets_an_estimated_word_end_the_turn(tmp_path: Path) -> None:
    """A3's second branch (open question 3, default yes): LISTENING with ``words == ()`` and ``t_eot_ms is None`` (a
    turn the machine opened on an onset no word followed) passes the flag ``True``, the estimated word opens the
    endpointer's run, and the run ends on the backstop, so the turn reaches THINKING with transcript ``Hello,``.
    BASE (and a session that admits IDLE only): the word is refused and the turn sits in LISTENING until a later real
    word, which is why the task includes this state. Here the onset is submitted directly, so the endpointer has no
    run open when the word arrives."""
    async with mic_rig(tmp_path, stt_kwargs=stt_kwargs(steps=True), script=[answer(REPLY)]) as rig:
        w = watch(rig)
        await rig.advance(800)
        await rig.onset()
        assert (rig.state.state, rig.state.words, rig.state.t_eot_ms) == (
            TurnState.LISTENING,
            (),
            None,
        ), f"rig premise: LISTENING before words; {rig.state.state}, {rig.state.words}"
        hello = await deliver(rig, w, "Hello,")
        await rig.until(
            lambda: bool(rig.commands_of("llm.start")),
            "A3: LISTENING before words: the estimated word must open the run and the backstop end the turn",
            limit_ms=4_000,
            step=40,
        )
        call = next(c for c in w.calls if c[0] is hello)
        starts = [c.transcript for c in rig.commands_of("llm.start")]
    assert call[2].state is TurnState.LISTENING and call[1] is True, (
        f"A3: LISTENING with no words passes {FLAG}=True; got {call[1]!r} in {call[2].state}"
    )
    assert starts == ["Hello,"], f"the turn's transcript is the word; got {starts}"


# =====================================================================================================================
# A6: the barge-in guard holds (negative; BASE-green guard)
# =====================================================================================================================


async def reach(rig: MicRig, target: str) -> None:
    """Drive the machine to ``target`` with the machine's own events, submitted directly (the endpointer never opens a
    run). ``LISTENING+words``: an onset and two words. ``ENDPOINTING``: plus an end of turn, no flush. ``THINKING``:
    a whole question with the LLM held. ``SPEAKING``: a whole question answered, the TTS held. ``PAUSED``: SPEAKING plus
    an onset. ``INTERRUPTED``: PAUSED plus a takeover."""
    await rig.advance(800)  # some audio is fed, so a word can be released at once
    if target in ("LISTENING+words", "ENDPOINTING"):
        await rig.onset()
        await rig.words("what is")
        if target == "ENDPOINTING":
            now = rig.clock.now_ms
            rig.submit(E.SttEndOfTurn(t_ms=now, t_eot_ms=now - 20, p_end=0.9))
            await rig.pump()
        return
    if target == "THINKING":
        rig.llm.hold()
        await rig.ask("what is this")
        return
    rig.tts.hold()
    await rig.ask("what is this")
    await rig.until(
        lambda: rig.state.state is TurnState.SPEAKING,
        "rig premise: SPEAKING",
        limit_ms=3_000,
        step=20,
    )
    if target == "SPEAKING":
        return
    now = rig.clock.now_ms
    rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
    await rig.pump()
    if target == "PAUSED":
        return
    # the flush's PlaybackStopped would take INTERRUPTED -> LISTENING at once; held, the state rests (no clock moves)
    rig.audio.hold_events()
    rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
    await rig.pump()


BUSY = ["SPEAKING", "PAUSED", "THINKING", "ENDPOINTING", "INTERRUPTED", "LISTENING+words"]


@pytest.mark.parametrize("target", BUSY)
async def test_estimated_word_in_each_busy_state_changes_nothing(
    tmp_path: Path, target: str
) -> None:
    """A6 (negative) with A3: in each busy state an estimated word (0 ms, start after every decision, so (f) and (g)
    admit it, and no run open in the endpointer, so only the flag stands in the way) is passed with the flag False,
    produces no decision, no ``stt.decision`` row, no ``UserSpeechOnset``, and none of ``PlaybackPause``,
    ``StopClip``, ``Unfreeze``, ``CancelTimer``, ``ArmTimer``; the only event it gives the machine is its
    ``SttWord`` (BASE). Its A10 note is checked when the feature exists (``opened: false``, the state named).

    **BASE-green guard** (BASE refuses every estimated word). It is red on a session that passes ``True`` outside A3's
    predicate. **Adversarial** for SPEAKING: the premise row below shows the same word *with a measured end* takes
    this very session to PAUSED, so the absence of a pause here is the guard's doing."""
    base_has_flag = FLAG in inspect.signature(Endpointer.observe).parameters
    async with mic_rig(tmp_path, stt_kwargs=stt_kwargs(), script=[answer(REPLY)]) as rig:
        w = watch(rig)
        await reach(rig, target)
        want = TurnState.LISTENING if target == "LISTENING+words" else TurnState[target]
        before = rig.state
        assert before.state is want, f"rig premise: {target}; the machine is in {before.state}"
        if target == "LISTENING+words":
            assert before.words, "rig premise: LISTENING holds words"
        since = len(rig.machine.log)
        hello = await deliver(rig, w, "Hello,")
        call = next(c for c in w.calls if c[0] is hello)
        stepped = [ev for ev, _, _ in rig.machine.log[since:]]
        cmds = [c for _, _, cs in rig.machine.log[since:] for c in cs]
        after = rig.state
    assert call[2].state is want, (
        f"premise: the word reached the endpointer in {call[2].state}, not {want}"
    )
    assert not call[1], (
        f"A3: in {target} the session passes {FLAG}=False (or omits it); it passed {call[1]!r}"
    )
    assert call[3] is None, f"A6: no decision in {target}; got {call[3]}"
    assert [r for r in decision_rows(rig) if r.get("onset_ms") == hello.start_ms] == [], (
        "A6: no stt.decision row"
    )
    assert [type(ev).__name__ for ev in stepped] == ["SttWord"], (
        f"A6: the estimated word gives the machine its SttWord only (BASE); stepped {stepped}"
    )
    assert (stepped[0].start_ms, stepped[0].end_ms) == (hello.start_ms, hello.end_ms)
    bad = [c for c in cmds if isinstance(c, FORBIDDEN)]
    assert bad == [], f"A6: in {target} no pause, clip stop, unfreeze or timer change; got {bad}"
    assert after.state is before.state, (
        f"A6: machine state as BASE ({before.state}); got {after.state}"
    )
    if base_has_flag:  # A10 on the busy path, once the feature exists
        mine = [n for n in notes(rig, NOTE_WORD) if n.get("start_ms") == hello.start_ms]
        assert mine == [
            {
                "what": NOTE_WORD,
                "start_ms": hello.start_ms,
                "end_ms": hello.end_ms,
                "machine_state": want.value,
                "opened": False,
            }
        ], f"A10: one note, opened=False, in {want.value}; got {mine}"


async def test_premise_a_measured_word_in_speaking_pauses_this_session(tmp_path: Path) -> None:
    """The adversarial premise of A6's SPEAKING row, on the same rig: the same 0 ms word **with** a measured end
    opens a run and takes SPEAKING to PAUSED (docs/02 §4). Without this, "no pause" above could be a rig that never
    could pause. BASE-green premise (not a feature row)."""
    async with mic_rig(tmp_path, stt_kwargs=stt_kwargs(), script=[answer(REPLY)]) as rig:
        w = watch(rig)
        await reach(rig, "SPEAKING")
        await deliver(rig, w, "Hello,", estimated=False)
        await rig.pump()
        moved = rig.machine.moved(TurnState.SPEAKING, TurnState.PAUSED)
    assert moved and isinstance(moved[0], E.UserSpeechOnset), (
        f"premise: SPEAKING -> PAUSED on the onset; {moved}"
    )


# =====================================================================================================================
# A7: the race (adversarial)
# =====================================================================================================================


async def test_queued_typed_turn_wins_the_race(tmp_path: Path) -> None:
    """A7. The session reads IDLE when the recogniser hands it an estimated word, so A3 passes ``True`` and the
    endpointer opens a run; but a typed question's ``UserSpeechOnset`` is already queued ahead of that onset. By the
    time ``_handle`` reaches the estimated onset the typed turn holds ENDPOINTING with its words, so A4 drops it: no
    ``machine.step`` for it, one ``stt.estimated_onset_dropped`` note (``onset_ms``, ``machine_state``), and the typed
    turn reaches ``LlmStart`` with its own transcript, with one LISTENING entry and no ``Unfreeze``.

    Construction: ``session.submit`` is wrapped so that, the instant the session submits the typed request's own onset
    (``_request_turn``, still in IDLE), the estimated word is handed to ``_on_stt_event`` (the STT entry point A3 names)
    synchronously, before the pump can step anything. Without A4 the estimated onset reaches ENDPOINTING and the
    machine's ``user.speech_onset`` row there unfreezes back to LISTENING: the failure this row exists for."""
    typed = "what is on my screen"
    async with mic_rig(tmp_path, stt_kwargs=stt_kwargs(), script=[answer(REPLY)]) as rig:
        w = watch(rig)
        await rig.advance(800)
        session = rig.session
        original = session.submit
        fired: list[Any] = []
        # The word's audio lies just *after* the typed question's moment (its ``at_ms`` is the clock at receipt, the
        # typed turn's ``t_eot``), so BASE's word routing keeps the ``SttWord`` out of the frozen transcript; only the
        # onset is this row's subject. Handed over directly, it needs no fed audio to pay for its release.
        at = rig.clock.now_ms + 40
        held = FakeStt.open_word("Hello,", at, at)

        def submit(event: Any) -> None:
            original(event)
            pending = getattr(session, "_pending_turn", None)
            if not fired and pending is not None and event is pending.onset:
                fired.append(session.state.state)
                session._on_stt_event(held)

        session.submit = submit
        await rig.type_question(typed)
        await rig.until(
            lambda: bool(rig.commands_of("llm.start")),
            "A7: the typed turn never started",
            limit_ms=3_000,
        )
        onsets = rig.machine.events(E.UserSpeechOnset)
        listening = rig.machine.moved(TurnState.IDLE, TurnState.LISTENING) + rig.machine.moved(
            TurnState.ENDPOINTING, TurnState.LISTENING
        )
        unfreezes = rig.machine.commands(C.Unfreeze)
        starts = [c.transcript for c in rig.commands_of("llm.start")]
    require_flag()
    assert fired == [TurnState.IDLE], f"rig premise: the word was handed over in IDLE; {fired}"
    call = next(c for c in w.calls if c[0] is held)
    assert call[1] is True and call[3] is not None and tuple(call[3]) == ("onset", at), (
        f"premise (A3): in IDLE the flag is True and the endpointer opens at {at}; got flag={call[1]!r}, {call[3]}"
    )
    assert len(onsets) == 1 and onsets[0].onset_ms != at, (
        f"A4: the estimated onset (onset_ms {at}) never reaches machine.step; only the typed onset is stepped; "
        f"got {onsets}"
    )
    dropped = notes(rig, NOTE_DROPPED)
    assert dropped == [
        {"what": NOTE_DROPPED, "onset_ms": at, "machine_state": TurnState.ENDPOINTING.value}
    ], f"A4: one drop note with the onset and the state it met; got {dropped}"
    assert starts == [typed], (
        f"A7: the typed turn's LlmStart carries its typed transcript; got {starts}"
    )
    assert len(listening) == 1 and unfreezes == [], (
        f"A7: one LISTENING entry and no Unfreeze; LISTENING entries {listening}, unfreezes {unfreezes}"
    )


# =====================================================================================================================
# A8: the mute still wins (negative; BASE-green guard)
# =====================================================================================================================


async def test_estimated_word_inside_a_muted_span_is_still_dropped(tmp_path: Path) -> None:
    """A8 (and A10's negative half): IDLE, muted; an estimated word whose audio starts inside the muted span is
    dropped before ``observe`` (T-0181 A6): the endpointer never sees it, no onset, no ``SttWord``, no
    ``stt.estimated_word`` note. In IDLE the flag would be ``True``, so a session that moved the note or the
    endpointer call ahead of the mute drop fails here. **BASE-green guard.**"""
    async with mic_rig(tmp_path, stt_kwargs=stt_kwargs()) as rig:
        w = watch(rig)
        await rig.advance(400)
        first_zero = await rig.engage()
        await rig.advance(1_200)
        start = first_zero + 160
        rig.rec_stt.inner.script.append(FakeStt.open_word("Hello,", start, start))
        since = len(rig.machine.log)
        await rig.advance(1_200)
        seen = [c for c in w.calls if isinstance(c[0], SttWordEvent)]
        stepped = [
            ev.type
            for ev, _, _ in rig.machine.log[since:]
            if ev.type in ("user.speech_onset", "stt.word")
        ]
        state = rig.state.state
        fed = fed_ms(rig)
    assert fed >= start + LAG_MS, (
        f"premise: the fed audio paid for the word's release ({fed} < {start + LAG_MS})"
    )
    assert state is TurnState.IDLE, f"premise: IDLE throughout; {state}"
    assert seen == [], f"A8: a muted word never reaches Endpointer.observe; it saw {seen}"
    assert stepped == [], f"A8: no onset and no word reach the machine; got {stepped}"
    assert notes(rig, NOTE_WORD) == [], (
        f"A10: a muted word gets no note; got {notes(rig, NOTE_WORD)}"
    )


# =====================================================================================================================
# A10: one note per estimated word
# =====================================================================================================================


async def test_estimated_word_note_per_word(tmp_path: Path) -> None:
    """A10 (closes T-0051 B39): every estimated word that reaches the endpointer is recorded, once, with exactly
    ``what``, ``start_ms``, ``end_ms``, ``machine_state`` (at the call) and ``opened`` (whether it opened a run), and
    never its text; a word with a measured end gets no such note. Script: IDLE, an estimated ``Hello,`` (opens: IDLE),
    then while that run is open a second estimated word ``there`` (refused: the run is open; LISTENING with words),
    then an honest ``friend`` (no note). The expected notes are read off ``Watch``'s record of each call (state at the
    call, decision) — the endpointer's own verdicts, not the session's."""
    async with mic_rig(tmp_path, stt_kwargs=stt_kwargs()) as rig:
        w = watch(rig)
        await rig.advance(800)
        first = await deliver(rig, w, "Hello,")
        await rig.advance(160, step=40)
        second = await deliver(rig, w, "there")
        await rig.advance(160, step=40)
        await deliver(rig, w, "friend", estimated=False)
        calls = w.estimated()
    require_flag()
    assert [c[0] for c in calls] == [first, second], (
        "premise: two estimated words reached the endpointer"
    )
    assert calls[0][3] is not None and calls[1][3] is None, (
        f"premise: the first opened (IDLE, flag True), the second met the open run; got {[c[3] for c in calls]}"
    )
    expected = [
        {
            "what": NOTE_WORD,
            "start_ms": c[0].start_ms,
            "end_ms": c[0].end_ms,
            "machine_state": c[2].state.value,
            "opened": c[3] is not None,
        }
        for c in calls
    ]
    assert expected[1]["machine_state"] == TurnState.LISTENING.value, f"premise: {expected}"
    got = notes(rig, NOTE_WORD)
    assert got == expected, (
        f"A10: one note per estimated word, in order, exact keys; expected {expected}, got {got}"
    )
    for n in got:
        assert set(n) == WORD_KEYS
        text = json.dumps(n)
        assert all(word not in text for word in ("Hello", "there", "friend")), (
            f"A10: no word text; {n}"
        )


# =====================================================================================================================
# A9: pinned files
# =====================================================================================================================


def test_pinned_files_unchanged() -> None:
    """A9: byte-identical to BASE (341215b7). **BASE-green guard.**"""
    moved = {
        path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        for path, sha in PINNED_SHA256.items()
        if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != sha
    }
    assert moved == {}, (
        f"A9: these files must be byte-identical to 341215b7; moved: {sorted(moved)}"
    )


def test_reactive_stt_is_the_rig_default() -> None:
    """Rig self-check: ``mic_rig`` builds the reactive fake these rows rely on (``deliver`` pays the release with fed
    audio). BASE-green."""
    assert reactive_stt().inner.reactive is True
