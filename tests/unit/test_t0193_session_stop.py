"""T-0193 group A through the session: the route, the session log, and what a stop does to a live loop, to speech and
to the next typed turn (docs/tasks/T-0193.md A-2..A-6, A-8; "Tests to write -> A -> test_t0193_session_stop.py").

Every row runs over ``tests/unit/_t0181_rig.mic_rig`` (read-only reuse): the session over the fakes, a socketless
``HudServer`` with one greeted page, ``RecordingAudio`` (``FakeAudioIO`` capture), a reactive ``FakeStt``, the
recording machine and a ``ScriptedClock``. A stop is sent **over the wire** as raw JSON into the greeted page
(``{"type": "astra.turn.stop", "turn_id": ...}``), so before group A lands the codec answers ``Unknown`` and every row
is red on its own named assertion, never on an import error. The live rows use the live kit over
``FakeLiveBrowser`` (loop on, ``tsl.LOOP``).

Expected values are written by hand from the criteria and the A0 rows of docs/02 §4 (merged before group A):

- A-2: one routed ``astra.turn.stop`` is exactly one ``UserStop(t_ms=clock.now_ms, turn_id=message.turn_id)`` stepped,
  whatever the state and whatever the turn id (the machine decides, Implementation notes "Purity"), and no
  ``astra.error`` answers it; an ungreeted socket's stop is the hello fence's ``no_hello`` and is never stepped;
- A-3: one ``note`` per stepped ``UserStop`` with data keys exactly ``{what, turn_id, state, acted}``;
  ``ended_by: "stopped"`` on the turn line of a turn a stop ended; ``tool_loop_end: "stopped"`` on a loop turn a stop
  ended, on the THINKING path and on the SPEAKING path (where it takes precedence over ``"interrupted"``);
- A-4: the adversarial row of the task: the fake browser blocks ``click_at`` on an ``asyncio.Event`` the row owns; the
  stop comes while it blocks; afterwards the row releases it. Two executors: a cooperative one (the cancel reaches
  it and it records ``execute.cancelled``) and a stubborn one that swallows the cancel, waits for the release and
  returns ``executed`` anyway (a late result). Either way: no ``llm.continue``, no further request to the LLM, no
  ``tool.execute``, the card ends ``cancelled`` / ``turn closed``, the browser is not opened, closed or restarted,
  and a typed question afterwards opens ``t2`` and is answered;
- A-5: speech is cut in the step that handles the stop (``flush`` reached the device before any virtual time
  passed), the page's ``astra.turn.state`` after the stop is exactly INTERRUPTED then IDLE, history's last entry is the
  played prefix the device reported plus ``" [interrupted]"``, and nothing is played or published as spoken after it.

**Readings** (listed in the test-author's report):

- A-3's ``state`` is the state the stop **arrived in** (the machine's state before the step), as a ``TurnState``
  value (``"SPEAKING"``); ``turn_id`` is the stop's own (the message's), not the machine's, so a stale click logs the
  id it named; ``acted`` is ``True`` exactly when the step changed the turn (THINKING -> IDLE, SPEAKING/PAUSED ->
  INTERRUPTED);
- A-3's ``ended_by`` is a turn-line ``extras`` key (where T-0162 writes ``"reset"`` and T-0181 ``"mic_muted"``), and
  it is absent from a line whose turn no stop ended (a no-op stop leaves the next line without it);
- A-4 "the agent browser is not navigated, reset or closed" is read off the fake's call log (no ``open``, no
  ``close``; the seam has no reset, and ``FakeLiveBrowser.__getattr__`` makes any other name a named failure) and its
  generation (unchanged);
- A-5 "the played prefix" is the ``played_text`` of the terminal playback event that took the machine from
  INTERRUPTED to IDLE (the device's report, ``FakeAudioIO``), joined with nothing (a tools-off turn has no earlier
  run); the row's premise is that this exit was a ``playback.stopped`` with a non-empty text that is a strict prefix
  of the answer, so the history row compares something;
- A-6's "byte-identical apart from the ``turn.stop`` note" compares the turn line (raw JSON, the per-session
  ``session_id``/``session_log`` keys removed in both arms, which have their own log directories), history, every
  published message and the TTS requests; the session-log records differ by the stop's ``wire.in``, its
  ``machine.step`` and its note, which are removed by kind and ``what`` before comparing;
- A-8's surface "on screen" is the turn's surface as the page last received it: the last ``astra.ui.surface`` for
  ``turn-t1`` before the stop and the last one after it carry the same ``messages``, none after the stop deletes it,
  no ``astra.session.cleared`` is published, and a page greeted after the stop is replayed the same surface.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on (``TurnPolicy(thinking=True)``, ack clip on) or
the loop policy (live), ``RecordingTts`` 200 ms a word, ``FakeAudioIO``, n = 1, deterministic per row. No measurement.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from local_astra.contracts.turn import TurnState
from tests.unit import test_session_tool_loop as tsl
from tests.unit._hud_rig import ui
from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_kit
from tests.unit._t0181_rig import LONG_ANSWER, QUESTION, MicRig, mic_rig
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool

STOP = "astra.turn.stop"
NOTE = "turn.stop"
NOTE_KEYS = {"what", "turn_id", "state", "acted"}
MARKER = " [interrupted]"
#: the second typed question, answered after a stop (A-4, A-6)
Q2 = "and what about tomorrow"
A2 = "Tomorrow is sunny. "
CID = tsl.call_id(1)


# =====================================================================================================
# reading the rig
# =====================================================================================================


def send_stop(rig: MicRig, turn_id: str, page: Any = None) -> None:
    """The page's ``astra.turn.stop``, as raw JSON (``Unknown`` before A-1, never a ``ValidationError`` here)."""
    (page or rig.page).receive(json.dumps({"type": STOP, "turn_id": turn_id}))


async def stop(rig: MicRig, turn_id: str | None = None) -> int:
    """Send a stop for ``turn_id`` (default: the machine's current turn) and pump it; no virtual time passes.
    Returns the machine-log index the stop's step starts at."""
    mark = len(rig.machine.log)
    send_stop(rig, rig.state.turn_id if turn_id is None else turn_id)
    await rig.pump()
    return mark


def stops(rig: MicRig, since: int = 0) -> list[Any]:
    return rig.stepped("user.stop", since)


def stop_moves(rig: MicRig, since: int = 0) -> list[tuple[TurnState, TurnState]]:
    """``(state before, state after)`` of every ``UserStop`` step from log index ``since`` (the fake device reports
    the flush's ``playback.stopped`` within the same pump, so the current state may already be past it)."""
    return [(b, a) for b, ev, a in rig.machine.transitions()[since:] if ev.type == "user.stop"]


def turn_states(messages: list[Any]) -> list[tuple[str, str]]:
    return [
        (m.state.value, m.turn_id)
        for m in messages
        if getattr(m, "type", None) == "astra.turn.state"
    ]


def errors(page: Any, since: int = 0) -> list[tuple[str, str]]:
    return [(m.code, m.message) for m in page.sent()[since:] if m.type == "astra.error"]


def line(rig: MicRig, turn_id: str) -> dict[str, Any]:
    found = [x for x in rig.raw_lines() if x["turn_id"] == turn_id]
    assert len(found) == 1, (
        f"one turn line for {turn_id}; got {[x['turn_id'] for x in rig.raw_lines()]}"
    )
    return found[0]


def history(rig: MicRig) -> list[tuple[str, str]]:
    return [(h.user_text, h.assistant_spoken) for h in rig.state.history]


async def until_idle(rig: MicRig, what: str, *, limit_ms: int = 3_000) -> None:
    await rig.until(lambda: rig.state.state is TurnState.IDLE, what, limit_ms=limit_ms, step=20)


async def until_speaking(rig: MicRig, *, limit_ms: int = 6_000) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.SPEAKING,
        "premise: the answer never started",
        limit_ms=limit_ms,
        step=40,
    )


async def ask_and_answer(rig: MicRig, question: str, reply: str, n_lines: int) -> None:
    """Type ``question`` with ``reply`` scripted as the next turn generate; wait for IDLE with ``n_lines`` lines."""
    rig.llm.script = [answer(reply)]
    await rig.type_question(question)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n_lines,
        f"the typed question {question!r} was not answered",
        limit_ms=20_000,
        step=40,
    )
    await rig.advance(300, step=40)


async def typed_speaking(rig: MicRig) -> None:
    """``QUESTION`` typed, ``LONG_ANSWER`` (two sentences) playing for 400 ms."""
    await rig.advance(400, step=40)
    await rig.type_question(QUESTION)
    await until_speaking(rig)
    await rig.advance(400, step=40)
    assert rig.state.state is TurnState.SPEAKING, f"premise: SPEAKING; {rig.state.state}"


# =====================================================================================================
# A-2: the route
# =====================================================================================================


@pytest.mark.parametrize(
    "where", ["idle", "speaking-current", "speaking-stale", "thinking-current"]
)
async def test_route_submits_one_userstop(tmp_path: Path, where: str) -> None:
    """A-2: one ``astra.turn.stop`` behind the hello fence is exactly one ``UserStop`` stepped, with the session clock
    at the send and the message's ``turn_id``, in every state and for any turn id (a session that decides by itself
    that a stop does not apply, in IDLE or for a stale id, submits nothing and is red here). No ``astra.error`` answers
    it and the server counts it as routed. A no-op stop changes nothing the machine holds."""
    async with mic_rig(tmp_path, script=[answer(LONG_ANSWER)]) as rig:
        if where == "idle":
            await rig.advance(400, step=40)
            target = "t1"  # the id the next turn will take: a stop in IDLE must not reach it
        elif where == "thinking-current":
            await rig.advance(400, step=40)
            rig.llm.hold()
            await rig.type_question(QUESTION)
            await rig.until(
                lambda: rig.state.state is TurnState.THINKING, "premise: THINKING", step=20
            )
            target = rig.state.turn_id
        else:
            await typed_speaking(rig)
            target = rig.state.turn_id if where == "speaking-current" else "t0"
        before_state = rig.state
        routed = rig.server.counts["routed"]
        sent_mark = len(rig.page.sent())
        t_send = rig.clock.now_ms
        since = await stop(rig, target)
        got = stops(rig, since)
        assert [(e.t_ms, e.turn_id) for e in got] == [(t_send, target)], (
            f"T-0193 A-2: one UserStop(t_ms={t_send}, turn_id={target!r}) stepped; got {got}"
        )
        assert errors(rig.page, sent_mark) == [], (
            f"A-2: no astra.error for a stop; {errors(rig.page, sent_mark)}"
        )
        assert rig.server.counts["routed"] == routed + 1, (
            f"A-2: the stop is routed once; {rig.server.counts}"
        )
        if where in ("idle", "speaking-stale"):
            assert (
                rig.state.state is before_state.state and rig.state.turn_id == before_state.turn_id
            ), f"A0-2: a stop in IDLE or for another turn changes nothing; {rig.state.state}"
        if where == "thinking-current":
            rig.llm.release()


async def test_ungreeted_stop_dropped(tmp_path: Path) -> None:
    """A-2 (adversarial): a socket that never said hello and sends a stop naming the speaking turn is answered
    ``no_hello`` and nothing is stepped; the answer plays on. Power: the same bytes from the greeted page act."""
    async with mic_rig(tmp_path, script=[answer(LONG_ANSWER)]) as rig:
        await typed_speaking(rig)
        stranger = rig.server.open()  # never greeted
        no_hello = rig.server.counts["no_hello"]
        since = len(rig.machine.log)
        send_stop(rig, rig.state.turn_id, page=stranger)
        await rig.pump()
        await rig.advance(200, step=40)
        assert stops(rig, since) == [], (
            f"A-2: an ungreeted stop is never stepped; {stops(rig, since)}"
        )
        assert rig.state.state is TurnState.SPEAKING, f"A-2: the answer plays on; {rig.state.state}"
        assert [c for c, _ in errors(stranger)] == ["no_hello"], stranger.sent()
        assert rig.server.counts["no_hello"] == no_hello + 1, rig.server.counts
        since = await stop(rig)
        assert len(stops(rig, since)) == 1, "power: the greeted page's stop is stepped"
        assert stop_moves(rig, since) == [(TurnState.SPEAKING, TurnState.INTERRUPTED)], (
            f"T-0193 A-2: the greeted page's stop for the speaking turn acts; {stop_moves(rig, since)}"
        )


# =====================================================================================================
# A-5: stopping speech
# =====================================================================================================


async def test_stop_while_speaking(tmp_path: Path) -> None:
    """A-5: with ``LONG_ANSWER`` (two sentences) playing, a stop flushes the device in the step that handles it (no
    virtual time passes), the page is sent exactly INTERRUPTED then IDLE for ``t1``, history's last entry is
    ``(QUESTION, <played prefix> + " [interrupted]")`` where the prefix is what the device reported, and after the
    stop nothing is played (no utterance, no clip of any kind) and no ``astra.spoken`` is published, through two more
    seconds. The turn line says ``ended_by: "stopped"`` (A-3)."""
    async with mic_rig(tmp_path, script=[answer(LONG_ANSWER)]) as rig:
        await typed_speaking(rig)
        turn = rig.state.turn_id
        flushes = rig.audio.calls.count("flush")
        plays = len(rig.audio.plays)
        tts_requests = len(rig.tts.requests)
        page_mark = len(rig.page.sent())
        pub_mark = len(rig.published.messages)
        since = await stop(rig)
        assert rig.audio.calls.count("flush") == flushes + 1, (
            f"T-0193 A-5: playback.flush ran in the stop's step; device calls {rig.audio.calls[-6:]}"
        )
        assert stop_moves(rig, since) == [(TurnState.SPEAKING, TurnState.INTERRUPTED)], (
            f"A-5: the stop interrupts the answer; {stop_moves(rig, since)}"
        )
        await until_idle(rig, "A-5: the stopped turn never reached IDLE")
        exits = [
            ev
            for b, ev, a in rig.machine.transitions()[since:]
            if (b, a) == (TurnState.INTERRUPTED, TurnState.IDLE)
        ]
        await rig.advance(2_000, step=40)
        after_plays = rig.audio.plays[plays:]
        after_pub = rig.published.messages[pub_mark:]
        page_after = rig.page.sent()[page_mark:]
        assert len(exits) == 1 and exits[0].type == "playback.stopped", (
            f"premise: the device reported the cut (playback.stopped) and that ended the turn; {exits}"
        )
        played = exits[0].played_text
        assert played and LONG_ANSWER.startswith(played) and len(played) < len(LONG_ANSWER), (
            f"premise: a non-empty strict prefix of the answer was played; {played!r}"
        )
        assert history(rig)[-1] == (QUESTION, played + MARKER), (
            f"T-0193 A-5: history keeps exactly the played prefix and the marker; {history(rig)[-1]}"
        )
        assert turn_states(page_after) == [("INTERRUPTED", turn), ("IDLE", turn)], (
            f"A-5: the page is sent INTERRUPTED then IDLE; {turn_states(page_after)}"
        )
        assert after_plays == [], f"A-5: nothing plays after the stop; {after_plays}"
        assert [m for m in after_pub if m.type == "astra.spoken"] == [], (
            "A-5: no astra.spoken after the stop"
        )
        assert rig.commands_of("play_clip", since) == [], "A-5: no clip of any kind after the stop"
        assert len(rig.tts.requests) >= tts_requests
        assert rig.commands_of("tts.speak", since) == [], "A-5: no tts.speak after the stop"
    assert line(rig, turn)["extras"].get("ended_by") == "stopped", line(rig, turn)["extras"]


# =====================================================================================================
# A-3: the note and the turn line
# =====================================================================================================


@pytest.mark.parametrize("where", ["thinking", "speaking", "paused", "idle", "speaking-stale"])
async def test_note_and_extras(tmp_path: Path, where: str) -> None:
    """A-3: each stepped stop writes exactly one ``note`` ``{what: "turn.stop", turn_id, state, acted}`` (keys exact;
    ``state`` the arrival state, ``turn_id`` the stop's own). A stop that acted puts ``ended_by: "stopped"`` on its
    turn's line; a no-op stop puts it on no line (the stale one's turn completes as usual)."""
    async with mic_rig(tmp_path, script=[answer(LONG_ANSWER)]) as rig:
        await rig.advance(400, step=40)
        if where == "thinking":
            rig.llm.hold()
            await rig.type_question(QUESTION)
            await rig.until(
                lambda: rig.state.state is TurnState.THINKING, "premise: THINKING", step=20
            )
            target, state, acted = "t1", "THINKING", True
        elif where == "idle":
            target, state, acted = "t1", "IDLE", False
        elif where == "paused":
            # the reflex lane's onset (A-R12 of T-0181): the answer pauses; no barge-in words, so the stop decides
            await rig.type_question(QUESTION)
            await until_speaking(rig)
            await rig.advance(400, step=40)
            rig.submit(E.UserSpeechOnset(t_ms=rig.clock.now_ms, onset_ms=rig.clock.now_ms))
            await rig.pump()
            assert rig.state.state is TurnState.PAUSED, f"premise: PAUSED; {rig.state.state}"
            target, state, acted = "t1", "PAUSED", True
        else:
            await rig.type_question(QUESTION)
            await until_speaking(rig)
            await rig.advance(400, step=40)
            target = "t1" if where == "speaking" else "t0"
            state, acted = "SPEAKING", where == "speaking"
        since = await stop(rig, target)
        if where == "paused":
            assert stop_moves(rig, since) == [(TurnState.PAUSED, TurnState.INTERRUPTED)], (
                f"T-0193 A0-3: a PAUSED stop interrupts; {stop_moves(rig, since)}"
            )
        if where == "thinking":
            rig.llm.release()
        if where == "idle":
            await ask_and_answer(rig, QUESTION, LONG_ANSWER, 1)
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 1,
            "the turn never ended",
            limit_ms=20_000,
            step=40,
        )
        await rig.advance(300, step=40)
        exits = [
            ev
            for b, ev, a in rig.machine.transitions()[since:]
            if (b, a) == (TurnState.INTERRUPTED, TurnState.IDLE)
        ]
        last = history(rig)[-1] if rig.state.history else None
    if where == "paused":
        assert len(exits) == 1 and exits[0].type == "playback.stopped", (
            f"premise: the device reported the cut; {exits}"
        )
        played = exits[0].played_text
        assert played and LONG_ANSWER.startswith(played), f"premise: a played prefix; {played!r}"
        assert last == (QUESTION, played + MARKER), (
            f"T-0193 A0-4 (PAUSED): history keeps the played prefix and the marker; {last}"
        )
    notes = rig.notes(NOTE)
    assert notes == [{"what": NOTE, "turn_id": target, "state": state, "acted": acted}], (
        f"T-0193 A-3: one turn.stop note with exactly {sorted(NOTE_KEYS)}; got {notes}"
    )
    ended = [x["extras"].get("ended_by") for x in rig.raw_lines()]
    want = ["stopped"] if acted else [None]
    assert ended == want, f"A-3: ended_by per turn line {ended}; want {want}"


@asynccontextmanager
async def loop_session(
    tmp_path: Path, *, step2: str = tsl.S3, stubborn: bool = False
) -> AsyncIterator[tuple[MicRig, FakeLiveBrowser, asyncio.Event, list[str]]]:
    """A live session (loop on): step 1 says ``S1`` and clicks (``VALID_CLICK``, call ``t1-call``), step 2 answers
    ``step2``. The click blocks on ``gate`` (the row's). ``stubborn``: the executor swallows a cancel, waits for the
    gate and returns ``executed`` anyway (the late result); otherwise a cancel propagates. ``seen`` records what the
    executor saw (``"cancelled"``, ``"returned"``)."""
    browser = FakeLiveBrowser(
        [live_frame("t0193-onset")],
        post_frames=[live_frame(f"t0193-after-{k}") for k in (1, 2, 3)],
    )
    gate = asyncio.Event()
    seen: list[str] = []

    async def blocked(fb: FakeLiveBrowser, inv: ToolInvocation) -> ToolOutcome:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            seen.append("cancelled")
            fb.calls.append(("execute.cancelled", inv.call_id))
            if not stubborn:
                raise
            await gate.wait()
        seen.append("returned")
        return fb.perform(inv)

    browser.handlers[CID] = blocked
    async with mic_rig(
        tmp_path,
        kit=live_kit(browser),
        policy=tsl.LOOP,
        script=[answer(tsl.S1), tool(VALID_CLICK)],
    ) as rig:
        browser.now_ms = lambda: rig.clock.now_ms
        rig.llm.script_steps([answer(step2)])
        await rig.advance(400, step=40)
        try:
            yield rig, browser, gate, seen
        finally:
            gate.set()  # a stubborn executor that swallowed a cancel must not hang the session's close
            await asyncio.sleep(0)


# =====================================================================================================
# A-4: stopping a live loop mid-click (adversarial: a late result must not restart the loop)
# =====================================================================================================


@pytest.mark.parametrize("stubborn", [False, True], ids=["cooperative", "late-result"])
async def test_stop_live_loop_mid_click(tmp_path: Path, stubborn: bool) -> None:
    """A-4: with ``click_at`` blocked in the fake browser, a stop ends the turn: the execution is cancelled (the
    executor sees the cancel), the card ends ``cancelled`` / ``turn closed``, and once the row releases the click --
    the stubborn executor then returns ``executed``, a late result -- no ``llm.continue`` follows, the LLM is asked
    nothing more, no ``tool.execute`` follows, and the browser is not opened, closed or restarted. Then a typed
    question opens ``t2`` and is answered (history and the spoken text), and its line has no ``ended_by``."""
    async with loop_session(tmp_path, stubborn=stubborn) as (rig, browser, gate, seen):
        await rig.type_question(tsl.QUESTION)
        await rig.until(
            lambda: ("execute", CID) in browser.calls,
            "premise: the click never reached the browser",
            limit_ms=10_000,
            step=20,
        )
        await rig.advance(100, step=20)
        assert rig.state.state is TurnState.THINKING and "returned" not in seen, (
            f"premise: the click is blocked while the turn awaits it; {rig.state.state}, {seen}"
        )
        generates, prefills = len(rig.llm.generates), len(rig.llm.prefills)
        calls_mark, generation = len(browser.calls), browser.generation
        since = await stop(rig, "t1")
        assert rig.state.state is TurnState.IDLE, f"A0-5: THINKING -> IDLE; {rig.state.state}"
        assert seen[:1] == ["cancelled"], f"T-0193 A-4: the execution task was cancelled; {seen}"
        await rig.advance(500, step=40)
        gate.set()
        await rig.advance(2_000, step=40)
        if stubborn:
            assert "returned" in seen, f"premise: the stubborn executor returned late; {seen}"
        assert rig.commands_of("llm.continue", since) == [], (
            "A-4: the late result restarted the loop (llm.continue)"
        )
        assert rig.commands_of("tool.execute", since) == [], "A-4: a tool.execute followed the stop"
        assert (len(rig.llm.generates), len(rig.llm.prefills)) == (generates, prefills), (
            "A-4: the LLM was asked again after the stop; "
            f"generates {generates}->{len(rig.llm.generates)}, prefills {prefills}->{len(rig.llm.prefills)}"
        )
        statuses = [(m.status, m.reason) for m in rig.tool_calls(CID)]
        assert statuses[-1] == ("cancelled", "turn closed"), (
            f"A-4: the card ends cancelled/turn closed; {statuses}"
        )
        later = browser.calls[calls_mark:]
        assert not [c for c in later if c[0] in ("open", "close", "execute")], (
            f"A-4: the browser was navigated, closed or driven after the stop; {later}"
        )
        assert browser.generation == generation and not browser.closed, (
            "A-4: the browser was not restarted"
        )
        assert rig.state.state is TurnState.IDLE and len(rig.lines()) == 1, (
            f"A-4: the stopped turn stays ended; {rig.state.state}, {len(rig.lines())} lines"
        )
        mark = len(rig.machine.log)
        await ask_and_answer(rig, Q2, A2, 2)
        starts = [c.transcript for c in rig.commands_of("llm.start", mark)]
        assert starts == [Q2], f"A-4: the next typed question opens a turn; llm.start {starts}"
        assert history(rig)[-1] == (Q2, A2), f"A-4: ... and is answered; {history(rig)[-1]}"
        assert rig.state.turn_id == "t2", rig.state.turn_id
    assert line(rig, "t1")["extras"].get("tool_loop_end") == "stopped", line(rig, "t1")["extras"]
    assert line(rig, "t2")["extras"].get("ended_by") is None, line(rig, "t2")["extras"]


async def test_late_tool_result_after_stop_is_inert(tmp_path: Path) -> None:
    """A-4 (adversarial, the machine's half through the session): after the stop, a ``ToolResult`` for the awaited
    call submitted directly on the pump (as a racing executor would) moves nothing: no ``llm.continue``, no request,
    no ``tts.speak``, no new turn line, history unchanged."""
    async with loop_session(tmp_path) as (rig, browser, gate, _seen):
        await rig.type_question(tsl.QUESTION)
        await rig.until(
            lambda: ("execute", CID) in browser.calls, "premise: no click", limit_ms=10_000, step=20
        )
        await rig.advance(100, step=20)
        since = await stop(rig, "t1")
        assert rig.state.state is TurnState.IDLE, (
            f"T-0193 A0-5/A-4: the stop ends the loop turn; {rig.state.state}"
        )
        kept = history(rig)
        generates = len(rig.llm.generates)
        rig.submit(E.ToolResult(t_ms=rig.clock.now_ms, call_id=CID, status="executed"))
        await rig.pump()
        await rig.advance(1_000, step=40)
        late = rig.stepped("tool.result", since)
        assert [ev.call_id for ev in late] == [CID], f"premise: the late result was stepped; {late}"
        assert (
            rig.commands_of("llm.continue", since) == []
            and rig.commands_of("tts.speak", since) == []
        )
        assert (
            len(rig.llm.generates) == generates and history(rig) == kept and len(rig.lines()) == 1
        )
        gate.set()


@pytest.mark.parametrize("path", ["thinking", "speaking", "paused"])
async def test_loop_end_label_is_stopped(tmp_path: Path, path: str) -> None:
    """A-3: a loop turn a stop ended has ``tool_loop_end: "stopped"`` and ``ended_by: "stopped"``. THINKING: the stop
    lands while the click blocks (a THINKING stop emits no ``tool.cancel`` for an executing call, so the label cannot
    come from a cancel). SPEAKING: the click returns, step 2 speaks ``LONG_ANSWER`` and the stop lands mid-speech; the
    turn closes at the INTERRUPTED exit with a truncation, and ``"stopped"`` takes precedence over ``"interrupted"``."""
    async with loop_session(tmp_path, step2=LONG_ANSWER) as (rig, browser, gate, _seen):
        await rig.type_question(tsl.QUESTION)
        await rig.until(
            lambda: ("execute", CID) in browser.calls, "premise: no click", limit_ms=10_000, step=20
        )
        if path == "thinking":
            await rig.advance(100, step=20)
            assert rig.state.state is TurnState.THINKING, f"premise: THINKING; {rig.state.state}"
        else:
            gate.set()
            await rig.until(
                lambda: rig.state.state is TurnState.SPEAKING and rig.state.tool_step >= 2,
                "premise: step 2 never spoke",
                limit_ms=10_000,
                step=40,
            )
            await rig.advance(400, step=40)
            assert rig.state.state is TurnState.SPEAKING, f"premise: SPEAKING; {rig.state.state}"
            if path == "paused":
                # PAUSED (the reflex lane's onset, no barge-in words): the stop acts as in SPEAKING
                rig.submit(E.UserSpeechOnset(t_ms=rig.clock.now_ms, onset_ms=rig.clock.now_ms))
                await rig.pump()
                assert rig.state.state is TurnState.PAUSED, f"premise: PAUSED; {rig.state.state}"
        since = await stop(rig, "t1")
        if path == "paused":
            assert stop_moves(rig, since) == [(TurnState.PAUSED, TurnState.INTERRUPTED)], (
                f"T-0193 A0-3: a PAUSED stop interrupts; {stop_moves(rig, since)}"
            )
        await until_idle(rig, "the stopped loop turn never reached IDLE")
        if path in ("speaking", "paused"):
            assert rig.commands_of("truncate_history_to_played", since), (
                "premise: the SPEAKING path closes with a truncation (what labels a voice interrupt 'interrupted')"
            )
        gate.set()
        await rig.advance(500, step=40)
    extras = line(rig, "t1")["extras"]
    assert extras.get("tool_loop_end") == "stopped", (
        f"T-0193 A-3 ({path}): tool_loop_end 'stopped'; got {extras.get('tool_loop_end')!r}"
    )
    assert extras.get("ended_by") == "stopped", (
        f"A-3 ({path}): ended_by 'stopped'; {extras.get('ended_by')!r}"
    )


# =====================================================================================================
# A-6: typed turns unaffected (negative)
# =====================================================================================================


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if k not in ("session_id", "session_log")}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _is_stop_record(record: dict[str, Any]) -> bool:
    data = record.get("data") or {}
    if record.get("kind") == "note" and data.get("what") == NOTE:
        return True
    if record.get("kind") == "wire.in":
        msg = data.get("msg")
        return isinstance(msg, dict) and msg.get("type") == STOP
    if record.get("kind") == "machine.step":
        return (data.get("event") or {}).get("type") == "user.stop"
    return False


async def _idle_arm(tmp: Path, *, stop_first: bool) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with mic_rig(tmp, script=[answer(LONG_ANSWER)]) as rig:
        await rig.advance(400, step=40)
        if stop_first:
            send_stop(rig, "t1")
        await rig.pump()
        await rig.advance(200, step=40)
        await rig.type_question(QUESTION)
        await rig.turn_done()
        await rig.advance(400, step=40)
        out = {
            "stops": len(stops(rig)),
            "lines": [_scrub(x) for x in rig.raw_lines()],
            "history": history(rig),
            "published": [encode(m) for m in rig.published.messages],
            "tts": list(rig.tts.requests),
            "plays": [(s, getattr(u, "text", None)) for u, s in rig.audio.plays],
        }
    records = [
        {"kind": r["kind"], "data": _scrub(r["data"])}
        for r in rig.log_records
        if r["kind"] not in ("log.started", "log.stopped")
    ]
    out["records"] = [r for r in records if not _is_stop_record(r)]
    out["stop_records"] = [r for r in records if _is_stop_record(r)]
    # per-session names (the session id inside attempt ids, the arm's tmp dir) are the only allowed difference
    text = json.dumps(out).replace(rig.session.session_id, "<sid>").replace(str(tmp), "<tmp>")
    normalised: dict[str, Any] = json.loads(text)
    return normalised


async def test_stop_in_idle_then_typed(tmp_path: Path) -> None:
    """A-6 (negative): a stop sent in IDLE, naming ``t1`` (the id the next turn takes), does nothing to the typed turn
    that follows: its turn line, history, every published message, every TTS request, every play and every session-log
    record equal a session that never stopped, apart from the stop's own records (``wire.in``, ``machine.step``, the
    ``turn.stop`` note). Anti-vacuity: the stopped arm stepped one ``UserStop`` and logged its note."""
    plain = await _idle_arm(tmp_path / "plain", stop_first=False)
    stopped = await _idle_arm(tmp_path / "stopped", stop_first=True)
    assert stopped["stops"] == 1, f"T-0193 A-2: the IDLE stop was stepped; {stopped['stops']}"
    assert [r["kind"] for r in stopped["stop_records"]].count("note") == 1, stopped["stop_records"]
    assert plain["history"] == [[QUESTION, LONG_ANSWER]], f"premise: answered; {plain['history']}"
    for key in ("lines", "history", "published", "tts", "plays", "records"):
        assert stopped[key] == plain[key], (
            f"A-6: a stop in IDLE changed the next typed turn's {key}"
        )


async def test_typed_turn_after_a_stop_is_answered_in_full(tmp_path: Path) -> None:
    """A-6: once a SPEAKING stop has reached IDLE, a typed question opens ``t2`` and is answered exactly: its
    ``llm.start`` carries the question, every sentence is requested from TTS and played, history ends with
    ``(Q2, A2)`` and ``t2``'s line has no ``ended_by``."""
    async with mic_rig(tmp_path, script=[answer(LONG_ANSWER)]) as rig:
        await typed_speaking(rig)
        await stop(rig, "t1")
        await until_idle(rig, "T-0193 A-5: the stopped turn never reached IDLE")
        mark, tts_mark = len(rig.machine.log), len(rig.tts.requests)
        await ask_and_answer(rig, Q2, A2, 2)
        assert [c.transcript for c in rig.commands_of("llm.start", mark)] == [Q2]
        spoken = "".join(t for _, t, _ in rig.tts.requests[tts_mark:])
        assert spoken.split() == A2.split(), (
            f"A-6: every word of the answer is requested; {spoken!r}"
        )
        assert history(rig)[-1] == (Q2, A2), history(rig)
        assert history(rig)[0][1].endswith(MARKER), f"premise: t1 was stopped; {history(rig)[0]}"
    assert line(rig, "t2")["extras"].get("ended_by") is None
    assert line(rig, "t1")["extras"].get("ended_by") == "stopped"


# =====================================================================================================
# A-8: untouched by a stop (negative)
# =====================================================================================================


async def test_untouched_toggles(tmp_path: Path) -> None:
    """A-8: a stop leaves the mic mute, the assist toggles, the vision source, the log toggle and the A2UI surface on
    screen as they were. Each is set away from its default first (mute on, page text on, vision ``shared``, logging
    off; a card on screen), so a stop that reset them would show. Read off the session's own state messages before
    and after, off the capture (still zeroed), off what is published after the stop, and off a page greeted after it."""
    browser = FakeLiveBrowser([live_frame("t0193-a8")])
    async with mic_rig(
        tmp_path,
        kit=live_kit(browser),
        policy=tsl.LOOP,
        script=[answer(LONG_ANSWER), ui(1)],
    ) as rig:
        browser.now_ms = lambda: rig.clock.now_ms
        await rig.advance(400, step=40)
        flags = {"page_text": True, "pdf_text": False, "cells": False, "at_spi": False}
        rig.page.receive(json.dumps({"type": "astra.assist.toggle", "flags": flags}))
        rig.page.receive(json.dumps({"type": "astra.vision.select", "source": "shared"}))
        await rig.engage()
        await rig.type_question(QUESTION)
        await until_speaking(rig)
        await rig.advance(400, step=40)

        def toggles() -> list[str]:
            s = rig.session
            msgs = [
                s.assists.state_message(),
                s.log_state_message(),
                s.mic_state_message(),
                s.vision_state_message(),
            ]
            return [encode(m) for m in msgs]

        rig.page.receive(json.dumps({"type": "astra.log.toggle", "on": False}))
        await rig.pump()
        before = toggles()
        assert (
            '"page_text":true' in before[0]
            and '"muted":true' in before[2]
            and '"shared"' in before[3]
        ), f"premise: the toggles are away from their defaults; {before}"
        surfaces = [m for m in rig.published.messages if m.type == "astra.ui.surface"]
        assert surfaces, "premise: a card is on screen before the stop"
        on_screen = surfaces[-1].messages
        fed = len(rig.rec_stt.fed)
        pub_mark = len(rig.published.messages)
        await stop(rig, "t1")
        await until_idle(rig, "T-0193 A-5: the stopped turn never reached IDLE")
        await rig.advance(400, step=40)
        after = toggles()
        after_pub = rig.published.messages[pub_mark:]
        late = rig.greet()
        replayed = [m for m in late.sent() if m.type == "astra.ui.surface"]
        zero_after = rig.rec_stt.zero()[fed:]
        stepped = len(stops(rig))
    assert stepped == 1, "T-0193 A-2: the stop was stepped"
    assert after == before, f"A-8: a stop changed a toggle; before {before}, after {after}"
    assert zero_after and all(zero_after), (
        "A-8: the mic stays muted after a stop (frames still zeroed)"
    )
    kinds = [m.type for m in after_pub]
    for kind in (
        "astra.assist.state",
        "astra.log.state",
        "astra.mic.state",
        "astra.vision.state",
        "astra.session.cleared",
    ):
        assert kind not in kinds, f"A-8: a stop published {kind}; {kinds}"
    after_surfaces = [m for m in after_pub if m.type == "astra.ui.surface"]
    assert all(m.messages == on_screen for m in after_surfaces), (
        "A-8: a stop changed the card on screen"
    )
    assert not any(
        "deleteSurface" in json.dumps(m.messages) for m in after_pub if m.type == "astra.ui.surface"
    )
    assert replayed and replayed[-1].messages == on_screen, (
        "A-8: a page greeted after the stop is replayed the same card"
    )


# =====================================================================================================
# THINKING: no clip, nothing voiced
# =====================================================================================================


async def test_stop_thinking_no_clip(tmp_path: Path) -> None:
    """A0-5 through the session (Goal: "Nothing new is spoken: no clip, no apology"): a typed question whose generate
    is held, so the turn sits in THINKING with the ack clip; the stop ends it in IDLE in its own step, aborts the
    generate once, and after it nothing is played (no clip, no utterance), no ``tts.speak`` or ``astra.spoken``,
    even once the held generate is released. History gets ``(QUESTION, "[interrupted]")``; the line says
    ``ended_by: "stopped"``."""
    async with mic_rig(tmp_path, script=[answer(LONG_ANSWER)]) as rig:
        await rig.advance(400, step=40)
        rig.llm.hold()
        await rig.type_question(QUESTION)
        await rig.until(lambda: rig.state.state is TurnState.THINKING, "premise: THINKING", step=20)
        await rig.advance(40, step=20)
        aborted = len(rig.llm.aborted)
        plays = len(rig.audio.plays)
        pub_mark = len(rig.published.messages)
        since = await stop(rig, "t1")
        assert rig.state.state is TurnState.IDLE, (
            f"T-0193 A0-5: THINKING -> IDLE in the stop's step; {rig.state.state}"
        )
        assert len(rig.llm.aborted) == aborted + 1, (
            f"A0-7: the live generate is aborted once; {rig.llm.aborted}"
        )
        rig.llm.release()
        await rig.advance(2_000, step=40)
        assert rig.audio.plays[plays:] == [], (
            f"nothing plays after a THINKING stop; {rig.audio.plays[plays:]}"
        )
        assert (
            rig.commands_of("play_clip", since) == [] and rig.commands_of("tts.speak", since) == []
        )
        assert [m for m in rig.published.messages[pub_mark:] if m.type == "astra.spoken"] == []
        assert history(rig) == [(QUESTION, "[interrupted]")], history(rig)
        assert len(rig.llm.aborted) == aborted + 1, f"A0-7: exactly one abort; {rig.llm.aborted}"
    assert line(rig, "t1")["extras"].get("ended_by") == "stopped"
