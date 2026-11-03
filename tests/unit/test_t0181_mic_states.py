"""T-0181 group A, A7 (as replanned), A0-S5, A8, A10 and A11: what a mute does in each state, what it leaves alone,
what it writes down, and what it never touches.

docs/tasks/T-0181.md: A7 (the "## Group A0 replan" table replaces the original), A0-S5, A8, A10, A11 (the replan's first
bullet), A-R2, A-R3, A-R11, A-R12 ("Tests to write -> Group A -> test_t0181_mic_states.py").

The rig is ``tests/unit/_t0181_rig.mic_rig`` (the session over the fakes, tools off, a socketless ``HudServer`` with one
greeted page, ``FakeAudioIO`` capture, a reactive ``FakeStt``). Mutes go over the wire as raw JSON; before group A
lands no frame is ever zeroed and the rows fail on ``engage``'s named A5 assertion or on their first own assertion,
never at collection.

**The A7 rows** (one parametrised row, ``test_a7_from_each_state_the_mute_does_its_row_and_the_session_reaches_idle``):
the mute engages in the named state; the machine is stepped with exactly one ``MicMuted``, from that state to the state
the table names (IDLE for an abandon, unchanged otherwise); then, with the fake LLM and TTS completing, the session is
IDLE with the expected number of turn lines within 15 s of virtual time from the mute (A0-S5), and the turns asked
are exactly the expected transcripts. One more second must open nothing new. Hand-timed scenarios:

- *voice question* ``what quokka narwhal zebra`` at audio 1000 ms, 200 ms a word, on a recogniser with steps (so the
  endpointer can end the run), released ~500 ms behind its audio;
- *typed question* ``tell me about the harbour`` (``LONG_ANSWER``, about five seconds of speech);
- *barge-in* ``wait stop that please`` from audio N, 400 ms into the answer, its onset submitted by the row (A-R12);
  the takeover is at N + 600. PAUSED mutes at N + 300; INTERRUPTED holds the playback terminal (``hold_events``) so the
  machine sits in INTERRUPTED, mutes there, then releases it. The relistened turn is asked with exactly the barge-in
  words whose audio starts before the first zeroed frame (A6, H-T181-9); no other barge-in word reaches any
  ``llm.start`` or any history entry.

**Readings** (listed in the report):

- A7 "IDLE: nothing opens" is checked with a content-blind word scripted inside the span;
- A7 "SPEAKING: plays to its drain, no PAUSED": a content-blind word scripted inside the span while the answer plays
  must not pause it (no onset after the mute, PAUSED never entered), and the answer's history entry is the whole
  answer;
- A8 "its bodies and turn line are byte-identical apart from A10's key": the request bodies are compared as the fake
  LLM received them (every turn ``Prompt``, dumped), and the turn lines as raw JSON with ``extras.mic_muted`` removed
  and the per-session ``session_id``/``session_log`` keys removed wherever they occur (each arm has its own log
  directory); both arms run the same virtual timeline (the muted arm's mute takes no virtual time);
- A10's ``mic.gate`` record is a ``note`` with data keys exactly ``{what, muted, frame_t_ms}`` on a mute and
  ``{what, muted, frame_t_ms, words_dropped}`` on an unmute (A-R2), ``frame_t_ms`` the first frame handled under the
  new state (the span's first zeroed frame; the first live frame after it); "the clock is named in a comment" is
  review-only (A-R12), as is "no duration is computed from a session time and an audio time";
- A10's ``mic_muted`` is an ``extras`` key (A-R11): ``true`` on a turn muted at any moment (muted throughout, or muted
  and unmuted while it ran), absent (not ``false``) on every other turn;
- A11's pins: ``turn_machine.py`` equals ``test_session_loop.TURN_MACHINE_SHA256`` (A0's re-pin), and
  ``contracts/commands.py``, ``turn.py``, ``codec.py`` and ``events.py`` equal their blobs at BASE 74ae8b25 (A0 merged
  before BASE, so group A changes none of them), pinned as sha256 literals (content, not a ref; A-R12: they become a
  closed range at merge if the merger prefers). "No new socket" is the differential module's row, not this one.

Regime: CPU over fakes, ``ScriptedClock``, ``TurnPolicy(thinking=True)`` (the ack clip on, as ``TurnPolicy()``), tools
off, ``RecordingTts`` 200 ms a word, ``FakeAudioIO`` capture, reactive ``FakeStt`` (``lag_ms`` 500), n = 1,
deterministic per row. No measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.unit._t0181_rig import (
    BARGE,
    GATE_NOTE,
    LONG_ANSWER,
    QUESTION,
    MicRig,
    barge_in,
    mic_rig,
    reactive_stt,
    zero_spans,
)
from tests.unit._tool_turn_support import answer
from tests.unit.test_session_loop import REPO, TURN_MACHINE_SHA256

VOICE = "what quokka narwhal zebra"
V_START = 1000
ANSWER = "A shop. "
FRAME_MS = 80
#: A0-S5: from the mute to IDLE, virtual time
LIVENESS_MS = 15_000

#: A11: sha256 of each file's blob at BASE 74ae8b25 (``git show 74ae8b25:<path> | sha256sum``, 2026-10-28); content
#: pins, so they hold whatever later ref is checked out
BASE_SHA256 = {
    "local_astra/contracts/commands.py": "776033c56d60eb06a2cab6560d2a10c7ed5d8943835c23e92907cb5026220057",
    "local_astra/contracts/turn.py": "9a8f98131bd905f856a36ab8a972ad1f89b6d4f4a7da3ef15558a291da2e3d58",
    "local_astra/contracts/codec.py": "ec10fa7c84ec4845197321c37111b91a65ec263cc2fa2228baf9ff6fb3f92338",
    "local_astra/contracts/events.py": "76aabfefc8faad5b7ee772f8a3ca6e5024a735aaa4a1d45fb42386acb4996934",
}


def moved(rig: MicRig) -> list[tuple[TurnState, TurnState]]:
    return [(b, a) for b, ev, a in rig.machine.transitions() if ev.type == "mic.muted"]


def starts(rig: MicRig) -> list[str]:
    return [c.transcript for c in rig.commands_of("llm.start")]


async def voice_question(rig: MicRig) -> list[Any]:
    spoken = rig.say(VOICE, V_START)
    await rig.advance(400, step=40)
    return spoken


# =====================================================================================================
# A7 / A0-S5: each state
# =====================================================================================================


@dataclass
class Muted:
    """What one A7 driver did: when the mute came, the span's first frame, and what the row must then see."""

    muted_at: int
    first_zero: int
    moved: tuple[TurnState, TurnState]
    lines: int
    starts: list[str]
    #: scripted words that must reach no llm.start and no history entry
    dropped: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] | None = None
    notes: dict[str, Any] = field(default_factory=dict)


async def mute_now(rig: MicRig) -> tuple[int, int]:
    at = rig.clock.now_ms
    return at, await rig.engage()


async def from_idle(rig: MicRig) -> Muted:
    await rig.advance(400, step=40)
    at, first_zero = await mute_now(rig)
    (word,) = rig.say("pelican", first_zero + 160)
    return Muted(at, first_zero, (TurnState.IDLE, TurnState.IDLE), 0, [], dropped=[word.text])


async def from_listening(rig: MicRig) -> Muted:
    await voice_question(rig)
    await rig.until(
        lambda: len(rig.state.words) >= 2,
        "premise: the machine never held two words",
        limit_ms=4_000,
        step=40,
    )
    assert rig.state.state is TurnState.LISTENING, f"premise: LISTENING; {rig.state.state}"
    at, first_zero = await mute_now(rig)
    return Muted(
        at, first_zero, (TurnState.LISTENING, TurnState.IDLE), 1, [], dropped=VOICE.split()
    )


async def from_endpointing_before_flush(rig: MicRig) -> Muted:
    await voice_question(rig)
    await rig.until(
        lambda: rig.state.state is TurnState.ENDPOINTING and rig.state.flush_t_ms is None,
        "premise: never in ENDPOINTING before the flush",
        limit_ms=6_000,
        step=20,
    )
    at, first_zero = await mute_now(rig)
    return Muted(
        at, first_zero, (TurnState.ENDPOINTING, TurnState.IDLE), 1, [], dropped=VOICE.split()
    )


async def from_endpointing_after_flush(rig: MicRig) -> Muted:
    await voice_question(rig)
    await rig.until(
        lambda: rig.state.state is TurnState.ENDPOINTING and rig.state.flush_t_ms is not None,
        "premise: never in ENDPOINTING after the flush",
        limit_ms=6_000,
        step=10,
    )
    at = rig.clock.now_ms
    rig.send_mute(True)
    await rig.pump()
    # the flush window is eot_settle_ms (120) plus one tick: read the step now, before engage advances the clock
    assert moved(rig) == [(TurnState.ENDPOINTING, TurnState.ENDPOINTING)], (
        f"A0-3/H-T181-8: one MicMuted in ENDPOINTING after the flush, no row; got {moved(rig)}"
    )
    first_zero = await rig.engage()
    return Muted(
        at,
        first_zero,
        (TurnState.ENDPOINTING, TurnState.ENDPOINTING),
        1,
        [VOICE],
        history=[(VOICE, ANSWER)],
    )


async def from_typed(rig: MicRig) -> Muted:
    await rig.advance(400, step=40)
    await rig.type_question("what is quokka")
    assert rig.state.state is TurnState.ENDPOINTING, f"premise: {rig.state.state}"
    at = rig.clock.now_ms
    rig.send_mute(True)
    await rig.pump()
    first_zero = await rig.engage()
    return Muted(
        at,
        first_zero,
        (TurnState.ENDPOINTING, TurnState.ENDPOINTING),
        1,
        ["what is quokka"],
        history=[("what is quokka", ANSWER)],
    )


async def from_thinking(rig: MicRig) -> Muted:
    rig.llm.hold()
    await voice_question(rig)
    await rig.until(
        lambda: rig.state.state is TurnState.THINKING,
        "premise: never THINKING",
        limit_ms=6_000,
        step=40,
    )
    at, first_zero = await mute_now(rig)
    await rig.advance(200, step=40)
    rig.llm.release()
    return Muted(
        at,
        first_zero,
        (TurnState.THINKING, TurnState.THINKING),
        1,
        [VOICE],
        history=[(VOICE, ANSWER)],
    )


async def from_speaking(rig: MicRig) -> Muted:
    await rig.advance(400, step=40)
    await rig.type_question(QUESTION)
    await rig.until(
        lambda: rig.state.state is TurnState.SPEAKING,
        "premise: never SPEAKING",
        limit_ms=6_000,
        step=40,
    )
    await rig.advance(400, step=40)
    at, first_zero = await mute_now(rig)
    (word,) = rig.say("pelican", first_zero + 160)
    return Muted(
        at,
        first_zero,
        (TurnState.SPEAKING, TurnState.SPEAKING),
        1,
        [QUESTION],
        dropped=[word.text],
        history=[(QUESTION, LONG_ANSWER)],
        notes={"never_paused_after": len(rig.machine.log)},
    )


async def from_paused(rig: MicRig) -> Muted:
    spoken = await barge_in(rig, BARGE, question=QUESTION)
    await rig.advance(300, step=20)
    assert rig.state.state is TurnState.PAUSED, f"premise: {rig.state.state}"
    at, first_zero = await mute_now(rig)
    kept = [w.text for w in spoken if w.start_ms < first_zero]
    dropped = [w.text for w in spoken if w.start_ms >= first_zero]
    assert kept and dropped, f"premise: the span splits the barge-in; {first_zero}, {spoken}"
    return Muted(
        at,
        first_zero,
        (TurnState.PAUSED, TurnState.PAUSED),
        2,
        [QUESTION, " ".join(kept)],
        dropped=dropped,
    )


async def from_interrupted(rig: MicRig) -> Muted:
    spoken = await barge_in(rig, BARGE, question=QUESTION)
    rig.audio.hold_events()
    await rig.until(
        lambda: rig.state.state is TurnState.INTERRUPTED,
        "premise: the takeover never interrupted",
        limit_ms=1_000,
        step=20,
    )
    at = rig.clock.now_ms
    rig.send_mute(True)
    await rig.pump()
    assert moved(rig) == [(TurnState.INTERRUPTED, TurnState.INTERRUPTED)], (
        f"A0-3/H-T181-9: one MicMuted in INTERRUPTED, no row; got {moved(rig)}"
    )
    rig.audio.release_events()
    first_zero = await rig.engage()
    kept = [w.text for w in spoken if w.start_ms < first_zero]
    dropped = [w.text for w in spoken if w.start_ms >= first_zero]
    assert kept and dropped, f"premise: the span splits the barge-in; {first_zero}, {spoken}"
    return Muted(
        at,
        first_zero,
        (TurnState.INTERRUPTED, TurnState.INTERRUPTED),
        2,
        [QUESTION, " ".join(kept)],
        dropped=dropped,
    )


Driver = Callable[[MicRig], Awaitable[Muted]]

#: (driver, the recogniser sends steps, the LLM script)
STATES: list[Any] = [
    pytest.param(from_idle, False, ANSWER, id="IDLE"),
    pytest.param(from_listening, True, ANSWER, id="LISTENING"),
    pytest.param(from_endpointing_before_flush, True, ANSWER, id="ENDPOINTING-before-flush"),
    pytest.param(from_endpointing_after_flush, True, ANSWER, id="ENDPOINTING-after-flush"),
    pytest.param(from_typed, True, ANSWER, id="typed"),
    pytest.param(from_thinking, True, ANSWER, id="THINKING"),
    pytest.param(from_speaking, False, LONG_ANSWER, id="SPEAKING"),
    pytest.param(from_paused, True, LONG_ANSWER, id="PAUSED"),
    pytest.param(from_interrupted, True, LONG_ANSWER, id="INTERRUPTED"),
]


@pytest.mark.parametrize(("drive", "steps", "reply"), STATES)
async def test_a7_from_each_state_the_mute_does_its_row_and_the_session_reaches_idle(
    tmp_path: Path, drive: Driver, steps: bool, reply: str
) -> None:
    """A7 (the replan's table) and A0-S5: one ``MicMuted`` from the named state to the table's state; IDLE with the
    expected turn lines within 15 s of the mute (the fakes completing), exactly the expected ``llm.start``
    transcripts, no dropped word in any ``llm.start`` or history entry, and nothing new one second later. The SPEAKING
    row also requires no onset and no PAUSED after the mute (the span word must not pause the answer)."""
    async with mic_rig(tmp_path, stt=reactive_stt(steps=steps), script=[answer(reply)]) as rig:
        want = await drive(rig)
        assert moved(rig) == [want.moved], (
            f"A7/A0: one MicMuted, {want.moved[0].name} -> {want.moved[1].name}; got {moved(rig)}"
        )
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= want.lines,
            f"A0-S5: not IDLE with {want.lines} turn line(s) within {LIVENESS_MS} ms of the mute",
            limit_ms=max(0, LIVENESS_MS - (rig.clock.now_ms - want.muted_at)),
        )
        await rig.advance(1_000)
        got = starts(rig)
        assert len(rig.lines()) == want.lines and rig.state.state is TurnState.IDLE, (
            f"A7: {len(rig.lines())} turn lines and {rig.state.state} one second after IDLE; want {want.lines}"
        )
        assert got == want.starts, f"A7: llm.start transcripts {got}; want {want.starts}"
        history = [(h.user_text, h.assistant_spoken) for h in rig.state.history]
        if want.history is not None:
            assert history == want.history, f"A7: history {history}; want {want.history}"
        said = " ".join([*got, *(u for u, _ in history)]).split()
        leaked = [w for w in want.dropped if w in said]
        assert not leaked, f"A6/A7: muted-span words {leaked} joined a turn; {got}, {history}"
        if "never_paused_after" in want.notes:
            since = want.notes["never_paused_after"]
            onsets = rig.stepped("user.speech_onset", since)
            paused = [a for _, _, a in rig.machine.transitions()[since:] if a is TurnState.PAUSED]
            assert (onsets, paused) == ([], []), (
                f"A7 SPEAKING: the span word paused the answer; onsets {onsets}, PAUSED {len(paused)}"
            )
            assert rig.stepped("user.takeover", since) == []


async def test_an_endpointing_before_flush_abandon_publishes_one_empty_final_and_ends_by_mic_muted(
    tmp_path: Path,
) -> None:
    """A0-S4 for the ENDPOINTING-before-flush abandon (A-RV3; A0-2's table: ENDPOINTING before the flush is an
    abandon, like LISTENING): the page gets exactly one final ``astra.transcript`` for ``t1`` and its text is empty
    (never a final carrying the question's words), and ``t1``'s turn line carries ``ended_by: "mic_muted"``. A session
    that recognises the abandon from LISTENING only finalises the turn as an ordinary one here."""
    async with mic_rig(tmp_path, stt=reactive_stt(steps=True), script=[answer(ANSWER)]) as rig:
        want = await from_endpointing_before_flush(rig)
        assert moved(rig) == [want.moved], f"premise: ENDPOINTING -> IDLE; {moved(rig)}"
        await rig.advance(3_000)
        finals = [m.text for m in rig.transcripts("t1") if m.final]
        lines = rig.raw_lines()
    assert finals == [""], (
        f"A0-S4: exactly one empty final astra.transcript for t1; finals {finals}"
    )
    assert [line["turn_id"] for line in lines] == ["t1"], [line["turn_id"] for line in lines]
    assert lines[0].get("extras", {}).get("ended_by") == "mic_muted", (
        f"A0-S4: ended_by 'mic_muted'; got extras {lines[0].get('extras')}"
    )


# =====================================================================================================
# A8: output and typed input are unaffected
# =====================================================================================================


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if k not in ("session_id", "session_log")}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


async def typed_arm(tmp_path: Path, *, mute: bool) -> dict[str, Any]:
    """A typed question answered with ``LONG_ANSWER``; with ``mute`` the switch is turned on (taking no virtual time)
    400 ms in and the question typed 200 ms later. Both arms run the same virtual timeline."""
    async with mic_rig(tmp_path, script=[answer(LONG_ANSWER)]) as rig:
        await rig.advance(400, step=40)
        mark = len(rig.rec_stt.fed)
        if mute:
            rig.send_mute(True)
        await rig.pump()
        await rig.advance(200, step=40)
        await rig.type_question(QUESTION)
        await rig.turn_done()
        await rig.advance(400, step=40)
        return {
            "zeroed": rig.rec_stt.zero()[mark:],
            "plays": [(source, getattr(u, "text", None)) for u, source in rig.audio.plays],
            "clips": len(rig.audio.clip_plays()),
            "prompts": [p.model_dump(mode="json") for _, p in rig.llm.gen_prompts],
            "lines": rig.raw_lines(),
            "history": [(h.user_text, h.assistant_spoken) for h in rig.state.history],
        }


async def test_a8_while_muted_the_ack_clip_and_every_utterance_still_play(tmp_path: Path) -> None:
    """A8: muted before a typed question, ``audio.play`` gets exactly what it gets unmuted: the ack clip and every
    utterance of the answer, in order (the muted arm's gate premise is that the frames fed after the mute are zeros)."""
    (tmp_path / "live").mkdir()
    (tmp_path / "muted").mkdir()
    live = await typed_arm(tmp_path / "live", mute=False)
    muted = await typed_arm(tmp_path / "muted", mute=True)
    assert live["clips"] >= 1 and live["history"] == [(QUESTION, LONG_ANSWER)], (
        f"premise: unmuted, the ack clip plays and the answer is spoken; {live['clips']}, {live['history']}"
    )
    assert muted["zeroed"] and all(muted["zeroed"]), (
        f"A5: the frames fed after the mute are zeros; {muted['zeroed'].count(False)} of "
        f"{len(muted['zeroed'])} were fed live"
    )
    assert muted["plays"] == live["plays"], (
        f"A8: while muted every clip and utterance plays as unmuted; {muted['plays']} != {live['plays']}"
    )
    assert muted["history"] == live["history"], (muted["history"], live["history"])


async def test_a8_a_typed_turn_muted_is_the_unmuted_turn_plus_the_key(tmp_path: Path) -> None:
    """A8 (and A10's key): the muted arm's request prompts equal the unmuted arm's, and its turn line equals the
    unmuted line once ``extras.mic_muted`` (``true`` in the muted arm, absent in the other) and the per-session
    ``session_id``/``session_log`` keys are removed."""
    (tmp_path / "live").mkdir()
    (tmp_path / "muted").mkdir()
    live = await typed_arm(tmp_path / "live", mute=False)
    muted = await typed_arm(tmp_path / "muted", mute=True)
    assert muted["prompts"] == live["prompts"], "A8: the typed turn's requests are as unmuted"
    assert len(live["lines"]) == len(muted["lines"]) == 1, (live["lines"], muted["lines"])
    live_line, muted_line = (_scrub(live["lines"][0]), _scrub(muted["lines"][0]))
    assert "mic_muted" not in live_line.get("extras", {}), live_line.get("extras")
    key = muted_line.get("extras", {}).pop("mic_muted", None)
    assert muted_line == live_line, (
        "A8: the muted turn line differs from the unmuted one beyond extras.mic_muted: "
        f"{sorted(k for k in set(muted_line) | set(live_line) if muted_line.get(k) != live_line.get(k))}"
    )
    assert key is True, (
        f"A10: a turn muted at any moment carries extras.mic_muted: true; got {key!r}"
    )
    assert muted["zeroed"] and all(muted["zeroed"]), "A5: the muted arm's frames are zeros"


# =====================================================================================================
# A10: the records and the turn-line key
# =====================================================================================================


async def test_a10_one_gate_record_per_change_at_the_first_frame_with_exact_keys(
    tmp_path: Path,
) -> None:
    """A10 / A-R2: over true, true (redundant), false, false (redundant), true, false, 400 ms apart, the log holds
    exactly four ``mic.gate`` notes, one per change, in order. A mute's data is exactly ``{what, muted: true,
    frame_t_ms}``, an unmute's exactly ``{what, muted: false, frame_t_ms, words_dropped: 0}`` (no words were spoken);
    ``frame_t_ms`` is the first frame handled under the new state: each zero span's first ``t_ms``, and the frame
    after its last (last + 80)."""
    async with mic_rig(tmp_path) as rig:
        await rig.advance(400, step=40)
        for muted in (True, True, False, False, True, False):
            await rig.mute(muted)
            await rig.advance(400, step=40)
        spans = zero_spans(rig.rec_stt)
    gates = rig.notes(GATE_NOTE)
    assert len(spans) == 2, f"premise/A5: two muted spans were fed; {spans}"
    want: list[dict[str, Any]] = []
    for first, last in spans:
        want.append({"what": GATE_NOTE, "muted": True, "frame_t_ms": first})
        want.append(
            {"what": GATE_NOTE, "muted": False, "frame_t_ms": last + FRAME_MS, "words_dropped": 0}
        )
    assert gates == want, f"A10: the mic.gate notes; got {gates}, want {want}"
    others = [
        r
        for r in rig.log_records
        if r.get("kind") != "note" and (r.get("data") or {}).get("what") == GATE_NOTE
    ]
    assert others == [] and rig.kind(GATE_NOTE) == [], "A-R2: mic.gate is a note, nothing else"


async def test_a10_mic_muted_is_on_muted_turns_only(tmp_path: Path) -> None:
    """A10 / A-R11: four typed turns. t1 unmuted; t2 asked and answered while muted; t3 after the unmute; t4 muted and
    unmuted while it is THINKING (the LLM held). ``extras.mic_muted`` is ``true`` on t2 and t4 and absent (the key
    itself) on t1 and t3."""
    async with mic_rig(tmp_path, script=[answer(ANSWER)]) as rig:
        await rig.advance(400, step=40)

        async def ask(text: str) -> None:
            await rig.type_question(text)
            await rig.turn_done()
            await rig.advance(300, step=40)

        await ask("first question")
        await rig.engage()
        await ask("second question")
        await rig.release()
        await ask("third question")
        rig.llm.hold()
        await rig.type_question("fourth question")
        await rig.until(lambda: rig.state.state is TurnState.THINKING, "premise: THINKING")
        await rig.engage()
        await rig.release()
        rig.llm.release()
        await rig.turn_done()
        await rig.advance(300, step=40)
        lines = rig.raw_lines()
    assert [line["turn_id"] for line in lines] == ["t1", "t2", "t3", "t4"], lines
    got = {line["turn_id"]: line.get("extras", {}).get("mic_muted", "absent") for line in lines}
    assert got == {"t1": "absent", "t2": True, "t3": "absent", "t4": True}, (
        f"A10: mic_muted: true only on turns muted at any moment, absent otherwise; got {got}"
    )


# =====================================================================================================
# A11: negatives
# =====================================================================================================


def stepped_types(rig: MicRig) -> list[str]:
    return [ev.type for ev, _, _ in rig.machine.log]


async def two_turns(tmp_path: Path, *, mute: bool) -> list[str]:
    """A voice question, then (muted, in the muted arm) a typed one; the event types the machine was stepped with."""
    async with mic_rig(tmp_path, stt=reactive_stt(steps=True), script=[answer(ANSWER)]) as rig:
        await voice_question(rig)
        await rig.turn_done()
        await rig.advance(400, step=40)
        if mute:
            await rig.engage()
        await rig.type_question(QUESTION)
        await rig.turn_done()
        if mute:
            await rig.release()
        await rig.advance(400, step=40)
        return stepped_types(rig)


async def test_a11_the_machine_files_are_pinned_and_only_mic_muted_is_new(tmp_path: Path) -> None:
    """A11 (as replanned): ``turn_machine.py`` is A0's re-pinned content (``TURN_MACHINE_SHA256``) and
    ``contracts/commands.py``, ``turn.py``, ``codec.py`` and ``events.py`` are byte-identical to BASE 74ae8b25. With the
    mute never toggled no ``MicMuted`` is stepped; with one mute exactly one is, and it is the only event type the
    muted arm steps that the unmuted arm does not; every type stepped is in ``EVENT_TYPES``."""
    # T-0192 B re-pin (B-RV6, both ends pinned): the machine, events.py and commands.py are these pins at T-0192 B's
    # literal base da29ddbe (the base end), and in the working tree exactly the merged 97409910 blobs (the tree end)
    from tests.unit.test_t0192_hand_back_machine import (
        T0192_BASE,
        T0192_BASE_SHA256,
        T0192_MACHINE,
        T0192_MOVED_SHA256,
        t0192_end_sha256,
    )

    # T-0193 A0 re-pin (A0-9; A0-R5, both ends pinned): the merged 97409910 blobs (and turn.py's A11 pin) are the
    # files at T-0193 A0's literal base fdb9fc01 (the base end), and the working tree is exactly each merged 373df531
    # blob (the tree end)
    from tests.unit.test_t0193_stop_row import T0193_BASE, T0193_BASE_SHA256, T0193_MOVED_SHA256

    # T-0216 re-pin: R-1: codec.py's two ends (literals in the T-0216 module)
    from tests.unit.test_t0216_hud_receive import (
        T0216_BASE,
        T0216_BASE_SHA256,
        T0216_CODEC,
        T0216_MOVED_SHA256,
        T0216C_BASE,
        T0216C_TREE_ENDS,
    )

    assert T0193_MOVED_SHA256[T0192_MACHINE] == TURN_MACHINE_SHA256
    for rel, digest in T0192_MOVED_SHA256.items():
        assert T0193_BASE_SHA256[rel] == digest
        got = t0192_end_sha256(rel, T0193_BASE)
        assert got == digest, (
            f"A11/A0-7 / T-0192 B: {rel} at {T0193_BASE[:8]} is not the merged 97409910 blob ({got} != {digest})"
        )
    for rel, digest in T0193_MOVED_SHA256.items():
        got = t0192_end_sha256(rel)
        assert got == digest, (
            f"T-0193 A0-R5: {rel} is not the merged 373df531 blob ({got} != {digest})"
        )
    for rel, digest in T0192_BASE_SHA256.items():
        assert BASE_SHA256.get(rel, digest) == digest, f"T-0192 B: {rel}'s base end is the A11 pin"
        assert t0192_end_sha256(rel, T0192_BASE) == digest, (
            f"T-0192 B: {rel} at da29ddbe is not its base end"
        )
    for rel, digest in BASE_SHA256.items():
        if rel in T0192_BASE_SHA256:
            continue
        if rel in T0193_BASE_SHA256:
            # T-0193 A0: turn.py's A11 pin is its content at fdb9fc01 (the base end); the tree end is the literal above
            assert (
                T0193_BASE_SHA256[rel] == digest and t0192_end_sha256(rel, T0193_BASE) == digest
            ), rel
            continue
        if rel == T0216_CODEC:
            # T-0216 re-pin: R-1 (both ends pinned, the T-0193 A0 way): codec.py's A11 pin is its content at T-0216's
            # literal base 423362b7 (the base end); the tree end is exactly the e705e79b blob
            assert digest == T0216_BASE_SHA256, (
                f"T-0216 re-pin: R-1: {rel}'s base end is the A11 pin"
            )
            assert t0192_end_sha256(rel, T0216_BASE) == digest, (
                f"T-0216 re-pin: R-1: {rel} at {T0216_BASE[:8]} is not its A11 pin"
            )
            # T-0216c re-pin: R-1 (both ends pinned): the e705e79b blob is codec.py at T-0216c's literal base 7489b9ed
            # (the base end); the working tree is one of ``T0216C_TREE_ENDS`` (the tree end; see its comment)
            assert t0192_end_sha256(rel, T0216C_BASE) == T0216_MOVED_SHA256, (
                f"T-0216c re-pin: R-1: {rel} at {T0216C_BASE[:8]} is not the e705e79b blob"
            )
            got = t0192_end_sha256(rel)
            assert got in T0216C_TREE_ENDS, (
                f"T-0216c re-pin: R-1: {rel} is none of the pinned tree ends ({got} not in {T0216C_TREE_ENDS})"
            )
            continue
        got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        assert got == digest, f"A11: {rel} is byte-identical to BASE 74ae8b25 ({got} != {digest})"
    (tmp_path / "live").mkdir()
    (tmp_path / "muted").mkdir()
    live = await two_turns(tmp_path / "live", mute=False)
    assert "mic.muted" not in live, "A11: with the mute never toggled, no MicMuted is stepped"
    muted = await two_turns(tmp_path / "muted", mute=True)
    assert muted.count("mic.muted") == 1, (
        f"A0-S1: one mute steps one MicMuted; {muted.count('mic.muted')}"
    )
    assert set(muted) - set(live) == {"mic.muted"}, (
        f"A11: the only new event type is mic.muted; {set(muted) - set(live)}"
    )
    known = {cls.model_fields["type"].default for cls in E.EVENT_TYPES}
    assert "mic.muted" in known and set(muted) <= known, set(muted) - known
