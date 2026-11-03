"""Egress: complete, ordered, typed, derived — and never the thinking (T-0028 criteria 9-12).

No socket here. The session's HUD egress is one injected publisher (docs/02 §4 spec delta (b)),
so everything the page would see is observable by handing the session a recorder. The seams are
fixed in ``tests/unit/_hud_rig.py``'s module docstring; in one line:

    Session(..., publish: Callable[[AstraMessage], None] | None = None)   # sync, never awaited
    Session.set_publisher(publish)                                        # what HudServer calls

The oracle for "what should be published" is the **turn machine's own command log** (``HudState``
→ ``astra.turn.state``, ``TtsSpeak`` → ``astra.spoken``, ``HudError`` → ``astra.error``) plus the
transcript the machine is holding — never the session's egress code. The machine is not this
task's file and is byte-identical through it (T-0007 ruling A6).

Regime: no measurement. ``ScriptedClock``, ``tests/fakes/``, Hypothesis derandomised
(the task asks for ``--hypothesis-seed=0``; ``derandomize=True`` pins the same thing from inside
the file, so the property is reproducible under the plain fast-suite command).
"""

from __future__ import annotations

import asyncio
import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.conductor.speech_gate import normalise_for_tts
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.codec import decode, encode
from local_astra.contracts.llm import Commit
from local_astra.contracts.protocol import (
    SERVER_TO_CLIENT,
    AstraError,
    AstraSpoken,
    AstraTranscript,
    AstraTurnState,
    AstraUiSurface,
)
from local_astra.contracts.turn import TurnState
from tests.unit._hud_rig import (
    UI_CANVAS,
    PushStt,
    Recorder,
    answer,
    frame,
    make_rig,
    require_symbol,
    running,
    thought,
    ui,
)

QUESTION = "show me the surface"

#: Hand-computed from ``QUESTION``: one partial per STT word, each the transcript so far
#: (``turn_machine._transcript`` joins the buffered words with single spaces).
PARTIALS = ["show", "show me", "show me the", "show me the surface"]

#: The states a clean turn passes through, read off docs/02 §4's rows: IDLE → LISTENING on the
#: onset, → ENDPOINTING on stt.end_of_turn, → THINKING on the settle tick, → SPEAKING on the
#: first answer commit, → IDLE on playback.drained ∧ llm.done.
STATES = ["LISTENING", "ENDPOINTING", "THINKING", "SPEAKING", "IDLE"]

#: Ruling A1's discriminating row. ``speech_gate.normalise_for_tts`` rewrites a file path, so the
#: committed ``source`` and the ``text`` handed to TTS differ; the expected rendering below is the
#: gate's own answer (an independent oracle — the gate is not this task's file and is consumed
#: unchanged), computed once at import.
PATH_SENTENCE = "The error is in /etc/hosts. "
PATH_SPOKEN = normalise_for_tts(PATH_SENTENCE)


def squashed(states: list[str]) -> list[str]:
    """Consecutive duplicates removed: the meter may repeat a state, the *changes* are the rows."""
    out: list[str] = []
    for state in states:
        if not out or out[-1] != state:
            out.append(state)
    return out


async def test_one_turn_produces_the_full_ordered_egress(tmp_path):
    """Criterion 9: state, transcript, spoken, surface — ordered, typed, nothing invented.

    The script is two answer sentences and one valid A2UI canvas, so by hand: five state changes,
    four partial transcripts plus one final, two ``astra.spoken`` (one per ``tts.speak``), one
    surface id ``turn-t1`` (``UiGate``'s, T-0010 A3/A7) and no error.

    The second sentence is the **discriminating row** ruling A1 asks for: ``/etc/hosts`` is
    rewritten by ``speech_gate.normalise_for_tts``, so ``TtsSpeak.source`` and ``TtsSpeak.text``
    differ and the assertion below can tell which of the two reached the wire. A1: it is the
    normalised ``text`` — what was *handed to TTS* (``AstraSpoken``'s own docstring); ``source``
    stays off the wire until T-0008b.
    """
    published = Recorder()
    script = [answer("Here it is. "), answer(PATH_SENTENCE), ui(canvas_idx=1)]
    async with running(
        tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=100, publish=published
    ) as rig:
        await rig.ask(QUESTION)
        await rig.speak_out()

        assert published.messages, (
            "criterion 9: a whole turn published nothing. The session's HUD egress is the "
            "injected publisher (docs/02 §4 spec delta (b)); today _gate_surface computes "
            "result.surface and drops it and there is no transcript/spoken publication at all."
        )

        # --- typing and the wire (criterion 9's negative half) -----------------------------
        for message in published.messages:
            raw = encode(message)
            assert message.type in SERVER_TO_CLIENT, (
                f"{message.type!r} is not a server→client type (contracts/protocol.py)"
            )
            assert not message.type.startswith("astra.canvas."), (
                "T-0128c A10: with no tap frames (this rig has no tap) a fake turn publishes zero "
                "astra.canvas.*; only the relay relays real tap frames: "
                f"{message.type} must never be sent here"
            )
            assert decode(raw) == message, (
                f"every egress message must re-decode under the real codec; {raw!r} did not"
            )
            if isinstance(message, AstraSpoken):
                # T-0030 ruling A9: the intent is "no field the models do not declare reaches the
                # wire", and the field this family must not grow is AstraSpoken.played_text
                # (T-0028 ruling A1, T-0030 criterion 2: the pane keeps showing the TTS-normalised
                # text). The played prefix has its own message since T-0030, so the scan is over
                # astra.spoken only -- a raw-substring scan over *every* message would forbid the
                # very message that legitimately carries it.
                assert "played_text" not in raw, (
                    "AstraSpoken carries the text handed to TTS and nothing else; the played "
                    f"prefix travels as astra.spoken.played (T-0030): {raw!r}"
                )
            if message.type == "astra.spoken.played":
                # ... and that message carries exactly its declared fields, nested ones included:
                # the same rule, checked against the contract rather than against a string.
                body = json.loads(raw)
                # `replay` joined the wire under T-0037 (ruling A2/A8): the default is serialised.
                assert set(body) == {"type", "turn_id", "played_ms", "utterances", "replay"}, (
                    f"astra.spoken.played must carry exactly its declared fields; got {body!r}"
                )
                for item in body["utterances"]:
                    assert set(item) == {
                        "utterance_id",
                        "played_text",
                        "played_words",
                        "complete",
                    }, f"PlayedUtterance must carry exactly its declared fields; got {item!r}"

        # --- turn.state (criterion 9, and one per change) -----------------------------------
        states = [m for m in published.messages if isinstance(m, AstraTurnState)]
        assert squashed([m.state for m in states]) == STATES, (
            f"criterion 9: one astra.turn.state per machine state change; got "
            f"{[m.state for m in states]}, docs/02 §4 says {STATES}"
        )
        assert {m.turn_id for m in states} == {"t1"}, (
            f"the states carry the turn they belong to; got {sorted({m.turn_id for m in states})}"
        )

        # --- transcript ---------------------------------------------------------------------
        transcripts = [m for m in published.messages if isinstance(m, AstraTranscript)]
        assert [m.text for m in transcripts if not m.final] == PARTIALS, (
            "criterion 9: one astra.transcript(final=false) per partial, each the transcript so "
            f"far; got {[m.text for m in transcripts if not m.final]}, expected {PARTIALS}"
        )
        finals = [m for m in transcripts if m.final]
        assert len(finals) == 1 and finals[0].text == QUESTION, (
            f"exactly one final=true transcript, carrying the whole turn; got {finals}"
        )

        # --- spoken: the machine's own utterance ids ------------------------------------------
        speaks = [c for c in rig.machine.commands(C.TtsSpeak) if c.text]
        spoken = [m for m in published.messages if isinstance(m, AstraSpoken)]
        assert [(m.utterance_id, m.text) for m in spoken] == [
            (c.utterance_id, c.text) for c in speaks
        ], (
            "criterion 9: one astra.spoken per utterance handed to TTS, carrying the *machine's* "
            "utterance_id (MachineState.utterance_seq — T-0008 criterion 8 keys the spoken pane "
            f"on it). The machine emitted {[(c.utterance_id, c.text) for c in speaks]}; the wire "
            f"carried {[(m.utterance_id, m.text) for m in spoken]}"
        )
        assert {m.turn_id for m in spoken} == {"t1"}
        discriminating = [c for c in speaks if c.text != c.source]
        assert discriminating, (
            "ruling A1's row is missing: no utterance had a normalised text different from its "
            f"source, so this test cannot tell them apart. The gate produced "
            f"{[(c.source, c.text) for c in speaks]}"
        )
        assert [m.text for m in spoken if m.text == PATH_SPOKEN] == [PATH_SPOKEN], (
            "ruling A1: astra.spoken carries the TTS-normalised text — "
            f"{PATH_SPOKEN!r}, not the committed source {PATH_SENTENCE!r}. The wire carried "
            f"{[m.text for m in spoken]}"
        )
        assert PATH_SENTENCE.strip() not in [m.text for m in spoken], (
            "…and the raw source is not on the wire (T-0008b owns AstraSpoken.played_text and "
            "the strike-through; nothing here sends a field or a value the models do not model)"
        )

        # --- surface -------------------------------------------------------------------------
        surfaces = [m for m in published.messages if isinstance(m, AstraUiSurface)]
        assert surfaces and {m.surface_id for m in surfaces} == {"turn-t1"}, (
            "criterion 9: the validated canvas reaches the page on the turn's one surface id "
            f"(UiGate: 'turn-<turn_id>'); got {[m.surface_id for m in surfaces]}"
        )
        assert [m.final for m in surfaces] == [False] * (len(surfaces) - 1) + [True], (
            f"final=true on the last surface of the turn and nowhere else; got "
            f"{[(m.canvas_idx, m.final) for m in surfaces]}"
        )
        assert {m.canvas_idx for m in surfaces} == {1}, (
            "the surface carries the canvas index of the commit it came from (the script's ui "
            f"commit is canvas_idx=1); got {[m.canvas_idx for m in surfaces]}"
        )

        # --- ordering ---------------------------------------------------------------------
        order = published.messages
        assert order.index(finals[0]) < order.index(spoken[0]), (
            "the user's transcript is final at end-of-turn (the freeze_after row), which is "
            "before any answer is voiced; the page must not show a partial question under a "
            "finished answer"
        )
        assert order.index(states[0]) == 0, (
            f"the first thing the page learns about a turn is that it opened; got {order[0]}"
        )
        assert isinstance(order[-1], AstraTurnState) and order[-1].state == "IDLE", (
            f"the turn ends with the state that says so; the last message was {order[-1]}"
        )
        assert not [m for m in published.messages if isinstance(m, AstraError)], (
            "a clean turn publishes no astra.error"
        )


async def test_thinking_ms_is_derived_from_llm_start_ms(tmp_path):
    """Criterion 10: the meter is ``clock.now_ms - llm_start_ms`` in THINKING, ``None`` elsewhere.

    The LLM is held so THINKING lasts 500 ms of virtual time (under ``interim_update_ms`` = 4 s,
    so no filler clip muddies the turn). No second clock and no timer may appear for the meter:
    ``ArmTimer``/``CancelTimer`` stay the session's only timers (T-0007 criterion 3), which is why
    the bound below is the *machine's* ``llm_start_ms`` and the *session's* one clock.
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("Here it is. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        rig.llm.hold()
        await rig.ask(QUESTION)
        await rig.until(
            lambda: rig.state.state is TurnState.THINKING, "the turn never reached THINKING"
        )
        llm_start = rig.state.llm_start_ms
        assert llm_start is not None
        await rig.advance(500, step=50)
        elapsed = rig.clock.now_ms - llm_start

        thinking = [
            m
            for m in published.messages
            if isinstance(m, AstraTurnState) and m.state == TurnState.THINKING
        ]
        assert thinking, "no astra.turn.state for THINKING at all (criterion 9)"
        metered = [m.thinking_ms for m in thinking]
        assert all(v is not None for v in metered), (
            f"criterion 10: thinking_ms is set while the state is THINKING; got {metered}"
        )
        assert metered == sorted(metered), (
            f"the meter counts up from llm.start; got {metered} (docs/02 §2.2's thinking meter)"
        )
        assert all(0 <= v <= elapsed for v in metered), (
            f"thinking_ms is now - llm_start_ms ({llm_start} ms), so it can never exceed the "
            f"{elapsed} ms that have passed; got {metered}"
        )
        assert max(metered) >= 400, (
            "criterion 10 / criterion 20: the meter *moves* — after 500 ms of THINKING the page "
            f"must have been told a number near 500; the largest it saw was {max(metered)}. A "
            "single turn.state per state change makes the meter a constant zero."
        )

        rig.llm.release()
        await rig.speak_out()
        outside = [
            m
            for m in published.messages
            if isinstance(m, AstraTurnState) and m.state != TurnState.THINKING
        ]
        assert outside and all(m.thinking_ms is None for m in outside), (
            "criterion 10, negative: thinking_ms is None in every state but THINKING; got "
            f"{[(m.state, m.thinking_ms) for m in outside]}"
        )


# --- criterion 11: the wire-side twin of "never store assistant thinking" ----------------------

SENTINEL = "zqxthought"


@st.composite
def commit_scripts(draw: st.DrawFn) -> list[Commit]:
    """Commit scripts whose thought segments carry tokens that appear nowhere else.

    Each thought token is ``zqxthought<i>`` — not a substring of any answer text, of the canvas,
    of the question or of any id the session mints, so finding one anywhere in the egress is
    proof it came from the thought channel and nothing else.
    """
    n = draw(st.integers(min_value=1, max_value=4))
    answers = ["Here it is. ", "The line is four. ", "It says hello. ", "Yes. "]
    script: list[Commit] = []
    for i in range(n):
        kind = draw(st.sampled_from(["thought", "answer", "ui"]))
        if kind == "thought":
            script.append(
                thought(
                    f"{SENTINEL}{i} because the screen says so. ",
                    canvas_idx=i,
                    open_=draw(st.booleans()),
                )
            )
        elif kind == "answer":
            script.append(answer(answers[i], canvas_idx=i))
        else:
            script.append(ui(canvas_idx=i))
    return script


async def _run_script(tmp_path, script: list[Commit], kept: dict | None = None) -> Recorder:
    published = Recorder()
    async with running(
        tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=100, publish=published
    ) as rig:
        await rig.ask(QUESTION)
        await rig.speak_out()
        if kept is not None:
            kept["history"] = json.dumps([h.model_dump(mode="json") for h in rig.state.history])
            kept["turns"] = "\n".join(rig.lines())
    return published


THOUGHT_TYPE = "astra.thought"


def _sentinel_outside_thought_text(raw: str) -> bool:
    """T-0180 B9: the sentinel is in ``raw`` somewhere other than an ``astra.thought``'s ``text``."""
    if SENTINEL not in raw:
        return False
    message = json.loads(raw)
    if message.get("type") != THOUGHT_TYPE:
        return True
    rest = {k: v for k, v in message.items() if k != "text"}
    return SENTINEL in json.dumps(rest)


@settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(script=commit_scripts())
def test_no_thought_token_ever_reaches_the_wire(tmp_path, script):
    """Criterion 11, narrowed by T-0180 (B9, R-1): a token that appears only in a ``thought``
    segment appears in no message except in the ``text`` of an ``astra.thought``.

    Not in ``astra.spoken``, not in ``astra.transcript``, not inside an ``astra.ui.surface``
    payload, not in an ``astra.error``, not in any other field of ``astra.thought``. The search is
    over the *encoded* messages, so a thought token hidden in a nested A2UI component or in an
    error string is caught too (CLAUDE.md: never voice or store assistant thinking; T-0007
    criterion 6 is the history-side twin). Anti-vacuity (B9): every thought token of the script
    **is** in some ``astra.thought`` text, since each is a committed thought of the one attempt;
    and none is in history or ``turns.jsonl``. The DOM half (``[data-model-thoughts]`` only) is
    group C's.
    """
    kept: dict = {}
    published = asyncio.run(_run_script(tmp_path, script, kept))
    wire = published.wire()
    offenders = [raw for raw in wire if _sentinel_outside_thought_text(raw)]
    assert not offenders, (
        "criterion 11 (narrowed by T-0180): thinking reached the HUD wire outside "
        f"astra.thought's text. The token is only ever in a thought segment of the script; it "
        f"turned up in {offenders[:2]}"
    )
    shown = " ".join(m.text for m in published.messages if m.type == THOUGHT_TYPE)
    tokens = [f"{SENTINEL}{c.canvas_idx} " for c in script if c.segment == "thought"]
    missing = [t for t in tokens if t not in shown]
    assert not missing, (
        f"T-0180 B9 anti-vacuity: the committed thought tokens {missing} are in no astra.thought "
        f"text (published types {published.types()})"
    )
    for name in ("history", "turns"):
        assert SENTINEL not in kept[name], f"T-0180 B9: a thought token reached {name}"
    if any(c.segment == "answer" for c in script):
        assert any(m.type == "astra.spoken" for m in published.messages), (
            "the property must not pass by publishing nothing: this script has an answer "
            f"commit, so something was voiced and must be on the wire. Types: "
            f"{published.types()}"
        )


# --- criterion 12: a slow client cannot slow the turn ------------------------------------------


def test_turn_state_is_the_only_droppable_kind_on_overflow():
    """Criterion 12's bounds, on the outbox itself: latest-wins for turn.state, close for the rest.

    ``Outbox(maxsize)`` is the bounded per-connection queue (see ``_hud_rig``'s docstring).
    Hand-computed: with ``maxsize=2``, three ``astra.turn.state`` messages leave the *newest*
    queued and count one drop, while a third ``astra.spoken`` cannot be dropped — losing a spoken
    line desyncs the transcript pane silently, so that connection is closed instead.
    """
    outbox_cls = require_symbol("Outbox")

    states = outbox_cls(maxsize=2)
    for i in range(3):
        assert states.offer(AstraTurnState(state=TurnState.THINKING, turn_id=f"t{i}")) is True, (
            "criterion 12: astra.turn.state is idempotent and latest-wins, so an overflow of it "
            "is a *drop*, never a closed connection"
        )
    queued = states.drain()
    assert [m.turn_id for m in queued] == ["t2"] or [m.turn_id for m in queued] == ["t1", "t2"], (
        "the newest state survives an overflow (latest-wins); the outbox holds "
        f"{[m.turn_id for m in queued]}"
    )
    assert states.dropped >= 1, (
        f"a dropped turn.state is counted, never silent; dropped={states.dropped}"
    )

    spoken = outbox_cls(maxsize=2)
    assert spoken.offer(AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="one")) is True
    assert spoken.offer(AstraSpoken(turn_id="t1", utterance_id="t1-u2", text="two")) is True
    assert spoken.offer(AstraSpoken(turn_id="t1", utterance_id="t1-u3", text="three")) is False, (
        "criterion 12: overflow of anything but turn.state closes *that* connection — the "
        "caller is told so by a False here, rather than silently losing a spoken line"
    )
    assert [m.utterance_id for m in spoken.drain()] == ["t1-u1", "t1-u2"], (
        "nothing already accepted is reordered or lost when the outbox refuses the next message"
    )


async def test_a_stalled_reader_does_not_delay_the_barge_in_pause(tmp_path):
    """Criterion 12, adversarial — the single most likely regression in this task.

    The twin of T-0007 criterion 14's ``test_a_slow_llm_client_does_not_delay_playback_pause``,
    with the slow thing on the *other* side: a HUD page that stopped reading (a backgrounded tab,
    a paused debugger). Its connection is real (``server.open()``), its outbox is never drained,
    and the TTS client is blocked at the same time. If publication ever waits for the wire — an
    ``await`` on a send inside the pump or inside a ``CommandExecutor`` handler — the
    ``user.speech_onset`` is not even read until the page comes back, and the assistant talks
    over the user for as long as a browser tab feels like it (docs/02 §4:
    ``pause_latency_budget_ms`` is 150 ms; ``.claude/rules/conductor.md``: 20 ms in the pump).
    """
    hud_server = require_symbol("HudServer")
    rig = make_rig(
        tmp_path,
        script=[answer("A long first sentence. "), answer("And a second one. ")],
        frames=[frame(0, "a")],
        ms_per_word=200,
    )
    server = hud_server(rig.session, frame_sink=_RefusingSink(), port=0)
    connection = server.open()  # no socket, and nothing will ever drain its outbox
    await rig.session.start()
    try:
        rig.tts.hold()
        await rig.ask("tell me everything")
        await rig.until(
            lambda: rig.tts.requests != [],
            "the answer never reached the (blocked) TTS client",
            limit_ms=5_000,
        )
        assert rig.state.state is TurnState.SPEAKING
        assert connection.sent(), (
            "the stalled connection must have been *offered* the turn's egress, or this test "
            "proves nothing about a slow reader"
        )
        audio_calls = len(rig.audio.calls)

        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.pump()

        hop = rig.audio.calls[audio_calls:]
        assert hop and hop[0] == "pause", (
            "criterion 12: with a stalled reader and a blocked TTS client the executor still "
            f"pauses playback first on the barge-in hop; it did {hop}"
        )
        assert rig.state.state is TurnState.PAUSED
        assert not rig.session.failures, (
            f"a stalled reader is not an error for the session: {rig.session.failures}"
        )
        rig.tts.release()
        await rig.pump()
    finally:
        await rig.session.aclose()


class _RefusingSink:
    """A frame sink for tests that post no frames: it accepts nothing and records nothing."""

    def ingest(self, message):
        return None


async def test_a_flush_delivered_by_the_stt_client_moves_the_chip_to_thinking(tmp_path):
    """Ruling A15: the ``SttFlushedEvent`` → ``events.SttFlushed`` adapter, over the stream.

    Every other row in this family submits the already-adapted ``events.SttFlushed`` straight to
    the pump, so ``Session._pump_stt``'s branch for the moshi ``Marker`` echo has never run: the
    branch could be ``pass`` and the suite would stay green. Here the flush travels the path a
    real ``MoshiSttClient`` uses — ``tests/fakes/stt.py``'s ``flushed()`` helper through
    ``SttClient.events()`` — and the observable is the one the page sees: ENDPOINTING is left for
    THINKING on the settle tick (docs/02 §4: "all words ≤ t_eot received" is what that row waits
    for), so the state chip moves.
    """
    published = Recorder()
    stt = PushStt()
    async with running(
        tmp_path,
        script=[answer("Here it is. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
        stt=stt,
    ) as rig:
        await rig.onset()
        await rig.words(QUESTION)
        await rig.advance(100, step=100)
        eot = rig.clock.now_ms - 20
        rig.submit(E.SttEndOfTurn(t_ms=rig.clock.now_ms, t_eot_ms=eot, p_end=0.9))
        await rig.pump()
        assert rig.state.state is TurnState.ENDPOINTING, (
            f"the end-of-turn row runs first; the machine is in {rig.state.state}"
        )

        stt.push(PushStt.flushed(eot))
        await rig.until(
            lambda: bool(rig.machine.events(E.SttFlushed)),
            "the SttFlushedEvent never reached the machine as events.SttFlushed — "
            "Session._pump_stt's adapter branch (ruling A15)",
            limit_ms=2_000,
        )
        await rig.advance(200, step=50)

        assert rig.state.state is not TurnState.ENDPOINTING, (
            "ruling A15: with the flush delivered, the settle tick takes ENDPOINTING to THINKING "
            f"(docs/02 §4); the machine is still in {rig.state.state}"
        )
        states = [m.state for m in published.messages if isinstance(m, AstraTurnState)]
        assert "THINKING" in states, (
            f"…and the chip is told so (criterion 9); the page saw {squashed(states)}"
        )


async def test_words_that_lag_the_end_of_turn_still_reach_the_prompt_and_the_final_transcript(
    tmp_path,
):
    """Ruling A17/A18(3): STT words arrive ~0.5 s after their audio — the wire must wait for them.

    This is the shape real speech has and none of the other rows do: ``stt.end_of_turn`` refers to
    an *audio* time, and the words whose audio is at or before it are still in flight when it
    arrives. docs/02 §4 attributes them to this turn (``start_ms <= t_eot``), so the transcript is
    only complete at the flush — which is exactly what the ENDPOINTING → THINKING row waits for.

    Publishing ``final=true`` at the freeze row instead truncates every spoken question on the
    page: the user asks "what is on this screen" and the transcript pane keeps "what is on". A2's
    ordering is still honoured — the flush is before the first ``astra.spoken`` of the turn — so
    nothing is traded away for this.
    """
    published = Recorder()
    early, late = "what is on", "this screen"
    async with running(
        tmp_path,
        script=[answer("A stack trace. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        await rig.onset()
        await rig.words(early)
        # the end-of-turn refers to audio that is 200 ms in the future of the last delivered word:
        # the tail of the utterance has been spoken but its words are still in the recogniser
        await rig.advance(100, step=100)
        eot = rig.clock.now_ms + 200
        rig.submit(E.SttEndOfTurn(t_ms=rig.clock.now_ms, t_eot_ms=eot, p_end=0.9))
        await rig.pump()
        assert rig.state.state is TurnState.ENDPOINTING

        # …and now they arrive, late, with audio times at or before t_eot (docs/02 §4)
        for index, word in enumerate(late.split()):
            await rig.advance(120, step=120)
            start = eot - 200 + index * 100
            rig.submit(
                E.SttWord(t_ms=rig.clock.now_ms, text=word, start_ms=start, end_ms=start + 100)
            )
        await rig.pump()
        rig.submit(E.SttFlushed(t_ms=rig.clock.now_ms, t_eot_ms=eot))
        await rig.pump()
        await rig.speak_out(limit_ms=10_000)

        whole = f"{early} {late}"
        starts = rig.machine.commands(C.LlmStart)
        assert [c.transcript for c in starts] == [whole], (
            "ruling A17: every word whose audio is at or before t_eot belongs to this turn, so "
            f"the prompt carries the whole question; the machine was given "
            f"{[c.transcript for c in starts]}"
        )

        transcripts = [m for m in published.messages if isinstance(m, AstraTranscript)]
        finals = [m for m in transcripts if m.final]
        assert len(finals) == 1, (
            f"criterion 9: exactly one final=true transcript per turn; got {len(finals)}"
        )
        assert finals[0].text == whole, (
            "ruling A17: the page's final transcript is the complete question, not the prefix "
            f"that had been delivered when stt.end_of_turn arrived. Expected {whole!r}, the wire "
            f"carried {finals[0].text!r}"
        )
        spoken = [m for m in published.messages if isinstance(m, AstraSpoken)]
        assert spoken, "the turn must have been answered, or the ordering below is vacuous"
        assert published.messages.index(finals[0]) < published.messages.index(spoken[0]), (
            "ruling A2, unchanged: the final transcript is still published before the first "
            "astra.spoken of the turn — the flush row is before the answer, not after it"
        )


# --- T-0054 criterion 8: the session's newest egress ---------------------------------------------


async def test_the_fence_notice_is_typed_egress_like_every_other_message(tmp_path):
    """Criterion 8/10: what the session publishes about a fence is a contract, on the wire.

    This module's rule (T-0028 criteria 9-12) is that the session's egress is *typed and derived*:
    every message is a ``contracts/protocol`` model, it round-trips through the production codec,
    and it is a server→client type. A fence notice published as a dict, or as a ``RealtimeEvent``,
    or as a message the page's direction table does not know, would be egress that no test of the
    page can ever see.

    The turn here is the one T-0054 exists for — a surface carried across two canvases — built
    with the fence fields the ``Commit`` contract has carried since T-0052.
    """
    from local_astra.contracts.protocol import AstraMessageAdapter

    script = [
        Commit(request_id="?", canvas_idx=1, segment="ui", text=UI_CANVAS, fence_close="open"),
        Commit(
            request_id="?",
            canvas_idx=2,
            segment="ui",
            text=UI_CANVAS,
            fence_close="inferred",
            fence_continued=True,
        ),
        answer("There you go. "),
    ]
    recorder = Recorder()
    async with running(
        tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=100, publish=recorder
    ) as rig:
        await rig.ask("show me the surface")
        await rig.speak_out()

    fences = [m for m in recorder.messages if getattr(m, "type", None) == "astra.ui.fence"]
    assert fences, (
        "T-0054 criteria 9 and 19: a surface reassembled from two canvases must say so on the "
        f"wire; the session published {sorted(set(recorder.types()))}"
    )
    for message in fences:
        assert message.type in set(SERVER_TO_CLIENT), (
            f"criterion 8: {message.type} is a server→client type; SERVER_TO_CLIENT holds "
            f"{sorted(SERVER_TO_CLIENT)}"
        )
        raw = encode(message)
        assert decode(raw) == message, (
            "criterion 9 of T-0028: every published message round-trips through the production "
            f"codec unchanged; {raw[:200]} did not"
        )
        assert AstraMessageAdapter.validate_json(raw) == message, (
            "criterion 8: …and is a member of the AstraMessage union, so hud_server's narrowing "
            "still type-checks with zero diff"
        )
        assert "replay" not in json.loads(raw), (
            "criterion 8: the fence notice has no replay field — it is not journalled, and a "
            f"replay flag would promise a catch-up that does not exist. Got {raw[:200]}"
        )
