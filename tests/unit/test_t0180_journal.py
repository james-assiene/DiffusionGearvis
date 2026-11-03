"""T-0180 group A: the hello-replay journal keeps the turn's thoughts (A4), the compact ``wire.out``
row (A5), and nothing else moves (A8).

docs/tasks/T-0180.md. Every row puts messages through ``HudServer.publish`` (the session's egress seam)
and asserts what a greeting page is sent, or what the session log holds; nothing is compared with the
journal's internals except where the task names them (``REPLAY_MAX_THOUGHTS``, ``truncated``).

THE SEAMS (names are the contract with the implementer; ``hud_server.py`` is group A's):

    conductor/hud_server.py
        REPLAY_MAX_THOUGHTS = 16
        REPLAY_MAX_MESSAGES gains that term and stays < DEFAULT_OUTBOX (512)
        TurnJournal.record keeps the latest astra.thought per attempt_id of the current turn, in
        first-seen order; the 17th attempt drops the oldest and sets ``truncated``; a thought of a
        superseded turn (T-0090 U10: seen, not the newest first-seen id) is published, never
        recorded, and clears nothing.
        TurnJournal.messages(): ... tool calls, thoughts, played reports (played stays last).
        HudServer.publish's wire.out row of a thought: every field but ``text``, plus ``n_chars``.

Readings recorded (the task does not say them in so many words; the reviewer should check them):

- ``n_chars`` is the number of **code points** of the carried ``text`` (the contract measures the cap
  in code points; the omitted tail is already in ``omitted_chars``).
- A thought whose ``turn_id`` was never seen opens that turn, as a tool call does under U10 (a turn's
  first published message may be its first thought; C1 "a thought-only turn creates an article").
- A4's "T-0177's ``astra.vision.state`` stays last" is pinned twice: outside live, no thought follows
  the reply's closing turn-state repeat and whatever follows it is ``astra.vision.*``; in live (T-0177's
  ``vision_rig``, merged from main at c0cb001e), the reply ends with exactly one ``astra.vision.state``
  followed by T-0177 B9's one replayed ``astra.vision.frame`` (T-0177 B-P3: the state is last apart from a
  trailing replayed frame), and every replayed thought precedes the closing repeat and both.

A8 is two rows that are green now and must stay green: the hello replies and ``wire.out`` rows of a
thought-free script equal BASE's byte for byte (BASE run in a subprocess over ``git archive BASE_A``,
origin checked), and the frozen paths do not change over ``BASE_A..TIP_A`` (two literals).

Regime: CPU, no socket, ``ScriptedClock``, fakes, no measurement; the property runs Hypothesis's
default example count.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts import protocol
from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import (
    AstraCanvasCommit,
    AstraCanvasStep,
    AstraError,
    AstraHello,
    AstraSessionCleared,
    AstraSpoken,
    AstraSpokenPlayed,
    AstraToolCall,
    AstraTranscript,
    AstraTurnState,
    AstraUiFence,
    AstraUiSurface,
    PlayedUtterance,
)
from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.unit import _t0181_pins as t0181
from tests.unit import test_session_tool_loop as tsl
from tests.unit._hud_rig import UI_CANVAS, Recorder, make_rig, require_symbol, running
from tests.unit._session_log_rig import make_log, of_kind, read_records

TASK = "T-0180"
#: T-0180's BASE: main at 7646b8b6 ("PROGRESS: T-0182 clip render run"), the branch point. A literal.
BASE_A = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
#: The group A implementation tip. Starts at BASE_A (no implementation yet); whoever commits group A's
#: implementation bumps it, and the merge closes it to the last group A implementation commit.
TIP_A = "5daa23bce6c23a155f389738233cddb9cd30aa08"  # T-0180 A implementation ("astra.thought on the wire...")
#: A8: group A modifies none of these.
FROZEN_A8 = (
    "plugin",
    "local_astra/conductor/canvas_relay.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "hud/src/denoise",
)
THOUGHT = "astra.thought"
SENTINEL = "THOUGHT_SENTINEL_Q7"
REPLAY_TRUNCATED = "replay_truncated"


class _NullSink:
    def ingest(self, message: Any) -> None:
        return None


def thought_model() -> Any:
    model = getattr(protocol, "AstraThought", None)
    assert model is not None, (
        "T-0180 A1: contracts/protocol.py gains AstraThought (type 'astra.thought'); the journal rows "
        "need it to publish one"
    )
    return model


def th(
    turn: str = "t4",
    attempt: str = "a1",
    *,
    end: str = "open",
    text: str = "Looking.",
    step: int = 1,
    kind: str = "step",
    omitted: int = 0,
) -> Any:
    return thought_model()(
        turn_id=turn,
        request_id=f"chatcmpl-s1-{turn}-r-{attempt}",
        attempt_id=f"{turn}-{attempt}",
        step=step,
        kind=kind,
        text=text,
        omitted_chars=omitted,
        end=end,
    )


def server_of(rig: Any, **kwargs: Any) -> Any:
    return require_symbol("HudServer")(rig.session, frame_sink=_NullSink(), port=0, **kwargs)


def make_server(tmp_path: Path) -> Any:
    return server_of(make_rig(tmp_path))


def greet(server: Any) -> Any:
    connection = server.open()
    connection.receive(encode(AstraHello()))
    return connection


def without_settings_state(reply: list[Any]) -> list[Any]:
    """T-0213 R-5 named re-pin: the reply with its one settings state (asserted last) set aside. Imported here, not
    at module level: the A8 BASE subprocess imports this module against BASE_A's ``local_astra``, which has no
    settings catalog."""
    from tests.unit import test_t0213_settings_wire as t0213

    return t0213.hello_without_settings_state(reply)


def reply_of(server: Any) -> list[Any]:
    # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
    return without_settings_state(list(greet(server).sent()))


#: the hello's live head (assist, turn, log state); T-0181 A4 pin move (A-R8): plus the mic state once the tree has it
LIVE_HEAD_LEN = len(t0181.hello_head(("astra.assist.state", "astra.turn.state", "astra.log.state")))


def thoughts_in(messages: list[Any]) -> list[Any]:
    return [m for m in messages if getattr(m, "type", "") == THOUGHT]


def without_replay(message: Any) -> Any:
    return message.model_copy(update={"replay": False})


def transcript(turn: str, text: str = "question") -> AstraTranscript:
    return AstraTranscript(turn_id=turn, text=text, final=True)


def spoken(turn: str, n: int) -> AstraSpoken:
    return AstraSpoken(turn_id=turn, utterance_id=f"{turn}-u{n}", text=f"Line {n}.")


def surface(turn: str) -> AstraUiSurface:
    return AstraUiSurface(
        surface_id=f"turn-{turn}",
        canvas_idx=1,
        final=True,
        messages=tuple(json.loads(line) for line in UI_CANVAS.splitlines()),
    )


def tool_call(turn: str, n: int, status: str = "proposed") -> AstraToolCall:
    return AstraToolCall(
        turn_id=turn,
        call_id=f"{turn}-c{n}",
        name="scroll",
        arguments={"direction": "down"},
        status=status,  # type: ignore[arg-type]
        source="draft",
    )


def played(turn: str, *utterances: str) -> AstraSpokenPlayed:
    return AstraSpokenPlayed(
        turn_id=turn,
        played_ms=900,
        utterances=[
            PlayedUtterance(utterance_id=u, played_text="Line", played_words=1, complete=False)
            for u in utterances
        ],
    )


# --- A4: what the journal keeps -------------------------------------------------------------------


def test_a4_the_latest_thought_per_attempt_is_replayed_in_first_seen_order(tmp_path):
    server = make_server(tmp_path)
    server.publish(transcript("t4"))
    a1_open = th(attempt="a1", text="First look.")
    a2_open = th(attempt="a2", text="Second attempt.", step=2, kind="reask")
    a1_done = th(attempt="a1", text="First look. Then the button.", end="done")
    for message in (a1_open, a2_open, a1_done):
        server.publish(message)
    got = thoughts_in(reply_of(server))
    assert [m.attempt_id for m in got] == ["t4-a1", "t4-a2"], (
        "A4: first-seen order, one per attempt"
    )
    assert all(m.replay is True for m in got), "A4: replayed copies carry replay=True"
    assert [without_replay(m) for m in got] == [a1_done, a2_open], (
        "A4: the latest message per attempt"
    )


def test_a4_the_seventeenth_attempt_drops_the_oldest_and_the_reply_says_so(tmp_path):
    from local_astra.conductor import hud_server

    assert getattr(hud_server, "REPLAY_MAX_THOUGHTS", None) == 16, (
        "A4/H-T180-5: 16 attempts per turn"
    )
    server = make_server(tmp_path)
    server.publish(transcript("t4"))
    for i in range(1, 18):
        server.publish(th(attempt=f"a{i}", step=i))
    reply = reply_of(server)
    assert [m.attempt_id for m in thoughts_in(reply)] == [f"t4-a{i}" for i in range(2, 18)], (
        "A4: 16 kept, the oldest (a1) dropped, order kept"
    )
    assert [m.code for m in reply if isinstance(m, AstraError)] == [REPLAY_TRUNCATED], (
        "A4: dropping a block is declared to the page (truncated)"
    )


def test_a4_an_update_of_a_held_attempt_at_the_cap_evicts_nothing(tmp_path):
    """The too-wide direction of the bound: a latest-wins update is not a new attempt."""
    server = make_server(tmp_path)
    server.publish(transcript("t4"))
    for i in range(1, 17):
        server.publish(th(attempt=f"a{i}", step=i))
    server.publish(th(attempt="a1", end="done", text="Looking. Done."))
    reply = reply_of(server)
    got = thoughts_in(reply)
    assert [m.attempt_id for m in got] == [f"t4-a{i}" for i in range(1, 17)]
    assert got[0].end == "done" and got[0].text == "Looking. Done."
    assert not [m for m in reply if isinstance(m, AstraError)], "A4: nothing was dropped"


def test_a4_adversarial_a_late_interrupted_thought_of_t3_after_t4_opened_reopens_nothing(tmp_path):
    """A4 (T-0090 U10): t3 thinks, t4 opens, then t3's attempt ends ``interrupted`` late. It is
    published live (the page decides), never recorded, and it clears none of t4."""
    server = make_server(tmp_path)
    live = greet(server)
    server.publish(transcript("t3"))
    server.publish(th("t3", "a1", text="Thinking about t3."))
    server.publish(transcript("t4", "the second question"))
    server.publish(spoken("t4", 1))
    late = th("t3", "a1", text="Thinking about t3.", end="interrupted")
    server.publish(late)
    server.publish(spoken("t4", 2))
    assert late in live.sent(), "A4: a superseded turn's thought is still published live"
    reply = reply_of(server)
    body = [
        (m.type, getattr(m, "turn_id", None))
        for m in reply[LIVE_HEAD_LEN:]
        if not isinstance(m, AstraTurnState)
    ]
    assert body == [("astra.transcript", "t4"), ("astra.spoken", "t4"), ("astra.spoken", "t4")], (
        f"A4: t4's journal is whole and no t3 thought is replayed; got {body}"
    )
    assert not [m for m in reply if isinstance(m, AstraError)], "A4: and nothing is declared lost"


def test_a4_a_thought_of_the_current_turn_is_recorded_after_a_superseded_one_was_ignored(tmp_path):
    """The too-wide direction of the U10 fence: after ignoring a late t3 thought, a t4 thought is
    still journalled (a fence keyed on "any thought after a turn change" would drop it)."""
    server = make_server(tmp_path)
    server.publish(transcript("t3"))
    server.publish(transcript("t4"))
    server.publish(th("t3", "a1", end="interrupted"))
    mine = th("t4", "a1", text="Thinking about t4.")
    server.publish(mine)
    got = thoughts_in(reply_of(server))
    assert [without_replay(m) for m in got] == [mine]


def test_a4_a_thought_of_a_never_seen_turn_opens_that_turn(tmp_path):
    """Reading (module docstring): a never-seen ``turn_id`` is the current turn (U10), so a thought
    can open its turn; the previous turn's lines are no longer replayed."""
    server = make_server(tmp_path)
    server.publish(transcript("t4"))
    server.publish(spoken("t4", 1))
    first = th("t5", "a1", text="A thought before anything else of t5.")
    server.publish(first)
    reply = reply_of(server)
    assert [without_replay(m) for m in thoughts_in(reply)] == [first]
    assert not [m for m in reply if isinstance(m, AstraSpoken)], "t4 is superseded"


def test_a4_a_current_turn_thought_reopens_the_journal_after_a_departure_clear():
    """T-0037 criterion 4 / U8: a departure clear forgets the turn; a later thought of the same
    (current) turn is journalled again, as a tool call is."""
    journal = require_symbol("TurnJournal")()
    journal.record(transcript("t4"))
    journal.record(th("t4", "a1"))
    journal.clear()
    assert thoughts_in(journal.messages()) == [], "a clear forgets the thoughts too"
    again = th("t4", "a1", end="done")
    journal.record(again)
    assert [without_replay(m) for m in thoughts_in(journal.messages())] == [again]


def test_a4_a_session_reset_clears_the_thoughts(tmp_path):
    server = make_server(tmp_path)
    server.publish(transcript("t4"))
    server.publish(th("t4", "a1", text=SENTINEL))
    assert thoughts_in(reply_of(server)), "premise: the thought is journalled before the reset"
    server.publish(AstraSessionCleared(resets=1, ended_turn_id="t4", memory_forgotten=0))
    reply = reply_of(server)
    assert thoughts_in(reply) == [] and SENTINEL not in "".join(encode(m) for m in reply)


def test_a4_replay_order_thoughts_after_tool_calls_before_played_and_the_repeat_last(tmp_path):
    """A4: the published order is scrambled on purpose; the reply's order is the journal's: transcript,
    spoken, surface, INTERRUPTED, tool calls, thoughts, played, then the closing turn-state repeat
    (T-0037). After that repeat only ``astra.vision.*`` may follow (T-0177; module docstring)."""
    thought_model()
    server = make_server(tmp_path)
    for message in (
        transcript("t4"),
        th("t4", "a1"),
        played("t4", "t4-u1"),
        tool_call("t4", 1),
        spoken("t4", 1),
        AstraTurnState(state=TurnState.INTERRUPTED, turn_id="t4"),
        th("t4", "a2", step=2),
        surface("t4"),
    ):
        server.publish(message)
    reply = reply_of(server)
    after_head = [m.type for m in reply[LIVE_HEAD_LEN:]]
    closing = max(i for i, m in enumerate(reply) if isinstance(m, AstraTurnState) and m.replay)
    tail = [m.type for m in reply[closing + 1 :]]
    assert all(t.startswith("astra.vision.") for t in tail), (
        f"A4: after the repeat only vision: {tail}"
    )
    assert after_head[: after_head.index("astra.turn.state", 5) + 1] == [
        "astra.transcript",
        "astra.spoken",
        "astra.ui.surface",
        "astra.turn.state",
        "astra.tool.call",
        THOUGHT,
        THOUGHT,
        "astra.spoken.played",
        "astra.turn.state",
    ], after_head


async def test_a4_in_live_the_thoughts_replay_before_the_vision_state_which_stays_last(tmp_path):
    """A4 + T-0177 A4: in a live session, after one answered turn and two thoughts of that turn, a new
    page's hello reply carries both thoughts (``replay=True``), before the closing turn-state repeat, and
    ends with the one ``astra.vision.state`` then the replayed ``astra.vision.frame`` (T-0177 B-P3)."""
    from tests.unit._tool_turn_support import answer
    from tests.unit.test_t0177_shared_frame import one_turn, vision_rig

    thought_model()
    async with vision_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await one_turn(rig)
        # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
        before = without_settings_state(list(greet(rig.server).sent()))
        turns = [m.turn_id for m in before if isinstance(m, AstraTranscript)]
        assert turns, f"premise: the answered turn is replayed; {[m.type for m in before]}"
        turn = turns[-1]
        first = th(turn, "a1", text="The page is a shop.", end="done")
        second = th(turn, "a2", text=SENTINEL, step=2, kind="reask")
        rig.server.publish(first)
        rig.server.publish(second)
        # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
        reply = without_settings_state(list(greet(rig.server).sent()))
    types = [m.type for m in reply]
    # T-0177 B-P3 (pre-authorised pin move): the state is last apart from a trailing replayed frame. The answered
    # turn's step-1 request published one ``astra.vision.frame`` (B7, blind or not), so B9 replays it after the state.
    assert types[-2:] == ["astra.vision.state", "astra.vision.frame"], (
        f"A4 + T-0177 B-P3: in live the hello ends with the state, then the replayed frame; {types}"
    )
    assert types.count("astra.vision.state") == 1, types
    assert types.count("astra.vision.frame") == 1, (
        f"T-0177 B9: one replayed frame, the latest; {types}"
    )
    assert reply[-1].replay is True, "T-0177 B9: the hello frame carries replay=True"
    got = thoughts_in(reply)
    assert [without_replay(m) for m in got] == [first, second], (
        f"A4: both thoughts of the live turn are replayed, in order; {types}"
    )
    assert all(m.replay is True for m in got)
    closing = max(i for i, m in enumerate(reply) if isinstance(m, AstraTurnState) and m.replay)
    assert all(reply.index(m) < closing for m in got), f"A4: thoughts precede the repeat; {types}"
    state_at = types.index("astra.vision.state")
    assert all(reply.index(m) < state_at for m in got), (
        f"T-0177 B-P3: thoughts come before the state and the frame; {types}"
    )


def test_a4_replay_max_messages_counts_the_thoughts_and_stays_under_the_outbox():
    from local_astra.conductor import hud_server as hs

    thoughts = getattr(hs, "REPLAY_MAX_THOUGHTS", None)
    assert thoughts == 16, "A4/H-T180-5"
    base_terms = 1 + hs.REPLAY_MAX_SPOKEN + 1 + 1 + hs.REPLAY_MAX_TOOL_CALLS + hs.REPLAY_MAX_PLAYED
    assert base_terms + thoughts <= hs.REPLAY_MAX_MESSAGES, "A4: the bound includes the new term"
    assert hs.REPLAY_MAX_MESSAGES < hs.DEFAULT_OUTBOX == 512, "A4: a replay never fills an outbox"
    # Behaviour, not only arithmetic: a journal filled past every bound holds exactly this many.
    journal = hs.TurnJournal()
    journal.record(transcript("t4"))
    for n in range(80):
        journal.record(spoken("t4", n))
    journal.record(surface("t4"))
    journal.record(AstraTurnState(state=TurnState.INTERRUPTED, turn_id="t4"))
    for n in range(20):
        journal.record(tool_call("t4", n))
    for n in range(20):
        journal.record(th("t4", f"a{n}"))
    for n in range(40):
        journal.record(played("t4", f"t4-u{n}"))
    held = journal.messages()
    assert len(held) == 1 + 64 + 1 + 1 + 16 + 16 + 32 <= hs.REPLAY_MAX_MESSAGES, len(held)
    assert len(thoughts_in(held)) == 16


# --- the U10 property over thoughts ---------------------------------------------------------------

_TURNS = ("t1", "t2", "t3")
_EVENT = st.one_of(
    st.tuples(st.just("tr"), st.sampled_from(_TURNS)),
    st.tuples(
        st.just("th"),
        st.sampled_from(_TURNS),
        st.integers(min_value=1, max_value=18),
        st.sampled_from(("open", "done", "interrupted")),
    ),
)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(events=st.lists(_EVENT, min_size=1, max_size=40))
def test_a4_property_replayed_thoughts_are_the_current_turns_latest_and_bounded(events):
    """A4 over arbitrary interleavings of three turns' transcripts and thoughts (attempts 1..18):

    - every replayed thought belongs to the newest first-seen turn id (U10: a superseded turn's
      thought is never recorded);
    - each is, field for field, the last thought published for its attempt;
    - at most 16 are replayed, each attempt at most once;
    - liveness: when the last event is a thought of the newest first-seen turn, it is replayed.
    """
    thought_model()
    journal = require_symbol("TurnJournal")()
    seen: list[str] = []
    last: dict[str, Any] = {}
    for event in events:
        if event[0] == "tr":
            message: Any = transcript(event[1])
        else:
            message = th(
                event[1], f"a{event[2]}", end=event[3], text=f"{event[1]} {event[2]} {event[3]}"
            )
            last[message.attempt_id] = message
        if message.turn_id not in seen:
            seen.append(message.turn_id)
        journal.record(message)
    current = seen[-1]
    got = thoughts_in(journal.messages())
    assert all(m.turn_id == current for m in got), [(m.turn_id, m.attempt_id) for m in got]
    assert all(without_replay(m) == last[m.attempt_id] for m in got)
    ids = [m.attempt_id for m in got]
    assert len(ids) == len(set(ids)) <= 16, ids
    final = events[-1]
    if final[0] == "th" and final[1] == current:
        assert f"{final[1]}-a{final[2]}" in ids, "liveness: the newest current-turn thought is kept"


# --- A5: the compact wire.out row -----------------------------------------------------------------


async def test_a5_the_wire_out_row_of_a_thought_carries_n_chars_and_never_the_text(tmp_path):
    """A5: every field but ``text``, plus ``n_chars``. The text is ``ab``, one astral character, a
    newline and the 19-character sentinel: 23 code points (25 UTF-16 units, 26 UTF-8 bytes)."""
    text = "ab\U0001f600\n" + SENTINEL
    message = th("t4", "a1", text=text, step=2, kind="reask", end="done")
    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server_of(rig).publish(message)
    rows = [r["data"] for r in of_kind(read_records(log.path), "wire.out")]
    assert len(rows) == 1, rows
    assert rows[0]["msg"] == {
        "type": THOUGHT,
        "turn_id": "t4",
        "request_id": "chatcmpl-s1-t4-r-a1",
        "attempt_id": "t4-a1",
        "step": 2,
        "kind": "reask",
        "omitted_chars": 0,
        "end": "done",
        "replay": False,
        "n_chars": 23,
    }, rows[0]["msg"]
    assert SENTINEL not in log.path.read_text(encoding="utf-8"), "A5: the text is not in wire.out"


# --- A8: nothing else moves -----------------------------------------------------------------------


def _stage_messages() -> list[list[Any]]:
    """A thought-free script in four stages; a hello reply is captured after each."""
    canvas = AstraCanvasStep(
        request_id="chatcmpl-s1-t1-r1",
        canvas_idx=0,
        step=1,
        max_steps=4,
        canvas_length=2,
        canvas=(5, 6),
        argmax=(5, 7),
        stable="80",
    )
    return [
        [
            AstraTurnState(state=TurnState.THINKING, turn_id="t1", thinking_ms=40),
            AstraTranscript(turn_id="t1", text="what is"),
            transcript("t1", "what is on this screen"),
            spoken("t1", 1),
            surface("t1"),
            tool_call("t1", 1),
            tool_call("t1", 1, status="dry_run"),
            spoken("t1", 2),
            AstraUiFence(turn_id="t1", canvases=1, close="written"),
            canvas,
            AstraCanvasCommit(
                request_id="chatcmpl-s1-t1-r1",
                canvas_idx=0,
                segment="answer",
                text="Line 1.",
                steps_used=3,
            ),
            AstraTurnState(state=TurnState.INTERRUPTED, turn_id="t1"),
            played("t1", "t1-u1", "t1-u2"),
            AstraError(code="llm_unavailable", message="vllm is down"),
        ],
        [transcript("t2", "and now"), tool_call("t1", 1, status="cancelled"), spoken("t2", 1)],
        [spoken("t2", n) for n in range(2, 72)],
        [
            AstraSessionCleared(resets=1, ended_turn_id="t2", memory_forgotten=0),
            transcript("t3", "hi"),
        ],
    ]


async def capture_without_thoughts(tmp: Path) -> dict[str, Any]:
    """The A8 sections, by the code under test: the encoded hello reply after each stage (no log), and
    the ``wire.out`` rows of the same script with a log and one greeted page attached. Run in this
    process (TIP) and, unchanged, in a subprocess over BASE_A's ``local_astra``."""
    server = server_of(make_rig(tmp / "hello"))
    replies: list[list[str]] = []
    for stage in _stage_messages():
        for message in stage:
            server.publish(message)
        replies.append([encode(m) for m in greet(server).sent()])
    log = make_log(tmp / "logs")
    async with running(tmp / "data", session_log=log, publish=Recorder()) as rig:
        logged = server_of(rig)
        greet(logged)
        for stage in _stage_messages():
            for message in stage:
                logged.publish(message)
    rows = [r["data"] for r in of_kind(read_records(log.path), "wire.out")]
    return {"hello": replies, "wire_out": rows}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0180_journal as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m.capture_without_thoughts(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


async def test_a8_without_a_thought_hello_replies_and_wire_out_rows_are_base_byte_for_byte(
    tmp_path,
):
    """A8 (green at BASE, must stay green): a script with no ``astra.thought`` gives exactly BASE_A's
    hello replies (encoded) and ``wire.out`` rows. The BASE side is a subprocess over ``git archive
    BASE_A`` with its ``local_astra`` first on ``sys.path`` (origin checked)."""
    gr.require_commits(BASE_A)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_A)
        base_dir = tsl._base_tree(tmp_path / "base_tree")
    out = tmp_path / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the BASE run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the BASE run imported local_astra from {data['origin']}"
    )
    base = data["sections"]
    now = await capture_without_thoughts(tmp_path / "now")
    # 14 + 3 + 70 + 2 published messages, one wire.out each
    assert len(base["hello"]) == 4 and len(base["wire_out"]) == 89, "premise: the script ran"
    for i, (b, n) in enumerate(zip(base["hello"], now["hello"], strict=True)):
        # T-0181 A4 pin move (A-R8): the one mic state, asserted right after the log state, set aside
        n = t0181.hello_without_mic_state(n)
        # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
        n = without_settings_state(n)
        assert b == n, f"A8: hello reply after stage {i + 1} differs from BASE {BASE_A[:8]}"
    assert len(base["wire_out"]) == len(now["wire_out"])
    for i, (b, n) in enumerate(zip(base["wire_out"], now["wire_out"], strict=True)):
        assert b == n, f"A8: wire.out row {i} differs from BASE {BASE_A[:8]}:\nbase {b}\nnow  {n}"


def test_a8_group_a_leaves_the_frozen_paths_alone_over_base_a_to_tip_a():
    """A8: ``plugin/**``, ``canvas_relay.py``, ``turn_machine.py``, ``speech_gate.py`` and
    ``hud/src/denoise/**`` are unchanged over ``BASE_A..TIP_A`` (two literals; TIP_A moves with group A's
    implementation commits and closes at the merge)."""
    gr.require_commits(BASE_A, TIP_A)
    assert gr.is_ancestor(BASE_A, TIP_A)
    assert gr.changed(BASE_A, TIP_A, *FROZEN_A8) == []
