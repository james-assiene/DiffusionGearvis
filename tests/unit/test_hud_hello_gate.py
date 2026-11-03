"""T-0166 H2, H4-H7 and the N2 greeted fan-out rows: nothing published reaches a connection ahead
of its hello reply (docs/tasks/T-0166.md, "Hold until hello").

The defect: ``HudServer.publish`` fans out to every accepted connection, greeted or not. While a
session is THINKING it publishes ``astra.turn.state`` at 20 Hz, so a reloading page can be sent a
live message before its hello reply, and the page's T-0162 D-3 "first message ever" rule then stops
its screen capture. After this task a message published to a connection that has not said a valid
hello is *held* and delivered after the hello reply, in publication order.

**What the rows read.** ``conn.sent()`` is "what this connection would put on the wire, in order"
(``tests/unit/_hud_rig.py``). Before the hello it may well list held messages (N3: the existing
outbox rows read it on connections that never greet, so held messages stay visible there); what
these rows pin is the order **after** the hello: the reply first, then every held message in
publication order (H2), minus only the losses H2 allows (the ``Outbox`` policy, the H4 reset cut,
the H5 departure cut). The wire-level half, "nothing held is *written* before the hello", is
``tests/contract/test_hud_hello_gate_wire.py`` (H1, H3, H4).

**The oracle.** The expected reply is ``list(server.hello_reply())`` taken immediately before the
hello (the task's own instruction: ``hello_reply()``'s order is T-0037/T-0043/T-0090/T-0162's and
does not change, N1), cross-checked by hand-derived type lists where the row's point depends on it.
The expected held tail is the list of messages this test itself published, in the order it
published them. Nothing is derived from the server's internals.

**The "must not change" rows** (N2, the H4 negative, the H7 bound) were green on main before the
change, on purpose, and never carried a ``pending`` marker; the other rows were pending on T-0166
until its implementation started (activated 2026-10-27).

Regime: no socket (``server.open()`` + ``receive(encode(AstraHello()))``), ``ScriptedClock``,
``tests/fakes/``, no wall time, no seed, n = 1 per row; the H7 row publishes 1 200 turn states
(60 s of virtual time at 20 Hz) and asserts counts only.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_astra.conductor.hud_server import (
    DEFAULT_OUTBOX,
    REPLAY_MAX_MESSAGES,
    REPLAY_MAX_PLAYED,
    REPLAY_MAX_SPOKEN,
    REPLAY_MAX_TOOL_CALLS,
)
from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import (
    PROTOCOL_VERSION,
    AstraAssistState,
    AstraCanvasStep,
    AstraError,
    AstraHello,
    AstraLogState,
    AstraSessionCleared,
    AstraSpoken,
    AstraText,
    AstraToolCall,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from tests.fakes.canvas_tap import (
    StateBox,
    StubPieces,
    commit,
    denoise,
    optional_piece_kwargs,
    pack,
    relay_attr,
)
from tests.unit import _t0181_pins as t0181
from tests.unit import test_t0213_settings_wire as t0213  # T-0213 R-5 named re-pin
from tests.unit._hud_rig import Recorder, running
from tests.unit._session_log_rig import make_log, of_kind, read_records
from tests.unit.test_hud_hello_replay import (
    LIVE_HEAD,
    greet,
    head_of,
    make_server,
    played,
    publish_a_turn,
    spoken,
    surface,
    transcript,
)
from tests.unit.test_hud_wire_tap import serve

#: vLLM's own request id shape, which the canvas relay carries (T-0162 amendment (a))
PRE_RESET_RELAY_ID = "chatcmpl-aaaaaaaaaaaaaaaa-aaaaaaaa"
RELAY_ID = "chatcmpl-0123456789abcdef-01234567"
#: an endpoint the relay constructor is handed; nothing here starts a relay on it
UNIT_ENDPOINT = "ipc:///nonexistent/t0166-unit.sock"


# --- helpers ---------------------------------------------------------------------------------


def wire(messages: list[Any]) -> list[str]:
    """Messages as the production codec puts them on the wire (the byte-level comparison)."""
    return [encode(m) for m in messages]


def kinds(messages: list[Any]) -> list[str]:
    return [f"{m.type}{'(replay)' if getattr(m, 'replay', False) else ''}" for m in messages]


def hello(connection: Any, protocol_version: int = PROTOCOL_VERSION) -> None:
    connection.receive(encode(AstraHello(protocol_version=protocol_version)))


def thinking(i: int, turn: str = "t1") -> AstraTurnState:
    """The i-th 20 Hz meter update of a THINKING turn (50 ms apart)."""
    return AstraTurnState(state=TurnState.THINKING, turn_id=turn, thinking_ms=50 * i)


def tool_call(call_id: str, *, turn: str = "t1", status: str = "proposed") -> AstraToolCall:
    return AstraToolCall(
        turn_id=turn,
        call_id=call_id,
        name="scroll",
        arguments={"direction": "down"},
        status=status,  # type: ignore[arg-type]
        source="draft",
    )


def canvas(request_id: str, step: int = 1) -> AstraCanvasStep:
    return AstraCanvasStep.model_validate(commit(request_id, 0, step))


class Publisher:
    """Publishes through ``server.publish`` and keeps what it published, in order (the oracle)."""

    def __init__(self, server: Any) -> None:
        self.server = server
        self.messages: list[Any] = []

    def __call__(self, message: Any) -> Any:
        self.server.publish(message)
        self.messages.append(message)
        return message


def assert_reply_then(sent: list[Any], reply: list[Any], tail: list[Any], criterion: str) -> None:
    """``sent`` is exactly ``reply`` followed by ``tail``, compared on the wire."""
    n = len(reply)
    assert wire(sent[:n]) == wire(reply), (
        f"{criterion}: after a valid hello the connection's first messages are the hello reply, "
        f"in hello_reply()'s own order, and no held message precedes it. Expected the reply "
        f"{kinds(reply)}; the connection's first {n} were {kinds(sent[:n])}"
    )
    assert wire(sent[n:]) == wire(tail), (
        f"{criterion}: after the reply come the held messages, in publication order, each "
        f"replay=False as published, and nothing else. Expected {kinds(tail)}; got "
        f"{kinds(sent[n:])}"
    )


# --- H2: held messages follow the reply ----------------------------------------------------------


async def test_a_message_published_before_hello_follows_the_reply(tmp_path):
    """H2, adversarial (primary): a spoken line, a tool call and 40 ``astra.turn.state`` (20 Hz for
    2 s) are published between ``open()`` and the hello.

    The spoken line and the tool call are journalled, so the reply carries a ``replay=True`` copy
    of each; they must *also* follow live exactly once (no dedupe against the reply, ruling Q3).
    """
    _, server = make_server(tmp_path)
    conn = server.open()
    publish = Publisher(server)
    line = spoken("t1-u1", "ZQXHELDLINE the first sentence.")
    call = tool_call("c-1")
    for i in range(40):
        publish(thinking(i))
        if i == 10:
            publish(line)
        if i == 25:
            publish(call)
    assert len(publish.messages) == 42, "premise: 40 states, one spoken line, one tool call"

    reply = list(server.hello_reply())
    hello(conn)
    sent = conn.sent()

    assert not conn.closed, f"H2: the hello closed the connection ({conn.close_reason})"
    assert isinstance(sent[0], AstraAssistState), (
        "H2: with no reset the connection's first message is the reply head, the assist state "
        f"(docs/02 §5 T-0037); it was {kinds(sent[:1])}, so a held message went out first"
    )
    assert_reply_then(sent, reply, publish.messages, "H2")
    live_lines = [m for m in sent if isinstance(m, AstraSpoken) and not m.replay]
    live_calls = [m for m in sent if isinstance(m, AstraToolCall) and not m.replay]
    assert wire(live_lines) == wire([line]) and wire(live_calls) == wire([call]), (
        "H2: the spoken line and the tool call each follow the reply live exactly once; got "
        f"{len(live_lines)} live lines and {len(live_calls)} live calls"
    )
    assert [m for m in reply if isinstance(m, AstraSpoken) and m.replay], (
        "premise (Q3, implementation notes): the journal also replays the line, so the live copy "
        f"is a second delivery that must not be deduped; the reply was {kinds(reply)}"
    )


async def test_twenty_hz_turn_state_never_precedes_the_reply(tmp_path):
    """H2: 40 states only (2 s of 20 Hz). The first message is the assist state (no reset), and the
    latest state the page is left on is the last one published (the chip is not left behind)."""
    _, server = make_server(tmp_path)
    conn = server.open()
    publish = Publisher(server)
    for i in range(40):
        publish(thinking(i))

    reply = list(server.hello_reply())
    hello(conn)
    sent = conn.sent()

    first = sent[0] if sent else None
    assert isinstance(first, AstraAssistState), (
        "H2 / T-0162 D-3: a page's first message is its hello reply's head; under 20 Hz "
        f"astra.turn.state it was {kinds(sent[:3])}"
    )
    assert_reply_then(sent, reply, publish.messages, "H2")
    states = [m for m in sent if isinstance(m, AstraTurnState)]
    assert encode(states[-1]) == encode(publish.messages[-1]), (
        f"H2: the last turn state the page applies is the newest published ({publish.messages[-1]})"
        f"; it was {states[-1]}"
    )


async def test_the_reply_is_unchanged_with_nothing_held(tmp_path):
    """N2/H2 negative (green on main on purpose): with nothing held the reply is today's, byte for
    byte, and a greeted page's later publishes follow it live. Uses T-0037's own turn shape."""
    _, server = make_server(tmp_path)
    publish_a_turn(server)
    reply = list(server.hello_reply())
    conn = greet(server)
    head_of(conn.sent())
    assert wire(conn.sent()) == wire(reply), (
        f"N2: nothing held, the reply is hello_reply() exactly; got {kinds(conn.sent())}"
    )
    after = Publisher(server)
    after(spoken("t1-u3", "One more line."))
    after(thinking(1))
    assert_reply_then(conn.sent(), reply, after.messages, "N2")


# --- H4: the reset cut -----------------------------------------------------------------------


async def test_a_reset_before_hello_cuts_what_was_held_before_it(tmp_path):
    """H4, adversarial: held = [spoken t1, relay canvas ``chatcmpl-A`` step, cleared(resets=1),
    spoken t2, turn.state]. Delivered = the reply (starting with the replayed cleared), then
    [spoken t2, turn.state]. The cut is by publication order, not by turn id or request id."""
    _, server = make_server(tmp_path)
    conn = server.open()
    server.publish(spoken("t1-u1", "ZQXPRERESET line of the old session.", turn="t1"))
    server.publish(canvas(PRE_RESET_RELAY_ID))
    server.publish(AstraSessionCleared(resets=1, ended_turn_id="t1", memory_forgotten=0))
    after = Publisher(server)
    after(spoken("t2-u1", "The new session speaks.", turn="t2"))
    after(thinking(3, turn="t2"))

    reply = list(server.hello_reply())
    # T-0181 A4 pin move (A-R8): the mic state right after the log state, fifth after a replayed reset
    # T-0213 R-5 named re-pin: then the one settings state, last
    assert [type(m) for m in reply] == t0213.hello_with_settings_last(
        t0181.hello_head(
            [
                AstraSessionCleared,
                AstraAssistState,
                AstraTurnState,
                AstraLogState,
                AstraSpoken,
                AstraTurnState,
            ]
        )
    ), f"premise (T-0162 A-5, T-0037): the post-reset reply by hand; got {kinds(reply)}"
    hello(conn)
    sent = conn.sent()

    assert sent and isinstance(sent[0], AstraSessionCleared) and sent[0].replay, (
        "H4 / T-0162 D-3: after a reset the connection's first message is the replayed cleared; "
        f"it was {kinds(sent[:2])}"
    )
    assert_reply_then(sent, reply, after.messages, "H4")
    raw = wire(sent)
    assert not [r for r in raw if "ZQXPRERESET" in r], (
        f"H4: a line held from before the reset reached the page: {kinds(sent)}"
    )
    assert not [r for r in raw if PRE_RESET_RELAY_ID in r], (
        f"H4: a canvas frame of {PRE_RESET_RELAY_ID}, held from before the reset, reached the page"
    )
    cleared = [m for m in sent if isinstance(m, AstraSessionCleared)]
    assert len(cleared) == 1 and cleared[0].replay, (
        "H4: exactly one cleared reaches the page, the replayed one (the live cleared it would "
        f"have been sent is cut with what was held before it); got {kinds(cleared)}"
    )


async def test_a_reset_after_hello_is_live_as_today(tmp_path):
    """H4 negative (green on main on purpose): a greeted page gets the live cleared, FIFO behind
    what was already queued for it (``test_queued_messages_go_out_before_the_echo``'s shape)."""
    _, server = make_server(tmp_path)
    conn = greet(server)
    reply = list(conn.sent())
    after = Publisher(server)
    after(spoken("t1-u1", "Already queued."))
    after(AstraSessionCleared(resets=1, ended_turn_id="t1", memory_forgotten=0))
    after(spoken("t2-u1", "After the reset.", turn="t2"))
    assert_reply_then(conn.sent(), reply, after.messages, "H4 negative")
    cleared = [m for m in conn.sent() if isinstance(m, AstraSessionCleared)]
    assert len(cleared) == 1 and not cleared[0].replay, kinds(cleared)


# --- H1/H2: the canvas relay's direct publish path ----------------------------------------------


async def test_relay_canvas_frames_before_hello_follow_the_reply(tmp_path):
    """H1/H2: ``canvas_relay.py`` calls ``HudServer.publish`` directly (docs/02 §5 "Canvas
    relay"), so its frames are held like any other publish. Three frames (two commits, which are
    never coalesced, and one denoise), well under the outbox's canvas cap of 8, so the ``Outbox``
    policy loses none of them and the tail is exactly what the relay published."""
    rig, server = make_server(tmp_path)
    conn = server.open()
    relayed: list[Any] = []

    def publish(message: Any) -> None:
        relayed.append(message)
        server.publish(message)

    cls = relay_attr("CanvasRelay")
    relay = cls(
        publish=publish,
        clock=rig.clock,
        state=StateBox(TurnState.THINKING, "t1"),
        session_log=None,
        endpoint=UNIT_ENDPOINT,
        **optional_piece_kwargs(cls, StubPieces()),
    )
    for frame in (commit(RELAY_ID, 0, 9), commit(RELAY_ID, 1, 9), denoise(RELAY_ID, 2, 1)):
        relay.receive(pack(frame))
        rig.clock.advance(40)
        relay.flush()
    assert [(m.canvas_idx, m.phase) for m in relayed] == [
        (0, "commit"),
        (1, "commit"),
        (2, "denoise"),
    ], f"premise: the relay published the three frames; it published {kinds(relayed)}"

    reply = list(server.hello_reply())
    hello(conn)
    assert_reply_then(conn.sent(), reply, relayed, "H1/H2 (canvas relay)")


# --- H6: the reply is never truncated or refused because of held messages ------------------------


async def test_a_full_hold_and_a_full_journal_do_not_close_the_connection(tmp_path):
    """H6, adversarial: the connection holds ``DEFAULT_OUTBOX - 1`` spoken lines of the open turn
    and the journal is at ``REPLAY_MAX_MESSAGES``; then hello. The reply (by hand: 3 live + the
    truncation notice + the full journal + the repeated state) goes first, complete, and all 511
    held lines follow in order; the connection stays open."""
    from local_astra.conductor import hud_server
    from local_astra.contracts import protocol

    _, server = make_server(tmp_path)
    # T-0180 A4 (activation pin move): the journal also holds the turn's latest thought per
    # attempt, up to REPLAY_MAX_THOUGHTS = 16 (the task's literal), between tool calls and played.
    replay_max_thoughts = getattr(hud_server, "REPLAY_MAX_THOUGHTS", None)
    thought_model = getattr(protocol, "AstraThought", None)
    assert replay_max_thoughts == 16 and thought_model is not None, (
        "T-0180 A1/A4: contracts/protocol.py gains AstraThought and hud_server.py REPLAY_MAX_THOUGHTS = 16"
    )
    by_hand = 1 + REPLAY_MAX_SPOKEN + 1 + 1 + REPLAY_MAX_TOOL_CALLS + 16 + REPLAY_MAX_PLAYED
    assert by_hand == REPLAY_MAX_MESSAGES < DEFAULT_OUTBOX, (
        "premise / H6: the journal's cap is transcript + spoken + surface + INTERRUPTED + tool "
        "calls + thoughts (T-0180) + played, strictly below the outbox bound"
    )
    # the journal's non-spoken slots, published before the page connects (so not held)
    server.publish(transcript("what is on this screen", final=True))
    server.publish(surface(1, final=True))
    server.publish(AstraTurnState(state=TurnState.INTERRUPTED, turn_id="t1"))
    for i in range(REPLAY_MAX_TOOL_CALLS):
        server.publish(tool_call(f"c-{i}"))
    for i in range(16):
        server.publish(
            thought_model(
                turn_id="t1",
                request_id=f"chatcmpl-s1-t1-r{i}",
                attempt_id=f"t1-a{i}",
                step=i + 1,
                kind="step",
                text=f"thought {i}",
                omitted_chars=0,
                end="done",
            )
        )
    for i in range(REPLAY_MAX_PLAYED):
        server.publish(played(f"t1-p{i}"))

    conn = server.open()
    publish = Publisher(server)
    for i in range(DEFAULT_OUTBOX - 1):
        publish(spoken(f"t1-u{i}", f"held line {i}"))
    assert not conn.closed, f"premise: 511 held lines fit the outbox ({conn.close_reason})"

    reply = list(server.hello_reply())
    # T-0213 R-5 named re-pin: + the one settings state, last
    assert len(t0213.hello_without_settings_state(reply)) == (
        len(LIVE_HEAD) + 1 + REPLAY_MAX_MESSAGES + 1
    ), f"premise: the journal is full and truncated; the reply is {len(reply)} messages"
    hello(conn)
    assert not conn.closed, (
        "H6: the hello reply is never refused because held messages fill the outbox; the "
        f"connection was closed: {conn.close_reason}"
    )
    assert_reply_then(conn.sent(), reply, publish.messages, "H6")


# --- H7: a socket that never says hello is bounded -----------------------------------------------


async def test_a_silent_socket_overflows_like_a_stalled_reader(tmp_path):
    """H7 (green on main on purpose; the bound must survive the hold): a connection that never
    greets holds up to ``DEFAULT_OUTBOX`` spoken lines and is closed, with the overflow named, on
    the next one. Both directions: still open at the bound, closed one past it. A greeted sibling
    whose writer keeps up receives every line, in order."""
    _, server = make_server(tmp_path)
    page = greet(server)
    silent = server.open()
    received = list(page.outbox.drain())
    lines: list[Any] = []
    for i in range(DEFAULT_OUTBOX + 1):
        lines.append(spoken(f"t1-u{i}", f"line {i}"))
        server.publish(lines[-1])
        if i == DEFAULT_OUTBOX - 1:
            assert not silent.closed, (
                f"H7, too wide: {DEFAULT_OUTBOX} held lines are within the bound; the silent "
                f"connection was closed early ({silent.close_reason})"
            )
        if i % 64 == 0:
            received.extend(page.outbox.drain())  # the greeted page's writer keeps up
    received.extend(page.outbox.drain())

    assert silent.closed and "overflow" in (silent.close_reason or ""), (
        "H7 / T-0028 c12: overflow of a non-droppable kind closes that connection with the named "
        f"overflow reason; closed={silent.closed}, reason={silent.close_reason!r}"
    )
    assert not page.closed, f"H7: the greeted sibling is unaffected ({page.close_reason})"
    got = [m for m in received if isinstance(m, AstraSpoken)]
    assert wire(got) == wire(lines), (
        f"H7: the greeted sibling received every line in order; got {len(got)} of {len(lines)}"
    )


async def test_a_minute_of_turn_state_keeps_a_silent_socket_open_and_bounded(tmp_path):
    """H7 (green on main on purpose): 60 s of virtual time at 20 Hz, ``astra.turn.state`` only, to
    a connection that never greets: it stays open and never holds more than ``DEFAULT_OUTBOX``
    (the state is latest-wins, T-0028 c12)."""
    rig, server = make_server(tmp_path)
    silent = server.open()
    worst = 0
    for i in range(60 * 20):
        rig.clock.advance(50)
        server.publish(thinking(i))
        worst = max(worst, len(silent.outbox))
        assert not silent.closed, f"H7: closed after {i + 1} states: {silent.close_reason}"
    assert worst <= DEFAULT_OUTBOX, f"H7: held {worst} > {DEFAULT_OUTBOX}"


# --- H3 (unit half): direct replies are not held -------------------------------------------------


async def test_errors_to_an_ungreeted_connection_are_not_held(tmp_path):
    """H3: ``astra.error`` about the connection's *own* frame (``malformed``, ``no_hello``,
    ``wrong_direction``) is a direct reply, not a publish, and is not held: it is ahead of every
    held message, and the reply and the held messages follow the hello as in H2."""
    _, server = make_server(tmp_path)
    conn = server.open()
    publish = Publisher(server)
    publish(spoken("t1-u1", "A held line."))
    for i in range(3):
        publish(thinking(i))

    conn.receive("{this is not json")
    conn.receive(encode(AstraText(text="are you there")))
    conn.receive(encode(AstraSpoken(turn_id="t9", utterance_id="t9-u1", text="wrong way")))
    reply = list(server.hello_reply())
    hello(conn)
    sent = conn.sent()

    codes = [m.code for m in sent[:3] if isinstance(m, AstraError)]
    assert codes == ["malformed", "no_hello", "wrong_direction"], (
        "H3 (T-0028 c4, T-0008 c2): the errors about the page's own frames go out ahead of every "
        f"held published message; the connection's first messages were {kinds(sent[:5])}"
    )
    assert_reply_then(sent[3:], reply, publish.messages, "H3/H2")


# --- H5: the departure cut -------------------------------------------------------------------


async def test_a_departure_discards_what_was_held(tmp_path):
    """H5 (ruling Q2): greeted A leaves while ungreeted B holds a spoken line; the grace (4000 ms)
    fires, the journal is cleared and B's hold with it. B's later hello gets the live head only
    (the journal is empty), and a message published after the departure follows it live."""
    rig, server = make_server(tmp_path)
    submitted: list[Any] = []
    real_submit = rig.session.submit

    def spy(event: Any) -> None:
        submitted.append(event)
        real_submit(event)

    rig.session.submit = spy  # type: ignore[method-assign]
    publish_a_turn(server)
    a = greet(server)
    b = server.open()
    a.close()
    server.publish(spoken("t1-u9", "ZQXHELDBEFOREDEPARTURE"))
    rig.clock.advance(4001)
    departed = [e for e in submitted if isinstance(e, E.ClientDisconnected)]
    assert len(departed) == 1, (
        "premise (G2c): the last greeted page left and the grace fired although an ungreeted "
        f"socket is open; ClientDisconnected submitted {len(departed)} times"
    )

    hello(b)
    sent = b.sent()
    # T-0213 R-5 named re-pin: then the one settings state, last
    assert [type(m) for m in sent] == t0213.hello_with_settings_last(LIVE_HEAD), (
        "H5: after a departure the ungreeted connection's hold is discarded with the journal; its "
        f"hello gets the live head only. It got {kinds(sent)}"
    )
    assert not [r for r in wire(sent) if "ZQXHELDBEFOREDEPARTURE" in r]
    later = Publisher(server)
    later(spoken("t2-u1", "After the departure.", turn="t2"))
    assert_reply_then(b.sent(), sent, later.messages, "H5")


async def test_a_refused_hello_keeps_holding(tmp_path):
    """H1/G2e: a hello with the wrong ``protocol_version`` is answered with its error, at once, and
    releases nothing: the valid hello that follows gets the reply first, then the held messages."""
    _, server = make_server(tmp_path)
    conn = server.open()
    publish = Publisher(server)
    publish(spoken("t1-u1", "A held line."))
    publish(thinking(1))
    hello(conn, PROTOCOL_VERSION + 1)
    publish(thinking(2))
    reply = list(server.hello_reply())
    hello(conn)
    sent = conn.sent()

    assert sent and isinstance(sent[0], AstraError) and sent[0].code == "protocol_version", (
        "H3/G2e: the refusal is a direct reply and goes out ahead of every held message; the "
        f"connection's first messages were {kinds(sent[:4])}"
    )
    assert_reply_then(sent[1:], reply, publish.messages, "G2e/H2")


# --- N2: greeted connections are unchanged --------------------------------------------------------


async def test_greeted_fan_out_is_unchanged(tmp_path):
    """N2 (green on main on purpose): greeted A and ungreeted B. Each publish is in A's outbox at
    once (B's hold delays nothing), and each publish writes exactly one ``wire.out`` whose
    ``connections`` counts both sockets (Q5) and whose ``dropped`` is false."""
    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig)
        a = server.open()
        hello(a)
        b = server.open()
        lines = [spoken(f"t1-u{i}", f"ZQXFANOUT{i}") for i in range(3)]
        for line in lines:
            server.publish(line)
            assert encode(a.outbox.pending()[-1]) == encode(line), (
                f"N2: a publish reaches the greeted page at once; A's outbox ends {kinds(a.sent()[-2:])}"
            )
        assert not b.closed
    rows = [
        r["data"]
        for r in of_kind(read_records(log.path), "wire.out")
        if "ZQXFANOUT" in str(r["data"]["msg"].get("text", ""))
    ]
    assert [(d["msg"]["text"], d["connections"], d["dropped"]) for d in rows] == [
        (f"ZQXFANOUT{i}", 2, False) for i in range(3)
    ], f"N2 / T-0043 c7: one wire.out per publish, connections counts all sockets; {rows}"


# --- review rows (reviewer A, 2026-10-27) --------------------------------------------------------


async def test_pre_hello_errors_are_bounded_and_overflow_closes_the_connection(tmp_path):
    """T1 (H3 x H7, T-0028 c12, ruling A12): direct replies are not held, but they are not free
    either. A page that never greets and sends ``DEFAULT_OUTBOX + 1`` malformed frames is still
    open with ``DEFAULT_OUTBOX`` errors queued, and is closed with the overflow named on the next
    one; the queue never exceeds ``DEFAULT_OUTBOX``. Both directions: not closed early, not left
    open (and not silently dropping the refused error)."""
    _, server = make_server(tmp_path)
    conn = server.open()
    worst = 0
    for i in range(DEFAULT_OUTBOX + 1):
        conn.receive("{this is not json")
        worst = max(worst, len(conn.outbox))
        assert len(conn.outbox) <= DEFAULT_OUTBOX, (
            f"T1: {len(conn.outbox)} queued after {i + 1} malformed frames; the bound is "
            f"{DEFAULT_OUTBOX}"
        )
        if i == DEFAULT_OUTBOX - 1:
            assert not conn.closed, (
                f"T1, too wide: {DEFAULT_OUTBOX} errors are within the bound; the connection was "
                f"closed early ({conn.close_reason})"
            )
    assert conn.closed and "overflow" in (conn.close_reason or ""), (
        "T1 / ruling A12: one error past the bound closes the connection with the overflow named, "
        f"never a silent drop; closed={conn.closed}, reason={conn.close_reason!r}, "
        f"worst queue {worst}"
    )


async def test_a_repeated_hello_queues_its_reply_behind_what_is_already_owed(tmp_path):
    """T2 (N2, T-0037 "a reconnect's second hello"): on an already-greeted page a second hello's
    reply is a live message like any other, FIFO behind what is already queued for the page. Only
    the *first* hello's reply jumps ahead of what was held."""
    _, server = make_server(tmp_path)
    conn = greet(server)
    first_reply = list(conn.sent())
    line = spoken("t1-u1", "ZQXQUEUEDLINE before the second hello.")
    server.publish(line)
    second_reply = list(server.hello_reply())
    hello(conn)
    assert_reply_then(conn.sent(), first_reply, [line, *second_reply], "T2")


@pytest.mark.parametrize("cut", ["reset", "departure"])
async def test_a_cut_never_drops_a_pending_error(tmp_path, cut):
    """T3 (H3 x H4/H5): the reset and departure cuts discard *published* messages held for an
    ungreeted page, never the direct reply about its own frame. Garbage, then the cut, then the
    hello: the first message is still the malformed error."""
    rig, server = make_server(tmp_path)
    if cut == "departure":
        greet(server).close()  # the last greeted page leaves: the grace starts
    conn = server.open()
    conn.receive("{this is not json")
    server.publish(spoken("t1-u1", "ZQXHELD line before the cut."))
    if cut == "reset":
        server.publish(AstraSessionCleared(resets=1, ended_turn_id="t1", memory_forgotten=0))
    else:
        rig.clock.advance(4001)
    reply = list(server.hello_reply())
    hello(conn)
    sent = conn.sent()
    assert sent and isinstance(sent[0], AstraError) and sent[0].code == "malformed", (
        f"T3 ({cut}): the error about the page's own frame survives the cut and goes out first; "
        f"the connection's first messages were {kinds(sent[:3])}"
    )
    assert_reply_then(sent[1:], reply, [], f"T3 ({cut})")
