"""T-0162 group A, criterion A-5 (and A-6's FIFO clause): the transport side of the reset.

- ``route`` turns ``astra.session.reset`` into exactly one ``E.SessionReset(t_ms=clock.now_ms)`` on
  the session's one pump (``session.submit``) -- the transport decides nothing;
- the journal is cleared when ``astra.session.cleared`` is published, so a page that reconnects
  after a Restart is never caught up on the turn the reset cut;
- ``hello_reply`` *begins* with ``AstraSessionCleared(replay=True, resets=n, ...)`` once the
  session has been reset (n >= 1), and is unchanged when it never was;
- messages already queued for a page go out before the echo (FIFO).

The reset is driven end to end through the real ``Session`` (started, over the fakes) so that the
cleared message the server sees is the one the session publishes -- whichever of the two keeps the
reset count. Inbound frames are raw JSON through ``HudConnection.receive``, so until the contract
declares ``astra.session.reset`` the row fails on its assertion (the codec reports an unknown type),
never on an import.

Regime: no socket (``start()`` is never called), ``ScriptedClock``; n = 1 per row.
"""

from __future__ import annotations

import json
from typing import Any

from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import AstraHello, AstraSpoken, AstraTranscript
from tests.unit import test_t0213_settings_wire as t0213  # T-0213 R-5 named re-pin
from tests.unit.test_hud_hello_replay import (
    LIVE_HEAD,
    after_the_live_messages,
    make_server,
    publish_a_turn,
    transcript,
    types_of,
)

CLEARED = "astra.session.cleared"
RESET_WIRE = json.dumps({"type": "astra.session.reset"})


def greet(server: Any) -> Any:
    connection = server.open()
    connection.receive(encode(AstraHello()))
    return connection


class SubmitSpy:
    """Wraps ``session.submit``: records every event and still delivers it to the pump."""

    def __init__(self, session: Any) -> None:
        self.events: list[Any] = []
        self._inner = session.submit
        session.submit = self

    def __call__(self, event: Any) -> None:
        self.events.append(event)
        self._inner(event)

    def resets(self) -> list[Any]:
        return [e for e in self.events if getattr(e, "type", None) == "session.reset"]


async def reset_through_the_wire(rig: Any, server: Any) -> Any:
    """A page's Restart: one raw ``astra.session.reset`` on a greeted connection, then the pump."""
    page = greet(server)
    page.receive(RESET_WIRE)
    await rig.pump()
    await rig.advance(200)
    return page


async def test_the_route_submits_one_session_reset_on_the_pump(tmp_path):
    rig, server = make_server(tmp_path)
    await rig.session.start()
    try:
        spy = SubmitSpy(rig.session)
        page = greet(server)
        routed = server.counts["routed"]
        now = rig.clock.now_ms
        page.receive(RESET_WIRE)
        assert server.counts["malformed"] == 0 and server.counts["unhandled"] == 0, (
            f"A-1: astra.session.reset is a known client->server type; counts {server.counts}"
        )
        resets = spy.resets()
        assert len(resets) == 1 and len(spy.events) == 1, (
            f"A-5: the route submits exactly one SessionReset and nothing else; got "
            f"{[getattr(e, 'type', '?') for e in spy.events]}"
        )
        assert isinstance(resets[0], E.SessionReset) and resets[0].t_ms == now, resets
        assert server.counts["routed"] == routed + 1, server.counts
        assert not [m for m in page.sent() if getattr(m, "type", "") == "astra.error"], (
            "the page is not answered with an error"
        )
    finally:
        await rig.session.aclose()


async def test_the_journal_is_cleared_when_cleared_is_published(tmp_path):
    rig, server = make_server(tmp_path)
    await rig.session.start()
    try:
        publish_a_turn(server)
        assert after_the_live_messages(greet(server).sent()), "premise: the journal holds t1"
        await reset_through_the_wire(rig, server)
        reply = greet(server).sent()
        assert reply and reply[0].type == CLEARED, (
            f"A-5: after a reset the hello begins with the cleared replay; got {types_of(reply)}"
        )
        rest = reply[1:]
        # T-0213 R-5 named re-pin: then the one settings state, last
        assert [type(m) for m in rest] == t0213.hello_with_settings_last(LIVE_HEAD), (
            "A-5: the journal was cleared with the echo, so nothing of the cut turn is replayed: "
            f"got {types_of(rest)}"
        )
    finally:
        await rig.session.aclose()


async def test_the_hello_after_a_reset_starts_with_the_cleared_replay(tmp_path):
    rig, server = make_server(tmp_path)
    await rig.session.start()
    try:
        live = await reset_through_the_wire(rig, server)
        echoes = [m for m in live.sent() if getattr(m, "type", "") == CLEARED]
        assert len(echoes) == 1 and echoes[0].replay is False, (
            f"A-3: the connected page gets one live echo; got {echoes}"
        )
        echo = echoes[0]
        reply = greet(server).sent()
        first = reply[0]
        assert first.type == CLEARED and first.replay is True, types_of(reply)
        assert (
            (first.resets, first.ended_turn_id, first.memory_forgotten)
            == (
                echo.resets,
                echo.ended_turn_id,
                echo.memory_forgotten,
            )
            == (1, None, 0)
        ), (first, echo)
        assert [type(m) for m in reply[1 : 1 + len(LIVE_HEAD)]] == list(LIVE_HEAD), types_of(reply)
        assert sum(m.type == CLEARED for m in reply) == 1, "one cleared per hello"

        # a second Restart: the prefix carries the session's count, n = 2
        await reset_through_the_wire(rig, server)
        reply = greet(server).sent()
        assert reply[0].type == CLEARED and reply[0].resets == 2, types_of(reply)
        assert sum(m.type == CLEARED for m in reply) == 1

        # a turn published after the reset is journalled and replayed as before, after the head
        server.publish(transcript("a new question", final=True, turn="t2"))
        reply = greet(server).sent()
        assert reply[0].type == CLEARED
        caught_up = after_the_live_messages(reply[1:])
        assert [m.text for m in caught_up if isinstance(m, AstraTranscript)] == ["a new question"]
    finally:
        await rig.session.aclose()


async def test_queued_messages_go_out_before_the_echo(tmp_path):
    """A-6: "Messages already in an outbox are delivered before the echo (FIFO)"."""
    rig, server = make_server(tmp_path)
    await rig.session.start()
    try:
        page = greet(server)
        server.publish(AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="Already queued."))
        page.receive(RESET_WIRE)
        await rig.pump()
        await rig.advance(200)
        sent = types_of(page.sent())
        assert CLEARED in sent, f"A-3: the echo reaches the connected page; sent {sent}"
        assert sent.index("astra.spoken") < sent.index(CLEARED), sent
    finally:
        await rig.session.aclose()


#: vLLM's own request id shape, which the canvas relay carries (never the session's machine id)
RELAYED_ID = "chatcmpl-0123456789abcdef-01234567"
FRESH_ID = "chatcmpl-fedcba9876543210-76543210"


def canvas_step(request_id: str) -> Any:
    from local_astra.contracts.protocol import AstraCanvasStep
    from tests.unit._examples import example_of

    return example_of(AstraCanvasStep).model_copy(update={"request_id": request_id})


def canvas_ids(connection: Any) -> list[str]:
    return [
        m.request_id for m in connection.sent() if getattr(m, "type", "") == "astra.canvas.step"
    ]


async def test_a_relayed_canvas_of_a_pre_reset_request_is_dropped_after_cleared(tmp_path):
    """Amendment (d) / R1: the relay publishes straight into the server, with vLLM's ids.

    A request relayed before the Restart keeps streaming its steps for a moment after the abort; once
    ``astra.session.cleared`` has been published, those frames stop at the server. A request the server first
    sees after the reset is delivered (the too-wide direction). Each check uses its own fresh page so the
    outbox's canvas coalescing cannot hide or merge a frame.
    """
    rig, server = make_server(tmp_path)
    await rig.session.start()
    try:
        before = greet(server)
        server.publish(canvas_step(RELAYED_ID))
        assert canvas_ids(before) == [RELAYED_ID], "premise: the relay frame reaches a page"
        await reset_through_the_wire(rig, server)
        stale_page = greet(server)
        assert stale_page.sent()[0].type == CLEARED, "premise: the reset was published"
        server.publish(canvas_step(RELAYED_ID))
        assert canvas_ids(stale_page) == [], (
            f"R1: a canvas step of {RELAYED_ID}, relayed before the reset, repaints the page after the echo"
        )
        fresh_page = greet(server)
        server.publish(canvas_step(FRESH_ID))
        assert canvas_ids(fresh_page) == [FRESH_ID], (
            "too wide: a request first relayed after the reset is dropped"
        )
    finally:
        await rig.session.aclose()


async def test_the_hello_without_a_reset_is_unchanged(tmp_path):
    """A-7 negative (green on main on purpose): no Restart, no cleared, today's reply exactly."""
    _, server = make_server(tmp_path)
    reply = greet(server).sent()
    # T-0213 R-5 named re-pin: then the one settings state, last
    assert [type(m) for m in reply] == t0213.hello_with_settings_last(LIVE_HEAD), types_of(reply)
    script = publish_a_turn(server)
    reply = greet(server).sent()
    assert CLEARED not in types_of(reply)
    assert [type(m) for m in after_the_live_messages(reply)][: len(script)] == [
        type(m) for m in script
    ], types_of(reply)
