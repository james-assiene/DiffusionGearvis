"""The endpoint itself: a real loopback socket, the real codec (T-0028 criteria 1, 2, 9, 13).

Binding ``127.0.0.1:0`` is loopback and free, so this runs in the fast suite with no GPU and no
route off the box. Every wait carries a 2 s budget (docs/tasks/T-0028.md, "Regime").

What is asserted here and nowhere else is the *socket's* behaviour: where it binds, which
handshakes it refuses, and that the bytes a page receives decode under
``local_astra/contracts/codec.py`` into server→client messages. The seams are fixed in
``tests/unit/_hud_rig.py``'s module docstring.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
import websockets
from websockets.asyncio.client import connect as ws_connect

from local_astra.config import Services
from local_astra.contracts import events as E
from local_astra.contracts.codec import Malformed, Unknown, decode
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import (
    SERVER_TO_CLIENT,
    AstraAssistState,
    AstraTurnState,
)
from tests.unit._hud_rig import HudClient, answer, require_symbol, served, ui

pytestmark = pytest.mark.contract

#: Ruling A13: the drain budget is *virtual* milliseconds of the session's ScriptedClock, not
#: wall seconds — a budget in seconds buys less session progress on a loaded machine, which is
#: exactly the one-in-four flake. 5 s of virtual time is 100 rounds; a scripted turn here needs a
#: handful.
BUDGET_MS = 5_000
#: "read whatever is already queued": a few rounds, no session progress expected.
SETTLE_MS = 500
#: Real seconds, and only for the things that really are wall-clock: a handshake and one recv on
#: a live socket (docs/tasks/T-0028.md, "Regime": a 2 s per-test budget).
HANDSHAKE_S = 2.0


def turn_finished(received: list) -> bool:
    """An ``IDLE`` turn state that *follows* a ``SPEAKING`` one (ruling A10).

    The hello reply already carries an ``IDLE`` (criterion 4), so "some IDLE arrived" is true
    before the turn even opens and a drain keyed on it returns immediately — the turn's own
    messages would still be in flight and the assertions after it would race the socket.
    """
    states = [m.state for m in received if isinstance(m, AstraTurnState)]
    if "SPEAKING" not in states:
        return False
    return "IDLE" in states[states.index("SPEAKING") :]


async def test_it_binds_loopback_and_serves_the_page_url_path(tmp_path):
    """Criterion 1: loopback only, the path ``hud/src/ws.js`` already defaults to, 404 elsewhere.

    ``getsockname()`` is the assertion CLAUDE.md's "all services bind to loopback only" and
    docs/03 E8 ask for: ``0.0.0.0`` and ``::`` are reachable from the network even though the
    page's URL says ``127.0.0.1``.
    """
    hud_server = require_symbol("HudServer")
    async with served(tmp_path) as box:
        addresses = [s.getsockname() for s in box.server.sockets]
        assert addresses, "the server must expose its listening socket(s) (criterion 1)"
        for address in addresses:
            assert address[0] in ("127.0.0.1", "::1"), (
                f"criterion 1 / docs/03 E8: the listener binds loopback, never {address[0]!r}"
            )
        assert box.server.url.endswith("/v1/realtime"), (
            f"the page defaults to /v1/realtime (hud/src/ws.js, T-0008); the server serves "
            f"{box.server.url}"
        )
        assert box.server.port == addresses[0][1] and box.server.port != 0

        # a legitimate connection on the right path works…
        async with HudClient.connect(box.server.url) as client:
            await client.hello()
            await client.drain(
                until=lambda got: any(isinstance(m, AstraTurnState) for m in got),
                rig=box.rig,
                budget_ms=BUDGET_MS,
            )
        # …and another path is refused without touching the session
        turns_before = box.rig.state.turn_seq
        wrong = box.server.url.replace("/v1/realtime", "/v1/whatever")
        with pytest.raises(websockets.exceptions.InvalidStatus) as caught:
            async with ws_connect(wrong, open_timeout=HANDSHAKE_S):
                pass
        assert caught.value.response.status_code == 404, (
            f"criterion 1: a request on another path is refused with 404; got "
            f"{caught.value.response.status_code}"
        )
        assert box.rig.state.turn_seq == turns_before and not box.rig.machine.log, (
            "…and no session state changed: the machine saw "
            f"{[type(e).__name__ for e, _, _ in box.rig.machine.log]}"
        )

        # the constructor refuses a non-loopback host *before* a socket exists (criterion 1)
        for host in ("0.0.0.0", "::", "10.0.0.5"):
            with pytest.raises(ValueError):
                hud_server(box.rig.session, frame_sink=box.frames, host=host, port=0)
        assert Services().conductor_port == 8000, (
            "criterion 1: the default port is config.Services.conductor_port (8000) — the number "
            "the page's default URL carries"
        )


async def test_a_remote_origin_is_refused_before_any_message(tmp_path):
    """Criterion 2, adversarial: a page on a remote site can open ``ws://127.0.0.1:8000``.

    Loopback binding does not stop it — the browser is on this machine. The handshake carries
    ``Origin: https://evil.example``; the answer must be HTTP 403 *before* any message is read,
    nothing may reach the session, and a legitimate connection afterwards must still work.
    """
    async with served(tmp_path) as box:
        with pytest.raises(websockets.exceptions.InvalidStatus) as caught:
            async with ws_connect(
                box.server.url, origin="https://evil.example", open_timeout=HANDSHAKE_S
            ):
                pass
        assert caught.value.response.status_code == 403, (
            "criterion 2: a non-loopback Origin is refused with 403 at the handshake; got "
            f"{caught.value.response.status_code}"
        )
        assert not box.rig.machine.log, (
            "…before any message is read: zero events were submitted, but the machine saw "
            f"{[type(e).__name__ for e, _, _ in box.rig.machine.log]}"
        )
        assert box.rig.session.assists.flags.any_on() is False

        async with HudClient.connect(box.server.url, origin="http://127.0.0.1:1420") as client:
            await client.hello()
            received = await client.drain(
                until=lambda got: any(isinstance(m, AstraTurnState) for m in got),
                rig=box.rig,
                budget_ms=BUDGET_MS,
            )
        assert any(isinstance(m, AstraAssistState) for m in received), (
            "criterion 2: the fence lets the real page through — a loopback Origin still gets "
            f"its hello reply; it got {[getattr(m, 'type', m) for m in received]}"
        )
        assert require_symbol("HudServer").origin_allowed(None) is True, (
            "an absent Origin (a native client, a file:// page) is allowed (criterion 2)"
        )
        assert require_symbol("HudServer").origin_allowed("https://evil.example") is False


async def test_two_clients_both_receive_and_the_last_disconnect_is_one_event(tmp_path):
    """Criterion 13: fan-out, a clean disconnect, and one ``client.disconnected`` at the end.

    A second HUD window (or the page reloading while the old socket lingers) must not end the
    session: ``events.ClientDisconnected`` takes the machine to IDLE and aborts what is live
    (docs/02 §4), so emitting it on the *first* disconnect would kill a turn the user is still
    watching in the other window.
    """
    async with served(tmp_path, script=[answer("Here it is. ")], ms_per_word=100) as box:
        rig = box.rig
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            async with HudClient.connect(box.server.url) as two:
                await two.hello()
                await two.drain(
                    until=lambda got: any(isinstance(m, AstraTurnState) for m in got),
                    rig=rig,
                    budget_ms=BUDGET_MS,
                )
                await rig.ask("what is on this screen")
                await rig.speak_out()
                for client in (one, two):
                    await client.drain(until=turn_finished, rig=rig, budget_ms=BUDGET_MS)
                states_one = [m.state for m in one.received if isinstance(m, AstraTurnState)]
                states_two = [m.state for m in two.received if isinstance(m, AstraTurnState)]
                assert states_one == states_two and "SPEAKING" in states_one, (
                    "criterion 13: both connections receive every egress message; "
                    f"one saw {states_one}, two saw {states_two}"
                )
            # `two` is gone; `one` is still connected
            await rig.pump()
            assert not rig.machine.events(E.ClientDisconnected), (
                "criterion 13: a disconnect while another page is connected changes nothing; "
                "the session was told the client disconnected"
            )
            rig.llm.script = [answer("Still here. ")]
            await rig.ask("are you still there")
            await rig.speak_out()
            assert len(rig.lines()) == 2, (
                f"…and the session keeps running with its history intact; lines={rig.lines()}"
            )

        await rig.until(
            lambda: bool(rig.machine.events(E.ClientDisconnected)),
            "the last disconnect never reached the session",
            limit_ms=require_symbol("RECONNECT_GRACE_MS") + 1_000,
        )
        assert len(rig.machine.events(E.ClientDisconnected)) == 1, (
            "criterion 13: exactly one client.disconnected, when the last connection goes; got "
            f"{rig.machine.events(E.ClientDisconnected)}"
        )


async def test_every_egress_message_decodes_and_is_server_to_client(tmp_path):
    """Criterion 9's typing half, over the real socket and the real codec.

    Nothing the page receives may be ``Malformed``/``Unknown`` under
    ``local_astra/contracts/codec.py``, every type must be in ``SERVER_TO_CLIENT``, and no
    ``astra.canvas.*`` may appear: amended by T-0128c A10 -- the relay (``canvas_relay.py``) is the
    only source of ``astra.canvas.*`` and it only relays real tap frames, so with no tap frames (this
    rig has no tap) a fake turn publishes zero ``astra.canvas.*``.
    """
    async with served(
        tmp_path, script=[answer("Here it is. "), ui(canvas_idx=1)], ms_per_word=100
    ) as box:
        async with HudClient.connect(box.server.url) as client:
            await client.hello()
            await box.rig.ask("show me the surface")
            await box.rig.speak_out()
            received = await client.drain(until=turn_finished, rig=box.rig, budget_ms=BUDGET_MS)

        assert len(received) >= 5, (
            f"a whole turn puts more than {len(received)} messages on the wire (criterion 9)"
        )
        bad = [m for m in received if isinstance(m, Malformed | Unknown)]
        assert not bad, f"the production codec could not decode what the server sent: {bad}"
        types = [m.type for m in received]
        assert set(types) <= SERVER_TO_CLIENT, (
            f"criterion 9: every message is a server→client type; got {sorted(set(types))}"
        )
        assert not [t for t in types if t.startswith("astra.canvas.")], (
            f"T-0128c A10: with no tap frames a fake turn publishes zero astra.canvas.*: {types}"
        )
        assert "astra.ui.surface" in types and "astra.spoken" in types, (
            f"…and the turn really produced both channels; it produced {sorted(set(types))}"
        )


async def test_the_listening_socket_is_the_only_socket_and_it_is_loopback(tmp_path):
    """Criterion 1 / docs/03 E8, in the fast suite: no listener on a routable address.

    The group-B twin (criterion 22) runs ``ss -tunp`` on the box; here the assertion is made on
    the socket objects themselves, which is what a unit-testable fence can say.
    """
    async with served(tmp_path) as box:
        for sock in box.server.sockets:
            assert sock.family in (socket.AF_INET, socket.AF_INET6)
            host = sock.getsockname()[0]
            assert host in ("127.0.0.1", "::1"), (
                f"docs/03 E8: {host!r} is reachable off the box; the listener must be loopback"
            )
        async with HudClient.connect(box.server.url) as client:
            await client.hello()
            raw = await asyncio.wait_for(client.ws.recv(), timeout=HANDSHAKE_S)
            assert not isinstance(decode(raw), Malformed | Unknown), (
                f"the hello reply decodes under the production codec; got {raw!r}"
            )
            peer = client.ws.local_address[0]
            assert peer in ("127.0.0.1", "::1"), (
                f"the client's own end is loopback too; it is {peer!r}"
            )


async def test_a_frame_posted_over_the_socket_reaches_the_turn(tmp_path):
    """Criterion 5 over the real wire: the page's frame is the frame the turn answers about.

    ``served(browser_source=True)`` wires the production shape of criterion 15 — one
    ``BrowserFrameSource`` is both the session's ``PerceptionClient`` and the server's frame sink.
    """
    from tests.unit._hud_rig import astra_frame

    posted = astra_frame("c")
    async with served(tmp_path, script=[answer("It is a stack trace. ")], ms_per_word=100) as box:
        async with HudClient.connect(box.server.url) as client:
            await client.hello()
            await client.send(posted)
            await client.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
            stored = await box.frames.get_frame(Source.SCREEN, at_ms=box.rig.clock.now_ms)
            assert stored is not None and stored.phash == posted.phash, (
                "criterion 5: a frame posted over the socket reaches the injected "
                f"BrowserFrameSource unmodified; the ring holds {stored}"
            )
            await box.rig.ask("what is on this screen")
            await box.rig.speak_out()
        phashes = [ep.phash for ep in box.rig.memory.episodes.values()]
        assert posted.phash in phashes, (
            f"…and it is the frame the turn snapshotted and stored; memory holds {phashes}"
        )


async def test_a_frame_is_never_stored_unvalidated(tmp_path):
    """Criterion 5, negative: the server adds no validation and rescues none.

    ``BrowserFrameSource`` counts the refusal (T-0008 criteria 14-18); the socket stays open.
    """
    from tests.unit._hud_rig import astra_frame

    async with served(tmp_path) as box:
        bad = astra_frame("d").model_copy(update={"phash": "nope"})
        async with HudClient.connect(box.server.url) as client:
            await client.hello()
            await client.send(bad)
            await client.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
            assert box.frames.rejected == 1, (
                "the frame reaches BrowserFrameSource and is counted there, not filtered out by "
                f"a second rule in the server; rejected={box.frames.rejected}"
            )
            await client.send(astra_frame("e"))
            await client.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
        assert box.frames.rejected == 1, "a good frame after a bad one is still accepted"


async def test_an_oversize_message_is_answered_and_the_socket_survives(tmp_path):
    """Ruling A13 / criterion 3, over a **real** socket: 8 MB in, ``malformed`` out, still alive.

    The unit twin (``tests/unit/test_hud_ingress.py``) hands the bytes to a connection object, so
    it says nothing about the library sitting under it: with ``websockets``' ``max_size`` below
    ``codec.MAX_MESSAGE_BYTES`` the frame never reaches the server at all and the page gets a bare
    1009 close instead of the error T-0008 criterion 2 expects. That choice is only pinned if a
    test really pushes 8 MB down a socket, which is what this row does (the Implementation notes
    ask for ``max_size`` to be picked deliberately; this is the deliberation, written down).

    Proved by mutation: ``MAX_SIZE = MAX_MESSAGE_BYTES // 2`` in the server makes this row red
    with the 1009 close named in the message, and green again unmutated.
    """
    from websockets.exceptions import ConnectionClosed

    from local_astra.contracts.codec import MAX_MESSAGE_BYTES
    from local_astra.contracts.protocol import AstraError
    from tests.fakes.hud_server import oversized_message
    from tests.unit._hud_rig import astra_frame

    huge = oversized_message()
    assert len(huge.encode()) > MAX_MESSAGE_BYTES, (
        "the adversarial message must really be over the codec's byte limit (T-0027)"
    )

    async with served(tmp_path) as box, HudClient.connect(box.server.url) as client:
        await client.hello()
        await client.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
        try:
            await client.send(huge)
            await client.drain(
                until=lambda got: any(
                    isinstance(m, AstraError) and m.code == "malformed" for m in got
                ),
                rig=box.rig,
                budget_ms=BUDGET_MS,
            )
        except ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd is not None else "?"
            raise AssertionError(
                f"ruling A13: the library killed the connection (close {code}) before the "
                "server could answer. websockets' max_size must sit *above* "
                "codec.MAX_MESSAGE_BYTES so an oversize message is reported as "
                "astra.error(code='malformed') (T-0008 criterion 2 expects the error, not a "
                "bare close, and the codec owns the size rule — T-0027)"
            ) from None

        errors = [m for m in client.received if isinstance(m, AstraError)]
        assert [m.code for m in errors] == ["malformed"], (
            f"criterion 3: one astra.error(malformed) and nothing else; got "
            f"{[m.code for m in errors]}"
        )
        assert box.server.counts.get("malformed") == 1, box.server.counts

        # …and the socket is still open and usable
        await client.send(astra_frame("h"))
        await client.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
        assert (
            box.frames.rejected == 0
            and (await box.frames.get_frame(Source.SCREEN, at_ms=box.rig.clock.now_ms)) is not None
        ), (
            "criterion 3: the connection survives the oversize message and the next "
            "legitimate frame is handled normally"
        )


async def test_a_reconnect_within_the_grace_window_is_not_a_disconnect(tmp_path):
    """Criterion 13's "never on a reconnect", with the page's own backoff (ruling A15).

    T-0008's page reconnects after a drop; ``events.ClientDisconnected`` takes the machine to
    IDLE and aborts whatever is live (docs/02 §4), so reporting one for a reload would kill the
    turn the user is waiting for. The grace window is ``RECONNECT_GRACE_MS`` on the *session's*
    clock (the transport owns no clock), so this drives it in virtual time: reconnect inside the
    window, then run past it and assert the session was never told.

    T-0166 (G2a) moved the grace cancel from ``HudServer.open()`` to the valid hello; this row's
    page greets, so it covers the hello path (the open-alone case is
    ``tests/unit/test_hud_reconnect_grace.py``).
    """
    grace = require_symbol("RECONNECT_GRACE_MS")
    assert grace >= 100, f"a grace window of {grace} ms is not a window"

    async with served(tmp_path) as box:
        rig = box.rig
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            await one.drain(until=None, rig=rig, budget_ms=SETTLE_MS)
        # the page is gone; give the server a moment of *real* time to notice the close
        await asyncio.sleep(0.05)
        await rig.advance(grace // 5, step=grace // 5)
        assert not rig.machine.events(E.ClientDisconnected), (
            "criterion 13: nothing is reported while the grace window is still open"
        )

        async with HudClient.connect(box.server.url) as two:
            await two.hello()
            received = await two.drain(
                until=lambda got: any(isinstance(m, AstraTurnState) for m in got),
                rig=rig,
                budget_ms=BUDGET_MS,
            )
            assert any(isinstance(m, AstraAssistState) for m in received), (
                "criterion 13: the reconnected page sends a fresh hello and gets the current "
                f"assist and turn state (criterion 4); it got {[m.type for m in received]}"
            )
            # run well past the window with the new page connected
            await rig.advance(grace * 2, step=50)
            assert not rig.machine.events(E.ClientDisconnected), (
                "criterion 13: a reconnect inside the grace window is not a disconnect — the "
                "session must never see client.disconnected here, which would take the machine "
                "to IDLE and abort a live turn (docs/02 §4)"
            )
            assert len(rig.lines()) == 0

        # …and when the last page really goes, it is reported exactly once
        await asyncio.sleep(0.05)
        await rig.until(
            lambda: bool(rig.machine.events(E.ClientDisconnected)),
            "the last disconnect never reached the session after the grace window",
            limit_ms=grace * 4,
        )
        assert len(rig.machine.events(E.ClientDisconnected)) == 1, (
            f"exactly one, at the end; got {rig.machine.events(E.ClientDisconnected)}"
        )


async def test_a_stalled_client_is_closed_on_the_wire_with_the_overflow_named(tmp_path):
    """Ruling A12/A18(1): overflow of a non-droppable kind closes *that* socket, visibly.

    Criterion 12 lets only ``astra.turn.state`` be dropped; anything else overflowing must close
    the connection "with a logged reason rather than silently losing a spoken line and desyncing
    the transcript". A log line is not enough on its own: the page has stopped reading, so it will
    not send another frame either, and without a close frame the peer keeps a socket it believes
    is live while the writer task outlives it. This row is about what the *client* sees.

    The overflow is forced rather than waited for: publishing more than ``outbox_size``
    non-droppable messages in one synchronous burst never lets the writer task run, which is
    exactly the state a stalled reader puts the server in, and is deterministic.

    Proved by mutation: replacing the closer task's body with ``asyncio.sleep(3600)`` in a scratch
    copy of the branch makes this red (the client waits for a close that never comes).
    """
    from websockets.exceptions import ConnectionClosed

    from local_astra.contracts.protocol import AstraSpoken

    grace = require_symbol("RECONNECT_GRACE_MS")
    async with served(tmp_path, outbox_size=4) as box:
        rig = box.rig
        async with HudClient.connect(box.server.url) as client:
            await client.hello()
            await client.drain(until=None, rig=rig, budget_ms=SETTLE_MS)

            # the page stops reading: nothing below awaits the socket, so the writer never runs
            for i in range(24):
                box.server.publish(
                    AstraSpoken(turn_id="t1", utterance_id=f"t1-u{i}", text="a spoken line")
                )

            closed: ConnectionClosed | None = None
            deadline = asyncio.get_running_loop().time() + 1.0
            try:
                while asyncio.get_running_loop().time() < deadline:
                    await asyncio.wait_for(client.ws.recv(), timeout=0.05)
            except ConnectionClosed as exc:
                closed = exc
            except TimeoutError:
                pass
            assert closed is not None, (
                "ruling A12: the client must be *told*. It stopped reading, the outbox overflowed "
                "with astra.spoken (not a droppable kind), and within a second the server owes it "
                "a close frame — otherwise the page holds a socket it thinks is live and the "
                "spoken pane silently desyncs (criterion 12)"
            )
            received = closed.rcvd
            assert received is not None and received.code == 1011, (
                f"the close is 1011 (internal error), not a normal 1000; got {received}"
            )
            assert "overflow" in (received.reason or "").lower(), (
                "…and the reason names the overflow, so the page can log why it was dropped; the "
                f"reason was {received.reason!r}"
            )

        await asyncio.sleep(0.05)
        alive = [
            task.get_name()
            for task in asyncio.all_tasks()
            if task.get_name().startswith(("hud-writer", "hud-closer")) and not task.done()
        ]
        assert not alive, (
            f"ruling A12: no per-connection task outlives the socket it serves; still running: "
            f"{alive}"
        )

        await rig.until(
            lambda: bool(rig.machine.events(E.ClientDisconnected)),
            "the closed connection was the last one, so client.disconnected is owed after the "
            "grace window (criterion 13)",
            limit_ms=grace * 4,
        )
        assert len(rig.machine.events(E.ClientDisconnected)) == 1
