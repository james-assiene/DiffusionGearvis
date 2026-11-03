"""T-0166 H1, H3, H4 over a real loopback socket: nothing published reaches the wire before the
connection's hello reply (docs/tasks/T-0166.md, "Hold until hello").

The unit half (``tests/unit/test_hud_hello_gate.py``) pins the order of ``conn.sent()``; this
module pins what a real ``websockets`` client actually *receives*: a client that connects and does
not say hello reads nothing published, however long it waits; an error about its own frame does
reach it (a direct reply, not a publish); after its hello it reads the reply first, then the held
messages in publication order.

**Deterministic under the fix**, at any load average: the server never writes a held message
before the hello, so the 300 ms silence window cannot be "too short" for a correct server. On main
it is red every time, because the server writes each publish as soon as its writer task runs.

The "connection accepted" moment is observed by wrapping ``server.open`` (the handler's first
call), so every publish below is made to a connection the server already holds; nothing here
races the handshake.

Regime: real ``websockets`` on 127.0.0.1, port 0, CPython 3.11; the session's clock is the rig's
``ScriptedClock`` and is never advanced (so the session publishes nothing of its own); turn states
published at 20 Hz for 200 ms of real time; a 300 ms ``recv`` timeout for the silence window; n = 1
per row.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_astra.contracts.codec import decode, encode
from local_astra.contracts.protocol import (
    AstraError,
    AstraSessionCleared,
    AstraSpoken,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from tests.unit._hud_rig import HudClient, served

pytestmark = pytest.mark.contract

#: the pre-hello silence window (docs/tasks/T-0166.md, Regime)
SILENCE_S = 0.3
#: real seconds for one expected frame on a live loopback socket
RECV_S = 2.0


def kinds(messages: list[Any]) -> list[str]:
    return [f"{m.type}{'(replay)' if getattr(m, 'replay', False) else ''}" for m in messages]


def thinking(i: int, turn: str = "t1") -> AstraTurnState:
    return AstraTurnState(state=TurnState.THINKING, turn_id=turn, thinking_ms=50 * i)


def watch_open(server: Any) -> asyncio.Event:
    """Set once the socket handler has created its connection (``server.open()``)."""
    opened = asyncio.Event()
    real = server.open

    def open_and_note() -> Any:
        connection = real()
        opened.set()
        return connection

    server.open = open_and_note
    return opened


async def nothing_arrives(client: HudClient, what: str) -> None:
    try:
        raw = await asyncio.wait_for(client.ws.recv(), timeout=SILENCE_S)
    except TimeoutError:
        return
    raise AssertionError(f"{what}: the client read {decode(raw)!r} before its hello")


async def read_n(client: HudClient, n: int) -> list[Any]:
    """Exactly the next ``n`` frames, decoded by the production codec."""
    got: list[Any] = []
    try:
        while len(got) < n:
            got.append(decode(await asyncio.wait_for(client.ws.recv(), timeout=RECV_S)))
    except TimeoutError:
        raise AssertionError(f"expected {n} frames, read {kinds(got)}") from None
    return got


def assert_sequence(got: list[Any], expected: list[Any], criterion: str) -> None:
    assert [encode(m) for m in got] == [encode(m) for m in expected], (
        f"{criterion}: on the wire, after the hello: the reply, then the held messages in "
        f"publication order. Expected {kinds(expected)}; read {kinds(got)}"
    )


async def test_nothing_published_reaches_the_wire_before_the_hello(tmp_path):
    """H1: connect, publish a spoken line and turn states at 20 Hz for 200 ms; a ``recv`` with a
    300 ms timeout gets nothing. After the hello the first frames are the reply, then the held
    messages in order."""
    async with served(tmp_path) as box:
        opened = watch_open(box.server)
        async with HudClient.connect(box.server.url) as client:
            await asyncio.wait_for(opened.wait(), timeout=RECV_S)
            held = [spoken := AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="Held.")]
            box.server.publish(spoken)
            for i in range(4):
                box.server.publish(state := thinking(i))
                held.append(state)
                await asyncio.sleep(0.05)
            await nothing_arrives(client, "H1")

            reply = list(box.server.hello_reply())
            await client.hello()
            got = await read_n(client, len(reply) + len(held))
        assert_sequence(got, reply + held, "H1/H2")


async def test_a_reset_before_the_hello_puts_the_cleared_replay_first_on_the_wire(tmp_path):
    """H4 / T-0162 D-3: a reset published while the page is ungreeted cuts what was held before
    it; the first decoded frame is ``astra.session.cleared`` with ``replay: true``, the pre-reset
    line never arrives, and exactly one cleared does."""
    async with served(tmp_path) as box:
        opened = watch_open(box.server)
        async with HudClient.connect(box.server.url) as client:
            await asyncio.wait_for(opened.wait(), timeout=RECV_S)
            box.server.publish(
                AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="ZQXPRERESET old line.")
            )
            box.server.publish(
                AstraSessionCleared(resets=1, ended_turn_id="t1", memory_forgotten=0)
            )
            post = [
                AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="The new session."),
                thinking(1, turn="t2"),
            ]
            for message in post:
                box.server.publish(message)
            await nothing_arrives(client, "H4")

            reply = list(box.server.hello_reply())
            await client.hello()
            got = await read_n(client, len(reply) + len(post))
            await nothing_arrives(client, "H4 (nothing after the expected frames)")
        assert isinstance(got[0], AstraSessionCleared) and got[0].replay, (
            f"H4: the first frame on the wire is the replayed cleared; read {kinds(got[:2])}"
        )
        assert_sequence(got, reply + post, "H4")
        assert not [m for m in got if "ZQXPRERESET" in encode(m)], kinds(got)
        assert len([m for m in got if isinstance(m, AstraSessionCleared)]) == 1, kinds(got)


async def test_an_error_before_hello_still_reaches_the_wire(tmp_path):
    """H3 (T-0028 c4, T-0008 c2): garbage before the hello is answered with
    ``astra.error(code="malformed")`` at once, while the held published messages are still not
    received; after the hello, the reply and then the held messages."""
    async with served(tmp_path) as box:
        opened = watch_open(box.server)
        async with HudClient.connect(box.server.url) as client:
            await asyncio.wait_for(opened.wait(), timeout=RECV_S)
            held = [
                AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="Held."),
                thinking(1),
            ]
            for message in held:
                box.server.publish(message)
            await client.send("{this is not json")
            (first,) = await read_n(client, 1)
            assert isinstance(first, AstraError) and first.code == "malformed", (
                "H3: the error about the page's own frame is the first thing on the wire, ahead "
                f"of every held message; read {kinds([first])}"
            )
            await nothing_arrives(client, "H3 (held messages after the error)")

            reply = list(box.server.hello_reply())
            await client.hello()
            got = await read_n(client, len(reply) + len(held))
        assert_sequence(got, reply + held, "H3/H2")
