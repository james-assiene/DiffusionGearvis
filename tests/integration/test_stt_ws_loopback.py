"""The same script as the unit double, over a **real** websocket on 127.0.0.1 — T-0007a.

Marked ``integration`` and *not* ``gpu``: no service and no model, but a real handshake, real
binary frames and a real close. This is the test that catches "works against an object, breaks on
a socket" — a client that never awaits the connect, sends ``str`` where the server wants binary,
or leaves the close handshake to the garbage collector (``scripts/demo/_listen.py::deliver``'s
measured 10 s hang). The server is ``tests/fakes/moshi_stt_double.py`` face (b), which follows the
``tests/fakes/hud_server.py`` precedent and decodes everything with the real ``msgpack.unpackb``.

No ``connect=`` is injected here on purpose: this exercises the client's *default* transport, the
one that will face moshi-server 0.6.4.
"""

from __future__ import annotations

import struct

import pytest

from local_astra.config import Services
from tests.fakes.moshi_stt_double import (
    EventDrain,
    SttLoopbackServer,
    build_stt_client,
    step,
    types_of,
    wait_until,
    word,
)

pytestmark = pytest.mark.integration

BLOCK_SAMPLES = 1920


def block(value: float = 0.5, n: int = BLOCK_SAMPLES) -> tuple[bytes, list[float]]:
    raw = struct.pack(f"<{n}f", *([value] * n))
    return raw, list(struct.unpack(f"<{n}f", raw))


async def test_the_script_survives_a_real_handshake_real_frames_and_a_real_close():
    with SttLoopbackServer() as server:
        client = build_stt_client(Services(stt=server.url))
        drain = EventDrain(client)
        raw, oracle = block()
        await client.feed(raw, 0)
        await wait_until(
            lambda: bool(server.messages_of("Audio")), timeout_s=5.0, what="the Audio frame"
        )

        assert server.request_paths[0].endswith("?auth_id=public_token")
        sent = server.messages_of("Audio")[0]
        assert set(sent) == {"type", "pcm"} and list(sent["pcm"]) == oracle

        server.push(word("hello", 0.5))
        server.push(word("there", 0.75))
        server.push(word("tail", 1.0))
        server.push(step(step_idx=1))  # in-order on the socket: its event proves all three landed

        seen = await drain.quiet(window_s=1.0)
        assert "step" in types_of(seen), f"nothing came back over the socket: {types_of(seen)}"
        words = [e for e in seen if e.type == "word"]
        assert [e.text for e in words[:2]] == ["hello", "there"]
        assert [e.start_ms for e in words[:2]] == [500, 750]

        # The marker is asked for only once the words are on this side, because the echo travels
        # a different path than the pushes. T-0038 may publish "tail" at the echo (it has no
        # EndWord), so the flush is asserted to be *last*, not to be alone.
        await client.mark(1500)
        tail = await drain.quiet(window_s=1.0)
        assert types_of(tail)[-1:] == ["flushed"], f"the flush must land last: {types_of(tail)}"
        assert tail[-1].t_eot_ms == 1500

        await client.close()
        await drain.expect_end(timeout_s=5.0)
        assert server.wait_for(lambda s: s.client_gone), "the socket was never closed"


async def test_the_client_talks_only_to_loopback_and_opens_one_connection():
    """docs/02 §9 on the wire, and ``batch_size = 2``: one connection per client, reused."""
    with SttLoopbackServer() as server:
        client = build_stt_client(Services(stt=server.url))
        raw, _ = block()
        for i in range(5):
            await client.feed(raw, i * 80)
        await wait_until(
            lambda: len(server.messages_of("Audio")) == 5, timeout_s=5.0, what="five blocks"
        )
        assert server.connections == 1
        assert [peer[0] for peer in server.peers] == ["127.0.0.1"]
        await client.close()
