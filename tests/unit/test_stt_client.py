"""``MoshiSttClient`` over the scripted connection double — T-0007a group A, criteria 1–8.

Pure: no socket, no service, no GPU (``tests/fakes/moshi_stt_double.py`` face (a)). The same
script runs over a real loopback socket in ``tests/integration/test_stt_ws_loopback.py``.

Seams this module fixes
=======================
``docs/tasks/T-0007a.md`` names the client and its behaviour but not every signature. These are
the names the tests call; the design behind them is the implementer's. Report anything wrong here
to the reviewer rather than editing the test (CLAUDE.md: the implementer never edits ``tests/``).

``local_astra/conductor/stt_client.py``

    MoshiSttClient(services=None, *, connect=None, auth_id="public_token",
                   max_reconnects=<int ≥ 1>, max_queued_blocks=<int ≥ 25>)
        ``services`` is a ``local_astra.config.Services`` (``None`` → ``Services()``: the
        documented loopback default). ``connect`` is the transport seam: an awaitable
        ``connect(url) -> conn`` where ``conn`` has ``async send(bytes)`` / ``async recv() ->
        bytes`` / ``async close()`` (criterion 1; typed as a local ``Protocol``, per the task's
        Implementation notes). ``auth_id`` defaults to the one token in
        ``services/moshi/stt-1b-en_fr.toml``.

    client.url -> str        ``services.stt`` plus ``?auth_id=<auth_id>``; what ``connect`` is given.
    await client.feed(pcm_24k_f32: bytes, t_ms: int) -> None      (``SttClient``)
    client.events() -> AsyncIterator[SttEvent]                    (``SttClient``; one stream)
    await client.close() -> None                                  (``SttClient``)
    await client.mark(t_eot_ms: int) -> int
        Sends ``{"type": "Marker", "id": <int>}``, returns the id, remembers ``id → t_eot_ms``.
    client.stats() -> Mapping[str, int]
        At least ``STT_STATS_KEYS``: ``estimated_word_times``, ``stale_markers``,
        ``unknown_messages``, ``dropped_blocks``, ``reconnects``, ``queued_blocks``.
    client.max_queued_blocks -> int      the documented outbound cap (≥ 2 s of 80 ms blocks).

Two time rules the criteria imply and the tests pin exactly (hand-computable, so an
implementation cannot be "nearly right"):

* **the fed-audio clock.** After ``feed(pcm, t_ms)`` the client's audio clock is
  ``t_ms + len(pcm) // 4 * 1000 // 24000`` — the *end* of the audio it has handed over (one
  1920-sample block fed at 1000 ms puts it at 1080 ms). It is the ``t_ms`` of a ``Step`` and the
  fallback ``start_ms`` of a ``Word`` the server did not timestamp (criterion 3). It never goes
  on the wire (criterion 2).
* **the connection base.** moshi's own clock restarts at 0 on every connection, so a server
  timestamp is read as ``base_ms + round(start_time * 1000)`` where ``base_ms`` is the ``t_ms``
  of the first block fed *on that connection* (criteria 3 and 7). After a reconnect the base
  moves; before any feed there is no base.

Reconnect is **eager**: a server-side close is not left for the next ``feed`` to discover
(criterion 7; a lazily reconnecting client would keep a dead socket across a silence and lose
the words that follow).

Assumption flagged for criterion 12's census
============================================
The server's word-timestamp field is spelled ``start_time`` (seconds) here — the upstream
delayed-streams-modeling spelling. Nobody has observed one on this box: ``scripts/demo/_listen.py``
reads ``m["text"]`` only. If the census finds another name, *this fixture* changes (and the client
with it); the criterion-3 behaviour — server time when there is one, fed clock when there is not,
``estimated_word_times`` counted — does not.
"""

from __future__ import annotations

import asyncio
import os
import struct
import time
from pathlib import Path

import pytest
import websockets.exceptions
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from local_astra.config import Services
from tests.fakes.moshi_stt_double import (
    STT_STATS_KEYS,
    EventDrain,
    SttConnectionDouble,
    SttConnector,
    build_stt_client,
    end_word,
    live_tasks,
    marker,
    settle,
    step,
    types_of,
    unknown,
    wait_until,
    word,
)

#: 80 ms at 24 kHz (T-0021's framing; the client is a consumer of it, not its author).
BLOCK_SAMPLES = 1920


def block(n: int = BLOCK_SAMPLES, value: float = 0.25) -> bytes:
    """``n`` f32le samples. 0.25 is exact in binary32, so the wire value is unambiguous."""
    return struct.pack(f"<{n}f", *([value] * n))


def pattern(n: int) -> tuple[bytes, list[float]]:
    """A block and the float32 values it decodes to, computed with ``struct``, not the client."""
    seed = [0.0, 1.0, -1.0, 0.5, -0.25, 0.1, -0.7, 1e-7]
    values = [seed[i % len(seed)] * (1.0 if i % 2 else -1.0) for i in range(n)]
    raw = struct.pack(f"<{n}f", *values)
    oracle = list(struct.unpack(f"<{n}f", raw))  # binary32 rounding, done independently
    return raw, oracle


def open_fds() -> int:
    return len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else -1


async def connected(**knobs) -> tuple[object, SttConnectionDouble, SttConnector, EventDrain]:
    """A client with one scripted connection, already connected by a first 1920-sample block."""
    double = SttConnectionDouble(**knobs)
    connector = SttConnector([double])
    client = build_stt_client(None, connect=connector, max_reconnects=0)
    drain = EventDrain(client)
    await client.feed(block(), 0)
    await wait_until(lambda: double.audio_count >= 1, what="the first block reaching the double")
    return client, double, connector, drain


# --- criterion 1: construction ------------------------------------------------------------------


async def test_construction_opens_nothing_and_refuses_a_non_loopback_endpoint():
    """Criterion 1: endpoint from ``Services``, loopback only, and the socket opens on ``feed``."""
    before = open_fds()
    connector = SttConnector([SttConnectionDouble()])
    client = build_stt_client(None, connect=connector)
    assert open_fds() == before, "constructing the client opened a file descriptor"
    assert connector.attempts == 0, "constructing the client connected"

    assert client.url.startswith("ws://127.0.0.1:8080/api/asr-streaming")
    assert "auth_id=public_token" in client.url  # services/moshi/stt-1b-en_fr.toml
    assert build_stt_client(None, connect=connector, auth_id="other").url.endswith("auth_id=other")

    with pytest.raises(ValidationError):  # the config validator, docs/02 §9
        Services(stt="ws://10.0.0.5:8080/api/asr-streaming")
    with pytest.raises(ValueError):  # ... and the client does not trust a bypassed validator
        build_stt_client(
            Services.model_construct(stt="ws://stt.example.com:8080/api/asr-streaming"),
            connect=connector,
        )

    await client.feed(block(), 0)  # lazy: the connection opens here, once
    await wait_until(lambda: connector.attempts == 1, what="the lazy connect")
    assert connector.urls == [client.url]
    await client.close()


# --- criterion 2: feed is a pure consumer -------------------------------------------------------


async def test_one_fed_block_becomes_one_audio_message_with_identical_samples():
    """Criterion 2: one block in, one ``Audio`` message out; samples bit-exact; no timestamps."""
    double = SttConnectionDouble()
    client = build_stt_client(None, connect=SttConnector([double]))
    big, big_oracle = pattern(1920)
    small, small_oracle = pattern(480)
    await client.feed(big, 0)
    await client.feed(small, 80)
    await wait_until(lambda: len(double.audio_messages()) == 2, what="two Audio messages")

    sent = double.audio_messages()
    assert [len(m["pcm"]) for m in sent] == [1920, 480], "blocks were merged, split or padded"
    assert [set(m) for m in sent] == [{"type", "pcm"}] * 2, "the wire carries no timestamps"
    assert list(sent[0]["pcm"]) == big_oracle
    assert list(sent[1]["pcm"]) == small_oracle

    # nothing survives a feed: the next block is on the wire alone, unchanged
    third, third_oracle = pattern(240)
    await client.feed(third, 160)
    await wait_until(lambda: len(double.audio_messages()) == 3, what="a third Audio message")
    assert list(double.audio_messages()[2]["pcm"]) == third_oracle
    await client.close()


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=1, max_value=200), min_size=1, max_size=6))
def test_every_sequence_of_blocks_arrives_as_the_same_sequence_of_messages(sizes):
    """Criterion 2 as a property: never merged, split, re-ordered or re-timestamped."""

    async def run() -> None:
        double = SttConnectionDouble()
        client = build_stt_client(None, connect=SttConnector([double]))
        oracles = []
        for i, n in enumerate(sizes):
            raw, oracle = pattern(n)
            oracles.append(oracle)
            await client.feed(raw, i * 80)
        await wait_until(lambda: len(double.audio_messages()) == len(sizes), what="every block")
        assert [list(m["pcm"]) for m in double.audio_messages()] == oracles
        await client.close()

    asyncio.run(run())


async def test_a_byte_length_that_is_not_a_multiple_of_four_raises_before_the_wire():
    """Criterion 2: a caller bug, not a service failure — and nothing reaches the socket."""
    double = SttConnectionDouble()
    connector = SttConnector([double])
    client = build_stt_client(None, connect=connector)
    with pytest.raises(ValueError):
        await client.feed(b"\x00" * 6, 0)
    await settle(rounds=3)
    assert connector.attempts == 0 and double.raw_sent == []
    await client.close()


# --- criterion 3: inbound mapping ---------------------------------------------------------------


async def test_word_step_marker_and_unknown_map_to_the_stt_event_union_in_arrival_order():
    """Criterion 3: total mapping, arrival order kept, an unknown type does not end the stream.

    Rewritten by T-0038 and pending until it lands: the group B census found an ``EndWord`` after
    every ``Word``, and criterion 1 publishes a word **when its end arrives**, so the script here
    gains the ends and the expected ``end_ms`` stop being copies of the starts. The property this
    row exists for — one event per message, in arrival order, an unknown never ending the stream
    — is unchanged.
    """
    client, double, _, drain = await connected()
    double.push(word("hello", 0.5))
    double.push(end_word(0.9))
    double.push(step(prs=(0.1, 0.2, 0.3, 0.4), step_idx=1))
    double.push(unknown("Frobnicate", payload=1))
    double.push(word("world", 1.25))
    double.push(end_word(1.5))

    events = await drain.take(4)
    assert types_of(events) == ["word", "step", "error", "word"]
    assert (events[0].text, events[0].start_ms, events[0].end_ms) == ("hello", 500, 900)
    assert (events[3].start_ms, events[3].end_ms) == (1250, 1500), "the word's own server times"
    assert tuple(events[1].pause_heads) == (0.1, 0.2, 0.3, 0.4)
    assert events[1].t_ms == 80, "the fed-audio clock: one 1920-sample block fed at t_ms=0"
    assert "Frobnicate" in events[2].message
    assert len(events[2].message) <= 200 and "Traceback" not in events[2].message
    assert client.stats()["unknown_messages"] == 1
    assert set(STT_STATS_KEYS) <= set(client.stats())
    await client.close()


async def test_a_word_with_no_server_timestamp_uses_the_fed_clock_and_invents_no_duration():
    """Criterion 3: the fallback is the fed clock, ``end_ms == start_ms``, and it is counted."""
    double = SttConnectionDouble()
    client = build_stt_client(None, connect=SttConnector([double]), max_reconnects=0)
    drain = EventDrain(client)
    await client.feed(block(), 1000)  # audio clock -> 1080; connection base -> 1000
    await wait_until(lambda: double.audio_count == 1, what="the first block")
    double.push(word("untimed"))
    double.push(word("timed", 0.05))

    # Collected across close(): T-0038 criterion 1 may hold a word back until its end arrives or
    # the stream closes, and neither word here has an EndWord. The *times* are the same either
    # way, which is what this row is about.
    events = await drain.quiet(window_s=0.25)
    await client.close()
    events += await drain.quiet(window_s=0.25)
    assert [e.text for e in events] == ["untimed", "timed"], types_of(events)
    assert (events[0].start_ms, events[0].end_ms) == (1080, 1080), "no duration is invented"
    assert (events[1].start_ms, events[1].end_ms) == (1050, 1050), "base + server seconds"
    assert client.stats()["estimated_word_times"] == 1


@pytest.mark.parametrize("message", [unknown("Error", message="cuda oom"), unknown("Frobnicate")])
async def test_an_unmapped_server_message_becomes_one_named_error_and_is_counted(message):
    """Criterion 3 / ruling A8: the mapping is total — an unguessed type is never swallowed.

    Swallowing the types we *guess* are harmless is how a server-sent ``Error`` (a model that
    failed to load, an out-of-memory) turns into "it transcribes nothing" with an empty log.
    ``Ready`` used to be one of the cases here; the group B census measured it and T-0038
    criterion 3 maps it, so it now lives in ``test_stt_client_endword.py`` — which asserts the
    opposite (counted, never an error event) for that one type only.
    """
    client, double, _, drain = await connected()
    double.push(message)
    events = await drain.take(1)
    assert types_of(events) == ["error"]
    assert str(message["type"]) in events[0].message
    assert len(events[0].message) <= 200 and "Traceback" not in events[0].message
    assert client.stats()["unknown_messages"] == 1

    # ... and the stream survives it: the next word still arrives (at close at the latest, since
    # T-0038 criterion 1 may hold a word that has no EndWord).
    double.push(word("after", 0.6))
    tail = await drain.quiet(window_s=0.25)
    await client.close()
    tail += await drain.quiet(window_s=0.25)
    assert [e.text for e in tail] == ["after"], f"an unknown type ended the stream: {tail}"
    assert client.stats()["unknown_messages"] == 1


# --- criterion 4: the marker ---------------------------------------------------------------------


async def test_a_marker_echo_yields_flushed_after_every_word_that_preceded_it():
    """Criterion 4 (docs/02 §4 row 6): the flush lands *after* the words it confirms."""
    client, double, _, drain = await connected()
    double.push(word("all", 0.1))
    double.push(word("five", 0.2))
    marker_id = await client.mark(3000)
    assert isinstance(marker_id, int)
    second_id = await client.mark(4000)
    assert second_id != marker_id, "marker ids are unique for the life of the connection"

    events = await drain.take(4)
    assert types_of(events) == ["word", "word", "flushed", "flushed"]
    assert [e.t_eot_ms for e in events[2:]] == [3000, 4000]

    on_the_wire = double.marker_messages()
    assert [set(m) for m in on_the_wire] == [{"type", "id"}] * 2
    assert [m["id"] for m in on_the_wire] == [marker_id, second_id]
    await client.close()


async def test_a_server_that_never_echoes_a_marker_never_produces_a_flushed_event():
    """Criterion 4, adversarial — the regression this task most needs to be impossible.

    ``scripts/demo/_listen.py``'s ``DRAIN_S = 2.5`` s tail heuristic exists because that spike had
    no marker. A client that re-invents it here (a drain timer, "recv went quiet", "N steps with no
    word", "we closed, so the turn must have flushed") silently defeats T-0007 criterion 16 while
    every other test stays green. The window below is deliberately longer than that 2.5 s.
    """
    client, double, _, drain = await connected(echo_markers=False)
    await client.mark(2000)

    async def keep_the_socket_busy() -> None:
        for i in range(80):  # a Step every 80 ms, exactly like the real server, for ~3.2 s
            double.push(step(prs=(0.9, 0.95, 0.99, 0.99), step_idx=i))
            await asyncio.sleep(0.04)

    pusher = asyncio.create_task(keep_the_socket_busy())
    seen = await drain.quiet(window_s=2.8)  # > DRAIN_S = 2.5 s
    pusher.cancel()

    assert "step" in types_of(seen), "the double did not deliver anything — the test is vacuous"
    assert "flushed" not in types_of(seen), "a flushed event was synthesised without a Marker echo"
    await client.close()
    tail = await drain.quiet(window_s=0.2)
    assert "flushed" not in types_of(tail), "close() synthesised a flushed event"
    assert double.marker_messages(), "the client never sent the Marker it was asked for"


async def test_a_stale_marker_id_is_ignored_and_counted():
    """Criterion 4: an echo for an unknown or already-consumed id is dropped, not guessed."""
    client, double, _, drain = await connected()
    double.push(marker(99_999))  # never requested (a leftover from an earlier flush)
    marker_id = await client.mark(1500)
    double.push(marker(marker_id))  # the same echo again, after it was consumed

    events = await drain.take(1)
    assert types_of(events) == ["flushed"] and events[0].t_eot_ms == 1500
    quiet = await drain.quiet(window_s=0.2)
    assert types_of(quiet) == [], f"the stale echoes produced {types_of(quiet)}"
    assert client.stats()["stale_markers"] == 2
    await client.close()


# --- criterion 5: failures are events -------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    ["connect_refused", "handshake_rejected", "server_closed", "garbage_frame", "send_timeout"],
)
async def test_each_of_five_failures_becomes_one_stt_error_event_and_the_stream_survives(failure):
    """Criterion 5: ``feed`` never raises a service failure; ``events()`` carries it, once."""
    double = SttConnectionDouble()
    script: list = [double]
    if failure == "connect_refused":
        script = [ConnectionRefusedError("connection refused")]
    elif failure == "handshake_rejected":
        script = [websockets.exceptions.InvalidHandshake("401 unauthorized")]
    elif failure == "server_closed":
        double = SttConnectionDouble(close_after_audio=1)
        script = [double]
    elif failure == "garbage_frame":
        double = SttConnectionDouble(garbage_after_audio=1)
        script = [double]
    elif failure == "send_timeout":
        double = SttConnectionDouble(send_error=TimeoutError("send timed out"))
        script = [double]

    client = build_stt_client(None, connect=SttConnector(script), max_reconnects=0)
    drain = EventDrain(client)
    await client.feed(block(), 0)  # returns normally whatever the socket does

    events = await drain.take(1)
    assert types_of(events) == ["error"]
    message = events[0].message
    assert 0 < len(message) <= 200, f"error text is {len(message)} chars: {message!r}"
    assert "Traceback" not in message and "\n" not in message

    if failure == "garbage_frame":  # an undecodable frame does not end the stream
        # A Step, not a Word: a Step is published the moment it arrives whatever T-0038's
        # delivery rule does with an unfinished word, so this row keeps testing the frame.
        double.push(step(prs=(0.1, 0.2, 0.3, 0.4), step_idx=3))
        assert types_of(await drain.take(1)) == ["step"]

    quiet = await drain.quiet(window_s=0.2)
    assert types_of(quiet) == [], f"a second event for one failure: {types_of(quiet)}"
    assert not drain.ended, "only close() may end events()"
    await client.close()


# --- criterion 6: close ---------------------------------------------------------------------------


async def test_close_returns_within_200ms_against_a_stuck_handshake_and_is_idempotent():
    """Criterion 6: the measured 10 s hang of ``_listen.py::deliver`` never happens again."""
    baseline = set(asyncio.all_tasks())
    client, double, _, drain = await connected(hang_close=True)

    started = time.perf_counter()
    try:
        await asyncio.wait_for(client.close(), 1.0)
    except TimeoutError as exc:
        raise AssertionError("close() did not return within 1 s against a stuck handshake") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 200, f"close() took {elapsed_ms:.0f} ms"

    await drain.expect_end()
    started = time.perf_counter()
    await asyncio.wait_for(client.close(), 1.0)  # idempotent, and still fast
    assert (time.perf_counter() - started) * 1000 < 200
    assert double.close_calls >= 1

    await settle()
    assert live_tasks(baseline) == set(), "the connection task outlived close()"


# --- criterion 7: reconnect -----------------------------------------------------------------------


async def test_reconnect_rebases_word_times_and_announces_the_discontinuity():
    """Criterion 7: a new moshi connection restarts the model clock at 0; the words do not."""
    first = SttConnectionDouble(close_after_audio=2, name="first")
    connector = SttConnector([first], factory=lambda i: SttConnectionDouble(name=f"reconnect-{i}"))
    client = build_stt_client(None, connect=connector, max_reconnects=2)
    drain = EventDrain(client)
    await client.feed(block(), 0)
    await client.feed(block(), 80)  # the server drops the connection here

    await wait_until(lambda: connector.attempts == 2, timeout_s=3.0, what="the eager reconnect")
    second = connector.handed_out[1]
    assert second.audio_messages() == [], "audio from before the disconnect was replayed"

    await client.feed(block(), 5000)  # the first block of the new connection: the new base
    await wait_until(lambda: len(second.audio_messages()) == 1, what="the post-reconnect block")
    second.push(word("later", 0.1))

    seen = await drain.quiet(window_s=0.3)
    await client.close()
    # T-0038: a word that has no EndWord lands at the latest here
    seen += await drain.quiet(window_s=0.3)
    kinds = types_of(seen)
    assert kinds.count("error") == 1, f"the discontinuity is announced exactly once: {kinds}"
    words = [e for e in seen if e.type == "word"]
    assert [w.text for w in words] == ["later"]
    assert words[0].start_ms == 5100, "the new connection's clock was not re-based"
    assert client.stats()["reconnects"] == 1


async def test_reconnects_are_bounded_and_the_client_does_not_spin():
    """Criterion 7 negative: no unbounded reconnect loop (``batch_size = 2`` on the whole box)."""
    connector = SttConnector(
        factory=lambda i: SttConnectionDouble(close_after_audio=1, name=f"c{i}")
    )
    client = build_stt_client(None, connect=connector, max_reconnects=2)
    drain = EventDrain(client)
    await client.feed(block(), 0)
    await drain.quiet(window_s=0.6)
    assert connector.attempts <= 3, f"{connector.attempts} connections for max_reconnects=2"
    await client.close()


# --- criterion 8: backpressure ---------------------------------------------------------------------


async def test_feed_returns_within_20ms_and_the_buffer_is_bounded_when_the_peer_stops_reading():
    """Criterion 8: the reflex lane is never blocked, and 60 s of audio cannot grow the buffer."""
    baseline = set(asyncio.all_tasks())
    double = SttConnectionDouble(stop_reading=True)
    connector = SttConnector([double])
    client = build_stt_client(None, connect=connector)
    drain = EventDrain(client)

    assert client.max_queued_blocks >= 25, "the cap must hold ≥ 2 s of 80 ms blocks"
    payload = block()
    worst_ms = 0.0
    depths = []
    for i in range(750):  # 60 s of audio at 80 ms a block
        started = time.perf_counter()
        await client.feed(payload, i * 80)
        worst_ms = max(worst_ms, (time.perf_counter() - started) * 1000)
        if i % 50 == 0:
            depths.append(client.stats()["queued_blocks"])
    depths.append(client.stats()["queued_blocks"])

    assert worst_ms < 20, f"the slowest feed took {worst_ms:.1f} ms (reflex-lane budget is 20 ms)"
    assert max(depths) <= client.max_queued_blocks, f"the outbound buffer grew to {max(depths)}"
    assert client.stats()["dropped_blocks"] > 0
    assert len(live_tasks(baseline)) <= 4, "a task per block"

    seen = await drain.quiet(window_s=0.2)
    assert types_of(seen).count("error") == 1, f"one overflow report, not one per block: {seen}"
    assert connector.attempts == 1, "one connection per client, reused (batch_size = 2)"
    await client.close()


async def test_a_marker_is_never_dropped_when_the_outbound_buffer_is_full():
    """Ruling A7: when the buffer is full the *oldest queued block* goes, never the ``Marker``.

    Criterion 4 beats one stale block: a dropped marker is an ``ENDPOINTING`` row that never
    fires (T-0003 criterion 6) and a turn that hangs, while a dropped block is 80 ms of audio the
    delayed-stream model will have moved past anyway. The marker still queues *behind* the audio
    it confirms — overtaking it would flush words that have not been sent yet (docs/02 §4 row 6).
    The eviction is a deliberate choice, not an overflow, so it is counted and **not** reported:
    ``feed``'s own overflow is what raises the one ``SttErrorEvent`` (criterion 8).
    """
    double = SttConnectionDouble(stop_reading=True)
    client = build_stt_client(None, connect=SttConnector([double]), max_reconnects=0)
    drain = EventDrain(client)

    # Block 0 is taken by the sender and parked inside send(); the queue then fills exactly.
    await client.feed(block(value=0.0), 0)
    await wait_until(lambda: double.send_calls == 1, what="the first send to park")
    cap = client.max_queued_blocks
    for i in range(1, cap + 1):
        await client.feed(block(value=float(i)), i * 80)
    assert client.stats()["queued_blocks"] == cap, "the buffer is not full; the test is vacuous"
    assert client.stats()["dropped_blocks"] == 0

    marker_id = await client.mark(cap * 80)
    double.release()
    await wait_until(lambda: bool(double.marker_messages()), what="the Marker reaching the wire")

    kinds = [m["type"] for m in double.decoded()]
    assert kinds[-1] == "Marker" and set(kinds[:-1]) == {"Audio"}, f"wire order: {kinds}"
    assert double.marker_messages() == [{"type": "Marker", "id": marker_id}]
    sent = [m["pcm"][0] for m in double.audio_messages()]
    assert sent == [0.0, *[float(i) for i in range(2, cap + 1)]], (
        "the block evicted to make room for the marker must be the oldest queued one"
    )
    assert client.stats()["dropped_blocks"] == 1
    tail = types_of(await drain.quiet(window_s=0.2))
    assert tail == ["flushed"], (
        f"the marker must survive the full buffer end to end — sent, echoed, flushed — and the "
        f"eviction must not be reported as an error: {tail}"
    )
    await client.close()
