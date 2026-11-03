"""``EndWord`` and ``Ready`` on the STT wire — T-0038 criteria 1–4.

Pure (``tests/fakes/moshi_stt_double.py`` face (a)): no socket, no service, no GPU.

Where the numbers come from
===========================
The T-0007a **group B census**, 2026-09-14, commit f114adf, live 0.6.4 speech server on CPU
(``--cpu``, ``batch_size 2``, ``asr_delay_in_tokens 6``), fixture
``tests/fixtures/audio/pocket_tts_sentence.wav`` (4.040 s, "One, two, three, four, five.") fed at
realtime pace plus 3.2 s of harness silence, n=1 — ``logs/t0007a-group-b-census-f114adf.log``.
That run recorded, on the wire:

* ``Word`` keys ``{'type', 'text', 'start_time'}``, ``start_time`` in **seconds**, the five words
  ``'one,' 'two,' 'three,' 'four,' 'five.'`` at ``0.72 1.44 2.24 3.04 3.76``;
* ``EndWord`` keys ``{'type', 'stop_time'}``, seconds, the four that arrived before the client
  stopped reading: ``1.44 2.24 3.04 3.76``;
* ``Ready`` keys ``{'type'}``, once;
* the client's ``stats()`` for that run: ``unknown_messages: 5`` (the ``Ready`` and the four
  ``EndWord`` messages), and its word events printed as ``720 ms .. 720 ms 'one,'`` … — i.e.
  every ``end_ms`` equal to its ``start_ms``, which is the bug T-0038 fixes.

The expected ``start_ms``/``end_ms`` below are those seconds multiplied by 1000 **by hand**
(``0.72 s -> 720 ms``) with the connection base at 0 (the first block is fed at ``t_ms = 0``),
never read back from the client. The fifth word's end is the only number not in the log: the
census's fifth ``EndWord`` was still inside the server's ~0.5 s lag when the client closed, so the
tests that need it use the fixture's own duration, **4.040 s -> 4040 ms**, and say so.

Delivery rule under test (criterion 1): a word event is emitted **when its ``EndWord`` arrives**;
a ``Word`` that is followed by another ``Word`` with no ``EndWord`` between them is emitted with
the second word's ``start_time`` as its end and counted in ``stats()["words_without_end"]``; a
trailing open word is emitted at the ``Marker`` echo or at ``close()`` with ``end_ms = start_ms``,
counted the same way, and **never dropped**.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fakes.moshi_stt_double import (
    EventDrain,
    SttConnectionDouble,
    SttConnector,
    build_stt_client,
    end_word,
    marker,
    ready,
    step,
    types_of,
    unknown,
    wait_until,
    word,
)

BLOCK_SAMPLES = 1920  # 80 ms at 24 kHz (T-0021's framing)

#: The census transcript, verbatim, with ``start_time`` in seconds and the ``stop_time`` that
#: closed each word. The fifth end is the fixture's length (4.040 s), not a censused value.
CENSUS: list[tuple[str, float, float]] = [
    ("one,", 0.72, 1.44),
    ("two,", 1.44, 2.24),
    ("three,", 2.24, 3.04),
    ("four,", 3.04, 3.76),
    ("five.", 3.76, 4.04),
]

#: The same five words in milliseconds, written out by hand (base 0). Kept as literals so a
#: reader can check them against the log without running anything.
CENSUS_MS: list[tuple[str, int, int]] = [
    ("one,", 720, 1440),
    ("two,", 1440, 2240),
    ("three,", 2240, 3040),
    ("four,", 3040, 3760),
    ("five.", 3760, 4040),
]


def block(n: int = BLOCK_SAMPLES, value: float = 0.25) -> bytes:
    """``n`` f32le samples; 0.25 is exact in binary32."""
    return struct.pack(f"<{n}f", *([value] * n))


async def connected(t0_ms: int = 0, **knobs: Any) -> tuple[Any, SttConnectionDouble, EventDrain]:
    """A client with one scripted connection, connected by a first block fed at ``t0_ms``."""
    double = SttConnectionDouble(**knobs)
    client = build_stt_client(None, connect=SttConnector([double]), max_reconnects=0)
    drain = EventDrain(client)
    await client.feed(block(), t0_ms)
    await wait_until(lambda: double.audio_count >= 1, what="the first block reaching the double")
    return client, double, drain


# --- criterion 1: EndWord closes the open word --------------------------------------------------


async def test_the_census_sentence_replays_as_five_words_with_their_real_ends():
    """Criterion 1: the five ``Word``/``EndWord`` pairs of the census, times in ms by hand."""
    client, double, drain = await connected()
    for text, start_s, stop_s in CENSUS:
        double.push(word(text, start_s))
        double.push(end_word(stop_s))

    events = await drain.take(5)
    assert types_of(events) == ["word"] * 5, (
        f"an EndWord must not reach the event stream as anything else: {types_of(events)}"
    )
    assert [(e.text, e.start_ms, e.end_ms) for e in events] == CENSUS_MS
    assert all(e.end_ms > e.start_ms for e in events), (
        "every censused word has a real duration; end_ms == start_ms is the bug (T-0038)"
    )
    stats = client.stats()
    assert stats["unknown_messages"] == 0, "the census's five error events per sentence are gone"
    assert stats.get("words_without_end") == 0
    assert stats.get("stray_end_words") == 0
    quiet = await drain.quiet(window_s=0.15)
    assert quiet == [], f"the pairs produced extra events: {types_of(quiet)}"
    await client.close()


async def test_a_word_is_not_emitted_until_its_end_arrives():
    """Criterion 1's delivery rule: a word with no end is not yet a word to the session."""
    client, double, drain = await connected()
    double.push(word("one,", 0.72))

    early = await drain.quiet(window_s=0.2)
    assert types_of(early) == [], f"the word was published before its end arrived: {early}"

    double.push(end_word(1.44))
    events = await drain.take(1)
    assert [(e.text, e.start_ms, e.end_ms) for e in events] == [("one,", 720, 1440)]
    await client.close()


async def test_a_word_with_no_end_is_closed_at_the_next_words_start_and_counted():
    """Criterion 1: ``Word``, ``Word``, ``EndWord`` — the first ends where the second begins."""
    client, double, drain = await connected()
    double.push(word("three,", 2.24))
    double.push(word("four,", 3.04))  # no EndWord came between these two
    double.push(end_word(3.76))

    events = await drain.take(2)
    assert [(e.text, e.start_ms, e.end_ms) for e in events] == [
        ("three,", 2240, 3040),
        ("four,", 3040, 3760),
    ]
    stats = client.stats()
    assert stats.get("words_without_end") == 1, (
        "a word closed by the next word's start is an estimate and must be counted"
    )
    assert stats["unknown_messages"] == 0
    await client.close()


async def test_a_trailing_open_word_is_delivered_at_close_never_dropped():
    """Criterion 1: ``close()`` flushes the open word with ``end_ms = start_ms`` and counts it."""
    client, double, drain = await connected()
    double.push(word("five.", 3.76))
    await asyncio.sleep(0.05)  # the message has been delivered; it is held, not published

    await client.close()
    tail = await drain.quiet(window_s=0.3)
    assert [(e.text, e.start_ms, e.end_ms) for e in tail] == [("five.", 3760, 3760)], (
        f"the last word of a turn must not be dropped (docs/02 §4 row 6): {types_of(tail)}"
    )
    assert client.stats().get("words_without_end") == 1


# --- criterion 2: a stray EndWord ---------------------------------------------------------------


async def test_an_endword_with_no_open_word_is_counted_and_emits_nothing():
    """Criterion 2: the server is ahead of us, not broken — count it, do not shout."""
    client, double, drain = await connected()
    double.push(end_word(1.44))

    quiet = await drain.quiet(window_s=0.2)
    assert types_of(quiet) == [], f"a stray EndWord produced {types_of(quiet)}"
    stats = client.stats()
    assert stats.get("stray_end_words") == 1
    assert stats["unknown_messages"] == 0, "a stray EndWord is not an unknown message type"

    double.push(word("two,", 1.44))  # and the stream still works afterwards
    double.push(end_word(2.24))
    events = await drain.take(1)
    assert [(e.text, e.start_ms, e.end_ms) for e in events] == [("two,", 1440, 2240)]
    await client.close()


# --- criterion 3: Ready -------------------------------------------------------------------------


async def test_ready_is_consumed_silently_and_counted_every_time():
    """Criterion 3: the census's one ``Ready`` per connection stops being an error event.

    Reading taken from the criterion's wording ("a second ``Ready`` is *also* just counted"):
    ``ready_seen`` is a counter, so two of them read 2. Flagged to the reviewer as the one place
    the criterion could also be read as a flag.
    """
    client, double, drain = await connected()
    double.push(ready())
    await wait_until(
        lambda: client.stats().get("ready_seen") == 1, what="the first Ready being counted"
    )

    double.push(ready())  # the server restarted its stream; still just a count
    await wait_until(
        lambda: client.stats().get("ready_seen") == 2, what="the second Ready being counted"
    )
    assert client.stats()["unknown_messages"] == 0

    double.push(word("one,", 0.72))
    double.push(end_word(1.44))
    events = await drain.take(1)
    assert types_of(events) == ["word"], f"Ready leaked onto the event stream: {types_of(events)}"
    assert (events[0].start_ms, events[0].end_ms) == (720, 1440)
    await client.close()


# --- criterion 4: everything else stays loud ----------------------------------------------------


async def test_a_word_without_text_is_still_an_unknown_and_opens_no_word():
    """Criterion 4 negative: ruling A8 holds; a malformed ``Word`` is loud and leaves no state."""
    client, double, drain = await connected()
    double.push({"type": "Word", "start_time": 0.72})  # no text: not a word at all

    events = await drain.take(1)
    assert types_of(events) == ["error"]
    assert len(events[0].message) <= 200 and "Traceback" not in events[0].message
    assert client.stats()["unknown_messages"] == 1

    double.push(end_word(1.44))  # nothing was opened, so this end has nothing to close
    quiet = await drain.quiet(window_s=0.2)
    assert types_of(quiet) == [], f"the rejected Word left an open word behind: {types_of(quiet)}"
    assert client.stats().get("stray_end_words") == 1
    assert client.stats()["unknown_messages"] == 1
    await client.close()


async def test_an_endword_without_a_numeric_stop_time_is_an_unknown_not_a_crash():
    """Criterion 4 negative: a malformed ``EndWord`` is loud, and the client keeps working."""
    client, double, drain = await connected()
    double.push({"type": "EndWord", "stop_time": "soon"})

    events = await drain.take(1)
    assert types_of(events) == ["error"]
    assert "EndWord" in events[0].message
    assert client.stats()["unknown_messages"] == 1

    double.push(word("two,", 1.44))
    double.push(end_word(2.24))
    good = [e for e in await drain.quiet(window_s=0.25) if e.type == "word" and e.text == "two,"]
    assert [(e.start_ms, e.end_ms) for e in good] == [(1440, 2240)], (
        "a malformed EndWord must not poison the words that follow it"
    )
    assert client.stats()["unknown_messages"] == 1, (
        "a well-formed EndWord is mapped, not counted as unknown"
    )
    await client.close()


@pytest.mark.parametrize("message", [unknown("Frobnicate", payload=1), unknown("Error", m="oom")])
async def test_any_other_type_still_becomes_one_named_error_event(message):
    """Criterion 4: mapping ``Ready``/``EndWord`` does not make the mapping permissive."""
    client, double, drain = await connected()
    double.push(message)
    double.push(word("one,", 0.72))
    double.push(end_word(1.44))

    events = await drain.take(2)
    assert types_of(events) == ["error", "word"]
    assert str(message["type"]) in events[0].message
    assert client.stats()["unknown_messages"] == 1
    assert client.stats().get("stray_end_words") == 0
    await client.close()


# --- adversarial: the end that precedes the start -----------------------------------------------


async def test_an_endword_before_the_words_start_clamps_and_is_counted():
    """Adversarial (T-0038's named failure mode): ``end_ms`` may never precede ``start_ms``.

    ``SttWordEvent`` would accept the inverted pair, T-0032 would take ``t_eot`` from it, and the
    turn would endpoint *before* the last word began. The only sound answer is the word's own
    start, marked as an estimate (``words_without_end``), never a negative duration.
    """
    client, double, drain = await connected()
    double.push(word("late", 2.0))
    early = await drain.quiet(window_s=0.15)
    assert types_of(early) == [], f"the word was published before its end arrived: {early}"

    double.push(end_word(1.5))  # 500 ms *before* the word started
    events = await drain.take(1)
    assert (events[0].text, events[0].start_ms, events[0].end_ms) == ("late", 2000, 2000)
    assert events[0].end_ms >= events[0].start_ms
    stats = client.stats()
    assert stats.get("words_without_end") == 1, "a clamped end is an estimate and is counted"
    assert stats["unknown_messages"] == 0
    await client.close()


async def test_a_step_between_a_word_and_its_end_does_not_disturb_the_pairing():
    """Adversarial: the real wire interleaves a ``Step`` every 80 ms between the two messages."""
    client, double, drain = await connected()
    double.push(word("one,", 0.72))
    double.push(step(prs=(0.1, 0.2, 0.3, 0.4), step_idx=9))
    double.push(end_word(1.44))

    events = await drain.take(2)
    assert types_of(events) == ["step", "word"], (
        f"the Step must not wait behind the open word: {types_of(events)}"
    )
    assert (events[1].start_ms, events[1].end_ms) == (720, 1440)
    assert tuple(events[0].pause_heads) == (0.1, 0.2, 0.3, 0.4)
    await client.close()


# --- the property: no Word is ever lost, and no pair ever inverts -------------------------------


@settings(
    deadline=None,
    max_examples=25,
    report_multiple_bugs=False,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(st.sampled_from("WE"), min_size=1, max_size=8).filter(lambda ops: "E" in ops))
def test_every_word_is_delivered_exactly_once_in_order_with_a_sane_duration(ops):
    """Criteria 1–2 as a property over arbitrary ``Word``/``EndWord`` interleavings.

    The oracle is independent of the client: whatever the order, by the time ``close()`` has
    returned the stream must carry **one word event per ``Word`` message, in the order the words
    arrived**, each with ``0 <= start_ms <= end_ms`` and the ``start_ms`` the server gave it —
    and no error event, because neither type is unknown any more (criteria 2 and 4). The k-th
    message on the wire is timestamped ``k * 100 ms``, so starts are strictly increasing and can
    be checked without knowing which rule closed which word.
    """

    async def run() -> None:
        client, double, drain = await connected()
        expected: list[tuple[str, int]] = []
        for k, op in enumerate(ops):
            t = (k + 1) * 0.1
            if op == "W":
                text = f"w{len(expected)}"
                expected.append((text, round(t * 1000)))
                double.push(word(text, t))
            else:
                double.push(end_word(t))
        seen = await drain.quiet(window_s=0.2)
        await client.close()
        seen += await drain.quiet(window_s=0.2)

        errors = [e.message for e in seen if e.type == "error"]
        assert errors == [], f"Word/EndWord traffic produced error events: {errors}"
        words = [e for e in seen if e.type == "word"]
        assert [(e.text, e.start_ms) for e in words] == expected, (
            f"ops={''.join(ops)} delivered {[(e.text, e.start_ms) for e in words]}"
        )
        assert all(0 <= e.start_ms <= e.end_ms for e in words), (
            f"ops={''.join(ops)} produced {[(e.start_ms, e.end_ms) for e in words]}"
        )

    asyncio.run(run())


# --- the marker case of the delivery rule -------------------------------------------------------


async def test_the_marker_echo_closes_the_open_word_before_the_flush():
    """Criteria 1 and 5 together: the flush confirms the word it is allowed to confirm."""
    client, double, drain = await connected()
    double.push(word("five.", 3.76))
    marker_id = await client.mark(4200)
    assert isinstance(marker_id, int)

    events = await drain.take(2)
    assert types_of(events) == ["word", "flushed"], f"order: {types_of(events)}"
    assert (events[0].text, events[0].start_ms, events[0].end_ms) == ("five.", 3760, 3760)
    assert events[1].t_eot_ms == 4200
    assert events[0].end_ms <= events[1].t_eot_ms
    assert client.stats().get("words_without_end") == 1
    assert double.marker_messages() == [marker(marker_id)]
    await client.close()


# --- the disconnect case of the delivery rule (ruling A7) ---------------------------------------


async def test_a_word_open_across_a_disconnect_is_published_on_the_old_clock_base():
    """Criterion 1 across a reconnect: a word may never be stitched onto the *next* clock base.

    moshi's model clock restarts at zero on every connection (docs/02 §2.3), so the only base a
    word opened on connection 1 can be closed against is connection 1's. Here the word opens at
    ``0.5 s`` on a base of 0 — ``start_ms = 500`` — the server then drops the connection, the
    session keeps feeding and the first block on connection 2 arrives at ``t_ms = 5000``, and the
    server's held ``EndWord(0.1)`` turns up on connection 2. Expected, by hand:

    * one word event, ``('one,', 500, 500)``: the disconnect publishes it with no known end
      (docs/02 §4 row 6 — the last word of a turn is never dropped), counted as an estimate;
    * ``stray_end_words == 1``: the late end has nothing left to close.

    Were the word instead held open across the disconnect, that ``EndWord`` would close it against
    the new base and publish ``(500, 5100)`` — a 4.6 s word spanning two clock bases, which T-0032
    would read as an end-of-turn 4.6 s after the word began (the reviewer's probe, ruling A7).
    """
    conn1 = SttConnectionDouble(close_after_audio=2, name="conn1")
    conn2 = SttConnectionDouble(name="conn2")
    connector = SttConnector([conn1, conn2])
    client = build_stt_client(None, connect=connector, max_reconnects=1)
    drain = EventDrain(client)

    await client.feed(block(), 0)  # connection 1's base: t_ms = 0
    await wait_until(lambda: conn1.audio_count >= 1, what="the first block reaching connection 1")
    conn1.push(word("one,", 0.5))
    held = await drain.quiet(window_s=0.15)
    assert types_of(held) == [], f"the word was published before its end arrived: {held}"

    await client.feed(block(), 80)  # the second block; the double closes on it
    await wait_until(lambda: connector.attempts >= 2, what="the client reconnecting")

    await client.feed(block(), 5000)  # connection 2's base: t_ms = 5000
    await wait_until(lambda: conn2.audio_count >= 1, what="the first block reaching connection 2")
    conn2.push(end_word(0.1))  # the end the server was holding when the socket died

    seen = await drain.quiet(window_s=0.25)
    await client.close()
    seen += await drain.quiet(window_s=0.25)

    words = [(e.text, e.start_ms, e.end_ms) for e in seen if e.type == "word"]
    assert words == [("one,", 500, 500)], (
        "a word opened on connection 1 is published on connection 1's base at the disconnect, "
        f"never closed against connection 2's; got {words}"
    )
    stats = client.stats()
    assert stats.get("stray_end_words") == 1, (
        "the EndWord arrives with nothing open: it is a stray, not the end of the old word; "
        f"stats said {dict(stats)}"
    )
    assert stats.get("words_without_end") == 1, "the disconnect publish is an estimate, counted"
    assert stats.get("reconnects") == 1, f"one reconnect was scripted; stats said {dict(stats)}"
