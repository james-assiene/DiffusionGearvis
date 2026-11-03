"""The bounded hold on a word moshi-server never closed — T-0051 group B, criteria 8-11.

Pure (``tests/fakes/moshi_stt_double.py`` face (a)): no socket, no service, no GPU, and — the
point of this module — **no wall clock**. The client's step clock is the *fed-audio* clock: after
``feed(pcm, t_ms)`` it stands at ``t_ms + len(pcm)//4 * 1000 // 24000``, it is what a ``Step``
carries as ``t_ms``, and nothing else moves it. Every deadline below is expressed in it.

What happened, and why this module exists
=========================================
Desk session 2026-09-16 17:09:34, ``logs/sessions/20260916-170934-3380576.jsonl`` @ 28d7080
(profile ``lean``, moshi-server 0.6.4 ``kyutai/stt-1b-en_fr`` **on the GPU**, ``asr_delay_in_tokens``
6, ``batch_size`` 2, n = 1 session / 2 turns / 7 words, no seed). The human's last word — ``see?``
— was recognised and then never published: ``MoshiSttClient`` held it waiting for an ``EndWord``
that never arrived. Audio never stopped (``capture_gated_blocks`` 1372 × 10 ms = 13 720 ms =
exactly ``played_ms``, and the 80 ms step stream has no gap in 787 s, ruling A2), so the wait was
unbounded, not starved: the word came out **21 040 ms** later, closed by the *next* word's
arrival, as ``start_ms`` 247200 → ``end_ms`` 268240 (seq 3844) — a start time that is not when it
was said and a 21-second duration nobody measured. It then opened a turn 800 ms into the
assistant's own reply.

The seam these rows fix
=======================
The task names the behaviour, not every signature. These are the names the tests call; object to
them here rather than editing a row (CLAUDE.md: the implementer never edits ``tests/``).

``local_astra/conductor/stt_client.py``

    STT_OPEN_WORD_HOLD_MS: int
        How much **step-clock** time a ``Word`` with no ``EndWord`` may be held before it is
        published anyway. Recommended 1000 ms (~2x the 510 ms emission delay measured on the GPU
        on 2026-09-16); the band and its sources are
        :func:`test_the_open_word_hold_sits_in_its_band`.

    MoshiSttClient.stats() gains
        ``held_word_timeouts``          words published because the hold expired (criterion 9)
        ``word_delivery_lag_count``     words for which a delivery lag was measured (criterion 11)
        ``word_delivery_lag_min_ms``    min  of ``arrival_audio_clock_ms - end_ms`` over them
        ``word_delivery_lag_median_ms`` median of the same
        ``word_delivery_lag_max_ms``    max  of the same
        ``late_end_words``              ``EndWord``s that arrived after the hold had already
                                        given up on their word — *we* were early, which is the
                                        opposite of what ``stray_end_words`` reports (ruling B38)

``local_astra/contracts/speech.py``

    SttWordEvent gains ``end_estimated: bool = False`` — ``True`` exactly where the client
    estimates the end today (closed by the next word's arrival, by the marker echo, by
    ``close()``, by a disconnect) plus the new hold; ``False`` when a real ``EndWord`` closed it.

Measured sensitivity (test-author, 2026-09-16), against a reference implementation of criteria
8-11 built in an isolated copy — ``scratchpad/t0051b-probe-1213617``, ``PYTHONPATH`` and cwd set
to the copy, the editable ``.pth`` printed from inside the run to prove the import origin
(`.claude/rules/tests.md`). All nine rows pass against it in 2.1 s. Five mutations, each named by
the direction it is wrong in:

  mutation (what it gets wrong)                                            rows reddened
  the hold measured from the word's audio ``start_ms``, not its arrival        2
  the ``EndWord`` no longer closes the word (the hold publishes everything)    5
  every word flagged ``end_estimated=True``                                    3
  the median reported as the mean                                             1
  the hold kept on ``time.monotonic()`` instead of the step clock             1
  the word suppressed instead of published (the "too wide" direction)         1

Regime for every row: pure arithmetic on the fed-audio clock, 1920-sample (80 ms) blocks at
24 kHz, connection base 0, n = 1 deterministic per row, no seed, no service. Word times are the
server's ``start_time``/``stop_time`` in seconds multiplied by 1000 **by hand**, never read back
from the client.
"""

from __future__ import annotations

import struct
import time
from typing import Any

import pytest

from local_astra.conductor import stt_client as _sc
from local_astra.contracts import events as E
from tests.fakes.moshi_stt_double import (
    EventDrain,
    SttConnectionDouble,
    SttConnector,
    build_stt_client,
    end_word,
    marker,
    step,
    wait_until,
    word,
)
from tests.fakes.stt import require_end_estimated

BLOCK_SAMPLES = 1920  # 80 ms at 24 kHz (T-0021's framing; the client is its consumer)
BLOCK_MS = 1000 * BLOCK_SAMPLES // 24_000
#: The delivery lag measured on the GPU on 2026-09-16, in the fed-audio clock (T-0051 *Regime*:
#: 874-923 ms in session time, ~490-590 ms in audio time). Used as the gap between a word's audio
#: end and the step clock at which its ``Word`` message turns up.
GPU_LAG_MS = 560
#: How long a row waits before concluding "nothing else is coming". The client is local and
#: in-process, so every event it is going to produce is produced within a millisecond of the
#: message that causes it; this is a fast-suite budget, not a measurement.
QUIET_S = 0.15
#: The frozen-clock window of the hold row: real time passes, the step clock does not.
FROZEN_WALL_S = 0.2


def block(n: int = BLOCK_SAMPLES, value: float = 0.25) -> bytes:
    """``n`` f32le samples; 0.25 is exact in binary32, so the wire value is unambiguous."""
    return struct.pack(f"<{n}f", *([value] * n))


def hold_ms() -> int:
    """``STT_OPEN_WORD_HOLD_MS``, or the named assertion saying which criterion owes it."""
    value = getattr(_sc, "STT_OPEN_WORD_HOLD_MS", None)
    assert isinstance(value, int), (
        "T-0051 criterion 9: local_astra/conductor/stt_client.py must export "
        "STT_OPEN_WORD_HOLD_MS — the step-clock time a Word with no EndWord may be held before "
        "it is published with end_estimated=True. Recommended 1000 ms, ~2x the 510 ms emission "
        f"delay measured on 2026-09-16; got {value!r} (docs/tasks/T-0051.md)"
    )
    return value


class Pump:
    """The capture pump, as the client sees it: 80 ms blocks, and the step clock they carry.

    ``clock`` is computed here from the feed rule (``t_ms + 80``) rather than read off the
    client, so every deadline in this module is hand-computed and an implementation that moved
    its own clock would not move the test's expectations with it.
    """

    def __init__(self, client: Any, double: SttConnectionDouble) -> None:
        self.client = client
        self.double = double
        self.clock = 0
        self.blocks = 0

    async def start(self) -> None:
        """The first block: it sets the connection base (``t_ms`` 0 → base 0) and connects."""
        await self.client.feed(block(), 0)
        self.clock = BLOCK_MS
        self.blocks = 1
        await wait_until(lambda: self.double.audio_count >= 1, what="the first block arriving")

    async def to(self, target_ms: int, *, with_step: bool = True) -> int:
        """Feed until the step clock is **at or past** ``target_ms``; send one ``Step`` of that time.

        Returns where the clock actually landed: audio arrives in whole 80 ms blocks, so a
        deadline that is not on the grid is reached at the first step after it — the same
        quantisation the endpointer's arrival stamp has (T-0051 ruling A23). Rows assert their
        premises against the returned value, never against the deadline they asked for.

        Feeding **and** a ``Step`` on purpose: the clock the hold is measured in is advanced by
        ``feed`` and read by every inbound message, so a client that checks its open word on a
        feed, on a ``Step``, or on anything else arriving is equally served — and none of those is
        a wall clock.
        """
        assert target_ms >= self.clock, (
            f"the pump only moves forward: it is at {self.clock} and was asked for {target_ms}"
        )
        while self.clock < target_ms:
            await self.client.feed(block(), self.clock)
            self.clock += BLOCK_MS
            self.blocks += 1
        if with_step:
            self.double.push(step(prs=(0.1, 0.2, 0.3, 0.4), step_idx=self.blocks))
        return self.clock


async def connected(**knobs: Any) -> tuple[Any, SttConnectionDouble, EventDrain, Pump]:
    """A client on one scripted connection, already connected, with its pump at 80 ms."""
    double = SttConnectionDouble(**knobs)
    client = build_stt_client(None, connect=SttConnector([double]), max_reconnects=0)
    drain = EventDrain(client)
    pump = Pump(client, double)
    await pump.start()
    return client, double, drain, pump


def words_of(events: list[Any]) -> list[Any]:
    return [e for e in events if getattr(e, "type", None) == "word"]


def shape(events: list[Any]) -> list[tuple]:
    return [
        (e.text, e.start_ms, e.end_ms, getattr(e, "end_estimated", "<missing>")) for e in events
    ]


# --- criterion 9: the hold, in the step clock ---------------------------------------------------


async def test_an_open_word_is_published_when_the_hold_expires():
    """**Criterion 9.** A ``Word`` whose ``EndWord`` never comes is published within the hold.

    Hand-computed on the fed-audio clock, with ``H = STT_OPEN_WORD_HOLD_MS``:

      clock 800   ``Word('see?', start_time=0.24)`` arrives — ``start_ms`` 240, so the recogniser
                  is 560 ms behind, which is the delay measured on the GPU on 2026-09-16
                  (490-590 ms in audio time). No ``EndWord`` ever follows it.
      clock 800   … and the wall clock now runs for 200 ms while **no audio is fed**: nothing may
                  be published, because the hold is not a timer.
      clock 800+H-80  one step before the deadline: still nothing.
      clock 800+H     published, ``end_estimated=True``, ``held_word_timeouts`` = 1.

    Three ways to get this wrong, each killed by a named assertion:

    * **the two-clock version** (ruling A3's class of bug): holding from the word's *audio*
      ``start_ms`` = 240 rather than from the step clock at which it arrived = 800 expires
      560 ms early, at ``240 + H``, which is inside the "still nothing" window above. The task's
      own rule is the arrival: the recogniser had examined audio only up to 240 when the message
      turned up at 800.
    * **a wall timer.** The step clock here covers ``H`` ms of audio in a few ms of wall time, so
      the publication is asserted to happen with **less than ``H`` ms of wall clock elapsed** —
      which an implementation sleeping ``H`` ms cannot do — and the frozen-clock window above
      kills a wall timer shorter than 200 ms. (Wall timers between those two bounds are not
      excluded by this row; the module-level statement that the client's clock is the fed-audio
      clock is what makes them wrong.)
    * **an invented duration.** ``end_ms`` must lie in ``[start_ms, start_ms + H]``: at the moment
      the hold expires the recogniser has examined audio up to ``arrival - lag = start_ms + H``,
      so that is the most an honest end can claim. Tonight's 21 040 ms is 21x outside it.

    Regime: fed-audio clock only, 80 ms blocks, base 0, n = 1 deterministic.
    """
    require_end_estimated()
    hold = hold_ms()
    assert hold >= 240, (
        f"this row needs a hold of at least three 80 ms steps to have an inside; H = {hold}"
    )
    client, double, drain, pump = await connected()
    arrival_ms = await pump.to(800)
    assert arrival_ms == 800, f"the pump is off the grid: {arrival_ms}"

    started = time.perf_counter()
    double.push(word("see?", 0.24))  # start_ms 240; the EndWord never comes

    frozen = await drain.quiet(window_s=FROZEN_WALL_S)
    assert words_of(frozen) == [], (
        "criterion 9: 200 ms of **wall** time passed with the step clock standing still at "
        f"{arrival_ms} ms of audio, and a word was published anyway: {shape(words_of(frozen))}. "
        "The hold is measured in the clock the recogniser is behind, not in a timer"
    )

    inside_ms = await pump.to(arrival_ms + hold - BLOCK_MS)
    assert arrival_ms < inside_ms < arrival_ms + hold, (
        f"the premise of the 'still nothing' half: the clock must stand inside the hold; it is at "
        f"{inside_ms}, {inside_ms - arrival_ms} ms after the arrival, against a hold of {hold}"
    )
    early = await drain.quiet(window_s=QUIET_S)
    assert words_of(early) == [], (
        f"criterion 9: the word arrived at step clock {arrival_ms} and the clock has reached "
        f"{pump.clock} — {pump.clock - arrival_ms} ms of examined audio, one step inside the hold "
        f"of {hold} ms — yet it was already published: {shape(words_of(early))}. A hold measured "
        f"from the word's own audio start ({240}) instead of from its arrival expires "
        f"{arrival_ms - 240} ms early, which is exactly here (ruling A3: a duration is never "
        "computed from two clocks)"
    )

    await pump.to(arrival_ms + hold)
    seen = await drain.quiet(window_s=QUIET_S)
    if not words_of(seen):  # up to two further steps of slack, as criterion 2's row allows
        await pump.to(arrival_ms + hold + 2 * BLOCK_MS)
        seen += await drain.quiet(window_s=QUIET_S)
    wall_ms = (time.perf_counter() - started) * 1000.0
    published = words_of(seen)

    assert len(published) == 1, (
        f"criterion 9: {hold} ms of step-clock time passed after a Word with no EndWord (arrival "
        f"{arrival_ms}, clock now {pump.clock}) and the word must be published. Got "
        f"{shape(published)}; on 2026-09-16 this word waited 21 040 ms and came out at the head of "
        "the next sentence"
    )
    event = published[0]
    assert event.text == "see?" and event.start_ms == 240, (
        f"the published word keeps its own text and server start time; got {shape(published)}"
    )
    assert event.end_estimated is True, (
        "criterion 8/9: a word the hold released was never closed by the server, so its end is an "
        f"estimate and must say so; got end_estimated={event.end_estimated!r}"
    )
    assert 240 <= event.end_ms <= 240 + hold, (
        f"criterion 9: the end may not be invented beyond the hold. At the deadline the recogniser "
        f"has examined audio up to start_ms + H = {240 + hold}, so an honest end lies in "
        f"[240, {240 + hold}]; got {event.end_ms} (a duration of {event.end_ms - 240} ms). "
        "Tonight's fabrication was 21 040 ms"
    )
    stats = client.stats()
    assert stats.get("held_word_timeouts") == 1, (
        "criterion 9: the hold firing is a fact about the recogniser and must be counted in "
        f"stats()['held_word_timeouts']; stats said {dict(stats)}"
    )
    assert stats.get("words_without_end", 0) >= 1, (
        "…and it is an estimate like every other, so T-0038's counter must not stop counting it; "
        f"stats said {dict(stats)}"
    )
    assert wall_ms < hold, (
        f"the whole row took {wall_ms:.1f} ms of wall clock while {pump.clock - arrival_ms} ms of "
        f"audio went by: an implementation that waited {hold} ms of *wall* time could not have "
        "published yet, and on the box it would hold the last word of every turn for as long as "
        "the machine is slow (`.claude/rules/conductor.md`: time arrives in the event)"
    )
    await client.close()
    assert client.stats().get("held_word_timeouts") == 1, (
        "…and close() must not count the same word a second time"
    )


async def test_the_open_word_hold_sits_in_its_band():
    """Criterion 9's new constant, pinned to a band whose bounds come from outside the module.

    Lower bound **600 ms**: the recogniser's own emission delay was 490-590 ms in the fed-audio
    clock on the GPU on 2026-09-16 (874-923 ms in session time, T-0051 *Regime*), so a hold
    shorter than that would pre-empt ``EndWord`` messages that are merely in flight — it would
    turn every word into an estimate and make criterion 10 unsatisfiable on the box.
    Upper bound **5000 ms**: the hold exists to bound the 21 040 ms wait observed at the desk, and
    it must sit well under ``EOT_SILENCE_MS`` + a turn's worth of speech so a held final word is
    released while its turn is still the current one. 1000 ms — about twice the measured delay —
    is the value the task recommends inside that band; anything in it is defensible **with its
    reason written down**.

    Not a tautology: both bounds are read from the session log and the task file, never from the
    value in the module.
    """
    hold = hold_ms()
    assert 600 <= hold <= 5000, (
        f"STT_OPEN_WORD_HOLD_MS={hold}: >= 600 ms because the recogniser's own delivery delay was "
        "490-590 ms of audio time on 2026-09-16 and a shorter hold pre-empts EndWords in flight; "
        "<= 5000 ms because it exists to bound the 21 040 ms wait that lost the human's last word"
    )
    assert hold < 21_040, (
        f"STT_OPEN_WORD_HOLD_MS={hold} would not have bounded tonight's 21 040 ms hold, which is "
        "the one case criterion 9 names"
    )


# --- criterion 10: a real EndWord still wins ----------------------------------------------------


@pytest.mark.parametrize("inside_ms", [400, None], ids=["400 ms in", "one step before expiry"])
async def test_an_endword_inside_the_hold_still_wins(inside_ms):
    """**Adversarial, criterion 10.** The row that kills "publish everything on the timeout".

    **The implementation this row kills**: one that lets the hold expire on *every* word — the
    ``EndWord`` handler no longer closes it and the timeout publishes it instead. That build
    satisfies criterion 9 completely, loses every real end time the server sends, and marks every
    word an estimate, which criterion 12 then refuses: the endpointer would open no turn at all.
    Measured: it reddens five of this module's nine rows, both of this one's cases among them. Here the ``EndWord`` arrives inside the hold —
    400 ms in, and (the sharp case) one 80 ms step before it expires — and the word must be
    published with the server's own ``stop_time`` and ``end_estimated=False``:

      clock 800        ``Word('five.', start_time=0.24)`` → ``start_ms`` 240
      clock 800 + d    ``EndWord(stop_time=1.0)`` → ``end_ms`` **1000**, by hand
      expected         ``('five.', 240, 1000, end_estimated=False)``, ``held_word_timeouts`` 0

    1000 is distinguishable from every value the timeout path could produce — the word's start
    (240), the honest ceiling at expiry (240 + H) and the step clock at publication — so "the
    timeout won anyway" cannot pass by coincidence. The clock is then run far past the deadline
    and the stream must stay silent: a word published twice is a word the endpointer sees twice.

    Regime: fed-audio clock only, base 0, n = 1 deterministic per case.
    """
    require_end_estimated()
    hold = hold_ms()
    delay = hold - BLOCK_MS if inside_ms is None else inside_ms
    client, double, drain, pump = await connected()
    arrival_ms = await pump.to(800)
    double.push(word("five.", 0.24))
    assert words_of(await drain.quiet(window_s=0.05)) == [], (
        "a word with no end is held, not published — and the hold starts when the client sees the "
        "Word, so this row waits for that before moving the clock"
    )
    at_ms = await pump.to(arrival_ms + delay)
    assert 0 < at_ms - arrival_ms < hold, (
        f"the premise of this row: the EndWord must land **inside** the hold. The pump asked for "
        f"{delay} ms and landed {at_ms - arrival_ms} ms after the arrival at {arrival_ms}, against "
        f"a hold of {hold} ms — recompute rather than relaxing the assertion"
    )
    double.push(end_word(1.0))

    seen = await drain.quiet(window_s=QUIET_S)
    published = words_of(seen)
    assert shape(published) == [("five.", 240, 1000, False)], (
        f"criterion 10: an EndWord arriving {at_ms - arrival_ms} ms into a hold of {hold} ms "
        "closes the word "
        "with the server's own stop_time (1.0 s → 1000 ms) and end_estimated=False. The timeout "
        "must not pre-empt it, and must not re-publish it afterwards. Got "
        f"{shape(published)}"
    )
    stats = client.stats()
    assert stats.get("held_word_timeouts") == 0, (
        f"criterion 10: nothing timed out here — the server closed the word. stats said "
        f"{dict(stats)}"
    )
    assert stats.get("words_without_end") == 0, (
        f"…and a word with a real end is not an estimate; stats said {dict(stats)}"
    )

    await pump.to(at_ms + 3 * hold)  # long past every deadline
    after = await drain.quiet(window_s=QUIET_S)
    assert words_of(after) == [], (
        f"criterion 10: the word was already published; the hold must not publish it again "
        f"{shape(words_of(after))}"
    )
    await client.close()


# --- criterion 8: which ends are estimates, and which are not -----------------------------------


async def test_a_word_closed_by_the_next_words_arrival_is_marked_estimated():
    """**Criterion 8**, on tonight's actual mechanism: ``see?`` ended where the next word began.

    The log's seq 3844: the held word's ``end_ms`` 268240 is exactly the next word's ``start_ms``,
    which is how anybody could tell the duration was invented. That behaviour stays (T-0038
    criterion 1) — what changes is that the event now *says* the end is an estimate, so the
    endpointer can refuse to open a turn on it (criterion 12).

    Hand-computed, base 0, the census's own words: ``Word('three,', 2.24)``, then ``Word('four,',
    3.04)`` with **no ``EndWord`` between them**, then ``EndWord(3.76)``. Expected

      ``('three,', 2240, 3040, end_estimated=True)``   — closed by the next word's start
      ``('four,',  3040, 3760, end_estimated=False)``  — closed by a real EndWord

    The second line is what makes this row two-sided: an implementation that flags every word as
    an estimate passes the first and fails the second, and it would make criterion 12 refuse every
    word the user ever says.

    The step clock does not move between the two messages, so the hold is not what closes the
    first word here; that is the separate path of the row above.
    """
    require_end_estimated()
    client, double, drain, pump = await connected()
    await pump.to(3200, with_step=False)
    double.push(word("three,", 2.24))
    double.push(word("four,", 3.04))
    double.push(end_word(3.76))

    seen = await drain.quiet(window_s=QUIET_S)
    assert shape(words_of(seen)) == [
        ("three,", 2240, 3040, True),
        ("four,", 3040, 3760, False),
    ], (
        "criterion 8: a word closed by the *next* word's arrival has an end nobody measured and "
        "must carry end_estimated=True; a word closed by a real EndWord must carry False. Got "
        f"{shape(words_of(seen))}"
    )
    stats = client.stats()
    assert stats.get("words_without_end") == 1, (
        f"T-0038's counter is unchanged by this task: one estimate here. stats said {dict(stats)}"
    )
    assert stats.get("held_word_timeouts", 0) == 0, (
        f"nothing was held: the step clock never moved between the two words. {dict(stats)}"
    )
    await client.close()


async def test_every_other_path_that_estimates_an_end_flags_the_word():
    """Criterion 8's "exactly where the client passes ``estimated=True`` today", path by path.

    Four of the five paths the criterion names (the fifth, the hold, has its own row above), each
    driven to publish one word and each asserted on the flag *and* on the times, because "it is an
    estimate" and "the end is the word's own start" are two different claims:

      marker echo   ``mark()`` → the server echoes ``Marker`` → the open word goes out first
                    (T-0038 criterion 5), ``('five.', 3760, 3760, True)``, then the flush
      close()       the last word of a turn is never dropped (docs/02 §4 row 6)
      disconnect    the server drops the socket; the word is published on the **old** clock base
      inverted end  an ``EndWord`` before the word's start is clamped to the start, never a
                    negative duration (T-0038's named failure mode)

    Regime: base 0 everywhere, hand-computed millisecond values from the census's seconds.
    """
    require_end_estimated()

    # (a) the marker echo
    client, double, drain, _ = await connected()
    double.push(word("five.", 3.76))
    marker_id = await client.mark(4200)
    seen = await drain.quiet(window_s=QUIET_S)
    assert shape(words_of(seen)) == [("five.", 3760, 3760, True)], (
        f"criterion 8, marker echo: {shape(words_of(seen))}"
    )
    assert [e.type for e in seen if e.type in ("word", "flushed")] == ["word", "flushed"], (
        f"T-0038 criterion 5: the word goes out before the flush that confirms it; {seen}"
    )
    assert double.marker_messages() == [marker(marker_id)]
    await client.close()

    # (b) close()
    client_b, double_b, drain_b, _ = await connected()
    double_b.push(word("five.", 3.76))
    held = await drain_b.quiet(window_s=QUIET_S)
    assert words_of(held) == [], (
        f"the word is held, not published, until something closes it {held}"
    )
    await client_b.close()
    tail = await drain_b.quiet(window_s=QUIET_S)
    assert shape(words_of(tail)) == [("five.", 3760, 3760, True)], (
        f"criterion 8, close(): {shape(words_of(tail))}"
    )

    # (c) the disconnect (the shape of tests/unit/test_stt_client_endword.py's ruling A7 row)
    conn1 = SttConnectionDouble(close_after_audio=2, name="conn1")
    conn2 = SttConnectionDouble(name="conn2")
    connector = SttConnector([conn1, conn2])
    client_c = build_stt_client(None, connect=connector, max_reconnects=1)
    drain_c = EventDrain(client_c)
    await client_c.feed(block(), 0)
    await wait_until(lambda: conn1.audio_count >= 1, what="the first block on connection 1")
    conn1.push(word("one,", 0.5))
    await client_c.feed(block(), 80)  # the double closes on this one
    await wait_until(lambda: connector.attempts >= 2, what="the client reconnecting")
    seen_c = await drain_c.quiet(window_s=QUIET_S)
    assert shape(words_of(seen_c)) == [("one,", 500, 500, True)], (
        "criterion 8, disconnect: a word open across a disconnect is published on the old base "
        f"with no known end, and that end is an estimate. Got {shape(words_of(seen_c))}"
    )
    await client_c.close()

    # (d) the inverted end, clamped
    client_d, double_d, drain_d, _ = await connected()
    double_d.push(word("late", 2.0))
    double_d.push(end_word(1.5))  # 500 ms *before* the word started
    seen_d = await drain_d.quiet(window_s=QUIET_S)
    assert shape(words_of(seen_d)) == [("late", 2000, 2000, True)], (
        "criterion 8, clamped end: an end before the start is not a measurement; the word is "
        f"published at its own start, flagged. Got {shape(words_of(seen_d))}"
    )
    await client_d.close()


async def test_the_flag_stays_on_the_stt_wire_and_off_the_session_event():
    """Criterion 8's negative: the field never appears on the HUD wire.

    ``end_estimated`` is a fact about the *recogniser's* output and it belongs on
    ``contracts/speech.py``'s ``SttWordEvent``, which the endpointer reads. What the session
    publishes — and therefore what the HUD and the turn record see — is ``contracts/events.py``'s
    ``SttWord``, which this task does not own and must not change; ``session.py`` is required to
    be byte-identical (criterion 16), so an implementer who widened the session event would have
    had to break that too. This row is the cheap guard that says so out loud.
    """
    require_end_estimated()
    assert "end_estimated" not in E.SttWord.model_fields, (
        "criterion 8, negative: the flag stays on the STT contract. contracts/events.py's SttWord "
        "is what reaches the HUD and the turn record, it is not a file this task owns, and "
        f"criterion 16 requires session.py to be byte-identical. Got {sorted(E.SttWord.model_fields)}"
    )


# --- criterion 11: the delivery lag, as three separate statistics --------------------------------


async def test_the_delivery_lag_is_recorded_per_word():
    """**Criterion 11.** ``arrival_audio_clock_ms - end_ms``, counted and summarised.

    This is the number that says whether the recogniser is keeping up, and it is the number
    criterion 2's rule is protecting: on 2026-09-16 it was 874/875/890/923 ms in session time,
    ~490-590 ms in the fed-audio clock, and the backstop was subtracting it from its own budget.

    Five words, each closed by a real ``EndWord`` pushed when the step clock stands at a chosen
    value. Hand-computed (seconds × 1000, base 0; ``arrival`` is the fed-audio clock at which the
    closing message is handled, which is where the pump was left):

      word      end_ms   arrival   lag
      one,        560      1120     560
      two,       1520      1920     400
      three,     2320      3520    1200   ← the CPU recogniser's order of magnitude (ruling A21)
      four,      3120      3600     480
      five.      3920      4560     640

    Sorted: 400, 480, 560, 640, 1200 → **count 5, min 400, median 560, max 1200**.

    The three statistics are asserted **separately**, never as a sum and never as "the keys are
    present": each is the kind of number a later session is compared against, so a client that
    reported the same value three times, or the arrival order's middle element, or the mean, has
    not recorded what criterion 11 asks for. The fixture is chosen so all four traps are visible —
    the mean is 656 (not the median 560), the middle element *in arrival order* is 1200 (not the
    median), and min, median and max are three different numbers.

    Regime: fed-audio clock only, base 0, n = 5 words in 1 deterministic stream.
    """
    schedule = [
        # text,    start_time, stop_time, arrival clock, lag by hand
        ("one,", 0.24, 0.56, 1120, 560),
        ("two,", 1.20, 1.52, 1920, 400),
        ("three,", 2.00, 2.32, 3520, 1200),
        ("four,", 2.80, 3.12, 3600, 480),
        ("five.", 3.60, 3.92, 4560, 640),
    ]
    lags = sorted(lag for *_, lag in schedule)
    assert lags == [400, 480, 560, 640, 1200], "recompute the table"
    assert sum(lags) // len(lags) != lags[2], "the fixture must separate the median from the mean"

    client, double, drain, pump = await connected()
    seen: list[Any] = []
    for text, start_s, stop_s, arrival, lag in schedule:
        assert round(stop_s * 1000) + lag == arrival, (
            f"{text}: recompute — arrival {arrival} is end {round(stop_s * 1000)} plus lag {lag}"
        )
        await pump.to(arrival, with_step=False)
        double.push(word(text, start_s))
        double.push(end_word(stop_s))
        seen += await drain.take(1)

    assert [(e.text, e.end_ms) for e in words_of(seen)] == [
        (text, round(stop_s * 1000)) for text, _, stop_s, _, _ in schedule
    ], f"the row's premise: five words with the server's own ends; got {shape(words_of(seen))}"

    stats = dict(client.stats())
    assert stats.get("word_delivery_lag_count") == 5, (
        f"criterion 11: five words were delivered and each one's lag is one observation; {stats}"
    )
    assert stats.get("word_delivery_lag_min_ms") == 400, (
        f"criterion 11: the minimum of {lags} is 400 ms (the word the recogniser was quickest "
        f"with); {stats}"
    )
    assert stats.get("word_delivery_lag_median_ms") == 560, (
        f"criterion 11: the median of {lags} is 560 ms — not the mean (656), not the middle of "
        f"the arrival order (1200), and not the min or the max; {stats}"
    )
    assert stats.get("word_delivery_lag_max_ms") == 1200, (
        f"criterion 11: the maximum of {lags} is 1200 ms — the observation that says the "
        f"recogniser fell behind, which is the whole reason to keep this number; {stats}"
    )
    await client.close()


# --- T-0038's behaviour that must not regress ---------------------------------------------------


async def test_a_marker_echo_still_closes_the_open_word():
    """T-0038 criterion 5, re-pinned beside the hold that could break it.

    docs/02 §4 row 6: the flush promises every word up to ``t_eot`` has been delivered, so a word
    still waiting for its end goes out **before** the ``SttFlushedEvent``. The hold introduces a
    second way for an open word to be published, and an implementation that moved the close into
    the timeout path only would leave the flush promising a word nobody had sent — the session
    then leaves ENDPOINTING with the last word of the sentence missing from the transcript, which
    is a worse version of the defect this task exists to fix.

    Hand-computed: ``Word('five.', 3.76)`` → ``start_ms`` 3760; ``mark(4200)``; the double echoes
    the marker; expected ``('five.', 3760, 3760, end_estimated=True)`` then ``flushed(4200)``, in
    that order, with **no** step-clock advance at all — the marker path must not need one.
    """
    require_end_estimated()
    client, double, drain, _ = await connected()
    double.push(word("five.", 3.76))
    marker_id = await client.mark(4200)
    assert isinstance(marker_id, int)

    events = await drain.take(2)
    assert [e.type for e in events] == ["word", "flushed"], (
        f"the open word is published before the flush that confirms it; got {[e.type for e in events]}"
    )
    assert shape([events[0]]) == [("five.", 3760, 3760, True)], (
        f"the marker echo closes the word with no invented end, flagged: {shape([events[0]])}"
    )
    assert events[1].t_eot_ms == 4200 and events[0].end_ms <= events[1].t_eot_ms
    assert client.stats().get("words_without_end") == 1
    assert client.stats().get("held_word_timeouts", 0) == 0, (
        "the marker closed it, not the hold — the step clock never moved"
    )
    await client.close()


# --- rulings B32/B33: four decisions that were written after these rows, and owed one each -------
#
# Every row below is green against ``cffd860`` and exists for the mutation it reddens: the ruling
# was made on an implementer's report, so the decision landed in ``docs/tasks/T-0051.md`` and
# nothing in ``tests/`` learned about it. The second reviewer mutated each ruling and the whole
# targeted set stayed at 192 passed. Each row names its mutation in the docstring; a row here that
# does not go red under that mutation has not earned its place.


async def test_the_delivery_lag_median_is_the_low_middle_of_an_even_count():
    """**Ruling B25(ii).** ``statistics.median_low``, not ``statistics.median`` — even n.

    Every number in the delivery-lag record must be a lag **some word actually had**; averaging
    the two middle observations of an even count reports a latency the recogniser never showed.
    The existing five-word row cannot see this: for odd n ``median_low`` and ``median`` return the
    same element, and its docstring excludes ``statistics.mean`` (656) — a *neighbour* of the
    function the ruling is about, which is how the gap hid.

    **The mutation this row exists for:** ``statistics.median_low(lags)`` → ``statistics.median(lags)``
    in ``MoshiSttClient.stats()``. Four server-closed words, hand-computed on the fed-audio clock
    (base 0, seconds × 1000 by hand, ``lag = arrival_audio_clock_ms - end_ms``):

      word      end_ms   arrival   lag
      one,        560      1040     480
      two,       1520      1920     400
      three,     2320      3520    1200
      four,      3920      4560     640

    Sorted: 400, 480, 640, 1200 → **count 4, min 400, median_low 480, max 1200**. The three
    values this row excludes are all different numbers: ``statistics.median`` is 560, the mean is
    680, and the high middle is 640. None of them is an observation.

    Regime: fed-audio clock only, 80 ms blocks, base 0, n = 4 words in 1 deterministic stream.
    """
    schedule = [
        # text,   start_time, stop_time, arrival clock, lag by hand
        ("one,", 0.24, 0.56, 1040, 480),
        ("two,", 1.20, 1.52, 1920, 400),
        ("three,", 2.00, 2.32, 3520, 1200),
        ("four,", 3.60, 3.92, 4560, 640),
    ]
    lags = sorted(lag for *_, lag in schedule)
    assert lags == [400, 480, 640, 1200], "recompute the table"
    assert len(lags) % 2 == 0 and lags[1] != lags[2], (
        "the premise of this row: an even count whose two middle observations differ, so "
        f"median_low ({lags[1]}) and median ({(lags[1] + lags[2]) / 2}) disagree; got {lags}"
    )

    client, double, drain, pump = await connected()
    seen: list[Any] = []
    for text, start_s, stop_s, arrival, lag in schedule:
        assert round(stop_s * 1000) + lag == arrival, (
            f"{text}: recompute — arrival {arrival} is end {round(stop_s * 1000)} plus lag {lag}"
        )
        landed = await pump.to(arrival, with_step=False)
        assert landed == arrival, (
            f"{text}: the pump is off the 80 ms grid — it was asked for {arrival} and landed on "
            f"{landed}, so the lag this row asserts is not the lag it computed"
        )
        double.push(word(text, start_s))
        double.push(end_word(stop_s))
        seen += await drain.take(1)

    assert [(e.text, e.end_ms, e.end_estimated) for e in words_of(seen)] == [
        (text, round(stop_s * 1000), False) for text, _, stop_s, _, _ in schedule
    ], f"the row's premise: four words the *server* closed; got {shape(words_of(seen))}"

    stats = dict(client.stats())
    assert stats.get("word_delivery_lag_count") == 4, (
        f"the premise of the even count: four observations, not five; {stats}"
    )
    assert stats.get("word_delivery_lag_median_ms") == 480, (
        f"ruling B25(ii): the median of {lags} is statistics.median_low = 480 ms — a lag the word "
        "'one,' actually had. statistics.median would report 560, the mean 680 and the high "
        f"middle 640, and no word was ever delivered with any of those; {stats}"
    )
    assert stats.get("word_delivery_lag_min_ms") == 400, f"min of {lags}; {stats}"
    assert stats.get("word_delivery_lag_max_ms") == 1200, f"max of {lags}; {stats}"
    await client.close()


async def test_only_server_closed_words_enter_the_delivery_lag_statistics():
    """**Ruling B25(i), adversarial.** A word this client closed is not evidence about the server.

    The statistic answers "is the recogniser keeping up". A held word's ``end_ms`` is a number
    *this client invented*, so counting it would measure our own guess and report it as the
    recogniser's latency — a client that graded its own homework, invisible in every row today
    because the five-word fixture's words are all server-closed. Held words are counted
    separately, as ``held_word_timeouts``.

    **The mutation this row exists for:** deleting the ``else:`` in ``_close_open_word`` so that
    every published word appends a lag. One stream, two words, hand-computed (base 0):

      clock 1040   ``Word('one,', 0.24)`` + ``EndWord(0.56)``  → end 560, lag **480**, server-closed
      clock 2000   ``Word('two,', 1.20)``, no ``EndWord`` ever → released by the hold at 3040

    Expected: ``word_delivery_lag_count`` **1** with min = median = max = 480, and
    ``held_word_timeouts`` **1**. Under the mutation the held word contributes
    ``3040 - 1200 = 1840`` — a 1.8 s "delivery lag" for a delivery that never happened — so the
    count reads 2 and the max 1840.

    Regime: fed-audio clock only, 80 ms blocks, base 0, n = 2 words in 1 deterministic stream.
    """
    require_end_estimated()
    hold = hold_ms()
    client, double, drain, pump = await connected()

    closed_at = await pump.to(1040, with_step=False)
    assert closed_at == 1040, f"the pump is off the grid: {closed_at}"
    double.push(word("one,", 0.24))
    double.push(end_word(0.56))
    first = await drain.take(1)
    assert shape(first) == [("one,", 240, 560, False)], (
        f"the row's premise: the server closed this one; got {shape(first)}"
    )

    held_at = await pump.to(2000, with_step=False)
    assert held_at == 2000, f"the pump is off the grid: {held_at}"
    double.push(word("two,", 1.20))
    assert words_of(await drain.quiet(window_s=0.05)) == [], (
        "the second word has no end yet, so it is held — and the hold starts when the client sees "
        "the Word, which is what this wait establishes"
    )
    released_at = await pump.to(held_at + hold)
    seen = await drain.quiet(window_s=QUIET_S)
    if not words_of(seen):  # up to two further steps of slack, as criterion 9's row allows
        released_at = await pump.to(released_at + 2 * BLOCK_MS)
        seen += await drain.quiet(window_s=QUIET_S)
    assert shape(words_of(seen)) == [("two,", 1200, 1200, True)], (
        f"criterion 9's premise for this row: the hold released the second word at step clock "
        f"{released_at} with an end this client supplied; got {shape(words_of(seen))}"
    )

    stats = dict(client.stats())
    assert stats.get("held_word_timeouts") == 1, (
        f"one word was released by the hold and it is counted there, not as a delivery; {stats}"
    )
    assert stats.get("word_delivery_lag_count") == 1, (
        "ruling B25(i): only the server-closed word is an observation of the recogniser's "
        "latency. The held word's end_ms (1200) is this client's own guess, and counting "
        f"{released_at} - 1200 = {released_at - 1200} ms as 'the recogniser's delivery lag' would "
        f"report our arithmetic as the server's behaviour; {stats}"
    )
    assert (
        stats.get("word_delivery_lag_min_ms"),
        stats.get("word_delivery_lag_median_ms"),
        stats.get("word_delivery_lag_max_ms"),
    ) == (480, 480, 480), (
        f"the one observation is 1040 - 560 = 480 ms, so all three summaries are it. A held word "
        f"in the list would push the max to {released_at - 1200}; {stats}"
    )
    await client.close()


async def test_a_delivery_lag_that_comes_out_negative_stays_signed():
    """**Ruling B27, adversarial.** The clocks disagreeing is the finding, not something to hide.

    A server ``EndWord`` whose ``stop_time`` lies **beyond the audio anybody has fed** claims an
    end for audio that does not exist yet. Clamping the lag at zero would erase exactly the
    disagreement between the recogniser's clock and the fed-audio clock that this whole task is
    about (ruling A3), and dropping the observation would make ``word_delivery_lag_count``
    disagree with the number of server-closed words — the record lying about how many
    observations it has. On a real server it cannot occur; if it ever appears at a desk, it *is*
    the finding, so it must be visible in ``stats()``.

    **The mutation this row exists for:** ``self._word_delivery_lags.append(x)`` →
    ``append(max(0, x))``. No other fixture in this module produces a negative. Hand-computed
    (base 0, seconds × 1000 by hand):

      clock 1040   ``Word('one,', 0.24)`` + ``EndWord(4.44)`` → end **4440**, 3400 ms past the
                   1040 ms of audio fed so far → lag **−3400**
      clock 2560   ``Word('two,', 1.20)`` + ``EndWord(1.52)`` → end 1520 → lag **+1040**

    Sorted: −3400, 1040 → **count 2, min −3400, median_low −3400, max 1040**. Under the clamp the
    list reads 0, 1040 and the min and median both become 0 — a client reporting perfect latency
    on the one wire inconsistency it should be shouting about.

    Regime: fed-audio clock only, 80 ms blocks, base 0, n = 2 words in 1 deterministic stream.
    """
    require_end_estimated()
    client, double, drain, pump = await connected()

    early = await pump.to(1040, with_step=False)
    assert early == 1040, f"the pump is off the grid: {early}"
    double.push(word("one,", 0.24))
    double.push(end_word(4.44))  # an end 3400 ms past every sample that has been fed
    first = await drain.take(1)
    assert shape(first) == [("one,", 240, 4440, False)], (
        "the server measured this end, however impossible it looks from here: the client passes "
        "it through as a measurement (it is not before the start, so T-0038's clamp does not "
        f"apply) rather than silently rewriting the server's number; got {shape(first)}"
    )

    late = await pump.to(2560, with_step=False)
    assert late == 2560, f"the pump is off the grid: {late}"
    double.push(word("two,", 1.20))
    double.push(end_word(1.52))
    second = await drain.take(1)
    assert shape(second) == [("two,", 1200, 1520, False)], (
        f"the row's second, ordinary observation; got {shape(second)}"
    )

    stats = dict(client.stats())
    assert stats.get("dropped_blocks") == 0, (
        f"the premise of every arrival in this row: no block was dropped on the way in; {stats}"
    )
    assert stats.get("word_delivery_lag_count") == 2, (
        "ruling B27: two words were closed by the server, so the record has two observations. "
        "Skipping the negative one would make the count disagree with the number of server-closed "
        f"words, which makes the record lie about its own n; {stats}"
    )
    assert stats.get("word_delivery_lag_min_ms") == -3400, (
        f"ruling B27: the lag is {early} - 4440 = -3400 ms and it stays signed. Clamping it to 0 "
        "hides a server claiming an end for audio nobody has fed — the two-clock disagreement "
        f"this task exists to make visible; {stats}"
    )
    assert stats.get("word_delivery_lag_median_ms") == -3400, (
        f"the low middle of [-3400, 1040] is -3400 (ruling B25(ii)); a clamp reports 0; {stats}"
    )
    assert stats.get("word_delivery_lag_max_ms") == 1040, (
        f"the maximum of [-3400, 1040] is the ordinary word's 1040 ms; {stats}"
    )
    await client.close()


async def test_feeding_alone_releases_a_held_word_with_no_message_arriving():
    """**Ruling B33, adversarial — the 21-second failure, reconstructed for T-0048's feed gate.**

    Every other row in this module that reaches the hold also pushes a ``Step`` in the same
    iteration, so an implementation that released a held word **only** on an inbound message
    satisfies all of them: deleting either expiry call site leaves 192 rows green, and only
    deleting both reddens anything. This row is the one that pins the *feed* side on its own. It
    pushes the ``Word`` and then **nothing else** — no ``Step``, no ``EndWord``, no ``Marker`` —
    and moves the step clock with audio alone.

    Why it matters: a client that stops receiving messages holds the word for ever, which is the
    21 040 ms wait of 2026-09-16 exactly, and **T-0048's feed gate is the configuration that stops
    the messages**. The proof that no message arrived is that the client emits no ``SttStepEvent``
    in the whole row: the only event it ever produces is the word.

    Hand-computed, ``H = STT_OPEN_WORD_HOLD_MS``, fed-audio clock, base 0:

      clock 800       ``Word('see?', 0.24)`` → ``start_ms`` 240, arrival 800 (the 560 ms measured
                      on the GPU on 2026-09-16). No ``EndWord`` ever.
      clock 800+H-80  fed there in 80 ms blocks, no message pushed: **still held** — feeding is not
                      by itself a release, the deadline is (the "too wide" direction).
      clock 800+H     fed there, still no message pushed: published, ``end_estimated=True``,
                      ``held_word_timeouts`` 1, end inside ``[start_ms, start_ms + H]`` (ruling B24).

    Regime: fed-audio clock only, 80 ms blocks, base 0, n = 1 deterministic, no Step messages.
    """
    require_end_estimated()
    hold = hold_ms()
    client, double, drain, pump = await connected()
    arrival_ms = await pump.to(800, with_step=False)
    assert arrival_ms == 800, f"the pump is off the grid: {arrival_ms}"
    double.push(word("see?", 0.24))
    assert words_of(await drain.quiet(window_s=0.05)) == [], (
        "the word has no end, so it is held — and the hold's zero is the clock at which the "
        "client saw the Word, which is what this wait establishes before the pump moves"
    )

    inside_ms = await pump.to(arrival_ms + hold - BLOCK_MS, with_step=False)
    assert arrival_ms < inside_ms < arrival_ms + hold, (
        f"the premise of the 'still held' half: {inside_ms} must lie inside the hold of {hold} ms "
        f"that started at {arrival_ms}"
    )
    early = await drain.quiet(window_s=QUIET_S)
    assert words_of(early) == [], (
        f"ruling B33, the too-wide direction: {inside_ms - arrival_ms} ms of audio has been fed "
        f"against a hold of {hold} ms, so feeding must not release the word yet — a feed-side "
        f"release that fires on any block would publish the word here: {shape(words_of(early))}"
    )

    released_at = await pump.to(arrival_ms + hold, with_step=False)
    seen = await drain.quiet(window_s=QUIET_S)
    if not words_of(seen):  # up to two further blocks of slack, as criterion 9's row allows
        released_at = await pump.to(released_at + 2 * BLOCK_MS, with_step=False)
        seen += await drain.quiet(window_s=QUIET_S)

    assert shape(words_of(seen)) == [("see?", 240, 240, True)], (
        f"ruling B33: {released_at - arrival_ms} ms of *fed audio* has gone by since the Word "
        f"arrived at {arrival_ms}, against a hold of {hold} ms, and **no message at all** was "
        "pushed in between. feed() must release the word on its own: an implementation that only "
        "expires the hold when something arrives on the socket holds this word until the next "
        "message — for ever if there is none, which is the 21 040 ms wait of 2026-09-16 and "
        f"exactly what T-0048's feed gate produces. Got {shape(words_of(seen))}"
    )
    assert [e.type for e in drain.seen] == ["word"], (
        "the proof that nothing arrived on the socket: a Step would have produced an SttStepEvent, "
        f"so the word above was released by feed() and by nothing else. Got "
        f"{[e.type for e in drain.seen]}"
    )
    stats = dict(client.stats())
    assert stats.get("held_word_timeouts") == 1, (
        f"the hold fired, on the feed side, and is counted exactly once; {stats}"
    )
    assert 240 <= words_of(seen)[0].end_ms <= 240 + hold, (
        f"ruling B24: the end may not be invented beyond the hold; got {words_of(seen)[0].end_ms}"
    )
    await client.close()


async def test_an_endword_that_arrives_after_the_hold_fired_gets_its_own_counter():
    """**Ruling B38, adversarial — the counter this change gives two opposite meanings.**

    ``stray_end_words`` exists to say *the server is ahead of us* (its own comment): an ``EndWord``
    for a word this client had not opened, which is a benign race on reconnect. The hold introduces
    a second way to reach that branch, and it means the opposite thing — **we gave up on a word the
    server did close**, i.e. ``STT_OPEN_WORD_HOLD_MS`` is shorter than this recogniser's delivery
    delay and we are inventing ends for words that were about to be measured. That is the more
    serious diagnosis, it is the number the next desk session reads, and a counter that means two
    things is how an evening gets wasted on the wrong one. The two conditions must land in
    **different** counters; this row does not care about the spelling beyond it naming the late
    arrival, but ``stats()`` must expose both (ruling B26: the seam names are the tests').

    Three halves, because a rename satisfies none of them on its own:

      (a) hold fires, **then** the real ``EndWord`` arrives → ``late_end_words`` 1,
          ``stray_end_words`` **0**. Hand-computed: ``Word('see?', 0.24)`` at clock 800, no end;
          the hold releases it at 800 + H with ``end_ms`` 240; ``EndWord(1.0)`` arrives after
          that. The word is *not* published a second time — the endpointer must not see it twice.
      (b) an ``EndWord`` with no word ever opened → ``stray_end_words`` 1, ``late_end_words``
          **0**. This is the condition the counter was built for and it must keep its meaning.
      (c) **ruling B40**: the credit a fired hold leaves behind is cleared by the next ``Word``.
          Hold fires on ``'see?'``; then ``Word('next,')`` opens and its own ``EndWord`` closes
          it; then an unmatched ``EndWord`` arrives → ``stray_end_words`` 1, ``late_end_words``
          **0**. Without the clearing, one early give-up would hold a permanent claim on the next
          unmatched end and report "we gave up on a word the server closed" about a message that
          arrived hours later and had nothing to do with it — a counter that accumulates a
          diagnosis it cannot support is worse than the shared counter it replaced.

    Regime: fed-audio clock only, 80 ms blocks, base 0, n = 1 deterministic stream per half.
    """
    require_end_estimated()
    hold = hold_ms()

    # (a) the hold fired first, and the server's real end turns up afterwards
    client, double, drain, pump = await connected()
    arrival_ms = await pump.to(800)
    double.push(word("see?", 0.24))
    assert words_of(await drain.quiet(window_s=0.05)) == [], "the word is held, not published"
    released_at = await pump.to(arrival_ms + hold)
    seen = await drain.quiet(window_s=QUIET_S)
    if not words_of(seen):  # up to two further steps of slack, as criterion 9's row allows
        released_at = await pump.to(released_at + 2 * BLOCK_MS)
        seen += await drain.quiet(window_s=QUIET_S)
    assert shape(words_of(seen)) == [("see?", 240, 240, True)], (
        f"the premise of this row: the hold published the word at {released_at} with an invented "
        f"end, before the server's own end arrived; got {shape(words_of(seen))}"
    )

    double.push(end_word(1.0))  # the measurement, 240 ms of audio late
    after = await drain.quiet(window_s=QUIET_S)
    assert words_of(after) == [], (
        "criterion 10's rule, from the other side: the word is already out, so the late EndWord "
        f"must not publish it a second time; got {shape(words_of(after))}"
    )
    stats = dict(client.stats())
    assert stats.get("late_end_words") == 1, (
        "ruling B38: an EndWord arriving after the hold gave up on its word says the hold is "
        f"shorter than this recogniser's delivery delay ({released_at - arrival_ms} ms here, "
        "against an emission delay of ~510 ms on the GPU on 2026-09-16) and that we are inventing "
        "ends for words the server was about to measure. stats() needs a counter named for that "
        f"arrival; got {stats}"
    )
    assert stats.get("stray_end_words") == 0, (
        "ruling B38: and it is **not** a stray. stray_end_words means 'the server is ahead of "
        "us, not broken' — the benign reconnect race — and the desk session that reads it must "
        f"not have to guess which of the two opposite diagnoses the number is reporting; {stats}"
    )
    assert stats.get("held_word_timeouts") == 1, (
        f"the hold fired exactly once and the late end does not un-count it; {stats}"
    )
    assert stats.get("word_delivery_lag_count") == 0, (
        "ruling B25(i): this word's published end is one this client invented, so nothing about "
        f"it is an observation of the recogniser's delivery lag; {stats}"
    )
    await client.close()

    # (b) the condition the counter was built for keeps its meaning
    client_b, double_b, drain_b, pump_b = await connected()
    await pump_b.to(1040, with_step=False)
    double_b.push(end_word(0.56))  # no Word was ever opened on this connection
    assert words_of(await drain_b.quiet(window_s=QUIET_S)) == [], (
        "an EndWord with no word to close publishes nothing (T-0038 ruling A2)"
    )
    stats_b = dict(client_b.stats())
    assert stats_b.get("stray_end_words") == 1, (
        f"ruling B38's other side: the server being ahead of us is still a stray; {stats_b}"
    )
    assert stats_b.get("late_end_words") == 0, (
        "…and it is not a late arrival: nothing was held and nothing was given up on, so a "
        f"rename of stray_end_words does not satisfy this criterion; {stats_b}"
    )
    assert stats_b.get("held_word_timeouts") == 0, f"nothing was held here; {stats_b}"
    await client_b.close()

    # (c) ruling B40: an intervening Word clears the credit the fired hold left behind
    client_c, double_c, drain_c, pump_c = await connected()
    arrival_c = await pump_c.to(800)
    double_c.push(word("see?", 0.24))
    assert words_of(await drain_c.quiet(window_s=0.05)) == [], "the word is held, not published"
    released_c = await pump_c.to(arrival_c + hold)
    seen_c = await drain_c.quiet(window_s=QUIET_S)
    if not words_of(seen_c):  # up to two further steps of slack, as criterion 9's row allows
        released_c = await pump_c.to(released_c + 2 * BLOCK_MS)
        seen_c += await drain_c.quiet(window_s=QUIET_S)
    assert shape(words_of(seen_c)) == [("see?", 240, 240, True)], (
        f"(c)'s premise, as in (a): the hold gave up on this word at {released_c}; got "
        f"{shape(words_of(seen_c))}"
    )

    double_c.push(word("next,", 1.60))  # the next word opens: the credit is spent, not banked
    double_c.push(end_word(1.92))  # …and the server closes this one itself
    assert shape(words_of(await drain_c.take(1))) == [("next,", 1600, 1920, False)], (
        "(c)'s premise: a whole word came and went between the give-up and the stray end"
    )
    double_c.push(end_word(2.40))  # unmatched, and nothing is owed to it any more
    assert words_of(await drain_c.quiet(window_s=QUIET_S)) == [], (
        "an EndWord with nothing open publishes nothing (T-0038 ruling A2)"
    )
    stats_c = dict(client_c.stats())
    assert stats_c.get("stray_end_words") == 1, (
        "ruling B40: a Word opened after the hold fired, so the unmatched end that follows it "
        "cannot be the end of the word we gave up on — the server is simply ahead of us. "
        f"{stats_c}"
    )
    assert stats_c.get("late_end_words") == 0, (
        "ruling B40: the credit a fired hold leaves is cleared by the next Word. An unbounded "
        "credit would let a single early give-up blame us for the next unmatched end whenever it "
        f"arrived, which is a diagnosis the number cannot support; {stats_c}"
    )
    assert stats_c.get("held_word_timeouts") == 1, (
        f"the hold did fire in this half, exactly once, and that count is unaffected; {stats_c}"
    )
    await client_c.close()
