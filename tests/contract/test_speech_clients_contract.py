"""What must hold for ``FakeStt``/``FakeTts`` *and* for the two real clients alike — T-0007a.

``.claude/rules/tests.md``: every contract test is parametrised over ``backend=["fake", "real"]``.
Here ``fake`` means "no service": each assertion runs against **both** the in-memory fake (the
object the session is developed against) and the real client driven by the doubles in
``tests/fakes/`` (the object that will face moshi-server and Pocket TTS). ``real`` is ``gpu``-marked
and runs the same assertions against the live services; it skips unless

* ``GET {tts}/health`` answers **200**, and
* ``GET {stt as http}`` answers **400** — the measured "route mounted, wants an upgrade" signal
  (Runs, 2026-09-13). A **404** is a healthy process with no ASR route, and a health check that
  expects 200 reports a healthy server as dead.

The flushed row used to exempt the service case with ``all(e.t_eot_ms >= 0 for e in flushed)`` —
**vacuously true of an empty list** — while ``LiveSttCase`` stopped reading on
``drain.quiet(2.0)``, which returns after two seconds of quiet whether or not a flush ever came.
The first two live census runs (T-0007a group B, f114adf and 3e9813b) showed no ``Marker`` echo,
so that row had been reporting a confirmation nobody had. It now asserts the flush arrives for the
service too, and ``LiveSttCase`` keeps feeding silence at realtime pace until it does or
``LIVE_FLUSH_DEADLINE_S`` elapses: moshi-server does not chew its backlog once the audio inflow
stops (3e9813b: 20 ``Step``s in 15 s of silence against 90 for the 10.4 s fed), so a marker behind
unprocessed audio is never reached, and a live session's capture pump would not stop feeding either.
Census run 3 (38a7643, ruling A14) did exactly that and **saw the echo**:
``{'type': 'Marker', 'id': 1}`` mapped to ``flushed(t_eot_ms=7240)``, 3034 ms after the mark on the
CPU server — ``logs/t0007a-group-b-census-38a7643.log``. So the echo is an observed fact of this
box, not an open question; if it fails to come here the row is **red on the box** — that is
criterion 12's answer arriving as a failure, and the client still may not synthesise a flush to
make it green (criterion 4's adversarial half).
"""

from __future__ import annotations

import asyncio
import struct
import time
import wave
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.config import Services
from local_astra.contracts.protocols import SttClient, TtsBackend, TtsUtterance
from local_astra.contracts.speech import TTS_OUTPUT_FORMAT, WordTiming
from tests.fakes import FakeStt, FakeTts
from tests.fakes.moshi_stt_double import (
    EventDrain,
    SttConnectionDouble,
    SttConnector,
    build_stt_client,
    end_word,
    wait_until,
    word,
)
from tests.fakes.pocket_tts_double import TtsDouble, build_tts_client

pytestmark = pytest.mark.contract

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "pocket_tts_sentence.wav"
SENTENCE = "One, two, three, four, five."
BLOCK_SAMPLES = 1920
SAMPLE_WIDTH = 2  # s16le
#: How long the live case keeps feeding silence after ``mark()`` before giving up on the echo.
LIVE_FLUSH_DEADLINE_S = 15.0


def _skip_unless_services_are_up() -> None:
    services = Services()
    probe = services.stt.replace("ws://", "http://", 1)
    with httpx.Client(timeout=2.0) as client:
        try:
            health = client.get(f"{services.tts}/health")
            asr = client.get(probe)
        except httpx.HTTPError as exc:
            pytest.skip(f"speech services are not up ({type(exc).__name__}: {exc})")
    if health.status_code != 200:
        pytest.skip(f"pocket-tts /health answered {health.status_code}")
    if asr.status_code != 400:
        pytest.skip(f"{probe} answered {asr.status_code}; 400 is the mounted-route signal")


def _fixture_blocks() -> list[bytes]:
    """The committed rendering as 1920-sample f32le blocks — what ``stt_blocks()`` would hand over."""
    with wave.open(str(FIXTURE), "rb") as reader:
        pcm = reader.readframes(reader.getnframes())
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    blocks = []
    for i in range(0, len(samples), BLOCK_SAMPLES):
        chunk = samples[i : i + BLOCK_SAMPLES]
        blocks.append(struct.pack(f"<{len(chunk)}f", *(s / 32768.0 for s in chunk)))
    return blocks


# --- the cases ------------------------------------------------------------------------------------


class SttCase:
    """One STT implementation, driven to produce a few words and (maybe) a flush."""

    name = "?"
    expected_texts: list[str] | None = None
    #: the ``t_eot_ms`` this case asked its marker for; the flush must carry it back.
    t_eot_ms = 1500
    client: Any

    async def run(self) -> list[Any]:
        raise NotImplementedError

    async def aclose(self) -> None:
        await self.client.close()


class FakeSttCase(SttCase):
    name = "FakeStt"
    expected_texts = ["one", "two"]

    def __init__(self) -> None:
        script = [
            *FakeStt.words("one two", start_ms=1000, ms_per_word=250),
            FakeStt.flushed(1500),
        ]
        self.client = FakeStt(script)

    async def run(self) -> list[Any]:
        return [e async for e in self.client.events()]


class DoubleSttCase(SttCase):
    name = "MoshiSttClient/double"
    expected_texts = ["one", "two"]

    def __init__(self) -> None:
        self.double = SttConnectionDouble()
        self.client = build_stt_client(None, connect=SttConnector([self.double]))

    async def run(self) -> list[Any]:
        """The script the live server was measured to speak (T-0038, criterion 6).

        The group B census (``logs/t0007a-group-b-census-f114adf.log``) found an ``EndWord``
        after every ``Word``, so the double sends them too — the fake and the service must speak
        the same wire. Everything the client offers up to the flush is collected rather than a
        fixed count, because *when* a word is published is T-0038's business and the contract
        rows below are only about what the stream carries and in which order.
        """
        drain = EventDrain(self.client)
        await self.client.feed(struct.pack("<1920f", *([0.0] * 1920)), 0)
        await wait_until(lambda: self.double.audio_count == 1, what="the first block")
        self.double.push(word("one", 0.5))
        self.double.push(end_word(0.7))
        self.double.push(word("two", 0.75))
        self.double.push(end_word(0.95))
        await self.client.mark(1500)
        return await drain.quiet(window_s=0.3)


class LiveSttCase(SttCase):
    """The real service: fed at realtime pace, and **still fed** while waiting for the flush."""

    name = "MoshiSttClient/service"

    def __init__(self) -> None:
        self.client = build_stt_client()

    async def run(self) -> list[Any]:
        events: list[Any] = []
        flushed = asyncio.Event()

        async def drain() -> None:
            async for event in self.client.events():
                events.append(event)
                if event.type == "flushed":
                    flushed.set()

        task = asyncio.create_task(drain())
        try:
            t_ms = 0
            for block in _fixture_blocks():
                await self.client.feed(block, t_ms)
                t_ms += len(block) // 4 * 1000 // TTS_OUTPUT_FORMAT.sample_rate
                await asyncio.sleep(0.08)  # realtime pace; the client pads nothing (T-0021)
            for _ in range(40):  # 3.2 s of trailing silence: the harness's, not the client's
                await self.client.feed(bytes(BLOCK_SAMPLES * 4), t_ms)
                t_ms += 80
                await asyncio.sleep(0.08)
            self.t_eot_ms = t_ms
            await self.client.mark(t_ms)
            # Keep the pump running: the model's clock only advances while audio arrives, so a
            # marker behind unprocessed audio is never reached on a stopped stream (3e9813b).
            deadline = time.monotonic() + LIVE_FLUSH_DEADLINE_S
            while not flushed.is_set() and time.monotonic() < deadline:
                await self.client.feed(bytes(BLOCK_SAMPLES * 4), t_ms)
                t_ms += 80
                await asyncio.sleep(0.08)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return events


def stt_cases(backend: str) -> list[SttCase]:
    if backend == "real":
        _skip_unless_services_are_up()
        return [LiveSttCase()]
    return [FakeSttCase(), DoubleSttCase()]


def tts_pair(backend: str) -> list[tuple[str, Any]]:
    if backend == "real":
        _skip_unless_services_are_up()
        return [("PocketTtsClient/service", build_tts_client())]
    double = TtsDouble.serving(FIXTURE.read_bytes())
    return [
        ("FakeTts", FakeTts()),
        ("PocketTtsClient/double", build_tts_client(None, transport=double.transport)),
    ]


# --- the assertions ---------------------------------------------------------------------------------


async def test_the_implementations_satisfy_the_service_protocols(backend):
    for case in stt_cases(backend):
        assert isinstance(case.client, SttClient), case.name
        await case.aclose()
    for name, tts in tts_pair(backend):
        assert isinstance(tts, TtsBackend), name


async def test_words_arrive_as_word_events_in_arrival_order_with_non_negative_times(backend):
    for case in stt_cases(backend):
        events = await case.run()
        words = [e for e in events if e.type == "word"]
        assert words, f"{case.name}: no word arrived — the assertion below would be vacuous"
        assert all(w.start_ms >= 0 and w.end_ms >= w.start_ms for w in words), case.name
        if case.expected_texts is not None:
            assert [w.text for w in words] == case.expected_texts, case.name
        await case.aclose()


async def test_a_flushed_event_carries_the_t_eot_it_was_asked_for(backend):
    """docs/02 §4 row 6: the flush confirms every word ≤ t_eot has been delivered."""
    for case in stt_cases(backend):
        events = await case.run()
        flushed = [e for e in events if e.type == "flushed"]
        if case.name == "MoshiSttClient/service":
            # Not "if one arrives": an empty list satisfied that and hid two runs with no echo.
            assert flushed, (
                f"{case.name}: no flushed event within {LIVE_FLUSH_DEADLINE_S}s of mark() while "
                f"audio kept flowing; the echo is criterion 4's contract and the client may not "
                f"synthesise one. Event types seen: {[e.type for e in events]}"
            )
            assert flushed[0].t_eot_ms == case.t_eot_ms, (
                f"{case.name}: the flush must carry the t_eot its marker was asked for, got "
                f"{flushed[0].t_eot_ms} for {case.t_eot_ms}"
            )
            await case.aclose()
            continue
        assert len(flushed) == 1, f"{case.name}: {[e.type for e in events]}"
        assert flushed[0].t_eot_ms == case.t_eot_ms
        assert [e.type for e in events].index("flushed") == len(events) - 1, (
            f"{case.name}: the flush must land after the words it confirms"
        )
        await case.aclose()


async def test_close_ends_the_stream_and_is_idempotent(backend):
    for case in stt_cases(backend):
        await case.run()
        await case.aclose()
        await case.aclose()  # idempotent
        left = await asyncio.wait_for(_collect(case.client.events()), 2.0)
        assert left == [], f"{case.name}: events() still yielded {left} after close()"


async def _collect(iterator: Any) -> list[Any]:
    return [e async for e in iterator]


async def test_speak_returns_an_utterance_of_whole_24k_s16le_frames(backend):
    for name, tts in tts_pair(backend):
        utterance = await tts.speak("u1", SENTENCE)
        assert isinstance(utterance, TtsUtterance), name
        chunks = [c async for c in utterance.audio()]
        total = sum(len(c) for c in chunks)
        assert total > 0, name
        frame_bytes = SAMPLE_WIDTH * TTS_OUTPUT_FORMAT.channels
        assert total % frame_bytes == 0, f"{name}: {total} bytes is not a whole number of frames"
        assert all(len(c) % frame_bytes == 0 for c in chunks), f"{name}: a chunk splits a frame"


async def test_words_is_none_or_a_list_of_word_timings(backend):
    for name, tts in tts_pair(backend):
        utterance = await tts.speak("u1", SENTENCE)
        timings = utterance.words()
        assert timings is None or all(isinstance(w, WordTiming) for w in timings), name


async def test_cancel_yields_nothing_afterwards(backend):
    for name, tts in tts_pair(backend):
        utterance = await tts.speak("u1", SENTENCE)
        await asyncio.wait_for(utterance.cancel(), 1.0)
        left = await asyncio.wait_for(_collect(utterance.audio()), 2.0)
        assert left == [], f"{name}: {sum(len(c) for c in left)} bytes arrived after cancel()"
        await asyncio.wait_for(utterance.cancel(), 1.0)  # idempotent


# --- T-0051 criteria 8-9: a word the server never closed ------------------------------------------


async def test_a_word_with_no_endword_is_published_within_the_hold(backend):
    """A ``Word`` whose ``EndWord`` never arrives is published, bounded, and says its end is a guess.

    **The behaviour, and why it is here.** ``tests/fakes/stt.py`` gained the ability to script such
    a word (``FakeStt.open_word``); ``.claude/rules/tests.md`` says a behaviour added to a fake
    needs a real-service row in the same PR, and this is it. The three implementations must agree
    on the same two statements: the word **is** published (it stays in the transcript and in the
    log — criterion 12 suppresses the *decision*, never the word), and its ``end_estimated`` says
    whether anybody measured its end.

    ``fake``
        ``FakeStt`` replays the scripted word, flag and all; ``MoshiSttClient`` over the scripted
        connection is given a ``Word`` and no ``EndWord``, and the fed-audio clock is walked past
        ``STT_OPEN_WORD_HOLD_MS`` — the hold's own timing is pinned by
        ``tests/unit/test_stt_client_open_word_hold.py``, this row is about what comes out.

    ``real`` (``gpu``)
        moshi-server cannot be *told* to withhold an ``EndWord``, so the service half asserts the
        other side of the same contract on real speech: every published word carries the field,
        at least one carries ``False`` (the census of 2026-09-14 saw an ``EndWord`` after every
        ``Word``, so a client that flagged everything as an estimate is wrong and would make
        criterion 12 refuse every word the user says), and any word that *is* flagged claims no
        more audio than the hold allows. **This arm has never run**: it is owed to a ``/launch``
        on the 5090 box, and T-0051 records it as owed rather than as passed.

    Regime: fake arm — in-process, fed-audio clock, base 0, n = 1 deterministic per case. Real arm
    — profile ``lean``, moshi-server 0.6.4, ``tests/fixtures/audio/pocket_tts_sentence.wav``
    (4.040 s, "One, two, three, four, five.") fed at realtime pace plus 3.2 s of silence, n = 1.
    """
    from tests.fakes.stt import require_end_estimated
    from tests.unit.test_stt_client_open_word_hold import Pump, hold_ms, shape, words_of

    require_end_estimated()
    hold = hold_ms()

    if backend == "real":
        _skip_unless_services_are_up()
        case = LiveSttCase()
        events = await case.run()
        await case.aclose()
        words = [e for e in events if e.type == "word"]
        assert words, (
            "the service published no word at all, so every assertion below would be vacuous; "
            f"event types seen: {[e.type for e in events]}"
        )
        assert all(isinstance(getattr(w, "end_estimated", None), bool) for w in words), (
            f"criterion 8: every word carries the flag; got {shape(words)}"
        )
        assert any(w.end_estimated is False for w in words), (
            "criterion 8, the negative half on the box: moshi-server 0.6.4 sends an EndWord after "
            "every Word (T-0007a group B census, 2026-09-14, "
            "logs/t0007a-group-b-census-f114adf.log), so at least one of these ends was measured. "
            f"A client that flags everything as an estimate makes criterion 12 deaf. Got "
            f"{shape(words)}"
        )
        assert all(w.end_ms - w.start_ms <= hold for w in words if w.end_estimated), (
            f"criterion 9: a word the client had to close itself may claim at most {hold} ms of "
            f"audio — on 2026-09-16 one claimed 21 040. Got {shape(words)}"
        )
        return

    # (a) the fake: the scripted word comes back with its flag intact
    fake = FakeStt([FakeStt.open_word("see?", start_ms=1000, end_ms=1200)])
    fake_events = [e async for e in fake.events()]
    await fake.close()
    assert shape(words_of(fake_events)) == [("see?", 1000, 1200, True)], (
        f"FakeStt must carry criterion 8's flag through unchanged; got {shape(fake_events)}"
    )

    # (b) the real client over the scripted connection: a Word, no EndWord, and the clock walks on
    double = SttConnectionDouble()
    client = build_stt_client(None, connect=SttConnector([double]))
    drain = EventDrain(client)
    pump = Pump(client, double)
    await pump.start()
    arrival_ms = await pump.to(800)
    double.push(word("see?", 0.24))  # start_ms 240; no EndWord will ever follow
    # The hold starts when the client *sees* the Word, so the row must let it be seen before the
    # step clock runs on; the word is held, not published, which is asserted rather than assumed.
    assert words_of(await drain.quiet(window_s=0.05)) == [], "the word is published on no end yet"
    await pump.to(arrival_ms + hold + 160)
    seen = await drain.quiet(window_s=0.2)
    published = words_of(seen)
    await client.close()

    assert len(published) == 1 and published[0].text == "see?", (
        f"criterion 9: the word must be published once the hold expires, not held for ever (on "
        f"2026-09-16 it waited 21 040 ms and arrived inside the next sentence). Got "
        f"{shape(published)}"
    )
    assert published[0].end_estimated is True, (
        f"criterion 8: nobody measured this end, and the event must say so; got {shape(published)}"
    )
    assert 240 <= published[0].end_ms <= 240 + hold, (
        f"criterion 9: the end may not be invented beyond the hold; got {shape(published)}"
    )
    assert client.stats().get("held_word_timeouts") == 1, (
        f"criterion 9: the hold firing is counted; stats said {dict(client.stats())}"
    )
