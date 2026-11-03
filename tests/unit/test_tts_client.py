"""``PocketTtsClient`` over ``httpx.MockTransport`` — T-0007a group A, criteria 9, 10, 11.

Pure: no service, no GPU, no socket (``tests/fakes/pocket_tts_double.py``). The one real artefact
is ``tests/fixtures/audio/pocket_tts_sentence.wav`` — a *recording* of Pocket TTS 3.1.0, because
the service is not deterministic (``tests/fixtures/audio/MANIFEST.md``), decoded here by the
``wave`` module as an independent oracle.

Seams this module fixes
=======================
``local_astra/conductor/tts_client.py``

    PocketTtsClient(services=None, *, transport=None, clock=None, chunk_ms=80)
        ``services`` is a ``local_astra.config.Services`` (``None`` → ``Services()``). ``transport``
        is an ``httpx.AsyncBaseTransport``, exactly as ``VllmClient(transport=…)`` does it.
        ``clock`` is a zero-argument callable returning **monotonic seconds** (default
        ``time.perf_counter``) — the only clock ``first_audio_latency_ms`` may use, so no unit test
        waits on wall-clock time. ``chunk_ms`` is the emitted chunk size (default 80 ms = 3840
        bytes of 24 kHz mono s16le).
    client.url -> str                    ``services.tts``; ``POST {url}/tts`` with form field ``text``
    await client.speak(utterance_id, text, voice="default") -> TtsUtterance

    TtsClientError(Exception) with ``.status: int | None``   — connect failure, non-200, timeout,
        and a mid-stream failure after the first chunk. Deliberately not named ``TtsError``: that
        is ``contracts.events.TtsError``, the event T-0007's session emits when it catches this.
    TtsFormatError(TtsClientError)                          — the body is not 24 kHz mono s16le.

    The utterance is ``contracts/protocols.py::TtsUtterance`` plus ``failed: bool`` (criterion 11:
    a mid-stream failure is loud, and the session must be able to see it after the fact).

Nothing here is a threshold: the two wall-clock bounds (100 ms cancel, and that ``speak`` issues
its request before ``audio()`` is awaited) are contract statements from
``contracts/protocols.py``/criterion 9, not the B5 budget, which this task does not gate.
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import time
import wave
from pathlib import Path

import httpx
import pytest

from local_astra.config import Services
from local_astra.contracts.speech import TTS_OUTPUT_FORMAT
from tests.fakes.pocket_tts_double import (
    BYTES_PER_MS,
    CANONICAL_HEADER_BYTES,
    STREAMING_DATA_SIZE,
    StepClock,
    TtsDouble,
    build_tts_client,
    canonical_wav,
    chunked,
    make_wav,
    ramp_pcm,
    require_utterance_surface,
    sine_pcm,
    tts_error_classes,
    wav_44100_stereo,
    wav_with_list_chunk,
)

SENTENCE = "One, two, three, four, five."
CHUNK_BYTES = 80 * BYTES_PER_MS  # 3840

#: What the live Pocket TTS 3.1.0 puts in front of a ~117 kB body (measured 2026-09-14 against
#: 127.0.0.1:8081 on CPU; docs/tasks/T-0007a.md ruling A10): a canonical 44-byte header declaring
#: ``data`` = 0x77359400 and ``RIFF`` = 0x77359424. Neither ``0`` nor ``0xFFFFFFFF``, so criterion
#: 10's two named placeholders do not cover it and ruling A6's short-body rule fired on every real
#: sentence (``test_speak_returns_an_utterance_of_whole_24k_s16le_frames[real]``).
POCKET_TTS_HEADER_DATA_SIZE = 2_000_000_000
POCKET_TTS_HEADER_RIFF_SIZE = 2_000_000_036

#: The longest utterance a declared size is believed about: 600 s of ``TTS_OUTPUT_FORMAT``
#: (ruling A10). Spelled out here from the rule, not imported from the client — a test that reads
#: the constant it is checking asserts nothing.
MAX_UTTERANCE_S = 600
MAX_PLAUSIBLE_DECLARED_BYTES = 600 * 24_000 * 2  # 28_800_000


def open_fds() -> int:
    return len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else -1


async def next_chunk(iterator, timeout_s: float = 1.0) -> bytes | None:
    """The next chunk, ``None`` at a clean end; a hang is an assertion, never a stuck test."""
    try:
        return await asyncio.wait_for(iterator.__anext__(), timeout_s)
    except StopAsyncIteration:
        return None
    except TimeoutError as exc:
        raise AssertionError(f"audio() produced nothing within {timeout_s}s") from exc


async def drain_audio(utterance, *, limit: int = 5000) -> list[bytes]:
    iterator = utterance.audio().__aiter__()
    out: list[bytes] = []
    while len(out) < limit:
        chunk = await next_chunk(iterator)
        if chunk is None:
            return out
        out.append(chunk)
    raise AssertionError(f"audio() yielded more than {limit} chunks")


# --- criterion 9: the sentence, and only the sentence -------------------------------------------


async def test_speak_posts_the_text_verbatim_and_a_non_default_voice_raises():
    """Criterion 9: one POST, form field ``text``, byte-for-byte the argument, issued in speak()."""
    before = open_fds()
    double = TtsDouble.serving(canonical_wav(sine_pcm(100)))
    client = build_tts_client(None, transport=double.transport)
    assert open_fds() == before, "constructing the client opened a file descriptor"
    assert client.url == Services().tts  # the endpoint comes from the config, not a literal
    assert double.requests == []

    text = "Sure — that's 42, isn't it?  "  # punctuation, unicode, trailing space: all preserved
    utterance = await client.speak("u1", text)
    require_utterance_surface(utterance)

    assert len(double.requests) == 1, "speak() must issue the request, not audio()"
    request = double.requests[0]
    assert request.method == "POST" and str(request.url) == "http://127.0.0.1:8081/tts"
    assert double.forms() == [{"text": [text]}], double.bodies

    with pytest.raises(ValueError):  # Pocket TTS 3.1.0 has no voice parameter
        await client.speak("u2", "hello", voice="af_heart")
    assert len(double.requests) == 1, "a rejected voice must not reach the service"


async def test_words_is_always_none():
    """Criterion 9 negative: Pocket TTS reports no word timings, so the client invents none.

    A character-rate estimate here is indistinguishable from a measurement to
    ``PipeWireAudioIO.play(..., source=…)``, and T-0006 criterion 6 would cut ``played_text`` on
    invented boundaries — history would then contain words nobody heard.
    """
    double = TtsDouble.serving(canonical_wav(sine_pcm(200)))
    client = build_tts_client(None, transport=double.transport)
    for i, text in enumerate(["one", "one two three four five", "", "a" * 400]):
        utterance = await client.speak(f"u{i}", text)
        assert utterance.words() is None
        await drain_audio(utterance)
        assert utterance.words() is None  # still None once the duration is known


# --- criterion 10: the wav is parsed, not assumed -------------------------------------------------


async def test_chunks_reconstruct_the_pcm_of_the_committed_pocket_tts_fixture(fixtures_dir):
    """Criterion 10 against the real rendering, with ``wave`` as the independent oracle."""
    path = fixtures_dir / "audio" / "pocket_tts_sentence.wav"
    body = path.read_bytes()
    with wave.open(str(path), "rb") as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) == (
            TTS_OUTPUT_FORMAT.sample_rate,
            TTS_OUTPUT_FORMAT.channels,
            2,
        )
        assert reader.getnframes() == 96_960  # MANIFEST.md
        oracle = reader.readframes(reader.getnframes())
    assert len(body) - len(oracle) == CANONICAL_HEADER_BYTES  # this file's header really is 44

    double = TtsDouble.serving(body)
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)
    chunks = await drain_audio(utterance)

    assert b"".join(chunks) == oracle, "the audio is not the file's PCM"
    assert not chunks[0].startswith(b"RIFF"), "header bytes were emitted as audio"
    assert max(len(c) for c in chunks) <= CHUNK_BYTES
    assert all(len(c) == CHUNK_BYTES for c in chunks[:-1])
    assert len(chunks) == math.ceil(len(oracle) / CHUNK_BYTES) == 51
    assert len(chunks[-1]) == len(oracle) % CHUNK_BYTES == 1920  # the last chunk may be short
    width = 2 if TTS_OUTPUT_FORMAT.sample_format == "s16le" else 4
    rate, channels = TTS_OUTPUT_FORMAT.sample_rate, TTS_OUTPUT_FORMAT.channels
    assert CHUNK_BYTES == 80 * rate * channels * width // 1000 == 3840  # 80 ms, per the contract


@pytest.mark.parametrize("data_size", [None, 0, STREAMING_DATA_SIZE])
async def test_a_wav_with_a_list_chunk_and_a_bogus_data_size_still_yields_exact_pcm(data_size):
    """Criterion 10, adversarial: this is the file that kills a fixed-offset-44 implementation."""
    pcm = ramp_pcm(5000)
    body = wav_with_list_chunk(pcm, data_size=data_size)
    assert body.index(b"data") + 8 != CANONICAL_HEADER_BYTES, "the LIST chunk must move ``data``"
    assert body[:CANONICAL_HEADER_BYTES].count(b"LIST") == 1

    double = TtsDouble.serving(body)
    client = build_tts_client(None, transport=double.transport)
    chunks = await drain_audio(await client.speak("u1", SENTENCE))
    assert b"".join(chunks) == pcm, "the data chunk was not located by walking the RIFF chunks"


async def test_a_body_that_ends_short_of_its_declared_data_size_raises():
    """Criterion 10 / ruling A6: a declared ``data`` size is authoritative, so a short body is loud.

    ``0`` and ``0xFFFFFFFF`` mean "read to EOF"; **any other** size is the server telling us how
    much audio it is sending, and a stream that stops before it is a truncated rendering. Handing
    it back as if it were complete is criterion 11's failure in another costume: the session would
    commit ``played_text`` for a sentence the user only heard the first fifth of.
    """
    tts_client_error, _ = tts_error_classes()
    pcm = ramp_pcm(20_000)  # 40 000 bytes of s16le
    body = make_wav(pcm)[: CANONICAL_HEADER_BYTES + 8_000]
    assert struct.unpack("<I", body[40:44])[0] == 40_000  # the file still *claims* 40 000 bytes
    assert len(body) - CANONICAL_HEADER_BYTES == 8_000  # ... and carries 8 000

    double = TtsDouble.serving(body)
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)

    delivered = []
    with pytest.raises(tts_client_error) as caught:
        async for chunk in utterance.audio():
            delivered.append(chunk)
    audio = b"".join(delivered)
    assert len(audio) <= 8_000 and audio == pcm[: len(audio)], "the short body was rewritten"
    assert utterance.failed is True, "a truncated rendering must not look like a complete one"
    assert "Traceback" not in str(caught.value) and len(str(caught.value)) <= 300


async def test_the_header_the_live_pocket_tts_sends_is_read_to_eof():
    """Ruling A10: the placeholder the real service streams is a placeholder, not a length.

    Pinned as the exact bytes the census saw, because this is the one header that matters and it
    is *not* one of criterion 10's two documented placeholders. Getting it wrong is not subtle:
    every real utterance raises, which is how it was found (T-0028 review A, against the service).
    """
    tts_client_error, _ = tts_error_classes()
    pcm = ramp_pcm(20_000)  # 40 000 bytes, far short of the two billion the header declares
    body = make_wav(
        pcm, data_size=POCKET_TTS_HEADER_DATA_SIZE, riff_size=POCKET_TTS_HEADER_RIFF_SIZE
    )
    assert len(body) - len(pcm) == CANONICAL_HEADER_BYTES, "the real header is canonical, 44 bytes"
    assert body[4:8] == struct.pack("<I", 2_000_000_036)  # 0x77359424
    assert body[40:44] == struct.pack("<I", 2_000_000_000)  # 0x77359400
    derived = MAX_UTTERANCE_S * TTS_OUTPUT_FORMAT.sample_rate * TTS_OUTPUT_FORMAT.channels * 2
    assert derived == MAX_PLAUSIBLE_DECLARED_BYTES == 28_800_000  # 600 s of the output format

    double = TtsDouble.serving(body)
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)
    chunks = await drain_audio(utterance)

    assert b"".join(chunks) == pcm, "the body is the audio; the declared size is a placeholder"
    assert utterance.failed is False
    assert tts_client_error  # named so the failure mode above is unambiguous if this ever raises


@pytest.mark.parametrize(
    ("declared", "authoritative"),
    [
        (40_000, True),  # ruling A6: a plausible length, and the body is short of it
        (MAX_PLAUSIBLE_DECLARED_BYTES, True),  # 600 s exactly: still believed
        (MAX_PLAUSIBLE_DECLARED_BYTES + 1, False),  # one byte past: nobody renders that
        (POCKET_TTS_HEADER_DATA_SIZE, False),  # what the service sends
        (STREAMING_DATA_SIZE, False),  # criterion 10's documented placeholder
        (0, False),  # ... and the other one
    ],
)
async def test_a_declared_data_size_is_a_length_only_while_it_is_plausible(declared, authoritative):
    """Rulings A6 + A10, as one table: where the short-body rule applies and where it does not.

    The boundary is pinned on both sides. Below it a short body is a truncated rendering and must
    be loud (a silent truncation makes ``played_text`` claim words nobody heard); above it the
    header is somebody's placeholder and the body is the audio.
    """
    tts_client_error, _ = tts_error_classes()
    pcm = ramp_pcm(4_000)  # 8 000 bytes: shorter than every declared size in the table
    double = TtsDouble.serving(make_wav(pcm, data_size=declared))
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)

    if authoritative:
        with pytest.raises(tts_client_error):
            await drain_audio(utterance)
        assert utterance.failed is True
    else:
        assert b"".join(await drain_audio(utterance)) == pcm
        assert utterance.failed is False


@pytest.mark.parametrize(
    ("body_kind", "why"),
    [
        ("44100_stereo", "would play at the wrong pitch and in the wrong channel count"),
        ("8bit", "is not s16le"),
        ("32bit", "is not s16le"),
        ("not_riff", "is not a wav at all"),
    ],
)
async def test_a_non_24k_mono_s16_wav_raises_instead_of_playing_at_the_wrong_rate(body_kind, why):
    """Criterion 10: the format is validated against ``TTS_OUTPUT_FORMAT`` before a byte is yielded."""
    _, tts_format_error = tts_error_classes()
    pcm = ramp_pcm(2000)
    bodies = {
        "44100_stereo": wav_44100_stereo(pcm),
        "8bit": make_wav(pcm, bits=8),
        "32bit": make_wav(pcm, bits=32),
        "not_riff": b'{"detail": "no model loaded"}',
    }
    double = TtsDouble.serving(bodies[body_kind])
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)
    with pytest.raises(tts_format_error):
        await drain_audio(utterance)


async def test_first_audio_latency_is_none_until_the_first_chunk_then_measured():
    """Criterion 10: measured from ``speak()`` to the first chunk, on the injected clock."""
    pcm = ramp_pcm(4000)
    clock = StepClock(10.0)  # an arbitrary origin: the client must measure a difference
    double = TtsDouble.streaming(chunked(canonical_wav(pcm), 1000))
    client = build_tts_client(None, transport=double.transport, clock=clock)

    utterance = await client.speak("u1", SENTENCE)
    assert utterance.first_audio_latency_ms is None
    iterator = utterance.audio().__aiter__()
    clock.advance(0.25)
    assert await next_chunk(iterator) is not None
    assert utterance.first_audio_latency_ms == pytest.approx(250.0)

    clock.advance(1.0)
    assert await next_chunk(iterator) is not None
    assert utterance.first_audio_latency_ms == pytest.approx(250.0), "measured once, at the first"


async def test_a_second_call_to_audio_raises_rather_than_replaying():
    """Criterion 10: the body is a stream, not a buffer; a replay would invent audio."""
    tts_client_error, _ = tts_error_classes()
    double = TtsDouble.serving(canonical_wav(sine_pcm(100)))
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)
    assert await drain_audio(utterance)
    with pytest.raises((tts_client_error, RuntimeError)):
        await drain_audio(utterance)


# --- criterion 11: cancel is fast and silent, failure is loud --------------------------------------


async def test_cancel_returns_within_100ms_and_no_chunk_follows():
    """Criterion 11: ``contracts/protocols.py::TtsUtterance.cancel`` — barge-in depends on it."""
    body = canonical_wav(ramp_pcm(200_000))
    # Raw chunks far larger than one 80 ms output chunk, so that several output chunks are
    # already decoded and buffered when the cancel lands: "no further chunk" has to mean the
    # buffer too, not just the socket (the mutation that yields one more from the buffer).
    double = TtsDouble.streaming(chunked(body, 20_000), delay_s=0.02, repeat_last=True)
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)
    iterator = utterance.audio().__aiter__()
    assert await next_chunk(iterator) is not None

    started = time.perf_counter()
    try:
        await asyncio.wait_for(utterance.cancel(), 1.0)
    except TimeoutError as exc:
        raise AssertionError("cancel() did not return within 1 s") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 100, f"cancel() took {elapsed_ms:.0f} ms"

    assert await next_chunk(iterator) is None, "a chunk arrived after cancel()"
    assert double.streams[0].closed, "the response was left open"
    assert not utterance.failed, "a cancel is not a failure"
    await asyncio.wait_for(utterance.cancel(), 1.0)  # idempotent


@pytest.mark.parametrize("when", ["before_first_anext", "during_header_parse"])
async def test_cancel_before_the_first_anext_and_during_header_parse_leak_nothing(when):
    """Criterion 11: the T-0005 M5 window — the generator does nothing until the first ``__anext__``.

    That is exactly the window barge-in lands in: ``speak()`` has already issued the request, so
    there is a live response to close even though no chunk has been asked for yet.
    """
    baseline = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    header = canonical_wav(ramp_pcm(4000))[:20]  # a *partial* RIFF header, then nothing
    double = TtsDouble.streaming([header], tail="hang")
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)
    iterator = utterance.audio().__aiter__()

    if when == "during_header_parse":
        pending = asyncio.create_task(iterator.__anext__())
        await asyncio.sleep(0.05)  # parked inside the header parse
        assert not pending.done()
    await asyncio.wait_for(utterance.cancel(), 1.0)

    if when == "during_header_parse":
        with pytest.raises((StopAsyncIteration, asyncio.CancelledError)):
            await asyncio.wait_for(pending, 1.0)
    assert await next_chunk(iterator) is None, "audio was delivered after a cancel"
    assert double.streams[0].closed, "the response was left open"

    for _ in range(20):
        await asyncio.sleep(0.01)
    leaked = {
        t for t in asyncio.all_tasks() if t not in baseline and t is not asyncio.current_task()
    }
    assert leaked == set(), f"cancel leaked {leaked}"


async def test_a_mid_stream_failure_raises_out_of_audio_rather_than_truncating_silently():
    """Criterion 11: a truncated stream that looks complete makes ``played_text`` lie."""
    tts_client_error, _ = tts_error_classes()
    body = canonical_wav(ramp_pcm(20_000))
    parts = chunked(body, 4096)
    double = TtsDouble.streaming(
        parts[:2], tail=httpx.ReadError("the connection died mid-sentence")
    )
    client = build_tts_client(None, transport=double.transport)
    utterance = await client.speak("u1", SENTENCE)
    assert not utterance.failed

    with pytest.raises(tts_client_error):
        await drain_audio(utterance)
    assert utterance.failed is True


@pytest.mark.parametrize("failure", ["status_500", "connect_error", "timeout"])
async def test_speak_raises_a_tts_client_error_on_non_200_connect_failure_and_timeout(failure):
    """Criterion 11: loud at ``speak()``, with the status and a bounded body, and no stack trace."""
    tts_client_error, _ = tts_error_classes()
    body = "server error: " + "x" * 2000
    doubles = {
        "status_500": lambda: TtsDouble.status_only(500, body),
        "connect_error": TtsDouble.connect_error,
        "timeout": TtsDouble.timing_out,
    }
    double = doubles[failure]()
    client = build_tts_client(None, transport=double.transport)
    with pytest.raises(tts_client_error) as caught:
        await client.speak("u1", SENTENCE)

    message = str(caught.value)
    assert "Traceback" not in message
    assert len(message) <= 300, f"{len(message)} chars of error text"
    if failure == "status_500":
        assert caught.value.status == 500
        assert "server error: xxxx" in message, "the cause must survive the truncation"
        assert body not in message, "the whole body was carried"
    else:
        assert caught.value.status is None
