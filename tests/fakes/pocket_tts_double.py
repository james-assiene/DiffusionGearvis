"""A scripted Pocket TTS 3.1.0 for T-0007a: ``httpx.MockTransport`` handlers and a wav builder.

``POST /tts`` (form field ``text``) → a RIFF body, and nothing else: the openapi of the running
service exposes ``/``, ``/health``, ``/tts`` only (``tests/fixtures/audio/MANIFEST.md``). The
variants here are the ones criteria 10 and 11 name:

* :func:`make_wav` — a wav built **by hand with ``struct``**, so a test never checks the client
  against another copy of the client's own parser: canonical 44-byte header, an arbitrary chunk
  (``LIST``) before ``data``, a bogus/streaming ``data`` size, any rate/channel/width.
* :class:`TtsDouble` — records every request (and decodes the form with ``urllib.parse``), and
  serves: a whole body, a trickling stream, a stream that fails after the first chunk, a stream
  that hangs after a partial header, 500, a connect error, a timeout.
* :class:`StepClock` — the injected monotonic-seconds source ``first_audio_latency_ms`` is
  measured against, moved by hand so no test waits on wall-clock time.

The committed rendering ``tests/fixtures/audio/pocket_tts_sentence.wav`` is the *real* body; the
service is not deterministic (MANIFEST.md), so it is a recording, and nothing here regenerates it.
"""

from __future__ import annotations

import asyncio
import struct
import urllib.parse
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx

#: 24 kHz mono s16le (``contracts.speech.TTS_OUTPUT_FORMAT``).
RATE = 24_000
BYTES_PER_MS = RATE * 2 // 1000  # 48
CANONICAL_HEADER_BYTES = 44

#: The streaming placeholder some writers put in ``data`` when the length is not known yet.
STREAMING_DATA_SIZE = 0xFFFFFFFF

TTS_CLIENT_SURFACE = ("speak", "url")
TTS_UTTERANCE_SURFACE = ("audio", "words", "cancel", "first_audio_latency_ms", "failed")


# --- wav bytes, built by hand --------------------------------------------------------------


def make_wav(
    pcm: bytes,
    *,
    sample_rate: int = RATE,
    channels: int = 1,
    bits: int = 16,
    data_size: int | None = None,
    extra_chunks: Sequence[tuple[bytes, bytes]] = (),
    riff_size: int | None = None,
) -> bytes:
    """A RIFF/WAVE file. ``data_size`` overrides the ``data`` chunk length (0 / 0xFFFFFFFF).

    ``extra_chunks`` are written between ``fmt `` and ``data`` — that is what kills an
    implementation that slices at offset 44.
    """
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,  # WAVE_FORMAT_PCM
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    middle = b"".join(
        cid + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")
        for cid, payload in extra_chunks
    )
    data = b"data" + struct.pack("<I", len(pcm) if data_size is None else data_size) + pcm
    body = b"WAVE" + fmt + middle + data
    size = len(body) if riff_size is None else riff_size
    return b"RIFF" + struct.pack("<I", size) + body


def sine_pcm(ms: int, *, freq: float = 220.0, amplitude: float = 0.4) -> bytes:
    """s16le mono 24 kHz test audio. Deterministic and non-silent (so "peak > 0" is meaningful)."""
    import math

    n = ms * RATE // 1000
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)),
    )


def ramp_pcm(n_samples: int) -> bytes:
    """s16le samples ``0, 1, 2, …`` — every byte position is identifiable in a chunk."""
    return struct.pack(f"<{n_samples}h", *(((i * 37) % 30000) - 15000 for i in range(n_samples)))


def canonical_wav(pcm: bytes) -> bytes:
    return make_wav(pcm)


def wav_with_list_chunk(pcm: bytes, *, data_size: int | None = STREAMING_DATA_SIZE) -> bytes:
    """The adversarial file: a ``LIST`` chunk before ``data`` *and* a bogus ``data`` size."""
    return make_wav(
        pcm,
        data_size=data_size,
        extra_chunks=((b"LIST", b"INFOISFT" + b"pocket-tts 3.1.0\x00"),),
    )


def wav_44100_stereo(pcm: bytes) -> bytes:
    return make_wav(pcm, sample_rate=44100, channels=2)


# --- the transport ---------------------------------------------------------------------------


class RecordingStream(httpx.AsyncByteStream):
    """A response body the test drives chunk by chunk; records delivery and ``aclose``."""

    def __init__(
        self,
        chunks: Sequence[bytes],
        *,
        delay_s: float = 0.0,
        tail: str | BaseException = "stop",
        repeat_last: bool = False,
    ) -> None:
        self.chunks = list(chunks)
        self.delay_s = delay_s
        self.tail = tail
        self.repeat_last = repeat_last
        self.delivered: list[bytes] = []
        self.closed = False
        self.exhausted = False
        self._closed_event = asyncio.Event()

    async def __aiter__(self) -> Any:
        for i, chunk in enumerate(self.chunks):
            if i:
                await asyncio.sleep(self.delay_s)
            self.delivered.append(chunk)
            yield chunk
        if self.repeat_last and self.chunks:
            while True:
                await asyncio.sleep(self.delay_s or 0.005)
                self.delivered.append(self.chunks[-1])
                yield self.chunks[-1]
        if isinstance(self.tail, BaseException):
            raise self.tail
        if self.tail == "hang":
            # A read that is parked when the caller closes the response: a real transport wakes it
            # with a read error rather than leaving it parked for ever, and ``cancel()`` (≤ 100 ms)
            # depends on that. Modelled here so the double cannot be *easier* than the socket.
            await self._closed_event.wait()
            raise httpx.ReadError("the response was closed mid-read")
        self.exhausted = True

    async def aclose(self) -> None:
        self.closed = True
        self._closed_event.set()


Responder = Callable[[httpx.Request], httpx.Response]


class TtsDouble:
    """Records what the client posted; serves one of the scripted bodies."""

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.requests: list[httpx.Request] = []
        self.bodies: list[bytes] = []
        self.streams: list[RecordingStream] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.read())
        self.requests.append(request)
        return self._responder(request)

    # --- what the client sent -----------------------------------------------------------

    def forms(self) -> list[dict[str, list[str]]]:
        """The posted form, decoded with ``urllib.parse`` — not with the client's encoder."""
        return [
            urllib.parse.parse_qs(body.decode("utf-8"), keep_blank_values=True)
            for body in self.bodies
        ]

    def texts(self) -> list[str]:
        out: list[str] = []
        for form in self.forms():
            values = form.get("text", [])
            out.append(values[0] if values else "<no text field>")
        return out

    # --- scripts -------------------------------------------------------------------------

    @classmethod
    def serving(cls, body: bytes, *, status: int = 200) -> TtsDouble:
        def responder(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, content=body, headers={"content-type": "audio/x-wav"})

        return cls(responder)

    @classmethod
    def streaming(
        cls,
        chunks: Sequence[bytes],
        *,
        delay_s: float = 0.0,
        tail: str | BaseException = "stop",
        repeat_last: bool = False,
    ) -> TtsDouble:
        double: TtsDouble

        def responder(request: httpx.Request) -> httpx.Response:
            stream = RecordingStream(chunks, delay_s=delay_s, tail=tail, repeat_last=repeat_last)
            double.streams.append(stream)
            return httpx.Response(200, stream=stream, headers={"content-type": "audio/x-wav"})

        double = cls(responder)
        return double

    @classmethod
    def status_only(cls, status: int, body: str) -> TtsDouble:
        def responder(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, text=body)

        return cls(responder)

    @classmethod
    def raising(cls, exc: BaseException) -> TtsDouble:
        def responder(request: httpx.Request) -> httpx.Response:
            raise exc

        return cls(responder)

    @classmethod
    def connect_error(cls) -> TtsDouble:
        return cls(_raiser(httpx.ConnectError))

    @classmethod
    def timing_out(cls) -> TtsDouble:
        return cls(_raiser(httpx.ReadTimeout))


def _raiser(exc_type: type[httpx.HTTPError]) -> Responder:
    def responder(request: httpx.Request) -> httpx.Response:
        raise exc_type("scripted failure", request=request)

    return responder


def chunked(body: bytes, size: int) -> list[bytes]:
    return [body[i : i + size] for i in range(0, len(body), size)]


# --- the injected clock ------------------------------------------------------------------------


class StepClock:
    """Monotonic seconds, moved by the test. ``first_audio_latency_ms`` is measured, not slept."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


# --- the client under test ---------------------------------------------------------------------


def tts_error_classes() -> tuple[type[BaseException], type[BaseException]]:
    """``(TtsClientError, TtsFormatError)`` or an assertion naming T-0007a's Implementation notes."""
    from local_astra.conductor import tts_client

    missing = [n for n in ("TtsClientError", "TtsFormatError") if not hasattr(tts_client, n)]
    assert not missing, (
        f"T-0007a: local_astra/conductor/tts_client.py must define {missing} "
        "(deliberately not named TtsError — that is the session's event class)"
    )
    return tts_client.TtsClientError, tts_client.TtsFormatError


def build_tts_client(services: Any = None, **kwargs: Any) -> Any:
    """``PocketTtsClient(services=None, *, transport=…, clock=…, chunk_ms=…)``, or an assertion."""
    from local_astra.conductor.tts_client import PocketTtsClient

    try:
        client = PocketTtsClient(services, **kwargs)
    except (TypeError, AttributeError) as exc:  # the stub still takes ``url: str``
        raise AssertionError(
            "T-0007a: PocketTtsClient(services=None, *, transport=…) is not implemented "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    missing = [name for name in TTS_CLIENT_SURFACE if not hasattr(client, name)]
    assert not missing, f"T-0007a: PocketTtsClient is missing {missing} (docs/tasks/T-0007a.md)"
    return client


def require_utterance_surface(utterance: Any) -> Any:
    missing = [name for name in TTS_UTTERANCE_SURFACE if not hasattr(utterance, name)]
    assert not missing, f"T-0007a: the utterance is missing {missing} (contracts/protocols.py)"
    return utterance


async def collect(aiterator: Any, *, limit: int = 10_000) -> list[bytes]:
    out: list[bytes] = []
    async for chunk in aiterator:
        out.append(chunk)
        if len(out) >= limit:
            break
    return out


def frames_of(pcm: bytes) -> Iterator[int]:
    """s16le samples, decoded with ``struct`` (the independent oracle)."""
    return iter(struct.unpack(f"<{len(pcm) // 2}h", pcm[: len(pcm) // 2 * 2]))
