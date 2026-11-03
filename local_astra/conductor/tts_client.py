"""Pocket TTS (CPU) client: sentence in, streaming 24 kHz audio out (docs/06 R2). Task T-0007a.

Pocket TTS takes a *whole sentence* (it is not streaming-text-in), so the speech gate hands over
committed sentences and this module changes none of them: ``speak`` posts the text it was given,
verbatim, as the form field ``text`` of ``POST /tts`` — the only route the service has besides
``/`` and ``/health``. ``first_audio_latency_ms`` is *measured* from ``speak`` to the first chunk
on the injected clock, never assumed.

Two things it refuses to invent:

* **word timings.** ``words()`` is always ``None``. Pocket TTS reports none, and a character-rate
  estimate would be indistinguishable from a measurement to the playback side, which is how
  ``played_text`` ends up claiming words nobody heard (docs/02 §4).
* **the end of a stream.** The wav is parsed by walking its RIFF chunks, not by slicing at
  offset 44, and a failure after the first chunk raises rather than stopping quietly: a truncated
  stream that looks complete is the same lie by another route.

``TtsClientError`` is deliberately *not* named ``TtsError`` — that is ``contracts.events.TtsError``,
the event T-0007's session emits when it catches this.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Sequence

import httpx

from local_astra.config import LOOPBACK_PREFIXES, Services
from local_astra.contracts.protocols import TtsUtterance
from local_astra.contracts.speech import TTS_OUTPUT_FORMAT, WordTiming

#: Pocket TTS 3.1.0's only synthesis route, and its only parameter.
TTS_PATH = "/tts"
TEXT_FIELD = "text"

#: The one voice the service has. Anything else is a lie by silence, so it raises.
DEFAULT_VOICE = "default"

#: Emitted chunk size, in milliseconds of 24 kHz mono s16le.
DEFAULT_CHUNK_MS = 80

#: s16le.
SAMPLE_WIDTH = 2

#: ``data`` sizes that mean "read to EOF": zero, and the canonical streaming placeholder.
STREAMING_DATA_SIZE = 0xFFFFFFFF

#: What Pocket TTS 3.1.0 actually streams (measured 2026-09-14 against the live service on
#: 127.0.0.1:8081, CPU): ``RIFF`` size 2000000036 (0x77359424) and ``data`` size 2000000000
#: (0x77359400) — 41 666 s of 24 kHz s16 mono — in front of a 117 120-byte body. It is neither
#: ``0`` nor ``0xFFFFFFFF``, so it needs naming (ruling A10).
POCKET_TTS_PLACEHOLDER_DATA_SIZE = 2_000_000_000

#: The longest single utterance worth believing a header about. Beyond it a declared size is a
#: placeholder somebody wrote once, not a length: the short-body rule (ruling A6) would otherwise
#: raise on every real sentence. Inside it the declared size stays authoritative.
MAX_UTTERANCE_S = 600
MAX_PLAUSIBLE_DATA_BYTES = (
    MAX_UTTERANCE_S * TTS_OUTPUT_FORMAT.sample_rate * TTS_OUTPUT_FORMAT.channels * 2
)

#: How much of a failing response body travels with the exception. A body is a clue, not a log.
MAX_BODY_CHARS = 200
_CAUSE_CHARS = 120

#: Rendering a long sentence on two pinned CPU cores is slow; a short read timeout would turn a
#: working service into a spurious failure. Connect stays short: nothing is listening or it is.
REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

_PCM_FORMAT = 1  # WAVE_FORMAT_PCM


class TtsClientError(Exception):
    """A failure of the TTS service or its stream. ``status`` is the HTTP status when there is one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class TtsFormatError(TtsClientError):
    """The body is not 24 kHz mono s16le — playing it would be the wrong pitch or the wrong rate."""


def _brief(exc: BaseException, limit: int = _CAUSE_CHARS) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def _is_placeholder_size(size: int) -> bool:
    """Does this declared ``data`` size mean "read to EOF" rather than a length? (ruling A10)"""
    return (
        size in (0, STREAMING_DATA_SIZE, POCKET_TTS_PLACEHOLDER_DATA_SIZE)
        or size > MAX_PLAUSIBLE_DATA_BYTES
    )


def _chunk_bytes(chunk_ms: int) -> int:
    rate = TTS_OUTPUT_FORMAT.sample_rate * TTS_OUTPUT_FORMAT.channels * SAMPLE_WIDTH
    return chunk_ms * rate // 1000


class _Reader:
    """A byte cursor over the response stream: nothing is read that the parser did not ask for."""

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self._buffer = bytearray()
        self._eof = False

    async def _pull(self) -> bool:
        if self._eof:
            return False
        try:
            block = await self._source.__anext__()
        except StopAsyncIteration:
            self._eof = True
            return False
        self._buffer += block
        return True

    async def exact(self, n: int) -> bytes | None:
        """Exactly ``n`` bytes, or ``None`` if the stream ends first."""
        while len(self._buffer) < n:
            if not await self._pull():
                return None
        out = bytes(self._buffer[:n])
        del self._buffer[:n]
        return out

    async def at_most(self, n: int) -> bytes:
        """Up to ``n`` bytes; empty at EOF."""
        while not self._buffer:
            if not await self._pull():
                return b""
        out = bytes(self._buffer[:n])
        del self._buffer[: len(out)]
        return out


class PocketTtsUtterance:
    """One ``POST /tts`` response, parsed and chunked as it arrives."""

    def __init__(
        self,
        utterance_id: str,
        response: httpx.Response,
        *,
        clock: Callable[[], float],
        started_s: float,
        chunk_bytes: int,
    ) -> None:
        self.utterance_id = utterance_id
        self.failed = False
        self._response = response
        self._clock = clock
        self._started_s = started_s
        self._chunk_bytes = chunk_bytes
        self._latency_ms: float | None = None
        self._consumed = False
        self._cancelled = False

    @property
    def first_audio_latency_ms(self) -> float | None:
        return self._latency_ms

    def words(self) -> Sequence[WordTiming] | None:
        """Always ``None``: Pocket TTS reports no word timings and this client invents none."""
        return None

    def audio(self) -> AsyncIterator[bytes]:
        return self._audio()

    def _is_cancelled(self) -> bool:
        """A method, not the attribute: ``cancel()`` sets it from another task, mid-``await``."""
        return self._cancelled

    async def cancel(self) -> None:
        """Idempotent, well inside 100 ms: close the response and stop yielding, silently."""
        if self._cancelled:
            return
        self._cancelled = True
        await self._response.aclose()

    # --- the stream -------------------------------------------------------------------------

    async def _audio(self) -> AsyncIterator[bytes]:
        if self._cancelled:
            return
        if self._consumed:
            raise TtsClientError(
                f"audio() for {self.utterance_id!r} has already been consumed; "
                "the body is a stream, not a buffer"
            )
        self._consumed = True
        reader = _Reader(self._response.aiter_bytes())
        try:
            async for chunk in self._pcm(reader):
                if self._is_cancelled():
                    return
                if self._latency_ms is None:
                    self._latency_ms = (self._clock() - self._started_s) * 1000.0
                yield chunk
        except TtsClientError:
            self.failed = True
            raise
        except Exception as exc:
            if self._is_cancelled():
                return  # the read that ``cancel()`` woke up: a cancel is not a failure
            self.failed = True
            raise TtsClientError(f"pocket-tts stream failed ({_brief(exc)})") from exc
        finally:
            await self._response.aclose()

    async def _pcm(self, reader: _Reader) -> AsyncIterator[bytes]:
        remaining = await self._parse_header(reader)
        frame = SAMPLE_WIDTH * TTS_OUTPUT_FORMAT.channels
        pending = bytearray()
        while remaining is None or remaining > 0:
            want = self._chunk_bytes - len(pending)
            if remaining is not None:
                want = min(want, remaining)
            block = await reader.at_most(want)
            if not block:
                break
            if remaining is not None:
                remaining -= len(block)
            pending += block
            if len(pending) >= self._chunk_bytes:
                yield bytes(pending[: self._chunk_bytes])
                del pending[: self._chunk_bytes]
        if remaining:
            # The declared ``data`` size is authoritative (the placeholders were resolved to
            # ``None`` above), so a body that stops short is a truncated stream, not a short
            # sentence — and a truncated stream that looks complete makes ``played_text`` lie.
            raise TtsClientError(
                f"pocket-tts stream ended {remaining} bytes short of the "
                "data size its header declared"
            )
        if len(pending) % frame:
            raise TtsClientError(
                f"pocket-tts stream ended {len(pending) % frame} bytes into a frame: "
                "the audio is truncated"
            )
        if pending:
            yield bytes(pending)

    async def _parse_header(self, reader: _Reader) -> int | None:
        """Walk the RIFF chunks to ``data``. Returns its length, or ``None`` for "read to EOF"."""
        riff = await reader.exact(12)
        if riff is None or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise TtsFormatError("pocket-tts did not answer with a RIFF/WAVE body")
        seen_format = False
        while True:
            head = await reader.exact(8)
            if head is None:
                raise TtsFormatError("pocket-tts body has no data chunk")
            chunk_id = head[:4]
            size = int.from_bytes(head[4:8], "little")
            if chunk_id == b"data":
                if not seen_format:
                    raise TtsFormatError("pocket-tts body has a data chunk before its fmt chunk")
                return None if _is_placeholder_size(size) else size
            payload = await reader.exact(size + (size % 2))  # chunks are word-aligned
            if payload is None:
                raise TtsFormatError(
                    f"pocket-tts body ends inside its {chunk_id.decode('ascii', 'replace')} chunk"
                )
            if chunk_id == b"fmt ":
                self._check_format(payload[:size])
                seen_format = True

    @staticmethod
    def _check_format(payload: bytes) -> None:
        if len(payload) < 16:
            raise TtsFormatError("pocket-tts fmt chunk is too short to describe the audio")
        audio_format = int.from_bytes(payload[0:2], "little")
        channels = int.from_bytes(payload[2:4], "little")
        sample_rate = int.from_bytes(payload[4:8], "little")
        bits = int.from_bytes(payload[14:16], "little")
        expected_bits = SAMPLE_WIDTH * 8
        if (audio_format, channels, sample_rate, bits) != (
            _PCM_FORMAT,
            TTS_OUTPUT_FORMAT.channels,
            TTS_OUTPUT_FORMAT.sample_rate,
            expected_bits,
        ):
            raise TtsFormatError(
                f"pocket-tts returned format {audio_format} {sample_rate} Hz {channels} ch "
                f"{bits}-bit, not {TTS_OUTPUT_FORMAT.sample_rate} Hz "
                f"{TTS_OUTPUT_FORMAT.channels} ch {TTS_OUTPUT_FORMAT.sample_format}"
            )


class PocketTtsClient:
    """``transport`` is an ``httpx.AsyncBaseTransport`` (tests inject ``httpx.MockTransport``)."""

    def __init__(
        self,
        services: Services | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
        chunk_ms: int = DEFAULT_CHUNK_MS,
    ) -> None:
        resolved = services if services is not None else Services()
        if not resolved.tts.startswith(LOOPBACK_PREFIXES):
            raise ValueError(f"non-loopback endpoint {resolved.tts!r} (docs/02 §9)")
        self.url = resolved.tts
        self.transport = transport
        self.chunk_ms = chunk_ms
        self._clock = clock
        self._chunk_bytes = _chunk_bytes(chunk_ms)
        self._client: httpx.AsyncClient | None = None

    async def speak(self, utterance_id: str, text: str, voice: str = DEFAULT_VOICE) -> TtsUtterance:
        """One POST, the sentence verbatim, issued here so the server renders before ``audio()``."""
        if voice != DEFAULT_VOICE:
            raise ValueError(
                f"pocket-tts has no voice parameter; {voice!r} would silently be the default"
            )
        client = self._http()
        request = client.build_request("POST", f"{self.url}{TTS_PATH}", data={TEXT_FIELD: text})
        started_s = self._clock()
        try:
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise TtsClientError(f"pocket-tts POST {TTS_PATH} failed ({_brief(exc)})") from exc
        if response.status_code != 200:
            try:
                body = (await response.aread()).decode("utf-8", "replace")
            finally:
                await response.aclose()
            raise TtsClientError(
                f"pocket-tts POST {TTS_PATH} answered {response.status_code}: "
                f"{body[:MAX_BODY_CHARS]}",
                status=response.status_code,
            )
        return PocketTtsUtterance(
            utterance_id,
            response,
            clock=self._clock,
            started_s=started_s,
            chunk_bytes=self._chunk_bytes,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self.transport, timeout=REQUEST_TIMEOUT)
        return self._client
