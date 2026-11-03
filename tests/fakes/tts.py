"""Fake ``TtsBackend``: synthesises silence at a fixed words-per-minute so timings are checkable."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from local_astra.contracts.speech import TTS_OUTPUT_FORMAT, WordTiming

BYTES_PER_MS = TTS_OUTPUT_FORMAT.sample_rate * 2 // 1000  # s16le mono


class FakeUtterance:
    def __init__(
        self,
        utterance_id: str,
        text: str,
        ms_per_word: int = 300,
        chunk_ms: int = 80,
        report_word_timings: bool = True,
    ) -> None:
        self.utterance_id = utterance_id
        self.text = text
        self.ms_per_word = ms_per_word
        self.chunk_ms = chunk_ms
        #: Pocket TTS reports no word timings at all (T-0007a criterion 9), so the fake can model
        #: that backend too; the executor estimates in that case (``contracts/protocols.py``).
        self.report_word_timings = report_word_timings
        #: A stream that died after its first chunk (T-0007a criterion 11); the fake never fails.
        self.failed = False
        self.cancelled = False
        self.chunks_sent = 0
        self._words = text.split()
        self.duration_ms = self.ms_per_word * len(self._words)
        self.first_audio_latency_ms_value: float | None = None

    async def audio(self) -> AsyncIterator[bytes]:
        sent = 0
        while sent < self.duration_ms and not self.cancelled:
            n = min(self.chunk_ms, self.duration_ms - sent)
            if self.first_audio_latency_ms_value is None:
                self.first_audio_latency_ms_value = 0.0
            self.chunks_sent += 1
            sent += n
            yield bytes(n * BYTES_PER_MS)

    def words(self) -> Sequence[WordTiming] | None:
        if not self.report_word_timings:
            return None
        return [
            WordTiming(text=w, start_ms=i * self.ms_per_word, end_ms=(i + 1) * self.ms_per_word)
            for i, w in enumerate(self._words)
        ]

    async def cancel(self) -> None:
        self.cancelled = True

    @property
    def first_audio_latency_ms(self) -> float | None:
        return self.first_audio_latency_ms_value


class FakeTts:
    def __init__(self, ms_per_word: int = 300, report_word_timings: bool = True) -> None:
        self.ms_per_word = ms_per_word
        self.report_word_timings = report_word_timings
        self.spoken: list[FakeUtterance] = []

    async def speak(self, utterance_id: str, text: str, voice: str = "default") -> FakeUtterance:
        u = FakeUtterance(
            utterance_id, text, self.ms_per_word, report_word_timings=self.report_word_timings
        )
        self.spoken.append(u)
        return u
