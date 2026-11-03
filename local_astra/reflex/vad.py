"""Silero VAD v6 wrapper (CPU thread). Task T-0007. Emits onset/offset with audio timestamps."""

from __future__ import annotations


class SileroVad:
    def __init__(
        self, threshold: float = 0.5, min_speech_ms: int = 80, min_silence_ms: int = 200
    ) -> None:
        self.threshold, self.min_speech_ms, self.min_silence_ms = (
            threshold,
            min_speech_ms,
            min_silence_ms,
        )

    def process(self, pcm_16k_f32: bytes, t_ms: int) -> list[tuple[str, int]]:
        """Return ``[("onset", t), ("offset", t)]`` events for this block."""
        raise NotImplementedError("T-0007")
