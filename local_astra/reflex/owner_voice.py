"""Owner-voice veto (docs/06 R5): runs at end-of-turn on ≥ 1.5 s of speech; never gates STT."""

from __future__ import annotations


class OwnerVoiceCheck:
    min_speech_ms = 1_500

    def score(self, pcm_16k_f32: bytes) -> float | None:
        """Similarity to the enrolled owner embedding, or None if too little speech."""
        raise NotImplementedError("T-0007")
