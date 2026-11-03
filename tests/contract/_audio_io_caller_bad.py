"""A caller that must *not* type-check: the vacuity guard for the mypy harness.

If ``mypy --strict`` reports nothing here, the harness is not actually reading the ``AudioIO``
protocol and the green in ``test_audio_io_protocol_static.py`` means nothing. ``_NotAudioIO`` is
missing ``capture`` and gets ``flush``'s return type wrong -- exactly the two shapes of drift the
static check exists to catch.
"""

from __future__ import annotations

from local_astra.contracts.protocols import AudioIO, TtsUtterance


class _NotAudioIO:
    async def play(self, utterance: TtsUtterance, *, source: str | None = None) -> None: ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def flush(self) -> str:  # the contract says milliseconds, not text
        return ""


not_audio: AudioIO = _NotAudioIO()
