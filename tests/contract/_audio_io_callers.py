"""Callers typed against the ``AudioIO`` *protocol* (T-0007 confirmation gap A22-adjacent).

Not collected by pytest (no ``test_`` prefix) and never imported at runtime: constructing
``PipeWireAudioIO`` starts a writer thread. ``test_audio_io_protocol_static.py`` type-checks this
module with ``mypy --strict``; the point is that the two implementations the conductor is handed
-- the real ``PipeWireAudioIO`` and ``tests/fakes/audio.py::FakeAudioIO`` -- satisfy the protocol
*as declared*, not merely as ``isinstance`` sees it. ``runtime_checkable`` only checks that the
member names exist, so a return type that disagrees with the protocol (``played_ms`` returning
``float`` against a protocol that promises ``int``) is invisible to every runtime test.

Annotate with ``AudioIO`` and nothing narrower: the session (docs/02 §5) only ever sees the
protocol.
"""

from __future__ import annotations

from local_astra.conductor.audio_io import PipeWireAudioIO
from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from local_astra.contracts.protocols import AudioIO, TtsUtterance
from tests.conftest import VirtualClock
from tests.fakes.audio import FakeAudioIO
from tests.fakes.audio_device import FakeAudioDevice

#: What the session is constructed with on the box, and in the fast suite. Both are declared as
#: the protocol: this is the assignment that fails if either class drifts from it.
module_io: AudioIO = PipeWireAudioIO(FakeAudioDevice())
fake_io: AudioIO = FakeAudioIO(VirtualClock())


async def speak_one(audio: AudioIO, utterance: TtsUtterance, spoken: str) -> None:
    """The playback half of the executor, written against the protocol only (docs/02 §4)."""
    await audio.play(utterance, source=spoken)
    await audio.pause()
    await audio.resume()
    audio.mark_drain()


async def first_terminal_event(audio: AudioIO) -> PlaybackStopped | PlaybackDrained | None:
    """The session's playback-event pump: the module's own ``played_ms``/``played_text``."""
    async for event in audio.playback_events():
        return event
    return None


async def positions(audio: AudioIO) -> tuple[float, int, bool]:
    """``played_ms`` is read as a duration, ``flush`` as the int the commands contract stores."""
    rendered: float = audio.played_ms()
    dropped: int = await audio.flush()
    return rendered, dropped, audio.is_rendering(0.0)


async def drive_both(utterance: TtsUtterance, spoken: str) -> None:
    for audio in (module_io, fake_io):
        await speak_one(audio, utterance, spoken)
        await first_terminal_event(audio)
        await positions(audio)
