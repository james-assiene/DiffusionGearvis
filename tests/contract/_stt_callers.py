"""Callers typed against the ``SttClient`` *protocol* (T-0032 criterion 7).

Not collected by pytest (no ``test_`` prefix) and never imported at runtime.
``test_stt_protocol_static.py`` type-checks this module with ``mypy --strict``: the two
implementations the conductor is ever handed -- the real ``MoshiSttClient`` and
``tests/fakes/stt.py::FakeStt`` -- must satisfy the protocol **as declared**, not merely as
``isinstance`` sees it. ``runtime_checkable`` checks member *names* only, which is how T-0007
ruling A13's ``played_ms -> int`` / ``-> float`` disagreement stayed invisible to every
conformance test in the suite.

Annotate with ``SttClient`` and nothing narrower: the session (docs/02 §5) only ever sees the
protocol, and ``mark`` is the method T-0032 adds to it (T-0007a ruling A3).
"""

from __future__ import annotations

from local_astra.conductor.stt_client import MoshiSttClient
from local_astra.contracts.protocols import SttClient
from local_astra.contracts.speech import SttEvent
from tests.fakes.stt import FakeStt

#: What the session is constructed with on the box, and in the fast suite.
module_stt: SttClient = MoshiSttClient()
fake_stt: SttClient = FakeStt()


async def feed_one(stt: SttClient, pcm_24k_f32: bytes, t_ms: int) -> None:
    """The capture pump, written against the protocol only (T-0032 criterion 1)."""
    await stt.feed(pcm_24k_f32, t_ms)


async def endpoint(stt: SttClient, t_eot_ms: int) -> int:
    """The producer task that marks the end of turn (criterion 5); the id is an ``int``."""
    marker_id: int = await stt.mark(t_eot_ms)
    return marker_id


async def first_event(stt: SttClient) -> SttEvent | None:
    async for event in stt.events():
        return event
    return None


async def drive_both(pcm_24k_f32: bytes) -> None:
    for stt in (module_stt, fake_stt):
        await feed_one(stt, pcm_24k_f32, 0)
        await endpoint(stt, 1000)
        await first_event(stt)
        await stt.close()
