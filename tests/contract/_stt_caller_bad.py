"""Vacuity guard for ``test_stt_protocol_static.py``: a client that drifts on ``mark`` only.

Every other member is right, so the *only* thing that can reject this module is the method
T-0032 adds to ``SttClient``. Before criterion 7 lands, ``mypy`` accepts it -- which is exactly
what makes the guard meaningful: a protocol that never declared ``mark`` would let a client whose
marker id is a string through, and the session would compare it to an int for ever.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from local_astra.contracts.protocols import SttClient
from local_astra.contracts.speech import SttEvent


class DriftingStt:
    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        return None

    async def mark(self, t_eot_ms: int) -> str:  # the drift: the protocol says ``int``
        return "marker-1"

    def events(self) -> AsyncIterator[SttEvent]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


drifting: SttClient = DriftingStt()
