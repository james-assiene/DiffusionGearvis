"""Fakes: in-memory implementations of the ``contracts.protocols`` interfaces.

Rules (``.claude/rules/tests.md``): a fake speaks the *same contract* as the real service; a
behaviour added here needs a real-service test in the same PR. T-0001 upgrades these to
replayers of traffic recorded in Phase 0 (SSE from vLLM, msgpack from moshi-server); until
then they are scripted.
"""

from tests.fakes.audio import FakeAudioIO
from tests.fakes.llm import FakeLlm
from tests.fakes.memory import FakeMemory
from tests.fakes.perception import FakePerception
from tests.fakes.stt import FakeStt
from tests.fakes.tts import FakeTts, FakeUtterance

__all__ = [
    "FakeAudioIO",
    "FakeLlm",
    "FakeMemory",
    "FakePerception",
    "FakeStt",
    "FakeTts",
    "FakeUtterance",
]
