"""Callers typed against the *protocol* (T-0017 criteria 4 and 5).

Not collected by pytest (no ``test_`` prefix). ``test_llm_backend_xargs.py`` does two things
with this module: it runs ``tap_commits`` against both the fake and the real client, and it
type-checks the module with ``mypy --strict`` — the criterion-4 assertion. Keep every call here
annotated with ``LlmBackend`` and nothing narrower: the point is that the *protocol* carries
``xargs``, not that the two concrete classes happen to.
"""

from __future__ import annotations

from local_astra.contracts.llm import Commit, LlmOptions, Prompt
from local_astra.contracts.protocols import LlmBackend

#: what T-0011's per-step tap will pass (docs/07 §5); string-valued, per criterion 5
TAP_XARGS: dict[str, str] = {"astra_tap": "1"}


async def tap_commits(backend: LlmBackend, prompt: Prompt, request_id: str) -> list[Commit]:
    """Drain ``generate`` with plugin arguments attached. Must typecheck under ``--strict``."""
    return [
        c async for c in backend.generate(prompt, request_id, LlmOptions(top_k=40), xargs=TAP_XARGS)
    ]


async def no_xargs_commits(backend: LlmBackend, prompt: Prompt, request_id: str) -> list[Commit]:
    """The pre-T-0017 call shape: positional, no ``xargs``. Criterion 1's "unaffected"."""
    return [c async for c in backend.generate(prompt, request_id, LlmOptions(top_k=40))]
