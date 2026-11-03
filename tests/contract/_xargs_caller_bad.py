"""Deliberately ill-typed callers: ``xargs`` values that are not strings (T-0017 criterion 5).

Not collected, never imported at runtime — ``test_llm_backend_xargs.py`` only runs ``mypy`` over
it and asserts *which* error each line produces. Every line below must be rejected by the type
checker, and rejected for the *value type*, not because ``xargs`` is unknown to the protocol.
"""

from __future__ import annotations

from local_astra.contracts.llm import Prompt
from local_astra.contracts.protocols import LlmBackend

INT_VALUED: dict[str, int] = {"n": 1}


async def int_valued_literal(backend: LlmBackend, prompt: Prompt, request_id: str) -> None:
    async for _ in backend.generate(prompt, request_id, xargs={"n": 1}):  # dict-item
        pass


async def int_valued_variable(backend: LlmBackend, prompt: Prompt, request_id: str) -> None:
    async for _ in backend.generate(prompt, request_id, xargs=INT_VALUED):  # arg-type
        pass
