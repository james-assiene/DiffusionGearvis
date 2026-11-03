"""Contracts: the only place where shapes are defined.

Rules (CLAUDE.md, .claude/rules/tests.md):

- every model is Pydantic v2, ``frozen=True``, ``extra="forbid"``;
- unions are discriminated on a literal ``type`` field so decoding is total;
- changing a wire format means changing the matching fake in ``tests/fakes/`` in the same PR;
- timestamps are integer milliseconds on the session's *audio* clock (``t_ms``), never wall time.
"""

from local_astra.contracts.base import Contract

__all__ = ["Contract"]
