"""Recover a tool call written at the tail of an unclosed thought (T-0135 A1).

``recover(thought)`` takes the text of a request's final open thought block (its trailing run of open ``thought``
commits, concatenated in order) and returns ``thought[k:]``, where ``k`` is the index of the **last**
``<|tool_call>``, iff ``thought[k:].rstrip()`` ends with ``<tool_call|>``; otherwise ``None`` (no tag, or any
non-whitespace text after the final ``<tool_call|>``).

It does not parse or validate the call: judging is ``live.classify``'s job. Stdlib only, linear in ``len(thought)``
(one ``rfind``, one ``rstrip``, one ``endswith``; no regular expression), and it never raises.
"""

from __future__ import annotations

TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"


def recover(thought: str) -> str | None:
    """The tail of ``thought`` from its last ``<|tool_call>``, iff it ends (after whitespace) in ``<tool_call|>``."""
    k = thought.rfind(TOOL_CALL_OPEN)
    if k < 0:
        return None
    tail = thought[k:]
    return tail if tail.rstrip().endswith(TOOL_CALL_CLOSE) else None
