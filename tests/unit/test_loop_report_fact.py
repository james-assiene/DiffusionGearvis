"""T-0178 B1: ``toolcall.loop.REPORT_FACT``, the fact the report step's continuation ends with (docs/tasks/T-0178.md
B1, ruling H-T178-3; docs/02 §4 "Steps").

What the task fixes and these rows pin: the name ``REPORT_FACT`` in ``local_astra/toolcall/loop.py`` (owned files:
"``REPORT_FACT`` and the append helper"), its wording (ruling H-T178-3, the recommended text, written out below by
hand), that it is plain text (the step text is sanitised with ``drop_special_literals`` before it is embedded, and a
special-token literal in assistant text is tokenized as a special id, T-0087 q9), and that it sits outside any
``<|tool_response>`` payload (implementation notes, "REPORT_FACT placement").

What the task does not fix: the append helper's name and signature, and any separator between the last tool
response and the fact. The append-only property (every earlier byte of the continuation is BASE's) is therefore
pinned through the request the session sends (``test_t0178_session.py``,
``test_report_fact_only_on_report_step``), never by calling a helper this module would have to guess the name of.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

from local_astra.toolcall import loop
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, drop_special_literals

#: ruling H-T178-3 (recommended default, James 2026-10-27 "go with the recommendation"), typed from the task file
H_T178_3 = (
    "You have used all your actions for this request. Do not call a tool. In one or two sentences, tell the user "
    "what you did, what you found, and where you stopped."
)


def _fact() -> str:
    fact = getattr(loop, "REPORT_FACT", None)
    assert fact is not None, "T-0178 B1: local_astra/toolcall/loop.py defines REPORT_FACT"
    assert isinstance(fact, str), f"T-0178 B1: REPORT_FACT is a str; got {type(fact).__name__}"
    return fact


def test_report_fact_is_the_ruled_wording():
    """B1: "The wording is ruling H-T178-3." Leading or trailing whitespace is the placement's (a separator), so the
    stripped text is compared."""
    fact = _fact()
    assert fact.strip() == H_T178_3, f"T-0178 B1 (H-T178-3): REPORT_FACT is {fact!r}"


def test_report_fact_is_non_empty_plain_text():
    """B1 / Implementation notes: non-empty, no special-token literal (each would be tokenized as a special id), so
    sanitising it is the identity, and it cannot open or close a tool response."""
    fact = _fact()
    assert fact.strip(), "T-0178 B1: REPORT_FACT is not empty"
    found = [lit for lit in SPECIAL_TOKENS if lit in fact]
    assert found == [], f"T-0178 B1: REPORT_FACT holds special-token literal(s) {found}"
    assert drop_special_literals(fact) == fact, "sanitising REPORT_FACT must not change it"
    for marker in ("<|tool_response>", "<tool_response|>", "response:", "<|tool_call>", "call:"):
        assert marker not in fact, (
            f"T-0178 implementation notes: the fact sits outside any payload; {marker!r} in {fact!r}"
        )


def test_report_fact_names_no_page_and_no_call():
    """Implementation notes ("the fact ... contains no page text") and D16 (no text built from a call): the fact is a
    module constant with no format placeholder a caller could fill."""
    fact = _fact()
    assert "{" not in fact and "}" not in fact and "%s" not in fact, (
        f"T-0178: REPORT_FACT is a fixed sentence, not a template; {fact!r}"
    )
