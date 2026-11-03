"""T-0092b criterion 3 / H11, cd-group half: ``NATIVE_TOOL_INSTRUCTIONS`` is paragraphs 2-3 of the eval prompt.

The conductor may not import ``local_astra.eval`` (the literal exists for that reason), and the fast suite pins
the literal's sha256 (``tests/unit/test_session_prompt.py``). This row compares the literal with
``twopass.NATIVE_SYSTEM_PROMPT`` itself, and the pinned sha with what twopass gives, so neither copy can drift.
"""

from __future__ import annotations

import hashlib

from local_astra.eval.cd import twopass
from tests.unit._tools_switch_support import NATIVE_TOOL_INSTRUCTIONS_SHA256, native_instructions


def _paragraphs_2_and_3() -> str:
    paragraphs = twopass.NATIVE_SYSTEM_PROMPT.split("\n\n")
    assert len(paragraphs) == 3 and paragraphs[0].startswith("You are Astra"), (
        "premise: the H-T84-0 prompt is persona + two paragraphs"
    )
    return "\n\n".join(paragraphs[1:3])


def test_the_literal_equals_paragraphs_2_and_3_of_twopass():
    assert native_instructions() == _paragraphs_2_and_3()


def test_the_fast_suites_pinned_sha_is_twopass_paragraphs_2_and_3():
    assert (
        hashlib.sha256(_paragraphs_2_and_3().encode()).hexdigest()
        == NATIVE_TOOL_INSTRUCTIONS_SHA256
    )
