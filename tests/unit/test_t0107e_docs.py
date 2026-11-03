"""T-0107e criterion 8: the rule recorded in docs/02 §5 and docs/03 Q13, as doc anchors.

Every row reads the working-tree docs (the deltas are this task's additions; there is nothing to compare against a
fixed commit); nothing is written. Each check is a pure function of a string, scoped to the block or span the
criterion names, so a keyword elsewhere cannot satisfy it; the adversarial rows run the same function on mutated
copies of the real document. The helpers ``section`` and ``q13_span`` are T-0107a's (``test_t0107a_docs``), imported.

Readings (listed in the test-author's report):

- the docs/02 block is the header's paragraph (up to the next blank line), as T-0107a's delta is one paragraph; "after
  T-0107a's delta" means the T-0107e header starts after the T-0107a header in the document, both inside §5;
- the block's contents are checked by keywords, never by wording: the rule (``user_declines`` or "clear no"), the
  transcript source ("transcript"), the reason (``the user said no`` or ``USER_DECLINED_REASON``), the precedence
  (``asked the user first``/``asked_first`` named with "precede"/"ahead of"/"before"), the log fields
  (``user_declined``, ``verdict_status``, ``asked_first``), and "P1–P9", "R1–R18", "unchanged";
- the Q13 delta is the lines of the Q13 span holding ``T-0107e`` and a ``2026-MM-DD`` date; it must name Q13b, the
  routed reading and the backstop, the re-read, and quote ``62/62`` and a zero count (``0``, ``0/n``) as a token.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.test_t0107a_docs import DELTA02 as DELTA_A
from tests.unit.test_t0107a_docs import q13_span, section

ROOT = Path(__file__).resolve().parents[2]
DOC02 = ROOT / "docs/02-architecture.md"
DOC03 = ROOT / "docs/03-test-plan.md"

DELTA_E = "**Spec delta (T-0107e, H-T107e).**"
ZERO = re.compile(r"(?<![\d/.,])0(?:/[\d,]+)?(?![\d.,]\d|\d)")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def delta_e_block(doc02: str) -> str:
    """Criterion 8, docs/02: exactly one T-0107e header, in §5, after T-0107a's; its paragraph."""
    assert doc02.count(DELTA_E) == 1, (
        f"T-0107e criterion 8: docs/02 holds {DELTA_E!r} exactly once; found {doc02.count(DELTA_E)}"
    )
    s5 = section(doc02, "## 5.")
    assert DELTA_E in s5, "T-0107e criterion 8: the T-0107e delta is in docs/02 §5"
    assert DELTA_A in s5, "premise: T-0107a's delta is in docs/02 §5"
    assert doc02.index(DELTA_E) > doc02.index(DELTA_A), (
        "T-0107e criterion 8: the T-0107e delta comes after T-0107a's"
    )
    start = doc02.index(DELTA_E)
    end = doc02.find("\n\n", start)
    return doc02[start : end if end >= 0 else len(doc02)]


def check_delta_e(doc02: str) -> None:
    block = delta_e_block(doc02)
    low = block.lower()
    need = {
        "the rule": "user_declines" in block or "clear no" in low,
        "the transcript source": "transcript" in low,
        "the reason": "the user said no" in block or "USER_DECLINED_REASON" in block,
        "the precedence over asked-first": (
            "asked the user first" in block or "asked_first" in block
        )
        and any(w in low for w in ("preced", "ahead of", "before")),
        "user_declined": "user_declined" in block,
        "verdict_status": "verdict_status" in block,
        "asked_first": "asked_first" in block,
        "P1–P9 unchanged": "P1–P9" in block,
        "R1–R18 unchanged": "R1–R18" in block,
        "unchanged": "unchanged" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0107e criterion 8: the docs/02 delta lacks {missing}: {block[:800]!r}"


def q13_delta(doc03: str) -> str:
    span = q13_span(doc03)
    lines = [ln for ln in span.splitlines() if "T-0107e" in ln and re.search(r"2026-\d\d-\d\d", ln)]
    assert lines, "T-0107e criterion 8: no dated T-0107e delta line in docs/03 Q13"
    return " ".join(lines)


def check_q13_delta(doc03: str) -> None:
    text = q13_delta(doc03)
    low = text.lower()
    need = {
        "Q13b": "Q13b" in text,
        "routed": "routed" in low,
        "backstop": "backstop" in low,
        "the re-read": "re-read" in low or "reread" in low,
        "62/62": "62/62" in text,
        "a zero count": ZERO.search(text.replace("62/62", "")) is not None,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0107e criterion 8: the Q13 delta lacks {missing}: {text[:800]!r}"


# =====================================================================================================


def test_doc02_delta_header_unique_after_t0107a() -> None:
    """Criterion 8, docs/02 §5: exactly one ``**Spec delta (T-0107e, H-T107e).**``, in §5 and after T-0107a's delta,
    stating the rule, the transcript source, the reason, the precedence, the log fields, and that P1–P9 and R1–R18 are
    unchanged; a duplicated header fails."""
    doc = read(DOC02)
    check_delta_e(doc)
    with pytest.raises(AssertionError):
        check_delta_e(doc + "\n\n" + DELTA_E + " again.\n")


def test_doc03_q13_delta_present() -> None:
    """Criterion 8, docs/03 Q13: a dated T-0107e delta in the Q13 span reporting Q13b's "after no" both routed and
    after the T-0107e backstop, with the criterion-9 re-read: it quotes 62/62 and a zero count."""
    check_q13_delta(read(DOC03))


def test_delta_outside_section5_fails() -> None:
    """Adversarial: the same delta paragraph moved out of §5 (to just after the ``## 6.`` heading, or to the top of the document),
    or placed in §5 before T-0107a's delta, fails the check -- the check is scoped, not a document-wide search. The
    real document passes first (premise)."""
    doc = read(DOC02)
    check_delta_e(doc)
    block = delta_e_block(doc)
    stripped = doc.replace(block, "", 1)
    assert DELTA_E not in stripped, "premise: the block is the only occurrence"
    s6 = stripped.index("\n## 6.")
    s6_line_end = stripped.index("\n", s6 + 1)
    to_s6 = stripped[:s6_line_end] + "\n\n" + block + stripped[s6_line_end:]
    to_top = block + "\n\n" + stripped
    before_a = stripped.replace(DELTA_A, block + "\n\n" + DELTA_A, 1)
    for name, mutated in (
        ("inside §6", to_s6),
        ("at the top", to_top),
        ("before T-0107a's", before_a),
    ):
        assert DELTA_E in mutated, f"premise: the {name} mutant holds the header"
        with pytest.raises(AssertionError):
            check_delta_e(mutated)


def test_q13_delta_outside_q13_fails() -> None:
    """Adversarial for the Q13 half: the dated T-0107e line moved out of the Q13 span (to the end of the document) fails;
    the line with ``62/62`` removed fails."""
    doc = read(DOC03)
    check_q13_delta(doc)
    span = q13_span(doc)
    lines = [ln for ln in span.splitlines() if "T-0107e" in ln and re.search(r"2026-\d\d-\d\d", ln)]
    moved = doc
    for ln in lines:
        moved = moved.replace(ln + "\n", "", 1)
    moved = moved + "\n" + "\n".join(lines) + "\n"
    with pytest.raises(AssertionError):
        check_q13_delta(moved)
    with pytest.raises(AssertionError):
        check_q13_delta(doc.replace("62/62", "sixty-two of sixty-two"))
