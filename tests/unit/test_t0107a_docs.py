"""T-0107a criterion 6: the ruling recorded in docs/01 §3, docs/02 §5 and docs/03 Q13, as doc anchors.

Every row reads repo files (and, for the "append" check, docs/01 at two fixed commits, ``git show
d3765de:docs/01-design.md`` and ``T0107A_TIP``, never the working tree, T-0094); nothing is written.
Each check is a pure function of a string, scoped to the paragraph, block or span the criterion names, so a keyword
elsewhere cannot satisfy it; the adversarial row runs the same function on a mutated copy of the real document.

Readings (listed in the test-author's report):

- "the computer-use paragraph" of docs/01 §3 is the one §3 paragraph containing "computer use" (as
  ``test_speaking_policy_doc.check_section3`` reads it); "append" means that paragraph at d3765de is a prefix of the
  paragraph now and the status sentence starts after it;
- the sentence's five contents are checked by keywords (pay/order, send, delet; click, typ, Enter, key; question; the
  human / model; D16 verbatim, T-0104), never by wording;
- the docs/02 block is the header's paragraph (up to the next blank line);
- the Q13 delta is a line of the Q13 span holding ``T-0107`` and a ``2026-10-DD`` date.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests import _git_range as gr

ROOT = Path(__file__).resolve().parents[2]
DOC01 = ROOT / "docs/01-design.md"
DOC02 = ROOT / "docs/02-architecture.md"
DOC03 = ROOT / "docs/03-test-plan.md"
#: T-0107a's base (literal, tests.md)
BASE_SHA = "d3765decf5a53949885edf401352415ed8749eda"
#: T-0107a's TIP: the branch's last commit touching non-test files (27777bf, the rulings line replacing the
#: predicate; the docs/01 §3 sentence landed in d4047c9). The "appended" check reads docs/01 here, never in the
#: working tree, so a later task's edit of the paragraph cannot redden it (T-0094). **Bump it** if the implementer
#: commits again to a non-test file; the same constant as ``test_t0107a_differential.T0107A_TIP``.
T0107A_TIP = "9fd1df320a1ed21edc07e36561ac6301c714c771"

STATUS = "*Status of D16 moment 2 (T-0107, H-T107, 2026-10-19):*"
DELTA02 = "**Spec delta (T-0107a, H-T107).**"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def section(doc: str, heading_prefix: str) -> str:
    m = re.search(rf"(?m)^{re.escape(heading_prefix)}", doc)
    assert m is not None, f"no heading {heading_prefix!r}"
    nxt = re.compile(r"(?m)^## ").search(doc, m.end())
    return doc[m.start() : nxt.start() if nxt else len(doc)]


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def computer_use_paragraph(doc01: str) -> str:
    paras = [p for p in paragraphs(section(doc01, "## 3. ")) if "computer use" in p]
    assert len(paras) == 1, f"docs/01 §3 has one computer-use paragraph; found {len(paras)}"
    return paras[0]


def check_section3_status(doc01: str, base01: str | None) -> None:
    """Criterion 6, docs/01 §3: one status sentence, appended to the computer-use paragraph, with its contents.
    ``base01`` is docs/01 at d3765de for the "appended" (paragraph-start) check; ``None`` skips that check only (the
    working-tree row, which a later task's legitimate edit of the paragraph's start must not redden)."""
    assert doc01.count(STATUS) == 1, (
        f"T-0107a criterion 6: docs/01 holds {STATUS!r} exactly once; found {doc01.count(STATUS)}"
    )
    para = computer_use_paragraph(doc01)
    assert STATUS in para, (
        "T-0107a criterion 6: the status sentence is not in docs/01 §3's computer-use paragraph"
    )
    if base01 is not None:
        old = computer_use_paragraph(base01).rstrip()
        assert para.startswith(old), (
            "T-0107a criterion 6: the sentence is appended; the paragraph's text at d3765de must be kept as its "
            "prefix"
        )
        assert para.index(STATUS) >= len(old), (
            "the status sentence must come after the paragraph's existing text"
        )
    sentence = para[para.index(STATUS) :]
    low = sentence.lower()
    need = {
        "payment or order": ("pay" in low or "payment" in low) and "order" in low,
        "send on the user's behalf": "send" in low,
        "deletion": "delet" in low,
        "routine clicks, typing, Enter, keys": all(
            w in low for w in ("click", "typ", "enter", "key")
        ),
        "never confirmed": "never" in low and "confirm" in low,
        "conductor enforces only the question rule": "conductor" in low and "question" in low,
        "the human accepts it rests on the model": "human" in low and "model" in low,
        "D16 stays verbatim (T-0104)": "verbatim" in low and "T-0104" in sentence,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0107a criterion 6: the §3 status sentence lacks {missing}: {sentence!r}"


def _show01(rev: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:docs/01-design.md"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def base01() -> str:
    gr.require_commits(BASE_SHA)
    return _show01(BASE_SHA)


def tip01() -> str:
    """docs/01 at ``T0107A_TIP`` (closed range, T-0094), after proving BASE is an ancestor of TIP."""
    gr.require_commits(BASE_SHA, T0107A_TIP)
    assert gr.is_ancestor(BASE_SHA, T0107A_TIP), f"{BASE_SHA} is not an ancestor of {T0107A_TIP}"
    return _show01(T0107A_TIP)


def delta02_block(doc02: str) -> str:
    assert doc02.count(DELTA02) == 1, (
        f"T-0107a criterion 6: docs/02 holds {DELTA02!r} exactly once; found {doc02.count(DELTA02)}"
    )
    s5 = section(doc02, "## 5.")
    assert DELTA02 in s5, "T-0107a criterion 6: the T-0107a delta is in docs/02 §5"
    start = doc02.index(DELTA02)
    end = doc02.find("\n\n", start)
    return doc02[start : end if end >= 0 else len(doc02)]


def q13_span(doc03: str) -> str:
    m = re.search(r"(?m)^(?:[-*] )?\**Q13\b", doc03)
    assert m is not None, "docs/03 has no Q13"
    nxt = re.compile(r"(?m)^((?:[-*] )?\**Q1[4-9]\b|## )").search(doc03, m.end())
    return doc03[m.start() : nxt.start() if nxt else len(doc03)]


# =====================================================================================================


def test_section3_status_sentence_in_computer_use_paragraph() -> None:
    """Criterion 6, docs/01 §3: the ``*Status of D16 moment 2 (T-0107, H-T107, 2026-10-19):*`` sentence is appended
    to the computer-use paragraph and records the five points of the criterion: in full (the "appended" check
    included) on docs/01 at ``T0107A_TIP`` against d3765de; and, the "appended" check aside, on the working tree."""
    check_section3_status(tip01(), base01())
    check_section3_status(read(DOC01), None)


def test_section3_rewritten_paragraph_start_fails() -> None:
    """Adversarial, the "appended" half (power of the fixed-commit check): docs/01 at ``T0107A_TIP`` with the
    computer-use paragraph's first word changed (the sentence prepended to a rewritten paragraph rather than
    appended) fails when checked against d3765de, and passes when the "appended" check is skipped."""
    doc, old = tip01(), base01()
    para = computer_use_paragraph(doc)
    mutated = doc.replace(para, "Rewritten " + para, 1)
    check_section3_status(mutated, None)  # premise: only the paragraph start differs
    with pytest.raises(AssertionError):
        check_section3_status(mutated, old)


def test_section3_sentence_elsewhere_fails() -> None:
    """Adversarial: the same sentence moved out of the computer-use paragraph (into a new §3 paragraph, or into
    §2) fails the check -- the check is scoped to the paragraph, not the document. Run on docs/01 at
    ``T0107A_TIP``."""
    doc, old = tip01(), base01()
    check_section3_status(doc, old)  # premise: the real document passes
    para = computer_use_paragraph(doc)
    sentence = para[para.index(STATUS) :]
    stripped = para[: para.index(STATUS)].rstrip()
    new_para = doc.replace(para, stripped + "\n\n" + sentence, 1)
    in_s2 = doc.replace(para, stripped, 1).replace("## 3. ", sentence + "\n\n## 3. ", 1)
    for name, mutated in (("a new §3 paragraph", new_para), ("§2", in_s2)):
        with pytest.raises(AssertionError):
            check_section3_status(mutated, old)
        assert STATUS in mutated, f"premise: the {name} mutant still holds the sentence"


def test_doc02_delta_header_unique() -> None:
    """Criterion 6, docs/02 §5: exactly one ``**Spec delta (T-0107a, H-T107).**``, in §5, covering the rule, the
    reason constant, both log fields, and that P1–P9 and R1–R18 are unchanged; a duplicated header fails."""
    doc = read(DOC02)
    block = delta02_block(doc)
    need = {
        "the reason constant": "asked the user first" in block or "ASKED_FIRST_REASON" in block,
        "asked_first": "asked_first" in block,
        "verdict_status": "verdict_status" in block,
        "the rule (a question)": "question" in block.lower() or "asks_user" in block,
        "P1–P9 unchanged": "P1–P9" in block,
        "R1–R18 unchanged": "R1–R18" in block,
        "unchanged": "unchanged" in block,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0107a criterion 6: the docs/02 delta lacks {missing}: {block[:600]!r}"
    with pytest.raises(AssertionError):
        delta02_block(doc + "\n\n" + DELTA02 + " again.\n")


def test_doc03_q13_delta_present() -> None:
    """Criterion 6, docs/03 Q13: a dated T-0107 delta in the Q13 span: "irreversible" now reads as high-stakes
    (pay/order, send, delete), and Q13a also reports the reading after the backstop."""
    span = q13_span(read(DOC03))
    lines = [ln for ln in span.splitlines() if "T-0107" in ln and re.search(r"2026-\d\d-\d\d", ln)]
    assert lines, "T-0107a criterion 6: no dated T-0107 delta line in docs/03 Q13"
    text = " ".join(lines)
    low = text.lower()
    need = {
        "high-stakes": "high-stakes" in low,
        "pay/order": "pay" in low and "order" in low,
        "send": "send" in low,
        "delete": "delet" in low,
        "Q13a after the backstop": "Q13a" in text and "backstop" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0107a criterion 6: the Q13 delta lacks {missing}: {text[:600]!r}"
