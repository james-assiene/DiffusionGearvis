"""T-0131 (owned files, docs/02): one §5 "Spec delta (T-0131)" paragraph, covering the optional body field, ReGround
in the execute stage, the turn-line keys and the live guidance (docs/tasks/T-0131.md).

Keywords, never wording (T-0107a's ``section``/``paragraphs`` helpers, imported). "One paragraph only": docs/02 at
the read revision, with that paragraph removed, has BASE f30365d's paragraphs exactly.

**Which revision is read.** The task says "read with ``git show`` at a fixed commit after the merge". **Pre-merge
conversion (2026-10-22, docs/tasks/T-0131.md Rulings):** ``DOC_REV`` is fixed at 047f82b, the last implementation
commit, as T-0111g's docs rows did; docs/02 is read with ``git show 047f82b:docs/02-architecture.md``, never from the
working tree, so main's later edits to docs/02 cannot redden the row. The merge may close it to ``M^2``. BASE is
always read with ``git show`` at its literal.

Before the conversion the row was ``pending(task="T-0131")``: at BASE there is no T-0131 paragraph, so it failed on
a named ``AssertionError``.

Regime: reads two git blobs (BASE and DOC_REV); nothing is written.
"""

from __future__ import annotations

from tests import _git_range as gr
from tests.unit import test_t0107a_docs as t7d

BASE_SHA = "f30365d9fae12dbf4f42b7e22d5a6e099b7303c4"
#: T-0131's last implementation commit, 047f82b, fixed at the pre-merge conversion (see the docstring)
DOC_REV = "047f82bb283cae10c7a019286e331f4687bb0e32"
HEADER = "**Spec delta (T-0131"


def _base_doc02() -> str:
    gr.require_commits(BASE_SHA)
    proc = gr._git("show", f"{BASE_SHA}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _doc02() -> str:
    gr.require_commits(DOC_REV)
    proc = gr._git("show", f"{DOC_REV}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_present() -> None:
    """The T-0131 delta is one §5 paragraph naming the body field (``mm_processor_kwargs``/``max_soft_tokens``,
    ``visual_tokens``), ReGround in the execute stage, the turn-line keys (``visual_tokens``, ``reground``) and the
    live guidance (``LIVE_TOOL_GUIDANCE``); docs/02 otherwise equals BASE's, paragraph for paragraph."""
    doc = _doc02()
    s5 = t7d.section(doc, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, (
        f"T-0131: docs/02 §5 holds one paragraph starting {HEADER!r}; found {len(paras)}"
    )
    para = paras[0]
    low = para.lower()
    need = {
        "mm_processor_kwargs": "mm_processor_kwargs" in para,
        "max_soft_tokens": "max_soft_tokens" in para,
        "visual_tokens": "visual_tokens" in para,
        "ReGround": "reground" in low,
        "execute stage": "execute" in low,
        "turn line": "turn line" in low or "turn-line" in low,
        "live guidance": "LIVE_TOOL_GUIDANCE" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0131: the docs/02 delta does not record {missing}"
    rest = [p for p in t7d.paragraphs(doc) if p is not para and not p.lstrip().startswith(HEADER)]
    assert rest == t7d.paragraphs(_base_doc02()), (
        "T-0131: docs/02 changes only by the one delta paragraph"
    )
