"""T-0111g B16: the docs/02 §5 "Spec delta (T-0111g)" paragraph and the docs/01 §3 status sentence.

Each check is a pure function of a string, scoped to the paragraph the criterion names (T-0107a's helpers, imported).
Keywords, never wording: the delta must record ``live``, the start-page origin, ``LIVE_FRAME_TIMEOUT_MS`` (the
implementation notes ask the value to be justified there), and say that it supersedes the "``live`` stays refused"
clauses of the T-0104, T-0107a and T-0107e deltas, which stay verbatim (compared with BASE 9bebf69, a literal).

T-0111g pre-merge conversion (.claude/rules/tests.md "pin both ends", T-0094): the two rows that compare a BASE blob
read the task's own docs at ``TIP_IMPL`` = e49b868 (the last implementation commit on ``T-0111g-live-wiring``), a
second literal, never the working tree. Main's T-0128 merge (fbb97b5) already edited docs/02 (§2.1), and once the
merge bumps ``CLOSED_AS_OF`` past 9bebf69 a BASE blob beside a live read of the same file is flagged by
``test_guards_use_fixed_refs``. Every commit after e49b868 on the branch is tests or docs-file bookkeeping only; after
the merge the orchestrator may close ``TIP_IMPL`` to the merge's ``M^2``. ``test_pinned_sentences_still_verbatim``
reads the working tree on purpose (no base: its pinned sentences are content, meant to hold on main).

Readings (reported): "records B5-B15" is checked by anchors for the behaviour a reader most needs (the swap, the
fresh frame, freshness, ``failed``, the closed window, assists dropped, egress), each by one keyword; the docs/01
sentence by "own"/"sandbox" browser, "loopback", "T-0111g", "H-T111c-E" and "desktop".

``test_pinned_sentences_still_verbatim`` is green at BASE and carries no marker; the other two are pending.

Regime: reads repo files and one blob at BASE; nothing is written.
"""

from __future__ import annotations

from pathlib import Path

from tests import _git_range as gr
from tests.unit import test_speaking_policy_doc as spd
from tests.unit import test_t0107a_docs as t7d

ROOT = Path(__file__).resolve().parents[2]
DOC01 = ROOT / "docs/01-design.md"
DOC02 = ROOT / "docs/02-architecture.md"
BASE_SHA = "9bebf69d9d3ac9aacb5a41bf8bd2f82569afc81a"
#: T-0111g's last implementation commit (pre-merge conversion): the B16 rows read the task's docs here, not live.
TIP_IMPL = "e49b868c583f6597d34c5c2c4584362327824dbe"
HEADER = "**Spec delta (T-0111g"
OLDER = ("**Spec delta (T-0104", "**Spec delta (T-0107a", "**Spec delta (T-0107e")


def delta(doc02: str, header: str) -> str:
    s5 = t7d.section(doc02, "## 5. ")
    paras = [p for p in t7d.paragraphs(s5) if p.lstrip().startswith(header)]
    assert len(paras) == 1, (
        f"docs/02 §5 holds one paragraph starting {header!r}; found {len(paras)}"
    )
    return paras[0]


def _blob(rev: str, path: str) -> str:
    """``path`` at the fixed commit ``rev`` (skips, never passes, when ``rev`` is absent)."""
    gr.require_commits(rev)
    proc = gr._git("show", f"{rev}:{path}")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_present_and_supersedes() -> None:
    """B16: the T-0111g delta records live, the origin and the timeout, supersedes the three older clauses, and
    those deltas stay verbatim (equal to BASE's paragraphs). Read at ``TIP_IMPL`` (fixed), compared with BASE."""
    doc02 = _blob(TIP_IMPL, "docs/02-architecture.md")
    para = delta(doc02, HEADER)
    low = para.lower()
    need = {
        "live": "`live`" in para or "tools = live" in low or '"live"' in para,
        "start-page origin": "origin" in low and ("start page" in low or "start-page" in low),
        "LIVE_FRAME_TIMEOUT_MS": "LIVE_FRAME_TIMEOUT_MS" in para,
        "supersedes": "supersede" in low,
        "the three older deltas": all(t in para for t in ("T-0104", "T-0107a", "T-0107e")),
        "they stay verbatim": "verbatim" in low,
        "swap": "swap" in low or "replace" in low,
        "fresh frame": "fresh" in low,
        "freshness / newest frame": "newest" in low,
        "failed": "`failed`" in para or "failed" in low,
        "window closed": "closed" in low,
        "assists dropped": "assist" in low,
        "egress": "egress" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0111g B16: the docs/02 §5 delta lacks {missing}"
    gr.require_commits(BASE_SHA)
    base = gr._git("show", f"{BASE_SHA}:docs/02-architecture.md")
    assert base.returncode == 0, base.stderr
    for header in OLDER:
        assert delta(doc02, header) == delta(base.stdout, header), (
            f"T-0111g B16: the {header} delta stays verbatim (the record)"
        )


def test_doc01_one_computer_use_paragraph_with_status() -> None:
    """B16: docs/01 §3 still has exactly one paragraph containing "computer use"; the status sentence is appended to
    it (BASE's paragraph is its prefix) and says: its own sandboxed browser, Astra's loopback pages only, T-0111g,
    real websites wait on H-T111c-E, the desktop stays out of scope. Read at ``TIP_IMPL`` (fixed), compared with
    BASE."""
    doc01 = _blob(TIP_IMPL, "docs/01-design.md")
    para = t7d.computer_use_paragraph(doc01)
    gr.require_commits(BASE_SHA)
    base = gr._git("show", f"{BASE_SHA}:docs/01-design.md")
    assert base.returncode == 0, base.stderr
    old = t7d.computer_use_paragraph(base.stdout).rstrip()
    assert para.startswith(old) and len(para.rstrip()) > len(old), (
        "T-0111g B16: one sentence appended to the paragraph; its BASE text is kept as the prefix"
    )
    added = para[len(old) :]
    low = added.lower()
    need = {
        "its own browser": "own" in low and "browser" in low,
        "sandboxed": "sandbox" in low,
        "loopback pages only": "loopback" in low,
        "T-0111g": "T-0111g" in added,
        "real websites wait on H-T111c-E": "H-T111c-E" in added and "website" in low,
        "the desktop stays out of scope": "desktop" in low,
        "behind tools = live": "live" in low,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0111g B16: the appended sentence lacks {missing}: {added!r}"


def test_pinned_sentences_still_verbatim() -> None:
    """B16: the pinned sentences of ``test_speaking_policy_doc.py`` and ``test_t0107a_docs.py`` are still in the one
    computer-use paragraph, verbatim (their own checks, imported). Green at BASE."""
    doc01 = DOC01.read_text(encoding="utf-8")
    spd.check_section3(doc01)
    t7d.check_section3_status(doc01, None)
