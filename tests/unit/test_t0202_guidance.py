"""T-0202 group A (A4, A6): ``prompt_builder.WAIT_TOOL_GUIDANCE`` and ``prompt_builder.with_wait_guidance``
(docs/tasks/T-0202.md).

**The surface these rows pin:**
- ``WAIT_TOOL_GUIDANCE`` is one ASCII module-level string literal (implicitly concatenated pieces are one literal),
  exactly H-T202-4's sentence, typed below from the task and pinned also by the sha256 of that typed text (computed
  by hand at BASE).
- ``with_wait_guidance(text: str) -> str`` returns ``text + " " + WAIT_TOOL_GUIDANCE``, pure.
- The constant contains no special-token literal (``drop_special_literals(c) == c``), no URL and no session label text
  (every module-level ``str`` constant of ``conductor/session.py`` whose name contains ``LABEL``).

Guard: ``test_guidance_versions_unchanged`` (A6) pins the v1, v2 and v3 live guidance shas, ``LIVE_GUIDANCE``'s
mapping and ``NATIVE_TOOL_INSTRUCTIONS``'s sha, all as full literals at BASE 3a755e2b. T-0211 B-R2 re-pinned its key
list to ``{v1, v2, v3, v4}`` (pending T-0211 until B's ``/implement``).

BASE (group A): 3a755e2b. Regime: CPU, pure, n = 1.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
from pathlib import Path
from typing import Any

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor import session as session_mod
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, drop_special_literals

REPO = Path(__file__).resolve().parents[2]
PB_PATH = REPO / "local_astra" / "conductor" / "prompt_builder.py"

#: H-T202-4, verbatim from the task.
WAIT_GUIDANCE_TEXT = (
    "If the latest screenshot shows the page still loading or the results you expect not there yet, call wait and "
    "look at the new screenshot instead of telling the user to wait or refresh."
)
#: sha256 of ``WAIT_GUIDANCE_TEXT`` as typed above (UTF-8), computed by hand at BASE.
WAIT_GUIDANCE_SHA256 = "6bcac15011ed8714addf1c3e7a53a44aa2b5e8b48a1bee4cd62474ca564beeed"

#: A6: the BASE shas (3a755e2b), full literals. v1/v2 agree with T-0191's pins; v3 and the mapping computed at BASE.
V1_SHA = "768a973bb546979536d2964f1c9767b764346843ef35921769b8cd75638af2ec"
V2_SHA = "518f56a1de3faff26c7fc1b8f938c4d233e59c5379f9acc431b74b7cefc41e50"
V3_SHA = "b20cd7239f29698a5bcf8f587e9b2a2b53de91e6f021bd63e74ae5ef40e082c5"
NATIVE_SHA = "6417aabda6c4e0b1bde587bdcea506f5cf2bbb12c896ac50074e9e63722ac55c"

URL_RE = re.compile(
    r"(?i)\b[a-z][a-z0-9+.\-]*://|\bwww\.|\b[a-z0-9\-]+\.(?:com|org|net|io|dev|html?)\b"
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def constant() -> str:
    c = getattr(pb, "WAIT_TOOL_GUIDANCE", None)
    assert isinstance(c, str), (
        "T-0202 A4: prompt_builder.WAIT_TOOL_GUIDANCE (a str) does not exist yet"
    )
    return c


def composer() -> Any:
    fn = getattr(pb, "with_wait_guidance", None)
    assert callable(fn), "T-0202 A4: prompt_builder.with_wait_guidance does not exist yet"
    return fn


def session_label_texts() -> dict[str, str]:
    """Every module-level ``str`` constant of ``session.py`` whose name contains ``LABEL`` (LABEL_NOW,
    LABEL_UPDATED, LABEL_UNCHANGED, SHARED_LABEL_*, OVERVIEW_LABEL, OVERVIEW_FRAME_LABEL, ...)."""
    return {
        k: v
        for k, v in vars(session_mod).items()
        if "LABEL" in k and k.isupper() and isinstance(v, str) and len(v) >= 20
    }


# ---------------------------------------------------------------------------------------------------- green at BASE


def test_guidance_versions_unchanged() -> None:
    """A6 (negative): ``LIVE_TOOL_GUIDANCE``, ``_V2`` and ``_V3`` keep their BASE shas; ``LIVE_GUIDANCE`` is exactly
    ``{v1, v2, v3, v4}`` (re-pinned by T-0211 B-R2: v4 is T-0211's own entry, ``LIVE_TOOL_GUIDANCE_V4``) with v1-v3
    mapping to those same objects (no wait variant); and ``NATIVE_TOOL_INSTRUCTIONS`` keeps its sha. None of them
    mentions the wait tool."""
    assert sha(pb.LIVE_TOOL_GUIDANCE) == V1_SHA
    assert sha(pb.LIVE_TOOL_GUIDANCE_V2) == V2_SHA
    assert sha(pb.LIVE_TOOL_GUIDANCE_V3) == V3_SHA
    assert sha(pb.NATIVE_TOOL_INSTRUCTIONS) == NATIVE_SHA
    assert "v4" in pb.LIVE_GUIDANCE, (
        f"T-0211 B3: LIVE_GUIDANCE['v4'] does not exist yet; {list(pb.LIVE_GUIDANCE)}"
    )
    assert sorted(pb.LIVE_GUIDANCE) == ["v1", "v2", "v3", "v4"], list(pb.LIVE_GUIDANCE)
    assert pb.LIVE_GUIDANCE["v4"] is getattr(pb, "LIVE_TOOL_GUIDANCE_V4", None)
    assert pb.LIVE_GUIDANCE["v1"] is pb.LIVE_TOOL_GUIDANCE
    assert pb.LIVE_GUIDANCE["v2"] is pb.LIVE_TOOL_GUIDANCE_V2
    assert pb.LIVE_GUIDANCE["v3"] is pb.LIVE_TOOL_GUIDANCE_V3
    for text in (*pb.LIVE_GUIDANCE.values(), pb.NATIVE_TOOL_INSTRUCTIONS):
        assert "call wait" not in text


def test_label_census_premise() -> None:
    """Premise (green at BASE): the label census sees the three T-0134 labels, so the "no label text" row is not
    vacuous."""
    labels = session_label_texts()
    for name in ("LABEL_NOW", "LABEL_UPDATED", "LABEL_UNCHANGED"):
        assert name in labels, sorted(labels)


# ---------------------------------------------------------------------------------------------------- needs A4


def test_constant_text_pinned() -> None:
    """A4: the constant is H-T202-4's text exactly (equality and sha256), ASCII, and in ``prompt_builder.py`` it is
    one module-level assignment whose value is one string literal (no f-string, no ``+``, no call)."""
    c = constant()
    assert c == WAIT_GUIDANCE_TEXT, c
    assert sha(c) == WAIT_GUIDANCE_SHA256
    assert c.isascii()
    tree = ast.parse(PB_PATH.read_text(encoding="utf-8"))
    assigns = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "WAIT_TOOL_GUIDANCE" for t in node.targets)
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "WAIT_TOOL_GUIDANCE"
        )
    ]
    assert len(assigns) == 1, (
        f"T-0202 A4: {len(assigns)} module-level assignments of WAIT_TOOL_GUIDANCE"
    )
    value = assigns[0].value
    assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
        ast.dump(value) if value else None
    )
    assert value.value == WAIT_GUIDANCE_TEXT


def test_with_wait_guidance_appends_one_space() -> None:
    """A4: ``with_wait_guidance(text)`` is ``text + " " + WAIT_TOOL_GUIDANCE`` for each live guidance version, the
    empty string, a text ending in a space and a text ending in a newline (no strip, no dedup, no second space); it
    takes one parameter ``text`` and leaves its argument and the constant unchanged (pure)."""
    fn = composer()
    c = constant()
    assert list(inspect.signature(fn).parameters) == ["text"], inspect.signature(fn)
    v3_before = pb.LIVE_TOOL_GUIDANCE_V3
    for text in (
        pb.LIVE_TOOL_GUIDANCE,
        pb.LIVE_TOOL_GUIDANCE_V2,
        pb.LIVE_TOOL_GUIDANCE_V3,
        "",
        "Guidance. ",
        "Guidance.\n",
        WAIT_GUIDANCE_TEXT,
    ):
        got = fn(text)
        assert got == text + " " + WAIT_GUIDANCE_TEXT, (text, got)
        assert fn(text) == got
    assert fn(pb.LIVE_TOOL_GUIDANCE_V3).count(WAIT_GUIDANCE_TEXT) == 1
    assert pb.LIVE_TOOL_GUIDANCE_V3 is v3_before and sha(pb.LIVE_TOOL_GUIDANCE_V3) == V3_SHA
    assert constant() == c


def test_no_special_literal_no_label_text() -> None:
    """A4 (adversarial): ``drop_special_literals(c) == c`` and no ``SPECIAL_TOKENS`` literal or bare ``<|`` / ``|>``
    is in it; it holds no URL; and it quotes none of ``session.py``'s label texts (``LABEL_*``, nor the T-0134
    "the browser screen now" core they share)."""
    c = constant()
    assert drop_special_literals(c) == c
    for lit in SPECIAL_TOKENS:
        assert lit not in c, lit
    assert "<|" not in c and "|>" not in c
    assert not URL_RE.search(c), URL_RE.search(c)
    for name, text in session_label_texts().items():
        assert text not in c, name
        assert text.strip("[]. ") not in c, name
    assert "browser screen now" not in c
