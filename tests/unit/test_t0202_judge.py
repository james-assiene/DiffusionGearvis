"""T-0202 group A (A3): the live judge with the two wait tool sets (docs/tasks/T-0202.md; modelled on
``tests/unit/test_t0140_judge.py``).

**The surface these rows pin:** ``live.classify(tool_text, tool_set=...)`` and ``live.repair_verdict(text,
tool_set=...)`` (T-0140's keyword, already at BASE) with ``tools.load(tools.TOOLS_V1_WAIT_PATH)`` and
``tools.load(tools.TOOLS_V1_OVERVIEW_WAIT_PATH)``. Group A changes no judge code: the verdicts below follow from the
schema A1 fixes (``wait`` takes the strict empty object) and from BASE's rule that a tool without a payload is never
``repairable``.

Hand verdicts with either wait set:
- ``call:wait{}`` is ``valid`` with arguments ``{}`` (tagged, untagged, after a closed thought; from ``classify`` and
  ``repair_verdict``).
- ``call:wait{seconds:2}`` is ``refused`` (``additionalProperties: false``). Its reason is not pinned: A3 says only
  "refused".
- a ``wait`` draft (cut, or with an argument, or followed by text) is never ``repairable``.

With a v1 kit (``tool_set=None`` or v1 passed explicitly), ``call:wait{}`` is ``refused`` with ``no tool name`` (the
T-0140 A6 pattern; green at BASE, a guard).

**The oracle for "every verdict on the five v1 tools is unchanged":** the same judge with ``tool_set=None`` (v1) for
the v1+wait set, and with the overview set for the overview+wait set, over the T-0140 corpus (every text of the
live-judge tables in ``tests/unit/test_toolcall_live.py``, T-0091's repair rows, T-0140's ``page_overview`` texts),
minus texts that call ``wait``, plus a Hypothesis alphabet over the T-0100 names and call fragments.

BASE (group A): 3a755e2b. Regime: CPU, pure, n = 1; Hypothesis 300 examples, database off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.toolcall import live, tools
from tests.unit import test_t0140_judge as t0140
from tests.unit import test_toolcall_live as tl
from tests.unit._toolcall_live_support import CLOSE, OPEN, Q

WAIT_VALID = OPEN + "call:wait{}" + CLOSE
WAIT_ARG = OPEN + "call:wait{seconds:2}" + CLOSE
#: Adversarial ``wait`` drafts: none may ever be ``repairable`` (A3).
WAIT_DRAFTS = [
    WAIT_ARG,
    OPEN + "call:wait{seconds:2}",
    OPEN + "call:wait{seconds:",
    OPEN + "call:wait{",
    OPEN + "call:wait{}",
    OPEN + "call:wait{}" + CLOSE + " Wait, no.",
    OPEN + "call:wait{} I will wait for the page" + CLOSE,
    OPEN + "call:wait{x:" + Q + "a",
    OPEN + "call:wait{x:" + Q + "a" + Q + "}" + CLOSE,
    OPEN + "call:wait{}<channel|>",
    OPEN + "call:wait{seconds:2,x:1}" + CLOSE,
    OPEN + "<|channel>thought\nloading<channel|>call:wait{seconds:2}" + CLOSE,
]


def _set(name: str) -> Any:
    p = getattr(tools, name, None)
    assert p is not None and Path(p).is_file(), (
        f"T-0202 A1/A2: tools.{name} (and its file) does not exist yet"
    )
    return tools.load(p)


def wait_set() -> Any:
    return _set("TOOLS_V1_WAIT_PATH")


def overview_wait_set() -> Any:
    return _set("TOOLS_V1_OVERVIEW_WAIT_PATH")


def both_sets() -> list[Any]:
    return [wait_set(), overview_wait_set()]


def as_tuple(v: Any) -> tuple[Any, ...]:
    return (v.status, v.name, v.arguments, v.reason, v.query_problem)


def corpus_without_wait() -> list[str]:
    return [t for t in t0140.corpus() if "call:wait" not in t]


# ---------------------------------------------------------------------------------------------------- green at BASE


def test_wait_refused_by_v1_kit() -> None:
    """A3 (guard; green at BASE and after; adversarial, the mutant makes the judge always load a wait set): with
    v1 given as ``None``, omitted, or explicitly, ``call:wait{}`` is ``refused`` with ``no tool name`` from both
    ``classify`` and ``repair_verdict``; with T-0140's overview set (no ``wait``) it is ``refused`` too."""
    for fn in (live.classify, live.repair_verdict):
        for kw in ({}, {"tool_set": None}, {"tool_set": tools.load()}):
            v = fn(WAIT_VALID, **kw)
            assert (v.status, v.reason) == ("refused", "no tool name"), (fn.__name__, kw, v)
        ov = tools.load(tools.TOOLS_V1_OVERVIEW_PATH)
        assert fn(WAIT_VALID, tool_set=ov).status == "refused", fn.__name__


def test_corpus_premise() -> None:
    """Premise (green at BASE): the corpus the negative row compares over is T-0140's (more than 180 texts), holds
    every verdict kind, and no text in it calls ``wait`` (so the filtered corpus is the whole corpus today)."""
    texts = corpus_without_wait()
    assert len(texts) > 180, len(texts)
    assert texts == t0140.corpus()
    assert {live.classify(t).status for t in texts} == {"valid", "repairable", "refused"}


# ---------------------------------------------------------------------------------------------------- needs A1/A2


def test_wait_valid_in_wait_sets() -> None:
    """A3: with either wait set, ``call:wait{}`` is ``valid`` with name ``wait`` and arguments ``{}``, tagged or
    untagged, from ``classify``; and from ``repair_verdict`` after a closed thought."""
    for ws in both_sets():
        for t in (WAIT_VALID, "call:wait{}" + CLOSE):
            v = live.classify(t, tool_set=ws)
            assert (v.status, v.name, v.arguments) == ("valid", "wait", {}), (ws.path, t, v)
        r = live.repair_verdict(
            OPEN + "<|channel>thought\nloading<channel|>call:wait{}" + CLOSE, tool_set=ws
        )
        assert (r.status, r.name, r.arguments) == ("valid", "wait", {}), (ws.path, r)


def test_wait_with_argument_refused() -> None:
    """A3 (adversarial): with either wait set, ``call:wait{seconds:2}`` is ``refused``; no ``wait`` draft
    (``{seconds:2}``, ``{}`` followed by model text, a truncated ``call:wait{``, and the other rows of
    ``WAIT_DRAFTS``) is ever ``repairable``, from ``classify`` or ``repair_verdict``. Power: the same set makes
    ``call:wait{}`` valid (so the refusals are not the set failing to load)."""
    for ws in both_sets():
        assert live.classify(WAIT_VALID, tool_set=ws).status == "valid", ws.path
        for fn in (live.classify, live.repair_verdict):
            assert fn(WAIT_ARG, tool_set=ws).status == "refused", (fn.__name__, ws.path)
            assert fn(OPEN + "call:wait{seconds:2,x:1}" + CLOSE, tool_set=ws).status == "refused"
            for t in WAIT_DRAFTS:
                assert fn(t, tool_set=ws).status != "repairable", (fn.__name__, ws.path, t)


@settings(max_examples=300, deadline=None, database=None)
@given(parts=st.lists(st.sampled_from(tl._FRAGMENTS), max_size=10))
def test_wait_draft_never_repairable(parts: list[str]) -> None:
    """A3 as a property (adversarial; the mutant extends the repairable rule to the set's names): whatever follows
    ``call:wait{`` (Hypothesis over the T-0100 fragments), neither wait set ever answers ``repairable``."""
    for ws in both_sets():
        t = OPEN + "call:wait{" + "".join(parts)
        assert live.classify(t, tool_set=ws).status != "repairable", (ws.path, t)
        assert live.repair_verdict(t, tool_set=ws).status != "repairable", (ws.path, t)


def test_v1_tool_verdicts_unchanged_against_wait_set() -> None:
    """A3 (negative): for every corpus text that does not call ``wait``, the v1+wait set gives the v1 verdict
    (``tool_set=None``) and the overview+wait set gives the overview set's verdict, field for field, from
    ``classify`` and ``repair_verdict``. The T-0100 bounds and the ``press_key`` list are read from the given set
    and agree with v1's."""
    ws, ows = wait_set(), overview_wait_set()
    ov = tools.load(tools.TOOLS_V1_OVERVIEW_PATH)
    for t in corpus_without_wait():
        for fn in (live.classify, live.repair_verdict):
            assert as_tuple(fn(t, tool_set=ws)) == as_tuple(fn(t, tool_set=None)), (fn.__name__, t)
            assert as_tuple(fn(t, tool_set=ows)) == as_tuple(fn(t, tool_set=ov)), (fn.__name__, t)


@settings(max_examples=300, deadline=None, database=None)
@given(name=st.sampled_from(tl._NAMES), parts=st.lists(st.sampled_from(tl._FRAGMENTS), max_size=10))
def test_v1_verdicts_unchanged_property(name: str, parts: list[str]) -> None:
    """A3 (negative) as a property over the T-0100 names and fragments: the v1+wait set gives the ``None`` verdict
    and the overview+wait set the overview set's, field for field."""
    ws, ows = wait_set(), overview_wait_set()
    ov = tools.load(tools.TOOLS_V1_OVERVIEW_PATH)
    t = OPEN + "call:" + name + "{" + "".join(parts)
    for fn in (live.classify, live.repair_verdict):
        assert as_tuple(fn(t, tool_set=ws)) == as_tuple(fn(t, tool_set=None)), (fn.__name__, t)
        assert as_tuple(fn(t, tool_set=ows)) == as_tuple(fn(t, tool_set=ov)), (fn.__name__, t)
