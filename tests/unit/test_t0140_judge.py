"""T-0140 A2 (group A1): the live judge's optional tool set (docs/tasks/T-0140.md).

**The surface these rows pin:** ``live.classify(tool_text, tool_set=None)`` and ``live.repair_verdict(text,
tool_set=None)``, ``tool_set`` a ``tools.ToolSet``; ``None`` means ``tools.load()`` and changes no verdict.

**The oracle for "every BASE verdict is unchanged":** ``local_astra/toolcall/live.py`` as committed at the literal
BASE 0bc60ff, executed as its own module over today's ``native``/``score``/``router``/``tools`` (none of which A1
changes beyond the ``tools`` path constant), the T-0100 pattern (``test_toolcall_live._live_at``). The corpus is
every text of the existing live-judge tables in ``tests/unit/test_toolcall_live.py`` (T-0091, T-0100, T-0102), the
T-0091 repair rows, the ``page_overview`` texts below, and a Hypothesis alphabet over call fragments.

``test_default_set_is_base`` calls the judge without the keyword, so it is green at BASE (and kills reviewer
mutation 7, the judge always on the overview set, through the default); its ``tool_set=None`` twin needs the keyword.

Hand verdicts with the overview set: ``call:page_overview{}`` is ``valid`` with ``{}``; ``{x:1}`` is ``refused``
(``additionalProperties: false``); a cut or otherwise unparsable ``page_overview`` draft is never ``repairable``
(it has no payload; ``native.draft_name`` reads v1 names only, and ``native.py`` is frozen). The reason of the
``{x:1}`` refusal is not pinned: A2 says "refused (schema)" and BASE's judge says "no tool name" for a parsed call
whose name ``draft_name`` does not know; flagged in the A1 report.

BASE (group A1): 0bc60ff (the task header's ca730f5 is stale). Regime: CPU, pure, n = 1; Hypothesis 300 examples.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.toolcall import live, tools
from tests.unit import test_toolcall_live as tl
from tests.unit._toolcall_live_support import CLOSE, OPEN, Q

#: Group A1's base: main when the branch was cut (a literal sha, never a moving ref; tests.md).
BASE_A1 = "0bc60ff"

OV_VALID = OPEN + "call:page_overview{}" + CLOSE
OV_EXTRA = OPEN + "call:page_overview{x:1}" + CLOSE
OV_TEXTS = [
    OV_VALID,
    OV_EXTRA,
    OPEN + "call:page_overview{",
    OPEN + "call:page_overview{}",
    OPEN + "call:page_overview{x:",
    OPEN + "call:page_overview{" + Q + "a",
    OPEN + "call:page_overview{}" + CLOSE + " Wait, no.",
    OPEN + "call:page_overview{}<channel|>",
    OPEN + "call:page_overview{full:true}" + CLOSE,
    OPEN
    + "call:page_overview{}"
    + CLOSE
    + OPEN
    + "call:scroll{direction:"
    + Q
    + "down"
    + Q
    + "}"
    + CLOSE,
    "call:page_overview{}" + CLOSE,
    OPEN + "<|channel>thought\nok<channel|>call:page_overview{}" + CLOSE,
]
#: T-0091's repair rows (``test_toolcall_live.test_c6_repair_verdict``'s parameters, rebuilt here: they are inline).
_CALL = "call:click_at{x:10,y:20}" + CLOSE
REPAIR_TEXTS = [
    OPEN + _CALL,
    _CALL,
    OPEN + "<|channel>thought\nok<channel|>" + _CALL,
    "<|channel>thought\nok<channel|>" + OPEN + _CALL,
    OPEN + "<|channel>thought\nnever closed",
    OPEN + "call:click_at{x:1200,y:5}" + CLOSE,
    OPEN + "call:click_at{x:488,y:5",
    OPEN + "call:navigate{url:" + Q + "x" + Q + "}" + CLOSE,
]


def _texts_of(table: Any) -> Iterator[str]:
    values = table.values() if isinstance(table, dict) else table
    for v in values:
        if isinstance(v, str):
            yield v
        elif isinstance(v, tuple) and v and isinstance(v[0], str):
            yield v[0]


def corpus() -> list[str]:
    """Every text of the live-judge tables (module docstring), deduplicated, in a stable order."""
    out: list[str] = []
    for name in sorted(dir(tl)):
        if name.endswith("_ROWS") and name not in ("X_ROWS", "Y_ROWS"):
            out += list(_texts_of(getattr(tl, name)))
    out += list(tl._ALL_T0100_TEXTS) + REPAIR_TEXTS + OV_TEXTS
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]


def base() -> Any:
    """The judge at BASE_A1 (skips when the commit is absent from this clone)."""
    return tl._live_at(BASE_A1)


def as_tuple(v: Any) -> tuple[Any, ...]:
    return (v.status, v.name, v.arguments, v.reason, v.query_problem)


def has_tool_set(fn: Any) -> None:
    assert "tool_set" in inspect.signature(fn).parameters, (
        f"T-0140 A2: live.{fn.__name__}(..., tool_set=None) does not exist yet"
    )


def overview_set() -> Any:
    p = getattr(tools, "TOOLS_V1_OVERVIEW_PATH", None)
    assert p is not None and Path(p).is_file(), (
        "T-0140 A1: tools.TOOLS_V1_OVERVIEW_PATH / tools_v1_overview.json does not exist yet"
    )
    return tools.load(p)


def test_corpus_premise() -> None:
    """Premise: the corpus is the tables it claims (198 at BASE: the 149 T-0100 texts and more) and holds every verdict kind."""
    texts = corpus()
    assert len(texts) > 180, len(texts)
    kinds = {live.classify(t).status for t in texts}
    assert kinds == {"valid", "repairable", "refused"}, kinds


# ---------------------------------------------------------------------------------------------------- green at BASE


def test_default_set_is_base() -> None:
    """A2 (green at BASE; reviewer mutation 7 through the default): ``classify(t)`` and ``repair_verdict(t)`` give
    BASE's verdict, field for field, for every corpus text."""
    b = base()
    for t in corpus():
        assert as_tuple(live.classify(t)) == as_tuple(b.classify(t)), t
        assert as_tuple(live.repair_verdict(t)) == as_tuple(b.repair_verdict(t)), t


def test_v1_refuses_overview_by_default() -> None:
    """A2/A8 (green at BASE; adversarial, the mutant always loads the overview set): with v1, ``page_overview{}``
    is ``refused`` with ``no tool name``, from ``classify`` and from ``repair_verdict``."""
    for fn in (live.classify, live.repair_verdict):
        v = fn(OV_VALID)
        assert (v.status, v.reason) == ("refused", "no tool name"), (fn.__name__, v)


# ---------------------------------------------------------------------------------------------------- needs A1


def test_default_set_is_base_explicit_none() -> None:
    """A2: ``tool_set=None`` and ``tool_set=tools.load()`` (v1 passed explicitly) give BASE's verdict, field for
    field, for every corpus text, from both functions."""
    has_tool_set(live.classify)
    has_tool_set(live.repair_verdict)
    b, v1 = base(), tools.load()
    for t in corpus():
        want_c, want_r = as_tuple(b.classify(t)), as_tuple(b.repair_verdict(t))
        for ts in (None, v1):
            assert as_tuple(live.classify(t, tool_set=ts)) == want_c, (t, ts)
            assert as_tuple(live.repair_verdict(t, tool_set=ts)) == want_r, (t, ts)


def test_v1_kit_refuses_overview() -> None:
    """A2/A8 (adversarial; the mutant always loads the overview set): with v1 given as ``None`` or explicitly,
    ``call:page_overview{}`` is ``refused`` with ``no tool name``."""
    has_tool_set(live.classify)
    for ts in (None, tools.load()):
        v = live.classify(OV_VALID, tool_set=ts)
        assert (v.status, v.reason) == ("refused", "no tool name"), (ts, v)


def test_overview_valid_and_refusals() -> None:
    """A2 with the overview set: ``page_overview{}`` is ``valid`` with arguments ``{}`` (``classify`` and
    ``repair_verdict``, tagged or not, after a closed thought); ``{x:1}`` and ``{full:true}`` are ``refused``; a cut
    draft is ``refused``, never ``repairable``."""
    has_tool_set(live.classify)
    has_tool_set(live.repair_verdict)
    ov = overview_set()
    for t in (OV_VALID, "call:page_overview{}" + CLOSE):
        v = live.classify(t, tool_set=ov)
        assert (v.status, v.name, v.arguments) == ("valid", "page_overview", {}), (t, v)
    r = live.repair_verdict(
        OPEN + "<|channel>thought\nok<channel|>call:page_overview{}" + CLOSE, tool_set=ov
    )
    assert (r.status, r.name, r.arguments) == ("valid", "page_overview", {}), r
    for t in (OV_EXTRA, OPEN + "call:page_overview{full:true}" + CLOSE):
        assert live.classify(t, tool_set=ov).status == "refused", t
        assert live.repair_verdict(t, tool_set=ov).status == "refused", t
    for t in (
        OPEN + "call:page_overview{",
        OPEN + "call:page_overview{x:",
        OPEN + "call:page_overview{}",
    ):
        assert live.classify(t, tool_set=ov).status == "refused", t


@settings(max_examples=300, deadline=None, database=None)
@given(parts=st.lists(st.sampled_from(tl._FRAGMENTS), max_size=10))
def test_overview_draft_never_repairable(parts: list[str]) -> None:
    """A2 (adversarial; the mutant extends the repairable rule to the set's names): no ``page_overview`` draft is
    ever ``repairable`` with the overview set, whatever follows ``call:page_overview{`` (Hypothesis over the T-0100
    fragments), and ``repair_verdict`` never answers ``repairable`` either."""
    has_tool_set(live.classify)
    ov = overview_set()
    t = OPEN + "call:page_overview{" + "".join(parts)
    assert live.classify(t, tool_set=ov).status != "repairable", t
    assert live.repair_verdict(t, tool_set=ov).status != "repairable", t


def test_overview_set_keeps_v1_verdicts() -> None:
    """A2 (the overview set's first five declarations are v1's, deep-equal): for every corpus text that does not
    name ``page_overview``, the overview set gives BASE's verdict; the T-0100 bounds and the ``press_key`` list are
    read from the given set and agree with v1's."""
    has_tool_set(live.classify)
    has_tool_set(live.repair_verdict)
    b, ov = base(), overview_set()
    for t in corpus():
        if "page_overview" in t:
            continue
        assert as_tuple(live.classify(t, tool_set=ov)) == as_tuple(b.classify(t)), t
        assert as_tuple(live.repair_verdict(t, tool_set=ov)) == as_tuple(b.repair_verdict(t)), t


@settings(max_examples=300, deadline=None, database=None)
@given(name=st.sampled_from(tl._NAMES), parts=st.lists(st.sampled_from(tl._FRAGMENTS), max_size=10))
def test_none_is_base_property(name: str, parts: list[str]) -> None:
    """A2 as a property (Hypothesis, 300 examples over the T-0100 names and fragments): ``tool_set=None`` gives
    BASE's verdicts field for field.

    T-0196 widening (docs/tasks/T-0196.md, Owned files): ``classify`` may also differ by exactly the T-0196 delta
    (``test_toolcall_live.t0196_delta``); ``repair_verdict`` stays field for field."""
    has_tool_set(live.classify)
    has_tool_set(live.repair_verdict)
    b = base()
    t = OPEN + "call:" + name + "{" + "".join(parts)
    new = live.classify(t, tool_set=None)
    assert as_tuple(new) == as_tuple(b.classify(t)) or tl.t0196_delta(t, new), t
    assert as_tuple(live.repair_verdict(t, tool_set=None)) == as_tuple(b.repair_verdict(t)), t


def test_load_payloads_still_v1() -> None:
    """A2 (green at BASE and after): ``load_payloads`` takes no tool set and is keyed on v1's sha; its payloads are
    the five v1 tools (no ``page_overview``)."""
    assert list(inspect.signature(live.load_payloads).parameters) == ["path"]
    p = live.load_payloads()
    assert (
        p.manifest["tools_sha256"]
        == "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
    )
    assert sorted(p.tools) == sorted(
        ("google_search", "click_at", "type_at", "scroll", "press_key")
    )
