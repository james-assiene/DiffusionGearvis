"""T-0127 criteria 1 and 2: rule CC (a Card written with ``children``) in ``repair_components``, and its reserved
namespace through the gate.

Goldens are hand-written from criterion 1: one distinct id -> ``child`` is that id, ``children`` removed, no new id;
several ids -> ``child`` = ``__gate_<card id>_body`` and a lifted ``{"id": "__gate_<card id>_body", "component":
"Column", "children": [<ids, order kept, first occurrence>]}`` right after the Card (H-T127-1). Every refusal row
carries a positive twin (the same input with the one refusing feature removed) so that it fails at the base, where
nothing is repaired, and cannot be satisfied by a rule that refuses everything.

Readings where the task is open (named in the test-author report):

- ``edits`` counts rule applications, so one CC repair is ``edits == 1`` (as BC/RB count in T-0108b).
- (c) mixed inline and string children: the Card comes first, the gate-made Column right after it ("follows the
  Card"), the lifted inline object somewhere after the Card; its position relative to the Column is not pinned.
- A cycle closed through string references (a child Column that references the Card) is refused like an inline
  one: the wrap would turn a pruned Card into a topology cycle, and a cycle yields no surface at all (T-0010 A2).
- A model-written component whose id is the would-be ``__gate_<card id>_body`` makes the several-ids repair refuse
  (criterion 2: never borrowed or duplicated).

Regime: CPU, pure, the vendored A2UI v0.9.1 schema; Hypothesis ``max_examples`` 120, derandomized, no database.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.conductor import ui_gate as ui_gate_module
from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.ui import A2UIValidator
from local_astra.ui import component_repair as component_repair_module

CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
CREATE = '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"' + CATALOG + '"}}'
V = A2UIValidator()


def repair(components: list[Any]) -> tuple[list[Any], int]:
    out, edits = component_repair_module.repair_components(components, V)
    n = edits if isinstance(edits, int) else len(edits)
    return out, n


def ok(component: Any) -> bool:
    return not V.component_errors(component)


def text(cid: str, s: str) -> dict[str, Any]:
    return {"id": cid, "component": "Text", "text": s}


def card_children(cid: str, *children: Any) -> dict[str, Any]:
    return {"id": cid, "component": "Card", "children": list(children)}


A, B, D = text("a", "Alpha"), text("b", "Beta"), text("d", "Delta")


def assert_untouched(inp: list[Any]) -> None:
    """The refused input comes back as the input list object, each component the identical object, 0 edits."""
    snapshot = copy.deepcopy(inp)
    out, edits = repair(inp)
    assert out is inp and edits == 0, f"refusal expected; got {json.dumps(out)} ({edits} edits)"
    assert inp == snapshot, "the input was mutated"


def assert_repaired_card(inp: list[Any], cid: str) -> list[Any]:
    """The positive twin: the Card ``cid`` comes back valid, with a string ``child`` and no ``children``."""
    out, edits = repair(inp)
    got = [c for c in out if isinstance(c, dict) and c.get("id") == cid]
    assert len(got) == 1 and ok(got[0]), f"twin: Card {cid} not repaired: {json.dumps(out)}"
    assert isinstance(got[0].get("child"), str) and "children" not in got[0]
    assert edits >= 1
    return out


# --- criterion 1(a): one distinct id ----------------------------------------------------------------------------


def test_cc_one_id_sets_child_and_creates_no_id() -> None:
    """1(a): ``children: ["a"]`` -> ``child: "a"``; nothing lifted, no new id, one edit; ``a`` the identical object."""
    inp = [card_children("c", "a"), A]
    out, edits = repair(inp)
    assert out == [{"id": "c", "component": "Card", "child": "a"}, A], json.dumps(out)
    assert out[1] is inp[1]
    assert edits == 1


def test_cc_one_id_written_twice_is_one_id() -> None:
    """1(a) with duplicates: ``["a", "a"]`` holds one distinct id, so no wrap (mutation 1: always wrap)."""
    out, _ = repair([card_children("c", "a", "a"), A])
    assert out == [{"id": "c", "component": "Card", "child": "a"}, A], json.dumps(out)


def test_cc_one_id_not_yet_defined_still_sets_child() -> None:
    """1(a): the id need not be defined on this canvas (it may arrive on the next one; the gate defers the Card)."""
    out, _ = repair([card_children("c", "later")])
    assert out == [{"id": "c", "component": "Card", "child": "later"}], json.dumps(out)


# --- criterion 1(b): several ids ---------------------------------------------------------------------------------


def test_cc_several_ids_wraps_in_the_gate_column() -> None:
    """1(b), H-T127-1: ``child`` = ``__gate_c_body``; the lifted Column right after the Card, order kept."""
    inp = [card_children("c", "b", "a", "d"), A, B, D]
    out, edits = repair(inp)
    assert out == [
        {"id": "c", "component": "Card", "child": "__gate_c_body"},
        {"id": "__gate_c_body", "component": "Column", "children": ["b", "a", "d"]},
        A,
        B,
        D,
    ], json.dumps(out)
    assert edits == 1
    assert all(out[i] is inp[i - 1] for i in (2, 3, 4))


def test_cc_duplicate_ids_collapse_in_first_occurrence_order() -> None:
    """1(b): ``["b", "a", "b", "d", "a"]`` -> Column ``["b", "a", "d"]`` (mutation 2: reversed order)."""
    out, _ = repair([card_children("c", "b", "a", "b", "d", "a"), A, B, D])
    assert out[:2] == [
        {"id": "c", "component": "Card", "child": "__gate_c_body"},
        {"id": "__gate_c_body", "component": "Column", "children": ["b", "a", "d"]},
    ], json.dumps(out)


def test_cc_keeps_the_cards_other_properties() -> None:
    """1: only ``children`` is replaced; a catalog property the model wrote (``weight``) stays."""
    out, _ = repair([{**card_children("c", "a", "b"), "weight": 1}, A, B])
    assert out[0] == {"id": "c", "component": "Card", "child": "__gate_c_body", "weight": 1}


# --- refusals (each with its positive twin) ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("refused", "twin"),
    [
        (["c"], ["a"]),
        (["c", "a"], ["d", "a"]),
        (["a", "c", "b"], ["a", "d", "b"]),
    ],
    ids=["own-id-alone", "own-id-first", "own-id-middle"],
)
def test_cc_refuses_the_cards_own_id(refused: list[str], twin: list[str]) -> None:
    """1: ``children`` holding the Card's own id is refused (identical object back); the twin without it repairs."""
    assert_untouched([card_children("c", *refused), A, B, D])
    assert_repaired_card([card_children("c", *twin), A, B, D], "c")


@pytest.mark.parametrize(
    ("refused", "twin"),
    [
        (["__gate_x"], ["x"]),
        (["a", "__gate_x"], ["a", "x"]),
        (["__gate_c_body", "a"], ["body", "a"]),
    ],
    ids=["reserved-alone", "reserved-second", "reserved-would-be-wrap-id"],
)
def test_cc_refuses_a_model_written_reserved_id(refused: list[str], twin: list[str]) -> None:
    """1 and 2: a ``__gate_``-prefixed string in ``children`` is never borrowed; the twin repairs."""
    assert_untouched([card_children("c", *refused), A])
    assert_repaired_card([card_children("c", *twin), A], "c")


def test_cc_does_not_duplicate_a_model_written_wrap_id() -> None:
    """2: the model already defined ``__gate_c_body``; wrapping would duplicate it, so the repair refuses. Twin: the
    same canvas without that component repairs."""
    squatter = {"id": "__gate_c_body", "component": "Column", "children": ["a"]}
    assert_untouched([card_children("c", "a", "b"), squatter, A, B])
    assert_repaired_card([card_children("c", "a", "b"), A, B], "c")


def test_cc_never_repairs_a_reserved_card() -> None:
    """2: a Card whose own id is in the reserved namespace is not repaired over; twin ``c`` is."""
    assert_untouched([card_children("__gate_c", "a", "b"), A, B])
    assert_repaired_card([card_children("c", "a", "b"), A, B], "c")


def test_cc_does_not_apply_when_a_string_child_exists() -> None:
    """1: "has no string ``child``": a Card with both ``child`` and ``children`` is left alone; twin repairs."""
    assert_untouched([{"id": "c", "component": "Card", "child": "a", "children": ["b"]}, A, B])
    assert_repaired_card([card_children("c", "b"), A, B], "c")


def test_cc_does_not_apply_to_empty_children() -> None:
    """1: ``children`` must be non-empty; twin repairs."""
    assert_untouched([card_children("c"), A])
    assert_repaired_card([card_children("c", "a"), A], "c")


def test_cc_refuses_an_inline_object_without_an_id() -> None:
    """1(c): an inline object with no id cannot be referenced; twin with an id repairs."""
    assert_untouched([card_children("c", {"component": "Text", "text": "Alpha"}, "b"), B])
    assert_repaired_card(
        [card_children("c", {"id": "a", "component": "Text", "text": "Alpha"}, "b"), B], "c"
    )


def test_cc_refuses_two_different_inline_definitions_of_one_id() -> None:
    """1(c) (mutant x3, a content-loss path): two inline objects share the id ``a`` with different texts; lifting
    one would drop the other's text, so the repair is refused. Twin: the two definitions are equal, and the Card
    repairs with ``child: "a"``."""
    assert_untouched(
        [
            card_children(
                "c",
                {"id": "a", "component": "Text", "text": "Alpha"},
                {"id": "a", "component": "Text", "text": "Other"},
            )
        ]
    )
    twin = assert_repaired_card(
        [
            card_children(
                "c",
                {"id": "a", "component": "Text", "text": "Alpha"},
                {"id": "a", "component": "Text", "text": "Alpha"},
            )
        ],
        "c",
    )
    assert [c for c in twin if c["id"] == "a"] == [
        {"id": "a", "component": "Text", "text": "Alpha"}
    ]


def test_cc_refuses_a_cycle_through_the_wrap_column() -> None:
    """1(c) (named in the task's tests): ``a`` references the Card, so ``c -> __gate_c_body -> a -> c`` would be a
    cycle; refused. Twin: ``a`` references ``b`` instead, and the repair is made."""
    loop = {"id": "a", "component": "Column", "children": ["c"]}
    assert_untouched([card_children("c", "a", "b"), loop, B])
    assert_repaired_card(
        [card_children("c", "a", "b"), {"id": "a", "component": "Column", "children": ["b"]}, B],
        "c",
    )


def test_cc_refuses_an_inline_cycle() -> None:
    """1(c): an inline Column that references the Card is not lifted (LI's cycle rule); twin repairs."""
    loop = {"id": "k", "component": "Column", "children": ["c"]}
    assert_untouched([card_children("c", loop, "b"), B])
    assert_repaired_card(
        [card_children("c", {"id": "k", "component": "Column", "children": ["b"]}, "b"), B], "c"
    )


# --- criterion 1(c): inline children go through LI-style lifting first -----------------------------------------


def test_cc_mixed_inline_and_string_children() -> None:
    """1(c): an inline Text is lifted unchanged and referenced; the Card wraps ``[a, b]``; the Column follows the Card;
    nothing is defined twice."""
    inline = {"id": "a", "component": "Text", "text": "Alpha"}
    inp = [card_children("c", inline, "b"), B]
    out, edits = repair(inp)
    assert out[0] == {"id": "c", "component": "Card", "child": "__gate_c_body"}, json.dumps(out)
    assert out[1] == {"id": "__gate_c_body", "component": "Column", "children": ["a", "b"]}
    assert sorted(c["id"] for c in out) == ["__gate_c_body", "a", "b", "c"], json.dumps(out)
    assert [c for c in out if c["id"] == "a"] == [inline]
    assert any(c is inp[1] for c in out)
    assert edits >= 1


def test_cc_inline_child_already_defined_is_referenced_not_duplicated() -> None:
    """1(c) with C6: an inline copy of an id defined at top level is referenced; ``a`` stays defined once."""
    out, _ = repair([card_children("c", {"id": "a", "component": "Text", "text": "Alpha"}), A])
    assert out == [{"id": "c", "component": "Card", "child": "a"}, A], json.dumps(out)


# --- idempotence ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inp",
    [
        [card_children("c", "a"), A],
        [card_children("c", "a", "b"), A, B],
        [card_children("c", {"id": "a", "component": "Text", "text": "Alpha"}, "b"), B],
    ],
    ids=["one", "several", "mixed"],
)
def test_cc_is_idempotent(inp: list[Any]) -> None:
    """A second pass over CC's output changes nothing: the same list object back, 0 edits."""
    out, edits = repair(inp)
    assert edits >= 1 and out != inp, "first pass: a repair was made"
    again, edits2 = repair(out)
    assert again is out and edits2 == 0, json.dumps(again)


# --- content preservation (Hypothesis) ------------------------------------------------------------------------

POOL = ["a", "b", "d", "e", "f", "g"]


def strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


@st.composite
def card_canvases(draw: st.DrawFn) -> tuple[list[Any], list[str]]:
    children = draw(st.lists(st.sampled_from(POOL), min_size=1, max_size=7))
    defined = draw(st.lists(st.sampled_from(POOL), unique=True, max_size=len(POOL)))
    words = st.text(alphabet="abcxyz XYZ:.0123", min_size=1, max_size=12)
    texts = [text(cid, draw(words)) for cid in defined]
    position = draw(st.integers(min_value=0, max_value=len(texts)))
    comps: list[Any] = [*texts[:position], card_children("c", *children), *texts[position:]]
    return comps, children


@settings(
    max_examples=120,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(card_canvases())
def test_cc_preserves_content(case: tuple[list[Any], list[str]]) -> None:
    """Criteria 1-2 over generated canvases: the Card is repaired; every valid input component is the identical
    object; no id is duplicated; the only new id is ``__gate_c_body`` and only when the children hold two or more
    distinct ids; the Column's children are exactly the distinct ids, each first occurrence before the next; every
    string in the output is the model's, except that id and the Column's kind."""
    inp, children = case
    out, _ = repair(inp)
    distinct = set(children)
    for c in inp:
        if ok(c):
            assert any(o is c for o in out), f"valid {c['id']} not passed through"
    ids = [c["id"] for c in out]
    assert len(ids) == len(set(ids)), f"duplicate id: {ids}"
    new_ids = set(ids) - {c["id"] for c in inp}
    got_card = next(c for c in out if c["id"] == "c")
    assert ok(got_card) and "children" not in got_card, f"Card not repaired: {got_card}"
    if len(distinct) == 1:
        assert new_ids == set() and got_card["child"] == children[0]
    else:
        assert new_ids == {"__gate_c_body"} and got_card["child"] == "__gate_c_body"
        column = next(c for c in out if c["id"] == "__gate_c_body")
        assert column["component"] == "Column"
        assert sorted(column["children"]) == sorted(distinct)
        first = [children.index(x) for x in column["children"]]
        assert first == sorted(first), f"order: {column['children']} from {children}"
    allowed = set(strings(inp)) | {"__gate_c_body", "Column", "child"}
    assert set(strings(out)) <= allowed, set(strings(out)) - allowed


# --- criterion 2 through the gate --------------------------------------------------------------------------------


def update(*components: str) -> str:
    return (
        '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
        + ",".join(components)
        + "]}}"
    )


#: a Row root, so the word "Column" appears nowhere the model wrote
NO_COLUMN = (
    CREATE
    + "\n"
    + update(
        '{"id":"root","component":"Row","children":["c"]}',
        '{"id":"c","component":"Card","children":["a","b"]}',
        '{"id":"a","component":"Text","text":"Alpha"}',
        '{"id":"b","component":"Text","text":"Beta"}',
    )
)


def emitted(result: GateResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in result.surface.messages if result.surface is not None else ():
        for c in (m.get("updateComponents") or {}).get("components", ()):
            out[c["id"]] = c
    return out


@pytest.mark.parametrize("raw", [NO_COLUMN, NO_COLUMN[:-1]], ids=["strict", "unguided"])
def test_cc_reaches_the_renderer_with_its_reserved_id_and_kind(raw: str) -> None:
    """Criterion 2: the id ``__gate_c_body`` and the kind ``Column`` on that one component are exempt from grounding,
    on both paths (on the unguided one CC fires before salvage sees the Card); no reserved id is resampled."""
    assert "Column" not in raw and "__gate_" not in raw, "premise"
    result = UiGate().gate(raw, turn_id="t", canvas_idx=0, final=False)
    got = emitted(result)
    assert got.get("c") == {"id": "c", "component": "Card", "child": "__gate_c_body"}, got
    assert got.get("__gate_c_body") == {
        "id": "__gate_c_body",
        "component": "Column",
        "children": ["a", "b"],
    }
    assert got["root"]["children"] == ["c"]
    assert not [cid for cid in result.resample if cid.startswith("__gate_")], result.resample


NEAR_RESERVED = ["__gate_zz_body", "__gate_a_body", "__gate_c_bodyX"]


@pytest.mark.parametrize("injected", NEAR_RESERVED)
@pytest.mark.parametrize("raw", [NO_COLUMN, NO_COLUMN[:-1]], ids=["strict", "unguided"])
def test_the_exemption_covers_only_this_cards_body_id(
    monkeypatch: pytest.MonkeyPatch, raw: str, injected: str
) -> None:
    """Criterion 2 (mutation 3: the exemption applies to any ``__gate_`` string): a stage that lifts, right after the
    gate-made Column, one more component whose id is reserved-looking but is not ``__gate_c_body`` (its kind and text
    are the model's) is caught by grounding; that id never reaches the renderer (all-or-nothing: the repair is
    reverted, the Card pruned as at the base). The extra component is unreferenced, so pruning cannot hide it: only
    grounding can."""
    real = component_repair_module.repair_components
    seen: list[str] = []

    def injecting(components: Any, validator: Any, *args: Any, **kwargs: Any) -> Any:
        out, edits = real(components, validator, *args, **kwargs)
        changed = []
        for c in out:
            changed.append(c)
            if isinstance(c, dict) and c.get("id") == "__gate_c_body":
                seen.append(c["id"])
                changed.append({"id": injected, "component": "Text", "text": "Alpha"})
        return changed, edits

    monkeypatch.setattr(component_repair_module, "repair_components", injecting)
    if hasattr(ui_gate_module, "repair_components"):
        monkeypatch.setattr(ui_gate_module, "repair_components", injecting)
    result = UiGate().gate(raw, turn_id="t", canvas_idx=0, final=False)
    assert seen, "setup: the stage produced the gate-made Column"
    text_out = json.dumps(list(result.surface.messages)) if result.surface is not None else ""
    assert injected not in text_out, f"criterion 2: {injected} reached the renderer: {text_out}"
