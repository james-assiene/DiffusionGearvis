"""T-0108b: ``repair_components(components, validator) -> (components, edits)``, the content-preserving
component stage of the UI gate (``local_astra/ui/component_repair.py``). Criteria 2-7 and the RB ruling (H-T108b (d)).

The module does not exist at the branch base (08c7776), so it is imported lazily and every row that calls it is
active since T-0108b went in progress (33db104): red on ``NotImplementedError`` until the module exists, green once it is right.

Goldens are hand-written from the rule definitions in docs/tasks/T-0108b.md (*Evidence*, *Rules simulated*, and the
RB shape in *Rulings*), with inputs taken from real components in ``tests/fixtures/a2ui/t0108b_components.jsonl``
(``row_id``) or written here. Nothing is computed by the code under test to make an expected value.

Readings this module makes where the task is open (named in the test-author report):

- ``edits`` is read as a count: an ``int``, or a sized collection of edit records (its ``len``).
- "Output shape": original order, lifted children right after their parent, the input list object back when nothing
  changed (criterion 4's ``is``, applied to the list as well as to each component).
- RB's lifted Text goes right after its Button, like any lifted child.
- A lifted child is the model's object *unchanged* (C4), so an inline child that itself needs a repair is refused.
- RB fires only on a Button whose keys are ``id``, ``component``, ``text``/``label`` and optionally ``variant``,
  ``weight``, ``accessibility``, ``checks``: anything else fails validation after the repair and is refused (3).

Regime: CPU, pure, the vendored A2UI v0.9.1 schema; Hypothesis ``max_examples`` 150, ``deadline=None``, derandomized.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.conductor.ui_instructions import UI_EXAMPLE
from local_astra.ui import A2UIValidator, parse_concatenated_json

MODULE = "local_astra.ui.component_repair"
REPO = Path(__file__).resolve().parents[2]
COMPONENTS = REPO / "tests" / "fixtures" / "a2ui" / "t0108b_components.jsonl"
QUIZ = REPO / "tests" / "fixtures" / "a2ui" / "quiz_example.jsonl"

#: H-T108b (d): the only strings the gate may create
RESERVED_EVENT = "__gate.clicked"
RESERVED_PREFIX = "__gate_"

V = A2UIValidator()


# --- access to the code under test ----------------------------------------------------------------------------


def repair_components_fn() -> Callable[..., Any]:
    try:
        module = importlib.import_module(MODULE)
    except ModuleNotFoundError as missing:
        if missing.name == MODULE:
            raise NotImplementedError(f"T-0108b: {MODULE} does not exist yet") from missing
        raise
    fn: Callable[..., Any] = module.repair_components
    return fn


def edit_count(edits: Any) -> int:
    assert not isinstance(edits, bool), f"edits is a bool: {edits!r}"
    if isinstance(edits, int):
        return edits
    return len(edits)


def repair(components: list[Any]) -> tuple[list[Any], int]:
    before = copy.deepcopy(components)
    out, edits = repair_components_fn()(components, V)
    assert components == before, "repair_components is pure: it must not mutate its input"
    assert isinstance(out, list), f"components come back as a list, not {type(out).__name__}"
    return out, edit_count(edits)


# --- oracles shared with the 9(i) guards (tests/unit/test_ui_gate_structural.py, test_ui_repair_rescore.py) -----


def _resolve(ref: str, doc: str) -> tuple[Any, str]:
    base, _, pointer = ref.partition("#")
    if base:
        doc = "common" if base.endswith("common_types.json") else "catalog"
    node: Any = {"catalog": V.catalog, "common": V.common_types}[doc]
    for part in [p for p in pointer.split("/") if p]:
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node, doc


@cache
def catalog_vocabulary(kind: str) -> frozenset[str]:
    """Every property name the vendored schema of ``kind`` can hold at any depth (C2): the ``properties`` keys
    reachable from ``catalog.components[kind]`` through ``$ref``s, without the catalog's function definitions
    (function arguments are not component properties). An oracle read from the schema, not from the gate."""
    found: set[str] = set()
    seen: set[tuple[str, str]] = set()

    def walk(node: Any, doc: str) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, doc)
            return
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and "anyFunction" not in ref and "/functions/" not in ref:
            target, tdoc = _resolve(ref, doc)
            key = (tdoc, ref.partition("#")[2])
            if key not in seen:
                seen.add(key)
                walk(target, tdoc)
        props = node.get("properties")
        if isinstance(props, dict):
            found.update(props)
        for k, v in node.items():
            if k not in ("$ref", "description", "enum", "default", "const"):
                walk(v, doc)

    walk(V.catalog["components"][kind], "catalog")
    return frozenset(found)


def grounding_strings(value: Any, kind: str | None = None) -> Iterator[str]:
    """Every string an emitted surface carries that grounding must find in the model's bodies: all string values,
    and every dict key except a key in the catalog vocabulary of the enclosing component's kind (criterion 9(i)).
    Keys outside any component (the message envelope) are still yielded."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        own = value.get("component")
        if isinstance(own, str) and own in V.component_kinds:
            kind = own
        vocab = catalog_vocabulary(kind) if kind is not None else frozenset()
        for key, item in value.items():
            if key not in vocab:
                yield key
            yield from grounding_strings(item, kind)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from grounding_strings(item, kind)


def in_bodies(text: str, bodies: list[str]) -> bool:
    escaped = json.dumps(text)[1:-1]
    return any(text in b or escaped in b for b in bodies)


def reserved_ok(text: str, bodies: list[str]) -> bool:
    """H-T108b (d), C7: exactly the reserved namespace is exempt -- the event name, and ``__gate_<id>_label`` where
    ``<id>`` is itself the model's (a Button id it wrote)."""
    if text == RESERVED_EVENT:
        return True
    found = re.fullmatch(r"__gate_(.+)_label", text)
    if found is not None and in_bodies(found.group(1), bodies):
        return True
    # T-0127 re-scope R6: CC's ``__gate_<id>_body`` where ``<id>`` is the model's, and the kind name ``Column`` of
    # that gate-made component. A per-string check cannot see which component carries ``Column``, so it is accepted
    # only when a body holds a Card written with ``children`` (the one thing CC wraps).
    found = re.fullmatch(r"__gate_(.+)_body", text)
    if found is not None:
        return in_bodies(found.group(1), bodies)
    return text == "Column" and any(_CARD_WITH_CHILDREN.search(b) for b in bodies)


#: a Card object written with ``children`` (either key order; a list of id strings holds no braces)
_CARD_WITH_CHILDREN = re.compile(
    r'\{[^{}]*"component"\s*:\s*"Card"[^{}]*"children"|\{[^{}]*"children"[^{}]*"component"\s*:\s*"Card"'
)


# --- small helpers --------------------------------------------------------------------------------------------


@cache
def rows() -> dict[str, dict[str, Any]]:
    """The fixture rows by ``row_id`` (cached; never mutated)."""
    out = {}
    for line in COMPONENTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["row_id"]] = r
    return out


def target(row_id: str) -> dict[str, Any]:
    r = rows()[row_id]
    c: dict[str, Any] = r["components"][r["component_index"]]
    return c


def top(row_id: str, cid: str) -> dict[str, Any]:
    found = [c for c in rows()[row_id]["components"] if isinstance(c, dict) and c.get("id") == cid]
    assert len(found) == 1, f"fixture: {row_id} has {len(found)} top-level {cid!r}"
    return found[0]


def string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)


def keys_of(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_of(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_of(v)}
    return set()


def scalars(value: Any) -> Counter[str]:
    """Non-string scalar leaves, typed (``True`` and ``1`` differ)."""
    if isinstance(value, dict):
        return sum((scalars(v) for v in value.values()), Counter())
    if isinstance(value, list):
        return sum((scalars(v) for v in value), Counter())
    if isinstance(value, str):
        return Counter()
    return Counter([f"{type(value).__name__}:{value!r}"])


def ok(component: Any) -> bool:
    return not V.component_errors(component)


GO = {"event": {"name": "go"}}


# --- criterion 4: valid input is untouched (negative) ----------------------------------------------------------

#: one schema-valid component of every basic-catalog kind, written by hand
VALID_OF_EVERY_KIND: list[dict[str, Any]] = [
    {"id": "t", "component": "Text", "text": "Hello", "variant": "h2"},
    {"id": "img", "component": "Image", "url": "https://example.invalid/a.png"},
    {"id": "ic", "component": "Icon", "name": "search"},
    {"id": "vid", "component": "Video", "url": "https://example.invalid/a.mp4"},
    {"id": "au", "component": "AudioPlayer", "url": "https://example.invalid/a.ogg"},
    {"id": "row", "component": "Row", "children": ["t", "ic"]},
    {"id": "root", "component": "Column", "children": ["row", "lst"]},
    {"id": "lst", "component": "List", "children": {"componentId": "t", "path": "/items"}},
    {"id": "card", "component": "Card", "child": "t"},
    {"id": "tabs", "component": "Tabs", "tabs": [{"title": "One", "child": "t"}]},
    {"id": "modal", "component": "Modal", "trigger": "btn", "content": "t"},
    {"id": "div", "component": "Divider"},
    {"id": "btn", "component": "Button", "child": "t", "action": GO},
    {"id": "tf", "component": "TextField", "label": "From"},
    {"id": "cb", "component": "CheckBox", "label": "Ok", "value": True},
    {
        "id": "cp",
        "component": "ChoicePicker",
        "options": [{"label": "A", "value": "a"}],
        "value": {"path": "/x"},
    },
    {"id": "sl", "component": "Slider", "value": 3, "max": 10},
    {"id": "dt", "component": "DateTimeInput", "value": ""},
]


def test_premise_every_kind_is_covered_and_valid() -> None:
    """Premise (guard): the hand-written list holds one valid component of each catalog kind."""
    assert {c["component"] for c in VALID_OF_EVERY_KIND} == set(V.component_kinds)
    assert [c["id"] for c in VALID_OF_EVERY_KIND if not ok(c)] == []


def _example_lists() -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {"every_kind": VALID_OF_EVERY_KIND}
    for name, text in (("UI_EXAMPLE", UI_EXAMPLE), ("quiz", QUIZ.read_text(encoding="utf-8"))):
        for i, m in enumerate(parse_concatenated_json(text)):
            body = m.get("updateComponents") if isinstance(m, dict) else None
            if isinstance(body, dict):
                out[f"{name}[{i}]"] = body["components"]
    for rid, r in rows().items():
        if r["class"] == "all_valid":
            out[rid] = r["components"]
    return out


@pytest.mark.parametrize("name", sorted(_example_lists()))
def test_valid_components_come_back_as_the_identical_objects(name: str) -> None:
    """Criterion 4 / mutation 2: an all-valid list is returned as the input list object, every component the
    identical object, with 0 edits (every kind, ``UI_EXAMPLE``, the quiz fixture, six real all-valid canvases)."""
    comps = _example_lists()[name]
    assert all(ok(c) for c in comps), f"premise: {name} is all-valid"
    out, edits = repair(comps)
    assert edits == 0, f"{name}: {edits} edits on valid input"
    assert out is comps, f"{name}: nothing changed, so the input list object comes back"
    assert all(a is b for a, b in zip(out, comps, strict=True))


def test_valid_neighbours_of_a_repaired_component_stay_identical() -> None:
    """Criterion 4: in a list that does change, every valid component is still the identical object."""
    comps = [*VALID_OF_EVERY_KIND, {"id": "f", "component": "TextField", "placeholder": "From"}]
    out, edits = repair(comps)
    assert edits >= 1
    for c in VALID_OF_EVERY_KIND:
        assert any(o is c for o in out), f"valid {c['id']} was copied or rewritten"


# --- per-rule goldens (criterion 2), from real components -----------------------------------------------------

#: (inputs, expected output), all expected values by hand from the rule definitions
GOLDEN: dict[str, tuple[Callable[[], list[Any]], list[Any]]] = {
    # BC, id-list form: children ["s_text"] -> child "s_text" (a key move; nothing lifted)
    "BC-id-list": (
        lambda: [target("button_children_id-2"), top("button_children_id-2", "s_text")],
        [
            {"id": "search_btn", "component": "Button", "action": GO, "child": "s_text"},
            {"id": "s_text", "component": "Text", "text": "Search Flights"},
        ],
    ),
    # BC, inline Text under children: referenced and lifted right after the Button
    "BC-inline-children": (
        lambda: [target("button_children_inline-0")],
        [
            {"id": "search", "component": "Button", "action": GO, "child": "btn_text"},
            {"id": "btn_text", "component": "Text", "text": "Search Flights"},
        ],
    ),
    # BC, a non-string child holding one inline Text
    "BC-inline-child": (
        lambda: [target("button_child_inline-1")],
        [
            {"id": "btn", "component": "Button", "action": GO, "child": "btn_text"},
            {"id": "btn_text", "component": "Text", "text": "Search Flights"},
        ],
    ),
    # C6 / adversarial (d) on real data: the inline child's id is already at top level -> referenced, not lifted
    "BC-C6-already-top-level": (
        lambda: [target("button_child_inline-0"), top("button_child_inline-0", "btn_text")],
        [
            {"id": "search_btn", "component": "Button", "action": GO, "child": "btn_text"},
            {"id": "btn_text", "component": "Text", "text": "Search"},
        ],
    ),
    # LI: five inline Texts in a List -> ids, each lifted in order right after the List
    "LI-list": (
        lambda: [target("container_inline_children-1")],
        [
            {"id": "steps", "component": "List", "children": ["s1", "s2", "s3", "s4", "s5"]},
            {"id": "s1", "component": "Text", "text": "Crack and whisk two eggs in a bowl."},
            {"id": "s2", "component": "Text", "text": "Add salt and pepper to taste."},
            {
                "id": "s3",
                "component": "Text",
                "text": "Melt butter in a non-stick pan over medium heat.",
            },
            {"id": "s4", "component": "Text", "text": "Pour in eggs and cook until set."},
            {"id": "s5", "component": "Text", "text": "Fold the omelette in half and serve."},
        ],
    ),
    # TF: placeholder -> label, placeholder removed
    "TF-placeholder": (
        lambda: [target("textfield_placeholder-2")],
        [{"id": "field1", "component": "TextField", "label": "City or airport"}],
    ),
    # TF: text -> label, text removed
    "TF-text": (
        lambda: [target("textfield_text-2")],
        [{"id": "t1", "component": "TextField", "label": "New York"}],
    ),
    # DT: missing value -> ""
    "DT": (
        lambda: [target("datetime_no_value-1")],
        [{"id": "date", "component": "DateTimeInput", "label": "Travel Date", "value": ""}],
    ),
    # CPO + CPV: string options -> {label, value}; missing value -> []
    "CPO-CPV-missing": (
        lambda: [target("choice_string_options-1")],
        [
            {
                "id": "class",
                "component": "ChoicePicker",
                "label": "Class",
                "options": [
                    {"label": "Economy", "value": "Economy"},
                    {"label": "Business", "value": "Business"},
                    {"label": "First Class", "value": "First Class"},
                ],
                "value": [],
            }
        ],
    ),
    # CPO + CPV: string options, string value "1" -> ["1"]
    "CPO-CPV-string": (
        lambda: [target("choice_string_options-2")],
        [
            {
                "id": "passengers",
                "component": "ChoicePicker",
                "label": "Passengers",
                "options": [{"label": s, "value": s} for s in ("1", "2", "3", "4", "5+")],
                "value": ["1"],
            }
        ],
    ),
    # CPV: string value -> [s]
    "CPV-string": (
        lambda: [target("choice_string_value-2")],
        [
            {
                "id": "passengers",
                "component": "ChoicePicker",
                "label": "Passengers",
                "options": [{"label": s, "value": s} for s in ("1", "2", "3")],
                "value": ["1"],
            }
        ],
    ),
    # adversarial (h), real: value "" -> [], never [""]
    "CPV-empty-string": (
        lambda: [target("choice_string_value-1")],
        [
            {
                "id": "class",
                "component": "ChoicePicker",
                "label": "Travel Class",
                "options": [
                    {"label": "Economy", "value": "eco"},
                    {"label": "Business", "value": "bus"},
                    {"label": "First Class", "value": "first"},
                ],
                "value": [],
            }
        ],
    ),
    # CPV: missing value -> []
    "CPV-missing": (
        lambda: [target("choice_no_value-0")],
        [
            {
                "id": "c1",
                "component": "ChoicePicker",
                "options": [
                    {"label": "Economy", "value": "eco"},
                    {"label": "Business", "value": "bus"},
                ],
                "value": [],
            }
        ],
    ),
    # RB (H-T108b (d)), real A0 Button: the reserved Text id, the reserved event, the label verbatim
    "RB-text": (
        lambda: [target("button_text_only-2")],
        [
            {
                "id": "btn",
                "component": "Button",
                "child": "__gate_btn_label",
                "action": {
                    "event": {"name": "__gate.clicked", "context": {"label": "Search Flights"}}
                },
            },
            {"id": "__gate_btn_label", "component": "Text", "text": "Search Flights"},
        ],
    ),
    # RB with ``label`` instead of ``text``, a kept ``variant``, a neighbour before it, and non-ASCII
    "RB-label-variant": (
        lambda: [
            {"id": "h", "component": "Text", "text": "Reise"},
            {"id": "go", "component": "Button", "label": "Søk «fly»", "variant": "primary"},
        ],
        [
            {"id": "h", "component": "Text", "text": "Reise"},
            {
                "id": "go",
                "component": "Button",
                "variant": "primary",
                "child": "__gate_go_label",
                "action": {"event": {"name": "__gate.clicked", "context": {"label": "Søk «fly»"}}},
            },
            {"id": "__gate_go_label", "component": "Text", "text": "Søk «fly»"},
        ],
    ),
    # adversarial (c): an id-only stub then the full Text, one distinct id -> the full one lifted, no duplicate
    "adv-c-stub-then-full": (
        lambda: [
            {
                "id": "b",
                "component": "Button",
                "children": [{"id": "t"}, {"id": "t", "component": "Text", "text": "Go"}],
                "action": GO,
            }
        ],
        [
            {"id": "b", "component": "Button", "action": GO, "child": "t"},
            {"id": "t", "component": "Text", "text": "Go"},
        ],
    ),
    # adversarial (d), LI form: an inline child whose id is already defined at top level is referenced
    "adv-d-LI-already-top-level": (
        lambda: [
            {"id": "s1", "component": "Text", "text": "first"},
            {
                "id": "l",
                "component": "List",
                "children": [
                    {"id": "s1", "component": "Text", "text": "first, again"},
                    {"id": "s2", "component": "Text", "text": "second"},
                ],
            },
        ],
        [
            {"id": "s1", "component": "Text", "text": "first"},
            {"id": "l", "component": "List", "children": ["s1", "s2"]},
            {"id": "s2", "component": "Text", "text": "second"},
        ],
    ),
    # adversarial (g): a valid data binding on ``value`` survives CPO
    "adv-g-binding-kept": (
        lambda: [
            {"id": "c", "component": "ChoicePicker", "options": ["A"], "value": {"path": "/x"}}
        ],
        [
            {
                "id": "c",
                "component": "ChoicePicker",
                "options": [{"label": "A", "value": "A"}],
                "value": {"path": "/x"},
            }
        ],
    ),
    # adversarial (k): the model's placeholder string "label" is kept, as the label's value
    "adv-k-placeholder-is-label": (
        lambda: [{"id": "f", "component": "TextField", "placeholder": "label"}],
        [{"id": "f", "component": "TextField", "label": "label"}],
    ),
    # TF with characters that need escaping
    "TF-escapes": (
        lambda: [{"id": "f", "component": "TextField", "placeholder": 'say "hi" \\ 東京'}],
        [{"id": "f", "component": "TextField", "label": 'say "hi" \\ 東京'}],
    ),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_premise_goldens_are_valid_and_inputs_are_not(name: str) -> None:
    """Premise (guard): every hand-written expected component validates, and each golden input holds at least one
    component the catalog rejects -- so the row tests a repair, not a pass-through."""
    make, want = GOLDEN[name]
    assert [c.get("id") for c in want if not ok(c)] == []
    assert any(not ok(c) for c in make())


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_per_rule_golden(name: str) -> None:
    """Criterion 2 (BC, LI, TF, DT, CPO, CPV, RB): the exact output, in order, with at least one edit; the lifted
    children are the model's objects unchanged (C4)."""
    make, want = GOLDEN[name]
    out, edits = repair(make())
    assert out == want, (
        f"{name}:\n got  {json.dumps(out, ensure_ascii=False)}\n want {json.dumps(want, ensure_ascii=False)}"
    )
    assert edits >= 1, f"{name}: a repair was made, so edits >= 1 (got {edits})"


# --- adversarial rows: the identical object comes back -------------------------------------------------------

UNTOUCHED: dict[str, Callable[[], list[Any]]] = {
    # (a) two distinct ids: BC must not choose (mutation 3)
    "a-two-ids": lambda: [{"id": "b", "component": "Button", "children": ["a", "c"], "action": GO}],
    "a-two-ids-real": lambda: [target("button_children_two-0")],
    # (b) the only child id is the Button's own (mutation 4)
    "b-own-id": lambda: [
        {
            "id": "b",
            "component": "Button",
            "children": [{"id": "b", "component": "Text", "text": "x"}],
            "action": GO,
        }
    ],
    "b-own-id-list": lambda: [{"id": "b", "component": "Button", "children": ["b"], "action": GO}],
    # (e) label and placeholder both present
    "e-label-and-placeholder": lambda: [target("textfield_label_and_placeholder-0")],
    # (f) options mixing strings and objects
    "f-mixed-options": lambda: [
        {
            "id": "c",
            "component": "ChoicePicker",
            "options": ["A", {"label": "B", "value": "b"}],
            "value": [],
        }
    ],
    # (i) a repair that still fails validation leaves the identical object (mutation 1)
    "i-option-without-value-real": lambda: [target("choice_no_value-1")],
    "i-textfield-extra-keys-real": lambda: [target("textfield_extra_keys-1")],
    "i-rb-with-unknown-key-real": lambda: [target("button_text_extra-0")],
    "i-tf-placeholder-and-bad-variant": lambda: [
        {"id": "f", "component": "TextField", "placeholder": "From", "variant": "huge"}
    ],
    "i-bc-lifted-child-invalid": lambda: [
        {
            "id": "b",
            "component": "Button",
            "children": [{"id": "t", "component": "Text", "text": "Go", "variant": "bold"}],
            "action": GO,
        }
    ],
    # BC with an inline child that has no id: nothing to reference
    "bc-inline-without-id-real": lambda: [target("button_children_inline-2")],
    # (j) a List with an inline object that has no ``component``
    "j-inline-without-component": lambda: [
        {"id": "l", "component": "List", "children": [{"id": "s1", "text": "x"}]}
    ],
    "j-list-items-real": lambda: [target("list_items-0")],
    # criterion 7: LI never lifts an object whose id equals its container's; a lift that makes a cycle is refused
    "7-li-own-id": lambda: [
        {
            "id": "l",
            "component": "List",
            "children": [{"id": "l", "component": "Text", "text": "x"}],
        }
    ],
    "7-li-cycle": lambda: [
        {
            "id": "a",
            "component": "Column",
            "children": [{"id": "b", "component": "Column", "children": ["a"]}],
        }
    ],
    # C4: a lifted child is the model's object unchanged, so an inline child that needs its own repair is refused
    "c4-nested-inline": lambda: [
        {
            "id": "l",
            "component": "List",
            "children": [
                {
                    "id": "r",
                    "component": "Row",
                    "children": [{"id": "t", "component": "Text", "text": "x"}],
                }
            ],
        }
    ],
    # RB rows (H-T108b (d))
    "rb-text-and-child": lambda: [
        {"id": "b", "component": "Button", "text": "Go", "child": "t", "action": GO}
    ],
    "rb-text-and-child-no-action": lambda: [
        {"id": "b", "component": "Button", "text": "Go", "child": "t"}
    ],
    "rb-empty-text": lambda: [{"id": "b", "component": "Button", "text": ""}],
    "rb-has-action": lambda: [{"id": "b", "component": "Button", "text": "Go", "action": GO}],
    "rb-own-id-in-reserved-namespace": lambda: [
        {"id": "__gate_x", "component": "Button", "text": "Go"}
    ],
    # rules fire only on the kind they name
    "tf-rule-not-on-other-kinds": lambda: [
        {"id": "x", "component": "Slider", "placeholder": "p", "max": 3}
    ],
    # F3 (reviewer A): a model-written reserved id is never lifted into the gate's namespace
    "f3-bc-inline-reserved-id": lambda: [
        {
            "id": "b",
            "component": "Button",
            "children": [{"id": "__gate_a_label", "component": "Text", "text": "A"}],
            "action": GO,
        }
    ],
    "f3-li-inline-reserved-id": lambda: [
        {
            "id": "c",
            "component": "Column",
            "children": [{"id": "__gate_x_label", "component": "Text", "text": "X"}],
        }
    ],
    # F3: two different inline definitions of one id are refused (BC and LI)
    "f3-bc-two-definitions-one-id": lambda: [
        {
            "id": "b",
            "component": "Button",
            "children": [
                {"id": "t", "component": "Text", "text": "Go"},
                {"id": "t", "component": "Text", "text": "Stop"},
            ],
            "action": GO,
        }
    ],
    "f3-li-two-definitions-one-id": lambda: [
        {
            "id": "c",
            "component": "Column",
            "children": [
                {"id": "t", "component": "Text", "text": "Go"},
                {"id": "t", "component": "Text", "text": "Stop"},
            ],
        }
    ],
    # F3: an id-only object with extra keys and no ``component`` is not referenced
    "f3-bc-inline-without-component": lambda: [
        {"id": "b", "component": "Button", "children": [{"id": "t", "text": "Go"}], "action": GO}
    ],
    # F2 / criterion 7: a cycle closed through two lifts (A -> K -> B -> J -> A) is refused
    "f2-two-lift-cycle": lambda: [
        {"id": "root", "component": "Column", "children": ["A"]},
        {
            "id": "A",
            "component": "Column",
            "children": [{"id": "K", "component": "Column", "children": ["B"]}],
        },
        {
            "id": "B",
            "component": "Column",
            "children": [{"id": "J", "component": "Column", "children": ["A"]}],
        },
    ],
}


@pytest.mark.parametrize("name", sorted(UNTOUCHED))
def test_premise_untouched_inputs_are_invalid(name: str) -> None:
    """Premise (guard): each row's component is catalog-invalid, so leaving it alone is a decision."""
    assert any(not ok(c) for c in UNTOUCHED[name]())


@pytest.mark.parametrize("name", sorted(UNTOUCHED))
def test_adversarial_input_comes_back_identical(name: str) -> None:
    """Adversarial rows (a), (b), (e), (f), (i), (j), criterion 7, C4 and RB: the identical list and objects, 0
    edits. ``edits == 0`` and ``is`` are both required: an equal copy is a rewrite."""
    comps = UNTOUCHED[name]()
    out, edits = repair(comps)
    assert out == comps, f"{name}: changed to {json.dumps(out, ensure_ascii=False)}"
    assert all(a is b for a, b in zip(out, comps, strict=True)), f"{name}: an equal copy came back"
    assert edits == 0, f"{name}: {edits} edits"


def test_model_written_reserved_id_is_not_borrowed_by_rb() -> None:
    """H-T108b (d): the model writes a Text with id ``__gate_btn_label`` next to a text-only Button ``btn``. RB
    must not produce a second ``__gate_btn_label`` nor point the Button at the model's Text: the namespace stays
    the gate's own. Either the Button is left unrepaired or the model's component is refused; ids stay unique and
    any child the Button gets says the Button's own string."""
    model_text = {"id": "__gate_btn_label", "component": "Text", "text": "Evil"}
    button = {"id": "btn", "component": "Button", "text": "Go"}
    out, _ = repair([button, model_text])
    ids = [c.get("id") for c in out]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
    by_id = {c["id"]: c for c in out}
    b = by_id.get("btn")
    assert b is not None
    if "child" in b:
        assert by_id[b["child"]].get("text") == "Go", f"the Button now says {by_id[b['child']]!r}"


# --- C1-C6 over generated components (Hypothesis) ------------------------------------------------------------

#: S: the strings the generated components are built from (non-ASCII, quotes, backslashes, the literal "label")
S = [
    "From",
    "Søk",
    "東京",
    'say "hi"',
    "back\\slash",
    "label",
    "value",
    "Search Flights",
    "1",
    "e.g. New York",
]
text_s = st.sampled_from(S)


@st.composite
def corrupted_canvas(draw: st.DrawFn) -> list[dict[str, Any]]:
    """A list of components, each from S, each either valid or corrupted by one of the observed operators
    (placeholder/text for label, a missing DateTimeInput value, string options, a string/empty/missing ChoicePicker
    value, inline children, a Button's single child under ``children`` or inline, a text-only Button)."""
    n = draw(st.integers(1, 6))
    out: list[dict[str, Any]] = []
    for i in range(n):
        cid, kid = f"c{i}", f"k{i}"
        op = draw(st.integers(0, 10))
        s = draw(text_s)
        if op == 0:
            out.append({"id": cid, "component": "Text", "text": s})
        elif op == 1:
            out.append(
                {
                    "id": cid,
                    "component": "TextField",
                    draw(st.sampled_from(["placeholder", "text"])): s,
                }
            )
        elif op == 2:
            comp: dict[str, Any] = {"id": cid, "component": "DateTimeInput"}
            if draw(st.booleans()):
                comp["label"] = s
            out.append(comp)
        elif op in (3, 4):
            opts = draw(st.lists(text_s, min_size=1, max_size=3))
            comp = {"id": cid, "component": "ChoicePicker"}
            comp["options"] = opts if op == 3 else [{"label": o, "value": o} for o in opts]
            value = draw(st.sampled_from(["missing", "string", "empty", "list"]))
            if value == "string":
                comp["value"] = draw(text_s)
            elif value == "empty":
                comp["value"] = ""
            elif value == "list":
                comp["value"] = [opts[0]]
            out.append(comp)
        elif op == 5:
            out.append(
                {"id": cid, "component": "Button", draw(st.sampled_from(["text", "label"])): s}
            )
        elif op == 6:
            out.append(
                {
                    "id": cid,
                    "component": "Button",
                    "children": [{"id": kid, "component": "Text", "text": s}],
                    "action": {"event": {"name": draw(text_s)}},
                }
            )
        elif op == 7:
            out.append(
                {
                    "id": cid,
                    "component": "Button",
                    "child": {"id": kid, "component": "Text", "text": s},
                    "action": {"event": {"name": draw(text_s)}},
                }
            )
        elif op == 8:
            out.append(
                {
                    "id": cid,
                    "component": draw(st.sampled_from(["Row", "Column", "List"])),
                    "children": [{"id": kid, "component": "Text", "text": s}, "c0"],
                }
            )
        elif op == 9:
            out.append({"id": cid, "component": "Button", "children": [kid], "action": GO})
        else:
            out.append({"id": cid, "component": "TextField", "label": s, "value": draw(text_s)})
    return out


def check_content_preservation(inp: list[Any], out: list[Any]) -> None:
    """C1-C5 and criterion 3 on one (input, output) pair. Every bound is from criterion 5 and the RB ruling."""
    top_in = {c["id"]: c for c in inp if isinstance(c, dict) and isinstance(c.get("id"), str)}
    inline = {
        x["id"]: x
        for c in inp
        if isinstance(c, dict)
        for prop in ("children", "child")
        for x in (c.get(prop) if isinstance(c.get(prop), list) else [c.get(prop)])
        if isinstance(x, dict) and isinstance(x.get("id"), str)
    }
    rb_text = {
        f"{RESERVED_PREFIX}{cid}_label": (c.get("text") if "text" in c else c.get("label"))
        for cid, c in top_in.items()
        if c.get("component") == "Button"
    }
    in_values = set(string_values(inp))
    reserved = {RESERVED_EVENT, *rb_text}
    for c in out:
        # criterion 3: a component either passes through as the model's object, or validates
        assert any(c is x for x in inp) or ok(c), f"criterion 3: emitted an invalid rewrite {c}"
        cid = c.get("id")
        # C4: no id is created (except RB's reserved one); a lifted component is the model's object, unchanged
        if cid in rb_text and cid not in top_in and cid not in inline:
            assert c == {"id": cid, "component": "Text", "text": rb_text[cid]}, f"C4/RB: {c}"
            continue
        assert cid in top_in or cid in inline, f"C4: created id {cid!r}"
        if cid not in top_in:
            assert c == inline[cid], f"C4: lifted {cid!r} was changed: {c} != {inline[cid]}"
            continue
        src = top_in[cid]
        # C1: every string value is the model's own, or "", or the reserved namespace
        for s in string_values(c):
            assert s in in_values or s == "" or s in reserved, f"C1: {s!r} in {c}"
        # C2: added keys are catalog properties of that kind (or label/value in a ChoicePicker option)
        kind = c["component"]
        allowed = catalog_vocabulary(kind) | (
            {"label", "value"} if kind == "ChoicePicker" else set()
        )
        for k in keys_of(c) - keys_of(src):
            assert k in allowed, f"C2: added key {k!r} to {kind} {cid}"
        # C3: no added non-string scalar; "" only as a DateTimeInput value; [] only as a ChoicePicker value
        assert not (scalars(c) - scalars(src)), (
            f"C3: added scalars {scalars(c) - scalars(src)} in {c}"
        )
        added_empty = Counter(string_values(c))[""] - Counter(string_values(src))[""]
        if added_empty > 0:
            assert kind == "DateTimeInput" and c.get("value") == "" and "value" not in src, (
                f"C3: {c}"
            )
        for k, v in c.items():
            if v == [] and src.get(k) != []:
                assert kind == "ChoicePicker" and k == "value", f"C3: added [] at {k} of {c}"
    # C5: no non-empty string value is lost (the generated inputs hold no C6 duplicate)
    out_values = set(string_values(out))
    lost = {s for s in in_values if s and s not in out_values}
    assert not lost, f"C5: lost {lost}"


@settings(
    max_examples=150, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow]
)
@given(corrupted_canvas())
def test_content_preservation_and_idempotence_on_generated_canvases(
    inp: list[dict[str, Any]],
) -> None:
    """Criteria 3, 5 (C1-C5) and 6 over generated canvases; criterion 4 for the valid ones among them."""
    out, edits = repair(inp)
    check_content_preservation(inp, out)
    for c in inp:
        if ok(c):
            assert any(o is c for o in out), f"criterion 4: valid {c['id']} was not passed through"
    if out != inp:
        assert edits >= 1
    if all(ok(c) for c in inp):
        assert out is inp and edits == 0
    again, edits2 = repair(out)
    assert again == out and edits2 == 0, f"criterion 6: a second pass changed {out} into {again}"


def test_content_preservation_and_idempotence_on_real_fixture_lists() -> None:
    """Criteria 3, 5 (C1, C4) and 6 on every component list of the real fixture (55 lists from three runs). C1 and
    C4 here also allow a C6 drop: an inline copy of an id defined at top level may disappear."""
    for rid, r in rows().items():
        comps = r["components"]
        out, _ = repair(comps)
        in_values = set(string_values(comps))
        rb = {
            f"{RESERVED_PREFIX}{c.get('id')}_label" for c in comps if c.get("component") == "Button"
        }
        # T-0127 re-scope R5: CC's documented id and the kind name of its gate-made Column are allowed
        cc = {"__gate_card_body", "Column"}
        for s in string_values(out):
            assert s in in_values or s == "" or s in rb or s == RESERVED_EVENT or s in cc, (
                f"C1 {rid}: {s!r}"
            )
        for c in out:
            assert any(c is x for x in comps) or ok(c), f"criterion 3 {rid}: invalid rewrite {c}"
        ids = [c.get("id") for c in out if isinstance(c, dict)]
        new_ids = set(ids) - {c.get("id") for c in comps if isinstance(c, dict)}
        inline_ids = {x.get("id") for x in _inline_objects(comps)}
        assert new_ids <= inline_ids | rb | {"__gate_card_body"}, (
            f"C4 {rid}: created {new_ids - inline_ids - rb}"
        )
        again, edits2 = repair(out)
        assert again == out and edits2 == 0, f"criterion 6 {rid}: not idempotent"


def _inline_objects(comps: list[Any]) -> Iterator[dict[str, Any]]:
    for c in comps:
        if not isinstance(c, dict):
            continue
        for prop in ("children", "child"):
            v = c.get(prop)
            for x in v if isinstance(v, list) else [v]:
                if isinstance(x, dict):
                    yield x


def test_real_fixture_rule_classes_are_repaired() -> None:
    """Criterion 2 on real data: in every fixture row of a repairable class (where the target's defect is the
    class's only one, checked by hand when the fixture was built), the target id comes back valid. Each class has 3
    real rows."""
    repairable = {
        "textfield_placeholder",
        "datetime_no_value",
        "choice_string_options",
        "choice_string_value",
        "button_text_only",
        "button_children_id",
        "container_inline_children",
    }
    failures = []
    for rid, r in rows().items():
        if r["class"] not in repairable:
            continue
        cid = target(rid)["id"]
        out, _ = repair(r["components"])
        got = [c for c in out if isinstance(c, dict) and c.get("id") == cid]
        if len(got) != 1 or not ok(got[0]):
            failures.append(f"{rid}: {json.dumps(got, ensure_ascii=False)[:200]}")
    assert not failures, "\n".join(failures)


# --- the oracle the 9(i) guards use: per kind, never global (mutation 14) --------------------------------------


def test_catalog_vocabulary_is_per_kind() -> None:
    """Guard on the test oracle: the key exemption of 9(i) is per enclosing kind, never a global list."""
    assert "label" in catalog_vocabulary("TextField") and "placeholder" not in catalog_vocabulary(
        "TextField"
    )
    assert "child" in catalog_vocabulary("Button") and "child" not in catalog_vocabulary("Text")
    assert {"event", "name", "context"} <= catalog_vocabulary("Button")
    assert "options" in catalog_vocabulary("ChoicePicker") and "options" not in catalog_vocabulary(
        "TextField"
    )
    msg = {
        "version": "v0.9.1",
        "updateComponents": {
            "surfaceId": "s",
            "components": [{"id": "f", "component": "TextField", "label": "x", "placeholder": "y"}],
        },
    }
    got = list(grounding_strings(msg))
    assert "placeholder" in got and "label" not in got and "x" in got and "updateComponents" in got


def test_f3_two_buttons_inlining_the_same_text_give_unique_ids() -> None:
    """F3 (reviewer A): two Buttons that both inline Text ``t`` never produce two top-level ``t``; any child a Button
    gets resolves to a Text that says the model's string."""
    inline = {"id": "t", "component": "Text", "text": "Go"}
    comps = [
        {"id": "b1", "component": "Button", "children": [dict(inline)], "action": GO},
        {"id": "b2", "component": "Button", "children": [dict(inline)], "action": GO},
    ]
    out, _ = repair(comps)
    ids = [c.get("id") for c in out]
    assert len(ids) == len(set(ids)), f"duplicate top-level ids: {ids}"
    by_id = {c["id"]: c for c in out}
    for b in ("b1", "b2"):
        child = by_id[b].get("child")
        if isinstance(child, str):
            assert by_id.get(child, {}).get("text") == "Go", f"{b} points at {by_id.get(child)}"
