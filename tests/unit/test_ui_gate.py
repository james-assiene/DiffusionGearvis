"""UI gate: fixer → schema → topology → resample → progressive surface (docs/12 §1.1). Task T-0010."""

from __future__ import annotations

import json
import re

import json_repair
import pytest

from local_astra.conductor.ui_gate import UiGate
from local_astra.contracts.protocol import AstraUiSurface
from local_astra.ui import A2UIValidator, parse_concatenated_json


@pytest.fixture
def gate() -> UiGate:  # function-scoped: per-turn resample budgets must not leak between tests
    return UiGate(validator=A2UIValidator())


@pytest.fixture(scope="module")
def quiz(quiz_example_text) -> list:
    return parse_concatenated_json(quiz_example_text)


def test_valid_surface_passes_unchanged_with_zero_repairs(gate, quiz_example_text, quiz):
    """Strict parse succeeds → the fixer is not consulted → repaired == 0 by definition.

    T-0010 A7: the one thing the gate *does* rewrite is ``surfaceId`` (the renderer keys on the
    id inside the messages, not on the envelope), so the comparison is equality modulo that key.
    """
    r = gate.gate(quiz_example_text, turn_id="t1", canvas_idx=2, final=True)
    assert r.report.ok and r.repaired == 0 and r.resample == ()
    assert isinstance(r.surface, AstraUiSurface)
    assert _blind_surface_ids(list(r.surface.messages)) == _blind_surface_ids(quiz)
    # A7: every surfaceId inside the messages is the gate's id for this turn, not the model's
    assert _surface_ids_in(r.surface.messages) == {r.surface.surface_id}
    assert r.surface.final is True and r.surface.canvas_idx == 2


def test_truncated_canvas_renders_the_reachable_prefix_as_non_final(gate, quiz_example_text):
    cut = quiz_example_text[: quiz_example_text.index('"id":"q3"')]
    r = gate.gate(cut, turn_id="t1", canvas_idx=0, final=False)
    assert r.surface is not None and r.surface.final is False
    assert r.repaired >= 1
    comps = [
        c["id"]
        for m in r.surface.messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
    ]
    assert "root" in comps and "q1" in comps and "q3" not in comps
    # the Column that referenced q3.. must not reference missing ids after pruning
    assert gate.validator.validate_messages(list(r.surface.messages)).ok


def test_invalid_component_is_pruned_listed_for_resample_and_parent_reference_dropped(gate, quiz):
    bad = json.loads(json.dumps(quiz))
    bad[1]["updateComponents"]["components"][2]["variant"] = "huge"  # quiz_title
    raw = "\n".join(json.dumps(m) for m in bad)
    r = gate.gate(raw, turn_id="t1", canvas_idx=1, final=True)
    assert r.resample == ("quiz_title",)
    assert r.surface is not None
    ids = {
        c["id"]
        for m in r.surface.messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
    }
    assert "quiz_title" not in ids
    quiz_col = next(
        c
        for m in r.surface.messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
        if c["id"] == "quiz"
    )
    assert "quiz_title" not in quiz_col["children"]


def test_resamples_are_capped(gate, quiz):
    bad = json.loads(json.dumps(quiz))
    for c in bad[1]["updateComponents"]["components"]:
        if c["component"] == "Text":
            c["variant"] = "huge"
    r = gate.gate("\n".join(json.dumps(m) for m in bad), turn_id="t1", canvas_idx=0, final=True)
    assert len(r.resample) == gate.max_resamples


def test_unrepairable_garbage_yields_no_surface_but_a_report(gate):
    r = gate.gate("total nonsense without braces", turn_id="t1", canvas_idx=0, final=True)
    assert r.surface is None and not r.report.ok


def test_a_message_missing_a_required_envelope_key_yields_no_surface(gate, quiz):
    """C2: ``createSurface`` without ``catalogId`` fails the *envelope* schema, which pruning
    cannot repair (there is no component to drop), so nothing is rendered. This is valid JSON, so
    the fixer is never consulted — the fixer's own fallbacks are pinned by the two
    ``…falls_back_to_unguided_repair`` tests below, not here."""
    bad = json.loads(json.dumps(quiz))
    del bad[0]["createSurface"]["catalogId"]
    r = gate.gate("\n".join(json.dumps(m) for m in bad), turn_id="t1", canvas_idx=0, final=True)
    assert r.surface is None and r.report.schema_errors
    assert r.repaired == 0  # strict parse succeeded: json_repair never ran


def test_surface_id_is_stable_per_turn(gate, quiz_example_text):
    a = gate.gate(quiz_example_text, turn_id="t9", canvas_idx=0, final=False)
    b = gate.gate(quiz_example_text, turn_id="t9", canvas_idx=1, final=True)
    assert a.surface is not None and b.surface is not None
    assert a.surface.surface_id == b.surface.surface_id
    # A7: what the renderer keys on is the id inside the messages; canvas 1 must morph canvas 0
    assert _surface_ids_in(a.surface.messages) == {a.surface.surface_id}
    assert _surface_ids_in(b.surface.messages) == {b.surface.surface_id}


# --- added by the test-author at /implement step 3 -------------------------------------------
# One test per acceptance criterion of docs/tasks/T-0010.md that the pending set left uncovered,
# plus the adversarial cases for "nothing unvalidated reaches @a2ui/lit" (CLAUDE.md).


def _ids(surface: AstraUiSurface) -> list[str]:
    """Component ids the HUD would mount, in message order."""
    return [
        c["id"]
        for m in surface.messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    ]


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _surface_ids_in(messages) -> set:
    """Every ``surfaceId`` value anywhere in the messages — what the renderer actually keys on."""
    out: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "surfaceId" and isinstance(value, str):
                    out.add(value)
                walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                walk(value)

    walk(messages)
    return out


def _blind_surface_ids(obj):
    """``obj`` with every ``surfaceId`` value replaced, so two messages can be compared modulo
    the A7 rewrite (which is the *only* edit the gate is allowed to make to a valid canvas)."""
    if isinstance(obj, dict):
        return {k: ("<sid>" if k == "surfaceId" else _blind_surface_ids(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_blind_surface_ids(v) for v in obj]
    return obj


def _tree(*surfaces) -> dict:
    """Components the renderer holds after applying these surfaces in order (id → component).

    A2UI replaces a component by id, so a later definition wins; this is the renderer's view of a
    multi-canvas turn and the only honest place to assert progressive-morph outcomes.
    """
    tree: dict[str, dict] = {}
    for surface in surfaces:
        for m in surface.messages:
            if "updateComponents" not in m:
                continue
            for c in m["updateComponents"]["components"]:
                if isinstance(c, dict) and isinstance(c.get("id"), str):
                    tree[c["id"]] = c
    return tree


def _names_index(problems, index: int) -> bool:
    """Does some problem point at the component at ``index``? The codebase labels an id-less
    component ``#<index>`` (``invalid_component_ids``, ``check_topology``) and schema errors use
    ``components/<index>``; any of those spellings counts."""
    return any(re.search(rf"(#|index |components/)\s*{index}\b", p) for p in problems)


def _resolves(document, ref: str) -> bool:
    """Does a local ``$ref`` point at something in ``document``? (RFC 6901, by hand.)"""
    node = document
    for token in ref.lstrip("#").split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"


def _create(surface_id: str = "s") -> dict:
    return {
        "version": "v0.9.1",
        "createSurface": {"surfaceId": surface_id, "catalogId": CATALOG_ID},
    }


def _update(components: list, surface_id: str = "s") -> dict:
    return {
        "version": "v0.9.1",
        "updateComponents": {"surfaceId": surface_id, "components": components},
    }


def _canvas(*messages) -> str:
    return "\n".join(json.dumps(m) for m in messages)


# --- criterion 1: the fixer ------------------------------------------------------------------


def test_repair_uses_the_strict_parser_first_and_reports_zero_repairs(
    gate, quiz_example_text, quiz
):
    """C1: valid input never reaches json_repair, so ``repaired`` is 0 and nothing is rewritten."""
    messages, repaired = gate.repair(quiz_example_text)
    assert repaired == 0
    assert messages == quiz  # byte-for-byte the same values the strict parser produces


def test_repair_recovers_every_complete_component_of_a_truncated_canvas(gate, quiz_example_text):
    """C1/C3: fallback repair; ``repaired`` counts fixer edits (log length), so it is >= 1 here."""
    cut = quiz_example_text[: quiz_example_text.index('"id":"q3"')]
    messages, repaired = gate.repair(cut)
    assert repaired >= 1  # at least the unclosed array and the unclosed object
    ids = [
        c["id"]
        for m in messages
        if isinstance(m, dict) and "updateComponents" in m
        for c in m["updateComponents"]["components"]
    ]
    # oracle: the ids literally present in the prefix — the fixture stops just before q3's id,
    # so exactly the 11 components root..q2_f are complete.
    complete = re.findall(r'"id":"([^"]+)"', cut)
    assert len(complete) == 11 and complete[-1] == "q2_f"
    assert ids == complete


def test_repair_never_invents_content(gate, quiz_example_text):
    """Adversarial: a fixer that hallucinates a component would show up as an unknown id or an
    unknown string. Every id and every string in the repaired output must come from the input."""
    cut = quiz_example_text[: quiz_example_text.index('"id":"q3"')]
    noisy = "```json\n" + cut  # the fence the model sometimes emits, never closed
    messages, _ = gate.repair(noisy)
    raw_ids = set(re.findall(r'"id":"([^"]+)"', noisy))
    got_ids = {
        c["id"]
        for m in messages
        if isinstance(m, dict) and "updateComponents" in m
        for c in m["updateComponents"]["components"]
    }
    assert got_ids and got_ids <= raw_ids
    invented = [s for s in _strings(messages) if s not in noisy]
    assert invented == []


# --- criterion 2: only a validated surface reaches the renderer -------------------------------


def test_a_cycle_yields_no_surface_and_the_report_names_it(gate, quiz):
    """Adversarial, T-0010 A2: valid JSON, schema-valid components, cyclic topology
    (quiz -> q1 -> quiz). Pruning is defined over schema-*invalid* components, so there is nothing
    to prune and the cycle is a topology failure: **no** surface, and the report names the cycle.
    An implementation that strips the back-edge, revalidates and emits the rest fails here — that
    is the point of the ruling, not an accident of this test."""
    bad = json.loads(json.dumps(quiz))
    for c in bad[1]["updateComponents"]["components"]:
        if c["id"] == "q1":
            c["children"] = ["quiz"]
    r = gate.gate("\n".join(json.dumps(m) for m in bad), turn_id="t1", canvas_idx=0, final=True)
    assert r.surface is None, "A2: a cycle is not broken heuristically, the surface is withheld"
    assert not r.report.ok
    named = [p for p in r.report.problems if "cycle" in p]
    assert named, r.report.problems
    assert any("quiz" in p and "q1" in p for p in named), named  # the cycle itself is named


def test_a_dangling_parent_reference_never_reaches_the_renderer(gate, quiz):
    """Adversarial: 'quiz' keeps a child id whose component was never emitted ('score')."""
    bad = json.loads(json.dumps(quiz))
    comps = bad[1]["updateComponents"]["components"]
    bad[1]["updateComponents"]["components"] = [c for c in comps if c["id"] != "score"]
    r = gate.gate("\n".join(json.dumps(m) for m in bad), turn_id="t1", canvas_idx=0, final=True)
    if r.surface is None:
        assert not r.report.ok
    else:
        report = gate.validator.validate_messages(list(r.surface.messages))
        assert report.ok, (report.schema_errors, report.problems)
        ids = _ids(r.surface)
        assert "score" not in ids
        assert {"root", "quiz", "q1", "deck"} <= set(ids)  # the rest of the surface survived


def test_valid_json_that_is_not_an_a2ui_message_yields_no_surface(gate):
    """Adversarial: the fixer happily returns an object; the envelope schema must still veto it."""
    r = gate.gate('{"hello":"world","components":[]}', turn_id="t1", canvas_idx=0, final=True)
    assert r.surface is None
    assert not r.report.ok and r.report.schema_errors


def test_two_bad_components_are_dropped_and_the_other_36_survive(gate, quiz):
    """C2: per-component granularity — one (or two) bad components must not discard the surface.
    Expected survivors are the fixture's 38 ids minus the two that were made invalid."""
    bad = json.loads(json.dumps(quiz))
    for c in bad[1]["updateComponents"]["components"]:
        if c["id"] in ("quiz_title", "score"):
            c["variant"] = "huge"  # not in the Text variant enum
    r = gate.gate("\n".join(json.dumps(m) for m in bad), turn_id="t1", canvas_idx=0, final=True)
    assert r.repaired == 0  # the input is valid JSON: the strict parser handles it
    assert set(r.resample) == {"quiz_title", "score"}
    assert len(r.resample) <= gate.max_resamples
    assert r.surface is not None
    expected = {c["id"] for c in quiz[1]["updateComponents"]["components"]} - {
        "quiz_title",
        "score",
    }
    assert len(expected) == 36
    assert set(_ids(r.surface)) == expected
    report = gate.validator.validate_messages(list(r.surface.messages))
    assert report.ok, (report.schema_errors, report.problems)


# --- criterion 3: progressive rendering -------------------------------------------------------


def test_later_canvases_are_validated_against_the_components_already_accepted(gate, quiz):
    """C3: canvas k is validated with ``prior=`` the accepted components. Canvas 1 here carries
    neither createSurface nor 'root', so without ``prior`` it is 'updateComponents before
    createSurface' plus a missing root; with it, the surface completes and 'b' resolves."""
    catalog_id = quiz[0]["createSurface"]["catalogId"]
    canvas0 = "\n".join(
        json.dumps(m)
        for m in [
            {"version": "v0.9.1", "createSurface": {"surfaceId": "s", "catalogId": catalog_id}},
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["a", "b"]},
                        {"id": "a", "component": "Text", "text": "first"},
                    ],
                },
            },
        ]
    )
    canvas1 = json.dumps(
        {
            "version": "v0.9.1",
            "updateComponents": {
                "surfaceId": "s",
                "components": [{"id": "b", "component": "Text", "text": "second"}],
            },
        }
    )
    r0 = gate.gate(canvas0, turn_id="t7", canvas_idx=0, final=False)
    # forward reference to 'b' is a warning while final=False, so canvas 0 still renders
    assert r0.report.ok, (r0.report.schema_errors, r0.report.problems)
    assert r0.surface is not None and r0.surface.final is False and r0.surface.canvas_idx == 0
    assert "a" in _ids(r0.surface)
    r1 = gate.gate(canvas1, turn_id="t7", canvas_idx=1, final=True)
    assert r1.report.ok, (r1.report.schema_errors, r1.report.problems)
    assert r1.surface is not None and r1.surface.final is True and r1.surface.canvas_idx == 1
    assert "b" in _ids(r1.surface)


def test_surface_id_differs_between_turns(gate, quiz_example_text):
    """Twin of test_surface_id_is_stable_per_turn: stability within a turn must not be bought
    with a constant id, or turn n+1 would morph turn n's stale components instead of mounting.

    T-0010 A7: the renderer never sees ``AstraUiSurface.surface_id`` — it sees the ``surfaceId``
    *inside* the messages. A model that writes ``"lesson_1"`` on both turns re-mounts turn 9's
    surface in turn 10 unless the gate rewrites it, so the ids are inspected where they are used.
    """
    a = gate.gate(quiz_example_text, turn_id="t9", canvas_idx=0, final=True)
    b = gate.gate(quiz_example_text, turn_id="t10", canvas_idx=0, final=True)
    assert a.surface is not None and b.surface is not None
    assert a.surface.surface_id != b.surface.surface_id
    inner_a, inner_b = _surface_ids_in(a.surface.messages), _surface_ids_in(b.surface.messages)
    assert inner_a == {a.surface.surface_id} and inner_b == {b.surface.surface_id}
    assert inner_a.isdisjoint(inner_b)


# --- T-0010 rulings A7-A9 and the reviewer's test gaps T1-T6 ----------------------------------
# Each test below names the ruling it pins. The reviewer's reproduction inputs are used verbatim
# where the task file quotes them (the '"id":"root"' cut, the id-less component, the Card delta,
# the cycle).


# --- T1: the fixer is grounded in the model's output ------------------------------------------


def test_the_fixer_may_not_invent_a_component_kind_for_a_bare_id(gate, quiz_example_text):
    """T1 / C1: the canvas is cut right after ``"id":"root"``, so the only thing the model wrote
    about that component is its id. The fixer is allowed to *close* and to *drop*; it is not
    allowed to add. An ungrounded schema-guided salvage fills the required ``component`` with a
    default (``Divider``, ``axis: "horizontal"``) — text that is nowhere in the input — and the
    HUD would render a divider the model never asked for."""
    cut = quiz_example_text[: quiz_example_text.index('"id":"root"') + len('"id":"root"')]
    messages, repaired = gate.repair(cut)
    assert repaired >= 1  # the canvas is malformed, so the fixer really ran
    components = [
        c for m in messages if "updateComponents" in m for c in m["updateComponents"]["components"]
    ]
    # oracle: the input contains exactly one component and exactly one key of it
    assert components == [{"id": "root"}], components
    assert [s for s in _strings(messages) if s not in cut] == []


def test_an_ungrounded_component_is_pruned_and_resampled_never_rendered(gate, quiz_example_text):
    """T1 / C2: the consequence of the test above at the gate. ``{"id": "root"}`` fails the
    catalog schema, so it is pruned and re-requested; a fixer that invented a ``Divider`` would
    hand the renderer a schema-valid component with an empty resample list instead."""
    cut = quiz_example_text[: quiz_example_text.index('"id":"root"') + len('"id":"root"')]
    r = gate.gate(cut, turn_id="t1", canvas_idx=0, final=False)
    assert r.resample == ("root",)
    if r.surface is not None:
        assert _ids(r.surface) == []
        assert gate.validator.validate_messages(list(r.surface.messages), final=False).ok


# --- T2/T3: schema-guided salvage, and what happens when it cannot be used --------------------


BAD_ENUM_CANVAS = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"' + CATALOG_ID + '"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":[\n'
    '{"id":"root","component":"Column","children":["a","bad","c"]},\n'
    '{"id":"a","component":"Text","text":"first"},\n'
    '{"id":"bad","component":"Text","variant":"%s","text":"second"},\n'
    '{"id":"c","component":"Text","text":"third'
)  # truncated mid-string: the strict parser fails, so the fixer path is the one under test


def test_schema_guided_salvage_drops_a_catalog_invalid_component_during_repair(gate):
    """T2 / C1: on the *fixer* path the components array is repaired against the derived catalog
    schema, and salvage mode drops an item that violates it (pinned in
    test_json_repair_contract.py::test_salvage_mode_drops_invalid_array_items_...). So ``bad``
    never reaches validation, it is reported for resample, and the edits are counted: an
    unguided-only repair keeps it and logs strictly fewer edits."""
    messages, repaired = gate.repair(BAD_ENUM_CANVAS % "huge")
    ids = [
        c["id"]
        for m in messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
    ]
    assert ids == ["root", "a", "c"]  # 'bad' was dropped by the schema-guided fixer, not by pruning
    control, control_repaired = UiGate(validator=A2UIValidator()).repair(BAD_ENUM_CANVAS % "body")
    control_ids = [
        c["id"]
        for m in control
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
    ]
    assert control_ids == ["root", "a", "bad", "c"]  # same canvas, valid variant: nothing dropped
    assert repaired > control_repaired  # the salvage edits are part of the repair count (C3)
    r = UiGate(validator=A2UIValidator()).gate(
        BAD_ENUM_CANVAS % "huge", turn_id="t1", canvas_idx=0, final=False
    )
    assert "bad" in r.resample  # dropped by the fixer is still "re-request this component"
    assert r.surface is not None and "bad" not in _ids(r.surface)


#: a catalog the *validator* resolves (components are validated by kind, ``#/components/Note``)
#: but whose ``anyComponent`` the derived fixer schema cannot: the pointer survives localisation
#: (its base ends in ``catalog.json``) and lands on a ``$defs`` entry that does not exist, so
#: json_repair raises ``PointerToNowhere``, an ``Unresolvable``. A *remote* ``$ref`` reproduces it
#: too, but `referencing` then tries to *retrieve* the URI — DNS and HTTP from a unit test — so
#: the dangling local pointer is the form used here (see the report).
UNRESOLVABLE_CATALOG = {
    "$id": "https://example.test/edu/catalog.json",
    "catalogId": "https://example.test/edu/catalog.json",
    "$defs": {"anyComponent": {"$ref": "https://example.test/edu/catalog.json#/$defs/missing"}},
    "components": {
        "Note": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "component": {"const": "Note"},
                "text": {"type": "string"},
            },
            "required": ["id", "component", "text"],
            "additionalProperties": False,
        }
    },
}

NOTE_CANVAS = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s",'
    '"catalogId":"https://example.test/edu/catalog.json"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":[\n'
    '{"id":"root","component":"Note","text":"hello'
)

#: the same canvas, well-formed but with a component the catalog rejects (``text`` must be a
#: string), behind an unclosed fence so the strict parser fails. Both conditions are needed to
#: reach the fixer's failure arm: A11 salvages only the components that failed, so a *valid*
#: component never calls the fixer and the ``Unresolvable`` is never raised.
BAD_NOTE_CANVAS = (
    "```json\n"
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s",'
    '"catalogId":"https://example.test/edu/catalog.json"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":[\n'
    '{"id":"root","component":"Note","text":123}]}}'
)


def test_a_fixer_schema_the_fixer_cannot_use_falls_back_to_unguided_repair():
    """T2 / C1: a failure *inside* the fixer is a fallback, never an exception out of the gate.

    The derived schema carries a ``$ref`` json_repair cannot follow, so salvaging the broken
    component raises ``Unresolvable``. The gate must absorb it, keep the component exactly as the
    model wrote it (unguided repair), and then treat it like any other component the catalog
    rejects: pruned, re-requested, and the rest of the surface still rendered.
    """
    gate = UiGate(validator=A2UIValidator(catalog=UNRESOLVABLE_CATALOG))
    messages, repaired = gate.repair(BAD_NOTE_CANVAS)
    assert repaired >= 1  # the fence: the fixer path really was taken
    components = [
        c for m in messages if "updateComponents" in m for c in m["updateComponents"]["components"]
    ]
    # the fixer could not run, so what comes back is the model's own value, not a salvage default
    assert components == [{"id": "root", "component": "Note", "text": 123}]

    r = UiGate(validator=A2UIValidator(catalog=UNRESOLVABLE_CATALOG)).gate(
        BAD_NOTE_CANVAS, turn_id="t1", canvas_idx=0, final=True
    )
    assert r.report.ok, (r.report.schema_errors, r.report.problems)
    assert r.surface is not None
    assert r.resample == ("root",)  # re-requested, because salvage could not fix it
    assert _ids(r.surface) == []  # and nothing unvalidated reached the renderer


def test_an_unusable_fixer_schema_still_lets_a_valid_component_through():
    """Twin of the above: the same unusable schema must not cost a component that needs no
    repair anything — A11(a) means the fixer is not even consulted for it."""
    gate = UiGate(validator=A2UIValidator(catalog=UNRESOLVABLE_CATALOG))
    r = gate.gate(NOTE_CANVAS, turn_id="t1", canvas_idx=0, final=True)
    assert r.surface is not None and r.report.ok
    assert _ids(r.surface) == ["root"]
    assert r.repaired >= 1


def test_a_catalog_without_any_component_falls_back_to_unguided_repair():
    """T2 / C1, the other fixer-schema failure: ``components_array_schema`` raises ``KeyError``
    when the catalog has no ``anyComponent``, and the gate repairs unguided instead of dying."""
    catalog = json.loads(json.dumps(UNRESOLVABLE_CATALOG))
    catalog["$defs"] = {}
    gate = UiGate(validator=A2UIValidator(catalog=catalog))
    r = gate.gate(NOTE_CANVAS, turn_id="t1", canvas_idx=0, final=True)
    assert r.surface is not None and r.report.ok and _ids(r.surface) == ["root"]


def test_the_derived_fixer_schema_is_self_contained_and_salvages_a_bad_enum():
    """T3 / C1 end to end on ``fixer_schema``: the derived document has no
    ``unevaluatedProperties`` and only local ``$ref``s (that is what C1 asks for, and what
    json_repair can actually follow), and with it salvage drops the component whose ``variant`` is
    not in the enum while keeping the valid ones — including the properties the catalog's
    ``allOf`` branches contribute (``justify`` lives in one of those), which is the whole reason
    the derivation exists. The survivors are re-checked against the *authoritative* validator, so
    a derivation that let junk through would fail here rather than at the renderer."""
    # imported here, not at module level: ``fixer_schema`` ships with the T-0010 implementation
    # and has no stub on main, so a top-level import would break collection of this whole file.
    from local_astra.ui.fixer_schema import components_array_schema

    validator = A2UIValidator()
    schema = components_array_schema(validator.catalog, validator.common_types)
    text = json.dumps(schema)
    assert "unevaluatedProperties" not in text
    refs = sorted(set(re.findall(r'"\$ref": "([^"]+)"', text)))
    assert [r for r in refs if not r.startswith("#")] == []
    # "local $refs only" is worth nothing if the pointers dangle: the catalog's components *and*
    # its functions (``anyFunction`` → ``#/functions/<name>``) must travel with the document.
    assert [r for r in refs if not _resolves(schema, r)] == []
    assert [r for r in refs if r.startswith("#/functions/")]  # the section is actually referenced
    salvaged, log = json_repair.loads(
        json.dumps(
            [
                {"id": "t1", "component": "Text", "variant": "h2", "text": "keep"},
                {"id": "t2", "component": "Text", "variant": "huge", "text": "drop"},
                {"id": "col", "component": "Column", "children": ["t1", "t2"], "justify": "center"},
            ]
        ),
        schema=schema,
        schema_repair_mode="salvage",
        logging=True,
    )
    assert [c["id"] for c in salvaged] == ["t1", "col"]
    assert salvaged[0] == {"id": "t1", "component": "Text", "variant": "h2", "text": "keep"}
    assert salvaged[1]["children"] == ["t1", "t2"] and salvaged[1]["justify"] == "center"
    assert log  # the drop is logged, so the gate can count it as a repair
    assert validator.component_errors(salvaged[0]) == []
    assert validator.component_errors(salvaged[1]) == []


# --- T5: progressive morph (delta discipline) -------------------------------------------------


def test_a_pruned_list_reference_is_restored_when_the_child_arrives(gate):
    """T5 / C3: canvas 0 references ``b``, which has not been generated yet; the reference is
    pruned so canvas 0 validates. When ``b`` arrives on canvas 1 the parent must be re-sent with
    the reference restored (A2UI replaces by id), or ``b`` stays an orphan the HUD never mounts."""
    c0 = _canvas(
        _create(),
        _update(
            [
                {"id": "root", "component": "Column", "children": ["a", "b"]},
                {"id": "a", "component": "Text", "text": "first"},
            ]
        ),
    )
    c1 = _canvas(_update([{"id": "b", "component": "Text", "text": "second"}]))
    r0 = gate.gate(c0, turn_id="t20", canvas_idx=0, final=False)
    r1 = gate.gate(c1, turn_id="t20", canvas_idx=1, final=True)
    assert r0.surface is not None and r1.surface is not None
    assert _tree(r0.surface)["root"]["children"] == ["a"]  # pruned while 'b' was missing
    tree = _tree(r0.surface, r1.surface)
    assert tree["root"]["children"] == ["a", "b"]  # restored on the canvas that brought 'b'
    report = gate.validator.validate_messages(
        [*r0.surface.messages, *r1.surface.messages], final=True
    )
    assert report.ok, (report.schema_errors, report.problems)
    assert report.topology is not None and set(report.topology.reachable) == {"root", "a", "b"}


def test_a_container_dropped_for_a_missing_child_is_re_emitted_when_the_child_arrives(gate):
    """T5 / T-0010 A9: ``Card.child`` is required, so a Card whose child has not arrived cannot be
    pruned into validity — it is dropped whole on canvas 0. When the child arrives on canvas 1 the
    Card must come back (and its parent's reference with it), exactly as the ``children: [...]``
    path does; otherwise the model's delta is lost and the card never renders."""
    c0 = _canvas(
        _create(),
        _update(
            [
                {"id": "root", "component": "Column", "children": ["card"]},
                {"id": "card", "component": "Card", "child": "body"},
            ]
        ),
    )
    c1 = _canvas(_update([{"id": "body", "component": "Text", "text": "late"}]))
    r0 = gate.gate(c0, turn_id="t21", canvas_idx=0, final=False)
    r1 = gate.gate(c1, turn_id="t21", canvas_idx=1, final=True)
    assert r0.surface is not None and r1.surface is not None
    assert "card" not in _ids(r0.surface)  # Card.child is required: nothing to render yet
    tree = _tree(r0.surface, r1.surface)
    assert "card" in tree, "A9: the Card must be re-emitted on the canvas that brings its child"
    assert tree["card"]["child"] == "body"
    assert tree["root"]["children"] == ["card"]
    report = gate.validator.validate_messages(
        [*r0.surface.messages, *r1.surface.messages], final=True
    )
    assert report.ok, (report.schema_errors, report.problems)
    assert report.topology is not None and set(report.topology.reachable) == {
        "root",
        "card",
        "body",
    }


# --- T6: A7 (one surface per turn, id rewritten) and A8 (id-less components) -------------------


def test_a_second_surface_in_one_canvas_is_dropped_and_named(gate):
    """T6 / A7: v1 renders one surface per turn. A canvas that creates two keeps the first and
    drops the second's messages, naming it — merging two surfaces under one id would silently
    collide their roots (both are called ``root`` here)."""
    raw = _canvas(
        _create("s1"),
        _update([{"id": "root", "component": "Text", "text": "one"}], "s1"),
        _create("s2"),
        _update([{"id": "root", "component": "Text", "text": "two"}], "s2"),
    )
    r = gate.gate(raw, turn_id="t3", canvas_idx=0, final=True)
    assert r.surface is not None
    assert _surface_ids_in(r.surface.messages) == {r.surface.surface_id}
    texts = [c.get("text") for c in _tree(r.surface).values()]
    assert texts == ["one"]  # the second surface's components are not merged in
    assert any("s2" in p for p in r.report.problems), r.report.problems


def test_an_id_less_component_is_dropped_by_index_and_the_surface_survives(gate, quiz):
    """T6 / A8: a component the model emitted without an ``id`` cannot be resampled (there is
    nothing to name in the request) but it must not take the other 37 components down with it.
    It is dropped by index, the drop is reported, and the surface is still emitted."""
    bad = json.loads(json.dumps(quiz))
    comps = bad[1]["updateComponents"]["components"]
    index = next(i for i, c in enumerate(comps) if c["id"] == "q3_p")
    del comps[index]["id"]
    r = gate.gate(_canvas(*bad), turn_id="t1", canvas_idx=0, final=True)
    assert r.surface is not None, r.report.problems + tuple(str(e) for e in r.report.schema_errors)
    expected = {c["id"] for c in quiz[1]["updateComponents"]["components"]} - {"q3_p"}
    assert len(expected) == 37
    assert set(_ids(r.surface)) == expected
    assert all(isinstance(c.get("id"), str) and c["id"] for c in _tree(r.surface).values())
    assert not [cid for cid in r.resample if cid.startswith("#")]  # nothing to re-request
    assert _names_index(r.report.problems, index), r.report.problems
    report = gate.validator.validate_messages(list(r.surface.messages))
    assert report.ok, (report.schema_errors, report.problems)


# --- T-0010 A11: salvage does not touch what already validated, grounding covers every leaf -----


def test_salvage_never_rewrites_a_component_that_already_validated(gate, quiz_example_text, quiz):
    """A11(a): one invalid neighbour must not cost the *valid* components their content.

    The canvas is the fixture with ``quiz_title``'s variant broken and an unclosed code fence in
    front, so the strict parser fails and the fixer path — the only one that salvages — is the one
    under test. Schema-guided salvage of the whole array rewrites the neighbours too: ``q1_c``'s
    data binding ``{"path": "/answers/q1"}`` comes back as ``[]`` and it gains ``filterable`` /
    ``displayStyle`` the model never wrote, i.e. a quiz that renders fine and silently stops
    recording answers. Ruling: a component that validates before salvage is passed through
    byte-identical; salvage runs only over the ones that failed.
    """
    bad = json.loads(json.dumps(quiz))
    for c in bad[1]["updateComponents"]["components"]:
        if c["id"] == "quiz_title":
            c["variant"] = "huge"  # not in the Text variant enum
    raw = "```json\n" + _canvas(*bad)  # unclosed fence: json.JSONDecodeError, so the fixer runs
    messages, repaired = gate.repair(raw)
    assert repaired >= 1

    original = {c["id"]: c for c in quiz[1]["updateComponents"]["components"]}
    got = {
        c["id"]: c
        for m in messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
        if isinstance(c.get("id"), str)
    }
    q1c = got["q1_c"]
    assert q1c["value"] == {"path": "/answers/q1"}  # the data binding, not a salvage default
    assert "filterable" not in q1c and "displayStyle" not in q1c
    assert q1c == original["q1_c"]
    # (a) is about every valid component, not just this one: only the broken one may differ
    assert {k: v for k, v in got.items() if k != "quiz_title"} == {
        k: v for k, v in original.items() if k != "quiz_title"
    }

    r = UiGate(validator=A2UIValidator()).gate(raw, turn_id="t11", canvas_idx=0, final=True)
    assert r.surface is not None, r.report.problems
    assert _tree(r.surface)["q1_c"] == original["q1_c"]  # and it reaches the renderer intact
    assert "quiz_title" in r.resample


def test_an_invented_non_string_leaf_never_reaches_the_renderer(gate):
    """A11(b): grounding covers every leaf type, not only strings.

    The model wrote ``"value": "yes"`` on a CheckBox — the wrong type. The fixer coerces it to the
    boolean ``true``: a value the model never wrote, which renders as a *ticked* box the user
    never ticked and which validates, so nothing downstream catches it. A non-string leaf must
    equal the parsed original at that path or the key is dropped; ``value`` is required on
    CheckBox, so what is left fails the catalog and is pruned and re-requested instead.
    """
    raw = "```json\n" + _canvas(
        _create(),
        _update(
            [
                {"id": "root", "component": "Column", "children": ["agree"]},
                {"id": "agree", "component": "CheckBox", "label": "Done", "value": "yes"},
            ]
        ),
    )
    messages, _ = gate.repair(raw)
    got = {
        c["id"]: c
        for m in messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
        if isinstance(c.get("id"), str)
    }
    agree = got.get("agree")
    if agree is not None:
        assert agree.get("value") is not True  # the fixer's boolean is not the model's "yes"
        assert "value" not in agree or agree["value"] == "yes"

    r = UiGate(validator=A2UIValidator()).gate(raw, turn_id="t12", canvas_idx=0, final=True)
    assert r.resample == ("agree",)
    assert r.surface is not None and "agree" not in _ids(r.surface)
    assert _tree(r.surface)["root"]["children"] == []  # the dangling reference went with it


def test_a_valid_component_contributes_no_repair_edits(gate):
    """A11(a) again, on the count rather than the content: salvage runs *only* over the components
    that failed, so adding well-formed components to a canvas cannot change ``repaired``.

    Oracle: two canvases that differ by eight valid ``Text`` components and nothing else. The
    text-level repairs (the unclosed fence, the closing brackets) are the same for both, and the
    one broken component is the same, so the edit counts must be equal. Salvaging the whole array
    logs an "inserted default value" per valid component instead, which is also what makes the Q11
    ``ui_repairs`` metric meaningless (docs/tasks/T-0010.md A6).
    """

    def canvas(extra: int) -> str:
        components = [
            {
                "id": "root",
                "component": "Column",
                "children": ["bad", *[f"t{i}" for i in range(extra)]],
            },
            {"id": "bad", "component": "Text", "variant": "huge", "text": "broken"},
            *({"id": f"t{i}", "component": "Text", "text": f"line {i}"} for i in range(extra)),
        ]
        return "```json\n" + _canvas(_create(), _update(components))

    _, without = gate.repair(canvas(0))
    _, with_eight = UiGate(validator=A2UIValidator()).repair(canvas(8))
    assert with_eight == without
