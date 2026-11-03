"""A2UI v0.9.1 validation and topology (docs/12 §1.1). The full JSON Schema is the oracle."""

from __future__ import annotations

import copy
import json

import pytest

from local_astra.ui import A2UIValidator, parse_concatenated_json
from local_astra.ui.a2ui_schema import (
    ACTIVE_CATALOG_ALIAS,
    BASIC_CATALOG_ID,
    COMMON_TYPES_ID,
    SERVER_TO_CLIENT_ID,
    component_children,
)


@pytest.fixture(scope="module")
def validator() -> A2UIValidator:
    return A2UIValidator()


@pytest.fixture(scope="module")
def quiz(quiz_example_text) -> list:
    return parse_concatenated_json(quiz_example_text)


def _full_ok(validator: A2UIValidator, msg) -> bool:
    """Oracle: the unmodified spec schema, slow path."""
    return validator.message_validator.is_valid(msg)


# --- vendored schemas ----------------------------------------------------------------------------


def test_vendored_schema_ids_are_the_spec_ids(validator):
    assert validator.server_to_client["$id"] == SERVER_TO_CLIENT_ID
    assert validator.common_types["$id"] == COMMON_TYPES_ID
    assert validator.catalog["$id"] == BASIC_CATALOG_ID
    # the message schema refers to the catalog by a relative path → the alias must be registered
    assert validator.registry.get_or_retrieve(ACTIVE_CATALOG_ALIAS) is not None


def test_basic_catalog_has_the_18_components(validator):
    assert validator.component_kinds == {
        "Text",
        "Image",
        "Icon",
        "Video",
        "AudioPlayer",
        "Row",
        "Column",
        "List",
        "Card",
        "Tabs",
        "Modal",
        "Divider",
        "Button",
        "TextField",
        "CheckBox",
        "ChoicePicker",
        "Slider",
        "DateTimeInput",
    }


# --- parsing ---------------------------------------------------------------------------------


def test_parse_concatenated_json_accepts_jsonl_and_pretty_printed_streams():
    assert parse_concatenated_json('{"a":1}\n{"b":2}\n') == [{"a": 1}, {"b": 2}]
    assert parse_concatenated_json('{\n  "a": [\n 1,\n 2\n ]\n}\n\n  {"b": {}}') == [
        {"a": [1, 2]},
        {"b": {}},
    ]
    assert parse_concatenated_json("   ") == []


def test_parse_concatenated_json_raises_on_truncation():
    with pytest.raises(json.JSONDecodeError):
        parse_concatenated_json('{"a": 1} {"b": ')


# --- the quiz fixture -----------------------------------------------------------------------------


def test_quiz_fixture_is_valid_and_has_38_reachable_components(validator, quiz):
    r = validator.validate_messages(quiz)
    assert r.ok, (r.schema_errors, r.problems)
    assert r.kinds == ("createSurface", "updateComponents", "updateDataModel")
    assert r.surface_ids == ("lesson_1",)
    assert r.topology is not None and r.topology.ok
    assert len(r.topology.reachable) == 38 and r.topology.reachable[0] == "root"
    assert r.topology.warnings == ()
    assert all(_full_ok(validator, m) for m in quiz)


def test_validate_text_matches_validate_messages(validator, quiz_example_text, quiz):
    assert validator.validate_text(quiz_example_text) == validator.validate_messages(quiz)


# --- fast path agrees with the full schema -------------------------------------------------------


def _mutations(quiz):
    upd = quiz[1]
    yield "valid", upd
    m = copy.deepcopy(upd)
    m["updateComponents"]["components"][2]["variant"] = "huge"
    yield "bad enum", m
    m = copy.deepcopy(upd)
    del m["updateComponents"]["components"][2]["text"]
    yield "missing required", m
    m = copy.deepcopy(upd)
    m["updateComponents"]["components"][2]["component"] = "Sparkle"
    yield "unknown kind", m
    m = copy.deepcopy(upd)
    m["updateComponents"]["components"][2]["colour"] = "red"
    yield "extra property", m
    m = copy.deepcopy(upd)
    m["updateComponents"]["components"][1]["children"] = "quiz_title"
    yield "children not a list", m
    m = copy.deepcopy(upd)
    m["version"] = "v0.8"
    yield "bad version", m
    m = copy.deepcopy(upd)
    m["updateComponents"]["extra"] = 1
    yield "extra envelope key", m
    m = copy.deepcopy(upd)
    m["updateComponents"]["components"] = []
    yield "empty components", m
    yield "create", quiz[0]
    yield "data model", quiz[2]
    yield "delete", {"version": "v0.9.1", "deleteSurface": {"surfaceId": "lesson_1"}}
    yield (
        "two kinds",
        {
            "version": "v0.9.1",
            "deleteSurface": {"surfaceId": "s"},
            "createSurface": {"surfaceId": "s", "catalogId": "c"},
        },
    )


def test_fast_path_verdict_equals_full_schema_verdict(validator, quiz):
    for label, msg in _mutations(quiz):
        fast = not validator.message_errors(msg)
        assert fast == _full_ok(validator, msg), label


def test_component_errors_point_at_the_field(validator, quiz):
    bad = copy.deepcopy(quiz[1]["updateComponents"]["components"][2])
    bad["variant"] = "huge"
    errs = validator.component_errors(bad)
    assert [e.path for e in errs] == ["variant"]
    assert "'huge' is not one of" in errs[0].message


def test_message_errors_carry_the_component_index(validator, quiz):
    m = copy.deepcopy(quiz[1])
    m["updateComponents"]["components"][5]["component"] = "Sparkle"
    errs = validator.message_errors(m)
    assert len(errs) == 1 and errs[0].path == "updateComponents/components/5/component"


# --- topology ---------------------------------------------------------------------------------


def _comps(*items):
    return [dict(c) for c in items]


def test_topology_reports_missing_reference(validator):
    r = validator.check_topology(
        _comps(
            {"id": "root", "component": "Column", "children": ["a", "ghost"]},
            {"id": "a", "component": "Text", "text": "x"},
        )
    )
    assert not r.ok and r.errors == ("'root' references missing component 'ghost'",)


def test_topology_requires_root_and_unique_ids(validator):
    r = validator.check_topology(
        _comps(
            {"id": "a", "component": "Text", "text": "x"},
            {"id": "a", "component": "Text", "text": "y"},
        )
    )
    assert set(r.errors) == {"duplicate id 'a'", "no component with id 'root'"}


def test_topology_detects_cycles_and_self_reference(validator):
    r = validator.check_topology(
        _comps(
            {"id": "root", "component": "Column", "children": ["a"]},
            {"id": "a", "component": "Card", "child": "root"},
        )
    )
    assert "cycle: root -> a -> root" in r.errors
    r2 = validator.check_topology(
        _comps({"id": "root", "component": "Column", "children": ["root"]})
    )
    assert "'root' references itself" in r2.errors


def test_topology_orphans_are_warnings_not_errors(validator):
    r = validator.check_topology(
        _comps(
            {"id": "root", "component": "Text", "text": "x"},
            {"id": "lonely", "component": "Text", "text": "y"},
        )
    )
    assert r.ok and r.warnings == ("unreachable from root: lonely",) and r.reachable == ("root",)


def test_topology_follows_template_children_tabs_buttons_and_modals(validator):
    comps = _comps(
        {
            "id": "root",
            "component": "Tabs",
            "tabs": [{"title": "A", "child": "lst"}, {"title": "B", "child": "btn"}],
        },
        {"id": "lst", "component": "List", "children": {"componentId": "row_t", "path": "/items"}},
        {"id": "row_t", "component": "Text", "text": {"path": "/name"}},
        {"id": "btn", "component": "Button", "child": "lbl", "action": {"event": {"name": "go"}}},
        {"id": "lbl", "component": "Text", "text": "Go"},
        {"id": "m", "component": "Modal", "trigger": "btn", "content": "lbl"},
    )
    assert component_children(comps[0]) == ["lst", "btn"]
    assert component_children(comps[1]) == ["row_t"]
    assert component_children(comps[5]) == ["btn", "lbl"]
    r = validator.check_topology(comps)
    assert r.errors == () and r.reachable == ("root", "lst", "row_t", "btn", "lbl")
    assert r.warnings == ("unreachable from root: m",)


def test_invalid_components_are_named_for_resample(validator, quiz):
    comps = copy.deepcopy(quiz[1]["updateComponents"]["components"])
    comps[3]["children"] = 5  # q1's children
    comps[7]["variant"] = "huge"  # q2_p? whichever: we only care that ids are reported
    r = validator.check_topology(comps)
    assert set(r.invalid_components) == {comps[3]["id"], comps[7]["id"]}
    assert not r.ok


# --- whole surface rules --------------------------------------------------------------------------


def test_surface_requires_create_before_update_and_known_catalog(validator, quiz):
    r = validator.validate_messages([quiz[1]])
    assert not r.ok and r.problems == (
        "[0] updateComponents for surface 'lesson_1' before createSurface",
    )
    create = copy.deepcopy(quiz[0])
    create["createSurface"]["catalogId"] = "example.com:nope"
    r2 = validator.validate_messages([create, quiz[1]])
    assert "[0] unknown catalogId 'example.com:nope'" in r2.problems


def test_surface_report_lists_schema_errors_with_message_index(validator, quiz):
    m = copy.deepcopy(quiz[1])
    m["updateComponents"]["components"][2]["variant"] = "huge"
    r = validator.validate_messages([quiz[0], m])
    assert not r.ok
    assert r.schema_errors[0].path == "[1]/updateComponents/components/2/variant"
    assert r.topology is not None and r.topology.invalid_components == ("quiz_title",)


def test_delete_then_recreate_is_allowed(validator, quiz):
    delete = {"version": "v0.9.1", "deleteSurface": {"surfaceId": "lesson_1"}}
    r = validator.validate_messages([quiz[0], delete, quiz[0]])
    assert r.ok and r.kinds == ("createSurface", "deleteSurface", "createSurface")
    r2 = validator.validate_messages([quiz[0], quiz[0]])
    assert r2.problems == ("[1] createSurface for existing surface 'lesson_1'",)


# --- custom catalog hook (what the edu catalog will use, T-0010) -------------------------------------


def test_custom_catalog_extends_any_component_and_topology(validator):
    edu = copy.deepcopy(validator.catalog)
    edu["$id"] = "https://example.invalid/edu/catalog.json"
    edu["catalogId"] = "local-astra:edu"
    edu["components"]["Math"] = {
        "type": "object",
        "allOf": [
            {"$ref": COMMON_TYPES_ID + "#/$defs/ComponentCommon"},
            {"$ref": "#/$defs/CatalogComponentCommon"},
            {
                "type": "object",
                "properties": {
                    "component": {"const": "Math"},
                    "latex": {"type": "string"},
                    "child": {"$ref": COMMON_TYPES_ID + "#/$defs/ComponentId"},
                },
                "required": ["component", "latex"],
            },
        ],
        "unevaluatedProperties": False,
    }
    edu["$defs"]["anyComponent"]["oneOf"].append({"$ref": "#/components/Math"})
    v = A2UIValidator(edu, extra_single_child_props={"Math": ("child",)})
    surface = [
        {"version": "v0.9.1", "createSurface": {"surfaceId": "s", "catalogId": "local-astra:edu"}},
        {
            "version": "v0.9.1",
            "updateComponents": {
                "surfaceId": "s",
                "components": [
                    {"id": "root", "component": "Math", "latex": "x^2", "child": "cap"},
                    {"id": "cap", "component": "Text", "text": "caption"},
                ],
            },
        },
    ]
    r = v.validate_messages(surface)
    assert r.ok, (r.schema_errors, r.problems)
    assert r.topology is not None and r.topology.reachable == ("root", "cap")
    # the stock validator does not know Math
    assert not validator.validate_messages(surface).ok
    # and Math still rejects garbage
    bad = copy.deepcopy(surface)
    bad[1]["updateComponents"]["components"][0]["latex"] = 5
    assert not v.validate_messages(bad).ok


# --- progressive rendering: components accumulate across messages and canvases ------------------------


def _create(sid="s"):
    return {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": BASIC_CATALOG_ID}}


def _update(comps, sid="s"):
    return {"version": "v0.9.1", "updateComponents": {"surfaceId": sid, "components": comps}}


def test_components_accumulate_across_update_messages(validator):
    msgs = [
        _create(),
        _update(
            [
                {"id": "root", "component": "Column", "children": ["a", "b"]},
                {"id": "a", "component": "Text", "text": "A"},
            ]
        ),
        _update([{"id": "b", "component": "Text", "text": "B"}]),
    ]
    r = validator.validate_messages(msgs)
    assert r.ok, (r.problems, r.schema_errors)
    assert r.topology is not None and r.topology.reachable == ("root", "a", "b")


def test_forward_reference_is_an_error_when_final_and_a_warning_when_partial(validator):
    msgs = [
        _create(),
        _update(
            [
                {"id": "root", "component": "Column", "children": ["a", "later"]},
                {"id": "a", "component": "Text", "text": "A"},
            ]
        ),
    ]
    final = validator.validate_messages(msgs, final=True)
    assert (
        not final.ok
        and "surface 's': 'root' references missing component 'later'" in final.problems
    )
    partial = validator.validate_messages(msgs, final=False)
    assert partial.ok and partial.topology is not None
    assert partial.topology.warnings == ("'root' references missing component 'later'",)
    assert partial.topology.reachable == ("root", "a")


def test_prior_components_from_earlier_canvases_complete_the_tree(validator):
    prior = {
        "s": [
            {"id": "root", "component": "Column", "children": ["a", "b"]},
            {"id": "a", "component": "Text", "text": "A"},
        ]
    }
    r = validator.validate_messages(
        [_update([{"id": "b", "component": "Text", "text": "B"}])], prior=prior
    )
    assert r.ok and r.topology is not None and r.topology.reachable == ("root", "a", "b")
    # a later definition replaces the earlier one
    r2 = validator.validate_messages(
        [
            _update(
                [
                    {"id": "a", "component": "Text", "text": "A2"},
                    {"id": "b", "component": "Text", "text": "B"},
                ]
            )
        ],
        prior=prior,
    )
    assert r2.ok and r2.topology is not None and len(r2.topology.reachable) == 3


def test_missing_root_is_soft_only_in_partial_mode(validator):
    msgs = [_create(), _update([{"id": "a", "component": "Text", "text": "A"}])]
    assert not validator.validate_messages(msgs).ok
    r = validator.validate_messages(msgs, final=False)
    assert r.ok and r.topology is not None and "no component with id 'root'" in r.topology.warnings


def test_non_object_component_is_reported_not_crashed(validator):
    r = validator.check_topology([{"id": "root", "component": "Text", "text": "x"}, "garbage", 5])
    assert not r.ok and r.errors == (
        "component #1 is not an object",
        "component #2 is not an object",
    )
    r2 = validator.validate_messages(
        [_create(), _update([{"id": "root", "component": "Text", "text": "x"}, "garbage"])]
    )
    assert not r2.ok and r2.schema_errors[0].path == "[1]/updateComponents/components/1"


# --- formats: DateTimeInput.min/max is a oneOf over date | time | date-time -----------------------------


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("2026-09-12", True),
        ("10:00:00Z", True),
        ("2026-09-12T10:00:00Z", True),
        ("2026-13-45", False),
        ("not a date", False),
        ("10:00:00", False),
    ],
)
def test_datetime_input_bounds_are_format_checked(validator, value, valid):
    c = {
        "id": "d",
        "component": "DateTimeInput",
        "value": {"path": "/when"},
        "enableDate": True,
        "min": value,
    }
    assert (not validator.component_errors(c)) is valid
    assert validator.component_validator.is_valid(c) is valid  # slow path agrees
