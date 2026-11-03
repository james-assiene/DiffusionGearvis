"""Behaviours of the pinned ``json_repair`` that the ui_gate design relies on (docs/12 §1.1).

Verified on json_repair 0.63.4 (services/versions.lock). If an upgrade changes any of these,
this test fails first and the gate is re-designed consciously, not by surprise.
"""

from __future__ import annotations

import json_repair
import pytest

TRUNCATED = (
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
    '{"id":"root","component":"Column","children":["a","b"]},'
    '{"id":"a","component":"Text","text":"Hello'
)

COMPONENT = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "component": {"type": "string", "enum": ["Text", "Column"]},
        "variant": {"type": "string", "enum": ["h1", "h2", "body"]},
        "text": {"type": "string"},
    },
    "required": ["id", "component", "text"],
    "additionalProperties": False,
}


def test_truncated_canvas_closes_strings_arrays_and_objects():
    out = json_repair.loads(TRUNCATED)
    comps = out["updateComponents"]["components"]
    assert [c["id"] for c in comps] == ["root", "a"]
    assert comps[1]["text"] == "Hello"


def test_logging_reports_each_repair_so_the_hud_can_count_them():
    _, log = json_repair.loads(TRUNCATED, logging=True)
    assert len(log) >= 2 and all({"text", "context"} <= set(entry) for entry in log)


def test_stream_stable_keeps_a_partial_string_and_drops_a_partial_key():
    assert json_repair.loads('{"text":"Hel', stream_stable=True) == {"text": "Hel"}
    assert json_repair.loads('{"text":"Hello","vari', stream_stable=True) == {"text": "Hello"}


def test_sloppy_model_output_is_repaired():
    out = json_repair.loads(
        "```json\n{id: 'root', component: \"Column\", children: ['a', 'b',], } // done\n```"
    )
    assert out == {"id": "root", "component": "Column", "children": ["a", "b"]}


def test_concatenated_objects_become_a_list_and_empty_input_is_an_empty_string():
    assert json_repair.loads('{"a":1}\n{"b":2}') == [{"a": 1}, {"b": 2}]
    assert json_repair.loads("") == ""
    assert json_repair.loads("nonsense") == ""


def test_salvage_mode_drops_invalid_array_items_and_strips_extra_properties():
    """This is why the gate applies the schema to the *components array*, not the envelope."""
    raw = (
        '[{"id":"a","component":"Text","text":"x"},'
        '{"id":"b","component":"Nope","text":"y"},'
        '{"id":"c","component":"Text","text":"z","variant":"big"},'
        '{"id":"d","component":"Text","text":"w","weight":2}]'
    )
    out = json_repair.loads(
        raw, schema={"type": "array", "items": COMPONENT}, schema_repair_mode="salvage"
    )
    assert out == [
        {"id": "a", "component": "Text", "text": "x"},
        {"id": "d", "component": "Text", "text": "w"},
    ]


def test_top_level_schema_violations_raise_value_error_in_both_modes():
    """The gate must catch ValueError from the fixer and fall back to unguided repair."""
    for mode in ("standard", "salvage"):
        with pytest.raises(ValueError):
            json_repair.loads(
                '{"id":"t","component":"Text","variant":"huge","text":"x"}',
                schema=COMPONENT,
                schema_repair_mode=mode,
            )
        with pytest.raises(ValueError):
            json_repair.loads(
                '{"id":"t","component":"Text"}', schema=COMPONENT, schema_repair_mode=mode
            )


def test_local_refs_and_all_of_are_followed_but_unevaluated_properties_are_not_salvaged():
    ref_schema = {"$defs": {"comp": COMPONENT}, "type": "array", "items": {"$ref": "#/$defs/comp"}}
    assert json_repair.loads(
        '[{"id":"a","component":"Text","text":"x"}]',
        schema=ref_schema,
        schema_repair_mode="salvage",
    ) == [{"id": "a", "component": "Text", "text": "x"}]
    all_of = {
        "type": "array",
        "items": {
            "allOf": [
                {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                {
                    "type": "object",
                    "properties": {"component": {"const": "Text"}, "text": {"type": "string"}},
                    "required": ["component", "text"],
                },
            ],
            "unevaluatedProperties": False,
        },
    }
    assert json_repair.loads(
        '[{"id":"a","component":"Text","text":"x"}]', schema=all_of, schema_repair_mode="salvage"
    ) == [{"id": "a", "component": "Text", "text": "x"}]
    with pytest.raises(ValueError):  # → the gate strips unevaluatedProperties from the fixer schema
        json_repair.loads(
            '[{"id":"a","component":"Text","text":"x","junk":1}]',
            schema=all_of,
            schema_repair_mode="salvage",
        )


def test_valid_input_passes_through_untouched_with_an_empty_log():
    out, log = json_repair.loads(
        '{"id":"a","component":"Text","text":"x"}',
        schema=COMPONENT,
        schema_repair_mode="salvage",
        logging=True,
    )
    assert out == {"id": "a", "component": "Text", "text": "x"} and log == []
