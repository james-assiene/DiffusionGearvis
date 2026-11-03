"""Derive the schema ``json_repair`` can be guided by (docs/12 §1.1, T-0010 C1).

``json_repair``'s schema mode is not a JSON Schema implementation: it follows *local* ``$ref``s
only and it does not evaluate ``unevaluatedProperties`` — with the vendored A2UI catalog as-is it
would strip every property the ``allOf`` branches contribute (pinned in
``tests/unit/test_json_repair_contract.py::test_local_refs_and_all_of_are_followed_but_unevaluated_properties_are_not_salvaged``).

So the fixer gets a *derived* document: ``common_types.json``'s ``$defs`` inlined next to the
catalog's, every absolute ``$ref`` into ``common_types.json`` rewritten to the local pointer, and
every ``unevaluatedProperties`` dropped. The authoritative verdict is still ``A2UIValidator``'s,
against the untouched vendored schemas; this derivation only tells the fixer what shape to aim at.

Localising every ``$ref`` is also the egress guard: ``referencing`` resolves a non-local absolute
``$ref`` by DNS + HTTP GET, so a catalog that references another document would make every
repaired canvas hit the network (loopback-only rule, CLAUDE.md). The self-contained assertion in
``tests/unit/test_ui_gate.py`` pins that no such ``$ref`` survives derivation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: documents whose ``$ref``s become local pointers, because both end up in the derived document:
#: ``common_types.json``'s ``$defs`` are merged into ``$defs``, the catalog keeps its own layout.
_INLINED = ("common_types.json", "catalog.json")


def _local_ref(ref: str) -> str | None:
    """``…/common_types.json#/$defs/X`` → ``#/$defs/X``; ``None`` when the ref is not inlined."""
    base, sep, fragment = ref.partition("#")
    if not sep or not fragment:
        return None
    if base and not base.endswith(_INLINED):
        return None
    return "#" + fragment


def _localise(node: Any) -> Any:
    """Drop ``unevaluatedProperties`` and rewrite inlined ``$ref``s to local pointers."""
    if isinstance(node, Mapping):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "unevaluatedProperties":
                continue
            if key == "$ref" and isinstance(value, str):
                out[key] = _local_ref(value) or value
            else:
                out[key] = _localise(value)
        return out
    if isinstance(node, list):
        return [_localise(v) for v in node]
    return node


def components_array_schema(
    catalog: Mapping[str, Any], common_types: Mapping[str, Any]
) -> dict[str, Any]:
    """Schema for an ``updateComponents.components`` array, self-contained and local-ref only.

    Raises ``KeyError`` if the catalog does not expose ``$defs/anyComponent`` — the caller falls
    back to unguided repair rather than guessing.
    """
    defs: dict[str, Any] = {
        **_localise(common_types.get("$defs", {})),
        **_localise(catalog["$defs"]),
    }
    if "anyComponent" not in defs:
        raise KeyError("anyComponent")
    return {
        "type": "array",
        "items": {"$ref": "#/$defs/anyComponent"},
        "$defs": defs,
        # ``anyComponent`` points at ``#/components/<kind>`` and ``anyFunction`` at
        # ``#/functions/<name>``; both sections must travel with the derived document.
        "components": _localise(catalog.get("components", {})),
        "functions": _localise(catalog.get("functions", {})),
    }
