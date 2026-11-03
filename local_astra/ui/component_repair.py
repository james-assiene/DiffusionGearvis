"""Content-preserving component repairs for the UI gate (T-0108b, docs/12 §1.1).

``repair_components(components, validator) -> (components, edits)`` rewrites a schema-invalid
component into a valid one when the model wrote everything the component needs, in the wrong
shape. Pure: no I/O, the input is never mutated.

Rules, each firing only on a component the catalog rejects:

- **BC** — a Button whose ``children``, or a non-string ``child``, holds exactly one distinct id
  gets ``child`` = that id (never the Button's own id). An inline component with that id is lifted
  to top level, unless the id is already defined at top level, where it is referenced (C6).
- **LI** — in Row, Column and List ``children``, each inline object with an ``id`` and a
  ``component`` is replaced by its id and lifted (or referenced, C6).
- **TF** — a TextField with no ``label`` gets ``label`` from ``placeholder`` (or ``text``); the
  old key is removed.
- **DT** — a DateTimeInput with no ``value`` gets ``value: ""``.
- **CPO** — ChoicePicker options that are all strings become ``{"label": s, "value": s}``.
- **CPV** — a missing ChoicePicker ``value`` becomes ``[]``; a string ``s`` becomes ``[s]``, or
  ``[]`` when ``s`` is ``""``.
- **RB** (H-T108b (d)) — a Button with a non-empty string ``text`` (or ``label``), no child and
  no ``action`` gets a lifted Text with the reserved id ``__gate_<id>_label`` carrying the model's
  string, ``child`` = that id, and ``action`` = the reserved event ``__gate.clicked`` with the
  model's string as ``context.label``; the old key is removed.
- **CC** (T-0127) — a Card with no ``child`` whose ``children`` is non-empty: inline objects are
  lifted LI-style (C6 applies) and replaced by their ids; one distinct id becomes ``child``; several
  (H-T127-1) become the children, in first-occurrence order, of a lifted Column with the reserved id
  ``__gate_<id>_body``, and ``child`` is that id. ``children`` is removed. Refused when an item is
  an id-less object, the Card's own id or a reserved id, when the wrap id is already written, or
  when the Card would end up inside its own subtree.

All-or-nothing per component: a repair is kept only if the component **and** every lifted child
validate; otherwise the identical object passes through. A valid component is always the identical
object, and a list in which nothing changed is returned as the input list object.

The repairs preserve content: every string value in the output is the model's own (a key rename
or a key move keeps the model's string), or ``""``; added keys are catalog properties of the kind;
the only added non-strings are ``""`` and ``[]``; no id is created and a lifted component is the
model's object, unchanged. The single exception is the reserved namespace RB and CC use: ids
starting with :data:`RESERVED_PREFIX` (``__gate_<id>_label``, ``__gate_<id>_body``), the event name
:data:`RESERVED_EVENT` and the kind of the gate-made component (``Text``, ``Column``) are the only
strings the gate creates, and a model-written id in that namespace is never lifted or borrowed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from local_astra.ui.a2ui_schema import component_children

#: H-T108b (d): the id prefix and event name the gate may create, and nothing else
RESERVED_PREFIX = "__gate_"
RESERVED_EVENT = "__gate.clicked"

_CONTAINERS = ("Row", "Column", "List")


class _Refused:
    """A rule applies to the component but cannot be applied without changing content."""


_REFUSED = _Refused()


class _Validator(Protocol):
    def component_errors(self, component: Any) -> Sequence[Any]: ...


def reserved_label_id(button_id: str) -> str:
    """The id of the Text RB lifts for the Button ``button_id``."""
    return f"{RESERVED_PREFIX}{button_id}_label"


def reserved_body_id(card_id: str) -> str:
    """The id of the Column CC lifts for the Card ``card_id`` written with several children."""
    return f"{RESERVED_PREFIX}{card_id}_body"


def repair_components(components: list[Any], validator: _Validator) -> tuple[list[Any], int]:
    """``(components, edits)``: ``components`` with every repairable invalid component repaired.

    Lifted children follow their parent; the original order is kept. ``edits`` counts the rule
    applications of the accepted repairs; it is 0, and the input list object comes back, when
    nothing changed.
    """
    top_ids = {_id(c) for c in components} - {""}
    every_id = top_ids | {_id(x) for c in components for x in _inline(c)} - {""}
    top = _top_by_id(components)
    lifted: set[str] = set()
    out: list[Any] = []
    edits = 0
    for component in components:
        if not isinstance(component, Mapping) or not validator.component_errors(component):
            out.append(component)
            continue
        repaired = _repair_one(component, top_ids | lifted, every_id, top)
        if repaired is None:
            out.append(component)
            continue
        new, children, n = repaired
        if validator.component_errors(new) or any(validator.component_errors(c) for c in children):
            out.append(component)
            continue
        out.append(new)
        out.extend(children)
        lifted.update(_id(c) for c in children)
        # criterion 7: later cycle checks see the references this repair created (multi-lift cycles)
        top[_id(new)] = new
        top.update((_id(c), c) for c in children if _id(c))
        edits += n
    if edits == 0:
        return components, 0
    return out, edits


def _top_by_id(components: Sequence[Any]) -> dict[str, Any]:
    return {cid: c for c in components if (cid := _id(c))}


def _repair_one(
    component: Mapping[str, Any],
    defined: set[str],
    every_id: set[str],
    top: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Any], int] | None:
    """The repaired component, its lifted children and the rule count; ``None`` if no rule fits."""
    cid = _id(component)
    if not cid or cid.startswith(RESERVED_PREFIX):
        return None  # the reserved namespace stays the gate's own
    kind = component.get("component")
    new = dict(component)
    children: list[Any] = []
    n = 0
    if kind == "Button":
        bc = _bc(new, cid, defined, top)
        if isinstance(bc, _Refused):
            return None
        if bc is not None:
            children.extend(bc)
            n += 1
        elif _rb(new, cid, every_id, children):
            n += 1
    elif kind == "Card":
        cc = _cc(new, cid, defined, every_id, top)
        if isinstance(cc, _Refused):
            return None
        if cc is not None:
            children.extend(cc)
            n += 1
    elif kind in _CONTAINERS:
        li = _li(new, cid, defined, top)
        if isinstance(li, _Refused):
            return None
        if li is not None:
            children.extend(li)
            n += 1
    elif kind == "TextField":
        n += _tf(new)
    elif kind == "DateTimeInput":
        if "value" not in new:
            new["value"] = ""
            n += 1
    elif kind == "ChoicePicker":
        n += _cpo(new) + _cpv(new)
    if n == 0:
        return None
    return new, children, n


def _bc(
    new: dict[str, Any], cid: str, defined: set[str], top: Mapping[str, Any]
) -> list[Any] | _Refused | None:
    """BC on ``new`` in place: the children to lift, ``None`` if BC does not apply, ``_REFUSED`` if it
    applies but must be refused."""
    child = new.get("child")
    children = new.get("children")
    if isinstance(child, str) or (not isinstance(children, list) and not isinstance(child, dict)):
        return None
    items: list[Any] = []
    if isinstance(children, list):
        items.extend(children)
    if isinstance(child, dict):
        items.append(child)
    target = _single_target(items, cid)
    if target is None:
        return _REFUSED
    tid, inline = target
    lift = _lift(tid, inline, cid, defined, top)
    if isinstance(lift, _Refused):
        return lift
    new.pop("children", None)
    new["child"] = tid
    return lift


def _single_target(items: Sequence[Any], cid: str) -> tuple[str, list[Any]] | None:
    """The one distinct id ``items`` hold and the inline objects carrying it, or ``None``."""
    ids: set[str] = set()
    inline: list[Any] = []
    for item in items:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, Mapping) and _id(item):
            ids.add(_id(item))
            inline.append(item)
        else:
            return None  # an inline object with no id cannot be referenced
    if len(ids) != 1:
        return None
    (tid,) = ids
    if tid == cid or tid.startswith(RESERVED_PREFIX):
        return None
    return tid, inline


def _lift(
    tid: str, inline: Sequence[Any], parent: str, defined: set[str], top: Mapping[str, Any]
) -> list[Any] | _Refused:
    """The children to lift for a reference to ``tid`` (``[]`` when it is already defined, C6),
    or ``_REFUSED`` when the inline objects cannot be lifted unchanged."""
    if tid in defined:
        return []
    full = [x for x in inline if "component" in x]
    if not full:
        # only id-only stubs (or bare id strings): referencing loses nothing
        return [] if all(set(x) == {"id"} for x in inline) else _REFUSED
    if any(x != full[0] and set(x) != {"id"} for x in inline):
        return _REFUSED  # two different definitions of one id: the gate does not choose
    child = full[0]
    if _reaches(child, parent, top):
        return _REFUSED  # criterion 7: a lift that would create a cycle is refused
    return [child]


def _reaches(start: Mapping[str, Any], target: str, top: Mapping[str, Any]) -> bool:
    """Whether ``target`` is reachable from ``start`` through references among ``top``.

    Inline objects are followed too (a later LI/BC may lift them), and ``top`` holds the repairs
    already accepted, so a cycle closed through two or more lifts is found (criterion 7).
    """
    refs: list[str] = []
    nodes: list[Mapping[str, Any]] = [start]
    seen: set[str] = set()
    while refs or nodes:
        if nodes:
            node = nodes.pop()
            refs.extend(component_children(node))
            for inline in _inline(node):
                if _id(inline) == target:
                    return True
                nodes.append(inline)
            continue
        ref = refs.pop()
        if ref == target:
            return True
        if ref in seen:
            continue
        seen.add(ref)
        found = top.get(ref)
        if isinstance(found, Mapping):
            nodes.append(found)
    return False


def _li(
    new: dict[str, Any], cid: str, defined: set[str], top: Mapping[str, Any]
) -> list[Any] | _Refused | None:
    """LI on ``new`` in place: the children to lift, ``None`` if LI does not apply, ``_REFUSED`` if it
    must be refused."""
    children = new.get("children")
    if not isinstance(children, list) or all(isinstance(x, str) for x in children):
        return None
    refs: list[Any] = []
    lifted: list[Any] = []
    seen = set(defined)
    inline_by_id: dict[str, Any] = {}
    for item in children:
        if isinstance(item, str):
            refs.append(item)
            continue
        if not isinstance(item, Mapping) or not _id(item) or "component" not in item:
            return _REFUSED
        tid = _id(item)
        if tid == cid or tid.startswith(RESERVED_PREFIX):
            return _REFUSED
        if inline_by_id.setdefault(tid, item) != item:
            return _REFUSED  # two different definitions of one id: the gate does not choose (C5)
        if tid not in seen:
            if _reaches(item, cid, top):
                return _REFUSED
            lifted.append(item)
            seen.add(tid)
        refs.append(tid)
    new["children"] = refs
    return lifted


def _cc(
    new: dict[str, Any],
    cid: str,
    defined: set[str],
    every_id: set[str],
    top: Mapping[str, Any],
) -> list[Any] | _Refused | None:
    """CC on ``new`` in place: the components to lift (the gate-made Column first, then the inline
    children LI-style), ``None`` if CC does not apply, ``_REFUSED`` if it must be refused."""
    children = new.get("children")
    if "child" in new or not isinstance(children, list) or not children:
        return None
    if _reaches({"component": "Column", "children": children}, cid, top):
        return _REFUSED  # the Card would sit inside its own subtree: a topology cycle
    ids: list[str] = []
    lifted: list[Any] = []
    seen = set(defined)
    inline_by_id: dict[str, Any] = {}
    for item in children:
        if isinstance(item, str):
            tid = item
        elif isinstance(item, Mapping) and _id(item) and "component" in item:
            tid = _id(item)
            if inline_by_id.setdefault(tid, item) != item:
                return _REFUSED  # two different definitions of one id: the gate does not choose
            if tid not in seen:
                lifted.append(item)
                seen.add(tid)
        else:
            return _REFUSED  # an inline object with no id cannot be referenced
        if tid == cid or tid.startswith(RESERVED_PREFIX):
            return _REFUSED
        if tid not in ids:
            ids.append(tid)
    if len(ids) == 1:
        del new["children"]
        new["child"] = ids[0]
        return lifted
    body = reserved_body_id(cid)
    if body in every_id:
        return _REFUSED  # never borrow or duplicate a model-written id in the reserved namespace
    del new["children"]
    new["child"] = body
    return [{"id": body, "component": "Column", "children": ids}, *lifted]


def _rb(new: dict[str, Any], cid: str, every_id: set[str], children: list[Any]) -> bool:
    """RB on ``new`` in place, appending the reserved Text to ``children``; whether it fired."""
    if "child" in new or "children" in new or "action" in new:
        return False
    keys = [k for k in ("text", "label") if k in new]
    if len(keys) != 1:
        return False
    label = new[keys[0]]
    if not isinstance(label, str) or not label:
        return False
    tid = reserved_label_id(cid)
    if tid in every_id:
        return False  # never borrow or duplicate a model-written id in the reserved namespace
    del new[keys[0]]
    new["child"] = tid
    new["action"] = {"event": {"name": RESERVED_EVENT, "context": {"label": label}}}
    children.append({"id": tid, "component": "Text", "text": label})
    return True


def _tf(new: dict[str, Any]) -> int:
    if "label" in new:
        return 0
    for key in ("placeholder", "text"):
        if isinstance(new.get(key), str):
            new["label"] = new.pop(key)
            return 1
    return 0


def _cpo(new: dict[str, Any]) -> int:
    options = new.get("options")
    if not isinstance(options, list) or not options or not all(isinstance(o, str) for o in options):
        return 0
    new["options"] = [{"label": o, "value": o} for o in options]
    return 1


def _cpv(new: dict[str, Any]) -> int:
    if "value" not in new:
        new["value"] = []
        return 1
    value = new["value"]
    if isinstance(value, str):
        new["value"] = [value] if value else []
        return 1
    return 0


def _inline(component: Any) -> list[Any]:
    """The inline objects under ``children``/``child`` of a component."""
    if not isinstance(component, Mapping):
        return []
    out: list[Any] = []
    for prop in ("children", "child"):
        value = component.get(prop)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, Mapping):
                out.append(item)
    return out


def _id(component: Any) -> str:
    cid = component.get("id") if isinstance(component, Mapping) else None
    return cid if isinstance(cid, str) else ""
