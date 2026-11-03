"""T-0108b inside ``UiGate``: the component stage on both paths, its count, grounding (C7), the unrepairable case,
RB through the gate, and the stale-stub fix (criteria 1, 4, 5 C7, 8, 10).

Rows that need the new code are active (were pending until 33db104). The rows named "guard" are guards that hold at the branch
base (08c7776, whose ``local_astra/`` equals 749312d) and must still hold after the task.

Readings this module makes where the task is open (named in the test-author report):

- "Strict path": on a strict-parseable canvas the component stage is the only stage that edits, so
  ``repaired == component_repairs`` there.
- Criterion 8's "updated only when ``cid`` is emitted on the current canvas" is checked on ``UiGate._as_written``
  directly (the criterion names it), and "a restored component that fails ``component_errors`` is not restored" by
  planting a stale stub in it: each half of the fix gets a row the other half cannot satisfy.

Regime: CPU, pure, the vendored A2UI v0.9.1 schema; deterministic, n = 1 per row.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.conductor import ui_gate as ui_gate_module
from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.conductor.ui_instructions import UI_EXAMPLE
from local_astra.ui import parse_concatenated_json

MODULE = "local_astra.ui.component_repair"
REPO = Path(__file__).resolve().parents[2]
NAME = "repair_components"
CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
CREATE = '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"' + CATALOG + '"}}'
TURN = "t"
SID = f"turn-{TURN}"
INJECTED = "zz-injected-not-in-raw-5b1e"


def update(*components: str) -> str:
    return (
        '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
        + ",".join(components)
        + "]}}"
    )


ROOT_HF = '{"id":"root","component":"Column","children":["h","f"]}'
H = '{"id":"h","component":"Text","text":"Your trip"}'
F_PLACEHOLDER = '{"id":"f","component":"TextField","placeholder":"From"}'
F_HUGE = '{"id":"f","component":"TextField","variant":"huge","label":"From"}'
#: strict-parseable, the TextField schema-invalid (the Goal's first defect)
PLACEHOLDER = CREATE + "\n" + update(ROOT_HF, H, F_PLACEHOLDER)
#: the same canvas missing its final ``}``: the unguided path
UNGUIDED = PLACEHOLDER[:-1]
#: an unrepairable TextField (criterion 9(ii)'s fixture)
HUGE = CREATE + "\n" + update(ROOT_HF, H, F_HUGE)


def gate_once(raw: str) -> GateResult:
    return UiGate().gate(raw, turn_id=TURN, canvas_idx=0, final=False)


def emitted(*results: GateResult) -> dict[str, Any]:
    """The surface as the renderer holds it after these results: by id, the last definition wins."""
    out: dict[str, Any] = {}
    for result in results:
        for m in result.surface.messages if result.surface is not None else ():
            body = m.get("updateComponents") if isinstance(m, dict) else None
            for c in (body or {}).get("components", ()):
                out[c["id"]] = c
    return out


def component_repairs(result: GateResult) -> int:
    value = getattr(result, "component_repairs", None)
    if value is None:
        raise NotImplementedError("T-0108b criterion 1: GateResult has no `component_repairs` yet")
    assert isinstance(value, int) and not isinstance(value, bool), f"component_repairs is {value!r}"
    return value


def module() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as missing:
        if missing.name == MODULE:
            raise NotImplementedError(f"T-0108b: {MODULE} does not exist yet") from missing
        raise


# --- criterion 1: the stage on both paths, and its count -------------------------------------------------------


def test_premises_of_the_canvases() -> None:
    """Guard: PLACEHOLDER and HUGE parse strictly, UNGUIDED does not; today's gate prunes ``f`` on all three."""
    parse_concatenated_json(PLACEHOLDER)
    parse_concatenated_json(HUGE)
    with pytest.raises(json.JSONDecodeError):
        parse_concatenated_json(UNGUIDED)


def test_strict_path_repairs_the_placeholder_textfield() -> None:
    """Criterion 1 (strict path), 4's spec delta to T-0010 A5 and mutation 12: a surface where ``f`` has its label,
    the placeholder gone, ``root`` still referencing it; ``component_repairs >= 1`` and, the component stage being the
    only stage that edits a strict-parseable canvas, ``repaired == component_repairs``."""
    result = gate_once(PLACEHOLDER)
    assert result.surface is not None, result.report
    got = emitted(result)
    assert got.get("f") == {"id": "f", "component": "TextField", "label": "From"}, got.get("f")
    assert got["root"]["children"] == ["h", "f"]
    n = component_repairs(result)
    assert n >= 1, "criterion 1: the TextField repair is counted"
    assert result.repaired == n, f"criterion 1: repaired includes it ({result.repaired} vs {n})"
    assert result.surface.repaired == result.repaired


def test_unguided_path_repairs_before_salvage() -> None:
    """Criterion 1 (unguided path; mutation 8): the same canvas without its final ``}`` keeps ``f``, repaired, and
    ``repaired`` counts both the structural/fixer edits and the component repair."""
    result = gate_once(UNGUIDED)
    assert result.surface is not None, result.report
    got = emitted(result)
    assert got.get("f") == {"id": "f", "component": "TextField", "label": "From"}, got.get("f")
    n = component_repairs(result)
    assert n >= 1
    assert result.repaired > n, f"repaired ({result.repaired}) also counts the closing edit"


#: the commit the acceptance rows were activated at; its ``ui_gate.py`` is the branch base's gate (749312d's).
#: A literal, never a moving ref (.claude/rules/tests.md).
BASE_GATE_SHA = "0a06f3af5ea8580c7cd8098da7716bf417cca46a"
GATE_PATH = "local_astra/conductor/ui_gate.py"


def base_gate_module() -> ModuleType:
    """``ui_gate.py`` exactly as at ``BASE_GATE_SHA``, compiled in memory from the git blob (it imports the rest of
    ``local_astra`` from the working tree, which criterion 14 keeps byte-identical). Skips when the commit is absent."""
    try:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASE_GATE_SHA}:{GATE_PATH}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"{BASE_GATE_SHA} is not in this clone; the base gate is unavailable")
    name = f"_ui_gate_at_{BASE_GATE_SHA}"
    module = ModuleType(name)
    module.__file__ = str(REPO / GATE_PATH)
    sys.modules[name] = module
    try:
        exec(compile(src, f"{BASE_GATE_SHA}:{GATE_PATH}", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.mark.parametrize("raw", [PLACEHOLDER, UNGUIDED], ids=["strict", "unguided"])
def test_guard_the_base_gate_prunes_the_placeholder_textfield(raw: str) -> None:
    """Guard on the premise of the two rows above, pinned to the base gate (``BASE_GATE_SHA``): there ``f`` does not
    reach the renderer with a label on either path, so those rows test the new stage and not a behaviour the base
    already had."""
    base = base_gate_module()
    assert "repair_components" not in base.__dict__, (
        "premise: the blob is the gate before the component stage"
    )
    result = base.UiGate().gate(raw, turn_id=TURN, canvas_idx=0, final=False)
    assert "f" not in emitted(result) or "label" not in emitted(result)["f"]


# --- criterion 5 C7: grounding against the original canvas on every path ---------------------------------------


def patch_repair(monkeypatch: pytest.MonkeyPatch, fake: Callable[..., Any]) -> None:
    monkeypatch.setattr(module(), NAME, fake)
    if hasattr(ui_gate_module, NAME):
        monkeypatch.setattr(ui_gate_module, NAME, fake)


#: F3 (reviewer A): strings next to the reserved namespace that the exemption must not cover
NEAR_RESERVED = ["__gate.clickedX", "__gatefoo", "__gate_zz_label"]


@pytest.mark.parametrize("injected", [INJECTED, *NEAR_RESERVED])
@pytest.mark.parametrize("raw", [PLACEHOLDER, UNGUIDED], ids=["strict", "unguided"])
def test_a_string_the_repair_invents_is_not_emitted(
    monkeypatch: pytest.MonkeyPatch, raw: str, injected: str
) -> None:
    """C7 / mutation 9: a ``repair_components`` that writes a string absent from ``raw`` (here as the TextField's
    label) is caught by grounding against the original canvas, on both paths; the rest still renders. F3: that
    holds for strings that only resemble the reserved namespace (``__gate.clickedX``, ``__gatefoo``, and
    ``__gate_<id>_label`` for an id the model never wrote)."""
    real = getattr(module(), NAME)
    calls: list[int] = []

    def inventing(components: list[Any], validator: Any) -> Any:
        calls.append(1)
        out, edits = real(components, validator)
        out = [
            {**c, "label": injected} if isinstance(c, dict) and c.get("id") == "f" else c
            for c in out
        ]
        return out, edits

    patch_repair(monkeypatch, inventing)
    result = gate_once(raw)
    assert calls, f"setup: the gate must reach {MODULE}.{NAME} (or a same-named ui_gate global)"
    assert result.surface is not None, result.report
    text = json.dumps(list(result.surface.messages))
    assert injected not in text, f"C7: an invented string reached the renderer: {text}"
    assert emitted(result)["h"]["text"] == "Your trip"


# --- criterion 4 / unrepairable defects -----------------------------------------------------------------------


def test_guard_an_unrepairable_textfield_is_still_pruned() -> None:
    """Guard (holds today, must hold after): ``variant: "huge"`` is pruned, the rest renders, ``repaired == 0``."""
    result = gate_once(HUGE)
    assert result.surface is not None and "f" not in emitted(result)
    assert emitted(result)["root"]["children"] == ["h"]
    assert result.repaired == 0


@pytest.mark.parametrize("name", ["HUGE", "UI_EXAMPLE"])
def test_no_component_repair_is_counted_without_one(name: str) -> None:
    """Criterion 4: an unrepairable canvas and an all-valid one report ``component_repairs == 0``."""
    result = gate_once({"HUGE": HUGE, "UI_EXAMPLE": UI_EXAMPLE}[name])
    assert component_repairs(result) == 0 and result.repaired == 0


# --- RB through the gate (H-T108b (d)) --------------------------------------------------------------------------

BTN = '{"id":"btn","component":"Button","text":"Search Flights"}'
RB_CANVAS = (
    CREATE + "\n" + update('{"id":"root","component":"Column","children":["h","btn"]}', H, BTN)
)


def test_rb_button_reaches_the_renderer_in_the_ruled_shape() -> None:
    """H-T108b (d) and C7's exemption: the text-only Button renders with the reserved Text id and event; its label
    is the model's string verbatim; nothing else is invented."""
    result = gate_once(RB_CANVAS)
    assert result.surface is not None and result.report.ok, result.report
    got = emitted(result)
    assert got.get("btn") == {
        "id": "btn",
        "component": "Button",
        "child": "__gate_btn_label",
        "action": {"event": {"name": "__gate.clicked", "context": {"label": "Search Flights"}}},
    }, got.get("btn")
    assert got.get("__gate_btn_label") == {
        "id": "__gate_btn_label",
        "component": "Text",
        "text": "Search Flights",
    }
    assert got["root"]["children"] == ["h", "btn"]
    assert component_repairs(result) >= 1


# --- criterion 8: the stale stub ---------------------------------------------------------------------------------

D = '{"id":"d","component":"Text","text":"Tuesday"}'
FULL = CREATE + "\n" + update('{"id":"root","component":"Column","children":["h","d"]}', H, D)
#: the same fence cut after ``{"id":"d"``: a stub
STUB = FULL[: FULL.index('{"id":"d"') + len('{"id":"d"')]


def three_canvases() -> tuple[UiGate, list[GateResult]]:
    gate = UiGate()
    results = [
        gate.gate(raw, turn_id=TURN, canvas_idx=i, final=False)
        for i, raw in enumerate([FULL, STUB, FULL])
    ]
    return gate, results


def test_guard_premises_of_the_three_canvas_example() -> None:
    """Guard: the first full canvas renders ``d``; the stub canvas is pruned to something renderable."""
    _, results = three_canvases()
    assert results[0].surface is not None and emitted(results[0])["d"]["text"] == "Tuesday"


def test_three_canvas_example_returns_a_surface_on_call_3() -> None:
    """Criterion 8 (reproduced at 749312d: call 3 had ``surface None``): the full canvas after a stub renders, and
    ``d`` is the full Text."""
    _, results = three_canvases()
    assert results[2].surface is not None, results[2].report
    assert results[2].report.ok
    assert emitted(*results)["d"] == {"id": "d", "component": "Text", "text": "Tuesday"}


def test_a_stub_not_emitted_does_not_overwrite_the_as_written_copy() -> None:
    """Criterion 8, first half (mutation 10): after the stub canvas, ``_as_written`` still holds the full ``d``."""
    gate = UiGate()
    gate.gate(FULL, turn_id=TURN, canvas_idx=0, final=False)
    gate.gate(STUB, turn_id=TURN, canvas_idx=1, final=False)
    assert gate._as_written[SID]["d"] == {"id": "d", "component": "Text", "text": "Tuesday"}, (
        gate._as_written[SID]
    )


def test_an_invalid_as_written_copy_is_not_restored() -> None:
    """Criterion 8, second half (mutation 11): a stale stub planted in ``_as_written`` -- the state main reaches --
    is not re-emitted; the next full canvas renders."""
    gate = UiGate()
    gate.gate(FULL, turn_id=TURN, canvas_idx=0, final=False)
    gate._as_written[SID]["d"] = {"id": "d"}
    result = gate.gate(FULL, turn_id=TURN, canvas_idx=1, final=False)
    assert result.surface is not None, result.report
    assert all(c.get("component") for c in emitted(result).values()), emitted(result)


def test_guard_a_valid_resend_updates_the_as_written_copy() -> None:
    """Criterion 8, wire-order companion (guard; too-wide fix): canvas 1 references ``d`` before it exists; canvas 2
    re-sends ``root`` without ``d``, validly; canvas 3 brings ``d``. The model's latest ``root`` wins: ``root`` is
    not re-emitted with ``d`` restored, and ``_as_written`` holds canvas 2's ``root``."""
    root_hd = '{"id":"root","component":"Column","children":["h","d"]}'
    root_h = '{"id":"root","component":"Column","children":["h"]}'
    gate = UiGate()
    r1 = gate.gate(CREATE + "\n" + update(root_hd, H), turn_id=TURN, canvas_idx=0, final=False)
    r2 = gate.gate(CREATE + "\n" + update(root_h, H), turn_id=TURN, canvas_idx=1, final=False)
    assert gate._as_written[SID]["root"] == {"id": "root", "component": "Column", "children": ["h"]}
    r3 = gate.gate(CREATE + "\n" + update(root_h, H, D), turn_id=TURN, canvas_idx=2, final=False)
    assert all(r.surface is not None for r in (r1, r2, r3))
    assert emitted(r1, r2, r3)["root"]["children"] == ["h"]


# --- adversarial: a repaired Button whose Text arrives on the next canvas (T-0010 A9) ---------------------------


def test_a_repaired_button_defers_and_restores_when_its_text_arrives() -> None:
    """T-0010 A9 with BC: canvas k has a Button whose single child sits under ``children`` and has not arrived; the
    repaired Button is deferred (a Button needs its child), and canvas k+1, which brings only the Text, restores it in
    the repaired shape, with ``root`` re-sent referencing it."""
    root = '{"id":"root","component":"Column","children":["b"]}'
    button = '{"id":"b","component":"Button","children":["t"],"action":{"event":{"name":"go"}}}'
    text = '{"id":"t","component":"Text","text":"Go"}'
    gate = UiGate()
    rk = gate.gate(CREATE + "\n" + update(root, button), turn_id=TURN, canvas_idx=0, final=False)
    rk1 = gate.gate(CREATE + "\n" + update(text), turn_id=TURN, canvas_idx=1, final=False)
    assert rk.surface is not None and rk1.surface is not None, (rk.report, rk1.report)
    got = emitted(rk, rk1)
    assert got.get("b") == {
        "id": "b",
        "component": "Button",
        "child": "t",
        "action": {"event": {"name": "go"}},
    }
    assert got["root"]["children"] == ["b"] and got["t"]["text"] == "Go"


# --- reviewer A, F1: RB on a canvas the model wrote no Text in; grounding all-or-nothing; resample ids --------

YES = '{"id":"y","component":"Button","text":"Yes"}'
NO = '{"id":"n","component":"Button","text":"No"}'
#: a Row of two text-only Buttons: the word ``Text`` (capitalised) occurs nowhere in the canvas
YES_NO = CREATE + "\n" + update('{"id":"root","component":"Row","children":["y","n"]}', YES, NO)


def rb_button(cid: str, label: str) -> dict[str, Any]:
    return {
        "id": cid,
        "component": "Button",
        "child": f"__gate_{cid}_label",
        "action": {"event": {"name": "__gate.clicked", "context": {"label": label}}},
    }


@pytest.mark.parametrize("raw", [YES_NO, YES_NO[:-1]], ids=["strict", "unguided"])
def test_f1_rb_renders_on_a_canvas_with_no_text_component(raw: str) -> None:
    """F1 and the 2026-10-17 clarification: the RB-made Text's ``"component": "Text"`` is part of the ruled shape and
    is exempt from grounding on that gate-made component only, so a canvas with no Text anywhere still renders both
    Buttons and both reserved Texts, on both paths; no reserved id is ever asked to be resampled."""
    assert "Text" not in raw, "premise: the canvas never writes the word Text"
    result = gate_once(raw)
    assert not any(cid.startswith("__gate_") for cid in result.resample), (
        f"F1: resample {result.resample}"
    )
    assert result.surface is not None, result.report
    got = emitted(result)
    assert got.get("y") == rb_button("y", "Yes") and got.get("n") == rb_button("n", "No"), got
    assert got.get("__gate_y_label") == {"id": "__gate_y_label", "component": "Text", "text": "Yes"}
    assert got.get("__gate_n_label") == {"id": "__gate_n_label", "component": "Text", "text": "No"}
    assert got["root"]["children"] == ["y", "n"]


@pytest.mark.parametrize(
    "raw", [RB_CANVAS, YES_NO, PLACEHOLDER, UNGUIDED], ids=["rb", "yes-no", "strict", "unguided"]
)
def test_f1_resample_never_names_a_reserved_id(raw: str) -> None:
    """The 2026-10-17 clarification: gate-reserved ids never appear in ``GateResult.resample``."""
    result = gate_once(raw)
    assert [cid for cid in result.resample if cid.startswith("__gate_")] == [], result.resample


@pytest.mark.parametrize("raw", [PLACEHOLDER, UNGUIDED], ids=["strict", "unguided"])
def test_f1_grounding_that_changes_a_repair_passes_the_original(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """All-or-nothing after grounding (2026-10-17 clarification): the repair returns ``f`` with the model's label
    plus an optional ``value`` the model never wrote. Grounding would strip ``value`` and leave a *valid* but changed
    TextField; instead the original placeholder-only ``f`` passes through unchanged, and so is pruned as at the
    base. The rest renders."""
    real = getattr(module(), NAME)

    def adding(components: list[Any], validator: Any) -> Any:
        out, edits = real(components, validator)
        out = [
            {**c, "value": INJECTED} if isinstance(c, dict) and c.get("id") == "f" else c
            for c in out
        ]
        return out, edits

    patch_repair(monkeypatch, adding)
    result = gate_once(raw)
    assert result.surface is not None, result.report
    got = emitted(result)
    assert INJECTED not in json.dumps(list(result.surface.messages))
    assert "f" not in got, (
        f"all-or-nothing after grounding: a grounded-changed repair was emitted: {got.get('f')}"
    )
    assert got["h"]["text"] == "Your trip"
    assert component_repairs(result) == 0, "the reverted repair is not counted (reviewer A)"


#: no ``Text`` substring anywhere (not even ``TextField``): a DateTimeInput with no ``value`` (rule DT)
WHEN = (
    CREATE
    + "\n"
    + update(
        '{"id":"root","component":"Column","children":["d"]}',
        '{"id":"d","component":"DateTimeInput","label":"When"}',
    )
)


@pytest.mark.parametrize("raw", [WHEN, WHEN[:-1]], ids=["strict", "unguided"])
def test_text_is_exempt_only_on_the_gate_made_text(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """Reviewer A / 2026-10-17 clarification: ``"Text"`` is exempt from grounding only as the ``component`` of the
    one RB-made Text. A non-RB repair (DT on ``d``) that carries an invented ``"Text"`` value (as ``d``'s label) is
    grounded like any other string: grounding would change it, so the original passes through (and is pruned), and
    ``"Text"`` never reaches the renderer. Fails if ``"Text"`` is exempted everywhere."""
    assert "Text" not in raw, "premise: the model never wrote Text"
    real = getattr(module(), NAME)

    def texting(components: list[Any], validator: Any) -> Any:
        out, edits = real(components, validator)
        out = [
            {**c, "label": "Text"} if isinstance(c, dict) and c.get("id") == "d" else c for c in out
        ]
        return out, edits

    patch_repair(monkeypatch, texting)
    result = gate_once(raw)
    assert result.surface is not None, result.report
    text = json.dumps(list(result.surface.messages))
    assert '"Text"' not in text, f"an invented Text string was emitted: {text}"
    assert "d" not in emitted(result), emitted(result).get("d")


# --- reviewer A, F2: a cycle formed through two lifts -------------------------------------------------------

TWO_LIFT_CYCLE = (
    CREATE
    + "\n"
    + update(
        '{"id":"root","component":"Column","children":["A"]}',
        '{"id":"A","component":"Column","children":[{"id":"K","component":"Column","children":["B"]}]}',
        '{"id":"B","component":"Column","children":[{"id":"J","component":"Column","children":["A"]}]}',
    )
)


def test_f2_two_lift_cycle_emits_what_the_base_emitted() -> None:
    """F2 / criterion 7 (2026-10-17 clarification: cycles through two or more lifts): lifting K out of A and J out of
    B would close A -> K -> B -> J -> A. The lifts are refused, A and B stay invalid and are pruned, and the gate
    emits exactly what the base gate (``BASE_GATE_SHA``) emitted -- a surface with ``root`` -- not "no surface"."""
    base = base_gate_module().UiGate().gate(TWO_LIFT_CYCLE, turn_id=TURN, canvas_idx=0, final=False)
    assert base.surface is not None and set(emitted(base)) == {"root"}, (
        "premise: the base emits root alone"
    )
    result = gate_once(TWO_LIFT_CYCLE)
    assert result.surface is not None, f"F2: no surface: {result.report}"
    assert list(result.surface.messages) == list(base.surface.messages)
