"""The structural stage inside ``UiGate``: where it sits, what it counts, what it must not do.
T-0108 criteria 1, 2, 3 (N5), 4 (unrepaired classes), 6, 7 and 9.

Most rows need code that does not exist at the branch base (``local_astra/ui/bracket_repair.py``,
``GateResult.structural_repairs``, the totality guards); they were pending until activation.
The rows whose docstrings say "Guard" hold at 14cde78 and must still hold after the task (valid
input handled as today, the D3 negative, the unrepaired classes' outcome on strict-valid canvases).

The expected outcomes of the unrepaired classes are the 14cde78 gate's, recorded in
``tests/fixtures/a2ui/t0108_canvases.jsonl`` (``baseline_fresh_gate``) when the fixture was built
at the branch base -- criterion 4's "keep today's handling" is a statement about that baseline.

Regime: CPU only, no service, deterministic, n = 1 per row.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import ui_gate as ui_gate_module
from local_astra.conductor.llm_client import UI_FENCE_MAX_CHARS
from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.conductor.ui_instructions import UI_EXAMPLE
from local_astra.ui import parse_concatenated_json
from tests.unit.test_component_repair import grounding_strings, reserved_ok

REPO = Path(__file__).resolve().parents[2]
CANVASES = REPO / "tests" / "fixtures" / "a2ui" / "t0108_canvases.jsonl"

#: the module the task names, and the function name ``tests/unit/test_bracket_repair.py`` pins
REPAIR_MODULE = "local_astra.ui.bracket_repair"
REPAIR_NAME = "repair_structure"

CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
#: every component catalog-valid, so salvage has nothing to do (the session rig's canvas)
VALID = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"lesson_1","catalogId":"' + CATALOG + '"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"lesson_1","components":['
    '{"id":"root","component":"Column","children":["a"]},'
    '{"id":"a","component":"Text","text":"one half"}]}}'
)
#: the same canvas with one duplicated opener: structurally broken, nothing else wrong
DUPLICATED = VALID.replace('{"id":"a"', '{"{"id":"a"')
#: …and with one spurious closer before the expected ``}``
SPURIOUS = VALID.replace('"one half"}', '"one half"]}')

INJECTED_TEXT = "zz-injected-not-in-raw-7f3a"
INJECTED_ID = "zz-injected-id-91c2"


def repair_module() -> Any:
    try:
        return importlib.import_module(REPAIR_MODULE)
    except ModuleNotFoundError as missing:
        if missing.name == REPAIR_MODULE:
            raise NotImplementedError(f"T-0108: {REPAIR_MODULE} does not exist yet") from missing
        raise


def structural_repairs(result: GateResult) -> int:
    value = getattr(result, "structural_repairs", None)
    assert isinstance(value, int) and not isinstance(value, bool), (
        "criterion 1 / *Owned files*: GateResult gains `structural_repairs` (int, default 0); "
        f"the result has {value!r}"
    )
    return value


def fixture_rows() -> list[dict[str, Any]]:
    return [json.loads(x) for x in CANVASES.read_text(encoding="utf-8").splitlines() if x.strip()]


def rows_of(*classes: str) -> list[dict[str, Any]]:
    out = [r for r in fixture_rows() if r["class"] in classes]
    assert out, f"fixture: no row of {classes}"
    return out


def strings_of(value: Any) -> Iterator[str]:
    """T-0108b criterion 9(i): every string value, and every key except one in the catalog vocabulary of the
    enclosing component's kind (C2); keys outside a component are still yielded."""
    yield from grounding_strings(value)


def grounded_in(text: str, bodies: list[str]) -> bool:
    """A string is grounded when some body holds it literally or JSON-escaped."""
    escaped = json.dumps(text)[1:-1]
    return any(text in b or escaped in b for b in bodies)


def gate_once(text: str, turn_id: str = "t", canvas_idx: int = 1) -> GateResult:
    return UiGate().gate(text, turn_id=turn_id, canvas_idx=canvas_idx, final=False)


# --- criterion 2: valid input is handled as today (guards) --------------------------------------


@pytest.mark.parametrize("name", ["VALID", "UI_EXAMPLE", "quiz"])
def test_a_strict_valid_canvas_is_handled_as_today(name: str, quiz_example_text: str) -> None:
    """Guard (T-0010 A5): strict-valid text never reaches a fixer: ``repaired == 0`` and the
    fixer step returns ``parse_concatenated_json``'s messages."""
    text = {"VALID": VALID, "UI_EXAMPLE": UI_EXAMPLE, "quiz": quiz_example_text}[name]
    gate = UiGate()
    messages, edits = gate.repair(text)
    assert edits == 0
    assert messages == [m for m in parse_concatenated_json(text) if isinstance(m, dict)]
    result = gate_once(text)
    assert result.repaired == 0 and result.surface is not None and result.surface.repaired == 0


@pytest.mark.parametrize("name", ["VALID", "UI_EXAMPLE", "quiz"])
def test_a_strict_valid_canvas_counts_zero_structural_repairs(
    name: str, quiz_example_text: str
) -> None:
    """Criterion 2: on valid input the new field reads 0, and the repair is never consulted."""
    text = {"VALID": VALID, "UI_EXAMPLE": UI_EXAMPLE, "quiz": quiz_example_text}[name]
    assert structural_repairs(gate_once(text)) == 0


@pytest.mark.parametrize("cls", ["duplicate_id", "inline_children"])
def test_strict_valid_real_canvases_gate_exactly_as_at_14cde78(cls: str) -> None:
    """Guard (criteria 2 and 4, unrepaired classes): same messages, ``repaired``, report problems
    and verdict as the 14cde78 gate recorded in the fixture."""
    for r in rows_of(cls):
        parse_concatenated_json(r["text"])  # precondition: these two are strict-valid
        base = r["baseline_fresh_gate"]
        result = UiGate().gate(
            r["text"], turn_id=r["request_id"], canvas_idx=r["canvas_idx"], final=False
        )
        assert (result.surface is not None) == base["surface"]
        assert result.repaired == base["repaired"] == 0
        # T-0127 re-scope R4 (criterion 3 and the no-surface ruling): loss lines are added only when a surface is
        # emitted. duplicate_id emits none, so its problems are the base's; inline_children loses its root Column
        # (LI refuses: the inline children need repairs themselves), so exactly one line naming root and Column.
        problems = list(result.report.problems)
        assert problems[: len(base["problems"])] == base["problems"]
        added = problems[len(base["problems"]) :]
        if cls == "duplicate_id":
            assert added == [], added
        else:
            assert len(added) == 1 and "root" in added[0] and "Column" in added[0], added
        got = list(result.surface.messages) if result.surface is not None else None
        assert got == base["messages"], f"{cls}: the emitted messages changed"


# --- criterion 1: the stage and its count --------------------------------------------------------


@pytest.mark.parametrize(
    "raw", [DUPLICATED, SPURIOUS], ids=["duplicated_opener", "spurious_closer"]
)
def test_a_structural_only_canvas_yields_a_surface_and_counts_its_edits(raw: str) -> None:
    """Criterion 1: ``edits`` = structural + json_repair log + salvage. Here the repaired text is
    strict-valid (json_repair's log is empty) and every component is catalog-valid (no salvage),
    so ``repaired`` is exactly the structural count, and that is the repair function's own."""
    with pytest.raises(json.JSONDecodeError):
        parse_concatenated_json(raw)
    result = gate_once(raw)
    assert result.surface is not None, f"the canvas yields a surface; {result.report}"
    counted = structural_repairs(result)
    _, edits = repair_module().__dict__[REPAIR_NAME](raw)
    assert counted == edits >= 1, (
        f"criterion 1: structural_repairs counts the structural edits ({edits}); it says {counted}"
    )
    assert result.repaired == counted, (
        f"criterion 1: with no json_repair log and no salvage, repaired == structural_repairs; "
        f"repaired={result.repaired}, structural_repairs={counted}"
    )
    assert result.surface.repaired == result.repaired


def test_the_repair_runs_only_after_strict_parsing_fails(
    monkeypatch: pytest.MonkeyPatch, quiz_example_text: str
) -> None:
    """Criterion 1 / mutation 7: valid text never reaches the structural pass; broken text does."""
    calls: list[str] = []
    module = repair_module()
    real = getattr(module, REPAIR_NAME)

    def spy(text: str) -> tuple[str, int]:
        calls.append(text)
        result: tuple[str, int] = real(text)
        return result

    patch_repair(monkeypatch, module, spy)
    for text in (VALID, UI_EXAMPLE, quiz_example_text):
        gate_once(text)
    assert calls == [], "criterion 1: the repair is called only after parse_concatenated_json fails"
    gate_once(DUPLICATED)
    assert calls == [DUPLICATED], f"…and then on the raw canvas, once; calls={len(calls)}"


# --- criterion 3 (N5): grounding against the original raw ----------------------------------------


def patch_repair(
    monkeypatch: pytest.MonkeyPatch, module: Any, fake: Callable[[str], tuple[str, int]]
) -> None:
    """Replace the repair where the gate can reach it: the module attribute, and a same-named
    global of ``ui_gate`` if the gate imported the function by name."""
    monkeypatch.setattr(module, REPAIR_NAME, fake)
    if hasattr(ui_gate_module, REPAIR_NAME):
        monkeypatch.setattr(ui_gate_module, REPAIR_NAME, fake)


INJECTED_VALID = VALID.replace('"children":["a"]}', f'"children":["a","{INJECTED_ID}"]}}').replace(
    '"text":"one half"}]}}',
    f'"text":"one half"}},{{"id":"{INJECTED_ID}","component":"Text","text":"{INJECTED_TEXT}"}}]}}}}',
)


@pytest.mark.parametrize(
    "path", ["strict_after_repair", "json_repair_after_repair"], ids=lambda p: p
)
def test_gate_grounds_repair_output_against_raw(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    """N5 (defense in depth; mutation 5): a repair that *invents* a component is caught by the
    gate's grounding against the original ``raw`` -- when its output parses strictly, and when it
    still needs json_repair. The rest of the canvas still renders, so the row is not vacuous."""
    parse_concatenated_json(INJECTED_VALID)  # the fake's output is valid JSON
    assert INJECTED_TEXT not in DUPLICATED and INJECTED_ID not in DUPLICATED
    returned = INJECTED_VALID if path == "strict_after_repair" else INJECTED_VALID[:-2]
    calls: list[str] = []

    def inventing_repair(text: str) -> tuple[str, int]:
        calls.append(text)
        return returned, 1

    patch_repair(monkeypatch, repair_module(), inventing_repair)
    result = gate_once(DUPLICATED)
    assert calls, (
        f"setup: the gate must reach the structural repair as {REPAIR_MODULE}.{REPAIR_NAME} (or a "
        f"same-named ui_gate global); the fake was never called"
    )
    assert result.surface is not None, (
        f"the grounded rest of the canvas still renders; {result.report}"
    )
    emitted = json.dumps(list(result.surface.messages))
    assert INJECTED_TEXT not in emitted and INJECTED_ID not in emitted, (
        f"N5 ({path}): a string that is not in raw reached the renderer: {emitted}"
    )
    texts = {c.get("text") for m in result.surface.messages for c in components(m)}
    assert "one half" in texts, f"the model's own Text survives; emitted {emitted}"


def components(message: Any) -> list[dict[str, Any]]:
    body = message.get("updateComponents") if isinstance(message, dict) else None
    comps = body.get("components") if isinstance(body, dict) else None
    return [c for c in comps or () if isinstance(c, dict)]


# --- criterion 4: unrepaired classes keep today's handling -----------------------------------------


@pytest.mark.parametrize("cls", ["header_cut", "non_a2ui_json"])
def test_canvases_with_no_surface_today_still_have_none(cls: str) -> None:
    """Guard (criteria 4 and 7): a canvas cut inside its header, and a D3-shaped canvas (non-A2UI
    JSON, 3ac00a1 arm D3), yield no surface -- at 14cde78 and after."""
    for r in rows_of(cls):
        assert r["baseline_fresh_gate"]["surface"] is False, "fixture: no surface at 14cde78"
        result = UiGate().gate(
            r["text"], turn_id=r["request_id"], canvas_idx=r["canvas_idx"], final=False
        )
        assert result.surface is None, f"{cls}: {r['request_id']} must not render"
        assert not result.report.ok


def test_unrepaired_classes_get_no_structural_edit_there() -> None:
    """Criterion 4: the structural pass has no rule for a duplicate id, inline children or a cut
    header (zero edits on those canvases), and never writes a missing ``component`` key."""
    fn = repair_module().__dict__[REPAIR_NAME]
    for r in rows_of("duplicate_id", "inline_children", "header_cut"):
        assert fn(r["text"]) == (r["text"], 0), f"{r['class']}: {r['request_id']} was edited"
    for r in rows_of("missing_component_key"):
        out, _ = fn(r["text"])
        assert out.count('"component"') == r["text"].count('"component"'), (
            "criterion 4: a component with no `component` key is not given one by the repair"
        )
        result = UiGate().gate(
            r["text"], turn_id=r["request_id"], canvas_idx=r["canvas_idx"], final=False
        )
        structural_repairs(result)
        for m in result.surface.messages if result.surface else ():
            for c in components(m):
                assert isinstance(c.get("component"), str), f"emitted without a kind: {c}"


# --- criterion 9: the gate never raises -------------------------------------------------------------

NESTING = {
    "[*3000": "[" * 3000,
    '{"a":*1000': '{"a":' * 1000,
    "[{*1000": "[{" * 1000,
    "[*8192": "[" * 8192,
    "[{*4096": "[{" * 4096,
    # Reviewer A, 1249 chars: broken (a spurious `]` after "root"), parses strictly *after* the
    # structural repair, then recurses in grounding/validation (500 nested lists). Only the outer
    # RecursionError guard of `gate()` can turn this into a result (mutation 10).
    "repaired-then-deep[*500": (
        '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"' + CATALOG + '"}}\n'
        '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
        '{"id":"root"],"component":"Text","text":' + "[" * 500 + "]" * 500 + "}]}}"
    ),
}
#: inputs that the 14cde78 gate already survives; they stay guards (no marker)
SURVIVED = {"]*8192": "]" * 8192, '{"{"*2048': '{"{"' * 2048}


def gate_without_raising(text: str) -> GateResult:
    try:
        return gate_once(text)
    except (RecursionError, ValueError) as err:
        raise AssertionError(
            f"criterion 9: UiGate.gate raised {type(err).__name__} on {len(text)} chars "
            f"({text[:12]!r}…): {err}"
        ) from err


@pytest.mark.parametrize("name", sorted(NESTING))
def test_deep_nesting_returns_a_result_naming_nesting(name: str) -> None:
    """Criterion 9 (H-T108 (e)): ``RecursionError`` (``parse_concatenated_json``) and
    ``ValueError`` (json_repair's nesting limit) never escape; the verdict names nesting."""
    text = NESTING[name]
    assert len(text) <= UI_FENCE_MAX_CHARS
    result = gate_without_raising(text)
    assert isinstance(result, GateResult)
    assert result.surface is None and not result.report.ok
    assert any("nest" in p.lower() for p in result.report.problems), (
        f"criterion 9: the problems list names nesting; it is {result.report.problems}"
    )


@pytest.mark.parametrize("name", sorted(SURVIVED))
def test_pathological_flat_input_returns_a_result(name: str) -> None:
    """Guard (criterion 9; ``'{"{"'*2048`` is json_repair's 0.86 s case, H-T108 (f) out of scope):
    these return a result with no surface today and must keep doing so."""
    text = SURVIVED[name]
    assert len(text) == UI_FENCE_MAX_CHARS
    result = gate_without_raising(text)
    assert isinstance(result, GateResult) and result.surface is None


# --- criterion 6: streaming and multi-canvas ---------------------------------------------------------


def cut_points(row: dict[str, Any]) -> list[int]:
    text = row["text"]
    cuts = {len(p) for p in row["prior_texts"] if text.startswith(p)}
    cuts |= set(range(64, len(text), 64))
    cuts.add(len(text))
    return sorted(c for c in cuts if c > 0)


def test_streaming_bodies_never_raise_stay_grounded_and_end_on_a_surface() -> None:
    """Criterion 6 (ruling A10): every repaired-class body, cut at each earlier canvas of its
    trial and at every 64th char, through one gate and one turn as the session drives it. No
    call raises; every emitted string is in the union of the bodies so far; the last call (the
    whole body) yields a surface."""
    failures: list[str] = []
    for r in fixture_rows():
        if not r["repaired_class"]:
            continue
        gate = UiGate()
        bodies: list[str] = []
        last: GateResult | None = None
        for i, cut in enumerate(cut_points(r)):
            body = r["text"][:cut]
            bodies.append(body)
            try:
                last = gate.gate(body, turn_id=r["request_id"], canvas_idx=i, final=False)
            except Exception as err:
                failures.append(f"{r['request_id']}#{r['ui_index']} raised at {cut}: {err!r}")
                break
            if last.surface is None:
                continue
            for s in strings_of(list(last.surface.messages)):
                # T-0108b (H-T108b (d), C7): exactly the reserved namespace is exempt
                if (
                    s != f"turn-{r['request_id']}"
                    and not grounded_in(s, bodies)
                    and not reserved_ok(s, bodies)
                ):
                    failures.append(f"{r['request_id']} cut {cut}: ungrounded {s!r}")
        else:
            if last is None or last.surface is None:
                failures.append(f"{r['request_id']}#{r['ui_index']} ({r['class']}): no surface")
    assert not failures, "criterion 6:\n" + "\n".join(failures)
