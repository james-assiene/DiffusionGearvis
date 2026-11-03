"""T-0184 C4, the renderer half: the v2 prompt example, gated, renders exactly its strings (contract; activated once B landed).

The example is read out of ``ui_instructions.UI_INSTRUCTIONS_V2`` (its one ``UI_OPEN … UI_CLOSE`` span), fenced and split
by the real ``llm_client.split_segments``, gated by the real ``UiGate()``, and the emitted messages are applied, one by
one, to the HUD's own engine (``hud/src/ui/engine.js``) in node, with both catalog ids from ``hud/src/ui/a2ui.js``
(``GATE_CATALOG_ID`` and group B's ``CARDS_V2_CATALOG_ID``, B1) and the trapping fake DOM of
``tests/hud/ui_render.test.mjs`` (imported between its ``// <fake-dom>`` markers, as ``ui_engine.test.mjs`` does). The
row reads every non-blank text node under the card root in document order.

**Expected, from the example alone (never from the JS):** the model's strings of the components ``root`` reaches, in
tree order, by role: Text ``text``; Stat ``label``, ``value``, ``detail``; StatTiles per tile ``title``, ``value``,
``secondary``; Comparison per column ``title``, ``badge``, then per row ``label`` and each cell (a string, or its
``text``); Bars ``title`` then each bar ``label``; Steps per step ``title``, ``detail``; Source ``text``; Actions each
label. They must appear in that order, byte-equal. Whatever text is left over may only be B4's derived strings: a number
(a Bars value, a tick in [0, max]), the Bars ``unit`` (alone or glued to a number), ``<percent>%`` of a tile, a step
ordinal ``1…n``. "Recommended", "From a web search" or any other word fails the row.

**Amended at activation (T-0184 B test-author, 2026-10-28):** B7a's ruling lets the scale end lie up to one step past
the largest value (0..1000 for 828, the example's own Bars). A plain number above the largest value is therefore
allowed only when the residue's plain numbers, less one ``String(value)`` per bar, form a valid tick scale
(``_is_scale``: 0 first, equal spacing, a step from the closed list, the end under one step past the max). At
activation the renderer drew 0, 250, 500, 750, 1000 here, and the old ``[0, max]`` bound refused 1000.

Pending until group B merged (the renderer for the v2 kinds and ``CARDS_V2_CATALOG_ID`` in ``a2ui.js``); activated
on ``T-0184-b-renderer`` after main (group C) was merged in. Not parametrised over ``backend``: no service is involved, only the page's own JavaScript in node.

Regime: node >= 18.19 (the box: v18.19.1), CPU, no socket; n = 1, deterministic.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN, split_segments
from local_astra.conductor.ui_gate import UiGate
from tests.unit.test_t0184_prompt_block import example_v2

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")

RUNNER = r"""
import { readFileSync, cpSync, mkdtempSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";
const [repo, messagesPath] = process.argv.slice(2);
const dir = mkdtempSync(join(tmpdir(), "astra-t0184c-"));
writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
cpSync(join(repo, "hud/src"), join(dir, "src"), { recursive: true });
const root = pathToFileURL(join(dir, "src") + "/");
const src = readFileSync(join(repo, "tests/hud/ui_render.test.mjs"), "utf8");
const m = /\/\/ <fake-dom>[^\n]*\n([\s\S]*?)\n\/\/ <\/fake-dom>/.exec(src);
if (!m) throw new Error("tests/hud/ui_render.test.mjs has no // <fake-dom> region");
const FAKE = await import("data:text/javascript," + encodeURIComponent(m[1]));
const a2ui = await import(new URL("ui/a2ui.js", root).href);
const { createEngine } = await import(new URL("ui/engine.js", root).href);
const out = { ids: { gate: a2ui.GATE_CATALOG_ID ?? null, v2: a2ui.CARDS_V2_CATALOG_ID ?? null }, texts: null, logs: [], roots: 0 };
if (out.ids.v2) {
  const dom = FAKE.makeDom();
  let card = null;
  const engine = createEngine({ document: dom.document, onAction: () => {}, log: (l) => out.logs.push(String(l)), catalogIds: [out.ids.gate, out.ids.v2] });
  engine.processor.onSurfaceCreated((s) => {
    out.roots += 1;
    card = s.element;
    dom.quiet(() => dom.document.body.appendChild(s.element));
  });
  for (const msg of JSON.parse(readFileSync(messagesPath, "utf8"))) engine.processor.processMessages([msg]);
  const texts = [];
  const walk = (n) => { for (const c of n.childNodes) { if (c.nodeType === 3 && c.data.trim()) texts.push(c.data); if (c.nodeType === 1) walk(c); } };
  if (card) walk(card);
  out.texts = texts;
  out.violations = dom.rec.violations;
}
process.stdout.write(JSON.stringify(out));
"""

_NUMBER = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"


def _ordered_strings(components: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """The model's strings ``root`` reaches, in tree order, and the data B4's derived strings may come from."""
    out: list[str] = []
    derived: dict[str, Any] = {"bar_values": [], "units": [], "percents": [], "steps": 0}
    seen: set[str] = set()

    def visit(cid: str) -> None:
        if cid in seen or cid not in components:
            return
        seen.add(cid)
        c = components[cid]
        kind = c.get("component")
        if kind == "Text":
            out.append(c["text"])
        elif kind in ("Column", "Row", "List"):
            for child in c.get("children", []):
                visit(child)
        elif kind == "Card":
            visit(c["child"])
        elif kind == "Divider":
            pass
        elif kind == "Stat":
            out.extend(c[k] for k in ("label", "value", "detail") if k in c)
        elif kind == "StatTiles":
            for t in c["tiles"]:
                out.extend(t[k] for k in ("title", "value", "secondary") if k in t)
                if "percent" in t:
                    derived["percents"].append(t["percent"])
        elif kind == "Comparison":
            for col in c["columns"]:
                out.extend(col[k] for k in ("title", "badge") if k in col)
            for row in c["rows"]:
                out.append(row["label"])
                out.extend(cell if isinstance(cell, str) else cell["text"] for cell in row["cells"])
        elif kind == "Bars":
            if "title" in c:
                out.append(c["title"])
            out.extend(b["label"] for b in c["bars"])
            derived["bar_values"].extend(b["value"] for b in c["bars"])
            if "unit" in c:
                derived["units"].append(c["unit"])
        elif kind == "Steps":
            for s in c["steps"]:
                out.extend(s[k] for k in ("title", "detail") if k in s)
            derived["steps"] += len(c["steps"])
        elif kind == "Source":
            out.append(c["text"])
        elif kind == "Actions":
            out.extend(c["labels"])
        else:
            raise AssertionError(
                f"the example uses {kind!r}, which this row does not read: extend the oracle"
            )

    visit("root")
    return [s for s in out if s.strip()], derived


#: B4 (ii) / B7a's closed list of tick steps: one significant digit, or 2.5, times a power of ten
#: (the reading of tests/hud/ui_cards_v2.test.mjs).
_NICE = (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)


def _is_scale(ticks: list[float], vmax: float) -> bool:
    """B4 (ii) with B7a's ruling (2026-10-28): ticks are 0 first, equally spaced by a step from
    ``_NICE`` times a power of ten, and the scale end lies at least at half the largest value and
    less than one step past it (so 0..1000 for a largest bar of 828)."""
    if len(ticks) < 2 or ticks[0] != 0:
        return False
    step = ticks[1] - ticks[0]
    if step <= 0 or any(abs(b - a - step) > 1e-9 * step for a, b in itertools.pairwise(ticks)):
        return False
    exp = 10.0 ** math.floor(math.log10(step))
    if not any(math.isclose(step / exp, n) for n in _NICE):
        return False
    return ticks[-1] >= vmax / 2 and ticks[-1] - step < vmax


def _beyond_max_ok(residue: list[str], derived: dict[str, Any]) -> set[str]:
    """The plain numbers above the largest bar that are the end of a valid tick scale: the
    residue's plain numbers, less one ``String(value)`` per bar, read as the ticks in order."""
    values = [float(v) for v in derived["bar_values"]]
    if not values:
        return set()
    left = list(values)
    ticks: list[float] = []
    for t in residue:
        if not re.fullmatch(_NUMBER, t):
            continue
        n = float(t)
        if n in left:
            left.remove(n)
        else:
            ticks.append(n)
    vmax = max(values)
    return (
        {t for t in residue if re.fullmatch(_NUMBER, t) and float(t) > vmax}
        if _is_scale(ticks, vmax)
        else set()
    )


def _allowed_residue(token: str, derived: dict[str, Any]) -> bool:
    units = [re.escape(u) for u in derived["units"]]
    if token in derived["units"]:
        return True
    m = (
        re.fullmatch(rf"({_NUMBER})({'|'.join(units)})?", token)
        if units
        else re.fullmatch(rf"({_NUMBER})", token)
    )
    if m:
        n = float(m.group(1))
        values = [float(v) for v in derived["bar_values"]]
        if values and (n in values or 0 <= n <= max(values)):
            return True  # String(value), or a tick on the 0..max scale
        if (
            derived["steps"]
            and m.group(2) is None
            and n.is_integer()
            and 1 <= n <= derived["steps"]
        ):
            return True  # a step ordinal
    p = re.fullmatch(rf"({_NUMBER})%", token)
    return bool(p) and any(math.isclose(float(p.group(1)), float(x)) for x in derived["percents"])


@pytest.mark.skipif(
    NODE is None, reason="node >= 18.19 is a dev prerequisite for the HUD JS layer (T-0008 Regime)"
)
def test_the_v2_example_renders_exactly_its_strings(tmp_path: Path) -> None:
    pieces = split_segments(None, UI_OPEN + example_v2() + UI_CLOSE, False)
    assert [p[0] for p in pieces] == ["ui"], pieces
    result = UiGate().gate(pieces[0][1], turn_id="ex", canvas_idx=0, final=True)
    assert result.surface is not None and result.complete, list(result.report.problems)[:3]
    messages = list(result.surface.messages)
    components = {
        c["id"]: c
        for m in messages
        if "updateComponents" in m
        for c in m["updateComponents"]["components"]
    }
    expected, derived = _ordered_strings(components)
    assert expected, "the example holds no model string"

    (tmp_path / "messages.json").write_text(json.dumps(messages), encoding="utf-8")
    (tmp_path / "run.mjs").write_text(RUNNER, encoding="utf-8")
    proc = subprocess.run(
        [str(NODE), str(tmp_path / "run.mjs"), str(REPO), str(tmp_path / "messages.json")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"the node runner failed: {proc.stderr[-800:]}"
    out = json.loads(proc.stdout)
    assert out["ids"]["v2"] == "gearvis:cards/v2", (
        f"B1: hud/src/ui/a2ui.js exports no CARDS_V2_CATALOG_ID: {out['ids']}"
    )
    assert out["roots"] == 1, (
        f"the engine handed out {out['roots']} card roots; logs: {out['logs'][:5]}"
    )
    assert out["violations"] == [], out["violations"]

    rendered = "\n".join(out["texts"])
    cursor, residue = 0, []
    for s in expected:
        at = rendered.find(s, cursor)
        assert at >= 0, (
            f"C4: {s!r} is missing, or out of tree order, in the rendered text {out['texts']}"
        )
        residue.append(rendered[cursor:at])
        cursor = at + len(s)
    residue.append(rendered[cursor:])
    tokens = [t for chunk in residue for t in chunk.split()]
    beyond = _beyond_max_ok(tokens, derived)
    extra = [t for t in tokens if not (_allowed_residue(t, derived) or t in beyond)]
    assert extra == [], f"C4/B4: the renderer wrote words the model did not: {extra}"
    if derived["units"] and derived["bar_values"]:
        assert any(u in rendered for u in derived["units"]), "B4: the Bars unit is not shown"
