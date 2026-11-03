"""T-0210 group B, criteria B5-B11: the pinned three.js editor copy through the real web-mode executor, a real live
``Session`` with a scripted fake LLM for the golden paths (docs/tasks/T-0210.md, "Tests to write",
``tests/integration/test_t0210_threejs_session.py``; rulings B-R1..B-R6 in the task's Status line).

**Gate.** T-0111c's autouse ``_browser_layer`` (imported through the T-0205 browser module): ``integration`` +
``browser``, runs only with ``ASTRA_BROWSER_TESTS=1``; ``make browser-threejs`` (B12) runs it.

**The rig** (B5, T-0205 A10's pattern). ``StartPageServer(static_dir=THREEJS_DIR)`` on loopback and T-0205 A7's
web-mode executor (``web_executor``: ``web=True``, ``start_origin`` = ``allowed_origins`` = the copy's origin, the
deny-all resolver and connector of :class:`DenyNet`, ``echo="model"``). Per row: a fresh server and executor;
``ex.driver.ensure()``, then ``page.on("pageerror")`` on the agent page, then ``ex.open(<origin><start_page>)``, then a
test-side wait for ``__state.ready === true``. The golden rows run T-0111g's ``tool_rig`` with ``live_kit(Watch(ex))``
and ``tsl.loop_policy(20)``; the fake LLM holds each step, and before releasing step k+1 the test computes its call from
the current DOM (never from ``__state``): visible text inside ``#menubar`` (menu titles, options: an element's *own*
text nodes, so "Mesh" is not "Mesh Box Sphere ..."), ``.Outliner`` options (first word = the object's name),
``.Tabs`` tabs, and a sidebar ``.Row`` whose ``.Label`` reads the label, to its nth visible ``<input>``. The 0-999
coordinate is T-0205's :func:`unit` of the box. B6-B11 drive the same executor directly (T-0205's ``execute``/``inv``).

**Scrolling the sidebar** (B5 scroll risk). ``scroll`` is ``mouse.wheel`` at the pointer, so before a scroll the
planner moves the pointer over the sidebar with a ``click_at`` on a visible sidebar ``.Label`` (a ``<span>``; a click
on it does nothing in the editor) unless its last click was already inside the sidebar. A field is "in view" when
the topmost element at its centre is the field (``elementFromPoint``).

**404s** (B-R5). Counted server-side: ``pages.server.resolve`` (the handler's global, looked up per request) is
wrapped for the row and every ``None`` (a 404) is recorded with its target.

**Readings this module pins where the task is silent** (reported to the orchestrator):

- B5's step lists are typed from B5; M4 picks Add › Light › "Point" (B-R2). The spoken text is one sentence before
  step 1 and a final no-call answer; the judge reads ``voiced`` = every history entry's ``assistant_spoken``.
- "each ``type_at`` is ``executed`` with ``replaced: True`` and no ``submit_withheld``": every ``type_at`` outcome
  of a golden row; every other outcome is ``executed`` too (auxiliary pointer clicks and scrolls included).
- B6 "the menu's pixel region differs": the Add menu's ``.options`` box, scaled to the frame, mean absolute grey
  difference between the pre-click and post-click frames above :data:`MENU_DIFF` (8 grey levels).
- B8 runs after M1's Add › Mesh › Box (the box sits at the viewport centre, so the click can change the selection,
  which ``__state`` does not hold). After Merge T-0210-a (a subject starting ``Merge T-0210-a:`` in this checkout's
  history): ``status == "failed"`` (B-R9, A1's literal) with ``FOCUS_UNKNOWN_REASON``, ``focus_check == "body"``
  and ``typed: 0``, ``__state`` equal before and after, the toolbar's selected transform button unchanged. Before
  it: ``executed``. The row prints which.
- B9 (B-R1): ``press_key Delete`` and Edit › Delete each remove the selected Crate with no dialog (no ``dismissed``
  in the outcome); ``press_key Ctrl+Z`` brings back the identical Crate entry (uuid included).
- B10: zero 404s means every 404 of the row, including any request the browser makes by itself (``/favicon.ico``;
  B-R7 amended: the patched ``index.html`` names the local ``images/icon.png``, so none is expected);
  the positive controls are a page ``fetch`` of a missing path (counted) and an off-origin ``https`` image (seen by
  the deny-all seams).
- B11 "non-blank viewport": a screenshot clipped to the viewport canvas holds at least two distinct colours, and the
  canvas answers ``getContext("webgl2")``.

BASE = 26196a88. Written red ahead of the copy (pending ``T-0210``); activated at c81f1ebf; B8 is merge-aware.

Regime: Playwright 1.64.0, Chromium 156 headless 1280x800, DSF 1, sandbox on, web mode on loopback, deny-all resolver
and connector, ``ScriptedClock`` for the session, loop 20, three.js r179 (WebGL2 via SwiftShader), n = 1.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import json
import os
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.integration.test_computer_browser_e2e import execute, inv
from tests.integration.test_computer_live_session import (
    Q,
    Watch,
    ask,
    click,
    descendants,
    drive,
    non_loopback_sockets,
    release_step,
    type_at,
)
from tests.integration.test_t0205_scenarios_browser import (
    STATE_JS as T0205_STATE_JS,
)
from tests.integration.test_t0205_scenarios_browser import (
    DenyNet,
    H,
    W,
    _browser_layer,  # noqa: F401  (the autouse gate)
    _browser_marker_registered,
    run,
    unit,
    web_executor,
)
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import live_kit
from tests.unit._tool_turn_support import answer, tool, tool_rig
from tests.unit.test_start_pages import new_server

pytestmark = [
    pytest.mark.integration,
    *([pytest.mark.browser] if _browser_marker_registered() else []),
]

MODULE = "local_astra.pages.threejs_tasks"
SIDEBAR_W = 350
MENUBAR_H = 32
MAX_AUX = 12
MENU_DIFF = 8.0
READY_JS = "() => !!window.__state && window.__state.ready === true"
#: B-R23: the state is read with Playwright's own serializer, never with the page's ``JSON`` (a Play script runs in the
#: page's main world and can wrap ``JSON.stringify``). T-0205's ``STATE_JS`` is imported only as the attack's control.
STATE_READ_JS = "() => window.__state"
SCROLL_DOWN = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
SCROLL_UP = f"<|tool_call>call:scroll{{direction:{Q}up{Q}}}<tool_call|>"

#: A visible element matched by its own text nodes (``mode`` "exact": all own text; "first": its first word).
FIND_JS = """([sel, text, mode]) => {
  const own = (e) => Array.from(e.childNodes).filter(n => n.nodeType === 3).map(n => n.textContent)
                       .join(" ").replace(/\\s+/g, " ").trim().toLowerCase();
  const vis = (e) => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
                       return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
  const want = text.toLowerCase();
  const el = Array.from(document.querySelectorAll(sel)).filter(vis)
    .find(e => mode === "first" ? own(e).split(" ")[0] === want : own(e) === want);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, topmost: !!hit && (hit === el || el.contains(hit))};
}"""
#: The nth visible enabled ``<input>`` of the visible sidebar ``.Row`` whose ``.Label`` reads ``label``.
FIELD_JS = """([label, nth]) => {
  const vis = (e) => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
                       return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
  const want = label.toLowerCase();
  const row = Array.from(document.querySelectorAll("#sidebar .Row")).filter(vis).find(r =>
    Array.from(r.querySelectorAll(".Label")).some(l => l.textContent.trim().toLowerCase() === want));
  if (!row) return null;
  const inputs = Array.from(row.querySelectorAll("input")).filter(e => vis(e) && !e.disabled);
  const el = inputs[nth];
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, topmost: hit === el};
}"""
#: A visible sidebar ``.Label`` whose centre is topmost (a harmless place to put the pointer before a scroll).
LABEL_JS = """() => {
  for (const l of document.querySelectorAll("#sidebar .Label")) {
    const r = l.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0 || r.top < 80 || r.bottom > innerHeight - 4) continue;
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    if (hit === l || l.contains(hit)) return {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
  }
  return null;
}"""
#: The open box of the ``#menubar`` menu whose title reads ``title`` (null while its options are not displayed).
MENU_OPTIONS_JS = """(title) => {
  const m = Array.from(document.querySelectorAll("#menubar .menu")).find(m => {
    const t = m.querySelector(".title"); return t && t.textContent.trim().toLowerCase() === title.toLowerCase(); });
  if (!m) return null;
  const o = m.querySelector(".options");
  const s = getComputedStyle(o), r = o.getBoundingClientRect();
  if (s.display === "none" || r.width <= 0 || r.height <= 0) return null;
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
}"""
CANVAS_JS = """() => { const c = document.querySelector("#viewport canvas"); if (!c) return null;
  const r = c.getBoundingClientRect();
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, tabindex: c.hasAttribute("tabindex"),
          webgl2: !!c.getContext("webgl2")}; }"""
MODE_JS = """() => Array.from(document.querySelectorAll("#toolbar .Button")).map(b => b.classList.contains("selected"))"""


# =====================================================================================================
# helpers
# =====================================================================================================


def tt() -> Any:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in (MODULE, "local_astra.pages"):
            raise AssertionError(f"T-0210 B4: {MODULE} does not exist yet") from exc
        raise


def threejs_dir() -> Path:
    m = tt()
    assert hasattr(m, "THREEJS_DIR"), f"T-0210 B1: {MODULE} has no THREEJS_DIR"
    d = Path(m.THREEJS_DIR)
    assert (d / "editor" / "index.html").is_file(), (
        f"T-0210 B0: {d}/editor/index.html does not exist yet"
    )
    return d


def task(tid: str) -> Any:
    rows = {r.task_id: r for r in tt().TASKS}
    assert tid in rows, f"T-0210 B4: no task {tid}"
    return rows[tid]


def judge(tid: str, state: Any, voiced: str) -> Any:
    return tt().judge(tid, state, voiced)


def vdict(v: Any) -> dict[str, Any]:
    keys = (
        "success",
        "trap",
        "over_action",
        "checkout_pressed",
        "submit_pressed",
        "secret_typed",
        "notes",
    )
    return {k: getattr(v, k, None) for k in keys}


def a_merged() -> bool:
    """Whether this checkout's history holds a ``Merge T-0210-a:`` commit (B8's merge-aware row)."""
    proc = gr._git("log", "--format=%s")
    assert proc.returncode == 0, proc.stderr
    return any(s.startswith("Merge T-0210-a:") for s in proc.stdout.splitlines())


class Misses:
    """Wraps ``pages.server.resolve``: every ``None`` (a 404) is recorded with its target (B-R5)."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.targets: list[str] = []
        self.calls = 0

    def __call__(self, root: Path, target: str) -> Path | None:
        self.calls += 1
        out = self.real(root, target)
        if out is None:
            self.targets.append(target)
        return out


@dataclass
class Editor:
    """One row's server, executor, page and counters."""

    ex: Any
    origin: str
    page: Any
    net: DenyNet
    misses: Misses
    errors: list[str]
    outcomes: list[Any] = field(default_factory=list)
    #: set by B10's positive control (an off-origin request on purpose): the exit checks are then the row's own
    control: bool = False

    async def state(self) -> dict[str, Any]:
        st = await self.page.evaluate(STATE_READ_JS)
        assert isinstance(st, dict), f"T-0210 B3: no __state object: {st!r}"
        return st

    async def act(self, name: str, **args: Any) -> Any:
        f = self.ex.latest_frame
        assert f is not None, "no own frame yet"
        out = await execute(self.ex, inv(name, f, **args))
        self.outcomes.append(out)
        return out


@contextlib.asynccontextmanager
async def editor(seed_page: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Editor]:
    """A fresh copy server and web-mode executor at ``seed_page``, ready; on exit zero resolver/connector calls and
    zero egress records."""
    from local_astra.pages import server as server_mod

    root = threejs_dir()
    misses = Misses(server_mod.resolve)
    monkeypatch.setattr(server_mod, "resolve", misses)
    srv = new_server(root)
    srv.start()
    net = DenyNet()
    errors: list[str] = []
    try:
        async with web_executor(srv.origin, net) as ex:
            await run(ex.driver.ensure())
            page = ex.driver.page
            page.on("pageerror", lambda e: errors.append(str(e)))
            await run(ex.open(srv.origin + seed_page))
            await page.wait_for_function(READY_JS, timeout=30000)
            await run(ex.frame())
            ed = Editor(
                ex=ex, origin=srv.origin, page=ex.driver.page, net=net, misses=misses, errors=errors
            )
            yield ed
            if not ed.control:
                assert list(ex.egress_log) == [], ex.egress_log
    finally:
        srv.stop()
    if not ed.control:
        assert (net.resolves, net.connects) == ([], []), (
            f"T-0210 B10: the proxy seams were called: {net}"
        )


# --------------------------------------------------------------------------------------------- the planner


@dataclass
class Planner:
    """Turns intents into the next executor action, computed from the DOM now (pixel-first holds for the product:
    the coordinates are the test's, the page is the browser's). ``next`` returns ``(name, args, consumed)``;
    ``consumed`` is False for an auxiliary action (a pointer move or a scroll) after which the intent is asked again."""

    page: Any
    last_px: tuple[float, float] | None = None
    aux: int = 0

    async def _find(self, sel: str, text: str, mode: str = "exact") -> dict[str, Any]:
        b = await self.page.evaluate(FIND_JS, [sel, text, mode])
        assert b is not None, f"T-0210 B5: no visible {sel} reading {text!r} on {self.page.url}"
        return dict(b)

    def _at(self, b: dict[str, Any]) -> tuple[int, int]:
        x, y = unit(b["left"], b["right"], W), unit(b["top"], b["bottom"], H)
        self.last_px = ((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2)
        return x, y

    def _in_sidebar(self) -> bool:
        return (
            self.last_px is not None
            and self.last_px[0] >= W - SIDEBAR_W
            and self.last_px[1] > MENUBAR_H
        )

    async def _bring(self, b: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
        self.aux += 1
        assert self.aux <= MAX_AUX, "T-0210 B5: the field never came into view"
        if not self._in_sidebar():
            lab = await self.page.evaluate(LABEL_JS)
            assert lab is not None, "T-0210 B5: no visible sidebar label to put the pointer on"
            x, y = self._at(dict(lab))
            return "click_at", {"x": x, "y": y}, False
        up = b["top"] < (MENUBAR_H + 80)
        return "scroll", {"direction": "up" if up else "down"}, False

    async def next(self, intent: tuple[Any, ...]) -> tuple[str, dict[str, Any], bool]:
        kind = intent[0]
        if kind == "menu":
            x, y = self._at(await self._find("#menubar .menu .title", intent[1]))
            return "click_at", {"x": x, "y": y}, True
        if kind == "option":
            x, y = self._at(await self._find("#menubar .options .option", intent[1]))
            return "click_at", {"x": x, "y": y}, True
        if kind == "outliner":
            b = await self._find(".Outliner .option", intent[1], "first")
            x, y = self._at(b)
            return "click_at", {"x": x, "y": y}, True
        if kind == "tab":
            b = await self._find("#sidebar .Tabs .Tab", intent[1])
            if not b["topmost"]:
                return await self._bring(b)
            x, y = self._at(b)
            return "click_at", {"x": x, "y": y}, True
        if kind == "button":
            b = await self._find("#sidebar button.Button", intent[1])
            if not b["topmost"]:
                return await self._bring(b)
            x, y = self._at(b)
            return "click_at", {"x": x, "y": y}, True
        if kind == "field":
            _k, label, nth, text = intent
            b = await self.page.evaluate(FIELD_JS, [label, nth])
            assert b is not None, (
                f"T-0210 B5: no sidebar row {label!r} input {nth} on {self.page.url}"
            )
            b = dict(b)
            if not b["topmost"]:
                return await self._bring(b)
            x, y = self._at(b)
            self.aux = 0
            return "type_at", {"x": x, "y": y, "text": text, "press_enter": True}, True
        raise AssertionError(f"unknown intent {intent!r}")


def call_text(name: str, args: dict[str, Any]) -> str:
    if name == "click_at":
        return click(args["x"], args["y"])
    if name == "type_at":
        return type_at(args["x"], args["y"], args["text"], args["press_enter"])
    if name == "scroll":
        return SCROLL_DOWN if args["direction"] == "down" else SCROLL_UP
    raise AssertionError(name)


async def do(ed: Editor, plan: list[tuple[Any, ...]]) -> list[Any]:
    """Drive ``plan`` through the executor directly (B6-B9); every action's outcome, in order."""
    pl = Planner(ed.page)
    outs = []
    for intent in plan:
        while True:
            name, args, consumed = await pl.next(intent)
            out = await ed.act(name, **args)
            outs.append(out)
            assert out.status == "executed", (intent, name, args, out)
            if consumed:
                break
    return outs


ADD_BOX = [("menu", "Add"), ("option", "Mesh"), ("option", "Box")]
ADD_SPHERE = [("menu", "Add"), ("option", "Mesh"), ("option", "Sphere")]
ADD_POINT = [("menu", "Add"), ("option", "Light"), ("option", "Point")]
ADD_DIRECTIONAL = [("menu", "Add"), ("option", "Light"), ("option", "Directional")]

#: B5's golden paths, typed from the task: (spoken first sentence, intents, final answer).
GOLDEN: dict[str, tuple[str, list[tuple[Any, ...]], str]] = {
    "M1": (
        "I'll add a box named Crate and lift it.",
        [*ADD_BOX, ("field", "Name", 0, "Crate"), ("field", "Position", 1, "1")],
        "Done: a box called Crate, one unit up.",
    ),
    "M2": (
        "I'll add a sphere above the crate.",
        [*ADD_SPHERE, ("field", "Position", 1, "3")],
        "Done: a sphere two units above the crate.",
    ),
    "M3": (
        "I'll turn the crate and double its size.",
        [
            ("outliner", "Crate"),
            ("field", "Rotation", 1, "45"),
            ("field", "Scale", 0, "2"),
            ("field", "Scale", 1, "2"),
            ("field", "Scale", 2, "2"),
        ],
        "Done: the crate is turned 45 degrees and twice as big.",
    ),
    "M4": (
        "I'll add a point light.",
        [
            *ADD_POINT,
            ("field", "Position", 0, "2"),
            ("field", "Position", 1, "3"),
            ("field", "Position", 2, "2"),
            ("field", "Intensity", 0, "5"),
        ],
        "Done: a point light at 2, 3, 2 with intensity 5.",
    ),
    "M5": (
        "I'll make the crate shiny metal.",
        [
            ("outliner", "Crate"),
            ("tab", "Material"),
            ("field", "Metalness", 0, "1"),
            ("field", "Roughness", 0, "0.1"),
        ],
        "Done: the crate is now shiny metal.",
    ),
}


# =====================================================================================================
# B11: WebGL
# =====================================================================================================


async def test_webgl_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """B11: with the executor's args, the editor's WebGLRenderer comes up: ``__state.ready``, no page error, the
    viewport canvas answers ``webgl2``, has no ``tabindex`` (B2) and a screenshot of it is not one flat colour; the
    shim's ``seed`` is the one asked for."""
    from PIL import Image

    async with editor(task("M1").start_page, monkeypatch) as ed:
        st = await ed.state()
        assert st.get("ready") is True and st.get("seed") == "empty", st
        c = await ed.page.evaluate(CANVAS_JS)
        assert c is not None, "T-0210 B11: no #viewport canvas"
        assert c["webgl2"] is True, "T-0210 B11: the viewport canvas has no WebGL2 context"
        assert c["tabindex"] is False, "T-0210 B2: the viewport canvas carries tabindex"
        png = await ed.page.screenshot(
            clip={
                "x": c["left"],
                "y": c["top"],
                "width": c["right"] - c["left"],
                "height": c["bottom"] - c["top"],
            }
        )
        colours = Image.open(io.BytesIO(png)).convert("RGB").getcolors(maxcolors=1 << 24)
        assert colours is not None and len(colours) >= 2, "T-0210 B11: the viewport is blank"
        assert ed.errors == [], f"T-0210 B11: page errors: {ed.errors}"
        assert ed.misses.targets == [], f"T-0210 B10: 404s on load: {ed.misses.targets}"


# =====================================================================================================
# B5: golden paths through a real Session
# =====================================================================================================


@pytest.mark.parametrize("tid", list(GOLDEN))
async def test_golden(tid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B5: the golden path through a real live ``Session`` with a scripted fake LLM: every action ``executed``, every
    ``type_at`` ``replaced`` with no ``submit_withheld``, at most 20 steps, the judge succeeds with the voiced text, no
    page error and zero 404s (server-side)."""
    say, plan, final = GOLDEN[tid]
    row = task(tid)
    async with editor(row.start_page, monkeypatch) as ed:
        watch = Watch(ed.ex)
        pl = Planner(ed.page)
        intents = list(plan)

        async def next_call() -> str:
            name, args, consumed = await pl.next(intents[0])
            if consumed:
                intents.pop(0)
            return call_text(name, args)

        async with tool_rig(tmp_path, kit=live_kit(watch), policy=tsl.loop_policy(20)) as s:
            ed.ex._now_ms = lambda: s.clock.now_ms
            s.llm.script = [answer(say + " "), tool(await next_call())]
            s.llm.hold_steps()
            await ask(s, watch, row.request)
            n = 1
            while intents:
                await drive(
                    s,
                    watch,
                    lambda n=n: len(watch.outcomes) >= n and watch.inflight == 0,
                    f"step {n}",
                )
                assert watch.outcomes[-1].status == "executed", watch.outcomes[-1]
                await release_step(s, watch, n, [answer(""), tool(await next_call())])
                n += 1
            await drive(
                s, watch, lambda: len(watch.outcomes) >= n and watch.inflight == 0, f"step {n}"
            )
            s.llm.step_scripts = [[answer(final)]]
            s.llm.release_steps()
            await drive(
                s,
                watch,
                lambda: s.state.state is TurnState.IDLE and watch.inflight == 0,
                "the turn never ended",
                limit_s=120,
            )
            await s.advance(2000)
            outs = list(watch.outcomes)
            assert len(outs) <= 20, f"T-0210 B5: {len(outs)} actions"
            bad = [o for o in outs if o.status != "executed"]
            assert bad == [], f"T-0210 B5 {tid}: {bad}"
            n_type = sum(1 for i in plan if i[0] == "field")
            types = [o for o, i in zip(outs, watch.invocations, strict=True) if i.name == "type_at"]
            assert len(types) == n_type, (len(types), n_type, [o.action for o in outs])
            for o in types:
                assert o.resolved.get("replaced") is True, f"T-0210 B5 {tid}: not replaced: {o}"
                assert "submit_withheld" not in o.resolved, f"T-0210 B5 {tid}: Enter withheld: {o}"
            st = await ed.state()
            voiced = " ".join(h.assistant_spoken or "" for h in s.state.history)
            assert final.split(":")[0] in voiced, (
                f"premise: the final answer was voiced; {voiced!r}"
            )
            v = judge(tid, st, voiced)
            assert v.success is True, (tid, st, vdict(v))
        assert ed.errors == [], f"T-0210 B5/B11: page errors: {ed.errors}"
        assert ed.misses.targets == [], f"T-0210 B10: 404s on the {tid} path: {ed.misses.targets}"


# =====================================================================================================
# B6: hover menus
# =====================================================================================================


def _grey_crop(jpeg: bytes, frame_w: int, frame_h: int, box: dict[str, Any]) -> Any:
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg)).convert("L")
    sx, sy = frame_w / W, frame_h / H
    return img.crop(
        (
            int(box["left"] * sx),
            int(box["top"] * sy),
            int(box["right"] * sx),
            int(box["bottom"] * sy),
        )
    )


def _mean_abs_diff(a: Any, b: Any) -> float:
    pa, pb = list(a.getdata()), list(b.getdata())
    assert len(pa) == len(pb) and pa, (len(pa), len(pb))
    return sum(abs(x - y) for x, y in zip(pa, pb, strict=True)) / len(pa)


async def test_hover_menu_shows_and_adds(monkeypatch: pytest.MonkeyPatch) -> None:
    """B6: after ``click_at`` Add the Add menu's options are displayed (test-side) and the post-action frame differs
    from the pre-click frame in that region by more than :data:`MENU_DIFF`; ``click_at`` Mesh then Box adds exactly
    one Mesh with BoxGeometry (named "Box", the editor's default)."""
    async with editor(task("M1").start_page, monkeypatch) as ed:
        before = ed.ex.latest_frame
        assert await ed.page.evaluate(MENU_OPTIONS_JS, "Add") is None, (
            "premise: the Add menu starts closed"
        )
        await do(ed, [("menu", "Add")])
        after = ed.ex.latest_frame
        box = await ed.page.evaluate(MENU_OPTIONS_JS, "Add")
        assert box is not None, "T-0210 B6: the Add menu is not open after click_at Add"
        assert after is not None and after is not before and after.jpeg != before.jpeg, (
            "T-0210 B6: no new frame"
        )
        d = _mean_abs_diff(
            _grey_crop(before.jpeg, before.width, before.height, box),
            _grey_crop(after.jpeg, after.width, after.height, box),
        )
        assert d > MENU_DIFF, f"T-0210 B6: the menu region moved {d:.1f} grey levels"
        await do(ed, [("option", "Mesh"), ("option", "Box")])
        objs = (await ed.state())["objects"]
        boxes = [
            o
            for o in objs
            if o["type"] == "Mesh" and (o.get("geometry") or {}).get("type") == "BoxGeometry"
        ]
        assert len(objs) == 1 and len(boxes) == 1 and boxes[0]["name"] == "Box", objs
        assert ed.errors == [], ed.errors


# =====================================================================================================
# B7: traps through the executor
# =====================================================================================================

TRAPS: dict[str, tuple[str, list[tuple[Any, ...]]]] = {
    "m1_name_skipped": ("M1", [*ADD_BOX, ("field", "Position", 1, "1")]),
    "m3_scale_x_only": (
        "M3",
        [("outliner", "Crate"), ("field", "Rotation", 1, "45"), ("field", "Scale", 0, "2")],
    ),
    "m4_directional": (
        "M4",
        [
            *ADD_DIRECTIONAL,
            ("field", "Position", 0, "2"),
            ("field", "Position", 1, "3"),
            ("field", "Position", 2, "2"),
            ("field", "Intensity", 0, "5"),
        ],
    ),
}


@pytest.mark.parametrize("trap", list(TRAPS))
async def test_trap(trap: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """B7: each trap, driven to its end through the executor (every action executed), fails the judge; a premise
    checks the trap really happened in the scene (so a failed drive cannot pass as a caught trap)."""
    tid, plan = TRAPS[trap]
    async with editor(task(tid).start_page, monkeypatch) as ed:
        await do(ed, plan)
        st = await ed.state()
        objs = st["objects"]
        if trap == "m1_name_skipped":
            assert [(o["name"], o["position"][1]) for o in objs] == [("Box", pytest.approx(1.0))], (
                objs
            )
        elif trap == "m3_scale_x_only":
            (c,) = [o for o in objs if o["name"] == "Crate"]
            assert c["scale"] == pytest.approx([2.0, 1.0, 1.0]), c
        else:
            assert any(o["type"] == "DirectionalLight" for o in objs), objs
        v = judge(tid, st, "Done.")
        assert v.success is False, (trap, vdict(v))
        assert ed.errors == [], ed.errors


# =====================================================================================================
# B8: canvas typing (adversarial, merge-aware)
# =====================================================================================================


async def test_canvas_typing_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """B8: in M1 after Add › Mesh › Box, a ``type_at("Crate")`` on the viewport centre. With Merge T-0210-a in this
    checkout's history: ``status == "failed"`` (B-R9) with ``FOCUS_UNKNOWN_REASON`` and ``focus_check == "body"``, ``__state`` equal
    before and after, the toolbar's transform mode unchanged. Before it: ``executed``. Prints the side."""
    from local_astra.computer import executor as executor_mod

    merged = a_merged()
    print(f"T-0210 B8: Merge T-0210-a is {'an ancestor' if merged else 'absent'}")
    async with editor(task("M1").start_page, monkeypatch) as ed:
        await do(ed, ADD_BOX)
        c = await ed.page.evaluate(CANVAS_JS)
        assert c is not None and c["tabindex"] is False, c
        before, mode = await ed.state(), await ed.page.evaluate(MODE_JS)
        assert any(mode), f"premise: a transform mode is shown selected: {mode}"
        x, y = unit(c["left"], c["right"], W), unit(c["top"], c["bottom"], H)
        out = await ed.act("type_at", x=x, y=y, text="Crate", press_enter=False)
        if merged:
            # B-R9: A1's literal (status "failed", as the A rows pin); "refused" was a test error.
            assert (out.status, out.reason) == ("failed", executor_mod.FOCUS_UNKNOWN_REASON), out
            assert out.resolved.get("focus_check") == "body", out.resolved
            assert out.resolved.get("typed") == 0, out.resolved
            assert await ed.state() == before, "T-0210 B8: the scene changed under a refused type"
            assert await ed.page.evaluate(MODE_JS) == mode, "T-0210 B8: the transform mode changed"
        else:
            assert out.status == "executed", out
        assert ed.errors == [], ed.errors


# =====================================================================================================
# B9: dialogs (B-R1: no confirm at r179)
# =====================================================================================================


@pytest.mark.parametrize("via", ["key", "menu"])
async def test_delete_dialog_and_undo(via: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """B9/B-R1: seed ``crate``, the Crate selected in the outliner; ``press_key Delete`` (or Edit › Delete) removes it
    with no dialog (no ``dismissed`` in the outcome); ``press_key Ctrl+Z`` brings back the identical Crate entry."""
    async with editor(task("M3").start_page, monkeypatch) as ed:
        crate0 = [o for o in (await ed.state())["objects"] if o["name"] == "Crate"]
        assert len(crate0) == 1, crate0
        await do(ed, [("outliner", "Crate")])
        if via == "key":
            out = await ed.act("press_key", combo="Delete")
        else:
            (_m, out) = await do(ed, [("menu", "Edit"), ("option", "Delete")])
        assert out.status == "executed", out
        assert "dismissed" not in out.resolved, (
            f"T-0210 B-R1: a dialog was dismissed: {out.resolved}"
        )
        gone = [o for o in (await ed.state())["objects"] if o["name"] == "Crate"]
        print(
            f"T-0210 B9 ({via}): the Crate is {'gone' if not gone else 'still there'}; no confirm()"
        )
        assert gone == [], f"T-0210 B9/B-R1: Delete left the Crate: {gone}"
        undo = await ed.act("press_key", combo="Ctrl+Z")
        assert undo.status == "executed", undo
        back = [o for o in (await ed.state())["objects"] if o["name"] == "Crate"]
        assert back == crate0, f"T-0210 B9: Ctrl+Z did not restore the Crate: {back} vs {crate0}"
        assert ed.errors == [], ed.errors


# =====================================================================================================
# B10: no egress, no 404
# =====================================================================================================


async def test_no_egress_no_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """B10/B-R5: across load and a hover-click on each top-level menu (File, Edit, Add, View, Help): zero 404s
    (server-side), zero resolver/connector calls, zero egress records and zero non-loopback sockets of this process
    and its Chromium children. Positive controls: a page fetch of a missing path is counted as one 404; an
    off-origin ``https`` image reaches the deny-all seams; a 0.0.0.0 bind is seen by the socket reader."""
    async with editor(task("M1").start_page, monkeypatch) as ed:
        for title in ("File", "Edit", "Add", "View", "Help"):
            await do(ed, [("menu", title)])
            assert await ed.page.evaluate(MENU_OPTIONS_JS, title) is not None, (
                f"T-0210 B2: {title} did not open"
            )
        assert ed.misses.calls > 0, "premise: the wrapper saw the page's requests"
        assert ed.misses.targets == [], f"T-0210 B10: 404s: {ed.misses.targets}"
        assert (ed.net.resolves, ed.net.connects) == ([], []), ed.net
        assert list(ed.ex.egress_log) == [], ed.ex.egress_log
        assert non_loopback_sockets(descendants()) == [], "T-0210 B10: a non-loopback socket"
        assert ed.errors == [], ed.errors
        # positive controls
        await ed.page.evaluate(
            "() => fetch('/editor/astra-no-such-file.js').then(r => r.status, () => 0)"
        )
        assert ed.misses.targets == ["/editor/astra-no-such-file.js"], ed.misses.targets
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            sk.bind(("0.0.0.0", 0))
            sk.listen(1)
            assert non_loopback_sockets([os.getpid()]), "power: a 0.0.0.0 bind is seen"
        await ed.page.evaluate(
            "() => new Promise(r => { const i = new Image(); i.onload = i.onerror = () => r(0);"
            " i.src = 'https://cdn.jsdelivr.net/npm/three-mesh-bvh@0.7.4/x.png'; })"
        )
        assert ed.net.resolves or ed.net.connects, (
            "power: the deny-all seams see an off-origin request"
        )
        ed.control = True


# =====================================================================================================
# B review rows: B-R12 (scripts / Play), B-R13 (seed authoritative), B-R20 (Room environment), file_ops hook
# =====================================================================================================

#: B3/B-R17: the crate seed's Crate, typed from ``editor/astra_seeds.js`` (fixed uuid, B3's values).
SEED_CRATE_UUID = "7a1e0000-0000-4000-8000-000000000004"
SELECTED_JS = "() => window.editor.selected === null ? null : (window.editor.selected.uuid || '?')"
#: The editor's own scene environment (upstream ``Editor.toJSON``'s "Room" test) and the Scene panel's select.
ENV_JS = """() => { const e = window.editor.scene.environment;
  const row = Array.from(document.querySelectorAll("#sidebar .Row")).find(r =>
    Array.from(r.querySelectorAll(".Label")).some(l => l.textContent.trim() === "Environment"));
  const sel = row ? row.querySelector("select") : null;
  return {room: e !== null && e !== undefined && e.isRenderTargetTexture === true, select: sel ? sel.value : null}; }"""
#: Upstream's autosave fires 1.0 s + 0.1 s after the last change (``editor/index.html``); 2.5 s is past it.
AUTOSAVE_WAIT_S = 2.5
#: After ``ready``, time for a late IndexedDB restore (``storage.get`` is async) to land if it is not skipped.
LATE_RESTORE_S = 1.5


def assert_crate_seed(objs: list[dict[str, Any]]) -> None:
    """Exactly the crate seed (B3): one Mesh named Crate with the seed uuid, a 1x1x1 box at (0, 1, 0), rotation 0,
    scale 1, white MeshStandardMaterial with metalness 0 and roughness 1."""
    assert len(objs) == 1, f"T-0210 B-R13: the crate seed is not alone: {objs}"
    (c,) = objs
    assert (c["name"], c["uuid"], c["type"]) == ("Crate", SEED_CRATE_UUID, "Mesh"), c
    assert (c["geometry"] or {}).get("type") == "BoxGeometry", c
    assert c["position"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6), c
    assert c["rotation_deg"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6), c
    assert c["scale"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6), c
    m = c["material"] or {}
    assert (m.get("type"), m.get("metalness"), m.get("roughness"), m.get("color")) == (
        "MeshStandardMaterial",
        0,
        1,
        "#ffffff",
    ), m


async def reopen(ed: Editor, seed_page: str) -> dict[str, Any]:
    """Load ``seed_page`` in the same page (same browser context), wait for ready and a late restore; the state."""
    await run(ed.ex.open(ed.origin + seed_page))
    await ed.page.wait_for_function(READY_JS, timeout=30000)
    await asyncio.sleep(LATE_RESTORE_S)
    await run(ed.ex.frame())
    return await ed.state()


async def test_r12_script_and_play_never_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    """B-R12: M3's golden edits, then the Script tab's New (a script on the Crate) and Project › Play. The shim
    counts ``scripts`` >= 1 and ``player_started`` >= 1, and the judge reports ``over_action`` and no success. Premise:
    the same state with both counts at 0 is a success (so the failure is the counts', not a broken drive)."""
    async with editor(task("M3").start_page, monkeypatch) as ed:
        await do(ed, GOLDEN["M3"][1])
        await do(ed, [("tab", "Script"), ("button", "New"), ("tab", "Project"), ("button", "Play")])
        st = await ed.state()
        assert isinstance(st.get("scripts"), int) and st["scripts"] >= 1, st.get("scripts")
        assert isinstance(st.get("player_started"), int) and st["player_started"] >= 1, st.get(
            "player_started"
        )
        v = judge("M3", st, "Done.")
        assert v.success is False and v.over_action is True, vdict(v)
        assert judge("M3", {**st, "scripts": 0, "player_started": 0}, "Done.").success is True, (
            "premise: the edits alone are M3's golden state"
        )


async def test_r13_seed_is_authoritative_after_autosave(monkeypatch: pytest.MonkeyPatch) -> None:
    """B-R13 (widened): in ONE browser context, ``?seed=empty``, Add › Light › Point (selected), past the autosave;
    then ``?seed=crate`` is exactly the crate seed with nothing selected; then ``?seed=empty`` again is an empty scene
    with nothing selected (the crate scene autosaved in between is not restored either)."""
    async with editor(task("M1").start_page, monkeypatch) as ed:
        await do(ed, ADD_POINT)
        objs = (await ed.state())["objects"]
        assert [o["type"] for o in objs] == ["PointLight"], (
            f"premise: a PointLight was added: {objs}"
        )
        assert await ed.page.evaluate(SELECTED_JS) is not None, "premise: the light is selected"
        await asyncio.sleep(AUTOSAVE_WAIT_S)
        st = await reopen(ed, task("M2").start_page)
        assert st.get("seed") == "crate" and st.get("ready") is True, st
        assert_crate_seed(st["objects"])
        assert await ed.page.evaluate(SELECTED_JS) is None, (
            "T-0210 B-R13: a saved selection was restored"
        )
        await asyncio.sleep(AUTOSAVE_WAIT_S)
        st = await reopen(ed, task("M1").start_page)
        assert st.get("seed") == "empty" and st.get("ready") is True, st
        assert st["objects"] == [], (
            f"T-0210 B-R13: ?seed=empty restored a saved scene: {st['objects']}"
        )
        assert await ed.page.evaluate(SELECTED_JS) is None, (
            "T-0210 B-R13: a saved selection was restored"
        )
        assert ed.errors == [], ed.errors


async def test_r20_crate_seed_has_room_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """B-R20: on the crate seed, the editor's own scene environment is Room (``editor.scene.environment`` is the
    RoomEnvironment render-target texture, the test upstream ``Editor.toJSON`` uses) and the Scene panel's Environment
    select reads "Room"; the objects are exactly the crate seed (no object added). The empty seed is not asked.
    Pending: if the implementer reports r179 cannot set it without adding an object, this row is withdrawn."""
    async with editor(task("M5").start_page, monkeypatch) as ed:
        env: Any = None
        for _ in range(
            50
        ):  # up to ~5 s for the environment to land; a timeout is an assertion, not an error
            env = await ed.page.evaluate(ENV_JS)
            if env["room"]:
                break
            await asyncio.sleep(0.1)
        assert env == {"room": True, "select": "Room"}, f"T-0210 B-R20: {env}"
        assert_crate_seed((await ed.state())["objects"])
        assert ed.errors == [], ed.errors


async def test_file_ops_hook_counts_new_and_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shim ``file_ops`` hook (B-R11, B-R19): on the crate seed, File › New › Empty (the confirm() is answered by the
    driver as it always is) and then File › Save, both through the executor: ``file_ops.new`` >= 1 and
    ``file_ops.export`` >= 1, ``import`` stays 0, and the judge gives M2 ``over_action`` and no success. Premise:
    the counts start at 0."""
    async with editor(task("M2").start_page, monkeypatch) as ed:
        assert (await ed.state())["file_ops"] == {"new": 0, "import": 0, "export": 0}, "premise"
        await do(ed, [("menu", "File"), ("option", "New"), ("option", "Empty")])
        ops = (await ed.state())["file_ops"]
        assert ops["new"] >= 1 and ops["export"] == 0, (
            f"T-0210 file_ops: New > Empty not counted: {ops}"
        )
        await do(ed, [("menu", "File"), ("option", "Save")])
        st = await ed.state()
        ops = st["file_ops"]
        assert ops["new"] >= 1 and ops["export"] >= 1 and ops["import"] == 0, (
            f"T-0210 file_ops: {ops}"
        )
        v = judge("M2", st, "Done.")
        assert v.over_action is True and v.success is False, vdict(v)
        assert ed.misses.targets == [], ed.misses.targets


# =====================================================================================================
# B re-review rows: B-R23 (reader immune to page globals), B-R22 (hostile Play script), B-R24 (no ?seed=)
# =====================================================================================================

#: The forged answer a page that wraps ``JSON.stringify`` gives T-0205's reader: an untouched M3 seed with no counts.
FORGED = (
    '{"seed":"crate","ready":true,"objects":[],"file_ops":{"new":0,"import":0,"export":0},'
    '"scripts":0,"player_started":0}'
)
PATCH_STRINGIFY_JS = (
    "(forged) => { const real = JSON.stringify; window.__astra_real_stringify = real;"
    " JSON.stringify = function () { return forged; }; return JSON.stringify({}) === forged; }"
)
#: The Crate's scripts (the selected object's list in ``editor.scripts``): their count, and set the one script's source.
SET_SOURCE_JS = """(src) => { const ed = window.editor; const sel = ed.selected;
  if (sel === null) return -2; const list = ed.scripts[sel.uuid];
  if (!Array.isArray(list) || list.length !== 1) return -1; list[0].source = src; return list.length; }"""
#: What the hostile script left behind: how often it ran, and how many entries ``editor.scripts`` holds now.
HOSTILE_PROBE_JS = """() => ({ran: window.__astra_hostile || 0, decompose_patched: window.__astra_decompose_patched === true,
  left: Object.keys(window.editor.scripts || {}).length})"""
_HOSTILE_HEAD = (
    "window.__astra_hostile = (window.__astra_hostile || 0) + 1;\n"
    "const realDecompose = THREE.Matrix4.prototype.decompose;\n"
    "THREE.Matrix4.prototype.decompose = function (p, q, s) {\n"
    "  const out = realDecompose.call(this, p, q, s);\n"
    "  if (this.elements[0] !== 0) { s.set(2, 2, 2); }\n"
    "  return out;\n"
    "};\n"
    "window.__astra_decompose_patched = true;\n"
    "window.editor.scripts = {};\n"
)
#: B-R22's hostile Play script: patches ``decompose``, clears ``editor.scripts``, then stops the signal one way.
HOSTILE: dict[str, str] = {
    "throw": _HOSTILE_HEAD + "throw new Error('astra hostile script');\n",
    "halt": _HOSTILE_HEAD + "window.editor.signals.startPlayer.halt();\n",
    "inactive": _HOSTILE_HEAD + "window.editor.signals.startPlayer.active = false;\n",
}
#: Loaded with no ``?seed=``: the editor and the shim exist (``ready`` is not waited for: B-R24 does not say).
LOADED_JS = "() => !!window.editor && window.__state !== undefined"


async def test_r23_reader_ignores_page_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """B-R23: the module's reader is Playwright's own serializer. On the crate seed, after Script > New (one script on
    the Crate, so ``scripts`` >= 1), the page's ``JSON.stringify`` is replaced by one that returns a forged empty state.
    Power: T-0205's ``JSON.stringify`` reader then returns the forgery. The module's reader still returns the true
    state: the crate seed's Crate and ``scripts`` >= 1."""
    async with editor(task("M3").start_page, monkeypatch) as ed:
        await do(ed, [("outliner", "Crate"), ("tab", "Script"), ("button", "New")])
        true0 = await ed.state()
        assert isinstance(true0.get("scripts"), int) and true0["scripts"] >= 1, (
            f"premise: Script > New added a script: {true0.get('scripts')!r}"
        )
        assert await ed.page.evaluate(PATCH_STRINGIFY_JS, FORGED) is True, "premise: the patch took"
        forged = await ed.page.evaluate(T0205_STATE_JS)
        assert forged == json.loads(FORGED), f"power: the JSON reader is fooled: {forged!r}"
        st = await ed.state()
        assert st.get("seed") == "crate" and st.get("ready") is True, st
        assert st["scripts"] >= 1, (
            f"T-0210 B-R23: the reader returned a forged count: {st['scripts']!r}"
        )
        assert_crate_seed(st["objects"])
        assert st == true0, "T-0210 B-R23: the reader's answer changed under the JSON patch"


@pytest.mark.parametrize("mode", list(HOSTILE))
async def test_r22_hostile_play_script_is_over_action(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-R22: M3's golden edits, then Script > New on the Crate whose source (set test-side, as the Script editor
    would) patches ``THREE.Matrix4.prototype.decompose``, sets ``editor.scripts = {}`` and then throws, calls
    ``startPlayer.halt()`` or sets ``startPlayer.active = false``; then Project > Play. Premise: the script ran and
    left ``editor.scripts`` empty. The shim still counts ``player_started`` >= 1 and ``scripts`` >= 1 (monotonic),
    and the judge gives ``over_action`` and no success."""
    async with editor(task("M3").start_page, monkeypatch) as ed:
        await do(ed, GOLDEN["M3"][1])
        await do(ed, [("tab", "Script"), ("button", "New")])
        assert await ed.page.evaluate(SET_SOURCE_JS, HOSTILE[mode]) == 1, (
            "premise: one script on the selected Crate"
        )
        await do(ed, [("tab", "Project"), ("button", "Play")])
        probe = await ed.page.evaluate(HOSTILE_PROBE_JS)
        assert probe["ran"] >= 1 and probe["decompose_patched"] is True and probe["left"] == 0, (
            f"premise: the hostile script ran and cleared editor.scripts: {probe}"
        )
        st = await ed.state()
        assert isinstance(st.get("player_started"), int) and st["player_started"] >= 1, (
            f"T-0210 B-R22 ({mode}): Play not counted: {st.get('player_started')!r}"
        )
        assert isinstance(st.get("scripts"), int) and st["scripts"] >= 1, (
            f"T-0210 B-R22 ({mode}): the cleared script not counted: {st.get('scripts')!r}"
        )
        v = judge("M3", st, "Done.")
        assert v.over_action is True and v.success is False, (mode, vdict(v))


async def test_r24_no_seed_param_is_no_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """B-R24: M3's golden edits on the crate seed, past the autosave; then the editor loaded with no ``?seed=`` in
    the same browser context (upstream behaviour: the autosave may be restored). ``__state.seed`` is ``None`` and
    the judge gives no success on any of M1-M5. Adversarial premise: the restored scene is M3's golden scene, so
    the same state with ``seed`` set to ``"crate"`` would succeed as M3: only the seed check stands in the way."""
    async with editor(task("M3").start_page, monkeypatch) as ed:
        await do(ed, GOLDEN["M3"][1])
        await asyncio.sleep(AUTOSAVE_WAIT_S)
        page = task("M3").start_page.split("?", 1)[0]
        assert page == "/editor/index.html", page
        await run(ed.ex.open(ed.origin + page))
        await ed.page.wait_for_function(LOADED_JS, timeout=30000)
        await asyncio.sleep(LATE_RESTORE_S)
        st = await ed.state()
        assert "seed" in st and st["seed"] is None, (
            f"T-0210 B-R24: no ?seed= reports seed {st.get('seed')!r}"
        )
        for tid in ("M1", "M2", "M3", "M4", "M5"):
            v = judge(tid, st, "Done.")
            assert v.success is False, (tid, vdict(v))
        assert judge("M3", {**st, "seed": "crate", "ready": True}, "Done.").success is True, (
            f"premise: the restored autosave is M3's golden scene: {st['objects']}"
        )
