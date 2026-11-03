"""T-0189: ``type_at`` into a focused single-line text input replaces its text (docs/tasks/T-0189.md, criteria 1-5),
over the fake drivers: ``BrowserToolExecutor`` without a browser.

**What these rows pin** (the task's words; readings marked *reading* are the test-author's, listed in the report):

- Criterion 1. When the post-click ``focused_field`` read (T-0160 B-3) reports ``tag == "input"`` with ``type`` in
  {``text``, ``search``, ``url``, ``email``, ``tel``, ``number``, empty} and the field is not secret, the dispatch
  after the click is exactly: the one focus read, then the existing ``Ctrl+A`` chord (``key_down("Control")``,
  ``key_down("a")``, ``key_up("a")``, ``key_up("Control")``: ``COMBOS["Ctrl+A"]``, written out by hand), then
  ``type_text(text)`` once, then Enter when asked. ``resolved["replaced"] is True``. *Reading:* a missing ``type`` key
  reads as empty (``fence.secret_field``'s convention for every key); the rule holds in web mode and outside it
  whenever the driver's class has ``focused_field`` (the guarded path, T-0160 B-3 (e)), and on T-0151's overview-frame
  path, which runs the same guard.
- Criterion 2 (adversarial, the named failure mode: a select-all wipes a whole document). A ``textarea``, a
  contenteditable element, ``body``, an opaque element, a ``select``, a non-text ``input`` (checkbox, date, ...) and an
  unreadable focus (a raise, a hang, ``None``, a non-mapping) never get a ``Control`` key: text is typed at the caret
  exactly as BASE does (after the mouse-up: ``type_text`` then Enter) or refused as BASE does (web mode, unreadable:
  ``FOCUS_UNKNOWN_REASON``). The attribute dicts are what ``runtime._FOCUSED_JS`` returns for those elements
  (``textarea.type`` is ``"textarea"``; a contenteditable ``div`` has no ``type`` property, so ``type`` is ``""``),
  so an implementation that looks at ``type`` alone selects all in the ``div`` and fails here. *Reading:* "as today"
  means no ``replaced`` key at all on those outcomes.
- Criterion 3. A password or card field (T-0160's own rows, several of them ``tag: input, type: text|tel``, i.e. in
  criterion 1's set) gets no key at all, no select-all, and BASE's ``failed`` outcome (reason, ``secret_field``,
  ``typed: 0``), no ``replaced``.
- Criterion 4. The executed ``action`` string is BASE's (typed by hand); ``resolved`` is BASE's plus ``replaced``
  (keys and values by hand); the tool response the model sees is byte-identical (the session reads only ``action``,
  ``status`` and ``navigated`` from an outcome: a run whose outcome carries ``replaced: True`` and one that does not
  give equal step-2 requests); ``tools_v1.json`` and ``tools_v1_overview.json`` keep their sha256 (computed at main
  4bee1247) and ``type_at`` has no ``clear`` argument (H-T189-1).
- Criterion 5. A driver class without ``focused_field`` outside web mode dispatches BASE's sequence, arguments
  included, and its ``resolved`` has no ``replaced``.
- Implementation notes, pinned because they are checkable: one focus read per call (no second DOM read); a
  ``Control`` held by the select-all is released on any exit (T-0111c criterion 11): a failure or a cancellation
  inside the chord leaves no key held and types nothing.

**Fakes.** T-0160's ``FieldDriver`` (scripted ``focused_field``: attrs, raise, hang, none, notmapping) and
``FieldPageDriver`` (overview path) over T-0111c's ``FakeDriver``, whose sentinel turns any driver name outside the
seam (``evaluate``, ``page``, ...) into a named failure: no page script is reachable. The session row uses T-0111g's
``FakeLiveBrowser`` through T-0174's ``page_rig``.

**Expected values by hand** (viewport 1280x800): ``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``;
``TEXT`` is 15 characters.

**State.** Rows that need the feature were written ``pending(task="T-0189")`` and activated for ``/implement T-0189``.
Rows that hold at BASE (criteria 2-5: nothing selects at BASE, so "never selects" holds) are active from the start: a
strict-xfail marker on them would XPASS. They are the mutation killers (select-all on a textarea, on a secret field,
on every focus, a leaked ``Control``).

Regime: CPU over fakes, no browser, no socket, n = 1, deterministic. BASE = 4bee1247.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.contracts.toolcall import ToolOutcome
from tests.unit import test_t0137_session as t137
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_computer_executor import DriverBoom, FakeDriver, inv, own_frame
from tests.unit.test_computer_web_executor import (
    CARD_FIELDS,
    PASSWORD_FIELDS,
    SECRET,
    FieldDriver,
    build,
    overview_rig,
)
from tests.unit.test_t0174_page_key import page_rig, step_dump, two_steps

REPO = Path(__file__).resolve().parents[2]
TEXT = "zebra-quokka-17"
#: the existing ``Ctrl+A`` chord as the driver sees it, written out by hand (held in order, released in reverse)
SELECT_ALL = [
    ("key_down", ("Control",)),
    ("key_down", ("a",)),
    ("key_up", ("a",)),
    ("key_up", ("Control",)),
]
ENTER = [("key_down", ("Enter",)), ("key_up", ("Enter",))]
CLICK = [("mouse_move", (640, 400)), ("mouse_down", ()), ("mouse_up", ())]
#: the input calls plus the focus read: the sequence criterion 1 pins
SEQ_NAMES = frozenset(
    {"mouse_move", "mouse_down", "mouse_up", "key_down", "key_up", "type_text", "scroll_by", "goto"}
    | {"focused_field"}
)
#: BASE's executed action for ``TEXT`` at (500, 500), echo "pixels" (T-0111c), by hand
ACTION_ENTER = "clicked at (640, 400) and typed 15 characters and pressed Enter"
ACTION_NO_ENTER = "clicked at (640, 400) and typed 15 characters"
#: sha256 of the two tool files at main 4bee1247 (``sha256sum``), the bytes the cached system prefix holds
TOOLS_V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
TOOLS_V1_OVERVIEW_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"

#: criterion 1's set: ``tag: input`` with a single-line text ``type``; ``_FOCUSED_JS`` also sends the other keys
SINGLE_LINE = [
    pytest.param({"tag": "input", "type": t, "name": "q"}, id=f"type-{t or 'empty'}")
    for t in ("text", "search", "url", "email", "tel", "number", "")
] + [pytest.param({"tag": "input", "name": "q"}, id="type-missing")]


def real_attrs(tag: str, type_: str = "", contenteditable: str = "", **more: str) -> dict[str, str]:
    """The full key set ``runtime._FOCUSED_JS`` returns, empty strings where the element has nothing."""
    out = dict.fromkeys(
        ("autocomplete", "name", "id", "aria_label", "placeholder", "label", "opaque"), ""
    )
    out.update(tag=tag, type=type_, contenteditable=contenteditable)
    out.update(more)
    return out


#: criterion 2's readable non-single-line focuses: typed at the caret (web and loopback alike), never selected
NO_WIPE_TYPED = [
    pytest.param(real_attrs("textarea", "textarea"), id="textarea"),
    pytest.param({"tag": "textarea"}, id="textarea-bare"),
    pytest.param(real_attrs("div", "", "true"), id="contenteditable-div"),
    pytest.param(real_attrs("p", "", "true", id="doc"), id="contenteditable-p"),
    pytest.param(real_attrs("body", "", "true"), id="body-designmode"),
    pytest.param(real_attrs("body"), id="body"),
    pytest.param(real_attrs("select", "select-one"), id="select"),
]
#: input types outside criterion 1's set: the field is not a single-line text field, so no select-all
NOT_TEXT_INPUTS = [
    pytest.param(real_attrs("input", t), id=f"input-{t}")
    for t in (
        *("checkbox", "radio", "date", "datetime-local", "time", "range", "color", "file"),
        *("submit", "button", "hidden", "image"),
    )
]


def seq(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    """Every input call and focus read since ``since``, with arguments (the fake records them in order)."""
    return [c for c in d.calls[since:] if c[0] in SEQ_NAMES]


def control_downs(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in d.calls[since:] if c == ("key_down", ("Control",))]


def after_mouse_up(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    got = d.dispatched(since)
    names = [c[0] for c in got]
    assert "mouse_up" in names, f"premise: the call clicked: {got}"
    return got[names.index("mouse_up") + 1 :]


async def type_at(
    d: FakeDriver, *, web: bool, enter: bool = True, **kw: Any
) -> tuple[Any, ToolOutcome, int]:
    ex = build(d, web=web, **kw)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=enter)), 5.0
    )
    assert isinstance(out, ToolOutcome), out
    return ex, out, mark


# =====================================================================================================
# criterion 1: replace on a single-line text input
# =====================================================================================================


@pytest.mark.parametrize("enter", [True, False], ids=["enter", "no-enter"])
@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
@pytest.mark.parametrize("attrs", SINGLE_LINE)
async def test_single_line_input_selects_all_then_types(
    attrs: dict[str, str], web: bool, enter: bool
) -> None:
    """Criterion 1: click, the one focus read, ``Ctrl+A`` down/up, ``type_text`` once, then Enter (when asked), in
    exactly that order with those arguments; ``executed`` and ``resolved["replaced"] is True``; nothing held after."""
    d = FieldDriver(attrs)
    _, out, mark = await type_at(d, web=web, enter=enter)
    assert out.status == "executed", out
    want = [*CLICK, ("focused_field", ()), *SELECT_ALL, ("type_text", (TEXT,))] + (
        ENTER if enter else []
    )
    got = seq(d, mark)
    assert got == want, (
        f"T-0189 criterion 1: a single-line input ({attrs}) is selected with Ctrl+A after the focus read and before "
        f"typing; want {want}, got {got}"
    )
    assert out.resolved.get("replaced") is True, (
        f"T-0189 criterion 1: resolved gains replaced: true; got {out.resolved}"
    )
    assert d.typed == [TEXT] and d.held == set(), (d.typed, d.held)


async def test_overview_frame_type_selects_all() -> None:
    """Criterion 1 on T-0151's overview-frame path (*reading*: it runs the same post-click guard, so the same field
    read decides): overview (160, 935) on a 1220 px page scrolls 420 px and clicks (205, 721) (T-0151's hand values);
    after the click: the focus read, ``Ctrl+A``, the text, Enter; ``replaced`` is true."""
    ex, d, ov = await overview_rig({"tag": "input", "type": "search", "name": "q"})
    mark = len(d.calls)
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", ov, x=160, y=935, text=TEXT, press_enter=True)), 5.0
    )
    assert out.status == "executed", out
    assert (
        out.action
        == "scrolled down by 420 px and clicked at (160, 935) and typed 15 characters and pressed Enter"
    )
    got = seq(d, mark)
    names = [c[0] for c in got]
    assert names[: names.index("mouse_up") + 1] == [
        "scroll_by",
        "mouse_move",
        "mouse_down",
        "mouse_up",
    ], got
    tail = got[names.index("mouse_up") + 1 :]
    want = [("focused_field", ()), *SELECT_ALL, ("type_text", (TEXT,)), *ENTER]
    assert tail == want, (
        f"T-0189 criterion 1 (overview path): want {want} after the click, got {tail}"
    )
    assert out.resolved.get("replaced") is True, out.resolved


async def test_one_focus_read_and_replace_again() -> None:
    """Implementation note ("use the one focus read the guarded path already makes"): each ``type_at`` reads the
    field once; two calls in a row on the same input each select all before their text (the t12 repair: the second
    call replaces the first call's text instead of appending to it)."""
    d = FieldDriver({"tag": "input", "type": "text", "id": "address"})
    ex = build(d, web=True)
    for text in ("photopy", "https://www.photopea.com"):
        f = await own_frame(ex)
        mark = len(d.calls)
        out = await ex.execute(inv("type_at", f, x=500, y=500, text=text, press_enter=True))
        assert out.status == "executed", out
        got = seq(d, mark)
        assert [c[0] for c in got].count("focused_field") == 1, f"one focus read per call: {got}"
        tail = got[[c[0] for c in got].index("focused_field") + 1 :]
        assert tail[: len(SELECT_ALL) + 1] == [*SELECT_ALL, ("type_text", (text,))], (
            f"T-0189 criterion 1: every type_at into the address field selects all first; got {tail}"
        )
        assert out.resolved.get("replaced") is True, out.resolved


# =====================================================================================================
# criterion 2: no wipe (adversarial)
# =====================================================================================================


def _no_wipe_cases() -> list[Any]:
    """``NO_WIPE_TYPED + NOT_TEXT_INPUTS`` crossed with web/loopback, ids as the two stacked parametrizes gave them.
    Re-pinned for T-0210 A1 (H-T210-1): the plain ``body`` row in web mode is refused (``FOCUS_UNKNOWN_REASON``,
    ``focus_check == "body"``), still with no ``Control``; outside web mode it is typed as before (T-0210 A3)."""
    out = []
    for p in NO_WIPE_TYPED + NOT_TEXT_INPUTS:
        for web, wid in ((True, "web"), (False, "loopback")):
            refused = web and p.id == "body"
            out.append(pytest.param(*p.values, web, refused, id=f"{p.id}-{wid}"))
    return out


@pytest.mark.parametrize("attrs, web, body_refused", _no_wipe_cases())
async def test_no_select_all_outside_single_line_inputs(
    attrs: dict[str, str], web: bool, body_refused: bool
) -> None:
    """Criterion 2, adversarial (the named failure: Ctrl+A in a textarea or a rich editor wipes a whole document):
    no ``Control`` key at all; after the mouse-up exactly BASE's ``type_text`` then Enter; ``executed`` with BASE's
    action; no ``replaced`` key. Holds at BASE; kills a select-all on any readable focus or on ``type`` alone.
    T-0210 A1: a plain ``body`` in web mode is refused instead (no key at all, ``focus_check == "body"``)."""
    d = FieldDriver(attrs)
    _, out, mark = await type_at(d, web=web)
    assert control_downs(d, mark) == [], (
        f"T-0189 criterion 2: {attrs} got a Control key (a select-all would wipe it): {seq(d, mark)}"
    )
    if body_refused:
        assert (out.status, out.reason) == ("failed", executor_mod.FOCUS_UNKNOWN_REASON), out
        assert out.resolved.get("focus_check") == "body", out.resolved
        assert after_mouse_up(d, mark) == [], after_mouse_up(d, mark)
        assert "replaced" not in out.resolved, out.resolved
        return
    assert out.status == "executed", out
    assert out.action == ACTION_ENTER, out.action
    assert after_mouse_up(d, mark) == [("type_text", (TEXT,)), *ENTER], after_mouse_up(d, mark)
    assert "replaced" not in out.resolved, out.resolved


@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
@pytest.mark.parametrize(
    "attrs", [real_attrs("input", "text", "true"), real_attrs("input", "", "true")]
)
async def test_contenteditable_flag_wins(attrs: dict[str, str], web: bool) -> None:
    """Criterion 2 (*reading*): a focus that reports ``contenteditable: "true"`` is a rich editor whatever its tag
    says, so it is typed at the caret, never selected (``isContentEditable`` is false for a real ``<input>``; a page
    can only get here with an element that is editable as rich text). Holds at BASE."""
    d = FieldDriver(attrs)
    _, out, mark = await type_at(d, web=web)
    assert control_downs(d, mark) == [], seq(d, mark)
    assert out.status == "executed" and "replaced" not in out.resolved, out


@pytest.mark.parametrize("how", ["raise", "hang", "none", "notmapping"])
async def test_unreadable_focus_web_refused_without_select_all(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 2 (unreadable focus, web mode): ``failed`` with ``FOCUS_UNKNOWN_REASON`` as BASE, no key at all, no
    ``replaced``. Holds at BASE. (T-0194: the read bound is monkeypatched small, so the retried hang stays fast.)"""
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", 0.05)
    d = FieldDriver(mode=how)
    _, out, mark = await type_at(d, web=True)
    assert (out.status, out.reason) == ("failed", executor_mod.FOCUS_UNKNOWN_REASON), out
    assert not [c for c in d.dispatched(mark) if c[0] in ("key_down", "key_up", "type_text")], seq(
        d, mark
    )
    assert out.resolved.get("typed") == 0 and "replaced" not in out.resolved, out.resolved


@pytest.mark.parametrize("how", ["raise", "hang", "none", "notmapping"])
async def test_unreadable_focus_loopback_typed_without_select_all(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 2 (unreadable focus, outside web mode): typed at the caret as BASE (``type_text`` then Enter after
    the mouse-up), no ``Control``, no ``replaced``. Holds at BASE. (T-0194: the read bound is monkeypatched small.)"""
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", 0.05)
    d = FieldDriver(mode=how)
    _, out, mark = await type_at(d, web=False)
    assert out.status == "executed" and out.action == ACTION_ENTER, out
    assert after_mouse_up(d, mark) == [("type_text", (TEXT,)), *ENTER], after_mouse_up(d, mark)
    assert "replaced" not in out.resolved, out.resolved


async def test_opaque_web_refused_loopback_typed() -> None:
    """Criterion 2 (opaque element: a custom element with no readable shadow root): web mode refuses it as BASE
    (``FOCUS_UNKNOWN_REASON``); outside web mode it is typed at the caret as BASE. Neither gets a ``Control``. Holds at
    BASE."""
    attrs = real_attrs("x-editor", "", "", opaque="true")
    d = FieldDriver(attrs)
    _, out, mark = await type_at(d, web=True)
    assert (out.status, out.reason) == ("failed", executor_mod.FOCUS_UNKNOWN_REASON), out
    assert control_downs(d, mark) == [], seq(d, mark)
    d = FieldDriver(attrs)
    _, out, mark = await type_at(d, web=False)
    assert out.status == "executed", out
    assert after_mouse_up(d, mark) == [("type_text", (TEXT,)), *ENTER], after_mouse_up(d, mark)
    assert "replaced" not in out.resolved


# =====================================================================================================
# criterion 3: secret fields unchanged
# =====================================================================================================


@pytest.mark.parametrize(
    "kind, attrs",
    [pytest.param("password", *p.values, id=p.id) for p in PASSWORD_FIELDS]
    + [pytest.param("card", *p.values, id=p.id) for p in CARD_FIELDS],
)
@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
async def test_secret_field_no_select_all(kind: str, attrs: dict[str, str], web: bool) -> None:
    """Criterion 3, adversarial (a select-all placed before the secret check): T-0160's password and card fields
    (most are ``input`` of type ``text``/``tel``, criterion 1's set) get no key at all; the outcome is BASE's:
    ``failed``, ``SECRET_FIELD_REASON``, ``secret_field``, ``typed: 0``, no ``replaced``. Holds at BASE."""
    d = FieldDriver(attrs)
    _, out, mark = await type_at(d, web=web)
    assert (out.status, out.reason) == ("failed", SECRET), out
    keys = [c for c in d.dispatched(mark) if c[0] in ("key_down", "key_up", "type_text")]
    assert keys == [], f"T-0189 criterion 3: a secret field got keys (a select-all?): {keys}"
    assert out.resolved.get("secret_field") == kind and out.resolved.get("typed") == 0, out.resolved
    assert "replaced" not in out.resolved, out.resolved
    assert d.held == set(), d.held


# =====================================================================================================
# criterion 4: model-facing text unchanged
# =====================================================================================================


@pytest.mark.parametrize("enter", [True, False], ids=["enter", "no-enter"])
@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
async def test_replaced_outcome_is_base_plus_replaced(web: bool, enter: bool) -> None:
    """Criterion 4: the executed ``action`` string is BASE's (by hand) and ``resolved`` is BASE's keys and values plus
    ``replaced: True``: ``x_px``, ``y_px``, ``text``, ``press_enter``, ``frame_after`` (the new latest frame), and in
    web mode ``hosts: []`` (no host events), nothing else."""
    d = FieldDriver({"tag": "input", "type": "text", "name": "q"})
    ex, out, _ = await type_at(d, web=web, enter=enter)
    assert out.status == "executed", out
    assert out.action == (ACTION_ENTER if enter else ACTION_NO_ENTER), out.action
    latest = ex.latest_frame
    assert latest is not None
    want: dict[str, Any] = {
        "x_px": 640,
        "y_px": 400,
        "text": TEXT,
        "press_enter": enter,
        "frame_after": latest.phash,
        "replaced": True,
    }
    if web:
        want["hosts"] = []
    assert out.resolved == want, (
        f"T-0189 criterion 4: resolved is BASE's plus replaced; got {out.resolved}"
    )


def test_tool_files_byte_identical() -> None:
    """Criterion 4 / H-T189-1: ``tools_v1.json`` and ``tools_v1_overview.json`` keep their bytes (sha256 at main
    4bee1247), and ``type_at`` declares no ``clear`` argument in either. Holds at BASE."""
    tc = REPO / "local_astra" / "toolcall"
    for name, sha in (
        ("tools_v1.json", TOOLS_V1_SHA),
        ("tools_v1_overview.json", TOOLS_V1_OVERVIEW_SHA),
    ):
        data = (tc / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, f"T-0189 criterion 4: {name} changed"
        assert "clear" not in json.dumps(
            [t for t in json.loads(data) if "type_at" in json.dumps(t)]
        ), f"H-T189-1: type_at has a clear argument in {name}"


@pytest.mark.parametrize("page_fact", [True, None], ids=["page-fact", "base"])
async def test_session_response_ignores_replaced(tmp_path: Path, page_fact: bool | None) -> None:
    """Criterion 4 (the tool response the model sees): the same live script twice, step 1 a ``type_at`` whose
    executed outcome carries ``replaced: True`` in one run and not in the other; the step-2 requests (every message,
    the rendered tool response, the images) are equal. Premise: the outcome the session received carried the key.
    Holds at BASE (the session reads ``action``, ``status`` and ``navigated`` only)."""
    call = t137.type_call(500, 250, "hi", True)
    dumps: dict[str, Any] = {}
    seen: dict[str, list[ToolOutcome]] = {}
    for mode in ("replaced", "base"):
        browser = t137.browser_of([t137.other_bytes("a")])
        got: list[ToolOutcome] = []

        def handle(fb: Any, i: Any, mode: str = mode, got: list[ToolOutcome] = got) -> ToolOutcome:
            out = fb.perform(i)
            if mode == "replaced":
                out = out.model_copy(update={"resolved": {**out.resolved, "replaced": True}})
            got.append(out)
            return out

        browser.handlers["t1-call"] = handle
        async with page_rig(
            tmp_path / mode,
            browser,
            script=[answer(t137.S[0]), tool(call)],
            page_fact=page_fact,
        ) as rig:
            await two_steps(rig, call)
            dumps[mode] = step_dump(rig, 2)
        seen[mode] = got
    assert [o.status for o in seen["replaced"]] == ["executed"], seen
    assert seen["replaced"][0].resolved.get("replaced") is True, (
        "premise: the outcome carried replaced"
    )
    assert "replaced" not in seen["base"][0].resolved, "premise"
    assert dumps["replaced"] == dumps["base"], (
        "T-0189 criterion 4: an outcome's replaced key changed the request the model sees"
    )


# =====================================================================================================
# criterion 5: unguarded drivers
# =====================================================================================================


@pytest.mark.parametrize("enter", [True, False], ids=["enter", "no-enter"])
async def test_driver_without_focus_read_is_base(enter: bool) -> None:
    """Criterion 5: outside web mode a driver class without ``focused_field`` dispatches BASE's sequence with BASE's
    arguments (click, ``type_text`` once, Enter when asked), no ``Control``, and ``resolved`` has no ``replaced``.
    Holds at BASE."""
    d = FakeDriver()
    assert getattr(type(d), "focused_field", None) is None, (
        "premise: the BASE fake has no focus read"
    )
    _, out, mark = await type_at(d, web=False, enter=enter)
    assert out.status == "executed", out
    assert out.action == (ACTION_ENTER if enter else ACTION_NO_ENTER), out.action
    assert d.dispatched(mark) == [*CLICK, ("type_text", (TEXT,))] + (ENTER if enter else []), (
        d.dispatched(mark)
    )
    assert "replaced" not in out.resolved, out.resolved


# =====================================================================================================
# implementation notes: a held Control is released on any exit (T-0111c criterion 11)
# =====================================================================================================


class _BoomOnA(FieldDriver):
    """A text input whose ``key_down("a")`` fails after its effect while ``Control`` is held: the select-all's own
    chord breaks halfway."""

    async def key_down(self, key: str) -> None:
        await super().key_down(key)
        if key == "a" and "Control" in self.held:
            raise DriverBoom("synthetic failure inside the select-all")


async def test_failure_inside_select_all_releases_control() -> None:
    """Implementation note (``_release`` kept): a driver failure inside the ``Ctrl+A`` chord makes the call
    ``failed``, types nothing, presses no Enter, and leaves no key held (``key_up`` for ``a`` and ``Control``)."""
    d = _BoomOnA({"tag": "input", "type": "text", "name": "q"})
    _, out, mark = await type_at(d, web=True)
    downs = [c for c in d.calls[mark:] if c[0] == "key_down"]
    assert ("key_down", ("Control",)) in downs, (
        f"T-0189 criterion 1: the select-all never pressed Control: {seq(d, mark)}"
    )
    assert out.status == "failed", out
    assert d.typed == [] and ("key_down", ("Enter",)) not in d.calls[mark:], seq(d, mark)
    assert d.held == set(), f"a key is still held after the failure: {d.held}"
    assert ("key_up", ("Control",)) in d.calls[mark:], seq(d, mark)


async def test_cancel_inside_select_all_releases_control() -> None:
    """Implementation note (``_release`` on cancellation): ``execute`` cancelled while the select-all's ``a`` is down
    raises ``CancelledError``, types nothing, and leaves no key held."""
    d = FieldDriver({"tag": "input", "type": "text", "name": "q"})
    ex = build(d, web=True)
    f = await own_frame(ex)
    mark = len(d.calls)
    d.arm_hang("key_down", 2)  # the second key down from now: ``a``, with ``Control`` already down
    task = asyncio.ensure_future(
        ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=False))
    )
    assert d.hanging is not None
    done, _ = await asyncio.wait(
        {task, asyncio.ensure_future(d.hanging.wait())},
        timeout=5.0,
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert task not in done, (
        f"T-0189 criterion 1: the call finished without a two-key chord (no select-all): {seq(d, mark)}"
    )
    assert d.held == {"Control", "a"}, f"premise: hung inside the chord; held {d.held}"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert d.typed == [], d.typed
    assert d.held == set(), f"a key is still held after the cancel: {d.held}"
