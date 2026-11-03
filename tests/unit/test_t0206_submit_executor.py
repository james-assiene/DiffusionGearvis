"""T-0206 A2-A8 at the seam: the submit stop in ``BrowserToolExecutor`` over scripted fake web drivers, no browser
(docs/tasks/T-0206.md, "Tests to write", ``tests/unit/test_t0206_submit_executor.py``).

**The fake.** ``SubmitMixin`` adds A2's two reads to T-0111c's ``FakeDriver`` (and T-0160's ``FieldDriver``, T-0151's
``PageDriver``, T-0192's ``UrlsMixin``, T-0204's ``GuardMixin``): ``submit_facts(x, y)`` answers ``at`` (A1's
``facts``) and ``submit_facts_focused()`` answers ``focused``; ``mode`` / ``fmode`` is ``"ok"``, ``"none"`` (``None``),
``"raise"`` (``DriverBoom`` whose message carries ``TOKEN``) or ``"hang"`` (until cancelled). Both reads are recorded in
``calls`` but are not seam calls (no ``arm_raise`` count, no scene change), so screenshots equal the plain fake's.

**Oracles.** ``SEND`` and ``OPEN`` are hand-built facts whose A1 verdicts are worked out in
``tests/unit/test_t0206_submit_predicate.py`` (``SEND``: "Request quote" with a filled Name, True; ``OPEN``: a ``q``
search form, False). BASE is the same executor over the plain fake (no reads on the class), same phash salt: an
executed row's status, action, reason and ``resolved`` are byte-identical to it. Geometry by hand (1280x800):
``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``.

**Readings (the test-author's, reported, not guessed silently):**

- A4 "the target field counts as ``filled`` when the text is non-empty": A1's field keys do not say which field is the
  target. The fake marks it twice, by the ``name``/``id`` ``focused_field`` also reports and by an extra
  ``"focused": True`` key; the row that depends on this (only the target is personal, still empty at the read) is
  :func:`test_type_enter_withheld_target_counts_filled`. Every other withheld row has another personal field filled.
- A4 "Space applies when the focused element is itself the control": ``submit_facts_focused()`` takes no argument, so
  the facts cannot tell Enter's read from Space's; at the seam only "Space with True facts is refused" is pinned. A
  Space typed into a text field of a filled form is pinned end to end (browser module).
- Fail open (H-T206-5) on ``press_key``'s and ``type_at``'s focused read: the key is sent; the note is pinned only for
  the click read (A3 names it ``live click: submit check``).
- ``SUBMIT_TIMEOUT_S`` is not stated as read at call time; the hang rows wait the real 0.3 s.
- The overview-frame row clicks a point already in view (no scroll), so "nothing dispatched" is exact there.

Regime: CPU, fake drivers, no browser, no socket, 1280x800, n = 1, deterministic salt; the hang rows sleep 0.3 s each.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime
from local_astra.contracts.toolcall import ToolOutcome
from tests.unit.test_computer_executor import DriverBoom, FakeDriver, inv, own_frame
from tests.unit.test_computer_web_executor import FieldDriver
from tests.unit.test_t0151_executor import PageDriver, register
from tests.unit.test_t0192_checkout_executor import CHECKOUT, UrlsMixin
from tests.unit.test_t0203_follow_executor import FollowField
from tests.unit.test_t0204_preclick import GuardField

CLOCK_MS = 1_700_000_020_206
START = "http://127.0.0.1:8123"
SALT = 0x0206_5EED_0206_5EED
#: A3, verbatim
SUBMIT_REASON = "this button sends the form; the user presses it themselves"
NOTE = {"what": "live click: submit check", "result": "unknown"}
TOKEN = "T0206TOKz4m8"
INPUT = frozenset(
    {"mouse_move", "mouse_down", "mouse_up", "key_down", "key_up", "type_text", "scroll_by", "goto"}
)
READS = ("submit_facts", "submit_facts_focused")


# ============================================================================================ facts (hand-built)


def _ctl(
    label: str, tag: str = "button", type_: str = "submit", structural: bool = True
) -> dict[str, Any]:
    return {
        "tag": tag,
        "type": type_,
        "role": "",
        "label": label,
        "href_nav": False,
        "structural": structural,
    }


def _fld(
    name: str, *, type_: str = "text", tag: str = "input", filled: bool = True, **kw: Any
) -> dict[str, Any]:
    out = {
        "tag": tag,
        "type": type_,
        "autocomplete": "",
        "name": name,
        "id": name,
        "placeholder": "",
        "label": name.title(),
        "filled": filled,
        "visible": True,
    }
    out.update(kw)
    return out


def facts(
    control: dict[str, Any], fields: list[dict[str, Any]], scope: str = "form"
) -> dict[str, Any]:
    return {"control": control, "scope": scope, "search_scope": False, "fields": fields}


#: True by A1: "Request quote" with Name filled (predicate module, ``d7-request-quote-after-name``)
SEND = facts(_ctl("Request quote"), [_fld("name"), _fld("phone", type_="tel", filled=False)])
#: False by A1: a search form (``q``, "Google Search")
OPEN = facts(_ctl("Google Search", tag="input"), [_fld("q")])
#: the facts of a click on a text field of a send form: the control is the input itself (label "Name"), False by (d)
AT_FIELD = facts(
    _ctl("Name", tag="input", type_="text", structural=False), [_fld("name", filled=False)]
)
NAME_ATTRS = {"tag": "input", "type": "text", "name": "name", "id": "name"}
TEXTAREA_ATTRS = {"tag": "textarea", "type": "textarea", "name": "description", "id": "description"}


# ============================================================================================ fakes


class SubmitMixin:
    """A2's two reads over a fake driver (module docstring)."""

    calls: list[tuple[str, tuple[Any, ...]]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.at: Any = OPEN
        self.focused: Any = OPEN
        self.mode = "ok"
        self.fmode = "ok"

    async def _answer(self, mode: str, value: Any) -> Any:
        if mode == "raise":
            raise DriverBoom(f"synthetic submit read failure near {TOKEN}")
        if mode == "hang":
            await asyncio.Event().wait()
        if mode == "none":
            return None
        return json.loads(json.dumps(value))

    async def submit_facts(self, x: float, y: float) -> Any:
        self.calls.append(("submit_facts", (x, y)))
        return await self._answer(self.mode, self.at)

    async def submit_facts_focused(self) -> Any:
        self.calls.append(("submit_facts_focused", ()))
        return await self._answer(self.fmode, self.focused)


class SubmitDriver(SubmitMixin, FakeDriver):
    pass


class SubmitField(SubmitMixin, FieldDriver):
    pass


class SubmitPage(SubmitMixin, PageDriver):
    pass


class SubmitUrls(SubmitMixin, UrlsMixin, FieldDriver):
    pass


class SubmitGuard(SubmitMixin, GuardField):
    pass


class SubmitFollow(SubmitMixin, FollowField):
    pass


# ============================================================================================ helpers


@pytest.fixture(autouse=True)
def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_mod.secrets, "randbits", lambda k: SALT)


def reason() -> str:
    value = getattr(executor_mod, "SUBMIT_REASON", None)
    assert value is not None, "T-0206 A3: executor.SUBMIT_REASON does not exist yet"
    assert value == SUBMIT_REASON, value
    return str(value)


def build(
    driver: Any, *, web: bool = True, notes: list[dict[str, Any]] | None = None, **kw: Any
) -> Any:
    if web:
        kw.setdefault("start_origin", START)
    ex = executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=driver.viewport, driver=driver, web=web, **kw
    )
    if notes is not None:
        ex.on_note = notes.append
    return ex


def names(d: Any, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def dispatched(d: Any, since: int = 0) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in d.calls[since:] if c[0] in INPUT]


def dump(out: ToolOutcome) -> str:
    return json.dumps([out.status, out.action, out.reason, out.resolved])


async def act(ex: Any, tool: str, args: dict[str, Any]) -> ToolOutcome:
    frame = await own_frame(ex) if tool in ("click_at", "type_at") else None
    out = await ex.execute(inv(tool, frame, **args))
    assert isinstance(out, ToolOutcome)
    return out


def assert_refused(out: ToolOutcome, d: Any, mark: int, tool: str) -> None:
    """A3/A4: refused with SUBMIT_REASON; nothing dispatched or captured after ``mark``."""
    r = reason()
    assert (out.status, out.reason) == ("refused", r), out
    assert out.action == f"refused {tool}: {r}", out.action
    assert out.resolved == {}, out.resolved
    assert dispatched(d, mark) == [], f"A3: dispatched a send: {dispatched(d, mark)}"
    after = names(d, mark)
    for name in ("screenshot", "full_page_screenshot", "focused_field"):
        assert name not in after, f"A3: {name} ran after a send was refused: {after}"


CLICK = {"x": 500, "y": 500}
TYPE = {"x": 500, "y": 500, "text": "Mrs Kettle", "press_enter": False}
TYPE_ENTER = {"x": 500, "y": 500, "text": "Mrs Kettle", "press_enter": True}


def test_constants_and_exports() -> None:
    """A3: ``SUBMIT_REASON`` verbatim, ``SUBMIT_TIMEOUT_S = 0.3``, both exported."""
    reason()
    assert getattr(executor_mod, "SUBMIT_TIMEOUT_S", None) == 0.3
    assert {"SUBMIT_REASON", "SUBMIT_TIMEOUT_S"} <= set(executor_mod.__all__)


def test_real_driver_has_both_reads() -> None:
    """A2/A7: ``PlaywrightDriver.submit_facts(x, y)`` and ``submit_facts_focused()`` exist on the class, async."""
    for name, params in (("submit_facts", ["self", "x", "y"]), ("submit_facts_focused", ["self"])):
        fn = getattr(runtime.PlaywrightDriver, name, None)
        assert fn is not None, f"T-0206 A2: PlaywrightDriver has no {name} yet"
        assert inspect.iscoroutinefunction(fn), f"A2: {name} is async"
        assert list(inspect.signature(fn).parameters) == params, (name, inspect.signature(fn))


# ============================================================================================ A3: clicks


async def test_click_on_send_refused() -> None:
    """A3: ``click_at`` whose read says True is refused with SUBMIT_REASON; nothing dispatched or captured; the read
    is at the CSS point (640, 400), after ``ensure()``."""
    d = SubmitDriver()
    ex = build(d)
    f = await own_frame(ex)
    d.at = SEND
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", f, **CLICK))
    assert_refused(out, d, mark, "click_at")
    reads = [c for c in d.calls[mark:] if c[0] == "submit_facts"]
    assert len(reads) == 1, names(d, mark)
    assert tuple(round(float(v)) for v in reads[0][1]) == (640, 400), reads
    seq = names(d, mark)
    assert seq.index("ensure") < seq.index("submit_facts"), seq


async def test_type_at_focus_click_on_send_refused() -> None:
    """A3: ``type_at`` whose point is a send control is refused before its focus click: no click, no focus read, no
    key."""
    d = SubmitField(NAME_ATTRS)
    ex = build(d)
    f = await own_frame(ex)
    d.at = SEND
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", f, **TYPE_ENTER))
    assert_refused(out, d, mark, "type_at")
    assert "submit_facts" in names(d, mark)


async def test_overview_frame_click_on_send_refused() -> None:
    """A3: the overview-frame click (T-0151) at a point in view reads first and is refused: no mouse event, no key."""
    d = SubmitPage(content_height=1220.0)
    ex = build(d, echo="model")
    await own_frame(ex)
    ov = await register(ex)
    d.at = SEND
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=500, y=100))
    r = reason()
    assert (out.status, out.reason, out.resolved) == ("refused", r, {}), out
    assert out.action == f"refused click_at: {r}", out.action
    assert [c for c in dispatched(d, mark) if c[0] != "scroll_by"] == [], dispatched(d, mark)
    assert "submit_facts" in names(d, mark)


async def test_click_open_form_executes_as_base() -> None:
    """A3/A5: a read that says False executes; status, action, reason and ``resolved`` equal BASE's (the plain fake);
    the read did run (the row is about its verdict)."""
    base = await act(build(FakeDriver()), "click_at", CLICK)
    d = SubmitDriver()
    d.at = OPEN
    ex = build(d)
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed", out
    assert "submit_facts" in names(d), names(d)
    assert dump(out) == dump(base)
    assert "submit_withheld" not in out.resolved


@pytest.mark.parametrize("mode", ["none", "raise", "hang"])
async def test_click_read_fails_open_with_note(mode: str) -> None:
    """A3/H-T206-5: a read that is ``None``, raises or outlives ``SUBMIT_TIMEOUT_S`` (0.3 s) lets the click proceed as
    BASE, and ``on_note`` gets exactly ``{"what": "live click: submit check", "result": "unknown"}`` once."""
    base = await act(build(FakeDriver()), "click_at", CLICK)
    d = SubmitDriver()
    d.at, d.mode = SEND, mode
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own_frame(ex)
    t0 = time.monotonic()
    out = await ex.execute(inv("click_at", f, **CLICK))
    took = time.monotonic() - t0
    assert out.status == "executed", out
    assert "submit_facts" in names(d), names(d)
    assert dump(out) == dump(base)
    assert [n for n in notes if n.get("what") == NOTE["what"]] == [NOTE], notes
    if mode == "hang":
        assert took < 2.0, f"A3: the read is bounded by SUBMIT_TIMEOUT_S: {took:.2f} s"


async def test_open_read_makes_no_note() -> None:
    """A3: a read that answers (True or False) makes no submit-check note."""
    for value in (OPEN, SEND):
        d = SubmitDriver()
        d.at = value
        notes: list[dict[str, Any]] = []
        ex = build(d, notes=notes)
        await act(ex, "click_at", CLICK)
        assert "submit_facts" in names(d), names(d)
        assert [n for n in notes if n.get("what") == NOTE["what"]] == [], notes


# ============================================================================================ A4: Enter and Space


@pytest.mark.parametrize("combo", ["Enter", "Ctrl+Enter", "Space"])
async def test_enter_space_refused(combo: str) -> None:
    """A4: a ``press_key`` chord holding Enter (``Enter``, ``Ctrl+Enter``) or Space, with focused facts True, is
    refused with SUBMIT_REASON; no key is sent."""
    d = SubmitDriver()
    ex = build(d)
    await own_frame(ex)
    d.focused = SEND
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo=combo))
    assert_refused(out, d, mark, "press_key")
    assert "submit_facts_focused" in names(d, mark)


@pytest.mark.parametrize("combo", ["Enter", "Space"])
async def test_enter_space_open_form_as_base(combo: str) -> None:
    """A4/A5: focused facts False: the key is sent and the outcome is BASE's (and the read ran)."""
    base = await act(build(FakeDriver()), "press_key", {"combo": combo})
    d = SubmitDriver()
    d.focused = OPEN
    out = await act(build(d), "press_key", {"combo": combo})
    assert "submit_facts_focused" in names(d)
    assert dump(out) == dump(base)


@pytest.mark.parametrize("combo", ["Tab", "Escape", "ArrowDown", "Ctrl+A", "Backspace"])
async def test_other_keys_never_read(combo: str) -> None:
    """A4 (active at BASE; the too-wide side): only chords holding Enter or Space read; other keys execute as BASE with no read."""
    base = await act(build(FakeDriver()), "press_key", {"combo": combo})
    d = SubmitDriver()
    d.at = d.focused = SEND
    out = await act(build(d), "press_key", {"combo": combo})
    assert not set(READS) & set(names(d)), names(d)
    assert dump(out) == dump(base)


@pytest.mark.parametrize("fmode", ["none", "raise", "hang"])
async def test_enter_read_fails_open(fmode: str) -> None:
    """H-T206-5 for the focused read: ``None``, a raise or a hang lets Enter through (the key is sent)."""
    d = SubmitDriver()
    d.focused, d.fmode = SEND, fmode
    ex = build(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo="Enter"))
    assert "submit_facts_focused" in names(d, mark), names(d, mark)
    assert out.status == "executed", out
    assert ("key_down", ("Enter",)) in dispatched(d, mark), dispatched(d, mark)


async def _type(
    d: Any, args: dict[str, Any], notes: list[dict[str, Any]] | None = None
) -> tuple[ToolOutcome, int]:
    ex = build(d, notes=notes)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", f, **args))
    assert isinstance(out, ToolOutcome)
    return out, mark


async def test_type_enter_withheld() -> None:
    """A4: ``type_at`` + Enter in a send form (another personal field already filled): the text is typed (T-0189
    replace applies to the single-line input), Enter is not pressed, ``executed``, the action is BASE's for the same
    call with ``press_enter=False``, ``resolved["submit_withheld"] is True``. The focused read comes after
    ``focused_field`` and before any key."""
    base, _ = await _type(FieldDriver(NAME_ATTRS), TYPE)
    d = SubmitField(NAME_ATTRS)
    d.at = AT_FIELD
    d.focused = facts(
        _ctl("Request quote"),
        [_fld("name", filled=False, focused=True), _fld("email", type_="email")],
    )
    out, mark = await _type(d, TYPE_ENTER)
    assert out.status == "executed", out
    assert out.action == base.action, (out.action, base.action)
    assert out.resolved.get("submit_withheld") is True, out.resolved
    keys = dispatched(d, mark)
    assert ("type_text", ("Mrs Kettle",)) in keys, keys
    assert ("key_down", ("Enter",)) not in keys, f"A4: Enter was pressed: {keys}"
    assert out.resolved.get("replaced") is True, out.resolved
    seq = names(d, mark)
    first_key = min(i for i, n in enumerate(seq) if n in ("key_down", "type_text"))
    assert seq.index("focused_field") < seq.index("submit_facts_focused") < first_key, seq


async def test_withheld_enter_sends_no_chord_and_no_late_follow() -> None:
    """Reviewer gap m18: a withheld Enter is BASE's ``press_enter=False`` call, driver call for driver call (the two
    reads aside), on a driver with T-0174/T-0203's navigation capability: no chord is held or released and no T-0203
    late read/follow (which only an action that sent Enter makes) follows the capture."""
    base_d = FollowField(NAME_ATTRS)
    base, _ = await _type(base_d, TYPE)
    d = SubmitFollow(NAME_ATTRS)
    d.at = AT_FIELD
    d.focused = facts(
        _ctl("Request quote"),
        [_fld("name", filled=False, focused=True), _fld("email", type_="email")],
    )
    out, _ = await _type(d, TYPE_ENTER)
    assert out.resolved.get("submit_withheld") is True, out.resolved
    seq = [c[0] for c in d.calls if c[0] not in READS]
    assert seq == [c[0] for c in base_d.calls], (seq, [c[0] for c in base_d.calls])
    assert out.action == base.action


async def test_type_enter_withheld_target_counts_filled() -> None:
    """A4: the target field counts as filled when the text is non-empty: D7 with only the Name field personal and
    still empty at the read (module docstring reading: the target is marked by name/id and ``focused``)."""
    d = SubmitField(NAME_ATTRS)
    d.at = AT_FIELD
    d.focused = facts(
        _ctl("Request quote"),
        [_fld("name", filled=False, focused=True), _fld("quantity", type_="number", filled=False)],
    )
    out, mark = await _type(d, TYPE_ENTER)
    assert out.status == "executed", out
    assert out.resolved.get("submit_withheld") is True, out.resolved
    assert ("key_down", ("Enter",)) not in dispatched(d, mark)


async def test_type_enter_open_form_presses_enter() -> None:
    """A4/A5 partner: focused facts False (a search form): Enter is pressed, the outcome equals BASE's, and there is no
    ``submit_withheld`` key."""
    base, _ = await _type(FieldDriver({"tag": "input", "type": "text", "name": "q"}), TYPE_ENTER)
    d = SubmitField({"tag": "input", "type": "text", "name": "q"})
    d.at = d.focused = OPEN
    out, mark = await _type(d, TYPE_ENTER)
    assert ("key_down", ("Enter",)) in dispatched(d, mark)
    assert "submit_facts_focused" in names(d, mark)
    assert dump(out) == dump(base)


async def test_textarea_never_withholds() -> None:
    """A4 (mutation 6; active at BASE, the too-wide side): a textarea target never withholds, because Enter there is a newline: Enter is sent and there
    is no ``submit_withheld``."""
    d = SubmitField(TEXTAREA_ATTRS)
    d.at = facts(_ctl("Description", tag="textarea", type_="textarea", structural=False), [])
    d.focused = facts(
        _ctl("Request quote"), [_fld("name"), _fld("description", tag="textarea", type_="textarea")]
    )
    out, mark = await _type(d, TYPE_ENTER)
    assert out.status == "executed", out
    assert ("key_down", ("Enter",)) in dispatched(d, mark), dispatched(d, mark)
    assert "submit_withheld" not in out.resolved, out.resolved


async def test_type_without_enter_never_withholds() -> None:
    """A4 (active at BASE): ``press_enter=False`` in a send form is BASE's outcome (no ``submit_withheld`` key)."""
    base, _ = await _type(FieldDriver(NAME_ATTRS), TYPE)
    d = SubmitField(NAME_ATTRS)
    d.at, d.focused = AT_FIELD, SEND
    out, _ = await _type(d, TYPE)
    assert dump(out) == dump(base)


# ============================================================================================ check order


async def test_page_changed_before_submit_read() -> None:
    """A3 order (active: holds at BASE, guards against a read placed too early): a popup adopted in ``ensure()`` is refused with PAGE_CHANGED_REASON and no submit read runs."""
    d = SubmitDriver()
    ex = build(d)
    f = await own_frame(ex)
    d.at = SEND
    d.open_popup()
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", f, **CLICK))
    assert (out.status, out.reason) == ("refused", executor_mod.PAGE_CHANGED_REASON), out
    assert not set(READS) & set(names(d, mark)), names(d, mark)


@pytest.mark.parametrize(
    ("tool", "args"),
    [("click_at", CLICK), ("press_key", {"combo": "Enter"}), ("type_at", TYPE_ENTER)],
)
async def test_checkout_before_submit_read(tool: str, args: dict[str, Any]) -> None:
    """A3 order (active at BASE): on a checkout page with a send control both guards could deny; the checkout's reason wins and no
    submit read runs."""
    d = SubmitUrls(NAME_ATTRS)
    ex = build(d)
    frame = await own_frame(ex) if tool != "press_key" else None
    d.at = d.focused = SEND
    d.visit(CHECKOUT)
    mark = len(d.calls)
    out = await ex.execute(inv(tool, frame, **args))
    assert (out.status, out.reason) == ("refused", executor_mod.CHECKOUT_REASON), out
    assert out.reason != SUBMIT_REASON
    assert not set(READS) & set(names(d, mark)), names(d, mark)


@pytest.mark.parametrize(("tool", "args"), [("click_at", CLICK), ("type_at", TYPE_ENTER)])
async def test_moved_before_submit_read(tool: str, args: dict[str, Any]) -> None:
    """A3 order (mutation 7; active at BASE): a page that moved near the point is refused with MOVED_REASON before any submit read,
    though the read would say True."""
    d = SubmitGuard(NAME_ATTRS)
    ex = build(d)
    f = await own_frame(ex)
    d.at = SEND
    d.shift_script.append(((630.0, 390.0, 20.0, 20.0),))
    # T-0234 B2 (named re-pin): a shift decides only without a crop, so the crop fails here
    d.region_script.append(RuntimeError("no crop"))
    mark = len(d.calls)
    out = await ex.execute(inv(tool, f, **args))
    assert (out.status, out.reason) == ("refused", executor_mod.MOVED_REASON), out
    assert out.reason != SUBMIT_REASON
    assert not set(READS) & set(names(d, mark)), names(d, mark)


@pytest.mark.parametrize(("tool", "args"), [("click_at", CLICK), ("type_at", TYPE)])
async def test_submit_read_after_moved_guard_before_dispatch(
    tool: str, args: dict[str, Any]
) -> None:
    """A3 order: with the moved guard passing, the read runs after its ``screenshot_region``/``shifts_since`` and
    before the first ``mouse_move``; on True the refusal is SUBMIT_REASON and nothing is dispatched or captured (the
    settle takes no shot)."""
    d = SubmitGuard(NAME_ATTRS)
    ex = build(d)
    f = await own_frame(ex)
    d.at = SEND
    mark = len(d.calls)
    out = await ex.execute(inv(tool, f, **args))
    assert_refused(out, d, mark, tool)
    seq = names(d, mark)
    assert "screenshot_region" in seq and "shifts_since" in seq and "submit_facts" in seq, seq
    assert seq.index("shifts_since") < seq.index("submit_facts"), seq


async def test_submit_read_before_mouse_move_when_open() -> None:
    """A3 order, executed partner: read after the guard, before ``mouse_move``."""
    d = SubmitGuard(NAME_ATTRS)
    ex = build(d)
    f = await own_frame(ex)
    d.at = OPEN
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", f, **CLICK))
    assert out.status == "executed", out
    seq = names(d, mark)
    assert "submit_facts" in seq, f"T-0206 A3: no submit read: {seq}"
    assert seq.index("shifts_since") < seq.index("submit_facts") < seq.index("mouse_move"), seq


# ============================================================================================ A7: outside web mode


SCRIPT = [
    ("click_at", CLICK),
    ("type_at", TYPE_ENTER),
    ("press_key", {"combo": "Enter"}),
    ("press_key", {"combo": "Space"}),
    ("scroll", {"direction": "down"}),
]


async def run_script(d: Any, **kw: Any) -> list[str]:
    ex = build(d, **kw)
    return [dump(await act(ex, tool, args)) for tool, args in SCRIPT]


async def test_non_web_never_reads() -> None:
    """A7 (active: holds at BASE): with ``web=False`` a driver whose reads say True is never read; the call sequence and
    every outcome equal the plain fake's."""
    base_d = FakeDriver()
    base = await run_script(base_d, web=False)
    d = SubmitDriver()
    d.at = d.focused = SEND
    got = await run_script(d, web=False)
    assert not set(READS) & set(names(d)), names(d)
    assert got == base
    assert d.calls == base_d.calls


async def test_web_driver_without_methods_is_base() -> None:
    """A7 (active): a web-mode fake whose *class* lacks the reads runs BASE's path (instance attributes are not the
    capability, as ``focused_field``)."""
    base = await run_script(FieldDriver(NAME_ATTRS))
    d = FieldDriver(NAME_ATTRS)

    async def says_true(*a: Any) -> Any:
        raise AssertionError("A7: an instance attribute was used as the capability")

    d.__dict__["submit_facts"] = says_true
    d.__dict__["submit_facts_focused"] = says_true
    assert await run_script(d) == base


# ============================================================================================ A8: privacy


async def test_token_nowhere() -> None:
    """A8: a token in a button label, a field label and name, and a read's exception message, through a refused click,
    a withheld Enter, a failed read and a refused Enter, is in no outcome, no ``take_events()`` entry, no note and no
    ``egress_log`` entry."""

    class Tracked(SubmitField):
        def __init__(self, *a: Any) -> None:
            super().__init__(*a)
            self.taken: list[dict[str, Any]] = []

        def take_events(self) -> list[dict[str, Any]]:
            got = super().take_events()
            self.taken.extend(got)
            return got

    d = Tracked(NAME_ATTRS)
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    tok_send = facts(
        _ctl(f"Submit {TOKEN}"), [_fld(f"name-{TOKEN}", label=f"Name {TOKEN}", focused=True)]
    )
    outs = []
    d.at = tok_send
    outs.append(await act(ex, "click_at", CLICK))
    d.at, d.focused = AT_FIELD, tok_send
    outs.append(await act(ex, "type_at", TYPE_ENTER))
    d.mode = "raise"
    outs.append(await act(ex, "click_at", CLICK))
    d.mode = "ok"
    outs.append(await ex.execute(inv("press_key", None, combo="Enter")))
    assert [o.status for o in outs] == ["refused", "executed", "executed", "refused"], outs
    assert outs[1].resolved.get("submit_withheld") is True, outs[1]
    blob = (
        repr(outs)
        + json.dumps(d.taken)
        + json.dumps(notes, default=str)
        + json.dumps(ex.egress_log)
    )
    assert TOKEN not in blob, f"A8: the token left the executor: {blob}"
    assert NOTE in notes, notes


async def test_raise_message_not_in_outcome() -> None:
    """A8: the read's exception message (with the token) is not in the outcome or the note."""
    d = SubmitDriver()
    d.at, d.mode = SEND, "raise"
    notes: list[dict[str, Any]] = []
    out = await act(build(d, notes=notes), "click_at", CLICK)
    assert "submit_facts" in names(d), names(d)
    assert TOKEN not in repr(out) and "synthetic" not in repr(out), out
    assert TOKEN not in json.dumps(notes, default=str), notes


# ============================================================================================ A5: schemas


@pytest.mark.parametrize("name", ["tools_v1.json", "tools_v1_overview.json"])
def test_tool_schemas_unchanged(name: str) -> None:
    """A5 (active): the model-facing tool schemas are byte-identical to BASE 0056daa2 (sha256 of ``git show
    0056daa2:local_astra/toolcall/<name>``, taken by the test-author on 2026-10-30)."""
    want = {
        "tools_v1.json": "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707",
        "tools_v1_overview.json": "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc",
    }[name]
    path = Path(executor_mod.__file__).resolve().parents[1] / "toolcall" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == want
