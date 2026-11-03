"""T-0160 group B, executor rows (B-2, B-3, B-4, B-6, B-8) over fake drivers: ``BrowserToolExecutor`` in web mode
without a browser (docs/tasks/T-0160.md).

**The surface these rows pin** (B-1..B-6's names; the readings marked *reading* are the test-author's where the task
names behaviour but not the seam, reported to the orchestrator):

- ``BrowserToolExecutor(..., web: bool = False, start_origin: str | None = None, proxy_resolve=None,
  proxy_connect=None)``. ``web`` and ``start_origin`` are the task's (Definitions, C-1). *Reading:* the two proxy
  seams are forwarded to ``PlaywrightDriver`` (B-1 names them there; B-7 needs them on "the real executor").
- ``executor.SECRET_FIELD_REASON`` (B-3 (h), exact words), ``executor.CARD_TEXT_REASON`` and
  ``executor.FOCUS_UNKNOWN_REASON`` (B-3 (a)/(d); names pinned, wording not), ``executor.FOCUS_TIMEOUT_S == 2.0``
  (T-0194 criterion 1; 0.5 s at T-0160).
- the driver's optional ``async focused_field() -> Mapping[str, str]``, looked up on the driver's *class* (as
  ``layout_metrics``): the deepest focused element's attributes over A-4's keys (``tag, type, autocomplete, name,
  id, aria_label, placeholder, label``), classified with ``fence.secret_field``.
- *Reading (B-6 hosts):* the driver reports each tunnelled host as a ``take_events()`` event ``{"kind": "host",
  "host": <host>}`` (the channel ``_report`` already maps into ``resolved``); the executor turns them into
  ``resolved["hosts"]`` (unique, at most 32) and ``resolved["hosts_more"]``.

**Fakes.** T-0111c's ``FakeDriver`` (its sentinel makes any unfaked driver name a named failure) plus a
``focused_field`` whose answer is scripted (attributes, a raise, or a hang); T-0151's ``PageDriver`` likewise for the
overview-frame path (B-3 (f)). Nothing here opens a socket.

**Expected values by hand** (viewport 1280x800): ``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``.
``4111 1111 1111 1111`` and ``5555555555554444`` pass Luhn (the classic test numbers); ``+33 6 12 34 56 78`` is a
10-digit phone (A-3's false row).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.contracts.toolcall import ToolOutcome
from tests.unit.test_computer_executor import DriverBoom, FakeDriver, inv, own_frame
from tests.unit.test_t0151_executor import PageDriver, register

CLOCK_MS = 1_700_000_016_000
START = "http://127.0.0.1:8123"
SECRET = "will not type into a password or card field; ask the user to type it themselves"
PASTE_OFF = "pasting is off on real websites"
OPENED = "opened the start page"
#: a typed text no reason, action or resolved value could contain by accident
TEXT = "zebra-quokka-17"
TEXT_FIELD = {"tag": "input", "type": "text", "name": "q"}
PASSWORD_FIELDS = [
    pytest.param({"tag": "input", "type": "password", "name": "pw"}, id="type-password"),
    pytest.param(
        {"tag": "input", "type": "text", "autocomplete": "current-password"}, id="ac-current"
    ),
    pytest.param(
        {"tag": "input", "type": "text", "autocomplete": "username new-password"}, id="ac-new"
    ),
]
CARD_FIELDS = [
    pytest.param({"tag": "input", "type": "text", "autocomplete": "cc-number"}, id="ac-cc-number"),
    pytest.param({"tag": "input", "type": "tel", "name": "card-number"}, id="name-card-number"),
    pytest.param({"tag": "input", "type": "text", "id": "CVV2"}, id="id-cvv"),
    pytest.param(
        {"tag": "input", "type": "text", "aria_label": "Card number"}, id="aria-card-number"
    ),
]
#: A-4's negative rows: not secret, so typing goes ahead (the guard must not over-refuse)
NOT_SECRET_FIELDS = [
    pytest.param({"tag": "input", "type": "text", "name": "giftcard_search"}, id="giftcard-search"),
    pytest.param({"tag": "input", "type": "text", "name": "postcode"}, id="postcode"),
    pytest.param({"tag": "input", "type": "text", "autocomplete": "cc-name"}, id="cc-name"),
    pytest.param({"tag": "input", "type": "search"}, id="search"),
]
CARD_TEXTS = ["4111 1111 1111 1111", "4111-1111-1111-1111", "pay with 5555555555554444 now"]
#: T-0194: the hang rows monkeypatch ``executor.FOCUS_TIMEOUT_S`` to this (a hang is retried once, so it costs two
#: bounds); the executor reads the module constant at call time
HANG_BOUND = 0.05


class FieldDriver(FakeDriver):
    """T-0111c's fake plus the optional ``focused_field`` (module docstring): ``mode`` is ``"ok"`` (return
    ``attrs``), ``"raise"`` (``DriverBoom``), ``"hang"`` (never return until cancelled), ``"none"`` (``None``: nothing
    holds the focus) or ``"notmapping"`` (a reply that is not a mapping: a list of pairs)."""

    def __init__(self, attrs: dict[str, str] | None = None, mode: str = "ok") -> None:
        super().__init__()
        self.attrs = dict(attrs if attrs is not None else TEXT_FIELD)
        self.mode = mode

    async def focused_field(self) -> Any:
        self._enter("focused_field")
        if self.mode == "raise":
            raise DriverBoom("synthetic focus read failure")
        if self.mode == "hang":
            await asyncio.Event().wait()
        if self.mode == "none":
            return None
        if self.mode == "notmapping":
            return [("tag", "input"), ("type", "password")]
        return dict(self.attrs)


class FieldPageDriver(PageDriver):
    """T-0151's overview fake with the same scripted ``focused_field``."""

    def __init__(self, attrs: dict[str, str] | None = None, mode: str = "ok") -> None:
        super().__init__(content_height=1220.0)
        self.attrs = dict(attrs if attrs is not None else TEXT_FIELD)
        self.mode = mode

    async def focused_field(self) -> dict[str, str]:
        self._enter("focused_field")
        if self.mode == "raise":
            raise DriverBoom("synthetic focus read failure")
        if self.mode == "hang":
            await asyncio.Event().wait()
        return dict(self.attrs)


def const(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0160 B-3: executor.{name} does not exist yet"
    return value


def require_keywords(cls: type, *names: str) -> None:
    """A missing T-0160 keyword is a named assertion, never a ``TypeError`` (the pending convention)."""
    params = inspect.signature(cls.__init__).parameters
    missing = [n for n in names if n not in params]
    assert not missing, f"T-0160 B-1: {cls.__name__} has no keyword(s) {missing} yet"


def new_executor(**kw: Any) -> Any:
    require_keywords(executor_mod.BrowserToolExecutor, "web", "start_origin")
    return executor_mod.BrowserToolExecutor(**kw)


def build(driver: FakeDriver, *, web: bool = True, **kw: Any) -> Any:
    kw.setdefault("start_origin", START if web else None)
    return new_executor(
        now_ms=lambda: CLOCK_MS, viewport=driver.viewport, driver=driver, web=web, **kw
    )


def names(d: FakeDriver, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def typed_anywhere(out: ToolOutcome, text: str) -> list[str]:
    """Where ``text`` (or its digits, for a card number) shows in the outcome's resolved values, action or reason."""
    needles = {text}
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        needles.add(digits)
    hay = {
        "resolved": json.dumps(out.resolved, default=str),
        "action": out.action or "",
        "reason": out.reason or "",
    }
    return [k for k, v in hay.items() for n in needles if n in v]


def assert_guard_failed(d: FakeDriver, out: ToolOutcome, mark: int, reason: str) -> None:
    """B-3 (c)/(d): ``failed`` with ``reason`` after the click: the button went down and up, the field was read after
    it (when the driver has a reader), no keystroke at all (no ``type_text``, no Enter), the frame re-captured."""
    assert (out.status, out.reason) == ("failed", reason), out
    seq = names(d, mark)
    assert "mouse_up" in seq, (
        f"B-3 (b): the guard runs after the click, which did not happen: {seq}"
    )
    assert not {"type_text", "key_down", "key_up"} & set(seq), (
        f"a keystroke was dispatched into a guarded field: {seq}"
    )
    if "focused_field" in seq:
        assert seq.index("focused_field") > seq.index("mouse_up"), seq
    assert "screenshot" in seq[seq.index("mouse_up") :], (
        f"B-3 (c): the frame was not re-captured: {seq}"
    )
    assert "typed" in out.resolved and out.resolved["typed"] == 0, (
        f"B-3 (c)/(d): a guarded outcome carries typed: 0, as the secret-field rows pin: {out.resolved}"
    )
    assert typed_anywhere(out, TEXT) == [], f"B-3 (g)/B-6: the typed text is logged: {out}"


# =====================================================================================================
# B-3 (h), (b): the names
# =====================================================================================================


def test_guard_constants() -> None:
    """B-3 (h) wording, exact; B-3 (b) ``FOCUS_TIMEOUT_S`` 2.0 s (T-0194 criterion 1; 0.5 s at T-0160), inside
    ``ACTION_TIMEOUT_S``; the other two reasons exist, are distinct and are not the secret-field wording."""
    assert const("SECRET_FIELD_REASON") == SECRET
    assert const("FOCUS_TIMEOUT_S") == 2.0
    assert const("FOCUS_TIMEOUT_S") < executor_mod.ACTION_TIMEOUT_S
    reasons = {SECRET, const("CARD_TEXT_REASON"), const("FOCUS_UNKNOWN_REASON")}
    assert len(reasons) == 3 and all(isinstance(r, str) and r for r in reasons), reasons


# =====================================================================================================
# B-3 (a): card-like text, every mode, before anything is dispatched
# =====================================================================================================


@pytest.mark.parametrize("text", CARD_TEXTS)
@pytest.mark.parametrize(
    "web, driver_cls",
    [(True, FieldDriver), (False, FieldDriver), (False, FakeDriver)],
    ids=["web", "loopback", "base-driver"],
)
async def test_card_text_refused_before_dispatch(text: str, web: bool, driver_cls: type) -> None:
    """B-3 (a), in every mode, the BASE driver class included: ``refused`` with ``CARD_TEXT_REASON``, no input
    dispatched (no click either), the field never read, and the number nowhere in the outcome (B-3 (g))."""
    d = driver_cls()
    ex = build(d, web=web)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", f, x=500, y=500, text=text, press_enter=True))
    assert (out.status, out.reason) == ("refused", const("CARD_TEXT_REASON")), out
    assert d.dispatched(mark) == [], d.dispatched(mark)
    assert "focused_field" not in names(d, mark)
    assert typed_anywhere(out, text) == [], f"B-3 (g): the card number is in the outcome: {out}"


@pytest.mark.parametrize("text", ["+33 6 12 34 56 78", "2026-10-26", "4111111111111112"])
async def test_not_card_text_types(text: str) -> None:
    """B-3 (a) does not over-refuse: A-3's false rows (a phone, a date, a Luhn-invalid 16 digits) type normally."""
    d = FieldDriver()
    ex = build(d)
    f = await own_frame(ex)
    out = await ex.execute(inv("type_at", f, x=500, y=500, text=text, press_enter=False))
    assert out.status == "executed", out
    assert d.typed == [text]


# =====================================================================================================
# B-3 (b), (c): the check after the click, a secret field fails
# =====================================================================================================


@pytest.mark.parametrize(
    "kind, attrs",
    [pytest.param("password", *p.values, id=p.id) for p in PASSWORD_FIELDS]
    + [pytest.param("card", *p.values, id=p.id) for p in CARD_FIELDS],
)
@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
async def test_secret_field_failed_without_keystroke(
    kind: str, attrs: dict[str, str], web: bool
) -> None:
    """B-3 (b)/(c)/(e): after the click the focused field is read; a password or card field is ``failed`` with
    ``SECRET_FIELD_REASON`` and ``resolved`` carries ``secret_field`` and ``typed: 0``. No ``type_text``, no Enter
    (``press_enter`` is true here), the frame re-captured (``frame_after`` is the new ``latest_frame``). With
    ``web=False`` a driver that has ``focused_field`` (the real one always has it) is guarded the same way (B-3 (e))."""
    d = FieldDriver(attrs)
    ex = build(d, web=web)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=True))
    assert_guard_failed(d, out, mark, SECRET)
    assert "focused_field" in names(d, mark), "B-3 (b): the focused field was never read"
    assert out.resolved.get("secret_field") == kind, out.resolved
    assert out.resolved.get("typed") == 0, out.resolved
    assert d.presses[-1] == (640, 400), d.presses
    latest = ex.latest_frame
    assert latest is not None and latest.phash != f.phash
    assert out.resolved.get("frame_after") == latest.phash, out.resolved


@pytest.mark.parametrize("attrs", [*NOT_SECRET_FIELDS, pytest.param(TEXT_FIELD, id="text")])
async def test_plain_field_reads_then_types(attrs: Any) -> None:
    """B-3 (b): on a field that is not secret the read sits between the click and the first keystroke, and the call
    is BASE's ``executed`` with BASE's action string."""
    d = FieldDriver(attrs)
    ex = build(d)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=True))
    assert out.status == "executed", out
    assert (
        out.action == f"clicked at (640, 400) and typed {len(TEXT)} characters and pressed Enter"
    ), out.action
    seq = names(d, mark)
    assert "focused_field" in seq, f"B-3 (b): the focused field was never read before typing: {seq}"
    assert seq.index("mouse_up") < seq.index("focused_field") < seq.index("type_text"), seq
    assert d.typed == [TEXT]


# =====================================================================================================
# B-3 (d), (e): fail closed in web mode; BASE outside it
# =====================================================================================================


@pytest.mark.parametrize("how", ["missing", "raise", "hang", "none", "notmapping"])
async def test_focus_unknown_fails_closed_in_web(how: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """B-3 (d), adversarial (the named failure mode: typing into a field that could not be checked): in web mode a
    driver class without ``focused_field``, a read that raises, a read that never returns, a read that answers
    ``None`` and one that answers something other than a mapping are all ``failed`` with ``FOCUS_UNKNOWN_REASON``,
    nothing typed; the hang is cut at ``FOCUS_TIMEOUT_S`` per read, two reads (T-0194: one retry; the bound is
    monkeypatched small here; 1.5 s allowed for a loaded box), well inside ``ACTION_TIMEOUT_S``. *Reading* (reviewer
    A, 2026-10-27): ``None`` and a non-mapping are "could not be checked" in web mode; the task names only missing,
    timeout and exception."""
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", HANG_BOUND)
    d: FakeDriver = FakeDriver() if how == "missing" else FieldDriver(mode=how)
    ex = build(d)
    f = await own_frame(ex)
    mark = len(d.calls)
    t0 = time.monotonic()
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=True)), 5.0
    )
    elapsed = time.monotonic() - t0
    assert_guard_failed(d, out, mark, const("FOCUS_UNKNOWN_REASON"))
    if how == "hang":
        assert elapsed < 2 * executor_mod.FOCUS_TIMEOUT_S + 1.5, (
            f"the focus read was not bounded by FOCUS_TIMEOUT_S: {elapsed:.2f} s"
        )


async def test_loopback_base_driver_types_as_base() -> None:
    """B-3 (e) / B-8: ``web=False`` with a driver class lacking ``focused_field`` runs BASE's path unchanged: the
    click, the keystrokes, Enter, BASE's action string and ``resolved``."""
    d = FakeDriver()
    ex = build(d, web=False)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", f, x=500, y=500, text="hello", press_enter=True))
    assert out.status == "executed", out
    assert out.action == "clicked at (640, 400) and typed 5 characters and pressed Enter"
    assert [c[0] for c in d.dispatched(mark)] == [
        "mouse_move",
        "mouse_down",
        "mouse_up",
        "type_text",
        "key_down",
        "key_up",
    ]
    assert {k: out.resolved[k] for k in ("x_px", "y_px", "text", "press_enter")} == {
        "x_px": 640,
        "y_px": 400,
        "text": "hello",
        "press_enter": True,
    }
    assert "secret_field" not in out.resolved and "typed" not in out.resolved


@pytest.mark.parametrize("how", ["none", "raise", "hang", "notmapping"])
async def test_loopback_unreadable_focus_types_as_base(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-3 (d)/(e) partners of the web-mode fail-closed rows (the guard too wide; reviewers A and B): with
    ``web=False`` a ``focused_field`` that answers ``None``, raises, hangs past ``FOCUS_TIMEOUT_S`` or answers a
    non-mapping types as BASE does: ``executed`` with BASE's action, BASE's dispatch after the read (the text once,
    then Enter), no ``secret_field`` or ``typed`` key. (d) fails closed in web mode only. (T-0194: the read bound is
    monkeypatched small, so the retried hang stays fast.)"""
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", HANG_BOUND)
    d = FieldDriver(mode=how)
    ex = build(d, web=False)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", f, x=500, y=500, text="hello", press_enter=True)), 5.0
    )
    assert out.status == "executed", out
    assert out.action == "clicked at (640, 400) and typed 5 characters and pressed Enter"
    seq = names(d, mark)
    assert "focused_field" in seq, "premise: the loopback guard read the field"
    after = [c[0] for c in d.dispatched(mark)]
    assert after[after.index("mouse_up") + 1 :] == ["type_text", "key_down", "key_up"], after
    assert d.typed == ["hello"]
    assert "secret_field" not in out.resolved and "typed" not in out.resolved


# =====================================================================================================
# B-3 (f): the overview-frame path is guarded the same way (adversarial)
# =====================================================================================================


async def overview_rig(
    attrs: dict[str, str] | None = None, mode: str = "ok"
) -> tuple[Any, FieldPageDriver, Any]:
    d = FieldPageDriver(attrs, mode)
    ex = build(d, echo="model")
    await own_frame(ex)
    ov = await register(ex)
    return ex, d, ov


async def test_overview_card_text_refused() -> None:
    """B-3 (f)+(a): a card number typed on an overview frame is refused before the scroll or the click."""
    ex, d, ov = await overview_rig()
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", ov, x=160, y=935, text=CARD_TEXTS[0], press_enter=False))
    assert (out.status, out.reason) == ("refused", const("CARD_TEXT_REASON")), out
    assert d.dispatched(mark) == [], d.dispatched(mark)
    assert typed_anywhere(out, CARD_TEXTS[0]) == []


@pytest.mark.parametrize(
    "attrs, mode, reason",
    [
        pytest.param({"tag": "input", "type": "password"}, "ok", SECRET, id="password"),
        pytest.param({"tag": "input", "autocomplete": "cc-number"}, "ok", SECRET, id="card"),
        pytest.param(None, "raise", "FOCUS_UNKNOWN_REASON", id="read-raises"),
        pytest.param(None, "hang", "FOCUS_UNKNOWN_REASON", id="read-hangs"),
    ],
)
async def test_overview_secret_field_guarded(
    attrs: Any, mode: str, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-3 (f)+(b)-(d), adversarial (the overview path left unguarded): a ``type_at`` on a registered overview frame
    scrolls and clicks as T-0151 does, then the same guard stops it: ``failed``, nothing typed, no Enter. (T-0194:
    the read bound is monkeypatched small, so the retried hang stays fast.)"""
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", HANG_BOUND)
    ex, d, ov = await overview_rig(attrs, mode)
    reason = SECRET if reason == SECRET else const(reason)
    mark = len(d.calls)
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", ov, x=160, y=935, text=TEXT, press_enter=True)), 5.0
    )
    assert_guard_failed(d, out, mark, reason)
    if reason == SECRET:
        assert out.resolved.get("secret_field") in ("password", "card"), out.resolved


async def test_overview_missing_reader_fails_closed_in_web() -> None:
    """B-3 (f)+(d): the overview path on a driver class without ``focused_field`` in web mode fails closed too."""
    d = PageDriver(content_height=1220.0)
    ex = build(d, echo="model")
    await own_frame(ex)
    ov = await register(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", ov, x=160, y=935, text=TEXT, press_enter=True))
    assert_guard_failed(d, out, mark, const("FOCUS_UNKNOWN_REASON"))


# =====================================================================================================
# B-4: navigation and paste keys
# =====================================================================================================


@pytest.mark.parametrize("combo", ["Ctrl+T", "Ctrl+L"])
async def test_web_nav_keys_open_the_start_page(combo: str) -> None:
    """B-4: in web mode ``Ctrl+T``/``Ctrl+L`` navigate the active page to ``<start origin>/web.html`` (no key is
    pressed), settle, capture, and are ``executed`` with ``opened the start page``."""
    d = FieldDriver()
    ex = build(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo=combo))
    assert (out.status, out.action) == ("executed", OPENED), out
    assert d.dispatched(mark) == [("goto", (f"{START}/web.html",))], d.dispatched(mark)
    seq = names(d, mark)
    assert seq.index("goto") < seq.index("settle") < seq.index("screenshot"), seq
    latest = ex.latest_frame
    assert latest is not None and out.resolved.get("frame_after") == latest.phash, out.resolved


async def test_web_paste_refused() -> None:
    """B-4: ``Ctrl+V`` in web mode is refused (the headed clipboard is the user's), nothing dispatched."""
    d = FieldDriver()
    ex = build(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo="Ctrl+V"))
    assert (out.status, out.reason) == ("refused", PASTE_OFF), out
    assert d.dispatched(mark) == []


@pytest.mark.parametrize("combo", ["Ctrl+T", "Ctrl+L", "Alt+Tab", "Ctrl+W"])
async def test_loopback_chrome_keys_refused_as_base(combo: str) -> None:
    """B-4 / B-8: with ``web=False`` the four chrome shortcuts are refused exactly as BASE, even when the executor
    knows a start origin (group C may build ``--no-web`` executors with one; reviewer B)."""
    d = FieldDriver()
    ex = build(d, web=False, start_origin=START)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo=combo))
    assert (out.status, out.reason) == ("refused", "not available in the browser"), out
    assert d.dispatched(mark) == []


@pytest.mark.parametrize("combo", ["Ctrl+W", "Alt+Tab"])
async def test_web_other_chrome_keys_stay_refused(combo: str) -> None:
    """B-4 names only ``Ctrl+T``/``Ctrl+L`` (open the start page) and ``Ctrl+V`` (refused) for web mode; ``Ctrl+W``
    and ``Alt+Tab`` keep BASE's refusal there, nothing dispatched (reviewer B)."""
    d = FieldDriver()
    ex = build(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo=combo))
    assert (out.status, out.reason) == ("refused", "not available in the browser"), out
    assert d.dispatched(mark) == []


async def test_loopback_paste_dispatches_as_base() -> None:
    """B-4 / B-8: with ``web=False`` ``Ctrl+V`` is BASE's chord, a known start origin notwithstanding."""
    d = FieldDriver()
    ex = build(d, web=False, start_origin=START)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo="Ctrl+V"))
    assert (out.status, out.action) == ("executed", "pressed Ctrl+V"), out
    assert d.dispatched(mark) == [
        ("key_down", ("Control",)),
        ("key_down", ("v",)),
        ("key_up", ("v",)),
        ("key_up", ("Control",)),
    ]


# =====================================================================================================
# B-2: no user data dir in web mode
# =====================================================================================================


@pytest.mark.parametrize("kind", ["path", "str"])
def test_web_refuses_user_data_dir(tmp_path: Path, kind: str) -> None:
    """B-2: in web mode any ``user_data_dir`` is a ``ValueError``, even a fresh temporary one the loopback fence
    accepts (premise: ``web=False`` accepts it)."""
    profile = tmp_path / "profile"
    profile.mkdir()
    given: Any = profile if kind == "path" else os.fspath(profile)
    new_executor(now_ms=lambda: CLOCK_MS, user_data_dir=given, web=False)
    with pytest.raises(ValueError):
        new_executor(now_ms=lambda: CLOCK_MS, user_data_dir=given, web=True, start_origin=START)


# =====================================================================================================
# B-6: hosts per executed action
# =====================================================================================================


def host_events(*hosts: str) -> list[dict[str, Any]]:
    return [{"kind": "host", "host": h} for h in hosts]


async def test_hosts_unique_per_action() -> None:
    """B-6: an executed action's ``resolved["hosts"]`` is the unique hosts tunnelled since the previous action (a
    preconnect repeats a host), host only; the next action starts afresh."""
    d = FieldDriver()
    ex = build(d)
    await own_frame(ex)
    d.events = host_events("shop.example", "cdn.example", "shop.example")
    out = await ex.execute(inv("scroll", None, direction="down"))
    assert out.status == "executed", out
    hosts = out.resolved.get("hosts")
    assert isinstance(hosts, list) and sorted(hosts) == ["cdn.example", "shop.example"], (
        out.resolved
    )
    assert "hosts_more" not in out.resolved
    d.events = host_events("pay.example")
    out = await ex.execute(inv("scroll", None, direction="up"))
    assert out.resolved.get("hosts") == ["pay.example"], out.resolved


async def test_loopback_resolved_has_no_hosts_keys() -> None:
    """B-6 / B-8 (the hosts report too wide): with ``web=False`` an executed action's ``resolved`` has neither
    ``hosts`` nor ``hosts_more``, as BASE, even when the driver reports host events (40 of them, past the cap)."""
    d = FieldDriver()
    ex = build(d, web=False)
    await own_frame(ex)
    d.events = host_events(*(f"h{i:02d}.example" for i in range(40)))
    out = await ex.execute(inv("scroll", None, direction="down"))
    assert out.status == "executed", out
    assert "hosts" not in out.resolved and "hosts_more" not in out.resolved, out.resolved
    d.events = []
    out = await ex.execute(inv("scroll", None, direction="up"))
    assert "hosts" not in out.resolved and "hosts_more" not in out.resolved, out.resolved


async def test_hosts_capped_at_32() -> None:
    """B-6: at most 32 hosts, ``hosts_more`` counts the rest (40 unique, each twice: 32 listed, 8 more)."""
    d = FieldDriver()
    ex = build(d)
    await own_frame(ex)
    all_hosts = [f"h{i:02d}.example" for i in range(40)]
    d.events = host_events(*(all_hosts + all_hosts))
    out = await ex.execute(inv("scroll", None, direction="down"))
    hosts = out.resolved.get("hosts")
    assert isinstance(hosts, list) and len(hosts) == 32 and len(set(hosts)) == 32, out.resolved
    assert set(hosts) <= set(all_hosts)
    assert out.resolved.get("hosts_more") == 8, out.resolved


async def test_hosts_exactly_32_has_no_more() -> None:
    """B-6 cap boundary (reviewer B): exactly 32 unique hosts are all listed and there is no ``hosts_more``; 33 give
    32 and ``hosts_more: 1``."""
    d = FieldDriver()
    ex = build(d)
    await own_frame(ex)
    hosts32 = [f"h{i:02d}.example" for i in range(32)]
    d.events = host_events(*hosts32)
    out = await ex.execute(inv("scroll", None, direction="down"))
    assert sorted(out.resolved.get("hosts") or []) == hosts32, out.resolved
    assert "hosts_more" not in out.resolved, out.resolved
    d.events = host_events(*hosts32, "h32.example")
    out = await ex.execute(inv("scroll", None, direction="up"))
    assert len(out.resolved.get("hosts") or []) == 32 and out.resolved.get("hosts_more") == 1, (
        out.resolved
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://shop.example/",
        "https://10.0.0.5/",
        "https://127.0.0.1:9/",
        "https://2130706433/",
        "https://shop.example:8443/",
    ],
)
async def test_web_open_refuses_non_public_https(url: str) -> None:
    """``open()`` in web mode (A-2 via ``fence.web_url_allowed``; reviewer B): http, a private IP literal, loopback
    on another port, a numeric loopback host and a non-443 port raise ``ValueError`` and navigate nowhere. The
    origin allow-list is set as the live session sets it (T-0111g); without one BASE accepts any loopback URL."""
    d = FieldDriver()
    ex = build(d, allowed_origins=frozenset({START}))
    await own_frame(ex)
    mark = len(d.calls)
    with pytest.raises(ValueError):
        await ex.open(url)
    assert "goto" not in names(d, mark), d.calls[mark:]


@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
async def test_open_public_https_only_in_web(web: bool) -> None:
    """``open()`` takes a public https URL in web mode and navigates to it; with ``web=False`` the same URL is BASE's
    ``ValueError`` (the widening is web mode's only). The start origin opens in both."""
    d = FieldDriver()
    ex = build(d, web=web, start_origin=START, allowed_origins=frozenset({START}))
    await own_frame(ex)
    mark = len(d.calls)
    if web:
        await ex.open("https://shop.example/a")
        assert ("goto", ("https://shop.example/a",)) in d.dispatched(mark), d.calls[mark:]
    else:
        with pytest.raises(ValueError):
            await ex.open("https://shop.example/a")
        assert "goto" not in names(d, mark)
    await ex.open(f"{START}/web.html")
    assert ("goto", (f"{START}/web.html",)) in d.dispatched(mark), d.calls[mark:]


async def test_guarded_outcome_keeps_events_but_never_text() -> None:
    """B-6 / B-3 (g): a guarded ``type_at`` carries its reason and ``secret_field`` kind; no value anywhere in it holds
    the typed text, and its other keys are host-only facts (no URL path, query or fragment)."""
    d = FieldDriver({"tag": "input", "type": "password"})
    ex = build(d)
    f = await own_frame(ex)
    d.events = host_events("login.example")
    out = await ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=False))
    assert out.reason == SECRET and out.resolved.get("secret_field") == "password", out
    assert out.resolved.get("hosts") == ["login.example"], (
        f"B-6 (reading, reviewer B): a guarded outcome still reports the hosts tunnelled since the last action: {out}"
    )
    blob = json.dumps(out.resolved, default=str)
    assert TEXT not in blob and "/" not in blob.replace('"', ""), blob
