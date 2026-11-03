"""T-0194: the focus check before typing gets 2 s and one retry (docs/tasks/T-0194.md, criteria 1-8), over the fake
drivers: ``BrowserToolExecutor`` without a browser.

**What these rows pin** (the task's words; readings marked *reading* are the test-author's, listed in the report):

- Criterion 1. ``executor.FOCUS_TIMEOUT_S == 2.0``, ``2 * FOCUS_TIMEOUT_S <= ACTION_TIMEOUT_S / 2`` at
  ``ACTION_TIMEOUT_S == 8.0`` (the task: "do not raise ``ACTION_TIMEOUT_S`` here").
- Criterion 2. A first ``focused_field()`` read that times out or raises is followed by exactly one more read with no
  driver call of any kind in between (no second click, key, ``settle`` or screenshot); the second answer is judged as a
  first answer is: a single-line text input is selected (T-0189) and replaced.
- Criterion 3 (adversarial). A second answer naming a password or card field is refused as T-0160 B-3 refuses it, on
  the own-frame and the overview-frame path.
- Criterion 4. Two failed reads (any order of timeout and raise) fail closed in web mode, bounded by
  ``2 * FOCUS_TIMEOUT_S`` plus slack; ``None``, a non-mapping, an unreadable element and a driver without
  ``focused_field`` are not retried.
- Criterion 5. When the first read answers, one read, BASE's dispatch, action and ``resolved`` (hand-written below).
- Criterion 6. A web-mode ``FOCUS_UNKNOWN`` outcome gains exactly ``resolved["focus_check"]``, one of six literals,
  the **last** read's kind; ``action`` and ``reason`` are BASE's (hand-written below); no exception message, URL or
  page text in any value.
- Criterion 7. Outside web mode two failed reads still type at the caret as BASE; the retry applies (two reads).
- Criterion 8 (adversarial). A cancellation during either read propagates: no further read, nothing typed; under a
  small ``ACTION_TIMEOUT_S`` the outer bound wins over the 2 s read bound.

**Fakes.** ``ScriptedFieldDriver`` / ``ScriptedFieldPageDriver`` (over T-0160's ``FieldDriver`` /
``FieldPageDriver``) pop one scripted answer per ``focused_field`` call: ``"hang"`` (never returns until cancelled),
``"raise"`` (``DriverBoom`` whose message carries a URL and page-like text, so a leak is visible), ``"none"``,
``"notmapping"``, or an attribute dict. A read past the script hangs (it is counted, and the rows count reads). The
base fakes' sentinel turns any driver name outside the seam into a named failure.

**Bounds.** Rows whose reads hang monkeypatch ``executor.FOCUS_TIMEOUT_S`` to 0.05 s (the task: "the executor reads
the module constant at call time"); the cancel rows run the real 2.0 s bound.

**Expected values by hand** (viewport 1280x800): ``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``;
``TEXT`` is 15 characters. BASE's ``FOCUS_UNKNOWN`` outcome for a web-mode ``type_at`` at (500, 500) with Enter:
``action = "failed type_at: " + reason``, ``reason = "could not check the field before typing; nothing was typed"``,
``resolved`` keys ``x_px, y_px, press_enter, typed, frame_after, hosts`` (the plan's keys without ``text``, then
T-0160 B-3 (c)'s ``typed: 0``, ``frame_after`` and B-6's ``hosts``).

**State.** Rows that need the feature were written ``pending(task="T-0194")`` and activated for ``/implement
T-0194``. Rows that hold at BASE (criteria 5, 8's first-read cancel and outer-bound rows, the loopback ``None`` row)
are active from the start: a strict-xfail marker on them would XPASS. They kill the reviewer's mutants (retry on
``None``, catch ``BaseException``).

Regime: CPU over fakes, no browser, no socket, n = 1, deterministic. BASE = 0b50f3a8 (611068bb's executor).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.contracts.toolcall import ToolOutcome
from tests.unit.test_computer_executor import DriverBoom, FakeDriver, inv, own_frame
from tests.unit.test_computer_web_executor import (
    SECRET,
    TEXT_FIELD,
    FieldDriver,
    FieldPageDriver,
    assert_guard_failed,
    build,
    const,
    names,
)
from tests.unit.test_t0151_executor import register
from tests.unit.test_t0189_type_replaces import real_attrs

TEXT = "zebra-quokka-17"
#: T-0160 B-3 (d)'s reason and the failed action at BASE 0b50f3a8, by hand (criterion 6: byte-identical)
FOCUS_UNKNOWN = "could not check the field before typing; nothing was typed"
FOCUS_UNKNOWN_ACTION = "failed type_at: could not check the field before typing; nothing was typed"
#: BASE's resolved keys on a web-mode FOCUS_UNKNOWN outcome of type_at (module docstring)
BASE_UNKNOWN_KEYS = frozenset({"x_px", "y_px", "press_enter", "typed", "frame_after", "hosts"})
#: re-pinned for T-0210 A1 (H-T210-1): ``"body"`` is the seventh literal (a plain ``body`` focus in web mode)
FOCUS_CHECKS = frozenset(
    {"missing", "timeout", "error", "none", "not_mapping", "unreadable", "body"}
)
SMALL = 0.05
#: a raise whose message carries a URL, a site name and an attribute-like value: none may reach ``resolved``
LEAK = "focus read failed on https://www.amazon.example/s?k=leak-marker-91 value=leak-attr-73"
LEAK_NEEDLES = ("leak-marker-91", "leak-attr-73", "amazon", "http", "focus read failed")

SELECT_ALL = [
    ("key_down", ("Control",)),
    ("key_down", ("a",)),
    ("key_up", ("a",)),
    ("key_up", ("Control",)),
]
ENTER = [("key_down", ("Enter",)), ("key_up", ("Enter",))]
CLICK = [("mouse_move", (640, 400)), ("mouse_down", ()), ("mouse_up", ())]
SEQ_NAMES = frozenset(
    {"mouse_move", "mouse_down", "mouse_up", "key_down", "key_up", "type_text", "scroll_by", "goto"}
    | {"focused_field"}
)
ACTION_ENTER = "clicked at (640, 400) and typed 15 characters and pressed Enter"
PASSWORD = {"tag": "input", "type": "password", "name": "pw"}
CARD = {"tag": "input", "type": "text", "autocomplete": "cc-number"}


async def _answer(d: Any) -> Any:
    d.reads += 1
    if not d.answers:
        await asyncio.Event().wait()  # a read nobody scripted: hangs until the bound cuts it
    a = d.answers.pop(0)
    if a == "hang":
        await asyncio.Event().wait()
    if a == "raise":
        raise DriverBoom(LEAK)
    if a == "none":
        return None
    if a == "notmapping":
        return [("tag", "input"), ("type", "text")]
    if d.suspend:
        await asyncio.sleep(d.suspend)  # a real read suspends before it answers (review finding 1)
    return dict(a)


class ScriptedFieldDriver(FieldDriver):
    """``focused_field`` pops one scripted answer per call (module docstring)."""

    def __init__(self, *answers: Any) -> None:
        super().__init__()
        self.answers = list(answers)
        self.reads = 0
        #: seconds a scripted attribute answer suspends before it returns (0: answers without suspending)
        self.suspend = 0.0

    async def focused_field(self) -> Any:
        self._enter("focused_field")
        return await _answer(self)


class ScriptedFieldPageDriver(FieldPageDriver):
    """T-0151's overview fake with the same scripted ``focused_field``."""

    def __init__(self, *answers: Any) -> None:
        super().__init__()
        self.answers = list(answers)
        self.reads = 0
        #: seconds a scripted attribute answer suspends before it returns (0: answers without suspending)
        self.suspend = 0.0

    async def focused_field(self) -> Any:  # type: ignore[override]
        self._enter("focused_field")
        return await _answer(self)


@pytest.fixture
def small_bound(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", SMALL)
    return SMALL


def seq(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in d.calls[since:] if c[0] in SEQ_NAMES]


def after_mouse_up(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    got = d.dispatched(since)
    ns = [c[0] for c in got]
    assert "mouse_up" in ns, f"premise: the call clicked: {got}"
    return got[ns.index("mouse_up") + 1 :]


def reads(d: FakeDriver, since: int) -> int:
    return names(d, since).count("focused_field")


async def type_at(d: FakeDriver, *, web: bool = True) -> tuple[Any, ToolOutcome, int, float]:
    ex = build(d, web=web)
    f = await own_frame(ex)
    mark = len(d.calls)
    t0 = time.monotonic()
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=True)), 10.0
    )
    assert isinstance(out, ToolOutcome), out
    return ex, out, mark, time.monotonic() - t0


def kind_of(answer: str) -> str:
    return {"hang": "timeout", "raise": "error"}[answer]


# =====================================================================================================
# criterion 1: the budget
# =====================================================================================================


def test_budget_constants() -> None:
    """Criterion 1: the read bound is 2.0 s and two reads fit in half the action bound (4.0 <= 4.0 at 8.0 s); the
    action bound itself is not raised."""
    assert executor_mod.ACTION_TIMEOUT_S == 8.0, executor_mod.ACTION_TIMEOUT_S
    assert const("FOCUS_TIMEOUT_S") == 2.0, (
        f"T-0194 criterion 1: FOCUS_TIMEOUT_S is {executor_mod.FOCUS_TIMEOUT_S}"
    )
    assert 2 * executor_mod.FOCUS_TIMEOUT_S <= executor_mod.ACTION_TIMEOUT_S / 2


# =====================================================================================================
# criterion 2: one retry after a timeout or a raise
# =====================================================================================================


@pytest.mark.parametrize("first", ["hang", "raise"])
async def test_retry_rescues_text_field(first: str, small_bound: float) -> None:
    """Criterion 2: first read hangs/raises, second answers a text input: ``executed`` with BASE's action; after the
    one click, two reads back to back (no driver call at all between them), then ``Ctrl+A``, the text, Enter;
    ``replaced`` is true and there is no ``focus_check``."""
    d = ScriptedFieldDriver(first, {"tag": "input", "type": "text"})
    _, out, mark, _ = await type_at(d)
    got = seq(d, mark)
    want = [*CLICK, ("focused_field", ()), ("focused_field", ()), *SELECT_ALL]
    want += [("type_text", (TEXT,)), *ENTER]
    assert got == want, f"T-0194 criterion 2: want {want}, got {got}"
    all_calls = names(d, mark)
    i = all_calls.index("focused_field")
    assert all_calls[i + 1] == "focused_field", (
        f"T-0194 criterion 2: a driver call between the two reads: {all_calls}"
    )
    assert out.status == "executed" and out.action == ACTION_ENTER, out
    assert out.resolved.get("replaced") is True, out.resolved
    assert "focus_check" not in out.resolved, out.resolved
    assert d.typed == [TEXT] and d.held == set(), (d.typed, d.held)


@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
@pytest.mark.parametrize("second", ["password", "text"])
async def test_retry_has_its_own_bound(
    second: str, web: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 2 ("reads once more with the same bound"), adversarial (review finding 1: one deadline shared by
    both reads, so the retry starts already expired): the first read hangs past ``FOCUS_TIMEOUT_S`` (0.3 s here), the
    second suspends 0.02 s before answering, as a real read does. With its own bound the second answer is judged:
    a password field is refused (both modes; outside web mode a shared deadline would type into it at the caret), a
    text input is selected and replaced. Two reads; no ``focus_check``."""
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", 0.3)
    d = ScriptedFieldDriver("hang", PASSWORD if second == "password" else TEXT_FIELD)
    d.suspend = 0.02
    _, out, mark, _ = await type_at(d, web=web)
    assert reads(d, mark) == 2, names(d, mark)
    if second == "password":
        assert_guard_failed(d, out, mark, SECRET)
        assert out.resolved.get("secret_field") == "password", out.resolved
        assert d.typed == [], f"T-0194 criterion 2/3: typed into a password field: {d.typed}"
    else:
        assert out.status == "executed" and out.action == ACTION_ENTER, out
        assert out.resolved.get("replaced") is True, out.resolved
        assert d.typed == [TEXT], d.typed
    assert "focus_check" not in out.resolved, out.resolved


# =====================================================================================================
# criterion 3: the secret-field refusal is never weakened (adversarial)
# =====================================================================================================


async def overview_type(d: ScriptedFieldPageDriver) -> tuple[ToolOutcome, int]:
    ex = build(d, echo="model")
    await own_frame(ex)
    ov = await register(ex)
    mark = len(d.calls)
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", ov, x=160, y=935, text=TEXT, press_enter=True)), 10.0
    )
    return out, mark


@pytest.mark.parametrize("path", ["own-frame", "overview"])
@pytest.mark.parametrize(
    "kind, attrs", [("password", PASSWORD), ("card", CARD)], ids=["password", "card"]
)
@pytest.mark.parametrize("first", ["hang", "raise"])
async def test_retry_still_refuses_secret(
    first: str, kind: str, attrs: dict[str, str], path: str, small_bound: float
) -> None:
    """Criterion 3, adversarial (the retried answer waved through, or judged by less than ``fence.secret_field``):
    first read hangs/raises, second names a password or card field: ``failed`` with ``SECRET_FIELD_REASON``,
    ``secret_field`` the kind, ``typed: 0``, no keystroke after the click, the frame re-captured; two reads; no
    ``focus_check`` (the field was checked)."""
    if path == "overview":
        d: Any = ScriptedFieldPageDriver(first, attrs)
        out, mark = await overview_type(d)
    else:
        d = ScriptedFieldDriver(first, attrs)
        _, out, mark, _ = await type_at(d)
    assert_guard_failed(d, out, mark, SECRET)
    assert out.resolved.get("secret_field") == kind, out.resolved
    assert reads(d, mark) == 2, f"premise: the retry read the field: {names(d, mark)}"
    assert "focus_check" not in out.resolved, out.resolved
    assert d.typed == [], d.typed


# =====================================================================================================
# criterion 4: still fails closed; definite answers are not retried
# =====================================================================================================


@pytest.mark.parametrize(
    "first, second",
    [("hang", "hang"), ("raise", "raise"), ("hang", "raise"), ("raise", "hang")],
    ids=["hang,hang", "raise,raise", "hang,raise", "raise,hang"],
)
async def test_two_failures_fail_closed(first: str, second: str, small_bound: float) -> None:
    """Criterion 4 (+6): two failed reads in web mode: ``failed`` with ``FOCUS_UNKNOWN_REASON``, nothing typed, the
    frame re-captured (``assert_guard_failed``); exactly two reads; within ``2 * FOCUS_TIMEOUT_S + 1.5 s``;
    ``focus_check`` names the **second** read's kind."""
    d = ScriptedFieldDriver(first, second, TEXT_FIELD)
    _, out, mark, elapsed = await type_at(d)
    assert_guard_failed(d, out, mark, const("FOCUS_UNKNOWN_REASON"))
    assert reads(d, mark) == 2, f"T-0194 criterion 2/4: one retry, then stop: {names(d, mark)}"
    assert elapsed < 2 * executor_mod.FOCUS_TIMEOUT_S + 1.5, f"not bounded: {elapsed:.2f} s"
    assert out.resolved.get("focus_check") == kind_of(second), (
        f"T-0194 criterion 6: focus_check describes the last read ({second}); got {out.resolved}"
    )


@pytest.mark.parametrize(
    "first, check",
    [
        pytest.param("none", "none", id="none"),
        pytest.param("notmapping", "not_mapping", id="notmapping"),
        pytest.param(real_attrs("x-editor", opaque="true"), "unreadable", id="unreadable-opaque"),
        pytest.param({"tag": "div", "type": ""}, "unreadable", id="unreadable-div"),
        pytest.param(None, "missing", id="missing"),
    ],
)
async def test_no_retry_on_definite_answers(first: Any, check: str, small_bound: float) -> None:
    """Criterion 4 (+6): a read that answers ``None``, a non-mapping, or an element web mode cannot judge (an opaque
    custom element, a plain ``div``) is not retried, although the script's next answer is a text input; a driver class
    without ``focused_field`` is not read at all. All fail closed as BASE; ``focus_check`` names the case."""
    d: FakeDriver = FakeDriver() if first is None else ScriptedFieldDriver(first, TEXT_FIELD)
    _, out, mark, _ = await type_at(d)
    assert_guard_failed(d, out, mark, const("FOCUS_UNKNOWN_REASON"))
    assert reads(d, mark) == (0 if first is None else 1), (
        f"T-0194 criterion 4: a definite answer is not retried: {names(d, mark)}"
    )
    assert out.resolved.get("focus_check") == check, (
        f"T-0194 criterion 6: focus_check names the case ({check}); got {out.resolved}"
    )


@pytest.mark.parametrize("second", [TEXT_FIELD, PASSWORD], ids=["then-text", "then-password"])
async def test_loopback_none_not_retried(second: dict[str, str], small_bound: float) -> None:
    """Criterion 4, outside web mode (a retry on ``None``): ``None`` means nothing holds the focus, which BASE types
    into at the caret: one read, no select-all, no ``replaced``, BASE's action. Holds at BASE."""
    d = ScriptedFieldDriver("none", second)
    _, out, mark, _ = await type_at(d, web=False)
    assert reads(d, mark) == 1, names(d, mark)
    assert out.status == "executed" and out.action == ACTION_ENTER, out
    assert after_mouse_up(d, mark) == [("type_text", (TEXT,)), *ENTER], after_mouse_up(d, mark)
    assert not {"replaced", "secret_field", "typed", "focus_check"} & set(out.resolved), (
        out.resolved
    )


# =====================================================================================================
# criterion 5: one read on the success path
# =====================================================================================================


@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
async def test_success_path_unchanged(web: bool) -> None:
    """Criterion 5: the first read answers: one read; dispatch, action and ``resolved`` exactly BASE's (by hand,
    T-0189 criterion 4's literal). Real 2.0 s bound. Holds at BASE."""
    d = ScriptedFieldDriver(TEXT_FIELD, PASSWORD)
    ex, out, mark, _ = await type_at(d, web=web)
    assert seq(d, mark) == [
        *CLICK,
        ("focused_field", ()),
        *SELECT_ALL,
        ("type_text", (TEXT,)),
        *ENTER,
    ], seq(d, mark)
    assert out.status == "executed" and out.action == ACTION_ENTER, out
    want: dict[str, Any] = {
        "x_px": 640,
        "y_px": 400,
        "text": TEXT,
        "press_enter": True,
        "frame_after": ex.latest_frame.phash,
        "replaced": True,
    }
    if web:
        want["hosts"] = []
    assert out.resolved == want, out.resolved


# =====================================================================================================
# criterion 6: the kind is logged, not shown to the model
# =====================================================================================================


@pytest.mark.parametrize(
    "script",
    [("raise", "raise"), ("hang", "raise"), ("raise", "hang"), ("none",), ("notmapping",)],
    ids=["raise,raise", "hang,raise", "raise,hang", "none", "notmapping"],
)
async def test_model_facing_unchanged(script: tuple[str, ...], small_bound: float) -> None:
    """Criterion 6: ``action`` and ``reason`` are BASE's bytes (by hand); ``resolved`` is BASE's keys plus exactly
    ``focus_check``, a fixed literal; no value carries the exception's message, the URL in it, ``amazon`` or
    ``http``."""
    d = ScriptedFieldDriver(*script)
    _, out, _, _ = await type_at(d)
    assert out.status == "failed", out
    assert out.reason == FOCUS_UNKNOWN, out.reason
    assert out.action == FOCUS_UNKNOWN_ACTION, out.action
    keys = set(out.resolved)
    assert keys >= BASE_UNKNOWN_KEYS and keys - BASE_UNKNOWN_KEYS == {"focus_check"}, (
        f"T-0194 criterion 6: resolved gains exactly focus_check; got {sorted(keys)}"
    )
    assert out.resolved["focus_check"] in FOCUS_CHECKS, out.resolved
    blob = json.dumps(out.resolved, default=str).lower()
    leaked = [n for n in LEAK_NEEDLES if n in blob]
    assert leaked == [], f"T-0194 criterion 6: resolved carries {leaked}: {out.resolved}"


async def test_focus_check_only_on_focus_unknown(small_bound: float) -> None:
    """Criterion 6, "no other outcome gains it": a secret-field refusal, a rescued text field and a loopback call
    whose reads both failed carry no ``focus_check``; the web-mode unknown outcome in the same run does (premise, so
    the row cannot pass on a build that never writes the key)."""
    outs: dict[str, ToolOutcome] = {}
    for name, script, web in (
        ("unknown", ("raise", "raise"), True),
        ("secret", ("raise", PASSWORD), True),
        ("rescued", ("raise", TEXT_FIELD), True),
        ("loopback", ("raise", "raise"), False),
        ("first-secret", (CARD,), True),
    ):
        _, outs[name], _, _ = await type_at(ScriptedFieldDriver(*script), web=web)
    assert outs["unknown"].resolved.get("focus_check") == "error", outs["unknown"].resolved
    for name in ("secret", "rescued", "loopback", "first-secret"):
        assert "focus_check" not in outs[name].resolved, (name, outs[name].resolved)


# =====================================================================================================
# criterion 7: outside web mode a failed read still types as BASE
# =====================================================================================================


@pytest.mark.parametrize(
    "first, second", [("hang", "hang"), ("raise", "raise")], ids=["hang,hang", "raise,raise"]
)
async def test_loopback_retry_then_types_as_base(
    first: str, second: str, small_bound: float
) -> None:
    """Criterion 7: ``web=False``, both reads fail: two reads, then typed at the caret as BASE: ``executed``, BASE's
    action, after the click exactly ``type_text`` then Enter (no ``Control``), no ``replaced``/``typed``/
    ``secret_field``/``focus_check``."""
    d = ScriptedFieldDriver(first, second)
    _, out, mark, _ = await type_at(d, web=False)
    assert reads(d, mark) == 2, (
        f"T-0194 criterion 7: the retry applies outside web mode: {names(d, mark)}"
    )
    assert out.status == "executed" and out.action == ACTION_ENTER, out
    assert after_mouse_up(d, mark) == [("type_text", (TEXT,)), *ENTER], after_mouse_up(d, mark)
    assert not {"replaced", "typed", "secret_field", "focus_check"} & set(out.resolved), (
        out.resolved
    )


# =====================================================================================================
# criterion 8: a cancellation is not retried (adversarial)
# =====================================================================================================


async def _cancel_at_read(script: tuple[Any, ...], k: int) -> tuple[ScriptedFieldDriver, int]:
    d = ScriptedFieldDriver(*script)
    ex = build(d)
    f = await own_frame(ex)
    mark = len(d.calls)
    task = asyncio.ensure_future(
        ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=True))
    )
    deadline = time.monotonic() + 5.0
    while d.reads < k and not task.done() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert not task.done(), (
        f"T-0194 criterion 2: the call ended before read {k} was in flight: {names(d, mark)}; {task}"
    )
    assert d.reads == k, f"premise: read {k} is in flight; {d.reads} reads"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5.0)
    await asyncio.sleep(0.1)
    return d, mark


@pytest.mark.parametrize(
    "script, k",
    [
        pytest.param(("hang", TEXT_FIELD), 1, id="first-read"),
        pytest.param(("raise", "hang", TEXT_FIELD), 2, id="second-read"),
    ],
)
async def test_cancel_during_read_not_retried(script: tuple[Any, ...], k: int) -> None:
    """Criterion 8, adversarial (``except BaseException`` around the read): the call is cancelled while read ``k``
    is in flight, inside the real 2.0 s bound: ``CancelledError`` propagates, no further read is made (the script's
    next answer, a text input, is never asked for), nothing is typed and no key goes down. The first-read row holds at
    BASE; the second-read row needs the retry to exist."""
    d, mark = await _cancel_at_read(script, k)
    assert d.reads == k, f"T-0194 criterion 8: a cancelled read was retried: {names(d, mark)}"
    assert d.typed == [], d.typed
    assert not [c for c in d.dispatched(mark) if c[0] in ("key_down", "key_up", "type_text")], seq(
        d, mark
    )


async def test_outer_action_bound_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 8, variant: ``ACTION_TIMEOUT_S = 0.1`` with the real 2.0 s read bound and a first read that hangs:
    the action bound cancels the read; that cancellation is not retried (the script's next answer, a text input, is
    never asked for): ``failed``, one read, nothing typed, within 1 s. Holds at BASE."""
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 0.1)
    d = ScriptedFieldDriver("hang", TEXT_FIELD)
    _, out, mark, elapsed = await type_at(d)
    assert out.status == "failed", out
    assert reads(d, mark) == 1, (
        f"T-0194 criterion 8: the outer cancellation was retried: {names(d, mark)}"
    )
    assert d.typed == [] and elapsed < 1.0, (d.typed, elapsed)
