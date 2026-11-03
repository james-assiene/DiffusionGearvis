"""T-0210 group A: in web mode a ``type_at`` whose focus click leaves the focus on a plain ``body`` is refused, not typed
(docs/tasks/T-0210.md, A1-A8), over the fake drivers: ``BrowserToolExecutor`` without a browser.

**What these rows pin** (the task's words; readings marked *reading* are the test-author's, listed in the report):

- A1. Web mode, the post-click focus read gives ``tag`` ``body`` (trimmed, casefolded) and ``contenteditable`` not
  ``"true"``: ``failed``, ``FOCUS_UNKNOWN_REASON``, action ``"failed type_at: " + reason`` (by hand, T-0194's BASE
  bytes); ``resolved`` is BASE's FOCUS_UNKNOWN keys (``x_px, y_px, press_enter, typed, frame_after, hosts``) plus
  ``focus_check == "body"`` (H-T210-1), ``typed: 0``, no ``text``; nothing dispatched after the click's ``mouse_up``
  (no key, no text, no ``Ctrl+A``, no Enter); the frame re-captured. ``FocusCheck`` gains ``"body"`` only. The
  overview-frame path is guarded alike. *Reading:* "ends exactly as an unreadable focus does today" includes T-0194's
  no-retry rule for definite answers (one read), and a ``body`` answer after a retried failed read is judged as a
  first answer (``"body"``, not ``"timeout"``/``"error"``).
- A2. ``body`` with ``contenteditable == "true"`` (designMode), ``input``/``textarea``/``select`` and a
  contenteditable ``div``/``p`` keep BASE's calls, order, action and ``resolved`` (hand-written, T-0189 criterion 4's
  literal).
- A3. Outside web mode a ``body`` focus is typed into as BASE.
- A4. ``press_key`` never reads the focus: every combo, both modes, a ``body`` focus and a text-input focus give the
  same driver calls and the same outcome (a differential oracle: the focus cannot matter), and ``Delete`` is
  hand-pinned.
- A5. ``click_at`` with a ``body`` focus executes as BASE (no focus read, BASE's action).
- A7. The module docstring's guard paragraph names the ``body`` rule; the Replace paragraph drops BASE's plain
  ``body`` from its caret list. *Reading:* the task fixes no wording, so only those two facts are pinned.
- A8. Frozen paths over BASE..TIP (two literals; ``TIP_IMPL`` set by the test-author after the implementation commit).

**Fakes.** T-0160's ``FieldDriver``/``FieldPageDriver`` and T-0194's ``ScriptedFieldDriver``; the attribute dicts are
``runtime._FOCUSED_JS``'s full key set (T-0189's ``real_attrs``).

**State.** Rows that needed A carried ``pending(task="T-0210")`` until activation for ``/implement T-0210`` (A,
implementation tip 3b12ae5c). Rows that hold at BASE (A2-A5, the A8 power premise) were
active from the start (a strict-xfail marker on them would XPASS); they kill the reviewer's mutations (1) the rule
outside web mode and (2) the contenteditable exemption dropped.

**Expected values by hand** (viewport 1280x800): ``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``;
``TEXT`` is 15 characters.

Regime: CPU over fakes, no browser, no socket, n = 1, deterministic. BASE = 4915fbe8.
"""

from __future__ import annotations

import asyncio
import typing
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.contracts.toolcall import ToolOutcome
from tests import _git_range as gr
from tests.unit.test_computer_executor import FakeDriver, inv, own_frame
from tests.unit.test_computer_web_executor import (
    TEXT_FIELD,
    FieldDriver,
    assert_guard_failed,
    build,
    names,
    overview_rig,
)
from tests.unit.test_t0189_type_replaces import real_attrs
from tests.unit.test_t0194_focus_retry import ScriptedFieldDriver

#: the literal commit T-0210 A branched from (main, 2026-10-30)
BASE_SHA = "4915fbe854ab89197c07e4b622685d9759cd61b0"
#: the implementation commit of T-0210 A (A8), set by the test-author at activation
TIP_IMPL: str | None = "3b12ae5c6bcceb373637793a5a041a2622e2fe8b"
#: A8's range base: the implementation branch merged main mid-way (8d6e41e1, bringing T-0209 A1's pages/rtx/**),
#: so the range starts at merge-base(main, TIP_IMPL) = feab84c1 (.claude/rules/tests.md: BASE = merge-base M^1 M^2,
#: not the declared fork point), not at BASE_SHA. feab84c1..TIP_IMPL still holds every A commit (cda835bd, 3b12ae5c).
RANGE_BASE = "feab84c1f40fb7447980d53324a336ed8d99dc6c"
#: A8's frozen paths
FROZEN = (
    "local_astra/computer/runtime.py",
    "local_astra/computer/fence.py",
    "local_astra/computer/egress_proxy.py",
    "local_astra/toolcall",
    "local_astra/conductor",
    "local_astra/contracts",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/pages",
)

TEXT = "zebra-quokka-17"
#: T-0160 B-3 (d)'s reason and failed action, by hand (T-0194 criterion 6's BASE bytes)
FOCUS_UNKNOWN = "could not check the field before typing; nothing was typed"
FOCUS_UNKNOWN_ACTION = "failed type_at: could not check the field before typing; nothing was typed"
#: BASE's resolved keys on a web-mode FOCUS_UNKNOWN own-frame type_at, plus T-0194's focus_check
REFUSED_KEYS = {"x_px", "y_px", "press_enter", "typed", "frame_after", "hosts", "focus_check"}
#: T-0194's six literals at BASE
BASE_CHECKS = {"missing", "timeout", "error", "none", "not_mapping", "unreadable"}
ACTION_ENTER = "clicked at (640, 400) and typed 15 characters and pressed Enter"
ACTION_NO_ENTER = "clicked at (640, 400) and typed 15 characters"
CLICK = [("mouse_move", (640, 400)), ("mouse_down", ()), ("mouse_up", ())]
SELECT_ALL = [
    ("key_down", ("Control",)),
    ("key_down", ("a",)),
    ("key_up", ("a",)),
    ("key_up", ("Control",)),
]
ENTER = [("key_down", ("Enter",)), ("key_up", ("Enter",))]
SEQ_NAMES = frozenset(
    {"mouse_move", "mouse_down", "mouse_up", "key_down", "key_up", "type_text", "scroll_by", "goto"}
    | {"focused_field"}
)
SMALL = 0.05

#: A1: a plain body, as ``_FOCUSED_JS`` reports it and in the spellings A1 names (trimmed, casefolded)
PLAIN_BODIES = [
    pytest.param(real_attrs("body"), id="body-real"),
    pytest.param({"tag": "BODY "}, id="BODY-space"),
    pytest.param({"tag": "body", "contenteditable": ""}, id="body-ce-empty"),
    pytest.param({"tag": " Body", "contenteditable": "false"}, id="Body-ce-false"),
]
#: A2: a designMode body (``isContentEditable`` true): still typed, never selected
EDITABLE_BODY = real_attrs("body", "", "true")


def seq(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in d.calls[since:] if c[0] in SEQ_NAMES]


def after_mouse_up(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    got = d.dispatched(since)
    ns = [c[0] for c in got]
    assert "mouse_up" in ns, f"premise: the call clicked: {got}"
    return got[ns.index("mouse_up") + 1 :]


async def type_at(
    d: FakeDriver, *, web: bool = True, enter: bool = True
) -> tuple[Any, ToolOutcome, int, Any]:
    ex = build(d, web=web)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", f, x=500, y=500, text=TEXT, press_enter=enter)), 10.0
    )
    assert isinstance(out, ToolOutcome), out
    return ex, out, mark, f


def assert_body_refused(ex: Any, d: FakeDriver, out: ToolOutcome, mark: int, f: Any) -> None:
    """A1's whole outcome for an own-frame ``type_at`` at (500, 500)."""
    assert (out.status, out.reason) == ("failed", FOCUS_UNKNOWN), (
        f"T-0210 A1: a plain body focus in web mode is refused as an unreadable focus; got {out}"
    )
    assert out.action == FOCUS_UNKNOWN_ACTION, out.action
    assert out.resolved.get("focus_check") == "body", (
        f"T-0210 A1 / H-T210-1: focus_check names the body case; got {out.resolved}"
    )
    assert set(out.resolved) == REFUSED_KEYS, (
        f"T-0210 A1: resolved is BASE's FOCUS_UNKNOWN keys plus focus_check; got {sorted(out.resolved)}"
    )
    assert out.resolved["typed"] == 0 and "text" not in out.resolved, out.resolved
    assert (out.resolved["x_px"], out.resolved["y_px"]) == (640, 400), out.resolved
    assert out.resolved["hosts"] == [], out.resolved
    assert after_mouse_up(d, mark) == [], (
        f"T-0210 A1: nothing is dispatched after the focus click: {after_mouse_up(d, mark)}"
    )
    assert d.typed == [] and d.held == set(), (d.typed, d.held)
    assert_guard_failed(d, out, mark, FOCUS_UNKNOWN)
    latest = ex.latest_frame
    assert latest is not None and latest.phash != f.phash, "T-0210 A1: the frame is re-captured"
    assert out.resolved["frame_after"] == latest.phash, out.resolved


# =====================================================================================================
# A1: refusal
# =====================================================================================================


@pytest.mark.parametrize("enter", [True, False], ids=["enter", "no-enter"])
@pytest.mark.parametrize("attrs", PLAIN_BODIES)
async def test_web_body_refused(attrs: dict[str, str], enter: bool) -> None:
    """A1, adversarial (the named failure: every character of ``type_at`` goes to the page's document ``keydown``
    shortcuts after a click on a canvas that cannot take focus): ``failed`` with ``FOCUS_UNKNOWN_REASON``,
    ``focus_check == "body"``, ``typed: 0``, no ``text``, nothing after ``mouse_up``, the frame re-captured, one read."""
    d = FieldDriver(attrs)
    ex, out, mark, f = await type_at(d, enter=enter)
    assert_body_refused(ex, d, out, mark, f)
    assert out.resolved["press_enter"] is enter, out.resolved
    assert names(d, mark).count("focused_field") == 1, (
        f"T-0210 A1 (as an unreadable focus, T-0194 criterion 4): a definite answer is read once: {names(d, mark)}"
    )


async def test_web_body_not_retried() -> None:
    """A1 *reading* (ends exactly as an unreadable focus does: a definite answer is not retried, T-0194 criterion 4):
    a ``body`` answer followed in the script by a text input is read once and refused."""
    d = ScriptedFieldDriver(real_attrs("body"), TEXT_FIELD)
    ex, out, mark, f = await type_at(d)
    assert d.reads == 1, f"a body answer was retried: {names(d, mark)}"
    assert_body_refused(ex, d, out, mark, f)


@pytest.mark.parametrize("first", ["hang", "raise"])
async def test_body_after_retry(first: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1, adversarial (a body hidden behind T-0194's retry): the first read times out or raises, the retry answers a
    plain ``body``: refused with ``focus_check == "body"`` (the second answer is judged as a first one), not
    ``"timeout"``/``"error"``, two reads, nothing typed."""
    monkeypatch.setattr(executor_mod, "FOCUS_TIMEOUT_S", SMALL)
    d = ScriptedFieldDriver(first, real_attrs("body"))
    ex, out, mark, f = await type_at(d)
    assert d.reads == 2, names(d, mark)
    assert_body_refused(ex, d, out, mark, f)


async def test_overview_type_on_body_refused() -> None:
    """A1 on T-0151's overview-frame path (same guard): it scrolls and clicks as T-0151 does, then refuses with
    ``focus_check == "body"``; nothing typed, no Enter."""
    ex, d, ov = await overview_rig(real_attrs("body"))
    mark = len(d.calls)
    out = await asyncio.wait_for(
        ex.execute(inv("type_at", ov, x=160, y=935, text=TEXT, press_enter=True)), 5.0
    )
    assert_guard_failed(d, out, mark, FOCUS_UNKNOWN)
    assert out.action == FOCUS_UNKNOWN_ACTION, out.action
    assert out.resolved.get("focus_check") == "body", out.resolved
    assert "scroll_by" in names(d, mark), "premise: the overview path scrolled before the click"
    assert d.typed == [] and d.held == set()


def test_focus_check_gains_body_only() -> None:
    """A1: ``FocusCheck`` is T-0194's six literals plus ``"body"``, nothing else."""
    got = set(typing.get_args(executor_mod.FocusCheck))
    assert got == BASE_CHECKS | {"body"}, f"T-0210 A1: FocusCheck gains 'body' only; got {got}"


# =====================================================================================================
# A2: still typed (guard too wide)
# =====================================================================================================


@pytest.mark.parametrize("enter", [True, False], ids=["enter", "no-enter"])
async def test_web_body_contenteditable_typed(enter: bool) -> None:
    """A2 (mutation 2, the contenteditable exemption dropped): a designMode ``body`` (``contenteditable: "true"``)
    in web mode is typed at the caret exactly as BASE: click, one read, the text once, Enter when asked; BASE's action
    and ``resolved``; no select-all, no ``replaced``. Holds at BASE."""
    d = FieldDriver(EDITABLE_BODY)
    ex, out, mark, _ = await type_at(d, enter=enter)
    assert out.status == "executed", out
    assert out.action == (ACTION_ENTER if enter else ACTION_NO_ENTER), out.action
    assert seq(d, mark) == [*CLICK, ("focused_field", ()), ("type_text", (TEXT,))] + (
        ENTER if enter else []
    ), seq(d, mark)
    assert out.resolved == {
        "x_px": 640,
        "y_px": 400,
        "text": TEXT,
        "press_enter": enter,
        "frame_after": ex.latest_frame.phash,
        "hosts": [],
    }, out.resolved


TYPABLE = [
    pytest.param(real_attrs("input", "text", name="q"), True, id="input-text"),
    pytest.param(real_attrs("input", "search"), True, id="input-search"),
    pytest.param(real_attrs("textarea", "textarea"), False, id="textarea"),
    pytest.param(real_attrs("select", "select-one"), False, id="select"),
    pytest.param(real_attrs("div", "", "true"), False, id="contenteditable-div"),
    pytest.param(real_attrs("p", "", "true", id="doc"), False, id="contenteditable-p"),
]


@pytest.mark.parametrize("attrs, replaced", TYPABLE)
async def test_typable_unchanged(attrs: dict[str, str], replaced: bool) -> None:
    """A2: ``input``, ``textarea``, ``select`` and a contenteditable ``div``/``p`` keep BASE's driver calls, order,
    action and ``resolved`` in web mode (a single-line input with T-0189's select-all and ``replaced: True``; the
    others typed at the caret). Hand-written expectations. Holds at BASE."""
    d = FieldDriver(attrs)
    ex, out, mark, _ = await type_at(d)
    assert out.status == "executed" and out.action == ACTION_ENTER, out
    want_seq = [*CLICK, ("focused_field", ())] + (SELECT_ALL if replaced else [])
    want_seq += [("type_text", (TEXT,)), *ENTER]
    assert seq(d, mark) == want_seq, seq(d, mark)
    want: dict[str, Any] = {
        "x_px": 640,
        "y_px": 400,
        "text": TEXT,
        "press_enter": True,
        "frame_after": ex.latest_frame.phash,
        "hosts": [],
    }
    if replaced:
        want["replaced"] = True
    assert out.resolved == want, out.resolved


# =====================================================================================================
# A3: outside web mode
# =====================================================================================================


@pytest.mark.parametrize("attrs", [*PLAIN_BODIES, pytest.param(EDITABLE_BODY, id="body-ce-true")])
async def test_loopback_body_typed(attrs: dict[str, str]) -> None:
    """A3 (mutation 1, the rule applied outside web mode): ``web=False`` with a driver that has ``focused_field``
    types into a ``body`` focus as BASE: click, one read, the text, Enter; BASE's action; ``resolved`` without
    ``typed``, ``focus_check`` or ``hosts``. Holds at BASE."""
    d = FieldDriver(attrs)
    ex, out, mark, _ = await type_at(d, web=False)
    assert out.status == "executed" and out.action == ACTION_ENTER, out
    assert seq(d, mark) == [*CLICK, ("focused_field", ()), ("type_text", (TEXT,)), *ENTER], seq(
        d, mark
    )
    assert out.resolved == {
        "x_px": 640,
        "y_px": 400,
        "text": TEXT,
        "press_enter": True,
        "frame_after": ex.latest_frame.phash,
    }, out.resolved


# =====================================================================================================
# A4: press_key unchanged
# =====================================================================================================


def _all_combos() -> list[str]:
    combos = set(executor_mod.COMBOS) | set(executor_mod.START_PAGE_COMBOS) | {"Ctrl+V"}
    return sorted(combos)


async def _press(
    attrs: dict[str, str], combo: str, web: bool
) -> tuple[FieldDriver, ToolOutcome, int, dict[str, Any]]:
    """``press_key combo`` with the given focus; also returns ``resolved`` without ``frame_after`` (a per-executor
    salted phash), after checking ``frame_after`` is that executor's latest frame when present."""
    d = FieldDriver(attrs)
    ex = build(d, web=web)
    await own_frame(ex)
    mark = len(d.calls)
    out = await asyncio.wait_for(ex.execute(inv("press_key", None, combo=combo)), 10.0)
    assert isinstance(out, ToolOutcome), out
    rest = dict(out.resolved)
    if "frame_after" in rest:
        assert rest.pop("frame_after") == ex.latest_frame.phash, out.resolved
    return d, out, mark, rest


@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
@pytest.mark.parametrize("combo", _all_combos())
async def test_press_key_never_reads_focus(combo: str, web: bool) -> None:
    """A4 (H-T210-2): ``press_key`` with a plain ``body`` focus makes the same driver calls and gives the same outcome
    as with a text-input focus (the focus is not an input to ``press_key``), and never reads the focus. Holds at BASE;
    kills a ``body`` rule leaking into ``press_key``."""
    d_body, out_body, m_body, r_body = await _press(real_attrs("body"), combo, web)
    d_text, out_text, m_text, r_text = await _press(TEXT_FIELD, combo, web)
    assert "focused_field" not in names(d_body, m_body), names(d_body, m_body)
    assert d_body.calls[m_body:] == d_text.calls[m_text:], (
        d_body.calls[m_body:],
        d_text.calls[m_text:],
    )
    assert (out_body.status, out_body.action, out_body.reason, r_body) == (
        out_text.status,
        out_text.action,
        out_text.reason,
        r_text,
    )


@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
async def test_press_key_delete_on_body_by_hand(web: bool) -> None:
    """A4 / A6e's unit partner: ``press_key Delete`` with a ``body`` focus is BASE's ``pressed Delete`` with exactly
    one Delete down and up. Holds at BASE."""
    d, out, mark, _ = await _press(real_attrs("body"), "Delete", web)
    assert (out.status, out.action) == ("executed", "pressed Delete"), out
    assert d.dispatched(mark) == [("key_down", ("Delete",)), ("key_up", ("Delete",))], d.dispatched(
        mark
    )


# =====================================================================================================
# A5: the click is unchanged
# =====================================================================================================


@pytest.mark.parametrize("web", [True, False], ids=["web", "loopback"])
async def test_click_on_body_focus_executes(web: bool) -> None:
    """A5: a ``click_at`` whose focus is (or ends) on ``body`` executes as BASE: the click only, no focus read,
    BASE's action. Holds at BASE."""
    d = FieldDriver(real_attrs("body"))
    ex = build(d, web=web)
    f = await own_frame(ex)
    mark = len(d.calls)
    out = await asyncio.wait_for(ex.execute(inv("click_at", f, x=500, y=500)), 10.0)
    assert (out.status, out.action) == ("executed", "clicked at (640, 400)"), out
    assert d.dispatched(mark) == CLICK, d.dispatched(mark)
    assert "focused_field" not in names(d, mark), names(d, mark)
    assert "focus_check" not in out.resolved and "typed" not in out.resolved, out.resolved


# =====================================================================================================
# A7: docs
# =====================================================================================================


def _paragraph(head: str) -> str:
    doc = executor_mod.__doc__ or ""
    start = doc.find(f"**{head}**")
    assert start >= 0, f"premise: the executor docstring has a {head!r} paragraph"
    end = doc.find("\n\n", start)
    return doc[start : end if end >= 0 else len(doc)]


def test_docstring_states_body_rule() -> None:
    """A7: the "Password and card guard" paragraph states the ``body`` rule (it names ``body``); the "Replace"
    paragraph no longer lists BASE's plain ``body`` among the caret focuses (BASE's words
    ``contenteditable, ``body``, opaque`` are gone) and still names a contenteditable ``body``. *Reading:* the wording
    is the implementer's; only these facts are pinned."""
    guard = _paragraph("Password and card guard")
    assert "body" in guard, f"T-0210 A7: the guard paragraph does not state the body rule: {guard}"
    replace = _paragraph("Replace")
    assert "contenteditable, ``body``, opaque" not in replace, (
        f"T-0210 A7: Replace still lists a plain body as typed at the caret: {replace}"
    )
    assert "body" in replace, f"T-0210 A7: Replace keeps a contenteditable body: {replace}"


# =====================================================================================================
# A8: frozen paths
# =====================================================================================================


def test_frozen_paths_over_range() -> None:
    """A8: ``runtime.py``, ``fence.py``, ``egress_proxy.py``, ``toolcall/**``, ``conductor/**`` (``prompt_builder.py``
    included), ``contracts/**``, ``cli.py``, ``config.py`` and ``pages/**`` are byte-identical over
    ``RANGE_BASE..TIP_IMPL`` (two literals). ``TIP_IMPL`` is A's implementation commit, set at activation;
    ``RANGE_BASE`` is ``merge-base(main, TIP_IMPL)`` because A's branch merged main mid-way."""
    assert TIP_IMPL is not None, (
        "T-0210 A8: TIP_IMPL is set by the test-author after the implementation"
    )
    #: A's first branch commit (the pending tests), made before the mid-way merge of main
    a_first = "cda835bd7b02f4f2298ab872b03b568807295982"
    gr.require_commits(BASE_SHA, RANGE_BASE, a_first, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, RANGE_BASE), (
        f"{BASE_SHA[:8]} is not an ancestor of {RANGE_BASE[:8]}"
    )
    assert gr.is_ancestor(RANGE_BASE, TIP_IMPL), (
        f"{RANGE_BASE[:8]} is not an ancestor of {TIP_IMPL[:8]}"
    )
    # premise: the range still holds all of A's branch (its first commit is not already in RANGE_BASE)
    assert gr.is_ancestor(a_first, TIP_IMPL) and not gr.is_ancestor(a_first, RANGE_BASE), (
        f"premise: {a_first[:8]} lies in {RANGE_BASE[:8]}..{TIP_IMPL[:8]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", RANGE_BASE, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(RANGE_BASE, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0210 A8: frozen paths differ in {RANGE_BASE[:8]}..{TIP_IMPL[:8]}: {changed}"
    )


def test_frozen_paths_power_premise() -> None:
    """A8 power (green at BASE): the predicate sees a known change to a frozen path (T-0206 A's driver submit reads
    in ``runtime.py``, ``7185d998``), so the guard above cannot pass by seeing nothing."""
    gr.require_commits("1940e3e7bcbfc24c2f8e489e4a9947f985165d7f", BASE_SHA)
    assert "local_astra/computer/runtime.py" in gr.changed(
        "1940e3e7bcbfc24c2f8e489e4a9947f985165d7f", BASE_SHA, *FROZEN
    )
