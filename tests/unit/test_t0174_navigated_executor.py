"""T-0174 B1-1/B1-2 at the seam: ``resolved["navigated"]`` in ``BrowserToolExecutor`` over scripted fake drivers, no
browser (docs/tasks/T-0174.md, "Tests to write", B1).

**The capability these rows pin** (the task names a "narrow capability protocol beside ``_FocusDriver``" and says
"mark at dispatch start, read after ``settle()``", but no names; *reading*, the test-author's, reported):

- ``driver.navigation_mark() -> object``: synchronous; an opaque mark of the active page's main-frame document,
  taken at dispatch start (after ``ensure()`` and its checks, before the first input call);
- ``driver.navigated_since(mark) -> bool | None``: synchronous; called after the action's last ``settle()`` returned
  and before the post-action screenshot; ``True`` when the active page now shows a different main-frame document than
  at ``mark``, ``None`` when the driver cannot tell.
- Both are looked up on the driver's *class* (as ``focused_field`` and ``layout_metrics``); a driver class without
  them has no capability and the executor reads nothing (the T-0111c fake raises a named ``AssertionError`` on any
  public name it lacks).
- The key is set only from a ``bool`` answer: ``None``, a non-bool or a raise leave it absent (the task's "where the
  signal is unavailable, leave the key absent rather than emit ``False``"); the action itself is unaffected.

**The fake.** ``NavMixin`` over T-0111c's ``FakeDriver`` (and T-0160's ``FieldDriver``, T-0151's ``PageDriver``):
the page commits a cross-document main-frame navigation on the k-th call of a named seam method (``commit_on``), or
right now (``commit_now``, a commit between two actions). ``navigation_mark`` hands out marks it remembers;
``navigated_since`` refuses a mark it never issued (so a fabricated mark is a named failure) and answers ``commits
now > commits at the mark``, or the scripted ``answer``. The mark and the read are recorded in ``calls`` but are not
seam calls (no ``arm_raise`` count, no scene change), so the fake's screenshots are those of the plain fake.

**Expected values by hand** (viewport 1280x800): ``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``; a
``scroll down`` is ``dy = 800 * 3 // 4 = 600``. The phash salt is fixed (``secrets.randbits`` patched) where two
executors' bytes are compared.

Activated for group B1 (the pending markers are off; the seam and the cap ruling are accepted in the task status);
the negative rows (no capability is BASE byte for byte, other keys
unchanged, no key on other tools, refused, failed and dry-run outcomes, an unknown answer) hold at BASE and are
active.

Regime: CPU, fake drivers, no browser, no socket, 1280x800, n = 1, deterministic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.perception import Source
from local_astra.contracts.toolcall import FrameGeometry, ToolOutcome
from tests.unit.test_computer_executor import DriverBoom, FakeDriver, inv, own_frame
from tests.unit.test_computer_web_executor import TEXT_FIELD, FieldDriver
from tests.unit.test_t0151_executor import PageDriver, register

CLOCK_MS = 1_700_000_017_400
START = "http://127.0.0.1:8123"
SALT = 0x0123_4567_89AB_CDEF
#: ``answer`` default: count the commits
COUNT = object()


class _Mark:
    def __init__(self, n: int, commits: int) -> None:
        self.n, self.commits = n, commits

    def __repr__(self) -> str:
        return f"<mark {self.n} at commit {self.commits}>"


class NavMixin:
    """The navigation capability on top of a fake driver (module docstring)."""

    calls: list[tuple[str, tuple[Any, ...]]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commits = 0
        self.answer: Any = COUNT
        self.marks: list[_Mark] = []
        self.reads: list[_Mark] = []
        self._sched: list[list[Any]] = []

    # -- test controls
    def commit_on(self, name: str, k: int = 1, *, arg: Any = None) -> None:
        """The page commits a new main-frame document during the k-th ``name`` seam call from now; with ``arg``, only
        calls whose first argument equals it count (``commit_on("key_down", arg="Enter")``: Enter's key-down, not a
        ``Control`` from a select-all chord, T-0189)."""
        self._sched.append([name, k, arg])

    def commit_now(self) -> None:
        """A commit with no input (a timer or redirect), between two actions."""
        self.commits += 1

    def _enter(self, name: str, *args: Any) -> None:
        super()._enter(name, *args)  # type: ignore[misc]
        for s in list(self._sched):
            if s[0] == name and (s[2] is None or (args[:1] == (s[2],))):
                s[1] -= 1
                if s[1] == 0:
                    self._sched.remove(s)
                    self.commits += 1

    # -- the capability
    def navigation_mark(self) -> object:
        self.calls.append(("navigation_mark", ()))
        mark = _Mark(len(self.marks), self.commits)
        self.marks.append(mark)
        return mark

    def navigated_since(self, mark: object) -> Any:
        self.calls.append(("navigated_since", (mark,)))
        assert any(mark is m for m in self.marks), (
            f"the executor passed {mark!r}, a mark this driver never issued"
        )
        assert isinstance(mark, _Mark)
        self.reads.append(mark)
        if self.answer == "raise":
            raise DriverBoom("synthetic navigation read failure")
        if self.answer is not COUNT:
            return self.answer
        return self.commits > mark.commits


class NavDriver(NavMixin, FakeDriver):
    pass


class NavFieldDriver(NavMixin, FieldDriver):
    pass


class NavPageDriver(NavMixin, PageDriver):
    pass


def build(driver: FakeDriver, **kw: Any) -> Any:
    if kw.get("web"):
        kw.setdefault("start_origin", START)
    return executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=driver.viewport, driver=driver, **kw
    )


def nav(out: ToolOutcome) -> bool:
    """B1-1: an executed click/type/press over a capable driver carries a ``bool`` ``navigated``."""
    assert out.status == "executed", out
    assert "navigated" in out.resolved, (
        f"T-0174 B1-1: an executed {out.action!r} carries no resolved['navigated'] yet: {out.resolved}"
    )
    value = out.resolved["navigated"]
    assert isinstance(value, bool), f"B1-1: navigated must be a bool, got {value!r}"
    return value


def names(d: FakeDriver, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


#: (tool, arguments, the seam call during which the page commits)
TOOLS = [
    pytest.param("click_at", {"x": 500, "y": 500}, "mouse_up", id="click_at"),
    pytest.param(
        "type_at",
        {"x": 500, "y": 500, "text": "kettle", "press_enter": True},
        "key_down",
        id="type-enter",
    ),
    pytest.param(
        "type_at",
        {"x": 500, "y": 500, "text": "kettle", "press_enter": False},
        "type_text",
        id="type",
    ),
    pytest.param("press_key", {"combo": "Enter"}, "key_down", id="press_key"),
]


async def act(ex: Any, tool: str, args: dict[str, Any]) -> ToolOutcome:
    frame = await own_frame(ex) if tool in ("click_at", "type_at") else None
    out = await ex.execute(inv(tool, frame, **args))
    assert isinstance(out, ToolOutcome)
    return out


# =====================================================================================================
# B1-1: the key on executed click_at / type_at / press_key
# =====================================================================================================


@pytest.mark.parametrize(("tool", "args", "during"), TOOLS)
async def test_commit_during_dispatch_is_true(tool: str, args: dict[str, Any], during: str) -> None:
    """B1-1: a cross-document main-frame commit during the action's own input gives ``navigated: True``."""
    d = NavDriver()
    ex = build(d)
    await own_frame(ex)
    d.commit_on(during)
    out = await act(ex, tool, args)
    assert d.commits == 1, "premise: the page committed during the action"
    assert nav(out) is True, out.resolved


@pytest.mark.parametrize(("tool", "args", "during"), TOOLS)
async def test_no_commit_is_false(tool: str, args: dict[str, Any], during: str) -> None:
    """B1-1 (W1-1's case: Go on a refused field, nothing loads): no commit gives ``navigated: False``, not absent."""
    d = NavDriver()
    ex = build(d)
    out = await act(ex, tool, args)
    assert d.commits == 0
    assert nav(out) is False, out.resolved


async def test_commit_during_settle_counts() -> None:
    """B1-1 ("before settle returned"): a commit the settle waits for counts, so the read comes after the settle."""
    d = NavDriver()
    ex = build(d)
    f = await own_frame(ex)
    d.commit_on("settle")
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert nav(out) is True, out.resolved


STALE = [
    pytest.param("between", id="between-actions"),
    pytest.param("ensure", id="during-ensure"),
]


@pytest.mark.parametrize("when", STALE)
async def test_commit_before_dispatch_is_stale(when: str) -> None:
    """B1-1 (adversarial, the task's stale case): a commit that landed before the dispatch began, between two
    actions (a timer or redirect after the previous settle) or while ``ensure()`` ran, is not this action's:
    ``False``. Kills a mark taken before ``ensure()`` and a "since the last read" counter."""
    d = NavDriver()
    ex = build(d)
    out = await ex.execute(inv("click_at", await own_frame(ex), x=500, y=500))
    assert out.status == "executed", out
    f = await own_frame(ex)
    if when == "between":
        d.commit_now()
    else:
        d.commit_on("ensure")
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert d.commits == 1, "premise: the page committed before the dispatch"
    assert nav(out) is False, out.resolved


async def test_commit_after_settle_returned_not_counted() -> None:
    """B1-1 ("observed ... before settle returned"): a commit during the post-action screenshot, after the settle
    returned, is not counted: ``False``. (Derived pin: with this seam the read sits between the settle and the
    capture; reported.)"""
    d = NavDriver()
    ex = build(d)
    f = await own_frame(ex)
    d.commit_on("screenshot")
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert d.commits == 1, "premise: the page committed during the capture"
    assert nav(out) is False, out.resolved


async def test_mark_at_dispatch_start_and_read_after_settle() -> None:
    """The seam's order (task notes: "mark at dispatch start, read after ``settle()``"): one mark, after ``ensure()``
    and before the first input; one read of that very mark, after the last ``settle()`` and before the post-action
    ``screenshot``."""
    d = NavDriver()
    ex = build(d)
    f = await own_frame(ex)
    start = len(d.calls)
    out = await ex.execute(inv("type_at", f, x=500, y=500, text="kettle", press_enter=True))
    assert out.status == "executed", out
    seq = names(d, start)
    assert seq.count("navigation_mark") == 1 and seq.count("navigated_since") == 1, seq
    mark, read = seq.index("navigation_mark"), seq.index("navigated_since")
    first_input = min(i for i, n in enumerate(seq) if n in ("mouse_move", "mouse_down", "goto"))
    last_settle = max(i for i, n in enumerate(seq) if n == "settle")
    assert seq.index("ensure") < mark < first_input, seq
    assert last_settle < read < seq.index("screenshot", last_settle), seq
    assert d.reads == [d.marks[-1]], (d.reads, d.marks)


async def test_cap_close_alone_is_false() -> None:
    """B1-1: a popup closed by the ``MAX_PAGES`` cap that never became active (a ``page_closed`` event, no page
    switch, no commit) is not a navigation: ``False``, with BASE's ``closed_pages``."""
    d = NavDriver()
    ex = build(d)
    f = await own_frame(ex)
    d.events.append({"kind": "page_closed"})
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.resolved.get("closed_pages") == 1, out.resolved
    assert nav(out) is False, out.resolved


async def test_key_set_is_base_plus_navigated() -> None:
    """B1-2 ("no new page event in ``_report``"): an executed click over a capable driver has exactly BASE's keys
    plus ``navigated``, and BASE's values."""
    d = NavDriver()
    ex = build(d)
    f = await own_frame(ex)
    d.commit_on("mouse_up")
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert nav(out) is True
    assert set(out.resolved) == {"x_px", "y_px", "frame_after", "navigated"}, out.resolved
    assert (out.resolved["x_px"], out.resolved["y_px"]) == (640, 400)
    assert out.resolved["frame_after"] == ex.latest_frame.phash
    assert out.action == "clicked at (640, 400)", out.action


# ----------------------------------------------------------------------------------------- web mode: hosts


async def test_hosts_without_commit_is_false() -> None:
    """B1-1 (adversarial: ``hosts`` is not read): the proxy tunnelled a host during the action (a prefetch, a
    beacon) but no document was committed: ``False``, and ``hosts`` is BASE's."""
    d = NavDriver()
    ex = build(d, web=True)
    f = await own_frame(ex)
    d.events.append({"kind": "host", "host": "en.wikipedia.org"})
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.resolved.get("hosts") == ["en.wikipedia.org"], f"premise: {out.resolved}"
    assert nav(out) is False, out.resolved


async def test_commit_without_hosts_is_true() -> None:
    """B1-1 (adversarial, the other side): a commit with no new host (a same-origin page, or a cached one): ``True``,
    and ``hosts`` is ``[]``."""
    d = NavDriver()
    ex = build(d, web=True)
    f = await own_frame(ex)
    d.commit_on("mouse_up")
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.resolved.get("hosts") == [], f"premise: {out.resolved}"
    assert nav(out) is True, out.resolved


async def test_web_start_page_key_is_navigated() -> None:
    """B1-1 (``press_key``): web mode's ``Ctrl+T`` opens the start page with ``goto`` (dispatch); its commit gives
    ``True``; BASE's action and keys otherwise."""
    d = NavDriver()
    ex = build(d, web=True)
    await own_frame(ex)
    d.commit_on("goto")
    out = await ex.execute(inv("press_key", None, combo="Ctrl+T"))
    assert out.action == "opened the start page", out.action
    assert nav(out) is True, out.resolved
    assert set(out.resolved) == {"combo", "frame_after", "hosts", "navigated"}, out.resolved


async def test_web_guarded_type_is_navigated() -> None:
    """B1-1 (``type_at`` on web mode's guarded path: click, read the field, type): Enter submits and commits: ``True``;
    the field read is not a navigation read (one mark, one read). The commit fires on Enter's key-down only (T-0189
    criterion 7: the guarded path now sends a ``Ctrl+A`` chord into a text field first, whose ``Control`` key-down is
    not a submit), so the row still means "Enter submits"."""
    d = NavFieldDriver(TEXT_FIELD)
    ex = build(d, web=True)
    f = await own_frame(ex)
    d.commit_on("key_down", arg="Enter")
    start = len(d.calls)
    out = await ex.execute(inv("type_at", f, x=500, y=500, text="kettle", press_enter=True))
    assert d.commits == 1, "premise: Enter's key-down committed"
    assert nav(out) is True, out.resolved
    seq = names(d, start)
    assert seq.count("navigation_mark") == 1 and seq.count("navigated_since") == 1, seq


# ----------------------------------------------------------------------------------------- the overview frame path


@pytest.mark.parametrize("commit", [True, False], ids=["commit", "no-commit"])
async def test_overview_frame_click(commit: bool) -> None:
    """B1-1 on T-0151's overview-frame path (an executed ``click_at`` too): a 1220 px page, overview (160, 935) is
    page y 1141, one scroll of 420 px, then the click; a commit during the click gives ``True``, none ``False``."""
    d = NavPageDriver(content_height=1220.0)
    ex = build(d, echo="model")
    await own_frame(ex)
    ov = await register(ex)
    if commit:
        d.commit_on("mouse_up")
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "executed" and out.action.startswith("scrolled down by 420 px and "), out
    assert nav(out) is commit, out.resolved


async def test_overview_frame_commit_after_settle_not_counted() -> None:
    """B1-1/B1-1b on the overview-frame path, mirroring ``test_commit_after_settle_returned_not_counted``: a commit
    during the post-action capture (after the last settle returned) is not counted: ``False``. Kills a read moved
    after ``_act_on_overview``'s capture."""
    d = NavPageDriver(content_height=1220.0)
    ex = build(d, echo="model")
    await own_frame(ex)
    ov = await register(ex)
    d.commit_on("screenshot")
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "executed" and out.action.startswith("scrolled down by 420 px and "), out
    assert d.commits == 1, "premise: the page committed during the capture"
    assert nav(out) is False, out.resolved


PENDING_TOOLS = [
    *TOOLS,
    pytest.param("overview", {"x": 160, "y": 935}, "mouse_up", id="overview-click"),
]


@pytest.mark.parametrize(("tool", "args", "during"), PENDING_TOOLS)
async def test_pending_navigation_reads_absent(
    tool: str, args: dict[str, Any], during: str
) -> None:
    """B1-1b: a main-frame navigation request still pending at the read is the driver's ``None``: the key is absent,
    never a guessed ``False``, even though the action dispatched input; the action and every other key are as
    without the capability."""
    d: Any = NavPageDriver(content_height=1220.0) if tool == "overview" else NavDriver()
    ex = build(d, echo="model")
    await own_frame(ex)
    d.answer = None
    d.commit_on(during)
    if tool == "overview":
        ov = await register(ex)
        out = await ex.execute(inv("click_at", ov, **args))
    else:
        out = await act(ex, tool, args)
    assert out.status == "executed", out
    assert d.reads, "premise: the executor asked the driver (B1-1b's read happened)"
    assert "navigated" not in out.resolved, out.resolved
    assert set(out.resolved) <= {
        "x_px",
        "y_px",
        "page_y_px",
        "scrolled_px",
        "text",
        "press_enter",
        "combo",
        "frame_after",
    }, out.resolved


# =====================================================================================================
# B1-1 absences and B1-2 negatives (active: they hold at BASE)
# =====================================================================================================


def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_mod.secrets, "randbits", lambda k: SALT)


SCRIPT = [
    ("click_at", {"x": 500, "y": 500}),
    ("type_at", {"x": 500, "y": 500, "text": "kettle", "press_enter": True}),
    ("press_key", {"combo": "Enter"}),
    ("scroll", {"direction": "down"}),
    ("press_key", {"combo": "Tab"}),
]


async def run_script(d: FakeDriver, *, commits: bool = False, **kw: Any) -> list[ToolOutcome]:
    ex = build(d, **kw)
    outs = []
    for tool, args in SCRIPT:
        if commits and isinstance(d, NavMixin):
            d.commit_on("settle")
        if tool == "press_key" and args["combo"] == "Tab":
            d.events.append({"kind": "page_closed"})
            d.events.append({"kind": "dialog", "type": "alert"})
        outs.append(await act(ex, tool, args))
    return outs


async def test_without_capability_resolved_is_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1-2 / mutation "emits False when the capability is absent": a driver class without the two methods yields
    BASE's ``resolved`` byte for byte (keys, order, values by hand) and BASE's actions, never a ``navigated``."""
    fixed_salt(monkeypatch)
    d = FakeDriver()
    outs = await run_script(d)
    phashes = [o.resolved["frame_after"] for o in outs]
    want = [
        ("clicked at (640, 400)", {"x_px": 640, "y_px": 400, "frame_after": phashes[0]}),
        (
            "clicked at (640, 400) and typed 6 characters and pressed Enter",
            {
                "x_px": 640,
                "y_px": 400,
                "text": "kettle",
                "press_enter": True,
                "frame_after": phashes[1],
            },
        ),
        ("pressed Enter", {"combo": "Enter", "frame_after": phashes[2]}),
        (
            "scrolled down by 600 px",
            {"direction": "down", "dx_px": 0, "dy_px": 600, "frame_after": phashes[3]},
        ),
        (
            "pressed Tab",
            {
                "combo": "Tab",
                "frame_after": phashes[4],
                "dismissed": [{"dialog": "alert"}],
                "closed_pages": 1,
            },
        ),
    ]
    for out, (action, resolved) in zip(outs, want, strict=True):
        assert out.status == "executed", out
        assert out.action == action, out.action
        assert json.dumps(out.resolved) == json.dumps(resolved), (out.resolved, resolved)
        assert "navigated" not in out.resolved


async def test_capability_leaves_other_keys_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1-2: the same script over a capable driver (a commit in every settle) and over the plain fake, with one salt:
    the actions are equal and every ``resolved`` minus ``navigated`` is equal, in BASE's key order."""
    fixed_salt(monkeypatch)
    base = await run_script(FakeDriver())
    got = await run_script(NavDriver(), commits=True)
    for b, g in zip(base, got, strict=True):
        assert (g.status, g.action, g.reason) == (b.status, b.action, b.reason)
        rest = {k: v for k, v in g.resolved.items() if k != "navigated"}
        assert json.dumps(rest) == json.dumps(b.resolved), (rest, b.resolved)


async def test_other_tools_have_no_key() -> None:
    """B1-1: ``scroll`` (even with a commit during the wheel), ``page_overview`` and ``google_search`` never carry
    the key."""
    d = NavPageDriver(content_height=1220.0)
    ex = build(d)
    await own_frame(ex)
    d.commit_on("scroll_by")
    out = await ex.execute(inv("scroll", None, direction="down"))
    assert out.status == "executed" and d.commits == 1, out
    assert "navigated" not in out.resolved, out.resolved
    out = await ex.execute(inv("page_overview"))
    assert out.status == "executed" and "navigated" not in out.resolved, out
    out = await ex.execute(inv("google_search", None, query="kettle"))
    assert out.status == "dry_run" and "navigated" not in out.resolved, out


async def test_refused_outcomes_have_no_key() -> None:
    """B1-1: ``refused`` outcomes (bad coordinates, the guard, the active page changed in ``ensure()``) carry no key,
    whatever committed."""
    d = NavDriver()
    ex = build(d, guard=lambda i: "no" if i.arguments.get("combo") == "Escape" else None)
    f = await own_frame(ex)
    d.commit_on("ensure")
    out = await ex.execute(inv("click_at", f, x=1000, y=5))
    assert out.status == "refused" and out.resolved == {}, out
    out = await ex.execute(inv("press_key", None, combo="Escape"))
    assert out.status == "refused" and out.resolved == {}, out
    d.open_popup()
    d.commit_on("ensure")
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.status == "refused" and out.resolved == {}, out


async def test_failed_outcomes_have_no_key() -> None:
    """B1-1: ``failed`` outcomes carry no key: a driver error after a commit in the same action, the password guard
    (web mode; a ``resolved`` with ``frame_after``) and T-0151's not-reached overview click."""
    d = NavDriver()
    ex = build(d)
    f = await own_frame(ex)
    d.commit_on("mouse_down")
    d.arm_raise(
        3
    )  # counted seam calls (ensure is not): mouse_move, mouse_down, then mouse_up raises
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.status == "failed" and "navigated" not in out.resolved, out

    fd = NavFieldDriver({"tag": "input", "type": "password", "name": "pw"})
    ex = build(fd, web=True)
    f = await own_frame(ex)
    fd.commit_on("mouse_up")
    out = await ex.execute(inv("type_at", f, x=500, y=500, text="hunter2", press_enter=True))
    assert out.status == "failed" and "frame_after" in out.resolved, out
    assert "navigated" not in out.resolved, out.resolved

    pd = NavPageDriver(content_height=1220.0)
    ex = build(pd, echo="model")
    await own_frame(ex)
    ov = await register(ex)
    pd.scroll_land = 100.0
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "failed" and "frame_after" in out.resolved, out
    assert "navigated" not in out.resolved, out.resolved


async def test_dry_run_has_no_key() -> None:
    """B1-1: dry-run outcomes never carry the key."""
    dry = DryRunToolExecutor()
    geom = FrameGeometry(phash="0123456789abcdef", source=Source.WINDOW, width=1280, height=800)
    for tool, args in SCRIPT:
        out = await dry.execute(
            inv(tool, geom if tool in ("click_at", "type_at") else None, **args)
        )
        assert out.status == "dry_run" and "navigated" not in out.resolved, out


UNKNOWN = [
    pytest.param(None, id="none"),
    pytest.param("raise", id="raises"),
    pytest.param(1, id="int-1"),
    pytest.param("true", id="str-true"),
]


@pytest.mark.parametrize("answer", UNKNOWN)
async def test_unknown_answer_leaves_key_absent(
    answer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1-1 ("leave the key absent rather than emit ``False``"): a driver that cannot tell (``None``), fails to read,
    or answers something that is not a ``bool`` gives no key, and the action is BASE's (executed, same keys)."""
    fixed_salt(monkeypatch)
    base = await run_script(FakeDriver())
    d = NavDriver()
    d.answer = answer
    got = await run_script(d, commits=True)
    for b, g in zip(base, got, strict=True):
        assert (g.status, g.action) == (b.status, b.action), g
        assert json.dumps(g.resolved) == json.dumps(b.resolved), (g.resolved, b.resolved)
