"""T-0203 group B at the seam: ``BrowserToolExecutor`` follows a navigation still in flight after the settle, over
scripted fake drivers, no browser (docs/tasks/T-0203.md, B1-B7; "Tests to write").

**The capability these rows pin** (A1, B1): ``driver.follow_navigation(mark, timeout_s) -> "none" | "loaded" |
"ended" | "timeout"``, async, looked up on the driver's *class* (as every optional capability); ``NAV_FOLLOW_S = 3.0``
is a module constant of ``executor`` read at call time; the note goes through T-0200's ``on_note``.

**The fake.** ``FollowMixin`` over T-0174's ``NavMixin`` (marks it remembers; a fabricated mark is a named failure):

- ``reads``: a script of ``navigated_since`` answers, one per read (``"raise"`` raises); once empty, NavMixin's
  commit count answers.
- ``follows``: a script, one item per ``follow_navigation`` call: a result string; ``("sleep", s, result)``;
  ``("budget", result)`` (sleeps its whole ``timeout_s``, as A4's bound allows); ``"hang"`` (until cancelled); or an
  exception to raise. Once empty, ``"none"``. A ``"loaded"`` result moves the scene (the new page's pixels), so a
  capture after it differs from one before it. Each call is recorded in ``calls`` (with its mark and ``timeout_s``)
  and in ``follow_log`` with its start time; it is not a seam call (no ``arm_raise`` count).

``ShotsMixin`` (T-0200) records every screenshot's returned bytes and scripts screenshot failures and blank pictures;
``UrlsMixin`` (T-0192) is the checkout read; ``PageDriver`` (T-0151) the overview frame.

**Oracles.** BASE is the executor at ``BASE_SHA`` (Merge T-0202-b + T-0203-a; it never follows), executed from
``git show`` as T-0200's differential does; with the phash salt patched, a negative row compares the driver call
sequence, outcomes, bytes and phashes byte for byte. A discarded capture's phash is BASE's own frame of the same bytes
under the same salt. Budgets are by hand: with ``ACTION_TIMEOUT_S = 8`` and ``CAPTURE_RESERVE_S = 1`` the deadline is
7 s after the call starts, so a follow right after a fast settle gets ``min(3.0, ~7) = 3.0``.

Regime: CPU over fakes, real asyncio time with budgets patched small where a row waits (under 3 s in total),
1280x800, n = 1, deterministic salt.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
import time
import types
from collections import deque
from collections.abc import Iterator
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.unit.test_computer_executor import CLOCK_MS, FakeDriver, fake_jpeg
from tests.unit.test_computer_web_executor import FieldDriver
from tests.unit.test_t0151_executor import PageDriver, register
from tests.unit.test_t0174_navigated_executor import NavMixin
from tests.unit.test_t0192_checkout_executor import CHECKOUT, SHOP, START, UrlsMixin
from tests.unit.test_t0200_blank_recapture import WHITE, ShotsMixin

#: T-0203 B's BASE: HEAD of T-0203-b-follow-exec before the B tests (Merge T-0202-b + T-0203-a). A literal.
BASE_SHA = "174214db3dd35864cb2797fad0ee324d92e756d5"
SALT = 0x5EED_0203_0203_5EED
W, H = 1280, 800
NOTE_WHAT = "live frame: followed navigation"
NOTE_KEYS = {"what", "path", "result", "late", "waited_ms"}
BLANK_WHAT = "live frame: blank capture retaken"


# ============================================================================================ the fakes


class FollowMixin(NavMixin):
    """``follow_navigation`` and scripted ``navigated_since`` answers (module docstring)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read_script: deque[Any] = deque()
        self.follow_script: deque[Any] = deque()
        self.follow_log: list[tuple[object, float, float]] = []
        self.follow_entered = asyncio.Event()
        self.on_follow: Any = None

    def navigated_since(self, mark: object) -> Any:
        if not self.read_script:
            return super().navigated_since(mark)
        self.calls.append(("navigated_since", (mark,)))
        assert any(mark is m for m in self.marks), f"a mark this driver never issued: {mark!r}"
        value = self.read_script.popleft()
        if value == "raise":
            raise RuntimeError("synthetic navigation read failure")
        return value

    async def follow_navigation(self, mark: object, timeout_s: float) -> str:
        self.calls.append(("follow_navigation", (mark, timeout_s)))
        self.follow_log.append((mark, timeout_s, time.monotonic()))
        assert any(mark is m for m in self.marks), f"follow with a mark never issued: {mark!r}"
        assert isinstance(timeout_s, (int, float)), timeout_s
        self.follow_entered.set()
        item = self.follow_script.popleft() if self.follow_script else "none"
        if self.on_follow is not None:
            self.on_follow()
        if isinstance(item, BaseException):
            raise item
        if item == "hang":
            await asyncio.Event().wait()
        if isinstance(item, tuple) and item[0] == "sleep":
            await asyncio.sleep(item[1])
            item = item[2]
        elif isinstance(item, tuple) and item[0] == "budget":
            await asyncio.sleep(max(0.0, timeout_s))
            item = item[1]
        if item == "loaded":
            self.scene += 1  # type: ignore[attr-defined]
        return str(item)


class FollowDriver(FollowMixin, FakeDriver):
    pass


class FollowShot(ShotsMixin, FollowMixin, FakeDriver):
    pass


class FollowSlowSettle(FollowMixin, FakeDriver):
    settle_s = 0.0

    async def settle(self) -> None:
        await super().settle()
        if self.settle_s:
            await asyncio.sleep(self.settle_s)


class FollowSlowShot(ShotsMixin, FollowMixin, FakeDriver):
    """A screenshot that hangs: the ``hang_at``-th screenshot from ``arm()`` sleeps ``hang_s`` (cancellable) before
    answering; ``settle_s`` makes the settle slow."""

    hang_at = 0
    hang_s = 30.0
    settle_s = 0.0
    _armed = 0

    def arm(self, n: int) -> None:
        self.hang_at, self._armed = n, 0

    async def screenshot(self) -> bytes:
        if self.hang_at:
            self._armed += 1
            if self._armed == self.hang_at:
                await asyncio.sleep(self.hang_s)
        out: bytes = await super().screenshot()
        return out

    async def settle(self) -> None:
        await super().settle()
        if self.settle_s:
            await asyncio.sleep(self.settle_s)


class FollowUrls(FollowMixin, UrlsMixin, FakeDriver):
    pass


class FollowPage(FollowMixin, PageDriver):
    pass


class FollowField(FollowMixin, FieldDriver):
    pass


class PlainNav(NavMixin, FakeDriver):
    """T-0174's capability without ``follow_navigation`` (the negative B6 class)."""


class Boom(RuntimeError):
    pass


# ============================================================================================ helpers


@pytest.fixture(autouse=True)
def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "randbits", lambda n: SALT)


def build(
    d: Any, *, module: Any = executor_mod, notes: list[dict[str, Any]] | None = None, **kw: Any
) -> Any:
    if kw.get("web"):
        kw.setdefault("start_origin", START)
    ex = module.BrowserToolExecutor(now_ms=lambda: CLOCK_MS, viewport=d.viewport, driver=d, **kw)
    if notes is not None:
        ex.on_note = notes.append
    return ex


_N = iter(range(1_000_000))


def call(
    name: str, frame: Frame | None = None, *, turn: str = "t1", **arguments: Any
) -> ToolInvocation:
    g = (
        FrameGeometry(
            phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
        )
        if frame is not None
        else None
    )
    return ToolInvocation(
        call_id=f"f{next(_N)}",
        turn_id=turn,
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


async def own(ex: Any) -> Frame:
    f = await ex.frame()
    assert isinstance(f, Frame)
    return f


TYPE_ENTER = {"x": 500, "y": 500, "text": "kettle", "press_enter": True}
TYPE_PLAIN = {"x": 500, "y": 500, "text": "kettle", "press_enter": False}
CLICK = {"x": 500, "y": 500}


async def act(ex: Any, tool: str, args: dict[str, Any]) -> ToolOutcome:
    frame = ex.latest_frame if tool in ("click_at", "type_at") else None
    if frame is None and tool in ("click_at", "type_at"):
        frame = await own(ex)
    out = await ex.execute(call(tool, frame, **args))
    assert isinstance(out, ToolOutcome)
    return out


def names(d: Any, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def follows(d: Any, since: int = 0) -> list[tuple[Any, ...]]:
    return [c[1] for c in d.calls[since:] if c[0] == "follow_navigation"]


def follow_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [n for n in notes if n.get("what") == NOTE_WHAT]
    for n in out:
        assert set(n) == NOTE_KEYS, f"T-0203 B5: the note has exactly {sorted(NOTE_KEYS)}: {n}"
        assert n["path"] == "action", n
        assert n["result"] in ("loaded", "ended", "timeout"), n
        assert type(n["late"]) is bool and type(n["waited_ms"]) is int, n
    return out


def const(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0203 B1: executor.{name} does not exist yet"
    return value


# -- BASE

_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "def _capture" in src and "follow_navigation" not in src, (
            "premise: BASE does not follow"
        )
        name = "_t0203b_base_executor"
        mod = types.ModuleType(name)
        mod.__file__ = f"<{BASE_SHA[:8]}>/local_astra/computer/executor.py"
        sys.modules[name] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        _BASE = mod
    return _BASE


async def base_phash(jpeg: bytes) -> str:
    """BASE's own frame of exactly these bytes under the fixed salt (generation 1): the phash a registered capture of
    these bytes carries."""
    d = FollowShot()
    d.queue(jpeg)
    f = await build(d, module=base_module()).frame()
    assert f.jpeg == jpeg
    return str(f.phash)


def _norm(calls: list[tuple[str, tuple[Any, ...]]]) -> list[Any]:
    out: list[Any] = []
    for name, args in calls:
        out.append([name, [repr(a) if not isinstance(a, (int, float, str)) else a for a in args]])
    return out


async def trace(
    module: Any, cls: type, steps: list[tuple[str, dict[str, Any]]], **kw: Any
) -> dict[str, Any]:
    """One executor over a fresh ``cls`` driver: a first frame, then ``steps``; the call sequence, outcomes, kept bytes
    and phashes (what B6 compares with BASE)."""
    d = cls()
    if "reads" in kw:
        d.read_script.extend(kw.pop("reads"))
    ex = build(d, module=module, **kw)
    rec: dict[str, Any] = {"outs": [], "frames": []}
    await own(ex)
    for tool, args in steps:
        out = await act(ex, tool, args)
        rec["outs"].append([out.status, out.action, out.reason, out.resolved])
        rec["frames"].append([ex.latest_frame.phash, ex.latest_frame.jpeg.hex()])
    rec["calls"] = _norm(d.calls)
    rec["driver"] = d
    return rec


def same(got: dict[str, Any], want: dict[str, Any], what: str) -> None:
    for key in ("outs", "frames", "calls"):
        assert json.dumps(got[key], default=str) == json.dumps(want[key], default=str), (
            f"T-0203 B6 {what}: {key} differs from BASE {BASE_SHA[:8]}:\n got  {got[key]}\n base {want[key]}"
        )


class Sleeps(list[float]):
    pass


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> Iterator[Sleeps]:
    """Every ``asyncio.sleep`` the executor awaits (the fakes here sleep only when a row scripts it)."""
    seen = Sleeps()
    real = asyncio.sleep

    async def rec(delay: float, result: Any = None) -> Any:
        seen.append(float(delay))
        return await real(delay, result)

    monkeypatch.setattr(asyncio, "sleep", rec)
    yield seen


# ============================================================================================ B1


ACTIONS = [
    pytest.param("click_at", CLICK, id="click_at"),
    pytest.param("type_at", TYPE_ENTER, id="type-enter"),
    pytest.param("type_at", TYPE_PLAIN, id="type"),
    pytest.param("press_key", {"combo": "Enter"}, id="press_key"),
]


@pytest.mark.parametrize(("tool", "args"), ACTIONS)
async def test_follow_called_on_none_then_reread(tool: str, args: dict[str, Any]) -> None:
    """B1: the read after the settle gives ``None`` (pending): one ``follow_navigation`` with the mark taken at
    dispatch start and ``min(NAV_FOLLOW_S, deadline - now) = 3.0``, then one more read, whose ``True`` is
    ``resolved["navigated"]``; the after-frame is captured after both (the new page's pixels)."""
    assert const("NAV_FOLLOW_S") == 3.0
    d = FollowDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.read_script.extend([None, True])
    d.follow_script.append("loaded")
    since = len(d.calls)
    out = await act(ex, tool, args)
    assert out.status == "executed", out
    seq = names(d, since)
    got = follows(d, since)
    assert len(got) == 1, f"T-0203 B1: one follow after a pending read: {seq}"
    mark, timeout_s = got[0]
    assert mark is d.marks[-1], "B1: the follow gets the action's own T-0174 mark"
    assert abs(timeout_s - 3.0) < 1e-6, (
        f"B1: min(NAV_FOLLOW_S, deadline - now) = 3.0, got {timeout_s}"
    )
    last_settle = max(i for i, n in enumerate(seq) if n == "settle")
    reads = [i for i, n in enumerate(seq) if n == "navigated_since"]
    f_at = seq.index("follow_navigation")
    shot = seq.index("screenshot", last_settle)
    assert last_settle < reads[0] < f_at < reads[1] < shot and len(reads) == 2, (
        f"B1 order: settle -> read -> follow -> re-read -> capture: {seq}"
    )
    assert out.resolved.get("navigated") is True, out.resolved
    assert out.resolved["frame_after"] == ex.latest_frame.phash
    assert ex.latest_frame.jpeg == fake_jpeg((W, H), d.generation, d.scene), (
        "the capture is after the follow"
    )
    fn = follow_notes(notes)
    assert [(n["result"], n["late"]) for n in fn] == [("loaded", False)], notes


@pytest.mark.parametrize(
    ("answers", "want"),
    [
        pytest.param([None, None], "absent", id="still-pending"),
        pytest.param([True, None], "absent", id="true-then-unknown"),
        pytest.param([None, False], False, id="ended-false"),
        pytest.param([None, "maybe"], "absent", id="non-bool"),
        pytest.param([None, "raise"], "absent", id="reread-raises"),
    ],
)
async def test_reread_is_the_answer(answers: list[Any], want: Any) -> None:
    """B1/B7: the second read is the answer, and the key is set only from a ``bool``: still pending after the follow
    (``None``), a non-bool or a raising read leave it absent, even when the first read was ``True``; the action is
    executed."""
    d = FollowDriver()
    ex = build(d)
    await own(ex)
    d.read_script.extend(answers)
    d.follow_script.append("timeout" if answers[1] is None else "ended")
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed", out
    assert len(follows(d)) == 1, names(d)
    if want == "absent":
        assert "navigated" not in out.resolved, out.resolved
    else:
        assert out.resolved.get("navigated") is want, out.resolved


async def test_follow_called_on_true_returns_none_cheaply(sleeps: Sleeps) -> None:
    """B1: a ``True`` read is followed too (the committed document may still be loading); a follow that answers
    ``"none"`` adds nothing else: no note, no sleep, the re-read's ``True``, one capture."""
    d = FollowDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.read_script.extend([True, True])
    d.follow_script.append("none")
    since = len(d.calls)
    sleeps.clear()
    out = await act(ex, "type_at", TYPE_ENTER)
    assert out.status == "executed", out
    assert len(follows(d, since)) == 1, names(d, since)
    assert out.resolved.get("navigated") is True, out.resolved
    assert notes == [], f"B5: no note for 'none': {notes}"
    assert sleeps == [], f"no fixed wait anywhere: {sleeps}"
    assert names(d, since).count("screenshot") == 1, names(d, since)


async def test_follow_needs_a_mark() -> None:
    """B1: no follow when no T-0174 mark was taken (the mark raised): the action is BASE's, no key."""

    class NoMark(FollowDriver):
        def navigation_mark(self) -> object:
            raise Boom("no mark")

    d = NoMark()
    ex = build(d)
    await own(ex)
    out = await act(ex, "type_at", TYPE_ENTER)
    assert out.status == "executed", out
    assert "follow_navigation" not in names(d) and "navigated_since" not in names(d), names(d)
    assert "navigated" not in out.resolved


async def test_instance_attribute_is_not_the_capability() -> None:
    """B1: the capability is looked up on the driver's *class*; a ``follow_navigation`` set on an instance of a class
    without it is never called (T-0174's class-lookup rule)."""
    d = PlainNav()
    called: list[Any] = []

    async def fake(mark: object, timeout_s: float) -> str:
        called.append((mark, timeout_s))
        return "loaded"

    d.follow_navigation = fake  # type: ignore[attr-defined]
    d.answer = None
    ex = build(d)
    await own(ex)
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed" and called == [], (out, called)


async def test_checkout_read_after_the_follow() -> None:
    """B1 order (web mode): settle -> read -> follow -> re-read -> checkout read (T-0192 A4) -> capture; a checkout
    the follow reached is reported (``checkout: True``)."""
    d = FollowUrls()
    ex = build(d, web=True)
    await own(ex)
    d.read_script.extend([None, True])
    d.follow_script.append("loaded")
    d.on_follow = lambda: d.visit(CHECKOUT)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed", out
    seq = names(d, since)
    last_settle = max(i for i, n in enumerate(seq) if n == "settle")
    tail = [
        n
        for n in seq[last_settle:]
        if n in ("navigated_since", "follow_navigation", "main_frame_urls", "screenshot")
    ]
    assert tail == [
        "navigated_since",
        "follow_navigation",
        "navigated_since",
        "main_frame_urls",
        "screenshot",
    ], seq
    assert out.resolved.get("checkout") is True, out.resolved


async def test_web_start_page_goto_follows() -> None:
    """B1: web mode's start-page ``goto`` (``Ctrl+L``) is followed like any other navigating action."""
    d = FollowUrls()
    d.url = f"{SHOP}/"
    ex = build(d, web=True)
    await own(ex)
    d.read_script.extend([None, True])
    d.follow_script.append("loaded")
    out = await ex.execute(call("press_key", None, combo="Ctrl+L"))
    assert out.status == "executed" and out.action == "opened the start page", out
    assert len(follows(d)) == 1, names(d)
    assert out.resolved.get("navigated") is True, out.resolved


async def test_overview_frame_click_follows() -> None:
    """B1: the overview-frame click (T-0151) follows a pending navigation the same way: one follow, the re-read's
    ``True``, the capture after it."""
    d = FollowPage(content_height=1220.0)
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes, echo="model")
    await own(ex)
    ov = await register(ex)
    d.read_script.extend([None, True])
    d.follow_script.append("loaded")
    since = len(d.calls)
    out = await ex.execute(call("click_at", ov, x=500, y=900))
    assert out.status == "executed", out
    seq = names(d, since)
    assert seq.count("follow_navigation") == 1, seq
    assert seq.index("follow_navigation") < max(
        i for i, n in enumerate(seq) if n == "screenshot"
    ), seq
    assert out.resolved.get("navigated") is True, out.resolved
    assert [n["result"] for n in follow_notes(notes)] == ["loaded"], notes


# ============================================================================================ B6 (negative)


NEG_STEPS = [
    ("click_at", CLICK),
    ("type_at", TYPE_PLAIN),
    ("press_key", {"combo": "Tab"}),
    ("scroll", {"direction": "down"}),
    ("wait", {}),
]


async def test_false_without_enter_base_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """B6 (negative, BASE differential): ``click_at``/``type_at``/``press_key`` whose read is ``False`` and that sent
    no Enter, then ``scroll`` and ``wait``, over a driver *with* ``follow_navigation``: BASE's exact driver call
    sequence, outcomes, ``resolved``, bytes and phashes; ``follow_navigation`` is never called."""
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.01)
    monkeypatch.setattr(base_module(), "WAIT_S", 0.01)
    want = await trace(base_module(), FollowDriver, NEG_STEPS)
    got = await trace(executor_mod, FollowDriver, NEG_STEPS)
    assert "follow_navigation" not in names(got["driver"])
    same(got, want, "False without Enter")


@pytest.mark.parametrize("answer", [None, True, False])
async def test_driver_without_follow_is_base(answer: Any) -> None:
    """B6 (negative): a driver class without ``follow_navigation`` (T-0174's capability only), whatever its read says:
    BASE's call sequence, outcomes, bytes and phashes for every navigating action, Enter included."""
    steps = [("type_at", TYPE_ENTER), ("click_at", CLICK), ("press_key", {"combo": "Enter"})]

    async def run(module: Any) -> dict[str, Any]:
        d = PlainNav()
        d.answer = answer
        ex = build(d, module=module)
        rec: dict[str, Any] = {"outs": [], "frames": []}
        await own(ex)
        for tool, args in steps:
            out = await act(ex, tool, args)
            rec["outs"].append([out.status, out.action, out.reason, out.resolved])
            rec["frames"].append([ex.latest_frame.phash, ex.latest_frame.jpeg.hex()])
        rec["calls"] = _norm(d.calls)
        return rec

    same(await run(executor_mod), await run(base_module()), f"no capability, read {answer!r}")


async def test_scroll_wait_overview_never_follow(monkeypatch: pytest.MonkeyPatch) -> None:
    """B6 (negative): ``scroll``, ``wait``, ``page_overview`` and ``google_search`` never follow (and read nothing),
    even when a navigation would be pending; BASE's call sequence and outcomes."""
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.01)
    monkeypatch.setattr(base_module(), "WAIT_S", 0.01)
    steps = [
        ("scroll", {"direction": "down"}),
        ("wait", {}),
        ("page_overview", {}),
        ("google_search", {"query": "kettle"}),
    ]

    async def run(module: Any) -> dict[str, Any]:
        d = FollowPage(content_height=1220.0)
        d.answer = None
        ex = build(d, module=module)
        rec: dict[str, Any] = {"outs": [], "frames": []}
        await own(ex)
        for tool, args in steps:
            out = await act(ex, tool, args)
            res = {k: v for k, v in out.resolved.items() if k != "overview"}
            rec["outs"].append([out.status, out.action, out.reason, res])
            rec["frames"].append([ex.latest_frame.phash, ex.latest_frame.jpeg.hex()])
        rec["calls"] = _norm(d.calls)
        rec["driver"] = d
        return rec

    got = await run(executor_mod)
    assert "follow_navigation" not in names(got["driver"]), names(got["driver"])
    same(got, await run(base_module()), "scroll/wait/overview/search")


async def test_late_start_enter_false_twice_one_extra_read() -> None:
    """B6/B2 (adversarial): an Enter action that reads ``False`` twice adds exactly one synchronous
    ``navigated_since`` after its capture and nothing else: removing that one read from the call sequence gives
    BASE's; same outcome, bytes and phash; no follow."""
    for tool, args in (
        ("type_at", TYPE_ENTER),
        ("press_key", {"combo": "Enter"}),
        ("press_key", {"combo": "Ctrl+Enter"}),
    ):
        want = await trace(base_module(), FollowDriver, [(tool, args)])
        got = await trace(executor_mod, FollowDriver, [(tool, args)])
        calls = got["calls"]
        shot = max(i for i, c in enumerate(calls) if c[0] == "screenshot")
        extra = [i for i, c in enumerate(calls) if c[0] == "navigated_since" and i > shot]
        assert len(extra) == 1, (
            f"T-0203 B2: one late re-check read after the capture for {tool} {args}: {calls}"
        )
        assert "follow_navigation" not in [c[0] for c in calls], calls
        got["calls"] = calls[: extra[0]] + calls[extra[0] + 1 :]
        same(got, want, f"Enter read False twice ({tool} {args})")


# ============================================================================================ B2


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        pytest.param("type_at", TYPE_ENTER, id="type-enter"),
        pytest.param("press_key", {"combo": "Enter"}, id="press-enter"),
        pytest.param("press_key", {"combo": "Ctrl+Enter"}, id="ctrl-enter"),
    ],
)
async def test_late_start_enter_recaptures_and_registers_only_kept(
    tool: str, args: dict[str, Any]
) -> None:
    """B2: an Enter action reads ``False``, captures, then the late check sees the navigation (``None``): it follows
    (``late: True``), re-reads (``True``) and captures again. Only the second capture is registered: it is
    ``latest_frame`` and ``frame_after``, and a ``click_at`` carrying the first capture's phash is refused."""
    d = FollowShot()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    before = await own(ex)
    d.read_script.extend([False, None, True])
    d.follow_script.append("loaded")
    # the first capture is a picture no frame of this executor ever showed (the other generation's gradient): the
    # fake's own picture repeats every 4 scene steps, and Ctrl+Enter's 4 key events would redraw the frame registered
    # before the action, which is rightly an own frame (reviewer of 312eecef)
    d.queue(fake_jpeg((W, H), 2, 1))
    since, shots_before = len(d.calls), len(d.returned)
    out = await act(ex, tool, args)
    assert out.status == "executed", out
    seq = names(d, since)
    shots = d.returned[shots_before:]
    assert len(shots) == 2, f"T-0203 B2: a re-capture after the late follow: {seq}"
    first = shots[0]
    assert first != shots[1], "premise: the follow moved the page"
    s1 = seq.index("screenshot", max(i for i, n in enumerate(seq) if n == "settle"))
    tail = [n for n in seq[s1:] if n in ("navigated_since", "follow_navigation", "screenshot")]
    assert tail == [
        "screenshot",
        "navigated_since",
        "follow_navigation",
        "navigated_since",
        "screenshot",
    ], seq
    assert ex.latest_frame.jpeg == shots[1], "B2: the kept capture is the second"
    assert out.resolved["frame_after"] == ex.latest_frame.phash
    assert out.resolved.get("navigated") is True, out.resolved
    fn = follow_notes(notes)
    assert [(n["result"], n["late"]) for n in fn] == [("loaded", True)], notes
    p1 = await base_phash(first)
    assert p1 != ex.latest_frame.phash
    assert p1 != before.phash, (
        "premise: the first capture is not the frame registered before the action"
    )
    stale = FrameGeometry(phash=p1, source=Source.WINDOW, width=W, height=H)
    refused = await ex.execute(
        ToolInvocation(
            call_id="stale",
            turn_id="t1",
            name="click_at",
            arguments=CLICK,
            source="draft",
            frame=stale,
        )
    )
    assert refused.status == "refused", (
        f"B2: the discarded first capture is not an own frame: {refused}"
    )


async def test_late_check_is_enter_only() -> None:
    """B2 / H-T203-3: a click (no Enter) that reads ``False`` makes no late read even if a navigation then starts;
    one capture."""
    d = FollowDriver()
    ex = build(d)
    await own(ex)
    d.read_script.extend([False, None, True])
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK)
    assert out.resolved.get("navigated") is False, out.resolved
    seq = names(d, since)
    assert seq.count("navigated_since") == 1 and seq.count("screenshot") == 1, seq
    assert "follow_navigation" not in seq


async def test_late_checkout_reads_are_ored() -> None:
    """B2.3: the late path makes a second checkout read and ``resolved["checkout"]`` is the OR of the two (the read
    drains, T-0192): a checkout seen by the first read and gone by the second is still reported."""
    d = FollowUrls()
    ex = build(d, web=True)
    await own(ex)
    d.read_script.extend([False, None, True])
    d.follow_script.append("loaded")
    d.visit_on("key_down", CHECKOUT)
    d.on_follow = lambda: d.visit(f"{SHOP}/after")
    since = len(d.calls)
    out = await ex.execute(call("press_key", None, combo="Enter"))
    assert out.status == "executed", out
    assert names(d, since).count("main_frame_urls") == 3, names(
        d, since
    )  # refusal check, first, second
    assert out.resolved.get("checkout") is True, out.resolved


async def test_late_recapture_raises_keeps_first() -> None:
    """B2: when the re-capture's first screenshot raises, the first capture is kept and registered, and the outcome is
    still ``executed``."""
    d = FollowShot()
    ex = build(d)
    await own(ex)
    first = fake_jpeg((W, H), 1, 7)
    d.queue(first, Boom("re-capture failed"))
    d.read_script.extend([False, None, True])
    d.follow_script.append("loaded")
    out = await act(ex, "type_at", TYPE_ENTER)
    assert out.status == "executed", out
    assert len(d.follow_log) == 1 and not d.script, (
        f"premise: the late follow ran and the re-capture was attempted: {names(d)}"
    )
    assert ex.latest_frame.jpeg == first, "B2: the first capture is kept"
    assert out.resolved["frame_after"] == ex.latest_frame.phash == await base_phash(first)


async def test_blank_recapture_after_follow(monkeypatch: pytest.MonkeyPatch) -> None:
    """B2.4: the late re-capture goes through ``_capture("action", deadline)``, so T-0200's blank re-capture applies:
    a blank first screenshot of the new page is taken again and the painted one kept (one T-0200 note, one follow
    note)."""
    monkeypatch.setattr(executor_mod, "BLANK_RECAPTURE_INTERVAL_S", 0.02)
    monkeypatch.setattr(executor_mod, "BLANK_RECAPTURE_ACTION_S", 0.3)
    d = FollowShot()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    first, painted = fake_jpeg((W, H), 1, 5), fake_jpeg((W, H), 1, 6)
    d.queue(first, WHITE, painted)
    d.read_script.extend([False, None, True])
    d.follow_script.append("loaded")
    out = await act(ex, "type_at", TYPE_ENTER)
    assert out.status == "executed", out
    assert ex.latest_frame.jpeg == painted, "B2.4: T-0200 re-captured the blank re-capture"
    assert [n["what"] for n in notes].count(BLANK_WHAT) == 1, notes
    blank = [n for n in notes if n["what"] == BLANK_WHAT]
    assert blank[0]["path"] == "action", (
        f"B2.4: the re-capture is _capture('action', deadline): {blank}"
    )
    assert [n["result"] for n in follow_notes(notes)] == ["loaded"], notes


# ============================================================================================ B3


async def test_follow_near_deadline_never_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """B3 (adversarial): with ``ACTION_TIMEOUT_S = 1.0`` and ``CAPTURE_RESERVE_S = 0.3`` (deadline 0.7 s) and a 0.5 s
    settle, the follow's budget is clipped to what is left before the deadline (at most ~0.2 s, not 3.0); a follow
    that spends its whole budget leaves the call ``executed`` within ``ACTION_TIMEOUT_S``, the capture inside the
    reserve. A settle past the deadline (0.75 s) gives a budget of at most 0 (if it is called at all) and is still
    ``executed``."""
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 1.0)
    monkeypatch.setattr(executor_mod, "CAPTURE_RESERVE_S", 0.3)
    for settle_s, bound in ((0.5, 0.2), (0.75, 0.0)):
        d = FollowSlowSettle()
        ex = build(d)
        await own(ex)
        d.settle_s = settle_s
        d.read_script.extend([None, None])
        d.follow_script.append(("budget", "timeout"))
        t0 = time.monotonic()
        out = await act(ex, "type_at", TYPE_ENTER)
        took = time.monotonic() - t0
        assert out.status == "executed", f"B3: settle {settle_s} s: {out}"
        assert took <= 1.0 + 0.1, f"B3: one execute() within ACTION_TIMEOUT_S: {took:.3f} s"
        for _mark, timeout_s, at in d.follow_log:
            left = 0.7 - (at - t0)
            assert timeout_s <= max(bound, left) + 0.02, (
                f"B1/B3: the follow's budget {timeout_s:.3f} s exceeds the time to the deadline ({left:.3f} s)"
            )
        if settle_s == 0.5:
            assert len(d.follow_log) == 1, "B1: the follow is called while time is left"


async def test_two_follows_share_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """B3: the after-settle follow and the late follow share ``NAV_FOLLOW_S`` (patched to 0.4 s): the first spends
    0.25 s and ends (re-read ``False``); the late check sees a new navigation and the second follow gets at most
    0.4 - 0.25 = 0.15 s (+ 20 ms slack). Two notes: ``ended`` (late ``False``), then ``late: True``."""
    monkeypatch.setattr(executor_mod, "NAV_FOLLOW_S", 0.4, raising=False)
    d = FollowDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.read_script.extend([None, False, None, True])
    d.follow_script.extend([("sleep", 0.25, "ended"), ("budget", "loaded")])
    out = await act(ex, "type_at", TYPE_ENTER)
    assert out.status == "executed", out
    assert len(d.follow_log) == 2, names(d)
    first, second = d.follow_log[0][1], d.follow_log[1][1]
    assert abs(first - 0.4) < 1e-6, f"B1: the first follow gets NAV_FOLLOW_S: {first}"
    assert second <= 0.4 - 0.25 + 0.02, (
        f"B3: the late follow gets what is left of 0.4 s: {second:.3f}"
    )
    fn = follow_notes(notes)
    assert [(n["result"], n["late"]) for n in fn] == [("ended", False), ("loaded", True)], notes
    assert out.resolved.get("navigated") is True, out.resolved


@pytest.mark.parametrize("late", [False, True], ids=["after-settle", "late"])
async def test_follow_raises_swallowed(late: bool) -> None:
    """B3: a ``follow_navigation`` that raises ``Exception`` is swallowed: the action is ``executed``, its after-frame
    captured and registered."""
    d = FollowDriver()
    ex = build(d)
    await own(ex)
    d.read_script.extend([False, None, None] if late else [None, None])
    d.follow_script.append(Boom("follow failed"))
    out = await act(ex, "type_at", TYPE_ENTER)
    assert out.status == "executed", out
    assert len(d.follow_log) == 1, names(d)
    assert out.resolved["frame_after"] == ex.latest_frame.phash


@pytest.mark.parametrize("late", [False, True], ids=["after-settle", "late"])
async def test_follow_cancel_registers_nothing(late: bool) -> None:
    """B4: a ``CancelledError`` in either follow propagates; nothing is registered and ``latest_frame`` is unchanged
    (on the late path the first capture was taken and is not kept)."""
    d = FollowShot()
    ex = build(d)
    before = await own(ex)
    d.read_script.extend([False, None] if late else [None])
    d.follow_script.append("hang")
    shots_before = len(d.returned)
    task = asyncio.ensure_future(act(ex, "type_at", TYPE_ENTER))
    try:
        await asyncio.wait_for(d.follow_entered.wait(), 2)
    except TimeoutError:
        task.cancel()
        raise AssertionError("T-0203 B4: follow_navigation was never awaited") from None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ex.latest_frame is before, "B4: latest_frame unchanged"
    if late:
        taken = d.returned[shots_before:]
        assert len(taken) == 1, "premise: the late path had captured once"
        p1 = await base_phash(taken[0])
        stale = FrameGeometry(phash=p1, source=Source.WINDOW, width=W, height=H)
        out = await ex.execute(
            ToolInvocation(
                call_id="c",
                turn_id="t1",
                name="click_at",
                arguments=CLICK,
                source="draft",
                frame=stale,
            )
        )
        assert out.status == "refused" or p1 == before.phash, (
            f"B4: the cancelled capture was registered: {out}"
        )


# ============================================================================================ B5


async def test_note_shape_and_absence() -> None:
    """B5: each follow that did not return ``"none"`` calls ``on_note`` with exactly
    ``{what, path, result, late, waited_ms}`` (``waited_ms`` an int near the follow's 0.1 s); none for ``"none"``; no
    URL, host or phash anywhere in it; a raising ``on_note`` is swallowed; ``resolved`` gains no key but
    ``navigated``."""
    d = FollowDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.read_script.extend([None, True])
    d.follow_script.append(("sleep", 0.1, "loaded"))
    out = await act(ex, "click_at", CLICK)
    fn = follow_notes(notes)
    assert len(fn) == 1 and len(notes) == 1, notes
    n = fn[0]
    assert n == {
        "what": NOTE_WHAT,
        "path": "action",
        "result": "loaded",
        "late": False,
        "waited_ms": n["waited_ms"],
    }
    assert 90 <= n["waited_ms"] <= 600, n
    text = json.dumps(n)
    assert (
        "http" not in text
        and "127.0.0.1" not in text
        and str(out.resolved["frame_after"]) not in text
    )

    want = await trace(base_module(), PlainNav, [("click_at", CLICK)])
    assert set(out.resolved) == set(want["outs"][0][3]) | {"navigated"}, (
        out.resolved,
        want["outs"][0][3],
    )

    notes.clear()
    d.read_script.extend([True, True])
    d.follow_script.append("none")
    await act(ex, "click_at", CLICK)
    assert notes == [], f"B5: no note for 'none': {notes}"

    def boom(_: dict[str, Any]) -> None:
        raise Boom("note sink failed")

    ex.on_note = boom
    d.read_script.extend([None, True])
    d.follow_script.append("timeout")
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed" and out.resolved.get("navigated") is True, out


# ============================================================================================ review (mutants)


async def test_late_recapture_hang_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """B2/B3 (reviewer M5/M16): the late re-capture's screenshot hangs (30 s). With ``ACTION_TIMEOUT_S = 1.0`` and
    ``CAPTURE_RESERVE_S = 0.5`` the re-capture is bounded inside the call: ``executed`` within ``ACTION_TIMEOUT_S``,
    and the first capture is kept (B2: a re-capture that does not complete keeps the first)."""
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 1.0)
    monkeypatch.setattr(executor_mod, "CAPTURE_RESERVE_S", 0.5)
    d = FollowSlowShot()
    ex = build(d)
    await own(ex)
    first = fake_jpeg((W, H), 2, 1)
    d.queue(first)
    d.read_script.extend([False, None, True])
    d.follow_script.append("loaded")
    d.arm(2)  # the action's second screenshot: the late re-capture
    t0 = time.monotonic()
    out = await act(ex, "type_at", TYPE_ENTER)
    took = time.monotonic() - t0
    assert out.status == "executed", f"B3: the late re-capture never fails the action: {out}"
    assert took <= 1.0 + 0.1, f"B3: one execute() within ACTION_TIMEOUT_S: {took:.3f} s"
    assert len(d.follow_log) == 1, "premise: the late follow ran"
    assert ex.latest_frame.jpeg == first, (
        "B2: the first capture is kept when the re-capture does not complete"
    )
    assert out.resolved["frame_after"] == ex.latest_frame.phash


async def test_late_follow_clipped_to_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """B2.1/B3 (reviewer M3b): the late follow gets what is left of the shared budget, *never past the deadline*: with
    a 0.6 s settle, ``ACTION_TIMEOUT_S = 1.0`` and ``CAPTURE_RESERVE_S = 0.3`` (deadline 0.7 s), a late follow that
    spends its whole budget gets at most ``deadline - now`` (+ 20 ms) and the call is ``executed`` within
    ``ACTION_TIMEOUT_S``."""
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 1.0)
    monkeypatch.setattr(executor_mod, "CAPTURE_RESERVE_S", 0.3)
    d = FollowSlowShot()
    ex = build(d)
    frame = await own(ex)
    d.settle_s = 0.6
    d.read_script.extend([False, None, True])
    d.follow_script.append(("budget", "loaded"))
    t0 = time.monotonic()
    out = await ex.execute(call("type_at", frame, **TYPE_ENTER))
    took = time.monotonic() - t0
    assert out.status == "executed", out
    assert took <= 1.0 + 0.1, f"B3: {took:.3f} s"
    for _mark, timeout_s, at in d.follow_log:
        left = 0.7 - (at - t0)
        assert timeout_s <= left + 0.02, (
            f"B2.1: the late follow's budget {timeout_s:.3f} s runs past the deadline ({left:.3f} s left)"
        )


async def test_late_read_that_raises_is_followed() -> None:
    """B2 (orchestrator ruling, reviewer M7b): the late check's read raising counts as "not ``False``": the executor
    follows (``late: True``) and captures again."""
    d = FollowShot()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.read_script.extend([False, "raise", True])
    d.follow_script.append("loaded")
    shots_before = len(d.returned)
    out = await act(ex, "type_at", TYPE_ENTER)
    assert out.status == "executed", out
    assert len(d.follow_log) == 1, f"ruling: a raising late read is followed: {names(d)}"
    assert len(d.returned) - shots_before == 2, "B2.4: the screen is captured again"
    assert [(n["result"], n["late"]) for n in follow_notes(notes)] == [("loaded", True)], notes
    assert out.resolved.get("navigated") is True, out.resolved


async def test_first_read_that_raises_is_followed() -> None:
    """B1 (reviewer M7): a first read that raises gives no ``bool``, so it is followed like ``None``; the re-read's
    ``True`` is the answer."""
    d = FollowDriver()
    ex = build(d)
    await own(ex)
    d.read_script.extend(["raise", True])
    d.follow_script.append("loaded")
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed", out
    assert len(d.follow_log) == 1, f"B1: a raising read is followed: {names(d)}"
    assert out.resolved.get("navigated") is True, out.resolved


async def test_follow_hanging_past_its_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """B3 (reviewer M10): a driver follow that ignores its ``timeout_s`` and hangs (A4's bound broken) does not hold
    the action: with ``NAV_FOLLOW_S = 0.3`` and ``ACTION_TIMEOUT_S = 3.0`` the call is ``executed`` (not ``failed`` or
    timed out) well before ``ACTION_TIMEOUT_S`` (within 0.3 s + 0.1 s slack + 0.6 s for the capture)."""
    monkeypatch.setattr(executor_mod, "NAV_FOLLOW_S", 0.3)
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 3.0)
    d = FollowDriver()
    ex = build(d)
    await own(ex)
    d.read_script.extend([None, None])
    d.follow_script.append("hang")
    t0 = time.monotonic()
    out = await act(ex, "click_at", CLICK)
    took = time.monotonic() - t0
    assert out.status == "executed", f"B3: a hanging follow never fails the action: {out}"
    assert took <= 0.3 + 0.1 + 0.6, f"B3: the follow is bounded by its own timeout: {took:.3f} s"
    assert "navigated" not in out.resolved, out.resolved
