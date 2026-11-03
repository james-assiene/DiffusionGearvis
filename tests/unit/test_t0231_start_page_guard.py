"""T-0231 group A: the T-0137 repeat guard never counts the session's own start page as a recurring screen; the
no-change repeat rule still applies there; every other page is judged as at BASE (docs/tasks/T-0231.md A1-A3).

**How the start page is known (R-A1, as ruled 2026-11-02 in docs/tasks/T-0231.md).** A1 says "by the URL or origin the executor reports, never by pixels".
The executor reports no URL to the session (T-0192: only the checkout boolean leaves its URL read), and
``computer/executor.py`` is not an owned file. What it does report, and what the rows script in the fake exactly as the
real web-mode executor does:

- ``press_key`` ``Ctrl+L``/``Ctrl+T`` executes as ``"opened the start page"`` (``executor.START_PAGE_ACTION``, a goto of
  ``start_page_url``) with ``resolved["navigated"] = True``;
- every other executed action carries ``resolved["navigated"]`` (``True`` for a typed address with Enter, ``False``
  for a click that stayed);
- a failed ``type_at`` (the focus check) dispatches nothing and recaptures; it never navigates.

``StartPageBrowser`` (below, a subclass of the T-0111g fake; the shared fake is not edited) also has the executor's
public ``start_page_url`` property and keeps ``url`` (a test control) in step. A session that reads any other name off
the browser gets the fake's named ``AssertionError``. Every turn starts on a site page (``PA``) except the 12:43 t6
replay, which starts on the start page as the logged turn did; whether the session can know that the *onset* page is
the start page is not pinned (nothing reports it), and the replay's verdicts do not depend on it (see its docstring).

**The A1/A3 tension (R-A3, as ruled: an optional argument or field marks start-page steps).** Recording a start-page step with ``screen=None`` and
leaving ``judge`` as it is would (a) disable the no-change repeat rule there (``same_action`` needs two equal non-None
screens), which A1 forbids, and (b) merge two visits of a site page that are separated only by start-page visits into
one run (``None`` is dropped before runs are collapsed), so ``A -> start -> A -> start -> A`` would no longer recur,
which A3 ("on every other page ... byte-identical to main's") forbids. The rows pin both behaviours (a) and (b) as
the criteria state them; how ``progress.py`` gets there (A2: an argument) is the implementer's.

**Oracles.** The frozen main rule is ``progress.py`` read at BASE with ``git show`` and executed as its own module
(``frozen``); the 12:43 t6 literals were hand-copied from the session log (calls, screens' executor phashes,
statuses) and the premise row proves that the frozen rule over them reproduces the logged refusals (steps 5, 7, 8, 9,
11, 12: ``recurring``). Reasons are typed from T-0137 A-1, never imported.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps; 20 for the
replay, as the logged ``loop20``), live, 1280x800 frames, T-0134c's label on, echo ``model``, ``repeat_guard=True``,
ReGround off; n = 1, deterministic. BASE = eb58ea00.

Not pending. Red at BASE: ``test_start_page_visits_never_stop`` and ``test_replay_1243_t6`` (named in each). The rest
are GUARD rows (green at BASE, must stay green): the no-change repeat on the start page, the site-page recurrence
across start-page visits, the pixel look-alike, the frozen-rule differential and the replay premise.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0137_session as g
from tests.unit._live_browser_support import FakeLiveBrowser, failed, live_frame
from tests.unit._tool_turn_support import answer, tool

#: BASE: main at eb58ea00 (``T-0232: evidence wording``), where T-0231 branched. A literal, never a moving ref.
BASE_SHA = "eb58ea00"
PROGRESS = "local_astra/conductor/progress.py"

Q = '<|"|>'
REPEAT_REASON = g.REPEAT_REASON
RECURRING_REASON = g.RECURRING_REASON
#: the executor's web-mode start page (``start_page_url_for(origin, web=True)``) and its action string, typed by hand
START_URL = "http://127.0.0.1:8765/web.html"
START_ACTION = "opened the start page"
FOCUS_REASON = "could not check the field before typing; nothing was typed"

#: executor phashes: a site page, the start page, two other site pages
PA, PS, PB, PC = "a1a1a1a1a1a1a1a1", "5ea75ea75ea75ea7", "b2b2b2b2b2b2b2b2", "c3c3c3c3c3c3c3c3"


def press(combo: str) -> str:
    return f"<|tool_call>call:press_key{{combo:{Q}{combo}{Q}}}<tool_call|>"


def type_call(x: int, y: int, text: str, enter: bool) -> str:
    e = "true" if enter else "false"
    return f"<|tool_call>call:type_at{{press_enter:{e},text:{Q}{text}{Q},x:{x},y:{y}}}<tool_call|>"


def click(x: int, y: int) -> str:
    return f"<|tool_call>call:click_at{{x:{x},y:{y}}}<tool_call|>"


def search(query: str) -> str:
    return f"<|tool_call>call:google_search{{query:{Q}{query}{Q}}}<tool_call|>"


CTRL_L = press("Ctrl+L")
WAIT = "<|tool_call>call:wait{}<tool_call|>"


# --------------------------------------------------------------------------------------------- the fake


class StartPageBrowser(FakeLiveBrowser):
    """The T-0111g fake with the web-mode executor's start-page reports (module docstring, R-A1). ``plan`` maps a call
    id to what the call does: ``("start",)`` (Ctrl+L: the start page), ``("go", url, frame)`` (navigates to ``url``,
    post-action capture ``frame``), ``("stay", frame)`` (executes, ``navigated: False``, capture ``frame``),
    ``("focus_fail", frame)`` (``failed`` with the focus reason, recapture ``frame``), ``("wait", frame)``."""

    def __init__(self, onset: Any, *, url: str, start_frames: list[Any]) -> None:
        super().__init__([onset], echo="model")
        self.url = url
        self.start_frames = list(start_frames)
        self.plan: dict[str, tuple[Any, ...]] = {}

    @property
    def start_page_url(self) -> str:
        """The executor's public ``start_page_url`` (web mode)."""
        return START_URL

    def _post(self, frame: Any) -> Any:
        self.post_frames.appendleft(frame)
        return self.capture_post()

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        step = self.plan.get(inv.call_id)
        if step is None or inv.name == "google_search":
            return await super().execute(inv)
        self.calls.append(("execute", inv.call_id))
        self.invocations.append(inv)
        kind = step[0]
        if kind == "start":
            after = self._post(self.start_frames.pop(0))
            self.url = START_URL
            return ToolOutcome(
                call_id=inv.call_id,
                status="executed",
                action=START_ACTION,
                reason=None,
                resolved={"navigated": True, "frame_after": after.phash},
            )
        if kind == "focus_fail":
            self._post(step[1])
            return failed(inv, FOCUS_REASON)
        if kind == "wait":
            after = self._post(step[1])
            return ToolOutcome(
                call_id=inv.call_id,
                status="executed",
                action="waited 1 s",
                reason=None,
                resolved={"frame_after": after.phash},
            )
        frame = step[-1]
        self.post_frames.appendleft(frame)
        out = self.perform(inv)
        if out.status != "executed":
            return out
        if kind == "go":
            self.url = step[1]
        return out.model_copy(update={"resolved": {**out.resolved, "navigated": kind == "go"}})


def site(tag: str, phash: str) -> Any:
    return live_frame(tag, phash=phash)


def start(tag: str) -> Any:
    return live_frame(tag, phash=PS)


async def guarded_turn(
    tmp: Path,
    browser: StartPageBrowser,
    calls: list[str],
    *,
    final: str = "Here it is.",
    kit: Any = None,
    max_steps: int = 8,
) -> Any:
    """One turn with ``repeat_guard=True``: step k makes ``calls[k-1]``, then ``final`` (an answer)."""
    saved = tsl.LOOP
    tsl.LOOP = tsl.loop_policy(max_steps)
    try:
        steps = [[answer(f"Step {i}."), tool(c)] for i, c in enumerate(calls[1:], 2)]
        steps.append([answer(final)])
        async with g.guard_rig(
            tmp, browser, script=[answer("Step 1."), tool(calls[0])], kit=kit
        ) as rig:
            await g.run(rig, tuple(steps), limit_ms=60_000)
    finally:
        tsl.LOOP = saved
    return rig


def cid(k: int) -> str:
    return tsl.call_id(k)


def no_progress(rig: Any) -> list[tuple[Any, Any]]:
    return [
        (r.get("call_id"), r.get("no_progress")) for r in g.all_records(rig) if "no_progress" in r
    ]


# =====================================================================================================
# A1: the start page never recurs
# =====================================================================================================


async def test_start_page_visits_never_stop(tmp_path: Path) -> None:
    """A1: from site page A, Ctrl+L, type an address, Ctrl+L, type another, Ctrl+L, and type a third address on the
    start page (its third visit in the request). The third typing executes; nothing in the turn is stopped as
    ``recurring``; the turn line counts no stops. Red at BASE: step 6 is refused with the recurring reason (the start
    page's phash has 3 runs: steps 2, 4, 6)."""
    b = StartPageBrowser(
        site("pa", PA), url="https://a.example/", start_frames=[start(f"s{i}") for i in range(3)]
    )
    b.plan = {
        cid(1): ("start",),
        cid(2): ("go", "https://b.example/", site("pb", PB)),
        cid(3): ("start",),
        cid(4): ("go", "https://c.example/", site("pc", PC)),
        cid(5): ("start",),
        cid(6): ("go", "https://d.example/", site("pd", "d4d4d4d4d4d4d4d4")),
    }
    calls = [
        CTRL_L,
        type_call(488, 451, "b.example", True),
        CTRL_L,
        type_call(488, 451, "c.example", True),
        CTRL_L,
        type_call(488, 451, "d.example", True),
    ]
    rig = await guarded_turn(tmp_path, b, calls)
    assert g.statuses(rig, cid(6)) == ["executed"], (
        f"T-0231 A1: an action on the start page is never stopped as recurring; step 6 {g.statuses(rig, cid(6))}, "
        f"card {g.card(rig, cid(6))}"
    )
    assert g.executes(b, cid(6)) == 1
    assert no_progress(rig) == [], f"T-0231 A1: no guard stop in the turn; {no_progress(rig)}"
    line = g.check_line_shape(g.guard_line(rig))
    assert (line["stops_repeat"], line["stops_recurring"]) == (0, 0), line


async def test_no_change_repeat_still_stopped_on_start_page(tmp_path: Path) -> None:
    """A1 (GUARD, too wide: the start page is not exempt from A-4): Ctrl+L from site page A, then the same address typed
    without Enter at the same place on the start page, each time with no change (the post-action capture is the same
    capture). Typings 1-3 execute; the fourth is refused with T-0137's repeat reason (``no_progress: "repeat"``) and
    never dispatched. Green at BASE; red for an implementation that only records the start page as ``screen=None``."""
    s0 = start("s0")
    b = StartPageBrowser(site("pa", PA), url="https://a.example/", start_frames=[s0])
    typed = type_call(488, 451, "amazon.com", False)
    b.plan = {cid(1): ("start",)} | {cid(k): ("stay", s0.model_copy()) for k in (2, 3, 4, 5)}
    rig = await guarded_turn(tmp_path, b, [CTRL_L, typed, typed, typed, typed])
    assert [g.statuses(rig, cid(k)) for k in (2, 3, 4)] == [["executed"]] * 3, "premise"
    assert g.statuses(rig, cid(5)) == ["refused"], (
        f"T-0231 A1: the no-change repeat rule still applies on the start page; step 5 {g.statuses(rig, cid(5))}"
    )
    assert g.card(rig, cid(5)).reason == REPEAT_REASON, g.card(rig, cid(5))
    assert g.executes(b, cid(5)) == 0
    assert no_progress(rig) == [(cid(5), "repeat")], no_progress(rig)


async def test_site_page_recurring_across_start_page_visits_still_stopped(tmp_path: Path) -> None:
    """A3 (GUARD, too wide; R-A3 as ruled): site page A, Ctrl+L, an address back to A, Ctrl+L, another address back to
    A, then a click on A (A's third visit, each separated only by a start-page visit). The click is refused as
    ``recurring``, as at BASE (A's verdict is byte-identical to main's: visits 3). Green at BASE; red for an
    implementation whose start-page steps drop out and merge A's runs."""
    b = StartPageBrowser(
        site("pa0", PA), url="https://a.example/", start_frames=[start("s0"), start("s1")]
    )
    b.plan = {
        cid(1): ("start",),
        cid(2): ("go", "https://a.example/", site("pa1", PA)),
        cid(3): ("start",),
        cid(4): ("go", "https://a.example/?q=2", site("pa2", PA)),
        cid(5): ("stay", site("pa3", PA)),
    }
    calls = [
        CTRL_L,
        type_call(488, 451, "a.example", True),
        CTRL_L,
        type_call(488, 451, "a.example/?q=2", True),
        click(300, 300),
    ]
    rig = await guarded_turn(tmp_path, b, calls)
    assert g.statuses(rig, cid(5)) == ["refused"], (
        f"T-0231 A3: a site page's third visit is still recurring; step 5 {g.statuses(rig, cid(5))}"
    )
    assert g.card(rig, cid(5)).reason == RECURRING_REASON, g.card(rig, cid(5))
    assert no_progress(rig) == [(cid(5), "recurring")], no_progress(rig)
    assert g.check_line_shape(g.guard_line(rig))["max_screen_visits"] == 3


async def test_lookalike_of_start_page_still_counts(tmp_path: Path) -> None:
    """A1 "never by pixels" (GUARD, adversarial): after one real start-page visit, a site page whose frame has the
    start page's executor phash (a blank look-alike, reached by typing an address) comes back three times; its third
    visit (step 7; the first was step 3, right after the real start page at step 2) is refused as ``recurring``. Green
    at BASE; red for an implementation that learns the start page's phash."""
    b = StartPageBrowser(site("pa", PA), url="https://a.example/", start_frames=[start("s0")])
    b.plan = {
        cid(1): ("start",),
        cid(2): ("go", "https://l.example/", site("look-1", PS)),
        cid(3): ("go", "https://b.example/", site("pb1", PB)),
        cid(4): ("go", "https://l.example/", site("look-2", PS)),
        cid(5): ("go", "https://b.example/", site("pb2", PB)),
        cid(6): ("go", "https://l.example/", site("look-3", PS)),
        cid(7): ("stay", site("look-4", PS)),
    }
    calls = [
        CTRL_L,
        type_call(488, 451, "l.example", True),
        click(100, 100),
        click(200, 200),
        click(150, 500),
        click(250, 250),
        click(400, 400),
    ]
    rig = await guarded_turn(tmp_path, b, calls)
    assert g.statuses(rig, cid(7)) == ["refused"], (
        f"T-0231 A1: a look-alike page is counted (the start page is known by URL, not pixels); "
        f"step 7 {g.statuses(rig, cid(7))}"
    )
    assert g.card(rig, cid(7)).reason == RECURRING_REASON
    assert no_progress(rig) == [(cid(7), "recurring")], no_progress(rig)


# =====================================================================================================
# A3: the 2026-11-02 12:43 t6 replay
# =====================================================================================================

#: executor phashes copied from logs/sessions/20261102-124323-2472863.jsonl, turn t6 (``resolved.frame_after``)
BD1 = "bd65546e7b221249"  # the start page after Ctrl+L (and the turn's onset frame)
BD2 = "bd65546efb229249"  # the start page recaptured after the failed focus check
F2C = "f2c0a23c392ecce0"  # amazon.com after the step-1 address
AMAZON_URL = "https://www.amazon.com/Grand-Theft-Auto-VI-PlayStation-Delivers/dp/B0H6K928WL"
Q1 = "pre-order gta vi for ps5 on amazon"
Q2 = "amazon pre-order gta vi ps5"
#: (step, name, arguments, screen of the step's request, logged status, logged no_progress), hand-copied
T6: list[tuple[int, str, dict[str, Any], str, str, str | None]] = [
    (
        1,
        "type_at",
        {"press_enter": True, "text": "amazon.com", "x": 488, "y": 451},
        BD1,
        "executed",
        None,
    ),
    (2, "press_key", {"combo": "Ctrl+L"}, F2C, "executed", None),
    (
        3,
        "type_at",
        {"press_enter": True, "text": "amazon.com", "x": 500, "y": 500},
        BD1,
        "failed",
        None,
    ),
    (4, "press_key", {"combo": "Ctrl+L"}, BD2, "executed", None),
    (
        5,
        "type_at",
        {"press_enter": True, "text": "amazon.com", "x": 500, "y": 500},
        BD1,
        "refused",
        "recurring",
    ),
    (6, "google_search", {"query": Q1}, BD1, "executed", None),
    (7, "click_at", {"x": 750, "y": 650}, BD1, "refused", "recurring"),
    (8, "click_at", {"x": 700, "y": 600}, BD1, "refused", "recurring"),
    (9, "click_at", {"x": 600, "y": 550}, BD1, "refused", "recurring"),
    (10, "google_search", {"query": Q2}, BD1, "executed", None),
    (
        11,
        "type_at",
        {"press_enter": True, "text": AMAZON_URL, "x": 500, "y": 500},
        BD1,
        "refused",
        "recurring",
    ),
    (12, "press_key", {"combo": "Ctrl+L"}, BD1, "refused", "recurring"),
    (13, "wait", {}, BD1, "executed", None),
    (14, "google_search", {"query": Q2}, BD1, "executed", None),
    (15, "google_search", {"query": Q2}, BD1, "executed", None),
]
STOPPED_AT_BASE = [5, 7, 8, 9, 11, 12]


def _call_text(name: str, args: Mapping[str, Any]) -> str:
    if name == "type_at":
        return type_call(args["x"], args["y"], args["text"], args["press_enter"])
    if name == "press_key":
        return press(args["combo"])
    if name == "click_at":
        return click(args["x"], args["y"])
    if name == "google_search":
        return search(args["query"])
    assert name == "wait"
    return WAIT


def frozen() -> types.ModuleType:
    """``progress.py`` at BASE, executed as its own module (the frozen main rule)."""
    gr.require_commits(BASE_SHA)
    source = gr._git("show", f"{BASE_SHA}:{PROGRESS}").stdout
    assert "def judge(" in source, "premise: BASE's progress.py"
    name = "t0231_frozen_progress"
    mod = types.ModuleType(name)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    try:
        exec(compile(source, f"{BASE_SHA}:{PROGRESS}", "exec"), mod.__dict__)
    finally:
        del sys.modules[name]
    return mod


def test_t6_literals_reproduce_the_logged_refusals_under_the_main_rule() -> None:
    """Premise of the replay (GUARD): the frozen main rule over the hand-copied t6 history refuses exactly the logged
    steps (5, 7, 8, 9, 11, 12) as ``recurring`` and lets every other browser action through, so the literals are the
    logged step sequence."""
    m = frozen()
    history: list[Any] = []
    stopped = []
    for step, name, args, screen, status, logged in T6:
        v = m.judge(history, name, args, screen, overview=False)
        if name in m.ACTIONS and step >= 2 and v.stop is not None:
            stopped.append(step)
            assert v.stop == logged, (step, v)
        else:
            assert logged is None, (step, v)
        history.append(
            m.Step(
                step=step,
                name=name,
                arguments=args,
                screen=screen,
                status=status,
                overview=False,
                no_change=None,
                outcome={},
                action=None,
            )
        )
    assert stopped == STOPPED_AT_BASE, stopped


async def test_replay_1243_t6(tmp_path: Path, monkeypatch: Any) -> None:
    """A3: the 12:43 t6 step sequence through the session (calls and executor phashes as logged; the turn starts on
    the start page, frame ``BD1``). Each action the guard refused at BASE now reaches the executor and gets the
    outcome the identical call got on the identical page in the log: a typed address on the start page fails the
    focus check (as step 3 did), a click there stays, Ctrl+L opens the start page, ``wait`` waits. The replay ends
    with no ``recurring`` (or any) refusal, every browser call dispatched, and zero stops on the turn line.

    Whether the onset page is known as the start page does not matter: either way every later step on the start page
    is exempt, and the only other screen (``F2C``) is visited once. Red at BASE: step 5 is refused as ``recurring``
    (``BD1``'s third run), and the five steps the log shows after it."""
    from tests.fakes.search import FakeSearch
    from tests.unit._tool_turn_support import with_search
    from tests.unit.test_t0202_session_wait import wait_kit

    onset = start("t6-onset").model_copy(update={"phash": BD1})
    b = StartPageBrowser(
        onset,
        url=START_URL,
        start_frames=[live_frame(f"t6-start-{k}", phash=BD1) for k in (2, 4, 12)],
    )
    recapture = live_frame("t6-start-recapture", phash=BD2)
    b.plan = {
        cid(1): ("go", "https://www.amazon.com/", live_frame("t6-amazon", phash=F2C)),
        cid(2): ("start",),
        cid(3): ("focus_fail", recapture),
        cid(4): ("start",),
        cid(5): ("focus_fail", recapture.model_copy()),
        cid(7): ("stay", recapture.model_copy()),
        cid(8): ("stay", recapture.model_copy()),
        cid(9): ("stay", recapture.model_copy()),
        cid(11): ("focus_fail", recapture.model_copy()),
        cid(12): ("start",),
        cid(13): ("wait", live_frame("t6-start-13", phash=BD1)),
    }
    kit = with_search(wait_kit(b, tmp_path / "kit", monkeypatch), FakeSearch())
    calls = [_call_text(name, args) for _, name, args, *_ in T6]
    rig = await guarded_turn(
        tmp_path / "run", b, calls, final="I could not open the page.", kit=kit, max_steps=20
    )
    assert no_progress(rig) == [], (
        f"T-0231 A3: the t6 replay ends with no guard refusal on the start page; {no_progress(rig)}"
    )
    for k in STOPPED_AT_BASE:
        assert g.executes(b, cid(k)) == 1, (
            f"T-0231 A3: step {k} ({T6[k - 1][1]}) reaches the executor; statuses {g.statuses(rig, cid(k))}"
        )
        assert g.card(rig, cid(k)).reason not in (RECURRING_REASON, REPEAT_REASON)
    line = g.check_line_shape(g.guard_line(rig))
    assert (line["stops_repeat"], line["stops_recurring"]) == (0, 0), line


# =====================================================================================================
# A3: every other page is judged as at BASE (differential against the frozen main rule)
# =====================================================================================================

NAMES = ("click_at",) * 5 + ("type_at", "scroll", "press_key", "google_search", "wait")
STATUSES = ("executed",) * 5 + ("refused", "failed", "timeout")
SCREENS = ("A", "A", "B", "C", None)


def _args(name: str, k: int) -> dict[str, Any]:
    xy = [(100, 100), (106, 109), (300, 100)][k]
    if name == "click_at":
        return {"x": xy[0], "y": xy[1]}
    if name == "type_at":
        return {"x": xy[0], "y": xy[1], "text": ["a", "a", "b"][k], "press_enter": k == 2}
    if name == "scroll":
        return {"direction": ["down", "down", "up"][k]}
    if name == "press_key":
        return {"combo": ["Ctrl+L", "Ctrl+L", "Enter"][k]}
    if name == "google_search":
        return {"query": "q"}
    return {}


raw_step = st.tuples(
    st.sampled_from(NAMES),
    st.sampled_from((0, 0, 1, 2)),
    st.sampled_from(SCREENS),
    st.sampled_from(STATUSES),
    st.sampled_from((False,) * 5 + (True,)),
    st.sampled_from((True, True, False, None)),
)
raw_hist = st.one_of(
    st.lists(raw_step, max_size=10),
    st.lists(raw_step, min_size=1, max_size=3).flatmap(lambda xs: st.permutations(xs * 3)),
)


def _steps(cls: Any, raw: list[tuple[Any, ...]]) -> list[Any]:
    return [
        cls(
            step=i,
            name=name,
            arguments=_args(name, k),
            screen=screen,
            status=status,
            overview=ov,
            no_change=nc,
            outcome={} if nc is None else {"screenshot": "unchanged" if nc else "updated"},
            action=f"{name} {k}",
        )
        for i, (name, k, screen, status, ov, nc) in enumerate(raw, 1)
    ]


@pytest.fixture(scope="module")
def main_rule() -> types.ModuleType:
    return frozen()


@settings(max_examples=300, deadline=None)
@given(
    raw=raw_hist,
    call=st.tuples(
        st.sampled_from(NAMES), st.sampled_from((0, 1, 2)), st.sampled_from(SCREENS), st.booleans()
    ),
)
def test_other_pages_judged_as_main(
    main_rule: types.ModuleType, raw: list[Any], call: tuple[Any, ...]
) -> None:
    """A3 (GUARD): with no start page in the turn, ``progress.judge`` (called as at BASE) gives the frozen main rule's
    verdict (``repeat_of``, ``stop``, ``no_change_repeats``, ``screen_visits``) on random histories over three screens
    and ``None``, every status, ``no_change`` value and overview flag; the module constants and reasons are BASE's."""
    from local_astra.conductor import progress

    name, k, screen, ov = call
    want = main_rule.judge(_steps(main_rule.Step, raw), name, _args(name, k), screen, overview=ov)
    got = progress.judge(_steps(progress.Step, raw), name, _args(name, k), screen, overview=ov)
    assert (got.repeat_of, got.stop, got.no_change_repeats, got.screen_visits) == (
        want.repeat_of,
        want.stop,
        want.no_change_repeats,
        want.screen_visits,
    ), (raw, call)
    for const in (
        "REPEAT_TOLERANCE",
        "MAX_NO_CHANGE_REPEATS",
        "MAX_SCREEN_VISITS",
        "ACTIONS",
        "NO_PROGRESS_REPEAT_REASON",
        "NO_PROGRESS_RECURRING_REASON",
    ):
        assert getattr(progress, const) == getattr(main_rule, const), const


# =====================================================================================================
# R-A3: the new ``start_page`` argument and field, set (review: the differential above only drives the defaults)
# =====================================================================================================


def _p(step: int, screen: str | None, *, start_page: bool = False) -> Any:
    from local_astra.conductor import progress

    return progress.Step(
        step=step,
        name="click_at",
        arguments={"x": 100 * step, "y": 100 * step},
        screen=screen,
        status="executed",
        overview=False,
        no_change=False,
        outcome={"screenshot": "updated"},
        action=f"step {step}",
        start_page=start_page,
    )


def test_proposed_on_start_page_has_no_visits_even_after_unmarked_runs() -> None:
    """R-A3 (pure; kills ``judge`` ignoring its ``start_page`` argument): the start page's phash ``PS`` has two
    *unmarked* runs in the history (the onset, and a return the session could not see as the start page), then a
    marked start-page step. A call proposed with ``start_page=True`` on ``PS`` has ``screen_visits == 0`` and is not
    stopped (by hand: unmarked it is PS, PB, PS, |, PS = 3 runs, ``recurring``)."""
    from local_astra.conductor import progress

    history = [_p(1, PS), _p(2, PB), _p(3, PS), _p(4, PS, start_page=True)]
    call = {"x": 900, "y": 900}
    v = progress.judge(history, "click_at", call, PS, overview=False, start_page=True)
    assert (v.screen_visits, v.stop) == (0, None), (
        f"T-0231 R-A3: a proposed start-page call has no visits and is never recurring; {v}"
    )
    unmarked = progress.judge(history, "click_at", call, PS, overview=False)
    assert (unmarked.screen_visits, unmarked.stop) == (3, "recurring"), (
        f"premise: the same call unmarked is the third run; {unmarked}"
    )


def test_none_screen_start_page_steps_still_split_runs() -> None:
    """R-A3 (pure; kills checking ``screen is None`` before the start-page split): A, a start-page step with no
    screen, A, another such step, then A proposed: three visits of A and ``recurring`` (by hand: each start-page step
    ends the run before it), never one merged run."""
    from local_astra.conductor import progress

    history = [_p(1, PA), _p(2, None, start_page=True), _p(3, PA), _p(4, None, start_page=True)]
    v = progress.judge(history, "click_at", {"x": 900, "y": 900}, PA, overview=False)
    assert (v.screen_visits, v.stop) == (3, "recurring"), (
        f"T-0231 R-A3: A, start, A, start, A is three visits of A; {v}"
    )


async def test_session_passes_start_page_after_an_unmarked_return(tmp_path: Path) -> None:
    """R-A1/R-A3 (session; kills ``_guard_stopped`` not passing ``start_page=True``): the turn starts on the start page
    (onset ``PS``, unmarked: nothing reports it), types an address (``PB``), clicks a link back to the start page
    (``navigated: True`` but not the start-page action, so unmarked, ``PS``), clicks to ``PB``, presses Ctrl+L (the
    start page, marked) and types an address there. That typing executes: its call is on the start page, so it has
    no visits, although ``PS`` already has two unmarked runs (unmarked it would be the third, ``recurring``)."""
    b = StartPageBrowser(start("onset"), url=START_URL, start_frames=[start("s4")])
    b.plan = {
        cid(1): ("go", "https://b.example/", site("pb1", PB)),
        cid(2): ("go", START_URL, start("back")),
        cid(3): ("go", "https://b.example/", site("pb2", PB)),
        cid(4): ("start",),
        cid(5): ("go", "https://c.example/", site("pc", PC)),
    }
    calls = [
        type_call(488, 451, "b.example", True),
        click(100, 100),
        click(200, 200),
        CTRL_L,
        type_call(488, 451, "c.example", True),
    ]
    rig = await guarded_turn(tmp_path, b, calls)
    assert g.statuses(rig, cid(5)) == ["executed"], (
        f"T-0231 R-A3: a call on the start page is never recurring; step 5 {g.statuses(rig, cid(5))}"
    )
    assert no_progress(rig) == [], no_progress(rig)


async def test_marked_history_steps_add_no_visit_to_a_lookalike(tmp_path: Path) -> None:
    """R-A3 (session; kills settled start-page steps recorded unmarked): three real start-page visits (steps 2, 4, 6,
    each reached by Ctrl+L), then step 6 types an address that lands on a site page with the start page's phash (a
    look-alike, not the start page). The click on it at step 7 executes: the marked start-page steps add no visit, so
    this is the look-alike's first. Recorded unmarked, ``PS`` would have three runs (2, 4, 6+7) and step 7 would be
    ``recurring``."""
    b = StartPageBrowser(
        site("pa", PA), url="https://a.example/", start_frames=[start(f"s{i}") for i in range(3)]
    )
    b.plan = {
        cid(1): ("start",),
        cid(2): ("go", "https://b.example/", site("pb", PB)),
        cid(3): ("start",),
        cid(4): ("go", "https://c.example/", site("pc", PC)),
        cid(5): ("start",),
        cid(6): ("go", "https://l.example/", site("look", PS)),
        cid(7): ("stay", site("look-2", PS)),
    }
    calls = [
        CTRL_L,
        type_call(488, 451, "b.example", True),
        CTRL_L,
        type_call(488, 451, "c.example", True),
        CTRL_L,
        type_call(488, 451, "l.example", True),
        click(400, 400),
    ]
    rig = await guarded_turn(tmp_path, b, calls)
    assert g.statuses(rig, cid(7)) == ["executed"], (
        f"T-0231 R-A3: start-page steps add no visit to a look-alike; step 7 {g.statuses(rig, cid(7))}"
    )
    assert no_progress(rig) == [], no_progress(rig)
