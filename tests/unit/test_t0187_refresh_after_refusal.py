"""T-0187: a "take a new frame" refusal is followed by a fresh frame at the next step (docs/tasks/T-0187.md).

The live tool loop over the fakes: ``feedback_rig`` (``_live_browser_support``; the session built with
``live_feedback=<screen label on>`` and ``action_outcome=True``, so a wrong ``LABEL_UPDATED`` or an outcome clause is
visible) over a ``FakeLiveBrowser``, a scripted ``FakeLlm`` and the ``ScriptedClock``. Criterion 3's reproduction runs
twice on CPU here (a fake that models the executor's page-serial rule, and the real ``BrowserToolExecutor`` over
``test_computer_executor``'s ``FakeDriver``) and once with real Chromium in
``tests/integration/test_t0187_late_popup_session.py``.

**Expected values.** The four "take a new frame" texts and their constant names are typed here from criterion 2 (and
checked once against the modules that define them); the labels are T-0134's literals (``test_t0134_label``); image URLs
are the real ``PromptBuilder``'s encoding of the frame (``builder_url``); the step tails are
``loop.step_prompt``'s rendering of hand-written payloads (``test_session_live.want_tail``/``status``). The refused
step's ``screenshot`` value is pinned to T-0177 B-R3 (merged at 2f955738 for ``SHARED_FRAME_REASON``):
``"updated"`` when a fresh frame replaced the image, ``"unchanged"`` when none could be taken.

**The fake page change.** ``RefreshBrowser`` (local to this module) adds to ``FakeLiveBrowser`` the executor's
criterion-15 rule as merged (ruling 683cc1e): a popup that opened after the last settle is adopted by the next
``execute()``/``frame()``; an adopting ``execute()`` refuses its action with the page-changed reason; a
``click_at``/``type_at`` carrying a frame of an earlier page is refused with it too. The real executor's side of that
contract is pinned by ``tests/unit/test_computer_executor.py`` (``test_adoption_in_ensure_refuses_the_pending_action``,
``test_openers_frame_after_a_newer_own_frame_is_refused``) and ``tests/integration/test_computer_browser_e2e.py``
(``test_late_popup_refuses_the_openers_stale_frame``, real Chromium); ``test_real_executor_*`` below runs the real
executor itself.

**Readings** (reported to the orchestrator, not guessed silently):

- Criterion 7's ``"tool_step": k`` is the refused step's number (criterion 1 calls the refused step k and the
  refreshed one k+1). The note is pinned as exactly the three keys the criterion writes.
- Criterion 2 names no constant. ``test_one_named_constant_holds_the_four_reasons`` finds it by value: exactly one
  module-level constant of ``session.py`` whose texts are the four (a set/tuple/list of them, or a mapping whose keys
  or values are them).
- Criterion 4 "nothing else changes": the step after a refusal with no fresh frame is exactly BASE's (the image
  stays, ``LABEL_UNCHANGED``, payload ``"unchanged"``).
- Criterion 5 "the T-0172 refusals": T-0172 adds no refusal reason to the loop (its re-ask makes a call or does not);
  no row is written for it.
- Criterion 6's differential runs three regimes against main at 2f955738 (BASE, which already has T-0162 C and T-0177
  A/B): tools ``off``, ``dry_run`` with the executor refusing with the page-changed text, and ``live`` with a refusal
  outside the set. ``_t0162c_pins.observed()`` is not applied: its strips are declared differences of tasks older than
  BASE, so applying it to one side would make the row red at BASE; nothing is normalised beyond the session id, the
  tmp dir, the log path and the git commit.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools live (loop on, 8 steps), 1280x800
frames, n = 1, deterministic.
"""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.llm import LlmOptions
from local_astra.contracts.perception import Frame
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.fakes.search import FakeSearch
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_session_web_search as wsw
from tests.unit._live_browser_support import (
    NOT_OWN,
    FakeLiveBrowser,
    b64,
    builder_url,
    data_url,
    failed,
    feedback_rig,
    hud_frame,
    live_feedback,
    live_frame,
    live_frame_timeout_ms,
    live_kit,
    refused,
    session_mod,
    turn_logs,
)
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool, tool_rig
from tests.unit.test_session_live import (
    ASK,
    NO,
    _wire,
    card,
    last_user,
    status,
    step_tail,
    want_tail,
)
from tests.unit.test_t0134_label import L1, LN, LU, label_after

#: Activated at /implement T-0187: the positive rows (criteria 1, 2, 3, 4, 7 and R4, and the criterion-6 repeat row
#: whose premise is the refresh) are red at BASE; the negatives of criteria 5 and 6 and the differential are green at
#: BASE by design.

Q = tsl.Q
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
CLICK2 = tsl.CLICK2
CLICK_ARGS, CLICK2_ARGS = tsl.CLICK_ARGS, tsl.CLICK2_ARGS
SEARCH = tsl.SEARCH_CALL

#: criterion 2, typed from the task and the modules' merged texts (never imported for the oracle)
PAGE_CHANGED = "the active page changed; take a new frame"
OVERVIEW_STALE = "the screen changed since the whole-page picture; take a new frame"
OVERVIEW_PAGE_CHANGED = "the page changed since the whole-page picture; take a new frame"
FRESHNESS = "the screen changed since this frame; take a new frame"
#: criterion 7's ``reason_kind``: the constant name of each
KIND = {
    PAGE_CHANGED: "PAGE_CHANGED_REASON",
    OVERVIEW_STALE: "OVERVIEW_STALE",
    OVERVIEW_PAGE_CHANGED: "OVERVIEW_PAGE_CHANGED",
    FRESHNESS: "FRESHNESS_REASON",
}
FOUR = frozenset(KIND)
#: criterion 7's note and B8's no-frame note (T-0111g, as merged)
REFRESHED = "live frame: refreshed after a page-change refusal"
NO_FRESH = "live frame: no fresh frame for the next step; the previous image stays"
#: T-0136 B-3's clause stem (typed; a substring of no other rig text)
CLAUSE_STEM = "Compared with the previous screenshot"

TYPE_HI = f"<|tool_call>call:type_at{{press_enter:false,text:{Q}hello{Q},x:500,y:250}}<tool_call|>"
PRESS = f"<|tool_call>call:press_key{{combo:{Q}Ctrl+C{Q}}}<tool_call|>"
SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
OV_CALL = "<|tool_call>call:page_overview{}<tool_call|>"
#: criterion 1's four actions: (the call, the tool name, its arguments as parsed)
ACTIONS = {
    "click_at": (VALID_CLICK, "click_at", CLICK_ARGS),
    "type_at": (
        TYPE_HI,
        "type_at",
        {"press_enter": False, "text": "hello", "x": 500, "y": 250},
    ),
    "press_key": (PRESS, "press_key", {"combo": "Ctrl+C"}),
    "scroll": (SCROLL, "scroll", {"direction": "down"}),
}
#: the fake's action for VALID_CLICK (500, 250 on 1280x800) and CLICK2 (120, 640), echo "pixels"
ACT1, ACT2 = "clicked at (640, 200)", "clicked at (153, 512)"

F0 = live_frame("t0187-onset")
F1 = live_frame("t0187-after-1")
F2 = live_frame("t0187-after-2")
FR = live_frame("t0187-fresh")
NEWER = live_frame("t0187-newer")


# --------------------------------------------------------------------------------------------- the fake


class RefreshBrowser(FakeLiveBrowser):
    """``FakeLiveBrowser`` plus two test controls (module docstring):

    - ``popup_after``: the call id after whose ``execute`` a popup opens (after the settle, as ``late_popup.html``);
      the next ``execute()`` or ``frame()`` adopts it (``page`` +1). An adopting ``execute()`` of a browser action
      is refused with the page-changed reason; a ``click_at``/``type_at`` whose frame was captured on an earlier page
      is refused with it as well (the executor's ``plan.frame_page != d.page_serial``). Nothing is dispatched.
    - ``fail_frames``: that many next ``frame()`` calls raise (a one-shot capture failure), then frames work again.
    - ``search_refusals``: call id -> reason; that ``google_search`` is refused with it (``FakeLiveBrowser`` answers
      every search ``dry_run`` before any handler, as the merged executor does; this is the criterion-5 adversarial
      row's synthetic executor refusal).
    """

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.page = 1
        self.page_of: dict[str, int] = {}
        self.popup_pending = False
        self.popup_after: str | None = None
        self.fail_frames = 0
        self.search_refusals: dict[str, str] = {}

    def _adopt(self) -> bool:
        if not self.popup_pending:
            return False
        self.popup_pending = False
        self.page += 1
        return True

    def _register(self, f: Frame) -> Frame:
        out = super()._register(f)
        assert out.phash is not None
        self.page_of[out.phash] = self.page
        return out

    async def frame(self) -> Frame:
        if self.fail_frames > 0 and not self.closed:
            self.fail_frames -= 1
            self.calls.append(("frame",))
            raise RuntimeError("the fake browser could not take this frame")
        self._adopt()  # frame() runs ensure() first
        return await super().frame()

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        if inv.name == "google_search" and inv.call_id in self.search_refusals:
            self.calls.append(("execute", inv.call_id))
            self.invocations.append(inv)
            return refused(inv, self.search_refusals[inv.call_id])
        if (
            inv.name in ("click_at", "type_at", "press_key", "scroll", "page_overview")
            and not self.closed
        ):
            adopted = self._adopt()
            fr = inv.frame
            stale = (
                inv.name in ("click_at", "type_at")
                and fr is not None
                and fr.phash in self.page_of
                and self.page_of[fr.phash] != self.page
            )
            if (adopted or stale) and inv.call_id not in self.handlers:
                self.calls.append(("execute", inv.call_id))
                self.invocations.append(inv)
                return refused(inv, PAGE_CHANGED)
        out = await super().execute(inv)
        if inv.call_id == self.popup_after:
            self.popup_pending = True
        return out


# --------------------------------------------------------------------------------------------- reading the rig


def label_on() -> Any:
    return live_feedback(screen_label=True, page_extent=False)


def kwargs(**more: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"live_feedback": label_on(), "action_outcome": True}
    out.update(more)
    return out


def step_body(rig: Any, k: int, turn: str = "t1") -> dict[str, Any]:
    return _wire(tsl.step_request(rig, k, turn))


def current_images(body: dict[str, Any]) -> list[str]:
    """The image URLs of the body's final user message, in order (the step's current image(s))."""
    return [
        p["image_url"]["url"] for p in last_user(body)["content"] if p.get("type") == "image_url"
    ]


def frame_calls(browser: Any) -> list[int]:
    return [i for i, c in enumerate(browser.calls) if c[0] == "frame"]


def execute_at(browser: Any, call_id: str) -> int:
    at = [i for i, c in enumerate(browser.calls) if c == ("execute", call_id)]
    assert len(at) == 1, f"one execute of {call_id}; calls {browser.calls}"
    return at[0]


def statuses(rig: Any, cid: str) -> list[str]:
    return [e.status for e in tsl.results_for(rig, cid)]


def refresh_notes(rig: Any) -> list[dict[str, Any]]:
    assert rig.log_records, "test bug: the session log is read back only after the rig closes"
    return [d for d in rig.kind("note") if d.get("what") == REFRESHED]


def notes_what(rig: Any) -> list[str]:
    assert rig.log_records, "test bug: the session log is read back only after the rig closes"
    return [str(d.get("what", "")) for d in rig.kind("note")]


async def loop_turn(rig: Any, steps: tuple[Any, ...], *, question: str = tsl.QUESTION) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(question)
    if steps:
        await tsl.until_steps(rig, len(steps), limit_ms=30_000, step=100)
    await rig.turn_done(limit_ms=30_000)
    await rig.advance(300)


def assert_fresh_step(body: dict[str, Any], fresh: Frame, old: Frame, *, why: str) -> None:
    """Criterion 1 at step k+1: exactly one current image, the fresh frame; the old image nowhere in the body;
    ``LABEL_NOW`` after it (never ``LABEL_UPDATED``/``LABEL_UNCHANGED``), and no outcome clause."""
    got = current_images(body)
    assert got == [builder_url(fresh)], (
        f"{why}: step k+1's final user message carries exactly the fresh frame; got {len(got)} image(s), "
        f"fresh={builder_url(fresh) in got}, old={builder_url(old) in got}"
    )
    assert b64(old.jpeg) not in json.dumps(body), (
        f"{why}: the previous image is replaced, never kept beside the fresh one"
    )
    label = label_after(body, fresh)
    assert label == L1 and label not in (LU, LN), f"{why}: LABEL_NOW, never LABEL_UPDATED; {label}"
    assert CLAUSE_STEM not in json.dumps(body), f"{why}: no outcome clause (T-0134 A3, T-0177 B2)"


# =====================================================================================================
# Criteria 1 and 7: an executor refusal that says "take a new frame"
# =====================================================================================================


CASES = [(PAGE_CHANGED, a) for a in ACTIONS] + [
    (OVERVIEW_STALE, "click_at"),
    (OVERVIEW_PAGE_CHANGED, "click_at"),
]


@pytest.mark.parametrize(
    ("reason", "action"),
    CASES,
    ids=[f"{KIND[r]}-{a}" for r, a in CASES],
)
async def test_an_executor_new_frame_refusal_is_followed_by_a_fresh_frame(
    tmp_path: Path, reason: str, action: str
) -> None:
    """Criteria 1, 2 and 7. The step-1 action is refused by the executor with ``reason``. Then:

    - one ``frame()`` is taken after the refusal;
    - step 2 shows exactly that fresh frame, labelled ``LABEL_NOW``, with no outcome clause, and no byte of the onset
      image;
    - its capture identity is ``latest_frame``'s (adversarial: the old image kept beside the fresh one);
    - the refused step's payload is ``refused`` with the reason and ``screenshot: "updated"`` (T-0177 B-R3);
    - one criterion-7 note is written, with ``reason_kind`` the constant's name and ``tool_step`` 1."""
    call, name, args = ACTIONS[action]
    browser = RefreshBrowser([F0, FR])
    browser.outcome("t1-call", lambda inv: refused(inv, reason))
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(call)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S3)],))
        frames = frame_calls(browser)
        assert len(frames) == 2 and frames[1] > execute_at(browser, "t1-call"), (
            f"criterion 1: one fresh frame, taken after the refusal; calls {browser.calls}"
        )
        s2 = step_body(rig, 2)
        assert_fresh_step(s2, FR, F0, why=f"criterion 1 ({KIND[reason]}, {action})")
        latest = browser.latest_frame
        assert latest is not None and current_images(s2) == [builder_url(latest)], (
            "adversarial: the one image's capture identity is latest_frame's"
        )
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, name, args, status("refused", reason, True))]
        ), (
            "criterion 1 / T-0177 B-R3: the refused step's payload says screenshot: updated; "
            f"{step_tail(rig, 2)[-300:]}"
        )
        c = card(rig, "t1-call")
        assert (c.status, c.reason) == ("refused", reason), c
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": KIND[reason], "tool_step": 1}
    ], f"criterion 7: one note per refresh; notes {rig.kind('note')}"


async def test_a_page_overview_refused_with_the_page_changed_reason_refreshes(
    tmp_path: Path,
) -> None:
    """Ruling R4. With the overview kit (T-0140), the step-1 ``page_overview`` is refused by the executor with the
    page-changed reason (the real executor's overview path refuses so when the page serial moved, ``executor.py``
    as merged). Step 2 shows exactly the fresh frame, labelled ``LABEL_NOW``; the payload says ``"updated"``; one
    criterion-7 note with ``tool_step`` 1."""
    from tests.unit.test_t0140_session import overview_kit

    browser = RefreshBrowser([F0, FR])
    browser.outcome("t1-call", lambda inv: refused(inv, PAGE_CHANGED))
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs=kwargs(),
        kit=overview_kit(browser),
    ) as rig:
        await loop_turn(rig, ([answer(S3)],))
        assert ("execute", "t1-call") in browser.calls, "premise: the overview reached the executor"
        frames = frame_calls(browser)
        assert len(frames) == 2 and frames[1] > execute_at(browser, "t1-call"), (
            f"R4: one fresh frame, taken after the refusal; calls {browser.calls}"
        )
        assert_fresh_step(step_body(rig, 2), FR, F0, why="R4 (page_overview)")
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "page_overview", {}, status("refused", PAGE_CHANGED, True))]
        ), step_tail(rig, 2)[-300:]
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": "PAGE_CHANGED_REASON", "tool_step": 1}
    ], f"criterion 7; notes {rig.kind('note')}"


async def test_the_session_freshness_refusal_is_followed_by_a_fresh_frame(tmp_path: Path) -> None:
    """Criteria 1, 2 and 7 for the session's own ``FRESHNESS_REASON`` (T-0111g B9), and the second adversarial row.

    1. Step 1's scroll captures a newer own frame and fails; the frame for step 2 fails once, so step 2 still shows
       the onset image.
    2. Step 2's click, read off that image, is refused by B9 before ``execute``.
    3. Step 3 shows a fresh frame, labelled ``LABEL_NOW``, and the refused step's payload says ``"updated"``.
    4. Step 3's click, which carries the fresh frame, is executed: B9 (``_live_stale_frame``) does not refuse it."""

    def capture_then_fail(fb: Any, inv: Any) -> Any:
        fb.post_frames.append(NEWER)
        fb.capture_post()
        fb.fail_frames = 1
        return failed(inv, "TimeoutError during scroll")

    browser = RefreshBrowser([F0, FR])
    browser.handlers["t1-call"] = capture_then_fail
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(SCROLL)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(
            rig, ([answer(S2), tool(CLICK2)], [answer(S2), tool(VALID_CLICK)], [answer(S3)])
        )
        assert current_images(step_body(rig, 2)) == [builder_url(F0)], (
            "premise: no fresh frame for step 2, the onset image stays"
        )
        assert (
            statuses(rig, "t1-call-s2") == ["refused"]
            and card(rig, "t1-call-s2").reason == FRESHNESS
        ), f"premise: B9 refuses the step-2 click; {card(rig, 't1-call-s2')}"
        assert ("execute", "t1-call-s2") not in browser.calls, "premise: B9 refuses before execute"
        s3 = step_body(rig, 3)
        assert_fresh_step(s3, FR, F0, why="criterion 1 (FRESHNESS_REASON)")
        assert f"reason:{Q}{FRESHNESS}{Q},screenshot:{Q}updated{Q}" in step_tail(rig, 3), (
            f"T-0177 B-R3: the B9 refusal's payload says updated; {step_tail(rig, 3)[-300:]}"
        )
        assert statuses(rig, "t1-call-s3") == ["executed"], (
            f"second adversarial: the click carrying the fresh frame is not refused by B9; {card(rig, 't1-call-s3')}"
        )
        inv = [i for i in browser.invocations if i.call_id == "t1-call-s3"]
        assert inv and inv[0].frame is not None and inv[0].frame.phash == FR.phash, (
            "the executed click carries the fresh frame's geometry"
        )
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": "FRESHNESS_REASON", "tool_step": 2}
    ], f"criterion 7; notes {rig.kind('note')}"


# =====================================================================================================
# Criterion 3 on CPU: the late popup through a Session
# =====================================================================================================


async def test_late_popup_loop_gets_unstuck(tmp_path: Path) -> None:
    """Criterion 3 (the t16 reproduction, fake page change) and the second adversarial row.

    1. Step 1's click executes, and a popup opens after its settle.
    2. Step 2's click, read off the opener's post-action frame, is refused with the page-changed reason.
    3. Step 3 shows a fresh frame of the popup. Its click carries that frame and is executed on the popup.

    Before the fix step 3 kept the opener's frame, and every later click was refused again."""
    browser = RefreshBrowser([F0, FR], post_frames=[F1, F2])
    browser.popup_after = "t1-call"
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(
            rig, ([answer(S2), tool(CLICK2)], [answer(S2), tool(VALID_CLICK)], [answer(S3)])
        )
        assert current_images(step_body(rig, 2)) == [builder_url(F1)], (
            "premise: step 2 is the opener's frame"
        )
        assert card(rig, "t1-call-s2").reason == PAGE_CHANGED, card(rig, "t1-call-s2")
        s3 = step_body(rig, 3)
        assert_fresh_step(s3, FR, F1, why="criterion 3")
        assert browser.page_of[FR.phash] == 2, "premise: the fresh frame is of the popup"
        assert statuses(rig, "t1-call-s3") == ["executed"], (
            f"criterion 3: the click on the fresh frame is executed; {card(rig, 't1-call-s3')}"
        )
        assert step_tail(rig, 4) == want_tail(
            rig,
            [
                (
                    S1,
                    "click_at",
                    CLICK_ARGS,
                    {"action": ACT1, "screenshot": "updated", "status": "executed"},
                ),
                (S2, "click_at", CLICK2_ARGS, status("refused", PAGE_CHANGED, True)),
                (
                    S2,
                    "click_at",
                    CLICK_ARGS,
                    {"action": ACT1, "screenshot": "updated", "status": "executed"},
                ),
            ],
        ), step_tail(rig, 4)[-600:]
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": "PAGE_CHANGED_REASON", "tool_step": 2}
    ]


async def test_two_refusals_in_a_row_refresh_twice(tmp_path: Path) -> None:
    """Criterion 7 "one note per refresh": steps 1 and 2 are both refused with the page-changed reason, so steps 2
    and 3 each show a new fresh frame, labelled ``LABEL_NOW``, and two notes say ``tool_step`` 1 and 2."""
    fr2 = live_frame("t0187-fresh-2")
    browser = RefreshBrowser([F0, FR, fr2])
    browser.outcome("t1-call", lambda inv: refused(inv, PAGE_CHANGED))
    browser.outcome("t1-call-s2", lambda inv: refused(inv, PAGE_CHANGED))
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert_fresh_step(step_body(rig, 2), FR, F0, why="the first refresh")
        assert_fresh_step(step_body(rig, 3), fr2, FR, why="the second refresh")
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": "PAGE_CHANGED_REASON", "tool_step": 1},
        {"what": REFRESHED, "reason_kind": "PAGE_CHANGED_REASON", "tool_step": 2},
    ]


async def test_label_now_is_read_once(tmp_path: Path) -> None:
    """T-0177 B-R2, reused: after the refresh, the next executed step goes back to ``LABEL_UPDATED`` (step 3 after
    step 2's executed click on the fresh frame)."""
    browser = RefreshBrowser([F0, FR], post_frames=[F1])
    browser.outcome("t1-call", lambda inv: refused(inv, PAGE_CHANGED))
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert current_images(step_body(rig, 2)) == [builder_url(FR)], (
            "criterion 1: step 2 shows exactly the fresh frame (the premise of read-once)"
        )
        assert label_after(step_body(rig, 2), FR) == L1
        assert statuses(rig, "t1-call-s2") == ["executed"]
        assert label_after(step_body(rig, 3), F1) == LU, "B-R2: LABEL_NOW is read once"


# =====================================================================================================
# Criterion 3 on CPU with the real executor
# =====================================================================================================


class AfterExecute:
    """Delegates every name to the real ``BrowserToolExecutor``; after the execute of ``popup_after`` it opens a
    popup on the fake driver (after the settle, as ``late_popup.html``). Records outcomes and the frames the session
    took."""

    def __init__(self, inner: Any, driver: Any, popup_after: str) -> None:
        self._inner = inner
        self._driver = driver
        self._popup_after = popup_after
        self.outcomes: list[Any] = []
        self.frames: list[Frame] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def frame(self) -> Frame:
        f = await self._inner.frame()
        self.frames.append(f)
        return f  # type: ignore[no-any-return]

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        out = await self._inner.execute(inv)
        self.outcomes.append(out)
        if inv.call_id == self._popup_after:
            self._driver.open_popup()
        return out  # type: ignore[no-any-return]


async def test_real_executor_late_popup_loop_gets_unstuck(tmp_path: Path) -> None:
    """Criterion 3 with the real ``BrowserToolExecutor`` over ``test_computer_executor.FakeDriver``.

    1. Step 1's click executes on the opener, and a popup opens after its settle.
    2. Step 2's click is refused by the executor itself with ``PAGE_CHANGED_REASON``, dispatching nothing.
    3. Step 3 shows the frame the session took after the refusal, of the popup. Its click is executed on the popup."""
    from tests.unit.test_computer_executor import make

    ex, driver = make()
    watch = AfterExecute(ex, driver, "t1-call")
    async with feedback_rig(
        tmp_path,
        watch,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs=kwargs(),
        kit=live_kit(watch),
    ) as rig:
        ex._now_ms = lambda: rig.clock.now_ms
        await loop_turn(
            rig, ([answer(S2), tool(CLICK2)], [answer(S2), tool(VALID_CLICK)], [answer(S3)])
        )
        got = [(o.status, o.reason) for o in watch.outcomes]
        assert got[:2] == [("executed", None), ("refused", PAGE_CHANGED)], got
        assert driver.page_serial == 1 and driver.active == "popup", (
            "premise: the popup was adopted"
        )
        assert len(watch.frames) == 2 and driver.shots[-1] == "popup", (
            f"criterion 3: a fresh frame of the popup was taken; shots {driver.shots}"
        )
        s3 = step_body(rig, 3)
        assert current_images(s3) == [data_url(watch.frames[-1].jpeg)], (
            "criterion 3: step 3 shows the popup's fresh frame, and only it"
        )
        assert label_after(s3, watch.frames[-1]) == L1
        assert got[2:] == [("executed", None)], (
            f"criterion 3: the click on the fresh frame is executed on the popup; {got}"
        )
        assert driver.presses[-1] == (640, 200), driver.presses


async def test_real_executor_refusal_is_not_weakened() -> None:
    """Criterion 6 and owned file ``executor.py``. The executor's reason is public as ``PAGE_CHANGED_REASON``, in
    ``__all__``, with the same text. After a popup is adopted, a click carrying the opener's frame is still refused
    with it and dispatches nothing. The popup's own frame is accepted."""
    from tests.unit.test_computer_executor import computer_executor, inv, make, own_frame

    mod = computer_executor()
    assert getattr(mod, "PAGE_CHANGED_REASON", None) == PAGE_CHANGED, (
        "criterion 2: executor.PAGE_CHANGED_REASON is the page-changed text"
    )
    assert "PAGE_CHANGED_REASON" in mod.__all__
    ex, fake = make()
    opener = await own_frame(ex)
    fake.open_popup()
    popup = await own_frame(ex)
    assert fake.page_serial == 1, "premise: frame()'s ensure() adopted the popup"
    for i in (
        inv("click_at", opener, x=500, y=500),
        inv("type_at", opener, x=500, y=500, text="abc", press_enter=False),
    ):
        mark = len(fake.calls)
        out = await ex.execute(i)
        assert (out.status, out.reason) == ("refused", mod.PAGE_CHANGED_REASON), (i.name, out)
        assert fake.dispatched(mark) == [], fake.dispatched(mark)
    out = await ex.execute(inv("click_at", popup, x=500, y=500))
    assert out.status == "executed", out


def test_one_named_constant_holds_the_four_reasons() -> None:
    """Criterion 2. ``session.py`` holds exactly one module-level constant whose texts are exactly the four. It may
    be a set, tuple or list of them, or a mapping whose keys or values are them. The session's own and the executor's
    constants still have the typed texts."""
    from local_astra.computer import executor

    s = session_mod()
    assert s.FRESHNESS_REASON == FRESHNESS
    assert (executor.OVERVIEW_STALE, executor.OVERVIEW_PAGE_CHANGED) == (
        OVERVIEW_STALE,
        OVERVIEW_PAGE_CHANGED,
    )
    found = []
    for name, value in vars(s).items():
        if name.startswith("__"):
            continue
        texts: set[Any] | None = None
        if isinstance(value, frozenset | set | tuple | list):
            texts = set(value)
        elif isinstance(value, Mapping):
            if set(value.keys()) == FOUR:
                texts = set(value.keys())
            else:
                with_values = (
                    set(value.values()) if all(isinstance(v, str) for v in value.values()) else None
                )
                texts = with_values
        if texts == FOUR:
            found.append(name)
    assert len(found) == 1, f"criterion 2: one named constant holds the four reasons; found {found}"
    tree = ast.parse(Path(s.__file__).read_text())
    top = {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(t, ast.Name)
    }
    assert found[0] in top, f"criterion 2: {found[0]} is assigned at module level in session.py"


# =====================================================================================================
# Criterion 4: no fresh frame
# =====================================================================================================


@pytest.mark.parametrize("mode", ["raise", "hang", "closed"])
async def test_no_fresh_frame_keeps_the_image_and_notes_it(tmp_path: Path, mode: str) -> None:
    """Criterion 4. The step-1 click is refused with the page-changed reason, and then no frame can be taken: the
    capture fails, hangs past ``LIVE_FRAME_TIMEOUT_MS`` on the session clock, or the browser closed. Step 2 keeps
    the onset image with BASE's label (``LABEL_UNCHANGED``), and the payload says ``"unchanged"`` (B-R3). B8's note
    is written, and no refresh note."""
    live_frame_timeout_ms()

    def refuse_then_break(fb: Any, inv: Any) -> Any:
        if mode == "closed":
            fb.close_window()
        else:
            fb.frame_mode = mode
        return refused(inv, PAGE_CHANGED)

    browser = RefreshBrowser([F0, FR])
    browser.handlers["t1-call"] = refuse_then_break
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S3)],))
        s2 = step_body(rig, 2)
        assert current_images(s2) == [builder_url(F0)], (
            f"criterion 4 ({mode}): the previous image stays"
        )
        assert b64(FR.jpeg) not in json.dumps(s2)
        if mode != "closed":
            assert label_after(s2, F0) == LN, (
                f"criterion 4 ({mode}): nothing else changes (BASE's label)"
            )
            assert len(frame_calls(browser)) == 2, (
                f"premise: a fresh frame was attempted; {browser.calls}"
            )
        if mode == "hang":
            assert browser.frame_hang_cancelled == 1, (
                "the hung capture was cancelled at the timeout"
            )
        assert f"reason:{Q}{PAGE_CHANGED}{Q},screenshot:{Q}unchanged{Q}" in step_tail(rig, 2), (
            f"B-R3: unchanged when no fresh frame could be taken; {step_tail(rig, 2)[-300:]}"
        )
    assert NO_FRESH in notes_what(rig), f"criterion 4 ({mode}): B8's note; notes {notes_what(rig)}"
    assert refresh_notes(rig) == [], "criterion 7: no refresh, no note"


# =====================================================================================================
# Criterion 5: every other refusal keeps today's image and label
# =====================================================================================================


#: executor refusals outside the set (executor.py as merged), and the session's executor-error refusal
OTHER_REASONS = [
    "bad coordinates",
    "bad arguments",
    "bad arguments: control character",
    NOT_OWN,
    "no frame",
    "frame from before a browser restart",
    "will not type a card number; ask the user to type it themselves",
    "will not type into a password or card field; ask the user to type it themselves",
    "could not check the field before typing; nothing was typed",
    "pasting is off on real websites",
    "not available in the browser",
    "unknown tool",
    "the page did not scroll to the point; nothing was clicked",
    # near misses of the set: a prefix of the page-changed text, and the set's common tail alone
    "the active page changed",
    "take a new frame",
]


@pytest.mark.parametrize("reason", OTHER_REASONS)
async def test_other_executor_refusals_keep_the_image(tmp_path: Path, reason: str) -> None:
    """Criterion 5, too wide. A step-1 click refused with any other reason takes no frame. Step 2 shows the onset
    image, labelled ``LABEL_UNCHANGED``, and the payload says ``"unchanged"``: BASE's step, byte for byte through the
    oracle tail. No refresh note is written."""
    browser = RefreshBrowser([F0, FR])
    browser.outcome("t1-call", lambda inv: refused(inv, reason))
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S3)],))
        assert len(frame_calls(browser)) == 1, (
            f"criterion 5: no frame after {reason!r}; {browser.calls}"
        )
        s2 = step_body(rig, 2)
        assert current_images(s2) == [builder_url(F0)], "criterion 5: the image stays"
        assert label_after(s2, F0) == LN, "criterion 5: the label stays"
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "click_at", CLICK_ARGS, status("refused", reason, False))]
        ), step_tail(rig, 2)[-300:]
    assert refresh_notes(rig) == []


async def test_executor_error_keeps_the_image(tmp_path: Path) -> None:
    """Criterion 5: an executor that raises gives the session's ``executor error`` refusal; the image stays."""

    def boom(fb: Any, inv: Any) -> Any:
        raise RuntimeError("the fake executor exploded")

    browser = RefreshBrowser([F0, FR])
    browser.handlers["t1-call"] = boom
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S3)],))
        assert card(rig, "t1-call").reason == "executor error"
        assert len(frame_calls(browser)) == 1, browser.calls
        assert current_images(step_body(rig, 2)) == [builder_url(F0)]
        assert label_after(step_body(rig, 2), F0) == LN


async def test_a_search_refused_at_the_cap_keeps_the_image(tmp_path: Path) -> None:
    """Criterion 5: a ``google_search`` refusal keeps the image. The refusal is the session's own (T-0176): a live
    kit with ``web_search`` set and a backend that always fails; steps 1-3's searches fail, so step 4's search is
    refused at the failed-search cap before the backend is called. Step 5 shows the onset image, as step 4 did, with
    the same label; no frame is taken after the onset, and no note is written."""
    search = FakeSearch(fail=TimeoutError())
    browser = RefreshBrowser([F0, FR])
    kit = wsw.web(live_kit(browser), search)
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(wsw.search_call("q1"))],
        session_kwargs=kwargs(),
        kit=kit,
    ) as rig:
        await loop_turn(
            rig,
            (
                [answer(S2), tool(wsw.search_call("q2"))],
                [answer(S2), tool(wsw.search_call("q3"))],
                [answer(S2), tool(wsw.search_call("q4"))],
                [answer(S3)],
            ),
        )
        assert search.queries == ["q1", "q2", "q3"], (
            f"premise: three searches ran and failed, the fourth never reached the backend; {search.queries}"
        )
        assert statuses(rig, "t1-call-s4") == ["refused"], "premise: the session refused the search"
        c = card(rig, "t1-call-s4")
        assert (c.status, c.reason) == ("refused", wsw.FAILED_CAP_CARD), c
        assert f"reason:{Q}{wsw.FAILED_CAP_REASON}{Q}" in step_tail(rig, 5), step_tail(rig, 5)[
            -400:
        ]
        assert len(frame_calls(browser)) == 1, (
            f"criterion 5: no frame after a refused search; {browser.calls}"
        )
        s4, s5 = step_body(rig, 4), step_body(rig, 5)
        assert current_images(s5) == current_images(s4) == [builder_url(F0)], (
            "criterion 5: the image stays"
        )
        assert label_after(s5, F0) == label_after(s4, F0) != L1, "criterion 5: the label stays"
    assert refresh_notes(rig) == []


@pytest.mark.parametrize("reason", [PAGE_CHANGED, FRESHNESS])
async def test_an_executor_refused_search_keeps_the_image_whatever_its_reason(
    tmp_path: Path, reason: str
) -> None:
    """Criterion 5, adversarial: a ``google_search`` the executor refuses keeps the image even when the reason is one
    of the set's texts (a rule keyed on the reason alone would refresh here). The refusal is the session's real
    path for an executor-refused search (``card == "refused"``, before any backend); the executor here is
    ``RefreshBrowser`` with ``search_refusals`` (the merged executor answers every search ``dry_run``, so this is a
    synthetic executor reason, not one the real executor gives)."""
    browser = RefreshBrowser([F0, FR])
    browser.search_refusals["t1-call"] = reason
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(SEARCH)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S3)],))
        assert statuses(rig, "t1-call") == ["refused"], "premise: the search was refused"
        c = card(rig, "t1-call")
        assert (c.status, c.reason) == ("refused", reason), c
        assert len(frame_calls(browser)) == 1, browser.calls
        s2 = step_body(rig, 2)
        assert current_images(s2) == [builder_url(F0)]
        assert label_after(s2, F0) != L1, "no LABEL_NOW after a refused search"
        assert f"reason:{Q}{reason}{Q},screenshot:{Q}unchanged{Q}" in step_tail(rig, 2), step_tail(
            rig, 2
        )[-300:]
    assert refresh_notes(rig) == []


async def test_ask_first_keeps_the_image(tmp_path: Path) -> None:
    """Criterion 5: T-0107a. Step 1's click executes; step 2 asks the user first, so its click is refused (the card
    says ``refused``, ``asked the user first``), nothing is dispatched and the turn ends with the question. No frame
    is taken (the post-action frame is the executor's own capture, not a ``frame()``), and no note is written."""
    browser = RefreshBrowser([F0, FR], post_frames=[F1])
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        rig.llm.script_steps([answer(ASK), tool(CLICK2)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1, limit_ms=30_000, step=100)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        c = card(rig, "t1-call-s2")
        assert (c.status, c.reason) == ("refused", "asked the user first"), c
        assert ("execute", "t1-call-s2") not in browser.calls, browser.calls
        assert len(frame_calls(browser)) == 1, (
            f"criterion 5: no frame after ask-first; {browser.calls}"
        )
        assert len(tsl.steps_of(rig)) == 1, "premise: the turn ended at the question"
    assert refresh_notes(rig) == []


async def test_user_declines_keeps_the_image(tmp_path: Path) -> None:
    """Criterion 5: T-0107e. The user's turn is a clear no, so the step-1 click is refused (the card says
    ``refused``, ``the user said no``); nothing is executed, no frame is taken after the onset, and no note is
    written."""
    browser = RefreshBrowser([F0, FR])
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await rig.ask(NO)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        c = card(rig, "t1-call")
        assert (c.status, c.reason) == ("refused", "the user said no"), c
        assert browser.executes() == 0, browser.calls
        assert len(frame_calls(browser)) == 1, (
            f"criterion 5: no frame after a decline; {browser.calls}"
        )
    assert refresh_notes(rig) == []


async def test_a_no_progress_stop_keeps_the_image(tmp_path: Path) -> None:
    """Criterion 5: a T-0137 ``repeat`` stop. Three identical clicks change nothing, and the fourth is refused by the
    guard. No frame is taken, and step 5 shows the same capture with ``LABEL_UNCHANGED``."""
    from tests.unit import test_t0137_session as g

    browser = g.browser_of([g.same(), g.same(), g.same()])
    async with g.guard_rig(
        tmp_path, browser, script=[answer(g.S[0]), tool(g.CLICK)], extra={"action_outcome": True}
    ) as rig:
        await g.run(
            rig,
            g.steps_of([(g.S[1], g.CLICK), (g.S[2], g.CLICK), (g.S[3], g.CLICK), (g.REPORT, "")]),
        )
        assert card(rig, "t1-call-s4").reason == g.REPEAT_REASON, (
            "premise: the guard stopped step 4"
        )
        assert len(frame_calls(browser)) == 1, (
            f"criterion 5: no frame after the stop; {browser.calls}"
        )
        s5 = step_body(rig, 5)
        assert current_images(s5) == [builder_url(g.F0)]
        assert label_after(s5, g.F0) == LN
    assert refresh_notes(rig) == []


# =====================================================================================================
# Criterion 6: the guard, the outcome and the non-live sessions
# =====================================================================================================


async def test_a_refused_step_is_never_a_repeat_instance(tmp_path: Path) -> None:
    """Criterion 6 (T-0137). Step 1's click is refused with the page-changed reason, and the fresh frame is a capture
    of the same screen (same executor phash, other bytes). Step 2's identical click is executed and carries no
    ``repeat_of_step``: a refused step is not an instance. Step 3's identical click repeats step 2.

    Pending, although criterion 6 is a negative: its premise is the refresh (step 2 shows the fresh capture), which
    BASE does not take."""
    from tests.unit import test_t0137_session as g

    same_screen = g.other_bytes("t0187-fresh")
    browser = RefreshBrowser([g.F0, same_screen], post_frames=[g.same(), g.same()], echo="model")
    browser.outcome("t1-call", lambda inv: refused(inv, PAGE_CHANGED))
    async with g.guard_rig(tmp_path, browser, script=[answer(g.S[0]), tool(g.CLICK)]) as rig:
        await g.run(rig, g.steps_of([(g.S[1], g.CLICK), (g.S[2], g.CLICK), (g.REPORT, "")]))
        assert current_images(step_body(rig, 2)) == [builder_url(same_screen)], "premise: refreshed"
        assert statuses(rig, "t1-call-s2") == ["executed"] and statuses(rig, "t1-call-s3") == [
            "executed"
        ]
        facts = g.facts_in(step_tail(rig, 4))
        assert [f["step"] for f in facts] == [2], (
            f"criterion 6: only executed steps are repeat instances; facts {facts}"
        )
    line = g.check_line_shape(g.guard_line(rig))
    assert line["repeats"] == 1 and line["stops_repeat"] == 0, line


async def test_a_refused_steps_screen_counts_toward_recurrence(tmp_path: Path) -> None:
    """Review ruling RV1 (criterion 6, T-0137 B-3/A-5): a step refused with a "take a new frame" reason stays in the
    guard's history exactly as at BASE (``_guard_settle`` then ``_guard_outcome``); only the fresh frame is added.

    Screens by executor phash: A, B, C. History A (executed, screen becomes B), B (executed, back to A), A (refused
    with the page-changed reason; the fresh frame shows C), C (executed, back to A), then a fifth call read off A.
    A's runs in the history are steps 1, 3 and the proposed step 5, so the fifth call is A's third visit: refused with
    ``RECURRING_REASON`` before ``execute``, ``screen_visits`` 3 (``max_screen_visits`` on the turn line). Without
    step 3 in the history, A's runs are steps 1 and 5 (two visits) and step 5 executes.

    Mutants this kills: the refused step skipped from the guard entirely, and the T-0177 B-R4 shape (its outcome
    applied but its ``_guard_settle`` skipped, so the outcome rewrites step 2's)."""
    from tests.unit import test_t0137_session as g

    pa, pb, pc = "a1a1a1a1a1a1a1a1", "b2b2b2b2b2b2b2b2", "c3c3c3c3c3c3c3c3"
    onset = live_frame("t0187-rv1-A-0", phash=pa)
    fresh_c = g.other_bytes("t0187-rv1-fresh", pc)
    post = [g.other_bytes("t0187-rv1-1", pb), g.other_bytes("t0187-rv1-2", pa)]
    post.append(g.other_bytes("t0187-rv1-4", pa))
    browser = RefreshBrowser([onset, fresh_c], post_frames=post, echo="model")
    browser.outcome("t1-call-s3", lambda inv: refused(inv, PAGE_CHANGED))
    calls = [g.click(100, 100), g.click(300, 300), g.click(500, 500), g.click(700, 700)]
    calls.append(g.click(900, 900))
    async with g.guard_rig(tmp_path, browser, script=[answer(g.S[0]), tool(calls[0])]) as rig:
        await g.run(rig, g.steps_of([(g.S[i], calls[i]) for i in range(1, 5)] + [(g.REPORT, "")]))
        assert [statuses(rig, f"t1-call-s{k}") for k in (2, 3, 4)] == [
            ["executed"],
            ["refused"],
            ["executed"],
        ], "premise: steps 2 and 4 executed, step 3 refused"
        assert card(rig, "t1-call-s3").reason == PAGE_CHANGED, card(rig, "t1-call-s3")
        assert current_images(step_body(rig, 4)) == [builder_url(fresh_c)], (
            "premise (criterion 1): step 4 shows the fresh frame of screen C"
        )
        assert statuses(rig, "t1-call-s5") == ["refused"], (
            "RV1: the refused step 3 was read off screen A, so step 5 is A's third visit and is stopped; "
            f"step 5 {statuses(rig, 't1-call-s5')}, card {card(rig, 't1-call-s5')}"
        )
        assert card(rig, "t1-call-s5").reason == g.RECURRING_REASON, card(rig, "t1-call-s5")
        assert g.executes(browser, "t1-call-s5") == 0, browser.calls
    recs = g.records_of(rig, "t1-call-s5")
    assert recs and recs[-1].get("no_progress") == "recurring", recs
    line = g.check_line_shape(g.guard_line(rig))
    assert (line["stops_recurring"], line["stops_repeat"], line["max_screen_visits"]) == (
        1,
        0,
        3,
    ), f"RV1: screen_visits 3 on A; {line}"
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": "PAGE_CHANGED_REASON", "tool_step": 3}
    ], f"RV1: only the fresh frame is added; notes {rig.kind('note')}"


async def test_freshness_after_reground_pass2_refreshes(tmp_path: Path) -> None:
    """Review ruling RV2 (criteria 1 and 2 at the second B9 site): with ``reground`` on, a capture lands while
    ReGround's pass 2 runs, so B9's re-check after pass 2 refuses the step-1 click with ``FRESHNESS_REASON`` and
    nothing is dispatched. Step 2 then shows exactly a fresh frame, labelled ``LABEL_NOW``, never the onset image or
    the capture that landed during pass 2; one criterion-7 note with ``tool_step`` 1."""
    from tests.unit._live_browser_support import jpeg_frame

    onset, newer, fresh = (jpeg_frame(f"t0187-rv2-{t}") for t in ("onset", "newer", "fresh"))
    assert len({onset.jpeg, newer.jpeg, fresh.jpeg}) == 3, "test bug: three distinct captures"
    browser = FakeLiveBrowser([onset, fresh], post_frames=[newer])
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs=kwargs(reground=True),
        reground_llm=True,
    ) as rig:
        rig.llm.script_pass2([answer("pass two. "), tool(CLICK2)])
        rig.llm.on_pass2 = browser.capture_post
        await loop_turn(rig, ([answer(S3)],))
        assert rig.llm.pass2_ids, "premise: ReGround's pass 2 ran (the first B9 check passed)"
        c = card(rig, "t1-call")
        assert (c.status, c.reason) == ("refused", FRESHNESS), (
            f"premise: B9 refuses after pass 2; {c}"
        )
        assert browser.executes() == 0, f"premise: nothing dispatched; {browser.calls}"
        s2 = step_body(rig, 2)
        assert b64(newer.jpeg) not in json.dumps(s2), (
            "RV2: the capture that landed during pass 2 is not the fresh frame"
        )
        assert_fresh_step(s2, fresh, onset, why="RV2 (B9 after ReGround pass 2)")
        assert f"reason:{Q}{FRESHNESS}{Q},screenshot:{Q}updated{Q}" in step_tail(rig, 2), (
            f"T-0177 B-R3: the refused step's payload says updated; {step_tail(rig, 2)[-300:]}"
        )
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": "FRESHNESS_REASON", "tool_step": 1}
    ], f"criterion 7; notes {rig.kind('note')}"


async def test_a_refused_step_is_never_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 6 (T-0136). With ``action_outcome`` on, the refused step 1 gets no measured outcome, and the label at
    step 2 has no outcome clause."""
    spy = turn_logs(monkeypatch)
    browser = RefreshBrowser([F0, FR])
    browser.outcome("t1-call", lambda inv: refused(inv, PAGE_CHANGED))
    async with feedback_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kwargs()
    ) as rig:
        await loop_turn(rig, ([answer(S3)],))
        assert CLAUSE_STEM not in json.dumps(step_body(rig, 2))
    log = spy.by_turn("t1")
    assert 1 not in log.action_outcomes, log.action_outcomes
    assert log.action_outcome_unmeasured.get(1) in (None, "not_pointer"), (
        log.action_outcome_unmeasured
    )
    assert log.action_outcome_counts.get("unmeasured", 0) == 0, log.action_outcome_counts


# --------------------------------------------------------------------------------------------- the differential

#: main at the T-0187 branch point (T-0177 B, T-0180 C and T-0191 A merged). A literal.
BASE_SHA = "2f955738"
REGIMES = ("off", "dry_run", "live_other_refusal")


def _bodies(rig: Any) -> list[dict[str, Any]]:
    from local_astra.conductor.llm_client import VllmClient
    from tests.unit.test_session_live import _refusing_transport

    client = VllmClient(transport=_refusing_transport())
    prefills = [
        {
            "phase": "prefill",
            "request_id": rid,
            "body": client.request_body(p, rid, LlmOptions(max_tokens=1)),
        }
        for rid, p in rig.llm.prefills
    ]
    generates = [
        {"phase": "generate", "request_id": rid, "body": client.request_body(p, rid, o, x)}
        for (rid, p, o), x in zip(rig.llm.generates, rig.llm.xargs_log, strict=True)
    ]
    return [*prefills, *generates]


def _normalise(tree: Any, session_id: str, tmp: Path, log_path: Path) -> Any:
    text = (
        json.dumps(tree, sort_keys=True)
        .replace(str(log_path), "<log>")
        .replace(session_id, "<session>")
        .replace(str(tmp), "<tmp>")
    )
    out = json.loads(text)
    for line in out["lines"]:
        regime = line.get("regime")
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


async def _scenario(tmp: Path, regime: str) -> dict[str, Any]:
    """One session of ``regime``: turn 1 a tool turn (``off``: a plain answer), turn 2 a plain answer.

    - ``off``: tools off.
    - ``dry_run``: the recording executor refuses the step-1 click with the page-changed text (never live, so no
      frame is ever taken), then step 2 answers.
    - ``live_other_refusal``: the fake live browser refuses the step-1 click with ``frame not from this browser``;
      step 2's click executes; step 3 answers."""
    from tests.fakes.tool_executor import RecordingToolExecutor
    from tests.unit import _tools_switch_support as T

    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [live_frame("t0187-d-onset"), live_frame("t0187-d-turn2")],
        post_frames=[live_frame(f"t0187-d-after-{k}") for k in (1, 2)],
    )
    if regime == "off":
        kit: Any = None
        first = [answer("A shop window. ")]
        steps: tuple[Any, ...] = ()
    elif regime == "dry_run":
        out = ToolOutcome(
            call_id="t1-call",
            status="refused",
            action=f"refused click_at: {PAGE_CHANGED}",
            reason=PAGE_CHANGED,
            resolved={},
        )
        kit = T.make_kit(executor=RecordingToolExecutor({"t1-call": out}))
        first = [answer(S1), tool(VALID_CLICK)]
        steps = ([answer(S3)],)
    else:
        browser.outcome("t1-call", lambda inv: refused(inv, NOT_OWN))
        kit = live_kit(browser)
        first = [answer(S1), tool(VALID_CLICK)]
        steps = ([answer(S2), tool(CLICK2)], [answer(S3)])
    async with tool_rig(tmp, kit=kit, policy=tsl.LOOP, script=first, frames=[hud_frame(0)]) as rig:
        browser.now_ms = lambda: rig.clock.now_ms
        if steps:
            rig.llm.script_steps(*steps)
        await rig.ask(tsl.QUESTION)
        if steps:
            await tsl.until_steps(rig, len(steps), limit_ms=30_000, step=100)
        await rig.until(lambda: len(rig.lines()) >= 1, f"{regime} turn 1", limit_ms=30_000)
        await rig.advance(300)
        rig.llm.script = [answer("A shop. ")]
        await rig.ask("what is on my screen now")
        await rig.until(
            lambda: rig.state.state.name == "IDLE" and len(rig.lines()) >= 2,
            f"{regime} turn 2",
            limit_ms=30_000,
        )
        await rig.advance(300)
        session_id = rig.session.session_id
        log_path = Path(rig.log.path)
        requests = _bodies(rig)
        lines = [json.loads(x) for x in rig.lines()]
    records = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    sections = {
        "requests": requests,
        "lines": lines,
        "llm_request": [r["data"] for r in records if r.get("kind") == "llm.request"],
    }
    got: dict[str, Any] = _normalise(sections, session_id, tmp, log_path)
    got["premise"] = {
        "frames": sum(1 for c in browser.calls if c[0] == "frame"),
        "executed": [i.call_id for i in browser.invocations],
        "notes": [(r.get("data") or {}).get("what") for r in records if r.get("kind") == "note"],
    }
    return got


async def _capture(tmp: Path) -> dict[str, Any]:
    return {regime: await _scenario(tmp / regime, regime) for regime in REGIMES}


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0187_refresh_after_refusal as m
out = pathlib.Path(sys.argv[1])
got = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": got}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BASE's sections, once per module: a subprocess over ``git archive BASE_SHA`` (origin checked)."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0187_diff_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    got: dict[str, Any] = asyncio.run(_capture(tmp_path_factory.mktemp("t0187_diff_now")))
    return got


def _compare(base: Any, now: Any, label: str, path: str = "") -> None:
    if isinstance(base, dict) and isinstance(now, dict) and base.keys() == now.keys():
        for key in base:
            _compare(base[key], now[key], label, f"{path}/{key}")
        return
    if isinstance(base, list) and isinstance(now, list) and len(base) == len(now):
        for i, (b, n) in enumerate(zip(base, now, strict=True)):
            _compare(b, n, label, f"{path}[{i}]")
        return
    assert base == now, (
        f"{label}: differs from BASE {BASE_SHA} at {path}:\nbase {json.dumps(base)[:800]}\nnow  {json.dumps(now)[:800]}"
    )


@pytest.mark.parametrize("regime", REGIMES)
def test_regime_equals_base(base: dict[str, Any], now: dict[str, Any], regime: str) -> None:
    """Criteria 5 and 6 (differential). The request bodies, the turn lines and the ``llm.request`` records equal
    BASE's exactly in each regime: tools off; dry_run with a page-changed refusal (non-live: no frame is ever taken);
    live with a refusal outside the set. Premises: two turns; a tool step outside ``off``; no frame and no refresh
    note in the non-live regimes; in live, only the onset frames. Green at BASE by design (a negative)."""
    got = now[regime]
    assert len(got["lines"]) == 2, f"premise: two turns; {len(got['lines'])}"
    phases = [r.get("phase") for r in got["llm_request"]]
    if regime != "off":
        assert "tool_step" in phases, f"premise: a tool step in {regime}; {phases}"
    premise = got["premise"]
    assert REFRESHED not in premise["notes"], premise
    if regime == "live_other_refusal":
        assert premise["frames"] == 2 and premise["executed"] == ["t1-call", "t1-call-s2"], premise
    else:
        assert premise["frames"] == 0 and premise["executed"] == [], premise
    for section in ("requests", "lines", "llm_request"):
        _compare(base[regime][section], got[section], f"T-0187 {regime} {section}")
