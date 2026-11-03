"""T-0204 B3 through a real ``Session``: a pre-click "moved" refusal is answered with a fresh frame at the next step,
with no ``session.py`` change (docs/tasks/T-0204.md, B3; "Tests to write" B, ``test_t0204_session_moved.py``).

**Rig.** T-0187's real-executor rig (``feedback_rig`` with the T-0134 screen label on and ``action_outcome``), the
real ``BrowserToolExecutor`` over ``test_t0204_preclick.GuardDriver`` (the four B1 capabilities over a hand-drawn
page). ``MoveBefore`` delegates every name to the executor and, just before the execute of one call id, inserts a
300 px block above the point on the fake page (the page moved while the model thought). Step 1's click (VALID_CLICK,
(500, 250) -> (640, 200)) is then refused by the executor itself; T-0187's ``NEW_FRAME_REASONS`` path takes a fresh
frame; step 2 shows it with ``LABEL_NOW`` and step 2's click on it is executed.

**Expected values** are T-0187's: the refusal text typed from the task (``FRESHNESS``), the payload
``status("refused", FRESHNESS, True)`` (T-0177 B-R3: ``screenshot: "updated"``), the note ``{"what": REFRESHED,
"reason_kind": "FRESHNESS_REASON", "tool_step": 1}`` and T-0134's ``L1``.

**Negative** (B3 "no ``session.py`` change"; Owned files: nothing under ``local_astra/conductor``): ``local_astra/conductor``
is identical over T-0204 B's closed range (merge-base 0efb0f58 and fork point e715c339 .. tip a3e61ffb, M cbb88b4b),
never against the working tree (closed by T-0207 ruling A2-R2: T-0207 A2 legitimately edits ``session.py``).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, tools live (loop on), 1280x800 frames, n = 1, deterministic.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from local_astra.computer import executor as executor_mod
from local_astra.contracts.perception import Frame
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.unit._live_browser_support import builder_url, data_url, feedback_rig, live_kit
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_live import card, status, step_tail, want_tail
from tests.unit.test_t0134_label import L1, label_after
from tests.unit.test_t0187_refresh_after_refusal import (
    REFRESHED,
    S1,
    S2,
    S3,
    current_images,
    kwargs,
    loop_turn,
    refresh_notes,
    step_body,
)
from tests.unit.test_t0204_preclick import (
    BASE_SHA,
    FRESHNESS,
    GuardDriver,
    build,
    detailed_page,
    lazy_insert,
)

#: VALID_CLICK's point on 1280x800 (echo "pixels"): to_pixels(500, 1280) = 640, to_pixels(250, 800) = 200
POINT = (640, 200)
CLICK_ARGS = {"x": 500, "y": 250}


@pytest.fixture(autouse=True)
def settle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-0204 C10: these rows pin the guard alone (B3 through a real ``Session``), as ``test_t0204_preclick.py`` does;
    the visual settle's real ``asyncio.sleep`` would race this rig's scripted clock. The settle is pinned in
    ``tests/unit/test_t0204_settle.py``."""
    monkeypatch.setattr(executor_mod, "STABLE_SCREEN", False, raising=False)


class MoveBefore:
    """Delegates every name to the real executor; inserts a block above the point just before ``move_before``'s
    execute (module docstring). Records outcomes and the frames the session took."""

    def __init__(self, inner: Any, driver: GuardDriver, move_before: str) -> None:
        self._inner = inner
        self._driver = driver
        self._move_before = move_before
        self.outcomes: list[ToolOutcome] = []
        self.frames: list[Frame] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def frame(self) -> Frame:
        f = await self._inner.frame()
        self.frames.append(f)
        return f  # type: ignore[no-any-return]

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        if inv.call_id == self._move_before:
            self._driver.page = lazy_insert(self._driver.page, 100, 300, (200, 30, 30))
        out = await self._inner.execute(inv)
        self.outcomes.append(out)
        return out  # type: ignore[no-any-return]


async def test_moved_refusal_then_fresh_frame(tmp_path: Path) -> None:
    """B3: step 1's click is refused by the executor with ``FRESHNESS_REASON`` (nothing pressed); the session takes
    one fresh frame after it (the moved page); step 2 shows exactly that frame labelled ``LABEL_NOW``; the refused
    step's payload is ``refused`` with the reason and ``screenshot: "updated"``; one T-0187 refresh note; step 2's
    click on the fresh frame is executed and is the only press."""
    d = GuardDriver(detailed_page())
    ex = build(d)
    watch = MoveBefore(ex, d, "t1-call")
    async with feedback_rig(
        tmp_path,
        watch,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs=kwargs(),
        kit=live_kit(watch),
    ) as rig:
        ex._now_ms = lambda: rig.clock.now_ms
        await loop_turn(rig, ([answer(S2), tool(VALID_CLICK)], [answer(S3)]))
        got = [(o.status, o.reason) for o in watch.outcomes]
        assert got[:1] == [("refused", FRESHNESS)], (
            f"B3: the executor refuses the moved click: {got}"
        )
        assert len(watch.frames) == 2, (
            f"B3/T-0187: one fresh frame after the refusal: {len(watch.frames)}"
        )
        fresh = watch.frames[-1]
        moved = Image.open(io.BytesIO(fresh.jpeg)).convert("RGB").getpixel(POINT)
        assert abs(moved[0] - 200) <= 24 and moved[1] <= 60, (
            f"premise: the fresh frame shows the inserted block at the point: {moved}"
        )
        s2 = step_body(rig, 2)
        assert current_images(s2) == [builder_url(fresh)], (
            "B3: step 2 carries exactly the fresh frame"
        )
        assert data_url(watch.frames[0].jpeg) not in str(s2), "B3: the onset image is replaced"
        assert label_after(s2, fresh) == L1, "B3/T-0134: the fresh frame is labelled LABEL_NOW"
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "click_at", CLICK_ARGS, status("refused", FRESHNESS, True))]
        ), step_tail(rig, 2)[-300:]
        c = card(rig, "t1-call")
        assert (c.status, c.reason) == ("refused", FRESHNESS), c
        assert got[1:] == [("executed", None)], f"B3: the click on the fresh frame executes: {got}"
        assert d.presses == [POINT], f"B3: the refused click pressed nothing: {d.presses}"
    assert refresh_notes(rig) == [
        {"what": REFRESHED, "reason_kind": "FRESHNESS_REASON", "tool_step": 1}
    ], f"T-0187 criterion 7: one refresh note; notes {rig.kind('note')}"


#: T-0204 B's closed range (T-0207 ruling A2-R2; tests.md "pin both ends"): M = ``Merge T-0204-b`` cbb88b4b,
#: TIP = M^2 = a3e61ffb (the B-R4 implementation tip), RANGE_BASE = merge-base(M^1, M^2) = 0efb0f58 (the branch merged
#: main at 971728b7, so the merge base is later than the declared fork point ``BASE_SHA`` e715c339, also checked).
B_MERGE = "cbb88b4b"
B_TIP = "a3e61ffb8a7416fc5c85a587eb548dd7fb870651"
B_RANGE_BASE = "0efb0f582e7dddf6026b5dc3994a957a496d9e54"
#: a later commit that holds M (T-0207 A2's BASE), and a known conductor change for the power row (T-0207 A2's impl)
B_LATER = "5a0602f2454526bba06db59aa0d200c6ea811853"
CONDUCTOR_CHANGED = (
    "5a0602f2454526bba06db59aa0d200c6ea811853",
    "3281676c431a3810b86735a43e0fa58d52fb36f2",
)


def test_conductor_diff_is_empty() -> None:
    """B3 (negative): no ``session.py`` (or any conductor) change over T-0204 B's closed range: ``local_astra/conductor``
    is identical between the range base and B's tip, and between the declared fork point ``BASE_SHA`` and the tip
    (both literals, never the working tree). Power: the same predicate sees T-0207 A2's ``session.py`` change."""
    gr.prove_range(B_RANGE_BASE, B_TIP, B_MERGE, later=B_LATER)
    gr.require_commits(BASE_SHA, *CONDUCTOR_CHANGED)
    assert gr.is_ancestor(BASE_SHA, B_TIP), f"{BASE_SHA[:8]} is not an ancestor of {B_TIP[:8]}"
    for base in (B_RANGE_BASE, BASE_SHA):
        changed = gr.changed(base, B_TIP, "local_astra/conductor")
        assert changed == [], (
            f"T-0204 B3: the conductor changed in {base[:8]}..{B_TIP[:8]}: {changed}"
        )
    assert "local_astra/conductor/session.py" in gr.changed(
        *CONDUCTOR_CHANGED, "local_astra/conductor"
    ), "power"
