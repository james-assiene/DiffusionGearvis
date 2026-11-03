"""T-0177 group B, B1-B6: the tool loop's step images under the frozen ``shared`` choice (docs/tasks/T-0177.md).

The rig is group A's ``vision_rig`` (``test_t0177_shared_frame``): the live tool rig over a ``VisionBrowser`` (the fake
live browser), the HUD perception serving the distinctive shared frame ``SH``, a socketless ``HudServer`` and one
greeted page. Selects go over the wire.

Expected values: images are compared with the data URL the real ``PromptBuilder`` writes for a frame
(``builder_url``); the labels are T-0134's literals (``test_t0134_label``) and A7's (``test_t0177_shared_frame``),
typed from the tasks; ``SHARED_FRAME_REASON`` is typed here from B2, never imported.

Readings (listed in the report, not guessed silently):

- B2 "step k+1 shows a fresh browser frame ... labelled ``LABEL_NOW``": the label rows run with the screen label on
  (``live_feedback(screen_label=True)``); with it off there is no browser label at all (T-0134). The refused step's
  payload ``screenshot`` value is not pinned (the criterion names the label only).
- B4's absent direction (a pointer action measured although its before frame is shared) has no public path once B2
  holds: every ``click_at``/``type_at`` read off a shared image is refused before dispatch, and only executed pointer
  actions are measured. The B4 row pins the too-wide direction (an action whose frames are both browser frames, in a
  shared turn, is measured and never noted ``shared_frame``).
- B6's adversarial row (three turns, a step-1 click off the same shared frame each time, no ``recurring`` stop) holds
  at BASE: the guard's history is per turn. It is kept as the criterion's named row; the discriminating B6 row is
  ``test_a_shared_screen_is_never_a_screen_visit`` (``max_screen_visits``).
- B5: at a shared step 1 no action has settled, so the "no ``page``/``screenshot`` fact" row holds at BASE by
  construction; its companion pins that an executed browser action in a shared turn still carries ``page``.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools live, loop on (8 steps), n = 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.perception import Source
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    FRESHNESS,
    b64,
    builder_url,
    image_urls,
    live_feedback,
    live_frame,
    refused,
    scene_frame,
    turn_extras,
    turn_logs,
)
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_live import ASK, NO, SCROLL, _wire, last_user
from tests.unit.test_t0134_label import L1, LN, LU, label_after
from tests.unit.test_t0177_shared_frame import (
    SH,
    SHARED_LABEL_SCREEN,
    VisionBrowser,
    select,
    shared_frame,
    vision_rig,
)

Q = tsl.Q
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
#: B2, typed from the task
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)
PRESS = f"<|tool_call>call:press_key{{combo:{Q}Ctrl+C{Q}}}<tool_call|>"
TYPE_HI = f"<|tool_call>call:type_at{{press_enter:false,text:{Q}hello{Q},x:500,y:250}}<tool_call|>"
CLICK2 = tsl.CLICK2
SEARCH = tsl.SEARCH_CALL
#: T-0136 B-3's clause stem and T-0137's recurring reason stem (typed; a substring of no other rig text)
CLAUSE_STEM = "Compared with the previous screenshot"

F0 = live_frame("t0177b-onset")
F1 = live_frame("t0177b-after-1")
F2 = live_frame("t0177b-after-2")
FR = live_frame("t0177b-fresh")


# --------------------------------------------------------------------------------------------- reading the rig


def step_body(rig: Any, k: int, turn: str = "t1") -> dict[str, Any]:
    return _wire(tsl.step_request(rig, k, turn))


def gen_body(rig: Any, turn: str = "t1") -> dict[str, Any]:
    return _wire(tsl.sent(rig, f"{turn}-gen"))


def current_images(body: dict[str, Any]) -> list[str]:
    """The image URLs of the body's final user message, in order (the step's current image(s))."""
    return [
        p["image_url"]["url"] for p in last_user(body)["content"] if p.get("type") == "image_url"
    ]


def text_after_current(body: dict[str, Any]) -> str | None:
    """The text part right after the final user message's last image, or ``None``."""
    parts = last_user(body)["content"]
    idx = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
    if not idx or idx[-1] + 1 >= len(parts) or parts[idx[-1] + 1].get("type") != "text":
        return None
    return str(parts[idx[-1] + 1]["text"])


def card(rig: Any, call_id: str) -> Any:
    msgs = rig.tool_calls(call_id)
    assert msgs, f"no astra.tool.call for {call_id}"
    return msgs[-1]


def frame_calls(browser: Any) -> int:
    return sum(1 for c in browser.calls if c[0] == "frame")


def executes(browser: Any, call_id: str) -> int:
    return sum(1 for c in browser.calls if c == ("execute", call_id))


async def loop_turn(
    rig: Any, steps: tuple[Any, ...], *, n: int, question: str = tsl.QUESTION
) -> None:
    """Ask; wait for ``n`` step requests and the turn's end."""
    rig.llm.script_steps(*steps)
    await rig.ask(question)
    if n:
        await tsl.until_steps(rig, n, limit_ms=20_000, step=100)
    await rig.turn_done(limit_ms=20_000)


def label_on() -> Any:
    return live_feedback(screen_label=True, page_extent=False)


# =====================================================================================================
# B1: follow the action
# =====================================================================================================


@pytest.mark.parametrize("call", [SCROLL, PRESS], ids=["scroll", "press_key"])
@pytest.mark.parametrize("feedback", ["off", "label"])
async def test_an_executed_action_swaps_in_the_browser_frame(
    tmp_path: Path, call: str, feedback: str
) -> None:
    """B1: under ``shared`` a step-1 ``scroll``/``press_key`` executes on the agent browser; step 2's final user
    message carries the post-action browser frame and no shared image (replaced, never kept beside it: the shared
    bytes are in no step-2 byte), and the label after it is the browser set's (``LABEL_UPDATED`` with the screen
    label on; none, and never the shared label, with it off)."""
    browser = VisionBrowser([F0], post_frames=[F1])
    kw = {"live_feedback": label_on()} if feedback == "label" else {}
    async with vision_rig(
        tmp_path, browser, script=[answer(S1), tool(call)], session_kwargs=kw
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        assert current_images(gen_body(rig)) == [builder_url(SH)], "premise: step 1 is shared"
        assert executes(browser, "t1-call") == 1, f"premise: the action executed; {browser.calls}"
        s2 = step_body(rig, 2)
    assert current_images(s2) == [builder_url(F1)], "B1: step 2 shows the post-action browser frame"
    assert b64(SH.jpeg) not in json.dumps(s2), (
        "B1: the shared image is replaced, never kept beside it"
    )
    after = text_after_current(s2)
    assert after != SHARED_LABEL_SCREEN and SHARED_LABEL_SCREEN not in json.dumps(s2), after
    if feedback == "label":
        assert label_after(s2, F1) == LU, "B1: the browser label set (LABEL_UPDATED)"
    else:
        assert after is None or not after.startswith("[This screenshot"), after


async def test_a_search_keeps_the_shared_image_and_its_label(tmp_path: Path) -> None:
    """B1: a ``google_search`` leaves the shared image and the shared label in place at step 2, and asks the browser
    for no frame."""
    async with vision_rig(tmp_path, script=[answer(S1), tool(SEARCH)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        s2 = step_body(rig, 2)
        assert current_images(s2) == [builder_url(SH)], "B1: the shared image stays after a search"
        assert text_after_current(s2) == SHARED_LABEL_SCREEN
        assert frame_calls(rig.browser) == 0, rig.browser.calls


async def test_a_refused_non_coordinate_call_keeps_the_shared_image(tmp_path: Path) -> None:
    """B1: a ``scroll`` the executor refuses leaves the shared image and its label in place at step 2 (no fresh
    frame: a refusal is not an action). T-0187 R3 (owned re-scope): the refusal reason is one outside the "take a new
    frame" set; a page-changed refusal on the shared screen is followed by a fresh browser frame."""
    browser = VisionBrowser([F0])
    browser.outcome("t1-call", lambda inv: refused(inv, "bad arguments"))
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        s2 = step_body(rig, 2)
    assert current_images(s2) == [builder_url(SH)], (
        "B1: a refused non-coordinate call keeps the shared image"
    )
    assert text_after_current(s2) == SHARED_LABEL_SCREEN
    assert frame_calls(browser) == 0, browser.calls


async def test_after_the_swap_the_turn_is_a_browser_turn_and_the_next_turn_is_shared_again(
    tmp_path: Path,
) -> None:
    """B1 "the rest of the turn behaves exactly like a browser turn" and B2's too-wide row: after the step-1 scroll
    swaps in ``F1``, the step-2 click read off ``F1`` is executed (its geometry is ``F1``'s), step 3 shows ``F2``
    labelled ``LABEL_UPDATED``. The next turn starts again from the frozen choice: its step 1 is the (new) shared
    frame, and an action never changed the choice (the line says ``shared``)."""
    later = shared_frame("t0177b-turn2", ts_ms=0)
    browser = VisionBrowser([F0], post_frames=[F1, F2])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(SCROLL)],
        session_kwargs={"live_feedback": label_on()},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]), n=2)
        assert current_images(step_body(rig, 2)) == [builder_url(F1)], "premise: the swap"
        invs = [i for i in browser.invocations if i.call_id == "t1-call-s2"]
        assert len(invs) == 1 and invs[0].frame is not None and invs[0].frame.phash == F1.phash, (
            f"B2 too wide: the click read off the browser frame is dispatched on it; {browser.calls}"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"]
        s3 = step_body(rig, 3)
        assert current_images(s3) == [builder_url(F2)] and label_after(s3, F2) == LU
        assert turn_extras(rig).get("vision_source") == "shared"
        await rig.advance(300)
        rig.perception.add(later.model_copy(update={"ts_ms": rig.clock.now_ms}))
        rig.llm.script = [answer("A shop. ")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        t2 = gen_body(rig, "t2")
    assert current_images(t2) == [builder_url(later)], (
        "B1: the next turn starts from the frozen shared choice"
    )
    assert turn_extras(rig, "t2").get("vision_source") == "shared"


# =====================================================================================================
# B2: coordinates read off a shared image
# =====================================================================================================


@pytest.mark.parametrize("call", [VALID_CLICK, TYPE_HI], ids=["click_at", "type_at"])
async def test_a_coordinate_call_off_the_shared_image_is_refused_with_the_shared_reason(
    tmp_path: Path, call: str
) -> None:
    """B2 (and the "not caught" row: a step-1 click read off the shared image is refused with this reason, not
    B9's): nothing is dispatched, the loop result is ``refused``, the card and the ``tool.call`` record carry exactly
    ``SHARED_FRAME_REASON``, step 2's response states it, and the freshness reason appears nowhere."""
    browser = VisionBrowser([FR])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(call)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        assert executes(browser, "t1-call") == 0, f"B2: nothing dispatched; {browser.calls}"
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["refused"]
        c = card(rig, "t1-call")
        assert (c.status, c.reason) == ("refused", SHARED_FRAME_REASON), c
        tail = tsl.tail(tsl.step_request(rig, 2))
        assert f"reason:{Q}{SHARED_FRAME_REASON}{Q}" in tail, tail[-400:]
        assert FRESHNESS not in tail, "B2: not B9's reason"
    records = [d for d in rig.tool_records() if d.get("call_id") == "t1-call"]
    assert records and (records[-1].get("status"), records[-1].get("reason")) == (
        "refused",
        SHARED_FRAME_REASON,
    ), records[-1:]


async def test_after_the_refusal_step2_shows_a_fresh_browser_frame_labelled_now(
    tmp_path: Path,
) -> None:
    """B2: after the refusal step 2 shows a fresh browser frame (one ``frame()`` call, after the refusal), labelled
    ``LABEL_NOW`` -- never ``LABEL_UPDATED`` -- with no outcome clause (``action_outcome`` on), and no shared byte.
    Its click, read off the fresh frame, is then executed (the fresh frame is a real own frame)."""
    browser = VisionBrowser([FR], post_frames=[F1])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"live_feedback": label_on(), "action_outcome": True},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]), n=2)
        s2 = step_body(rig, 2)
        assert frame_calls(browser) == 1, f"B2: one fresh frame after the refusal; {browser.calls}"
        assert current_images(s2) == [builder_url(FR)], "B2: step 2 shows the fresh browser frame"
        assert b64(SH.jpeg) not in json.dumps(s2), "B2: never both images"
        got = label_after(s2, FR)
        assert got == L1 and got not in (LU, LN), f"B2: LABEL_NOW, never LABEL_UPDATED; {got}"
        assert CLAUSE_STEM not in json.dumps(s2), "B2: no outcome clause"
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"], (
            f"the click read off the fresh frame executes; {browser.calls}"
        )


async def test_with_no_fresh_frame_the_shared_image_stays_and_a_note_is_written(
    tmp_path: Path,
) -> None:
    """B2: when no fresh frame can be taken (``frame()`` raises) the shared image and its label stay at step 2, the
    browser was asked once, and a note is written."""
    browser = VisionBrowser([])
    browser.frame_mode = "raise"
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        s2 = step_body(rig, 2)
        assert card(rig, "t1-call").reason == SHARED_FRAME_REASON, card(rig, "t1-call")
        assert frame_calls(browser) == 1, f"B2: a fresh frame was attempted once; {browser.calls}"
        assert current_images(s2) == [builder_url(SH)], "B2: the shared image stays"
        assert text_after_current(s2) == SHARED_LABEL_SCREEN
    notes = [str(n.get("what", "")) for n in rig.kind("note")]
    assert [n for n in notes if "frame" in n.lower()], f"B2: a note is written; notes {notes}"


async def test_no_reground_pass_two_for_a_shared_click(tmp_path: Path) -> None:
    """B2 "refused before ReGround pass 2": with ``reground`` on, the shared-image click sends no pass-2 request
    (no ``tool.reground`` record, no request beyond step 1, step 2 and their prefill) and is refused with the
    shared reason."""
    browser = VisionBrowser([FR])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"reground": True},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        assert card(rig, "t1-call").reason == SHARED_FRAME_REASON, card(rig, "t1-call")
        ids = [rid for rid, _, _ in rig.llm.generates]
    assert rig.kind("tool.reground") == [], rig.kind("tool.reground")
    assert not [r for r in ids if "t1-gen" not in r], f"B2: no pass-2 request; generates {ids}"


@pytest.mark.parametrize("case", ["ask_first", "user_declines"])
async def test_ask_first_and_decline_keep_precedence(tmp_path: Path, case: str) -> None:
    """B2 "Ask-first (T-0107a) and decline (T-0107e) keep precedence": under ``shared`` their reasons, verbatim, are
    the card's, never the shared reason; nothing is executed."""
    browser = VisionBrowser([FR])
    script = [answer(ASK if case == "ask_first" else S1), tool(VALID_CLICK)]
    async with vision_rig(tmp_path, browser, script=script) as rig:
        await select(rig, "shared")
        await rig.ask(tsl.QUESTION if case == "ask_first" else NO)
        await rig.turn_done(limit_ms=20_000)
        assert browser.executes() == 0, browser.calls
        want = "asked the user first" if case == "ask_first" else "the user said no"
        assert card(rig, "t1-call").reason == want, card(rig, "t1-call")


async def test_a_click_in_a_browser_turn_after_a_shared_turn_is_executed(tmp_path: Path) -> None:
    """B2 too wide (the frame-kind fact, not the session's history): turn 1 is shared (its image stays in history,
    H-T177-4); after a select to ``browser`` turn 2's step-1 click, read off the browser's own frame, is executed on
    that frame."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done()
        await rig.advance(300)
        await select(rig, "browser")
        rig.llm.script = [answer(S1), tool(VALID_CLICK)]
        rig.llm.script_steps([answer(S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1, turn="t2", limit_ms=20_000, step=100)
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        t2 = gen_body(rig, "t2")
        assert current_images(t2) == [builder_url(F0)], "premise: turn 2 shows the browser frame"
        assert builder_url(SH) in image_urls(t2), "premise: turn 1's shared image is in history"
        assert executes(browser, "t2-call") == 1, browser.calls
        assert [e.status for e in tsl.results_for(rig, "t2-call")] == ["executed"]


async def test_a_click_off_a_shared_frame_riding_on_history_is_refused(tmp_path: Path) -> None:
    """B2 (adversarial: the current image rides on a held history image). Turn 1 is shared over ``SH``; turn 2 is
    shared over the same bytes, so its frame rides on turn 1's history image and the final user message carries no
    image. Its step-1 click is still read off the shared frame: refused with the shared reason, nothing
    dispatched."""
    browser = VisionBrowser([FR, live_frame("t0177b-fresh-2")])
    async with vision_rig(tmp_path, browser, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done()
        await rig.advance(300)
        rig.llm.script = [answer(S1), tool(VALID_CLICK)]
        rig.llm.script_steps([answer(S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1, turn="t2", limit_ms=20_000, step=100)
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        t2 = gen_body(rig, "t2")
        assert current_images(t2) == [] and image_urls(t2) == [builder_url(SH)], (
            "premise: turn 2's shared frame rides on turn 1's history image"
        )
        assert executes(browser, "t2-call") == 0, browser.calls
        assert card(rig, "t2-call").reason == SHARED_FRAME_REASON, card(rig, "t2-call")


# =====================================================================================================
# B4: outcome comparison
# =====================================================================================================

#: real JPEG own frames (T-0136's scenes): a white screen, and a black box at the click point of CLICK2 (153, 512)
WHITE = scene_frame("t0177b-white")
TICK2 = scene_frame("t0177b-tick2", blocks=[(137, 500, 169, 524, (0, 0, 0))])


async def test_an_action_on_browser_frames_in_a_shared_turn_is_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 too wide: in a shared turn, after the B2 refusal at step 1, the step-2 click is read off the fresh browser
    frame and its after frame is a browser frame: it is measured (``action_outcomes`` holds step 2, ``near``: a box
    at the click point), never unmeasured ``shared_frame``; no note carries ``shared_frame`` for it."""
    spy = turn_logs(monkeypatch)
    browser = VisionBrowser([WHITE], post_frames=[TICK2])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"live_feedback": label_on(), "action_outcome": True},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]), n=2)
        assert card(rig, "t1-call").reason == SHARED_FRAME_REASON, "premise: B2 at step 1"
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"]
    log = spy.by_turn("t1")
    assert 2 in log.action_outcomes and log.action_outcomes[2].kind == "near", (
        f"B4: measured; outcomes {log.action_outcomes}, unmeasured {log.action_outcome_unmeasured}"
    )
    assert log.action_outcome_unmeasured.get(2) is None, log.action_outcome_unmeasured
    shared_notes = [n for n in rig.kind("note") if "shared_frame" in json.dumps(n)]
    assert shared_notes == [], shared_notes


# =====================================================================================================
# B5: page facts
# =====================================================================================================


def _navigated(fb: Any, inv: Any) -> Any:
    out = fb.perform(inv)
    return out.model_copy(update={"resolved": {**out.resolved, "navigated": True}})


async def test_a_shared_step1_carries_no_page_or_screenshot_fact(tmp_path: Path) -> None:
    """B5 (the criterion's row; holds at BASE by construction, no action has settled): with ``page_fact`` on, the
    shared step-1 request carries no ``page`` and no ``screenshot`` fact. Companion (too wide): the executed step-1
    scroll, a browser action, still gives step 2 ``page: "new page loaded"``."""
    browser = VisionBrowser([F0], post_frames=[F1])
    browser.handlers["t1-call"] = _navigated
    async with vision_rig(
        tmp_path, browser, script=[answer(S1), tool(SCROLL)], session_kwargs={"page_fact": True}
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        g = json.dumps(gen_body(rig))
        assert "new page loaded" not in g and f"screenshot:{Q}" not in g, "B5: no fact at step 1"
        tail = tsl.tail(tsl.step_request(rig, 2))
    assert f"page:{Q}new page loaded{Q}" in tail, tail[-300:]


async def test_a_refused_shared_click_carries_no_page_fact(tmp_path: Path) -> None:
    """B5 "only on executed browser actions": the B2-refused click (whose ``resolved`` the session never sees) gives
    step 2 no ``page`` fact."""
    browser = VisionBrowser([FR])
    browser.handlers["t1-call"] = _navigated
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"page_fact": True},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        assert card(rig, "t1-call").reason == SHARED_FRAME_REASON, card(rig, "t1-call")
        tail = tsl.tail(tsl.step_request(rig, 2))
    assert "page loaded" not in tail, tail[-300:]


# =====================================================================================================
# B6: the repeat guard
# =====================================================================================================


@pytest.mark.parametrize("choice", ["browser", "shared"])
async def test_a_shared_screen_is_never_a_screen_visit(tmp_path: Path, choice: str) -> None:
    """B6: ``repeat_guard`` on; step 1 searches (the image stays), step 2 scrolls (a judged browser action read off
    the step-1 image). Under ``shared`` that image is the shared frame: ``_guard_screen`` is ``None``, so the judged
    call has no screen visit (``max_screen_visits`` 0). Too wide (control): under ``browser`` the same call is a
    first visit to the browser screen (``max_screen_visits`` 1)."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(SEARCH)],
        session_kwargs={"repeat_guard": True},
    ) as rig:
        if choice == "shared":
            await select(rig, "shared")
        await loop_turn(rig, ([answer(S2), tool(SCROLL)], [answer(S3)]), n=2)
        assert executes(browser, "t1-call-s2") == 1, f"premise: the scroll ran; {browser.calls}"
        s2 = step_body(rig, 2)
        want = builder_url(SH) if choice == "shared" else builder_url(F0)
        assert current_images(s2) == [want], "premise: step 2 still shows the step-1 image"
        counts = turn_extras(rig).get("repeat_guard")
    assert isinstance(counts, dict), counts
    assert counts.get("max_screen_visits") == (0 if choice == "shared" else 1), (
        f"B6: shared frames never count as screen visits; {counts}"
    )


async def test_three_step1_clicks_off_one_shared_frame_across_turns_never_stop_recurring(
    tmp_path: Path,
) -> None:
    """B6 adversarial (the criterion's row): three turns, each a step-1 click off the same shared frame (refused,
    B2), then a step-2 click off the fresh browser frame; no turn has a ``recurring`` stop and no card says so.
    Holds at BASE (the guard's history is per turn): see the module docstring."""
    frames = [live_frame(f"t0177b-fresh-{i}") for i in range(3)]
    browser = VisionBrowser(frames)
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"repeat_guard": True},
    ) as rig:
        await select(rig, "shared")
        for n in (1, 2, 3):
            if n > 1:
                await rig.advance(300)
                rig.llm.script = [answer(S1), tool(VALID_CLICK)]
            rig.llm.script_steps([answer(S2), tool(CLICK2)], [answer(S3)])
            await rig.ask(tsl.QUESTION)
            await tsl.until_steps(rig, 2, turn=f"t{n}", limit_ms=20_000, step=100)
            await rig.until(
                lambda n=n: len(rig.lines()) >= n, f"turn {n} never ended", limit_ms=20_000
            )
        for n in (1, 2, 3):
            counts = turn_extras(rig, f"t{n}").get("repeat_guard") or {}
            assert counts.get("stops_recurring") == 0, (n, counts)
        reasons = [str(getattr(m, "reason", "")) for m in rig.tool_calls()]
        assert not [r for r in reasons if "recur" in r.lower() or "same screen" in r.lower()], (
            reasons
        )


async def test_the_shared_label_stays_on_a_camera_frame_after_a_search(tmp_path: Path) -> None:
    """B1 with a camera source: a search keeps the camera image and ``SHARED_LABEL_CAMERA`` (A7) at step 2."""
    from tests.unit.test_t0177_shared_frame import SHARED_LABEL_CAMERA

    cam = shared_frame("t0177b-camera", source=Source.CAMERA, width=896, height=504)
    async with vision_rig(tmp_path, script=[answer(S1), tool(SEARCH)], hud_frames=[cam]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        s2 = step_body(rig, 2)
    assert (
        current_images(s2) == [builder_url(cam)] and text_after_current(s2) == SHARED_LABEL_CAMERA
    )


# =====================================================================================================
# Group B review rulings B-R2..B-R4 (docs/tasks/T-0177.md "## Group B review rulings")
# =====================================================================================================

#: T-0136 B-3's clause, typed; with LABEL_UPDATED it reads as one bracketed label (T-0136's EXAMPLE shape)
OUTCOME_CLAUSE = " Compared with the previous screenshot: {outcome}."
#: T-0136 A2's text of a ``near`` outcome (a change touching the click's near box), typed
NEAR_TEXT = "changed near your click"


async def test_label_now_is_read_once_step3_is_updated_with_its_outcome(tmp_path: Path) -> None:
    """B-R2: LABEL_NOW is read once. Step 1's click off the shared frame is refused (B2), step 2 shows the fresh
    browser frame ``WHITE`` labelled LABEL_NOW; step 2's click (read off ``WHITE``) is executed and leaves ``TICK2``
    (a black box at the click point). Step 3's label is LABEL_UPDATED with the measured outcome clause, exactly
    ``[<LABEL_UPDATED> Compared with the previous screenshot: changed near your click.]``."""
    browser = VisionBrowser([WHITE], post_frames=[TICK2])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"live_feedback": label_on(), "action_outcome": True},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]), n=2)
        assert card(rig, "t1-call").reason == SHARED_FRAME_REASON, "premise: B2 at step 1"
        assert label_after(step_body(rig, 2), WHITE) == L1, "premise: step 2 is LABEL_NOW"
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"]
        s3 = step_body(rig, 3)
    want = "[" + LU[1:-1] + OUTCOME_CLAUSE.format(outcome=NEAR_TEXT) + "]"
    assert current_images(s3) == [builder_url(TICK2)], "premise: step 3 shows the post-click frame"
    assert label_after(s3, TICK2) == want, (
        f"B-R2: after LABEL_NOW the next executed browser step is LABEL_UPDATED with its outcome clause; "
        f"got {label_after(s3, TICK2)!r}"
    )


@pytest.mark.parametrize("fresh", [True, False], ids=["fresh-frame", "no-frame"])
async def test_the_refused_steps_payload_says_whether_the_image_was_replaced(
    tmp_path: Path, fresh: bool
) -> None:
    """B-R3 (first half): the ``shared_refused`` step's response keeps ``screenshot: "updated"`` when a fresh browser
    frame replaced the shared image at step 2 (``loop.screenshot_payload``'s definition), and ``"unchanged"`` when
    none could be taken (``frame()`` raises; the shared image stays)."""
    browser = VisionBrowser([FR] if fresh else [])
    if not fresh:
        browser.frame_mode = "raise"
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        assert card(rig, "t1-call").reason == SHARED_FRAME_REASON, "premise: B2 at step 1"
        s2 = step_body(rig, 2)
        tail = tsl.tail(tsl.step_request(rig, 2))
    shown = builder_url(FR) if fresh else builder_url(SH)
    assert current_images(s2) == [shown], "premise: step 2's image"
    want, other = ("updated", "unchanged") if fresh else ("unchanged", "updated")
    assert f"screenshot:{Q}{want}{Q}" in tail and f"screenshot:{Q}{other}{Q}" not in tail, (
        f"B-R3: the refused step's payload says screenshot {want!r}; tail {tail[-400:]}"
    )


def _spy_guard_outcome(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int | None, bool, Any]]:
    """Every ``Session._guard_outcome`` call: (the guard step it targets -- the newest -- , ``updated``, the turn)."""
    from local_astra.conductor.session import Session

    calls: list[tuple[int | None, bool, Any]] = []
    real = Session._guard_outcome

    def spy(self: Any, turn: Any, updated: bool) -> None:
        key = next(reversed(turn.guard_steps)) if turn.guard_steps else None
        calls.append((key, updated, turn))
        real(self, turn, updated)

    monkeypatch.setattr(Session, "_guard_outcome", spy)
    return calls


class _OnceBlindBrowser(VisionBrowser):
    """``frame()`` raises on its first call only (a capture that fails once, then works)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.failed_once = False

    async def frame(self) -> Any:
        if not self.failed_once:
            self.failed_once = True
            self.calls.append(("frame",))
            raise RuntimeError("the fake browser could not take a frame (once)")
        return await super().frame()


async def test_a_refused_shared_call_never_rewrites_an_earlier_guard_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-R3 (second half: "It never calls ``_guard_outcome``: an earlier guard step's ``no_change``/``outcome`` is
    never rewritten by a refused call"). ``repeat_guard`` on, shared. Step 1 scrolls (executed on the browser); its
    post-action frame is not the browser's latest and the fresh frame fails once, so the shared image stays at step
    2 and the scroll's guard step reads no change (``screenshot: "unchanged"``). Step 2's click is read off the
    shared image: refused (B2); the fresh frame ``FR`` now works and replaces it. Step 3's click off ``FR`` is
    executed. ``_guard_outcome`` is called after step 1 (targeting step 1) and after step 3 (targeting step 3),
    never after the refused step 2; step 1's guard step still reads ``no_change`` True, ``{"screenshot":
    "unchanged"}`` at the end of the turn.

    (Reading: no turn line field or later payload exposes the outcome of a step read off the shared image -- its
    screen is ``None`` (B6), so it is never an instance of a repeat -- and once a browser frame is shown no shared
    image returns within the turn (B1), so the ruling's ``_guard_outcome`` is pinned by name with a spy.)"""
    calls = _spy_guard_outcome(monkeypatch)
    browser = _OnceBlindBrowser([FR], post_frames=[F1, F2])

    def scrolled_elsewhere(fb: Any, inv: Any) -> Any:
        out = fb.perform(inv)
        return out.model_copy(
            update={"resolved": {**out.resolved, "frame_after": "not-the-latest"}}
        )

    browser.handlers["t1-call"] = scrolled_elsewhere
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(SCROLL)],
        session_kwargs={"repeat_guard": True},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(
            rig, ([answer(S2), tool(VALID_CLICK)], [answer(S2), tool(CLICK2)], [answer(S3)]), n=3
        )
        assert executes(browser, "t1-call") == 1, f"premise: the step-1 scroll ran; {browser.calls}"
        assert current_images(step_body(rig, 2)) == [builder_url(SH)], (
            "premise: the shared image stayed"
        )
        assert card(rig, "t1-call-s2").reason == SHARED_FRAME_REASON, "premise: B2 at step 2"
        assert current_images(step_body(rig, 3)) == [builder_url(FR)], "premise: FR replaced it"
        assert [e.status for e in tsl.results_for(rig, "t1-call-s3")] == ["executed"]
    assert calls, "premise: the guard filled step outcomes"
    first = calls[0][2].guard_steps[1]
    assert (first.no_change, first.outcome) == (True, {"screenshot": "unchanged"}), (
        f"B-R3: the earlier guard step is not rewritten by the refused call; {first}"
    )
    assert [(key, updated) for key, updated, _ in calls] == [(1, False), (3, True)], (
        f"B-R3: _guard_outcome is never called for a shared_refused step; calls (target, updated) "
        f"{[(key, updated) for key, updated, _ in calls]}"
    )


async def test_a_refused_shared_call_never_enters_the_repeat_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-R4: the B2 refusal comes before the T-0137 guard. ``repeat_guard`` on, shared: step 1 searches (the shared
    image stays), step 2's click off it is refused (B2, at loop step 2, where the guard judges), step 3's click off
    the fresh frame ``FR`` is executed. The guard judges exactly one call -- step 3's click, on ``FR``'s screen --
    and the history it judges over holds no step 2; the turn line's ``repeat_guard`` counts are those of a turn whose
    only judged action is one first visit (no repeats, no stops, ``max_screen_visits`` 1)."""
    from local_astra.conductor import progress

    judged: list[tuple[str, dict[str, Any], str | None, list[int]]] = []
    real_judge = progress.judge

    def spy(history: Any, name: str, arguments: Any, screen: Any, *, overview: bool) -> Any:
        judged.append((name, dict(arguments), screen, [s.step for s in history]))
        return real_judge(history, name, arguments, screen, overview=overview)

    monkeypatch.setattr(progress, "judge", spy)
    browser = VisionBrowser([FR], post_frames=[F1])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(SEARCH)],
        session_kwargs={"repeat_guard": True},
    ) as rig:
        await select(rig, "shared")
        await loop_turn(
            rig, ([answer(S2), tool(VALID_CLICK)], [answer(S2), tool(CLICK2)], [answer(S3)]), n=3
        )
        assert current_images(step_body(rig, 2)) == [builder_url(SH)], "premise: the search kept SH"
        assert card(rig, "t1-call-s2").reason == SHARED_FRAME_REASON, "premise: B2 at step 2"
        assert executes(browser, "t1-call-s2") == 0, browser.calls
        assert [e.status for e in tsl.results_for(rig, "t1-call-s3")] == ["executed"]
        counts = turn_extras(rig).get("repeat_guard")
    assert [(name, args.get("x"), args.get("y")) for name, args, _, _ in judged] == [
        ("click_at", 120, 640)
    ], f"B-R4: only step 3's click is judged, never the refused shared click; judged {judged}"
    assert judged[0][2] is not None, "premise: step 3's click has a browser screen"
    assert 2 not in judged[0][3], (
        f"B-R4: the refused shared call never enters the guard's steps; history steps {judged[0][3]}"
    )
    assert counts == {
        "repeats": 0,
        "no_change_repeats": 0,
        "stops_repeat": 0,
        "stops_recurring": 0,
        "max_screen_visits": 1,
    }, f"B-R4: the turn line's repeat_guard counts; {counts}"
