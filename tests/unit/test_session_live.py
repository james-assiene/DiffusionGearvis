"""T-0111g B5-B15 (session side): ``tools = live`` over the fake live browser (docs/tasks/T-0111g.md).

Every row runs ``tests/unit/_live_browser_support.live_rig``: the T-0114 ``tool_rig`` with the loop on and a live kit
whose ``executor`` and ``live_browser`` are one ``FakeLiveBrowser``; the HUD perception serves a sentinel frame that
must never reach a request. The interfaces the rows assume are pinned in that module's docstring.

Expected values: request images are compared with the data URL the real ``PromptBuilder`` writes for the frame
(``builder_url``, the B8 oracle); continuation texts with ``loop.step_prompt``/``render_response`` (T-0113's, proven
against the template) over payloads written out by hand: BASE's ``{action, status}`` or ``status_payload`` plus
``screenshot``. Nothing is computed with the swap or the payload builder under test.

Readings (reported):

- the "fresh frame" after ``failed``/``timeout`` makes the swapped image new, so that step's payload says
  ``screenshot: "updated"``; a ``refused`` step's says ``"unchanged"``;
- B11's precedence: with the browser closed, every live call (a coordinate call included) is refused with A2's
  reason, ahead of B9's freshness reason;
- B13's ``tool.cancel(interrupted)`` cannot be observed as a machine command: after R11 the machine holds no
  pending ``tool`` (``state.tool is None`` while the call runs) and ``turn_machine.py`` is frozen, so a barge-in
  during an execute emits no ``ToolCancel``; the row pins what is observable (the execute is cancelled, no result
  is submitted, the next onset frame is fresh);
- B15's adversarial secret is carried by the fake's egress entries (an extra ``url``/``path`` key the real executor
  never writes): the session must project to ``{host, scheme}``.

Every row is ``pending(task="T-0111g")``; at BASE each fails on the kit's missing ``live_browser`` field.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), n = 1.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.conductor.prompt_builder import NO_IMAGE_STATEMENT
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.perception import AssistFlags, Source
from local_astra.contracts.protocol import AstraAssistToggle, AstraFrameText
from local_astra.contracts.turn import TurnState
from local_astra.toolcall import loop
from tests.fakes.llm import _refusing_transport
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    ADOPTION,
    CLOSED,
    FRESHNESS,
    RESTARTED,
    FakeLiveBrowser,
    b64,
    builder_url,
    failed,
    hud_frame,
    image_urls,
    live_frame,
    live_frame_timeout_ms,
    live_rig,
    refused,
    turn_extras,
)
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool, tool_rig

Q = '<|"|>'
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
CLICK2 = tsl.CLICK2
SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
CLICK_ARGS, CLICK2_ARGS = tsl.CLICK_ARGS, tsl.CLICK2_ARGS
#: the fake's action for VALID_CLICK (500, 250 on 1280x800) and CLICK2 (120, 640)
ACT1, ACT2 = "clicked at (640, 200)", "clicked at (153, 512)"
TIMEOUT_REASON = "TimeoutError during click_at"
ASK = "Shall I place the order for 42.99 euros? "
NO = "No, leave it."


# --------------------------------------------------------------------------------------------- reading the rig


def _wire(prompt: Any) -> dict[str, Any]:
    """The body the real client sends for ``prompt`` (``VllmClient.request_body``, as the fake LLM builds it)."""
    body: dict[str, Any] = VllmClient(transport=_refusing_transport()).request_body(
        prompt, "x", None, {}
    )
    return body


def pass1_body(rig: Any, turn: str = "t1") -> dict[str, Any]:
    """The wire body of ``turn``'s step-1 request (its last attempt)."""
    return _wire(tsl.sent(rig, f"{turn}-gen"))


def step_body(rig: Any, k: int, turn: str = "t1") -> dict[str, Any]:
    return _wire(tsl.step_request(rig, k, turn))


def stripped(body: dict[str, Any]) -> list[Any]:
    """The messages with every image URL replaced by a placeholder (everything else byte for byte)."""
    out = json.loads(json.dumps(body["messages"]))
    for m in out:
        if isinstance(m.get("content"), list):
            for part in m["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    part["image_url"]["url"] = "<image>"
    return out


def step_tail(rig: Any, k: int, turn: str = "t1") -> str:
    return tsl.tail(tsl.step_request(rig, k, turn))


def want_tail(rig: Any, steps: list[tuple[str, str, Any, Any]], turn: str = "t1") -> str:
    return tsl.tail(tsl.oracle(tsl.sent(rig, f"{turn}-gen"), steps))


def executed(action: str, updated: bool = True) -> dict[str, Any]:
    return {
        "action": action,
        "screenshot": "updated" if updated else "unchanged",
        "status": "executed",
    }


def status(kind: str, reason: str, updated: bool) -> dict[str, Any]:
    return {"reason": reason, "screenshot": "updated" if updated else "unchanged", "status": kind}


def inv_of(browser: FakeLiveBrowser, call_id: str) -> Any:
    found = [i for i in browser.invocations if i.call_id == call_id]
    assert len(found) == 1, (
        f"one execute of {call_id}; got {[i.call_id for i in browser.invocations]}"
    )
    return found[0]


def card(rig: Any, call_id: str) -> Any:
    msgs = rig.tool_calls(call_id)
    assert msgs, f"no astra.tool.call for {call_id}"
    return msgs[-1]


def frame_calls(browser: FakeLiveBrowser) -> list[int]:
    return [i for i, c in enumerate(browser.calls) if c[0] == "frame"]


def everything_logged(rig: Any) -> str:
    return json.dumps(rig.log_records)


async def three_steps(rig: Any, steps: tuple[Any, ...], *, limit_ms: int = 20_000) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(tsl.QUESTION)
    await tsl.until_steps(rig, len(steps), limit_ms=limit_ms, step=100)
    await rig.turn_done(limit_ms=limit_ms)


async def next_turn(
    rig: Any, n: int, question: str, script: list[Any], steps: tuple[Any, ...] = ()
) -> None:
    rig.llm.script = list(script)
    if steps:
        rig.llm.script_steps(*steps)
    await rig.ask(question)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"turn {n} never ended",
        limit_ms=20_000,
    )
    await rig.advance(300)


F0 = live_frame("onset")
F1 = live_frame("after-1")
F2 = live_frame("after-2")
FR = live_frame("fresh")


# =====================================================================================================
# B6: the turn's frame
# =====================================================================================================


async def test_onset_frame_is_browser_frame_hud_never_attached(tmp_path: Path) -> None:
    """B6 (adversarial, mutation 7): the onset image is the live browser's ``frame()``; the HUD sentinel frame's
    bytes appear in no request body and no logged prompt, and ``perception.get_frame`` is never asked."""
    browser = FakeLiveBrowser([F0])
    async with live_rig(tmp_path, browser, script=[answer("It is a shop. ")]) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        body = pass1_body(rig)
        assert image_urls(body) == [builder_url(F0)], image_urls(body)
        assert rig.perception.frame_calls == [], (
            f"T-0111g B6: perception.get_frame is never called in live; {rig.perception.frame_calls}"
        )
        sentinel = b64(hud_frame().jpeg)
        assert sentinel not in json.dumps(rig.llm.bodies), "the HUD frame reached a request body"
        extras = turn_extras(rig)
        assert (extras.get("frame_source"), extras.get("browser")) == ("window", "open"), extras
        assert extras.get("browser_egress_blocked") == [], extras
    assert sentinel not in everything_logged(rig), "the HUD frame reached the session log"


@pytest.mark.parametrize("mode", ["raise", "hang", "closed"])
async def test_frame_failure_no_fallback(tmp_path: Path, mode: str) -> None:
    """B6: when ``frame()`` fails, times out (``LIVE_FRAME_TIMEOUT_MS`` on the session clock: a wall-time wait
    would never end here) or the browser is closed, the turn carries no image and the no-image wording; nothing falls
    back to the HUD frame."""
    live_frame_timeout_ms()
    browser = FakeLiveBrowser([F0])
    if mode == "closed":
        browser.close_window()
    else:
        browser.frame_mode = mode
    async with live_rig(tmp_path, browser, script=[answer("I cannot see it. ")]) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done(limit_ms=20_000)
        body = pass1_body(rig)
        assert image_urls(body) == [], f"T-0111g B6 ({mode}): no image; got {len(image_urls(body))}"
        assert NO_IMAGE_STATEMENT in json.dumps(body), "the existing no-image wording applies"
        assert rig.perception.frame_calls == [], "no fallback to perception.get_frame"
        assert b64(hud_frame().jpeg) not in json.dumps(rig.llm.bodies)
        extras = turn_extras(rig)
        want = "closed" if mode == "closed" else "open"
        assert (extras.get("frame_source"), extras.get("browser")) == ("none", want), extras


# =====================================================================================================
# B7: capture identity
# =====================================================================================================


@pytest.mark.parametrize("where", ["across_turns", "within_turn"])
async def test_capture_identity_collision_both_kept(tmp_path: Path, where: str) -> None:
    """B7 (adversarial, mutation 9): two own frames with an equal phash and different bytes are two captures:
    neither is dropped as "seen", and each resolves to its own bytes (history included)."""
    a = live_frame("A", phash="00aa00aa00aa00aa")
    b = live_frame("B", phash="00aa00aa00aa00aa")
    browser = FakeLiveBrowser([a, b])
    async with live_rig(tmp_path, browser, script=[answer("Noted. ")]) as rig:
        if where == "across_turns":
            await rig.ask("look at this")
            await rig.turn_done()
            await next_turn(rig, 2, "and now", [answer("Changed. ")])
            urls = image_urls(pass1_body(rig, "t2"))
            assert urls == [builder_url(a), builder_url(b)], (
                "T-0111g B7: turn 1's image resolves to its own bytes and turn 2's same-phash frame is attached; "
                f"got {len(urls)} images, a={builder_url(a) in urls}, b={builder_url(b) in urls}"
            )
        else:
            await rig.onset()
            rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash="scene-2"))
            await rig.pump()
            await rig.words("compare these")
            await rig.end_of_turn()
            await rig.turn_done()
            assert len(frame_calls(browser)) == 2, f"premise: two snapshots; {browser.calls}"
            urls = image_urls(pass1_body(rig))
            assert builder_url(b) in urls and builder_url(a) in urls, (
                f"T-0111g B7: both captures held; a={builder_url(a) in urls}, b={builder_url(b) in urls}"
            )


async def test_identical_bytes_one_identity(tmp_path: Path) -> None:
    """B7 (the too-wide direction): identical bytes are one capture, so a turn-2 frame equal to turn 1's rides the
    history image and is not attached twice."""
    again = F0.model_copy(update={"ts_ms": 99_999})
    browser = FakeLiveBrowser([F0, again])
    async with live_rig(tmp_path, browser, script=[answer("Noted. ")]) as rig:
        await rig.ask("look at this")
        await rig.turn_done()
        await next_turn(rig, 2, "and now", [answer("Same. ")])
        urls = image_urls(pass1_body(rig, "t2"))
        assert urls == [builder_url(F0)], f"T-0111g B7: one image for one capture; got {len(urls)}"


# =====================================================================================================
# B8: the post-action frame and the swap
# =====================================================================================================


async def test_step_k1_carries_frame_after_and_geometry(tmp_path: Path) -> None:
    """B8 (mutations 1, 4, 18): step k+1's request is step 1's with only the newest image replaced by the builder's
    data URL of ``latest_frame`` (whose phash is ``frame_after``); its call's geometry phash is that ``frame_after``;
    the payload is ``{action, screenshot: "updated", status: "executed"}``."""
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2])
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        p1, s2, s3 = pass1_body(rig), step_body(rig, 2), step_body(rig, 3)
        assert image_urls(p1) == [builder_url(F0)]
        assert image_urls(s2) == [builder_url(F1)], "step 2 shows the post-action frame of step 1"
        assert image_urls(s3) == [builder_url(F2)], "step 3 shows the post-action frame of step 2"
        assert stripped(s2)[:-1] == stripped(p1) and stripped(s3)[:-1] == stripped(p1), (
            "T-0111g B8: every other byte of the step-1 prompt is unchanged"
        )
        assert inv_of(browser, "t1-call").frame.phash == F0.phash
        assert inv_of(browser, "t1-call-s2").frame.phash == F1.phash, (
            "step 2's geometry is frame_after"
        )
        s1 = (S1, "click_at", CLICK_ARGS, executed(ACT1))
        assert step_tail(rig, 2) == want_tail(rig, [s1])
        assert step_tail(rig, 3) == want_tail(
            rig, [s1, (S2, "click_at", CLICK2_ARGS, executed(ACT2))]
        )


async def test_refused_keeps_image(tmp_path: Path) -> None:
    """B8 (mutation 2): after a ``refused`` step the image and the geometry are unchanged, ``screenshot:
    "unchanged"``; the next click still resolves against the onset frame and executes. T-0187 (owned re-scope): the
    refusal reason is one outside the "take a new frame" set, whose refusals are followed by a fresh frame."""
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    browser.outcome("t1-call", lambda inv: refused(inv, "bad coordinates"))
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert image_urls(step_body(rig, 2)) == [builder_url(F0)]
        assert inv_of(browser, "t1-call-s2").frame.phash == F0.phash
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "click_at", CLICK_ARGS, status("refused", "bad coordinates", False))]
        )
        assert len(frame_calls(browser)) == 1, f"no fresh frame after a refusal; {browser.calls}"


async def test_failed_takes_fresh_frame(tmp_path: Path) -> None:
    """B8 (adversarial, mutation 3): after ``failed: browser restarted`` step 2 carries a fresh own frame of the new
    generation, taken after the result, and its click executes (without it every later click is refused)."""

    def restart(inv: Any) -> Any:
        browser.restart()
        return failed(inv, RESTARTED)

    browser = FakeLiveBrowser([F0, FR], post_frames=[F1])
    browser.outcome("t1-call", restart)
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        frames = frame_calls(browser)
        executes = [i for i, c in enumerate(browser.calls) if c == ("execute", "t1-call")]
        assert len(frames) == 2 and frames[1] > executes[0], (
            f"one fresh frame after the step; {browser.calls}"
        )
        assert image_urls(step_body(rig, 2)) == [builder_url(FR)]
        assert inv_of(browser, "t1-call-s2").frame.phash == FR.phash
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"], (
            "T-0111g B8: the click on the fresh frame executes"
        )
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "click_at", CLICK_ARGS, status("failed", RESTARTED, True))]
        )


async def test_timeout_takes_fresh_frame(tmp_path: Path) -> None:
    """B8 (mutation 3): after the session's own ``timeout`` (10 s on the session clock) step 2 carries a fresh frame
    and says ``screenshot: "updated"``; the hung execute is cancelled."""
    browser = FakeLiveBrowser([F0, FR])
    browser.hang("t1-call")
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1, limit_ms=14_000, step=250)
        await rig.turn_done()
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["timeout"]
        assert browser.cancelled == ["t1-call"]
        assert image_urls(step_body(rig, 2)) == [builder_url(FR)]
        tail = step_tail(rig, 2)
        assert f"screenshot:{Q}updated{Q},status:{Q}timeout{Q}" in tail, tail[-300:]


async def test_latest_mismatch_takes_fresh_frame(tmp_path: Path) -> None:
    """B8 (mutation 20): an executed step whose ``latest_frame`` phash differs from ``resolved["frame_after"]`` is
    never attached: a fresh frame is taken instead, and a note says so."""

    def mismatch(fb: FakeLiveBrowser, inv: Any) -> Any:
        out = fb.perform(inv)
        return out.model_copy(update={"resolved": {**out.resolved, "frame_after": "0" * 16}})

    browser = FakeLiveBrowser([F0, FR], post_frames=[F1])
    browser.handlers["t1-call"] = mismatch
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        urls = image_urls(step_body(rig, 2))
        assert builder_url(F1) not in urls, (
            "T-0111g B8: the mismatching latest_frame is never attached"
        )
        assert urls == [builder_url(FR)], "a fresh frame instead"
        assert inv_of(browser, "t1-call-s2").frame.phash == FR.phash
    assert any("frame" in str(n.get("what", "")).lower() for n in rig.kind("note")), rig.kind(
        "note"
    )


# =====================================================================================================
# B9: freshness
# =====================================================================================================


@pytest.mark.parametrize("shared", [False, True], ids=["distinct_phash", "shared_phash"])
async def test_freshness_refuses_older_own_frame(tmp_path: Path, shared: bool) -> None:
    """B9, absent direction (mutations 10, 11): the step-1 action captured a newer own frame and then failed, and no
    fresh frame can be taken, so step 2 still shows the older own frame of the same page (valid geometry, accepted by
    the executor). Its click is refused before ``execute`` with B9's reason; with a shared phash, a phash comparison
    would accept it."""
    newer = live_frame("newer", phash=F0.phash if shared else None)

    def capture_then_fail(fb: FakeLiveBrowser, inv: Any) -> Any:
        fb.post_frames.append(newer)
        fb.capture_post()
        fb.frame_mode = "raise"
        return failed(inv, TIMEOUT_REASON)

    browser = FakeLiveBrowser([F0])
    browser.handlers["t1-call"] = capture_then_fail
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert image_urls(step_body(rig, 2)) == [builder_url(F0)], (
            "premise: the previous image stays"
        )
        assert browser.executes() == 1, (
            f"T-0111g B9: nothing dispatched for the stale frame; {browser.calls}"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["refused"]
        assert f"reason:{Q}{FRESHNESS}{Q}" in step_tail(rig, 3), step_tail(rig, 3)[-300:]
        assert card(rig, "t1-call-s2").reason == FRESHNESS


async def test_freshness_accepts_newest_with_shared_phash(tmp_path: Path) -> None:
    """B9, too-wide direction: the newest own frame is accepted even when it shares its phash with an older one."""
    after = live_frame("scrolled", phash=F0.phash)
    browser = FakeLiveBrowser([F0], post_frames=[after])
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert image_urls(step_body(rig, 2)) == [builder_url(after)]
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"], (
            f"T-0111g B9: the newest frame is accepted; {[c for c in browser.calls]}"
        )


# =====================================================================================================
# B10: results
# =====================================================================================================


async def test_failed_settles_failed_card_refused(tmp_path: Path) -> None:
    """B10 (mutation 5): an executor ``failed`` settles the loop result as ``failed`` with
    ``status_payload("failed", reason)``; the card stays ``refused`` with the executor's reason."""
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome("t1-call", lambda inv: failed(inv, TIMEOUT_REASON))
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S3)],))
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["failed"]
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "click_at", CLICK_ARGS, status("failed", TIMEOUT_REASON, True))]
        )
        c = card(rig, "t1-call")
        assert (c.status, c.reason) == ("refused", TIMEOUT_REASON), c
    recs = [r for r in rig.kind("tool.result") if r.get("call_id") == "t1-call"]
    assert [r.get("status") for r in recs] == ["failed"], recs


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda inv: refused(inv, ADOPTION), ADOPTION),
        (lambda inv: failed(inv, TIMEOUT_REASON), TIMEOUT_REASON),
        (lambda inv: failed(inv, RESTARTED), RESTARTED),
        (lambda inv: refused(inv, CLOSED), CLOSED),
        (None, "executor error"),
    ],
    ids=["adoption", "executor_timeout", "restarted", "closed", "executor_raises"],
)
async def test_adoption_and_timeout_reasons_verbatim(
    tmp_path: Path, make: Any, reason: str
) -> None:
    """B10: the executor's reasons reach the model (the step-2 response) and the card verbatim; an executor that
    raises still gives ``executor error``, as BASE does."""

    def boom(inv: Any) -> Any:
        raise RuntimeError("the fake executor exploded")

    browser = FakeLiveBrowser([F0, FR])
    browser.outcome("t1-call", make if make is not None else boom)
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S3)],))
        # the reason, verbatim, followed by the live payload's ``screenshot`` key (B8: every live step carries it)
        assert f"reason:{Q}{reason}{Q},screenshot:" in step_tail(rig, 2), step_tail(rig, 2)[-300:]
        assert card(rig, "t1-call").reason == reason, card(rig, "t1-call")


# =====================================================================================================
# B11: the window closed
# =====================================================================================================


async def test_window_closed_publishes_once_and_refuses(tmp_path: Path) -> None:
    """B11 (mutations 12, 19): once ``closed``: one HUD error saying computer use is off for this session, one
    session-log note, later turns carry no image (``frame_source: none``, ``browser: closed``), the ``tools`` field is
    unchanged, every live call is refused with A2's reason, and the voice session keeps answering."""
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S3)],))
        browser.close_window()
        await next_turn(
            rig,
            2,
            "click it again",
            [answer(S1), tool(SCROLL)],
            ([answer(S2), tool(CLICK2)], [answer(S3)]),
        )
        await next_turn(rig, 3, "are you still there", [answer("Still here. ")])
        errors = [
            m
            for m in rig.published.messages
            if getattr(m, "type", None) == "astra.error"
            and "computer use is off for this session" in m.message.lower()
        ]
        assert len(errors) == 1, f"T-0111g B11: published once; got {errors}"
        for turn in ("t2", "t3"):
            ex = turn_extras(rig, turn)
            assert (ex.get("frame_source"), ex.get("browser")) == ("none", "closed"), (turn, ex)
            assert image_urls(pass1_body(rig, turn)) == []
        assert (
            pass1_body(rig, "t2")["tools"]
            == pass1_body(rig, "t1")["tools"]
            == pass1_body(rig, "t3")["tools"]
        )
        for cid in ("t2-call", "t2-call-s2"):
            assert [e.status for e in tsl.results_for(rig, cid)] == ["refused"], cid
            assert card(rig, cid).reason == CLOSED, card(rig, cid)
        assert any("Still here" in (s or "") for _, s in rig.audio.plays), (
            "the voice session keeps running"
        )
    closed_notes = [n for n in rig.kind("note") if "closed" in str(n.get("what", "")).lower()]
    assert len(closed_notes) == 1, f"T-0111g B11: one session-log note; {closed_notes}"


# =====================================================================================================
# B12-B14: safety
# =====================================================================================================


@pytest.mark.parametrize("case", ["ask_first", "user_declines"])
async def test_ask_first_and_decline_never_execute(tmp_path: Path, case: str) -> None:
    """B12: T-0107a's and T-0107e's refusals apply unchanged in live; the executor is never called."""
    browser = FakeLiveBrowser([F0])
    script = [answer(ASK if case == "ask_first" else S1), tool(VALID_CLICK)]
    async with live_rig(tmp_path, browser, script=script) as rig:
        await rig.ask(tsl.QUESTION if case == "ask_first" else NO)
        await rig.turn_done()
        assert browser.executes() == 0, f"T-0111g B12: never executed; {browser.calls}"
        want = "asked the user first" if case == "ask_first" else "the user said no"
        assert card(rig, "t1-call").reason == want, card(rig, "t1-call")


async def test_barge_in_mid_execute_no_late_result(tmp_path: Path) -> None:
    """B13: a takeover while the fake blocks inside ``execute`` cancels it (``CancelledError`` seen), no result is
    submitted after, the next turn reaches IDLE and its onset frame is a fresh own frame."""
    fresh = live_frame("after-barge")
    browser = FakeLiveBrowser([F0, fresh])
    browser.hang("t1-call")
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.until(lambda: bool(browser.invocations), "premise: executing", limit_ms=6_000)
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        assert browser.cancelled == ["t1-call"], "T-0111g B13: the execute task got CancelledError"
        rig.llm.script = [answer("Okay. ")]
        await rig.words("stop that")
        await rig.end_of_turn()
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 2, "t2 never ended"
        )
        assert tsl.results_for(rig, "t1-call") == [], "no ToolResult after the cancel"
        cancel_at = browser.calls.index(("execute.cancelled", "t1-call"))
        assert frame_calls(browser) and frame_calls(browser)[-1] > cancel_at, (
            "the next onset frame is taken after the cancel"
        )
        assert image_urls(pass1_body(rig, "t2"))[-1] == builder_url(fresh)


async def test_assists_dropped_in_live(tmp_path: Path) -> None:
    """B14 (mutation 16): assist text describes the HUD's capture, so in live it is not attached; the flags are
    still frozen and logged, and the line says ``assists_dropped: true``. Positive control: the same setup in
    ``dry_run`` attaches it."""
    canary = "ASSIST-CANARY-T0111G"

    async def run(rig: Any) -> None:
        rig.session.apply_assist_toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
        rig.session.accept_frame_text(
            AstraFrameText(
                source=Source.SCREEN, ts_ms=rig.clock.now_ms, kind="page_text", text=canary
            )
        )
        await rig.ask("what does the page say")
        await rig.turn_done()

    async with tool_rig(tmp_path / "dry", policy=tsl.LOOP, script=[answer("It says hi. ")]) as dry:
        await run(dry)
        assert canary in json.dumps(dry.llm.bodies), "premise: dry_run attaches the assist text"
    async with live_rig(
        tmp_path / "live", FakeLiveBrowser([F0]), script=[answer("It says hi. ")]
    ) as rig:
        await run(rig)
        assert canary not in json.dumps(rig.llm.bodies), "T-0111g B14: no assist block in live"
        record = rig.record("t1")
        assert record.regime.assist_flags.page_text is True, (
            "the flags are frozen and logged as BASE logs them"
        )
        assert record.extras.get("assists_dropped") is True, record.extras


# =====================================================================================================
# B15 and B5: logging and lifecycle
# =====================================================================================================


async def test_egress_line_per_turn_unique(tmp_path: Path) -> None:
    """B15: ``browser_egress_blocked`` is the unique ``{host, scheme}`` pairs added during the turn, ``[]`` when
    none; exactly those two keys."""

    def blocked(fb: FakeLiveBrowser, inv: Any) -> Any:
        fb.block(("evil.example", "https"), ("evil.example", "https"), ("tracker.example", "http"))
        return fb.perform(inv)

    browser = FakeLiveBrowser([F0])
    browser.handlers["t1-call"] = blocked
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S3)],))
        await next_turn(rig, 2, "anything else", [answer("No. ")])
        got = turn_extras(rig, "t1").get("browser_egress_blocked")
        assert isinstance(got, list) and len(got) == 2, got
        assert {(r["host"], r["scheme"]) for r in got} == {
            ("evil.example", "https"),
            ("tracker.example", "http"),
        }
        assert all(set(r) == {"host", "scheme"} for r in got), got
        assert turn_extras(rig, "t2").get("browser_egress_blocked") == [], (
            "only this turn's additions"
        )


async def test_no_url_path_or_query_anywhere(tmp_path: Path) -> None:
    """B15 (adversarial, mutation 8): a step on ``/demo/contact.html?secret=Q7Z9`` whose egress entries carry the
    URL: neither ``Q7Z9`` nor ``/demo/contact`` appears in ``turns.jsonl``, the session log or any HUD message."""
    url = "http://127.0.0.1:41234/demo/contact.html?secret=Q7Z9"

    def blocked(fb: FakeLiveBrowser, inv: Any) -> Any:
        fb.block(
            ("cdn.example", "https"), extra={"url": url, "path": "/demo/contact.html?secret=Q7Z9"}
        )
        return fb.perform(inv)

    browser = FakeLiveBrowser([F0])
    browser.handlers["t1-call"] = blocked
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(S3)],))
        lines = rig.turns_path.read_text()
        hud = tsl.published_json(rig)
        assert turn_extras(rig).get("browser_egress_blocked") == [
            {"host": "cdn.example", "scheme": "https"}
        ], "premise: the blocked attempt was read, projected to {host, scheme}"
    logged = everything_logged(rig)
    for secret in ("Q7Z9", "/demo/contact"):
        for where, text in (("turns.jsonl", lines), ("the session log", logged), ("the HUD", hud)):
            assert secret not in text, f"T-0111g B15: {secret!r} reached {where}"


@pytest.mark.parametrize("close_fails", [False, True], ids=["clean", "close_raises"])
async def test_aclose_order_and_close_failure_logged(tmp_path: Path, close_fails: bool) -> None:
    """B5 (mutation 14): ``aclose`` cancels executions first, then closes the browser once; a close failure is one
    ``error`` record and ``aclose`` still completes (no exception escapes)."""
    browser = FakeLiveBrowser([F0])
    browser.hang("t1-call")
    if close_fails:
        browser.close_error = RuntimeError("close boom T0111G")
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.until(lambda: bool(browser.invocations), "premise: executing", limit_ms=6_000)
    closes = [i for i, c in enumerate(browser.calls) if c == ("close",)]
    assert len(closes) == 1, f"T-0111g B5: the browser is closed once; {browser.calls}"
    assert browser.calls.index(("execute.cancelled", "t1-call")) < closes[0], browser.calls
    errors = [
        r for r in rig.log_records if r.get("kind") == "error" and "close boom" in json.dumps(r)
    ]
    assert len(errors) == (1 if close_fails else 0), errors


@pytest.mark.parametrize("repair_off", [None, "payloads stale"])
async def test_tools_regime_label_live(tmp_path: Path, repair_off: str | None) -> None:
    """B5: the regime reads ``native:<sha12>:live[:repair_off]:loop<N>``."""
    async with live_rig(
        tmp_path, FakeLiveBrowser([F0]), script=[answer("Done. ")], repair_off=repair_off
    ) as rig:
        await rig.ask("hello")
        await rig.turn_done()
        want = f"native:{T.TOOLS_SHA12}:live" + (":repair_off" if repair_off else "") + ":loop8"
        assert rig.record("t1").regime.tools == want
    assert loop  # the module under test is imported at module level (tests.md)


# =====================================================================================================
# The group B review rows (Rulings, "on the group B reviews of 5294eda"): A1-A4
# =====================================================================================================

SCROLL_ACT = "scrolled down by 600 px"  # the fake's action for SCROLL


def last_user(body: dict[str, Any]) -> dict[str, Any]:
    """The final ``user`` message of a wire body (a step request ends with the assistant continuation)."""
    users = [m for m in body["messages"] if m.get("role") == "user"]
    assert users, body["messages"]
    return users[-1]


def part_kinds(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return ["text"]
    return [str(p.get("type")) for p in content]


async def test_blind_turn_step_says_unchanged_and_stays_blind(tmp_path: Path) -> None:
    """Review A1 (implementer fix; red at 6d2b3af) and the ruling on the blind turn: the onset ``frame()`` fails,
    so the turn carries no image. A scroll then executes (it needs no frame) and the browser captures a post-action
    frame, but the next step's request still carries no image, so that step's payload says ``screenshot:
    "unchanged"`` (written out by hand), never ``"updated"`` beside the no-image statement. The turn stays blind: a
    later coordinate call is refused with ``no frame``. The next turn takes a new onset frame."""
    # no onset frame queued: turn 1's onset raises, and turn 2's onset frame (F2) is queued only before turn 2,
    # so the frame turn 2 is handed is unambiguous whether or not the blind turn takes a post-action capture
    browser = FakeLiveBrowser([], post_frames=[F1])
    browser.frame_mode = "raise"
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert image_urls(pass1_body(rig)) == [], "premise: the onset frame failed, step 1 is blind"
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"], (
            "premise: the scroll executed"
        )
        s2 = step_body(rig, 2)
        assert image_urls(s2) == [], (
            f"the ruling on the blind turn: no image mid-turn either; got {len(image_urls(s2))}"
        )
        assert NO_IMAGE_STATEMENT in json.dumps(s2), "the no-image wording still applies"
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "scroll", {"direction": "down"}, executed(SCROLL_ACT, updated=False))]
        ), (
            "T-0111g review A1: the step's prompt did not change, so its payload says screenshot: unchanged; "
            f"got ...{step_tail(rig, 2)[-160:]}"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["refused"], (
            "the blind turn stays blind: the coordinate call has no frame"
        )
        assert card(rig, "t1-call-s2").reason == "no frame", card(rig, "t1-call-s2")
        browser.frame_mode = "ok"
        browser.frames.append(F2)
        calls_before, returned_before = len(frame_calls(browser)), len(browser.frames_returned)
        await next_turn(rig, 2, "and now", [answer("Now I see it. ")])
        onset = browser.frames_returned[returned_before:]
        assert len(frame_calls(browser)) == calls_before + 1 and len(onset) == 1, (
            f"the next turn takes one new onset capture; frame() calls {browser.calls[-4:]}, returned {onset}"
        )
        assert onset[0].jpeg == F2.jpeg, "premise: the fake handed turn 2 the frame queued for it"
        assert image_urls(pass1_body(rig, "t2")) == [builder_url(onset[0])], (
            "the next turn carries exactly one image: its own fresh onset capture"
        )


@pytest.mark.parametrize("step1", ["executed", "failed"])
async def test_blind_turn_takes_no_frame_after_failed_onset(tmp_path: Path, step1: str) -> None:
    """Review A1 (blind guard; kills A1m2): a turn whose onset ``frame()`` failed stays blind, so no ``frame()`` call
    happens for the rest of the turn. Whether step 1 executes (a scroll) or fails (``browser restarted``, after which
    a sighted turn would take a fresh frame), the turn makes exactly the one failed onset call, step 2 carries no
    image, and step 1's payload says ``screenshot: "unchanged"`` (written out by hand)."""
    browser = FakeLiveBrowser([], post_frames=[F1])
    browser.frame_mode = "raise"
    if step1 == "failed":
        browser.outcome("t1-call", lambda inv: failed(inv, RESTARTED))
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await three_steps(rig, ([answer(S3)],))
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == [step1], "premise"
        assert len(frame_calls(browser)) == 1, (
            f"T-0111g review A1: a blind turn takes no frame after its failed onset; calls {browser.calls}"
        )
        assert image_urls(step_body(rig, 2)) == [], "step 2 is blind too"
        payload = (
            executed(SCROLL_ACT, updated=False)
            if step1 == "executed"
            else status("failed", RESTARTED, False)
        )
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "scroll", {"direction": "down"}, payload)]
        ), step_tail(rig, 2)[-200:]


async def test_post_action_frame_with_same_bytes_says_unchanged(tmp_path: Path) -> None:
    """Review A1 (prompt-diff layer; kills A1m, A1m3): an executed scroll at the bottom of a page captures a
    post-action frame with the same JPEG bytes as the image already shown. Step 2's prompt is then byte for byte
    step 1's, so step 1's payload says ``screenshot: "unchanged"`` (by hand), even though a new frame object was
    taken. Step 2's click, read off that same image, still executes."""
    at_bottom = F0.model_copy(update={"ts_ms": 55_555})
    browser = FakeLiveBrowser([F0], post_frames=[at_bottom])
    async with live_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await three_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        p1, s2 = pass1_body(rig), step_body(rig, 2)
        assert image_urls(s2) == [builder_url(F0)] == image_urls(p1), "premise: the same image"
        assert stripped(s2)[:-1] == stripped(p1), "every message but the continuation is step 1's"
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "scroll", {"direction": "down"}, executed(SCROLL_ACT, updated=False))]
        ), (
            "T-0111g review A1: the step's prompt did not change, so it says unchanged; got "
            f"...{step_tail(rig, 2)[-160:]}"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"], (
            f"step 2's click executes; {browser.calls}"
        )


async def test_ride_on_post_action_image_first_in_current_message(tmp_path: Path) -> None:
    """Review A2 (kills X1, X2): turn 2's onset frame has turn 1's bytes, so it rides on the history image and turn
    2's own user message carries none. After turn 2's click, step 2's final user message carries the post-action
    frame as its **first** part, before its text (where the prompt builder puts a current frame), the history
    image is untouched, every other message is unchanged, and step 2's click resolves against that frame and
    executes."""
    again = F0.model_copy(update={"ts_ms": 77_777})
    browser = FakeLiveBrowser([F0, again], post_frames=[F1])
    async with live_rig(tmp_path, browser, script=[answer("A shop. ")]) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        await next_turn(
            rig,
            2,
            tsl.QUESTION,
            [answer(S1), tool(VALID_CLICK)],
            ([answer(S2), tool(CLICK2)], [answer(S3)]),
        )
        p1 = pass1_body(rig, "t2")
        assert image_urls(p1) == [builder_url(F0)], "premise: one image, the history one"
        assert "image_url" not in part_kinds(last_user(p1)), (
            "premise: turn 2's own message rides on the history image"
        )
        s2 = step_body(rig, 2, "t2")
        current = last_user(s2)
        assert part_kinds(current)[0] == "image_url" and "text" in part_kinds(current)[1:], (
            f"T-0111g review A2: the post-action image comes first, then the text; parts {part_kinds(current)}"
        )
        assert current["content"][0]["image_url"]["url"] == builder_url(F1), (
            "the first part is the post-action frame"
        )
        assert image_urls(s2) == [builder_url(F0), builder_url(F1)], (
            "the history image is untouched and the post-action frame is the only new image"
        )
        p1_users = [m for m in p1["messages"] if m.get("role") == "user"]
        s2_messages = s2["messages"][: len(p1["messages"])]
        assert s2_messages[:-1] == p1["messages"][:-1], "every earlier message is unchanged"
        assert [p for p in current["content"][1:]] == p1_users[-1]["content"], (
            "the current message's own parts are unchanged after the new image"
        )
        assert inv_of(browser, "t2-call-s2").frame.phash == F1.phash, (
            "step 2's click resolves against F1"
        )
        assert [e.status for e in tsl.results_for(rig, "t2-call-s2")] == ["executed"]
        assert step_tail(rig, 2, "t2") == want_tail(
            rig, [(S1, "click_at", CLICK_ARGS, executed(ACT1))], "t2"
        )


def _prompt_urls(prompt: Any) -> list[str]:
    from local_astra.contracts.llm import ImagePart

    return [p.url for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]


@pytest.mark.parametrize("which", ["first_turn", "second_turn"])
async def test_capture_wait_step_one_carries_onset_frame(tmp_path: Path, which: str) -> None:
    """Review A3 (kills X5, X6): the onset ``frame()`` blocks on an event that is released only once the machine has
    emitted ``LlmStart``. No step-1 request goes out before the frame lands, and step 1's image is F0. The prefill
    waits too: no prefill ``llm.request`` goes out while the frame is still being taken, and any prefill of the turn
    carries F0 as its newest image.

    ``second_turn`` runs an ungated turn first. On a first turn a prefill that skipped the wait would find no prefix
    yet and send nothing, so the prefill wait (X6) is observable only once an earlier turn left a stale prefix."""
    prior = live_frame("prior-turn")
    browser = FakeLiveBrowser([prior, F0] if which == "second_turn" else [F0])
    turn = "t1"
    async with live_rig(tmp_path, browser, script=[answer("It is a shop. ")]) as rig:
        if which == "second_turn":
            await rig.ask("what is on the screen")
            await rig.turn_done()
            await rig.advance(300)
            rig.llm.script = [answer("Still a shop. ")]
            turn = "t2"
        gate = asyncio.Event()
        browser.frame_gate = gate
        prefills_before = len(rig.llm.prefills)
        starts_before = len(rig.commands(C.LlmStart))
        await rig.onset()
        assert rig.state.turn_id == turn, f"premise: turn {turn} opened; {rig.state.turn_id}"
        await rig.words("what is shown")
        await rig.end_of_turn()
        await rig.until(
            lambda: len(rig.commands(C.LlmStart)) > starts_before,
            "premise: LlmStart",
            limit_ms=1_500,
        )
        assert browser.frame_active == 1, (
            f"premise: the onset frame is still being taken; {browser.calls}"
        )
        assert not tsl.generates_for(rig, f"{turn}-gen"), (
            "T-0111g review A3: the step-1 request waits for the turn's live frame"
        )
        assert rig.llm.prefills[prefills_before:] == [], (
            "T-0111g review A3: the prefill waits for the turn's live frame; sent before it landed: "
            f"{[(rid, len(_prompt_urls(p))) for rid, p in rig.llm.prefills[prefills_before:]]}"
        )
        gate.set()
        await rig.turn_done()
        want = (
            [builder_url(prior), builder_url(F0)] if which == "second_turn" else [builder_url(F0)]
        )
        assert image_urls(pass1_body(rig, turn)) == want, (
            "T-0111g review A3: step 1 carries F0 (after the history image on a second turn); got "
            f"{len(image_urls(pass1_body(rig, turn)))} image(s)"
        )
        late = [_prompt_urls(p)[-1:] for _, p in rig.llm.prefills[prefills_before:]]
        assert all(urls == [builder_url(F0)] for urls in late), (
            f"every prefill of the turn carries F0 as its newest image; {late}"
        )


async def test_frame_after_turn_end_never_joins_next_turn(tmp_path: Path) -> None:
    """Review A4 (kills X4): turn 1's onset ``frame()`` is still blocked when turn 1 ends (the transcriber fails
    while it listens), and it lands only after turn 2 opened. That late frame never joins turn 2: turn 2 carries
    only its own onset frame."""
    late = live_frame("late")
    browser = FakeLiveBrowser([late, F0])
    gate = asyncio.Event()
    browser.frame_gate = gate
    async with live_rig(tmp_path, browser, script=[answer("A shop. ")]) as rig:
        await rig.onset()
        rig.submit(E.SttError(t_ms=rig.clock.now_ms, message="transcriber hiccup"))
        await rig.pump()
        assert rig.state.state is TurnState.IDLE, f"premise: turn 1 ended; {rig.state.state}"
        await rig.advance(100)
        await rig.onset()
        assert rig.state.turn_id != "t1", f"premise: turn 2 opened; {rig.state.turn_id}"
        turn2 = rig.state.turn_id
        gate.set()
        await rig.pump()
        await rig.words("what is shown")
        await rig.end_of_turn()
        await rig.turn_done()
        assert len(frame_calls(browser)) == 2, f"premise: two captures; {browser.calls}"
        urls = image_urls(pass1_body(rig, turn2))
        assert builder_url(late) not in urls, (
            "T-0111g review A4: a frame that landed after its turn ended joined the next turn"
        )
        assert urls == [builder_url(F0)], (
            f"turn 2 carries its own onset frame; {len(urls)} image(s)"
        )


async def test_frame_calls_are_serialised(tmp_path: Path) -> None:
    """Review A4 (kills X9): an onset snapshot and a scene-change snapshot of one turn never call ``frame()`` at the
    same time; both captures still land, in order."""
    a, b = live_frame("first"), live_frame("second")
    browser = FakeLiveBrowser([a, b])
    gate = asyncio.Event()
    browser.frame_gate = gate
    async with live_rig(tmp_path, browser, script=[answer("Noted. ")]) as rig:
        await rig.onset()
        rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash="scene-2"))
        await rig.pump()
        await rig.advance(200)
        assert browser.frame_max_active == 1, (
            f"T-0111g review A4: frame() calls overlapped ({browser.frame_max_active} at once)"
        )
        gate.set()
        await rig.pump()
        await rig.words("compare these")
        await rig.end_of_turn()
        await rig.turn_done()
        assert browser.frame_max_active == 1, browser.frame_max_active
        assert len(frame_calls(browser)) == 2, f"premise: two snapshots; {browser.calls}"
        urls = image_urls(pass1_body(rig))
        assert urls == [builder_url(a), builder_url(b)], f"both captures, in order; {len(urls)}"
