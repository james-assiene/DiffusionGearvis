"""T-0150 group A (session side): A1-A5, A7, A8 over the fakes (docs/tasks/T-0150.md).

**The surface these rows pin** (the task's names; where it is silent the reading is stated here and in the report):

- ``Session(..., auto_overview=True)`` (keyword; absent means off). With it on, in live, each turn whose step-1
  request is built decides once: refused before capture (A1.2: no ``ImagePart`` in the final user message; or the
  prompt plus one overview over ``prompts.budget.max_images``), else one ``LiveBrowser.execute(ToolInvocation(
  name="page_overview", arguments={}, frame=None, ...))``, bounded by ``session.AUTO_OVERVIEW_TIMEOUT_MS`` (2_000,
  A8) on the session clock.
- A2: an executed outcome with ``resolved["overview"]`` whose ``overview_of`` gives a ``Frame`` is placed by
  ``loop.place_overview`` in the step-1 request (and its over-long retry, and the step-1 repair), with T-0140's
  ``OVERVIEW_LABEL``; nowhere else.
- A5: the turn line's ``auto_overview == {"shown", "fits", "failed", "refused"}`` (ints) on every live turn line when
  on; ``llm.request``'s ``overview == {"phash", "screens", "clipped", "width", "height"}`` on every request carrying
  the picture; one ``note`` ``{"what": "auto overview timed out", "timeout_ms"}`` on a timeout.

Readings (stated, not guessed silently; see the report):

- **R1 (A3 "fits").** A fits page *is* captured (A1.3 has no fits exception; the executor decides fits), so it
  counts one capture and ``fits`` 1, and every request body equals the off turn's.
- **R2 (A5 "sum to 0").** A turn whose step-1 request is never built counts all four at 0. The rows produce it with a
  takeover while the generate still waits for the turn's live frame (``LlmStart`` was issued, step 1 never built):
  the criterion's own example ("a barge-in before ``LlmStart``") leaves no turn line on this machine, so the row's
  case is the nearest reachable one.
- **R3 (A1.5, Q-5).** A live snapshot *can* be requested while step 1 is being built: a ``UserTakeover`` in THINKING
  (no step executed yet) aborts the generate and issues ``SnapshotFrame`` for the new turn (turn_machine
  ``_on_takeover``). ``test_no_overlap_with_frame`` drives that machine path (with a capture that, like a real one,
  takes event-loop turns to unwind after a cancel) and, as a second case, the session's snapshot seam
  (``Session.snapshot_frame``) directly while the capture is held.
- **R4 (A1.3 exceptions).** An executor exception is a case the criterion does not classify (``failed`` names the
  executor's ``failed``, the timeout and an image-less ``executed``). The row pins only what A5 states: exactly one
  count, no overview, and the exception's text (a URL) in no log record.
- **R5 (Q-3).** With two own frames the step-1 final user message is ``[f1, "[screen 2]", overview, OVERVIEW_LABEL,
  f2, screen label, ...]``: ``place_overview`` inserts before the newest image.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on, ReGround off unless a row says so, T-0134c's label on unless a row says so, deals-like
page 1220/800 CSS px (1.525 screens, docs/tasks/T-0150.md "Measured scope"); n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import Frame, Source
from local_astra.toolcall import native, tools
from tests.fakes.llm import http_error, is_caption_prompt
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    MAX_OVERVIEW_SCREENS,
    FakeLiveBrowser,
    b64,
    builder_url,
    feedback_rig,
    image_urls,
    jpeg_frame,
    live_feedback,
    live_frame,
    live_kit,
    session_mod,
)
from tests.unit._t0177_pins import published_json_outside_frames  # T-0177 B-P2
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    tool,
)
from tests.unit.test_session_live import (
    CLICK2,
    _wire,
    last_user,
    pass1_body,
    step_body,
)
from tests.unit.test_t0134_label import L1, all_bodies, next_turn

S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
#: T-0140's OVERVIEW_LABEL at 1.525 screens, not clipped, written out by hand from the criterion's template
LABEL_15 = (
    "[Whole page, scaled to fit one screen: about 1.5 screens tall. "
    "The next image is the browser screen.]"
)
#: the label's stem and closing sentence (substrings of no other rig text)
OV_STEM = "Whole page, scaled to fit one screen"
OV_CLOSE = "The next image is the browser screen"
#: the fake executor's action string at 1.525 screens (A3's template, by hand)
OV_ACTION = "showed the whole page (about 1.5 screens tall) as a picture before the screen"
TIMEOUT_NOTE = "auto overview timed out"
A5_KEYS = ("shown", "fits", "failed", "refused")
REQUEST_KEYS = {"phash", "screens", "clipped", "width", "height"}
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"

#: the scaled overview the executor keeps (sentinel bytes: the session only carries it)
OV = Frame(
    source=Source.WINDOW,
    ts_ms=0,
    width=320,
    height=800,
    jpeg=b"\xff\xd8" + b"AUTO-OVERVIEW-PIXELS-T0150|" * 8,
    phash="5150515051505150",
)
#: the session's capture identity of the overview's bytes, by hand (T-0111g B7's rule)
OV_IDENTITY = "capture:" + hashlib.sha256(OV.jpeg).hexdigest()[:32]
#: deals.html at 1280x800 (the planner's measurement): 1220/800 = 1.525 screens
DEALS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1220.0}
#: the other four start pages: 800/800 = 1.0 screens, fits
FITS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 800.0}

F0 = live_frame("onset")
F1 = live_frame("after-1")
F2 = live_frame("after-2")


# --------------------------------------------------------------------------------------------- seams


def counts(**kw: int) -> dict[str, int]:
    return {k: int(kw.get(k, 0)) for k in A5_KEYS}


def timeout_ms() -> int:
    value = getattr(session_mod(), "AUTO_OVERVIEW_TIMEOUT_MS", None)
    assert value == 2_000 and type(value) is int, (
        f"T-0150 A8: session.AUTO_OVERVIEW_TIMEOUT_MS = 2_000 does not exist yet (got {value!r})"
    )
    return value


def label_on() -> Any:
    """T-0134c's live default: the screen label, no extent."""
    return live_feedback(screen_label=True, page_extent=False)


def label_off() -> Any:
    """A2's ``LiveFeedback(False, False)``."""
    return live_feedback(screen_label=False, page_extent=False)


def deals_browser(
    frames: Any = (F0,), post: Any = (F1, F2), metrics: dict[str, float] | None = None
) -> FakeLiveBrowser:
    browser = FakeLiveBrowser(list(frames), post_frames=list(post), echo="model")
    browser.overview_image = OV
    browser.page_metrics = dict(metrics if metrics is not None else DEALS)
    return browser


def rig_for(
    tmp: Path,
    browser: Any,
    *,
    on: bool,
    script: Any,
    feedback: Any = "on",
    extra: dict[str, Any] | None = None,
    reground_llm: bool = False,
    kit: Any = None,
) -> Any:
    """``feedback_rig`` with ``Session(..., live_feedback=..., auto_overview=True)`` when ``on`` (off: the keyword is
    not passed at all, A6 "Wiring"). A session without the keyword is a named ``AssertionError``."""
    kw: dict[str, Any] = {}
    fb = label_on() if feedback == "on" else feedback
    if fb is not None:
        kw["live_feedback"] = fb
    if on:
        kw["auto_overview"] = True
    kw.update(extra or {})
    return feedback_rig(
        tmp, browser, script=script, session_kwargs=kw, reground_llm=reground_llm, kit=kit
    )


async def run_turn(rig: Any, steps: tuple[Any, ...], question: str = tsl.QUESTION) -> None:
    if steps:
        rig.llm.script_steps(*steps)
    await rig.ask(question)
    await tsl.until_steps(rig, len(steps), limit_ms=20_000, step=100)
    await rig.turn_done(limit_ms=20_000)
    await rig.advance(300)


def ov_url() -> str:
    return builder_url(OV)


def ov_in(body: dict[str, Any]) -> int:
    """How many times the overview's pixels appear in ``body`` (as a data URL or raw base64)."""
    return json.dumps(body).count(b64(OV.jpeg))


def kinds(parts: list[dict[str, Any]]) -> list[str]:
    out = []
    for p in parts:
        if p.get("type") == "image_url":
            out.append("image:OV" if p["image_url"]["url"] == ov_url() else "image")
        else:
            out.append("text:" + str(p.get("text", ""))[:48])
    return out


def ov_part() -> dict[str, Any]:
    """The overview's part on the wire, by T-0111g's oracle (``builder_url``) and the body shape of the frame's part."""
    return {"type": "image_url", "image_url": {"url": ov_url()}}


def label_part() -> dict[str, Any]:
    return {"type": "text", "text": LABEL_15}


def same_shape(part: dict[str, Any], like: dict[str, Any]) -> dict[str, Any]:
    """``like``'s image part with the overview's URL (so extra wire keys of the client's image parts are kept)."""
    out = json.loads(json.dumps(like))
    out["image_url"]["url"] = ov_url()
    return out


def requests(rig: Any) -> list[dict[str, Any]]:
    return rig.kind("llm.request")


def ov_records(rig: Any) -> list[dict[str, Any]]:
    return [d for d in requests(rig) if "overview" in d]


def captures(browser: FakeLiveBrowser) -> list[Any]:
    return [i for i in browser.invocations if i.name == "page_overview"]


def auto_key(rig: Any, turn: str = "t1") -> Any:
    return dict(rig.record(turn).extras).get("auto_overview")


async def off_and_on(
    tmp: Path,
    make_browser: Callable[[], FakeLiveBrowser],
    *,
    script: Any,
    steps: tuple[Any, ...] = (),
    feedback: Any = "on",
    drive: Any = None,
) -> dict[str, dict[str, Any]]:
    """The same scripted turn with ``auto_overview`` off and on: bodies, records and turn lines of each."""
    out: dict[str, dict[str, Any]] = {}
    for arm in ("off", "on"):
        browser = make_browser()
        async with rig_for(
            tmp / arm, browser, on=arm == "on", script=list(script), feedback=feedback
        ) as rig:
            if drive is None:
                await run_turn(rig, steps)
            else:
                await drive(rig, steps)
            got: dict[str, Any] = {
                "rig": rig,
                "browser": browser,
                "bodies": all_bodies(rig),
                "pass1": pass1_body(rig),
                "steps": [_wire(p) for _, p in tsl.steps_of(rig)],
                "lines": [json.loads(x) for x in rig.lines()],
            }
        got["requests"] = requests(rig)
        out[arm] = got
    return out


# =====================================================================================================
# A1/A2: the step-1 placement
# =====================================================================================================


async def test_step1_overview_before_current_frame(tmp_path: Path) -> None:
    """A1/A2 (reviewer mutation 1, the overview after the current frame): on a 1.525-screen page the step-1 final user
    message is the off turn's with ``[overview ImagePart, OVERVIEW_LABEL(1.5)]`` inserted right before the current
    frame: ``[OV, LABEL_15, F0, L1, ...]``; exactly two images, the current frame newest; the step-1 click resolves
    against F0 and executes (``_sent_identity``/``frame_identity``/B9 key on it); the generate ``llm.request`` has
    ``overview`` with the five keys (values by hand); exactly one ``page_overview`` invocation reached the browser,
    with ``arguments == {}`` and no frame."""
    runs = await off_and_on(
        tmp_path,
        deals_browser,
        script=[answer(S1), tool(VALID_CLICK)],
        steps=([answer(S3)],),
    )
    off, on = runs["off"], runs["on"]
    own_off = last_user(off["pass1"])["content"]
    assert own_off[0]["image_url"]["url"] == builder_url(F0) and own_off[1] == {
        "type": "text",
        "text": L1,
    }, f"premise: the off turn shows F0 then L1; {kinds(own_off)}"
    parts = last_user(on["pass1"])["content"]
    want = [same_shape(ov_part(), own_off[0]), label_part(), *own_off]
    assert parts == want, f"T-0150 A2: the step-1 final user message; got {kinds(parts)}"
    assert image_urls(on["pass1"]) == [ov_url(), builder_url(F0)], (
        "A2: one overview image, before the current frame, which stays the newest"
    )
    browser = on["browser"]
    clicks = [i for i in browser.invocations if i.call_id == "t1-call"]
    assert len(clicks) == 1 and clicks[0].frame is not None and clicks[0].frame.phash == F0.phash, (
        "A2: the step-1 click resolves against the current frame, never the overview"
    )
    rig = on["rig"]
    assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"], (
        "A2: the click is executed (not refused as stale): the current frame keys B9"
    )
    caps = captures(browser)
    assert len(caps) == 1, f"T-0150 A1: exactly one capture; got {len(caps)}"
    assert caps[0].arguments == {} and caps[0].frame is None, caps[0]
    gen = [d for d in on["requests"] if d.get("phase") == "generate"]
    assert len(gen) == 1 and gen[0].get("overview") == {
        "phash": OV.phash,
        "screens": pytest.approx(1.525),
        "clipped": False,
        "width": 320,
        "height": 800,
    }, gen
    assert set(gen[0]["overview"]) == REQUEST_KEYS


async def test_overview_step1_only(tmp_path: Path) -> None:
    """A2 / H-T150-3 (reviewer mutation 3's step half): after an executed step-1 click, the step-2 and step-3 bodies
    carry no overview and no label; no prefill and no caption request carries it; the next turn's request (history)
    does not; exactly one ``llm.request`` record has the ``overview`` key, the step-1 generate."""
    browser = deals_browser(frames=(F0, live_frame("t2-onset")))
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert ov_in(pass1_body(rig)) == 1, "premise: step 1 carries the overview"
        for k in (2, 3):
            body = step_body(rig, k)
            assert ov_in(body) == 0 and OV_STEM not in json.dumps(body), (
                f"T-0150 A2: step {k} carries the overview; {kinds(last_user(body)['content'])}"
            )
        browser.page_metrics = dict(
            FITS
        )  # turn 2's page fits, so any overview in its request is turn 1's
        await next_turn(rig, 2, "and now", [answer("Still the deals. ")])
        t2 = pass1_body(rig, "t2")
        assert OV_STEM not in json.dumps(t2) and OV.phash not in json.dumps(t2), (
            "A2: never in history"
        )
        assert rig.llm.prefills, "premise: prefills were sent"
        for rid, p in rig.llm.prefills:
            assert ov_in(_wire(p)) == 0, f"A2: the prefill {rid} carries the overview"
        for rid, p, _ in rig.llm.generates:
            if is_caption_prompt(p):
                assert ov_in(_wire(p)) == 0, f"A2: the caption request {rid} carries the overview"
    recs = ov_records(rig)
    assert [(d.get("phase"), d.get("request_id")) for d in recs] == [("generate", "t1-gen")], (
        f"A5: exactly the step-1 request records an overview; {[(d.get('phase'), d.get('step')) for d in recs]}"
    )


async def test_no_capture_after_navigation(tmp_path: Path) -> None:
    """H-T150-3 (reviewer mutation 3, a capture at every step or after navigation): the turn starts on a page that
    fits (one capture, ``fits``); the step-1 click navigates to a 1.525-screen page; steps 2 and 3 take no capture and
    carry no overview. Exactly one capture in the turn."""
    browser = deals_browser(metrics=FITS)
    browser.url_metrics["/demo/deals.html"] = dict(DEALS)

    def navigates(fb: FakeLiveBrowser, inv: Any) -> Any:
        fb.navigate("/demo/deals.html")
        return fb.perform(inv)

    browser.handlers["t1-call"] = navigates
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert browser.url == "/demo/deals.html", "premise: the click navigated"
        assert len(captures(browser)) == 1, (
            f"T-0150 H-T150-3: one capture per turn, at step 1; got {len(captures(browser))}"
        )
        for k in (2, 3):
            assert ov_in(step_body(rig, k)) == 0, f"step {k} carries an overview"
        assert auto_key(rig) == counts(fits=1), auto_key(rig)


@pytest.mark.parametrize("reground", [False, True])
async def test_loop_base_clean_with_label_off(tmp_path: Path, reground: bool) -> None:
    """A2 (adversarial; reviewer mutation 2, ``loop_base = prompt`` when no label, and mutation 11, the overview in the
    ReGround pass-2 body): with ``LiveFeedback(False, False)`` step 1 carries the overview (premise) and step 2
    carries none. With ReGround on, the pass-2 body of the step-1 click carries no overview image and no label (the
    crop replaces the current frame in a prompt that never held the overview)."""
    on0, on1, on2 = jpeg_frame("onset"), jpeg_frame("after-1"), jpeg_frame("after-2")
    browser = deals_browser(frames=(on0,), post=(on1, on2))
    extra = {"reground": True} if reground else {}
    async with rig_for(
        tmp_path,
        browser,
        on=True,
        script=[answer(S1), tool(VALID_CLICK)],
        feedback=label_off(),
        extra=extra,
        reground_llm=reground,
    ) as rig:
        llm = rig.llm
        if reground:
            base_is_pass2 = llm.is_pass2

            def is_pass2(self: Any, prompt: Any) -> bool:
                # the overview is an image the browser never returned too: only the crop marks a pass-2 request
                stripped = prompt.model_copy(
                    update={
                        "messages": tuple(
                            m.model_copy(
                                update={
                                    "content": tuple(
                                        p
                                        for p in m.content
                                        if not (isinstance(p, ImagePart) and p.url == ov_url())
                                    )
                                }
                            )
                            for m in prompt.messages
                        )
                    }
                )
                return bool(base_is_pass2(stripped))

            llm.is_pass2 = types.MethodType(is_pass2, llm)
            llm.script_pass2(
                [answer(""), tool("<|tool_call>call:click_at{x:510,y:260}<tool_call|>")]
            )
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        p1 = pass1_body(rig)
        assert ov_in(p1) == 1 and LABEL_15 in json.dumps(p1), "premise: step 1 carries the overview"
        assert "This screenshot is the browser screen now" not in json.dumps(p1), (
            "premise: label off"
        )
        s2 = step_body(rig, 2)
        assert ov_in(s2) == 0 and OV_STEM not in json.dumps(s2), (
            f"T-0150 A2: with the label off step 2 carries the overview; {kinds(last_user(s2)['content'])}"
        )
        if reground:
            assert llm.pass2_prompts, "premise: pass 2 ran for the step-1 click"
            for rid, p in llm.pass2_prompts:
                body = _wire(p)
                assert ov_in(body) == 0 and OV_STEM not in json.dumps(body), (
                    f"T-0150 A2: the pass-2 request {rid} carries the overview"
                )


async def test_reground_pass2_clean_label_on(tmp_path: Path) -> None:
    """A2 (reviewer mutation 11 with the label on): ReGround's pass 2 for the step-1 click carries no overview image
    and no overview label; step 1 does (premise)."""
    on0, on1 = jpeg_frame("onset"), jpeg_frame("after-1")
    browser = deals_browser(frames=(on0,), post=(on1,))
    async with rig_for(
        tmp_path,
        browser,
        on=True,
        script=[answer(S1), tool(VALID_CLICK)],
        extra={"reground": True},
        reground_llm=True,
    ) as rig:
        llm = rig.llm
        base_is_pass2 = llm.is_pass2

        def is_pass2(self: Any, prompt: Any) -> bool:
            urls = {p.url for m in prompt.messages for p in m.content if isinstance(p, ImagePart)}
            if ov_url() in urls:
                return False  # a request carrying the overview is never the crop request (the row's reading)
            return bool(base_is_pass2(prompt))

        llm.is_pass2 = types.MethodType(is_pass2, llm)
        llm.script_pass2([answer(""), tool("<|tool_call>call:click_at{x:510,y:260}<tool_call|>")])
        await run_turn(rig, ([answer(S3)],))
        assert ov_in(pass1_body(rig)) == 1, "premise: step 1 carries the overview"
        assert llm.pass2_prompts, "premise: pass 2 ran"
        # a mutant that puts the overview in pass 2 makes the fake serve it as a generate: then the step-1 click has
        # no pass-2 request at all, or one that carries the picture
        for rid, p in llm.pass2_prompts:
            body = _wire(p)
            assert ov_in(body) == 0 and OV_STEM not in json.dumps(body), rid
    regs = [d for d in requests(rig) if d.get("phase") == "reground"]
    assert regs and all("overview" not in d for d in regs), regs
    assert all(OV_STEM not in json.dumps(d) for d in regs), (
        "A2: the pass-2 record carries the label"
    )


async def test_repair_of_step1_carries_overview(tmp_path: Path) -> None:
    """A2/A5: a repairable step-1 call's ``tool_repair`` request re-sends pass 1 as sent, so its body holds the same
    overview image and label once, before the current frame, and its record has the ``overview`` key equal to the
    generate's; ``shown`` is 1 (the repair is not counted again); step 2 has none."""
    browser = deals_browser()
    async with rig_for(
        tmp_path, browser, on=True, script=[answer(S1), tool(REPAIRABLE_CLICK)]
    ) as rig:
        rig.llm.script_continuation(tool(REPAIRED_CLICK))
        await run_turn(rig, ([answer(S3)],))
        reps = rig.llm.continuations()
        assert reps, "premise: a repair request was sent"
        body = _wire(reps[0][1])
        assert ov_in(body) == 1 and LABEL_15 in json.dumps(body), (
            "T-0150 A2: the step-1 repair re-sends the overview"
        )
        urls = image_urls(body)
        assert urls == [ov_url(), builder_url(F0)], f"A2: overview, then the current frame; {urls}"
        assert ov_in(step_body(rig, 2)) == 0, "A2: step 2 carries none"
        assert len(captures(browser)) == 1
        assert auto_key(rig) == counts(shown=1), auto_key(rig)
    gen = [d for d in requests(rig) if d.get("phase") == "generate"]
    rep = [d for d in requests(rig) if d.get("phase") == "tool_repair"]
    assert gen and rep, [d.get("phase") for d in requests(rig)]
    assert "overview" in gen[-1] and rep[-1].get("overview") == gen[-1]["overview"], (
        f"T-0150 A5: the repair record carries the overview key; got {rep[-1].get('overview')}"
    )


async def test_once_per_turn_on_retry(tmp_path: Path) -> None:
    """A1.4/A2/A5 (adversarial; reviewer mutation 4, a second capture on the over-long retry): a 413 on the first
    step-1 attempt makes the retry's body carry the same overview (same image, same label, before the current frame)
    with exactly one capture and ``shown`` 1; both attempts' ``llm.request`` records carry the same ``overview``
    key."""
    browser = deals_browser()
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        rig.llm.fail_generates(http_error(413), None)
        await run_turn(rig, ([answer(S3)],))
        attempts = tsl.generates_for(rig, "t1-gen")
        assert len(attempts) == 2, f"premise: one over-long retry; {len(attempts)} attempts"
        for i, (_, p) in enumerate(attempts):
            body = _wire(p)
            assert image_urls(body) == [ov_url(), builder_url(F0)], (
                i,
                kinds(last_user(body)["content"]),
            )
            assert LABEL_15 in json.dumps(body), i
        assert len(captures(browser)) == 1, (
            f"T-0150 A1.4: the retry re-uses the turn's result; {len(captures(browser))} captures"
        )
        assert auto_key(rig) == counts(shown=1), auto_key(rig)
    gen = [d for d in requests(rig) if d.get("phase") == "generate"]
    assert len(gen) == 2 and all("overview" in d for d in gen), gen
    assert gen[0]["overview"] == gen[1]["overview"]


async def test_capture_after_frame_landed(tmp_path: Path) -> None:
    """A1.1: while the turn's onset frame is still being taken (the fake's ``frame_gate``), no capture is made; once it
    lands, one capture, and step 1 carries F0 and the overview."""
    browser = deals_browser()
    browser.frame_gate = asyncio.Event()
    async with rig_for(tmp_path, browser, on=True, script=[answer(S3)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.advance(600)
        assert not browser.frames_returned, "premise: the onset frame is still pending"
        assert captures(browser) == [], "T-0150 A1.1: a capture before the turn's live frame landed"
        browser.frame_gate.set()
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        assert len(captures(browser)) == 1
        assert image_urls(pass1_body(rig)) == [ov_url(), builder_url(F0)]
        frame_at = browser.calls.index(("frame",))
        exec_at = next(i for i, c in enumerate(browser.calls) if c[0] == "execute")
        assert frame_at < exec_at, browser.calls


async def test_two_frame_layout(tmp_path: Path) -> None:
    """Q-3 (pinned): a turn with two own frames (the onset and one scene change) gives ``[f1, "[screen 2]", overview,
    OVERVIEW_LABEL, f2, L1, ...]``: the off turn's parts with the overview and its label inserted immediately before
    the newest image."""
    g = live_frame("scene-2")

    async def drive(rig: Any, steps: tuple[Any, ...]) -> None:
        await rig.onset()
        rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash="scene-2"))
        await rig.pump()
        await rig.advance(200)
        await rig.words(tsl.QUESTION)
        await rig.end_of_turn()
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)

    runs = await off_and_on(
        tmp_path, lambda: deals_browser(frames=(F0, g)), script=[answer(S3)], drive=drive
    )
    own_off = last_user(runs["off"]["pass1"])["content"]
    assert image_urls(runs["off"]["pass1"]) == [builder_url(F0), builder_url(g)], (
        "premise: two own frames"
    )
    assert own_off[1].get("type") == "text" and "screen 2" in own_off[1]["text"], (
        f"premise: the second frame's label; {kinds(own_off)}"
    )
    parts = last_user(runs["on"]["pass1"])["content"]
    want = [*own_off[:2], same_shape(ov_part(), own_off[2]), label_part(), *own_off[2:]]
    assert parts == want, f"T-0150 Q-3: the 2-frame layout; got {kinds(parts)}"


# =====================================================================================================
# A3: fits is a no-op
# =====================================================================================================


async def test_fits_is_byte_identical(tmp_path: Path) -> None:
    """A3 (adversarial; the mutant always inserts a label, and reviewer mutation 8, ``shown`` counted on fits): on a
    page of 1.0 screens every request body of the turn (step 1, step 2, prefills' and captions' included) equals the
    same scripted turn's with ``auto_overview`` off, byte for byte, and so does every ``llm.request`` record; the turn
    lines differ only by the ``auto_overview`` key, which is ``fits`` 1. One capture (reading R1)."""
    runs = await off_and_on(
        tmp_path,
        lambda: deals_browser(metrics=FITS),
        script=[answer(S1), tool(VALID_CLICK)],
        steps=([answer(S3)],),
    )
    off, on = runs["off"], runs["on"]
    assert len(captures(on["browser"])) == 1, "R1: the fits page is captured once"
    assert json.dumps(on["bodies"]) == json.dumps(off["bodies"]), "T-0150 A3: a body differs"
    assert json.dumps(on["steps"]) == json.dumps(off["steps"]) and len(on["steps"]) == 1

    def scrub(recs: list[dict[str, Any]]) -> str:
        # the engine attempt id carries a per-session random prefix; everything else is compared byte for byte
        return json.dumps([{k: v for k, v in d.items() if k != "attempt_id"} for d in recs])

    assert scrub(on["requests"]) == scrub(off["requests"]), (
        "T-0150 A3: an llm.request record differs"
    )
    (line_on,), (line_off,) = on["lines"], off["lines"]
    key = line_on.get("extras", {}).pop("auto_overview", None)
    assert key == counts(fits=1), key
    for line in (line_on, line_off):
        # per-session identity, not behaviour: a random session id and the rig's own log path
        line.pop("session_id", None)
        line.get("extras", {}).pop("session_log", None)
        line.pop("session_log", None)
    assert line_on == line_off, "T-0150 A3: the turn lines differ by more than A5's key"


# =====================================================================================================
# A1.2: refused before capture
# =====================================================================================================


async def test_refused_blind_turn(tmp_path: Path) -> None:
    """A1.2 (reviewer mutation 5, a capture before the refusal check): a blind turn (the onset ``frame()`` fails) gives
    ``refused`` 1, zero capture calls, no ``overview_of`` read, and step 1 identical to the off turn's."""

    def make() -> FakeLiveBrowser:
        b = deals_browser(frames=())
        b.frame_mode = "raise"
        return b

    runs = await off_and_on(tmp_path, make, script=[answer(S3)])
    on = runs["on"]
    assert image_urls(on["pass1"]) == [], "premise: blind"
    assert captures(on["browser"]) == [] and on["browser"].overview_reads == [], (
        f"T-0150 A1.2: nothing is dispatched on a blind turn; calls {on['browser'].calls}"
    )
    assert json.dumps(on["pass1"]) == json.dumps(runs["off"]["pass1"])
    assert on["lines"][0]["extras"].get("auto_overview") == counts(refused=1)


async def test_refused_rider_turn(tmp_path: Path) -> None:
    """A1.2: turn 2's onset frame has turn 1's bytes, so its frame rides on the history image (no ``ImagePart`` in
    its final user message): ``refused`` 1, no capture for turn 2, and turn 2's step 1 identical to the off session's
    (turn 1 captured and showed it; the overview never entered history)."""
    again = F0.model_copy(update={"ts_ms": 77_777})

    async def drive(rig: Any, steps: tuple[Any, ...]) -> None:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        await next_turn(rig, 2, tsl.QUESTION, [answer(S3)])

    runs = {}
    for arm in ("off", "on"):
        browser = deals_browser(frames=(F0, again))
        async with rig_for(
            tmp_path / arm, browser, on=arm == "on", script=[answer("Deals. ")]
        ) as rig:
            await drive(rig, ())
            runs[arm] = (rig, browser, pass1_body(rig, "t2"), [json.loads(x) for x in rig.lines()])
    rig, browser, t2, lines = runs["on"]
    own = last_user(t2)["content"]
    assert "image_url" not in [p["type"] for p in own], "premise: turn 2 rides on history"
    assert len(captures(browser)) == 1, (
        f"T-0150 A1.2: only turn 1 captured; {[i.call_id for i in captures(browser)]}"
    )
    assert json.dumps(t2) == json.dumps(runs["off"][2]), "turn 2's step 1 equals the off session's"
    assert [x["extras"].get("auto_overview") for x in lines] == [counts(shown=1), counts(refused=1)]


async def _frames_turn(rig: Any, n: int) -> None:
    """Onset plus a scene change per further frame, then the question (one turn, ``n`` own frames)."""
    await rig.onset()
    for i in range(1, n):
        rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash=f"scene-{i + 1}"))
        await rig.pump()
        await rig.advance(200)
    await rig.words(tsl.QUESTION)
    await rig.end_of_turn()


async def test_refused_image_slots(tmp_path: Path) -> None:
    """A1.2 and R-s, both directions (``.claude/rules/tests.md``): with ``max_images`` = 4 and 4 images already in the
    step-1 prompt (a history image plus three own frames) the auto overview is refused before capture (no capture for
    that turn, ``refused`` 1, 4 images sent, none evicted); with 3 images (three own frames, no history) it is shown
    and step 1 carries exactly 4 images, the overview immediately before the newest frame."""
    a, b, c, d = (live_frame(t) for t in ("hist-a", "own-b", "own-c", "own-d"))
    browser = deals_browser(frames=(a, b, c, d), metrics=FITS)
    async with rig_for(tmp_path / "full", browser, on=True, script=[answer("A shop. ")]) as rig:
        assert rig.session.prompts.budget.max_images == 4, "premise: the engine's four image slots"
        await rig.ask("what is on the screen")
        await rig.turn_done()
        assert len(captures(browser)) == 1, "premise: turn 1 captured (fits)"
        browser.page_metrics = dict(DEALS)
        rig.llm.script = [answer(S3)]
        await _frames_turn(rig, 3)
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        await rig.advance(300)
        p1 = pass1_body(rig, "t2")
        assert len(image_urls(p1)) == 4 and ov_in(p1) == 0, (
            f"T-0150 A1.2: 4 images and no overview; {len(image_urls(p1))}"
        )
        assert len(captures(browser)) == 1, "T-0150 A1.2: nothing is captured without an image slot"
        assert auto_key(rig, "t2") == counts(refused=1), auto_key(rig, "t2")
    b2, c2, d2 = (live_frame(t) for t in ("own-b2", "own-c2", "own-d2"))
    browser = deals_browser(frames=(b2, c2, d2))
    async with rig_for(tmp_path / "three", browser, on=True, script=[answer(S3)]) as rig:
        await _frames_turn(rig, 3)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        urls = image_urls(pass1_body(rig))
        assert urls == [builder_url(b2), builder_url(c2), ov_url(), builder_url(d2)], (
            "T-0150 A1.2: with a free slot the overview is shown before the newest frame; "
            f"{kinds(last_user(pass1_body(rig))['content'])}"
        )
        assert auto_key(rig) == counts(shown=1)


# =====================================================================================================
# A1.3: the executor's outcomes, the timeout
# =====================================================================================================


@pytest.mark.parametrize("case", ["moved", "no_frame"])
async def test_failed_outcomes(tmp_path: Path, case: str) -> None:
    """A1.3/A5: an executor ``failed`` (the page moved) and an ``executed`` outcome whose ``overview_of`` gives
    ``None`` each give ``failed`` 1, no image, no ``overview`` request key, and step 1 identical to the off turn's."""

    def make() -> FakeLiveBrowser:
        b = deals_browser()
        if case == "moved":
            b.overview_moves = True
        else:
            b.overview_of = lambda key: None  # type: ignore[method-assign]
        return b

    runs = await off_and_on(tmp_path, make, script=[answer(S3)])
    on = runs["on"]
    assert len(captures(on["browser"])) == 1, "premise: captured"
    assert ov_in(on["pass1"]) == 0 and OV_STEM not in json.dumps(on["pass1"])
    assert json.dumps(on["pass1"]) == json.dumps(runs["off"]["pass1"])
    assert on["lines"][0]["extras"].get("auto_overview") == counts(failed=1)
    assert not [d for d in on["requests"] if "overview" in d]


async def test_executor_exception_text_never_logged(tmp_path: Path) -> None:
    """A5 (B15) and reading R4: an executor whose capture raises with a URL in its message gives step 1 without an
    overview, exactly one count, and the URL in no session-log record and no turn line."""
    secret = "http://127.0.0.1:9/secret-path?q=T0150-URL-SENTINEL"
    browser = deals_browser()

    async def boom(fb: FakeLiveBrowser, inv: Any) -> Any:
        raise RuntimeError(f"capture failed at {secret}")

    real_execute = browser.execute

    async def execute(inv: Any) -> Any:
        if inv.name == "page_overview":
            browser.invocations.append(inv)
            return await boom(browser, inv)
        return await real_execute(inv)

    browser.execute = execute  # type: ignore[method-assign]
    async with rig_for(tmp_path, browser, on=True, script=[answer(S3)]) as rig:
        await run_turn(rig, ())
        assert image_urls(pass1_body(rig)) == [builder_url(F0)], (
            "step 1 is sent without an overview"
        )
        got = auto_key(rig)
        assert isinstance(got, dict) and sum(got.values()) == 1, got
        lines = "\n".join(rig.lines())
    assert "T0150-URL-SENTINEL" not in json.dumps(rig.log_records), (
        "T-0150 A5 (B15): the exception's text reached the session log"
    )
    assert "T0150-URL-SENTINEL" not in lines


async def test_capture_timeout(tmp_path: Path) -> None:
    """A1.3/A5/A8 (reviewer mutation 12, no session-side timeout): a capture that never returns, with the clock
    advanced past ``AUTO_OVERVIEW_TIMEOUT_MS``: the capture is cancelled, step 1 is sent without an overview (equal to
    the off turn's), ``failed`` 1, exactly one note ``{"what": "auto overview timed out", "timeout_ms": 2000}``, and the
    turn completes normally (spoken, IDLE)."""
    limit = timeout_ms()

    def make() -> FakeLiveBrowser:
        b = deals_browser()
        b.overview_hang = True
        return b

    runs = await off_and_on(tmp_path, make, script=[answer(S3)])
    on = runs["on"]
    browser = on["browser"]
    assert any(c[0] == "overview.cancelled" for c in browser.calls), (
        f"T-0150 A1.3: the capture is cancelled on expiry; calls {browser.calls}"
    )
    assert json.dumps(on["pass1"]) == json.dumps(runs["off"]["pass1"])
    assert on["lines"][0]["extras"].get("auto_overview") == counts(failed=1)
    rig = on["rig"]
    notes = [d for d in rig.kind("note") if d.get("what") == TIMEOUT_NOTE]
    assert notes == [{"what": TIMEOUT_NOTE, "timeout_ms": limit}], notes
    assert rig.spoken_sources().replace(" ", "") == S3.replace(" ", ""), "the turn completed"
    # the step-1 request went out only after the deadline (the session waited, then gave up)
    gens = [
        r
        for r in rig.log_records
        if r["kind"] == "llm.request" and r["data"].get("phase") == "generate"
    ]
    assert len(gens) == 1


async def test_slow_capture_within_timeout_is_shown(tmp_path: Path) -> None:
    """A1.3 (the timeout is not too short; ``.claude/rules/tests.md`` both directions): a capture released 1.5 s after
    it started (under 2 s) is shown, with no timeout note."""
    timeout_ms()
    browser = deals_browser()
    browser.overview_gate = asyncio.Event()
    async with rig_for(tmp_path, browser, on=True, script=[answer(S3)]) as rig:
        rig.llm.script_steps()
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: (
                ("overview.start", captures(browser)[0].call_id) in browser.calls
                if captures(browser)
                else False
            ),
            "the capture never started",
            limit_ms=6_000,
        )
        await rig.advance(1_500)
        browser.overview_gate.set()
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        assert image_urls(pass1_body(rig)) == [ov_url(), builder_url(F0)]
        assert auto_key(rig) == counts(shown=1)
    assert [d for d in rig.kind("note") if d.get("what") == TIMEOUT_NOTE] == []


@pytest.mark.parametrize("case", ["closed", "unavailable"])
async def test_executor_refusal_counts(tmp_path: Path, case: str) -> None:
    """A1.3/A5: the executor's own refusals count as ``refused``: a browser closed after the onset frame landed (the
    executor refuses with A2's reason) and a browser without the capture (``page overview is not available``) each
    give ``refused`` 1 and step 1 sent with the current frame and no overview."""
    browser = deals_browser()
    if case == "closed":
        real_frame = browser.frame

        async def frame_then_close() -> Frame:
            f = await real_frame()
            browser.close_window()
            return f

        browser.frame = frame_then_close  # type: ignore[method-assign]
    else:
        browser.overview_available = False
    async with rig_for(tmp_path, browser, on=True, script=[answer(S3)]) as rig:
        await run_turn(rig, ())
        assert len(captures(browser)) == 1, f"premise: the capture was dispatched; {browser.calls}"
        assert image_urls(pass1_body(rig)) == [builder_url(F0)], "step 1 is sent, no overview"
        assert auto_key(rig) == counts(refused=1), auto_key(rig)


# =====================================================================================================
# A1.5: no overlap with frame()
# =====================================================================================================


def _interleaving(browser: FakeLiveBrowser) -> list[str]:
    return [
        c[0]
        for c in browser.calls
        if c[0] in ("frame", "overview.start", "overview.end", "overview.cancelled")
    ]


def _no_frame_inside_capture(browser: FakeLiveBrowser) -> None:
    seq = _interleaving(browser)
    inside = False
    for i, what in enumerate(seq):
        if what == "overview.start":
            inside = True
        elif what in ("overview.end", "overview.cancelled"):
            inside = False
        elif what == "frame":
            assert not inside, (
                f"T-0150 A1.5: frame() called while the capture ran; sequence {seq} (at {i})"
            )


async def test_no_overlap_with_frame_takeover(tmp_path: Path) -> None:
    """A1.5 (adversarial; R3/Q-5, the machine path): a takeover while the capture is held aborts the generate and
    requests the new turn's live snapshot; its ``frame()`` is called only after the capture ended (here: after the
    cancelled capture finished unwinding, ``overview_cancel_ticks``). The new turn then completes."""
    browser = deals_browser(frames=(F0, live_frame("t2-onset")))
    browser.overview_gate = asyncio.Event()
    browser.overview_cancel_ticks = 6
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: any(c[0] == "overview.start" for c in browser.calls),
            "the capture never started",
            limit_ms=6_000,
        )
        frames_before = len([c for c in browser.calls if c[0] == "frame"])
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        await rig.advance(400)
        assert len([c for c in browser.calls if c[0] == "frame"]) > frames_before, (
            "premise: the takeover requested a live snapshot"
        )
        browser.overview_gate.set()
        _no_frame_inside_capture(browser)
        rig.llm.script = [answer("Okay. ")]
        await rig.words("stop that")
        await rig.end_of_turn()
        await rig.until(lambda: len(rig.lines()) >= 1, "no turn line", limit_ms=20_000)
        await rig.advance(300)
    _no_frame_inside_capture(browser)
    _outer_cancel_not_a_timeout(rig)


def _outer_cancel_not_a_timeout(rig: Any) -> None:
    """Review A F2/F1: a takeover's cancel of the generate that is waiting on the capture is not a timeout: no
    ``auto overview timed out`` note, no ``generate`` request for the aborted turn (``t1-gen``), and t1's line counts
    all four at 0 (step 1 was never sent; A5 "sum to 0"). The turn that follows runs its own generate."""
    notes = [d for d in rig.kind("note") if d.get("what") == TIMEOUT_NOTE]
    assert notes == [], f"T-0150 A1.3 (review F2): an outer cancel was logged as a timeout; {notes}"
    t1 = [
        d for d in requests(rig) if d.get("phase") == "generate" and d.get("request_id") == "t1-gen"
    ]
    assert t1 == [], (
        f"T-0150 A1.3 (review F2): the aborted turn's step 1 was sent; {len(t1)} request(s)"
    )
    assert tsl.generates_for(rig, "t1-gen") == [], "the backend saw a t1-gen generate"
    lines = {json.loads(x)["turn_id"]: json.loads(x) for x in rig.lines()}
    assert "t1" in lines, f"premise: the aborted turn has a line; {sorted(lines)}"
    assert lines["t1"]["extras"].get("auto_overview") == counts(), (
        f"T-0150 A5 (review F2): t1 never sent step 1; got {lines['t1']['extras'].get('auto_overview')}"
    )
    later = [
        d for d in requests(rig) if d.get("phase") == "generate" and d.get("request_id") != "t1-gen"
    ]
    assert later, "premise: the next turn's generate ran"


async def test_takeover_inside_timeout_unwind(tmp_path: Path) -> None:
    """Review A F1 (adversarial; the race): a capture that never returns and unwinds slowly after a cancel
    (``overview_cancel_ticks`` 50). The clock reaches the deadline (the session cancels the capture: ``expired``), and
    while the cancelled capture is still unwinding a ``UserTakeover`` aborts the generate. The takeover's cancel wins:
    t1's step 1 is never sent, no timeout note is written for t1, t1 counts all 0, and only the next turn's generate
    runs."""
    limit = timeout_ms()
    browser = deals_browser(frames=(F0, live_frame("t2-onset")))
    browser.overview_hang = True
    browser.overview_cancel_ticks = 50
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: any(c[0] == "overview.start" for c in browser.calls),
            "the capture never started",
            limit_ms=6_000,
            step=10,
        )
        started = rig.clock.now_ms
        await rig.advance(limit - 100 - (rig.clock.now_ms - started), step=50)
        assert not any(c[0] == "overview.cancelled" for c in browser.calls), (
            "premise: not expired yet"
        )
        rig.clock.advance(200)  # the deadline fires: the capture is cancelled and starts unwinding
        for _ in range(3):
            await asyncio.sleep(0)
        assert not any(c[0] == "overview.cancelled" for c in browser.calls), (
            "premise: the cancelled capture is still unwinding"
        )
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        await rig.until(
            lambda: any(c[0] == "overview.cancelled" for c in browser.calls),
            "premise: the cancelled capture never finished unwinding",
            limit_ms=2_000,
            step=10,
        )
        browser.overview_hang = (
            False  # the next turn's capture returns (so any timeout note is t1's)
        )
        rig.llm.script = [answer("Okay. ")]
        await rig.words("stop that")
        await rig.end_of_turn()
        await rig.until(lambda: len(rig.lines()) >= 2, "no second turn line", limit_ms=20_000)
        await rig.advance(300)
    _outer_cancel_not_a_timeout(rig)


async def test_cancelled_capture_still_chains_snapshots(tmp_path: Path) -> None:
    """A1.5 / review A F3 (adversarial; the mutant drops the chain spawned when the capture starts behind a pending
    snapshot): a live snapshot is held inside ``frame()`` (the fake's ``frame_gate``); the session's
    ``_auto_overview`` starts behind it and times out (2,100 ms) while still waiting, so its capture task ends first.
    A second snapshot requested then must not call ``frame()`` while the first is still inside it; once the gate is
    released exactly one more ``frame()`` runs, never two at once, and the chain is done. Driven through the session's
    seams (``snapshot_frame``, ``_auto_overview``, ``_live_snap``): the machine reaches this window only between the
    generate's snapshot wait and the capture, which the fakes cannot hold open (the test-author report)."""
    timeout_ms()
    s = session_mod()
    browser = deals_browser(frames=(F0, live_frame("second")))
    browser.frame_gate = asyncio.Event()
    async with rig_for(tmp_path, browser, on=True, script=[]) as rig:
        session = rig.session
        await session.snapshot_frame(rig.clock.now_ms)
        await rig.pump()
        held = session._live_snap
        assert held is not None and not held.done(), (
            "premise: the first snapshot is held in frame()"
        )
        frames = [c[0] for c in browser.calls].count("frame")
        assert frames == 1, f"premise: one frame() call; {browser.calls}"
        turn = s._TurnLog(turn_id="tseam", seq=0)
        step1 = Prompt(
            messages=(
                Message(role="system", content=(TextPart(text="sys"),)),
                Message(
                    role="user",
                    content=(ImagePart(url=builder_url(F0), phash=F0.phash), TextPart(text="q")),
                ),
            ),
            prefix_len=0,
        )
        auto = asyncio.ensure_future(session._auto_overview(turn, step1))
        await rig.pump()
        assert captures(browser) == [], "premise: the capture waits behind the held snapshot"
        await rig.advance(2_100)
        await rig.until(
            lambda: auto.done(), "_auto_overview never returned after its deadline", limit_ms=500
        )
        assert auto.exception() is None, auto.exception()
        assert not held.done(), "premise: the first snapshot is still held"
        await session.snapshot_frame(rig.clock.now_ms)
        await rig.pump()
        await rig.advance(200)
        assert [c[0] for c in browser.calls].count("frame") == frames, (
            "T-0150 A1.5 (review F3): a second snapshot called frame() while the first was still inside it; "
            f"calls {[c[0] for c in browser.calls]}"
        )
        browser.frame_gate.set()
        await rig.advance(300)
        assert [c[0] for c in browser.calls].count("frame") == frames + 1, browser.calls
        assert browser.frame_max_active == 1, "two frame() calls ran at once"
        assert session._live_snap is not None and session._live_snap.done(), "the chain is done"
        assert captures(browser) == [], "the timed-out capture never dispatched"


async def test_no_overlap_with_frame_seam(tmp_path: Path) -> None:
    """A1.5 (adversarial; R3/Q-5, the session's snapshot seam): a live snapshot requested through
    ``Session.snapshot_frame`` while the capture is held (not cancelled) calls ``frame()`` only after the capture
    returned; the turn completes with the overview shown."""
    browser = deals_browser(frames=(F0, live_frame("mid")))
    browser.overview_gate = asyncio.Event()
    async with rig_for(tmp_path, browser, on=True, script=[answer(S3)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: any(c[0] == "overview.start" for c in browser.calls),
            "the capture never started",
            limit_ms=6_000,
        )
        await rig.session.snapshot_frame(rig.clock.now_ms)
        await rig.advance(500)
        assert [c[0] for c in browser.calls].count("frame") == 1, (
            f"T-0150 A1.5: frame() ran while the capture was held; {_interleaving(browser)}"
        )
        browser.overview_gate.set()
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        _no_frame_inside_capture(browser)
        assert [c[0] for c in browser.calls].count("frame") == 2, (
            "the requested snapshot ran after the capture"
        )
        assert ov_in(pass1_body(rig)) == 1


# =====================================================================================================
# A2/A4: never an own frame, never spoken or stored, outside the tool path
# =====================================================================================================


async def test_never_own_frame_or_stored(tmp_path: Path) -> None:
    """A2/A4 (adversarial; reviewer mutation 7, the overview's phash registered in ``_frame_geometry`` or the turn's
    frames): across an overview turn (click, report) and a second turn, the overview's phash and its capture identity
    are in no history message, ``Episode``, turn frames (the turn line's ``frames``), ``_frame_geometry`` or
    ``_frame_bytes``; the step-1 ``click_at`` on the current frame is ``executed``, not refused as stale; the label
    (stem, "1.5 screens", closing sentence), the pixels and the action string appear in no ``TtsSpeak``, history, HUD
    message, ``Episode`` or ``machine.events()``; the voiced text is exactly the model's answers (no narration)."""
    browser = deals_browser(frames=(F0, live_frame("t2-onset")))
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await run_turn(rig, ([answer(S3)],))
        await next_turn(rig, 2, "and now", [answer("Still the deals. ")])
        sent = json.dumps(all_bodies(rig))
        assert OV_STEM in sent and b64(OV.jpeg) in sent, "premise: the overview was sent"
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"], (
            "A2: the step-1 click on the current frame executes"
        )
        session = rig.session
        for name in ("_frame_geometry", "_frame_bytes"):
            held = getattr(session, name)
            keys = set(held)
            assert OV.phash not in keys and OV_IDENTITY not in keys, (
                f"T-0150 A2: the overview is registered in {name}"
            )
            if name == "_frame_geometry":
                assert all(getattr(g, "phash", None) != OV.phash for g in held.values()), name
            else:
                assert OV.jpeg not in held.values(), name
        for t in ("t1", "t2"):
            frames = list(rig.record(t).extras.get("frames", []))
            assert OV.phash not in frames and OV_IDENTITY not in frames, (t, frames)
        needles = (
            OV_STEM,
            "1.5 screens",
            OV_CLOSE,
            b64(OV.jpeg),
            OV.phash,
            OV_IDENTITY,
            OV_ACTION,
            "showed the whole page",
        )
        speaks = [c.text + "|" + c.source for c in rig.machine.commands(C.TtsSpeak)]
        assert speaks, "premise: something was spoken"
        stores = {
            "TtsSpeak": json.dumps(speaks),
            "history": tsl.history_text(rig),
            # T-0177 B-P2: "the HUD" reads "any HUD message other than astra.vision.frame" (H-T177-2)
            "HUD": published_json_outside_frames(rig.published.messages),
            "Episode": json.dumps(
                [e.model_dump(mode="json") for e in rig.memory.episodes.values()]
            ),
            "machine.events()": json.dumps([repr(e) for e in rig.machine.events()]),
        }
        for where, text in stores.items():
            for n in needles:
                assert n not in text, f"T-0150 A2/A4: {n[:40]!r} reached {where}"
        voiced = "".join(c.source for c in rig.machine.commands(C.TtsSpeak))
        assert voiced.replace(" ", "") == (S1 + S3 + "Still the deals. ").replace(" ", ""), (
            f"A4: no narration is added; voiced {voiced!r}"
        )


async def test_outside_tool_path(tmp_path: Path) -> None:
    """A1.6/R-a: the capture produces no tool card, no ``tool.*`` record, no ``ToolDraft``/``ToolResult`` and no
    machine event (the machine's events and commands, and the tool records, equal the off turn's); the step-1 body's
    ``tools`` are v1's declarations exactly (no ``page_overview``)."""
    runs = await off_and_on(
        tmp_path, deals_browser, script=[answer(S1), tool(VALID_CLICK)], steps=([answer(S3)],)
    )
    off, on = runs["off"], runs["on"]
    assert ov_in(on["pass1"]) == 1, "premise: the overview was shown"
    v1 = native.tools_field(tools.load())
    assert (
        on["pass1"].get("tools") == off["pass1"].get("tools") == json.loads(json.dumps(list(v1)))
    ), "T-0150 R-a: the step-1 tools are v1's"
    assert "page_overview" not in json.dumps(on["pass1"].get("tools"))
    ron, roff = on["rig"], off["rig"]
    assert [m.call_id for m in ron.tool_calls()] == [m.call_id for m in roff.tool_calls()], (
        "T-0150 A1.6: a tool card for the auto capture"
    )
    assert "page_overview" not in tsl.published_json(ron), "A1.6: the HUD heard of the capture"

    def tool_records(run: dict[str, Any]) -> list[tuple[str, Any]]:
        return [
            (r["kind"], r["data"].get("call_id"))
            for r in run["rig"].log_records
            if str(r["kind"]).startswith("tool.")
        ]

    assert tool_records(on) == tool_records(off), "T-0150 A1.6: a tool.* record for the capture"
    assert "page_overview" not in json.dumps(
        [r for r in ron.log_records if str(r["kind"]).startswith("tool.")]
    )
    ev_on = [repr(e) for e in ron.machine.events()]
    ev_off = [repr(e) for e in roff.machine.events()]
    assert len(ev_on) == len(ev_off) and [type(e).__name__ for e in ron.machine.events()] == [
        type(e).__name__ for e in roff.machine.events()
    ], "T-0150 A1.6: a machine event for the capture"
    assert not [e for e in ev_on if "page_overview" in e or "auto-overview" in e]
    assert [type(c).__name__ for c in ron.machine.commands()] == [
        type(c).__name__ for c in roff.machine.commands()
    ], "A1.6: the machine's commands differ"


# =====================================================================================================
# A5: the turn line
# =====================================================================================================

OUTCOMES: dict[str, tuple[dict[str, Any], dict[str, int]]] = {
    "shown": ({}, counts(shown=1)),
    "fits": ({"page_metrics": FITS}, counts(fits=1)),
    "failed": ({"overview_moves": True}, counts(failed=1)),
    "refused_executor": ({"overview_available": False}, counts(refused=1)),
    "refused_blind": ({"frame_mode": "raise"}, counts(refused=1)),
}


@pytest.mark.parametrize("case", list(OUTCOMES))
async def test_turn_line_counts(tmp_path: Path, case: str) -> None:
    """A5 (reviewer mutation 8, ``shown`` on fits; mutation 10's on half): each scripted outcome gives the key with
    exactly one count at 1 (written out per case); a second, toolless turn of the same session gets its own counts
    (not accumulated); no new log kind and no new span against the off session."""
    attrs, want = OUTCOMES[case]
    runs: dict[str, Any] = {}
    for arm in ("off", "on"):
        browser = deals_browser(frames=(F0, live_frame("t2-onset")))
        for k, v in attrs.items():
            setattr(browser, k, dict(v) if isinstance(v, dict) else v)
        async with rig_for(tmp_path / arm, browser, on=arm == "on", script=[answer(S3)]) as rig:
            await run_turn(rig, ())
            await next_turn(rig, 2, "and now", [answer("Again. ")])
            runs[arm] = (rig, [json.loads(x) for x in rig.lines()])
    rig, lines = runs["on"]
    assert lines[0]["extras"].get("auto_overview") == want, (
        case,
        lines[0]["extras"].get("auto_overview"),
    )
    second = lines[1]["extras"].get("auto_overview")
    assert isinstance(second, dict) and set(second) == set(A5_KEYS) and sum(second.values()) == 1, (
        second
    )
    off_rig, off_lines = runs["off"]
    assert all("auto_overview" not in x["extras"] for x in off_lines), "A5: no key when off"
    on_kinds = {r["kind"] for r in rig.log_records}
    off_kinds = {r["kind"] for r in off_rig.log_records}
    assert on_kinds <= off_kinds, f"A5: no new log kind; {sorted(on_kinds - off_kinds)}"
    for a, b in zip(lines, off_lines, strict=True):
        assert set(a.get("spans", {})) == set(b.get("spans", {})), "A5: no new span"


async def test_turn_line_zero_without_step1(tmp_path: Path) -> None:
    """A5 (reading R2): a turn taken over while its generate still waits for the onset frame (step 1 never built)
    gives a line with all four counts 0 and no capture; the turn that follows counts its own capture."""
    browser = deals_browser(frames=(F0, live_frame("t2-onset")))
    browser.frame_gate = asyncio.Event()
    async with rig_for(tmp_path, browser, on=True, script=[answer(S3)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.advance(300)
        assert not browser.frames_returned, "premise: the onset frame is held"
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        browser.frame_gate.set()
        rig.llm.script = [answer("Okay. ")]
        await rig.words("never mind")
        await rig.end_of_turn()
        await rig.until(lambda: len(rig.lines()) >= 1, "no turn line", limit_ms=20_000)
        await rig.advance(300)
        lines = [json.loads(x) for x in rig.lines()]
    zero = [x for x in lines if x["extras"].get("auto_overview") == counts()]
    assert zero, (
        "T-0150 A5: a turn that never sent step 1 counts 0 (reading R2); lines "
        f"{[(x.get('turn_id'), x['extras'].get('auto_overview')) for x in lines]}"
    )
    assert all("auto_overview" in x["extras"] for x in lines), (
        "A5: every live turn line carries the key"
    )


# =====================================================================================================
# A7/A8: refusals, constants
# =====================================================================================================


async def test_session_refusals(tmp_path: Path) -> None:
    """A7: ``Session(auto_overview=True)`` raises ``ValueError`` with a ``dry_run`` kit and with a live kit offering
    ``page_overview``; too-wide companion: the live v1 kit is accepted (every other row)."""
    from tests.unit.test_t0140_session import overview_kit

    browser = deals_browser()
    with pytest.raises(ValueError):
        async with rig_for(
            tmp_path / "dry", browser, on=True, script=[], feedback=None, kit=T.make_kit()
        ):
            pass
    with pytest.raises(ValueError):
        async with rig_for(tmp_path / "ov", browser, on=True, script=[], kit=overview_kit(browser)):
            pass
    async with rig_for(tmp_path / "ok", deals_browser(), on=True, script=[]) as rig:
        assert rig.session is not None


def test_timeout_constant() -> None:
    """A8: ``AUTO_OVERVIEW_TIMEOUT_MS = 2_000`` in ``session.py`` (chosen, not measured)."""
    assert timeout_ms() == 2_000


def test_max_screens_constant() -> None:
    """A8 (green at BASE): ``session.OVERVIEW_MAX_SCREENS`` still equals ``executor.MAX_OVERVIEW_SCREENS`` (4)."""
    from local_astra.computer import executor as executor_mod

    assert session_mod().OVERVIEW_MAX_SCREENS == executor_mod.MAX_OVERVIEW_SCREENS == 4
    assert MAX_OVERVIEW_SCREENS == 4


def test_live_kit_is_v1_by_default() -> None:
    """R-a (green at BASE): the live kit the rows run on offers v1's tools (no ``page_overview``)."""
    kit = live_kit(FakeLiveBrowser())
    assert kit.tools == tuple(native.tools_field(tools.load()))
    assert kit.tools_version == V1_SHA, kit.tools_version
