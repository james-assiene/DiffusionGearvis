"""T-0151 group A2 (session side): A2-1..A2-8 over the fakes (docs/tasks/T-0151.md).

**The surface these rows pin** (the task's names; where it is silent the reading is stated here and in the report):

- ``Session(..., overview_frame=True)`` (keyword; absent means off). With it on, in live, each turn whose step-1 request
  is built decides once (A2-2/A2-3): refused before capture when the final user message has no ``ImagePart`` or more
  than one; else T-0150's capture (one ``LiveBrowser.execute(ToolInvocation(name="page_overview", arguments={},
  frame=None, ...))``, chained after ``_live_snap``, bounded by ``AUTO_OVERVIEW_TIMEOUT_MS``); ``shown`` when it is
  executed with an image, ``resolved["screens"] <= OVERVIEW_FRAME_MAX_SCREENS`` (2.0) and
  ``getattr(browser, "overview_frame_of")(key)`` returns a ``Frame``; ``too_tall`` above the cap (the picture read
  through ``overview_of`` and discarded); ``fits``; ``failed``; ``refused``.
- A2-4: the step-1 request as sent is ``label_current_image(swap_newest_image(unlabelled, overview),
  OVERVIEW_FRAME_LABEL.format(screens=s))``: one image (the overview), the label right after it, no live frame and no
  T-0134 label; the over-long retry and the step-1 repair carry the same.
- A2-5: the overview is registered under its capture identity in ``_frame_geometry`` (geometry phash = the
  executor's); the step-1 call's ``frame_identity`` is that identity; B9 exempts exactly the step-1 call on this turn's
  overview identity while ``latest_frame`` is still the live frame the overview replaced.
- A2-8: the turn line's ``overview_frame == {"shown", "fits", "too_tall", "failed", "refused"}`` (ints);
  ``llm.request``'s ``overview_frame == {"phash", "screens", "width", "height"}`` on every request carrying it; the
  step-1 call's ``tool.call`` record has ``"frame_kind": "overview"``.

**The fake** is ``_live_browser_support.FakeLiveBrowser`` with its T-0151 group A2 additions (``overview_frame_of``
mirroring A1-1, a click on a registered overview frame mirroring A1-2..A1-5); see that module's docstring.

Readings (stated, not guessed silently; see the report):

- **R1 (A2-8 request ``phash``), ruled O-A2 (2f33f3f):** the record's ``phash`` is the executor's phash of the
  picture exactly (what T-0150's ``overview`` record carried), never the session's capture identity.
- **R2 (A2-8 "sum to 0").** As T-0150's R2: the criterion's example (a barge-in before ``LlmStart``) leaves no turn
  line on this machine; the row uses a takeover while the generate still waits for the turn's live frame.
- **R3 (A2-5, the B9 seam).** A step >= 2 call carrying the overview identity, and a call carrying an earlier turn's
  overview identity, are unreachable through the rig when placement is right (the overview is in no step >= 2 request
  and no history). The row reaches them through the session's B9 predicate ``_live_stale_frame`` (named by the task's
  implementation notes: "it may take the turn"): a spy records the step-1 call as the session checks it, and re-asks
  the same predicate, with the same other arguments, for a copy of that call with ``loop_step=2`` or another turn's
  overview identity. This assumes the exemption lives in that predicate.
- **R4 (A2-2 timeout note).** "the same ... timeout note": the rows accept one ``note`` whose ``what`` contains
  ``timed out`` (not the live frame's) with ``timeout_ms == AUTO_OVERVIEW_TIMEOUT_MS``; the exact text is not pinned.
- **R5 (A2-8 ``frame_kind``).** "its ``tool.call`` record": a call has several ``tool.call`` records (proposed,
  executed); the rows pin the call's last one (its final status) and that no other call's record has the key.
- **R6 (A2-8 ``tool.result``).** "carries the executor's ``resolved`` as at BASE": at BASE the executed ``tool.call``
  record carries ``resolved`` and ``tool.result`` carries the rendered payload; the rows pin both as they are.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on, ReGround off, T-0134c's label on unless a row says so, echo ``model``; deals-like page
1220/800 CSS px (1.525 screens), overview 838x798 (docs/tasks/T-0151.md "Measured scope"); n = 1, deterministic.

BASE_A2 = 177c0a1. Activated for group A2 (rulings O-A2 in the task file; the pending markers are off): every row is
red until group A2 is implemented, except the two marked green at BASE.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.perception import Frame, Source
from local_astra.toolcall import native, tools
from tests.fakes.llm import http_error, is_caption_prompt
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    FRESHNESS,
    FakeLiveBrowser,
    b64,
    builder_url,
    feedback_rig,
    image_urls,
    live_feedback,
    live_frame,
    live_kit,
    refused,
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
    SCROLL,
    _wire,
    card,
    last_user,
    pass1_body,
    step_body,
    step_tail,
)
from tests.unit.test_t0134_label import L1, LN, LU, STEM, all_bodies, label_after, next_turn

S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
Q = tsl.Q

#: A2-4's label, typed from the criterion (H-T151-5 approved "as written"); the template and two renderings by hand
LABEL_TEMPLATE = (
    "[This screenshot is the whole web page, about {screens:.1f} screens tall, scaled down to fit; the browser "
    "screen shows only part of it. click_at and type_at coordinates refer to this image, and the browser scrolls to "
    "the point before it clicks.]"
)
LABEL_15 = (
    "[This screenshot is the whole web page, about 1.5 screens tall, scaled down to fit; the browser "
    "screen shows only part of it. click_at and type_at coordinates refer to this image, and the browser scrolls to "
    "the point before it clicks.]"
)
LABEL_20 = LABEL_15.replace("about 1.5 screens", "about 2.0 screens")
#: the label's stem (a substring of no other rig text)
OV_STEM = "This screenshot is the whole web page"
KEYS = ("shown", "fits", "too_tall", "failed", "refused")
REQUEST_KEYS = {"phash", "screens", "width", "height"}
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"

#: the overview the executor keeps for deals.html (838x798, "Measured scope"); sentinel bytes, the session only
#: carries it
OV = Frame(
    source=Source.WINDOW,
    ts_ms=0,
    width=838,
    height=798,
    jpeg=b"\xff\xd8" + b"OVERVIEW-FRAME-PIXELS-T0151|" * 8,
    phash="5151515151515151",
)
#: the session's capture identity of the overview's bytes, by hand (T-0111g B7's rule)
OV_IDENTITY = "capture:" + hashlib.sha256(OV.jpeg).hexdigest()[:32]
#: a second turn's overview (another capture)
OV_B = Frame(
    source=Source.WINDOW,
    ts_ms=0,
    width=838,
    height=798,
    jpeg=b"\xff\xd8" + b"OVERVIEW-FRAME-PIXELS-T0151-B|" * 8,
    phash="5151b5151b5151b5",
)
OV_B_IDENTITY = "capture:" + hashlib.sha256(OV_B.jpeg).hexdigest()[:32]
#: deals.html at 1280x800 (the planner's measurement): 1220/800 = 1.525 screens
DEALS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1220.0}
FITS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 800.0}

#: the tea towel button (A1-7): overview (160, 935); by hand on 1220/800 from S = 0: P = (to_px(160, 1280),
#: to_px(935, 1220)) = (205, 1141); 1141 is not in [0, 800); dy = clamp(1141 - 400, 0, 420) - 0 = 420; click y 721
TOWEL = "<|tool_call>call:click_at{x:160,y:935}<tool_call|>"
TOWEL_ACTION = "scrolled down by 420 px and clicked at (160, 935)"

F0 = live_frame("onset")
F1 = live_frame("after-1")
F2 = live_frame("after-2")
G = live_frame("t2-onset")


# --------------------------------------------------------------------------------------------- seams


def counts(**kw: int) -> dict[str, int]:
    return {k: int(kw.get(k, 0)) for k in KEYS}


def label_on() -> Any:
    return live_feedback(screen_label=True, page_extent=False)


def label_off() -> Any:
    return live_feedback(screen_label=False, page_extent=False)


def deals_browser(
    frames: Any = (F0,),
    post: Any = (F1, F2),
    metrics: dict[str, float] | None = None,
    image: Frame = OV,
) -> FakeLiveBrowser:
    browser = FakeLiveBrowser(list(frames), post_frames=list(post), echo="model")
    browser.overview_image = image
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
    kit: Any = None,
) -> Any:
    """``feedback_rig`` with ``Session(..., live_feedback=..., overview_frame=True)`` when ``on`` (off: the keyword
    is not passed at all, A2-9 "Wiring"). A session without the keyword is a named ``AssertionError``."""
    import inspect

    if on:
        params = inspect.signature(session_mod().Session.__init__).parameters
        assert "overview_frame" in params, (
            "T-0151 A2: Session(..., overview_frame=True) does not exist yet"
        )
    kw: dict[str, Any] = {}
    fb = label_on() if feedback == "on" else feedback
    if fb is not None:
        kw["live_feedback"] = fb
    if on:
        kw["overview_frame"] = True
    kw.update(extra or {})
    return feedback_rig(tmp, browser, script=script, session_kwargs=kw, kit=kit)


async def run_turn(rig: Any, steps: tuple[Any, ...], question: str = tsl.QUESTION) -> None:
    if steps:
        rig.llm.script_steps(*steps)
    await rig.ask(question)
    await tsl.until_steps(rig, len(steps), limit_ms=20_000, step=100)
    await rig.turn_done(limit_ms=20_000)
    await rig.advance(300)


def ov_url(frame: Frame = OV) -> str:
    return builder_url(frame)


def ov_in(body: dict[str, Any], frame: Frame = OV) -> int:
    """How many times the overview's pixels appear in ``body``."""
    return json.dumps(body).count(b64(frame.jpeg))


def kinds(parts: list[dict[str, Any]]) -> list[str]:
    out = []
    for p in parts:
        if p.get("type") == "image_url":
            out.append("image:OV" if p["image_url"]["url"] == ov_url() else "image")
        else:
            out.append("text:" + str(p.get("text", ""))[:48])
    return out


def same_shape(like: dict[str, Any], frame: Frame = OV) -> dict[str, Any]:
    """``like``'s image part with the overview's URL (extra wire keys of the client's image parts kept)."""
    out = json.loads(json.dumps(like))
    out["image_url"]["url"] = ov_url(frame)
    return out


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def requests(rig: Any) -> list[dict[str, Any]]:
    return rig.kind("llm.request")


def ov_records(rig: Any) -> list[dict[str, Any]]:
    return [d for d in requests(rig) if "overview_frame" in d]


def captures(browser: FakeLiveBrowser) -> list[Any]:
    return [i for i in browser.invocations if i.name == "page_overview"]


def reads(browser: FakeLiveBrowser, name: str) -> list[Any]:
    return [c for c in browser.calls if c[0] == name]


def frame_key(rig: Any, turn: str = "t1") -> Any:
    return dict(rig.record(turn).extras).get("overview_frame")


def check_request_key(rec: Any, screens: float = 1.525) -> None:
    """A2-8's request key: exactly the four keys; ``phash`` is the executor's phash of the picture exactly
    (O-A2 R1), never the capture identity; the rest by hand."""
    assert isinstance(rec, dict) and set(rec) == REQUEST_KEYS, rec
    assert rec["phash"] == OV.phash, (
        f"O-A2 R1: the record's phash is the executor's phash exactly; {rec}"
    )
    assert rec["screens"] == pytest.approx(screens) and (rec["width"], rec["height"]) == (
        838,
        798,
    ), rec


def scrub(recs: list[dict[str, Any]]) -> str:
    # the engine attempt id carries a per-session random prefix; everything else is compared byte for byte
    return json.dumps([{k: v for k, v in d.items() if k != "attempt_id"} for d in recs])


def step1_records(rig: Any) -> list[dict[str, Any]]:
    return [
        d for d in requests(rig) if d.get("phase") == "generate" and d.get("request_id") == "t1-gen"
    ]


async def off_and_on(
    tmp: Path,
    make_browser: Callable[[], FakeLiveBrowser],
    *,
    script: Any,
    steps: tuple[Any, ...] = (),
    feedback: Any = "on",
    drive: Any = None,
) -> dict[str, dict[str, Any]]:
    """The same scripted turn with ``overview_frame`` off and on: bodies, records and turn lines of each."""
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
                "lines": [json.loads(x) for x in rig.lines()],
            }
        got["requests"] = requests(rig)
        out[arm] = got
    return out


async def _frames_turn(rig: Any, n: int) -> None:
    """Onset plus a scene change per further frame, then the question (one turn, ``n`` own frames)."""
    await rig.onset()
    for i in range(1, n):
        rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash=f"scene-{i + 1}"))
        await rig.pump()
        await rig.advance(200)
    await rig.words(tsl.QUESTION)
    await rig.end_of_turn()


async def _two_frame_drive(rig: Any, steps: tuple[Any, ...]) -> None:
    await _frames_turn(rig, 2)
    await rig.turn_done(limit_ms=20_000)
    await rig.advance(300)


# =====================================================================================================
# A2-4: placement
# =====================================================================================================


async def test_step1_only_overview(tmp_path: Path) -> None:
    """A2-3/A2-4/A2-5 (``shown``): on a 1.525-screen page the step-1 final user message is the off turn's with the
    live frame replaced by the overview and the T-0134 label replaced by ``OVERVIEW_FRAME_LABEL(1.5)``: ``[OV, LABEL_15,
    ...transcript]``; exactly one image in the body; the generate record has ``overview_frame`` and no ``live_label``;
    the step-1 click reaches the browser with the overview's executor phash and size as its frame and is executed (B9
    exempts it); exactly one capture (``arguments == {}``, no frame) and one ``overview_frame_of`` read."""
    runs = await off_and_on(
        tmp_path, deals_browser, script=[answer(S1), tool(TOWEL)], steps=([answer(S3)],)
    )
    off, on = runs["off"], runs["on"]
    own_off = last_user(off["pass1"])["content"]
    assert own_off[0]["image_url"]["url"] == builder_url(F0) and own_off[1] == text_part(L1), (
        f"premise: the off turn shows F0 then L1; {kinds(own_off)}"
    )
    parts = last_user(on["pass1"])["content"]
    want = [same_shape(own_off[0]), text_part(LABEL_15), *own_off[2:]]
    assert parts == want, f"T-0151 A2-4: the step-1 final user message; got {kinds(parts)}"
    assert image_urls(on["pass1"]) == [ov_url()], "A2-4: exactly one image, the overview"
    assert STEM not in json.dumps(on["pass1"]), "A2-4: no T-0134 label on the overview request"
    gen = step1_records(on["rig"])
    assert len(gen) == 1, gen
    check_request_key(gen[0].get("overview_frame"))
    assert "live_label" not in gen[0], "A2-4: the record has no live_label key"
    browser = on["browser"]
    clicks = [i for i in browser.invocations if i.call_id == "t1-call"]
    assert len(clicks) == 1 and clicks[0].frame is not None, clicks
    fr = clicks[0].frame
    assert (fr.phash, fr.source, fr.width, fr.height) == (OV.phash, Source.WINDOW, 838, 798), (
        f"T-0151 A2-5: the step-1 call's frame is the overview's geometry (executor phash); got {fr}"
    )
    assert [e.status for e in tsl.results_for(on["rig"], "t1-call")] == ["executed"], (
        "A2-5: the step-1 call on the overview is executed, not refused as stale"
    )
    caps = captures(browser)
    assert len(caps) == 1 and caps[0].arguments == {} and caps[0].frame is None, caps
    assert len(reads(browser, "overview_frame_of")) == 1, browser.calls


async def test_replaces_not_adds(tmp_path: Path) -> None:
    """A2-4 (adversarial; reviewer mutation 1, ``place_overview`` as T-0150 did): no live-frame bytes appear anywhere
    in the step-1 body (premise: they do in the off turn's), and the overview's appear exactly once."""
    runs = await off_and_on(tmp_path, deals_browser, script=[answer(S3)])
    assert b64(F0.jpeg) in json.dumps(runs["off"]["pass1"]), "premise: the off turn carries F0"
    on = runs["on"]["pass1"]
    assert b64(F0.jpeg) not in json.dumps(on), (
        f"T-0151 A2-4: the live frame is in the step-1 body beside the overview; {kinds(last_user(on)['content'])}"
    )
    assert ov_in(on) == 1


async def test_label_with_feedback_off(tmp_path: Path) -> None:
    """A2-4: with ``LiveFeedback(False, False)`` the overview label is still placed: the off turn's ``[F0, ...]``
    becomes ``[OV, LABEL_15, ...]``; the record has ``overview_frame`` and no ``live_label``."""
    runs = await off_and_on(tmp_path, deals_browser, script=[answer(S3)], feedback=label_off())
    own_off = last_user(runs["off"]["pass1"])["content"]
    assert own_off[0]["image_url"]["url"] == builder_url(F0), kinds(own_off)
    assert STEM not in json.dumps(runs["off"]["pass1"]), "premise: label off"
    parts = last_user(runs["on"]["pass1"])["content"]
    assert parts == [same_shape(own_off[0]), text_part(LABEL_15), *own_off[1:]], kinds(parts)
    (gen,) = step1_records(runs["on"]["rig"])
    check_request_key(gen.get("overview_frame"))
    assert "live_label" not in gen


async def test_history_image_allowed(tmp_path: Path) -> None:
    """H-T151-3 (approved: history images in earlier messages are allowed): turn 1's page fits (``fits``); turn 2 on
    the 1.525-screen page has turn 1's frame as a history image and one own frame G: it is ``shown``, the history
    image stays, G is replaced by the overview (premise: the off session's turn 2 carries ``[F0, G]``)."""

    async def drive(rig: Any, steps: tuple[Any, ...]) -> None:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        rig.browser.page_metrics = dict(DEALS)
        await next_turn(rig, 2, tsl.QUESTION, [answer(S3)])

    runs = {}
    for arm in ("off", "on"):
        browser = deals_browser(frames=(F0, G), metrics=FITS)
        async with rig_for(
            tmp_path / arm, browser, on=arm == "on", script=[answer("A shop. ")]
        ) as rig:
            await drive(rig, ())
            runs[arm] = (rig, pass1_body(rig, "t2"))
    off_t2 = runs["off"][1]
    assert image_urls(off_t2) == [builder_url(F0), builder_url(G)], (
        "premise: a history image, then G"
    )
    rig, t2 = runs["on"]
    assert image_urls(t2) == [builder_url(F0), ov_url()], (
        f"T-0151 H-T151-3: the history image stays and G is replaced; {image_urls(t2)}"
    )
    assert [i for i, p in enumerate(last_user(t2)["content"]) if p.get("type") == "image_url"] == [
        0
    ]
    assert LABEL_15 in json.dumps(last_user(t2)), "the label follows the overview"
    assert frame_key(rig, "t1") == counts(fits=1) and frame_key(rig, "t2") == counts(shown=1)


async def test_loop_base_clean_label_on_and_off(tmp_path: Path) -> None:
    """A2-4 (adversarial; reviewer mutation 11, the overview kept in ``loop_base``), with the label on and off: after
    the step-1 click, steps 2 and 3 carry no overview and no overview label; no prefill and no caption request carries
    it; the next turn's step 1 (history) carries none; only ``t1-gen`` records ``overview_frame``."""
    for name, feedback in (("on", "on"), ("off", label_off())):
        browser = deals_browser(frames=(F0, G))
        async with rig_for(
            tmp_path / name,
            browser,
            on=True,
            script=[answer(S1), tool(VALID_CLICK)],
            feedback=feedback,
        ) as rig:
            await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
            assert ov_in(pass1_body(rig)) == 1, "premise: step 1 carries the overview"
            for k in (2, 3):
                body = step_body(rig, k)
                assert ov_in(body) == 0 and OV_STEM not in json.dumps(body), (
                    f"T-0151 A2-4 ({name}): step {k} carries the overview; {kinds(last_user(body)['content'])}"
                )
            browser.page_metrics = dict(FITS)  # turn 2 fits, so any overview in it is turn 1's
            await next_turn(rig, 2, "and now", [answer("Still the deals. ")])
            t2 = pass1_body(rig, "t2")
            assert ov_in(t2) == 0 and OV_STEM not in json.dumps(t2), "A2-4: never in history"
            assert rig.llm.prefills, "premise: prefills were sent"
            for rid, p in rig.llm.prefills:
                assert ov_in(_wire(p)) == 0, f"A2-4: the prefill {rid} carries the overview"
            for rid, p, _ in rig.llm.generates:
                if is_caption_prompt(p):
                    assert ov_in(_wire(p)) == 0, f"A2-4: the caption request {rid} carries it"
        recs = ov_records(rig)
        assert [(d.get("phase"), d.get("request_id")) for d in recs] == [("generate", "t1-gen")], (
            f"A2-8 ({name}): only the step-1 request records the overview frame; "
            f"{[(d.get('phase'), d.get('request_id')) for d in recs]}"
        )


async def test_retry_and_repair_carry(tmp_path: Path) -> None:
    """A2-2/A2-4/A2-8: a 413 on the first step-1 attempt makes the retry carry the same overview and label (one image)
    with exactly one capture and ``shown`` 1, both records with the same ``overview_frame``; a repairable step-1 call's
    repair body carries it once (one image), its record carries the generate's key, and the repaired call is executed
    against the overview geometry."""
    browser = deals_browser()
    async with rig_for(
        tmp_path / "retry", browser, on=True, script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.fail_generates(http_error(413), None)
        await run_turn(rig, ([answer(S3)],))
        attempts = tsl.generates_for(rig, "t1-gen")
        assert len(attempts) == 2, f"premise: one over-long retry; {len(attempts)} attempts"
        for i, (_, p) in enumerate(attempts):
            body = _wire(p)
            assert image_urls(body) == [ov_url()], (i, kinds(last_user(body)["content"]))
            assert LABEL_15 in json.dumps(body) and b64(F0.jpeg) not in json.dumps(body), i
        assert len(captures(browser)) == 1, (
            f"A2-2: the retry re-uses the turn's result; {len(captures(browser))}"
        )
        assert frame_key(rig) == counts(shown=1), frame_key(rig)
    gen = step1_records(rig)
    assert len(gen) == 2 and gen[0].get("overview_frame") == gen[1].get("overview_frame"), gen
    check_request_key(gen[0].get("overview_frame"))

    browser = deals_browser()
    async with rig_for(
        tmp_path / "repair", browser, on=True, script=[answer(S1), tool(REPAIRABLE_CLICK)]
    ) as rig:
        rig.llm.script_continuation(tool(REPAIRED_CLICK))
        await run_turn(rig, ([answer(S3)],))
        reps = rig.llm.continuations()
        assert reps, "premise: a repair request was sent"
        body = _wire(reps[0][1])
        assert image_urls(body) == [ov_url()] and LABEL_15 in json.dumps(body), (
            f"T-0151 A2-4: the step-1 repair re-sends the overview; {image_urls(body)}"
        )
        clicks = [i for i in browser.invocations if i.call_id == "t1-call"]
        assert (
            len(clicks) == 1 and clicks[0].frame is not None and clicks[0].frame.phash == OV.phash
        ), clicks
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"]
        assert len(captures(browser)) == 1 and frame_key(rig) == counts(shown=1)
    gen = [d for d in requests(rig) if d.get("phase") == "generate"]
    rep = [d for d in requests(rig) if d.get("phase") == "tool_repair"]
    assert gen and rep, [d.get("phase") for d in requests(rig)]
    assert (
        "overview_frame" in gen[0] and rep[-1].get("overview_frame") == gen[0]["overview_frame"]
    ), (
        f"A2-8: the repair record carries the overview_frame key; got {rep[-1].get('overview_frame')}"
    )
    assert "live_label" not in rep[-1], (
        "A2-4: no T-0134 label on the repair of the overview request"
    )


# =====================================================================================================
# A2-3: the cap and the other outcomes
# =====================================================================================================


@pytest.mark.parametrize(
    "content,want",
    [(1600.0, "shown"), (1600.08, "too_tall"), (2240.0, "too_tall")],
    ids=["2.0", "2.0001", "2.8"],
)
async def test_cap(tmp_path: Path, content: float, want: str) -> None:
    """A2-3 / H-T151-4 (adversarial; reviewer mutation 12, the cap compared with ``<`` or removed): screens 2.0
    (1600/800) gives ``shown`` with the label at "2.0"; 2.0001 (1600.08/800) and 2.8 (2240/800) give ``too_tall``, a
    step 1 byte-identical to off (body and record), ``overview_frame_of`` never called and the picture read once
    through ``overview_of`` (discarded)."""
    metrics = {**DEALS, "content_height": content}
    runs = await off_and_on(tmp_path, lambda: deals_browser(metrics=metrics), script=[answer(S3)])
    on = runs["on"]
    browser = on["browser"]
    assert len(captures(browser)) == 1, "premise: captured"
    assert on["lines"][0]["extras"].get("overview_frame") == counts(**{want: 1})
    if want == "shown":
        parts = last_user(on["pass1"])["content"]
        assert parts[0]["image_url"]["url"] == ov_url() and parts[1] == text_part(LABEL_20), kinds(
            parts
        )
        (gen,) = step1_records(on["rig"])
        check_request_key(gen.get("overview_frame"), screens=2.0)
        return
    assert json.dumps(on["pass1"]) == json.dumps(runs["off"]["pass1"]), (
        "A2-3: too_tall step 1 differs from off"
    )
    assert scrub(step1_records(on["rig"])) == scrub(step1_records(runs["off"]["rig"]))
    assert reads(browser, "overview_frame_of") == [], (
        f"T-0151 A2-3: a too_tall picture was offered for registration; {browser.calls}"
    )
    assert len(reads(browser, "overview_of")) == 1, (
        f"A2-3: read through overview_of and discarded; {browser.calls}"
    )
    assert browser.overviews == {}, "the picture is discarded (read once)"


def _closing(b: FakeLiveBrowser) -> None:
    real_frame = b.frame

    async def frame_then_close() -> Frame:
        f = await real_frame()
        b.close_window()
        return f

    b.frame = frame_then_close  # type: ignore[method-assign]


def _raising(b: FakeLiveBrowser) -> None:
    real_execute = b.execute

    async def execute(inv: Any) -> Any:
        if inv.name == "page_overview":
            b.invocations.append(inv)
            raise RuntimeError("capture failed at http://127.0.0.1:9/secret?q=T0151-URL-SENTINEL")
        return await real_execute(inv)

    b.execute = execute  # type: ignore[method-assign]


def _stale_between(b: FakeLiveBrowser) -> None:
    """A capture, then another own frame becomes latest before the registration (A1-1: ``None``)."""
    real = b.perform_overview

    def perform_overview(inv: Any) -> Any:
        out = real(inv)
        b.land(live_frame("between"))
        return out

    b.perform_overview = perform_overview  # type: ignore[method-assign]


CASES: dict[
    str, tuple[Callable[[FakeLiveBrowser], None] | None, dict[str, Any], dict[str, int], bool]
] = {
    # name: (setup, attributes, counts, captured)
    "fits": (None, {"page_metrics": FITS}, counts(fits=1), True),
    "failed_moved": (None, {"overview_moves": True}, counts(failed=1), True),
    "failed_timeout": (None, {"overview_hang": True}, counts(failed=1), True),
    "failed_exception": (_raising, {}, counts(failed=1), True),
    "failed_none_registration": (None, {"overview_frame_none": True}, counts(failed=1), True),
    "failed_stale_registration": (_stale_between, {}, counts(failed=1), True),
    "refused_unavailable": (None, {"overview_available": False}, counts(refused=1), True),
    "refused_closed": (_closing, {}, counts(refused=1), True),
    "refused_blind": (None, {"frame_mode": "raise"}, counts(refused=1), False),
}


@pytest.mark.parametrize("case", list(CASES))
async def test_fits_failed_refused_identical(tmp_path: Path, case: str) -> None:
    """A2-2/A2-3: fits, an executor ``failed``, the timeout, an exception, a ``None`` registration (scripted, and by
    A1-1's stale rule), the executor's refusals and a blind turn each give their one count and a step 1 byte-identical
    to the off turn's (body and ``llm.request`` record, no ``overview_frame`` key); the blind turn makes no capture call
    and no read; the exception's text (a URL) reaches no log record or turn line."""
    setup, attrs, want, captured = CASES[case]

    def make() -> FakeLiveBrowser:
        b = deals_browser()
        for k, v in attrs.items():
            setattr(b, k, dict(v) if isinstance(v, dict) else v)
        if setup is not None:
            setup(b)
        return b

    runs = await off_and_on(tmp_path, make, script=[answer(S3)])
    on, off = runs["on"], runs["off"]
    browser = on["browser"]
    assert on["lines"][0]["extras"].get("overview_frame") == want, (
        case,
        on["lines"][0]["extras"].get("overview_frame"),
    )
    assert json.dumps(on["pass1"]) == json.dumps(off["pass1"]), (
        f"T-0151 A2-3 ({case}): step 1 differs from off"
    )
    assert scrub(step1_records(on["rig"])) == scrub(step1_records(off["rig"])), case
    assert not ov_records(on["rig"]), case
    if captured:
        assert len(captures(browser)) == 1, f"premise: captured; {browser.calls}"
    else:
        assert captures(browser) == [] and reads(browser, "overview_of") == [], browser.calls
        assert reads(browser, "overview_frame_of") == [], browser.calls
    if case == "failed_exception":
        assert "T0151-URL-SENTINEL" not in json.dumps(on["rig"].log_records)
        assert "T0151-URL-SENTINEL" not in json.dumps(on["lines"])


async def test_rider_and_two_frame_turns_refused(tmp_path: Path) -> None:
    """A2-2 / H-T151-3 (a multi-frame final message falls back): a rider turn (turn 2's onset frame has turn 1's
    bytes, so its final user message has no image) and a T-0059 two-frame turn each count ``refused`` 1, make no
    capture call and send step 1 identical to the off session's."""
    again = F0.model_copy(update={"ts_ms": 77_777})
    runs = {}
    for arm in ("off", "on"):
        browser = deals_browser(frames=(F0, again))
        async with rig_for(
            tmp_path / arm, browser, on=arm == "on", script=[answer("Deals. ")]
        ) as rig:
            await rig.ask("what is on the screen")
            await rig.turn_done()
            await next_turn(rig, 2, tsl.QUESTION, [answer(S3)])
            runs[arm] = (rig, browser, pass1_body(rig, "t2"), [json.loads(x) for x in rig.lines()])
    rig, browser, t2, lines = runs["on"]
    assert "image_url" not in [p["type"] for p in last_user(t2)["content"]], "premise: turn 2 rides"
    assert len(captures(browser)) == 1, (
        f"A2-2: only turn 1 captured; {[i.call_id for i in captures(browser)]}"
    )
    assert json.dumps(t2) == json.dumps(runs["off"][2]), "turn 2's step 1 equals the off session's"
    assert [x["extras"].get("overview_frame") for x in lines] == [
        counts(shown=1),
        counts(refused=1),
    ]

    two = await off_and_on(
        tmp_path / "two",
        lambda: deals_browser(frames=(F0, live_frame("scene-2"))),
        script=[answer(S3)],
        drive=_two_frame_drive,
    )
    assert len(image_urls(two["off"]["pass1"])) == 2, "premise: a two-frame turn"
    assert captures(two["on"]["browser"]) == [], (
        f"T-0151 A2-2: a multi-frame turn was captured; {two['on']['browser'].calls}"
    )
    assert json.dumps(two["on"]["pass1"]) == json.dumps(two["off"]["pass1"])
    assert two["on"]["lines"][0]["extras"].get("overview_frame") == counts(refused=1)


async def test_timeout_note(tmp_path: Path) -> None:
    """A2-2 (the same ``AUTO_OVERVIEW_TIMEOUT_MS`` bound and timeout note; reading R4): a capture that never returns
    is cancelled; one note with ``timed out`` and ``timeout_ms`` 2000; step 1 equals off; the turn completes."""
    limit = getattr(session_mod(), "AUTO_OVERVIEW_TIMEOUT_MS", None)
    assert limit == 2_000, limit

    def make() -> FakeLiveBrowser:
        b = deals_browser()
        b.overview_hang = True
        return b

    runs = await off_and_on(tmp_path, make, script=[answer(S3)])
    on = runs["on"]
    assert any(c[0] == "overview.cancelled" for c in on["browser"].calls), on["browser"].calls
    rig = on["rig"]
    notes = [
        d
        for d in rig.kind("note")
        if "timed out" in str(d.get("what", "")) and "live frame" not in str(d.get("what", ""))
    ]
    assert len(notes) == 1 and notes[0].get("timeout_ms") == limit, notes
    assert json.dumps(on["pass1"]) == json.dumps(runs["off"]["pass1"])
    assert rig.spoken_sources().replace(" ", "") == S3.replace(" ", ""), "the turn completed"


async def test_capture_after_frame_landed_and_no_overlap(tmp_path: Path) -> None:
    """A2-2 (T-0150 A1's moment and chaining): while the onset frame is still being taken nothing is captured; once it
    lands, one capture; a live snapshot requested through ``Session.snapshot_frame`` while the capture is held calls
    ``frame()`` only after it returned; the overview is shown."""
    browser = deals_browser(frames=(F0, live_frame("mid")))
    browser.frame_gate = asyncio.Event()
    browser.overview_gate = asyncio.Event()
    async with rig_for(tmp_path, browser, on=True, script=[answer(S3)]) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.advance(600)
        assert not browser.frames_returned, "premise: the onset frame is pending"
        assert captures(browser) == [], "A2-2: a capture before the turn's live frame landed"
        browser.frame_gate.set()
        await rig.until(
            lambda: any(c[0] == "overview.start" for c in browser.calls),
            "the capture never started",
            limit_ms=6_000,
        )
        await rig.session.snapshot_frame(rig.clock.now_ms)
        await rig.advance(500)
        assert [c[0] for c in browser.calls].count("frame") == 1, (
            f"A2-2: frame() ran while the capture was held; {browser.calls}"
        )
        browser.overview_gate.set()
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        assert [c[0] for c in browser.calls].count("frame") == 2, browser.calls
        assert image_urls(pass1_body(rig)) == [ov_url()]
        assert frame_key(rig) == counts(shown=1)


# =====================================================================================================
# A2-5: identity and freshness (B9)
# =====================================================================================================


def _spy_b9(session: Any, seen: list[dict[str, Any]], replays: dict[str, Any]) -> None:
    """Reading R3: wrap the session's B9 predicate. At each check it records the call; for a ``click_at`` it also
    re-asks the same predicate, with the same other arguments, for copies of the call (``replays``: name -> a function
    of the call giving the copy). Never invokes anything discovered by pattern: the name is the task's."""
    original = session._live_stale_frame

    def spy(*args: Any, **kwargs: Any) -> Any:
        found = [i for i, a in enumerate(args) if type(a).__name__ == "_ToolCall"]
        named = [k for k, v in kwargs.items() if type(v).__name__ == "_ToolCall"]
        assert len(found) + len(named) == 1, (args, kwargs)
        call = args[found[0]] if found else kwargs[named[0]]
        verdict = original(*args, **kwargs)
        entry: dict[str, Any] = {
            "call_id": call.call_id,
            "loop_step": call.loop_step,
            "frame_identity": call.frame_identity,
            "stale": verdict,
            "replays": {},
        }
        if call.name in ("click_at", "type_at"):
            for name, make in replays.items():
                copy = make(call)
                a2, k2 = list(args), dict(kwargs)
                if found:
                    a2[found[0]] = copy
                else:
                    k2[named[0]] = copy
                entry["replays"][name] = original(*a2, **k2)
        seen.append(entry)
        return verdict

    session._live_stale_frame = spy


async def test_b9_exemption_scoped(tmp_path: Path) -> None:
    """A2-5 (adversarial; reviewer mutation 10, the exemption not limited to step 1, to the turn's identity, or to an
    unchanged latest frame):

    - turn 1's step-1 call carries ``frame_identity`` = the overview's capture identity, is not stale and executes;
      ``_frame_geometry`` holds that identity with the executor's phash;
    - the same call at loop step 2 (reading R3) is stale;
    - turn 2 (another overview, shown): its step-1 call is exempt, but the same call carrying turn 1's overview
      identity is stale.
    """
    seen: list[dict[str, Any]] = []
    browser = deals_browser(frames=(F0, G), post=(F1, F2, live_frame("after-3")))
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        _spy_b9(
            rig.session,
            seen,
            {
                "step2": lambda c: dataclasses.replace(c, loop_step=2),
                "turn1_identity": lambda c: dataclasses.replace(c, frame_identity=OV_IDENTITY),
            },
        )
        await run_turn(rig, ([answer(S3)],))
        geometry = rig.session._frame_geometry.get(OV_IDENTITY)
        assert geometry is not None and geometry.phash == OV.phash, (
            f"T-0151 A2-5: the overview is registered under its capture identity; {geometry}"
        )
        browser.overview_image = OV_B
        await next_turn(
            rig, 2, "and the next one", [answer(S1), tool(VALID_CLICK)], ([answer(S3)],)
        )
        await tsl.until_steps(rig, 1, turn="t2", limit_ms=20_000, step=100)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
    t1 = [e for e in seen if e["call_id"] == "t1-call"]
    t2 = [e for e in seen if e["call_id"] == "t2-call"]
    assert t1 and t2, f"premise: both step-1 calls were checked; {seen}"
    assert t1[0]["frame_identity"] == OV_IDENTITY, (
        f"T-0151 A2-5: the step-1 call's frame_identity is the overview's; {t1[0]['frame_identity']}"
    )
    assert t1[0]["stale"] is False, "A2-5: B9 refused the step-1 overview call"
    assert t1[0]["replays"]["step2"] is True, "A2-5: the exemption is not limited to step 1"
    assert t2[0]["frame_identity"] == OV_B_IDENTITY and t2[0]["stale"] is False, t2[0]
    assert t2[0]["replays"]["turn1_identity"] is True, (
        "A2-5: the exemption is not limited to this turn's overview identity"
    )
    for t in ("t1-call", "t2-call"):
        assert [e.status for e in tsl.results_for(rig, t)] == ["executed"], t


async def test_b9_refuses_after_takeover_snapshot(tmp_path: Path) -> None:
    """A2-5 (adversarial; mutation 10's third shape): a snapshot lands (another own frame becomes ``latest_frame``)
    after the overview was registered and before the step-1 call: the call is refused with ``FRESHNESS_REASON`` before
    ``execute`` (nothing reaches the browser for it; the fake would have refused with its own stale reason)."""
    browser = deals_browser()
    real = browser.overview_frame_of

    def then_snapshot(key: Any) -> Any:
        out = real(key)
        browser.land(live_frame("takeover-snapshot"))
        return out

    browser.overview_frame_of = then_snapshot  # type: ignore[method-assign]
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await run_turn(rig, ([answer(S3)],))
        assert image_urls(pass1_body(rig)) == [ov_url()], "premise: the overview was shown"
        assert [i.call_id for i in browser.invocations if i.call_id == "t1-call"] == [], (
            "T-0151 A2-5: the call reached the browser after the latest frame changed"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["refused"]
        assert card(rig, "t1-call").reason == FRESHNESS
        assert f"reason:{Q}{FRESHNESS}{Q}" in step_tail(rig, 2), step_tail(rig, 2)[-300:]


async def test_step1_scroll_on_overview_allowed(tmp_path: Path) -> None:
    """Q-3 (default: allowed): a step-1 ``scroll`` on the overview request is executed (it uses no coordinates); step 2
    shows the live screen after it (F1, ``LU``), never the overview."""
    browser = deals_browser()
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(SCROLL)]) as rig:
        await run_turn(rig, ([answer(S3)],))
        assert image_urls(pass1_body(rig)) == [ov_url()], "premise: shown"
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"]
        s2 = step_body(rig, 2)
        assert image_urls(s2) == [builder_url(F1)] and ov_in(s2) == 0, image_urls(s2)
        assert label_after(s2, F1) == LU


# =====================================================================================================
# A2-6: step 2
# =====================================================================================================


@pytest.mark.parametrize("case", ["executed_same_frame", "refused"])
async def test_step2_live_frame_updated(tmp_path: Path, case: str) -> None:
    """A2-6 (adversarial; reviewer mutation 14, ``updated`` computed against ``loop_base`` as at BASE): after a step-1
    call that the fake executes with an unchanged ``frame_after`` (the post-action capture is F0 again), and after a
    refused step-1 call, step 2 shows the live frame F0 (no overview), the payload says ``"updated"`` and the label is
    ``LU``. Premise: the same scripted turn with the overview frame off says ``"unchanged"``. With the towel click the
    step-1 action string reaches the continuation exactly as the executor returned it."""

    def make() -> FakeLiveBrowser:
        b = deals_browser(frames=(F0,), post=(F0, F2))
        if case == "refused":
            b.outcome("t1-call", lambda i: refused(i, "bad coordinates"))
        return b

    runs = await off_and_on(tmp_path, make, script=[answer(S1), tool(TOWEL)], steps=([answer(S3)],))
    off_tail = step_tail(runs["off"]["rig"], 2)
    assert f"{Q}unchanged{Q}" in off_tail, f"premise: off says unchanged; {off_tail[-300:]}"
    rig = runs["on"]["rig"]
    s2 = step_body(rig, 2)
    assert image_urls(s2) == [builder_url(F0)] and ov_in(s2) == 0, (
        f"T-0151 A2-6: step 2 shows the live frame; {image_urls(s2)}"
    )
    tail = step_tail(rig, 2)
    assert f"{Q}updated{Q}" in tail and f"{Q}unchanged{Q}" not in tail, (
        f"T-0151 A2-6: the settled step-1 payload says updated; {tail[-300:]}"
    )
    assert label_after(s2, F0) == LU, "A2-6: the label opening is LABEL_UPDATED"
    if case == "executed_same_frame":
        assert TOWEL_ACTION in tail, f"A2-6: the step-1 action string; {tail[-300:]}"


async def test_updated_override_is_step2_only(tmp_path: Path) -> None:
    """A2-6 (review A; the mutant widens the override from step 2 to every step): a three-step turn with the overview
    shown at step 1. The step-1 towel click captures F1 (step 2 shows F1, ``updated``); the step-2 click on F1 leaves
    the screen unchanged (its post-action capture is F1 again), so the step-3 request's payload says
    ``screenshot: "unchanged"`` and its label is ``LN`` (``LABEL_UNCHANGED``) after F1, exactly as at BASE."""
    browser = deals_browser(frames=(F0,), post=(F1, F1))
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(TOWEL)]) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert image_urls(pass1_body(rig)) == [ov_url()], (
            "premise: the overview was shown at step 1"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"], "premise"
        s2, s3 = step_body(rig, 2), step_body(rig, 3)
        assert label_after(s2, F1) == LU, "premise: step 2 shows F1 after the step-1 action"
        tail3 = step_tail(rig, 3)
        # the step-3 continuation embeds both settled payloads in order: step 1's "updated", then step 2's
        up, un = f"{Q}updated{Q}", f"{Q}unchanged{Q}"
        assert (
            tail3.count(up) == 1 and tail3.count(un) == 1 and tail3.index(up) < tail3.index(un)
        ), f"T-0151 A2-6: step 2's settled payload says unchanged; {tail3[-400:]}"
        assert image_urls(s3) == [builder_url(F1)] and ov_in(s3) == 0, image_urls(s3)
        assert label_after(s3, F1) == LN, "A2-6: the step-3 label opens with LABEL_UNCHANGED"


# =====================================================================================================
# A2-7: never spoken, never stored
# =====================================================================================================


async def test_never_registered_in_turn_frames_or_stored(tmp_path: Path) -> None:
    """A2-5/A2-7: across an overview turn (towel click, report) and a second turn, the overview's phash and capture
    identity are in no turn frames (turn line ``frames``), history or ``Episode``; the label (stem, "1.5 screens"),
    the pixels, the capture's action string and the towel action string reach no ``TtsSpeak``, history, HUD message,
    ``Episode`` or ``machine.events()``; the voiced text is exactly the model's answers."""
    browser = deals_browser(frames=(F0, G))
    async with rig_for(tmp_path, browser, on=True, script=[answer(S1), tool(TOWEL)]) as rig:
        await run_turn(rig, ([answer(S3)],))
        await next_turn(rig, 2, "and now", [answer("Still the deals. ")])
        sent = json.dumps(all_bodies(rig))
        assert OV_STEM in sent and b64(OV.jpeg) in sent, "premise: the overview was sent"
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"]
        for t in ("t1", "t2"):
            frames = list(rig.record(t).extras.get("frames", []))
            line = next(json.loads(x) for x in rig.lines() if json.loads(x)["turn_id"] == t)
            frames += list(line.get("frames", []) or []) + list(
                line.get("extras", {}).get("frames", []) or []
            )
            assert OV.phash not in frames and OV_IDENTITY not in frames, (t, frames)
        needles = (
            OV_STEM,
            "1.5 screens",
            b64(OV.jpeg),
            OV.phash,
            OV_IDENTITY,
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
                assert n not in text, f"T-0151 A2-7: {n[:40]!r} reached {where}"
        voiced = "".join(c.source for c in rig.machine.commands(C.TtsSpeak))
        assert voiced.replace(" ", "") == (S1 + S3 + "Still the deals. ").replace(" ", ""), (
            f"A2-7: no narration is added; voiced {voiced!r}"
        )


async def test_outside_tool_path(tmp_path: Path) -> None:
    """A2-2 (outside the model's tool path): the capture produces no tool card, no ``tool.*`` record and no machine
    event (the call ids of the tool records, the machine's event and command types equal the off turn's); the step-1
    body's ``tools`` are v1's declarations exactly."""
    runs = await off_and_on(
        tmp_path, deals_browser, script=[answer(S1), tool(VALID_CLICK)], steps=([answer(S3)],)
    )
    off, on = runs["off"], runs["on"]
    assert ov_in(on["pass1"]) == 1, "premise: shown"
    v1 = native.tools_field(tools.load())
    assert on["pass1"].get("tools") == off["pass1"].get("tools") == json.loads(json.dumps(list(v1)))
    ron, roff = on["rig"], off["rig"]
    assert [m.call_id for m in ron.tool_calls()] == [m.call_id for m in roff.tool_calls()]
    assert "page_overview" not in tsl.published_json(ron)

    def tool_records(run: dict[str, Any]) -> list[tuple[str, Any]]:
        return [
            (r["kind"], r["data"].get("call_id"))
            for r in run["rig"].log_records
            if str(r["kind"]).startswith("tool.")
        ]

    assert tool_records(on) == tool_records(off), "A2-2: a tool.* record for the capture"
    assert [type(e).__name__ for e in ron.machine.events()] == [
        type(e).__name__ for e in roff.machine.events()
    ]
    assert [type(c).__name__ for c in ron.machine.commands()] == [
        type(c).__name__ for c in roff.machine.commands()
    ]


# =====================================================================================================
# A2-8: logging
# =====================================================================================================

OUTCOMES: dict[str, tuple[dict[str, Any], dict[str, int]]] = {
    "shown": ({}, counts(shown=1)),
    "fits": ({"page_metrics": FITS}, counts(fits=1)),
    "too_tall": ({"page_metrics": {**DEALS, "content_height": 2240.0}}, counts(too_tall=1)),
    "failed": ({"overview_moves": True}, counts(failed=1)),
    "refused": ({"overview_available": False}, counts(refused=1)),
}


@pytest.mark.parametrize("case", list(OUTCOMES))
async def test_turn_line_counts(tmp_path: Path, case: str) -> None:
    """A2-8: each scripted outcome gives the key with exactly one count at 1 (written out per case), all five keys; a
    second turn of the same session gets its own counts (one at 1, not accumulated); no key when off; no new log kind
    and no new span against the off session."""
    attrs, want = OUTCOMES[case]
    runs: dict[str, Any] = {}
    for arm in ("off", "on"):
        browser = deals_browser(frames=(F0, G))
        for k, v in attrs.items():
            setattr(browser, k, dict(v) if isinstance(v, dict) else v)
        async with rig_for(tmp_path / arm, browser, on=arm == "on", script=[answer(S3)]) as rig:
            await run_turn(rig, ())
            await next_turn(rig, 2, "and now", [answer("Again. ")])
            runs[arm] = (rig, [json.loads(x) for x in rig.lines()])
    rig, lines = runs["on"]
    assert lines[0]["extras"].get("overview_frame") == want, (
        case,
        lines[0]["extras"].get("overview_frame"),
    )
    second = lines[1]["extras"].get("overview_frame")
    assert isinstance(second, dict) and set(second) == set(KEYS) and sum(second.values()) == 1, (
        second
    )
    off_rig, off_lines = runs["off"]
    assert all("overview_frame" not in x["extras"] for x in off_lines), "A2-8: no key when off"
    assert not [d for d in requests(off_rig) if "overview_frame" in d]
    on_kinds = {r["kind"] for r in rig.log_records}
    off_kinds = {r["kind"] for r in off_rig.log_records}
    assert on_kinds <= off_kinds, f"A2-8: no new log kind; {sorted(on_kinds - off_kinds)}"
    for a, b in zip(lines, off_lines, strict=True):
        assert set(a.get("spans", {})) == set(b.get("spans", {})), "A2-8: no new span"


async def test_turn_line_zero_without_step1(tmp_path: Path) -> None:
    """A2-8 (reading R2): a turn taken over while its generate still waits for the onset frame (step 1 never built)
    gives a line with all five counts 0 and no capture; every live turn line carries the key."""
    browser = deals_browser(frames=(F0, G))
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
    assert [x for x in lines if x["extras"].get("overview_frame") == counts()], (
        f"A2-8: a turn that never sent step 1 counts 0; {[x['extras'].get('overview_frame') for x in lines]}"
    )
    assert all("overview_frame" in x["extras"] for x in lines), (
        "A2-8: every live turn line carries the key"
    )


async def test_tool_records(tmp_path: Path) -> None:
    """A2-8 (readings R5, R6): the step-1 towel call's last ``tool.call`` record has ``"frame_kind": "overview"``; the
    step-2 call's (on the live frame) has none, and the off session has none at all. The executed record's
    ``resolved`` is the executor's, by hand from the fake (A1-4: x 205, viewport y 721, page y 1141, scrolled 420,
    ``frame_after`` F1), and ``tool.result`` carries the action string as at BASE."""
    runs = await off_and_on(
        tmp_path,
        deals_browser,
        script=[answer(S1), tool(TOWEL)],
        steps=([answer(S2), tool(CLICK2)], [answer(S3)]),
    )

    def calls_of(run: dict[str, Any], cid: str) -> list[dict[str, Any]]:
        return [d for d in run["rig"].kind("tool.call") if d.get("call_id") == cid]

    on = runs["on"]
    t1 = calls_of(on, "t1-call")
    assert t1 and t1[-1].get("frame_kind") == "overview", (
        f"T-0151 A2-8: frame_kind on the step-1 call's record; {[d.get('frame_kind') for d in t1]}"
    )
    executed = [d for d in t1 if d.get("status") == "executed"]
    assert executed and executed[-1]["resolved"] == {
        "x_px": 205,
        "y_px": 721,
        "page_y_px": 1141,
        "scrolled_px": 420,
        "frame_after": F1.phash,
    }, executed
    assert all("frame_kind" not in d for d in calls_of(on, "t1-call-s2")), calls_of(
        on, "t1-call-s2"
    )
    assert "frame_kind" not in json.dumps(runs["off"]["rig"].kind("tool.call")), (
        "A2-8: no key when off"
    )
    results = [d for d in on["rig"].kind("tool.result") if d.get("call_id") == "t1-call"]
    assert results and TOWEL_ACTION in json.dumps(results[-1]), results


# =====================================================================================================
# A2-1: refusals; constants
# =====================================================================================================


async def test_session_refusals(tmp_path: Path) -> None:
    """A2-1: ``Session(overview_frame=True)`` raises ``ValueError`` with a ``dry_run`` kit, with
    ``auto_overview=True``, with a live kit offering ``page_overview`` and with ``reground=True``. Too-wide companion:
    the live v1 kit alone is accepted."""
    from tests.unit.test_t0140_session import overview_kit

    browser = deals_browser()
    with pytest.raises(ValueError):
        async with rig_for(
            tmp_path / "dry", browser, on=True, script=[], feedback=None, kit=T.make_kit()
        ):
            pass
    with pytest.raises(ValueError):
        async with rig_for(
            tmp_path / "auto", browser, on=True, script=[], extra={"auto_overview": True}
        ):
            pass
    with pytest.raises(ValueError):
        async with rig_for(tmp_path / "ov", browser, on=True, script=[], kit=overview_kit(browser)):
            pass
    with pytest.raises(ValueError):
        async with rig_for(tmp_path / "rg", browser, on=True, script=[], extra={"reground": True}):
            pass
    async with rig_for(tmp_path / "ok", deals_browser(), on=True, script=[]) as rig:
        assert rig.session is not None


def test_constants() -> None:
    """A2-3/A2-4: ``OVERVIEW_FRAME_MAX_SCREENS == 2.0`` and ``OVERVIEW_FRAME_LABEL`` is the criterion's template
    exactly (H-T151-4, H-T151-5)."""
    s = session_mod()
    assert getattr(s, "OVERVIEW_FRAME_MAX_SCREENS", None) == 2.0
    assert getattr(s, "OVERVIEW_FRAME_LABEL", None) == LABEL_TEMPLATE
    assert (
        LABEL_TEMPLATE.format(screens=1.525) == LABEL_15
        and LABEL_TEMPLATE.format(screens=2.0) == LABEL_20
    )


def test_label_by_hand() -> None:
    """Green at BASE (the oracle): the hand-typed renderings are the template's at 1.525 and 2.0 screens."""
    assert LABEL_TEMPLATE.format(screens=1.525) == LABEL_15
    assert LABEL_TEMPLATE.format(screens=2.0) == LABEL_20


def test_live_kit_is_v1() -> None:
    """Green at BASE (A2-9 "tools stay v1"): the live kit the rows run on offers v1's tools."""
    kit = live_kit(FakeLiveBrowser())
    assert kit.tools == tuple(native.tools_field(tools.load()))
    assert kit.tools_version == V1_SHA, kit.tools_version
