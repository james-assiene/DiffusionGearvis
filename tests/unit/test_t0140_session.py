"""T-0140 group A2 (session side): A6, A7, A8, A9, A10 over the fakes (docs/tasks/T-0140.md).

**The surface these rows pin** (the task's names; where it is silent the reading is stated here and in the report):

- ``ToolKit`` gains exactly one optional field typed ``ToolSet | None`` with default ``None`` (v1); its name is the
  implementer's, found by its annotation (``kit_set_field``). An *overview kit* (``overview_kit``) is the live kit with
  ``tools``/``tools_version`` from ``tools_v1_overview.json`` and that field set to the loaded overview ``ToolSet``
  (what A5's ``load_tool_kit`` builds when on).
- A6: every pass-1 ``tool`` commit is judged with ``live.classify(text, tool_set=<the kit's set>)``.
- A7: ``loop.place_overview(prompt, frame, text) -> Prompt`` inserts ``ImagePart(<frame's data URL>)`` and
  ``TextPart(text)`` immediately before the newest ``ImagePart`` of the final user message (the current frame,
  wherever ``show_current_image`` put it). The step request right after an executed ``page_overview`` that returned an
  image carries it once; on the wire that is an ``image_url`` part whose URL is the builder's data URL of the overview
  frame (``builder_url``, T-0111g's oracle), then a text part ``OVERVIEW_LABEL``, then the current frame.
- **Reading (A7, O-A1-6):** an overview takes no post-action screen frame, and A7 says the settled step's payload says
  ``"unchanged"`` *unconditionally*; so the step after an overview shows the same current frame as the step before it
  (a fresh ``frame()`` whose pixels differ would say ``"updated"``). The fake's fresh frames always differ, so a session
  that takes one fails ``test_screenshot_unchanged_after_overview``.
- A8: the session refuses before dispatch (``refused``, the fixed reason, no ``execute`` reaches the browser and no
  ``overview_of`` is read) on a blind turn (``no screen this turn``) and when the step request would carry more than
  ``session.prompts.budget.max_images`` images with the overview (``no image slot for an overview``). The spec's
  ``PromptBudget`` is ``prompt_builder.Budget`` (no class of that name exists).
- A10: ``TurnRecord.extras["page_overview"] == {"calls", "shown", "fits", "refused", "failed"}`` on every turn line of a
  session whose kit offers the tool (zeros on a turn without a call); ``llm.request``'s ``overview`` key ``{"phash",
  "screens", "clipped", "width", "height"}`` on the one request that carries the image. ``calls`` counts every
  ``page_overview`` call the judge accepted; ``refused`` the session's A8 refusals and the executor's own; a
  judge-refused (schema) call counts in neither (O-A2 Q4).
- O-A2 rulings applied: Q1 the HUD never carries the overview image or ``OVERVIEW_LABEL`` (the card may show the action
  string); Q2 no fresh frame after an executed overview; Q3 a call whose final user message has no current-frame image
  (a rider turn's step 1) is refused with ``no screen this turn``; Q11 "never in a continuation" holds by construction.

Activated (no ``pending``): the A2 rows are red until A2 is implemented; ``test_v1_kit_refuses_overview_call`` (A8's
last sentence), ``test_judge_reads_given_set_bounds_and_keys`` and ``test_executor_page_size_unreadable`` (O-A1-11's
carried rows; A1's code) are green at BASE_A2 52d2452. Checked against a throwaway scratch implementation: every row
green, and reviewer mutations 1, 2, 4 each turn rows red (the test-author report).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on, ReGround off unless a row says so, T-0134c's label on; n = 1, deterministic.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import types
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import Frame, Source
from local_astra.toolcall import live, loop, native, tools
from tests.fakes.llm import is_caption_prompt
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    MAX_OVERVIEW_SCREENS,
    OVERVIEW_FITS,
    OVERVIEW_MOVED,
    OVERVIEW_UNAVAILABLE,
    FakeLiveBrowser,
    b64,
    builder_url,
    data_url,
    feedback_rig,
    image_urls,
    jpeg_frame,
    live_feedback,
    live_frame,
    live_kit,
    overview_action,
    session_mod,
)
from tests.unit._t0177_pins import published_json_outside_frames  # T-0177 B-P2
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_live import (
    CLICK2,
    _wire,
    card,
    executed,
    last_user,
    pass1_body,
    status,
    step_body,
    step_tail,
    want_tail,
)
from tests.unit.test_t0134_label import L1, LN, LU, all_bodies, next_turn

Q = '<|"|>'
OV_CALL = "<|tool_call>call:page_overview{}<tool_call|>"
CLICK3 = tsl.CLICK3
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
#: A8's fixed reasons, typed from the criterion
NO_SCREEN = "no screen this turn"
NO_SLOT = "no image slot for an overview"
#: A8: the v1 judge's reason for a name it does not know (BASE's verdict; ``test_t0140_judge``)
NO_TOOL_NAME = "no tool name"
#: the stem of every overview label (A7); a substring of no other text of the rigs
OV_STEM = "Whole page, scaled to fit one screen"
V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
A10_KEYS = ("calls", "shown", "fits", "refused", "failed")


def overview_label(screens: float, clipped: bool) -> str:
    """A7's ``OVERVIEW_LABEL``, by the criterion's template (``{C}`` as A3's)."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return (
        f"[Whole page, scaled to fit one screen: about {screens:.1f} screens tall{c}. "
        "The next image is the browser screen.]"
    )


def counts(**kw: int) -> dict[str, int]:
    return {k: int(kw.get(k, 0)) for k in A10_KEYS}


#: the scaled overview the executor keeps: 512x800 is a 2.5-screen page scaled by 0.4 (A1's executor rows), but the
#: session only carries it, so its bytes are a sentinel (unlike every own frame's)
OV = Frame(
    source=Source.WINDOW,
    ts_ms=0,
    width=320,
    height=800,
    jpeg=b"\xff\xd8" + b"OVERVIEW-PIXELS-T0140|" * 8,
    phash="0e0e0e0e0e0e0e0e",
)
#: 7440 CSS px over an 800 px viewport: 9.3 screens, clipped at 4 (by hand)
TALL = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 7440.0}
#: 2000 CSS px: 2.5 screens, not clipped
MID = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 2000.0}
#: 840 = 1.05 x 800: fits (A3's boundary)
FITS_PAGE = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 840.0}

F0 = live_frame("onset")
F1 = live_frame("after-1")
F2 = live_frame("after-2")


# --------------------------------------------------------------------------------------------- seams


def overview_set() -> Any:
    p = getattr(tools, "TOOLS_V1_OVERVIEW_PATH", None)
    assert p is not None and Path(p).is_file(), "T-0140 A1: tools_v1_overview.json is missing"
    return tools.load(p)


def kit_set_field() -> str:
    """A6: the one ``ToolKit`` field annotated ``ToolSet | None`` with default ``None`` (its name is free)."""
    cls = session_mod().ToolKit
    found = [f for f in dataclasses.fields(cls) if "ToolSet" in str(f.type) and f.default is None]
    assert len(found) == 1, (
        "T-0140 A6: ToolKit gains one optional field `<name>: ToolSet | None = None` (None means v1); fields "
        f"{[(f.name, str(f.type)) for f in dataclasses.fields(cls)]}"
    )
    return found[0].name


def overview_kit(browser: Any) -> Any:
    """The live kit over ``browser`` offering ``page_overview`` (A5's kit when on)."""
    ov = overview_set()
    return dataclasses.replace(
        live_kit(browser),
        tools=tuple(native.tools_field(ov)),
        tools_version=ov.version,
        **{kit_set_field(): ov},
    )


def tall_browser(
    frames: Any = (F0,), post: Any = (F1, F2), metrics: dict[str, float] | None = None
) -> FakeLiveBrowser:
    browser = FakeLiveBrowser(list(frames), post_frames=list(post), echo="model")
    browser.overview_image = OV
    browser.page_metrics = dict(metrics if metrics is not None else TALL)
    return browser


def label_on() -> Any:
    """T-0134c's live default: the screen label, no extent."""
    return live_feedback(screen_label=True, page_extent=False)


def place_overview() -> Any:
    fn = getattr(loop, "place_overview", None)
    assert callable(fn), "T-0140 A7: loop.place_overview(prompt, frame, text) does not exist yet"
    return fn


def overview_invocations(browser: FakeLiveBrowser) -> list[Any]:
    return [i for i in browser.invocations if i.name == "page_overview"]


def ov_url() -> str:
    return builder_url(OV)


def ov_in(body: dict[str, Any]) -> int:
    """How many times the overview's pixels appear in ``body`` (as a data URL or raw base64)."""
    return json.dumps(body).count(b64(OV.jpeg))


def kinds(parts: list[dict[str, Any]]) -> list[str]:
    out = []
    for p in parts:
        if p.get("type") == "image_url":
            url = p["image_url"]["url"]
            out.append("image:OV" if url == ov_url() else "image")
        else:
            out.append("text:" + str(p.get("text", ""))[:48])
    return out


async def run_turn(rig: Any, steps: tuple[Any, ...], question: str = tsl.QUESTION) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(question)
    await tsl.until_steps(rig, len(steps), limit_ms=20_000, step=100)
    await rig.turn_done(limit_ms=20_000)
    await rig.advance(300)


def ov_records(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if "overview" in d]


# =====================================================================================================
# A7: where the picture goes (the pure helper)
# =====================================================================================================


def _prompt(*users: tuple[Any, ...]) -> Prompt:
    msgs = [Message(role="system", content=(TextPart(text="sys"),))]
    for i, parts in enumerate(users):
        msgs.append(Message(role="user", content=tuple(parts)))
        if i < len(users) - 1:
            msgs.append(Message(role="assistant", content=(TextPart(text=f"said {i}"),)))
    return Prompt(messages=tuple(msgs), prefix_len=0)


def _img(f: Frame) -> ImagePart:
    return ImagePart(url=data_url(f.jpeg), phash=f.phash)


def test_place_overview_pure() -> None:
    """A7 (reviewer mutation 1, the overview after the current frame): ``place_overview`` puts ``ImagePart(<the
    overview's data URL>)`` and ``TextPart(text)`` immediately before the newest image of the final user message,
    whatever precedes it (a frame label, earlier frames, a screen label after it); a history image and every other
    message are untouched; the input prompt is unchanged. Expected parts written out by hand."""
    fn = place_overview()
    hist, a, b = live_frame("hist"), live_frame("a"), live_frame("b")
    text = overview_label(9.3, True)
    p = _prompt(
        (TextPart(text="earlier"), _img(hist)),
        (_img(a), TextPart(text="lbl-2"), _img(b), TextPart(text=LN), TextPart(text="q")),
    )
    before = p.model_dump()
    got = fn(p, OV, text)
    assert p.model_dump() == before, "A7: the helper is pure (the input is not changed)"
    users = [m for m in got.messages if m.role == "user"]
    final = list(users[-1].content)
    assert [type(x).__name__ for x in final] == [
        "ImagePart",
        "TextPart",
        "ImagePart",
        "TextPart",
        "ImagePart",
        "TextPart",
        "TextPart",
    ], final
    assert final[0] == _img(a) and final[1] == TextPart(text="lbl-2"), final
    assert isinstance(final[2], ImagePart) and final[2].url == data_url(OV.jpeg), (
        "A7: the overview's ImagePart (the builder's encoding of its JPEG)"
    )
    assert final[3] == TextPart(text=text), final
    assert final[4] == _img(b), (
        "A7: the current frame follows, unchanged, and stays the newest image"
    )
    assert final[5:] == [TextPart(text=LN), TextPart(text="q")], final
    assert got.messages[:-1] == p.messages[:-1], (
        "A7: every earlier message (the history image) is untouched"
    )
    # the newest image of the whole prompt is still the current frame (``_sent_identity``'s reading)
    newest = [x for m in got.messages for x in m.content if isinstance(x, ImagePart)][-1]
    assert newest == _img(b)
    # a single-frame turn: [F, q] -> [OV, label, F, q]
    one = fn(_prompt((_img(a), TextPart(text="q"))), OV, text)
    assert [type(x).__name__ for x in one.messages[-1].content] == [
        "ImagePart",
        "TextPart",
        "ImagePart",
        "TextPart",
    ]
    assert one.messages[-1].content[2] == _img(a)


# =====================================================================================================
# A7: where the picture goes (the session)
# =====================================================================================================


async def test_overview_placed_before_current_frame(tmp_path: Path) -> None:
    """A7 (reviewer mutation 1): step 1 calls ``page_overview`` on a 9.3-screen page; the step-2 request's final user
    message is ``[overview image, OVERVIEW_LABEL, current frame (F0), LN, <step 1's own parts>]`` (the label clipped,
    by hand), with exactly one overview image in the body; the current frame stays the newest image, so step 2's click
    resolves against F0 (``_sent_identity``) and executes."""
    browser = tall_browser()
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert [i.call_id for i in overview_invocations(browser)] == ["t1-call"], (
            "premise: the overview was executed once"
        )
        p1, s2 = pass1_body(rig), step_body(rig, 2)
        assert ov_in(p1) == 0, "the step-1 request carries no overview (it was not taken yet)"
        parts = last_user(s2)["content"]
        own = last_user(p1)["content"]
        assert own[0]["type"] == "image_url" and own[0]["image_url"]["url"] == builder_url(F0), (
            f"premise: step 1 shows F0 first; {kinds(own)}"
        )
        assert own[1] == {"type": "text", "text": L1}, f"premise: step 1's label; {kinds(own)}"
        want = [
            {"type": "image_url", "url": ov_url()},
            {"type": "text", "text": overview_label(9.3, True)},
            {"type": "image_url", "url": builder_url(F0)},
            {"type": "text", "text": LN},
        ]
        got = [
            {"type": "image_url", "url": p["image_url"]["url"]}
            if p.get("type") == "image_url"
            else {"type": "text", "text": p.get("text")}
            for p in parts[:4]
        ]
        assert got == want, f"T-0140 A7: step 2's final user message; parts {kinds(parts)}"
        assert parts[4:] == own[2:], "A7: the message's own parts follow unchanged"
        assert image_urls(s2) == [ov_url(), builder_url(F0)], (
            f"A7: one overview image, before the current frame; {len(image_urls(s2))} images"
        )
        assert ov_in(s2) == 1
        inv = [i for i in browser.invocations if i.call_id == "t1-call-s2"]
        assert len(inv) == 1 and inv[0].frame is not None and inv[0].frame.phash == F0.phash, (
            "A7: the current frame stays the newest image, so the click resolves against F0, never the overview"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"]


async def test_overview_one_step_only(tmp_path: Path) -> None:
    """A7 (reviewer mutation 2, the overview kept in later steps): step 3 (after the click) carries no overview and no
    overview label; nor does any prefill, caption request or repair continuation; nor does the next turn's request
    (history), the HUD, an ``Episode`` or a turn line. Exactly one ``llm.request`` record has the ``overview`` key."""
    browser = tall_browser(frames=(F0, F2))
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert ov_in(step_body(rig, 2)) == 1, "premise: step 2 carries the overview"
        s3 = step_body(rig, 3)
        assert ov_in(s3) == 0 and OV_STEM not in json.dumps(s3), (
            f"T-0140 A7: step 3 still carries the overview; {kinds(last_user(s3)['content'])}"
        )
        assert image_urls(s3) == [builder_url(F1)], "step 3 shows the post-click frame only"
        await next_turn(rig, 2, "and now", [answer("Still the shop. ")])
        t2 = pass1_body(rig, "t2")
        assert ov_in(t2) == 0 and OV_STEM not in json.dumps(t2), "A7: never in history"
        assert rig.llm.prefills, "premise: prefills were sent"
        for rid, p in rig.llm.prefills:
            assert ov_in(_wire(p)) == 0, f"A7: the prefill {rid} carries the overview"
        for rid, p, _ in rig.llm.generates:
            if is_caption_prompt(p):
                assert ov_in(_wire(p)) == 0, f"A7: the caption request {rid} carries the overview"
        # T-0115 Q5 ("never in a continuation", i.e. a repair): holds by construction, O-A2 Q11 (repairs are
        # step 1 only, the overview first appears at step 2); ``rig.llm.continuations()`` also lists the step
        # requests, so it is not the place to look
        # T-0177 B-P2: "never on the HUD" reads "never in any message other than astra.vision.frame" (H-T177-2)
        hud = published_json_outside_frames(rig.published.messages)
        assert b64(OV.jpeg) not in hud and OV_STEM not in hud, (
            "A7 (T-0177 B-P2): never on the HUD outside astra.vision.frame"
        )
        episodes = json.dumps([e.model_dump(mode="json") for e in rig.memory.episodes.values()])
        assert b64(OV.jpeg) not in episodes and OV.phash not in episodes, "A7: never in an Episode"
    recs = ov_records(rig)
    assert [(d.get("phase"), d.get("step")) for d in recs] == [("tool_step", 2)], (
        f"A7/A10: exactly the step-2 request records an overview; {[(d.get('phase'), d.get('step')) for d in recs]}"
    )


async def test_no_overview_in_reground_pass2(tmp_path: Path) -> None:
    """A7: with ReGround on, a click at the step right after the overview runs pass 2; the pass-2 request carries no
    overview image and no overview label (its prompt is the step's, without the overview), while the step request
    it came from does (premise)."""
    on0, on1 = jpeg_frame("onset"), jpeg_frame("after-1")
    browser = tall_browser(frames=(on0,), post=(on1,))
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on(), "reground": True},
        reground_llm=True,
        kit=overview_kit(browser),
    ) as rig:
        llm = rig.llm
        base_is_pass2 = llm.is_pass2

        def is_pass2(self: Any, prompt: Any) -> bool:
            # the overview is a known image too: only the crop marks a pass-2 request
            urls = {p.url for m in prompt.messages for p in m.content if isinstance(p, ImagePart)}
            if ov_url() in urls:
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
            return bool(base_is_pass2(prompt))

        llm.is_pass2 = types.MethodType(is_pass2, llm)
        llm.script_pass2([answer(""), tool("<|tool_call>call:click_at{x:130,y:650}<tool_call|>")])
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert ov_in(step_body(rig, 2)) == 1, "premise: step 2 carries the overview"
        assert llm.pass2_prompts, "premise: pass 2 ran for step 2's click"
        for rid, p in llm.pass2_prompts:
            body = _wire(p)
            assert ov_in(body) == 0 and OV_STEM not in json.dumps(body), (
                f"T-0140 A7: the pass-2 request {rid} carries the overview"
            )


async def test_history_rider_branch(tmp_path: Path) -> None:
    """A7 (adversarial; the mutant places the overview before ``show_current_image``, so the swap replaces the
    overview instead of the frame): turn 2's onset frame has turn 1's bytes, so it rides on the history image. Step 1
    clicks (post-action frame F1, placed first by ``show_current_image``), step 2 calls ``page_overview``, step 3
    clicks again, step 4 reports. Step 3's final user message is ``[overview, OVERVIEW_LABEL, F1, LN, <own parts>]``
    and its click resolves against F1; step 4's is ``[F2, LU, <own parts>]`` with no overview."""
    again = F0.model_copy(update={"ts_ms": 77_777})
    browser = tall_browser(frames=(F0, again), post=(F1, F2), metrics=MID)
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer("A shop. ")],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        await next_turn(
            rig,
            2,
            tsl.QUESTION,
            [answer(S1), tool(VALID_CLICK)],
            ([answer(S2), tool(OV_CALL)], [answer(S2), tool(CLICK3)], [answer(S3)]),
        )
        p1 = pass1_body(rig, "t2")
        own = last_user(p1)["content"]
        assert "image_url" not in [p["type"] for p in own], "premise: turn 2 rides on history"
        s2 = last_user(step_body(rig, 2, "t2"))["content"]
        assert s2[0]["image_url"]["url"] == builder_url(F1) and s2[1] == {
            "type": "text",
            "text": LU,
        }, f"premise: show_current_image placed F1 first; {kinds(s2)}"
        s3 = last_user(step_body(rig, 3, "t2"))["content"]
        assert kinds(s3)[:4] == [
            "image:OV",
            "text:" + overview_label(2.5, False)[:48],
            "image",
            "text:" + LN[:48],
        ], f"T-0140 A7: step 3's final user message; {kinds(s3)}"
        assert s3[1]["text"] == overview_label(2.5, False)
        assert s3[2]["image_url"]["url"] == builder_url(F1), (
            "A7: the current frame follows the overview"
        )
        assert s3[4:] == own, "the message's own parts follow unchanged"
        assert image_urls(step_body(rig, 3, "t2")) == [builder_url(F0), ov_url(), builder_url(F1)]
        inv = [i for i in browser.invocations if i.call_id == "t2-call-s3"]
        assert len(inv) == 1 and inv[0].frame is not None and inv[0].frame.phash == F1.phash, (
            "A7: step 3's click resolves against F1"
        )
        s4 = last_user(step_body(rig, 4, "t2"))["content"]
        assert s4[0]["image_url"]["url"] == builder_url(F2) and s4[1] == {
            "type": "text",
            "text": LU,
        }, f"A7: the following swap replaces the frame; {kinds(s4)}"
        assert s4[2:] == own and ov_in(step_body(rig, 4, "t2")) == 0, kinds(s4)


async def test_screenshot_unchanged_after_overview(tmp_path: Path) -> None:
    """A7 (reviewer mutation 4, ``updated`` computed with the overview): the overview step's payload says
    ``screenshot: "unchanged"`` (the whole response written out by hand from A3's action string) and step 2's screen
    label reads "your last action did not change it" (LN). Reading (module docstring): no fresh frame is shown, the
    current frame is F0 again."""
    browser = tall_browser()
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S3)],))
        action = overview_action(9.3, True)
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "page_overview", {}, executed(action, updated=False))]
        ), f"T-0140 A7: the payload says unchanged; got ...{step_tail(rig, 2)[-220:]}"
        parts = last_user(step_body(rig, 2))["content"]
        imgs = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
        assert [parts[i]["image_url"]["url"] for i in imgs] == [ov_url(), builder_url(F0)], kinds(
            parts
        )
        assert parts[imgs[-1] + 1] == {"type": "text", "text": LN}, (
            f"T-0140 A7: the label reads 'did not change it'; {kinds(parts)}"
        )


# =====================================================================================================
# A8: when it is refused
# =====================================================================================================


async def test_refusals_blind_turn(tmp_path: Path) -> None:
    """A8 (and R-s): a blind turn (the onset ``frame()`` fails) refuses ``page_overview`` before dispatch with ``no
    screen this turn``: the card and the loop payload say so (payload by hand), the browser sees no
    ``page_overview`` execute and no ``overview_of`` read, step 2 stays blind; the turn line counts it."""
    browser = tall_browser(frames=())
    browser.frame_mode = "raise"
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S3)],))
        assert image_urls(pass1_body(rig)) == [], "premise: blind"
        assert overview_invocations(browser) == [] and browser.overview_reads == [], (
            f"T-0140 A8: nothing is dispatched or captured on a blind turn; calls {browser.calls}"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["refused"]
        assert card(rig, "t1-call").reason == NO_SCREEN, card(rig, "t1-call")
        assert step_tail(rig, 2) == want_tail(
            rig, [(S1, "page_overview", {}, status("refused", NO_SCREEN, False))]
        ), step_tail(rig, 2)[-200:]
        assert image_urls(step_body(rig, 2)) == [] and OV_STEM not in json.dumps(step_body(rig, 2))
        extras = dict(rig.record("t1").extras)
    assert extras.get("page_overview") == counts(calls=1, refused=1), extras.get("page_overview")


async def _four_images(rig: Any, frames: tuple[Frame, ...]) -> None:
    """Onset plus a scene change per further frame, then the question (one turn, ``len(frames)`` own frames)."""
    await rig.onset()
    for i in range(1, len(frames)):
        rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash=f"scene-{i + 1}"))
        await rig.pump()
        await rig.advance(200)
    await rig.words(tsl.QUESTION)
    await rig.end_of_turn()


async def test_refusals_image_slots(tmp_path: Path) -> None:
    """A8 and R-s (both directions, ``.claude/rules/tests.md``): with ``max_images`` = 4 and 4 images already in the
    request (a history image plus three own frames of this turn) the overview is refused before dispatch with ``no
    image slot for an overview`` and no history image is evicted for it; with 3 images (three own frames, no history)
    it is shown, and step 2 carries exactly 4 images, the overview immediately before the newest frame."""
    budget = None
    a, b, c, d = (live_frame(t) for t in ("hist-a", "own-b", "own-c", "own-d"))
    browser = tall_browser(frames=(a, b, c, d), post=(F1,))
    async with feedback_rig(
        tmp_path / "full",
        browser,
        script=[answer("A shop. ")],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        budget = rig.session.prompts.budget.max_images
        assert budget == 4, "premise: the engine's four image slots"
        await rig.ask("what is on the screen")
        await rig.turn_done()
        rig.llm.script = [answer(S1), tool(OV_CALL)]
        rig.llm.script_steps([answer(S3)])
        await _four_images(rig, (b, c, d))
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        await rig.advance(300)
        p1 = pass1_body(rig, "t2")
        assert len(image_urls(p1)) == 4, f"premise: 4 images on the wire; {len(image_urls(p1))}"
        assert overview_invocations(browser) == [] and browser.overview_reads == [], (
            "T-0140 A8: nothing is dispatched or captured without an image slot"
        )
        assert card(rig, "t2-call").reason == NO_SLOT, card(rig, "t2-call")
        s2 = step_body(rig, 2, "t2")
        assert len(image_urls(s2)) == 4 and ov_in(s2) == 0, "R-s: no history image evicted for it"
        assert step_tail(rig, 2, "t2") == want_tail(
            rig, [(S1, "page_overview", {}, status("refused", NO_SLOT, False))], "t2"
        )
        extras = dict(rig.record("t2").extras)
    assert extras.get("page_overview") == counts(calls=1, refused=1), extras
    # the guard is not too wide: 3 images + the overview = 4 is allowed
    b2, c2, d2 = (live_frame(t) for t in ("own-b2", "own-c2", "own-d2"))
    browser = tall_browser(frames=(b2, c2, d2), post=(F1,))
    async with feedback_rig(
        tmp_path / "three",
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        rig.llm.script_steps([answer(S3)])
        await _four_images(rig, (b2, c2, d2))
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        assert len(image_urls(pass1_body(rig))) == 3, "premise: three own frames"
        s2 = step_body(rig, 2)
        urls = image_urls(s2)
        assert urls == [builder_url(b2), builder_url(c2), ov_url(), builder_url(d2)], (
            f"T-0140 A8: with a free slot the overview is shown before the newest frame; {kinds(last_user(s2)['content'])}"
        )
        parts = last_user(s2)["content"]
        at = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
        assert parts[at[2] + 1] == {"type": "text", "text": overview_label(9.3, True)}
        assert at[3] == at[2] + 2, "the overview label sits right before the current frame"


async def test_refusals_executor_outcomes(tmp_path: Path) -> None:
    """A3 through the session, and A10's counts: a fitting page gives the fits string with no image placed; a page
    that moved gives ``failed`` (payload by hand) and no image; a browser without the capture gives ``refused``
    (``page overview is not available``). None of the three places an overview or writes the ``overview`` request
    key."""
    cases = {
        "fits": (FITS_PAGE, {}, counts(calls=1, fits=1)),
        "moved": (TALL, {"overview_moves": True}, counts(calls=1, failed=1)),
        "unavailable": (TALL, {"overview_available": False}, counts(calls=1, refused=1)),
    }
    for name, (metrics, attrs, want) in cases.items():
        browser = tall_browser(metrics=metrics)
        for k, v in attrs.items():
            setattr(browser, k, v)
        async with feedback_rig(
            tmp_path / name,
            browser,
            script=[answer(S1), tool(OV_CALL)],
            session_kwargs={"live_feedback": label_on()},
            kit=overview_kit(browser),
        ) as rig:
            await run_turn(rig, ([answer(S3)],))
            assert len(overview_invocations(browser)) == 1, f"premise {name}: dispatched"
            s2 = step_body(rig, 2)
            assert ov_in(s2) == 0 and OV_STEM not in json.dumps(s2), (
                f"A3/A7 {name}: no image placed"
            )
            tail = step_tail(rig, 2)
            if name == "fits":
                # the page did not change: the same frame, so "unchanged" (by hand)
                assert tail == want_tail(
                    rig, [(S1, "page_overview", {}, executed(OVERVIEW_FITS, updated=False))]
                ), tail[-220:]
            elif name == "moved":
                assert Q + OVERVIEW_MOVED + Q in tail and Q + "failed" + Q in tail, tail[-220:]
            else:
                assert tail == want_tail(
                    rig, [(S1, "page_overview", {}, status("refused", OVERVIEW_UNAVAILABLE, False))]
                ), tail[-220:]
            extras = dict(rig.record("t1").extras)
        assert extras.get("page_overview") == want, (name, extras.get("page_overview"))
        assert ov_records(rig) == [], f"A10 {name}: no request carries an overview"


# =====================================================================================================
# A9: never spoken, never stored
# =====================================================================================================


async def test_never_voiced_or_stored(tmp_path: Path) -> None:
    """A9 (adversarial): across an overview turn (overview, click, report) and a second turn, the overview label (its
    stem, its "9.3 screens", its closing sentence), the action string's own words and the overview's pixels appear in
    no ``TtsSpeak`` (text or source), no history entry, no ``Episode`` and no ``machine.events()``; the label and the
    pixels appear in no HUD message. The step-2 request carries them (premise). No narration is added: the voiced text
    is exactly the model's own answer text (S1, S3, then turn 2's)."""
    browser = tall_browser(frames=(F0, F2))
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        await next_turn(rig, 2, "and now", [answer("Still a shop. ")])
        sent = json.dumps(all_bodies(rig))
        assert OV_STEM in sent and "showed the whole page" in sent and b64(OV.jpeg) in sent, (
            "premise"
        )
        label_needles = (
            OV_STEM,
            "9.3 screens",
            "The next image is the browser screen",
            b64(OV.jpeg),
        )
        action_needles = ("showed the whole page", "as a picture before the screen")
        speaks = [c.text + "|" + c.source for c in rig.machine.commands(C.TtsSpeak)]
        assert speaks, "premise: something was spoken"
        stores = {
            "TtsSpeak": json.dumps(speaks),
            "history": tsl.history_text(rig),
            "Episode": json.dumps(
                [e.model_dump(mode="json") for e in rig.memory.episodes.values()]
            ),
            "machine.events()": json.dumps([repr(e) for e in rig.machine.events()]),
        }
        for where, text in stores.items():
            for n in label_needles + action_needles:
                assert n not in text, f"T-0140 A9: {n[:40]!r} reached {where}"
        # O-A2 Q1: the HUD half is ``test_overview_image_and_label_not_on_hud`` (the card may show the action string,
        # which also says "9.3 screens")
        voiced = "".join(c.source for c in rig.machine.commands(C.TtsSpeak))
        assert voiced.replace(" ", "") == (S1 + S3 + "Still a shop. ").replace(" ", ""), (
            f"A9: no narration is added; voiced {voiced!r}"
        )


async def test_overview_image_and_label_not_on_hud(tmp_path: Path) -> None:
    """A9 as ruled (O-A2 Q1): across an overview turn and a following step, no HUD message carries the overview's
    pixels (data URL or base64), its phash, or ``OVERVIEW_LABEL`` (its stem and its closing sentence). The overview
    call's tool card may show its action string like every other card, so that is not pinned. The step-2 request
    carries the image and the label (premise)."""
    browser = tall_browser()
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        s2 = json.dumps(step_body(rig, 2))
        assert b64(OV.jpeg) in s2 and OV_STEM in s2, "premise: step 2 carries image and label"
        # T-0177 B-P2: "not on the HUD" reads "in no message other than astra.vision.frame" (H-T177-2: the panel
        # shows exactly the images sent, the overview included)
        hud = published_json_outside_frames(rig.published.messages)
        for n in (b64(OV.jpeg), OV.phash, OV_STEM, "The next image is the browser screen"):
            assert n not in hud, (
                f"T-0140 A9 (O-A2 Q1, T-0177 B-P2): {n[:40]!r} reached the HUD outside astra.vision.frame"
            )
        # ...and inside one it is the image step 2 sent, as an overview, with its exact bytes (T-0177 B7)
        shown = [
            (i.kind, i.jpeg_b64)
            for m in rig.published.messages
            if getattr(m, "type", None) == "astra.vision.frame"
            for i in m.images
        ]
        assert ("overview", b64(OV.jpeg)) in shown, (
            f"T-0177 B7/H-T177-2: the step-2 overview the model saw is published in an astra.vision.frame; "
            f"published kinds {[k for k, _ in shown]}"
        )


async def test_rider_step1_overview_refused(tmp_path: Path) -> None:
    """O-A2 Q3: turn 2's onset frame has turn 1's bytes, so its final user message carries no current-frame image; a
    step-1 ``page_overview`` is refused before dispatch with ``no screen this turn`` and nothing is captured."""
    again = F0.model_copy(update={"ts_ms": 77_777})
    browser = tall_browser(frames=(F0, again), post=(F1,))
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer("A shop. ")],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        await next_turn(rig, 2, tsl.QUESTION, [answer(S1), tool(OV_CALL)], ([answer(S3)],))
        own = last_user(pass1_body(rig, "t2"))["content"]
        assert "image_url" not in [p["type"] for p in own], "premise: rides on history"
        assert overview_invocations(browser) == [] and browser.overview_reads == [], browser.calls
        assert card(rig, "t2-call").reason == NO_SCREEN, card(rig, "t2-call")
        extras = dict(rig.record("t2").extras)
    assert extras.get("page_overview") == counts(calls=1, refused=1), extras.get("page_overview")


async def test_judge_refused_call_not_counted(tmp_path: Path) -> None:
    """O-A2 Q4: with the overview kit, ``call:page_overview{x:1}`` is refused by the judge (schema); it counts in
    neither ``calls`` nor ``refused`` (the key is still there, all zeros), and nothing reaches the browser."""
    browser = tall_browser()
    bad = "<|tool_call>call:page_overview{x:1}<tool_call|>"
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(bad)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        assert overview_invocations(browser) == [], browser.calls
        assert card(rig, "t1-call").status == "refused", card(rig, "t1-call")
        extras = dict(rig.record("t1").extras)
    assert extras.get("page_overview") == counts(), extras.get("page_overview")


# =====================================================================================================
# A10: logging
# =====================================================================================================


async def test_turn_line_and_request_keys(tmp_path: Path) -> None:
    """A10: an overview turn's line carries ``page_overview == {"calls": 1, "shown": 1, "fits": 0, "refused": 0,
    "failed": 0}``; a later toolless turn of the same session carries the key with zeros; the step-2 ``llm.request``
    record (only it) carries ``overview == {"phash": <the overview's>, "screens": 9.3, "clipped": true, "width": 320,
    "height": 800}``. A v1 live kit (not offered) adds no key to any line and no ``overview`` to any request. No new
    log kind and no new span against the v1 session running a click turn and the same call."""
    browser = tall_browser(frames=(F0, F2))
    async with feedback_rig(
        tmp_path / "on",
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        await next_turn(rig, 2, "and now", [answer("Still a shop. ")])
        e1, e2 = dict(rig.record("t1").extras), dict(rig.record("t2").extras)
    assert e1.get("page_overview") == counts(calls=1, shown=1), e1.get("page_overview")
    assert e2.get("page_overview") == counts(), e2.get("page_overview")
    recs = ov_records(rig)
    assert len(recs) == 1 and recs[0].get("step") == 2, recs
    got = recs[0]["overview"]
    assert got == {
        "phash": OV.phash,
        "screens": pytest.approx(9.3),
        "clipped": True,
        "width": 320,
        "height": 800,
    } and set(got) == {"phash", "screens", "clipped", "width", "height"}, got
    on_kinds = {r["kind"] for r in rig.log_records}
    on_spans = set(json.loads(rig.lines()[0])["spans"])
    # not offered: the same call on a v1 kit, then a click turn
    v1 = tall_browser(frames=(F0, F2))
    async with feedback_rig(
        tmp_path / "v1",
        v1,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
    ) as off:
        await off.ask(tsl.QUESTION)
        await off.turn_done(limit_ms=20_000)
        await off.advance(300)
        await next_turn(
            off,
            2,
            tsl.QUESTION,
            [answer(S1), tool(VALID_CLICK)],
            ([answer(S2), tool(CLICK2)], [answer(S3)]),
        )
        lines = [dict(off.record(t).extras) for t in ("t1", "t2")]
    assert all("page_overview" not in x for x in lines), "A10: no key when not offered"
    assert ov_records(off) == [], "A10: no overview request key when not offered"
    off_kinds = {r["kind"] for r in off.log_records}
    assert on_kinds <= off_kinds, f"A10: no new log kind; {sorted(on_kinds - off_kinds)}"
    off_spans = set(json.loads(off.lines()[1])["spans"]) | set(json.loads(off.lines()[0])["spans"])
    assert on_spans <= off_spans, f"A10: no new span; {sorted(on_spans - off_spans)}"


# =====================================================================================================
# A6: judging it; A8's last sentence
# =====================================================================================================


async def test_judged_with_kit_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A6: with the overview kit, ``call:page_overview{}`` is valid and executes; every ``live.classify`` call the
    session makes for the turn's commits (step 1's overview, step 2's click) is given the kit's set (its sha), never
    ``None``."""
    seen: list[Any] = []
    real = live.classify

    def spy(*a: Any, **k: Any) -> Any:
        bound = inspect.signature(real).bind(*a, **k)
        seen.append(bound.arguments.get("tool_set"))
        return real(*a, **k)

    monkeypatch.setattr(live, "classify", spy)
    browser = tall_browser()
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
        kit=overview_kit(browser),
    ) as rig:
        await run_turn(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert [i.call_id for i in overview_invocations(browser)] == ["t1-call"], (
            "T-0140 A6: the overview kit's judge accepts page_overview and it executes"
        )
        assert card(rig, "t1-call").status == "executed", card(rig, "t1-call")
    want = overview_set().version
    assert len(seen) >= 2, f"premise: both commits were judged; {len(seen)}"
    versions = [getattr(s, "version", None) for s in seen]
    assert versions == [want] * len(seen), (
        f"T-0140 A6: every pass-1 commit is judged with the kit's set; got {versions}"
    )


async def test_v1_kit_refuses_overview_call(tmp_path: Path) -> None:
    """A8's last sentence (green at BASE; reviewer mutation 7, the judge always on the overview set): a v1 live kit
    (the default) judges ``call:page_overview{}`` as at BASE, ``refused`` with ``no tool name``; nothing reaches the
    browser and nothing is captured."""
    browser = tall_browser()
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CALL)],
        session_kwargs={"live_feedback": label_on()},
    ) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        assert overview_invocations(browser) == [] and browser.overview_reads == [], browser.calls
        assert card(rig, "t1-call").status == "refused", card(rig, "t1-call")
        assert card(rig, "t1-call").reason == NO_TOOL_NAME, card(rig, "t1-call")
        assert "page_overview" not in dict(rig.record("t1").extras)


# =====================================================================================================
# O-A1-11's carried rows (A1's code; green at BASE_A2)
# =====================================================================================================


def _synthetic_set(tmp_path: Path) -> Any:
    """v1's five declarations with click_at's x maximum 499 (not 999) and a press_key list without ``Escape`` but with
    ``F13``: a set whose bounds and key list differ from v1's."""
    decls = json.loads(tools.TOOLS_V1_PATH.read_text())
    for d in decls:
        if d["name"] == "click_at":
            d["parameters"]["properties"]["x"]["maximum"] = 499
        if d["name"] == "press_key":
            enum = d["parameters"]["properties"]["combo"]["enum"]
            assert "Escape" in enum and "F13" not in enum, "premise: v1's list"
            enum.remove("Escape")
            enum.append("F13")
    p = tmp_path / "synthetic_tools.json"
    p.write_text(json.dumps(decls, indent=2))
    return tools.load(p)


def test_judge_reads_given_set_bounds_and_keys(tmp_path: Path) -> None:
    """O-A1-11 (reviewer A's surviving mutant, a helper that ignores ``tool_set``): against a set whose click_at x
    maximum is 499 and whose press_key list drops ``Escape`` and adds ``F13``, a complete and a truncated click at
    x=600 are refused (out of the set's range), a truncated ``Escape`` is refused (not in the set's list), a complete
    ``F13`` is valid and a truncated one repairable; v1 gives the other verdict on each (by hand from v1's schema: 999
    and its list), so the rows are not satisfied by a judge that refuses everything or reads v1 always."""
    syn = _synthetic_set(tmp_path)
    open_ = "<|tool_call>"
    # (text, v1 verdict, the synthetic set's verdict)
    rows = [
        (open_ + "call:click_at{x:600,y:10}<tool_call|>", "valid", "refused"),
        (open_ + "call:click_at{x:600,y:", "repairable", "refused"),
        (open_ + "call:press_key{combo:" + Q + "Escape" + Q, "repairable", "refused"),
        (open_ + "call:press_key{combo:" + Q + "F13" + Q + "}<tool_call|>", "refused", "valid"),
        (open_ + "call:press_key{combo:" + Q + "F13" + Q, "refused", "repairable"),
    ]
    for text, v1_status, syn_status in rows:
        assert live.classify(text).status == v1_status, (text, live.classify(text))
        got = live.classify(text, tool_set=syn)
        assert got.status == syn_status, (
            f"O-A1-11: the judge reads the given set's bounds and key list; {text!r} gave {got}"
        )


async def test_executor_page_size_unreadable() -> None:
    """O-A1-9/O-A1-11: a metrics read that fails before the capture gives ``failed`` with ``the page size could not be
    read``; nothing is captured (no ``full_page_screenshot`` call) and no overview id is kept."""
    from tests.unit.test_computer_executor import inv
    from tests.unit.test_t0140_executor import OverviewDriver, build

    class Broken(OverviewDriver):
        async def layout_metrics(self) -> dict[str, float]:
            self._enter("layout_metrics")
            raise RuntimeError("CDP said no")

    d = Broken()
    ex = build(d)
    out = await ex.execute(inv("page_overview", None))
    assert out.status == "failed" and out.reason == "the page size could not be read", out
    assert d.full_page_calls == [], "nothing is captured"
    assert "overview" not in out.resolved, out.resolved


def test_overview_max_screens_agree() -> None:
    """O-A2-12: the session copies the clip constant (it imports nothing from ``computer/``); the copy and the
    executor's agree, and both are H-T140-2's 4 (the label's ``{C}`` and the action string's must say the same)."""
    from local_astra.computer import executor as executor_mod

    s = session_mod()
    assert getattr(s, "OVERVIEW_MAX_SCREENS", None) == executor_mod.MAX_OVERVIEW_SCREENS == 4, (
        getattr(s, "OVERVIEW_MAX_SCREENS", None),
        executor_mod.MAX_OVERVIEW_SCREENS,
    )
    assert MAX_OVERVIEW_SCREENS == 4, "the support module's hand copy (A3's constant)"
