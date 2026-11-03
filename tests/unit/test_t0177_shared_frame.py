"""T-0177 group A, A6/A7/A10 (partial): step 1 under the ``shared`` vision choice, over the fake live browser.

The rig (``vision_rig``) is T-0114's ``tool_rig`` with a live kit over a ``VisionBrowser`` (``FakeLiveBrowser`` plus
a no-op ``reset_page``), the HUD perception serving distinctive *shared* frames, and a real ``HudServer`` (no socket)
around the session with one greeted page (``rig.page``). Selects go over the wire as raw JSON through that page, so
before A1 lands a row fails on ``select``'s named assertion (the codec answers ``Unknown``), never at collection.

Expected values: request images are compared with the data URL the real ``PromptBuilder`` writes for the frame
(``builder_url``); the label strings are copied by hand from docs/tasks/T-0177.md A7, never imported.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools live, loop on (8 steps), n = 1.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.prompt_builder import NO_IMAGE_STATEMENT
from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocol import AstraFrame, AstraHello
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    FakeLiveBrowser,
    b64,
    builder_url,
    image_urls,
    live_feedback,
    live_frame,
    live_kit,
    turn_extras,
)
from tests.unit._tool_turn_support import answer
from tests.unit.test_session_live import _wire, last_user

SELECT = "astra.vision.select"
#: A7, copied by hand from the task
SHARED_LABEL_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_LABEL_CAMERA = "[This image is the user's camera now, not your browser; click_at and type_at cannot act on it.]"
#: T-0174 A2's forward-pointing regex, typed from that criterion
FORWARD = r"(?i)\b(next|new|upcoming|following)\s+screenshot|\bwill\s+(show|see|appear)|\bwait\b"

#: The one private name this module reads off ``Session`` (``test_frame_geometry_source_is_the_frames_own``): the map
#: every frame's ``FrameGeometry`` is registered in at BASE 7646b8b6. This CONSTRAINS THE IMPLEMENTATION'S NAMING: the
#: attribute must keep this name and stay a mapping whose values are ``FrameGeometry`` (keys are free). If it is
#: renamed, change this constant (test-author) rather than working around it.
FRAME_GEOMETRY_ATTR = "_frame_geometry"

F0 = live_frame("t0177-onset")


def shared_frame(
    tag: str,
    *,
    source: Source = Source.SCREEN,
    ts_ms: int = 0,
    width: int = 1568,
    height: int = 590,
) -> Frame:
    """A distinctive HUD frame (the user's shared tab/window/camera)."""
    return Frame(
        source=source,
        ts_ms=ts_ms,
        width=width,
        height=height,
        jpeg=b"\xff\xd8" + f"SHARED-T0177-{tag}|".encode() * 8,
        phash=hashlib.sha256(tag.encode()).hexdigest()[:16],
        title="the user's shared tab",
        app="firefox",
    )


SH = shared_frame("screen-a")


class VisionBrowser(FakeLiveBrowser):
    """``FakeLiveBrowser`` with the ``reset_page`` a session reset calls (T-0162 reviewer B)."""

    async def reset_page(self) -> None:
        self.calls.append(("reset_page",))


@asynccontextmanager
async def vision_rig(
    tmp_path: Path,
    browser: Any = None,
    *,
    script: Any = (),
    hud_frames: Any = None,
    machine_cls: Any = None,
    session_kwargs: dict[str, Any] | None = None,
    kit: Any = "live",
    ingress: bool = False,
) -> AsyncIterator[Any]:
    """The live tool rig plus a socketless ``HudServer`` and one greeted page (module docstring).

    ``session_kwargs``: extra ``Session`` keywords (``live_feedback``, ``overview_frame``, ``auto_overview``); a keyword
    the session does not take is a named ``AssertionError``. ``kit``: ``"live"`` (the live kit over ``browser``), or
    any kit object (``None``: tools off; ``"default"``: dry_run). ``ingress``: the production shape of ``cli`` -- one
    real ``BrowserFrameSource`` is both the session's ``PerceptionClient`` and the server's frame sink, so HUD frames
    arrive only as ``astra.frame`` messages through ``rig.page`` (``post``); ``rig.perception`` is that source."""
    import inspect

    from local_astra.conductor.hud_server import HudServer
    from local_astra.perception.browser_source import BrowserFrameSource
    from tests.unit import _tool_turn_support as tts
    from tests.unit._hud_rig import _NullSink

    browser = browser if browser is not None else VisionBrowser([F0])
    extra: dict[str, Any] = {}
    if machine_cls is not None:
        extra["machine_cls"] = machine_cls
    kwargs = dict(session_kwargs or {})
    real = tts.Session
    missing = [k for k in kwargs if k not in inspect.signature(real.__init__).parameters]
    assert not missing, f"Session(..., {', '.join(missing)}=...) does not exist"
    sources: list[Any] = []
    wrap = bool(kwargs) or ingress
    if wrap:

        class Wrapped(real):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                for key, value in kwargs.items():
                    k.setdefault(key, value)
                if ingress:
                    clock = k["clock"]
                    source = BrowserFrameSource(now_ms=lambda: clock.now_ms)
                    sources.append(source)
                    a = (
                        *a[:5],
                        source,
                        *a[6:],
                    )  # machine, llm, stt, tts, audio, perception, memory
                super().__init__(*a, **k)

        tts.Session = Wrapped  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path,
            kit=live_kit(browser) if kit == "live" else kit,
            policy=tsl.LOOP,
            script=script,
            frames=list(hud_frames) if hud_frames is not None else [SH],
            **extra,
        ) as rig:
            if wrap:
                tts.Session = real  # type: ignore[misc]
            browser.now_ms = lambda: rig.clock.now_ms
            sink: Any = sources[0] if ingress else _NullSink()
            if ingress:
                rig.perception = sink
            server = HudServer(rig.session, frame_sink=sink, port=0)
            recorder = rig.published

            def tee(message: Any) -> None:
                server.publish(message)
                recorder(message)

            rig.session.set_publisher(tee)
            page = server.open()
            page.receive(encode(AstraHello()))
            rig.browser, rig.server, rig.page = browser, server, page  # type: ignore[attr-defined]
            yield rig
    finally:
        tts.Session = real  # type: ignore[misc]


async def select(rig: Any, source: str, page: Any = None) -> None:
    """One ``astra.vision.select`` over the wire, then the pump."""
    page = page if page is not None else rig.page
    before = dict(rig.server.counts)
    page.receive(json.dumps({"type": SELECT, "source": source}))
    after = rig.server.counts
    assert (after["unhandled"], after["malformed"]) == (before["unhandled"], before["malformed"]), (
        f"T-0177 A1/A3: {SELECT} is a known client->server type routed to the session; counts {dict(after)}"
    )
    await rig.pump()


def gen_body(rig: Any, turn: str = "t1") -> dict[str, Any]:
    return _wire(tsl.sent(rig, f"{turn}-gen"))


def frame_calls(browser: Any) -> int:
    return sum(1 for c in browser.calls if c[0] == "frame")


async def one_turn(rig: Any, question: str = "what is on my screen") -> None:
    await rig.ask(question)
    await rig.turn_done()


# =====================================================================================================
# A6 under browser: B6 stands verbatim (green at BASE)
# =====================================================================================================


async def test_browser_choice_never_carries_the_hud_frame(tmp_path: Path) -> None:
    """A6 last paragraph (B6 verbatim under ``browser``, the default): the step-1 image is the browser's own frame and
    the distinctive HUD frame's bytes appear in no body. Green at BASE and must stay green."""
    async with vision_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await one_turn(rig)
        assert image_urls(gen_body(rig)) == [builder_url(F0)]
        assert b64(SH.jpeg) not in json.dumps(rig.llm.bodies), "the HUD frame reached a body"
        assert turn_extras(rig).get("frame_source") == "window"


# =====================================================================================================
# A6/A10 under shared
# =====================================================================================================


async def test_shared_step1_carries_the_shared_frame_and_never_asks_the_browser(
    tmp_path: Path,
) -> None:
    """A6 (adversarial): the browser has a distinctive frame ready; under ``shared`` step 1 carries the HUD frame
    only, ``frame()`` is never called and the browser's bytes appear in no body. A10: ``vision_source: "shared"``,
    ``frame_source: "screen"``."""
    async with vision_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await one_turn(rig)
        assert image_urls(gen_body(rig)) == [builder_url(SH)], "A6: step 1 shows the shared frame"
        assert frame_calls(rig.browser) == 0, (
            f"A6: no browser frame() for the snapshot; {rig.browser.calls}"
        )
        assert b64(F0.jpeg) not in json.dumps(rig.llm.bodies), (
            "A6: the browser frame reached a body"
        )
        extras = turn_extras(rig)
        assert (extras.get("vision_source"), extras.get("frame_source")) == ("shared", "screen"), (
            extras
        )


async def test_shared_with_no_frame_is_blind_with_no_fallback(tmp_path: Path) -> None:
    """A6: no HUD frame under ``shared``: the existing no-image wording, no image, no browser fallback; A10
    ``frame_source: "none"``."""
    async with vision_rig(tmp_path, script=[answer("I cannot see it. ")], hud_frames=[]) as rig:
        await select(rig, "shared")
        await one_turn(rig)
        body = gen_body(rig)
        assert image_urls(body) == [] and NO_IMAGE_STATEMENT in json.dumps(body), image_urls(body)
        assert frame_calls(rig.browser) == 0 and b64(F0.jpeg) not in json.dumps(rig.llm.bodies)
        extras = turn_extras(rig)
        assert (extras.get("vision_source"), extras.get("frame_source")) == ("shared", "none"), (
            extras
        )


async def test_back_to_browser_drops_the_shared_frame(tmp_path: Path) -> None:
    """A6 adversarial (the other direction): shared then browser before the turn; the HUD bytes appear nowhere and
    the line says ``vision_source: "browser"``."""
    async with vision_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await select(rig, "browser")
        await one_turn(rig)
        assert image_urls(gen_body(rig)) == [builder_url(F0)]
        assert b64(SH.jpeg) not in json.dumps(rig.llm.bodies)
        assert turn_extras(rig).get("vision_source") == "browser"


# =====================================================================================================
# A7: the shared label
# =====================================================================================================


async def test_the_shared_label_follows_the_image_with_live_feedback_off(tmp_path: Path) -> None:
    """A7: with ``live_feedback`` off (the rig's default ``None``) the step-1 final user message carries exactly
    ``SHARED_LABEL_SCREEN`` right after the shared image; never on a prefill; the ``llm.request`` record's
    ``live_label.text`` is it. Premise: the literal points nowhere forward and never says "browser screen"."""
    import re

    assert (
        re.search(FORWARD, SHARED_LABEL_SCREEN) is None
        and "browser screen" not in SHARED_LABEL_SCREEN
    )
    async with vision_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await one_turn(rig)
        body = gen_body(rig)
        parts = last_user(body)["content"]
        idx = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
        assert len(idx) == 1 and parts[idx[0]]["image_url"]["url"] == builder_url(SH), parts
        assert parts[idx[0] + 1] == {"type": "text", "text": SHARED_LABEL_SCREEN}, parts[idx[0] + 1]
        assert json.dumps(body).count("user's shared screen now") == 1
        for _, p in rig.llm.prefills:
            assert "shared screen now" not in json.dumps(_wire(p)), "A7: no label on a prefill"
    gens = [d for d in rig.kind("llm.request") if d.get("request_id") == "t1-gen"]
    assert gens and all(
        (d.get("live_label") or {}).get("text") == SHARED_LABEL_SCREEN for d in gens
    ), gens


@pytest.mark.parametrize(
    ("screen_label", "page_extent"), [(False, False), (True, True)], ids=["off", "label+extent"]
)
async def test_the_shared_label_under_every_live_feedback(
    tmp_path: Path, screen_label: bool, page_extent: bool
) -> None:
    """A7 "placed whatever ``live_feedback`` is" and "never contains 'browser screen', an extent clause or an outcome
    clause": under ``LiveFeedback(False, False)`` and ``LiveFeedback(True, True)`` the step-1 final user message is the
    shared image followed by exactly ``SHARED_LABEL_SCREEN``, and no text of that message says "browser screen". The
    browser has an extent scripted for its own frame, so a T-0134 label leaking onto the shared image would carry an
    extent clause too."""
    feedback = live_feedback(screen_label=screen_label, page_extent=page_extent)
    browser = VisionBrowser([F0])
    browser.extents[F0.phash] = {"above_px": 0.0, "below_px": 1600.0, "viewport_px": 800.0}
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer("A shop. ")],
        session_kwargs={"live_feedback": feedback},
    ) as rig:
        await select(rig, "shared")
        await one_turn(rig)
        parts = last_user(gen_body(rig))["content"]
    idx = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
    assert len(idx) == 1 and parts[idx[0]]["image_url"]["url"] == builder_url(SH), parts
    assert parts[idx[0] + 1] == {"type": "text", "text": SHARED_LABEL_SCREEN}, parts[idx[0] + 1]
    texts = [str(p.get("text", "")) for p in parts if p.get("type") == "text"]
    assert not [t for t in texts if "browser screen" in t], (
        f"A7: no 'browser screen' wording on a shared step; {texts}"
    )
    assert sum(1 for t in texts if t.startswith("[")) == 1, (
        f"A7: one bracketed label after the shared image, no second (extent/outcome) label; {texts}"
    )


# =====================================================================================================
# A6/H-T177-5: the newest HUD frame's source, through the real ingress (BrowserFrameSource)
# =====================================================================================================


def posted(
    tag: str, *, source: Source = Source.SCREEN, width: int = 1568, height: int = 590
) -> tuple[AstraFrame, Frame]:
    """An ``astra.frame`` the real ``BrowserFrameSource`` accepts (16-hex phash, JPEG magic), and the ``Frame`` whose
    builder URL is the oracle for it (the URL depends on the bytes only)."""
    jpeg = b"\xff\xd8\xff\xe0" + f"POSTED-T0177-{tag}|".encode() * 8
    phash = hashlib.sha256(tag.encode()).hexdigest()[:16]
    message = AstraFrame(
        source=source,
        ts_ms=0,
        width=width,
        height=height,
        jpeg_b64=b64(jpeg),
        phash=phash,
        title=f"the user's {source.value}",
        app="firefox",
    )
    frame = Frame(source=source, ts_ms=0, width=width, height=height, jpeg=jpeg, phash=phash)
    return message, frame


async def post(rig: Any, message: AstraFrame, *, then_ms: int = 1000) -> None:
    """One HUD frame over the wire (the page's own path), then ``then_ms`` of session time."""
    before = rig.server.counts["frames_rejected"]
    rig.page.receive(encode(message))
    await rig.pump()
    assert rig.server.counts["frames_rejected"] == before, "premise: the source accepted the frame"
    await rig.advance(then_ms)


async def test_the_newest_hud_frame_is_a_window_so_shared_shows_the_window(
    tmp_path: Path,
) -> None:
    """A6 / H-T177-5 (adversarial: ``config.source`` read instead of the newest frame's source): the page posts a
    screen frame, then a window frame; under ``shared`` step 1 carries the window frame only, labelled as the shared
    screen (a window is a screen for A7), and the line says ``frame_source: "window"``. The screen frame's bytes and
    the browser's appear in no body."""
    scr_msg, scr = posted("older-screen")
    win_msg, win = posted("newest-window", source=Source.WINDOW, width=1200, height=900)
    async with vision_rig(tmp_path, script=[answer("A shop. ")], ingress=True) as rig:
        await post(rig, scr_msg)
        await post(rig, win_msg)
        await select(rig, "shared")
        await one_turn(rig)
        body = gen_body(rig)
        assert image_urls(body) == [builder_url(win)], "H-T177-5: the newest HUD frame's source"
        assert {"type": "text", "text": SHARED_LABEL_SCREEN} in last_user(body)["content"]
        dumped = json.dumps(rig.llm.bodies)
        assert b64(scr.jpeg) not in dumped and b64(F0.jpeg) not in dumped
        assert frame_calls(rig.browser) == 0, rig.browser.calls
        extras = turn_extras(rig)
        assert (extras.get("vision_source"), extras.get("frame_source")) == ("shared", "window")


async def test_a_camera_frame_gets_the_camera_label(tmp_path: Path) -> None:
    """A7: a camera source gives ``SHARED_LABEL_CAMERA``; A10 ``frame_source: "camera"``. Driven through the page's
    own ingress (orchestrator ruling 3): a screen frame, then a camera frame, posted as ``astra.frame``; the camera is
    the newest HUD frame's source (H-T177-5), so how the implementation learns it is its own business."""
    scr_msg, scr = posted("screen-before-camera")
    cam_msg, cam = posted("camera-a", source=Source.CAMERA, width=896, height=504)
    async with vision_rig(tmp_path, script=[answer("A mug. ")], ingress=True) as rig:
        await post(rig, scr_msg)
        await post(rig, cam_msg)
        await select(rig, "shared")
        await one_turn(rig)
        body = gen_body(rig)
        assert image_urls(body) == [builder_url(cam)], "A6: the camera frame is the shared frame"
        assert {"type": "text", "text": SHARED_LABEL_CAMERA} in last_user(body)["content"]
        assert SHARED_LABEL_SCREEN not in json.dumps(body) and b64(scr.jpeg) not in json.dumps(body)
        extras = turn_extras(rig)
        assert (extras.get("vision_source"), extras.get("frame_source")) == ("shared", "camera")


async def test_frame_geometry_source_is_the_frames_own(tmp_path: Path) -> None:
    """A6 "``FrameGeometry.source`` is the frame's own ``Source``, never ``WINDOW``": after a shared turn over a screen
    frame (1568x590, a size no browser frame of the rig has), the session's geometry for that frame says ``SCREEN``.
    Read from ``Session.<FRAME_GEOMETRY_ATTR>`` (the map every frame's geometry is registered in at BASE), found by
    size, not by key, so the key the implementation chooses is free."""
    async with vision_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await one_turn(rig)
        assert image_urls(gen_body(rig)) == [builder_url(SH)], "premise: the shared turn"
        table = getattr(rig.session, FRAME_GEOMETRY_ATTR, None)
        assert hasattr(table, "values"), (
            f"T-0177 A6 (test seam): Session.{FRAME_GEOMETRY_ATTR} is the frame-geometry map; got {type(table)}"
        )
        geometry = list(table.values())
    mine = [g for g in geometry if (g.width, g.height) == (SH.width, SH.height)]
    assert mine and all(g.source == Source.SCREEN for g in mine), (
        f"A6: the shared frame's geometry carries its own source; {mine} (all: {geometry})"
    )


# =====================================================================================================
# A6: scene changes append shared frames; a pre-reset frame is stale
# =====================================================================================================


async def test_a_scene_change_appends_a_second_shared_frame(tmp_path: Path) -> None:
    """A6 "HUD scene changes append shared frames under T-0059's rules": the onset snapshot takes ``SH``; a
    ``SceneChanged`` while the user speaks takes the newer shared frame, appended after it (oldest first). Neither
    snapshot calls the browser's ``frame()``, and the browser's bytes are in no body."""
    later = shared_frame("screen-b", ts_ms=600)
    async with vision_rig(tmp_path, script=[answer("They differ. ")], hud_frames=[SH]) as rig:
        await select(rig, "shared")
        await rig.onset()
        await rig.advance(600)
        rig.perception.add(later)
        rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash=later.phash))
        await rig.pump()
        await rig.words("compare these")
        await rig.end_of_turn()
        await rig.turn_done()
        urls = image_urls(gen_body(rig))
        assert urls == [builder_url(SH), builder_url(later)], (
            f"A6: both shared frames, oldest first; got {len(urls)} images "
            f"(SH {builder_url(SH) in urls}, later {builder_url(later) in urls})"
        )
        assert frame_calls(rig.browser) == 0 and b64(F0.jpeg) not in json.dumps(rig.llm.bodies)


async def test_a_pre_reset_shared_frame_is_stale(tmp_path: Path) -> None:
    """A6 "the frame dates from before the last reset, the turn is blind" (T-0162 A-3), and the guard's width: the
    shared frame posted before a reset is never shown (no-image wording, no browser fallback); one posted after the
    reset is the next turn's frame. The choice survives the reset (T-0162 D-3), so both turns are shared."""
    from local_astra.contracts.protocol import AstraSessionReset

    async with vision_rig(tmp_path, script=[answer("I cannot see it. ")], hud_frames=[SH]) as rig:
        await select(rig, "shared")
        rig.page.receive(encode(AstraSessionReset()))
        await rig.pump()
        await rig.advance(500)
        await one_turn(rig)
        body = gen_body(rig)
        assert image_urls(body) == [] and NO_IMAGE_STATEMENT in json.dumps(body), (
            f"A6: a pre-reset shared frame is stale; {len(image_urls(body))} images"
        )
        assert b64(SH.jpeg) not in json.dumps(rig.llm.bodies) and frame_calls(rig.browser) == 0
        assert turn_extras(rig).get("frame_source") == "none", turn_extras(rig)
        fresh = shared_frame("after-reset", ts_ms=rig.clock.now_ms + 100)
        await rig.advance(200)
        rig.perception.add(fresh)
        rig.llm.script = [answer("Now I see it. ")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        urls = image_urls(gen_body(rig, "t2"))
        assert urls == [builder_url(fresh)], (
            f"guard width: a post-reset shared frame is shown; {len(urls)} images"
        )
        assert turn_extras(rig, "t2").get("vision_source") == "shared"


# =====================================================================================================
# A5: frozen per turn
# =====================================================================================================


async def test_a_select_during_thinking_applies_from_the_next_turn(tmp_path: Path) -> None:
    """A5 adversarial: ``shared`` is selected while a browser turn is THINKING (its generate held). That turn stays a
    browser turn to its end: its step-2 request, which takes a fresh frame after a failed click (T-0111g B8), shows
    the browser's fresh frame and no shared bytes, and its line says ``vision_source: "browser"``. The next turn is
    shared."""
    from local_astra.contracts.turn import TurnState
    from tests.unit._live_browser_support import RESTARTED, failed
    from tests.unit._tool_turn_support import VALID_CLICK, tool

    fr = live_frame("t0177-fresh")

    def restart(inv: Any) -> Any:
        browser.restart()
        return failed(inv, RESTARTED)

    browser = VisionBrowser([F0, fr])
    browser.outcome("t1-call", restart)
    async with vision_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(tsl.S3)])
        rig.llm.hold()
        await rig.ask(tsl.QUESTION)
        assert rig.state.state is TurnState.THINKING, f"premise: THINKING; {rig.state.state}"
        await select(rig, "shared")
        rig.llm.release()
        await tsl.until_steps(rig, 1, limit_ms=20_000, step=100)
        await rig.turn_done(limit_ms=20_000)
        t1 = [_wire(p) for rid, p in rig.llm.gen_prompts if "t1-gen" in rid]
        assert len(t1) >= 2, f"premise: step 1 and step 2; {[r for r, _ in rig.llm.gen_prompts]}"
        assert not [b for b in t1 if b64(SH.jpeg) in json.dumps(b)], (
            "A5: no request of the open turn carries the shared frame"
        )
        assert image_urls(_wire(tsl.step_request(rig, 2)))[-1] == builder_url(fr), (
            "A5: the fresh frame after the failed click is the browser's"
        )
        assert turn_extras(rig).get("vision_source") == "browser", turn_extras(rig)
        await rig.advance(300)
        rig.llm.script = [answer("A shop. ")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        current = [
            p["image_url"]["url"]
            for p in last_user(gen_body(rig, "t2"))["content"]
            if p.get("type") == "image_url"
        ]
        assert current == [builder_url(SH)], f"A5: the next turn is shared; {len(current)} images"
        assert turn_extras(rig, "t2").get("vision_source") == "shared"


def _decoded(url: str) -> str:
    """The bytes of a ``data:`` URL, as latin-1 text (to find a sentinel)."""
    import base64

    return base64.b64decode(url.split(",", 1)[1]).decode("latin-1")


@pytest.mark.parametrize("scene", [True, False], ids=["scene-change", "no-scene-change"])
@pytest.mark.parametrize("frozen,later_choice", [("browser", "shared"), ("shared", "browser")])
async def test_a_select_while_the_user_speaks_applies_from_the_next_turn(
    tmp_path: Path, frozen: str, later_choice: str, scene: bool
) -> None:
    """A5, both directions, at the reads that follow the freeze (reviewer B, mutants 04a/04c). The turn opens (onset)
    under ``frozen``; the page then selects ``later_choice``; with ``scene``, a new HUD frame arrives with a
    ``SceneChanged`` (a second ``snapshot_frame`` of the same turn); then words and end of turn. ``overview_frame`` is
    on and the page is 1.525 screens (T-0151's deals page), so step 1 also runs the overview decision after the select.
    Everything the turn did follows the frozen choice:

    - every request of the turn (prefills and generates) carries only the frozen source's bytes, never both (shared:
      the HUD frames, whose bytes carry ``SHARED-T0177``; browser: every other image, its own frames or the overview);
    - under a frozen ``shared``, the browser's ``frame()`` is never called and nothing is dispatched;
    - the overview counts are the frozen choice's, with T-0151's five keys;
    - the line says ``vision_source`` = ``frozen`` and ``frame_source`` = ``window`` (browser) or ``screen`` (shared).

    Why two arms: the scene change makes step 1 a two-frame turn under either choice (browser: a second own frame;
    shared: the later HUD frame), and T-0151 refuses the overview on a multi-frame step 1 whatever the source, so with
    a scene change the count is ``refused`` 1 in both directions and cannot tell the choices apart. The
    ``no-scene-change`` arm is the overview arm: browser ``shown`` 1 with one capture, shared ``refused`` 1 with none.
    The ``scene-change`` arm is the ``snapshot_frame`` arm. A read of the live choice after the freeze in
    ``snapshot_frame`` (04a) breaks the scene arm and one in the overview decision (04c) the other, in each direction."""
    later = shared_frame("screen-later", ts_ms=600)
    browser = overview_browser(DEALS)
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer("A deals page. ")],
        hud_frames=[SH],
        session_kwargs={"overview_frame": True},
    ) as rig:
        if frozen == "shared":
            await select(rig, "shared")
        await rig.onset()
        await select(rig, later_choice)
        await rig.advance(600)
        if scene:
            rig.perception.add(later)
            rig.submit(E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash=later.phash))
            await rig.pump()
        await rig.words("what is on my screen")
        await rig.end_of_turn()
        await rig.turn_done()
        prefills = [_wire(p) for _, p in rig.llm.prefills]
        gens = [_wire(p) for rid, p in rig.llm.gen_prompts if "t1-" in rid]
        extras = turn_extras(rig)
    assert gens, "premise: the turn sent a generate"
    urls = [u for body in (*prefills, *gens) for u in image_urls(body)]
    shared_urls = {builder_url(SH), builder_url(later)}
    # the HUD frames' bytes are distinctive (``SHARED-T0177-``); every other image is the browser's (own frames,
    # including any the fake mints, or the overview picture)
    has_shared = any(u in shared_urls or "SHARED-T0177" in _decoded(u) for u in urls)
    has_browser = any(u not in shared_urls and "SHARED-T0177" not in _decoded(u) for u in urls)
    counts = extras.get("overview_frame")
    step1_images = len([p for p in last_user(gens[0])["content"] if p.get("type") == "image_url"])
    if frozen == "browser":
        assert has_browser and not has_shared, (
            f"A5: a turn frozen as browser carries only browser bytes; shared {has_shared}, browser {has_browser}"
        )
        if scene:
            assert step1_images == 2 and frame_calls(browser) == 2, (
                f"premise: the scene change took a second own frame; {step1_images} images, {browser.calls}"
            )
            assert counts == {**dict.fromkeys(OVERVIEW_KEYS, 0), "refused": 1}, counts
            assert overview_executes(browser) == [], browser.calls
        else:
            assert counts == {**dict.fromkeys(OVERVIEW_KEYS, 0), "shown": 1}, (
                f"A5: the overview decision follows the frozen browser choice; {counts}; step-1 images "
                f"{step1_images}; browser calls {browser.calls}"
            )
            assert len(overview_executes(browser)) == 1, browser.calls
        assert (extras.get("vision_source"), extras.get("frame_source")) == ("browser", "window"), (
            extras
        )
    else:
        assert has_shared and not has_browser, (
            f"A5: a turn frozen as shared carries only shared bytes; shared {has_shared}, browser {has_browser}"
        )
        assert frame_calls(browser) == 0, (
            f"A5: no browser frame() under a frozen shared; {browser.calls}"
        )
        assert step1_images == (2 if scene else 1), (
            f"premise: step 1 shows {'both HUD frames' if scene else 'the HUD frame'}; {step1_images}"
        )
        assert overview_executes(browser) == [] and browser.overview_frame_reads == [], (
            browser.calls
        )
        assert counts == {**dict.fromkeys(OVERVIEW_KEYS, 0), "refused": 1}, (
            f"A5: the overview decision follows the frozen shared choice; {counts}"
        )
        assert (extras.get("vision_source"), extras.get("frame_source")) == ("shared", "screen"), (
            extras
        )


@pytest.mark.parametrize("frozen,later_choice", [("browser", "shared"), ("shared", "browser")])
async def test_the_auto_overview_follows_the_frozen_choice(
    tmp_path: Path, frozen: str, later_choice: str
) -> None:
    """A5 x A8, the automatic overview (reviewer B round 2, mutant 04d): ``auto_overview`` on, the 9.3-screen page
    with a 320x800 overview picture, no scene change. The turn opens (onset) under ``frozen``, the page selects
    ``later_choice`` while the user speaks, then words and end of turn; ``_auto_overview`` runs at step 1, after the
    select. Browser frozen: ``shown`` 1 of T-0150's four keys and one ``page_overview`` capture. Shared frozen:
    ``refused`` 1, no capture and no ``frame()`` call. A read of the live choice in ``_auto_overview`` (04d) swaps
    the two outcomes."""
    browser = overview_browser(TALL)
    browser.overview_image = OV.model_copy(update={"width": 320, "height": 800})
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer("A long page. ")],
        hud_frames=[SH],
        session_kwargs={"auto_overview": True},
    ) as rig:
        if frozen == "shared":
            await select(rig, "shared")
        await rig.onset()
        await select(rig, later_choice)
        await rig.advance(600)
        await rig.words("what is on my screen")
        await rig.end_of_turn()
        await rig.turn_done()
        extras = turn_extras(rig)
    counts = extras.get("auto_overview")
    assert extras.get("vision_source") == frozen, extras
    if frozen == "browser":
        assert counts == {**dict.fromkeys(AUTO_KEYS, 0), "shown": 1}, (
            f"A5: the automatic overview follows the frozen browser choice; {counts}; {browser.calls}"
        )
        assert len(overview_executes(browser)) == 1, browser.calls
    else:
        assert counts == {**dict.fromkeys(AUTO_KEYS, 0), "refused": 1}, (
            f"A5: the automatic overview follows the frozen shared choice; {counts}; {browser.calls}"
        )
        assert overview_executes(browser) == [], f"A5/A8: refused before capture; {browser.calls}"
        assert frame_calls(browser) == 0, (
            f"A5: no browser frame() under a frozen shared; {browser.calls}"
        )


# =====================================================================================================
# A8: overview never touches a shared step 1
# =====================================================================================================

#: the scaled whole-page picture the fake executor keeps (sentinel bytes; the session only carries it)
OV = Frame(
    source=Source.WINDOW,
    ts_ms=0,
    width=838,
    height=798,
    jpeg=b"\xff\xd8" + b"OVERVIEW-T0177|" * 8,
    phash="7177717771777177",
)
#: deals.html at 1280x800: 1.525 screens (T-0151's page); TALL: 9.3 screens (T-0140's)
DEALS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1220.0}
TALL = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 7440.0}
OV_CALL = "<|tool_call>call:page_overview{}<tool_call|>"
OVERVIEW_KEYS = {"shown", "fits", "too_tall", "failed", "refused"}
AUTO_KEYS = {"shown", "fits", "failed", "refused"}


def overview_browser(metrics: dict[str, float]) -> VisionBrowser:
    browser = VisionBrowser([F0, live_frame("t0177-t2")], echo="model")
    browser.overview_image = OV
    browser.page_metrics = dict(metrics)
    return browser


def overview_executes(browser: Any) -> list[Any]:
    return [i for i in browser.invocations if i.name == "page_overview"]


def shared_overview_reason() -> str:
    from tests.unit._live_browser_support import session_mod

    reason = getattr(session_mod(), "SHARED_OVERVIEW_REASON", None)
    assert isinstance(reason, str) and reason.strip(), (
        f"T-0177 A8: session.SHARED_OVERVIEW_REASON is a non-empty string; got {reason!r}"
    )
    return reason


async def test_the_overview_frame_is_refused_on_a_shared_step1(tmp_path: Path) -> None:
    """A8: with ``overview_frame`` on, a 1.525-screen page and the choice ``shared``, ``_decide_overview_frame``
    dispatches nothing (no ``page_overview`` reaches the browser), counts ``refused`` (the five keys unchanged, one
    at 1), notes ``overview frame: shared screen``, and step 1 shows the shared frame. Control (too wide): the same
    page under ``browser`` shows the overview (``shown`` 1, one capture)."""
    arms: dict[str, Any] = {}
    for arm in ("browser", "shared"):
        browser = overview_browser(DEALS)
        async with vision_rig(
            tmp_path / arm,
            browser,
            script=[answer("A deals page. ")],
            session_kwargs={"overview_frame": True},
        ) as rig:
            if arm == "shared":
                await select(rig, "shared")
            await one_turn(rig)
            body = gen_body(rig)
        arms[arm] = (rig, browser, body, turn_extras(rig).get("overview_frame"))
    _, b_browser, _, b_counts = arms["browser"]
    assert b_counts == {**dict.fromkeys(OVERVIEW_KEYS, 0), "shown": 1}, (
        f"premise: under browser the overview frame is shown; {b_counts}"
    )
    assert len(overview_executes(b_browser)) == 1
    rig, browser, body, counts = arms["shared"]
    assert overview_executes(browser) == [] and browser.overview_frame_reads == [], (
        f"A8: nothing dispatched on a shared step 1; {browser.calls}"
    )
    assert counts == {**dict.fromkeys(OVERVIEW_KEYS, 0), "refused": 1}, counts
    assert image_urls(body) == [builder_url(SH)] and b64(OV.jpeg) not in json.dumps(body)
    notes = [str(d.get("what", "")) for d in rig.kind("note")]
    assert [n for n in notes if "overview frame: shared screen" in n], (
        f"A8: the refusal is noted; notes {notes}"
    )


async def test_the_auto_overview_is_refused_before_capture_on_a_shared_step1(
    tmp_path: Path,
) -> None:
    """A8: with ``auto_overview`` on and a 9.3-screen page, under ``shared`` the automatic overview refuses before
    capture (no ``page_overview`` execute, ``refused`` 1 of the four keys) and step 1 carries only the shared frame.
    Control: under ``browser`` the same page is captured and shown."""
    arms: dict[str, Any] = {}
    for arm in ("browser", "shared"):
        browser = overview_browser(TALL)
        browser.overview_image = OV.model_copy(update={"width": 320, "height": 800})
        async with vision_rig(
            tmp_path / arm,
            browser,
            script=[answer("A long page. ")],
            session_kwargs={"auto_overview": True},
        ) as rig:
            if arm == "shared":
                await select(rig, "shared")
            await one_turn(rig)
            body = gen_body(rig)
        arms[arm] = (browser, body, turn_extras(rig).get("auto_overview"))
    b_browser, _, b_counts = arms["browser"]
    assert b_counts == {**dict.fromkeys(AUTO_KEYS, 0), "shown": 1} and overview_executes(
        b_browser
    ), f"premise: under browser the automatic overview is taken and shown; {b_counts}"
    browser, body, counts = arms["shared"]
    assert overview_executes(browser) == [], f"A8: refused before capture; {browser.calls}"
    assert counts == {**dict.fromkeys(AUTO_KEYS, 0), "refused": 1}, counts
    assert image_urls(body) == [builder_url(SH)] and b64(OV.jpeg) not in json.dumps(body)


async def test_page_overview_off_a_shared_image_is_refused(tmp_path: Path) -> None:
    """A8: a ``page_overview`` call made while the current image is shared settles ``refused`` with
    ``SHARED_OVERVIEW_REASON`` (read from ``session.py``, checked non-empty): nothing reaches the browser, the card and
    the call's final ``tool.call`` record carry exactly that reason, and the turn line's ``page_overview`` keys are
    T-0140's. Control: under ``browser`` the same call is dispatched and not refused for that reason."""
    from tests.unit._tool_turn_support import tool
    from tests.unit.test_t0140_session import overview_kit

    reason = shared_overview_reason()
    arms: dict[str, Any] = {}
    for arm in ("browser", "shared"):
        browser = overview_browser(TALL)
        browser.overview_image = OV.model_copy(update={"width": 320, "height": 800})
        async with vision_rig(
            tmp_path / arm,
            browser,
            script=[answer(tsl.S1), tool(OV_CALL)],
            kit=overview_kit(browser),
        ) as rig:
            if arm == "shared":
                await select(rig, "shared")
            rig.llm.script_steps([answer(tsl.S3)])
            await rig.ask(tsl.QUESTION)
            await tsl.until_steps(rig, 1, limit_ms=20_000, step=100)
            await rig.turn_done(limit_ms=20_000)
            final = rig.tool_calls("t1-call")[-1] if rig.tool_calls("t1-call") else None
        arms[arm] = (rig, browser, final)
    _, b_browser, b_final = arms["browser"]
    assert len(overview_executes(b_browser)) == 1, "premise: under browser the call is dispatched"
    assert b_final is not None and b_final.reason != reason, "too wide: refused under browser"
    rig, browser, final = arms["shared"]
    assert overview_executes(browser) == [], f"A8: nothing dispatched; {browser.calls}"
    assert final is not None and (final.status, final.reason) == ("refused", reason), final
    records = [d for d in rig.tool_records() if d.get("call_id") == "t1-call"]
    assert records and (records[-1].get("status"), records[-1].get("reason")) == (
        "refused",
        reason,
    ), records[-1:]
    counts = turn_extras(rig).get("page_overview")
    assert isinstance(counts, dict) and set(counts) == {
        "calls",
        "shown",
        "fits",
        "refused",
        "failed",
    }, counts


# =====================================================================================================
# A11: assists unchanged (T-0111g B14 under both sources)
# =====================================================================================================


async def test_assists_are_dropped_under_shared_too(tmp_path: Path) -> None:
    """A11 / H-T177-6: in live under ``shared``, assist text is still not attached (the HUD capture is now the
    model's view, the tempting reason to attach it), the flags are frozen and logged, and the line says
    ``assists_dropped: true``."""
    from local_astra.contracts.perception import AssistFlags
    from local_astra.contracts.protocol import AstraAssistToggle, AstraFrameText

    canary = "ASSIST-CANARY-T0177"
    async with vision_rig(tmp_path, script=[answer("It says hi. ")]) as rig:
        await select(rig, "shared")
        rig.session.apply_assist_toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
        rig.session.accept_frame_text(
            AstraFrameText(
                source=Source.SCREEN, ts_ms=rig.clock.now_ms, kind="page_text", text=canary
            )
        )
        await one_turn(rig, "what does the page say")
        assert image_urls(gen_body(rig)) == [builder_url(SH)], "premise: a shared turn"
        assert canary not in json.dumps(rig.llm.bodies), "A11: no assist block under shared"
        record = rig.record("t1")
    assert record.regime.assist_flags.page_text is True
    assert record.extras.get("assists_dropped") is True, record.extras


# =====================================================================================================
# A10: non-live turn lines are unchanged (green at BASE by design)
# =====================================================================================================


@pytest.mark.parametrize("mode", ["off", "dry_run"])
async def test_non_live_lines_carry_no_vision_keys(tmp_path: Path, mode: str) -> None:
    """A10 "Non-live turn lines are unchanged" (and A3: a select outside live changes nothing): with HUD frames and a
    raw ``astra.vision.select`` sent before the turn, an ``off`` / ``dry_run`` line has neither ``vision_source`` nor
    ``frame_source``, and the step-1 body carries the HUD frame (non-live input is unchanged). Green at BASE."""
    async with vision_rig(
        tmp_path, script=[answer("A shop. ")], kit=None if mode == "off" else "default"
    ) as rig:
        rig.page.receive(json.dumps({"type": SELECT, "source": "shared"}))
        await rig.pump()
        await one_turn(rig)
        assert image_urls(gen_body(rig)) == [builder_url(SH)], "premise: the HUD frame is shown"
        extras = turn_extras(rig)
    assert "vision_source" not in extras and "frame_source" not in extras, extras
