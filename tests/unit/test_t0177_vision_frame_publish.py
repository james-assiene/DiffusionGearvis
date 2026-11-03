"""T-0177 group B, B7-B9: the published ``astra.vision.frame`` (the "What I see" panel's only source).

Rigs: group A's ``vision_rig`` (the live tool rig, a socketless ``HudServer`` and one greeted page, ``kit=None`` for
``off`` and ``"default"`` for ``dry_run``); T-0131's ``one_click_turn`` for the ReGround row; bare ``Outbox`` objects
for B8. What the session published is ``rig.published.messages`` (the recorder teed beside the server).

Oracles: a published image's bytes are compared with the bytes of the request body's own data URL (the wire body the
fake LLM's real client builds, base64-decoded here) *and* with the fixture frame's bytes; its ``identity`` with the
``phash`` of the ``ImagePart`` in the request's ``Prompt``. Nothing is recomputed with the code under test.

Readings (listed in the report):

- ``history_images`` counts the image parts outside the request's final user message (turn 2 over a browser frame,
  with turn 1's shared image in history: 1). In the riding case it is not pinned (whether the ridden image itself is
  counted is open).
- Outside live the HUD's capture is the user's shared screen, so a non-live frame's ``source`` is ``"shared"`` with
  ``kind`` the frame's own source (``"screen"``).
- The step-1 prefill and generate send the same images, so they publish one frame between them (B7's change rule);
  T-0171 resends, T-0172 re-asks and degenerate retries resend the same images, so whether they reach the publish
  point is not observable through the wire (no row; named in the report).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, loop on (8 steps), n = 1.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.hud_server import HudServer, Outbox
from local_astra.contracts.codec import MAX_MESSAGE_BYTES, encode
from local_astra.contracts.llm import ImagePart, Prompt
from local_astra.contracts.protocol import AstraHello, AstraSessionReset, AstraSpoken
from local_astra.memory.captioner import IDLE_BEFORE_CAPTION_MS
from tests.fakes.hud_server import vision_frame, vision_image
from tests.fakes.llm import is_caption_prompt
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import b64, live_frame
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_live import SCROLL, _wire, last_user
from tests.unit.test_t0177_shared_frame import (
    DEALS,
    OV,
    SH,
    VisionBrowser,
    overview_browser,
    select,
    shared_frame,
    vision_rig,
)

FRAME = "astra.vision.frame"
STATE = "astra.vision.state"
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
F0 = live_frame("t0177p-onset")
F1 = live_frame("t0177p-after-1")
F2 = live_frame("t0177p-after-2")


# --------------------------------------------------------------------------------------------- reading the rig


def frames(rig: Any, turn: str | None = None) -> list[Any]:
    """Every published ``astra.vision.frame``, in order (of ``turn`` only, when given)."""
    return [
        m
        for m in rig.published.messages
        if getattr(m, "type", None) == FRAME and (turn is None or m.turn_id == turn)
    ]


def gen_prompt(rig: Any, turn: str = "t1") -> Prompt:
    return tsl.sent(rig, f"{turn}-gen")


def body_bytes(body: dict[str, Any], *, final_only: bool = True) -> list[bytes]:
    """The bytes of every data URL of the body's final user message (or of the whole body), in order."""
    messages = [last_user(body)] if final_only else body["messages"]
    out: list[bytes] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    out.append(base64.b64decode(url.split(",", 1)[1]))
    return out


def final_identities(prompt: Prompt) -> list[str | None]:
    users = [m for m in prompt.messages if m.role == "user"]
    return [p.phash for p in users[-1].content if isinstance(p, ImagePart)]


def decoded(image: Any) -> bytes:
    assert image.jpeg_b64 is not None, f"premise: the image carries its bytes; {image}"
    return base64.b64decode(image.jpeg_b64)


def published_bytes(frame: Any) -> list[bytes]:
    return [decoded(i) for i in frame.images]


def greet(rig: Any) -> list[Any]:
    page = rig.server.open()
    page.receive(encode(AstraHello()))
    # T-0213 R-5 named re-pin: the one settings state is the reply's last message; the rows compare what precedes it.
    # Imported here, not at module level: BASE-tree subprocesses (T-0201's) import this module against an old
    # ``local_astra`` with no settings catalog.
    from tests.unit import test_t0213_settings_wire as t0213

    return t0213.hello_without_settings_state(page.sent())


async def loop_turn(rig: Any, steps: tuple[Any, ...], *, n: int) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(tsl.QUESTION)
    if n:
        await tsl.until_steps(rig, n, limit_ms=20_000, step=100)
    await rig.turn_done(limit_ms=20_000)


async def next_turn(rig: Any, n: int, script: list[Any], question: str = "and now") -> None:
    await rig.advance(300)
    rig.llm.script = list(script)
    await rig.ask(question)
    await rig.until(lambda: len(rig.lines()) >= n, f"turn {n} never ended", limit_ms=20_000)


def require_frames(rig: Any, turn: str | None = None) -> list[Any]:
    got = frames(rig, turn)
    assert got, (
        f"T-0177 B7: the session publishes {FRAME} for the requests it sends; published types "
        f"{sorted({getattr(m, 'type', '?') for m in rig.published.messages})}"
    )
    return got


# =====================================================================================================
# B7: the key row -- exactly the bytes the body sent
# =====================================================================================================


async def test_each_published_image_is_exactly_the_bytes_the_browser_body_sent(
    tmp_path: Path,
) -> None:
    """B7 key row, browser source: a click turn (step 1 ``F0``, step 2 ``F1``). Two frames are published for t1,
    steps 1 and 2; each one's images are the body's final user message's, in order: identity = the ``ImagePart``'s
    phash, ``jpeg_b64`` decodes to exactly the body's data-URL bytes (and the fixture's), ``kind`` window,
    1280x800, ``source`` browser, not blind, no history image, not a replay."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await loop_turn(rig, ([answer(S3)],), n=1)
        sent = {1: gen_prompt(rig), 2: tsl.step_request(rig, 2)}
        got = require_frames(rig, "t1")
    assert [f.step for f in got] == [1, 2], [(f.step, len(f.images)) for f in got]
    for f, fixture in zip(got, (F0, F1), strict=True):
        prompt = sent[f.step]
        assert published_bytes(f) == body_bytes(_wire(prompt)) == [fixture.jpeg], (
            f"B7: step {f.step}"
        )
        assert [i.identity for i in f.images] == final_identities(prompt)
        assert [(i.kind, i.width, i.height) for i in f.images] == [("window", 1280, 800)]
        assert (f.turn_id, f.source, f.blind, f.history_images, f.replay) == (
            "t1",
            "browser",
            False,
            0,
            False,
        ), f


async def test_each_published_image_is_exactly_the_bytes_the_shared_body_sent(
    tmp_path: Path,
) -> None:
    """B7 key row, shared choice: step 1 publishes the shared frame (``source`` shared, ``kind`` screen, 1568x590,
    the body's bytes); after the executed scroll step 2 publishes the browser frame with ``source`` browser -- the
    source of the images actually sent, not the turn's choice."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        sent = {1: gen_prompt(rig), 2: tsl.step_request(rig, 2)}
        got = require_frames(rig, "t1")
    assert [(f.step, f.source) for f in got] == [(1, "shared"), (2, "browser")], [
        (f.step, f.source) for f in got
    ]
    one, two = got
    assert published_bytes(one) == body_bytes(_wire(sent[1])) == [SH.jpeg]
    assert [(i.identity, i.kind, i.width, i.height) for i in one.images] == [
        (final_identities(sent[1])[0], "screen", 1568, 590)
    ]
    assert published_bytes(two) == body_bytes(_wire(sent[2])) == [F1.jpeg]
    assert [i.kind for i in two.images] == ["window"] and not one.blind and not two.blind


async def test_the_overview_frame_is_what_is_published_when_it_is_what_is_sent(
    tmp_path: Path,
) -> None:
    """B7 adversarial (the most likely failure: a frame built from another ``Prompt`` than the body's): with
    ``overview_frame`` on, the browser source and the 1.525-screen page, step 1's generate sends only the overview
    picture (T-0151). The last frame published for step 1 carries exactly that picture's bytes (``kind`` overview,
    its own size), never the live frame the overview replaced."""
    browser = overview_browser(DEALS)
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer("A deals page. ")],
        session_kwargs={"overview_frame": True},
    ) as rig:
        await rig.ask("what is on my screen")
        await rig.turn_done()
        body = _wire(gen_prompt(rig))
        assert body_bytes(body) == [OV.jpeg], "premise: step 1 sends only the overview picture"
        got = [f for f in require_frames(rig, "t1") if f.step == 1]
    last = got[-1]
    assert published_bytes(last) == [OV.jpeg], (
        f"B7: the published frame is the body's; {[i.kind for i in last.images]}"
    )
    assert [(i.kind, i.width, i.height) for i in last.images] == [("overview", OV.width, OV.height)]
    assert last.source == "browser" and frames(rig)[-1] == last


# =====================================================================================================
# B7: when a frame is published
# =====================================================================================================


async def test_a_frame_is_published_only_when_the_images_change(tmp_path: Path) -> None:
    """B7 "only when the request's (source, identities, blind) differs from the last one published": turn 1 searches
    at step 1 and answers at step 2 (prefill, generate and step 2 all show ``F0``): one frame. Turn 2 shows a new
    browser frame ``F1``: one more frame, for t2."""
    browser = VisionBrowser([F0, F1])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(tsl.SEARCH_CALL)]) as rig:
        await loop_turn(rig, ([answer(S3)],), n=1)
        assert len(rig.llm.prefills) >= 1, "premise: a prefill was sent"
        first = require_frames(rig, "t1")
        await next_turn(rig, 2, [answer("A shop. ")])
        second = frames(rig, "t2")
    assert len(first) == 1 and published_bytes(first[0]) == [F0.jpeg], [
        (f.step, len(f.images)) for f in first
    ]
    assert len(second) == 1 and published_bytes(second[0]) == [F1.jpeg], second


async def test_the_prefill_publishes_before_any_generate(tmp_path: Path) -> None:
    """B7 "the turn's prefill": while the user is still speaking (a prefill sent, no generate yet), the frame of the
    prefill's images is already published."""
    async with vision_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await rig.onset()
        await rig.advance(600)
        await rig.words("what is on")
        await rig.until(lambda: bool(rig.llm.prefills), "no prefill was sent", limit_ms=5_000)
        assert rig.llm.generates == [], "premise: no generate yet"
        got = require_frames(rig)
    assert len(got) == 1 and published_bytes(got[0]) == [SH.jpeg] and got[0].step == 1, got


async def test_the_captioner_publishes_nothing(tmp_path: Path) -> None:
    """B7 "No frame is published for the captioner": outside live, after one answered turn the idle captioner sends
    its caption request; the number of published frames does not move."""
    async with vision_rig(tmp_path, script=[answer("Yes. ")], kit=None) as rig:
        await rig.ask("what is this")
        await rig.turn_done()
        before = len(require_frames(rig))
        rig.llm.hold()
        await rig.advance(IDLE_BEFORE_CAPTION_MS + 1_000, step=500)
        captions = [p for _, p, _ in rig.llm.generates if is_caption_prompt(p)]
        assert captions, "premise: the captioner sent a request"
        assert any(
            isinstance(part, ImagePart) for m in captions[0].messages for part in m.content
        ), "premise: the caption request carries an image"
        after = len(frames(rig))
        rig.llm.release()
    assert after == before, f"B7: the captioner published {after - before} frame(s)"


@pytest.mark.parametrize("mode", ["off", "dry_run"])
async def test_non_live_publishes_too(tmp_path: Path, mode: str) -> None:
    """B7 "in all modes": outside live the step-1 request's HUD frame is published: the body's bytes, ``kind``
    screen, ``source`` shared (reading: the HUD capture is the user's shared screen), not blind."""
    async with vision_rig(
        tmp_path, script=[answer("A shop. ")], kit=None if mode == "off" else "default"
    ) as rig:
        await rig.ask("what is on my screen")
        await rig.turn_done()
        body = _wire(gen_prompt(rig))
        got = require_frames(rig, "t1")
    assert len(got) == 1, got
    f = got[0]
    assert published_bytes(f) == body_bytes(body) == [SH.jpeg]
    assert (f.source, f.blind, [i.kind for i in f.images]) == ("shared", False, ["screen"]), f


@pytest.mark.parametrize("choice", ["shared", "browser"])
async def test_a_blind_request_publishes_blind(tmp_path: Path, choice: str) -> None:
    """B7: a blind request publishes ``blind: true, images: []`` with the turn's source: shared with no HUD frame;
    browser with a ``frame()`` that fails."""
    browser = VisionBrowser([])
    if choice == "browser":
        browser.frame_mode = "raise"
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer("I cannot see it. ")],
        hud_frames=[] if choice == "shared" else None,
    ) as rig:
        if choice == "shared":
            await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done(limit_ms=20_000)
        assert body_bytes(_wire(gen_prompt(rig)), final_only=False) == [], "premise: blind"
        got = require_frames(rig, "t1")
    assert [(f.blind, f.images, f.source) for f in got] == [(True, [], choice)], got


async def test_history_images_counts_the_images_outside_the_final_message(tmp_path: Path) -> None:
    """B7 / H-T177-4. Turn 1 shared (``SH``); turn 2 browser (``F0``) with turn 1's ``SH`` in history:
    ``history_images`` 1, and the published image is ``F0`` only (the history image is not shown as current)."""
    browser = VisionBrowser([F0])
    async with vision_rig(tmp_path, browser, script=[answer("A shop. ")]) as rig:
        await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done()
        await select(rig, "browser")
        await next_turn(rig, 2, [answer("A page. ")])
        t2 = _wire(gen_prompt(rig, "t2"))
        two = require_frames(rig, "t2")
    assert SH.jpeg in body_bytes(t2, final_only=False), (
        "premise: turn 1's shared image is in history"
    )
    assert two and published_bytes(two[-1]) == [F0.jpeg], two
    assert two[-1].history_images == 1, f"reading: images outside the final user message; {two[-1]}"


async def test_a_frame_riding_on_a_history_image_is_published(tmp_path: Path) -> None:
    """B7 "A frame riding on a held history image is included instead". Turn 1 is shared (``SH``) and its step-1
    scroll swaps in the browser frame ``F1`` (the last frame published is browser ``F1``). Turn 2 is shared again
    over the same bytes, so its frame rides on turn 1's history image (the final user message holds no image): the
    frame published for t2 includes it -- ``SH``'s bytes, exactly the history data URL's, ``source`` shared, not
    blind -- and differs from the last one, so it is published.

    (Premise shape, test-author 2026-10-28: a shared -> browser -> shared sequence of answered turns re-sends ``SH``
    in turn 3's final message instead of riding, so the riding case is two consecutive shared turns with a browser
    frame published in between.)"""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        one = require_frames(rig, "t1")
        assert published_bytes(one[-1]) == [F1.jpeg], (
            "premise: the last frame published is browser F1"
        )
        await next_turn(rig, 2, [answer("The shop again. ")])
        t2 = _wire(gen_prompt(rig, "t2"))
        two = frames(rig, "t2")
    assert body_bytes(t2) == [] and SH.jpeg in body_bytes(t2, final_only=False), (
        "premise: turn 2's frame rides on turn 1's history image"
    )
    assert two, "B7: the riding frame is published (it differs from the last one, browser F1)"
    f = two[-1]
    assert (f.source, f.blind, f.step) == ("shared", False, 1), f
    assert published_bytes(f) == [SH.jpeg], [i.identity for i in f.images]
    assert [i.identity for i in f.images] == [SH.phash], "the ridden image's identity"


async def test_an_image_too_large_for_the_wire_is_omitted(tmp_path: Path) -> None:
    """B7 / A1 ``too_large``: a shared frame whose base64 alone exceeds ``codec.MAX_MESSAGE_BYTES`` is published with
    ``jpeg_b64`` null and ``omitted: "too_large"``, its identity and size kept, and the message encodes within the
    limit. (Too wide: every other row's images carry their bytes.)"""
    raw = b"\xff\xd8" + b"BIG-T0177|" * ((MAX_MESSAGE_BYTES * 3) // 4 // 10 + 10_000)
    assert len(base64.b64encode(raw)) > MAX_MESSAGE_BYTES, "premise: the bytes alone are too large"
    big = shared_frame("big").model_copy(update={"jpeg": raw})
    async with vision_rig(tmp_path, script=[answer("A big screen. ")], hud_frames=[big]) as rig:
        await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done(limit_ms=20_000)
        prompt = gen_prompt(rig)
        got = require_frames(rig, "t1")
    f = got[-1]
    assert [(i.jpeg_b64, i.omitted) for i in f.images] == [(None, "too_large")], [
        (i.omitted, i.identity) for i in f.images
    ]
    assert [(i.identity, i.width, i.height) for i in f.images] == [
        (final_identities(prompt)[0], big.width, big.height)
    ]
    assert len(encode(f).encode("utf-8")) <= MAX_MESSAGE_BYTES


async def test_a_reground_crop_is_never_published(tmp_path: Path) -> None:
    """B7 / H-T177-7: with ReGround on, the pass-2 crop request publishes nothing: every published image is one of
    the turn's step requests' images, and no published image is a crop's bytes."""
    from tests.unit.test_session_reground import OK_PASS2, one_click_turn

    rig, _, _ = await one_click_turn(tmp_path, [OK_PASS2])
    crops = {
        base64.b64decode(p.url.split(",", 1)[1])
        for _, prompt in rig.llm.pass2_prompts
        for m in prompt.messages
        for p in m.content
        if isinstance(p, ImagePart)
    }
    assert rig.llm.pass2_prompts and crops, "premise: a pass-2 request carried an image"
    turn_images = {
        b
        for prompt in [gen_prompt(rig), tsl.step_request(rig, 2)]
        for b in body_bytes(_wire(prompt), final_only=False)
    }
    got = require_frames(rig, "t1")
    shown = [b for f in got for b in published_bytes(f)]
    assert shown and set(shown) <= turn_images, (
        "B7: only the turn's own request images are published"
    )
    assert not set(shown) & (crops - turn_images), "H-T177-7: a crop was published"


async def test_no_frame_is_published_for_a_pre_reset_turn(tmp_path: Path) -> None:
    """B7 "No frame is published for ... a pre-reset turn": turn 1's generate is held; the page resets the
    session; the held request is released. Nothing published after ``astra.session.cleared`` is a t1 frame (and
    turn 1 did publish before the reset: the premise)."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(S3)])
        rig.llm.hold()
        await rig.ask(tsl.QUESTION)
        require_frames(rig, "t1")
        rig.page.receive(encode(AstraSessionReset()))
        await rig.pump()
        rig.llm.release()
        await rig.advance(3_000)
        types = [getattr(m, "type", None) for m in rig.published.messages]
        assert "astra.session.cleared" in types, "premise: the reset was published"
        cut = len(types) - types[::-1].index("astra.session.cleared")
        late = [m for m in rig.published.messages[cut:] if getattr(m, "type", None) == FRAME]
    assert [m for m in late if m.turn_id == "t1"] == [], late


# =====================================================================================================
# B7: the session log is redacted
# =====================================================================================================


async def test_no_base64_of_a_frame_reaches_the_log_file(tmp_path: Path) -> None:
    """B7: the ``wire.out`` record of a published frame is redacted: after a shared turn and a browser step the log
    file holds a ``wire.out`` for ``astra.vision.frame`` but not the base64 of either image."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(SCROLL)]) as rig:
        await select(rig, "shared")
        await loop_turn(rig, ([answer(S3)],), n=1)
        require_frames(rig, "t1")
        path = rig.log.path
    text = Path(path).read_text()
    outs = [
        r
        for r in rig.log_records
        if r["kind"] == "wire.out" and (r["data"].get("msg") or {}).get("type") == FRAME
    ]
    assert outs, "premise: the frames' wire.out records are in the log"
    for secret in (b64(SH.jpeg), b64(F1.jpeg)):
        assert secret not in text, "B7: base64 of a frame reached the session log"


# =====================================================================================================
# B8: droppable
# =====================================================================================================


def _vf(step: int) -> Any:
    return vision_frame(
        "t1", step, images=[vision_image(b"\xff\xd8" + bytes([step]) * 8, identity=f"id-{step}")]
    )


def _spoken(i: int) -> AstraSpoken:
    return AstraSpoken(turn_id="t1", utterance_id=f"u{i}", text=f"line {i}")


def _held_frames(box: Outbox) -> list[Any]:
    return [m for m in box.pending() if getattr(m, "type", None) == FRAME]


def test_a_newer_frame_replaces_the_unsent_one() -> None:
    """B8: at most one unsent frame per outbox; a newer one replaces it, and the line queued between them stays."""
    box = Outbox(512)
    assert box.offer(_vf(1)) and box.offer(_spoken(1)) and box.offer(_vf(2))
    assert [m.step for m in _held_frames(box)] == [2], box.pending()
    assert [m.utterance_id for m in box.pending() if isinstance(m, AstraSpoken)] == ["u1"]


def test_a_held_frame_never_costs_a_non_droppable_message() -> None:
    """B8: an outbox full with one held frame accepts the next spoken line (the frame goes); one full of spoken lines
    takes a frame without refusing (the frame is dropped), so the frame is never why a connection closes."""
    box = Outbox(2)
    assert box.offer(_spoken(1)) and box.offer(_vf(1))
    assert box.offer(_spoken(2)), "B8: a frame never makes a non-droppable message overflow"
    assert [getattr(m, "utterance_id", None) for m in box.pending()] == ["u1", "u2"]
    full = Outbox(2)
    assert full.offer(_spoken(1)) and full.offer(_spoken(2))
    assert full.offer(_vf(1)) is True, "B8: a frame into a full outbox is dropped, never fatal"
    assert _held_frames(full) == []


async def test_a_stuck_connection_is_never_closed_by_frames(tmp_path: Path) -> None:
    """B8 through the server: a page that never reads (outbox of 3) is sent 40 frames, then a spoken line: it stays
    open, holds one frame (the newest), and the spoken line is queued."""
    async with vision_rig(tmp_path) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0, outbox_size=3)
        conn = server.open()
        conn.receive(encode(AstraHello()))
        conn.outbox.drain()
        for k in range(1, 41):
            server.publish(_vf(k))
        server.publish(_spoken(1))
        assert not conn.closed, f"B8: frames closed the connection ({conn.close_reason})"
        assert [m.step for m in _held_frames(conn.outbox)] == [40]
        assert [m.utterance_id for m in conn.outbox.pending() if isinstance(m, AstraSpoken)] == [
            "u1"
        ]


class _Sink:
    def ingest(self, message: Any) -> None:
        return None


# =====================================================================================================
# B9: hello and reset
# =====================================================================================================


@pytest.mark.parametrize("mode", ["live", "off"])
async def test_the_hello_ends_with_the_latest_frame_replayed(tmp_path: Path, mode: str) -> None:
    """B9: after a turn, a new page's hello ends with the latest published frame, ``replay: true`` and otherwise
    equal; in live right after A4's state. Exactly one frame in the reply (the frame stays out of the T-0037
    journal)."""
    browser = VisionBrowser([F0], post_frames=[F1])
    script = [answer(S1), tool(VALID_CLICK)] if mode == "live" else [answer("A shop. ")]
    async with vision_rig(
        tmp_path, browser, script=script, kit="live" if mode == "live" else None
    ) as rig:
        if mode == "live":
            await loop_turn(rig, ([answer(S3)],), n=1)
        else:
            await rig.ask("what is on my screen")
            await rig.turn_done()
        latest = require_frames(rig)[-1]
        reply = greet(rig)
    kinds = [getattr(m, "type", None) for m in reply]
    assert kinds[-1] == FRAME and kinds.count(FRAME) == 1, kinds
    assert reply[-1] == latest.model_copy(update={"replay": True}), (reply[-1], latest)
    if mode == "live":
        assert kinds[-2] == STATE, kinds
    else:
        assert STATE not in kinds, kinds


@pytest.mark.parametrize("mode", ["live", "off"])
async def test_a_reset_clears_the_frame_slot(tmp_path: Path, mode: str) -> None:
    """B9: after a reset a new page's hello carries no frame (live: it still ends with the state; the choice
    survives, T-0162 D-3); the next turn's frame is the slot again."""
    async with vision_rig(
        tmp_path, script=[answer("A shop. ")], kit="live" if mode == "live" else None
    ) as rig:
        if mode == "live":
            await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done()
        require_frames(rig)
        rig.page.receive(encode(AstraSessionReset()))
        await rig.pump()
        await rig.advance(500)
        after_reset = greet(rig)
        kinds = [getattr(m, "type", None) for m in after_reset]
        assert FRAME not in kinds, f"B9: a reset clears the slot; {kinds}"
        if mode == "live":
            assert kinds[-1] == STATE and after_reset[-1].source == "shared", kinds
        fresh = shared_frame("after-reset", ts_ms=rig.clock.now_ms + 100)
        await rig.advance(200)
        rig.perception.add(fresh)
        rig.llm.script = [answer("Now I see it. ")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        again = greet(rig)
    assert [getattr(m, "type", None) for m in again][-1] == FRAME
    assert published_bytes(again[-1]) == [fresh.jpeg] and again[-1].replay is True


async def test_a_reset_forgets_the_published_key(tmp_path: Path) -> None:
    """B-R1 (review ruling, adversarial: a static start page after ``reset_page``). Live, browser source, the browser
    returns the identical frame ``F0`` before and after a reset. Turn 1 publishes ``F0``; the page resets; turn 2's
    request carries the same ``F0`` (same capture identity as the last frame published before the reset). After the
    reset exactly one frame is published, with ``F0``'s bytes, and a new page's hello ends with that frame
    (``replay: true``): the reset forgot the published key, so the empty slot is filled again."""
    browser = VisionBrowser([F0, F0])
    async with vision_rig(tmp_path, browser, script=[answer("A page. ")]) as rig:
        await rig.ask("what is on my screen")
        await rig.turn_done()
        before = require_frames(rig)
        assert published_bytes(before[-1]) == [F0.jpeg], "premise: turn 1 published F0"
        rig.page.receive(encode(AstraSessionReset()))
        await rig.pump()
        await rig.advance(500)
        types = [getattr(m, "type", None) for m in rig.published.messages]
        assert "astra.session.cleared" in types, "premise: the reset was published"
        cut = len(types) - types[::-1].index("astra.session.cleared")
        await next_turn(rig, 2, [answer("The same page. ")])
        last_gen = [p for rid, p, _ in rig.llm.generates if not is_caption_prompt(p)][-1]
        assert body_bytes(_wire(last_gen)) == [F0.jpeg], "premise: turn 2 sends the identical F0"
        late = [m for m in rig.published.messages[cut:] if getattr(m, "type", None) == FRAME]
        reply = greet(rig)
    assert len(late) == 1, (
        f"B-R1: after a reset the first request with an image publishes it even when its identity equals the "
        f"last one published before the reset; {len(late)} frame(s) published after the reset"
    )
    assert published_bytes(late[0]) == [F0.jpeg] and late[0].source == "browser", late[0]
    assert [i.identity for i in late[0].images] == [i.identity for i in before[-1].images], (
        "premise: the same capture identity as the frame published before the reset"
    )
    kinds = [getattr(m, "type", None) for m in reply]
    assert kinds[-1] == FRAME, f"B-R1/B9: the hello ends with the post-reset frame; {kinds}"
    assert reply[-1] == late[0].model_copy(update={"replay": True}), (reply[-1], late[0])


# =====================================================================================================
# B10: the only new wire output
# =====================================================================================================


async def test_the_only_new_message_type_in_a_live_browser_turn_is_the_frame(
    tmp_path: Path,
) -> None:
    """B10 (partial, by type): a live browser click turn publishes nothing of a type outside the BASE set plus
    ``astra.vision.frame`` (the BASE set is every server-to-client type ``protocol.py`` had before A, typed from the
    A1 direction table: the three vision types are the only additions, and ``astra.vision.state`` is published only
    on a select or a hello)."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await loop_turn(rig, ([answer(S3)],), n=1)
        require_frames(rig, "t1")
        types = {getattr(m, "type", None) for m in rig.published.messages}
    assert STATE not in types and "astra.vision.select" not in types, types


# =====================================================================================================
# B-P1: the pre-authorised normaliser itself (tests/unit/_t0177_pins.py), pure
# =====================================================================================================
#
# The older rows discount B7's frames only through ``without_vision_frames``; these rows pin that it strips exactly
# the declared difference (a well-formed frame as a model, a dict or a JSON string, and its ``wire.out`` record, as
# an element of a list or tuple) and nothing else, and that a malformed frame is refused rather than discounted.
# Expected outputs are written out by hand.


def _pin_frame() -> Any:
    return vision_frame(
        "t1", 2, images=[vision_image(b"\xff\xd8pin-frame", identity="pin-id", kind="window")]
    )


def _wire_out(msg: dict[str, Any]) -> dict[str, Any]:
    """A session-log ``wire.out`` record as the HUD server writes it (``data.msg`` is the dumped message)."""
    return {
        "kind": "wire.out",
        "t_ms": 10,
        "data": {"n_bytes": 99, "msg": msg, "dropped": False, "connections": 1},
    }


def _logged(frame: Any) -> dict[str, Any]:
    """The frame as B7 logs it: every image's bytes redacted (the redacted string's form is not pinned)."""
    msg = frame.model_dump(mode="json")
    for image in msg["images"]:
        image["jpeg_b64"] = "<redacted>"
    return msg


def test_b_p1_strips_a_frame_as_a_model_a_dict_a_json_string_and_its_wire_out_record() -> None:
    from tests.unit._t0177_pins import without_vision_frames

    frame = _pin_frame()
    spoken = _spoken(1)
    as_dict = frame.model_dump(mode="json")
    tree = {
        "messages": [spoken, frame, "x"],
        "dicts": [{"type": "astra.spoken", "text": "a"}, as_dict],
        "lines": (encode(frame), '{"type": "astra.spoken"}'),
        "records": [_wire_out(_logged(frame)), _wire_out(spoken.model_dump(mode="json"))],
    }
    got = without_vision_frames(tree)
    assert got == {
        "messages": [spoken, "x"],
        "dicts": [{"type": "astra.spoken", "text": "a"}],
        "lines": ('{"type": "astra.spoken"}',),
        "records": [_wire_out(spoken.model_dump(mode="json"))],
    }, f"B-P1: exactly the four frame forms go; got {got}"
    assert tree["messages"] == [spoken, frame, "x"], "B-P1: the input is not mutated"


def test_b_p1_strips_nothing_else() -> None:
    """Near misses stay: the state message, a dict naming another type, prose mentioning the type, a ``wire.out`` of
    another message, a record of another kind carrying a frame-shaped ``msg``, and a blind frame's sibling fields."""
    from local_astra.contracts.protocol import AstraVisionState
    from tests.unit._t0177_pins import without_vision_frames

    state = AstraVisionState.model_validate(
        {"type": "astra.vision.state", **{k: v for k, v in _state_fields().items()}}
    )
    frame_msg = _pin_frame().model_dump(mode="json")
    near = [
        state,
        {"type": "astra.vision.frames", "turn_id": "t1"},
        {"kind": "astra.vision.frame"},
        "the astra.vision.frame message is new",
        '["astra.vision.frame"]',
        _wire_out({"type": "astra.vision.state"}),
        {"kind": "wire.in", "data": {"msg": frame_msg}},
        {"kind": "note", "data": {"what": "vision.select", "from": "browser", "to": "shared"}},
    ]
    assert without_vision_frames(near) == near, (
        "B-P1: nothing but a frame or its wire.out is stripped"
    )
    assert without_vision_frames(tuple(near)) == tuple(near)


def _state_fields() -> dict[str, Any]:
    """An ``AstraVisionState``'s fields (examples from the contract's own sampler, not from the session)."""
    from local_astra.contracts.protocol import AstraVisionState
    from tests.unit._examples import example_of

    sample = example_of(AstraVisionState).model_dump(mode="json")
    sample.pop("type", None)
    return dict(sample)


def test_b_p1_a_frame_nested_as_a_dict_value_is_not_stripped() -> None:
    """B-P1 removes frames only as elements of a list or tuple: a frame held as a dict value (a slot, a ``last``
    field) is not a published element and stays, as does a ``wire.out`` record held that way."""
    from tests.unit._t0177_pins import without_vision_frames

    frame = _pin_frame()
    record = _wire_out(_logged(frame))
    tree = {
        "last": frame,
        "as_dict": frame.model_dump(mode="json"),
        "as_json": encode(frame),
        "record": record,
    }
    assert without_vision_frames(tree) == tree, "B-P1: a frame as a dict value is kept"
    assert without_vision_frames({"outer": {"last": frame}}) == {"outer": {"last": frame}}


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(
            {"type": FRAME, "step": 1, "source": "browser", "blind": False}, id="dict-missing-turn"
        ),
        pytest.param(
            {
                "type": FRAME,
                "turn_id": "t1",
                "step": 0,
                "source": "browser",
                "blind": False,
                "history_images": 0,
            },
            id="dict-step-0",
        ),
        pytest.param(
            '{"type": "astra.vision.frame", "turn_id": "t1", "step": 1, "source": "nowhere", "blind": false, '
            '"history_images": 0}',
            id="json-bad-source",
        ),
        pytest.param("model-blind-with-images", id="model-blind-with-images"),
        pytest.param("wire-out-image-without-identity", id="wire-out-image-without-identity"),
        pytest.param("wire-out-bad-kind", id="wire-out-bad-kind"),
        pytest.param("wire-out-missing-turn", id="wire-out-missing-turn"),
    ],
)
def test_b_p1_a_malformed_frame_raises_rather_than_being_discounted(bad: Any) -> None:
    from local_astra.contracts.protocol import AstraVisionFrame
    from tests.unit._t0177_pins import VisionFramePinError, without_vision_frames

    good = _logged(_pin_frame())
    if bad == "model-blind-with-images":
        # model_construct skips validation: a model that names the type but is not A1's shape
        bad = AstraVisionFrame.model_construct(
            turn_id="t1",
            step=1,
            source="browser",
            blind=True,
            images=list(_pin_frame().images),
            history_images=0,
            replay=False,
        )
    elif bad == "wire-out-image-without-identity":
        msg = {**good, "images": [{k: v for k, v in good["images"][0].items() if k != "identity"}]}
        bad = _wire_out(msg)
    elif bad == "wire-out-bad-kind":
        bad = _wire_out({**good, "images": [{**good["images"][0], "kind": "desktop"}]})
    elif bad == "wire-out-missing-turn":
        bad = _wire_out({k: v for k, v in good.items() if k != "turn_id"})
    with pytest.raises(VisionFramePinError):
        without_vision_frames([_spoken(1), bad])


def test_b_p1_is_chained_by_t0162c_observed() -> None:
    """B-P1 through the chain the older differentials call: ``_t0162c_pins.observed`` strips a frame and its
    ``wire.out`` and keeps the rest."""
    from tests.unit._t0162c_pins import observed

    frame = _pin_frame()
    keep = {"kind": "note", "data": {"what": "x"}}
    assert observed([frame.model_dump(mode="json"), keep, _wire_out(_logged(frame))]) == [keep]
