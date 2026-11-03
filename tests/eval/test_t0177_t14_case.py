"""T-0177 group A eval case (CPU): James's t14, replayed over the fakes (docs/tasks/T-0177.md, Evidence and A6/A7).

**The t14 shape.** In ``logs/sessions/20261027-230926-966483.jsonl`` the HUD posted 1568x590 ``astra.frame`` messages
(``source: screen``) at 1 Hz, the user asked about the integral on the screen, and the request carried only the agent
browser's ``capture:`` image labelled as the browser screen; the model answered "I don't see an integral on the
screen". Here:

- a live session (``vision_rig(ingress=True)``: one real ``BrowserFrameSource`` is both the session's perception and
  the server's frame sink, so HUD frames arrive only through the page), with the T-0134 screen label on (the live
  default at the time of t14);
- the page posts the committed fixture ``tests/fixtures/t0177/integral_1568x590.jpg`` (the integral from 0 to pi of
  x sin(x) dx) as ``astra.frame`` three times, 1 s apart;
- the shared arm then sends ``astra.vision.select`` ``shared``; the browser arm sends nothing;
- then ``astra.text`` "what integral is on my screen?".

**Rows.**

- ``test_the_fixture_is_the_committed_integral`` (active): the fixture's sha256 is MANIFEST.md's and it decodes as a
  1568x590 JPEG.
- ``test_t14_under_shared`` (red until A3/A6/A7/A10 land):the step-1 request carries exactly one image, the fixture's bytes (compared
  with the data URL the real ``PromptBuilder`` writes for it), followed by ``SHARED_LABEL_SCREEN``; no ``capture:``
  identity is in the turn's ``frames``; the browser's frame bytes are in no body; the ``llm.request`` record's
  ``live_label.text`` is the shared label; the line says ``vision_source: "shared"``, ``frame_source: "screen"``.
- ``test_t14_under_browser`` (green at BASE: "Under browser, BASE behaviour"): the step-1 request carries the
  browser's own capture and the browser-screen label, and the fixture's bytes are in no body.

Marked ``gpu`` + ``eval`` although CPU only: every row under ``tests/eval`` is deselected by a path-only run (T-0046,
``tests/unit/test_pytest_config.py``). It spends no GPU time, so select it by the ``eval`` marker (``-m gpu`` also
selects it, but agents' ``-m gpu`` commands are gated behind ``/launch``)::

    uv run pytest tests/eval/test_t0177_t14_case.py -q -m eval -rxX

Regime: CPU over fakes, ``ScriptedClock``, thinking on, tools live, loop on (8 steps), T-0134 screen label on, a
typed turn, 1280x800 browser frames, the 1568x590 quality-85 fixture, n = 1, deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.codec import encode
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocol import AstraFrame, AstraText
from tests.unit._live_browser_support import (
    b64,
    builder_url,
    live_feedback,
    turn_extras,
)
from tests.unit._tool_turn_support import answer
from tests.unit.test_session_live import last_user
from tests.unit.test_t0177_shared_frame import (
    F0,
    SHARED_LABEL_SCREEN,
    gen_body,
    select,
    vision_rig,
)

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "t0177" / "integral_1568x590.jpg"
MANIFEST = FIXTURE.parent / "MANIFEST.md"
QUESTION = "what integral is on my screen?"
WIDTH, HEIGHT = 1568, 590


def fixture_frame() -> tuple[AstraFrame, Frame]:
    jpeg = FIXTURE.read_bytes()
    phash = hashlib.sha256(jpeg).hexdigest()[:16]
    message = AstraFrame(
        source=Source.SCREEN,
        ts_ms=0,
        width=WIDTH,
        height=HEIGHT,
        jpeg_b64=b64(jpeg),
        phash=phash,
        title="worksheet-3.pdf",
        app="firefox",
    )
    frame = Frame(source=Source.SCREEN, ts_ms=0, width=WIDTH, height=HEIGHT, jpeg=jpeg, phash=phash)
    return message, frame


async def t14(tmp: Path, *, shared: bool) -> dict[str, Any]:
    message, frame = fixture_frame()
    async with vision_rig(
        tmp,
        script=[answer("It is the integral from 0 to pi of x sin x. ")],
        ingress=True,
        session_kwargs={"live_feedback": live_feedback(screen_label=True)},
    ) as rig:
        for _ in range(3):
            before = rig.server.counts["frames_rejected"]
            rig.page.receive(encode(message))
            await rig.pump()
            assert rig.server.counts["frames_rejected"] == before, (
                "premise: the HUD source accepted the fixture frame"
            )
            await rig.advance(1000)
        if shared:
            await select(rig, "shared")
        rig.page.receive(encode(AstraText(text=QUESTION)))
        await rig.pump()
        await rig.turn_done()
        body = gen_body(rig)
        bodies = json.dumps(rig.llm.bodies)
        extras = turn_extras(rig)
    return {
        "frame": frame,
        "body": body,
        "bodies": bodies,
        "extras": extras,
        "llm_request": [d for d in rig.kind("llm.request") if d.get("request_id") == "t1-gen"],
    }


def _texts_after_image(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    parts = last_user(body)["content"]
    idx = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
    following = [str(parts[i + 1].get("text")) if i + 1 < len(parts) else "<end>" for i in idx]
    return [parts[i]["image_url"]["url"] for i in idx], following


def test_the_fixture_is_the_committed_integral() -> None:
    """The fixture is MANIFEST.md's file (sha256) and decodes as a 1568x590 JPEG."""
    from PIL import Image

    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    listed = re.findall(r"\*\*sha256\*\* `([0-9a-f]{64})`", MANIFEST.read_text())
    assert listed == [digest], f"MANIFEST.md lists {listed}; the file is {digest}"
    with Image.open(FIXTURE) as img:
        assert (img.format, img.size) == ("JPEG", (WIDTH, HEIGHT))


async def test_t14_under_shared(tmp_path: Path) -> None:
    """The t14 case fixed (A6, A7, A10): under ``shared`` the request carries the integral and the shared label, no
    agent-browser capture."""
    got = await t14(tmp_path, shared=True)
    urls, following = _texts_after_image(got["body"])
    assert urls == [builder_url(got["frame"])], f"A6: one image, the shared fixture; {len(urls)}"
    assert following == [SHARED_LABEL_SCREEN], f"A7: the shared label follows it; {following}"
    assert "browser screen" not in json.dumps(last_user(got["body"])), "A7: no browser wording"
    frames = got["extras"].get("frames") or []
    assert not [f for f in frames if str(f).startswith("capture:")], (
        f"t14: no agent-browser capture in the turn; frames {frames}"
    )
    assert b64(F0.jpeg) not in got["bodies"], "A6: the browser's frame reached a body"
    assert got["llm_request"] and all(
        (d.get("live_label") or {}).get("text") == SHARED_LABEL_SCREEN for d in got["llm_request"]
    ), got["llm_request"]
    assert (got["extras"].get("vision_source"), got["extras"].get("frame_source")) == (
        "shared",
        "screen",
    ), got["extras"]


async def test_t14_under_browser(tmp_path: Path) -> None:
    """Under ``browser`` (no select), BASE behaviour, the shape of t14 itself: the request carries the agent
    browser's own capture labelled as the browser screen, and the integral's bytes are in no body. Green at BASE."""
    got = await t14(tmp_path, shared=False)
    urls, following = _texts_after_image(got["body"])
    assert urls == [builder_url(F0)], f"the browser's own capture; {len(urls)} images"
    assert following and "browser screen now" in following[0], following
    assert b64(got["frame"].jpeg) not in got["bodies"], "the shared fixture reached a body"
    frames = got["extras"].get("frames") or []
    assert [f for f in frames if str(f).startswith("capture:")], frames
