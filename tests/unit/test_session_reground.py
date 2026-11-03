"""T-0131 B1-B10 (session side): ReGround inside the live execute stage, over the fakes (docs/tasks/T-0131.md).

Every row runs ``_live_browser_support.reground_rig``: T-0111g's live session (loop on, one ``FakeLiveBrowser`` as
executor and live browser, the HUD sentinel frame) with real-JPEG frames (``jpeg_frame``), a fake LLM that tells a
pass-2 request apart (an image no frame of the browser has: the crop) and serves it from ``script_pass2``, and a
session built with ``reground=True``.

**The interfaces these rows pin** (listed in the test-author's report for the orchestrator to accept):

- ``Session(..., reground: bool = False)`` (keyword, as the accepted ``visual_tokens``); ``cli`` passes
  ``Profile.reground``/``--reground``;
- ``session.REGROUND_TIMEOUT_MS: int``, ``0 < value <= TOOL_EXEC_TIMEOUT_MS // 2`` (B3), on the session clock;
- the pass-2 request is one ``llm.generate`` with its own engine id (not the step's, not the repair's);
- session-log records: ``llm.request`` with ``data.phase == "reground"``, ``data.step`` and ``data.attempt_id``;
  ``tool.reground`` with ``data`` keys ``call_id``, ``p1``, ``p2``, ``crop`` (``{"x0", "y0", "w", "h"}``),
  ``outcome`` (a B3 label), ``ms`` and ``tool_text`` (pass 2's tool text, at most 512 characters; ``None`` when
  there was none). ``p1``/``p2`` are the implementer's shape (pass 1's and pass 2's points);
- the turn line (live, ``reground = true``): ``extras["reground"] == {"requests", "ok", "fallback", "skipped",
  "ms"}`` (orchestrator ruling: ``skipped_no_point`` is counted as ``skipped``, not under ``fallback``);
- B8 (orchestrator ruling): the final card, the payload ``action`` and step k+1's continuation show the executed
  (regrounded) arguments; the ``proposed`` card at ``LlmDone`` keeps pass 1's;
- ``telemetry.spans.SPAN_NAMES`` holds ``"tool.reground"``.

Expected values: the executed point is checked against B4's rule over the crop the session logged
(``tool_executor.to_pixels``, the executor's own formula); the pass-2 prompt against the step-1 prompt the fake
received, part by part; the crop's pixels against the frame's own JPEG at the logged rect.

Readings (reported): ``skipped_no_point`` calls (``scroll``/``press_key``/``google_search``) send no request and
write a ``tool.reground`` record with that outcome, counted as ``skipped`` and not in ``requests``; the turn
line's ``fallback`` counts every request whose outcome is not ``ok``, and ``ms`` holds one entry per request
(fallbacks included, a timeout's or an error's too); a takeover during pass 2 cancels the execute
(no ``ToolResult`` follows, as T-0111g B13's row reads it).

Every row is ``pending(task="T-0131")``; at BASE each fails on the missing ``Session(reground=...)`` keyword (a
named ``AssertionError`` raised by the rig) or on a missing name.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps),
1280x800 frames, n = 1, deterministic.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import ImagePart
from tests.unit import test_session_tool_loop as tsl
from tests.unit._hud_rig import split_thoughts
from tests.unit._live_browser_support import (
    FRESHNESS,
    HANG,
    FakeLiveBrowser,
    data_url,
    jpeg_frame,
    reground_rig,
)
from tests.unit._tool_turn_support import VALID_CLICK, answer, thought, tool, tool_rig

Q = '<|"|>'
W, H = 1280, 800
SENTINEL = "REGROUND-SENTINEL pass two voiced this."
SECRET_THOUGHT = "REGROUND-THOUGHT never shown."
ASK = "Shall I place the order for 42.99 euros? "
P1 = {"x": 500, "y": 250}


def click(x: int, y: int) -> str:
    return f"<|tool_call>call:click_at{{x:{x},y:{y}}}<tool_call|>"


def type_call(x: int, y: int, text: str = "mug", enter: bool = True) -> str:
    e = "true" if enter else "false"
    return f"<|tool_call>call:type_at{{press_enter:{e},text:{Q}{text}{Q},x:{x},y:{y}}}<tool_call|>"


SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
PRESS = f"<|tool_call>call:press_key{{combo:{Q}Enter{Q}}}<tool_call|>"
OK_PASS2 = [answer(SENTINEL), tool(click(900, 100))]


def session_mod() -> Any:
    return importlib.import_module("local_astra.conductor.session")


def reground_timeout_ms() -> int:
    s = session_mod()
    value = getattr(s, "REGROUND_TIMEOUT_MS", None)
    assert isinstance(value, int), "T-0131 B3: session.REGROUND_TIMEOUT_MS does not exist yet"
    assert 0 < value <= s.TOOL_EXEC_TIMEOUT_MS // 2, value
    return value


def frames() -> FakeLiveBrowser:
    return FakeLiveBrowser(
        [jpeg_frame("onset")], post_frames=[jpeg_frame("after-1"), jpeg_frame("after-2")]
    )


def inv_of(browser: FakeLiveBrowser, call_id: str) -> Any:
    found = [i for i in browser.invocations if i.call_id == call_id]
    assert len(found) == 1, (
        f"one execute of {call_id}; got {[i.call_id for i in browser.invocations]}"
    )
    return found[0]


def regrounds(rig: Any) -> list[dict[str, Any]]:
    return [r["data"] for r in rig.log_records if r["kind"] == "tool.reground"]


def reground_requests(rig: Any) -> list[dict[str, Any]]:
    return [
        r["data"]
        for r in rig.log_records
        if r["kind"] == "llm.request" and r["data"].get("phase") == "reground"
    ]


def crop_of(record: dict[str, Any]) -> tuple[int, int, int, int]:
    c = record["crop"]
    return int(c["x0"]), int(c["y0"]), int(c["w"]), int(c["h"])


def assert_regrounded(
    args: dict[str, Any], crop: tuple[int, int, int, int], x2: int, y2: int
) -> None:
    x0, y0, w, h = crop
    assert abs(to_pixels(args["x"], W) - (x0 + to_pixels(x2, w))) <= 1, (args, crop)
    assert abs(to_pixels(args["y"], H) - (y0 + to_pixels(y2, h))) <= 1, (args, crop)


async def one_click_turn(
    tmp_path: Path,
    pass2: list[Any],
    *,
    script: Any = None,
    steps: tuple[Any, ...] = ([answer(tsl.S3)],),
    reground: bool | None = True,
    browser: FakeLiveBrowser | None = None,
    before: Any = None,
) -> tuple[Any, FakeLiveBrowser, dict[str, Any]]:
    """One live turn: step 1 ``S1`` + ``VALID_CLICK`` (500, 250), then ``steps``; pass 2 scripted by ``pass2``.
    Returns the rig (log read back after close), the browser and the turn line's extras."""
    browser = browser if browser is not None else frames()
    async with reground_rig(
        tmp_path,
        browser,
        script=script if script is not None else [answer(tsl.S1), tool(VALID_CLICK)],
        reground=reground,
    ) as rig:
        rig.llm.script_pass2(*pass2)
        if before is not None:
            before(rig)
        rig.llm.script_steps(*steps)
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        extras = dict(rig.record("t1").extras)
    return rig, browser, extras


# =====================================================================================================
# B2/B3 ok: the regrounded point executes
# =====================================================================================================


async def test_click_regrounded_point_executed(tmp_path: Path) -> None:
    """B2/B3 ``ok`` (mutation 4): pass 2's (900, 100) in the crop is what executes, mapped by B4 over the logged
    crop (512x320 around (640, 200) px); the pass-2 request is step 1's prompt with only the call frame's image
    replaced by a 512x320 JPEG cut from that frame's bytes (R-a), under its own engine id."""
    rig, browser, _ = await one_click_turn(tmp_path, [OK_PASS2])
    assert len(rig.llm.pass2_prompts) == 1, f"one pass-2 request; got {rig.llm.pass2_ids}"
    (rec,) = regrounds(rig)
    assert rec["outcome"] == "ok", rec
    crop = crop_of(rec)
    assert crop[2:] == (512, 320) and abs(crop[0] - 384) <= 1 and abs(crop[1] - 40) <= 1, crop
    args = dict(inv_of(browser, "t1-call").arguments)
    assert_regrounded(args, crop, 900, 100)
    assert (args["x"], args["y"]) != (500, 250), (
        "T-0131 B3: ok executes the regrounded point, not pass 1's"
    )
    # the request: step 1's prompt, one image swapped
    rid, p2 = rig.llm.pass2_prompts[0]
    p1 = tsl.sent(rig, "t1-gen")
    assert rid not in {r for r, _, _ in rig.llm.generates} - {rid} and rid != "t1-gen", rid
    assert len(p2.messages) == len(p1.messages)
    swapped = []
    for m2, m1 in zip(p2.messages, p1.messages, strict=True):
        assert m2.role == m1.role and len(m2.content) == len(m1.content)
        for a, b in zip(m2.content, m1.content, strict=True):
            if a != b:
                swapped.append((a, b))
    assert len(swapped) == 1, f"T-0131 B2: exactly one part differs; {len(swapped)}"
    new, old = swapped[0]
    assert isinstance(new, ImagePart) and isinstance(old, ImagePart)
    assert old.url == data_url(browser.frames_returned[0].jpeg), (
        "the replaced image is the call's own frame"
    )
    assert p2.prefix_len == p1.prefix_len and p2.thinking == p1.thinking and p2.tools == p1.tools
    from PIL import Image

    crop_img = Image.open(io.BytesIO(base64.b64decode(new.url.split(",", 1)[1])))
    assert crop_img.format == "JPEG" and crop_img.size == (512, 320), "the crop at its native size"
    src = Image.open(io.BytesIO(browser.frames_returned[0].jpeg)).convert("RGB")
    x0, y0, _, _ = crop
    for fx, fy in ((6 * 64 + 32, 1 * 64 + 32), (10 * 64 + 32, 4 * 64 + 32)):
        a, b = crop_img.convert("RGB").getpixel((fx - x0, fy - y0)), src.getpixel((fx, fy))
        assert isinstance(a, tuple) and isinstance(b, tuple)
        assert all(abs(int(p) - int(q)) <= 12 for p, q in zip(a, b, strict=True)), (a, b)


@pytest.mark.parametrize(
    ("pass2", "outcome"),
    [
        ([[answer(SENTINEL)]], "pass2_unparseable"),
        ([[tool("<|tool_call>call:click_at{x:1200,y:5}<tool_call|>")]], "pass2_unparseable"),
        ([[tool(SCROLL)]], "pass2_no_point"),
        ([[tool(type_call(900, 100))]], "pass2_other_tool"),
        ([RuntimeError("vLLM returned HTTP 500")], "error"),
        ([HANG], "pass2_timeout"),
    ],
    ids=["no-call", "invalid", "no-point", "other-tool", "error", "timeout"],
)
async def test_each_fallback_executes_pass1(tmp_path: Path, pass2: list[Any], outcome: str) -> None:
    """B3: every fallback executes pass 1's call unchanged and is logged with its outcome; a timeout aborts the
    pass-2 engine id and still executes, inside the execute deadline."""
    if outcome == "pass2_timeout":
        reground_timeout_ms()
    rig, browser, extras = await one_click_turn(tmp_path, pass2)
    assert dict(inv_of(browser, "t1-call").arguments) == P1, (
        "T-0131 B3: the fallback executes pass 1"
    )
    (rec,) = regrounds(rig)
    assert rec["outcome"] == outcome, rec
    if outcome == "pass2_timeout":
        assert rig.llm.pass2_ids[0] in rig.llm.aborted, (
            "B3: the timed-out pass-2 request is aborted"
        )
    got = extras.get("reground", {})
    assert {k: got.get(k) for k in ("requests", "ok", "fallback", "skipped")} == {
        "requests": 1,
        "ok": 0,
        "fallback": 1,
        "skipped": 0,
    }, extras
    assert set(got) == {"requests", "ok", "fallback", "skipped", "ms"} and len(got["ms"]) == 1, (
        extras
    )


async def test_timeout_aborts_and_keeps_pass1(tmp_path: Path) -> None:
    """B3/B7: ``REGROUND_TIMEOUT_MS`` on the session clock (0 < value <= TOOL_EXEC_TIMEOUT_MS // 2); a hanging pass 2
    is aborted at the timeout, pass 1 executes before the execute deadline, and the result is not ``timeout``."""
    limit = reground_timeout_ms()
    rig, browser, _ = await one_click_turn(tmp_path, [HANG])
    starts = [t for kind, k, rid, t in rig.llm.events if kind == "start" and k == "pass2"]
    ends = [t for kind, k, rid, t in rig.llm.events if kind == "end" and k == "pass2"]
    assert starts and ends and ends[0] - starts[0] <= limit + 100, (starts, ends, limit)
    assert dict(inv_of(browser, "t1-call").arguments) == P1
    statuses = [r.get("status") for r in rig.kind("tool.result")]
    assert statuses, "premise: the call's tool.result was logged"
    assert "timeout" not in statuses, statuses


# =====================================================================================================
# B2 isolation, R-b, B5
# =====================================================================================================


async def test_pass2_never_voiced_or_stored(tmp_path: Path) -> None:
    """B2 isolation (adversarial, mutation 7): pass 2's answer sentence and thought appear in no ``TtsSpeak``, no
    audio play, no history entry, no HUD message, no ``Episode`` and no machine event; no machine event carries a
    pass-2 engine id.

    Narrowed by T-0180 (B1, R-1, B-R5): pass 2's thought may reach the HUD only as the ``text`` of an
    ``astra.thought`` with ``kind: "reground"``. Every other message, and every other field of ``astra.thought``, is
    searched as before; pass 2's answer is in no ``astra.thought`` text. Anti-vacuity: the thought **is** in a
    ``reground`` block's text."""
    pass2 = [thought(SECRET_THOUGHT), answer(SENTINEL), tool(click(900, 100))]
    rig, _, _ = await one_click_turn(tmp_path, [pass2])
    assert rig.llm.pass2_ids, "premise: pass 2 ran"
    p2ids = set(rig.llm.pass2_ids)
    for ev in rig.machine.events():
        assert getattr(ev, "request_id", None) not in p2ids, (
            f"T-0131 B2: a pass-2 event reached the machine: {ev}"
        )
    speaks = [c.source for c in rig.machine.commands(C.TtsSpeak)]
    # T-0180 B-R5: the HUD searched is every message but astra.thought's ``text`` field
    split = split_thoughts(rig.published.messages)
    hud = split.elsewhere
    history = json.dumps([h.model_dump(mode="json") for h in rig.state.history])
    episodes = repr(rig.session.memory.episodes) if hasattr(rig.session.memory, "episodes") else ""
    plays = repr(getattr(rig.audio, "plays", ""))
    for secret in (SENTINEL, SECRET_THOUGHT, "REGROUND-"):
        assert not any(secret in s for s in speaks), "voiced"
        for where, text in (
            ("HUD", hud),
            ("history", history),
            ("episode", episodes),
            ("audio", plays),
        ):
            assert secret not in text, f"T-0131 B2: pass 2's text reached the {where}"
    for step, kind, _, text in split.thoughts:
        assert SENTINEL not in text, (
            f"T-0131 B2 (narrowed by T-0180 B2): pass 2's answer reached astra.thought text ({step}, {kind})"
        )
        assert "REGROUND-" not in text.replace(SECRET_THOUGHT, ""), (
            f"T-0131 B2 (narrowed by T-0180): only pass 2's thought may be in astra.thought text; {text!r}"
        )
        if kind != "reground":
            assert SECRET_THOUGHT not in text, (
                f"T-0180 B6: pass 2's thought is shown under kind {kind!r}, not 'reground'"
            )
    assert any(SECRET_THOUGHT in text for text in split.texts(kind="reground")), (
        "T-0180 B1 anti-vacuity (B-R5): pass 2's committed thought is in a kind='reground' astra.thought text; "
        f"published thoughts {split.thoughts}"
    )


async def test_keyboard_pass2_not_executed(tmp_path: Path) -> None:
    """R-b (mutation 6): a coordinate-free pass 2 (``press_key Enter``) is observed (``pass2_no_point``) and never
    executed: the one execute is pass 1's click, unchanged."""
    rig, browser, _ = await one_click_turn(tmp_path, [[tool(PRESS)]])
    assert [i.name for i in browser.invocations] == ["click_at"], browser.invocations
    assert dict(browser.invocations[0].arguments) == P1
    assert regrounds(rig)[0]["outcome"] == "pass2_no_point"


async def test_crop_never_becomes_geometry(tmp_path: Path) -> None:
    """B5 (mutation 8): the invocation's frame is the full own frame's (1280x800, the executor's phash), never the
    crop's; the crop is in neither ``_frame_bytes`` nor ``_frame_geometry``; step 2's call reads the post-action
    frame as before."""
    browser = frames()
    rig, _, _ = await one_click_turn(
        tmp_path,
        [OK_PASS2, OK_PASS2],
        steps=([answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)]),
        browser=browser,
    )
    first = inv_of(browser, "t1-call")
    assert (first.frame.width, first.frame.height) == (W, H)
    assert first.frame.phash == browser.frames_returned[0].phash
    second = inv_of(browser, "t1-call-s2")
    assert (second.frame.width, second.frame.height) == (W, H)
    assert second.frame.phash == browser.frames_returned[1].phash, (
        "step 2's geometry is frame_after"
    )
    crops = {
        base64.b64decode(p.url.split(",", 1)[1])
        for _, prompt in rig.llm.pass2_prompts
        for m in prompt.messages
        for p in m.content
        if isinstance(p, ImagePart)
        and p.url not in {data_url(f.jpeg) for f in browser.frames_returned}
    }
    assert crops, "premise: two crops were sent"
    held = set(rig.session._frame_bytes.values())
    assert not (crops & held), "T-0131 B5: a crop was registered in _frame_bytes"
    assert all((g.width, g.height) == (W, H) for g in rig.session._frame_geometry.values())


async def test_freshness_rechecked_after_pass2(tmp_path: Path) -> None:
    """B5 (mutation 9): the newest own frame changes while pass 2 runs (a capture lands), so after pass 2 the call
    is refused with T-0111g's freshness reason and nothing is dispatched. Too-wide companion: without the change
    the same call executes."""
    rig, browser, _ = await one_click_turn(
        tmp_path, [OK_PASS2], before=lambda r: setattr(r.llm, "on_pass2", r.browser.capture_post)
    )
    assert browser.executes() == 0, (
        f"T-0131 B5: nothing dispatched after a stale pass 2; {browser.calls}"
    )
    cards = [m for m in rig.published.messages if getattr(m, "call_id", None) == "t1-call"]
    assert cards and cards[-1].status == "refused" and cards[-1].reason == FRESHNESS, cards[-1]
    _rig2, browser2, _ = await one_click_turn(tmp_path / "fresh", [OK_PASS2])
    assert browser2.executes() == 1


async def test_session_takes_no_frame_during_reground(tmp_path: Path) -> None:
    """B5: the session takes no ``frame()`` during ReGround: between the pass-2 start and the execute, the fake's
    call log holds no ``frame``."""
    _rig, browser, _ = await one_click_turn(
        tmp_path, [OK_PASS2], before=lambda r: setattr(r.llm, "on_pass2", lambda: browser_mark(r))
    )
    calls = browser.calls
    mark = calls.index(("mark",))
    execute = calls.index(("execute", "t1-call"))
    assert mark < execute and ("frame",) not in calls[mark:execute], calls


def browser_mark(rig: Any) -> None:
    rig.browser.calls.append(("mark",))


# =====================================================================================================
# B6, B7
# =====================================================================================================


async def test_no_overlap_with_step_generate(tmp_path: Path) -> None:
    """B6 (mutation 10): every pass-2 request starts after its step's generate stream ended and with no generate
    or repair of the session live (step 1 and step 2 both regrounded)."""
    rig, _, _ = await one_click_turn(
        tmp_path,
        [OK_PASS2, OK_PASS2],
        steps=([answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)]),
    )
    assert len(rig.llm.pass2_ids) == 2, rig.llm.pass2_ids
    assert rig.llm.pass2_overlaps == [(), ()], (
        f"T-0131 B6: generates live at pass-2 start {rig.llm.pass2_overlaps}"
    )
    order = [(kind, k, rid) for kind, k, rid, _ in rig.llm.events]
    first_p2 = order.index(("start", "pass2", rig.llm.pass2_ids[0]))
    # Engine ids are session-prefixed (``{session_id}-t1-gen-{n}``, ``_RequestIds``; T-0131 B6 row fix), so the
    # step generate is matched by substring.
    assert any(
        e[0] == "end" and e[1] == "generate" and "-t1-gen-" in e[2] for e in order[:first_p2]
    ), order


async def test_barge_in_aborts_pass2_no_late_result(tmp_path: Path) -> None:
    """B7 (mutation 11): a takeover while pass 2 hangs cancels the execute and aborts the pass-2 engine id; nothing
    is dispatched and no ``ToolResult`` follows."""
    browser = frames()
    async with reground_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_pass2(HANG)
        await rig.ask(tsl.QUESTION)
        await rig.until(lambda: bool(rig.llm.pass2_ids), "premise: pass 2 started", limit_ms=10_000)
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        await rig.advance(500)
        assert rig.llm.pass2_ids[0] in rig.llm.aborted, "T-0131 B7: the pass-2 engine id is aborted"
        assert browser.executes() == 0, browser.calls
        assert tsl.results_for(rig, "t1-call") == [], "no ToolResult after the cancel"


# ----------------------------------------------------------------------------- the group B review round (68cd08f)
#
# F2: a takeover captures a fresh own frame, so after a swallowed cancel the B5 freshness re-check refuses the call
# and the barge-in row above stays green (M16). These rows cancel with **no** new own frame, so only the cancel
# itself can stop the dispatch. F1: a cancel that lands while pass 2's timeout is unwinding.


async def hanging_pass2(rig: Any) -> int:
    """Ask, wait until pass 2 hangs; returns how many ``frame()`` calls the browser has served so far."""
    rig.llm.script_pass2(HANG)
    await rig.ask(tsl.QUESTION)
    await rig.until(lambda: bool(rig.llm.pass2_ids), "premise: pass 2 started", limit_ms=10_000)
    assert not any(e[0] == "end" and e[1] == "pass2" for e in rig.llm.events), (
        "premise: pass 2 still hangs"
    )
    assert "t1-call" in rig.session._executions, "premise: the execute is in flight"
    assert ("execute", "t1-call") not in rig.browser.calls, "premise: nothing dispatched yet"
    return int(rig.browser.calls.count(("frame",)))


def assert_no_new_frame(
    browser: FakeLiveBrowser, frames_before: int, *, before_dispatch_only: bool = False
) -> None:
    """Premise (F2): up to the first dispatch (or to now, when nothing was dispatched) no new own frame was taken,
    so the B5 freshness re-check had nothing to refuse on and cannot mask a swallowed cancel. A dispatch's own
    post-action capture is not a ``frame()`` call. ``before_dispatch_only``: a row whose call is settled with a
    result (the loop then legitimately frames step 2) checks only the window before a dispatch."""
    calls = browser.calls
    first = next((i for i, c in enumerate(calls) if c[0] == "execute"), None)
    if first is None and before_dispatch_only:
        return
    upto = first if first is not None else len(calls)
    taken = calls[:upto].count(("frame",))
    assert taken == frames_before, (
        f"premise (F2): a new own frame was taken while pass 2 hung; {calls}"
    )


def tool_result_records(rig: Any, call_id: str = "t1-call") -> list[dict[str, Any]]:
    return [r for r in rig.kind("tool.result") if r.get("call_id") == call_id]


async def test_tool_cancel_during_pass2_dispatches_nothing(tmp_path: Path) -> None:
    """F2 (review A at 68cd08f, kills M16): a ``tool.cancel`` (the session's command executor, ``cancel_tool``)
    while pass 2 hangs, with no new own frame captured, aborts the pass-2 engine id, dispatches nothing and
    submits no ``ToolResult``, through and past ``REGROUND_TIMEOUT_MS``."""
    limit = reground_timeout_ms()
    browser = frames()
    async with reground_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        before = await hanging_pass2(rig)
        rig.session.cancel_tool(C.ToolCancel(call_id="t1-call", reason="interrupted"))
        await rig.pump()
        await rig.advance(limit + 1_000)
        assert_no_new_frame(browser, before)
        assert rig.llm.pass2_ids[0] in rig.llm.aborted, "T-0131 B7: the pass-2 engine id is aborted"
        assert browser.executes() == 0, f"T-0131 B7: dispatched after tool.cancel; {browser.calls}"
        assert tsl.results_for(rig, "t1-call") == [], "T-0131 B7: a ToolResult after tool.cancel"
    assert tool_result_records(rig) == [], "no tool.result after the cancel (log read after close)"


async def test_turn_close_during_pass2_dispatches_nothing(tmp_path: Path) -> None:
    """F2, turn close (review A at 68cd08f, kills M16): the client disconnects while pass 2 hangs, which closes the
    turn (``_close_loop`` cancels the execution) with no new own frame captured; nothing is dispatched and no
    ``ToolResult`` is submitted, through and past ``REGROUND_TIMEOUT_MS``."""
    limit = reground_timeout_ms()
    browser = frames()
    async with reground_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        before = await hanging_pass2(rig)
        rig.submit(E.ClientDisconnected(t_ms=rig.clock.now_ms))
        await rig.pump()
        assert "t1-call" not in rig.session._executions, (
            "premise: the turn close cancelled the execution"
        )
        await rig.advance(limit + 1_000)
        assert_no_new_frame(browser, before)
        assert rig.llm.pass2_ids[0] in rig.llm.aborted, "T-0131 B7: the pass-2 engine id is aborted"
        assert browser.executes() == 0, f"T-0131 B7: dispatched after turn close; {browser.calls}"
        assert tsl.results_for(rig, "t1-call") == [], "T-0131 B7: a ToolResult after turn close"


async def test_cancel_right_after_pass2_expiry_dispatches_nothing(tmp_path: Path) -> None:
    """F1 case (a) (review A at 68cd08f, HIGH): pass 2's ``REGROUND_TIMEOUT_MS`` expires (the clock fires its
    deadline, which cancels the pass-2 task) and the execution is cancelled synchronously, before the loop runs
    again, as a turn close or ``tool.cancel`` arriving in the same instant would. The cancel must not be swallowed
    as ``pass2_timeout``: nothing is dispatched, and since a cancel settles with no result, no ``ToolResult`` and
    no ``tool.result`` record exist for the call."""
    limit = reground_timeout_ms()
    browser = frames()
    async with reground_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        before = await hanging_pass2(rig)
        execution = rig.session._executions["t1-call"]
        rig.clock.advance(limit)  # the pass-2 deadline fires now; the loop has not run since
        rig.session._cancel_execution(execution)
        await rig.pump()
        await rig.advance(1_000)
        assert_no_new_frame(browser, before)
        assert browser.executes() == 0, (
            f"T-0131 F1: the cancel was swallowed as pass2_timeout and the call dispatched; {browser.calls}"
        )
        assert tsl.results_for(rig, "t1-call") == [], "a cancel settles with no ToolResult"
    assert tool_result_records(rig) == [], (
        "no tool.result for a cancelled call (log read after close)"
    )


async def test_both_deadlines_in_one_advance_dispatch_nothing(tmp_path: Path) -> None:
    """F1 case (b) (review A at 68cd08f, HIGH): one clock advance past both deadlines (pass 2's
    ``REGROUND_TIMEOUT_MS``, then the execute's ``TOOL_EXEC_TIMEOUT_MS``) before the loop runs. The execute deadline
    settles the call as ``timeout``; the call is then never dispatched, and that ``timeout`` is the only result
    (one ``ToolResult``, one ``tool.result`` record)."""
    reground_timeout_ms()
    exec_ms = session_mod().TOOL_EXEC_TIMEOUT_MS
    browser = frames()
    async with reground_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        before = await hanging_pass2(rig)
        rig.clock.advance(exec_ms + 500)  # both deadlines fire in this one call, in time order
        await rig.pump()
        await rig.advance(1_000)
        assert_no_new_frame(browser, before, before_dispatch_only=True)
        assert browser.executes() == 0, (
            f"T-0131 F1: dispatched after the call was settled as timeout; {browser.calls}"
        )
        results = tsl.results_for(rig, "t1-call")
        assert [r.status for r in results] == ["timeout"], results
    statuses = [r.get("status") for r in tool_result_records(rig)]  # the log is read after close
    assert statuses == ["timeout"], tool_result_records(rig)


async def test_pass2_text_never_reaches_machine_events(tmp_path: Path) -> None:
    """F4 (review A at 68cd08f): pass 2's answer sentence and thought appear nowhere in what the machine received
    (``repr(rig.machine.events())``) nor in any command it emitted, so no pass-2 commit was routed through
    ``submit``. Companion premise: step 1's own answer did reach the machine (the scan can see commits)."""
    pass2 = [thought(SECRET_THOUGHT), answer(SENTINEL), tool(click(900, 100))]
    rig, _, _ = await one_click_turn(tmp_path, [pass2])
    assert rig.llm.pass2_ids, "premise: pass 2 ran"
    events = repr(rig.machine.events())
    assert tsl.S1.strip()[:20] in events, (
        "premise: step 1's commits are visible in the machine's events"
    )
    commands = repr(rig.machine.commands())
    for secret in (SENTINEL, SECRET_THOUGHT, "REGROUND-"):
        assert secret not in events, f"T-0131 B2: pass 2's text {secret!r} reached the machine"
        assert secret not in commands, (
            f"T-0131 B2: pass 2's text {secret!r} is in a machine command"
        )


@pytest.mark.parametrize(
    "pass2",
    [OK_PASS2, HANG, RuntimeError("vLLM returned HTTP 500")],
    ids=["ok", "timeout", "error"],
)
async def test_pass2_engine_id_released(tmp_path: Path, pass2: Any) -> None:
    """M11c (review A at 68cd08f): once pass 2 ends (ok, timed out or failed), its engine id is released: the
    session's live request-id table (``session._ids``, ``_RequestIds``) holds no pass-2 id, so no later abort can
    target it and no tombstone outlives it."""
    if pass2 == HANG:
        reground_timeout_ms()
    browser = frames()
    async with reground_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_pass2(pass2)
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: browser.executes() == 1, "premise: pass 1 or 2 executed", limit_ms=20_000
        )
        assert rig.llm.pass2_ids, "premise: pass 2 ran"
        live = dict(rig.session._ids._live)
        leaked = {k: v for k, v in live.items() if v in set(rig.llm.pass2_ids)}
        assert not leaked, f"T-0131 M11c: the pass-2 engine id was not released: {leaked}"


# =====================================================================================================
# B1, B10
# =====================================================================================================


@pytest.mark.parametrize("reground", [False, None], ids=["explicit-false", "default"])
async def test_reground_off_never_requests(tmp_path: Path, reground: bool | None) -> None:
    """B9/B10 (a bomb): with ``reground`` off (explicit or the default) no pass-2 request is made, no
    ``tool.reground`` record or ``reground`` phase is written, the turn line has no ``reground`` key, and pass 1
    executes."""
    rig, browser, extras = await one_click_turn(tmp_path, [OK_PASS2], reground=reground)
    assert rig.llm.pass2_prompts == [], "T-0131 B10: a pass-2 request with reground off"
    assert regrounds(rig) == [] and reground_requests(rig) == []
    assert "reground" not in extras, extras
    assert dict(inv_of(browser, "t1-call").arguments) == P1
    reground_timeout_ms()  # the seam exists (so this row is red until the implementation lands)


async def test_ask_first_never_regrounded(tmp_path: Path) -> None:
    """B10: an ask-first call (T-0107a) is never regrounded and never executed."""
    rig, browser, _ = await one_click_turn(
        tmp_path, [OK_PASS2], script=[answer(ASK), tool(VALID_CLICK)], steps=()
    )
    assert rig.llm.pass2_prompts == [] and browser.executes() == 0
    assert regrounds(rig) == [] or all(r["outcome"] != "ok" for r in regrounds(rig))
    reground_timeout_ms()


@pytest.mark.parametrize("call", [SCROLL, PRESS], ids=["scroll", "press_key"])
async def test_non_coordinate_calls_skipped(tmp_path: Path, call: str) -> None:
    """B1/B9 (ruling: ``skipped`` is its own count): ``scroll`` and ``press_key`` send no pass-2 request, are logged
    ``skipped_no_point``, execute unchanged, and the turn line counts them under ``skipped`` only."""
    rig, browser, extras = await one_click_turn(
        tmp_path, [OK_PASS2], script=[answer(tsl.S1), tool(call)]
    )
    assert rig.llm.pass2_prompts == []
    assert [r["outcome"] for r in regrounds(rig)] == ["skipped_no_point"], regrounds(rig)
    assert len(browser.invocations) == 1
    assert extras.get("reground") == {
        "requests": 0,
        "ok": 0,
        "fallback": 0,
        "skipped": 1,
        "ms": [],
    }, extras


async def test_no_reground_in_dry_run(tmp_path: Path) -> None:
    """B10: in ``dry_run`` no reground runs: a ``dry_run`` session built with ``reground=True`` either refuses at
    construction naming ``live``, or makes no pass-2 request."""
    from tests.unit import _tool_turn_support as tts

    real = tts.Session
    import inspect

    assert "reground" in inspect.signature(real.__init__).parameters, (
        "T-0131 B1: Session(..., reground=...) does not exist yet (pinned seam)"
    )

    class Regrounding(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            k.setdefault("reground", True)
            super().__init__(*a, **k)

    tts.Session = Regrounding  # type: ignore[misc]
    try:
        try:
            async with tool_rig(tmp_path, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
                await rig.ask(tsl.QUESTION)
                await rig.turn_done()
                generates = list(rig.llm.generates)
        except ValueError as exc:
            assert "live" in str(exc), exc
            return
    finally:
        tts.Session = real  # type: ignore[misc]
    assert all(rid.startswith("t1-") for rid, _, _ in generates)
    assert not any(r["kind"] == "tool.reground" for r in rig.log_records)


# =====================================================================================================
# B2/B9 logging, B8 the card
# =====================================================================================================


async def test_llm_request_phase_and_turn_line(tmp_path: Path) -> None:
    """B2/B9: the pass-2 request is logged ``llm.request`` with ``phase: "reground"``, ``step`` and
    ``attempt_id``; ``tool.reground`` carries ``p1``, ``p2``, ``crop``, ``outcome``, ``ms`` and pass 2's tool text
    truncated to 512 characters; the turn line gains
    ``reground: {requests: 1, ok: 1, fallback: 0, skipped: 0, ms: [n]}`` (``skipped`` per the B9 ruling);
    ``tool.reground`` is a span name."""
    from local_astra.telemetry import spans

    long_text = "m" * 700
    pass2 = [answer(SENTINEL), tool(type_call(900, 100, long_text))]
    rig, browser, extras = await one_click_turn(
        tmp_path,
        [pass2],
        script=[answer(tsl.S1), tool(type_call(500, 250, "mug"))],
    )
    (req,) = reground_requests(rig)
    assert req.get("step") == 1 and req.get("attempt_id"), req
    assert req["attempt_id"] == rig.llm.pass2_ids[0] or rig.llm.pass2_ids[0].endswith(
        str(req["attempt_id"])
    )
    (rec,) = regrounds(rig)
    assert {"p1", "p2", "crop", "outcome", "ms", "tool_text"} <= set(rec), sorted(rec)
    assert rec["outcome"] == "ok" and isinstance(rec["ms"], int) and rec["ms"] >= 0
    assert set(rec["crop"]) == {"x0", "y0", "w", "h"}
    assert isinstance(rec["tool_text"], str) and 0 < len(rec["tool_text"]) <= 512
    assert type_call(900, 100, long_text).startswith(rec["tool_text"]) or rec[
        "tool_text"
    ] in type_call(900, 100, long_text), "the logged text is pass 2's tool text, truncated"
    assert SENTINEL not in json.dumps(rec), "pass 2's answer text is not the tool text"
    assert extras.get("reground") == {
        "requests": 1,
        "ok": 1,
        "fallback": 0,
        "skipped": 0,
        "ms": [rec["ms"]],
    }, extras
    assert "tool.reground" in spans.SPAN_NAMES
    executed = dict(inv_of(browser, "t1-call").arguments)
    assert executed["text"] == "mug" and executed["press_enter"] is True, "pass 1's text and Enter"


async def test_card_shows_executed_point(tmp_path: Path) -> None:
    """B8 (orchestrator ruling, 2026-10-22): the final card state, the payload ``action`` and step 2's continuation
    (the replayed call and its payload) show the executed, regrounded arguments; the ``proposed`` card published at
    ``LlmDone`` keeps pass 1's point (pass 2 has not run yet). Expected strings are hand-built: the call in the
    native format ``click()`` writes, the action in the fake browser's ``clicked at (px, py)`` wording."""
    rig, browser, _ = await one_click_turn(tmp_path, [OK_PASS2])
    executed = dict(inv_of(browser, "t1-call").arguments)
    assert (executed["x"], executed["y"]) != (500, 250), "premise: the regrounded point executed"
    cards = [m for m in rig.published.messages if getattr(m, "call_id", None) == "t1-call"]
    assert any(m.status == "proposed" for m in cards), f"premise: a proposed card; {cards}"
    assert cards[-1].status == "executed", cards[-1]
    assert {k: cards[-1].arguments[k] for k in ("x", "y")} == {
        k: executed[k] for k in ("x", "y")
    }, cards[-1]
    for m in cards:
        point = (m.arguments.get("x"), m.arguments.get("y"))
        if m.status == "proposed":
            assert point == (500, 250), (
                "B8 ruling: the proposed card at LlmDone keeps pass 1's point"
            )
        else:
            assert point != (500, 250), m
    act = f"clicked at ({executed['x'] * W // 1000}, {executed['y'] * H // 1000})"  # the fake's action
    assert cards[-1].action == act
    tail = tsl.tail(tsl.step_request(rig, 2))
    assert act in tail and "clicked at (640, 200)" not in tail, tail
    assert click(executed["x"], executed["y"]) in tail, (
        "B8 ruling: step 2's continuation replays the executed call"
    )
    assert click(500, 250) not in tail, "B8 ruling: pass 1's call is not replayed"
