"""T-0201 group B through the session: the screen reminder appended to a labelled live step's continuation, behind
``Session(screen_reminder: bool = False)`` (docs/tasks/T-0201.md B1-B8).

Rigs: T-0134's live rig (``_live_browser_support.feedback_rig``: loop on, one ``FakeLiveBrowser`` as executor and
live browser, the screen label on through ``live_feedback``) and T-0177's ``vision_rig`` for the shared screen and
camera. ``tests/fakes/llm.py``'s ``is_step_prompt`` learns (additively, read off ``session.SCREEN_REMINDERS`` at call
time) that a step continuation may end with a reminder, exactly as T-0178 taught it ``REPORT_FACT``; before group B
the table does not exist and the fake is unchanged.

**Oracles.** The reminder texts and the separator are H-T201-1/H-T201-2, typed by hand in
``test_t0201_loop_reminder.py`` (``H_T201_1``, ``SEP``). The expected request is the same scenario run with
``screen_reminder=False`` (whose bytes B1 pins to BASE's through a subprocess run over ``git archive`` of BASE,
origin-proved, the T-0201 A A5/A6 pattern). Nothing is recomputed with the helper under test.

**Readings** (in the test-author's report):

- R-S1 (B1 "every existing session test"): the existing suite running unchanged is the implementer's and
  verifier's evidence; ``test_off_is_byte_identical_to_base`` adds four scenarios that reach every place B touches
  (labelled steps with a T-0171 resend; a five-step loop one char inside ``MAX_CONTINUATION``, so a room reserved
  with the keyword off would compact; a T-0172 re-ask; the report step), compared with BASE byte for byte: every
  prefill and generate wire body, every ``llm.request`` and ``note`` record, and every HUD message. Both the
  keyword absent and ``screen_reminder=False`` given.
- R-S2 (B3 ``updated`` "with or without outcome/extent clause"): one row each for the plain label, the T-0136
  outcome clause and the T-0134 extent clause.
- R-S3 (B4 "not live"; ruled in review 2026-10-29): ``screen_reminder=True`` with tools ``dry_run`` or off raises
  ``ValueError`` naming ``live`` at construction (the ``live_feedback`` precedent: a no-op is refused);
  ``screen_reminder=False`` constructs and runs in both.
- R-S4 (B4 "riding a history image"): a live step >= 2 always has a current frame (``show_current_image`` places it
  first), so the only request that rides a history image is a step-1 request, which never gets a reminder; covered
  by the step-1 row.
- R-S5 (B6 "T-0135 recovery resend"): a call recovered from an open thought sends no request of its own; the row
  pins that the recovered step sent exactly one request, with its reminder, and that the step after it carries
  only its own reminder.
- R-S6 (B7): the room is reserved on every non-report step of a session with the keyword on (the task's formula
  has no "when placed"); the rows place an ``updated`` reminder (130 chars with the separator) where the largest is
  ``unchanged`` (142), so a room of the *placed* reminder is told apart from the maximum. "The size check counts
  the reminder": a step whose continuation fits only without it, with nothing compaction may elide (two executed
  steps, ``keep_last`` 2), ends the loop ``over_budget`` with the keyword on and is sent with it off.
- R-S7 (B8 "no new note"): the set of note ``what`` values and of record kinds is the same with the keyword on and
  off over the same scenario.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``live`` (1280x800
frames, ReGround off unless a row turns it on, screen label on), loop on (8 steps unless a row sets 2), n = 1,
deterministic.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import session as session_mod
from local_astra.contracts import commands as C
from local_astra.contracts.llm import Prompt
from local_astra.contracts.perception import Source
from local_astra.toolcall import loop
from local_astra.toolcall.announced_call import REASK_NUDGE
from tests import _git_range as gr
from tests.fakes.llm import engine_empty_error, is_caption_prompt, is_reask_prompt
from tests.fakes.search import FakeSearch
from tests.unit import test_session_loop_compaction as cmp
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0142_session as t0142
from tests.unit._live_browser_support import (
    FRESHNESS,
    FakeLiveBrowser,
    feedback_rig,
    jpeg_frame,
    live_feedback,
    live_frame,
    live_kit,
    refused,
    scene_frame,
)
from tests.unit._tool_turn_support import VALID_CLICK, answer, thought, tool, with_search
from tests.unit.test_session_live import _wire, last_user
from tests.unit.test_t0177_shared_frame import VisionBrowser, select, shared_frame, vision_rig
from tests.unit.test_t0201_loop_reminder import H_T201_1, MAX_ROOM, SEP, STEM

#: BASE: main at 620b1245 (``T-0201 A merged``), where group B branched. A literal, never a moving ref (tests.md).
BASE_SHA = "620b1245"

KW = "screen_reminder"
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
CLICK2, CLICK3 = tsl.CLICK2, tsl.CLICK3
Q = tsl.Q
SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
ANN_CLICK = "I will click the Sign in button."
R_STOPPED = "The search limit was reached, so I could not search again."
REPORT = "I opened the menu and clicked the item, and stopped there."
OVER_MESSAGE = "tool loop: continuation over budget"
COMPACTED = "tool loop compacted"
RESP_CLOSE = "<tool_response|>"

#: the labels (T-0134 A3, T-0177 A7, T-0136 B-3, T-0134 A4), typed by hand from those task files
L1 = "[This screenshot is the browser screen now.]"
LU = "[This screenshot is the browser screen now, after your last action.]"
LN = "[This screenshot is the browser screen now; your last action did not change it.]"
LU_OUTCOME = (
    "[This screenshot is the browser screen now, after your last action. Compared with the previous screenshot: "
    "changed elsewhere (top right).]"
)
LU_EXTENT = (
    "[This screenshot is the browser screen now, after your last action. Page beyond this screen: about 2.0 screens "
    "below, nothing above.]"
)
SHARED_SCREEN = "[This screenshot is the user's shared screen now, not your browser; click_at and type_at cannot act on it.]"
SHARED_CAMERA = "[This image is the user's camera now, not your browser; click_at and type_at cannot act on it.]"

F0 = live_frame("t0201b-onset")
F1 = live_frame("t0201b-after-1")
F2 = live_frame("t0201b-after-2")
F3 = live_frame("t0201b-after-3")
FR = live_frame("t0201b-fresh")

ABSENT: Any = object()


# --------------------------------------------------------------------------------------------- the rig


def require_keyword() -> None:
    params = inspect.signature(session_mod.Session.__init__).parameters
    assert KW in params, "T-0201 B1: Session(..., screen_reminder: bool = False) does not exist yet"
    assert params[KW].default is False, (
        f"T-0201 B1: screen_reminder defaults to False; got {params[KW].default!r}"
    )


def label_on(page_extent: bool = False) -> Any:
    return live_feedback(screen_label=True, page_extent=page_extent)


def kwargs_of(
    on: Any, extra: dict[str, Any] | None = None, *, feedback: Any = "on"
) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    fb = label_on() if feedback == "on" else feedback
    if fb is not None:
        kw["live_feedback"] = fb
    if on is not ABSENT:
        require_keyword()
        kw[KW] = on
    kw.update(extra or {})
    return kw


@asynccontextmanager
async def live(
    tmp: Path,
    browser: Any,
    *,
    script: Any,
    on: Any = True,
    extra: dict[str, Any] | None = None,
    feedback: Any = "on",
    kit: Any = None,
    reground_llm: bool = False,
) -> AsyncIterator[Any]:
    async with feedback_rig(
        tmp,
        browser,
        script=script,
        session_kwargs=kwargs_of(on, extra, feedback=feedback),
        kit=kit,
        reground_llm=reground_llm,
    ) as rig:
        yield rig


async def run_steps(
    rig: Any, steps: tuple[Any, ...], *, n: int | None = None, limit_ms: int = 40_000
) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(tsl.QUESTION)
    want = len(steps) if n is None else n
    if want:
        await tsl.until_steps(rig, want, limit_ms=limit_ms, step=100)
    await rig.turn_done(limit_ms=limit_ms)
    await rig.advance(300)


def tail(prompt: Prompt) -> str:
    return prompt.messages[-1].text()


def attempts(rig: Any, k: int, turn: str = "t1") -> list[Prompt]:
    """Every request sent under step ``k``'s machine id that is not a re-ask, in order."""
    return [p for _, p in tsl.generates_for(rig, tsl.step_id(k, turn)) if not is_reask_prompt(p)]


def records(rig: Any, rid: str | None = None) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if rid is None or d.get("request_id") == rid]


def every_body(rig: Any) -> list[dict[str, Any]]:
    return [_wire(p) for _, p, _ in rig.llm.generates]


def text_after_current(body: dict[str, Any]) -> str | None:
    parts = last_user(body)["content"]
    idx = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
    if not idx or idx[-1] + 1 >= len(parts) or parts[idx[-1] + 1].get("type") != "text":
        return None
    return str(parts[idx[-1] + 1]["text"])


def assert_reminded(got: Prompt, key: str, what: str) -> str:
    """``got``'s continuation ends with ``SEP + H_T201_1[key]`` right after a tool response, the stem appears once in
    the whole request; returns the continuation without the suffix."""
    want = SEP + H_T201_1[key]
    t = tail(got)
    assert t.endswith(want), (
        f"T-0201 B2/B3 ({what}): the continuation ends with SEP + the {key!r} reminder; ends ...{t[-220:]!r}"
    )
    head = t[: -len(want)]
    assert head.endswith(RESP_CLOSE), (
        f"T-0201 B2 ({what}): after the last tool response; ...{head[-120:]!r}"
    )
    assert json.dumps(got.model_dump(mode="json")).count(STEM) == 1, (
        f"T-0201 B2/B5 ({what}): exactly one reminder in the request"
    )
    return head


def assert_no_reminder(rig: Any, what: str) -> None:
    for rid, p, _ in rig.llm.generates:
        assert STEM not in json.dumps(_wire(p)), f"T-0201 B4 ({what}): {rid} carries a reminder"
    for d in records(rig):
        assert KW not in d, f"T-0201 B8 ({what}): {d.get('request_id')} record has {KW}"


def no_caption_reminder(rig: Any) -> None:
    for rid, p, _ in rig.llm.generates:
        if is_caption_prompt(p):
            assert STEM not in json.dumps(_wire(p)), (
                f"T-0201 B4: the caption request {rid} carries a reminder"
            )


def three_browser() -> FakeLiveBrowser:
    """Step-1 click executes (F0 -> F1, step 2 ``updated``); the step-2 click is refused (step 3 ``unchanged``)."""
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2])
    browser.outcome("t1-call-s2", lambda i: refused(i, "bad coordinates"))
    return browser


THREE_STEPS = ([answer(S2), tool(CLICK2)], [answer(S3)])


async def three(tmp: Path, on: Any, extra: dict[str, Any] | None = None) -> Any:
    async with live(
        tmp, three_browser(), script=[answer(S1), tool(VALID_CLICK)], on=on, extra=extra
    ) as rig:
        await run_steps(rig, THREE_STEPS)
    return rig


# =====================================================================================================
# B2: the reminder after the last tool response
# =====================================================================================================


async def test_on_appends_after_last_response(tmp_path: Path) -> None:
    """B2: with the keyword on, steps 2 (``updated``) and 3 (``unchanged``) are the keyword-off requests with
    ``SEP + reminder`` appended to the continued final assistant message: every other ``Prompt`` field, every earlier
    message and part equal; on the wire only the suffix of ``messages[-1]["content"]`` differs. Step 1 is identical
    (no continuation)."""
    off = await three(tmp_path / "off", False)
    on = await three(tmp_path / "on", True)
    assert tsl.sent(on, "t1-gen") == tsl.sent(off, "t1-gen"), "T-0201 B4: step 1 is unchanged"
    for k, key, label in ((2, "updated", LU), (3, "unchanged", LN)):
        a, b = tsl.step_request(on, k), tsl.step_request(off, k)
        assert text_after_current(_wire(a)) == label, f"premise: step {k} is labelled {label}"
        assert tail(b).endswith(RESP_CLOSE), f"premise: the off step {k} ends with a tool response"
        head = assert_reminded(a, key, f"step {k}")
        assert head == tail(b), f"T-0201 B2: step {k}'s continuation before the reminder is BASE's"
        for field in Prompt.model_fields:
            if field != "messages":
                assert getattr(a, field) == getattr(b, field), f"T-0201 B2: Prompt.{field} changed"
        assert a.messages[:-1] == b.messages[:-1], f"T-0201 B2: step {k}'s earlier messages changed"
        wa, wb = _wire(a), _wire(b)
        ca, cb = wa["messages"][-1]["content"], wb["messages"][-1]["content"]
        sa = ca if isinstance(ca, str) else "".join(p.get("text", "") for p in ca)
        sb = cb if isinstance(cb, str) else "".join(p.get("text", "") for p in cb)
        assert sa == sb + SEP + H_T201_1[key], f"T-0201 B2: the wire suffix of step {k}"
        masked = copy.deepcopy(wa)
        masked["messages"][-1]["content"] = cb
        assert masked == wb, (
            f"T-0201 B2: the step-{k} wire body differs only in messages[-1].content"
        )


# =====================================================================================================
# B3: the key follows the label placed
# =====================================================================================================


async def _case_browser(tmp: Path, case: str) -> tuple[Prompt, list[dict[str, Any]]]:
    extra: dict[str, Any] = {}
    feedback: Any = "on"
    if case == "updated":
        browser = FakeLiveBrowser([F0], post_frames=[F1])
    elif case == "unchanged":
        browser = FakeLiveBrowser([F0])
        browser.outcome("t1-call", lambda i: refused(i, "bad coordinates"))
    elif case == "updated_outcome":
        w0 = scene_frame("t0201b-white")
        badge = scene_frame("t0201b-badge", blocks=[(1100, 20, 1220, 80, (0, 0, 0))])
        browser = FakeLiveBrowser([w0], post_frames=[badge])
        extra["action_outcome"] = True
    elif case == "updated_extent":
        browser = FakeLiveBrowser([F0], post_frames=[F1])
        ext = {"above_px": 0.0, "below_px": 1600.0, "viewport_px": 800.0}
        browser.extents = {f.phash: dict(ext) for f in (F0, F1)}
        feedback = label_on(page_extent=True)
    elif case == "now_new_frame":
        # T-0187: a refusal whose reason asks for a new frame; step 2 shows a fresh frame labelled LABEL_NOW
        browser = FakeLiveBrowser([F0, FR], post_frames=[F1])
        browser.outcome("t1-call", lambda i: refused(i, FRESHNESS))
    else:
        raise AssertionError(case)
    async with live(
        tmp, browser, script=[answer(S1), tool(VALID_CLICK)], extra=extra, feedback=feedback
    ) as rig:
        await run_steps(rig, ([answer(S3)],))
        s2 = tsl.step_request(rig, 2)
    return s2, records(rig, "t1-gen-s2")


async def _case_shared(tmp: Path, case: str) -> tuple[Prompt, list[dict[str, Any]]]:
    kw = kwargs_of(True)
    if case == "now_shared_refusal":
        browser = VisionBrowser([FR], post_frames=[F1])
        async with vision_rig(
            tmp, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=kw
        ) as rig:
            await select(rig, "shared")
            await run_steps(rig, ([answer(S3)],))
            s2 = tsl.step_request(rig, 2)
        return s2, records(rig, "t1-gen-s2")
    hud = None
    if case == "shared_camera":
        hud = [shared_frame("t0201b-camera", source=Source.CAMERA, width=896, height=504)]
    async with vision_rig(
        tmp, script=[answer(S1), tool(tsl.SEARCH_CALL)], hud_frames=hud, session_kwargs=kw
    ) as rig:
        await select(rig, "shared")
        await run_steps(rig, ([answer(S3)],))
        s2 = tsl.step_request(rig, 2)
    return s2, records(rig, "t1-gen-s2")


KEY_CASES = {
    "updated": (LU, "updated"),
    "unchanged": (LN, "unchanged"),
    "updated_outcome": (LU_OUTCOME, "updated"),
    "updated_extent": (LU_EXTENT, "updated"),
    "now_new_frame": (L1, "now"),
    "now_shared_refusal": (L1, "now"),
    "shared_screen": (SHARED_SCREEN, "shared_screen"),
    "shared_camera": (SHARED_CAMERA, "shared_camera"),
}


@pytest.mark.parametrize("case", sorted(KEY_CASES))
async def test_key_follows_label(tmp_path: Path, case: str) -> None:
    """B3: the step-2 reminder is the one for the label placed on that request: ``LABEL_UPDATED`` (plain, with the
    T-0136 outcome clause, with the T-0134 extent clause) -> ``updated``; ``LABEL_UNCHANGED`` -> ``unchanged``;
    T-0177 B2's ``label_now_next`` (T-0187's fresh frame after a page-change refusal, and after a shared-coordinate
    refusal) -> ``now``; the shared screen and camera labels -> ``shared_screen`` / ``shared_camera``. The step's
    ``llm.request`` record carries ``screen_reminder: <key>`` (B8)."""
    label, key = KEY_CASES[case]
    if case.startswith("shared") or case == "now_shared_refusal":
        s2, recs = await _case_shared(tmp_path, case)
    else:
        s2, recs = await _case_browser(tmp_path, case)
    assert text_after_current(_wire(s2)) == label, (
        f"premise ({case}): step 2's label is {label!r}; got {text_after_current(_wire(s2))!r}"
    )
    assert_reminded(s2, key, case)
    assert [r.get(KW) for r in recs] == [key], f"T-0201 B8 ({case}): the record's key; {recs}"


# =====================================================================================================
# B4: where it is never placed
# =====================================================================================================


async def _b4(tmp: Path, case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if case == "step1":
        rig = await three(tmp, True)
        assert STEM not in json.dumps(_wire(tsl.sent(rig, "t1-gen"))), (
            "T-0201 B4: step 1 carries a reminder"
        )
        assert all(KW not in d for d in records(rig, "t1-gen")), records(rig, "t1-gen")
        assert STEM in json.dumps(_wire(tsl.step_request(rig, 2))), (
            "premise (too wide): step 2 carries one"
        )
        no_caption_reminder(rig)
        return
    if case == "report":
        monkeypatch.setattr(tsl, "LOOP", tsl.loop_policy(2))
        browser = FakeLiveBrowser([F0], post_frames=[F1, F2])
        async with live(tmp, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
            await run_steps(rig, ([answer(S2), tool(CLICK2)], [answer(REPORT)]))
            s2, s3 = tsl.step_request(rig, 2), tsl.step_request(rig, 3)
        assert loop.REPORT_FACT in tail(s3), "premise: step 3 is the report step"
        assert text_after_current(_wire(s3)) == LU, "premise: the report step is labelled"
        assert STEM not in json.dumps(_wire(s3)), (
            "T-0201 B4 / H-T201-4: no reminder on the report step"
        )
        assert tail(s3).rstrip().endswith(loop.REPORT_FACT.strip()), "the report fact stays last"
        assert_reminded(s2, "updated", "report row, step 2 (too wide)")
        r3 = records(rig, "t1-gen-s3")
        assert r3 and all(KW not in d and d.get("report_fact") is True for d in r3), r3
        return
    if case == "labels_off":
        browser = FakeLiveBrowser([F0], post_frames=[F1])
        async with live(tmp, browser, script=[answer(S1), tool(VALID_CLICK)], feedback=None) as rig:
            await run_steps(rig, ([answer(S3)],))
            assert "[This screenshot" not in json.dumps(_wire(tsl.step_request(rig, 2))), (
                "premise: no label"
            )
        assert_no_reminder(rig, "labels off")
        return
    if case == "blind":
        browser = FakeLiveBrowser([], post_frames=[F1])
        browser.frame_mode = "raise"
        async with live(tmp, browser, script=[answer(S1), tool(SCROLL)]) as rig:
            await run_steps(rig, ([answer(S3)],))
            assert "image_url" not in json.dumps(_wire(tsl.step_request(rig, 2))), "premise: blind"
        assert_no_reminder(rig, "blind")
        return
    if case == "reground":
        browser = FakeLiveBrowser(
            [jpeg_frame("t0201b-onset")],
            post_frames=[jpeg_frame("t0201b-after-1"), jpeg_frame("t0201b-after-2")],
        )
        async with live(
            tmp,
            browser,
            script=[answer(S1), tool(VALID_CLICK)],
            extra={"reground": True},
            reground_llm=True,
        ) as rig:
            pass2 = [answer(""), tool("<|tool_call>call:click_at{x:900,y:100}<tool_call|>")]
            rig.llm.script_pass2(list(pass2), list(pass2))
            await run_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
            s2 = tsl.step_request(rig, 2)
        assert_reminded(s2, "updated", "reground row, step 2 (premise)")
        assert len(rig.llm.pass2_prompts) >= 2, (
            f"premise: pass 2 ran at steps 1 and 2; {rig.llm.pass2_prompts}"
        )
        for rid, p in rig.llm.pass2_prompts:
            assert STEM not in json.dumps(_wire(p)), (
                f"T-0201 B4: the pass-2 request {rid} carries a reminder"
            )
        no_caption_reminder(rig)
        return
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["step1", "report", "labels_off", "blind", "reground"])
async def test_no_reminder_step1_report_unlabelled_blind_reground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """B4 (with the keyword on): no reminder on step 1 (and no caption request), on the report step
    (``step == max_tool_steps + 1``, H-T201-4; its label stays and ``REPORT_FACT`` stays last), on a request with no
    placed label (labels off; a blind turn), in ReGround's pass 2 (at step 1 and at a step >= 2 whose request carries
    one); outside live it is refused (R-S3, ``test_refused_outside_live``). Each case has a too-wide companion where it applies: the same session's step 2
    carries the reminder."""
    require_keyword()
    await _b4(tmp_path, case, monkeypatch)


# =====================================================================================================
# B5: never persisted, never stored, never spoken
# =====================================================================================================


def four_browser() -> FakeLiveBrowser:
    """Step-1 click executes (F1; step 2 ``updated``), step-2 click refused (step 3 ``unchanged``), step-3 click
    executes (F2; step 4 ``updated``)."""
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2, F3])
    browser.outcome("t1-call-s2", lambda i: refused(i, "bad coordinates"))
    return browser


FOUR_STEPS = ([answer(S2), tool(CLICK2)], [answer("And the third."), tool(CLICK3)], [answer(S3)])


async def test_not_persisted_step_k_plus_1(tmp_path: Path) -> None:
    """B5 (adversarial: a reminder kept in ``turn.loop_steps`` or in the step base): over four steps every step k >= 2
    is the keyword-off step k plus *its own* reminder only, so step k+1 holds none of step k's (exactly one stem per
    request, and the continuation before it is byte-identical to the off run's)."""
    async with live(
        tmp_path / "off", four_browser(), script=[answer(S1), tool(VALID_CLICK)], on=False
    ) as off:
        await run_steps(off, FOUR_STEPS)
    async with live(tmp_path / "on", four_browser(), script=[answer(S1), tool(VALID_CLICK)]) as on:
        await run_steps(on, FOUR_STEPS)
    for k, key in ((2, "updated"), (3, "unchanged"), (4, "updated")):
        b = tsl.step_request(off, k)
        assert STEM not in json.dumps(b.model_dump(mode="json")), (
            "premise: the off run carries none"
        )
        head = assert_reminded(tsl.step_request(on, k), key, f"step {k}")
        assert head == tail(b), (
            f"T-0201 B5: step {k} is BASE step {k} plus its own reminder; the continuation before it differs\n"
            f"  base ...{tail(b)[-200:]!r}\n  got  ...{head[-200:]!r}"
        )
        assert tsl.step_request(on, k).messages[:-1] == b.messages[:-1]


async def test_reminder_never_in_history_or_spoken(tmp_path: Path) -> None:
    """B5: across a three-step labelled turn and a second turn, no ``TtsSpeak`` (text or source), ``tts.utterance``
    record, history entry, turn line, HUD message (``astra.spoken`` included), ``Episode`` or machine event carries
    the reminder's stem or a phrase of it; the requests do (premise)."""
    browser = FakeLiveBrowser([F0, F2], post_frames=[F1])
    async with live(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await run_steps(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        rig.llm.script = [answer("Still a shop. ")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        await rig.advance(300)
        sent = json.dumps(every_body(rig))
        speaks = [c.text + "|" + c.source for c in rig.machine.commands(C.TtsSpeak)]
        stores = {
            "TtsSpeak": json.dumps(speaks),
            "history": tsl.history_text(rig),
            "HUD": tsl.published_json(rig),
            "Episode": json.dumps(
                [e.model_dump(mode="json") for e in rig.memory.episodes.values()]
            ),
            "machine.events()": json.dumps([repr(e) for e in rig.machine.events()]),
            "turn lines": "\n".join(rig.lines()),
        }
    assert STEM in sent, "premise: the requests carry the reminder"
    assert speaks, "premise: something was spoken"
    stores["tts.utterance"] = json.dumps(rig.kind("tts.utterance"))
    needles = (STEM, "last image in the user's message above", "You have the current")
    for where, text in stores.items():
        for n in needles:
            assert n not in text, f"T-0201 B5: {n!r} reached {where}"


# =====================================================================================================
# B6: resends and the re-ask carry the prompt as sent
# =====================================================================================================


async def test_resends_and_reask_carry_it(tmp_path: Path) -> None:
    """B6: the T-0171 resend and the T-0179 degenerate retry of step 2 send the same prompt as the first attempt,
    reminder included, and both ``llm.request`` records carry the key; a T-0172 re-ask of step 2 is
    ``reask_prompt(<step 2 as sent>, said)``: the step's messages, then its continuation (reminder included) followed by
    the announcement as one assistant message, then the nudge; T-0135 recovery (R-S5) sends step 2 once, with its
    reminder, and step 3 carries only its own."""
    # T-0171
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    async with live(tmp_path / "empty", browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        rig.llm.fail_steps(engine_empty_error)
        await run_steps(rig, ([answer(S3)],), n=0)
        sent = attempts(rig, 2)
    assert len(sent) == 2, f"premise: step 2 was resent once; {len(sent)}"
    assert sent[0] == sent[1], "T-0201 B6: the T-0171 resend is the prompt as sent"
    assert_reminded(sent[1], "updated", "T-0171 resend")
    assert [d.get(KW) for d in records(rig, "t1-gen-s2")] == ["updated", "updated"], records(
        rig, "t1-gen-s2"
    )
    # T-0179
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    async with live(
        tmp_path / "degenerate",
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        extra={"degeneracy_guard": True},
    ) as rig:
        await run_steps(rig, ([thought(t0142.COLLAPSE)], [answer(S3)]), n=1)
        sent = attempts(rig, 2)
    notes = [d.get("what") for d in rig.kind("note")]
    assert "degenerate request: retrying once" in notes, f"premise: a T-0179 retry; {notes}"
    assert len(sent) == 2 and sent[0] == sent[1], (
        "T-0201 B6: the T-0179 retry is the prompt as sent"
    )
    assert_reminded(sent[1], "updated", "T-0179 retry")
    assert [d.get(KW) for d in records(rig, "t1-gen-s2")] == ["updated", "updated"]
    # T-0172
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    async with live(
        tmp_path / "reask",
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        extra={"reask_announced_calls": True},
    ) as rig:
        rig.llm.script_reasks([answer(R_STOPPED)])
        await run_steps(rig, ([answer(ANN_CLICK)],))
        step2 = attempts(rig, 2)
        reasks = list(rig.llm.reask_prompts)
    assert len(step2) == 1 and len(reasks) == 1, (
        f"premise: one step-2 request and one re-ask; {reasks}"
    )
    s2, re = step2[0], reasks[0][1]
    assert_reminded(s2, "updated", "the re-asked step")
    assert re.messages[: len(s2.messages) - 1] == s2.messages[:-1], (
        "T-0201 B6: the step's messages, unchanged"
    )
    assert len(re.messages) == len(s2.messages) + 1
    said = re.messages[-2]
    assert said.role == "assistant" and said.text().startswith(tail(s2)), (
        "T-0201 B6: the re-ask continues the step's continuation as sent, reminder included"
    )
    assert said.text()[len(tail(s2)) :].strip() == ANN_CLICK, said.text()[-200:]
    assert re.messages[-1].role == "user" and re.messages[-1].text() == REASK_NUDGE
    assert re.continue_final_message is False
    # T-0135 (R-S5)
    open_call = thought(f"the next link is lower down so I click it {CLICK2}", open_=True)
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2])
    async with live(
        tmp_path / "recover",
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        extra={"recover_thought_calls": True},
    ) as rig:
        await run_steps(rig, ([open_call], [answer(S3)]), n=2)
        s2_all, s3 = attempts(rig, 2), tsl.step_request(rig, 3)
        executed = [e.status for e in tsl.results_for(rig, "t1-call-s2")]
    assert executed == ["executed"], f"premise: the recovered step-2 call executed; {executed}"
    assert len(s2_all) == 1, f"R-S5: the recovered step sent one request; {len(s2_all)}"
    assert_reminded(s2_all[0], "updated", "recovered step 2")
    assert_reminded(s3, "updated", "step 3 after recovery")


# =====================================================================================================
# B7: compaction makes room; the size check counts the reminder
# =====================================================================================================

#: the budget loop: three searches, a click (so the last settled step is ``updated``), then an answer at step 5
BUDGET_STEPS = (
    [answer("Second search."), tool(cmp.search_call(2))],
    [answer("Third search."), tool(cmp.search_call(3))],
    [answer("Opening it."), tool(VALID_CLICK)],
    [answer(S3)],
)


def budget_kit(browser: Any) -> Any:
    return with_search(
        live_kit(browser), FakeSearch({cmp.query(k): cmp.rows(k) for k in (1, 2, 3)})
    )


async def budget_run(
    tmp: Path, on: Any, max_chars: int | None, monkeypatch: pytest.MonkeyPatch
) -> Any:
    if max_chars is not None:
        monkeypatch.setattr(session_mod, "MAX_CONTINUATION", max_chars)
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    try:
        async with live(
            tmp,
            browser,
            script=[answer(tsl.PREFIX), tool(cmp.search_call(1))],
            on=on,
            kit=budget_kit(browser),
        ) as rig:
            await run_steps(rig, BUDGET_STEPS, n=0, limit_ms=60_000)
    finally:
        monkeypatch.undo()
    return rig


def compactions(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if d.get("what") == COMPACTED]


async def test_budget_reserves_room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B7 (adversarial, R-S6; the T-0178 B1a pattern). ``L5`` is the keyword-off step-5 continuation (four executed
    steps, the last a click, so the placed reminder is ``updated``: 130 chars with the separator, against
    ``MAX_ROOM`` 142 for ``unchanged``). With the keyword on:

    - ``MAX_CONTINUATION = L5 + MAX_ROOM``: compaction's budget is ``L5`` exactly, nothing is elided, and step 5 is
      the off step 5 plus its reminder (an over-reserving room compacts here);
    - ``MAX_CONTINUATION = L5 + MAX_ROOM - 1``: the budget is ``L5 - 1``, so step 5 is compacted (one note, at step 5
      only) and still sent, reminder last, within ``MAX_CONTINUATION`` (a room of the *placed* reminder, ``L5 + 11``,
      or none, does not compact here; with no room at all and the size check counting it, the loop would end
      ``over_budget``).

    With the keyword off, ``MAX_CONTINUATION = L5`` compacts nothing (no room is reserved when off)."""
    require_keyword()
    base = await budget_run(tmp_path / "measure", False, None, monkeypatch)
    s5 = tsl.step_request(base, 5)
    L5 = len(tail(s5))
    assert compactions(base) == [] and base.record("t1").extras.get("tool_loop_end") == "final", (
        "premise"
    )
    assert len(SEP + H_T201_1["updated"]) == 130 and MAX_ROOM == 142, (
        "premise: the reminder lengths"
    )
    assert L5 > 4_000, f"premise: three full search responses; {L5}"
    # off, exactly at the budget
    off = await budget_run(tmp_path / "off", False, L5, monkeypatch)
    assert compactions(off) == [], (
        f"T-0201 B1/B7: no room reserved with the keyword off; {compactions(off)}"
    )
    assert tail(tsl.step_request(off, 5)) == tail(s5)
    # on, room exactly the maximum
    fits = await budget_run(tmp_path / "fits", True, L5 + MAX_ROOM, monkeypatch)
    assert compactions(fits) == [], (
        f"T-0201 B7: compaction runs against MAX_CONTINUATION - max(len(SEP + r)); at L5 + {MAX_ROOM} nothing is "
        f"elided; notes {compactions(fits)}"
    )
    assert assert_reminded(tsl.step_request(fits, 5), "updated", "fits") == tail(s5)
    assert fits.record("t1").extras.get("tool_loop_end") == "final"
    # on, one char less
    tight = await budget_run(tmp_path / "tight", True, L5 + MAX_ROOM - 1, monkeypatch)
    notes = compactions(tight)
    assert [n.get("step") for n in notes] == [5], (
        f"T-0201 B7: at L5 + {MAX_ROOM - 1} step 5 is compacted (budget L5 - 1), and only step 5; {notes}"
    )
    assert notes[0]["chars_before"] == L5 and notes[0]["elided"], notes[0]
    sent = tsl.step_request(tight, 5)
    assert_reminded(sent, "updated", "tight")
    assert len(tail(sent)) <= L5 + MAX_ROOM - 1, f"T-0201 B7: sent {len(tail(sent))} chars"
    assert tight.record("t1").extras.get("tool_loop_end") == "final", tight.record("t1").extras
    assert [d.get(KW) for d in records(tight, "t1-gen-s5")] == ["updated"]


async def test_size_check_counts_the_reminder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B7 "the size check counts the reminder" (R-S6): step 3 after two searches (nothing compaction may elide,
    ``keep_last`` 2; labelled ``unchanged``) with ``MAX_CONTINUATION = L3 + len(SEP + unchanged) - 1``: with the
    keyword on the loop ends ``over_budget`` (``LlmError(t1-gen-s3, "tool loop: continuation over budget")``, nothing
    sent for step 3); with it off step 3 is sent. At ``L3 + len(SEP + unchanged)`` the keyword-on step 3 is sent."""
    require_keyword()
    steps = ([answer("Second search."), tool(cmp.search_call(2))], [answer(S3)])

    async def run(tmp: Path, on: Any, max_chars: int | None) -> Any:
        if max_chars is not None:
            monkeypatch.setattr(session_mod, "MAX_CONTINUATION", max_chars)
        browser = FakeLiveBrowser([F0])
        try:
            async with live(
                tmp,
                browser,
                script=[answer(tsl.PREFIX), tool(cmp.search_call(1))],
                on=on,
                kit=budget_kit(browser),
            ) as rig:
                await run_steps(rig, steps, n=0, limit_ms=60_000)
        finally:
            monkeypatch.undo()
        return rig

    measure = await run(tmp_path / "measure", False, None)
    s3 = tsl.step_request(measure, 3)
    assert text_after_current(_wire(s3)) == LN, (
        "premise: step 3 after a search is labelled unchanged"
    )
    L3 = len(tail(s3))
    room = len(SEP + H_T201_1["unchanged"])
    over = await run(tmp_path / "over", True, L3 + room - 1)
    errors = [
        e.message
        for e in over.machine.events()
        if type(e).__name__ == "LlmError" and e.request_id == "t1-gen-s3"
    ]
    assert errors == [OVER_MESSAGE], (
        f"T-0201 B7: the size check counts the reminder; LlmError {errors}"
    )
    assert attempts(over, 3) == [], "T-0201 B7: step 3 is not sent over budget"
    assert over.record("t1").extras.get("tool_loop_end") == "over_budget"
    assert over.record("t1").extras.get("tool_steps") == 2, (
        "T-0201 B7: the turn line's tool_steps is the last step actually sent (2), not the refused step 3; "
        f"extras {over.record('t1').extras}"
    )
    off = await run(tmp_path / "off", False, L3 + room - 1)
    assert tail(tsl.step_request(off, 3)) == tail(s3), "the off step 3 is sent (too-wide companion)"
    edge = await run(tmp_path / "edge", True, L3 + room)
    assert assert_reminded(tsl.step_request(edge, 3), "unchanged", "edge") == tail(s3)
    assert edge.record("t1").extras.get("tool_loop_end") == "final"


async def test_report_step_room_is_the_report_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B7 with B4 (review 2026-10-29, mutant M5: the reminder's room on the report step too). Keyword on,
    ``max_tool_steps`` 4, so step 5 (three searches and a click done) is the report step: no reminder, and its room is
    ``len(REPORT_SEPARATOR + REPORT_FACT)`` (162), never the reminder's ``MAX_ROOM`` (142). With
    ``MAX_CONTINUATION = L5 + 150`` (``L5``: the step-5 continuation before the fact, measured on the same turn at
    ``max_tool_steps`` 8 with the keyword off) a room of 142 leaves ``L5`` uncompacted and the fact pushes it over
    (``over_budget``); a room of 162 compacts step 5 and sends it with ``report_fact``. The row pins the latter."""
    require_keyword()
    fact_room = len(loop.REPORT_SEPARATOR + loop.REPORT_FACT)
    assert fact_room == 162 and MAX_ROOM == 142, "premise: the two rooms"
    base = await budget_run(tmp_path / "measure", False, None, monkeypatch)
    L5 = len(tail(tsl.step_request(base, 5)))
    assert L5 - (L5 + 150 - fact_room) > 0 and L5 <= L5 + 150 - MAX_ROOM, "premise: inside the band"
    monkeypatch.setattr(tsl, "LOOP", tsl.loop_policy(4))
    monkeypatch.setattr(session_mod, "MAX_CONTINUATION", L5 + 150)
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    async with live(
        tmp_path / "report",
        browser,
        script=[answer(tsl.PREFIX), tool(cmp.search_call(1))],
        kit=budget_kit(browser),
    ) as rig:
        await run_steps(rig, BUDGET_STEPS, n=0, limit_ms=60_000)
    monkeypatch.undo()
    extras = rig.record("t1").extras
    assert extras.get("tool_loop_end") != "over_budget", (
        f"T-0201 B7/B4: the report step reserves the fact's room, not the reminder's; extras {extras}"
    )
    notes = compactions(rig)
    assert [n.get("step") for n in notes] == [5] and notes[0]["chars_before"] == L5, (
        f"T-0201 B7: the report step is compacted against MAX_CONTINUATION - {fact_room}; {notes}"
    )
    s5 = tsl.step_request(rig, 5)
    assert loop.REPORT_FACT in tail(s5) and STEM not in json.dumps(_wire(s5)), (
        "the report step: fact, no reminder"
    )
    assert len(tail(s5)) <= L5 + 150
    marked = [d.get("request_id") for d in records(rig) if d.get("report_fact") is True]
    assert marked == ["t1-gen-s5"], marked
    assert all(KW not in d for d in records(rig, "t1-gen-s5"))


@asynccontextmanager
async def _plain_rig(tmp: Path, kit: Any, on: bool) -> AsyncIterator[Any]:
    """``tool_rig`` (tools ``dry_run`` with ``kit="default"``, off with ``kit=None``) built with
    ``Session(..., screen_reminder=on)``."""
    from tests.unit import _tool_turn_support as tts

    real = tts.Session

    class WithKw(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            k.setdefault(KW, on)
            super().__init__(*a, **k)

    tts.Session = WithKw  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp, kit=kit, policy=tsl.LOOP, script=[answer(S1), tool(VALID_CLICK)]
        ) as rig:
            yield rig
    finally:
        tts.Session = real  # type: ignore[misc]


@pytest.mark.parametrize("tools", ["dry_run", "off"])
async def test_refused_outside_live(tmp_path: Path, tools: str) -> None:
    """R-S3 (ruled, review 2026-10-29; mutants M9: no guard, M18: the guard only for tools off):
    ``screen_reminder=True`` with tools ``dry_run`` or off raises ``ValueError`` naming ``live``."""
    require_keyword()
    kit: Any = "default" if tools == "dry_run" else None
    with pytest.raises(ValueError, match="live"):
        async with _plain_rig(tmp_path, kit, True):
            pass


@pytest.mark.parametrize("tools", ["dry_run", "off"])
async def test_false_outside_live_constructs(tmp_path: Path, tools: str) -> None:
    """R-S3, the too-wide companion: ``screen_reminder=False`` with tools ``dry_run`` or off builds a session that
    answers a turn (dry_run: through the tool loop's step 2), and no request carries a reminder."""
    require_keyword()
    kit: Any = "default" if tools == "dry_run" else None
    async with _plain_rig(tmp_path, kit, False) as rig:
        if tools == "dry_run":
            rig.llm.script_steps([answer(S3)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        await rig.advance(300)
        if tools == "dry_run":
            assert tsl.steps_of(rig), "premise: the dry_run loop sent a step"
    assert len(rig.lines()) == 1
    assert_no_reminder(rig, tools)


# =====================================================================================================
# B8: the record key
# =====================================================================================================


async def test_record_key_only_when_placed(tmp_path: Path) -> None:
    """B8 / R-S7: with the keyword on, ``screen_reminder`` is on the step-2 (``updated``) and step-3 (``unchanged``)
    records only, never on step 1's; with it off (and absent) on no record. No new note and no new record kind."""
    on = await three(tmp_path / "on", True)
    off = await three(tmp_path / "off", False)
    absent = await three(tmp_path / "absent", ABSENT)
    got = [(d.get("request_id"), d.get(KW)) for d in records(on) if KW in d]
    assert got == [("t1-gen-s2", "updated"), ("t1-gen-s3", "unchanged")], f"T-0201 B8: {got}"
    for name, rig in (("off", off), ("absent", absent)):
        assert [d for d in records(rig) if KW in d] == [], f"T-0201 B8 ({name}): no key"
    whats = {
        name: sorted({str(d.get("what")) for d in r.kind("note")})
        for name, r in (("on", on), ("off", off))
    }
    assert whats["on"] == whats["off"], f"T-0201 B8: no new note; {whats}"
    kinds_on = {r["kind"] for r in on.log_records}
    kinds_off = {r["kind"] for r in off.log_records}
    assert kinds_on == kinds_off, f"T-0201 B8: no new record kind; {sorted(kinds_on ^ kinds_off)}"


# =====================================================================================================
# B1: off is BASE
# =====================================================================================================


def _norm(obj: Any, session_id: str, tmp: str) -> Any:
    text = json.dumps(obj, sort_keys=True, default=str)
    return json.loads(text.replace(session_id, "<sid>").replace(tmp, "<tmp>"))


def _dump(rig: Any, tmp: Path) -> dict[str, Any]:
    sid = rig.session.session_id
    return _norm(
        {
            "prefills": [_wire(p) for _, p in rig.llm.prefills],
            "bodies": every_body(rig),
            "records": [
                {k: r.get(k) for k in ("seq", "kind", "turn_id", "session_ms", "data")}
                for r in rig.log_records
                if r["kind"] in ("llm.request", "note")
            ],
            "hud": [m.model_dump(mode="json") for m in rig.published.messages],
        },
        sid,
        str(tmp),
    )


def _base_kwargs(mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    kw: dict[str, Any] = {"live_feedback": label_on()}
    if mode == "false":
        kw[KW] = False
    kw.update(extra or {})
    return kw


async def _scenario(tmp: Path, mode: str) -> dict[str, Any]:
    """B1's four scenarios (R-S1), with ``screen_reminder`` absent (``mode="absent"``) or ``False``."""
    out: dict[str, Any] = {}
    # 1. labelled steps (updated, unchanged) with a T-0171 resend of step 2
    async with feedback_rig(
        tmp / "labels",
        three_browser(),
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs=_base_kwargs(mode),
    ) as rig:
        rig.llm.fail_steps(engine_empty_error)
        await run_steps(rig, THREE_STEPS, n=0)
    out["labels"] = _dump(rig, tmp / "labels")
    # 2. the budget loop, one char inside MAX_CONTINUATION at step 5 (measured in a first pass)
    saved = session_mod.MAX_CONTINUATION
    lengths: list[int] = []
    for name in ("measure", "budget"):
        browser = FakeLiveBrowser([F0], post_frames=[F1])
        if lengths:
            session_mod.MAX_CONTINUATION = lengths[0] + 1
        try:
            async with feedback_rig(
                tmp / name,
                browser,
                script=[answer(tsl.PREFIX), tool(cmp.search_call(1))],
                session_kwargs=_base_kwargs(mode),
                kit=budget_kit(browser),
            ) as rig:
                await run_steps(rig, BUDGET_STEPS, n=0, limit_ms=60_000)
                lengths.append(len(tail(tsl.step_request(rig, 5))))
        finally:
            session_mod.MAX_CONTINUATION = saved
    out["budget"] = _dump(rig, tmp / "budget")
    # 3. a T-0172 re-ask at step 2
    async with feedback_rig(
        tmp / "reask",
        FakeLiveBrowser([F0], post_frames=[F1]),
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs=_base_kwargs(mode, {"reask_announced_calls": True}),
    ) as rig:
        rig.llm.script_reasks([answer(R_STOPPED)])
        await run_steps(rig, ([answer(ANN_CLICK)],))
    out["reask"] = _dump(rig, tmp / "reask")
    # 4. the report step at max_tool_steps 2
    saved_loop = tsl.LOOP
    tsl.LOOP = tsl.loop_policy(2)
    try:
        async with feedback_rig(
            tmp / "report",
            FakeLiveBrowser([F0], post_frames=[F1, F2]),
            script=[answer(S1), tool(VALID_CLICK)],
            session_kwargs=_base_kwargs(mode),
        ) as rig:
            await run_steps(rig, ([answer(S2), tool(CLICK2)], [answer(REPORT)]))
    finally:
        tsl.LOOP = saved_loop
    out["report"] = _dump(rig, tmp / "report")
    return out


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0201_session_reminder as m
out = pathlib.Path(sys.argv[1])
data = asyncio.run(m._scenario(out.parent / "run", "absent"))
out.write_text(json.dumps({"origin": local_astra.__file__, "data": data}))
"""


def _base_capture(root: Path) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(root / "base_tree")
    out = root / "base_out" / "data.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _BASE_RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    got = json.loads(out.read_text())
    assert Path(got["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {got['origin']}"
    )
    data: dict[str, Any] = got["data"]
    return data


async def test_off_is_byte_identical_to_base(tmp_path: Path) -> None:
    """B1 (R-S1): ``Session(screen_reminder: bool = False)``; with the keyword absent and with ``False`` given, the
    four scenarios' prefill and generate wire bodies, ``llm.request`` and ``note`` records and HUD messages equal
    BASE's (620b1245, run in a subprocess over ``git archive`` of BASE, origin-proved). Premises: the labelled run
    resent step 2 (T-0171), the re-ask was sent, the report step carried ``REPORT_FACT``, and the budget run sent
    step 5 uncompacted one char inside its budget."""
    require_keyword()
    base = await asyncio.to_thread(_base_capture, tmp_path)
    for mode in ("absent", "false"):
        now = await _scenario(tmp_path / mode, mode)
        assert set(now) == set(base)
        for name in sorted(base):
            b, n = base[name], now[name]
            assert n == b, (
                f"T-0201 B1 ({mode}, {name}): differs from BASE; bodies {len(b['bodies'])} vs {len(n['bodies'])}, "
                f"records {len(b['records'])} vs {len(n['records'])}, hud {len(b['hud'])} vs {len(n['hud'])}"
            )
    notes = [r["data"].get("what") for r in base["labels"]["records"] if r["kind"] == "note"]
    assert "the engine returned an empty retryable response; retrying once" in notes, notes
    assert [r for r in base["reask"]["records"] if r["data"].get("phase") == "announce_reask"], (
        "premise: re-ask"
    )
    assert any("report_fact" in r["data"] for r in base["report"]["records"]), (
        "premise: the report step"
    )
    assert not [r for r in base["budget"]["records"] if r["data"].get("what") == COMPACTED], (
        "premise: uncompacted"
    )
    assert STEM not in json.dumps(base), "premise: BASE has no reminder"
