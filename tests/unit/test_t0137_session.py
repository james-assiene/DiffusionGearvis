"""T-0137 group B (session side): B-1..B-8 over the fakes (docs/tasks/T-0137.md).

**The surface these rows pin** (the task's names; where it is silent the reading is stated here and in the report):

- ``Session(..., repeat_guard=True)`` (keyword; absent means off, B-9 "off passes no keyword"). ``ValueError`` unless
  the kit's mode is ``live``; never refused beside ReGround, the auto overview, the page overview or the overview
  frame (B-1).
- The screen key (B-2) is the **executor phash** of the live frame a step's request showed, never the capture
  identity. The rows script captures whose executor phash and bytes are set independently (``live_frame(tag,
  phash=...)``): equal phash with different bytes is one screen that B8 reports ``updated``; equal bytes are the same
  capture, which B8 reports ``unchanged`` (T-0111g review A1). ``FakeLiveBrowser`` already scripts both (Q-1: no fake
  capability had to be added; its ``_own`` map is keyed by phash and accepts several captures under one phash).
- The repeat fact (B-6): an executed repeat's payload is BASE's plus ``repeat_of_step: {"step": j, "action": <step
  j's action string>, **<step j's outcome>}``, rendered in the continuation by ``loop.render_response`` (keys sorted);
  the expected payloads are written out by hand and the rendered fact is also checked against the literal shape of
  the B-6 example (``repeat_of_step:{action:<|"|>...<|"|>,screenshot:<|"|>...<|"|>,step:j}``).
- The stop (B-5): refused before ``execute``; the card ``refused`` with the reason; the loop result
  ``{**status_payload("refused", REASON), "repeat_of_step": {...}}`` (the key only for ``stop == "repeat"``), then
  B8's ``screenshot: "unchanged"``.
- Logging (B-8): the turn line's ``repeat_guard`` key (``TurnRecord.extras``) with the five int counts; the
  ``tool.call`` records' ``repeat_of_step: j`` and ``no_progress: "repeat" | "recurring"``.

**Readings** (flagged in the report, not resolved here):

- **S-R1 (the stop's fact).** B-5 says ``"repeat_of_step": {...}``; the rows read it as B-6's mapping for the latest
  instance (``{"step": j, "action": ..., <j's outcome>}``; "Tests to write" shows ``{...step:3...}`` at step 4).
- **S-R2 (records).** "the call's ``tool.call`` records gain ...": a call has several records (proposed, then its
  final state); the guard decides after the proposed record, so the rows pin the key on the call's **last** record
  and its absence from every record of every other call (T-0151's R5 reading).
- **S-R3 (turn-line counts).** B-8 names ``repeats`` and ``no_change_repeats`` without defining them. The rows pin
  them only on scripts where every reading agrees (one executed repeat that changed the screen: ``repeats == 1``,
  ``no_change_repeats == 0``); ``stops_repeat``, ``stops_recurring`` and ``max_screen_visits`` are pinned where they
  are unambiguous.
- **S-R4 (S-OUT-1 key, Q-3).** O-1 fixes T-0136's contract as ``_TurnLog.action_outcomes: dict[int, Outcome]`` with no
  change ``kind == "none"`` / ``.text == "no visible change"``, and the settled payload unchanged. S-OUT-1's
  "``outcome`` holds T-0136's key with its value" therefore names no key: the row accepts exactly one key besides
  ``step``, ``action`` and ``screenshot`` in the fact, whose value is ``"no visible change"``. Q-3, pinned to T-0136's
  merge **dd637f0** (``Merge T-0136 group B``): ``session._TurnLog.action_outcomes``, ``action_outcome.Outcome.kind ==
  "none"`` and ``action_outcome.NO_VISIBLE_CHANGE == "no visible change"``. The row reaches T-0136's measurement by
  monkeypatching ``action_outcome.outcome_of_jpeg`` (every binding, ``patch_bound``) to return ``Outcome(kind="none",
  ...)`` and reads ``_TurnLog.action_outcomes`` through T-0136 B's ``TurnLogSpy`` (``turn_logs``) as its premise.
- **S-R5 (compaction).** "a continuation budget small enough that compaction elides step 1-2 responses": the row
  monkeypatches ``session.MAX_CONTINUATION`` to 2470 chars and makes every action string long (the executor's action
  plus a fixed padding), so the real ``loop.compact`` elides steps 1 and 2 before step 6 is judged (computed with
  ``loop.continuation_chars`` over the hand-written payloads; the row asserts that premise from the step-6 request).

Expected values are hand-written payloads rendered through ``loop.step_prompt`` (T-0113's oracle, proven against the
template; ``tests.unit.test_session_live.want_tail``), the reason strings are typed from A-1, never imported.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on, T-0134c's label on, echo ``model``, ReGround off unless a row says so; n = 1,
deterministic. BASE_B = 4382224 (see ``test_t0137_differential``).

Activated for ``/implement T-0137`` group B (no pending marker): at BASE_B every row fails on a named
``AssertionError`` (the ``repeat_guard`` keyword does not exist), never on a ``TypeError``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.toolcall import loop
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0151_session as s151
from tests.unit._live_browser_support import (
    FRESHNESS,
    FakeLiveBrowser,
    failed,
    feedback_rig,
    jpeg_frame,
    live_feedback,
    live_frame,
    live_kit,
    refused,
    reground_rig,
    session_mod,
    turn_logs,
)
from tests.unit._tool_turn_support import answer, patch_bound, tool
from tests.unit.test_session_live import card, next_turn, step_tail, want_tail
from tests.unit.test_t0140_session import overview_kit

Q = '<|"|>'
S = [f"Step {i} text." for i in range(1, 10)]
REPORT = "Done, the button is pressed."
#: A-1's reasons, typed from the task (never imported from ``progress``)
REPEAT_REASON = (
    "the same action did not change this screen the last two times; it was not done again"
)
RECURRING_REASON = (
    "this screen has come back for the third time in this request; the action was not done"
)
GUARD_KEYS = (
    "repeats",
    "no_change_repeats",
    "stops_repeat",
    "stops_recurring",
    "max_screen_visits",
)
STUCK = "I can't get past this button; which one should I use?"


def click(x: int, y: int) -> str:
    return f"<|tool_call>call:click_at{{x:{x},y:{y}}}<tool_call|>"


def type_call(x: int, y: int, text: str, enter: bool = False) -> str:
    e = "true" if enter else "false"
    return f"<|tool_call>call:type_at{{press_enter:{e},text:{Q}{text}{Q},x:{x},y:{y}}}<tool_call|>"


def press(combo: str) -> str:
    return f"<|tool_call>call:press_key{{combo:{Q}{combo}{Q}}}<tool_call|>"


SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
CLICK = click(500, 250)
CLICK_ARGS = {"x": 500, "y": 250}
#: the fake's model-unit action string for CLICK (echo "model"; T-0134 A2)
ACT = "clicked at (500, 250)"

#: one screen (executor phash ``P``): the onset capture, the same capture again (identical bytes: B8 ``unchanged``),
#: and captures of that screen with other bytes (a caret, a hover, a cart badge: B8 ``updated``)
P = "7137713771377137"
F0 = live_frame("onset", phash=P)


def same() -> Any:
    """The onset capture again: identical bytes, so the same capture identity (B8 ``unchanged``)."""
    return F0.model_copy()


def other_bytes(tag: str, phash: str = P) -> Any:
    """A capture of the screen ``phash`` whose bytes differ (B8 ``updated``; same executor phash)."""
    return live_frame(f"{phash}-{tag}", phash=phash)


# --------------------------------------------------------------------------------------------- seams


def require_keyword(*names: str) -> None:
    params = inspect.signature(session_mod().Session.__init__).parameters
    missing = [n for n in names if n not in params]
    assert not missing, f"T-0137 B-1: Session(..., {', '.join(missing)}=...) does not exist yet"


def label_on() -> Any:
    return live_feedback(screen_label=True, page_extent=False)


def guard_rig(
    tmp: Path,
    browser: Any,
    *,
    script: Any,
    on: bool = True,
    extra: dict[str, Any] | None = None,
    kit: Any = None,
    machine_cls: Any = None,
) -> Any:
    """``feedback_rig`` with ``Session(..., live_feedback=<label on>, repeat_guard=True)`` when ``on`` (off: no
    keyword). The keyword's absence is a named ``AssertionError``."""
    if on:
        require_keyword("repeat_guard")
    kw: dict[str, Any] = {"live_feedback": label_on()}
    if on:
        kw["repeat_guard"] = True
    kw.update(extra or {})
    return feedback_rig(
        tmp, browser, script=script, session_kwargs=kw, kit=kit, machine_cls=machine_cls
    )


async def run(rig: Any, steps: tuple[Any, ...], *, limit_ms: int = 30_000) -> None:
    if steps:
        rig.llm.script_steps(*steps)
    await rig.ask(tsl.QUESTION)
    await tsl.until_steps(rig, len(steps), limit_ms=limit_ms, step=100)
    await rig.turn_done(limit_ms=limit_ms)
    await rig.advance(300)


def browser_of(post: list[Any], *, frames: Any = None) -> FakeLiveBrowser:
    return FakeLiveBrowser(
        list(frames if frames is not None else [F0]), post_frames=post, echo="model"
    )


def ex(action: str, updated: bool, fact: dict[str, Any] | None = None) -> dict[str, Any]:
    """An executed step's settled payload, by hand: BASE's ``{action, status}`` plus B8's ``screenshot`` (plus B-6's
    fact when given)."""
    out: dict[str, Any] = {
        "action": action,
        "screenshot": "updated" if updated else "unchanged",
        "status": "executed",
    }
    if fact is not None:
        out["repeat_of_step"] = fact
    return out


def fact(step: int, action: str, screenshot: str, **more: str) -> dict[str, Any]:
    return {"action": action, "screenshot": screenshot, "step": step, **more}


def refusal(reason: str, fact_: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"reason": reason, "screenshot": "unchanged", "status": "refused"}
    if fact_ is not None:
        out["repeat_of_step"] = fact_
    return out


def rendered_fact(step: int, action: str, screenshot: str) -> str:
    """B-6's rendered example shape, typed by hand (keys in sorted order)."""
    return f"repeat_of_step:{{action:{Q}{action}{Q},screenshot:{Q}{screenshot}{Q},step:{step}}}"


_ENTRY = re.compile(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|(-?\d+))')


def facts_in(text: str) -> list[dict[str, Any]]:
    """Every ``repeat_of_step:{...}`` mapping rendered in ``text``, parsed (flat entries: quoted strings or ints)."""
    out = []
    for body in re.findall(r"repeat_of_step:\{(.*?)\}", text):
        entry: dict[str, Any] = {}
        for k, s, n in _ENTRY.findall(body):
            entry[k] = s if n == "" else int(n)
        out.append(entry)
    return out


def statuses(rig: Any, cid: str) -> list[str]:
    return [e.status for e in tsl.results_for(rig, cid)]


def records_of(rig: Any, cid: str) -> list[dict[str, Any]]:
    """The ``tool.call`` records of ``cid``. The rig fills ``log_records`` only after ``session.aclose()``, so this is
    read after the ``async with`` block; an empty log is a named failure (a read inside the block is vacuous)."""
    assert rig.log_records, "test bug: the session log is read back only after the rig closes"
    return [r for r in rig.tool_records() if r.get("call_id") == cid]


def all_records(rig: Any) -> list[dict[str, Any]]:
    assert rig.log_records, "test bug: the session log is read back only after the rig closes"
    return list(rig.tool_records())


def guard_line(rig: Any, turn: str = "t1") -> Any:
    return dict(rig.record(turn).extras).get("repeat_guard")


def check_line_shape(line: Any) -> dict[str, int]:
    assert isinstance(line, dict) and set(line) == set(GUARD_KEYS), (
        f"T-0137 B-8: the turn line's repeat_guard has exactly {GUARD_KEYS}; got {line}"
    )
    assert all(type(v) is int for v in line.values()), f"B-8: ints; got {line}"
    return line


def executes(browser: FakeLiveBrowser, cid: str) -> int:
    return browser.calls.count(("execute", cid))


def dispatched(browser: FakeLiveBrowser) -> int:
    """How many of the model's calls reached the executor (the T-0151 overview capture, ``*-overview-frame``, is the
    session's own and not counted)."""
    return sum(1 for c in browser.calls if c[0] == "execute" and "-call" in str(c[1]))


def steps_of(texts_calls: list[tuple[str, str]]) -> tuple[list[Any], ...]:
    return tuple([answer(t), tool(c)] if c else [answer(t)] for t, c in texts_calls)


def machine_trace(rig: Any) -> list[dict[str, Any]]:
    """Every command the machine emitted for a non-``Tick`` event, as data, without the clock fields (``*_ms``).
    Tick-driven commands are left out: a refusal settled without an executor await may land one pump earlier than an
    executor's refusal, which moves a Tick's commands without any behavioural difference."""
    out = []
    for ev, _, cmds in rig.machine.log:
        if isinstance(ev, E.Tick):
            continue
        for c in cmds:
            d = c.model_dump(mode="json")
            out.append({k: v for k, v in d.items() if not k.endswith("_ms")})
    return out


def speaks_after(rig: Any, cid: str) -> list[Any]:
    """The ``TtsSpeak`` commands the machine emitted after it was fed ``ToolResult(cid)``."""
    seen, out = False, []
    for ev, _, cmds in rig.machine.log:
        if isinstance(ev, E.ToolResult) and ev.call_id == cid:
            seen = True
            continue
        if seen:
            out.extend(c for c in cmds if isinstance(c, C.TtsSpeak))
    assert seen, f"premise: the machine was fed ToolResult({cid})"
    return out


# =====================================================================================================
# B-1: the keyword
# =====================================================================================================


async def test_session_refusal(tmp_path: Path) -> None:
    """B-1: ``Session(repeat_guard=True)`` raises ``ValueError`` with a ``dry_run`` kit and with tools off; it is not
    refused in ``live`` beside ReGround, the auto overview, the page overview (the overview kit) or the overview frame
    (each built and started over the fakes)."""
    require_keyword("repeat_guard")
    from tests.unit import _tools_switch_support as T
    from tests.unit._tool_turn_support import tool_rig

    real = session_mod().Session
    for name, kit in (("dry_run", T.make_kit()), ("off", None)):

        class Guarded(real):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                k.setdefault("repeat_guard", True)
                super().__init__(*a, **k)

        from tests.unit import _tool_turn_support as tts

        tts.Session = Guarded  # type: ignore[misc]
        try:
            with pytest.raises(ValueError):
                async with tool_rig(tmp_path / name, kit=kit):
                    pass
        finally:
            tts.Session = real  # type: ignore[misc]
    combos: dict[str, tuple[dict[str, Any], Any]] = {
        "reground": ({"reground": True}, None),
        "auto_overview": ({"auto_overview": True}, None),
        "page_overview": ({}, "overview"),
        "overview_frame": ({"overview_frame": True}, None),
    }
    for name, (kw, kit) in combos.items():
        browser = browser_of([])
        made_kit = overview_kit(browser) if kit == "overview" else live_kit(browser)
        async with guard_rig(
            tmp_path / name, browser, script=[answer(REPORT)], extra=kw, kit=made_kit
        ) as rig:
            assert rig.session.tool_kit.mode == "live", name


# =====================================================================================================
# B-6: the repeat fact
# =====================================================================================================


async def test_repeat_fact_after_no_change(tmp_path: Path) -> None:
    """B-6 / B-3 S-OUT-2: a step-1 click on screen P; its post-action capture is the same capture (B8 ``unchanged``).
    At step 2 the same click executes (the executor runs twice) and its payload gains ``repeat_of_step: {step: 1,
    action: "clicked at (500, 250)", screenshot: "unchanged"}``. The step-3 continuation is exactly the hand-written
    steps rendered (step 1's payload has no fact) and holds B-6's rendered shape inside step 2's response."""
    browser = browser_of([same(), same()])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], "")]))
        assert executes(browser, "t1-call") == 1 and executes(browser, "t1-call-s2") == 1, (
            browser.calls
        )
        want = [
            (S[0], "click_at", CLICK_ARGS, ex(ACT, False)),
            (S[1], "click_at", CLICK_ARGS, ex(ACT, False, fact(1, ACT, "unchanged"))),
        ]
        got = step_tail(rig, 3)
        assert got == want_tail(rig, want), (
            f"T-0137 B-6: the step-3 continuation; got ...{got[-500:]}"
        )
        assert got.count(rendered_fact(1, ACT, "unchanged")) == 1, got[-500:]
        assert "repeat_of_step" not in step_tail(rig, 2), (
            "B-6: step 1 is the original, not a repeat"
        )


async def test_repeat_fact_over_click(tmp_path: Path) -> None:
    """B-6 / B-2 (the over-click; reviewer mutation 5, the capture identity as the screen key): the step-1 click
    changes the bytes but not the executor phash (a cart badge), so B8 says ``updated``. The step-2 identical click
    gets ``repeat_of_step`` with ``screenshot: "updated"`` and is not stopped (it executes)."""
    browser = browser_of([other_bytes("badge-1"), other_bytes("badge-2")])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], "")]))
        assert statuses(rig, "t1-call-s2") == ["executed"], "B-6: the over-click is not stopped"
        assert executes(browser, "t1-call-s2") == 1
        want = [
            (S[0], "click_at", CLICK_ARGS, ex(ACT, True)),
            (S[1], "click_at", CLICK_ARGS, ex(ACT, True, fact(1, ACT, "updated"))),
        ]
        got = step_tail(rig, 3)
        assert got == want_tail(rig, want), (
            "T-0137 B-2/B-6: equal executor phash with different bytes is the same screen, so the step-2 click "
            f"is a repeat of step 1 with screenshot updated; got ...{got[-500:]}"
        )
        assert rendered_fact(1, ACT, "updated") in got


async def test_no_fact_for_failed_repeat(tmp_path: Path) -> None:
    """B-6: a failed repeat gets no fact (its payload is BASE's ``status_payload("failed", reason)`` plus B8's
    ``screenshot``, which after a fresh frame of another capture says ``updated``)."""
    browser = browser_of([same()], frames=[F0, other_bytes("fresh")])
    browser.outcome("t1-call-s2", lambda inv: failed(inv, "TimeoutError during click_at"))
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], "")]))
        assert statuses(rig, "t1-call-s2") == ["failed"], "premise"
        got = step_tail(rig, 3)
        assert "repeat_of_step" not in got, (
            f"T-0137 B-6: a failed repeat gets no fact; ...{got[-400:]}"
        )
    recs = records_of(rig, "t1-call-s2")
    assert recs, "premise: the failed call has tool.call records"
    assert not [r for r in recs if "repeat_of_step" in r], recs


# =====================================================================================================
# B-5: the stop
# =====================================================================================================


async def test_stop_after_two_no_change_repeats(tmp_path: Path) -> None:
    """B-5 (adversarial; the mutant dispatches anyway): four identical clicks on an unchanging screen (every
    post-action capture is the same capture). The executor is invoked exactly 3 times; step 4 is refused before
    ``execute`` (nothing dispatched, no frame taken); its response is ``{reason: REPEAT_REASON, repeat_of_step:
    {action, screenshot: "unchanged", step: 3}, screenshot: "unchanged", status: "refused"}`` (S-R1); the card is
    ``refused`` with that reason; the call's last ``tool.call`` record has ``no_progress: "repeat"`` (S-R2); the turn
    line counts ``stops_repeat == 1``, ``stops_recurring == 0`` and ``max_screen_visits == 1``."""
    browser = browser_of([same(), same(), same()])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], CLICK), (REPORT, "")]))
        assert dispatched(browser) == 3, (
            f"T-0137 B-5: the fourth identical click was dispatched; {browser.calls}"
        )
        assert executes(browser, "t1-call-s4") == 0
        assert statuses(rig, "t1-call-s4") == ["refused"], statuses(rig, "t1-call-s4")
        assert browser.calls.count(("frame",)) == 1, (
            f"B-5: nothing captured for the stop; {browser.calls}"
        )
        want = [
            (S[0], "click_at", CLICK_ARGS, ex(ACT, False)),
            (S[1], "click_at", CLICK_ARGS, ex(ACT, False, fact(1, ACT, "unchanged"))),
            (S[2], "click_at", CLICK_ARGS, ex(ACT, False, fact(2, ACT, "unchanged"))),
            (S[3], "click_at", CLICK_ARGS, refusal(REPEAT_REASON, fact(3, ACT, "unchanged"))),
        ]
        got = step_tail(rig, 5)
        assert got == want_tail(rig, want), (
            f"T-0137 B-5: the step-5 continuation; got ...{got[-700:]}"
        )
        c = card(rig, "t1-call-s4")
        assert (c.status, c.reason) == ("refused", REPEAT_REASON), c
    recs = records_of(rig, "t1-call-s4")
    assert recs and recs[-1].get("no_progress") == "repeat", recs
    others = [r for r in all_records(rig) if r.get("call_id") != "t1-call-s4"]
    assert others and not [r for r in others if "no_progress" in r], others
    line = check_line_shape(guard_line(rig))
    assert (line["stops_repeat"], line["stops_recurring"], line["max_screen_visits"]) == (
        1,
        0,
        1,
    ), line


async def test_changed_screen_never_stops(tmp_path: Path) -> None:
    """B-3 consistency rule (the too-wide direction): four identical clicks whose captures each change the bytes of
    the same screen (B8 ``updated`` every time, the model was told the screen changed) all execute: no stop, and each
    repeat's fact says ``updated``."""
    browser = browser_of([other_bytes(f"v{i}") for i in range(1, 5)])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], CLICK), (REPORT, "")]))
        assert dispatched(browser) == 4, (
            f"T-0137 B-3: a click the model was told changed the screen was counted as no change; {browser.calls}"
        )
        got = step_tail(rig, 5)
        assert [f["step"] for f in facts_in(got)] == [1, 2, 3], facts_in(got)
        assert all(f["screenshot"] == "updated" for f in facts_in(got)), facts_in(got)
        assert REPEAT_REASON not in got and RECURRING_REASON not in got
    line = check_line_shape(guard_line(rig))
    assert (line["stops_repeat"], line["stops_recurring"]) == (0, 0), line


async def test_stop_recurring(tmp_path: Path) -> None:
    """B-5 (recurring; reviewer mutation 5 on the stop path): scripted screens A, B, A, B (each visit a capture with
    new bytes but its screen's executor phash), then a fifth call read off A (a click that repeats nothing) is refused
    with ``RECURRING_REASON``, dispatches nothing and carries no ``repeat_of_step``; its last ``tool.call`` record has
    ``no_progress: "recurring"``; the turn line has ``stops_recurring == 1`` and ``max_screen_visits == 3``."""
    pa, pb = "a1a1a1a1a1a1a1a1", "b2b2b2b2b2b2b2b2"
    onset = live_frame("A-0", phash=pa)
    post = [
        other_bytes("1", pb),
        other_bytes("2", pa),
        other_bytes("3", pb),
        other_bytes("4", pa),
    ]
    browser = browser_of(post, frames=[onset])
    calls = [click(100, 100), click(300, 300), click(500, 500), click(700, 700), click(900, 900)]
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(calls[0])]) as rig:
        await run(
            rig,
            steps_of(
                [
                    (S[1], calls[1]),
                    (S[2], calls[2]),
                    (S[3], calls[3]),
                    (S[4], calls[4]),
                    (REPORT, ""),
                ]
            ),
        )
        assert dispatched(browser) == 4, (
            f"T-0137 B-5: the call on A's third visit was dispatched; {browser.calls}"
        )
        assert statuses(rig, "t1-call-s5") == ["refused"]
        tail = step_tail(rig, 6)
        coords = [(100, 100), (300, 300), (500, 500), (700, 700)]
        want: list[tuple[str, str, Any, Any]] = [
            (S[i], "click_at", {"x": x, "y": y}, ex(f"clicked at ({x}, {y})", True))
            for i, (x, y) in enumerate(coords)
        ]
        want.append((S[4], "click_at", {"x": 900, "y": 900}, refusal(RECURRING_REASON)))
        assert tail == want_tail(rig, want), (
            "T-0137 B-5: step 5's response is the recurring refusal, without a fact (no repeats here); "
            f"...{tail[-500:]}"
        )
        c = card(rig, "t1-call-s5")
        assert (c.status, c.reason) == ("refused", RECURRING_REASON), c
    recs = records_of(rig, "t1-call-s5")
    assert recs and recs[-1].get("no_progress") == "recurring", recs
    assert all("repeat_of_step" not in r for r in recs), recs
    line = check_line_shape(guard_line(rig))
    assert (line["stops_recurring"], line["stops_repeat"], line["max_screen_visits"]) == (
        1,
        0,
        3,
    ), line


async def test_fact_names_step_j_not_newest(tmp_path: Path) -> None:
    """B-6 (review B: the mutant builds the fact from the newest settled step): a click on screen P whose capture is
    the same capture (``unchanged``), then ``press_key Escape`` whose capture changes P's bytes (``updated``), then the
    same click again. Step 3 repeats step 1, so its fact is step 1's own: ``{step: 1, action: "clicked at (500, 250)",
    screenshot: "unchanged"}``, never step 2's (``pressed Escape``, ``updated``)."""
    browser = browser_of([same(), other_bytes("after-escape"), other_bytes("after-3")])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(rig, steps_of([(S[1], press("Escape")), (S[2], CLICK), (REPORT, "")]))
        assert dispatched(browser) == 3, browser.calls
        want = [
            (S[0], "click_at", CLICK_ARGS, ex(ACT, False)),
            (S[1], "press_key", {"combo": "Escape"}, ex("pressed Escape", True)),
            (S[2], "click_at", CLICK_ARGS, ex(ACT, True, fact(1, ACT, "unchanged"))),
        ]
        got = step_tail(rig, 4)
        assert got == want_tail(rig, want), (
            f"T-0137 B-6: the fact names step j (1) with step j's action and outcome; ...{got[-600:]}"
        )


async def test_toggle_recurring_has_no_fact(tmp_path: Path) -> None:
    """B-5 (review B: the mutant drops ``stop == "repeat"`` when attaching the fact): a toggle, click c on A, click
    d on B, c on A, d on B (every capture changes the screen, so no repeat is no-change), then c on A again. That call
    repeats step 3 but is stopped as ``recurring`` (A's third visit), and a recurring refusal carries no
    ``repeat_of_step``: ``{reason: RECURRING_REASON, screenshot: "unchanged", status: "refused"}``. Steps 3 and 4
    executed with their facts (steps 1 and 2)."""
    pa, pb = "a1a1a1a1a1a1a1a1", "b2b2b2b2b2b2b2b2"
    onset = live_frame("A-0", phash=pa)
    post = [other_bytes("1", pb), other_bytes("2", pa), other_bytes("3", pb), other_bytes("4", pa)]
    browser = browser_of(post, frames=[onset])
    c, d = click(200, 200), click(800, 600)
    ca, da = "clicked at (200, 200)", "clicked at (800, 600)"
    cargs, dargs = {"x": 200, "y": 200}, {"x": 800, "y": 600}
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(c)]) as rig:
        await run(rig, steps_of([(S[1], d), (S[2], c), (S[3], d), (S[4], c), (REPORT, "")]))
        assert dispatched(browser) == 4, browser.calls
        assert card(rig, "t1-call-s5").reason == RECURRING_REASON
        want = [
            (S[0], "click_at", cargs, ex(ca, True)),
            (S[1], "click_at", dargs, ex(da, True)),
            (S[2], "click_at", cargs, ex(ca, True, fact(1, ca, "updated"))),
            (S[3], "click_at", dargs, ex(da, True, fact(2, da, "updated"))),
            (S[4], "click_at", cargs, refusal(RECURRING_REASON)),
        ]
        got = step_tail(rig, 6)
        assert got == want_tail(rig, want), (
            f"T-0137 B-5: a recurring stop carries no repeat_of_step, even on a repeat; ...{got[-700:]}"
        )
    recs = records_of(rig, "t1-call-s5")
    assert recs and recs[-1].get("no_progress") == "recurring", recs


async def test_failed_steps_count_as_visits(tmp_path: Path) -> None:
    """B-3 / A-5 R-4 (review B: the mutant records only executed steps in the guard history): click on A (executed;
    the screen becomes B), a click read off B that fails (a fresh frame shows A), a click on A (executed; B), a click
    on B that fails (fresh: A), then a click on A. Every step's screen counts, failed ones included, so the fifth call
    is A's third visit: refused with ``RECURRING_REASON``, nothing dispatched for it."""
    pa, pb = "a1a1a1a1a1a1a1a1", "b2b2b2b2b2b2b2b2"
    frames = [live_frame("A-0", phash=pa), other_bytes("fresh-1", pa), other_bytes("fresh-2", pa)]
    browser = browser_of([other_bytes("1", pb), other_bytes("2", pb)], frames=frames)
    for cid in ("t1-call-s2", "t1-call-s4"):
        browser.outcome(cid, lambda inv: failed(inv, "TimeoutError during click_at"))
    calls = [click(100, 100), click(300, 300), click(500, 500), click(700, 700), click(900, 900)]
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(calls[0])]) as rig:
        await run(rig, steps_of([(S[i], calls[i]) for i in range(1, 5)] + [(REPORT, "")]))
        assert statuses(rig, "t1-call-s2") == ["failed"] and statuses(rig, "t1-call-s4") == [
            "failed"
        ], "premise: steps 2 and 4 failed"
        assert statuses(rig, "t1-call-s3") == ["executed"], "premise: step 3 executed"
        assert statuses(rig, "t1-call-s5") == ["refused"], (
            "T-0137 B-3/A-5: a failed step was read off its screen too; A's third visit is refused; "
            f"{statuses(rig, 't1-call-s5')}"
        )
        assert card(rig, "t1-call-s5").reason == RECURRING_REASON
        assert executes(browser, "t1-call-s5") == 0


async def test_form_fill_not_stopped(tmp_path: Path) -> None:
    """B-5 (adversarial, the too-wide direction): four ``type_at`` on different fields of one screen (equal executor
    phash, B8 ``updated`` each time), then a click on the submit button: all five execute, with no fact and no stop;
    the turn line's ``max_screen_visits`` is 1."""
    fields = [
        type_call(500, 200, "Ada"),
        type_call(500, 300, "Lovelace"),
        type_call(500, 400, "ada@example.org"),
        type_call(500, 500, "London"),
        click(500, 700),
    ]
    browser = browser_of([other_bytes(f"typed-{i}") for i in range(1, 6)])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(fields[0])]) as rig:
        await run(rig, steps_of([(S[i], fields[i]) for i in range(1, 5)] + [(REPORT, "")]))
        assert dispatched(browser) == 5, (
            f"T-0137 B-5: a form fill on one screen was refused; {browser.calls}"
        )
        tail = step_tail(rig, 6)
        assert (
            "repeat_of_step" not in tail
            and REPEAT_REASON not in tail
            and RECURRING_REASON not in tail
        )
    line = check_line_shape(guard_line(rig))
    assert line == {
        "repeats": 0,
        "no_change_repeats": 0,
        "stops_repeat": 0,
        "stops_recurring": 0,
        "max_screen_visits": 1,
    }, line


async def test_budget_spent_by_refusals(tmp_path: Path) -> None:
    """B-5 budget / no latch (H-T137-5): a model that keeps calling the same click after the stop is refused at
    every step until ``max_tool_steps`` (8): the executor runs 3 times, steps 4-8 are each refused with
    ``REPEAT_REASON``, ``tool_steps_executed`` is 8, and the turn ends as BASE's turn does when the executor itself
    refuses steps 4-8 (the same ``tool_loop_end``, the same machine commands and the same speech: R16 as at BASE)."""
    texts = [(S[i], CLICK) for i in range(1, 8)] + [(REPORT, "")]
    runs: dict[str, Any] = {}
    for arm in ("guard", "base"):
        browser = browser_of([same(), same(), same()])
        if arm == "base":
            for k in range(4, 9):
                browser.outcome(f"t1-call-s{k}", lambda inv: refused(inv, FRESHNESS))
        async with guard_rig(
            tmp_path / arm, browser, script=[answer(S[0]), tool(CLICK)], on=arm == "guard"
        ) as rig:
            await run(rig, steps_of(texts), limit_ms=60_000)
            runs[arm] = (
                rig,
                browser,
                dict(rig.record("t1").extras),
                machine_trace(rig),
                rig.spoken_sources(),
            )
    rig, browser, extras, trace, spoken = runs["guard"]
    assert dispatched(browser) == 3, (
        f"T-0137 B-5: a refused-by-guard step was dispatched; {browser.calls}"
    )
    for k in range(4, 9):
        assert statuses(rig, f"t1-call-s{k}") == ["refused"], k
        assert card(rig, f"t1-call-s{k}").reason == REPEAT_REASON, k
    _, _, base_extras, base_trace, base_spoken = runs["base"]
    assert extras.get("tool_steps_executed") == base_extras.get("tool_steps_executed") == 8, (
        extras,
        base_extras,
    )
    assert extras.get("tool_loop_end") == base_extras.get("tool_loop_end"), (extras, base_extras)
    assert trace == base_trace, (
        "T-0137 B-5/B-7: the machine saw something other than BASE's refusal path"
    )
    assert spoken == base_spoken


async def test_b9_first(tmp_path: Path) -> None:
    """B-4 (adversarial; reviewer mutation 8, the guard before B9): three identical clicks on an unchanging screen,
    then a scroll that captures a newer own frame (same phash, other bytes) and fails with no fresh frame, so step 5's
    identical click is read off a stale frame. It would be the guard's stop, but B9 comes first: refused with
    ``FRESHNESS_REASON``, the payload BASE's B9 payload (no ``repeat_of_step``), the card's reason B9's, no
    ``no_progress`` on any record, ``stops_repeat == 0``."""
    newer = other_bytes("newer")

    def capture_then_fail(fb: FakeLiveBrowser, inv: Any) -> Any:
        fb.post_frames.append(newer)
        fb.capture_post()
        fb.frame_mode = "raise"
        return failed(inv, "TimeoutError during scroll")

    browser = browser_of([same(), same(), same()])
    browser.handlers["t1-call-s4"] = capture_then_fail
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(
            rig,
            steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], SCROLL), (S[4], CLICK), (REPORT, "")]),
        )
        assert statuses(rig, "t1-call-s4") == ["failed"], "premise: the scroll failed"
        assert statuses(rig, "t1-call-s5") == ["refused"]
        assert executes(browser, "t1-call-s5") == 0
        assert card(rig, "t1-call-s5").reason == FRESHNESS, (
            f"T-0137 B-4: B9 runs before the guard; got {card(rig, 't1-call-s5').reason!r}"
        )
        tail = step_tail(rig, 6)
        assert loop.render_response("click_at", refusal(FRESHNESS)) in tail, tail[-500:]
        assert REPEAT_REASON not in tail
        assert [f["step"] for f in facts_in(tail)] == [1, 2], (
            f"B-4: only the executed repeats carry facts; {facts_in(tail)}"
        )
    assert not [r for r in all_records(rig) if "no_progress" in r], "B-5: no stop here"
    line = check_line_shape(guard_line(rig))
    assert line["stops_repeat"] == 0 and line["stops_recurring"] == 0, line


# =====================================================================================================
# B-3: where the outcome comes from
# =====================================================================================================

#: a long tail for every action string (S-R5): elision then shortens a response enough to matter
PAD = " and the page answered with a long description of what it did" * 5
#: S-R5: with the hand-written payloads every request from step 2 to step 7 fits for budgets 2310..2630 and steps 1
#: and 2 are elided before step 6 is judged (``loop.continuation_chars``); 2470 sits in the middle of that window
COMPACT_BUDGET = 2470


async def test_compaction_does_not_erase_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-3 (adversarial; reviewer mutation 6, the outcomes read back from ``turn.loop_steps``): clicks on one
    unchanging screen with presses between, every action string long, ``MAX_CONTINUATION`` lowered (S-R5): steps 1
    and 2's responses are elided by the real ``loop.compact`` before step 6 is judged. The repeats of the click are
    steps 2 and 5 (both no change), so step 6's identical click is still stopped with ``REPEAT_REASON``."""
    require_keyword("repeat_guard")
    s = session_mod()
    monkeypatch.setattr(s, "MAX_CONTINUATION", COMPACT_BUDGET)
    browser = browser_of([same() for _ in range(5)])

    def padded(fb: FakeLiveBrowser, inv: Any) -> Any:
        out = fb.perform(inv)
        return out.model_copy(update={"action": out.action + PAD})

    for cid in ("t1-call", *(f"t1-call-s{k}" for k in range(2, 7))):
        browser.handlers[cid] = padded
    script = [
        (S[1], CLICK),
        (S[2], press("Escape")),
        (S[3], press("Tab")),
        (S[4], CLICK),
        (S[5], CLICK),
        (REPORT, ""),
    ]
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)]) as rig:
        await run(rig, steps_of(script), limit_ms=60_000)
        extras = dict(rig.record("t1").extras)
        assert (
            extras.get("tool_compactions", 0) >= 1 and extras.get("tool_loop_end") != "over_budget"
        ), f"premise (S-R5): compaction ran and the turn stayed in budget; {extras}"
        s6 = step_tail(rig, 6)
        elided = loop.render_response("click_at", loop.elided_payload({"status": "executed"}))
        assert s6.count(elided) >= 2 and s6.count("repeat_of_step") == 1, (
            f"premise (S-R5): steps 1 and 2 are elided in the step-6 request; ...{s6[:900]}"
        )
        assert dispatched(browser) == 5, (
            f"T-0137 B-3: the outcomes were read from the compacted steps; step 6 was dispatched; {browser.calls}"
        )
        assert statuses(rig, "t1-call-s6") == ["refused"]
        assert card(rig, "t1-call-s6").reason == REPEAT_REASON


def _action_outcome() -> Any:
    try:
        return importlib.import_module("local_astra.conductor.action_outcome")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the merge
        raise AssertionError(
            "T-0137 S-OUT-1 needs T-0136's merged action_outcome module (O-1); it is absent at BASE_B"
        ) from exc


@pytest.mark.parametrize("t0136", ["on", "off"])
async def test_t0136_outcome_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, t0136: str
) -> None:
    """B-3 S-OUT-1 (reviewer mutation 7, B8's ``screenshot`` used while T-0136 is on; S-R4): four identical clicks
    whose captures change the bytes of the same screen (B8 ``updated``) while T-0136's measurement says ``no visible
    change`` (``outcome_of_jpeg`` patched to ``Outcome(kind="none")``). With T-0136 on: steps 1-3 count as no change
    and the fourth is stopped; step 2's fact carries ``screenshot: "updated"`` and one more key whose value is ``no
    visible change``. With T-0136 off (no ``action_outcome`` keyword) the same script does not stop: the fallback
    reads ``updated``."""
    require_keyword("repeat_guard")
    extra: dict[str, Any] = {}
    spy = turn_logs(monkeypatch)
    if t0136 == "on":
        require_keyword("action_outcome")
        ao = _action_outcome()
        none = ao.Outcome(kind="none", region=None, changed_cells=0, cells=200)
        assert none.kind == "none" and none.text == ao.NO_VISIBLE_CHANGE == "no visible change", (
            none.text,
            ao.NO_VISIBLE_CHANGE,
        )
        patch_bound(monkeypatch, ao, "outcome_of_jpeg", lambda *a, **k: none)
        extra["action_outcome"] = True
    post = [jpeg_frame(f"badge-{i}", seed=i).model_copy(update={"phash": P}) for i in range(1, 5)]
    onset = jpeg_frame("onset", seed=0).model_copy(update={"phash": P})
    browser = browser_of(post, frames=[onset])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)], extra=extra) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], CLICK), (REPORT, "")]))
        tail = step_tail(rig, 5)
        facts = facts_in(tail)
        outcomes = spy.by_turn("t1").action_outcomes
        if t0136 == "on":
            assert sorted(outcomes) == [1, 2, 3] and all(
                o.kind == "none" for o in outcomes.values()
            ), (
                f"premise (T-0136 B-5, dd637f0): _TurnLog.action_outcomes holds none for steps 1-3; {outcomes}"
            )
        else:
            assert outcomes == {}, f"premise: T-0136 off measures nothing; {outcomes}"
        if t0136 == "off":
            assert dispatched(browser) == 4, (
                f"T-0137 S-OUT-2: B8 said updated, so no stop; {browser.calls}"
            )
            assert [f.get("screenshot") for f in facts] == ["updated"] * 3, facts
            return
        assert dispatched(browser) == 3, (
            f"T-0137 S-OUT-1: T-0136 said no visible change three times; the fourth was dispatched; {browser.calls}"
        )
        assert card(rig, "t1-call-s4").reason == REPEAT_REASON
        first = facts[0]
        assert (first.get("step"), first.get("action"), first.get("screenshot")) == (
            1,
            ACT,
            "updated",
        ), first
        others = {k: v for k, v in first.items() if k not in ("step", "action", "screenshot")}
        assert list(others.values()) == ["no visible change"], (
            f"T-0137 S-OUT-1 (S-R4): the fact carries T-0136's key with its value beside screenshot; {first}"
        )


#: T-0136's other labels (A2-2's vocabulary, typed by hand): kind, region, the ``.text`` the fact must carry
OTHER_LABELS = {
    "near": ("near", None, "changed near your click"),
    "elsewhere": ("elsewhere", "top right", "changed elsewhere (top right)"),
    "page": ("page", None, "page changed"),
}


@pytest.mark.parametrize("label", list(OTHER_LABELS))
async def test_t0136_other_labels_never_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    """B-3 S-OUT-1, the other arms of ``test_t0136_outcome_used`` (review B: the mutant sets ``no_change = True``
    whenever T-0136 measured): ``no_change`` is ``False`` for every T-0136 label other than ``no visible change``. The
    same four identical clicks with the patched label ``near`` / ``elsewhere`` / ``page`` all execute (no stop), and
    each repeat's fact carries that label's text beside ``screenshot: "updated"``."""
    require_keyword("repeat_guard", "action_outcome")
    kind, region, text = OTHER_LABELS[label]
    ao = _action_outcome()
    outcome = ao.Outcome(kind=kind, region=region, changed_cells=12, cells=200)
    assert outcome.text == text, (outcome.text, text)
    patch_bound(monkeypatch, ao, "outcome_of_jpeg", lambda *a, **k: outcome)
    spy = turn_logs(monkeypatch)
    post = [jpeg_frame(f"badge-{i}", seed=i).model_copy(update={"phash": P}) for i in range(1, 5)]
    onset = jpeg_frame("onset", seed=0).model_copy(update={"phash": P})
    browser = browser_of(post, frames=[onset])
    async with guard_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], extra={"action_outcome": True}
    ) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], CLICK), (REPORT, "")]))
        outcomes = spy.by_turn("t1").action_outcomes
        assert sorted(outcomes) == [1, 2, 3, 4] and all(
            o.kind == kind for o in outcomes.values()
        ), f"premise: T-0136 measured {kind} for every step; {outcomes}"
        assert dispatched(browser) == 4, (
            f"T-0137 S-OUT-1: T-0136 said {text!r}, which is a change; a click was stopped; {browser.calls}"
        )
        facts = facts_in(step_tail(rig, 5))
        assert [f.get("step") for f in facts] == [1, 2, 3], facts
        for f in facts:
            others = [v for k, v in f.items() if k not in ("step", "action", "screenshot")]
            assert f.get("screenshot") == "updated" and others == [text], (
                f"T-0137 S-OUT-1: the fact carries T-0136's label {text!r}; {f}"
            )


# =====================================================================================================
# B-2: the overview step and ReGround
# =====================================================================================================


async def test_overview_step1_excluded(tmp_path: Path) -> None:
    """B-2 / A-3 ``overview`` (reviewer mutation 10): with the overview frame on, step 1 clicks (160, 935) on the
    overview; its post-action capture has the replaced live frame's executor phash, and step 2 clicks (160, 935) on
    that live frame. Numerically equal, but step 1 was read off the overview: no fact and no stop (both execute)."""
    post = [other_bytes("after-overview", s151.F0.phash), other_bytes("after-2", s151.F0.phash)]
    browser = s151.deals_browser(frames=(s151.F0,), post=post)
    async with guard_rig(
        tmp_path, browser, script=[answer(S[0]), tool(s151.TOWEL)], extra={"overview_frame": True}
    ) as rig:
        await run(rig, steps_of([(S[1], s151.TOWEL), (S[2], "")]))
        assert s151.frame_key(rig) == s151.counts(shown=1), "premise: step 1 showed the overview"
        assert dispatched(browser) == 2, browser.calls
        assert "repeat_of_step" not in step_tail(rig, 3), (
            "T-0137 B-2/A-3: a step-1 call read off the overview frame is never the same action as a live one"
        )


async def test_overview_step1_screen_is_replaced_frame(tmp_path: Path) -> None:
    """B-2 ("for a T-0151 overview step 1 the screen key is the live frame the overview replaced"): step 1 (overview)
    on P = the replaced frame's executor phash; steps 2-5 on Q, P, Q, P (distinct clicks). Step 5 is P's third visit
    only if step 1's key is P, so it is refused with ``RECURRING_REASON``."""
    p, q = s151.F0.phash, "c3c3c3c3c3c3c3c3"
    assert p is not None
    post = [other_bytes("1", q), other_bytes("2", p), other_bytes("3", q), other_bytes("4", p)]
    browser = s151.deals_browser(frames=(s151.F0,), post=post)
    calls = [click(100, 100), click(300, 300), click(700, 700), click(900, 900)]
    async with guard_rig(
        tmp_path, browser, script=[answer(S[0]), tool(s151.TOWEL)], extra={"overview_frame": True}
    ) as rig:
        await run(
            rig,
            steps_of(
                [
                    (S[1], calls[0]),
                    (S[2], calls[1]),
                    (S[3], calls[2]),
                    (S[4], calls[3]),
                    (REPORT, ""),
                ]
            ),
        )
        assert s151.frame_key(rig) == s151.counts(shown=1), "premise: step 1 showed the overview"
        assert statuses(rig, "t1-call-s5") == ["refused"], (
            "T-0137 B-2: the overview step's screen key is the replaced live frame's phash, so step 5 is P's third "
            f"visit; statuses {statuses(rig, 't1-call-s5')}"
        )
        assert card(rig, "t1-call-s5").reason == RECURRING_REASON


async def test_reground_matches_model_args(tmp_path: Path) -> None:
    """B-4 ReGround: with ``reground=True`` pass 2 moves each executed point (step 1 to the crop's (900, 100), step 2
    to its (100, 900), far apart). The model's own arguments are equal (500, 250), so step 2 is a repeat of step 1:
    its response carries the fact with step 1's executor action string (the fake's pixel echo of step 1's executed
    arguments: ``x * 1280 // 1000``, ``y * 800 // 1000``)."""
    require_keyword("repeat_guard")
    onset = jpeg_frame("onset")
    browser = FakeLiveBrowser([onset], post_frames=[onset.model_copy(), onset.model_copy()])
    async with reground_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], session_kwargs={"repeat_guard": True}
    ) as rig:
        rig.llm.script_pass2(
            [answer("p2"), tool(click(900, 100))], [answer("p2"), tool(click(100, 900))]
        )
        rig.llm.script_steps(*steps_of([(S[1], CLICK), (S[2], "")]))
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        invs = {i.call_id: dict(i.arguments) for i in browser.invocations}
        a1, a2 = invs["t1-call"], invs["t1-call-s2"]
        assert max(abs(a1["x"] - a2["x"]), abs(a1["y"] - a2["y"])) > 10 and a1 != CLICK_ARGS, (
            f"premise: pass 2 moved both points apart; {a1} {a2}"
        )
        act1 = f"clicked at ({a1['x'] * 1280 // 1000}, {a1['y'] * 800 // 1000})"
        tail = step_tail(rig, 3)
        assert rendered_fact(1, act1, "unchanged") in tail, (
            f"T-0137 B-4: repeats are judged on call.arguments, not pass 2's; ...{tail[-500:]}"
        )


async def test_stop_sends_no_pass2(tmp_path: Path) -> None:
    """B-4 (review B: the mutant runs the guard after ReGround): with ``reground=True``, four identical clicks on an
    unchanging screen; the fourth is stopped before pass 2, so exactly three pass-2 requests are sent (steps 1-3) and
    three calls dispatched."""
    require_keyword("repeat_guard")
    onset = jpeg_frame("onset")
    browser = FakeLiveBrowser([onset], post_frames=[onset.model_copy() for _ in range(4)])
    async with reground_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], session_kwargs={"repeat_guard": True}
    ) as rig:
        rig.llm.script_pass2(*([answer("p2"), tool(click(900, 100))] for _ in range(4)))
        rig.llm.script_steps(*steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], CLICK), (REPORT, "")]))
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        assert statuses(rig, "t1-call-s4") == ["refused"], "premise: the fourth click is stopped"
        assert card(rig, "t1-call-s4").reason == REPEAT_REASON
        assert dispatched(browser) == 3, browser.calls
        assert len(rig.llm.pass2_ids) == 3, (
            f"T-0137 B-4: the guard runs before ReGround, so a stopped call sends no pass 2; {rig.llm.pass2_ids}"
        )


async def test_blind_turn_inert(tmp_path: Path) -> None:
    """B-2/B-3 (blind: the screen is ``None``): the onset frame fails, so the turn is blind; four identical scrolls
    all execute, with no fact and no stop."""
    browser = FakeLiveBrowser([], post_frames=[same() for _ in range(4)], echo="model")
    browser.frame_mode = "raise"
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(SCROLL)]) as rig:
        await run(rig, steps_of([(S[1], SCROLL), (S[2], SCROLL), (S[3], SCROLL), (REPORT, "")]))
        assert dispatched(browser) == 4, f"T-0137 B-2: a blind turn was stopped; {browser.calls}"
        tail = step_tail(rig, 5)
        assert (
            "repeat_of_step" not in tail
            and REPEAT_REASON not in tail
            and RECURRING_REASON not in tail
        )
    line = check_line_shape(guard_line(rig))
    assert (line["repeats"], line["stops_repeat"], line["stops_recurring"]) == (0, 0, 0), line


# =====================================================================================================
# B-7: speech
# =====================================================================================================

ACT_SENTINEL = " ACT-SENTINEL-T0137"


def _sentinel_actions(browser: FakeLiveBrowser) -> None:
    def tagged(fb: FakeLiveBrowser, inv: Any) -> Any:
        out = fb.perform(inv)
        return out.model_copy(update={"action": out.action + ACT_SENTINEL})

    for cid in ("t1-call", "t1-call-s2", "t1-call-s3"):
        browser.handlers[cid] = tagged


def _texts_everywhere(rig: Any) -> str:
    """Every TtsSpeak source and text, the history, the Episodes and the machine's events, as one string."""
    speaks = [(c.source, c.text) for c in rig.speaks()]
    history = [h.model_dump(mode="json") for h in rig.state.history]
    episodes = [e.model_dump(mode="json") for e in getattr(rig.memory, "episodes", {}).values()]
    events = [e.model_dump(mode="json") for e in rig.machine.events()]
    return json.dumps(
        {"speaks": speaks, "history": history, "episodes": episodes, "events": events}
    )


@pytest.mark.parametrize("after", ["answer", "silent"])
async def test_stuck_speech_committed_only(tmp_path: Path, after: str) -> None:
    """B-7 (adversarial; reviewer mutation 9, a synthesised "I'm stuck" or the reason voiced): three identical clicks
    on an unchanging screen, the fourth stopped, then a step with no call. With the committed answer ``STUCK`` the
    speech after the stop is exactly ``STUCK`` and the history holds it; with no answer text nothing is voiced after
    the stop. Both reasons and the action strings (a sentinel appended by the executor, so it reaches the repeat facts)
    reach no TtsSpeak, history, Episode or machine event. The machine's command sequence equals the same script with
    the executor refusing step 4 with B9's reason and the guard off (no turn-machine difference).

    T-0234 A1 (named re-pin): in ``silent`` the empty step 5 follows a refused click, so it is re-asked once (both
    arms); the re-ask is scripted empty too, and the premise reads step 5's first request (both carry the same
    prompt)."""
    final = [answer(STUCK)] if after == "answer" else []
    extra: list[Any] = (
        [] if after == "answer" else [[]]
    )  # T-0234 A1: the re-ask of the empty step 5
    runs: dict[str, Any] = {}
    for arm in ("guard", "base"):
        browser = browser_of([same(), same(), same()])
        _sentinel_actions(browser)
        if arm == "base":
            browser.outcome("t1-call-s4", lambda inv: refused(inv, FRESHNESS))
        async with guard_rig(
            tmp_path / arm, browser, script=[answer(S[0]), tool(CLICK)], on=arm == "guard"
        ) as rig:
            rig.llm.script_steps(
                [answer(S[1]), tool(CLICK)],
                [answer(S[2]), tool(CLICK)],
                [answer(S[3]), tool(CLICK)],
                final,
                *extra,
            )
            await rig.ask(tsl.QUESTION)
            await rig.turn_done(limit_ms=30_000)
            await rig.advance(500)
            runs[arm] = rig
    g, b = runs["guard"], runs["base"]
    assert card(g, "t1-call-s4").reason == REPEAT_REASON, "premise: the guard stopped step 4"
    step5 = tsl.generates_for(g, tsl.step_id(5))
    assert len(step5) == (1 if after == "answer" else 2), (
        f"T-0234 A1: step 5 is re-asked only when it came back empty; {[r for r, _ in step5]}"
    )
    assert ACT_SENTINEL.strip() in tsl.tail(step5[0][1]), (
        "premise: the action strings reached the repeat facts"
    )
    after_stop = "".join(c.source for c in speaks_after(g, "t1-call-s4"))
    if after == "answer":
        assert after_stop == STUCK, (
            f"T-0137 B-7: after the stop only the committed answer is voiced; {after_stop!r}"
        )
        assert g.state.history and STUCK in g.state.history[-1].assistant_spoken, g.state.history
    else:
        assert after_stop == "", (
            f"T-0137 B-7: a step without answer text voices nothing; {after_stop!r}"
        )
    everywhere = _texts_everywhere(g)
    for needle in (REPEAT_REASON, RECURRING_REASON, ACT_SENTINEL.strip(), "repeat_of_step"):
        assert needle not in everywhere, (
            f"T-0137 B-7: {needle!r} reached speech, history, an Episode or an event"
        )
    assert machine_trace(g) == machine_trace(b), (
        "T-0137 B-7: the machine saw something other than a refusal"
    )
    assert g.spoken_sources() == b.spoken_sources()
    assert [h.model_dump(mode="json") for h in g.state.history] == [
        h.model_dump(mode="json") for h in b.state.history
    ]


# =====================================================================================================
# B-8: logging
# =====================================================================================================


async def test_turn_line_and_records(tmp_path: Path) -> None:
    """B-8 (S-R2, S-R3): a click, the same click again (every capture changes the bytes of one screen, so the repeat
    executes with a fact), a click elsewhere, then the report; then a toolless turn. Turn 1's ``repeat_guard`` is
    ``{repeats: 1, no_change_repeats: 0, stops_repeat: 0, stops_recurring: 0, max_screen_visits: 1}``; turn 2's is
    all zeros; ``repeat_of_step: 1`` is on the last ``tool.call`` record of step 2's call and on no record of any other
    call; no record has ``no_progress``. The session-log kinds and the turn's span names are a subset of the same
    script's with the guard off (no new span or log kind)."""
    kinds: dict[str, set[str]] = {}
    spans: dict[str, set[str]] = {}
    for arm in ("guard", "off"):
        browser = browser_of(
            [other_bytes(f"c{i}") for i in range(1, 4)], frames=[F0, other_bytes("t2")]
        )
        async with guard_rig(
            tmp_path / arm, browser, script=[answer(S[0]), tool(CLICK)], on=arm == "guard"
        ) as rig:
            await run(rig, steps_of([(S[1], CLICK), (S[2], click(900, 900)), (REPORT, "")]))
            await next_turn(rig, 2, "what is on the screen", [answer("A shop. ")])
            if arm == "guard":
                grig = rig
        kinds[arm] = {r["kind"] for r in rig.log_records}
        spans[arm] = set(rig.record("t1").spans)
    assert kinds["guard"] <= kinds["off"], (
        f"T-0137 B-8: a new log kind {kinds['guard'] - kinds['off']}"
    )
    assert spans["guard"] <= spans["off"], f"T-0137 B-8: a new span {spans['guard'] - spans['off']}"
    line = check_line_shape(guard_line(grig, "t1"))
    assert line == {
        "repeats": 1,
        "no_change_repeats": 0,
        "stops_repeat": 0,
        "stops_recurring": 0,
        "max_screen_visits": 1,
    }, line
    assert check_line_shape(guard_line(grig, "t2")) == dict.fromkeys(GUARD_KEYS, 0)
    recs = records_of(grig, "t1-call-s2")
    assert recs and recs[-1].get("repeat_of_step") == 1, (
        f"T-0137 B-6: the repeat call's tool.call record gains repeat_of_step: 1; {recs}"
    )
    for cid in ("t1-call", "t1-call-s3"):
        assert all("repeat_of_step" not in r for r in records_of(grig, cid)), (
            cid,
            records_of(grig, cid),
        )
    assert not [r for r in all_records(grig) if "no_progress" in r]


async def test_off_no_key(tmp_path: Path) -> None:
    """B-8 "Off: no key": the same four identical clicks as the stop row with the session built without the keyword:
    all four execute, no turn line has ``repeat_guard``, no record has ``repeat_of_step`` or ``no_progress``, and no
    continuation carries a fact. (Pending with the module: the keyword's existence is checked first, so the row is red
    until the keyword exists.)"""
    require_keyword("repeat_guard")
    browser = browser_of([same(), same(), same(), same()])
    async with guard_rig(tmp_path, browser, script=[answer(S[0]), tool(CLICK)], on=False) as rig:
        await run(rig, steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], CLICK), (REPORT, "")]))
        assert dispatched(browser) == 4
        assert "repeat_of_step" not in step_tail(rig, 5)
        lines = [json.loads(x) for x in rig.lines()]
    assert all("repeat_guard" not in x.get("extras", {}) for x in lines), lines
    flat = json.dumps(all_records(rig))
    assert "repeat_of_step" not in flat and "no_progress" not in flat
