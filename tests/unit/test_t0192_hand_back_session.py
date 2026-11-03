"""T-0192 group B through the session: the checkout hand-back (docs/tasks/T-0192.md B1, B4, B5).

Every row runs over the session fakes: ``live_rig`` (T-0111g: ``FakeLiveBrowser`` behind a live kit, loop on) or
``tool_rig`` (dry run), the recording machine, ``RecordingAudio`` and the ``ScriptedClock``. The executor's group A
outputs are scripted on the fake browser (``resolved_extra`` for an arrival, ``outcome`` for a refusal), so these rows
test only what the session does with them. Expected values are written by hand from B1/B4.

What the rows fix where the task is silent (in the test-author's report):

- **Which events hand back** (B4, exact): an ``executed`` outcome whose ``resolved.get("checkout") is True``, or a
  ``refused`` outcome whose ``reason == executor.CHECKOUT_REASON``. Not: ``1``, ``"true"``, ``False``, absent; a
  ``failed`` outcome carrying ``checkout: True``; a reason that only resembles ``CHECKOUT_REASON``; anything without a
  live browser. Those rows hold at BASE (no hand-back exists) and stay unmarked: they are the guard's "too wide" side.
- **The log row**: the handed-back call's ``tool.result`` row is BASE's four keys plus ``"hand_back": "checkout"``
  (nothing else, so no URL, path or page text can ride along); every other ``tool.result`` row carries no
  ``hand_back`` value (absent or ``None``).
- **The clip** is read off the audio device (``RecordingAudio.clip_plays()``, played with ``source=""``): one
  ``at_checkout`` play with B1's sentence; never a ``tts.utterance``, never history.
- **The model is never asked again**: no ``LlmContinue`` after the handed-back result, so the scripted next step (a
  drafted "I will complete the purchase") is never requested and never voiced (adversarial, Tests to write).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), tools
``live`` unless a row says dry run; n = 1 and deterministic per row. No measurement, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_astra import preflight
from local_astra.computer import executor as executor_mod
from local_astra.conductor import session as session_mod
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.toolcall import ToolOutcome
from local_astra.reflex import clips
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import FakeLiveBrowser, hud_frame, live_rig, refused
from tests.unit._tool_turn_support import VALID_CLICK, WatchingExecutor, answer, tool, tool_rig
from tests.unit.test_session_live import F0, FR, next_turn
from tests.unit.test_t0140_session import overview_kit
from tests.unit.test_t0182_doctor_clips import default_dir  # noqa: F401  (fixture)

AT_CHECKOUT = "at_checkout"
#: B1, by hand from the task file
AT_CHECKOUT_TEXT = (
    "I've reached the checkout, so I'll stop here. Over to you to finish the purchase."
)
#: ``clips.ENDING_KINDS`` at BASE (da29ddbe ``reflex/clips.py`` line 24, by hand)
BASE_ENDING_KINDS = {"out_of_steps", "not_finished", "lost_thread"}
#: the ``tool.result`` log row's keys at BASE (da29ddbe ``Session._settle_result``, by hand)
BASE_RESULT_KEYS = {"call_id", "step", "status", "response"}
CID = "t1-call"
#: B-RV3: the other ways to reach or meet a checkout (tools_v1 / tools_v1_overview native calls, by hand)
TYPE_ENTER = f"<|tool_call>call:type_at{{press_enter:true,text:{tsl.Q}blue mug{tsl.Q},x:500,y:250}}<tool_call|>"
PRESS_ENTER = f"<|tool_call>call:press_key{{combo:{tsl.Q}Enter{tsl.Q}}}<tool_call|>"
OV_CALL = "<|tool_call>call:page_overview{}<tool_call|>"
CID2 = tsl.call_id(2)
#: the adversarial next step: drafted after an arrival, it must never be requested or voiced
DEAD = "I will complete the purchase now."


def checkout_reason() -> str:
    reason = getattr(executor_mod, "CHECKOUT_REASON", None)
    assert reason == "this is a checkout page; the user finishes the purchase themselves", (
        f"premise (T-0192 A3): executor.CHECKOUT_REASON; got {reason!r}"
    )
    return reason


# --- reading the rig ---------------------------------------------------------------------------------------------


def results(rig: Any) -> list[Any]:
    return list(rig.machine.events(E.ToolResult))


def hand_backs(rig: Any) -> list[Any]:
    return [getattr(ev, "hand_back", None) for ev in results(rig)]


def checkout_clips(rig: Any) -> list[str]:
    return [
        str(getattr(u, "text", ""))
        for u in rig.audio.clip_plays()
        if getattr(u, "kind", None) == AT_CHECKOUT
    ]


def continues(rig: Any) -> list[Any]:
    return list(rig.commands(C.LlmContinue))


def history_text(rig: Any) -> str:
    return " | ".join(f"{h.user_text} / {h.assistant_spoken}" for h in rig.state.history)


def utterance_sources(rig: Any) -> list[str]:
    return [str(r.get("source", "")) for r in rig.kind("tts.utterance")]


def final_status(rig: Any, cid: str) -> str:
    msgs = rig.tool_calls(cid)
    assert msgs, f"premise: {cid} published a card"
    return str(msgs[-1].status)


def assert_handed_back(
    rig: Any, *, cid: str, status: str, turn: str = "t1", played: str = tsl.S1
) -> None:
    """B4 at session level, the whole contract of one handed-back turn (by hand)."""
    mine = [ev for ev in results(rig) if ev.call_id == cid]
    assert len(mine) == 1, f"premise: one ToolResult for {cid}; {results(rig)}"
    assert getattr(mine[0], "hand_back", None) == "checkout", (
        f"T-0192 B4: {cid} ({status}) is submitted with hand_back='checkout'; got {mine[0]!r}"
    )
    assert mine[0].status == status, mine[0]
    assert not [c for c in continues(rig) if c.call_id == cid], (
        f"T-0192 B2/B3 L3: no llm.continue after a hand-back; {continues(rig)}"
    )
    rows = [r for r in rig.kind("tool.result") if r.get("call_id") == cid]
    assert len(rows) == 1, rows
    assert rows[0].get("hand_back") == "checkout" and set(rows[0]) == BASE_RESULT_KEYS | {
        "hand_back"
    }, f"T-0192 B4: the tool.result row gains exactly hand_back='checkout'; {rows[0]}"
    assert rig.record(turn).extras.get("tool_loop_end") == "handed_back", (
        f"T-0192 B4: tool_loop_end; got {rig.record(turn).extras.get('tool_loop_end')!r}"
    )
    assert final_status(rig, cid) == status, (
        f"B4: the HUD card is the usual one; {rig.statuses(cid)}"
    )
    entry = [h for h in rig.state.history if h.assistant_spoken == played]
    assert entry, f"T-0192 B2: history holds the played text {played!r}; {history_text(rig)}"
    assert AT_CHECKOUT_TEXT not in history_text(rig), "H8/L6: the clip never enters history"
    assert all(AT_CHECKOUT_TEXT not in s for s in utterance_sources(rig)), (
        "the clip is never a tts.utterance"
    )


def assert_not_handed_back(rig: Any, *, cid: str = CID) -> None:
    assert all(h is None for h in hand_backs(rig)), f"T-0192 B4 (too wide): {results(rig)}"
    assert checkout_clips(rig) == [], checkout_clips(rig)
    assert [c.call_id for c in continues(rig)][:1] == [cid], (
        f"the loop continued after {cid}: {continues(rig)}"
    )
    rows = rig.kind("tool.result")
    assert rows and all(r.get("hand_back") is None for r in rows), rows
    assert rig.record("t1").extras.get("tool_loop_end") == "final", rig.record("t1").extras


# =====================================================================================================================
# B1: the clip registry
# =====================================================================================================================


def test_b1_clip_text_and_ending_kinds() -> None:
    """B1: ``CLIP_TEXT["at_checkout"]`` is the ruled sentence; ``clips.ENDING_KINDS`` is BASE's plus ``at_checkout``;
    the registry lists it and it plays by default (filler clips off)."""
    assert session_mod.CLIP_TEXT.get(AT_CHECKOUT) == AT_CHECKOUT_TEXT, session_mod.CLIP_TEXT
    assert set(clips.ENDING_KINDS) == BASE_ENDING_KINDS | {AT_CHECKOUT}, clips.ENDING_KINDS
    registry = clips.registered()
    assert registry.get(AT_CHECKOUT) == AT_CHECKOUT_TEXT, registry
    assert AT_CHECKOUT in clips.default_enabled(registry, filler_clips=False)


def test_b1_doctor_reports_the_clip(default_dir: Path) -> None:  # noqa: F811
    """B1 "doctor reports it": with no assets rendered, ``collect_clips`` lists ``at_checkout`` as ``missing`` (so the
    clips row WARNs naming it until ``render-clips`` runs, implementation notes)."""
    got = preflight.collect_clips()["clips"]
    assert got.get(AT_CHECKOUT) == "missing", got


# =====================================================================================================================
# B4: the session submits the hand-back
# =====================================================================================================================


async def test_b4_executed_arrival_hands_back_and_ends_the_turn(tmp_path: Path) -> None:
    """B4: an executed click whose ``resolved["checkout"] is True``: the result is submitted with
    ``hand_back="checkout"``, the turn ends with the at_checkout clip, the scripted next step (``DEAD``) is never
    requested or voiced, the log row and the turn line say so, history holds the played S1 only."""
    browser = FakeLiveBrowser([F0, FR])
    browser.resolved_extra[CID] = {"checkout": True}
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(500)
        assert checkout_clips(rig) == [AT_CHECKOUT_TEXT], f"T-0192 B1/B2: {rig.audio.clip_plays()}"
        assert DEAD not in rig.spoken_sources(), "adversarial: the drafted purchase step was voiced"
        assert not rig.llm.step_prompts, (
            f"the model was asked again: {[r for r, _ in rig.llm.step_prompts]}"
        )
    assert_handed_back(rig, cid=CID, status="executed")
    assert all(DEAD not in s for s in utterance_sources(rig)) and DEAD not in history_text(rig)


async def test_b4_refused_on_a_checkout_page_hands_back(tmp_path: Path) -> None:
    """B4: a click refused with ``CHECKOUT_REASON`` hands back the same way (card ``refused``)."""
    reason = checkout_reason()
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(CID, lambda inv: refused(inv, reason))
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(500)
        assert checkout_clips(rig) == [AT_CHECKOUT_TEXT], rig.audio.clip_plays()
        assert DEAD not in rig.spoken_sources()
    assert_handed_back(rig, cid=CID, status="refused")


async def test_b4_hand_back_at_step_2_keeps_earlier_played_text_only(tmp_path: Path) -> None:
    """B2/B4 at step 2: step 1 voiced S1 and executed; step 2's text S2 is withheld (R15) and its click arrives at a
    checkout; the turn ends with one clip, history holds S1 only, one llm.continue (step 2's) and none after."""
    browser = FakeLiveBrowser([F0, FR])
    browser.resolved_extra[CID2] = {"checkout": True}
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(tsl.S2), tool(tsl.CLICK2)], [answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(500)
        assert checkout_clips(rig) == [AT_CHECKOUT_TEXT], rig.audio.clip_plays()
        assert [c.call_id for c in continues(rig)] == [CID], continues(rig)
        assert tsl.S2 not in rig.spoken_sources() and DEAD not in rig.spoken_sources()
    assert_handed_back(rig, cid=CID2, status="executed")
    assert tsl.S2 not in history_text(rig) and DEAD not in history_text(rig)


async def test_b5_the_next_turn_at_checkout_hands_back_again(tmp_path: Path) -> None:
    """B5: turn 1 arrives at a checkout and hands back; turn 2's click (still at checkout) is refused with
    ``CHECKOUT_REASON`` and hands back again: one at_checkout clip per turn, both turn lines ``handed_back``."""
    reason = checkout_reason()
    browser = FakeLiveBrowser([F0, FR])
    browser.resolved_extra[CID] = {"checkout": True}
    t2 = tsl.call_id(1, "t2")
    browser.outcome(t2, lambda inv: refused(inv, reason))
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(500)
        await next_turn(
            rig, 2, "click buy now", [answer(tsl.S2), tool(VALID_CLICK)], ([answer(DEAD)],)
        )
        await rig.advance(500)
        assert checkout_clips(rig) == [AT_CHECKOUT_TEXT, AT_CHECKOUT_TEXT], rig.audio.clip_plays()
        assert DEAD not in rig.spoken_sources()
    assert_handed_back(rig, cid=CID, status="executed")
    assert_handed_back(rig, cid=t2, status="refused", turn="t2", played=tsl.S2)


# =====================================================================================================================
# B-RV2: the turn line says handed_back only when the machine really handed back
# =====================================================================================================================


def _arrival_racing(rig_box: dict[str, Any], ending: str) -> Any:
    """An arrival outcome (``checkout: True``) whose execution first queues ``ending`` on the session: the ending
    event is submitted while the click is in flight, so it reaches the machine ahead of the checkout ``ToolResult``.
    ``"none"`` queues nothing (the positive control)."""

    def make(inv: Any) -> ToolOutcome:
        rig = rig_box["rig"]
        now = rig.clock.now_ms
        if ending == "takeover":
            rig.session.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        elif ending == "disconnected":
            rig.session.submit(E.ClientDisconnected(t_ms=now))
        return ToolOutcome(
            call_id=inv.call_id,
            status="executed",
            action="clicked at (640, 400)",
            reason=None,
            resolved={"checkout": True, "frame_after": None},
        )

    return make


@pytest.mark.parametrize(
    ("ending", "want_end"),
    [("takeover", "takeover"), ("disconnected", "disconnected"), ("none", "handed_back")],
    ids=["takeover_first", "disconnect_first", "control_hand_back"],
)
async def test_brv2_an_ending_queued_ahead_of_the_checkout_result_wins(
    tmp_path: Path, ending: str, want_end: str
) -> None:
    """B-RV2: ``tool_loop_end == "handed_back"`` only when the machine's closing commands played ``at_checkout``.

    A ``UserTakeover`` (or ``ClientDisconnected``) queued ahead of a checkout ``ToolResult`` ends the turn as at BASE
    for that event (``Session._close_loop``: ``takeover`` for a takeover in THINKING, ``disconnected``); the late
    result is inert (R13), so no ``at_checkout`` clip plays and the turn line never says ``handed_back``. The control
    arm (nothing queued) is the same rig handing back for real. Expected labels by hand from H-T114-5 / B-RV2."""
    rig_box: dict[str, Any] = {}
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(CID, _arrival_racing(rig_box, ending))
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig_box["rig"] = rig
        rig.llm.script_steps([answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await rig.advance(3000)
        await rig.advance(500)
        assert browser.executes() == 1, f"premise: the click executed once; {browser.calls}"
        clips_now = checkout_clips(rig)
        assert DEAD not in rig.spoken_sources(), "the drafted purchase step was voiced"
    end = rig.record("t1").extras.get("tool_loop_end")
    if ending == "none":
        assert clips_now == [AT_CHECKOUT_TEXT], (
            f"control: the hand-back plays; {rig.audio.clip_plays()}"
        )
        assert end == "handed_back", f"control: a real hand-back logs handed_back; got {end!r}"
        return
    assert clips_now == [], f"B-RV2: the {ending} ended the turn first, no at_checkout; {clips_now}"
    assert end == want_end, (
        f"B-RV2: the turn ended by {ending} logs tool_loop_end={want_end!r} as at BASE, never 'handed_back'; "
        f"got {end!r}"
    )


# =====================================================================================================================
# B-RV3: every way the executor reports a checkout hands back, not only click_at
# =====================================================================================================================


def _executed(inv: Any, action: str, resolved: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(
        call_id=inv.call_id, status="executed", action=action, reason=None, resolved=resolved
    )


async def test_brv3_type_at_with_enter_arriving_at_checkout_hands_back(tmp_path: Path) -> None:
    """B-RV3 / B4: a ``type_at`` with ``enter=true`` whose submit lands on a checkout (``checkout: True``) hands back
    exactly like a click."""
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(
        CID, lambda inv: _executed(inv, "typed 3 characters and pressed Enter", {"checkout": True})
    )
    script = [answer(tsl.S1), tool(TYPE_ENTER)]
    async with live_rig(tmp_path, browser, script=script) as rig:
        rig.llm.script_steps([answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(500)
        assert [inv.name for inv in browser.invocations] == ["type_at"], browser.invocations
        assert browser.invocations[0].arguments.get("press_enter") is True, browser.invocations[0]
        assert checkout_clips(rig) == [AT_CHECKOUT_TEXT], rig.audio.clip_plays()
        assert not rig.llm.step_prompts and DEAD not in rig.spoken_sources()
    assert_handed_back(rig, cid=CID, status="executed")


async def test_brv3_press_key_arriving_at_checkout_hands_back(tmp_path: Path) -> None:
    """B-RV3 / B4: a ``press_key`` (Enter on a focused "Proceed to checkout") that lands on a checkout hands back."""
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(CID, lambda inv: _executed(inv, "pressed Enter", {"checkout": True}))
    script = [answer(tsl.S1), tool(PRESS_ENTER)]
    async with live_rig(tmp_path, browser, script=script) as rig:
        rig.llm.script_steps([answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(500)
        assert [inv.name for inv in browser.invocations] == ["press_key"], browser.invocations
        assert checkout_clips(rig) == [AT_CHECKOUT_TEXT], rig.audio.clip_plays()
        assert not rig.llm.step_prompts and DEAD not in rig.spoken_sources()
    assert_handed_back(rig, cid=CID, status="executed")


async def test_brv3_page_overview_refused_at_checkout_hands_back(tmp_path: Path) -> None:
    """B-RV3 / B4 with A3: ``page_overview`` on a checkout page is refused with ``CHECKOUT_REASON``; the refusal
    hands back (card ``refused``)."""
    reason = checkout_reason()
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(CID, lambda inv: refused(inv, reason))
    script = [answer(tsl.S1), tool(OV_CALL)]
    kit = overview_kit(browser)
    async with tool_rig(
        tmp_path, kit=kit, policy=tsl.LOOP, script=script, frames=[hud_frame(0)]
    ) as rig:
        browser.now_ms = lambda: rig.clock.now_ms
        rig.llm.script_steps([answer(DEAD)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(500)
        assert [inv.name for inv in browser.invocations] == ["page_overview"], browser.invocations
        assert checkout_clips(rig) == [AT_CHECKOUT_TEXT], rig.audio.clip_plays()
        assert not rig.llm.step_prompts and DEAD not in rig.spoken_sources()
    assert_handed_back(rig, cid=CID, status="refused")


# =====================================================================================================================
# B4 negatives (the guard's "too wide" side): green at BASE, never pending
# =====================================================================================================================


@pytest.mark.parametrize(
    "resolved",
    [{"checkout": 1}, {"checkout": "true"}, {"checkout": False}, {"checkout": None}, {}],
    ids=["one", "string_true", "false", "none", "absent"],
)
async def test_b4_only_exact_true_hands_back(tmp_path: Path, resolved: dict[str, Any]) -> None:
    """B4 "exact ``True``": anything else in ``resolved["checkout"]`` is an ordinary executed result."""
    browser = FakeLiveBrowser([F0, FR])
    browser.resolved_extra[CID] = resolved
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1)
        await rig.turn_done(limit_ms=20_000)
    assert_not_handed_back(rig)


async def test_b4_failed_with_checkout_true_does_not_hand_back(tmp_path: Path) -> None:
    """B4 names an ``executed`` outcome only: a ``failed`` outcome that carries ``checkout: True`` is an ordinary
    failure."""
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(
        CID,
        lambda inv: ToolOutcome(
            call_id=inv.call_id,
            status="failed",
            action="failed click_at: the page went away",
            reason="the page went away",
            resolved={"checkout": True},
        ),
    )
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1)
        await rig.turn_done(limit_ms=20_000)
    assert_not_handed_back(rig)


@pytest.mark.parametrize(
    "variant",
    ["dot", "upper", "prefix", "other"],
)
async def test_b4_only_the_exact_checkout_reason_hands_back(tmp_path: Path, variant: str) -> None:
    """B4 "whose reason is ``CHECKOUT_REASON``": near misses (a trailing dot, upper case, a prefix) and another
    refusal are ordinary refusals; the loop continues."""
    base = checkout_reason()
    reason = {
        "dot": base + ".",
        "upper": base.upper(),
        "prefix": base[:20],
        "other": "frame not from this browser",
    }[variant]
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(CID, lambda inv: refused(inv, reason))
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1)
        await rig.turn_done(limit_ms=20_000)
    assert_not_handed_back(rig)


def test_b3_a_result_without_hand_back_dumps_as_at_base() -> None:
    """B3 "with no hand_back, the machine equals the frozen a9a11add machine", carried to what the session logs: a
    ``ToolResult`` without a hand-back serialises exactly as at BASE (no ``hand_back`` key), so every session-log and
    machine-dump differential of earlier tasks (T-0121/T-0172, T-0174, T-0179, T-0180, T-0181, T-0183, T-0191) stays
    byte-identical. Green at BASE; the prototype showed 24 of those rows red when ``hand_back: null`` was dumped."""
    ev = E.ToolResult(t_ms=7, call_id="c1", status="executed")
    want = {"type": "tool.result", "t_ms": 7, "call_id": "c1", "status": "executed"}
    assert ev.model_dump(mode="json") == want, ev.model_dump(mode="json")
    assert ev.model_dump() == want


class _CheckoutDryRun:
    """A dry-run executor that answers with the checkout facts (no browser): ``refused`` + ``CHECKOUT_REASON`` for
    ``mode="refused"``, ``dry_run`` with ``resolved={"checkout": True}`` otherwise."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def execute(self, inv: Any) -> ToolOutcome:
        if self.mode == "refused":
            return refused(inv, checkout_reason())
        return ToolOutcome(
            call_id=inv.call_id,
            status="dry_run",
            action=f"recorded {inv.name}",
            reason=None,
            resolved={"checkout": True},
        )


@pytest.mark.parametrize("mode", ["refused", "dry_run"])
async def test_b4_no_browser_nothing_changes(tmp_path: Path, mode: str) -> None:
    """B4 "No browser / dry-run: nothing changes": the same facts from a dry-run executor never hand back."""
    executor = WatchingExecutor(inner=_CheckoutDryRun(mode))
    async with tool_rig(
        tmp_path, policy=tsl.LOOP, executor=executor, script=[answer(tsl.S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1)
        await rig.turn_done(limit_ms=20_000)
    assert_not_handed_back(rig)
