"""T-0202 group B, B7 (and B9's frozen paths): ``wait`` through a real ``Session`` with no ``session.py`` change
(docs/tasks/T-0202.md).

**The rig** (B7's setup): a real ``Session`` (T-0134's ``feedback_rig``, loop on, 8 steps, T-0134c's label on) whose
live kit is ``cli.load_tool_kit(profile, live_browser=<executor>, wait_tool=True)``; the live browser is the real
``BrowserToolExecutor`` over T-0200's recording ``ShotDriver`` (scripted screenshots), wrapped so the rows see each
invocation, outcome and the after-frame of each call (T-0134's ``ExecutorWatch``). The fake LLM scripts ``type_at``,
then ``call:wait{}``, then an answer. ``WAIT_S`` is patched to 12.3 ms (the session's tool deadline runs on the
virtual clock, which the rig advances much faster than wall time).

**Oracles.** The settled payloads are typed by hand in the template's rendering (keys sorted, ``<|"|>`` quotes,
T-0113), the labels are T-0134 A3's texts typed from that task, the reason H-T202-5's text typed from this one.

**Readings (reported):** R1 as in ``test_t0202_executor_wait`` (the action is ``WAIT_ACTION``, ``"waited 1 s"``,
also under the patched ``WAIT_S``). R2 ("a wait counts as a step"): with every step a tool call, the turn runs
exactly ``max_tool_steps`` calls whether they are waits or scrolls.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, loop on (8 steps), live, 1280x800, repair as
``load_tool_kit`` decides, n = 1, deterministic. BASE 5821b628.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.computer import executor as executor_mod
from local_astra.config import Profile
from tests import _git_range as gr
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import builder_url, feedback_rig, live_feedback
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_computer_executor import CLOCK_MS, fake_jpeg
from tests.unit.test_session_live import last_user, step_body
from tests.unit.test_t0134_echo import ExecutorWatch
from tests.unit.test_t0200_blank_recapture import ShotDriver

BASE_SHA = "5821b628"
#: B's implementation tip (closed by the test-author; nothing after it touches local_astra/). A literal.
TIP_SHA = "be60168a"
Q = '<|"|>'
FAST = 0.0123
WAIT = "<|tool_call>call:wait{}<tool_call|>"
SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
TYPE = f"<|tool_call>call:type_at{{press_enter:true,text:{Q}milk{Q},x:500,y:100}}<tool_call|>"
WAIT_RUN_REASON = "you have already waited three times in a row; act on the screen or answer"
LU = "[This screenshot is the browser screen now, after your last action.]"
LN = "[This screenshot is the browser screen now; your last action did not change it.]"
DONE = "The results are on the screen."
#: four distinct painted 1280x800 screenshots
A, B, C, D = (fake_jpeg((1280, 800), 1, k) for k in range(4))


def payload(action: str, screenshot: str) -> str:
    return f"{{action:{Q}{action}{Q},screenshot:{Q}{screenshot}{Q},status:{Q}executed{Q}}}"


def refusal(reason: str) -> str:
    return f"{{reason:{Q}{reason}{Q},screenshot:{Q}unchanged{Q},status:{Q}refused{Q}}}"


def wait_kit(browser: Any, tmp: Path, monkeypatch: Any, wait_tool: bool = True) -> Any:
    if not wait_tool:  # the v1 kit (the control arm), as at BASE
        T.patch_detok_load(monkeypatch, T.Calls(), tmp)
        return cli.load_tool_kit(Profile(name="lean", tools="live"), live_browser=browser)[0]  # type: ignore[arg-type]
    assert "wait_tool" in inspect.signature(cli.load_tool_kit).parameters, (
        "T-0202 B8: cli.load_tool_kit(..., wait_tool=...) does not exist yet"
    )
    T.patch_detok_load(monkeypatch, T.Calls(), tmp)
    kit, _ = cli.load_tool_kit(
        Profile(name="lean", tools="live"),  # type: ignore[arg-type]
        live_browser=browser,
        wait_tool=True,
    )
    return kit


async def run_turn(
    tmp: Path,
    monkeypatch: Any,
    shots: list[bytes],
    calls: list[str],
    *,
    forever: bytes | None = None,
    extra: dict[str, Any] | None = None,
    limit_ms: int = 40_000,
    wait_tool: bool = True,
) -> tuple[Any, ExecutorWatch, ShotDriver]:
    """One turn: step 1 calls ``calls[0]``, each later step the next call, then ``DONE``; screenshots in order."""
    monkeypatch.setattr(executor_mod, "WAIT_S", FAST, raising=False)
    monkeypatch.setattr(executor_mod.secrets, "randbits", lambda k: 0)
    d = ShotDriver()
    d.queue(*shots)
    d.forever = forever
    ex = executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=d.viewport, driver=d, echo="model"
    )
    watch = ExecutorWatch(ex)
    kit = wait_kit(watch, tmp, monkeypatch, wait_tool)
    kw = {"live_feedback": live_feedback(screen_label=True, page_extent=False), **(extra or {})}
    steps = [[answer(f"Step {i}. "), tool(c)] for i, c in enumerate(calls[1:], 2)]
    steps.append([answer(DONE)])
    async with feedback_rig(
        tmp, watch, script=[answer("Step 1. "), tool(calls[0])], session_kwargs=kw, kit=kit
    ) as rig:
        ex._now_ms = lambda: rig.clock.now_ms
        rig.llm.script_steps(*steps)
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=limit_ms)
        await rig.advance(300)
        return rig, watch, d


def outcome(watch: ExecutorWatch, k: int) -> Any:
    return watch.outcomes[k]


def tail(rig: Any, k: int) -> str:
    return tsl.tail(tsl.step_request(rig, k))


async def test_wait_step_response_updated(tmp_path: Path, monkeypatch: Any) -> None:
    """B7: type_at (after-frame B), wait (after-frame C): step 3's continuation carries the generic executed payload
    ``{action: "waited 1 s", screenshot: "updated", status: "executed"}``, with no ``page`` key."""
    rig, watch, _ = await run_turn(tmp_path, monkeypatch, [A, B, C], [TYPE, WAIT])
    assert [i.name for i in watch.invocations] == ["type_at", "wait"], watch.invocations
    assert outcome(watch, 1).status == "executed", outcome(watch, 1)
    t3 = tail(rig, 3)
    assert payload("waited 1 s", "updated") in t3, (
        f"T-0202 B7: the wait's payload; tail {t3[-400:]}"
    )
    assert "page:" not in t3.split("waited 1 s")[-1], t3[-400:]


async def test_wait_step_response_unchanged(tmp_path: Path, monkeypatch: Any) -> None:
    """B7: the wait's after-frame has the type_at after-frame's pixels: ``screenshot: "unchanged"``; step 3's label is
    T-0134's ``LN`` after that frame."""
    rig, watch, _ = await run_turn(tmp_path, monkeypatch, [A, B, B], [TYPE, WAIT])
    assert outcome(watch, 1).status == "executed", outcome(watch, 1)
    assert payload("waited 1 s", "unchanged") in tail(rig, 3), tail(rig, 3)[-400:]


async def test_wait_frame_is_next_step_image(tmp_path: Path, monkeypatch: Any) -> None:
    """B7: step 3's final user message holds the wait's after-frame (C's bytes), immediately followed by ``LU``; the
    type_at after-frame (B) is not step 3's current image."""
    rig, watch, _ = await run_turn(tmp_path, monkeypatch, [A, B, C], [TYPE, WAIT])
    assert outcome(watch, 1).status == "executed", outcome(watch, 1)
    after = watch.frames_returned[-1]
    assert after.jpeg == C and outcome(watch, 1).resolved["frame_after"] == after.phash
    parts = last_user(step_body(rig, 3))["content"]
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    i = [k for k, p in enumerate(parts) if p.get("type") == "image_url"]
    assert urls and urls[0] == builder_url(after), (
        "T-0202 B7: the wait's frame is the current image"
    )
    assert parts[i[0] + 1] == {"type": "text", "text": LU}, parts[i[0] + 1]


async def test_repeat_guard_never_judges_wait(tmp_path: Path, monkeypatch: Any) -> None:
    """B7 (T-0137 on): type_at, then three waits on a screen that never changes: all three are executed and no
    continuation carries a repeat fact or a no-progress refusal (``wait`` is not in ``progress.ACTIONS``)."""
    rig, watch, _ = await run_turn(
        tmp_path,
        monkeypatch,
        [A, B],
        [TYPE, WAIT, WAIT, WAIT],
        forever=B,
        extra={"repeat_guard": True},
    )
    assert [o.status for o in watch.outcomes] == ["executed"] * 4, watch.outcomes
    for k in (3, 4, 5):
        t = tail(rig, k)
        assert "repeat_of_step" not in t and "did not change this screen" not in t, t[-400:]


async def test_fourth_wait_refused_loop_continues(tmp_path: Path, monkeypatch: Any) -> None:
    """B7 (adversarial; mutant: the run limit ends the turn or is not enforced): type_at, then four waits: the fourth's
    response is the refused payload with ``WAIT_RUN_REASON``, and the loop goes on to step 7, the answer."""
    rig, watch, _ = await run_turn(
        tmp_path, monkeypatch, [A, B, C], [TYPE, WAIT, WAIT, WAIT, WAIT], forever=C
    )
    assert [o.status for o in watch.outcomes] == ["executed"] * 4 + ["refused"], watch.outcomes
    assert watch.outcomes[-1].reason == WAIT_RUN_REASON
    assert refusal(WAIT_RUN_REASON) in tail(rig, 6), tail(rig, 6)[-400:]
    # step 1 (type_at) and steps 2-5 (the four waits) are calls; the refusal goes back in step 6's request, which
    # answers: step requests s2..s6 (five), and step 6 exists only if the loop went on after the refusal
    assert len(tsl.steps_of(rig)) == 5, "T-0202 B7: step 6 (the answer) follows the refusal"
    assert [c for c in rig.llm.step_prompts if "-s7-" in c[0]] == [], "no seventh step"


@pytest.mark.parametrize("kind", ["wait", "scroll"])
async def test_wait_counts_as_step(tmp_path: Path, monkeypatch: Any, kind: str) -> None:
    """B7/H-T202-3 (reading R2): eleven scripted calls alternating a scroll with ``kind``: the turn executes exactly
    ``max_tool_steps`` (8) calls whether the odd ones are waits or scrolls (the scroll arm is the green control)."""
    call = WAIT if kind == "wait" else SCROLL
    calls = [SCROLL if k % 2 == 0 else call for k in range(11)]
    rig, watch, _ = await run_turn(
        tmp_path, monkeypatch, [A], calls, forever=B, wait_tool=kind == "wait"
    )
    assert len(watch.invocations) == tsl.LOOP.max_tool_steps == 8, [
        (i.name, o.status) for i, o in zip(watch.invocations, watch.outcomes, strict=True)
    ]
    assert all(o.status == "executed" for o in watch.outcomes), watch.outcomes
    # premise of the hand-typed payload shape (the same rendering every row reads): the first scroll's response
    assert payload("scrolled down by 600 px", "updated") in tail(rig, 2), tail(rig, 2)[-400:]


FROZEN = (
    "local_astra/conductor/",
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/payloads/",
)


def frozen_changes(base: str, tip: str) -> list[str]:
    """The frozen paths that differ between two commits (never the working tree); ``tip`` must descend from
    ``base``."""
    gr.require_commits(base, tip)
    assert gr.is_ancestor(base, tip), f"premise: {base} is an ancestor of {tip}"
    return gr.changed(base, tip, *FROZEN)


def test_conductor_diff_empty() -> None:
    """B7/B9 (negative): nothing under ``local_astra/conductor/`` (session, turn machine, progress, announced_call,
    warmup, tool_executor, prompt_builder: A's change is already in BASE) nor the loop, judge, v1 tools or payloads
    differs between BASE 5821b628 and TIP be60168a (B's implementation; closed, T-0094: a fixed range, never the
    working tree). Skips with the missing sha when a commit is absent."""
    got = frozen_changes(BASE_SHA, TIP_SHA)
    assert got == [], f"T-0202 B7/B9: frozen paths changed in {BASE_SHA}..{TIP_SHA}: {got}"
