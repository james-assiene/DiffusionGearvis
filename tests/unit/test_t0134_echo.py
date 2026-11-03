"""T-0134 A1/A2 (A-echo): ``BrowserToolExecutor(..., echo="pixels" | "model")`` over T-0111c's fake driver
(docs/tasks/T-0134.md).

**The surface these rows pin:** ``BrowserToolExecutor(..., echo: Literal["pixels", "model"] = "pixels")``; any other
value raises ``ValueError``. With ``echo="model"`` an executed ``click_at`` reads ``clicked at (X, Y)`` and a
``type_at`` ``clicked at (X, Y) and typed N characters[ and pressed Enter]``, ``X, Y`` the 0-999 arguments of the
invocation the executor executed; ``resolved`` keeps ``x_px``/``y_px``. Scroll and key actions are unchanged.

Expected values by hand: ``to_pixels(c, size) = (2*c*(size-1) + 999) // 1998``, so on 1280x800
``(500, 250) -> (640, 200)`` and ``(120, 640) -> (154, 512)``; ``scroll down`` moves ``800 * 3 // 4 = 600`` px.

``test_pixels_default_is_base`` holds at BASE c29a8d9 and carries no marker (A8). Every other row is
``pending(task="T-0134")``: at BASE the keyword does not exist (a named ``AssertionError`` from ``make_echo``).

``test_echo_uses_executed_arguments`` (adversarial; mutant: echo the model's original call, not the arguments the
executor ran with) runs a real ``Session(..., reground=True)`` over the fake LLM with the *real* executor (fake
driver) as its live browser: pass 2 rewrites the click, and step 2's continuation must carry the rewritten 0-999
values, not pass 1's and not pixels.

Regime: CPU, fake driver (no browser), 1280x800, ``ScriptedClock`` for the session row, n = 1, deterministic.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.perception import Frame
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import feedback_rig
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_computer_executor import FakeDriver, computer_executor, inv, own_frame

PENDING_NOTE = "T-0134 A1: BrowserToolExecutor has no `echo` keyword yet"


def make_echo(echo: Any = None, fake: FakeDriver | None = None) -> tuple[Any, FakeDriver]:
    """The executor over the fake driver with ``echo=echo`` (``None``: the keyword is not passed)."""
    cls = computer_executor().BrowserToolExecutor
    fake = fake or FakeDriver()
    kw: dict[str, Any] = {}
    if echo is not None:
        assert "echo" in inspect.signature(cls.__init__).parameters, PENDING_NOTE
        kw["echo"] = echo
    return cls(now_ms=lambda: 1, viewport=fake.viewport, driver=fake, **kw), fake


async def _outcomes(ex: Any) -> dict[str, Any]:
    f: Frame = await own_frame(ex)
    out: dict[str, Any] = {}
    click = await ex.execute(inv("click_at", f, x=500, y=250))
    out["click"] = click
    f = ex.latest_frame
    out["type"] = await ex.execute(inv("type_at", f, x=120, y=640, text="hello", press_enter=False))
    f = ex.latest_frame
    out["type_enter"] = await ex.execute(
        inv("type_at", f, x=120, y=640, text="mugs!", press_enter=True)
    )
    out["scroll"] = await ex.execute(inv("scroll", None, direction="down"))
    out["key"] = await ex.execute(inv("press_key", None, combo="Enter"))
    for k, o in out.items():
        assert o.status == "executed", (k, o)
    return out


BASE_ACTIONS = {
    "click": "clicked at (640, 200)",
    "type": "clicked at (154, 512) and typed 5 characters",
    "type_enter": "clicked at (154, 512) and typed 5 characters and pressed Enter",
    "scroll": "scrolled down by 600 px",
    "key": "pressed Enter",
}
MODEL_ACTIONS = {
    "click": "clicked at (500, 250)",
    "type": "clicked at (120, 640) and typed 5 characters",
    "type_enter": "clicked at (120, 640) and typed 5 characters and pressed Enter",
    "scroll": "scrolled down by 600 px",
    "key": "pressed Enter",
}


async def test_model_echo_click_and_type() -> None:
    """A2 (reviewer mutation 4): with ``echo="model"`` the click and both type actions read the call's 0-999
    values; ``resolved`` keeps the pixels (640, 200) and (154, 512); no pixel number is in the action string."""
    ex, _ = make_echo("model")
    out = await _outcomes(ex)
    for k in ("click", "type", "type_enter"):
        assert out[k].action == MODEL_ACTIONS[k], (k, out[k].action)
    assert (out["click"].resolved["x_px"], out["click"].resolved["y_px"]) == (640, 200)
    assert (out["type"].resolved["x_px"], out["type"].resolved["y_px"]) == (154, 512)
    assert out["type_enter"].resolved["press_enter"] is True
    for k, px in (("click", ("640", "200")), ("type", ("154", "512"))):
        assert not any(p in out[k].action for p in px), (k, out[k].action)


async def test_pixels_default_is_base() -> None:
    """A1/A8 (green at BASE, no marker): without the keyword the wording is BASE's, written out by hand."""
    ex, _ = make_echo(None)
    out = await _outcomes(ex)
    assert {k: o.action for k, o in out.items()} == BASE_ACTIONS


async def test_explicit_pixels_equals_default() -> None:
    """A1: ``echo="pixels"`` is the default: the same strings as BASE's and the same ``resolved`` pixels."""
    ex, _ = make_echo("pixels")
    out = await _outcomes(ex)
    assert {k: o.action for k, o in out.items()} == BASE_ACTIONS
    assert getattr(ex, "echo", None) == "pixels", "A5 reads the mode by duck typing (browser.echo)"
    ex2, _ = make_echo("model")
    assert getattr(ex2, "echo", None) == "model"


@pytest.mark.parametrize("junk", ["Model", "px", "", "0-999", 1, True])
def test_echo_refuses_junk(junk: Any) -> None:
    """A1: any value other than "pixels" or "model" raises ``ValueError`` at construction."""
    cls = computer_executor().BrowserToolExecutor
    assert "echo" in inspect.signature(cls.__init__).parameters, PENDING_NOTE
    fake = FakeDriver()
    with pytest.raises(ValueError):
        cls(now_ms=lambda: 1, viewport=fake.viewport, driver=fake, echo=junk)


async def test_scroll_and_key_unchanged() -> None:
    """A2: scroll and key actions read the same under both modes (hand values)."""
    ex, _ = make_echo("model")
    out = await _outcomes(ex)
    assert out["scroll"].action == "scrolled down by 600 px"
    assert out["key"].action == "pressed Enter"
    assert out["scroll"].resolved["dy_px"] == 600


# --------------------------------------------------------------------------------------------- the adversarial row


class ExecutorWatch:
    """The real executor as the live browser, plus ``frames_returned`` (what the pass-2 fake LLM needs to tell the
    crop apart). Delegates every other name."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.frames_returned: list[Frame] = []
        self.invocations: list[Any] = []
        self.outcomes: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def frame(self) -> Frame:
        f = await self._inner.frame()
        self.frames_returned.append(f)
        return f

    async def execute(self, i: Any) -> Any:
        self.invocations.append(i)
        out = await self._inner.execute(i)
        self.outcomes.append(out)
        if self._inner.latest_frame is not None:
            self.frames_returned.append(self._inner.latest_frame)
        return out


async def test_echo_uses_executed_arguments(tmp_path: Path) -> None:
    """A2 (adversarial; mutant: echo the model's original call): a ReGround-rewritten click executes at pass 2's
    point, and step 2's continuation reads ``clicked at (X', Y')`` with the rewritten 0-999 values; neither pass 1's
    ``(500, 250)`` nor the pixel form of either point appears as the action."""
    ex, _ = make_echo("model")
    watch = ExecutorWatch(ex)
    async with feedback_rig(
        tmp_path,
        watch,
        script=[answer(tsl.S1), tool(VALID_CLICK)],
        session_kwargs={"reground": True},
        reground_llm=True,
    ) as rig:
        ex._now_ms = lambda: rig.clock.now_ms
        rig.llm.script_pass2(
            [answer(""), tool("<|tool_call>call:click_at{x:900,y:100}<tool_call|>")]
        )
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        (executed,) = [i for i in watch.invocations if i.call_id == "t1-call"]
        x, y = int(executed.arguments["x"]), int(executed.arguments["y"])
        assert (x, y) != (500, 250), "premise: pass 2 rewrote the click"
        (out,) = [o for o in watch.outcomes if o.call_id == "t1-call"]
        assert out.status == "executed", out
        assert out.action == f"clicked at ({x}, {y})", out.action
        tail = tsl.tail(tsl.step_request(rig, 2))
        assert f"clicked at ({x}, {y})" in tail, tail[-300:]
        for wrong in (
            "clicked at (500, 250)",
            "clicked at (640, 200)",
            f"clicked at ({to_pixels(x, 1280)}, {to_pixels(y, 800)})",
        ):
            if wrong != f"clicked at ({x}, {y})":
                assert wrong not in tail, (wrong, tail[-300:])
