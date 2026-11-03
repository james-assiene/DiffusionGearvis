"""T-0198 criteria 4, 5 and 1 through the live session: the session sets the browser's ``raise_window`` where a live
turn's vision choice is frozen, and a conversational turn never fronts (docs/tasks/T-0198.md).

**Rigs.** T-0177's ``vision_rig`` (a live tool loop over a ``FakeLiveBrowser``, a socketless ``HudServer``, one greeted
page; ``select`` sends ``astra.vision.select`` over the wire). ``RaiseBrowser`` is T-0177's ``VisionBrowser`` with a
``raise_window`` *property* (so ``inspect.getattr_static`` finds it on the class) whose setter records each value with
the length of ``browser.calls`` at that moment, so a row can tell whether the value was set before the turn's first
``execute``. ``AnsweringBrowser`` has no such attribute, but its ``__getattr__`` would answer one (and records the
question); the criterion says the session checks with ``inspect.getattr_static``, so it must neither ask nor set.
The fronting rows run the real ``BrowserToolExecutor`` headed (``headless=False``) over
``test_t0198_front_executor.FrontDriver``, through the live loop and through the loop-off ``execute_tool`` path.

**Readings (reported):**

- C4 "sets ... where a live turn's vision choice is fixed": once per turn (the freeze runs once per turn), so each turn
  adds exactly one value; the rows pin the per-turn sequence.
- C5 "a conversational turn": a live turn whose model answers without a tool call; its first snapshot calls
  ``browser.frame()``. The row ends with a tool turn that fronts once (so the rig can front at all).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools live (loop on unless named), n = 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import live_kit
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_computer_executor import CLOCK_MS, INPUT
from tests.unit.test_t0177_shared_frame import F0, VisionBrowser, one_turn, select, vision_rig
from tests.unit.test_t0198_front_executor import FRONT, FrontDriver, fronts


class RaiseBrowser(VisionBrowser):
    """``VisionBrowser`` with a recorded ``raise_window`` (module docstring). Starts at ``None`` (never set)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self._raise: bool | None = None
        self.raise_log: list[tuple[bool, int]] = []

    @property
    def raise_window(self) -> bool | None:
        return self._raise

    @raise_window.setter
    def raise_window(self, value: bool) -> None:
        self.raise_log.append((value, len(self.calls)))
        self._raise = value


class AnsweringBrowser(VisionBrowser):
    """No ``raise_window``; a ``__getattr__`` that would answer for it, and a ``__setattr__`` that records it."""

    def __init__(self, *a: Any, **k: Any) -> None:
        object.__setattr__(self, "asked", [])
        object.__setattr__(self, "set_raise", [])
        super().__init__(*a, **k)

    def __getattr__(self, name: str) -> Any:
        if name == "raise_window":
            self.asked.append(name)
            return True
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "raise_window":
            self.set_raise.append(value)
        object.__setattr__(self, name, value)


def values(b: RaiseBrowser) -> list[bool]:
    return [v for v, _ in b.raise_log]


def executes(b: Any) -> list[int]:
    return [i for i, c in enumerate(b.calls) if c[0] == "execute"]


async def tool_turn(rig: Any, turn: str = "t1") -> None:
    """One live loop turn ``turn`` (the n-th): step 1 clicks (``VALID_CLICK``), step 2 answers."""
    from local_astra.contracts.turn import TurnState

    n = int(turn[1:])
    rig.llm.script = [answer(tsl.S1), tool(VALID_CLICK)]
    rig.llm.script_steps([answer(tsl.S3)])
    await rig.ask(tsl.QUESTION)
    await tsl.until_steps(rig, 1, turn=turn, limit_ms=20_000, step=100)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"{turn} never ended",
        limit_ms=20_000,
    )


# =====================================================================================================
# C4: the session sets raise_window at the freeze
# =====================================================================================================


async def test_vision_browser_turn_sets_true(tmp_path: Path) -> None:
    """C4: a browser-vision tool turn sets ``raise_window = True`` once, before the turn's first ``execute``."""
    b = RaiseBrowser([F0])
    async with vision_rig(tmp_path, b) as rig:
        await tool_turn(rig)
    assert executes(b), f"premise: the turn executed its click; {b.calls}"
    assert values(b) == [True], f"T-0198 C4: {b.raise_log}"
    assert b.raise_log[0][1] <= executes(b)[0], (
        f"T-0198 C4: set at the freeze, before the first execute; {b.raise_log}, calls {b.calls}"
    )


async def test_vision_shared_turn_sets_raise_window_false(tmp_path: Path) -> None:
    """C4, H-T198-2: with ``shared`` selected, the turn sets ``raise_window = False`` once; back to ``browser``, the
    next turn sets ``True``."""
    b = RaiseBrowser([F0])
    async with vision_rig(tmp_path, b, script=[answer("I see a page. ")]) as rig:
        await select(rig, "shared")
        await one_turn(rig)
        assert values(b) == [False], f"T-0198 C4: {b.raise_log}"
        await select(rig, "browser")
        await rig.advance(300)
        await tool_turn(rig, "t2")
    assert values(b) == [False, True], f"T-0198 C4: {b.raise_log}"


async def test_select_mid_turn_applies_next_turn(tmp_path: Path) -> None:
    """C4 with T-0177 A5: ``shared`` selected while a browser turn is THINKING leaves that turn's value ``True`` to its
    end; the next turn sets ``False``."""
    from local_astra.contracts.turn import TurnState

    b = RaiseBrowser([F0])
    async with vision_rig(tmp_path, b, script=[answer("A shop. ")]) as rig:
        rig.llm.hold()
        await rig.ask(tsl.QUESTION)
        assert rig.state.state is TurnState.THINKING, f"premise: THINKING; {rig.state.state}"
        await select(rig, "shared")
        assert values(b) == [True], (
            f"T-0198 C4: a select mid-turn changed the open turn; {b.raise_log}"
        )
        rig.llm.release()
        await rig.turn_done(limit_ms=20_000)
        assert values(b) == [True], f"T-0198 C4: {b.raise_log}"
        await rig.advance(300)
        rig.llm.script = [answer("A shop. ")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
    assert values(b) == [True, False], f"T-0198 C4: {b.raise_log}"


async def test_browser_without_attribute_untouched(tmp_path: Path) -> None:
    """C4: a ``LiveBrowser`` without ``raise_window`` is untouched: its answering ``__getattr__`` is never asked, the
    attribute is never set, in a browser turn and a shared turn. The same turns on ``RaiseBrowser`` set it (the
    premise that the session sets it at all)."""
    seen = RaiseBrowser([F0])
    async with vision_rig(tmp_path / "a", seen, script=[answer("A page. ")]) as rig:
        await one_turn(rig)
    assert values(seen) == [True], f"premise (C4): the session sets raise_window; {seen.raise_log}"
    b = AnsweringBrowser([F0])
    async with vision_rig(tmp_path / "b", b, script=[answer("A page. ")]) as rig:
        await one_turn(rig)
        await select(rig, "shared")
        await rig.advance(300)
        rig.llm.script = [answer("A screen. ")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
    assert b.asked == [], (
        f"T-0198 C4: the session consulted __getattr__ for raise_window: {b.asked}"
    )
    assert b.set_raise == [] and "raise_window" not in vars(b), (
        f"T-0198 C4: raise_window was set on a browser without it: {b.set_raise}"
    )


# =====================================================================================================
# C5, C1: the real executor through the session
# =====================================================================================================


def headed() -> tuple[Any, FrontDriver]:
    from local_astra.computer import executor as executor_mod

    d = FrontDriver()
    ex = executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=d.viewport, driver=d, headless=False
    )
    return ex, d


async def test_conversational_turn_never_fronts(tmp_path: Path) -> None:
    """C5 adversarial: two live turns that answer without a tool call each take the browser's ``frame()`` and front
    nothing; a third turn that clicks fronts exactly once, before its click."""
    ex, d = headed()
    async with vision_rig(tmp_path, ex, kit=live_kit(ex), script=[answer("Hello there. ")]) as rig:
        ex._now_ms = lambda: rig.clock.now_ms
        await one_turn(rig, "hello")
        await rig.advance(300)
        rig.llm.script = [answer("Nice to meet you. ")]
        await rig.ask("how are you")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
        assert d.shots, "premise: the conversational turns took the browser's frame"
        assert fronts(d) == [], f"T-0198 C5: a conversational turn fronted {fronts(d)}"
        await rig.advance(300)
        mark = len(d.calls)
        await tool_turn(rig, "t3")
    assert d.dispatched(mark), f"premise: the tool turn clicked; {d.calls[mark:]}"
    assert len(fronts(d)) == 1, f"T-0198 C1: the tool turn fronts once; {fronts(d)}"
    i_front = next(i for i, c in enumerate(d.calls) if c[0] == FRONT)
    assert i_front < min(i for i, c in enumerate(d.calls) if c[0] in INPUT), d.calls[mark:]


async def test_loop_off_execute_tool_path_same_policy(tmp_path: Path) -> None:
    """C1 on the loop-off path (``Session.execute_tool``, the machine's default policy): a turn's one click fronts once,
    before the click; the next turn's click fronts once more."""
    from tests.unit._tool_turn_support import tool_rig

    ex, d = headed()
    async with tool_rig(
        tmp_path, kit=live_kit(ex), script=[answer(tsl.S1), tool(VALID_CLICK)]
    ) as rig:
        ex._now_ms = lambda: rig.clock.now_ms
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=20_000)
        assert d.dispatched(), f"premise: the loop-off turn clicked; {d.calls}"
        assert len(fronts(d)) == 1, f"T-0198 C1: {fronts(d)}"
        i_front = next(i for i, c in enumerate(d.calls) if c[0] == FRONT)
        assert i_front < min(i for i, c in enumerate(d.calls) if c[0] in INPUT), d.calls
        await rig.advance(300)
        mark = len(d.calls)
        rig.llm.script = [answer(tsl.S1), tool(VALID_CLICK)]
        await rig.ask("again")
        await rig.until(lambda: len(rig.lines()) >= 2, "turn 2 never ended", limit_ms=20_000)
    assert d.dispatched(mark), f"premise: the second turn clicked; {d.calls[mark:]}"
    assert len(fronts(d)) == 2, f"T-0198 C1: once per turn; {fronts(d)}"
