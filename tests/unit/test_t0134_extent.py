"""T-0134 A4 (A-extent): the page extent in the label, at the session over the fake live browser and at the executor
over a fake driver (docs/tasks/T-0134.md).

**The surface these rows pin** (the task lets the implementer choose between an ``extent_of(identity)`` lookup and a
field on the executor's own frame record; the rows need one name, so they take the lookup; if the implementer
chooses otherwise, say so and the rows move):

- ``BrowserToolExecutor.extent_of(phash: str) -> dict | None``: the extent captured with the own frame whose
  ``phash`` is given, ``{"above_px", "below_px", "viewport_px"}`` in CSS px, or ``None`` when the read failed, was
  unstable (the metrics read before and after the screenshot differ) or the frame is unknown;
- the driver method (``runtime.PlaywrightDriver``, A4's "one driver method that reads layout metrics"):
  ``async layout_metrics() -> {"scroll_y", "viewport_height", "content_height"}`` (CSS px), read before and after the
  screenshot of every own capture. ``above = scroll_y``, ``below = content_height - scroll_y - viewport_height``;
- the session reads the extent of the frame it shows from the live browser (``FakeLiveBrowser.extent_of`` accepts
  the executor phash, the capture identity or the frame), and ``{P}`` is `` Page beyond this screen: {below} below,
  {above} above.``, each ``nothing`` under 0.05 screens, else ``about {v:.1f} screens``.

Hand table (viewport 800 px): 0 and 39.2 px (0.049) -> ``nothing``; 40 px (0.05) -> ``about 0.1 screens``; 480 px
(0.6) -> ``about 0.6 screens``; 1880 px (2.35, stored as 2.3500000000000001) -> ``about 2.4 screens``.

Reading (reported): the existing fake drivers (``tests/unit/test_computer_executor.py``,
``test_computer_live_options.py``) close by a raising sentinel, so an executor that calls ``layout_metrics`` on a
driver without it reddens ~all their rows (R11 in the report). These rows assume the executor looks the method up on
the driver's *class* and reads no extent when it is absent.

All rows ``pending(task="T-0134")``.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, loop on, 1280x800, n = 1, deterministic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.toolcall import ToolOutcome
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    FakeLiveBrowser,
    feedback_rig,
    live_feedback,
    live_frame,
)
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_computer_executor import FakeDriver, computer_executor, inv, own_frame
from tests.unit.test_session_live import last_user, pass1_body, step_body

STEM = "This screenshot is the browser screen now"
CLAUSE_RE = re.compile(
    r"^ Page beyond this screen: (nothing|about \d+\.\d screens) below, (nothing|about \d+\.\d screens) above\.$"
)
SCROLL = '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'
F0 = live_frame("onset")
F1 = live_frame("after-1")


def ext(above: float, below: float, viewport: float = 800.0) -> dict[str, float]:
    return {"above_px": above, "below_px": below, "viewport_px": viewport}


def both() -> Any:
    return live_feedback(screen_label=True, page_extent=True)


def labels_of(body: dict[str, Any]) -> list[str]:
    out = []
    for m in body["messages"]:
        if isinstance(m["content"], list):
            out += [
                p["text"]
                for p in m["content"]
                if p.get("type") == "text" and STEM in p.get("text", "")
            ]
    return out


def the_label(body: dict[str, Any]) -> str:
    found = labels_of(body)
    assert len(found) == 1, f"T-0134 A3: one label per request; got {found}"
    parts = last_user(body)["content"]
    i = [k for k, p in enumerate(parts) if p.get("type") == "text" and p.get("text") == found[0]]
    assert i and parts[i[0] - 1].get("type") == "image_url", "the label follows the current frame"
    return found[0]


def clause(label: str) -> str:
    """``{P}``: the text between the label's first sentence and its closing bracket."""
    for head in (
        "[This screenshot is the browser screen now.",
        "[This screenshot is the browser screen now, after your last action.",
        "[This screenshot is the browser screen now; your last action did not change it.",
    ):
        if label.startswith(head) and label.endswith("]"):
            return label[len(head) : -1]
    raise AssertionError(f"not an A3 label: {label!r}")


# =====================================================================================================
# the session side
# =====================================================================================================

TABLE = [
    pytest.param(
        ext(0.0, 1880.0),
        " Page beyond this screen: about 2.4 screens below, nothing above.",
        id="0-2.35",
    ),
    pytest.param(
        ext(480.0, 40.0),
        " Page beyond this screen: about 0.1 screens below, about 0.6 screens above.",
        id="0.6-0.05",
    ),
    pytest.param(
        ext(39.2, 0.0), " Page beyond this screen: nothing below, nothing above.", id="0.049-0"
    ),
]


@pytest.mark.parametrize(("extent", "want"), TABLE)
async def test_extent_clause_format(tmp_path: Path, extent: dict[str, float], want: str) -> None:
    """A4: the hand table (0, 0.049, 0.05, 0.6 and 2.35 screens; above and below; ``nothing``) on a toolless live
    turn: step 1's label is ``[This screenshot is the browser screen now.{P}]`` with ``{P}`` written out."""
    browser = FakeLiveBrowser([F0])
    browser.extents = {F0.phash: extent}
    async with feedback_rig(
        tmp_path, browser, script=[answer("A shop. ")], session_kwargs={"live_feedback": both()}
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        label = the_label(pass1_body(rig))
        assert label == "[This screenshot is the browser screen now." + want + "]", label


async def test_extent_keyed_by_capture(tmp_path: Path) -> None:
    """A4 (adversarial; reviewer mutation 3, the latest extent rather than the shown frame's): step 1 scrolls; the
    browser captures F1 (its extent: 0.75 screens above) but reports a ``frame_after`` that is not F1, and the fresh
    frame fails, so step 2 still shows F0 and must carry F0's extent (2.0 screens below, nothing above), with the
    ``unchanged`` wording."""
    browser = FakeLiveBrowser([F0], post_frames=[F1])
    browser.extents = {F0.phash: ext(0.0, 1600.0), F1.phash: ext(600.0, 1000.0)}

    def scrolled(i: Any) -> ToolOutcome:
        browser.capture_post()  # F1 is now latest_frame, with its own extent
        browser.frame_mode = "raise"  # and no fresh frame can be taken
        return ToolOutcome(
            call_id=i.call_id,
            status="executed",
            action="scrolled down by 600 px",
            reason=None,
            resolved={"frame_after": "0123456789abcdef"},
        )

    browser.outcome("t1-call", scrolled)
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer("I will scroll."), tool(SCROLL)],
        session_kwargs={"live_feedback": both()},
    ) as rig:
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1, limit_ms=20_000)
        await rig.turn_done(limit_ms=20_000)
        assert browser.latest_frame is not None and browser.latest_frame.phash == F1.phash, (
            "premise"
        )
        s2 = step_body(rig, 2)
        assert the_label(s2) == (
            "[This screenshot is the browser screen now; your last action did not change it."
            " Page beyond this screen: about 2.0 screens below, nothing above.]"
        ), the_label(s2)


async def test_unstable_or_failed_read_omits_clause(tmp_path: Path) -> None:
    """A4 (reviewer mutation 9): the shown frame's extent is unknown (an unstable or failed read: ``None``), so
    ``{P}`` is empty (``L1`` exactly) and a ``note`` record says so (its data mentions the extent)."""
    browser = FakeLiveBrowser([F0])
    browser.extents = {F0.phash: None}
    async with feedback_rig(
        tmp_path, browser, script=[answer("A shop. ")], session_kwargs={"live_feedback": both()}
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        assert the_label(pass1_body(rig)) == "[This screenshot is the browser screen now.]"
    notes = [r["data"] for r in rig.log_records if r["kind"] == "note"]
    assert any("extent" in str(n).lower() for n in notes), (
        f"T-0134 A4: no note about the extent; {notes}"
    )


async def test_extent_numbers_only(tmp_path: Path) -> None:
    """A4: across a three-step turn every ``{P}`` matches the numbers-only pattern (no page text, title, URL or DOM
    content), and is not empty where the extent is known."""
    f2 = live_frame("after-2")
    browser = FakeLiveBrowser([F0], post_frames=[F1, f2])
    browser.extents = {
        F0.phash: ext(0.0, 2000.0),
        F1.phash: ext(600.0, 1400.0),
        f2.phash: ext(1200.0, 800.0),
    }
    async with feedback_rig(
        tmp_path,
        browser,
        script=[answer("I will scroll."), tool(SCROLL)],
        session_kwargs={"live_feedback": both()},
    ) as rig:
        rig.llm.script_steps([answer("Again."), tool(SCROLL)], [answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 2, limit_ms=20_000)
        await rig.turn_done(limit_ms=20_000)
        got = [
            clause(the_label(b)) for b in (pass1_body(rig), step_body(rig, 2), step_body(rig, 3))
        ]
        for c in got:
            assert CLAUSE_RE.match(c), c
        assert got == [
            " Page beyond this screen: about 2.5 screens below, nothing above.",
            " Page beyond this screen: about 1.8 screens below, about 0.8 screens above.",
            " Page beyond this screen: about 1.0 screens below, about 1.5 screens above.",
        ], got


# =====================================================================================================
# the executor side
# =====================================================================================================


class MetricsDriver(FakeDriver):
    """T-0111c's fake driver plus ``layout_metrics``: a page ``content_height`` tall, scrolled by ``scroll_by``.
    ``unstable``: every other read moves by 1 px (so a read before and after one screenshot differ); ``broken``:
    the read raises."""

    def __init__(self, content_height: float = 2680.0) -> None:
        super().__init__((1280, 800))
        self.content_height = content_height
        self.scroll_y = 0.0
        self.unstable = False
        self.broken = False
        self.reads = 0

    async def layout_metrics(self) -> dict[str, float]:
        self.calls.append(("layout_metrics", ()))
        self.reads += 1
        if self.broken:
            raise RuntimeError("CDP: Page.getLayoutMetrics failed")
        jitter = float(self.reads % 2) if self.unstable else 0.0
        return {
            "scroll_y": self.scroll_y + jitter,
            "viewport_height": 800.0,
            "content_height": self.content_height,
        }

    async def scroll_by(self, dx: float, dy: float) -> None:
        await super().scroll_by(dx, dy)
        self.scroll_y = max(0.0, min(self.content_height - 800.0, self.scroll_y + dy))


def make(driver: MetricsDriver) -> Any:
    ex = computer_executor().BrowserToolExecutor(
        now_ms=lambda: 1, viewport=driver.viewport, driver=driver
    )
    assert callable(getattr(ex, "extent_of", None)), (
        "T-0134 A4: BrowserToolExecutor.extent_of does not exist yet"
    )
    return ex


async def test_executor_extent_per_capture() -> None:
    """A4 at the executor: the onset frame's extent is 0 above and 1880 below (2680 - 800); after ``scroll down``
    (600 px) the post-action frame's is 600 above and 1280 below, and the onset frame's is unchanged (keyed by
    capture). The metrics were read around each screenshot."""
    d = MetricsDriver()
    ex = make(d)
    f0 = await own_frame(ex)
    out = await ex.execute(inv("scroll", None, direction="down"))
    assert out.status == "executed", out
    f1 = ex.latest_frame
    assert ex.extent_of(f0.phash) == {"above_px": 0.0, "below_px": 1880.0, "viewport_px": 800.0}
    assert ex.extent_of(f1.phash) == {"above_px": 600.0, "below_px": 1280.0, "viewport_px": 800.0}
    assert d.reads >= 4, f"read before and after each of the two screenshots; {d.reads}"
    assert ex.extent_of("0" * 16) is None


@pytest.mark.parametrize("fault", ["unstable", "broken"])
async def test_executor_unstable_or_failed_read(fault: str) -> None:
    """A4: metrics that differ before and after the screenshot, or a read that raises, give no extent for that
    frame (``None``), and the frame itself is still taken (the capture never fails because of the extent)."""
    d = MetricsDriver()
    setattr(d, fault, True)
    ex = make(d)
    f0 = await own_frame(ex)
    assert f0.width == 1280, "the frame is taken"
    assert ex.extent_of(f0.phash) is None
    assert any(c[0] == "layout_metrics" for c in d.calls), "premise: the metrics were read"


# =====================================================================================================
# Review A of group A (2026-10-22): o12 and o14
# =====================================================================================================


async def test_extent_divides_by_the_viewport(tmp_path: Path) -> None:
    """Review A o12 (mutant: dividing by 800): a 600 px viewport with 300 px above and 1500 px below is 0.5 and 2.5
    screens (by hand; over 800 it would read 0.4 and 1.9)."""
    browser = FakeLiveBrowser([F0])
    browser.extents = {F0.phash: ext(300.0, 1500.0, 600.0)}
    async with feedback_rig(
        tmp_path, browser, script=[answer("A shop. ")], session_kwargs={"live_feedback": both()}
    ) as rig:
        await rig.ask("what is on the screen")
        await rig.turn_done()
        assert the_label(pass1_body(rig)) == (
            "[This screenshot is the browser screen now."
            " Page beyond this screen: about 2.5 screens below, about 0.5 screens above.]"
        ), the_label(pass1_body(rig))


class HungMetricsDriver(MetricsDriver):
    """``layout_metrics`` never answers in any useful time (an hour)."""

    async def layout_metrics(self) -> dict[str, float]:
        import asyncio

        self.calls.append(("layout_metrics", ()))
        self.reads += 1
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def test_executor_hung_metrics_read_is_bounded() -> None:
    """Review A o14 (mutant: ``METRICS_TIMEOUT_S = 3600``): a hung metrics read is bounded by
    ``executor.METRICS_TIMEOUT_S`` (well under the 8 s action bound): ``frame()`` returns within the timeout plus a
    2 s margin, the frame is taken and its extent is ``None``. (The session's note for an unknown extent is
    ``test_unstable_or_failed_read_omits_clause``'s: the session sees the same ``None``.)"""
    import asyncio
    import time

    mod = computer_executor()
    bound = float(getattr(mod, "METRICS_TIMEOUT_S", 0.0))
    assert 0 < bound < mod.ACTION_TIMEOUT_S, f"METRICS_TIMEOUT_S {bound}"
    d = HungMetricsDriver()
    ex = make(d)
    t0 = time.monotonic()
    try:
        f0 = await asyncio.wait_for(own_frame(ex), bound + 2.0)
    except TimeoutError as exc:
        raise AssertionError(
            f"T-0134 A4: a hung metrics read held frame() past {bound + 2.0} s"
        ) from exc
    elapsed = time.monotonic() - t0
    assert elapsed <= bound + 2.0, elapsed
    assert f0.width == 1280 and ex.extent_of(f0.phash) is None
    assert d.reads >= 1, "premise: the metrics were read"
