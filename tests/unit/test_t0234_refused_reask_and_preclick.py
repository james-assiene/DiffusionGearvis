"""T-0234: the empty-step re-ask is also armed after a *refused* browser action (A, amends T-0231 R-B3), and the
pre-click guard decides on pixels when the crop was taken, a layout-shift rect near the point no longer refusing on
its own (B, amends T-0204 B2) (docs/tasks/T-0234.md).

**Evidence** (logs/sessions/20261102-155518-245443.jsonl): 4 of 6 pre-click refusals had ``changed_share`` 0.0 (a
shift rect alone refused), and 4 empty steps right after a refusal ended their turns silent.

**Rigs.** Group A reuses T-0231's live rig (``test_t0231_empty_step.turn`` over T-0207's ``claim_turn``: one
``FakeLiveBrowser`` as executor and live browser, real JPEG frames, loop on 8 steps) and its assertions
(``assert_reasked``, ``assert_not_reasked``, ``assert_empty_end``). A refusal is ``FakeLiveBrowser.outcome`` with
``_live_browser_support.refused``; the evidence's own reason is the T-0204 ``MOVED_REASON`` text, typed (``FRESHNESS``).
A ``failed`` outcome is ``_live_browser_support.failed``; a ``timeout`` is ``FakeLiveBrowser.hang`` (the session cuts
it after ``TOOL_EXEC_TIMEOUT_MS`` on the virtual clock, T-0114 criterion 10). Group B reuses T-0204's guard fakes
(``test_t0204_preclick.GuardDriver``: a hand-drawn page whose crops are real JPEGs) and its geometry: a click at
(500, 500) is the point (640, 400), box (576, 336, 128, 128), grown by the 40 px margin to x 536-744, y 296-504.
Changed shares are counted by hand on the crop's 8x8 grid (256 blocks): a 4x8-block patch is 32/256 = 0.125
(>= 0.10, moved), a 4x6-block patch 24/256 = 0.094 (< 0.10, not moved).

**Oracles.** A3 is a subprocess run over ``git archive`` of BASE (origin-proved), T-0231 B5's pattern; the scenario
set includes the shape this task touches (a refused click followed by a step that is *not* empty). B's verdicts come
from the task text and the hand counts above, never from the executor.

**Readings** (reported, not guessed):

- A1 "timeout": the session's own ``timeout`` status for a browser action (``_settle_result``); the fake's
  ``OutcomeStatus`` has no ``timeout``, so the row hangs the executor.
- B1/B3: a click with a shift near the point and an unchanged crop executes. Whether it writes a guard note is not
  pinned (T-0204 B6 writes none for a plain "not moved"; the task does not say whether "shift but pixels still"
  is one). Pinned: no ``refused`` note, and any guard note there has ``shift`` ``True``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time) for A, real asyncio time for B (well under 2 s), thinking on,
tools ``live`` (1280x800), loop on (8 steps), n = 1, deterministic. BASE = 0516a4f8 (main with the task file).

Not pending: every row marked RED below fails until T-0234 lands; GUARD rows are green at BASE and must stay green.
"""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from local_astra.computer import executor as executor_mod
from local_astra.conductor.tool_executor import to_pixels
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0207_session_check as t207
from tests.unit import test_t0231_empty_step as t231
from tests.unit._live_browser_support import failed, refused
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_t0204_preclick import (
    CLICK,
    FRESHNESS,
    NEAR,
    TYPE,
    Boom,
    GuardDriver,
    W,
    act,
    assert_moved_refusal,
    block_page,
    build,
    checked,
    const,
    count,
    dispatched,
    names,
    own,
    settle_off,  # noqa: F401  (autouse: T-0204 C10, the visual settle off; it reads ``shifts_since`` too)
)

REPO = Path(__file__).resolve().parents[2]
#: BASE: main at 0516a4f8 (``T-0234: task file``), where T-0234 branched. A literal, never a moving ref.
BASE_SHA = "0516a4f8"

EMPTY = t231.EMPTY
CALL_STEP = t231.CALL_STEP
FINAL = t231.FINAL
S1, S2, S3 = t231.S1, t231.S2, t231.S3
ARMING = t231.ARMING
NEVER = [answer("never sent")]


def _refuse(cid: str, reason: str = FRESHNESS) -> Any:
    browser = t207.browser_of()
    browser.outcome(cid, lambda inv: refused(inv, reason))
    return browser


# =====================================================================================================
# A1: re-asked after a refused browser action
# =====================================================================================================


@pytest.mark.parametrize("action", list(ARMING))
async def test_empty_after_each_refused_browser_action_is_reasked(
    tmp_path: Path, action: str
) -> None:
    """A1 (RED): step 1 makes each of the four browser actions and the browser refuses it with the moved text (the
    15:55 evidence); step 2 comes back with zero commits: it is re-asked once (T-0231's notes, the ``empty_reask``
    phase, the same prompt bytes) and the re-ask's call executes. Red at BASE: step 2 is sent once and the turn ends
    silent (``empty_reask_armed`` reads ``executed`` only)."""
    rig = await t231.turn(
        tmp_path,
        EMPTY,
        CALL_STEP,
        FINAL,
        script=[answer(S1), tool(ARMING[action])],
        browser=_refuse("t1-call"),
    )
    assert t207.results(rig, "t1-call") == ["refused"], f"premise: the step-1 {action} was refused"
    assert t207.inv_of(rig, "t1-call").name == action
    t231.assert_reasked(rig, 2, "call")
    assert "t1-call-s2" in t207.invoked(rig), "T-0234 A2: the re-ask's call executes as usual"
    assert S3 in rig.spoken_sources(), "T-0234 A2: the loop goes on to step 3"


async def test_refused_then_still_empty_ends_with_lost_thread(tmp_path: Path) -> None:
    """A1/A2 (RED; the t4-t6 evidence): step 1's click is refused (moved), step 2 empty, its re-ask empty too: the
    turn ends as T-0231 B3 (one ``lost_thread`` clip, ``empty_step: true``, nothing "degenerate", no third
    request). Red at BASE: no re-ask, no clip."""
    rig = await t231.turn(tmp_path, EMPTY, EMPTY, NEVER, browser=_refuse("t1-call"))
    assert t207.results(rig, "t1-call") == ["refused"], "premise"
    t231.assert_reasked(rig, 2, "empty")
    assert t231.attempts(rig, tsl.step_id(3)) == [], "T-0231 B3: step 3 is never sent"
    t231.assert_empty_end(rig, "refused then still empty")
    assert "never sent" not in rig.spoken_sources()


async def test_refused_reasks_share_the_turn_cap(tmp_path: Path) -> None:
    """A2 (RED): the cap of 2 re-asks per turn counts refused-armed re-asks. Step 1's click refused, step 2 empty, its
    re-ask clicks (refused again); step 3 empty, its re-ask clicks (refused again); step 4 empty: sent once (the cap)
    and the turn ends with ``lost_thread``; step 5 never sent."""
    browser = t207.browser_of()
    for cid in ("t1-call", "t1-call-s2", "t1-call-s3"):
        browser.outcome(cid, lambda inv: refused(inv, FRESHNESS))
    rig = await t231.turn(
        tmp_path,
        EMPTY,
        CALL_STEP,
        EMPTY,
        [answer("Next one."), tool(t231.CLICK3)],
        EMPTY,
        NEVER,
        browser=browser,
    )
    t231.assert_reasked(rig, 2, "call")
    assert t207.results(rig, "t1-call-s2") == ["refused"], "premise: the re-ask's click refused"
    t231.assert_reasked(rig, 3, "call")
    assert t207.results(rig, "t1-call-s3") == ["refused"], "premise"
    assert [n["step"] for n in t231.asked(rig)] == [2, 3], t231.asked(rig)
    assert len(t231.attempts(rig, tsl.step_id(4))) == 1, (
        f"T-0234 A2: the third empty step is not re-asked; {t231.attempts(rig, tsl.step_id(4))}"
    )
    assert t231.attempts(rig, tsl.step_id(5)) == []
    t231.assert_empty_end(rig, "refused, after the cap")


async def test_refused_then_answer_step_voiced_without_check(tmp_path: Path) -> None:
    """A2 (RED; re-pinned after review R-1): with ``completion_check=True``, step 1's click refused, step 2 empty, its
    re-ask a no-call answer. T-0207's check arms only after an *executed* browser action (A2.1, unchanged by
    T-0234), so the re-ask's answer is voiced with no check, exactly as a non-empty step 2 answering after a refused
    click on BASE."""
    rig = await t231.turn(
        tmp_path,
        EMPTY,
        t207.CLAIM_STEP,
        on=True,
        checks=(t207.NO_CALL_CHECK,),
        browser=_refuse("t1-call"),
    )
    t231.assert_reasked(rig, 2, "answer")
    assert rig.llm.check_prompts == [], "T-0207 A2.1: no check after a refused-only turn"
    voiced = rig.spoken_sources()
    assert t207.CLAIM in voiced and t207.CHECK_SAYS not in voiced, voiced


# --------------------------------------------------------------------------------------------- A1: not armed


async def test_empty_after_failed_click_not_reasked(tmp_path: Path) -> None:
    """A1 (GUARD, too wide): step 1's click ``failed``; step 2 empty: not re-asked (sent once, no note, no
    ``empty_reask`` phase). An implementation arming on "not executed" fails here. Green at BASE."""
    browser = t207.browser_of()
    browser.outcome("t1-call", lambda inv: failed(inv, "the page crashed"))
    rig = await t231.turn(tmp_path, EMPTY, NEVER, browser=browser)
    assert t207.results(rig, "t1-call") == ["failed"], "premise"
    t231.assert_not_reasked(rig, 2, "after a failed click")


async def test_empty_after_timed_out_click_not_reasked(tmp_path: Path) -> None:
    """A1 (GUARD, too wide): step 1's click hangs and the session cuts it (``timeout``, T-0114 criterion 10); step 2
    empty: not re-asked. Green at BASE."""
    browser = t207.browser_of()
    browser.hang("t1-call")
    rig = await t231.turn(tmp_path, EMPTY, NEVER, browser=browser)
    assert t207.results(rig, "t1-call") == ["timeout"], f"premise: {t207.results(rig, 't1-call')}"
    t231.assert_not_reasked(rig, 2, "after a timed-out click")


async def test_empty_after_refused_wait_not_reasked(tmp_path: Path, monkeypatch: Any) -> None:
    """A1 (GUARD, too wide): ``wait`` is not a browser action (R-B3). Step 1 clicks (executed), step 2's ``wait`` is
    refused, step 3 empty: not re-asked. An implementation arming on any refusal fails here. Green at BASE."""
    from tests.unit.test_t0202_session_wait import wait_kit

    browser = t207.browser_of()
    browser.outcome("t1-call-s2", lambda inv: refused(inv, "too many waits"))
    rig = await t231.turn(
        tmp_path,
        [answer(S2), tool(t231.WAIT)],
        EMPTY,
        NEVER,
        browser=browser,
        kit_of=lambda b: wait_kit(b, tmp_path / "kit", monkeypatch),
    )
    assert t207.results(rig, "t1-call") == ["executed"], "premise"
    assert t207.results(rig, "t1-call-s2") == ["refused"], "premise: the wait was refused"
    t231.assert_not_reasked(rig, 3, "after a refused wait")


async def test_refused_click_then_search_then_empty_not_reasked(tmp_path: Path) -> None:
    """A1 (GUARD, too wide): the arming reads the *previous* step only. Step 1's click is refused, step 2 searches
    (executed), step 3 empty: not re-asked (a "refused at any point this turn" flag fails here). Green at BASE."""
    rig = await t231.turn(
        tmp_path,
        [answer(S2), tool(t231.SEARCH_CALL)],
        EMPTY,
        NEVER,
        browser=_refuse("t1-call"),
        kit_of=t207._search_kit,
    )
    assert t207.results(rig, "t1-call") == ["refused"], "premise"
    assert t207.results(rig, "t1-call-s2") == ["executed"], "premise: the search executed"
    t231.assert_not_reasked(rig, 3, "after a search that followed a refused click")


# =====================================================================================================
# A3: a turn with no empty step is BASE's, byte for byte
# =====================================================================================================


async def _scenario(tmp: Path) -> dict[str, Any]:
    """A3's scenarios (no empty step), shaped on what A touches: a refused click then a click then an answer; a
    refused click then an answer with the completion check on; a failed click then an answer; an executed click,
    a click, an answer."""
    out: dict[str, Any] = {}

    async def one(name: str, **k: Any) -> None:
        rig = await t207.claim_turn(tmp / name, **k)
        out[name] = t207._dump(rig, tmp / name)

    await one(
        "refused_click_answer",
        on=t207.ABSENT,
        steps=(CALL_STEP, FINAL),
        browser=_refuse("t1-call"),
    )
    await one(
        "refused_claim_checked",
        on=True,
        steps=(t207.CLAIM_STEP,),
        checks=(t207.NO_CALL_CHECK,),
        browser=_refuse("t1-call"),
    )
    fb = t207.browser_of()
    fb.outcome("t1-call", lambda inv: failed(inv, "the page crashed"))
    await one("failed_click_answer", on=t207.ABSENT, steps=(FINAL,), browser=fb)
    await one("click_click_answer", on=t207.ABSENT, steps=(CALL_STEP, FINAL))
    return out


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0234_refused_reask_and_preclick as m
out = pathlib.Path(sys.argv[1])
data = asyncio.run(m._scenario(out.parent / "run"))
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


async def test_no_empty_step_is_byte_identical_to_base(tmp_path: Path) -> None:
    """A3 (GUARD): four turns with no empty step, three of them after a refused or failed click -- wire bodies,
    ``llm.request``/``llm.done``/``note``/``tool.call`` records, HUD messages, machine events with their command lists
    and turn-line extras -- equal BASE's (0516a4f8, subprocess over ``git archive``, origin-proved). Green at BASE."""
    base = await asyncio.to_thread(_base_capture, tmp_path)
    now = await _scenario(tmp_path / "now")
    assert set(now) == set(base)
    for name in sorted(base):
        b, n = base[name], now[name]
        for part in sorted(b):
            assert n[part] == b[part], (
                f"T-0234 A3 ({name}): {part} differs from BASE ({len(b[part])} vs {len(n[part])} items)"
            )
    blob = json.dumps(base["refused_click_answer"]["machine"])
    assert '"refused"' in blob, "premise: the BASE turn carried a refused result"
    assert t231.EMPTY_KEY not in json.dumps([s["extras"] for s in base.values()])


# =====================================================================================================
# B: pixels decide when the crop was taken
# =====================================================================================================


NOTE_WHAT = "live click: page moved check"
NOTE_KEYS = {"what", "result", "shift", "changed_share", "ms"}
DESPITE = "executed_despite_shift"  # R-2


def guard_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """T-0204 B6's notes with R-2's ``executed_despite_shift`` result admitted; each checked for the exact shape."""
    out = [n for n in notes if n.get("what") == NOTE_WHAT]
    for n in out:
        assert set(n) == NOTE_KEYS, f"T-0204 B6: the note has exactly {sorted(NOTE_KEYS)}: {n}"
        assert n["result"] in ("refused", "allowed_after_cap", "unavailable", DESPITE), n
        assert n["shift"] is None or type(n["shift"]) is bool, n
        assert n["changed_share"] is None or type(n["changed_share"]) is float, n
        assert type(n["ms"]) is int and n["ms"] >= 0, n
    return out


def _no_refused_note(notes: list[dict[str, Any]], share: float | None = None) -> None:
    """R-2: a click that executes despite a shift near the point writes exactly one ``executed_despite_shift`` note
    with ``shift`` ``True`` and the measured ``changed_share`` (below the threshold; ``share`` when given)."""
    got = guard_notes(notes)
    assert [n["result"] for n in got] == [DESPITE], (
        f"T-0234 R-2: one {DESPITE!r} note for a click executed despite a nearby shift: {got}"
    )
    (n,) = got
    assert n["shift"] is True, f"T-0234 B3/R-2: the note records the shift: {n}"
    assert isinstance(n["changed_share"], float) and n["changed_share"] < 0.10, n
    if share is not None:
        assert n["changed_share"] == pytest.approx(share, abs=0.004), n


@pytest.mark.parametrize("rects", NEAR)
@pytest.mark.parametrize(("tool_name", "args"), [("click_at", CLICK), ("type_at", TYPE)])
async def test_shift_near_point_unchanged_crop_executes(
    rects: Any, tool_name: str, args: dict[str, Any]
) -> None:
    """B1 (RED; the 15:55 evidence, ``changed_share`` 0.0): a shift rect intersecting the grown box (T-0204's NEAR
    set: at the point, at each margin, the whole viewport, a second rect, a source that slid out) on a static page:
    the crop was taken and nothing in it changed, so the call is checked (one region call, one shift read) and
    executes, dispatching as usual. Red at BASE: refused with the moved reason."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(rects)
    since = len(d.calls)
    out = await act(ex, tool_name, args, f)
    assert out.status == "executed", f"T-0234 B1: pixels decide; the crop is unchanged: {out}"
    checked(d, since)
    assert dispatched(d, since), "T-0234 B1: the call was dispatched"
    _no_refused_note(notes)


async def test_shifting_banner_page_never_reaches_the_cap() -> None:
    """B1 (RED; Amazon/Nintendo): a page that reports a shift near the point on every read but whose pixels at the
    point never change: three clicks in one turn each are checked and execute; no ``refused`` and no
    ``allowed_after_cap`` note (the cap is never reached)."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    d.shift_default = ((630.0, 390.0, 20.0, 20.0),)
    for _ in range(3):
        f = ex.latest_frame or await own(ex)
        since = len(d.calls)
        out = await act(ex, "click_at", CLICK, f)
        assert out.status == "executed", out
        checked(d, since)
    results = [n["result"] for n in guard_notes(notes)]
    assert results == [DESPITE] * 3, f"T-0234 R-2: three executed-despite-shift notes: {notes}"


async def test_shift_near_point_changed_crop_refused() -> None:
    """B1/B4 (GUARD): a shift near the point and a dark patch of 4x8 blocks on the crop's grid (32/256 = 0.125 >=
    0.10): refused with the moved reason, nothing dispatched, ``latest_frame`` unchanged; one ``refused`` note with
    ``shift`` ``True`` and ``changed_share`` 0.125 (B3)."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    ImageDraw.Draw(d.page).rectangle((576, 336, 607, 399), fill=(60, 60, 60))
    d.shift_script.append(((630.0, 390.0, 20.0, 20.0),))
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert_moved_refusal(out)
    checked(d, since)
    assert dispatched(d, since) == [] and ex.latest_frame is f
    (n,) = guard_notes(notes)
    assert (n["result"], n["shift"]) == ("refused", True), n
    assert n["changed_share"] == pytest.approx(32 / 256, abs=0.02), n


async def test_shift_near_point_crop_change_below_threshold_executes() -> None:
    """B1 (RED; adversarial, "any change plus a shift" is too wide): a shift near the point and a dark patch of 4x6
    blocks (24/256 = 0.094 < 0.10) in the crop's top-left corner (crop cols 0-3, rows 0-5: none of the point's 3x3
    blocks, cols/rows 7-9, R-2): executed, one ``executed_despite_shift`` note. Red at BASE (the shift alone
    refused)."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    ImageDraw.Draw(d.page).rectangle((576, 336, 607, 383), fill=(60, 60, 60))
    d.shift_script.append(((630.0, 390.0, 20.0, 20.0),))
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    checked(d, since)
    assert out.status == "executed", f"T-0234 B1: 24 of 256 blocks is under the share: {out}"
    _no_refused_note(notes, 24 / 256)


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(Boom("synthetic region failure"), id="raises"),
        pytest.param(ValueError("nothing inside"), id="valueerror"),
        pytest.param("hang", id="hangs"),
    ],
)
async def test_failed_crop_shift_near_point_refused(
    fault: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2 (GUARD; BASE's rule): the crop could not be taken (raises, or outlives ``PRECLICK_TIMEOUT_S``, patched to
    0.1 s) and a shift rect is near the point: refused with the moved reason, nothing dispatched; one note
    ``refused`` with ``shift`` ``True`` and ``changed_share`` ``None``."""
    monkeypatch.setattr(executor_mod, "PRECLICK_TIMEOUT_S", 0.1)
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(((630.0, 390.0, 20.0, 20.0),))
    d.region_script.append(fault)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert_moved_refusal(out)
    assert dispatched(d, since) == [] and count(d, "screenshot", since) == 0, names(d, since)
    assert [(n["result"], n["shift"], n["changed_share"]) for n in guard_notes(notes)] == [
        ("refused", True, None)
    ], notes


async def test_failed_crop_shift_far_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """B2 (GUARD, too wide): the crop failed and the only shift is outside the grown box (x 760): executed, noted
    ``unavailable`` (fail open, T-0204 B5)."""
    monkeypatch.setattr(executor_mod, "PRECLICK_TIMEOUT_S", 0.1)
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(((760.0, 390.0, 10.0, 10.0),))
    d.region_script.append(Boom("no crop"))
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    assert [n["result"] for n in guard_notes(notes)] == ["unavailable"], notes


@pytest.mark.parametrize(
    "shifts", [pytest.param((), id="no-shift"), pytest.param(None, id="shift-unknown")]
)
async def test_no_shift_changed_crop_refused(shifts: Any) -> None:
    """B4 (GUARD): no shift entry (or the recorder unavailable) and the grey block under the point turned dark: every
    block differs (share 1.0): refused as before, the note says ``shift`` ``False`` (``None``) and ``changed_share``
    1.0."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.page = block_page(60)
    d.shift_script.append(shifts)
    out = await act(ex, "click_at", CLICK, f)
    assert_moved_refusal(out)
    (n,) = guard_notes(notes)
    assert n["result"] == "refused" and n["shift"] is (False if shifts == () else None), n
    assert n["changed_share"] == pytest.approx(1.0, abs=0.01), n


def test_constants_unchanged() -> None:
    """B3 (GUARD): thresholds, the cap and the reason are T-0204's."""
    assert {
        k: const(k)
        for k in (
            "PRECLICK_HALF_PX",
            "PRECLICK_SHIFT_MARGIN_PX",
            "PRECLICK_BLOCK_DELTA",
            "PRECLICK_CHANGED_SHARE",
            "PRECLICK_TIMEOUT_S",
            "MAX_MOVED_REFUSALS",
        )
    } == {
        "PRECLICK_HALF_PX": 64,
        "PRECLICK_SHIFT_MARGIN_PX": 40,
        "PRECLICK_BLOCK_DELTA": 16,
        "PRECLICK_CHANGED_SHARE": 0.10,
        "PRECLICK_TIMEOUT_S": 0.3,
        "MAX_MOVED_REFUSALS": 2,
    }
    assert const("MOVED_REASON") == FRESHNESS


# =====================================================================================================
# R-2: with a shift near the point, the point's 3x3 blocks decide too
# =====================================================================================================
#
# The reviewer's synthetic probe: a white page, the kept screenshot a JPEG at quality 80 (the reference), the crop a
# PNG (lossless). Crop box (576, 336)-(704, 464) for a click at (500, 500) -> (640, 400). The point's block is crop
# block (8, 8) (offset 64 // 8); its 3x3 neighbourhood is crop cols/rows 7-9 = page x 632-655, y 392-415. Every
# drawn square is aligned to the 8 px grid of both the page and the crop (576 and 336 are multiples of 8), so a
# block is either untouched white or changes by a full black/white swap (delta ~255 > 16). Counts by hand.

WHITE = (255, 255, 255)


class WhiteDriver(GuardDriver):
    """``GuardDriver`` over a white page: ``screenshot`` is JPEG q80, ``screenshot_region`` a PNG crop."""

    def __init__(self) -> None:
        super().__init__(Image.new("RGB", (W, 800), WHITE))

    async def screenshot(self) -> bytes:
        self._enter("screenshot")
        self.shots.append(self.active)
        buf = io.BytesIO()
        self.page.convert("RGB").save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    async def screenshot_region(self, x: float, y: float, w: float, h: float) -> bytes:
        self._enter("screenshot_region", x, y, w, h)
        self.region_log.append((0.0, (x, y, w, h)))
        item = self.region_script.popleft() if self.region_script else None
        if isinstance(item, BaseException):
            raise item
        vw, vh = self.viewport
        x0, y0 = max(0, round(x)), max(0, round(y))
        x1, y1 = min(vw, round(x + w)), min(vh, round(y + h))
        buf = io.BytesIO()
        self.page.crop((x0, y0, x1, y1)).save(buf, format="PNG")
        return buf.getvalue()


def icon(page: Image.Image, x: int, y: int, size: int = 16) -> None:
    ImageDraw.Draw(page).rectangle((x, y, x + size - 1, y + size - 1), fill=(0, 0, 0))


def unicon(page: Image.Image, x: int, y: int, size: int = 16) -> None:
    ImageDraw.Draw(page).rectangle((x, y, x + size - 1, y + size - 1), fill=WHITE)


NEAR_RECT = ((630.0, 390.0, 20.0, 20.0),)


async def _white_click(
    move: Any, shifts: Any, start: Any = None, units: tuple[int, int] = (500, 500)
) -> tuple[Any, Any, list[dict[str, Any]], int]:
    d = WhiteDriver()
    if start is not None:
        start(d.page)
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    move(d.page)
    d.shift_script.append(shifts)
    since = len(d.calls)
    out = await act(ex, "click_at", {"x": units[0], "y": units[1]}, f)
    return out, d, notes, since


def test_white_rig_geometry() -> None:
    """Premise (by hand): the click (500, 500) is the point (640, 400), so the crop is (576, 336)-(704, 464)."""
    assert (to_pixels(500, 1280), to_pixels(500, 800)) == (640, 400)


@pytest.mark.parametrize(
    ("dx", "changed"),
    [pytest.param(8, 4, id="moved-8px-4-blocks"), pytest.param(24, 8, id="moved-24px-8-blocks")],
)
async def test_r2_small_icon_moved_under_point_refused(dx: int, changed: int) -> None:
    """R-2 (a) (RED): a 16 px icon at page (632, 392) (crop cols/rows 7-8, inside the point's 3x3) moves right by
    ``dx`` with a shift rect near the point. 8 px: cols 7 and 9 of rows 7-8 change (col 8 stays black): 4 blocks,
    share 4/256 = 0.016. 24 px: cols 7, 8 (old) and 10, 11 (new) of rows 7-8: 8 blocks, 8/256 = 0.031. Both under
    10 %, but blocks of the point's 3x3 changed: refused with the moved reason, one ``refused`` note with ``shift``
    ``True`` and the share. Red at 623edb49 (whole-crop share only: executed)."""
    out, d, notes, since = await _white_click(
        lambda pg: (unicon(pg, 632, 392), icon(pg, 632 + dx, 392)),
        NEAR_RECT,
        start=lambda pg: icon(pg, 632, 392),
    )
    checked(d, since)
    assert_moved_refusal(out)
    assert dispatched(d, since) == []
    (n,) = guard_notes(notes)
    assert (n["result"], n["shift"]) == ("refused", True), n
    assert n["changed_share"] == pytest.approx(changed / 256, abs=0.004), n


async def test_r2_small_icon_moved_without_shift_executes() -> None:
    """R-2 (d) (GUARD, too wide): the same 8 px icon move (4 blocks, 0.016, inside the 3x3) with **no** shift entry:
    the 10 % share rule alone applies, so the click executes, with no guard note."""
    out, d, notes, since = await _white_click(
        lambda pg: (unicon(pg, 632, 392), icon(pg, 640, 392)),
        (),
        start=lambda pg: icon(pg, 632, 392),
    )
    checked(d, since)
    assert out.status == "executed", f"T-0234 R-2: no shift, 4/256 < 10 %: {out}"
    assert guard_notes(notes) == [], notes


async def test_r2_no_shift_share_rule_unchanged() -> None:
    """R-2 (d) (GUARD): no shift, 26 whole blocks blackened far from the point (crop rows 0-1, cols 0-12): 26/256 =
    0.102 >= 0.10: refused; the note says ``shift`` ``False``."""
    out, d, notes, since = await _white_click(
        lambda pg: ImageDraw.Draw(pg).rectangle(
            (576, 336, 576 + 13 * 8 - 1, 336 + 15), fill=(0, 0, 0)
        ),
        (),
    )
    checked(d, since)
    assert_moved_refusal(out)
    (n,) = guard_notes(notes)
    assert (n["result"], n["shift"]) == ("refused", False), n
    assert n["changed_share"] == pytest.approx(26 / 256, abs=0.004), n


#: the four desk refusals of 2026-11-02 15:55 had ``changed_share`` 0.0 with a shift rect near the point; their
#: rect shapes are not in the log, so four plausible ones: a banner over the point, a loaded image below it, the
#: whole viewport (A-R1), and a carousel slid out (zero current rect, non-zero previous)
DESK = [
    pytest.param(((0.0, 380.0, 1280.0, 60.0),), id="banner-over-point"),
    pytest.param(((560.0, 440.0, 300.0, 200.0),), id="image-below"),
    pytest.param(((0.0, 0.0, 1280.0, 800.0),), id="whole-viewport"),
    pytest.param(((600.0, 350.0, 400.0, 120.0), (0.0, 0.0, 0.0, 0.0)), id="carousel-slid-out"),
]


@pytest.mark.parametrize("rects", DESK)
async def test_r2_desk_cases_share_zero_execute(rects: Any) -> None:
    """R-2 (b) (RED for the note): a shift near the point and nothing in the crop changed (share 0.0): executed, one
    ``executed_despite_shift`` note with ``shift`` ``True`` and ``changed_share`` 0.0."""
    out, d, notes, since = await _white_click(
        lambda pg: None, rects, start=lambda pg: icon(pg, 632, 392)
    )
    checked(d, since)
    assert out.status == "executed", f"T-0234 R-2: the desk case executes: {out}"
    _no_refused_note(notes, 0.0)


async def test_r2_distant_blocks_with_shift_execute_and_note() -> None:
    """R-2 (c) (RED for the note): a shift near the point and an icon in the crop's top-left corner (page (576, 336),
    crop cols/rows 0-1) moved 8 px right: crop cols 0 and 2 of rows 0-1 change, 4 blocks = 0.016, none in the point's
    3x3 (cols/rows 7-9): executed, one ``executed_despite_shift`` note, ``shift`` ``True``, ``changed_share``
    4/256."""
    out, d, notes, since = await _white_click(
        lambda pg: (unicon(pg, 576, 336), icon(pg, 584, 336)),
        NEAR_RECT,
        start=lambda pg: icon(pg, 576, 336),
    )
    checked(d, since)
    assert out.status == "executed", f"T-0234 R-2: distant blocks only, 0.016 < 10 %: {out}"
    _no_refused_note(notes, 4 / 256)


def _edge_units() -> tuple[int, int]:
    """A click whose crop is clamped at the right edge to 80 px (10 block columns x 16 rows = 160 blocks), its left
    edge on the 8 px grid: the point p has p - 64 = 1200, so p = 1264; the x unit is found by brute force over
    ``to_pixels`` (geometry, not the guard)."""
    xs = [u for u in range(1000) if to_pixels(u, 1280) == 1264]
    assert xs, "premise: some x unit maps to pixel 1264"
    return xs[0], 500


@pytest.mark.parametrize(
    ("blocks", "verdict"),
    [
        pytest.param(16, "refused", id="16-of-160-exactly-0.10"),
        pytest.param(15, "executed", id="15-of-160"),
    ],
)
async def test_r2_clamped_crop_exact_threshold(blocks: int, verdict: str) -> None:
    """R-2 (e) boundary for ``>=``: the crop at the right viewport edge is clamped to (1200, 336)-(1280, 464), 80x128
    = 160 blocks. No shift. ``blocks`` whole 8x8 blocks blackened from (1200, 336), ten per block row, left to right
    (crop rows 0-1): 16/160 = 0.10 exactly is moved (refused); 15/160 = 0.094 is not (executed). Green at 623edb49
    (the share rule already uses ``>=``); it guards a ``>`` mutant."""

    def paint(pg: Image.Image) -> None:
        dr = ImageDraw.Draw(pg)
        for i in range(blocks):
            bx, by = 1200 + (i % 10) * 8, 336 + (i // 10) * 8
            dr.rectangle((bx, by, bx + 7, by + 7), fill=(0, 0, 0))

    out, d, notes, since = await _white_click(paint, (), units=_edge_units())
    checked(d, since)
    region = next(c[1] for c in d.calls[since:] if c[0] == "screenshot_region")
    x0, y0 = max(0.0, float(region[0])), max(0.0, float(region[1]))
    x1 = min(1280.0, float(region[0]) + float(region[2]))
    assert (x0, y0, x1 - x0) == (1200.0, 336.0, 80.0), f"premise: the clamped crop: {region}"
    if verdict == "refused":
        assert_moved_refusal(out)
        (n,) = guard_notes(notes)
        assert n["changed_share"] == pytest.approx(0.10, abs=1e-9), n
    else:
        assert out.status == "executed", out
        assert guard_notes(notes) == [], notes


# =====================================================================================================
# R-2 re-review: the 3x3 window itself (on the white rig, counts and blocks by hand)
# =====================================================================================================
#
# The point (640, 400) is crop block (col 8, row 8), page x 640-647, y 400-407. A side neighbour is col 9 (x 648-655),
# a corner neighbour col 7 / row 7 (x 632-639, y 392-399), two blocks away col 10 (x 656-663). One changed block of
# 256 is a share of 0.004.


def _block(pg: Image.Image, x: int, y: int) -> None:
    ImageDraw.Draw(pg).rectangle((x, y, x + 7, y + 7), fill=(0, 0, 0))


def _units_for(px: int, size: int) -> int:
    """The coordinate unit that maps to pixel ``px`` (brute force over ``to_pixels``: geometry, not the guard)."""
    us = [u for u in range(1000) if to_pixels(u, size) == px]
    assert us, f"premise: some unit maps to pixel {px} on {size}"
    return us[0]


@pytest.mark.parametrize(
    "at",
    [pytest.param((648, 400), id="side-col9"), pytest.param((632, 392), id="corner-col7-row7")],
)
async def test_r2_window_one_block_away_refused(at: tuple[int, int]) -> None:
    """R-2 window (GUARD vs a 1x1 window): a shift near the point and one block changed one step from the point's
    block (side or corner): share 1/256, inside the 3x3: refused."""
    out, d, notes, since = await _white_click(lambda pg: _block(pg, *at), NEAR_RECT)
    checked(d, since)
    assert_moved_refusal(out)
    (n,) = guard_notes(notes)
    assert (n["result"], n["shift"]) == ("refused", True), n
    assert n["changed_share"] == pytest.approx(1 / 256, abs=0.004), n


@pytest.mark.parametrize(
    "at",
    [pytest.param((656, 400), id="side-col10"), pytest.param((624, 384), id="corner-col6-row6")],
)
async def test_r2_window_two_blocks_away_executes(at: tuple[int, int]) -> None:
    """R-2 window (GUARD vs a 5x5 window): a shift near the point and one block changed two steps from the point's
    block: share 1/256, outside the 3x3: executed, one ``executed_despite_shift`` note."""
    out, d, notes, since = await _white_click(lambda pg: _block(pg, *at), NEAR_RECT)
    checked(d, since)
    assert out.status == "executed", f"T-0234 R-2: two blocks away is outside the 3x3: {out}"
    _no_refused_note(notes, 1 / 256)


@pytest.mark.parametrize(
    ("point", "block", "rect"),
    [
        pytest.param((640, 4), (640, 0), ((630.0, 0.0, 20.0, 20.0),), id="top-edge-y4"),
        pytest.param((4, 400), (0, 400), ((0.0, 390.0, 20.0, 20.0),), id="left-edge-x4"),
    ],
)
async def test_r2_window_point_in_first_block_row_or_col_refused(
    point: tuple[int, int], block: tuple[int, int], rect: Any
) -> None:
    """R-2 window at the top/left edge (GUARD vs an unclipped ``cy - 1`` / ``cx - 1``): the point is at y 4 (x 4), so
    the crop is clamped to start at 0 and the point's block is row 0 (col 0). A shift near the point and only the
    point's own block changed (1 block, under 10 %): refused."""
    units = (_units_for(point[0], 1280), _units_for(point[1], 800))
    out, d, notes, since = await _white_click(lambda pg: _block(pg, *block), rect, units=units)
    checked(d, since)
    assert_moved_refusal(out)
    (n,) = guard_notes(notes)
    assert (n["result"], n["shift"]) == ("refused", True), n
    assert n["changed_share"] < 0.10, n


@pytest.mark.parametrize(
    "at",
    [pytest.param((1268, 788), id="block-7-7"), pytest.param((1260, 780), id="block-6-6-clamp")],
)
async def test_r2_window_point_in_last_partial_block_refused(at: tuple[int, int]) -> None:
    """R-2 window at the bottom-right edge: the point (1276, 796) gives a crop clamped to (1212, 732)-(1280, 800),
    68x68 px: 8x8 whole blocks plus a 4 px partial strip (not counted, T-0204 B2 (b)); the point lies in the partial
    strip (offset 64 -> block 8, past the last whole block 7). A shift near the point and one block changed: the last
    whole block next to it (crop block (7, 7), page x 1268-1275, y 788-795), or crop block (6, 6) (page x 1260-1267,
    y 780-787): 1/64 under 10 %, refused. Reading (the code's "clipped to it"): the point's block is clamped to the
    last whole block (7, 7), so its 3x3 is blocks 6-7; ``block-6-6-clamp`` kills a mutant without the clamp (whose
    window 7-9 misses block 6)."""
    units = (_units_for(1276, 1280), _units_for(796, 800))
    out, d, notes, since = await _white_click(
        lambda pg: _block(pg, *at), ((1260.0, 780.0, 20.0, 20.0),), units=units
    )
    checked(d, since)
    region = next(c[1] for c in d.calls[since:] if c[0] == "screenshot_region")
    assert tuple(round(float(v)) for v in region[:2]) == (1212, 732), f"premise: the crop {region}"
    assert_moved_refusal(out)
    (n,) = guard_notes(notes)
    assert (n["result"], n["shift"]) == ("refused", True) and n["changed_share"] < 0.10, n


async def test_r2_window_unreadable_shift_executes() -> None:
    """R-2 (GUARD vs "shift not False"): ``shifts_since`` answers ``None`` (the recorder cannot tell), the point's
    own block changed (1/256, under 10 %): the 3x3 rule needs a shift near the point, so the 10 % rule alone decides:
    executed, no guard note."""
    out, d, notes, since = await _white_click(lambda pg: _block(pg, 640, 400), None)
    checked(d, since)
    assert out.status == "executed", f"T-0234 R-2: no known shift, 1/256: {out}"
    assert guard_notes(notes) == [], notes
