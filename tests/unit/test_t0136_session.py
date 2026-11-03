"""T-0136 group B (session side): B-1..B-6 over the fakes (docs/tasks/T-0136.md).

**The surface these rows pin** (the task's names; where it is silent the reading is stated here and in the report):

- ``Session(..., action_outcome: bool = False)``; ``session.OUTCOME_CLAUSE == " Compared with the previous
  screenshot: {outcome}."``; the step label is ``"[" + LABEL_UPDATED + OUTCOME_CLAUSE.format(outcome=o.text) +
  extent_clause + "]"`` (B-3), typed here by hand from the criterion's example.
- The detector is called through the module attribute ``action_outcome.outcome_of_jpeg`` (the wiring rows replace it
  with a recording spy; ``session.outcome_of_jpeg`` is patched too, so a ``from ... import`` binding is covered).
  The spy's parameters are the A2-1 names (``before_jpeg, after_jpeg, point, pointers``), so keyword calls work.
- B-5: ``_TurnLog.action_outcomes: dict[int, action_outcome.Outcome]`` and ``_TurnLog.action_outcome_unmeasured:
  dict[int, str]``, keyed by the loop step of the action (the step that drafted the call: ``t1-call`` is step 1,
  ``t1-call-s2`` step 2). The rows read the turn's ``_TurnLog`` through ``_live_browser_support.turn_logs`` (a
  recording subclass installed on the module).
- B-4: ``llm.request``'s ``live_label`` gains ``outcome == {"kind", "region", "changed_cells", "cells"}`` exactly when
  the clause is placed; the turn line's ``TurnRecord.extras["action_outcome"]`` is ``{"none", "near", "elsewhere",
  "page", "unmeasured"}`` (all five keys, ints) on every live turn line; each unmeasured action writes one ``note``
  whose ``data`` holds the reason literal as a value.

Readings (stated, not guessed silently; see the report):

- **R1 (B-4 counts).** "one per executed pointer action": the counts sum to the turn's *executed* ``click_at`` /
  ``type_at`` actions. A ``not_pointer`` step (a scroll, a key, a refused click) is in ``action_outcome_unmeasured``
  and writes a note, but is not counted in the turn line.
- **R2 (B-2 ``pointers``).** "the session's previous executed pointer point": session-wide, so a turn's first click
  carries the previous turn's last executed point (the mouse stays where it was). One row pins it alone
  (``test_pointers_previous_point_across_turns``) so it can be ruled on separately. The order is ``(previous,
  current)`` and the first action of a session has ``(current,)``.
- **R3 (B-1 messages).** The ``ValueError``s name the keyword (``action_outcome``); the not-live one also names
  ``live``.
- **R4 (B-2 "never blocks the event loop for more than 20 ms").** Not a row here: the detector's cost is A2-6's eval
  (p95 <= 20 ms), and a wall-clock bound on a fake session would be a timing row under load. Stated in the report.

The rows are independent of the detector's calibrated constants: the measured rows that call the real detector use
``scene_frame`` JPEGs whose change is large and unambiguous (solid black on white, far from or at the click point; a
whole frame turned black for ``page``; the same white frame at JPEG quality 90 and 80 for ``none``), and the wiring
rows replace the detector with a spy.

Activated for group B (readings R1-R5 ruled O-B in the task file; the pending marker is off): at BASE_B (2366600)
every row is red, the rig failing on the missing keyword (a named ``AssertionError``) or a named assertion failing.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on, ReGround off unless a row says so, T-0134c's label on (extent off unless a row says so),
echo ``pixels`` (the fake's default); n = 1, deterministic.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import action_outcome
from local_astra.contracts import commands as C
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    BLACK,
    FakeLiveBrowser,
    feedback_rig,
    live_feedback,
    live_frame,
    refused,
    scene_frame,
    session_mod,
    turn_logs,
)
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    tool,
)
from tests.unit.test_session_live import CLICK2, SCROLL, _wire, pass1_body, step_body
from tests.unit.test_t0134_label import (
    L1,
    LN,
    LU,
    STEM,
    all_bodies,
    label_after,
    labels,
    next_turn,
    three,
)

Q = tsl.Q
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3

#: B-3, typed from the criterion (never imported from session.py)
CLAUSE_TEMPLATE = " Compared with the previous screenshot: {outcome}."
#: the clause's stem: a substring of no other rig text
CLAUSE_STEM = "Compared with the previous screenshot"
#: LABEL_UPDATED, typed (T-0134 A3; equal to LU without its brackets)
UPDATED = "This screenshot is the browser screen now, after your last action."
#: B-3's example, verbatim from the task file
EXAMPLE = (
    "[This screenshot is the browser screen now, after your last action. Compared with the previous screenshot: "
    "changed elsewhere (top right).]"
)
REASONS = ("not_pointer", "blind", "no_after_frame", "overview_frame", "scrolled", "decode")
COUNT_KEYS = ("none", "near", "elsewhere", "page", "unmeasured")
OUTCOME_KEYS = {"kind", "region", "changed_cells", "cells"}
TYPE_MUGS = f"<|tool_call>call:type_at{{press_enter:false,text:{Q}mugs{Q},x:500,y:250}}<tool_call|>"

# --------------------------------------------------------------------------------------------- scenes
# VALID_CLICK (500, 250) resolves on the fake to (640, 200); CLICK2 (120, 640) to (153, 512). By A2-3's definitional
# geometry at 1280x800 the near box of (640, 200) is x 320..960, y 120..280.

#: a white screen
W0 = scene_frame("t0136-white")
#: the same white screen re-encoded at quality 80: other bytes, the same pixels within JPEG noise (``none``)
W0_Q80 = scene_frame("t0136-white-q80", quality=80)
#: a black badge at the top right (x 1100..1220, y 20..80): outside the near box and every hover zone of (640, 200),
#: in the top-right ninth (x >= 853, y < 267)
BADGE = scene_frame("t0136-badge", blocks=[(1100, 20, 1220, 80, BLACK)])
#: a black 32x24 box centred on (640, 200): a strong change at the click point (``near``)
TICK = scene_frame("t0136-tick", blocks=[(624, 188, 656, 212, BLACK)])
#: the whole screen black (``page``: every cell changed)
DARK = scene_frame("t0136-dark", background=BLACK)
#: a black box at the bottom left (x 40..160, y 700..760): bottom-left ninth, far from (640, 200) and (153, 512)'s
#: hover zones (x 0..353, y 480..544) and near boxes (y up to 592)
CORNER = scene_frame("t0136-corner", blocks=[(40, 700, 160, 760, BLACK)])
#: the badge kept and the bottom-left box added: from BADGE, a change at the bottom left only
BADGE_CORNER = scene_frame(
    "t0136-badge-corner", blocks=[(1100, 20, 1220, 80, BLACK), (40, 700, 160, 760, BLACK)]
)

#: the non-JPEG fake frames (``decode``)
F0 = live_frame("onset")
F1 = live_frame("after-1")
F2 = live_frame("after-2")


def with_clause(outcome_text: str, extent: str = "") -> str:
    return "[" + UPDATED + CLAUSE_TEMPLATE.format(outcome=outcome_text) + extent + "]"


def label_on(page_extent: bool = False) -> Any:
    return live_feedback(screen_label=True, page_extent=page_extent)


def outcome_rig(
    tmp: Path,
    browser: Any,
    *,
    script: Any,
    feedback: Any = "on",
    on: bool = True,
    extra: dict[str, Any] | None = None,
    kit: Any = None,
) -> Any:
    """``feedback_rig`` with ``Session(..., live_feedback=<label on>, action_outcome=True)`` (``on=False``: the
    keyword is not passed at all). A session without the keyword is a named ``AssertionError``."""
    kw: dict[str, Any] = {}
    fb = label_on() if feedback == "on" else feedback
    if fb is not None:
        kw["live_feedback"] = fb
    if on:
        kw["action_outcome"] = True
    kw.update(extra or {})
    return feedback_rig(tmp, browser, script=script, session_kwargs=kw, kit=kit)


class DetectorSpy:
    """Stands in for ``action_outcome.outcome_of_jpeg``: records every call, returns ``result``."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[bytes, bytes, tuple[int, int], tuple[tuple[int, int], ...]]] = []

    def __call__(self, before_jpeg: bytes, after_jpeg: bytes, point: Any, pointers: Any) -> Any:
        self.calls.append(
            (
                bytes(before_jpeg),
                bytes(after_jpeg),
                (int(point[0]), int(point[1])),
                tuple((int(p[0]), int(p[1])) for p in pointers),
            )
        )
        return self.result


#: what the spy returns: a distinctive outcome no scene in this module produces
SPY_OUTCOME = action_outcome.Outcome(
    kind="elsewhere", region="bottom left", changed_cells=7, cells=4000
)
SPY_TEXT = "changed elsewhere (bottom left)"


def patch_detector(monkeypatch: pytest.MonkeyPatch, result: Any = SPY_OUTCOME) -> DetectorSpy:
    spy = DetectorSpy(result)
    monkeypatch.setattr(action_outcome, "outcome_of_jpeg", spy)
    monkeypatch.setattr(session_mod(), "outcome_of_jpeg", spy, raising=False)
    return spy


def turn_log(spy: Any, turn: str = "t1") -> Any:
    return spy.by_turn(turn)


def outcomes(log: Any) -> dict[int, Any]:
    got = getattr(log, "action_outcomes", None)
    assert isinstance(got, dict), (
        f"T-0136 B-5: _TurnLog.action_outcomes (dict[int, Outcome]) does not exist; got {got!r}"
    )
    return got


def unmeasured(log: Any) -> dict[int, str]:
    got = getattr(log, "action_outcome_unmeasured", None)
    assert isinstance(got, dict), (
        f"T-0136 B-5: _TurnLog.action_outcome_unmeasured (dict[int, str]) does not exist; got {got!r}"
    )
    return got


def _values(v: Any) -> list[Any]:
    if isinstance(v, dict):
        return [x for y in v.values() for x in _values(y)]
    if isinstance(v, list):
        return [x for y in v for x in _values(y)]
    return [v]


def reason_notes(rig: Any, reason: str) -> list[dict[str, Any]]:
    """``note`` records whose data holds ``reason`` as a value (B-4)."""
    return [d for d in rig.kind("note") if reason in _values(d)]


def counts(**kw: int) -> dict[str, int]:
    return {k: int(kw.get(k, 0)) for k in COUNT_KEYS}


def clause_free(rig: Any) -> None:
    sent = json.dumps(all_bodies(rig))
    assert CLAUSE_STEM not in sent, (
        f"T-0136 B-2/B-3: an unmeasured action placed a clause; {[labels(b) for b in all_bodies(rig)]}"
    )


def live_labels(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if "live_label" in d]


def executed(
    call_id: str, x: int, y: int, *, frame_after: str = "not-a-capture", **extra: Any
) -> Any:
    """A handler: ``call_id`` is executed at viewport ``(x, y)`` with ``frame_after`` (default: no capture of the
    fake's, so B8 takes a fresh frame)."""

    def make(inv: ToolInvocation) -> ToolOutcome:
        return ToolOutcome(
            call_id=inv.call_id,
            status="executed",
            action=f"clicked at ({x}, {y})",
            reason=None,
            resolved={"x_px": x, "y_px": y, "frame_after": frame_after, **extra},
        )

    return make


# =====================================================================================================
# B-3: the clause
# =====================================================================================================


def test_outcome_clause_constant() -> None:
    """B-3 / H-T136-7: ``session.OUTCOME_CLAUSE`` is the criterion's template exactly."""
    got = getattr(session_mod(), "OUTCOME_CLAUSE", None)
    assert got == CLAUSE_TEMPLATE, f"T-0136 B-3: OUTCOME_CLAUSE; got {got!r}"


async def test_clause_text_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B-3 (the criterion's example, real detector): a click at (640, 200) on a white screen after which a black badge
    shows at the top right gives step 2 exactly ``EXAMPLE``, right after the post-action frame, once; step 1 keeps
    ``L1``. ``_TurnLog.action_outcomes[1]`` is that ``elsewhere``/``top right`` outcome."""
    spy = turn_logs(monkeypatch)
    browser = FakeLiveBrowser([W0], post_frames=[BADGE])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S3)],))
        assert label_after(pass1_body(rig), W0) == L1, "B-3: step 1 is unchanged"
        assert label_after(step_body(rig, 2), BADGE) == EXAMPLE
    got = outcomes(turn_log(spy))
    assert set(got) == {1}, got
    o = got[1]
    assert isinstance(o, action_outcome.Outcome) and (o.kind, o.region) == (
        "elsewhere",
        "top right",
    ), o
    assert o.text == "changed elsewhere (top right)"


@pytest.mark.parametrize(
    ("after", "text"),
    [
        (TICK, "changed near your click"),
        (DARK, "page changed"),
        (CORNER, "changed elsewhere (bottom left)"),
        (W0_Q80, "no visible change"),
    ],
    ids=["near", "page", "elsewhere", "none"],
)
async def test_clause_each_kind(tmp_path: Path, after: Any, text: str) -> None:
    """B-3 / A2-2 (real detector, unambiguous scenes): the step-2 label after a click on the white screen is
    ``[LABEL_UPDATED + clause(text)]`` for a box at the click point (near), the whole screen turning black (page), a
    box at the bottom left (elsewhere) and the same screen re-encoded at quality 80 (none: other bytes, so the screen
    label opens with ``LABEL_UPDATED``; the clause says ``no visible change``)."""
    browser = FakeLiveBrowser([W0], post_frames=[after])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S3)],))
        assert label_after(step_body(rig, 2), after) == with_clause(text)


async def test_clause_with_extent(tmp_path: Path) -> None:
    """B-3: with the page extent on the clause sits between the opening and the extent clause: ``[LABEL_UPDATED +
    outcome clause + " Page beyond this screen: about 2.0 screens below, nothing above." + "]"`` (1600/800 below, 0
    above, typed by hand from T-0134 A4)."""
    browser = FakeLiveBrowser([W0], post_frames=[TICK])
    ext = {"above_px": 0.0, "below_px": 1600.0, "viewport_px": 800.0}
    browser.extents = {f.phash: dict(ext) for f in (W0, TICK)}
    async with outcome_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], feedback=label_on(True)
    ) as rig:
        await three(rig, ([answer(S3)],))
        want = with_clause(
            "changed near your click",
            " Page beyond this screen: about 2.0 screens below, nothing above.",
        )
        assert label_after(step_body(rig, 2), TICK) == want


async def test_type_at_measured(tmp_path: Path) -> None:
    """B-2: an executed ``type_at`` is measured like a click (its point is ``resolved``'s (640, 200)): a box at the
    point gives ``changed near your click``."""
    browser = FakeLiveBrowser([W0], post_frames=[TICK])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(TYPE_MUGS)]) as rig:
        await three(rig, ([answer(S3)],))
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"], "premise"
        assert label_after(step_body(rig, 2), TICK) == with_clause("changed near your click")


async def test_fresh_after_frame_measured(tmp_path: Path) -> None:
    """B-2 ("B8 obtained a fresh after frame"): when ``latest_frame`` is not the action's ``frame_after`` B8 takes a
    fresh frame; the action is measured against that frame (the badge): ``EXAMPLE``."""
    browser = FakeLiveBrowser([W0, BADGE])
    browser.outcome("t1-call", executed("t1-call", 640, 200))
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S3)],))
        assert label_after(step_body(rig, 2), BADGE) == EXAMPLE


async def test_no_clause_after_unchanged_or_step1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-3 (adversarial; reviewer mutation 6, the clause after ``LABEL_UNCHANGED``): a measured click whose
    post-action capture is the very frame the model saw (same bytes and phash) leaves the payload ``unchanged``; step
    2's label is exactly ``LN`` although the action was measured (``none``, without decoding: the spy is never
    called). Step 1 is exactly ``L1``. A refused click at step 2 then gives step 3 exactly ``LN`` too."""
    det = patch_detector(monkeypatch)
    spy = turn_logs(monkeypatch)
    browser = FakeLiveBrowser([W0], post_frames=[W0])
    browser.outcome("t1-call-s2", lambda i: refused(i, "bad coordinates"))
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert label_after(pass1_body(rig), W0) == L1
        tail2 = tsl.tail(tsl.step_request(rig, 2))
        assert f"{Q}unchanged{Q}" in tail2, "premise: the payload says unchanged"
        assert label_after(step_body(rig, 2), W0) == LN, "T-0136 B-3: LABEL_UNCHANGED is unchanged"
        assert label_after(step_body(rig, 3), W0) == LN
        clause_free(rig)
    log = turn_log(spy)
    got = outcomes(log)
    assert set(got) == {1} and got[1].kind == "none", (
        f"premise: the step-1 action was measured (identical identities: none); {got}"
    )
    assert det.calls == [], "B-2: identical identities give none without decoding"
    assert unmeasured(log) == {2: "not_pointer"}, unmeasured(log)


async def test_identical_identity_none_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-2 (adversarial: "identical identities give none without decoding"; the mutant decodes first): the
    post-action capture has the onset frame's bytes (not a JPEG: decoding would fail) under another executor phash.
    The session keys frames by capture identity (sha256 of the bytes, T-0111g B7), so the identities are equal: the
    action is ``none`` (``NO_VISIBLE_CHANGE``), the detector is never called, no ``decode`` note is written, and the
    payload says ``unchanged`` so step 2 is exactly ``LN``."""
    det = patch_detector(monkeypatch)
    spy = turn_logs(monkeypatch)
    same = live_frame("same-bytes").model_copy(update={"jpeg": F0.jpeg})
    assert same.phash != F0.phash, "premise"
    browser = FakeLiveBrowser([F0], post_frames=[same])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S3)],))
        assert label_after(step_body(rig, 2), same) == LN
    assert rig.kind("llm.request"), "premise: the session log was read (after the rig closed)"
    assert reason_notes(rig, "decode") == [], rig.kind("note")
    assert det.calls == []
    got = outcomes(turn_log(spy))
    assert set(got) == {1} and got[1].kind == "none", got
    assert got[1].text == action_outcome.NO_VISIBLE_CHANGE == "no visible change"
    # O-B2 F2: the shortcut still reports the frame's cell grid (1280x800 decoded at 1/DRAFT_SCALE, cut into CELL_PX
    # cells, ceil on both axes: 80 x 50 = 4000 at DRAFT_SCALE 2 / CELL_PX 8), and no changed cell
    s, c = action_outcome.DRAFT_SCALE, action_outcome.CELL_PX
    grid = -(-(800 // s) // c) * -(-(1280 // s) // c)
    assert (got[1].changed_cells, got[1].cells) == (0, grid), (got[1], grid)


# =====================================================================================================
# B-2: unmeasured actions
# =====================================================================================================


async def _unmeasured_case(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, reason: str, *, patched: bool = True
) -> tuple[Any, Any, Any]:
    """One turn whose step-1 action is unmeasured with ``reason``; returns (rig, the turn's _TurnLog, the spy)."""
    det = patch_detector(monkeypatch) if patched else None
    spy = turn_logs(monkeypatch)
    extra: dict[str, Any] = {}
    call = VALID_CLICK
    if reason == "scrolled":
        browser = FakeLiveBrowser([W0], post_frames=[BADGE])
        browser.resolved_extra["t1-call"] = {"scrolled_px": 120}
    elif reason == "no_after_frame":
        browser = FakeLiveBrowser([W0])

        def gone(fb: FakeLiveBrowser, inv: ToolInvocation) -> ToolOutcome:
            fb.frame_mode = "raise"  # B8's fresh frame fails
            return executed("t1-call", 640, 200)(inv)

        browser.handlers["t1-call"] = gone
    elif reason == "decode":
        browser = FakeLiveBrowser([F0], post_frames=[F1])
    elif reason == "blind":
        browser = FakeLiveBrowser([], post_frames=[BADGE])
        browser.frame_mode = "raise"
        browser.outcome("t1-call", executed("t1-call", 640, 200))
    elif reason == "not_pointer":
        browser = FakeLiveBrowser([W0], post_frames=[BADGE])
        call = SCROLL
    else:
        raise AssertionError(reason)
    async with outcome_rig(tmp, browser, script=[answer(S1), tool(call)], extra=extra) as rig:
        await three(rig, ([answer(S3)],))
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"], (
            f"premise ({reason}): the step-1 action was executed"
        )
    return rig, turn_log(spy), det


@pytest.mark.parametrize("reason", ["scrolled", "no_after_frame", "decode", "blind", "not_pointer"])
async def test_unmeasured_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """B-2/B-4/B-5: each unmeasured reason (the overview frame has its own row): no clause in any body; the detector
    is never called (``decode`` runs the real one on the fake's non-JPEG bytes and must not fail the turn); one
    ``note`` holding the reason; ``action_outcome_unmeasured == {1: reason}`` and ``action_outcomes == {}``; the
    turn line counts ``unmeasured`` 1, except ``not_pointer`` (R1: not a pointer action, not counted)."""
    rig, log, det = await _unmeasured_case(
        tmp_path, monkeypatch, reason, patched=reason != "decode"
    )
    clause_free(rig)
    if det is not None:
        assert det.calls == [], f"T-0136 B-2 ({reason}): the action was measured; {det.calls}"
    assert reason_notes(rig, reason), (
        f"T-0136 B-4: no note with the reason {reason!r}; notes {rig.kind('note')}"
    )
    assert unmeasured(log) == {1: reason}, unmeasured(log)
    assert outcomes(log) == {}, outcomes(log)
    want = counts() if reason == "not_pointer" else counts(unmeasured=1)
    assert rig.record("t1").extras.get("action_outcome") == want, rig.record("t1").extras


@pytest.mark.parametrize("reason", ["no_after_frame", "decode"])
async def test_unmeasured_at_step2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """O-B2 F1 (adversarial; the mutant drops the ``fresh`` gate in ``_live_step_screen``, which survives at step 1
    because no live frame is held yet): step 1 (white -> badge) is measured; the step-2 click on the badge frame is
    unmeasured with ``reason``. ``no_after_frame``: B8's fresh frame fails at step 2 (the step-2 handler sets
    ``frame_mode = "raise"`` and returns a ``frame_after`` that is no capture), so ``live_frame`` still holds step 2's
    frame and must not be taken as the after frame. ``decode``: the step-2 post-action capture is not a JPEG.
    ``action_outcome_unmeasured == {2: reason}``, key 2 is absent from ``action_outcomes``, the turn line counts
    ``elsewhere`` 1 and ``unmeasured`` 1, one note holds the reason, and step 3's body carries no clause."""
    spy = turn_logs(monkeypatch)
    if reason == "no_after_frame":
        browser = FakeLiveBrowser([W0], post_frames=[BADGE])

        def gone(fb: FakeLiveBrowser, inv: ToolInvocation) -> ToolOutcome:
            fb.frame_mode = "raise"
            return executed("t1-call-s2", 153, 512)(inv)

        browser.handlers["t1-call-s2"] = gone
    else:
        browser = FakeLiveBrowser([W0], post_frames=[BADGE, F1])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert [e.status for e in tsl.results_for(rig, "t1-call-s2")] == ["executed"], "premise"
        assert label_after(step_body(rig, 2), BADGE) == EXAMPLE, "premise: step 1 was measured"
        s3 = step_body(rig, 3)
        assert CLAUSE_STEM not in json.dumps(s3), (
            f"T-0136 B-2: the unmeasured step-2 action placed a clause; {labels(s3)}"
        )
    log = turn_log(spy)
    assert unmeasured(log) == {2: reason}, unmeasured(log)
    got = outcomes(log)
    assert 2 not in got and set(got) == {1}, {k: v.kind for k, v in got.items()}
    assert rig.record("t1").extras.get("action_outcome") == counts(elsewhere=1, unmeasured=1)
    assert len(reason_notes(rig, reason)) == 1, rig.kind("note")


async def test_scrolled_zero_is_measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B-2 (the ``scrolled`` guard is not too wide): ``resolved["scrolled_px"] == 0`` is measured; the spy sees the
    call and step 2 carries its text."""
    det = patch_detector(monkeypatch)
    browser = FakeLiveBrowser([W0], post_frames=[BADGE])
    browser.resolved_extra["t1-call"] = {"scrolled_px": 0}
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S3)],))
        assert label_after(step_body(rig, 2), BADGE) == with_clause(SPY_TEXT)
    assert len(det.calls) == 1, det.calls


async def test_decode_either_frame(tmp_path: Path) -> None:
    """B-2 ``decode``: a decodable before frame and a non-JPEG after frame is unmeasured too (real detector); the
    turn ends normally and step 2's label is exactly ``LU``."""
    spy_rig_browser = FakeLiveBrowser([W0], post_frames=[F1])
    async with outcome_rig(
        tmp_path, spy_rig_browser, script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        await three(rig, ([answer(S3)],))
        assert label_after(step_body(rig, 2), F1) == LU
    assert reason_notes(rig, "decode"), rig.kind("note")


async def test_refused_click_not_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B-2 ``not_pointer`` ("not an executed click_at/type_at"): a refused click at step 2 after a measured step 1
    is ``not_pointer`` at key 2, and is not counted in the turn line (R1)."""
    det = patch_detector(monkeypatch)
    spy = turn_logs(monkeypatch)
    browser = FakeLiveBrowser([W0], post_frames=[BADGE])
    browser.outcome("t1-call-s2", lambda i: refused(i, "bad coordinates"))
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert label_after(step_body(rig, 3), BADGE) == LN
    log = turn_log(spy)
    assert set(outcomes(log)) == {1} and unmeasured(log) == {2: "not_pointer"}
    assert len(det.calls) == 1
    assert rig.record("t1").extras.get("action_outcome") == counts(elsewhere=1)


# --------------------------------------------------------------------------------------------- overview frame

#: deals.html at 1280x800 (T-0151): 1220/800 CSS px
DEALS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1220.0}
#: a decodable overview picture (838x798), so a mutant that measures it does not stop at ``decode``
OV = scene_frame(
    "t0136-overview",
    width=838,
    height=798,
    blocks=[(700, 10, 830, 60, BLACK)],
    phash="3136313631363136",
)
#: a step-1 click on the overview at (500, 300): by hand, P = (to_px(500, 1280), to_px(300, 1220)) = (640, 366),
#: inside the first screen, so the browser does not scroll (``scrolled_px == 0``): the reason is the overview alone
OV_CLICK = "<|tool_call>call:click_at{x:500,y:300}<tool_call|>"


async def test_overview_frame_unmeasured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B-2 / H-T136-3 (adversarial; reviewer mutation 7, overview-frame steps measured): with the overview frame shown
    at step 1 (decodable pictures throughout), the step-1 click (no scroll) is unmeasured ``overview_frame``: step 2
    keeps T-0151 A2-6's forced ``updated`` with exactly ``LU`` and no clause, and the detector is not called for it.
    The step-2 click on the live frame is measured (the guard is not too wide): step 3 carries the spy's text, and the
    spy's only call is that one, before = the badge frame step 2 showed."""
    det = patch_detector(monkeypatch)
    spy = turn_logs(monkeypatch)
    browser = FakeLiveBrowser([W0], post_frames=[BADGE, CORNER])
    browser.overview_image = OV
    browser.page_metrics = dict(DEALS)
    async with outcome_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(OV_CLICK)],
        extra={"overview_frame": True},
    ) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        ov_click = [i for i in browser.invocations if i.call_id == "t1-call"]
        assert ov_click and ov_click[0].frame is not None and ov_click[0].frame.phash == OV.phash, (
            "premise: the step-1 click was on the overview frame"
        )
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"], "premise"
        assert label_after(step_body(rig, 2), BADGE) == LU, (
            "T-0136 H-T136-3: step 2 after an overview-frame action keeps LU, no clause"
        )
        assert label_after(step_body(rig, 3), CORNER) == with_clause(SPY_TEXT)
    assert reason_notes(rig, "overview_frame"), rig.kind("note")
    assert [c[:2] for c in det.calls] == [(BADGE.jpeg, CORNER.jpeg)], (
        f"T-0136 B-2: only the step-2 action is measured; {[(len(a), len(b)) for a, b, _, _ in det.calls]}"
    )
    log = turn_log(spy)
    assert unmeasured(log) == {1: "overview_frame"} and set(outcomes(log)) == {2}, (
        unmeasured(log),
        outcomes(log),
    )


# =====================================================================================================
# B-2: the inputs (wiring, spy)
# =====================================================================================================


async def test_pointers_and_before_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B-2 (adversarial: the before frame re-derived from ``loop_base``; the point taken from the arguments): two
    clicks. Call 1: before F0 (step 1's image), after F1, point (640, 200), pointers ``((640, 200),)``. Call 2:
    before F1 (the image step 2 sent, not ``loop_base``'s F0), after F2, point ``resolved``'s scripted (901, 444)
    (the arguments map to (153, 512)), pointers ``((640, 200), (901, 444))``. Both labels carry the spy's text."""
    det = patch_detector(monkeypatch)
    browser = FakeLiveBrowser([F0], post_frames=[F1, F2])
    browser.resolved_extra["t1-call-s2"] = {"x_px": 901, "y_px": 444}
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        assert label_after(step_body(rig, 2), F1) == with_clause(SPY_TEXT)
        assert label_after(step_body(rig, 3), F2) == with_clause(SPY_TEXT)
    assert det.calls == [
        (F0.jpeg, F1.jpeg, (640, 200), ((640, 200),)),
        (F1.jpeg, F2.jpeg, (901, 444), ((640, 200), (901, 444))),
    ], [(c[2], c[3], c[0] == F0.jpeg, c[1] == F1.jpeg) for c in det.calls]


async def test_pointers_previous_point_across_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-2, reading R2 ("the session's previous executed pointer point"): turn 2's first click carries turn 1's last
    executed point: pointers ``((640, 200), (153, 512))``."""
    det = patch_detector(monkeypatch)
    G0 = live_frame("t2-onset")
    browser = FakeLiveBrowser([F0, G0], post_frames=[F1, F2])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S3)],))
        await next_turn(rig, 2, "and the next one", [answer(S1), tool(CLICK2)], ([answer(S3)],))
    assert [c[2:] for c in det.calls] == [
        ((640, 200), ((640, 200),)),
        ((153, 512), ((640, 200), (153, 512))),
    ], det.calls


# =====================================================================================================
# B-5: the _TurnLog contract; B-4: logging
# =====================================================================================================


async def test_turnlog_records_and_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B-4/B-5 (real detector): click (badge: elsewhere/top right), scroll (not_pointer), click with ``scrolled_px``
    (scrolled), report; then a toolless turn.

    - ``action_outcomes == {1: Outcome(elsewhere, top right, n, cells)}`` with ``0 < n < cells``;
      ``action_outcome_unmeasured == {2: "not_pointer", 3: "scrolled"}``;
    - only step 2's ``live_label`` record has ``outcome``, equal to ``{kind, region, changed_cells, cells}`` of that
      Outcome; step 1's, step 3's and step 4's have no ``outcome`` key, and their labels no clause;
    - turn 1's line: ``action_outcome == {elsewhere: 1, unmeasured: 1, others 0}`` (R1); turn 2's: all zeros;
    - one note each for ``not_pointer`` and ``scrolled``."""
    spy = turn_logs(monkeypatch)
    S2F = scene_frame(
        "t0136-after-scroll", blocks=[(1100, 20, 1220, 80, BLACK), (40, 300, 300, 340, BLACK)]
    )
    S3F = scene_frame(
        "t0136-after-3", blocks=[(1100, 20, 1220, 80, BLACK), (600, 600, 700, 640, BLACK)]
    )
    browser = FakeLiveBrowser([W0, live_frame("t2")], post_frames=[BADGE, S2F, S3F])
    browser.resolved_extra["t1-call-s3"] = {"scrolled_px": -200}
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S2), tool(SCROLL)], [answer(S2), tool(CLICK2)], [answer(S3)]))
        assert label_after(step_body(rig, 2), BADGE) == EXAMPLE
        assert label_after(step_body(rig, 3), S2F) == LU
        assert label_after(step_body(rig, 4), S3F) == LU
        await next_turn(rig, 2, "what is on the page", [answer("A shop. ")])
    log = turn_log(spy)
    got = outcomes(log)
    assert set(got) == {1}, got
    o = got[1]
    assert (o.kind, o.region) == ("elsewhere", "top right") and 0 < o.changed_cells < o.cells, o
    assert unmeasured(log) == {2: "not_pointer", 3: "scrolled"}, unmeasured(log)
    steps = {
        d.get("step", 1): d["live_label"]
        for d in live_labels(rig)
        if d.get("phase") in ("generate", "tool_step")
        and str(d.get("request_id", "")).startswith("t1")
    }
    assert set(steps) == {1, 2, 3, 4}, sorted(steps)
    assert steps[2].get("outcome") == {
        "kind": "elsewhere",
        "region": "top right",
        "changed_cells": o.changed_cells,
        "cells": o.cells,
    }, steps[2]
    assert steps[2].get("text") == EXAMPLE, steps[2]
    for k in (1, 3, 4):
        assert "outcome" not in steps[k], (k, steps[k])
    assert rig.record("t1").extras.get("action_outcome") == counts(elsewhere=1, unmeasured=1)
    assert rig.record("t2").extras.get("action_outcome") == counts(), rig.record("t2").extras
    assert len(reason_notes(rig, "not_pointer")) == 1 and len(reason_notes(rig, "scrolled")) == 1


async def test_outcome_record_shape(tmp_path: Path) -> None:
    """B-4: the ``outcome`` record of a placed clause has exactly the four keys, ``region`` ``None`` unless
    ``elsewhere`` (near: ``None``), ints for the cell counts."""
    browser = FakeLiveBrowser([W0], post_frames=[TICK])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S3)],))
    rec = [d["live_label"] for d in live_labels(rig) if d.get("step") == 2]
    assert len(rec) == 1, live_labels(rig)
    out = rec[0].get("outcome")
    assert isinstance(out, dict) and set(out) == OUTCOME_KEYS, out
    assert out["kind"] == "near" and out["region"] is None, out
    assert isinstance(out["changed_cells"], int) and isinstance(out["cells"], int), out


async def test_repair_and_other_bodies(tmp_path: Path) -> None:
    """B-3 ("every body of that step, its repair included"; reading R5 in the report): repairs exist for step 1 only
    (T-0114: a repairable step >= 2 draft is refused ``repair: step 1 only``), and step 1 never carries a clause.
    So: the step-1 call is repairable and repaired; the repair body carries exactly ``[L1]`` and its record's
    ``live_label`` has no ``outcome``; the repaired click (white -> badge) is measured and step 2 carries ``EXAMPLE``
    in its one body; a repairable step-2 draft sends no repair request; no prefill and no caption request carries
    the clause; across every body the clause appears exactly once (step 2)."""
    from tests.fakes.llm import is_caption_prompt, is_step_prompt

    browser = FakeLiveBrowser([W0], post_frames=[BADGE, CORNER])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(REPAIRABLE_CLICK)]) as rig:
        rig.llm.script_continuation(tool(REPAIRED_CLICK))
        rig.llm.script_steps([answer(S2), tool(REPAIRABLE_CLICK)], [answer(S3)])
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1, limit_ms=20_000, step=100)
        await rig.turn_done(limit_ms=20_000)
        await rig.advance(300)
        assert [e.status for e in tsl.results_for(rig, "t1-call")] == ["executed"], "premise"
        assert label_after(step_body(rig, 2), BADGE) == EXAMPLE
        reps = [r for r in rig.llm.continuations() if not is_step_prompt(r[1])]
        assert len(reps) == 1, f"premise: one (step-1) repair request; {[r[0] for r in reps]}"
        assert labels(_wire(reps[0][1])) == [L1], labels(_wire(reps[0][1]))
        bodies = all_bodies(rig)
        assert sum(json.dumps(b).count(CLAUSE_STEM) for b in bodies) == 1, [
            labels(b) for b in bodies
        ]
        for rid, p in rig.llm.prefills:
            assert CLAUSE_STEM not in json.dumps(_wire(p)), f"the prefill {rid} carries the clause"
        for rid, p, _ in rig.llm.generates:
            if is_caption_prompt(p):
                assert CLAUSE_STEM not in json.dumps(_wire(p)), rid
    rep = [d for d in rig.kind("llm.request") if d.get("phase") == "tool_repair"]
    assert len(rep) == 1, [d.get("phase") for d in rig.kind("llm.request")]
    assert (
        rep[0].get("live_label", {}).get("text") == L1 and "outcome" not in rep[0]["live_label"]
    ), rep[0]


# =====================================================================================================
# B-1: the keyword
# =====================================================================================================


def test_keyword_default_false() -> None:
    """B-1 (reviewer mutation 10, the default ``True``): ``Session.__init__`` has ``action_outcome`` with default
    ``False``."""
    params = inspect.signature(session_mod().Session.__init__).parameters
    assert "action_outcome" in params, "T-0136 B-1: Session(action_outcome=...) does not exist"
    assert params["action_outcome"].default is False, params["action_outcome"]


async def test_keyword_refusals(tmp_path: Path) -> None:
    """B-1 (R3): ``ValueError`` naming ``action_outcome`` with a ``dry_run`` kit (and naming ``live``), with
    ``live_feedback=None``, and with ``LiveFeedback(screen_label=False)``; companion (not too wide): the label on, with
    and without the extent, starts."""
    # reviewer B (O-B2): the dry-run arm runs with the label on, so only the tools refusal can fire (with
    # ``feedback=None`` the label refusal fired first and a mutant dropping the mode check survived); T-0134's own
    # ``live_feedback needs tools = 'live'`` refusal does not name ``action_outcome``, so it cannot pass this row
    with pytest.raises(ValueError, match=r"action_outcome[^\n]*tools = 'live'") as dry:
        async with outcome_rig(
            tmp_path / "dry",
            FakeLiveBrowser([W0]),
            script=[],
            feedback=label_on(),
            kit=T.make_kit(),
        ):
            pass
    assert "live_feedback needs" not in str(dry.value), dry.value
    with pytest.raises(ValueError, match="action_outcome"):
        async with outcome_rig(tmp_path / "none", FakeLiveBrowser([W0]), script=[], feedback=None):
            pass
    with pytest.raises(ValueError, match="action_outcome"):
        async with outcome_rig(
            tmp_path / "nolabel",
            FakeLiveBrowser([W0]),
            script=[],
            feedback=live_feedback(screen_label=False),
        ):
            pass
    for name, fb in (("label", label_on()), ("extent", label_on(True))):
        async with outcome_rig(
            tmp_path / name, FakeLiveBrowser([W0]), script=[], feedback=fb
        ) as rig:
            assert rig.session is not None


@pytest.mark.parametrize("combo", ["overview_frame", "auto_overview", "page_overview", "reground"])
async def test_keyword_composes(tmp_path: Path, combo: str) -> None:
    """B-1: ``action_outcome=True`` composes with the overview frame, the automatic overview, the page overview tool
    and ReGround: the session starts."""
    browser = FakeLiveBrowser([W0])
    kit = None
    extra: dict[str, Any] = {}
    if combo == "page_overview":
        from tests.unit.test_t0140_session import overview_kit

        kit = overview_kit(browser)
    else:
        extra[combo] = True
    async with outcome_rig(tmp_path, browser, script=[], extra=extra, kit=kit) as rig:
        assert rig.session is not None


async def test_explicit_false_is_off(tmp_path: Path) -> None:
    """B-1/B-7: ``action_outcome=False`` given explicitly behaves as off: the same measured scene places no clause,
    the turn line has no ``action_outcome`` key and no ``live_label`` record an ``outcome``."""
    browser = FakeLiveBrowser([W0], post_frames=[BADGE])
    async with outcome_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        on=False,
        extra={"action_outcome": False},
    ) as rig:
        await three(rig, ([answer(S3)],))
        assert label_after(step_body(rig, 2), BADGE) == LU
    assert "action_outcome" not in rig.record("t1").extras
    assert all("outcome" not in d["live_label"] for d in live_labels(rig))


# =====================================================================================================
# B-6: never spoken or stored
# =====================================================================================================


async def test_never_voiced_or_stored(tmp_path: Path) -> None:
    """B-6 (adversarial): across a measured two-click turn and a second turn, the clause stem and the outcome texts
    reach no ``TtsSpeak`` (text or source), history, HUD message, ``Episode`` or ``machine.events()``; the requests
    carry them (premise)."""
    browser = FakeLiveBrowser([W0, live_frame("t2")], post_frames=[BADGE, BADGE_CORNER])
    async with outcome_rig(tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await three(rig, ([answer(S2), tool(CLICK2)], [answer(S3)]))
        await next_turn(rig, 2, "and now", [answer("Still a shop. ")])
        sent = json.dumps(all_bodies(rig))
        needles = (CLAUSE_STEM, "changed elsewhere", "top right", "bottom left")
        for n in needles:
            assert n in sent, f"premise: the requests carry {n!r}"
        speaks = [c.text + "|" + c.source for c in rig.machine.commands(C.TtsSpeak)]
        assert speaks, "premise: something was spoken"
        stores = {
            "TtsSpeak": json.dumps(speaks),
            "history": tsl.history_text(rig),
            "HUD": tsl.published_json(rig),
            "Episode": json.dumps(
                [e.model_dump(mode="json") for e in rig.memory.episodes.values()]
            ),
            "machine.events()": json.dumps([repr(e) for e in rig.machine.events()]),
        }
        for where, text in stores.items():
            for n in needles:
                assert n not in text, f"T-0136 B-6: {n!r} reached {where}"
        assert STEM in sent
