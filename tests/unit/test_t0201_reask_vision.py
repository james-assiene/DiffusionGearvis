"""T-0201 group A (A1-A6): a T-0172 re-ask's "What I see" frame is the re-asked request's, never blind by accident
(docs/tasks/T-0201.md).

**The bug** (``logs/sessions/20261029-181336-282350.jsonl`` seq 1329 and 2962): during a re-ask the session published
``astra.vision.frame {blind: true, images: []}`` beside a request that carried one image. A re-ask's prompt is
``reask_prompt(<the step's prompt as sent>, <step text>)``, whose final user message holds only ``REASK_NUDGE``; the
frame must be the one of the request it re-asks (T-0177 B7's key ``(source, identities, blind)``).

Rigs: T-0177's ``vision_rig`` (the live tool rig, a socketless ``HudServer``, one greeted page; what the session
published is ``rig.published.messages``) with ``reask_announced_calls=True`` (T-0172's keyword) and the fake's re-ask
route (``script_reasks``, ``reask_prompts``, ``reask_bodies``). The re-ask's publish point is located by a tap on the
session log: ``cut`` is the number of published messages when the T-0172 B7 note ``announced call re-asked`` is
recorded (logged before the re-ask is built and sent); every frame published after it in these rows is the re-ask's
(each row's re-ask ends the turn: it answers without a call).

Oracles: published bytes against the fixture frames' bytes and the step request's wire body (T-0177's ``body_bytes``);
identities against the ``phash`` of the step prompt's ``ImagePart`` s; the BASE rows (A5, A6) against a subprocess
run of the same scenario over ``git archive`` of BASE (origin-proved). Nothing is recomputed with the code under test.

**Readings** (listed in the test-author's report):

- R-1 (A1 "publishes nothing new"): no ``astra.vision.frame`` is published after ``cut``, and the last frame
  published (the HUD slot) is still the step's, not blind.
- R-2 (A2, adversarial): the task names the wrong fix "newest image anywhere ... publishes turn 1's image". Read as
  the newest image of the history (outside the step's own final user message): turn 1's ``SH``, which the row's
  turn 2 carries in history and which must never be shown. Corrected (review, 2026-10-29): an earlier reading
  argued that the *positionally* newest image of the re-ask prompt is always the step's frame, so a "newest image
  anywhere" fallback (and a re-ask published from the step's ``unlabelled`` prompt) were equivalent to the fix.
  That is false: ``unlabelled`` never holds T-0151's overview frame (swapped into the sent prompt only, default-on
  in live on 1.05-2.0-screen pages since T-0152), T-0150's automatic overview or T-0140's step overview, and
  T-0150/T-0140 put two images in the final user message, where "newest image anywhere" publishes one. Both
  mutants are killed by the A1 rows ``test_reask_step1_overview_frame_publishes_nothing_new`` (``unlabelled``)
  and ``test_reask_step1_auto_overview_two_images_publishes_nothing_new`` (both).
- R-3 (A3 "a reset or new turn cleared the key"): neither can happen between a step and its own re-ask through the
  public seams (a reset ends the turn and a new turn aborts the re-ask), so the row clears ``Session._vision_key``
  (T-0177 B9's slot key, which ``_apply_reset`` clears; the task's notes keep it untouched) at the ``announced call
  re-asked`` note. The re-ask must then publish exactly one frame equal to the step's frame in ``source``, ``step``,
  ``images`` (identities and bytes) and ``history_images``, not blind.
- R-4 (A4): "still publishes ``blind: true``": every frame of the turn is blind with the turn's choice; with the
  key cleared (R-3) the re-ask publishes exactly one blind frame with the turn's ``vision_source``.
- R-5 (A5): "byte-identical to BASE" is read on the published ``astra.vision.frame`` messages of one live turn that
  sends a prefill, a step-1 generate, a T-0171 resend at step 2 and a T-0179 degenerate retry at step 3 (premises
  asserted), compared with the same scenario over BASE's tree. Captioner and ReGround pass 2 are T-0177's rows
  (``test_the_captioner_publishes_nothing``, ``test_a_reground_crop_is_never_published``), which run unchanged.
- R-6 (A6): the re-ask's wire body (``reask_bodies``, the real client's ``request_body``) and its ``llm.request``
  record (``phase: announce_reask``) at step 1 and step 2 equal BASE's, with the session id and log paths
  normalised. A6 pins what is *sent*, not what is *shown*: it is green for any choice of the prompt the re-ask's
  frame is published from, so it says nothing about the overview steps (whose sent prompt differs from
  ``unlabelled``); those are pinned by the two overview A1 rows (R-2).

**BASE-green guards** (deliberate, named): ``test_reask_step1_publishes_nothing_new`` and
``test_reask_after_key_reset_publishes_reasked_images[step1]`` (at a single-image step 1 without an overview, BASE's
history fallback already rides on the step-1 frame, which is one of the turn's frames: there the bug is at step >= 2,
as in both logged cases, steps 12 and 5; with T-0151's overview frame or T-0150's automatic overview BASE is red at
step 1 too, see the two overview A1 rows), ``test_reask_of_blind_request_stays_blind`` (every case), ``test_other_paths_publish_as_base``,
``test_reask_body_and_record_unchanged``, ``test_reask_of_blind_request_after_a_seen_turn_stays_blind`` and
``test_publish_never_matches_the_nudge_text``. Every other row is red at BASE 7b4bca27 (pending until activation at
the start of ``/implement T-0201`` group A).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), tools
``live`` (1280x800 frames, ReGround off, labels at the session default), n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.llm import Prompt
from tests import _git_range as gr
from tests.fakes.llm import engine_empty_error, is_reask_prompt
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0142_session as t0142
from tests.unit._live_browser_support import live_frame, turn_extras
from tests.unit._tool_turn_support import VALID_CLICK, answer, thought, tool
from tests.unit.test_session_live import _wire
from tests.unit.test_t0177_shared_frame import (
    AUTO_KEYS,
    DEALS,
    OV,
    OVERVIEW_KEYS,
    SH,
    TALL,
    VisionBrowser,
    overview_browser,
    select,
    vision_rig,
)
from tests.unit.test_t0177_vision_frame_publish import (
    FRAME,
    body_bytes,
    final_identities,
    frames,
    published_bytes,
)

#: BASE: main at 7b4bca27 (``T-0199 merged``), where group A branched. A literal, never a moving ref (tests.md).
BASE_SHA = "7b4bca2703e4c4b01e1494e8ef4ae28a40030cd6"

REASK_KW = {"reask_announced_calls": True}
NOTE_ASKED = "announced call re-asked"
PHASE = "announce_reask"
ANN_CLICK = "I will click the Sign in button."
R_STOPPED = "The search limit was reached, so I could not search again."
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
CLICK2 = tsl.CLICK2
F0 = live_frame("t0201-onset")
F1 = live_frame("t0201-after-1")
F2 = live_frame("t0201-after-2")


# --------------------------------------------------------------------------------------------- the tap


class ReaskTap:
    """Taps the session log: ``cut`` is ``len(rig.published.messages)`` when the ``announced call re-asked`` note is
    recorded (``None`` until then). ``clear_key``: also forget the published key there (R-3)."""

    def __init__(self, rig: Any, *, clear_key: bool = False) -> None:
        self.cut: int | None = None
        log = rig.log
        real: Callable[..., Any] = log.record

        def record(kind: str, data: Any, *a: Any, **k: Any) -> Any:
            if kind == "note" and isinstance(data, dict) and data.get("what") == NOTE_ASKED:
                assert self.cut is None, "premise: one re-ask per turn"
                self.cut = len(rig.published.messages)
                if clear_key:
                    rig.session._vision_key = None  # R-3: what a reset does to the key (T-0177 B9)
            return real(kind, data, *a, **k)

        log.record = record

    def after(self, rig: Any) -> list[Any]:
        assert self.cut is not None, (
            f"premise: the re-ask was sent (no {NOTE_ASKED!r} note; re-asks {rig.llm.reask_prompts})"
        )
        return [m for m in rig.published.messages[self.cut :] if getattr(m, "type", None) == FRAME]

    def before(self, rig: Any) -> list[Any]:
        assert self.cut is not None, f"premise: the re-ask was sent (no {NOTE_ASKED!r} note)"
        return [m for m in rig.published.messages[: self.cut] if getattr(m, "type", None) == FRAME]


def one_reask(rig: Any, machine_id: str) -> Prompt:
    found = [p for rid, p in rig.llm.reask_prompts if f"-{machine_id}-" in rid]
    assert len(rig.llm.reask_prompts) == 1 and len(found) == 1, (
        f"premise: exactly one re-ask under {machine_id}; {[r for r, _ in rig.llm.reask_prompts]}"
    )
    return found[0]


def step_prompt_of(rig: Any, machine_id: str) -> Prompt:
    """The step's own request (its last attempt that is not a re-ask), as the backend was handed it."""
    found = [p for _, p in tsl.generates_for(rig, machine_id) if not is_reask_prompt(p)]
    assert found, f"premise: a request was sent for {machine_id}"
    return found[-1]


def shape(f: Any) -> tuple[Any, ...]:
    return (f.turn_id, f.step, f.source, f.blind, f.history_images, [i.identity for i in f.images])


async def wait_turns(rig: Any, n: int) -> None:
    await rig.until(lambda: len(rig.lines()) >= n, f"turn {n} never ended", limit_ms=40_000)
    await rig.advance(300)


async def step2_reask(rig: Any, *, clear_key: bool = False) -> ReaskTap:
    """Turn 1, live browser: step 1 clicks (``F0`` -> ``F1``), step 2 only announces a click, the re-ask answers."""
    tap = ReaskTap(rig, clear_key=clear_key)
    rig.llm.script_steps([answer(ANN_CLICK)])
    rig.llm.script_reasks([answer(R_STOPPED)])
    await rig.ask(tsl.QUESTION)
    await wait_turns(rig, 1)
    return tap


async def step1_reask(rig: Any, *, clear_key: bool = False) -> ReaskTap:
    """Turn 1, live browser: step 1 only announces a click (``F0``), the re-ask answers."""
    tap = ReaskTap(rig, clear_key=clear_key)
    rig.llm.script_reasks([answer(R_STOPPED)])
    await rig.ask(tsl.QUESTION)
    await wait_turns(rig, 1)
    return tap


async def history_then_step2_reask(rig: Any, *, clear_key: bool = False) -> ReaskTap:
    """Turn 1 shared (``SH``, answered); turn 2 browser: step 1 clicks (``F0`` -> ``F1``), step 2 only announces,
    the re-ask answers. Turn 2's requests carry turn 1's ``SH`` in history."""
    await select(rig, "shared")
    rig.llm.script = [answer("A shop. ")]
    await rig.ask("what is on my screen")
    await wait_turns(rig, 1)
    await select(rig, "browser")
    tap = ReaskTap(rig, clear_key=clear_key)
    rig.llm.script = [answer(S1), tool(VALID_CLICK)]
    rig.llm.script_steps([answer(ANN_CLICK)])
    rig.llm.script_reasks([answer(R_STOPPED)])
    await rig.ask("and now click it")
    await wait_turns(rig, 2)
    return tap


# =====================================================================================================
# A1: the re-ask's key is the step's; already published, so nothing new
# =====================================================================================================


async def test_reask_step2_publishes_nothing_new(tmp_path: Path) -> None:
    """A1, step >= 2, live browser: step 2 publishes ``F1`` (premise: exactly the step body's bytes); its re-ask
    publishes no frame (R-1), and the last frame published is still step 2's ``F1``, not blind. BASE published a
    blind frame with no image here (the seq 1329 bug)."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], session_kwargs=REASK_KW
    ) as rig:
        tap = await step2_reask(rig)
        one_reask(rig, tsl.step_id(2))
        step2 = step_prompt_of(rig, tsl.step_id(2))
        before, after = tap.before(rig), tap.after(rig)
        every = frames(rig, "t1")
    assert before and published_bytes(before[-1]) == body_bytes(_wire(step2)) == [F1.jpeg], (
        f"premise: step 2 published F1; {[shape(f) for f in before]}"
    )
    assert after == [], (
        f"T-0201 A1: the re-ask's key equals step 2's, so it publishes nothing new; got {[shape(f) for f in after]}"
    )
    last = every[-1]
    assert (last.step, last.source, last.blind) == (2, "browser", False), shape(last)
    assert published_bytes(last) == [F1.jpeg]
    assert [i.identity for i in last.images] == final_identities(step2)


async def test_reask_step1_publishes_nothing_new(tmp_path: Path) -> None:
    """A1, step 1, live browser: the step-1 request publishes ``F0``; its re-ask (a new assistant message and the
    nudge after the step-1 prompt) publishes nothing, and the slot is still ``F0``, not blind."""
    browser = VisionBrowser([F0])
    async with vision_rig(
        tmp_path, browser, script=[answer(ANN_CLICK)], session_kwargs=REASK_KW
    ) as rig:
        tap = await step1_reask(rig)
        one_reask(rig, tsl.step_id(1))
        step1 = step_prompt_of(rig, tsl.step_id(1))
        before, after = tap.before(rig), tap.after(rig)
        every = frames(rig, "t1")
    assert before and published_bytes(before[-1]) == body_bytes(_wire(step1)) == [F0.jpeg], (
        f"premise: step 1 published F0; {[shape(f) for f in before]}"
    )
    assert after == [], (
        f"T-0201 A1: the step-1 re-ask publishes nothing new; got {[shape(f) for f in after]}"
    )
    last = every[-1]
    assert (last.step, last.source, last.blind, published_bytes(last)) == (
        1,
        "browser",
        False,
        [F0.jpeg],
    ), shape(last)


async def test_reask_step1_overview_frame_publishes_nothing_new(tmp_path: Path) -> None:
    """A1, step 1 with T-0151's overview frame (default-on in live on 1.05-2.0-screen pages since T-0152): the
    1.525-screen page, ``overview_frame`` on, the browser source. Step 1 sends only the whole-page picture ``OV``
    (swapped into the sent prompt; premise: the step body's bytes, ``shown`` 1) and publishes it. Its re-ask
    publishes nothing new (R-1), and the slot is still the overview picture, never the live frame it replaced and
    never blind. Kills a re-ask that rebuilds its frame from the conversation's own images (which never hold the
    swapped-in overview): that publishes a second frame after the cut."""
    browser = overview_browser(DEALS)
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(ANN_CLICK)],
        session_kwargs={**REASK_KW, "overview_frame": True},
    ) as rig:
        tap = await step1_reask(rig)
        one_reask(rig, tsl.step_id(1))
        step1 = step_prompt_of(rig, tsl.step_id(1))
        before, after = tap.before(rig), tap.after(rig)
        every = frames(rig, "t1")
        counts = turn_extras(rig).get("overview_frame")
    assert counts == {**dict.fromkeys(OVERVIEW_KEYS, 0), "shown": 1}, (
        f"premise: the overview frame is shown at step 1; {counts}"
    )
    assert body_bytes(_wire(step1)) == [OV.jpeg], "premise: step 1 sends only the overview picture"
    assert before and published_bytes(before[-1]) == [OV.jpeg], (
        f"premise: step 1 published the overview picture; {[shape(f) for f in before]}"
    )
    assert after == [], (
        f"T-0201 A1: the overview step's re-ask publishes nothing new; got {[shape(f) for f in after]}"
    )
    last = every[-1]
    assert (last.step, last.source, last.blind, published_bytes(last)) == (
        1,
        "browser",
        False,
        [OV.jpeg],
    ), shape(last)
    assert [(i.kind, i.width, i.height) for i in last.images] == [("overview", OV.width, OV.height)]


async def test_reask_step1_auto_overview_two_images_publishes_nothing_new(tmp_path: Path) -> None:
    """A1, step 1 with T-0150's automatic overview: the 9.3-screen page with a 320x800 whole-page picture,
    ``auto_overview`` on, the browser source. Step 1's final user message carries two images, the live frame and
    the overview (premise: ``shown`` 1, the step body's final-message bytes are two and include ``OV``, and the
    step published exactly those two). Its re-ask publishes nothing new (R-1), and the slot still holds both
    images, not blind. Kills a re-ask frame built from the conversation's images (no overview there) and a
    "newest image anywhere" fallback (one image where the step published two)."""
    browser = overview_browser(TALL)
    browser.overview_image = OV.model_copy(update={"width": 320, "height": 800})
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(ANN_CLICK)],
        session_kwargs={**REASK_KW, "auto_overview": True},
    ) as rig:
        tap = await step1_reask(rig)
        one_reask(rig, tsl.step_id(1))
        step1 = step_prompt_of(rig, tsl.step_id(1))
        before, after = tap.before(rig), tap.after(rig)
        every = frames(rig, "t1")
        counts = turn_extras(rig).get("auto_overview")
    assert counts == {**dict.fromkeys(AUTO_KEYS, 0), "shown": 1}, (
        f"premise: the automatic overview is shown at step 1; {counts}"
    )
    sent = body_bytes(_wire(step1))
    assert len(sent) == 2 and OV.jpeg in sent, (
        f"premise: step 1's final user message carries two images, one the overview; {len(sent)}"
    )
    assert before and published_bytes(before[-1]) == sent, (
        f"premise: step 1 published both of its images; {[shape(f) for f in before]}"
    )
    assert after == [], (
        f"T-0201 A1: the two-image step's re-ask publishes nothing new; got {[shape(f) for f in after]}"
    )
    last = every[-1]
    assert (last.step, last.source, last.blind) == (1, "browser", False), shape(last)
    assert published_bytes(last) == sent, shape(last)
    assert [i.identity for i in last.images] == final_identities(step1), shape(last)


# =====================================================================================================
# A2 (adversarial): an older turn's image in history is never what the re-ask shows
# =====================================================================================================


@pytest.mark.parametrize("clear", [False, True], ids=["kept-key", "cleared-key"])
async def test_reask_with_history_image_not_history_not_blind(tmp_path: Path, clear: bool) -> None:
    """A2 (adversarial, R-2). Turn 1 is shared (``SH``, answered); turn 2 is the browser: step 1 clicks, step 2
    (``F1``, ``history_images`` 1: premise) only announces, and is re-asked. Turn 1's ``SH`` is the newest image of
    turn 2's history.

    ``kept-key``: the re-ask publishes nothing new, so the slot stays step 2's ``F1`` (never ``SH``, never blind).
    ``cleared-key`` (R-3): the re-ask's one frame is ``F1`` -- never the history image ``SH`` -- with ``source``
    browser and ``history_images`` equal to step 2's."""
    browser = VisionBrowser([F0], post_frames=[F1])
    async with vision_rig(tmp_path, browser, session_kwargs=REASK_KW) as rig:
        tap = await history_then_step2_reask(rig, clear_key=clear)
        one_reask(rig, tsl.step_id(2, "t2"))
        step2 = step_prompt_of(rig, tsl.step_id(2, "t2"))
        before, after = tap.before(rig), tap.after(rig)
        every = frames(rig, "t2")
    assert body_bytes(_wire(step2)) == [F1.jpeg] and body_bytes(_wire(step2), final_only=False) == [
        SH.jpeg,
        F1.jpeg,
    ], "premise: step 2 sends F1, with turn 1's SH in its history"
    mine = [f for f in before if f.turn_id == "t2"]
    assert mine and published_bytes(mine[-1]) == [F1.jpeg], (
        f"premise: t2 step 2 published F1; {[shape(f) for f in mine]}"
    )
    assert mine[-1].history_images == 1, f"premise: one history image; {shape(mine[-1])}"
    shown = [b for f in after for b in published_bytes(f)]
    assert SH.jpeg not in shown, (
        f"T-0201 A2: the re-ask never shows turn 1's history image; {[shape(f) for f in after]}"
    )
    if not clear:
        assert after == [], (
            f"T-0201 A2: the re-ask publishes nothing new; got {[shape(f) for f in after]}"
        )
        assert (every[-1].blind, published_bytes(every[-1])) == (False, [F1.jpeg]), shape(every[-1])
        return
    assert len(after) == 1, (
        f"T-0201 A2/A3: with the key cleared the re-ask publishes one frame; {[shape(f) for f in after]}"
    )
    f = after[0]
    assert published_bytes(f) == [F1.jpeg], f"T-0201 A2: the current frame; {shape(f)}"
    assert [i.identity for i in f.images] == final_identities(step2), shape(f)
    assert (f.step, f.source, f.blind, f.history_images) == (
        2,
        "browser",
        False,
        mine[-1].history_images,
    ), f"T-0201 A2: history_images equals the step's ({mine[-1].history_images}); {shape(f)}"


# =====================================================================================================
# A3: the key was cleared before the re-ask's publish
# =====================================================================================================


@pytest.mark.parametrize("at", [pytest.param(1, id="step1"), pytest.param(2, id="step2")])
async def test_reask_after_key_reset_publishes_reasked_images(tmp_path: Path, at: int) -> None:
    """A3 (R-3): with the published key forgotten at the re-ask, the re-ask publishes exactly one frame: the
    re-asked request's images (bytes of the step body, identities of its ``ImagePart`` s), not blind, with the step's
    ``source``, ``step``, ``images`` and ``history_images``."""
    if at == 2:
        browser = VisionBrowser([F0], post_frames=[F1])
        script, run, want = [answer(S1), tool(VALID_CLICK)], step2_reask, F1
    else:
        browser = VisionBrowser([F0])
        script, run, want = [answer(ANN_CLICK)], step1_reask, F0
    async with vision_rig(tmp_path, browser, script=script, session_kwargs=REASK_KW) as rig:
        tap = await run(rig, clear_key=True)
        one_reask(rig, tsl.step_id(at))
        stepped = step_prompt_of(rig, tsl.step_id(at))
        before, after = tap.before(rig), tap.after(rig)
    assert before and before[-1].step == at, (
        f"premise: step {at} published; {[shape(f) for f in before]}"
    )
    step_frame = before[-1]
    assert len(after) == 1, (
        f"T-0201 A3: the re-ask publishes its frame once the key was cleared; got {[shape(f) for f in after]}"
    )
    f = after[0]
    assert not f.blind, f"T-0201 A3: never blind; {shape(f)}"
    assert published_bytes(f) == body_bytes(_wire(stepped)) == [want.jpeg], shape(f)
    assert [i.identity for i in f.images] == final_identities(stepped), shape(f)
    assert (f.turn_id, f.step, f.source, f.history_images) == (
        "t1",
        at,
        step_frame.source,
        step_frame.history_images,
    ), (shape(f), shape(step_frame))
    assert f.images == step_frame.images, "T-0201 A3: the step's images, field for field"


# =====================================================================================================
# A4 (negative): a re-ask of a request with no image anywhere stays blind
# =====================================================================================================


@pytest.mark.parametrize("clear", [False, True], ids=["kept-key", "cleared-key"])
@pytest.mark.parametrize("choice", ["browser", "shared"])
async def test_reask_of_blind_request_stays_blind(tmp_path: Path, choice: str, clear: bool) -> None:
    """A4 (R-4, BASE-green guard): the step-1 request carries no image anywhere (``shared``: no HUD frame; ``browser``:
    ``frame()`` raises). Every t1 frame is blind with the turn's choice; with the key cleared the re-ask publishes
    exactly one frame, ``blind: true, images: []``, ``source`` the turn's choice."""
    browser = VisionBrowser([])
    if choice == "browser":
        browser.frame_mode = "raise"
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(ANN_CLICK)],
        hud_frames=[] if choice == "shared" else None,
        session_kwargs=REASK_KW,
    ) as rig:
        if choice == "shared":
            await select(rig, "shared")
        tap = await step1_reask(rig, clear_key=clear)
        reask = one_reask(rig, tsl.step_id(1))
        after = tap.after(rig)
        every = frames(rig, "t1")
    assert body_bytes(_wire(reask), final_only=False) == [], "premise: no image anywhere"
    assert every and all((f.blind, f.images, f.source) == (True, [], choice) for f in every), [
        shape(f) for f in every
    ]
    if clear:
        assert [(f.blind, f.images, f.source, f.step) for f in after] == [(True, [], choice, 1)], [
            shape(f) for f in after
        ]
    else:
        assert after == [], [shape(f) for f in after]


@pytest.mark.parametrize("clear", [False, True], ids=["kept-key", "cleared-key"])
async def test_reask_of_blind_request_after_a_seen_turn_stays_blind(
    tmp_path: Path, clear: bool
) -> None:
    """A4 (R-4, BASE-green guard; a blind request publishing an image): turn 1 shows the browser's ``F0`` (published);
    turn 2's ``frame()`` raises, so its step-1 request carries no image anywhere (premise) and publishes blind. The
    re-ask never shows ``F0`` (the last image the panel showed): kept key, nothing new; cleared key, one blind frame
    with the turn's choice."""
    browser = VisionBrowser([F0])
    async with vision_rig(tmp_path, browser, session_kwargs=REASK_KW) as rig:
        rig.llm.script = [answer("A page. ")]
        await rig.ask("what is on my screen")
        await wait_turns(rig, 1)
        assert [published_bytes(f) for f in frames(rig, "t1")] == [[F0.jpeg]], (
            "premise: turn 1 showed F0"
        )
        browser.frame_mode = "raise"
        tap = ReaskTap(rig, clear_key=clear)
        rig.llm.script = [answer(ANN_CLICK)]
        rig.llm.script_reasks([answer(R_STOPPED)])
        await rig.ask("and now click it")
        await wait_turns(rig, 2)
        reask = one_reask(rig, tsl.step_id(1, "t2"))
        before, after = tap.before(rig), tap.after(rig)
    assert body_bytes(_wire(reask), final_only=False) == [], "premise: no image anywhere"
    assert [(f.blind, f.source) for f in before if f.turn_id == "t2"] == [(True, "browser")], (
        f"premise: t2 step 1 published blind; {[shape(f) for f in before]}"
    )
    want = [("t2", 1, "browser", True, 0, [])] if clear else []
    assert [shape(f) for f in after] == want, (
        f"T-0201 A4: the re-ask of a blind request stays blind; {[shape(f) for f in after]}"
    )


def test_publish_never_matches_the_nudge_text() -> None:
    """Implementation note "never match on ``REASK_NUDGE`` text" (BASE-green guard): the session's code (tokenised;
    comments are not code) names no ``REASK_NUDGE`` and holds no string literal containing the nudge's text, so the
    re-ask's frame cannot come from recognising the nudge. Prose cannot trip it: only ``NAME`` and ``STRING`` tokens
    are read, and a string is matched on the nudge's whole text."""
    import io
    import tokenize

    from local_astra.conductor import session as session_mod
    from local_astra.toolcall.announced_call import REASK_NUDGE

    assert len(REASK_NUDGE) > 20, "premise: the nudge is a sentence"
    source = Path(session_mod.__file__).read_text()
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    names = [t for t in tokens if t.type == tokenize.NAME and t.string == "REASK_NUDGE"]
    literals = [t for t in tokens if t.type == tokenize.STRING and REASK_NUDGE in t.string]
    assert names == [] and literals == [], (
        f"T-0201 A (notes): session.py matches on the re-ask nudge (lines "
        f"{[t.start[0] for t in names + literals]}); pass the re-asked prompt to the publish point instead"
    )


# =====================================================================================================
# A5, A6 (negative): every other publish path, and the re-ask request itself, as at BASE
# =====================================================================================================


def _norm(obj: Any, session_id: str, tmp: str) -> Any:
    text = json.dumps(obj, sort_keys=True, default=str)
    return json.loads(text.replace(session_id, "<sid>").replace(tmp, "<tmp>"))


async def _paths_scenario(tmp_path: Path) -> dict[str, Any]:
    """A5's turn: a prefill and a step-1 generate (``F0``), step 2 resent once after an empty engine response
    (T-0171, ``F1``), step 3 retried once after a degenerate thought (T-0179, ``F2``), then an answer."""
    browser = VisionBrowser([F0], post_frames=[F1, F2])
    async with vision_rig(
        tmp_path,
        browser,
        script=[answer(S1), tool(VALID_CLICK)],
        session_kwargs={"degeneracy_guard": True},
    ) as rig:
        rig.llm.fail_steps(engine_empty_error)
        rig.llm.script_steps(
            [answer(S2), tool(CLICK2)],
            [thought(t0142.COLLAPSE)],
            [answer(S3)],
        )
        await rig.ask(tsl.QUESTION)
        await wait_turns(rig, 1)
        sid = rig.session.session_id
    notes = [d.get("what") for d in rig.kind("note")]
    published = [m.model_dump(mode="json") for m in frames(rig)]
    return _norm(
        {
            "frames": published,
            "prefills": len(rig.llm.prefills),
            "notes": notes,
            "steps": [rid for rid, _ in rig.llm.step_prompts],
        },
        sid,
        str(tmp_path),
    )


async def _reask_scenario(tmp_path: Path) -> dict[str, Any]:
    """A6's turns: a step-2 re-ask and a step-1 re-ask (two sessions)."""
    out: dict[str, Any] = {}
    for name, browser, script, run in (
        (
            "step2",
            VisionBrowser([F0], post_frames=[F1]),
            [answer(S1), tool(VALID_CLICK)],
            step2_reask,
        ),
        ("step1", VisionBrowser([F0]), [answer(ANN_CLICK)], step1_reask),
    ):
        tmp = tmp_path / name
        async with vision_rig(tmp, browser, script=script, session_kwargs=REASK_KW) as rig:
            await run(rig)
            sid = rig.session.session_id
        records = [d for d in rig.kind("llm.request") if d.get("phase") == PHASE]
        out[name] = _norm({"bodies": list(rig.llm.reask_bodies), "records": records}, sid, str(tmp))
    return out


async def _capture(tmp_path: Path) -> dict[str, Any]:
    return {
        "paths": await _paths_scenario(tmp_path / "paths"),
        "reask": await _reask_scenario(tmp_path / "reask"),
    }


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0201_reask_vision as m
out = pathlib.Path(sys.argv[1])
data = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "data": data}))
"""


def _base_capture(tmp_path: Path) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp_path / "base_tree")
    out = tmp_path / "base_out" / "data.json"
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
    # each scenario's own tmp dir is normalised to <tmp> by ``_norm``, so the base (<root>/base_out/run) and the
    # observed side (<root>/now) compare
    data: dict[str, Any] = got["data"]
    return data


@pytest.fixture(scope="module")
def base_and_now(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any]]:
    root = tmp_path_factory.mktemp("t0201_base")
    base = _base_capture(root)
    now = asyncio.run(_capture(root / "now"))
    return base, now


def test_other_paths_publish_as_base(base_and_now: tuple[dict[str, Any], dict[str, Any]]) -> None:
    """A5 (R-5, BASE-green guard): the prefill, generate, step, T-0171 resend and T-0179 retry publishes of one live
    turn equal BASE's message for message."""
    base, now = base_and_now
    b, n = base["paths"], now["paths"]
    assert n["prefills"] >= 1, "premise: a prefill was sent"
    assert "the engine returned an empty retryable response; retrying once" in n["notes"], (
        f"premise: a T-0171 resend; notes {n['notes']}"
    )
    assert "degenerate request: retrying once" in n["notes"], (
        f"premise: a T-0179 retry; {n['notes']}"
    )
    assert len(n["frames"]) >= 3, f"premise: frames for F0, F1, F2; {len(n['frames'])}"
    assert n == b, (
        "T-0201 A5: the other publish paths are byte-identical to BASE\n"
        f"  base frames {[(f['step'], f['blind'], f['source']) for f in b['frames']]}\n"
        f"  now  frames {[(f['step'], f['blind'], f['source']) for f in n['frames']]}"
    )


def test_reask_body_and_record_unchanged(
    base_and_now: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """A6 (R-6, BASE-green guard): the step-1 and step-2 re-asks' wire bodies and ``llm.request`` records equal
    BASE's."""
    base, now = base_and_now
    for name in ("step1", "step2"):
        b, n = base["reask"][name], now["reask"][name]
        assert len(n["bodies"]) == 1 and len(n["records"]) == 1, f"premise ({name}): one re-ask"
        assert n["bodies"] == b["bodies"], (
            f"T-0201 A6 ({name}): the re-ask's wire body equals BASE's"
        )
        assert n["records"] == b["records"], (
            f"T-0201 A6 ({name}): its llm.request record equals BASE's"
        )
