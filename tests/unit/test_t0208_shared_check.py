"""T-0208 group A2: the completion check on a shared-start turn (docs/tasks/T-0208.md A2.1-A2.7; H-T208-1 ruled yes).

A2.1 replaces T-0207 A2.1's clause ``turn.vision_source == "browser"`` by "the step request's current image is not a
user's shared frame" (the session's one frame-kind fact ``_is_shared`` over the identity of the current image in the
prompt the step sent). So under the ``shared`` choice the check fires once the step's image is a browser frame (an
executed action swapped it in, T-0177 B1), and never while the step still shows the shared image.

**Rigs.** T-0177's ``vision_rig`` (the live tool rig over a ``VisionBrowser``, the HUD perception serving the shared
frame ``SH``, a socketless ``HudServer``; selects over the wire) with ``Session(completion_check=...)`` passed only
when the row gives it, driven by T-0207's ``drive`` (steps, checks and re-asks scripted on the fake LLM's T-0207
routes). Browser frames are real JPEGs (``jpeg_frame``) so T-0200's blank predicate can decode the fresh capture;
``SH`` is T-0177's distinctive non-JPEG shared payload (never decoded by the check: the check's frame is a browser
frame).

**Oracles.** The check prompt is T-0207 A1's ``check_prompt`` over the step prompt the fake received, with the fresh
frame keyed by its capture identity typed by hand (``held``); image kinds are told apart by their bytes (the shared
frame's base64 in a data URL, or one of the browser frames this module made), never by the session's own fact. The
prefix property renders through ``tests/cd/oracle/gemma4_template.py`` (T-0172's ``render``; the row skips, naming
what is missing, without jinja2 or the pinned template: run it with ``uv run --group cd``). BASE equality (A2.5) is a
subprocess run over ``git archive`` of BASE, origin-proved (T-0207's ``_base_capture`` pattern).

**Readings** (each in the test-author's report):

- R-A1 (orchestrator ruling, docs/tasks/T-0208.md Status): "one source" / A2.3's "no body holds both a shared and a
  browser image" is read over the current turn's images (its user message(s) and the step's own images), never over
  history. A single-turn row's scan (``assert_one_source``) asserts that premise (one turn line) and reads every
  image of every request body; a two-turn row (review 2) names its turn count and reads each body from its turn's
  own user message on (the index of the last user message of that turn's step-1 body; the body's earlier messages
  must equal that body's, a premise checked per request), so history images of the other source are excluded.
- R-A2 (orchestrator ruling, docs/tasks/T-0208.md Status): on a turn whose step 1 was a shared frame, an executed
  ``wait`` alone does not arm; any other executed browser action arms as A2.1. Read with the T-0202 wait kit
  (``cli.load_tool_kit(..., wait_tool=True)``) over the ``VisionBrowser``; the fake's ``wait`` executes and takes
  its after-frame (as the executor's does, T-0202 R1). The ruling's two sides are two rows: wait only (absent) and
  wait then scroll (too wide).
- A2.3's "no fresh browser frame can be taken" is T-0177 B-R3's path: the scroll executes, its post-action frame is
  not the browser's latest, and the one fresh ``frame()`` fails, so step 2 still carries the shared image and its
  label. Its fresh ``frame()`` works afterwards, so a session that armed on ``browser_actions > 0`` alone would take a
  frame and send a check: the row's "no check" cannot be met by a frame failure.
- A2.4's "blind shared turn" is ``hud_frames=[]`` under ``shared`` (T-0177 A6): step 1 has no image and, after the
  executed scroll, step 2 has none either (it "stays blind", observed at BASE). A step with no image is not a shared
  step, so the image clause alone would arm; only the non-blind condition blocks.
- A2.4's power is A2.2's row: each blocked case differs from it in the one named feature.
- A2.5 compares with BASE e5301165 (the branch point of ``T-0208-a2-shared-check``). Its rows guard a negative
  criterion, so they are green before the implementation by construction; their power is the T-0207 BASE harness
  they reuse (whose captures include every body, record, machine event and HUD message).

**Mutation the reviewer must kill:** arming on ``browser_actions > 0`` alone (the current-image clause dropped) turns
``test_shared_step_image_never_checked`` red (a check fires with the shared image in its body, beside the fresh
browser frame).

**Review 2 mutation:** reading the prompt's oldest image instead of its newest in ``_check_armed`` (the first
``ImagePart.phash`` for ``self._sent_identity(prompt)``) turns the two-turn rows red
(``test_shared_turn_then_browser_turn_checks``, ``test_browser_turn_then_shared_turn_stuck_not_checked``,
``test_same_frame_shared_turns_check_body_has_no_shared_bytes``); every single-turn row stays green under it.
**R-A2 too wide:** a ``wait`` that disarms the whole turn turns ``test_r_a2_wait_then_scroll_arms`` red.

Q-1 (where the sent current-image identity is available relative to ``_check_armed``): in ``Session._run_step`` the
step's final ``prompt`` (after the swap, the label, the T-0201 reminder and the overview) is bound before
``held = [] if self._check_armed(turn) else None``, so the clause can read ``self._is_shared(
self._sent_identity(prompt))`` before the stream; no move to the fire point is needed (test-author's reading of the
code at e5301165, for the implementer to confirm).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``live`` (1280x800
browser frames, ReGround off, no live feedback unless named), loop on (8 steps), n = 1, deterministic.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from local_astra.toolcall.completion_check import check_prompt
from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0207_session_check as t7
from tests.unit._live_browser_support import b64, builder_url, image_urls, jpeg_frame
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_live import _wire, last_user
from tests.unit.test_t0177_shared_frame import (
    SH,
    SHARED_LABEL_SCREEN,
    VisionBrowser,
    select,
    vision_rig,
)
from tests.unit.test_t0177_step_screens import (
    SHARED_FRAME_REASON,
    _OnceBlindBrowser,
    current_images,
    text_after_current,
)

REPO = Path(__file__).resolve().parents[2]
#: BASE: main at e5301165 (``T-0208: James rules H-T208-1 yes``), where group A2 branched. A literal (tests.md).
BASE_SHA = "e53011658c54b4d212ea3bbe52d7af8e94102d06"
#: The implementation tip, set by the test-author after the implementation commit (A2.7's range end).
TIP_IMPL: str | None = "b611e5de286586fa96ad35f31d97478e0e758668"
#: A2.7, verbatim
FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "local_astra/toolcall",
    "local_astra/computer",
    "local_astra/conductor/prompt_builder.py",
    "hud",
    "local_astra/cli.py",
    "local_astra/config.py",
    "tests/unit/test_turn_invariants.py",
)

KW = t7.KW
ABSENT = t7.ABSENT
S1 = t7.S1
S2 = t7.S2
SCROLL = t7.SCROLL
CLAIM = t7.CLAIM
CLAIM_STEP = t7.CLAIM_STEP
CHECK_SAYS = t7.CHECK_SAYS
NO_CALL_CHECK = t7.NO_CALL_CHECK

#: browser frames of this module (real JPEGs, distinct from T-0207's and T-0177's)
B0 = jpeg_frame("t0208-a2-onset")
B1 = jpeg_frame("t0208-a2-after-1")
B2 = jpeg_frame("t0208-a2-after-2")
BR = jpeg_frame("t0208-a2-fresh")
BROWSER_FRAMES = (B0, B1, B2, BR)
SHARED_FRAMES = (SH,)
#: step 1: words, then an executed scroll
SCROLL_STEP = (answer(S1), tool(SCROLL))


# --------------------------------------------------------------------------------------------- the rig


def _session_kwargs(on: Any) -> dict[str, Any]:
    if on is ABSENT:
        return {}
    t7.require_keyword()
    return {KW: on}


async def shared_turn(
    tmp: Path,
    browser: Any,
    *,
    on: Any = True,
    script: Any = SCROLL_STEP,
    steps: Any = (CLAIM_STEP,),
    checks: Any = (NO_CALL_CHECK,),
    hud_frames: Any = None,
    share: bool = True,
) -> Any:
    """One live turn in ``vision_rig``: ``select shared`` over the wire (``share``), then step 1 ``script`` and
    ``steps``; the check route scripted with ``checks``."""
    tmp.mkdir(parents=True, exist_ok=True)
    async with vision_rig(
        tmp,
        browser,
        script=list(script),
        session_kwargs=_session_kwargs(on),
        hud_frames=hud_frames,
    ) as rig:
        if share:
            await select(rig, "shared")
        await t7.drive(rig, *steps, checks=checks)
    return rig


def swap_browser() -> Any:
    """A2.2: step 1 shared; the scroll's after-frame ``B1`` swaps in at step 2 (T-0177 B1); the check's fresh frame
    is ``BR``."""
    return VisionBrowser([BR], post_frames=[B1, B2])


def _scrolled_elsewhere(fb: Any, inv: Any) -> Any:
    """T-0177 B-R3's handler: the action executes, but its after-frame is not the browser's latest."""
    out = fb.perform(inv)
    return out.model_copy(update={"resolved": {**out.resolved, "frame_after": "not-the-latest"}})


def stuck_browser() -> Any:
    """A2.3 (T-0177 B2's no-fresh-frame path): the scroll executes, its after-frame is not the latest and the one
    fresh ``frame()`` after it fails, so step 2 keeps the shared image; ``frame()`` works afterwards (``BR``)."""
    browser = _OnceBlindBrowser([BR], post_frames=[B1, B2])
    browser.handlers["t1-call"] = _scrolled_elsewhere
    return browser


# --------------------------------------------------------------------------------------------- reading the rig


def _bytes_of(url: str) -> bytes:
    assert url.startswith("data:") and "," in url, url[:60]
    return base64.b64decode(url.split(",", 1)[1])


def image_kind(url: str) -> str:
    """``shared`` or ``browser`` for a request image, by its bytes (an oracle independent of ``_is_shared``)."""
    data = _bytes_of(url)
    if any(f.jpeg == data for f in SHARED_FRAMES):
        return "shared"
    if any(f.jpeg == data for f in BROWSER_FRAMES):
        return "browser"
    raise AssertionError(
        f"an image of unknown kind in a request body ({len(data)} bytes); the scan cannot classify it"
    )


def all_bodies(rig: Any) -> list[tuple[str, dict[str, Any]]]:
    """Every request the session sent (prefills and generates: step 1, steps, re-asks, resends, checks), as wire
    bodies."""
    out = [(f"prefill {rid}", _wire(p)) for rid, p in rig.llm.prefills]
    out += [(rid, _wire(p)) for rid, p, _ in rig.llm.generates]
    return out


_TURN_OF_RID = re.compile(r"-(t\d+)-(?:gen(?:-s\d+)?|prefill)-\d+$")


def turn_of(rid: str) -> str:
    """The turn (``t1``, ``t2``, ...) of a request id (``...-t2-gen-s2-1``; a prefill's is ``prefill <id>``)."""
    found = _TURN_OF_RID.search(rid)
    assert found is not None, f"the scan cannot place request {rid!r} in a turn"
    return found.group(1)


def turn_start(rig: Any, turn: str) -> int:
    """The index of ``turn``'s own user message in its step-1 body: every message before it is history (an earlier
    turn's), every message from it on is the current turn's. Read off the body the session sent at step 1 (its last
    user message is the turn's question, T-0111g); the turn's later requests extend that prefix (T-0114), and a check
    replaces only the final user message (T-0207 A1)."""
    messages = gen_body(rig, turn)["messages"]
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    assert users, f"premise: {turn}'s step-1 body has a user message"
    return users[-1]


def current_turn_urls(rig: Any, rid: str, body: dict[str, Any]) -> list[str]:
    """R-A1: the images of the current turn in ``body`` (its user message(s) and the steps' own images), history
    excluded. Premise: the body's history prefix is the step-1 body's (same messages, same order)."""
    turn = turn_of(rid)
    start = turn_start(rig, turn)
    head = gen_body(rig, turn)["messages"][:start]
    assert body["messages"][:start] == head, (
        f"premise ({rid}): the body's history is {turn}'s step-1 history, so its current-turn part starts at {start}"
    )
    return image_urls({"messages": body["messages"][start:]})


def assert_one_source(rig: Any, what: str, *, turns: int = 1) -> None:
    """A2.3 "no body holds both a shared-frame image and an agent-browser image", read under R-A1 over the current
    turn's images: each body's history messages (an earlier turn's) are excluded. A single-turn row keeps its
    premise (``turns=1``: one turn line, so every image is a current-turn image, and the whole body is scanned,
    as before); a multi-turn row names its turn count and is scanned from each turn's own user message on."""
    assert len(rig.lines()) == turns, (
        f"premise ({what}): a {turns}-turn session; turn lines {len(rig.lines())}"
    )
    bodies = all_bodies(rig)
    assert bodies, f"premise ({what}): the session sent requests"
    for rid, body in bodies:
        urls = image_urls(body) if turns == 1 else current_turn_urls(rig, rid, body)
        kinds = {image_kind(u) for u in urls}
        assert kinds != {"shared", "browser"}, (
            f"T-0208 A2.3 ({what}): request {rid} holds both a shared-frame image and an agent-browser image "
            f"in its current turn (images {[image_kind(u) for u in urls]})"
        )


def step_body(rig: Any, k: int) -> dict[str, Any]:
    return _wire(tsl.step_request(rig, k))


def gen_body(rig: Any, turn: str = "t1") -> dict[str, Any]:
    return _wire(tsl.sent(rig, f"{turn}-gen"))


def _no_check_and_voiced(rig: Any, what: str) -> None:
    t7.assert_no_check(rig, what)
    assert t7.notes(rig, t7.NOTE_ASKED) == [] and t7.outcomes(rig) == [], (
        f"T-0208 ({what}): no completion check note; notes {t7.notes(rig)}"
    )
    assert t7.turn_key(rig) == {"on": True, "asked": 0, "outcome": None}, (
        f"T-0207 A2.11 / T-0208 ({what}): {t7.turn_key(rig)}"
    )
    assert rig.spoken_sources().endswith(CLAIM), (
        f"T-0208 ({what}): the draft is voiced as at BASE; voiced {rig.spoken_sources()!r}"
    )


# =====================================================================================================
# Premises
# =====================================================================================================


def test_frames_are_distinct_and_classifiable() -> None:
    """Premise of the scan: every frame of the module has its own bytes, the browser frames are real JPEGs that are
    not blank (T-0200), and the shared payload is none of them."""
    from local_astra.computer.executor import blank_frame

    jpegs = [f.jpeg for f in (*BROWSER_FRAMES, *SHARED_FRAMES)]
    assert len(set(jpegs)) == len(jpegs)
    assert not any(blank_frame(f.jpeg) for f in BROWSER_FRAMES)
    assert image_kind(builder_url(SH)) == "shared" and image_kind(builder_url(B1)) == "browser"


# =====================================================================================================
# A2.2: shared start, browser finish
# =====================================================================================================


async def test_shared_start_browser_finish_checks_once(tmp_path: Path) -> None:
    """A2.1/A2.2: under ``shared`` step 1 carries the shared image, an executed ``scroll`` swaps in the browser frame
    ``B1`` at step 2, step 2 answers with no call. The check fires once under ``t1-gen-s2`` with T-0207 A2.4's shape
    (``phase: "completion_check"``, its record keys, ``fresh_frame``); its prompt is ``check_prompt(<step 2 as sent>,
    <step text>, <fresh frame BR>, report=False)``; its body holds zero shared bytes and exactly one image more than
    step 2's body (the fresh browser frame, last); the draft is never voiced, stored or published, the check's
    answer is voiced; no body of the turn holds both kinds of image."""
    rig = await shared_turn(tmp_path, swap_browser())
    assert current_images(gen_body(rig)) == [builder_url(SH)], "premise: step 1 is the shared image"
    assert t7.results(rig, t7.call_id(1)) == ["executed"], "premise: the scroll executed"
    s2 = step_body(rig, 2)
    assert current_images(s2) == [builder_url(B1)], "premise: step 2 shows the browser frame (B1)"
    engine, prompt = t7.assert_one_check(rig, t7.step_id(2), "shared start, browser finish")
    want = check_prompt(tsl.step_request(rig, 2), t7.step_text(CLAIM), t7.held(BR), report=False)
    assert prompt == want, (
        "T-0208 A2.2 / T-0207 A2.4: the check prompt is check_prompt(<step 2 as sent>, <step text>, <fresh frame>)"
    )
    (req,) = t7.check_requests(rig)
    assert set(req) == t7.REQUEST_KEYS and req["phase"] == t7.PHASE, sorted(req)
    assert (req["request_id"], req["attempt_id"], req["step"]) == (t7.step_id(2), engine, 2), req
    assert req["fresh_frame"]["phash"] in (BR.phash, t7.identity(BR)), req["fresh_frame"]
    (body,) = rig.llm.check_bodies
    text = json.dumps(body)
    assert b64(SH.jpeg) not in text, "T-0208 A2.2: the check body holds zero shared bytes"
    imgs, step_imgs = image_urls(body), image_urls(s2)
    assert len(imgs) == len(step_imgs) + 1, (len(imgs), len(step_imgs))
    assert _bytes_of(imgs[-1]) == BR.jpeg, (
        "T-0208 A2.2: the one extra image is the fresh browser frame"
    )
    assert t7.outcomes(rig) == ["no_call"], t7.notes(rig)
    assert t7.turn_key(rig) == {"on": True, "asked": 1, "outcome": "no_call"}, t7.turn_key(rig)
    voiced = rig.spoken_sources()
    assert CHECK_SAYS in voiced and CLAIM not in voiced, voiced
    for where, blob in (
        ("astra.spoken", t7.spoken_published(rig)),
        ("history", t7.history_json(rig)),
        ("turn record", t7.record_json(rig)),
        ("memory", t7.memory_json(rig)),
    ):
        assert CLAIM not in blob, f"T-0207 A2.5 / T-0208 A2.2: the draft is never in {where}"
    assert_one_source(rig, "A2.2")


async def test_shared_check_prefix_property(tmp_path: Path) -> None:
    """A2.2 "the T-0207 A1.3 prefix property holds against step 2's rendered body": rendered through the pinned Gemma
    4 template, step 2's prompt as sent is a byte prefix of the check prompt as sent, the system turn is identical and
    starts with ``<|think|>``, and the check has exactly one more ``<|image|>``, inside its final user turn."""
    from tests.contract.test_t0172_reask_prefix import _oracle, render, system_turn

    tpl = _oracle()
    rig = await shared_turn(tmp_path, swap_browser())
    (_, prompt) = t7.assert_one_check(rig, t7.step_id(2), "prefix")
    a, b = render(tpl, tsl.step_request(rig, 2)), render(tpl, prompt)
    assert b.startswith(a), (
        "T-0208 A2.2 / T-0207 A1.3: the step's rendering is a prefix of the check's"
    )
    assert system_turn(a) == system_turn(b)
    assert system_turn(b).startswith("<bos><|turn>system\n<|think|>"), system_turn(b)[:60]
    assert b.count("<|image|>") == a.count("<|image|>") + 1
    assert b.rindex("<|image|>") > b.rindex("<|turn>user\n"), (
        "the extra image is in the final user turn"
    )


# =====================================================================================================
# A2.3: one source per request (adversarial)
# =====================================================================================================


async def test_shared_step_image_never_checked(tmp_path: Path) -> None:
    """A2.3 (adversarial, "the model sees one source"; the mutation row): under ``shared`` the step-1 scroll executes
    but no fresh browser frame can be taken (T-0177 B2's path), so step 2 still carries the shared image and its
    label; step 2 answers with no call. No check fires: no check request, no ``completion check asked`` note, no
    further ``frame()`` (the one call is the failed post-action attempt; ``frame()`` would work now, so a session that
    armed would have sent a check), ``asked: 0``; the draft is voiced, and the machine sequence and voiced text equal
    the keyword-absent run's (BASE's behaviour). R-A1: no body of the turn holds both a shared and a browser image.

    Under the mutation "arm on ``browser_actions > 0`` alone" this row is red: the check fires and its body carries
    the shared image beside the fresh browser frame."""
    rig = await shared_turn(tmp_path / "on", stuck_browser())
    assert t7.results(rig, t7.call_id(1)) == ["executed"], "premise: the scroll executed"
    s2 = step_body(rig, 2)
    assert current_images(s2) == [builder_url(SH)], "premise: step 2 still carries the shared image"
    assert text_after_current(s2) == SHARED_LABEL_SCREEN, "premise: with its shared label"
    _no_check_and_voiced(rig, "A2.3 shared step image")
    assert t7.frame_calls(rig) == 1, (
        f"T-0208 A2.3: no fresh frame for a check (only the failed post-action attempt); {rig.browser.calls}"
    )
    assert_one_source(rig, "A2.3")
    base = await shared_turn(tmp_path / "absent", stuck_browser(), on=ABSENT)
    assert t7.machine_seq(rig) == t7.machine_seq(base), (
        "T-0208 A2.3: the turn reaches the machine as with the keyword absent"
    )
    assert rig.spoken_sources() == base.spoken_sources()


# =====================================================================================================
# Review 2: two-turn rows (the step's newest image, never the oldest; R-A1 history images kept)
# =====================================================================================================

T1_SAYS = "It is a small shop. "
T2_QUESTION = "now add it to the cart"


async def next_turn(
    rig: Any,
    turn: str,
    *,
    script: Any,
    steps: Any = (CLAIM_STEP,),
    checks: Any = (NO_CALL_CHECK,),
    question: str = T2_QUESTION,
) -> None:
    """Turn ``turn`` (``t2``) of a running rig: step 1 ``script``, then ``steps``; checks scripted; waits until the
    turn's line is written."""
    n = int(turn[1:])
    rig.llm.script = list(script)
    rig.llm.script_steps(*steps)
    rig.llm.script_checks(*checks)
    await rig.ask(question)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
        f"{turn} never ended",
        limit_ms=40_000,
    )
    await rig.advance(300)


async def test_shared_turn_then_browser_turn_checks(tmp_path: Path) -> None:
    """Review 2 (A2.1 "the step request's current image", the newest image of the prompt the step sent): turn 1 runs
    under ``shared`` and answers, so ``SH`` stays in history (H-T177-4). After ``select browser``, turn 2's step 1
    (browser frame ``B0``) scrolls, step 2 (``B1``) claims with no call: the check fires once under ``t2-gen-s2``
    with the fresh frame ``BR``, ``asked: 1``, the check's answer voiced. The session's oldest image is ``SH``, so a
    session that read the oldest image instead of the newest would leave this turn unchecked (the mutant this row
    kills). R-A1: no body holds both kinds of image in its current turn."""
    browser = VisionBrowser([B0, BR], post_frames=[B1, B2])
    async with vision_rig(
        tmp_path, browser, script=[answer(T1_SAYS)], session_kwargs=_session_kwargs(True)
    ) as rig:
        await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done()
        await rig.advance(300)
        await select(rig, "browser")
        await next_turn(rig, "t2", script=SCROLL_STEP)
    assert current_images(gen_body(rig, "t1")) == [builder_url(SH)], "premise: turn 1 is shared"
    t2 = gen_body(rig, "t2")
    assert current_images(t2) == [builder_url(B0)], "premise: turn 2's step 1 is the browser frame"
    assert image_urls(t2)[0] == builder_url(SH), (
        "premise: turn 1's shared image is in history, the oldest image of every turn-2 request"
    )
    assert t7.results(rig, t7.call_id(1, "t2")) == ["executed"], "premise: the scroll executed"
    s2 = _wire(tsl.step_request(rig, 2, "t2"))
    assert current_images(s2) == [builder_url(B1)], "premise: step 2 shows the browser frame B1"
    assert image_urls(s2)[0] == builder_url(SH), "premise: SH is still the oldest image at step 2"
    t7.assert_one_check(rig, t7.step_id(2, "t2"), "shared turn, then browser turn")
    (body,) = rig.llm.check_bodies
    assert current_images(body) == [builder_url(BR)], "the check's fresh frame is BR"
    assert t7.turn_key(rig, "t1") == {"on": True, "asked": 0, "outcome": None}, t7.turn_key(
        rig, "t1"
    )
    assert t7.turn_key(rig, "t2") == {"on": True, "asked": 1, "outcome": "no_call"}, (
        f"T-0208 A2.1 (review 2): {t7.turn_key(rig, 't2')}"
    )
    voiced = rig.spoken_sources()
    assert CHECK_SAYS in voiced and CLAIM not in voiced, voiced
    assert_one_source(rig, "shared then browser", turns=2)


class _BlindOnDemandBrowser(VisionBrowser):
    """``frame()`` raises once when ``blind_next`` is set (a capture that fails once, then works), and works
    otherwise."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.blind_next = False

    async def frame(self) -> Any:
        if self.blind_next:
            self.blind_next = False
            self.calls.append(("frame",))
            raise RuntimeError("the fake browser could not take a frame (once)")
        return await super().frame()


async def test_browser_turn_then_shared_turn_stuck_not_checked(tmp_path: Path) -> None:
    """Review 2, the reverse: turn 1 runs under ``browser`` and answers (its frame ``B0`` stays in history); after
    ``select shared``, turn 2's step 1 (``SH``) scrolls, the scroll executes but no fresh browser frame can be taken
    (A2.3's path: its after-frame is not the latest and the one fresh ``frame()`` fails), so step 2's newest image is
    still ``SH``; step 2 claims with no call. No check: ``asked: 0``, the draft voiced. The oldest image of every
    turn-2 request is the browser frame ``B0``, so a session that read the oldest image would arm and check.
    ``frame()`` works again after the failure, so the row's "no check" is not met by a frame failure."""
    browser = _BlindOnDemandBrowser([B0, BR], post_frames=[B1, B2])
    browser.handlers["t2-call"] = _scrolled_elsewhere
    async with vision_rig(
        tmp_path, browser, script=[answer(T1_SAYS)], session_kwargs=_session_kwargs(True)
    ) as rig:
        await rig.ask("what is on the page")
        await rig.turn_done()
        await rig.advance(300)
        await select(rig, "shared")
        browser.blind_next = True
        await next_turn(rig, "t2", script=SCROLL_STEP)
    assert current_images(gen_body(rig, "t1")) == [builder_url(B0)], "premise: turn 1 is browser"
    t2 = gen_body(rig, "t2")
    assert current_images(t2) == [builder_url(SH)], "premise: turn 2's step 1 is the shared image"
    assert image_urls(t2)[0] == builder_url(B0), "premise: B0 is the oldest image of turn 2"
    assert t7.results(rig, t7.call_id(1, "t2")) == ["executed"], "premise: the scroll executed"
    s2 = _wire(tsl.step_request(rig, 2, "t2"))
    assert current_images(s2) == [builder_url(SH)], "premise: step 2 still carries the shared image"
    assert image_urls(s2)[0] == builder_url(B0), "premise: B0 is still the oldest image at step 2"
    assert not browser.blind_next, "premise: the one failing frame() was consumed in turn 2"
    t7.assert_no_check(rig, "browser then shared, stuck")
    assert t7.notes(rig, t7.NOTE_ASKED) == [], t7.notes(rig)
    assert t7.turn_key(rig, "t2") == {"on": True, "asked": 0, "outcome": None}, (
        f"T-0208 A2.3 (review 2, reverse): {t7.turn_key(rig, 't2')}"
    )
    assert rig.spoken_sources().endswith(CLAIM), rig.spoken_sources()
    assert_one_source(rig, "browser then shared, stuck", turns=2)


async def test_same_frame_shared_turns_check_body_has_no_shared_bytes(tmp_path: Path) -> None:
    """Reviewer 1's note (R-A1, A2.2 "zero shared bytes" read over the check's own image): turn 1 is shared over
    ``SH``; turn 2 is shared over the same bytes, so the builder does not attach ``SH`` twice: it rides on turn 1's
    history message and turn 2's step-1 user message carries no image. Turn 2's scroll executes and swaps in ``B1``
    at step 2, step 2 claims with no call, the check fires once (``asked: 1``). The check's final user message holds
    exactly the fresh browser frame ``BR``, with no shared bytes; ``SH`` appears only in history."""
    browser = VisionBrowser([BR], post_frames=[B1, B2])
    async with vision_rig(
        tmp_path, browser, script=[answer(T1_SAYS)], session_kwargs=_session_kwargs(True)
    ) as rig:
        await select(rig, "shared")
        await rig.ask("what is on my screen")
        await rig.turn_done()
        await rig.advance(300)
        await next_turn(rig, "t2", script=SCROLL_STEP)
    t2 = gen_body(rig, "t2")
    assert current_images(t2) == [] and image_urls(t2) == [builder_url(SH)], (
        "premise: turn 2's shared frame rides on turn 1's history image"
    )
    assert t7.results(rig, t7.call_id(1, "t2")) == ["executed"], "premise: the scroll executed"
    s2 = _wire(tsl.step_request(rig, 2, "t2"))
    assert current_images(s2) == [builder_url(B1)], "premise: step 2 shows the browser frame B1"
    t7.assert_one_check(rig, t7.step_id(2, "t2"), "same-frame shared turns")
    (body,) = rig.llm.check_bodies
    last = last_user(body)
    assert current_images(body) == [builder_url(BR)], (
        f"T-0208 A2.2 / R-A1: the check's final user message holds only the fresh browser frame; images "
        f"{[image_kind(u) for u in current_images(body)]}"
    )
    assert b64(SH.jpeg) not in json.dumps(last), (
        "T-0208 A2.2 / R-A1: no shared bytes in the check's final user message"
    )
    assert [image_kind(u) for u in image_urls(body)].count("shared") == 1, (
        "R-A1: SH appears once, in history only"
    )
    assert t7.turn_key(rig, "t2") == {"on": True, "asked": 1, "outcome": "no_call"}, t7.turn_key(
        rig, "t2"
    )
    assert_one_source(rig, "same-frame shared turns", turns=2)


# =====================================================================================================
# R-A2: a wait alone does not arm on a shared-start turn
# =====================================================================================================

WAIT = "<|tool_call>call:wait{}<tool_call|>"


def _waited(fb: Any, inv: Any) -> Any:
    """The executor's ``wait`` (T-0202 R1): an executed ``waited 1 s`` with its after-frame."""
    out = fb.perform(inv)
    return out.model_copy(update={"action": "waited 1 s"})


async def _wait_turn(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, steps: Any, source: str = "shared"
) -> Any:
    """One turn in ``vision_rig`` (``source`` selected; ``shared`` by default) with the T-0202 wait kit
    (``cli.load_tool_kit(..., wait_tool=True)`` over the ``VisionBrowser``) and the check on: step 1 calls ``wait``,
    then ``steps``. Under ``shared`` step 1 shows the shared image ``SH`` and ``BR`` is the check's fresh frame; under
    ``browser`` step 1 takes the browser frame ``B0`` and ``BR`` stays the check's fresh frame."""
    from tests.unit.test_t0202_session_wait import wait_kit

    tmp.mkdir(parents=True, exist_ok=True)
    first = SH if source == "shared" else B0
    browser = VisionBrowser([BR] if source == "shared" else [B0, BR], post_frames=[B1, B2])
    browser.handlers["t1-call"] = _waited
    kit = wait_kit(browser, tmp, monkeypatch)
    async with vision_rig(
        tmp, browser, script=[answer(S1), tool(WAIT)], session_kwargs=_session_kwargs(True), kit=kit
    ) as rig:
        await select(rig, source)
        await t7.drive(rig, *steps, checks=(NO_CALL_CHECK,))
    assert current_images(gen_body(rig)) == [builder_url(first)], (
        f"premise: step 1 is the {source} image"
    )
    assert [i.name for i in browser.invocations][:1] == ["wait"], browser.invocations
    assert t7.results(rig, t7.call_id(1)) == ["executed"], "premise: the wait executed"
    assert current_images(step_body(rig, 2)) == [builder_url(B1)], (
        "premise: step 2 shows the browser frame (the wait's after-frame)"
    )
    return rig


async def test_r_a2_wait_alone_does_not_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-A2 (orchestrator ruling, T-0208 Status): on a turn whose step 1 was a shared frame, an executed ``wait``
    alone does not arm the check. Shared start, step 1 ``wait`` (executed), step 2 shows the browser frame ``B1`` and
    claims with no call: no check, no note, ``asked: 0``, the draft voiced as at BASE."""
    rig = await _wait_turn(tmp_path, monkeypatch, (CLAIM_STEP,))
    _no_check_and_voiced(rig, "R-A2 wait only")
    assert_one_source(rig, "R-A2 wait only")


async def test_r_a2_wait_then_scroll_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R-A2 (too wide): any other executed browser action arms as A2.1. Shared start, step 1 ``wait``, step 2 (``B1``)
    an executed ``scroll``, step 3 (``B2``) claims with no call: the check fires once under ``t1-gen-s3``,
    ``asked: 1``, the check's answer voiced."""
    rig = await _wait_turn(tmp_path, monkeypatch, ((answer(S2), tool(SCROLL)), CLAIM_STEP))
    assert t7.results(rig, t7.call_id(2)) == ["executed"], "premise: the scroll executed"
    assert current_images(step_body(rig, 3)) == [builder_url(B2)], "premise: step 3 shows B2"
    t7.assert_one_check(rig, t7.step_id(3), "R-A2 wait then scroll")
    assert t7.turn_key(rig) == {"on": True, "asked": 1, "outcome": "no_call"}, t7.turn_key(rig)
    voiced = rig.spoken_sources()
    assert CHECK_SAYS in voiced and CLAIM not in voiced, voiced
    assert_one_source(rig, "R-A2 wait then scroll")


async def test_r_a2_browser_wait_only_still_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-A2's scope (too wide): the ruling subtracts a ``wait`` only on a turn whose step 1 was a shared frame. On a
    browser-source turn an executed ``wait`` is a browser action and arms the check as T-0207 A2.1 says. Browser
    source, check on, step 1 ``wait`` (executed, after-frame ``B1``), step 2 (``B1``) claims with no call: the check
    fires once under ``t1-gen-s2``, ``asked: 1``, the check's answer voiced. Kills mutant N1 (waits subtracted on
    every turn)."""
    rig = await _wait_turn(tmp_path, monkeypatch, (CLAIM_STEP,), source="browser")
    t7.assert_one_check(rig, t7.step_id(2), "R-A2 scope: browser wait only")
    assert len(rig.llm.check_bodies) == 1, "T-0207 A2.1: exactly one check request"
    assert current_images(rig.llm.check_bodies[0]) == [builder_url(BR)], (
        "T-0207 A2.3: the check carries the one fresh browser frame"
    )
    assert t7.turn_key(rig) == {"on": True, "asked": 1, "outcome": "no_call"}, t7.turn_key(rig)
    voiced = rig.spoken_sources()
    assert CHECK_SAYS in voiced and CLAIM not in voiced, voiced
    assert_one_source(rig, "R-A2 scope: browser wait only")


# =====================================================================================================
# A2.4: not when
# =====================================================================================================


def _blind_browser() -> Any:
    return VisionBrowser([], post_frames=[B1, B2])


NOT_WHEN_SHARED: dict[str, dict[str, Any]] = {
    # only a refused shared-image click (T-0177 B2), then an answer: step 2 shows a fresh browser frame (BR), so
    # only "an executed browser action" blocks
    "refused_click": dict(script=(answer(S1), tool(VALID_CLICK)), browser=swap_browser),
    # only a google_search, then an answer: the shared image stays
    "search_only": dict(script=(answer(S1), tool(t7.SEARCH_CALL)), browser=swap_browser),
    # a blind shared turn (no HUD frame), an executed scroll, an answer: step 2 stays blind, so only "non-blind" blocks
    "blind": dict(script=(answer(S1), tool(SCROLL)), browser=_blind_browser, hud_frames=[]),
}


@pytest.mark.parametrize("name", list(NOT_WHEN_SHARED))
async def test_not_when_shared(tmp_path: Path, name: str) -> None:
    """A2.4 (too wide): a shared turn whose only call was a refused shared-image click, or a ``google_search``, or a
    blind shared turn with an executed scroll, ends with no check: no request, no note, ``asked: 0``, the draft
    voiced; no body holds both kinds of image. Power: A2.2's row (an executed scroll on a non-blind shared turn)."""
    case = NOT_WHEN_SHARED[name]
    rig = await shared_turn(
        tmp_path,
        case["browser"](),
        script=case["script"],
        hud_frames=case.get("hud_frames"),
    )
    s2 = step_body(rig, 2)
    if name == "refused_click":
        card = rig.tool_calls(t7.call_id(1))[-1]
        assert (card.status, card.reason) == ("refused", SHARED_FRAME_REASON), card
        assert current_images(s2) == [builder_url(BR)], (
            "premise: step 2 shows a fresh browser frame, so the image clause alone would arm"
        )
    elif name == "search_only":
        assert current_images(s2) == [builder_url(SH)], "premise: the shared image stays"
    else:
        assert image_urls(gen_body(rig)) == [], "premise: step 1 is blind"
        assert t7.results(rig, t7.call_id(1)) == ["executed"], "premise: the scroll executed"
        assert image_urls(s2) == [], (
            "premise: the turn stays blind at step 2 (no shared image either), so only the non-blind condition "
            "blocks"
        )
    _no_check_and_voiced(rig, f"A2.4 {name}")
    assert_one_source(rig, f"A2.4 {name}")


# =====================================================================================================
# A2.5: off and browser unchanged (BASE e5301165, subprocess)
# =====================================================================================================


async def _scenarios(tmp: Path) -> dict[str, Any]:
    """A2.5's scenarios: ``completion_check=False`` under ``shared`` (A2.2's swap and A2.3's stuck path) and under
    ``browser`` (T-0207's claim turn), and the check on under ``browser`` (T-0207's claim turn)."""
    out: dict[str, Any] = {}
    rig = await shared_turn(tmp / "off_shared_swap", swap_browser(), on=False)
    out["off_shared_swap"] = t7._dump(rig, tmp / "off_shared_swap")
    rig = await shared_turn(tmp / "off_shared_stuck", stuck_browser(), on=False)
    out["off_shared_stuck"] = t7._dump(rig, tmp / "off_shared_stuck")
    rig = await t7.claim_turn(tmp / "off_browser", False)
    out["off_browser"] = t7._dump(rig, tmp / "off_browser")
    rig = await t7.claim_turn(tmp / "on_browser", True)
    out["on_browser"] = t7._dump(rig, tmp / "on_browser")
    out["on_browser"]["check_bodies"] = t7._norm(
        rig.llm.check_bodies, rig.session.session_id, str(tmp / "on_browser")
    )
    return out


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0208_shared_check as m
out = pathlib.Path(sys.argv[1])
data = asyncio.run(m._scenarios(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "data": data}, default=str))
"""


@pytest.fixture(scope="module")
def base_capture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A2.5's scenarios at BASE, in a subprocess over ``git archive`` of BASE, origin-proved."""
    root = tmp_path_factory.mktemp("t0208_base")
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


def _compare(base: dict[str, Any], now: dict[str, Any], name: str) -> None:
    for part in sorted(base):
        assert now[part] == base[part], (
            f"T-0208 A2.5 ({name}): {part} differs from BASE e5301165 "
            f"({len(base[part])} vs {len(now[part])} items)"
        )


async def test_off_byte_identical_both_sources(
    tmp_path: Path, base_capture: dict[str, Any]
) -> None:
    """A2.5: with ``completion_check=False`` every request body (prefill, generate, the body as sent), ``llm.request``
    / ``llm.done`` / ``note`` / ``tool.call`` record, HUD message, machine event with its command list and turn-line
    extras equals BASE's, under ``shared`` (the swap and the stuck path) and under ``browser``. Premises: BASE's
    shared runs sent the shared image at step 1 and executed the scroll; no check anywhere."""
    now = await _scenarios(tmp_path)
    for name in ("off_shared_swap", "off_shared_stuck", "off_browser"):
        _compare(base_capture[name], json.loads(json.dumps(now[name], default=str)), name)
        assert t7.PHASE not in json.dumps(base_capture[name]["records"]), name
    assert b64(SH.jpeg) in json.dumps(base_capture["off_shared_swap"]["bodies"][0]), (
        "premise: step 1 carried the shared image at BASE"
    )


async def test_browser_source_unchanged(tmp_path: Path, base_capture: dict[str, Any]) -> None:
    """A2.5 "under browser with the check on ... unchanged": T-0207's claim turn under ``browser`` with the check on
    sends the same check body, and the same records, HUD messages and machine sequence, as at BASE. Premise: BASE
    sent exactly one check."""
    now = json.loads(json.dumps((await _scenarios(tmp_path))["on_browser"], default=str))
    base = base_capture["on_browser"]
    assert len(base["check_bodies"]) == 1, "premise: BASE checked the browser claim turn once"
    _compare(base, now, "on_browser")


# =====================================================================================================
# A2.7: frozen paths
# =====================================================================================================


def test_frozen_paths_over_range() -> None:
    """A2.7: ``turn_machine.py``, ``contracts/**``, ``toolcall/**``, ``computer/**``, ``prompt_builder.py``,
    ``hud/**``, ``cli.py``, ``config.py`` and ``tests/unit/test_turn_invariants.py`` are byte-identical over
    ``BASE_SHA..TIP_IMPL`` (two literals), and docs/02 §4 is unedited. Red until the test-author sets ``TIP_IMPL``
    after the implementation commit."""
    assert TIP_IMPL is not None, (
        "T-0208 A2.7: TIP_IMPL is set by the test-author after the implementation commit"
    )
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:8]} is not an ancestor of {TIP_IMPL[:8]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0208 A2.7: frozen paths differ in {BASE_SHA[:8]}..{TIP_IMPL[:8]}: {changed}"
    )
    from tests.unit import test_t0107a_docs as t7d

    s4_base = t7d.section(t7._doc02_at(BASE_SHA), "## 4. ")
    s4_now = t7d.section(t7._doc02_at(TIP_IMPL), "## 4. ")
    assert s4_base and s4_now == s4_base, "T-0208 A2.7: the §4 invariants are unedited"


def test_frozen_paths_power_premise() -> None:
    """A2.7 power (green at BASE): the predicate sees a change a known task made to a frozen path (T-0172's
    ``announced_call.py``, ``3e4d45b7`` to BASE), so the guard above cannot pass by seeing nothing."""
    gr.require_commits(BASE_SHA, "3e4d45b7")
    assert "local_astra/toolcall/announced_call.py" in gr.changed("3e4d45b7", BASE_SHA, *FROZEN)
