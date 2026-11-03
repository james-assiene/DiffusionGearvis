"""T-0231 group B: a ``tool_step`` request that comes back with zero commits right after an executed browser action is
re-asked once (docs/tasks/T-0231.md B1-B5).

**The fake.** ``FakeLlm.script_steps`` with an empty list is a step that yields zero commits: no thought, no answer, no
call, and no engine error (the T-0171 retryable kind is ``engine_empty_error``, raised, and is a different path). The
re-ask resends the step's prompt bytes, so the fake routes it as one more step request (``is_step_prompt``) and it
consumes the next step script: ``steps=([], [answer(S2), tool(CLICK2)])`` is "step 2 empty, its re-ask a call". No
fake change was needed.

**Rigs.** T-0207's live rig (``test_t0207_session_check.claim_turn``: T-0134's ``feedback_rig``, one
``FakeLiveBrowser`` as executor and live browser, real JPEG frames, loop on 8 steps), built with
``Session(..., completion_check=True)`` where a row needs the check (B2) and without the keyword elsewhere. The search
rows use T-0207's ``_search_kit`` (``FakeSearch``, so the search *executes*); the wait row the T-0202 kit
(``cli.load_tool_kit(..., wait_tool=True)``) over the fake browser, whose ``wait`` is scripted to the executor's
``waited 1 s``.

**Oracles.** B2's "exactly as for a normal step" is the same turn in which step 2 itself committed the re-ask's stream
(``as_step``): the voiced text, the executed calls and the step-3 request must match it. B5 is a subprocess run over
``git archive`` of BASE (origin-proved), the T-0207 A2.2 pattern, over T-0207's ``_dump`` sections. The clip text is
H-T179-4's sentence typed by hand.

**Readings, as ruled** (docs/tasks/T-0231.md, "Rulings", orchestrator 2026-11-02):

- R-B1 the outcome note is ``{"what": "empty step re-ask outcome", "request_id", "step", "outcome"}``.
- R-B2 the re-ask's ``llm.request`` phase is ``"empty_reask"`` (the step's own record stays ``"tool_step"``); both
  records carry the step's machine id, distinct ``attempt_id`` and the same ``prompt``.
- R-B3 "browser action": ``click_at``/``type_at``/``press_key``/``scroll`` with status ``executed``; ``wait`` is not
  one (unlike T-0207's ``_BROWSER_ACTIONS``). A refused click is not executed: not re-asked. **Amended by T-0234
  A1:** a refused browser action arms the re-ask too (``failed``/``timeout`` still do not).
- R-B4 a T-0171 retry that itself comes back empty counts as that step's re-ask: no third attempt, and the turn ends
  as in B3. Whether the empty-step notes are written for it is not pinned (the ruling does not say).
- R-B5 B3 "nothing in history": the clip text never enters history or a ``TtsSpeak``; what step 1 really spoke stays
  what it was.
- R-B6 after the cap of 2 re-asks, a later empty step is sent once and ends the turn as in B3.
- R-B7 the empty-step end plays ``lost_thread`` but the HUD and the turn line never say "degenerate" (no HUD message
  and no turn-line value carries the word; ``tool_loop_end`` is not ``"degenerate"``; no ``degenerate_retries``).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``live`` (1280x800 real
JPEG frames, ReGround off, no live feedback), loop on (8 steps), n = 1, deterministic. BASE = eb58ea00.

Not pending: every re-ask row is red until B lands (named in each docstring); the guard rows (marked GUARD) are green
at BASE and must stay green.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts.toolcall import ToolOutcome
from tests import _git_range as gr
from tests.fakes.llm import engine_empty_error
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0207_session_check as t207
from tests.unit._live_browser_support import refused
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool

REPO = Path(__file__).resolve().parents[2]
#: BASE: main at eb58ea00 (``T-0232: evidence wording``), where T-0231 branched. A literal, never a moving ref.
BASE_SHA = "eb58ea00"

NOTE_ASKED = "empty step re-asked"
NOTE_OUTCOME = "empty step re-ask outcome"  # R-B1
ASKED_KEYS = frozenset({"what", "request_id", "step"})
OUTCOME_KEYS = frozenset({"what", "request_id", "step", "outcome"})
PHASE = "empty_reask"  # R-B2
OUTCOMES = frozenset({"call", "answer", "empty"})
EMPTY_KEY = "empty_step"
LOST_THREAD = "lost_thread"
LOST_THREAD_TEXT = "Sorry, I lost my train of thought. Could you ask again?"  # H-T179-4, verbatim
#: T-0171 criterion 5, verbatim
RETRY_NOTE = "the engine returned an empty retryable response; retrying once"

Q = tsl.Q
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
CLICK2, CLICK2_ARGS = tsl.CLICK2, tsl.CLICK2_ARGS
CLICK3 = tsl.CLICK3
SEARCH_CALL = tsl.SEARCH_CALL
WAIT = "<|tool_call>call:wait{}<tool_call|>"
STEP1 = [answer(S1), tool(VALID_CLICK)]
EMPTY: list[Any] = []
CALL_STEP = [answer(S2), tool(CLICK2)]
FINAL = [answer(S3)]


# --------------------------------------------------------------------------------------------- reading the rig


def asked(rig: Any) -> list[dict[str, Any]]:
    return t207.notes(rig, NOTE_ASKED)


def outcome_notes(rig: Any) -> list[dict[str, Any]]:
    """R-B1: the outcome notes."""
    return t207.notes(rig, NOTE_OUTCOME)


def stray_notes(rig: Any) -> list[dict[str, Any]]:
    """Any other note naming an empty-step re-ask (a misnamed outcome note shows here)."""
    return [
        d
        for d in t207.notes(rig)
        if str(d.get("what", "")).startswith("empty step")
        and d.get("what") not in (NOTE_ASKED, NOTE_OUTCOME)
    ]


def attempts(rig: Any, machine_id: str) -> list[str]:
    return [rid for rid, _ in tsl.generates_for(rig, machine_id)]


def requests_for(rig: Any, machine_id: str) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if d.get("request_id") == machine_id]


def clips(rig: Any, kind: str) -> list[Any]:
    return [u for u in rig.audio.clip_plays() if getattr(u, "kind", None) == kind]


def assert_reasked(rig: Any, k: int, outcome: str) -> None:
    """B1: one ``empty step re-asked`` note for step ``k`` (exact keys), one outcome note ``outcome``, and exactly two
    engine attempts under ``t1-gen-s{k}``, fresh ids, the same prompt; two ``llm.request`` records, distinct
    ``attempt_id``, the same ``prompt``."""
    mid = tsl.step_id(k)
    notes = asked(rig)
    assert [n for n in notes if n.get("step") == k] == [
        {"what": NOTE_ASKED, "request_id": mid, "step": k}
    ], f"T-0231 B1: one {NOTE_ASKED!r} note {{what, request_id, step}} for step {k}; got {notes}"
    assert all(set(n) == ASKED_KEYS for n in notes), notes
    outs = [n for n in outcome_notes(rig) if n.get("step") == k]
    assert len(outs) == 1 and outs[0].get("request_id") == mid, (
        f"T-0231 B1 (R-B1): one outcome note for step {k}; got {outcome_notes(rig)}"
    )
    assert set(outs[0]) == OUTCOME_KEYS and outs[0].get("outcome") == outcome, (
        f"T-0231 B1/R-B1: the outcome note {{what, request_id, step, outcome}} says {outcome!r}; got {outs[0]}"
    )
    assert stray_notes(rig) == [], stray_notes(rig)
    # the T-0207 completion check is a later attempt under the same machine id: it is not the step or its re-ask
    checks = {rid for rid, _ in t207.checks_of(rig, mid)}
    gens = [(rid, p) for rid, p in tsl.generates_for(rig, mid) if rid not in checks]
    assert len(gens) == 2, (
        f"T-0231 B1: the empty step is resent once: 2 engine attempts under {mid}; got {[r for r, _ in gens]}"
    )
    (a1, p1), (a2, p2) = gens
    assert a1 != a2, f"T-0231 B1: a fresh attempt id; got {a1} twice"
    assert p1 == p2, "T-0231 B1: the re-ask sends the same prompt"
    assert tsl.tail(p1) == tsl.tail(p2)
    assert t207._wire(p1) == t207._wire(p2), "T-0231 B1: the same prompt bytes on the wire"
    every = requests_for(rig, mid)
    recs = [r for r in every if r.get("phase") in ("tool_step", PHASE)]
    others = [r.get("phase") for r in every if r.get("phase") not in ("tool_step", PHASE)]
    assert set(others) <= {t207.PHASE} and len(others) == len(checks), (
        f"T-0231 B1: under {mid} only the step, its re-ask and T-0207 checks; other phases {others}"
    )
    assert len(recs) == 2, f"T-0231 B1 (R-B2): two llm.request records under {mid}; got {recs}"
    assert recs[0]["attempt_id"] != recs[1]["attempt_id"], recs
    assert [r.get("phase") for r in recs] == ["tool_step", PHASE], (
        f"T-0231 R-B2: the step's record, then the re-ask's with phase {PHASE!r}; got {[r.get('phase') for r in recs]}"
    )
    assert recs[0]["prompt"] == recs[1]["prompt"], "T-0231 B1: the logged prompts are equal"


def assert_not_reasked(rig: Any, k: int, why: str) -> None:
    mid = tsl.step_id(k)
    assert asked(rig) == [] and outcome_notes(rig) == [] and stray_notes(rig) == [], (
        f"T-0231 B4 ({why}): no empty-step re-ask; notes {asked(rig)} {outcome_notes(rig)}"
    )
    assert not [d for d in rig.kind("llm.request") if d.get("phase") == PHASE], why
    assert len(attempts(rig, mid)) == 1, (
        f"T-0231 B4 ({why}): {mid} is sent once; got {attempts(rig, mid)}"
    )


async def turn(
    tmp: Path,
    *steps: Any,
    on: Any = t207.ABSENT,
    checks: Any = (),
    script: Any = STEP1,
    browser: Any = None,
    kit_of: Any = None,
    setup: Any = None,
) -> Any:
    return await t207.claim_turn(
        tmp,
        on,
        steps=steps,
        checks=checks,
        script=script,
        browser=browser,
        kit_of=kit_of,
        setup=setup,
    )


# =====================================================================================================
# B1/B2: re-asked once, recovered
# =====================================================================================================


async def test_empty_after_click_reasked_then_call_continues(tmp_path: Path) -> None:
    """B1/B2: step 1 clicks (executed), step 2 comes back with zero commits, its re-ask makes a call: the call executes
    and the loop goes on to step 3, which answers. Red at BASE: step 2 is sent once and the turn ends silent (12:12)."""
    rig = await turn(tmp_path, EMPTY, CALL_STEP, FINAL)
    assert_reasked(rig, 2, "call")
    assert t207.invoked(rig) == ["t1-call", "t1-call-s2"], (
        f"T-0231 B2: the re-ask's call executes; invocations {t207.invoked(rig)}"
    )
    assert t207.inv_of(rig, "t1-call-s2").arguments == CLICK2_ARGS
    assert len(attempts(rig, tsl.step_id(3))) == 1, "T-0231 B2: step 3 is sent, once"
    assert S3 in rig.spoken_sources(), "T-0231 B2: step 3's answer is voiced"
    assert rig.record("t1").extras.get(EMPTY_KEY) is None, (
        "T-0231 B3: empty_step is written only when the turn ended still empty"
    )


#: R-B3's four browser actions as the step-1 call (review: a mutant arming on ``click_at`` alone survived)
ARMING = {
    "click_at": VALID_CLICK,
    "type_at": f"<|tool_call>call:type_at{{press_enter:true,text:{Q}vinland saga{Q},x:500,y:250}}<tool_call|>",
    "press_key": f"<|tool_call>call:press_key{{combo:{Q}Ctrl+L{Q}}}<tool_call|>",
    "scroll": f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>",
}


@pytest.mark.parametrize("action", list(ARMING))
async def test_empty_after_each_browser_action_is_reasked(tmp_path: Path, action: str) -> None:
    """B1 / R-B3: step 1 makes each browser action in turn (``type_at`` with Enter, ``press_key`` Ctrl+L, ``scroll``
    and ``click_at``), it executes, and step 2 comes back empty: step 2 is re-asked and its call executes."""
    rig = await turn(tmp_path, EMPTY, CALL_STEP, FINAL, script=[answer(S1), tool(ARMING[action])])
    assert t207.results(rig, "t1-call") == ["executed"], f"premise: the step-1 {action} executed"
    assert t207.inv_of(rig, "t1-call").name == action
    assert_reasked(rig, 2, "call")
    assert "t1-call-s2" in t207.invoked(rig)


async def test_reask_thought_is_a_retry_block(tmp_path: Path) -> None:
    """B1 / T-0180 B6: the re-ask's thought is published as ``kind: "retry"`` for step 2 under the re-ask's attempt id
    (the empty attempt published nothing)."""
    from tests.unit._tool_turn_support import thought
    from tests.unit.test_t0180_publish import published_thoughts

    secret = "ZQX-T0231-REASK-THOUGHT click the second item. "
    rig = await turn(tmp_path, EMPTY, [thought(secret, 0), answer(S2, 1), tool(CLICK2)], FINAL)
    assert_reasked(rig, 2, "call")
    first, reask = attempts(rig, tsl.step_id(2))[:2]
    mine = [m for m in published_thoughts(rig) if m.request_id == tsl.step_id(2)]
    assert mine, "premise: the re-ask's thought was published"
    assert {m.attempt_id for m in mine} == {reask}, (first, reask, [m.attempt_id for m in mine])
    assert {(m.step, m.kind) for m in mine} == {(2, "retry")}, (
        f"T-0231 / T-0180 B6: the re-ask is a retry block; {[(m.step, m.kind) for m in mine]}"
    )


async def test_recovered_call_continues_exactly_as_a_normal_step(tmp_path: Path) -> None:
    """B2 ("exactly as for a normal step"): the re-asked turn and the turn in which step 2 itself made the re-ask's call
    voice the same text, execute the same calls with the same arguments, and send the same step-3 prompt. Red at
    BASE (the re-asked turn never reaches step 3)."""
    got = await turn(tmp_path / "reasked", EMPTY, CALL_STEP, FINAL)
    want = await turn(tmp_path / "as_step", CALL_STEP, FINAL)
    assert len(attempts(got, tsl.step_id(3))) == 1, (
        f"T-0231 B2: step 3 was sent after the re-ask; engine ids {[r for r, _, _ in got.llm.generates]}"
    )
    assert got.spoken_sources() == want.spoken_sources()
    assert [(i.name, dict(i.arguments)) for i in got.browser.invocations] == [
        (i.name, dict(i.arguments)) for i in want.browser.invocations
    ]
    assert t207._wire(tsl.step_request(got, 3)) == t207._wire(tsl.step_request(want, 3)), (
        "T-0231 B2: the step-3 prompt is the normal step's"
    )


async def test_empty_after_click_reasked_then_answer_runs_completion_check(tmp_path: Path) -> None:
    """B2: with ``completion_check=True``, step 2 empty, its re-ask a no-call claim: the claim goes through the T-0207
    check (one check request for step 2, its ``asked`` note), the check's answer is voiced and the claim is not. Red at
    BASE: no re-ask, no check, nothing voiced after step 1."""
    rig = await turn(tmp_path, EMPTY, t207.CLAIM_STEP, on=True, checks=(t207.NO_CALL_CHECK,))
    assert_reasked(rig, 2, "answer")
    assert len(t207.checks_of(rig, tsl.step_id(2))) == 1, (
        f"T-0231 B2: the completion check runs on the re-ask's answer; checks {rig.llm.check_prompts}"
    )
    assert [n.get("step") for n in t207.notes(rig, t207.NOTE_ASKED)] == [2]
    voiced = rig.spoken_sources()
    assert t207.CHECK_SAYS in voiced, f"T-0231 B2: the check's answer is voiced; {voiced!r}"
    assert t207.CLAIM not in voiced, "T-0231 B2/T-0207: the unchecked claim is never voiced"


# =====================================================================================================
# B3: still empty
# =====================================================================================================


def assert_empty_end(rig: Any, what: str) -> None:
    """B3 / R-B7: one ``lost_thread`` clip (H-T179-4's sentence), ``empty_step: true`` and ``system_clip:
    "lost_thread"`` on the turn line; neither the HUD nor the turn line says "degenerate"; the clip text never in
    history, a ``TtsSpeak`` or a ``tts.utterance``."""
    lost = clips(rig, LOST_THREAD)
    assert len(lost) == 1, (
        f"T-0231 B3 ({what}): one lost_thread clip; clips "
        f"{[getattr(u, 'kind', None) for u in rig.audio.clip_plays()]}"
    )
    assert str(getattr(lost[0], "text", "")) == LOST_THREAD_TEXT
    record = rig.record("t1")
    extras = dict(record.extras)
    assert extras.get(EMPTY_KEY) is True, f"T-0231 B3 ({what}): empty_step: true; {extras}"
    assert extras.get("system_clip") == LOST_THREAD, extras
    line = json.dumps(record.model_dump(mode="json"), default=str)
    assert "degenerate" not in line, (
        f"T-0231 R-B7 ({what}): the turn line never says degenerate; {line[:1500]}"
    )
    assert extras.get("tool_loop_end") == "empty_step", (
        f"T-0231 B3 ({what}): tool_loop_end is empty_step; {extras.get('tool_loop_end')!r}"
    )
    assert "degenerate_retries" not in extras
    hud = tsl.published_json(rig)
    assert "degenerate" not in hud, (
        f"T-0231 R-B7 ({what}): no HUD message says degenerate; "
        f"...{hud[max(0, hud.find('degenerate') - 300) :][:600]}"
    )
    assert LOST_THREAD_TEXT not in t207.history_json(rig), (
        "T-0231 B3: the clip never enters history"
    )
    assert not any(LOST_THREAD_TEXT in c.source for c in rig.machine.commands(C.TtsSpeak))
    assert not [r for r in rig.kind("tts.utterance") if LOST_THREAD_TEXT in json.dumps(r)]


async def test_still_empty_ends_with_lost_thread(tmp_path: Path) -> None:
    """B3: step 2 empty, its re-ask empty too: the turn ends with one ``lost_thread`` clip (H-T179-4's sentence,
    ``source=""``), ``empty_step: true`` and ``system_clip: "lost_thread"`` on the turn line, nothing voiced beyond
    what step 1 voiced, the clip text never in history or a ``TtsSpeak``, no completion check (``completion_check``
    on: there is no answer to check), no third request, and (R-B7) no "degenerate" on the HUD or the turn line. Red
    at BASE: no re-ask, no clip, no key."""
    rig = await turn(tmp_path, EMPTY, EMPTY, FINAL, on=True)
    assert_reasked(rig, 2, "empty")
    assert attempts(rig, tsl.step_id(3)) == [], "T-0231 B3: the turn ends; step 3 is never sent"
    assert_empty_end(rig, "still empty")
    assert rig.llm.check_prompts == [], "T-0231 B3: no completion check of an empty step"
    voiced = rig.spoken_sources()
    assert voiced.strip() in ("", S1), (
        f"T-0231 B3: nothing is invented as speech (only step 1's own words may be voiced); {voiced!r}"
    )


# =====================================================================================================
# B4: scope
# =====================================================================================================


async def test_at_most_two_reasks_per_turn(tmp_path: Path) -> None:
    """B4 / R-B6: steps 2 and 3 are empty and each re-ask makes a call; step 4 is empty too: it is sent once (the cap
    of 2 per turn) and the turn ends as in B3 (``lost_thread``, ``empty_step: true``, nothing "degenerate"); step 5 is
    never sent. Red at BASE (no re-ask of step 2)."""
    rig = await turn(
        tmp_path,
        EMPTY,
        CALL_STEP,
        EMPTY,
        [answer("Next one."), tool(CLICK3)],
        EMPTY,
        [answer("never sent")],
    )
    assert_reasked(rig, 2, "call")
    assert_reasked(rig, 3, "call")
    assert [n["step"] for n in asked(rig)] == [2, 3], f"T-0231 B4: two re-asks; {asked(rig)}"
    assert len(attempts(rig, tsl.step_id(4))) == 1, (
        f"T-0231 B4: the third empty step is not re-asked; {attempts(rig, tsl.step_id(4))}"
    )
    assert "never sent" not in rig.spoken_sources()
    assert attempts(rig, tsl.step_id(5)) == [], "T-0231 R-B6: the turn ends at step 4"
    assert_empty_end(rig, "after the cap")


async def test_empty_after_search_not_reasked(tmp_path: Path) -> None:
    """B4 (GUARD, too wide): step 1 clicks (executed), step 2 searches (executed through ``FakeSearch``), step 3 is
    empty: not re-asked, because the previous step was a search (an implementation that arms on "a browser action
    this turn", T-0207's rule, fails here). Green at BASE."""
    rig = await turn(
        tmp_path,
        [answer(S2), tool(SEARCH_CALL)],
        EMPTY,
        [answer("never sent")],
        kit_of=t207._search_kit,
    )
    assert t207.results(rig, "t1-call-s2") == ["executed"], "premise: the search executed"
    assert_not_reasked(rig, 3, "after google_search")


async def test_empty_after_first_step_search_not_reasked(tmp_path: Path) -> None:
    """B4 (GUARD): step 1 searches (executed), step 2 is empty: not re-asked. Green at BASE."""
    rig = await turn(
        tmp_path,
        EMPTY,
        [answer("never sent")],
        script=[answer(S1), tool(SEARCH_CALL)],
        kit_of=t207._search_kit,
    )
    assert t207.results(rig, "t1-call") == ["executed"], "premise"
    assert_not_reasked(rig, 2, "after google_search")


async def test_empty_after_wait_not_reasked(tmp_path: Path, monkeypatch: Any) -> None:
    """B4 (GUARD, too wide): step 1 clicks, step 2 waits (executed, ``waited 1 s``), step 3 is empty: not re-asked
    (B4 names ``wait`` as non-browser; R-B3). Green at BASE."""
    from tests.unit.test_t0202_session_wait import wait_kit

    def waited(fb: Any, inv: Any) -> ToolOutcome:
        after = fb.capture_post()
        return ToolOutcome(
            call_id=inv.call_id,
            status="executed",
            action="waited 1 s",
            reason=None,
            resolved={"frame_after": after.phash},
        )

    browser = t207.browser_of()
    browser.handlers["t1-call-s2"] = waited
    rig = await turn(
        tmp_path,
        [answer(S2), tool(WAIT)],
        EMPTY,
        [answer("never sent")],
        browser=browser,
        kit_of=lambda b: wait_kit(b, tmp_path / "kit", monkeypatch),
    )
    assert t207.results(rig, "t1-call-s2") == ["executed"], "premise: the wait executed"
    assert_not_reasked(rig, 3, "after wait")


async def test_empty_after_refused_click_reasked(tmp_path: Path) -> None:
    """B1 as amended by T-0234 A1 (named re-pin; was ``test_empty_after_refused_click_not_reasked``, R-B3's
    "a refused click is not executed: not re-asked"): step 1's click is refused by the browser, step 2 is empty: it
    is re-asked once and the re-ask's call executes. Red until T-0234 A lands."""
    browser = t207.browser_of()
    browser.outcome("t1-call", lambda inv: refused(inv, "the element is covered"))
    rig = await turn(tmp_path, EMPTY, CALL_STEP, FINAL, browser=browser)
    assert t207.results(rig, "t1-call") == ["refused"], "premise"
    assert_reasked(rig, 2, "call")
    assert "t1-call-s2" in t207.invoked(rig)


async def test_empty_first_generate_not_reasked(tmp_path: Path) -> None:
    """B4 (GUARD): ``generate`` (step 1) coming back empty is unchanged: sent once, no re-ask note. Green at BASE."""
    rig = await turn(tmp_path, [answer("never sent")], script=[])
    assert asked(rig) == [] and outcome_notes(rig) == []
    assert len(attempts(rig, "t1-gen")) == 1, attempts(rig, "t1-gen")


async def test_t0171_retry_unchanged_and_not_double_counted(tmp_path: Path) -> None:
    """B4 (GUARD; R-B4): step 2's first attempt is the engine's empty retryable response (T-0171): one T-0171 retry
    note, two attempts, the retry's call executes; no empty-step note. Green at BASE."""
    rig = await turn(
        tmp_path,
        CALL_STEP,
        FINAL,
        setup=lambda r: r.llm.fail_steps(engine_empty_error),
    )
    assert len(t207.notes(rig, RETRY_NOTE)) == 1, t207.notes(rig)
    assert len(attempts(rig, tsl.step_id(2))) == 2, attempts(rig, tsl.step_id(2))
    assert asked(rig) == [] and outcome_notes(rig) == [], (
        "T-0231 B4: a T-0171 retry is not also an empty-step re-ask"
    )
    assert "t1-call-s2" in t207.invoked(rig)


async def test_t0171_retry_that_comes_back_empty_is_the_reask(tmp_path: Path) -> None:
    """B4 / R-B4: step 2's first attempt is the engine's empty retryable response (T-0171) and T-0171's retry comes back
    with zero commits: that retry is the step's re-ask, so there is no third attempt; the turn ends as in B3
    (``lost_thread``, ``empty_step: true``, nothing "degenerate"); step 3 is never sent. Red at BASE (no clip, no
    key)."""
    rig = await turn(
        tmp_path,
        EMPTY,
        CALL_STEP,
        FINAL,
        setup=lambda r: r.llm.fail_steps(engine_empty_error),
    )
    assert len(t207.notes(rig, RETRY_NOTE)) == 1, t207.notes(rig)
    assert len(attempts(rig, tsl.step_id(2))) == 2, (
        f"T-0231 R-B4: the T-0171 retry counts as the re-ask (no third attempt); {attempts(rig, tsl.step_id(2))}"
    )
    assert attempts(rig, tsl.step_id(3)) == [], "T-0231 R-B4: the turn ends at step 2"
    assert "t1-call-s2" not in t207.invoked(rig)
    assert_empty_end(rig, "T-0171 retry empty")


# =====================================================================================================
# B5: a turn with no empty step is BASE's, byte for byte
# =====================================================================================================


async def _scenario(tmp: Path) -> dict[str, Any]:
    """B5's BASE scenarios (no empty step): a click, a click, an answer; a claim after a click with the completion
    check on; a search then an answer; and a dry-run tool turn."""
    from tests.unit import _tool_turn_support as tts

    out: dict[str, Any] = {}

    async def one(name: str, **k: Any) -> None:
        rig = await t207.claim_turn(tmp / name, **k)
        out[name] = t207._dump(rig, tmp / name)

    await one("click_click_answer", on=t207.ABSENT, steps=(CALL_STEP, FINAL))
    await one(
        "claim_checked",
        on=True,
        steps=(t207.CLAIM_STEP,),
        checks=(t207.NO_CALL_CHECK,),
    )
    await one(
        "search_answer",
        on=t207.ABSENT,
        script=[answer(S1), tool(SEARCH_CALL)],
        steps=(FINAL,),
        kit_of=t207._search_kit,
    )
    async with tts.tool_rig(tmp / "dry_run", script=STEP1, policy=tsl.LOOP) as rig:
        rig.llm.script_steps(CALL_STEP, FINAL)
        await rig.ask(t207.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        await rig.advance(300)
    out["dry_run"] = t207._dump(rig, tmp / "dry_run")
    return out


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0231_empty_step as m
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
    """B5 (GUARD): four turns with no empty step -- prefill and generate wire bodies (as sent, with options, ids and
    plugin arguments), ``llm.request``/``llm.done``/``note``/``tool.call`` records, HUD messages, machine events with
    their command lists and turn-line extras -- equal BASE's (eb58ea00, run in a subprocess over ``git archive``,
    origin-proved). Green at BASE."""
    base = await asyncio.to_thread(_base_capture, tmp_path)
    now = await _scenario(tmp_path / "now")
    assert set(now) == set(base)
    for name in sorted(base):
        b, n = base[name], now[name]
        for part in sorted(b):
            assert n[part] == b[part], (
                f"T-0231 B5 ({name}): {part} differs from BASE ({len(b[part])} vs {len(n[part])} items)"
            )
    assert t207.CHECK_SAYS in json.dumps(base["claim_checked"]["machine"]), (
        "premise: the check ran in the BASE claim turn"
    )
    assert EMPTY_KEY not in json.dumps([s["extras"] for s in base.values()])
