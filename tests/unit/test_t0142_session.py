"""T-0142 group B (A3-A6, session side): with ``Session(..., degeneracy_guard=True)`` a request whose committed thought
collapses into repetition is aborted and the turn ends; ReGround pass 2 falls back instead (docs/tasks/T-0142.md).

Every row runs T-0111g's session over the fakes (``tool_rig`` with the live kit over a ``FakeLiveBrowser``, the
``dry_run`` kit or tools off), built through ``guard_rig`` below, which passes ``degeneracy_guard`` only when the row
gives it (``ABSENT``: not passed, the BASE session). A session without the keyword is a named ``AssertionError``.

**The interfaces these rows pin** (A3-A6; the readings are listed in the test-author's report):

- R-3: ``Session(..., degeneracy_guard: bool = False)``; ``session.DEGENERATE_MESSAGE: str``, non-empty and not
  ``OVER_BUDGET_MESSAGE``.
- R-4: the trip ``note`` has exactly the keys ``what, request_id, attempt_id, canvas_idx, unit, repeats, kind``;
  ``what == "degenerate thought: request aborted"``; ``request_id`` is the machine id (``t1-gen``, ``t1-gen-s<k>``, or
  the pass-2 key ``<call_id>-reground``), ``attempt_id`` the engine id the fake saw, ``canvas_idx`` that of the commit
  that tripped.
- R-5: the abort reaches the backend (``FakeLlm.aborted`` holds the engine id) *before* ``E.LlmError`` is submitted
  (the order of ``backend.abort`` calls and ``Session.submit`` calls, recorded by wrapping both).
- R-6 ("logged as dropped"): a commit the session receives from that attempt after the trip is in the log as its
  ``llm.commit`` record and is marked dropped, either by ``"dropped": true`` on that record or by a ``note`` whose
  ``what`` contains ``dropped`` and whose ``attempt_id`` is the attempt's. Whether the session keeps reading the
  stream after the abort is not pinned; ``LateLlm`` keeps serving it (an engine whose canvases were already in
  flight), so a session that reads on must drop what it reads.
- R-7: a ReGround pass-2 trip has outcome ``pass2_timeout`` (A3.5: "pass 2's existing failure outcome", and the
  test list's "same outcome and records as a pass-2 timeout at BASE"); timings (``ms``, ``first_ms``, ``tool_ms``,
  the turn line's ``reground.ms``) are not compared.
- R-8: ``degeneracy_guard: {"on": true, "tripped": n}`` is on every turn line (``TurnRecord.extras``) of every mode
  when the guard is on, ``n`` the turn's tripped requests (pass 2 included); no key with ``False``.
- R-9: ``cli`` passes ``degeneracy_guard=True`` to ``Session`` in ``off``, ``dry_run`` and ``live``; there is no
  ``--degeneracy*`` flag and no ``Profile`` field naming it (R-n).

Expected values come from the fixture (``tests/fixtures/t0142/thoughts.jsonl``: the logged S2#15 commits and the
test-author's census of the commit at which K = 8 first holds) and from hand-written scripts.

Every row fails at BASE e1ae76e on a named ``AssertionError`` (the missing keyword or constant), except
``test_no_flag_or_profile_key`` (R-n, a negative pin: green at BASE and after).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps) in live
and ``dry_run``, 1280x800 frames, ReGround off unless a row says otherwise, n = 1, deterministic.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from local_astra.config import Profile
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import (
    FakeLiveBrowser,
    hud_frame,
    live_frame,
    live_kit,
    make_reground_llm,
)
from tests.unit._tool_turn_support import (
    VALID_CLICK,
    ToolLlm,
    answer,
    assert_concurrency,
    thought,
    tool,
)
from tests.unit.test_cli_session import CLI, flags_of
from tests.unit.test_cli_t0131 import build
from tests.unit.test_session_loop import POLICY
from tests.unit.test_session_reground import click, regrounds
from tests.unit.test_session_reground import frames as jpeg_browser

GUARD = "degeneracy_guard"
NOTE = "degenerate thought: request aborted"
NOTE_KEYS = {"what", "request_id", "attempt_id", "canvas_idx", "unit", "repeats", "kind"}
ABSENT: Any = object()
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "t0142" / "thoughts.jsonl"
TOK = re.compile(r"\w+|[^\w\s]+")
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
#: 12 back-to-back "the" after a clean clause: trips (>= 8) at this commit, by hand
COLLAPSE = "I see the cart page now and" + " the" * 12
LATE = "ZQX-LATE-T0142 your order is placed."
STEP1 = [answer(S1), tool(VALID_CLICK)]


# --------------------------------------------------------------------------------------------- seams


def session_mod() -> Any:
    return importlib.import_module("local_astra.conductor.session")


def require_guard() -> None:
    params = inspect.signature(session_mod().Session.__init__).parameters
    assert GUARD in params, (
        "T-0142 A3: Session(..., degeneracy_guard: bool = False) does not exist yet"
    )
    assert params[GUARD].default is False, "T-0142 A3/R-n: Session's default stays False"


def degenerate_message() -> str:
    s = session_mod()
    value = getattr(s, "DEGENERATE_MESSAGE", None)
    assert isinstance(value, str) and value.strip(), (
        "T-0142 A3: session.DEGENERATE_MESSAGE (a non-empty str module constant) does not exist yet"
    )
    assert value != s.OVER_BUDGET_MESSAGE, "T-0142 A3: its own message, not the over_budget one"
    return value


def retried_message() -> str:
    """T-0179 B-6a/B-11: the message of a request whose one degenerate retry tripped too (supersedes H-T142-1's "the
    turn ends with ``DEGENERATE_MESSAGE``" for a request that is retried)."""
    s = session_mod()
    value = getattr(s, "DEGENERATE_RETRIED_MESSAGE", None)
    assert isinstance(value, str) and value.strip(), "T-0179 B-6a: DEGENERATE_RETRIED_MESSAGE"
    assert value != s.DEGENERATE_MESSAGE
    return value


class LateLlm(ToolLlm):
    """``ToolLlm`` that records what it hands out after an abort and, with ``late=True``, keeps serving an aborted
    stream (canvases already in flight when the abort lands). Every other name is the fake's."""

    def __init__(self, script: Any = (), *, late: bool = False) -> None:
        super().__init__(list(script))
        self.late = late
        #: (engine id, segment, text) of every commit handed to the session after its request was aborted
        self.after_abort: list[tuple[str, str, str]] = []

    async def abort(self, request_id: str) -> None:
        if not self.late:
            await super().abort(request_id)
            return
        self.aborted.append(request_id)  # recorded, but the stream goes on

    async def generate(  # type: ignore[override]
        self,
        prompt: Any,
        request_id: str,
        options: Any = None,
        *,
        xargs: Any = None,
        cd: Any = None,
    ) -> Any:
        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            if request_id in self.aborted:
                self.after_abort.append((request_id, c.segment, c.text))
            yield c


@asynccontextmanager
async def guard_rig(
    tmp_path: Path,
    *,
    mode: str = "live",
    guard: Any = True,
    script: Any = STEP1,
    llm: Any = None,
    browser: Any = None,
    reground: bool = False,
) -> AsyncIterator[Any]:
    """The session over the fakes with ``degeneracy_guard=guard`` (``ABSENT``: not passed). ``mode`` is ``live``
    (loop on, live kit), ``dry_run`` (loop on) or ``off`` (tools off). ``rig.order`` records ``("abort", engine id)``
    and ``("submit", event type, request id)`` in call order."""
    from tests.unit import _tool_turn_support as tts

    kwargs: dict[str, Any] = {}
    if guard is not ABSENT:
        require_guard()
        kwargs[GUARD] = guard
    if reground:
        kwargs["reground"] = True
    real = tts.Session

    class Guarded(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in kwargs.items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    if browser is None:
        browser = FakeLiveBrowser(
            [live_frame("onset")], post_frames=[live_frame(f"after-{i}") for i in range(1, 8)]
        )
    if llm is None:
        llm = make_reground_llm(browser, script) if reground else LateLlm(script)
    else:
        llm.script = list(script)
    order: list[tuple[Any, ...]] = []
    inner_abort = llm.abort

    async def abort(request_id: str) -> None:
        order.append(("abort", request_id))
        await inner_abort(request_id)

    llm.abort = abort
    if mode == "live":
        kit, policy, hud = live_kit(browser), tsl.LOOP, [hud_frame(0)]
    elif mode == "dry_run":
        kit, policy, hud = "default", tsl.LOOP, None
    else:
        kit, policy, hud = None, POLICY, None
    tts.Session = Guarded  # type: ignore[misc]
    try:
        async with tts.tool_rig(tmp_path, kit=kit, policy=policy, llm=llm, frames=hud) as rig:
            browser.now_ms = lambda: rig.clock.now_ms
            rig.browser = browser
            rig.order = order
            submit = rig.session.submit

            def spy(event: Any) -> None:
                order.append(("submit", type(event).__name__, getattr(event, "request_id", None)))
                submit(event)

            rig.session.submit = spy
            yield rig
    finally:
        tts.Session = real  # type: ignore[misc]


async def run(rig: Any, *steps: Any, question: str = tsl.QUESTION) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(question)
    await rig.turn_done("T-0142: the turn never ended", limit_ms=40_000)
    await rig.advance(300)


def fixture_row(task: str, repeat: int) -> dict[str, Any]:
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    (row,) = [
        r for r in rows if r["class"] == "collapsed" and r["task"] == task and r["repeat"] == repeat
    ]
    return row


def commits_of(row: dict[str, Any]) -> list[Commit]:
    return [
        thought(c["text"], c["canvas_idx"], open_=bool(c["thought_open"])) for c in row["thoughts"]
    ]


S215 = fixture_row("S2", 15)


def notes(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if d.get("what") == NOTE]


def attempts(rig: Any, request_id: str) -> list[str]:
    return [d["attempt_id"] for d in rig.kind("llm.request") if d.get("request_id") == request_id]


def errors(rig: Any, request_id: str) -> list[Any]:
    return [e for e in rig.machine.events(E.LlmError) if e.request_id == request_id]


def dones(rig: Any, request_id: str) -> list[Any]:
    return [e for e in rig.machine.events(E.LlmDone) if e.request_id == request_id]


def spoken(rig: Any) -> str:
    return " ".join(f"{c.text} {c.source}" for c in rig.machine.commands(C.TtsSpeak))


def history_text(rig: Any) -> str:
    return " ".join(f"{h.user_text} || {h.assistant_spoken}" for h in rig.state.history)


def hud_dump(rig: Any) -> str:
    return json.dumps([m.model_dump(mode="json") for m in rig.published.messages])


def run_present(text: str, unit: str, repeats: int) -> bool:
    toks = TOK.findall(text.casefold())
    block = TOK.findall(unit.casefold()) * repeats
    return bool(block) and any(
        toks[i : i + len(block)] == block for i in range(len(toks) - len(block) + 1)
    )


def check_note(
    note: dict[str, Any], *, request_id: str, attempt_id: str, canvas_idx: int, text: str
) -> None:
    """R-4, and that the reported unit really repeats ``repeats`` times in the thought text so far (a check)."""
    assert set(note) == NOTE_KEYS, f"T-0142 A3 (R-4): the note's keys; got {sorted(note)}"
    assert (note["request_id"], note["attempt_id"], note["canvas_idx"]) == (
        request_id,
        attempt_id,
        canvas_idx,
    ), note
    assert note["kind"] in ("ngram", "char_run"), note
    assert isinstance(note["repeats"], int) and isinstance(note["unit"], str), note
    if note["kind"] == "ngram":
        assert note["repeats"] >= 8 and run_present(text, note["unit"], note["repeats"]), note
    else:
        assert note["repeats"] >= 64 and note["unit"] * note["repeats"] in text, note


def dropped_logged(rig: Any, attempt_id: str, text: str) -> bool:
    for r in rig.log_records:
        d = r.get("data") or {}
        if (
            r["kind"] == "llm.commit"
            and d.get("attempt_id") == attempt_id
            and d.get("text") == text
            and d.get("dropped") is True
        ):
            return True
        if (
            r["kind"] == "note"
            and "dropped" in str(d.get("what"))
            and d.get("attempt_id") == attempt_id
        ):
            return True
    return False


def logged_commit(rig: Any, attempt_id: str, text: str) -> bool:
    return any(
        r["kind"] == "llm.commit"
        and r["data"].get("attempt_id") == attempt_id
        and r["data"].get("text") == text
        for r in rig.log_records
    )


# =====================================================================================================
# A3/A4: a step trips, the request is aborted, the turn ends
# =====================================================================================================


async def test_trip_aborts_and_ends(tmp_path: Path) -> None:
    """A3 (mutation 4) / A4: the logged S2#15 step-2 commits trip at the commit the census says (K = 8: the first,
    ``the.`` x 8): one note with the A3 fields, the step's engine id aborted *before* ``LlmError`` is submitted,
    ``LlmError(t1-gen-s2, DEGENERATE_MESSAGE)``, no ``LlmDone`` for it, ``tool_loop_end == "degenerate"``, the turn
    over well before ``llm_timeout_ms``, nothing of the thought spoken or stored (history holds step 1's words).

    T-0179 B-11 (supersedes H-T142-1's "no retry"): the step is retried once, and the S2#15 commits are scripted for
    both attempts, so each attempt trips at the census commit with its own note and abort, both aborts precede the one
    ``LlmError``, and its message is ``DEGENERATE_RETRIED_MESSAGE``."""
    degenerate_message()
    message = retried_message()
    async with guard_rig(tmp_path) as rig:
        await run(rig, commits_of(S215), commits_of(S215))
        extras = dict(rig.record("t1").extras)
        errs, gen_done = errors(rig, "t1-gen-s2"), dones(rig, "t1-gen")
        speech, hist, order = spoken(rig), history_text(rig), list(rig.order)
    both = attempts(rig, "t1-gen-s2")
    assert len(both) == 2, f"T-0179 B-11: the tripped step and its one retry; {both}"
    k = S215["k8_commit"]
    assert k == 1, "premise (fixture census): S2#15 holds the.x8 in its first commit"
    tripping = S215["thoughts"][k - 1]
    found = notes(rig)
    assert len(found) == 2, f"T-0142 A3.1 / T-0179 B-11: one trip note per attempt; got {found}"
    for note, attempt in zip(found, both, strict=True):
        check_note(
            note,
            request_id="t1-gen-s2",
            attempt_id=attempt,
            canvas_idx=tripping["canvas_idx"],
            text="".join(c["text"] for c in S215["thoughts"][:k]),
        )
        assert attempt in rig.llm.aborted, f"T-0142 A3.2: {attempt} aborted; {rig.llm.aborted}"
    assert [e.message for e in errs] == [message], (
        f"T-0142 A3.4 / T-0179 B-11: one LlmError with DEGENERATE_RETRIED_MESSAGE; {errs}"
    )
    error_at = order.index(("submit", "LlmError", "t1-gen-s2"))
    assert order.index(("abort", both[0])) < order.index(("abort", both[1])) < error_at, (
        f"T-0142 A3 (R-5): both aborts before the error; {order}"
    )
    assert dones(rig, "t1-gen-s2") == [], (
        "T-0142 A3.3: no llm.done of the tripped attempt reaches the machine"
    )
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    assert gen_done, "premise: step 1 ended"
    assert errs[0].t_ms - gen_done[-1].t_ms < POLICY.llm_timeout_ms, (
        "T-0142: the turn ends before the LLM timeout"
    )
    assert "the. the." not in speech and "the. the." not in hist, (
        "T-0142 A4: the thought is never voiced or stored"
    )
    assert "menu" in hist, f"premise: history holds what step 1 spoke; {hist!r}"


async def test_late_answer_after_trip_never_voiced(tmp_path: Path) -> None:
    """A3.3 / A4 (adversarial; mutation 5, and the mutant that drops only thought commits after the trip): an
    engine that keeps serving the aborted stream delivers an answer, a thought and a call after the tripping
    commit; none is voiced, stored, shown on the HUD, executed or handed to the machine, and each one the session
    received is logged as dropped (R-6).

    T-0179 B-11: the step is retried once; the same late-serving collapse is scripted for both attempts, and every
    assertion holds over both (each attempt's late commits are logged and dropped)."""
    degenerate_message()
    llm = LateLlm(late=True)
    step2 = [
        thought(COLLAPSE, 0, open_=True),
        answer(LATE, 1),
        thought(" and then more planning.", 1, open_=True),
        tool(tsl.CLICK3, canvas_idx=1),
    ]
    async with guard_rig(tmp_path, llm=llm) as rig:
        await run(rig, step2, step2, [answer(S3)])
        extras = dict(rig.record("t1").extras)
        speech, hist, hud = spoken(rig), history_text(rig), hud_dump(rig)
        late_commits = [
            e
            for e in rig.machine.events(E.LlmCommit)
            if e.request_id == "t1-gen-s2" and (e.segment == "answer" or "planning" in e.text)
        ]
        executed = [i.call_id for i in rig.browser.invocations]
    both = attempts(rig, "t1-gen-s2")
    assert len(both) == 2, f"T-0179 B-11: the tripped step and its one retry; {both}"
    assert len(notes(rig)) == 2, "T-0142 A3: each attempt tripped"
    assert "ZQX-LATE" not in speech, f"T-0142 A4: the late answer was voiced: {speech!r}"
    assert "ZQX-LATE" not in hist, "T-0142 A4: the late answer reached history"
    assert "ZQX-LATE" not in hud, "T-0142 A3.3: the late answer reached the HUD"
    assert late_commits == [], f"T-0142 A3.3: late commits reached the machine: {late_commits}"
    assert dones(rig, "t1-gen-s2") == [], "T-0142 A3.3: the late llm.done reached the machine"
    assert executed == ["t1-call"], f"T-0142 A3.3: the late call executed; {executed}"
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    for rid, _segment, text in llm.after_abort:
        if rid not in both:
            continue
        assert logged_commit(rig, rid, text), (
            f"T-0142 A3.3: a received commit is logged: {text[:40]!r}"
        )
        assert dropped_logged(rig, rid, text), (
            f"T-0142 A3.3 (R-6): not logged as dropped: {text[:40]!r}"
        )


async def test_straddle_trips_in_session(tmp_path: Path) -> None:
    """A1/A3 (adversarial; mutation 2, each commit judged alone): 5 repeats closing canvas 0 and 3 opening canvas 1
    trip at canvas 1 on the concatenation; the step's call is never executed."""
    degenerate_message()
    c0 = "I look at the page and" + " the" * 5
    c1 = " the" * 3 + " then I click."
    step2 = [
        thought(c0, 0, open_=True),
        thought(c1, 1, open_=True),
        answer(S2, 1),
        tool(tsl.CLICK3, canvas_idx=1),
    ]
    async with guard_rig(tmp_path) as rig:
        # T-0179 B-11: the straddle is scripted for the step and its one retry; each trips at canvas 1
        await run(rig, step2, step2, [answer(S3)])
        extras = dict(rig.record("t1").extras)
        executed = [i.call_id for i in rig.browser.invocations]
    both = attempts(rig, "t1-gen-s2")
    assert len(both) == 2, f"T-0179 B-11: the tripped step and its one retry; {both}"
    found = notes(rig)
    assert len(found) == 2, f"T-0142 A1: the straddling collapse trips each attempt; notes {found}"
    for note, attempt in zip(found, both, strict=True):
        check_note(note, request_id="t1-gen-s2", attempt_id=attempt, canvas_idx=1, text=c0 + c1)
        assert note["repeats"] == 8, note
    assert executed == ["t1-call"] and extras.get("tool_loop_end") == "degenerate", (
        executed,
        extras,
    )


async def test_drafts_never_judged(tmp_path: Path) -> None:
    """A8 / R-o (adversarial; mutation 1, judging anything but committed thought): clean thought commits beside a
    repetitive *answer* and a call whose draft text repeats (``la`` x 20) never trip; the turn runs as without the
    guard, and the turn line says ``tripped: 0``. (The session takes no denoising-tap input at BASE: the draft
    channel it can see is the non-thought commit text.) T-0142b R-11: the answer is ``the`` x 5, under
    ``MIN_REPEATS_ANSWER`` (6), since T-0142b judges answers too."""
    degenerate_message()
    q = '<|"|>'
    typed = "la " * 20
    type_call = f"<|tool_call>call:type_at{{press_enter:false,text:{q}{typed.strip()}{q},x:305,y:448}}<tool_call|>"
    step2 = [thought("I check the field first.", 0), answer("Typing it now.", 0), tool(type_call)]
    step3 = [thought("Done typing.", 0), answer(" the" * 5, 0)]
    async with guard_rig(tmp_path) as rig:
        await run(rig, step2, step3)
        extras = dict(rig.record("t1").extras)
        executed = [i.call_id for i in rig.browser.invocations]
    assert notes(rig) == [], f"T-0142 A8: non-thought text tripped: {notes(rig)}"
    assert errors(rig, "t1-gen-s2") == [] and errors(rig, "t1-gen-s3") == []
    assert executed == ["t1-call", "t1-call-s2"], executed
    assert extras.get("tool_loop_end") == "final", extras.get("tool_loop_end")
    assert extras.get(GUARD) == {"on": True, "tripped": 0}, extras.get(GUARD)


async def test_guard_is_per_request(tmp_path: Path) -> None:
    """A3 (adversarial; a guard that accumulates across a turn's requests): 5 repeats ending step 1's thought and
    3 opening step 2's are two requests, judged apart; nothing trips."""
    degenerate_message()
    step1 = [thought("Plan: click the button, then" + " the" * 5), answer(S1), tool(VALID_CLICK)]
    async with guard_rig(tmp_path, script=step1) as rig:
        await run(rig, [thought(" the" * 3 + " next step."), answer(S3)])
        extras = dict(rig.record("t1").extras)
    assert notes(rig) == [], notes(rig)
    assert extras.get("tool_loop_end") == "final" and extras.get(GUARD) == {
        "on": True,
        "tripped": 0,
    }, extras


async def test_guard_false_never_trips(tmp_path: Path) -> None:
    """A8 (mutation 7, the guard on with ``False``): ``degeneracy_guard=False`` over the S2#15 commits writes no note,
    no degenerate error and no turn-line key; the turn ends ``final`` as at BASE."""
    message = degenerate_message()
    async with guard_rig(tmp_path, guard=False) as rig:
        await run(rig, commits_of(S215))
        extras = dict(rig.record("t1").extras)
    assert notes(rig) == []
    assert [e for e in rig.machine.events(E.LlmError) if e.message == message] == []
    assert extras.get("tool_loop_end") == "final" and GUARD not in extras, extras


async def test_tools_off_generate_trips(tmp_path: Path) -> None:
    """A3 (the generate, tools off): the turn's generate trips: note for ``t1-gen``, its engine id aborted,
    ``LlmError(t1-gen, DEGENERATE_MESSAGE)``, no ``tool_loop_end`` (no loop), the later answer never voiced and
    history without assistant text.

    T-0179 B-11: the generate is retried once and the fake replays the same collapsing script, so both attempts trip
    (a note and an abort each), ``tripped: 2``, and the one ``LlmError`` carries ``DEGENERATE_RETRIED_MESSAGE``."""
    degenerate_message()
    message = retried_message()
    script = [thought(COLLAPSE, 0, open_=True), answer(LATE, 1)]
    async with guard_rig(tmp_path, mode="off", script=script) as rig:
        await rig.ask("what is on this screen")
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        extras = dict(rig.record("t1").extras)
        speech, hist = spoken(rig), history_text(rig)
    both = attempts(rig, "t1-gen")
    assert len(both) == 2, f"T-0179 B-11: the tripped generate and its one retry; {both}"
    found = notes(rig)
    assert len(found) == 2, found
    for note, attempt in zip(found, both, strict=True):
        check_note(note, request_id="t1-gen", attempt_id=attempt, canvas_idx=0, text=COLLAPSE)
        assert attempt in rig.llm.aborted
    assert [e.message for e in errors(rig, "t1-gen")] == [message]
    assert "tool_loop_end" not in extras, extras
    assert extras.get(GUARD) == {"on": True, "tripped": 2}, extras.get(GUARD)
    assert "ZQX-LATE" not in speech and "ZQX-LATE" not in hist and " the the" not in hist


async def test_dry_run_step1_trips(tmp_path: Path) -> None:
    """A3 (the generate with the loop on, ``dry_run``): step 1 trips before its call; nothing executes, the turn
    line says ``degenerate``.

    T-0179 B-11: the fake replays the collapsing script for the one retry, so both attempts of ``t1-gen`` trip and the
    turn ends with ``DEGENERATE_RETRIED_MESSAGE``."""
    degenerate_message()
    message = retried_message()
    script = [thought(COLLAPSE, 0, open_=True), answer(S1, 1), tool(VALID_CLICK)]
    async with guard_rig(tmp_path, mode="dry_run", script=script) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        extras = dict(rig.record("t1").extras)
        executed = list(rig.executor.invocations)
    assert len(attempts(rig, "t1-gen")) == 2, attempts(rig, "t1-gen")
    assert [n["request_id"] for n in notes(rig)] == ["t1-gen", "t1-gen"], notes(rig)
    assert [e.message for e in errors(rig, "t1-gen")] == [message], errors(rig, "t1-gen")
    assert executed == [], executed
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")


async def test_next_turn_runs_after_trip(tmp_path: Path) -> None:
    """docs/02 §4 (one live generate plus one prefill; ``llm.abort`` idempotent): after a tripped turn the engine ids
    are released and the next turn is answered and voiced."""
    degenerate_message()
    async with guard_rig(tmp_path) as rig:
        await run(rig, [thought(COLLAPSE, 0, open_=True)])
        rig.llm.script = [answer("Second answer here.")]
        await rig.ask("and now")
        await rig.until(
            lambda: len(rig.lines()) >= 2,
            "T-0142: the turn after a trip never ended",
            limit_ms=30_000,
        )
        await rig.advance(600)
        speech = spoken(rig)
        assert_concurrency(rig)
    assert "Second answer" in speech, speech
    assert rig.record("t2").extras.get(GUARD) == {"on": True, "tripped": 0}


# =====================================================================================================
# A3.5: ReGround pass 2 falls back
# =====================================================================================================


def _reground_view(rig: Any, extras: dict[str, Any]) -> dict[str, Any]:
    recs = [
        # timings differ by construction; the engine id carries the random session id
        {k: v for k, v in r.items() if k not in ("ms", "first_ms", "tool_ms", "attempt_id")}
        for r in regrounds(rig)
    ]
    reground = {k: v for k, v in dict(extras.get("reground") or {}).items() if k != "ms"}
    return {
        "regrounds": recs,
        "executed": [(i.call_id, dict(i.arguments)) for i in rig.browser.invocations],
        "results": [r.get("status") for r in rig.kind("tool.result")],
        "spoken": [c.source for c in rig.machine.commands(C.TtsSpeak)],
        "turn": {k: extras.get(k) for k in ("tool_loop_end", "tool_steps", "tool_steps_executed")},
        "reground": reground,
        "errors": [(e.request_id, e.message) for e in rig.machine.events(E.LlmError)],
    }


async def _pass2_turn(tmp_path: Path, guard: Any, pass2: Any) -> tuple[Any, dict[str, Any]]:
    async with guard_rig(tmp_path, guard=guard, reground=True, browser=jpeg_browser()) as rig:
        rig.llm.script_pass2(pass2)
        await run(rig, [answer(S3)])
        extras = dict(rig.record("t1").extras)
    return rig, extras


async def test_pass2_collapse_falls_back(tmp_path: Path) -> None:
    """A3.5 (mutation 6, a pass-2 trip ending the turn) / R-7: a ReGround pass 2 whose thought collapses (and would
    then have regrounded to (900, 100)) gives the outcome and records of a pass-2 timeout at BASE (the guard not
    passed, pass 2 hanging): pass 1's (500, 250) executes, the turn goes on to ``final``; apart from the note (for
    the pass-2 key and engine id), the abort of that id, the turn-line key and the timings."""
    from tests.unit._live_browser_support import HANG
    from tests.unit.test_session_reground import reground_timeout_ms

    degenerate_message()
    limit = reground_timeout_ms()
    collapse = [
        thought(COLLAPSE, 0, open_=True),
        thought(" the" * 20, 1, open_=True),
        tool(click(900, 100), canvas_idx=1),
    ]
    base, base_extras = await _pass2_turn(tmp_path / "base", ABSENT, HANG)
    rig, extras = await _pass2_turn(tmp_path / "guard", True, collapse)
    want, got = _reground_view(base, base_extras), _reground_view(rig, extras)
    assert [r["outcome"] for r in want["regrounds"]] == ["pass2_timeout"], "premise: BASE timed out"
    assert want["executed"] == [("t1-call", {"x": 500, "y": 250})], want["executed"]
    assert got == want, f"T-0142 A3.5: a pass-2 trip is a pass-2 timeout\nbase {want}\nnow  {got}"
    (p2,) = [d for d in rig.kind("llm.request") if d.get("phase") == "reground"]
    (note,) = notes(rig)
    check_note(
        note, request_id=p2["request_id"], attempt_id=p2["attempt_id"], canvas_idx=0, text=COLLAPSE
    )
    assert p2["attempt_id"] in rig.llm.aborted, rig.llm.aborted
    (rec,) = regrounds(rig)
    assert rec["attempt_id"] == p2["attempt_id"], rec
    assert rec["ms"] < limit, (
        f"T-0142: the trip ends pass 2 before REGROUND_TIMEOUT_MS ({rec['ms']} vs {limit})"
    )
    assert extras.get(GUARD) == {"on": True, "tripped": 1}, extras.get(GUARD)
    assert GUARD not in base_extras


async def test_tripped_counts_requests(tmp_path: Path) -> None:
    """A5 / R-8: a pass-2 trip and then a step trip in one turn are counted per tripped request; the step's trip ends
    the turn.

    T-0179 B-11: pass 2 is never retried (out of scope), the step is retried once and trips again (the collapse is
    scripted for both attempts), so three requests tripped: ``tripped: 3``, three notes, one ``LlmError`` with
    ``DEGENERATE_RETRIED_MESSAGE``."""
    degenerate_message()
    message = retried_message()
    async with guard_rig(tmp_path, reground=True, browser=jpeg_browser()) as rig:
        rig.llm.script_pass2([thought(COLLAPSE, 0, open_=True), tool(click(900, 100))])
        step2 = [thought(COLLAPSE, 0, open_=True), answer(S3)]
        await run(rig, step2, step2)
        extras = dict(rig.record("t1").extras)
    assert len(notes(rig)) == 3, notes(rig)
    assert len(attempts(rig, "t1-gen-s2")) == 2, attempts(rig, "t1-gen-s2")
    assert extras.get(GUARD) == {"on": True, "tripped": 3}, extras.get(GUARD)
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    assert [e.message for e in errors(rig, "t1-gen-s2")] == [message]


# =====================================================================================================
# A5: the turn-line key; A6: cli
# =====================================================================================================


@pytest.mark.parametrize(
    ("mode", "guard", "expected"),
    [
        ("live", True, {"on": True, "tripped": 0}),
        ("dry_run", True, {"on": True, "tripped": 0}),
        ("off", True, {"on": True, "tripped": 0}),
        ("live", False, None),
        ("off", False, None),
    ],
)
async def test_turn_line_key(tmp_path: Path, mode: str, guard: Any, expected: Any) -> None:
    """A5 (R-8): with the guard on every turn line of every mode carries ``degeneracy_guard: {"on": true,
    "tripped": 0}`` on a clean turn; with ``False`` there is no key."""
    script = STEP1 if mode != "off" else [answer("Here it is.")]
    async with guard_rig(tmp_path, mode=mode, guard=guard, script=script) as rig:
        if mode == "off":
            await rig.ask("what is on this screen")
            await rig.turn_done(limit_ms=30_000)
        else:
            await run(rig, [answer(S3)])
        extras = dict(rig.record("t1").extras)
    if expected is None:
        assert GUARD not in extras, extras
    else:
        assert extras.get(GUARD) == expected, (
            f"T-0142 A5: turn line {GUARD}; got {extras.get(GUARD)}"
        )


@pytest.mark.parametrize("mode", ["off", "dry_run", "live"])
def test_cli_turns_it_on(tmp_path: Path, capsys: Any, monkeypatch: Any, mode: str) -> None:
    """A6 (R-9): ``_build_graph`` passes ``degeneracy_guard=True`` to ``Session`` in every mode."""
    calls = build(tmp_path, capsys, monkeypatch, mode)
    got = calls.session_kwargs[-1].get(GUARD, ABSENT)
    assert got is True, (
        f"T-0142 A6: cli passes {GUARD}=True in {mode}; got {'not passed' if got is ABSENT else got}"
    )


# =====================================================================================================
# Review A additions (2026-10-23): the captioner, pass 2 with False, a pass-2 straddle, the issuing client
# =====================================================================================================


async def _caption_run(tmp_path: Path, guard: Any, script: list[Any]) -> tuple[Any, Any, list[Any]]:
    """Tools off, no turn: one uncaptioned episode, then one ``captioner.run_once()``. The fake LLM keeps serving
    an aborted stream (``late=True``), so a guard that reads on past its trip is seen in ``after_abort``.
    Returns the rig, the LLM and the ``set_caption`` calls."""
    from local_astra.contracts.memory import Episode
    from local_astra.contracts.perception import Source

    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8" + b"CAPTION-T0142|" * 8)
    llm = LateLlm(late=True)
    async with guard_rig(tmp_path / "rig", mode="off", guard=guard, script=script, llm=llm) as rig:
        stored: list[Any] = []
        memory = rig.session.memory
        inner_set = memory.set_caption

        async def set_caption(episode_id: int, caption: str) -> None:
            stored.append((episode_id, caption))
            await inner_set(episode_id, caption)

        memory.set_caption = set_caption
        await memory.ingest(
            Episode(ts_ms=0, source=Source.SCREEN, thumbnail_path=str(thumb), phash="cap0142")
        )
        await rig.session.captioner.run_once()
        await rig.pump()
        session_id = rig.session.session_id
    rig.session_id_ = session_id
    return rig, llm, stored


async def test_captioner_collapse_aborted(tmp_path: Path) -> None:
    """A3 ("the captioner if it thinks"; review A mutants 13, 14, 2c): with the guard on, a caption whose thought
    collapses gives exactly one note, an abort of the session-prefixed engine id (``<session>-caption-1``), no
    ``set_caption``, and nothing read from the stream past the tripping commit."""
    degenerate_message()
    script = [
        thought(COLLAPSE, 0, open_=True),
        thought(" more the", 1, open_=True),
        answer("A cart page.", 2),
    ]
    rig, llm, stored = await _caption_run(tmp_path, True, script)
    engine_id = f"{rig.session_id_}-caption-1"
    found = notes(rig)
    assert len(found) == 1, f"T-0142 A3: one note for the collapsed caption; got {found}"
    check_note(
        found[0],
        request_id=found[0]["request_id"],
        attempt_id=engine_id,
        canvas_idx=0,
        text=COLLAPSE,
    )
    assert str(found[0]["request_id"]).endswith("caption-1"), found[0]
    assert engine_id in llm.aborted, f"T-0142 A3: the prefixed caption id is aborted; {llm.aborted}"
    assert stored == [], f"T-0142 A3: a collapsed caption is never stored; {stored}"
    read_on = [c for c in llm.after_abort if c[0] == engine_id]
    assert read_on == [], f"T-0142 A3.3: the caption stream was read past the trip: {read_on}"


async def test_captioner_straddle_trips(tmp_path: Path) -> None:
    """A1/A3 (review A mutant 2c, the caption judged per commit): 5 + 3 repeats over two caption canvases trip at
    canvas 1; nothing is stored."""
    degenerate_message()
    c0 = "A photo of" + " the" * 5
    c1 = " the" * 3 + " shop."
    script = [thought(c0, 0, open_=True), thought(c1, 1, open_=True), answer("A shop page.", 2)]
    rig, _llm, stored = await _caption_run(tmp_path, True, script)
    (note,) = notes(rig)
    check_note(
        note,
        request_id=note["request_id"],
        attempt_id=f"{rig.session_id_}-caption-1",
        canvas_idx=1,
        text=c0 + c1,
    )
    assert stored == [], stored


async def test_captioner_unguarded_with_false(tmp_path: Path) -> None:
    """A8 for the captioner (the guard too wide): with ``False`` the same collapsing caption writes no note,
    aborts nothing and is stored as at BASE (its answer text)."""
    degenerate_message()
    script = [thought(COLLAPSE, 0, open_=True), answer("A cart page.", 1)]
    rig, llm, stored = await _caption_run(tmp_path, False, script)
    assert notes(rig) == []
    assert not [a for a in llm.aborted if a.endswith("caption-1")], llm.aborted
    assert [c for _, c in stored] == ["A cart page."], stored


async def test_pass2_collapse_with_false_is_base(tmp_path: Path) -> None:
    """A8 (review A mutant 20, the pass-2 guard ignoring the flag): ``degeneracy_guard=False`` with the collapsing
    pass 2 of ``test_pass2_collapse_falls_back`` gives no note, no abort of the pass-2 id, and BASE's outcome
    (the keyword not passed): pass 2's call regrounds (``ok``)."""
    degenerate_message()
    collapse = [
        thought(COLLAPSE, 0, open_=True),
        thought(" the" * 20, 1, open_=True),
        tool(click(900, 100), canvas_idx=1),
    ]
    base, base_extras = await _pass2_turn(tmp_path / "base", ABSENT, collapse)
    rig, extras = await _pass2_turn(tmp_path / "false", False, collapse)
    want, got = _reground_view(base, base_extras), _reground_view(rig, extras)
    assert [r["outcome"] for r in want["regrounds"]] == ["ok"], "premise: BASE regrounds"
    assert got == want, f"T-0142 A8: False is BASE for pass 2\nbase {want}\nnow  {got}"
    assert notes(rig) == []
    (p2,) = [d for d in rig.kind("llm.request") if d.get("phase") == "reground"]
    assert p2["attempt_id"] not in rig.llm.aborted, (
        f"T-0142 A8: pass 2 aborted with False; {rig.llm.aborted}"
    )
    assert GUARD not in extras


async def test_pass2_straddle_trips(tmp_path: Path) -> None:
    """A1/A3.5 (review A mutant 2b, pass 2 judged per commit): 5 repeats closing pass-2 canvas 0 and 3 opening
    canvas 1 trip at canvas 1; the outcome is ``pass2_timeout`` and pass 1's point executes."""
    degenerate_message()
    c0 = "The button sits at" + " the" * 5
    c1 = " the" * 3 + " top."
    pass2 = [
        thought(c0, 0, open_=True),
        thought(c1, 1, open_=True),
        tool(click(900, 100), canvas_idx=1),
    ]
    rig, extras = await _pass2_turn(tmp_path, True, pass2)
    (p2,) = [d for d in rig.kind("llm.request") if d.get("phase") == "reground"]
    (note,) = notes(rig)
    check_note(
        note, request_id=p2["request_id"], attempt_id=p2["attempt_id"], canvas_idx=1, text=c0 + c1
    )
    (rec,) = regrounds(rig)
    assert rec["outcome"] == "pass2_timeout", rec
    assert next((i.call_id, dict(i.arguments)) for i in rig.browser.invocations) == (
        "t1-call",
        {"x": 500, "y": 250},
    )
    assert extras.get("tool_loop_end") == "final", extras.get("tool_loop_end")


async def test_pass2_abort_on_issuing_client(tmp_path: Path) -> None:
    """A3.2 with T-0132 B4 (review A mutant 15): when pass 2 goes through a distinct ``reground_llm`` (single-image
    prompt, budget 280 vs 1120), a pass-2 trip aborts that engine id on ``reground_llm`` and never on the main
    client; the outcome is ``pass2_timeout`` and pass 1's point executes (the T-0132 B4 two-client pattern)."""
    from tests.unit._live_browser_support import reground_rig
    from tests.unit.test_session_reground_budget import SecondLlm
    from tests.unit.test_session_reground_budget import frames as two_frames

    degenerate_message()
    require_guard()
    second = SecondLlm()
    second.timed = [
        [(0, thought(COLLAPSE, 0, open_=True)), (0, tool(click(900, 100), canvas_idx=1))]
    ]
    browser = two_frames()
    kwargs = {
        "visual_tokens": 1120,
        "reground_llm": second,
        "reground_visual_tokens": 280,
        GUARD: True,
    }
    async with reground_rig(tmp_path, browser, script=STEP1, session_kwargs=kwargs) as rig:
        second.clock = rig.clock
        rig.llm.script_steps([answer(S3)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
        await rig.advance(300)
        executed = [(i.call_id, dict(i.arguments)) for i in browser.invocations]
    assert len(second.calls) == 1, f"premise: pass 2 went to reground_llm; {second.calls}"
    rid = second.calls[0][0]
    (note,) = notes(rig)
    assert note["attempt_id"] == rid, note
    assert rid in second.aborted, (
        f"T-0142 A3.2 / T-0132 B4: abort on reground_llm; {second.aborted}"
    )
    assert rid not in rig.llm.aborted, (
        f"T-0142 A3.2 / T-0132 B4: abort sent to self.llm; {rig.llm.aborted}"
    )
    (rec,) = regrounds(rig)
    assert rec["outcome"] == "pass2_timeout", rec
    assert executed[0] == ("t1-call", {"x": 500, "y": 250}), executed


def test_no_flag_or_profile_key() -> None:
    """R-n (negative pin, green at BASE and after): there is no ``--degeneracy*`` flag and no ``Profile`` field."""
    assert not [f for f in flags_of(CLI) if "degenera" in f]
    assert not [f for f in Profile.model_fields if "degenera" in f]
    assert cli is not None
