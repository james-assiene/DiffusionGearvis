"""T-0172 group B (B1-B8, B11, B12): the session re-asks once when a tools-on step only announces an action and makes
no call (docs/tasks/T-0172.md).

Every row runs the session over the fakes (``tests/unit/_tool_turn_support.tool_rig``: tools ``dry_run`` and the loop
on unless a row says otherwise; ``_live_browser_support.feedback_rig`` for the live rows), built through
``reask_rig`` below, which passes ``reask_announced_calls`` only when the row gives it (``ABSENT``: not passed, the
BASE session). A session without the keyword is a named ``AssertionError`` (``require_keyword``), never a
``TypeError``.

**The fake (re-scope R1).** A re-ask is a second request under one machine request id whose prompt is
``announced_call.reask_prompt(<the step's prompt as sent>, <step text>)``. Before R1 ``FakeLlm`` replayed ``script``
for every non-continuation prompt, so a step-1 re-ask could not be scripted apart from step 1, and a step-k re-ask
replayed step 1's stream. ``tests/fakes/llm.py`` gained (additively) a re-ask route told apart structurally by
``is_reask_prompt`` (last message a user turn whose whole text is ``REASK_NUDGE``): ``script_reasks``,
``fail_reasks``, ``hold_reasks``/``release_reasks``, ``reask_prompts``, ``reask_bodies`` (built by the real
``VllmClient.request_body``) and ``reask_overruns``. No real-service contract row: the route only chooses which
scripted stream answers; the wire body stays the real client's.

**Readings where the task is silent** (each listed in the test-author's report):

- R-a "the announcement followed by the re-ask's own text" (B3/B4: ``spoken_prefix``, ``step_text``, ``LoopStep.text``,
  the withheld note's ``chars``): superseded by B4a (amendment): exactly one space when neither side has whitespace at
  the join, otherwise unchanged (``joined``); with an empty re-ask text it is the announcement.
- R-b the failure note ``announced call re-ask failed`` "(error type)": the exception's class name appears among the
  note's values; the note carries no prompt or answer text.
- R-c "over-budget prompt" (B5) is read as the engine refusing the re-ask as over-long (HTTP 413, the T-0020 seam):
  it is not retried with less history.
- R-d a re-ask cut by a barge-in, a takeover, a disconnect or the shared ``llm_timeout`` has outcome ``aborted``
  (B7's list), on the outcome note and on the turn line.
- R-e calls other than the re-ask's carry no ``after_reask: true`` (absent or false).
- R-f the ``announced call re-asked`` note's ``chars`` is the announcement's length (every row's announcement is one
  sentence, so the step text and the matched sentence have the same length).

**Which rows are pending.** Every row that needs the keyword or the re-ask. Active (green at BASE a7811bed and
after): the fake's premise, ``test_absent_is_base_tree`` (the keyword-less session equals BASE's tree over the same
three turns, so the implementation cannot move the default path), ``test_frozen_paths_over_range`` and
``test_no_reask_without_keyword``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps unless a row
says otherwise), tools ``dry_run`` (``live`` where named: 1280x800 frames, ReGround off), n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit, ImagePart, Prompt
from local_astra.contracts.turn import TurnState
from local_astra.telemetry import session_log
from local_astra.toolcall import loop, router
from local_astra.toolcall.announced_call import REASK_NUDGE, announces, reask_prompt
from local_astra.toolcall.ask_first import ASKED_FIRST_REASON, asks_user
from local_astra.toolcall.user_declines import USER_DECLINED_REASON, user_declines
from tests import _git_range as gr
from tests.fakes.llm import engine_empty_error, http_error, is_reask_prompt, is_step_prompt
from tests.fakes.search import FakeSearch
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0135_live_recovery as t0135
from tests.unit import test_t0142_session as t0142
from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame
from tests.unit._t0177_pins import without_vision_frames  # T-0177 B-P1
from tests.unit._tool_turn_support import (
    NAMELESS,
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    assert_concurrency,
    thought,
    tool,
)
from tests.unit.test_session_loop import POLICY
from tests.unit.test_t0107a_differential import SECTIONS
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

REPO = Path(__file__).resolve().parents[2]
#: BASE: main at a7811bed (after ``Merge T-0172-a``, T-0171 and T-0163's cli work merged), where group B branched.
#: A literal, never a moving ref (tests.md).
BASE_SHA = "a7811bed39a5dc1e32973588bc4320f5745b0c81"
#: The implementation tip, bumped by the test-author after each implementation commit: e58c7ce1 (group B impl; was
#: BASE). BASE..TIP also spans the merge of main (b09bf10f, T-0163), which touches none of ``FROZEN``.
TIP_IMPL = "e58c7ce15fd8538437eadce3954d63de9d0b894c"
#: docs/02 as the implementation left it, read at a literal (bumped with ``TIP_IMPL``); never the working tree
DOC_REV = TIP_IMPL
#: docs/02 the paragraph is added to: the implementation's parent 5d083e4c, which holds main's T-0163 docs/02 edit
#: (dbbdd56d, merged at b09bf10f) -- comparing with BASE a7811bed would charge T-0163's edit to this task
DOC_PARENT = "5d083e4c3b8cc9b2139741bfce05dfbe34783439"

KEY = "reask_announced_calls"
TURN_KEY = "announced_call_reask"
PHASE = "announce_reask"
NOTE_ASKED = "announced call re-asked"
NOTE_OUTCOME = "announced call re-ask outcome"
NOTE_FAILED = "announced call re-ask failed"
ASKED_KEYS = frozenset({"what", "request_id", "step", "verb", "chars"})
OUTCOME_KEYS = frozenset({"what", "request_id", "step", "outcome"})
OUTCOMES = frozenset({"call", "no_call", "refused", "failed", "aborted"})
AFTER = "after_reask"
#: T-0171's retry note (verbatim from its criterion 5)
RETRY_NOTE = "the engine returned an empty retryable response; retrying once"
ABSENT: Any = object()

Q = '<|"|>'
#: the two smoke records (T-0125 at b372eda: t2-gen-s7 seq 904, t3-gen-s2 seq 1047), verbatim
ANN_WEATHER = "I will search for the current weather in Paris again."
ANN_PYTHON = "I will search for the latest stable version of Python."
ANN_CLICK = "I will click the Sign in button."
ANN_ADD = "I will add the blue mug to your cart."
ANN_REMOVE = "I will remove the blue mug from the cart."
ANN_TYPE = "I will type the email address into the field."
#: a step-1 announcement long enough to still be playing when the re-ask is cut (one sentence: ``tail`` reads it)
ANN_LONG = "I will click the Sign in button at the top right corner of the page to open the login form for your account."
#: a re-ask's words before its call, and a re-ask's whole answer when it makes none
R_CALL = "Searching now."
R_STOPPED = "The search limit was reached, so I could not search again."
R_ASKS = "Shall I place the order now?"
#: a re-ask that only announces again (adversarial: never a second re-ask)
R_AGAIN = "I will search for it now."
PY_QUERY = "latest stable version of Python"
PY_CALL = f"<|tool_call>call:google_search{{query:{Q}{PY_QUERY}{Q}}}<tool_call|>"
FIRST_QUERY = "latest Python release"
FIRST_CALL = f"<|tool_call>call:google_search{{query:{Q}{FIRST_QUERY}{Q}}}<tool_call|>"
SECRET = "ZQX-T0172-THOUGHT"
REPORT = tsl.REPORT
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
CLICK_ARGS, CLICK2, CLICK2_ARGS = tsl.CLICK_ARGS, tsl.CLICK2, tsl.CLICK2_ARGS
RECORDED = tsl.RECORDED_CLICK
QUESTION = tsl.QUESTION


# --------------------------------------------------------------------------------------------- seams


def session_cls() -> Any:
    from local_astra.conductor import session as session_mod

    return session_mod.Session


def require_keyword() -> None:
    params = inspect.signature(session_cls().__init__).parameters
    assert KEY in params, f"T-0172 B8: Session(..., {KEY}: bool | None = None) does not exist yet"
    assert params[KEY].default is None, (
        f"T-0172 B8/implementation notes: Session's {KEY} default stays None (defaults live in cli); "
        f"got {params[KEY].default!r}"
    )


@asynccontextmanager
async def reask_rig(
    tmp_path: Path,
    *,
    reask: Any = True,
    mode: str = "dry_run",
    script: Sequence[Commit] = (),
    policy: Any = None,
    executor: Any = None,
    search: Any = None,
    extra: dict[str, Any] | None = None,
    machine_cls: Any = None,
    browser: Any = None,
) -> AsyncIterator[Any]:
    """The session over the fakes with ``reask_announced_calls=reask`` (``ABSENT``: not passed). ``mode``: ``dry_run``
    (the default kit, loop on), ``live`` (``feedback_rig`` over ``browser``) or ``off`` (tools off, loop off).
    ``extra``: other ``Session`` keywords (``degeneracy_guard``, ``recover_thought_calls``)."""
    from tests.unit import _tool_turn_support as tts

    kwargs: dict[str, Any] = dict(extra or {})
    if reask is not ABSENT:
        require_keyword()
        kwargs[KEY] = reask
    if mode == "live":
        if browser is None:
            browser = FakeLiveBrowser(
                [live_frame("onset")], post_frames=[live_frame(f"after-{i}") for i in range(1, 8)]
            )
        async with feedback_rig(
            tmp_path, browser, script=list(script), session_kwargs=kwargs, machine_cls=machine_cls
        ) as rig:
            yield rig
        return
    real = tts.Session

    class WithReask(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in kwargs.items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    extra_rig: dict[str, Any] = {}
    if machine_cls is not None:
        extra_rig["machine_cls"] = machine_cls
    if mode == "off":
        kit, chosen = None, policy or POLICY
    else:
        kit, chosen = "default", policy or tsl.LOOP
    tts.Session = WithReask  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path,
            script=list(script),
            kit=kit,
            policy=chosen,
            executor=executor,
            search=search,
            **extra_rig,
        ) as rig:
            yield rig
    finally:
        tts.Session = real  # type: ignore[misc]


async def run(
    rig: Any,
    *steps: Sequence[Commit],
    reasks: Sequence[Sequence[Commit]] = (),
    question: str = QUESTION,
    limit_ms: int = 40_000,
) -> None:
    rig.llm.script_steps(*steps)
    rig.llm.script_reasks(*reasks)
    await rig.ask(question)
    await rig.turn_done("T-0172: the turn never ended", limit_ms=limit_ms)
    await rig.advance(300)


# --------------------------------------------------------------------------------------------- reading the rig


def step_id(k: int, turn: str = "t1") -> str:
    return tsl.step_id(k, turn)


def call_id(k: int, turn: str = "t1") -> str:
    return tsl.call_id(k, turn)


def reask_requests(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if d.get("phase") == PHASE]


def reasks_of(rig: Any, machine_id: str) -> list[tuple[str, Prompt]]:
    pat = re.compile(rf"-{re.escape(machine_id)}-\d+$")
    return [(rid, p) for rid, p in rig.llm.reask_prompts if pat.search(rid)]


def original(rig: Any, machine_id: str) -> tuple[str, Prompt]:
    """The step's own request (its last attempt that is not a re-ask), as the backend was handed it."""
    found = [(rid, p) for rid, p in tsl.generates_for(rig, machine_id) if not is_reask_prompt(p)]
    assert found, f"no request was sent for {machine_id}"
    return found[-1]


def notes(rig: Any, what: str | None = None) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if what is None or d.get("what") == what]


def turn_key(rig: Any, turn: str = "t1") -> Any:
    return rig.record(turn).extras.get(TURN_KEY, ABSENT)


def step_text(raw: str) -> str:
    """B1's ``step_text``: the answer channel with thoughts removed (``router.route(...).spoken_prefix``, the
    definition the task cites)."""
    return router.route(raw).spoken_prefix


def joined(a: str, r: str) -> set[str]:
    """B4a (amendment, replaces R-a): the announcement and the re-ask's own text joined with one space when neither
    side has whitespace at the join, otherwise as they are. One exact value (a set of one, so the rows read the same)."""
    if not r:
        return {a}
    if a[-1:].isspace() or r[:1].isspace():
        return {a + r}
    return {a + " " + r}


def images(prompt: Prompt) -> list[tuple[str, str | None]]:
    return [
        (p.url, p.phash) for m in prompt.messages for p in m.content if isinstance(p, ImagePart)
    ]


def invoked(rig: Any) -> list[tuple[str, dict[str, Any]]]:
    return [(i.call_id, dict(i.arguments)) for i in rig.executor.invocations]


def speaks_json(rig: Any) -> str:
    return json.dumps([c.model_dump(mode="json") for c in rig.speaks()])


def history_json(rig: Any) -> str:
    return json.dumps([h.model_dump(mode="json") for h in rig.state.history])


def assert_no_reask(rig: Any, what: str) -> None:
    assert rig.llm.reask_prompts == [], (
        f"T-0172 B1 ({what}): no re-ask is sent; the fake saw {[r for r, _ in rig.llm.reask_prompts]}"
    )
    assert reask_requests(rig) == [], (
        f"T-0172 B1 ({what}): no llm.request with phase {PHASE!r}; got {reask_requests(rig)}"
    )


def assert_one_reask(rig: Any, machine_id: str, what: str) -> tuple[str, Prompt]:
    found = reasks_of(rig, machine_id)
    assert len(found) == 1, (
        f"T-0172 B1/B5 ({what}): exactly one re-ask under {machine_id}; the fake saw "
        f"{[r for r, _ in rig.llm.reask_prompts]} (llm.request phases "
        f"{[(d.get('phase'), d.get('request_id')) for d in rig.kind('llm.request')]})"
    )
    assert len(rig.llm.reask_prompts) == 1, (
        f"T-0172 B5 ({what}): at most one re-ask per turn; {[r for r, _ in rig.llm.reask_prompts]}"
    )
    return found[0]


def outcome_notes(rig: Any) -> list[dict[str, Any]]:
    return notes(rig, NOTE_OUTCOME)


def withheld(rig: Any) -> list[tuple[Any, Any]]:
    return [(n.get("step"), n.get("chars")) for n in notes(rig, "step text withheld")]


def log_index(rig: Any, kind: str, pred: Any) -> list[int]:
    return [
        i for i, r in enumerate(rig.log_records) if r["kind"] == kind and pred(r.get("data") or {})
    ]


def machine_index(rig: Any, pred: Any) -> list[int]:
    return [i for i, (ev, _, _) in enumerate(rig.machine.log) if pred(ev)]


# =====================================================================================================
# Premises (active): the fake routes a re-ask on its own script
# =====================================================================================================


def test_fake_routes_reask_prompts():
    """R1 premise: ``reask_prompt`` over a step-1 prompt and over a step-3 continuation is a re-ask to the fake and
    neither a step nor a continuation; the prompts it was built from are not re-asks; the smoke records and the rows'
    announcements are announcements (``announces``), the rows' negatives are not."""
    from tests.fakes.llm import engine_empty_prompt

    step1 = engine_empty_prompt()
    step3 = loop.step_prompt(
        step1,
        [
            loop.LoopStep(S1, "click_at", CLICK_ARGS, RECORDED),
            loop.LoopStep(S2, "click_at", CLICK2_ARGS, RECORDED),
        ],
    )
    assert is_step_prompt(step3) and not is_reask_prompt(step3) and not is_reask_prompt(step1)
    for sent in (step1, step3):
        r = reask_prompt(sent, ANN_WEATHER)
        assert is_reask_prompt(r) and not is_step_prompt(r), sent.continue_final_message
        assert r.continue_final_message is False
    for text, verb in ((ANN_WEATHER, "search"), (ANN_PYTHON, "search"), (ANN_CLICK, "click")):
        a = announces(text)
        assert a is not None and a.verb == verb, (text, a)
    for text in (ANN_ADD, ANN_REMOVE, ANN_TYPE, R_AGAIN, ANN_LONG):
        assert announces(text) is not None, text
    for text in (R_CALL, R_STOPPED, R_ASKS, "I will remember that.", REPORT, S3):
        assert announces(text) is None, text
    assert asks_user(R_ASKS) and not asks_user(ANN_ADD + " " + R_CALL)


# =====================================================================================================
# B1/B2: when it fires, and the request it sends
# =====================================================================================================


async def test_fires_once_step_k(tmp_path):
    """B1/B2 (the smoke's turn 3): a step-2 answer that only announces a search, with no call, sends exactly one
    re-ask under ``t1-gen-s2``: a fresh engine id under the same machine id, its ``llm.request`` record has phase
    ``announce_reask``, the step, both ids, the options and the prompt sent; that prompt is
    ``reask_prompt(<the step-2 prompt as sent>, <step text>)``; every body of the turn has the same system message,
    tools, template arguments and images; at most one live generate at any time; the re-asked note."""
    search = FakeSearch()
    async with reask_rig(
        tmp_path,
        search=search,
        script=[thought("The user wants the latest Python."), tool(FIRST_CALL)],
    ) as rig:
        await run(
            rig,
            [thought(f"{SECRET} the search failed, try again"), answer(ANN_PYTHON)],
            reasks=[[answer(R_STOPPED)]],
        )
        sid = step_id(2)
        step_engine, sent = original(rig, sid)
        engine, prompt = assert_one_reask(rig, sid, "step 2")
        assert_concurrency(rig)
    want = reask_prompt(sent, step_text(ANN_PYTHON))
    assert prompt == want, (
        "T-0172 B2: the re-ask prompt is reask_prompt(<the step prompt as sent>, <step text>)\n"
        f"  last messages want {[m.model_dump(mode='json') for m in want.messages[-2:]]}\n"
        f"  got  {[m.model_dump(mode='json') for m in prompt.messages[-2:]]}"
    )
    assert engine != step_engine and re.search(rf"-{re.escape(sid)}-\d+$", engine), (
        f"T-0172 B2: a fresh engine id under the same machine id; step {step_engine}, re-ask {engine}"
    )
    assert int(engine.rsplit("-", 1)[1]) > int(step_engine.rsplit("-", 1)[1]), (step_engine, engine)
    (req,) = reask_requests(rig)
    assert {"phase", "step", "request_id", "attempt_id", "options", "prompt"} <= set(req), sorted(
        req
    )
    assert (req["request_id"], req["attempt_id"], req["step"]) == (sid, engine, 2), req
    assert req["prompt"] == session_log.redact(prompt.model_dump(mode="json")), (
        "T-0172 B2: the logged prompt is the re-ask prompt sent"
    )
    # B2: the cached prefix is untouched -- the same system message, tools and template arguments in every request of
    # the turn, the step's images, and the re-ask's head is the step's messages without its continued final message
    # (``bodies`` holds every tools-on generate the real client would send; captions and prefills are not in it)
    assert len(rig.llm.bodies) == 3, (
        f"premise: step 1, step 2 and the re-ask; {len(rig.llm.bodies)}"
    )
    systems = {json.dumps(b["messages"][0], sort_keys=True) for b in rig.llm.bodies}
    tools = {json.dumps(b.get("tools"), sort_keys=True) for b in rig.llm.bodies}
    kwargs = {json.dumps(b.get("chat_template_kwargs"), sort_keys=True) for b in rig.llm.bodies}
    assert len(systems) == 1 and len(tools) == 1 and len(kwargs) == 1, (
        f"T-0172 B2: one system message, tool list and template kwargs across the turn; got {len(systems)}, "
        f"{len(tools)}, {len(kwargs)}"
    )
    assert prompt.messages[: len(sent.messages) - 1] == sent.messages[:-1], (
        "T-0172 B2: nothing before the appended messages changes"
    )
    assert images(prompt) == images(sent), "T-0172 B2: the re-ask carries the step's images"
    assert REASK_NUDGE not in json.dumps(prompt.messages[0].model_dump(mode="json")), (
        "the nudge never enters the system message"
    )
    asked = notes(rig, NOTE_ASKED)
    assert len(asked) == 1 and set(asked[0]) == ASKED_KEYS, (
        f"T-0172 B7: one re-asked note with keys {sorted(ASKED_KEYS)}; got {asked}"
    )
    assert (asked[0]["request_id"], asked[0]["step"], asked[0]["verb"], asked[0]["chars"]) == (
        sid,
        2,
        "search",
        len(ANN_PYTHON),
    ), asked[0]
    assert search.queries == [FIRST_QUERY], (
        f"premise: step 1 searched; the re-ask made no call; {search.queries}"
    )


async def test_step_k_call_withheld(tmp_path):
    """B3/B4 (the smoke's turn 3, the re-ask commits the search): the re-ask's ``google_search`` executes as step 2's
    call; nothing of step 2 (the announcement nor the re-ask's words) is voiced, the withheld note counts both; one
    ``llm.done`` record and one ``LlmDone`` for ``t1-gen-s2``, after the re-ask, carrying the valid call; no TtsSpeak
    between step 2's first commit and its ``LlmDone``; the report of step 3 is voiced; outcome ``call``."""
    search = FakeSearch()
    async with reask_rig(tmp_path, search=search, script=[tool(FIRST_CALL)]) as rig:
        await run(
            rig,
            [answer(ANN_PYTHON)],
            [answer(REPORT)],
            reasks=[[thought(SECRET), answer(R_CALL), tool(PY_CALL)]],
        )
        sid = step_id(2)
        assert_one_reask(rig, sid, "step 2")
        dones = rig.dones(sid)
        commits = machine_index(
            rig, lambda ev: isinstance(ev, E.LlmCommit) and ev.request_id == sid
        )
        done_at = machine_index(rig, lambda ev: isinstance(ev, E.LlmDone) and ev.request_id == sid)
        between = (
            [
                c
                for _, _, cmds in rig.machine.log[commits[0] : done_at[0] + 1]
                for c in cmds
                if isinstance(c, C.TtsSpeak)
            ]
            if commits and done_at
            else None
        )
        spoken = speaks_json(rig)
        history = history_json(rig)
    assert search.queries == [FIRST_QUERY, PY_QUERY], (
        f"T-0172 B3: the re-ask's search executes as step 2's call; FakeSearch saw {search.queries}"
    )
    assert len(dones) == 1 and dones[0].tool is not None and dones[0].tool.status == "valid", (
        f"T-0172 B3: one LlmDone({sid}) carrying the re-ask's valid call; got {dones}"
    )
    done_records = log_index(rig, "llm.done", lambda d: d.get("request_id") == sid)
    reask_at = log_index(rig, "llm.request", lambda d: d.get("phase") == PHASE)
    assert len(done_records) == 1 and reask_at and done_records[0] > reask_at[0], (
        f"T-0172 B3: one llm.done record for {sid}, after the re-ask's request; done {done_records}, "
        f"re-ask {reask_at}"
    )
    assert between == [], f"T-0172 B3: no TtsSpeak while step 2 and its re-ask run; got {between}"
    for text in (ANN_PYTHON, R_CALL, SECRET, REASK_NUDGE):
        assert text not in spoken, f"T-0172 B3: {text!r} is voiced"
    assert REPORT in rig.spoken_sources(), rig.spoken_sources()
    for text in (ANN_PYTHON, R_CALL, REASK_NUDGE, SECRET):
        assert text not in history, f"T-0172 B3/B6: history holds {text!r}"
    a = step_text(ANN_PYTHON)
    held = withheld(rig)
    assert held in [[(2, len(j))] for j in joined(a, R_CALL)], (
        f"T-0172 B3: the withheld note counts the announcement and the re-ask's words ({sorted(joined(a, R_CALL))}); "
        f"got {held}"
    )
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["call"], outcome_notes(rig)
    assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "call"}, turn_key(rig)


async def test_step_k_no_call_voiced(tmp_path):
    """B3 (the smoke's turn 2 shape): a re-ask that makes no call ends the step as one: the announcement, then the
    re-ask's answer, voiced; one ``LlmDone`` with no tool after the re-ask; no second re-ask; history holds both and
    never the nudge; outcome ``no_call``."""
    async with reask_rig(tmp_path, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await run(rig, [answer(ANN_WEATHER)], reasks=[[thought(SECRET), answer(R_STOPPED)]])
        sid = step_id(2)
        assert_one_reask(rig, sid, "step 2")
        dones = rig.dones(sid)
        sources = rig.spoken_sources()
        history = rig.state.history[-1].assistant_spoken
        spoken = speaks_json(rig)
    assert len(dones) == 1 and dones[0].tool is None, dones
    a = step_text(ANN_WEATHER)
    (j,) = joined(a, R_STOPPED)
    assert sources in (S1 + j, S1 + " " + j), (
        f"T-0172 B3: step 1, then the announcement and the re-ask's answer as one step; sources {sources!r}"
    )
    assert ANN_WEATHER in history and R_STOPPED in history, history
    assert REASK_NUDGE not in history and SECRET not in history and SECRET not in spoken, history
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["no_call"], outcome_notes(rig)
    assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "no_call"}, turn_key(rig)


async def test_step1_voiced_then_call_after_drain(tmp_path):
    """B3 step 1: the announcement is voiced as today; the re-ask's call executes only after the announcement's
    run drained (T-0089 T11/T14), once; the step-2 request continues the original step-1 prompt (never the re-ask
    prompt) with the announcement as step 1's text; every ``tool.call`` record of ``t1-call`` has ``after_reask:
    true``; the report is voiced and history is ``"<announcement> <report>"``."""
    async with reask_rig(tmp_path, script=[answer(ANN_CLICK)]) as rig:
        await run(rig, [answer(REPORT)], reasks=[[tool(VALID_CLICK)]])
        sid = step_id(1)
        assert_one_reask(rig, sid, "step 1")
        _, pass1 = original(rig, sid)
        drained = machine_index(rig, lambda ev: isinstance(ev, E.PlaybackDrained))
        executes = [
            i
            for i, (_, _, cmds) in enumerate(rig.machine.log)
            if any(isinstance(c, C.ToolExecute) and c.call_id == call_id(1) for c in cmds)
        ]
        step2 = tsl.step_request(rig, 2)
        sources = rig.spoken_sources()
        history = rig.state.history[-1].assistant_spoken
    assert sources.startswith(ANN_CLICK), (
        f"T-0172 B3: the announcement is voiced at step 1; {sources!r}"
    )
    assert invoked(rig)[:1] == [(call_id(1), CLICK_ARGS)] and len(executes) == 1, (
        f"T-0172 B3: the re-ask's call executes once; invocations {invoked(rig)}, ToolExecute at {executes}"
    )
    # review fix: T-0089 T11/T14 emit ToolExecute in the drain event's own tuple, so "after the drain" is that index
    # or later; strictly-before an earlier index is what a call executed before the drain would show
    assert drained and drained[0] <= executes[0], (
        f"T-0172 B3 (T-0089 T11/T14): the call executes after the announcement drained; drained {drained}, "
        f"execute {executes}"
    )
    assert not is_reask_prompt(pass1) and REASK_NUDGE not in json.dumps(
        step2.model_dump(mode="json")
    ), "T-0172 B4: no later request carries the nudge"
    wants = [
        tsl.oracle(pass1, [(text, "click_at", CLICK_ARGS, RECORDED)])
        for text in joined(step_text(ANN_CLICK), "")
    ]
    assert step2 in wants, (
        "T-0172 B4: step 2 = step_prompt(<the step-1 prompt as sent>, [<announcement>, the re-ask's call])\n"
        f"  got tail {tsl.tail(step2)[-300:]!r}"
    )
    recs = [r for r in rig.tool_records() if r.get("call_id") == call_id(1)]
    assert recs and all(r.get(AFTER) is True for r in recs), (
        f"T-0172 B4: every tool.call record of the re-ask's call carries after_reask: true; {recs}"
    )
    assert history == f"{ANN_CLICK} {REPORT}", history


async def test_later_steps_unchanged(tmp_path):
    """B4: after a step-2 re-ask whose call executes, step 3's prompt is exactly ``step_prompt(<step-1 prompt as
    sent>, [step 1, (<announcement + re-ask words>, the call, its result)])``: no nudge, no extra user turn, no
    thought of either request; ``after_reask: true`` on the re-ask's call records only (R-e)."""
    async with reask_rig(tmp_path, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await run(
            rig,
            [thought(f"{SECRET} one"), answer(ANN_CLICK)],
            [answer(S3)],
            reasks=[[thought(f"{SECRET} two"), answer(R_CALL), tool(CLICK2)]],
        )
        assert_one_reask(rig, step_id(2), "step 2")
        _, pass1 = original(rig, step_id(1))
        step3 = tsl.step_request(rig, 3)
        later = [p for rid, p in tsl.generates_for(rig, step_id(3))]
    a = step_text(ANN_CLICK)
    wants = [
        tsl.oracle(
            pass1,
            [(S1, "click_at", CLICK_ARGS, RECORDED), (text, "click_at", CLICK2_ARGS, RECORDED)],
        )
        for text in joined(a, R_CALL)
    ]
    assert step3 in wants, (
        "T-0172 B4: step 3 is step_prompt(base, steps) with the re-asked step's text and call\n"
        f"  got tail {tsl.tail(step3)[-400:]!r}"
    )
    for p in later:
        text = json.dumps(p.model_dump(mode="json"))
        assert REASK_NUDGE not in text and SECRET not in text, (
            "T-0172 B4: no later request carries them"
        )
        assert not is_reask_prompt(p)
    assert invoked(rig) == [(call_id(1), CLICK_ARGS), (call_id(2), CLICK2_ARGS)], invoked(rig)
    s2 = [r for r in rig.tool_records() if r.get("call_id") == call_id(2)]
    others = [r for r in rig.tool_records() if r.get("call_id") != call_id(2)]
    assert s2 and all(r.get(AFTER) is True for r in s2), s2
    assert others and all(r.get(AFTER) is not True for r in others), (
        f"T-0172 B4: only the re-ask's call carries after_reask: true; {others}"
    )


# =====================================================================================================
# B1: when it does not fire (and the minimal pair that does)
# =====================================================================================================

#: case -> (blocked, power): each a dict of rig keywords and the turn's scripts. ``power`` differs from ``blocked`` in
#: the one feature the case is about, so a guard that is too wide (blocked fires) and one that is absent (power does
#: not) are both caught.
STEP1 = [answer(S1), tool(VALID_CLICK)]
OKAY = [[answer("Okay.")]]
NOT_WHEN: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "call_present": (
        dict(script=STEP1, steps=([answer(ANN_CLICK), tool(CLICK2)], [answer(S3)])),
        dict(script=STEP1, steps=([answer(ANN_CLICK)],), reasks=OKAY),
    ),
    "refused_draft": (
        dict(script=STEP1, steps=([answer(ANN_CLICK), tool(NAMELESS)],)),
        dict(script=STEP1, steps=([answer(ANN_CLICK)],), reasks=OKAY),
    ),
    "repairable_step1": (
        dict(
            script=[answer(ANN_CLICK), tool(REPAIRABLE_CLICK)],
            continuation=[tool(REPAIRED_CLICK)],
            steps=([answer(S3)],),
        ),
        dict(script=[answer(ANN_CLICK)], reasks=OKAY),
    ),
    "recovered_thought_call": (
        dict(
            mode="live",
            extra={"recover_thought_calls": True},
            script=STEP1,
            steps=([answer(ANN_TYPE), t0135.d5()], [answer(S3)]),
        ),
        dict(
            mode="live",
            extra={"recover_thought_calls": False},
            script=STEP1,
            steps=([answer(ANN_TYPE), t0135.d5()],),
            reasks=OKAY,
        ),
    ),
    "degenerate_trip": (
        dict(
            extra={"degeneracy_guard": True},
            script=STEP1,
            steps=([answer(ANN_CLICK), thought(t0142.COLLAPSE)],),
        ),
        dict(
            extra={"degeneracy_guard": False},
            script=STEP1,
            steps=([answer(ANN_CLICK), thought(t0142.COLLAPSE)],),
            reasks=OKAY,
        ),
    ),
    "budget_spent": (
        dict(
            policy=tsl.loop_policy(2),
            script=STEP1,
            steps=([answer(S2), tool(CLICK2)], [answer(ANN_CLICK)]),
        ),
        dict(
            policy=tsl.loop_policy(3),
            script=STEP1,
            steps=([answer(S2), tool(CLICK2)], [answer(ANN_CLICK)]),
            reasks=OKAY,
        ),
    ),
    "thought_only": (
        dict(script=STEP1, steps=([thought(ANN_CLICK)],)),
        dict(script=STEP1, steps=([answer(ANN_CLICK)],), reasks=OKAY),
    ),
    "deny_verb": (
        dict(script=STEP1, steps=([answer("I will remember that.")],)),
        dict(script=STEP1, steps=([answer("I will search for that.")],), reasks=OKAY),
    ),
    "question": (
        dict(script=STEP1, steps=([answer("Should I search for the weather?")],)),
        dict(script=STEP1, steps=([answer("I will search for the weather.")],), reasks=OKAY),
    ),
    "engine_empty_twice": (
        dict(script=STEP1, steps=([answer(ANN_CLICK)],), step_errors=(engine_empty_error,) * 2),
        dict(
            script=STEP1,
            steps=([answer(ANN_CLICK)],),
            step_errors=(engine_empty_error,),
            reasks=OKAY,
        ),
    ),
}


async def _not_when_turn(tmp: Path, case: dict[str, Any], reask: Any = True) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with reask_rig(
        tmp,
        reask=reask,
        mode=case.get("mode", "dry_run"),
        script=case["script"],
        policy=case.get("policy"),
        extra=case.get("extra"),
    ) as rig:
        if "continuation" in case:
            rig.llm.script_continuation(*case["continuation"])
        if "step_errors" in case:
            rig.llm.fail_steps(*case["step_errors"])
        await run(rig, *case.get("steps", ()), reasks=case.get("reasks", ()))
        extras = dict(rig.record("t1").extras)
    return {"rig": rig, "extras": extras}


@pytest.mark.parametrize("name", list(NOT_WHEN))
async def test_not_when(tmp_path, name):
    """B1 (too wide): a call present, a refused draft, a repairable step-1 draft, a recovered thought call (live), a
    T-0142 trip, the step budget spent, an announcement only in a thought, a deny verb, a question, and a step the
    engine answered empty twice (T-0171) send no re-ask; the turn line says ``asked: 0``."""
    blocked, _ = NOT_WHEN[name]
    got = await _not_when_turn(tmp_path, blocked)
    assert_no_reask(got["rig"], name)
    assert got["extras"].get(TURN_KEY) == {"on": True, "asked": 0, "outcome": None}, (
        f"T-0172 B7 ({name}): {got['extras'].get(TURN_KEY, 'absent')}"
    )


@pytest.mark.parametrize("name", list(NOT_WHEN))
async def test_not_when_power(tmp_path, name):
    """B1 (absent): each ``test_not_when`` case with its one blocking feature removed sends exactly one re-ask, so
    the negative rows are not vacuous."""
    _, power = NOT_WHEN[name]
    got = await _not_when_turn(tmp_path, power)
    rig = got["rig"]
    assert len(rig.llm.reask_prompts) == 1 and len(reask_requests(rig)) == 1, (
        f"T-0172 B1 ({name} power): one re-ask; the fake saw {[r for r, _ in rig.llm.reask_prompts]}, "
        f"records {reask_requests(rig)}"
    )
    assert got["extras"].get(TURN_KEY, {}).get("asked") == 1, got["extras"].get(TURN_KEY)


@pytest.mark.parametrize("reask", [None, False], ids=["none", "false"])
async def test_not_when_keyword_off(tmp_path, reask):
    """B1/B8: with the keyword ``None``, ``False`` or not given, the smoke's step-2 announcement sends no re-ask and
    is voiced as at BASE (one request for the step); tools off with ``False`` voices "I will search for that." with
    one generate. ``None``/absent write no turn-line key; ``False`` writes ``on: false``."""
    async with reask_rig(tmp_path / "loop", reask=reask, script=STEP1) as rig:
        await run(rig, [answer(ANN_WEATHER)])
        assert_no_reask(rig, f"keyword {reask!r}")
        assert len(tsl.generates_for(rig, step_id(2))) == 1
        assert rig.spoken_sources().endswith(ANN_WEATHER), rig.spoken_sources()
        key = turn_key(rig)
    want = {"on": False, "asked": 0, "outcome": None} if reask is False else ABSENT
    assert key == want, f"T-0172 B7/B8: turn line {TURN_KEY} for keyword {reask!r}: {key}"
    if reask is ABSENT:
        return
    async with reask_rig(
        tmp_path / "off", reask=reask, mode="off", script=[answer("I will search for that.")]
    ) as rig:
        await rig.ask("search the weather")
        await rig.turn_done()
        assert_no_reask(rig, "tools off")
        assert len(rig.llm.generates) == 1, [r for r, _, _ in rig.llm.generates]
        off_key = turn_key(rig)
    assert off_key is ABSENT, f"T-0172 B7: tools off writes no {TURN_KEY}; got {off_key}"


async def test_no_reask_without_keyword(tmp_path):
    """B12 (active, green at BASE and after): a session built without the keyword (every rig-built session, and
    ``Session``'s default) never re-asks: the smoke's step-2 announcement is voiced, one request for the step, no
    nudge anywhere, and the turn line has no ``announced_call_reask`` key."""
    async with reask_rig(tmp_path, reask=ABSENT, script=STEP1) as rig:
        await run(rig, [answer(ANN_WEATHER)])
        assert rig.llm.reask_prompts == [] and reask_requests(rig) == []
        assert len(tsl.generates_for(rig, step_id(2))) == 1, (
            [r for r, _, _ in rig.llm.generates],
            [type(c).__name__ for c in rig.machine.commands()],
            rig.spoken_sources(),
        )
        assert rig.spoken_sources() == S1 + ANN_WEATHER, rig.spoken_sources()
        assert turn_key(rig) is ABSENT
    assert REASK_NUDGE not in json.dumps(rig.log_records)


# =====================================================================================================
# B5: never twice, never worse
# =====================================================================================================

FAILURES = {
    "transport": lambda: httpx.ConnectError("connection refused"),
    "engine_empty": lambda: engine_empty_error,
    "over_long_413": lambda: http_error(413),
}


@pytest.mark.parametrize("name", list(FAILURES))
async def test_failure_falls_back(tmp_path, name):
    """B5 (R-b, R-c): a re-ask that fails (transport, the engine's empty response, an over-long prompt) writes one
    ``announced call re-ask failed`` note naming the error type, is never retried (one re-ask request, no T-0171 retry
    note for it), raises no ``llm.error``/``HudError``, and the step ends as without the re-ask: one ``LlmDone`` with
    no tool, the announcement voiced; outcome ``failed``."""
    async with reask_rig(tmp_path, script=STEP1) as rig:
        rig.llm.fail_reasks(FAILURES[name]())
        await run(rig, [answer(ANN_WEATHER)])
        sid = step_id(2)
        assert_one_reask(rig, sid, name)
        dones = rig.dones(sid)
        errors = [e for e in rig.machine.events(E.LlmError) if e.request_id == sid]
        hud_errors = rig.commands(C.HudError)
        sources = rig.spoken_sources()
    failed = notes(rig, NOTE_FAILED)
    assert len(failed) == 1, f"T-0172 B5 ({name}): one failure note; got {failed}"
    expected_type = {
        "transport": "ConnectError",
        "engine_empty": None,
        "over_long_413": "HTTPStatusError",
    }[name]
    values = json.dumps(failed[0])
    if expected_type is not None:
        assert expected_type in values, (
            f"T-0172 B5 (R-b): the note names {expected_type}; {failed[0]}"
        )
    for text in (ANN_WEATHER, REASK_NUDGE):
        assert text not in values, f"the failure note carries no text: {failed[0]}"
    assert [n for n in notes(rig, RETRY_NOTE) if n.get("request_id") == sid] == [], (
        f"T-0172 B5: no T-0171 retry of the re-ask; {notes(rig, RETRY_NOTE)}"
    )
    assert len(dones) == 1 and dones[0].tool is None, (
        f"T-0172 B5: one LlmDone without a tool; {dones}"
    )
    assert errors == [] and hud_errors == [], (
        f"T-0172 B5: never an llm.error from the re-ask alone; {errors} {hud_errors}"
    )
    assert sources.endswith(ANN_WEATHER), f"T-0172 B5: the announcement is voiced; {sources!r}"
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["failed"], outcome_notes(rig)
    assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "failed"}, turn_key(rig)


async def test_engine_empty_failure_type(tmp_path):
    """B5 (R-b), the empty-response case: the note names the type the real client raises for it (read off
    ``engine_empty_error``, so the row follows T-0171's type)."""
    expected = type(engine_empty_error("x")).__name__
    async with reask_rig(tmp_path, script=STEP1) as rig:
        rig.llm.fail_reasks(engine_empty_error)
        await run(rig, [answer(ANN_WEATHER)])
    failed = notes(rig, NOTE_FAILED)
    assert len(failed) == 1 and expected in json.dumps(failed[0]), (expected, failed)


async def test_trip_in_reask_ends_turn(tmp_path):
    """B5: a T-0142 trip inside the re-ask ends the turn the T-0142 way: the trip note for the re-ask's attempt, one
    ``LlmError(t1-gen-s2, DEGENERATE_MESSAGE)``, no ``LlmDone`` for the step, no step-2 call, loop end
    ``degenerate``."""
    message = t0142.degenerate_message()
    async with reask_rig(tmp_path, script=STEP1, extra={"degeneracy_guard": True}) as rig:
        await run(rig, [answer(ANN_CLICK)], reasks=[[thought(t0142.COLLAPSE), tool(CLICK2)]])
        sid = step_id(2)
        engine, _ = assert_one_reask(rig, sid, "trip")
        errors = [e.message for e in rig.machine.events(E.LlmError) if e.request_id == sid]
        dones = rig.dones(sid)
        extras = dict(rig.record("t1").extras)
    trips = t0142.notes(rig)
    assert [n.get("attempt_id") for n in trips] == [engine], (
        f"T-0172 B5: the trip note names the re-ask's attempt {engine}; {trips}"
    )
    assert errors == [message] and dones == [], (errors, dones)
    assert invoked(rig) == [(call_id(1), CLICK_ARGS)], invoked(rig)
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    # B-F4 / B5b: a trip inside the re-ask is a failed re-ask
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["failed"], outcome_notes(rig)
    assert extras.get(TURN_KEY) == {"on": True, "asked": 1, "outcome": "failed"}, extras.get(
        TURN_KEY
    )


# =====================================================================================================
# B6: abort, barge-in, disconnect and the timeout win (adversarial)
# =====================================================================================================


async def _until_reask_held(rig: Any, machine_id: str) -> None:
    await rig.until(
        lambda: bool(reasks_of(rig, machine_id)),
        f"T-0172 B1: the re-ask under {machine_id} was never sent (re-asks {rig.llm.reask_prompts})",
        limit_ms=10_000,
    )


@pytest.mark.parametrize("how", ["takeover_step_k", "disconnect_step_k", "takeover_step1"])
async def test_barge_in_during_reask(tmp_path, how):
    """B6 (adversarial): a takeover (with its onset) or a disconnect while the re-ask is in flight aborts the re-ask's
    engine request; when the engine would still answer (released afterwards) its call never executes, no further
    request is sent for the turn, the nudge is in no history entry; outcome ``aborted`` (R-d)."""
    step1 = how == "takeover_step1"
    script = [answer(ANN_LONG)] if step1 else STEP1
    async with reask_rig(tmp_path, script=script) as rig:
        rig.llm.script_steps([answer(ANN_CLICK)], [answer(S3)])
        rig.llm.script_reasks([tool(CLICK2 if not step1 else VALID_CLICK)])
        rig.llm.hold_reasks()
        await rig.ask(QUESTION)
        sid = step_id(1 if step1 else 2)
        await _until_reask_held(rig, sid)
        engine = reasks_of(rig, sid)[0][0]
        before = len(rig.llm.generates)
        now = rig.clock.now_ms
        if how == "disconnect_step_k":
            rig.submit(E.ClientDisconnected(t_ms=now))
        else:
            rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
            rig.submit(E.UserTakeover(t_ms=now + 1, onset_ms=now))
        await rig.until(
            lambda: rig.state.turn_id != "t1" or rig.state.state is TurnState.IDLE,
            "T-0172 B6: the turn never closed",
        )
        await rig.advance(500)
        rig.llm.release_reasks()
        await rig.advance(3_000)
        t1_after = [
            rid for rid, _, _ in rig.llm.generates[before:] if re.search(r"-t1-(gen|repair)", rid)
        ]
        history = history_json(rig)
    assert engine in rig.llm.aborted, (
        f"T-0172 B6: the re-ask's engine request {engine} is aborted; aborted {rig.llm.aborted}"
    )
    executed = [cid for cid, _ in invoked(rig)]
    banned = call_id(1) if step1 else call_id(2)
    assert banned not in executed, (
        f"T-0172 B6: no call from the aborted re-ask executes; {executed}"
    )
    assert t1_after == [], f"T-0172 B6: no further request for t1; {t1_after}"
    assert REASK_NUDGE not in history, "T-0172 B6: history never holds the nudge"
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["aborted"], outcome_notes(rig)
    assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "aborted"}, turn_key(rig)


async def test_timeout_during_reask(tmp_path):
    """B6 (adversarial): the re-ask shares the step's ``llm_timeout`` and never re-arms it. Step 2's request is held
    15 s of session time before its announcement arrives; the re-ask is then held: the turn fails with ``llm timeout``
    at the step's own deadline (LlmContinue + 20 s), not 20 s after the re-ask started; the re-ask is aborted, its call
    never executes, no step-3 request, outcome ``aborted``."""
    timeout = POLICY.llm_timeout_ms
    async with reask_rig(tmp_path, script=STEP1) as rig:
        rig.llm.script_steps([answer(ANN_CLICK)], [answer(S3)])
        rig.llm.script_reasks([tool(CLICK2)])
        rig.llm.hold_steps()
        rig.llm.hold_reasks()
        await rig.ask(QUESTION)
        await tsl.until_steps(rig, 1)
        armed = rig.machine.events(E.ToolResult)[-1].t_ms
        await rig.advance(15_000, step=500)
        rig.llm.release_steps()
        await _until_reask_held(rig, step_id(2))
        started = rig.clock.now_ms
        assert started - armed >= 15_000, "premise: the re-ask starts 15 s into the step"
        await rig.until(
            lambda: bool(rig.machine.events(E.LlmError)) or rig.state.state is TurnState.IDLE,
            "T-0172 B6: the shared llm_timeout never fired",
            limit_ms=30_000,
            step=100,
        )
        errors = rig.machine.events(E.LlmError)
        speaks_before = len(rig.speaks())
        engine = reasks_of(rig, step_id(2))[0][0]
        rig.llm.release_reasks()
        await rig.advance(3_000)
        fails = [c for c in rig.commands(C.HudError)]
    assert errors or rig.state.state is TurnState.IDLE, "premise: the turn ended"
    deadline = armed + timeout
    tick_end = next(
        (ev.t_ms for ev, st, cmds in rig.machine.log
         if any(isinstance(c, C.HudError) for c in cmds)),
        None,
    )  # fmt: skip
    assert tick_end is not None and deadline <= tick_end <= deadline + 200, (
        f"T-0172 B6: llm timeout at the step's deadline {deadline} (armed {armed}), never re-armed by the re-ask "
        f"(started {started}); the turn failed at {tick_end}"
    )
    assert any("timeout" in getattr(c, "message", "") for c in fails), fails
    assert engine in rig.llm.aborted, f"the re-ask {engine} is aborted; {rig.llm.aborted}"
    assert call_id(2) not in [cid for cid, _ in invoked(rig)], invoked(rig)
    assert tsl.generates_for(rig, step_id(3)) == [], "no step-3 request"
    assert len(rig.speaks()) == speaks_before, "nothing is voiced after the timeout"
    assert REASK_NUDGE not in history_json(rig)
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["aborted"], outcome_notes(rig)


# =====================================================================================================
# B5: once per turn (adversarial), and per turn rather than per session
# =====================================================================================================


@pytest.mark.parametrize("where", ["later_step", "reask_announces_again"])
async def test_once_per_turn_across_steps(tmp_path, where):
    """B5 (adversarial): one ``announce_reask`` per turn. ``later_step``: step 2 announces, its re-ask calls, step 3
    announces again with no call: no second re-ask, step 3's announcement is voiced. ``reask_announces_again``: the
    re-ask itself only announces again: no second re-ask, both are voiced."""
    async with reask_rig(tmp_path, script=STEP1) as rig:
        if where == "later_step":
            await run(
                rig,
                [answer(ANN_CLICK)],
                [answer(ANN_WEATHER)],
                reasks=[[answer(R_CALL), tool(CLICK2)], [answer("Second re-ask.")]],
            )
        else:
            await run(
                rig, [answer(ANN_WEATHER)], reasks=[[answer(R_AGAIN)], [answer("Second re-ask.")]]
            )
        sources = rig.spoken_sources()
    assert len(rig.llm.reask_prompts) == 1 and len(reask_requests(rig)) == 1, (
        f"T-0172 B5 ({where}): at most one re-ask per turn; the fake saw {[r for r, _ in rig.llm.reask_prompts]}"
    )
    assert "Second re-ask." not in sources
    if where == "later_step":
        assert sources.endswith(ANN_WEATHER), sources
        assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "call"}, turn_key(rig)
    else:
        assert ANN_WEATHER in sources and R_AGAIN in sources, sources
        assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "no_call"}, turn_key(rig)


async def test_once_per_turn_not_per_session(tmp_path):
    """B5 (too wide): the limit is per turn: two turns that each announce at step 1 get one re-ask each."""
    async with reask_rig(tmp_path, script=[answer(ANN_CLICK)]) as rig:
        rig.llm.script_reasks([answer("Okay.")], [answer("Okay again.")])
        await rig.ask(QUESTION)
        await rig.turn_done()
        await rig.advance(300)
        await rig.ask("and again please")
        await rig.until(lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "t2")
        per_turn = [len(reasks_of(rig, step_id(1, t))) for t in ("t1", "t2")]
        keys = [turn_key(rig, t) for t in ("t1", "t2")]
    assert per_turn == [1, 1], f"T-0172 B5: one re-ask in each turn; {per_turn}"
    assert keys == [{"on": True, "asked": 1, "outcome": "no_call"}] * 2, keys


# =====================================================================================================
# The nudge never bypasses confirmation (T-0107a ask-first, T-0107e honour-no) -- adversarial
# =====================================================================================================


@pytest.mark.parametrize("path", ["asks_first", "user_declined"])
async def test_reask_call_still_confirmed(tmp_path, path):
    """B4 (adversarial; the nudge asks for "that call now", it must not bypass T-0107): ``asks_first``: the re-ask
    asks the user before placing the order and drafts a valid click; the step's voiced text (announcement + question)
    asks, so the call is refused with T-0107a's reason and never executes; the question is voiced. ``user_declined``:
    the user's turn is a clear no; the re-ask's valid click is refused with T-0107e's reason and never executes. Both
    records carry ``after_reask: true``; outcome ``refused``. Power: the same re-ask call without the question executes
    (``test_step_k_call_withheld``)."""
    if path == "asks_first":
        question, script = QUESTION, STEP1
        steps: tuple[list[Commit], ...] = ([answer(ANN_ADD)],)
        reask = [answer(R_ASKS), tool(CLICK2)]
        cid, reason = call_id(2), ASKED_FIRST_REASON
        assert asks_user(ANN_ADD + " " + R_ASKS) and asks_user(ANN_ADD + R_ASKS), "premise"
    else:
        question, script = "no stop", [answer(ANN_REMOVE)]
        steps = ([answer(S3)],)
        reask = [tool(VALID_CLICK)]
        cid, reason = call_id(1), USER_DECLINED_REASON
        assert user_declines(question), "premise: the transcript is a clear no"
    async with reask_rig(tmp_path, script=script) as rig:
        await run(rig, *steps, reasks=[reask], question=question)
        assert len(rig.llm.reask_prompts) == 1, "premise: the re-ask was sent"
        sources = rig.spoken_sources()
    assert cid not in [c for c, _ in invoked(rig)], (
        f"T-0172 (T-0107): the re-ask's call is never executed on the {path} path; {invoked(rig)}"
    )
    recs = [r for r in rig.tool_records() if r.get("call_id") == cid]
    assert recs and recs[-1].get("status") == "refused" and recs[-1].get("reason") == reason, (
        f"T-0172 (T-0107): {cid} is refused with {reason!r}; records {recs}"
    )
    assert all(r.get(AFTER) is True for r in recs), recs
    if path == "asks_first":
        assert R_ASKS in sources, f"the question is voiced; {sources!r}"
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["refused"], outcome_notes(rig)


# =====================================================================================================
# B7: logging
# =====================================================================================================


async def test_turn_line_key(tmp_path):
    """B7: with the keyword on and tools on, every turn line carries ``announced_call_reask``: the re-asked turn
    ``{on: true, asked: 1, outcome: "call"}``, a later toolless turn ``{on: true, asked: 0, outcome: null}``; the
    value is a plain mapping in ``extras`` and nowhere else on the line."""
    async with reask_rig(tmp_path, script=STEP1) as rig:
        await run(rig, [answer(ANN_CLICK)], [answer(S3)], reasks=[[tool(CLICK2)]])
        rig.llm.script = [answer("It is a small shop.")]
        await rig.ask("what is on the page")
        await rig.until(lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "t2")
        lines = [json.loads(x) for x in rig.lines()]
        keys = [turn_key(rig, t) for t in ("t1", "t2")]
    assert keys == [
        {"on": True, "asked": 1, "outcome": "call"},
        {"on": True, "asked": 0, "outcome": None},
    ], keys
    for line in lines:
        rest = {k: v for k, v in line.items() if k != "extras"}
        assert TURN_KEY not in json.dumps(rest), (
            f"T-0172 B7: the key lives in extras only; {sorted(rest)}"
        )


async def test_notes_have_no_text(tmp_path):
    """B7 (.claude/rules/conductor.md: notes carry counts): the re-asked and outcome notes have exactly their keys,
    carry no announcement, re-ask or nudge text, and ``chars`` is an int; no new session-log kind and no new span
    beside a keyword-``None`` run of the same turn."""

    async def turn(tmp: Path, reask: Any) -> tuple[Any, Any]:
        async with reask_rig(tmp, reask=reask, script=STEP1) as rig:
            await run(
                rig, [answer(ANN_CLICK)], [answer(S3)], reasks=[[answer(R_CALL), tool(CLICK2)]]
            )
            rec = rig.record("t1")
        return rig, rec

    rig, rec = await turn(tmp_path / "on", True)
    base, base_rec = await turn(tmp_path / "none", None)
    asked, outcome = notes(rig, NOTE_ASKED), outcome_notes(rig)
    assert len(asked) == 1 and set(asked[0]) == ASKED_KEYS, asked
    assert len(outcome) == 1 and set(outcome[0]) == OUTCOME_KEYS, outcome
    assert outcome[0]["outcome"] in OUTCOMES and isinstance(asked[0]["chars"], int), (
        asked,
        outcome,
    )
    for note in (*asked, *outcome):
        text = json.dumps(note)
        for banned in (ANN_CLICK, R_CALL, REASK_NUDGE, "Sign in"):
            assert banned not in text, f"T-0172 B7: a note carries {banned!r}: {note}"
    assert rec.unknown_spans() == [] and set(rec.spans) == set(base_rec.spans), (
        f"T-0172 B7: no new span; {sorted(set(rec.spans) ^ set(base_rec.spans))}"
    )
    kinds = {r["kind"] for r in rig.log_records}
    base_kinds = {r["kind"] for r in base.log_records}
    # review fix: the keyword-None run happens to write no ``note`` (B7 requires notes), so the vocabulary is the
    # kinds that run wrote plus every kind BASE's session.py passes to ``_log`` (read at the literal BASE_SHA).
    # Release re-pin (James 2026-11-01): the history-free release (one squashed commit, T-0157 §Recipe) has no
    # BASE_SHA, so the log-kind half skips naming it (the note and span rows above still run).
    gr.require_commits(BASE_SHA, why=gr.NO_HISTORY)
    src = gr._git("show", f"{BASE_SHA}:local_astra/conductor/session.py").stdout
    base_vocab = set(re.findall(r"self\._log\(\s*\"([a-z_.]+)\"", src))
    assert "note" in base_vocab and "llm.request" in base_vocab, "premise: BASE writes these kinds"
    vocab = base_kinds | base_vocab
    assert kinds <= vocab, f"T-0172 B7: no new log kind; new {sorted(kinds - vocab)}"


# =====================================================================================================
# B8: the keyword
# =====================================================================================================


def _construct(tmp_path: Path, value: Any) -> str | None:
    """The message of the ``ValueError`` a tools-off ``Session(..., reask_announced_calls=value)`` raises, or
    ``None`` if it constructs."""
    from tests.fakes import FakeMemory, FakeStt
    from tests.unit._tool_turn_support import ToolLlm
    from tests.unit.test_session_loop import (
        RecordingAudio,
        RecordingMachine,
        RecordingPerception,
        RecordingTts,
        ScriptedClock,
        SessionConfig,
    )

    clock = ScriptedClock()
    try:
        session_cls()(
            RecordingMachine(POLICY),
            ToolLlm([]),
            FakeStt(),
            RecordingTts(),
            RecordingAudio(clock),
            RecordingPerception([]),
            FakeMemory(),
            clock=clock,
            config=SessionConfig(data_dir=tmp_path, turns_path=tmp_path / "turns.jsonl"),
            **{KEY: value},
        )
    except ValueError as exc:
        return str(exc)
    return None


def test_keyword_refusal(tmp_path):
    """B8: ``True`` without a tool kit raises a ``ValueError`` naming ``dry_run`` and ``live``; ``None`` and
    ``False`` construct (too-wide companion)."""
    require_keyword()
    msg = _construct(tmp_path / "true", True)
    assert msg is not None and "dry_run" in msg and "live" in msg, (
        f"T-0172 B8: tools off with {KEY}=True is refused naming dry_run and live; got {msg!r}"
    )
    for value in (None, False):
        (tmp_path / str(value)).mkdir()
        assert _construct(tmp_path / str(value), value) is None, value


# =====================================================================================================
# B12: off is BASE
# =====================================================================================================


def _drop_key(tree: Any, path: tuple[str, ...] = ()) -> tuple[Any, list[tuple[str, ...]]]:
    """``tree`` without any ``announced_call_reask`` key, and the paths it was removed from."""
    found: list[tuple[str, ...]] = []
    if isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            if k == TURN_KEY:
                found.append(path)
                continue
            out[k], sub = _drop_key(v, (*path, str(k)))
            found += sub
        return out, found
    if isinstance(tree, list):
        items = []
        for i, v in enumerate(tree):
            x, sub = _drop_key(v, (*path, str(i)))
            items.append(x)
            found += sub
        return items, found
    if isinstance(tree, str) and TURN_KEY in tree:
        found.append((*path, "<in a string>"))
    return tree, found


async def _capture(tmp: Path, reask: Any) -> dict[str, Any]:
    """Three turns over one session (``_PolicyMachine``, every section of T-0121's capture): the smoke's step-2
    announcement with no call (after a search), a step-1 announcement with no call, and an announcement followed by
    its call. Uses only names BASE's tree has when ``reask`` is ``ABSENT``."""
    tmp.mkdir(parents=True, exist_ok=True)
    search = FakeSearch()
    async with reask_rig(
        tmp,
        reask=reask,
        search=search,
        script=[answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)],
        machine_cls=_PolicyMachine,
    ) as rig:
        rig.llm.script_steps([thought("the snippets lack the temperature"), answer(ANN_WEATHER)])
        await rig.ask(QUESTION)
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 1, "turn 1"
        )
        await rig.advance(300)
        rig.llm.script = [answer(ANN_CLICK)]
        await rig.ask("click the sign in button")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "turn 2"
        )
        await rig.advance(300)
        rig.llm.script = [answer(S1), tool(VALID_CLICK)]
        rig.llm.script_steps([answer(REPORT)])
        await rig.ask("open the menu")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 3, "turn 3"
        )
        await rig.advance(300)
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out = _with_history(rig, tmp, session_id, log_path)
    for r in out["log"]:
        regime = r["data"].get("regime") if isinstance(r["data"], dict) else None
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    out["invocations"] = [[i.call_id, dict(i.arguments)] for i in rig.executor.invocations]
    out["queries"] = list(search.queries)
    return out


CAPTURED = (*SECTIONS, "invocations", "queries")


@pytest.mark.parametrize("reask", [None, False], ids=["none", "false"])
async def test_off_is_base(tmp_path, reask):
    """B12: with the keyword ``None`` or ``False`` the three turns of ``_capture`` (machine steps, HUD, prompts,
    request bodies, session-log records, turn lines, history, executor actions, search queries) equal the
    keyword-less session's, except the ``announced_call_reask`` key, which ``False`` adds to every turn line's extras
    (``{on: false, asked: 0, outcome: null}``) and ``None`` adds nowhere."""
    now = await _capture(tmp_path / "now", reask)
    base = await _capture(tmp_path / "base", ABSENT)
    clean, where = _drop_key(now)
    _, where_base = _drop_key(base)
    assert where_base == [], f"premise: no key without the keyword; {where_base}"
    if reask is None:
        assert where == [], f"T-0172 B12: keyword None adds no key; found at {where}"
    else:
        assert where and all(p[-1] == "extras" for p in where), (
            f"T-0172 B12: keyword False adds the key to turn-line extras only; found at {where}"
        )
        values = [r["extras"].get(TURN_KEY) for r in now["records"]]
        assert values == [{"on": False, "asked": 0, "outcome": None}] * 3, values
    for section in CAPTURED:
        _assert_equal(base[section], clean[section], f"T-0172 B12 keyword {reask!r} {section!r}:")


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0172_session_reask as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run", m.ABSENT))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


def test_absent_is_base_tree(tmp_path):
    """B12 (active, green at BASE and after): the keyword-less session (``Session``'s default) runs ``_capture``'s
    three turns exactly as BASE a7811bed's tree does (``local_astra`` from ``git archive``, origin-proved): machine
    steps, HUD, prompts, request bodies, session-log records, turn lines, history, executor actions, queries. An
    implementation that moves the default path (the decision before ``LlmDone``, a changed ``llm.done`` record, an
    extra request) fails here even if keyword ``None`` and absent agree with each other."""
    gr.require_commits(BASE_SHA)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp_path / "base_tree")
    out = tmp_path / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _BASE_RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA[:8]} tree"
    )
    base = data["sections"]
    # T-0177 B-P1 (pre-authorised pin move): the observed side without the published astra.vision.frame messages
    # and their wire.out records (shape checked); the BASE side is not normalised.
    now = without_vision_frames(asyncio.run(_capture(tmp_path / "now", ABSENT)))
    for section in CAPTURED:
        _assert_equal(
            base[section], now[section], f"T-0172 B12 keyword absent vs BASE {section!r}:"
        )
    assert base["queries"] == [QUERY_PREMISE] and any(
        ANN_WEATHER in json.dumps(s) for s in base["machine"]
    ), "premise: the scenario ran the smoke's announcement after a search"


QUERY_PREMISE = tsl.QUERY

#: B12, written out from the criterion: byte-identical over BASE..TIP
FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/thought_call.py",
    "tests/unit/test_turn_invariants.py",
)


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_frozen_paths_over_range():
    """B12 (active): the frozen paths are byte-identical over ``BASE_SHA..TIP_IMPL`` (two literals, never the working
    tree), no file added or removed, and docs/02 §4 (the invariants) is unedited between BASE and ``DOC_REV``. Power:
    the same predicate sees T-0171's ``llm_client.py`` change (its BASE 3e4d45b7 to this BASE)."""
    gr.require_commits(BASE_SHA, TIP_IMPL, DOC_REV, "3e4d45b7")
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:8]} is not an ancestor of {TIP_IMPL[:8]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    assert gr.changed(BASE_SHA, TIP_IMPL, *FROZEN, diff_filter="ADR") == []
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0172 B12: frozen paths differ in {BASE_SHA[:8]}..{TIP_IMPL[:8]}: {changed}"
    )
    from tests.unit import test_t0107a_docs as t7d

    s4_base = t7d.section(_doc02_at(DOC_PARENT), "## 4. ")
    s4_now = t7d.section(_doc02_at(DOC_REV), "## 4. ")
    assert s4_base and s4_now == s4_base, "T-0172 B12: the §4 invariants are unedited"
    assert "local_astra/conductor/llm_client.py" in gr.changed("3e4d45b7", BASE_SHA, *FROZEN), (
        "power"
    )


# =====================================================================================================
# B11: docs/02 §5
# =====================================================================================================

HEADER = "**Spec delta (T-0172"


def test_doc02_delta():
    """B11: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0172``, naming the rule and its mode,
    the census and its regime, the triggers, the prompt shape and prefix property, voicing at step 1 and >= 2, once
    per turn, the fallback, abort wins, the keys, the flag and the default; removing it leaves docs/02 equal to BASE's
    paragraph for paragraph. Keywords, never wording. Both sides are literals (``DOC_REV``, ``BASE_SHA``)."""
    from tests.unit import test_t0107a_docs as t7d

    gr.require_commits(BASE_SHA, DOC_REV)
    doc = _doc02_at(DOC_REV)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, f"T-0172 B11: one §5 paragraph starting {HEADER!r}; found {len(paras)}"
    flat = paras[0].replace("`", "")
    low = flat.lower()
    need = {
        "announces": "announces" in flat,
        "tail": re.search(r"\btail\b", low) is not None,
        "census (106)": "106" in flat,
        "regime": "regime" in low or "3e4d45b7" in flat,
        "announce_reask": "announce_reask" in flat,
        "reask_prompt / prefix": "reask_prompt" in flat and "prefix" in low,
        "REASK_NUDGE": "REASK_NUDGE" in flat,
        "step 1 and later steps": re.search(r"step 1", low) is not None,
        "once per turn": "once" in low and "turn" in low,
        "fallback": "fail" in low,
        "abort": "abort" in low or "barge" in low,
        "after_reask": "after_reask" in flat,
        TURN_KEY: TURN_KEY in flat,
        KEY: KEY in flat,
        "--announced-call-reask": "--announced-call-reask" in flat,
        "DEFAULT_ANNOUNCED_CALL_REASK": "DEFAULT_ANNOUNCED_CALL_REASK" in flat,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0172 B11: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    rest.remove(paras[0])
    gr.require_commits(DOC_PARENT)
    assert gr.is_ancestor(BASE_SHA, DOC_PARENT) and gr.is_ancestor(DOC_PARENT, DOC_REV)
    assert rest == t7d.paragraphs(_doc02_at(DOC_PARENT)), (
        "T-0172 B11: docs/02 changes only by the one delta paragraph (against DOC_PARENT, which holds T-0163's edit)"
    )


# =====================================================================================================
# Review rows (reviews A and B of e58c7ce1): B5b, B4a, B-F1, B-F2, B-F5
# =====================================================================================================

#: B5b: (step, re-ask stream before the error); the error comes after the stream (``fail_reasks_after_script``)
B5B_CASES = {
    "step_k-text": (2, [answer(R_CALL)]),
    "step_k-call": (2, [answer(R_CALL), tool(CLICK2)]),
    "step_1-text": (1, [answer(R_CALL)]),
    "step_1-call": (1, [answer(R_CALL), tool(VALID_CLICK)]),
}


@pytest.mark.parametrize("case", list(B5B_CASES))
async def test_failed_reask_voices_none_of_its_text(tmp_path, case):
    """B5b (amendment; reviews A-F1, B-F3): the re-ask commits text (and, in ``*-call``, a valid call), then the
    transport fails. Nothing of the re-ask's text is voiced, stored in history or written to ``turns.jsonl``; no call
    from it executes; outcome ``failed``; no ``llm.error``; at step >= 2 exactly the announcement is voiced (R15), at
    step 1 the announcement was already voiced and nothing follows it."""
    step, stream = B5B_CASES[case]
    script = [answer(ANN_CLICK)] if step == 1 else STEP1
    steps: tuple[list[Commit], ...] = () if step == 1 else ([answer(ANN_WEATHER)],)
    async with reask_rig(tmp_path, script=script) as rig:
        rig.llm.fail_reasks_after_script(
            httpx.RemoteProtocolError("peer closed connection mid-stream")
        )
        await run(rig, *steps, reasks=[stream])
        sid = step_id(step)
        assert_one_reask(rig, sid, case)
        sources = rig.spoken_sources()
        errors = [e for e in rig.machine.events(E.LlmError) if e.request_id == sid]
        history = history_json(rig)
        lines = "\n".join(rig.lines())
        key = turn_key(rig)
    assert R_CALL not in sources and R_CALL not in speaks_json(rig), (
        f"T-0172 B5b ({case}): the failed re-ask's text is voiced; sources {sources!r}"
    )
    assert R_CALL not in history and R_CALL not in lines, (
        f"T-0172 B5b ({case}): the failed re-ask's text is stored"
    )
    want_calls = [] if step == 1 else [(call_id(1), CLICK_ARGS)]
    assert invoked(rig) == want_calls, (
        f"T-0172 B5b ({case}): no call from the re-ask; {invoked(rig)}"
    )
    assert errors == [] and rig.commands(C.HudError) == [], (errors, rig.commands(C.HudError))
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["failed"], outcome_notes(rig)
    assert key == {"on": True, "asked": 1, "outcome": "failed"}, key
    if step == 1:
        assert sources == ANN_CLICK, (
            f"T-0172 B5b: the step-1 announcement and nothing else; {sources!r}"
        )
    else:
        assert sources in (S1 + ANN_WEATHER, S1 + " " + ANN_WEATHER), (
            f"T-0172 B5b: at step 2 exactly the announcement is voiced (R15); {sources!r}"
        )


#: B4a: (announcement commit, re-ask answer, the exact joined step text)
JOIN_CASES = {
    "no-space-either-side": (
        ANN_CLICK,
        "Okay, I could not find it.",
        "I will click the Sign in button. Okay, I could not find it.",
    ),
    "announcement-ends-in-space": (
        ANN_CLICK + " ",
        "Okay, I could not find it.",
        "I will click the Sign in button. Okay, I could not find it.",
    ),
}


@pytest.mark.parametrize("case", list(JOIN_CASES))
async def test_join_one_space(tmp_path, case):
    """B4a (amendment): the announcement and the re-ask's text are joined with exactly one space when neither side
    has whitespace at the join, and as they are otherwise. Voiced (a no-call re-ask at step 2) and in history the step
    reads exactly the joined text; with a call, step 3's prompt embeds exactly it as step 2's ``LoopStep.text``."""
    ann, r, want = JOIN_CASES[case]
    async with reask_rig(tmp_path / "voiced", script=STEP1) as rig:
        await run(rig, [answer(ann)], reasks=[[answer(r)]])
        sources = rig.spoken_sources()
        history = rig.state.history[-1].assistant_spoken
    assert sources in (S1 + want, S1 + " " + want), f"T-0172 B4a ({case}): voiced {sources!r}"
    assert history.endswith(want), f"T-0172 B4a ({case}): history {history!r}"
    async with reask_rig(tmp_path / "call", script=STEP1) as rig:
        await run(rig, [answer(ann)], [answer(S3)], reasks=[[answer(r), tool(CLICK2)]])
        _, pass1 = original(rig, step_id(1))
        step3 = tsl.step_request(rig, 3)
    expected = tsl.oracle(
        pass1, [(S1, "click_at", CLICK_ARGS, RECORDED), (want, "click_at", CLICK2_ARGS, RECORDED)]
    )
    assert step3 == expected, (
        f"T-0172 B4a ({case}): step 2's LoopStep.text is exactly {want!r}; tail {tsl.tail(step3)[-300:]!r}"
    )
    assert withheld(rig) == [(2, len(want))], withheld(rig)


@pytest.mark.parametrize("label", [True, False], ids=["labelled", "unlabelled-power"])
async def test_live_labelled_reask_prompt(tmp_path, label):
    """B2 (B-F1): live with the screen label on (``LiveFeedback(True, False)``), a step-2 announcement's re-ask prompt
    is ``reask_prompt(<the labelled step-2 prompt as sent>, step_text)``. Mutant power: the prompt the mutant
    ``reask_prompt(unlabelled, ...)`` builds (the sent prompt with its label part removed) differs from it, so a
    session that rebuilt the re-ask from the unlabelled prompt fails the equality. ``unlabelled-power``: with the
    label off the two coincide (the premise that the label is the only difference)."""
    from local_astra.contracts.llm import TextPart
    from tests.unit._live_browser_support import live_feedback

    extra = {"live_feedback": live_feedback(screen_label=True, page_extent=False)} if label else {}
    async with reask_rig(tmp_path, mode="live", script=STEP1, extra=extra) as rig:
        await run(rig, [answer(ANN_CLICK)], reasks=[[answer(R_STOPPED)]])
        sid = step_id(2)
        _, prompt = assert_one_reask(rig, sid, "live step 2")
        _, sent = original(rig, sid)
    stripped = sent.model_copy(
        update={
            "messages": tuple(
                m.model_copy(
                    update={
                        "content": tuple(
                            p
                            for p in m.content
                            if not (isinstance(p, TextPart) and "browser screen now" in p.text)
                        )
                    }
                )
                for m in sent.messages
            )
        }
    )
    want = reask_prompt(sent, step_text(ANN_CLICK))
    mutant = reask_prompt(stripped, step_text(ANN_CLICK))
    assert prompt == want, "T-0172 B2 (B-F1): the re-ask continues the labelled step prompt as sent"
    if label:
        assert "browser screen now" in json.dumps(sent.model_dump(mode="json")), "premise: labelled"
        assert prompt != mutant, "power: the unlabelled-rebuild mutant is told apart"
    else:
        assert stripped == sent and mutant == want, "premise: no label, nothing to strip"


@pytest.mark.parametrize("recover", [True, False], ids=["recovered", "recovery-off-power"])
async def test_recovered_call_inside_reask(tmp_path, recover):
    """B4 (B-F2): live with ``recover_thought_calls=True``, the re-ask answers and leaves a valid ``type_at`` at the
    tail of an open thought; the recovered call executes as step 2's call, every ``tool.call`` record of it carries
    ``from_thought: true`` and ``after_reask: true``, outcome ``call``. Power (the ``recovered=None`` mutant): with
    recovery off the same re-ask executes nothing and the outcome is ``no_call``."""
    browser = t0135.browser3()
    async with reask_rig(
        tmp_path,
        mode="live",
        browser=browser,
        script=STEP1,
        extra={"recover_thought_calls": recover},
    ) as rig:
        await run(
            rig, [answer(ANN_TYPE)], [answer(S3)], reasks=[[answer("Typing it now."), t0135.d5()]]
        )
        assert_one_reask(rig, step_id(2), "live step 2")
    # the session log is read back once the rig has closed (flushed)
    recs = [r for r in rig.tool_records() if r.get("call_id") == call_id(2)]
    outcomes = [n.get("outcome") for n in outcome_notes(rig)]
    executed = t0135.invocations(browser)
    if recover:
        assert (call_id(2), t0135.TYPE_ARGS) in executed, f"T-0172 B4 (B-F2): executed {executed}"
        assert recs and all(r.get("from_thought") is True and r.get(AFTER) is True for r in recs), (
            recs
        )
        assert outcomes == ["call"], outcomes
    else:
        assert call_id(2) not in [c for c, _ in executed] and outcomes == ["no_call"], (
            executed,
            outcomes,
        )


async def test_repairable_step1_reask(tmp_path):
    """B4/P5' (B-F5): a step-1 re-ask that drafts a repairable call is repaired as any step-1 draft: the repair
    request (``t1-repair``) carries no ``REASK_NUDGE``, the call executes after the repair with the repaired
    arguments, and the outcome is ``call``."""
    async with reask_rig(tmp_path, script=[answer(ANN_CLICK)]) as rig:
        rig.llm.script_continuation(tool(REPAIRED_CLICK))
        await run(rig, [answer(REPORT)], reasks=[[tool(REPAIRABLE_CLICK)]])
        assert_one_reask(rig, step_id(1), "step 1")
        repairs = [(rid, p) for rid, p, _, _ in rig.llm.continuations() if "-t1-repair-" in rid]
        repair_done = machine_index(
            rig, lambda ev: isinstance(ev, E.LlmDone) and ev.request_id == "t1-repair"
        )
        executes = [
            i
            for i, (_, _, cmds) in enumerate(rig.machine.log)
            if any(isinstance(c, C.ToolExecute) and c.call_id == call_id(1) for c in cmds)
        ]
    assert len(repairs) == 1, (
        f"T-0172 (B-F5): one repair request; {[r for r, _, _, _ in rig.llm.continuations()]}"
    )
    assert REASK_NUDGE not in json.dumps(repairs[0][1].model_dump(mode="json")), (
        "T-0172 (B-F5): the repair request never carries the nudge"
    )
    assert invoked(rig) == [(call_id(1), {"x": 12, "y": 5})], invoked(rig)
    assert repair_done and executes and repair_done[0] <= executes[0], (repair_done, executes)
    assert [n.get("outcome") for n in outcome_notes(rig)] == ["call"], outcome_notes(rig)
