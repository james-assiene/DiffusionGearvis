"""T-0176 (session side): Gearvis's own search limit -- 5 counted searches per question, 3 failed ones, the notice at
the cap, the refusal of the first search after withdrawal, the forced answer step, the plain wording
(docs/tasks/T-0176.md criteria 1-7, 9, 10).

Every row runs the T-0114 ``tool_rig`` with the loop on, a ``FakeSearch`` (T-0125's ``fail=``/``block=`` and T-0176's
additive ``outcomes=`` modes) and a kit with ``web_search="ddgs"``, built through ``wrig`` below (``live``: T-0111g's
``feedback_rig`` over ``FakeLiveBrowser``). Steps are scripted (``FakeLlm.script_steps``; T-0172's re-ask route
``script_reasks``). Expected values are written out by hand from the criteria (the cap reasons, the card reasons, the
counts) or rendered with T-0113's ``loop.search_payload``/``status_payload``/``render_response``/``step_prompt`` over
the fixture rows; nothing is computed with the code under test. The two limit notes' wording is the implementer's
(criterion 3 gives examples), so rows read ``session.SEARCH_LIMIT_NOTE``/``SEARCH_FAILED_LIMIT_NOTE`` and pin only
their bound and where they appear.

The failure mode the task names (James's desk session t7): after the cap the model kept drafting ``google_search``,
each refused, and the turn ended on the announcement of a call that never ran. The adversarial row is
``test_forced_step_search_ends_turn``: the forced-step search must be refused before execution and end the turn;
settling it through the execution path keeps the loop going and reddens the row.

Readings where the task is silent (listed in the test-author's report):

- "the turn ends at that step" is read as: no step ``k+1`` request, no ``llm.continue`` with ``step > k``, the turn
  line written; ``tool_loop_end`` is not pinned;
- the forced-step draft as the machine sees it is pinned as ``LlmDone.tool.status == "refused"`` and one
  ``tool.cancel(refused)``; its machine-side reason is not pinned (the card's is, criterion 7);
- a refused-at-cap search's note has a ``failures`` key (criterion 8 "every ``web search`` note"); its value is not
  pinned (no search ran);
- a reset is delivered as ``hud_server.route`` delivers it (``E.SessionReset`` on the pump, T-0162 A-5).

Every row but the two negative ones (``test_search_off_byte_identical``, criterion 10, and the premise row) is
pending for T-0176: at BASE a0badcd1 the cap is 3, the reasons say ``search failed``, and nothing withdraws search.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps unless a row
says 12), tools ``dry_run`` unless a row says ``live`` (1280x720 HUD frame, FakeLiveBrowser frames), n = 1,
deterministic. No measurement, no network.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import session as session_mod
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.protocols import SearchFailed
from local_astra.toolcall import loop, thought_call
from local_astra.toolcall.announced_call import announces
from tests.fakes.search import DEFAULT_ROWS, FakeSearch
from tests.unit import _tool_turn_support as tts
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_session_web_search as wsw
from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame, live_kit
from tests.unit._tool_turn_support import WatchingExecutor, answer, thought, tool

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "local_astra"

Q = '<|"|>'
PREFIX, REPORT = tsl.PREFIX, tsl.REPORT
UNTRUSTED_NOTE = wsw.UNTRUSTED_NOTE
#: criteria 5 and 7, written out (H-T176-1)
CAP_REASON = wsw.CAP_REASON
FAILED_CAP_REASON = wsw.FAILED_CAP_REASON
CAP_CARD = wsw.CAP_CARD
FAILED_CAP_CARD = wsw.FAILED_CAP_CARD
REASONS = wsw.REASONS
TIMEOUT_MS = wsw.TIMEOUT_MS
#: the t7 announcement, verbatim (logs/sessions/20261027-230926-966483.jsonl, the turn's last voiced sentence)
ANN_T7 = (
    "I will search for the latest volume of the Vinland Saga Deluxe edition available on Amazon."
)

#: twelve distinct queries, one per call
QN = tuple(f"vinland saga deluxe volume {k}" for k in range(1, 13))
search_call = wsw.search_call


def args(k: int) -> dict[str, Any]:
    return {"query": QN[k - 1]}


def search_steps(first: int, last: int) -> list[list[Any]]:
    """Steps ``first..last`` (each >= 2), each ``Step k.`` and a search for ``QN[k-1]``."""
    return [[answer(f"Step {k}."), tool(search_call(QN[k - 1]))] for k in range(first, last + 1)]


STEP1 = [answer(PREFIX), tool(search_call(QN[0]))]


# --------------------------------------------------------------------------------------------- the rig


@asynccontextmanager
async def wrig(
    tmp_path: Path,
    search: Any,
    *,
    steps: Sequence[Sequence[Any]],
    script: Sequence[Any] = tuple(STEP1),
    reasks: Sequence[Sequence[Any]] = (),
    policy: Any = tsl.LOOP,
    web_search: str | None = "ddgs",
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    """The dry-run session over the fakes with ``kit.search=search``, ``kit.web_search=web_search`` and
    ``Session(..., **session_kwargs)`` (a keyword the session does not take is a named failure)."""
    executor = WatchingExecutor()
    kit = wsw.web(T.make_kit(executor=executor), search, web_search)
    kwargs = dict(session_kwargs or {})
    real = tts.Session
    params = inspect.signature(real.__init__).parameters
    missing = [k for k in kwargs if k not in params]
    assert not missing, f"premise: Session(..., {', '.join(missing)}=...) does not exist"

    class WithKwargs(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in kwargs.items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    tts.Session = WithKwargs  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path, kit=kit, executor=executor, policy=policy, script=list(script)
        ) as rig:
            rig.llm.script_steps(*steps)
            if reasks:
                rig.llm.script_reasks(*reasks)
            yield rig
    finally:
        tts.Session = real  # type: ignore[misc]


@asynccontextmanager
async def live_wrig(
    tmp_path: Path,
    search: Any,
    *,
    steps: Sequence[Sequence[Any]],
    script: Sequence[Any] = tuple(STEP1),
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    browser = FakeLiveBrowser(
        [live_frame("onset")], post_frames=[live_frame(f"after-{i}") for i in range(1, 12)]
    )
    kit = wsw.web(live_kit(browser), search)
    async with feedback_rig(
        tmp_path, browser, script=list(script), session_kwargs=session_kwargs, kit=kit
    ) as rig:
        rig.llm.script_steps(*steps)
        yield rig


async def run_turn(
    rig: Any,
    requests: int,
    *,
    question: str = tsl.QUESTION,
    turn: str = "t1",
    limit_ms: int = 40_000,
    step: int = 50,
) -> None:
    """Ask; wait until ``requests`` step requests of ``turn`` were sent, then for the turn's end."""
    await rig.ask(question)
    if requests:
        await tsl.until_steps(rig, requests, turn=turn, limit_ms=limit_ms, step=step)
    await rig.turn_done(f"T-0176: the turn {turn} never ended", limit_ms=limit_ms)
    await rig.advance(300)


# --------------------------------------------------------------------------------------------- reading the rig


def limit_note() -> str:
    note = getattr(session_mod, "SEARCH_LIMIT_NOTE", None)
    assert isinstance(note, str) and note.strip(), (
        "T-0176 criterion 3: session.SEARCH_LIMIT_NOTE (a non-empty str module constant) does not exist yet"
    )
    assert len(note) <= 40, f"T-0176 criterion 3: at most 40 characters; {len(note)}: {note!r}"
    assert note != UNTRUSTED_NOTE
    return note


def failed_limit_note() -> str:
    note = getattr(session_mod, "SEARCH_FAILED_LIMIT_NOTE", None)
    assert isinstance(note, str) and note.strip(), (
        "T-0176 criterion 3: session.SEARCH_FAILED_LIMIT_NOTE (a non-empty str module constant) does not exist yet"
    )
    assert note != UNTRUSTED_NOTE
    return note


def executed_payload(
    k: int, *, rows: Sequence[Any] = DEFAULT_ROWS, limit: str | None = None
) -> Any:
    """Search ``k``'s executed payload: ``search_payload`` over the fixture rows, the untrusted note (T-0125
    criterion 6) and, for the search that reaches the cap, ``limit`` (T-0176 criterion 3)."""
    payload = {**loop.search_payload(wsw.results_of(rows, QN[k - 1])), "note": UNTRUSTED_NOTE}
    if limit is not None:
        payload["limit"] = limit
    return payload


def rendered(payload: Any) -> str:
    return loop.render_response("google_search", payload)


def response(rig: Any, k: int, turn: str = "t1") -> str:
    return wsw.response_of(rig, tsl.call_id(k, turn))


def status(rig: Any, k: int, turn: str = "t1") -> str:
    return wsw.result_status(rig, tsl.call_id(k, turn))


def card(rig: Any, k: int, turn: str = "t1") -> Any:
    return wsw.card_text(rig, tsl.call_id(k, turn))


def note_statuses(rig: Any) -> list[str]:
    return [n["status"] for n in wsw.web_notes(rig)]


def assert_refused_at_execution(rig: Any, k: int, reason: str, card_reason: str) -> None:
    """Criterion 4: the call settled ``refused`` through the execution path with ``status_payload("refused",
    reason)``; its card is refused with ``card_reason``."""
    assert status(rig, k) == "refused", f"T-0176 criterion 4: search {k} settles refused"
    want = rendered(loop.status_payload("refused", reason))
    got = response(rig, k)
    assert got == want, (
        f"T-0176 criteria 4/5: search {k}'s response\n  want {want!r}\n  got  {got!r}"
    )
    c = card(rig, k)
    assert (c.status, c.reason) == ("refused", card_reason), (
        f"T-0176 criterion 7: search {k}'s card; {c}"
    )


def assert_forced_refusal(rig: Any, k: int, card_reason: str, what: str) -> None:
    """Criterion 6: step ``k``'s ``google_search`` draft reached the machine refused (R15), was never executed,
    has no result, and the turn ended at step ``k``."""
    cid, sid = tsl.call_id(k), tsl.step_id(k)
    dones = rig.dones(sid)
    assert len(dones) == 1 and dones[0].tool is not None, (
        f"T-0176 criterion 6 ({what}): one LlmDone({sid}) carrying the draft; got {dones}"
    )
    assert dones[0].tool.status == "refused", (
        f"T-0176 criterion 6 ({what}): the machine sees a refused draft; got {dones[0].tool}"
    )
    executes = [c for c in rig.commands(C.ToolExecute) if c.call_id == cid]
    assert executes == [], f"T-0176 criterion 6 ({what}): no tool.execute for {cid}; {executes}"
    cancels = [c.reason for c in rig.commands(C.ToolCancel) if c.call_id == cid]
    assert cancels == ["refused"], (
        f"T-0176 criterion 6 ({what}): R15's tool.cancel(refused) for {cid}; {cancels}"
    )
    assert tsl.results_for(rig, cid) == [], f"T-0176 criterion 6 ({what}): no ToolResult for {cid}"
    records = [r for r in rig.kind("tool.result") if r.get("call_id") == cid]
    assert records == [], f"T-0176 criterion 6 ({what}): no tool.result record for {cid}; {records}"
    later = [c for c in rig.commands(C.LlmContinue) if c.step > k]
    assert later == [], f"T-0176 criterion 6 ({what}): no llm.continue after step {k}; {later}"
    nxt = tsl.generates_for(rig, tsl.step_id(k + 1))
    assert nxt == [], (
        f"T-0176 criterion 6 ({what}): the turn ends at step {k}; a {tsl.step_id(k + 1)} request was sent"
    )
    c = card(rig, k)
    assert (c.status, c.reason) == ("refused", card_reason), (
        f"T-0176 criterion 7 ({what}): the forced-step card; {c}"
    )
    mine = [n for n in wsw.web_notes(rig) if n.get("call_id") == cid]
    assert [(n["status"], n["step"]) for n in mine] == [("refused", k)], (
        f"T-0176 criterion 9 ({what}): one refused web search note for {cid}; {mine}"
    )
    assert "failures" in mine[0], (
        f"T-0176 criterion 8: every web search note has failures; {mine[0]}"
    )


# =====================================================================================================
# Premise (active)
# =====================================================================================================


def test_fixture_premises() -> None:
    """The t7 announcement is an announcement (T-0172's re-ask fires on it) and the thought-call fixture is a
    recoverable tail call (T-0135)."""
    assert announces(ANN_T7) is not None and announces(ANN_T7).verb == "search"  # type: ignore[union-attr]
    call = search_call(QN[6])
    assert thought_call.recover(f"I should check once more. {call}") == call
    assert len(set(QN)) == len(QN)


# =====================================================================================================
# Criteria 1, 3, 4, 7, 9: five counted, the notice, the sixth refused
# =====================================================================================================


async def test_five_counted_then_sixth_refused(tmp_path: Path) -> None:
    """Five executed searches; the sixth never reaches the backend, gets ``SEARCH_CAP_REASON`` and the Gearvis card;
    its step text stays withheld and the loop continues to step 7 (criterion 4); 5 ``executed`` notes, one
    ``refused``, ``searches == 6`` (criterion 9)."""
    search = FakeSearch()
    async with wrig(tmp_path, search, steps=[*search_steps(2, 6), [answer(REPORT)]]) as rig:
        await run_turn(rig, 6)
    assert search.queries == list(QN[:5]), (
        f"T-0176 criteria 1/4: five searches reach the backend, the sixth never; {search.queries}"
    )
    for k in range(1, 6):
        assert status(rig, k) == "executed", f"search {k}"
    assert_refused_at_execution(rig, 6, CAP_REASON, CAP_CARD)
    assert tsl.generates_for(rig, tsl.step_id(7)), (
        "T-0176 criterion 4: the loop continues to step 7"
    )
    held = [n.get("step") for n in tsl.notes(rig, "step text withheld")]
    assert 6 in held, f"T-0176 criterion 4 (R15 unchanged): step 6's text stays withheld; {held}"
    assert note_statuses(rig) == ["executed"] * 5 + ["refused"], note_statuses(rig)
    assert rig.record("t1").extras.get("searches") == 6, rig.record("t1").extras
    assert REPORT in rig.spoken_sources(), "premise: the step-7 answer is voiced"


async def test_fifth_response_carries_limit_note(tmp_path: Path) -> None:
    """Criterion 3: search 5's rendered response is ``search_payload`` + the untrusted note + ``limit`` =
    ``SEARCH_LIMIT_NOTE``; searches 1-4 and the refused sixth carry no ``limit`` key."""
    note = limit_note()
    search = FakeSearch()
    async with wrig(tmp_path, search, steps=[*search_steps(2, 6), [answer(REPORT)]]) as rig:
        await run_turn(rig, 6)
    for k in range(1, 5):
        want = rendered(executed_payload(k))
        assert response(rig, k) == want, (
            f"T-0176 criterion 3: search {k} gains no key\n  want {want!r}\n  got  {response(rig, k)!r}"
        )
        assert "limit:" not in response(rig, k)
    want5 = rendered(executed_payload(5, limit=note))
    assert response(rig, 5) == want5, (
        f"T-0176 criterion 3: search 5 carries the limit note\n  want {want5!r}\n  got  {response(rig, 5)!r}"
    )
    assert f"limit:{Q}{note}{Q}" in response(rig, 5)
    assert "limit:" not in response(rig, 6), "T-0176 criterion 3: no other response gains the key"
    tail6 = tsl.tail(tsl.step_request(rig, 6))
    assert f"limit:{Q}{note}{Q}" in tail6, "the step-6 request carries search 5's notice"


async def test_limit_and_untrusted_notes_uncut_at_worst_case_bounds(tmp_path: Path) -> None:
    """Criterion 3, last item: T-0125's worst-case rig (5 results of 200/500/600-char title/URL/snippet, a 500-char
    query) run as search 5: the untrusted note and the limit note both appear verbatim in the rendered response."""
    note = limit_note()
    query = "q" * 500
    rows = [
        {
            "title": f"{i}" + "t" * 199,
            "url": f"https://h{i}.example/" + "u" * 481,
            "snippet": "s" * 600,
        }
        for i in range(5)
    ]
    assert all(len(r["url"]) == 500 and len(r["title"]) == 200 for r in rows)
    search = FakeSearch(outcomes=[None, None, None, None, rows])
    steps = [*search_steps(2, 4), [answer("Step 5."), tool(search_call(query))], [answer(REPORT)]]
    async with wrig(tmp_path, search, steps=steps) as rig:
        await run_turn(rig, 5)
        tail = tsl.tail(tsl.step_request(rig, 6))
    block = response(rig, 5)
    for what, text in (("untrusted", UNTRUSTED_NOTE), ("limit", note)):
        assert Q + text + Q in block, (
            f"T-0176 criterion 3: the {what} note is uncut at the worst-case bounds (shorten the limit note, never "
            f"the untrusted note or MAX_RESPONSE); {block[-300:]!r}"
        )
        assert Q + text + Q in tail
    assert len(block) <= loop.MAX_RESPONSE


async def test_zero_result_search_counts(tmp_path: Path) -> None:
    """Criterion 1: a search that returned zero rows counts. Five empty searches; the sixth is refused."""
    search = FakeSearch(default=())
    async with wrig(tmp_path, search, steps=[*search_steps(2, 6), [answer(REPORT)]]) as rig:
        await run_turn(rig, 6)
    assert search.queries == list(QN[:5]), search.queries
    assert response(rig, 5) == rendered(executed_payload(5, rows=(), limit=limit_note()))
    assert_refused_at_execution(rig, 6, CAP_REASON, CAP_CARD)


# =====================================================================================================
# Criterion 6: the forced answer step (adversarial: the t7 failure)
# =====================================================================================================


async def test_forced_step_search_ends_turn(tmp_path: Path) -> None:
    """The t7 failure. Searches 1-5 run; step 6's search is refused at execution (criterion 4); step 7 (the forced
    answer step) searches again: no backend call, no ``tool.execute``/``tool.result``/``llm.continue`` for it, no
    step-8 request, a refused card with the Gearvis reason and one note. The backend saw exactly five queries.

    Mutation target: settling the forced-step search through the execution path (as the sixth was) keeps the turn
    looping into step 8."""
    search = FakeSearch()
    steps = [*search_steps(2, 7), [answer(REPORT)]]
    async with wrig(tmp_path, search, steps=steps) as rig:
        await run_turn(rig, 6)
    assert search.queries == list(QN[:5]), (
        f"T-0176 criterion 6: the backend saw exactly five queries; {search.queries}"
    )
    assert_refused_at_execution(rig, 6, CAP_REASON, CAP_CARD)
    assert_forced_refusal(rig, 7, CAP_CARD, "step 7")
    assert note_statuses(rig) == ["executed"] * 5 + ["refused", "refused"], note_statuses(rig)
    assert rig.record("t1").extras.get("searches") == 7, (
        f"T-0176 criterion 9: both refused-at-cap searches count; {rig.record('t1').extras}"
    )
    assert REPORT not in rig.spoken_sources(), "premise: step 8's script was never asked for"


async def test_forced_step_reask_search_also_refused(tmp_path: Path) -> None:
    """Criterion 6 (the T-0172 re-ask): the forced step only announces a search (the t7 sentence); the re-ask yields
    ``google_search``, which is refused before execution, and the turn ends at step 7."""
    search = FakeSearch()
    steps = [*search_steps(2, 6), [answer(ANN_T7)], [answer(REPORT)]]
    reasks = [[answer("Searching now."), tool(search_call(QN[6]))]]
    async with wrig(
        tmp_path,
        search,
        steps=steps,
        reasks=reasks,
        session_kwargs={"reask_announced_calls": True},
    ) as rig:
        await run_turn(rig, 6)
        sent_reasks = [rid for rid, _ in rig.llm.reask_prompts]
    assert len(sent_reasks) == 1 and tsl.step_id(7) in sent_reasks[0], (
        f"premise (T-0172 unchanged): one re-ask under {tsl.step_id(7)}; {sent_reasks}"
    )
    assert search.queries == list(QN[:5]), search.queries
    assert_forced_refusal(rig, 7, CAP_CARD, "re-ask")


async def test_forced_step_recovered_thought_search_also_refused(tmp_path: Path) -> None:
    """Criterion 6 (thought-call recovery, T-0135, live): at the forced step the only commit is an open thought ending
    in a ``google_search`` call; the recovered call is refused before execution and the turn ends at step 7."""
    search = FakeSearch()
    recovered = thought(f"I should check once more. {search_call(QN[6])}", open_=True)
    steps = [*search_steps(2, 6), [recovered], [answer(REPORT)]]
    async with live_wrig(
        tmp_path, search, steps=steps, session_kwargs={"recover_thought_calls": True}
    ) as rig:
        await run_turn(rig, 6)
        browser = rig.browser
    assert search.queries == list(QN[:5]), search.queries
    assert all(i.call_id != tsl.call_id(7) for i in browser.invocations), (
        f"T-0176 criterion 6: the recovered search never reaches the executor; {browser.invocations}"
    )
    assert_forced_refusal(rig, 7, CAP_CARD, "thought-call recovery")


async def test_forced_step_non_search_call_runs(tmp_path: Path) -> None:
    """Criterion 6: a ``click_at`` at the forced step (live) executes as today (one invocation, an executed card, a
    step-8 request); the ``google_search`` drafted at step 8 is refused before execution and ends the turn."""
    search = FakeSearch()
    steps = [
        *search_steps(2, 6),
        [answer("Step 7."), tool(tsl.CLICK2)],
        [answer("Step 8."), tool(search_call(QN[7]))],
        [answer(REPORT)],
    ]
    async with live_wrig(tmp_path, search, steps=steps) as rig:
        await run_turn(rig, 7)
        browser = rig.browser
    assert search.queries == list(QN[:5]), search.queries
    clicks = [i for i in browser.invocations if i.call_id == tsl.call_id(7)]
    assert [(i.name, dict(i.arguments)) for i in clicks] == [("click_at", tsl.CLICK2_ARGS)], (
        f"T-0176 criterion 6: the forced step's click runs; {browser.invocations}"
    )
    assert status(rig, 7) == "executed", "T-0176 criterion 6: the click's result is as today"
    assert card(rig, 7).status == "executed", card(rig, 7)
    assert tsl.generates_for(rig, tsl.step_id(8)), "the loop continues after the click"
    assert all(i.call_id != tsl.call_id(8) for i in browser.invocations), browser.invocations
    assert_forced_refusal(rig, 8, CAP_CARD, "step 8 after a click")


# =====================================================================================================
# Criterion 1: failures outside the cap, the failure bound
# =====================================================================================================


async def test_failed_searches_do_not_count(tmp_path: Path) -> None:
    """Criterion 1 (H-T176-2): two ``SearchFailed("error")``, then five executed searches (the fifth carries the
    limit note); the 8th call is refused with ``SEARCH_CAP_REASON``. Too wide (failures counted) refuses the 6th or
    7th; absent lets the 8th run."""
    note = limit_note()
    search = FakeSearch(outcomes=[SearchFailed("error", "x"), SearchFailed("error", "x")])
    steps = [*search_steps(2, 8), [answer(REPORT)]]
    async with wrig(tmp_path, search, steps=steps, policy=tsl.loop_policy(12)) as rig:
        await run_turn(rig, 8)
    assert search.queries == list(QN[:7]), search.queries
    for k in (1, 2):
        assert status(rig, k) == "failed"
        assert response(rig, k) == rendered(loop.status_payload("failed", REASONS["error"])), (
            f"T-0176 criterion 3: a failure below the bound gains no key; {response(rig, k)!r}"
        )
    for k in range(3, 7):
        assert response(rig, k) == rendered(executed_payload(k)), f"search {k}"
    assert response(rig, 7) == rendered(executed_payload(7, limit=note)), (
        f"T-0176 criterion 3: the fifth counted search carries the notice; {response(rig, 7)!r}"
    )
    assert_refused_at_execution(rig, 8, CAP_REASON, CAP_CARD)
    assert note_statuses(rig) == ["failed", "failed"] + ["executed"] * 5 + ["refused"]


async def test_three_failures_withdraw(tmp_path: Path) -> None:
    """Criterion 1/3/4/7: failures ``error``, ``rate_limited`` and a timeout via the session deadline (10 s, a backend
    that never returns). The third failure's response carries ``SEARCH_FAILED_LIMIT_NOTE``; the 4th call gets
    ``SEARCH_FAILED_CAP_REASON`` without a backend call; the forced step-5 search is refused and ends the turn; both
    cards say ``Gearvis limit: 3 failed searches per question``."""
    note = failed_limit_note()
    search = FakeSearch(
        outcomes=[SearchFailed("error", "x"), SearchFailed("rate_limited", "x"), asyncio.Event()]
    )
    steps = [*search_steps(2, 5), [answer(REPORT)]]
    async with wrig(tmp_path, search, steps=steps) as rig:
        await run_turn(rig, 4, limit_ms=60_000, step=250)
    assert search.queries == list(QN[:3]), (
        f"T-0176 criterion 4: the 4th search never reaches the backend; {search.queries}"
    )
    assert response(rig, 1) == rendered(loop.status_payload("failed", REASONS["error"]))
    assert response(rig, 2) == rendered(loop.status_payload("failed", REASONS["rate_limited"]))
    assert status(rig, 3) == "timeout", "premise: the session deadline settled search 3"
    want3 = rendered(
        {**loop.status_payload("timeout", f"no result within {TIMEOUT_MS} ms"), "limit": note}
    )
    assert response(rig, 3) == want3, (
        f"T-0176 criterion 3: the third failure carries the failed-limit notice\n  want {want3!r}\n"
        f"  got  {response(rig, 3)!r}"
    )
    assert_refused_at_execution(rig, 4, FAILED_CAP_REASON, FAILED_CAP_CARD)
    assert_forced_refusal(rig, 5, FAILED_CAP_CARD, "failure bound")
    assert note_statuses(rig) == ["failed", "failed", "timeout", "refused", "refused"]


async def test_failure_bound_after_counted_searches(tmp_path: Path) -> None:
    """Criterion 2: either bound withdraws, whichever comes first. executed, error, executed, error, error: the 6th
    call is refused with ``SEARCH_FAILED_CAP_REASON`` though only two searches counted; the 5th response (the third
    failure) carries the failed-limit notice and no counted response carries ``limit``."""
    note = failed_limit_note()
    err = SearchFailed("error", "x")
    search = FakeSearch(outcomes=[None, err, None, err, err])
    async with wrig(tmp_path, search, steps=[*search_steps(2, 6), [answer(REPORT)]]) as rig:
        await run_turn(rig, 6)
    assert search.queries == list(QN[:5]), search.queries
    assert response(rig, 1) == rendered(executed_payload(1))
    assert response(rig, 3) == rendered(executed_payload(3))
    assert response(rig, 5) == rendered(
        {**loop.status_payload("failed", REASONS["error"]), "limit": note}
    ), response(rig, 5)
    assert "limit:" not in response(rig, 4)
    assert_refused_at_execution(rig, 6, FAILED_CAP_REASON, FAILED_CAP_CARD)


# =====================================================================================================
# Criterion 1 (reset), H-T176-5
# =====================================================================================================


async def _second_turn(rig: Any, lines: int) -> str:
    rig.llm.script = [answer(PREFIX), tool(search_call(QN[6]))]
    rig.llm.script_steps(
        *[[answer(f"Again {k}."), tool(search_call(QN[5 + k]))] for k in range(2, 7)],
        [answer(REPORT)],
    )
    await rig.ask("and the next volume")
    await rig.until(
        lambda: len(rig.lines()) == lines and rig.state.state.name == "IDLE",
        "T-0176: the second turn never ended",
        limit_ms=40_000,
    )
    await rig.advance(300)
    return str(rig.records()[-1].turn_id)


@pytest.mark.parametrize(
    "case", ["next_turn", "reset_after_turn", "reset_mid_turn", "failed_next_turn"]
)
async def test_withdrawal_resets_per_turn_and_on_reset(tmp_path: Path, case: str) -> None:
    """Criterion 1 (both counts reset per turn and on session reset) and H-T176-5: after a turn that reached a bound
    (or a reset that cut a turn holding four counted searches and a fifth in flight), the next turn has exactly five
    again: its sixth search is refused with ``SEARCH_CAP_REASON``."""
    block = asyncio.Event()
    if case == "failed_next_turn":
        err = SearchFailed("error", "x")
        search = FakeSearch(outcomes=[err, err, err])
        steps1: list[list[Any]] = [*search_steps(2, 4), [answer(REPORT)]]
    elif case == "reset_mid_turn":
        search = FakeSearch(outcomes=[None, None, None, None, block])
        steps1 = search_steps(2, 5)
    else:
        search = FakeSearch()
        steps1 = [*search_steps(2, 6), [answer(REPORT)]]
    async with wrig(tmp_path, search, steps=steps1) as rig:
        if case == "reset_mid_turn":
            await rig.ask(tsl.QUESTION)
            await rig.until(lambda: len(search.queries) == 5, "premise: search 5 is in flight")
            now = rig.clock.now_ms
            rig.session.submit(E.SessionReset(t_ms=now))
            await rig.pump()
            await rig.advance(200)
            await rig.until(lambda: len(rig.lines()) == 1, "the reset-cut turn wrote its line")
        else:
            await run_turn(rig, len(steps1))
            if case == "reset_after_turn":
                rig.session.submit(E.SessionReset(t_ms=rig.clock.now_ms))
                await rig.pump()
                await rig.advance(200)
        first = list(search.queries)
        tid = await _second_turn(rig, 2)
    assert first == list(QN[: {"failed_next_turn": 3}.get(case, 5)]), f"premise [{case}]: {first}"
    second = search.queries[len(first) :]
    assert second == list(QN[6:11]), (
        f"T-0176 criterion 1 [{case}]: the next turn gets a fresh five; it searched {second}"
    )
    k6 = tsl.call_id(6, tid)
    assert wsw.result_status(rig, k6) == "refused", f"[{case}] the next turn's sixth is refused"
    assert wsw.response_of(rig, k6) == rendered(loop.status_payload("refused", CAP_REASON)), (
        f"T-0176 [{case}]: the next turn's cap is the counted one, at five; {wsw.response_of(rig, k6)!r}"
    )


# =====================================================================================================
# Criterion 10 / H-T176-3: the prefix cache
# =====================================================================================================


async def test_tools_field_and_system_turn_identical_across_steps(tmp_path: Path) -> None:
    """Criterion 10 / implementation note 1: every request of the turn, before and after withdrawal (steps 1-7),
    has one ``tools`` value, one system message and one set of template kwargs. Premise: withdrawal happened (the
    sixth search refused with the Gearvis reason), so the row is not vacuous."""
    search = FakeSearch()
    async with wrig(tmp_path, search, steps=[*search_steps(2, 7), [answer(REPORT)]]) as rig:
        await run_turn(rig, 6)
        bodies = list(rig.llm.bodies)
    assert_refused_at_execution(rig, 6, CAP_REASON, CAP_CARD)
    assert len(bodies) == 7, f"premise: steps 1-7, one request each; {len(bodies)}"
    tools = {json.dumps(b.get("tools"), sort_keys=True) for b in bodies}
    systems = {json.dumps(b["messages"][0], sort_keys=True) for b in bodies}
    kwargs = {json.dumps(b.get("chat_template_kwargs"), sort_keys=True) for b in bodies}
    assert (len(tools), len(systems), len(kwargs)) == (1, 1, 1), (
        f"T-0176 criterion 10: one tools value, system message and template kwargs across the turn; "
        f"{len(tools)}, {len(systems)}, {len(kwargs)}"
    )
    assert '"google_search"' in next(iter(tools)), "the declaration is never removed (note 1)"


async def test_forced_step_body_is_todays_step_prompt(tmp_path: Path) -> None:
    """Criterion 6 / H-T176-3: the forced step's request is ``loop.step_prompt(<step-1 prompt as sent>, steps 1-6)``
    with the payloads written out by hand (four plain executed searches, the fifth with ``limit``, the sixth refused
    with ``SEARCH_CAP_REASON``): no new prompt text."""
    note = limit_note()
    search = FakeSearch()
    async with wrig(tmp_path, search, steps=[*search_steps(2, 7), [answer(REPORT)]]) as rig:
        await run_turn(rig, 6)
        pass1 = tsl.sent(rig, "t1-gen")
        forced = tsl.step_request(rig, 7)
        step_bodies = list(rig.llm.step_bodies)
    loop_steps = [(PREFIX, "google_search", args(1), executed_payload(1))]
    loop_steps += [(f"Step {k}.", "google_search", args(k), executed_payload(k)) for k in (2, 3, 4)]
    loop_steps.append(("Step 5.", "google_search", args(5), executed_payload(5, limit=note)))
    loop_steps.append(
        ("Step 6.", "google_search", args(6), loop.status_payload("refused", CAP_REASON))
    )
    want = tsl.oracle(pass1, loop_steps)
    assert tsl.tail(forced) == tsl.tail(want), (
        f"T-0176 criterion 6: the forced step's text\n  want {tsl.tail(want)[-600:]!r}\n"
        f"  got  {tsl.tail(forced)[-600:]!r}"
    )
    assert forced == want, "T-0176 criterion 6: every other prompt field is the step-1 prompt's"
    assert len(step_bodies) == 6 and step_bodies[-1].get("continue_final_message") is True


# =====================================================================================================
# Criteria 5, 7: wording
# =====================================================================================================


def _assigned(name: str) -> ast.expr:
    tree = ast.parse(Path(session_mod.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    raise AssertionError(f"T-0176 criterion 5: session.py assigns no module-level {name}")


def _rebuilt(name: str, **override: Any) -> Any:
    """``name``'s module-level expression evaluated over the module's namespace with ``override`` applied: what the
    constant would read if the numbers it is built from changed."""
    ns = dict(vars(session_mod))
    ns.update(override)
    return eval(compile(ast.Expression(_assigned(name)), "<session.py>", "eval"), ns)


def test_wording_constants() -> None:
    """Criterion 5: the exact strings; neither starts with ``search failed``; each contains its number and is built
    from the constant (rebuilt with the number changed, the text changes with it). Criterion 7: ``search limit
    reached`` no longer appears anywhere under ``local_astra/`` (prose included)."""
    got = {
        n: getattr(session_mod, n, None)
        for n in (
            "SEARCH_CAP_REASON",
            "SEARCH_FAILED_CAP_REASON",
            "MAX_SEARCHES_PER_TURN",
            "MAX_FAILED_SEARCHES_PER_TURN",
        )
    }
    assert got == {
        "SEARCH_CAP_REASON": CAP_REASON,
        "SEARCH_FAILED_CAP_REASON": FAILED_CAP_REASON,
        "MAX_SEARCHES_PER_TURN": 5,
        "MAX_FAILED_SEARCHES_PER_TURN": 3,
    }, f"T-0176 criteria 1/5: {got}"
    for reason in (CAP_REASON, FAILED_CAP_REASON):
        assert not reason.lower().startswith("search failed"), reason
    assert "5" in CAP_REASON and "3" in FAILED_CAP_REASON
    assert _rebuilt("SEARCH_CAP_REASON") == CAP_REASON, "premise: the expression rebuilds the value"
    assert _rebuilt("SEARCH_CAP_REASON", MAX_SEARCHES_PER_TURN=7) == CAP_REASON.replace("5", "7"), (
        "T-0176 criterion 5: SEARCH_CAP_REASON is built from MAX_SEARCHES_PER_TURN"
    )
    assert _rebuilt("SEARCH_FAILED_CAP_REASON", MAX_FAILED_SEARCHES_PER_TURN=7) == (
        FAILED_CAP_REASON.replace("3", "7")
    ), "T-0176 criterion 5: SEARCH_FAILED_CAP_REASON is built from MAX_FAILED_SEARCHES_PER_TURN"
    hits = []
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "search limit reached" in text:
                hits.append(str(path.relative_to(REPO)))
    assert hits == [], (
        f"T-0176 criterion 7: 'search limit reached' still appears (code or prose) in {hits}"
    )


# =====================================================================================================
# Criterion 10: search off, byte-identical (negative; active)
# =====================================================================================================


async def test_search_off_byte_identical(tmp_path: Path) -> None:
    """Criterion 10: a FakeSearch kit without ``web_search`` runs seven searches as BASE does -- every one reaches the
    backend, every card is ``dry_run``, every step request is ``loop.step_prompt`` over the plain
    ``search_payload`` (no note, no limit, no refusal), no ``web search`` note and no ``searches``. Passes at BASE
    (a negative row: it fails only if T-0176 leaks outside ``web_search``)."""
    search = FakeSearch()
    async with wrig(
        tmp_path, search, steps=[*search_steps(2, 7), [answer(REPORT)]], web_search=None
    ) as rig:
        await run_turn(rig, 7)
        pass1 = tsl.sent(rig, "t1-gen")
        requests = {k: tsl.step_request(rig, k) for k in range(2, 9)}
    assert search.queries == list(QN[:7]), search.queries
    plain = [
        (
            PREFIX if k == 1 else f"Step {k}.",
            "google_search",
            args(k),
            loop.search_payload(wsw.results_of(DEFAULT_ROWS, QN[k - 1])),
        )
        for k in range(1, 8)
    ]
    for k, got in requests.items():
        assert got == tsl.oracle(pass1, plain[: k - 1]), f"criterion 10: step {k}'s request"
    for k in range(1, 8):
        assert rig.statuses(tsl.call_id(k)) == ["proposed", "dry_run"], rig.statuses(tsl.call_id(k))
    assert wsw.web_notes(rig) == [] and "searches" not in rig.record("t1").extras
