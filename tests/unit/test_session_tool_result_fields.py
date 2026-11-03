"""T-0162 group C, criteria C-1 (session side) and C-2: the session fills ``AstraToolCall.step`` and
``AstraToolCall.result`` from data, never from model prose (docs/tasks/T-0162.md, Group C; docs/tasks/T-0125.md
criterion 11 and the planner's three T-0162 gaps).

What the rows read: only what the session publishes to the HUD (``rig.tool_calls``), through the T-0114
``tool_rig`` / T-0111g ``live_rig`` / T-0125 ``web_rig`` fakes. Expected values are written out by hand from the
criteria and the fixture rows (``tests/fakes/search.DEFAULT_ROWS``'s hosts are ``weather.example.org`` and
``forecast.example.org``); nothing is computed with the code under test.

The mapping the rows pin (C-1/C-2, with T-0125's gaps 1 and 2 applied):

- ``result`` is ``None`` on ``proposed``/``repairing`` and set on the call's one final message
  (``dry_run``/``executed``/``refused``/``cancelled``; ``ToolCallStatus`` is not widened);
- ``result.outcome`` is the call's real outcome: ``failed`` for a live executor failure (the card stays
  ``refused``, T-0111g B10), ``timeout`` for the session deadline and for a backend timeout (the card stays
  ``cancelled``), and for a search ``SearchFacts.status`` (T-0125 criterion 11);
- browser hosts come from the executor's ``resolved["hosts"]`` (T-0160 B-6); search hosts and ``results`` from
  ``SearchFacts`` (gap 2), never from ``resolved``;
- the wire carries at most 3 hosts; ``hosts_more`` counts every distinct host beyond the 3 shown, i.e. for a browser
  action ``len(resolved["hosts"]) - 3`` plus ``resolved.get("hosts_more", 0)`` (the executor's own count beyond its
  32) -- the test-author's reading, listed in the report;
- ``step`` is the call's loop step (1 for ``t1-call``, 2 for ``t1-call-s2``) on every message of the call -- the task
  does not say "final only" for ``step``, so it is pinned on all of them (listed in the report).

Written pending on T-0162 (group C, 8b5cf8a2); activated with group C's implementation (no ``pending`` marker). At
the base ``AstraToolCall`` had neither field, so each row failed on the named ``fields_of`` assertion; the C-4
negative row held on the base and must keep holding.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), tools
``dry_run`` unless a row says ``live``; n = 1 and deterministic per row. No measurement, no network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.protocols import SearchFailed
from local_astra.contracts.toolcall import ToolOutcome
from tests.fakes.search import FakeSearch
from tests.fakes.tool_executor import RecordingToolExecutor
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_session_web_search as ws
from tests.unit._live_browser_support import FakeLiveBrowser, failed, live_rig, refused
from tests.unit._t0177_pins import without_vision_frames  # T-0177 B-P1
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool, tool_rig
from tests.unit.test_session_live import F0, FR, three_steps
from tests.unit.test_session_loop import thought

CID = "t1-call"
FINAL = ("dry_run", "executed", "refused", "cancelled")
NON_FINAL = ("proposed", "repairing")
#: DEFAULT_ROWS' hosts, by hand, in rank order
SEARCH_HOSTS = ["weather.example.org", "forecast.example.org"]
OUTCOMES = {"executed", "dry_run", "refused", "failed", "cancelled", "timeout"}


# --- reading the published messages ---------------------------------------------------------------


def fields_of(msg: Any) -> tuple[Any, Any]:
    names = set(type(msg).model_fields)
    assert {"step", "result"} <= names, (
        "T-0162 C-1: AstraToolCall gains `step: int >= 1 | None` and `result: ToolCallResult | None`; its fields "
        f"are {sorted(names)}"
    )
    return msg.step, msg.result


def as_dict(result: Any) -> dict[str, Any]:
    return {
        "outcome": result.outcome,
        "hosts": list(result.hosts),
        "hosts_more": result.hosts_more,
        "results": result.results,
    }


def final_of(rig: Any, cid: str = CID) -> tuple[Any, dict[str, Any]]:
    """The call's one final message and its ``result`` as a dict; every non-final message has ``result is None``."""
    msgs = rig.tool_calls(cid)
    assert msgs, f"premise: {cid} published at least one astra.tool.call"
    finals = [m for m in msgs if m.status in FINAL]
    assert len(finals) == 1 and msgs[-1] is finals[0], (
        f"premise: {cid} ends in exactly one final card; {[m.status for m in msgs]}"
    )
    for m in msgs:
        _, result = fields_of(m)
        if m.status in NON_FINAL:
            assert result is None, (
                f"T-0162 C-1: result is set only on a final message; {cid} {m.status} carries {result!r}"
            )
    _, result = fields_of(finals[0])
    assert result is not None, (
        f"T-0162 C-1: the final {finals[0].status!r} message of {cid} carries a result; {finals[0]!r}"
    )
    return finals[0], as_dict(result)


def steps_of(rig: Any, cid: str) -> list[Any]:
    return [fields_of(m)[0] for m in rig.tool_calls(cid)]


def result_json(rig: Any) -> str:
    """Every published tool call's ``result``, as JSON (``null`` where absent)."""
    out = []
    for m in rig.tool_calls():
        _, result = fields_of(m)
        out.append(None if result is None else result.model_dump(mode="json"))
    return json.dumps(out)


# =====================================================================================================
# C-1/C-2: browser and dry-run calls
# =====================================================================================================


async def test_dry_run_calls_carry_their_step_and_a_dry_run_result(tmp_path: Path) -> None:
    """Two dry-run clicks in one loop turn: step 1 then step 2 on every message of each call; the final card's
    result is ``dry_run`` with no hosts and no result count."""
    async with tool_rig(
        tmp_path, policy=tsl.LOOP, script=[answer(tsl.S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)])
        await tsl.loop_turn(rig, 3)
    want = {"outcome": "dry_run", "hosts": [], "hosts_more": 0, "results": None}
    for k, cid in ((1, CID), (2, tsl.call_id(2))):
        card, result = final_of(rig, cid)
        assert card.status == "dry_run", f"premise: {cid} is a dry run; {card}"
        assert result == want, f"T-0162 C-1/C-2: {cid}'s result\n  want {want}\n  got  {result}"
        assert steps_of(rig, cid) == [k] * len(rig.tool_calls(cid)), (
            f"T-0162 C-1: every message of {cid} carries step {k}; {steps_of(rig, cid)}"
        )


@pytest.mark.parametrize(
    ("resolved", "hosts", "more"),
    [
        ({}, [], 0),  # not web mode: the executor reports no hosts
        ({"hosts": ["shop.example", "cdn.shop.example"]}, ["shop.example", "cdn.shop.example"], 0),
        (
            {"hosts": ["a.example", "b.example", "c.example", "d.example", "e.example"]},
            ["a.example", "b.example", "c.example"],
            2,
        ),
        (
            # the executor's own cap (32) was exceeded by 7: 2 beyond the 3 shown + 7
            {
                "hosts": ["a.example", "b.example", "c.example", "d.example", "e.example"],
                "hosts_more": 7,
            },
            ["a.example", "b.example", "c.example"],
            9,
        ),
        # reviewer A (optional): the executor's own overflow of 1 under the 3 shown
        ({"hosts": ["a.example"], "hosts_more": 1}, ["a.example"], 1),
    ],
    ids=["no_hosts", "two_hosts", "five_hosts", "five_plus_executor_more", "one_plus_one_more"],
)
async def test_live_executed_click_reports_the_executors_hosts(
    tmp_path: Path, resolved: dict[str, Any], hosts: list[str], more: int
) -> None:
    """C-2: ``hosts`` from ``outcome.resolved["hosts"]`` (T-0160 B-6), at most 3 in the executor's order;
    ``hosts_more`` the distinct hosts beyond them (adversarial: more than 3 hosts)."""
    browser = FakeLiveBrowser([F0, FR])
    browser.resolved_extra[CID] = resolved
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(tsl.S3)],))
    card, result = final_of(rig)
    assert card.status == "executed", f"premise: the live click executed; {card}"
    want = {"outcome": "executed", "hosts": hosts, "hosts_more": more, "results": None}
    assert result == want, f"T-0162 C-2\n  want {want}\n  got  {result}"
    assert steps_of(rig, CID) == [1] * len(rig.tool_calls(CID))


async def test_live_executor_failure_is_failed_from_the_field(tmp_path: Path) -> None:
    """C-2: a live executor ``failed`` outcome (T-0160's guarded field, which still carries ``hosts``): the card stays
    ``refused`` (``ToolCallStatus`` is not widened) and ``result.outcome`` is ``failed``."""
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(
        CID,
        lambda inv: ToolOutcome(
            call_id=inv.call_id,
            status="failed",
            action="failed type_at: password field",
            reason="password field",
            resolved={"hosts": ["login.example"]},
        ),
    )
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(tsl.S3)],))
    card, result = final_of(rig)
    assert card.status == "refused", f"premise (T-0111g B10): the card stays refused; {card}"
    want = {"outcome": "failed", "hosts": ["login.example"], "hosts_more": 0, "results": None}
    assert result == want, f"T-0162 C-2\n  want {want}\n  got  {result}"


async def test_failed_is_never_derived_from_wording(tmp_path: Path) -> None:
    """C-2, the too-wide direction: a *refused* outcome whose reason reads like a failure or a timeout is
    ``refused``; only the field says ``failed``."""
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(CID, lambda inv: refused(inv, "failed: timeout while the page was loading"))
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(tsl.S3)],))
    card, result = final_of(rig)
    assert card.status == "refused"
    assert result["outcome"] == "refused", f"T-0162 C-2: never from wording; {result}"


async def test_live_failure_without_hosts(tmp_path: Path) -> None:
    """C-2: the fake's plain ``failed`` (no ``hosts`` key) gives ``failed`` with no hosts."""
    browser = FakeLiveBrowser([F0, FR])
    browser.outcome(CID, lambda inv: failed(inv, "TimeoutError during click_at"))
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(tsl.S3)],))
    _, result = final_of(rig)
    assert result == {"outcome": "failed", "hosts": [], "hosts_more": 0, "results": None}, result


async def test_session_deadline_is_timeout(tmp_path: Path) -> None:
    """C-2: an executor that never answers is cut by the session deadline: the card is ``cancelled``
    (``timeout``), ``result.outcome`` is ``timeout``."""
    ex = RecordingToolExecutor(hang=True)
    async with ws.web_rig(
        tmp_path,
        FakeSearch(),
        executor=ex,
        web_search=None,
        script=[answer(tsl.PREFIX), tool(VALID_CLICK)],
    ) as rig:
        await rig.ask("click the sign in button")
        await rig.until(lambda: bool(ex.invocations), "premise: the click runs", limit_ms=5_000)
        await tsl.until_steps(rig, 1, limit_ms=12_000, step=250)
        await rig.turn_done(limit_ms=30_000)
    card, result = final_of(rig)
    assert (card.status, card.reason) == ("cancelled", "timeout"), f"premise: {card}"
    assert result == {"outcome": "timeout", "hosts": [], "hosts_more": 0, "results": None}, result


@pytest.mark.parametrize(
    "bad",
    [
        "<img src=x onerror=alert(1)>",
        "https://evil.example/path",
        "evil.example/path?q=1",
        'x.example"><script>',
        "",
    ],
    ids=["img_onerror", "url", "path_query", "quote_markup", "empty"],
)
async def test_markup_or_a_url_in_the_executors_hosts_never_reaches_the_wire(
    tmp_path: Path, bad: str
) -> None:
    """Adversarial (C-1 "host only"): a ``resolved["hosts"]`` entry that is not a plain host is not published; the
    good host beside it is, and the call still ends in exactly one final card with a result."""
    browser = FakeLiveBrowser([F0, FR])
    browser.resolved_extra[CID] = {"hosts": [bad, "good.example"]}
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        await three_steps(rig, ([answer(tsl.S3)],))
    _, result = final_of(rig)
    assert "good.example" in result["hosts"], f"T-0162 C-2: the plain host is kept; {result}"
    for host in result["hosts"]:
        assert host and not any(ch in host for ch in "<>\"'/:?#@ "), (
            f"T-0162 C-1: hosts are host only; {host!r} published"
        )
    assert len(result["hosts"]) <= 3


# =====================================================================================================
# C-2 with T-0125's SearchFacts (gap 2): search calls
# =====================================================================================================


@pytest.mark.parametrize(
    ("kwargs", "card_status", "outcome", "hosts", "results"),
    [
        ({}, "executed", "executed", SEARCH_HOSTS, 2),
        ({"fail": TimeoutError("slow")}, "cancelled", "timeout", [], None),
        ({"fail": SearchFailed("network", "x")}, "cancelled", "failed", [], None),
        ({"fail": SearchFailed("rate_limited", "x")}, "cancelled", "failed", [], None),
        ({"fail": RuntimeError("boom")}, "cancelled", "failed", [], None),
    ],
    ids=["results", "backend_timeout", "network", "rate_limited", "other_exception"],
)
async def test_search_result_from_search_facts(
    tmp_path: Path,
    kwargs: dict[str, Any],
    card_status: str,
    outcome: str,
    hosts: list[str],
    results: int | None,
) -> None:
    """C-2 (gap 2): a search's ``result`` is its ``SearchFacts``: outcome = status, hosts, hosts_more, results.
    Gap 1: ``proposed`` then the final, no ``dry_run``."""
    rig = await ws.one_search_turn(tmp_path, FakeSearch(**kwargs))
    card, result = final_of(rig)
    assert card.status == card_status, f"premise (T-0125 criterion 7): {card}"
    assert rig.statuses(CID) == ["proposed", card_status], rig.statuses(CID)
    want = {"outcome": outcome, "hosts": hosts, "hosts_more": 0, "results": results}
    assert result == want, f"T-0162 C-2\n  want {want}\n  got  {result}"


async def test_search_with_no_results(tmp_path: Path) -> None:
    """C-2: an executed search whose backend returned no rows: ``executed``, no hosts, and a count of 0 (``None`` is
    also accepted: T-0125 criterion 11 says ``None`` "when no results came back", which may read as this case)."""
    rig = await ws.one_search_turn(tmp_path, FakeSearch(default=()))
    _, result = final_of(rig)
    assert (result["outcome"], result["hosts"], result["hosts_more"]) == ("executed", [], 0), result
    assert result["results"] in (0, None), result


async def test_search_hosts_beyond_three(tmp_path: Path) -> None:
    """Adversarial (more than 3 hosts): six rows over five distinct hosts (one repeated, one with ``www.``): the
    first 3 distinct in rank order, ``hosts_more`` 2, ``results`` 6 (the rows, not the hosts)."""
    rows = [
        {"title": "1", "url": "https://one.example/a", "snippet": "s"},
        {"title": "2", "url": "https://www.two.example/b", "snippet": "s"},
        {"title": "3", "url": "https://one.example/c", "snippet": "s"},
        {"title": "4", "url": "https://three.example/", "snippet": "s"},
        {"title": "5", "url": "https://four.example/", "snippet": "s"},
        {"title": "6", "url": "https://five.example/", "snippet": "s"},
    ]
    rig = await ws.one_search_turn(tmp_path, FakeSearch(default=rows))
    _, result = final_of(rig)
    want = {
        "outcome": "executed",
        "hosts": ["one.example", "two.example", "three.example"],
        "hosts_more": 2,
        "results": 6,
    }
    assert result == want, f"T-0162 C-2\n  want {want}\n  got  {result}"


async def test_search_deadline_is_timeout(tmp_path: Path) -> None:
    """C-2: a backend that never returns is cut by the session deadline: ``timeout``."""
    search = FakeSearch(block=asyncio.Event())
    async with ws.web_rig(
        tmp_path, search, script=[answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)]
    ) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.until(lambda: bool(search.queries), "premise: the search started", limit_ms=5_000)
        await tsl.until_steps(rig, 1, limit_ms=12_000, step=250)
        await rig.turn_done(limit_ms=30_000)
    card, result = final_of(rig)
    assert card.status == "cancelled", card
    assert result == {"outcome": "timeout", "hosts": [], "hosts_more": 0, "results": None}, result


async def test_cap_refused_search(tmp_path: Path) -> None:
    """C-2: the search over the cap is refused without a backend call: ``refused``, no count. T-0176 re-scope
    (criterion 1): the cap is 5, so it is the sixth search (was the fourth at a cap of 3)."""
    qs = ws.QS7
    search = FakeSearch()
    steps = [[answer(f"Step {k}."), tool(ws.search_call(qs[k - 1]))] for k in range(2, 7)] + [
        [answer(tsl.REPORT)]
    ]
    async with ws.web_rig(
        tmp_path,
        search,
        script=[answer(tsl.PREFIX), tool(ws.search_call(qs[0]))],
        steps=steps,
    ) as rig:
        await tsl.loop_turn(rig, 7)
    assert search.queries == list(qs[:5]), f"premise: five searches ran; {search.queries}"
    sixth = tsl.call_id(6)
    card, result = final_of(rig, sixth)
    assert card.status == "refused", card
    assert result == {"outcome": "refused", "hosts": [], "hosts_more": 0, "results": None}, result
    assert steps_of(rig, sixth) == [6] * len(rig.tool_calls(sixth)), steps_of(rig, sixth)
    _, first = final_of(rig, CID)
    assert first["outcome"] == "executed" and first["results"] == 2, first


async def test_search_cancelled_after_the_backend_call(tmp_path: Path) -> None:
    """C-2: a takeover while the backend holds the query: the cancel path's card, ``result.outcome`` ``cancelled``."""
    search = FakeSearch(block=asyncio.Event())
    async with ws.web_rig(
        tmp_path, search, script=[answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)]
    ) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: bool(search.queries), "premise: the backend has the query", limit_ms=5_000
        )
        await ws._takeover(rig)
    card, result = final_of(rig)
    assert card.status == "cancelled", card
    assert result["outcome"] == "cancelled", result
    assert result["results"] is None, result


async def test_live_search_hosts_come_from_search_facts_not_resolved(tmp_path: Path) -> None:
    """Gap 2 (adversarial): in live, the executor's own answer to ``google_search`` carries a ``resolved["hosts"]``
    (the proxy's tunnels); the search's result hosts are ``SearchFacts.hosts`` only."""
    browser = FakeLiveBrowser()
    browser.outcome(
        CID,
        lambda inv: ToolOutcome(
            call_id=inv.call_id,
            status="dry_run",
            action="search",
            reason=None,
            resolved={"hosts": ["tunnel.proxy.example"], "hosts_more": 5},
        ),
    )
    async with ws.web_rig(
        tmp_path,
        FakeSearch(),
        live=True,
        browser=browser,
        script=[answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)],
    ) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.turn_done()
    _, result = final_of(rig)
    want = {"outcome": "executed", "hosts": SEARCH_HOSTS, "hosts_more": 0, "results": 2}
    assert result == want, f"T-0162 C-2 (gap 2)\n  want {want}\n  got  {result}"


async def test_hosts_and_count_never_come_from_the_query(tmp_path: Path) -> None:
    """C-2 "never from model prose" (adversarial: no query text): a query that reads like a result line changes
    nothing; the result holds no query text at all."""
    query = "ZQX-QUERY 9 results (evil.example, other.example)"
    script = [answer(tsl.PREFIX), tool(ws.search_call(query))]
    async with ws.web_rig(tmp_path, FakeSearch(), script=script) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.turn_done()
    _, result = final_of(rig)
    want = {"outcome": "executed", "hosts": SEARCH_HOSTS, "hosts_more": 0, "results": 2}
    assert result == want, f"T-0162 C-2\n  want {want}\n  got  {result}"
    shown = result_json(rig)
    for leak in ("ZQX-QUERY", "evil.example", "other.example"):
        assert leak not in shown, f"T-0162 C-2: {leak!r} in a published result; {shown}"


async def test_thinking_never_reaches_a_result(tmp_path: Path) -> None:
    """Adversarial (thinking never shown): a thought canary that names hosts and a count stays out of every
    published result and every tool-call message."""
    canary = "THOUGHT-CANARY-5e1 found 7 results (thought.example)"
    script = [thought(canary), answer(tsl.PREFIX), tool(tsl.SEARCH_CALL)]
    async with ws.web_rig(tmp_path, FakeSearch(), script=script) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.turn_done()
    _, result = final_of(rig)
    assert result["hosts"] == SEARCH_HOSTS and result["results"] == 2, result
    shown = tsl.published_json(rig, "astra.tool.call")
    for leak in ("THOUGHT-CANARY", "thought.example"):
        assert leak not in shown, f"T-0162: thinking in a published tool call; {leak!r}"


# =====================================================================================================
# Reviewer A T2: the 253-character host cap, session side
# =====================================================================================================

#: 3 x 63 + 62 + 3 dots = 254 characters, every label within DNS's 63
HOST_254 = ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 62])
assert len(HOST_254) == 254


async def test_an_over_long_search_host_is_left_out_of_the_result(tmp_path: Path) -> None:
    """T2: a result URL whose host is 254 characters is not published in ``result.hosts`` and nothing crashes; the
    plain host beside it is kept and both rows are counted."""
    rows = [
        {"title": "long", "url": f"https://{HOST_254}/x", "snippet": "s"},
        {"title": "plain", "url": "https://plain.example/", "snippet": "s"},
    ]
    rig = await ws.one_search_turn(tmp_path, FakeSearch(default=rows))
    _, result = final_of(rig)
    assert result["hosts"] == ["plain.example"], (
        f"T-0162 T2: the 254-char host is dropped; {result}"
    )
    assert (result["outcome"], result["results"]) == ("executed", 2), result
    assert HOST_254 not in result_json(rig)


# =====================================================================================================
# Amendment 2026-10-27 (after reviewer A): a guarded type_at's text is masked on the page
# =====================================================================================================

MASK = "•••"  # "•••"
PLAIN_TEXT = "hunter2-ZQX"
CARD_TEXT = "4111 1111 1111 1111"  # Luhn-valid: fence.card_like holds


def type_call(text: str) -> str:
    q = tsl.Q
    return (
        f"<|tool_call>call:type_at{{press_enter:false,text:{q}{text}{q},x:500,y:250}}<tool_call|>"
    )


def guarded(reason: str, resolved: dict[str, Any]) -> Any:
    return lambda inv: ToolOutcome(
        call_id=inv.call_id,
        status="failed",
        action="failed type_at: guarded field",
        reason=reason,
        resolved=resolved,
    )


async def type_turn(tmp_path: Path, text: str, outcome: Any = None) -> Any:
    browser = FakeLiveBrowser([F0, FR])
    if outcome is not None:
        browser.outcome(CID, outcome)
    async with live_rig(tmp_path, browser, script=[answer(tsl.S1), tool(type_call(text))]) as rig:
        await three_steps(rig, ([answer(tsl.S3)],))
    return rig


def texts(rig: Any, cid: str = CID) -> list[tuple[str, Any]]:
    return [(m.status, m.arguments.get("text")) for m in rig.tool_calls(cid)]


def _reasons() -> dict[str, str]:
    from local_astra.computer import executor as ex

    return {
        "secret": ex.SECRET_FIELD_REASON,
        "card": ex.CARD_TEXT_REASON,
        "focus_unknown": ex.FOCUS_UNKNOWN_REASON,
    }


@pytest.mark.parametrize(
    ("reason_key", "resolved"),
    [
        (
            None,
            {"secret_field": "password", "typed": 0},
        ),  # the field kind alone, an unlisted reason
        ("secret", {}),
        ("card", {}),
        ("focus_unknown", {}),
    ],
    ids=["secret_field_key", "secret_reason", "card_reason", "focus_unknown_reason"],
)
async def test_a_guarded_type_at_final_message_masks_the_text(
    tmp_path: Path, reason_key: str | None, resolved: dict[str, Any]
) -> None:
    """Amendment (b): the final ``astra.tool.call`` of a ``type_at`` the guard refused carries ``text == "•••"``;
    every other argument is the call's own. Each trigger is pinned on its own: ``resolved["secret_field"]`` with a
    reason outside the three, and each of the three reasons with no ``secret_field``."""
    reason = "password field (guarded)" if reason_key is None else _reasons()[reason_key]
    rig = await type_turn(tmp_path, PLAIN_TEXT, guarded(reason, resolved))
    final = rig.tool_calls(CID)[-1]
    assert final.status == "refused", f"premise (T-0111g B10): {final}"
    assert final.arguments == {"x": 500, "y": 250, "text": MASK, "press_enter": False}, (
        f"T-0162 amendment: the guarded text is masked on the final message; {final.arguments}"
    )
    assert PLAIN_TEXT not in rig.tool_calls(CID)[-1].model_dump_json()


async def test_card_like_text_is_masked_from_the_proposed_message_on(tmp_path: Path) -> None:
    """Amendment (a): ``fence.card_like(text)`` masks every message of the call, ``proposed`` included, even when the
    executor (here the fake) does not refuse it."""
    rig = await type_turn(tmp_path, CARD_TEXT)
    seen = texts(rig)
    assert seen and seen[0][0] == "proposed", f"premise: the call was proposed; {seen}"
    assert all(t == MASK for _, t in seen), f"T-0162 amendment (a): masked from proposed on; {seen}"
    assert "4111" not in tsl.published_json(rig, "astra.tool.call")


@pytest.mark.parametrize(
    "outcome",
    [None, lambda inv: refused(inv, "frame not from this browser")],
    ids=["executed", "refused_unguarded"],
)
async def test_an_unguarded_type_at_keeps_its_text(tmp_path: Path, outcome: Any) -> None:
    """The too-wide direction: an executed ``type_at``, and one refused for a reason that is not a guard's, keep
    their text on every message."""
    rig = await type_turn(tmp_path, PLAIN_TEXT, outcome)
    seen = texts(rig)
    assert len(seen) >= 2 and all(t == PLAIN_TEXT for _, t in seen), (
        f"T-0162 amendment: an unguarded type_at keeps its text; {seen}"
    )


async def test_the_hello_replay_carries_the_masked_copy(tmp_path: Path) -> None:
    """Amendment: the journal is latest-wins, so a page that reconnects after a guarded ``type_at`` is replayed the
    masked final, never the proposed message's text. The session's own published messages are fed, in order, to a
    real ``HudServer`` whose journal is then replayed to a fresh hello."""
    from local_astra.contracts.protocol import AstraTurnState
    from local_astra.contracts.turn import TurnState
    from tests.unit.test_hud_hello_replay import make_server
    from tests.unit.test_hud_server_reset import greet

    rig = await type_turn(tmp_path / "s", PLAIN_TEXT, guarded(_reasons()["secret"], {}))
    published = rig.tool_calls(CID)
    assert published[-1].arguments.get("text") == MASK, f"premise: {texts(rig)}"
    _, server = make_server(tmp_path / "h")
    server.publish(AstraTurnState(state=TurnState.THINKING, turn_id="t1", thinking_ms=None))
    for m in published:
        server.publish(m)
    replay = [m for m in greet(server).sent() if getattr(m, "type", None) == "astra.tool.call"]
    assert replay, "premise: the open turn's tool call is replayed"
    assert [m.arguments.get("text") for m in replay] == [MASK], (
        f"T-0162 amendment: the hello replays the masked copy only; {[m.arguments for m in replay]}"
    )


# =====================================================================================================
# C-4 (negative): holds on the base, must keep holding (not pending)
# =====================================================================================================


async def test_a_turn_without_tool_calls_publishes_no_tool_fields(tmp_path: Path) -> None:
    """C-4: with no tool calls, no message carries ``step`` or ``result`` (the byte-identical golden is
    ``test_session_tools_off_golden.py``; this row checks the loop-on rig, where the fields exist)."""
    async with tool_rig(tmp_path, policy=tsl.LOOP, script=[answer("Plain answer here.")]) as rig:
        await rig.ask("say something")
        await rig.turn_done()
    assert rig.tool_calls() == [], "premise: no tool call"
    # astra.canvas.step has a field of its own named ``step``; every other message is checked.
    # T-0177 B-P1: so has astra.vision.frame (B7), stripped shape-checked by ``without_vision_frames``.
    shown = json.dumps(
        [
            m.model_dump(mode="json")
            for m in without_vision_frames(list(rig.published.messages))
            if not str(getattr(m, "type", "")).startswith("astra.canvas.")
        ]
    )
    assert '"step"' not in shown and '"result"' not in shown, shown[:400]
