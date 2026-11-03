"""T-0172 B10: the logged announced-call records replayed through a fake-LLM session (docs/tasks/T-0172.md).

Cases: ``tests/eval/t0172_announced_call_cases.jsonl`` (provenance in its first line). For every ``expect: reask``
record the turn is rebuilt over the fakes (tools ``dry_run``, loop on, 8 steps): steps 1..k-1 commit the logged
``prior_calls`` in order, step k commits the logged thought tail and answer and no call. Then:

- (a) the re-ask fires once, under step k's machine id, with the B2 prompt shape
  (``reask_prompt(<step k's prompt as sent>, <the logged answer's step text>)``);
- (b) a scripted re-ask committing the record's call executes it (a ``google_search`` reaches ``FakeSearch``; a
  ``type_at`` reaches the executor) and nothing of step k is voiced;
- (c) a scripted no-call re-ask voices the announcement then its answer, and no second re-ask is sent.

For every ``expect: no_reask`` record (the T-0160 web records: announced and called, or a no-call answer the rule does
not hit) the same rebuild sends no re-ask: with the logged call, the call executes as logged.

CPU over fakes; ``ScriptedClock``; n = 1 per record, deterministic. The live-browser step-1 positive is synthetic
(derived from W2-2 with its tool commit removed) and replayed in ``dry_run``: the re-ask path is mode-independent
up to the executor, which the unit rows cover in ``live`` (``test_not_when_power[recovered_thought_call]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import live
from local_astra.toolcall.announced_call import REASK_NUDGE, announces, reask_prompt
from tests.fakes.search import FakeSearch
from tests.unit import test_t0172_session_reask as m
from tests.unit._tool_turn_support import answer, thought, tool

CASES_PATH = Path(__file__).with_name("t0172_announced_call_cases.jsonl")
ROWS = [json.loads(x) for x in CASES_PATH.read_text().splitlines() if x.strip()]
CASES = {r["id"]: r for r in ROWS if "id" in r}
REASK = sorted(k for k, r in CASES.items() if r["expect"] == "reask")
NO_REASK = sorted(k for k, r in CASES.items() if r["expect"] == "no_reask")
#: CPU only: the ``gpu`` marker gates selection (every tests/eval row is deselected without ``-m gpu``, T-0046), as
#: the T-0142b, T-0170 and T-0172 census evals. Run: ``uv run pytest tests/eval/test_t0172_reask_cases.py -m gpu``
pytestmark = pytest.mark.gpu

NO_CALL_ANSWER = "I could not do that, the search limit was reached."


def test_cases_premises():
    """Active: the file has provenance, both smoke records and at least two T-0160 web records; ``announces`` hits
    every ``reask`` record's answer with the logged verb, and the no-call web negatives are rule misses."""
    assert "provenance" in ROWS[0]
    assert {"t0125-t2-s7", "t0125-t3-s2"} <= set(REASK)
    assert len([k for k in CASES if k.startswith("t0160-")]) >= 2
    for k in REASK:
        a = announces(CASES[k]["answer"])
        assert a is not None and a.verb == CASES[k]["verb"], (k, a)
        assert live.classify(CASES[k]["reask_call"]).status == "valid", k
    for k in NO_REASK:
        r = CASES[k]
        if "call" not in r:
            assert announces(r["answer"]) is None, k
        else:
            assert (
                announces(r["answer"]) is not None and live.classify(r["call"]).status == "valid"
            ), k


def _steps(r: dict[str, Any], tail: list[Any]) -> tuple[list[Any], list[list[Any]]]:
    calls = [tool(c) for c in r["prior_calls"]]
    if not calls:
        return tail, []
    return [calls[0]], [*([c] for c in calls[1:]), tail]


async def _replay(tmp: Path, r: dict[str, Any], reasks: list[list[Any]], after: list[Any]) -> Any:
    search = FakeSearch()
    tail = [thought(r.get("thought_tail", "")), answer(r["answer"])]
    if "call" in r:
        tail.append(tool(r["call"]))
    script, steps = _steps(r, tail)
    async with m.reask_rig(tmp, search=search, script=script) as rig:
        await m.run(rig, *steps, after, reasks=reasks, question=r["question"])
    rig.search = search
    return rig


@pytest.mark.parametrize("case", REASK)
async def test_reask_fires_with_prompt_shape(tmp_path, case):
    """B10 (a)."""
    r = CASES[case]
    rig = await _replay(tmp_path, r, [[answer(NO_CALL_ANSWER)]], [answer(m.REPORT)])
    sid = m.step_id(r["step"])
    _, prompt = m.assert_one_reask(rig, sid, case)
    _, sent = m.original(rig, sid)
    assert prompt == reask_prompt(sent, m.step_text(r["answer"])), (
        f"T-0172 B10(a) {case}: the B2 shape"
    )


@pytest.mark.parametrize("case", REASK)
async def test_reask_call_executes_unvoiced(tmp_path, case):
    """B10 (b)."""
    r = CASES[case]
    rig = await _replay(tmp_path, r, [[tool(r["reask_call"])]], [answer(m.REPORT)])
    k = r["step"]
    if "reask_query" in r:
        assert rig.search.queries[-1:] == [r["reask_query"]], rig.search.queries
    else:
        assert (m.call_id(k), r["reask_args"]) in m.invoked(rig), m.invoked(rig)
    spoken = m.speaks_json(rig)
    if k > 1:
        assert r["answer"].strip() not in spoken, (
            f"T-0172 B10(b) {case}: nothing of step {k} is voiced"
        )
    assert REASK_NUDGE not in spoken and m.REPORT in rig.spoken_sources()


@pytest.mark.parametrize("case", REASK)
async def test_no_call_reask_voices_both(tmp_path, case):
    """B10 (c)."""
    r = CASES[case]
    rig = await _replay(tmp_path, r, [[answer(NO_CALL_ANSWER)], [answer("A second re-ask.")]], [])
    sources = rig.spoken_sources()
    a = m.step_text(r["answer"]).strip()
    assert (
        a in sources
        and NO_CALL_ANSWER in sources
        and sources.index(a) < sources.index(NO_CALL_ANSWER)
    ), f"T-0172 B10(c) {case}: the announcement, then the re-ask's answer; {sources!r}"
    assert len(rig.llm.reask_prompts) == 1 and "A second re-ask." not in sources


@pytest.mark.parametrize("case", NO_REASK)
async def test_web_negatives_send_no_reask(tmp_path, case):
    """B10 negatives (the T-0160 web records): announced and called, or a rule miss: no re-ask; a logged call
    executes as logged."""
    r = CASES[case]
    rig = await _replay(tmp_path, r, [[answer("never sent")]], [answer(m.REPORT)])
    m.assert_no_reask(rig, case)
    if "call" in r:
        args = live.classify(r["call"]).arguments
        assert (m.call_id(r["step"]), dict(args)) in m.invoked(rig), m.invoked(rig)
