"""T-0207 A2.12: the seq 2493 / seq 5538 shaped records replayed through a fake-LLM live session (docs/tasks/T-0207.md).

Cases: ``tests/eval/t0207_completion_cases.jsonl`` (provenance in its first line; synthetic, shaped like James's
session). For each record the turn is rebuilt over the fakes (tools ``live``, ``FakeLiveBrowser``, loop on, 8 steps):
step 1 and the following steps commit the record's ``prior_calls`` in order, the last of which captures a post-action
frame with the same bytes and phash as the one before it (``last_action_unchanged``); the final step commits the
record's answer and no call. With ``Session(completion_check=True)``:

- (a) the check fires once, under the final step's machine id, with A2.4's shape (``check_prompt(<the step prompt as
  sent>, <step text>, <the fresh frame>, report=False)``, ``phase: "completion_check"``), and the answer is never in
  ``astra.spoken``;
- (b) a check that commits the record's ``click_at`` executes it on the fresh frame, and nothing of the final step is
  voiced;
- (c) a no-call check answer is voiced alone (the draft never);
- (d) a check failure (transport) voices the draft, as the keyword-absent turn does.

CPU over fakes; ``ScriptedClock``; n = 1 per record, deterministic. Marked ``gpu`` like every tests/eval row
(T-0046; ruling A2-R1), although it spends no GPU time; the same (a)-(d) behaviour is pinned on the CPU by
``tests/unit/test_t0207_session_check.py`` (``test_fires_once_with_shape``, ``test_check_call_executes_withheld``,
``test_draft_never_voiced_or_stored``, ``test_failure_falls_back_to_base``). Run:
``uv run pytest tests/eval/test_t0207_cases.py -m gpu``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.toolcall import live
from local_astra.toolcall.completion_check import check_prompt
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0207_session_check as m
from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame
from tests.unit._tool_turn_support import answer, thought, tool

CASES_PATH = Path(__file__).with_name("t0207_completion_cases.jsonl")
ROWS = [json.loads(x) for x in CASES_PATH.read_text().splitlines() if x.strip()]
CASES = {r["id"]: r for r in ROWS if "id" in r}
IDS = sorted(CASES)
pytestmark = pytest.mark.gpu

PRIOR_TEXT = "Let me continue."
AFTER = "I clicked it again and stopped there."


def test_cases_premises() -> None:
    """Active: provenance, both record shapes, every prior call and check call classifies valid, the last prior call is
    the logged action and is a browser action."""
    assert "provenance" in ROWS[0]
    assert {r["seq"] for r in CASES.values()} == {2493, 5538}
    for r in CASES.values():
        for c in (*r["prior_calls"], r["check_call"]):
            assert live.classify(c).status == "valid", c
        assert "click_at" in r["prior_calls"][-1] and r["last_action_unchanged"] is True


def _frames(tag: str) -> tuple[FakeLiveBrowser, Any]:
    onset, p1, p2, fresh, p4 = (
        jpeg_frame(f"{tag}-{n}") for n in ("onset", "p1", "p2", "fresh", "p4")
    )
    # the last prior action captures the same bytes (and phash) as the action before it
    browser = FakeLiveBrowser([onset, fresh], post_frames=[p1, p2, p2.model_copy(), p4])
    return browser, fresh


async def _replay(
    tmp: Path, r: dict[str, Any], on: Any, *, check: list[Any], fail: bool = False
) -> Any:
    calls = r["prior_calls"]
    final = [*([thought(r["thought"])] if r["thought"] else []), answer(r["answer"])]
    steps = [*([answer(PRIOR_TEXT), tool(c)] for c in calls[1:]), final, [answer(AFTER)]]
    browser, fresh = _frames(r["id"])
    async with m.live(tmp, browser, on=on, script=[answer(PRIOR_TEXT), tool(calls[0])]) as rig:
        if fail:
            rig.llm.fail_checks(httpx.ConnectError("connection refused"))
        await m.drive(rig, *steps, checks=(check,), question=r["question"])
    rig.fresh = fresh  # type: ignore[attr-defined]
    return rig


def _k(r: dict[str, Any]) -> int:
    return len(r["prior_calls"]) + 1


@pytest.mark.parametrize("cid", IDS)
async def test_a_fires_once_with_shape(tmp_path: Path, cid: str) -> None:
    r = CASES[cid]
    rig = await _replay(tmp_path, r, True, check=[answer(r["check_says"])])
    k = _k(r)
    _, prompt = m.assert_one_check(rig, m.step_id(k), cid)
    sent = tsl.step_request(rig, k)
    assert prompt == check_prompt(sent, m.step_text(r["answer"]), m.held(rig.fresh), report=False)
    (req,) = m.check_requests(rig)
    assert (req["phase"], req["request_id"], req["step"]) == ("completion_check", m.step_id(k), k)
    assert r["answer"] not in m.spoken_published(rig)


@pytest.mark.parametrize("cid", IDS)
async def test_b_check_click_executes_nothing_voiced(tmp_path: Path, cid: str) -> None:
    r = CASES[cid]
    rig = await _replay(tmp_path, r, True, check=[tool(r["check_call"])])
    k = _k(r)
    cids = [i.call_id for i in rig.browser.invocations]
    assert m.call_id(k) in cids and m.results(rig, m.call_id(k)) == ["executed"], cids
    assert m.inv_of(rig, m.call_id(k)).frame.phash == rig.fresh.phash
    assert r["answer"] not in rig.spoken_sources() and r["answer"] not in m.spoken_published(rig)
    assert len([x for x in rig.llm.check_prompts if re.search(rf"-{m.step_id(k)}-\d+$", x[0])]) == 1


@pytest.mark.parametrize("cid", IDS)
async def test_c_no_call_check_voiced_alone(tmp_path: Path, cid: str) -> None:
    r = CASES[cid]
    rig = await _replay(tmp_path, r, True, check=[answer(r["check_says"])])
    sources = rig.spoken_sources()
    assert sources.endswith(r["check_says"]) and r["answer"] not in sources, sources
    assert r["answer"] not in m.history_json(rig)


@pytest.mark.parametrize("cid", IDS)
async def test_d_failure_voices_draft_as_base(tmp_path: Path, cid: str) -> None:
    r = CASES[cid]
    on = await _replay(tmp_path / "on", r, True, check=[answer(r["check_says"])], fail=True)
    base = await _replay(tmp_path / "absent", r, m.ABSENT, check=[answer(r["check_says"])])
    assert r["answer"] in base.spoken_sources(), "premise: BASE voices the claim"
    assert on.spoken_sources() == base.spoken_sources()
    assert m.outcomes(on) == ["failed"], m.notes(on)
