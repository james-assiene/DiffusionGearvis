"""T-0170 criteria 1, 3 and 5 through the session: an implicit thought is never voiced, published or stored.

The rig is T-0092c's ``tool_rig`` (``tests/unit/_tool_turn_support``) with the **real** ``VllmClient`` in id mode over
``httpx.MockTransport`` (T-0088a's subset detokeniser, ``tests/unit/_llm_id_support``), as
``test_session_tool_wire.py`` drives it. ``LoopWire`` serves the tools-on generates in order (step 1, then the step-2
continuation), empty streams to prefills and captions, and the fake ``/tokenize``. Search is T-0114's ``FakeSearch``.

The step-2 canvas is the incident's (``logs/t0125-search-smoke-b372eda/phase1/sessions/20261027-112444-1064873.jsonl``,
seq 92, request ``t1-gen-s2``, attempt ``11cb09b2dc51-t1-gen-s2-3``, main b372edaa): the logged answer text with
``<channel|>`` re-inserted at the ``sentence.The`` splice, split over three canvases here so the hold is exercised.

``test_llm_commit_record_keys`` drives the three ``llm.commit`` record sites (pass 1 / loop step, the repair, ReGround
pass 2) with the scripted fakes yielding a ``Commit(implicit_thought=True)``, so the record shape is pinned without
the client. The base key sets are written out from the b372edaa incident log (seq 27 thought, seq 29 tool, seq 92
answer).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run`` (ReGround
row: live kit over ``FakeLiveBrowser``), n = 1 and deterministic per row.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.contracts import commands as C
from local_astra.contracts.llm import Commit
from tests.fakes import vllm_sse as V
from tests.fakes.llm import fake_tokenize
from tests.fakes.search import FakeSearch
from tests.unit import _llm_id_support as S
from tests.unit import test_session_reground as tr
from tests.unit import test_session_tool_loop as tsl
from tests.unit._llm_id_support import (
    CHANNEL_CLOSE,
    TOOL_CALL,
    TOOL_CALL_CLOSE,
    TOOL_RESPONSE,
    TURN_CLOSE,
)
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    answer,
    tool,
    tool_rig,
)

#: the incident's reasoning (seq 92's text before the splice), verbatim
REASONING = (
    "The user wants to know the height of the Eiffel Tower in one sentence.\n"
    "The search results show it is 330 meters (1,083 ft) with the antenna, and 300 meters of structural height.\n"
    "I will provide this information in one sentence."
)
#: the incident's answer (seq 92's text after the splice), verbatim
ANSWER = (
    "The Eiffel Tower is three hundred and thirty meters tall, or one thousand and eighty-three feet, "
    "including its antenna."
)
THOUGHT_PHRASES = ("user wants", "search results show", "I will provide")
EMPTY_SSE = "data: [DONE]\n\n"

#: ``llm.commit`` data keys at b372edaa (the incident log: seq 27 thought, seq 92 answer; seq 29 tool adds stop_id)
BASE_KEYS = frozenset(
    {
        "request_id",
        "attempt_id",
        "canvas_idx",
        "segment",
        "text",
        "thought_open",
        "fence_close",
        "fence_continued",
    }
)


class LoopWire:
    """Tools-on generates get the next prepared stream; prefills (``max_tokens: 1``) and tools-off requests an empty
    one; ``/tokenize`` the fake's map."""

    def __init__(self, streams: Sequence[list[str]]) -> None:
        self.streams = list(streams)
        self.bodies: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(await request.aread())
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"tokens": fake_tokenize(body["prompt"])})
        assert request.url.path.endswith("/chat/completions"), f"unexpected {request.url}"
        lines: list[str] | None = None
        if body.get("max_tokens") != 1 and "tools" in body and self.streams:
            self.bodies.append(body)
            lines = self.streams.pop(0)
        text = V.sse_text(lines) if lines else EMPTY_SSE
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)


def search_call_ids(query: str) -> list[int]:
    """``<|tool_call>call:google_search{query:<|"|>…<|"|>}<tool_call|>`` as ids."""
    quote = S.LITERAL_ID['<|"|>']
    return [
        TOOL_CALL,
        *S.prose("call:google_search{query:"),
        quote,
        *S.prose(query),
        quote,
        *S.prose("}"),
        TOOL_CALL_CLOSE,
    ]


def incident_chunks() -> list[list[int]]:
    """Seq 92's canvas with the orphan close re-inserted, over three canvases (the close in the last)."""
    first, rest = REASONING.split("\n", 1)
    return [
        S.prose(first + "\n"),
        S.prose(rest),
        [CHANNEL_CLOSE, *S.prose(ANSWER), TURN_CLOSE],
    ]


def implicit_commit(text: str, canvas_idx: int = 0) -> Commit:
    if "implicit_thought" not in Commit.model_fields:
        raise AssertionError("T-0170 criterion 1: Commit has no field `implicit_thought` yet")
    return Commit(
        request_id="?", canvas_idx=canvas_idx, segment="thought", text=text, implicit_thought=True
    )


def commit_records(rig: Any, request_id: str | None = None) -> list[dict[str, Any]]:
    return [
        r for r in rig.kind("llm.commit") if request_id is None or r.get("request_id") == request_id
    ]


def assert_never_spoken_or_stored(rig: Any, phrases: Sequence[str], spoken_answer: str) -> None:
    """Criterion 5's surfaces: ``TtsSpeak`` (source and text), ``astra.spoken``, history and ``turns.jsonl``. A turn
    that voiced nothing would pass vacuously, so the answer must be voiced and stored."""
    speaks = rig.machine.commands(C.TtsSpeak)
    hud = [m.model_dump(mode="json") for m in rig.published.messages]
    spoken = [m["text"] for m in hud if m.get("type") == "astra.spoken"]
    history = [h.assistant_spoken for h in rig.state.history]
    turns = rig.turns_path.read_text() if rig.turns_path.exists() else ""
    for phrase in phrases:
        for c in speaks:
            assert phrase not in c.source and phrase not in c.text, (
                f"T-0170 criterion 5: reasoning reached TTS: source={c.source!r}"
            )
        assert not any(phrase in s for s in spoken), f"T-0170 criterion 5: astra.spoken {spoken}"
        assert not any(phrase in h for h in history), f"T-0170 criterion 5: history {history}"
        assert phrase not in turns, "T-0170 criterion 5: turns.jsonl holds the reasoning"
    assert spoken_answer in "".join(c.source for c in speaks), (
        f"premise: the answer is voiced; sources {[c.source for c in speaks]}"
    )
    assert any(spoken_answer in h for h in history), f"premise: the answer is stored; {history}"


def assert_implicit_records(rig: Any, request_id: str, thought: str, answer_text: str) -> None:
    recs = commit_records(rig, request_id)
    thoughts = [r for r in recs if r["segment"] == "thought"]
    assert "".join(r["text"] for r in thoughts) == thought, (
        f"T-0170 criterion 1: {request_id}'s thought records hold the implicit thought; "
        f"{[(r['segment'], r['text'][:40]) for r in recs]}"
    )
    assert thoughts and all(r.get("implicit_thought") is True for r in thoughts), (
        f"T-0170 criterion 1: each is marked implicit_thought: true; {thoughts}"
    )
    assert "".join(r["text"] for r in recs if r["segment"] == "answer") == answer_text


async def test_step2_orphan_not_voiced_or_stored(tmp_path: Path) -> None:
    """Criterion 5 through the session (tools on, loop on, step 2): the step-1 call searches, the step-2 continuation
    is the incident canvas; nothing of the reasoning is voiced, published as spoken, stored in history or written to
    ``turns.jsonl``; the answer sentence is."""
    step1 = [*S.thought("t"), *S.prose(tsl.PREFIX), *search_call_ids(tsl.QUERY), TOOL_RESPONSE]
    wire = LoopWire([S.stream_lines([step1]), S.stream_lines(incident_chunks())])
    llm = S.client(wire.transport())
    search = FakeSearch()
    async with tool_rig(tmp_path, policy=tsl.LOOP, search=search, llm=llm) as rig:
        await rig.ask(tsl.QUESTION)
        await rig.turn_done("T-0170: the two-step search turn never ended", limit_ms=40_000)
    await llm.aclose()
    assert search.queries == [tsl.QUERY], "premise: step 1's call searched"
    assert len(wire.bodies) == 2 and wire.bodies[1].get("continue_final_message") is True, (
        "premise: step 2 is a continuation"
    )
    assert_never_spoken_or_stored(rig, THOUGHT_PHRASES, ANSWER)
    assert_implicit_records(rig, "t1-gen-s2", REASONING, ANSWER)


async def test_step1_orphan_not_voiced(tmp_path: Path) -> None:
    """Criteria 3 and 5, loop off, the first canvas: text before an orphan close arrives over two canvases before the
    close; none of it is voiced (the hold), and the answer after the close is."""
    chunks = [
        S.prose("The user greets me. "),
        S.prose("I will greet back."),
        [CHANNEL_CLOSE, *S.prose("Hello there."), TURN_CLOSE],
    ]
    wire = LoopWire([S.stream_lines(chunks)])
    llm = S.client(wire.transport())
    async with tool_rig(tmp_path, llm=llm) as rig:
        await rig.ask("hi")
        await rig.turn_done("T-0170: the turn never ended")
    await llm.aclose()
    assert_never_spoken_or_stored(rig, ("user greets", "greet back"), "Hello there.")
    assert_implicit_records(rig, "t1-gen", "The user greets me. I will greet back.", "Hello there.")


def _assert_keys(recs: Sequence[dict[str, Any]], flagged_text: str) -> None:
    flagged = [r for r in recs if r["text"] == flagged_text]
    assert len(flagged) == 1, (
        f"premise: one record of the implicit thought; {[(r['segment'], r['text']) for r in recs]}"
    )
    assert set(flagged[0]) == BASE_KEYS | {"implicit_thought"}, (
        f"T-0170 criterion 1: a flagged thought record adds implicit_thought; it has {sorted(flagged[0])}"
    )
    assert flagged[0]["implicit_thought"] is True
    for r in recs:
        if r is flagged[0]:
            continue
        want = BASE_KEYS | ({"stop_id"} if r["segment"] == "tool" else set())
        assert set(r) == want, (
            f"T-0170 criterion 1: every other record keeps exactly its b372edaa keys; a {r['segment']} record has "
            f"{sorted(r)}"
        )


@pytest.mark.parametrize("site", ["pass1", "repair", "reground_pass2"])
async def test_llm_commit_record_keys(tmp_path: Path, site: str) -> None:
    """Criterion 1: ``implicit_thought: true`` only on the records of flagged thought commits, at each of the three
    record sites; every other record's key set equals b372edaa's."""
    secret = "IMPLICIT-THOUGHT never voiced."
    if site == "pass1":
        async with tool_rig(
            tmp_path, script=[implicit_commit(secret), answer("Sure.", canvas_idx=1)]
        ) as rig:
            await rig.ask("hi")
            await rig.turn_done()
        recs = commit_records(rig)
    elif site == "repair":
        async with tool_rig(
            tmp_path,
            script=[answer("I will click it."), tool(REPAIRABLE_CLICK)],
            continuation=[implicit_commit(secret), tool(REPAIRED_CLICK, canvas_idx=0)],
        ) as rig:
            await rig.ask("click it")
            await rig.turn_done()
        recs = commit_records(rig)
        assert any(r["request_id"] == "t1-repair" for r in recs), "premise: the repair ran"
    else:
        pass2 = [implicit_commit(secret), answer(tr.SENTINEL), tool(tr.click(900, 100))]
        rig, _browser, _extras = await tr.one_click_turn(tmp_path, [pass2])
        assert rig.llm.pass2_ids, "premise: pass 2 ran"
        recs = commit_records(rig)
    _assert_keys(recs, secret)
    assert not any(secret in c.source for c in rig.machine.commands(C.TtsSpeak))
