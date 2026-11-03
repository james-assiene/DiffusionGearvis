"""T-0092c criterion 6 and the draft/route parity row, over the real wire.

A ``Session`` with the real ``VllmClient`` in id mode (T-0088a's subset detokeniser, ``tests/unit/_llm_id_support``)
over ``httpx.MockTransport``. Pass 1 is a T-0087 recorded stream, verbatim (``tests/fixtures/llm/t0087_streams.jsonl``,
raw.jsonl ``f871d037…``, commit 724c0c4); a repair continuation, when one is asked for, is built by
``tests/fakes/vllm_sse.py`` from ids. ``/tokenize`` answers with the fake's deterministic map.

Oracle: D = the host decode of the recorded ids up to the first stop id, and ``router.route(D)`` -- the eval's
definition (T-0085). The rows check:

* the concatenated ``TtsSpeak.source`` over the turn equals ``drop_special_literals(route(D).spoken_prefix)`` for
  click-01, a google_search with no sentence, and key-02's stray ``<tool_call|>`` (criterion 6);
* nothing voiced or stored holds a special literal, ``call:`` or an argument value;
* the turn then judges and dry-runs the call (red at the base 4f8d4e0: no ``astra.tool.call`` is published);
* the draft the session tokenises is ``route(D).draft_call_text.rstrip()`` on every recorded tool stream (criterion
  1's parity row). The recorded calls are all valid, so the verdict is forced to ``repairable`` to make the draft
  observable on ``/tokenize``; ``classify`` is replaced wherever it is bound (``patch_bound``).

Regime: CPU, fixtures only, ``ScriptedClock``, n = 1 per recorded stream.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_astra.toolcall import live, router
from local_astra.toolcall.special_tokens import drop_special_literals
from tests.fakes import vllm_sse as V
from tests.fakes.llm import fake_tokenize
from tests.unit import _llm_id_support as S
from tests.unit._tool_turn_support import (
    assert_history_clean,
    assert_never_voiced,
    patch_bound,
    tool_rig,
)

ROWS = {f"{r['arm']}/{r['case']}": r for r in S.recorded()}
TOOL_ROWS = [k for k in ROWS if k.startswith("b/") and not k.startswith("b/chat-")]
SPEECH_ROWS = ["b/click-01", "b/search-01", "b/key-02"]
EMPTY_SSE = "data: [DONE]\n\n"


class ToolWire:
    """Serves pass 1 (one prepared stream), the repair continuation, prefills/captions (empty) and ``/tokenize``."""

    def __init__(self, pass1: list[str], continuation: list[str] | None = None) -> None:
        self.pass1 = [pass1]
        self.continuation = continuation
        self.tokenized: list[str] = []
        self.bodies: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(await request.aread())
        if request.url.path.endswith("/tokenize"):
            self.tokenized.append(body["prompt"])
            return httpx.Response(200, json={"tokens": fake_tokenize(body["prompt"])})
        assert request.url.path.endswith("/chat/completions"), f"unexpected {request.url}"
        self.bodies.append(body)
        lines: list[str] | None = None
        if body.get("continue_final_message"):
            lines = self.continuation
        elif body.get("max_tokens") != 1 and "tools" in body and self.pass1:
            lines = self.pass1.pop(0)
        text = V.sse_text(lines) if lines else EMPTY_SSE
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)


def decoded(row: dict[str, Any]) -> str:
    ids, _stop = V.until_stop(S.recorded_ids(row))
    return S.decode(ids)


def client(wire: ToolWire) -> Any:
    return S.client(wire.transport())


async def run_case(tmp_path: Any, wire: ToolWire) -> Any:
    llm = client(wire)
    async with tool_rig(tmp_path, llm=llm) as rig:
        await rig.ask("do it")
        await rig.turn_done()
    await llm.aclose()
    return rig


@pytest.mark.parametrize("key", SPEECH_ROWS)
async def test_c6_the_voiced_text_is_the_routers_spoken_prefix_on_the_recorded_wire(tmp_path, key):
    """Criterion 6 on click-01, search-01 (no sentence) and key-02 (stray close), then the call is dry-run."""
    row = ROWS[key]
    route = router.route(decoded(row))
    assert route.tag == "native", "premise: a native tool stream"
    wire = ToolWire(row["lines"])
    rig = await run_case(tmp_path, wire)
    expected = drop_special_literals(route.spoken_prefix)
    assert rig.spoken_sources() == expected, (
        f"T-0092c criterion 6: the voiced sources are drop_special_literals(route(D).spoken_prefix) = {expected!r}; "
        f"got {rig.spoken_sources()!r}"
    )
    args = live.classify("<|tool_call>" + route.draft_call_text).arguments or {}
    assert_never_voiced(rig, arguments=args)
    assert_history_clean(rig, arguments=args)
    assert rig.statuses() == ["proposed", "dry_run"], (
        f"T-0092c criteria 1/3: the recorded call is judged and dry-run; published {rig.statuses()}"
    )
    assert [i.arguments for i in rig.executor.invocations] == [args], rig.executor.invocations


async def test_c1_a_valid_call_split_across_canvases_on_the_wire(tmp_path):
    """Flow "a valid call split across canvases": the call's ids straddle two SSE chunks; one tool commit, one
    execution with the call's arguments, and only the sentence is voiced."""
    call = S.call("call:click_at{x:1,y:2}")
    ids = [*S.thought("t"), *S.prose("I will click it."), *call, S.TOOL_RESPONSE]
    cut = len(ids) - len(call) + 5
    lines = S.stream_lines([ids[:cut], ids[cut:]])
    rig = await run_case(tmp_path, ToolWire(lines))
    assert rig.spoken_sources() == "I will click it."
    assert rig.statuses() == ["proposed", "dry_run"], rig.statuses()
    assert [i.arguments for i in rig.executor.invocations] == [{"x": 1, "y": 2}]


@pytest.mark.parametrize("key", TOOL_ROWS)
async def test_c1_the_draft_is_the_routers_draft_on_the_parity_fixtures(tmp_path, monkeypatch, key):
    """Criterion 1: ``draft = router.route(tool.text).draft_call_text`` equals ``route(D).draft_call_text`` on every
    recorded tool stream, observed as the text the session sends to ``/tokenize`` (``draft.rstrip()``, F1)."""
    real = live.classify

    def forced(text: str) -> Any:
        v = real(text)
        return live.Verdict(
            "repairable", v.name, v.arguments, "forced for the parity row", v.query_problem
        )

    patch_bound(monkeypatch, live, "classify", forced)
    row = ROWS[key]
    expected = router.route(decoded(row)).draft_call_text.rstrip()
    continuation = S.stream_lines([[*S.call("call:click_at{x:1,y:2}"), S.TOOL_RESPONSE]])
    wire = ToolWire(row["lines"], continuation)
    await run_case(tmp_path, wire)
    assert wire.tokenized == [expected], (
        f"T-0092c criterion 1: the session's draft is route(D).draft_call_text; /tokenize got {wire.tokenized}, "
        f"expected [{expected!r}]"
    )


def test_the_parity_set_is_the_ten_recorded_tool_streams():
    assert len(TOOL_ROWS) == 10, TOOL_ROWS
