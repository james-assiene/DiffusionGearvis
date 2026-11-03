"""T-0088 criterion 3: ``LlmBackend.generate(..., *, xargs=None, cd: CdRequest | None = None)``.

``backend`` is the fake or the real client, as in T-0017 (``test_llm_backend_xargs.py``): the real side is
``VllmClient`` over ``httpx.MockTransport`` — the real request builder and SSE reader, no GPU. The live engine
row is gpu row 3 of ``test_llm_tools_wire_gpu.py``. Both sides carry the same assertions.

Expected wire values come from ``CdRequest``'s own fields, written by hand below (types included): the JSON on
the wire must carry ``1.0`` as a float and ``64``/``24`` as ints, never as strings (T-0080d finding 3).
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.cd import CdRequest
from local_astra.contracts.llm import VLLM_XARGS_KEY, Commit, Message, Prompt, TextPart
from local_astra.contracts.protocols import LlmBackend
from tests.fakes import FakeLlm

pytestmark = pytest.mark.contract

PROMPT = Prompt(messages=(Message(role="user", content=(TextPart(text="hi"),)),), prefix_len=1)
SSE = 'data: {"choices":[{"delta":{"content":"Four."}}]}\n\ndata: [DONE]\n\n'
SCRIPT = (Commit(request_id="?", canvas_idx=0, segment="answer", text="Four."),)

AUTOMATON = "AAAA"  # any ASCII string: CdRequest checks only its sha here
DRAFT = base64.b64encode((48).to_bytes(4, "little") + (6639).to_bytes(4, "little")).decode()
CD = CdRequest(
    mode="perstep_mar",
    automaton=AUTOMATON,
    automaton_sha256=hashlib.sha256(AUTOMATON.encode()).hexdigest(),
    record_tag="t0088-cd",
    entropy_bound=1.0,
    confidence_threshold=0.005,
    max_bucket=64,
    draft=DRAFT,
    draft_noise=0.0,
    draft_start_step=24,
)
#: by hand from the fields above (``CdRequest.to_xargs`` omits unset optionals and prefixes ``astra_cd_``)
EXPECTED_CD = {
    "astra_cd_mode": "perstep_mar",
    "astra_cd_self_conditioning": "model",
    "astra_cd_automaton": AUTOMATON,
    "astra_cd_automaton_sha256": hashlib.sha256(AUTOMATON.encode()).hexdigest(),
    "astra_cd_record_tag": "t0088-cd",
    "astra_cd_entropy_bound": 1.0,
    "astra_cd_confidence_threshold": 0.005,
    "astra_cd_max_bucket": 64,
    "astra_cd_draft": DRAFT,
    "astra_cd_draft_noise": 0.0,
    "astra_cd_draft_start_step": 24,
}


@dataclass
class Harness:
    backend: LlmBackend
    fake: FakeLlm | None = None
    raw: list[bytes] = field(default_factory=list)

    def sent(self) -> list[dict[str, Any]]:
        """The ``vllm_xargs`` that left the boundary, as JSON (types as the engine would parse them)."""
        if self.fake is not None:
            return (
                [json.loads(json.dumps(self.fake.last_extra_body))]
                if self.fake.last_extra_body
                else []
            )
        return [json.loads(b)[VLLM_XARGS_KEY] for b in self.raw]


@pytest.fixture(params=["fake", "vllm"])
def harness(request: pytest.FixtureRequest) -> Harness:
    if request.param == "fake":
        fake = FakeLlm(SCRIPT)
        return Harness(backend=fake, fake=fake)
    raw: list[bytes] = []

    def handler(req: httpx.Request) -> httpx.Response:
        raw.append(req.read())
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=SSE)

    return Harness(backend=VllmClient(transport=httpx.MockTransport(handler)), raw=raw)


def _accepts_cd(backend: Any) -> None:
    params = inspect.signature(backend.generate).parameters
    assert "cd" in params, (
        f"criterion 3: {type(backend).__name__}.generate takes no `cd` keyword yet"
    )


async def _run(backend: Any, **kw: Any) -> list[Any]:
    _accepts_cd(backend)
    return [c async for c in backend.generate(PROMPT, "r1", None, **kw)]


def test_the_protocol_declares_cd_keyword_only_defaulting_to_none() -> None:
    params = inspect.signature(LlmBackend.generate).parameters
    assert "cd" in params, "criterion 3: LlmBackend.generate has no `cd` keyword"
    assert params["cd"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["cd"].default is None
    assert params["xargs"].kind is inspect.Parameter.KEYWORD_ONLY, "T-0017's xargs is unchanged"


async def test_cd_values_reach_the_wire_with_their_json_types(harness: Harness) -> None:
    got = await _run(harness.backend, cd=CD)
    assert [c.text for c in got] == ["Four."]
    (sent,) = harness.sent()
    assert {k: v for k, v in sent.items() if k.startswith("astra_cd_")} == EXPECTED_CD
    for key in ("astra_cd_entropy_bound", "astra_cd_confidence_threshold", "astra_cd_draft_noise"):
        assert type(sent[key]) is float, key
    for key in ("astra_cd_max_bucket", "astra_cd_draft_start_step"):
        assert type(sent[key]) is int, key
    assert sent["astra_request_id"] == "r1"


async def test_cd_and_xargs_merge_when_they_do_not_collide(harness: Harness) -> None:
    await _run(harness.backend, xargs={"astra_tap": "1"}, cd=CD)
    (sent,) = harness.sent()
    assert sent == {**EXPECTED_CD, "astra_tap": "1", "astra_request_id": "r1"}


async def test_a_key_in_both_xargs_and_cd_raises_before_any_request(harness: Harness) -> None:
    _accepts_cd(harness.backend)
    with pytest.raises(ValueError) as excinfo:
        async for _ in harness.backend.generate(
            PROMPT, "r1", None, xargs={"astra_cd_mode": "end_projection"}, cd=CD
        ):
            pass
    assert "astra_cd_mode" in str(excinfo.value)
    assert harness.sent() == [], "nothing was sent"


async def test_astra_request_id_wins_over_xargs_with_cd(harness: Harness) -> None:
    await _run(harness.backend, xargs={"astra_request_id": "spoofed"}, cd=CD)
    (sent,) = harness.sent()
    assert sent["astra_request_id"] == "r1"


async def test_no_cd_sends_no_cd_key(harness: Harness) -> None:
    """The negative half: ``cd=None`` (and omitting it) adds nothing."""
    await _run(harness.backend, cd=None)
    (sent,) = harness.sent()
    assert sent == {"astra_request_id": "r1"}


def test_fake_records_the_cd_of_each_generate() -> None:
    import asyncio

    fake = FakeLlm(SCRIPT)

    async def go() -> None:
        async for _ in fake.generate(PROMPT, "a", cd=CD):
            pass
        async for _ in fake.generate(PROMPT, "b"):
            pass

    asyncio.run(go())
    assert fake.cd_log == [CD, None]
    assert fake.last_cd is None
