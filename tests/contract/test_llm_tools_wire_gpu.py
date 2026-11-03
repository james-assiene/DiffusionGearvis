"""T-0088 gpu rows (``-m gpu``, via ``/launch`` only; never in the fast suite): the tools-on wire against the live
engine.

Regime (the T-0080d umbrella): lean profile, vLLM v0.29.0, ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev
``2ea8372``, template ``36e3a42e…``, tokenizer ``cc8d3a0c…``, tools_v1 ``bbfc53cc…``, CL 256, 48 steps, thinking
on, prefix caching on, gpu-mem 0.86; engine only, n = 1 per row, no seed. The engine is already up on loopback
(``VllmEndpoint()``'s default); every row prints one ``T-0088 gpu regime [<row>]:`` line.

Rows:

1. click-01's prompt (``twopass.native_pass1_body``) through ``VllmClient`` in id mode: status 200, every non-role
   chunk has ``token_ids``, one ``tool`` commit that ``native.parse_call`` accepts, ``stop_id`` 50.
2. A tools-on prompt that forces several canvases: at least 2 content chunks, each with ``token_ids``; the
   host-decoded thought equals the served ``reasoning`` and the answer the served ``content``; per-chunk host time
   p50/max is **reported**. The recorded chunks are also compared with ``tests/fakes/vllm_sse.py`` built from the
   same ids and partition: a difference means the fake changes in T-0088's PR (CLAUDE.md), and the row fails.
3. One CD request (``cd=CdRequest``): status 200 and an ``ASTRA_CD`` record joined on its tag in the engine log.
   The payload is precompiled in the ``cd`` group before the launch (``compile_toolcall(..., scope="click_at",
   thought="T4", call_format="native")`` then ``artifact.encode_automaton``) and named by ``ASTRA_T0088_CD_PAYLOAD``:
   ``tests/fixtures/cd`` holds no precompiled payload (reported as an ambiguity; T-0091 adds them), and this module
   may not import ``local_astra.decoding`` (``test_cd_isolation.py``). The engine log path comes from ``ASTRA_ENGINE_LOG``; without it the row fails, naming the variable.
4. ``/tokenize``: the ids equal T-0087 q10's on its ten drafts (``tests/fixtures/llm/t0087_q10_tokenize.json``).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests import chunk_shape_diff

pytestmark = [pytest.mark.gpu, pytest.mark.contract]

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
Q10 = REPO / "tests" / "fixtures" / "llm" / "t0087_q10_tokenize.json"
MULTI_CANVAS_USER = "Write the numbers from 1 to 300 separated by spaces."


def _regime(row: str, **extra: Any) -> None:
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True
        ).stdout
    )
    print(
        f"T-0088 gpu regime [{row}]: head={head.strip()[:9]} dirty={dirty} lean vLLM0.29.0 rev2ea8372 CL256 "
        f"steps48 thinking=on prefix_caching=on gpu_mem=0.86 n=1 seed=none {json.dumps(extra)}"
    )


class _Recording(httpx.AsyncBaseTransport):
    """The real transport, with every ``data:`` line of each response kept (for the chunk-shape checks)."""

    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.lines: list[list[tuple[float, str]]] = []
        self.statuses: list[int] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        resp = await self.inner.handle_async_request(request)
        self.statuses.append(resp.status_code)
        kept: list[tuple[float, str]] = []
        self.lines.append(kept)
        stream = resp.stream

        async def tee() -> Any:
            buf = b""
            async for part in stream:  # type: ignore[union-attr]
                buf += part
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.startswith(b"data: "):
                        kept.append((time.perf_counter(), line.decode()))
                yield part

        return httpx.Response(
            resp.status_code, headers=resp.headers, content=tee(), request=request
        )

    async def aclose(self) -> None:
        await self.inner.aclose()


def _prompt_from_body(body: dict[str, Any], **kw: Any) -> Any:
    from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart

    msgs = []
    for m in body["messages"]:
        content = m["content"]
        if isinstance(content, str):
            parts: tuple[Any, ...] = (TextPart(text=content),)
        else:
            parts = tuple(
                TextPart(text=p["text"])
                if p["type"] == "text"
                else ImagePart(url=p["image_url"]["url"])
                for p in content
            )
        msgs.append(Message(role=m["role"], content=parts))
    return Prompt(messages=tuple(msgs), prefix_len=1, tools=tuple(body["tools"]), **kw)


def _case(case_id: str) -> dict[str, Any]:
    for line in (SET_DIR / "cases.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] == case_id:
            return dict(row)
    raise AssertionError(f"no case {case_id}")


def _click01_body() -> dict[str, Any]:
    from local_astra.eval.cd import twopass
    from local_astra.toolcall import tools

    return twopass.native_pass1_body(
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4", _case("click-01"), SET_DIR, tools.load()
    )


async def _generate(
    prompt: Any, rec: _Recording | None = None, **kw: Any
) -> tuple[list[Any], _Recording]:
    from local_astra.conductor.llm_client import VllmClient
    from local_astra.toolcall.detok import HostDetokenizer

    rec = _Recording() if rec is None else rec
    cl = VllmClient(transport=rec, detokenizer=HostDetokenizer.load())
    try:
        got = [c async for c in cl.generate(prompt, f"t0088-{uuid.uuid4().hex[:8]}", **kw)]
    finally:
        await cl.aclose()
    return got, rec


def _chunks(lines: list[tuple[float, str]]) -> list[dict[str, Any]]:
    return [json.loads(ln[len("data: ") :]) for _, ln in lines if ln != "data: [DONE]"]


def _record_row2(
    lines: list[tuple[float, str]], built: list[dict[str, Any]] | None, build_error: str | None
) -> None:
    """T-0095: with ``ASTRA_T0088_RECORD_DIR`` set, write row 2's live SSE lines verbatim (``row2_live.jsonl``) and
    ``V.build_chunks``' chunks one JSON object per line (``row2_built.jsonl``; a single ``{"build_error": ...}`` line
    if building raised). Unset: write nothing."""
    out = os.environ.get("ASTRA_T0088_RECORD_DIR")
    if not out:
        return
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "row2_live.jsonl").write_text("".join(f"{ln}\n" for _, ln in lines))
    rows = [{"build_error": build_error}] if built is None else built
    (d / "row2_built.jsonl").write_text("".join(json.dumps(c) + "\n" for c in rows))


def test_1_one_tool_case_in_id_mode() -> None:
    from local_astra.toolcall import native
    from local_astra.toolcall.score import ParseFailure

    got, rec = asyncio.run(_generate(_prompt_from_body(_click01_body())))
    chunks = _chunks(rec.lines[0])
    _regime(
        "1", case="click-01", chunks=len(chunks), commits=[(c.segment, len(c.text)) for c in got]
    )
    assert rec.statuses == [200]
    assert all("token_ids" in c["choices"][0] for c in chunks[1:]), (
        "every non-role chunk carries ids"
    )
    tools_ = [c for c in got if c.segment == "tool"]
    assert len(tools_) == 1
    assert not isinstance(
        native.parse_call(tools_[0].text.removeprefix("<|tool_call>")), ParseFailure
    )
    assert tools_[0].stop_id == 50


def test_2_a_multi_canvas_tools_on_response() -> None:
    from local_astra.contracts.llm import Message, TextPart
    from local_astra.toolcall.detok import HostDetokenizer
    from tests.fakes import vllm_sse as V

    body = _click01_body()
    body["messages"] = [body["messages"][0], {"role": "user", "content": MULTI_CANVAS_USER}]
    prompt = _prompt_from_body(body)
    assert prompt.messages[-1] == Message(role="user", content=(TextPart(text=MULTI_CANVAS_USER),))
    # T-0095: the recording lives outside _generate, and both files are written in the finally, before any
    # assertion that can fail, even when the client raises (row2_built.jsonl then holds a "not built: ..." line)
    rec = _Recording()
    built: list[dict[str, Any]] | None = None
    build_error: str | None = "not built: the row stopped before V.build_chunks"
    try:
        got, _ = asyncio.run(_generate(prompt, rec=rec))
        chunks = _chunks(rec.lines[0])
        content_chunks = [c for c in chunks[1:]]
        det = HostDetokenizer.load()
        try:
            built = V.build_chunks(
                V.StreamSpec(chunks=[c["choices"][0]["token_ids"] for c in content_chunks]), det
            )
            build_error = None
        except Exception as e:  # recorded, then the assertions below say what is wrong
            build_error = repr(e)
    except BaseException as e:
        build_error = f"not built: {e!r}"
        raise
    finally:
        _record_row2(rec.lines[0] if rec.lines else [], built, build_error)
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning", "") for c in content_chunks)
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in content_chunks)
    # report-only: per-chunk host time, replaying the recorded lines through the client over MockTransport
    stamps: list[float] = []
    wire = V.Wire(
        streams=[[ln for _, ln in rec.lines[0]]],
        on_chunk=lambda k: stamps.append(time.perf_counter()),
    )

    async def replay() -> list[float]:
        from local_astra.conductor.llm_client import VllmClient

        cl = VllmClient(transport=wire.transport(), detokenizer=det)
        seen: list[float] = []
        async for _ in cl.generate(prompt, "replay"):
            seen.append(time.perf_counter())
        await cl.aclose()
        return seen

    seen = asyncio.run(replay())
    per_chunk = [max(0.0, b - a) * 1000 for a, b in zip(stamps[1:], seen, strict=False)]
    _regime(
        "2",
        content_chunks=len(content_chunks),
        per_chunk_host_ms_p50=round(statistics.median(per_chunk), 3) if per_chunk else None,
        per_chunk_host_ms_max=round(max(per_chunk), 3) if per_chunk else None,
    )
    assert len(content_chunks) >= 2, "the prompt must force more than one canvas"
    assert all(isinstance(c["choices"][0].get("token_ids"), list) for c in content_chunks)
    assert "".join(c.text for c in got if c.segment == "thought") == reasoning
    assert "".join(c.text for c in got if c.segment == "answer") == content
    assert built is not None, f"V.build_chunks failed on the live ids: {build_error}"

    def strip(c: dict[str, Any]) -> str:
        return json.dumps(
            {
                k: v
                for k, v in c.items()
                if k not in ("id", "created", "prompt_token_ids", "prompt_text")
            }
        )

    same = [strip(c) for c in built] == [strip(c) for c in chunks]
    if not same:
        # T-0095: built by hand and raised through pytest.fail so that pytest does not truncate the values
        diff = chunk_shape_diff.first_mismatch(
            [json.loads(strip(c)) for c in built], [json.loads(strip(c)) for c in chunks]
        ) or (
            "no key path differs, yet the serialised chunks do:\n"
            f"built: {[strip(c) for c in built]!r}\nlive: {[strip(c) for c in chunks]!r}"
        )
        pytest.fail(
            "the live multi-canvas shape differs from tests/fakes/vllm_sse.py: change the fake in this PR\n"
            + diff,
            pytrace=False,
        )
    assert same


def test_3_one_cd_request_is_accepted_and_joined() -> None:
    from local_astra.contracts.cd import CdRequest
    from local_astra.contracts.llm import Message, TextPart
    from local_astra.eval.cd import records

    log_path = os.environ.get("ASTRA_ENGINE_LOG")
    assert log_path, (
        "row 3 needs ASTRA_ENGINE_LOG: the engine's log file, where the plugin writes ASTRA_CD records"
    )
    log = records.EngineLog(log_path)
    payload_path = os.environ.get("ASTRA_T0088_CD_PAYLOAD")
    assert payload_path, (
        "row 3 needs ASTRA_T0088_CD_PAYLOAD: a file holding artifact.encode_automaton(compile_toolcall(tools.load(), "
        "scope='click_at', thought='T4', call_format='native')), written in the cd group before the launch "
        "(the fast suite must not import local_astra.decoding: tests/unit/test_cd_isolation.py)"
    )
    payload = Path(payload_path).read_text().strip()
    tag = f"t0088-{uuid.uuid4().hex[:8]}"
    cd = CdRequest(
        mode="perstep_mar",
        automaton=payload,
        automaton_sha256=hashlib.sha256(payload.encode("ascii")).hexdigest(),
        record_tag=tag,
        entropy_bound=1.0,
        confidence_threshold=0.005,
    )
    body = _click01_body()
    base = _prompt_from_body(body)
    msgs = (
        *base.messages,
        Message(role="assistant", content=(TextPart(text="I will click it.<|tool_call>"),)),
    )
    prompt = base.model_copy(update={"messages": msgs, "continue_final_message": True})
    _got, rec = asyncio.run(_generate(prompt, cd=cd))
    joined = log.wait(tag, max_tokens=256, timeout_s=10.0)
    _regime("3", tag=tag, status=rec.statuses, records=0 if joined is None else len(joined.records))
    assert rec.statuses == [200]
    assert joined is not None and joined.records, f"no ASTRA_CD record joined on {tag}"


def test_4_tokenize_equals_t0087_q10() -> None:
    from local_astra.conductor.llm_client import VllmClient

    rows = json.loads(Q10.read_text())["rows"]
    assert len(rows) == 10

    async def go() -> list[list[int]]:
        cl = VllmClient()
        assert inspect.iscoroutinefunction(cl.tokenize), "ruling E5: tokenize is async"
        out = []
        for r in rows:
            out.append(await cl.tokenize(r["prompt"]))
        await cl.aclose()
        return out

    got = asyncio.run(go())
    _regime("4", equal=sum(g == r["tokens"] for g, r in zip(got, rows, strict=True)))
    assert got == [r["tokens"] for r in rows]
