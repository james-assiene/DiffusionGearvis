"""Shared helpers for the T-0088 rows (tools-on request body, id mode, the ``tool`` commit, typed CD, tokenize).

Nothing here touches T-0088's new surface at import time: every new name (``Prompt.tools``, ``Commit.stop_id``,
``LlmWireError``, ``VllmClient(detokenizer=...)``, ``tokenize``, ``cd=``) is reached inside a row through the
helpers below, which turn "it does not exist yet" into an ``AssertionError`` naming the criterion. So each row is
red on its own named assertion, and collection never errors (the T-0088a lazy-import pattern).

API the rows assume (docs/tasks/T-0088.md; where the task is silent, the choice is stated here and in the
test-author's report):

- ``Prompt(..., tools=tuple[dict, ...] | None, continue_final_message=bool)``; ``Commit(..., stop_id=int | None)``;
  ``Segment`` admits ``"tool"``;
- ``VllmClient(endpoint=None, transport=None, *, detokenizer=None)`` (``detokenizer`` passed by keyword);
- ``llm_client.LlmWireError`` (any exception class);
- ``generate(prompt, request_id, options=None, *, xargs=None, cd=None)``;
- ``async def tokenize(text) -> list[int]`` on the protocol and on both backends (ruling E5).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from local_astra.conductor import llm_client
from local_astra.contracts import llm as contracts_llm
from local_astra.contracts.llm import ImagePart, LlmOptions, Message, Prompt, TextPart
from local_astra.toolcall import native, tools
from local_astra.toolcall.special_tokens import SPECIAL_IDS
from tests.fakes import vllm_sse as V
from tests.unit._detok_support import detok_mod, fixture, write_tokenizer

REPO = Path(__file__).resolve().parents[2]
STREAMS = REPO / "tests" / "fixtures" / "llm" / "t0087_streams.jsonl"
GOLDEN = REPO / "tests" / "fixtures" / "llm" / "request_bodies_9669258.json"
#: The T-0088 base: main with T-0088a merged. A literal SHA, never the moving ``main`` (tests.md).
BASE_SHA = "9669258"

#: Pieces the rows spell markers and prose from that the T-0088a subset lacks, with their ids in the pinned
#: ``tokenizer.json`` (``cc8d3a0c…``; looked up with ``tokenizers`` 0.23.2 on 2026-10-12: ``<cc>`` = ``< cc >``,
#: ``<genui>`` = ``< gen ui >``, ``<a2ui>`` = ``< a 2 ui >``, ``</a2ui>`` = ``</ a 2 ui >``, ``{"a":1}`` =
#: ``{" a ": 1 }``). The ids only label pieces here: what the rows depend on is the piece text.
EXTRA_PIECES: dict[str, int] = {
    "cc": 767,
    "gen": 2568,
    "ui": 2450,
    "a": 236746,
    "</": 954,
    "think": 36345,
    '{"': 14937,
    '":': 1083,
    "Here": 8291,
    "Hi": 10979,
}

TOOL_CALL, TOOL_CALL_CLOSE, STRING_DELIM = 48, 49, 52
CHANNEL_OPEN, CHANNEL_CLOSE = 100, 101
EOS, TOOL_RESPONSE, TURN_CLOSE = 1, 50, 106
HEADER = ("<|channel>", "thought", "\n")

LITERAL_ID = {lit: i for i, lit in SPECIAL_IDS.items()}


def need(obj: Any, name: str, why: str) -> Any:
    if not hasattr(obj, name):
        owner = getattr(obj, "__name__", None) or type(obj).__name__
        raise AssertionError(f"T-0088 {why}: `{owner}.{name}` does not exist yet")
    return getattr(obj, name)


# --------------------------------------------------------------------------- tokens


@cache
def _detok_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="t0088-detok-"))


@cache
def detok() -> Any:
    """``HostDetokenizer`` over the T-0088a subset plus ``EXTRA_PIECES`` (written once per process)."""

    def add(tok: dict[str, Any]) -> None:
        vocab = tok["model"]["vocab"]
        for piece, i in EXTRA_PIECES.items():
            assert vocab.get(piece, i) == i
            vocab[piece] = i

    path, sha = write_tokenizer(_detok_dir(), add)
    return detok_mod().HostDetokenizer(path, sha)


@cache
def vocab() -> dict[str, int]:
    v: dict[str, int] = dict(fixture()["tokenizer"]["model"]["vocab"])
    v.update(EXTRA_PIECES)
    return v


def ids(*pieces: str) -> list[int]:
    """Ids of ``pieces``: a special literal is its special id, anything else a vocabulary piece (``▁`` spelled)."""
    v = vocab()
    return [LITERAL_ID[p] if p in LITERAL_ID else v[p] for p in pieces]


def words(text: str) -> list[str]:
    """Pieces for simple prose: ASCII letters/digits/punctuation one by one through the byte pieces when needed.

    Every character becomes its own piece when the subset has it, else its byte-fallback piece(s)."""
    v = vocab()
    out: list[str] = []
    for ch in text:
        piece = "▁" if ch == " " else ch
        if piece in v:
            out.append(piece)
        else:
            out.extend(f"<0x{b:02X}>" for b in ch.encode())
    return out


def prose(text: str) -> list[int]:
    return ids(*words(text))


def header() -> list[int]:
    return ids(*HEADER)


def thought(text: str) -> list[int]:
    """``<|channel>thought\\n`` + text + ``<channel|>``."""
    return [*header(), *prose(text), CHANNEL_CLOSE]


def call(text: str = "call:click_at{x:1,y:2}") -> list[int]:
    """``<|tool_call>`` + text + ``<tool_call|>`` (the text spelled piece by piece)."""
    return [TOOL_CALL, *prose(text), TOOL_CALL_CLOSE]


def decode(seq: Sequence[int]) -> str:
    return str(detok().decode(list(seq)))


# --------------------------------------------------------------------------- the recorded streams


@cache
def _streams_text() -> str:
    return STREAMS.read_text()


def recorded() -> list[dict[str, Any]]:
    """The 24 recorded streams (provenance line dropped); fresh copies."""
    rows = [json.loads(line) for line in _streams_text().splitlines() if line.strip()]
    assert "provenance" in rows[0]
    return rows[1:]


def recorded_chunks(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(ln[len("data: ") :]) for ln in row["lines"] if ln != V.DONE_LINE]


def recorded_ids(row: dict[str, Any]) -> list[int]:
    return [t for c in recorded_chunks(row) for t in c["choices"][0].get("token_ids", [])]


def served(row: dict[str, Any]) -> tuple[str, str, Any]:
    """``(reasoning, content, stop_reason)`` the engine served on a recorded stream."""
    reasoning = content = ""
    stop = None
    for c in recorded_chunks(row):
        ch = c["choices"][0]
        reasoning += ch["delta"].get("reasoning", "")
        content += ch["delta"].get("content", "")
        if ch.get("finish_reason") is not None:
            stop = ch.get("stop_reason")
    return reasoning, content, stop


# --------------------------------------------------------------------------- prompts


TOOLS: tuple[dict[str, Any], ...] = tuple(native.tools_field(tools.load()))

SYSTEM = Message(role="system", content=(TextPart(text="You are Astra."),))
USER = Message(role="user", content=(TextPart(text="Click the Sign in button."),))


def make_prompt(**kw: Any) -> Prompt:
    """``Prompt(**kw)``; a field T-0088 adds that does not exist yet is a named AssertionError, not a crash."""
    kw.setdefault("messages", (SYSTEM, USER))
    kw.setdefault("prefix_len", 1)
    try:
        return Prompt(**kw)
    except ValidationError as exc:
        missing = [
            e["loc"][0]
            for e in exc.errors()
            if e["type"] == "extra_forbidden"
            and e["loc"]
            and e["loc"][0] in ("tools", "continue_final_message")
        ]
        if missing:
            raise AssertionError(f"T-0088 criterion 1: Prompt has no field {missing}") from exc
        raise


def tools_prompt(**kw: Any) -> Prompt:
    kw.setdefault("tools", TOOLS)
    return make_prompt(**kw)


def continuation_prompt(assistant_text: str, **kw: Any) -> Prompt:
    msgs = (SYSTEM, USER, Message(role="assistant", content=(TextPart(text=assistant_text),)))
    return tools_prompt(messages=msgs, prefix_len=1, continue_final_message=True, **kw)


def make_commit(**kw: Any) -> Any:
    kw.setdefault("request_id", "r")
    kw.setdefault("canvas_idx", 0)
    kw.setdefault("text", "x")
    try:
        return contracts_llm.Commit(**kw)
    except ValidationError as exc:
        if any(e["type"] == "extra_forbidden" and e["loc"] == ("stop_id",) for e in exc.errors()):
            raise AssertionError("T-0088 criterion 1: Commit has no field `stop_id`") from exc
        if kw.get("segment") == "tool" and any(e["loc"] == ("segment",) for e in exc.errors()):
            raise AssertionError('T-0088 criterion 1: Segment does not admit "tool"') from exc
        raise


# --------------------------------------------------------------------------- the client


def client(transport: httpx.AsyncBaseTransport, *, detokenizer: Any = "subset") -> Any:
    """``VllmClient(transport=..., detokenizer=...)``; ``"subset"`` means :func:`detok`."""
    params = inspect.signature(llm_client.VllmClient.__init__).parameters
    if "detokenizer" not in params:
        raise AssertionError(
            "T-0088 criterion 5: VllmClient(..., detokenizer: HostDetokenizer | None = None) does not exist yet"
        )
    det = detok() if detokenizer == "subset" else detokenizer
    return llm_client.VllmClient(transport=transport, detokenizer=det)


def wire_error() -> type[BaseException]:
    err: type[BaseException] = need(llm_client, "LlmWireError", "criterion 5")
    return err


def stream_lines(chunks: Sequence[Sequence[int]], **kw: Any) -> list[str]:
    return V.build_lines(V.StreamSpec(chunks=[list(c) for c in chunks], **kw), detok())


async def collect(cl: Any, prompt: Prompt, request_id: str = "r1", **kw: Any) -> list[Any]:
    return [c async for c in cl.generate(prompt, request_id, **kw)]


async def run_chunks(
    chunks: Sequence[Sequence[int]], prompt: Prompt | None = None, **spec: Any
) -> tuple[list[Any], V.Wire]:
    """Serve ``chunks`` (one canvas each) through the real client in id mode; the commits and the wire."""
    wire = V.Wire(streams=[stream_lines(chunks, **spec)])
    cl = client(wire.transport())
    try:
        got = await collect(cl, prompt if prompt is not None else tools_prompt())
    finally:
        await cl.aclose()
    return got, wire


def run(chunks: Sequence[Sequence[int]], prompt: Prompt | None = None, **spec: Any) -> list[Any]:
    """Synchronous :func:`run_chunks` for Hypothesis rows (one event loop per example)."""
    return asyncio.run(run_chunks(chunks, prompt, **spec))[0]


def texts(commits: Sequence[Any], segment: str) -> str:
    return "".join(c.text for c in commits if c.segment == segment)


def tool_commits(commits: Sequence[Any]) -> list[Any]:
    return [c for c in commits if c.segment == "tool"]


def stop_id(commit: Any) -> Any:
    if "stop_id" not in type(commit).model_fields:
        raise AssertionError("T-0088 criterion 1: Commit has no field `stop_id`")
    return commit.stop_id


# --------------------------------------------------------------------------- the tools-off golden battery

#: 1x1 JPEG as an inline data URL (the client refuses anything else)
_JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP////////////////////////////////////////////////////////////////////////////////////8AAAsIAAEAAQEBEQD/xAAUAAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAA/AH//2Q=="


def golden_battery() -> list[tuple[str, Prompt, LlmOptions | None, dict[str, str] | None]]:
    """The tools-off prompts whose bodies were captured at ``BASE_SHA`` (``request_bodies_9669258.json``).

    Built from constructors that exist on main, so the battery itself is identical before and after T-0088."""
    frame = Message(
        role="user", content=(ImagePart(url=_JPEG, phash="0" * 16), TextPart(text="What is this?"))
    )
    hist = (
        SYSTEM,
        Message(role="user", content=(TextPart(text="hi"),)),
        Message(role="assistant", content=(TextPart(text="Hello. ünïcode — ok."),)),
        frame,
    )
    return [
        ("plain", Prompt(messages=(SYSTEM, USER), prefix_len=1), None, None),
        ("thinking_off", Prompt(messages=(SYSTEM, USER), prefix_len=2, thinking=False), None, None),
        (
            "options",
            Prompt(messages=(SYSTEM, USER), prefix_len=1),
            LlmOptions(top_k=40, top_p=0.9, max_tokens=256, stop=("</a2ui>",)),
            None,
        ),
        ("xargs", Prompt(messages=(SYSTEM, USER), prefix_len=1), None, {"astra_tap": "1"}),
        (
            "history_and_frame",
            Prompt(messages=hist, prefix_len=4),
            LlmOptions(max_tokens=512),
            None,
        ),
    ]
