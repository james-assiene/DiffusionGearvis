"""T-0087: the offline parts of ``scripts/gpu/t0080d_native_stream_probe.py`` (cd group; no engine, no GPU).

The probe is loaded by path with ``importlib`` (``scripts/gpu`` is not a package); nothing here reaches a real
engine. Every expectation is hand-built: the captures below are written out by hand, their ids come from the pinned
tokenizer (``vocab.encode`` of hand-written segments; the decode is checked against the hand text), and the regions
are hand-written literals. Nothing is computed by the code under test and then compared with itself.

Surface proposed here (the task names only ``main(argv) -> int``; the reviewer may rename, the tests follow):

- ``CASE_IDS``: the 12 fixed case ids (criterion 1).
- ``arm_body(body, arm) -> dict``: arm ``"a"``..``"e"`` applied to a ``twopass.native_pass1_body`` (criterion 1).
- ``reassemble(lines) -> dict``: raw SSE lines (``"data: {...}"``, blank lines, ``"data: [DONE]"``) to
  ``{"reasoning": str, "content": str, "chunks": [{"reasoning", "content", "token_ids"}...], "finish_reason",
  "stop_reason", "done": bool}``; a chunk field the delta did not carry is ``None`` (q2, q5, q7).
- ``reasoning_separated(reasoning, content, ids) -> bool`` (q3) and ``content_identical(content, ids) -> bool``
  (q4). The host decode is ``replay.detokenize`` of the ids before the first stop id (1, 106, 50). Its thought
  region is the text between the ``<|channel>thought\\n`` header and ``<channel|>`` (what the gemma4 reasoning
  parser serves; ``tests/cd/fakes/vllm_server.split_served``). Its answer region is the text after that
  ``<channel|>``.
- ``split_special_pieces(texts) -> int`` (q6): how many special pieces of the pinned tokenizer have their text split
  across a chunk boundary of the per-chunk ``content`` strings.
- ``redact(body) -> dict`` (criterion 1): a copy with each image data URL replaced by ``{"bytes", "sha256"}`` of
  the decoded image.
- ``reading(results) -> dict`` (criterion 6). ``results`` is::

      {"arms": {arm: {case_id: {"kind": "tool" | "none", "status": int, "q3": bool, "q4": bool,
                                "split_specials": int, "stop_reason": Any}}},
       "q8": [{"hits": int | None, "queries": int | None}, ...],   # with tools, 3 cases
       "q9": {"string": [{"status": int, "prompt_ends_48": bool}, ...], "list": [...]},
       "q10": [bool, ...]}                                          # 10 tool cases

  It returns ``{"reading": "A" | "B", "arm": "b" | "c" | None, "stop_id": ..., "tools_in_prefill": bool,
  "continuation_content": ..., "seed_tokenizer": ...}``. The positive values are pinned (``"stop_reason"``,
  ``"string"``, ``"tokenize"``); each replan branch is pinned only as "not the positive value".
- Ruling P7: arms (b)-(e) carry ``return_token_ids: true`` and q3/q4 for each are judged against that arm's own
  ids (the response ids for (d), the concatenated per-chunk ids for a stream arm). Arm (a) carries none; its
  comparison with (d)'s decode is ``report["arms"]["a"][cid]["vs_d"] = {"q3", "q4", "label": "noise-confounded"}``
  and decides nothing. ``b_ids_equal_e_ids`` is recorded per case (on (b)'s or (e)'s row) and decides nothing.
- Rulings after review (proposed names, the reviewer may rename): ``reassemble`` also returns ``error`` (an
  in-band ``data: {"error": ...}`` object, else ``None``); ``results["arms"][arm][cid]`` also carries ``done``,
  ``finish_reason``, ``error`` (P11) and ``q7`` (P13); ``reading`` also returns ``streamed_ids`` (P13) and, through
  ``main``, ``complete`` (P11: false with transport errors, every decision then reading "re-run (human)");
  ``report.json`` stores ``ids[arm][cid]`` for (b)-(e) and the stream arms' per-chunk content strings as
  ``chunk_contents[arm][cid]`` or a row's ``chunk_contents`` (P12).
- ``main(argv)`` accepts ``--base-url``, ``--repo`` (the tree whose cleanliness and sha are checked) and
  ``--out-dir`` (the default stays ``logs/t0087-probe-<sha>/``); every other option has a default.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.decoding import vocab
from local_astra.eval.cd import replay, twopass
from local_astra.toolcall import tools as _tools
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gpu" / "t0080d_native_stream_probe.py"
SET_DIR = REPO / "tests" / "eval" / "toolcall"
MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
TOOLS = ("google_search", "click_at", "type_at", "scroll", "press_key")


def probe() -> ModuleType:
    assert SCRIPT.exists(), f"T-0087 not implemented: {SCRIPT.relative_to(REPO)} does not exist"
    name = "astra_t0087_native_stream_probe"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def attr(name: str) -> Any:
    value = getattr(probe(), name, None)
    assert value is not None, (
        f"{SCRIPT.relative_to(REPO)} has no {name} (T-0087; name proposed here)"
    )
    return value


# --------------------------------------------------------------------------- hand-built captures

THOUGHT = "The user wants the weather in Lisbon."
ANSWER = 'Checking now.<|tool_call>call:google_search{query:<|"|>weather lisbon<|"|>}<tool_call|>'
#: the default arm's content: the special pieces dropped (T-0080d finding 1)
ANSWER_DEFAULT = "Checking now.call:google_search{query:weather lisbon}"
#: adversarial: the call tags kept, the string delimiters dropped
ANSWER_NO_QUOTES = "Checking now.<|tool_call>call:google_search{query:weather lisbon}<tool_call|>"
SEGMENTS = [
    "<|channel>thought\n",
    THOUGHT,
    "<channel|>",
    "Checking now.",
    "<|tool_call>",
    "call:google_search{query:",
    '<|"|>',
    "weather lisbon",
    '<|"|>',
    "}",
    "<tool_call|>",
]
NO_TOOL_THOUGHT = "Just a greeting."
NO_TOOL_ANSWER = "Hello there!"
NO_TOOL_SEGMENTS = ["<|channel>thought\n", NO_TOOL_THOUGHT, "<channel|>", NO_TOOL_ANSWER]


def _seg_ids(segments: Sequence[str]) -> list[list[int]]:
    return [vocab.encode(s) for s in segments]


def tool_ids() -> list[int]:
    """The tool capture's ids: the hand segments encoded one by one, then ``<|tool_response>`` (50)."""
    ids = [t for seg in _seg_ids(SEGMENTS) for t in seg] + [50]
    assert replay.detokenize(ids[:-1]) == "".join(SEGMENTS), (
        "fixture: the segments do not round-trip"
    )
    assert ids.count(48) == 1 and ids.count(49) == 1 and ids.count(52) == 2 and ids.count(101) == 1
    return ids


def no_tool_ids() -> list[int]:
    ids = [t for seg in _seg_ids(NO_TOOL_SEGMENTS) for t in seg] + [106]
    assert replay.detokenize(ids[:-1]) == "".join(NO_TOOL_SEGMENTS), "fixture: no round trip"
    return ids


def _chunk(delta: dict[str, Any], **choice: Any) -> str:
    body = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None, **choice}],
    }
    return "data: " + json.dumps(body, ensure_ascii=False)


def sse(
    reasoning: Sequence[str],
    content: Sequence[str],
    *,
    stop_reason: Any = 50,
    done: bool = True,
    field: str = "reasoning",
) -> list[str]:
    """Raw SSE lines: one chunk per reasoning piece, one per content piece, the final chunk, ``[DONE]``."""
    lines: list[str] = []
    for r in reasoning:
        lines += [_chunk({field: r}), ""]
    for c in content:
        lines += [_chunk({"content": c}), ""]
    lines += [_chunk({}, finish_reason="stop", stop_reason=stop_reason), ""]
    if done:
        lines += ["data: [DONE]", ""]
    return lines


CONTENT_PIECES = [
    "Checking now.",
    "<|tool_call>",
    "call:google_search{query:",
    '<|"|>weather lisbon<|"|>',
    "}<tool_call|>",
]
REASONING_PIECES = ["The user wants", " the weather in Lisbon."]


# =========================================================================== criterion 1: cases and arms


def test_the_twelve_cases_are_dev_two_per_tool_and_two_no_tool() -> None:
    ids = list(attr("CASE_IDS"))
    rows = {
        r["id"]: r
        for r in (
            json.loads(x) for x in (SET_DIR / "cases.jsonl").read_text().splitlines() if x.strip()
        )
    }
    assert len(ids) == 12 and len(set(ids)) == 12, ids
    assert all(i in rows and rows[i]["split"] == "dev" for i in ids), ids
    per_tool = {t: sum(1 for i in ids if rows[i]["tool"] == t) for t in TOOLS}
    assert per_tool == dict.fromkeys(TOOLS, 2), per_tool
    assert sum(1 for i in ids if rows[i]["kind"] != "tool" and rows[i]["tool"] is None) == 2
    assert hashlib.sha256((SET_DIR / "cases.jsonl").read_bytes()).hexdigest().startswith("460cd1c3")


def _base_body() -> dict[str, Any]:
    case = next(
        json.loads(x)
        for x in (SET_DIR / "cases.jsonl").read_text().splitlines()
        if x.strip() and json.loads(x)["id"] == "click-01"
    )
    return twopass.native_pass1_body(MODEL, case, SET_DIR, _tools.load())


ARM_FIELDS: dict[str, dict[str, Any]] = {
    "a": {"stream": True},
    "b": {"stream": True, "skip_special_tokens": False, "return_token_ids": True},
    "c": {
        "stream": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "return_token_ids": True,
    },
    "d": {"stream": False, "skip_special_tokens": False, "return_token_ids": True},
    "e": {"stream": True, "skip_special_tokens": False, "return_token_ids": True},
}


@pytest.mark.parametrize("arm", sorted(ARM_FIELDS))
def test_each_arm_is_the_native_pass1_body_plus_its_fields(arm: str) -> None:
    """Criterion 1: the native pass-1 body (messages, ``tools``, ``tool_choice`` ``none``, thinking on) plus the
    arm's fields; (a) leaves ``skip_special_tokens`` to the engine; only (c) sets the spacing flag. The base body
    is not mutated (it is shared by five arms). Ruling P7 (reverses P2 for (b) and (c)): (b), (c), (d) and (e)
    carry ``return_token_ids: true``; (a), today's ``llm_client`` baseline, carries none."""
    base = _base_body()
    before = copy.deepcopy(base)
    out = attr("arm_body")(base, arm)
    assert base == before, "arm_body mutated the shared pass-1 body"
    for key in ("model", "messages", "tools", "tool_choice", "chat_template_kwargs"):
        assert out[key] == base[key], key
    for key, value in ARM_FIELDS[arm].items():
        assert out.get(key, "<absent>") == value, (arm, key, out.get(key, "<absent>"))
    if arm == "a":
        assert "skip_special_tokens" not in out
    if arm != "c":
        assert "spaces_between_special_tokens" not in out
    if arm == "a":
        assert "return_token_ids" not in out, "ruling P7: arm (a) carries no return_token_ids"
    else:
        assert out.get("return_token_ids") is True, (
            f"ruling P7: arm ({arm}) carries return_token_ids: true"
        )


def test_arm_e_is_the_same_request_as_arm_b() -> None:
    """Ruling P7: (b) now carries ``return_token_ids: true`` too, so (e) is the same request as (b) (a replicate,
    the noise level); (a) is (b) without ``skip_special_tokens`` and without ``return_token_ids``."""
    base = _base_body()
    a = attr("arm_body")(base, "a")
    b = attr("arm_body")(base, "b")
    e = attr("arm_body")(base, "e")
    assert b.get("return_token_ids") is True, "ruling P7: arm (b) carries return_token_ids: true"
    assert e == b, "ruling P7: arm (e) is the same request as arm (b)"
    assert {k: v for k, v in b.items() if k not in ("skip_special_tokens", "return_token_ids")} == a


# =========================================================================== SSE reassembly (q2, q5, q7)


def test_reassembly_joins_reasoning_and_content_per_chunk() -> None:
    got = attr("reassemble")(sse(REASONING_PIECES, CONTENT_PIECES))
    assert got["reasoning"] == THOUGHT
    assert got["content"] == ANSWER
    assert [c["content"] for c in got["chunks"] if c["content"]] == CONTENT_PIECES
    assert [c["reasoning"] for c in got["chunks"] if c["reasoning"]] == REASONING_PIECES
    assert len(got["chunks"]) == len(REASONING_PIECES) + len(CONTENT_PIECES) + 1
    assert got["finish_reason"] == "stop" and got["stop_reason"] == 50
    assert got["done"] is True


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reassembly_accepts_either_reasoning_field_and_names_it(field: str) -> None:
    """Ruling P6: vLLM's delta field is recorded verbatim; either name reassembles, and the one seen is reported."""
    got = attr("reassemble")(sse(REASONING_PIECES, CONTENT_PIECES, field=field))
    assert got["reasoning"] == THOUGHT and got["content"] == ANSWER
    assert got["reasoning_field"] == field


def test_reassembly_reports_no_reasoning_field_when_none_was_seen() -> None:
    """Ruling P6, the other direction: a stream with no reasoning delta names no field (it does not default one)."""
    got = attr("reassemble")(sse([], CONTENT_PIECES))
    assert got["reasoning_field"] is None


def test_a_stream_cut_before_done_is_not_complete() -> None:
    """q2: a stream without ``[DONE]`` did not complete (the text so far is still reassembled)."""
    got = attr("reassemble")(sse(REASONING_PIECES, CONTENT_PIECES[:2], done=False)[:-4])
    assert got["done"] is False
    assert got["content"] == "Checking now."


def test_arm_e_chunks_carry_token_ids_that_concatenate_to_the_response() -> None:
    """q7: per-chunk ``token_ids`` (on the choice) survive reassembly, in order; q4 on (e) uses them."""
    seg = _seg_ids(SEGMENTS)
    groups = [
        ("reasoning", THOUGHT, seg[0] + seg[1] + seg[2]),
        ("content", "Checking now.", seg[3]),
        ("content", "<|tool_call>", seg[4]),
        (
            "content",
            'call:google_search{query:<|"|>weather lisbon<|"|>}',
            seg[5] + seg[6] + seg[7] + seg[8] + seg[9],
        ),
        ("content", "<tool_call|>", seg[10]),
    ]
    lines: list[str] = []
    for field, text, ids in groups:
        lines += [_chunk({field: text}, token_ids=ids), ""]
    lines += [
        _chunk({}, finish_reason="stop", stop_reason=50, token_ids=[50]),
        "",
        "data: [DONE]",
        "",
    ]
    got = attr("reassemble")(lines)
    flat = [t for c in got["chunks"] for t in (c["token_ids"] or [])]
    assert flat == tool_ids()
    assert got["content"] == ANSWER
    assert attr("content_identical")(got["content"], flat) is True


def test_a_chunk_without_token_ids_reports_none() -> None:
    """q7's negative: a stream without per-chunk ids (arm (a), ruling P7) is recorded as absent, not as an empty
    list."""
    got = attr("reassemble")(sse(REASONING_PIECES, CONTENT_PIECES))
    assert all(c["token_ids"] is None for c in got["chunks"])


# =========================================================================== q3 and q4 against the host decode


def test_q4_served_content_with_the_specials_is_identical_to_the_answer_region() -> None:
    assert attr("content_identical")(ANSWER, tool_ids()) is True
    assert attr("content_identical")(NO_TOOL_ANSWER, no_tool_ids()) is True


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(ANSWER_DEFAULT, id="default-arm-specials-dropped"),
        pytest.param(ANSWER_NO_QUOTES, id="adversarial-string-delimiters-dropped"),
        pytest.param("<channel|>" + ANSWER, id="stray-channel-close"),
        pytest.param(ANSWER + "<|tool_response>", id="stop-text-in-content"),
        pytest.param(ANSWER.replace("<|tool_call>", "<|tool_call> "), id="space-after-special"),
        pytest.param(ANSWER[:-1], id="one-byte-short"),
    ],
)
def test_q4_any_byte_difference_is_not_identical(content: str) -> None:
    assert attr("content_identical")(content, tool_ids()) is False


def test_q3_reasoning_equals_the_thought_region_and_stays_out_of_content() -> None:
    assert attr("reasoning_separated")(THOUGHT, ANSWER, tool_ids()) is True
    assert attr("reasoning_separated")(NO_TOOL_THOUGHT, NO_TOOL_ANSWER, no_tool_ids()) is True


@pytest.mark.parametrize(
    ("reasoning", "content"),
    [
        pytest.param(None, THOUGHT + ANSWER, id="no-reasoning-field"),
        pytest.param(THOUGHT, THOUGHT + ANSWER, id="thought-also-in-content"),
        pytest.param("The user wants", ANSWER, id="reasoning-truncated"),
        pytest.param("thought\n" + THOUGHT, ANSWER, id="header-left-in-reasoning"),
    ],
)
def test_q3_fails_when_the_thought_is_not_separated(reasoning: str | None, content: str) -> None:
    assert attr("reasoning_separated")(reasoning, content, tool_ids()) is False


# =========================================================================== q6 split special pieces


@pytest.mark.parametrize(
    ("texts", "splits"),
    [
        pytest.param(CONTENT_PIECES, 0, id="pieces-whole"),
        pytest.param([ANSWER], 0, id="one-chunk"),
        pytest.param(
            ["Checking now.<|tool", "_call>call:google_search{query:"],
            1,
            id="adversarial-tool-call-split",
        ),
        pytest.param(['query:<|"', "|>weather"], 1, id="quote-split"),
        pytest.param(["}<tool_", "call|>"], 1, id="close-split"),
        pytest.param(["<|tool", "_ca", "ll>"], 1, id="one-piece-over-three-chunks"),
        pytest.param(["a<|tool", "_call>b<|", '"|>c'], 2, id="two-splits"),
        pytest.param(["<|tool_call>", '<|"|>'], 0, id="boundary-between-pieces"),
        pytest.param(["x <", "| y |", "> z"], 0, id="plain-angle-and-pipe-text"),
        pytest.param(["<channel", "|>Hi"], 1, id="channel-close-split"),
    ],
)
def test_q6_counts_special_pieces_split_across_chunks(texts: list[str], splits: int) -> None:
    assert attr("split_special_pieces")(texts) == splits


def test_adversarial_a_split_tool_call_reassembles_cleanly_but_fails_q6() -> None:
    """A capture splitting ``<|tool_call>`` over two chunks: the joined content is still byte-identical (so q4
    cannot catch it), q6 must, and the arm cannot give reading A."""
    pieces = [
        "Checking now.<|tool",
        "_call>call:google_search{query:",
        '<|"|>weather lisbon<|"|>',
        "}<tool_call|>",
    ]
    got = attr("reassemble")(sse(REASONING_PIECES, pieces))
    assert got["content"] == ANSWER
    assert attr("content_identical")(got["content"], tool_ids()) is True
    splits = attr("split_special_pieces")([c["content"] for c in got["chunks"] if c["content"]])
    assert splits == 1
    res = results()
    for arm in ("b", "c"):
        res["arms"][arm]["search-01"]["split_specials"] = splits
    assert attr("reading")(res)["reading"] == "B"


# =========================================================================== criterion 6: the reading

TOOL_CASE_IDS = [
    f"{t}-{n:02d}" for t in ("search", "click", "type", "scroll", "key") for n in (1, 2)
]
NO_TOOL_CASE_IDS = ["none-01", "none-02"]


def results(**arm_overrides: Any) -> dict[str, Any]:
    """Every condition of reading A holding on every arm; tests then break exactly one thing."""
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in "abcde":
        arms[arm] = {}
        for cid in TOOL_CASE_IDS + NO_TOOL_CASE_IDS:
            tool = cid in TOOL_CASE_IDS
            arms[arm][cid] = {
                "kind": "tool" if tool else "none",
                "status": 200,
                "q3": True,
                "q4": True,
                "split_specials": 0,
                "stop_reason": 50 if tool else 106,
                # ruling P11: the four completeness facts; ruling P13: q7 on the stream arms
                "done": True,
                "finish_reason": "stop",
                "error": None,
                "q7": arm in "bce",
            }
    return {
        "arms": arms,
        "q8": [
            {"hits": 970, "queries": 1000},
            {"hits": 95, "queries": 100},
            {"hits": 1000, "queries": 1000},
        ],
        "q9": {
            "string": [{"status": 200, "prompt_ends_48": True}] * 3,
            "list": [{"status": 200, "prompt_ends_48": True}] * 3,
        },
        "q10": [True] * 10,
        **arm_overrides,
    }


def _set(res: dict[str, Any], arm: str, cid: str, **kv: Any) -> None:
    res["arms"][arm][cid].update(kv)


def test_reading_a_prefers_b_when_both_qualify() -> None:
    out = attr("reading")(results())
    assert out["reading"] == "A" and out["arm"] == "b"
    assert out["stop_id"] == "stop_reason"
    assert out["tools_in_prefill"] is True
    assert out["continuation_content"] == "string"
    assert out["seed_tokenizer"] == "tokenize"


@pytest.mark.parametrize(
    "breaks_b",
    [
        pytest.param({"status": 400}, id="status-400"),
        pytest.param({"q3": False}, id="q3"),
        pytest.param({"q4": False}, id="q4"),
        pytest.param({"split_specials": 1}, id="split"),
    ],
)
def test_reading_a_on_c_when_only_b_fails(breaks_b: dict[str, Any]) -> None:
    res = results()
    _set(res, "b", "click-02", **breaks_b)
    out = attr("reading")(res)
    assert out["reading"] == "A" and out["arm"] == "c"


@pytest.mark.parametrize(
    ("cid", "breaks"),
    [
        pytest.param("type-01", {"status": 500}, id="status-on-a-tool-case"),
        pytest.param("none-01", {"status": 400}, id="status-on-a-no-tool-case"),
        pytest.param("none-02", {"q3": False}, id="q3-on-a-no-tool-case"),
        pytest.param("scroll-02", {"q4": False}, id="q4-on-a-tool-case"),
        pytest.param("none-01", {"split_specials": 1}, id="split-on-a-no-tool-case"),
    ],
)
def test_reading_b_when_neither_b_nor_c_qualifies(cid: str, breaks: dict[str, Any]) -> None:
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, cid, **breaks)
    out = attr("reading")(res)
    assert out["reading"] == "B" and out["arm"] is None


def test_reading_b_when_b_and_c_fail_on_different_conditions() -> None:
    """Reading A needs one arm meeting every condition, not each condition met by some arm."""
    res = results()
    _set(res, "b", "key-01", q4=False)
    _set(res, "c", "click-01", status=400)
    assert attr("reading")(res)["reading"] == "B"


def test_reading_ignores_arms_a_d_and_e() -> None:
    """Only (b) and (c) can give A: perfect (a), (d), (e) do not rescue a failing (b) and (c), and failing ones do
    not spoil a perfect (b)."""
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "search-02", q4=False)
    assert attr("reading")(res)["reading"] == "B"
    res = results()
    for arm in ("a", "d", "e"):
        for cid in TOOL_CASE_IDS:
            _set(res, arm, cid, status=400, q3=False, q4=False, split_specials=3)
    out = attr("reading")(res)
    assert out["reading"] == "A" and out["arm"] == "b"


def test_q4_on_a_no_tool_case_does_not_decide_the_reading() -> None:
    """Too-wide direction: q4 is required on the 10 tool cases only (criterion 6)."""
    res = results()
    for arm in ("b", "c"):
        for cid in NO_TOOL_CASE_IDS:
            _set(res, arm, cid, q4=False)
    out = attr("reading")(res)
    assert out["reading"] == "A" and out["arm"] == "b"


def test_adversarial_dropped_string_delimiters_read_b_not_a() -> None:
    """The task's failure mode: content keeps ``<|tool_call>``/``<tool_call|>`` but drops ``<|"|>``. q4 comes from
    the probe's own comparison on that capture, for (b) and (c), on one tool case."""
    got = attr("reassemble")(sse(REASONING_PIECES, [ANSWER_NO_QUOTES]))
    q4 = attr("content_identical")(got["content"], tool_ids())
    assert q4 is False
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "search-01", q4=q4)
    assert attr("reading")(res)["reading"] == "B"


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(None, id="missing"),
        pytest.param(107, id="int-not-a-stop"),
        pytest.param("50", id="string"),
        pytest.param("<|tool_response>", id="stop-text"),
        pytest.param(True, id="bool-is-not-an-int"),
    ],
)
def test_stop_id_replans_unless_every_tool_case_gives_a_stop_int(bad: Any) -> None:
    res = results()
    for arm in "abcde":
        _set(res, arm, "click-01", stop_reason=bad)
    out = attr("reading")(res)
    assert out["reading"] == "A"
    assert out["stop_id"] != "stop_reason"


@pytest.mark.parametrize("stop", [1, 106, 50])
def test_stop_id_accepts_each_stop_and_ignores_no_tool_cases(stop: int) -> None:
    res = results()
    for arm in "abcde":
        for cid in TOOL_CASE_IDS:
            _set(res, arm, cid, stop_reason=stop)
        for cid in NO_TOOL_CASE_IDS:
            _set(res, arm, cid, stop_reason=None)
    assert attr("reading")(res)["stop_id"] == "stop_reason"


def test_stop_id_checks_only_the_chosen_arm() -> None:
    """Ruling P3: with (b) chosen, a bad stop in (a), (c), (d) or (e) does not spoil the stop id; in (b) it does.
    With (c) chosen ((b) fails q4), a bad stop in (b) does not spoil it; in (c) it does."""
    res = results()
    for arm in "acde":
        _set(res, arm, "click-01", stop_reason=None)
    out = attr("reading")(res)
    assert out["arm"] == "b" and out["stop_id"] == "stop_reason"
    res = results()
    _set(res, "b", "click-01", stop_reason=None)
    assert attr("reading")(res)["stop_id"] != "stop_reason"
    res = results()
    _set(res, "b", "key-02", q4=False, stop_reason=None)
    out = attr("reading")(res)
    assert out["arm"] == "c" and out["stop_id"] == "stop_reason"
    _set(res, "c", "type-02", stop_reason=107)
    assert attr("reading")(res)["stop_id"] != "stop_reason"


def test_tools_in_prefill_fails_on_a_mixed_q8() -> None:
    """Ruling P4: every case must be at or above 0.9; one case below fails, even with the others at 1.0."""
    res = results(
        q8=[
            {"hits": 1000, "queries": 1000},
            {"hits": 1000, "queries": 1000},
            {"hits": 899, "queries": 1000},
        ]
    )
    assert attr("reading")(res)["tools_in_prefill"] is False


def test_continuation_content_is_not_silently_switched_to_list() -> None:
    """Ruling P5: a failed string continuation re-plans T-0091; it does not pick list content even when the list
    form is accepted ending in 48."""
    res = results()
    res["q9"]["string"] = [{"status": 400, "prompt_ends_48": False}] * 3
    assert attr("reading")(res)["continuation_content"] not in ("string", "list")


def test_tools_in_prefill_at_the_boundary_and_below() -> None:
    """q8 at least 0.9 (inclusive, B11): exactly 9/10 counts; 899/1000 on every case does not; an unreadable
    ``/metrics`` (None) decides nothing (ruling P18: it is a re-run, not False; rows below)."""
    res = results(q8=[{"hits": 9, "queries": 10}] * 3)
    assert attr("reading")(res)["tools_in_prefill"] is True
    res = results(q8=[{"hits": 899, "queries": 1000}] * 3)
    assert attr("reading")(res)["tools_in_prefill"] is False
    res = results(q8=[{"hits": None, "queries": None}] * 3)
    assert attr("reading")(res)["tools_in_prefill"] is not True, (
        "an unreadable /metrics never gives tools"
    )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"status": 400, "prompt_ends_48": False}, id="refused"),
        pytest.param({"status": 200, "prompt_ends_48": False}, id="accepted-but-prompt-not-48"),
    ],
)
def test_continuation_content_is_string_only_if_string_is_accepted_ending_in_48(
    bad: dict[str, Any],
) -> None:
    res = results()
    res["q9"]["string"] = [{"status": 200, "prompt_ends_48": True}] * 2 + [bad]
    assert attr("reading")(res)["continuation_content"] != "string"


def test_continuation_content_is_string_whatever_the_list_form_does() -> None:
    res = results()
    res["q9"]["list"] = [{"status": 400, "prompt_ends_48": False}] * 3
    assert attr("reading")(res)["continuation_content"] == "string"


@pytest.mark.parametrize(
    "q10",
    [
        pytest.param([True] * 9 + [False], id="nine-of-ten"),
        pytest.param([True] * 9, id="only-nine-measured"),
        pytest.param([], id="none-measured"),
    ],
)
def test_seed_tokenizer_is_tokenize_only_on_ten_of_ten(q10: list[bool]) -> None:
    assert attr("reading")(results(q10=q10))["seed_tokenizer"] != "tokenize"


# =========================================================================== image redaction


def _find_digests(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        found = [obj] if {"bytes", "sha256"} <= set(obj) else []
        return found + [d for v in obj.values() for d in _find_digests(v)]
    if isinstance(obj, list):
        return [d for v in obj for d in _find_digests(v)]
    return []


def test_redact_replaces_the_image_data_url_by_its_size_and_sha256() -> None:
    body = _base_body()
    before = copy.deepcopy(body)
    raw = (SET_DIR / "screens" / "d01_login.jpg").read_bytes()
    out = attr("redact")(body)
    assert body == before, "redact mutated the body that is sent to the engine"
    text = json.dumps(out)
    assert "base64" not in text and "data:image" not in text
    assert base64.b64encode(raw).decode()[:64] not in text
    assert {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} in [
        {"bytes": d["bytes"], "sha256": d["sha256"]} for d in _find_digests(out)
    ]
    assert out["messages"][0] == body["messages"][0] and out["tools"] == body["tools"]


# =========================================================================== main(): loopback, clean tree, files


class _Engine(BaseHTTPRequestHandler):
    """A canned loopback engine: every chat request answers (stream or not), plus ``/metrics``, ``/tokenize``,
    ``/v1/models`` and ``/health``. It only has to let the probe run; nothing here is a measurement."""

    calls: list[tuple[str, str]] = []
    queries = 0

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, code: int, ctype: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        type(self).calls.append(("GET", self.path))
        if self.path.endswith("/metrics"):
            q = type(self).queries
            text = f"vllm:prefix_cache_queries_total {q}\nvllm:prefix_cache_hits_total {q * 95 // 100}\n"
            self._send(200, "text/plain", text.encode())
        elif self.path.endswith("/models"):
            self._send(200, "application/json", json.dumps({"data": [{"id": MODEL}]}).encode())
        else:
            self._send(200, "application/json", b"{}")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        type(self).calls.append(("POST", self.path))
        if self.path.endswith("/tokenize"):
            ids = vocab.encode(str(body.get("prompt", "")))
            self._send(
                200, "application/json", json.dumps({"tokens": ids, "count": len(ids)}).encode()
            )
            return
        type(self).queries += 1000
        ids = tool_ids()
        if body.get("stream"):
            lines = sse(REASONING_PIECES, CONTENT_PIECES)
            self._send(200, "text/event-stream", ("\n".join(lines) + "\n").encode())
            return
        msg = {"role": "assistant", "reasoning": THOUGHT, "content": ANSWER}
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": "stop",
                    "stop_reason": 50,
                    "token_ids": ids,
                }
            ],
            "prompt_token_ids": [2, 48],
        }
        self._send(200, "application/json", json.dumps(payload).encode())


@pytest.fixture
def engine() -> Iterator[str]:
    _Engine.calls = []
    _Engine.queries = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway committed tree with the inputs a probe might read relative to ``--repo``."""
    top = tmp_path / "repo"
    shutil.copytree(
        SET_DIR, top / "tests" / "eval" / "toolcall", ignore=shutil.ignore_patterns("__pycache__")
    )
    (top / "services").mkdir()
    shutil.copy(REPO / "services" / "versions.lock", top / "services" / "versions.lock")
    shutil.copy(REPO / ".gitignore", top / ".gitignore")
    _git(top, "init", "-q")
    _git(top, "add", "-A")
    _git(top, "commit", "-q", "-m", "fixture")
    return top


@pytest.fixture
def sockets(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Record (and pass through) every client connect and every name resolution in this process."""
    seen: dict[str, list[Any]] = {"connect": [], "getaddrinfo": []}
    real_connect = socket.socket.connect
    real_gai = socket.getaddrinfo

    def connect(self: socket.socket, address: Any) -> Any:
        seen["connect"].append(address)
        return real_connect(self, address)

    def gai(host: Any, *args: Any, **kw: Any) -> Any:
        seen["getaddrinfo"].append(host)
        return real_gai(host, *args, **kw)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", gai)
    return seen


def _main(argv: list[str]) -> int:
    main = attr("main")
    try:
        rc = main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return int(rc)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://10.0.0.5:8001/v1", id="private-v4"),
        pytest.param("http://192.0.2.1:8001/v1", id="test-net"),
        pytest.param("http://[2001:db8::1]:8001/v1", id="v6"),
        pytest.param("http://example.com:8001/v1", id="name-never-resolved"),
    ],
)
def test_a_non_loopback_url_is_refused_before_any_socket(
    url: str, repo: Path, tmp_path: Path, sockets: dict[str, list[Any]]
) -> None:
    rc = _main(["--base-url", url, "--repo", str(repo), "--out-dir", str(tmp_path / "out")])
    assert rc != 0
    assert sockets["connect"] == [], sockets["connect"]
    assert "example.com" not in sockets["getaddrinfo"]


def test_a_dirty_tree_is_refused_before_any_socket(
    engine: str, repo: Path, tmp_path: Path, sockets: dict[str, list[Any]]
) -> None:
    (repo / "services" / "versions.lock").write_text("edited\n")
    rc = _main(["--base-url", engine, "--repo", str(repo), "--out-dir", str(tmp_path / "out")])
    assert rc != 0
    assert sockets["connect"] == [] and _Engine.calls == [], _Engine.calls


def test_a_clean_tree_on_loopback_runs_and_writes_no_base64(
    engine: str, repo: Path, tmp_path: Path, sockets: dict[str, list[Any]]
) -> None:
    """The too-wide twin of the two refusals (the same tree committed, a loopback URL): the probe reaches the
    engine, and nothing it writes holds image bytes: no base64 anywhere, the screens appear only as sha256."""
    out = tmp_path / "out"
    logs_before = (
        sorted(p.name for p in (REPO / "logs").glob("t0087-probe-*"))
        if (REPO / "logs").is_dir()
        else []
    )
    _main(["--base-url", engine, "--repo", str(repo), "--out-dir", str(out)])
    posts = [p for m, p in _Engine.calls if m == "POST" and p.endswith("/chat/completions")]
    assert len(posts) >= 12 * 5, (
        f"the probe did not run its arms against the loopback engine: {_Engine.calls[:5]}"
    )
    assert all(a[0] in ("127.0.0.1", "::1") for a in sockets["connect"] if isinstance(a, tuple))
    files = [p for p in out.rglob("*") if p.is_file()]
    assert {"report.json", "report.md"} <= {p.name for p in files}, sorted(p.name for p in files)
    screens = {
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (SET_DIR / "screens").glob("*.jpg")
    }
    blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)
    assert "base64" not in blob and "data:image" not in blob
    for p in (SET_DIR / "screens").glob("*.jpg"):
        assert base64.b64encode(p.read_bytes()).decode()[100:164] not in blob, p.name
    assert any(h in blob for h in screens), (
        "no screen sha256 recorded: the image parts were not redacted in place"
    )
    logs_after = (
        sorted(p.name for p in (REPO / "logs").glob("t0087-probe-*"))
        if (REPO / "logs").is_dir()
        else []
    )
    assert logs_after == logs_before, (
        "--out-dir was ignored: the probe wrote under the repository's logs/"
    )


# =========================================================================== ruling P7: each arm against its own ids

#: a second sample of the same tool case: own ids and own text agree, but the ids differ from the first sample's
#: (the sampler is unseeded, so two requests may serve two different, each self-consistent, answers)
THOUGHT_Y = "The user asks about Porto weather."
SEGMENTS_Y = [
    "<|channel>thought\n",
    THOUGHT_Y,
    "<channel|>",
    "Looking it up.",
    "<|tool_call>",
    "call:google_search{query:",
    '<|"|>',
    "weather porto",
    '<|"|>',
    "}",
    "<tool_call|>",
]
#: the first sample with the string delimiters left out of the ids (a self-consistent no-quotes answer)
SEGMENTS_NQ = [s for s in SEGMENTS if s != '<|"|>']


def capture(segments: Sequence[str], stop: int = 50) -> dict[str, Any]:
    """A served answer: one reasoning group (header, thought, close: the thought text and those three segments'
    ids), then one content group per later segment (its text and its ids), then the stop id."""
    seg = _seg_ids(segments)
    ids = [t for g in seg for t in g]
    assert replay.detokenize(ids) == "".join(segments), "fixture: the segments do not round-trip"
    groups = [("reasoning", segments[1], seg[0] + seg[1] + seg[2])]
    groups += [("content", text, g) for text, g in zip(segments[3:], seg[3:], strict=True)]
    return {"groups": groups, "stop": stop}


def with_texts(cap: dict[str, Any], texts: Sequence[str]) -> dict[str, Any]:
    """The same ids with other content texts (one per content group): a capture whose text disagrees with its ids."""
    content = [g for g in cap["groups"] if g[0] == "content"]
    assert len(texts) == len(content)
    return {
        "groups": [cap["groups"][0]]
        + [("content", t, g[2]) for t, g in zip(texts, content, strict=True)],
        "stop": cap["stop"],
    }


X = capture(SEGMENTS)
Y = capture(SEGMENTS_Y)
NQ = capture(SEGMENTS_NQ)
#: arm (a)'s served content for sample X: the special pieces dropped (T-0080d finding 1); no ids are asked for
X_DEFAULT = with_texts(X, ["" if s.startswith("<") else s for s in SEGMENTS[3:]])
#: sample X's ids, served with sample NQ's text: content that disagrees with its own ids
X_IDS_NQ_TEXT = with_texts(X, ["" if s == '<|"|>' else s for s in SEGMENTS[3:]])


def _ids_of(cap: dict[str, Any]) -> list[int]:
    return [t for g in cap["groups"] for t in g[2]] + [cap["stop"]]


def test_the_p7_fixtures_are_what_they_claim() -> None:
    """The fixtures, checked by hand: X and Y differ in ids and each agrees with its own text; X_IDS_NQ_TEXT's
    content is NQ's answer region but its ids are X's; arm (a)'s content is X's answer without the specials."""
    assert _ids_of(X) != _ids_of(Y)
    assert _ids_of(X) == tool_ids()
    joined = "".join(g[1] for g in X_IDS_NQ_TEXT["groups"][1:])
    assert joined == "".join(SEGMENTS_NQ[3:]) == ANSWER_NO_QUOTES
    assert _ids_of(X_IDS_NQ_TEXT) == _ids_of(X)
    assert "".join(g[1] for g in X_DEFAULT["groups"][1:]) == ANSWER_DEFAULT


def _arm_of(body: dict[str, Any]) -> str:
    """Which arm a chat body is, from criterion 1's fields alone: (b) and (e) are the same request under P7."""
    if not body.get("stream"):
        return "d"
    if "skip_special_tokens" not in body:
        return "a"
    if body.get("spaces_between_special_tokens") is False:
        return "c"
    return "be"


class _Scripted(_Engine):
    """A loopback engine serving a per-arm capture: ``plan[arm]`` for ``a``, ``c`` and ``d`` (every non-stream
    request, so q8 and q9 too); ``plan["be"]`` is a list, the n-th (b)/(e)-shaped request of a case getting the
    n-th entry (so (b) and (e) can serve different samples without the double knowing which came first). Ids
    are served only when the body asks for ``return_token_ids`` and the capture has ``"ids"`` (default true), as
    vLLM does.

    Review rows (P11-P15) add: every POST body is recorded (``bodies``); ``/metrics`` counts per request, a 1-token
    prefill adding 1000 queries and 0 hits, any other chat request 1000 queries and 950 hits, so a delta over the
    wrong window shows; ``plan["prompt"](body)`` gives a non-stream response's ``prompt_token_ids`` (default
    ``[2, 48]``); a capture may drop its final chunk (``"final": False``) and add raw ``"tail"`` lines before
    ``[DONE]``; ``plan["hook"](handler, body) -> bool`` may answer a chat request itself (400, 3xx, reset);
    ``/tokenize`` prepends BOS (2) unless the body says ``add_special_tokens: false``, as vLLM does."""

    plan: dict[str, Any] = {}
    seen: dict[str, int] = {}
    bodies: list[tuple[str, dict[str, Any]]] = []
    hits = 0

    def do_GET(self) -> None:
        cls = type(self)
        if self.path.endswith("/metrics"):
            cls.calls.append(("GET", self.path))
            if cls.plan.get("metrics") is False:
                self._send(404, "text/plain", b"not found")
                return
            text = (
                f"vllm:prefix_cache_queries_total {cls.queries}\n"
                f"vllm:prefix_cache_hits_total {cls.hits}\n"
            )
            self._send(200, "text/plain", text.encode())
            return
        super().do_GET()

    def finish(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            super().finish()

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        cls = type(self)
        cls.calls.append(("POST", self.path))
        cls.bodies.append((self.path, body))
        if self.path.endswith("/tokenize"):
            ids = vocab.encode(str(body.get("prompt", "")))
            if body.get("add_special_tokens") is not False:
                ids = [2, *ids]
            self._send(200, "application/json", json.dumps({"tokens": ids}).encode())
            return
        hook = cls.plan.get("hook")
        if hook is not None and hook(self, body):
            return
        cls.queries += 1000
        cls.hits += 0 if body.get("max_tokens") == 1 else 950
        arm = _arm_of(body)
        if arm == "be":
            key = json.dumps(body.get("messages"), sort_keys=True)
            i = cls.seen.get(key, 0)
            cls.seen[key] = i + 1
            cap = cls.plan["be"][i % len(cls.plan["be"])]
        else:
            cap = cls.plan[arm]
        want = body.get("return_token_ids") is True and cap.get("ids", True)
        if body.get("stream"):
            lines: list[str] = []
            for field, text, ids in cap["groups"]:
                lines += [_chunk({field: text}, **({"token_ids": ids} if want else {})), ""]
            for extra in cap.get("tail", []):
                lines += [extra, ""]
            if cap.get("final", True):
                last = {"token_ids": [cap["stop"]]} if want else {}
                lines += [_chunk({}, finish_reason="stop", stop_reason=cap["stop"], **last), ""]
            if cap.get("done", True):
                lines += ["data: [DONE]", ""]
            self._send(200, "text/event-stream", ("\n".join(lines) + "\n").encode())
            return
        msg = {
            "role": "assistant",
            "reasoning": "".join(t for f, t, _ in cap["groups"] if f == "reasoning"),
            "content": "".join(t for f, t, _ in cap["groups"] if f == "content"),
        }
        choice: dict[str, Any] = {
            "index": 0,
            "message": msg,
            "finish_reason": "stop",
            "stop_reason": cap["stop"],
        }
        if want:
            choice["token_ids"] = _ids_of(cap)
        prompt = cls.plan.get("prompt")
        payload = {
            "choices": [choice],
            "prompt_token_ids": prompt(body) if prompt is not None else [2, 48],
        }
        self._send(200, "application/json", json.dumps(payload).encode())


class _QuietServer(ThreadingHTTPServer):
    """The scripted engine's server: a handler that resets its own connection must not print a traceback."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class _Ran:
    """What one scripted run left behind: the exit code, the request bodies, the output directory."""

    def __init__(self, rc: int, bodies: list[tuple[str, dict[str, Any]]], out: Path) -> None:
        self.rc = rc
        self.bodies = bodies
        self.out = out

    @property
    def report(self) -> dict[str, Any]:
        path = self.out / "report.json"
        assert path.is_file(), f"the probe wrote no report.json under {self.out}"
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return report

    def chat_bodies(self) -> list[dict[str, Any]]:
        return [b for p, b in self.bodies if p.endswith("/chat/completions")]

    def files_text(self) -> str:
        return "\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in self.out.rglob("*")
            if p.is_file()
        )


def _run_full(repo: Path, out: Path, base_url: str | None = None, **plan: Any) -> _Ran:
    """``main`` against a scripted loopback engine (or ``base_url`` instead of its own URL)."""
    handler = type(
        "_Plan",
        (_Scripted,),
        {"plan": plan, "seen": {}, "calls": [], "queries": 0, "hits": 0, "bodies": []},
    )
    srv = _QuietServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = base_url or f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        rc = _main(["--base-url", url, "--repo", str(repo), "--out-dir", str(out)])
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)
    return _Ran(rc, handler.bodies, out)


def _run(repo: Path, out: Path, **plan: Any) -> dict[str, Any]:
    """``main`` against a scripted loopback engine; returns ``report.json``."""
    return _run_full(repo, out, **plan).report


def _tool_cases(report: dict[str, Any]) -> list[str]:
    rows = {
        r["id"]: r
        for r in (
            json.loads(x) for x in (SET_DIR / "cases.jsonl").read_text().splitlines() if x.strip()
        )
    }
    cids = list(report["arms"]["b"])
    assert len(cids) == 12, cids
    return [c for c in cids if rows[c]["tool"] is not None]


def _verdict(report: dict[str, Any]) -> tuple[Any, Any]:
    return report["reading"]["reading"], report["reading"]["arm"]


def test_p7_own_ids_that_differ_from_d_still_read_a(repo: Path, tmp_path: Path) -> None:
    """Adversarial (P7): (d) serves sample X, while (b), (c) and (e) each serve sample Y, whose content agrees with
    its own ids. Judged against its own ids, (b) meets every condition, so the reading is A on (b); judging (b)'s
    text against (d)'s ids would read B on sampler noise alone."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=Y, d=X, be=[Y])
    tool = _tool_cases(report)
    for arm in ("b", "c", "e"):
        bad = [
            cid
            for cid, r in report["arms"][arm].items()
            if r.get("q3") is not True or (cid in tool and r.get("q4") is not True)
        ]
        assert bad == [], f"ruling P7: arm ({arm}) q3/q4 must be judged against its own ids: {bad}"
    assert _verdict(report) == ("A", "b"), (
        f"ruling P7: own ids agree with own content, so the reading is A on (b), not {_verdict(report)}"
    )


def test_p7_content_that_disagrees_with_its_own_ids_reads_b(repo: Path, tmp_path: Path) -> None:
    """Adversarial (P7): (b), (c) and (e) serve sample X's ids with content that drops ``<|"|>``, and (d) serves a
    self-consistent answer whose text is exactly that content. Against their own ids, (b) and (c) fail q4 on every
    tool case: reading B. Against (d)'s ids they would wrongly pass."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X_IDS_NQ_TEXT, d=NQ, be=[X_IDS_NQ_TEXT])
    tool = _tool_cases(report)
    for arm in ("b", "c", "e"):
        passed = [cid for cid in tool if report["arms"][arm][cid].get("q4") is not False]
        assert passed == [], (
            f"ruling P7: arm ({arm}) content disagrees with its own ids, yet q4 passed on {passed}"
        )
    assert _verdict(report) == ("B", None), (
        f"ruling P7: content that disagrees with its own ids reads B, not {_verdict(report)}"
    )


@pytest.mark.parametrize(
    "arm_a",
    [
        pytest.param(X_DEFAULT, id="a-drops-the-specials"),
        pytest.param(X, id="a-agrees-with-d"),
    ],
)
def test_p7_a_disagreement_in_arm_a_does_not_change_the_reading(
    arm_a: dict[str, Any], repo: Path, tmp_path: Path
) -> None:
    """Adversarial (P7): arm (a) is the noise-confounded baseline and decides nothing. With (b)-(e) all serving
    sample X, the reading is A on (b) whether (a)'s content agrees with (d)'s decode or not."""
    report = _run(repo, tmp_path / "out", a=arm_a, c=X, d=X, be=[X])
    assert _verdict(report) == ("A", "b"), (
        f"ruling P7: arm (a) decides nothing, yet the reading is {_verdict(report)}"
    )


def test_p7_arm_a_comparison_with_d_is_recorded_and_labelled_noise_confounded(
    repo: Path, tmp_path: Path
) -> None:
    """Ruling P7: arm (a)'s comparison with arm (d)'s decode is recorded per case as a diagnostic labelled
    ``noise-confounded``. Proposed surface: ``report["arms"]["a"][cid]["vs_d"] == {"q3": bool, "q4": bool,
    "label": "noise-confounded"}``. The values are real: (a) drops the specials, so q4 against (d)'s decode is
    False on every tool case while q3 holds (its reasoning is (d)'s thought)."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    tool = _tool_cases(report)
    for cid, r in report["arms"]["a"].items():
        vs = r.get("vs_d")
        assert isinstance(vs, dict), (
            f"ruling P7: arm (a) {cid} records no comparison with (d) (vs_d)"
        )
        assert vs.get("label") == "noise-confounded", (
            f"ruling P7: arm (a) {cid} comparison with (d) is not labelled noise-confounded: {vs}"
        )
        assert vs.get("q3") is True, f"arm (a) {cid}: vs_d q3 should hold (same thought): {vs}"
        if cid in tool:
            assert vs.get("q4") is False, (
                f"arm (a) {cid}: vs_d q4 should fail (specials dropped): {vs}"
            )


def _b_equal_e(report: dict[str, Any]) -> dict[str, Any]:
    """``b_ids_equal_e_ids`` per case, from arm (b)'s or arm (e)'s row (either place is accepted)."""
    got: dict[str, Any] = {}
    for cid in report["arms"]["b"]:
        for arm in ("b", "e"):
            row = report["arms"][arm][cid]
            if "b_ids_equal_e_ids" in row:
                got[cid] = row["b_ids_equal_e_ids"]
    return got


@pytest.mark.parametrize(
    ("be", "equal"),
    [
        pytest.param([X, X], True, id="replicates-agree"),
        pytest.param([X, Y], False, id="replicates-differ"),
    ],
)
def test_p7_b_ids_equal_e_ids_is_recorded_per_case_and_decides_nothing(
    be: list[dict[str, Any]], equal: bool, repo: Path, tmp_path: Path
) -> None:
    """Ruling P7: (e) is a replicate of (b); ``b_ids_equal_e_ids`` is recorded per case as the noise level. One of
    (b)/(e) serves X and the other X or Y (each self-consistent): the flag follows the ids, and the reading is A on
    (b) either way."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=be)
    flags = _b_equal_e(report)
    assert sorted(flags) == sorted(report["arms"]["b"]), (
        f"ruling P7: b_ids_equal_e_ids is not recorded on every case: {sorted(flags)}"
    )
    assert set(flags.values()) == {equal}, f"b_ids_equal_e_ids should be {equal}: {flags}"
    assert _verdict(report) == ("A", "b"), (
        f"ruling P7: b_ids_equal_e_ids decides nothing, yet the reading is {_verdict(report)}"
    )


# =========================================================================== review rows (rulings P11-P15, survivors)

NO_TOOL = capture(NO_TOOL_SEGMENTS, stop=106)
#: sample X's ids with ``<|tool_call>`` split over two content chunks (the joined content is still X's answer)
SPLIT = with_texts(
    X,
    [
        "Checking now.<|tool",
        "_call>",
        "call:google_search{query:",
        '<|"|>',
        "weather lisbon",
        '<|"|>',
        "}",
        "<tool_call|>",
    ],
)
#: sample X served with no per-chunk ids even when asked
NOIDS = {**X, "ids": False}
#: the in-band error vLLM sends inside a 200 stream
STREAM_ERROR = {
    "object": "error",
    "message": "EngineCore encountered an issue.",
    "type": "InternalServerError",
    "param": None,
    "code": 500,
}
#: both reviews' truncated stream: 200, the text stops after ``call:google_search{query:``, an in-band error
#: object, then ``[DONE]``; no final chunk, so no finish_reason. Its own ids agree with its own text.
TRUNC = {
    "groups": X["groups"][:4],
    "stop": 50,
    "final": False,
    "tail": ["data: " + json.dumps({"error": STREAM_ERROR})],
}
#: criterion 5's draft call text of sample X, by hand: the call after ``<|tool_call>``, up to and including the
#: close tag (``router.route`` on X's host decode; no trailing whitespace to strip)
DRAFT_X = 'call:google_search{query:<|"|>weather lisbon<|"|>}<tool_call|>'
#: criterion 4's T4 continuation of sample X: the spoken prefix plus the call-open tag
CONTINUATION_X = "Checking now.<|tool_call>"


def test_the_review_fixtures_are_what_they_claim() -> None:
    assert "".join(g[1] for g in SPLIT["groups"][1:]) == ANSWER
    assert _ids_of(SPLIT) == _ids_of(X)
    assert (
        "".join(g[1] for g in TRUNC["groups"][1:])
        == "Checking now.<|tool_call>call:google_search{query:"
    )
    assert ANSWER.startswith(CONTINUATION_X) and ANSWER.endswith(DRAFT_X)
    assert ANSWER == CONTINUATION_X + DRAFT_X


def _first_data_url(body: Any) -> str | None:
    m = re.search(r'data:image/[^"\\]+', json.dumps(body))
    return m.group(0) if m else None


def _echo_400(h: _Scripted, body: dict[str, Any]) -> bool:
    """Every chat request is a 400 whose JSON echoes the request's image data URL, as pydantic does."""
    url = _first_data_url(body)
    echo = f", input_value='{url[:200]}...', input_type=str" if url else ""
    msg = f"1 validation error for ChatCompletionRequest\n  Input should be a valid URL [type=url_parsing{echo}]"
    data = json.dumps({"object": "error", "message": msg, "type": "BadRequestError", "code": 400})
    h._send(400, "application/json", data.encode())
    return True


def _reset_first_be(h: _Scripted, body: dict[str, Any]) -> bool:
    """The first (b)/(e)-shaped stream: a 200, one reasoning chunk, then a TCP reset (an OSError mid-stream)."""
    state = type(h).plan["state"]
    if _arm_of(body) != "be" or state.get("reset"):
        return False
    state["reset"] = True
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Connection", "close")
    h.end_headers()
    h.wfile.write((_chunk({"reasoning": THOUGHT}) + "\n\n").encode())
    h.wfile.flush()
    h.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    h.connection.close()
    h.close_connection = True
    return True


def _q9_prompt(string_ids: list[int], list_ids: list[int]) -> Any:
    """``prompt_token_ids`` per q9 form (the last message is the assistant continuation); ``[2, 48]`` otherwise."""

    def prompt(body: dict[str, Any]) -> list[int]:
        last = body["messages"][-1]
        if last.get("role") != "assistant":
            return [2, 48]
        return string_ids if isinstance(last.get("content"), str) else list_ids

    return prompt


def _q9_bodies(ran: _Ran) -> list[dict[str, Any]]:
    return [b for b in ran.chat_bodies() if b["messages"][-1].get("role") == "assistant"]


def _decisions(reading: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in reading.items() if k != "why"}


# ----------------------------------------------------------------------------- survivor: q6 in the probe output


def test_review_q6_is_counted_on_each_stream_arms_own_content_chunks(
    repo: Path, tmp_path: Path
) -> None:
    """Survivors "q6 hard-coded to 0" and "q6 counted on reasoning chunks": (b), (c) and (e) split
    ``<|tool_call>`` over two content chunks (the reasoning is one whole chunk), so q6 is 1 on every case and the
    reading is B, though the joined content is byte-identical."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=SPLIT, d=X, be=[SPLIT])
    for arm in ("b", "c", "e"):
        got = {cid: r.get("split_specials") for cid, r in report["arms"][arm].items()}
        assert set(got.values()) == {1}, f"q6: arm ({arm}) split <|tool_call> on every case: {got}"
    assert _verdict(report) == ("B", None), f"q6 must block reading A: {_verdict(report)}"


# ----------------------------------------------------------------------------- survivor: missing own ids


def test_review_missing_own_ids_fail_q3_and_q4(repo: Path, tmp_path: Path) -> None:
    """Survivor "missing own ids passing q3/q4": (b), (c) and (e) stream X's text without per-chunk ids. With no
    ids of its own an arm's q3/q4 are False (not passed, not None), and the reading is B."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=NOIDS, d=X, be=[NOIDS])
    for arm in ("b", "c", "e"):
        bad = {
            cid: (r.get("q3"), r.get("q4"))
            for cid, r in report["arms"][arm].items()
            if r.get("q3") is not False or r.get("q4") is not False
        }
        assert bad == {}, f"arm ({arm}) has no ids of its own, so q3/q4 must be False: {bad}"
    assert _verdict(report) == ("B", None), f"no own ids cannot read A: {_verdict(report)}"


def test_review_missing_own_ids_are_listed_in_confounds(repo: Path, tmp_path: Path) -> None:
    """Survivor "the confounds list": an arm without own ids is named in ``confounds`` with its cases (P7 log);
    a run where every arm has ids lists no such confound for (b)."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=NOIDS, d=X, be=[NOIDS])
    confounds = report.get("confounds") or []
    for arm in ("b", "c", "e"):
        named = [c for c in confounds if f"({arm})" in c and "search-01" in c]
        assert named, f"confounds does not name arm ({arm}) and its cases: {confounds}"
    clean = _run(repo, tmp_path / "out2", a=X_DEFAULT, c=X, d=X, be=[X])
    assert not [c for c in clean.get("confounds") or [] if "(b)" in c], clean.get("confounds")


# ----------------------------------------------------------------------------- survivors: q8


def test_review_q8_prefills_carry_tools_and_the_control_does_not(
    repo: Path, tmp_path: Path
) -> None:
    """Survivors "with-tools prefill sent without tools" and "control sent with tools" (criterion 3): six 1-token
    prefills, three with ``tools`` and three without, each followed by its generate request on the same messages
    with the same ``tools`` as the pass-1 body."""
    ran = _run_full(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    chats = ran.chat_bodies()
    pre = [i for i, b in enumerate(chats) if b.get("max_tokens") == 1]
    assert len(pre) == 6, f"criterion 3: 3 + 3 one-token prefills, got {len(pre)}"
    with_tools = [i for i in pre if chats[i].get("tools")]
    without = [i for i in pre if "tools" not in chats[i]]
    assert len(with_tools) == 3, f"criterion 3: three prefills carry tools, got {len(with_tools)}"
    assert len(without) == 3, (
        f"criterion 3: the three control prefills carry no tools, got {len(without)}"
    )
    for i in pre:
        gen = chats[i + 1]
        assert gen.get("max_tokens") != 1 and gen["messages"] == chats[i]["messages"], (
            f"prefill {i} is not followed by its generate request"
        )
        assert gen.get("tools"), f"the generate request after prefill {i} carries no tools"
        if i in with_tools:
            assert gen["tools"] == chats[i]["tools"], (
                "criterion 3: identical tools in prefill and generate"
            )


def test_review_q8_delta_is_measured_over_the_generate_request_only(
    repo: Path, tmp_path: Path
) -> None:
    """Survivor "delta over prefill plus generate": the double's ``/metrics`` gives a cold prefill 1000 queries and
    0 hits, a generate 1000 queries and 950 hits. Over the generate only that is 950/1000 = 0.95 (tools in the
    prefill); over both it would be 950/2000."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    got = [(r.get("hits"), r.get("queries")) for r in report["q8"]]
    assert got == [(950, 1000)] * 3, f"q8 delta must cover the generate request only: {got}"
    assert report["reading"]["tools_in_prefill"] is True


@pytest.mark.parametrize(
    "q8",
    [
        pytest.param([{"hits": 1000, "queries": 1000}] * 2, id="two-of-three"),
        pytest.param([], id="empty"),
    ],
)
def test_review_q8_needs_all_three_cases(q8: list[dict[str, Any]]) -> None:
    """Survivors "the q8 count check dropped" and missing data ``q8 = []``: fewer than three measured cases never
    give tools in the prefill, however good the measured ones are."""
    assert attr("reading")(results(q8=q8))["tools_in_prefill"] is False, (
        f"q8: {len(q8)} measured cases cannot give tools in the prefill"
    )


# ----------------------------------------------------------------------------- survivors: q9


def test_review_q9_prompt_must_end_in_48_not_merely_contain_it(repo: Path, tmp_path: Path) -> None:
    """Survivor "``48 in prompt`` instead of the last id": both forms are served ``prompt_token_ids = [48, 2]``."""
    report = _run(
        repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X], prompt=_q9_prompt([48, 2], [48, 2])
    )
    got = [r.get("prompt_ends_48") for r in report["q9"]["string"]]
    assert got == [False] * 3, f"q9: [48, 2] does not end in 48: {got}"
    assert report["reading"]["continuation_content"] != "string"


def test_review_q9_string_and_list_are_not_swapped(repo: Path, tmp_path: Path) -> None:
    """Survivor "string and list swapped": the string form is served a prompt without 48, the list form one ending
    in 48. The string rows are False, the list rows True, and the continuation content is not ``string``."""
    report = _run(
        repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X], prompt=_q9_prompt([2, 7], [2, 48])
    )
    assert [r.get("prompt_ends_48") for r in report["q9"]["string"]] == [False] * 3, report["q9"]
    assert [r.get("prompt_ends_48") for r in report["q9"]["list"]] == [True] * 3, report["q9"]
    assert report["reading"]["continuation_content"] != "string"


def test_review_q9_continuation_bodies_carry_the_flags_and_the_hand_content(
    repo: Path, tmp_path: Path
) -> None:
    """Survivor "the continuation flags dropped" (criterion 4): three string and three list continuations of
    ``spoken_prefix + "<|tool_call>"``, each with ``continue_final_message: true`` and
    ``add_generation_prompt: false``."""
    ran = _run_full(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    bodies = _q9_bodies(ran)
    contents = [b["messages"][-1]["content"] for b in bodies]
    assert contents.count(CONTINUATION_X) == 3, contents
    assert contents.count([{"type": "text", "text": CONTINUATION_X}]) == 3, contents
    assert len(bodies) == 6, len(bodies)
    for b in bodies:
        assert b.get("continue_final_message") is True, "q9: continue_final_message must be true"
        assert b.get("add_generation_prompt") is False, "q9: add_generation_prompt must be false"


def test_review_q9_fewer_than_three_native_cases_replans(repo: Path, tmp_path: Path) -> None:
    """Survivor "fewer than 3 not replanning" (P9), at the ``main`` level: arm (d) never routes ``native`` (a no-tool
    answer), so q9 has no case and the continuation content is a replan."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=NO_TOOL, be=[X])
    assert len(report["q9"]["string"]) < 3, report["q9"]
    assert report["reading"]["continuation_content"] not in ("string", "list"), (
        "P9: fewer than 3 native cases must replan"
    )


@pytest.mark.parametrize(
    "string",
    [
        pytest.param([{"status": 200, "prompt_ends_48": True}] * 2, id="two-of-three"),
        pytest.param([], id="empty"),
    ],
)
def test_review_q9_needs_three_string_continuations(string: list[dict[str, Any]]) -> None:
    """Survivors "fewer than 3 not replanning" and missing data ``q9 = []`` in ``reading``."""
    res = results()
    res["q9"]["string"] = string
    assert attr("reading")(res)["continuation_content"] not in ("string", "list"), (
        f"P5/P9: {len(string)} string continuations must replan"
    )


# ----------------------------------------------------------------------------- survivors: q10


def test_review_q10_sends_the_hand_draft_without_special_tokens_and_matches(
    repo: Path, tmp_path: Path
) -> None:
    """Survivors "a trailing space in ``vocab.encode``" and "``add_special_tokens: true``" (criterion 5): ten
    ``/tokenize`` bodies carry the hand draft with ``add_special_tokens: false`` (the double prepends BOS
    otherwise, as vLLM does), all ten are equal, and the seed tokenizer is ``/tokenize``."""
    ran = _run_full(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    tok = [b for p, b in ran.bodies if p.endswith("/tokenize")]
    assert len(tok) == 10, f"criterion 5: ten /tokenize requests, got {len(tok)}"
    assert all(b.get("prompt") == DRAFT_X for b in tok), [b.get("prompt") for b in tok]
    assert all(b.get("add_special_tokens") is False for b in tok), (
        "criterion 5: add_special_tokens false"
    )
    report = ran.report
    assert [r.get("equal") for r in report["q10_rows"]] == [True] * 10, report["q10_rows"]
    assert report["reading"]["seed_tokenizer"] == "tokenize"


def test_review_q10_fewer_than_ten_drafts_is_not_tokenize(repo: Path, tmp_path: Path) -> None:
    """Survivor "fewer than 10 at the ``main`` level" (P9): no native draft, so no comparison, so not ``/tokenize``."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=NO_TOOL, be=[X])
    assert report["reading"]["seed_tokenizer"] != "tokenize", (
        "P9: fewer than 10 drafts is not /tokenize"
    )


# ----------------------------------------------------------------------------- survivor: the reading under B


def test_review_under_reading_b_the_stop_id_is_never_stop_reason() -> None:
    """Survivor "under B, ``stop_ok`` forced true" (P9): every arm gives a stop int, but no arm qualifies."""
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "click-01", q4=False)
    out = attr("reading")(res)
    assert out["reading"] == "B"
    assert out["stop_id"] != "stop_reason", "P9: under reading B the stop id is a replan"


# ----------------------------------------------------------------------------- survivors: refusals


def test_review_a_non_empty_out_dir_is_refused_before_any_socket(
    engine: str, repo: Path, tmp_path: Path, sockets: dict[str, list[Any]]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("earlier run\n")
    rc = _main(["--base-url", engine, "--repo", str(repo), "--out-dir", str(out)])
    assert rc != 0, "a non-empty --out-dir must be refused"
    assert sockets["connect"] == [] and _Engine.calls == [], _Engine.calls
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
    assert (out / "keep.txt").read_text() == "earlier run\n"


def test_review_a_cases_file_other_than_the_pinned_one_is_refused_before_any_socket(
    engine: str, repo: Path, tmp_path: Path, sockets: dict[str, list[Any]]
) -> None:
    cases = repo / "tests" / "eval" / "toolcall" / "cases.jsonl"
    cases.write_bytes(cases.read_bytes() + b"\n")
    _git(repo, "commit", "-q", "-am", "cases edited")
    rc = _main(["--base-url", engine, "--repo", str(repo), "--out-dir", str(tmp_path / "out")])
    assert rc != 0, "a cases.jsonl other than 460cd1c3... must be refused"
    assert sockets["connect"] == [] and _Engine.calls == [], _Engine.calls


# ----------------------------------------------------------------------------- survivor: transport errors


def test_review_a_transport_error_is_counted_and_exits_nonzero(repo: Path, tmp_path: Path) -> None:
    """Survivor "the ``transport_errors`` counter": one stream is reset by the peer after its 200."""
    ran = _run_full(
        repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X], hook=_reset_first_be, state={}
    )
    assert ran.rc != 0, "a transport error must make the run exit non-zero"
    assert ran.report.get("transport_errors") == 1, ran.report.get("transport_errors")


# ----------------------------------------------------------------------------- survivors: reading B's missing data


@pytest.mark.parametrize("c_ok", [False, True], ids=["c-empty-too", "c-perfect"])
def test_review_an_empty_arm_never_qualifies(c_ok: bool) -> None:
    res = results()
    res["arms"]["b"] = {}
    if not c_ok:
        res["arms"]["c"] = {}
    out = attr("reading")(res)
    assert (out["reading"], out["arm"]) == (("A", "c") if c_ok else ("B", None)), out


@pytest.mark.parametrize(
    "breaks",
    [
        pytest.param({"q3": None}, id="q3-none"),
        pytest.param({"split_specials": None}, id="split-specials-none"),
        pytest.param({"status": 201}, id="status-201"),
        pytest.param({"status": 204}, id="status-204"),
    ],
)
def test_review_missing_or_non_200_data_never_qualifies(breaks: dict[str, Any]) -> None:
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "type-02", **breaks)
    out = attr("reading")(res)
    assert out["reading"] == "B", f"{breaks} must not qualify: {out['reading']}"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(ANSWER + " ", id="trailing-space"),
        pytest.param("\n" + ANSWER, id="leading-newline"),
        pytest.param(ANSWER + "\n", id="trailing-newline"),
    ],
)
def test_review_q4_is_not_compared_after_strip(content: str) -> None:
    """Survivor "q4 compared after ``.strip()``": q4 is byte identity (criterion 2)."""
    assert attr("content_identical")(content, tool_ids()) is False, (
        f"q4 is byte identity: {content!r}"
    )


# ----------------------------------------------------------------------------- P11: truncated streams


def test_p11_reassembly_surfaces_an_in_band_error_object() -> None:
    """P11: a ``data: {"error": ...}`` line inside a 200 stream is kept verbatim (proposed key ``error``); a stream
    without one reports ``None``."""
    lines = sse(REASONING_PIECES, CONTENT_PIECES[:3])[:-4]
    lines += ["data: " + json.dumps({"error": STREAM_ERROR}), "", "data: [DONE]", ""]
    got = attr("reassemble")(lines)
    assert got.get("error", "<absent>") == STREAM_ERROR, (
        f"P11: in-band error not surfaced: {got.get('error', '<absent>')}"
    )
    assert got["done"] is True and got["finish_reason"] is None
    clean = attr("reassemble")(sse(REASONING_PIECES, CONTENT_PIECES))
    assert clean.get("error", "<absent>") is None, clean.get("error", "<absent>")


@pytest.mark.parametrize(
    "breaks",
    [
        pytest.param({"done": False}, id="not-done"),
        pytest.param({"finish_reason": None}, id="no-finish-reason"),
        pytest.param({"error": STREAM_ERROR}, id="in-band-error"),
    ],
)
def test_p11_a_truncated_case_never_qualifies(breaks: dict[str, Any]) -> None:
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "search-01", **breaks)
    out = attr("reading")(res)
    assert out["reading"] == "B", f"P11: {breaks} on (b) and (c) must not read A: {out['reading']}"


def test_p11_any_finish_reason_counts_as_finished() -> None:
    """P11's too-wide twin: a non-null ``finish_reason`` other than ``stop`` (``length``) is not a truncation."""
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "search-01", finish_reason="length")
    assert attr("reading")(res)["reading"] == "A"


def test_p11_the_truncated_capture_of_both_reviews_does_not_read_a(
    repo: Path, tmp_path: Path
) -> None:
    """P11, the reviewers' capture: (b), (c) and (e) answer 200, stop after ``call:google_search{query:``, send an
    in-band error object, then ``[DONE]``. Own ids agree with own text and no piece is split, so only P11 stands
    between this and reading A."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=TRUNC, d=X, be=[TRUNC])
    row = report["results"]["arms"]["b"]["search-01"]
    assert row.get("finish_reason", "<absent>") is None, (
        f"P11: results row lacks finish_reason: {row}"
    )
    assert row.get("error") == STREAM_ERROR, f"P11: results row lacks the in-band error: {row}"
    assert _verdict(report)[0] != "A", f"P11: a truncated stream read {_verdict(report)}"


def test_p11_results_rows_carry_the_four_completeness_facts(repo: Path, tmp_path: Path) -> None:
    """P11: ``results["arms"]`` carries status, done, finish_reason and error per case, so the reading can be
    re-derived; on a clean run they are 200, True, "stop", None, and the reading is complete."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    for arm in ("b", "c"):
        for cid, row in report["results"]["arms"][arm].items():
            got = {k: row.get(k, "<absent>") for k in ("status", "done", "finish_reason", "error")}
            assert got == {"status": 200, "done": True, "finish_reason": "stop", "error": None}, (
                f"P11: results row ({arm}) {cid}: {got}"
            )
    assert report["reading"].get("complete", "<absent>") is True, "P11: a clean run is complete"


def test_p11_an_oserror_mid_stream_marks_the_reading_incomplete(repo: Path, tmp_path: Path) -> None:
    """P11: one stream is reset after its 200. The run exits non-zero, the reading says ``complete: false``, every
    decision reads "re-run (human)", and ``report.md`` shows the count."""
    ran = _run_full(
        repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X], hook=_reset_first_be, state={}
    )
    assert ran.rc != 0
    verdict = ran.report["reading"]
    assert verdict.get("complete", "<absent>") is False, f"P11: complete must be false: {verdict}"
    for key in ("reading", "stop_id", "tools_in_prefill", "continuation_content", "seed_tokenizer"):
        assert "re-run (human)" in str(verdict.get(key)), (
            f"P11: decision {key} = {verdict.get(key)!r}"
        )
    md = (ran.out / "report.md").read_text(encoding="utf-8")
    assert re.search(r"transport[ _]errors?\W{0,6}1\b", md, re.IGNORECASE), (
        "P11: report.md does not show the transport error count"
    )


# ----------------------------------------------------------------------------- P12: re-derivable report


def _stored_chunks(report: dict[str, Any], arm: str, cid: str) -> Any:
    """The per-chunk content strings: ``report["chunk_contents"][arm][cid]`` or the row's ``chunk_contents``."""
    top = (report.get("chunk_contents") or {}).get(arm, {}).get(cid)
    return top if top is not None else report["arms"][arm][cid].get("chunk_contents")


def test_p12_q3_q4_and_q6_rederive_from_report_json_alone(repo: Path, tmp_path: Path) -> None:
    """P12: ``report.json`` stores each arm's own ids for (b)-(e) (``report["ids"][arm][cid]``) and each stream
    arm's per-chunk content strings. From those alone, q3, q4 and q6 of (b) and (c) are recomputed with the
    hand-pinned comparison functions and must match the stored values. (b) serves the split capture (q6 = 1),
    (c) content that disagrees with its ids (q4 False), so the values are not all trivial."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X_IDS_NQ_TEXT, d=X, be=[SPLIT])
    tool = _tool_cases(report)
    for arm in ("b", "c", "d", "e"):
        stored = (report.get("ids") or {}).get(arm) or {}
        assert sorted(stored) == sorted(report["arms"]["b"]), (
            f"P12: report.json lacks arm ({arm})'s own ids"
        )
    for arm in ("b", "c"):
        for cid, row in report["arms"][arm].items():
            ids = report["ids"][arm][cid]
            assert ids == _ids_of(X), f"P12: stored ids of ({arm}) {cid} are not the served ones"
            chunks = _stored_chunks(report, arm, cid)
            assert isinstance(chunks, list), f"P12: no per-chunk content strings for ({arm}) {cid}"
            assert "".join(chunks) == row["content"]
            q3 = attr("reasoning_separated")(row["reasoning"], row["content"], ids)
            q4 = attr("content_identical")(row["content"], ids)
            q6 = attr("split_special_pieces")([c for c in chunks if c])
            assert (q3, q4, q6) == (row["q3"], row["q4"], row["split_specials"]), (arm, cid)
    assert {report["arms"]["b"][c]["split_specials"] for c in tool} == {1}
    assert {report["arms"]["c"][c]["q4"] for c in tool} == {False}


# ----------------------------------------------------------------------------- P13: is B a real fallback?


def test_p13_streamed_ids_true_when_one_stream_arm_has_q7_on_all_cases() -> None:
    res = results()
    _set(res, "b", "click-01", q7=False)
    _set(res, "c", "type-01", q7=False)
    out = attr("reading")(res)
    assert out.get("streamed_ids", "<absent>") is True, (
        f"P13: (e) has q7 on all 12: {out.get('streamed_ids', '<absent>')}"
    )


def test_p13_streamed_ids_false_when_every_stream_arm_misses_a_case() -> None:
    res = results()
    for arm, cid in (("b", "click-01"), ("c", "type-01"), ("e", "none-02")):
        _set(res, arm, cid, q7=False)
    out = attr("reading")(res)
    assert out.get("streamed_ids", "<absent>") is False, (
        f"P13: streamed_ids: {out.get('streamed_ids', '<absent>')}"
    )
    assert out["reading"] == "A", "P13: q7 does not decide reading A"
    assert "escalate" not in json.dumps(_decisions(out)).lower(), (
        "P13: escalate only under reading B"
    )


def test_p13_reading_b_without_streamed_ids_escalates_to_the_human() -> None:
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "key-01", q4=False)
    for arm in ("b", "c", "e"):
        _set(res, arm, "none-01", q7=False)
    out = attr("reading")(res)
    assert out["reading"] == "B"
    assert out.get("streamed_ids", "<absent>") is False, (
        f"P13: no stream arm has q7 on all 12: {out.get('streamed_ids', '<absent>')}"
    )
    assert "escalate" in json.dumps(_decisions(out)).lower(), (
        f"P13: B without streamed ids must escalate: {out}"
    )


def test_p13_reading_b_with_streamed_ids_does_not_escalate() -> None:
    res = results()
    for arm in ("b", "c"):
        _set(res, arm, "key-01", q4=False)
    out = attr("reading")(res)
    assert out["reading"] == "B", out["reading"]
    assert out.get("streamed_ids", "<absent>") is True, (
        f"P13: every stream arm has q7 on all 12: {out.get('streamed_ids', '<absent>')}"
    )
    assert "escalate" not in json.dumps(_decisions(out)).lower()


def test_p13_a_run_without_streamed_ids_reports_escalate(repo: Path, tmp_path: Path) -> None:
    """P13 through ``main``: (b), (c) and (e) carry no per-chunk ids, so q7 fails, the reading is B, and the
    report says escalate."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=NOIDS, d=X, be=[NOIDS])
    verdict = report["reading"]
    assert verdict.get("streamed_ids", "<absent>") is False, f"P13: {verdict}"
    assert "escalate" in json.dumps(_decisions(verdict)).lower(), f"P13: {verdict}"


# ----------------------------------------------------------------------------- P14: redaction of responses


def test_p14_a_400_echoing_the_image_data_url_writes_no_base64(repo: Path, tmp_path: Path) -> None:
    """P14: every chat request is a loopback 400 whose JSON echoes ``input_value='data:image/jpeg;base64,...'``.
    No written file holds base64; the 400s are still recorded."""
    ran = _run_full(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X], hook=_echo_400)
    blob = ran.files_text()
    assert '"status": 400' in blob, "the 400 responses were not recorded"
    assert "base64" not in blob and "data:image" not in blob, (
        "P14: a response echoed base64 into a file"
    )
    for p in (SET_DIR / "screens").glob("*.jpg"):
        assert base64.b64encode(p.read_bytes()).decode()[:64] not in blob, p.name


# ----------------------------------------------------------------------------- P15: loopback edges


def test_p15_localhost_is_refused_before_any_socket(
    engine: str, repo: Path, tmp_path: Path, sockets: dict[str, list[Any]]
) -> None:
    """P15: only an IP literal in 127.0.0.0/8 or ``::1``; ``localhost`` is a name and is never resolved."""
    url = engine.replace("127.0.0.1", "localhost")
    rc = _main(["--base-url", url, "--repo", str(repo), "--out-dir", str(tmp_path / "out")])
    assert rc != 0, "P15: localhost must be refused"
    assert sockets["connect"] == [] and _Engine.calls == [], _Engine.calls
    assert "localhost" not in sockets["getaddrinfo"], "P15: localhost was resolved"


def test_p15_a_redirect_is_recorded_and_not_followed(repo: Path, tmp_path: Path) -> None:
    """P15: every chat request answers 302 to another loopback server. The probe records 302 and never reaches the
    target; nothing qualifies."""
    target = type("_Target", (_Engine,), {"calls": [], "queries": 0})
    tsrv = ThreadingHTTPServer(("127.0.0.1", 0), target)
    tt = threading.Thread(target=tsrv.serve_forever, daemon=True)
    tt.start()
    where = f"http://127.0.0.1:{tsrv.server_address[1]}/v1/chat/completions"

    def redirect(h: _Scripted, body: dict[str, Any]) -> bool:
        h.send_response(302)
        h.send_header("Location", where)
        h.send_header("Content-Length", "0")
        h.end_headers()
        return True

    try:
        ran = _run_full(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X], hook=redirect)
    finally:
        tsrv.shutdown()
        tsrv.server_close()
        tt.join(timeout=5)
    assert target.calls == [], f"P15: the redirect was followed: {target.calls[:3]}"
    report = ran.report
    statuses = {r.get("status") for arm in "bc" for r in report["arms"][arm].values()}
    assert statuses == {302}, f"P15: the 3xx is recorded as the status: {statuses}"
    assert _verdict(report)[0] != "A"


# =========================================================================== re-review rows (rulings P17, P18, q8 pairing)

#: sample X without ``[DONE]``: the final chunk (finish_reason "stop") arrives, then the stream just ends
NO_DONE = {**X, "done": False}
Q8_WITH_TOOLS = ["click-01", "type-02", "key-01"]
Q8_CONTROL = ["click-05", "scroll-01", "type-05"]


def _incomplete_ok(verdict: dict[str, Any]) -> None:
    """Ruling P17: under ``complete: false`` no arm, and ``fallback`` / ``streamed_ids`` are re-runs too."""
    assert verdict.get("complete", "<absent>") is False, f"P11: complete must be false: {verdict}"
    assert verdict.get("arm", "<absent>") is None, (
        f"P17: an incomplete run names no arm: {verdict.get('arm')!r}"
    )
    for key in ("fallback", "streamed_ids"):
        assert "re-run" in str(verdict.get(key, "<absent>")), (
            f"P17: {key} = {verdict.get(key, '<absent>')!r}"
        )


@pytest.mark.parametrize("shape", ["A-shaped", "B-shaped"])
def test_p17_an_incomplete_reading_states_nothing(shape: str) -> None:
    """P17 on ``reading``: results that would read A, or B, plus one transport error."""
    res = results(transport_errors=1)
    if shape == "B-shaped":
        for arm in ("b", "c"):
            _set(res, arm, "key-01", q4=False)
    _incomplete_ok(attr("reading")(res))


def _console_reading_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if "reading" in line]


def _md_arm_lines(md: str) -> list[str]:
    """``report.md`` lines that name a chosen arm (``- arm: `b```), not the per-arm table rows."""
    return [
        line for line in md.splitlines() if re.match(r"^\s*[-*]\s*arm\s*:\s*`?[bc]`?\s*$", line)
    ]


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param({"c": X, "be": [X]}, id="A-shaped"),
        pytest.param({"c": NOIDS, "be": [NOIDS]}, id="B-shaped"),
    ],
)
def test_p17_an_oserror_run_names_no_arm_anywhere(
    plan: dict[str, Any], repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P17 through ``main``: one stream is reset after its 200. Whether the rest would read A (on (c)) or B, the
    reading names no arm, ``fallback`` and ``streamed_ids`` read re-run, and neither the final console line nor
    ``report.md`` shows an arm."""
    ran = _run_full(
        repo, tmp_path / "out", a=X_DEFAULT, d=X, hook=_reset_first_be, state={}, **plan
    )
    assert ran.rc != 0
    _incomplete_ok(ran.report["reading"])
    lines = _console_reading_lines(capsys.readouterr().err)
    assert lines, "the probe printed no reading line"
    assert not re.search(r"\barm\W+[bc]\b", lines[-1]), (
        f"P17: the console line names an arm: {lines[-1]!r}"
    )
    md = (ran.out / "report.md").read_text(encoding="utf-8")
    assert _md_arm_lines(md) == [], f"P17: report.md names an arm: {_md_arm_lines(md)}"


def test_p17_a_complete_run_still_names_its_arm(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P17's too-wide twin: with no transport error the arm is named, in the reading, the console and report.md."""
    ran = _run_full(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    assert ran.rc == 0 and _verdict(ran.report) == ("A", "b"), _verdict(ran.report)
    lines = _console_reading_lines(capsys.readouterr().err)
    assert lines and re.search(r"\barm\W+b\b", lines[-1]), lines
    md = (ran.out / "report.md").read_text(encoding="utf-8")
    assert len(_md_arm_lines(md)) == 1, "report.md does not name the chosen arm"


@pytest.mark.parametrize(
    "q8",
    [
        pytest.param([{"hits": None, "queries": None}] * 3, id="all-unreadable"),
        pytest.param(
            [{"hits": 1000, "queries": 1000}] * 2 + [{"hits": None, "queries": 1000}],
            id="one-hits-none",
        ),
        pytest.param(
            [{"hits": 1000, "queries": 1000}] * 2 + [{"hits": 1000, "queries": None}],
            id="one-queries-none",
        ),
    ],
)
def test_p18_unreadable_q8_metrics_are_a_re_run(q8: list[dict[str, Any]]) -> None:
    out = attr("reading")(results(q8=q8))
    assert "re-run" in str(out["tools_in_prefill"]), (
        f"P18: tools_in_prefill = {out['tools_in_prefill']!r}"
    )


def test_p18_a_measured_miss_is_still_false_not_a_re_run() -> None:
    """P18's too-wide twin: a readable /metrics below 0.9 is a decision (False), not a re-run."""
    out = attr("reading")(
        results(q8=[{"hits": 1000, "queries": 1000}] * 2 + [{"hits": 0, "queries": 1000}])
    )
    assert out["tools_in_prefill"] is False, out["tools_in_prefill"]


def test_p18_an_engine_without_metrics_gives_a_re_run(repo: Path, tmp_path: Path) -> None:
    """P18 through ``main``: ``/metrics`` answers 404, so q8 has no numbers and tools_in_prefill is a re-run."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X], metrics=False)
    got = report["reading"]["tools_in_prefill"]
    assert "re-run" in str(got), f"P18: /metrics unreadable, tools_in_prefill = {got!r}"


def _case_of(messages: Any) -> str | None:
    """Which dev case a body's messages belong to, by rebuilding each case's pass-1 messages."""
    rows = [json.loads(x) for x in (SET_DIR / "cases.jsonl").read_text().splitlines() if x.strip()]
    tools = _tools.load()
    for row in rows:
        if (
            row["id"] in Q8_WITH_TOOLS + Q8_CONTROL
            and twopass.native_pass1_body(MODEL, row, SET_DIR, tools)["messages"] == messages
        ):
            return str(row["id"])
    return None


def test_q8_pairing_tools_on_click01_type02_key01_and_control_on_the_other_three(
    repo: Path, tmp_path: Path
) -> None:
    """Re-review (reviewer A's surviving swap): every ``report["q8"]`` row is with tools, on exactly click-01,
    type-02 and key-01; every ``report["q8_control"]`` row is without, on click-05, scroll-01 and type-05; and the
    request bodies agree: the prefills carrying ``tools`` are those three cases' messages, the ones without are the
    control's."""
    ran = _run_full(repo, tmp_path / "out", a=X_DEFAULT, c=X, d=X, be=[X])
    report = ran.report
    assert [r.get("with_tools") for r in report["q8"]] == [True] * 3, report["q8"]
    assert sorted(r.get("case") for r in report["q8"]) == sorted(Q8_WITH_TOOLS), report["q8"]
    assert [r.get("with_tools") for r in report["q8_control"]] == [False] * 3, report["q8_control"]
    assert sorted(r.get("case") for r in report["q8_control"]) == sorted(Q8_CONTROL), report[
        "q8_control"
    ]
    pre = [b for b in ran.chat_bodies() if b.get("max_tokens") == 1]
    with_tools = sorted(str(_case_of(b["messages"])) for b in pre if b.get("tools"))
    without = sorted(str(_case_of(b["messages"])) for b in pre if "tools" not in b)
    assert with_tools == sorted(Q8_WITH_TOOLS), f"the prefills carrying tools are on {with_tools}"
    assert without == sorted(Q8_CONTROL), f"the control prefills (no tools) are on {without}"


def test_a_stream_with_a_finish_reason_but_no_done_is_not_done(repo: Path, tmp_path: Path) -> None:
    """Optional re-review row: (b), (c) and (e) send the final chunk (finish_reason "stop") but no ``[DONE]``; no
    transport error. q2 is not met, so ``results`` says not done and the reading is not A."""
    report = _run(repo, tmp_path / "out", a=X_DEFAULT, c=NO_DONE, d=X, be=[NO_DONE])
    assert report.get("transport_errors") == 0, report.get("transport_errors")
    for arm in ("b", "c"):
        got = {cid: r.get("done") for cid, r in report["results"]["arms"][arm].items()}
        assert set(got.values()) == {False}, f"q2: no [DONE] is not done: ({arm}) {got}"
    assert _verdict(report)[0] != "A", f"a stream without [DONE] read {_verdict(report)}"


# =========================================================================== criterion 8 (negative; green by design)

T87_UNCHANGED = ("local_astra", "plugin", "services")
#: T-0087's merge commit 323c3d9; BASE = ``git merge-base 323c3d9^1 323c3d9^2``, TIP = ``323c3d9^2`` (T-0094 criterion 2)
T0087_MERGE = "323c3d9"
T0087_BASE = "e0174f9"
T0087_TIP = "c08612c"


def _t87_changes(base: str, tip: str, repo: Path = REPO) -> tuple[list[str], list[str]]:
    """Criterion 8's two halves over ``base..tip``: any change under ``T87_UNCHANGED``; and stored logs modified,
    deleted or renamed (``--diff-filter=MDR``; a new log is allowed)."""
    return (
        gr.changed(base, tip, *T87_UNCHANGED, repo=repo),
        gr.changed(base, tip, "logs", diff_filter="MDR", repo=repo),
    )


def test_t0087_changes_nothing_under_local_astra_plugin_services_or_stored_logs() -> None:
    """Criterion 8 over T-0087's closed range (T-0094): no change under ``local_astra``, ``plugin`` or ``services``, and
    no tracked file under ``logs/`` modified, deleted or renamed, between BASE and TIP. The untracked-file half of the
    old guard is retired: it only meant something while the branch was open."""
    gr.prove_range(T0087_BASE, T0087_TIP, T0087_MERGE)
    code, stored = _t87_changes(T0087_BASE, T0087_TIP)
    assert code == [], code
    assert stored == [], stored


def test_the_t0087_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the first half sees ``executor.py`` (T-0089/T-0092)."""
    gr.require_commits(T0087_BASE, gr.LATER)
    code, _stored = _t87_changes(T0087_BASE, gr.LATER)
    assert "local_astra/conductor/executor.py" in code, code


def test_the_t0087_logs_half_reports_a_modified_or_deleted_log_and_allows_a_new_one(
    tmp_path: Path,
) -> None:
    """T-0094 (adversarial, the ``logs`` half): no later commit modifies a stored log, so its power is shown in a
    scratch repo: a modified and a deleted log are reported, a new one is not, and nothing outside ``logs`` is."""
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel in ("logs/a.jsonl", "logs/b.jsonl", "logs/c.jsonl", "local_astra/x.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("base\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    rev = [*git, "rev-parse", "HEAD"]
    base = subprocess.run(rev, capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "logs/a.jsonl").write_text("edited\n")
    (tmp_path / "logs/b.jsonl").unlink()
    (tmp_path / "logs/new.jsonl").write_text("new probe output\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "tip"], check=True)
    tip = subprocess.run(rev, capture_output=True, text=True, check=True).stdout.strip()
    code, stored = _t87_changes(base, tip, repo=tmp_path)
    assert code == []
    assert stored == ["logs/a.jsonl", "logs/b.jsonl"]
