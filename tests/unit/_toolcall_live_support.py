"""Shared helpers for the T-0091 rows (payloads, classifier, repair builder, executor contracts, dry-run executor).

Nothing here imports T-0091's new modules at import time. ``live()``, ``contracts()``, ``executor_mod()`` and
``protocol()`` import them inside a row and turn "it does not exist yet" into an ``AssertionError`` naming the
criterion, so each row is red on its own named assertion and collection never errors (the T-0088 lazy-import
pattern, ``tests/unit/_llm_id_support.py``). Stdlib + pydantic only: no cd package (``test_cd_isolation``).

API the rows assume (docs/tasks/T-0091.md and its 2026-10-12 amendment; where the task is silent, the choice is
stated here and in the test-author's report):

- ``local_astra.toolcall.live``:
  - ``PayloadStale`` (an ``Exception``); ``load_payloads(path=None)`` reads ``toolcall/payloads/native_T4.json`` by
    default and returns an object the rows only pass on to ``repair_request``;
  - ``load_calibration(path=None)`` reads ``docs/cd/calibration-bfcl.json`` by default, checks its sha
    (``4084026c…``) and returns an object the rows only pass on to ``repair_request`` (the task names a
    ``calibration`` argument but not where it comes from);
  - ``classify(tool_text) -> Verdict`` and ``repair_verdict(text) -> Verdict``; a ``Verdict`` has the attributes
    ``status``, ``name``, ``arguments``, ``reason`` and ``query_problem``;
  - ``repair_prompt(pass1, spoken_prefix) -> Prompt | Refusal``; a ``Refusal`` has ``reason`` in ``{"template",
    "marker"}`` (``"continuation_guard: template"`` / ``"…: marker"`` is accepted too: the task spells it that way);
  - ``repair_request(name, draft_ids, stop_id, record_tag, payloads, calibration) -> CdRequest`` (positional, in that
    order);
  - ``REPAIR_OPTIONS == LlmOptions(max_tokens=256)`` (the task gives the value, not the name).
- ``local_astra.contracts.toolcall``: ``FrameGeometry``, ``ToolInvocation``, ``ToolOutcome`` with the fields of
  criterion 7, built by keyword. ``ToolOutcome.resolved`` holds ``x_px``/``y_px`` for a click or a type (the names of
  criterion 7's formula).
- ``local_astra.contracts.protocols.ToolExecutor`` (``runtime_checkable``): ``mode`` and ``async execute(inv)``.
- ``local_astra.conductor.tool_executor.DryRunToolExecutor()``: no constructor argument; ``mode == "dry_run"`` (the
  value of the T-0080d switch it serves).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
import struct
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PAYLOADS = REPO / "local_astra" / "toolcall" / "payloads" / "native_T4.json"
CALIBRATION = REPO / "docs" / "cd" / "calibration-bfcl.json"
CALIBRATION_SHA256 = "4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a"
QARGMAX_CALIBRATION = REPO / "docs" / "cd" / "calibration-bfcl-perstep_mar_qargmax.json"
EVAL = REPO / "tests" / "fixtures" / "toolcall" / "native_t4_eval.json"
#: T-0091's base: main with T-0088 merged. A literal SHA, never the moving ``main`` (tests.md).
BASE_SHA = "bd2c124"

V1 = ("google_search", "click_at", "type_at", "scroll", "press_key")
OPEN, CLOSE, Q = "<|tool_call>", "<tool_call|>", '<|"|>'
STOPS = (1, 106, 50)


def _import(modname: str, why: str) -> ModuleType:
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        if exc.name == modname:
            raise AssertionError(f"T-0091 {why}: `{modname}` does not exist yet ({exc})") from exc
        raise


def need(obj: Any, name: str, why: str) -> Any:
    if not hasattr(obj, name):
        owner = getattr(obj, "__name__", None) or type(obj).__name__
        raise AssertionError(f"T-0091 {why}: `{owner}.{name}` does not exist yet")
    return getattr(obj, name)


def live() -> ModuleType:
    return _import("local_astra.toolcall.live", "criteria 1-6")


def contracts() -> ModuleType:
    return _import("local_astra.contracts.toolcall", "criterion 7 (contracts)")


def executor_mod() -> ModuleType:
    return _import("local_astra.conductor.tool_executor", "criterion 7 (DryRunToolExecutor)")


def protocol() -> Any:
    from local_astra.contracts import protocols

    return need(protocols, "ToolExecutor", "criterion 7 (protocol)")


def fn(name: str, why: str) -> Any:
    return need(live(), name, why)


# --------------------------------------------------------------------------- criterion 1: the payload file


def payload_doc(path: Path = PAYLOADS) -> Any:
    assert path.is_file(), (
        f"T-0091 criterion 1: {path.relative_to(REPO) if path.is_relative_to(REPO) else path} does not exist "
        "(scripts/build_toolcall_payloads.py writes it)"
    )
    return json.loads(path.read_bytes())


def walk(node: Any, trail: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield trail, node
    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from walk(v, (*trail, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, (*trail, i))


def tool_entries(doc: Any) -> dict[str, dict[str, Any]]:
    """name -> the per-tool object (the one holding ``automaton``), wherever the layout puts it: its name is its
    ``name`` field or the key it sits under."""
    out: dict[str, dict[str, Any]] = {}
    for trail, node in walk(doc):
        if isinstance(node, Mapping) and "automaton" in node:
            name = (
                node.get("name")
                if isinstance(node.get("name"), str)
                else (trail[-1] if trail else None)
            )
            assert isinstance(name, str), f"criterion 1: a payload at {trail} names no tool"
            assert name not in out, f"criterion 1: two payloads for {name!r}"
            out[name] = dict(node)
    return out


def scalars(doc: Any) -> list[Any]:
    return [v for _t, v in walk(doc) if not isinstance(v, (Mapping, list))]


def eval_fixture() -> Any:
    return json.loads(EVAL.read_bytes())


@cache
def payloads() -> Any:
    return fn("load_payloads", "criterion 1 (load check)")()


@cache
def calibration() -> Any:
    return fn("load_calibration", "criterion 5 (calibration, sha 4084026c… checked)")()


# --------------------------------------------------------------------------- criterion 5: the CD request


def draft_ids_of(req: Any) -> list[int]:
    """The ids of a ``CdRequest.draft`` (the documented wire: standard base64 of little-endian int32 ids)."""
    raw = base64.b64decode(req.draft.encode("ascii"), validate=True)
    return list(struct.unpack(f"<{len(raw) // 4}i", raw))


def sha256_ascii(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


# --------------------------------------------------------------------------- criterion 7: contracts


def geometry(width: int, height: int, *, phash: str = "ph0", source: str = "screen") -> Any:
    from local_astra.contracts.perception import Source

    return need(contracts(), "FrameGeometry", "criterion 7 (contracts)")(
        phash=phash, source=Source(source), width=width, height=height
    )


def invocation(
    name: str,
    arguments: Mapping[str, Any],
    *,
    frame: Any = None,
    call_id: str = "c1",
    turn_id: str = "t1",
    source: str = "draft",
) -> Any:
    return need(contracts(), "ToolInvocation", "criterion 7 (contracts)")(
        call_id=call_id,
        turn_id=turn_id,
        name=name,
        arguments=dict(arguments),
        source=source,
        frame=frame,
    )


def dry_run() -> Any:
    return need(executor_mod(), "DryRunToolExecutor", "criterion 7 (DryRunToolExecutor)")()


def run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def native(name: str, args: Sequence[tuple[str, str]]) -> str:
    """A native call ``<|tool_call>call:NAME{k:v,...}<tool_call|>`` from already-rendered values."""
    return OPEN + "call:" + name + "{" + ",".join(f"{k}:{v}" for k, v in args) + "}" + CLOSE


def s(text: str) -> str:
    """A native string value."""
    return Q + text + Q
