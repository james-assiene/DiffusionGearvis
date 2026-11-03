"""Warm each repair automaton once before the first turn (T-0101, H-T101a).

On a fresh engine boot the first H-T84-2 repair on a given automaton pays the plugin's Triton JIT compile (1.5-2.5 s
measured at a794a98; later repairs 213-346 ms). ``warm_repair(llm, kit)`` sends one repair request per distinct
``automaton_sha256`` of ``kit.payloads`` -- sequentially, in tool-name order, through the caller's real client and the
real CD path -- so the compile lands in session start instead of the first user turn.

- Each request is the session's own repair shape: ``cd`` is ``live.repair_request(name, ids, 50, tag, payloads,
  calibration)`` (only ``draft`` and ``record_tag`` differ from a turn's), options ``live.REPAIR_OPTIONS``, the prompt
  ``live.repair_prompt`` of one fixed minimal pass-1 prompt (no screen, memory or user data) carrying ``kit.tools``.
  The draft is the engine's ids (``llm.tokenize``) of ``CANONICAL_DRAFTS[name]``, a fixed valid native call.
- Invisible: ``xargs`` ``{"astra_tap": "off"}`` keeps it out of the denoising view; request ids start ``warmup-``
  and record tags ``warmup:``. Commits are counted and discarded: nothing goes to TTS, the HUD, history, memory or
  a session log.
- Reported, never gating: a request that raises, ends 4xx/5xx or outlives its deadline (the remaining total budget
  when it starts; ruling W1; aborted) is ``failed: <reason>`` and the next one is still tried; once the budget is
  spent the rest are ``skipped``. A payload with no canonical draft is ``skipped: no canonical draft``. Two tools
  sharing one sha are warmed once and both carry that request's outcome (ruling W2). The output's
  ``live.repair_verdict`` status is reported only; an invalid output is still ``ok``.
- ``warm_repair`` never raises except ``CancelledError`` (a running request is aborted first) and never touches the
  kit.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from local_astra.contracts.llm import Commit, Message, Prompt, TextPart
from local_astra.toolcall import live, native

if TYPE_CHECKING:
    from local_astra.conductor.session import ToolKit
    from local_astra.contracts.protocols import LlmBackend

__all__ = [
    "CANONICAL_DRAFTS",
    "DEFAULT_TOTAL_BUDGET_S",
    "TAP_OFF",
    "WarmEntry",
    "WarmReport",
    "summary",
    "warm_repair",
]

DEFAULT_TOTAL_BUDGET_S: Final = 30.0
#: the tap opt-out (``tap.tap_enabled`` honours ``astra_tap: "off"``): a warm-up never reaches the denoising view
TAP_OFF: Final[Mapping[str, str]] = MappingProxyType({"astra_tap": "off"})
#: pass 1's stop id the seed ends in (``<|tool_response>``, the native call's usual stop)
_STOP_ID: Final = 50
_D: Final = native.STRING_DELIM
_C: Final = native.TOOL_CALL_CLOSE
#: one fixed, valid native call per v1 tool (``live.classify`` says ``valid``; in-range coordinates, an in-list
#: combo), as the router hands a draft to the repair: the text after ``<|tool_call>``
CANONICAL_DRAFTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "click_at": f"call:click_at{{x:500,y:500}}{_C}",
        "google_search": f"call:google_search{{query:{_D}weather{_D}}}{_C}",
        "press_key": f"call:press_key{{combo:{_D}Enter{_D}}}{_C}",
        "scroll": f"call:scroll{{direction:{_D}down{_D}}}{_C}",
        "type_at": f"call:type_at{{x:500,y:500,text:{_D}hello{_D},press_enter:false}}{_C}",
    }
)
_PASS1_TEXT: Final = "Warm-up request: no screen is attached. Call one tool."
_TAG_UNSAFE: Final = re.compile(r"[^A-Za-z0-9_.:-]")
_REASON_MAX: Final = 160


@dataclass(frozen=True)
class WarmEntry:
    """One tool's warm-up: ``ok`` | ``failed: <reason>`` | ``skipped`` | ``skipped: <why>``."""

    status: str
    wall_ms: float
    commits: int
    verdict: str | None


@dataclass(frozen=True)
class WarmReport:
    per_tool: Mapping[str, WarmEntry]
    total_ms: float


def _pass1(kit: ToolKit) -> Prompt:
    """The fixed minimal pass-1 prompt: one user line, the session's tools field, thinking on (a session setting)."""
    return Prompt(
        messages=(Message(role="user", content=(TextPart(text=_PASS1_TEXT),)),),
        prefix_len=0,
        thinking=True,
        tools=kit.tools,
    )


def _tag(name: str) -> str:
    return ("warmup:" + _TAG_UNSAFE.sub("_", name))[:64]


def _reason(exc: BaseException) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return f"HTTP {status}"
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text[:_REASON_MAX]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


async def _abort(llm: LlmBackend, request_id: str) -> None:
    with contextlib.suppress(Exception):
        await llm.abort(request_id)


async def _one(
    llm: LlmBackend, kit: ToolKit, name: str, request_id: str, deadline_s: float
) -> WarmEntry:
    """One warm-up request with ``deadline_s`` to finish (tokenize included)."""
    assert kit.payloads is not None and kit.calibration is not None
    t0 = time.perf_counter()
    commits = 0
    found: Commit | None = None
    try:
        async with asyncio.timeout(deadline_s):
            ids = await llm.tokenize(CANONICAL_DRAFTS[name].rstrip())
            prompt = live.repair_prompt(_pass1(kit), "")
            if isinstance(prompt, live.Refusal):
                return WarmEntry(f"failed: continuation_guard: {prompt.reason}", _ms(t0), 0, None)
            request = live.repair_request(
                name, ids, _STOP_ID, _tag(name), kit.payloads, kit.calibration
            )
            stream = llm.generate(
                prompt, request_id, live.REPAIR_OPTIONS, xargs=TAP_OFF, cd=request
            )
            async with contextlib.aclosing(stream):  # type: ignore[type-var]
                async for commit in stream:
                    commits += 1
                    if commit.segment == "tool" and found is None:
                        found = commit
    except asyncio.CancelledError:
        await _abort(llm, request_id)
        raise
    except TimeoutError:
        await _abort(llm, request_id)
        return WarmEntry(
            f"failed: timeout after {_ms(t0):.0f} ms (aborted)", _ms(t0), commits, None
        )
    except Exception as exc:
        return WarmEntry(f"failed: {_reason(exc)}", _ms(t0), commits, None)
    verdict = live.repair_verdict(found.text).status if found is not None else None
    return WarmEntry("ok", _ms(t0), commits, verdict)


async def warm_repair(
    llm: LlmBackend, kit: ToolKit, *, total_budget_s: float = DEFAULT_TOTAL_BUDGET_S
) -> WarmReport:
    """One warm-up request per distinct payload ``automaton_sha256`` (module docstring)."""
    t0 = time.perf_counter()
    deadline = t0 + total_budget_s
    per: dict[str, WarmEntry] = {}
    payloads = kit.payloads
    if payloads is None or kit.calibration is None:
        return WarmReport(per_tool=MappingProxyType(per), total_ms=_ms(t0))
    groups: dict[str, list[str]] = {}
    for name in sorted(payloads.tools):
        groups.setdefault(payloads.tools[name].automaton_sha256, []).append(name)
    n = 0
    for names in groups.values():
        chosen = next((nm for nm in names if nm in CANONICAL_DRAFTS), None)
        if chosen is None:
            for nm in names:
                per[nm] = WarmEntry("skipped: no canonical draft", 0.0, 0, None)
            continue
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            entry = WarmEntry("skipped", 0.0, 0, None)
        else:
            n += 1
            entry = await _one(llm, kit, chosen, f"warmup-{n}-{chosen}", remaining)
        for nm in names:
            per[nm] = entry
    ordered = {name: per[name] for name in sorted(per)}
    return WarmReport(per_tool=MappingProxyType(ordered), total_ms=_ms(t0))


def summary(report: WarmReport) -> str:
    """``<tool> <ms> ms <status>, ..., total <ms> ms`` in tool-name order (the session's stdout line)."""
    parts = [f"{t} {e.wall_ms:.0f} ms {e.status}" for t, e in sorted(report.per_tool.items())]
    return ", ".join([*parts, f"total {report.total_ms:.0f} ms"])
