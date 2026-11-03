"""T-0099: the live repair harness's report, on hand-built run directories (unit; no engine, no socket).

Every row builds a run directory by hand -- the drafts, the repairs, the spans and the outcomes are chosen here, and
every expected number is computed by hand in the row (``Fraction``; the Wilson interval by hand in a comment). The
harness does not exist yet, so the rows are red on the named assertion in ``harness()``, never on an ImportError at
collection (the module is loaded lazily, by path, as ``test_t0093_report.py`` loads T-0093's).

**The assumed surface** (``scripts/gpu/t0099_repair_live.py``; importing it opens no socket and starts nothing):

- ``build_report(run_dir: Path) -> dict`` reads only ``turns.jsonl``, ``sessions/*.jsonl`` and ``manifest.json``
  (criterion 9). An *invalid draft* is pass 1's kept ``tool`` commit (an ``llm.commit`` record of segment ``tool``
  from the turn's generate request) judged ``repairable`` by the **unpatched** ``local_astra.toolcall.live.classify``,
  re-run over the logged commit text -- never read from a ``tool.call`` record (criterion notation).
- ``gates(report) -> {"S": {"G1": bool, ..., "G8": bool}, "F": {...}}``: criterion 5, per arm; ``True`` holds.
- ``render_markdown(report) -> str``: report.md's text.
- ``forced_classify()``: a context manager; inside it ``live.classify`` is criterion 3's wrapper (a ``valid`` verdict
  becomes ``Verdict("repairable", name, arguments, FORCED_REASON, query_problem)``, anything else is the real
  verdict); on exit -- an exception included -- ``live.classify`` is the original object again.
- ``FORCED_REASON == "forced: valid draft (T-0099 arm F)"``, ``K == 3``, ``STRESS_SHA256`` (the pinned sha of
  ``tests/eval/toolcall_stress/cases.jsonl``), ``STRESS_PATH``; ``load_stress_cases(path=None)`` (the 30 cases in file
  order; ``ValueError`` when the file's sha256 is not ``STRESS_SHA256``, both read at call time) and
  ``load_forced_cases()`` (arm F: the 40 dev tool cases of ``tests/eval/toolcall/cases.jsonl`` in file order). Cases
  expose ``case_id``.

**The run directory** (criterion 9; the fixture below writes it):

- ``turns.jsonl``: the sessions' own ``TurnRecord`` lines; ``sessions/<log>``: one ``SessionLog`` per session (one
  session per (case, repetition, arm), one typed turn ``t1`` each; pass 1 is ``t1-gen``, the repair ``t1-repair``,
  the call ``t1-call``, as the turn machine names them).
- ``manifest.json``: ``{"regime": {...}, "sessions": [{"arm", "session_id", "log", "forced"}], "cases": [{"case_id",
  "category", "arm", "rep", "order", "session_id", "session_log", "turn_id", "gt", "forced", "barge_in",
  "barge_in_attempted", "timed_out", "failures"}]}``; ``arm`` is ``"S"`` or ``"F"``; ``category`` is the stress
  category in arm S (``K-in``, ``K-out``, ``S-extra``, ``T-hard``, ``C-pixel``, ``U-under``) and the dev case's tool
  in arm F; ``gt`` is the case's ground truth (``None`` for K-out).
- The forced marker (criterion 3) reaches the log where the session puts it: the ``reason`` of the ``ToolDraft`` on
  pass 1's ``LlmDone`` event (a forced draft's ``tool.call`` records say ``schema_valid: false``).

**The report keys these rows read** (rates ``{"k", "n"}``, distributions ``{"n", "p50", "p90"}``, nearest-rank):

- ``void``: reasons, empty for a valid run; criterion 8's first reason contains
  ``"integrity: repair on a valid draft outside arm F"``, its second names the forced marker (contains ``"forced"``).
- ``regime``: the manifest's regime plus ``prompt_sha256`` = sha256 of the UTF-8 text of the system message (its text
  parts in order) of the first ``llm.request`` record of the **first** session in manifest order; ``stress_sha256``,
  ``tools_sha256``, ``k``, ``arm_sizes`` (``{"S": int, "F": int}``) and ``commit`` carried through.
- ``report["S"]``: ``yield`` (k invalid drafts / n routed tool turns), ``yield_by_category`` ({category: rate}),
  ``yield_by_reason`` ({unpatched ``classify`` reason: count}), ``refused_drafts`` (drafts ``classify`` refuses:
  nameless and the like), ``unrouted`` (turns with no pass-1 tool commit); ``repairs`` (``{"n", "outcomes": {final
  status: count}, "ms": dist of the tool.repair span}``; the final status is the last ``tool.call`` record's
  ``status``, J6); ``rescue`` (``{"k", "n", "wilson95": [lo, hi] | None, "reading"}``: k = repairs whose verdict is
  valid, K-out excluded (that is coercion, never rescue); n = every arm-S repair fired); ``rescue_exact`` (``{"k",
  "n", "reading"}``: k = repaired-valid and exact, U-under by criterion 1's rule; n = repairs on gt-bearing
  categories); ``coercion`` (``{"n", "cases": [{"case_id", "rep", "combo"}]}``).
  ``reading`` is ``"read"`` with >= 10 repairs on gt-bearing arm-S turns, else starts ``"insufficient yield (n = <that
  count>)"`` and ``wilson95`` is ``None``.
- ``report["F"]``: ``repairs`` (as above), ``agreement`` (repaired call equal to the draft / repairs fired),
  ``repair_timeout`` (count), ``barge_in`` (``{"attempted": int, "delivered": int, "repair_done_before_takeover": int,
  "reading": "read" | "unread..."}``: attempted = the manifest's ``barge_in_attempted``; delivered = ``outcome ==
  "interrupted"``, the line carries a call, and the log shows ``LlmAbort`` of the turn's repair request;
  repair_done_before_takeover = the repair's ``llm.done`` (with its verdict) accepted by a step whose ``state_before``
  is ``PAUSED``, ruling Q8). ``BARGE_IN_MAX_ATTEMPTS == 5`` (Q8).

Regime: CPU, pure, hand-built data; no measurement. The ``gpu`` rows that read a real run are
``tests/eval/test_t0099_repair_live.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import ToolDraft
from local_astra.telemetry.turns_log import Regime, TurnRecord
from local_astra.toolcall import live, tools

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0099_repair_live.py"
T0093 = REPO / "scripts" / "gpu" / "t0093_toolcall_live.py"
STRESS = REPO / "tests" / "eval" / "toolcall_stress" / "cases.jsonl"
V1_CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"

FORCED = "forced: valid draft (T-0099 arm F)"
TOOLS_ON = "native:bbfc53ccdd80:dry_run"
GIT = "2c584d9" + "0" * 33
WALL = "2026-10-14T18:00:00+01:00"
SYSTEM_TEXT = "You are Astra, a local assistant that can see the user's screen."
GATE_KEYS = frozenset(f"G{i}" for i in range(1, 9))

#: the T-0093 helpers the implementation notes name for reuse (a rename there must turn the fast suite red)
T0093_HELPERS = (
    "_MeteredLlm",
    "_PlaybackOnly",
    "_CaseMemory",
    "_fake",
    "_wait_idle",
    "check_loopback",
    "read_metrics",
    "_NoRedirect",
    "_OPENER",
    "_Turn",
    "_voiced",
    "_executes_before_drain",
    "_stale",
    "_percentile",
    "_dist",
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def harness() -> ModuleType:
    """The harness, loaded by path. Red here, by name, until T-0099 lands."""
    assert HARNESS.is_file(), (
        f"T-0099: {HARNESS.relative_to(REPO)} does not exist yet (the report is built by it)"
    )
    module = _load(HARNESS, "_t0099_repair_live_unit")
    for attr in ("build_report", "gates", "render_markdown", "forced_classify", "main"):
        assert callable(getattr(module, attr, None)), (
            f"T-0099 report surface: {HARNESS.name} has no callable {attr!r} (see this module's docstring)"
        )
    return module


# =====================================================================================================
# The hand-built run
# =====================================================================================================


def native(name: str | None, arguments: dict[str, Any], *, close: bool = True) -> str:
    """A native call as pass 1's ``tool`` commit carries it (``<|"|>``-quoted strings)."""

    def value(v: Any) -> str:
        return f'<|"|>{v}<|"|>' if isinstance(v, str) else json.dumps(v)

    body = ",".join(f"{k}:{value(v)}" for k, v in arguments.items())
    return f"<|tool_call>call:{name or ''}{{{body}}}" + ("<tool_call|>" if close else "")


#: drafts used below, each checked against the unpatched judge in ``test_fixture_drafts_judge_as_named``
D_F5 = native(
    "press_key", {"combo": "F5"}
)  # refused, "combo not in enum" (T-0100 H-T99d; repairable until 0756893)
#: an alias of a list entry (Control -> Ctrl, T-0100 Q1): repairable "schema invalid" before and after T-0100. The
#: repair rows below draft it where they drafted F5 until 0756893 (an out-of-list combo is no longer repaired).
D_KEY_ALIAS = native("press_key", {"combo": "Control+C"})  # repairable, "schema invalid"
D_ENTER = native("press_key", {"combo": "Enter"})  # valid
D_UNCLOSED = (
    '<|tool_call>call:type_at{x:500,y:36,text:<|"|>usb-c'  # repairable, "unterminated string"
)
D_NAMELESS = native(None, {"combo": "F5"})  # refused
D_CLICK = native("click_at", {"x": 800, "y": 380})  # valid
D_CLICK_PX = native(
    "click_at", {"x": 1024, "y": 304}
)  # refused, "coordinate out of range" (T-0100 H-T99e)
D_SCROLL_AMOUNT = native(
    "scroll", {"direction": "down", "amount": 3}
)  # repairable, "schema invalid"
D_TYPE_NO_ENTER = native("type_at", {"x": 500, "y": 36, "text": "wireless mouse"})  # repairable

GT_SCROLL_DOWN = {"name": "scroll", "direction": "down"}
GT_STORE_SEARCH_U = {
    "name": "type_at",
    "bbox": [297, 13, 703, 59],
    "text": "wireless mouse",
    "press_enter": None,
}
GT_STORE_SEARCH_T = {
    "name": "type_at",
    "bbox": [297, 13, 703, 59],
    "text": "{hdmi,usb-c} cable",
    "press_enter": True,
}
GT_CLICK = {"name": "click_at", "bbox": [740, 358, 839, 402]}


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


@dataclass(frozen=True)
class Spec:
    """One session: one typed turn ``t1``. Every field's default is the healthy value."""

    case: str
    arm: str = "S"
    category: str | None = "T-hard"
    gt: dict[str, Any] | None = None
    rep: int = 0
    #: pass 1's ``tool`` commit text; ``None``: the turn is not routed (no tool commit, no call)
    draft: str | None = None
    #: ``schema_valid`` on the ``proposed``/``repairing`` records; ``None``: what the session writes
    record_valid: bool | None = None
    #: arm F's forcing: the LlmDone draft reads ``repairable`` with the forced marker
    forced: bool = False
    #: ``None`` (no repair), ``valid``/``invalid`` (a verdict), ``timeout``, ``open`` (no verdict, no abort), ``barge``
    repair: str | None = None
    #: the repair's call for ``valid``/``invalid``
    repair_call: dict[str, Any] | None = None
    repair_ms: int = 400
    #: ``tool_repair`` ``llm.request`` records written (G6)
    repair_requests: int = 1
    done_ms: int = 1500
    speech: str = "Okay, on it. "
    #: ``barge``: whether the takeover step carries ``LlmAbort`` of the repair; ``timeout``: whether the tick does
    abort: bool = True
    #: G2: the repair's own ``answer`` commit is stepped through the machine and voiced
    repair_voiced: bool = False
    #: G3: a ``ToolExecute`` in the takeover step of a barged turn
    execute_on_barge: bool = False
    #: G7: a ``ToolExecute`` for the call after its repair failed
    execute_dead: bool = False
    #: G7: the executed arguments, over the repair's
    execute_args: dict[str, Any] | None = None
    #: criterion 8b: the forced marker on a ``tool.call`` record of this session
    marker_on_record: bool = False
    #: manifest fields over the healthy ones
    manifest: tuple[tuple[str, Any], ...] = ()
    system_text: str = SYSTEM_TEXT


def _step(t: int, event: Any, before: str, after: str, commands: list[Any]) -> dict[str, Any]:
    return {
        "_t": t,
        "kind": "machine.step",
        "data": {
            "event": event.model_dump(mode="json"),
            "state_before": before,
            "state_after": after,
            "turn_seq_before": 1,
            "turn_seq_after": 1,
            "commands": [c.model_dump(mode="json") for c in commands],
        },
    }


def _rec(t: int, kind: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"_t": t, "kind": kind, "data": data}


def _tool_call(
    status: str,
    source: str,
    c: dict[str, Any],
    *,
    schema_valid: bool | None,
    repair_ms: int | None = None,
    reason: str | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """The ``tool.call`` record, keys as ``Session._publish_tool`` writes them."""
    return {
        "status": status,
        "call_id": "t1-call",
        "source": source,
        "name": c.get("name") or "",
        "arguments": dict(c.get("arguments") or {}),
        "parse": "ok",
        "schema_valid": schema_valid,
        "query_problem": None,
        "repair_ms": repair_ms,
        "action": f"recorded {c.get('name')}" if status == "dry_run" else None,
        "reason": reason,
        "resolved": {} if status == "dry_run" else None,
        "spoken_prefix": prefix,
    }


def _request(rid: str, phase: str, system_text: str, user_text: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "request_id": rid,
        "attempt_id": f"e-{rid}",
        "options": {"top_k": None, "top_p": None, "max_tokens": None, "stop": []},
        "xargs": None,
        "prompt": {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ],
            "prefix_len": 2,
            "thinking": True,
            "tools": None,
            "continue_final_message": phase == "tool_repair",
        },
        "drop_history_turns": 0,
        "prompt_tokens": 0,
    }


def turn_records(sp: Spec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The session log of one turn and its ``TurnRecord`` kwargs (``spans``, ``outcome``, ``tool``)."""
    tid, gen, rep = "t1", "t1-gen", "t1-repair"
    s = 1000
    t_fc, t_done = s + 800, s + sp.done_ms
    spans: dict[str, tuple[int, int]] = {"llm.first_canvas": (s, t_fc), "llm.done": (s, t_done)}
    out: list[dict[str, Any]] = [
        _rec(s, "llm.request", _request(gen, "generate", sp.system_text, sp.case)),
        _step(
            s,
            E.UserText(t_ms=s, text=sp.case),
            "IDLE",
            "THINKING",
            [
                C.LlmStart(request_id=gen, transcript=sp.case),
                C.HudState(state="THINKING", turn_id=tid),
            ],
        ),
        _rec(
            t_fc,
            "llm.commit",
            {"request_id": gen, "canvas_idx": 0, "segment": "answer", "text": sp.speech},
        ),
        _step(
            t_fc,
            E.LlmCommit(t_ms=t_fc, request_id=gen, canvas_idx=0, segment="answer", text=sp.speech),
            "THINKING",
            "SPEAKING",
            [C.TtsSpeak(utterance_id="t1-u1", text=sp.speech, source=sp.speech)],
        ),
    ]
    final: dict[str, Any] | None = None
    outcome = "spoken"
    executed: tuple[dict[str, Any], str] | None = None
    t_drain = t_done + 300
    if sp.draft is None:
        out.append(_rec(t_done, "llm.done", {"request_id": gen, "commits": 1}))
        out.append(
            _step(t_done, E.LlmDone(t_ms=t_done, request_id=gen), "SPEAKING", "SPEAKING", [])
        )
    else:
        verdict = live.classify(sp.draft)
        out.append(
            _rec(
                t_done,
                "llm.commit",
                {
                    "request_id": gen,
                    "canvas_idx": 1,
                    "segment": "tool",
                    "text": sp.draft,
                    "stop_id": 50,
                },
            )
        )
        out.append(_rec(t_done, "llm.done", {"request_id": gen, "commits": 2}))
        if sp.repair is not None:
            status, reason = (
                "repairable",
                FORCED if sp.forced else (verdict.reason or "schema invalid"),
            )
        else:
            status, reason = verdict.status, verdict.reason
        draft = ToolDraft(
            call_id="t1-call",
            status=status,
            name=verdict.name if status != "refused" else None,
            arguments=verdict.arguments if status != "refused" else None,
            reason=reason,
            source="draft",
        )
        cmds: list[Any] = [C.CancelTimer(name="llm_timeout")]
        if sp.repair is not None:
            cmds.append(C.ToolRepairStart(request_id=rep, draft_request_id=gen, call_id="t1-call"))
        elif status == "refused":
            cmds.append(C.ToolCancel(call_id="t1-call", reason="refused"))
        out.append(
            _step(
                t_done,
                E.LlmDone(t_ms=t_done, request_id=gen, tool=draft),
                "SPEAKING",
                "SPEAKING",
                cmds,
            )
        )
        pcall = {"name": verdict.name, "arguments": verdict.arguments or {}}
        rv = sp.record_valid
        if rv is None:
            rv = (
                (verdict.status == "valid" and not sp.forced)
                if verdict.arguments is not None
                else None
            )
        first_status = "refused" if status == "refused" else "proposed"
        out.append(
            _rec(
                t_done,
                "tool.call",
                _tool_call(
                    first_status,
                    "draft",
                    pcall,
                    schema_valid=rv,
                    reason=(FORCED if sp.marker_on_record else reason)
                    if first_status == "refused"
                    else None,
                    prefix=sp.speech,
                ),
            )
        )
        if status == "refused":
            final = out[-1]["data"]
        if sp.marker_on_record and first_status != "refused":
            out.append(
                _rec(
                    t_done,
                    "tool.call",
                    _tool_call(
                        "proposed", "draft", pcall, schema_valid=rv, reason=FORCED, prefix=sp.speech
                    ),
                )
            )
        if sp.repair is None and status == "valid":
            executed = (pcall, "draft")
        if sp.repair is not None:
            out.append(
                _rec(
                    t_done,
                    "tool.call",
                    _tool_call("repairing", "draft", pcall, schema_valid=rv, prefix=sp.speech),
                )
            )
            for _ in range(sp.repair_requests):
                out.append(
                    _rec(
                        t_done, "llm.request", _request(rep, "tool_repair", sp.system_text, sp.case)
                    )
                )
            t_rep = t_done + sp.repair_ms
            if sp.repair in ("valid", "invalid"):
                rc = sp.repair_call
                assert rc is not None, f"fixture: {sp.case} needs a repair_call"
                if sp.repair_voiced:
                    said = "Pressing it now. "
                    out.append(
                        _rec(
                            t_rep - 10,
                            "llm.commit",
                            {"request_id": rep, "canvas_idx": 0, "segment": "answer", "text": said},
                        )
                    )
                    out.append(
                        _step(
                            t_rep - 10,
                            E.LlmCommit(
                                t_ms=t_rep - 10,
                                request_id=rep,
                                canvas_idx=0,
                                segment="answer",
                                text=said,
                            ),
                            "SPEAKING",
                            "SPEAKING",
                            [C.TtsSpeak(utterance_id="t1-u2", text=said, source=said)],
                        )
                    )
                rtext = native(rc["name"], rc["arguments"])
                out.append(
                    _rec(
                        t_rep,
                        "llm.commit",
                        {
                            "request_id": rep,
                            "canvas_idx": 0,
                            "segment": "tool",
                            "text": rtext,
                            "stop_id": 50,
                        },
                    )
                )
                out.append(_rec(t_rep, "llm.done", {"request_id": rep, "commits": 1}))
                rv2 = live.repair_verdict(rtext)
                spans["tool.repair"] = (t_done, t_rep)
                if rv2.status == "valid":
                    rdraft = ToolDraft(
                        call_id="t1-call",
                        status="valid",
                        name=rv2.name,
                        arguments=rv2.arguments,
                        source="repair",
                    )
                    out.append(
                        _step(
                            t_rep,
                            E.LlmDone(t_ms=t_rep, request_id=rep, tool=rdraft),
                            "SPEAKING",
                            "SPEAKING",
                            [C.CancelTimer(name="tool_repair")],
                        )
                    )
                    executed = ({"name": rv2.name, "arguments": rv2.arguments}, "repair")
                else:
                    rdraft = ToolDraft(
                        call_id="t1-call", status="refused", reason=rv2.reason, source="repair"
                    )
                    out.append(
                        _step(
                            t_rep,
                            E.LlmDone(t_ms=t_rep, request_id=rep, tool=rdraft),
                            "SPEAKING",
                            "SPEAKING",
                            [
                                C.CancelTimer(name="tool_repair"),
                                C.ToolCancel(call_id="t1-call", reason="repair_failed"),
                            ],
                        )
                    )
                    final = _tool_call(
                        "refused",
                        "repair",
                        pcall,
                        schema_valid=False,
                        repair_ms=sp.repair_ms,
                        reason=rv2.reason,
                        prefix=sp.speech,
                    )
                    out.append(_rec(t_rep, "tool.call", final))
                    if sp.execute_dead:
                        executed = (rc, "repair")
                t_drain = max(t_drain, t_rep + 100)
            elif sp.repair == "timeout":
                t_to = t_done + 3000
                tcmds: list[Any] = [C.LlmAbort(request_id=rep)] if sp.abort else []
                tcmds.append(C.ToolCancel(call_id="t1-call", reason="repair_timeout"))
                out.append(_step(t_to, E.Tick(t_ms=t_to), "SPEAKING", "SPEAKING", tcmds))
                final = _tool_call(
                    "cancelled",
                    "draft",
                    pcall,
                    schema_valid=rv,
                    reason="repair_timeout",
                    prefix=sp.speech,
                )
                out.append(_rec(t_to, "tool.call", final))
                t_drain = t_to + 100
            elif sp.repair == "barge":
                t_cut = t_done + 200
                bcmds: list[Any] = [C.LlmAbort(request_id=rep)] if sp.abort else []
                bcmds += [
                    C.TtsCancel(),
                    C.PlaybackFlush(),
                    C.ToolCancel(call_id="t1-call", reason="interrupted"),
                ]
                if sp.execute_on_barge:
                    bcmds.append(
                        C.ToolExecute(
                            call_id="t1-call",
                            turn_id=tid,
                            name=str(pcall["name"]),
                            arguments=dict(pcall["arguments"]),
                            source="draft",
                        )
                    )
                out.append(
                    _step(
                        t_cut,
                        E.UserTakeover(t_ms=t_cut, onset_ms=t_cut),
                        "SPEAKING",
                        "INTERRUPTED",
                        bcmds,
                    )
                )
                final = _tool_call(
                    "cancelled",
                    "draft",
                    pcall,
                    schema_valid=rv,
                    reason="interrupted",
                    prefix=sp.speech,
                )
                out.append(_rec(t_cut, "tool.call", final))
                out.append(
                    _step(
                        t_cut + 100,
                        E.PlaybackStopped(t_ms=t_cut + 100, played_ms=300, played_text="Okay,"),
                        "INTERRUPTED",
                        "IDLE",
                        [C.HudState(state="IDLE", turn_id=tid)],
                    )
                )
                spans["playback.stopped"] = (t_cut + 100, t_cut + 100)
                outcome = "interrupted"
            elif sp.repair == "paused_done":
                # Q8: the onset pauses the turn; the repair's verdict is accepted in PAUSED; the machine's own takeover
                # fires 600 ms after the onset, with no repair left to abort
                t_on, t_rep, t_cut = t_done + 100, t_done + sp.repair_ms, t_done + 700
                rc = sp.repair_call
                assert rc is not None, f"fixture: {sp.case} needs a repair_call"
                out.append(
                    _step(
                        t_on,
                        E.UserSpeechOnset(t_ms=t_on, onset_ms=t_on),
                        "SPEAKING",
                        "PAUSED",
                        [C.PlaybackPause()],
                    )
                )
                rtext = native(rc["name"], rc["arguments"])
                out.append(
                    _rec(
                        t_rep,
                        "llm.commit",
                        {
                            "request_id": rep,
                            "canvas_idx": 0,
                            "segment": "tool",
                            "text": rtext,
                            "stop_id": 50,
                        },
                    )
                )
                out.append(_rec(t_rep, "llm.done", {"request_id": rep, "commits": 1}))
                rv2 = live.repair_verdict(rtext)
                rdraft = ToolDraft(
                    call_id="t1-call",
                    status="valid",
                    name=rv2.name,
                    arguments=rv2.arguments,
                    source="repair",
                )
                out.append(
                    _step(
                        t_rep,
                        E.LlmDone(t_ms=t_rep, request_id=rep, tool=rdraft),
                        "PAUSED",
                        "PAUSED",
                        [C.CancelTimer(name="tool_repair")],
                    )
                )
                spans["tool.repair"] = (t_done, t_rep)
                out.append(
                    _step(
                        t_cut,
                        E.UserTakeover(t_ms=t_cut, onset_ms=t_on),
                        "PAUSED",
                        "INTERRUPTED",
                        [
                            C.TtsCancel(),
                            C.PlaybackFlush(),
                            C.ToolCancel(call_id="t1-call", reason="interrupted"),
                        ],
                    )
                )
                final = _tool_call(
                    "cancelled",
                    "repair",
                    pcall,
                    schema_valid=True,
                    repair_ms=sp.repair_ms,
                    reason="interrupted",
                    prefix=sp.speech,
                )
                out.append(_rec(t_cut, "tool.call", final))
                out.append(
                    _step(
                        t_cut + 100,
                        E.PlaybackStopped(t_ms=t_cut + 100, played_ms=300, played_text="Okay,"),
                        "INTERRUPTED",
                        "IDLE",
                        [C.HudState(state="IDLE", turn_id=tid)],
                    )
                )
                spans["playback.stopped"] = (t_cut + 100, t_cut + 100)
                outcome = "interrupted"
            elif sp.repair == "open":
                final = _tool_call(
                    "cancelled",
                    "draft",
                    pcall,
                    schema_valid=rv,
                    reason="turn closed",
                    prefix=sp.speech,
                )
    if outcome != "interrupted":
        dcmds: list[Any] = []
        if executed is not None:
            ecall, source = executed
            args = sp.execute_args if sp.execute_args is not None else ecall["arguments"]
            dcmds.append(
                C.ToolExecute(
                    call_id="t1-call",
                    turn_id=tid,
                    name=ecall["name"],
                    arguments=args,
                    source=source,
                )
            )
            spans["tool.execute"] = (t_drain, t_drain + 2)
        dcmds += [C.CommitHistory(user_only=False), C.HudState(state="IDLE", turn_id=tid)]
        out.append(
            _step(
                t_drain,
                E.PlaybackDrained(t_ms=t_drain, played_ms=900, played_text=sp.speech),
                "SPEAKING",
                "IDLE",
                dcmds,
            )
        )
        spans["playback.stopped"] = (t_drain, t_drain)
        if executed is not None and not sp.execute_dead:
            # (a call already final -- ``execute_dead`` -- gets no further record: ``_publish_tool`` drops it)
            ecall, source = executed
            args = sp.execute_args if sp.execute_args is not None else ecall["arguments"]
            done = _tool_call(
                "dry_run",
                source,
                {"name": ecall["name"], "arguments": args},
                schema_valid=True,
                repair_ms=sp.repair_ms if source == "repair" else None,
                prefix=sp.speech,
            )
            out.append(_rec(t_drain + 2, "tool.call", done))
            final = done
        elif sp.repair == "open" and final is not None:
            out.append(_rec(t_drain + 5, "tool.call", final))
    return out, {"spans": spans, "outcome": outcome, "tool": final}


def regime_line() -> Regime:
    return Regime(
        profile="lean",
        canvas_length=256,
        max_denoising_steps=48,
        thinking=True,
        vllm_version="vllm/vllm-openai:v0.29.0",
        model="nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        stt="fake",
        tts="fake",
        audio_path="null",
        git_commit=GIT,
        tools=TOOLS_ON,
    )


STRESS_SHA = "d" * 64


def build_run(root: Path, specs: list[Spec], *, name: str = "run") -> Path:
    """``turns.jsonl``, ``sessions/*.jsonl`` and ``manifest.json`` for ``specs``, one session each, in list order."""
    run = root / name
    (run / "sessions").mkdir(parents=True)
    lines: list[str] = []
    sessions: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for i, sp in enumerate(specs):
        sid = f"{i:04d}{'s' if sp.arm == 'S' else 'f'}{'0' * 7}"
        log = f"20261014-180000-{i:03d}.jsonl"
        records, kw = turn_records(sp)
        record = TurnRecord(
            turn_id="t1",
            session_id=sid,
            regime=regime_line(),
            input="text",
            canvases=2,
            session_log=str(run / "sessions" / log),
            **kw,
        )
        records.append(_rec(9000, "turn.record", record.model_dump(mode="json")))
        body: list[dict[str, Any]] = [
            {
                "seq": 0,
                "t_ms": 0,
                "wall": WALL,
                "session_ms": None,
                "kind": "log.started",
                "turn_id": None,
                "data": {"pid": 1, "path": str(run / "sessions" / log)},
            }
        ]
        for r in records:
            t = r.pop("_t")
            body.append(
                {
                    "seq": len(body),
                    "t_ms": t,
                    "wall": WALL,
                    "session_ms": t,
                    "kind": r["kind"],
                    "turn_id": "t1",
                    "data": r["data"],
                }
            )
        (run / "sessions" / log).write_text("".join(json.dumps(b) + "\n" for b in body))
        lines.append(record.to_line())
        forced = sp.arm == "F"
        sessions.append({"arm": sp.arm, "session_id": sid, "log": log, "forced": forced})
        barge = sp.repair == "barge"
        attempted = sp.repair in ("barge", "paused_done")
        cases.append(
            {
                "case_id": sp.case,
                "category": sp.category,
                "arm": sp.arm,
                "rep": sp.rep,
                "order": i,
                "session_id": sid,
                "session_log": log,
                "turn_id": "t1",
                "gt": sp.gt,
                "forced": forced,
                "barge_in": barge,
                "barge_in_attempted": attempted,
                "timed_out": False,
                "failures": [],
                **dict(sp.manifest),
            }
        )
    (run / "turns.jsonl").write_text("".join(x + "\n" for x in lines))
    manifest = {
        "regime": {
            "commit": GIT,
            "profile": "lean",
            "canvas_length": 256,
            "max_denoising_steps": 48,
            "thinking": True,
            "tools": TOOLS_ON,
            "stress_sha256": STRESS_SHA,
            "tools_sha256": tools.load().version,
            "k": 3,
            "arm_sizes": {
                "S": sum(1 for s in specs if s.arm == "S"),
                "F": sum(1 for s in specs if s.arm == "F"),
            },
            "concurrency": 1,
            "seed": None,
        },
        "sessions": sessions,
        "cases": cases,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return run


# --- reusable specs -------------------------------------------------------------------------------------------


def s_repaired(case: str = "hard-03", **kw: Any) -> Spec:
    """Arm S, T-hard: an unclosed draft, repaired to the exact call."""
    base = Spec(
        case=case,
        category="T-hard",
        gt=GT_STORE_SEARCH_T,
        draft=D_UNCLOSED,
        repair="valid",
        repair_call=call("type_at", x=500, y=36, text="{hdmi,usb-c} cable", press_enter=True),
    )
    return replace(base, **kw)


def f_forced(case: str = "click-03", **kw: Any) -> Spec:
    """Arm F: a valid draft, forced into the repair, which returns the same call."""
    base = Spec(
        case=case,
        arm="F",
        category="click_at",
        gt=GT_CLICK,
        draft=D_CLICK,
        forced=True,
        repair="valid",
        repair_call=call("click_at", x=800, y=380),
    )
    return replace(base, **kw)


def clean_specs() -> list[Spec]:
    return [s_repaired(), f_forced()]


@pytest.fixture
def h() -> ModuleType:
    return harness()


# =====================================================================================================
# The fixture is what it says (not a row of the harness; it runs regardless)
# =====================================================================================================


def test_fixture_drafts_judge_as_named() -> None:
    """The drafts above judge, under the unpatched ``live.classify``, as their comments say -- so the rows' expected
    values rest on the defined judge, not on the fixture's say-so."""
    expect = {
        D_KEY_ALIAS: ("repairable", "schema invalid"),
        D_ENTER: ("valid", None),
        D_UNCLOSED: ("repairable", "unterminated string"),
        D_NAMELESS: ("refused", None),
        D_CLICK: ("valid", None),
        D_SCROLL_AMOUNT: ("repairable", "schema invalid"),
        D_TYPE_NO_ENTER: ("repairable", "schema invalid"),
    }
    harness()  # pending until the harness exists; the checks below are the fixture's
    for text, (status, reason) in expect.items():
        v = live.classify(text)
        assert v.status == status, (text, v)
        if reason is not None:
            assert v.reason == reason, (text, v)


def test_fixture_h99_drafts_judge_refused() -> None:
    """T-0100 rewrote these two fixture premises (they were ``repairable`` "schema invalid" until 0756893): F5 is not
    in the press_key list (H-T99d) and x = 1024 is outside 0-999 (H-T99e) -- both refused at detection, name kept."""
    for text, name, reason in (
        (D_F5, "press_key", "combo not in enum"),
        (D_CLICK_PX, "click_at", "coordinate out of range"),
    ):
        v = live.classify(text)
        assert (v.status, v.name, v.reason) == ("refused", name, reason), (text, v)


# =====================================================================================================
# Yield and the forcing (criteria 6a, 7, 8)
# =====================================================================================================


def test_yield_is_rejudged_from_the_commit_not_the_record(h: ModuleType, tmp_path: Path) -> None:
    """Adversarial: the records lie both ways and arm F's records all say ``schema_valid: false``.

    Arm S: kout-01 (draft Control+C, schema invalid, its record says valid), hard-03 (unclosed, record as written),
    cpix-01 (a valid click whose record says invalid, no repair), sx-01 unrouted. Invalid drafts re-judged from the
    commits: kout-01 and hard-03 -> k = 2; routed tool turns: 3 -> yield 2/3. Arm F's two forced drafts (records
    false) are not arm S and are valid anyway: they add nothing."""
    specs = [
        Spec(
            case="kout-01",
            category="K-out",
            gt=None,
            draft=D_KEY_ALIAS,
            record_valid=True,
            repair="invalid",
            repair_call=call("press_key", combo="F5"),
        ),
        s_repaired(),
        Spec(case="cpix-01", category="C-pixel", gt=GT_CLICK, draft=D_CLICK, record_valid=False),
        Spec(case="sx-01", category="S-extra", gt=GT_SCROLL_DOWN, draft=None),
        f_forced("click-03"),
        f_forced("click-04", rep=0),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    s = report["S"]
    assert Fraction(s["yield"]["k"], s["yield"]["n"]) == Fraction(2, 3)
    assert (s["yield"]["k"], s["yield"]["n"]) == (2, 3)
    assert s["yield_by_category"]["K-out"] == {"k": 1, "n": 1}
    assert s["yield_by_category"]["T-hard"] == {"k": 1, "n": 1}
    assert s["yield_by_category"]["C-pixel"] == {"k": 0, "n": 1}
    assert s["yield_by_reason"] == {"schema invalid": 1, "unterminated string": 1}
    assert s["unrouted"] == 1
    assert report["void"] == [], report["void"]


def test_nameless_drafts_are_counted_separately(h: ModuleType, tmp_path: Path) -> None:
    """A nameless draft is refused by ``classify``: never an invalid draft, never repaired, counted on its own."""
    specs = [
        Spec(case="under-02", category="U-under", gt=GT_STORE_SEARCH_U, draft=D_NAMELESS),
        f_forced(),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    # Q5: counted separately, not in the yield denominator
    assert (report["S"]["yield"]["k"], report["S"]["yield"]["n"]) == (0, 0)
    assert report["S"]["refused_drafts"] == 1
    assert report["S"]["repairs"]["n"] == 0


def test_forced_repairs_never_count_as_rescues(h: ModuleType, tmp_path: Path) -> None:
    """Adversarial, the most likely failure: arm F's three forced repairs all return a valid, exact call. Arm S has
    two turns: one invalid draft repaired to valid (a rescue) and one valid draft that is never repaired.

    Arm S by hand: yield 1/2, repairs 1, rescue 1/1, rescue-exact 1/1, outcomes {dry_run: 1}. Pooling arm F in would
    read repairs 4, rescue 4/4 and outcomes {dry_run: 4}. Arm F: repairs 3, agreement 3/3."""
    specs = [
        s_repaired(),
        Spec(
            case="kin-01",
            category="K-in",
            gt={"name": "press_key", "combo": "Enter"},
            draft=D_ENTER,
        ),
        f_forced("click-03"),
        f_forced("click-04"),
        f_forced("click-05"),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    s, f = report["S"], report["F"]
    assert (s["yield"]["k"], s["yield"]["n"]) == (1, 2)
    assert s["repairs"]["n"] == 1
    assert s["repairs"]["outcomes"] == {"dry_run": 1}
    assert (s["rescue"]["k"], s["rescue"]["n"]) == (1, 1)
    assert (s["rescue_exact"]["k"], s["rescue_exact"]["n"]) == (1, 1)
    assert s["coercion"]["n"] == 0
    assert f["repairs"]["n"] == 3
    assert (f["agreement"]["k"], f["agreement"]["n"]) == (3, 3)
    assert report["void"] == [], report["void"]


@pytest.mark.parametrize("leak", ["repair_on_valid_draft", "marker_on_record", "none"])
def test_leak_of_forcing_into_arm_s_voids(h: ModuleType, tmp_path: Path, leak: str) -> None:
    """Criterion 8: an arm-S repair whose pass-1 draft re-judges ``valid`` voids the run, and so does an arm-S
    ``tool.call`` record carrying the forced marker. The control (``none``) has arm F's forced repairs on valid
    drafts and its marker where the session puts it, and is not void: the check is not too wide."""
    specs = clean_specs()
    if leak == "repair_on_valid_draft":
        specs.append(
            Spec(
                case="kin-01",
                category="K-in",
                gt={"name": "press_key", "combo": "Enter"},
                draft=D_ENTER,
                repair="valid",
                repair_call=call("press_key", combo="Enter"),
            )
        )
    elif leak == "marker_on_record":
        specs.append(
            Spec(
                case="under-02",
                category="U-under",
                gt=GT_STORE_SEARCH_U,
                draft=D_NAMELESS,
                marker_on_record=True,
            )
        )
    report = h.build_report(build_run(tmp_path, specs))
    void = [str(v) for v in report["void"]]
    if leak == "none":
        assert void == []
        assert all(h.gates(report)[arm]["G1"] for arm in ("S", "F"))
    elif leak == "repair_on_valid_draft":
        assert any("integrity: repair on a valid draft outside arm F" in v for v in void), void
        assert h.gates(report)["S"]["G1"] is False
    else:
        assert any("forced" in v.lower() for v in void), void
        assert h.gates(report)["S"]["G1"] is False


# =====================================================================================================
# Outcomes, spans, percentiles (criterion 6b-d, 6h; J6, J9)
# =====================================================================================================


def test_outcomes_by_final_status(h: ModuleType, tmp_path: Path) -> None:
    """J6: keyed by the call's last ``tool.call`` status. Arm S: one repair to valid (dry_run), two to invalid
    (refused), one timeout (cancelled, reason repair_timeout); a valid draft executed without repair is no repair.
    By hand: S repairs 4, outcomes {dry_run: 1, refused: 2, cancelled: 1}. Arm F: one forced repair to valid and one
    timeout -> F outcomes {dry_run: 1, cancelled: 1}, repair_timeout 1."""
    specs = [
        s_repaired(),
        s_repaired(
            "hard-01",
            repair="invalid",
            repair_call=call("type_at", x=500, y=36, text="x", press_enter="yes"),
        ),
        Spec(
            case="kout-01",
            category="K-out",
            gt=None,
            draft=D_KEY_ALIAS,
            repair="invalid",
            repair_call=call("press_key", combo="F5"),
        ),
        s_repaired("hard-02", repair="timeout"),
        Spec(
            case="kin-01",
            category="K-in",
            gt={"name": "press_key", "combo": "Enter"},
            draft=D_ENTER,
        ),
        f_forced("click-03"),
        f_forced("click-04", repair="timeout"),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    assert report["S"]["repairs"]["n"] == 4
    assert report["S"]["repairs"]["outcomes"] == {"dry_run": 1, "refused": 2, "cancelled": 1}
    assert report["F"]["repairs"]["outcomes"] == {"dry_run": 1, "cancelled": 1}
    assert report["F"]["repair_timeout"] == 1
    assert report["void"] == [], report["void"]


def test_tool_repair_span_is_the_repair_span(h: ModuleType, tmp_path: Path) -> None:
    """``tool.repair`` p50/p90 come from the turn's ``tool.repair`` span, not ``llm.done``. Arm S repairs at 250, 350
    and 450 ms (llm.done spans 1500, 1700, 1900): nearest-rank p50 = 2nd of 3 = 350, p90 = ceil(2.7) = 3rd = 450,
    n 3. Read from ``llm.done`` it would be 1700/1900. Arm F's 120 ms repair is its own (n 1, p50 120)."""
    specs = [
        s_repaired("hard-01", repair_ms=450, done_ms=1500),
        s_repaired("hard-02", repair_ms=250, done_ms=1700),
        s_repaired("hard-03", repair_ms=350, done_ms=1900),
        f_forced(repair_ms=120),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    assert report["S"]["repairs"]["ms"] == {"n": 3, "p50": 350, "p90": 450}
    assert report["F"]["repairs"]["ms"] == {"n": 1, "p50": 120, "p90": 120}


def test_percentiles_are_nearest_rank_with_n(h: ModuleType, tmp_path: Path) -> None:
    """J9 on five repairs of 100, 200, 300, 400 and 900 ms: p50 = ceil(2.5) = 3rd = 300; p90 = ceil(4.5) = 5th = 900
    (linear interpolation would give 700 for p90); n 5."""
    specs = [
        s_repaired(f"hard-0{i + 1}", repair_ms=ms) for i, ms in enumerate([900, 100, 300, 200, 400])
    ]
    specs.append(f_forced())
    report = h.build_report(build_run(tmp_path, specs))
    assert report["S"]["repairs"]["ms"] == {"n": 5, "p50": 300, "p90": 900}


def test_arm_f_latency_is_labelled_a_floor(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 6: arm F's ``tool.repair`` line is labelled as the floor on valid drafts."""
    report = h.build_report(build_run(tmp_path, clean_specs()))
    md = h.render_markdown(report)
    assert "floor on valid drafts, not repair latency on invalid drafts" in md


# =====================================================================================================
# Rescue, rescue-exact, coercion, U-under (criteria 1, 6e-g, 7)
# =====================================================================================================


def _scroll_repairs(n_exact: int, n_wrong: int, n_invalid: int) -> list[Spec]:
    out: list[Spec] = []
    calls = (
        [call("scroll", direction="down")] * n_exact
        + [call("scroll", direction="up")] * n_wrong
        + [call("scroll", direction="sideways")] * n_invalid
    )
    for i, c in enumerate(calls):
        out.append(
            Spec(
                case=f"sx-0{i % 4 + 1}",
                rep=i // 4,
                category="S-extra",
                gt=GT_SCROLL_DOWN,
                draft=D_SCROLL_AMOUNT,
                repair="valid" if c["arguments"]["direction"] != "sideways" else "invalid",
                repair_call=c,
            )
        )
    return out


def test_rescue_read_only_at_ten(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 7 at the boundary.

    Ten: 10 S-extra repairs (gt-bearing): 5 exact (down), 2 valid but wrong (up), 3 invalid. Rescue = 7/10, read;
    Wilson 95 % by hand (z = 1.96, z^2 = 3.8416): denominator 1 + 3.8416/10 = 1.38416; centre (0.7 + 0.19208) /
    1.38416 = 0.644492; half 1.96 * sqrt(0.7*0.3/10 + 3.8416/400) / 1.38416 = 1.96 * sqrt(0.030604) / 1.38416 =
    0.247719 -> [0.396773, 0.892211]. Rescue-exact 5/10, read.

    Nine: 9 S-extra repairs plus one K-out repair (coerced to Enter): 10 repairs fired, but only 9 on gt-bearing
    turns -> "insufficient yield (n = 9)", no Wilson interval. Dropping the threshold, or counting the K-out repair
    toward it, reads it."""
    ten = [*_scroll_repairs(5, 2, 3), f_forced()]
    report = h.build_report(build_run(tmp_path, ten, name="ten"))
    r = report["S"]["rescue"]
    assert (r["k"], r["n"]) == (7, 10)
    assert r["reading"] == "read"
    lo, hi = r["wilson95"]
    assert lo == pytest.approx(0.396773, abs=1e-3) and hi == pytest.approx(0.892211, abs=1e-3)
    e = report["S"]["rescue_exact"]
    assert (e["k"], e["n"], e["reading"]) == (5, 10, "read")

    nine = [
        *_scroll_repairs(5, 2, 2),
        Spec(
            case="kout-01",
            category="K-out",
            gt=None,
            draft=D_KEY_ALIAS,
            repair="valid",
            repair_call=call("press_key", combo="Enter"),
        ),
        f_forced(),
    ]
    report9 = h.build_report(build_run(tmp_path, nine, name="nine"))
    r9 = report9["S"]["rescue"]
    assert str(r9["reading"]).startswith("insufficient yield (n = 9)"), r9
    assert r9["wilson95"] is None
    assert str(report9["S"]["rescue_exact"]["reading"]).startswith("insufficient yield (n = 9)")


def test_coercion_counted_on_k_out(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 6g: a K-out turn whose repair produced a valid call is coercion, with the combo chosen -- never a
    rescue (repaired-valid-but-wrong). Arm S: kout-01 (an aliased in-list draft, Control+C, since T-0100 refuses F5)
    repaired to Enter, kout-02 (the same draft) repaired to an invalid call, hard-03 repaired exactly. By hand: coercion 1 ([kout-01, rep 0, Enter]); rescue k = 1 (hard-03
    only), n = 3 repairs fired; rescue-exact 1/1 (K-out carries no gt)."""
    specs = [
        Spec(
            case="kout-01",
            category="K-out",
            gt=None,
            draft=D_KEY_ALIAS,
            repair="valid",
            repair_call=call("press_key", combo="Enter"),
        ),
        Spec(
            case="kout-02",
            category="K-out",
            gt=None,
            draft=D_KEY_ALIAS,
            repair="invalid",
            repair_call=call("press_key", combo="Ctrl+S"),
        ),
        s_repaired(),
        f_forced(),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    s = report["S"]
    assert s["coercion"]["n"] == 1
    assert [(c["case_id"], c["rep"], c["combo"]) for c in s["coercion"]["cases"]] == [
        ("kout-01", 0, "Enter")
    ]
    assert (s["rescue"]["k"], s["rescue"]["n"]) == (1, 3)
    assert (s["rescue_exact"]["k"], s["rescue_exact"]["n"]) == (1, 1)


def test_h99_refusals_are_refused_drafts_never_yield_nor_coercion(
    h: ModuleType, tmp_path: Path
) -> None:
    """T-0100 criterion 12 (Q5): kout-01 drafting F5 (H-T99d) and cpix-03 drafting x = 1024 (H-T99e) are refused at
    detection -- no repair, nothing executed. Re-judged with the unpatched ``classify`` they are ``refused``: counted
    in ``refused_drafts`` (2), out of the yield denominator, never coercion. By hand: arm S routed = kin-01 only
    (valid) -> yield 0/1, yield_by_reason {}, repairs 0, coercion 0; G1-G8 hold, not void."""
    specs = [
        Spec(case="kout-01", category="K-out", gt=None, draft=D_F5),
        Spec(case="cpix-03", category="C-pixel", gt=GT_CLICK, draft=D_CLICK_PX),
        Spec(
            case="kin-01",
            category="K-in",
            gt={"name": "press_key", "combo": "Enter"},
            draft=D_ENTER,
        ),
        f_forced(),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    s = report["S"]
    assert s["refused_drafts"] == 2, s["refused_drafts"]
    assert (s["yield"]["k"], s["yield"]["n"]) == (0, 1)
    assert s["yield_by_reason"] == {}
    assert s["repairs"]["n"] == 0
    assert s["coercion"]["n"] == 0 and s["coercion"]["cases"] == []
    assert report["void"] == [], report["void"]
    g = h.gates(report)
    assert all(g["S"].values()) and all(g["F"].values()), g


def test_u_under_rule(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 1's U-under rule: rescued = valid, inside the bbox, text equal; ``press_enter`` reported, not scored.
    under-02 repaired with press_enter true (gt says nothing): exact. under-02 rep 1 repaired with "wireless mice":
    not exact. hard-03 repaired with press_enter false where the gt says true: not exact (score.exact). By hand:
    rescue 3/3, rescue-exact 1/3."""
    specs = [
        Spec(
            case="under-02",
            category="U-under",
            gt=GT_STORE_SEARCH_U,
            draft=D_TYPE_NO_ENTER,
            repair="valid",
            repair_call=call("type_at", x=500, y=36, text="wireless mouse", press_enter=True),
        ),
        Spec(
            case="under-02",
            rep=1,
            category="U-under",
            gt=GT_STORE_SEARCH_U,
            draft=D_TYPE_NO_ENTER,
            repair="valid",
            repair_call=call("type_at", x=500, y=36, text="wireless mice", press_enter=False),
        ),
        s_repaired(
            repair_call=call("type_at", x=500, y=36, text="{hdmi,usb-c} cable", press_enter=False)
        ),
        f_forced(),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    s = report["S"]
    assert (s["rescue"]["k"], s["rescue"]["n"]) == (3, 3)
    assert (s["rescue_exact"]["k"], s["rescue_exact"]["n"]) == (1, 3)


# =====================================================================================================
# Hard gates (criterion 5)
# =====================================================================================================


def test_clean_run_holds_every_gate(h: ModuleType, tmp_path: Path) -> None:
    """The control for every gate row below: the clean run holds G1-G8 in both arms and is not void."""
    report = h.build_report(build_run(tmp_path, clean_specs()))
    g = h.gates(report)
    assert set(g) == {"S", "F"}
    for arm in ("S", "F"):
        assert set(g[arm]) == GATE_KEYS, g[arm]
        assert all(g[arm].values()), (arm, g[arm])
    assert report["void"] == []


def test_g2_repair_commit_voiced_fails(h: ModuleType, tmp_path: Path) -> None:
    """R9: a ``TtsSpeak`` whose source comes from the repair request's commits fails G2 in that arm (F here) only."""
    specs = [s_repaired(), f_forced(repair_voiced=True)]
    g = h.gates(h.build_report(build_run(tmp_path, specs)))
    assert g["F"]["G2"] is False
    assert g["S"]["G2"] is True


def test_g6_two_repairs_fail(h: ModuleType, tmp_path: Path) -> None:
    """G6: two ``tool_repair`` requests in one turn fail G6 in that arm only."""
    specs = [s_repaired(repair_requests=2), f_forced()]
    g = h.gates(h.build_report(build_run(tmp_path, specs)))
    assert g["S"]["G6"] is False
    assert g["F"]["G6"] is True


@pytest.mark.parametrize("how", ["after_repair_failed", "repair_call_schema_invalid"])
def test_g7_executed_refused_call_fails(h: ModuleType, tmp_path: Path, how: str) -> None:
    """G7: a ``ToolExecute`` for a call whose repair failed (final ``refused``), and an executed ``source == "repair"``
    call that is schema-invalid under tools_v1 re-judged from the log (``combo: F5`` is out of the enum), each fail G7
    in arm S and leave arm F holding."""
    if how == "after_repair_failed":
        bad = s_repaired(
            repair="invalid",
            repair_call=call("type_at", x=500, y=36, text="x", press_enter="yes"),
            execute_dead=True,
        )
    else:
        bad = Spec(
            case="kout-01",
            category="K-out",
            gt=None,
            draft=D_KEY_ALIAS,
            repair="valid",
            repair_call=call("press_key", combo="Enter"),
            execute_args={"combo": "F5"},
        )
    g = h.gates(h.build_report(build_run(tmp_path, [bad, f_forced()])))
    assert g["S"]["G7"] is False
    assert g["F"]["G7"] is True


@pytest.mark.parametrize(
    ("variant", "holds"),
    [
        ("open_at_close", False),
        ("timeout_with_abort", True),
        ("timeout_without_abort", False),
        ("verdict_after_deadline", False),
    ],
)
def test_g8_open_repair_at_close_fails(
    h: ModuleType, tmp_path: Path, variant: str, holds: bool
) -> None:
    """G8: every repair ends with a verdict before 3000 ms or as ``repair_timeout`` with its abort logged; none open at
    close. A repair with no verdict and no abort fails; a timeout with its ``LlmAbort`` holds (the control); a timeout
    without the abort fails; a verdict 3200 ms after the repair started fails."""
    sp = {
        "open_at_close": s_repaired(repair="open"),
        "timeout_with_abort": s_repaired(repair="timeout"),
        "timeout_without_abort": s_repaired(repair="timeout", abort=False),
        "verdict_after_deadline": s_repaired(repair_ms=3200),
    }[variant]
    g = h.gates(h.build_report(build_run(tmp_path, [sp, f_forced()])))
    assert g["S"]["G8"] is holds
    assert g["F"]["G8"] is True


@pytest.mark.parametrize("aborts", [3, 2])
def test_barge_during_repair_row(h: ModuleType, tmp_path: Path, aborts: int) -> None:
    """Criterion 4: three arm-F barge-ins during the repair, each interrupted with a call on its line. With all three
    repair aborts logged the row reads 3 delivered; with one abort missing it reads 2 and is "unread" -- and the run
    is not void for it. The manifest says ``barge_in: true`` for all three either way: the report re-derives it."""
    specs: list[Spec] = [s_repaired()]
    for i, case in enumerate(("click-01", "click-03", "click-06")):
        specs.append(
            f_forced(case, repair="barge", abort=i < aborts, manifest=(("barge_in", True),))
        )
    specs.append(f_forced("click-07"))
    report = h.build_report(build_run(tmp_path, specs))
    row = report["F"]["barge_in"]
    assert row["delivered"] == aborts
    if aborts == 3:
        assert row["reading"] == "read"
    else:
        assert str(row["reading"]).startswith("unread"), row
    assert report["void"] == [], report["void"]
    g = h.gates(report)
    assert g["F"]["G3"] is True
    # G8's barge-abort clause: a repair aborted by the takeover ends neither by verdict nor by timeout, and holds;
    # the barge-in whose abort is missing leaves its repair open at close, and fails
    assert g["F"]["G8"] is (aborts == 3)


def test_g3_execute_on_interrupted_turn_fails(h: ModuleType, tmp_path: Path) -> None:
    """G3: a ``ToolExecute`` on an interrupted turn (here in the takeover step of an arm-F barge-in) fails G3 in arm F
    only; the same barge-in without it holds (``test_barge_during_repair_row``)."""
    specs = [s_repaired(), f_forced("click-01", repair="barge", execute_on_barge=True)]
    g = h.gates(h.build_report(build_run(tmp_path, specs)))
    assert g["F"]["G3"] is False
    assert g["S"]["G3"] is True


def test_barge_in_counts_are_separate(h: ModuleType, tmp_path: Path) -> None:
    """Ruling Q8: attempted, delivered and "repair finished before the takeover window" are separate counts. Arm F:
    three barge-ins that abort the repair (delivered), one whose repair verdict is accepted in PAUSED before the
    machine's takeover (attempted, not delivered: no repair abort), one turn with no barge-in. By hand: attempted 4,
    delivered 3, repair done before the takeover 1; reading "read" (3 delivered); the run is not void and G3/G8 hold."""
    specs: list[Spec] = [s_repaired()]
    for case in ("click-01", "click-03", "click-06"):
        specs.append(f_forced(case, repair="barge"))
    specs.append(f_forced("click-08", repair="paused_done", repair_ms=300))
    specs.append(f_forced("click-07"))
    report = h.build_report(build_run(tmp_path, specs))
    row = report["F"]["barge_in"]
    assert row.get("attempted") == 4, row
    assert row.get("delivered") == 3, row
    assert row.get("repair_done_before_takeover") == 1, row
    assert row.get("reading") == "read", row
    assert report["void"] == [], report["void"]
    g = h.gates(report)
    assert g["F"]["G3"] is True and g["F"]["G8"] is True


def test_barge_in_attempts_are_capped_at_five(h: ModuleType) -> None:
    """Ruling Q8: at most 5 attempts, in run order among the pre-registered candidates (not "until 3 delivered")."""
    assert getattr(h, "BARGE_IN_MAX_ATTEMPTS", None) == 5


# =====================================================================================================
# The regime, the patch and the reuse (criteria 3, 9; implementation notes)
# =====================================================================================================


def test_regime_block(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 9: the commit, the live system prompt's sha256 from the **first** session's ``llm.request`` (a later
    session carries another prompt, so "any" or "last" reads differently), the stress sha, the tools sha, k and the
    arm sizes."""
    specs = [
        s_repaired(),
        s_repaired("hard-01", system_text=SYSTEM_TEXT + " (changed)"),
        f_forced(),
    ]
    report = h.build_report(build_run(tmp_path, specs))
    regime = report["regime"]
    assert regime["prompt_sha256"] == hashlib.sha256(SYSTEM_TEXT.encode("utf-8")).hexdigest()
    assert regime["commit"] == GIT
    assert regime["stress_sha256"] == STRESS_SHA
    assert regime["tools_sha256"] == tools.load().version
    assert regime["k"] == 3
    assert regime["arm_sizes"] == {"S": 2, "F": 1}


def test_forced_classify_wraps_and_restores(h: ModuleType) -> None:
    """Criterion 3: inside the patch a valid draft judges ``repairable`` with the marker and its own name and
    arguments; an invalid one judges as the real judge says; after the block -- even one that raised -- ``live.classify``
    is the original object."""
    original = live.classify
    with h.forced_classify():
        assert live.classify is not original
        v = live.classify(D_CLICK)
        assert (v.status, v.name, v.arguments, v.reason) == (
            "repairable",
            "click_at",
            {"x": 800, "y": 380},
            FORCED,
        )
        assert live.classify(D_F5) == original(D_F5)
        assert live.classify(D_NAMELESS) == original(D_NAMELESS)
    assert live.classify is original
    with pytest.raises(RuntimeError, match="inside a session"), h.forced_classify():
        raise RuntimeError("inside a session")
    assert live.classify is original
    assert h.FORCED_REASON == FORCED


def test_stress_and_forced_case_loaders(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 2: arm S is the pinned stress file (30 cases, file order, k = 3), arm F the 40 dev tool cases in file
    order; a stress file that is not the pinned one is refused."""
    assert h.K == 3
    assert hashlib.sha256(STRESS.read_bytes()).hexdigest() == h.STRESS_SHA256
    stress_ids = [json.loads(x)["id"] for x in STRESS.read_text().splitlines() if x.strip()]
    assert [c.case_id for c in h.load_stress_cases()] == stress_ids
    dev_tool = [
        r["id"]
        for r in (json.loads(x) for x in V1_CASES.read_text().splitlines() if x.strip())
        if r["split"] == "dev" and r["kind"] == "tool"
    ]
    assert len(dev_tool) == 40
    assert [c.case_id for c in h.load_forced_cases()] == dev_tool
    other = tmp_path / "cases.jsonl"
    other.write_bytes(STRESS.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        h.load_stress_cases(other)


def test_imported_t0093_helpers_exist() -> None:
    """The T-0093 helpers the harness reuses exist under their names, and the harness loads with them (a rename there
    turns this row red in the fast suite)."""
    t0093 = _load(T0093, "_t0093_for_t0099_unit")
    missing = [name for name in T0093_HELPERS if not hasattr(t0093, name)]
    assert missing == [], f"T-0093 helpers renamed or gone: {missing}"
    harness()


@pytest.fixture(autouse=True)
def _classify_untouched() -> Iterator[None]:
    """No row may leave ``live.classify`` patched (criterion 3)."""
    original = live.classify
    yield
    leaked = live.classify is not original
    live.classify = original  # type: ignore[assignment]
    assert not leaked, "a row left local_astra.toolcall.live.classify patched"


# =====================================================================================================
# T-0101 (added rows): the warm-up's regime key and void (criterion 6), first-use rows (criterion 7), and the
# T-0099 keys unchanged (criterion 10)
#
# The assumed surface, beyond T-0099's:
# - ``report["first_use_repair"]``: a dict keyed by payload ``automaton_sha256``; for every sha some ``tool_repair``
#   request used, the first such request in manifest session order (either arm) as ``{"tool", "arm", "case_id",
#   "rep", "ms"}`` (``ms`` = that turn's ``tool.repair`` span length, ``None`` when it has none). A sha no repair used
#   has no entry (or ``None``).
# - ``report["first_use_repair_max_ms"]``: the largest ``ms`` among those entries.
# - ``report["regime"]["repair_warmup"]``: the manifest's ``regime.repair_warmup``, or ``{"on": False}`` when the
#   manifest has none (a pre-T-0101 run dir). ``per_tool[<tool>]["status"]`` is the warm-up status of that tool.
# - a warm-up with ``on: true`` leaving a tool not ``ok`` voids the run with
#   ``integrity: repair warm-up incomplete (<tool>: <status>)``.
# The sha of a repair is read from the ``cd`` the session logs on its ``tool_repair`` ``llm.request`` record
# (``request.model_dump(mode="json", exclude={"automaton"})``, ``Session._run_repair``); ``with_cd`` adds it to the
# hand-built logs, which T-0099's fixture leaves out.
# =====================================================================================================

#: the commit T-0101's branch was cut at (main with T-0100 merged): the T-0099 harness as it was before T-0101
T0101_BASE = "878e65fcb009ed06321637dc6bd3f27c1da07653"
D_SCROLL_OK = native("scroll", {"direction": "down"})  # valid
D_SEARCH = native("google_search", {"query": "usb c cable"})  # valid


def with_cd(run: Path, specs: list[Spec]) -> Path:
    """Add the logged ``cd`` (real payload sha of the drafted tool) to every ``tool_repair`` ``llm.request``."""
    payloads, calibration = live.load_payloads(), live.load_calibration()
    for i, sp in enumerate(specs):
        if sp.draft is None or sp.repair is None:
            continue
        name = live.classify(sp.draft).name
        assert name is not None, f"fixture: {sp.case}'s draft names no tool"
        path = run / "sessions" / f"20261014-180000-{i:03d}.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        for r in rows:
            d = r.get("data") or {}
            if r.get("kind") == "llm.request" and d.get("phase") == "tool_repair":
                cd = live.repair_request(
                    name, [238, 239], 50, str(d["attempt_id"]), payloads, calibration
                )
                d["cd"] = cd.model_dump(mode="json", exclude={"automaton"})
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def set_warmup(run: Path, warmup: dict[str, Any] | None) -> Path:
    manifest = json.loads((run / "manifest.json").read_text())
    if warmup is None:
        manifest["regime"].pop("repair_warmup", None)
    else:
        manifest["regime"]["repair_warmup"] = warmup
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return run


def warm(statuses: dict[str, str] | None = None) -> dict[str, Any]:
    """A warm-up regime key: every payload tool ``ok`` unless ``statuses`` says otherwise."""
    per = {
        t: {"status": "ok", "wall_ms": 300, "commits": 1}
        for t in sorted(live.load_payloads().tools)
    }
    for t, s in (statuses or {}).items():
        per[t]["status"] = s
    return {
        "on": True,
        "per_tool": per,
        "total_ms": 1500,
        "ended_wall": "2026-10-14T17:59:00+01:00",
    }


def first_use_specs() -> list[Spec]:
    """Session order: type_at first at 2100 ms (arm S), scroll first at 1900 (S), type_at again at 300 (S, rep 1),
    press_key first at 2500 (S, repaired to an invalid call), click_at first at 1800 (arm F only), click_at again
    at 250 (F), scroll again at 200 (F). google_search never repaired."""
    return [
        s_repaired("hard-03", repair_ms=2100),
        Spec(
            case="sx-01",
            category="S-extra",
            gt=GT_SCROLL_DOWN,
            draft=D_SCROLL_AMOUNT,
            repair="valid",
            repair_call=call("scroll", direction="down"),
            repair_ms=1900,
        ),
        s_repaired("hard-03", rep=1, repair_ms=300),
        Spec(
            case="kout-01",
            category="K-out",
            gt=None,
            draft=D_KEY_ALIAS,
            repair="invalid",
            repair_call=call("press_key", combo="F5"),
            repair_ms=2500,
        ),
        f_forced("click-03", repair_ms=1800),
        f_forced("click-04", repair_ms=250),
        Spec(
            case="scroll-01",
            arm="F",
            category="scroll",
            gt=GT_SCROLL_DOWN,
            draft=D_SCROLL_OK,
            forced=True,
            repair="valid",
            repair_call=call("scroll", direction="down"),
            repair_ms=200,
        ),
    ]


def _sha(tool: str) -> str:
    return live.load_payloads().tools[tool].automaton_sha256


FIRST_USE_EXPECTED = {
    "type_at": {"tool": "type_at", "arm": "S", "case_id": "hard-03", "rep": 0, "ms": 2100},
    "scroll": {"tool": "scroll", "arm": "S", "case_id": "sx-01", "rep": 0, "ms": 1900},
    "press_key": {"tool": "press_key", "arm": "S", "case_id": "kout-01", "rep": 0, "ms": 2500},
    "click_at": {"tool": "click_at", "arm": "F", "case_id": "click-03", "rep": 0, "ms": 1800},
}


def _assert_first_use(report: dict[str, Any]) -> None:
    rows = report.get("first_use_repair")
    assert isinstance(rows, dict), f"criterion 7: no first_use_repair dict in the report ({rows!r})"
    for tool, want in FIRST_USE_EXPECTED.items():
        got = rows.get(_sha(tool))
        assert got is not None, f"criterion 7: no first-use row for {tool}"
        assert {k: got.get(k) for k in want} == want, (tool, got)
    assert rows.get(_sha("google_search")) is None, "google_search was never repaired: no first use"
    assert report.get("first_use_repair_max_ms") == 2500


def test_first_use_rows_per_sha_in_session_order(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 7 (adversarial for "first use from arm S only" and "last use"): by hand from ``first_use_specs``,
    type_at's first use is hard-03 rep 0 (2100, not rep 1's 300), click_at's is arm F's click-03 (arm S never repaired
    it), scroll's is arm S's sx-01 (1900, not arm F's 200); max 2500 (press_key). A clean warm-up: not void."""
    specs = first_use_specs()
    run = set_warmup(with_cd(build_run(tmp_path, specs), specs), warm())
    report = h.build_report(run)
    _assert_first_use(report)
    assert report["regime"].get("repair_warmup") == warm()
    assert report["void"] == [], report["void"]


def test_first_use_rows_on_pre_t0101_dir(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 7: a run dir with no ``regime.repair_warmup`` (a794a98's) reads ``{"on": false}`` and still gets the
    same first-use rows; not void."""
    specs = first_use_specs()
    run = set_warmup(with_cd(build_run(tmp_path, specs), specs), None)
    report = h.build_report(run)
    assert report["regime"].get("repair_warmup") == {"on": False}
    _assert_first_use(report)
    assert report["void"] == [], report["void"]


@pytest.mark.parametrize(
    ("tool", "status"),
    [("scroll", "failed: HTTP 500"), ("type_at", "skipped")],
    ids=["failed", "skipped"],
)
def test_incomplete_warmup_voids(h: ModuleType, tmp_path: Path, tool: str, status: str) -> None:
    """Criterion 6: a warm-up that left a tool not ``ok`` voids the run, naming it (G1 fails in both arms). In the same
    row, the controls on the same clean T-0099 pair: a clean warm-up and a warm-up-off run are not void (the check is
    not too wide; a control alone passes on today's harness, so it rides with the red half)."""
    specs = clean_specs()
    runs = {}
    for label, warmup in (
        ("bad", warm({tool: status})),
        ("clean", warm()),
        ("off", {"on": False}),
    ):
        runs[label] = (
            warmup,
            set_warmup(with_cd(build_run(tmp_path, specs, name=label), specs), warmup),
        )
    reports = {label: h.build_report(run) for label, (_, run) in runs.items()}
    for label, (warmup, _) in runs.items():
        assert reports[label]["regime"].get("repair_warmup") == warmup, label
    void = [str(v) for v in reports["bad"]["void"]]
    want = f"integrity: repair warm-up incomplete ({tool}: {status})"
    assert any(want in v for v in void), (want, void)
    g = h.gates(reports["bad"])
    assert g["S"]["G1"] is False and g["F"]["G1"] is False
    for label in ("clean", "off"):
        assert reports[label]["void"] == [], (label, reports[label]["void"])
        assert all(h.gates(reports[label])[arm]["G1"] for arm in ("S", "F")), label


def _harness_at(sha: str) -> ModuleType:
    """The T-0099 harness as committed at ``sha``, executed in memory (no file written); skips when the commit is absent."""
    import subprocess

    shown = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{sha}:scripts/gpu/t0099_repair_live.py"],
        capture_output=True,
        text=True,
    )
    if shown.returncode != 0:
        pytest.skip(f"commit {sha[:12]} is not in this clone: {shown.stderr.strip()[:200]}")
    name = "_t0099_repair_live_pre_t0101"
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(HARNESS)  # its REPO is parents[2] of this path
    sys.modules[name] = module
    try:
        exec(compile(shown.stdout, str(HARNESS), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def test_existing_keys_unchanged(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 10: on a run dir with timeouts, barge-ins, refusals and first uses, the report equals the pre-T-0101
    harness's (``T0101_BASE``, the oracle) plus only ``first_use_repair``/``first_use_repair_max_ms`` at the top and
    ``repair_warmup`` in the regime."""
    specs = [
        *first_use_specs(),
        s_repaired("hard-02", repair="timeout"),
        Spec(case="kout-02", category="K-out", gt=None, draft=D_F5),
        Spec(case="under-02", category="U-under", gt=GT_STORE_SEARCH_U, draft=D_NAMELESS),
        f_forced("click-01", repair="barge"),
        Spec(
            case="search-01",
            arm="F",
            category="google_search",
            gt={"name": "google_search", "query": "usb c cable"},
            draft=D_SEARCH,
            forced=True,
            repair="valid",
            repair_call=call("google_search", query="usb c cable"),
        ),
    ]
    run = with_cd(build_run(tmp_path, specs), specs)
    old = _harness_at(T0101_BASE).build_report(run)
    new = h.build_report(run)
    assert set(new) - set(old) == {"first_use_repair", "first_use_repair_max_ms"}, sorted(
        set(new) ^ set(old)
    )
    assert set(old) <= set(new)
    for key in old:
        if key == "regime":
            assert set(new["regime"]) - set(old["regime"]) == {"repair_warmup"}
            assert {k: v for k, v in new["regime"].items() if k != "repair_warmup"} == old["regime"]
        else:
            assert json.loads(json.dumps(new[key])) == json.loads(json.dumps(old[key])), key


def test_markdown_first_use_table(h: ModuleType, tmp_path: Path) -> None:
    """Criterion 7: report.md has a first-use section with one table row per first use (tool, arm, case, ms)."""
    specs = first_use_specs()
    run = set_warmup(with_cd(build_run(tmp_path, specs), specs), warm())
    md = h.render_markdown(h.build_report(run))
    assert "first use" in md.lower(), "criterion 7: no first-use section in report.md"
    rows = [line for line in md.splitlines() if line.startswith("|")]
    for tool, want in FIRST_USE_EXPECTED.items():
        hit = [
            r
            for r in rows
            if tool in r
            and str(want["case_id"]) in r
            and str(want["ms"]) in r
            and f" {want['arm']} " in r
        ]
        assert hit, f"criterion 7: no first-use table row for {tool} ({want})"
