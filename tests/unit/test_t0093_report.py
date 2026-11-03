"""T-0093: the live tool-call eval's report, on hand-built ``turns.jsonl`` and session logs (unit; no engine).

The report is what the hard gates of criterion 2 (and the amendment's added gates and void conditions, approved
unchanged at H-T93) are read from, and what criterion 3's quality lines are printed from. These rows build a run
directory by hand -- every number below is chosen here and its expected value computed by hand in the row -- and
run the report over it. The harness does not exist yet, so the rows are red on the named assertion in ``harness()``,
never on an ImportError at collection (the module is loaded lazily, by path, as ``test_cd_toolcall_cli.py`` does).

**The assumed surface** (``scripts/gpu/t0093_toolcall_live.py``; importing it opens no socket and starts nothing):

- ``build_report(run_dir: Path) -> dict``: reads the run directory below and returns the report.
- ``gates(report: dict) -> dict[str, bool]``: exactly the keys of ``GATES``; ``True`` means the gate holds.
- ``render_markdown(report: dict) -> str``: report.md's text.

**The run directory** (what the harness writes under ``logs/t0093-live-<sha>/``, criterion 1):

- ``turns.jsonl``: the session's own ``TurnRecord`` lines, unchanged (``local_astra/`` does not change, criterion 5).
- ``sessions/<name>.jsonl``: the session logs (``SessionLog``, one file per session).  ``TurnRecord.session_log`` is
  the absolute path of the turn's log; the rows write it as ``<run_dir>/sessions/<name>``.
- ``manifest.json``, the harness's sidecar (the session cannot record these, and ``local_astra/`` may not change):
  ``{"sessions": [{"arm", "session_id", "log", "startup_ms"}], "cases": [{"case_id", "kind", "arm", "session_id",
  "session_log", "turn_id", "gt", "barge_in", "prefix_hit_tokens", "prefix_query_tokens"}]}``.
  ``arm`` is ``"on"`` (``tools = dry_run``) or ``"off"``; ``kind`` is ``tool`` | ``no_tool`` | ``plain`` | ``q12``;
  ``log`` and ``session_log`` are the log's file name under ``sessions/``; ``gt`` is the eval case's ground truth
  (``tests/eval/toolcall/cases.jsonl``) or ``null``.  Turn ids repeat across sessions (``t1``, ``t2`` ... in each), so
  a turn is joined to its records by (session, turn id), never by turn id alone.
  T-0096 (H-T96, criterion 1) adds per case ``prefill_query_tokens`` (P), ``prefill_hit_tokens`` (Hp) and
  ``prefills_before_pass1``, and to the manifest ``regime`` the keys ``prefix_rule = "H-T96"``,
  ``prefix_block_tokens = 32`` and ``cache_config_block_size`` (the engine's label, reported only; the fixture writes
  ``"16"`` so that a gate reading A from it is caught).
- ``prefix_ids.jsonl`` (T-0096 criterion 1d): one line per (case, arm) with ``case_id, arm, session_id,
  prefill_request_id, generate_request_id, prefill_ids, generate_ids``. The fixture writes ``range(S)`` then disjoint
  ranges (the prefill's tail from 1_000_000, pass 1's suffix from 2_000_000), so the longest common prefix is S
  exactly and the lengths are P and Q. The session logs carry a ``llm.request`` record for the prefill and for pass 1
  with image digests in ``prompt.messages[].content[].url.digest`` (shape copied from a 4b0bd92 record).

**The report keys these rows read** (rates are ``{"k": int, "n": int}``, distributions ``{"n", "p50", "p90"}``):

- ``void``: a list of reasons, empty for a valid run;
- ``route_recall``: tools-on tool cases routed by pass 1's ``tool`` commit (J8); ``n`` is every tool case;
- ``draft_valid``, ``draft_exact``: over the tools-on tool cases that were not barged in (J5); ``barge_in["n"]``
  counts the barged-in tool cases, reported separately;
- ``query_problem``: ``k`` drafts with ``query_problem`` set, ``n`` the routed ``google_search`` cases (J5);
- ``spoken_prefix``: ``k`` routed tool cases whose spoken prefix is not blank, ``n`` the routed tool cases;
- ``false_tool``: tools-on ``no_tool`` and ``plain`` cases with any tool call or ``ToolExecute``;
- ``repairs``: ``{"n", "outcomes": {final status: count}, "ms": distribution of the tool.repair span}``;
- ``action_latency_ms``: ``tool.execute`` span start minus pass 1's ``llm.done`` span end (amendment);
- Q12 (J1, amended by T-0097 rulings K1/K2): the cases are the 7 of ``tests/eval/surface_placement_cases.jsonl``;
  ``expect_surface`` is read from there by ``case_id`` (the manifest also carries it). ``q12_shown`` (primary):
  ``{"on": rate, "off": rate}`` over the positive cases, a case counting when its own turn's log has a ``ui.gate``
  record with ``gated`` and ``surface_emitted`` both true; ``q12_all_ui_valid`` (secondary): the same over
  ``extras.ui_valid is True`` (the session's AND over the turn's gated ui commits); ``q12_negative_shown`` and
  ``q12_negative_all_ui_valid`` over the negative controls; ``q12_gates``, ``q12_cases`` and ``q12_inconsistent``
  (reported, never voided or gated). The old ``q12_surface`` / ``q12_negative`` keys are gone (T-0097 criterion 7);
- ``first_canvas_ms`` / ``llm_done_ms``: ``{"on": dist, "off": dist}`` of span end minus span start (relative to
  ``llm.start``, implementation note), and ``first_canvas_ms["ratio"]`` = p50 on / p50 off;
- ``prefix_hit``: ``{"on": float, "off": float}``, pooled Σhit/Σquery per arm over the whole final request (J2),
  reported and, since H-T96, not gated; ``startup_ms``: ``{"on": float, "off": float}``;
- T-0096 (criterion 8): ``prefix_rule``, ``prefix_block_tokens``, ``prefix_b11_cases`` (``{on: [...], off: [...]}``,
  one row per case) and ``prefix_b11`` (pooled ΣHs/ΣS per arm). The ``prefix_hit_on`` gate is B11 per tools-on case:
  S = LCP(prefill ids, generate ids), Hs = min(H, S), holds iff S > 0, Hs >= floor(S/32)*32 and Hs/S >= 9/10;
- ``stale_terminals``: ``{"dropped": int, "reached_machine": int}``.

Percentiles are nearest-rank (J9; p50 of an even count may be either middle value, so the rows keep p50 on odd
counts or on equal middle values). Rulings J0-J22 are in ``docs/tasks/T-0093.md``; J17 (human) replaces the argument-value rows with "the voiced
sources are a prefix of the spoken prefix", and the refusal rows call ``main`` in-process with every connect refused.

Regime: CPU, pure, hand-built data; no measurement. The ``gpu`` rows that read a real run are
``tests/eval/test_toolcall_live.py``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
import zlib
from collections.abc import Callable
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
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0093_toolcall_live.py"

#: Criterion 2 and the amendment, approved unchanged at H-T93, one key per gate.
GATES = frozenset(
    {
        "run_valid",  # amendment: every tools-on line reads TOOLS_ON exactly; repair off is invalid
        "no_literal_voiced",  # criterion 2: no special literal, ``call:`` or argument value in TtsSpeak
        "no_execute_on_interrupted",  # criterion 2: 0 ToolExecute on an interrupted turn
        "prefix_hit_on",  # criterion 2: prefix hit with tools on >= 0.9 (B11)
        "first_canvas_ratio",  # criterion 2: llm.first_canvas p50 on <= 1.10 x off
        "false_tool_rate",  # criterion 2: no-tool and plain cases <= 1/13
        "no_execute_before_drain",  # amendment: 0 ToolExecute before the turn's own playback.drained
        "no_stale_terminal_reached_machine",  # amendment: 0 stale-terminal notes that reached the machine
    }
)

TOOLS_ON = "native:bbfc53ccdd80:dry_run"
GIT = "64412895855e7fd89863554b8c27b9428de54716"
WALL = "2026-10-12T18:00:00+01:00"
STALE_TEXT = "turn one's tail. "
LOGS = {"on": "20261012-180000-101.jsonl", "off": "20261012-180000-102.jsonl"}
SESSION_IDS = {"on": "a0a0a0a0a0a0", "off": "b0b0b0b0b0b0"}
STARTUP_MS = {"on": 1650, "off": 1300}

# --- T-0096: the prefix measurement of a healthy case, per arm --------------------------------------------------
# Every hit is a multiple of 32 (criterion 5e). Tools on: S = 1220 = 38*32 + 4, H = 1216 = 38*32 -> Hs = 1216 >=
# floor(1220/32)*32 = 1216 and Hs/S = 1216/1220 >= 9/10: holds, missing exactly the 4-token partial block. The whole
# request (J2) reads 1216/1280 = 0.95. Tools off: S = 1580 = 49*32 + 12, H = 1568 = 49*32 -> holds; J2 1568/1600 =
# 0.98. The prefills: on Hp 1184 = 37*32 of P 1225 (tail P - S = 5); off Hp 1024 = 32*32 of P 1587 (tail 7).
ON_HQ = (1216, 1280)
ON_S = 1220
ON_PREFILL = (1184, 1225)
OFF_HQ = (1568, 1600)
OFF_S = 1580
OFF_PREFILL = (1024, 1587)
#: the engine's ``cache_config_info`` ``block_size`` label as the fixture writes it: reported only, never A (M5)
CACHE_CONFIG_BLOCK_SIZE = "16"
SIDECAR = "prefix_ids.jsonl"
SYSTEM_TEXT = "You are Astra, a local assistant that can see the user's screen."
PHASH = "b2dfede5393217a6"


def harness() -> ModuleType:
    """The harness, loaded by path. Red here, by name, until T-0093 criterion 1 lands."""
    assert HARNESS.is_file(), (
        f"T-0093 criterion 1: {HARNESS.relative_to(REPO)} does not exist yet (the report is built by it)"
    )
    name = "_t0093_toolcall_live_unit"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    for attr in ("build_report", "gates", "render_markdown"):
        assert callable(getattr(module, attr, None)), (
            f"T-0093 report surface: {HARNESS.name} has no callable {attr!r} (see this module's docstring)"
        )
    return module


# =====================================================================================================
# The hand-built run
# =====================================================================================================


@dataclass(frozen=True)
class Repair:
    ms: int
    #: ``{"name", "arguments"}`` of a valid repair; ``None`` when the repair is refused
    call: dict[str, Any] | None = None


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict[str, Any]
    schema_valid: bool
    query_problem: str | None = None
    stop_id: int | None = 50
    repair: Repair | None = None
    #: pass 1's ``llm.done`` span end -> the turn's drain (= the action latency when the call runs at the drain)
    gap: int = 400


@dataclass(frozen=True)
class Turn:
    case: str
    kind: str
    gt: dict[str, Any] | None = None
    #: ``(TtsSpeak.text, TtsSpeak.source)`` per sentence
    speech: tuple[tuple[str, str], ...] = ()
    call: Call | None = None
    barge_in: bool = False
    ui_valid: bool | None = None
    fc: int = 1000
    done: int = 1500
    #: pass 1's ``/metrics`` delta (H, Q): the whole final request (J2)
    hit: tuple[int | None, int | None] = ON_HQ
    #: T-0096: S, the shared prefix of the prefill's and pass 1's ids (``None``: no ``prefix_ids.jsonl`` line)
    shared: int | None = ON_S
    #: T-0096: the prefill's own ``/metrics`` delta (Hp, P)
    prefill: tuple[int | None, int | None] = ON_PREFILL
    #: T-0096: prefills started before pass 1's generate
    prefills: int = 1
    #: T-0096: the image digest on the prefill's and on pass 1's ``llm.request`` (``None``: one per case, both equal)
    digests: tuple[str, str] | None = None
    #: ``regime.tools`` override for this line
    tools: str | None = None
    #: ``auto`` (at the drain when the final call is valid), ``llm_done`` (early), ``after_drain`` (on the
    #: repair's done, after the drain), ``interrupted`` (in the barge-in step), ``after_stop`` (in the barge-in's
    #: ``playback.stopped`` step), ``bare`` (a ToolExecute in the drain step and no other tool evidence)
    execute: str = "auto"
    #: emit the turn's own ``playback.drained`` step
    drain: bool = True
    #: ``"stale_dropped"``: a stale-terminal note before the drain; ``"stale_reached"``: the note and the same
    #: terminal stepped through the machine on the next step; ``"stale_reached_later"``: the same with a tick step in
    #: between; ``"echo"``: no note here, but the terminal noted by an earlier turn is stepped at this turn's start
    stale: str | None = None
    #: J1: a Q12 case's ``expect_surface`` (``None`` for every other kind)
    expect_surface: bool | None = None
    #: J17: the ``spoken_prefix`` the ``tool.call`` records carry (default: the voiced sources, as the session does)
    spoken_prefix: str | None = None
    #: the line's ``outcome`` (default: interrupted / spoken / silent from the script)
    outcome: str | None = None
    #: ``state_before`` of pass 1's ``llm.done`` step (default: SPEAKING once voiced, else THINKING)
    done_state: str | None = None
    #: manifest fields over the harness's defaults (J19: ``failures``, ``timed_out``, ``prefill_done``, ...)
    manifest: tuple[tuple[str, Any], ...] = ()
    #: T-0097: ``(canvas_idx, gated, surface_emitted)`` per ``ui`` canvas, in order; one ``llm.commit`` (segment
    #: ``ui``) and one ``ui.gate`` record each, under this turn's id. ``ui_valid`` above stays independent, so a
    #: row can make the flag and the log disagree.
    ui_gates: tuple[tuple[int, bool, bool], ...] = ()
    #: T-0097: ``ui.gate`` records written in this turn's stretch of the log under *another* turn id (``None``:
    #: a record the session wrote with no turn open): ``(turn_id, canvas_idx, gated, surface_emitted)``
    stray_ui_gates: tuple[tuple[str | None, int, bool, bool], ...] = ()
    #: T-0097 criterion 5: ``extras.speech_refused`` (``None``: the key is absent from ``extras``) and the number
    #: of ``speech.refused`` records the turn's log carries
    speech_refused: int | None = 0
    refusal_records: int = 0


def tool_call_record(
    status: str,
    cid: str,
    source: str,
    name: str,
    arguments: dict[str, Any],
    *,
    schema_valid: bool | None,
    query_problem: str | None = None,
    repair_ms: int | None = None,
    action: str | None = None,
    reason: str | None = None,
    spoken_prefix: str = "",
) -> dict[str, Any]:
    """The ``tool.call`` record, keys as T-0092c criterion 7 (and the session's own records) carry them."""
    return {
        "status": status,
        "call_id": cid,
        "source": source,
        "name": name,
        "arguments": dict(arguments),
        "parse": "ok",
        "schema_valid": schema_valid,
        "query_problem": query_problem,
        "repair_ms": repair_ms,
        "action": action,
        "reason": reason,
        "resolved": {} if action else None,
        "spoken_prefix": spoken_prefix,
    }


def ui_gate_record(canvas_idx: int, gated: bool, emitted: bool, raw_len: int) -> dict[str, Any]:
    """A ``ui.gate`` record's ``data``, field set hand-copied from ``session.py`` (T-0097's fixture change).

    Gated (``_gate_surface``): ``canvas_idx, gated, ok, repairs, resamples, surface_emitted, surface_messages,
    problems, raw_len``; ``ok`` follows ``surface_emitted`` (the session's two read the same; the reviewer's
    ``ok``-for-``surface_emitted`` mutation is expected to survive). ``problems`` is ``[]`` as at 4b0bd92.
    Not gated (``_log_ui_not_gated``): the same plus ``reason``, with ``ok: False`` and ``surface_emitted:
    False`` -- except that ``emitted=True`` here writes the malformed ``gated: False, surface_emitted: True``
    record one row needs.
    """
    if gated:
        return {
            "canvas_idx": canvas_idx,
            "gated": True,
            "ok": emitted,
            "repairs": 0,
            "resamples": 0,
            "surface_emitted": emitted,
            "surface_messages": 2 if emitted else 0,
            "problems": [],
            "raw_len": raw_len,
        }
    return {
        "canvas_idx": canvas_idx,
        "gated": False,
        "reason": "no turn was open when the canvas arrived",
        "ok": False,
        "surface_emitted": emitted,
        "surface_messages": 0,
        "repairs": 0,
        "resamples": 0,
        "problems": [],
        "raw_len": raw_len,
    }


UI_TEXT = '{"surfaceUpdate": {"surfaceId": "main"}}'


def case_digest(case: str) -> str:
    """A case's own image digest (8 hex digits, as the session log writes it)."""
    return f"{zlib.crc32(case.encode()):08x}"


def image_part(digest: str) -> dict[str, Any]:
    """An image content part as the session log redacts it; the field set is copied from a 4b0bd92 ``llm.request``
    record (``{"type": "image_url", "url": {"redacted", "bytes", "digest", "phash", "mime"}, "phash"}``)."""
    return {
        "type": "image_url",
        "url": {
            "redacted": "image",
            "bytes": 91208,
            "digest": digest,
            "phash": PHASH,
            "mime": "image/jpeg",
        },
        "phash": PHASH,
    }


def request_prompt(user_content: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``prompt`` of a ``llm.request`` record: the system message and one user message (no history)."""
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
            {"role": "user", "content": user_content},
        ],
        "prefix_len": 2,
        "thinking": True,
        "assists": {"page_text": False, "pdf_text": False, "cells": False, "at_spi": False},
        "tools": None,
        "continue_final_message": False,
    }


def sidecar_ids(turn: Turn) -> tuple[list[int], list[int]]:
    """``(prefill_ids, generate_ids)``: ``range(S)``, then disjoint ranges of length P - S and Q - S."""
    s = turn.shared
    assert s is not None
    p = turn.prefill[1] if turn.prefill[1] is not None else s
    q = turn.hit[1] if turn.hit[1] is not None else s
    assert p >= s and q >= s, f"fixture: S {s} exceeds P {p} or Q {q} ({turn.case})"
    shared = list(range(s))
    return (
        shared + list(range(1_000_000, 1_000_000 + p - s)),
        shared + list(range(2_000_000, 2_000_000 + q - s)),
    )


def native(name: str, arguments: dict[str, Any]) -> str:
    """A native call's text as the tool commit carries it (the report never parses it; shape only)."""
    body = ",".join(
        f'{k}:<|"|>{v}<|"|>' if isinstance(v, str) else f"{k}:{json.dumps(v)}"
        for k, v in arguments.items()
    )
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


def step(
    t: int, event: E.Event | dict[str, Any], before: str, after: str, commands: list[Any]
) -> dict[str, Any]:
    ev = event if isinstance(event, dict) else event.model_dump(mode="json")
    return {
        "_t": t,
        "kind": "machine.step",
        "data": {
            "event": ev,
            "state_before": before,
            "state_after": after,
            "turn_seq_before": 1,
            "turn_seq_after": 1,
            "commands": [c if isinstance(c, dict) else c.model_dump(mode="json") for c in commands],
        },
    }


def rec(t: int, kind: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"_t": t, "kind": kind, "data": data}


def regime(tools: str) -> Regime:
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
        tools=tools,
    )


def turn_records(
    turn: Turn, tid: str, sid: str, s: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The session-log records of one turn (in the session's order) and its TurnRecord kwargs. ``s`` = llm.start."""
    gen, rep, cid = f"{tid}-gen", f"{tid}-repair", f"{tid}-call"
    call = turn.call
    out: list[dict[str, Any]] = []
    spans: dict[str, tuple[int, int]] = {
        "llm.first_canvas": (s, s + turn.fc),
        "llm.done": (s, s + turn.done),
    }
    spoken = "".join(src for text, src in turn.speech if text)
    prefix = turn.spoken_prefix if turn.spoken_prefix is not None else spoken
    out.append(
        step(
            s,
            E.UserText(t_ms=s, text=turn.case),
            "IDLE",
            "THINKING",
            [
                C.LlmStart(request_id=gen, transcript=turn.case),
                C.HudState(state="THINKING", turn_id=tid),
            ],
        )
    )
    if turn.stale == "echo":
        out.append(
            step(
                s,
                E.PlaybackDrained(t_ms=s, played_ms=777, played_text=STALE_TEXT),
                "THINKING",
                "THINKING",
                [],
            )
        )
    # T-0096: the anticipatory prefill's request, written before the turn's first step (as at 4b0bd92), and pass 1's,
    # each with the case's image digest (criterion 5d compares them over the shared messages)
    d_prefill, d_gen = turn.digests or (case_digest(turn.case),) * 2
    out.insert(
        0,
        rec(
            s,
            "llm.request",
            {
                "phase": "prefill",
                "request_id": f"{tid}-prefill",
                "attempt_id": f"{sid}-{tid}-prefill-1",
                "options": None,
                "xargs": None,
                "prompt": request_prompt([image_part(d_prefill)]),
                "drop_history_turns": 0,
                "prompt_tokens": 0,
            },
        ),
    )
    out.append(
        rec(
            s,
            "llm.request",
            {
                "phase": "generate",
                "request_id": gen,
                "attempt_id": f"{sid}-{gen}-2",
                "options": {"top_k": None, "top_p": None, "max_tokens": None, "stop": []},
                "xargs": None,
                "prompt": request_prompt(
                    [
                        image_part(d_gen),
                        {"type": "text", "text": "[screen] The user's screen is attached."},
                        {"type": "text", "text": turn.case},
                    ]
                ),
                "drop_history_turns": 0,
                "prompt_tokens": 0,
            },
        )
    )
    t_fc = s + turn.fc
    if turn.speech:
        out.append(
            rec(
                t_fc,
                "llm.commit",
                {"request_id": gen, "canvas_idx": 0, "segment": "answer", "text": spoken},
            )
        )
        speaks = [
            C.TtsSpeak(utterance_id=f"{tid}-u{i + 1}", text=text, source=src)
            for i, (text, src) in enumerate(turn.speech)
        ]
        out.append(
            step(
                t_fc,
                E.LlmCommit(t_ms=t_fc, request_id=gen, canvas_idx=0, segment="answer", text=spoken),
                "THINKING",
                "SPEAKING",
                [C.StopClip(), *speaks, C.HudState(state="SPEAKING", turn_id=tid)],
            )
        )
        for sp in [sp for sp in speaks if sp.text]:  # the session logs no utterance for ``text=""``
            out.append(
                rec(
                    t_fc,
                    "tts.utterance",
                    {
                        "utterance_id": sp.utterance_id,
                        "text": sp.text,
                        "source": sp.source,
                        "kind": "speech",
                    },
                )
            )
    else:
        out.append(
            rec(
                t_fc,
                "llm.commit",
                {"request_id": gen, "canvas_idx": 0, "segment": "thought", "text": "hmm"},
            )
        )
        out.append(
            step(
                t_fc,
                E.LlmCommit(t_ms=t_fc, request_id=gen, canvas_idx=0, segment="thought", text="hmm"),
                "THINKING",
                "THINKING",
                [],
            )
        )
    # T-0097: each ``ui`` canvas is committed and then gated (or logged as not gated), in canvas order
    for canvas_idx, gated, emitted in turn.ui_gates:
        out.append(
            rec(
                t_fc,
                "llm.commit",
                {"request_id": gen, "canvas_idx": canvas_idx, "segment": "ui", "text": UI_TEXT},
            )
        )
        out.append(rec(t_fc, "ui.gate", ui_gate_record(canvas_idx, gated, emitted, len(UI_TEXT))))
    for other, canvas_idx, gated, emitted in turn.stray_ui_gates:
        stray = rec(t_fc, "ui.gate", ui_gate_record(canvas_idx, gated, emitted, len(UI_TEXT)))
        stray["turn_id"] = other
        out.append(stray)
    for n in range(1, turn.refusal_records + 1):
        out.append(rec(t_fc, "speech.refused", {"count": n, "sources": ["<|"], "truncated": False}))
    state = "SPEAKING" if turn.speech else "THINKING"
    t_done = s + turn.done
    final: dict[str, Any] | None = None
    exec_call: tuple[str, dict[str, Any], str] | None = (
        None  # name, arguments, source of the executed call
    )
    draft: ToolDraft | None = None
    if call is not None:
        out.append(
            rec(
                t_done,
                "llm.commit",
                {
                    "request_id": gen,
                    "canvas_idx": 1,
                    "segment": "tool",
                    "text": native(call.name, call.arguments),
                    "stop_id": call.stop_id,
                },
            )
        )
        status = (
            "valid"
            if call.schema_valid
            else ("repairable" if call.repair is not None else "refused")
        )
        draft = ToolDraft(
            call_id=cid,
            status=status,
            name=call.name,
            arguments=call.arguments,
            reason=None if call.schema_valid else "schema invalid",
            source="draft",
        )
        if call.schema_valid:
            exec_call = (call.name, call.arguments, "draft")
        elif call.repair is not None and call.repair.call is not None:
            exec_call = (call.repair.call["name"], call.repair.call["arguments"], "repair")
    out.append(rec(t_done, "llm.done", {"request_id": gen, "commits": 2 if call else 1}))

    def execute_cmd(name: str, arguments: dict[str, Any], source: str) -> C.ToolExecute:
        return C.ToolExecute(
            call_id=cid, turn_id=tid, name=name, arguments=arguments, source=source
        )

    done_cmds: list[Any] = [C.CancelTimer(name="llm_timeout")]
    if call is not None and call.repair is not None:
        done_cmds.append(C.ToolRepairStart(request_id=rep, draft_request_id=gen, call_id=cid))
    if turn.execute == "llm_done":
        assert call is not None
        done_cmds.append(execute_cmd(call.name, call.arguments, "draft"))
        spans["tool.execute"] = (t_done, t_done + 2)
    out.append(
        step(
            t_done,
            E.LlmDone(t_ms=t_done, request_id=gen, tool=draft),
            turn.done_state or state,
            state,
            done_cmds,
        )
    )
    if call is not None:
        out.append(
            rec(
                t_done,
                "tool.call",
                tool_call_record(
                    "proposed",
                    cid,
                    "draft",
                    call.name,
                    call.arguments,
                    schema_valid=call.schema_valid,
                    query_problem=call.query_problem,
                    spoken_prefix=prefix,
                ),
            )
        )
    gap = call.gap if call is not None else 300
    t_drain = s + turn.done + gap
    repair_tail: list[dict[str, Any]] = []
    if call is not None and call.repair is not None:
        r = call.repair
        t_rep = t_drain + 50 if turn.execute == "after_drain" else t_done + r.ms
        repair_ms = t_rep - t_done
        spans["tool.repair"] = (t_done, t_rep)
        out.append(
            rec(
                t_done,
                "tool.call",
                tool_call_record(
                    "repairing",
                    cid,
                    "draft",
                    call.name,
                    call.arguments,
                    schema_valid=False,
                    spoken_prefix=prefix,
                ),
            )
        )
        out.append(rec(t_done, "llm.request", {"phase": "tool_repair", "request_id": rep}))
        rtext = native(r.call["name"], r.call["arguments"]) if r.call else "<|tool_call>call:"
        out.append(
            rec(
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
        out.append(rec(t_rep, "llm.done", {"request_id": rep, "commits": 1}))
        rdraft = (
            ToolDraft(
                call_id=cid,
                status="valid",
                name=r.call["name"],
                arguments=r.call["arguments"],
                source="repair",
            )
            if r.call
            else ToolDraft(call_id=cid, status="refused", reason="schema invalid", source="repair")
        )
        rcmds: list[Any] = [C.CancelTimer(name="tool_repair")]
        if r.call is None:
            rcmds.append(C.ToolCancel(call_id=cid, reason="refused"))
        if turn.execute == "after_drain":
            assert r.call is not None
            rcmds.append(execute_cmd(r.call["name"], r.call["arguments"], "repair"))
            spans["tool.execute"] = (t_rep, t_rep + 2)
        rstep = step(t_rep, E.LlmDone(t_ms=t_rep, request_id=rep, tool=rdraft), state, state, rcmds)
        rfinal: list[dict[str, Any]] = []
        if r.call is None:
            final = tool_call_record(
                "refused",
                cid,
                "repair",
                call.name,
                call.arguments,
                schema_valid=False,
                repair_ms=repair_ms,
                reason="schema invalid",
                spoken_prefix=prefix,
            )
            rfinal.append(rec(t_rep, "tool.call", final))
        elif turn.execute == "after_drain":
            final = tool_call_record(
                "dry_run",
                cid,
                "repair",
                r.call["name"],
                r.call["arguments"],
                schema_valid=True,
                repair_ms=repair_ms,
                action=f"recorded {r.call['name']}",
                spoken_prefix=prefix,
            )
            rfinal.append(rec(t_rep + 2, "tool.call", final))
        if turn.execute == "after_drain":
            repair_tail = [rstep, *rfinal]
        else:
            out.extend([rstep, *rfinal])
        repair_ms_final: int | None = repair_ms
    else:
        repair_ms_final = None

    if turn.stale is not None and turn.stale != "echo":
        # a dropped note carries the shared text (the "echo" turn re-steps it); a reached one has its own, so that
        # under J7 it cannot also match an earlier turn's note
        stale_text = STALE_TEXT if turn.stale == "stale_dropped" else f"the tail before {tid}. "
        out.append(
            rec(
                t_drain - 20,
                "note",
                {
                    "what": "stale playback terminal",
                    "run_id": 1,
                    "run_turn": "t0",
                    "current_turn": tid,
                    "type": "playback.drained",
                    "played_ms": 777,
                    "played_text": stale_text,
                },
            )
        )
        if turn.stale == "stale_reached_later":
            out.append(step(t_drain - 15, E.Tick(t_ms=t_drain - 15), state, state, []))
        if turn.stale in ("stale_reached", "stale_reached_later"):
            out.append(
                step(
                    t_drain - 20,
                    E.PlaybackDrained(t_ms=t_drain - 20, played_ms=777, played_text=stale_text),
                    state,
                    state,
                    [],
                )
            )

    if turn.barge_in:
        t_cut = s + turn.done + 200
        cut_cmds: list[Any] = [C.PlaybackFlush()]
        if call is not None:
            cut_cmds.append(C.ToolCancel(call_id=cid, reason="interrupted"))
        if turn.execute == "interrupted":
            assert call is not None
            cut_cmds.append(execute_cmd(call.name, call.arguments, "draft"))
            spans["tool.execute"] = (t_cut, t_cut + 2)
        out.append(
            step(
                t_cut,
                E.UserTakeover(t_ms=t_cut, onset_ms=t_cut),
                "SPEAKING",
                "INTERRUPTED",
                cut_cmds,
            )
        )
        if call is not None:
            final = tool_call_record(
                "cancelled",
                cid,
                "draft",
                call.name,
                call.arguments,
                schema_valid=call.schema_valid,
                reason="interrupted",
                spoken_prefix=prefix,
            )
            out.append(rec(t_cut, "tool.call", final))
        out.append(
            step(
                t_cut + 100,
                E.PlaybackStopped(t_ms=t_cut + 100, played_ms=300, played_text=spoken[:5]),
                "INTERRUPTED",
                "IDLE",
                [
                    *(
                        [execute_cmd(call.name, call.arguments, "draft")]
                        if turn.execute == "after_stop" and call is not None
                        else []
                    ),
                    C.HudState(state="IDLE", turn_id=tid),
                ],
            )
        )
        spans["playback.stopped"] = (t_cut + 100, t_cut + 100)
    elif turn.drain:
        drain_cmds: list[Any] = []
        at_drain = turn.execute == "auto" and exec_call is not None
        if at_drain:
            assert exec_call is not None
            drain_cmds.append(execute_cmd(*exec_call))
            spans["tool.execute"] = (t_drain, t_drain + 2)
        if turn.execute == "bare":
            drain_cmds.append(
                C.ToolExecute(
                    call_id=cid,
                    turn_id=tid,
                    name="click_at",
                    arguments={"x": 1, "y": 1},
                    source="draft",
                )
            )
        drain_cmds += [C.CommitHistory(user_only=False), C.HudState(state="IDLE", turn_id=tid)]
        out.append(
            step(
                t_drain,
                E.PlaybackDrained(t_ms=t_drain, played_ms=100 * len(spoken), played_text=spoken),
                state,
                "IDLE",
                drain_cmds,
            )
        )
        spans["playback.stopped"] = (t_drain, t_drain)
        if at_drain:
            assert exec_call is not None and call is not None
            name, arguments, source = exec_call
            final = tool_call_record(
                "dry_run",
                cid,
                source,
                name,
                arguments,
                schema_valid=True,
                query_problem=call.query_problem if source == "draft" else None,
                repair_ms=repair_ms_final,
                action=f"recorded {name}",
                spoken_prefix=prefix,
            )
            out.append(rec(t_drain + 2, "tool.call", final))
        out.extend(repair_tail)
    if turn.execute == "llm_done" and call is not None:
        final = tool_call_record(
            "dry_run",
            cid,
            "draft",
            call.name,
            call.arguments,
            schema_valid=call.schema_valid,
            action=f"recorded {call.name}",
            spoken_prefix=prefix,
        )
        out.append(rec(t_done + 2, "tool.call", final))
    outcome = turn.outcome or (
        "interrupted" if turn.barge_in else ("spoken" if turn.speech else "silent")
    )
    kwargs = {
        "spans": spans,
        "outcome": outcome,
        "tool": final,
        "extras": {
            "ui_valid": turn.ui_valid,
            **({} if turn.speech_refused is None else {"speech_refused": turn.speech_refused}),
        },
    }
    return out, kwargs


@dataclass
class Case:
    on: Turn
    off: Turn


def plain_off(turn: Turn) -> Turn:
    """The tools-off arm of a case: the same question, no call, ordinary speech."""
    speech = turn.speech or (("I cannot do that from here.", "I cannot do that from here. "),)
    return replace(
        turn, call=None, speech=speech, barge_in=False, execute="auto", stale=None, tools=None
    )


def build_run(
    root: Path,
    cases: list[Case],
    *,
    on_tools: str = TOOLS_ON,
    repair_off_note: bool = False,
    h_t96: bool = True,
) -> Path:
    """Write ``turns.jsonl``, ``sessions/*.jsonl`` and ``manifest.json`` for ``cases`` under ``root``; arms alternate.

    ``h_t96`` (T-0096): write ``prefix_ids.jsonl``, the per-case prefill fields and the regime's ``prefix_rule``;
    ``False`` writes a run shaped as 4b0bd92 was (none of them)."""
    run = root / f"t0093-live-{GIT[:7]}"
    (run / "sessions").mkdir(parents=True)
    logs: dict[str, list[dict[str, Any]]] = {"on": [], "off": []}
    clock = {"on": 1000, "off": 1000}
    nth = {"on": 0, "off": 0}
    lines: list[str] = []
    manifest_cases: list[dict[str, Any]] = []
    sidecar: list[str] = []
    if repair_off_note:
        logs["on"].append(
            rec(0, "note", {"what": "tool repair off", "reason": "model_revision mismatch"})
        )
    for i, case in enumerate(cases):
        order = ("on", "off") if i % 2 == 0 else ("off", "on")
        for arm in order:
            turn = case.on if arm == "on" else case.off
            nth[arm] += 1
            tid = f"t{nth[arm]}"
            sid = SESSION_IDS[arm]
            s = clock[arm]
            records, kw = turn_records(turn, tid, sid, s)
            clock[arm] = s + 20_000
            tools = turn.tools if turn.tools is not None else (on_tools if arm == "on" else "off")
            record = TurnRecord(
                turn_id=tid,
                session_id=sid,
                regime=regime(tools),
                input="text",
                canvases=2,
                session_log=str(run / "sessions" / LOGS[arm]),
                **kw,
            )
            records.append(rec(s + 19_000, "turn.record", record.model_dump(mode="json")))
            for r in records:
                r.setdefault("turn_id", tid)  # a stray ``ui.gate`` keeps the id it was written with
            logs[arm].extend(records)
            lines.append(record.to_line())
            manifest_cases.append(
                {
                    "case_id": turn.case,
                    "kind": turn.kind,
                    "arm": arm,
                    "session_id": sid,
                    "session_log": LOGS[arm],
                    "turn_id": tid,
                    "gt": turn.gt,
                    "barge_in": turn.barge_in,
                    "expect_surface": turn.expect_surface,
                    "prefix_hit_tokens": turn.hit[0],
                    "prefix_query_tokens": turn.hit[1],
                    # the harness's own per-case fields, at their healthy values (J19 voids on the others)
                    "barge_in_attempted": turn.barge_in,
                    "prefill_done": True,
                    "max_inflight": 1,
                    "engine_idle_before": True,
                    "timed_out": False,
                    "failures": [],
                    **(
                        {
                            "prefill_hit_tokens": turn.prefill[0],
                            "prefill_query_tokens": turn.prefill[1],
                            "prefills_before_pass1": turn.prefills,
                        }
                        if h_t96
                        else {}
                    ),
                    **dict(turn.manifest),
                }
            )
            if h_t96 and turn.shared is not None:
                prefill_ids, generate_ids = sidecar_ids(turn)
                sidecar.append(
                    json.dumps(
                        {
                            "case_id": turn.case,
                            "arm": arm,
                            "session_id": sid,
                            "prefill_request_id": f"{tid}-prefill",
                            "generate_request_id": f"{tid}-gen",
                            "prefill_ids": prefill_ids,
                            "generate_ids": generate_ids,
                        },
                        separators=(",", ":"),
                    )
                )
    for arm, records in logs.items():
        path = run / "sessions" / LOGS[arm]
        body = [
            {
                "seq": 0,
                "t_ms": 0,
                "wall": WALL,
                "session_ms": None,
                "kind": "log.started",
                "turn_id": None,
                "data": {"pid": 1, "path": str(path)},
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
                    "turn_id": r.get("turn_id"),
                    "data": r["data"],
                }
            )
        path.write_text("".join(json.dumps(b, separators=(",", ":")) + "\n" for b in body))
    (run / "turns.jsonl").write_text("".join(line + "\n" for line in lines))
    regime_: dict[str, Any] = {"commit": GIT, "profile": "lean", "tools": TOOLS_ON}
    if h_t96:
        regime_ |= {
            "prefix_rule": "H-T96",
            "prefix_block_tokens": 32,
            "cache_config_block_size": CACHE_CONFIG_BLOCK_SIZE,
        }
        (run / SIDECAR).write_text("".join(line + "\n" for line in sidecar))
    manifest = {
        "regime": regime_,
        "sessions": [
            {
                "arm": arm,
                "session_id": SESSION_IDS[arm],
                "log": LOGS[arm],
                "startup_ms": STARTUP_MS[arm],
            }
            for arm in ("on", "off")
        ],
        "cases": manifest_cases,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return run


# --- the reference run: 11 tool + 8 no-tool + 5 plain + 3 Q12 cases, each in both arms --------------------------

GT = {
    "click-01": {"bbox": [358, 477, 642, 521], "name": "click_at"},
    "click-02": {"bbox": [541, 437, 642, 458], "name": "click_at"},
    "search-01": {"keywords": ["weather", "lisbon"], "name": "google_search"},
    "search-02": {"keywords": ["british museum"], "name": "google_search"},
    "type-01": {
        "bbox": [358, 279, 642, 326],
        "name": "type_at",
        "press_enter": False,
        "text": "anna.berg@northwind.example",
    },
    "scroll-01": {"direction": "down", "name": "scroll"},
    "key-01": {"combo": "Ctrl+Z", "name": "press_key"},
    "key-02": {"combo": "Ctrl+C", "name": "press_key"},
    "key-03": {"combo": "Tab", "name": "press_key"},
    "click-03": {"bbox": [740, 358, 839, 402], "name": "click_at"},
    "type-02": {
        "bbox": [297, 13, 703, 59],
        "name": "type_at",
        "press_enter": True,
        "text": "usb-c hub",
    },
    "scroll-02": {"direction": "up", "name": "scroll"},
}

MATHS = ("Since 17 < 20 and 23 > 20, it is 391.", "Since 17 < 20 and 23 > 20, it is 391. ")


def say(*sentences: str) -> tuple[tuple[str, str], ...]:
    return tuple((s, s + " ") for s in sentences)


def tool_turns() -> list[Turn]:
    """Eleven tools-on tool cases (five barged in, J18); the expected metrics are worked out in the rows."""
    return [
        # 1. a valid click inside the bbox: valid, exact, spoken, executed at the drain 400 ms after llm.done
        Turn(
            "click-01",
            "tool",
            GT["click-01"],
            say("I will click Sign in."),
            Call("click_at", {"x": 500, "y": 499}, True, gap=400),
        ),
        # 2. google_search with no spoken sentence: valid, exact, executed 100 ms after llm.done
        Turn(
            "search-01",
            "tool",
            GT["search-01"],
            (),
            Call("google_search", {"query": "weather lisbon"}, True, gap=100),
        ),
        # 3. schema-valid but narrated query: valid, query_problem set, therefore not exact (score.exact)
        Turn(
            "search-02",
            "tool",
            GT["search-02"],
            say("Let me look that up."),
            Call(
                "google_search",
                {"query": "let me search british museum"},
                True,
                query_problem="narration",
                gap=400,
            ),
        ),
        # 4. press_enter missing: repairable; repair (180 ms) valid; executed at the drain 400 ms after llm.done
        Turn(
            "type-01",
            "tool",
            GT["type-01"],
            say("Typing the address."),
            Call(
                "type_at",
                {"x": 500, "y": 302, "text": "anna.berg@northwind.example"},
                False,
                repair=Repair(
                    180,
                    {
                        "name": "type_at",
                        "arguments": {
                            "x": 500,
                            "y": 302,
                            "text": "anna.berg@northwind.example",
                            "press_enter": False,
                        },
                    },
                ),
                gap=400,
            ),
        ),
        # 5. a direction outside the enum: repairable; repair (220 ms) refused
        Turn(
            "scroll-01",
            "tool",
            GT["scroll-01"],
            say("Scrolling now."),
            Call("scroll", {"direction": "sideways"}, False, repair=Repair(220)),
        ),
        # 6. a combo outside the enum, cut by length (stop_id None) -- still a routed call; repair (220 ms) refused
        Turn(
            "key-01",
            "tool",
            GT["key-01"],
            (),
            Call("press_key", {"combo": "Control+Z"}, False, stop_id=None, repair=Repair(220)),
        ),
        # 7. a valid click, barge-in during the spoken prefix: cancelled, never executed
        Turn(
            "click-02",
            "tool",
            GT["click-02"],
            say("Opening the link."),
            Call("click_at", {"x": 591, "y": 447}, True),
            barge_in=True,
        ),
        # 8-11. J18: four more barge-ins on routed tool turns (five in all), each a valid, exact draft
        Turn(
            "click-03",
            "tool",
            GT["click-03"],
            say("Opening that menu."),
            Call("click_at", {"x": 789, "y": 380}, True),
            barge_in=True,
        ),
        Turn(
            "type-02",
            "tool",
            GT["type-02"],
            say("Searching the shop."),
            Call("type_at", {"x": 500, "y": 36, "text": "usb-c hub", "press_enter": True}, True),
            barge_in=True,
        ),
        Turn(
            "scroll-02",
            "tool",
            GT["scroll-02"],
            say("Going back up."),
            Call("scroll", {"direction": "up"}, True),
            barge_in=True,
        ),
        Turn(
            "key-02",
            "tool",
            GT["key-02"],
            say("Copying it."),
            Call("press_key", {"combo": "Ctrl+C"}, True),
            barge_in=True,
        ),
    ]


#: T-0096: a healthy case's prefix fields per arm (see ``ON_HQ`` / ``OFF_HQ``)
ON_PREFIX: dict[str, Any] = {"hit": ON_HQ, "shared": ON_S, "prefill": ON_PREFILL}
OFF_PREFIX: dict[str, Any] = {"hit": OFF_HQ, "shared": OFF_S, "prefill": OFF_PREFILL}


def reference_cases() -> list[Case]:
    turns: list[Turn] = tool_turns()
    chats = [
        say("You are welcome."),
        (MATHS,),  # guard width: a maths answer with < and > is ordinary speech
        say("You can call me any time."),  # guard width: ``call`` without the colon
        *[say(f"Here is answer {n}.") for n in range(4, 9)],
    ]
    for n, speech in enumerate(chats, start=1):
        turns.append(
            Turn(f"chat-0{n}", "no_tool", None, speech, stale="stale_dropped" if n == 4 else None)
        )
    for n in range(1, 6):
        turns.append(Turn(f"plain-0{n}", "plain", None, say(f"Plain answer {n}.")))
    # J1: two positives and one negative control that over-triggers with tools on. T-0097: with tools on,
    # flight_booking shows a surface at canvas 1 and a later canvas fails the gate (shown, not every ui commit
    # valid: the session's AND reads False); primes_table's one ui canvas was never gated (not shown, ui_valid
    # None); capital_of_france shows one surface. The tools-off arm is set below.
    for name, ok, expect, gates in (
        ("flight_booking", False, True, ((1, True, True), (2, True, False))),
        ("primes_table", None, True, ((0, False, False),)),
        ("capital_of_france", True, False, ((1, True, True),)),
    ):
        turns.append(
            Turn(
                name,
                "q12",
                None,
                say("Here it is."),
                ui_valid=ok,
                expect_surface=expect,
                ui_gates=gates,
            )
        )
    assert len(turns) == 27
    cases: list[Case] = []
    for i, t in enumerate(turns):
        # first canvas: 14 cases at 1050 (on) / 1000 (off), 13 at 4000 / 900 -> medians 1050 and 1000 (the means
        # would be 2470.4 and 951.9); llm.done is 500 ms after the first canvas in both arms
        on_fc, off_fc = (1050, 1000) if i < 14 else (4000, 900)
        on = replace(t, fc=on_fc, done=on_fc + 500, **ON_PREFIX)
        off = replace(
            plain_off(t),
            fc=off_fc,
            done=off_fc + 500,
            **OFF_PREFIX,
            ui_valid=True if t.expect_surface else None,
            # tools off: one surface at canvas 1 on each positive, no ui canvas on the negative
            ui_gates=((1, True, True),) if t.expect_surface else (),
        )
        cases.append(Case(on, off))
    return cases


def edit(cases: list[Case], case: str, fn: Callable[[Case], Case]) -> list[Case]:
    hit = [c for c in cases if c.on.case == case]
    assert len(hit) == 1, f"fixture: no single case {case!r}"
    return [fn(c) if c.on.case == case else c for c in cases]


def on(case: str, **changes: Any) -> Callable[[Case], Case]:
    return lambda c: Case(replace(c.on, **changes), c.off)


def off(case: str, **changes: Any) -> Callable[[Case], Case]:
    return lambda c: Case(c.on, replace(c.off, **changes))


def report_of(tmp_path: Path, cases: list[Case] | None = None, **kw: Any) -> dict[str, Any]:
    run = build_run(tmp_path, reference_cases() if cases is None else cases, **kw)
    report = harness().build_report(run)
    assert isinstance(report, dict), f"build_report returns a dict; got {type(report).__name__}"
    return report


def gates_of(report: dict[str, Any]) -> dict[str, bool]:
    g = harness().gates(copy.deepcopy(report))
    assert isinstance(g, dict), f"gates returns a dict; got {type(g).__name__}"
    missing = GATES - set(g)
    assert not missing, f"T-0093 H-T93: gates dropped {sorted(missing)}"
    return g


def only_failing(g: dict[str, bool]) -> list[str]:
    return sorted(k for k, v in g.items() if v is not True)


def key(report: dict[str, Any], name: str) -> Any:
    """A report key, red on a named assertion (not a KeyError) while T-0097 is not implemented."""
    assert name in report, f"T-0097: the report has no {name!r} key"
    return report[name]


#: J1's 7 Q12 cases and their ``expect_surface``, hand-copied from ``tests/eval/surface_placement_cases.jsonl``
#: (the harness reads that file by ``case_id``; a manifest value that disagrees voids the run)
Q12_EXPECT = {
    "flight_booking": True,
    "flight_booking_with_memory": True,
    "primes_table": True,
    "omelette_steps": True,
    "capital_of_france": False,
    "spider_legs": False,
    "cloud_joke": False,
}

Gates = tuple[tuple[int, bool, bool], ...]


def q12_case(
    name: str,
    *,
    on_gates: Gates = (),
    on_valid: bool | None = None,
    off_gates: Gates = (),
    off_valid: bool | None = None,
    on_extra: tuple[tuple[str, Any], ...] = (),
    off_extra: tuple[tuple[str, Any], ...] = (),
) -> Case:
    """One Q12 case in both arms, at the reference run's latencies and prefix hits."""
    base = Turn(name, "q12", None, say("Here it is."), expect_surface=Q12_EXPECT[name])
    return Case(
        replace(
            base,
            fc=1050,
            done=1550,
            **ON_PREFIX,
            ui_gates=on_gates,
            ui_valid=on_valid,
            **dict(on_extra),
        ),
        replace(
            base,
            fc=1000,
            done=1500,
            **OFF_PREFIX,
            ui_gates=off_gates,
            ui_valid=off_valid,
            **dict(off_extra),
        ),
    )


def with_q12(*q12: Case) -> list[Case]:
    """The reference run's 24 tool / no-tool / plain cases, then exactly these Q12 cases (in this order).

    The run stays valid and inside every gate: the 24 keep their 14 x 1050 / 10 x 4000 first canvases on (14 x 1000
    / 10 x 900 off), and each Q12 case adds 1050 on and 1000 off, so p50 is 1050 on and 1000 off (ratio 1.05) for
    any number of Q12 cases; the five barge-ins and the 13 no-tool / plain cases are untouched."""
    return [c for c in reference_cases() if c.on.kind != "q12"] + list(q12)


def q12_row(report: dict[str, Any], case_id: str, arm: str) -> dict[str, Any]:
    rows = [
        r for r in key(report, "q12_cases") if r.get("case_id") == case_id and r.get("arm") == arm
    ]
    assert len(rows) == 1, (
        f"criterion 5: one q12_cases row per (case, arm); {case_id}/{arm}: {rows}"
    )
    return rows[0]


def counts(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """``(gated, emitted, failed, not_gated)`` of a q12_cases row or a q12_gates cell."""
    return (row.get("gated"), row.get("emitted"), row.get("failed"), row.get("not_gated"))


def pairs(entries: Any) -> set[tuple[str, str]]:
    """``q12_inconsistent``'s (case, arm) pairs: criterion 6 names the pair, not its shape, so a ``{case_id, arm}``
    mapping or a two-item sequence are both read."""
    assert isinstance(entries, list), f"criterion 6: q12_inconsistent is a list; got {entries!r}"
    out: set[tuple[str, str]] = set()
    for e in entries:
        if isinstance(e, dict):
            out.add((str(e.get("case_id")), str(e.get("arm"))))
        else:
            assert isinstance(e, list | tuple) and len(e) >= 2, f"criterion 6: entry {e!r}"
            out.add((str(e[0]), str(e[1])))
    return out


# =====================================================================================================
# The surface and the reference run
# =====================================================================================================


def test_the_gates_are_exactly_the_approved_set(tmp_path: Path) -> None:
    """H-T93: criterion 2's five gates plus the amendment's two and its void conditions -- no fewer, no others."""
    g = harness().gates(report_of(tmp_path))
    assert set(g) == GATES, (
        f"gates: extra {sorted(set(g) - GATES)}, missing {sorted(GATES - set(g))}"
    )
    assert all(isinstance(v, bool) for v in g.values()), g


def test_the_reference_run_is_valid_and_passes_every_gate(tmp_path: Path) -> None:
    """The reference run is built to pass everything; a gate that fails here is too wide."""
    report = report_of(tmp_path)
    assert report["void"] == [], (
        f"amendment: a valid run has no void reason; got {report['void']!r}"
    )
    g = gates_of(report)
    assert only_failing(g) == [], f"the reference run fails {only_failing(g)}"


# =====================================================================================================
# Criterion 3 and the amendment: the reported metrics, computed by hand
# =====================================================================================================


def test_route_recall_draft_valid_exact_and_query_problem(tmp_path: Path) -> None:
    """Criterion 3. Eleven tools-on tool cases, all routed (key-01's call was cut by length, ``stop_id`` None, and is
    still a call: routing is read from the tool commit and its ids, not from how the stream stopped).
    J5: draft valid and exact leave out the five barged-in cases (click-02, click-03, type-02, scroll-02, key-02;
    reported on their own), so n = 6. Valid drafts:
    click-01, search-01, search-02 = 3. Exact (score.exact against the case's gt): click-01 (500, 499) in
    [358, 477, 642, 521]; search-01 has both keywords = 2. search-02 has its keywords but a narrated query, so it is
    not exact; it is the one ``query_problem`` of the 2 routed google_search cases."""
    r = report_of(tmp_path)
    assert r["route_recall"] == {"k": 11, "n": 11}, r["route_recall"]
    assert r["draft_valid"] == {"k": 3, "n": 6}, r["draft_valid"]
    assert r["draft_exact"] == {"k": 2, "n": 6}, r["draft_exact"]
    assert r["query_problem"] == {"k": 1, "n": 2}, r["query_problem"]
    assert r["barge_in"]["n"] == 5, r["barge_in"]


def test_an_unrouted_tool_case_lowers_recall_and_stays_in_the_valid_and_exact_denominators(
    tmp_path: Path,
) -> None:
    """Criterion 3, beside the eval's 40/40 (over the tool cases): key-03 answered in speech with no call is a miss
    for recall and counts in ``n`` for draft valid and exact (J5: 6 + 1 not barged in)."""
    cases = reference_cases()
    extra = Turn("key-03", "tool", GT["key-03"], say("I pressed nothing."), fc=1050, done=1550)
    cases.insert(11, Case(extra, plain_off(replace(extra, fc=1000, done=1500, **OFF_PREFIX))))
    r = report_of(tmp_path, cases)
    assert r["route_recall"] == {"k": 11, "n": 12}, r["route_recall"]
    assert r["spoken_prefix"] == {"k": 9, "n": 11}, r["spoken_prefix"]  # J5: over routed cases only
    assert r["draft_valid"] == {"k": 3, "n": 7}, r["draft_valid"]
    assert r["draft_exact"] == {"k": 2, "n": 7}, r["draft_exact"]


def test_a_barged_in_draft_stays_out_of_the_valid_and_exact_denominators(tmp_path: Path) -> None:
    """J5: the barge-in moves from key-02 to click-01 (valid and exact), and key-02, now not barged in, drafts an
    invalid combo. The unbarged six are search-01 (valid, exact), search-02 (valid), type-01, scroll-01, key-01 and
    key-02 (all invalid): draft valid 2/6 and exact 1/6, and still five barge-ins."""
    cases = edit(reference_cases(), "click-01", on("click-01", barge_in=True))
    cases = edit(
        cases,
        "key-02",
        on("key-02", barge_in=False, call=Call("press_key", {"combo": "Control+C"}, False)),
    )
    r = report_of(tmp_path, cases)
    assert r["draft_valid"] == {"k": 2, "n": 6}, r["draft_valid"]
    assert r["draft_exact"] == {"k": 1, "n": 6}, r["draft_exact"]
    assert r["barge_in"]["n"] == 5, r["barge_in"]


def test_spoken_prefix_rate(tmp_path: Path) -> None:
    """Criterion 3, beside the eval's 27/40: routed tool cases whose spoken prefix is not blank. Non-blank:
    click-01, search-02, type-01, scroll-01 and the five barged-in cases = 9 of the 11 routed (search-01 and key-01
    spoke nothing)."""
    r = report_of(tmp_path)
    assert r["spoken_prefix"] == {"k": 9, "n": 11}, r["spoken_prefix"]


def test_repairs_their_outcomes_and_the_repair_span(tmp_path: Path) -> None:
    """Amendment: repairs fired with n and outcomes, and the ``tool.repair`` p50/p90. Three repairs: type-01 ends
    ``dry_run``, scroll-01 and key-01 end ``refused``; spans 180, 220, 220 ms -> p50 220, p90 220."""
    r = report_of(tmp_path)
    rep = r["repairs"]
    assert rep["n"] == 3, rep
    assert rep["outcomes"] == {"dry_run": 1, "refused": 2}, rep["outcomes"]
    assert (rep["ms"]["n"], rep["ms"]["p50"], rep["ms"]["p90"]) == (3, 220, 220), rep["ms"]


def test_action_latency_is_execute_start_minus_pass1_done_end(tmp_path: Path) -> None:
    """Amendment: ``tool.execute`` span start minus pass 1's ``llm.done`` span end, on the session clock. Executed:
    click-01 400, search-01 100, search-02 400, type-01 400 (its repair took 180 of those) -> n 4, p50 400, p90 400.
    Counted from ``llm.start`` (the span's start) instead, every value would be 1550 ms larger."""
    r = report_of(tmp_path)
    lat = r["action_latency_ms"]
    assert (lat["n"], lat["p50"], lat["p90"]) == (4, 400, 400), lat


def test_action_latency_reads_pass_1s_done_not_the_repairs(tmp_path: Path) -> None:
    """T-0092c criterion 7: the ``llm.done`` span stays pass 1's. type-01's call runs on the repair's done 50 ms after
    a drain that came 400 ms after pass 1's done: its latency is 450, not 0. Values 100, 400, 400, 450 -> p50 400
    (with the repair's ``llm.done`` record as the base it would be 0, 100, 400, 400 -> p50 250). J9: nearest-rank
    p90 of four values is the ceil(3.6) = 4th, 450 (linear interpolation would give 435)."""
    cases = edit(reference_cases(), "type-01", on("type-01", execute="after_drain"))
    lat = report_of(tmp_path, cases)["action_latency_ms"]
    assert lat["n"] == 4 and lat["p50"] == 400, lat
    assert lat["p90"] == 450, f"J9: nearest-rank p90; got {lat['p90']}"


def test_q12_surface_rate_on_and_off(tmp_path: Path) -> None:
    """Criterion 3 and J1, as amended by T-0097 (K1): a positive case is surfaced when its own turn logged a
    ``ui.gate`` record with ``gated`` and ``surface_emitted`` true; the rate is over the positive cases. Tools on:
    flight_booking emitted at canvas 1 (a later canvas failed) -> yes; primes_table's one canvas was not gated -> no;
    1 of 2. Tools off: both emitted at canvas 1, 2 of 2. The negative control capital_of_france is on its own line
    (K2): a surface with tools on (1 of 1), none off (0 of 1) -- never mixed into the rate."""
    r = report_of(tmp_path)
    assert key(r, "q12_shown") == {"on": {"k": 1, "n": 2}, "off": {"k": 2, "n": 2}}, r["q12_shown"]
    assert key(r, "q12_negative_shown") == {"on": {"k": 1, "n": 1}, "off": {"k": 0, "n": 1}}, r[
        "q12_negative_shown"
    ]


def test_first_canvas_and_done_are_reported_relative_to_llm_start(tmp_path: Path) -> None:
    """Implementation note: typed turns make B3 unmeasurable, so ``llm.first_canvas`` and ``llm.done`` are span end
    minus span start. 27 turns per arm: on 14 x 1050 and 13 x 4000 -> p50 1050; off 14 x 1000 and 13 x 900 ->
    p50 1000; ratio 1.05. ``llm.done`` is 500 later in each: 1550 and 1500."""
    r = report_of(tmp_path)
    fc = r["first_canvas_ms"]
    assert (fc["on"]["n"], fc["on"]["p50"]) == (27, 1050), fc["on"]
    assert (fc["off"]["n"], fc["off"]["p50"]) == (27, 1000), fc["off"]
    assert fc["ratio"] == pytest.approx(1.05), fc["ratio"]
    done = r["llm_done_ms"]
    assert (done["on"]["p50"], done["off"]["p50"]) == (1550, 1500), done


def test_prefix_hit_and_startup_on_vs_off(tmp_path: Path) -> None:
    """J2: the pooled Σhit/Σquery of each arm is reported (the gate is per case). Every on turn hit 950 of 1000
    prefix tokens, every off turn 980 of 1000. Startup (the detokeniser load, amendment) is the manifest's 1650 on, 1300 off."""
    r = report_of(tmp_path)
    assert r["prefix_hit"]["on"] == pytest.approx(0.95), r["prefix_hit"]
    assert r["prefix_hit"]["off"] == pytest.approx(0.98), r["prefix_hit"]
    assert r["startup_ms"] == {"on": 1650, "off": 1300}, r["startup_ms"]


def test_the_dropped_stale_terminals_are_counted(tmp_path: Path) -> None:
    """Amendment: report the count of dropped stale terminals. The reference run has one note (chat-04) whose
    terminal never reached the machine."""
    st = report_of(tmp_path)["stale_terminals"]
    assert st == {"dropped": 1, "reached_machine": 0}, st


def test_the_markdown_puts_the_eval_beside_the_live_numbers(tmp_path: Path) -> None:
    """Criterion 3: recall, valid and exact beside the eval's 40/40, 40/40 and 39/40; the spoken prefix beside 27/40."""
    h = harness()
    md = h.render_markdown(h.build_report(build_run(tmp_path, reference_cases())))
    for ref in ("40/40", "39/40", "27/40"):
        assert ref in md, f"criterion 3: report.md does not show the eval's {ref}"
    for live in ("11/11", "3/6", "2/6", "9/11"):
        assert live in md, f"criterion 3: report.md does not show the live {live}"


# =====================================================================================================
# T-0097: Q12 "a surface was shown" (primary) beside "every ui commit valid" (secondary)
# =====================================================================================================


def test_q12_shown_and_all_valid_on_the_reference_run(tmp_path: Path) -> None:
    """T-0097 criteria 1-6 on the reference run, where the two definitions differ on flight_booking (on).

    Tools on: flight_booking ``(1, T, T), (2, T, F)``, ui_valid False; primes_table ``(0, F, F)`` (not gated),
    ui_valid None; capital_of_france (negative) ``(1, T, T)``, ui_valid True. Tools off: each positive ``(1, T, T)``
    with ui_valid True; the negative has no ui canvas and ui_valid None.

    - shown, positive: on 1/2 (flight_booking), off 2/2; every ui commit valid: on 0/2 (False, None), off 2/2.
    - negative: shown on 1/1, off 0/1; all valid on 1/1, off 0/1.
    - gates (gated / emitted / failed / not gated): positive on = flight 2/1/1/0 + primes 0/0/0/1 = 2/1/1/1;
      positive off = 1/1/0/0 twice = 2/2/0/0; negative on 1/1/0/0; negative off 0/0/0/0.
    - consistency: False with a failed record, None with no gated record, True with an emitting record: none.
    """
    r = report_of(tmp_path)
    assert key(r, "q12_shown") == {"on": {"k": 1, "n": 2}, "off": {"k": 2, "n": 2}}, r["q12_shown"]
    assert key(r, "q12_all_ui_valid") == {"on": {"k": 0, "n": 2}, "off": {"k": 2, "n": 2}}, r[
        "q12_all_ui_valid"
    ]
    assert key(r, "q12_negative_shown") == {"on": {"k": 1, "n": 1}, "off": {"k": 0, "n": 1}}
    assert key(r, "q12_negative_all_ui_valid") == {
        "on": {"k": 1, "n": 1},
        "off": {"k": 0, "n": 1},
    }
    g = key(r, "q12_gates")
    assert counts(g["positive"]["on"]) == (2, 1, 1, 1), g
    assert counts(g["positive"]["off"]) == (2, 2, 0, 0), g
    assert counts(g["negative"]["on"]) == (1, 1, 0, 0), g
    assert counts(g["negative"]["off"]) == (0, 0, 0, 0), g
    assert len(key(r, "q12_cases")) == 6, (
        "criterion 5: one row per Q12 (case, arm): 3 cases x 2 arms"
    )
    fb = q12_row(r, "flight_booking", "on")
    assert (fb["expect_surface"], fb["shown"], fb["all_ui_valid"]) == (True, True, False), fb
    assert counts(fb) == (2, 1, 1, 0), fb
    assert (fb["first_emit_canvas"], fb["speech_refused"]) == (1, 0), fb
    pt = q12_row(r, "primes_table", "on")
    assert (pt["shown"], pt["all_ui_valid"], pt["first_emit_canvas"]) == (False, None, None), pt
    assert counts(pt) == (0, 0, 0, 1), pt
    cf = q12_row(r, "capital_of_france", "off")
    assert (cf["expect_surface"], cf["shown"], cf["all_ui_valid"]) == (False, False, None), cf
    assert pairs(key(r, "q12_inconsistent")) == set(), r["q12_inconsistent"]


def test_one_surface_then_failed_gates_is_shown_not_all_valid(tmp_path: Path) -> None:
    """Adversarial (the failure mode of finding 6): flight_booking with tools on gets a surface at canvas 1, then
    six canvases (2..7) that fail the gate; the session's AND makes ui_valid False. The surface reached the panel
    and was never retracted, so it was shown: shown 1/1, every ui commit valid 0/1, gates 7 gated / 1 emitted /
    6 failed / 0 not gated, first emit at canvas 1. Keeping the AND, or reading only the last record, gives 0/1.
    Tools off: one surface at canvas 1, ui_valid True -> 1/1 under both. No negative case: 0/0 under both."""
    failing = tuple((k, True, False) for k in range(2, 8))
    r = report_of(
        tmp_path,
        with_q12(
            q12_case(
                "flight_booking",
                on_gates=((1, True, True), *failing),
                on_valid=False,
                off_gates=((1, True, True),),
                off_valid=True,
            )
        ),
    )
    assert key(r, "q12_shown") == {"on": {"k": 1, "n": 1}, "off": {"k": 1, "n": 1}}
    assert key(r, "q12_all_ui_valid") == {"on": {"k": 0, "n": 1}, "off": {"k": 1, "n": 1}}
    assert counts(key(r, "q12_gates")["positive"]["on"]) == (7, 1, 6, 0), r["q12_gates"]
    row = q12_row(r, "flight_booking", "on")
    assert (row["shown"], row["all_ui_valid"], row["first_emit_canvas"]) == (True, False, 1), row
    assert counts(row) == (7, 1, 6, 0), row
    assert key(r, "q12_negative_shown") == {"on": {"k": 0, "n": 0}, "off": {"k": 0, "n": 0}}
    assert pairs(key(r, "q12_inconsistent")) == set(), (
        "ui_valid False beside six failed records agrees"
    )


def test_failed_gates_before_and_after_a_surface(tmp_path: Path) -> None:
    """primes_table with tools on, in seq order: three failed gates (canvases 0, 1, 1), a surface at canvas 2, five
    failed gates (canvases 3..7); ui_valid False. Shown (neither the first nor the last record decides);
    ``first_emit_canvas`` 2 (the first record's canvas is 0); gates 9 / 1 / 8 / 0.

    Criterion 5's ``speech_refused``: on, ``extras`` has no ``speech_refused`` key and the log has three
    ``speech.refused`` records -> 3 (the 4b0bd92 primes_table shape). Off, ``extras.speech_refused`` is 1 beside
    three records -> 1 (the flag wins when present). Off's ui: one surface at canvas 2, ui_valid True."""
    gates = (
        (0, True, False),
        (1, True, False),
        (1, True, False),
        (2, True, True),
        *((k, True, False) for k in range(3, 8)),
    )
    r = report_of(
        tmp_path,
        with_q12(
            q12_case(
                "primes_table",
                on_gates=gates,
                on_valid=False,
                off_gates=((2, True, True),),
                off_valid=True,
                on_extra=(("speech_refused", None), ("refusal_records", 3)),
                off_extra=(("speech_refused", 1), ("refusal_records", 3)),
            )
        ),
    )
    row = q12_row(r, "primes_table", "on")
    assert row["shown"] is True, row
    assert row["first_emit_canvas"] == 2, row
    assert counts(row) == (9, 1, 8, 0), row
    assert row["speech_refused"] == 3, f"criterion 5: no extras key -> count the records; {row}"
    assert q12_row(r, "primes_table", "off")["speech_refused"] == 1, "criterion 5: extras first"
    assert key(r, "q12_shown")["on"] == {"k": 1, "n": 1}, r["q12_shown"]
    assert key(r, "q12_all_ui_valid")["on"] == {"k": 0, "n": 1}, r["q12_all_ui_valid"]


def test_not_gated_records_never_count_as_shown(tmp_path: Path) -> None:
    """Too wide. Tools on: primes_table logs three not-gated canvases (0, 1, 2) -> not shown, not_gated 3, gated 0;
    flight_booking logs one malformed ``gated: False, surface_emitted: True`` record -> not shown, gated 0, emitted
    0, not_gated 1; omelette_steps logs ``(1, T, T)`` -> shown (the other direction: the rule is not "never").
    Every ui_valid on is None except omelette's True. Shown on 1/3; gates on 1 / 1 / 0 / 4. Tools off: no ui canvas,
    ui_valid None -> 0/3 under both."""
    r = report_of(
        tmp_path,
        with_q12(
            q12_case(
                "primes_table",
                on_gates=((0, False, False), (1, False, False), (2, False, False)),
            ),
            q12_case("flight_booking", on_gates=((1, False, True),)),
            q12_case("omelette_steps", on_gates=((1, True, True),), on_valid=True),
        ),
    )
    pt, fb = q12_row(r, "primes_table", "on"), q12_row(r, "flight_booking", "on")
    assert (pt["shown"], counts(pt)) == (False, (0, 0, 0, 3)), pt
    assert (fb["shown"], counts(fb)) == (False, (0, 0, 0, 1)), fb
    assert (fb["first_emit_canvas"], pt["first_emit_canvas"]) == (None, None), (fb, pt)
    assert q12_row(r, "omelette_steps", "on")["shown"] is True
    assert key(r, "q12_shown") == {"on": {"k": 1, "n": 3}, "off": {"k": 0, "n": 3}}
    assert counts(key(r, "q12_gates")["positive"]["on"]) == (1, 1, 0, 4), r["q12_gates"]


def test_every_gate_emits_counts_under_both(tmp_path: Path) -> None:
    """Too strict. flight_booking with tools on: two surfaces (canvases 1 and 2), ui_valid True -> shown and every
    ui commit valid, 1/1 under both; gates 2 / 2 / 0 / 0; first emit at canvas 1. Tools off: no ui canvas, ui_valid
    None -> 0/1 under both (so neither rate is a constant)."""
    r = report_of(
        tmp_path,
        with_q12(
            q12_case("flight_booking", on_gates=((1, True, True), (2, True, True)), on_valid=True)
        ),
    )
    assert key(r, "q12_shown") == {"on": {"k": 1, "n": 1}, "off": {"k": 0, "n": 1}}
    assert key(r, "q12_all_ui_valid") == {"on": {"k": 1, "n": 1}, "off": {"k": 0, "n": 1}}
    row = q12_row(r, "flight_booking", "on")
    assert counts(row) == (2, 2, 0, 0) and row["first_emit_canvas"] == 1, row
    assert pairs(key(r, "q12_inconsistent")) == set()


def test_a_surface_in_another_turn_does_not_count(tmp_path: Path) -> None:
    """Criterion 1's join: flight_booking is the 25th case, so its tools-on turn is ``t25``; plain-05's is ``t24``.
    In flight_booking's stretch of the tools-on log the session wrote its own ``(1, T, F)`` and two emitting records
    that are not its turn's: one under ``t24`` and one with ``turn_id: None``. Not shown (0/1), gates 1 / 0 / 1 / 0,
    no first emit. Tools off: ``(1, T, T)``, ui_valid True -> 1/1 (a join that drops every record would give 0/1)."""
    r = report_of(
        tmp_path,
        with_q12(
            q12_case(
                "flight_booking",
                on_gates=((1, True, False),),
                on_valid=False,
                off_gates=((1, True, True),),
                off_valid=True,
                on_extra=(("stray_ui_gates", (("t24", 2, True, True), (None, 3, True, True))),),
            )
        ),
    )
    row = q12_row(r, "flight_booking", "on")
    assert row["shown"] is False, f"criterion 1: a record of another turn was counted; {row}"
    assert counts(row) == (1, 0, 1, 0) and row["first_emit_canvas"] is None, row
    assert key(r, "q12_shown") == {"on": {"k": 0, "n": 1}, "off": {"k": 1, "n": 1}}
    assert counts(key(r, "q12_gates")["positive"]["on"]) == (1, 0, 1, 0), r["q12_gates"]


def test_flag_and_log_disagreement_is_listed_not_voided(tmp_path: Path) -> None:
    """Criterion 6, tools on: flight_booking ui_valid True with only a failed record (no emit); primes_table
    ui_valid False with only an emit (no failure); omelette_steps ui_valid None with a gated record. Each is listed.
    Not listed (the too-wide direction): flight_booking_with_memory, ui_valid None beside a *not-gated* record;
    capital_of_france, True beside an emit. Tools off: positives ``(1, T, T)`` True, negative none / None: none
    listed. The run stays valid and every gate holds (H-T93: reported, never voided)."""
    r = report_of(
        tmp_path,
        with_q12(
            q12_case(
                "flight_booking",
                on_gates=((1, True, False),),
                on_valid=True,
                off_gates=((1, True, True),),
                off_valid=True,
            ),
            q12_case(
                "primes_table",
                on_gates=((1, True, True),),
                on_valid=False,
                off_gates=((1, True, True),),
                off_valid=True,
            ),
            q12_case(
                "omelette_steps",
                on_gates=((1, True, True),),
                on_valid=None,
                off_gates=((1, True, True),),
                off_valid=True,
            ),
            q12_case(
                "flight_booking_with_memory",
                on_gates=((0, False, False),),
                on_valid=None,
                off_gates=((1, True, True),),
                off_valid=True,
            ),
            q12_case("capital_of_france", on_gates=((1, True, True),), on_valid=True),
        ),
    )
    assert pairs(key(r, "q12_inconsistent")) == {
        ("flight_booking", "on"),
        ("primes_table", "on"),
        ("omelette_steps", "on"),
    }, r["q12_inconsistent"]
    assert r["void"] == [], f"criterion 6: a disagreement adds no void reason; got {r['void']}"
    assert only_failing(gates_of(r)) == [], "criterion 6: a disagreement moves no gate"
    # the log is primary: primes_table and omelette_steps shown, flight_booking not -> 2/4 (with memory: not)
    assert key(r, "q12_shown")["on"] == {"k": 2, "n": 4}, r["q12_shown"]
    assert key(r, "q12_all_ui_valid")["on"] == {"k": 1, "n": 4}, r["q12_all_ui_valid"]


def test_negative_control_never_mixes_into_the_rate(tmp_path: Path) -> None:
    """Criterion 3, both directions. Tools on: positives flight_booking ``(1, T, T)`` True and primes_table none /
    None; negatives capital_of_france ``(1, T, T)`` True, spider_legs none / None, cloud_joke ``(1, T, T), (2, T, F)``
    False. Tools off: both positives ``(1, T, T)`` True; the three negatives none / None.

    - positive shown on 1/2, off 2/2; all valid on 1/2, off 2/2 (n is 2, not 5: no negative enters);
    - negative shown on 2/3 (capital, cloud), off 0/3; all valid on 1/3 (capital), off 0/3 (n is 3, not 5);
    - gates: positive on 1/1/0/0, off 2/2/0/0; negative on 3/2/1/0, off 0/0/0/0."""
    r = report_of(
        tmp_path,
        with_q12(
            q12_case(
                "flight_booking",
                on_gates=((1, True, True),),
                on_valid=True,
                off_gates=((1, True, True),),
                off_valid=True,
            ),
            q12_case("primes_table", off_gates=((1, True, True),), off_valid=True),
            q12_case("capital_of_france", on_gates=((1, True, True),), on_valid=True),
            q12_case("spider_legs"),
            q12_case("cloud_joke", on_gates=((1, True, True), (2, True, False)), on_valid=False),
        ),
    )
    assert key(r, "q12_shown") == {"on": {"k": 1, "n": 2}, "off": {"k": 2, "n": 2}}
    assert key(r, "q12_all_ui_valid") == {"on": {"k": 1, "n": 2}, "off": {"k": 2, "n": 2}}
    assert key(r, "q12_negative_shown") == {"on": {"k": 2, "n": 3}, "off": {"k": 0, "n": 3}}
    assert key(r, "q12_negative_all_ui_valid") == {
        "on": {"k": 1, "n": 3},
        "off": {"k": 0, "n": 3},
    }
    g = key(r, "q12_gates")
    assert counts(g["positive"]["on"]) == (1, 1, 0, 0), g
    assert counts(g["positive"]["off"]) == (2, 2, 0, 0), g
    assert counts(g["negative"]["on"]) == (3, 2, 1, 0), g
    assert counts(g["negative"]["off"]) == (0, 0, 0, 0), g
    assert q12_row(r, "cloud_joke", "on")["expect_surface"] is False


def test_old_q12_keys_are_gone(tmp_path: Path) -> None:
    """Criterion 7 (K2): a redefined key would mix definitions across runs silently; a missing one fails loudly."""
    r = report_of(tmp_path)
    assert "q12_surface" not in r, (
        "criterion 7: q12_surface is gone (now q12_shown / q12_all_ui_valid)"
    )
    assert "q12_negative" not in r, "criterion 7: q12_negative is gone (K2)"


def _line(md: str, label: str, *, exclude: str | None = None) -> str:
    hits = [
        line
        for line in md.splitlines()
        if label in line and (exclude is None or exclude not in line)
    ]
    assert len(hits) == 1, f"criterion 8: one report.md line labelled {label!r}; got {hits}"
    return hits[0]


def _ints_after(line: str, label: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", line.split(label, 1)[1])]


SHOWN_LABEL = (
    "surface shown (primary; Q12 / T-0055 `surface_accepted`; component count not recorded)"
)
ALL_VALID_LABEL = "every ui commit valid (secondary; `extras.ui_valid`)"
GATES_LABEL = "ui.gate on positive cases: gated / emitted / failed / not gated"
NEG_SHOWN_LABEL = "negative control, surface shown"
NEG_ALL_VALID_LABEL = "negative control, every ui commit valid"


def test_q12_markdown_rows(tmp_path: Path) -> None:
    """Criterion 8. The reference run with capital_of_france (negative) changed so the two definitions differ there
    too: tools on ``(1, T, T), (2, T, F)``, ui_valid False. Then, on / off:

    - surface shown: 1/2, 2/2; every ui commit valid: 0/2, 2/2;
    - positive gates: on 2 / 1 / 1 / 1, off 2 / 2 / 0 / 0 (the integers after the label, in that order);
    - negative control shown: 1/1, 0/1 -- the old row put ui_valid (0/1, 0/1) under this label;
    - negative control every ui commit valid: 0/1, 0/1.

    No inconsistency in that run, so no line names one; a second run whose primes_table (on) reads ui_valid None
    beside a gated failure gets one line naming primes_table and ``on``. (The line's wording is not fixed by the
    task; this row asks for the word "inconsistent" on it.)"""
    h = harness()
    cases = edit(
        reference_cases(),
        "capital_of_france",
        on("capital_of_france", ui_gates=((1, True, True), (2, True, False)), ui_valid=False),
    )
    md = h.render_markdown(h.build_report(build_run(tmp_path / "a", cases)))
    shown = _line(md, SHOWN_LABEL, exclude="negative control")
    assert re.search(r"1/2\D+2/2", shown.split(SHOWN_LABEL, 1)[1]), shown
    valid = _line(md, ALL_VALID_LABEL)
    assert re.search(r"0/2\D+2/2", valid.split(ALL_VALID_LABEL, 1)[1]), valid
    assert _ints_after(_line(md, GATES_LABEL), GATES_LABEL) == [2, 1, 1, 1, 2, 2, 0, 0], md
    neg = _line(md, NEG_SHOWN_LABEL)
    assert re.search(r"1/1\D+0/1", neg.split(NEG_SHOWN_LABEL, 1)[1]), (
        f"criterion 8: the negative 'surface shown' row reads the log (1/1 on), not ui_valid (0/1): {neg}"
    )
    neg_valid = _line(md, NEG_ALL_VALID_LABEL)
    assert re.search(r"0/1\D+0/1", neg_valid.split(NEG_ALL_VALID_LABEL, 1)[1]), neg_valid
    assert not [x for x in md.splitlines() if "inconsistent" in x.lower()], (
        "criterion 8: the inconsistency line appears only when q12_inconsistent is non-empty"
    )

    bad = edit(reference_cases(), "primes_table", on("primes_table", ui_gates=((1, True, False),)))
    md2 = h.render_markdown(h.build_report(build_run(tmp_path / "b", bad)))
    lines = [x for x in md2.splitlines() if "inconsistent" in x.lower()]
    assert len(lines) == 1, f"criterion 8: one line listing q12_inconsistent; got {lines}"
    assert "primes_table" in lines[0] and "on" in lines[0], lines[0]


def _flip(v: bool | None) -> bool | None:
    return {True: False, False: True, None: True}[v]


@pytest.mark.parametrize("mutation", ["strip", "flip"])
def test_q12_changes_move_no_gate(tmp_path: Path, mutation: str) -> None:
    """Criterion 9. Every Q12 turn of the reference run, both arms: ``strip`` drops its ui.gate records and flips
    its ui_valid (True <-> False, None -> True); ``flip`` keeps the records with ``surface_emitted`` inverted (and
    flips ui_valid the same way). ``gates()`` is all-true in both runs (its value at 6a8600b on the reference run),
    ``void`` is ``[]`` in both, and every non-Q12 key but ``run_dir`` (the directory differs) is equal.

    The mutation is not vacuous: tools-off shown is 2/2 in the reference and 0/2 after either (strip: no record;
    flip: ``(1, T, F)``), and the mutated run has a flag/log disagreement listed (strip: off positives read True with
    no emitting record)."""

    def mutate(t: Turn) -> Turn:
        if t.kind != "q12":
            return t
        gates = () if mutation == "strip" else tuple((c, g, not e) for c, g, e in t.ui_gates)
        return replace(t, ui_gates=gates, ui_valid=_flip(t.ui_valid))

    ref = reference_cases()
    mut = [Case(mutate(c.on), mutate(c.off)) for c in ref]
    h = harness()
    a = h.build_report(build_run(tmp_path / "a", ref))
    b = h.build_report(build_run(tmp_path / "b", mut))
    assert key(a, "q12_shown")["off"] == {"k": 2, "n": 2}, a["q12_shown"]
    assert key(b, "q12_shown")["off"] == {"k": 0, "n": 2}, b["q12_shown"]
    assert pairs(key(b, "q12_inconsistent")), "fixture: the mutated run should disagree somewhere"
    assert gates_of(a) == gates_of(b) == dict.fromkeys(GATES, True), (gates_of(a), gates_of(b))
    assert a["void"] == b["void"] == [], (a["void"], b["void"])
    rest_a = {k: v for k, v in a.items() if not k.startswith("q12_") and k != "run_dir"}
    rest_b = {k: v for k, v in b.items() if not k.startswith("q12_") and k != "run_dir"}
    assert set(rest_a) == set(rest_b), sorted(set(rest_a) ^ set(rest_b))
    moved = sorted(k for k in rest_a if rest_a[k] != rest_b[k])
    assert moved == [], f"criterion 9: non-Q12 keys moved: {moved}"


# =====================================================================================================
# Void conditions (amendment; approved at H-T93)
# =====================================================================================================


@pytest.mark.parametrize(
    "tools",
    [TOOLS_ON + ":repair_off", "off", "native:000000000000:dry_run", "native:bbfc53ccdd80:live"],
    ids=["repair_off_suffix", "off", "another_tool_set", "another_mode"],
)
def test_one_tools_on_line_with_another_regime_voids_the_run(tmp_path: Path, tools: str) -> None:
    """Amendment: void unless *every* tools-on line reads exactly ``native:bbfc53ccdd80:dry_run``. One line is
    enough. ``:repair_off`` is the mutant that reads the prefix only."""
    cases = edit(reference_cases(), "plain-03", on("plain-03", tools=tools))
    report = report_of(tmp_path, cases)
    assert report["void"], f"amendment: a tools-on line reading {tools!r} must void the run"
    assert gates_of(report)["run_valid"] is False


def test_a_run_with_repair_off_is_invalid(tmp_path: Path) -> None:
    """Amendment: a run with repair off is invalid -- as the session announces it (T-0092b criterion 2): the start
    note and the ``:repair_off`` suffix on every line."""
    report = report_of(tmp_path, on_tools=TOOLS_ON + ":repair_off", repair_off_note=True)
    assert report["void"], "amendment: repair off must void the run"
    assert gates_of(report)["run_valid"] is False


def test_the_tools_off_lines_read_off_without_voiding(tmp_path: Path) -> None:
    """Guard width: the void check reads the tools-on arm. The reference's off lines read ``off`` and it is valid."""
    report = report_of(tmp_path)
    assert report["void"] == [] and gates_of(report)["run_valid"] is True


# =====================================================================================================
# Gate: nothing but the spoken prefix is voiced
# =====================================================================================================


def test_a_quote_delimiter_in_tts_source_fails_the_gate(tmp_path: Path) -> None:
    """Adversarial (task file): ``<|"|>`` in ``TtsSpeak.source``, with clean ``text``, fails."""
    assert '<|"|>' in SPECIAL_TOKENS
    speech = (("I will click Sign in.", 'I will click <|"|>Sign in. '),)
    cases = edit(reference_cases(), "click-01", on("click-01", speech=speech))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is False, 'a <|"|> in TtsSpeak.source must fail the gate'
    assert only_failing(g) == ["no_literal_voiced"], only_failing(g)


@pytest.mark.parametrize(
    ("case", "speech"),
    [
        ("click-01", (("Clicking <|tool_call> now.", "Clicking <|tool_call> now. "),)),
        ("search-02", (("Okay call:google_search.", "Okay call:google_search. "),)),
    ],
    ids=["literal_in_text", "call_colon"],
)
def test_a_literal_or_call_in_speech_fails_the_gate(
    tmp_path: Path, case: str, speech: tuple[tuple[str, str], ...]
) -> None:
    """Criterion 2 and J17: no ``TtsSpeak`` text or source holds a special literal or ``call:`` -- on a tool turn
    too, where the voiced sources are the spoken prefix itself (the fixture's ``tool.call`` records carry them), so
    only the literal rule can fail the turn. (J17 supersedes J11's argument-value rows: see the J17 rows below.)"""
    cases = edit(reference_cases(), case, on(case, speech=speech))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is False, f"{speech[0][0]!r} must fail the gate"


def _fixed_choice(case: str, call: Call, sentence: str) -> list[Case]:
    return edit(reference_cases(), case, on(case, call=call, speech=say(sentence)))


@pytest.mark.parametrize(
    ("case", "call", "sentence"),
    [
        ("scroll-01", Call("scroll", {"direction": "down"}, True), "Scrolling down."),
        ("key-01", Call("press_key", {"combo": "Enter"}, True), "I'll press Enter."),
    ],
    ids=["scroll_direction", "press_key_combo"],
)
def test_saying_a_fixed_choice_argument_passes(
    tmp_path: Path, case: str, call: Call, sentence: str
) -> None:
    """J11 (human): fixed choices are not checked -- "Scrolling down." with ``direction="down"`` and "I'll press
    Enter." with ``combo="Enter"`` are normal speech. (J3's "every string value" reading fails both.)"""
    g = gates_of(report_of(tmp_path, _fixed_choice(case, call, sentence)))
    assert g["no_literal_voiced"] is True, f"J11: {sentence!r} names a fixed choice, not free text"


@pytest.mark.parametrize(
    "sentence",
    ["Scrolling down call:scroll.", 'Scrolling <|"|>down.'],
    ids=["call_colon", "literal"],
)
def test_literals_and_call_are_still_checked_on_a_fixed_choice_turn(
    tmp_path: Path, sentence: str
) -> None:
    """J11, unchanged part: special literals and ``call:`` are checked everywhere, also on a turn whose call has
    only fixed-choice arguments."""
    call = Call("scroll", {"direction": "down"}, True)
    g = gates_of(report_of(tmp_path, _fixed_choice("scroll-01", call, sentence)))
    assert g["no_literal_voiced"] is False, f"{sentence!r} must fail the gate"


def test_a_literal_in_a_tools_off_turn_fails_too(tmp_path: Path) -> None:
    """Criterion 2 counts every turn of the run, not only the tools-on arm."""
    speech = (("Done.", "Done.<|channel> "),)
    cases = edit(reference_cases(), "plain-02", off("plain-02", speech=speech))
    assert gates_of(report_of(tmp_path, cases))["no_literal_voiced"] is False


def test_a_literal_in_the_source_of_an_empty_text_tts_speak_passes(tmp_path: Path) -> None:
    """J4 (T-0092c criterion 6): the source is checked only on a ``TtsSpeak`` with non-empty text; a refused or
    trimmed source travels with ``text=""`` and is never voiced."""
    speech = (
        ("You can call me any time.", "You can call me any time. "),
        ("", "<|tool_call>call:"),
    )
    cases = edit(reference_cases(), "chat-03", on("chat-03", speech=speech))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is True, only_failing(g)


@pytest.mark.parametrize(
    "sentence",
    [
        "Since 17 < 20 and 23 > 20, it is 391.",
        "If a<b and b>c, then nothing follows.",
        "You can call me at nine.",
        "The answer is 3 <= 4.",
    ],
)
def test_ordinary_angle_brackets_and_the_word_call_pass(tmp_path: Path, sentence: str) -> None:
    """Guard width (tests rule): a maths answer with ``<`` and ``>`` (an over-wide ``<[^>]*>`` would match
    ``< 20 and 23 >``), or the word ``call`` without a colon, is ordinary speech and must not fail the gate."""
    speech = ((sentence, sentence + " "),)
    cases = edit(reference_cases(), "chat-02", on("chat-02", speech=speech))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is True, (
        f"{sentence!r} is not a literal, call: or an argument value"
    )


# =====================================================================================================
# Gates: no ToolExecute on an interrupted turn, or before the turn's own drain
# =====================================================================================================


def test_a_tool_execute_on_an_interrupted_turn_fails_the_gate(tmp_path: Path) -> None:
    """Criterion 2: 0 ``ToolExecute`` on an interrupted turn. click-02 is the scripted barge-in."""
    cases = edit(reference_cases(), "click-02", on("click-02", execute="interrupted"))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_execute_on_interrupted"] is False


def test_a_cancelled_barge_in_passes_the_interrupted_gate(tmp_path: Path) -> None:
    """Guard width: the reference's barge-in (``ToolCancel(interrupted)``, no execute) and its four executed,
    uninterrupted calls pass."""
    assert gates_of(report_of(tmp_path))["no_execute_on_interrupted"] is True


def test_a_tool_execute_before_the_turns_drain_fails_the_gate(tmp_path: Path) -> None:
    """Amendment (T-0092a): 0 ``ToolExecute`` before the turn's own ``playback.drained``. click-01's call runs on
    pass 1's ``llm.done`` step, a step before its drain."""
    cases = edit(reference_cases(), "click-01", on("click-01", execute="llm_done"))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_execute_before_drain"] is False


def test_a_tool_execute_in_a_turn_that_never_drained_fails_the_gate(tmp_path: Path) -> None:
    """Amendment: a ``ToolExecute`` with no drain of its turn at all is before the drain too."""
    cases = edit(reference_cases(), "click-01", on("click-01", execute="llm_done", drain=False))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_execute_before_drain"] is False


def test_a_silent_valid_call_executed_on_llm_done_without_a_drain_passes(tmp_path: Path) -> None:
    """J15 (T-0089 R1): a valid call in THINKING with nothing voiced executes on ``llm.done``; there is no drain to
    wait for, and the turn is exempt. search-01 voices nothing and its call runs on pass 1's done step."""
    cases = edit(reference_cases(), "search-01", on("search-01", execute="llm_done", drain=False))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_execute_before_drain"] is True, (
        f"J15: a silent THINKING call is exempt; failing {only_failing(g)}"
    )


def test_a_voiced_turn_executing_on_llm_done_without_a_drain_still_fails(tmp_path: Path) -> None:
    """J15 guard width: the exemption is only for a turn that voiced nothing. search-02 spoke "Let me look that up."
    and then executes on ``llm.done`` with no drain: it fails, as it does with its drain still to come."""
    cases = edit(reference_cases(), "search-02", on("search-02", execute="llm_done", drain=False))
    assert gates_of(report_of(tmp_path, cases))["no_execute_before_drain"] is False
    cases = edit(reference_cases(), "search-02", on("search-02", execute="llm_done"))
    assert gates_of(report_of(tmp_path / "with_drain", cases))["no_execute_before_drain"] is False


def test_a_tool_execute_after_the_drain_on_a_later_step_passes(tmp_path: Path) -> None:
    """Guard width: a repair that ends after the drain runs its call on the repair's ``llm.done`` step, after the
    drain step -- legitimate, and not only the drain step itself may carry a ``ToolExecute``."""
    cases = edit(reference_cases(), "type-01", on("type-01", execute="after_drain"))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_execute_before_drain"] is True, only_failing(g)


# =====================================================================================================
# Gate: stale terminals
# =====================================================================================================


def test_a_stale_terminal_note_that_reached_the_machine_fails_the_gate(tmp_path: Path) -> None:
    """Amendment: 0 stale-terminal notes that reached the machine. chat-05's note is followed by the same terminal
    (type, ``played_ms``, ``played_text``) as a ``machine.step`` event."""
    cases = edit(reference_cases(), "chat-05", on("chat-05", stale="stale_reached"))
    report = report_of(tmp_path, cases)
    assert report["stale_terminals"]["reached_machine"] == 1, report["stale_terminals"]
    assert gates_of(report)["no_stale_terminal_reached_machine"] is False


def test_a_stale_terminal_reached_after_a_step_in_between_fails_the_gate(tmp_path: Path) -> None:
    """J7: *any* later ``machine.step`` of the session counts, not only the next one. chat-05's note, then a tick
    step, then the same terminal."""
    cases = edit(reference_cases(), "chat-05", on("chat-05", stale="stale_reached_later"))
    report = report_of(tmp_path, cases)
    assert report["stale_terminals"]["reached_machine"] == 1, report["stale_terminals"]
    assert gates_of(report)["no_stale_terminal_reached_machine"] is False


def test_a_stale_terminal_reached_in_a_later_turn_of_the_session_fails_the_gate(
    tmp_path: Path,
) -> None:
    """J7: chat-04's dropped note (tools-on session) and, a turn later in the same session, chat-05 steps that
    terminal (same type, ``played_ms`` and ``played_text``) through the machine."""
    cases = edit(reference_cases(), "chat-05", on("chat-05", stale="echo"))
    report = report_of(tmp_path, cases)
    assert report["stale_terminals"] == {"dropped": 1, "reached_machine": 1}, report[
        "stale_terminals"
    ]
    assert gates_of(report)["no_stale_terminal_reached_machine"] is False


def test_the_same_terminal_in_another_session_did_not_reach_this_one(tmp_path: Path) -> None:
    """Guard width (J7, "of the same session"): the identical terminal stepped in the tools-off session does not
    make the tools-on session's dropped note one that reached the machine."""
    cases = edit(reference_cases(), "chat-05", off("chat-05", stale="echo"))
    report = report_of(tmp_path, cases)
    assert report["stale_terminals"] == {"dropped": 1, "reached_machine": 0}, report[
        "stale_terminals"
    ]
    assert gates_of(report)["no_stale_terminal_reached_machine"] is True


def test_a_dropped_stale_terminal_and_the_turns_own_drain_pass(tmp_path: Path) -> None:
    """Guard width: the reference's chat-04 has a dropped note and then its own drain (another ``played_text``);
    the own drain is not the stale one."""
    report = report_of(tmp_path)
    assert gates_of(report)["no_stale_terminal_reached_machine"] is True


# =====================================================================================================
# Gates: prefix hit, the first-canvas ratio, false tools
# =====================================================================================================


def key96(report: dict[str, Any], name: str) -> Any:
    """A T-0096 report key, red on a named assertion (not a KeyError) until criterion 8 lands."""
    assert name in report, f"T-0096 criterion 8: the report has no {name!r} key"
    return report[name]


def prefix_edit(
    cases: list[Case],
    case: str,
    arm: str = "on",
    *,
    S: int,
    H: int,
    Q: int = 1400,
    P: int | None = None,
    Hp: int = 0,
) -> list[Case]:
    """One (case, arm)'s prefix measurement: S shared ids, pass 1's (H, Q), the prefill's (Hp, P = S + 5 unless
    given)."""
    change = {"shared": S, "hit": (H, Q), "prefill": (Hp, S + 5 if P is None else P)}
    return edit(cases, case, on(case, **change) if arm == "on" else off(case, **change))


def b11_row(report: dict[str, Any], case_id: str, arm: str = "on") -> dict[str, Any]:
    table = key96(report, "prefix_b11_cases")
    assert isinstance(table, dict) and arm in table, (
        f"criterion 8: prefix_b11_cases[{arm!r}]: {table!r}"
    )
    rows = [r for r in table[arm] if r.get("case_id") == case_id]
    assert len(rows) == 1, (
        f"criterion 8: one prefix_b11_cases row per case; {case_id}/{arm}: {rows}"
    )
    return dict(rows[0])


def b11_numbers(row: dict[str, Any]) -> tuple[Any, ...]:
    """``(S, P, Hp, H, Q, hit_on_prefix, allowed_miss, missed, hit_beyond_prefix, prefill_tail)`` of a row."""
    return tuple(
        row.get(k)
        for k in (
            "S",
            "P",
            "Hp",
            "H",
            "Q",
            "hit_on_prefix",
            "allowed_miss",
            "missed",
            "hit_beyond_prefix",
            "prefill_tail",
        )
    )


def ratio_of(row: dict[str, Any]) -> float:
    value = row.get("ratio")
    assert isinstance(value, int | float | Fraction), f"criterion 8: ratio {value!r}"
    return float(value)


def test_prefix_hit_below_point_nine_with_tools_on_fails(tmp_path: Path) -> None:
    """H-T96 (rewritten from J2's premise; T-0096): every tools-on case has S 1360, H 1184 = 37*32, Q 1400, so
    Hs = 1184 and Hs/S = Fraction(1184, 1360) = 74/85 = 0.8706 < 9/10 (the block clause fails too: 1184 < 1344).
    The pooled B11 value per arm is ΣHs/ΣS = 74/85 on; off keeps 1568/1580 = 392/395."""
    cases = reference_cases()
    for c in [c.on.case for c in cases]:
        cases = prefix_edit(cases, c, S=1360, H=1184, Q=1400)
    report = report_of(tmp_path, cases)
    b11 = key96(report, "prefix_b11")
    assert float(b11["on"]) == pytest.approx(float(Fraction(74, 85))), b11
    assert float(b11["off"]) == pytest.approx(float(Fraction(392, 395))), b11
    assert gates_of(report)["prefix_hit_on"] is False


def test_one_tools_on_case_below_point_nine_fails_even_when_the_pool_passes(tmp_path: Path) -> None:
    """H-T96, per case (J2's per-case intent carried over): plain-03 at S 1360, H 1184 (Hs/S = 74/85 < 9/10) among
    26 cases at Hs 1216 of S 1220. Pooled B11 on = (26*1216 + 1184) / (26*1220 + 1360) = 32800/33080 = 0.9915 and
    the whole request (J2) (26*1216 + 1184) / (26*1280 + 1400) = 32800/34680 = 0.9458; the gate still fails."""
    cases = prefix_edit(reference_cases(), "plain-03", S=1360, H=1184, Q=1400)
    report = report_of(tmp_path, cases)
    assert report["prefix_hit"]["on"] == pytest.approx(32800 / 34680), report["prefix_hit"]
    assert float(key96(report, "prefix_b11")["on"]) == pytest.approx(32800 / 33080)
    assert b11_row(report, "plain-03")["holds"] is False
    assert gates_of(report)["prefix_hit_on"] is False


def test_one_tools_on_case_on_the_block_line_passes(tmp_path: Path) -> None:
    """Rewritten from ``test_one_tools_on_case_at_exactly_point_nine_passes`` (T-0096). Under H-T96 a case at
    exactly Hs/S = 9/10 can never pass: Hs is a multiple of 32 (criterion 5e), so 9S/10 = 32k needs 9 | k, S >= 320,
    and the miss S/10 >= 32 exceeds S mod 32. The line that *can* be stood on is the block clause's: plain-03 at
    S 1375 = 42*32 + 31 with H 1344 = 42*32 misses 31 = S mod 32 tokens (all of the partial block, nothing else):
    Fraction(1344, 1375) = 0.977 >= 9/10. Holds."""
    cases = prefix_edit(reference_cases(), "plain-03", S=1375, H=1344, Q=1400)
    report = report_of(tmp_path, cases)
    row = b11_row(report, "plain-03")
    assert (row.get("allowed_miss"), row.get("missed"), row.get("holds")) == (31, 31, True), row
    assert gates_of(report)["prefix_hit_on"] is True


def test_prefix_hit_is_gated_on_the_tools_on_arm_only(tmp_path: Path) -> None:
    """Guard width, H-T96: the gate reads tools on. Every tools-off case at S 1360, H 672 (Hs/S = 42/85 = 0.494, and
    half its blocks missed) while every tools-on case holds -> the gate holds. Pooling the arms would give
    (27*1216 + 27*672) / (27*1220 + 27*1360) = 1888/2580 = 0.7318 and fail it. The off arm is measured the same way
    and reported: prefix_b11 off = 42/85."""
    cases = reference_cases()
    for c in [c.on.case for c in cases]:
        cases = prefix_edit(cases, c, "off", S=1360, H=672, Q=1600)
    report = report_of(tmp_path, cases)
    assert float(key96(report, "prefix_b11")["off"]) == pytest.approx(float(Fraction(42, 85)))
    assert b11_row(report, "plain-03", "off")["holds"] is False
    assert gates_of(report)["prefix_hit_on"] is True


def _with_first_canvas(on_fc: int, off_fc: int) -> list[Case]:
    return [
        Case(
            replace(c.on, fc=on_fc, done=on_fc + 500), replace(c.off, fc=off_fc, done=off_fc + 500)
        )
        for c in reference_cases()
    ]


def test_first_canvas_at_exactly_one_point_one_times_off_passes(tmp_path: Path) -> None:
    """Criterion 2: p50 on <= 1.10 x p50 off. 1100 against 1000 is on the line and passes."""
    report = report_of(tmp_path, _with_first_canvas(1100, 1000))
    assert gates_of(report)["first_canvas_ratio"] is True


def test_first_canvas_more_than_one_point_one_times_off_fails(tmp_path: Path) -> None:
    """Criterion 2: 1150 against 1000 is a 15 % regression and fails."""
    report = report_of(tmp_path, _with_first_canvas(1150, 1000))
    assert report["first_canvas_ms"]["ratio"] == pytest.approx(1.15)
    assert gates_of(report)["first_canvas_ratio"] is False


def test_first_canvas_is_gated_on_the_median_not_the_mean(tmp_path: Path) -> None:
    """Criterion 2 names p50. The reference arms' means are 2470.4 and 951.9 (ratio 2.60); the medians 1050 and 1000
    (1.05) pass."""
    assert gates_of(report_of(tmp_path))["first_canvas_ratio"] is True


def _false_call() -> Call:
    return Call("google_search", {"query": "thanks"}, True, gap=300)


def test_one_false_tool_in_thirteen_passes(tmp_path: Path) -> None:
    """Criterion 2: <= 1/13 over the 8 no-tool and 5 plain cases of the tools-on arm. One is on the line."""
    cases = edit(reference_cases(), "chat-01", on("chat-01", call=_false_call()))
    report = report_of(tmp_path, cases)
    assert report["false_tool"] == {"k": 1, "n": 13}, report["false_tool"]
    assert gates_of(report)["false_tool_rate"] is True


def test_two_false_tools_in_thirteen_fail(tmp_path: Path) -> None:
    """Criterion 2: 2/13 > 1/13. A plain case counts like a no-tool case."""
    cases = edit(reference_cases(), "chat-01", on("chat-01", call=_false_call()))
    cases = edit(cases, "plain-04", on("plain-04", call=_false_call()))
    report = report_of(tmp_path, cases)
    assert report["false_tool"] == {"k": 2, "n": 13}, report["false_tool"]
    assert gates_of(report)["false_tool_rate"] is False


def test_a_no_tool_turn_with_a_tool_execute_counts_as_a_false_tool(tmp_path: Path) -> None:
    """Adversarial (task file): a no-tool turn whose only tool evidence is a ``ToolExecute`` command in its
    ``machine.step`` (no tool commit, no ``tool.call``, no ``TurnRecord.tool``) is a false tool. With a second,
    ordinary false call the rate is 2/13 and fails."""
    cases = edit(reference_cases(), "chat-06", on("chat-06", execute="bare"))
    report = report_of(tmp_path, cases)
    assert report["false_tool"] == {"k": 1, "n": 13}, report["false_tool"]
    cases = edit(cases, "plain-01", on("plain-01", call=_false_call()))
    report = report_of(tmp_path / "second", cases)
    assert report["false_tool"]["k"] == 2, report["false_tool"]
    assert gates_of(report)["false_tool_rate"] is False


def test_tool_cases_and_the_tools_off_arm_are_not_false_tools(tmp_path: Path) -> None:
    """Guard width: tool cases' calls are not false; ``n`` is the tools-on arm's 13, not both arms' 26."""
    report = report_of(tmp_path)
    assert report["false_tool"] == {"k": 0, "n": 13}, report["false_tool"]


# =====================================================================================================
# J17 (human): what is voiced is the turn's spoken prefix
# =====================================================================================================


def test_saying_the_value_to_type_in_the_spoken_prefix_passes(tmp_path: Path) -> None:
    """J17: the model may say what it will type, because that sentence is the spoken prefix itself. J11's
    free-text rule failed this sentence ("Anna Berg" is ``type_at.text``)."""
    sentence = 'I will type "Anna Berg" in the display name field.'
    call = Call("type_at", {"x": 500, "y": 302, "text": "Anna Berg", "press_enter": False}, True)
    cases = edit(reference_cases(), "type-01", on("type-01", call=call, speech=say(sentence)))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is True, (
        f"J17: {sentence!r} is the spoken prefix; failing {only_failing(g)}"
    )


@pytest.mark.parametrize(
    ("case", "voiced", "prefix"),
    [
        ("click-01", ("I will click Sign in.", "And then more."), "I will click Sign in. "),
        (
            "search-02",
            ("Let me look that up.", "Searching let me search british museum."),
            "Let me look that up. ",
        ),
        ("click-01", ("I will press Sign in.",), "I will click Sign in. "),
    ],
    ids=["beyond_the_prefix", "free_text_beyond_the_prefix", "differs_from_the_prefix"],
)
def test_voiced_text_that_is_not_a_prefix_of_the_spoken_prefix_fails(
    tmp_path: Path, case: str, voiced: tuple[str, ...], prefix: str
) -> None:
    """J17: on a tool turn the concatenated voiced sources must be a prefix of the turn's cleaned ``spoken_prefix``
    (its first ``tool.call`` record). Voicing past it, or something else, fails."""
    cases = edit(reference_cases(), case, on(case, speech=say(*voiced), spoken_prefix=prefix))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is False, (
        f"J17: {' '.join(voiced)!r} is not a prefix of {prefix!r}"
    )


@pytest.mark.parametrize(
    ("voiced", "prefix"),
    [
        (
            (("I will click Sign in.", "I will click Sign in. "),),
            "I will click Sign in. Then I wait. ",
        ),
        ((("I will click Sign in.", "I will  click\nSign in. "),), "I will click Sign in. "),
    ],
    ids=["a_strict_prefix_after_a_barge_in", "whitespace_normalised"],
)
def test_a_voiced_prefix_of_the_spoken_prefix_passes(
    tmp_path: Path, voiced: tuple[tuple[str, str], ...], prefix: str
) -> None:
    """J17 guard width: a prefix, not an equal string (a barge-in cuts the speech), compared after whitespace
    normalisation."""
    cases = edit(reference_cases(), "click-01", on("click-01", speech=voiced, spoken_prefix=prefix))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is True, (
        f"J17: a voiced prefix of {prefix!r}; failing {only_failing(g)}"
    )


# =====================================================================================================
# Text vs source: each side alone is checked (kills M5b/M5c)
# =====================================================================================================


@pytest.mark.parametrize(
    ("text", "source"),
    [
        ("Done <|tool_call> now.", "Done now. "),
        ("Done now.", "Done <|tool_call> now. "),
        ("Done call:scroll now.", "Done now. "),
        ("Done now.", "Done call:scroll now. "),
    ],
    ids=["literal_text_only", "literal_source_only", "call_text_only", "call_source_only"],
)
def test_a_literal_or_call_on_either_side_alone_fails(
    tmp_path: Path, text: str, source: str
) -> None:
    """Criterion 2 (J4 scope): ``TtsSpeak.text`` and ``.source`` are each checked on their own. On a no-tool turn,
    so that nothing but the literal rule applies."""
    cases = edit(reference_cases(), "chat-01", on("chat-01", speech=((text, source),)))
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_literal_voiced"] is False, f"text {text!r} / source {source!r} must fail"


# =====================================================================================================
# J15's other half: the exemption is only for a turn that voiced nothing (kills M3, M47)
# =====================================================================================================


def test_a_thinking_call_in_a_turn_that_has_voiced_fails(tmp_path: Path) -> None:
    """J15: search-02 voiced "Let me look that up." on an earlier step; its call then executes on ``llm.done`` in a
    THINKING step with no drain. Voicing is read over the whole turn, not the executing step, so it fails."""
    cases = edit(
        reference_cases(),
        "search-02",
        on("search-02", execute="llm_done", drain=False, done_state="THINKING"),
    )
    g = gates_of(report_of(tmp_path, cases))
    assert g["no_execute_before_drain"] is False, "J15: a turn that voiced is not exempt"


# =====================================================================================================
# J18: barge-ins
# =====================================================================================================


def _integrity(report: dict[str, Any]) -> list[str]:
    return [v for v in report["void"] if str(v).startswith("integrity:")]


def test_fewer_than_five_barge_ins_void_the_run(tmp_path: Path) -> None:
    """J18: four delivered (key-02's did not land) -> void (``integrity:``)."""
    cases = edit(reference_cases(), "key-02", on("key-02", barge_in=False))
    report = report_of(tmp_path, cases)
    assert _integrity(report), f"J18: 4 barge-ins must void the run; void {report['void']}"
    assert gates_of(report)["run_valid"] is False


def test_a_barge_in_on_an_unrouted_tool_turn_does_not_count(tmp_path: Path) -> None:
    """J18: "delivered on a routed tool turn". key-02's barge-in lands on a turn with no call, so four count."""
    cases = edit(reference_cases(), "key-02", on("key-02", call=None))
    report = report_of(tmp_path, cases)
    assert _integrity(report), (
        f"J18: a barge-in on an unrouted turn must not count; void {report['void']}"
    )


def test_five_barge_ins_on_routed_tool_turns_are_valid(tmp_path: Path) -> None:
    """J18 guard width: the reference's five barge-ins void nothing."""
    report = report_of(tmp_path)
    assert report["void"] == [] and report["barge_in"]["n"] == 5, (
        report["void"],
        report["barge_in"],
    )


def test_the_barge_in_candidates_are_three_per_tool_and_never_a_silent_caller() -> None:
    """J18: 15 pre-registered candidates, 3 per tool, all dev tool cases; search-03 and search-06 (silent callers in
    T-0084e dev) are not among them. (Whether each spoke a prefix in T-0084e dev is the reviewer's check: the
    artifact is not in the tree.)"""
    h = harness()
    candidates = list(h.BARGE_IN_CANDIDATES)
    rows = {
        r["id"]: r
        for r in (
            json.loads(x)
            for x in (REPO / "tests/eval/toolcall/cases.jsonl").read_text().splitlines()
        )
    }
    assert len(candidates) == 15 and len(set(candidates)) == 15, candidates
    off_split = [
        c for c in candidates if rows.get(c, {}).get("split") != "dev" or rows[c]["kind"] != "tool"
    ]
    assert off_split == [], f"J18: not dev tool cases: {off_split}"
    per_tool: dict[str, int] = {}
    for c in candidates:
        per_tool[rows[c]["tool"]] = per_tool.get(rows[c]["tool"], 0) + 1
    assert per_tool == {
        t: 3 for t in ("google_search", "click_at", "type_at", "scroll", "press_key")
    }, per_tool
    assert not {"search-03", "search-06"} & set(candidates), candidates


# =====================================================================================================
# J19: errored and incomplete cases void the run
# =====================================================================================================


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "error"},
        {"manifest": (("failures", ["RuntimeError('boom')"]),)},
        {"manifest": (("timed_out", True),)},
        {"manifest": (("prefill_done", False),)},
        {"manifest": (("max_inflight", 2),)},
    ],
    ids=["error_outcome", "failures", "timed_out", "prefill_not_done", "two_in_flight"],
)
def test_an_errored_or_incomplete_tools_on_case_voids_the_run(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """J19: any of these on one tools-on case (plain-02) voids the run (``integrity:``)."""
    cases = edit(reference_cases(), "plain-02", on("plain-02", **change))
    report = report_of(tmp_path, cases)
    assert _integrity(report), f"J19: {change} must void the run; void {report['void']}"
    assert gates_of(report)["run_valid"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "error"},
        {"manifest": (("failures", ["x"]),)},
        {"manifest": (("timed_out", True),)},
    ],
    ids=["error_outcome", "failures", "timed_out"],
)
def test_an_errored_tools_off_case_voids_the_run_too(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """J19: the error, failure and time-out voids are not limited to the tools-on arm."""
    cases = edit(reference_cases(), "plain-02", off("plain-02", **change))
    assert _integrity(report_of(tmp_path, cases)), change


def test_report_md_shows_the_errored_timed_out_and_not_idle_counts(tmp_path: Path) -> None:
    """J19: report.md shows the counts -- here 1 errored, 2 timed out, 3 not idle (distinct, so each is read off its
    own line)."""
    cases = edit(reference_cases(), "plain-01", on("plain-01", outcome="error"))
    for name in ("plain-02", "plain-03"):
        cases = edit(cases, name, on(name, manifest=(("timed_out", True),)))
    for name in ("chat-05", "chat-06", "chat-07"):
        cases = edit(cases, name, on(name, manifest=(("engine_idle_before", False),)))
    h = harness()
    md = h.render_markdown(h.build_report(build_run(tmp_path, cases)))
    for word, count in (("errored", 1), ("timed.out", 2), ("idle", 3)):
        assert re.search(rf"(?im)^.*{word}.*\b{count}\b", md), (
            f"J19: report.md has no {word} count of {count}"
        )


def test_an_engine_that_was_not_idle_does_not_void(tmp_path: Path) -> None:
    """Accepted at J16: checking the engine is idle before a case is reported, not a void reason."""
    cases = edit(
        reference_cases(), "chat-05", on("chat-05", manifest=(("engine_idle_before", False),))
    )
    assert report_of(tmp_path, cases)["void"] == []


# =====================================================================================================
# J14: integrity voids, one row per mutant (reviewer B: M23-M29b)
# =====================================================================================================


def _lines(run: Path) -> list[str]:
    return (run / "turns.jsonl").read_text().splitlines(keepends=True)


def test_m23_a_case_with_two_turn_lines_voids_the_run(tmp_path: Path) -> None:
    run = build_run(tmp_path, reference_cases())
    lines = _lines(run)
    (run / "turns.jsonl").write_text("".join([*lines, lines[0]]))
    assert _integrity(harness().build_report(run)), "J14 (M23): a case with 2 turns.jsonl lines"


def test_m23_a_case_with_no_turn_line_voids_the_run(tmp_path: Path) -> None:
    run = build_run(tmp_path, reference_cases())
    (run / "turns.jsonl").write_text("".join(_lines(run)[1:]))
    assert _integrity(harness().build_report(run)), "J14 (M23): a case with no turns.jsonl line"


def test_m24_a_missing_session_log_voids_the_run(tmp_path: Path) -> None:
    run = build_run(tmp_path, reference_cases())
    (run / "sessions" / LOGS["off"]).unlink()
    assert _integrity(harness().build_report(run)), "J14 (M24): a session log is missing"


def test_m25_a_log_that_dropped_records_voids_the_run(tmp_path: Path) -> None:
    run = build_run(tmp_path, reference_cases())
    path = run / "sessions" / LOGS["on"]
    records = [json.loads(x) for x in path.read_text().splitlines()]
    gap = {**records[5], "kind": "dropped", "turn_id": None, "data": {"count": 3, "since_seq": 4}}
    records.insert(5, gap)
    for i, r in enumerate(records):
        r["seq"] = i
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert _integrity(harness().build_report(run)), "J14 (M25): the log dropped records"


def test_m26_a_tools_off_line_with_another_regime_voids_the_run(tmp_path: Path) -> None:
    cases = edit(reference_cases(), "plain-03", off("plain-03", tools=TOOLS_ON))
    assert _integrity(report_of(tmp_path, cases)), "J14 (M26): a tools-off line reads tools-on"


def test_m28_no_no_tool_or_plain_case_fails_the_false_tool_gate(tmp_path: Path) -> None:
    """A rate over nothing is not a pass."""
    cases = [c for c in reference_cases() if c.on.kind not in ("no_tool", "plain")]
    assert gates_of(report_of(tmp_path, cases))["false_tool_rate"] is False


def test_m29_no_prefix_measurement_fails_the_prefix_gate(tmp_path: Path) -> None:
    """H-T96 (rewritten, T-0096): B11 over no measured case is not a pass. Every tools-on case keeps its whole-request
    hit (1216 of 1280, which J2 passed) but has no B11 measurement: no ``prefix_ids.jsonl`` line and no prefill
    delta. S is unread for each, so the gate fails (it never falls back to the whole request)."""
    cases = [
        Case(replace(c.on, shared=None, prefill=(None, None)), c.off) for c in reference_cases()
    ]
    report = report_of(tmp_path, cases)
    assert report["prefix_hit"]["on"] == pytest.approx(0.95), report["prefix_hit"]
    assert gates_of(report)["prefix_hit_on"] is False


def test_m29b_one_unmeasured_tools_on_case_fails_the_prefix_gate(tmp_path: Path) -> None:
    """H-T96 (rewritten, T-0096): *every* tools-on case must be measured; plain-04 with no sidecar line and no
    prefill delta (its whole request still 1216/1280) is not skipped: its row reads "unread" and the gate fails."""
    cases = edit(reference_cases(), "plain-04", on("plain-04", shared=None, prefill=(None, None)))
    report = report_of(tmp_path, cases)
    row = b11_row(report, "plain-04")
    assert row.get("holds") is not True and "unread" in json.dumps(row), row
    assert gates_of(report)["prefix_hit_on"] is False


# =====================================================================================================
# Refusals: main refuses before any socket, writes nothing (one row per class; reviewer A)
# =====================================================================================================


@pytest.fixture
def refusal_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The harness with a clean tree, a healthy fake kit and every connect refused and recorded.

    ``_git`` is faked so that only the refusal under test can fire (the dirty-tree refusal has its own row); the kit
    is a stand-in carrying the two fields ``main`` reads, so no tokenizer file is loaded. A connect is refused with
    ``ConnectionRefusedError`` (an ``OSError``) and recorded, so a missing refusal shows up as a recorded connect
    -- never as a real packet."""
    import socket
    import types

    from local_astra import cli

    h = harness()
    connects: list[object] = []

    def refuse(address: object) -> None:
        connects.append(address)
        raise ConnectionRefusedError(f"refusal rig: no connect allowed ({address!r})")

    monkeypatch.setattr(socket.socket, "connect", lambda self, a: refuse(a))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, a: refuse(a))
    monkeypatch.setattr(socket, "create_connection", lambda a, *x, **k: refuse(a))
    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else GIT + "\n"

    monkeypatch.setattr(h, "_git", git)
    kit = types.SimpleNamespace(repair_off=None, tools_version="bbfc53ccdd80" + "0" * 52)
    monkeypatch.setattr(cli, "load_tool_kit", lambda profile: (kit, None))
    out = tmp_path / "x-none"
    return types.SimpleNamespace(h=h, cli=cli, kit=kit, connects=connects, status=status, out=out)


def _refused(rig: Any, argv: list[str]) -> None:
    rc = rig.h.main(argv)
    assert rc == rig.h.EXIT_REFUSED == 2, f"exit {rc}, not the refusal 2"
    assert rig.connects == [], f"a refusal connected first: {rig.connects}"
    assert not rig.out.exists(), "a refusal wrote the out-dir"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8001/v1",
        "http://10.0.0.1:8001/v1",
        "ftp://127.0.0.1/v1",
        "http://[::ffff:127.0.0.1]:8001/v1",
        "http://127.0.0.1@8.8.8.8:8001/v1",
        "http://0.0.0.0:8001/v1",
    ],
    ids=["a_name", "not_loopback", "not_http", "v4_mapped", "userinfo", "wildcard"],
)
def test_a_base_url_that_is_not_a_loopback_ip_literal_is_refused(
    refusal_rig: Any, url: str
) -> None:
    _refused(refusal_rig, ["--base-url", url, "--out-dir", str(refusal_rig.out)])


def test_a_dirty_tree_is_refused(refusal_rig: Any) -> None:
    refusal_rig.status["porcelain"] = " M local_astra/cli.py\n"
    _refused(
        refusal_rig, ["--base-url", "http://127.0.0.1:9/v1", "--out-dir", str(refusal_rig.out)]
    )


def test_a_non_empty_out_dir_is_refused_and_left_alone(refusal_rig: Any, tmp_path: Path) -> None:
    full = tmp_path / "x-full"
    full.mkdir()
    (full / "f").write_text("x")
    rc = refusal_rig.h.main(["--base-url", "http://127.0.0.1:9/v1", "--out-dir", str(full)])
    assert (rc, refusal_rig.connects) == (2, []), (rc, refusal_rig.connects)
    assert sorted(p.name for p in full.iterdir()) == ["f"] and (full / "f").read_text() == "x"


def test_a_cases_file_that_is_not_the_pinned_one_is_refused(
    refusal_rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refusal_rig.h, "CASES_SHA256", "0" * 64)
    _refused(
        refusal_rig, ["--base-url", "http://127.0.0.1:9/v1", "--out-dir", str(refusal_rig.out)]
    )


@pytest.mark.parametrize("what", ["kit_unavailable", "repair_off", "another_tool_set"])
def test_a_kit_that_cannot_run_the_eval_is_refused(
    refusal_rig: Any, monkeypatch: pytest.MonkeyPatch, what: str
) -> None:
    """The amendment: a run with repair off is invalid, so it is not started; nor without the kit, nor on another
    tool set."""
    cli = refusal_rig.cli
    if what == "kit_unavailable":

        def boom(profile: Any) -> Any:
            raise cli.ToolsUnavailable("no tokenizer")

        monkeypatch.setattr(cli, "load_tool_kit", boom)
    elif what == "repair_off":
        refusal_rig.kit.repair_off = "payload stale"
    else:
        refusal_rig.kit.tools_version = "0" * 64
    _refused(
        refusal_rig, ["--base-url", "http://127.0.0.1:9/v1", "--out-dir", str(refusal_rig.out)]
    )


def test_no_engine_exits_failed_after_loopback_connects_only(refusal_rig: Any) -> None:
    """Not a refusal: with every precondition met and nothing listening, ``main`` tries the engine (loopback only),
    exits 1 and writes nothing."""
    rc = refusal_rig.h.main(
        ["--base-url", "http://127.0.0.1:9/v1", "--out-dir", str(refusal_rig.out)]
    )
    assert rc == refusal_rig.h.EXIT_FAILED == 1, rc
    assert refusal_rig.connects, "the engine was never tried"
    assert all(str(a[0]) == "127.0.0.1" for a in refusal_rig.connects), refusal_rig.connects
    assert not refusal_rig.out.exists()


# =====================================================================================================
# Further mutant kills (reviewer B's battery: M14, M19, M19b, M22, M29, M33, M35, M38)
# =====================================================================================================


def test_m29_a_run_with_no_tools_on_case_fails_the_prefix_gate(tmp_path: Path) -> None:
    """H-T96 (rewritten, T-0096): B11 over no tools-on case at all is not a pass (the manifest and the sidecar lost
    their tools-on arm; every tools-off case still holds). No tools-on row is reported."""
    run = build_run(tmp_path, reference_cases())
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["cases"] = [c for c in manifest["cases"] if c["arm"] != "on"]
    manifest["sessions"] = [s for s in manifest["sessions"] if s["arm"] != "on"]
    (run / "manifest.json").write_text(json.dumps(manifest))
    side = [x for x in (run / SIDECAR).read_text().splitlines() if json.loads(x)["arm"] != "on"]
    (run / SIDECAR).write_text("".join(x + "\n" for x in side))
    h = harness()
    report = h.build_report(run)
    assert key96(report, "prefix_b11_cases")["on"] == [], report["prefix_b11_cases"]
    assert h.gates(report)["prefix_hit_on"] is False


def test_m14_the_same_terminal_stepped_before_the_note_did_not_reach_the_machine(
    tmp_path: Path,
) -> None:
    """J7 reads *later* steps only: chat-03 steps a terminal identical to the one chat-04 notes later in the same
    session; that earlier step is not the dropped terminal reaching the machine."""
    cases = edit(reference_cases(), "chat-03", on("chat-03", stale="echo"))
    report = report_of(tmp_path, cases)
    assert report["stale_terminals"] == {"dropped": 1, "reached_machine": 0}, report[
        "stale_terminals"
    ]
    assert gates_of(report)["no_stale_terminal_reached_machine"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "interrupted"},  # the line says interrupted; the steps show no takeover
        {
            "barge_in": True,
            "outcome": "spoken",
            "execute": "after_stop",
        },  # the steps do; the line does not
    ],
    ids=["line_only", "steps_only"],
)
def test_m19_an_execute_on_a_turn_interrupted_by_either_record_fails(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """Criterion 2: a turn is interrupted if its line says so *or* its steps went INTERRUPTED; either alone is
    enough for its ``ToolExecute`` to fail the gate."""
    cases = edit(reference_cases(), "click-01", on("click-01", **change))
    assert gates_of(report_of(tmp_path, cases))["no_execute_on_interrupted"] is False


def test_m38_a_playback_stopped_is_not_a_drain(tmp_path: Path) -> None:
    """Amendment: only the turn's own ``playback.drained`` releases a call. click-02's call runs in its
    ``playback.stopped`` step after the barge-in, with no drain: it is before the drain."""
    cases = edit(reference_cases(), "click-02", on("click-02", execute="after_stop"))
    assert gates_of(report_of(tmp_path, cases))["no_execute_before_drain"] is False


def test_m22_the_repair_off_note_alone_voids_the_run(tmp_path: Path) -> None:
    """Amendment: a run with repair off is invalid, as announced by the session's start note, whatever the lines'
    ``regime.tools`` say."""
    report = report_of(tmp_path, repair_off_note=True)
    assert report["void"], "the repair-off note must void the run"
    assert gates_of(report)["run_valid"] is False


def test_m33_query_problem_counts_routed_searches_only(tmp_path: Path) -> None:
    """J5: ``query_problem`` is over *routed* google_search cases. An unrouted search (search-04, answered in
    speech) leaves it at 1 of 2."""
    cases = reference_cases()
    extra = Turn(
        "search-04",
        "tool",
        {"keywords": ["banana bread", "egg"], "name": "google_search"},
        say("No idea."),
    )
    cases.insert(11, Case(extra, plain_off(replace(extra, **OFF_PREFIX))))
    r = report_of(tmp_path, cases)
    assert r["query_problem"] == {"k": 1, "n": 2}, r["query_problem"]


def test_m35_startup_is_the_p50_over_the_sessions_of_each_arm(tmp_path: Path) -> None:
    """J16: ``startup_ms`` is the p50 over each arm's sessions. Three tools-on sessions at 100, 200 and 900 ms
    (nearest rank p50 200, p90 900) and three tools-off at 5, 6 and 50 (p50 6)."""
    run = build_run(tmp_path, reference_cases())
    manifest = json.loads((run / "manifest.json").read_text())
    for s in manifest["sessions"]:
        s["startup_ms"] = 100 if s["arm"] == "on" else 5
    manifest["sessions"] += [
        {"arm": "on", "session_id": "a1a1a1a1a1a1", "log": LOGS["on"], "startup_ms": 200},
        {"arm": "on", "session_id": "a2a2a2a2a2a2", "log": LOGS["on"], "startup_ms": 900},
        {"arm": "off", "session_id": "b1b1b1b1b1b1", "log": LOGS["off"], "startup_ms": 6},
        {"arm": "off", "session_id": "b2b2b2b2b2b2", "log": LOGS["off"], "startup_ms": 50},
    ]
    (run / "manifest.json").write_text(json.dumps(manifest))
    startup = harness().build_report(run)["startup_ms"]
    assert (startup["on"], startup["off"]) == (200, 6), startup


# =====================================================================================================
# T-0096 (H-T96): the prefix gate measured as B11 is written
# =====================================================================================================
# Notation (task file): P, Hp = the prefill's /metrics query and hit deltas; Q, H = pass 1's (J2's numbers);
# S = LCP(prefill ids, generate ids); Hs = min(H, S); A = 32. Expected values are worked by hand in each docstring.

#: T-0096 branched from ``main`` here (docs/tasks/T-0096.md); criteria 7 and 11 compare against this build's report.
BASE_SHA = "cb4d654"
#: T-0096's merge commit 9e6e9cb; BASE = ``git merge-base 9e6e9cb^1 9e6e9cb^2`` = ``BASE_SHA``, TIP = ``9e6e9cb^2``
#: (T-0094b criterion 3: criteria 7 and 11 compare T-0096's own two harness blobs, never BASE with the live file; the
#: merge is not an ancestor of ``gr.LATER``, so the range is proven against ``gr.LATER_B``).
T0096_MERGE = "9e6e9cb"
T0096_BASE = "cb4d654"
T0096_TIP = "b693dc1"
#: The power pair of criterion 11's predicate (T-0094b ruling P7): T-0097's 1dcc542 (parent a2ea8f2) replaced the
#: Q12 report keys.
Q12_PRIOR_PARENT = "a2ea8f2"
Q12_PRIOR = "1dcc542"


def base_harness(tmp_path: Path, rev: str = BASE_SHA) -> ModuleType:
    """The harness as it was at ``rev`` (``BASE_SHA`` by default: an independent oracle for "nothing else moves");
    skipped, with the reason, when that commit is not in this clone. T-0094b: ``rev`` loads T-0096's TIP blob too."""
    try:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:scripts/gpu/t0093_toolcall_live.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"commit {rev} (T-0096's base) is not readable here: {exc}")
    path = tmp_path / "base_harness" / f"t0093_at_{rev}.py"
    path.parent.mkdir(parents=True)
    path.write_text(src)
    name = f"_t0093_toolcall_live_{rev}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    # the copy sits outside the repo: point its data paths back at the tree's
    module.REPO = REPO  # type: ignore[attr-defined]
    module.SET_DIR = REPO / "tests" / "eval" / "toolcall"  # type: ignore[attr-defined]
    module.Q12_PATH = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"  # type: ignore[attr-defined]
    return module


def unread(row: dict[str, Any]) -> bool:
    """Criterion 5: a case whose measurement is not verified reads "unread" and does not hold."""
    return row.get("holds") is not True and "unread" in json.dumps(row, default=str)


def rewrite_sidecar(run: Path, case: str, arm: str, fn: Callable[[str], str | None]) -> None:
    """Replace (or drop, on ``None``) the ``prefix_ids.jsonl`` line of one (case, arm)."""
    out: list[str] = []
    hit = 0
    for line in (run / SIDECAR).read_text().splitlines():
        row = json.loads(line)
        if (row["case_id"], row["arm"]) == (case, arm):
            hit += 1
            new = fn(line)
            if new is not None:
                out.append(new)
        else:
            out.append(line)
    assert hit == 1, f"fixture: {hit} sidecar lines for {case}/{arm}"
    (run / SIDECAR).write_text("".join(x + "\n" for x in out))


def edit_ids(field_: str, change: Callable[[list[int]], list[int]]) -> Callable[[str], str]:
    def fn(line: str) -> str:
        row = json.loads(line)
        row[field_] = change(list(row[field_]))
        return json.dumps(row)

    return fn


def test_b11_rows_carry_the_registered_fields(tmp_path: Path) -> None:
    """Criterion 8 on the reference run (the verified control: nothing voids, every case holds in both arms).

    click-01 on: S 1220, P 1225, Hp 1184, H 1216, Q 1280 -> hit_on_prefix 1216, allowed_miss 1220 mod 32 = 4,
    missed 4, ratio Fraction(1216, 1220) = 304/305, holds, hit_beyond_prefix 0, prefill_prewarmed 1184/1225,
    prefill_tail 5. click-01 off: S 1580, P 1587, Hp 1024, H 1568, Q 1600 -> 1568, 12, 12, 392/395, holds, 0,
    1024/1587, 7. Pooled B11: on 27*1216 / (27*1220) = 304/305, off 392/395. The rule and A come from the manifest's
    regime (``"H-T96"``, 32), never from the engine's ``block_size`` label (written here as ``"16"``)."""
    report = report_of(tmp_path)
    assert report["void"] == [], report["void"]
    assert key96(report, "prefix_rule") == "H-T96"
    assert key96(report, "prefix_block_tokens") == 32
    table = key96(report, "prefix_b11_cases")
    assert sorted(table) == ["off", "on"] and len(table["on"]) == len(table["off"]) == 27, table
    on_row, off_row = b11_row(report, "click-01"), b11_row(report, "click-01", "off")
    assert b11_numbers(on_row) == (1220, 1225, 1184, 1216, 1280, 1216, 4, 4, 0, 5), on_row
    assert b11_numbers(off_row) == (1580, 1587, 1024, 1568, 1600, 1568, 12, 12, 0, 7), off_row
    assert ratio_of(on_row) == pytest.approx(float(Fraction(304, 305))), on_row
    assert ratio_of(off_row) == pytest.approx(float(Fraction(392, 395))), off_row
    assert float(on_row["prefill_prewarmed"]) == pytest.approx(1184 / 1225), on_row
    assert float(off_row["prefill_prewarmed"]) == pytest.approx(1024 / 1587), off_row
    for arm in ("on", "off"):
        bad = [r.get("case_id") for r in table[arm] if r.get("holds") is not True or unread(r)]
        assert bad == [], f"the verified control: {arm} rows that do not hold: {bad}"
    b11 = key96(report, "prefix_b11")
    assert float(b11["on"]) == pytest.approx(304 / 305) and float(b11["off"]) == pytest.approx(
        392 / 395
    ), b11
    assert gates_of(report)["prefix_hit_on"] is True


def test_memory_hit_case_with_its_prefix_fully_hit_passes(tmp_path: Path) -> None:
    """Criterion 10a (the 4b0bd92 failure, under H-T96): flight_booking_with_memory with tools on, S 1360 = 42*32 +
    16, P 1365, H 1344 = 42*32, Q 1599 (the recollections and the question are the suffix). Hs = 1344 >= 1344,
    missed 16 = allowed 16, Hs/S = Fraction(1344, 1360) = 84/85 >= 9/10: holds, and the gate holds. The whole
    request is still reported: its prefix_hit_cases row reads hit 1344, query 1599 (= 0.8405, which J2 failed)."""
    cases = with_q12(q12_case("flight_booking_with_memory"))
    cases = prefix_edit(cases, "flight_booking_with_memory", S=1360, P=1365, H=1344, Q=1599)
    report = report_of(tmp_path, cases)
    row = b11_row(report, "flight_booking_with_memory")
    assert b11_numbers(row)[:8] == (1360, 1365, 0, 1344, 1599, 1344, 16, 16), row
    assert row.get("holds") is True and ratio_of(row) == pytest.approx(84 / 85), row
    j2 = [
        r for r in report["prefix_hit_cases"]["on"] if r["case_id"] == "flight_booking_with_memory"
    ]
    assert j2 == [{"case_id": "flight_booking_with_memory", "hit": 1344, "query": 1599}], j2
    assert gates_of(report)["prefix_hit_on"] is True


def test_prefill_that_missed_half_its_prefix_fails(tmp_path: Path) -> None:
    """Criterion 10b: plain-03 with tools on, S 1360, P 1365, H 672 = 21*32, Q 1400. Hs = 672 < floor(1360/32)*32
    = 1344; missed 688 > allowed 16; Hs/S = Fraction(672, 1360) = 42/85 = 0.494. Fails, and so does the gate."""
    report = report_of(tmp_path, prefix_edit(reference_cases(), "plain-03", S=1360, H=672))
    row = b11_row(report, "plain-03")
    assert b11_numbers(row) == (1360, 1365, 0, 672, 1400, 672, 16, 688, 0, 5), row
    assert row.get("holds") is False and ratio_of(row) == pytest.approx(42 / 85), row
    assert gates_of(report)["prefix_hit_on"] is False


def test_only_the_last_partial_block_may_miss(tmp_path: Path) -> None:
    """Criteria 10c and 10d, adversarial (kills a ratio-only gate, and ceil for floor). plain-03, S 1370 = 42*32 + 26:

    - H 1344 = 42*32: missed 26 = allowed 26, Fraction(1344, 1370) = 672/685 = 0.981 -> holds (a ceil would demand
      1376 and fail it);
    - H 1312 = 41*32: Fraction(1312, 1370) = 656/685 = 0.958 >= 9/10, yet missed 58 > 26 -> fails, and the gate
      fails with it."""
    c = report_of(tmp_path / "c", prefix_edit(reference_cases(), "plain-03", S=1370, H=1344))
    row = b11_row(c, "plain-03")
    assert (row.get("allowed_miss"), row.get("missed"), row.get("holds")) == (26, 26, True), row
    assert gates_of(c)["prefix_hit_on"] is True
    d = report_of(tmp_path / "d", prefix_edit(reference_cases(), "plain-03", S=1370, H=1312))
    row = b11_row(d, "plain-03")
    assert ratio_of(row) == pytest.approx(656 / 685) and ratio_of(row) >= 0.9, row
    assert (row.get("allowed_miss"), row.get("missed"), row.get("holds")) == (26, 58, False), row
    assert gates_of(d)["prefix_hit_on"] is False


def test_block_aligned_prefix_must_hit_in_full(tmp_path: Path) -> None:
    """Criterion 10e, both directions: plain-03 at S 1344 = 42*32 (allowed miss 0). H 1312: missed 32, ratio
    Fraction(1312, 1344) = 41/42 = 0.976 -> fails. H 1344: missed 0, ratio 1 -> holds."""
    short = report_of(tmp_path / "a", prefix_edit(reference_cases(), "plain-03", S=1344, H=1312))
    row = b11_row(short, "plain-03")
    assert (row.get("allowed_miss"), row.get("missed"), row.get("holds")) == (0, 32, False), row
    assert gates_of(short)["prefix_hit_on"] is False
    full = report_of(tmp_path / "b", prefix_edit(reference_cases(), "plain-03", S=1344, H=1344))
    row = b11_row(full, "plain-03")
    assert (row.get("allowed_miss"), row.get("missed"), row.get("holds")) == (0, 0, True), row
    assert gates_of(full)["prefix_hit_on"] is True


def test_hits_beyond_the_prefix_never_count(tmp_path: Path) -> None:
    """Criterion 10f (kills H for Hs): plain-03 at S 1360, H 1408 = 44*32, Q 1600. hit_on_prefix = min(1408, 1360)
    = 1360, hit_beyond_prefix 48, missed 0, ratio exactly Fraction(1) (H/S would be 22/21). Pooled B11 on =
    (26*1216 + 1360) / (26*1220 + 1360) = 32976/33080 (with H uncapped it would be 33024/33080)."""
    report = report_of(tmp_path, prefix_edit(reference_cases(), "plain-03", S=1360, H=1408, Q=1600))
    row = b11_row(report, "plain-03")
    assert (row.get("hit_on_prefix"), row.get("hit_beyond_prefix"), row.get("missed")) == (
        1360,
        48,
        0,
    ), row
    assert ratio_of(row) == float(Fraction(1)) and row.get("holds") is True, row
    assert float(key96(report, "prefix_b11")["on"]) == pytest.approx(32976 / 33080, abs=1e-9)
    assert gates_of(report)["prefix_hit_on"] is True


def test_denominator_is_the_shared_prefix_not_the_prefill_length(tmp_path: Path) -> None:
    """Criterion 10g, adversarial (kills P for S): plain-03 at S 1340, P 1350, H 1312 = 41*32. Against S:
    floor(1340/32)*32 = 1312 <= 1312, missed 28 = allowed 28, Fraction(1312, 1340) = 328/335 -> holds. Against P it
    would need floor(1350/32)*32 = 1344 and fail. prefill_tail = P - S = 10."""
    report = report_of(tmp_path, prefix_edit(reference_cases(), "plain-03", S=1340, P=1350, H=1312))
    row = b11_row(report, "plain-03")
    assert b11_numbers(row) == (1340, 1350, 0, 1312, 1400, 1312, 28, 28, 0, 10), row
    assert row.get("holds") is True and ratio_of(row) == pytest.approx(328 / 335), row
    assert gates_of(report)["prefix_hit_on"] is True


def test_tiny_prefix_fails_on_the_point_nine_clause(tmp_path: Path) -> None:
    """Criterion 10h (the only row that kills dropping the 0.9 clause): plain-03 at S 40 = 32 + 8, P 45, H 32.
    The block clause holds (32 >= 32; missed 8 = allowed 8), but Fraction(32, 40) = 4/5 < 9/10 -> fails."""
    report = report_of(tmp_path, prefix_edit(reference_cases(), "plain-03", S=40, H=32))
    row = b11_row(report, "plain-03")
    assert (row.get("allowed_miss"), row.get("missed")) == (8, 8), row
    assert ratio_of(row) == pytest.approx(0.8) and row.get("holds") is False, row
    assert gates_of(report)["prefix_hit_on"] is False


def test_no_shared_prefix_is_not_a_pass(tmp_path: Path) -> None:
    """Criterion 10i: plain-03 whose prefill and pass 1 share no leading id (S 0), with the whole request still at
    1216 of 1280 (J2 passed it). Hs = min(1216, 0) = 0, hit_beyond_prefix 1216; S = 0 fails (no 0/0)."""
    report = report_of(
        tmp_path, prefix_edit(reference_cases(), "plain-03", S=0, H=1216, Q=1280, P=1225)
    )
    row = b11_row(report, "plain-03")
    assert (row.get("S"), row.get("hit_on_prefix"), row.get("hit_beyond_prefix")) == (0, 0, 1216), (
        row
    )
    assert row.get("holds") is not True, row
    assert gates_of(report)["prefix_hit_on"] is False


def test_b11_gate_reads_tools_on_only(tmp_path: Path) -> None:
    """Criterion 10j and its reverse. plain-03 at S 1360, H 672 (fails B11): with tools off only, the gate holds
    (the off row is reported, holds False); with tools on only, the gate fails."""
    off_only = report_of(
        tmp_path / "off", prefix_edit(reference_cases(), "plain-03", "off", S=1360, H=672)
    )
    assert b11_row(off_only, "plain-03", "off").get("holds") is False
    assert gates_of(off_only)["prefix_hit_on"] is True
    on_only = report_of(tmp_path / "on", prefix_edit(reference_cases(), "plain-03", S=1360, H=672))
    assert b11_row(on_only, "plain-03", "off").get("holds") is True
    assert gates_of(on_only)["prefix_hit_on"] is False


def test_one_failing_case_among_passing_fails(tmp_path: Path) -> None:
    """Criterion 3 (every tools-on case, not the pool): chat-05 at S 1370, H 1312 (missed 58 > allowed 26) among 26
    healthy cases. Pooled B11 on = (26*1216 + 1312) / (26*1220 + 1370) = 32928/33090 = 0.9951 >= 0.9; the gate
    still fails."""
    report = report_of(tmp_path, prefix_edit(reference_cases(), "chat-05", S=1370, H=1312))
    assert float(key96(report, "prefix_b11")["on"]) == pytest.approx(32928 / 33090)
    assert b11_row(report, "chat-05").get("holds") is False
    assert gates_of(report)["prefix_hit_on"] is False


def test_s_is_the_longest_common_prefix_of_the_sidecar_ids(tmp_path: Path) -> None:
    """Criterion 2 (S derived from the ids, never a count of agreeing positions): plain-03's two id lists agree
    everywhere in their first 1220 positions except position 704. S is 704 = 22*32 (1219 positions agree). With H
    1216: hit_on_prefix 704, hit_beyond_prefix 512, allowed 0, missed 0 -> holds."""
    run = build_run(tmp_path, reference_cases())

    def poke(ids: list[int]) -> list[int]:
        ids[704] = 3_000_000
        return ids

    rewrite_sidecar(run, "plain-03", "on", edit_ids("generate_ids", poke))
    report = harness().build_report(run)
    row = b11_row(report, "plain-03")
    assert (row.get("S"), row.get("hit_on_prefix"), row.get("hit_beyond_prefix")) == (704, 704, 512)
    assert (row.get("allowed_miss"), row.get("missed"), row.get("holds")) == (0, 0, True), row
    assert report["void"] == [], report["void"]


def _drop_line(line: str) -> None:
    return None


def _truncate_line(line: str) -> str:
    return line[: len(line) // 2]


#: criterion 5's voids, each on plain-02: (case edit, sidecar rewrite, arm, strings the void reason must carry)
VOIDS: dict[
    str, tuple[dict[str, Any], Callable[[str], str | None] | None, str, tuple[str, ...]]
] = {
    # 5a: 1226 prefill ids against P 1225
    "5a_prefill_ids_not_P": (
        {},
        edit_ids("prefill_ids", lambda ids: [*ids, 999_999]),
        "on",
        ("1226", "1225"),
    ),
    # 5b: 1279 generate ids against Q 1280
    "5b_generate_ids_not_Q": (
        {},
        edit_ids("generate_ids", lambda ids: ids[:-1]),
        "on",
        ("1279", "1280"),
    ),
    "5c_two_prefills": ({"prefills": 2}, None, "on", ("2",)),
    "5c_no_prefill": ({"prefills": 0}, None, "on", ("0",)),
    "5d_image_digests_differ": (
        {"digests": ("494a9ca4", "0badf00d")},
        None,
        "on",
        ("494a9ca4", "0badf00d"),
    ),
    # 5e: Hp 1200 = 37.5 * 32
    "5e_prefill_hit_not_a_multiple_of_32": ({"prefill": (1200, 1225)}, None, "on", ("1200",)),
    "5f_missing_line": ({}, _drop_line, "on", ()),
    "5f_unparsable_line": ({}, _truncate_line, "on", ()),
    # either arm: the same 5a on the tools-off arm (1588 ids against P 1587)
    "5a_off_arm": (
        {},
        edit_ids("prefill_ids", lambda ids: [*ids, 999_999]),
        "off",
        ("1588", "1587"),
    ),
}


@pytest.mark.parametrize("which", sorted(VOIDS))
def test_unverified_measurement_voids(tmp_path: Path, which: str) -> None:
    """Criterion 5: one unverified (case, arm) voids the run with one ``integrity:`` reason naming the case, the arm
    and the numbers; the run is not valid, the case's row reads "unread", and (tools on) the prefix gate fails. The
    unparsable line cannot be attributed to its case by its own text, so that variant asks only that some reason
    names plain-02 (whose line is then missing). The verified control is
    ``test_b11_rows_carry_the_registered_fields`` (the reference run voids nothing)."""
    change, rewrite, arm, numbers = VOIDS[which]
    cases = reference_cases()
    if change:
        cases = edit(cases, "plain-02", (on if arm == "on" else off)("plain-02", **change))
    run = build_run(tmp_path, cases)
    if rewrite is not None:
        rewrite_sidecar(run, "plain-02", arm, rewrite)
    report = harness().build_report(run)
    reasons = [str(v) for v in report["void"]]
    naming = [v for v in reasons if "plain-02" in v]
    assert naming and all(v.startswith("integrity:") for v in naming), reasons
    if which != "5f_unparsable_line":
        assert len(reasons) == 1, f"criterion 5: exactly one void reason; got {reasons}"
        assert re.search(rf"\b{arm}\b", naming[0]), f"the reason names the arm {arm!r}: {naming[0]}"
        for n in numbers:
            assert re.search(rf"(?<![0-9a-f]){n}(?![0-9a-f])", naming[0]), (
                f"criterion 5: the reason carries {n}: {naming[0]}"
            )
    g = gates_of(report)
    assert g["run_valid"] is False
    assert unread(b11_row(report, "plain-02", arm)), b11_row(report, "plain-02", arm)
    if arm == "on":
        assert g["prefix_hit_on"] is False


@pytest.mark.parametrize(
    ("where", "change", "voids"),
    [
        ("pass1_hit_on", {"shared": 1370, "hit": (1360, 1400), "prefill": (0, 1375)}, True),
        ("prefill_hit_on", {"shared": 1370, "hit": (1344, 1400), "prefill": (1360, 1375)}, True),
        ("pass1_hit_off", {"shared": 1370, "hit": (1360, 1400), "prefill": (0, 1375)}, True),
        ("pass1_hit_1408", {"shared": 1420, "hit": (1408, 1500), "prefill": (0, 1425)}, False),
        ("zero_hits", {"shared": 1370, "hit": (0, 1400), "prefill": (0, 1375)}, False),
    ],
)
def test_granularity_check_flags_an_odd_multiple_of_16(
    tmp_path: Path, where: str, change: dict[str, Any], voids: bool
) -> None:
    """Criterion 5e, both directions. A hit of 1360 = 42.5*32 (a multiple of 16, not of 32) anywhere in the run --
    pass 1's or the prefill's, either arm -- voids it ("cache granularity is not the registered 32"), naming the case
    and 1360. Not voided: H 1408 = 44*32 (S 1420: floor(1420/32)*32 = 1408, Fraction(1408, 1420) >= 9/10, holds), and
    a hit of zero (a cold prefill and pass 1; 0 is a multiple of everything, criterion 5e says "nonzero")."""
    arm = "off" if where.endswith("_off") else "on"
    cases = edit(reference_cases(), "plain-02", (on if arm == "on" else off)("plain-02", **change))
    report = report_of(tmp_path, cases)
    if voids:
        naming = [str(v) for v in report["void"] if "plain-02" in str(v)]
        assert len(report["void"]) == 1 and naming, report["void"]
        assert naming[0].startswith("integrity:") and "1360" in naming[0], naming
        assert gates_of(report)["run_valid"] is False
    else:
        assert report["void"] == [], report["void"]
        row = b11_row(report, "plain-02", arm)
        assert not unread(row), row


@pytest.mark.parametrize("shape", ["4b0bd92", "rule_without_sidecar", "sidecar_without_rule"])
def test_pre_h_t96_run_is_never_scored_passing(tmp_path: Path, shape: str) -> None:
    """Criterion 12, adversarial (a silent fallback to J2). A run without ``regime.prefix_rule == "H-T96"`` or
    without ``prefix_ids.jsonl`` -- here built as 4b0bd92 was, and each half on its own -- whose whole-request hits
    are all >= 0.95 (1216/1280 per tools-on case, which J2 passed) gives ``prefix_hit_on`` False, the gate row reading
    "unread: run predates H-T96". Never True. (No row reads the real 4b0bd92 directory.)"""
    run = build_run(tmp_path, reference_cases(), h_t96=shape != "4b0bd92")
    if shape == "rule_without_sidecar":
        (run / SIDECAR).unlink()
    if shape == "sidecar_without_rule":
        manifest = json.loads((run / "manifest.json").read_text())
        del manifest["regime"]["prefix_rule"]
        (run / "manifest.json").write_text(json.dumps(manifest))
    h = harness()
    report = h.build_report(run)
    assert all(r["hit"] * 20 >= r["query"] * 19 for r in report["prefix_hit_cases"]["on"])
    assert gates_of(report)["prefix_hit_on"] is False
    if shape != "rule_without_sidecar":
        assert key96(report, "prefix_rule") is None
    gate_line = _line(h.render_markdown(report), "| prefix_hit_on |")
    assert "predates H-T96" in gate_line, gate_line


def _cells(md: str, label: str) -> tuple[list[float], list[float]]:
    """The numbers in the tools-on and the tools-off cell of the one table row labelled ``label``."""
    line = _line(md, label)
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    assert len(cells) >= 3, f"criterion 9: {label!r} is not a table row with on/off cells: {line}"

    def nums(cell: str) -> list[float]:
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cell)]

    return nums(cells[-2]), nums(cells[-1])


def _shows(values: list[float], expected: float) -> bool:
    """A rendered share: as a fraction to at least two decimals, or as a percentage."""
    return any(abs(v - expected) <= 0.0051 or abs(v / 100 - expected) <= 0.00051 for v in values)


WHOLE_REQUEST_LABEL = "whole request (J2), reported, not gated since H-T96"


def test_whole_request_numbers_unchanged(tmp_path: Path) -> None:
    """Criterion 7: ``prefix_hit`` and ``prefix_hit_cases`` keep their keys, values and definition (equal to the
    report of the harness at ``BASE_SHA`` on the same directory), and report.md labels them as the whole request.
    Reference run with plain-03 at H 672, Q 1400 (S 1360): on (26*1216 + 672) / (26*1280 + 1400) = 32288/34680 =
    0.9310; off 27*1568 / (27*1600) = 0.98."""
    gr.prove_range(T0096_BASE, T0096_TIP, T0096_MERGE, later=gr.LATER_B)
    run = build_run(tmp_path, prefix_edit(reference_cases(), "plain-03", S=1360, H=672))
    h = harness()
    report = h.build_report(run)
    base_mod = base_harness(tmp_path)
    tip_mod = base_harness(tmp_path / "tip", T0096_TIP)
    base_src = Path(base_mod.__file__ or "").read_text()
    assert base_src != Path(tip_mod.__file__ or "").read_text(), "premise: two blobs"
    base = base_mod.build_report(run)
    tip = tip_mod.build_report(run)
    assert report["prefix_hit"]["on"] == pytest.approx(32288 / 34680), report["prefix_hit"]
    assert report["prefix_hit"]["off"] == pytest.approx(0.98), report["prefix_hit"]
    assert tip["prefix_hit"] == base["prefix_hit"]
    assert tip["prefix_hit_cases"] == base["prefix_hit_cases"]
    assert {"case_id": "plain-03", "hit": 672, "query": 1400} in report["prefix_hit_cases"]["on"]
    on_cell, off_cell = _cells(h.render_markdown(report), WHOLE_REQUEST_LABEL)
    assert _shows(on_cell, 32288 / 34680) and _shows(off_cell, 0.98), (on_cell, off_cell)


def _variants() -> dict[str, list[Case]]:
    ref = reference_cases()
    two_false = edit(ref, "chat-01", on("chat-01", call=_false_call()))
    two_false = edit(two_false, "plain-04", on("plain-04", call=_false_call()))
    return {
        "reference": ref,
        "b11_fails": prefix_edit(ref, "plain-03", S=1360, H=672),
        "no_shared_prefix": prefix_edit(ref, "plain-03", S=0, H=1216, Q=1280, P=1225),
        "beyond_the_prefix": prefix_edit(ref, "plain-03", S=1360, H=1408, Q=1600),
        "four_barge_ins": edit(ref, "key-02", on("key-02", barge_in=False)),
        "two_false_tools": two_false,
        "slow_first_canvas": _with_first_canvas(1150, 1000),
        "executed_when_interrupted": edit(ref, "click-02", on("click-02", execute="interrupted")),
    }


@pytest.mark.parametrize("variant", sorted(_variants()))
def test_prefix_change_moves_no_other_gate(tmp_path: Path, variant: str) -> None:
    """Criterion 11 (negative; a guard that holds before T-0096 and must keep holding): on runs with verified prefix
    data, the 7 other gates, ``void`` and every non-prefix report key equal those of the harness at ``BASE_SHA``; the
    only keys the report may add start with ``prefix``. The variants move the prefix numbers (B11 failing, S 0, hits
    beyond S) and, separately, the other gates (a J18 void, two false tools, a slow first canvas, an interrupted
    execute), so a change that couples them shows here."""
    gr.prove_range(T0096_BASE, T0096_TIP, T0096_MERGE, later=gr.LATER_B)
    run = build_run(tmp_path, _variants()[variant])
    base = base_harness(tmp_path)
    tip = base_harness(tmp_path / "tip", T0096_TIP)
    assert Path(base.__file__ or "").read_text() != Path(tip.__file__ or "").read_text(), (
        "premise: two blobs"
    )
    assert _criterion_11_moves(base, tip, run) == []


def _criterion_11_moves(
    base: ModuleType, tip: ModuleType, run: Path, allowed: tuple[str, ...] = ("prefix",)
) -> list[str]:
    """Criterion 11's predicate over two harness modules on one run (T-0094b: T-0096's own two blobs): the gates other
    than ``prefix_hit_on`` and ``void`` that differ (or are missing), report keys added outside ``allowed`` prefixes,
    keys removed, and keys outside ``allowed`` whose value moved."""
    new, old = tip.build_report(run), base.build_report(run)
    g_new, g_old = tip.gates(copy.deepcopy(new)), base.gates(copy.deepcopy(old))
    out = [
        f"gate {k}" for k in sorted(GATES - {"prefix_hit_on"}) if k not in g_new or k not in g_old
    ]
    out += [
        f"gate {k}"
        for k in sorted(GATES - {"prefix_hit_on"})
        if k in g_new and k in g_old and g_new[k] != g_old[k]
    ]
    if new["void"] != old["void"]:
        out.append("void")
    out += [f"added {k}" for k in sorted(set(new) - set(old)) if not k.startswith(allowed)]
    out += [f"removed {k}" for k in sorted(set(old) - set(new))]
    out += [
        f"moved {k}"
        for k in sorted(set(old) & set(new))
        if k != "run_dir" and not k.startswith(allowed) and old[k] != new[k]
    ]
    return out


def test_the_criterion_11_predicate_has_power(tmp_path: Path) -> None:
    """T-0094b criterion 5 (adversarial): over 1dcc542's parent and 1dcc542 (T-0097) the same predicate on the
    reference run names the replaced Q12 keys (``q12_surface`` removed, ``q12_shown`` added); a predicate that is
    always empty fails here."""
    gr.require_commits(Q12_PRIOR_PARENT, Q12_PRIOR)
    assert gr.is_ancestor(Q12_PRIOR, T0096_BASE)
    run = build_run(tmp_path, reference_cases())
    old = base_harness(tmp_path / "old", Q12_PRIOR_PARENT)
    new = base_harness(tmp_path / "new", Q12_PRIOR)
    moves = _criterion_11_moves(old, new, run)
    assert "removed q12_surface" in moves and "added q12_shown" in moves, moves


def test_the_criterion_11_allowance_is_not_too_strict(tmp_path: Path) -> None:
    """T-0094b criterion 6, too strict: with the ``prefix`` allowance emptied the predicate over T-0096's own pair
    reports exactly the four keys T-0096 added (``prefix_b11``, ``prefix_b11_cases``, ``prefix_block_tokens``,
    ``prefix_rule``); a pair reduced to TIP vs TIP fails here."""
    gr.prove_range(T0096_BASE, T0096_TIP, T0096_MERGE, later=gr.LATER_B)
    run = build_run(tmp_path, reference_cases())
    base = base_harness(tmp_path)
    tip = base_harness(tmp_path / "tip", T0096_TIP)
    assert _criterion_11_moves(base, tip, run, allowed=()) == [
        "added prefix_b11",
        "added prefix_b11_cases",
        "added prefix_block_tokens",
        "added prefix_rule",
    ]


def test_prefix_markdown_rows(tmp_path: Path) -> None:
    """Criterion 9. The reference run with three changes:

    - plain-03 on: S 1360, P 1365, Hp 0, H 672, Q 1600 -> fails (Hs 672, allowed miss 1360 mod 32 = 16);
    - chat-02 on: S 1360, P 1370, Hp 1344, H 1408, Q 1600 -> holds (beyond 48);
    - chat-02 off: S 1500, P 1503, Hp 0, H 1568, Q 1600 -> holds (beyond 68).

    Expected, on / off:

    - gate row: "B11-prefix (H-T96): 1 tools-on case(s) failing or unread", naming plain-03 with 1360, 672, 16, and
      not chat-02;
    - B11-prefix pooled: on (25*1216 + 672 + 1360) / (25*1220 + 1360 + 1360) = 32432/33220 = 0.9763; off
      (26*1568 + 1500) / (26*1580 + 1500) = 42268/42580 = 0.9927;
    - whole request (J2): on (25*1216 + 672 + 1408) / (25*1280 + 2*1600) = 32480/35200 = 0.9227; off 0.98;
    - hit beyond the prefix, totals: on 48, off 68;
    - prefill pre-warmed share p50 (nearest rank, 14th of 27): on 1184/1225 = 0.9665 (one 0, 25 x 0.9665, one
      1344/1370 = 0.9810); off 1024/1587 = 0.6452 (one 0, 26 x 0.6452);
    - prefill tail P - S: on min 5, max 10; off min 3, max 7."""
    cases = prefix_edit(reference_cases(), "plain-03", S=1360, P=1365, H=672, Q=1600)
    cases = prefix_edit(cases, "chat-02", S=1360, P=1370, Hp=1344, H=1408, Q=1600)
    cases = prefix_edit(cases, "chat-02", "off", S=1500, P=1503, H=1568, Q=1600)
    h = harness()
    md = h.render_markdown(h.build_report(build_run(tmp_path, cases)))

    gate = _line(md, "| prefix_hit_on |")
    assert "B11-prefix (H-T96): 1 tools-on case(s) failing or unread" in gate, gate
    assert "chat-02" not in gate, gate
    named = gate.split("plain-03", 1)
    assert len(named) == 2, f"criterion 9: the gate row names plain-03: {gate}"
    assert {1360, 672, 16} <= set(_ints_after(gate, "plain-03")), gate

    on_, off_ = _cells(md, "B11-prefix pooled")
    assert _shows(on_, 32432 / 33220) and _shows(off_, 42268 / 42580), (on_, off_)
    on_, off_ = _cells(md, WHOLE_REQUEST_LABEL)
    assert _shows(on_, 32480 / 35200) and _shows(off_, 0.98), (on_, off_)
    on_, off_ = _cells(md, "hit beyond the prefix")
    assert 48 in on_ and 68 in off_, (on_, off_)
    on_, off_ = _cells(md, "pre-warmed")
    assert _shows(on_, 1184 / 1225) and _shows(off_, 1024 / 1587), (on_, off_)
    on_, off_ = _cells(md, "prefill tail")
    assert {5, 10} <= set(on_) and {3, 7} <= set(off_), (on_, off_)


@pytest.mark.parametrize(
    ("field_", "criterion"),
    [
        ("prefill_query_tokens", "5a"),
        ("prefix_query_tokens", "5b"),
        ("prefills_before_pass1", "5c"),
    ],
)
def test_a_missing_count_voids_and_is_never_skipped(
    tmp_path: Path, field_: str, criterion: str
) -> None:
    """Ruling M6's test gaps (reviewer A): plain-02 (tools on) keeps both id lists in ``prefix_ids.jsonl`` and its
    pass-1 hit H 1216, but one count is ``None`` in the manifest: P (5a), Q (5b) or ``prefills_before_pass1`` (5c).
    A count that cannot be checked is not a check that passed: the run voids with an ``integrity:`` reason naming
    plain-02, the row reads "unread", and the prefix gate fails."""
    cases = edit(reference_cases(), "plain-02", on("plain-02", manifest=((field_, None),)))
    report = report_of(tmp_path, cases)
    naming = [str(v) for v in report["void"] if "plain-02" in str(v)]
    assert naming and all(v.startswith("integrity:") for v in naming), (criterion, report["void"])
    g = gates_of(report)
    assert g["run_valid"] is False and g["prefix_hit_on"] is False, g
    assert unread(b11_row(report, "plain-02")), b11_row(report, "plain-02")
