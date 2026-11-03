"""T-0220: the GPU check of T-0215, shared-screen handoff with ``--search off`` and no ``google_search`` declared,
harness ``scripts/gpu/t0220_handoff_nosearch.py`` (unit; CPU; no engine, no socket, no browser).
docs/tasks/T-0220.md A1-A8, N1 and "Tests to write".

BASE 3846e7e0 (main; ``Merge T-0215:`` 5237da2d is an ancestor). The rows that need the harness carry
``pending(task="T-0220")`` until ``/implement``; a missing file is a named assertion ("does not exist yet") from
:func:`harness`, never an import error. Rows that need no harness (N1's byte pins, the eval's import fence) are
active and green at BASE by design (negative criteria).

**The harness surface these rows use** (the orchestrator's interface names of 2026-10-31; the signatures below are
the test-author's reading, reported to the orchestrator; a thin adapter here may reshape a call, never compute a
metric)

Modules and constants
    ``A`` (T-0211's harness loaded by path under a fixed ``sys.modules`` name; ``A.C`` is T-0208's, ``A.D``
    T-0205's); ``HARD_LIMIT_S``, ``RUN_BUDGET_S``; ``ANCHOR`` (names both anchor commits); ``PROCESS_START_S`` and a
    module-level ``time`` name the budget clock reads (the rows patch both on the harness and on ``A``).
A2
    ``reference_kit(profile)`` builds the reference through ``cli.load_tool_kit(profile, search=False, **opts)`` with
    ``opts = A.kit_options(profile)`` read at call time (a cache, if any, exposes ``cache_clear``); ``nosearch_kit(kit)``
    reads the reference for the lean profile and returns ``dataclasses.replace(kit, tools=, tools_version=,
    tool_set=)``, raising when A2's checks fail.
A3/A4
    No public P-A function is called (``pa_problems``/``pa_check`` signatures are not fixed): P-A and every refusal
    run through ``main(["--base-url", URL, "--out-dir", DIR])``.
A5-A8 and the report
    ``order(seed)``; ``async _run_all(run)`` (or ``_run_all(run, turns)``), which calls
    ``A.C._one_turn(run, hid, arm, repeat, search_kit=nosearch_kit)``; ``finish_turn(out, hid, arm, repeat, trial,
    steps, seam_versions=None, names=(), sha="") -> (trial, steps)`` (``seam_versions`` given here as a list of
    the versions the seam handed out); ``write_report(out, probes, manifest, stopped)`` writing ``report.json`` and
    ``report.md`` in the shapes the eval module's docstring lists (``label``, ``violations``,
    ``undeclared_search_calls``, ``used_browser``, ``no_browser_claim``, ``reading = {reading, mark, line}``);
    ``main``.

The synthetic manifest (``_manifest``) holds only what the eval reads plus T-0208's regime keys; ``write_report``
must read it with ``.get`` (no other manifest key is promised).

Readings and gaps (reported to the orchestrator, not resolved silently)

1. **A6 (d) versus ``A.C._one_turn`` as written.** ``_one_turn`` records ``trial["tools_version"]`` from the kit
   ``cli.load_tool_kit`` gave *before* ``search_kit`` runs, so a runner's trial carries the **on-set** sha. Ruling
   R-2 (orchestrator, 2026-10-31): ``finish_turn(..., seam_versions=, names=, sha=)`` sets ``tools_version`` to the
   version the seam handed out (the runner's own kept as ``tools_version_loaded``); (d) applies only when the turn
   sent a request, (e) only with a turn line. Hand trials here carry the runner's on-set sha and the seam's version
   goes in through ``seam_versions`` (the on-set sha there is the seeded (d) fault).
2. Ruling R-3: a raised shared turn's line has ``ph3: None`` (not ``A.C._not_run``'s ``False``), so it is errored,
   never void.
3. A5: ``RUN_BUDGET_S = 1320`` cannot bind before the hard-limit check (a turn starts only at ``elapsed <= 1260``),
   so no row can tell it from an absent run-budget check; the constant and the 1260/1261 boundary are pinned.
4. A not-run turn carries no A6 evidence and is no violation (it counts errored), as in T-0214.
5. The A1 pin rows alter a runner's bytes through ``Path.read_bytes``/``read_text`` and through every
   ``_git`` the harness holds (``hash-object``, ``rev-parse <rev>:<path>``, ``ls-tree``, ``ls-files``, ``show``,
   ``cat-file``); a harness that hashes through ``open()`` directly is not reached by them.
6. A3's ``manifest.json`` contents are pinned only by the eval's manifest row over the synthetic dir (``main``'s
   ``_measure`` is never reached on CPU).

Regime: CPU, pure, hand-built session logs and stubbed ``_one_turn``/``main``; no engine; n = 1.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import json
import random
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0201_ab_harness as t201
from tests.unit import test_t0205_demo_harness as t205
from tests.unit import test_t0208_handoff_harness as t208
from tests.unit import test_t0208_handoff_judge as J
from tests.unit import test_t0210_threejs_harness as t210

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0220_handoff_nosearch.py"
EVAL = REPO / "tests" / "eval" / "test_t0220_handoff_nosearch.py"
#: the task's base (main at ``T-0215: merged``), a literal
BASE = "3846e7e0bd2626d6dde415a8f23cff9c07c03714"
#: A1: the read-only runners and their git blob shas at BASE (``git rev-parse 3846e7e0:<path>``)
RUNNER_BLOBS = {
    "scripts/gpu/t0208_handoff_run.py": "56c0ef8084629187cf57f47ddbd788008af39c9e",
    "scripts/gpu/t0211_guidance_ab.py": "41296154114b7ffbbe1e6da422d2034d4c337adc",
    "scripts/gpu/t0205_demo_run.py": "c65be5949dc5ce38b971704f46a53133f760fc65",
}
#: N1: their tests, byte-identical too
RUNNER_TESTS = (
    "tests/unit/test_t0208_handoff_harness.py",
    "tests/unit/test_t0211_ab_harness.py",
    "tests/unit/test_t0205_demo_harness.py",
)
#: the real merges on main (``git log --merges``), and lookalikes
MERGE_T0208_B = (
    "43eb2f9a9148f990c99d4db0a416fec5ee076ea2 Merge T-0208-b: loopback shared-screen handoff scenarios H1-H3 (drawn "
    "images, judge, real ingress)"
)
MERGE_T0211_A = (
    "c3b0c432f67f4cb6bbb9b5575bfbb08077205ec4 Merge T-0211-a: T-0208 P-H3 adopts the P-R3 switch rule; _one_turn "
    "search_kit seam"
)
MERGE_T0211_B = (
    "8e635a14117a6861fc0579e7b4cd196c52301056 Merge T-0211-b: live guidance v4 (cite the source page; act in your own "
    "browser from a shared screen), opt-in; replay_live_guidance.py"
)
MERGE_T0215 = "5237da2d4f67fbc53beaf8f5b9a3972cce1f93dd Merge T-0215: --search off in live does not offer google_search"
LOOKALIKES_T0215 = (
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'T-0215-search-off' into main",
    "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f Merge T-0215b: something else",
)
OTHERS = (
    *t208.OTHERS,
    "cf2a915367362d7f588d4b61e25fe679596382b7 Merge T-0208-c: shared-screen handoff GPU harness (H1-H3 one-source "
    "probes, completion-check audit, locked wrapper)",
    t208.MERGE_A2,
)
HIDS = ("H1", "H2", "H3")
ARMS = ("shared", "typed")
UNBLOCKS = "withdrawal unblocks the handoff"
NO_CHANGE = "no change"
MIXED = "mixed"
READINGS = (UNBLOCKS, NO_CHANGE, MIXED)
ON_FILES = {
    (False, False): "tools_v1.json",
    (True, False): "tools_v1_overview.json",
    (False, True): "tools_v1_wait.json",
    (True, True): "tools_v1_overview_wait.json",
}
COMBOS = list(ON_FILES)
COMBO_IDS = ["v1", "overview", "wait", "overview+wait"]
SHARED_LABEL_SCREEN = t208.SHARED_LABEL_SCREEN
SHARED_FRAME_REASON = t208.SHARED_FRAME_REASON
L_NOW = t208.L_NOW
FAILED = {"page": "home", "cart": [], **J.SAFE}
SYS_HEAD = (
    "You are Gearvis, a local assistant that can see the user's screen and hear their voice. Answer what you can "
    "actually see; say so when you cannot.\n"
)
SYS_TAIL = "\nKeep answers short."
Q = '<|"|>'


# =====================================================================================================
# helpers
# =====================================================================================================


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0220 A1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0220_handoff_nosearch_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), f"T-0220: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def _A() -> ModuleType:
    a = api("A")
    assert isinstance(a, ModuleType) and str(a.__file__).endswith("t0211_guidance_ab.py"), a
    return a


@functools.cache
def _eval() -> ModuleType:
    from importlib import import_module

    return import_module("tests.eval.test_t0220_handoff_nosearch")


def _held(h: ModuleType) -> list[ModuleType]:
    """The harness and every ``scripts/gpu`` module it holds by name, four levels down."""
    seen: dict[int, ModuleType] = {id(h): h}
    frontier = [h]
    for _ in range(4):
        nxt = []
        for mod in frontier:
            for v in vars(mod).values():
                if (
                    isinstance(v, ModuleType)
                    and id(v) not in seen
                    and "/scripts/gpu/" in str(getattr(v, "__file__", "") or "")
                ):
                    seen[id(v)] = v
                    nxt.append(v)
        frontier = nxt
    return list(seen.values())


def _patch_name(mp: pytest.MonkeyPatch, h: ModuleType, name: str, value: Any) -> None:
    for mod in _held(h):
        if hasattr(mod, name):
            mp.setattr(mod, name, value)


def _patch_object(mp: pytest.MonkeyPatch, h: ModuleType, original: Any, value: Any) -> int:
    n = 0
    for mod in _held(h):
        for k, v in list(vars(mod).items()):
            if v is original:
                mp.setattr(mod, k, value)
                n += 1
    return n


def _lean() -> Any:
    from local_astra.config import load

    return load().profiles["lean"]


def _toolset() -> ModuleType:
    from local_astra.toolcall import tools

    return tools


def _on_set(ov: bool, wait: bool) -> Any:
    tools = _toolset()
    return tools.load(Path(tools.__file__).with_name(ON_FILES[(ov, wait)]))


def _ns_path(ov: bool, wait: bool) -> Path:
    from local_astra import cli

    return Path(cli._NOSEARCH_PATHS[(ov, wait)])


def _ns_set(ov: bool, wait: bool) -> Any:
    return _toolset().load(_ns_path(ov, wait))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tools_field(s: Any) -> list[dict[str, Any]]:
    from local_astra.toolcall import native

    return [dict(t) for t in native.tools_field(s)]


def _head_combo() -> tuple[bool, bool]:
    from local_astra import cli

    ns = _A().D._live_args()
    lean = _lean()
    return (
        bool(cli.page_overview_of(lean, ns, "live").page_overview),
        bool(cli.wait_tool_of(ns, "live")),
    )


def _patch_detok(mp: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``HostDetokenizer.load`` gives T-0088a's subset (the real 31 MB file is never read)."""
    from local_astra.toolcall.detok import HostDetokenizer
    from tests.unit._tools_switch_support import subset_detok

    det = subset_detok(tmp_path / "detok")
    mp.setattr(HostDetokenizer, "load", classmethod(lambda cls: det))


def _set_combo(mp: pytest.MonkeyPatch, ov: bool, wait: bool) -> None:
    from local_astra import cli

    real = cli.page_overview_of
    mp.setattr(
        cli,
        "page_overview_of",
        lambda *a, **k: dataclasses.replace(real(*a, **k), page_overview=ov),
    )
    mp.setattr(cli, "wait_tool_of", lambda *a, **k: wait)


def _reset_reference(h: ModuleType) -> None:
    fn = getattr(h, "reference_kit", None)
    clear = getattr(fn, "cache_clear", None)
    if callable(clear):
        clear()


class _Exec:
    """The turn's executor stand-in (identity only; never driven)."""

    mode = "live"


_EXEC = _Exec()


def _turn_kit(ov: bool, wait: bool) -> Any:
    """The kit ``A.C._one_turn`` gets from ``cli.load_tool_kit`` (its exact kwargs shape: the options only when
    on), before its ``search_kit`` runs."""
    from local_astra import cli

    kw: dict[str, Any] = {}
    if ov:
        kw["page_overview"] = True
    if wait:
        kw["wait_tool"] = True
    kit, _ = cli.load_tool_kit(_lean(), live_browser=_EXEC, **kw)
    return kit


def _walk(obj: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    out = [(path, obj)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _walk(v, (*path, i))
    return out


def _undeclared(report: dict[str, Any]) -> int:
    vals = [
        v
        for p, v in _walk(report)
        if p
        and p[-1] == "undeclared_search_calls"
        and isinstance(v, int)
        and not isinstance(v, bool)
    ]
    assert vals, "T-0220 A6: report.json has no undeclared_search_calls count"
    return max(vals)


def _line(report: dict[str, Any]) -> str:
    line = _eval()._reading_line(report)
    assert isinstance(line, str) and line, (
        "T-0220 A8: the report has no reading line",
        list(report),
    )
    return line


def _order0() -> list[tuple[str, str, int]]:
    """A5's order, by hand: T-0208's handoff-major list shuffled once by ``random.Random(0)``."""
    turns = [(h, a, r) for h in HIDS for a in ARMS for r in range(3)]
    random.Random(0).shuffle(turns)
    return turns


# =====================================================================================================
# hand-built turns: session logs, turn lines, trials and steps
# =====================================================================================================


def _txt(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


SHARED_IMG = {
    "type": "image_url",
    "image_url": {"redacted": "image", "bytes": 81234, "digest": "5e1f00aa"},
}
BROWSER_IMG = {
    "type": "image_url",
    "image_url": {"redacted": "image", "bytes": 40211, "digest": "77c0ffee"},
}


def _spec(hid: str, arm: str, r: int, **kw: Any) -> dict[str, Any]:
    """One turn: ``acts`` (an executed browser action), ``success`` (the golden state and words), ``search``
    (``None``, ``"refused"`` or ``"executed"``: a ``google_search`` attempt at step 1), ``fault`` (one A6 fault),
    ``safety`` (a non-loopback connection), ``not_run`` (``A.C._not_run``'s line)."""
    return {
        "hid": hid,
        "arm": arm,
        "repeat": r,
        "acts": True,
        "success": True,
        "search": None,
        "fault": None,
        "safety": False,
        "not_run": False,
        **kw,
    }


def _call_text(name: str, args: dict[str, Any]) -> str:
    inner = ",".join(
        f"{k}:{Q}{v}{Q}" if isinstance(v, str) else f"{k}:{v}" for k, v in args.items()
    )
    return f"<|tool_call>call:{name}{{{inner}}}<tool_call|>"


def _turn_files(out: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write one turn's session log and append its turn line to ``turns.jsonl`` (what the real ``Session`` does
    during the turn); return the runner's trial and step lines (before ``finish_turn``)."""
    from local_astra import cli
    from local_astra.pages import handoff_tasks
    from local_astra.toolcall import live, loop

    hid, arm, r, fault = spec["hid"], spec["arm"], spec["repeat"], spec["fault"]
    ov, wait = _head_combo()
    ns, on = _ns_set(ov, wait), _on_set(ov, wait)
    ns_tools, on_tools = _tools_field(ns), _tools_field(on)
    guidance = cli._tool_guidance("v3", True, wait, search=False)
    assert isinstance(guidance, str) and "google_search" not in guidance, (
        "premise: v3 search-off text"
    )
    system = SYS_HEAD + guidance + SYS_TAIL
    if fault == "system-names-google-search":
        system += " Use google_search for facts you do not know."
    sid = f"t0220{hid}{arm}{r}"
    name = f"turn-{hid}-{arm}-{r}.jsonl"
    row = next(x for x in handoff_tasks.HANDOFFS if x.handoff_id == hid)
    text = row.request if arm == "shared" else row.control_request
    first_img, label = (
        (SHARED_IMG, SHARED_LABEL_SCREEN) if arm == "shared" else (BROWSER_IMG, L_NOW)
    )
    recs: list[dict[str, Any]] = []

    def add(kind: str, data: dict[str, Any], turn: Any = "t1") -> None:
        recs.append(
            {"seq": len(recs), "t_ms": 10 * len(recs), "kind": kind, "turn_id": turn, "data": data}
        )

    def req(
        phase: str,
        rid: str,
        k: int,
        user: list[dict[str, Any]],
        assistant: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [_txt(system)]},
            {"role": "user", "content": user},
        ]
        if assistant is not None:
            messages.append({"role": "assistant", "content": [_txt(assistant)]})
        data: dict[str, Any] = {
            "phase": phase,
            "request_id": rid,
            "attempt_id": f"{sid}-{rid}-{k}",
            "options": None,
            "xargs": None,
            "prompt": {
                "messages": messages,
                "tools": copy.deepcopy(tools if tools is not None else ns_tools),
                "thinking": True,
                "assists": "pixels",
                "prefix_len": 0,
                "continue_final_message": assistant is not None,
            },
        }
        if phase == "tool_step":
            data["step"] = int(rid.rsplit("-s", 1)[1])
        add("llm.request", data)

    add("log.started", {"pid": 4242, "path": name}, None)
    add("machine.step", {"event": {"type": "user.text", "text": text}, "commands": []})
    req("prefill", "t1-prefill", 1, [first_img])
    req("generate", "t1-gen", 2, [first_img, _txt(label), _txt(text)])
    thought = f"The user wants help with {hid}."
    if spec["search"] is not None:
        # adversarial for row (b): the model's own thought names the tool (never a system message)
        thought += " I could use google_search to look the part up first."
    add(
        "llm.commit",
        {
            "request_id": "t1-gen",
            "attempt_id": f"{sid}-t1-gen-2",
            "segment": "thought",
            "text": thought,
        },
    )

    # the plan: (call name or None, status, reason, arguments) per step
    plan: list[tuple[str | None, str | None, str | None, dict[str, Any]]] = []
    if spec["search"] == "refused":
        plan.append(("google_search", "refused", live.NOT_DECLARED, {"query": f"{hid} part"}))
    elif spec["search"] == "executed":
        plan.append(("google_search", "executed", None, {"query": f"{hid} part"}))
    if spec["acts"]:
        plan.append(("click_at", "executed", None, {"x": 358, "y": 357}))
    elif arm == "shared":
        plan.append(("click_at", "refused", SHARED_FRAME_REASON, {"x": 300, "y": 300}))
    plan.append((None, None, None, {}))

    steps: list[dict[str, Any]] = []
    continuation = ""
    for k, (call, status, reason, args) in enumerate(plan, start=1):
        rid = "t1-gen" if k == 1 else f"t1-gen-s{k}"
        if k > 1:
            user = [BROWSER_IMG, _txt(L_NOW)]
            if fault == "search-not-connected" and k == 2:
                user.append(
                    _txt(
                        f"<|tool_response>response:google_search{{note:{Q}{loop.SEARCH_NOT_CONNECTED}{Q}}}<tool_response|>"
                    )
                )
            tools = on_tools if fault == "declares-google-search" and k == 2 else None
            req("tool_step", rid, k + 1, user, assistant=continuation, tools=tools)
        cid = None
        if call is not None:
            cid = "t1-call" if k == 1 else f"t1-call-s{k}"
            add(
                "tool.call",
                {
                    "status": status,
                    "call_id": cid,
                    "source": "draft",
                    "name": call,
                    "arguments": dict(args),
                    "reason": reason,
                },
            )
            if status == "executed":
                response = (
                    f"<|tool_response>response:{call}{{status:{Q}executed{Q}}}<tool_response|>"
                    if call != "google_search"
                    else f"<|tool_response>response:google_search{{results:{Q}a loopback page{Q}}}<tool_response|>"
                )
                add(
                    "tool.result",
                    {"call_id": cid, "step": k, "status": "executed", "response": response},
                )
            continuation += _call_text(call, args)
        steps.append(
            {
                "task": hid,
                "arm": arm,
                "repeat": r,
                "step": k,
                "http_status": 200,
                "retries": [],
                "status": status,
                "reason": reason,
                "call_id": cid,
                "call_name": call,
                "executed_name": call if status == "executed" else None,
                "arguments": dict(args),
                "attempt_id": f"{sid}-{rid}-{k + 1}",
                "error": None,
            }
        )

    if spec["success"] and spec["acts"]:
        state, voiced = copy.deepcopy(J.GOLD[hid][0]), [J.GOLD[hid][1]]
    elif arm == "shared" and not spec["acts"]:
        state, voiced = (
            copy.deepcopy(FAILED),
            ["I cannot act on your screen; I could not finish that."],
        )
    else:
        state, voiced = copy.deepcopy(FAILED), [f"I could not finish that ({hid})."]
    add(
        "machine.step",
        {"event": {}, "commands": [{"type": "tts.speak", "text": voiced[0], "source": voiced[0]}]},
    )
    sha12 = (on.version if fault == "regime-tools-on-sha" else ns.version)[:12]
    line = {
        "turn_id": "t1",
        "session_id": sid,
        "regime": {
            "profile": "lean",
            "canvas_length": 256,
            "max_denoising_steps": 48,
            "thinking": True,
            "git_commit": "c0ffee2" + "0" * 33,
            "seed": None,
            "tools": f"native:{sha12}:live:loop20",
        },
        "input": "text",
        "outcome": "spoken",
        "session_log": str(out / "sessions" / name),
        "extras": {
            "tool_loop_end": "final",
            "vision_source": "shared" if arm == "shared" else "browser",
            "browser_egress_blocked": [],
        },
    }
    add("turn.record", line)
    add("log.stopped", {"reason": "session closed", "records": len(recs) + 1, "dropped": 0}, None)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    (out / "sessions" / name).write_text(
        "".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8"
    )
    with (out / "turns.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")

    ok = bool(spec["success"] and spec["acts"])
    trial: dict[str, Any] = {
        "task": hid,
        "arm": arm,
        "repeat": r,
        "state": state,
        "voiced": voiced,
        "shown": "",
        "clips": [],
        "success": ok,
        "exception": None,
        "timed_out": False,
        "non_loopback": ["10.0.0.1:443"] if spec["safety"] else [],
        "browser_egress_blocked": [],
        "url_leaks": [],
        "sentinel_requests": [],
        "log_complete": True,
        "turn_id": "t1",
        "session_id": sid,
        "tool_steps": len(plan),
        "tool_steps_executed": sum(1 for p in plan if p[1] == "executed"),
        "tool_loop_end": "final",
        "ph3": True if arm == "shared" else None,
        "completion_checks": 0,
        "completion_check_outcomes": [],
        "step1_thought": thought,
        # the runner's own (A.C._one_turn records the kit it loaded, before the seam); R-2: finish_turn restamps it
        "tools_version": on.version,
        "wall_s": 12.0 + r,
        "engine_up_after": True,
        "resolver_calls": 0,
        "connector_calls": 0,
        "egress_log": [],
        "session_log": name,
    }
    return trial, steps


def _manifest(stopped: str | None = None) -> dict[str, Any]:
    """The manifest fields the eval reads (A3) and T-0208's regime keys, from HEAD's cli."""
    from local_astra import cli

    A = _A()
    lean = _lean()
    ov, wait = _head_combo()
    ns = _ns_set(ov, wait)
    guidance = cli._tool_guidance("v3", True, wait, search=False)
    return {
        "regime": {
            "commit": "c0ffee2" + "0" * 33,
            "profile": "lean",
            "canvas_length": 256,
            "max_denoising_steps": 48,
            "thinking": True,
            "engine_version": "vllm-T0220-0.29.0",
            "tools": f"native:{ns.version[:12]}:live:loop20",
            "turns": 18,
            "concurrency": 1,
            "order_seed": 0,
            "engine_seed": None,
            "turn_timeout_s": 300,
            "hard_limit_s": 1800,
            "run_budget_s": 1320,
            "input": "typed (UserText) T0220; fake STT/TTS",
        },
        "live_default": A.C.live_default(lean),
        "search": False,
        "declared_names": list(ns.names),
        "tools_sha256": {"on": A.D.cli_kit_sha256(ov, wait), "nosearch": ns.version},
        "tool_guidance_sha256": hashlib.sha256(str(guidance).encode("utf-8")).hexdigest(),
        "stopped": stopped,
    }


def _finish(
    out: Path,
    hid: str,
    arm: str,
    r: int,
    trial: dict[str, Any],
    steps: list[dict[str, Any]],
    seam: str | None = None,
) -> Any:
    """``finish_turn`` as ``_run_all`` calls it: the six positional arguments and (R-2) the versions the seam
    handed out, the reference names and the no-search sha."""
    fn = api("finish_turn")
    params = inspect.signature(fn).parameters
    assert len(params) >= 6, (
        f"T-0220: finish_turn(out, hid, arm, repeat, trial, steps, ...) expected, not {list(params)}"
    )
    ov, wait = _head_combo()
    ns = _ns_set(ov, wait)
    extra = {
        "seam_versions": [seam] if seam is not None else [],
        "names": tuple(ns.names),
        "sha": ns.version,
    }
    got = fn(out, hid, arm, r, trial, steps, **{k: v for k, v in extra.items() if k in params})
    assert isinstance(got, tuple) and len(got) == 2, got
    return got


def _write_run(
    out: Path, specs: list[dict[str, Any]], stopped: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """A run dir written with the harness's ``finish_turn`` and ``write_report`` over hand turns; the report."""
    A = _A()
    G = A.G
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    (out / "turns.jsonl").touch()
    G._write_json(out / "order.json", [[s["hid"], s["arm"], s["repeat"]] for s in specs])
    manifest = _manifest(stopped)
    G._write_json(out / "manifest.json", manifest)
    trials, steps = [], []
    for s in specs:
        seam: str | None = None
        if s["not_run"]:
            trial, st = (
                A.C._not_run(
                    s["hid"], s["arm"], s["repeat"], "harness: RuntimeError('T-0220 rig: died')"
                ),
                [],
            )
            if s["arm"] == "shared":
                trial["ph3"] = None  # R-3
        else:
            trial, st = _turn_files(out, s)
            ov, wait = _head_combo()
            bad = s["fault"] == "wrong-tools-version"
            seam = (_on_set(ov, wait) if bad else _ns_set(ov, wait)).version
        trial, st = _finish(
            out, s["hid"], s["arm"], s["repeat"], dict(trial), [dict(x) for x in st], seam
        )
        trials.append(trial)
        steps += st
    G._append_jsonl(out / "trials.jsonl", trials)
    G._append_jsonl(out / "steps.jsonl", steps)
    probes = t208._probes()
    G._write_json(out / "probes.json", probes)
    api("write_report")(out, probes, manifest, stopped)
    return out, json.loads((out / "report.json").read_text(encoding="utf-8"))


def _full(
    k: int, *, safety: int = 0, not_run_typed: int = 0, search_refused: int = 0
) -> list[dict[str, Any]]:
    """18 turns in A5's order: ``k`` of the 9 shared turns act (sorted by handoff and repeat), the rest refuse the
    shared frame and answer; every typed turn acts and succeeds, except the first ``not_run_typed`` (not run) and
    the first ``safety`` (a non-loopback connection); the first ``search_refused`` shared turns try ``google_search``
    first (refused, not declared)."""
    shared = sorted((h, r) for h, a, r in _order0() if a == "shared")
    typed = sorted((h, r) for h, a, r in _order0() if a == "typed")
    acting = set(shared[:k])
    searching = set(shared[:search_refused])
    nr = set(typed[:not_run_typed])
    unsafe = set(typed[not_run_typed : not_run_typed + safety])
    out = []
    for h, a, r in _order0():
        if a == "shared":
            out.append(
                _spec(
                    h,
                    a,
                    r,
                    acts=(h, r) in acting,
                    success=(h, r) in acting and r != 2,
                    search="refused" if (h, r) in searching else None,
                )
            )
        else:
            out.append(_spec(h, a, r, not_run=(h, r) in nr, safety=(h, r) in unsafe))
    return out


# =====================================================================================================
# A1: reuse, read-only; N1
# =====================================================================================================


def test_reuse_runner_objects() -> None:
    """A1: ``A`` is T-0211's harness under a fixed ``sys.modules`` name; ``A.C`` is T-0208's and the only copy of it
    among the modules the harness holds; ``A.D`` is T-0205's; ``used_browser``/``no_browser_claim`` and T-0208's
    reused functions, where the harness names them, are the runners' own objects."""
    h, A = harness(), _A()
    assert sys.modules.get(A.__name__) is A and A.__name__ != "__main__", A.__name__
    assert str(A.C.__file__).endswith("t0208_handoff_run.py"), A.C.__file__
    assert str(A.D.__file__).endswith("t0205_demo_run.py"), A.D.__file__
    copies = [
        m for m in _held(h) if str(getattr(m, "__file__", "")).endswith("t0208_handoff_run.py")
    ]
    assert copies and all(m is A.C for m in copies), [m.__name__ for m in copies]
    for name in ("used_browser", "no_browser_claim"):
        if hasattr(h, name):
            assert getattr(h, name) is getattr(A, name), f"A1: {name} is not A's own"
    for name in (
        "arm_counts",
        "step1_action",
        "probe_verdicts",
        "run_probes",
        "probe_ph1",
        "probe_ph2",
    ):
        if hasattr(h, name):
            assert getattr(h, name) is getattr(A.C, name), f"A1: {name} is not A.C's own"
    text = HARNESS.read_text(encoding="utf-8")
    assert "t0211_guidance_ab.py" in text, "A1: T-0211's harness is not loaded by path"
    assert "_one_turn" in text and "search_kit" in text, (
        "A1: turns do not go through A.C._one_turn's seam"
    )


#: A1's reused functions; the harness must not ``def`` them
REUSED = (
    "used_browser",
    "no_browser_claim",
    "arm_counts",
    "step1_action",
    "probe_verdicts",
    "run_probes",
    "probe_ph1",
    "probe_ph2",
    "_one_turn",
    "_not_run",
)
RUNNER_NAMES = {"A", "C", "D", "G", "T", "P", "F", "R"}


def _root(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute | ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def test_never_assigns_runner_attributes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1 (AST and at run time): no assignment to, or ``setattr``/``delattr`` on, an attribute of ``A``/``A.C``/
    ``A.D`` (or any runner name); no ``def`` of a reused function; ``vars(A)``, ``vars(A.C)``, ``vars(A.D)``,
    ``vars(A.G)`` identical by identity before and after ``nosearch_kit`` and a stub turn through ``_run_all``."""
    harness()
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    bad = [
        ast.unparse(t)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign)
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(t, ast.Attribute) and _root(t) in RUNNER_NAMES
    ]
    assert bad == [], f"A1: the harness writes into an imported runner: {bad}"
    calls = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in ("setattr", "delattr")
        and n.args
        and _root(n.args[0]) in RUNNER_NAMES
    ]
    assert calls == [], calls
    redefined = sorted(
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name in REUSED
    )
    assert redefined == [], f"A1: the harness redefines reused functions: {redefined}"

    h, A = harness(), _A()
    _patch_detok(monkeypatch, tmp_path)
    start = 70.0
    run, clock = _run(tmp_path, monkeypatch, start)
    _stub_turns(monkeypatch, run.out, clock, start, [5.0, 10.0])
    mods = (A, A.C, A.D, A.G)
    before = [dict(vars(x)) for x in mods]
    _reset_reference(h)
    h.nosearch_kit(_turn_kit(*_head_combo()))
    _call_run_all(h, run, [tuple(x) for x in h.order(0)][:1])
    for x, snap in zip(mods, before, strict=True):
        now = vars(x)
        changed = sorted(k for k in set(snap) | set(now) if snap.get(k, ...) is not now.get(k, ...))
        assert changed == [], f"A1: {x.__name__} attributes changed: {changed}"


def test_pins_are_literal_base_blobs() -> None:
    """A1: the three runners' git blob shas are literals in the harness, and they are BASE's (computed here with
    ``git rev-parse 3846e7e0:<path>``, not read from the harness)."""
    harness()
    text = HARNESS.read_text(encoding="utf-8")
    for rel, blob in RUNNER_BLOBS.items():
        got = _git_out("rev-parse", f"{BASE}:{rel}")
        if got is None:
            pytest.skip(f"{BASE[:8]} is not in this checkout")
        assert got.strip() == blob, (rel, got)
        assert blob in text, f"A1: the harness does not pin {rel} at blob {blob}"


def _git_out(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_show(rev: str, rel: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:{rel}"], check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.mark.parametrize("rel", [*RUNNER_BLOBS, *RUNNER_TESTS])
def test_runners_unchanged(rel: str) -> None:
    """N1 (negative criterion, green at BASE by design): the T-0208, T-0211 and T-0205 harnesses and their unit
    tests equal their bytes at BASE 3846e7e0 (a literal). Skipped when that commit is absent. (cli.py, session.py
    and thresholds.py are N1's too; a later task may edit them, so they are the reviewer's BASE..TIP diff, not a
    row here.)"""
    pinned = _git_show(BASE, rel)
    if pinned is None:
        pytest.skip(f"{BASE[:8]} is not in this checkout")
    assert (REPO / rel).read_bytes() == pinned, f"N1: {rel} differs from {BASE[:8]}"


def test_eval_imports_no_harness() -> None:
    """The eval is the oracle (green by design): it imports nothing under ``scripts/`` and, from ``local_astra``,
    only the handoff judge and the tool-set loader; it names no harness file outside its docstring."""
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = {a.name for a in node.names}
            if mod == "local_astra.pages" and names <= {"handoff_tasks"}:
                continue
            if mod == "local_astra.toolcall" and names <= {"tools"}:
                continue
            if mod.startswith(("local_astra", "scripts", "tests")):
                bad.append((mod, sorted(names)))
        elif isinstance(node, ast.Import):
            bad += [
                (a.name, [])
                for a in node.names
                if a.name.startswith(("local_astra", "scripts", "tests"))
            ]
    assert bad == [], bad
    body = EVAL.read_text(encoding="utf-8").split('"""', 2)[2]
    assert "t0220_handoff_nosearch.py" not in body and "scripts/gpu" not in body


# =====================================================================================================
# A2: the no-search kit
# =====================================================================================================


@pytest.mark.parametrize("ov,wait", COMBOS, ids=COMBO_IDS)
def test_nosearch_kit_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ov: bool, wait: bool
) -> None:
    """A2, each ``page_overview`` x ``wait_tool``: with cli's live options moved to the combination, the turn's kit
    (cli's on-set) becomes the no-search kit: its names are the on-set file's minus ``google_search`` in order, each
    declaration unchanged; ``tools`` is built from its ``tool_set``; ``tools_version`` is the sha256 of
    ``cli._NOSEARCH_PATHS[(ov, wait)]``'s bytes and differs from ``D.cli_kit_sha256``; no search."""
    h, A = harness(), _A()
    _patch_detok(monkeypatch, tmp_path)
    _set_combo(monkeypatch, ov, wait)
    _reset_reference(h)
    assert dict(A.kit_options(_lean())) == {"page_overview": ov, "wait_tool": wait}, (
        "premise: the options moved"
    )
    turn = _turn_kit(ov, wait)
    on = _on_set(ov, wait)
    assert turn.tools_version == on.version == A.D.cli_kit_sha256(ov, wait), (
        "premise: the turn kit is the on-set"
    )
    got = h.nosearch_kit(turn)
    want = [d for d in on.declarations if d["name"] != "google_search"]
    names = [t["function"]["name"] for t in got.tools]
    assert names == [d["name"] for d in want], f"A2: names {names}"
    assert got.tool_set is not None and list(got.tool_set.names) == names, (
        "A2: the judge's set is the declared one"
    )
    assert [json.dumps(d) for d in got.tool_set.declarations] == [json.dumps(d) for d in want], (
        "A2: a remaining declaration is not the on-set's"
    )
    assert list(got.tools) == _tools_field(got.tool_set), (
        "A2: tools are not built from the kit's set"
    )
    assert got.tools_version == _sha(_ns_path(ov, wait)), f"A2: tools_version {got.tools_version}"
    assert got.tools_version != A.D.cli_kit_sha256(ov, wait)
    assert got.search is None and got.web_search is None
    assert turn.tools_version == on.version, "A2: the turn's own kit was edited"
    _reset_reference(h)


def test_nosearch_kit_keeps_executor_and_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2: every field other than ``tools``/``tools_version``/``tool_set`` is the turn's own by identity (the
    executor, the live browser, the payloads and calibration of an on repair, the mode); ``search`` and
    ``web_search`` stay ``None``; a second call gives an equal kit."""
    h = harness()
    _patch_detok(monkeypatch, tmp_path)
    _reset_reference(h)
    turn = _turn_kit(*_head_combo())
    assert turn.repair_off is None and turn.payloads is not None, "premise: repair on"
    got = h.nosearch_kit(turn)
    assert got is not turn
    for f in dataclasses.fields(turn):
        if f.name in ("tools", "tools_version", "tool_set"):
            continue
        if f.name in ("search", "web_search"):
            assert getattr(got, f.name) is None, f.name
            continue
        assert getattr(got, f.name) is getattr(turn, f.name), f"A2: {f.name} is not the turn's own"
    assert got.executor is _EXEC and got.live_browser is _EXEC and got.mode == "live"
    again = h.nosearch_kit(turn)
    assert again.tools == got.tools and again.tools_version == got.tools_version


NOSEARCH_MUTANTS = ("still-declares-google-search", "version-is-the-on-sha", "another-nosearch-set")


def _bad_load(case: str) -> Callable[..., Any]:
    """``cli.load_tool_kit`` whose ``search=False`` answer is wrong in one way; every other call is the real one."""
    from local_astra import cli

    real = cli.load_tool_kit

    def load(profile: Any, *, search: bool = True, **k: Any) -> Any:
        if search:
            return real(profile, **k)
        if case == "still-declares-google-search":
            return real(profile, **k)
        if case == "version-is-the-on-sha":
            kit, det = real(profile, search=False, **k)
            on = real(profile, **k)[0]
            return dataclasses.replace(kit, tools_version=on.tools_version), det
        if case == "another-nosearch-set":
            return real(profile, search=False)
        raise KeyError(case)

    return load


@pytest.mark.parametrize("case", NOSEARCH_MUTANTS)
def test_nosearch_kit_refuses_a_wrong_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """A2 (the absent direction; the too-wide one is ``test_nosearch_kit_fields``): a reference whose ``search=False``
    set still declares ``google_search``, carries the on-set's sha, or is another no-search set (no ``wait`` with
    the wait tool on) makes ``nosearch_kit`` raise."""
    from local_astra import cli

    h = harness()
    _patch_detok(monkeypatch, tmp_path)
    turn = _turn_kit(*_head_combo())
    assert _head_combo() == (False, True), "premise: HEAD's live kit is the wait set"
    monkeypatch.setattr(cli, "load_tool_kit", _bad_load(case))
    _reset_reference(h)
    with pytest.raises(Exception):  # noqa: B017 - any refusal; the message is the harness's
        h.nosearch_kit(turn)
    _reset_reference(h)


def test_pa_guidance() -> None:
    """A3's premise at HEAD (the too-wide side of the P-A refusals below): T-0205's session keywords carry
    ``cli._tool_guidance("v3", True, wait, search=False)``, which names no ``google_search``; the guidance resolves to
    v3; and HEAD's main passes P-A (``test_rig_at_head_goes_to_the_engine``)."""
    from local_astra import cli

    A = _A()
    lean = _lean()
    ns = A.D._live_args()
    _ov, wait = _head_combo()
    text = cli._tool_guidance("v3", True, wait, search=False)
    assert dict(A.D.session_keywords(lean))["tool_guidance"] == text
    assert isinstance(text, str) and "google_search" not in text
    assert cli.live_guidance_of(lean, ns, "live").value == "v3"
    assert callable(api("main"))


# =====================================================================================================
# A3/A4: refusals before any socket, through main
# =====================================================================================================

#: the read-only object queries a git stub passes to the real git (A1 pins may read blobs this way)
_PASS_THROUGH = ("hash-object", "ls-tree", "ls-files", "show", "cat-file")


class Rig:
    """Every real socket refused and recorded; a clean tree and the merges through every ``_git`` the harness
    holds (object queries go to the real git, altered for ``altered`` paths); HEAD's real tool kit through
    ``cli.load_tool_kit`` over the subset detokeniser; Chromium present; an idle box where a seam exists; ``_measure``
    a named failure (nothing may get past the engine check on CPU)."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        from local_astra import cli

        self.h = h = harness()
        self.cli = cli
        self.mp = mp
        self.sockets: list[str] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (
            *OTHERS,
            MERGE_T0208_B,
            MERGE_T0211_A,
            MERGE_T0211_B,
            MERGE_T0215,
        )
        self.out = tmp_path / "out"
        self.altered: set[str] = set()
        rig = self

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                rig.sockets.append(what)
                raise ConnectionRefusedError(f"T-0220 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def altered_rel(args: tuple[str, ...]) -> str | None:
            for rel in rig.altered:
                if any(a == rel or a.endswith(":" + rel) or a.endswith("/" + rel) for a in args):
                    return rel
            return None

        def git(*args: str) -> str:
            if args[:1] == ("status",):
                return rig.porcelain
            if "log" in args:
                if any("%H" in a for a in args):
                    return "\n".join(rig.merges) + "\n"
                return "\n".join(m.split(" ", 1)[1] for m in rig.merges) + "\n"
            if "merge-base" in args:
                return ""
            if (args[:1] and args[0] in _PASS_THROUGH) or (
                args[:1] == ("rev-parse",) and any(":" in a for a in args[1:])
            ):
                out = subprocess.run(
                    ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
                ).stdout
                rel = altered_rel(args)
                if rel is not None:
                    blob = RUNNER_BLOBS[rel]
                    out = (
                        out.replace(blob, "0" * 40)
                        if blob in out
                        else out + "# T-0220 rig: altered\n"
                    )
                return out
            return t31.GIT + "\n"

        _patch_name(mp, h, "_git", git)
        _patch_detok(mp, tmp_path)
        mp.setattr(cli, "live_prerequisites", lambda *a, **k: None, raising=False)
        for seam, value in (
            ("_nvidia_smi", lambda *a, **k: t201.REAL_ENGINE_ONLY),
            ("_loadavg", lambda *a, **k: (0.5, 0.5, 0.5)),
            ("_listening_ports", lambda *a, **k: {8001}),
            ("_chromium_path", lambda *a, **k: "/opt/chromium/chrome"),
        ):
            _patch_name(mp, h, seam, value)

        def measure(*a: Any, **k: Any) -> Any:
            raise AssertionError("T-0220 rig: main got past the engine check on CPU")

        _patch_name(mp, h, "_measure", measure)

        real_rb, real_rt = Path.read_bytes, Path.read_text

        def rel_of(p: Path) -> str | None:
            try:
                return Path(p).resolve().relative_to(REPO.resolve()).as_posix()
            except ValueError:
                return None

        def read_bytes(p: Path) -> bytes:
            data = real_rb(p)
            return data + b"\n# T-0220 rig: altered\n" if rel_of(p) in rig.altered else data

        def read_text(p: Path, *a: Any, **k: Any) -> str:
            data = real_rt(p, *a, **k)
            return data + "\n# T-0220 rig: altered\n" if rel_of(p) in rig.altered else data

        mp.setattr(Path, "read_bytes", read_bytes)
        mp.setattr(Path, "read_text", read_text)

    def argv(self) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out)]


#: case -> a pattern the stderr line must match
REFUSALS: dict[str, str] = {
    "url-not-loopback": r"loopback|10\.0\.0\.1",
    "dirty-tree": r"session\.py",
    "out-dir-exists": r"exist",
    "no-merge-t0208-b": r"T-0208-b",
    "no-merge-t0215": r"T-0215",
    "branch-merge-only": r"T-0215",
    "pin-t0208": r"t0208_handoff_run",
    "pin-t0211": r"t0211_guidance_ab",
    "pin-t0205": r"t0205_demo_run",
    "kit-search-false-declares-google-search": r"google_search|tool|kit",
    "kit-search-false-on-sha": r"sha|tool|kit|version",
    "pa-guidance-names-google-search": r"google_search|guidance|P-A",
    "pa-session-keywords-differ": r"guidance|keyword|P-A",
    "pa-guidance-v4": r"v3|guidance",
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals_before_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """A3/A4: each refusal exits 2 with a stderr line naming it, before any socket, leaving the out-dir as it was:
    A.C.main's own (loopback, dirty tree, existing out-dir, no ``Merge T-0208-b:``), no ``Merge T-0215:`` (a branch
    merge or a lookalike subject does not count), each of A1's three pins off, a ``cli.load_tool_kit`` whose
    ``search=False`` still declares ``google_search`` (or carries the on-set's sha), and P-A failing (the guidance
    names ``google_search``; T-0205's keywords carry another text; the guidance resolves to v4)."""
    rig = Rig(tmp_path, monkeypatch)
    h, cli, A = rig.h, rig.cli, _A()
    argv = rig.argv()
    if case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-t0208-b":
        rig.merges = tuple(m for m in rig.merges if m != MERGE_T0208_B)
    elif case == "no-merge-t0215":
        rig.merges = tuple(m for m in rig.merges if m != MERGE_T0215)
    elif case == "branch-merge-only":
        rig.merges = (*(m for m in rig.merges if m != MERGE_T0215), *LOOKALIKES_T0215)
    elif case.startswith("pin-"):
        rel = next(r for r in RUNNER_BLOBS if case.split("-", 1)[1] in r)
        rig.altered = {rel}
    elif case == "kit-search-false-declares-google-search":
        monkeypatch.setattr(cli, "load_tool_kit", _bad_load("still-declares-google-search"))
    elif case == "kit-search-false-on-sha":
        monkeypatch.setattr(cli, "load_tool_kit", _bad_load("version-is-the-on-sha"))
    elif case == "pa-guidance-names-google-search":
        real = cli._tool_guidance

        def guidance(version: Any, web: bool, wait_tool: bool, search: bool = True) -> Any:
            t = real(version, web, wait_tool, search=search)
            return None if t is None else t + " Use google_search for facts you do not know."

        monkeypatch.setattr(cli, "_tool_guidance", guidance)
    elif case == "pa-session-keywords-differ":
        real_kw = A.D.session_keywords

        def keywords(profile: Any) -> Any:
            kw = dict(real_kw(profile))
            kw["tool_guidance"] = (
                str(kw["tool_guidance"]) + " T-0220 rig: a sentence the cli does not give."
            )
            return kw

        assert _patch_object(monkeypatch, h, real_kw, keywords) >= 1
    elif case == "pa-guidance-v4":
        monkeypatch.setattr(
            cli, "live_guidance_of", lambda p, a, m: cli.LiveGuidance("v4", "default")
        )
    else:
        raise KeyError(case)
    _reset_reference(h)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = capsys.readouterr().err
    _reset_reference(h)
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert re.search(REFUSALS[case], err, re.IGNORECASE | re.DOTALL), (case, err[-2000:])


def test_rig_at_head_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A4, the too-wide direction: at HEAD (every merge, the pins intact, the real kit, P-A passing) ``main``
    refuses nothing; it asks for the engine and, finding none, exits 1."""
    rig = Rig(tmp_path, monkeypatch)
    _reset_reference(rig.h)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    _reset_reference(rig.h)
    assert rc == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


def test_no_prefix_cache_counters_is_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A4: an engine that serves the model but has no prefix-cache counters on ``/metrics`` is exit 1, before any
    turn (``_measure`` never reached)."""
    from local_astra.conductor.llm_client import VllmEndpoint

    rig = Rig(tmp_path, monkeypatch)
    A = _A()
    model = VllmEndpoint().model

    def get_text(url: str, *a: Any, **k: Any) -> str | None:
        if url.rstrip("/").endswith("/models"):
            return json.dumps({"data": [{"id": model}]})
        return None

    monkeypatch.setattr(A.G, "_get_text", get_text)
    monkeypatch.setattr(A.T, "_healthy", lambda *a, **k: True)
    _reset_reference(rig.h)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    _reset_reference(rig.h)
    assert rc == 1, f"exit {rc}\n{err[-2000:]}"
    assert re.search(r"metric|prefix", err, re.IGNORECASE), err[-2000:]


# =====================================================================================================
# A5: order, budget, errored turns
# =====================================================================================================


def test_order_and_constants() -> None:
    """A5: ``order(0)`` is ``A.C.order(0)`` and the hand oracle (18 turns, each once); ``HARD_LIMIT_S = 1800``,
    ``RUN_BUDGET_S = 1320``; T-0208's per-turn timeout is 300 s and its errored stop is ``G.ERRORED_STOP``."""
    h, A = harness(), _A()
    got = [tuple(x) for x in h.order(0)]
    assert got == [tuple(x) for x in A.C.order(0)] == _order0()
    assert len(set(got)) == 18
    assert (h.HARD_LIMIT_S, h.RUN_BUDGET_S) == (30 * 60, 22 * 60)
    assert A.C.TURN_TIMEOUT_S == 300 and A.G.ERRORED_STOP == 3


def _run(tmp_path: Path, mp: pytest.MonkeyPatch, start: float) -> tuple[Any, Any]:
    h, A = harness(), _A()
    _patch_detok(mp, tmp_path)
    _reset_reference(h)
    clock = t210._Clock(start)
    for mod in (h, A):
        mp.setattr(mod, "time", clock, raising=False)
        mp.setattr(mod, "PROCESS_START_S", start, raising=False)
    out = tmp_path / "out"
    (out / "sessions").mkdir(parents=True)
    (out / "turns.jsonl").touch()
    known = {
        "base_url": "http://127.0.0.1:8001/v1",
        "out": out,
        "profile": _lean(),
        "sentinel": b"T0220-BUDGET-SENTINEL",
        "manifest": {"stopped": None},
        "vram": t205._FakeVram(),
    }
    cls = getattr(h, "_Run", A.C._Run)
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in known.items() if k in names}), clock


def _stub_turns(
    mp: pytest.MonkeyPatch,
    out: Path,
    clock: Any,
    start: float,
    after: list[float],
    raise_on: Callable[[int, str], bool] = lambda k, arm: False,
) -> list[tuple[str, str, int, Any]]:
    """``A.C._one_turn`` stubbed: records ``(hid, arm, repeat, search_kit)``, sets the clock to ``start + after[k]``,
    writes a clean turn (session log and turn line) or raises when ``raise_on(k, arm)``."""
    A = _A()
    seen: list[tuple[str, str, int, Any]] = []

    async def one_turn(
        run: Any, hid: str, arm: str, repeat: int, *a: Any, search_kit: Any = None, **k: Any
    ) -> Any:
        await asyncio.sleep(0)
        seen.append((hid, arm, repeat, search_kit))
        n = len(seen)
        clock.now = start + after[min(n, len(after)) - 1]
        if raise_on(n, arm):
            raise RuntimeError(f"T-0220 rig: turn {n} raised")
        if search_kit is not None:
            # as A.C._one_turn does: the loaded kit through the caller's seam before the Session is built
            search_kit(_turn_kit(*_head_combo()))
        trial, steps = _turn_files(out, _spec(hid, arm, repeat))
        return trial, steps, None

    mp.setattr(A.C, "_one_turn", one_turn)
    return seen


def _call_run_all(h: ModuleType, run: Any, turns: list[tuple[str, str, int]]) -> None:
    fn = api("_run_all")
    params = list(inspect.signature(fn).parameters)
    if len(params) == 1:
        asyncio.run(fn(run))
    else:
        assert len(params) == 2, f"T-0220: _run_all(run) or _run_all(run, turns), not {params}"
        asyncio.run(fn(run, turns))


def _trials(out: Path) -> list[dict[str, Any]]:
    p = out / "trials.jsonl"
    return (
        [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if p.is_file()
        else []
    )


@pytest.mark.parametrize(("second", "want"), [(1260.0, 2), (1261.0, 1)])
def test_budget_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second: float, want: int
) -> None:
    """A5: turn 1 ends at elapsed ``second``; turn 2 starts iff ``second + 480 + 60 <= 1800`` (1260 starts, a fake
    clock at 1261 stops before the turn: a dropped check lets both start, a ``RUN_BUDGET_S`` check alone lets 1261
    start). A stop sets ``stopped`` and the report written after it is ``incomplete`` with its mark."""
    h = harness()
    start = 9000.0
    run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(monkeypatch, run.out, clock, start, [second, second + 1.0, second + 2.0])
    _call_run_all(h, run, [tuple(x) for x in h.order(0)][:3])
    assert len(seen) == want, (second, len(seen))
    if want == 1:
        assert isinstance(run.stopped, str) and run.stopped
        api("write_report")(run.out, t208._probes(), _manifest(run.stopped), run.stopped)
        report = json.loads((run.out / "report.json").read_text(encoding="utf-8"))
        assert report.get("label") == "incomplete", (report.get("label"), report.get("violations"))
        assert "provisional (run incomplete)" in _line(report)
        assert (run.out / "report.md").is_file()


def test_turns_go_through_the_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1/A5: every turn is ``A.C._one_turn(run, hid, arm, repeat, search_kit=nosearch_kit)`` in ``order(0)``, one at
    a time; each trial line carries its handoff and arm."""
    h = harness()
    start = 100.0
    run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(monkeypatch, run.out, clock, start, [10.0 * k for k in range(1, 19)])
    turns = [tuple(x) for x in h.order(0)][:4]
    _call_run_all(h, run, turns)
    got = [(a, b, c) for a, b, c, _ in seen][: len(turns)]
    assert got == turns, got
    ov, wait = _head_combo()
    for *_x, sk in seen:
        # the seam is nosearch_kit (or a recording wrapper of it): judged by what it hands out
        assert callable(sk), f"A1: no search_kit seam passed ({sk!r})"
        out_kit = sk(_turn_kit(ov, wait))
        assert out_kit.tools_version == _ns_set(ov, wait).version, out_kit.tools_version
        assert "google_search" not in [t["function"]["name"] for t in out_kit.tools]
    assert [(t["task"], t["arm"]) for t in _trials(run.out)][: len(turns)] == [
        (a, b) for a, b, _ in turns
    ]


def test_errored_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A5: after ``G.ERRORED_STOP`` consecutive errored turns the run stops (nothing more is started)."""
    h = harness()
    start = 100.0
    run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(
        monkeypatch,
        run.out,
        clock,
        start,
        [5.0 * k for k in range(1, 19)],
        raise_on=lambda k, arm: True,
    )
    _call_run_all(h, run, [tuple(x) for x in h.order(0)])
    assert len(seen) == _A().G.ERRORED_STOP
    assert isinstance(run.stopped, str) and run.stopped


def test_not_run_is_errored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A5 (adversarial): the first shared and the first typed turn raise: both are not-run lines (never dropped,
    success False, ``engine_up_after`` None, the exception named; the shared one ``ph3: None`` by R-3, so the run is
    not void), the run goes on through all 18 turns, and the reading carries "provisional (errored 2)"."""
    h = harness()
    start = 100.0
    run, clock = _run(tmp_path, monkeypatch, start)
    first: dict[str, int] = {}

    def raise_on(k: int, arm: str) -> bool:
        first.setdefault(arm, k)
        return first[arm] == k

    seen = _stub_turns(
        monkeypatch, run.out, clock, start, [5.0 * k for k in range(1, 19)], raise_on=raise_on
    )
    turns = [tuple(x) for x in h.order(0)]
    _call_run_all(h, run, turns)
    assert len(seen) == 18, len(seen)
    tr = _trials(run.out)
    assert [(t["task"], t["arm"], t["repeat"]) for t in tr] == turns
    raised = [t for t in tr if "T-0220 rig" in str(t.get("exception"))]
    assert sorted(t["arm"] for t in raised) == ["shared", "typed"], raised
    for t in raised:
        assert (
            t.get("success") is False and "engine_up_after" in t and t["engine_up_after"] is None
        ), t
    assert [t.get("ph3") for t in raised if t["arm"] == "shared"] == [None], (
        "R-3: a raised shared turn is ph3 None"
    )
    api("write_report")(run.out, t208._probes(), _manifest(None), None)
    report = json.loads((run.out / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "measured", (report.get("label"), report.get("violations"))
    assert "provisional (errored 2)" in _line(report), _line(report)


# =====================================================================================================
# A6: withdrawal evidence per turn, over hand-built session logs
# =====================================================================================================

EVIDENCE_VOID = (
    "declares-google-search",
    "system-names-google-search",
    "search-not-connected",
    "wrong-tools-version",
    "regime-tools-on-sha",
    "executed-google-search",
)


def _one(tmp_path: Path, **kw: Any) -> dict[str, Any]:
    _out, report = _write_run(tmp_path / "run", [_spec("H1", "shared", 0, **kw)])
    return report


def test_evidence_clean_turn_with_a_refused_search_attempt(tmp_path: Path) -> None:
    """A6 (adversarial, the too-wide direction): a clean shared turn whose model tried ``call:google_search`` first
    (ended ``refused``, not declared) and then clicked: not void, ``undeclared_search_calls == 1``; its thought text
    and its continuation naming ``google_search`` (never a system message) do not trip row (b); the label is
    ``measured``."""
    report = _one(tmp_path, search="refused")
    log = (tmp_path / "run" / "sessions" / "turn-H1-shared-0.jsonl").read_text(encoding="utf-8")
    assert log.count("google_search") >= 3, (
        "premise: the thought, the call and the continuation name it"
    )
    assert report.get("violations") == [], report.get("violations")
    assert report.get("label") == "measured", report.get("label")
    assert _undeclared(report) == 1


def test_evidence_clean_turn_premise(tmp_path: Path) -> None:
    """A6 (too-wide): a clean turn with no search attempt is not void and counts zero undeclared calls."""
    report = _one(tmp_path)
    assert report.get("violations") == [] and report.get("label") == "measured", report.get(
        "violations"
    )
    assert _undeclared(report) == 0


@pytest.mark.parametrize("case", EVIDENCE_VOID)
def test_evidence_rows(tmp_path: Path, case: str) -> None:
    """A6, each ``void``: a request whose ``prompt.tools`` declares ``google_search`` (a); a system message naming it
    (b); ``SEARCH_NOT_CONNECTED`` in a tool response (c); the trial's ``tools_version`` the on-set's (d); the turn
    line's ``Regime.tools`` the on-set's sha12 (e); an executed ``google_search`` (counted, and void)."""
    if case == "executed-google-search":
        report = _one(tmp_path, search="executed")
        assert _undeclared(report) == 1
    else:
        report = _one(tmp_path, fault=case)
    assert report.get("violations"), case
    assert all("H1" in str(v) for v in report["violations"]), report["violations"]
    assert report.get("label") == "void", (case, report.get("label"))
    assert "provisional (run void)" in _line(report)


# =====================================================================================================
# A8: the reading and its marks, exact Fractions
# =====================================================================================================

READING_CASES: dict[str, tuple[dict[str, Any], str]] = {
    "shared-9-of-9": (dict(k=9), UNBLOCKS),
    "shared-5-of-9": (dict(k=5), UNBLOCKS),
    "shared-5-of-9-with-a-refused-search": (dict(k=5, search_refused=1), UNBLOCKS),
    "shared-4-of-9": (dict(k=4), MIXED),
    "shared-2-of-9": (dict(k=2), MIXED),
    "shared-1-of-9": (dict(k=1), NO_CHANGE),
    "shared-0-of-9": (dict(k=0), NO_CHANGE),
    "shared-5-of-9-safety-event": (dict(k=5, safety=1), MIXED),
    "shared-1-of-9-safety-event": (dict(k=1, safety=1), NO_CHANGE),
}


@pytest.mark.parametrize("case", list(READING_CASES))
def test_reading_lines(tmp_path: Path, case: str) -> None:
    """A8: exactly one pre-registered phrase: shared used_browser >= 5/9 with A6 clean and zero safety events ->
    "withdrawal unblocks the handoff"; <= 1/9 -> "no change"; else "mixed" (exact ``Fraction`` edges at 5/9 and
    1/9; a safety event at 5/9 is mixed)."""
    kw, want = READING_CASES[case]
    _out, report = _write_run(tmp_path / "run", _full(**kw))
    ub = report.get("used_browser") or {}
    shared = ub.get("shared") or {}
    assert Fraction(int(shared.get("k")), int(shared.get("n"))) == Fraction(kw["k"], 9), shared
    line = _line(report)
    assert want in line, (case, line)
    for other in READINGS:
        if other != want:
            assert other not in line, (case, other, line)
    assert "provisional" not in line, (case, line)


MARKS: dict[str, tuple[dict[str, Any], str | None, list[str], list[str]]] = {
    "measured": (dict(k=5), None, [], ["provisional"]),
    "incomplete": (
        dict(k=5),
        "T-0220 rig: budget",
        ["provisional (run incomplete)"],
        ["provisional (errored"],
    ),
    "errored-1": (dict(k=5, not_run_typed=1), None, [], ["provisional"]),
    "errored-2": (
        dict(k=5, not_run_typed=2),
        None,
        ["provisional (errored 2)"],
        ["provisional (run"],
    ),
}


@pytest.mark.parametrize("case", list(MARKS))
def test_reading_marks(tmp_path: Path, case: str) -> None:
    """A8: "provisional (run <label>)" when not ``measured``; "provisional (errored N)" when N >= 2 (one errored
    turn marks nothing)."""
    kw, stopped, want, absent = MARKS[case]
    _out, report = _write_run(tmp_path / "run", _full(**kw), stopped=stopped)
    line = _line(report)
    for w in want:
        assert w in line, (case, w, line)
    for a in absent:
        assert a not in line, (case, a, line)


# =====================================================================================================
# A8: the anchors from the local logs
# =====================================================================================================

ANCHOR_DIRS = {
    "t0208": Path("logs") / "t0208-handoff-9a079c3",
    "t0211": Path("logs") / "t0211-guidance-ab-e743c56",
}


def _main_checkout() -> Path | None:
    out = _git_out("rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(out.strip()).parent if out and out.strip() else None


def _anchor_dir(rel: Path) -> Path | None:
    roots = [REPO]
    main = _main_checkout()
    if main is not None:
        roots.append(main)
    for root in roots:
        d = root / rel
        if (d / "trials.jsonl").is_file() and (d / "steps.jsonl").is_file():
            return d
    return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_anchor_constants_from_local_logs(tmp_path: Path) -> None:
    """A8 (skipped when a local log is absent): ``A.used_browser`` recomputes T-0208 C's shared **1/9** (ruling R-1;
    the task's 0/9 was its shared success) from ``logs/t0208-handoff-9a079c3/`` and T-0211 C v3 part H's shared 6/9
    from ``logs/t0211-guidance-ab-e743c56/``; the success anchors (0/9, 7/9; 3/9, 6/9) with the runners' own success
    rules; the harness's ``ANCHOR`` (keys ``t0208_c``, ``t0211_c_v3_h``, values ``(k, n)``) holds the recomputed
    used_browser pairs, and a harness-built report.md quotes each anchor on one line with its commit."""
    d08, d11 = _anchor_dir(ANCHOR_DIRS["t0208"]), _anchor_dir(ANCHOR_DIRS["t0211"])
    if d08 is None or d11 is None:
        pytest.skip(f"{ANCHOR_DIRS} absent (local logs, not in the repo)")
    A = _A()

    def by_turn(
        steps: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]
    ) -> dict[Any, list[Any]]:
        out: dict[Any, list[Any]] = {}
        for s in steps:
            out.setdefault(key(s), []).append(s)
        return out

    k08 = lambda x: (x["task"], x["arm"], int(x["repeat"]))  # noqa: E731
    tr08, st08 = _jsonl(d08 / "trials.jsonl"), by_turn(_jsonl(d08 / "steps.jsonl"), k08)
    sh08 = [t for t in tr08 if t["arm"] == "shared"]
    ty08 = [t for t in tr08 if t["arm"] == "typed"]
    assert (sum(A.used_browser(st08.get(k08(t), [])) for t in sh08), len(sh08)) == (1, 9)
    assert (sum(A.C.succeeded(t, st08.get(k08(t), [])) for t in sh08), len(sh08)) == (0, 9)
    assert (sum(A.C.succeeded(t, st08.get(k08(t), [])) for t in ty08), len(ty08)) == (7, 9)

    k11 = lambda x: (x["task"], x["arm"], x["mode"], int(x["repeat"]))  # noqa: E731
    tr11 = [
        t for t in _jsonl(d11 / "trials.jsonl") if t.get("part") == "H" and t.get("arm") == "v3"
    ]
    st11 = by_turn([s for s in _jsonl(d11 / "steps.jsonl") if s.get("part") == "H"], k11)
    sh11 = [t for t in tr11 if t["mode"] == "shared"]
    ty11 = [t for t in tr11 if t["mode"] == "typed"]
    assert (sum(A.used_browser(st11.get(k11(t), [])) for t in sh11), len(sh11)) == (6, 9)
    assert (sum(A.success(t) for t in sh11), len(sh11)) == (3, 9)
    assert (sum(A.success(t) for t in ty11), len(ty11)) == (6, 9)

    anchor = json.loads(json.dumps(api("ANCHOR"), default=str))
    for key, pair in (("t0208_c", [1, 9]), ("t0211_c_v3_h", [6, 9])):
        assert key in anchor, (key, list(anchor))
        assert any(v == pair for _p, v in _walk(anchor[key])), (key, pair, anchor[key])
    run = _eval().load_run(_synthetic(tmp_path, "clean"))
    _eval().test_anchor_quoted_unpaired(run)


# =====================================================================================================
# the synthetic 18-turn run dir and its seeded faults (binding lesson, T-0214 C2)
# =====================================================================================================


def _synthetic_specs() -> list[dict[str, Any]]:
    """18 turns by hand: 5 of 9 shared turns act (the first of them tries ``google_search`` first, refused), one
    typed turn not run, no safety event; the reading by hand: "withdrawal unblocks the handoff", label measured."""
    return _full(5, search_refused=1, not_run_typed=1)


def _synthetic(root: Path, kind: str) -> Path:
    stopped = "T-0220 rig: the hard limit is near" if kind == "incomplete" else None
    out, _ = _write_run(
        root / f"t0220-handoff-nosearch-{kind}", _synthetic_specs(), stopped=stopped
    )
    return out


def _run_rows(ev: ModuleType) -> dict[str, Callable[[Any], None]]:
    marks = {mk.name for mk in getattr(ev, "pytestmark", []) if hasattr(mk, "name")}
    assert "gpu" in marks, marks
    return {
        name: fn
        for name, fn in vars(ev).items()
        if name.startswith("test_")
        and callable(fn)
        and list(inspect.signature(fn).parameters) == ["run"]
    }


def _failing(ev: ModuleType, path: Path) -> dict[str, str]:
    run = ev.load_run(path)
    failed = {}
    for name, fn in _run_rows(ev).items():
        try:
            fn(copy.deepcopy(run))
        except AssertionError as exc:
            failed[name] = repr(exc)[:300]
    return failed


def _rw_jsonl(path: Path, edit: Callable[[list[dict[str, Any]]], None]) -> None:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    edit(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _f_declared(out: Path) -> None:
    ov, wait = _head_combo()
    on_tools = _tools_field(_on_set(ov, wait))

    def edit(rows: list[dict[str, Any]]) -> None:
        req = next(r for r in rows if r.get("kind") == "llm.request")
        req["data"]["prompt"]["tools"] = on_tools

    _rw_jsonl(out / "sessions" / "turn-H2-typed-0.jsonl", edit)


def _f_not_connected(out: Path) -> None:
    from local_astra.toolcall import loop

    def edit(rows: list[dict[str, Any]]) -> None:
        res = next(r for r in rows if r.get("kind") == "tool.result")
        res["data"]["response"] = (
            f"<|tool_response>response:google_search{{note:{Q}{loop.SEARCH_NOT_CONNECTED}{Q}}}<tool_response|>"
        )

    _rw_jsonl(out / "sessions" / "turn-H1-typed-2.jsonl", edit)


def _f_used_browser(out: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> None:
        hit = [
            s
            for s in rows
            if (s["task"], s["arm"], s["repeat"]) == ("H1", "shared", 1)
            and s.get("status") == "executed"
            and s.get("call_name") == "click_at"
        ]
        assert hit, "T-0220 rig: H1/shared/r1 has no executed click"
        for s in hit:
            s["status"] = "refused"

    _rw_jsonl(out / "steps.jsonl", edit)


def _replace(path: Path, old: str, new: str, *, on_line_with: str | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    if on_line_with is None:
        assert old in text, f"T-0220 rig: {path.name} lacks {old!r}"
        path.write_text(text.replace(old, new), encoding="utf-8")
        return
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if on_line_with in ln and old in ln]
    assert idx, f"T-0220 rig: no {path.name} line with {on_line_with!r} and {old!r}"
    lines[idx[0]] = lines[idx[0]].replace(old, new, 1)
    path.write_text("\n".join(lines), encoding="utf-8")


def _f_anchor(out: Path) -> None:
    _replace(out / "report.md", "6/9", "5/9", on_line_with="e743c565")


def _f_undeclared(out: Path) -> None:
    p = out / "report.json"
    rep = json.loads(p.read_text(encoding="utf-8"))
    rep["undeclared_search_calls"] = int(rep.get("undeclared_search_calls") or 0) + 1
    p.write_text(json.dumps(rep), encoding="utf-8")


def _f_unpaired(out: Path) -> None:
    _replace(out / "report.md", "engine unseeded", "engine seeded")


FAULTS: dict[str, tuple[Callable[[Path], None], str]] = {
    "a-google-search-declaration-injected": (_f_declared, "test_withdrawal_evidence_re_derived"),
    "b-search-not-connected-injected": (_f_not_connected, "test_withdrawal_evidence_re_derived"),
    "c-used-browser-flipped": (_f_used_browser, "test_used_browser_re_derived"),
    "d-anchor-number-altered": (_f_anchor, "test_anchor_quoted_unpaired"),
    "e-undeclared-count-off-by-one": (_f_undeclared, "test_undeclared_search_calls_re_derived"),
    "f-unpaired-note-altered": (_f_unpaired, "test_anchor_quoted_unpaired"),
}


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("t0220-synth")


@functools.cache
def _built(root: str, kind: str) -> Path:
    return _synthetic(Path(root), kind)


def test_synthetic_run_dir_premise(synth_root: Path) -> None:
    """Premise: 18 trials (one not run, typed), 17 session logs, label ``measured``, shared used_browser 5/9, one
    undeclared search call, the reading by hand ("withdrawal unblocks the handoff"), the voiced texts in report.md."""
    out = _built(str(synth_root), "clean")
    tr = _trials(out)
    assert len(tr) == 18 and {t["arm"] for t in tr} == set(ARMS)
    assert sum(1 for t in tr if "engine_up_after" in t and t["engine_up_after"] is None) == 1
    assert len(list((out / "sessions").glob("*.jsonl"))) == 17
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "measured", (report.get("label"), report.get("violations"))
    assert _undeclared(report) == 1
    assert UNBLOCKS in _line(report)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert J.H1_SAID in md and "I cannot act on your screen; I could not finish that." in md


def test_every_eval_run_row_passes_on_a_harness_built_run_dir(synth_root: Path) -> None:
    """The too-wide direction: every eval run row passes on the clean dir and on the clean incomplete dir; every
    seeded fault targets a real row."""
    ev = _eval()
    rows = _run_rows(ev)
    assert len(rows) >= 10, sorted(rows)
    targets = {r for _, r in FAULTS.values()} | {"test_label_re_derived", "test_reading_re_derived"}
    assert targets <= set(rows), sorted(targets - set(rows))
    assert _failing(ev, _built(str(synth_root), "clean")) == {}
    assert _failing(ev, _built(str(synth_root), "incomplete")) == {}
    rep = json.loads(
        (_built(str(synth_root), "incomplete") / "report.json").read_text(encoding="utf-8")
    )
    assert rep["label"] == "incomplete" and "provisional (run incomplete)" in _line(rep)


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_seeded_fault_fails_its_row(synth_root: Path, tmp_path: Path, fault: str) -> None:
    """The absent direction: each seeded fault fails the eval row named for it (a ``google_search`` declaration
    injected; one ``SEARCH_NOT_CONNECTED`` injected; one ``used_browser`` flipped; one anchor number altered; the
    undeclared count; the unpaired note)."""
    ev = _eval()
    how, row = FAULTS[fault]
    clean = _built(str(synth_root), "clean")
    assert row not in _failing(ev, clean), (fault, row)
    out = tmp_path / clean.name
    shutil.copytree(clean, out)
    how(out)
    failed = _failing(ev, out)
    print(f"T-0220 {fault}: failing rows {sorted(failed)}")
    assert row in failed, (fault, row, sorted(failed))


def test_incomplete_mark_dropped_fails_its_row(synth_root: Path, tmp_path: Path) -> None:
    """The provisional mark dropped from report.json and report.md of an ``incomplete`` dir fails the reading row."""
    ev = _eval()
    src = _built(str(synth_root), "incomplete")
    out = tmp_path / src.name
    shutil.copytree(src, out)
    _replace(out / "report.json", "provisional (run incomplete)", "")
    _replace(out / "report.md", "provisional (run incomplete)", "")
    assert "test_reading_re_derived" in _failing(ev, out)
