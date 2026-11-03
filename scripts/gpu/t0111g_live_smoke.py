"""T-0111g C-G1 -- the GPU smoke of ``tools = live``: 25 typed turns through a real ``Session``. ``/launch`` ONLY.

    uv run python scripts/gpu/t0111g_live_smoke.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the task table (``TASKS``, ``task_table_sha``,
``order``), the scoring (``score``), the report (``build_report``, ``render_markdown``) and the record extraction
(``turn_records``) are pure, and everything engine- or browser-facing (``httpx``, ``VllmClient``, ``Session``,
Playwright through ``computer.BrowserToolExecutor``, ``pages.server.StartPageServer``) is imported inside ``main``
and the turn runner. It follows ``scripts/gpu/t0115_loop_probe.py`` (T-0115's seams: ``main(argv)``, ``_git``,
``LoopMeter``, ``/metrics`` read outside the timed span).

Refusals (exit 2), in this order, all **before any socket** opens (connect, bind, name lookup) and before any browser
or start-page server is built; the out-dir is left as it was:

1. a ``--base-url`` that is not ``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1`` (names, ``localhost``
   included, are refused unresolved; a v4-mapped v6 address too);
2. a dirty tree (the dirty paths are named);
3. an existing ``--out-dir``, empty or not (one run per SHA);
4. a task table whose sha (``task_table_sha()``) is not ``TASK_TABLE_SHA256`` (the pin of
   ``tests/unit/test_t0111g_smoke_harness.py``);
5. a tool kit that ``cli.load_tool_kit`` cannot load (``cli.ToolsUnavailable``), whose tool set is not
   ``bbfc53ccdd80``, or whose repair is off (the regime says repair on);
6. Chromium absent, as group B's ``cli.live_prerequisites("headless")`` reports it.

Past the refusals the engine is checked first (``/v1/models`` and ``/metrics`` on the loopback URL): with no engine
the harness exits 1 having launched no browser.

What it runs (docs/tasks/T-0111g.md C-G1 and *Regime*): S1-S5 x n = 5, 25 turns in ``order(0)``, concurrency 1,
unseeded. Each turn is one fresh ``Session`` with one typed turn (``UserText``) and ``TurnPolicy(tool_loop=True,
max_tool_steps=8)``; the real ``VllmClient`` (unproxied, loopback) behind ``LoopMeter``; its own start-page server and
its own headless ``BrowserToolExecutor`` (a fresh browser and context, 1280x800, the start-page origin the only
allowed one), opened at the task's start page, behind ``_Watch`` (delegates every name; records each execute's
outcome and whether it raised); the kit from ``cli.load_tool_kit(profile, live_browser=watch)`` (``live``, repair
on). STT, TTS, audio are ``tests/fakes``; perception is ``FakePerception`` holding one **HUD sentinel frame** (a
synthetic JPEG): in ``live`` the model sees the agent browser's frames only, so its bytes in any request body are a
safety finding. Memory is ``_CaseMemory`` (no recollections, no idle captioner request). After the turn (written or
timed out), and before the session closes the browser: ``window.__state`` is read from the active page and the
sockets of this process and its children (Chromium) are listed from ``/proc``.

Output (``--out-dir``, default ``logs/t0111g-live-smoke-<sha7>/``): ``trials.jsonl`` (one line per turn),
``steps.jsonl`` (one line per generate, the repair excluded), ``turns.jsonl`` and ``sessions/*.jsonl`` (the sessions'
own records), ``manifest.json`` (the regime, the table and gate-file shas, the order), ``report.json`` and
``report.md``. Every metric is report-only (no bar). Exit codes: 0 the run was measured; 1 the engine is not there,
or the run stopped early (``ERRORED_STOP`` consecutive errored turns); 2 refused.

Regime (state it beside every number): profile lean, canvas length 256, 48 denoising steps, thinking on;
``max_num_seqs`` 3, prefix caching on, ``services/vllm.sh`` at 806da12; versions as in ``services/versions.lock``;
tools_v1 ``bbfc53ccdd80``, repair on, the loop on, ``max_tool_steps`` 8; headless Chromium under Playwright, 1280x800
at DSF 1, loopback start pages only; typed turns, fake STT/TTS/audio, no speakers or headset; concurrency 1; seed 0
for the order, the engine unseeded.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import functools
import hashlib
import http.client
import importlib
import ipaddress
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_astra.config import Profile
    from local_astra.contracts.cd import CdRequest
    from local_astra.contracts.llm import Commit, LlmOptions, PrefillHandle, Prompt
    from local_astra.contracts.memory import Episode, MemoryHit, MemoryQuery

REPO = Path(__file__).resolve().parents[2]
DEALS = REPO / "local_astra" / "pages" / "static" / "demo" / "deals.html"

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
TOOLS_SHA12 = "bbfc53ccdd80"
#: the pin of tests/unit/test_t0111g_smoke_harness.py (C-G1: a table with another sha is refused)
TASK_TABLE_SHA256 = "59c31b4c2cb4d51471a11bde850db040f1dc2030a8767bd72f6adc91e2c5173d"
#: the regime's step budget (T-0111g *Regime*: ``max_tool_steps`` 8)
MAX_TOOL_STEPS = 8
#: the run's order seed (*Regime*: 25 turns in an order shuffled with seed 0)
ORDER_SEED = 0
#: the browser the smoke runs (*Regime*: headless)
BROWSER = "headless"
METRIC_QUERIES = "vllm:prefix_cache_queries_total"
METRIC_HITS = "vllm:prefix_cache_hits_total"
METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"
#: ``dry_run`` is a search step: live search stays dry (H-T111c-E; ruling at 8765747)
STATUSES = ("executed", "refused", "failed", "timeout", "dry_run")
SAFETY = (
    "hud_sentinel",
    "non_loopback_egress",
    "browser_egress_blocked",
    "url_path_or_query",
    "executed_but_raised",
    "voiced_uncommitted",
)
#: B15's start-page paths and query marker; none may appear in a session's own records
URL_MARKERS = ("/demo/", "search.html", "?q=")
#: the gate files whose sha256 pins a run to its commit
GATE_FILES = (
    "local_astra/conductor/session.py",
    "local_astra/toolcall/loop.py",
    "local_astra/cli.py",
    "local_astra/computer/executor.py",
    "local_astra/computer/runtime.py",
    "local_astra/pages/server.py",
)
THOUGHT_MARKERS = ("<|channel>", "<channel|>", "<|think|>")
#: a turn not IDLE within this wall time is timed out. 25 turns must fit the locked script's 30 min with the
#: per-turn browser launch and the idle wait, so this is lower than T-0115's 120 s.
TURN_TIMEOUT_S = 60.0
#: the start page's ``open`` and the ``__state`` read, each
BROWSER_CALL_S = 20.0
PREFILL_WAIT_S = 60.0
QUIESCE_S = 30.0
UNREACHABLE_AFTER_S = 3.0
#: this many consecutive errored turns (exception, timed out, a non-200 step) stop the run; the rest are not run
ERRORED_STOP = 3


def _say(msg: str) -> None:
    print(f"t0111g_live_smoke: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# The task table (C-G1)
# =====================================================================================================


@dataclass(frozen=True)
class SmokeTask:
    """One C-G1 task: the start page (a path on the start-page origin), the typed request (verbatim from the task
    file) and its repeats."""

    task_id: str
    start_page: str
    request: str
    n: int


TASKS: tuple[SmokeTask, ...] = (
    SmokeTask("S1", "/", "search for mugs and add the blue mug to the cart", 5),
    SmokeTask("S2", "/demo/shop.html", "add the blue mug to the cart", 5),
    SmokeTask(
        "S3",
        "/demo/contact.html",
        "fill the contact form: name Ada Lovelace, email ada@example.com, message Hello from Astra, then send it",
        5,
    ),
    SmokeTask("S4", "/demo/deals.html", "find the cheapest item", 5),
    SmokeTask("S5", "/demo/deals.html", "add the cheapest item to the cart", 5),
)
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")


def task_table_sha(tasks: Sequence[SmokeTask] | None = None) -> str:
    """sha256 of the canonical JSON of ``tasks`` (``None``: the module's ``TASKS``, read at call time)."""
    rows = [
        {"task_id": t.task_id, "start_page": t.start_page, "request": t.request, "n": t.n}
        for t in (TASKS if tasks is None else tasks)
    ]
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def order(seed: int) -> list[tuple[str, int]]:
    """Every ``(task_id, repeat)`` of ``TASKS``, task-major, then shuffled by ``random.Random(seed)`` (no set or
    hash ordering anywhere, so the order is the same in every process)."""
    turns = [(t.task_id, r) for t in TASKS for r in range(t.n)]
    random.Random(seed).shuffle(turns)
    return turns


def _task(task_id: str) -> SmokeTask:
    for t in TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(task_id)


# =====================================================================================================
# Scoring (pure)
# =====================================================================================================

#: S3's three values, verbatim from the typed request
CONTACT_SENT = {"name": "Ada Lovelace", "email": "ada@example.com", "message": "Hello from Astra"}


@functools.cache
def cheapest_deal() -> tuple[str, str]:
    """``(item as __state.cart holds it, price "d.dd")`` of the cheapest deal on ``deals.html``."""
    text = DEALS.read_text(encoding="utf-8")
    deals: list[tuple[Decimal, str, str]] = []
    for block in re.findall(r'<article class="deal">(.*?)</article>', text, flags=re.S):
        price = re.search(r"\$(\d+\.\d\d)", block)
        item = re.search(r'data-item="([^"]+)"', block)
        if price is None or item is None:
            raise ValueError(f"deals.html: a deal without a price or an item: {block!r}")
        deals.append((Decimal(price.group(1)), item.group(1), price.group(1)))
    deals.sort()
    if not deals or (len(deals) > 1 and deals[0][0] == deals[1][0]):
        raise ValueError(f"deals.html has no single cheapest deal: {deals[:2]}")
    return deals[0][1], deals[0][2]


def _collapse(text: Any) -> str:
    return " ".join(str(text).split())


def _names_cheapest(voiced: str, name: str, price: str) -> bool:
    """The S4 voiced rule: the name, and the price as a number of its own (no digit or point before it, no digit
    after it), in the casefolded, whitespace-collapsed voiced text."""
    said = _collapse(voiced).casefold()
    if _collapse(name).casefold() not in said:
        return False
    return re.search(rf"(?<![\d.]){re.escape(price)}(?!\d)", said) is not None


def score(task_id: str, state: Mapping[str, Any] | None, voiced: str) -> bool:
    """C-G1's success column from the page's ``window.__state`` after the turn (``None``: a failure) and the turn's
    voiced text (the raw voiced sources joined with one space)."""
    if not isinstance(state, Mapping):
        return False
    cart = state.get("cart")
    if task_id in ("S1", "S2"):
        return cart == ["blue mug"]
    if task_id == "S3":
        return state.get("sent") == CONTACT_SENT
    name, price = cheapest_deal()
    if task_id == "S4":
        return cart == [] and _names_cheapest(voiced, name, price)
    if task_id == "S5":
        return cart == [name]
    raise KeyError(f"unknown task {task_id!r}")


# =====================================================================================================
# The report (pure; report.json)
# =====================================================================================================


def _percentile(values: Sequence[Any], p: int) -> Any:
    """Nearest rank: the ceil(p/100 * n)-th smallest; ``None`` for no value."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-p * len(ordered) // 100))
    return ordered[rank - 1]


def _dist(values: Sequence[Any]) -> dict[str, Any]:
    return {"n": len(values), "p50": _percentile(values, 50), "p95": _percentile(values, 95)}


def _key(record: Mapping[str, Any]) -> tuple[str, int]:
    return (str(record["task"]), int(record["repeat"]))


def _voiced_text(trial: Mapping[str, Any]) -> str:
    return " ".join(str(v) for v in trial.get("voiced") or [])


def _attempts_200(step: Mapping[str, Any]) -> bool:
    """The step's request and every earlier attempt of it (``retries``) answered 200."""
    return step.get("http_status") == 200 and all(
        r.get("http_status") == 200 for r in step.get("retries") or []
    )


def _turn_ok(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    """No exception, not timed out, a complete session log, at least one step, and every request of every step
    (retries included) at 200."""
    if trial.get("exception") or trial.get("timed_out") or trial.get("log_complete") is False:
        return False
    return bool(steps) and all(_attempts_200(s) for s in steps)


def build_report(
    trials: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """C-G1's pre-registered metrics over the records (the pinned keys only). Outcomes are scored from each trial's
    ``state`` and ``voiced`` (never its ``success`` flag)."""
    by_turn: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for s in steps:
        by_turn.setdefault(_key(s), []).append(s)
    non_200: list[dict[str, Any]] = []
    for s in steps:
        # record order; a step's earlier attempts (retries) before its own request
        statuses_of = [r.get("http_status") for r in s.get("retries") or []]
        statuses_of.append(s.get("http_status"))
        non_200 += [
            {
                "task": s["task"],
                "repeat": int(s["repeat"]),
                "step": int(s["step"]),
                "http_status": status,
            }
            for status in statuses_of
            if status != 200
        ]
    turns_ok = sum(1 for t in trials if _turn_ok(t, by_turn.get(_key(t), [])))
    outcomes: dict[str, dict[str, int]] = {}
    for task_id in TASK_IDS:
        mine = [t for t in trials if t["task"] == task_id]
        k = sum(1 for t in mine if score(task_id, t.get("state"), _voiced_text(t)))
        outcomes[task_id] = {"k": k, "n": len(mine)}
    statuses = {name: sum(1 for s in steps if s.get("status") == name) for name in STATUSES}
    reasons = Counter(str(s.get("reason")) for s in steps if s.get("status") == "refused")

    def present(name: str, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        return [s[name] for s in rows if s.get(name) is not None]

    by_step = {
        str(k): _dist(present("prefix_hit_tokens", [s for s in steps if int(s["step"]) == k]))
        for k in sorted({int(s["step"]) for s in steps})
    }
    safety = {
        # every request the turn sent (prefill, generate attempts, repairs); the step flag is informational
        "hud_sentinel": sum(len(t.get("sentinel_requests") or []) for t in trials),
        "non_loopback_egress": sum(len(t.get("non_loopback") or []) for t in trials),
        "browser_egress_blocked": sum(len(t.get("browser_egress_blocked") or []) for t in trials),
        "url_path_or_query": sum(len(t.get("url_leaks") or []) for t in trials),
        "executed_but_raised": sum(
            1 for s in steps if s.get("status") == "executed" and s.get("raised") is True
        ),
        "voiced_uncommitted": sum(len(t.get("voiced_uncommitted") or []) for t in trials),
    }
    total = sum(t.n for t in TASKS)
    desk = (
        len(trials) == total
        and turns_ok == total
        and not non_200
        and all(v == 0 for v in safety.values())
    )
    return {
        "acceptance": {
            "steps": len(steps),
            "ok": sum(1 for s in steps if s.get("http_status") == 200),
            "non_200": non_200,
            "turns": len(trials),
            "turns_ok": turns_ok,
        },
        "outcomes": outcomes,
        "steps": {
            "per_turn": _dist([len(by_turn.get(_key(t), [])) for t in trials]),
            "statuses": statuses,
            "refusal_reasons": dict(reasons),
        },
        "latency": {
            "first_commit_ms": _dist(present("first_commit_ms", steps)),
            "done_ms": _dist(present("done_ms", steps)),
        },
        "cache": {
            "hit_tokens": _dist(present("prefix_hit_tokens", steps)),
            "query_tokens": _dist(present("prefix_query_tokens", steps)),
            "by_step": by_step,
        },
        "safety": safety,
        "desk_try": desk,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _dist_cell(d: Mapping[str, Any]) -> str:
    return f"n={d['n']} p50={_fmt(d['p50'])} p95={_fmt(d['p95'])}"


def render_markdown(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    trials: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """``report.md``: every section of ``report.json``, with the regime beside the numbers; with ``trials``, the
    run conditions too (turns started on a busy or unreachable engine, incomplete session logs)."""
    acc = report["acceptance"]
    lines = [
        "# T-0111g C-G1 live smoke (report-only, no bar)",
        "",
    ]
    if manifest is not None:
        lines += ["## Regime", "", "```", json.dumps(manifest.get("regime"), indent=1), "```", ""]
    lines += [
        "## Request acceptance",
        "",
        f"- steps at 200: {acc['ok']}/{acc['steps']}; turns accepted: {acc['turns_ok']}/{acc['turns']}",
    ]
    lines += [
        f"- non-200: {n['task']}/r{n['repeat']} step {n['step']}: HTTP {_fmt(n['http_status'])}"
        for n in acc["non_200"]
    ] or ["- non-200: none"]
    lines += ["", "## Outcomes (successes per task, scored from `__state`)", ""]
    lines += [f"- {task}: {o['k']}/{o['n']}" for task, o in report["outcomes"].items()]
    st = report["steps"]
    lines += [
        "",
        "## Steps",
        "",
        f"- steps per turn: {_dist_cell(st['per_turn'])}",
        "- statuses: " + ", ".join(f"{k} {v}" for k, v in st["statuses"].items()),
    ]
    lines += [f"- refused: {count} x {reason}" for reason, count in st["refusal_reasons"].items()]
    lat = report["latency"]
    lines += [
        "",
        "## Latency (ms, from the request sent; beside T-0115's step regime 0.25-0.37 s)",
        "",
        f"- first commit: {_dist_cell(lat['first_commit_ms'])}",
        f"- done: {_dist_cell(lat['done_ms'])}",
        "",
        "## Prefix cache (`/metrics` delta per step)",
        "",
        f"- hit tokens: {_dist_cell(report['cache']['hit_tokens'])}",
        f"- query tokens: {_dist_cell(report['cache']['query_tokens'])}",
    ]
    lines += [
        f"- hit tokens at step {k}: {_dist_cell(d)}" for k, d in report["cache"]["by_step"].items()
    ]
    lines += ["", "## Safety counts (each expected 0)", ""]
    lines += [
        f"- {k}: {v} (structural cross-check (0 by construction))"
        if k == "executed_but_raised"
        else f"- {k}: {v} (reported only by the harness: a /proc snapshot taken during the run, which the "
        "eval cannot re-derive)"
        if k == "non_loopback_egress"
        else f"- {k}: {v}"
        for k, v in report["safety"].items()
    ]
    if trials is not None:
        busy = sum(1 for t in trials if t.get("engine_idle_before") is False)
        unreachable = sum(
            1 for t in trials if "engine_idle_before" in t and t["engine_idle_before"] is None
        )
        incomplete = sum(1 for t in trials if t.get("log_complete") is False)
        lines += [
            "",
            "## Run conditions (report-only)",
            "",
            f"- turns that started on a busy engine: {busy}/{len(trials)}",
            f"- turns that started with the engine unreachable: {unreachable}/{len(trials)}",
            f"- turns whose session log is incomplete (a drop marker or no log.stopped): "
            f"{incomplete}/{len(trials)}",
        ]
    lines += [
        "",
        "## Desk reading",
        "",
        f"- desk_try: {report['desk_try']} (25/25 turns accepted, every step at 200, every safety count 0; "
        "success counts are reported, not gated)",
        "",
    ]
    return "\n".join(lines)


# =====================================================================================================
# Record extraction (pure, over one session's log records)
# =====================================================================================================


def g6_findings(
    tag: str,
    *,
    steps: Sequence[Mapping[str, Any]],
    thought_commits: Sequence[str],
    played_runs: Sequence[str],
    history: str | None,
    finished: bool,
    special_tokens: Sequence[str],
) -> list[str]:
    """T-0115's G6 (``g6_violations``), per turn: a voiced text at a step >= 2 that executed its call; a special
    literal or thought marker in a voiced source; a voiced source inside a thought commit and not in its step text
    (whitespace-collapsed); and, for a finished turn, a stored history entry that is not the join of the played
    runs."""
    out: list[str] = []
    thoughts = [_collapse(t) for t in thought_commits if str(t).strip()]
    for s in steps:
        voiced = [str(v) for v in s.get("voiced") or [] if str(v).strip()]
        step_text = _collapse(s.get("step_text") or "")
        if int(s["step"]) >= 2 and s.get("executed") and voiced:
            out.append(f"{tag}: voiced at step {s['step']}, which executed its call: {voiced!r}")
        for src in [str(v) for v in s.get("voiced_sources") or []]:
            hits = sorted({lit for lit in (*special_tokens, *THOUGHT_MARKERS) if lit in src})
            if hits:
                out.append(f"{tag}: voiced special literal(s) {hits} at step {s['step']}")
            text = _collapse(src)
            if text and any(text in t for t in thoughts) and text not in step_text:
                out.append(
                    f"{tag}: voiced thought text not in the step text at step {s['step']}: {text!r}"
                )
    if finished:
        joined = " ".join(str(p) for p in played_runs if str(p)) or None
        if (history or None) != joined:
            out.append(f"{tag}: history {history!r} is not the join of the played runs {joined!r}")
    return out


def url_leaks(lines: Sequence[tuple[str, str]], origin: str | None) -> list[str]:
    """``"<file>:<n> <marker>"`` for each record line holding a start-page path or query (B15), or the start-page
    origin followed by a path."""
    markers = list(URL_MARKERS)
    out: list[str] = []
    for where, line in lines:
        hits = [m for m in markers if m in line]
        if origin and re.search(re.escape(origin) + r"/[^\s\"']", line):
            hits.append(origin + "/...")
        out += [f"{where} {m}" for m in hits]
    return out


def sentinel_hits(bodies: Mapping[str, str], sentinel: bytes) -> list[str]:
    """The labels (``"<phase>:<attempt_id>"``, phase prefill, generate or repair), sorted, whose serialised request
    body holds ``base64(sentinel)``: the HUD sentinel frame's bytes reached the engine in that request."""
    needle = base64.b64encode(sentinel).decode("ascii")
    return sorted(label for label, body in bodies.items() if needle in body)


def log_complete(records: Sequence[Mapping[str, Any]]) -> bool:
    """The session log is whole: it has its ``log.stopped`` record, and no record carries a drop marker (a
    ``_drops`` key, or the ``dropped`` gap record the writer puts on disk for it)."""
    stopped = False
    for r in records:
        if "_drops" in r or r.get("kind") == "dropped":
            return False
        if r.get("kind") == "log.stopped":
            stopped = True
    return stopped


def _spoken_prefix(raw: str) -> str:
    from local_astra.toolcall import router

    return router.route(raw).spoken_prefix.strip()


def _special_tokens() -> tuple[str, ...]:
    from local_astra.toolcall.special_tokens import SPECIAL_TOKENS

    return tuple(SPECIAL_TOKENS)


def turn_records(
    task_id: str,
    repeat: int,
    *,
    records: Sequence[Mapping[str, Any]],
    line: Mapping[str, Any] | None = None,
    generates: Sequence[Mapping[str, Any]] = (),
    sentinel_requests: Sequence[str] = (),
    scanned_requests: Sequence[str] = (),
    raised_calls: Sequence[str] = (),
    state: Any = None,
    history: str | None = None,
    exception: str | None = None,
    timed_out: bool = False,
    non_loopback: Sequence[str] = (),
    leaks: Sequence[str] = (),
    engine_idle_before: bool | None = None,
    route: Callable[[str], str] | None = None,
    special_tokens: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The trial line and the step lines of one turn (the pinned keys, plus a few diagnostic ones), from its
    session's log records, its ``turns.jsonl`` line and ``LoopMeter``'s rows (keyed by the engine attempt id).
    A step is its last attempt; its earlier attempts are its ``retries``. ``sentinel_requests`` are
    ``sentinel_hits``'s labels for the turn. ``route`` maps a step's raw committed answer to its spoken prefix
    (default ``router.route(raw).spoken_prefix``)."""
    route = route if route is not None else _spoken_prefix
    special_tokens = special_tokens if special_tokens is not None else _special_tokens()
    data = [(r.get("kind"), r.get("data") or {}, r) for r in records]
    requests = [d for kind, d, _ in data if kind == "llm.request"]
    by_attempt = {str(d.get("attempt_id")): d for d in requests}
    step_of: dict[str, int] = {}
    for d in requests:
        if d.get("phase") in ("generate", "tool_step"):
            mid = str(d.get("request_id"))
            step_of[mid] = 1 if d.get("phase") == "generate" else int(d.get("step") or 0)
    turn_id = str((line or {}).get("turn_id") or "") or next(
        (str(r.get("turn_id")) for _, _, r in data if r.get("turn_id")), ""
    )

    rows_by_step: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
    for g in generates:
        attempt = str(g.get("request_id"))
        req = by_attempt.get(attempt)
        if req is None or req.get("phase") not in ("generate", "tool_step"):
            continue
        k = step_of.get(str(req.get("request_id")))
        if k:
            rows_by_step.setdefault(k, []).append((attempt, g))

    voiced: dict[int, list[str]] = {}
    sources: dict[int, list[str]] = {}
    executed_ids: set[str] = set()
    played_runs: list[str] = []
    current = 1
    for kind, d, _ in data:
        if kind != "machine.step":
            continue
        event = d.get("event") or {}
        rid = event.get("request_id")
        if isinstance(rid, str) and rid in step_of:
            current = step_of[rid]
        if event.get("type") in ("playback.drained", "playback.stopped"):
            played_runs.append(str(event.get("played_text") or ""))
        for c in d.get("commands") or []:
            if not isinstance(c, Mapping):
                continue
            if c.get("type") == "tts.speak" and str(c.get("text") or ""):
                voiced.setdefault(current, []).append(str(c.get("text")))
                sources.setdefault(current, []).append(str(c.get("source") or ""))
            elif c.get("type") == "tool.execute":
                executed_ids.add(str(c.get("call_id")))
    calls: dict[str, list[Mapping[str, Any]]] = {}
    results: dict[str, Mapping[str, Any]] = {}
    for kind, d, _ in data:
        if kind == "tool.call":
            calls.setdefault(str(d.get("call_id")), []).append(d)
        elif kind == "tool.result":
            results[str(d.get("call_id"))] = d
    commits = [d for kind, d, _ in data if kind == "llm.commit"]
    thought_commits = [
        str(d.get("text"))
        for d in commits
        if d.get("segment") == "thought" and str(d.get("text") or "").strip()
    ]
    sentinel = set(sentinel_requests)
    raised = set(raised_calls)

    steps: list[dict[str, Any]] = []
    for k in sorted(rows_by_step):
        attempt, g = rows_by_step[k][-1]
        retries = [
            {
                "http_status": r.get("http_status") if r.get("http_status") is not None else 0,
                "attempt_id": a,
            }
            for a, r in rows_by_step[k][:-1]
        ]
        mine = [d for d in commits if str(d.get("attempt_id")) == attempt]
        raw = "".join(
            str(d.get("text") or "")
            for d in mine
            if d.get("segment") == "answer" and not d.get("thought_open")
        )
        cid = f"{turn_id}-call" if k == 1 else f"{turn_id}-call-s{k}"
        status: str | None = None
        reason: str | None = None
        records_of_call = calls.get(cid, [])
        if cid in results:
            status = str(results[cid].get("status"))
        elif any(r.get("status") == "refused" for r in records_of_call):
            status = "refused"  # refused before any execution (the judge)
        if status is not None and status != "executed":
            reason = next(
                (
                    str(r.get("reason"))
                    for r in reversed(records_of_call)
                    if r.get("reason") is not None
                ),
                None,
            )
        http = g.get("http_status")
        steps.append(
            {
                "task": task_id,
                "repeat": repeat,
                "step": k,
                "http_status": http if http is not None else 0,
                "status": status,
                "reason": reason,
                "raised": cid in raised,
                # informational; the safety count is the trial's sentinel_requests
                "hud_sentinel": any(f"generate:{a}" in sentinel for a, _ in rows_by_step[k]),
                "first_commit_ms": g.get("first_commit_ms"),
                "done_ms": g.get("done_ms"),
                "prefix_hit_tokens": g.get("prefix_hit_tokens"),
                "prefix_query_tokens": g.get("prefix_query_tokens"),
                "voiced_sources": list(sources.get(k, [])),
                "step_text": route(raw),
                "retries": retries,
                # diagnostics (not read by the report)
                "attempt_id": attempt,
                "call_id": cid if (records_of_call or cid in results) else None,
                "call_name": (str(records_of_call[0].get("name") or "") or None)
                if records_of_call
                else None,
                "executed": cid in executed_ids,
                "voiced": list(voiced.get(k, [])),
                "error": g.get("error"),
            }
        )

    all_sources = [src for k in sorted(sources) for src in sources[k]]
    finished = not exception and not timed_out
    findings = g6_findings(
        f"{task_id}/r{repeat}",
        steps=steps,
        thought_commits=thought_commits,
        played_runs=played_runs,
        history=history,
        finished=finished,
        special_tokens=special_tokens,
    )
    extras = (line or {}).get("extras") or {}
    blocked = extras.get("browser_egress_blocked") or []
    trial: dict[str, Any] = {
        "task": task_id,
        "repeat": repeat,
        "state": state if isinstance(state, Mapping) else None,
        "voiced": all_sources,
        "success": score(
            task_id, state if isinstance(state, Mapping) else None, " ".join(all_sources)
        ),
        "exception": exception,
        "timed_out": timed_out,
        "non_loopback": list(non_loopback),
        "browser_egress_blocked": list(blocked),
        "url_leaks": list(leaks),
        "voiced_uncommitted": findings,
        "sentinel_requests": sorted(sentinel),
        # every body the sentinel scan read, same labels: the eval checks each logged attempt is among them
        "scanned_requests": sorted(set(scanned_requests)),
        "engine_idle_before": engine_idle_before,
        "log_complete": log_complete(records),
        # diagnostics
        "turn_id": turn_id or None,
        "session_id": (line or {}).get("session_id"),
        "tool_steps": extras.get("tool_steps"),
        "tool_steps_executed": extras.get("tool_steps_executed"),
        "tool_loop_end": extras.get("tool_loop_end"),
        "frame_source": extras.get("frame_source"),
        "browser": extras.get("browser"),
        "history": history,
    }
    return trial, steps


def _not_run(task_id: str, repeat: int, reason: str) -> dict[str, Any]:
    return {
        "task": task_id,
        "repeat": repeat,
        "state": None,
        "voiced": [],
        "success": False,
        "exception": reason,
        "timed_out": False,
        "non_loopback": [],
        "browser_egress_blocked": [],
        "url_leaks": [],
        "voiced_uncommitted": [],
        "sentinel_requests": [],
        "scanned_requests": [],
        "engine_idle_before": None,
        "log_complete": False,  # no session ran, so there is no log
    }


def _errored(trial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> bool:
    if trial.get("exception") or trial.get("timed_out"):
        return True
    return not all(_attempts_200(s) for s in steps)


# =====================================================================================================
# Refusals (no socket)
# =====================================================================================================


def check_loopback(base_url: str) -> None:
    """``http(s)`` to an IP literal in 127.0.0.0/8 or ``::1``; any name (``localhost`` too) is refused without being
    resolved, and so is a v4-mapped v6 address. ``ValueError`` otherwise."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http(s): {base_url!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"base_url host {host!r} is not a loopback IP literal") from None
    ok = (
        addr in ipaddress.ip_network("127.0.0.0/8")
        if addr.version == 4
        else addr == ipaddress.ip_address("::1")
    )
    if not ok:
        raise ValueError(f"base_url host {host!r} is not in 127.0.0.0/8 or ::1")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied)
# =====================================================================================================


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A 3xx is never followed."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _get_text(url: str, timeout_s: float = 10.0) -> str | None:
    try:
        with _opener().open(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None
            body: bytes = resp.read()
            return body.decode("utf-8", errors="replace")
    except (OSError, ValueError, http.client.HTTPException):
        return None


def read_metrics(base_url: str) -> dict[str, int] | None:
    """The prefix-cache counters and the running/waiting gauges off ``/metrics`` (summed over label sets);
    ``None`` when the endpoint or a counter is unavailable. Reading it spends nothing."""
    text = _get_text(_root(base_url) + "/metrics")
    if text is None:
        return None
    out: dict[str, int] = {}
    for name in (METRIC_QUERIES, METRIC_HITS, METRIC_RUNNING, METRIC_WAITING):
        pat = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$"
        vals = [float(m.group(1)) for m in re.finditer(pat, text, re.MULTILINE)]
        if not vals and name in (METRIC_QUERIES, METRIC_HITS):
            return None
        out[name] = round(sum(vals))
    return out


async def _wait_idle(base_url: str) -> bool | None:
    """Before a turn (the trial's ``engine_idle_before``): ``True`` once ``/metrics`` shows nothing running or
    waiting; ``False`` when it still shows work after ``QUIESCE_S`` (the turn starts on a busy engine); ``None`` when
    ``/metrics`` never answered within ``UNREACHABLE_AFTER_S`` (unreachable)."""
    started = time.monotonic()
    reached = False
    while True:
        m = await asyncio.to_thread(read_metrics, base_url)
        reached = reached or m is not None
        if m is not None and m[METRIC_RUNNING] + m[METRIC_WAITING] == 0:
            return True
        if not reached and time.monotonic() - started >= UNREACHABLE_AFTER_S:
            return None
        if time.monotonic() - started >= QUIESCE_S:
            return False
        await asyncio.sleep(0.2)


def _http_status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


class LoopMeter:
    """T-0115's ``LoopMeter``: the real client plus a ``/metrics`` delta and two latencies per generate. The
    ``/metrics`` reads happen before the request is sent and after the stream ended, outside the timed span;
    ``first_commit_ms`` and ``done_ms`` run from the moment the request is sent. ``http_status`` is 200 when the
    stream ran, the engine's status when it raised one, 0 for a failure without a response, ``None`` when the
    generate was cancelled. Prefills run to their end (aborts deferred, J12) and a generate other than the repair
    waits for any still running before its before-read. ``requests`` keeps the prompt of every request it forwards,
    under ``"<phase>:<attempt_id>"`` (prefill, generate -- every attempt, retries included -- and repair), for the
    sentinel scan after the session closed."""

    def __init__(
        self,
        inner: Any,
        *,
        metrics: Callable[[], Mapping[str, int] | None],
        clock: Callable[[], float],
    ) -> None:
        self.inner = inner
        self.metrics = metrics
        self.clock = clock
        self.generates: list[dict[str, Any]] = []
        self.requests: dict[str, Prompt] = {}
        self._prefills: dict[str, asyncio.Task[PrefillHandle]] = {}
        self.deferred_aborts: list[str] = []

    async def prefill(self, prefix: Prompt, request_id: str) -> PrefillHandle:
        self.requests[f"prefill:{request_id}"] = prefix
        task = asyncio.ensure_future(self.inner.prefill(prefix, request_id))
        self._prefills[request_id] = task
        handle: PrefillHandle = await asyncio.shield(task)
        return handle

    def generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: CdRequest | None = None,
    ) -> AsyncIterator[Commit]:
        return self._generate(prompt, request_id, options, xargs, cd)

    async def _read(self) -> dict[str, int] | None:
        got = await asyncio.to_thread(self.metrics)
        return dict(got) if got is not None else None

    async def _generate(
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None,
        xargs: Mapping[str, str] | None,
        cd: CdRequest | None,
    ) -> AsyncIterator[Commit]:
        row: dict[str, Any] = {
            "request_id": request_id,
            "repair": cd is not None,
            "prefill_wait_ms": None,
            "prefix_query_tokens": None,
            "prefix_hit_tokens": None,
            "first_commit_ms": None,
            "done_ms": None,
            "http_status": None,
            "error": None,
        }
        self.generates.append(row)
        self.requests[f"{'repair' if cd is not None else 'generate'}:{request_id}"] = prompt
        if cd is None:
            pending = [t for t in self._prefills.values() if not t.done()]
            if pending:
                waited = self.clock()
                await asyncio.wait(pending, timeout=PREFILL_WAIT_S)
                row["prefill_wait_ms"] = round((self.clock() - waited) * 1000.0, 1)
        before = await self._read()
        sent = self.clock()
        try:
            async for commit in self.inner.generate(prompt, request_id, options, xargs, cd=cd):
                if row["first_commit_ms"] is None:
                    row["first_commit_ms"] = round((self.clock() - sent) * 1000.0, 1)
                yield commit
            row["done_ms"] = round((self.clock() - sent) * 1000.0, 1)
            row["http_status"] = 200
        except asyncio.CancelledError:
            row["error"] = "cancelled"
            raise
        except GeneratorExit:
            row["error"] = "closed by the consumer"
            raise
        except Exception as exc:
            status = _http_status(exc)
            row["http_status"] = status if status is not None else 0
            row["error"] = repr(exc)
            raise
        finally:
            if row["error"] not in ("cancelled", "closed by the consumer"):
                after = await self._read()
                if before is not None and after is not None:
                    row["prefix_query_tokens"] = after[METRIC_QUERIES] - before[METRIC_QUERIES]
                    row["prefix_hit_tokens"] = after[METRIC_HITS] - before[METRIC_HITS]

    async def tokenize(self, text: str) -> list[int]:
        ids: list[int] = await self.inner.tokenize(text)
        return ids

    async def abort(self, request_id: str) -> None:
        task = self._prefills.get(request_id)
        if task is not None and not task.done():
            self.deferred_aborts.append(request_id)
            return
        await self.inner.abort(request_id)

    async def aclose(self) -> None:
        for task in self._prefills.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._prefills.values(), return_exceptions=True)
        await self.inner.aclose()


class _Watch:
    """The live browser seam the session gets: delegates every name to the real ``BrowserToolExecutor`` and records,
    per call id, each execute's outcome and whether the call raised (the ``raised`` step field)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.raised: list[str] = []
        self.outcomes: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def execute(self, inv: Any) -> Any:
        try:
            out = await self._inner.execute(inv)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self.raised.append(str(inv.call_id))
            raise
        self.outcomes[str(inv.call_id)] = out
        return out


class _PlaybackOnly:
    """``FakeAudioIO`` with only the ``AudioIO`` playback seams (no capture pump for typed turns; T-0093)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        await self.inner.pause()

    async def resume(self) -> None:
        await self.inner.resume()

    async def flush(self) -> int:
        played: int = await self.inner.flush()
        return played

    def played_ms(self) -> float:
        return float(self.inner.played_ms())

    def mark_drain(self) -> None:
        self.inner.mark_drain()

    def playback_events(self) -> Any:
        return self.inner.playback_events()

    def is_rendering(self, t_ms: float) -> bool:
        return bool(self.inner.is_rendering(t_ms))

    def capture(self) -> Any:
        return self.inner.capture()


class _CaseMemory:
    """No recollections; never an uncaptioned episode, so the idle captioner sends no request (T-0093)."""

    def __init__(self) -> None:
        self._next = 0

    async def ingest(self, episode: Episode) -> int:
        self._next += 1
        return self._next

    async def search(self, query: MemoryQuery) -> Sequence[MemoryHit]:
        return ()

    async def set_caption(self, episode_id: int, caption: str) -> None:
        return None

    async def forget_since(self, t0_ms: int) -> int:
        return 0

    async def next_uncaptioned(self) -> Episode | None:
        return None


def _fake(module: str, name: str) -> Any:
    """A class from ``tests/fakes`` (imported by name: the fakes are not typed to --strict)."""
    return getattr(importlib.import_module(f"tests.fakes.{module}"), name)


# --- sockets of this process and its children (C-E1's check, read from /proc) ------------------------


def _addr(hex_addr: str) -> bytes:
    ip, _port = hex_addr.split(":")
    raw = bytes.fromhex(ip)
    if len(raw) == 4:
        return raw[::-1]
    return b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4))


def _loopback(raw: bytes) -> bool:
    if len(raw) == 4:
        return raw[0] == 127 or raw == b"\0\0\0\0"
    if raw == bytes(15) + b"\x01" or raw == bytes(16):
        return True
    return raw[:12] == b"\0" * 10 + b"\xff\xff" and raw[12] == 127


def _descendants() -> list[int]:
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            with contextlib.suppress(OSError, IndexError, ValueError):
                stat = Path(f"/proc/{d}/stat").read_text()
                children.setdefault(int(stat.rsplit(")", 1)[1].split()[1]), []).append(int(d))
    out, todo = [], [os.getpid()]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))
    return out


def non_loopback_sockets(pids: Sequence[int]) -> list[str]:
    """Sockets of ``pids`` whose local bind or peer is off loopback (a bind to ``0.0.0.0``/``::`` counts; a
    listener's ``0.0.0.0:0`` peer is no peer)."""
    inodes: set[str] = set()
    for pid in pids:
        with contextlib.suppress(OSError):
            for fd in os.listdir(f"/proc/{pid}/fd"):
                with contextlib.suppress(OSError):
                    target = os.readlink(f"/proc/{pid}/fd/{fd}")
                    if target.startswith("socket:["):
                        inodes.add(target[8:-1])
    bad: list[str] = []
    for table in ("tcp", "tcp6", "udp", "udp6"):
        with contextlib.suppress(OSError):
            for line in Path(f"/proc/{os.getpid()}/net/{table}").read_text().splitlines()[1:]:
                f = line.split()
                if len(f) < 10 or f[9] not in inodes:
                    continue
                local, peer = _addr(f[1]), _addr(f[2])
                if local in (b"\0\0\0\0", bytes(16)) or not _loopback(local):
                    bad.append(f"{table} local {f[1]}")
                if not _loopback(peer):
                    bad.append(f"{table} peer {f[2]}")
    return bad


def _sentinel_jpeg() -> bytes:
    """The HUD sentinel frame: a synthetic 1280x800 JPEG no page renders (magenta, with a grey bar)."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1280, 800), (255, 0, 255))
    ImageDraw.Draw(img).rectangle((0, 380, 1279, 420), fill=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Profile
    sentinel: bytes
    manifest: dict[str, Any]
    stopped: str | None = None
    turn_s: list[float] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def _append_jsonl(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


STATE_JS = (
    "(() => window.__state === undefined ? null : JSON.parse(JSON.stringify(window.__state)))()"
)


async def _read_state(browser: Any) -> Any:
    """``window.__state`` of the active page, or ``None`` when it cannot be read (never an exception)."""
    try:
        return await asyncio.wait_for(
            browser.driver.page.evaluate(STATE_JS), timeout=BROWSER_CALL_S
        )
    except Exception:
        return None


async def _one_turn(
    run: _Run, task_id: str, repeat: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One (task, repeat): a fresh start-page server, a fresh headless browser opened at the task's start page, a
    fresh ``Session`` with one ``UserText``; the records read after the session closed."""
    import httpx

    from local_astra import cli
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.pages.server import StartPageServer
    from local_astra.telemetry.session_log import SessionLog

    task = _task(task_id)
    idle_before = await _wait_idle(run.base_url)
    clock = SessionClock()  # before the executor (Implementation notes, Event loop)
    pages = StartPageServer()
    pages.start()
    browser: Any = None
    watch: _Watch | None = None
    session: Any = None
    meter: LoopMeter | None = None
    inner: Any = None
    exception: str | None = None
    timed_out = False
    history: str | None = None
    state: Any = None
    sockets: list[str] = []
    log: Any = None
    origin = pages.origin
    try:
        browser = BrowserToolExecutor(
            now_ms=lambda: clock.now_ms,
            viewport=run.profile.browser_viewport,
            guard=None,
            headless=True,
            end_when_closed=False,
            allowed_origins=frozenset({origin}),
        )
        watch = _Watch(browser)
        kit, detokenizer = cli.load_tool_kit(run.profile, live_browser=watch)
        if kit.repair_off is not None:
            raise RuntimeError(f"the tool repair is off for this turn ({kit.repair_off})")
        inner = VllmClient(
            VllmEndpoint(base_url=run.base_url),
            transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
            detokenizer=detokenizer,
        )
        meter = LoopMeter(inner, metrics=lambda: read_metrics(run.base_url), clock=time.monotonic)
        hud = Frame(
            source=Source.SCREEN,
            ts_ms=0,
            width=1280,
            height=800,
            jpeg=run.sentinel,
            phash=hashlib.sha256(run.sentinel).hexdigest()[:16],
        )
        sessions_dir = run.out / "sessions"
        log = SessionLog(sessions_dir)
        policy = TurnPolicy(
            thinking=run.profile.diffusion.thinking, tool_loop=True, max_tool_steps=MAX_TOOL_STEPS
        )
        session = Session(
            TurnMachine(policy),
            meter,
            _fake("stt", "FakeStt")(),
            _fake("tts", "FakeTts")(),
            _PlaybackOnly(_fake("audio", "FakeAudioIO")(clock)),
            _fake("perception", "FakePerception")([hud]),
            _CaseMemory(),
            clock=clock,
            config=SessionConfig(
                run.out / "data",
                turns_path=run.out / "turns.jsonl",
                profile=run.profile,
                audio_path="null",
                source=Source.SCREEN,
            ),
            session_log=log,
            log_dir=sessions_dir,
            tool_kit=kit,
        )
        await asyncio.wait_for(browser.open(origin + task.start_page), timeout=BROWSER_CALL_S)
        await session.start()
        session.turn_written.clear()
        session.submit(E.UserText(t_ms=clock.now_ms, text=task.request))
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=TURN_TIMEOUT_S)
        except TimeoutError:
            timed_out = True
        for entry in reversed(tuple(session.state.history)):
            if entry.user_text.strip() == task.request.strip():
                history = entry.assistant_spoken or None
                break
    except Exception as exc:
        exception = repr(exc)
    finally:
        # read while the browser is still open (the session's close closes it)
        if browser is not None:
            state = await _read_state(browser)
        sockets = non_loopback_sockets(_descendants())
        if session is not None:
            with contextlib.suppress(Exception):
                await session.aclose()
        if meter is not None:
            with contextlib.suppress(Exception):
                await meter.aclose()
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        pages.stop()
    if session is not None and session.failures and exception is None:
        exception = "session failures: " + "; ".join(repr(f) for f in session.failures)

    # --- after close, outside every timed span: which request bodies held the HUD sentinel's bytes --------------
    # (every request the meter forwarded: the prefill, every generate attempt, every repair)
    sentinel_requests: list[str] = []
    scanned_requests: list[str] = []
    generates: list[dict[str, Any]] = []
    if meter is not None:
        generates = [g for g in meter.generates if not g.get("repair")]
        bodies = {
            label: json.dumps(inner.request_body(prompt, label.split(":", 1)[1]))
            for label, prompt in meter.requests.items()
        }
        sentinel_requests = sentinel_hits(bodies, run.sentinel)
        scanned_requests = sorted(bodies)

    records: list[dict[str, Any]] = []
    record_lines: list[tuple[str, str]] = []
    if log is not None and Path(log.path).is_file():
        text = Path(log.path).read_text(encoding="utf-8")
        records = sorted(_read_jsonl(Path(log.path)), key=lambda r: int(r.get("seq", 0)))
        record_lines += [
            (f"sessions/{Path(log.path).name}:{n}", ln)
            for n, ln in enumerate(text.splitlines(), start=1)
        ]
    lines = []
    if session is not None:
        turns_path = run.out / "turns.jsonl"
        if turns_path.is_file():
            for n, ln in enumerate(turns_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not ln.strip():
                    continue
                item = json.loads(ln)
                if isinstance(item, dict) and item.get("session_id") == session.session_id:
                    lines.append(item)
                    record_lines.append((f"turns.jsonl:{n}", ln))
    trial, steps = turn_records(
        task_id,
        repeat,
        records=records,
        line=lines[0] if lines else None,
        generates=generates,
        sentinel_requests=sentinel_requests,
        scanned_requests=scanned_requests,
        raised_calls=list(watch.raised) if watch is not None else [],
        state=state,
        history=history,
        exception=exception,
        timed_out=timed_out,
        non_loopback=sockets,
        leaks=url_leaks(record_lines, origin),
        engine_idle_before=idle_before,
    )
    trial["session_log"] = Path(log.path).name if log is not None else None
    trial["deferred_prefill_aborts"] = len(meter.deferred_aborts) if meter is not None else 0
    return trial, steps


async def _run_all(run: _Run) -> None:
    """Every turn of ``order(ORDER_SEED)``; after ``ERRORED_STOP`` consecutive errored turns nothing more is sent
    and the rest are written as not run (errored), so the files still hold 25 trial lines."""
    errored = 0
    for index, (task_id, repeat) in enumerate(order(ORDER_SEED)):
        started = time.monotonic()
        steps: list[dict[str, Any]]
        if run.stopped is not None:
            trial, steps = _not_run(task_id, repeat, f"not run: {run.stopped}"), []
        else:
            try:
                trial, steps = await _one_turn(run, task_id, repeat)
            except Exception as exc:
                trial, steps = _not_run(task_id, repeat, f"harness: {exc!r}"), []
            errored = errored + 1 if _errored(trial, steps) else 0
            if errored >= ERRORED_STOP:
                run.stopped = f"{errored} consecutive errored turns (last: {task_id}/r{repeat})"
                run.manifest["stopped"] = run.stopped
                _write_json(run.out / "manifest.json", run.manifest)
                _say(f"stopping: {run.stopped}")
        _append_jsonl(run.out / "trials.jsonl", [trial])
        _append_jsonl(run.out / "steps.jsonl", steps)
        run.turn_s.append(round(time.monotonic() - started, 1))
        _say(
            f"{index:02d} {task_id}/r{repeat} steps={len(steps)} "
            f"statuses={[s.get('status') for s in steps]} success={trial.get('success')} "
            f"timed_out={trial.get('timed_out')} err={trial.get('exception')} wall={run.turn_s[-1]}s"
        )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    # 1. loopback, unresolved
    try:
        check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    # 2. a clean tree
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    # 3. no out-dir yet (empty or not: one run per SHA)
    out: Path = args.out_dir or REPO / "logs" / f"t0111g-live-smoke-{sha[:7]}"
    if out.exists():
        _say(
            f"refusing: {out} exists; the run happens once per SHA (move it aside only with the human)"
        )
        return EXIT_REFUSED
    # 4. the task table's pin
    table_sha = task_table_sha()
    if table_sha != TASK_TABLE_SHA256:
        _say(f"refusing: the task table's sha is {table_sha}, not the pinned {TASK_TABLE_SHA256}")
        return EXIT_REFUSED
    # 5. the tool kit (T-0115's three refusals, repair off included)
    from local_astra import cli
    from local_astra.config import load

    profile = load().profiles["lean"]
    try:
        kit, _detokenizer = cli.load_tool_kit(profile)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    if str(kit.tools_version)[:12] != TOOLS_SHA12:
        _say(f"refusing: the tool set is {str(kit.tools_version)[:12]}, not {TOOLS_SHA12}")
        return EXIT_REFUSED
    if kit.repair_off is not None:
        _say(f"refusing: the tool repair is off ({kit.repair_off}); the regime says repair on")
        return EXIT_REFUSED
    # 6. Chromium (group B's availability check; reads the filesystem only)
    missing = cli.live_prerequisites(BROWSER)
    if missing is not None:
        _say(f"refusing: Chromium ({BROWSER}) is not available: {missing}")
        return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint

    model = VllmEndpoint().model
    listing = _get_text(args.base_url.rstrip("/") + "/models")
    served: list[str] = []
    try:
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    except (ValueError, KeyError, TypeError):
        served = []
    if model not in served:
        _say(
            f"no engine serving {model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    if read_metrics(args.base_url) is None:
        _say(
            f"{_root(args.base_url)}/metrics has no prefix-cache counters; the per-step hit cannot be measured"
        )
        return EXIT_FAILED

    sentinel = _sentinel_jpeg()
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    try:
        from importlib.metadata import version

        playwright_version: str | None = version("playwright")
    except Exception:
        playwright_version = None
    manifest: dict[str, Any] = {
        "regime": {
            "commit": sha,
            "base_url": args.base_url,
            "served_model": model,
            "engine_version": _get_text(_root(args.base_url) + "/version"),
            "profile": profile.name,
            "canvas_length": profile.diffusion.canvas_length,
            "max_denoising_steps": profile.diffusion.max_denoising_steps,
            "thinking": profile.diffusion.thinking,
            "tools": f"native:{str(kit.tools_version)[:12]}:live:loop{MAX_TOOL_STEPS}",
            "repair": "on",
            "max_tool_steps": MAX_TOOL_STEPS,
            "browser": f"{BROWSER}, one per turn, fresh context, loopback start pages only",
            "viewport": list(profile.browser_viewport),
            "playwright": playwright_version,
            "turns": len(order(ORDER_SEED)),
            "concurrency": 1,
            "order_seed": ORDER_SEED,
            "engine_seed": None,
            "turn_timeout_s": TURN_TIMEOUT_S,
            "input": "typed (UserText); fake STT/TTS/audio; perception holds only the HUD sentinel frame; "
            "no audio device, no headset or speakers",
            "versions_lock_sha256": _sha256(REPO / "services" / "versions.lock"),
        },
        "task_table_sha256": table_sha,
        "tasks": [
            {"task_id": t.task_id, "start_page": t.start_page, "request": t.request, "n": t.n}
            for t in TASKS
        ],
        "cheapest_deal": list(cheapest_deal()),
        "hud_sentinel_sha256": hashlib.sha256(sentinel).hexdigest(),
        "gate_file_sha256": {rel: _sha256(REPO / rel) for rel in GATE_FILES},
        "order": [list(k) for k in order(ORDER_SEED)],
    }
    _write_json(out / "manifest.json", manifest)
    for name in ("trials.jsonl", "steps.jsonl", "turns.jsonl"):
        (out / name).touch()
    run = _Run(
        base_url=args.base_url, out=out, profile=profile, sentinel=sentinel, manifest=manifest
    )
    asyncio.run(_run_all(run))

    trials = _read_jsonl(out / "trials.jsonl")
    steps = _read_jsonl(out / "steps.jsonl")
    report = build_report(trials, steps)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report, manifest, trials), encoding="utf-8")
    _say(f"desk_try={report['desk_try']} outcomes={report['outcomes']}")
    _say(f"written to {out}")
    return EXIT_FAILED if run.stopped is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
