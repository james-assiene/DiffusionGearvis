"""T-0099 -- the live tool-call repair, on elicited invalid drafts (arm S) and forced valid drafts (arm F). ``/launch`` ONLY.

    uv run python scripts/gpu/t0099_repair_live.py --base-url http://127.0.0.1:8001/v1

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never boots
one. **Importing this file opens no socket and starts nothing**: the report half (``build_report``, ``gates``,
``render_markdown``) reads files only, and everything that talks to the engine is imported inside ``main``. The
T-0093 harness is imported read-only, by path, for the helpers the task names (``_MeteredLlm``, ``_Turn``, ...).

Refusals (exit 2), in this order and all **before any socket**: T-0093's (a ``--base-url`` that is not ``http(s)``
to a loopback IP literal; a dirty tree; an existing, non-empty ``--out-dir``; a v1 case file or Q12 file that is not
the pinned one, or a dev split that is not 40 tool + 8 no-tool cases; a tool kit that cannot load, whose repair is
off, or whose tool set is not ``bbfc53ccdd80``), plus: a stress file (``STRESS_PATH``) whose sha256 is not
``STRESS_SHA256`` (H-T99a).

What it runs (docs/tasks/T-0099.md criteria 2-4):

- **Arm S** (stress): the 30 cases of ``tests/eval/toolcall_stress/cases.jsonl`` in file order, ``K`` = 3 times,
  repetition-major (all 30, then all 30 again, ...). **Arm F** (forced): the 40 dev tool cases of
  ``tests/eval/toolcall/cases.jsonl`` in file order, once each. Concurrency 1, one boot, no seed.
- **The session.** As T-0093 builds it: one fresh ``Session`` per (case, repetition, arm) with one typed turn; the real
  ``VllmClient`` (unproxied, loopback) behind T-0093's ``_MeteredLlm``; fake STT, TTS (300 ms/word), audio and
  perception (the case's screen as the frame); no recollections; tools on (``native:bbfc53ccdd80:dry_run``), repair
  on.
- **The forcing (criterion 3, H-T99c).** Each arm-F session runs inside ``forced_classify()``: there
  ``local_astra.toolcall.live.classify`` turns a ``valid`` verdict into ``repairable`` with ``FORCED_REASON``; the
  previous object is restored in a ``finally``. Arm S never runs inside it. The manifest records ``forced`` per session.
- **Barge-in during a repair (criterion 4).** On arm-F cases in ``BARGE_IN_CANDIDATES`` (the dev tool cases that spoke a
  prefix at 4b0bd92), in run order, at most ``BARGE_IN_MAX_ATTEMPTS`` attempts (ruling Q8): once the turn is SPEAKING
  **and** the live generate is the turn's repair request, one ``UserSpeechOnset`` is submitted; the machine's own
  takeover timer does the rest (no synthetic takeover). Delivered = the line reads ``outcome == "interrupted"``,
  carries a call, and the session log has the ``LlmAbort`` of the repair request. An attempt whose repair verdict is
  accepted while the turn is PAUSED is counted as "repair done before the takeover". Fewer than ``BARGE_IN_N``
  delivered reads "unread"; it never voids the run.

The report re-judges every pass-1 ``tool`` commit with the **unpatched** ``live.classify`` (never a ``tool.call``
record) and the repair's ``tool`` commit with ``live.repair_verdict``. Output (``--out-dir``, default
``logs/t0099-repair-live-<sha7>/``): ``turns.jsonl``, ``sessions/*.jsonl``, ``manifest.json``, ``report.json``,
``report.md``; ``data/`` holds the sessions' thumbnails. Exit codes: 0 the run is not void and G1-G8 hold in both arms;
1 otherwise, or the engine is not there; 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from local_astra.toolcall import live, tools
from local_astra.toolcall import score as _score

if TYPE_CHECKING:
    from local_astra.config import Profile

REPO = Path(__file__).resolve().parents[2]
T0093_PATH = REPO / "scripts" / "gpu" / "t0093_toolcall_live.py"
SET_DIR = REPO / "tests" / "eval" / "toolcall"
V1_PATH = SET_DIR / "cases.jsonl"
#: H-T99a: the stress set and its pinned sha256 (both read at call time, never bound as defaults)
STRESS_PATH = REPO / "tests" / "eval" / "toolcall_stress" / "cases.jsonl"
STRESS_SHA256 = "dea52a3763419d00f8d03cffd36c90dc7b3f4353203ee0d6e4cbd296228c79eb"
STRESS_N = 30
K = 3
FORCED_N = 40

FORCED_REASON = "forced: valid draft (T-0099 arm F)"
TOOLS_ON = "native:bbfc53ccdd80:dry_run"
TOOLS_SHA12 = "bbfc53ccdd80"
REPAIR_OFF_NOTE = "tool repair off"
#: the machine's repair deadline (``TurnPolicy.tool_repair_timeout_ms``, T-0089 R6)
REPAIR_DEADLINE_MS = 3000
#: criterion 7 (H-T99b): rescue is read only with at least this many repairs fired on gt-bearing arm-S turns
READ_MIN = 10
WILSON_Z = 1.96
#: criterion 1: K-out carries no ground truth
NO_GT_CATEGORY = "K-out"
U_UNDER = "U-under"

#: Criterion 4: the arm-F cases that spoke a prefix at 4b0bd92 (tools on, first ``tool.call`` record's
#: ``spoken_prefix`` non-empty; ``logs/t0093-live-4b0bd92``), in run order. Pre-registered here before the run.
#: criterion 4's row is read with at least this many delivered
BARGE_IN_N = 3
#: ruling Q8: at most this many attempts, in run order among the candidates (so at most 5 of 40 arm-F turns barged)
BARGE_IN_MAX_ATTEMPTS = 5
BARGE_IN_CANDIDATES: tuple[str, ...] = (
    "search-04",
    "click-01",
    "click-02",
    "click-03",
    "click-05",
    "click-06",
    "click-08",
    "click-09",
    "click-10",
    "scroll-01",
    "scroll-03",
    "scroll-04",
    "scroll-05",
    "key-01",
    "key-02",
    "key-03",
    "key-04",
    "key-05",
)

GATE_KEYS = tuple(f"G{i}" for i in range(1, 9))
ARMS = ("S", "F")
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
#: J19 as T-0093: stop after this many consecutive cases in which the engine cannot be reached
UNREACHABLE_STOP = 3
CASE_TIMEOUT_S = 180.0


def _say(msg: str) -> None:
    print(f"t0099_repair_live: {msg}", file=sys.stderr, flush=True)


# =====================================================================================================
# T-0093's helpers, imported read-only by path (implementation notes: reuse, no edit)
# =====================================================================================================


def _load_t0093() -> ModuleType:
    name = "_t0093_toolcall_live_for_t0099"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, T0093_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {T0093_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the file executes
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_t93 = _load_t0093()
_MeteredLlm = _t93._MeteredLlm
_PlaybackOnly = _t93._PlaybackOnly
_CaseMemory = _t93._CaseMemory
_fake = _t93._fake
_wait_idle = _t93._wait_idle
check_loopback = _t93.check_loopback
read_metrics = _t93.read_metrics
_NoRedirect = _t93._NoRedirect
_OPENER = _t93._OPENER
_Turn = _t93._Turn
_voiced = _t93._voiced
_executes_before_drain = _t93._executes_before_drain
_stale = _t93._stale
_percentile = _t93._percentile
_dist = _t93._dist

#: the unpatched judge, captured when this module loads (criterion notation: every invalid draft is re-judged by it)
_CLASSIFY: Callable[[str], live.Verdict] = live.classify


def _unpatched_classify(text: str) -> live.Verdict:
    fn = _CLASSIFY
    if (
        getattr(fn, "__module__", None) != live.__name__
        or getattr(fn, "__qualname__", None) != "classify"
    ):
        raise RuntimeError("live.classify was already patched when the T-0099 harness loaded")
    return fn(text)


# =====================================================================================================
# The forcing (criterion 3, H-T99c)
# =====================================================================================================


@contextlib.contextmanager
def forced_classify() -> Iterator[None]:
    """Inside: ``live.classify`` returns ``Verdict("repairable", name, arguments, FORCED_REASON, query_problem)`` where
    the judge it replaced says ``valid``, and that judge's verdict otherwise. On exit -- an exception included -- the
    replaced object is ``live.classify`` again."""
    previous = live.classify

    def forced(tool_text: str) -> live.Verdict:
        v = previous(tool_text)
        if v.status == "valid":
            return live.Verdict("repairable", v.name, v.arguments, FORCED_REASON, v.query_problem)
        return v

    live.classify = forced  # type: ignore[assignment]
    try:
        yield
    finally:
        live.classify = previous  # type: ignore[assignment]


# =====================================================================================================
# The cases
# =====================================================================================================


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Case:
    case_id: str
    #: the stress category in arm S, the dev case's tool in arm F
    category: str
    text: str
    gt: dict[str, Any] | None
    screen: str | None
    screen_size: tuple[int, int] | None


def _case(r: Mapping[str, Any], category: str) -> Case:
    size = r.get("screen_size")
    return Case(
        case_id=str(r["id"]),
        category=category,
        text=str(r["user"]),
        gt=r.get("gt"),
        screen=r.get("screen"),
        screen_size=(int(size[0]), int(size[1])) if size else None,
    )


def load_stress_cases(path: Path | None = None) -> list[Case]:
    """Arm S's cases in file order. ``ValueError`` when the file is not the pinned one (``STRESS_PATH`` and
    ``STRESS_SHA256`` are read at call time)."""
    p = Path(path) if path is not None else STRESS_PATH
    got = _sha256(p)
    if got != STRESS_SHA256:
        raise ValueError(f"the stress file {p} has sha256 {got}, not the pinned {STRESS_SHA256}")
    return [_case(r, str(r["category"])) for r in _read_jsonl(p)]


def load_forced_cases() -> list[Case]:
    """Arm F: the 40 dev tool cases of ``tests/eval/toolcall/cases.jsonl`` in file order (the pinned file)."""
    got = _sha256(V1_PATH)
    if got != _t93.CASES_SHA256:
        raise ValueError(f"{V1_PATH} has sha256 {got}, not the pinned {_t93.CASES_SHA256}")
    rows = [r for r in _read_jsonl(V1_PATH) if r.get("split") == "dev" and r.get("kind") == "tool"]
    if len(rows) != FORCED_N:
        raise ValueError(f"the dev split has {len(rows)} tool cases, not {FORCED_N}")
    return [_case(r, str(r["tool"])) for r in rows]


# =====================================================================================================
# The report (pure: reads the run directory, opens no socket)
# =====================================================================================================


def _rate(k: int, n: int) -> dict[str, int]:
    return {"k": k, "n": n}


def _span(line: Mapping[str, Any] | None, name: str) -> tuple[Any, Any] | None:
    spans = (line or {}).get("spans")
    if not isinstance(spans, Mapping):
        return None
    pair = spans.get(name)
    if not isinstance(pair, list | tuple) or len(pair) != 2:
        return None
    return pair[0], pair[1]


def _span_len(line: Mapping[str, Any] | None, name: str) -> Any:
    pair = _span(line, name)
    return None if pair is None else pair[1] - pair[0]


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """The Wilson score interval of ``k`` in ``n``; ``None`` when ``n`` is 0."""
    if n <= 0:
        return None
    p = k / n
    zz = z * z
    denom = 1 + zz / n
    centre = (p + zz / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _call(name: Any, arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


def _u_under_exact(call: Mapping[str, Any], gt: Mapping[str, Any]) -> bool:
    """Criterion 1's U-under rule: ``type_at`` inside the bbox with the text equal; ``press_enter`` not scored."""
    args = call.get("arguments")
    if call.get("name") != gt.get("name") or not isinstance(args, Mapping):
        return False
    x, y, text, bbox = args.get("x"), args.get("y"), args.get("text"), gt.get("bbox")
    if type(x) is not int or type(y) is not int or not isinstance(text, str):
        return False
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return bool(x0 <= x <= x1 and y0 <= y <= y1) and text.strip() == str(gt.get("text", "")).strip()


def _exact(category: str | None, call: Mapping[str, Any], gt: Mapping[str, Any] | None) -> bool:
    if not isinstance(gt, Mapping):
        return False
    if category == U_UNDER:
        return _u_under_exact(call, gt)
    return bool(_score.exact(call, gt))


@dataclass
class _Facts:
    """One case's turn, re-derived from its session log."""

    meta: dict[str, Any]
    turn: Any  # T-0093's ``_Turn``
    arm: str
    draft_text: str | None
    draft: live.Verdict | None
    repaired: bool
    repair_ids: set[str]
    repair_requests: int
    repair_verdict: live.Verdict | None

    @property
    def tag(self) -> str:
        return f"{self.meta.get('case_id')} ({self.arm}, rep {self.meta.get('rep')})"

    @property
    def category(self) -> str | None:
        c = self.meta.get("category")
        return None if c is None else str(c)

    @property
    def gt(self) -> dict[str, Any] | None:
        gt = self.meta.get("gt")
        return gt if isinstance(gt, dict) else None

    @property
    def gt_bearing(self) -> bool:
        return self.category != NO_GT_CATEGORY and self.gt is not None

    @property
    def repaired_valid(self) -> bool:
        return self.repair_verdict is not None and self.repair_verdict.status == "valid"

    def repaired_call(self) -> dict[str, Any] | None:
        v = self.repair_verdict
        return _call(v.name, v.arguments) if v is not None and v.status == "valid" else None


def _facts(meta: dict[str, Any], turn: Any) -> _Facts:
    commits = turn.tool_commits()
    text = str(commits[-1].get("text") or "") if commits else None
    draft = _unpatched_classify(text) if text is not None else None
    requests = [d for d in turn.data("llm.request") if d.get("phase") == "tool_repair"]
    ids = {str(d.get("request_id")) for d in requests if d.get("request_id") is not None}
    ids |= {
        str(c.get("request_id"))
        for c in turn.commands("tool.repair_start")
        if c.get("request_id") is not None
    }
    found = next(
        (
            d
            for d in turn.data("llm.commit")
            if d.get("segment") == "tool" and str(d.get("request_id")) in ids
        ),
        None,
    )
    verdict = live.repair_verdict(str(found.get("text") or "")) if found is not None else None
    return _Facts(
        meta=meta,
        turn=turn,
        arm=str(meta.get("arm")),
        draft_text=text,
        draft=draft,
        repaired=bool(turn.repaired()),
        repair_ids=ids,
        repair_requests=len(requests),
        repair_verdict=verdict,
    )


def _steps_with(turn: Any) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    """``(step, event, commands)`` per ``machine.step`` of the turn, in ``seq`` order."""
    out = []
    for s in turn.steps():
        event = s.get("event") or {}
        commands = [c for c in (s.get("commands") or []) if isinstance(c, Mapping)]
        out.append((s, event if isinstance(event, dict) else {}, commands))
    return out


def _aborted(f: _Facts) -> bool:
    """The session log has an ``LlmAbort`` of the turn's repair request."""
    return any(str(c.get("request_id")) in f.repair_ids for c in f.turn.commands("llm.abort"))


def _g2(f: _Facts) -> list[dict[str, Any]]:
    """J4/J17 (T-0093's ``_voiced``) and R9: no ``TtsSpeak`` out of a step fed by the repair request's commits."""
    out = list(_voiced(f.turn))
    for _, event, commands in _steps_with(f.turn):
        if str(event.get("request_id")) not in f.repair_ids:
            continue
        for c in commands:
            if c.get("type") == "tts.speak" and c.get("text"):
                out.append(
                    {
                        "log": f.turn.log,
                        "turn_id": f.turn.turn_id,
                        "utterance_id": c.get("utterance_id"),
                        "field": "repair",
                        "found": [f"voiced from repair request {event.get('request_id')}"],
                    }
                )
    return out


def _g7(f: _Facts) -> list[dict[str, Any]]:
    """No ``ToolExecute`` for a call that ended refused/cancelled; every executed repair call schema-valid."""
    out: list[dict[str, Any]] = []
    calls = f.turn.calls()
    executes = f.turn.commands("tool.execute")
    cancels = f.turn.commands("tool.cancel")
    v1 = tools.load()
    for c in executes:
        cid = c.get("call_id")
        mine = [d for d in calls if d.get("call_id") == cid]
        last = mine[-1] if mine else None
        dead_reasons = {"refused", "repair_failed", "repair_timeout", "cancelled"}
        cancelled = [x.get("reason") for x in cancels if x.get("call_id") == cid]
        if (
            last is not None
            and (last.get("status") in dead_reasons or last.get("reason") in dead_reasons)
        ) or cancelled:
            out.append(
                {
                    "log": f.turn.log,
                    "call_id": cid,
                    "why": f"executed a call whose final status is "
                    f"{last.get('status') if last else None} (cancels {cancelled})",
                }
            )
        if c.get("source") == "repair" and not _score.schema_valid(
            _call(c.get("name"), c.get("arguments")), v1
        ):
            out.append(
                {
                    "log": f.turn.log,
                    "call_id": cid,
                    "why": f"executed repair call {c.get('name')} {c.get('arguments')} is not schema-valid",
                }
            )
    return out


def _g8(f: _Facts) -> list[dict[str, Any]]:
    """Every repair ends with a verdict before its deadline, as ``repair_timeout`` with its abort logged, or aborted by
    a barge-in; none open at close."""
    if not f.repaired:
        return []
    start: int | None = None
    verdict_at: int | None = None
    for _, event, commands in _steps_with(f.turn):
        if start is None and any(c.get("type") == "tool.repair_start" for c in commands):
            t = event.get("t_ms")
            start = int(t) if isinstance(t, int | float) else None
        if (
            verdict_at is None
            and event.get("type") == "llm.done"
            and str(event.get("request_id")) in f.repair_ids
            and event.get("tool") is not None
        ):
            t = event.get("t_ms")
            verdict_at = int(t) if isinstance(t, int | float) else None
    aborted = _aborted(f)
    timed_out = any(c.get("reason") == "repair_timeout" for c in f.turn.commands("tool.cancel"))
    if verdict_at is not None and start is not None and verdict_at - start < REPAIR_DEADLINE_MS:
        return []
    if timed_out and aborted:
        return []
    if f.turn.interrupted() and aborted:
        return []
    why = (
        f"verdict {verdict_at - start} ms after the repair started (deadline {REPAIR_DEADLINE_MS})"
        if verdict_at is not None and start is not None
        else "repair_timeout without the repair's LlmAbort"
        if timed_out
        else "no verdict, no timeout, no abort: the repair was open at close"
    )
    return [{"log": f.turn.log, "case": f.tag, "why": why}]


def _first_prompt_sha(records: Sequence[Mapping[str, Any]]) -> str | None:
    """Q3: sha256 of the system message's text parts (in order) of the first ``llm.request`` record."""
    for r in records:
        if r.get("kind") != "llm.request":
            continue
        messages = ((r.get("data") or {}).get("prompt") or {}).get("messages") or []
        for m in messages:
            if isinstance(m, Mapping) and m.get("role") == "system":
                parts = m.get("content") or []
                text = "".join(
                    str(p.get("text") or "")
                    for p in parts
                    if isinstance(p, Mapping) and p.get("type") == "text"
                )
                return hashlib.sha256(text.encode("utf-8")).hexdigest()
        return None
    return None


def _arm_report(arm: str, facts: list[_Facts]) -> dict[str, Any]:
    """Criterion 6's numbers for one arm, from its own turns only (criterion 7: arm F never enters an arm-S figure)."""
    mine = [f for f in facts if f.arm == arm]
    routed = [f for f in mine if f.draft is not None]
    invalid = [f for f in routed if f.draft is not None and f.draft.status == "repairable"]
    refused = [f for f in routed if f.draft is not None and f.draft.status == "refused"]
    repaired = [f for f in mine if f.repaired]
    outcomes: dict[str, int] = {}
    for f in repaired:
        status = f.turn.final_status() or "none"
        outcomes[status] = outcomes.get(status, 0) + 1
    repair_ms = [v for f in repaired if (v := _span_len(f.turn.line, "tool.repair")) is not None]
    action: list[Any] = []
    for f in repaired:
        execute, done = _span(f.turn.line, "tool.execute"), _span(f.turn.line, "llm.done")
        if execute is not None and done is not None:
            action.append(execute[0] - done[1])
    timeouts = sum(
        1
        for f in repaired
        if any(c.get("reason") == "repair_timeout" for c in f.turn.commands("tool.cancel"))
    )
    out: dict[str, Any] = {
        "cases": len(mine),
        "routed": len(routed),
        "repairs": {
            "n": len(repaired),
            "outcomes": dict(sorted(outcomes.items())),
            "ms": _dist(repair_ms),
        },
        "repair_timeout": timeouts,
        "action_latency_ms": _dist(action),
        "max_inflight": max(
            (
                int(f.meta["max_inflight"])
                for f in mine
                if isinstance(f.meta.get("max_inflight"), int)
            ),
            default=None,
        ),
    }
    if arm == "S":
        categories: list[str] = []
        for f in mine:
            if f.category is not None and f.category not in categories:
                categories.append(f.category)
        by_reason: dict[str, int] = {}
        for f in invalid:
            reason = str(f.draft.reason) if f.draft is not None else "None"
            by_reason[reason] = by_reason.get(reason, 0) + 1
        # criterion 6e-g: rescue over every repair fired, K-out excluded from k (coercion, never rescue)
        rescued = [f for f in repaired if f.repaired_valid and f.category != NO_GT_CATEGORY]
        gt_repairs = [f for f in repaired if f.gt_bearing]
        exact = [
            f
            for f in gt_repairs
            if (c := f.repaired_call()) is not None and _exact(f.category, c, f.gt)
        ]
        coerced = [f for f in repaired if f.category == NO_GT_CATEGORY and f.repaired_valid]
        read = len(gt_repairs) >= READ_MIN
        # ruling Q5: refused (nameless) drafts are counted only as ``refused_drafts``, never in the yield denominator
        judged = [f for f in routed if f.draft is not None and f.draft.status != "refused"]
        yield_hi = wilson(len(invalid), len(judged))
        unread = (
            f"insufficient yield (n = {len(gt_repairs)}); yield {len(invalid)}/{len(judged)}, Wilson 95 % upper "
            f"bound {'-' if yield_hi is None else f'{yield_hi[1]:.3f}'}; no conclusion on rescue"
        )
        rescue_ci = wilson(len(rescued), len(repaired)) if read else None
        exact_ci = wilson(len(exact), len(gt_repairs)) if read else None
        out.update(
            {
                "yield": _rate(len(invalid), len(judged)),
                "yield_by_category": {
                    cat: _rate(
                        sum(1 for f in invalid if f.category == cat),
                        sum(1 for f in judged if f.category == cat),
                    )
                    for cat in categories
                },
                "yield_by_reason": dict(sorted(by_reason.items())),
                "refused_drafts": len(refused),
                "unrouted": len(mine) - len(routed),
                "no_call": sum(1 for f in mine if not f.turn.calls()),
                "rescue": {
                    "k": len(rescued),
                    "n": len(repaired),
                    "wilson95": list(rescue_ci) if rescue_ci is not None else None,
                    "reading": "read" if read else unread,
                },
                "rescue_exact": {
                    "k": len(exact),
                    "n": len(gt_repairs),
                    "wilson95": list(exact_ci) if exact_ci is not None else None,
                    "reading": "read" if read else unread,
                },
                "coercion": {
                    "n": len(coerced),
                    "cases": [
                        {
                            "case_id": f.meta.get("case_id"),
                            "rep": f.meta.get("rep"),
                            "combo": (f.repair_verdict.arguments or {}).get("combo")
                            if f.repair_verdict is not None
                            else None,
                        }
                        for f in coerced
                    ],
                },
            }
        )
    else:
        agree = [
            f
            for f in repaired
            if f.draft is not None
            and f.repaired_valid
            and f.repair_verdict is not None
            and (f.repair_verdict.name, f.repair_verdict.arguments)
            == (f.draft.name, f.draft.arguments)
        ]
        exact_f = [
            f
            for f in repaired
            if (c := f.repaired_call()) is not None and _exact(f.category, c, f.gt)
        ]
        delivered = [
            f
            for f in mine
            if (f.turn.line or {}).get("outcome") == "interrupted"
            and (f.turn.line or {}).get("tool") is not None
            and _aborted(f)
        ]
        attempted = [f for f in mine if f.meta.get("barge_in_attempted")]
        # ruling Q8: the repair's verdict accepted while the turn was PAUSED (the onset's pause, before the takeover)
        done_paused = [
            f
            for f in attempted
            if any(
                step.get("state_before") == "PAUSED"
                and event.get("type") == "llm.done"
                and str(event.get("request_id")) in f.repair_ids
                and event.get("tool") is not None
                for step, event, _ in _steps_with(f.turn)
            )
        ]
        out.update(
            {
                "repairs_fired": _rate(len(repaired), len(routed)),
                "agreement": _rate(len(agree), len(repaired)),
                "exact": _rate(len(exact_f), len(repaired)),
                "barge_in": {
                    "delivered": len(delivered),
                    "cases": [f.meta.get("case_id") for f in delivered],
                    "attempted": len(attempted),
                    "repair_done_before_takeover": len(done_paused),
                    "reading": "read"
                    if len(delivered) >= BARGE_IN_N
                    else f"unread: {len(delivered)} of {BARGE_IN_N} delivered",
                },
            }
        )
    return out


def build_report(run_dir: Path) -> dict[str, Any]:
    """The report of a run directory: reads only ``turns.jsonl``, ``sessions/*.jsonl`` and ``manifest.json``."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sessions: list[dict[str, Any]] = list(manifest.get("sessions") or [])
    cases: list[dict[str, Any]] = list(manifest.get("cases") or [])
    regime_in: dict[str, Any] = dict(manifest.get("regime") or {})
    problems: dict[str, dict[str, list[Any]]] = {arm: {g: [] for g in GATE_KEYS} for arm in ARMS}
    global_void: list[str] = []

    lines = _read_jsonl(run_dir / "turns.jsonl") if (run_dir / "turns.jsonl").is_file() else []
    by_session: dict[str, list[dict[str, Any]]] = {}
    for line in lines:
        by_session.setdefault(str(line.get("session_id")), []).append(line)

    arm_of_session: dict[str, str] = {}
    forced_of_session: dict[str, bool] = {}
    log_session: dict[str, str] = {}
    for c in cases:
        arm_of_session.setdefault(str(c.get("session_id")), str(c.get("arm")))
        log_session.setdefault(str(c.get("session_log")), str(c.get("session_id")))
    for s in sessions:
        sid = str(s.get("session_id"))
        arm_of_session.setdefault(sid, str(s.get("arm")))
        log_session.setdefault(str(s.get("log")), sid)
        forced_of_session[sid] = s.get("forced") is True

    def flag(arm: str | None, gate: str, item: Any) -> None:
        if arm in problems:
            problems[arm][gate].append(item)
        else:
            global_void.append(str(item))

    if manifest.get("stopped"):
        global_void.append(f"integrity: the run stopped early: {manifest.get('stopped')}")
    sizes = regime_in.get("arm_sizes")
    if isinstance(sizes, Mapping):
        for arm in ARMS:
            got = sum(1 for c in cases if c.get("arm") == arm)
            if sizes.get(arm) != got:
                global_void.append(
                    f"integrity: arm {arm} has {got} cases in the manifest, not {sizes.get(arm)}"
                )
    # T-0101 criterion 6: a warm-up that ran and left any payload tool not ``ok`` voids the run
    repair_warmup = regime_in.get("repair_warmup")
    if not isinstance(repair_warmup, Mapping):
        repair_warmup = {"on": False}
    if repair_warmup.get("on") is True:
        per_tool = repair_warmup.get("per_tool")
        per_tool = per_tool if isinstance(per_tool, Mapping) else {}
        for tool in sorted({*live.load_payloads().tools, *per_tool}):
            entry = per_tool.get(tool)
            status = entry.get("status") if isinstance(entry, Mapping) else "missing"
            if status != "ok":
                global_void.append(f"integrity: repair warm-up incomplete ({tool}: {status})")

    # --- the session logs ----------------------------------------------------------------------------
    logs: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(log_session):
        arm = arm_of_session.get(log_session[name])
        path = run_dir / "sessions" / name
        if not path.is_file():
            flag(arm, "G1", f"integrity: session log {name} is missing")
            continue
        records = sorted(_read_jsonl(path), key=lambda r: int(r.get("seq", 0)))
        logs[name] = records
        dropped = [r for r in records if r.get("kind") == "dropped"]
        if dropped:
            flag(
                arm, "G1", f"integrity: session log {name} dropped records ({len(dropped)} gap(s))"
            )
        for r in records:
            d = r.get("data") or {}
            if r.get("kind") == "note" and d.get("what") == REPAIR_OFF_NOTE:
                flag(arm, "G1", f"repair off in {name}: {d.get('reason')}")
                break
        sid = log_session[name]
        if not forced_of_session.get(sid, False):
            marked = [
                r
                for r in records
                if r.get("kind") in ("tool.call", "machine.step")
                and FORCED_REASON in json.dumps(r.get("data"), ensure_ascii=False)
            ]
            if marked:
                flag(
                    arm,
                    "G1",
                    f"integrity: the forced marker in {name}, a session that was not forced (arm {arm}; "
                    f"{len(marked)} record(s), first {marked[0].get('kind')})",
                )
        _dropped, hits = _stale(records)
        for h in hits:
            flag(arm, "G5", {"log": name, **h})

    # --- the cases --------------------------------------------------------------------------------------
    facts: list[_Facts] = []
    for c in cases:
        arm = str(c.get("arm"))
        sid = str(c.get("session_id"))
        tag = f"{c.get('case_id')} ({arm}, rep {c.get('rep')})"
        if arm not in ARMS:
            global_void.append(f"integrity: {tag} has arm {arm!r}")
            continue
        if (c.get("forced") is True) != (arm == "F") or forced_of_session.get(sid, arm == "F") != (
            arm == "F"
        ):
            flag(arm, "G1", f"integrity: {tag} is forced={c.get('forced')} in arm {arm}")
        found = by_session.get(sid) or []
        if len(found) != 1:
            flag(arm, "G1", f"integrity: {tag} has {len(found)} turns.jsonl lines, not 1")
        name = str(c.get("session_log"))
        tid = c.get("turn_id")
        turn = _Turn(log=name, turn_id=None if tid is None else str(tid))
        turn.records = [
            r for r in logs.get(name, []) if tid is not None and str(r.get("turn_id")) == str(tid)
        ]
        turn.line = found[0] if found else None
        line = turn.line or {}
        if (line.get("regime") or {}).get("tools") != TOOLS_ON:
            flag(
                arm,
                "G1",
                f"integrity: {tag}'s line reads tools={(line.get('regime') or {}).get('tools')!r}, not {TOOLS_ON!r}",
            )
        if line.get("outcome") == "error":
            flag(arm, "G1", f"integrity: {tag} ended in an error outcome")
        if c.get("timed_out"):
            flag(arm, "G1", f"integrity: {tag} timed out")
        if c.get("failures"):
            flag(arm, "G1", f"integrity: {tag} recorded session failures: {c.get('failures')}")
        f = _facts(c, turn)
        facts.append(f)
        # criterion 8a: a repair on a draft the unpatched judge calls valid, outside arm F
        if arm == "S" and f.repaired and f.draft is not None and f.draft.status == "valid":
            flag(arm, "G1", f"integrity: repair on a valid draft outside arm F: {tag}")
        problems[arm]["G2"] += _g2(f)
        if turn.interrupted():
            for cmd in turn.commands("tool.execute"):
                problems[arm]["G3"].append(
                    {"log": name, "case": tag, "call_id": cmd.get("call_id")}
                )
        problems[arm]["G4"] += _executes_before_drain(turn)
        if f.repair_requests > 1:
            problems[arm]["G6"].append(
                {"log": name, "case": tag, "tool_repair_requests": f.repair_requests}
            )
        problems[arm]["G7"] += _g7(f)
        problems[arm]["G8"] += _g8(f)

    # a session with more or fewer than one line (criterion 5 G1: exactly one turn line per session)
    for sid, arm in arm_of_session.items():
        n_lines = len(by_session.get(sid) or [])
        if n_lines != 1 and not any(str(c.get("session_id")) == sid for c in cases):
            flag(arm, "G1", f"integrity: session {sid} has {n_lines} turns.jsonl lines, not 1")

    void: list[str] = []
    for item in [*global_void, *problems["S"]["G1"], *problems["F"]["G1"]]:
        if str(item) not in void:
            void.append(str(item))

    first_log = None
    if sessions:
        first_log = str(sessions[0].get("log"))
    elif cases:
        first_log = str(cases[0].get("session_log"))
    regime = {
        **regime_in,
        "prompt_sha256": _first_prompt_sha(logs.get(first_log or "", [])),
        "stress_sha256": regime_in.get("stress_sha256"),
        "tools_sha256": regime_in.get("tools_sha256"),
        "k": regime_in.get("k"),
        "arm_sizes": regime_in.get("arm_sizes"),
        "commit": regime_in.get("commit"),
        "repair_warmup": repair_warmup,
    }
    first_use = _first_use_repair(sessions, cases, logs, by_session)
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "git_commit": sorted({str((line.get("regime") or {}).get("git_commit")) for line in lines}),
        "regime": regime,
        "void": void,
        "global_void": [str(v) for v in global_void],
    }
    for arm in ARMS:
        report[arm] = {**_arm_report(arm, facts), "gate_problems": problems[arm]}
    report["first_use_repair"] = first_use
    report["first_use_repair_max_ms"] = max(
        (v["ms"] for v in first_use.values() if v.get("ms") is not None), default=None
    )
    return report


def _first_use_repair(
    sessions: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    logs: Mapping[str, list[dict[str, Any]]],
    by_session: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """T-0101 criterion 7: per payload sha, the first ``tool_repair`` request in manifest session order (either arm):
    ``{tool, arm, case_id, rep, ms}``, ``ms`` the turn line's ``tool.repair`` span. The sha is read from the request's
    logged ``cd``; a request logged without one (ruling W4) is skipped."""
    tool_of = {p.automaton_sha256: name for name, p in live.load_payloads().tools.items()}
    case_of = {str(c.get("session_id")): c for c in cases}
    order = [(str(s.get("session_id")), str(s.get("log")), s.get("arm")) for s in sessions]
    if not order:
        order = [(str(c.get("session_id")), str(c.get("session_log")), c.get("arm")) for c in cases]
    out: dict[str, dict[str, Any]] = {}
    for sid, log, arm in order:
        for r in logs.get(log, []):
            d = r.get("data") or {}
            if r.get("kind") != "llm.request" or d.get("phase") != "tool_repair":
                continue
            cd = d.get("cd")
            sha = cd.get("automaton_sha256") if isinstance(cd, Mapping) else None
            if sha is None or str(sha) in out:
                continue
            c = case_of.get(sid) or {}
            mine = by_session.get(sid) or []
            out[str(sha)] = {
                "tool": tool_of.get(str(sha)),
                "arm": c.get("arm", arm),
                "case_id": c.get("case_id"),
                "rep": c.get("rep"),
                "ms": _span_len(mine[0] if mine else None, "tool.repair"),
            }
    return out


def gates(report: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    """Criterion 5 per arm: ``True`` holds. G1 also fails on a void reason that belongs to no arm."""
    glob = report.get("global_void") or []
    out: dict[str, dict[str, bool]] = {}
    for arm in ARMS:
        found = (report.get(arm) or {}).get("gate_problems") or {}
        out[arm] = {g: not found.get(g) for g in GATE_KEYS}
        out[arm]["G1"] = out[arm]["G1"] and not glob
    return out


def _kn(rate: Mapping[str, Any] | None) -> str:
    if not rate:
        return "-"
    return f"{rate.get('k')}/{rate.get('n')}"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _dist_cells(d: Mapping[str, Any] | None) -> str:
    d = d or {}
    return f"n {d.get('n', 0)}, p50 {_fmt(d.get('p50'))}, p90 {_fmt(d.get('p90'))}"


def _ci(value: Any) -> str:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return "-"
    return f"[{value[0]:.3f}, {value[1]:.3f}]"


GATE_TEXT = {
    "G1": "run valid (tools line, repair on, no error/timeout/failure, one line per session, marker only when forced)",
    "G2": "only the spoken prefix voiced; nothing voiced from a repair request (R9)",
    "G3": "no ToolExecute on interrupted turns",
    "G4": "no ToolExecute before the turn's own drain",
    "G5": "no stale terminal reached the machine",
    "G6": "at most one tool_repair request per turn",
    "G7": "no ToolExecute for a refused/failed/timed-out/cancelled call; executed repair calls schema-valid",
    "G8": f"every repair ends with a verdict before {REPAIR_DEADLINE_MS} ms, or repair_timeout with its abort",
}


def render_markdown(report: Mapping[str, Any]) -> str:
    """report.md: the regime, the void, G1-G8 per arm, then criterion 6's numbers per arm."""
    g = gates(report)
    commits = ", ".join(f"`{c}`" for c in report.get("git_commit") or []) or "-"
    out = [f"# T-0099 live tool-call repair, run at {commits}", ""]
    out.append(
        "Dry run; nothing executed (`tools = dry_run`). Arm F's numbers never enter an arm-S figure."
    )
    out.append("")
    regime = report.get("regime")
    if isinstance(regime, Mapping):
        out += ["## Regime", ""]
        out += [f"- {k}: `{v}`" for k, v in regime.items()]
        out.append("")
    void = report.get("void") or []
    out += ["## Void", ""]
    out += [f"- {v}" for v in void] if void else ["- none: the run is valid"]
    out += ["", "## Hard gates (criterion 5, H-T99b; no number is read unless all hold)", ""]
    out += ["| gate | arm S | arm F | what |", "|---|---|---|---|"]
    for key in GATE_KEYS:
        cells = []
        for arm in ARMS:
            n = len((((report.get(arm) or {}).get("gate_problems") or {}).get(key)) or [])
            cells.append("yes" if g[arm][key] else f"**NO** ({n})")
        out.append(f"| {key} | {cells[0]} | {cells[1]} | {GATE_TEXT[key]} |")
    s = report.get("S") or {}
    f = report.get("F") or {}
    rs = s.get("repairs") or {}
    out += [
        "",
        "## Arm S: elicited drafts on out-of-schema requests (not a production invalid-rate estimate)",
        "",
        f"- yield (invalid drafts / routed tool turns minus refused drafts (Q5), re-judged with the unpatched "
        f"`live.classify`): "
        f"{_kn(s.get('yield'))}",
        f"- by category: {', '.join(f'{c} {_kn(r)}' for c, r in (s.get('yield_by_category') or {}).items()) or '-'}",
        f"- by parse reason: {s.get('yield_by_reason') or {}}",
        f"- refused drafts (nameless and the like): {s.get('refused_drafts', 0)}; unrouted turns: "
        f"{s.get('unrouted', 0)}; turns with no call: {s.get('no_call', 0)}",
        f"- repairs fired: {rs.get('n', 0)}; outcomes by final status (J6): {rs.get('outcomes') or {}}; "
        f"repair_timeout: {s.get('repair_timeout', 0)}",
        f"- `tool.repair` ms (nearest rank): {_dist_cells(rs.get('ms'))}",
    ]
    rescue = s.get("rescue") or {}
    rexact = s.get("rescue_exact") or {}
    if rescue.get("reading") == "read":
        out.append(
            f"- rescue (repaired to valid / repairs fired, K-out excluded from k): {_kn(rescue)}, "
            f"Wilson 95 % {_ci(rescue.get('wilson95'))}"
        )
        out.append(
            f"- rescue-exact (gt-bearing categories; U-under: bbox and text, press_enter not scored): "
            f"{_kn(rexact)}, Wilson 95 % {_ci(rexact.get('wilson95'))}"
        )
    else:
        out.append(f"- rescue: {rescue.get('reading')}")
    coercion = s.get("coercion") or {}
    out += [
        f"- coercion (K-out turns repaired to a valid call; never a rescue): {coercion.get('n', 0)} "
        f"{[(c.get('case_id'), c.get('rep'), c.get('combo')) for c in coercion.get('cases') or []]}",
        f"- action latency of repaired calls (`tool.execute` start - pass 1 `llm.done` end) ms: "
        f"{_dist_cells(s.get('action_latency_ms'))}",
    ]
    rf = f.get("repairs") or {}
    barge = f.get("barge_in") or {}
    out += [
        "",
        "## Arm F: the repair forced on valid drafts (the seeded path's floor on the live path)",
        "",
        f"- repairs fired / routed: {_kn(f.get('repairs_fired'))}; outcomes by final status: "
        f"{rf.get('outcomes') or {}}",
        f"- agreement (repaired call equal to the draft): {_kn(f.get('agreement'))}; exact against gt: "
        f"{_kn(f.get('exact'))}",
        f"- `tool.repair` ms, floor on valid drafts, not repair latency on invalid drafts: {_dist_cells(rf.get('ms'))}",
        f"- repair_timeout: {f.get('repair_timeout', 0)}",
        f"- barge-in during a repair (criterion 4): {barge.get('delivered', 0)} delivered of "
        f"{barge.get('attempted', 0)} attempted (cap {BARGE_IN_MAX_ATTEMPTS}) {barge.get('cases') or []}; repair done "
        f"before the takeover (verdict accepted in PAUSED): {barge.get('repair_done_before_takeover', 0)}; "
        f"{barge.get('reading')}",
        "",
    ]
    first_use = report.get("first_use_repair") or {}
    rows = sorted(
        (v for v in first_use.values() if isinstance(v, Mapping)), key=lambda v: str(v.get("tool"))
    )
    out += [
        "## Repair first use per automaton (T-0101 criterion 7; either arm, manifest session order)",
        "",
        "| tool | arm | case | rep | `tool.repair` ms |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {v.get('tool')} | {v.get('arm')} | {v.get('case_id')} | {v.get('rep')} | {'-' if v.get('ms') is None else v.get('ms')} |"
        for v in rows
    ]
    out += ["", f"- slowest first use: {_fmt(report.get('first_use_repair_max_ms'))} ms", ""]
    return "\n".join(out)


# =====================================================================================================
# The engine side (sockets: loopback only, unproxied)
# =====================================================================================================


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


@dataclass
class _Run:
    base_url: str
    out: Path
    profile: Profile
    manifest: dict[str, Any]
    #: ruling Q8: barge-ins submitted so far (capped at ``BARGE_IN_MAX_ATTEMPTS``)
    barge_attempts: int = 0
    order: int = 0


async def _barge_on_repair(session: Any, stop: asyncio.Event) -> bool:
    """Criterion 4: one ``UserSpeechOnset`` once the turn is SPEAKING and the live generate is its repair request."""
    from local_astra.contracts import events as E
    from local_astra.contracts.turn import TurnState

    while not stop.is_set():
        state = session.state
        if state.state is TurnState.SPEAKING and state.active_generate == f"{state.turn_id}-repair":
            now = session.clock.now_ms
            session.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
            return True
        await asyncio.sleep(0.005)
    return False


async def _session(run: _Run, case: Case, arm: str, rep: int) -> bool:
    """One (case, repetition, arm) in a fresh session, recorded; ``False`` when the engine could not be reached."""
    import httpx

    from local_astra import cli
    from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
    from local_astra.conductor.session import Session, SessionClock, SessionConfig
    from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
    from local_astra.contracts import events as E
    from local_astra.contracts.perception import Frame, Source
    from local_astra.telemetry.session_log import SessionLog

    forced = arm == "F"
    index = run.order
    run.order += 1
    idle, reachable, _ = await _wait_idle(run.base_url)
    sessions_dir = run.out / "sessions"
    attempt = (
        forced
        and case.case_id in BARGE_IN_CANDIDATES
        and run.barge_attempts < BARGE_IN_MAX_ATTEMPTS
    )
    submitted = timed_out = False
    with forced_classify() if forced else contextlib.nullcontext():
        started = time.perf_counter()
        kit, detokenizer = cli.load_tool_kit(run.profile)
        inner = VllmClient(
            VllmEndpoint(base_url=run.base_url),
            transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
            detokenizer=detokenizer,
        )
        llm = _MeteredLlm(inner, run.base_url)
        clock = SessionClock()
        frames = []
        if case.screen is not None and case.screen_size is not None:
            jpeg = (SET_DIR / case.screen).read_bytes()
            frames.append(
                Frame(
                    source=Source.SCREEN,
                    ts_ms=0,
                    width=case.screen_size[0],
                    height=case.screen_size[1],
                    jpeg=jpeg,
                    phash=hashlib.sha256(jpeg).hexdigest()[:16],
                )
            )
        log = SessionLog(sessions_dir)
        session = Session(
            TurnMachine(TurnPolicy(thinking=run.profile.diffusion.thinking)),
            llm,
            _fake("stt", "FakeStt")(),
            _fake("tts", "FakeTts")(),
            _PlaybackOnly(_fake("audio", "FakeAudioIO")(clock)),
            _fake("perception", "FakePerception")(frames),
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
        try:
            await session.start()
            startup_ms = round((time.perf_counter() - started) * 1000.0, 1)
            session.turn_written.clear()
            stop = asyncio.Event()
            barge = asyncio.ensure_future(_barge_on_repair(session, stop)) if attempt else None
            session.submit(E.UserText(t_ms=clock.now_ms, text=case.text))
            try:
                await asyncio.wait_for(session.turn_written.wait(), timeout=CASE_TIMEOUT_S)
            except TimeoutError:
                timed_out = True
            stop.set()
            submitted = bool(await barge) if barge is not None else False
        finally:
            await session.aclose()
            await llm.aclose()

    line: dict[str, Any] | None = None
    turns_path = run.out / "turns.jsonl"
    if turns_path.is_file():
        mine = [r for r in _read_jsonl(turns_path) if r.get("session_id") == session.session_id]
        line = mine[0] if mine else None
    delivered = False
    if submitted and line and line.get("outcome") == "interrupted" and line.get("tool") is not None:
        records = _read_jsonl(log.path) if log.path.is_file() else []
        repair_ids = {
            str((r.get("data") or {}).get("request_id"))
            for r in records
            if r.get("kind") == "llm.request"
            and (r.get("data") or {}).get("phase") == "tool_repair"
        }
        delivered = any(
            c.get("type") == "llm.abort" and str(c.get("request_id")) in repair_ids
            for r in records
            if r.get("kind") == "machine.step"
            for c in ((r.get("data") or {}).get("commands") or [])
            if isinstance(c, Mapping)
        )
    if submitted:
        run.barge_attempts += 1
    run.manifest["sessions"].append(
        {
            "arm": arm,
            "session_id": session.session_id,
            "log": log.path.name,
            "forced": forced,
            "startup_ms": startup_ms,
        }
    )
    run.manifest["cases"].append(
        {
            "case_id": case.case_id,
            "category": case.category,
            "arm": arm,
            "rep": rep,
            "order": index,
            "session_id": session.session_id,
            "session_log": log.path.name,
            "turn_id": line.get("turn_id") if line else None,
            "gt": case.gt,
            "forced": forced,
            "barge_in": delivered,
            "barge_in_attempted": submitted,
            "deferred_prefill_aborts": len(llm.deferred_aborts),
            "max_inflight": llm.max_inflight,
            "engine_idle_before": idle,
            "engine_reachable": reachable,
            "startup_ms": startup_ms,
            "timed_out": timed_out,
            "failures": [repr(f) for f in session.failures],
        }
    )
    _write_json(run.out / "manifest.json", run.manifest)
    tool = (line or {}).get("tool") or {}
    _say(
        f"{index:03d} {case.case_id} [{arm} rep {rep}] outcome={line.get('outcome') if line else None} "
        f"tool={tool.get('status')}{'/' + str(tool.get('reason')) if tool.get('reason') else ''}"
        + (" BARGE-IN" if delivered else "")
    )
    return reachable


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


async def _run_all(run: _Run, stress: Sequence[Case], forced: Sequence[Case]) -> None:
    """Arm S (repetition-major), then arm F; J19: stop after ``UNREACHABLE_STOP`` consecutive unreachable cases."""
    plan = [(c, "S", rep) for rep in range(K) for c in stress] + [(c, "F", 0) for c in forced]
    unreachable = 0
    for case, arm, rep in plan:
        unreachable = 0 if await _session(run, case, arm, rep) else unreachable + 1
        if unreachable >= UNREACHABLE_STOP:
            run.manifest["stopped"] = (
                f"the engine could not be reached for {unreachable} consecutive cases "
                f"(last: {case.case_id} [{arm} rep {rep}])"
            )
            _write_json(run.out / "manifest.json", run.manifest)
            _say(f"stopping: {run.manifest['stopped']}")
            return


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--warm-repair",
        action="store_true",
        help="T-0101: one warm-up repair per payload automaton before case 0 (default off)",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    try:
        dirty = _git("status", "--porcelain").rstrip("\n")
        sha = _git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _say(f"refusing: {REPO} is not a readable git tree ({exc})")
        return EXIT_REFUSED
    if dirty:
        _say(f"refusing: the tree at {REPO} is dirty:\n{dirty}")
        return EXIT_REFUSED
    out: Path = args.out_dir or REPO / "logs" / f"t0099-repair-live-{sha[:7]}"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _say(f"refusing: {out} exists and is not empty; move it aside first")
        return EXIT_REFUSED
    try:
        _t93.load_cases()  # T-0093's refusals: the pinned v1 and Q12 files, a 40 + 8 dev split
        stress = load_stress_cases()
        forced = load_forced_cases()
    except (OSError, ValueError, KeyError) as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED

    from local_astra import cli
    from local_astra.config import load

    profile = load().profiles["lean"]
    try:
        kit, detokenizer = cli.load_tool_kit(profile)
    except cli.ToolsUnavailable as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    if kit.repair_off is not None:
        _say(
            f"refusing: the tool repair is off ({kit.repair_off}); a run with repair off is invalid"
        )
        return EXIT_REFUSED
    if kit.tools_version[:12] != TOOLS_SHA12:
        _say(f"refusing: the tool set is {kit.tools_version[:12]}, not {TOOLS_SHA12}")
        return EXIT_REFUSED

    # --- sockets from here on: loopback, unproxied ---------------------------------------------------
    from local_astra.conductor.llm_client import VllmEndpoint

    model = VllmEndpoint().model
    listing = _t93._get_text(args.base_url.rstrip("/") + "/models")
    served: list[str] = []
    with contextlib.suppress(ValueError, KeyError, TypeError):
        served = [str(m["id"]) for m in json.loads(listing or "")["data"]]
    if model not in served:
        _say(
            f"no engine serving {model} at {args.base_url} (served: {served}); the locked script boots it"
        )
        return EXIT_FAILED
    if read_metrics(args.base_url) is None:
        _say(f"{_t93._root(args.base_url)}/metrics is not readable; the idle check cannot run")
        return EXIT_FAILED

    # --- T-0101 criterion 6: the repair warm-up, once, after the idle and loopback checks, before case 0 ---------
    repair_warmup: dict[str, Any] = {"on": False}
    if args.warm_repair:
        from datetime import datetime

        async def warm() -> dict[str, Any]:
            import httpx

            from local_astra.conductor.llm_client import VllmClient
            from local_astra.toolcall import warmup

            await _wait_idle(args.base_url)
            client = VllmClient(
                VllmEndpoint(base_url=args.base_url),
                transport=httpx.AsyncHTTPTransport(),  # an explicit transport: no proxy from the environment
                detokenizer=detokenizer,
            )
            try:
                done = await warmup.warm_repair(client, kit)
            finally:
                await client.aclose()
            return {
                "on": True,
                "per_tool": {
                    t: {
                        "status": e.status,
                        "wall_ms": e.wall_ms,
                        "commits": e.commits,
                        "verdict": e.verdict,
                    }
                    for t, e in done.per_tool.items()
                },
                "total_ms": done.total_ms,
                "ended_wall": datetime.now().astimezone().isoformat(),
            }

        repair_warmup = asyncio.run(warm())
        _say(
            "repair warm-up: "
            + ", ".join(
                f"{t} {v['wall_ms']:.0f} ms {v['status']}"
                for t, v in repair_warmup["per_tool"].items()
            )
            + f", total {repair_warmup['total_ms']:.0f} ms"
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions").mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "regime": {
            "commit": sha,
            "base_url": args.base_url,
            "served_model": model,
            "engine_version": _t93._get_text(_t93._root(args.base_url) + "/version"),
            "versions_lock_sha256": _sha256(REPO / "services" / "versions.lock"),
            "cases_sha256": _t93.CASES_SHA256,
            "stress_sha256": STRESS_SHA256,
            "tools_sha256": kit.tools_version,
            "tools": TOOLS_ON,
            "repair": "on",
            "repair_deadline_ms": REPAIR_DEADLINE_MS,
            "profile": profile.name,
            "canvas_length": profile.diffusion.canvas_length,
            "max_denoising_steps": profile.diffusion.max_denoising_steps,
            "thinking": profile.diffusion.thinking,
            "k": K,
            "arm_sizes": {"S": len(stress) * K, "F": len(forced)},
            "order": "arm S (file order x k, repetition-major), then arm F (file order)",
            "forced_reason": FORCED_REASON,
            "seed": None,
            "concurrency": 1,
            "input": "typed (UserText); fixture frames; fake STT/TTS (300 ms/word)/audio/perception; no audio",
            "barge_in": f"arm F, at most {BARGE_IN_MAX_ATTEMPTS} attempts (Q8) in run order among "
            f"{list(BARGE_IN_CANDIDATES)}; read with >= {BARGE_IN_N} delivered",
            "repair_warmup": repair_warmup,
        },
        "sessions": [],
        "cases": [],
    }
    _write_json(out / "manifest.json", manifest)
    run = _Run(base_url=args.base_url, out=out, profile=profile, manifest=manifest)
    asyncio.run(_run_all(run, stress, forced))

    report = build_report(out)
    _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    g = gates(report)
    for arm in ARMS:
        for key in GATE_KEYS:
            _say(f"arm {arm} gate {key}: {'holds' if g[arm][key] else 'FAILS'}")
    for reason in report["void"]:
        _say(f"void: {reason}")
    _say(f"written to {out}")
    ok = not report["void"] and all(all(v.values()) for v in g.values())
    return EXIT_OK if ok else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
