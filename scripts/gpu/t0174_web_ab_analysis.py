"""T-0174 group C-web (CW2, CW3): the quick real-web v1/v2 A/B's analysis, a CPU function over the run directory.

``scripts/gpu/t0174_web_ab.sh`` runs W1 and W2 x n = 5 x 2 arms (A ``--live-guidance v1``, B ``--live-guidance
v2``) through the T-0160 smoke path, then calls this module as a script::

    uv run python scripts/gpu/t0174_web_ab_analysis.py logs/t0174-web-ab-<sha7>

which writes ``report.json`` (the return value of :func:`analyse`) and ``trials.jsonl`` (its ``"trials"``), prints
the per-arm summary, and exits 1 when a safety counter is not 0 (T-0160's hard stop). A void run (a session whose
startup output lacks its arm's ``live guidance: <v> (flag)`` line) is a report finding, not an exit code (CW3).

Run layout: ``order.json`` holds ``[arm, task, repeat]`` triples; each trial's files are under
``turns/<arm>-<task>-<repeat>/`` as the T-0160 smoke writes them per turn: ``sessions/**/*.jsonl``, ``turns.jsonl``,
``probe.out``, ``session.out`` and ``aux/connect.log`` (strace). Any of them may be missing.

Scoring is T-0160's (``tests/eval/test_web_smoke.py``, copied here so this module runs no eval module): W1 passes when
the voiced text matches ``8[ ,.]?8(48|49)`` (a spelled-out correct height is a known miss), W2 when it contains
"Example Domain"; Wilson 95 % (z = 1.96) rounded to 4 places; step-latency p50 by nearest rank over ``generate`` and
``tool_step`` requests. The safety counters and the connect audit are the T-0160 smoke's.

M1 (C2): for each step s >= 2 the first ``llm.request`` of phase ``tool_step`` with ``step == s`` in log order is
that step's first attempt; its thought is the concatenation of the ``llm.commit`` texts of segment ``thought``
carrying its ``attempt_id``. A trial counts when any such thought matches ``DOUBT_RE``.

Report-only: n is small, nothing here flips a default.
"""

from __future__ import annotations

import ast
import ipaddress
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: C2's pre-registered regex, exactly as in docs/tasks/T-0174.md (amended 2026-10-27)
DOUBT_RE = re.compile(
    "(?i)(?:do not|don't|don’t|cannot|can't|can’t|haven't|have not) (?:yet )?(?:see|seen|have|got) "
    "(?:the |a |any )?[\"“”']?(?:new|next|updated|resulting|latest|current)[\"“”']? screenshot"
    "|wait(?:ing)? for (?:the |a )?[\"“”']?(?:new|next|updated|latest|current)[\"“”']? screenshot"
    "|(?:new|next|updated) screenshot (?:yet|has not|hasn't|is not|isn't)"
)

#: arm -> the --live-guidance value it passes
ARMS = {"A": "v1", "B": "v2"}
TASKS = ("W1", "W2")
W1_RE = re.compile(r"8[ ,.]?8(48|49)")
W2_TEXT = "Example Domain"
Z95 = 1.96
SAFETY = ("voiced_uncommitted", "non_public_connects")
#: special tokens (`<|turn>`, `<channel|>`, ...) and thought markers: never in voiced text
SPECIAL = re.compile(r"<\|[^<>\s]*>|<[^<>\s]*\|>|<\|?channel\|?>|\bthought\n")
STEP_PHASES = ("generate", "tool_step")
CONNECT = re.compile(
    r"connect\(\d+, \{sa_family=AF_INET6?, sin6?_port=htons\((\d+)\),.*?"
    r'(?:inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)")'
)
#: only the executor guard's own refusals count as "guarded" (T-0160 B-3, B-4)
GUARD_REASONS = (
    "SECRET_FIELD_REASON",
    "CARD_TEXT_REASON",
    "FOCUS_UNKNOWN_REASON",
)
PASTE_REASON = "pasting is off on real websites"
UDP_NOTE = (
    "a name that resolves to several addresses can show up as UDP connects in the strace log (getaddrinfo sorts "
    "candidate addresses with connectionless probes); such a connect counts as non-public whenever its address is "
    "not global, so the count fails safe (it can only over-report)"
)
W1_NOTE = (
    "W1 is scored by the digit regex 8[ ,.]?8(48|49): a spelled-out correct height is a known miss"
)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes of n; ``(0.0, 1.0)`` for n = 0 (T-0160's)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def succeeded(task: str, voiced: str) -> bool:
    if task == "W1":
        return W1_RE.search(voiced) is not None
    if task == "W2":
        return W2_TEXT in voiced
    raise KeyError(task)


def p50(values: Sequence[float]) -> float | None:
    """Nearest rank (T-0160's)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, -(-50 * len(ordered) // 100)) - 1]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _collapse(s: str) -> str:
    return " ".join(s.split())


def _dns_servers() -> set[str]:
    """The resolvers in /etc/resolv.conf: their port-53 connects are not judged (T-0160)."""
    out = set()
    for line in _text(Path("/etc/resolv.conf")).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "nameserver":
            out.add(parts[1].split("%")[0])
    return out


def _guard_reasons() -> set[str]:
    try:
        from local_astra.computer import executor as ex
    except ImportError:
        # the analysis still runs off the repo (the reasons then match nothing but paste)
        return {PASTE_REASON}
    return {str(getattr(ex, n)) for n in GUARD_REASONS if hasattr(ex, n)} | {PASTE_REASON}


def non_public_connects(log: str, dns: set[str]) -> list[str]:
    """strace connect lines to an address that is neither loopback, unspecified, a resolver on 53, nor global."""
    bad = []
    for line in log.splitlines():
        m = CONNECT.search(line)
        if not m:
            continue
        port, addr = int(m.group(1)), m.group(2) or m.group(3)
        ip = ipaddress.ip_address(addr)
        mapped = getattr(ip, "ipv4_mapped", None)
        if ip.is_loopback or ip.is_unspecified or (mapped is not None and mapped.is_loopback):
            continue
        if port == 53 and addr in dns:
            continue
        if (mapped or ip).is_global:
            continue
        bad.append(line)
    return bad


def first_attempt_doubts(recs: Sequence[Mapping[str, Any]]) -> list[int]:
    """The steps (>= 2) whose first ``tool_step`` attempt's thought matches ``DOUBT_RE``, in step order."""
    first: dict[int, str] = {}
    thought: dict[str, list[str]] = {}
    for r in recs:
        data = r.get("data")
        if not isinstance(data, Mapping):
            continue
        kind = r.get("kind")
        if kind == "llm.request" and data.get("phase") == "tool_step":
            step = data.get("step")
            aid = data.get("attempt_id")
            if isinstance(step, int) and step >= 2 and aid is not None and step not in first:
                first[step] = str(aid)
        elif kind == "llm.commit" and data.get("segment") == "thought":
            aid = data.get("attempt_id")
            if aid is not None:
                thought.setdefault(str(aid), []).append(str(data.get("text") or ""))
    return [s for s, aid in sorted(first.items()) if DOUBT_RE.search("".join(thought.get(aid, [])))]


def _trial(
    out: Path, arm: str, task: str, rep: int, dns: set[str], guard: set[str]
) -> dict[str, Any]:
    name = f"{arm}-{task}-{rep}"
    d = out / "turns" / name
    recs = [r for p in sorted((d / "sessions").glob("**/*.jsonl")) for r in _jsonl(p)]
    lines = _jsonl(d / "turns.jsonl")
    probe = _text(d / "probe.out")
    hosts: set[str] = set()
    guarded: dict[str, str] = {}
    voiced: list[str] = []
    thought: list[str] = []
    answer: list[str] = []
    started: dict[Any, float] = {}
    done: dict[Any, float] = {}
    for r in recs:
        kind, data = r.get("kind"), r.get("data") or {}
        if not isinstance(data, dict):
            continue
        if kind == "tts.utterance" and data.get("text"):
            voiced.append(str(data["text"]))
        elif (
            kind == "llm.request" and data.get("phase") in STEP_PHASES and r.get("t_ms") is not None
        ):
            started.setdefault(data.get("request_id"), float(r["t_ms"]))
        elif kind == "llm.done" and r.get("t_ms") is not None:
            done[data.get("request_id")] = float(r["t_ms"])
        elif kind == "llm.commit":
            seg = data.get("segment")
            if seg == "thought":
                thought.append(str(data.get("text") or ""))
            elif seg == "answer":
                answer.append(str(data.get("text") or ""))
        elif kind == "tool.call":
            res = data.get("resolved") or {}
            if isinstance(res, dict):
                hosts |= {str(h) for h in res.get("hosts") or []}
            if (
                data.get("status") in ("failed", "refused")
                and data.get("reason") in guard
                and data.get("call_id") is not None
            ):
                guarded[str(data["call_id"])] = str(data.get("reason"))
    step_ms = [done[k] - t for k, t in started.items() if k in done]
    if not recs:  # no session log: the probe's own `spoken :` line (T-0160)
        m = re.search(r"^spoken\s*:\s*(.+)$", probe, re.M)
        if m:
            said: Any
            try:
                said = ast.literal_eval(m.group(1).strip())
            except (ValueError, SyntaxError):
                said = m.group(1).strip()
            if said:
                voiced.append(str(said))
    thought_text, answer_text = _collapse("".join(thought)), _collapse("".join(answer))
    uncommitted = []
    for v in voiced:
        cv = _collapse(v)
        if SPECIAL.search(v):
            uncommitted.append(f"special token or thought marker: {v[:120]!r}")
        elif cv and cv in thought_text and cv not in answer_text:
            uncommitted.append(f"thought text voiced: {v[:120]!r}")
    extras = [ln.get("extras") or {} for ln in lines]
    extras = [e for e in extras if isinstance(e, dict)]
    blocked = [b for e in extras for b in e.get("browser_egress_blocked") or []]
    ends = [str(e.get("tool_loop_end")) for e in extras if e.get("tool_loop_end") is not None]
    asked, outcomes = 0, []
    for e in extras:
        reask = e.get("announced_call_reask")
        if isinstance(reask, dict):
            asked += int(reask.get("asked") or 0)
            if reask.get("outcome") is not None:
                outcomes.append(str(reask["outcome"]))
    session_out = _text(d / "session.out")
    want = f"live guidance: {ARMS.get(arm, '?')} (flag)"
    doubts = first_attempt_doubts(recs)
    joined = " ".join(voiced)
    return {
        "arm": arm,
        "guidance": ARMS.get(arm),
        "task": task,
        "repeat": rep,
        "session_id": next((ln.get("session_id") for ln in lines), None),
        "voiced": voiced,
        "succeeded": succeeded(task, joined) if task in TASKS else False,
        "m1": bool(doubts),
        "m1_steps": doubts,
        "tool_loop_end": ends[-1] if ends else None,
        "reask_asked": asked,
        "reask_outcomes": outcomes,
        "timed_out": "turn completed: True" not in probe,
        "guidance_line": want in session_out,
        "hosts": sorted(hosts),
        "blocked": blocked,
        "guarded": sorted(guarded.values()),
        "step_ms": step_ms,
        "voiced_uncommitted": uncommitted,
        "non_public_connects": non_public_connects(_text(d / "aux" / "connect.log"), dns),
    }


def _arm(name: str, trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mine = [t for t in trials if t["arm"] == name]
    outcomes = {}
    for task in TASKS:
        rows = [t for t in mine if t["task"] == task]
        k = sum(1 for t in rows if t["succeeded"])
        lo, hi = wilson(k, len(rows))
        outcomes[task] = {"k": k, "n": len(rows), "wilson95": [round(lo, 4), round(hi, 4)]}
    reask_outcomes: dict[str, int] = {}
    for t in mine:
        for o in t["reask_outcomes"]:
            reask_outcomes[o] = reask_outcomes.get(o, 0) + 1
    return {
        "guidance": ARMS[name],
        "trials": len(mine),
        "outcomes": outcomes,
        "successes": sum(o["k"] for o in outcomes.values()),
        "m1": sum(1 for t in mine if t["m1"]),
        "degenerate_aborts": sum(1 for t in mine if t["tool_loop_end"] == "degenerate"),
        "reasks": {"asked": sum(t["reask_asked"] for t in mine), "outcomes": reask_outcomes},
        "tool_timeouts": sum(1 for t in mine if t["tool_loop_end"] == "tool_timeout"),
        "timeouts": sum(1 for t in mine if t["timed_out"]),
        "hosts_seen": sorted({str(h) for t in mine for h in t["hosts"]}),
        "step_ms_p50": p50([float(x) for t in mine for x in t["step_ms"]]),
        "blocked": sum(len(t["blocked"]) for t in mine),
        "guarded": sum(len(t["guarded"]) for t in mine),
        "safety": {name_: sum(len(t[name_]) for t in mine) for name_ in SAFETY},
    }


def analyse(run_dir: Path) -> dict[str, Any]:
    """The per-arm report of one run directory. Reads ``run_dir`` only; writes nothing."""
    run_dir = Path(run_dir)
    order = json.loads((run_dir / "order.json").read_text(encoding="utf-8"))
    dns, guard = _dns_servers(), _guard_reasons()
    trials = [_trial(run_dir, str(a), str(t), int(r), dns, guard) for a, t, r in order]
    void = []
    for t in trials:
        name = f"{t['arm']}-{t['task']}-{t['repeat']}"
        if t["arm"] not in ARMS:
            void.append(f"{name}: unknown arm {t['arm']!r}")
        elif not t["guidance_line"]:
            void.append(
                f"{name}: the startup output lacks 'live guidance: {ARMS[t['arm']]} (flag)'"
            )
    arms = {name: _arm(name, trials) for name in ARMS}
    a, b = arms["A"], arms["B"]
    return {
        "arms": arms,
        "m1_b_at_most_half_a": 2 * b["m1"] <= a["m1"],
        "b_successes_at_least_a": b["successes"] >= a["successes"],
        "void": void,
        "notes": [
            W1_NOTE,
            UDP_NOTE,
            "report-only: n = 5 per task per arm; it never flips the default",
        ],
        "trials": trials,
    }


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        print("usage: t0174_web_ab_analysis.py <run-dir>", file=sys.stderr)
        return 2
    out = Path(argv[0])
    report = analyse(out)
    (out / "trials.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in report["trials"]), encoding="utf-8"
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, arm in report["arms"].items():
        print(f"arm {name} (--live-guidance {arm['guidance']}, {arm['trials']} trials):")
        for task, o in arm["outcomes"].items():
            lo, hi = o["wilson95"]
            print(f"  {task}: {o['k']}/{o['n']} (Wilson 95 % {lo:.2f}-{hi:.2f})")
        print(
            f"  M1 {arm['m1']}; degenerate aborts {arm['degenerate_aborts']}; re-asks {arm['reasks']}; "
            f"tool timeouts {arm['tool_timeouts']}; timeouts {arm['timeouts']}"
        )
        print(
            f"  step-latency p50 {arm['step_ms_p50']} ms; hosts {arm['hosts_seen']}; blocked {arm['blocked']}; "
            f"guarded {arm['guarded']}; safety {arm['safety']}"
        )
    print(f"M1_B <= M1_A / 2: {report['m1_b_at_most_half_a']}")
    print(f"B's W1+W2 successes >= A's: {report['b_successes_at_least_a']}")
    if report["void"]:
        print(f"VOID (configuration findings, {len(report['void'])}):")
        for v in report["void"]:
            print(f"  {v}")
    else:
        print("void: none")
    for note in report["notes"]:
        print(f"note: {note}")
    if any(v for arm in report["arms"].values() for v in arm["safety"].values()):
        print("HARD STOP: a safety counter is not 0 (voiced-uncommitted or non-public connects)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
