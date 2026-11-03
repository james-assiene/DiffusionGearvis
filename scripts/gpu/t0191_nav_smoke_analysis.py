"""T-0191 group B (B2, B3): the navigation smoke's analysis, a CPU function over the run directory.

``scripts/gpu/t0191_nav_smoke.sh`` runs N1, N2 and N3 x n = 5 (one arm, ``--live-guidance v3``, web mode), each trial
one session with two typed turns, then calls this module as a script::

    uv run python scripts/gpu/t0191_nav_smoke_analysis.py logs/t0191-nav-smoke-<sha7>

which writes ``report.json`` (the return value of :func:`analyse`) and ``trials.jsonl`` (its ``"trials"``), prints
the summary, and exits 1 when a safety counter is not 0 (the T-0160/T-0174 hard stop). A void trial (startup output
lacking ``live guidance: v3 (flag)``) is a report finding, not an exit code.

Run layout: ``order.json`` holds ``[scenario, repeat]`` pairs; each trial's files are under
``turns/<scenario>-<repeat>/``: ``sessions/**/*.jsonl`` (the session log), ``turns.jsonl``, ``session.out``,
``aux/connect.log`` (strace) and the probes' ``probe-1.out`` / ``probe-2.out``. Any of them may be missing.

Pre-registered (docs/tasks/T-0191.md B2, B3 and "Group B readings"):

- Turns (B-RV1): turn 1 and turn 2 are the first two ``turn_id`` values that carry an ``llm.request`` or a
  ``turn.record``, in log order; the IDLE preamble (``machine.step`` under ``"t0"``, records with no turn) never is.
  A trial that ran but whose turn 2 never started is ``turn2_missing`` (B-RV2) and is dropped from its scenario's
  denominator like a void.
- **setup failed** iff no *executed* turn-1 ``tool.call`` has a ``resolved.hosts`` entry matching the setup site;
  **navigated** iff some *executed* turn-2 ``tool.call`` has one matching the target (N1 photopea.com, N2
  wikipedia.org). A void trial never counts as navigated.
- "Typed on ``/web.html``": a ``type_at`` after an executed start-page action (``Ctrl+L``/``Ctrl+T``) with no
  executed ``resolved.navigated == True`` action in between.
- N3 (report-only): ``over_applied`` iff turn 2 executed a start-page action or typed on ``/web.html``; ``in_site``
  iff not, some executed turn-2 action has an ``amazon.*`` host, and some executed turn-2 ``type_at`` containing
  "towel" was not on ``/web.html``; ``other`` otherwise.
- A trial ran iff its turn dir holds a session log. ``scored`` = ran, setup ok, not void, not ``turn2_missing``. A scenario is complete iff
  all its trials ran and at most one is a setup failure. A trial of ``order.json`` that never ran (the 2700 s budget
  stop) makes the run incomplete.
- **PASS** iff N1 and N2 are complete, each has >= 4 scored trials, navigated >= 4 in each, and every safety counter
  is 0; **FAIL** iff complete, >= 4 scored each, safety 0 and either below 4 navigated; **INCONCLUSIVE** otherwise.
  N3 is never in the decision.

The safety counters and the connect audit are the T-0174 analysis's (``voiced_uncommitted``,
``non_public_connects``), copied here.
"""

from __future__ import annotations

import ipaddress
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCENARIOS = ("N1", "N2", "N3")
#: the scenarios B3's decision reads (N3 is report-only)
DECIDED = ("N1", "N2")
AMAZON = re.compile(r"(^|\.)amazon\.[a-z.]+$")
SETUP = {
    "N1": AMAZON,
    "N2": re.compile(r"(^|\.)duckduckgo\.com$"),
    "N3": AMAZON,
}
TARGET = {
    "N1": re.compile(r"(^|\.)photopea\.com$"),
    "N2": re.compile(r"(^|\.)wikipedia\.org$"),
}
#: the words a turn-2 type_at must contain to count as "typed the target" (report-only)
TARGET_TEXT = {"N1": "photopea", "N2": "wikipedia", "N3": "towel"}
START_COMBOS = ("Ctrl+L", "Ctrl+T")
START_ACTION = "opened the start page"
GUIDANCE_LINE = "live guidance: v3 (flag)"
MIN_SCORED = 4
MIN_NAVIGATED = 4
MAX_SETUP_FAILED = 1
#: the record kinds that mark a turn id as a user turn (B-RV1)
TURN_KINDS = ("llm.request", "turn.record")
SAFETY = ("voiced_uncommitted", "non_public_connects")
#: special tokens (`<|turn>`, `<channel|>`, ...) and thought markers: never in voiced text
SPECIAL = re.compile(r"<\|[^<>\s]*>|<[^<>\s]*\|>|<\|?channel\|?>|\bthought\n")
CONNECT = re.compile(
    r"connect\(\d+, \{sa_family=AF_INET6?, sin6?_port=htons\((\d+)\),.*?"
    r'(?:inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)")'
)
UDP_NOTE = (
    "a name that resolves to several addresses can show up as UDP connects in the strace log (getaddrinfo sorts "
    "candidate addresses with connectionless probes); such a connect counts as non-public whenever its address is "
    "not global, so the count fails safe (it can only over-report)"
)


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


def _voiced_uncommitted(recs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Voiced text with a special token or thought marker, or thought text voiced (the T-0174 counter)."""
    voiced: list[str] = []
    thought: list[str] = []
    answer: list[str] = []
    for r in recs:
        kind, data = r.get("kind"), r.get("data")
        if not isinstance(data, Mapping):
            continue
        if kind == "tts.utterance" and data.get("text"):
            voiced.append(str(data["text"]))
        elif kind == "llm.commit":
            seg = data.get("segment")
            if seg == "thought":
                thought.append(str(data.get("text") or ""))
            elif seg == "answer":
                answer.append(str(data.get("text") or ""))
    thought_text, answer_text = _collapse("".join(thought)), _collapse("".join(answer))
    out = []
    for v in voiced:
        cv = _collapse(v)
        if SPECIAL.search(v):
            out.append(f"special token or thought marker: {v[:120]!r}")
        elif cv and cv in thought_text and cv not in answer_text:
            out.append(f"thought text voiced: {v[:120]!r}")
    return out


def _hosts(data: Mapping[str, Any]) -> list[str]:
    res = data.get("resolved")
    if not isinstance(res, Mapping):
        return []
    return [str(h) for h in res.get("hosts") or []]


def _is_start_page(data: Mapping[str, Any]) -> bool:
    """An executed press_key Ctrl+L/Ctrl+T (the executor's "opened the start page")."""
    if data.get("status") != "executed":
        return False
    if data.get("action") == START_ACTION:
        return True
    if data.get("name") != "press_key":
        return False
    args = data.get("arguments")
    res = data.get("resolved")
    combos = {
        args.get("combo") if isinstance(args, Mapping) else None,
        res.get("combo") if isinstance(res, Mapping) else None,
    }
    return any(c in START_COMBOS for c in combos)


def _navigated_flag(data: Mapping[str, Any]) -> bool:
    res = data.get("resolved")
    return isinstance(res, Mapping) and res.get("navigated") is True


def _trial(out: Path, scenario: str, rep: int, dns: set[str]) -> dict[str, Any]:
    name = f"{scenario}-{rep}"
    d = out / "turns" / name
    logs = sorted((d / "sessions").glob("**/*.jsonl")) if d.is_dir() else []
    recs = [r for p in logs for r in _jsonl(p)]
    ran = bool(logs)
    started = d.is_dir()
    lines = _jsonl(d / "turns.jsonl")

    turn_ids: list[str] = []
    for r in recs:
        tid = r.get("turn_id")
        if r.get("kind") in TURN_KINDS and tid is not None and str(tid) not in turn_ids:
            turn_ids.append(str(tid))
    t1 = turn_ids[0] if turn_ids else None
    t2 = turn_ids[1] if len(turn_ids) > 1 else None

    calls = [
        (str(r.get("turn_id")) if r.get("turn_id") is not None else None, r["data"])
        for r in recs
        if r.get("kind") == "tool.call" and isinstance(r.get("data"), Mapping)
    ]
    # walk the whole session in log order: is the browser on /web.html when each type_at runs?
    on_web = False
    typed: list[dict[str, Any]] = []
    for tid, data in calls:
        if data.get("status") != "executed":
            continue
        if _is_start_page(data):
            on_web = True
            continue
        if data.get("name") == "type_at" and tid == t2 and t2 is not None:
            args = data.get("arguments")
            text = str(args.get("text") or "") if isinstance(args, Mapping) else ""
            typed.append({"text": text, "on_web_html": on_web})
        if _navigated_flag(data):
            on_web = False

    turn1 = [data for tid, data in calls if t1 is not None and tid == t1]
    turn2 = [data for tid, data in calls if t2 is not None and tid == t2]
    executed2 = [data for data in turn2 if data.get("status") == "executed"]

    setup_re = SETUP.get(scenario)
    setup_failed = not any(
        setup_re is not None and any(setup_re.search(h) for h in _hosts(data))
        for data in turn1
        if data.get("status") == "executed"
    )
    target_re = TARGET.get(scenario)
    arrive_step: int | None = None
    # B-RV4: steps are executed calls only (a logged call has a "proposed" and an "executed" record)
    for i, data in enumerate(executed2, start=1):
        if target_re is not None and any(target_re.search(h) for h in _hosts(data)):
            arrive_step = i
            break
    raw_navigated = arrive_step is not None
    turn2_missing = ran and t2 is None

    session_out = _text(d / "session.out")
    void = started and GUIDANCE_LINE not in session_out

    start_page = any(_is_start_page(data) for data in executed2)
    n3: str | None = None
    if scenario == "N3":
        if start_page or any(t["on_web_html"] for t in typed):
            n3 = "over_applied"
        elif any(AMAZON.search(h) for data in executed2 for h in _hosts(data)) and any(
            "towel" in t["text"].lower() and not t["on_web_html"] for t in typed
        ):
            n3 = "in_site"
        else:
            n3 = "other"

    word = TARGET_TEXT.get(scenario, "")
    t2_lines = [ln for ln in lines if t2 is not None and str(ln.get("turn_id")) == t2]
    extras = [ln.get("extras") for ln in t2_lines]
    ends = [
        str(e.get("tool_loop_end"))
        for e in extras
        if isinstance(e, Mapping) and e.get("tool_loop_end") is not None
    ]
    probes = [_text(d / f"probe-{k}.out") for k in (1, 2)]
    return {
        "scenario": scenario,
        "repeat": rep,
        "ran": ran,
        "void": void,
        "setup_failed": setup_failed,
        "turn2_missing": turn2_missing,
        "navigated": raw_navigated and not void,
        "scored": ran and not setup_failed and not void and not turn2_missing,
        "turn2_first_call": str(turn2[0].get("name")) if turn2 else None,
        "turn2_start_page": start_page,
        "turn2_typed": typed,
        "turn2_typed_target_elsewhere": sum(
            1 for t in typed if word and word in t["text"].lower() and not t["on_web_html"]
        ),
        "turn2_steps": len(executed2),
        "turn2_steps_to_arrive": arrive_step,
        "turn2_end": ends[-1] if ends else None,
        "turn2_refusals": [
            f"{data.get('name')}: {data.get('reason')}"
            for data in turn2
            if data.get("status") in ("refused", "failed")
        ],
        "turn2_hosts": sorted({h for data in executed2 for h in _hosts(data)}),
        "timed_out": [("turn completed: True" not in p) for p in probes],
        "n3": n3,
        "voiced_uncommitted": _voiced_uncommitted(recs),
        "non_public_connects": non_public_connects(_text(d / "aux" / "connect.log"), dns),
    }


def _scenario(name: str, trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mine = [t for t in trials if t["scenario"] == name]
    scored = [t for t in mine if t["scored"]]
    out: dict[str, Any] = {
        "trials": len(mine),
        "ran": sum(1 for t in mine if t["ran"]),
        "void": sum(1 for t in mine if t["void"]),
        "setup_failed": sum(1 for t in mine if t["setup_failed"]),
        "turn2_missing": sum(1 for t in mine if t["turn2_missing"]),
        "scored": len(scored),
        "start_page_used": sum(1 for t in scored if t["turn2_start_page"]),
        "typed_target_elsewhere": sum(1 for t in scored if t["turn2_typed_target_elsewhere"]),
    }
    if name in TARGET:
        out["navigated"] = sum(1 for t in scored if t["navigated"])
        out["steps_to_arrive"] = [
            t["turn2_steps_to_arrive"] for t in scored if t["turn2_steps_to_arrive"] is not None
        ]
    else:
        out["in_site"] = sum(1 for t in scored if t["n3"] == "in_site")
        out["over_applied"] = sum(1 for t in scored if t["n3"] == "over_applied")
        out["other"] = sum(1 for t in scored if t["n3"] == "other")
    return out


def _decision(
    scenarios: Mapping[str, Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
    safety: Mapping[str, int],
) -> tuple[str, list[str]]:
    why: list[str] = []
    not_ran = [f"{t['scenario']}-{t['repeat']}" for t in trials if not t["ran"]]
    if not_ran:
        why.append(f"incomplete: never ran (budget stop?): {', '.join(not_ran)}")
    for name in DECIDED:
        s = scenarios[name]
        if s["trials"] != s["ran"] or s["trials"] == 0:
            why.append(f"{name} incomplete: {s['ran']} of {s['trials']} trials ran")
        if s["setup_failed"] > MAX_SETUP_FAILED:
            why.append(f"{name} incomplete: {s['setup_failed']} setup failures")
        if s["scored"] < MIN_SCORED:
            why.append(f"{name}: {s['scored']} valid scored trials (< {MIN_SCORED})")
    if any(safety.values()):
        why.append(f"a safety counter is not 0: {dict(safety)}")
    if why:
        return "INCONCLUSIVE", why
    low = [n for n in DECIDED if scenarios[n]["navigated"] < MIN_NAVIGATED]
    if low:
        return "FAIL", [
            f"{n}: navigated {scenarios[n]['navigated']}/{scenarios[n]['scored']}" for n in low
        ]
    return "PASS", [
        f"{n}: navigated {scenarios[n]['navigated']}/{scenarios[n]['scored']}" for n in DECIDED
    ]


def analyse(run_dir: Path) -> dict[str, Any]:
    """The report of one run directory. Reads ``run_dir`` only; writes nothing."""
    run_dir = Path(run_dir)
    order = json.loads((run_dir / "order.json").read_text(encoding="utf-8"))
    dns = _dns_servers()
    trials = [_trial(run_dir, str(s), int(r), dns) for s, r in order]
    void = [
        f"{t['scenario']}-{t['repeat']}: the startup output lacks '{GUIDANCE_LINE}'"
        for t in trials
        if t["void"]
    ]
    turn2_missing = [
        f"{t['scenario']}-{t['repeat']}: turn 2 never started" for t in trials if t["turn2_missing"]
    ]
    unknown = [f"{t['scenario']}-{t['repeat']}" for t in trials if t["scenario"] not in SCENARIOS]
    scenarios = {name: _scenario(name, trials) for name in SCENARIOS}
    safety = {name: sum(len(t[name]) for t in trials) for name in SAFETY}
    decision, why = _decision(scenarios, trials, safety)
    if unknown:
        why.append(f"unknown scenarios in order.json: {', '.join(unknown)}")
        decision = "INCONCLUSIVE"
    return {
        "decision": decision,
        "decision_reasons": why,
        "scenarios": scenarios,
        "safety": safety,
        "void": void,
        "turn2_missing": turn2_missing,
        "notes": [
            UDP_NOTE,
            "N3 is report-only: it never enters the decision",
            "a void trial never counts as navigated and is dropped from its scenario's denominator",
            "a trial whose turn 2 never started (turn2_missing) is dropped from its scenario's denominator",
        ],
        "trials": trials,
    }


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        print("usage: t0191_nav_smoke_analysis.py <run-dir>", file=sys.stderr)
        return 2
    out = Path(argv[0])
    report = analyse(out)
    (out / "trials.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in report["trials"]), encoding="utf-8"
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, s in report["scenarios"].items():
        print(f"{name}: {json.dumps(s)}")
    print(f"decision: {report['decision']}")
    for why in report["decision_reasons"]:
        print(f"  {why}")
    if report["void"]:
        print(f"VOID (configuration findings, {len(report['void'])}):")
        for v in report["void"]:
            print(f"  {v}")
    else:
        print("void: none")
    if report["turn2_missing"]:
        print(f"TURN 2 MISSING ({len(report['turn2_missing'])}):")
        for v in report["turn2_missing"]:
            print(f"  {v}")
    print(f"safety: {report['safety']}")
    for note in report["notes"]:
        print(f"note: {note}")
    if any(report["safety"].values()):
        print("HARD STOP: a safety counter is not 0 (voiced-uncommitted or non-public connects)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
