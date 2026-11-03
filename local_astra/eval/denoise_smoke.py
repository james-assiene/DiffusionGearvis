"""T-0128c C2: the pure gates G1-G7 of the `/launch` denoising smoke.

The GPU driver ``scripts/gpu/t0128c_denoise_smoke.py`` collects the evidence (page samples, the
session log, the engine log, ``ss -tnp``) into ``logs/t0128c-smoke-<sha>/evidence.json``; this
module decides the gates from that evidence alone, with no I/O. The evidence format is defined in
``tests/unit/test_denoise_smoke_checks.py``'s docstring; the gates are the task's Regime
(``docs/tasks/T-0128c.md``).
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from local_astra.bench.b16_tap import parse_tap_release_lines

GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")

#: the Regime's bars (docs/tasks/T-0128c.md "Gates")
MIN_FRAMES_PER_TURN = 5
MIN_DISTINCT_PER_TURN = 3
#: the typed turns the smoke sends (scripts/gpu/t0128c_denoise_smoke.py ``PROMPTS``); O-C2 (1)
EXPECTED_TURNS = 3

#: the page's denoise-frame label (hud/src/denoise/model.js ``label``), never the commit form
_DENOISE_LABEL = re.compile(r"^canvas \d+ · step \d+/\d+$")

_PID = re.compile(r"\bpid=(\d+),")


def _msg(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The published message of a ``wire.out`` record, else None."""
    if record.get("kind") != "wire.out":
        return None
    data = record.get("data")
    if not isinstance(data, Mapping):
        return None
    msg = data.get("msg")
    return msg if isinstance(msg, Mapping) else None


def _wire_msgs(session_log: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for rec in session_log:
        msg = _msg(rec)
        if msg is not None:
            yield msg


def relayed_frames(session_log: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One ``{request_id, state, turn_id}`` per relayed ``astra.canvas.step``, in log order.

    ``state``/``turn_id`` are those of the most recent non-replay ``astra.turn.state`` published
    before the frame (``"IDLE"``/``None`` before any). A dropped row was still relayed.
    """
    state: str = "IDLE"
    turn_id: str | None = None
    out: list[dict[str, Any]] = []
    for msg in _wire_msgs(session_log):
        kind = msg.get("type")
        if kind == "astra.turn.state":
            if msg.get("replay"):
                continue
            state = str(msg.get("state"))
            tid = msg.get("turn_id")
            turn_id = None if tid is None else str(tid)
        elif kind == "astra.canvas.step":
            out.append({"request_id": msg.get("request_id"), "state": state, "turn_id": turn_id})
    return out


def _last_frames(samples: Sequence[Mapping[str, Any]]) -> int:
    for s in reversed(samples):
        v = s.get("frames")
        if v is not None:
            return int(v)
    return 0


def _g1(turns: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    for turn in turns:
        tid = turn.get("turn_id")
        samples = turn.get("samples") or []
        if not samples:
            problems.append(f"turn {tid}: no page samples")
            continue
        base = samples[0].get("frames")
        decoded = _last_frames(samples) - (0 if base is None else int(base))
        if decoded < MIN_FRAMES_PER_TURN:
            problems.append(
                f"turn {tid}: {decoded} frames decoded by the page (< {MIN_FRAMES_PER_TURN})"
            )
    return problems


def _g2(turns: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    for turn in turns:
        tid = turn.get("turn_id")
        samples = turn.get("samples") or []
        if not samples:
            problems.append(f"turn {tid}: no page samples")
            continue
        base = samples[0]
        for field in ("digest", "label"):
            carried = base.get(field)
            values = {
                s.get(field)
                for s in samples[1:]
                if s.get(field) is not None and s.get(field) != carried
            }
            if field == "label":
                # O-C2 (2): only denoise-form labels show steps moving; a commit-only stream
                # ("canvas k · committed at step s/N") across canvases is not the denoising view
                values = {v for v in values if _DENOISE_LABEL.match(str(v))}
            if len(values) < MIN_DISTINCT_PER_TURN:
                what = "denoise step labels" if field == "label" else "digests"
                problems.append(
                    f"turn {tid}: {len(values)} distinct new {what} "
                    f"(< {MIN_DISTINCT_PER_TURN}; carried over: {carried!r})"
                )
    return problems


def _g3(
    turns: Sequence[Mapping[str, Any]],
    relayed: Sequence[Mapping[str, Any]],
    release_ids: set[str],
) -> list[str]:
    problems: list[str] = []
    for turn in turns:
        tid = turn.get("turn_id")
        for s in turn.get("samples") or []:
            m = s.get("malformed")
            if m:
                problems.append(f"turn {tid}: page malformed count {m} at t_ms={s.get('t_ms')}")
    seen: set[str] = set()
    for f in relayed:
        rid = str(f["request_id"])
        if rid in seen:
            continue
        seen.add(rid)
        if rid not in release_ids:
            problems.append(f"relayed request {rid} has no engine tap release line")
    return problems


def _g4(relayed: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    first: dict[str, Mapping[str, Any]] = {}
    for f in relayed:
        first.setdefault(str(f["request_id"]), f)
    for rid, f in first.items():
        if f["state"] == "IDLE":
            problems.append(
                f"relayed request {rid} was first seen while IDLE (turn {f['turn_id']})"
            )
    return problems


def _is_loopback(addr: str) -> bool:
    """``127.0.0.1:8001``, ``[::1]:8081``, ``[::ffff:127.0.0.1]:8080``, ``127.0.0.53%lo:53``."""
    host = addr.rsplit(":", 1)[0] if ":" in addr else addr
    host = host.strip("[]").split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def _g5(ss_lines: Sequence[str], pid: int) -> list[str]:
    problems: list[str] = []
    n = 0
    for ln in ss_lines:
        if pid not in {int(p) for p in _PID.findall(ln)}:
            continue
        cols = ln.split()
        if len(cols) < 5:
            problems.append(f"unparseable ss line of pid {pid}: {ln!r}")
            continue
        n += 1
        local, peer = cols[3], cols[4]
        for which, addr in (("local", local), ("peer", peer)):
            if not _is_loopback(addr):
                problems.append(f"conductor socket with non-loopback {which} address {addr}")
    if n == 0:
        problems.append(f"no ss line of the conductor (pid={pid}): no evidence is not a pass")
    return problems


def _error_text(msg: Mapping[str, Any]) -> str:
    return f"astra.error {msg.get('code')}: {msg.get('message')}"


def _record_turn_id(rec: Mapping[str, Any]) -> str | None:
    tid = rec.get("turn_id")
    if tid is None and isinstance(rec.get("data"), Mapping):
        tid = rec["data"].get("turn_id")
    return None if tid is None else str(tid)


def _teardown_start(
    turns: Sequence[Mapping[str, Any]], session_log: Sequence[Mapping[str, Any]]
) -> int | None:
    """Index of the first ``client.disconnected`` machine step after the last turn's
    ``turn.record`` (the last turn = the last row of ``turns``).

    Ruling O-C5 (2): the harness's own page disconnecting once the last turn has closed is its
    teardown, not the session's. None when there is no turn, no ``turn.record`` of the last turn,
    or no such disconnect: then nothing is teardown.
    """
    if not turns:
        return None
    last_tid = str(turns[-1].get("turn_id"))
    last_record = None
    for i, rec in enumerate(session_log):
        if rec.get("kind") == "turn.record" and _record_turn_id(rec) == last_tid:
            last_record = i
    if last_record is None:
        return None
    for i in range(last_record + 1, len(session_log)):
        rec = session_log[i]
        if rec.get("kind") != "machine.step":
            continue
        data = rec.get("data")
        event = data.get("event") if isinstance(data, Mapping) else None
        if isinstance(event, Mapping) and event.get("type") == "client.disconnected":
            return i
    return None


def teardown_errors(
    turns: Sequence[Mapping[str, Any]], session_log: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Reported, not gated (O-C5 (2)): each ``astra.error`` G6 ignored as the harness's teardown,
    formatted like G6's problems (``"astra.error <code>: <message>"``)."""
    start = _teardown_start(turns, session_log)
    if start is None:
        return []
    out: list[str] = []
    for rec in session_log[start + 1 :]:
        msg = _msg(rec)
        if msg is not None and msg.get("type") == "astra.error":
            out.append(_error_text(msg))
    return out


def _g6(turns: Sequence[Mapping[str, Any]], session_log: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    spoke: set[str] = set()
    start = _teardown_start(turns, session_log)
    for i, rec in enumerate(session_log):
        msg = _msg(rec)
        if msg is None:
            continue
        kind = msg.get("type")
        if kind == "astra.spoken" and not msg.get("replay"):
            text = msg.get("text")
            if isinstance(text, str) and text.strip():
                spoke.add(str(msg.get("turn_id")))
        elif kind == "astra.error" and (start is None or i < start):
            # O-C5 (2): errors after the teardown start are reported (teardown_errors), not gated
            problems.append(_error_text(msg))
    for turn in turns:
        tid = str(turn.get("turn_id"))
        if tid not in spoke:
            problems.append(f"turn {tid}: nothing spoken")
    return problems


def _g7(rows: Sequence[Mapping[str, str | int]]) -> list[str]:
    if not rows:
        return ["no canvas tap release line in the engine log"]
    problems: list[str] = []
    for r in rows:
        if r["errors"] != 0 or r["commit_overrun"] != 0:
            problems.append(
                f"tap release {r['req_id']}: errors={r['errors']} "
                f"commit_overrun={r['commit_overrun']}"
            )
    return problems


def gates(evidence: Mapping[str, Any]) -> dict[str, list[str]]:
    """Every gate's problems (``[]`` = pass). A malformed release line raises ``ValueError``."""
    turns: Sequence[Mapping[str, Any]] = evidence.get("turns") or []
    log: Sequence[Mapping[str, Any]] = evidence.get("session_log") or []
    rows = parse_tap_release_lines(str(evidence.get("release_log") or ""))
    relayed = relayed_frames(log)
    release_ids = {str(r["req_id"]) for r in rows}
    # O-C2 (1): the per-turn gates hold only over exactly the typed turns; fewer (zero included)
    # or more is a problem in each of them, never a vacuous pass
    count: list[str] = []
    if len(turns) != EXPECTED_TURNS:
        count = [f"{len(turns)} turns in the evidence, expected exactly {EXPECTED_TURNS}"]
    return {
        "G1": count + _g1(turns),
        "G2": count + _g2(turns),
        "G3": _g3(turns, relayed, release_ids),
        "G4": _g4(relayed),
        "G5": _g5(evidence.get("ss_lines") or [], int(evidence["conductor_pid"])),
        "G6": count + _g6(turns, log),
        "G7": _g7(rows),
    }
