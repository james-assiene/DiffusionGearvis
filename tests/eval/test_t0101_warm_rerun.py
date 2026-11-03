"""T-0101 criterion 8 (H-T101b): the pre-registered warm re-run of the T-0099 harness, re-derived (``gpu``, ``eval``).

Run through ``/launch`` on the 5090 box, by ``scripts/locked/t0101_warm_rerun.sh`` after the harness ran with
``--warm-repair`` on a fresh boot::

    T0101_RUN_DIR=logs/t0101-warm-rerun-<sha> uv run pytest tests/eval/test_t0101_warm_rerun.py -q -m gpu -rA -s

Without ``T0101_RUN_DIR`` every row skips with that reason. The rows spend no GPU time (they read ``turns.jsonl``,
``sessions/*.jsonl`` and ``manifest.json``); they are ``gpu`` because the data only exists after a reviewed run.

- ``report.json`` agrees with ``build_report(run_dir)``; the run is not void; G1-G8 hold in both arms; criterion 6:
  ``regime.repair_warmup.on`` with every payload tool ``ok``, and the warm-up ended no later than the first session's
  ``log.started``; no warm-up request id is in any session log.
- ``first_use_repair`` is **re-derived here** with the stdlib from the raw files (manifest session order, the ``cd``
  sha on each ``tool_repair`` ``llm.request``, the turn line's ``tool.repair`` span) and must equal the report's.
- **C1** every payload tool has a first use and each first-use span is <= 600 ms; **C2** arm-S ``tool.repair`` p90
  (nearest rank) <= 1500 ms when arm S fired >= 1 repair, else "unread (n = 0)" and C1 decides; **C3** arm-F
  ``tool.repair`` p50 in [203, 248] ms; **C4** (printed, never asserted) the slowest warm-up request >= 1000 ms.
  The reading (R1-R5) is the human's and the reviewer's; these rows only decide C1-C3.

Regime: T-0099's at a794a98 plus ``--warm-repair``: lean, engine only, fresh boot; vLLM v0.29.0; NVFP4 rev
``2ea8372``; template ``36e3a42e``; tokenizer ``cc8d3a0c``; CL 256, 48 steps; thinking on; prefix caching on;
``max_num_seqs 3``; RTX 5090, driver 595.84, gpu-mem 0.86; ``native:bbfc53ccdd80:dry_run``, repair on, deadline
3000 ms; typed turns; fake STT/audio; fake TTS 300 ms/word; fixture frames; concurrency 1; no seed; arm S 30 x k = 3,
arm F 40.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import live
from tests.eval.test_t0099_repair_live import report_disagreements
from tests.unit.test_t0099_report import GATE_KEYS, REPO, harness

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0101_RUN_DIR"
C1_MAX_MS = 600
C2_MAX_MS = 1500
C3_RANGE_MS = (203, 248)
C4_MIN_MS = 1000


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(
            f"{ENV} is not set: no T-0101 run directory (logs/t0101-warm-rerun-<sha>/) to read"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    for name in ("turns.jsonl", "manifest.json", "report.json", "report.md"):
        assert (path / name).is_file(), f"the run directory has no {name}"
    return path


@pytest.fixture(scope="module")
def report(run_dir: Path) -> dict[str, Any]:
    rebuilt = harness().build_report(run_dir)
    assert isinstance(rebuilt, dict)
    return rebuilt


@pytest.fixture(scope="module")
def manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _nearest_rank(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[max(1, math.ceil(q * len(s))) - 1]


def _span_ms(line: dict[str, Any] | None) -> float | None:
    pair = ((line or {}).get("spans") or {}).get("tool.repair")
    if not isinstance(pair, list | tuple) or len(pair) != 2:
        return None
    return float(pair[1] - pair[0])


def _lines_by_session(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for line in _jsonl(run_dir / "turns.jsonl"):
        out.setdefault(str(line.get("session_id")), []).append(line)
    return out


def _rederived_first_use(run_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Stdlib: per payload sha, the first ``tool_repair`` request in manifest session order (either arm)."""
    tool_of = {p.automaton_sha256: name for name, p in live.load_payloads().tools.items()}
    case_of = {str(c.get("session_id")): c for c in manifest.get("cases") or []}
    lines = _lines_by_session(run_dir)
    out: dict[str, dict[str, Any]] = {}
    for s in manifest.get("sessions") or []:
        sid = str(s.get("session_id"))
        records = sorted(
            _jsonl(run_dir / "sessions" / str(s["log"])), key=lambda r: int(r.get("seq", 0))
        )
        for r in records:
            d = r.get("data") or {}
            if r.get("kind") != "llm.request" or d.get("phase") != "tool_repair":
                continue
            sha = str((d.get("cd") or {}).get("automaton_sha256"))
            if sha in out:
                continue
            c = case_of.get(sid, {})
            mine = lines.get(sid) or []
            out[sha] = {
                "tool": tool_of.get(sha),
                "arm": c.get("arm", s.get("arm")),
                "case_id": c.get("case_id"),
                "rep": c.get("rep"),
                "ms": _span_ms(mine[0] if mine else None),
            }
    return out


def _arm_repair_ms(run_dir: Path, manifest: dict[str, Any], arm: str) -> list[float]:
    lines = _lines_by_session(run_dir)
    out = []
    for c in manifest.get("cases") or []:
        if c.get("arm") != arm:
            continue
        mine = lines.get(str(c.get("session_id"))) or []
        ms = _span_ms(mine[0] if mine else None)
        if ms is not None:
            out.append(ms)
    return out


# --- the run is what it says ---------------------------------------------------------------------------


def test_stored_report_agrees_with_the_rederived_one(run_dir: Path, report: dict[str, Any]) -> None:
    stored = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    differ = report_disagreements(stored, report)
    assert differ == [], f"report.json differs from build_report(run_dir) on {differ}"


def test_not_void_and_gates_hold(report: dict[str, Any]) -> None:
    assert report["void"] == [], f"the run is void: {report['void']}"
    g = harness().gates(report)
    for arm in ("S", "F"):
        assert set(g[arm]) == GATE_KEYS
        failing = sorted(k for k, v in g[arm].items() if v is not True)
        assert failing == [], f"arm {arm}: gates failing {failing} (no number is read)"


def test_warmup_complete_and_before_the_first_session(
    run_dir: Path, manifest: dict[str, Any]
) -> None:
    """Criterion 6: on, every payload tool ``ok``; ended no later than the first session's ``log.started``; no
    ``warmup-`` request id in any session log (criterion 2)."""
    rw = manifest["regime"].get("repair_warmup")
    assert isinstance(rw, dict) and rw.get("on") is True, rw
    per = rw.get("per_tool") or {}
    assert sorted(per) == sorted(live.load_payloads().tools), per
    bad = {t: v.get("status") for t, v in per.items() if v.get("status") != "ok"}
    assert bad == {}, f"criterion 6: warm-up incomplete {bad}"
    first = _jsonl(run_dir / "sessions" / str(manifest["sessions"][0]["log"]))
    started = next(r["wall"] for r in first if r.get("kind") == "log.started")
    assert datetime.fromisoformat(str(rw["ended_wall"])) <= datetime.fromisoformat(str(started)), (
        rw["ended_wall"],
        started,
    )
    for log in (run_dir / "sessions").glob("*.jsonl"):
        assert "warmup-" not in log.read_text(encoding="utf-8"), (
            f"a warm-up request id in {log.name}"
        )


def test_first_use_rederived_agrees(
    run_dir: Path, manifest: dict[str, Any], report: dict[str, Any]
) -> None:
    mine = _rederived_first_use(run_dir, manifest)
    theirs = report.get("first_use_repair") or {}
    keys = ("tool", "arm", "case_id", "rep", "ms")
    got = {
        sha: {k: (float(v[k]) if k == "ms" and v.get(k) is not None else v.get(k)) for k in keys}
        for sha, v in theirs.items()
        if v is not None
    }
    assert got == mine, f"first_use_repair disagrees with the stdlib re-derivation:\n{got}\n{mine}"
    worst = max((v["ms"] for v in mine.values() if v["ms"] is not None), default=None)
    assert report.get("first_use_repair_max_ms") == worst


# --- C1-C4 (H-T101b) -----------------------------------------------------------------------------------


def test_c1_every_first_use_within_600_ms(run_dir: Path, manifest: dict[str, Any]) -> None:
    first = _rederived_first_use(run_dir, manifest)
    seen = {v["tool"]: v for v in first.values()}
    missing = sorted(set(live.load_payloads().tools) - set(seen))
    assert missing == [], f"C1: no first use for {missing}"
    over = {t: v["ms"] for t, v in seen.items() if v["ms"] is None or v["ms"] > C1_MAX_MS}
    print(f"C1 first uses (ms): { {t: v['ms'] for t, v in sorted(seen.items())} }")
    assert over == {}, f"C1 fails (> {C1_MAX_MS} ms or no span): {over}"


def test_c2_arm_s_p90(run_dir: Path, manifest: dict[str, Any], report: dict[str, Any]) -> None:
    ms = _arm_repair_ms(run_dir, manifest, "S")
    if not ms:
        print("C2: unread (n = 0); C1 decides")
        assert report["S"]["repairs"]["n"] == 0
        return
    p90 = _nearest_rank(ms, 0.9)
    print(f"C2: arm-S tool.repair p90 {p90} ms (nearest rank, n = {len(ms)})")
    assert report["S"]["repairs"]["ms"]["p90"] == p90
    assert p90 <= C2_MAX_MS, f"C2 fails: arm-S p90 {p90} ms > {C2_MAX_MS}"


def test_c3_arm_f_p50(run_dir: Path, manifest: dict[str, Any], report: dict[str, Any]) -> None:
    ms = _arm_repair_ms(run_dir, manifest, "F")
    assert ms, "C3: arm F fired no repair"
    p50 = _nearest_rank(ms, 0.5)
    print(f"C3: arm-F tool.repair p50 {p50} ms (nearest rank, n = {len(ms)})")
    assert report["F"]["repairs"]["ms"]["p50"] == p50
    lo, hi = C3_RANGE_MS
    assert lo <= p50 <= hi, f"C3 fails: arm-F p50 {p50} ms outside [{lo}, {hi}]"


def test_c4_slowest_warmup_reported(manifest: dict[str, Any]) -> None:
    """C4 is reported, never asserted (R3 reads it)."""
    per = (manifest["regime"].get("repair_warmup") or {}).get("per_tool") or {}
    walls = {t: v.get("wall_ms") for t, v in per.items()}
    slowest = max((w for w in walls.values() if isinstance(w, int | float)), default=None)
    holds = slowest is not None and slowest >= C4_MIN_MS
    print(
        f"C4: warm-up wall ms {walls}; slowest {slowest}; C4 {'holds' if holds else 'does not hold'}"
    )
