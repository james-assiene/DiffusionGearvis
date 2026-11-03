"""T-0099: the live repair run's report, re-derived from the raw files (``gpu``, ``eval``).

Run through ``/launch`` on the 5090 box, by ``scripts/locked/t0099_repair_live.sh`` after the harness::

    T0099_RUN_DIR=logs/t0099-repair-live-<sha> uv run pytest tests/eval/test_t0099_repair_live.py -q -m gpu -rA

Without ``T0099_RUN_DIR`` every row skips with that reason. The rows spend no GPU time themselves (they read
``turns.jsonl``, ``sessions/*.jsonl`` and ``manifest.json``); they are ``gpu`` because the data only exists after a
``/launch``-reviewed run.

- The report is **re-derived** with the harness's ``build_report`` and the stored ``report.json`` must agree with it.
- G1-G8 hold in both arms (criterion 5; H-T99b: no number is read otherwise).
- The regime carries the prompt sha, the stress sha (= the pinned file's), ``k == 3`` and the arm sizes (90, 40).
- Arm S's yield is re-judged here, independently of the harness, from every arm-S pass-1 ``tool`` commit with the
  unpatched ``live.classify`` (the notation of the criteria), and must equal the report's.
- **No threshold on rescue** (H-T99b): only the reading rule's shape is checked.

The surface assumed is stated in ``tests/unit/test_t0099_report.py``. Regime: the T-0093 umbrella regime at 4b0bd92
(lean; vLLM v0.29.0; NVFP4 rev 2ea8372; template ``36e3a42e``; tokenizer ``cc8d3a0c``; CL 256, 48 steps; thinking on;
prefix caching on), ``native:bbfc53ccdd80:dry_run``, repair on, deadline 3000 ms; typed turns, fake STT/audio, fake TTS
300 ms/word, fixture frames; concurrency 1; no seed; arm S 30 x k = 3, arm F 40.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import live
from tests.unit.test_t0099_report import GATE_KEYS, REPO, harness

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0099_RUN_DIR"
STRESS = REPO / "tests" / "eval" / "toolcall_stress" / "cases.jsonl"


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(
            f"{ENV} is not set: no T-0099 run directory (logs/t0099-repair-live-<sha>/) to read"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    for name in ("turns.jsonl", "manifest.json", "report.json", "report.md"):
        assert (path / name).is_file(), f"criterion 9: the run directory has no {name}"
    assert any((path / "sessions").glob("*.jsonl")), "criterion 9: no session log"
    return path


@pytest.fixture(scope="module")
def report(run_dir: Path) -> dict[str, Any]:
    rebuilt = harness().build_report(run_dir)
    assert isinstance(rebuilt, dict)
    return rebuilt


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _resolved(path: Any, base: Path) -> Path | None:
    """``path`` as an absolute, resolved directory; a relative one is read against ``base`` (the repo root, where the
    locked script runs the harness and where ``T0099_RUN_DIR`` is resolved)."""
    if not isinstance(path, str) or not path:
        return None
    p = Path(path)
    return (p if p.is_absolute() else base / p).resolve()


def report_disagreements(
    stored: dict[str, Any], rebuilt: dict[str, Any], base: Path = REPO
) -> list[str]:
    """The top-level keys on which the stored ``report.json`` and the re-derived report differ. Every key is compared
    by value after a JSON round trip, except ``run_dir``, which is the same directory spelled relative (as the locked
    script passed it) or absolute (as ``run_dir`` here is): it is compared as a resolved path, and a missing or
    unresolvable one still differs."""
    a, b = json.loads(json.dumps(stored)), json.loads(json.dumps(rebuilt))
    out: list[str] = []
    for key in sorted(set(a) | set(b)):
        if key == "run_dir":
            ra, rb = _resolved(a.get(key), base), _resolved(b.get(key), base)
            if ra is None or rb is None or ra != rb:
                out.append(key)
        elif key not in a or key not in b or a[key] != b[key]:
            out.append(key)
    return out


def test_stored_report_agrees_with_the_rederived_one(run_dir: Path, report: dict[str, Any]) -> None:
    stored = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    differ = report_disagreements(stored, report)
    assert differ == [], f"report.json differs from build_report(run_dir) on {differ}"


def test_the_run_is_not_void(report: dict[str, Any]) -> None:
    assert report["void"] == [], f"the run is void: {report['void']}"


def test_hard_gates_hold_in_both_arms(report: dict[str, Any]) -> None:
    g = harness().gates(report)
    for arm in ("S", "F"):
        assert set(g[arm]) == GATE_KEYS
        failing = sorted(k for k, v in g[arm].items() if v is not True)
        assert failing == [], f"arm {arm}: gates failing {failing} (H-T99b: no number is read)"


def test_regime_block(run_dir: Path, report: dict[str, Any]) -> None:
    regime = report["regime"]
    prompt = str(regime.get("prompt_sha256"))
    assert len(prompt) == 64 and all(c in "0123456789abcdef" for c in prompt), prompt
    assert regime["stress_sha256"] == hashlib.sha256(STRESS.read_bytes()).hexdigest()
    assert regime["k"] == 3
    assert regime["arm_sizes"] == {"S": 90, "F": 40}
    assert str(regime["tools_sha256"]).startswith("bbfc53ccdd80")
    assert regime.get("commit")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert [c["arm"] for c in cases] == ["S"] * 90 + ["F"] * 40, "criterion 2: arm S then arm F"
    stress_ids = [r["id"] for r in _jsonl(STRESS)]
    assert [c["case_id"] for c in cases[:90]] == stress_ids * 3, (
        "criterion 2: repetition-major, file order"
    )
    assert all(c["forced"] is False for c in cases[:90]) and all(
        c["forced"] is True for c in cases[90:]
    )


def test_arm_s_yield_rejudged_independently(run_dir: Path, report: dict[str, Any]) -> None:
    """Every arm-S pass-1 ``tool`` commit (segment ``tool`` from the turn's ``generate`` request) re-judged with the
    unpatched ``live.classify``; invalid = ``repairable``; ``refused`` (nameless) drafts are counted apart and are not in
    the denominator (ruling Q5)."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    invalid = routed = refused = 0
    for c in manifest["cases"]:
        if c["arm"] != "S":
            continue
        records = _jsonl(run_dir / "sessions" / c["session_log"])
        gen = {
            str(r["data"].get("request_id"))
            for r in records
            if r.get("kind") == "llm.request" and (r.get("data") or {}).get("phase") == "generate"
        }
        commits = [
            r["data"]["text"]
            for r in records
            if r.get("kind") == "llm.commit"
            and r["data"].get("segment") == "tool"
            and str(r["data"].get("request_id")) in gen
        ]
        if commits:
            status = live.classify(commits[-1]).status
            if status == "refused":
                refused += 1  # ruling Q5: counted on their own, never in the yield denominator
                continue
            routed += 1
            invalid += status == "repairable"
    assert (report["S"]["yield"]["k"], report["S"]["yield"]["n"]) == (invalid, routed)
    assert report["S"]["refused_drafts"] == refused


def test_rescue_reading_follows_the_rule_no_threshold(report: dict[str, Any]) -> None:
    """H-T99b: rescue is read only at >= 10 gt-bearing repairs; no bar on its value."""
    r = report["S"]["rescue"]
    if r["reading"] == "read":
        lo, hi = r["wilson95"]
        assert 0.0 <= lo <= hi <= 1.0
    else:
        assert str(r["reading"]).startswith("insufficient yield (n = "), r
        assert r["wilson95"] is None
