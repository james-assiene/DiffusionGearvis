"""T-0093 criterion 2: the hard gates, read from a real run's report (``gpu``, ``eval``).

Run through ``/launch`` on the 5090 box, after the harness has written its run directory::

    T0093_RUN_DIR=logs/t0093-live-<sha> uv run pytest tests/eval/test_toolcall_live.py -q -m gpu -rA

Without ``T0093_RUN_DIR`` every row skips with that reason. The rows spend no GPU time themselves -- they read
``turns.jsonl``, the session logs and ``manifest.json`` -- but they are ``gpu`` because the data only exists after a
``/launch``-reviewed run, and they never run in the fast suite (``addopts`` deselects ``gpu``; ``tests/eval`` is
outside the fast selection).

The report is **re-derived from the raw logs** with the harness's ``build_report`` (the blast-radius note: one
reviewer re-derives the gates from the raw logs), and the stored ``report.json`` must agree with it, so a report that
was edited by hand, or written by another build of the harness, fails here.

The gates are the eight of ``tests/unit/test_t0093_report.GATES`` (criterion 2's five, the amendment's two and its
void conditions; approved unchanged at H-T93). The surface they assume is stated in that module's docstring.

Regime (state it beside every number this prints): umbrella regime of ``docs/tasks/T-0080d.md`` (lean; vLLM v0.29.0;
DiffusionGemma NVFP4 rev 2ea8372; template ``36e3a42e``; tokenizer ``cc8d3a0c``; tools_v1 ``bbfc53cc``; CL 256, 48
steps; thinking on; prefix caching on), engine only, ``tools = dry_run`` vs ``off``, typed input, fixture frames, no
audio, no seed, n = 1 per case per arm.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_t0093_report import GATES, REPO, harness

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0093_RUN_DIR"


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(
            f"{ENV} is not set: no T-0093 run directory (logs/t0093-live-<sha>/) to read the gates from"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    for name in ("turns.jsonl", "manifest.json", "report.json", "report.md"):
        assert (path / name).is_file(), f"T-0093 criterion 1: the run directory has no {name}"
    assert (path / "sessions").is_dir() and any((path / "sessions").glob("*.jsonl")), (
        "T-0093 criterion 1: the run directory holds no session log"
    )
    return path


@pytest.fixture(scope="module")
def report(run_dir: Path) -> dict[str, Any]:
    rebuilt = harness().build_report(run_dir)
    assert isinstance(rebuilt, dict)
    return rebuilt


def test_the_run_is_not_void(report: dict[str, Any]) -> None:
    """Amendment: every tools-on line reads ``native:bbfc53ccdd80:dry_run``; a run with repair off is invalid."""
    assert report["void"] == [], f"the run is void: {report['void']}"


def test_no_integrity_void(report: dict[str, Any]) -> None:
    """J14 and J19: no ``integrity:`` void -- every case has one line, every log is complete, no case errored, timed
    out, failed, ran with an unfinished prefill or with more than one request in flight."""
    integrity = [v for v in report["void"] if str(v).startswith("integrity:")]
    assert integrity == [], f"integrity voids: {integrity}"


def test_five_barge_ins_were_delivered(report: dict[str, Any]) -> None:
    """Criterion 2 and J18: exactly the five scripted barge-ins, delivered on routed tool turns."""
    assert report["barge_in"]["n"] == 5, f"barge-ins delivered: {report['barge_in']}"


@pytest.mark.parametrize("gate", sorted(GATES))
def test_hard_gate(report: dict[str, Any], gate: str) -> None:
    """Criterion 2 and the amendment, one row per gate, re-derived from the raw logs."""
    g = harness().gates(report)
    assert set(g) == GATES, (
        f"gates: extra {sorted(set(g) - GATES)}, missing {sorted(GATES - set(g))}"
    )
    shown = {
        k: report.get(k)
        for k in ("void", "prefix_hit", "first_canvas_ms", "false_tool", "stale_terminals")
    }
    assert g[gate] is True, (
        f"T-0093 hard gate {gate!r} fails; report: {json.dumps(shown, default=str)}"
    )


def test_the_stored_report_agrees_with_the_logs(run_dir: Path, report: dict[str, Any]) -> None:
    """The ``report.json`` the human reads is the one the raw logs give."""
    stored = json.loads((run_dir / "report.json").read_text())
    h = harness()
    assert h.gates(stored) == h.gates(report), (
        "report.json's gates differ from the ones re-derived from the logs"
    )
    for key in (
        "route_recall",
        "draft_valid",
        "draft_exact",
        "spoken_prefix",
        "false_tool",
        "repairs",
    ):
        assert stored.get(key) == report.get(key), f"report.json's {key!r} differs from the logs"


def test_the_run_is_named_for_the_commit_every_turn_ran_at(run_dir: Path) -> None:
    """Criterion 1: ``logs/t0093-live-<sha>/``; every turn line's ``regime.git_commit`` is that commit."""
    prefix = "t0093-live-"
    assert run_dir.name.startswith(prefix), run_dir.name
    sha = run_dir.name[len(prefix) :]
    assert len(sha) >= 7, run_dir.name
    lines = [json.loads(x) for x in (run_dir / "turns.jsonl").read_text().splitlines() if x.strip()]
    assert lines, "turns.jsonl is empty"
    other = sorted(
        {r["regime"]["git_commit"] for r in lines if not r["regime"]["git_commit"].startswith(sha)}
    )
    assert other == [], f"turn lines written at another commit: {other}"


def test_every_case_ran_once_per_arm(run_dir: Path) -> None:
    """Criterion 1: each case runs once with tools off and once with tools on (the dev split: 40 tool + 8 no-tool,
    the 7 Q12 cases of J1 and the 5 plain questions of J10)."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    per_arm = Counter((c["case_id"], c["arm"]) for c in manifest["cases"])
    cases = {case for case, _ in per_arm}
    bad = sorted(c for c in cases if (per_arm[(c, "on")], per_arm[(c, "off")]) != (1, 1))
    assert bad == [], f"cases not run exactly once per arm: {bad}"
    kinds = Counter(c["kind"] for c in manifest["cases"] if c["arm"] == "on")
    assert kinds["tool"] == 40 and kinds["no_tool"] == 8 and kinds["plain"] == 5, kinds
    # J1: the Q12 cases are the 7 of surface_placement_cases.jsonl
    q12_file = REPO / "tests" / "eval" / "surface_placement_cases.jsonl"
    q12_ids = {json.loads(x)["id"] for x in q12_file.read_text().splitlines() if x.strip()}
    ran = {c["case_id"] for c in manifest["cases"] if c["kind"] == "q12" and c["arm"] == "on"}
    assert kinds["q12"] == 7 and ran == q12_ids, (
        f"J1: Q12 cases {sorted(ran)}, expected {sorted(q12_ids)}"
    )


def test_the_run_is_scored_under_h_t96(report: dict[str, Any]) -> None:
    """T-0096 (H-T96): a run after T-0096 is scored by B11 as written -- the manifest says so and A is the registered
    32 (never the engine's ``block_size`` label). A run without it reads "predates H-T96" and cannot pass."""
    assert report.get("prefix_rule") == "H-T96", f"prefix_rule {report.get('prefix_rule')!r}"
    assert report.get("prefix_block_tokens") == 32, (
        f"prefix_block_tokens {report.get('prefix_block_tokens')!r}"
    )
