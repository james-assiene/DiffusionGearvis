"""T-0127 criterion 9: ``scripts/replay_ui_gate.py <session.jsonl>``, the CPU pre-desk check, on the committed fixture.

The script gates each turn's ``ui`` commits in order and prints, per turn, ``turn_id``, ``complete``, ``lost``, the
unreachable ids and the repairs by rule; it exits non-zero if any turn's last call is not complete. On the fixture it
reports 5/6 complete (t3 incomplete: its two Lists and two Texts lost); with CC disabled (= BASE behaviour) 0/6.

Readings where the task is open (named in the test-author report): the output format is not specified, so these rows
read only (i) a ``k/n`` complete count printed somewhere in stdout, (ii) t3's lost ids printed in the output, and
(iii) the exit status. "Non-zero only because of t3" is checked by replaying the fixture without t3's records (exit
0, 5/5). "CC disabled" is the validator proxy of ``tests/unit/test_ui_gate_desk_20261020.py``, applied in-process
(``runpy``), since a subprocess cannot be patched.

Regime: CPU only, no GPU, no network; read-only on the fixture; deterministic, n = 1.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_ui_gate_desk_20261020 import disable_cc

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "replay_ui_gate.py"
FIXTURE = REPO / "tests" / "fixtures" / "a2ui" / "desk_20261020_card_children.jsonl"
T3_LOST = ("h2", "inglist", "h2b", "steplist")


def require_script() -> None:
    if not SCRIPT.exists():
        raise NotImplementedError(
            "T-0127 criterion 9: scripts/replay_ui_gate.py does not exist yet"
        )


def in_process(
    path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str]:
    require_script()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(path)])
    code = 0
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    return code, capsys.readouterr().out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replay_reports_five_of_six_and_fails_only_on_t3(tmp_path: Path) -> None:
    """Criterion 9 on the fixture, as the CLI agents run: 5/6 complete, non-zero exit, t3's lost ids printed; the
    fixture is untouched. Without t3's records the same replay exits 0 with 5/5."""
    require_script()
    before = sha(FIXTURE)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(FIXTURE)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert sha(FIXTURE) == before, "criterion 9: the replay is read-only"
    assert proc.returncode != 0, proc.stdout
    assert "5/6" in proc.stdout, proc.stdout
    for cid in T3_LOST:
        assert cid in proc.stdout, f"t3's lost {cid} is not reported: {proc.stdout}"
    for turn in ("t3", "t5", "t6", "t8", "t9", "t10"):
        assert turn in proc.stdout, f"{turn} missing from the report"

    without_t3 = tmp_path / "without_t3.jsonl"
    kept = [
        line
        for line in FIXTURE.read_text().splitlines(keepends=True)
        if json.loads(line)["turn_id"] != "t3"
        and json.loads(line)["data"].get("msg", {}).get("surface_id") != "turn-t3"
    ]
    without_t3.write_text("".join(kept))
    proc2 = subprocess.run(
        [sys.executable, str(SCRIPT), str(without_t3)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert proc2.returncode == 0, proc2.stdout
    assert "5/5" in proc2.stdout, proc2.stdout


def test_replay_with_cc_disabled_reports_zero_of_six(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 9 "at BASE it reports 0/6": with CC disabled every turn is incomplete; non-zero exit."""
    disable_cc(monkeypatch)
    code, out = in_process(FIXTURE, monkeypatch, capsys)
    assert code != 0, out
    assert "0/6" in out, out
