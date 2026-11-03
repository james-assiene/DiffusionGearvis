"""T-0079 -- the H2 speed-aware BFCL re-selection tool (``scripts/cd_reselect.py``), CPU only.

Every row runs the tool as a subprocess (``sys.executable scripts/cd_reselect.py ...``, cwd = the repo) on an
evidence root built in ``tmp_path``:

- the campaign: ``tests/cd/fixtures/t0071b_b2_campaign.json`` (the recorded b2 campaign's fields, verbatim; see
  the MANIFEST) or a copy of it whose ``perstep_mar`` / ``perstep_mar_qargmax`` cells are replaced by a
  hand-built grid;
- the pre-registered frozen files the re-selection supersedes (``t0071b_b2_frozen/``, byte for byte);
- per cell, a synthetic artifact at the cell's recorded relative path with ``n`` rows whose per-row step counts
  average to exactly the cell's ``mean_steps``, and one ``ASTRA_CD`` record per row in a synthetic engine log,
  each carrying a per-cell constant ``mean_accepted_per_step`` = ``ACCEPTED_BASE[arm] + <cell index>``. The
  step-weighted mean over one cell's records is therefore that constant exactly, which tells the row both
  *which* cell the tool chose and that its stats came from the engine log of *that* cell.

Expected knobs come from H2's text (docs/tasks/T-0071b.md, "H2 -- human decision"), never from the tool:
perstep_mar -> eb 1.0 / ct 0.005 (0.52, 6.88 steps); perstep_mar_qargmax -> eb 0.3 / ct 0.005 (0.52, 7.99 steps).

The tool does not exist before T-0079's implementation; ``_run`` asserts the file exists *before* invoking it,
so no row can pass on Python's own "can't open file" exit status 2 (which would read as a refusal).

Regime of the recorded inputs: launch2 worktree at 067ee04, dev cut BFCL 130-229 (n 100), thinking off,
max_tokens 256, JSON whitespace, CL 256, 48 steps, NVFP4 rev 2ea8372, vLLM v0.29.0, no seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.eval.cd import calibration

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts" / "cd_reselect.py"
FIXTURES = REPO / "tests" / "cd" / "fixtures"
CAMPAIGN_FIXTURE = FIXTURES / "t0071b_b2_campaign.json"
FROZEN_FIXTURES = FIXTURES / "t0071b_b2_frozen"

EVIDENCE_DIR = "logs/t0071b-calib-bfcl-067ee04"
CAMPAIGN_REL = f"{EVIDENCE_DIR}/campaign-bfcl.json"
ENGINE_REL = f"{EVIDENCE_DIR}/engine.log"
OLD_FROZEN = ("calibration-bfcl.json", "calibration-bfcl-perstep_mar_qargmax.json")

OUT_MAR = "calibration-bfcl.json"
OUT_QARGMAX = "calibration-bfcl-perstep_mar_qargmax.json"
DECISION_DATE = "2026-10-08"

#: mean_accepted_per_step written for cell i of each arm's grid: base + i (exact in float64).
ACCEPTED_BASE = {"perstep_mar": 100.0, "perstep_mar_qargmax": 200.0}

SHA_OLD_MAR = "d70f588801f85338caa2b70a4cf25949a11546337ee63c4cfc5ac17cbf6424a7"
SHA_OLD_QARGMAX = "42c689a3223d20bcd2e7ab8bf12ae568bce74dcfbcb20a6833955767743313c3"
EXPECTED_COMMIT = "067ee043f82dd810f1c590b9be94f802852578b8"


# --------------------------------------------------------------------------- evidence builders


def _fixture_campaign() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(CAMPAIGN_FIXTURE.read_text())
    return data


def _cell(
    idx: int,
    eb: float,
    ct: float,
    score: float,
    steps: float,
    *,
    n: int = 100,
    eligible: bool = True,
    stopped: str | None = None,
    arm: str = "perstep_mar",
) -> dict[str, Any]:
    """A synthetic grid cell with every field a recorded cell has."""
    return {
        "entropy_bound": eb,
        "confidence_threshold": ct,
        "score": score,
        "n": n,
        "eligible": eligible,
        "stopped": stopped,
        "mean_steps": steps,
        "mean_steps_per_canvas": steps,
        "errors": {},
        "wall_s": 600.0,
        "artifact": f"{EVIDENCE_DIR}/syn-{arm}-{idx:02d}.jsonl",
        "run_id": f"syn{arm[-4:]}{idx:02d}",
    }


def _record_line(tag: str, arm: str, steps: int, accepted: float) -> str:
    rec = {
        "record_tag": tag,
        "arm": arm,
        "canvas_idx": 0,
        "committed_ids": [1],
        "state_set_after": [0],
        "in_final": True,
        "steps": steps,
        "bucket": 64,
        "stop_pos": 0,
        "error": None,
        "mean_accepted_per_step": accepted,
        "shadow_argmax_ids": None,
        "argmax_in_language": None,
        "map_equals_argmax": None,
    }
    return "(EngineCore pid=1) INFO 10-07 17:58:32 [record.py:100] ASTRA_CD " + json.dumps(rec)


def _build_evidence(root: Path, campaign: dict[str, Any] | None = None) -> Path:
    """An evidence root shaped like the launch worktree; returns it. ``campaign=None`` copies the fixture's bytes."""
    ev = root / "evidence"
    evdir = ev / EVIDENCE_DIR
    evdir.mkdir(parents=True)
    if campaign is None:
        shutil.copyfile(CAMPAIGN_FIXTURE, ev / CAMPAIGN_REL)
        campaign = _fixture_campaign()
    else:
        (ev / CAMPAIGN_REL).write_text(json.dumps(campaign, indent=1) + "\n")
    for name in OLD_FROZEN:
        shutil.copyfile(FROZEN_FIXTURES / name, evdir / name)
    log_lines: list[str] = ["(APIServer pid=1) INFO boot line without a record"]
    for arm, grid in campaign["grids"].items():
        for i, cell in enumerate(grid["cells"]):
            n = int(cell["n"])
            total = round(float(cell["mean_steps"]) * n)
            assert total / n == float(cell["mean_steps"]), (
                f"fixture: {cell['mean_steps']} not k/{n}"
            )
            q, r = divmod(total, n)
            accepted = ACCEPTED_BASE[arm] + i
            lines = [
                json.dumps({"kind": "regime", "task": "bfcl", "cut": "dev", "arm": arm, "n": n})
            ]
            for j in range(n):
                s = q + 1 if j < r else q
                tag = f"t0070.{cell['run_id']}.{j}"
                lines.append(
                    json.dumps(
                        {"id": f"bfcl-dev-{130 + j}", "steps": [s], "knobs": {"record_tag": tag}}
                    )
                )
                log_lines.append(_record_line(tag, arm, s, accepted))
            (ev / cell["artifact"]).write_text("\n".join(lines) + "\n")
    (ev / ENGINE_REL).write_text("\n".join(log_lines) + "\n")
    return ev


def _with_grid(arm: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    campaign = copy.deepcopy(_fixture_campaign())
    campaign["grids"][arm]["cells"] = cells
    return campaign


def _run(
    ev: Path,
    out: Path,
    *,
    task: str = "bfcl",
    campaign: Path | None = None,
    decision_date: str = DECISION_DATE,
) -> subprocess.CompletedProcess[str]:
    assert TOOL.is_file(), f"{TOOL.relative_to(REPO)} does not exist yet (T-0079 criterion 1)"
    cmd = [
        sys.executable,
        str(TOOL),
        "--evidence-root",
        str(ev),
        "--campaign",
        str(campaign if campaign is not None else ev / CAMPAIGN_REL),
        "--engine-log",
        str(ev / ENGINE_REL),
        "--out-dir",
        str(out),
        "--task",
        task,
        "--decision-date",
        decision_date,
    ]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)


def _ok(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
    )


def _refused(proc: subprocess.CompletedProcess[str], out: Path) -> None:
    assert proc.returncode == 2, (
        f"expected a refusal (exit 2), got {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    written = sorted(p.name for p in out.rglob("*") if p.is_file()) if out.exists() else []
    assert written == [], f"a refusal wrote {written}"


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_bytes())
    return data


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _knobs(data: dict[str, Any]) -> tuple[float, float]:
    return float(data["entropy_bound"]), float(data["confidence_threshold"])


# --------------------------------------------------------------------------- criterion 2


def test_h2_picks_on_recorded_grid(tmp_path: Path) -> None:
    """H2's text: perstep_mar eb 1.0 / ct 0.005 (exact 0.52, 6.9 steps); qargmax eb 0.3 / ct 0.005 (0.52, 8.0)."""
    ev = _build_evidence(tmp_path)
    out = tmp_path / "out"
    _ok(_run(ev, out))
    mar = _load(out / OUT_MAR)
    qam = _load(out / OUT_QARGMAX)
    assert mar["arm"] == "perstep_mar"
    assert _knobs(mar) == (1.0, 0.005)
    assert qam["arm"] == "perstep_mar_qargmax"
    assert _knobs(qam) == (0.3, 0.005)
    # chosen_stats: score / steps from the recorded grid (H2: 0.52 at 6.88 and 7.99); accepted per step from
    # the synthetic engine log of the chosen cell (index 17 of perstep_mar, 14 of qargmax: 100 + 17, 200 + 14).
    for data, steps, accepted in ((mar, 6.88, 117.0), (qam, 7.99, 214.0)):
        cs = data["chosen_stats"]
        assert cs["score"] == 0.52
        assert cs["n"] == 100
        assert cs["mean_steps_per_canvas"] == steps
        assert cs["stopped"] is None
        assert cs["mean_accepted_per_step"] == accepted


# --------------------------------------------------------------------------- criterion 1: the rule


# Boundary rows. The task file's example (best 0.56, probe 0.51) is NOT a float trap for ``best - 0.05``:
# in float64 0.56 - 0.05 == 0.51 exactly (checked: ``repr(0.56 - 0.05) == '0.51'``); it traps only the
# ``best - score > 0.05`` formulation. The pairs below trap each naive formulation at least once:
#   (0.54, 0.49): 0.54 - 0.05 = 0.49000000000000005 > 0.49, and 0.54 - 0.49 > 0.05;
#   (0.46, 0.41): 0.46 - 0.05 = 0.41000000000000003 > 0.41, 0.41 + 0.05 < 0.46, and 0.46 - 0.41 > 0.05;
#   (0.56, 0.51): 0.56 - 0.51 = 0.05000000000000004 > 0.05 (the task file's pair).
# Exactly, every probe is best - 1/20 (Fraction(k - 5, 100) >= Fraction(k, 100) - Fraction(1, 20)), so it qualifies.
BOUNDARY_PAIRS = [(0.54, 0.49), (0.46, 0.41), (0.56, 0.51)]


def _boundary_grid(best: float, probe_score: float) -> list[dict[str, Any]]:
    """Best (17 steps); a cell 0.03 below best (9 steps); the probe (6 steps, the fewest); a low cell."""
    return [
        _cell(0, 0.1, 0.0005, best, 17.0),
        _cell(1, 0.03, 0.005, round(best - 0.03, 2), 9.0),
        _cell(2, 0.3, 0.005, probe_score, 6.0),
        _cell(3, 0.003, 5e-05, round(best - 0.10, 2), 25.0),
    ]


@pytest.mark.parametrize(("best", "probe"), BOUNDARY_PAIRS, ids=lambda v: f"{v}")
def test_boundary_inclusive_float_trap(tmp_path: Path, best: float, probe: float) -> None:
    """Adversarial (criterion 1, "computed exactly", "inclusive"): the probe sits exactly at best - 0.05 and has
    the fewest steps, so it must be chosen; a naive float comparison drops it."""
    # premise of the row (not a check of the tool): at least one naive float form drops the probe
    assert (best - 0.05 > probe) or (probe + 0.05 < best) or (best - probe > 0.05)
    ev = _build_evidence(tmp_path, _with_grid("perstep_mar", _boundary_grid(best, probe)))
    out = tmp_path / "out"
    _ok(_run(ev, out))
    mar = _load(out / OUT_MAR)
    assert _knobs(mar) == (0.3, 0.005)
    assert mar["chosen_stats"]["mean_accepted_per_step"] == 102.0  # cell index 2
    assert mar["chosen_stats"]["score"] == probe


@pytest.mark.parametrize(("best", "probe"), BOUNDARY_PAIRS, ids=lambda v: f"{v}")
def test_just_below_excluded(tmp_path: Path, best: float, probe: float) -> None:
    """The same cell one count lower (best - 0.06) does not qualify; the cell 0.03 below best at 9 steps is the
    fewest that does (a guard that is too wide, e.g. a 0.06 margin or a loose epsilon, picks the probe)."""
    below = round(probe - 0.01, 2)
    ev = _build_evidence(tmp_path, _with_grid("perstep_mar", _boundary_grid(best, below)))
    out = tmp_path / "out"
    _ok(_run(ev, out))
    mar = _load(out / OUT_MAR)
    assert _knobs(mar) == (0.03, 0.005)
    assert mar["chosen_stats"]["mean_accepted_per_step"] == 101.0  # cell index 1


@pytest.mark.parametrize(
    "disqualifier",
    [
        {"eligible": False},
        {"stopped": "e: schema_valid 3/20 below the floor"},
        {"n": 60, "score": 0.95},
        {"n": 120, "score": 0.95},
    ],
    ids=["ineligible", "stopped", "n_below_cut", "n_above_cut"],
)
def test_ineligible_never_chosen_nor_best(tmp_path: Path, disqualifier: dict[str, Any]) -> None:
    """A pool-excluded cell with the top score (0.70) and the fewest steps (3) neither wins nor sets "best".
    Were it best, the threshold would be 0.65 and no eligible cell would qualify; were it chosen, eb 1.0 / ct 5e-05.
    The pool's best is 0.56, so 0.51 qualifies and wins at 6 steps.
    The n rows use 0.95 (57 answers at n 60, 114 at n 120: whole counts, so the integer-count refusal cannot
    mask them). Pooled, they would be best however the count is read (0.95, or 57/100 = 0.57) and win on steps,
    so a pool that drops the ``n == cut`` condition fails this row (reviewer F1: the earlier 0.70 at n 50 read as
    35/100 could neither be best nor qualify, so that mutant survived)."""
    x = _cell(0, 1.0, 5e-05, 0.70, 3.0)
    x.update(disqualifier)
    cells = [
        x,
        _cell(1, 0.1, 0.0005, 0.56, 17.0),
        _cell(2, 0.3, 0.005, 0.51, 6.0),
        _cell(3, 0.03, 0.005, 0.53, 9.0),
    ]
    ev = _build_evidence(tmp_path, _with_grid("perstep_mar", cells))
    out = tmp_path / "out"
    _ok(_run(ev, out))
    mar = _load(out / OUT_MAR)
    assert _knobs(mar) == (0.3, 0.005)
    assert mar["chosen_stats"]["mean_accepted_per_step"] == 102.0  # cell index 2


def test_tie_on_steps_larger_eb(tmp_path: Path) -> None:
    """Two qualifying cells at 7.0 steps: eb 1.0 wins over eb 0.3 although it has the lower score (0.52 < 0.53)
    and the smaller ct, so neither a score nor a ct tie-break reproduces the expected choice."""
    cells = [
        _cell(0, 0.1, 0.0005, 0.56, 17.0),
        _cell(1, 0.3, 0.005, 0.53, 7.0),
        _cell(2, 1.0, 0.0005, 0.52, 7.0),
        _cell(3, 0.01, 0.005, 0.54, 9.0),
    ]
    ev = _build_evidence(tmp_path, _with_grid("perstep_mar", cells))
    out = tmp_path / "out"
    _ok(_run(ev, out))
    mar = _load(out / OUT_MAR)
    assert _knobs(mar) == (1.0, 0.0005)
    assert mar["chosen_stats"]["mean_accepted_per_step"] == 102.0  # cell index 2


@pytest.mark.parametrize("arm", ["perstep_mar", "perstep_mar_qargmax"])
def test_full_tie_refuses(tmp_path: Path, arm: str) -> None:
    """Same steps and same eb (ct differs): no third key, exit 2, and nothing written -- not even the
    other arm's file (the tie sits in either the first or the second arm)."""
    cells = [
        _cell(0, 0.1, 0.0005, 0.56, 17.0, arm=arm),
        _cell(1, 1.0, 0.005, 0.53, 7.0, arm=arm),
        _cell(2, 1.0, 0.0005, 0.52, 7.0, arm=arm),
    ]
    ev = _build_evidence(tmp_path, _with_grid(arm, cells))
    out = tmp_path / "out"
    _refused(_run(ev, out), out)


@pytest.mark.parametrize("arm", ["perstep_mar", "perstep_mar_qargmax"])
def test_score_not_integer_count_refuses(tmp_path: Path, arm: str) -> None:
    """0.515 at n 100 is 51.5 correct answers: not a count, so the exact comparison is undefined -> exit 2."""
    cells = [
        _cell(0, 0.1, 0.0005, 0.56, 17.0, arm=arm),
        _cell(1, 0.3, 0.005, 0.515, 6.0, arm=arm),
        _cell(2, 0.03, 0.005, 0.53, 9.0, arm=arm),
    ]
    ev = _build_evidence(tmp_path, _with_grid(arm, cells))
    out = tmp_path / "out"
    _refused(_run(ev, out), out)


# --------------------------------------------------------------------------- criterion 3: frozen files


def test_frozen_file_loads_only_for_its_arm(tmp_path: Path) -> None:
    ev = _build_evidence(tmp_path)
    out = tmp_path / "out"
    _ok(_run(ev, out))
    mar, qam = out / OUT_MAR, out / OUT_QARGMAX
    # names follow calibration.default_path
    assert calibration.default_path(out, "bfcl", "perstep_mar") == mar
    assert calibration.default_path(out, "bfcl", "perstep_mar_qargmax") == qam
    for arm in ("perstep_mar", "perstep_mar_sc"):
        knobs, sha = calibration.load_frozen(mar, task="bfcl", arm=arm)
        assert knobs == {"entropy_bound": 1.0, "confidence_threshold": 0.005}
        assert sha == _sha(mar)
    for arm in ("perstep_mar_qargmax", "perstep_mar_qargmax_sc"):
        knobs, sha = calibration.load_frozen(qam, task="bfcl", arm=arm)
        assert knobs == {"entropy_bound": 0.3, "confidence_threshold": 0.005}
        assert sha == _sha(qam)
    for path, other in (
        (mar, "perstep_mar_qargmax"),
        (mar, "perstep_mar_qargmax_sc"),
        (qam, "perstep_mar"),
        (qam, "perstep_mar_sc"),
    ):
        with pytest.raises(ValueError):
            calibration.load_frozen(path, task="bfcl", arm=other)
    with pytest.raises(ValueError):
        calibration.load_frozen(mar, task="sudoku", arm="perstep_mar")


def test_frozen_file_keeps_t0071b_fields(tmp_path: Path) -> None:
    """Every field of the T-0071b frozen file survives: task, cut, arm, knobs, expected_commit, limit,
    stock_reference (from the campaign), chosen_stats; plus grid_results = that arm's recorded cells."""
    ev = _build_evidence(tmp_path)
    out = tmp_path / "out"
    _ok(_run(ev, out))
    campaign = _fixture_campaign()
    for name, arm in ((OUT_MAR, "perstep_mar"), (OUT_QARGMAX, "perstep_mar_qargmax")):
        data = _load(out / name)
        old = _load(FROZEN_FIXTURES / name)
        missing = sorted(set(old) - set(data))
        assert missing == [], f"{name} lost T-0071b fields {missing}"
        assert data["task"] == "bfcl"
        assert data["cut"] == "dev"
        assert data["arm"] == arm
        assert data["expected_commit"] == EXPECTED_COMMIT
        assert data["limit"] is None
        assert data["stock_reference"] == campaign["frozen_by_arm"][arm]["stock_reference"]
        assert data["stock_reference"] == old["stock_reference"]
        assert data["grid_results"] == campaign["grids"][arm]["cells"]
        assert isinstance(data["content_sha256"], str) and len(data["content_sha256"]) == 64


def test_provenance_fields(tmp_path: Path) -> None:
    ev = _build_evidence(tmp_path)
    out = tmp_path / "out"
    _ok(_run(ev, out))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    input_sha = _sha(ev / CAMPAIGN_REL)
    old_rule = _load(FROZEN_FIXTURES / OUT_MAR)["rule"]
    superseded = {
        OUT_MAR: (SHA_OLD_MAR, 0.1, 0.0005, 0.56, 17.77),
        OUT_QARGMAX: (SHA_OLD_QARGMAX, 1.0, 0.0005, 0.56, 17.91),
    }
    for name, (old_sha, eb, ct, score, steps) in superseded.items():
        data = _load(out / name)
        assert data["decision"] == {
            "by": "human",
            "date": "2026-10-08",
            "ref": "docs/tasks/T-0071b.md#H2",
        }
        rule = data["rule"]
        assert isinstance(rule, str) and "0.05" in rule, rule
        assert rule != old_rule, "the pre-registered rule text was kept"
        src = data["source_campaign"]
        assert src["sha256"] == input_sha
        assert Path(src["path"]).name == "campaign-bfcl.json"
        sup = data["superseded"]
        assert sup["sha256"] == old_sha
        assert Path(sup["path"]).name == name
        assert float(sup["entropy_bound"]) == eb
        assert float(sup["confidence_threshold"]) == ct
        assert float(sup["score"]) == score
        assert float(sup["mean_steps"]) == steps
        by = json.dumps(data["reselected_by"])
        assert "cd_reselect.py" in by, by
        assert head in by, f"reselected_by does not name HEAD {head}: {by}"


# --------------------------------------------------------------------------- criteria 4 and 6


@pytest.mark.parametrize("name", OLD_FROZEN)
def test_superseded_sha_mismatch_refuses(tmp_path: Path, name: str) -> None:
    """Criterion 3: ``superseded`` names the pre-registered file the campaign recorded (``frozen_by_arm[arm].sha256``).
    If the file in the evidence root is not those bytes, the provenance would be a lie: exit 2, nothing written
    (either arm, so the second arm's mismatch also leaves no first-arm file)."""
    ev = _build_evidence(tmp_path)
    _ok(_run(ev, tmp_path / "control"))  # the untampered evidence is accepted
    old = ev / EVIDENCE_DIR / name
    os.chmod(old, stat.S_IRUSR | stat.S_IWUSR)
    old.write_bytes(old.read_bytes() + b"\n")
    out = tmp_path / "out"
    _refused(_run(ev, out), out)


def test_decision_date_is_the_argument(tmp_path: Path) -> None:
    """Criterion 4: the bytes are a function of the inputs, the decision date and the commit. Another
    ``--decision-date`` changes ``decision.date`` (and so the bytes) and nothing else: same knobs and stats."""
    ev = _build_evidence(tmp_path)
    a, b = tmp_path / "out-a", tmp_path / "out-b"
    _ok(_run(ev, a))
    _ok(_run(ev, b, decision_date="2026-10-09"))
    for name in (OUT_MAR, OUT_QARGMAX):
        da, db = _load(a / name), _load(b / name)
        assert da["decision"]["date"] == "2026-10-08"
        assert db["decision"]["date"] == "2026-10-09"
        assert (a / name).read_bytes() != (b / name).read_bytes()
        strip = ("decision", "content_sha256")
        assert {k: v for k, v in da.items() if k not in strip} == {
            k: v for k, v in db.items() if k not in strip
        }


def test_deterministic_bytes(tmp_path: Path) -> None:
    ev = _build_evidence(tmp_path)
    a, b = tmp_path / "out-a", tmp_path / "out-b"
    _ok(_run(ev, a))
    _ok(_run(ev, b))
    for name in (OUT_MAR, OUT_QARGMAX):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_inputs_untouched(tmp_path: Path) -> None:
    """Every input is made read-only (0444) before the run, so an open for writing fails; bytes are
    compared before and after."""
    ev = _build_evidence(tmp_path)
    inputs = sorted(p for p in ev.rglob("*") if p.is_file())
    assert len(inputs) == 2 + 2 + 36  # campaign, engine log, two old frozen files, 36 artifacts
    before = {p: _sha(p) for p in inputs}
    for p in inputs:
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        out = tmp_path / "out"
        _ok(_run(ev, out))
        after = {p: _sha(p) for p in inputs}
        assert after == before
        assert sorted(p for p in ev.rglob("*") if p.is_file()) == inputs, (
            "the tool wrote into the evidence"
        )
        assert _sha(ev / f"{EVIDENCE_DIR}/{OUT_MAR}") == SHA_OLD_MAR
        assert _sha(ev / f"{EVIDENCE_DIR}/{OUT_QARGMAX}") == SHA_OLD_QARGMAX
    finally:
        for p in inputs:
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def test_refuses_sudoku_existing_output_and_non_dev(tmp_path: Path) -> None:
    ev = _build_evidence(tmp_path)
    # control: the same inputs with nothing wrong succeed, so a refusal below is the refusal under test
    _ok(_run(ev, tmp_path / "control"))

    # --task sudoku (H2 keeps Sudoku): refused, with the BFCL campaign and with a Sudoku-labelled one
    out = tmp_path / "sudoku"
    _refused(_run(ev, out, task="sudoku"), out)
    sudoku = copy.deepcopy(_fixture_campaign())
    sudoku["task"] = "sudoku"
    sud_path = tmp_path / "campaign-sudoku.json"
    sud_path.write_text(json.dumps(sudoku, indent=1) + "\n")
    out = tmp_path / "sudoku2"
    _refused(_run(ev, out, task="sudoku", campaign=sud_path), out)

    # a campaign whose cut is not dev
    for cut in ("test", None):
        bad = copy.deepcopy(_fixture_campaign())
        bad["cut"] = cut
        bad_path = tmp_path / f"campaign-cut-{cut}.json"
        bad_path.write_text(json.dumps(bad, indent=1) + "\n")
        out = tmp_path / f"cut-{cut}"
        _refused(_run(ev, out, campaign=bad_path), out)


@pytest.mark.parametrize("existing", [OUT_MAR, OUT_QARGMAX])
def test_refuses_existing_output(tmp_path: Path, existing: str) -> None:
    """An existing output file (either arm's) is never overwritten, and the other arm's file is not written."""
    ev = _build_evidence(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = b'{"sentinel": true}\n'
    (out / existing).write_bytes(sentinel)
    proc = _run(ev, out)
    assert proc.returncode == 2, (
        f"exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert (out / existing).read_bytes() == sentinel
    assert sorted(p.name for p in out.iterdir()) == [existing]


# Criterion 5 (the docs/cd publication row) lives in tests/unit/test_cd_published_calibration.py: it needs
# no torch, and a pending marker under tests/cd is invisible to scripts/check_pending.py (tests/cd is
# collected only with ASTRA_CD_TESTS=1), which test_suite_budget.py::test_the_pending_report_is_complete rejects.
