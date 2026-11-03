"""T-0070 criterion 10: the calibration grid, the selection rule, the frozen ``calibration.json`` and the
test-cut runner's refusal to run ``perstep_mar``/``perstep_mar_sc`` without it.

Grid (SPEC §3.4): entropy_bound in {0.003, 0.01, 0.03, 0.1, 0.3, 1.0} x confidence_threshold in
{5e-5, 5e-4, 5e-3}, ``perstep_mar``, dev cuts only. Selection: maximise exact call (BFCL) / solved
(Sudoku); ties -> fewer mean steps, then the larger entropy_bound. Expected choices below are by hand.
Regime: CPU, fake server.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import Reply

EB = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CT = (5e-5, 5e-4, 5e-3)


def _choice(x: object) -> tuple[float, float]:
    return float(cs.field(x, "entropy_bound")), float(cs.field(x, "confidence_threshold"))


def test_grid_is_the_pre_registered_18_points() -> None:
    got = {
        (float(cs.field(p, "entropy_bound")), float(cs.field(p, "confidence_threshold")))
        for p in es.calibration().grid()
    }
    assert got == set(itertools.product(EB, CT))
    assert len(list(es.calibration().grid())) == 18


def _res(eb: float, ct: float, score: float, steps: float) -> dict[str, float]:
    return {"entropy_bound": eb, "confidence_threshold": ct, "score": score, "mean_steps": steps}


def test_select_maximises_the_score() -> None:
    r = [_res(0.1, 5e-4, 0.60, 10), _res(0.03, 5e-3, 0.62, 30), _res(1.0, 5e-5, 0.55, 5)]
    assert _choice(es.calibration().select(r)) == (0.03, 5e-3)


def test_select_breaks_score_ties_on_fewer_mean_steps() -> None:
    r = [_res(1.0, 5e-4, 0.60, 22.0), _res(0.01, 5e-3, 0.60, 21.5), _res(0.3, 5e-5, 0.50, 3.0)]
    assert _choice(es.calibration().select(r)) == (0.01, 5e-3)


def test_select_breaks_remaining_ties_on_the_larger_entropy_bound() -> None:
    r = [_res(0.03, 5e-4, 0.60, 20.0), _res(0.3, 5e-3, 0.60, 20.0), _res(0.1, 5e-5, 0.60, 20.0)]
    assert _choice(es.calibration().select(r)) == (0.3, 5e-3)


def test_freeze_writes_a_file_whose_sha_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    sha = es.calibration().freeze(
        {
            "entropy_bound": 0.1,
            "confidence_threshold": 5e-4,
            "task": "sudoku",
            "cut": "dev",
            "arm": "perstep_mar",
        },
        path,
    )
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    assert data["entropy_bound"] == 0.1 and data["confidence_threshold"] == 5e-4


@pytest.mark.parametrize("arm", ["perstep_mar", "perstep_mar_sc"])
def test_runner_refuses_mar_without_the_frozen_file(tmp_path: Path, arm: str) -> None:
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        try:
            res = es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, limit=2)
        except ValueError:
            res = None
        sent = len(srv.requests)
    assert sent == 0
    assert res is None or res.exit_code != 0


def test_runner_refuses_a_calibration_edited_after_freezing(tmp_path: Path) -> None:
    path = es.freeze_calibration(tmp_path, 0.03, 5e-4)
    data = json.loads(path.read_text())
    data["entropy_bound"] = 0.3
    path.write_text(json.dumps(data))
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        try:
            res = es.run(
                srv,
                tmp_path,
                task="sudoku",
                cut="test",
                arm="perstep_mar",
                limit=2,
                calibration=path,
            )
        except ValueError:
            res = None
        sent = len(srv.requests)
    assert sent == 0
    assert res is None or res.exit_code != 0


def test_runner_needs_no_calibration_for_the_other_arms(tmp_path: Path) -> None:
    """Companion: the refusal is not too wide."""
    cs.warm_sudoku("test")
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm="perstep_mf", limit=2)
    assert res.exit_code == 0, res.stopped


def test_driver_runs_the_grid_on_the_dev_cut_and_freezes_the_winner(tmp_path: Path) -> None:
    """The fake solves only under (0.03, 5e-4); every request is perstep_mar on a dev puzzle."""
    cs.warm_sudoku("dev")
    dev_by_id = es.sudoku_by_id("dev")
    test_prompts = set(es.prompt_to_id("sudoku", "test"))

    def decide(i: int, rid: str, body: dict) -> Reply | str:  # type: ignore[type-arg]
        x = body["vllm_xargs"]
        if (x.get("astra_cd_entropy_bound"), x.get("astra_cd_confidence_threshold")) == (
            0.03,
            5e-4,
        ):
            return "solved"
        return "wrong"

    repo, head = es.clean_repo(tmp_path)
    out = tmp_path / "artifacts"
    out.mkdir()
    with es.fake(es.sudoku_script("dev", decide), tmp_path) as srv:
        path = es.calibration().calibrate(
            task="sudoku",
            base_url=srv.url,
            engine_log=srv.engine_log,
            out_dir=out,
            repo=repo,
            expected_commit=head,
            limit=4,
        )
        bodies = list(srv.requests)
    assert len(bodies) == 18 * 4
    assert not {b["messages"][-1]["content"] for b in bodies} & test_prompts
    assert all(b["vllm_xargs"]["astra_cd_mode"] == "perstep_mar" for b in bodies)
    assert {
        (
            b["vllm_xargs"]["astra_cd_entropy_bound"],
            b["vllm_xargs"]["astra_cd_confidence_threshold"],
        )
        for b in bodies
    } == set(itertools.product(EB, CT))
    data = json.loads(Path(path).read_text())
    assert (data["entropy_bound"], data["confidence_threshold"]) == (0.03, 5e-4)
    assert len(dev_by_id) == 100


# ------------------------------------------------------------------ E11: calibration provenance


@pytest.mark.parametrize(
    "variant",
    [
        {"drop": ("task",)},
        {"drop": ("cut",)},
        {"drop": ("arm",)},
        {"cut": "test"},
        {"task": "bfcl"},
    ],
    ids=["no-task", "no-cut", "no-arm", "cut-test", "other-task"],
)
def test_e11_runner_refuses_a_calibration_without_dev_provenance(
    tmp_path: Path, variant: dict[str, object]
) -> None:
    path = es.freeze_calibration(tmp_path, **variant)  # type: ignore[arg-type]
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        try:
            res = es.run(
                srv,
                tmp_path,
                task="sudoku",
                cut="test",
                arm="perstep_mar",
                limit=2,
                calibration=path,
            )
        except ValueError:
            res = None
        sent = len(srv.requests)
    assert sent == 0
    assert res is None or res.exit_code != 0


@pytest.mark.parametrize("arm", ["perstep_mar", "perstep_mar_sc"])
def test_e11_complete_dev_calibration_is_accepted(tmp_path: Path, arm: str) -> None:
    """Companion: the provenance check is not too wide (Mar+SC reuses Mar's file)."""
    cs.warm_sudoku("test")
    path = es.freeze_calibration(tmp_path)
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        res = es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, limit=2, calibration=path)
        sent = len(srv.requests)
    assert res.exit_code == 0, res.stopped
    assert sent == 2


# ------------------------------------------------------------------ T-0076 criterion 8: the qargmax arms

QA, QA_SC = "perstep_mar_qargmax", "perstep_mar_qargmax_sc"
#: the arm whose frozen file each calibrated arm must run on (Q3: an SC twin takes its base arm's file)
SOURCE = {"perstep_mar": "perstep_mar", "perstep_mar_sc": "perstep_mar", QA: QA, QA_SC: QA}


def _known(arm: str) -> None:
    """Precondition of every refusal row below: an unknown arm is refused too, which would make a refusal row
    pass for the wrong reason before T-0076."""
    specs = es.need("local_astra.eval.cd.arms").ARM_SPECS
    assert arm in specs, f"T-0076 not implemented: {arm} is not in ARM_SPECS {sorted(specs)}"


def _run_with(tmp_path: Path, arm: str, calibration: Path | None) -> tuple[object, int, bool]:
    """``(result or None, requests sent, raised ValueError)``."""
    extra = {} if calibration is None else {"calibration": calibration}
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        try:
            res = es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, limit=2, **extra)
            raised = False
        except ValueError:
            res, raised = None, True
        sent = len(srv.requests)
    return res, sent, raised


def test_qargmax_is_a_calibrated_arm() -> None:
    cal = es.calibration()
    assert QA in cal.CALIBRATED_ARMS, cal.CALIBRATED_ARMS
    assert {"perstep_mar", "perstep_mar_sc"} <= set(cal.CALIBRATED_ARMS), cal.CALIBRATED_ARMS


@pytest.mark.parametrize("arm", [QA, QA_SC])
def test_runner_refuses_qargmax_without_the_frozen_file(tmp_path: Path, arm: str) -> None:
    _known(arm)
    res, sent, raised = _run_with(tmp_path, arm, None)
    assert sent == 0
    assert raised or (res is not None and res.exit_code != 0)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("arm", "file_arm"),
    [
        (QA, "perstep_mar"),
        (QA_SC, "perstep_mar"),
        ("perstep_mar", QA),
        ("perstep_mar_sc", QA),
        # ruling A2 (review B T3): an SC twin's file is never a source, not even for its own family
        ("perstep_mar", "perstep_mar_sc"),
        ("perstep_mar_sc", "perstep_mar_sc"),
        (QA, QA_SC),
        (QA_SC, QA_SC),
    ],
    ids=[
        "qargmax-on-mar",
        "qargmax_sc-on-mar",
        "mar-on-qargmax",
        "mar_sc-on-qargmax",
        "mar-on-mar_sc-file",
        "mar_sc-on-mar_sc-file",
        "qargmax-on-qargmax_sc-file",
        "qargmax_sc-on-qargmax_sc-file",
    ],
)
def test_runner_refuses_a_frozen_file_from_another_calibration_source(
    tmp_path: Path, arm: str, file_arm: str
) -> None:
    """Adversarial (criterion 8): ``perstep_mar_qargmax`` must not run on ``perstep_mar``'s knobs (nor the
    reverse; nor an SC twin on the other family's file) -- ``ValueError`` and zero requests sent."""
    _known(arm)
    path = es.freeze_calibration(tmp_path, arm=file_arm)
    with es.fake(es.sudoku_script("test", lambda i, rid, b: "solved"), tmp_path) as srv:
        with pytest.raises(ValueError):
            es.run(srv, tmp_path, task="sudoku", cut="test", arm=arm, limit=2, calibration=path)
        sent = len(srv.requests)
    assert sent == 0


@pytest.mark.parametrize("arm", ["perstep_mar", "perstep_mar_sc", QA, QA_SC])
def test_runner_accepts_the_arms_own_calibration_source(tmp_path: Path, arm: str) -> None:
    """Companion (the check is not too wide): each arm runs on its source's file -- Mar and Mar+SC on
    ``perstep_mar``'s, qargmax and its SC twin on ``perstep_mar_qargmax``'s."""
    _known(arm)
    cs.warm_sudoku("test")
    path = es.freeze_calibration(tmp_path, arm=SOURCE[arm])
    res, sent, raised = _run_with(tmp_path, arm, path)
    assert not raised and res is not None
    assert res.exit_code == 0, res.stopped  # type: ignore[attr-defined]
    assert sent == 2


def test_driver_calibrates_the_qargmax_arm(tmp_path: Path) -> None:
    """``calibrate(arm="perstep_mar_qargmax")``: the 18-point grid on the dev cut with the new mode on every
    request, and the frozen file names the new arm (so the runner's source check can tell it from Mar's)."""
    cs.warm_sudoku("dev")
    test_prompts = set(es.prompt_to_id("sudoku", "test"))

    def decide(i: int, rid: str, body: dict) -> Reply | str:  # type: ignore[type-arg]
        x = body["vllm_xargs"]
        if (x.get("astra_cd_entropy_bound"), x.get("astra_cd_confidence_threshold")) == (0.3, 5e-3):
            return "solved"
        return "wrong"

    repo, head = es.clean_repo(tmp_path)
    out = tmp_path / "artifacts"
    out.mkdir()
    with es.fake(es.sudoku_script("dev", decide), tmp_path) as srv:
        path = es.calibration().calibrate(
            task="sudoku",
            base_url=srv.url,
            engine_log=srv.engine_log,
            out_dir=out,
            repo=repo,
            expected_commit=head,
            limit=2,
            arm=QA,
            path=out / "calibration-sudoku-perstep_mar_qargmax.json",
        )
        bodies = list(srv.requests)
    assert len(bodies) == 18 * 2
    assert not {b["messages"][-1]["content"] for b in bodies} & test_prompts
    assert {b["vllm_xargs"]["astra_cd_mode"] for b in bodies} == {QA}
    assert {b["vllm_xargs"].get("astra_cd_self_conditioning", "model") for b in bodies} == {"model"}
    data = json.loads(Path(path).read_text())
    assert (data["entropy_bound"], data["confidence_threshold"]) == (0.3, 5e-3)
    assert data["arm"] == QA and data["cut"] == "dev" and data["task"] == "sudoku"
    # load_frozen accepts the file it froze (its arm is a calibrated arm)
    es.calibration().load_frozen(path, task="sudoku")


@pytest.mark.parametrize(
    ("arm", "file_arm"),
    [
        ("perstep_mar", "perstep_mar_sc"),
        ("perstep_mar_sc", "perstep_mar_sc"),
        (QA, QA_SC),
        (QA_SC, QA_SC),
    ],
)
def test_load_frozen_refuses_an_sc_twin_file_for_its_own_family(
    tmp_path: Path, arm: str, file_arm: str
) -> None:
    """Ruling A2 (review B T3): ``load_frozen(..., arm=...)`` refuses a file naming an SC twin even for the twin's
    own family. Kills ``cal_twin_file_same_family`` (accept a twin file within its family). Companion: the
    family's base file is accepted for the same ``arm``."""
    bad = es.freeze_calibration(tmp_path / "twin", arm=file_arm)
    with pytest.raises(ValueError):
        es.calibration().load_frozen(bad, task="sudoku", arm=arm)
    good = es.freeze_calibration(tmp_path / "base", arm=SOURCE[arm])
    knobs, _ = es.calibration().load_frozen(good, task="sudoku", arm=arm)
    assert knobs == {"entropy_bound": 0.03, "confidence_threshold": 5e-4}


def _calibrate(srv: object, tmp_path: Path, out: Path, **kw: object) -> Path:
    repo, head = es.clean_repo(tmp_path)
    return es.calibration().calibrate(  # type: ignore[no-any-return]
        task="sudoku",
        base_url=srv.url,  # type: ignore[attr-defined]
        engine_log=srv.engine_log,  # type: ignore[attr-defined]
        out_dir=out,
        repo=repo,
        expected_commit=head,
        limit=1,
        **kw,
    )


def test_calibrate_qargmax_default_path_leaves_mars_file_untouched(tmp_path: Path) -> None:
    """Ruling A1 (review B T1): with no ``path``, ``calibrate(arm="perstep_mar_qargmax")`` writes
    ``<out>/calibration-sudoku-perstep_mar_qargmax.json`` and leaves a pre-existing ``calibration-sudoku.json``
    (Mar's) byte-identical. Kills ``default_path_same`` (both arms default to ``calibration-<task>.json``, which
    ``freeze`` overwrites silently)."""
    cs.warm_sudoku("dev")
    out = tmp_path / "artifacts"
    out.mkdir()
    mar_file = out / "calibration-sudoku.json"
    es.calibration().freeze(
        {
            "entropy_bound": 0.01,
            "confidence_threshold": 5e-5,
            "task": "sudoku",
            "cut": "dev",
            "arm": "perstep_mar",
        },
        mar_file,
    )
    before = mar_file.read_bytes()
    with es.fake(es.sudoku_script("dev", lambda i, rid, b: "wrong"), tmp_path) as srv:
        path = _calibrate(srv, tmp_path, out, arm=QA)
    assert Path(path) == out / "calibration-sudoku-perstep_mar_qargmax.json", path
    assert json.loads(Path(path).read_text())["arm"] == QA
    assert mar_file.read_bytes() == before, "calibrate(arm=qargmax) overwrote Mar's frozen file"


@pytest.mark.parametrize("arm", ["perstep_mar_sc", QA_SC])
def test_calibrate_refuses_an_sc_twin_before_running(tmp_path: Path, arm: str) -> None:
    """Ruling A2 (review B T2): ``calibrate`` runs only a grid arm; an SC twin is refused with ``ValueError``
    before any request is sent and no file is written. Kills widening ``GRID_ARMS`` to ``CALIBRATED_ARMS``."""
    out = tmp_path / "artifacts"
    out.mkdir()
    with es.fake(es.sudoku_script("dev", lambda i, rid, b: "solved"), tmp_path) as srv:
        with pytest.raises(ValueError):
            _calibrate(srv, tmp_path, out, arm=arm)
        sent = len(srv.requests)
    assert sent == 0
    assert list(out.glob("calibration*.json")) == []
