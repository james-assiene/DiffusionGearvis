"""T-0081b criteria 1 and 7: the stage-2 arm set of ``scripts/gpu/cd_toolcall.py`` (hermetic: no engine, no request).

``stage2_cells(selection, commit)`` derives the stage-2 cells from a dev-stage selection of the same commit; after
T-0081b they are the chosen (thought, scope) under ``end_projection``, ``perstep_mar``, ``perstep_mar_qargmax`` and
``perstep_mar+seed``, in that order (the locked script ``t0080_toolcall.sh`` reads them through this function). The
stage-1 cells are unchanged, and the seed constants (noise 0.0, start step 24) are not CLI options.

The script is loaded by path; importing it opens no socket (its docstring's promise). Regime: CPU, pure.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "scripts" / "gpu" / "cd_toolcall.py"
COMMIT = "abc1234abc1234abc1234abc1234abc1234abc1"


def cli() -> ModuleType:
    assert CLI.is_file(), f"{CLI.relative_to(REPO)} does not exist"
    name = "_t0081b_cd_toolcall_cli_unit"
    spec = importlib.util.spec_from_file_location(name, CLI)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    return m


def selection(tmp_path: Path, choice: str, commit: str = COMMIT) -> Path:
    p = tmp_path / "selection-dev.json"
    p.write_text(json.dumps({"stage": "dev", "rule": "stage1", "choice": choice, "commit": commit}))
    return p


@pytest.mark.parametrize("thought", ["T1", "T2", "T4"])
def test_stage2_cells_add_the_seed_arm_last(tmp_path: Path, thought: str) -> None:
    """Criterion 7: the chosen (thought, named) under the four stage-2 arms, in exactly this order."""
    got = cli().stage2_cells(selection(tmp_path, f"{thought}/named/perstep_mar"), COMMIT)
    assert list(got) == [
        f"{thought}/named/end_projection",
        f"{thought}/named/perstep_mar",
        f"{thought}/named/perstep_mar_qargmax",
        f"{thought}/named/perstep_mar+seed",
    ]


def test_stage2_cells_still_refuse_another_commits_selection(tmp_path: Path) -> None:
    """Unchanged (T-0080b criterion 6): a selection written at another commit is refused."""
    with pytest.raises(ValueError):
        cli().stage2_cells(selection(tmp_path, "T4/named/perstep_mar", commit="def5678"), COMMIT)


def test_stage1_cells_are_unchanged() -> None:
    """Criterion 7: stage 1 is still ``T1|T2|T4 / named / perstep_mar`` (no seed cell)."""
    assert tuple(cli().STAGE1_CELLS) == (
        "T1/named/perstep_mar",
        "T2/named/perstep_mar",
        "T4/named/perstep_mar",
    )


def test_the_seed_constants_are_not_cli_options() -> None:
    """Criterion 1: noise 0.0 and start step 24 are module constants, not overridable from the CLI -- no option
    mentions a noise, a start step or a draft. (Read from the parser's option strings; nothing is invoked.)"""
    parser = cli()._parser()
    opts = [o for a in parser._actions for o in a.option_strings]
    assert opts, "fixture: no options found"
    bad = [o for o in opts if any(w in o.lower() for w in ("noise", "start", "draft"))]
    assert bad == [], bad


# =========================================================================== T-0084d criterion 4: --compare-native
# ``scripts/gpu/cd_toolcall.py --compare-native --stage dev|test [--selection PATH]``. Hermetic: ``twopass.run`` is
# replaced (monkeypatch, restored after the row) by a stand-in that records the config and writes a hand-built
# artifact (``tests/cd/_native_artifacts.full_rows``) for the configured cells; ``report`` and the rule run for real on
# it. No engine, no socket.
#
# Pins (test-author proposal; criterion 4 names the file and its fields): the file is
# ``<out-dir>/selection-native-<stage>.json`` with keys ``stage``, ``rule``, ``native_thought``, ``choice``, ``winner``,
# ``commit`` (= ``--expected-commit``), ``artifact_sha256`` and ``cells``. Cell order is not pinned (sets).

from tests.cd import _native_artifacts as na  # noqa: E402

ARMS4 = ("end_projection", "perstep_mar", "perstep_mar_qargmax", "perstep_mar+seed")


class _FakeRun:
    def __init__(self, walls: dict[str, float]) -> None:
        self.walls = walls
        self.cfgs: list[object] = []

    def __call__(self, cfg: object) -> object:
        from local_astra.eval.cd import twopass

        self.cfgs.append(cfg)
        cells = list(cfg.cells)  # type: ignore[attr-defined]
        stage = str(cfg.stage)  # type: ignore[attr-defined]
        art = na.write_artifact(
            Path(cfg.out_dir) / f"20261011T000000Z-toolcall-{stage}.jsonl",  # type: ignore[attr-defined]
            stage,
            cells,
            na.full_rows(cells, walls=self.walls),
        )
        return twopass.TwoPassResult(exit_code=0, artifact=art, stopped=None, n=5)


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--engine-log",
        str(tmp_path / "vllm.log"),
        "--out-dir",
        str(tmp_path / "out"),
        "--expected-commit",
        COMMIT,
        "--repo",
        str(tmp_path),
        *extra,
    ]


def _run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walls: dict[str, float], *extra: str
) -> tuple[int, _FakeRun]:
    from local_astra.eval.cd import twopass

    fake = _FakeRun(walls)
    monkeypatch.setattr(twopass, "run", fake)
    try:
        rc = cli().main(_argv(tmp_path, *extra))
    except (
        SystemExit
    ) as exc:  # argparse: an unknown option (``--compare-native`` before T-0084d) exits 2
        rc = exc.code if isinstance(exc.code, int) else 2
    return int(rc), fake


def test_compare_native_dev_runs_the_16_cells_and_writes_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 4: dev default cells = cc ``T1/named/{4}`` + native ``{T1,T2,T4}/named/{4}`` (16). On the hand-built
    artifact (native T4 end_projection at pass-2 wall 10, all else 500; every draft invalid) the rule gives θ_n T4,
    ``cd_on_invalid:native/T4/named/end_projection``, winner native; ``selection-native-dev.json`` records it with
    the commit, the artifact's sha256 and the cells."""
    rc, fake = _run_main(
        tmp_path,
        monkeypatch,
        {"native/T4/named/end_projection": 10.0},
        "--compare-native",
        "--stage",
        "dev",
    )
    assert rc == 0, f"--compare-native --stage dev exited {rc}"
    assert len(fake.cfgs) == 1
    cells = list(fake.cfgs[0].cells)  # type: ignore[attr-defined]
    want = {f"T1/named/{a}" for a in ARMS4} | {
        f"native/{t}/named/{a}" for t in ("T1", "T2", "T4") for a in ARMS4
    }
    assert len(cells) == 16 and set(cells) == want, cells
    path = tmp_path / "out" / "selection-native-dev.json"
    assert path.is_file(), sorted(p.name for p in (tmp_path / "out").iterdir())
    sel = json.loads(path.read_text())
    assert sel["stage"] == "dev"
    assert "rule" in sel
    assert sel["native_thought"] == "T4"
    assert sel["choice"] == "cd_on_invalid:native/T4/named/end_projection"
    assert sel["winner"] == "native"
    assert sel["commit"] == COMMIT
    art = next((tmp_path / "out").glob("*-toolcall-dev.jsonl"))
    assert sel["artifact_sha256"] == na.sha256(art)
    assert set(sel["cells"]) == want


def _native_selection(tmp_path: Path, thought: str = "T2", **over: object) -> Path:
    p = tmp_path / "selection-native-dev.json"
    data = {
        "stage": "dev",
        "rule": "T-0084d",
        "native_thought": thought,
        "choice": f"cd_on_invalid:native/{thought}/named/end_projection",
        "winner": "native",
        "commit": COMMIT,
        **over,
    }
    p.write_text(json.dumps(data))
    return p


def test_compare_native_test_runs_cc_t1_and_native_theta_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 4: test cells = cc ``T1/named/{4}`` + native ``θ_n/named/{4}`` (8), θ_n = the selection's
    ``native_thought`` (T2). Native T2 end_projection at wall 10 -> ``selection-native-test.json`` with stage test,
    θ_n T2 and that choice."""
    sel = _native_selection(tmp_path, "T2")
    rc, fake = _run_main(
        tmp_path,
        monkeypatch,
        {"native/T2/named/end_projection": 10.0},
        "--compare-native",
        "--stage",
        "test",
        "--selection",
        str(sel),
    )
    assert rc == 0, f"--compare-native --stage test exited {rc}"
    cells = list(fake.cfgs[0].cells)  # type: ignore[attr-defined]
    want = {f"T1/named/{a}" for a in ARMS4} | {f"native/T2/named/{a}" for a in ARMS4}
    assert len(cells) == 8 and set(cells) == want, cells
    out = json.loads((tmp_path / "out" / "selection-native-test.json").read_text())
    assert (out["stage"], out["native_thought"], out["winner"]) == ("test", "T2", "native")
    assert out["choice"] == "cd_on_invalid:native/T2/named/end_projection"
    assert out["commit"] == COMMIT


def test_compare_native_test_without_a_selection_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test stage takes θ_n from a dev selection file: none given -> exit 2, nothing run. (The flag itself must be
    accepted -- the dev invocation runs -- so argparse's exit 2 on an unknown option does not satisfy this row.)"""
    rc, fake = _run_main(tmp_path, monkeypatch, {}, "--compare-native", "--stage", "dev")
    assert len(fake.cfgs) == 1, f"--compare-native --stage dev did not run (exit {rc})"
    rc, fake = _run_main(tmp_path, monkeypatch, {}, "--compare-native", "--stage", "test")
    assert rc == 2
    assert fake.cfgs == []


def test_existing_dev_invocation_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 4, negative: without ``--compare-native`` the dev stage runs ``STAGE1_CELLS`` and writes
    ``selection-dev.json`` (stage 1's choice, T4 on a tie), and no ``selection-native-dev.json``."""
    rc, fake = _run_main(tmp_path, monkeypatch, {}, "--stage", "dev")
    assert rc == 0
    assert list(fake.cfgs[0].cells) == list(cli().STAGE1_CELLS)  # type: ignore[attr-defined]
    out = tmp_path / "out"
    assert json.loads((out / "selection-dev.json").read_text())["choice"] == "T4/named/perstep_mar"
    assert not (out / "selection-native-dev.json").exists()


def test_compare_native_with_no_choice_exits_1_and_writes_no_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F2: when the rule chooses nothing -- every pass-2 row CS-false (every CD policy ineligible) and every
    draft invalid (both draft-only policies at schema 0) -- ``--compare-native --stage dev`` exits 1 and writes no
    ``selection-native-dev.json``."""
    from local_astra.eval.cd import twopass

    class _NoChoice(_FakeRun):
        def __call__(self, cfg: object) -> object:
            res = super().__call__(cfg)
            art = Path(res.artifact)  # type: ignore[attr-defined]
            lines = [json.loads(x) for x in art.read_text().splitlines() if x.strip()]
            for x in lines:
                if x.get("pass") == 2:
                    x["cs"] = False
            art.write_text("".join(json.dumps(x) + "\n" for x in lines))
            return res

    fake = _NoChoice({})
    monkeypatch.setattr(twopass, "run", fake)
    try:
        rc = cli().main(_argv(tmp_path, "--compare-native", "--stage", "dev"))
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 2
    assert len(fake.cfgs) == 1, f"the run did not happen (exit {rc})"
    assert rc == 1
    assert not (tmp_path / "out" / "selection-native-dev.json").exists()
