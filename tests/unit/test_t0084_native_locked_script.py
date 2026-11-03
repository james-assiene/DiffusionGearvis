"""T-0084d criteria 5 and 6: the locked script ``scripts/locked/t0084_native.sh``, hermetically (fast suite, no GPU,
no engine, no request). Harness as ``tests/unit/test_t0082_replay_locked_script.py`` (the t0080 pattern the task
names): the script is copied into a throwaway top-level directory and run with a stub ``PATH`` directory that shadows
every command able to reach an engine, a container or the GPU (``STUBBED``); ``FORBIDDEN`` commands fail loudly. Each
stub appends its argv to a call log. The copy has no ``services/`` directory, so even an unstubbed engine start has
nothing to run.

CLI (criterion 5): ``t0084_native.sh --stage dev|test [--selection PATH]``. Conventions (T-0071b C11): a refusal prints
``REFUSED`` and exits 2 before ``make``, the boot or the driver; a failure before the driver prints ``FAIL``; the
driver's status is the script's (``driver exit N``); ``nvidia-smi`` never decides the exit code. The out-dir is
``logs/t0084-native-<stage>-<SHA>`` and is refused if it exists. ``--stage test`` is refused unless the selection has
``stage == dev``, ``commit == HEAD`` and ``winner == native``.

Pins where the task is silent (reported): the driver is the two-pass CLI ``cd_toolcall.py --compare-native --stage
<stage>`` (plus ``--selection`` on test, as given); the script reads the selection with a command outside the stubbed
set (``python3``, as t0080_toolcall.sh does), not ``uv``; ``commit`` is compared with the full HEAD sha (the CLI
writes ``--expected-commit``, which the locked scripts pass as the full sha).

Regime: CPU; bash; the stubbed git reports HEAD ``abc1234abc1234abc1234abc1234abc1234abc1`` (short ``abc1234``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
LOCKED = REPO / "scripts" / "locked" / "t0084_native.sh"
BRANCH_BASE = "9e84f21"  # T-0084d's fork point (main after T-0084c)
HEAD = "abc1234abc1234abc1234abc1234abc1234abc1"
SHORT = "abc1234"

STUBBED = (
    "git",
    "ss",
    "docker",
    "sg",
    "setsid",
    "sleep",
    "uptime",
    "make",
    "curl",
    "nvidia-smi",
    "uv",
)
FORBIDDEN = ("sudo", "systemctl", "podman", "vllm", "wget", "astra")
RECOGNISABLE = re.compile(r"(?m)^.*\b(driver exit \d+|REFUSED|FAIL)\b")


def _script() -> Path:
    assert LOCKED.is_file(), f"T-0084d not implemented: {LOCKED.relative_to(REPO)} does not exist"
    return LOCKED


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f'#!/bin/bash\necho "{name} $*" >> "$T_CALLS"\n{body}\n')
    p.chmod(0o755)


def run_locked(
    tmp_path: Path,
    args: list[str],
    *,
    driver_exit: int = 0,
    smi_fails: bool = False,
    dirty: bool = False,
    make_exit: int = 0,
    curl_exit: int = 0,
    out_dir_exists: str | None = None,
    pre_dirs: list[str] | None = None,
    port_bound: bool = False,
    containers: bool = False,
    vllm_drift: bool = False,
    engine_alive: bool = False,
) -> tuple[int, str, list[list[str]], list[str]]:
    """Run a copy of the locked script; returns (exit code, console output, argv of every ``uv`` call, the ordered
    call log). ``out_dir_exists`` names a stage whose ``logs/t0084-native-<stage>-<SHA>`` already exists."""
    text = _script().read_text()
    names = "|".join(re.escape(n) for n in STUBBED + FORBIDDEN)
    absolute = re.findall(r"/(?:usr/)?(?:local/)?s?bin/(" + names + r")\b", text)
    if absolute:
        pytest.fail(
            f"the locked script calls {sorted(set(absolute))} by absolute path; the harness cannot stub it"
        )
    top = tmp_path / "top"
    (top / "scripts" / "locked").mkdir(parents=True)
    shutil.copy(LOCKED, top / "scripts" / "locked" / LOCKED.name)
    if out_dir_exists is not None:
        (top / "logs" / f"t0084-native-{out_dir_exists}-{SHORT}").mkdir(parents=True)
    for d in pre_dirs or ():
        (top / "logs" / d).mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.argv"
    calls_log = tmp_path / "calls.log"
    _stub(
        bin_dir,
        "git",
        'case "$*" in\n'
        '  *--show-toplevel*) echo "$T_TOP" ;;\n'
        f'  *"rev-parse --short"*) echo {SHORT} ;;\n'
        f'  *"rev-parse HEAD"*) echo {HEAD} ;;\n'
        '  *"status --porcelain"*) if [ -n "$T_DIRTY" ]; then echo " M x.py"; fi ;;\n'
        '  *diff*services/vllm.sh*) if [ -n "$T_VLLM_DRIFT" ]; then echo "+--max-model-len 4096"; fi ;;\n'
        "  *) ;;\nesac\nexit 0",
    )
    _stub(
        bin_dir,
        "docker",
        'case "$*" in\n'
        '  *ps*) if [ -n "$T_CONTAINERS" ]; then echo "deadbeef0001 vllm/vllm-openai:v0.29.0"; fi ;;\n'
        "esac\nexit 0",
    )
    for quiet in ("sg", "sleep", "uptime"):
        _stub(bin_dir, quiet, "exit 0")
    _stub(
        bin_dir,
        "ss",
        'if [ -n "$T_PORT_BOUND" ]; then echo "LISTEN 0 4096 127.0.0.1:8001 0.0.0.0:*"; fi\nexit 0',
    )
    _stub(bin_dir, "setsid", 'if [ -n "$T_ENGINE_ALIVE" ]; then exec /bin/sleep 60; fi\nexit 0')
    _stub(bin_dir, "make", 'exit "$T_MAKE_EXIT"')
    _stub(bin_dir, "curl", 'exit "$T_CURL_EXIT"')
    _stub(
        bin_dir,
        "nvidia-smi",
        'if [ -n "$T_SMI_FAIL" ]; then echo "NVIDIA-SMI has failed" >&2; exit 9; fi\n'
        'echo "NVIDIA GeForce RTX 5090, 595.84, 1000 MiB, 31000 MiB"',
    )
    _stub(
        bin_dir,
        "uv",
        'printf "%s\\n" "$@" >> "$T_UV"; echo "--end--" >> "$T_UV"; exit "$T_DRIVER_EXIT"',
    )
    for loud in FORBIDDEN:
        _stub(bin_dir, loud, 'echo "unexpected command in the locked-script harness" >&2; exit 97')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "T_CALLS": str(calls_log),
        "T_TOP": str(top),
        "T_UV": str(uv_log),
        "T_DIRTY": "1" if dirty else "",
        "T_SMI_FAIL": "1" if smi_fails else "",
        "T_MAKE_EXIT": str(make_exit),
        "T_CURL_EXIT": str(curl_exit),
        "T_DRIVER_EXIT": str(driver_exit),
        "T_PORT_BOUND": "1" if port_bound else "",
        "T_CONTAINERS": "1" if containers else "",
        "T_VLLM_DRIFT": "1" if vllm_drift else "",
        "T_ENGINE_ALIVE": "1" if engine_alive else "",
    }
    proc = subprocess.run(
        ["bash", str(top / "scripts" / "locked" / LOCKED.name), *args],
        cwd=top,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    uv_calls: list[list[str]] = []
    if uv_log.is_file():
        cur: list[str] = []
        for line in uv_log.read_text().splitlines():
            if line == "--end--":
                uv_calls.append(cur)
                cur = []
            else:
                cur.append(line)
    calls = calls_log.read_text().splitlines() if calls_log.is_file() else []
    return proc.returncode, proc.stdout + proc.stderr, uv_calls, calls


def _driver_calls(uv_calls: list[list[str]]) -> list[list[str]]:
    return [argv for argv in uv_calls if any(a.endswith("cd_toolcall.py") for a in argv)]


def _flag(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _selection(tmp_path: Path, **over: Any) -> Path:
    """A dev selection that the test stage accepts (stage dev, commit HEAD, winner native); ``over`` changes fields
    (a value ``None`` removes the key)."""
    data: dict[str, Any] = {
        "stage": "dev",
        "rule": "T-0084d",
        "native_thought": "T2",
        "choice": "cd_on_invalid:native/T2/named/end_projection",
        "winner": "native",
        "commit": HEAD,
        "artifact_sha256": "ab" * 32,
    }
    for k, v in over.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    p = tmp_path / "sel" / "selection-native-dev.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return p


def _valid(tmp_path: Path, stage: str) -> list[str]:
    if stage == "dev":
        return ["--stage", "dev"]
    return ["--stage", "test", "--selection", str(_selection(tmp_path))]


# --------------------------------------------------------------------------- shape


def test_script_parses() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_script())], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_happy_path_wheel_boot_health_then_one_compare_native_driver(
    tmp_path: Path, stage: str
) -> None:
    """Criterion 5: ``make plugin-wheel``, the engine boot, ``/health``, then the two-pass CLI with
    ``--compare-native`` exactly once, ``--stage`` as given (``--selection`` passed through on test, absent on dev), a
    loopback base URL, the out-dir ``logs/t0084-native-<stage>-<SHA>`` and ``--expected-commit`` HEAD; exit 0."""
    args = _valid(tmp_path, stage)
    rc, output, uv_calls, calls = run_locked(tmp_path, args)
    assert rc == 0, output
    drivers = _driver_calls(uv_calls)
    assert len(drivers) == 1, f"driver calls: {uv_calls}\n{output}"
    argv = drivers[0]
    assert "--compare-native" in argv, argv
    assert _flag(argv, "--stage") == stage, argv
    sel = _flag(argv, "--selection")
    if stage == "test":
        assert sel is not None and Path(sel).resolve() == Path(args[-1]).resolve(), argv
    else:
        assert sel is None, argv
    out = _flag(argv, "--out-dir")
    assert out is not None and out.rstrip("/").endswith(f"logs/t0084-native-{stage}-{SHORT}"), argv
    assert _flag(argv, "--expected-commit") == HEAD, argv
    base = _flag(argv, "--base-url")
    assert base is not None and re.match(
        r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?/", base
    ), argv
    first_uv = next(i for i, c in enumerate(calls) if c.startswith("uv ") and "cd_toolcall.py" in c)
    wheel = [i for i, c in enumerate(calls) if c.startswith("make ") and "plugin-wheel" in c]
    boot = [i for i, c in enumerate(calls) if c.startswith("setsid ")]
    health = [i for i, c in enumerate(calls) if c.startswith("curl ") and "/health" in c]
    assert wheel and boot and health, calls
    assert wheel[0] < boot[0], calls
    assert wheel[0] < health[0] < first_uv, calls


def test_the_engine_is_started_engine_only_with_services_vllm_sh() -> None:
    code = "\n".join(_code_lines(_script().read_text()))
    assert "services/vllm.sh" in code
    assert not re.search(r"\bastra\s+up\b", code), "the full stack is started, not the engine alone"


def test_every_nvidia_smi_pipeline_ends_with_or_true() -> None:
    """Criterion 5: every ``nvidia-smi … || true``."""
    lines = [ln for ln in _code_lines(_script().read_text()) if "nvidia-smi" in ln]
    assert lines, "the script prints no nvidia-smi line"
    bad = [ln.strip() for ln in lines if not ln.rstrip().endswith("|| true")]
    assert bad == [], f"nvidia-smi lines without '|| true': {bad}"


# --------------------------------------------------------------------------- exit-code capture


@pytest.mark.parametrize("driver_exit", [0, 1, 2, 3])
@pytest.mark.parametrize("smi_fails", [False, True], ids=["smi-ok", "smi-fails"])
def test_the_drivers_exit_code_is_the_scripts(
    tmp_path: Path, driver_exit: int, smi_fails: bool
) -> None:
    """Criterion 5: the driver runs in an ``||`` list, so the script exits with the driver's code (a driver refusal, 2,
    included) and prints ``driver exit <code>``; a failing ``nvidia-smi`` neither stops the launch nor replaces it."""
    rc, output, uv_calls, _ = run_locked(
        tmp_path, _valid(tmp_path, "test"), driver_exit=driver_exit, smi_fails=smi_fails
    )
    assert len(_driver_calls(uv_calls)) == 1, f"the driver did not run:\n{output}"
    assert rc == driver_exit, output
    assert f"driver exit {driver_exit}" in output


# --------------------------------------------------------------------------- refusals

SEL_CASES: dict[str, dict[str, Any]] = {
    "sel_winner_cc": {"winner": "cc"},
    "sel_no_winner": {"winner": None},
    "sel_other_sha": {"commit": "def5678def5678def5678def5678def5678def56"},
    "sel_short_other_sha": {"commit": "def5678"},
    # review F3: HEAD's own 7-char short sha is not HEAD (the selection records the full sha)
    "sel_short_head": {"commit": SHORT},
    "sel_no_commit": {"commit": None},
    "sel_stage_test": {"stage": "test"},
    "sel_no_stage": {"stage": None},
}
ENV_CASES: dict[str, dict[str, Any]] = {
    "dirty": {"dirty": True},
    "out_dir_exists_dev": {"out_dir_exists": "dev"},
    "port_bound": {"port_bound": True},
    "containers": {"containers": True},
    "vllm_sh_changed": {"vllm_drift": True},
}


def _refusal(tmp_path: Path, case: str) -> tuple[list[str], dict[str, Any]]:
    if case in SEL_CASES:
        sel = _selection(tmp_path, **SEL_CASES[case])
        return ["--stage", "test", "--selection", str(sel)], {}
    if case in ENV_CASES:
        return ["--stage", "dev"], ENV_CASES[case]
    if case == "selection_not_json":
        p = tmp_path / "garbage.json"
        p.write_text("not json {")
        return ["--stage", "test", "--selection", str(p)], {}
    args = {
        "no_stage": [],
        "bad_stage": ["--stage", "prod"],
        "empty_stage": ["--stage", ""],
        "unknown_flag": ["--stage", "dev", "--cell", "T1/named/perstep_mar"],
        "test_without_selection": ["--stage", "test"],
        "selection_missing_file": ["--stage", "test", "--selection", str(tmp_path / "nope.json")],
    }
    return args[case], {}


REFUSALS = [
    "no_stage",
    "bad_stage",
    "empty_stage",
    "unknown_flag",
    "test_without_selection",
    "selection_missing_file",
    "selection_not_json",
    *SEL_CASES,
    *ENV_CASES,
]


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_exit_2_before_anything_runs(tmp_path: Path, case: str) -> None:
    """Criterion 5: no stage or an unknown one, an unknown flag, ``--stage test`` without a readable selection or with a
    selection whose ``winner`` is not native, whose ``commit`` is not HEAD (another SHA, full or short) or whose
    ``stage`` is not dev, a dirty tree, an existing ``logs/t0084-native-<stage>-<SHA>``, a bound Astra port, running
    containers, and ``services/vllm.sh`` changed since 806da12 -> ``REFUSED``, exit 2, before the wheel build, the boot,
    the health check or the driver."""
    args, kw = _refusal(tmp_path, case)
    rc, output, uv_calls, calls = run_locked(tmp_path, args, **kw)
    assert rc == 2, (rc, output)
    assert "REFUSED" in output, output
    assert _driver_calls(uv_calls) == [], "the driver ran"
    started = [c for c in calls if c.split(" ", 1)[0] in ("make", "setsid", "curl", "uv")]
    assert started == [], f"commands ran before the refusal: {started}"


def test_an_existing_test_out_dir_refuses_the_test_stage(tmp_path: Path) -> None:
    """Once per SHA: ``logs/t0084-native-test-<SHA>`` existing refuses ``--stage test`` with a valid selection."""
    rc, output, uv_calls, _ = run_locked(tmp_path, _valid(tmp_path, "test"), out_dir_exists="test")
    assert rc == 2 and "REFUSED" in output, output
    assert _driver_calls(uv_calls) == []


@pytest.mark.parametrize(
    ("stage", "pre"),
    [
        ("test", "t0084-native-dev-{s}"),
        ("dev", "t0084-native-test-{s}"),
        ("dev", "t0080-toolcall-dev-{s}"),
        ("dev", "t0082-replay-dev-{s}"),
        ("dev", "t0084-native-dev-def5678"),
    ],
)
def test_other_out_dirs_do_not_refuse(tmp_path: Path, stage: str, pre: str) -> None:
    """The other side of the once-per-SHA refusal (a guard too wide): the other stage's out-dir, the t0080/t0082
    evidence of this SHA and this stage's out-dir of another SHA do not refuse; the driver runs once."""
    rc, output, uv_calls, _ = run_locked(
        tmp_path, _valid(tmp_path, stage), pre_dirs=[pre.format(s=SHORT)]
    )
    assert rc == 0, output
    assert len(_driver_calls(uv_calls)) == 1


def test_a_selection_with_extra_fields_and_the_full_head_is_accepted(tmp_path: Path) -> None:
    """The selection check is not too wide: the CLI's real file carries more keys (cells, tables, artifact sha); with
    stage dev, commit HEAD and winner native it is accepted."""
    sel = _selection(tmp_path, cells=["T1/named/perstep_mar"], tables={"step1": {}}, policies={})
    rc, output, uv_calls, _ = run_locked(tmp_path, ["--stage", "test", "--selection", str(sel)])
    assert rc == 0, output
    assert len(_driver_calls(uv_calls)) == 1


EARLY_FAILURES: dict[str, dict[str, Any]] = {
    "make": {"make_exit": 2},
    "unhealthy": {"curl_exit": 7},
    "health_timeout": {"curl_exit": 7, "engine_alive": True},
}


@pytest.mark.parametrize("case", list(EARLY_FAILURES))
def test_a_failure_before_the_driver_is_loud_and_never_runs_the_driver(
    tmp_path: Path, case: str
) -> None:
    rc, output, uv_calls, _ = run_locked(tmp_path, _valid(tmp_path, "dev"), **EARLY_FAILURES[case])
    assert rc != 0, output
    assert _driver_calls(uv_calls) == [], "the driver ran"
    assert RECOGNISABLE.search(output), f"no recognisable exit line in:\n{output}"


# --------------------------------------------------------------------------- criterion 6 (negative)


CRITERION_6 = (
    "scripts/locked/t0080_toolcall.sh",
    "scripts/locked/t0082_replay.sh",
    "local_astra/bench/thresholds.py",
)
#: T-0084d's merge commit 2e89e87; BASE = ``git merge-base 2e89e87^1 2e89e87^2`` = ``BRANCH_BASE`` (9e84f21), TIP =
#: ``2e89e87^2`` (T-0094b criterion 2, ruling P9: the rows guard the task's diff; immutability of locked scripts is
#: by convention, not by these rows).
T0084D_MERGE = "2e89e87"
T0084D_BASE = "9e84f21"
T0084D_TIP = "9abdb3b"
#: Power pair (T-0094b ruling P2): T-0082b's 59ec87c (parent 78b5698) added the replay script.
C6_PRIOR_PARENT = "78b5698"
C6_PRIOR = "59ec87c"


@pytest.mark.parametrize("rel", CRITERION_6)
def test_criterion_6_files_are_byte_identical_to_the_fork_point(rel: str) -> None:
    """Criterion 6: ``t0080_toolcall.sh`` and ``t0082_replay.sh`` are byte-identical, and ``thresholds.py`` is
    unchanged, between T-0084d's BASE (the fork point) and TIP (T-0094b: a closed range, never the working tree)."""
    assert T0084D_BASE == BRANCH_BASE
    gr.prove_range(T0084D_BASE, T0084D_TIP, T0084D_MERGE)
    changed = gr.changed(T0084D_BASE, T0084D_TIP, rel)
    assert changed == [], f"{rel} changed in {T0084D_BASE}..{T0084D_TIP}"


def test_the_criterion_6_guard_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over 59ec87c's parent and 59ec87c the same predicate over ``CRITERION_6``
    reports the replay script, reached through the tuple; a predicate that reports nothing fails here."""
    gr.require_commits(C6_PRIOR_PARENT, C6_PRIOR)
    assert gr.is_ancestor(C6_PRIOR, T0084D_BASE)
    replay = next(p for p in CRITERION_6 if p.endswith("_replay.sh"))
    assert gr.changed(C6_PRIOR_PARENT, C6_PRIOR, *CRITERION_6) == [replay]
