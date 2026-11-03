"""T-0082b criterion 1: the locked script ``scripts/locked/t0082_replay.sh``, hermetically (fast suite, no GPU, no
engine, no request). Harness as ``tests/unit/test_cd_toolcall_locked_script.py`` (T-0080b criterion 6, the pattern the
task names): the script is copied into a throwaway top-level directory and run with a stub ``PATH`` directory that
shadows every command able to reach an engine, a container or the GPU (``STUBBED``); ``FORBIDDEN`` commands fail
loudly. Each stub appends its argv to a call log. The copy has no ``services/`` directory, so even an unstubbed engine
start has nothing to run.

CLI (criterion 1): ``t0082_replay.sh --stage dev|test --pass1-from PATH --pass1-sha256 HEX``. Conventions (T-0071b
C11): a refusal prints ``REFUSED`` and exits 2 before ``make``, the boot or the driver; a failure before the driver
prints ``FAIL``; the driver's status is the script's (``driver exit N``); ``nvidia-smi`` never decides the exit code.
The out-dir is ``logs/t0082-replay-<stage>-<SHA>`` and is refused if it exists; no ``selection-dev.json`` is needed.

Pins where the task is silent (reported): the script checks the artifact's sha256 itself, before the boot (the
"sha" refusal of the task's test list), with a command outside the stubbed set (``sha256sum`` or ``python3``, not
``uv``); a missing ``--pass1-from`` file is refused the same way. Cells are not pinned (the task gives no ``--cell``
to the locked script).

Regime: CPU; bash; the stubbed git reports HEAD ``abc1234abc1234abc1234abc1234abc1234abc1`` (short ``abc1234``).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
LOCKED = REPO / "scripts" / "locked" / "t0082_replay.sh"
T0080 = REPO / "scripts" / "locked" / "t0080_toolcall.sh"
BRANCH_BASE = "a8757f9"  # T-0082b's fork point (main with T-0082a merged)
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
ARTIFACT_TEXT = '{"kind": "regime", "stage": "test"}\n{"kind": "trailer", "stopped": null}\n'


def _script() -> Path:
    assert LOCKED.is_file(), f"T-0082b not implemented: {LOCKED.relative_to(REPO)} does not exist"
    return LOCKED


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f'#!/bin/bash\necho "{name} $*" >> "$T_CALLS"\n{body}\n')
    p.chmod(0o755)


def _artifact(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "evidence" / "20261001-toolcall-test.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(ARTIFACT_TEXT)
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


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
    call log). ``out_dir_exists`` names a stage whose ``logs/t0082-replay-<stage>-<SHA>`` already exists."""
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
        (top / "logs" / f"t0082-replay-{out_dir_exists}-{SHORT}").mkdir(parents=True)
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


def _valid(tmp_path: Path, stage: str = "test") -> tuple[list[str], Path, str]:
    src, sha = _artifact(tmp_path)
    return ["--stage", stage, "--pass1-from", str(src), "--pass1-sha256", sha], src, sha


# --------------------------------------------------------------------------- shape


def test_script_parses() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_script())], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_happy_path_wheel_boot_health_then_one_replay_driver_without_a_selection(
    tmp_path: Path, stage: str
) -> None:
    """Criterion 1: ``make plugin-wheel``, the engine boot, ``/health``, then ``scripts/gpu/cd_toolcall.py`` exactly
    once with ``--stage``, ``--pass1-from`` and ``--pass1-sha256`` as given, a loopback base URL, the out-dir
    ``logs/t0082-replay-<stage>-<SHA>`` and ``--expected-commit`` HEAD; no ``--selection`` (none exists in the
    copy), exit 0."""
    args, src, sha = _valid(tmp_path, stage)
    rc, output, uv_calls, calls = run_locked(tmp_path, args)
    assert rc == 0, output
    drivers = _driver_calls(uv_calls)
    assert len(drivers) == 1, f"driver calls: {uv_calls}\n{output}"
    argv = drivers[0]
    assert _flag(argv, "--stage") == stage, argv
    assert _flag(argv, "--pass1-from") is not None
    assert Path(str(_flag(argv, "--pass1-from"))).resolve() == src.resolve(), argv
    assert _flag(argv, "--pass1-sha256") == sha, argv
    assert _flag(argv, "--selection") is None, argv
    assert "--cell" not in argv and not any(a.startswith("--cell=") for a in argv), (
        "R2: the locked script passes no --cell (the CLI's REPLAY_CELLS apply)",
        argv,
    )
    out = _flag(argv, "--out-dir")
    assert out is not None and out.rstrip("/").endswith(f"logs/t0082-replay-{stage}-{SHORT}"), argv
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
    # the engine is backgrounded (``setsid ... &``), so the setsid stub's own log line may land after curl's: only
    # the orders the script makes deterministic are pinned (make finishes before the boot is spawned; health and
    # the driver run in the foreground)
    assert wheel[0] < boot[0], calls
    assert wheel[0] < health[0] < first_uv, calls


def test_the_engine_is_started_engine_only_with_services_vllm_sh() -> None:
    """Criterion 1: engine-only ``services/vllm.sh``, never ``astra up`` (code lines only)."""
    code = "\n".join(_code_lines(_script().read_text()))
    assert "services/vllm.sh" in code
    assert not re.search(r"\bastra\s+up\b", code), "the full stack is started, not the engine alone"


def test_every_nvidia_smi_pipeline_ends_with_or_true() -> None:
    """Criterion 1: ``nvidia-smi ... || true``."""
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
    """Criterion 1, the real exit code: the script exits with the driver's code (a driver refusal, 2, included),
    prints ``driver exit <code>``, and a failing ``nvidia-smi`` neither stops the launch nor replaces the code."""
    args, _, _ = _valid(tmp_path)
    rc, output, uv_calls, _ = run_locked(
        tmp_path, args, driver_exit=driver_exit, smi_fails=smi_fails
    )
    assert len(_driver_calls(uv_calls)) == 1, f"the driver did not run:\n{output}"
    assert rc == driver_exit, output
    assert f"driver exit {driver_exit}" in output


# --------------------------------------------------------------------------- refusals


def _refusal_args(tmp_path: Path, case: str) -> tuple[list[str], dict[str, Any]]:
    args, src, sha = _valid(tmp_path)
    if case == "no_stage":
        return ["--pass1-from", str(src), "--pass1-sha256", sha], {}
    if case == "bad_stage":
        return ["--stage", "prod", "--pass1-from", str(src), "--pass1-sha256", sha], {}
    if case == "no_pass1_from":
        return ["--stage", "test", "--pass1-sha256", sha], {}
    if case == "no_pass1_sha256":
        return ["--stage", "test", "--pass1-from", str(src)], {}
    if case == "missing_artifact":
        return ["--stage", "test", "--pass1-from", str(src) + ".gone", "--pass1-sha256", sha], {}
    if case == "sha_mismatch":
        wrong = hashlib.sha256(ARTIFACT_TEXT.encode() + b"x").hexdigest()
        return ["--stage", "test", "--pass1-from", str(src), "--pass1-sha256", wrong], {}
    if case == "unknown_flag":
        return [*args, "--selection", "x.json"], {}
    kw: dict[str, dict[str, Any]] = {
        "dirty": {"dirty": True},
        "out_dir_exists": {"out_dir_exists": "test"},
        "port_bound": {"port_bound": True},
        "containers": {"containers": True},
        "vllm_sh_changed": {"vllm_drift": True},
    }
    return args, kw[case]


REFUSALS = [
    "no_stage",
    "bad_stage",
    "no_pass1_from",
    "no_pass1_sha256",
    "missing_artifact",
    "sha_mismatch",
    "unknown_flag",
    "dirty",
    "out_dir_exists",
    "port_bound",
    "containers",
    "vllm_sh_changed",
]


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_exit_2_before_anything_runs(tmp_path: Path, case: str) -> None:
    """Criterion 1/2: missing or bad flags, a missing artifact, an artifact whose sha256 is not ``--pass1-sha256``, a
    dirty tree, an existing ``logs/t0082-replay-<stage>-<SHA>`` (a stage runs once per SHA), a bound Astra port,
    running containers and ``services/vllm.sh`` changed since 806da12 -> ``REFUSED``, exit 2, before the wheel
    build, the boot, the health check or the driver."""
    args, kw = _refusal_args(tmp_path, case)
    rc, output, uv_calls, calls = run_locked(tmp_path, args, **kw)
    assert rc == 2, (rc, output)
    assert "REFUSED" in output, output
    assert _driver_calls(uv_calls) == [], "the driver ran"
    started = [c for c in calls if c.split(" ", 1)[0] in ("make", "setsid", "curl", "uv")]
    assert started == [], f"commands ran before the refusal: {started}"


def test_an_out_dir_of_the_other_stage_does_not_refuse(tmp_path: Path) -> None:
    """The other side of the once-per-SHA refusal (a guard too wide): ``logs/t0082-replay-dev-<SHA>`` existing does
    not refuse ``--stage test``; the driver runs once."""
    args, _, _ = _valid(tmp_path, "test")
    rc, output, uv_calls, _ = run_locked(tmp_path, args, out_dir_exists="dev")
    assert rc == 0, output
    assert len(_driver_calls(uv_calls)) == 1


def test_a_t0080_out_dir_of_this_sha_does_not_refuse(tmp_path: Path) -> None:
    """The replay's out-dir is its own: an existing ``logs/t0080-toolcall-test-<SHA>`` (the live stage's evidence)
    does not refuse the replay."""
    args, _, _ = _valid(tmp_path, "test")
    rc, output, uv_calls, _ = run_locked(tmp_path, args, pre_dirs=[f"t0080-toolcall-test-{SHORT}"])
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
    args, _, _ = _valid(tmp_path)
    rc, output, uv_calls, _ = run_locked(tmp_path, args, **EARLY_FAILURES[case])
    assert rc != 0, output
    assert _driver_calls(uv_calls) == [], "the driver ran"
    assert RECOGNISABLE.search(output), f"no recognisable exit line in:\n{output}"


# --------------------------------------------------------------------------- negative


#: T-0082b's merge commit b6166c3; BASE = ``git merge-base b6166c3^1 b6166c3^2`` = ``BRANCH_BASE`` (a8757f9), TIP =
#: ``b6166c3^2`` (T-0094b criterion 2, ruling P9).
T0082B_MERGE = "b6166c3"
T0082B_BASE = "a8757f9"
T0082B_TIP = "8fe86eb"
NEGATIVE = ("scripts/locked/t0080_toolcall.sh",)
#: Power pair (T-0094b ruling P2): T-0080b's 276001d (parent 5770faa) added the script.
NEGATIVE_PRIOR_PARENT = "5770faa"
NEGATIVE_PRIOR = "276001d"


def test_t0080_toolcall_sh_is_unchanged() -> None:
    """Criterion 6 (negative): ``scripts/locked/t0080_toolcall.sh`` is byte-identical between T-0082b's BASE (the
    branch base) and TIP (its once-per-SHA rule is what T-0080c criterion 7 rests on; T-0094b: a closed range, never
    the working tree). Green before the implementation by construction."""
    assert T0082B_BASE == BRANCH_BASE and (str(T0080.relative_to(REPO)),) == NEGATIVE
    gr.prove_range(T0082B_BASE, T0082B_TIP, T0082B_MERGE)
    assert gr.changed(T0082B_BASE, T0082B_TIP, *NEGATIVE) == []


def test_the_negative_guard_has_power() -> None:
    """T-0094b criterion 5 (adversarial): over 276001d's parent and 276001d the same predicate reports the script,
    reached through the tuple; a predicate that reports nothing fails here."""
    gr.require_commits(NEGATIVE_PRIOR_PARENT, NEGATIVE_PRIOR)
    assert gr.is_ancestor(NEGATIVE_PRIOR, T0082B_BASE)
    assert gr.changed(NEGATIVE_PRIOR_PARENT, NEGATIVE_PRIOR, *NEGATIVE) == list(NEGATIVE)
