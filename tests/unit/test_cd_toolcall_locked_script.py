"""T-0080b criterion 6: the locked script ``scripts/locked/t0080_toolcall.sh``, hermetically (fast suite, no GPU, no
engine, no request).

The script is copied into a throwaway top-level directory and run with a stub ``PATH`` directory that shadows every
command able to reach an engine, a container or the GPU (``STUBBED``); ``FORBIDDEN`` commands fail loudly. Each stub
appends its argv to a call log, so the rows see what ran and in which order. The copy has no ``services/``
directory, so even an unstubbed engine start has nothing to run. Conventions are the repository's locked-script
ones (``scripts/locked/t0071b_calib.sh``, T-0071b C11): a refusal prints ``REFUSED`` and exits 2 before anything
runs; a failure before the driver prints ``FAIL``; the driver's status is the script's (real exit-code capture,
printed as ``driver exit N``); ``nvidia-smi`` never decides the exit code.

Assumed CLI (criterion 6 names ``--stage dev|test`` and "a ``selection-dev.json`` from the same SHA's dev stage is
given"; how it is given is not spelled out -- see the report): ``t0080_toolcall.sh --stage dev`` and
``t0080_toolcall.sh --stage test --selection <path/to/selection-dev.json>``. The rows' selection files name their
commit in the content (``commit``, ``expected_commit``, ``head``) *and* sit in a per-SHA dev directory, so a script
that checks either the content or the path discriminates; a file from another commit differs in both.

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

REPO = Path(__file__).resolve().parents[2]
LOCKED = REPO / "scripts" / "locked" / "t0080_toolcall.sh"
HEAD = "abc1234abc1234abc1234abc1234abc1234abc1"
SHORT = "abc1234"
OTHER = "def5678def5678def5678def5678def5678def5"

#: commands the harness replaces; anything reaching a real engine, container or GPU is among them
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
#: commands the script must not call at all in the harness (they fail loudly)
FORBIDDEN = ("sudo", "systemctl", "podman", "vllm", "wget", "astra")
RECOGNISABLE = re.compile(r"(?m)^.*\b(driver exit \d+|REFUSED|FAIL)\b")


def _script() -> Path:
    assert LOCKED.is_file(), f"T-0080b not implemented: {LOCKED.relative_to(REPO)} does not exist"
    return LOCKED


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f'#!/bin/bash\necho "{name} $*" >> "$T_CALLS"\n{body}\n')
    p.chmod(0o755)


def _selection(top: Path, sha_full: str) -> Path:
    """A dev-stage selection file of commit ``sha_full``, in that commit's per-SHA dev directory."""
    d = top / "logs" / f"t0080-toolcall-dev-{sha_full[:7]}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "selection-dev.json"
    content = {
        "stage": "dev",
        "rule": "stage1",
        "commit": sha_full,
        "expected_commit": sha_full,
        "head": sha_full,
        "choice": "T4/named/perstep_mar",
    }
    path.write_text(json.dumps(content, indent=2) + "\n")
    return path


def run_locked(
    tmp_path: Path,
    args: list[str],
    *,
    driver_exit: int = 0,
    smi_fails: bool = False,
    dirty: bool = False,
    make_exit: int = 0,
    curl_exit: int = 0,
    selection_of: str | None = None,
    out_dir_exists: bool = False,
    port_bound: bool = False,
    vllm_drift: bool = False,
    engine_alive: bool = False,
) -> tuple[int, str, list[list[str]], list[str]]:
    """Run a copy of the locked script; returns (exit code, console output, argv of every ``uv`` call, the
    ordered call log). ``selection_of`` writes a selection file of that commit and appends
    ``--selection <path>`` to ``args``."""
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
    if out_dir_exists:  # a previous run of this stage at this SHA left its evidence directory
        stage = args[args.index("--stage") + 1] if "--stage" in args else "dev"
        (top / "logs" / f"t0080-toolcall-{stage}-{SHORT}").mkdir(parents=True)
    args = list(args)
    if selection_of is not None:
        args += ["--selection", str(_selection(top, selection_of))]
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
    for quiet in ("docker", "sg", "sleep", "uptime"):
        _stub(bin_dir, quiet, "exit 0")
    _stub(
        bin_dir,
        "ss",
        'if [ -n "$T_PORT_BOUND" ]; then echo "LISTEN 0 4096 127.0.0.1:8001 0.0.0.0:*"; fi\nexit 0',
    )
    # the engine start: exits at once (an engine that died during boot) unless the row keeps it alive; the real
    # sleep is reached by absolute path so the stubbed one cannot shorten it
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


# --------------------------------------------------------------------------- shape


def test_script_parses() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_script())], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_dev_stage_builds_the_wheel_boots_checks_health_then_runs_the_driver_once(
    tmp_path: Path,
) -> None:
    """Criterion 6, the happy path in order: ``make plugin-wheel``, the engine boot, ``/health``, then the driver
    ``scripts/gpu/cd_toolcall.py --stage dev`` exactly once, against a loopback base URL; exit 0."""
    rc, output, uv_calls, calls = run_locked(tmp_path, ["--stage", "dev"])
    assert rc == 0, output
    drivers = _driver_calls(uv_calls)
    assert len(drivers) == 1, f"driver calls: {uv_calls}\n{output}"
    argv = drivers[0]
    assert _flag(argv, "--stage") == "dev", argv
    base = _flag(argv, "--base-url")
    assert base is not None and re.match(
        r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?/", base
    ), argv
    first_uv = next(i for i, c in enumerate(calls) if c.startswith("uv ") and "cd_toolcall.py" in c)
    wheel = [i for i, c in enumerate(calls) if c.startswith("make ") and "plugin-wheel" in c]
    health = [i for i, c in enumerate(calls) if c.startswith("curl ") and "/health" in c]
    assert wheel and wheel[0] < first_uv, (
        f"make plugin-wheel did not run before the driver: {calls}"
    )
    assert health and health[0] < first_uv, f"/health was not checked before the driver: {calls}"


def test_the_engine_is_started_engine_only_with_services_vllm_sh() -> None:
    """Criterion 6: engine-only ``services/vllm.sh`` (STT, TTS and the conductor stay down): the script starts the
    engine through that file and never through ``astra up``. (Code lines only; comments are prose.)"""
    code = "\n".join(_code_lines(_script().read_text()))
    assert "services/vllm.sh" in code
    assert not re.search(r"\bastra\s+up\b", code), "the full stack is started, not the engine alone"


def test_every_nvidia_smi_pipeline_ends_with_or_true() -> None:
    """Criterion 6: ``nvidia-smi ... || true`` -- a GPU fault in the informational lines never decides the exit."""
    lines = [ln for ln in _code_lines(_script().read_text()) if "nvidia-smi" in ln]
    assert lines, "the script prints no nvidia-smi line (criterion 6 asks for one)"
    bad = [ln.strip() for ln in lines if not ln.rstrip().endswith("|| true")]
    assert bad == [], f"nvidia-smi lines without '|| true': {bad}"


# --------------------------------------------------------------------------- exit-code capture


@pytest.mark.parametrize("driver_exit", [0, 1, 3])
@pytest.mark.parametrize("smi_fails", [False, True], ids=["smi-ok", "smi-fails"])
def test_the_drivers_exit_code_is_the_scripts(
    tmp_path: Path, driver_exit: int, smi_fails: bool
) -> None:
    """Real exit-code capture: the script exits with the driver's code (not swallowed by a pipe or by ``set -e``),
    prints ``driver exit <code>``, and a failing ``nvidia-smi`` (exit 9, no output) neither stops the launch before
    the driver nor replaces its code."""
    rc, output, uv_calls, _ = run_locked(
        tmp_path, ["--stage", "dev"], driver_exit=driver_exit, smi_fails=smi_fails
    )
    assert len(_driver_calls(uv_calls)) == 1, f"the driver did not run:\n{output}"
    assert rc == driver_exit, output
    assert f"driver exit {driver_exit}" in output


# --------------------------------------------------------------------------- refusals and stage gating

REFUSALS: dict[str, tuple[list[str], dict[str, Any]]] = {
    "no_stage": ([], {}),
    "bad_stage": (["--stage", "prod"], {}),
    "dirty": (["--stage", "dev"], {"dirty": True}),
    "test_without_selection": (["--stage", "test"], {}),
    "test_with_another_shas_selection": (["--stage", "test"], {"selection_of": OTHER}),
    # review of 528d385: the refusals the implementer added, each pinned on its own
    "dev_out_dir_exists": (
        ["--stage", "dev"],
        {"out_dir_exists": True},
    ),  # a stage runs once per SHA
    "test_out_dir_exists": (["--stage", "test"], {"selection_of": HEAD, "out_dir_exists": True}),
    "port_bound": (["--stage", "dev"], {"port_bound": True}),
    "vllm_sh_changed": (["--stage", "dev"], {"vllm_drift": True}),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals_exit_2_before_anything_runs(tmp_path: Path, case: str) -> None:
    """Criterion 6 refusals: no/unknown stage, a dirty tree, and ``--stage test`` without a ``selection-dev.json``
    from this SHA's dev stage are refused with ``REFUSED`` and exit 2 before the wheel build, the boot or the driver."""
    args, kw = REFUSALS[case]
    rc, output, uv_calls, calls = run_locked(tmp_path, args, **kw)
    assert rc == 2, (rc, output)
    assert "REFUSED" in output, output
    assert _driver_calls(uv_calls) == [], "the driver ran"
    started = [c for c in calls if c.split(" ", 1)[0] in ("make", "setsid", "curl", "uv")]
    assert started == [], f"commands ran before the refusal: {started}"


def test_test_stage_runs_with_this_shas_dev_selection(tmp_path: Path) -> None:
    """The other side of the stage gate (a gate needs a row for being too wide): ``--stage test`` with the
    selection written by *this* SHA's dev stage is not refused; the driver runs once with ``--stage test``."""
    rc, output, uv_calls, _ = run_locked(tmp_path, ["--stage", "test"], selection_of=HEAD)
    assert rc == 0, output
    drivers = _driver_calls(uv_calls)
    assert len(drivers) == 1, f"driver calls: {uv_calls}\n{output}"
    assert _flag(drivers[0], "--stage") == "test", drivers[0]


EARLY_FAILURES: dict[str, dict[str, Any]] = {
    "make": {"make_exit": 2},  # the plugin wheel build fails (set -e would be silent)
    "unhealthy": {"curl_exit": 7},  # the engine never answers /health (and its process exits)
    # the engine process stays up but /health never answers: the boot loop runs out (health timeout)
    "health_timeout": {"curl_exit": 7, "engine_alive": True},
}


@pytest.mark.parametrize("case", list(EARLY_FAILURES))
def test_a_failure_before_the_driver_is_loud_and_never_runs_the_driver(
    tmp_path: Path, case: str
) -> None:
    rc, output, uv_calls, _ = run_locked(tmp_path, ["--stage", "dev"], **EARLY_FAILURES[case])
    assert rc != 0, output
    assert _driver_calls(uv_calls) == [], "the driver ran"
    assert RECOGNISABLE.search(output), f"no recognisable exit line in:\n{output}"
