"""Fake ``nvidia-smi`` and ``docker`` executables on ``PATH``, for the launcher's start guard.

``services/vllm.sh`` asks the driver one question before it starts the container (T-0058
criterion 3): *is there enough free device memory for the utilisation the argv will demand?*
Testing that guard needs the answer to be chosen by the test, not by whichever box the suite runs
on — a laptop with no driver, CI with no GPU, and the 5090 with an engine already resident would
otherwise each produce a different verdict. So the rows put a **fake** ``nvidia-smi`` first on
``PATH`` and drive the guard from its output.

Two properties this module exists to guarantee, both safety rather than convenience:

* **Nothing real starts.** A fake ``docker`` is installed first on ``PATH`` in every row and
  writes a *sentinel* file when it is invoked. The rows run the launcher under
  ``ASTRA_DRY_RUN=1``, where ``services/vllm.sh`` replaces ``docker`` with ``printf`` and prints
  the argv it would have run, so the sentinel must never appear; if a future edit lost that
  branch, the fake docker catches it instead of the real one starting a 26 B model.
* **The script under test is the repository's.** :func:`launcher_tree` copies
  ``services/vllm.sh`` byte-for-byte into a temporary tree whose layout the script expects
  (``$ROOT/plugin/astra_vllm_plugin/dist/*.whl`` — the launcher refuses with "no plugin wheel"
  when the wheel is missing, and this repo does not keep built wheels in git). Only the
  surroundings are faked.

The output shapes are ``nvidia-smi``'s CSV surface, not an invention: a header line naming each
queried field with its unit in brackets, one row per device (or per compute app), units inside
the cells unless ``nounits`` is in ``--format``. ``tests/fixtures/box/nvidia_smi_*_coldboot.csv``
are verbatim recordings of the same two queries on the 5090 box on 2026-09-16, and
``tests/contract/test_vllm_sh_start_guard.py`` carries a ``gpu``-marked row asserting the real
``nvidia-smi`` still answers in this shape (``.claude/rules/tests.md``: a behaviour added to a
fake needs a real-service row).
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VLLM_SH = REPO / "services" / "vllm.sh"

#: The failure modes criterion 4 requires the guard to survive by **proceeding**: a box whose
#: driver query is missing, broken, silent, or answering in a shape the guard did not expect.
#: ``missing`` installs no file at all (``nvidia-smi`` is simply not on ``PATH``).
SMI_MODES = ("ok", "missing", "exit1", "empty", "na", "units")

#: The two compute apps recorded at 2026-09-16 22:17 (``tests/fixtures/box/MANIFEST.md``):
#: Chrome's GPU process and the recogniser. The Chrome argv is abbreviated here — the *parser*
#: rows use the verbatim fixture; these rows only need a holder list the guard can print.
#: The recogniser's pid and Chrome's, as recorded. A row asserts the pid because it is a figure
#: the launcher can only have got from the driver: the recogniser's *name* also appears in the
#: refusal's fixed remedy text, so it proves nothing about the process list (ruling A17).
CHROME_PID = 11419
RECOGNISER_PID = 3380309
RECORDED_PROCESSES = (
    (CHROME_PID, "/opt/google/chrome/chrome --type=gpu-process --ozone-platform=x11", 50),
    (RECOGNISER_PID, "moshi-server", 3180),
)

_SMI_TEMPLATE = '''#!/usr/bin/env python3
"""A fake nvidia-smi (tests/fakes/gpu_cli.py). Answers from a recorded configuration."""
import sys

CFG = {config}

args = sys.argv[1:]
if CFG["log"]:
    with open(CFG["log"], "a") as handle:
        handle.write("\\x00".join(args) + "\\n")

mode = CFG["mode"]
if mode == "exit1":
    sys.stderr.write("Failed to initialize NVML: Driver/library version mismatch\\n")
    sys.exit(1)
if mode == "empty":
    sys.exit(0)

kind = None
fields = []
fmt = "csv"
for arg in args:
    key, _, value = arg.partition("=")
    if key == "--query-gpu":
        kind, fields = "gpu", [f.strip() for f in value.split(",") if f.strip()]
    elif key == "--query-compute-apps":
        kind, fields = "apps", [f.strip() for f in value.split(",") if f.strip()]
    elif key == "--format":
        fmt = value
flags = [part.strip() for part in fmt.split(",")]
noheader = "noheader" in flags
nounits = "nounits" in flags

GPU = {{
    "memory.total": CFG["total_mib"],
    "memory.used": CFG["used_mib"],
    "memory.free": CFG["free_mib"],
}}


def cell(value):
    if mode == "na":
        return "[N/A]"
    if mode == "units":                       # a driver answering in a unit nobody expected
        return "%.2f GiB" % (value / 1024.0)
    return str(value) if nounits else "%d MiB" % value


def header(name):
    if nounits or mode == "units":
        return name
    if name.startswith("memory") or name in ("used_gpu_memory", "used_memory"):
        return ("used_gpu_memory" if name == "used_memory" else name) + " [MiB]"
    return name


if kind == "gpu":
    if not noheader:
        print(", ".join(header(f) for f in fields))
    print(", ".join(cell(GPU[f]) if f in GPU else "[N/A]" for f in fields))
elif kind == "apps":
    if not noheader:
        print(", ".join(header(f) for f in fields))
    for pid, name, used in CFG["processes"]:
        # nvidia-smi names this field used_gpu_memory; `used_memory` is the spelling
        # docs/tasks/T-0058.md criterion 3 uses, and both answer here.
        row = {{"pid": str(pid), "process_name": name,
               "used_gpu_memory": cell(used), "used_memory": cell(used)}}
        print(", ".join(row.get(f, "[N/A]") for f in fields))
else:
    print("fake nvidia-smi (tests/fakes/gpu_cli.py), mode=%s" % mode)
'''

_DOCKER_TEMPLATE = """#!/usr/bin/env bash
# A fake `docker` (tests/fakes/gpu_cli.py). Invoking it is the failure: it records the argv in a
# sentinel file so a test can assert the launcher never reached the container.
printf '%s\\n' "$@" >> {sentinel}
exit 0
"""


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def install_nvidia_smi(
    directory: Path,
    *,
    mode: str = "ok",
    free_mib: int = 28391,
    total_mib: int = 32607,
    used_mib: int = 3696,
    processes: tuple[tuple[int, str, int], ...] = RECORDED_PROCESSES,
    log: Path | None = None,
) -> Path | None:
    """Put a fake ``nvidia-smi`` in ``directory``; ``mode="missing"`` installs nothing.

    ``log``, when given, receives one NUL-joined argv per invocation, so a row can assert *which*
    questions the guard asked (and that it asked at all).
    """
    assert mode in SMI_MODES, f"unknown fake nvidia-smi mode {mode!r}; known: {SMI_MODES}"
    directory.mkdir(parents=True, exist_ok=True)
    if mode == "missing":
        return None
    # ``repr`` of a dict, not ``json.dumps``: the template is a *Python* script, and JSON's
    # ``null`` is a NameError there.
    config = repr(
        {
            "mode": mode,
            "free_mib": free_mib,
            "total_mib": total_mib,
            "used_mib": used_mib,
            "processes": [list(row) for row in processes],
            "log": str(log) if log else None,
        }
    )
    return _write_executable(directory / "nvidia-smi", _SMI_TEMPLATE.format(config=config))


def install_docker_sentinel(directory: Path, sentinel: Path) -> Path:
    """A fake ``docker`` whose invocation is recorded in ``sentinel`` (which must not appear)."""
    directory.mkdir(parents=True, exist_ok=True)
    return _write_executable(directory / "docker", _DOCKER_TEMPLATE.format(sentinel=str(sentinel)))


def launcher_tree(root: Path) -> Path:
    """A temporary tree holding a byte-identical copy of ``services/vllm.sh`` and a fake wheel.

    The script derives ``ROOT`` from its own location and refuses to start when no plugin wheel
    exists under ``$ROOT/plugin/astra_vllm_plugin/dist`` (built wheels are not in git), so the
    copy carries an empty file with a wheel's name. Nothing else about the layout is invented:
    ``services/`` is copied as it is in the repository, which is what decides whether
    ``--generation-config`` appears in the argv.
    """
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "services", root / "services", dirs_exist_ok=True)
    copied = root / "services" / "vllm.sh"
    assert copied.read_bytes() == VLLM_SH.read_bytes(), "the copy must be the repository's script"
    dist = root / "plugin" / "astra_vllm_plugin" / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "astra_vllm_plugin-0.0.0-py3-none-any.whl").write_bytes(b"")
    return copied


#: The tools the launcher may reach besides the fakes. ``PATH`` is built from this list rather
#: than from ``/usr/bin``, for one reason: **this box has a real ``nvidia-smi``**. With the system
#: directories on ``PATH``, the "nvidia-smi is absent" case of criterion 4 silently becomes "the
#: guard asked the 5090 instead", the row passes for the wrong reason, and the box's own free
#: memory decides a unit test. The same exclusion guarantees that no row here can reach a real
#: container runtime whatever the script does. If a guard legitimately needs a tool that is not
#: here, add it to this list — do not put ``/usr/bin`` back.
TOOLBOX = (
    "bash",
    "sh",
    "ls",
    "cat",
    "printf",
    "echo",
    "basename",
    "dirname",
    "head",
    "tail",
    "tr",
    "cut",
    "sed",
    "grep",
    "awk",
    "gawk",
    "mawk",
    "sort",
    "wc",
    "expr",
    "bc",
    "numfmt",
    "env",
    "readlink",
    "mktemp",
    "date",
    "stat",
    "id",
    "uname",
    "true",
    "false",
    "test",
    "sleep",
    "timeout",
    "tee",
    "xargs",
    "find",
    "python3",
)


def toolbox(root: Path) -> Path:
    """A ``PATH`` directory holding only :data:`TOOLBOX`, by symlink. No driver, no runtime."""
    root.mkdir(parents=True, exist_ok=True)
    for name in TOOLBOX:
        found = shutil.which(name, path="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
        link = root / name
        if found and not link.exists():
            link.symlink_to(found)
    return root


def run_launcher(
    script: Path,
    *,
    path_dirs: list[Path],
    env: dict[str, str] | None = None,
    timeout_s: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Run the copied launcher with ``path_dirs`` first on ``PATH`` and ``ASTRA_DRY_RUN=1``.

    ``PATH`` is the fakes plus :func:`toolbox` — never the system directories, so "nvidia-smi is
    absent" really is absent (see :data:`TOOLBOX`). ``ASTRA_DRY_RUN=1`` is always set: criterion 5
    says the guard must fire on that path too, and it is the only setting in which running the
    engine's launcher inside a test cannot start the engine.
    """
    tools = toolbox(script.parent.parent / ".toolbox")
    environment = {
        "PATH": ":".join([*(str(p) for p in path_dirs), str(tools)]),
        "HOME": str(script.parent.parent),
        "ASTRA_DRY_RUN": "1",
        **(env or {}),
    }
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=environment,
        cwd=str(script.parent.parent),
    )
