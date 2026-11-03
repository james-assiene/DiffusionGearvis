"""The evidence extract the CD container wrappers print instead of a raw tail (T-0077c a683f8e, generalised for the
T-0073 wrappers): every ``ASTRA_*`` marker line, pytest's FAILURES/ERRORS sections and short test summary, and the
final summary line; captured ``DEBUG`` log lines inside those sections are dropped (counted). A red row's message
survives whatever floods the run's output."""

from __future__ import annotations

import re

_SECTION = re.compile(r"^=+ (.+?) =+$")
_KEEP_SECTIONS = ("FAILURES", "ERRORS", "short test summary info")


def evidence(stdout: str, marker: str = "ASTRA_") -> str:
    keep: list[str] = []
    dropped = 0
    section = None
    for ln in stdout.splitlines():
        m = _SECTION.match(ln.strip())
        if m:
            name = m.group(1).strip()
            section = name if any(name.startswith(k) for k in _KEEP_SECTIONS) else None
            if section or re.search(r"\d+ (passed|failed)", name):
                keep.append(ln)
            continue
        if marker in ln:
            keep.append(ln)
        elif section:
            if ln.lstrip().startswith("DEBUG "):
                dropped += 1
            else:
                keep.append(ln)
    if dropped:
        keep.append(f"[evidence: {dropped} captured DEBUG log lines dropped]")
    return "\n".join(keep)


def flood_run_stdout(tmp_path: object, message: str) -> tuple[int, str]:
    """A real ``-rA`` pytest run (the containers' flags) of a passing row printing an ``ASTRA_`` line and a red row that
    logs 5 000 ``torch.__trace`` DEBUG lines and fails with ``message``: (exit code, stdout)."""
    import subprocess
    import sys
    from pathlib import Path

    t = Path(str(tmp_path)) / "test_flood.py"
    t.write_text(
        "import logging\n"
        "def test_ok():\n"
        "    print('ASTRA_TIER2_MATMUL_PRECISION high')\n"
        "def test_red():\n"
        "    lg = logging.getLogger('torch.__trace')\n"
        "    lg.setLevel(logging.DEBUG)\n"
        "    for i in range(5000):\n"
        "        lg.debug('trace %d %s', i, 'x' * 60)\n"
        f"    assert False, {message!r}\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-rA",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "--noconftest",
            "-o",
            "addopts=",
            "--log-level=DEBUG",
            str(t),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode, proc.stdout
