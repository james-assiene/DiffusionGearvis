"""T-0067 tier 2: the summary parser that stops a container pytest run from passing vacuously.

``tests/contract/test_vllm_cd_cuda_gpu.py`` requires the inner run to report exactly N passed and nothing but
warnings besides. These rows pin that rule on canned summary lines and, once, on real pytest output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.pytest_summary import parse_summary, require_exact

OK = [
    ("17 passed in 41.20s", 17),
    ("==================== 17 passed in 41.20s ====================", 17),
    ("3 passed, 2 warnings in 65.01s (0:01:05)", 3),
    ("========= 3 passed, 1 warning in 9.10s =========", 3),
]

BAD = [
    ("16 passed, 1 skipped in 40.00s", 17),
    ("17 passed, 1 xfailed in 40.00s", 17),
    ("17 passed, 1 xpassed in 40.00s", 17),
    ("17 passed, 1 deselected in 40.00s", 17),
    ("17 passed, 1 error in 40.00s", 17),
    ("16 passed, 1 failed in 40.00s", 17),
    ("18 passed in 40.00s", 17),
    ("no tests ran in 0.01s", 17),
    ("======= 17 skipped in 1.00s =======", 17),
]


@pytest.mark.parametrize(("line", "n"), OK)
def test_exact_pass_accepted(line: str, n: int) -> None:
    require_exact(f"collected {n} items\n...\n{line}\n", n)


@pytest.mark.parametrize(("line", "n"), BAD)
def test_anything_but_exact_passes_rejected(line: str, n: int) -> None:
    with pytest.raises(AssertionError):
        require_exact(f"...\n{line}\n", n)


def test_counts_parsed_by_kind() -> None:
    assert parse_summary("= 1 failed, 16 passed, 1 skipped, 2 warnings, 1 error in 3.00s =") == {
        "failed": 1,
        "passed": 16,
        "skipped": 1,
        "warnings": 2,
        "errors": 1,
    }
    assert parse_summary("no tests ran in 0.01s") == {}


def test_no_summary_line_rejected() -> None:
    crash = "INTERNALERROR> Traceback (most recent call last):\nINTERNALERROR> RuntimeError: boom\n"
    with pytest.raises(AssertionError, match="no pytest summary line"):
        require_exact(crash, 17)


def test_the_last_summary_wins_over_a_stray_earlier_one() -> None:
    # a row's captured output may contain a summary-looking line; the run's own summary is the last one
    out = "Captured stdout call\n17 passed in 1.00s\n...\n=== 16 passed, 1 skipped in 40.00s ===\n"
    with pytest.raises(AssertionError):
        require_exact(out, 17)


def test_real_pytest_output(tmp_path: Path) -> None:
    """Oracle: pytest's own output, not a hand-typed line. Two passes and a skip must be rejected for 3, and
    the same file with the skip removed accepted."""
    body = "import pytest\ndef test_a(): pass\ndef test_b(): pass\n"
    (tmp_path / "test_skip.py").write_text(body + "def test_c(): pytest.skip('x')\n")
    (tmp_path / "test_ok.py").write_text(body + "def test_c(): pass\n")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}

    def run(name: str) -> str:
        argv = [sys.executable, "-m", "pytest", "-rA", "-p", "no:cacheprovider", "--noconftest"]
        argv += ["-c", os.devnull, "--rootdir", str(tmp_path), str(tmp_path / name)]
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False, env=env
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc.stdout

    with pytest.raises(AssertionError):
        require_exact(run("test_skip.py"), 3)
    assert require_exact(run("test_ok.py"), 3)["passed"] == 3
