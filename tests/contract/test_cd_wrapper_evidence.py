"""T-0077c: the T-0073 CD container wrappers print an evidence extract, so a red inner row keeps its message through a
torch.__trace DEBUG flood (tier 2 at a671719 lost the numerics module's 12 failure texts). Fast suite, CPU, no
container: the wrappers themselves are ``gpu``-marked as whole modules.

- per wrapper: a real ``-rA`` pytest run with a 5 000-line flood and a red row, through ``evidence``: the message,
  the ``ASTRA_`` marker line, the FAILED line and the summary survive; controls: the flood is reproduced and the old
  40 000-character tail loses the message;
- per wrapper (AST): it prints ``evidence(proc.stdout)`` and its returncode assertion carries it, with no raw tail
  longer than 8 000 characters printed unconditionally;
- the locked tier-2 script checks only these wrappers' host summary (``check_summary``, last summary line "1 passed"),
  so inner summaries printed in the extract do not move it (the script's own function, run by bash).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tests.contract._evidence import evidence, flood_run_stdout

REPO = Path(__file__).resolve().parents[2]
WRAPPERS = {
    "numerics": "tests/contract/test_vllm_cd_numerics_gpu.py",
    "repeats": "tests/contract/test_vllm_cd_cuda_gpu.py",
}
MESSAGES = {
    "numerics": "control (compensated=False) is not red on CUDA: max 0.62 ulp, 1.2% > 0.51 ulp",
    "repeats": "control: the input does not discriminate on this device",
}


@pytest.mark.parametrize("which", list(WRAPPERS))
def test_flooded_red_row_keeps_its_message(which: str, tmp_path: Path) -> None:
    msg = MESSAGES[which]
    rc, out = flood_run_stdout(tmp_path, msg)
    assert rc == 1 and out.count("DEBUG") >= 5000, out[-2000:]
    assert msg not in out[-40000:], "control: the old 40 000-character tail would have kept it"
    ev = evidence(out)
    assert msg in ev and "ASTRA_TIER2_MATMUL_PRECISION high" in ev
    assert "FAILED test_flood.py::test_red" in ev and "1 failed, 1 passed" in ev
    assert "trace 4999" not in ev and "captured DEBUG log lines dropped" in ev
    assert len(ev) < 20000, len(ev)


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


#: the function of each wrapper that runs the CUDA module in the container
RUNS = {"numerics": "test_criterion5_numerics_module_on_cuda", "repeats": "_pytest_in_container"}


@pytest.mark.parametrize("which", list(WRAPPERS))
def test_wrapper_prints_the_evidence_extract(which: str) -> None:
    """In the function that runs the CUDA module: ``evidence(proc.stdout)`` is printed, the ``returncode == 0``
    assertion's message carries it, and no 40 000/20 000-character raw tail is printed."""
    tree = ast.parse((REPO / WRAPPERS[which]).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == RUNS[which])
    printed = [c for c in _calls(fn, "print") if _calls(c, "evidence")]
    assert printed, f"{which}: {RUNS[which]} prints no evidence extract"
    rc = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assert) and "returncode == 0" in ast.unparse(n.test)
    ]
    assert len(rc) == 1 and _calls(rc[0], "evidence"), (
        f"{which}: the returncode assertion lacks the evidence"
    )
    body = ast.unparse(fn)
    assert "[-40000:]" not in body and "[-20000:]" not in body, (
        f"{which}: a raw long tail is still printed"
    )


def test_locked_check_summary_reads_the_hosts_own_line(tmp_path: Path) -> None:
    src = (REPO / "scripts" / "locked" / "t0077c_tier2.sh").read_text()
    assert 'check_inner "numerics"' not in src and 'check_inner "repeats"' not in src
    start = src.index("check_summary() {")
    fn = tmp_path / "check_summary.sh"
    fn.write_text(src[start : src.index("\n}\n", start) + 3])
    host = tmp_path / "host.txt"
    inner = "================== 2 failed, 49 passed, 15 warnings in 43.70s =================="
    for last, want in (("1 passed in 120.00s (0:02:00)", 0), ("1 failed in 120.00s (0:02:00)", 1)):
        host.write_text(
            "==================================== PASSES ====================================\n"
            f"--- container pytest evidence ---\n{inner}\n"
            f"============================== {last} ==============================\n"
        )
        proc = subprocess.run(
            ["bash", "-c", f'source "{fn}"; check_summary x "{host}" 1'],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == want, (last, proc.stdout)
