"""T-0077c criteria 2-5, 7 and 8 on the 5090 (tier 2, ``-m gpu``, ``/launch``): ``tests/cd/cuda/test_cd_logmatmul_cuda.py``
inside the pinned container, **twice** (two container processes), with equal output digests (T-0077b wrapper form).

T-0073 wrapper pattern: ``docker run --rm --gpus all --network none``, the repository read-only at ``/repo``, no
``/hostsite``, ``PYTHONDONTWRITEBYTECODE=1``, T-0067's runner (5090 preflight, ``ASTRA_TIER2_REGIME`` line),
``--noconftest -o addopts= -p no:cacheprovider``, ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, ``ASTRA_CD_REQUIRE_CUDA=1``.
The four dev automata of the A/B are compiled on the host (``scripts/gpu/cd_microbench.write_payloads``, which needs
the ``cd`` group: run this row with ``uv run --group cd pytest ... -m gpu``) and mounted read-only at
``/cdpayloads`` (``ASTRA_CD_PAYLOADS``); the same payload directory serves both runs.

Pass = per run: exit 0, the runner on a 5090, the inner summary exactly ``LOGMATMUL_CUDA_PASSED`` (K_c) passed and nothing
else; the ``ASTRA_T0077C_DIGEST`` set of each run complete (every case × field of ``DIGEST_CASES``, one digest
each) and the two runs' digests equal; the image's triton is 3.7.1 (reported in the regime line); the host tree
clean.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc

REPO = Path(__file__).resolve().parents[2]
MODULE = "tests/cd/cuda/test_cd_logmatmul_cuda.py"
RUNNER = "/repo/tests/container_scripts/astra_cd_tier2_runner.py"
RUNNER_REGIME = "ASTRA_TIER2_REGIME"
PAYLOADS = "/cdpayloads"
#: K_c, rows of ``test_cd_logmatmul_cuda.py`` (its docstring): criterion 2 2, 3a 8, 3b 2, 4 8, 5 11 + 4, 7 4, 8 4
LOGMATMUL_CUDA_PASSED = 43
BUCKETS = (64, 128, 256, 512)
CORE = (
    "s1_b64",
    "s2_b64",
    "s31_b64_dfa",
    "s31_b64_nfa_leff",
    "s32_b32",
    "s91_b128_nfa",
    "s91_b128_dfa_leff",
    "s205_b256",
    "last_state_s31_b64",
    "last_state_s91_b128",
    "empty_s31_b64",
)
REGIMES = ("wide", "gauged")
INNERS = (32, 91, 205, 400, 512)
VARIANTS = ("comp", "plain")
FIELDS = ("log_z", "q", "entropy", "log_a", "log_b")
EXPECTED_DIGESTS = frozenset(
    [f"lm:{r}{i}-{v}" for r in REGIMES for i in INNERS for v in VARIANTS]
    + [f"lm:dev:b{b}-{v}" for b in BUCKETS for v in VARIANTS]
    + [f"cm:core:{n}.{f}" for n in CORE for f in FIELDS]
    + [f"cm:dev:b{b}.{f}" for b in BUCKETS for f in FIELDS]
)
_DIGEST = re.compile(r"ASTRA_T0077C_DIGEST (\S+) ([0-9a-f]{64})")
_REGIME = re.compile(r"ASTRA_T0077C_REGIME (\{.*\})")


_SECTION = re.compile(r"^=+ (.+?) =+$")
#: pytest sections kept whole in the evidence (the PASSES section is not: ASTRA lines are kept from it line by line)
_KEEP_SECTIONS = ("FAILURES", "ERRORS", "short test summary info")


def evidence(stdout: str) -> str:
    """What a red row needs on the host, whatever floods the run's output: every ``ASTRA_T0077C_*`` line (digests,
    bars, diffs, regime), every line of pytest's FAILURES/ERRORS sections and its short test summary, and the final
    summary line; captured ``DEBUG`` log lines inside those sections are dropped (counted)."""
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
        if "ASTRA_T0077C_" in ln:
            keep.append(ln)
        elif section:
            if ln.lstrip().startswith("DEBUG "):
                dropped += 1
            else:
                keep.append(ln)
    if dropped:
        keep.append(f"[evidence: {dropped} captured DEBUG log lines dropped]")
    return "\n".join(keep)


def run_report(i: int, stdout: str, returncode: int) -> str:
    """What the host prints for container run ``i``: the evidence extract, which keeps the inner summary line once;
    the raw tail only when the run failed. ``scripts/locked/t0077c_tier2.sh``'s ``check_inner`` counts summary lines
    in the host output (exactly one ``43 passed`` per run, runs + 1 in all), so a green run must print its inner
    summary exactly once (launch review, 2026-10-06)."""
    out = f"--- container pytest evidence, run {i} ({MODULE}) ---\n{evidence(stdout)}"
    if returncode != 0:
        out += (
            f"\n--- container pytest output tail, run {i} (exit {returncode}) ---\n{stdout[-8000:]}"
        )
    return out


def _host(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, f"{argv} exited {proc.returncode}: {proc.stderr}"
    return proc.stdout.strip()


def digests(stdout: str) -> dict[str, str]:
    """``name -> sha256`` from one run's output; a name printed twice with different digests fails."""
    out: dict[str, str] = {}
    for name, h in _DIGEST.findall(stdout):
        assert out.get(name, h) == h, f"{name} printed with two digests in one run"
        out[name] = h
    return out


def _write_payloads(dest: Path) -> dict[str, Any]:
    gpu = str(REPO / "scripts" / "gpu")
    if gpu not in sys.path:
        sys.path.insert(0, gpu)
    try:
        import cd_microbench as mb

        manifest: dict[str, Any] = mb.write_payloads(dest)
    except (ImportError, SystemExit) as exc:
        pytest.fail(
            f"cannot compile the dev automata on the host ({exc!r}); run this row with "
            "`uv run --group cd pytest tests/contract/test_vllm_cd_logmatmul_gpu.py -m gpu`"
        )
    return manifest


def _docker_argv(image: str, payloads: Path) -> list[str]:
    argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
    argv += ["-v", f"{REPO}:/repo:ro", "-v", f"{payloads}:{PAYLOADS}:ro", "-w", "/repo"]
    env = {
        "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "ASTRA_CD_TESTS": "1",
        "ASTRA_CD_REQUIRE_CUDA": "1",
        "ASTRA_CD_DEVICE": "cuda",
        "ASTRA_CD_STOCK": "image",
        "ASTRA_CD_PAYLOADS": PAYLOADS,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    args = ["-rA", "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=", MODULE]
    return [*argv, "--entrypoint", "python3", image, RUNNER, *args]


@pytest.mark.gpu
def test_logmatmul_module_twice_on_cuda_with_equal_digests() -> None:
    from tests.pytest_summary import require_exact

    image = vc.require_container()
    host = {
        "image": image,
        "driver": _host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "host_head": _host(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "host_tree_dirty": bool(_host(["git", "-C", str(REPO), "status", "--porcelain"])),
    }
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="t0077c-cdpayloads-") as tmp:
        manifest = _write_payloads(Path(tmp))
        for i in (1, 2):
            proc = vc._run(_docker_argv(image, Path(tmp)), timeout=7200)
            inner = next(
                (
                    json.loads(ln[len(RUNNER_REGIME) + 1 :])
                    for ln in proc.stdout.splitlines()
                    if ln.startswith(RUNNER_REGIME + " ")
                ),
                None,
            )
            reg = _REGIME.search(proc.stdout)
            kern = json.loads(reg.group(1)) if reg else None
            print(
                f"T-0077c run {i} regime: "
                + json.dumps({**host, "inner": inner, "kernels": kern}, sort_keys=True)
            )
            print(run_report(i, proc.stdout, proc.returncode))
            runs.append({"proc": proc, "inner": inner, "kernels": kern})
    automata = {b: (m["id"], m["n_states"]) for b, m in manifest.items()}
    print(f"T-0077c dev automata: {automata}")

    sets = []
    for i, r in enumerate(runs, 1):
        proc, inner = r["proc"], r["inner"]
        assert inner is not None, f"run {i}: no {RUNNER_REGIME} line\n{proc.stdout[-8000:]}"
        assert inner["cuda_available"] and "5090" in inner["device"], f"run {i}: {inner}"
        assert proc.returncode == 0, f"run {i}:\n{evidence(proc.stdout)}\n{proc.stderr[-4000:]}"
        require_exact(proc.stdout, LOGMATMUL_CUDA_PASSED)
        assert r["kernels"] is not None and r["kernels"]["triton"] == "3.7.1", (
            f"run {i}: image triton {r['kernels']}"
        )
        d = digests(proc.stdout)
        assert set(d) == EXPECTED_DIGESTS, (
            f"run {i}: digest set missing {sorted(EXPECTED_DIGESTS - set(d))[:10]}, "
            f"extra {sorted(set(d) - EXPECTED_DIGESTS)[:10]}"
        )
        sets.append(d)
    differ = sorted(k for k in EXPECTED_DIGESTS if sets[0][k] != sets[1][k])
    assert differ == [], f"output bits differ between the two processes: {differ}"
    assert not host["host_tree_dirty"], (
        "launch evidence from a dirty tree (CLAUDE.md: never against a dirty tree)"
    )


def test_digest_parser_and_expected_set() -> None:
    """Host-side control (fast suite, no container): the parser reads lines anywhere in a line, refuses two digests
    for one name, and the expected set is 28 entry digests + 15 marginals cases × 5 fields. The ``gpu`` row above is the only one tier 2 runs
    (host summary exactly "1 passed")."""
    h1, h2 = "a" * 64, "b" * 64
    out = f"x\nASTRA_T0077C_DIGEST cm:core:s1_b64.q {h1}\n.. ASTRA_T0077C_DIGEST lm:wide32-comp {h2}\n"
    assert digests(out) == {"cm:core:s1_b64.q": h1, "lm:wide32-comp": h2}
    with pytest.raises(AssertionError):
        digests(out + f"ASTRA_T0077C_DIGEST cm:core:s1_b64.q {h2}\n")
    assert len(EXPECTED_DIGESTS) == 20 + 8 + 55 + 20


def test_evidence_survives_a_debug_flood(tmp_path: Path) -> None:
    """A failing row's numbers reach the host: a real pytest run (``-rA``, like the container's) of a row that logs
    5 000 ``torch.__trace`` DEBUG lines, passes rows that print ``ASTRA_T0077C_*`` lines, and fails one row after
    printing its ``ASTRA_T0077C_BAR`` line, goes through ``evidence``: the BAR line, the assertion text, the digest
    line from the passing row and the short summary all survive; no DEBUG line does; the extract is small."""
    t = tmp_path / "test_flood.py"
    t.write_text(
        "import logging\n"
        "def test_ok():\n"
        "    print('ASTRA_T0077C_DIGEST lm:wide32-comp ' + 'a' * 64)\n"
        "def test_red():\n"
        "    lg = logging.getLogger('torch.__trace')\n"
        "    lg.setLevel(logging.DEBUG)\n"
        "    for i in range(5000):\n"
        "        lg.debug('trace %d %s', i, 'x' * 60)\n"
        "    print('ASTRA_T0077C_BAR comp_bars gauged inner 32 seed 1 kernel max/mean/share 2.6100/0.66000/0.4100 "
        "torch max/mean/share 2.7700/0.68000/0.4300 bar mean<=0.75,max<=2.5 FAIL')\n"
        "    assert 2.61 <= 2.5, 'gauged inner 32 seed 1: mean 0.660, max 2.610 ulp(1.0) (bar 0.75 / 2.5)'\n"
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
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert proc.stdout.count("DEBUG") >= 5000, "the flood was not reproduced (control)"
    assert "ASTRA_T0077C_BAR comp_bars gauged" not in proc.stdout[-40000:], (
        "control: the old 40 000-char tail kept it"
    )
    ev = evidence(proc.stdout)
    assert "ASTRA_T0077C_BAR comp_bars gauged inner 32 seed 1" in ev
    assert "mean 0.660, max 2.610 ulp(1.0) (bar 0.75 / 2.5)" in ev
    assert "ASTRA_T0077C_DIGEST lm:wide32-comp " + "a" * 64 in ev
    assert "FAILED test_flood.py::test_red" in ev and "1 failed, 1 passed" in ev
    assert "trace 4999" not in ev and "captured DEBUG log lines dropped" in ev
    assert len(ev) < 20000, len(ev)


LOCKED_TIER2 = REPO / "scripts" / "locked" / "t0077c_tier2.sh"


def locked_check_inner(text: str, k: int, runs: int) -> bool:
    """``check_inner`` of ``scripts/locked/t0077c_tier2.sh``, with the regexes read from the script itself (so this
    row follows the script if it changes): exactly ``runs`` lines matching the "<k> passed" summary and exactly
    ``runs + 1`` summary lines in all."""
    src = LOCKED_TIER2.read_text()
    body = src[src.index("check_inner() {") :]
    good_pat = re.search(r'good=\$\(grep -Ec "(.+?)" "\$file"\)', body)
    all_pat = re.search(r"all=\$\(grep -Ec '(.+?)' \"\$file\"\)", body)
    assert good_pat and all_pat, "check_inner's grep patterns not found in the locked script"
    good_re = re.compile(good_pat.group(1).replace("$k", str(k)).replace("\\$", "$"), re.M)
    all_re = re.compile(all_pat.group(1), re.M)
    return len(good_re.findall(text)) == runs and len(all_re.findall(text)) == runs + 1


def _inner_stdout(summary: str) -> str:
    """A container run's stdout in the inner pytest's ``-rA`` shape (abridged)."""
    return (
        'ASTRA_TIER2_REGIME {"cuda_available": true, "device": "NVIDIA GeForce RTX 5090"}\n'
        "============================= test session starts ==============================\n"
        "tests/cd/cuda/test_cd_logmatmul_cuda.py ...........................................\n"
        "==================================== PASSES ====================================\n"
        "----------------------------- Captured stdout call -----------------------------\n"
        "ASTRA_T0077C_DIGEST lm:wide32-comp " + "a" * 64 + "\n"
        "=========================== short test summary info ============================\n"
        "PASSED tests/cd/cuda/test_cd_logmatmul_cuda.py::test_criterion8_regime_line\n"
        f"{summary}\n"
    )


def _host_output(runs: list[tuple[str, int]], report: Any) -> str:
    """The host's ``-rA`` output as the locked script tees it: each run's printed report inside the PASSES (or
    FAILURES) section, then the host's own summary line."""
    lines = ["==================================== PASSES ===================================="]
    for i, (summary, rc) in enumerate(runs, 1):
        lines.append(report(i, _inner_stdout(summary), rc))
    host_ok = all(rc == 0 for _, rc in runs)
    lines.append(
        "============================== 1 passed in 300.12s (0:05:00) =============================="
        if host_ok
        else "============================== 1 failed in 300.12s (0:05:00) =============================="
    )
    return "\n".join(lines) + "\n"


GREEN = "================== 43 passed, 18 warnings in 52.69s =================="
RED = "================== 2 failed, 41 passed, 18 warnings in 52.69s =================="


def test_green_two_run_output_passes_the_locked_check_inner() -> None:
    """Launch review 2026-10-06: a fully green launch must read "2 x '43 passed'" to ``check_inner``. Control: the
    previous print format (evidence and the raw tail on every run) shows each summary twice and fails it."""
    out = _host_output([(GREEN, 0), (GREEN, 0)], run_report)
    assert locked_check_inner(out, LOGMATMUL_CUDA_PASSED, 2), out

    def old_report(i: int, stdout: str, rc: int) -> str:
        return f"--- evidence {i} ---\n{evidence(stdout)}\n--- tail {i} ---\n{stdout[-8000:]}"

    assert not locked_check_inner(_host_output([(GREEN, 0), (GREEN, 0)], old_report), 43, 2), (
        "control: the a683f8e format should fail check_inner (each summary printed twice)"
    )


@pytest.mark.parametrize("which", [1, 2])
def test_one_red_run_fails_the_locked_check_inner(which: int) -> None:
    runs = [(GREEN, 0), (GREEN, 0)]
    runs[which - 1] = (RED, 1)
    out = _host_output(runs, run_report)
    assert "2 failed, 41 passed" in out, "the red run's summary is visible on the host"
    assert not locked_check_inner(out, LOGMATMUL_CUDA_PASSED, 2), out


def _shell_check_inner(text: str, tmp_path: Path, k: int, runs: int) -> int:
    """The locked script's own ``check_inner`` (extracted from the file, run by bash) on ``text``: its exit code."""
    out = tmp_path / "host.txt"
    out.write_text(text)
    fn = tmp_path / "check_inner.sh"
    src = LOCKED_TIER2.read_text()
    start = src.index("check_inner() {")
    fn.write_text(src[start : src.index("\n}\n", start) + 3])
    proc = subprocess.run(
        ["bash", "-c", f'source "{fn}"; check_inner logmatmul "{out}" {k} {runs}'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode


def test_the_locked_shell_check_inner_agrees(tmp_path: Path) -> None:
    """Independent of the Python port: bash runs the script's function on the same host outputs."""
    old = lambda i, s, rc: f"--- e {i}\n{evidence(s)}\n--- t {i}\n{s[-8000:]}"  # noqa: E731
    assert (
        _shell_check_inner(_host_output([(GREEN, 0), (GREEN, 0)], run_report), tmp_path, 43, 2) == 0
    )
    assert (
        _shell_check_inner(_host_output([(GREEN, 0), (RED, 1)], run_report), tmp_path, 43, 2) == 1
    )
    assert _shell_check_inner(_host_output([(GREEN, 0), (GREEN, 0)], old), tmp_path, 43, 2) == 1
