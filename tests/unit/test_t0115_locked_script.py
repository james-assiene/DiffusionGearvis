"""T-0115 criterion 8: the locked script ``scripts/locked/t0115_loop_probe.sh``, hermetically (unit; CPU; no engine,
no container, no request).

``tests/fakes/locked_scripts.py`` loads ``scripts/locked/*.py`` helpers and does not fit a shell script, so this is
the T-0084 pattern instead (``tests/unit/test_t0084_native_locked_script.py``): the script is copied into a throwaway
top-level directory and run with a stub ``PATH`` directory that shadows every command able to reach an engine, a
container or the GPU (``STUBBED``); ``FORBIDDEN`` commands fail loudly. Each stub appends its argv to a call log; the
``uv`` stub also records the proxy variables and the two run-dir variables it was started with, creates the files a
harness would write under any ``--out-dir`` it is given, and answers a ``compact`` probe from the copy's
``local_astra/toolcall/loop.py``. The copy holds placeholders for the files the steps name (both harnesses, the
contract row, both eval rows, ``loop.py`` with ``def compact``) unless a row removes one.

What the rows pin, from criterion 8 and T-0108a's boot regime (``scripts/locked/t0108a_ui_fields_ab.sh``):

- no arguments are taken (any argument refuses); refusals print ``REFUSED`` and exit 2 before the wheel build, the
  boot, the health check or any step: a dirty tree (naming its paths), an Astra port bound, running containers,
  ``services/vllm.sh`` changed since ``806da12``, an existing ``logs/t0115-loop-probe-<SHA>`` or
  ``logs/t0115-launch-<SHA>`` (one run per SHA), the T-0115a harness absent, ``loop.compact`` absent;
- the steps run in order: the T-0114 gpu contract row (``-m gpu``), the T-0115 harness (out-dir
  ``logs/t0115-loop-probe-<SHA>``), the T-0115a harness (``logs/t0115a-loop-requests-<SHA>``), the two eval rows with
  ``T0115_RUN_DIR`` / ``T0115A_RUN_DIR`` set to those dirs, then the engine is stopped; the proxy variables are unset
  for every step;
- exit 0 when every step passes, 1 when any fails;
- ``bash -n`` is clean.

Readings where the task is open (named in the test-author report): the T-0115a run dir is not a refusal of its own
(its harness refuses a non-empty out-dir); "stop the engine" is read as a ``docker`` call after the last step
(T-0108a's ``stop_engine`` trap); how the script probes ``loop.compact`` is free (a ``grep`` of ``loop.py`` or a ``uv
run`` import both work here).

Regime: CPU; bash; the stubbed git reports HEAD ``abc1234abc1234abc1234abc1234abc1234abc1`` (short ``abc1234``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LOCKED = REPO / "scripts" / "locked" / "t0115_loop_probe.sh"
HEAD = "abc1234abc1234abc1234abc1234abc1234abc1"
SHORT = "abc1234"
OUT = f"logs/t0115-loop-probe-{SHORT}"
OUT_A = f"logs/t0115a-loop-requests-{SHORT}"
AUX = f"logs/t0115-launch-{SHORT}"

CONTRACT = "tests/contract/test_llm_step_continuation.py"
H115 = "scripts/gpu/t0115_loop_probe.py"
H115A = "scripts/gpu/t0115a_loop_requests.py"
EVAL = "tests/eval/test_loop_probe.py"
EVAL_A = "tests/eval/test_loop_requests.py"
LOOP_PY = "local_astra/toolcall/loop.py"
PLACEHOLDERS = (H115, H115A, CONTRACT, EVAL, EVAL_A)
PROXIES = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")

STUBBED = (
    "timeout",
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

UV_STUB = r"""
for a in "$@"; do printf '%s\n' "$a" >> "$T_UV"; done
printf 'ENV' >> "$T_UV"
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY T0115_RUN_DIR T0115A_RUN_DIR; do
  printf ' %s=%s' "$v" "${!v-}" >> "$T_UV"
done
printf '\n--end--\n' >> "$T_UV"
case "$*" in *compact*) grep -q "def compact" "$T_TOP/local_astra/toolcall/loop.py"; exit $? ;; esac
prev=""
for a in "$@"; do
  if [ "$prev" = "--out-dir" ]; then
    mkdir -p "$a"
    for f in report.json report.md trials.jsonl steps.jsonl requests.jsonl manifest.json; do : > "$a/$f"; done
  fi
  prev="$a"
done
if [ -n "$T_FAIL_MATCH" ] && [[ "$*" == *"$T_FAIL_MATCH"* ]]; then exit "$T_FAIL_CODE"; fi
exit 0
"""

#: ``timeout [OPTION]... DURATION COMMAND...``: the call (with its duration) is logged by ``_stub``; the command runs
#: unwrapped, so the rig sees it as it would without the wrapper
TIMEOUT_STUB = r"""
while [ "$#" -gt 0 ] && [ "${1#-}" != "$1" ]; do
  case "$1" in -k|-s|--kill-after|--signal) shift 2 ;; *) shift ;; esac
done
shift
exec "$@"
"""

#: T-0117's merge commit (B4: the launch refuses unless it is an ancestor of HEAD)
T0117 = "ebff09e"


def _script() -> Path:
    assert LOCKED.is_file(), f"T-0115 criterion 8: {LOCKED.relative_to(REPO)} does not exist yet"
    return LOCKED


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f'#!/bin/bash\necho "{name} $*" >> "$T_CALLS"\n{body}\n')
    p.chmod(0o755)


@dataclass
class UvCall:
    argv: list[str]
    env: dict[str, str]

    def names(self, path: str) -> bool:
        return any(a == path or a.endswith("/" + path) for a in self.argv)

    def flag(self, name: str) -> str | None:
        for i, a in enumerate(self.argv):
            if a == name and i + 1 < len(self.argv):
                return self.argv[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return None


@dataclass
class Result:
    rc: int
    output: str
    uv: list[UvCall]
    calls: list[str]

    def steps(self) -> list[UvCall]:
        """The uv calls that run a step (pytest or a harness), not a ``compact`` probe."""
        return [c for c in self.uv if any(c.names(p) for p in PLACEHOLDERS)]

    def started(self) -> list[str]:
        boot = [c for c in self.calls if c.split(" ", 1)[0] in ("make", "setsid", "curl")]
        return boot + [" ".join(c.argv) for c in self.steps()]


def run_locked(
    tmp_path: Path,
    args: tuple[str, ...] = (),
    *,
    dirty: bool = False,
    port_bound: bool = False,
    containers: bool = False,
    vllm_drift: bool = False,
    pre_dirs: tuple[str, ...] = (),
    absent: tuple[str, ...] = (),
    no_compact: bool = False,
    not_ancestor: bool = False,
    fail_match: str = "",
    fail_code: int = 1,
) -> Result:
    text = _script().read_text()
    names = "|".join(re.escape(n) for n in STUBBED + FORBIDDEN)
    absolute = re.findall(r"/(?:usr/)?(?:local/)?s?bin/(" + names + r")\b", text)
    if absolute:
        pytest.fail(
            f"the locked script calls {sorted(set(absolute))} by absolute path; it cannot be stubbed"
        )
    top = tmp_path / "top"
    (top / "scripts" / "locked").mkdir(parents=True)
    shutil.copy(LOCKED, top / "scripts" / "locked" / LOCKED.name)
    for rel in PLACEHOLDERS:
        if rel not in absent:
            (top / rel).parent.mkdir(parents=True, exist_ok=True)
            (top / rel).write_text("# placeholder\n")
    (top / LOOP_PY).parent.mkdir(parents=True)
    (top / LOOP_PY).write_text("def step_prompt(): ...\n" if no_compact else "def compact(): ...\n")
    for d in pre_dirs:
        (top / d).mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"
    calls_log = tmp_path / "calls.log"
    _stub(
        bin_dir,
        "git",
        'case "$*" in\n'
        '  *"merge-base --is-ancestor"*) if [ -n "$T_NOT_ANCESTOR" ]; then exit 1; fi; exit 0 ;;\n'
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
    for quiet in ("sg", "sleep", "uptime", "setsid", "make", "curl"):
        _stub(bin_dir, quiet, "exit 0")
    _stub(
        bin_dir,
        "ss",
        'if [ -n "$T_PORT_BOUND" ]; then echo "LISTEN 0 4096 127.0.0.1:8001 0.0.0.0:*"; fi\nexit 0',
    )
    _stub(bin_dir, "nvidia-smi", 'echo "NVIDIA GeForce RTX 5090, 595.84, 1000 MiB, 31000 MiB"')
    _stub(bin_dir, "uv", UV_STUB)
    _stub(bin_dir, "timeout", TIMEOUT_STUB)
    for loud in FORBIDDEN:
        _stub(bin_dir, loud, 'echo "unexpected command in the locked-script harness" >&2; exit 97')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "T_CALLS": str(calls_log),
        "T_TOP": str(top),
        "T_UV": str(uv_log),
        "T_DIRTY": "1" if dirty else "",
        "T_PORT_BOUND": "1" if port_bound else "",
        "T_CONTAINERS": "1" if containers else "",
        "T_VLLM_DRIFT": "1" if vllm_drift else "",
        "T_NOT_ANCESTOR": "1" if not_ancestor else "",
        "T_FAIL_MATCH": fail_match,
        "T_FAIL_CODE": str(fail_code),
        **dict.fromkeys(PROXIES, "http://proxy.invalid:3128"),
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
    uv: list[UvCall] = []
    if uv_log.is_file():
        argv: list[str] = []
        seen: dict[str, str] = {}
        for line in uv_log.read_text().splitlines():
            if line == "--end--":
                uv.append(UvCall(argv, seen))
                argv, seen = [], {}
            elif line.startswith("ENV"):
                seen = dict(kv.split("=", 1) for kv in line[3:].split())
            else:
                argv.append(line)
    calls = calls_log.read_text().splitlines() if calls_log.is_file() else []
    return Result(proc.returncode, proc.stdout + proc.stderr, uv, calls)


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


# --------------------------------------------------------------------------- shape


def test_script_parses() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_script())], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_boot_regime_pins_and_proxies() -> None:
    """T-0108a's boot regime: ``services/vllm.sh`` pinned at ``806da12``, the engine started alone (never the full
    stack), and every proxy variable unset (``env -u`` for each of the six)."""
    code = "\n".join(_code_lines(_script().read_text()))
    assert "806da12" in code, "the services/vllm.sh pin 806da12 is gone"
    assert "services/vllm.sh" in code
    assert not re.search(r"\bastra\s+up\b", code), "the full stack is started, not the engine alone"
    missing = [v for v in PROXIES if not re.search(rf"-u\s+{v}\b", code)]
    assert missing == [], f"proxy variables not unset: {missing}"


# --------------------------------------------------------------------------- the steps


def test_the_five_steps_run_in_order(tmp_path: Path) -> None:
    """Criterion 8: boot (wheel, engine, health), then the contract row (``-m gpu``), the T-0115 harness, the T-0115a
    harness and the two eval rows with their run dirs, then the engine stop; exit 0; no proxy variable reaches a
    step; the harnesses get loopback base URLs and the out-dirs of *Launch*."""
    res = run_locked(tmp_path)
    assert res.rc == 0, res.output
    steps = res.steps()

    def first(path: str) -> int:
        found = [i for i, c in enumerate(steps) if c.names(path)]
        assert found, f"no step runs {path}: {[c.argv for c in steps]}\n{res.output}"
        return found[0]

    order = [first(CONTRACT), first(H115), first(H115A)]
    assert order == sorted(order) and len(set(order)) == 3, order
    assert min(first(EVAL), first(EVAL_A)) > order[-1], [c.argv for c in steps]
    contract = steps[first(CONTRACT)]
    assert "gpu" in (contract.flag("-m") or ""), contract.argv
    for path, out in ((H115, OUT), (H115A, OUT_A)):
        call = steps[first(path)]
        got = call.flag("--out-dir")
        assert got is not None and got.rstrip("/").endswith(out), (path, call.argv)
        base = call.flag("--base-url") or ""
        assert re.match(r"^https?://(127\.0\.0\.1|\[::1\])(:\d+)?/", base), (path, call.argv)
    probe, requests = steps[first(EVAL)], steps[first(EVAL_A)]
    assert probe.env.get("T0115_RUN_DIR", "").rstrip("/").endswith(OUT), probe.env
    assert requests.env.get("T0115A_RUN_DIR", "").rstrip("/").endswith(OUT_A), requests.env
    leaked = [(c.argv[-1], v) for c in steps for v in PROXIES if c.env.get(v)]
    assert leaked == [], f"proxy variables reached a step: {leaked}"

    calls = res.calls

    def at(pred: str) -> int:
        found = [i for i, c in enumerate(calls) if re.search(pred, c)]
        assert found, f"no call matches {pred!r}: {calls}"
        return found[0]

    uv_steps = [i for i, c in enumerate(calls) if c.startswith("uv ") and "compact" not in c]
    assert at(r"^make .*plugin-wheel") < at(r"^setsid ") < at(r"^curl .*/health") < uv_steps[0], (
        calls
    )
    assert any(c.startswith("docker ") for c in calls[uv_steps[-1] + 1 :]), (
        f"the engine is not stopped after the last step: {calls[uv_steps[-1] :]}"
    )


@pytest.mark.parametrize("failing", [CONTRACT, H115A], ids=["contract-row", "t0115a-harness"])
def test_a_failing_step_exits_1(tmp_path: Path, failing: str) -> None:
    """Exit 1 if any step fails (here the first step, and a middle one)."""
    res = run_locked(tmp_path, fail_match=failing, fail_code=1)
    assert res.rc == 1, res.output


# --------------------------------------------------------------------------- refusals


REFUSALS: dict[str, dict[str, object]] = {
    "an-argument": {"args": ("--table", "T-0115")},
    "dirty-tree": {"dirty": True},
    "port-bound": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (OUT,)},
    "aux-dir-exists": {"pre_dirs": (AUX,)},
    "t0115a-harness-absent": {"absent": (H115A,)},
    "loop-compact-absent": {"no_compact": True},
    # B4 (orchestrator, 2026-10-17): T-0117 (ebff09e) must be an ancestor of HEAD
    "t0117-not-an-ancestor": {"not_ancestor": True},
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refusals_exit_2_before_anything_runs(tmp_path: Path, case: str) -> None:
    """Criterion 8: each refusal prints ``REFUSED`` and exits 2 before the wheel build, the boot, the health check
    or any step; the dirty tree is named by its path."""
    kw = REFUSALS[case]
    res = run_locked(tmp_path, **kw)  # type: ignore[arg-type]
    assert res.rc == 2, (res.rc, res.output)
    assert "REFUSED" in res.output, res.output
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"
    if case == "dirty-tree":
        assert "x.py" in res.output, f"the dirty-tree refusal does not name the path: {res.output}"


def test_other_run_dirs_do_not_refuse(tmp_path: Path) -> None:
    """The other side of the once-per-SHA refusal (a guard too wide): this run's dirs of another SHA and an earlier
    task's dirs of this SHA do not refuse; every step runs and the script exits 0."""
    pre = (
        "logs/t0115-loop-probe-def5678",
        "logs/t0115-launch-def5678",
        f"logs/t0108a-launch-{SHORT}",
    )
    res = run_locked(tmp_path, pre_dirs=pre)
    assert res.rc == 0, res.output
    assert any(c.names(H115) for c in res.steps()), res.output


# --------------------------------------------------------------------------- B4: T-0117 is an ancestor


def test_an_ancestor_t0117_does_not_refuse_and_is_checked(tmp_path: Path) -> None:
    """The other side of B4 (a guard too wide, and a guard absent that passes vacuously): when ``ebff09e`` is an
    ancestor of HEAD the script does not refuse -- every step runs and it exits 0 -- and it did ask git, with
    ``merge-base --is-ancestor ebff09e<...> HEAD``, before the boot."""
    res = run_locked(tmp_path)
    assert res.rc == 0, res.output
    asked = [
        i
        for i, c in enumerate(res.calls)
        if re.match(rf"^git (-C \S+ )?merge-base --is-ancestor {T0117}[0-9a-f]* (HEAD|{HEAD})$", c)
    ]
    assert asked, f"B4: the script never checks that {T0117} is an ancestor of HEAD: {res.calls}"
    boot = [i for i, c in enumerate(res.calls) if c.split(" ", 1)[0] in ("make", "setsid", "curl")]
    assert boot and asked[0] < boot[0], res.calls
    assert any(c.names(H115) for c in res.steps()), res.output


# --------------------------------------------------------------------------- B3: wall-time bounds


def _seconds(token: str) -> float | None:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", token)
    if m is None:
        return None
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _timeout_of(calls: list[str], path: str) -> float | None:
    """The duration (s) of the ``timeout`` call that runs ``path``, options skipped; ``None`` if none does."""
    for c in calls:
        words = c.split()
        if not words or words[0] != "timeout" or not any(w.endswith(path) for w in words):
            continue
        rest = words[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in ("-k", "-s", "--kill-after", "--signal") else rest[1:]
        return _seconds(rest[0]) if rest else None
    return None


def test_the_harnesses_run_under_their_wall_time_bounds(tmp_path: Path) -> None:
    """B3 (orchestrator, 2026-10-17): the T-0115 harness runs under ``timeout 90m`` (5 400 s) and the T-0115a harness
    under ``timeout 30m`` (1 800 s); any spelling of the same duration is accepted. Both still run (the stub runs the
    wrapped command) and the script exits 0."""
    res = run_locked(tmp_path)
    assert res.rc == 0, res.output
    assert _timeout_of(res.calls, H115) == 5400, [c for c in res.calls if c.startswith("timeout")]
    assert _timeout_of(res.calls, H115A) == 1800, [c for c in res.calls if c.startswith("timeout")]
    assert any(c.names(H115) for c in res.steps()) and any(c.names(H115A) for c in res.steps())
