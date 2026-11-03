"""T-0191 group B, B1 and B4: ``scripts/gpu/t0191_nav_smoke.sh``, the GPU navigation smoke, checked without a GPU.

B1: the script is a copy of ``scripts/gpu/t0174_web_ab.sh`` (its refusals, hard stops, strace connect audit,
listener checks, per-session fresh data dir and startup void check) with one arm, ``--live-guidance v3``, in web mode
(no ``--no-web``), plus the T-0174 smoke's other flags. Each trial is one session with **two** typed turns through
``scripts/demo/text_mode_probe.py``:

- N1: "Open amazon.com.", then "Now go to photopea.com."
- N2: "Open duckduckgo.com/html/?q=photo+editor.", then "Open Wikipedia."
- N3 (report-only): "Open amazon.com.", then "Search Amazon for bath towels."

n = 5 per scenario; the order is shuffled with seed 0 and written before any turn. Extra refusal: HEAD contains
``Merge T-0191-a:``. A trial is void if its startup output lacks ``live guidance: v3 (flag)``. B4: the run stops
at 2700 s (ruled 2026-10-28 at 6f78c5c2 for 15 sessions; a budget-stopped run is "incomplete"); each turn is bounded by the T-0174 smoke's ``TURN_TIMEOUT_S``
(60 s).

**Readings this module pins (the test-author's; see the report for the ambiguous ones):**

- the output dir is ``logs/t0191-nav-smoke-<sha7>/`` (the task's Regime line);
- ``order.json`` holds ``[scenario, rep]`` pairs, ``random.Random(0).shuffle`` of the 15 pairs enumerated as nested
  loops over scenario (N1, N2, N3) and rep (0-4) in either nesting (as T-0174 CW3's "any pre-shuffle
  enumeration"); each trial lives in ``turns/<scenario>-<rep>/`` and its session's ``--log-dir`` is that dir's
  ``sessions``;
- the T-0174 smoke's refusals are all kept (``Merge T-0160 group C impl``, ``Merge T-0174-a:``), and the T-0191-a
  guard accepts ``Merge T-0191-a:`` and refuses the bare name, a tests-only ``Merge T-0191-a tests: ...`` and the
  look-alike ``Merge T-0191-ab: ...``;
- void is a report finding (as T-0174 CW3): a completed run with void trials exits 0, ``report.json`` names them;
- the budget stop (B4) is checked statically (the literal 2700 and the word "budget" in the script's code): a
  dynamic row would dictate how the script reads the clock.

How the rows run the script: as ``test_t0174_web_ab_script.py`` does, in a scratch git repo with ``strace``,
``ss``, ``curl``, ``uv`` and ``sleep`` replaced by stubs on ``PATH``. The ``uv`` stub also records, at each probe,
whether the session's ports are bound, how many sessions have started and whether ``order.json`` exists; the
``strace`` stub records whether ``order.json`` exists when the session starts. The T-0174 smoke runs through the
same stubs as the reference for "the T-0174 smoke's other flags".

Regime: CPU, scratch repos in tmp, stubbed tools; no GPU, no network. The real script is never run against this box.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "gpu" / "t0191_nav_smoke.sh"
ANALYSIS = REPO / "scripts" / "gpu" / "t0191_nav_smoke_analysis.py"
T0174 = REPO / "scripts" / "gpu" / "t0174_web_ab.sh"
T0174_ANALYSIS = REPO / "scripts" / "gpu" / "t0174_web_ab_analysis.py"

MERGE_160 = "Merge T-0160 group C impl: wiring (fixture)"
MERGE_174A = "Merge T-0174-a: live guidance v2 behind --live-guidance (fixture)"
MERGE_191A = (
    "Merge T-0191-a: live guidance v3 (how to go to a different website, web mode only) (fixture)"
)
ALL_MERGES = (MERGE_160, MERGE_174A, MERGE_191A)

#: B1's prompts, typed from docs/tasks/T-0191.md (setup, then turn 2)
PROMPTS = {
    "N1": ("Open amazon.com.", "Now go to photopea.com."),
    "N2": ("Open duckduckgo.com/html/?q=photo+editor.", "Open Wikipedia."),
    "N3": ("Open amazon.com.", "Search Amazon for bath towels."),
}
SCENARIOS = ("N1", "N2", "N3")
N = 5
PROXIES = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
    "DDGS_PROXY",
    "PRIMP_PROXY",
)
HEALTHY = {"CURL_ENGINE": "200", "CURL_TTS": "200", "CURL_STT": "400"}
STUBBED = ("strace", "ss", "curl", "uv", "sleep")
SLEEP = shutil.which("sleep") or "/usr/bin/sleep"
TURN_RE = re.compile(r"logs/t0(?:174-web-ab|191-nav-smoke)-[0-9a-f]{7}/turns/[^/\s]+")
T0191_TURN_RE = re.compile(r"^logs/t0191-nav-smoke-([0-9a-f]{7})/turns/(N[123])-(\d+)/sessions/?$")


def _text() -> str:
    assert SCRIPT.is_file(), f"T-0191 B1: {SCRIPT.relative_to(REPO)} does not exist yet"
    return SCRIPT.read_text(encoding="utf-8")


def _code(text: str) -> str:
    """Continuation lines joined, comment-only lines dropped."""
    out, buf = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("#") and not buf:
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    return "\n".join(out)


def _all_code() -> str:
    """The script's code plus the analysis module's text (the connect audit may live in either)."""
    extra = ANALYSIS.read_text(encoding="utf-8") if ANALYSIS.is_file() else ""
    return _code(_text()) + "\n" + extra


# --------------------------------------------------------------------------------------------- the stubbed run

STRACE_STUB = """#!{py}
import glob, json, os, signal, sys, time
from pathlib import Path

stub = Path(os.environ["STUB_DIR"])
mode = os.environ.get("STUB_MODE", "ok")
args = sys.argv[1:]
order_seen = bool(glob.glob("logs/t0191-nav-smoke-*/order.json") or glob.glob("logs/t0174-web-ab-*/order.json"))
with open(stub / "strace.log", "a") as f:
    f.write(json.dumps({{"unbuffered": os.environ.get("PYTHONUNBUFFERED"), "argv": args,
                         "order_seen": order_seen}}) + "\\n")
if mode == "exit":
    sys.exit(0)
g = None
for i, a in enumerate(args):
    if a == "--live-guidance" and i + 1 < len(args):
        g = args[i + 1]
    elif a.startswith("--live-guidance="):
        g = a.split("=", 1)[1]
if g is not None:
    if mode == "swap_guidance":
        g = {{"v1": "v2", "v2": "v1", "v3": "v2"}}.get(g, g)
    print(f"astra session: live guidance: {{g}} (flag)", flush=True)
print("astra session: reask: announced_call_reask=on (default)", flush=True)
if mode != "no_notice":
    print("astra session: web browsing is on: Gearvis can open public websites. (stub)", flush=True)
if mode == "unsafe" and "--log-dir" in args:
    d = Path(args[args.index("--log-dir") + 1])
    d.mkdir(parents=True, exist_ok=True)
    rec = {{"seq": 1, "kind": "tts.utterance", "t_ms": 1, "turn_id": "t1", "data": {{"text": "<|turn>model hi"}}}}
    (d / "stub.jsonl").write_text(json.dumps(rec) + "\\n")
done = []
signal.signal(signal.SIGINT, lambda *_: done.append(1))
signal.signal(signal.SIGTERM, lambda *_: done.append(1))
if mode != "never_bind":
    (stub / "bound").touch()
deadline = time.monotonic() + 60
while not done and time.monotonic() < deadline:
    time.sleep(0.01)
if mode != "stuck_port":
    (stub / "bound").unlink(missing_ok=True)
"""

UV_STUB = """#!{py}
import glob, json, os, sys
from pathlib import Path

stub = Path(os.environ["STUB_DIR"])
args = sys.argv[1:]
sessions = 0
if (stub / "strace.log").is_file():
    sessions = sum(1 for x in (stub / "strace.log").read_text().splitlines() if x.strip())
with open(stub / "uv.log", "a") as f:
    f.write(json.dumps({{
        "unbuffered": os.environ.get("PYTHONUNBUFFERED"),
        "argv": args,
        "bound": (stub / "bound").exists(),
        "sessions": sessions,
        "order_seen": bool(glob.glob("logs/t0191-nav-smoke-*/order.json")),
    }}) + "\\n")
if any(a.endswith("text_mode_probe.py") for a in args):
    print("turn completed: True  wall 1.0s (stub)")
    sys.exit(0)
if "pytest" in args:
    print("stub pytest: 0 failed")
    sys.exit(0)
if "python" not in args:
    print("STUB-UV", file=sys.stderr)
    sys.exit(97)
rest = args[args.index("python") + 1 :]
os.execv(sys.executable, [sys.executable, *rest])
"""

SS_STUB = """#!/bin/sh
if [ -n "${STUB_PREBOUND:-}" ]; then printf 'LISTEN 0 128 127.0.0.1:%s 0.0.0.0:*\\n' "$STUB_PREBOUND"; fi
A="${STUB_LISTEN_ADDR:-127.0.0.1}"
if [ -e "$STUB_DIR/bound" ]; then
  printf 'LISTEN 0 128 %s:8000 0.0.0.0:*\\nLISTEN 0 128 %s:1420 0.0.0.0:*\\n' "$A" "$A"
fi
exit 0
"""

CURL_STUB = """#!/bin/sh
case "$*" in
  *127.0.0.1:8001/*) printf '%s' "${CURL_ENGINE:-000}" ;;
  *127.0.0.1:8081/*) printf '%s' "${CURL_TTS:-000}" ;;
  *127.0.0.1:8080/*) printf '%s' "${CURL_STT:-000}" ;;
  *) printf 000 ;;
esac
"""


@dataclass
class Run:
    proc: subprocess.CompletedProcess[str]
    repo: Path
    stubs: Path

    @property
    def output(self) -> str:
        return self.proc.stdout + self.proc.stderr

    def _log(self, name: str) -> list[dict[str, Any]]:
        p = self.stubs / name
        if not p.is_file():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    @property
    def sessions(self) -> list[dict[str, Any]]:
        return self._log("strace.log")

    @property
    def uv(self) -> list[dict[str, Any]]:
        return self._log("uv.log")

    @property
    def probes(self) -> list[dict[str, Any]]:
        return [u for u in self.uv if any(a.endswith("text_mode_probe.py") for a in u["argv"])]

    def out_dir(self, prefix: str) -> Path:
        logs = self.repo / "logs"
        dirs = sorted(d for d in logs.glob(f"{prefix}-*") if d.is_dir()) if logs.is_dir() else []
        assert len(dirs) == 1, (prefix, dirs, self.output[-3000:])
        return dirs[0]


@pytest.fixture(scope="module")
def sysbin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The system's tools, minus the stubbed ones (so a missing stub means a missing tool)."""
    d = tmp_path_factory.mktemp("sysbin")
    for src in ("/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"):
        root = Path(src)
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if entry.name in STUBBED or (d / entry.name).exists() or (d / entry.name).is_symlink():
                continue
            (d / entry.name).symlink_to(entry)
    return d


def _git(repo: Path, *a: str) -> None:
    git = shutil.which("git")
    assert git
    subprocess.run(
        [git, "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.invalid", *a],
        check=True,
        capture_output=True,
    )


def _sha7(repo: Path) -> str:
    git = shutil.which("git")
    assert git
    return subprocess.run(
        [git, "-C", str(repo), "rev-parse", "--short=7", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def scratch(
    tmp: Path,
    sysbin: Path,
    which: str,
    *,
    merges: Iterable[str] = ALL_MERGES,
    mode: str = "ok",
    env: dict[str, str] | None = None,
    args: Iterable[str] = (),
    strace: bool = True,
    before: Callable[[Path], None] | None = None,
) -> Run:
    """Run the T-0191 script (``which="t0191"``) or the T-0174 reference (``"t0174"``) in a scratch repo."""
    if which == "t0191":
        rel, text, analysis = "scripts/gpu/t0191_nav_smoke.sh", _text(), ANALYSIS
    else:
        rel, text, analysis = (
            "scripts/gpu/t0174_web_ab.sh",
            T0174.read_text("utf-8"),
            T0174_ANALYSIS,
        )
    repo = tmp / "repo"
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_text(text)
    if analysis.is_file():
        (repo / "scripts" / "gpu" / analysis.name).write_text(analysis.read_text(encoding="utf-8"))
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    for subject in merges:
        _git(repo, "checkout", "-qb", "side")
        _git(repo, "commit", "-q", "--allow-empty", "-m", "s")
        _git(repo, "checkout", "-q", "main")
        _git(repo, "merge", "-q", "--no-ff", "-m", subject, "side")
        _git(repo, "branch", "-qD", "side")
    if before is not None:
        before(repo)
    stubs = tmp / "stubs"
    stubs.mkdir()
    bodies = {
        "ss": SS_STUB,
        "curl": CURL_STUB,
        "uv": UV_STUB.format(py=sys.executable),
        "sleep": f"#!/bin/sh\nexec {SLEEP} 0.01\n",
    }
    if strace:
        bodies["strace"] = STRACE_STUB.format(py=sys.executable)
    for name, body in bodies.items():
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    full = {
        "PATH": f"{stubs}:{sysbin}",
        "HOME": str(tmp),
        "LANG": "C",
        "STUB_DIR": str(stubs),
        "STUB_MODE": mode,
        "PYTHONPATH": str(REPO),
        **(env or {}),
    }
    proc = subprocess.run(
        ["bash", str(repo / rel), *args],
        cwd=repo,
        env=full,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return Run(proc, repo, stubs)


@pytest.fixture(scope="module")
def ok_runs(tmp_path_factory: pytest.TempPathFactory, sysbin: Path) -> Callable[[str], Run]:
    """One healthy stubbed run per script, shared by the rows that read it (run on first use, in the test)."""
    cache: dict[str, Run] = {}

    def get(which: str) -> Run:
        if which not in cache:
            cache[which] = scratch(
                tmp_path_factory.mktemp(f"ok-{which}"), sysbin, which, env=HEALTHY
            )
        return cache[which]

    return get


def _flag_values(argv: list[str], flag: str) -> list[str]:
    found = []
    for i, a in enumerate(argv):
        if a == flag:
            found.append(argv[i + 1] if i + 1 < len(argv) else "")
        elif a.startswith(flag + "="):
            found.append(a.split("=", 1)[1])
    return found


def _without_guidance(argv: list[str]) -> list[str]:
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--live-guidance":
            skip = i + 1 < len(argv)
            continue
        if a.startswith("--live-guidance="):
            continue
        out.append(a)
    return out


def _normalise(argv: list[str]) -> list[str]:
    return [TURN_RE.sub("<TURN>", a) for a in argv]


def _trial_of(argv: list[str]) -> tuple[str, int]:
    assert "--log-dir" in argv, argv
    m = T0191_TURN_RE.match(argv[argv.index("--log-dir") + 1])
    assert m, (
        f"B1: the session's --log-dir is not logs/t0191-nav-smoke-<sha7>/turns/<scenario>-<rep>/sessions: {argv}"
    )
    return m.group(2), int(m.group(3))


def _enumerations() -> list[list[tuple[str, int]]]:
    """The 15 (scenario, rep) pairs as nested loops, in each of the two nestings."""
    a = [(s, r) for s in SCENARIOS for r in range(N)]
    b = [(s, r) for r in range(N) for s in SCENARIOS]
    return [a, b]


def _oracle_orders() -> list[list[tuple[str, int]]]:
    out = []
    for o in _enumerations():
        o = list(o)
        random.Random(0).shuffle(o)
        out.append(o)
    return out


def test_premise_oracle_orders_differ_from_unshuffled() -> None:
    """Active premise: two distinct enumerations of 15 pairs; each accepted order holds them all and is neither an
    enumeration nor its seed-1 shuffle (so an unshuffled or wrong-seed order fails the order row)."""
    enums = _enumerations()
    assert len({tuple(e) for e in enums}) == 2
    full = sorted((s, r) for s in SCENARIOS for r in range(N))
    for e, o in zip(enums, _oracle_orders(), strict=True):
        e1 = list(e)
        random.Random(1).shuffle(e1)
        assert sorted(o) == full and o not in enums and o != e1


# --------------------------------------------------------------------------------------------- static rows


def test_script_parses_and_is_strict() -> None:
    text = _text()
    assert text.startswith("#!") and "set -euo pipefail" in text
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_static_copies_of_the_t0174_audits() -> None:
    """B1 "keeping its ... strace connect audit, listener checks": the process-group listener check and the connect
    audit (non-public connects by ``is_global``, the resolver from ``resolv.conf``, the UDP note) are kept (script
    or analysis module); no ``--no-web``, no ``--no-sandbox`` anywhere in the script's code."""
    code = _all_code()
    for word in (
        "pgrep -g",
        "ss -tlnpH",
        "is_global",
        "resolv.conf",
        "UDP connects",
        "fails safe",
        "strace",
    ):
        assert word in code, word
    assert "--no-web" not in _code(_text()) and "--no-sandbox" not in _code(_text())


def test_no_grep_pipelines_under_pipefail() -> None:
    """As T-0160/T-0174: no ``ss | ...``, ``| grep -q`` or ``printf | tr`` pipeline in the shell code."""
    code = _code(_text())
    shell = code.split("<<'PY'", 1)[0] + (code.rsplit("\nPY\n", 1)[-1] if "<<'PY'" in code else "")
    pipe = r"(?<!\|)\|(?!\|)"
    for bad in (
        rf"\bss\b[^\n|]*{pipe}",
        rf"{pipe}\s*grep\s+-[a-zA-Z]*q",
        rf"printf[^\n|]*{pipe}\s*tr\b",
    ):
        hits = re.findall(rf"^.*{bad}.*$", shell, re.M)
        assert hits == [], f"a pipeline under pipefail: {hits}"


def test_prompts_n_timeout_and_seed_verbatim() -> None:
    """B1: the six prompts verbatim, n = 5, the T-0174 smoke's 60 s turn timeout, seed 0."""
    code = _code(_text())
    for setup, turn2 in PROMPTS.values():
        assert setup in code and turn2 in code, (setup, turn2)
    assert re.search(r"\bN=5\b", code) and re.search(r"TURN_TIMEOUT_S=60\b", code)
    assert "Random(0)" in _all_code()


def test_budget_stop_2700_s() -> None:
    """B4 (Group B readings, 6f78c5c2): the run stops at 2700 s and a budget-stopped run is "incomplete". Checked statically (see the module
    docstring): the script's code holds the literal 2700 (not 1800) and says "budget" and "incomplete"."""
    code = _code(_text())
    assert re.search(r"\b2700\b", code), "B4: no 2700 s budget in the script's code"
    assert not re.search(r"\b1800\b", code), (
        "B4: the pre-N3 1800 s budget is still in the script's code"
    )
    assert re.search(r"(?i)budget", code) and re.search(r"(?i)incomplete", code), (
        "B4: the budget stop does not say 'budget' / 'incomplete' in the script's code"
    )


# --------------------------------------------------------------------------------------------- refusals (prelude)


def _refused(run: Run, *words: str) -> None:
    assert run.proc.returncode == 2, (run.proc.returncode, run.output[-3000:])
    assert "REFUSED" in run.proc.stderr, run.output[-3000:]
    for w in words:
        assert w in run.proc.stderr, (w, run.proc.stderr)
    assert run.sessions == [] and run.uv == [], "a refusal comes before any uv call or session"


def test_refuses_arguments(tmp_path: Path, sysbin: Path) -> None:
    _refused(scratch(tmp_path, sysbin, "t0191", env=HEALTHY, args=["--n", "3"]), "no arguments")


def test_refuses_a_dirty_tree(tmp_path: Path, sysbin: Path) -> None:
    run = scratch(
        tmp_path, sysbin, "t0191", env=HEALTHY, before=lambda r: (r / "stray.txt").write_text("x")
    )
    _refused(run, "dirty")


@pytest.mark.parametrize(
    "merges",
    [
        [MERGE_160, MERGE_174A],
        [MERGE_160, MERGE_174A, "Merge T-0191-a tests: guidance v3 rows (fixture)"],
        [MERGE_160, MERGE_174A, "Merge T-0191-a"],
        [MERGE_160, MERGE_174A, "Merge T-0191-ab: a look-alike (fixture)"],
    ],
    ids=["no_t0191a", "tests_only", "bare_name", "look_alike"],
)
def test_refuses_without_the_t0191a_merge(tmp_path: Path, sysbin: Path, merges: list[str]) -> None:
    """B1's extra refusal (wrong HEAD): HEAD must contain ``Merge T-0191-a:`` (so v3 exists). With healthy services
    and every other merge present, the script would otherwise go on to run sessions."""
    _refused(scratch(tmp_path, sysbin, "t0191", merges=merges, env=HEALTHY), "Merge T-0191-a")


def test_accepts_the_real_t0191a_subject(ok_runs: Callable[[str], Run]) -> None:
    """The guard is not too wide: with ``Merge T-0191-a: live guidance v3 ...`` (the real subject's shape) as an
    ancestor, nothing is refused and all 15 sessions run."""
    run = ok_runs("t0191")
    assert "REFUSED" not in run.proc.stderr, run.output[-3000:]
    assert len(run.sessions) == len(SCENARIOS) * N, (len(run.sessions), run.output[-3000:])


@pytest.mark.parametrize(
    ("merges", "word"),
    [
        ([MERGE_174A, MERGE_191A], "Merge T-0160 group C impl"),
        ([MERGE_160, MERGE_191A], "Merge T-0174-a"),
    ],
    ids=["no_t0160", "no_t0174a"],
)
def test_keeps_the_t0174_merge_refusals(
    tmp_path: Path, sysbin: Path, merges: list[str], word: str
) -> None:
    """B1 "keeping its refusals": the T-0174 smoke's two merge refusals stay."""
    _refused(scratch(tmp_path, sysbin, "t0191", merges=merges, env=HEALTHY), word)


@pytest.mark.parametrize("var", PROXIES)
def test_refuses_a_proxy_variable(tmp_path: Path, sysbin: Path, var: str) -> None:
    _refused(scratch(tmp_path, sysbin, "t0191", env={**HEALTHY, var: "http://127.0.0.1:3128"}), var)


def test_refuses_without_strace(tmp_path: Path, sysbin: Path) -> None:
    _refused(scratch(tmp_path, sysbin, "t0191", env=HEALTHY, strace=False), "strace")


@pytest.mark.parametrize("port", ["8000", "1420"])
def test_refuses_a_bound_port(tmp_path: Path, sysbin: Path, port: str) -> None:
    _refused(
        scratch(tmp_path, sysbin, "t0191", env={**HEALTHY, "STUB_PREBOUND": port}), f"port {port}"
    )


@pytest.mark.parametrize(
    ("down", "word"), [("CURL_ENGINE", "engine"), ("CURL_TTS", "tts"), ("CURL_STT", "stt")]
)
def test_refuses_an_unhealthy_service(tmp_path: Path, sysbin: Path, down: str, word: str) -> None:
    _refused(scratch(tmp_path, sysbin, "t0191", env={**HEALTHY, down: "000"}), word)


def test_refuses_an_existing_output_dir(tmp_path: Path, sysbin: Path) -> None:
    """One run per SHA in ``logs/t0191-nav-smoke-<sha7>/`` (an empty dir does not dirty the tree)."""
    made: list[str] = []

    def mk(repo: Path) -> None:
        name = f"t0191-nav-smoke-{_sha7(repo)}"
        (repo / "logs" / name).mkdir(parents=True)
        made.append(name)

    _refused(scratch(tmp_path, sysbin, "t0191", env=HEALTHY, before=mk), made[0], "exists")


# --------------------------------------------------------------------------------------------- the run (stubbed)


def test_one_arm_v3_web_mode_and_the_t0174_flags(ok_runs: Callable[[str], Run]) -> None:
    """B1: 15 sessions; each passes ``--live-guidance v3`` exactly once and no ``--no-web``; with that pair removed,
    the whole strace + session command line equals the T-0174 smoke's (per-trial paths normalised)."""
    ref = ok_runs("t0174")
    assert ref.sessions, ref.output[-2000:]
    want = _normalise(_without_guidance(ref.sessions[0]["argv"]))
    run = ok_runs("t0191")
    assert len(run.sessions) == len(SCENARIOS) * N, (len(run.sessions), run.output[-3000:])
    for s in run.sessions:
        argv = s["argv"]
        _trial_of(argv)
        assert _flag_values(argv, "--live-guidance") == ["v3"], argv
        assert "--no-web" not in argv and not any(a.startswith("--no-web") for a in argv), argv
        assert _normalise(_without_guidance(argv)) == want, (argv, want)


def test_fresh_data_dir_per_session(ok_runs: Callable[[str], Run]) -> None:
    """B1 "per-session fresh data dir": every session's ``--data-dir`` is its own, inside its trial's dir."""
    run = ok_runs("t0191")
    assert run.sessions, run.output[-3000:]
    dirs = []
    for s in run.sessions:
        argv = s["argv"]
        scen, rep = _trial_of(argv)
        d = _flag_values(argv, "--data-dir")
        assert len(d) == 1 and f"/turns/{scen}-{rep}/" in d[0], argv
        dirs.append(d[0])
    assert len(set(dirs)) == len(dirs), dirs


def test_order_seed_0_written_before_the_first_turn(ok_runs: Callable[[str], Run]) -> None:
    """B1: N1, N2, N3 x n = 5 in one order shuffled with seed 0, written to ``order.json`` as ``[scenario, rep]``
    before any turn (the first session and the first probe both see the file); sessions run in that order; the
    output dir is ``logs/t0191-nav-smoke-<sha7>/`` with HEAD's sha."""
    run = ok_runs("t0191")
    out = run.out_dir("t0191-nav-smoke")
    assert out.name == f"t0191-nav-smoke-{_sha7(run.repo)}", out
    order = [(str(s), int(r)) for s, r in json.loads((out / "order.json").read_text())]
    assert order in _oracle_orders(), order
    assert [_trial_of(s["argv"]) for s in run.sessions] == order
    assert run.sessions[0]["order_seen"] is True, (
        "B1: the first session started before order.json was written"
    )
    assert run.probes and run.probes[0]["order_seen"] is True


def test_two_probes_per_session_setup_then_turn_two(ok_runs: Callable[[str], Run]) -> None:
    """B1: each session gets exactly two typed turns through ``text_mode_probe.py``, the scenario's setup prompt then
    its turn-2 prompt, both while that session (and only it) is up, each with ``--timeout-s 60``."""
    run = ok_runs("t0191")
    order = [_trial_of(s["argv"]) for s in run.sessions]
    probes = run.probes
    assert len(probes) == 2 * len(order), (len(probes), len(order))
    for i, (scen, _) in enumerate(order):
        pair = probes[2 * i : 2 * i + 2]
        for turn, (p, prompt) in enumerate(zip(pair, PROMPTS[scen], strict=True), start=1):
            argv = p["argv"]
            assert prompt in argv, (scen, turn, argv)
            assert _flag_values(argv, "--timeout-s") == ["60"], argv
            assert p["bound"] is True, f"B1: {scen} turn {turn}'s probe ran with no session up"
            assert p["sessions"] == i + 1, (
                f"B1: {scen} turn {turn} ran in session {p['sessions']}, not {i + 1}"
            )


def test_completed_run_writes_the_report(ok_runs: Callable[[str], Run]) -> None:
    """The script runs the analysis on the finished run: ``report.json`` holds a B3 decision and no void finding
    (every stub session printed ``live guidance: v3 (flag)``). The stub sessions log nothing, so every trial is a
    setup failure and the decision is INCONCLUSIVE."""
    run = ok_runs("t0191")
    assert run.proc.returncode == 0, run.output[-3000:]
    report = json.loads((run.out_dir("t0191-nav-smoke") / "report.json").read_text())
    assert report["void"] == [], report["void"]
    assert report["decision"] == "INCONCLUSIVE", report["decision"]


def test_void_when_the_startup_line_is_missing(tmp_path: Path, sysbin: Path) -> None:
    """B1: every session announces ``live guidance: v2 (flag)`` instead of v3: every trial is void and named in
    ``report.json``. As T-0174 CW3, void is a report finding: the completed run exits 0."""
    run = scratch(tmp_path, sysbin, "t0191", mode="swap_guidance", env=HEALTHY)
    report = json.loads((run.out_dir("t0191-nav-smoke") / "report.json").read_text())
    void = report["void"]
    assert isinstance(void, list) and void, report
    for s, r in ((s, r) for s in SCENARIOS for r in range(N)):
        assert any(f"{s}-{r}" in str(v) for v in void), (s, r, void)
    assert run.proc.returncode == 0, (run.proc.returncode, run.output[-3000:])


def test_no_eval_call_and_unbuffered(ok_runs: Callable[[str], Run]) -> None:
    """As T-0174: no ``pytest`` call, and every session, probe and analysis call runs with ``PYTHONUNBUFFERED=1``
    (the ``python -c`` one-liners excepted)."""
    run = ok_runs("t0191")
    assert run.uv and not [u for u in run.uv if any("pytest" in a for a in u["argv"])]
    assert run.sessions and all(s["unbuffered"] == "1" for s in run.sessions)
    rest = [u for u in run.uv if "-c" not in u["argv"]]
    assert rest and all(u["unbuffered"] == "1" for u in rest), [
        u for u in rest if u["unbuffered"] != "1"
    ]


HARD_STOPS = {
    "exit": ("exited before binding", {}),
    "never_bind": ("never bound", {}),
    "no_notice": ("web-mode notice is absent", {}),
    "nonloopback": ("non-loopback listener", {"STUB_LISTEN_ADDR": "0.0.0.0"}),
    "stuck_port": ("still bound after SIGINT", {}),
    "unsafe": ("a safety counter is not 0", {}),
}


@pytest.mark.parametrize("case", list(HARD_STOPS))
def test_hard_stops(tmp_path: Path, sysbin: Path, case: str) -> None:
    """B1 "keeping its hard stops": each of the T-0174 smoke's hard stops (exit 1), under the same stub modes as
    ``test_t0174_web_ab_script.py``'s rows."""
    phrase, extra = HARD_STOPS[case]
    mode = "ok" if case == "nonloopback" else case
    run = scratch(tmp_path, sysbin, "t0191", mode=mode, env={**HEALTHY, **extra})
    assert run.proc.returncode == 1, (case, run.proc.returncode, run.output[-3000:])
    assert "HARD STOP" in run.output and phrase in run.output, (case, run.output[-3000:])


def test_premise_enumerations_shape() -> None:
    """Active premise: the itertools product equals the scenario-major enumeration (the oracle is not hand-typed)."""
    assert list(itertools.product(SCENARIOS, range(N))) == _enumerations()[0]
