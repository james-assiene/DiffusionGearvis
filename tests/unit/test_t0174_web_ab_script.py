"""T-0174 group C-web, CW1: ``scripts/gpu/t0174_web_ab.sh``, the quick real-web v1/v2 A/B, checked without a GPU.

The script is a copy of ``scripts/locked/t0160_web_smoke.sh`` with two arms (A ``--live-guidance v1``, B
``--live-guidance v2``), 20 trials jointly shuffled with order seed 0, output in ``logs/t0174-web-ab-<sha7>/`` and
one more refusal (HEAD must contain the ``Merge T-0174-a:`` merge). Its analysis is
``scripts/gpu/t0174_web_ab_analysis.py`` (``analyse(run_dir)``, pinned in ``test_t0174_web_ab_analysis.py``).

How the rows run the script: in a scratch git repo (the script, and the analysis module when it exists), with
``strace``, ``ss``, ``curl``, ``uv`` and ``sleep`` replaced by stubs on ``PATH`` (every other tool is the system's):

- ``curl`` answers the health probes as told (``000`` = down);
- ``strace`` records its argv and ``PYTHONUNBUFFERED`` and plays the session: it prints the guidance line it was
  asked for (``live guidance: <v> (flag)``) and the web-mode notice, marks ports 8000 and 1420 bound (``ss`` then
  lists them) until SIGINT; modes make it exit at once, never bind, omit the notice, bind 0.0.0.0, keep the ports
  after SIGINT, print the *other* guidance version, or write a session log that voices a special token;
- ``uv`` records its argv and ``PYTHONUNBUFFERED``; a ``text_mode_probe.py`` call prints ``turn completed: True``, a
  ``pytest`` call exits 0, a ``python ...`` call runs the real interpreter (``PYTHONPATH`` = this repo) so the order
  step and the analysis run for real; anything else exits 97;
- ``sleep`` sleeps 10 ms.

Nothing reaches a service, a GPU or the network. The T-0160 script runs through the same stubs as a control
(active rows): its flags are the reference for "every other flag is the T-0160 smoke's", and its hard stops show the
stubs reproduce each hard-stop condition.

Rulings (CW3, 2026-10-27, task file 47b94512): the merge guard accepts ``Merge T-0174-a:`` (the real subject is
``Merge T-0174-a: live guidance v2 ...``) and refuses the bare ``Merge T-0174-a``, a tests-only ``Merge T-0174-a
tests: ...`` and ``Merge T-0174-ab: ...``; "any pre-shuffle enumeration": the order is ``random.Random(0).shuffle``
of the 20 triples enumerated as nested loops over arm (A, B), task (W1, W2) and rep (0-4) in any of the six nestings;
the script calls no eval module (no ``pytest``); void is a report finding and a completed run exits 0.
``order.json`` holds ``[arm, task, rep]`` triples and each trial lives in ``turns/<arm>-<task>-<rep>/``.

Regime: CPU, scratch repos in tmp, stubbed tools; no GPU, no network.
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

from tests.eval import test_web_smoke as oracle

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "gpu" / "t0174_web_ab.sh"
ANALYSIS = REPO / "scripts" / "gpu" / "t0174_web_ab_analysis.py"
T0160 = REPO / "scripts" / "locked" / "t0160_web_smoke.sh"

MERGE_160 = "Merge T-0160 group C impl: wiring (fixture)"
MERGE_174A = "Merge T-0174-a: live guidance v2 behind --live-guidance (fixture)"
GUIDANCE = {"A": "v1", "B": "v2"}
TASKS = ("W1", "W2")
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
TURN_RE = re.compile(r"logs/t0(?:160-web-smoke|174-web-ab)-[0-9a-f]{7}/turns/[^/\s]+")
T0174_TURN_RE = re.compile(
    r"^logs/t0174-web-ab-([0-9a-f]{7})/turns/([AB])-(W[12])-(\d+)/sessions/?$"
)


def _text() -> str:
    assert SCRIPT.is_file(), f"T-0174 CW1: {SCRIPT.relative_to(REPO)} does not exist yet"
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
import json, os, signal, sys, time
from pathlib import Path

stub = Path(os.environ["STUB_DIR"])
mode = os.environ.get("STUB_MODE", "ok")
args = sys.argv[1:]
with open(stub / "strace.log", "a") as f:
    f.write(json.dumps({{"unbuffered": os.environ.get("PYTHONUNBUFFERED"), "argv": args}}) + "\\n")
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
        g = {{"v1": "v2", "v2": "v1"}}.get(g, g)
    print(f"astra session: live guidance: {{g}} (flag)", flush=True)
print("astra session: reask: announced_call_reask=on (default)", flush=True)
if mode != "no_notice":
    print("astra session: web browsing is on: Gearvis can open public websites. (stub)", flush=True)
if mode == "unsafe" and "--log-dir" in args:
    d = Path(args[args.index("--log-dir") + 1])
    d.mkdir(parents=True, exist_ok=True)
    rec = {{"seq": 1, "kind": "tts.utterance", "t_ms": 1, "data": {{"text": "<|turn>model hi"}}}}
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
import json, os, sys
from pathlib import Path

stub = Path(os.environ["STUB_DIR"])
args = sys.argv[1:]
with open(stub / "uv.log", "a") as f:
    f.write(json.dumps({{"unbuffered": os.environ.get("PYTHONUNBUFFERED"), "argv": args}}) + "\\n")
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

    def out_dir(self, prefix: str) -> Path:
        dirs = (
            sorted((self.repo / "logs").glob(f"{prefix}-*"))
            if (self.repo / "logs").is_dir()
            else []
        )
        dirs = [d for d in dirs if d.is_dir()]
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
    merges: Iterable[str] = (MERGE_160, MERGE_174A),
    mode: str = "ok",
    env: dict[str, str] | None = None,
    args: Iterable[str] = (),
    strace: bool = True,
    before: Callable[[Path], None] | None = None,
) -> Run:
    """Run the T-0174 script (``which="t0174"``) or the T-0160 control (``"t0160"``) in a scratch repo."""
    if which == "t0174":
        rel, text = "scripts/gpu/t0174_web_ab.sh", _text()
    else:
        rel, text = "scripts/locked/t0160_web_smoke.sh", T0160.read_text(encoding="utf-8")
    repo = tmp / "repo"
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_text(text)
    if which == "t0174" and ANALYSIS.is_file():
        (repo / "scripts" / "gpu" / ANALYSIS.name).write_text(ANALYSIS.read_text(encoding="utf-8"))
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
        timeout=180,
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


def _guidance_of(argv: list[str]) -> list[str]:
    found = []
    for i, a in enumerate(argv):
        if a == "--live-guidance":
            found.append(argv[i + 1] if i + 1 < len(argv) else "")
        elif a.startswith("--live-guidance="):
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


def _turn_of(argv: list[str]) -> tuple[str, str, int]:
    assert "--log-dir" in argv, argv
    m = T0174_TURN_RE.match(argv[argv.index("--log-dir") + 1])
    assert m, (
        f"CW1: the session's --log-dir is not logs/t0174-web-ab-<sha7>/turns/<arm>-<task>-<rep>/: {argv}"
    )
    return m.group(2), m.group(3), int(m.group(4))


def _enumerations() -> list[list[tuple[str, str, int]]]:
    """The 20 (arm, task, rep) triples as nested loops, in each of the six nestings (CW3: any enumeration)."""
    axes: list[list[Any]] = [list(GUIDANCE), list(TASKS), list(range(N))]
    out = []
    for perm in itertools.permutations(range(3)):
        rows = []
        for combo in itertools.product(*(axes[i] for i in perm)):
            v = dict(zip(perm, combo, strict=True))
            rows.append((v[0], v[1], v[2]))
        out.append(rows)
    return out


def _oracle_orders() -> list[list[tuple[str, str, int]]]:
    out = []
    for o in _enumerations():
        o = list(o)
        random.Random(0).shuffle(o)
        out.append(o)
    return out


# --------------------------------------------------------------------------------------------- the control (active)


def test_control_t0160_runs_clean_through_the_stubs(ok_runs: Callable[[str], Run]) -> None:
    """Active premise: the stubs carry the T-0160 smoke through all 10 turns to its OK line, one session each."""
    run = ok_runs("t0160")
    assert run.proc.returncode == 0 and "OK: T-0160 web smoke" in run.output, run.output[-3000:]
    assert len(run.sessions) == 2 * N, len(run.sessions)


HARD_STOPS = {
    "exit": ("exited before binding", {}),
    "never_bind": ("never bound", {}),
    "no_notice": ("web-mode notice is absent", {}),
    "nonloopback": ("non-loopback listener", {"STUB_LISTEN_ADDR": "0.0.0.0"}),
    "stuck_port": ("still bound after SIGINT", {}),
    "unsafe": ("a safety counter is not 0", {}),
}


def _hard_stop(tmp: Path, sysbin: Path, which: str, case: str) -> Run:
    phrase, extra = HARD_STOPS[case]
    mode = "ok" if case == "nonloopback" else case
    run = scratch(tmp, sysbin, which, mode=mode, env={**HEALTHY, **extra})
    assert run.proc.returncode == 1, (case, run.proc.returncode, run.output[-3000:])
    assert "HARD STOP" in run.output and phrase in run.output, (case, run.output[-3000:])
    return run


@pytest.mark.parametrize("case", list(HARD_STOPS))
def test_control_t0160_hard_stops_through_the_stubs(
    tmp_path: Path, sysbin: Path, case: str
) -> None:
    """Active premise: each stub mode trips the T-0160 smoke's matching hard stop (exit 1)."""
    _hard_stop(tmp_path, sysbin, "t0160", case)


# --------------------------------------------------------------------------------------------- static rows


def test_script_parses_and_is_strict() -> None:
    text = _text()
    assert text.startswith("#!") and "set -euo pipefail" in text
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_static_copies_of_the_t0160_audits() -> None:
    """CW1 "its refusals, hard stops, strace connect audit": the process-group listener check and the connect
    audit (non-public connects judged by ``is_global``, the resolver from ``resolv.conf``, the UDP note) are
    copied (script or analysis module)."""
    code = _all_code()
    for word in ("pgrep -g", "ss -tlnpH", "is_global", "resolv.conf", "UDP connects", "fails safe"):
        assert word in code, word
    assert "--no-web" not in _code(_text()) and "--no-sandbox" not in _code(_text())


def test_no_grep_pipelines_under_pipefail() -> None:
    """As T-0160's review fix: no ``ss | ...``, ``| grep -q`` or ``printf | tr`` pipeline in the shell code."""
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


def test_tasks_and_timeout_verbatim() -> None:
    """CW1 (T-0160's regime): W1 and W2 verbatim, n = 5, a 60 s turn timeout, seed 0."""
    code = _code(_text())
    for prompt in oracle.PROMPTS.values():
        assert prompt in code, prompt
    assert re.search(r"\bN=5\b", code) and re.search(r"TURN_TIMEOUT_S=60\b", code)
    assert "Random(0)" in _all_code()


# --------------------------------------------------------------------------------------------- refusals (prelude)


def _refused(run: Run, *words: str) -> None:
    assert run.proc.returncode == 2, (run.proc.returncode, run.output[-3000:])
    assert "REFUSED" in run.proc.stderr, run.output[-3000:]
    for w in words:
        assert w in run.proc.stderr, (w, run.proc.stderr)
    assert run.sessions == [] and run.uv == [], "a refusal comes before any uv call or session"


def test_refuses_arguments(tmp_path: Path, sysbin: Path) -> None:
    _refused(scratch(tmp_path, sysbin, "t0174", env=HEALTHY, args=["--n", "3"]), "no arguments")


def test_refuses_a_dirty_tree(tmp_path: Path, sysbin: Path) -> None:
    run = scratch(
        tmp_path, sysbin, "t0174", env=HEALTHY, before=lambda r: (r / "stray.txt").write_text("x")
    )
    _refused(run, "dirty")


@pytest.mark.parametrize(
    "merges",
    [
        [MERGE_160],
        [MERGE_160, "Merge T-0174-a tests: activation rows (fixture)"],
        [MERGE_160, "Merge T-0174-a"],
        [MERGE_160, "Merge T-0174-ab: a look-alike (fixture)"],
    ],
    ids=["no_t0174a", "tests_only", "bare_name", "look_alike"],
)
def test_refuses_without_the_t0174a_merge(tmp_path: Path, sysbin: Path, merges: list[str]) -> None:
    """CW1's extra refusal: HEAD must contain the ``Merge T-0174-a:`` merge (so ``--live-guidance`` exists). With
    healthy services, the script would otherwise go on to run sessions."""
    _refused(scratch(tmp_path, sysbin, "t0174", merges=merges, env=HEALTHY), "Merge T-0174-a")


def test_keeps_the_t0160_impl_merge_refusal(tmp_path: Path, sysbin: Path) -> None:
    """T-0160's own merge refusal is kept (a copy of its refusals)."""
    run = scratch(tmp_path, sysbin, "t0174", merges=[MERGE_174A], env=HEALTHY)
    _refused(run, "Merge T-0160 group C impl")


@pytest.mark.parametrize("var", PROXIES)
def test_refuses_a_proxy_variable(tmp_path: Path, sysbin: Path, var: str) -> None:
    run = scratch(tmp_path, sysbin, "t0174", env={**HEALTHY, var: "http://127.0.0.1:3128"})
    _refused(run, var)


def test_refuses_without_strace(tmp_path: Path, sysbin: Path) -> None:
    _refused(scratch(tmp_path, sysbin, "t0174", env=HEALTHY, strace=False), "strace")


@pytest.mark.parametrize("port", ["8000", "1420"])
def test_refuses_a_bound_port(tmp_path: Path, sysbin: Path, port: str) -> None:
    _refused(
        scratch(tmp_path, sysbin, "t0174", env={**HEALTHY, "STUB_PREBOUND": port}), f"port {port}"
    )


@pytest.mark.parametrize(
    ("down", "word"),
    [("CURL_ENGINE", "engine"), ("CURL_TTS", "tts"), ("CURL_STT", "stt")],
)
def test_refuses_an_unhealthy_service(tmp_path: Path, sysbin: Path, down: str, word: str) -> None:
    _refused(scratch(tmp_path, sysbin, "t0174", env={**HEALTHY, down: "000"}), word)


def test_refuses_an_existing_output_dir(tmp_path: Path, sysbin: Path) -> None:
    """CW1: one run per SHA in ``logs/t0174-web-ab-<sha7>/`` (an empty dir does not dirty the tree)."""
    made: list[str] = []

    def mk(repo: Path) -> None:
        name = f"t0174-web-ab-{_sha7(repo)}"
        (repo / "logs" / name).mkdir(parents=True)
        made.append(name)

    run = scratch(tmp_path, sysbin, "t0174", env=HEALTHY, before=mk)
    _refused(run, made[0], "exists")


# --------------------------------------------------------------------------------------------- the run (stubbed)


def test_each_arm_passes_exactly_its_guidance_and_the_t0160_flags(
    ok_runs: Callable[[str], Run],
) -> None:
    """CW1: 20 sessions; each passes ``--live-guidance`` exactly once, v1 in arm A's turn dirs and v2 in arm B's;
    with that pair removed, the whole strace + session command line equals the T-0160 smoke's (per-turn paths
    normalised)."""
    ref = ok_runs("t0160")
    assert ref.sessions, ref.output[-2000:]
    want = _normalise(ref.sessions[0]["argv"])
    run = ok_runs("t0174")
    assert len(run.sessions) == 2 * len(TASKS) * N, (len(run.sessions), run.output[-3000:])
    for s in run.sessions:
        argv = s["argv"]
        arm, _, _ = _turn_of(argv)
        assert _guidance_of(argv) == [GUIDANCE[arm]], (arm, argv)
        assert _normalise(_without_guidance(argv)) == want, (argv, want)


def test_twenty_trials_jointly_shuffled_with_seed_0(ok_runs: Callable[[str], Run]) -> None:
    """CW1: W1 and W2 x n = 5 x 2 arms, in one order shuffled with seed 0 (written to ``order.json`` as
    ``[arm, task, rep]`` before any turn); sessions and probes run in that order, each probe with its task's prompt
    and ``--timeout-s 60``; the output dir is ``logs/t0174-web-ab-<sha7>/`` with HEAD's sha."""
    run = ok_runs("t0174")
    out = run.out_dir("t0174-web-ab")
    assert out.name == f"t0174-web-ab-{_sha7(run.repo)}", out
    order = [tuple(x) for x in json.loads((out / "order.json").read_text())]
    assert order in [[tuple(x) for x in o] for o in _oracle_orders()], order
    assert [_turn_of(s["argv"]) for s in run.sessions] == order
    probes = [u["argv"] for u in run.uv if any(a.endswith("text_mode_probe.py") for a in u["argv"])]
    assert len(probes) == len(order), probes
    for (_, task, _), argv in zip(order, probes, strict=True):
        assert oracle.PROMPTS[task] in argv and "60" in argv[argv.index("--timeout-s") + 1 :][:1], (
            argv
        )


def test_arm_recorded_per_trial_and_not_void(ok_runs: Callable[[str], Run]) -> None:
    """CW1/CW2: ``trials.jsonl`` holds one line per (arm, task, repeat), and ``report.json`` (the analysis's
    return value) has both arms and no void finding when every session printed its arm's guidance line."""
    run = ok_runs("t0174")
    assert run.proc.returncode == 0, run.output[-3000:]
    out = run.out_dir("t0174-web-ab")
    trials = [json.loads(x) for x in (out / "trials.jsonl").read_text().splitlines() if x.strip()]
    keys = sorted((t["arm"], t["task"], int(t["repeat"])) for t in trials)
    assert keys == sorted((a, t, r) for a in GUIDANCE for t in TASKS for r in range(N)), keys
    report = json.loads((out / "report.json").read_text())
    assert set(report["arms"]) == {"A", "B"} and report["void"] == [], report


def test_void_when_a_session_runs_the_other_guidance(tmp_path: Path, sysbin: Path) -> None:
    """CW1: every session announces the other arm's version (``live guidance: v2 (flag)`` in arm A and v1 in arm
    B): the report is void. CW3 ruling: void is a report finding, so the completed run still exits 0."""
    run = scratch(tmp_path, sysbin, "t0174", mode="swap_guidance", env=HEALTHY)
    report = json.loads((run.out_dir("t0174-web-ab") / "report.json").read_text())
    assert report["void"], report
    assert run.proc.returncode == 0, (run.proc.returncode, run.output[-3000:])


def test_no_eval_call(ok_runs: Callable[[str], Run]) -> None:
    """CW3 ruling: the script calls no eval module (the T-0160 eval assumes one trial per (task, rep)): no ``uv``
    call runs ``pytest``, and the script's code names no ``tests/eval`` module and no ``T0160_RUN_DIR``."""
    run = ok_runs("t0174")
    assert run.proc.returncode == 0, run.output[-3000:]
    assert run.uv, "the stubbed run made no uv call at all"
    calls = [u["argv"] for u in run.uv if any("pytest" in a for a in u["argv"])]
    assert calls == [], calls
    code = _code(_text())
    assert "tests/eval" not in code and "T0160_RUN_DIR" not in code and "pytest" not in code


def test_pythonunbuffered(ok_runs: Callable[[str], Run]) -> None:
    """Every session, probe and analysis call runs with ``PYTHONUNBUFFERED=1`` (the ``python -c`` order
    one-liners excepted, as in T-0160)."""
    run = ok_runs("t0174")
    assert run.sessions and all(s["unbuffered"] == "1" for s in run.sessions), run.sessions[:2]
    rest = [u for u in run.uv if "-c" not in u["argv"]]
    assert rest and all(u["unbuffered"] == "1" for u in rest), [
        u for u in rest if u["unbuffered"] != "1"
    ]


@pytest.mark.parametrize("case", list(HARD_STOPS))
def test_hard_stops(tmp_path: Path, sysbin: Path, case: str) -> None:
    """CW1: each of the T-0160 smoke's hard stops (exit 1), under the same stub mode as the control row."""
    _hard_stop(tmp_path, sysbin, "t0174", case)


def test_premise_oracle_orders_differ_from_unshuffled() -> None:
    """Active premise: six distinct enumerations; every accepted order holds the 20 triples, is none of the
    enumerations (so an unshuffled order fails), and differs from each enumeration's seed-1 shuffle."""
    enums = _enumerations()
    assert len({tuple(e) for e in enums}) == 6
    full = sorted((a, t, r) for a in GUIDANCE for t in TASKS for r in range(N))
    seed1 = []
    for e in enums:
        e1 = list(e)
        random.Random(1).shuffle(e1)
        seed1.append(e1)
    for o in _oracle_orders():
        assert sorted(o) == full
        assert o not in enums and o not in seed1
