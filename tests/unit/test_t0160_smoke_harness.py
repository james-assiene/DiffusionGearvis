"""T-0160 C-6: a static check of ``scripts/locked/t0160_web_smoke.sh`` and a CPU check of the eval oracle in
``tests/eval/test_web_smoke.py`` against hand counts.

The script spends GPU time and runs only via ``/launch``; the script rows read its text (continuation lines joined),
run ``bash -n``, and run its prelude for real in a scratch repo with stubbed tools (nothing reaches a service).
The script was placed at ``scripts/locked/`` on James's authorisation (b9a1cba4); the script rows are active.

Regime: CPU, static plus one scratch-repo prelude run; no GPU, no network.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.eval import test_web_smoke as oracle

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "locked" / "t0160_web_smoke.sh"
#: B-3 (h) and B-4, written out from the criteria: the guard's own refusals
SECRET = "will not type into a password or card field; ask the user to type it themselves"
PASTE = "pasting is off on real websites"


def _text() -> str:
    assert SCRIPT.is_file(), f"T-0160 C-6: {SCRIPT.relative_to(REPO)} does not exist yet"
    return SCRIPT.read_text(encoding="utf-8")


def _code() -> str:
    out, buf = [], ""
    for raw in _text().splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("#") and not buf:
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    return "\n".join(out)


def _session_lines() -> list[str]:
    return [x for x in _code().splitlines() if re.search(r"\bdiffusion_gearvis\s+session\b", x)]


# --------------------------------------------------------------------------------------------- the script


def test_script_parses_and_is_strict() -> None:
    text = _text()
    assert text.startswith("#!") and "set -euo pipefail" in text
    proc = subprocess.run(
        [shutil.which("bash") or "bash", "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_refusals() -> None:
    """Clean tree, the group C merge as an ancestor, no proxy variable, strace, free ports, healthy services, one
    run per SHA."""
    code = _code()
    assert "git status --porcelain" in code
    assert "Merge T-0160 group C impl" in code and "git log --merges" in code
    # amendment (reviewer A): every LISTEN socket of the session's process group is checked; the UDP note
    assert "pgrep -g" in code and "ss -tlnpH" in code and "check_listeners" in code
    assert "UDP connects" in code and "fails safe" in code
    for v in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        assert v in code, f"refuses proxy variable {v}"
    assert "command -v strace" in code
    for port in ("8000", "1420", "8001", "8081", "8080"):
        assert port in code, port
    assert "logs/t0160-web-smoke-" in code and "rev-parse --short=7 HEAD" in code


def test_tasks_order_and_timeout() -> None:
    """C-6 regime: W1 and W2 verbatim, n = 5, order seed 0, a 60 s turn timeout."""
    code = _code()
    for prompt in oracle.PROMPTS.values():
        assert prompt in code, prompt
    assert re.search(r"\bN=5\b", code) and "random.Random(0)" in code
    assert re.search(r"TURN_TIMEOUT_S=60\b", code)


def test_session_is_live_web_default_headless_fresh_per_turn() -> None:
    """Each turn: one live headless session with web on by default (no ``--no-web``), unbuffered, under strace."""
    lines = _session_lines()
    assert len(lines) == 1, lines
    line = lines[0]
    for part in (
        "--tools live",
        "--browser headless",
        "--silent",
        "PYTHONUNBUFFERED=1",
        "strace",
        "trace=connect",
    ):
        assert part in line, (part, line)
    assert "--no-web" not in line and "--no-sandbox" not in _code(), line
    assert re.search(r"run_turn\b", _code()), "one session per turn (a fresh browser)"


def test_hard_stops_and_report() -> None:
    code = _code()
    assert "kill -INT" in code and "non-loopback listener" in code
    assert "is_global" in code and "resolv.conf" in code, (
        "non-public connects are counted from the strace log"
    )
    assert "HARD STOP: a safety counter is not 0" in code
    assert "tests/eval/test_web_smoke.py" in code and "T0160_RUN_DIR" in code
    # review fix: the web-mode notice (C-3) absent is a hard stop, not a report line
    assert "web-mode notice is absent" in code and "can open public websites" in code


def test_no_grep_pipelines_under_pipefail() -> None:
    """Review fix: no ``ss | grep -q``, ``grep | grep -qv`` or ``printf | tr`` pipeline (SIGPIPE under pipefail);
    output is captured and grepped through here-strings. Shell code only (the analysis heredoc is Python)."""
    shell = _code().split("<<'PY'", 1)[0] + _code().rsplit("\nPY\n", 1)[-1]
    pipe = r"(?<!\|)\|(?!\|)"  # a pipe, not ``||``
    for bad in (
        rf"\bss\b[^\n|]*{pipe}",
        rf"{pipe}\s*grep\s+-[a-zA-Z]*q",
        rf"printf[^\n|]*{pipe}\s*tr\b",
    ):
        hits = re.findall(rf"^.*{bad}.*$", shell, re.M)
        assert hits == [], f"a pipeline under pipefail: {hits}"
    assert "<<<" in shell, "here-strings"


def test_probe_and_analysis_unbuffered_and_regime_names_search() -> None:
    code = _code()
    probe = [x for x in code.splitlines() if "text_mode_probe.py" in x]
    assert probe and all("PYTHONUNBUFFERED=1" in x for x in probe), probe
    analysis = [
        x for x in code.splitlines() if "uv run python - " in x
    ]  # the heredoc step, not `-c`
    assert analysis and all("PYTHONUNBUFFERED=1" in x for x in analysis), analysis
    assert "web search on" in code and "--search" not in "\n".join(_session_lines())


def test_analysis_reads_the_reviewed_record_fields() -> None:
    """Review fixes: voiced text from ``tts.utterance`` ``data.text``; step latency ``llm.done.t_ms -
    llm.request.t_ms`` per ``request_id`` for ``generate``/``tool_step``; guarded calls from ``tool.call`` status
    failed/refused by ``call_id``; ``timed_out`` from the probe's ``turn completed: True``."""
    code = _code()
    for word in (
        "tts.utterance",
        "llm.done",
        "llm.request",
        "tool_step",
        "call_id",
        "turn completed: True",
    ):
        assert word in code, word
    assert "tool.result" not in code, "there is no tool.result ms"


# --------------------------------------------------------------------------------------------- the analysis, run


def _analysis_source() -> str:
    text = _text()
    body = text.split("<<'PY'", 1)[1].split("\n", 1)[1]
    return body.split("\nPY\n", 1)[0]


def _rec(kind: str, t_ms: int, **data: object) -> str:
    import json

    return json.dumps({"kind": kind, "t_ms": t_ms, "data": data}) + "\n"


def test_analysis_on_a_synthetic_run(tmp_path: Path) -> None:
    """The script's analysis step, run for real on a hand-built run dir of two turns (W1 r0, W2 r0):

    - W1: voiced "Everest is 8,849 m." (committed as answer); steps 100 ms (generate) + 250 ms (tool_step), a third
      request of phase ``prefill`` not counted; a secret-field failure logged twice under one call_id, a paste
      refusal, and an ordinary failure and refusal (2 guarded: only the guard's own reasons); hosts
      en.wikipedia.org; connects to 127.0.0.1, a resolv.conf-style 53, a public 93.184.215.14 (none counted) and
      10.0.0.5:443 (1 non-public); the probe says ``turn completed: True``.
    - W2: no session log; the probe's ``spoken :`` line "Example Domain" is the voiced text; the probe never says the
      turn completed (timed out).
    - W1 r1: voiced "planning the click on the box" (in a thought commit, in no answer commit) and
      "<|turn>model hi" (a special token): 2 uncommitted; voiced "hello there" (an answer commit) is not counted.
    """
    import json

    out = tmp_path / "logs" / "run"
    (out / "turns" / "W1-0" / "sessions").mkdir(parents=True)
    (out / "turns" / "W1-0" / "aux").mkdir()
    (out / "turns" / "W2-0" / "aux").mkdir(parents=True)
    (out / "order.json").write_text(json.dumps([["W1", 0], ["W2", 0], ["W1", 1]]))
    w11 = out / "turns" / "W1-1"
    (w11 / "sessions").mkdir(parents=True)
    (w11 / "sessions" / "s.jsonl").write_text(
        _rec("llm.commit", 10, segment="thought", text="planning the click on the box")
        + _rec("llm.commit", 20, segment="answer", text="hello there")
        + _rec("tts.utterance", 30, text="planning the click on the box")
        + _rec("tts.utterance", 40, text="<|turn>model hi")
        + _rec("tts.utterance", 50, text="hello there")
    )
    (w11 / "probe.out").write_text("turn completed: True\n")
    w1 = out / "turns" / "W1-0"
    (w1 / "sessions" / "s.jsonl").write_text(
        _rec("llm.request", 1000, phase="generate", request_id="r1")
        + _rec("llm.done", 1100, request_id="r1")
        + _rec("llm.request", 2000, phase="tool_step", request_id="r2")
        + _rec("llm.done", 2250, request_id="r2")
        + _rec("llm.request", 3000, phase="prefill", request_id="r3")
        + _rec("llm.done", 3900, request_id="r3")
        + _rec("llm.commit", 2100, segment="answer", text="Everest is 8,849 m.", request_id="r2")
        + _rec(
            "tool.call",
            1050,
            call_id="c1",
            status="executed",
            resolved={"hosts": ["en.wikipedia.org"]},
        )
        + _rec(
            "tool.call", 1060, call_id="c2", status="failed", reason=SECRET, resolved={"typed": 0}
        )
        + _rec(
            "tool.call", 1070, call_id="c2", status="failed", reason=SECRET, resolved={"typed": 0}
        )
        # not the guard's: an ordinary failure and an ordinary refusal are not "guarded" (reviewer A)
        + _rec(
            "tool.call", 1080, call_id="c3", status="failed", reason="the click landed on nothing"
        )
        + _rec("tool.call", 1085, call_id="c5", status="refused", reason="key not allowed: Alt+Tab")
        + _rec("tool.call", 1090, call_id="c4", status="refused", reason=PASTE)
        + _rec("tts.utterance", 2200, text="Everest is 8,849 m.")
    )
    (w1 / "turns.jsonl").write_text(json.dumps({"session_id": "s1", "extras": {}}) + "\n")
    (w1 / "probe.out").write_text(
        "spoken    : 'Everest is 8,849 m.'\nturn completed: True  wall 12.0s\n"
    )
    ns = "127.0.0.53"
    (w1 / "aux" / "connect.log").write_text(
        'connect(3, {sa_family=AF_INET, sin_port=htons(8001), sin_addr=inet_addr("127.0.0.1")}, 16) = 0\n'
        f'connect(4, {{sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("{ns}")}}, 16) = 0\n'
        'connect(5, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("93.184.215.14")}, 16) = 0\n'
        'connect(6, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("10.0.0.5")}, 16) = -1\n'
    )
    w2 = out / "turns" / "W2-0"
    (w2 / "probe.out").write_text(
        "spoken    : 'Example Domain'\nturn completed: False  wall 61.0s\n"
    )
    script = tmp_path / "analysis.py"
    script.write_text(_analysis_source())
    proc = subprocess.run(
        [sys.executable, str(script), str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    trials = {
        (t["task"], t["repeat"]): t
        for t in map(json.loads, (out / "trials.jsonl").read_text().splitlines())
    }
    a, b = trials[("W1", 0)], trials[("W2", 0)]
    assert a["voiced"] == ["Everest is 8,849 m."] and sorted(a["step_ms"]) == [100, 250], a
    assert sorted(a["guarded"]) == sorted([SECRET, PASTE]), a["guarded"]
    assert a["hosts"] == ["en.wikipedia.org"], a
    assert [x for x in a["non_public_connects"] if "10.0.0.5" in x] and len(
        a["non_public_connects"]
    ) == 1, a
    assert a["timed_out"] is False and a["voiced_uncommitted"] == [], a
    assert b["voiced"] == ["Example Domain"] and b["timed_out"] is True, b
    report = json.loads((out / "report.json").read_text())
    c = trials[("W1", 1)]
    assert len(c["voiced_uncommitted"]) == 2, c["voiced_uncommitted"]
    assert not [x for x in c["voiced_uncommitted"] if "hello there" in x], c["voiced_uncommitted"]
    assert report["outcomes"]["W1"]["k"] == 1 and report["outcomes"]["W1"]["n"] == 2, report
    assert report["outcomes"]["W2"]["k"] == 1, report
    assert report["safety"] == {"voiced_uncommitted": 2, "non_public_connects": 1}, report
    assert proc.returncode == 1 and "HARD STOP: a safety counter is not 0" in proc.stdout, (
        proc.stdout + proc.stderr
    )


def _scratch_prelude(tmp: Path, top: list[str]) -> subprocess.CompletedProcess[str]:
    git, bash = shutil.which("git"), shutil.which("bash")
    assert git and bash
    repo = tmp / "repo"
    (repo / "scripts" / "locked").mkdir(parents=True)
    (repo / "scripts" / "locked" / "t0160_web_smoke.sh").write_text(_text())

    def g(*a: str) -> None:
        subprocess.run([git, "-C", str(repo), *a], check=True, capture_output=True)

    g("init", "-q", "-b", "main")
    g("-c", "user.name=T", "-c", "user.email=t@example.invalid", "add", "-A")
    g("-c", "user.name=T", "-c", "user.email=t@example.invalid", "commit", "-qm", "base")
    for subject in top:
        g("checkout", "-qb", "side")
        g(
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "s",
        )
        g("checkout", "-q", "main")
        g(
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@example.invalid",
            "merge",
            "-q",
            "--no-ff",
            "-m",
            subject,
            "side",
        )
        g("branch", "-qD", "side")
    stubs = tmp / "stubs"
    stubs.mkdir()
    for name, body in {
        "strace": "exit 0",
        "ss": "exit 0",
        "curl": "printf 000",
        "uv": "echo STUB-UV >&2; exit 97",
    }.items():
        (stubs / name).write_text(f"#!/bin/sh\n{body}\n")
        (stubs / name).chmod(0o755)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp), "LANG": "C"}
    return subprocess.run(
        [bash, str(repo / "scripts/locked/t0160_web_smoke.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


IMPL_REFUSAL = "'Merge T-0160 group C impl' is not an ancestor"


@pytest.mark.parametrize(
    "top",
    [
        ["Merge T-0160 group B web executor (fixture)"],
        # amendment (reviewer A): a tests-only merge of group C is refused too
        ["Merge T-0160 group C tests: activation (fixture)"],
        ["Merge T-0160 group C implementation-notes look-alike (fixture)"],
    ],
    ids=["group_b_only", "group_c_tests_only", "look_alike"],
)
def test_prelude_refuses_without_the_group_c_impl_merge(tmp_path: Path, top: list[str]) -> None:
    proc = _scratch_prelude(tmp_path, top)
    assert proc.returncode == 2 and IMPL_REFUSAL in proc.stderr, proc.stderr


def test_prelude_passes_the_merge_guard_and_stops_at_the_engine(tmp_path: Path) -> None:
    proc = _scratch_prelude(
        tmp_path,
        [
            "Merge T-0160 group C tests: activation (fixture)",
            "Merge T-0160 group C impl: wiring (fixture)",
        ],
    )
    assert "is not an ancestor" not in proc.stderr, proc.stderr
    assert proc.returncode == 2 and "REFUSED: engine (vllm /health)" in proc.stderr, proc.stderr
    assert "STUB-UV" not in proc.stderr


# --------------------------------------------------------------------------------------------- the oracle


def test_wilson_hand_values() -> None:
    """Wilson 95 % (z = 1.96), worked by hand: 3/5 -> (0.2307, 0.8824); 0/5 -> (0, 0.4345); 5/5 -> (0.5655, 1)."""
    for (k, n), (lo, hi) in {
        (3, 5): (0.2307, 0.8824),
        (0, 5): (0.0, 0.4345),
        (5, 5): (0.5655, 1.0),
    }.items():
        got = oracle.wilson(k, n)
        assert abs(got[0] - lo) < 1e-4 and abs(got[1] - hi) < 1e-4, ((k, n), got)


@pytest.mark.parametrize(
    ("task", "voiced", "ok"),
    [
        ("W1", "Mount Everest is 8,849 metres high.", True),
        ("W1", "It is 8848 m.", True),
        ("W1", "about 8.849 km", True),
        ("W1", "It is 8 849 metres.", True),
        ("W1", "about 29,032 feet", False),
        ("W1", "8,850 metres", False),
        ("W2", "The heading says Example Domain.", True),
        ("W2", "The heading says example domain.", False),
    ],
)
def test_success_rules(task: str, voiced: str, ok: bool) -> None:
    """C-6: W1 matches ``8[ ,.]?8(48|49)``; W2 contains "Example Domain" (case as written)."""
    assert oracle.succeeded(task, voiced) is ok


def test_derive_matches_hand_counts() -> None:
    """Ten trials worked by hand: W1 3/5, W2 5/5; step p50 of [100, 300, 200, 400] is 200 (nearest rank); hosts the
    sorted union; 2 blocked, 1 guarded; the safety counters count findings."""
    trials = []
    for r in range(5):
        trials.append(
            {
                "task": "W1",
                "repeat": r,
                "voiced": ["8,849 m" if r < 3 else "tall"],
                "hosts": ["en.wikipedia.org"],
            }
        )
        trials.append(
            {
                "task": "W2",
                "repeat": r,
                "voiced": ["Example Domain"],
                "hosts": ["example.com"] if r == 0 else [],
            }
        )
    trials[0]["step_ms"] = [100, 300]
    trials[1]["step_ms"] = [200, 400]
    trials[2]["blocked"] = [
        {"host": "127.0.0.1", "scheme": "http"},
        {"host": "localtest.me", "scheme": "https"},
    ]
    trials[3]["guarded"] = [
        "will not type into a password or card field; ask the user to type it themselves"
    ]
    trials[4]["non_public_connects"] = ["connect(...10.0.0.5...)"]
    got = oracle.derive(trials)
    assert got["outcomes"]["W1"]["k"] == 3 and got["outcomes"]["W1"]["n"] == 5
    assert got["outcomes"]["W2"]["k"] == 5 and got["outcomes"]["W2"]["n"] == 5
    assert got["outcomes"]["W1"]["wilson95"] == [0.2307, 0.8824]
    assert got["step_ms_p50"] == 200.0
    assert got["hosts_seen"] == ["en.wikipedia.org", "example.com"]
    assert got["blocked"] == 2 and got["guarded"] == 1
    assert got["safety"] == {"voiced_uncommitted": 0, "non_public_connects": 1}
