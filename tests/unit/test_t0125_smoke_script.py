"""T-0125 criterion 13: a static check of ``scripts/locked/t0125_search_smoke.sh`` (refusals, phases, hard stops).

The script spends GPU time and is run only via ``/launch``; these rows read its text (continuation lines joined)
and run ``bash -n`` on it. They pin what the criterion names, not how the script is written: each refusal and hard
stop must appear as a check the script makes, and phase 2 must run the default (no ``--search``). Expected strings
are written out from the criterion; the proxy variable names come from the backend's ``PROXY_ENV`` (T-0111b), which
is what the script must mirror.

Written ahead of T-0125 as pending rows (at BASE the script did not exist); activated with the implementation.

T-0168 (amends T-0125 13.1 and 13.3): the script runs on current main. The ancestor refusal for
``Merge T-0160 group C`` becomes a requirement for ``Merge T-0160 group C impl`` (the guard of
``t0160_web_smoke.sh``), phase 2 passes ``--no-web`` and hard-stops on any sign of web mode, and the regime says
so. ``test_prelude_refuses_after_t0160_group_c`` is replaced (a spec change, not edited to fit the code).

Regime: CPU; static rows read the text, the T2 rows run the real prelude in a scratch repo with stubbed
``strace``/``ss``/``curl``/``uv`` (nothing reaches the network or the GPU).
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "locked" / "t0125_search_smoke.sh"
PROMPTS = (
    "Search the web for the height of the Eiffel Tower and tell me in one sentence.",
    "Search the web for today's weather in Paris.",
    "Search the web for the latest Python release.",
)


def _text() -> str:
    assert SCRIPT.is_file(), f"T-0125 criterion 13: {SCRIPT.relative_to(REPO)} does not exist yet"
    return SCRIPT.read_text(encoding="utf-8")


def _joined() -> list[str]:
    """Logical lines: backslash continuations joined, comments dropped."""
    out: list[str] = []
    buf = ""
    for raw in _text().splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("#") and not buf:
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def _code() -> str:
    return "\n".join(_joined())


def _session_lines() -> list[str]:
    """The real session starts. A ``--dry-run`` command (T-0168's CPU pre-flight) prints its configuration and
    exits, so it is not a session start and is excluded (``--tools dry-run`` is a tool mode, not this flag)."""
    return [
        line
        for line in _joined()
        if re.search(r"\b(?:diffusion_gearvis|astra)\s+session\b", line)
        and "--tools" in line
        and not re.search(r"(?<!\S)--dry-run(?!\S)", line)
    ]


def test_script_parses_and_is_strict() -> None:
    text = _text()
    assert text.startswith("#!"), "a shebang"
    assert re.search(r"set -[a-z]*e[a-z]*u|set -euo pipefail|set -eu", text), "set -e/-u"
    bash = shutil.which("bash")
    assert bash is not None
    proc = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_refuses_a_dirty_tree() -> None:
    code = _code()
    assert re.search(r"git (?:-C \S+ )?status --porcelain|git diff --quiet", code), (
        "T-0125 criterion 13.1: the script refuses a dirty tree"
    )


def test_merge_ancestry_guards() -> None:
    """T-0168 criterion 1 (amends T-0125 13.1): `Merge T-0125` and `Merge T-0160 group C impl` are both required;
    the old "group C is an ancestor" refusal is gone; the impl check reads a captured history via a here-string."""
    code = _code()
    assert "Merge T-0125" in code, (
        "T-0125 criterion 13.1: a `Merge T-0125` must be an ancestor of HEAD"
    )
    assert "Merge T-0160 group C impl" in code, (
        "T-0168 criterion 1: a `Merge T-0160 group C impl` must be an ancestor of HEAD"
    )
    assert re.search(r"merge-base --is-ancestor|git log[^\n]*--merges", code), (
        "T-0125 criterion 13.1: ancestry is checked against HEAD's history"
    )
    # the removed refusal: no grep pattern on `group C` that is not followed by ` impl`, and not its message
    old = [
        line
        for line in _joined()
        if re.search(r"\bgrep\b", line) and re.search(r"Merge T-0160 group C(?! impl)", line)
    ]
    assert old == [] and T0160_REFUSAL not in code, (
        f"T-0168 criterion 1: the refusal for `Merge T-0160 group C` being an ancestor is removed; {old}"
    )
    impl_checks = [
        line for line in _joined() if "group C impl" in line and re.search(r"\bgrep\b", line)
    ]
    assert impl_checks, "T-0168 criterion 1: a grep for `Merge T-0160 group C impl`"
    for line in impl_checks:
        assert "<<<" in line and "|" not in line.split("grep", 1)[0], (
            f"T-0168 criterion 1 (T-0125 amendment 3 (f)): here-string over captured MERGES, no pipe; {line}"
        )
    refusals = [line for line in _joined() if "REFUSED" in line and "group C impl" in line]
    assert refusals and all(re.search(r"T-0168|13\.1", line) for line in refusals), (
        f"T-0168 criterion 1: the impl refusal message cites T-0168 / T-0125 13.1 (amended); {refusals}"
    )


def test_refuses_proxy_variables() -> None:
    from local_astra.search.ddgs_backend import PROXY_ENV

    code = _code()
    missing = [name for name in PROXY_ENV if name not in code]
    assert missing == [], (
        f"T-0125 criterion 13.1: the script refuses each proxy variable; missing {missing}"
    )


def test_refuses_without_strace_and_with_ports_bound() -> None:
    code = _code()
    assert re.search(r"command -v strace|which strace|type -P strace", code), (
        "T-0125 criterion 13.1: refuses when strace is absent"
    )
    for port in ("8000", "1420"):
        assert port in code, (
            f"T-0125 criterion 13.1: port {port} must be free (and checked after SIGINT)"
        )
    assert "SIGINT" in code or "kill -INT" in code or "kill -s INT" in code, (
        "the session is stopped with SIGINT"
    )


def test_refuses_unhealthy_services() -> None:
    code = _code().lower()
    for what in ("tts", "stt"):
        assert what in code, f"T-0125 criterion 13.1: the {what.upper()} health check"
    assert "health" in code or "status" in code or "doctor" in code, "a health probe"


def test_phase1_dry_run_opt_in_under_strace() -> None:
    lines = _session_lines()
    phase1 = [line for line in lines if "--tools dry-run" in line]
    assert len(phase1) == 1, f"T-0125 criterion 13.2: one dry-run session command; {phase1}"
    line = phase1[0]
    for part in (
        "--search ddgs",
        "--silent",
        "--serve-hud",
        "strace",
        "-f",
        "trace=connect",
        "connect.log",
    ):
        assert part in line, f"T-0125 criterion 13.2: phase 1 runs with {part!r}; {line}"
    code = _code()
    assert "text_mode_probe.py" in code
    for prompt in PROMPTS:
        assert prompt in code, f"T-0125 criterion 13.2: the typed turn {prompt!r}"


NO_WEB = re.compile(r"(?<!\S)--no-web(?!\S)")


def test_phase1_has_no_no_web() -> None:
    """T-0168 criterion 5: phase 1 is unchanged; dry-run has no web mode, so its command has no ``--no-web``."""
    phase1 = [line for line in _session_lines() if "--tools dry-run" in line]
    assert len(phase1) == 1, f"one dry-run session command; {phase1}"
    assert "--no-web" not in phase1[0], f"T-0168 criterion 5: phase 1 has no --no-web; {phase1[0]}"


def test_phase2_live_default_passes_no_search_flag() -> None:
    lines = _session_lines()
    phase2 = [line for line in lines if "--tools live" in line]
    assert len(phase2) == 1, f"T-0125 criterion 13.3: one live session command; {phase2}"
    assert NO_WEB.search(phase2[0]), (
        f"T-0168 criterion 2: phase 2 runs live with --no-web (web off); {phase2[0]}"
    )
    assert "--browser headless" in phase2[0], phase2[0]
    assert "--search" not in phase2[0], (
        f"T-0125 criterion 13.3: phase 2 runs the default (no --search flag); {phase2[0]}"
    )
    assert "search: ddgs (default)" in _code(), (
        "T-0125 criterion 13.3: phase 2 checks the search line reads `search: ddgs (default)`"
    )


def _web_notice_prefix() -> str:
    """The start of ``cli.WEB_STARTUP_NOTICE`` after the ``astra session: `` prefix, up to its first colon
    (``web browsing is on`` as the task names it), read from ``cli`` so a reworded notice reddens this row."""
    from local_astra.cli import WEB_STARTUP_NOTICE

    lead = "astra session: "
    assert WEB_STARTUP_NOTICE.startswith(lead), WEB_STARTUP_NOTICE
    prefix = WEB_STARTUP_NOTICE[len(lead) :].split(":", 1)[0]
    assert len(prefix) >= 10, prefix
    return prefix


_READS_SESSION_OUT = re.compile(r"session_out|session\.out")


def _egress_greps() -> list[str]:
    return [line for line in _joined() if re.search(r"\bgrep\b", line) and "egress=" in line]


def test_no_egress_grep_on_real_start() -> None:
    """T-0168 criterion 3 (amended): a real start prints no tools line (``cli.tools_line`` is printed only under
    ``--dry-run``), so no ``egress=`` grep may read a real start's ``session.out``."""
    bad = [line for line in _egress_greps() if _READS_SESSION_OUT.search(line)]
    assert bad == [], (
        f"T-0168 criterion 3: an egress= grep reads the real start's session.out; {bad}"
    )


def _preflight_pattern(needle: str) -> tuple[str, str]:
    """(grep flags, pattern) of the pre-flight's grep on ``needle`` (the one egress grep not on session.out)."""
    found: list[tuple[str, str]] = []
    for line in _egress_greps():
        if _READS_SESSION_OUT.search(line):
            continue
        for m in re.finditer(r"\bgrep((?:\s+-[A-Za-z]+)*)\s+'([^']*)'", line):
            if needle in m.group(2):
                found.append((m.group(1), m.group(2)))
    assert len(found) == 1, (
        f"T-0168 criterion 3a: one pre-flight grep on {needle!r} (not on session.out); {found}"
    )
    return found[0]


def _grep_matches(flags: str, pattern: str, text: str) -> bool:
    grep = shutil.which("grep")
    assert grep is not None
    args = [f for f in flags.split() if f]
    proc = subprocess.run([grep, *args, "--", pattern], input=text, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    return proc.returncode == 0


def _real_dry_run(tmp: Path, capsys: pytest.CaptureFixture[str], *, no_web: bool) -> str:
    """Phase 2's command plus ``--dry-run`` through the real ``cli.main``, in-process. Only
    ``cli.live_prerequisites`` is stubbed (the caller's monkeypatch): it checks Playwright's pinned headless shell
    is installed, which a fresh clone or CI may lack, and is not what this row tests. ``tools_line`` and every
    other step of the dry-run are the real ones."""
    from local_astra import cli

    argv = ["session", "--silent", "--serve-hud", "--tools", "live", "--browser", "headless"]
    argv += ["--no-web"] if no_web else []
    argv += ["--dry-run", "--data-dir", str(tmp / "data"), "--log-dir", str(tmp / "sessions")]
    capsys.readouterr()
    rc = cli.main(argv)
    out, err = capsys.readouterr()
    assert rc == 0, (rc, out, err)
    return out + err


def test_preflight_matches_the_real_dry_run_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """T-0168 criterion 3a against the real CLI's dry-run output (CPU, in-process; ``--dry-run`` opens nothing:
    measured 2026-10-27 under ``strace -e connect,execve``, zero connects, no data/log dir created). The script's
    own loopback pattern matches phase 2's dry-run output and its web pattern does not; with ``--no-web`` dropped
    the web pattern matches and the loopback one does not, so the pre-flight would refuse. Playwright's install
    check is stubbed (as in test_t0111g_smoke_harness / test_t0150_ab_harness), so the row is independent of it."""
    from local_astra import cli

    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None)
    lb_flags, lb = _preflight_pattern("egress=loopback")
    web_flags, web = _preflight_pattern("egress=web")
    off = _real_dry_run(tmp_path / "off", capsys, no_web=True)
    assert _grep_matches(lb_flags, lb, off), f"loopback pattern {lb!r} misses the real line\n{off}"
    assert not _grep_matches(web_flags, web, off), (
        f"web pattern {web!r} matches web-off output\n{off}"
    )
    on = _real_dry_run(tmp_path / "on", capsys, no_web=False)
    assert _grep_matches(web_flags, web, on), f"web pattern {web!r} misses the web-mode line\n{on}"
    assert not _grep_matches(lb_flags, lb, on), (
        f"loopback pattern {lb!r} matches web-mode output\n{on}"
    )


def _notice_check_fragment() -> str:
    """The script's own phase-2 notice check: the ``if … fi`` block (or the single logical line) whose ``grep``
    tests for the notice prefix, outside comments."""
    prefix = _web_notice_prefix()
    raw = _text().splitlines()
    at = [
        i
        for i, line in enumerate(raw)
        if prefix in line and re.search(r"\bgrep\b", line) and not line.lstrip().startswith("#")
    ]
    assert len(at) == 1, f"T-0168 criterion 3b: one check on {prefix!r}; lines {at}"
    i = at[0]
    if not re.match(r"\s*if\b", raw[i]):
        return raw[i]
    depth, out = 0, []
    for line in raw[i:]:
        out.append(line)
        depth += len(re.findall(r"(?:^|;|\s)if\s", " " + line))
        depth -= len(re.findall(r"(?:^|;|\s)fi\b", " " + line))
        if depth <= 0:
            break
    return "\n".join(out)


def _run_notice_check(tmp: Path, session_out: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    assert bash
    d = tmp / "phase2"
    d.mkdir(parents=True)
    (d / "session.out").write_text(session_out)
    prog = (
        "set -euo pipefail\n"
        f'name=phase2; dir={shlex.quote(str(d))}; session_out="$dir/session.out"; LOG={shlex.quote(str(tmp / "run.log"))}\n'
        + _notice_check_fragment()
        + "\necho CHECK-PASSED\n"
    )
    return subprocess.run([bash, "-c", prog], capture_output=True, text=True, timeout=30)


def test_phase2_notice_hard_stop(tmp_path: Path) -> None:
    """T-0168 criterion 3b, executed: the script's notice check run against a fixture ``session.out``. With
    ``cli.WEB_STARTUP_NOTICE`` present it exits 1 with ``HARD STOP: phase2:``; without it the check passes. Both
    fixtures carry the real search startup notice (``web search is on …``, which phase 2 does print) so a check
    wider than the web notice is caught too."""
    from local_astra.cli import WEB_STARTUP_NOTICE
    from tests.unit._t0125_pins import SEARCH_DEFAULT, STARTUP

    base = f"{STARTUP}\nregime: profile=lean\n{SEARCH_DEFAULT}\n"
    clean = _run_notice_check(tmp_path / "clean", base)
    assert clean.returncode == 0 and "CHECK-PASSED" in clean.stdout, (
        clean.returncode,
        clean.stdout,
        clean.stderr,
    )
    assert "HARD STOP" not in clean.stdout + clean.stderr, clean.stdout
    web = _run_notice_check(tmp_path / "web", f"{WEB_STARTUP_NOTICE}\n{base}")
    assert web.returncode == 1 and "CHECK-PASSED" not in web.stdout, (
        web.returncode,
        web.stdout,
        web.stderr,
    )
    assert "HARD STOP: phase2:" in web.stdout + web.stderr, (web.stdout, web.stderr)


def test_regime_states_web_off() -> None:
    """T-0168 criterion 4: the header comment and the run.log regime lines say phase 2 runs with --no-web and that
    the web path is covered by t0160_web_smoke.sh; no "re-plan" remains."""
    text = _text()
    header = text.split("\nset -", 1)[0]
    assert NO_WEB.search(header) and "t0160_web_smoke.sh" in header, (
        "T-0168 criterion 4: the header comment says phase 2 runs with --no-web, web path: t0160_web_smoke.sh"
    )
    echoes = [line for line in _joined() if re.match(r'\s*echo "#', line)]
    assert any(NO_WEB.search(line) for line in echoes), (
        f"T-0168 criterion 4: a run.log regime line names --no-web; {echoes}"
    )
    assert any("t0160_web_smoke" in line for line in echoes), (
        f"T-0168 criterion 4: a run.log regime line names t0160_web_smoke.sh; {echoes}"
    )
    assert "re-plan" not in text.lower(), (
        "T-0168 criterion 4: the `re-plan this script` comments are removed"
    )


def test_hard_stops() -> None:
    code = _code()
    assert "web search" in code and "executed" in code, (
        "T-0125 criterion 13.4: the stop on no `web search` note with status executed and results >= 1"
    )
    assert "provider refused" in code, (
        "T-0125 criterion 13.4: the distinct `provider refused` message when every engine refused"
    )
    assert "resolv.conf" in code and re.search(r"\b53\b", code), (
        "T-0125 criterion 13.4: non-loopback connects allowed only to a resolv.conf nameserver on port 53"
    )
    assert "ALLOWED_HOSTS" in code, "T-0125 criterion 13.4: or to an ALLOWED_HOSTS address"
    assert re.search(r"\bss\b[^\n]*-l|listen", code, re.I), (
        "T-0125 criterion 13.4: the non-loopback listener stop"
    )


def test_output_dir_per_sha() -> None:
    code = _code()
    assert "logs/t0125-search-smoke-" in code, (
        "T-0125 criterion 13: logs/t0125-search-smoke-<sha7>/"
    )
    assert re.search(r"rev-parse --short(?:=7)? HEAD|rev-parse HEAD", code), "the sha names the run"


def test_sessions_run_unbuffered() -> None:
    """Amendment 3 (f): the session runs with ``PYTHONUNBUFFERED=1`` (its stdout checks must see the lines)."""
    code = _code()
    exported = re.search(r"^\s*export\s+PYTHONUNBUFFERED=1\b", code, re.M)
    for line in _session_lines():
        assert exported or "PYTHONUNBUFFERED=1" in line, (
            f"T-0125 amendment 3 (f): the session command runs with PYTHONUNBUFFERED=1; {line}"
        )


# --- T2: the prelude, run for real in a scratch repo ------------------------------------------------------------

FILLER = 450  # merges with ~200-char subjects: far more than a 64 KB pipe holds
ENGINE_REFUSAL = "REFUSED: engine (vllm /health)"
T0125_REFUSAL = "T-0125's merge commit is not an ancestor"
#: the refusal T-0168 removes (T-0125 13.1 as first written); must never be printed again
T0160_REFUSAL = "'Merge T-0160 group C' is an ancestor of HEAD"
#: T-0168 criterion 1: the refusal it adds, worded as t0160_web_smoke.sh's guard
T0160_IMPL_REFUSAL = "'Merge T-0160 group C impl' is not an ancestor"


def _blob(data: bytes) -> bytes:
    return b"data %d\n" % len(data) + data + b"\n"


def _stream(script: bytes, top: list[str]) -> bytes:
    """base (the script) and a side commit, ``FILLER`` merges of side into main, then ``top`` merges (last on top)."""
    out: list[bytes] = []
    t = 1_700_000_000
    mark = 0

    def commit(ref: str, msg: str, parents: list[int], files: dict[str, bytes]) -> int:
        nonlocal t, mark
        mark += 1
        t += 1
        out.append(b"commit " + ref.encode() + b"\nmark :%d\n" % mark)
        out.append(b"committer T <t@example.invalid> %d +0000\n" % t)
        out.append(_blob(msg.encode()))
        if parents:
            out.append(b"from :%d\n" % parents[0])
            for p in parents[1:]:
                out.append(b"merge :%d\n" % p)
        for path, data in files.items():
            out.append(b"M 100755 inline " + path.encode() + b"\n" + _blob(data))
        return mark

    head = commit("refs/heads/main", "base", [], {"scripts/locked/t0125_search_smoke.sh": script})
    side = commit("refs/heads/side", "side", [head], {"side.txt": b"side\n"})
    for i in range(FILLER):
        head = commit("refs/heads/main", f"Merge T-9{i:03d} filler " + "x" * 200, [head, side], {})
    for subject in top:
        head = commit("refs/heads/main", subject, [head, side], {})
    return b"".join(out) + b"done\n"


def _scratch_run(
    tmp: Path, top: list[str], *, healthy: bool = False, dry_out: str = "", dry_rc: int = 0
) -> subprocess.CompletedProcess[str]:
    """Run the real script in a scratch repo. Stubs: ``ss`` (nothing listening); ``curl`` (every probe fails, or
    with ``healthy`` the engine/TTS answer 200 and the STT route 400); ``strace`` logs its argv to
    ``<tmp>/strace.calls`` and exits 0 (so phase 1's session "exits before binding"); ``uv`` logs its argv to
    ``<tmp>/uv.calls`` and, for an argv containing ``--dry-run``, prints ``dry_out`` and exits ``dry_rc``; any
    other ``uv`` call prints ``STUB-UV-CALLED`` and exits 97."""
    git = shutil.which("git")
    bash = shutil.which("bash")
    assert git and bash
    repo = tmp / "repo"
    repo.mkdir(parents=True)

    def run(*a: str, **k: Any) -> Any:
        return subprocess.run([git, "-C", str(repo), *a], check=True, capture_output=True, **k)

    run("init", "-q", "-b", "main")
    run("fast-import", "--quiet", "--done", input=_stream(_text().encode(), top))
    run("checkout", "-q", "-f", "main")
    stubs = tmp / "stubs"
    stubs.mkdir()
    (tmp / "dry.out").write_text(dry_out)
    log = "{ printf CALL; printf '\\t%s' \"$@\"; printf '\\n'; } >> " + shlex.quote(str(tmp))
    curl = 'case "$*" in *8080*) printf 400;; *) printf 200;; esac' if healthy else "printf 000"
    uv = (
        f"{log}/uv.calls\n"
        'for a in "$@"; do if [ "$a" = --dry-run ]; then\n'
        f"  cat {shlex.quote(str(tmp / 'dry.out'))}; exit {dry_rc}\n"
        "fi; done\n"
        "echo STUB-UV-CALLED >&2; exit 97"
    )
    for name, body in {
        "strace": f"{log}/strace.calls\nexit 0",
        "ss": "exit 0",  # nothing listening
        "curl": curl,
        "uv": uv,
    }.items():
        p = stubs / name
        p.write_text(f"#!/bin/sh\n{body}\n")
        p.chmod(0o755)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp), "LANG": "C"}
    return subprocess.run(
        [bash, str(repo / "scripts/locked/t0125_search_smoke.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _no_merge_refusal(proc: subprocess.CompletedProcess[str]) -> None:
    for refusal in (T0125_REFUSAL, T0160_IMPL_REFUSAL, T0160_REFUSAL):
        assert refusal not in proc.stderr, f"a false merge refusal ({refusal!r})\n{proc.stderr}"


@pytest.mark.parametrize("trial", range(3))
def test_prelude_passes_the_merge_guard_with_a_long_history(tmp_path: Path, trial: int) -> None:
    """T2 (amendment 3 (f), T-0168 criterion 1): > 64 KB of merge subjects with ``Merge T-0125`` and then
    ``Merge T-0160 group C impl`` on top: the merge guards pass (no pipefail/SIGPIPE false refusal) and the run
    stops at the engine health check (exit 2, that message)."""
    proc = _scratch_run(
        tmp_path,
        ["Merge T-0125 search wiring (fixture)", "Merge T-0160 group C impl web egress (fixture)"],
    )
    _no_merge_refusal(proc)
    assert proc.returncode == 2 and ENGINE_REFUSAL in proc.stderr, (proc.returncode, proc.stderr)
    assert "STUB-UV-CALLED" not in proc.stderr


def test_prelude_refuses_without_t0160_group_c_impl(tmp_path: Path) -> None:
    """T2, T-0168 criterion 1: ``Merge T-0125`` alone (the pre-web history) is refused with the impl message."""
    proc = _scratch_run(tmp_path, ["Merge T-0125 search wiring (fixture)"])
    assert proc.returncode == 2 and T0160_IMPL_REFUSAL in proc.stderr, (
        proc.returncode,
        proc.stderr,
    )
    assert T0125_REFUSAL not in proc.stderr, proc.stderr
    assert ENGINE_REFUSAL not in proc.stderr and "STUB-UV-CALLED" not in proc.stderr, proc.stderr


@pytest.mark.parametrize(
    "subject",
    [
        "Merge T-0160 group C tests web rows (fixture)",  # a tests-only merge
        "Merge T-0160 group C implementation (fixture)",  # `impl\b`: a look-alike word
    ],
)
def test_prelude_refuses_a_tests_only_t0160_merge(tmp_path: Path, subject: str) -> None:
    """T2, adversarial (T-0168 criterion 1): neither a tests-only group C merge nor a look-alike satisfies the
    guard. The refusal must be the impl one (not the removed "is an ancestor" refusal, not the T-0125 one), and
    the run must not reach the health checks."""
    proc = _scratch_run(tmp_path, ["Merge T-0125 search wiring (fixture)", subject])
    assert proc.returncode == 2 and T0160_IMPL_REFUSAL in proc.stderr, (
        proc.returncode,
        proc.stderr,
    )
    assert T0160_REFUSAL not in proc.stderr and T0125_REFUSAL not in proc.stderr, proc.stderr
    assert ENGINE_REFUSAL not in proc.stderr and "STUB-UV-CALLED" not in proc.stderr, proc.stderr


def test_prelude_passes_with_impl_below_t0125(tmp_path: Path) -> None:
    """T2, T-0168 criterion 1: order does not matter; the impl merge older than ``Merge T-0125`` passes both
    guards and the run stops at the engine check."""
    proc = _scratch_run(
        tmp_path,
        ["Merge T-0160 group C impl (fixture)", "Merge T-0125 search wiring (fixture)"],
    )
    _no_merge_refusal(proc)
    assert proc.returncode == 2 and ENGINE_REFUSAL in proc.stderr, (proc.returncode, proc.stderr)
    assert "STUB-UV-CALLED" not in proc.stderr


BOTH_MERGES = [
    "Merge T-0125 search wiring (fixture)",
    "Merge T-0160 group C impl web egress (fixture)",
]
# the shape of the real dry-run tools line (docs/tasks/T-0168.md implementation notes; cli.tools_line)
LOOPBACK_LINE = (
    "tools=live native:bbfc53ccdd80 repair=on loop=on(8) browser=headless egress=loopback "
    "viewport=1280x800 visual_tokens=1120"
)
WEB_LINE = LOOPBACK_LINE.replace("egress=loopback", "egress=web")
DRY_HEAD = "profile=lean thinking=True canvas_length=256 max_denoising_steps=48\nregime: profile=lean mic=none\n"
DRY_TAIL = "astra session: search: ddgs (default)\n"


def _calls(tmp: Path, name: str) -> list[list[str]]:
    path = tmp / f"{name}.calls"
    if not path.exists():
        return []
    return [
        line.split("\t")[1:] for line in path.read_text().splitlines() if line.startswith("CALL")
    ]


def _without_dirs(argv: list[str]) -> list[str]:
    """argv minus ``--dry-run`` and the ``--data-dir``/``--log-dir`` pairs (the pre-flight may use its own dirs)."""
    out: list[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("--data-dir", "--log-dir"):
            skip = True
            continue
        if a != "--dry-run":
            out.append(a)
    return out


def _phase2_uv_argv() -> list[str]:
    """Phase 2's session command as the script spells it: the argv ``uv`` receives (after ``uv``, before ``>``)."""
    phase2 = [line for line in _session_lines() if "--tools live" in line]
    assert len(phase2) == 1, phase2
    tokens = shlex.split(phase2[0].split(">", 1)[0])
    return tokens[tokens.index("uv") + 1 :]


@pytest.mark.parametrize(
    ("case", "dry_out", "dry_rc", "own"),
    [
        # a web tools line beside a loopback one: only the egress=web check can refuse, whatever the order
        ("web", f"{DRY_HEAD}{LOOPBACK_LINE}\n{WEB_LINE}\n{DRY_TAIL}", 0, r"egress=web"),
        ("no-tools-line", f"{DRY_HEAD}{DRY_TAIL}", 0, r"loopback|tools="),
        # a good loopback line, but the dry-run failed: only the exit-status check can refuse
        (
            "nonzero",
            f"{DRY_HEAD}{LOOPBACK_LINE}\n{DRY_TAIL}",
            3,
            r"exit|status|rc\b|non-?zero|fail",
        ),
    ],
    ids=["web", "no-tools-line", "nonzero"],
)
def test_preflight_refuses_web_egress(
    tmp_path: Path, case: str, dry_out: str, dry_rc: int, own: str
) -> None:
    """T-0168 criterion 3a, run for real: the prelude passes every refusal (both merges, healthy stubbed services),
    then the CPU pre-flight (phase 2's command plus ``--dry-run``) refuses with exit 2 and its own ``REFUSED:``
    message, before any phase-1 command starts."""
    proc = _scratch_run(tmp_path, BOTH_MERGES, healthy=True, dry_out=dry_out, dry_rc=dry_rc)
    _no_merge_refusal(proc)
    assert ENGINE_REFUSAL not in proc.stderr, proc.stderr
    dry = [argv for argv in _calls(tmp_path, "uv") if "--dry-run" in argv]
    assert len(dry) == 1, (
        f"T-0168 criterion 3a: one pre-flight dry-run; uv calls {_calls(tmp_path, 'uv')}"
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    refusals = [
        line for line in (proc.stdout + proc.stderr).splitlines() if line.startswith("REFUSED:")
    ]
    assert len(refusals) == 1 and re.search(own, refusals[0], re.I), (
        f"T-0168 criterion 3a ({case}): its own refusal message; {refusals}"
    )
    assert _calls(tmp_path, "strace") == [], "T-0168 criterion 3a: refused before phase 1 started"
    left = sorted(p.name for p in (tmp_path / "repo" / "logs").glob("t0125-search-smoke-*"))
    assert left == [], (
        "T-0168 criterion 3a: a refused pre-flight is a refusal, so it leaves no per-SHA output dir behind "
        f"(the dir would burn the SHA's single run); found {left}"
    )
    assert "STUB-UV-CALLED" not in proc.stderr, proc.stderr


def test_preflight_passes_loopback(tmp_path: Path) -> None:
    """T-0168 criterion 3a, the other direction: the real dry-run shape with ``egress=loopback`` passes the
    pre-flight and the run proceeds to phase 1 (which the strace stub then stops); the dry-run the stub saw is
    exactly phase 2's command plus ``--dry-run`` (so with ``--no-web``)."""
    proc = _scratch_run(
        tmp_path, BOTH_MERGES, healthy=True, dry_out=f"{DRY_HEAD}{LOOPBACK_LINE}\n{DRY_TAIL}"
    )
    out = proc.stdout + proc.stderr
    assert "REFUSED" not in out, out
    dry = [argv for argv in _calls(tmp_path, "uv") if "--dry-run" in argv]
    assert len(dry) == 1, (
        f"T-0168 criterion 3a: one pre-flight dry-run; uv calls {_calls(tmp_path, 'uv')}"
    )
    assert "--no-web" in dry[0], dry[0]
    assert _without_dirs(dry[0]) == _without_dirs(_phase2_uv_argv()), (
        f"T-0168 criterion 3a: the pre-flight is phase 2's exact command plus --dry-run; {dry[0]}"
    )
    phase1 = _calls(tmp_path, "strace")
    assert len(phase1) == 1 and "--tools" in phase1[0], (
        f"the run proceeds to phase 1 after the pre-flight; {phase1}\n{out}"
    )
    assert phase1[0][phase1[0].index("--tools") + 1] == "dry-run", phase1


def test_prelude_refuses_without_the_t0125_merge(tmp_path: Path) -> None:
    """T2, the other direction: no ``Merge T-0125`` at all (a ``Merge T-01250`` look-alike does not count). The
    impl merge is present (T-0168), so the T-0125 refusal is tested on its own whatever the check order."""
    proc = _scratch_run(
        tmp_path, ["Merge T-0160 group C impl (fixture)", "Merge T-01250 look-alike (fixture)"]
    )
    assert proc.returncode == 2 and T0125_REFUSAL in proc.stderr, (proc.returncode, proc.stderr)
    assert T0160_IMPL_REFUSAL not in proc.stderr and ENGINE_REFUSAL not in proc.stderr, proc.stderr


# --- T-0163 criterion 12 (additive, static): the smoke stays runnable with a Brave key present -------------------
#
# The script reports ``brave api: configured|not configured`` from the production loader in a child (never the path,
# never the key), adds ``brave_api.ALLOWED_HOSTS`` to its connect allowlist only under the configured branch, and
# names ``brave_api`` among the refusers in its "provider refused" stop only then. "Under the configured branch" is
# read from the text: a line that pulls in the Brave hosts, or names ``brave_api`` as a refuser, must itself mention
# ``configured`` or sit (by indentation, in the bash or in the embedded Python) inside a block whose header does. Active
# since T-0163's implementation.


def _enclosed_by(lines: list[str], i: int, word: str) -> bool:
    """``word`` (case-insensitive) is on line ``i`` or on a less-indented header line enclosing it."""
    if word in lines[i].lower():
        return True
    indent = len(lines[i]) - len(lines[i].lstrip())
    for j in range(i - 1, -1, -1):
        line = lines[j]
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        if ind < indent:
            if word in line.lower():
                return True
            indent = ind
            if ind == 0:
                break
    return False


def _code_lines() -> list[str]:
    """Raw lines without comments (bash ``#`` lines and Python ``#`` lines alike)."""
    return [x if not x.lstrip().startswith("#") else "" for x in _text().splitlines()]


def test_t0163_reports_brave_api_status_from_the_loader() -> None:
    code = _code()
    assert "local_astra.search.brave_key" in code or "brave_key" in code, (
        "T-0163 criterion 12: the status comes from the production loader (brave_key), in a child"
    )
    reports = [x for x in _joined() if "brave api: " in x]
    assert reports, "T-0163 criterion 12: a `brave api: configured|not configured` report line"
    assert any("LOG" in x or "tee" in x for x in reports) or re.search(
        r'\{[^{}]*brave api: [^{}]*\}\s*>>?\s*"\$LOG"', code, re.S
    ), f"T-0163 criterion 12: the status line goes to the run log; {reports}"
    assert "not configured" in code and "configured" in code
    text = _text()
    for never in (".config/diffusion_gearvis", "brave_api_key", "X-Subscription-Token"):
        assert never not in text, (
            f"T-0163 criterion 12: the script reaches the key only through the loader; it names {never!r}"
        )


def test_t0163_brave_hosts_allowed_only_when_configured() -> None:
    lines = _code_lines()
    pulls = [
        i
        for i, x in enumerate(lines)
        if "brave_api" in x and ("ALLOWED_HOSTS" in x or "import" in x)
    ]
    assert pulls, "T-0163 criterion 12: the connect audit reads brave_api.ALLOWED_HOSTS"
    bad = [lines[i].strip() for i in pulls if not _enclosed_by(lines, i, "configured")]
    assert bad == [], (
        f"T-0163 criterion 12: brave_api.ALLOWED_HOSTS is allowed only under the configured branch; {bad}"
    )
    # the keyless allowlist is still read as before
    assert "from local_astra.search.ddgs_backend import ALLOWED_HOSTS, ENGINES" in _text()


def test_t0163_provider_refused_names_brave_api() -> None:
    lines = _code_lines()
    named = [i for i, x in enumerate(lines) if re.search(r"""["']brave_api["']""", x)]
    assert named, (
        "T-0163 criterion 12: the provider-refused stop names brave_api among the refusers"
    )
    bad = [lines[i].strip() for i in named if not _enclosed_by(lines, i, "configured")]
    assert bad == [], (
        f"T-0163 criterion 12: brave_api is named as a refuser only when configured; {bad}"
    )
    assert "provider refused" in _code(), "the stop itself is unchanged"


def test_t0163_enclosed_by_has_power() -> None:
    """The branch reader on hand-made fragments: an unconditional import is caught; one under ``if configured:`` (or
    a bash ``if [ "$BRAVE" = configured ]``) or on a line that says ``configured`` passes."""
    flat = [
        "allowed = set()",
        "from local_astra.search.brave_api import ALLOWED_HOSTS as B",
        "allowed |= B",
    ]
    assert not _enclosed_by(flat, 1, "configured")
    nested = [
        "if brave == 'configured':",
        "    from local_astra.search.brave_api import ALLOWED_HOSTS as B",
    ]
    assert _enclosed_by(nested, 1, "configured")
    deeper = ["if brave_status == 'configured':", "    for h in hosts:", "        x = 'brave_api'"]
    assert _enclosed_by(deeper, 2, "configured")
    sibling = ["if other:", "    pass", "if x:", "    y = 'brave_api'"]
    assert not _enclosed_by(sibling, 3, "configured")
    one_line = ["refusers = ENGINES + ('brave_api',) if brave == 'configured' else ENGINES"]
    assert _enclosed_by(one_line, 0, "configured")


# --- T-0163 criterion 15 (reviewer B F2): the criterion-12 branches, executed -------------------------------------
#
# The rows above read the text; these run the script's own fragments. The bash status lines run under
# ``set -euo pipefail`` with a stubbed ``uv`` (what the loader child prints, and its exit status); the analysis
# heredoc's ``brave`` branches are taken from its AST and executed with both words. Survivors these kill:
# ``"configured" in brave``, an inverted comparison, ``BRAVE=configured`` hard-coded, a literal ``"configured"`` argv,
# and a dropped ``|| echo "not configured"``.


def _brave_status_fragment() -> str:
    """The top-level (unindented, non-comment) bash lines that set or print ``BRAVE``."""
    lines = [
        x
        for x in _text().splitlines()
        if "BRAVE" in x and x[:1] not in (" ", "\t", "#") and "<<'PY'" not in x
    ]
    assert lines, "T-0163 criterion 12: the script sets BRAVE at top level"
    return "\n".join(lines)


def _run_status(tmp: Path, printed: str, rc: int) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    assert bash
    stubs = tmp / "stubs"
    stubs.mkdir(parents=True)
    uv = stubs / "uv"
    uv.write_text(f"#!/bin/sh\nprintf %s {shlex.quote(printed)}\nexit {rc}\n")
    uv.chmod(0o755)
    prog = (
        "set -euo pipefail\n"
        f"LOG={shlex.quote(str(tmp / 'run.log'))}\n"
        + _brave_status_fragment()
        + '\nprintf "RESULT[%s]\\n" "$BRAVE"\n'
    )
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp), "LANG": "C"}
    return subprocess.run([bash, "-c", prog], env=env, capture_output=True, text=True, timeout=30)


#: (what the loader child prints, its exit status) -> the BRAVE the script must settle on
STATUS_CASES = {
    "configured": ("configured\n", 0, "configured"),
    "not_configured": ("not configured\n", 0, "not configured"),
    "lookalike": ("configuredX\n", 0, "not configured"),
    "capitalised": ("Configured\n", 0, "not configured"),
    "child_failed_silent": ("", 1, "not configured"),
    "child_failed_after_printing": ("configured\n", 1, "not configured"),
    "garbage": ("Traceback (most recent call last)\n", 0, "not configured"),
}


@pytest.mark.parametrize("case", list(STATUS_CASES))
def test_t0163_brave_status_normalised(tmp_path: Path, case: str) -> None:
    printed, rc, want = STATUS_CASES[case]
    proc = _run_status(tmp_path, printed, rc)
    assert proc.returncode == 0, (
        f"[{case}] the status lines must survive a failed child under set -e; {proc.stderr}"
    )
    got = re.findall(r"RESULT\[(.*)\]", proc.stdout)
    assert got == [want], f"[{case}] BRAVE={got}; want {want!r}\n{proc.stdout}"
    assert f"# brave api: {want}" in (tmp_path / "run.log").read_text(), (
        "the status goes to the run log"
    )


def test_t0163_brave_status_lines_pinned() -> None:
    frag = _brave_status_fragment()
    assert '|| echo "not configured"' in frag, f"the bash normaliser for a failed child; {frag}"
    assert re.search(
        r'if \[ "\$BRAVE" != "?configured"? \]; then BRAVE="not configured"; fi', frag
    ), frag


def test_t0163_analyse_gets_brave_as_argv() -> None:
    code = _code()
    assert re.search(r'uv run python - "\$name" "\$dir" "\$BRAVE" <<\'PY\'', code), (
        'T-0163 criterion 12: the analysis gets the status as its third argv, "$BRAVE" (not a literal)'
    )


def _analyse_ast() -> Any:
    import ast

    text = _text()
    start = text.index("<<'PY'")
    body = text[text.index("\n", start) + 1 : text.index("\nPY\n", start)]
    return ast.parse(body)


def test_t0163_analyse_branches_compare_exactly() -> None:
    """Every comparison on ``brave`` in the analysis is ``brave == "configured"`` (two at least: the refusers and the
    hosts), and ``brave`` is ``sys.argv[3]``."""
    import ast

    tree = _analyse_ast()
    compares = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Compare)
        and any(isinstance(x, ast.Name) and x.id == "brave" for x in ast.walk(n))
    ]
    assert len(compares) >= 2, [ast.unparse(c) for c in compares]
    for c in compares:
        assert ast.unparse(c) == "brave == 'configured'", ast.unparse(c)
    sources = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(x, ast.Name) and x.id == "brave" and isinstance(x.ctx, ast.Store)
            for t in n.targets
            for x in ast.walk(t)
        )
    ]
    assert len(sources) == 1 and "sys.argv[3]" in sources[0], sources
    uses = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "brave" and isinstance(n.ctx, ast.Load)
    ]
    assert len(uses) == len(compares), "brave is read only in those comparisons"


@pytest.mark.parametrize("brave", ["configured", "not configured", "", "configuredX"])
def test_t0163_analyse_branches_executed(brave: str) -> None:
    """The analysis' own ``refusers`` and ``hosts`` statements, executed: only the exact word adds ``brave_api`` to
    the refusers and ``api.search.brave.com`` to the allowed hosts."""
    import ast

    tree = _analyse_ast()

    def mentions(node: ast.AST, *names: str) -> bool:
        return any(isinstance(x, ast.Name) and x.id in names for x in ast.walk(node))

    picked = [
        n
        for n in tree.body
        if (isinstance(n, ast.Assign) and mentions(n, "refusers", "hosts"))
        or (isinstance(n, ast.If) and mentions(n.test, "brave"))
    ]
    assert picked, "the refusers/hosts statements"
    ns: dict[str, Any] = {
        "ENGINES": ("duckduckgo", "brave", "mojeek"),
        "ALLOWED_HOSTS": frozenset({"html.duckduckgo.com"}),
        "brave": brave,
    }
    exec(compile(ast.Module(body=picked, type_ignores=[]), "<analyse>", "exec"), ns)
    on = brave == "configured"
    assert ("brave_api" in tuple(ns["refusers"])) is on, (brave, ns["refusers"])
    assert set(ns["refusers"]) - {"brave_api"} == {"duckduckgo", "brave", "mojeek"}
    want_hosts = {"html.duckduckgo.com"} | ({"api.search.brave.com"} if on else set())
    assert set(ns["hosts"]) == want_hosts, (brave, ns["hosts"])
