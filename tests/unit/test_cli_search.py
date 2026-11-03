"""T-0125 (cli side): ``--search {off,ddgs}``, its resolution and source, the refusals, the default-on degrade, the
startup and search lines, and the backend close (docs/tasks/T-0125.md criteria 1-3, 8).

Two rigs, both over the existing CLI fakes (``tests/unit/_tools_switch_support`` and ``test_cli_tools_live``'s
``fake_live``): ``--dry-run`` (prints the plan, constructs nothing), and a real start that builds the graph and
stops before serving (``T.stop_before_serving``), read back through the ``Session`` kwargs. A real start
constructs a default ``DdgsSearch()`` (it imports ddgs and builds clients, connects nowhere): every row unsets the
proxy variables, and the ``search`` extra is installed (``uv sync --all-extras``). Prerequisite failures are made
through the environment: ``sys.modules["ddgs"] = None`` (the extra missing), a proxy variable (T-0111b criterion
6's ``ValueError``), and the backend's construction check raising ``RuntimeError`` (ddgs internals moved).

Expected lines are written out from criterion 2/3. The surface the rows assume where the task is silent (listed in
the test-author's report):

- the refusal for ``--search ddgs`` with tools off names ``--tools live`` and ``--tools dry-run``; with
  ``--no-tool-loop`` it names ``--no-tool-loop``;
- under ``--dry-run`` the search line is printed as at a real start, and is the last line of the tools block (no
  ``--serve-hud``);
- the degrade's session-log note is not checked here: the CLI rigs run ``--no-log`` and a logged run would write
  under the repo's ``logs/``.

Written ahead of T-0125 as pending rows; activated with the implementation.

Regime: CPU over fakes; no browser, no service, no network; no measurement.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli
from tests.unit import _t0172_pins as t0172
from tests.unit import _t0202_pins as t0202  # T-0202 D3 re-pin
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import run_cli
from tests.unit.test_cli_tools_live import LIVE_LINE, Events, _Hud, _serve_rig, argv, fake_live

REPO = Path(__file__).resolve().parents[2]
PROXY_ENV = ("DDGS_PROXY", "PRIMP_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
STARTUP = (
    "astra session: web search is on (keyless: duckduckgo, then brave, then mojeek): each search query the model "
    "makes is sent off this machine and logged in the session log; results are untrusted web text; --search off "
    "turns it off"
)
SEARCH_PREFIX = "astra session: search:"
OFF_PREFIX = "astra session: web search is off: "


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":99")


def _has_helper() -> None:
    assert hasattr(cli, "_search_backend"), "T-0125 criterion 3: cli._search_backend(name) exists"


def dry(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = ""
) -> tuple[int, str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, "--dry-run", *flags, config_body=body))
    got = capsys.readouterr()
    return code, got.out, got.err


def real(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, body: str = ""
) -> tuple[int, str, str, T.Calls, list[Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    made = fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, *flags, config_body=body))
    got = capsys.readouterr()
    return code, got.out, got.err, calls, made


def search_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(SEARCH_PREFIX)]


def not_argparse(err: str) -> None:
    assert "unrecognized arguments" not in err and "invalid choice" not in err, (
        f"T-0125 criterion 2: --search is a known flag; argparse refused it: {err}"
    )


# =====================================================================================================
# Criterion 2: the resolution table
# =====================================================================================================

LIVE = ("--tools", "live", "--browser", "headless")
DRY = ("--tools", "dry-run")
#: id -> (flags, profile body, search line or None, kit web_search at a real start)
TABLE: dict[str, tuple[tuple[str, ...], str, str | None, str | None]] = {
    "off_unset": ((), "", None, None),
    "dry_unset": (DRY, "", None, None),
    "dry_ddgs_flag": ((*DRY, "--search", "ddgs"), "", "astra session: search: ddgs (flag)", "ddgs"),
    "dry_ddgs_profile": (DRY, 'search = "ddgs"\n', "astra session: search: ddgs (profile)", "ddgs"),
    "live_unset": (LIVE, "", "astra session: search: ddgs (default)", "ddgs"),
    "live_off_flag": ((*LIVE, "--search", "off"), "", "astra session: search: off (flag)", None),
    "live_off_profile": (LIVE, 'search = "off"\n', "astra session: search: off (profile)", None),
    "live_ddgs_flag": (
        (*LIVE, "--search", "ddgs"),
        "",
        "astra session: search: ddgs (flag)",
        "ddgs",
    ),
    "live_ddgs_profile": (
        LIVE,
        'search = "ddgs"\n',
        "astra session: search: ddgs (profile)",
        "ddgs",
    ),
}


@pytest.mark.parametrize("case", list(TABLE))
@pytest.mark.parametrize("start", ["dry_run_flag", "real"])
def test_resolution_table(tmp_path, capsys, monkeypatch, case: str, start: str) -> None:
    _has_helper()
    flags, body, line, web = TABLE[case]
    if start == "dry_run_flag":
        code, out, err = dry(tmp_path, capsys, monkeypatch, *flags, body=body)
    else:
        code, out, err, calls, _ = real(tmp_path, capsys, monkeypatch, *flags, body=body)
    not_argparse(err)
    assert code == 0, f"{case}: exit {code}\n{out}{err}"
    assert search_lines(out) == ([line] if line else []), (
        f"T-0125 criterion 2/3 [{case}, {start}]: search line(s) {search_lines(out)}"
    )
    if start == "real":
        kit = calls.kit()
        if kit is None:
            assert web is None
            return
        assert getattr(kit, "web_search", "missing") == web, f"[{case}] kit.web_search"
        if web is None:
            assert kit.search is None, f"[{case}] search off: kit.search is None"
        else:
            assert type(kit.search).__name__ == "DdgsSearch", f"[{case}] kit.search {kit.search!r}"


@pytest.mark.parametrize("case", ["dry_run_opt_in", "live_default"])
def test_kit_carries_web_search_ddgs(tmp_path, capsys, monkeypatch, case: str) -> None:
    flags = (*DRY, "--search", "ddgs") if case == "dry_run_opt_in" else LIVE
    code, _out, err, calls, _ = real(tmp_path, capsys, monkeypatch, *flags)
    not_argparse(err)
    assert code == 0, err
    kit = calls.kit()
    assert getattr(kit, "web_search", None) == "ddgs", "T-0125 criterion 3"
    from local_astra.search.ddgs_backend import DdgsSearch

    assert isinstance(kit.search, DdgsSearch)


def test_flag_beats_profile(tmp_path, capsys, monkeypatch) -> None:
    _code, out, err = dry(
        tmp_path, capsys, monkeypatch, *LIVE, "--search", "ddgs", body='search = "off"\n'
    )
    not_argparse(err)
    assert search_lines(out) == ["astra session: search: ddgs (flag)"], out
    _code, out, err = dry(
        tmp_path / "b", capsys, monkeypatch, *LIVE, "--search", "off", body='search = "ddgs"\n'
    )
    assert search_lines(out) == ["astra session: search: off (flag)"], out


def test_unknown_search_value_is_argparse_error(tmp_path, capsys, monkeypatch) -> None:
    _has_helper()
    for value in ("on", "duckduckgo", "brave-api"):
        code, _out, err = dry(tmp_path, capsys, monkeypatch, *LIVE, "--search", value)
        assert code == 2 and "--search" in err, (value, err)


# =====================================================================================================
# Criterion 2: explicit refusals and the default-on degrade
# =====================================================================================================


def _break(monkeypatch: pytest.MonkeyPatch, how: str) -> str:
    """Make a prerequisite fail; returns a string the reason must contain."""
    if how == "extra_missing":
        for name in [m for m in list(sys.modules) if m == "ddgs" or m.startswith("ddgs.")]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, "ddgs", None)
        return "uv sync --all-extras"  # T-0222 re-pin: R-1 (BASE: "uv sync --extra search")
    if how == "proxy_env":
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
        return "HTTPS_PROXY"
    from local_astra.search import ddgs_backend

    def moved() -> None:
        raise RuntimeError("ZQX ddgs internals moved since 9.16.1")

    monkeypatch.setattr(ddgs_backend, "_check_default_client", moved)
    return "ZQX ddgs internals moved"


@pytest.mark.parametrize(
    "how", ["tools_off", "loop_off", "extra_missing", "proxy_env", "internals_moved"]
)
def test_explicit_refusals(tmp_path, capsys, monkeypatch, how: str) -> None:
    _has_helper()
    if how == "tools_off":
        flags: tuple[str, ...] = ("--search", "ddgs")
        names = ("--tools live", "--tools dry-run")
    elif how == "loop_off":
        flags = (*DRY, "--no-tool-loop", "--search", "ddgs")
        names = ("--no-tool-loop",)
    else:
        flags = (*LIVE, "--search", "ddgs")
        names = (_break(monkeypatch, how),)
    code, out, err, calls, made = real(tmp_path, capsys, monkeypatch, *flags)
    not_argparse(err)
    assert code == 2, f"T-0125 criterion 2 [{how}]: exit 2; got {code}\n{out}{err}"
    for name in names:
        assert name in err, f"T-0125 criterion 2 [{how}]: the refusal names {name!r}; {err}"
    assert calls.session_kwargs == [] and made == [], (
        f"[{how}]: nothing is built before the refusal"
    )


def test_prerequisites_never_checked_under_dry_run(tmp_path, capsys, monkeypatch) -> None:
    _has_helper()
    _break(monkeypatch, "proxy_env")
    code, out, err = dry(tmp_path, capsys, monkeypatch, *LIVE, "--search", "ddgs")
    assert code == 0, err
    assert search_lines(out) == ["astra session: search: ddgs (flag)"], out


@pytest.mark.parametrize("how", ["extra_missing", "proxy_env", "internals_moved"])
def test_default_on_degrades_cleanly(tmp_path, capsys, monkeypatch, how: str) -> None:
    """Adversarial: a default-on live start whose prerequisite fails starts with search off -- never half-wired
    (``kit.search`` left set), never failing open, never exiting."""
    _has_helper()
    why = _break(monkeypatch, how)
    code, out, err, calls, _ = real(tmp_path, capsys, monkeypatch, *LIVE)
    assert code == 0, f"T-0125 criterion 2 [{how}]: the session still starts; exit {code}\n{err}"
    kit = calls.kit()
    assert (
        kit is not None and kit.search is None and getattr(kit, "web_search", "missing") is None
    ), f"T-0125 criterion 2 [{how}]: kit.search and kit.web_search are both None; {kit}"
    off = [line for line in err.splitlines() if line.startswith(OFF_PREFIX)]
    assert len(off) == 1, f"T-0125 criterion 2 [{how}]: one stderr line; {err}"
    assert off[0].endswith("; --search off silences this") and (
        how == "extra_missing" or why in off[0]
    ), off
    if how == "extra_missing":
        assert "uv sync --all-extras" in off[0], (
            off
        )  # T-0222 re-pin: R-1 (BASE: "uv sync --extra search")
    assert STARTUP not in err, "no startup line when search is off"
    lines = search_lines(out)
    assert len(lines) == 1 and lines[0].startswith("astra session: search: off (default: "), lines
    outs = out.splitlines()
    regime = [i for i, line in enumerate(outs) if line.startswith("astra session: regime:")]
    assert regime and outs.index(lines[0]) < regime[0], (
        "the search line comes before the regime line"
    )


# =====================================================================================================
# Criterion 3: the lines
# =====================================================================================================


def serve(
    tmp_path: Path, capsys: Any, monkeypatch: Any, *flags: str, **live_kw: Any
) -> tuple[int, str, str, Events]:
    """A served start (``test_cli_tools_live._serve_rig``'s doubles, any flags): the HUD double SIGINTs once
    started, so ``_serve`` runs its start-up and then the close."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    events = Events()
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    T.point_cli_at(
        monkeypatch, payloads=tmp_path / "nowhere" / "native_T4.json"
    )  # repair off: no warm-up
    hud = live_kw.pop("hud", _Hud)
    log_dir = live_kw.pop("log_dir", None)
    hud.events = events
    monkeypatch.setattr(cli, "HudServer", hud)
    fake_live(monkeypatch, events, **live_kw)
    args = argv(tmp_path, *flags)
    if log_dir is not None:  # the ``--log-dir`` seam (cli.log_dir_of): a logged run under tmp_path
        args.remove("--no-log")
        args += ["--log-dir", str(log_dir)]
    code = run_cli(args)
    got = capsys.readouterr()
    return code, got.out, got.err, events


class _WatchingHud(_Hud):
    """Records, when the HUD server starts, whether the startup line was already on stderr."""

    seen_at_start: bool | None = None

    async def start(self) -> None:
        getvalue = getattr(sys.stderr, "getvalue", None)
        assert getvalue is not None, "premise: capsys captures stderr"
        type(self).seen_at_start = STARTUP in getvalue()
        await super().start()


def test_startup_line_after_the_server_started(tmp_path, capsys, monkeypatch) -> None:
    """Amendment 3 (g): the startup line prints after the HUD server and the start pages are both up."""
    _WatchingHud.seen_at_start = None
    code, out, err, events = serve(tmp_path, capsys, monkeypatch, *LIVE, hud=_WatchingHud)
    assert code == 0, f"{code}\n{out}{err}"
    assert events.index_of("pages.start") < events.index_of("hud.start"), list(events)
    assert _WatchingHud.seen_at_start is False, (
        "T-0125 amendment 3 (g): the startup line was printed before graph.server.start()"
    )
    assert err.splitlines().count(STARTUP) == 1, err


def test_startup_label_is_read_from_engines(tmp_path, capsys, monkeypatch) -> None:
    """T3: the engine list in the startup line is ``ENGINES``, never retyped."""
    from local_astra.search import ddgs_backend

    monkeypatch.setattr(ddgs_backend, "ENGINES", ("duckduckgo", "mojeek"))
    code, out, err, _ = serve(tmp_path, capsys, monkeypatch, *LIVE)
    assert code == 0, f"{code}\n{out}{err}"
    want = STARTUP.replace("duckduckgo, then brave, then mojeek", "duckduckgo, then mojeek")
    assert err.splitlines().count(want) == 1, (
        f"T-0125 criterion 3: the label follows ENGINES; {err}"
    )


@pytest.mark.parametrize("start", ["stopped", "served"])
@pytest.mark.parametrize("case", ["dry_run_opt_in", "live_default"])
def test_one_backend_construction_per_real_start(
    tmp_path, capsys, monkeypatch, start, case
) -> None:
    """T3 / amendment 3 (c): ``cli._search_backend`` is called exactly once per real start."""
    _has_helper()
    original = cli._search_backend
    calls: list[str] = []

    def counting(name: str) -> Any:
        calls.append(name)
        return original(name)

    monkeypatch.setattr(cli, "_search_backend", counting)
    flags = (*DRY, "--search", "ddgs") if case == "dry_run_opt_in" else LIVE
    if start == "stopped":
        code, out, err, _, _ = real(tmp_path, capsys, monkeypatch, *flags)
    else:
        code, out, err, _ = serve(tmp_path, capsys, monkeypatch, *flags)
    assert code == 0, f"{code}\n{out}{err}"
    assert calls == ["ddgs"], (
        f"T-0125 amendment 3 (c): one construction per real start; calls {calls}"
    )


def _spy_backend(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Wrap ``cli._search_backend``: the real backend is built (and its label kept), and every ``aclose`` of it is
    recorded."""
    _has_helper()
    original = cli._search_backend
    closes: list[str] = []

    def spying(name: str) -> Any:
        backend, label = original(name)
        real_close = backend.aclose

        async def aclose() -> None:
            closes.append(name)
            await real_close()

        backend.aclose = aclose
        return backend, label

    monkeypatch.setattr(cli, "_search_backend", spying)
    return closes


class _BindFailsHud(_Hud):
    async def start(self) -> None:
        type(self).events.append(("hud.start.failed",))
        raise OSError(98, "Address already in use")


def test_bind_failure_closes_the_backend_and_prints_one_line(tmp_path, capsys, monkeypatch) -> None:
    """Amendment 4 (c)/(e): the HUD server cannot bind: exit 1, exactly one stderr line (the bind error), no startup
    line, and the backend built at the real start is closed."""
    from tests.unit.test_cli_tools_live import failure_lines

    closes = _spy_backend(monkeypatch)
    code, out, err, events = serve(tmp_path, capsys, monkeypatch, *LIVE, hud=_BindFailsHud)
    assert ("hud.start.failed",) in events, f"premise: the bind failed; {list(events)}"
    assert code == 1, f"T-0125 amendment 4: exit 1 on a bind failure; got {code}\n{out}{err}"
    lines = failure_lines(err)
    assert len(lines) == 1 and ("Address already in use" in lines[0] or "98" in lines[0]), (
        f"T-0125 amendment 4: exactly one stderr line, the bind error; {lines}"
    )
    assert STARTUP not in err
    assert closes == ["ddgs"], (
        f"T-0125 amendment 4 (c): the backend is closed on a bind failure; {closes}"
    )


def test_static_server_failure_closes_the_backend(tmp_path, capsys, monkeypatch) -> None:
    """T-0197 criterion 5 (named spec change, superseding T-0125 amendment 4 (c) / reviewer A's "static server
    failure exits with its code"): at a real start with ``--serve-hud`` in live, the HUD static server cannot bind
    (its construction raises ``OSError``): the session is **not** ended, ``_serve`` is reached, one stderr line names
    ``127.0.0.1:<hud_port>``, and the backend the real-start check built is **not** closed before the session runs.
    The row keeps its name so the supersession is visible in the history."""
    from local_astra import hud_static

    closes = _spy_backend(monkeypatch)
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    stubbed = cli._serve
    closes_at_serve: list[list[str]] = []

    async def serve(graph: Any, stop: Any = None) -> int:
        closes_at_serve.append(list(closes))
        return await stubbed(graph, stop)

    monkeypatch.setattr(cli, "_serve", serve)
    tried: list[int] = []

    def busy(root: Any, port: int, *a: Any, **k: Any) -> Any:
        tried.append(int(port))
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(hud_static, "HudStaticServer", busy)
    port = 1420  # the profile default: write_config sets no services table; nothing binds here (``busy`` raises)
    code = run_cli(argv(tmp_path, *LIVE, "--serve-hud"))
    out, err = capsys.readouterr()
    assert tried == [port], f"premise: the HUD static server was tried once, on {port}; {tried}"
    assert code == 0, (
        f"T-0197 criterion 5: a busy HUD port never ends the session; exit {code}\n{out}{err}"
    )
    assert closes_at_serve == [[]], (
        f"T-0197 criterion 5: the session is served and the search backend is not closed first; {closes_at_serve}"
    )
    named = [line for line in err.splitlines() if f"127.0.0.1:{port}" in line]
    assert len(named) == 1 and named[0].startswith("astra session: "), (
        f"T-0197 criterion 5: one `astra session: ` stderr line names 127.0.0.1:{port}; {err}"
    )
    assert "Traceback" not in err


class _BuildBoom(RuntimeError):
    pass


def test_build_failure_after_the_check_closes_the_backend(tmp_path, capsys, monkeypatch) -> None:
    """Amendment 4 (c): the real-start check built the backend, then ``build_session`` fails: the backend is
    closed."""
    closes = _spy_backend(monkeypatch)

    def boom(*a: Any, **k: Any) -> Any:
        raise _BuildBoom("ZQX build failed")

    monkeypatch.setattr(cli, "_build_graph", boom)
    try:
        code, _out, _err, _, _ = real(tmp_path, capsys, monkeypatch, *LIVE)
    except _BuildBoom:
        code = None
    assert code != 0, "premise: the start failed"
    assert closes == ["ddgs"], (
        f"T-0125 amendment 4 (c): the backend is closed when the build fails; {closes}"
    )


def test_degraded_start_writes_the_log_note(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 2: a default-on start that degrades writes one session-log ``note {what: "web search off", why}``
    (through the ``--log-dir`` seam, under tmp_path)."""
    from tests.unit._session_log_rig import read_records

    _has_helper()
    _break(monkeypatch, "proxy_env")
    logs = tmp_path / "sessions"
    code, out, err, _ = serve(tmp_path, capsys, monkeypatch, *LIVE, log_dir=logs)
    assert code == 0, f"{code}\n{out}{err}"
    files = sorted(logs.rglob("*.jsonl"))
    assert files, f"premise: a session log was written under {logs}; {out}"
    notes = [
        r["data"]
        for f in files
        for r in read_records(f)
        if r.get("kind") == "note" and r.get("data", {}).get("what") == "web search off"
    ]
    assert len(notes) == 1, f"T-0125 criterion 2: one `web search off` note; {notes}"
    assert "HTTPS_PROXY" in str(notes[0].get("why")), notes[0]


@pytest.mark.parametrize("case", ["dry_run", "live_default"])
def test_startup_line_once(tmp_path, capsys, monkeypatch, case: str) -> None:
    """Criterion 3 as amended (amendment 2): exactly one startup line, printed only once ``_serve`` has started;
    none under ``--dry-run``, none when search is off, none for a start stopped before serving, none when the
    start-up fails (that failure stays the only stderr line, as ``test_cli_tools_live`` pins)."""
    flags = (*DRY, "--search", "ddgs") if case == "dry_run" else LIVE
    code, out, err, events = serve(tmp_path, capsys, monkeypatch, *flags)
    not_argparse(err)
    assert code == 0, f"{code}\n{out}{err}\n{list(events)}"
    assert err.splitlines().count(STARTUP) == 1, (
        f"T-0125 criterion 3: exactly one startup line; {err}"
    )
    code, _out, err, _ = serve(tmp_path / "off", capsys, monkeypatch, *LIVE, "--search", "off")
    assert code == 0 and STARTUP not in err, err
    _code, _out, err = dry(tmp_path / "dry", capsys, monkeypatch, *flags)
    assert STARTUP not in err, f"T-0125 criterion 3: no startup line under --dry-run; {err}"
    _code, _out, err, _, _ = real(tmp_path / "stopped", capsys, monkeypatch, *flags)
    assert STARTUP not in err, f"T-0125 amendment 2: none before _serve has started; {err}"


class ZqxStartBoom(RuntimeError):
    pass


def test_no_startup_line_when_start_up_fails(tmp_path, capsys, monkeypatch) -> None:
    """Amendment 2: a live start-up that fails (the start page cannot open) exits 1 and prints no startup line."""
    _has_helper()
    code, out, err, events = serve(
        tmp_path, capsys, monkeypatch, *LIVE, open_error=ZqxStartBoom("no page")
    )
    assert code == 1 and "ZqxStartBoom" in err, (code, out, err, list(events))
    assert STARTUP not in err, f"T-0125 amendment 2: no startup line when start-up fails; {err}"


# T-0160 ruling 2: web on by default in live (LIVE_LINE says egress=web)
def test_search_line_placement(tmp_path, capsys, monkeypatch) -> None:
    _has_helper()
    # --dry-run, live: the search line follows every tools-block line (last, without --serve-hud)
    _code, out, _err = dry(tmp_path, capsys, monkeypatch, *LIVE)
    outs = out.splitlines()
    # T-0172 R6 pin move (B9): the reask line is now the last tools-on line, right after the search line
    assert outs[-2:] == ["astra session: search: ddgs (default)", t0172.REASK_LINE], outs[-12:]
    # T-0202 D3 re-pin: the default live line read through admit_line (one wait=on(default) token required when the
    # default is on; the wait set's sha12 read back as v1's)
    tools = [
        t0202.admit_line(line, "search placement") for line in outs if line.startswith("tools=")
    ]
    assert len(tools) == 1 and LIVE_LINE.match(tools[0]), f"tools_line unchanged: {tools}"
    # a real live start: after the tools block, then the reask line (T-0172 R6), then the regime line
    _code, out, _err, _, _ = real(tmp_path / "r", capsys, monkeypatch, *LIVE)
    outs = out.splitlines()
    regime = [i for i, line in enumerate(outs) if line.startswith("astra session: regime:")]
    assert regime and outs[regime[0] - 2 : regime[0]] == [
        "astra session: search: ddgs (default)",
        t0172.REASK_LINE,
    ], outs
    # no line in off, none in dry-run with search off; their tools lines unchanged (T-0183 A3a re-scope: the off
    # line is BASE's plus the default budget's suffix)
    _code, out, _err = dry(tmp_path / "o", capsys, monkeypatch)
    assert search_lines(out) == [] and [x for x in out.splitlines() if x.startswith("tools=")] == [
        "tools=off visual_tokens=1120"
    ]
    _code, out, _err = dry(tmp_path / "d", capsys, monkeypatch, *DRY)
    assert search_lines(out) == []
    assert [x for x in out.splitlines() if x.startswith("tools=")] == [
        f"tools=dry_run native:{T.TOOLS_SHA12} repair=on loop=on(8)"
    ] or all(
        x.startswith(f"tools=dry_run native:{T.TOOLS_SHA12} ")
        for x in out.splitlines()
        if x.startswith("tools=")
    )


# =====================================================================================================
# Criteria 1-2: what search-off never loads (fresh interpreter)
# =====================================================================================================

_SCRIPT = r"""
import json, sys, tempfile
from pathlib import Path
import pytest
from local_astra import cli
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import run_cli
from tests.unit.test_cli_tools_live import Events, argv, fake_live

flags = json.loads(sys.argv[1])
dry = sys.argv[2] == "dry"
tmp = Path(tempfile.mkdtemp())
with pytest.MonkeyPatch.context() as mp:
    for name in PROXY_NAMES:
        mp.delenv(name, raising=False)
    mp.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(mp, calls)
    T.stop_before_serving(mp, calls)
    T.patch_detok_load(mp, calls, tmp)
    fake_live(mp, Events())
    code = run_cli(argv(tmp, *(["--dry-run"] if dry else []), *flags))
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("ddgs", "primp") or m == "local_astra.search.ddgs_backend")
print("RESULT " + json.dumps({"code": code, "loaded": loaded, "helper": hasattr(cli, "_search_backend")}))
""".replace("PROXY_NAMES", repr(PROXY_ENV))


def _fresh(
    flags: tuple[str, ...], dry_run: bool, *, home: Path | None = None, script: str = _SCRIPT
) -> dict[str, Any]:
    # T-0163 criterion 10 (amended): a fresh child never sees the real HOME, so it never reaches the real Brave
    # Search API key file (the default key path follows HOME)
    if home is None:
        home = Path(tempfile.mkdtemp(prefix="t0163_home_"))
    env = {**os.environ, "HOME": str(home)}
    proc = subprocess.run(
        [sys.executable, "-c", script, json.dumps(list(flags)), "dry" if dry_run else "real"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    found = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert found, (
        f"the fresh interpreter printed no result\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    return dict(json.loads(found[-1][len("RESULT ") :]))


@pytest.mark.parametrize(
    ("case", "flags", "dry_run"),
    [
        ("off", (), False),
        ("dry_run_default", DRY, False),
        ("live_search_off", (*LIVE, "--search", "off"), False),
        ("dry_run_flag_live", LIVE, True),  # test_dry_run_flag_builds_no_backend
    ],
)
def test_off_never_imports_ddgs(case: str, flags: tuple[str, ...], dry_run: bool) -> None:
    got = _fresh(flags, dry_run)
    assert got["helper"], (
        "T-0125 criterion 3: cli._search_backend exists (the absence below is not vacuous)"
    )
    assert got["code"] == 0, got
    assert got["loaded"] == [], (
        f"T-0125 criterion 2 [{case}]: search off loads no ddgs; {got['loaded']}"
    )


def test_off_never_imports_ddgs_has_power() -> None:
    """The same harness sees the import when search is on (live default, a real start)."""
    got = _fresh(LIVE, False)
    assert got["code"] == 0, got
    assert "local_astra.search.ddgs_backend" in got["loaded"] and "ddgs" in got["loaded"], got


#: T-0163 criterion 10: ``_SCRIPT`` with an ``open`` audit hook on the Brave key path and the backend type reported
_HOME_SCRIPT = (
    "import os, sys\n"
    "opened = []\n"
    "def _audit(event, args):\n"
    "    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):\n"
    "        name = os.fsdecode(args[0])\n"
    "        if name.endswith('brave_api_key'):\n"
    "            opened.append(os.path.abspath(name))\n"
    "sys.addaudithook(_audit)\n" + _SCRIPT + "kit = calls.kit()\n"
    'print("RESULT " + json.dumps({"code": code, "opened": opened, '
    '"search": type(kit.search).__name__ if kit is not None else None}))\n'
)


def test_fresh_child_reads_the_key_only_under_its_tmp_home(tmp_path: Path) -> None:
    """T-0163 criterion 10 (power row for ``_fresh``'s tmp ``HOME``): a child started as ``_fresh`` starts it, with a
    canary key at the default path inside its tmp ``HOME``, opens the key there and nowhere else (so it cannot pick
    up a key outside that ``HOME``) and builds the Brave fallback from it; with no key in the tmp ``HOME`` it stays
    keyless, again opening nothing outside it."""
    home = tmp_path / "home"
    key = home / ".config" / "diffusion_gearvis" / "brave_api_key"
    key.parent.mkdir(parents=True)
    key.write_text("BSAzq" + "GHJKMNPQRSTVWXYZghjkmnpqrstvwxy" + "\n")
    key.chmod(0o600)
    got = _fresh(LIVE, False, home=home, script=_HOME_SCRIPT)
    assert got["code"] == 0, got
    outside = [p for p in got["opened"] if not Path(p).is_relative_to(home)]
    assert outside == [], f"the child opened a key path outside its tmp HOME: {outside}"
    assert got["opened"] == [str(key)], (
        f"the child reads the key under its tmp HOME, once; {got['opened']}"
    )
    assert got["search"] == "FallbackSearch", got
    empty = tmp_path / "empty_home"
    empty.mkdir()
    got = _fresh(LIVE, False, home=empty, script=_HOME_SCRIPT)
    assert got["code"] == 0, got
    assert all(Path(p).is_relative_to(empty) for p in got["opened"]), got["opened"]
    assert got["search"] == "DdgsSearch", got


# =====================================================================================================
# Criterion 3/8: the close; criterion 1: static
# =====================================================================================================


def test_backend_closed_after_session_even_if_its_close_raises(
    tmp_path, capsys, monkeypatch
) -> None:
    _has_helper()
    events = Events()
    original = cli._search_backend

    class Spy:
        async def search(self, query: str) -> Any:
            raise AssertionError("no turn runs in this row")

        async def aclose(self) -> None:
            events.append(("search.close",))
            raise RuntimeError("close boom")

    def spy_backend(name: str) -> Any:
        _backend, label = original(name)
        return Spy(), label

    monkeypatch.setattr(cli, "_search_backend", spy_backend)
    code = run_cli(_serve_rig(tmp_path, monkeypatch, events))
    out, err = capsys.readouterr()
    assert _Hud.events is events
    assert code == 0, f"{code}\n{out}{err}\n{list(events)}"
    closes = [e for e in events if e == ("search.close",)]
    assert len(closes) == 1, (
        f"T-0125 criterion 3: run_session closes the backend once; {list(events)}"
    )
    order = [
        events.index_of(n) for n in ("hud.start", "browser.close", "search.close", "pages.stop")
    ]
    assert order == sorted(order), (
        f"T-0125 criteria 3/8: the backend closes after the session, and a raising close does not stop the rest; "
        f"{list(events)}"
    )
    # amendment 2026-10-27 (e): the raising close is one stderr line (the session log is closed by then)
    boom = [line for line in err.splitlines() if "close boom" in line]
    assert len(boom) == 1 and boom[0].startswith("astra session: "), (
        f"T-0125 criterion 8: one stderr line names the close failure; {err}"
    )


def test_no_config_file_read_for_search() -> None:
    _has_helper()
    for rel in ("local_astra/cli.py", "local_astra/config.py", "local_astra/conductor/session.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert ".config/diffusion_gearvis" not in text, (
            f"T-0125 criterion 1: {rel} names a search config file"
        )


def test_ddgs_not_imported_at_cli_module_level() -> None:
    """Reviewer mutation: ``DdgsSearch`` imported at the top of ``cli.py``."""
    _has_helper()
    tree = ast.parse((REPO / "local_astra/cli.py").read_text(encoding="utf-8"))
    top = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom | ast.Import)
        and any("ddgs" in (getattr(n, "module", "") or "") or "ddgs" in a.name for a in n.names)
    ]
    assert top == [], "T-0125 criterion 3: the backend is imported only inside _search_backend"
