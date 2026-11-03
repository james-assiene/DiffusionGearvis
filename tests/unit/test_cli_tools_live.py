"""T-0111g B2, B3, B5 (cli side): ``astra session --tools live``, its startup refusals, the ``--dry-run`` line and the
kit ``cli`` builds (docs/tasks/T-0111g.md).

**The seams these rows assume** (listed in the test-author's report):

- ``--tools live`` (argv spelling ``live``) and ``--browser {headed,headless}`` (``args.browser``, ``None`` when
  absent: the profile decides).
- ``cli`` constructs ``BrowserToolExecutor`` and ``pages.server.StartPageServer`` (the composition root). The rows
  replace both wherever ``cli`` reaches them (the defining module and a ``cli`` attribute of the same name, the
  ``CONCRETE`` pattern of ``test_cli_session.py``).
- ``cli.live_prerequisites(browser: str) -> str | None``: the availability check of B2 (``None`` = available, else
  the refusal message). Rows that need a *passing* check (a box without Chromium must still run them) replace it
  with ``lambda browser: None``; the refusal rows never replace it and drive it through the environment only:
  ``playwright`` made unimportable (``sys.modules[name] = None`` for every ``playwright*`` module), the pinned
  Chromium looked up where Playwright looks (``$PLAYWRIGHT_BROWSERS_PATH``, pointed at an empty directory), and
  ``DISPLAY``/``WAYLAND_DISPLAY`` unset.
- Startup (B5): the start-page server is started in ``build_session`` (the executor needs its origin for
  ``allowed_origins``); the start page is opened (``open(url)`` with ``url`` under that origin) on the loop before
  the HUD endpoint's ``start()``; at shutdown the session closes the browser and then ``cli`` stops the server.

Regime: CPU over fakes; no browser, no service, nothing served on a real port; no measurement.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import re
import signal
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, preflight
from tests.unit import _t0202_pins as t0202
from tests.unit import _tools_switch_support as T
from tests.unit._live_browser_support import FakeLiveBrowser
from tests.unit.test_cli_session import explode_clients, run_cli

ORIGIN = "http://127.0.0.1:45678"
#: T-0222 re-pin: R-1 (``uv sync --extra X`` uninstalls the other extras; BASE: ``uv sync --extra browser && ...``).
P1_FIX = "uv sync --all-extras && uv run playwright install chromium"
#: T-0132 re-scope R3, then T-0133 re-scope R4 (human ruling 2026-10-22: live defaults to 1120 with ReGround off):
#: the ``$``-anchored line takes the optional resolved T-0131 suffixes; `` visual_tokens=N`` is on every default live
#: line, and `` reground=on`` appears only when ReGround is on (``--reground`` or ``reground = true``). Both are
#: optional and non-capturing, so the six groups the rows read are unchanged. Green at BASE and after.
#: T-0160 C-1/C-2 pin move (T-0160 ruling 2: web on by default in live): a default live line says ``egress=web``;
#: ``LOOPBACK_LIVE_LINE`` is BASE's regex, the line of ``--no-web`` (``test_cli_tools_web.py``).
LIVE_LINE = re.compile(
    r"^tools=live native:([0-9a-f]{12}) repair=(on|off\(.+\)) loop=on\((\d+)\) "
    r"browser=(headed|headless) egress=web viewport=(\d+)x(\d+)"
    r"(?: visual_tokens=(?:280|560|1120))?(?: reground=on)?$"
)
LOOPBACK_LIVE_LINE = re.compile(LIVE_LINE.pattern.replace(" egress=web ", " egress=loopback "))


# --------------------------------------------------------------------------------------------- doubles


def _patch_both(monkeypatch: pytest.MonkeyPatch, module: str, name: str, value: Any) -> None:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        mod = None
    if mod is not None:
        monkeypatch.setattr(mod, name, value, raising=False)
    monkeypatch.setattr(cli, name, value, raising=False)


def bomb_browser(monkeypatch: pytest.MonkeyPatch, why: str) -> None:
    """``PlaywrightDriver``, ``BrowserToolExecutor``, the start-page server and Playwright itself explode."""

    def boom(name: str) -> type:
        def init(self: Any, *a: Any, **k: Any) -> None:
            raise AssertionError(f"T-0111g {why}: {name} was constructed")

        return type(name, (), {"__init__": init})

    _patch_both(
        monkeypatch, "local_astra.computer.runtime", "PlaywrightDriver", boom("PlaywrightDriver")
    )
    _patch_both(
        monkeypatch,
        "local_astra.computer.executor",
        "BrowserToolExecutor",
        boom("BrowserToolExecutor"),
    )
    _patch_both(monkeypatch, "local_astra.pages.server", "StartPageServer", boom("StartPageServer"))
    for mod, name in (
        ("playwright.async_api", "async_playwright"),
        ("playwright.sync_api", "sync_playwright"),
    ):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue

        def launched(*a: Any, _n: str = name, **k: Any) -> Any:
            raise AssertionError(f"T-0111g {why}: playwright {_n}() was started")

        monkeypatch.setattr(m, name, launched)


def bomb_processes(monkeypatch: pytest.MonkeyPatch, why: str) -> None:
    import subprocess

    def spawned(*a: Any, **k: Any) -> Any:
        raise AssertionError(f"T-0111g {why}: a process was started: {a[:1]!r}")

    monkeypatch.setattr(subprocess, "Popen", spawned)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawned)
    monkeypatch.setattr(os, "posix_spawn", spawned, raising=False)
    monkeypatch.setattr(os, "fork", spawned, raising=False)


def bomb_binds(monkeypatch: pytest.MonkeyPatch, why: str) -> None:
    import http.server

    def bound(*a: Any, **k: Any) -> Any:
        raise AssertionError(f"T-0111g {why}: something was bound or served")

    monkeypatch.setattr(socket.socket, "bind", bound)
    monkeypatch.setattr(http.server.ThreadingHTTPServer, "__init__", bound)


def bomb_session(monkeypatch: pytest.MonkeyPatch, calls: T.Calls, tmp_path: Path) -> None:
    def no_session(*a: Any, **k: Any) -> Any:
        raise AssertionError("T-0111g B2: the session must never be constructed here")

    monkeypatch.setattr(cli, "Session", no_session)
    explode_clients(monkeypatch)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    monkeypatch.setattr(
        preflight, "gather_regime", lambda *a, **k: preflight.Facts(), raising=False
    )


class Events(list):  # type: ignore[type-arg]
    def index_of(self, name: str) -> int:
        found = [i for i, e in enumerate(self) if e[0] == name]
        assert found, f"T-0111g B5: no {name!r} happened; events {list(self)}"
        return found[0]


def fake_live(
    monkeypatch: pytest.MonkeyPatch,
    events: Events,
    *,
    open_error: BaseException | None = None,
    check: bool = True,
) -> list[Any]:
    """The spy executor and the fake start-page server, wherever ``cli`` reaches them; returns the executors made."""
    made: list[Any] = []

    class SpyExecutor(FakeLiveBrowser):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            self.kwargs = kwargs
            made.append(self)

        async def open(self, url: str) -> None:
            events.append(("browser.open", url))
            if open_error is not None:
                raise open_error

        async def close(self) -> None:
            events.append(("browser.close",))

    class FakePages:
        def __init__(self, static_dir: Any = None) -> None:
            events.append(("pages.init",))

        @property
        def origin(self) -> str:
            return ORIGIN

        def start(self) -> None:
            events.append(("pages.start",))

        def stop(self) -> None:
            events.append(("pages.stop",))

    _patch_both(monkeypatch, "local_astra.computer.executor", "BrowserToolExecutor", SpyExecutor)
    _patch_both(monkeypatch, "local_astra.pages.server", "StartPageServer", FakePages)
    if check:
        monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return made


def null_playwright(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in list(sys.modules) if m == "playwright" or m.startswith("playwright.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "playwright", None)
    with pytest.raises(ImportError):
        importlib.import_module("playwright.async_api")


def argv(tmp_path: Path, *extra: str, config_body: str = "") -> list[str]:
    config = T.write_config(tmp_path)
    if config_body:
        config.write_text(config.read_text() + config_body)
    return T.session_argv(config, tmp_path, *extra)


def _out(capsys: pytest.CaptureFixture[str]) -> tuple[str, str]:
    got = capsys.readouterr()
    return got.out, got.err


# --------------------------------------------------------------------------------------------- B2: argv


def test_argv_live_accepted(tmp_path, capsys, monkeypatch):
    """B2: ``--tools live`` and ``--browser headed|headless`` parse; ``--browser`` takes nothing else."""
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    explode_clients(monkeypatch)
    code = run_cli([*argv(tmp_path, "--tools", "live", "--browser", "headless"), "--dry-run"])
    out, err = _out(capsys)
    assert code == 0, (
        f"T-0111g B2: --tools live --browser headless --dry-run exits 0; {code}\n{out}{err}"
    )
    assert any(line.startswith("tools=live ") for line in out.splitlines()), out
    assert run_cli([*argv(tmp_path, "--tools", "live", "--browser", "kiosk"), "--dry-run"]) == 2
    _out(capsys)


def test_no_tool_loop_refused(tmp_path, capsys, monkeypatch):
    """B2 (mutation 6; replaces R3's deleted row): ``live`` with the loop off by explicit choice is refused before
    anything is constructed: ``--no-tool-loop`` exits 2 naming the loop; ``tool_loop = false`` in the profile is
    refused at load (B1) or with exit 2, naming the loop."""
    calls = T.Calls()
    bomb_session(monkeypatch, calls, tmp_path)
    bomb_browser(monkeypatch, "B2 (loop off)")
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    code = run_cli(argv(tmp_path, "--tools", "live", "--no-tool-loop", "--browser", "headless"))
    out, err = _out(capsys)
    assert code == 2, f"T-0111g B2: --tools live --no-tool-loop exits 2; got {code}\n{out}{err}"
    assert "loop" in err.lower() and "live" in err, err
    assert not calls.detok_loads, "refused before anything is loaded"
    try:
        code = run_cli(
            argv(
                tmp_path, "--browser", "headless", config_body="tools = 'live'\ntool_loop = false\n"
            )
        )
        out, err = _out(capsys)
        message = out + err
        assert code != 0, f"refused: non-zero exit; got {code}"
    except AssertionError:
        raise
    except Exception as exc:  # refused at config load (B1)
        message = str(exc)
    assert "loop" in message.lower() and "live" in message, message
    # premise of the refusal (the too-wide direction): with the loop on, live is not refused for the loop
    code = run_cli([*argv(tmp_path, "--tools", "live", "--browser", "headless"), "--dry-run"])
    out, err = _out(capsys)
    assert "loop" not in err.lower() or code == 0, err


def _refused_names(tmp_path, capsys, monkeypatch, *extra: str, why: str) -> str:
    calls = T.Calls()
    bomb_session(monkeypatch, calls, tmp_path)
    bomb_browser(monkeypatch, why)
    bomb_processes(monkeypatch, why)
    code = run_cli(argv(tmp_path, "--tools", "live", *extra))
    out, err = _out(capsys)
    assert code == 2, (
        f"T-0111g {why}: refused with exit 2 before anything is constructed; got {code}\n{out}{err}"
    )
    assert not calls.detok_loads, f"T-0111g {why}: nothing is loaded before the refusal"
    return err


def test_missing_playwright_names_p1(tmp_path, capsys, monkeypatch):
    """B2: ``playwright`` not importable -> exit 2 naming P1 and the fix; no process is started (the check never
    launches)."""
    null_playwright(monkeypatch)
    err = _refused_names(
        tmp_path, capsys, monkeypatch, "--browser", "headless", why="B2 (no playwright)"
    )
    assert "P1" in err and P1_FIX in err, f"the message names P1 and `{P1_FIX}`; it said:\n{err}"


def test_missing_chromium_names_p1(tmp_path, capsys, monkeypatch):
    """B2: the pinned Chromium absent (an empty ``PLAYWRIGHT_BROWSERS_PATH``) -> exit 2 naming P1 and the fix;
    the check reads the path and launches nothing."""
    empty = tmp_path / "no-browsers"
    empty.mkdir()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(empty))
    err = _refused_names(
        tmp_path, capsys, monkeypatch, "--browser", "headless", why="B2 (no chromium)"
    )
    assert "P1" in err and P1_FIX in err, f"the message names P1 and `{P1_FIX}`; it said:\n{err}"


def _revisions() -> dict[str, str]:
    """The installed Playwright's browser revisions, read from its ``browsers.json`` (where Playwright reads them);
    the row is skipped, saying so, when Playwright is not installed (its refusal is then P1 for another reason)."""
    import importlib.util
    import json

    spec = importlib.util.find_spec("playwright")
    if spec is None or spec.origin is None:
        pytest.skip(
            "premise: playwright is installed (the `browser` extra); without it P1 is refused anyway"
        )
    listing = json.loads(
        (Path(spec.origin).parent / "driver" / "package" / "browsers.json").read_text()
    )
    return {str(b["name"]): str(b["revision"]) for b in listing["browsers"]}


def _scratch_browsers(root: Path, *, chromium: bool, shell: bool) -> Path:
    """A scratch ``PLAYWRIGHT_BROWSERS_PATH`` in Playwright's layout: ``chromium-<rev>/chrome-linux64/chrome`` and
    ``chromium_headless_shell-<rev>/chrome-headless-shell-linux64/chrome-headless-shell`` (what headless launches)."""
    revs = _revisions()
    root.mkdir()
    paths = []
    if chromium:
        paths.append(root / f"chromium-{revs['chromium']}" / "chrome-linux64" / "chrome")
    if shell:
        paths.append(
            root
            / f"chromium_headless_shell-{revs['chromium-headless-shell']}"
            / "chrome-headless-shell-linux64"
            / "chrome-headless-shell"
        )
    for exe in paths:
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"#!/bin/false\n")
        exe.chmod(0o755)
    return root


def test_headless_without_headless_shell_names_p1(tmp_path, monkeypatch):
    """Review B-F2 (implementer fix; red at 6d2b3af): headless launches the headless shell, not ``chrome``. With a
    browsers path holding only ``chromium-<rev>`` (no ``chromium_headless_shell-<rev>``),
    ``live_prerequisites("headless")`` refuses and names P1 and the fix. Too-wide companion: with both executables
    present it does not refuse."""
    only = _scratch_browsers(tmp_path / "only-chromium", chromium=True, shell=False)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(only))
    got = cli.live_prerequisites("headless")
    assert got is not None, (
        "T-0111g review B-F2: chromium-<rev> alone is not what headless launches; live_prerequisites passed"
    )
    assert "P1" in got and P1_FIX in got, got
    both = _scratch_browsers(tmp_path / "both", chromium=True, shell=True)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(both))
    assert cli.live_prerequisites("headless") is None, (
        "with chrome and the headless shell both present, headless is available"
    )


def test_headed_without_display_refused(tmp_path, capsys, monkeypatch):
    """B2: headed (the profile default) with neither ``DISPLAY`` nor ``WAYLAND_DISPLAY`` -> exit 2 suggesting
    ``--browser headless``; and ``--browser headless`` without a display is not refused for it (too-wide)."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    err = _refused_names(tmp_path, capsys, monkeypatch, why="B2 (no display)")
    assert "--browser headless" in err, err
    monkeypatch.undo()
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    explode_clients(monkeypatch)
    fake_live(monkeypatch, Events(), check=False)
    code = run_cli([*argv(tmp_path, "--tools", "live", "--browser", "headless"), "--dry-run"])
    _, err = _out(capsys)
    assert code == 0 and "--browser headless" not in err, (
        f"headless needs no display ({code}):\n{err}"
    )


# --------------------------------------------------------------------------------------------- B3: --dry-run


@pytest.mark.parametrize(
    ("flag", "body", "browser", "viewport"),
    [
        (["--browser", "headless"], "", "headless", (1280, 800)),
        ([], "browser_viewport = [1024, 768]\n", "headed", (1024, 768)),
    ],
    ids=["headless_default_viewport", "headed_profile_viewport"],
)
# T-0160 ruling 2: web on by default in live (LIVE_LINE says egress=web)
def test_dry_run_line_live(tmp_path, capsys, monkeypatch, flag, body, browser, viewport):
    """B3: the line reads ``tools=live native:<sha12> repair=<on|off(...)> loop=on(<N>) browser=<...>
    egress=loopback viewport=<W>x<H>``, from the profile and the flags."""
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    explode_clients(monkeypatch)
    fake_live(monkeypatch, Events())
    code = run_cli([*argv(tmp_path, "--tools", "live", *flag, config_body=body), "--dry-run"])
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    # T-0202 D3 re-pin: the line read through admit_line (the default wait token and the wait set's sha12)
    found = [m for m in (LIVE_LINE.match(t0202.admit_line(line)) for line in out.splitlines()) if m]
    assert len(found) == 1, f"T-0111g B3: one live tools line; printed:\n{out}"
    m = found[0]
    assert m.group(1) == T.TOOLS_SHA12 and m.group(2) == "on" and m.group(3) == "20", (
        f"T-0178 A3: a default web-live line says loop=on(20); {m.group(0)}"
    )
    assert m.group(4) == browser and (int(m.group(5)), int(m.group(6))) == viewport, m.group(0)


# T-0160 ruling 2: web on by default in live (LIVE_LINE says egress=web)
def test_dry_run_launches_nothing(tmp_path, capsys, monkeypatch):
    """B3 (adversarial, mutation 15): ``--dry-run`` with ``live`` constructs no ``PlaywrightDriver``, no executor and
    no start-page server, starts no Playwright and binds nothing; it still prints the live line."""
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    explode_clients(monkeypatch)
    bomb_browser(monkeypatch, "B3 (--dry-run)")
    bomb_binds(monkeypatch, "B3 (--dry-run)")
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    code = run_cli([*argv(tmp_path, "--tools", "live", "--browser", "headless"), "--dry-run"])
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    # T-0202 D3 re-pin: the line read through admit_line (the default wait token and the wait set's sha12)
    assert any(LIVE_LINE.match(t0202.admit_line(line)) for line in out.splitlines()), out


@pytest.mark.parametrize(
    ("flags", "want"),
    [
        # T-0183 A3a: each BASE line plus `` visual_tokens=1120`` (the budget, 1120 by default in every mode)
        ([], "tools=off visual_tokens=1120"),
        (
            ["--tools", "dry-run"],
            f"tools=dry_run native:{T.TOOLS_SHA12} repair=on loop=on(8) visual_tokens=1120",
        ),
        (
            ["--tools", "dry-run", "--no-tool-loop"],
            f"tools=dry_run native:{T.TOOLS_SHA12} repair=on visual_tokens=1120",
        ),
    ],
    ids=["off", "dry_run_loop", "dry_run_no_loop"],
)
def test_off_and_dry_run_lines_unchanged(tmp_path, capsys, monkeypatch, flags, want):
    """B3/B17: the ``off`` and ``dry_run`` lines are BASE's (9bebf69), byte for byte (written out from
    ``cli.tools_line`` at BASE and T-0114 criterion 14), plus T-0183 A3a's `` visual_tokens=1120`` suffix."""
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    explode_clients(monkeypatch)
    code = run_cli([*argv(tmp_path, *flags), "--dry-run"])
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    lines = [line for line in out.splitlines() if line.startswith("tools=")]
    assert lines == [want], f"T-0111g B17: the --dry-run tools line changed: {lines}"


# --------------------------------------------------------------------------------------------- B5: the kit


class _FixedClock:
    def __init__(self) -> None:
        self.now_ms = 424_242

    def call_at(self, at_ms: int, callback: Any) -> Any:
        raise AssertionError("the graph is never started in this row")


@pytest.mark.parametrize("browser", ["headed", "headless"])
def test_kit_wiring(tmp_path, capsys, monkeypatch, browser):
    """B5: ``kit.mode == "live"``, ``kit.executor is kit.live_browser`` (the one executor ``cli`` built), no
    guard, the session clock as ``now_ms``, ``headless`` from ``--browser``, ``end_when_closed == headed``,
    ``allowed_origins == {start-page origin}``, the profile's viewport; the start-page server was started."""
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    monkeypatch.setattr(cli, "SessionClock", _FixedClock)
    events = Events()
    made = fake_live(monkeypatch, events)
    code = run_cli(argv(tmp_path, "--tools", "live", "--browser", browser))
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    assert len(made) == 1, f"T-0111g B5: cli builds one BrowserToolExecutor; built {len(made)}"
    kit = calls.kit()
    assert kit is not None and kit.mode == "live", kit
    assert getattr(kit, "live_browser", None) is made[0] and kit.executor is made[0], (
        "T-0111g B5: kit.executor is kit.live_browser, the executor cli built"
    )
    kw = made[0].kwargs
    assert kw.get("guard") is None, (
        f"T-0111g B5: no confirmation guard (H-T107); {kw.get('guard')!r}"
    )
    assert callable(kw.get("now_ms")) and kw["now_ms"]() == 424_242, "the session clock is now_ms"
    assert kw.get("headless") is (browser == "headless"), kw
    assert kw.get("end_when_closed") is (browser == "headed"), kw
    assert frozenset(kw.get("allowed_origins") or ()) == frozenset({ORIGIN}), kw
    assert tuple(kw.get("viewport", (1280, 800))) == (1280, 800), kw
    assert events.index_of("pages.start") >= 0


class _Hud:
    events: Events

    def __init__(self, session: Any, **kwargs: Any) -> None:
        self.session = session
        self.port = 0
        self.url = "ws://127.0.0.1:0/v1/realtime"

    async def start(self) -> None:
        type(self).events.append(("hud.start",))
        asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGINT)

    async def aclose(self) -> None:
        type(self).events.append(("hud.close",))


def _serve_rig(tmp_path, monkeypatch, events: Events, **kw: Any) -> list[str]:
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    T.point_cli_at(
        monkeypatch, payloads=tmp_path / "nowhere" / "native_T4.json"
    )  # repair off: no warm-up
    _Hud.events = events
    monkeypatch.setattr(cli, "HudServer", _Hud)
    fake_live(monkeypatch, events, **kw)
    return argv(tmp_path, "--tools", "live", "--browser", "headless")


def test_startup_and_shutdown_order(tmp_path, capsys, monkeypatch):
    """B5: the start-page server is started, then the start page opened in the browser, before the HUD endpoint
    starts; at shutdown (Ctrl-C) the browser is closed and then the start-page server stopped."""
    events = Events()
    code = run_cli(_serve_rig(tmp_path, monkeypatch, events))
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}\nevents {list(events)}"
    opened = [e for e in events if e[0] == "browser.open"]
    assert len(opened) == 1 and str(opened[0][1]).startswith(ORIGIN), events
    # review X25: the start page is the origin's root, exactly
    assert opened[0][1] == ORIGIN + "/", f"T-0111g B5: the start page is {ORIGIN}/; opened {opened}"
    order = [
        events.index_of(n)
        for n in ("pages.start", "browser.open", "hud.start", "browser.close", "pages.stop")
    ]
    assert order == sorted(order), f"T-0111g B5: start/stop order; events {list(events)}"


class StartBoom(RuntimeError):
    pass


def test_startup_failure_exits_1_before_hud(tmp_path, capsys, monkeypatch):
    """B5: a failure opening the start page exits 1 with one stderr line naming the exception type, before the
    HUD endpoint binds; the start-page server is stopped."""
    events = Events()
    code = run_cli(_serve_rig(tmp_path, monkeypatch, events, open_error=StartBoom("no page")))
    out, err = _out(capsys)
    assert code == 1, f"T-0111g B5: exit 1; got {code}\n{out}{err}"
    named = [line for line in err.splitlines() if "StartBoom" in line]
    assert len(named) == 1, f"one stderr line names StartBoom; stderr:\n{err}"
    # review B-F1 (tightened): that line is the failure's only stderr line
    assert failure_lines(err) == named, f"the start failure is one stderr line; stderr:\n{err}"
    assert not [e for e in events if e[0] == "hud.start"], (
        f"the HUD endpoint never started; {list(events)}"
    )
    assert ("pages.stop",) in events, f"nothing is left bound; {list(events)}"
    # review X5 (B): the browser that failed to open the start page is closed, before the start pages stop
    assert ("browser.close",) in events, f"T-0111g B5: the browser is closed; {list(events)}"
    assert events.index_of("browser.close") < events.index_of("pages.stop"), list(events)


#: the one other stderr line this rig prints: ``_serve_rig`` points the payloads nowhere so no warm-up runs
REPAIR_OFF_PREFIX = "astra session: tool repair off for this session:"


def failure_lines(err: str) -> list[str]:
    """Every non-empty stderr line except the rig's own repair-off announcement."""
    return [
        line for line in err.splitlines() if line.strip() and not line.startswith(REPAIR_OFF_PREFIX)
    ]


def test_startup_failure_multiline_message_is_one_line(tmp_path, capsys, monkeypatch):
    """Review B-F1 (implementer fix; red at 6d2b3af): Playwright's launch errors are multi-line boxes. A start-page
    failure whose message spans several lines still gives exactly one non-empty stderr line: the exception type and
    the message's first line, whitespace collapsed; nothing of the later lines."""
    events = Events()
    message = (
        "  launch   failed:\tno sandbox  \n╔══════════╗\n║ BOX-LINE-2 ║\n╚══════════╝\nTAIL-LINE"
    )
    code = run_cli(_serve_rig(tmp_path, monkeypatch, events, open_error=StartBoom(message)))
    out, err = _out(capsys)
    assert code == 1, f"T-0111g B5: exit 1; got {code}\n{out}{err}"
    lines = failure_lines(err)
    assert len(lines) == 1, f"T-0111g review B-F1: one stderr line; got {len(lines)}:\n{err}"
    assert "StartBoom" in lines[0] and "launch failed: no sandbox" in lines[0], lines[0]
    for later in ("BOX-LINE-2", "TAIL-LINE", "╔"):
        assert later not in err, (
            f"only the first line of the message is printed; {later!r} in:\n{err}"
        )


@pytest.mark.parametrize("where", ["load_tool_kit", "_build_graph"])
def test_start_pages_stopped_when_build_fails(tmp_path, capsys, monkeypatch, where):
    """Review X18/X19: in ``live`` the start-page server is bound before the kit and the graph are built, so when
    ``load_tool_kit`` (the detokeniser refused) or ``_build_graph`` (the audio device refused) raises, the server is
    stopped and the session exits 1 without serving."""
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    events = Events()
    fake_live(monkeypatch, events)

    def refuse(*a: Any, **k: Any) -> Any:
        events.append((where,))
        if where == "load_tool_kit":
            raise cli.ToolsUnavailable("--tools live refused: detokeniser BUILD-BOOM")
        raise cli.AudioDeviceUnavailable("no device BUILD-BOOM")

    monkeypatch.setattr(cli, where, refuse)
    code = run_cli(argv(tmp_path, "--tools", "live", "--browser", "headless"))
    out, err = _out(capsys)
    assert (where,) in events, f"premise: {where} ran and raised; {list(events)}"
    assert code == 1 and "BUILD-BOOM" in err, f"premise: refused with exit 1; {code}\n{out}{err}"
    assert not calls.graphs, "nothing was served"
    assert events.index_of("pages.start") < events.index_of(where)
    assert ("pages.stop",) in events, (
        f"T-0111g B5: the start-page server is stopped when {where} raises; {list(events)}"
    )


# T-0160 ruling 2: web on by default in live (LIVE_LINE says egress=web)
def test_profile_viewport_reaches_executor_and_dry_run_line(tmp_path, capsys, monkeypatch):
    """Review X8: a non-default ``browser_viewport`` in the profile is the executor's ``viewport`` keyword and the
    ``--dry-run`` line's ``viewport=<W>x<H>``."""
    monkeypatch.setenv("DISPLAY", ":99")
    body = "browser_viewport = [1440, 900]\n"
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    made = fake_live(monkeypatch, Events())
    code = run_cli(argv(tmp_path, "--tools", "live", "--browser", "headless", config_body=body))
    out, err = _out(capsys)
    assert code == 0 and len(made) == 1, f"{code}\n{out}{err}"
    assert tuple(made[0].kwargs.get("viewport", ())) == (1440, 900), (
        f"T-0111g B5: the profile's viewport reaches the executor; {made[0].kwargs}"
    )
    code = run_cli(
        [*argv(tmp_path, "--tools", "live", "--browser", "headless", config_body=body), "--dry-run"]
    )
    out, err = _out(capsys)
    assert code == 0, f"{code}\n{out}{err}"
    # T-0202 D3 re-pin: the line read through admit_line (the default wait token and the wait set's sha12)
    found = [m for m in (LIVE_LINE.match(t0202.admit_line(line)) for line in out.splitlines()) if m]
    assert len(found) == 1 and (found[0].group(5), found[0].group(6)) == ("1440", "900"), out
