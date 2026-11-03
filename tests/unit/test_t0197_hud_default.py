"""T-0197: ``astra session`` serves the HUD page by default (docs/tasks/T-0197.md, criteria 1-9).

Seam: ``hud_static.HudStaticServer`` replaced by a recorder (as ``test_hud_static.py::test_the_hud_verb_prints_that_url``
does), over ``T.fake_clients`` / ``T.stop_before_serving`` / ``fake_live``; every row writes its own ``hud_port`` (a free
port) and ``conductor_port`` into the profile, so the expected URL is written out by hand from criterion 1:
``http://127.0.0.1:<hud_port>/?ws=ws://127.0.0.1:<conductor_port>/v1/realtime``. Rows run at the real default (the
shared argv builder's ``--no-serve-hud`` pin is lifted with ``_t0197_pins.hud_default()``) unless they say otherwise.

Regime: CPU over fakes, no measurement; two rows bind a real loopback socket on a port the OS chose.
"""

from __future__ import annotations

import errno
import socket
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, hud_static, preflight
from tests.unit import _t0197_pins as pins
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import run_cli
from tests.unit.test_cli_tools_live import Events, _Hud, fake_live

LIVE = ("--tools", "live", "--browser", "headless")
MODES: dict[str, tuple[str, ...]] = {
    "off": (),
    "dry_run": ("--tools", "dry-run"),
    "live-web": LIVE,
    "live-no-web": (*LIVE, "--no-web"),
}
PROXY_ENV = ("DDGS_PROXY", "PRIMP_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
REAL_SERVER = hud_static.HudStaticServer


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("ASTRA_AUDIO_SOURCE", raising=False)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def page_url(hud: int, conductor: int) -> str:
    """Criterion 1's URL, written out by hand."""
    return f"http://127.0.0.1:{hud}/?ws=ws://127.0.0.1:{conductor}/v1/realtime"


def make_recorder(raise_: BaseException | None = None) -> type:
    class Recorder:
        built: list[tuple[Path, int]] = []
        closed: list[int] = []
        events: list[str] | None = None

        def __init__(self, root: Any, port: int, *a: Any, **k: Any) -> None:
            type(self).built.append((Path(root), int(port)))
            if type(self).events is not None:
                type(self).events.append("bind")
            if raise_ is not None:
                raise raise_
            self.port = int(port)
            self.address = ("127.0.0.1", self.port)

        def serve_forever(self) -> None:
            return None

        def close(self) -> None:
            type(self).closed.append(self.port)

    Recorder.built = []
    Recorder.closed = []
    return Recorder


class Run:
    def __init__(self, code: int, out: str, err: str, calls: T.Calls, hud: type | None) -> None:
        self.code, self.out, self.err, self.calls, self.hud = code, out, err, calls, hud


def ports_body(hud: int, conductor: int) -> str:
    return f"\n[profiles.lean.services]\nhud_port = {hud}\nconductor_port = {conductor}\n"


def start(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: pytest.MonkeyPatch,
    *flags: str,
    hud: type | None,
    ports: tuple[int, int],
    dry: bool = False,
    drop: tuple[str, ...] = (),
) -> Run:
    """``astra session <flags>`` at the real default (no opt-out pin), a real start stopped before serving or
    ``--dry-run``; ``hud`` replaces ``hud_static.HudStaticServer`` (``None``: the real class)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    monkeypatch.setattr(hud_static, "HudStaticServer", hud if hud is not None else REAL_SERVER)
    fake_live(monkeypatch, Events())
    if "--silent" in drop:
        # a speaking session opens the output device: the fake one (T-0181's seam)
        from tests.conftest import VirtualClock
        from tests.fakes.audio import FakeAudioIO

        monkeypatch.setattr(
            cli, "open_audio", lambda args: FakeAudioIO(VirtualClock()), raising=False
        )
    config = T.write_config(tmp_path)
    config.write_text(config.read_text() + ports_body(*ports))
    with pins.hud_default():
        argv = T.session_argv(config, tmp_path, *flags)
    argv = [a for a in argv if a not in drop]
    if dry:
        argv.append("--dry-run")
    capsys.readouterr()
    code = run_cli(argv)
    got = capsys.readouterr()
    return Run(code, got.out, got.err, calls, hud)


def is_at_lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(pins.IS_AT)]


# --- criterion 1: default on, every mode -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "drop"),
    [
        *[(m, ()) for m in MODES],
        ("off", ("--silent",)),
        ("live-no-web", ("--silent",)),
    ],
    ids=[*MODES, "off-not-silent", "live-no-web-not-silent"],
)
def test_default_serves_in_every_mode(tmp_path, capsys, monkeypatch, mode, drop) -> None:
    """Criterion 1: neither flag → one ``HudStaticServer(hud_static.HUD_DIR, hud_port)``, the session runs (``_serve``
    reached), and stdout carries ``astra session: the HUD page is at <url>`` exactly once with the full URL."""
    hud_port, conductor = free_port(), free_port()
    rec = make_recorder()
    run = start(
        tmp_path, capsys, monkeypatch, *MODES[mode], hud=rec, ports=(hud_port, conductor), drop=drop
    )
    assert run.code == 0, f"{mode}: exit {run.code}\n{run.out}{run.err}"
    assert len(run.calls.graphs) == 1, f"premise: the session was built and served; {run.err}"
    assert rec.built == [(Path(hud_static.HUD_DIR), hud_port)], (
        f"T-0197 criterion 1 ({mode}): one HudStaticServer(HUD_DIR, {hud_port}) by default; built {rec.built}"
    )
    want = pins.IS_AT + page_url(hud_port, conductor)
    assert is_at_lines(run.out) == [want], (
        f"T-0197 criterion 1 ({mode}): exactly one {want!r} on stdout; got {is_at_lines(run.out)}\n{run.out}"
    )
    assert pins.IS_AT not in run.err, run.err


# --- criterion 2: placement ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["off", "live-web"])
def test_line_follows_the_regime_line(tmp_path, capsys, monkeypatch, mode) -> None:
    """Criterion 2: the HUD line is the first start-up line after ``astra session: regime: …``."""
    hud_port, conductor = free_port(), free_port()
    rec = make_recorder()
    run = start(tmp_path, capsys, monkeypatch, *MODES[mode], hud=rec, ports=(hud_port, conductor))
    assert run.code == 0, f"{run.out}{run.err}"
    lines = run.out.splitlines()
    regime = [i for i, x in enumerate(lines) if x.startswith(pins.REGIME_PREFIX)]
    assert len(regime) == 1, f"premise: one regime line\n{run.out}"
    at = [i for i, x in enumerate(lines) if x.startswith(pins.IS_AT)]
    assert at == [regime[0] + 1], (
        f"T-0197 criterion 2 ({mode}): the HUD line is the line right after the regime line; regime at "
        f"{regime[0]}, HUD at {at}\n{run.out}"
    )


def test_bind_precedes_regime_gathering(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 2 (bind before the regime line), through the order of two recorded events: the HUD construction
    and ``preflight.gather_regime`` (whose result the regime line prints)."""
    hud_port, conductor = free_port(), free_port()
    rec = make_recorder()
    order: list[str] = []
    rec.events = order
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    monkeypatch.setattr(hud_static, "HudStaticServer", rec)
    fake_live(monkeypatch, Events())
    patched = preflight.gather_regime

    def gather(*a: Any, **k: Any) -> Any:
        order.append("regime")
        return patched(*a, **k)

    monkeypatch.setattr(preflight, "gather_regime", gather)
    config = T.write_config(tmp_path)
    config.write_text(config.read_text() + ports_body(hud_port, conductor))
    with pins.hud_default():
        argv = T.session_argv(config, tmp_path)
    code = run_cli(argv)
    out = capsys.readouterr()
    assert code == 0, f"{out.out}{out.err}"
    assert order == ["bind", "regime"], (
        f"T-0197 criterion 2: the HUD binds before the regime line is measured and printed; order {order}"
    )


def test_dry_run_default_prints_would_be_served_last_and_binds_nothing(
    tmp_path, capsys, monkeypatch
) -> None:
    """Criterion 2: under ``--dry-run`` with neither flag, ``astra session: the HUD page would be served at <url>``
    exactly once, as the last stdout line; nothing is constructed (``--dry-run`` constructs nothing)."""
    for mode in MODES:
        hud_port, conductor = free_port(), free_port()
        rec = make_recorder()
        run = start(
            tmp_path / mode,
            capsys,
            monkeypatch,
            *MODES[mode],
            hud=rec,
            ports=(hud_port, conductor),
            dry=True,
        )
        assert run.code == 0, f"{mode}: {run.out}{run.err}"
        want = pins.WOULD_BE + page_url(hud_port, conductor)
        lines = run.out.splitlines()
        assert [x for x in lines if x.startswith(pins.WOULD_BE)] == [want] and lines[-1] == want, (
            f"T-0197 criterion 2 ({mode}): {want!r} once, as the last stdout line\n{run.out}"
        )
        assert rec.built == [], f"--dry-run binds nothing; built {rec.built}"
        assert pins.IS_AT not in run.out + run.err


# --- criterion 3: opt-out ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["real", "dry"])
@pytest.mark.parametrize("mode", ["off", "live-web"])
def test_no_serve_hud_constructs_nothing(tmp_path, capsys, monkeypatch, kind, mode) -> None:
    """Criterion 3: ``--no-serve-hud`` constructs no ``HudStaticServer`` and prints no HUD line, real start and
    ``--dry-run``; the session still runs."""
    hud_port, conductor = free_port(), free_port()
    rec = make_recorder()
    run = start(
        tmp_path,
        capsys,
        monkeypatch,
        *MODES[mode],
        "--no-serve-hud",
        hud=rec,
        ports=(hud_port, conductor),
        dry=kind == "dry",
    )
    assert "unrecognized arguments" not in run.err, (
        f"T-0197 criterion 3: --no-serve-hud parses; {run.err}"
    )
    assert run.code == 0, f"{run.out}{run.err}"
    if kind == "real":
        assert len(run.calls.graphs) == 1, f"premise: the session ran; {run.err}"
    assert rec.built == [], (
        f"T-0197 criterion 3: --no-serve-hud constructs nothing; built {rec.built}"
    )
    assert "HUD page" not in run.out + run.err, (
        f"T-0197 criterion 3: no HUD line with --no-serve-hud\n{run.out}{run.err}"
    )


# --- criterion 4: compatibility ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["real", "dry"])
@pytest.mark.parametrize("mode", ["off", "live-no-web"])
def test_serve_hud_is_a_noop(tmp_path, capsys, monkeypatch, kind, mode) -> None:
    """Criterion 4: ``--serve-hud`` behaves like no flag (exit, stdout, stderr byte-equal, one construction); with
    both flags the last wins (``--serve-hud --no-serve-hud`` → none; ``--no-serve-hud --serve-hud`` → served)."""
    hud_port, conductor = free_port(), free_port()
    dry = kind == "dry"
    runs: dict[str, Run] = {}
    for name, extra in (
        ("none", ()),
        ("serve", ("--serve-hud",)),
        ("serve-then-no", ("--serve-hud", "--no-serve-hud")),
        ("no-then-serve", ("--no-serve-hud", "--serve-hud")),
    ):
        rec = make_recorder()
        runs[name] = start(
            tmp_path / name,
            capsys,
            monkeypatch,
            *MODES[mode],
            *extra,
            hud=rec,
            ports=(hud_port, conductor),
            dry=dry,
        )
    norm = {
        k: (
            r.code,
            r.out.replace(str(tmp_path / k), "<tmp>"),
            r.err.replace(str(tmp_path / k), "<tmp>"),
        )
        for k, r in runs.items()
    }
    for name, r in runs.items():
        assert "unrecognized arguments" not in r.err and "usage:" not in r.err, (
            f"T-0197 criterion 4 ({name}): never a usage error\n{r.err}"
        )
    assert norm["serve"] == norm["none"], (
        f"T-0197 criterion 4: --serve-hud is byte-equal to no flag\nnone: {norm['none']}\nserve: {norm['serve']}"
    )
    assert norm["no-then-serve"] == norm["none"], (
        "T-0197 criterion 4: last wins (--serve-hud last → served)"
    )
    want_built = [] if dry else [(Path(hud_static.HUD_DIR), hud_port)]
    for name in ("none", "serve", "no-then-serve"):
        assert runs[name].hud.built == want_built, (name, runs[name].hud.built)  # type: ignore[union-attr]
        line = (pins.WOULD_BE if dry else pins.IS_AT) + page_url(hud_port, conductor)
        assert runs[name].out.count(line) == 1, (name, runs[name].out)
    assert runs["serve-then-no"].hud.built == [], (
        "T-0197 criterion 4: last wins (--no-serve-hud last → none)"
    )  # type: ignore[union-attr]
    assert "HUD page" not in runs["serve-then-no"].out + runs["serve-then-no"].err
    assert runs["serve-then-no"].code == 0


def test_help_names_both_flags(capsys) -> None:
    """Criterion 4: ``session --help`` names ``--serve-hud`` and ``--no-serve-hud`` and says the default is on."""
    code = run_cli(["session", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--serve-hud" in out and "--no-serve-hud" in out, (
        f"T-0197 criterion 4: --help names both flags\n{out}"
    )
    entries = out.split("\n  -")
    found = [e for e in entries if e.lstrip("-").startswith(("serve-hud", "no-serve-hud"))]
    assert found, f"T-0197 criterion 4: --help has an entry for the flag\n{out}"
    block = found[0]
    assert "default" in block.lower() and ("true" in block.lower() or " on" in block.lower()), (
        f"T-0197 criterion 4: the flag's help says it is on by default; {block!r}"
    )


# --- criterion 5: a busy port never ends the session (adversarial) ------------------------------------------------


def _spy_backend(monkeypatch: pytest.MonkeyPatch) -> list[str]:
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


BUSY = {
    "EADDRINUSE": OSError(errno.EADDRINUSE, "Address already in use"),
    "EACCES": OSError(errno.EACCES, "Permission denied"),
}


def _check_busy_line(line: str, hud_port: int, conductor: int) -> None:
    url = page_url(hud_port, conductor)
    assert line.startswith("astra session: "), line
    assert f"127.0.0.1:{hud_port}" in line, (
        f"T-0197 criterion 5: the line names 127.0.0.1:{hud_port}; {line!r}"
    )
    assert "in use" in line, f"T-0197 criterion 5: the line says the port is in use; {line!r}"
    assert url in line, f"T-0197 criterion 5: the line gives the full page URL {url}; {line!r}"
    assert "continu" in line, f"T-0197 criterion 5: the line says the session continues; {line!r}"


@pytest.mark.parametrize("why", sorted(BUSY))
@pytest.mark.parametrize("mode", ["off", "live-web"])
def test_busy_port_one_line_and_continues(tmp_path, capsys, monkeypatch, why, mode) -> None:
    """Criterion 5 (adversarial: the failure mode is the old T-0125 amendment 4 (c) exit): the server's construction
    raises ``OSError``: exit equals the ``--no-serve-hud`` run's; ``_serve`` is reached; stdout is byte-equal to the
    opt-out run's (no ``is at``); stderr is the opt-out run's plus exactly one line, the first after the regime line
    in time, naming ``127.0.0.1:<hud_port>``, "in use", the URL; no traceback; one construction, on that port (no
    retry, no other port); in live the search backend is not closed."""
    hud_port, conductor = free_port(), free_port()
    closes = _spy_backend(monkeypatch)
    opt = start(
        tmp_path / "opt",
        capsys,
        monkeypatch,
        *MODES[mode],
        "--no-serve-hud",
        hud=make_recorder(),
        ports=(hud_port, conductor),
    )
    assert opt.code == 0, f"premise: the opt-out run starts; {opt.out}{opt.err}"
    opt_closes = list(closes)
    rec = make_recorder(BUSY[why])
    busy = start(
        tmp_path / "busy", capsys, monkeypatch, *MODES[mode], hud=rec, ports=(hud_port, conductor)
    )
    assert busy.code == opt.code, (
        f"T-0197 criterion 5 ({why}): a busy HUD port never ends the session; exit {busy.code}\n{busy.out}{busy.err}"
    )
    assert len(busy.calls.graphs) == 1, (
        f"T-0197 criterion 5: the session runs on (_serve reached); {busy.err}"
    )
    assert rec.built == [(Path(hud_static.HUD_DIR), hud_port)], (
        f"T-0197 criterion 5: one attempt, on {hud_port}; no retry, no other port; built {rec.built}"
    )
    assert "Traceback" not in busy.out + busy.err, busy.err
    norm_out = busy.out.replace(str(tmp_path / "busy"), "<tmp>")
    assert norm_out == opt.out.replace(str(tmp_path / "opt"), "<tmp>"), (
        f"T-0197 criterion 5: stdout as with --no-serve-hud (no `is at`)\nbusy:\n{busy.out}\nopt:\n{opt.out}"
    )
    busy_err = busy.err.replace(str(tmp_path / "busy"), "<tmp>").splitlines()
    opt_err = opt.err.replace(str(tmp_path / "opt"), "<tmp>").splitlines()
    added = [x for x in busy_err if x not in opt_err]
    assert len(added) == 1, f"T-0197 criterion 5: exactly one added stderr line; added {added}"
    assert [x for x in busy_err if x != added[0]] == opt_err, (busy_err, opt_err)
    _check_busy_line(added[0], hud_port, conductor)
    if mode.startswith("live"):
        assert closes == opt_closes, (
            f"T-0197 criterion 5: the search backend is not closed on a busy HUD port; closes {closes[len(opt_closes) :]}"
        )


def test_real_bind_busy_port(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 5 with a real socket held on ``hud_port`` (criterion 7: no probe of the busy port): the real
    ``HudStaticServer`` fails to bind, one stderr line, exit 0, the session ran, nobody connected to the holder."""
    conductor = free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(4)
        held.setblocking(False)
        hud_port = int(held.getsockname()[1])
        run = start(tmp_path, capsys, monkeypatch, hud=None, ports=(hud_port, conductor))
        try:
            conn, _ = held.accept()
            conn.close()
            probed = True
        except BlockingIOError:
            probed = False
    assert run.code == 0, (
        f"T-0197 criterion 5: the session continues; exit {run.code}\n{run.out}{run.err}"
    )
    assert len(run.calls.graphs) == 1, run.err
    lines = [x for x in run.err.splitlines() if f"127.0.0.1:{hud_port}" in x]
    assert len(lines) == 1, f"T-0197 criterion 5: one stderr line; {run.err}"
    _check_busy_line(lines[0], hud_port, conductor)
    assert pins.IS_AT not in run.out
    assert not probed, (
        "T-0197 criterion 7: the busy port is never probed (a connection reached its holder)"
    )


# --- criterion 6: lifetime ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("exit_", ["return", "startup_failure"])
def test_served_hud_closed_on_every_exit(tmp_path, capsys, monkeypatch, exit_) -> None:
    """Criterion 6: the default-served HUD is closed on the way out of a normal return (a real server on a free port:
    afterwards the port is free) and of a start-up failure after the bind (T-0111g B5: the start page fails to
    open; exit 1)."""
    hud_port, conductor = free_port(), free_port()
    if exit_ == "return":
        built: list[Any] = []

        class Spy(REAL_SERVER):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                built.append(self)

        run = start(tmp_path, capsys, monkeypatch, hud=Spy, ports=(hud_port, conductor))
        assert run.code == 0, f"{run.out}{run.err}"
        assert len(built) == 1, (
            f"T-0197 criterion 1: the default served the page; built {len(built)}"
        )
        assert port_is_free(hud_port), (
            f"T-0197 criterion 6: {hud_port} is free after the session returned"
        )
        return
    from tests.unit.test_cli_tools_live import StartBoom

    rec = make_recorder()
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    T.point_cli_at(monkeypatch, payloads=tmp_path / "nowhere" / "native_T4.json")
    events = Events()
    _Hud.events = events
    monkeypatch.setattr(cli, "HudServer", _Hud)
    fake_live(monkeypatch, events, open_error=StartBoom("no page"))
    monkeypatch.setattr(hud_static, "HudStaticServer", rec)
    config = T.write_config(tmp_path)
    config.write_text(config.read_text() + ports_body(hud_port, conductor))
    with pins.hud_default():
        argv = T.session_argv(config, tmp_path, *LIVE)
    code = run_cli(argv)
    out = capsys.readouterr()
    assert code == 1, f"premise: T-0111g B5 start failure exits 1; {code}\n{out.out}{out.err}"
    assert len(rec.built) == 1, (
        f"T-0197 criterion 1: the default served the page; built {rec.built}"
    )
    assert rec.closed == [hud_port], (
        f"T-0197 criterion 6: the served HUD is closed once on a start-up failure; closed {rec.closed}"
    )


# --- criterion 7: loopback only, no browser ----------------------------------------------------------------------


def test_no_browser_opened(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 7 / H-T197-1: the default serves the page and opens no browser (``webbrowser`` is never asked)."""
    import webbrowser

    asked: list[str] = []

    def nope(*a: Any, **k: Any) -> bool:
        asked.append(repr(a))
        return False

    for name in ("open", "open_new", "open_new_tab"):
        monkeypatch.setattr(webbrowser, name, nope)
    monkeypatch.setattr(
        webbrowser, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("webbrowser.get"))
    )
    hud_port, conductor = free_port(), free_port()
    rec = make_recorder()
    run = start(tmp_path, capsys, monkeypatch, hud=rec, ports=(hud_port, conductor))
    assert run.code == 0, f"{run.out}{run.err}"
    assert len(rec.built) == 1, (
        f"T-0197 criterion 1: the page is served by default; built {rec.built}"
    )
    assert asked == [], f"T-0197 criterion 7: no browser opened; {asked}"


# --- criterion 8: the `hud` verb is unchanged --------------------------------------------------------------------


def test_hud_verb_unchanged_on_busy_port(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 8 against criterion 5, on the same busy port: ``astra hud`` still exits 1 with one
    ``astra hud: cannot bind 127.0.0.1:<port> …`` stderr line, while the session on that port runs on."""
    hud_port, conductor = free_port(), free_port()
    rec = make_recorder(BUSY["EADDRINUSE"])
    session = start(tmp_path, capsys, monkeypatch, hud=rec, ports=(hud_port, conductor))
    assert session.code == 0 and len(session.calls.graphs) == 1, (
        f"T-0197 criterion 5: the session continues; {session.code}\n{session.err}"
    )
    assert rec.built == [(Path(hud_static.HUD_DIR), hud_port)], (
        f"T-0197 criterion 1: the session tried {hud_port} by default; built {rec.built}"
    )
    rec.built.clear()
    config = tmp_path / "hud.toml"
    config.write_text("profile = 'lean'\n[profiles.lean]\n" + ports_body(hud_port, conductor))
    code = run_cli(["--config", str(config), "hud"])
    got = capsys.readouterr()
    assert code == 1, (
        f"T-0197 criterion 8: the hud verb still exits 1 on a busy port; {code}\n{got.err}"
    )
    lines = [x for x in got.err.splitlines() if x.strip()]
    assert len(lines) == 1 and lines[0].startswith(
        f"astra hud: cannot bind 127.0.0.1:{hud_port} "
    ), f"T-0197 criterion 8: one `astra hud: cannot bind …` line; {lines}"
    assert "http://" not in got.out, (
        f"criterion 8: no URL printed by a hud verb that did not bind; {got.out}"
    )


# --- review RV3 (b): the `hud` verb's busy message, byte for byte against BASE 110694b4 ---------------------------


def test_hud_verb_busy_message_is_bases_byte_for_byte(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 8 / RV3: on a busy port ``astra hud`` writes exactly BASE 110694b4's stderr (its ``serve_hud``
    f-string, written out here by hand with the port and the exception substituted) and nothing on stdout; exit 1."""
    hud_port, conductor = free_port(), free_port()
    exc = OSError(errno.EADDRINUSE, "Address already in use")
    monkeypatch.setattr(hud_static, "HudStaticServer", make_recorder(exc))
    config = tmp_path / "hud.toml"
    config.write_text("profile = 'lean'\n[profiles.lean]\n" + ports_body(hud_port, conductor))
    capsys.readouterr()
    code = run_cli(["--config", str(config), "hud"])
    got = capsys.readouterr()
    want = (
        f"astra hud: cannot bind 127.0.0.1:{hud_port} ([Errno {errno.EADDRINUSE}] Address already in use). The "
        f"port is in the profile (services.hud_port); free it or change it there — this never moves to another "
        f"port.\n"
    )
    assert code == 1, code
    assert got.err == want, f"T-0197 criterion 8: BASE's bytes\nwant {want!r}\ngot  {got.err!r}"
    assert got.out == "", got.out


# --- review RV3 (a): one ordered buffer: the busy line is the first line after the regime line ---------------------


@pytest.mark.parametrize("mode", ["off", "live-web"])
def test_busy_line_first_after_regime_in_one_stream(tmp_path, capsys, monkeypatch, mode) -> None:
    """Criterion 2 with criterion 5: stdout and stderr written to one ordered buffer; the busy stderr line is the
    line right after ``astra session: regime: …`` (not printed at the bind, before the tools block)."""
    import contextlib
    import io

    hud_port, conductor = free_port(), free_port()
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    monkeypatch.setattr(hud_static, "HudStaticServer", make_recorder(BUSY["EADDRINUSE"]))
    fake_live(monkeypatch, Events())
    config = T.write_config(tmp_path)
    config.write_text(config.read_text() + ports_body(hud_port, conductor))
    with pins.hud_default():
        argv = T.session_argv(config, tmp_path, *MODES[mode])
    both = io.StringIO()
    with contextlib.redirect_stdout(both), contextlib.redirect_stderr(both):
        code = run_cli(argv)
    assert code == 0, both.getvalue()
    lines = both.getvalue().splitlines()
    regime = [i for i, x in enumerate(lines) if x.startswith(pins.REGIME_PREFIX)]
    busy = [i for i, x in enumerate(lines) if f"127.0.0.1:{hud_port}" in x]
    assert len(regime) == 1 and len(busy) == 1, both.getvalue()
    assert busy[0] == regime[0] + 1, (
        f"T-0197 criterion 2 (RV3, {mode}): the busy line is the first line after the regime line; regime at "
        f"{regime[0]}, busy at {busy[0]}\n{both.getvalue()}"
    )
    _check_busy_line(lines[busy[0]], hud_port, conductor)


# --- review RV2: a failure between the bind and the session closes the served HUD once -----------------------------


class _RegimeBoom(RuntimeError):
    pass


@pytest.mark.parametrize("raise_", ["error", "keyboard_interrupt"])
def test_served_hud_closed_when_regime_gathering_fails(
    tmp_path, capsys, monkeypatch, raise_
) -> None:
    """Criterion 6 / RV2: ``preflight.gather_regime`` raises after the bind (an error, or a ``KeyboardInterrupt``
    landing there): the served HUD is closed exactly once."""
    hud_port, conductor = free_port(), free_port()
    rec = make_recorder()
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    monkeypatch.setattr(hud_static, "HudStaticServer", rec)
    fake_live(monkeypatch, Events())
    boom: BaseException = (
        KeyboardInterrupt() if raise_ == "keyboard_interrupt" else _RegimeBoom("regime")
    )

    def gather(*a: Any, **k: Any) -> Any:
        raise boom

    monkeypatch.setattr(preflight, "gather_regime", gather)
    config = T.write_config(tmp_path)
    config.write_text(config.read_text() + ports_body(hud_port, conductor))
    with pins.hud_default():
        argv = T.session_argv(config, tmp_path)
    with pytest.raises(type(boom)):
        run_cli(argv)
    capsys.readouterr()
    assert rec.built == [(Path(hud_static.HUD_DIR), hud_port)], f"premise: bound once; {rec.built}"
    assert rec.closed == [hud_port], (
        f"T-0197 criterion 6 (RV2, {raise_}): the served HUD is closed exactly once; closed {rec.closed}"
    )
    assert calls.graphs == [], "premise: the failure came before the session was served"


# --- review RV1: tests/integration/test_hud_live.py readiness is the conductor listening ---------------------------


def test_hud_live_readiness_is_the_conductor_listening(tmp_path, capsys, monkeypatch) -> None:
    """RV1: ``test_hud_live.ready_url`` fed a TIP session's stdout line by line (a served start over the fakes, the
    HUD on by default) first answers at ``_serve``'s "open the HUD page and connect to" line, not at the earlier
    ``the HUD page is at …?ws=ws://…/v1/realtime`` line, and answers with the conductor's ws URL."""
    from tests.integration.test_hud_live import ready_url

    hud_port, conductor = free_port(), free_port()
    events = Events()
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    _Hud.events = events
    monkeypatch.setattr(cli, "HudServer", _Hud)
    monkeypatch.setattr(hud_static, "HudStaticServer", make_recorder())
    fake_live(monkeypatch, events)
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = T.write_config(tmp_path)
    config.write_text(config.read_text() + ports_body(hud_port, conductor))
    with pins.hud_default():
        argv = T.session_argv(config, tmp_path)
    code = run_cli(argv)
    lines = capsys.readouterr().out.splitlines()
    assert code == 0, lines
    hud_line = [i for i, x in enumerate(lines) if x.startswith(pins.IS_AT)]
    connect = [i for i, x in enumerate(lines) if "open the HUD page and connect to" in x]
    assert len(hud_line) == 1 and len(connect) == 1 and hud_line[0] < connect[0], (
        f"premise: the HUD page line precedes the connect line\n{lines}"
    )
    first = next((i for i, x in enumerate(lines) if ready_url(x) is not None), None)
    assert first == connect[0], (
        f"RV1: readiness is the connect line ({connect[0]}), not {first}: {lines[first] if first is not None else None}"
    )
    assert ready_url(lines[connect[0]]) == lines[connect[0]].rsplit(" ", 1)[-1]
