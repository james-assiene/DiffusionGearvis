"""``astra session``: the production object graph, Ctrl-C, and loud failure (criteria 15, 16, 19).

The CLI is the composition root — the one place allowed to name the concrete clients — so the
fast suite may not let it construct any of them. Two devices for that: ``--dry-run``, which must
construct nothing at all, and ``explode_clients``, which replaces every concrete client with a
class that raises if anybody instantiates it.

Seams (fixed in ``tests/unit/_hud_rig.py``'s docstring):

    cli.main(argv, *, runner=None, probe=None) -> int
    cli.build_session(config, args) -> SessionGraph   # .session .server .frames .describe()
    cli.run_session(graph) -> int                     # serves until interrupted, then closes

Regime: no measurement; nothing here starts a service, opens a device or writes to ``artifacts/``.

T-0111g re-scopes (docs/tasks/T-0111g.md, "Named re-scopes"): R2 dropped the ``"live"`` param of
``test_tools_live_or_junk_on_argv_is_refused_naming_the_non_goal``; R3 deleted
``test_tools_live_in_the_config_is_refused_and_the_session_never_starts`` (``tools = "live"`` is a mode now), replaced
by ``tests/unit/test_cli_tools_live.py::test_no_tool_loop_refused`` and
``tests/unit/test_config_tools_live.py::test_live_with_loop_false_refused``.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import atexit
import contextlib
import os
import re
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from local_astra import cli
from tests.unit._mic_rig import (
    BOX_MIC,
    ENV_MIC,
    ENV_SOURCE,
    RIG_MIC,
    RIG_SINK,
    Recorded,
    env_source,
    pin_default_source,
    record_open_device,
    stub_default_source,
)

REPO = Path(__file__).resolve().parents[2]
RUNG = REPO / "scripts" / "demo" / "session_rung.py"
CLI = REPO / "local_astra" / "cli.py"

#: The concrete clients ``astra session`` wires, and where they are defined. Patching both the
#: defining module and the ``cli`` attribute covers a top-level ``from … import X`` and a
#: function-local one, so the test does not depend on where the implementer puts the import.
CONCRETE = {
    "VllmClient": "local_astra.conductor.llm_client",
    "MoshiSttClient": "local_astra.conductor.stt_client",
    "PocketTtsClient": "local_astra.conductor.tts_client",
    "PipeWireAudioIO": "local_astra.conductor.audio_io",
    "SqliteMemory": "local_astra.memory.store",
    "BrowserFrameSource": "local_astra.perception.browser_source",
}

#: Flags that decide the *regime* a turn runs in — the ones a line in ``turns.jsonl`` has to be
#: able to state (CLAUDE.md: a number without its regime cannot be compared). ``--image`` and
#: ``--listen`` are the rung's own input modes and ``--timeout-s`` is its harness, so they are not
#: in the parity set; if the reviewer disagrees, this tuple is the one place to change.
REGIME_FLAGS = ("--config", "--data-dir", "--sink", "--audio-path")


@pytest.fixture(autouse=True)
def _no_ambient_capture_source(monkeypatch):
    """T-0045 ruling A16: no row here may inherit the box's shell for its capture source.

    ``ASTRA_AUDIO_SOURCE`` decides which microphone a session opens (ruling A14's precedence), and
    it is the variable the human was told to export on 2026-09-15 — so a suite that leaves it
    alone measures whatever shell it was started from. Cleared for every row; the rows whose
    subject *is* the variable set it themselves (``env_source``).
    """
    monkeypatch.delenv(ENV_SOURCE, raising=False)


class Exploding:
    """Stands in for a concrete client: constructing one is the failure under test."""

    name = "?"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            f"criterion 15: --dry-run constructs no client and opens no device; "
            f"{type(self).name} was instantiated"
        )


def explode_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every concrete client explode on construction, wherever the CLI reaches it."""
    assert CLI.is_file()
    body = CLI.read_text()
    named = [name for name in CONCRETE if name in body]
    assert named, (
        "criterion 15: local_astra/cli.py is the composition root and must name the concrete "
        f"clients; it names none of {sorted(CONCRETE)} — this test would pass vacuously"
    )
    for name, module in CONCRETE.items():
        bomb = type(name, (Exploding,), {"name": name})
        monkeypatch.setattr(f"{module}.{name}", bomb, raising=False)
        if hasattr(cli, name):
            monkeypatch.setattr(cli, name, bomb, raising=False)
    monkeypatch.setattr("local_astra.conductor.audio_device.open_device", _no_device, raising=False)
    if hasattr(cli, "open_device"):
        monkeypatch.setattr(cli, "open_device", _no_device, raising=False)


def _no_device(*args: object, **kwargs: object) -> object:
    raise AssertionError("criterion 15: --dry-run opens no audio device")


def run_cli(argv: list[str], **kwargs: object) -> int:
    """``cli.main`` with argparse's ``SystemExit`` turned into the exit code it carries."""
    try:
        return int(cli.main(argv, **kwargs))  # type: ignore[arg-type]
    except SystemExit as exit_code:  # argparse: an unknown verb or a bad flag
        return int(exit_code.code or 0)
    except TypeError as exc:  # an unimplemented keyword seam must not look like a crash
        raise AssertionError(
            f"T-0028: cli.main(argv, *, runner=None, probe=None) — {exc}"
        ) from None


def flags_of(path: Path) -> set[str]:
    """Every ``--flag`` string passed to ``add_argument`` in a module, read statically."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "add_argument":
            continue
        out |= {
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        }
    return out


def test_dry_run_constructs_nothing_and_prints_the_resolved_graph(tmp_path, capsys, monkeypatch):
    """Criterion 15: ``--dry-run`` prints the profile, the endpoints and the graph, exits 0.

    Every client is a bomb, so a graph built here fails loudly; and the printed endpoints must be
    the ones ``config.Services`` resolved (loopback-validated, docs/02 §9), which is checked by
    looking for the vLLM endpoint of the ``lean`` profile in the output.
    """
    explode_clients(monkeypatch)
    # T-0045 ruling A16: the box's default capture source is *pinned*, never read off whatever
    # PipeWire graph the runner happens to have. Unpinned, this row is red on a headless runner or
    # in a container: `astra session` refuses to start deaf (ruling A3) before it reaches what
    # this row is about, so the failure names the microphone instead of the subject.
    pin_default_source(monkeypatch, BOX_MIC)
    from local_astra.config import load

    services = load(None).active().services

    code = run_cli(["session", "--profile", "lean", "--dry-run", "--data-dir", str(tmp_path)])
    printed = capsys.readouterr()
    out = printed.out + printed.err
    assert code == 0, f"criterion 15: --dry-run exits 0; it exited {code}. Output:\n{out}"
    assert "lean" in out, f"the resolved profile is printed; got:\n{out}"
    assert services.vllm in out and services.tts in out, (
        f"criterion 15: the resolved endpoints are printed and they come from config.Services "
        f"({services.vllm}, {services.tts}); got:\n{out}"
    )
    assert "/v1/realtime" in out, (
        f"…and the page URL the endpoint will serve (criterion 1); got:\n{out}"
    )
    assert not list(tmp_path.iterdir()), (
        f"--dry-run writes nothing either; it left {list(tmp_path.iterdir())}"
    )


def test_the_session_flags_are_a_superset_of_the_rungs_regime_flags():
    """Criterion 15: the two ways of starting a session cannot disagree about the regime.

    A read-only parity pin against ``scripts/demo/session_rung.py`` (T-0007's file, untouched
    here): every regime flag it accepts, ``astra session`` accepts too — otherwise a turn started
    from the CLI cannot even describe the run the rung could.
    """
    rung_flags = flags_of(RUNG)
    assert set(REGIME_FLAGS) <= rung_flags, (
        f"the parity pin is stale: the rung no longer accepts {set(REGIME_FLAGS) - rung_flags}. "
        "Fix REGIME_FLAGS here (and say so to the reviewer), do not weaken the assertion below."
    )
    cli_flags = flags_of(CLI)
    missing = set(REGIME_FLAGS) - cli_flags
    assert not missing, (
        f"criterion 15: `astra session` must accept the rung's regime flags too; it is missing "
        f"{sorted(missing)} (it has {sorted(cli_flags)})"
    )
    assert {"--profile", "--turns", "--source", "--dry-run"} <= cli_flags, (
        f"criterion 15 names these flags explicitly; the CLI has {sorted(cli_flags)}"
    )


def test_the_cli_never_accepts_a_url_argument():
    """Criterion 15, negative: every endpoint comes from ``config.Services``, never from argv.

    A URL on the command line bypasses the loopback validator — which is the only thing standing
    between this process and a non-loopback service (docs/02 §9).
    """
    urlish = [flag for flag in flags_of(CLI) if "url" in flag or "endpoint" in flag]
    assert not urlish, f"criterion 15: the CLI never accepts a URL argument; it accepts {urlish}"
    assert "add_argument" in CLI.read_text(), "the static scan must have something to read"
    assert "session" in CLI.read_text(), "…and the verb it is about"


def test_a_busy_port_exits_non_zero_naming_the_port(tmp_path, capsys, monkeypatch):
    """Criterion 19: a port already in use is loud, and the CLI does not quietly move.

    Silently moving the endpoint is how the page ends up talking to nothing: ``hud/src/ws.js``
    defaults to the configured port, so a session that fell back to another one is invisible.
    """
    # T-0045 ruling A16: the box's default capture source is *pinned*, never read off whatever
    # PipeWire graph the runner happens to have. Unpinned, this row is red on a headless runner or
    # in a container: `astra session` refuses to start deaf (ruling A3) before it reaches what
    # this row is about, so the failure names the microphone instead of the subject.
    pin_default_source(monkeypatch, BOX_MIC)
    monkeypatch.setattr(cli, "build_session", _fake_graph_factory(tmp_path), raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = squatter.getsockname()[1]
        config = tmp_path / "astra.toml"
        config.write_text(f"profile = 'lean'\n[profiles.lean.services]\nconductor_port = {port}\n")
        code = run_cli(["--config", str(config), "session", "--data-dir", str(tmp_path)])
        out = capsys.readouterr()
        message = out.out + out.err
    assert code != 0, f"criterion 19: a busy port exits non-zero; it exited {code}"
    assert str(port) in message, f"…with the cause named — the port. It said:\n{message}"
    assert "fall" not in message.lower() and "instead" not in message.lower(), (
        f"criterion 19, negative: no fallback to another port; it said:\n{message}"
    )


def free_port() -> int:
    """A port nothing is listening on right now (ruling T-0039 A16).

    The rows below used to bind the profile default (8000); two runs of this module on one box
    then collided and the SIGINT row failed with ``fired == []``. Bind-to-zero-and-release is the
    same seam ``test_a_silent_session_opens_no_microphone_and_says_so`` already uses.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def config_on_free_ports(tmp_path: Path) -> tuple[Path, int, int]:
    """An ``astra.toml`` whose conductor and HUD ports are this run's alone (A16)."""
    conductor, hud = free_port(), free_port()
    path = tmp_path / "astra.toml"
    path.write_text(
        "profile = 'lean'\n[profiles.lean.services]\n"
        f"conductor_port = {conductor}\nhud_port = {hud}\n"
    )
    return path, conductor, hud


def port_is_free(port: int) -> bool:
    """True if nothing is listening: the check criterion 16 makes about the way out."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _fake_graph_factory(tmp_path: Path):
    """``build_session`` over the fakes: a real ``Session`` and a real ``HudServer``, no clients."""

    def build(config: object, args: object) -> object:
        from tests.unit._hud_rig import make_rig, require_symbol

        hud_server = require_symbol("HudServer")
        rig = make_rig(tmp_path)
        port = getattr(getattr(config, "active", lambda: None)(), "services", None)
        server = hud_server(
            rig.session,
            frame_sink=_Sink(),
            port=port.conductor_port if port is not None else 0,
        )
        return _Graph(session=rig.session, server=server, rig=rig)

    return build


class _Sink:
    def ingest(self, message: object) -> None:
        return None


class _Graph:
    """The shape ``cli.run_session`` consumes: a session, a server, and a description."""

    def __init__(self, session: object, server: object, rig: object) -> None:
        self.session = session
        self.server = server
        self.frames = _Sink()
        self.rig = rig

    def describe(self) -> str:
        return "fake graph"


def test_sigint_closes_the_session_and_writes_no_extra_line(tmp_path, monkeypatch):
    """Criterion 16: Ctrl-C closes the session, stops the server, exits 0, writes no line.

    Records belong to turn-ending rows only (T-0007 criterion 12): a line written at shutdown
    would claim a turn that never ended. Nothing here ran a turn, so ``turns.jsonl`` must not
    exist at all when the process is interrupted.

    **The signal is fired from inside the loop, not off a wall clock** (T-0031 finding, one
    failure in five parallel runs). A ``threading.Timer(0.5, …)`` assumes the CLI has reached
    ``_serve``'s ``StopSignals.arm`` — the ``signal.signal`` call that takes SIGINT over
    (T-0040) — within half a second of real time; under ``-n 8`` that is not true — ``build_session``
    constructs the whole graph first, and ``Session._build_regime`` forks ``git rev-parse HEAD`` on
    the way — so the SIGINT lands on the *default* handler and a ``KeyboardInterrupt`` is raised at
    an arbitrary point instead.
    Hooking ``server.start`` is the same instant Ctrl-C means in the criterion — the endpoint is
    up and the loop is serving — and it carries no timing assumption at all. The timer that
    remains is a backstop so a CLI that never serves fails loudly instead of hanging the suite;
    the assertions below are unchanged, and one of them is that the backstop did *not* fire.
    """
    graph_factory = _fake_graph_factory(tmp_path)
    built: list[object] = []
    # T-0045 ruling A16: the box's default capture source is *pinned*, never read off whatever
    # PipeWire graph the runner happens to have. Unpinned, this row is red on a headless runner or
    # in a container: `astra session` refuses to start deaf (ruling A3) before it reaches what
    # this row is about, so the failure names the microphone instead of the subject.
    pin_default_source(monkeypatch, BOX_MIC)
    fired: list[str] = []
    lock = threading.Lock()

    def interrupt(reason: str) -> None:
        with lock:
            if fired:
                return
            fired.append(reason)
        os.kill(os.getpid(), signal.SIGINT)

    def build(config: object, args: object) -> object:
        graph = graph_factory(config, args)
        serving = graph.server.start

        async def start_then_interrupt() -> None:
            await serving()
            # inside the loop, endpoint bound, signal handler installed: this is "Ctrl-C now"
            asyncio.get_running_loop().call_soon(interrupt, "serving")

        graph.server.start = start_then_interrupt
        built.append(graph)
        return graph

    monkeypatch.setattr(cli, "build_session", build, raising=False)
    # Ruling T-0039 A16: this row binds the real conductor endpoint, so it takes a port of its
    # own through the config seam. On the profile default (8000) two concurrent runs of this
    # module collide and the row fails with ``fired == []``.
    config_path, conductor_port, _hud_port = config_on_free_ports(tmp_path)
    backstop = threading.Timer(20.0, interrupt, args=("backstop",))
    backstop.start()
    try:
        code = run_cli(["--config", str(config_path), "session", "--data-dir", str(tmp_path)])
    except KeyboardInterrupt:
        raise AssertionError(
            "criterion 16: the CLI must handle SIGINT and exit 0; the KeyboardInterrupt escaped "
            "main() instead"
        ) from None
    finally:
        backstop.cancel()

    assert fired == ["serving"], (
        "the interrupt must be the one fired once the endpoint was serving; "
        f"what fired was {fired} (a 'backstop' entry means the CLI never started serving)"
    )
    assert code == 0, f"criterion 16: SIGINT exits 0; it exited {code}"
    assert built, "the CLI must have built its graph (the patch point is cli.build_session)"
    session = built[0].session  # type: ignore[attr-defined]
    assert session.stt.closed, (
        "criterion 16: the session is closed on the way out (aclose closes the STT client)"
    )
    turns = tmp_path / "turns.jsonl"
    assert not turns.exists() or turns.read_text().strip() == "", (
        "criterion 16, negative: no turns.jsonl line is written at shutdown; it holds "
        f"{turns.read_text()!r}"
    )
    assert port_is_free(conductor_port), (
        f"criterion 16: the endpoint is released on the way out; {conductor_port} is still bound"
    )


def test_a_second_sigint_during_shutdown_is_absorbed(tmp_path, capsys, monkeypatch):
    """T-0040 criterion 1: Ctrl-C twice still exits 0, prints no traceback, frees both ports.

    Ctrl-C in a terminal goes to the whole foreground process group, so ``uv`` and the Python
    child each get one and ``uv`` forwards its copy: the session sees two. The first is the one
    criterion 16 handles; the second lands while the CLI is already inside
    ``static.close()`` -> ``HTTPServer.shutdown()`` -> ``Event.wait()`` and — today — escapes as a
    ``KeyboardInterrupt`` traceback with exit 130 (reproduced live, logs/t0039-live-82d8462/run.log).

    Both signals are fired from inside the shutdown path rather than off a wall clock, for the
    reason the single-SIGINT row above gives: the first once the endpoint is serving, the second
    from the first statement of the static server's ``close()`` — the exact window the live
    traceback came from. What is asserted is the criterion: exit 0, no ``KeyboardInterrupt``
    anywhere, nothing left listening on either port.
    """
    graph_factory = _fake_graph_factory(tmp_path)
    built: list[object] = []
    fired: list[str] = []
    lock = threading.Lock()

    def interrupt(reason: str) -> None:
        with lock:
            if reason in fired:
                return
            fired.append(reason)
        os.kill(os.getpid(), signal.SIGINT)

    def build(config: object, args: object) -> object:
        graph = graph_factory(config, args)
        serving = graph.server.start

        async def start_then_interrupt() -> None:
            await serving()
            asyncio.get_running_loop().call_soon(interrupt, "serving")

        graph.server.start = start_then_interrupt
        built.append(graph)
        return graph

    real_static = cli.hud_static.HudStaticServer

    class SecondCtrlC(real_static):  # type: ignore[valid-type, misc]
        """The real static server, interrupted the instant the CLI starts closing it."""

        def close(self) -> None:
            interrupt("closing")
            super().close()

    monkeypatch.setattr(cli, "build_session", build, raising=False)
    monkeypatch.setattr(cli.hud_static, "HudStaticServer", SecondCtrlC)
    config_path, conductor_port, hud_port = config_on_free_ports(tmp_path)
    backstop = threading.Timer(20.0, interrupt, args=("backstop",))
    backstop.start()
    try:
        code = run_cli(
            [
                "--config",
                str(config_path),
                "session",
                "--silent",
                "--serve-hud",
                "--data-dir",
                str(tmp_path),
            ]
        )
    except KeyboardInterrupt:
        printed = capsys.readouterr()
        raise AssertionError(
            "T-0040 criterion 1: the second SIGINT must be absorbed; a KeyboardInterrupt escaped "
            f"main() instead (this is the live traceback). stdout so far:\n{printed.out}"
        ) from None
    finally:
        backstop.cancel()

    assert fired == ["serving", "closing"], (
        "both interrupts must be the fired-from-inside ones: the first once the endpoint was "
        f"serving, the second while the static server was closing; what fired was {fired} "
        "('backstop' means the CLI never got that far)"
    )
    assert code == 0, f"T-0040 criterion 1: two SIGINTs still exit 0; it exited {code}"
    assert built, "the CLI must have built its graph (the patch point is cli.build_session)"
    session = built[0].session  # type: ignore[attr-defined]
    assert session.stt.closed, (
        "criterion 16 still holds under the second signal: the session is closed on the way out"
    )
    assert port_is_free(conductor_port), (
        f"T-0040 criterion 1: the conductor endpoint is unbound after the way out; "
        f"{conductor_port} is still listening"
    )
    assert port_is_free(hud_port), (
        f"T-0040 criterion 1: the static server is unbound too; {hud_port} is still listening"
    )
    turns = tmp_path / "turns.jsonl"
    assert not turns.exists() or turns.read_text().strip() == "", (
        "criterion 16, negative: no turns.jsonl line is written at shutdown"
    )


# --- ruling A8: the ignore is the CLI's, and must not leak into an in-process caller ------------


#: What a fresh interpreter — and so a pytest worker, and any embedder that never touched the
#: signals — has installed for the two stop signals. This is the pre-call state ruling A6 refuses
#: to restore, and therefore the only one in which ruling A8's leak is visible at all: a row that
#: recorded whatever it happened to find (``SIG_IGN``, once an earlier row has leaked one) would
#: assert that ``SIG_IGN`` was restored and pass vacuously.
FRESH_DISPOSITIONS: dict[signal.Signals, object] = {
    signal.SIGINT: signal.default_int_handler,
    signal.SIGTERM: signal.SIG_DFL,
}


def dispositions() -> dict[signal.Signals, object]:
    """What is installed right now for the two stop signals."""
    return {sig: signal.getsignal(sig) for sig in FRESH_DISPOSITIONS}


def install(state: dict[signal.Signals, object]) -> None:
    for sig, handler in state.items():
        signal.signal(sig, handler)  # type: ignore[arg-type]


def describe(handler: object) -> str:
    """``SIG_IGN`` / ``SIG_DFL`` / ``default_int_handler`` read back as their names."""
    known = {
        signal.SIG_IGN: "SIG_IGN",
        signal.SIG_DFL: "SIG_DFL",
        signal.default_int_handler: "default_int_handler",
    }
    for value, name in known.items():
        if handler is value:
            return name
    return repr(handler)


def _run_an_interrupted_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, through_sys_argv: bool = False
) -> None:
    """Serve a session over the fakes, Ctrl-C it once it is serving, assert it exited 0.

    The same seam as ``test_sigint_closes_the_session_and_writes_no_extra_line`` — the SIGINT is
    fired from inside the loop the instant the endpoint is up, so nothing here depends on a wall
    clock — reduced to what the rows below need: a session that has *serviced a stop signal* and
    returned. ``through_sys_argv`` is the difference between the two ways ``main`` is entered:
    ``cli.main([...])`` (an in-process caller) and ``cli.main()`` with ``sys.argv`` set (what the
    ``astra`` console script does).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    graph_factory = _fake_graph_factory(tmp_path)
    # T-0045 ruling A16: the box's default capture source is *pinned*, never read off whatever
    # PipeWire graph the runner happens to have. Unpinned, this row is red on a headless runner or
    # in a container: `astra session` refuses to start deaf (ruling A3) before it reaches what
    # this row is about, so the failure names the microphone instead of the subject.
    pin_default_source(monkeypatch, BOX_MIC)
    fired: list[str] = []
    lock = threading.Lock()

    def interrupt(reason: str) -> None:
        with lock:
            if fired:
                return
            fired.append(reason)
        os.kill(os.getpid(), signal.SIGINT)

    def build(config: object, args: object) -> object:
        graph = graph_factory(config, args)
        serving = graph.server.start

        async def start_then_interrupt() -> None:
            await serving()
            asyncio.get_running_loop().call_soon(interrupt, "serving")

        graph.server.start = start_then_interrupt
        return graph

    monkeypatch.setattr(cli, "build_session", build, raising=False)
    config_path, _conductor_port, _hud_port = config_on_free_ports(tmp_path)
    argv = ["--config", str(config_path), "session", "--data-dir", str(tmp_path)]
    backstop = threading.Timer(20.0, interrupt, args=("backstop",))
    backstop.start()
    try:
        if through_sys_argv:
            monkeypatch.setattr(sys, "argv", ["astra", *argv])
            code = int(cli.main())
        else:
            code = run_cli(argv)
    except KeyboardInterrupt:  # pragma: no cover - criterion 16 has its own row for this
        raise AssertionError(
            "criterion 16: the CLI handles the SIGINT and exits 0; it escaped main() instead"
        ) from None
    finally:
        backstop.cancel()
    assert fired == ["serving"], (
        "the scenario must interrupt the session from inside the serving loop; what fired was "
        f"{fired} (a 'backstop' entry means the CLI never started serving, so nothing below is "
        "about a serviced stop signal)"
    )
    assert code == 0, f"criterion 16: the interrupted session exits 0; it exited {code}"


def test_an_interrupted_in_process_session_restores_both_dispositions(tmp_path, monkeypatch):
    """Ruling A8: after an in-process ``cli.main([… session …])`` the signals are the caller's.

    Ruling A6 keeps ``SIG_IGN`` installed through interpreter exit when the previous disposition
    was the interpreter's own default — right for the console entry point, where the duplicates
    ``uv`` forwards keep arriving during finalisation and nobody is left to interrupt; wrong for a
    caller that passed an explicit argv (a test, an embedder), which goes on living deaf. Ruling
    A8: an explicit argv means the CLI does not own the process, so both dispositions are restored
    (ruling A3's full restore), whatever they were.

    Both signals, not just SIGINT: A7's note — a caller that had a handler for only one of the two
    keeps that one while the other stays ignored — is the shape of the leak, and SIGTERM is the one
    that makes children unkillable (next row).
    """
    entry = dispositions()  # whatever this worker is in right now, restored below
    try:
        install(FRESH_DISPOSITIONS)
        before = dispositions()
        _run_an_interrupted_session(tmp_path, monkeypatch)
        after = dispositions()
    finally:
        install(entry)
    leaked = {
        sig.name: f"{describe(before[sig])} -> {describe(after[sig])}"
        for sig in before
        if after[sig] is not before[sig]
    }
    assert not leaked, (
        "T-0040 ruling A8: an interrupted in-process session restores both dispositions; these "
        f"came back changed: {leaked}. SIG_IGN is inherited across fork+exec, so the caller and "
        "every child it spawns afterwards are deaf for the rest of their lives"
    )


def test_children_spawned_after_an_interrupted_session_still_die_on_sigterm(tmp_path, monkeypatch):
    """Ruling A8, the consequence the Stop hook found: a child spawned afterwards is killable.

    An ignored disposition survives ``fork``+``exec``, so a worker left with SIGTERM at
    ``SIG_IGN`` hands that ignore to every process it starts: in
    ``tests/unit/test_cli_down_status.py`` the ``sleep 30`` victims then outlive the SIGTERM
    ``astra down`` sends them and three rows go red whenever this module ran first in the same
    worker (reproduced: session-then-down 3 failed, down-then-session all passed).

    Nothing but this row's own ``sleep`` child is signalled, and it is reaped with SIGKILL — which
    cannot be ignored — whatever the outcome.
    """
    from tests.unit._box_rig import reap

    entry = dispositions()
    child: subprocess.Popen[bytes] | None = None
    status: int | None = None
    try:
        install(FRESH_DISPOSITIONS)
        _run_an_interrupted_session(tmp_path, monkeypatch)
        child = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        child.send_signal(signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            status = child.wait(timeout=2.0)
    finally:
        install(entry)
        reap(child)
    assert status is not None, (
        "T-0040 ruling A8: a `sleep 30` spawned after an interrupted in-process session must die "
        "on SIGTERM; it was still running 2 s later, because it inherited the SIG_IGN the session "
        "left behind (this is `astra down` failing to stop the services it started)"
    )
    assert status == -signal.SIGTERM, (
        "…and it must die *of* the SIGTERM (a child killed by signal 15 reports -15); "
        f"it exited with {status}"
    )


def test_the_process_owning_mode_is_argv_none(tmp_path, monkeypatch):
    """Ruling A8: "keep the ignore through exit" is the console entry point's mode alone.

    One scenario, run twice, told apart only by how ``main`` was entered. ``cli.main([...])`` is an
    in-process caller: both dispositions come back (the row above) and no ``atexit`` hook is left
    behind. ``cli.main()`` with ``sys.argv`` set is what the ``astra`` console script does — the
    CLI owns the process — and there ruling A6/A7 stands: the ignore stays installed through
    interpreter exit, because ``uv``'s forwarded copy of the signal (150–350 ms late) and a human
    holding Ctrl-C keep arriving while CPython joins the non-daemon threads, which it does *before*
    the ``atexit`` hooks run. That second half is what keeps the A8 fix from being "delete A6".

    Limitation, stated rather than guessed: ``atexit`` exposes no listing of its callbacks, so the
    hook is pinned one-sidedly through CPython's ``atexit._ncallbacks()`` — the in-process half
    must register nothing new. Whether the process-owning half registers one is deliberately not
    asserted: the hook is belt-and-braces (ruling A6) and A8 forbids it only in the other mode.
    """
    count = getattr(atexit, "_ncallbacks", None)
    hook = getattr(cli, "_ignore_stop_signals", None)
    entry = dispositions()
    try:
        if hook is not None:  # normalise: an earlier row in this worker may have registered it
            atexit.unregister(hook)
        install(FRESH_DISPOSITIONS)
        registered_before = count() if count is not None else None
        _run_an_interrupted_session(tmp_path / "in-process", monkeypatch)
        after_in_process = dispositions()
        registered_after = count() if count is not None else None

        install(FRESH_DISPOSITIONS)
        _run_an_interrupted_session(tmp_path / "console", monkeypatch, through_sys_argv=True)
        after_console = dispositions()
    finally:
        install(entry)

    restored = {
        sig.name: describe(after_in_process[sig])
        for sig in FRESH_DISPOSITIONS
        if after_in_process[sig] is not FRESH_DISPOSITIONS[sig]
    }
    assert not restored, (
        "T-0040 ruling A8: `cli.main(argv)` with an explicit argv is not the process-owning mode, "
        f"so it restores both dispositions; it left {restored}"
    )
    if registered_before is not None and registered_after is not None:
        assert registered_after == registered_before, (
            "T-0040 ruling A8: the atexit hook that re-deafens the process at exit belongs to the "
            "process-owning mode only; an explicit-argv call registered "
            f"{registered_after - registered_before} new atexit callback(s)"
        )
    kept = {
        sig.name: describe(after_console[sig])
        for sig in FRESH_DISPOSITIONS
        if after_console[sig] is not signal.SIG_IGN
    }
    assert not kept, (
        "T-0040 rulings A6/A7: `cli.main()` (argv=None, sys.argv — the console entry point) owns "
        "the process, so a serviced stop leaves the stop signals ignored through interpreter exit "
        f"rather than restoring the interpreter default; these were put back: {kept}. Restoring "
        "them here is what printed `Exception ignored in threading._shutdown: KeyboardInterrupt` "
        "on ~40 % of live runs at b3fae52"
    )


def test_run_session_returns_zero_when_the_serve_loop_is_cancelled(tmp_path):
    """Criterion 16, the seam itself: ``run_session`` closes both halves and returns 0.

    Driven without a signal so the assertion is about the shutdown *path*, not about signal
    delivery: cancelling the serve loop is what SIGINT does to it.
    """
    graph = _fake_graph_factory(tmp_path)(None, argparse.Namespace())

    async def drive() -> int:
        task = asyncio.ensure_future(cli.run_session(graph))  # type: ignore[attr-defined]
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            return int(await task)
        except asyncio.CancelledError:
            return 0

    code = asyncio.run(drive())
    assert code == 0, f"criterion 16: run_session returns 0 after a clean stop; got {code}"
    assert graph.session.stt.closed, (  # type: ignore[attr-defined]
        "…and it closed the session on the way out"
    )


def _explode_device(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    """Make ``open_device`` raise, wherever the CLI reaches it, and leave the rest constructible."""

    def boom(*args: object, **kwargs: object) -> object:
        raise exc

    monkeypatch.setattr("local_astra.conductor.audio_device.open_device", boom, raising=False)
    if hasattr(cli, "open_device"):
        monkeypatch.setattr(cli, "open_device", boom, raising=False)


def test_a_missing_portaudio_exits_non_zero_naming_the_device(tmp_path, capsys, monkeypatch):
    """Criterion 19 / ruling A18(4): the device that could not be opened is named.

    A missing PortAudio is the failure this box actually hits, and the criterion's negative is
    that the CLI does not quietly pick a different device — a session on a device nobody chose is
    a recording whose regime is a guess (docs/01 §8: headset and speakers are not comparable).
    """
    # T-0045 ruling A16: the box's default capture source is *pinned*, never read off whatever
    # PipeWire graph the runner happens to have. Unpinned, this row is red on a headless runner or
    # in a container: `astra session` refuses to start deaf (ruling A3) before it reaches what
    # this row is about, so the failure names the microphone instead of the subject.
    pin_default_source(monkeypatch, BOX_MIC)
    _explode_device(monkeypatch, RuntimeError("PortAudio library not found"))
    code = run_cli(
        [
            "session",
            "--sink",
            "alsa_output.pci-0000_00_1f.3.analog-stereo",
            "--data-dir",
            str(tmp_path),
        ]
    )
    printed = capsys.readouterr()
    message = printed.out + printed.err
    assert code != 0, f"criterion 19: a device that cannot be opened exits non-zero; got {code}"
    assert "alsa_output.pci-0000_00_1f.3.analog-stereo" in message, (
        f"…naming the device it was asked for; it said:\n{message}"
    )
    assert "PortAudio library not found" in message, (
        f"…and the cause it was given, not a swallowed one; it said:\n{message}"
    )
    assert "default" not in message.lower().replace("the default sink", ""), (
        "criterion 19, negative: no fallback to another device is announced; it said:\n" + message
    )


def test_a_bad_profile_is_not_reported_as_an_audio_device_failure(tmp_path, capsys, monkeypatch):
    """Ruling A18(4): a profile that does not exist is a *configuration* error, and says so.

    ``session_cmd``'s ``except`` wrapping the whole of ``build_session`` turns every startup
    failure into "cannot open the audio device", so the one message the user acts on points at
    the wrong thing. Nothing about the audio device is true here: it was never reached.
    """
    # If the profile lookup were to fall through to the device, this makes it loud rather than
    # letting the row pass because PortAudio happens to be absent on the machine anyway.
    _explode_device(monkeypatch, RuntimeError("PortAudio library not found"))
    try:
        code = run_cli(["session", "--profile", "no-such-profile", "--data-dir", str(tmp_path)])
    except Exception as exc:  # a traceback is not "loud with the cause named"
        raise AssertionError(
            "criterion 19: an unknown profile must exit non-zero with the cause named, not raise "
            f"{type(exc).__name__}: {exc}"
        ) from None
    printed = capsys.readouterr()
    message = printed.out + printed.err
    assert code != 0, f"an unknown profile exits non-zero; got {code}"
    assert "no-such-profile" in message, (
        f"…naming the profile that does not exist; it said:\n{message}"
    )
    for wrong in ("audio device", "portaudio", "sink"):
        assert wrong not in message.lower(), (
            f"ruling A18(4): this is a configuration error, not an audio one; the message blames "
            f"{wrong!r}:\n{message}"
        )


# --- T-0032 criterion 9: the two defects that made the microphone wiring invisible --------------

#: The capture seams T-0021 shipped, on the concrete class only (ruling A1 kept them off the
#: ``AudioIO`` protocol). ``stt_frames`` is the name on ``audio_capture.CaptureStream`` — *not* on
#: the audio object the session is handed, which is the whole defect.
CAPTURE_SEAMS = ("stt_blocks", "capture_stats", "close_capture")


class _AudioWithSeams:
    """A real fake audio object whose *capture* seam names are chosen by the test.

    Everything else delegates, so the session/CLI see a working playback object; only the four
    capture names are under the test's control.
    """

    SEAM_NAMES = ("stt_frames", "stt_blocks", "capture_stats", "close_capture")

    def __init__(self, inner: object, seams: tuple[str, ...]) -> None:
        self._inner = inner
        self._seams = set(seams)

    def __getattr__(self, name: str) -> object:
        if name in self.SEAM_NAMES:
            if name not in self._seams:
                raise AttributeError(name)
            if name == "stt_frames":  # the name that does not exist on the real class
                return lambda: None
        return getattr(self._inner, name)


def _serve_briefly(graph: object) -> int:
    """Run ``cli.run_session`` until it is cancelled — the shutdown path SIGINT takes."""

    async def drive() -> int:
        task = asyncio.ensure_future(cli.run_session(graph))  # type: ignore[attr-defined]
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            return int(await task)
        except asyncio.CancelledError:
            return 0

    return asyncio.run(drive())


def _hasattr_names(path: Path) -> set[str]:
    """Every string literal the module passes to ``hasattr`` — read statically, not guessed."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "hasattr":
            out |= {
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            }
    return out


def test_the_wired_check_tests_the_real_capture_seam(tmp_path, capsys):
    """Criterion 9(a): "the microphone path is not wired yet" must test a seam that **exists**.

    ``cli.py`` asks ``hasattr(audio, "stt_frames")`` — a name that is on
    ``audio_capture.CaptureStream``, never on the audio object the session holds — so the line
    fires on every real run and would go on firing after T-0032 wires the microphone. The oracle
    below is ``PipeWireAudioIO``'s *actual attribute set*, not a string literal the CLI also uses,
    so renaming the seam breaks this check loudly.

    **Adversarial: the positive case is asserted too.** An audio object that has ``stt_frames``
    and not ``stt_blocks`` is exactly as deaf as one with no capture seam at all, and must be
    reported as unwired — a check that never fires is as useless as one that always does.
    """
    from local_astra.conductor.audio_io import PipeWireAudioIO

    real = set(dir(PipeWireAudioIO))
    assert "stt_frames" not in real, (
        "the premise of criterion 9(a): PipeWireAudioIO has no attribute 'stt_frames' "
        "(audio_capture.CaptureStream does), so hasattr(audio, 'stt_frames') is always False"
    )
    for seam in CAPTURE_SEAMS:
        assert seam in real, f"T-0021 shipped {seam} on PipeWireAudioIO; it is missing"

    checked = _hasattr_names(CLI)
    assert checked, "criterion 9(a): the CLI must still check for the capture seam"
    assert checked <= real, (
        f"criterion 9(a): the CLI tests names that PipeWireAudioIO does not have: "
        f"{sorted(checked - real)}. On the box the check is then always negative and the "
        "microphone wiring is invisible."
    )

    unwired = _fake_graph_factory(tmp_path)(None, argparse.Namespace())
    unwired.session.audio = _AudioWithSeams(unwired.session.audio, ("stt_frames",))
    assert _serve_briefly(unwired) == 0
    said = capsys.readouterr()
    assert "not wired" in (said.out + said.err), (
        "criterion 9(a), the positive case: an audio object with stt_frames but no stt_blocks is "
        f"deaf and must be reported as unwired; the CLI said:\n{said.out + said.err}"
    )

    wired = _fake_graph_factory(tmp_path / "wired")(None, argparse.Namespace())
    wired.session.audio = _AudioWithSeams(wired.session.audio, CAPTURE_SEAMS)
    assert _serve_briefly(wired) == 0
    printed = capsys.readouterr()
    assert "not wired" not in (printed.out + printed.err), (
        "…and an audio object that *has* the seam must not be announced as unwired; the CLI "
        f"said:\n{printed.out + printed.err}"
    )


def test_vllm_takes_its_endpoint_from_the_profile_services(tmp_path, monkeypatch):
    """Criterion 9(b): ``VllmClient()`` ignores ``services.vllm`` — a profile that moves the port.

    ``MoshiSttClient``/``PocketTtsClient`` already read their endpoint off ``Services`` (T-0007a
    ruling A5); the LLM client is constructed with no argument, so a profile pointing at
    ``http://127.0.0.1:9001/v1`` reaches a client still talking to 8001 and the session answers
    from whichever engine happens to be on the default port. *Negative:* the loopback validators
    stay the only gate — a non-loopback endpoint is not constructible through this path.
    """
    # T-0045 ruling A16: the box's default capture source is *pinned*, never read off whatever
    # PipeWire graph the runner happens to have. Unpinned, this row is red on a headless runner or
    # in a container: `astra session` refuses to start deaf (ruling A3) before it reaches what
    # this row is about, so the failure names the microphone instead of the subject.
    pin_default_source(monkeypatch, BOX_MIC)
    from pydantic import ValidationError

    from local_astra.config import load
    from tests.conftest import VirtualClock
    from tests.fakes.audio import FakeAudioIO

    config_path = tmp_path / "astra.toml"
    config_path.write_text(
        "profile = 'lean'\n[profiles.lean.services]\nvllm = 'http://127.0.0.1:9001/v1'\n"
    )
    config = load(config_path)
    assert config.active().services.vllm == "http://127.0.0.1:9001/v1"

    monkeypatch.setattr(cli, "open_audio", lambda args: FakeAudioIO(VirtualClock()), raising=False)
    args = argparse.Namespace(
        profile=None,
        data_dir=str(tmp_path),
        turns=None,
        audio_path="null",
        source="screen",
        sink=None,
    )
    graph = cli.build_session(config, args)
    endpoint = getattr(graph.session.llm, "endpoint", None)
    assert endpoint is not None and endpoint.base_url == "http://127.0.0.1:9001/v1", (
        "criterion 9(b): build_session must construct "
        "VllmClient(VllmEndpoint(base_url=profile.services.vllm)); the client is pointing at "
        f"{getattr(endpoint, 'base_url', endpoint)!r}"
    )

    bad = tmp_path / "bad.toml"
    bad.write_text("profile = 'lean'\n[profiles.lean.services]\nvllm = 'http://10.0.0.5:8001/v1'\n")
    with pytest.raises(ValidationError):
        load(bad)


# --- T-0039 group C: ``astra session --silent`` (criteria 17, 18, 19) ----------------------------
#
# The defect this is aimed at is the ordering one named in criterion 17: ``build_session`` opens
# the PortAudio device *before* ``HudServer`` exists, so on this box (no ``libportaudio2``, P1)
# the command cannot start at all. ``--silent`` installs ``NullAudioDevice`` and must therefore
# reach a bound endpoint with ``sounddevice`` unimportable -- which is how these rows are written:
# the import is made to raise ``OSError`` for real, not mocked away.

SILENT_WHY = "T-0039 criterion 15: local_astra/conductor/null_audio.py provides NullAudioDevice"


@contextlib.contextmanager
def sounddevice_unimportable():
    """``import sounddevice`` raises ``OSError`` -- what this box does without libportaudio2.

    A meta-path finder rather than a ``sys.modules`` injection: ``audio_device.py`` imports the
    library *inside* ``start()``, and only a finder makes that import fail the way a missing
    shared object does.
    """

    class Refuse:
        def find_spec(self, name, path=None, target=None):
            if name == "sounddevice" or name.startswith("sounddevice."):
                raise OSError(
                    "PortAudio library not found (libportaudio2 is not installed on this box)"
                )
            return None

    saved = sys.modules.pop("sounddevice", None)
    finder = Refuse()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        if saved is not None:
            sys.modules["sounddevice"] = saved


def silent_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    """The ``session`` namespace ``build_session``/``describe_graph`` read, with ``--silent`` on."""
    values: dict[str, object] = {
        "profile": None,
        "data_dir": str(tmp_path),
        "turns": None,
        "audio_path": "speakers",
        "source": "screen",
        "sink": None,
        "silent": True,
        "dry_run": False,
        "serve_hud": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def spy_null_device(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every ``NullAudioDevice`` the composition root builds, wherever it reaches it."""
    from tests.fakes.audio_device import attr_or_fail, import_or_fail

    module = import_or_fail("local_astra.conductor.null_audio", SILENT_WHY)
    real = attr_or_fail(module, "NullAudioDevice", SILENT_WHY)
    made: list[object] = []

    class Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(module, "NullAudioDevice", Spy)
    if hasattr(cli, "NullAudioDevice"):
        monkeypatch.setattr(cli, "NullAudioDevice", Spy)
    return made


def test_the_session_parser_has_the_silent_flag():
    """Criterion 17: ``--silent`` is a ``session`` flag, and it is part of the regime.

    ``REGIME_FLAGS`` above is the set the rung and the CLI must agree on; ``--silent`` decides
    what was playing, so it belongs to the same family (criterion 18: no number from a silent run
    can be mistaken for an acoustic one).
    """
    flags = flags_of(CLI)
    assert "--silent" in flags, (
        f"criterion 17: `astra session --silent` is the flag the human runs on a box with no "
        f"PortAudio; the session parser declares {sorted(f for f in flags)}"
    )


def test_dry_run_silent_says_sink_silent_and_constructs_nothing(tmp_path, capsys, monkeypatch):
    """Criterion 18: ``--dry-run --silent`` prints ``sink=silent``; criterion 15 stays honest.

    The whole point is that nobody reading a ``--silent`` regime can mistake it for an acoustic
    one, so the resolved-graph line has to say it where ``sink=`` is read.
    """
    explode_clients(monkeypatch)
    with sounddevice_unimportable():
        code = run_cli(
            ["session", "--silent", "--dry-run", "--data-dir", str(tmp_path)],
        )
    printed = capsys.readouterr()
    out = printed.out + printed.err
    assert code == 0, f"criterion 17: --dry-run --silent exits 0; it exited {code}. Output:\n{out}"
    assert "sink=silent" in out, (
        f"criterion 18: ``describe_graph``'s sink= field reads 'silent' under --silent; the CLI "
        f"printed:\n{out}"
    )
    assert not list(tmp_path.iterdir()), (
        f"--dry-run writes nothing either; it left {list(tmp_path.iterdir())}"
    )


def test_silent_builds_the_graph_with_the_null_device_and_never_opens_one(tmp_path, monkeypatch):
    """Criteria 17 and 18, adversarial: ``sounddevice`` cannot be imported and ``open_device`` is a bomb.

    This is the criterion tested against the real ordering defect rather than against a mock of
    it: if ``build_session`` still reaches the device before the endpoint exists, the import
    raises ``OSError`` and the graph never gets built. The regime half is asserted on the same
    graph: ``Regime.audio_path`` is what every ``turns.jsonl`` line will carry.
    """
    from local_astra.config import load

    made = spy_null_device(monkeypatch)
    _explode_device(monkeypatch, AssertionError("criterion 17: --silent never calls open_device"))
    args = silent_args(tmp_path)
    config = load(None)

    with sounddevice_unimportable():
        graph = cli.build_session(config, args)

    assert made, (
        "criterion 17: --silent builds the graph with NullAudioDevice; the composition root "
        "constructed none"
    )
    assert len(made) == 1, f"one session, one device; it built {len(made)}"
    assert graph.session.config.audio_path == "silent", (
        f"criterion 18: every turns.jsonl line of a silent run says so -- Regime.audio_path is "
        f"'silent', not {graph.session.config.audio_path!r} (a silent run's timings are not "
        f"comparable to an acoustic run's)"
    )
    assert "sink=silent" in graph.describe(), (
        f"criterion 18: the graph description names the sink it is not using; it says:\n"
        f"{graph.describe()}"
    )


def test_without_silent_a_missing_portaudio_is_still_fatal(tmp_path, capsys, monkeypatch):
    """Criterion 17, the other half (T-0006 ruling A1): no silent fallback to a null device.

    The failure mode of adding ``NullAudioDevice`` is that it becomes the *fallback*: a session
    that cannot open the sink the user asked for quietly plays nothing and every number it
    records is a fiction. Without ``--silent``, an unimportable ``sounddevice`` is still
    ``AudioDeviceUnavailable`` and still exit 1.
    """
    # T-0045 ruling A16: the box's default capture source is *pinned*, never read off whatever
    # PipeWire graph the runner happens to have. Unpinned, this row is red on a headless runner or
    # in a container: `astra session` refuses to start deaf (ruling A3) before it reaches what
    # this row is about, so the failure names the microphone instead of the subject.
    pin_default_source(monkeypatch, BOX_MIC)
    with sounddevice_unimportable():
        code = run_cli(["session", "--data-dir", str(tmp_path)])
    printed = capsys.readouterr()
    message = printed.out + printed.err
    assert code == 1, (
        f"criterion 17 / T-0006 ruling A1: without --silent a device that cannot be opened exits "
        f"1; it exited {code}. It said:\n{message}"
    )
    assert "audio device" in message.lower(), (
        f"…naming the failure as an audio device one (AudioDeviceUnavailable); it said:\n{message}"
    )
    assert "silent" not in message.lower(), (
        f"…and never announcing a fallback to a device nobody chose; it said:\n{message}"
    )


def test_a_silent_session_opens_no_microphone_and_says_so(tmp_path, capsys, monkeypatch):
    """Criterion 19: no capture stream, no STT audio, and stdout says what is happening.

    ``--silent`` is one device object for both directions (ambiguity 4, decided yes), so the
    microphone is off too. The line matters because the alternative is a session that looks like
    it is listening and is not.
    """
    from local_astra.config import load

    made = spy_null_device(monkeypatch)
    _explode_device(monkeypatch, AssertionError("criterion 19: --silent never calls open_device"))
    # T-0159 O-A1: the relay's start awaits the piece load; an empty HF cache makes it fail at once
    # (the A5 path) instead of a 355 ms real load outliving this row's 0.15 s serve
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "t0159-empty-hf-cache"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config_path = tmp_path / "astra.toml"
    config_path.write_text(f"profile = 'lean'\n[profiles.lean.services]\nconductor_port = {port}\n")
    config = load(config_path)

    with sounddevice_unimportable():
        # T-0175 amendment: the session log goes under tmp_path, never into the repo's logs/sessions/
        graph = cli.build_session(config, silent_args(tmp_path, log_dir=str(tmp_path / "logs")))
        assert made, "criterion 17: the graph is built on the null device"
        assert _serve_briefly(graph) == 0
    written = sorted((tmp_path / "logs").glob("*.jsonl"))
    assert written, "T-0175: the row's session log is not under its tmp log dir"
    repo_sessions = cli.log_dir_of(argparse.Namespace(log_dir=None))
    assert not any((repo_sessions / p.name).exists() for p in written), (
        f"T-0175: the row wrote its session log into the repo's {repo_sessions}"
    )
    said = capsys.readouterr()
    message = said.out + said.err
    assert "silent" in message.lower(), (
        f"criterion 18/19: the startup line says the session is silent -- no number from it can "
        f"be mistaken for an acoustic one. It said:\n{message}"
    )
    stats = graph.session.audio.capture_stats()
    yielded = getattr(stats, "yielded_ms", getattr(stats, "blocks", 0))
    assert not yielded, (
        f"criterion 19: a --silent session opens no capture stream and feeds the STT no audio; "
        f"the capture stats report {stats}"
    )


# --- T-0045: the microphone `astra session` opens ----------------------------------------------
#
# The defect these rows exist for: ``open_audio`` calls ``open_device(sink=args.sink)`` and never
# passes a ``source``, so ``PortAudioDevice`` opens an output-only stream, ``stt_blocks()`` never
# yields and every live turn records ``stt_frames_fed: 0``. Nothing in the fast suite noticed,
# because every capture row drives the device object directly.
#
# The seam the CLI must use to learn which microphone to open when no flag says
# (T-0045 criterion 1: *read from the box, never guessed*):
#
#     preflight.default_source() -> str | None
#
# the **PipeWire node name** of the default capture source (what ``open_device(source=…)`` and
# ``pw-dump`` call a node — *not* the human-readable description ``wpctl status`` prints), or
# ``None`` when the graph could not be read. The rows below stub it, so no row here asks this box
# anything; ``tests/unit/test_default_source.py`` is where the real one is pinned against the
# recorded graph. The helpers and the node names live in ``tests/unit/_mic_rig.py``.


def run_recorded(argv: list[str]) -> int:
    """``run_cli`` with the recording device's sentinel treated as "the device call happened"."""
    try:
        return run_cli(argv)
    except Recorded:
        return 1


def device_call(calls: list[dict[str, object]], argv: list[str], output: str) -> dict[str, object]:
    """The single ``open_device`` call ``argv`` made, or the assertion that says what happened."""
    assert calls, (
        f"T-0045 criterion 1: `astra {' '.join(argv)}` must reach "
        f"open_device(sink=…, source=…); it called open_device not once. The CLI said:\n{output}"
    )
    assert len(calls) == 1, f"one session, one device; it opened {len(calls)}"
    call = calls[0]
    assert call["args"] == (), (
        f"open_device takes its nodes by keyword (sink=, source=); it was called with positional "
        f"{call['args']!r}"
    )
    keywords = {key: value for key, value in call.items() if not key.startswith(("args", "_"))}
    assert "source" in call, (
        f"T-0045 criterion 1: `astra {' '.join(argv)}` must name the capture source in the "
        f"open_device call — without a `source` PortAudioDevice opens an output-only stream and "
        f"every turn of the session records stt_frames_fed: 0 (the 2026-09-15 defect). It called "
        f"open_device({keywords})"
    )
    return call


def mic_token(output: str) -> str:
    """The ``mic=<node|none>`` token of the resolved-graph line, beside ``sink=`` (criterion 2)."""
    sink_lines = [line for line in output.splitlines() if "sink=" in line]
    assert sink_lines, f"the resolved graph states the sink it plays through; it printed:\n{output}"
    matches = [re.search(r"\bmic=(\S+)", line) for line in sink_lines]
    found = [m.group(1) for m in matches if m]
    assert found, (
        "T-0045 criterion 2: the resolved graph prints `mic=<node name|none>` beside `sink=` — a "
        "session that cannot say which microphone it opened cannot say what heard the turn "
        f"(CLAUDE.md's regime rule). The sink line(s) read {sink_lines!r}"
    )
    return found[0]


def regime_mic(output: str) -> str:
    """The ``mic=`` token of the ``regime:`` line — the one a **real** start prints.

    ``describe_graph``'s ``sink=… mic=…`` line is what ``--dry-run`` prints; a session that is
    actually starting prints the regime line instead (criterion 19 with criterion 2: the regime a
    number is quoted with has to say which microphone heard the turn). ``mic_token`` reads the
    first line, this reads the second.
    """
    lines = [line for line in output.splitlines() if "regime:" in line]
    assert lines, (
        f"criterion 19: a session that starts prints its regime line; it printed:\n{output}"
    )
    match = re.search(r"\bmic=(\S+)", lines[-1])
    assert match, (
        f"T-0045 criterion 2: the regime line carries `mic=<node name|none>` — a turn whose "
        f"numbers cannot say which microphone heard it is not comparable to another's "
        f"(CLAUDE.md's regime rule). The line read {lines[-1]!r}"
    )
    return match.group(1)


def pin_regime_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gather nothing for the regime line: no docker inspect, no nvidia-smi, no engine request.

    A real start gathers those facts; they are not the subject of the microphone rows, and asking
    for them makes a unit row depend on what happens to be running on this box.
    """
    from local_astra import preflight

    monkeypatch.setattr(
        preflight, "gather_regime", lambda *a, **k: preflight.Facts(), raising=False
    )


def test_the_session_parser_has_the_microphone_flags():
    """Criterion 1: the capture source is a flag of ``astra session`` — ``--mic`` / ``--no-mic``.

    ``--source`` is taken: it is the *perception* source (screen/camera), and this row pins that
    too, because the cheap wrong fix is to overload it and make `--source screen` mean a
    microphone named ``screen``.
    """
    flags = flags_of(CLI)
    missing = sorted({"--mic", "--no-mic"} - flags)
    assert not missing, (
        f"criterion 1: `astra session` takes its capture source on the command line "
        f"(--mic <PipeWire node name> / --no-mic); the parser declares no {missing}. Today the "
        f"only route is the ASTRA_AUDIO_SOURCE environment variable (T-0035 criterion 7)."
    )
    from local_astra.contracts.perception import Source

    source_choices = {mode.value for mode in Source}
    assert "--source" in flags and RIG_MIC not in source_choices, (
        "criterion 1: --source keeps meaning the perception source (screen/camera); the "
        f"microphone flag is --mic. Source values: {sorted(source_choices)}"
    )


def test_the_mic_flag_is_the_capture_source_open_device_is_given(tmp_path, capsys, monkeypatch):
    """Criterion 1: ``--mic <node>`` arrives at ``open_device(sink=…, source=<node>)``.

    The value is not interpreted on the way: what the human typed is what the device layer is
    asked for, so ``open_device``'s own refusal (T-0035 criterion 4) is about the node the human
    named and not about something the CLI invented.
    """
    calls = record_open_device(monkeypatch)
    argv = ["session", "--mic", RIG_MIC, "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    call = device_call(calls, argv, output)
    assert call["source"] == RIG_MIC, (
        f"criterion 1: --mic {RIG_MIC} is the capture source; open_device got "
        f"source={call['source']!r}"
    )
    assert call["sink"] == RIG_SINK, (
        f"…and --sink still chooses the playback node; open_device got sink={call['sink']!r}"
    )


def test_with_no_flag_the_microphone_is_the_boxs_default_capture_source(
    tmp_path, capsys, monkeypatch
):
    """Criterion 1: no flag → the *default source the box reports*, not None and not a guess.

    This is the origin defect stated as a test: a session started the way the human starts one
    (``uv run astra session --serve-hud``) opened an output-only stream and recorded
    ``stt_frames_fed: 0`` on every turn.
    """
    asked = stub_default_source(monkeypatch, BOX_MIC)
    calls = record_open_device(monkeypatch)
    argv = ["session", "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    call = device_call(calls, argv, output)
    assert call["source"] == BOX_MIC, (
        f"criterion 1: with no --mic the session opens the box's default capture source "
        f"({BOX_MIC!r}, what preflight.default_source() answered); open_device was given "
        f"source={call['source']!r} — a source of None is the deaf session this task exists for"
    )
    assert asked, (
        "criterion 1: …and it is *read* from the box (preflight.default_source()), never guessed "
        "from a name in the CLI; the collector was never called"
    )


@pytest.mark.parametrize("off", [["--no-mic"], ["--mic", "none"]])
def test_no_mic_opens_the_sink_with_no_capture_source(off, tmp_path, capsys, monkeypatch):
    """Criterion 1: ``--no-mic`` (≡ ``--mic none``) is the only way to an output-only session."""
    pin_default_source(monkeypatch, BOX_MIC)
    calls = record_open_device(monkeypatch)
    argv = ["session", *off, "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    call = device_call(calls, argv, output)
    assert call["source"] is None, (
        f"criterion 1: `astra session {' '.join(off)}` opens no capture source; open_device got "
        f"source={call['source']!r}"
    )
    assert call["sink"] == RIG_SINK, (
        f"…and the playback half is untouched: --sink still decides it; got sink={call['sink']!r}"
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], BOX_MIC),
        (["--mic", RIG_MIC], RIG_MIC),
        (["--no-mic"], "none"),
        (["--mic", "none"], "none"),
    ],
)
def test_dry_run_prints_the_resolved_microphone_beside_the_sink(
    flags, expected, tmp_path, capsys, monkeypatch
):
    """Criteria 2 and 3: ``--dry-run`` prints ``mic=<node|none>`` beside ``sink=``, and builds nothing.

    "Resolved" is the point: the default case prints the node the box reported, not the word
    ``default`` — the human reads this line to find out which microphone is about to listen.
    """
    explode_clients(monkeypatch)
    pin_default_source(monkeypatch, BOX_MIC)
    code = run_cli(["session", *flags, "--dry-run", "--data-dir", str(tmp_path)])
    output = "".join(capsys.readouterr())
    assert code == 0, f"criterion 15: --dry-run exits 0; it exited {code}. Output:\n{output}"
    assert mic_token(output) == expected, (
        f"criterion 3: `astra session {' '.join(flags)} --dry-run` prints the resolved capture "
        f"source; it printed mic={mic_token(output)!r} where {expected!r} was resolved. Output:\n"
        f"{output}"
    )
    assert not list(tmp_path.iterdir()), (
        f"--dry-run writes nothing either; it left {list(tmp_path.iterdir())}"
    )


def test_a_box_that_cannot_say_its_default_microphone_prints_mic_unknown(
    tmp_path, capsys, monkeypatch
):
    """Ruling A7: a fact that was not gathered is ``mic=unknown`` — never ``none``, never a guess.

    ``preflight.default_source()`` answers ``None`` when the graph cannot be read (no PipeWire
    session, no ``pw-dump``: ``tests/unit/test_default_source.py``'s second row). The dry run
    still exits 0 — it starts nothing, so it has nothing to refuse — and says which of the three
    cases it is in the regime line's existing vocabulary: ``none`` means *nobody chose one*
    (``--no-mic``), a node name means *this one will listen*, and ``unknown`` means *this box
    would not say*. Collapsing the third into either of the others is how a deaf session looks
    like a deliberate one.

    Pinned, not read from this box: with no stub the answer is whatever graph the box running the
    suite happens to have, and this row would assert nothing on a developer machine.
    """
    explode_clients(monkeypatch)
    pin_default_source(monkeypatch, None)
    code = run_cli(["session", "--dry-run", "--data-dir", str(tmp_path)])
    output = "".join(capsys.readouterr())
    assert code == 0, (
        f"ruling A7: an ungathered default source is printed, not refused -- `astra session "
        f"--dry-run` starts nothing; it exited {code}:\n{output}"
    )
    assert mic_token(output) == "unknown", (
        f"ruling A7: a box whose graph cannot be read prints mic=unknown beside sink=, which is "
        f"neither --no-mic's `none` nor a guessed node; it printed mic={mic_token(output)!r}:\n"
        f"{output}"
    )


def test_a_box_that_cannot_name_a_microphone_refuses_to_start_rather_than_run_deaf(
    tmp_path, capsys, monkeypatch
):
    """Ruling A3 (with A15), the origin defect: a **real** start with no capture source exits 1.

    The twin of the row above, in the world where it matters. ``--dry-run`` opens nothing, so it
    prints ``mic=unknown`` and exits 0 (ruling A7); dropping ``--dry-run`` is the only path on
    which a deaf session can actually happen, and it is the path the human took on 2026-09-15:
    the session served, looked alive, and every turn recorded ``stt_frames_fed: 0``.

    Three things are asserted, and the first is the one that discriminates: **no device call at
    all**. A CLI that has lost the refusal still exits 1 here — the recording ``open_device``
    raises, ``open_audio`` wraps it as ``AudioDeviceUnavailable`` — so an exit code alone would
    pass while the box opened an output-only stream.

    The fix line is not quoted: it is read off the doctor's own ``audio.source`` row for a box
    with no capture source, so the two places a human meets this state cannot drift apart.
    """
    from tests.unit import _box_rig as box_rig

    doctor_fix = box_rig.by_name(
        box_rig.evaluate(**box_rig.good_box(sources=(), default_source=None))
    )["audio.source"].fix
    assert doctor_fix.strip(), "fixture check: the doctor's no-capture-source row carries a fix"

    # Nothing anywhere names a microphone: no flag, no readable graph, and no exported
    # ASTRA_AUDIO_SOURCE either (ruling A14 puts the variable above the graph default, so a row
    # about "the box cannot name one" has to say that the environment does not name one either).
    env_source(monkeypatch, None)
    pin_default_source(monkeypatch, None)
    calls = record_open_device(monkeypatch)
    argv = ["session", "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    code = run_recorded(argv)
    message = "".join(capsys.readouterr())

    assert calls == [], (
        f"ruling A3: with no capture source to open, `astra {' '.join(argv)}` refuses *before* the "
        f"device — an output-only stream is the deaf session of 2026-09-15. It called open_device "
        f"{len(calls)} time(s): {calls}"
    )
    assert code == 1, (
        f"ruling A3: …and exits 1 (not 0, and not argparse's 2); it exited {code}. It said:\n"
        f"{message}"
    )
    assert doctor_fix in message, (
        f"ruling A3: …carrying the same fix line the doctor's audio.source row prints for a box "
        f"with no capture source ({doctor_fix!r}); it said:\n{message}"
    )
    assert "--no-mic" in message, (
        f"…which is what makes it actionable: the human either plugs a microphone in or asks for "
        f"a session that does not listen. It said:\n{message}"
    )
    for wrong in ("listening on port", "open the HUD page"):
        assert wrong not in message, (
            f"ruling A3, negative: it must not serve anyway; it printed {wrong!r}:\n{message}"
        )


def test_a_microphone_node_that_does_not_exist_exits_1_naming_it(tmp_path, capsys, monkeypatch):
    """Criterion 4, adversarial: the refusal reaches the human; it never runs output-only.

    ``open_device`` refuses an absent node before the stream exists (T-0035 criterion 4). The
    failure this row is aimed at is the CLI swallowing that refusal and carrying on without a
    microphone: a session that serves, looks alive and hears nothing is exactly the 2026-09-15
    evening, and it must instead be a non-zero exit naming the node asked for and the ones that
    exist.
    """
    from local_astra.conductor.audio_device import UnknownNodeError

    absent = "astra-no-such-mic"
    _explode_device(
        monkeypatch,
        UnknownNodeError(
            f"no such PipeWire node: ['{absent}'] -- the graph has sinks ['{RIG_SINK}'] and "
            f"sources ['{BOX_MIC}']"
        ),
    )
    code = run_cli(["session", "--mic", absent, "--data-dir", str(tmp_path)])
    message = "".join(capsys.readouterr())
    assert code == 1, (
        f"criterion 4: a microphone node the graph does not have exits 1; it exited {code}. "
        f"It said:\n{message}"
    )
    assert absent in message, f"…naming the node it was asked for; it said:\n{message}"
    assert BOX_MIC in message, (
        f"…and the capture sources that do exist, which is the line the human acts on; it "
        f"said:\n{message}"
    )
    for wrong in ("listening on port", "open the HUD page"):
        assert wrong not in message, (
            f"criterion 4, negative: it must not serve a deaf session anyway; it printed "
            f"{wrong!r}:\n{message}"
        )


def test_silent_still_opens_no_device_and_no_microphone(tmp_path, capsys, monkeypatch):
    """Criteria 1 and 6, negative: ``--silent`` is unchanged — and it accepts the new flags.

    Two halves, because a criterion-6 negative alone would pass before the feature exists: the
    parser must take ``--silent --no-mic`` (it takes neither microphone flag today), and the
    silent graph must still be built on ``NullAudioDevice`` without a single ``open_device`` call.

    The halves want opposite worlds, so they get one each (T-0045 ruling A6). ``explode_clients``
    makes every concrete client fatal to construct, which is what a ``--dry-run`` must survive and
    what ``build_session`` — whose whole job is constructing them — cannot. The first half runs
    inside its own ``monkeypatch.context()``; by the second half the bombs are gone and only the
    device seams below are patched, exactly as the sibling silent row builds its graph.
    """
    from local_astra.config import load

    with monkeypatch.context() as dry_run_only:
        explode_clients(dry_run_only)
        code = run_cli(
            ["session", "--silent", "--no-mic", "--dry-run", "--data-dir", str(tmp_path)]
        )
        output = "".join(capsys.readouterr())
    assert code == 0, (
        f"criterion 1: `--silent` and the microphone flags are not mutually exclusive; "
        f"`astra session --silent --no-mic --dry-run` exited {code}:\n{output}"
    )
    assert "sink=silent" in output, (
        f"criterion 6: --silent is unchanged — the graph still says sink=silent:\n{output}"
    )

    made = spy_null_device(monkeypatch)
    _explode_device(monkeypatch, AssertionError("criterion 6: --silent never calls open_device"))
    with sounddevice_unimportable():
        graph = cli.build_session(load(None), silent_args(tmp_path))
    assert len(made) == 1, (
        f"criterion 6: a silent session is still one NullAudioDevice and no open_device call; it "
        f"built {len(made)}"
    )
    assert graph.session.config.audio_path == "silent", (
        f"…and its turns still say audio_path=silent; it says {graph.session.config.audio_path!r}"
    )


def test_silent_with_a_microphone_is_a_usage_error(tmp_path, capsys, monkeypatch):
    """Ruling A2 (with A18): ``--silent --mic X`` exits 2 and names both flags; ``--mic none`` is fine.

    ``--silent`` opens no audio device at all, so it cannot also open a microphone: one of the two
    flags is a typo and the human has to be told which two contradict. Argparse's exit 2, not a
    silent precedence rule — a session that quietly ignored ``--mic`` would be a run whose regime
    line and whose command line disagree.

    Second half, from criterion 1 (``--mic none`` ≡ ``--no-mic``) and ruling A2 (``--silent
    --no-mic`` is accepted): ``--silent --mic none`` asks for nothing contradictory and must be
    accepted, so this row cannot be satisfied by refusing every ``--mic`` beside ``--silent``.
    Both halves are dry runs: with the refusal deleted the first would otherwise serve forever.
    """
    pin_default_source(monkeypatch, BOX_MIC)
    code = run_cli(
        ["session", "--silent", "--mic", RIG_MIC, "--dry-run", "--data-dir", str(tmp_path)]
    )
    message = "".join(capsys.readouterr())
    assert code == 2, (
        f"ruling A2: `astra session --silent --mic {RIG_MIC}` is a usage error (argparse exits 2); "
        f"it exited {code} and said:\n{message}"
    )
    assert "--silent" in message and RIG_MIC in message, (
        f"…naming both halves of the contradiction, so the human can see which flag to drop; it "
        f"said:\n{message}"
    )

    code = run_cli(
        ["session", "--silent", "--mic", "none", "--dry-run", "--data-dir", str(tmp_path)]
    )
    accepted = "".join(capsys.readouterr())
    assert code == 0, (
        f"criterion 1 + ruling A2: `--mic none` *is* --no-mic, and `--silent --no-mic` is "
        f"accepted, so `--silent --mic none` asks for nothing contradictory; it exited {code} and "
        f"said:\n{accepted}"
    )
    assert mic_token(accepted) == "none", (
        f"…and it resolves to no microphone; it printed mic={mic_token(accepted)!r}:\n{accepted}"
    )


def test_the_perception_source_is_never_handed_to_the_microphone(tmp_path, capsys, monkeypatch):
    """Criterion 1, adversarial: ``--source camera`` is a screen/camera choice, not a node name.

    The cheapest wrong fix is ``open_device(source=args.source)``: it type-checks, it makes the
    stream duplex, and every session then asks PipeWire for a node called ``screen`` — refused by
    T-0035's graph check, or, on a box where the check is skipped, linked to the default source
    by luck. The microphone comes from ``--mic``/the box's default; ``--source`` never touches it.
    """
    pin_default_source(monkeypatch, BOX_MIC)
    calls = record_open_device(monkeypatch)
    argv = ["session", "--source", "camera", "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    call = device_call(calls, argv, output)
    assert call["source"] == BOX_MIC, (
        f"criterion 1: the perception source must not be read as a capture node; open_device was "
        f"given source={call['source']!r} where the box's default microphone {BOX_MIC!r} was "
        f"resolved"
    )


# --- T-0045 ruling A14: ASTRA_AUDIO_SOURCE is a step of the precedence, not a hidden fallback ---
#
# ``open_device`` consults the variable itself when it is given no source::
#
#     if source is None:
#         source = os.environ.get("ASTRA_AUDIO_SOURCE")      # audio_device.py, unchanged (c. 6)
#
# so while ``resolve_mic`` ignores it, "no capture source" and "the source the environment names"
# are the *same value* at that seam: ``astra session --no-mic`` prints ``mic=none`` on the
# resolved-graph line and the regime line while a duplex stream is open on the environment's node,
# and a box whose graph cannot be read exits 1 although the variable names a microphone that
# exists — a regression on what this task's Goal calls "the only route today". The variable is the
# workaround the human was told to export on 2026-09-15, so this is live, not hypothetical.
#
# Ruled precedence, highest first (ruling A14):
#
#     --mic <node>   >   --no-mic   >   ASTRA_AUDIO_SOURCE   >   the graph's default source
#
# and the variable is *printed* where it is what gets opened, because ``mic=`` is a claim about
# which microphone heard the turn. ``audio_device.py`` stays byte-identical (criterion 6), so the
# off path has to defeat the fallback by clearing the variable around the call — which is what
# ``_env`` (the recorder's snapshot of the environment at the moment of the call) pins.


def test_the_environment_source_outranks_the_graph_default_and_is_printed(
    tmp_path, capsys, monkeypatch
):
    """Ruling A14: with no flag, ``ASTRA_AUDIO_SOURCE`` decides — and the line says which node.

    The symptom without this: someone who exported the rig's monitor gets the webcam, because the
    graph default silently outranks the variable. Both are set here, to different nodes, so the
    row can only pass by choosing the ruled one.
    """
    pin_default_source(monkeypatch, BOX_MIC)
    env_source(monkeypatch, ENV_MIC)
    pin_regime_facts(monkeypatch)
    calls = record_open_device(monkeypatch)
    argv = ["session", "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    call = device_call(calls, argv, output)
    assert call["source"] == ENV_MIC, (
        f"ruling A14: {ENV_SOURCE}={ENV_MIC!r} outranks the graph's default source "
        f"({BOX_MIC!r}); open_device was asked for source={call['source']!r}"
    )
    assert regime_mic(output) == ENV_MIC, (
        f"ruling A14: …and the run's regime names it, because `mic=` is the claim about which "
        f"microphone heard the turn; it printed mic={regime_mic(output)!r}:\n{output}"
    )
    assert BOX_MIC not in output, (
        f"…and the node that was *not* opened ({BOX_MIC!r}) is not printed as if it had been; it "
        f"said:\n{output}"
    )

    # …and the resolved-graph line a human reads *before* starting says the same thing.
    run_cli(["session", "--dry-run", "--sink", RIG_SINK, "--data-dir", str(tmp_path)])
    dry = "".join(capsys.readouterr())
    assert mic_token(dry) == ENV_MIC, (
        f"ruling A14: `--dry-run` prints the microphone the session would open, and the "
        f"environment is the step that decided it; it printed mic={mic_token(dry)!r}:\n{dry}"
    )


def test_the_mic_flag_outranks_the_environment_source(tmp_path, capsys, monkeypatch):
    """Ruling A14, top of the precedence: what the human typed beats what the shell remembers."""
    pin_default_source(monkeypatch, BOX_MIC)
    env_source(monkeypatch, ENV_MIC)
    pin_regime_facts(monkeypatch)
    calls = record_open_device(monkeypatch)
    argv = ["session", "--mic", RIG_MIC, "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    call = device_call(calls, argv, output)
    assert call["source"] == RIG_MIC, (
        f"ruling A14: --mic {RIG_MIC} outranks {ENV_SOURCE}={ENV_MIC!r}; open_device was asked "
        f"for source={call['source']!r}"
    )
    assert regime_mic(output) == RIG_MIC, (
        f"…and the printed microphone is the one that was opened; it printed "
        f"mic={regime_mic(output)!r}:\n{output}"
    )


@pytest.mark.parametrize("off", [["--no-mic"], ["--mic", "none"]])
def test_no_microphone_defeats_the_environment_source(off, tmp_path, capsys, monkeypatch):
    """Ruling A14, the lie: asking for no microphone must *defeat* the fallback, not pass it through.

    Adversarial, and the reason this ruling exists: with the variable exported, ``--no-mic``
    reaches ``open_device(source=None)``, which reads the variable and opens a duplex stream on
    that node — while every line the human reads says ``mic=none``. ``audio_device.py`` may not
    change (criterion 6), so the only way to keep the printed line true is to clear the variable
    around the call; ``_env`` is what the device layer would have seen at that moment.
    """
    pin_default_source(monkeypatch, BOX_MIC)
    env_source(monkeypatch, ENV_MIC)
    pin_regime_facts(monkeypatch)
    calls = record_open_device(monkeypatch)
    argv = ["session", *off, "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    call = device_call(calls, argv, output)
    assert call["source"] is None, (
        f"criterion 1: `astra session {' '.join(off)}` opens no capture source; open_device got "
        f"source={call['source']!r}"
    )
    assert call["_env"] is None, (
        f"ruling A14: …and the environment cannot re-open one behind its back — open_device falls "
        f"back to {ENV_SOURCE} whenever source is None, so with the variable still set "
        f"({call['_env']!r}) this session prints mic=none and listens on {ENV_MIC!r}"
    )
    assert regime_mic(output) == "none", (
        f"…and the run's regime says so; it printed mic={regime_mic(output)!r}:\n{output}"
    )
    assert ENV_MIC not in output, f"…without naming a node it did not open; it said:\n{output}"


def test_an_unreadable_graph_with_the_environment_set_still_opens_that_microphone(
    tmp_path, capsys, monkeypatch
):
    """Ruling A14: the refusal is for a box that can name *no* microphone — the variable names one.

    A box with no readable PipeWire graph and ``ASTRA_AUDIO_SOURCE`` exported is precisely the
    route the Goal calls "the only route today"; refusing it (ruling A3's exit 1) would be a
    regression introduced by this task. The refusal row above is the same box state with the
    variable unset, and the two rows differ in nothing else.
    """
    pin_default_source(monkeypatch, None)
    env_source(monkeypatch, ENV_MIC)
    pin_regime_facts(monkeypatch)
    calls = record_open_device(monkeypatch)
    argv = ["session", "--sink", RIG_SINK, "--data-dir", str(tmp_path)]
    run_recorded(argv)
    output = "".join(capsys.readouterr())
    assert "no capture source found" not in output, (
        f"ruling A14: {ENV_SOURCE}={ENV_MIC!r} *is* a capture source the human named, so this box "
        f"is not the one ruling A3 refuses; it said:\n{output}"
    )
    call = device_call(calls, argv, output)
    assert call["source"] == ENV_MIC, (
        f"ruling A14: …and it is the node the device is asked for; open_device got "
        f"source={call['source']!r}"
    )
    assert regime_mic(output) == ENV_MIC, (
        f"…and the node the run's regime names; it printed mic={regime_mic(output)!r}:\n{output}"
    )


# --- T-0043 criterion 15: --no-log, --log-dir, and saying which it is ---------------------------


def log_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    """The ``session`` namespace with T-0043's two new flags, ``--silent`` on (no device)."""
    return silent_args(tmp_path, **{"no_log": False, "log_dir": None, **overrides})


def test_the_session_parser_has_the_logging_flags():
    """Criterion 15: ``astra session`` gains ``--no-log`` and ``--log-dir``."""
    flags = flags_of(CLI)
    missing = {"--no-log", "--log-dir"} - flags
    assert not missing, (
        f"criterion 15: `astra session` accepts {sorted(missing)}; the session parser declares "
        f"{sorted(flags)}"
    )


def test_dry_run_says_where_the_log_will_go_and_when_it_is_off(tmp_path, capsys, monkeypatch):
    """Criterion 15: ``--dry-run``/``describe_graph`` reports ``log=<path>`` or ``log=off``.

    A resolved-graph line that does not say whether the run will be recorded is the same failure
    as ``sink=`` not saying which device is playing: the human cannot tell what they are about to
    get.
    """
    explode_clients(monkeypatch)
    log_dir = tmp_path / "somewhere"

    code = run_cli(
        [
            "session",
            "--dry-run",
            "--silent",
            "--data-dir",
            str(tmp_path / "data"),
            "--log-dir",
            str(log_dir),
        ]
    )
    printed = capsys.readouterr()
    out = printed.out + printed.err
    assert code == 0, f"criterion 15: --dry-run exits 0; it exited {code}. Output:\n{out}"
    assert f"log={log_dir}" in out, (
        f"criterion 15: the resolved graph names the log directory it was given; it printed:\n{out}"
    )
    assert not log_dir.exists(), "…and --dry-run still writes nothing: it created the log directory"

    code = run_cli(
        ["session", "--dry-run", "--silent", "--no-log", "--data-dir", str(tmp_path / "data2")]
    )
    out = capsys.readouterr().out
    assert code == 0 and "log=off" in out, (
        f"criterion 15: with --no-log the same line reads log=off; it printed:\n{out}"
    )


def test_no_log_builds_a_session_with_no_log_and_creates_no_file(tmp_path, monkeypatch):
    """Criterion 15: with ``--no-log`` no file is created in the log directory at all.

    Not "an empty file", not "a file that is never written to": none. A session started with
    ``--no-log`` leaves no trace in ``logs/sessions/``.
    """
    from local_astra.config import load

    spy_null_device(monkeypatch)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    graph = cli.build_session(load(None), log_args(tmp_path, no_log=True, log_dir=str(log_dir)))

    built = getattr(graph.session, "session_log", "<no session_log attribute>")
    assert built is None, (
        f"criterion 15: --no-log starts with logging off; the session holds {built!r}"
    )
    assert not list(log_dir.iterdir()), (
        f"…and no file was created in {log_dir}; it holds {list(log_dir.iterdir())}"
    )
    assert "log=off" in graph.describe(), (
        f"…and the graph says so; it describes itself as:\n{graph.describe()}"
    )


async def test_log_dir_puts_the_file_where_it_was_asked_to(tmp_path, monkeypatch):
    """Criterion 15 and 1: ``--log-dir DIR`` is the directory the session's log is created in."""
    from local_astra.config import load

    spy_null_device(monkeypatch)
    log_dir = tmp_path / "elsewhere"
    graph = cli.build_session(load(None), log_args(tmp_path, log_dir=str(log_dir)))

    log = getattr(graph.session, "session_log", None)
    assert log is not None and hasattr(log, "path"), (
        "criterion 15: logging is on by default — the flag is --no-log, not --log; the session "
        f"holds {log!r}"
    )
    assert Path(log.path).parent == log_dir, (
        f"criterion 15: --log-dir decides where the file goes; it is at {log.path}"
    )
    await log.start()
    await log.aclose()
    assert Path(log.path).is_file(), (
        f"…and starting the log creates exactly that file; {log_dir} holds "
        f"{list(log_dir.iterdir()) if log_dir.exists() else 'nothing'}"
    )


def test_run_session_prints_the_exact_log_path(tmp_path, capsys):
    """Criterion 15: ``run_session`` prints one line naming the log file it is writing.

    The human at the desk needs the path at the moment the session starts, not after it ends:
    that line is what they paste into ``session_log_summary.py`` while the session is running.
    """
    from tests.unit._hud_rig import Recorder, make_rig, require_symbol
    from tests.unit._session_log_rig import make_log

    log = make_log(tmp_path / "logs")
    rig = make_rig(tmp_path / "data", publish=Recorder(), session_log=log)
    hud_server = require_symbol("HudServer")
    graph = _Graph(
        session=rig.session,
        server=hud_server(rig.session, frame_sink=_Sink(), port=0),
        rig=rig,
    )

    async def drive() -> int:
        task = asyncio.ensure_future(cli.run_session(graph))  # type: ignore[attr-defined]
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            return int(await task)
        except asyncio.CancelledError:
            return 0

    code = asyncio.run(drive())
    out = capsys.readouterr().out
    assert code == 0, f"the session came up and went down cleanly; it returned {code}"
    assert str(log.path) in out, (
        f"criterion 15: the startup line names the exact log path ({log.path}); it printed:\n{out}"
    )


# --- T-0043 ruling B1: --no-log is a start state, not a lock ------------------------------------
#
# The row ruling B1 called for and the first group-B pass could not write, because the seam did
# not exist: "the session carries the **resolved** log directory (the ``--log-dir`` value or the
# default) regardless of whether logging starts on, and the toggle opens there. Do not make
# ``--no-log`` a state the page cannot leave — a checkbox that cannot be ticked is worse than no
# checkbox."
#
# The observable is a file, not an attribute: the panel's checkbox exists to say whether this
# session is being recorded (criterion 19, ruling B8), so "the box can be ticked" means a session
# started with --no-log ends up writing a real file in the directory the command line named.
#
# ** THIS ROW IS EXPECTED TO FAIL TODAY, AND ruling B9 says why. ** The one line that closes it
# lived in ``local_astra/cli.py``: ``build_session`` passed ``session_log=`` without
# ``log_dir=log_dir_of(args)``, so ``--no-log`` left the session with nowhere to write and the
# checkbox could not be ticked. That line landed with this row's marker removed in the same commit
# (ruling B22), which is what the strict marker was for: it was listed by name in every run, and
# being strict it failed loudly the moment the line arrived rather than quietly starting to pass.
# A bare red would have been forgotten, and once this branch merged it would have become a
# blocking failure in every agent's checks — which is how a real regression gets waved through.


async def test_a_session_started_with_no_log_can_still_be_switched_on(tmp_path, monkeypatch):
    """Ruling B1 (criteria 15 and 17): --no-log starts logging off; the page can turn it on.

    Driven through the two real seams, not through an attribute: ``cli.build_session`` builds the
    graph the command line asked for, and the toggle arrives as the JSON a panel puts on the
    socket, through ``HudServer.route``'s hello fence. What is asserted is the file on disk and
    the state the server echoed back — the panel renders that echo verbatim and nothing else.
    """
    import json as _json

    from local_astra.config import load
    from local_astra.contracts.codec import encode
    from local_astra.contracts.protocol import AstraHello
    from tests.unit._hud_rig import require_symbol
    from tests.unit._session_log_rig import kinds, read_records

    spy_null_device(monkeypatch)
    log_dir = tmp_path / "logs"  # named on the command line, and does not exist yet
    graph = cli.build_session(load(None), log_args(tmp_path, no_log=True, log_dir=str(log_dir)))
    session = graph.session

    assert getattr(session, "session_log", "<no session_log attribute>") is None, (
        "premise (criterion 15): --no-log starts with logging off, or this row proves nothing "
        f"about turning it on; the session holds {getattr(session, 'session_log', None)!r}"
    )
    server = require_symbol("HudServer")(session, frame_sink=_Sink(), port=0)
    conn = server.open()
    conn.receive(encode(AstraHello()))
    conn.receive(_json.dumps({"type": "astra.log.toggle", "on": True}))

    # Bounded and clock-free: opening the file is a task the toggle spawned and the first record
    # is written by the log's own writer task, so let the loop run both. No sleep on the wall.
    def written() -> list[Path]:
        if not log_dir.exists():
            return []
        return [p for p in sorted(log_dir.glob("*.jsonl")) if p.stat().st_size > 0]

    for _ in range(500):
        if written():
            break
        await asyncio.sleep(0)

    try:
        files = written()
        states = [m for m in conn.sent() if getattr(m, "type", None) == "astra.log.state"]
        held = list(log_dir.iterdir()) if log_dir.exists() else "nothing (no directory)"
        assert files, (
            "ruling B1: ticking 'Log this session' on a session started with --no-log writes a "
            f"file in the directory the command line named ({log_dir}); it holds {held}. The "
            f"server echoed error={[getattr(m, 'error', None) for m in states]}. '--no-log' is a "
            "start state, not a lock: the session carries the *resolved* log directory either "
            "way and the toggle opens there. The missing line is build_session's, in "
            "local_astra/cli.py — it passes session_log= and not the directory — and ruling B9 "
            "queued it behind T-0049 group B, which holds that file"
        )
        assert len(files) == 1, f"…exactly one file, not one per click; {[p.name for p in files]}"
        assert kinds(read_records(files[0]))[:1] == ["log.started"], (
            "criterion 17: the file the toggle opened begins with its own log.started; it begins "
            f"{kinds(read_records(files[0]))[:3]}"
        )
        assert states, "criterion 17: every routed toggle is echoed with one astra.log.state"
        last = states[-1]
        assert last.on is True and last.error is None, (
            "ruling B8: `on` means 'this session is writing a file' — after a tick that opened "
            f"one, the page is told so; it was told on={last.on!r} error={last.error!r}"
        )
        assert last.path == str(files[0]), (
            f"criterion 19: …and the path the panel renders verbatim is the file just opened "
            f"({files[0]}); the echo says {last.path!r}"
        )
    finally:
        log = getattr(session, "session_log", None)
        if log is not None:
            await log.aclose()


# --- T-0092b: `--tools`, the startup kit, staleness, and the off-loop load ------------------------
#
# Every assumed name is in tests/unit/_tools_switch_support.py's docstring. The fixtures are copies:
# versions.lock under a temporary ``services/`` (``cli.SERVICES_DIR``), the payload and calibration files
# (``live.DEFAULT_PAYLOADS``/``DEFAULT_CALIBRATION``), and T-0088a's subset tokenizer behind
# ``HostDetokenizer.load`` — never the real 31 MB file.

from tests.unit import _tools_switch_support as T  # noqa: E402

TOOLS_LINE = re.compile(r"\btools=(off|dry_run native:([0-9a-f]{12}) repair=(on|off\((.+)\)))")
REFUSAL_NAMES = ("docs/01 §3", "T-0080d")


def _tools_token(output: str) -> re.Match[str]:
    found = [m for m in (TOOLS_LINE.search(line) for line in output.splitlines()) if m]
    assert len(found) == 1, (
        "T-0092b criterion 1: `astra session --dry-run` prints one `tools=off` or "
        f"`tools=dry_run native:<sha12> repair=on|off(<reason>)`; it printed:\n{output}"
    )
    return found[0]


def _bomb_session(monkeypatch: pytest.MonkeyPatch) -> None:
    def bomb(*a: object, **k: object) -> object:
        raise AssertionError("T-0092b: the session must never be constructed here")

    monkeypatch.setattr(cli, "Session", bomb)


def test_the_session_parser_has_tools_and_up_does_not(monkeypatch):
    """Criterion 1: ``astra session --tools``; ``astra up`` gets no flag (it starts no conductor)."""
    assert "--tools" in flags_of(CLI), "T-0092b criterion 1: `astra session --tools {off,dry-run}`"

    def no_up(*a: object, **k: object) -> int:
        raise AssertionError(
            "T-0092b criterion 1: `astra up --tools` must be an argv error, not a start"
        )

    monkeypatch.setattr(cli, "up", no_up)
    assert run_cli(["up", "--tools", "off"]) == 2


#: T-0111g re-scope R2: ``"live"`` left this list when ``--tools live`` became a mode (docs/tasks/T-0111g.md B2,
#: "Named re-scopes"); its acceptance is ``tests/unit/test_cli_tools_live.py``.
@pytest.mark.parametrize("value", ["junk", "dry_run"])
def test_tools_live_or_junk_on_argv_is_refused_naming_the_non_goal(
    tmp_path, capsys, monkeypatch, value
):
    """Criterion 1: any ``--tools`` value but the modes is refused at argv, naming docs/01 §3 and T-0080d, and the
    session never starts."""
    _bomb_session(monkeypatch)
    pin_regime_facts(monkeypatch)
    code = run_cli(T.session_argv(T.write_config(tmp_path), tmp_path, "--tools", value))
    out = capsys.readouterr()
    message = out.out + out.err
    assert code != 0, f"refused: a non-zero exit; got {code}"
    for name in REFUSAL_NAMES:
        assert name in message, (
            f"T-0092b criterion 1: the refusal names {name!r}; it said:\n{message}"
        )


@pytest.mark.parametrize(
    ("profile", "flag", "want"),
    [
        (None, [], "off"),
        ("dry_run", [], "dry_run"),
        (None, ["--tools", "dry-run"], "dry_run"),
        ("dry_run", ["--tools", "off"], "off"),
        ("off", ["--tools", "off"], "off"),
    ],
    ids=["default", "profile", "flag-on", "flag-off-overrides", "both-off"],
)
def test_dry_run_prints_the_tools_line_and_the_flag_overrides_the_profile(
    tmp_path, capsys, monkeypatch, profile, flag, want
):
    """Criterion 1: ``--dry-run`` prints ``tools=off`` or ``tools=dry_run native:bbfc53ccdd80 repair=on``; the
    flag beats the profile. Still constructs no client and opens no device."""
    explode_clients(monkeypatch)
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    code = run_cli(
        [*T.session_argv(T.write_config(tmp_path, profile), tmp_path), "--dry-run", *flag]
    )
    out = capsys.readouterr()
    assert code == 0, f"--dry-run exits 0; it exited {code}:\n{out.out}{out.err}"
    m = _tools_token(out.out + out.err)
    if want == "off":
        assert m.group(1) == "off", m.group(0)
        assert not calls.detok_loads, "tools off: the detokeniser is never loaded"
    else:
        assert m.group(2) == T.TOOLS_SHA12 == T.tools_sha256()[:12], m.group(0)
        assert m.group(3) == "on", (
            f"the shipped payloads, calibration and lock are consistent: {m.group(0)}"
        )
    assert not list((tmp_path / "data").glob("*")) if (tmp_path / "data").exists() else True


def test_dry_run_prints_repair_off_with_its_reason(tmp_path, capsys, monkeypatch):
    """Criterion 1: ``repair=off(<reason>)`` — here the payload file is missing, and the reason names it."""
    explode_clients(monkeypatch)
    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    missing = tmp_path / "nowhere" / "native_T4.json"
    T.point_cli_at(monkeypatch, payloads=missing)
    code = run_cli(
        [*T.session_argv(T.write_config(tmp_path), tmp_path), "--dry-run", "--tools", "dry-run"]
    )
    out = capsys.readouterr()
    assert code == 0, out.out + out.err
    m = _tools_token(out.out + out.err)
    assert m.group(3) and m.group(3).startswith("off("), m.group(0)
    assert str(missing) in (m.group(4) or ""), f"the reason names the missing file: {m.group(0)}"


@pytest.mark.parametrize("kind", ["missing", "bad sha", "anything"])
def test_a_detokeniser_load_failure_exits_1_naming_the_file_and_starts_nothing(
    tmp_path, capsys, monkeypatch, kind
):
    """Criterion 2(a): any exception from ``HostDetokenizer.load()`` refuses dry_run — exit 1, the message names
    the pinned file and the reason, no session is constructed and nothing is served."""
    from local_astra.toolcall import detok

    path = detok.pinned_tokenizer_path()
    exc: BaseException = {
        "missing": FileNotFoundError(f"{path} is missing: fetch the tokenizer files"),
        "bad sha": detok.TokenizerFileError(f"{path}: sha256 0000 is not the expected cc8d3a0c"),
        "anything": RuntimeError("the decoder table could not be built"),
    }[kind]
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path, raise_=exc)
    code = run_cli(T.session_argv(T.write_config(tmp_path), tmp_path, "--tools", "dry-run"))
    out = capsys.readouterr()
    message = out.out + out.err
    assert calls.detok_loads, "premise: the load was attempted"
    assert code == 1, f"T-0092b criterion 2(a): exit 1; got {code}:\n{message}"
    assert str(path) in message, f"…naming the file {path}; it said:\n{message}"
    assert str(exc).split(": ")[-1] in message, f"…and the reason ({exc}); it said:\n{message}"
    assert not calls.session_kwargs and not calls.graphs, (
        "…and no session was constructed or served"
    )


def _run_dry_run(tmp_path, monkeypatch, capsys, *extra_cfg: object, **point: object):
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    T.spy_loaders(monkeypatch, calls)
    T.point_cli_at(monkeypatch, **point)  # type: ignore[arg-type]
    config = T.write_config(tmp_path, *extra_cfg) if extra_cfg else T.write_config(tmp_path)
    code = run_cli(T.session_argv(config, tmp_path, "--tools", "dry-run"))
    out = capsys.readouterr()
    assert code == 0, f"premise: the dry-run session started (exit {code}):\n{out.out}{out.err}"
    return calls, out


def test_a_consistent_kit_starts_with_repair_on_and_is_injected(tmp_path, monkeypatch, capsys):
    """Criterion 2: (a)-(c) run once each, the client gets the loaded detokeniser, the session gets the kit with
    payloads, calibration and a ``DryRunToolExecutor``; repair is on and no repair line is printed."""
    import asyncio as _asyncio

    from local_astra.conductor.tool_executor import DryRunToolExecutor

    calls, out = _run_dry_run(tmp_path, monkeypatch, capsys, services=T.lock_copy(tmp_path))
    assert (
        len(calls.detok_loads) == 1 and calls.payload_loads == 1 and calls.calibration_loads == 1
    ), (
        calls.detok_loads,
        calls.payload_loads,
        calls.calibration_loads,
    )
    assert calls.vllm_kwargs and calls.vllm_kwargs[-1].get("detokenizer") is not None, (
        f"T-0092b criterion 2: VllmClient(..., detokenizer=…); it got {calls.vllm_kwargs}"
    )
    kit = calls.kit()
    assert kit is not None, "T-0092b criterion 2: the session receives the kit by injection"
    assert T.kit_attr(kit, "repair_off") is None
    assert tuple(T.kit_attr(kit, "tools")) == T.tools_field()
    assert T.kit_attr(kit, "payloads") is not None and T.kit_attr(kit, "calibration") is not None
    assert isinstance(T.kit_attr(kit, "executor"), DryRunToolExecutor)
    assert calls.executors == 1
    assert not [line for line in out.err.splitlines() if "repair" in line.lower()], out.err
    assert _asyncio.run(T.regime_tools_of(kit, tmp_path / "replay")) == T.REGIME_ON


def _edit_manifest(key: str, value: object):
    def edit(doc: dict) -> None:
        doc["manifest"][key] = value

    return edit


STALE = {
    "model_revision": ("model_revision",),
    "tokenizer": ("tokenizer_sha256",),
    "canvas": ("canvas_length",),
    "missing payload": ("native_T4.json",),
    "calibration sha": ("calibration",),
    "lock model_revision": ("model_revision",),
    # coordinator follow-up after review: the lock itself unreadable or shapeless is a reason, never a crash
    "missing lock": ("versions.lock",),
    "lock without [gpu]": ("versions.lock",),
}


def _lock_without_gpu(tmp_path):  # type: ignore[no-untyped-def]
    """A ``services/`` copy whose versions.lock has every table but ``[gpu]``."""
    import tomllib

    services = T.lock_copy(tmp_path)
    lock = services / "versions.lock"
    kept, skipping = [], False
    for line in lock.read_text().splitlines():
        if line.strip().startswith("["):
            skipping = line.strip() == "[gpu]"
        if not skipping:
            kept.append(line)
    lock.write_text("\n".join(kept) + "\n")
    assert "gpu" not in tomllib.loads(lock.read_text()), "premise: the copy has no [gpu] table"
    return services


@pytest.mark.parametrize("branch", list(STALE))
def test_each_staleness_branch_turns_repair_off_and_says_so(tmp_path, monkeypatch, capsys, branch):
    """Criterion 2, staleness: each mismatch or ``PayloadStale`` turns repair **off** (it never refuses the session):
    one stderr line naming the reason, the kit carries the reason, and the regime gains ``:repair_off``.

    The six branches: the manifest's ``model_revision`` against versions.lock (edited on either side), its
    ``tokenizer_sha256`` against the detokeniser's pinned sha, its ``canvas_length`` against the profile (the
    profile is set to 128), a missing payload file, and an edited calibration copy."""
    import asyncio as _asyncio

    point: dict[str, object] = {"services": T.lock_copy(tmp_path)}
    cfg: tuple[object, ...] = ()
    if branch == "model_revision":
        point["payloads"] = T.payload_copy(tmp_path, _edit_manifest("model_revision", "0" * 40))
    elif branch == "tokenizer":
        point["payloads"] = T.payload_copy(tmp_path, _edit_manifest("tokenizer_sha256", "0" * 64))
    elif branch == "canvas":
        cfg = (None,)
        monkeypatch.setattr(T, "write_config", _with_canvas(T.write_config, 128))
    elif branch == "missing payload":
        point["payloads"] = tmp_path / "gone" / "native_T4.json"
    elif branch == "calibration sha":
        point["calibration"] = T.calibration_copy(tmp_path, edit=True)
    elif branch == "lock model_revision":
        point["services"] = T.lock_copy(tmp_path, model_revision="1" * 40)
    elif branch == "missing lock":
        empty = tmp_path / "services-empty"
        empty.mkdir()
        point["services"] = empty
    elif branch == "lock without [gpu]":
        point["services"] = _lock_without_gpu(tmp_path)
    try:
        calls, out = _run_dry_run(tmp_path, monkeypatch, capsys, *cfg, **point)
    except AssertionError:
        raise
    except Exception as exc:  # a staleness branch that crashes is the failure under test, named
        raise AssertionError(
            f"T-0092b criterion 2: {branch} crashed the startup instead of turning repair off: {exc!r}"
        ) from None
    kit = calls.kit()
    assert kit is not None, (
        "the session still starts: staleness turns repair off, it does not refuse"
    )
    reason = T.kit_attr(kit, "repair_off")
    assert reason, (
        f"T-0092b criterion 2: {branch} turns repair off; the kit says repair_off={reason!r}"
    )
    for needle in STALE[branch]:
        assert needle in str(reason), f"the reason names {needle!r}: {reason!r}"
    # a repair-off kit carries no payloads and no calibration, whichever branch turned it off
    assert T.kit_attr(kit, "payloads") is None, (
        f"T-0092b criterion 2: repair off ({branch}) leaves the kit without payloads; it holds "
        f"{T.kit_attr(kit, 'payloads')!r}"
    )
    assert T.kit_attr(kit, "calibration") is None, (
        f"T-0092b criterion 2: repair off ({branch}) leaves the kit without a calibration; it holds "
        f"{T.kit_attr(kit, 'calibration')!r}"
    )
    assert "Traceback" not in out.err + out.out, f"no traceback, a reason:\n{out.err}"
    lines = [line for line in out.err.splitlines() if "repair" in line.lower()]
    assert len(lines) == 1 and "off" in lines[0] and str(reason) in lines[0], (
        f"T-0092b criterion 2: one stderr line announces repair off with its reason; stderr was:\n{out.err}"
    )
    assert calls.vllm_kwargs[-1].get("detokenizer") is not None, "the detokeniser is still wired"
    assert _asyncio.run(T.regime_tools_of(kit, tmp_path / "replay")) == T.REGIME_REPAIR_OFF


def _with_canvas(write, canvas):  # type: ignore[no-untyped-def]
    def wrapped(tmp_path, tools=None, **_):  # type: ignore[no-untyped-def]
        return write(tmp_path, tools, canvas_length=canvas)

    return wrapped


def test_the_too_wide_direction_the_shipped_files_leave_repair_on(tmp_path, monkeypatch, capsys):
    """Criterion 2, the guard's other side: a lock copy and payload/calibration copies that agree keep repair on
    (a staleness check keyed on the wrong field, or on the file path, would turn it off here)."""
    calls, _ = _run_dry_run(
        tmp_path,
        monkeypatch,
        capsys,
        services=T.lock_copy(tmp_path),
        payloads=T.payload_copy(tmp_path),
        calibration=T.calibration_copy(tmp_path, edit=False),
    )
    assert T.kit_attr(calls.kit(), "repair_off") is None


def test_the_detokeniser_is_loaded_off_the_event_loop(tmp_path, monkeypatch, capsys):
    """Criterion 2: (a)-(c) run in ``build_session``, before ``asyncio.run`` — never on the loop.

    The real path runs end to end: ``asyncio.run(_serve(...))`` starts the session over the fakes, then the
    server's ``start`` fails as a busy port does, so the command ends. ``HostDetokenizer.load`` records whether an
    event loop was running on its calling thread; a load moved into ``run_session``/``_serve`` is caught."""
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)

    async def busy(self, *a: object, **k: object) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr(cli.HudServer, "start", busy)
    code = run_cli(T.session_argv(T.write_config(tmp_path), tmp_path, "--tools", "dry-run"))
    capsys.readouterr()
    assert calls.session_kwargs, "premise: the session was built"
    assert code == 1, f"premise: the busy server ended the run; exit {code}"
    assert calls.detok_loads == [False], (
        f"T-0092b criterion 2: HostDetokenizer.load runs once, with no event loop running on the calling "
        f"thread; per load (loop running?) = {calls.detok_loads}"
    )


def test_with_tools_off_nothing_of_the_kit_is_loaded_or_constructed(tmp_path, monkeypatch, capsys):
    """Criterion 7: with ``off``, ``HostDetokenizer``, ``load_payloads``, ``load_calibration`` and the executor are
    never called or constructed; the client gets no detokeniser and the session no kit."""
    from local_astra.toolcall.detok import HostDetokenizer

    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.spy_loaders(monkeypatch, calls, bomb=True)

    def no_detok(*a: object, **k: object) -> object:
        raise AssertionError(
            "T-0092b criterion 7: with tools = off, HostDetokenizer is never loaded"
        )

    monkeypatch.setattr(HostDetokenizer, "load", classmethod(no_detok))
    monkeypatch.setattr(HostDetokenizer, "__init__", no_detok)
    code = run_cli(T.session_argv(T.write_config(tmp_path, "off"), tmp_path))
    capsys.readouterr()
    assert code == 0 and calls.session_kwargs, (
        f"premise: the tools-off session started (exit {code})"
    )
    assert calls.vllm_kwargs[-1].get("detokenizer") is None
    assert calls.session_kwargs[-1].get(T.KIT_KWARG) is None


# --- T-0128c A8: the denoise relay's wiring (docs/tasks/T-0128c.md, group A) ---------------------
#
# Seams (task, implementation notes; the relay API is ruling O-A's): ``build_session`` builds a
# ``canvas_relay.CanvasRelay`` with ``publish`` = the ``HudServer``'s publish, ``state`` =
# ``session.turn_state_message``, the session clock and the session's log, its endpoint from
# ``ASTRA_TAP_ENDPOINT`` (default ``tcp://127.0.0.1:8002``), and carries it as ``SessionGraph.relay``;
# ``run_session`` starts it after the server and closes it before the server. These rows never start a
# relay on 8002: every graph that is served gets an ``ipc://`` endpoint in a pytest temp dir, or a
# refused one. The relay's module is reached through ``tests/fakes/canvas_tap.relay_attr`` so a missing
# module is an assertion naming T-0128c, not a collection error.

T0128C_ENV = "ASTRA_TAP_ENDPOINT"
#: what A8's one stderr line is about: a line that names neither cannot be the relay's
T0128C_WHAT = ("denois", "canvas", "tap", "relay")


def _relay_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str | None) -> object:
    """A real ``build_session`` graph, ``--silent`` with logging on, on free ports, ``endpoint`` in the
    environment (``None``: unset)."""
    from local_astra.config import load

    spy_null_device(monkeypatch)
    _explode_device(monkeypatch, AssertionError("T-0128c: --silent never calls open_device"))
    if endpoint is None:
        monkeypatch.delenv(T0128C_ENV, raising=False)
    else:
        monkeypatch.setenv(T0128C_ENV, endpoint)
    # T-0159 ruling O-A1: ``relay.start()`` awaits the piece load (``HostDetokenizer.load``, 355 ms
    # on the pinned file) before it reads. These rows serve for 0.3 s, so a real load would still
    # be running when they cancel. An empty HF cache makes the load fail at once: the relay takes
    # the A5 path (no pieces, one stderr line, one note), which the ruling accepts here.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "t0159-empty-hf-cache"))
    config_path, _, _ = config_on_free_ports(tmp_path)
    with sounddevice_unimportable():
        return cli.build_session(
            load(config_path), log_args(tmp_path / "data", log_dir=str(tmp_path / "logs"))
        )


def _relay_of(graph: object) -> object:
    from tests.fakes.canvas_tap import relay_attr

    cls = relay_attr("CanvasRelay")
    relay = getattr(graph, "relay", None)
    assert isinstance(relay, cls), (
        f"T-0128c A8: the session graph carries a CanvasRelay as ``graph.relay``; it has {relay!r}"
    )
    return relay


def _ipc(tmp_path_factory: pytest.TempPathFactory) -> str:
    return f"ipc://{tmp_path_factory.mktemp('t0128c') / 'tap.sock'}"


def _serve_and_probe(graph: object, hold_s: float = 0.3) -> tuple[int, bool]:
    """``run_session`` until the HUD port accepts a connection, ``hold_s`` more, then cancelled.
    Returns (exit code, whether the HUD port accepted a connection while it ran)."""

    async def drive() -> tuple[int, bool]:
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(cli.run_session(graph))  # type: ignore[arg-type]
        served = False
        deadline = loop.time() + 3.0
        while not served and not task.done() and loop.time() < deadline:
            try:
                port = int(graph.server.port)  # type: ignore[attr-defined]
                _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                served = True
            except Exception:
                await asyncio.sleep(0.02)
        await asyncio.sleep(hold_s)
        task.cancel()
        try:
            code = int(await task)
        except asyncio.CancelledError:
            code = 0
        except Exception as exc:
            raise AssertionError(
                f"T-0128c A8: the session runs without the relay; run_session raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return code, served

    with sounddevice_unimportable():
        return asyncio.run(drive())


def _log_notes(graph: object) -> list[object]:
    import json

    log = getattr(graph.session, "session_log", None)  # type: ignore[attr-defined]
    assert log is not None, "premise: logging is on in these rows"
    path = Path(log.path)
    assert path.is_file(), f"premise: the session log was written at {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row.get("data") for row in rows if row.get("kind") == "note"]


def _one_line_about(err: str, *needles: str) -> list[str]:
    return [line for line in err.splitlines() if any(n.lower() in line.lower() for n in needles)]


def test_the_session_graph_has_a_relay_wired_to_the_server_the_session_and_its_log(
    tmp_path, monkeypatch, tmp_path_factory
):
    """T-0128c (Tests to write, group A): the graph builds a relay, endpoint from the environment,
    publishing through ``HudServer.publish``, admitting by ``session.turn_state_message`` and writing
    its turn note to the session's ``SessionLog``.

    Observed through behaviour, not attribute names: the three production methods are replaced on
    their classes before the graph is built, a frame is handed to ``relay.receive`` while the patched
    state says THINKING (admitted, A4), and the turn is closed (one ``denoise`` note, A7)."""
    from local_astra.contracts.protocol import AstraTurnState
    from local_astra.contracts.turn import TurnState
    from tests.fakes.canvas_tap import denoise, pack

    state = {"now": TurnState.THINKING}
    published: list[object] = []
    noted: list[tuple[str, object]] = []
    monkeypatch.setattr(
        cli.Session,
        "turn_state_message",
        lambda self, state_=None, turn_id=None: AstraTurnState(state=state["now"], turn_id="t1"),
    )
    monkeypatch.setattr(cli.HudServer, "publish", lambda self, message: published.append(message))
    monkeypatch.setattr(
        cli.SessionLog, "record", lambda self, kind, data, **kw: noted.append((kind, data))
    )
    endpoint = _ipc(tmp_path_factory)
    graph = _relay_graph(tmp_path, monkeypatch, endpoint)
    relay = _relay_of(graph)
    assert relay.endpoint == endpoint, (  # type: ignore[attr-defined]
        f"A1: the relay's endpoint is ASTRA_TAP_ENDPOINT ({endpoint}); it is {relay.endpoint!r}"  # type: ignore[attr-defined]
    )
    relay.receive(pack(denoise("gen", 0, 1)))  # type: ignore[attr-defined]
    relay.flush()  # type: ignore[attr-defined]
    steps = [m for m in published if getattr(m, "type", "") == "astra.canvas.step"]
    assert [(m.request_id, m.step) for m in steps] == [("gen", 1)], (  # type: ignore[attr-defined]
        "the relay publishes through HudServer.publish and admits by session.turn_state_message; "
        f"HudServer.publish saw {published}"
    )
    state["now"] = TurnState.IDLE
    relay.flush()  # type: ignore[attr-defined]
    notes = [d for k, d in noted if k == "note" and isinstance(d, dict) and "denoise" in d]
    assert len(notes) == 1, (
        f"A7: the closed turn's note goes to the session's SessionLog; it recorded {noted}"
    )


def test_with_no_environment_the_graphs_relay_is_on_the_locked_endpoint(tmp_path, monkeypatch):
    """A1: ``ASTRA_TAP_ENDPOINT`` unset -> the relay's endpoint is ``tcp://127.0.0.1:8002``. Built,
    never started (the fast suite never touches 8002)."""
    graph = _relay_graph(tmp_path, monkeypatch, None)
    assert _relay_of(graph).endpoint == "tcp://127.0.0.1:8002"  # type: ignore[attr-defined]


def test_run_session_starts_the_relay_after_the_server_and_closes_it_before(
    tmp_path, monkeypatch, tmp_path_factory, capsys
):
    """Group A wiring: ``run_session`` awaits ``relay.start()`` once, after ``server.start()``
    returned, and ``relay.aclose()`` once, finished before ``server.aclose()`` begins (the relay
    publishes through the server). Recorded by wrapping the four methods on the instances."""
    graph = _relay_graph(tmp_path, monkeypatch, _ipc(tmp_path_factory))
    relay = _relay_of(graph)
    events: list[str] = []

    def wrap(obj: object, name: str, label: str) -> None:
        real = getattr(obj, name)

        async def wrapper(*args: object, **kwargs: object) -> object:
            events.append(f"{label}.{name}")
            try:
                return await real(*args, **kwargs)
            finally:
                events.append(f"{label}.{name}.done")

        setattr(obj, name, wrapper)

    for name in ("start", "aclose"):
        wrap(graph.server, name, "server")  # type: ignore[attr-defined]
        wrap(relay, name, "relay")
    code, served = _serve_and_probe(graph)
    capsys.readouterr()
    assert code == 0 and served, (
        f"premise: the session served and stopped cleanly ({code}, {served})"
    )
    for name in ("relay.start", "relay.aclose", "server.start", "server.aclose"):
        assert events.count(name) == 1, f"{name} awaited once; events {events}"
    assert events.index("relay.start") > events.index("server.start.done"), (
        f"the relay starts after the server: {events}"
    )
    assert events.index("relay.aclose.done") < events.index("server.aclose"), (
        f"the relay is closed before the server: {events}"
    )


@pytest.mark.parametrize("why", ["start raises", "pyzmq missing"])
def test_a_relay_that_cannot_start_says_so_once_and_the_session_still_serves(
    why, tmp_path, monkeypatch, tmp_path_factory, capsys
):
    """A8: the relay cannot start -> one stderr line naming the reason, one session-log ``note``
    naming it, and the session serves (the HUD port accepts a connection; exit 0 on stop).

    ``start raises``: ``relay.start`` raises ``OSError`` with a unique reason. ``pyzmq missing``:
    ``import zmq`` fails for the whole build and serve (a meta-path finder, the
    ``sounddevice_unimportable`` device), the reason is ``zmq``."""
    reason = "t0128c-tap-refused-by-test"
    hide = contextlib.nullcontext()
    if why == "pyzmq missing":
        reason = "zmq"
        hide = _zmq_unimportable()
    with hide:
        graph = _relay_graph(tmp_path, monkeypatch, _ipc(tmp_path_factory))
        if why == "start raises":
            relay = _relay_of(graph)

            async def refuse() -> None:
                raise OSError(reason)

            relay.start = refuse  # type: ignore[attr-defined]
        code, served = _serve_and_probe(graph)
    err = capsys.readouterr().err
    assert code == 0 and served, (
        f"A8: the session runs without the relay (exit {code}, HUD port served: {served})\n{err}"
    )
    lines = _one_line_about(err, reason)
    assert len(lines) == 1, f"A8: one stderr line names the reason {reason!r}; stderr:\n{err}"
    assert any(w in lines[0].lower() for w in T0128C_WHAT), (
        f"A8: the line says what is off (the denoise relay / tap): {lines[0]!r}"
    )
    notes = [n for n in _log_notes(graph) if reason.lower() in str(n).lower()]
    assert len(notes) == 1, f"A8: the session log records one note naming {reason!r}: {notes}"


def test_a_refused_tap_endpoint_says_so_once_and_the_session_still_serves(
    tmp_path, monkeypatch, capsys
):
    """A1 + A8: ``ASTRA_TAP_ENDPOINT=tcp://0.0.0.0:8002`` is refused (``ValueError`` before any
    socket); ``build_session`` does not fail, one stderr line names the variable or the value, the
    session log records a note, and the session serves."""
    bad = "tcp://0.0.0.0:8002"
    graph = _relay_graph(tmp_path, monkeypatch, bad)
    code, served = _serve_and_probe(graph)
    err = capsys.readouterr().err
    assert code == 0 and served, (
        f"A8: the session runs without the relay (exit {code}, HUD port served: {served})\n{err}"
    )
    lines = _one_line_about(err, T0128C_ENV, "0.0.0.0")
    assert len(lines) == 1, f"A8: one stderr line names {T0128C_ENV} or {bad!r}; stderr:\n{err}"
    notes = [n for n in _log_notes(graph) if T0128C_ENV in str(n) or "0.0.0.0" in str(n)]
    assert len(notes) == 1, f"A8: the session log records one note naming the endpoint: {notes}"


def test_the_relays_turn_note_lands_in_the_log_the_hud_toggle_opened(
    tmp_path, monkeypatch, tmp_path_factory, capsys
):
    """Reviewer 2 T3 (kills C3, the relay holding ``session.session_log`` from build time): the
    session serves with logging on; the HUD toggles the log off, then on (a new ``SessionLog``
    file); a turn with one admitted frame then closes. The relay's ``denoise`` note is in the new
    file, and the first file holds none."""
    import json

    from local_astra.contracts.protocol import AstraLogToggle, AstraTurnState
    from local_astra.contracts.turn import TurnState
    from tests.fakes.canvas_tap import denoise, pack

    state = {"now": TurnState.THINKING}
    monkeypatch.setattr(
        cli.Session,
        "turn_state_message",
        lambda self, state_=None, turn_id=None: AstraTurnState(state=state["now"], turn_id="t1"),
    )
    graph = _relay_graph(tmp_path, monkeypatch, _ipc(tmp_path_factory))
    relay = _relay_of(graph)
    session = graph.session  # type: ignore[attr-defined]
    first_path = Path(session.session_log.path)

    async def until(pred, what: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 3.0
        while not pred():
            assert loop.time() < deadline, f"T3 premise: {what}"
            await asyncio.sleep(0.01)

    async def drive() -> Path:
        task = asyncio.ensure_future(cli.run_session(graph))  # type: ignore[arg-type]
        await asyncio.sleep(0.3)  # run_session has started the session, the server, the relay
        assert not task.done(), f"T3 premise: the session is serving ({task})"
        session.apply_log_toggle(AstraLogToggle(on=False))
        await until(lambda: session.session_log is None, "the toggle turned the log off")
        session.apply_log_toggle(AstraLogToggle(on=True))
        await until(lambda: session.session_log is not None, "the toggle opened a new log")
        new_path = Path(session.session_log.path)
        relay.receive(pack(denoise("gen", 0, 1)))  # type: ignore[attr-defined]
        relay.flush()  # type: ignore[attr-defined]
        state["now"] = TurnState.IDLE
        relay.flush()  # type: ignore[attr-defined]
        await asyncio.sleep(0.1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return new_path

    with sounddevice_unimportable():
        new_path = asyncio.run(drive())
    capsys.readouterr()

    def denoise_notes(path: Path) -> list[object]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [
            r["data"]
            for r in rows
            if r.get("kind") == "note"
            and isinstance(r.get("data"), dict)
            and "denoise" in r["data"]
        ]

    assert new_path != first_path, "premise: the toggle opened a second file"
    assert len(denoise_notes(new_path)) == 1, (
        f"T3: the turn's denoise note goes to the log open now ({new_path}); it holds "
        f"{denoise_notes(new_path)}, the first file {denoise_notes(first_path)}"
    )
    assert denoise_notes(first_path) == [], denoise_notes(first_path)


@contextlib.contextmanager
def _zmq_unimportable():
    """``import zmq`` raises ``ImportError`` (pyzmq not installed), for the block's duration."""

    class Refuse:
        def find_spec(self, name, path=None, target=None):
            if name == "zmq" or name.startswith("zmq."):
                raise ImportError(f"No module named {name!r} (pyzmq hidden by T-0128c)")
            return None

    saved = {k: v for k, v in sys.modules.items() if k == "zmq" or k.startswith("zmq.")}
    for k in saved:
        del sys.modules[k]
    finder = Refuse()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k == "zmq" or k.startswith("zmq.")]:
            del sys.modules[k]
        sys.modules.update(saved)
