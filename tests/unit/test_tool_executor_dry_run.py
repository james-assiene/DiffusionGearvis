"""T-0091 criterion 8 (negative): the dry-run executor opens no socket, spawns no process and writes no file (unit).

The double closes by delegation (tests rule): the *real* ``socket``, ``subprocess``, ``os``, ``builtins``/``io`` and
``sqlite3`` modules stay in place, and only the dangerous entry points are replaced -- each raises an
``AssertionError`` naming what was attempted. Everything else (reading a file, ``os.getcwd``, ``json``) is the real
thing. Construction and ``execute`` both run inside the trap. The event loop is created before the trap is armed,
because a selector loop makes its self-pipe with ``socket.socketpair`` (that is the loop's, not the executor's).

The self-check rows prove the trap fires for each dangerous form and lets the harmless ones through, so a green
executor row means "did nothing dangerous", not "the trap was off".
"""

from __future__ import annotations

import asyncio
import builtins
import io
import os
import socket
import sqlite3
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.unit import _toolcall_live_support as S

_WRITE_MODE_CHARS = set("wax+")
_OS_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _raiser(what: str) -> Callable[..., Any]:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(f"T-0091 criterion 8: the dry-run executor {what}")

    return boom


class _NoSocket(socket.socket):
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise AssertionError("T-0091 criterion 8: the dry-run executor opened a socket")


class _NoPopen(subprocess.Popen):  # type: ignore[type-arg]
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise AssertionError("T-0091 criterion 8: the dry-run executor spawned a process")


def _guarded_open(real: Callable[..., Any]) -> Callable[..., Any]:
    def opener(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if _WRITE_MODE_CHARS & set(str(mode)):
            raise AssertionError(
                f"T-0091 criterion 8: the dry-run executor opened {file!r} for writing ({mode!r})"
            )
        return real(file, mode, *args, **kwargs)

    return opener


def _guarded_os_open(real: Callable[..., Any]) -> Callable[..., Any]:
    def os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> Any:
        if flags & _OS_WRITE_FLAGS:
            raise AssertionError(
                f"T-0091 criterion 8: the dry-run executor opened {path!r} for writing (os.open)"
            )
        return real(path, flags, *args, **kwargs)

    return os_open


@contextmanager
def trap() -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(socket, "socket", _NoSocket)
        for name in ("create_connection", "create_server", "socketpair", "fromfd", "getaddrinfo"):
            mp.setattr(socket, name, _raiser(f"used socket.{name}"))
        mp.setattr(subprocess, "Popen", _NoPopen)
        for name in dir(os):
            if name.startswith(("exec", "spawn", "posix_spawn", "fork")) or name in (
                "system",
                "popen",
            ):
                mp.setattr(os, name, _raiser(f"spawned a process (os.{name})"))
        for name in (
            "mkdir",
            "makedirs",
            "remove",
            "unlink",
            "rmdir",
            "rename",
            "replace",
            "truncate",
        ):
            mp.setattr(os, name, _raiser(f"changed the filesystem (os.{name})"))
        mp.setattr(os, "open", _guarded_os_open(os.open))
        mp.setattr(builtins, "open", _guarded_open(builtins.open))
        mp.setattr(io, "open", _guarded_open(io.open))
        mp.setattr(sqlite3, "connect", _raiser("opened a database (sqlite3.connect)"))
        yield
    finally:
        mp.undo()


def _run_trapped(make: Callable[[], Any]) -> Any:
    """Build the invocation, then build the executor and run one ``execute`` inside the trap, on a loop made
    before it."""
    inv = make()
    loop = asyncio.new_event_loop()
    try:

        async def go() -> Any:
            with trap():
                ex = S.dry_run()
                return await ex.execute(inv)

        return loop.run_until_complete(go())
    finally:
        loop.close()


# --------------------------------------------------------------------------- the trap fires (guard absent)


def test_trap_self_check_a_socket_is_refused_by_name() -> None:
    with trap(), pytest.raises(AssertionError, match="opened a socket"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_trap_self_check_a_connection_is_refused_by_name() -> None:
    with trap(), pytest.raises(AssertionError, match=r"socket\.create_connection"):
        socket.create_connection(("127.0.0.1", 9))


def test_trap_self_check_a_process_is_refused_by_name() -> None:
    with trap(), pytest.raises(AssertionError, match="spawned a process"):
        subprocess.run(["true"], check=False)
    with trap(), pytest.raises(AssertionError, match=r"os\.system"):
        os.system("true")


def test_trap_self_check_an_asyncio_subprocess_is_refused() -> None:
    loop = asyncio.new_event_loop()
    try:

        async def go() -> None:
            with trap():
                await asyncio.create_subprocess_exec("true")

        with pytest.raises(AssertionError, match="spawned a process"):
            loop.run_until_complete(go())
    finally:
        loop.close()


@pytest.mark.parametrize(
    "how", ["open", "io.open", "Path.write_text", "Path.touch", "os.open", "mkdir"]
)
def test_trap_self_check_a_write_is_refused_by_name(tmp_path: Path, how: str) -> None:
    target = tmp_path / "x.txt"
    with trap(), pytest.raises(AssertionError, match="criterion 8"):
        if how == "open":
            open(target, "w")  # noqa: SIM115
        elif how == "io.open":
            io.open(target, "ab")  # noqa: SIM115, UP020
        elif how == "Path.write_text":
            target.write_text("x")
        elif how == "Path.touch":
            target.touch()
        elif how == "os.open":
            os.open(target, os.O_WRONLY | os.O_CREAT)
        else:
            (tmp_path / "d").mkdir()
    assert not target.exists()


# --------------------------------------------------------------------------- the trap delegates (guard too wide)


def test_trap_self_check_reading_and_harmless_calls_are_delegated(tmp_path: Path) -> None:
    """Guard too wide: reading a file, listing a directory, ``os.getcwd`` and a ``print`` to stdout still work."""
    src = tmp_path / "r.txt"
    src.write_text("hello")
    with trap():
        assert open(src).read() == "hello"  # noqa: SIM115
        assert src.read_bytes() == b"hello"
        assert os.getcwd() and os.listdir(tmp_path) == ["r.txt"]
        assert socket.AF_INET is not None and subprocess.PIPE == -1


# --------------------------------------------------------------------------- the executor


CALLS: dict[str, tuple[str, dict[str, Any], bool]] = {
    "click_framed": ("click_at", {"x": 500, "y": 500}, True),
    "click_frameless": ("click_at", {"x": 500, "y": 500}, False),
    "type_framed": ("type_at", {"x": 1, "y": 2, "text": "hi", "press_enter": True}, True),
    "type_frameless": ("type_at", {"x": 1, "y": 2, "text": "hi", "press_enter": True}, False),
    "search": ("google_search", {"query": "weather in Lisbon"}, False),
    "scroll": ("scroll", {"direction": "down"}, True),
    "key": ("press_key", {"combo": "Enter"}, False),
}


@pytest.mark.parametrize("row", list(CALLS))
def test_the_dry_run_executor_opens_no_socket_spawns_no_process_writes_no_file(row: str) -> None:
    """Criterion 8 (and T-0080d default 1, H1/H2): building the executor and executing every v1 call, with and without
    a frame, touches neither the network, a process nor the filesystem; the outcome is ``dry_run`` or ``refused``."""
    name, args, framed = CALLS[row]
    out = _run_trapped(
        lambda: S.invocation(name, args, frame=S.geometry(1280, 720) if framed else None)
    )
    assert out.status in ("dry_run", "refused"), out.status


# --------------------------------------------------------------------------- T-0092c: the T-0091 review rows


@pytest.mark.parametrize(
    ("name", "args", "framed", "reason"),
    [
        ("click_at", {"x": 1000, "y": 5}, True, "bad coordinates"),
        ("click_at", {"x": -1, "y": 5}, True, "bad coordinates"),
        ("type_at", {"x": 1, "y": 2, "text": 7, "press_enter": True}, True, "bad arguments"),
        ("type_at", {"x": 1, "y": 2, "text": "hi"}, True, "bad arguments"),
        ("navigate", {"url": "https://example.org"}, True, "unknown tool"),
    ],
)
def test_t0092c_the_dry_run_refuses_bad_coordinates_bad_arguments_and_unknown_tools(
    name, args, framed, reason
):
    """T-0092c (the T-0091 review rows): out-of-range coordinates, a bad ``type_at`` argument and an unknown tool are
    refused with their reason, inside the trap. Green at the base: these pin T-0091's executor."""
    out = _run_trapped(
        lambda: S.invocation(name, args, frame=S.geometry(1280, 720) if framed else None)
    )
    assert (out.status, out.reason) == ("refused", reason), out


@pytest.mark.parametrize(
    ("name", "args", "reason"),
    [
        ("navigate", {"url": "https://example.org"}, "unknown tool"),
        ("scroll", {"direction": 5}, "bad arguments"),
    ],
)
async def test_t0092c_a_session_publishes_the_executors_refusal_from_the_tool_execute_handler(
    tmp_path, name, args, reason
):
    """T-0092c criterion 3: a ``ToolExecute`` driven through the session's handler directly, with the real dry-run
    executor in the kit, publishes ``refused`` with the executor's reason (and nothing executes anywhere)."""
    from local_astra.contracts import commands as C
    from tests.unit import _tool_turn_support as TT

    executor = TT.WatchingExecutor(inner=S.dry_run())
    async with TT.tool_rig(tmp_path, script=[TT.answer("Done. ")], executor=executor) as rig:
        await rig.ask("hello")
        await rig.turn_done()
        await rig.session.executor.execute(
            C.ToolExecute(
                call_id="t1-call", turn_id="t1", name=name, arguments=args, source="draft"
            )
        )
        await rig.pump()
    assert len(executor.invocations) == 1, (
        "T-0092c criterion 5: the ToolExecute handler calls the session"
    )
    msgs = rig.tool_calls("t1-call")
    assert msgs and (msgs[-1].status, msgs[-1].reason) == ("refused", reason), (
        f"T-0092c criterion 3: the executor's refusal is published; got {msgs}"
    )
