"""T-0101 criterion 4 (H-T101a): ``cli.run_session`` awaits the repair warm-up before ``graph.session.start()``.

The graph is duck-typed (as ``tests/unit/test_cli_session.py`` serves ``run_session`` one): a recording session that
holds a ``FakeLlm`` as ``llm`` and the kit as ``tool_kit``, and a server whose ``start`` says "serving". Every call
lands in one ordered event list, so "before" means before in that list -- a warm-up started as a background task and
not awaited leaves ``session.start`` ahead of every warm-up request, and fails.

The warm-up itself is the real ``local_astra.toolcall.warmup.warm_repair`` over the fake (its surface is stated in
``tests/unit/test_repair_warmup.py``); ``run_session`` is driven until the server serves, then cancelled (the path
SIGINT takes).

Regime: CPU, asyncio, ``FakeLlm``; real payloads and calibration; no engine, no socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from local_astra import cli
from local_astra.conductor.session import ToolKit
from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.llm import Commit, LlmOptions, Prompt
from local_astra.toolcall import live, native
from local_astra.toolcall import tools as toolset
from tests.fakes.llm import FakeLlm, http_error

TOOLS = sorted(live.load_payloads().tools)
LINE_PREFIX = "astra session: repair warm-up: "
VALID_OUT = 'call:press_key{combo:<|"|>Enter<|"|>}<tool_call|>'


class _Llm(FakeLlm):
    """``FakeLlm`` that also writes each generate's start and end, and each tokenize, into the shared event list."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.script_continuation(FakeLlm.tool_commit(VALID_OUT, stop_id=50))

    async def generate(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: Any = None,
    ) -> AsyncIterator[Commit]:
        self.events.append(f"generate:{request_id}")
        try:
            async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
                yield c
        finally:
            self.events.append(f"generate.end:{request_id}")

    async def tokenize(self, text: str) -> list[int]:
        self.events.append("tokenize")
        return await super().tokenize(text)


class _Audio:
    def stt_blocks(self) -> None:  # the seam run_session checks for (T-0032 criterion 9(a))
        return None


class _Session:
    def __init__(self, llm: _Llm, kit: ToolKit | None, *, with_kit: bool = True) -> None:
        self.llm = llm
        if with_kit:
            self.tool_kit = kit
        self.audio = _Audio()
        self.session_log = None
        self.events = llm.events

    async def start(self) -> None:
        self.events.append("session.start")

    async def aclose(self) -> None:
        self.events.append("session.aclose")


class _Server:
    url = "ws://127.0.0.1:0/ws"
    port = 0

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.serving = asyncio.Event()

    async def start(self) -> None:
        self.events.append("server.start")
        self.serving.set()

    async def aclose(self) -> None:
        self.events.append("server.aclose")


class _Graph:
    def __init__(self, session: _Session, server: _Server) -> None:
        self.session = session
        self.server = server
        self.frames = None
        self.silent = False

    def describe(self) -> str:
        return "warm-up graph"


def _kit(*, repair_off: str | None = None, payloads: bool = True) -> ToolKit:
    tool_set = toolset.load()
    return ToolKit(
        mode="dry_run",
        tools=tuple(native.tools_field(tool_set)),
        tools_version=tool_set.version,
        executor=DryRunToolExecutor(),
        payloads=live.load_payloads() if payloads else None,
        calibration=live.load_calibration() if payloads else None,
        repair_off=repair_off,
    )


def _serve(
    kit: ToolKit | None, *, with_kit: bool = True, llm_setup: Any = None
) -> tuple[list[str], _Llm, _Graph]:
    events: list[str] = []
    llm = _Llm(events)
    if llm_setup is not None:
        llm_setup(llm)
    session = _Session(llm, kit, with_kit=with_kit)
    server = _Server(events)
    graph = _Graph(session, server)

    async def drive() -> None:
        task = asyncio.ensure_future(cli.run_session(graph))  # type: ignore[arg-type]
        waiter = asyncio.ensure_future(server.serving.wait())
        await asyncio.wait({task, waiter}, timeout=20, return_when=asyncio.FIRST_COMPLETED)
        waiter.cancel()
        if task.done():
            task.result()  # run_session ended on its own: surface why
        assert server.serving.is_set(), f"run_session never served; events {events}"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    return events, llm, graph


def _warm_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(LINE_PREFIX)]


def test_warmup_completes_before_session_start() -> None:
    """Adversarial: every warm-up request (one per payload tool) has started *and ended* before ``session.start`` --
    a fire-and-forget background task puts ``session.start`` first and fails here."""
    events, llm, _ = _serve(_kit())
    assert "session.start" in events, events
    start = events.index("session.start")
    begun = [i for i, e in enumerate(events) if e.startswith("generate:")]
    ended = [i for i, e in enumerate(events) if e.startswith("generate.end:")]
    assert len(begun) == len(TOOLS) == 5, (
        f"criterion 4: one warm-up request per tool; events {events}"
    )
    assert all(rid.startswith("warmup-") for rid, _, _ in llm.generates)
    assert max(begun + ended) < start, (
        f"criterion 4: the warm-up must finish before session.start; {events}"
    )
    assert len(ended) == 5


def _assert_control_warms(capsys: pytest.CaptureFixture[str]) -> None:
    """The positive control each negative row carries: the same rig with a full kit warms all five tools and prints
    one line. Without it a negative row passes against a ``run_session`` that never warms anything (today's)."""
    _, llm, _ = _serve(_kit())
    assert len(llm.generates) == 5, "control: a tools-on, repair-on kit is warmed (criterion 4)"
    assert len(_warm_lines(capsys.readouterr().out)) == 1, "control: one warm-up line"


def _assert_no_warmup(events: list[str], llm: _Llm, out: str, what: str) -> None:
    assert llm.generates == [] and llm.tokenized == [], (
        f"criterion 4: no warm-up request for {what}"
    )
    assert "session.start" in events, f"{what}: the session still starts"
    assert _warm_lines(out) == [], f"criterion 4: no warm-up line for {what}"


def test_no_warmup_when_tools_off(capsys: pytest.CaptureFixture[str]) -> None:
    """Tools off: the session's ``tool_kit`` is ``None``."""
    _assert_control_warms(capsys)
    events, llm, _ = _serve(None)
    _assert_no_warmup(events, llm, capsys.readouterr().out, "tools off")


@pytest.mark.parametrize(
    "case",
    ["as_load_tool_kit_builds_it", "repair_off_with_payloads", "repair_on_without_payloads"],
)
def test_no_warmup_when_repair_off(case: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Repair off as ``cli.load_tool_kit`` builds it (a reason, no payloads, no calibration); a kit whose
    ``repair_off`` is set although payloads are present (the gate reads ``repair_off``, not only the payloads); and a
    kit with no payloads (the gate reads the payloads, not only ``repair_off``)."""
    _assert_control_warms(capsys)
    kit = {
        "as_load_tool_kit_builds_it": _kit(repair_off="payload stale", payloads=False),
        "repair_off_with_payloads": _kit(repair_off="payload stale"),
        "repair_on_without_payloads": _kit(payloads=False),
    }[case]
    events, llm, _ = _serve(kit)
    _assert_no_warmup(events, llm, capsys.readouterr().out, case)


def test_no_warmup_without_kit_attribute(capsys: pytest.CaptureFixture[str]) -> None:
    """A duck-typed session with no ``tool_kit`` attribute at all: no warm-up, no line, no error, the session starts."""
    _assert_control_warms(capsys)
    events, llm, _ = _serve(_kit(), with_kit=False)
    _assert_no_warmup(events, llm, capsys.readouterr().out, "a session without tool_kit")


def test_warmup_failure_still_starts_session(capsys: pytest.CaptureFixture[str]) -> None:
    """Every warm-up request ends 500: the session still starts, the kit is the same object with repair still on, and
    the line reports each tool failed (H-T101a: a failed warm-up never blocks the session)."""
    kit = _kit()

    def fail_all(llm: _Llm) -> None:
        llm.continuation_errors = [http_error(500)] * 5

    events, llm, graph = _serve(kit, llm_setup=fail_all)
    assert len(llm.generates) == 5, "every tool is still tried"
    assert "session.start" in events
    assert graph.session.tool_kit is kit and kit.repair_off is None, (
        "criterion 3: repair is not turned off"
    )
    lines = _warm_lines(capsys.readouterr().out)
    assert len(lines) == 1, lines
    for tool in TOOLS:
        assert re.search(rf"\b{tool} \d+(\.\d+)? ms failed", lines[0]), (tool, lines[0])


def test_warmup_stdout_line(capsys: pytest.CaptureFixture[str]) -> None:
    """Criterion 4: exactly one line ``astra session: repair warm-up: <tool> <ms> ms <status>, ..., total <ms> ms``,
    tools in name order, each ``ok`` here."""
    _serve(_kit())
    lines = _warm_lines(capsys.readouterr().out)
    assert len(lines) == 1, lines
    line = lines[0]
    assert re.search(r", total \d+(\.\d+)? ms$", line), line
    positions = []
    for tool in TOOLS:
        m = re.search(rf"\b{tool} \d+(\.\d+)? ms ok\b", line)
        assert m, (tool, line)
        positions.append(m.start())
    assert positions == sorted(positions), f"tools out of name order: {line}"
