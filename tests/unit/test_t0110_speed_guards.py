"""T-0110 guards: the fast suite got faster by not repeating work, and every shortcut is pinned (unit; CPU).

- Part A (criteria 3-4): ``tests/unit/_harness_reading_memo.py`` memoizes the T-0098 harness's ``ui_reading`` on one
  harness object inside the report-building helpers. Rows 1-4 pin that it is installed where it should be and nowhere
  else, that it changes no report, that its key is neither too narrow (row 2: every field but ``request_id`` misses)
  nor too wide (row 3: the premise that ``request_id`` changes no reading, on the real ``ui_reading`` for every
  fixture commit list), and that callers cannot share a mutable reading (row 4).
- Part B (T-0110b; T-0110 criterion 5): the session rigs' pump and the virtual-clock fakes' pollers yield with
  ``asyncio.sleep(0)`` instead of sleeping 0.3-1 ms of wall time. Row 5 records every event-loop timer scheduled from
  code under ``tests/`` during a fake-backend voice turn: none outside ``TIMER_ALLOWED`` (red at ``FORK_SHA``: the pump
  slept 1 ms); its ``selector.select`` timeouts are all zero too, but a spinning poller hides any sleep from that
  coarse check, so it is only the companion. Row 6 is the other
  direction: the real ``PipeWireAudioIO`` on ``FakeAudioDevice`` keeps its real writer-thread waits (T-0057
  criterion 13). Row 7 is a source guard over both sides.
- Part C (criterion 8): ``StubService.close()`` no longer waits for ``serve_forever``'s default 0.5 s poll.

The reviewers' M1-M5 discrimination check (criterion 6) compares the branch against the fork commit
2e182de4fdec40533409cad6c8dd710239f58040 for parts A and C (T-0110), and against ``FORK_SHA`` for part B (T-0110b).

Regime: CPU, pure, the fakes in ``tests/fakes/`` on the virtual clock, the real ``UiGate`` and vendored A2UI v0.9.1
schema at the working tree; row 8 measures wall time on loopback (n = 20 sequential closes, bound 1.0 s in total).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import tests.unit.test_t0098c_ui_prompt_ab as t98c
import tests.unit.test_t0103_c3_confirm as t103
import tests.unit.test_t0108a_ui_fields_ab as t108a
from local_astra.contracts.turn import TurnState
from tests.fakes.http_service import StubService
from tests.unit._harness_reading_memo import ReadingMemo, memoize_ui_reading
from tests.unit._tool_turn_support import SPOKEN, VALID_CLICK, answer, thought, tool, tool_rig
from tests.unit.test_session_loop import ScriptedClock, frame, running
from tests.unit.test_t0098_prompt_ab import (
    ANOMALY,
    ERROR,
    FALSE,
    INEXACT,
    INVALID,
    NONE,
    SILENT,
    SPEAK,
    UNCLOSED,
)
from tests.unit.test_t0098c_ui_prompt_ab import (
    ERROR_C,
    FENCE_THOUGHT,
    OPEN_SURFACE,
    PROSE_UI,
    Q12_TOOL,
    SHOWN,
    SHOWN_FAIL,
    WITH_SURFACE,
)

#: The commit T-0110b branched from after its rebase (``main`` at 68d5566, the T-0110b task file on top of the
#: T-0115 merge). Criterion 6's M1-M5 red sets for part B are compared against this commit, and row 7 compares the
#: real-thread waits with it (``audio_device.py`` and the two contract files are unchanged since T-0110's 2e182de).
#: A literal, never a moving ref (.claude/rules/tests.md).
FORK_SHA = "68d556645448c79f7f4d32295a7413717231edb2"

REPO = Path(__file__).resolve().parents[2]


# =====================================================================================================
# Part A: the harness reading memo
# =====================================================================================================


@contextmanager
def recording_installs(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> Iterator[list[Any]]:
    """Record every ``(harness object, memo)`` the report helper of ``module`` installs."""
    installed: list[Any] = []

    def spy(h: ModuleType) -> ReadingMemo:
        memo = memoize_ui_reading(h)
        installed.append((h, memo))
        return memo

    monkeypatch.setattr(module, "memoize_ui_reading", spy)
    yield installed


def _lines(run: Path) -> list[dict[str, Any]]:
    return [
        json.loads(x) for x in (run / "trials.jsonl").read_text(encoding="utf-8").splitlines() if x
    ]


def _report_103(tmp_path: Path) -> dict[str, Any]:
    return t103.report_103(tmp_path)


def _report_108a(tmp_path: Path) -> dict[str, Any]:
    return t108a.report_108a(tmp_path)


#: table -> (its report helper's module, the helper, the plain harness loader, the baseline run writer)
TABLES: dict[str, tuple[ModuleType, Callable[[Path], dict[str, Any]], Callable[[], ModuleType], Callable[[Path], Path]]] = {
    "T-0098c": (t98c, t98c.report_c, t98c.harness_c, lambda d: t98c.write_run_c(d, t98c.baseline_c())),
    "T-0103": (t103, _report_103, t103.harness_103, lambda d: t103.write_run_103(d, t103.baseline_103())),
    "T-0108a": (t108a, _report_108a, t108a.harness_108a, lambda d: t108a.write_run_108a(d, t108a.baseline_108a())),
}  # fmt: skip


@pytest.mark.parametrize("table", list(TABLES))
def test_memo_report_equals_unmemoized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    """Criteria 3 and 4. The table's report helper, on its baseline fixture, installs exactly one memo, on the harness
    object whose ``build_report`` it calls (every trial line goes through it: ``calls`` = the run's line count, counted
    from the file; ``misses`` = the run's distinct commit lists, counted from the file), and its report is
    ``json.dumps(sort_keys=True)``-equal to ``build_report`` on a fresh, unwrapped harness over the same fixture. The
    memo is not process-global: a harness loaded afterwards has the plain ``ui_reading``."""
    module, helper, loader, write = TABLES[table]
    if module is t108a:
        monkeypatch.setattr(
            t108a, "_REPORTS", {}
        )  # report_108a's own cross-row cache would skip the build
    with recording_installs(monkeypatch, module) as installed:
        memoized = helper(tmp_path / "memo")
    assert len(installed) == 1, (
        f"T-0110 criterion 3: {table}'s report helper installs the memo exactly once; it installed {len(installed)}"
    )
    h, memo = installed[0]
    assert h.ui_reading is memo, (
        "the memo is installed on the harness object build_report looks ui_reading up in"
    )

    plain_h = loader()
    assert not isinstance(plain_h.ui_reading, ReadingMemo), (
        "T-0110 criterion 3: a harness loaded after a report helper ran has a memoized ui_reading -- the memo leaked "
        "past its row (process-global)"
    )
    run = write(tmp_path / "plain")
    plain = plain_h.build_report(run)

    lines = _lines(run)
    distinct = {json.dumps(line["commits"], sort_keys=True) for line in lines}
    assert memo.calls == len(lines), (
        f"T-0110 criterion 3: build_report read {memo.calls} trials through the memo; the run has {len(lines)} lines"
    )
    assert memo.misses == len(distinct) < memo.calls, (
        f"T-0110: {memo.misses} misses for {len(distinct)} distinct commit lists over {memo.calls} calls"
    )
    assert json.dumps(memoized, sort_keys=True) == json.dumps(plain, sort_keys=True), (
        f"T-0110 criterion 4: {table}'s report with the memo differs from the unmemoized report"
    )


def _counting() -> tuple[list[tuple[Any, dict[str, Any]]], Callable[..., dict[str, Any]]]:
    seen: list[tuple[Any, dict[str, Any]]] = []

    def fake(commits: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append((json.loads(json.dumps(commits)), dict(kwargs)))
        return {"call": len(seen), "nested": {"ids": ["a"]}}

    return seen, fake


def _c(
    segment: str = "ui", text: str = "{}", *, open_: bool = False, idx: int = 0
) -> dict[str, Any]:
    return {"segment": segment, "text": text, "thought_open": open_, "canvas_idx": idx}


BASE_COMMITS = [_c("thought", "OK."), _c("ui", '{"a":1}')]

#: Each differs from (BASE_COMMITS, request_id="r1", cards=True) in exactly one thing ``ui_reading`` may read.
MISSES: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
    "thought_open": ([_c("thought", "OK."), _c("ui", '{"a":1}', open_=True)], {"cards": True}),
    "canvas_idx": ([_c("thought", "OK."), _c("ui", '{"a":1}', idx=1)], {"cards": True}),
    "segment": ([_c("thought", "OK."), _c("answer", '{"a":1}')], {"cards": True}),
    "text": ([_c("thought", "OK."), _c("ui", '{"a":2}')], {"cards": True}),
    "thought_text": ([_c("thought", "OK!"), _c("ui", '{"a":1}')], {"cards": True}),
    "order": ([_c("ui", '{"a":1}'), _c("thought", "OK.")], {"cards": True}),
    "one_more_commit": ([*BASE_COMMITS, _c("ui", '{"a":1}', idx=1)], {"cards": True}),
    "cards": (BASE_COMMITS, {"cards": False}),
    "cards_absent": (BASE_COMMITS, {}),
    "unknown_kwarg": (BASE_COMMITS, {"cards": True, "strict": True}),
}  # fmt: skip


def test_memo_key_separates_every_field_but_request_id() -> None:
    """Adversarial (a memo too wide reads one trial's surface for another's). With a counting fake: a different
    ``request_id`` alone is a hit (same reading, no call); each of a different ``thought_open``, ``canvas_idx``,
    ``segment``, text, commit order or count, ``cards`` value, absent ``cards`` and an unknown keyword is a miss that
    reaches the fake with exactly its own arguments."""
    seen, fake = _counting()
    memo = ReadingMemo(fake)
    first = memo(BASE_COMMITS, request_id="r1", cards=True)
    again = memo(BASE_COMMITS, request_id="a-wholly-different-id", cards=True)
    assert len(seen) == 1 and again == first, (
        f"T-0110: a different request_id alone must hit the memo; the fake ran {len(seen)} times"
    )
    for why, (commits, kwargs) in MISSES.items():
        before = len(seen)
        got = memo(commits, request_id="r1", **kwargs)
        assert len(seen) == before + 1, f"T-0110: a different {why} must miss the memo; it hit"
        assert seen[-1] == (commits, {"request_id": "r1", **kwargs}), (why, seen[-1])
        assert got == {"call": before + 1, "nested": {"ids": ["a"]}}, (why, got)
    assert (memo.calls, memo.misses) == (2 + len(MISSES), 1 + len(MISSES))


def test_memo_is_installed_on_one_object_only() -> None:
    """Criterion 3: ``memoize_ui_reading`` wraps the one object it is given; a second harness object keeps its own
    ``ui_reading`` and two installs never share a cache."""
    seen, fake = _counting()
    a, b = ModuleType("a"), ModuleType("b")
    a.ui_reading = fake  # type: ignore[attr-defined]
    b.ui_reading = fake  # type: ignore[attr-defined]
    memo_a = memoize_ui_reading(a)
    assert a.ui_reading is memo_a and b.ui_reading is fake
    memo_c = memoize_ui_reading(b)
    a.ui_reading(BASE_COMMITS, request_id="x")
    b.ui_reading(BASE_COMMITS, request_id="x")
    assert len(seen) == 2 and memo_a is not memo_c, "two harness objects must not share one memo"


OUTCOMES = (
    SPEAK, SILENT, UNCLOSED, NONE, FALSE, ANOMALY, INVALID, INEXACT, ERROR,
    SHOWN, OPEN_SURFACE, FENCE_THOUGHT, PROSE_UI, SHOWN_FAIL, Q12_TOOL, WITH_SURFACE, ERROR_C,
    t108a.COMPLETE, t108a.PRUNED, t108a.FAILED_LAST,
)  # fmt: skip


def _fixture_commit_lists() -> dict[str, list[dict[str, Any]]]:
    """Every commit list the T-0098c / T-0103 / T-0108a run fixtures can write: each outcome of the fixture
    vocabulary through both commit builders, on every case row (an outcome a builder refuses for a row is skipped)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for build in (t98c.commits_for_c, t108a.commits_108a):
        for row in t98c.case_rows_c():
            for outcome in OUTCOMES:
                try:
                    commits = build(outcome, row)
                except (AssertionError, KeyError, TypeError):
                    continue
                out.setdefault(json.dumps(commits, sort_keys=True), commits)
    return out


def _readers() -> list[tuple[str, Callable[..., Any]]]:
    readers = [("working tree", t98c.harness_c().ui_reading)]
    for name, module in (("T-0103 oracle", t103), ("T-0108a oracle", t108a)):
        try:
            readers.append((f"{name} at {module.BASE_SHA[:7]}", module.old_harness().ui_reading))
        except pytest.skip.Exception:
            continue  # the frozen oracle's commit is not in this clone
    return readers


def test_ui_reading_ignores_request_id_on_every_fixture_commit_list(tmp_path: Path) -> None:
    """The memo's premise (it guards against a memo that is too wide; row 2 guards against one that is absent). The
    enumeration covers the three baseline runs' commit lists (checked from their trials.jsonl), and on every
    enumerated list the real ``ui_reading`` -- the working tree's and each frozen oracle's that a memo is installed
    on -- returns the same reading for two different request ids, with every ``cards`` value it accepts."""
    lists = _fixture_commit_lists()
    for table, (_m, _h, _l, write) in TABLES.items():
        missing = {
            json.dumps(line["commits"], sort_keys=True) for line in _lines(write(tmp_path / table))
        } - set(lists)
        assert not missing, (
            f"T-0110: the enumeration misses {len(missing)} of {table}'s baseline commit lists"
        )
    assert sum(any(c.get("segment") == "ui" for c in v) for v in lists.values()) >= 10, (
        "premise: the enumeration gates real surfaces"
    )
    for name, reader in _readers():
        cards_options: list[dict[str, Any]] = [{}]
        if "cards" in inspect.signature(reader).parameters:
            cards_options = [{"cards": True}, {"cards": False}]
        for commits in lists.values():
            for kw in cards_options:
                a = reader(commits, request_id="t0110-r0-flight_booking-A0", **kw)
                b = reader(commits, request_id="zz", **kw)
                assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (
                    f"T-0110 premise broken ({name}, {kw}): ui_reading depends on request_id for {commits!r:.300}"
                )


def test_memo_returns_independent_copies() -> None:
    """Every call returns its own deep copy: mutating one reading (top level or nested) changes neither the next
    caller's reading nor the memo's."""
    seen, fake = _counting()
    memo = ReadingMemo(fake)
    one = memo(BASE_COMMITS, request_id="a")
    one["call"] = 99
    one["nested"]["ids"].append("mutated")
    two = memo(BASE_COMMITS, request_id="b")
    assert len(seen) == 1 and two == {"call": 1, "nested": {"ids": ["a"]}}, two
    assert two is not one and two["nested"] is not one["nested"]
    assert memo(BASE_COMMITS, request_id="c") is not two


# =====================================================================================================
# Part B: the zero-wait rig
# =====================================================================================================


@contextmanager
def select_timeouts() -> Iterator[list[float | None]]:
    """Record every timeout the running event loop passes to its selector (the loop's only way to wait)."""
    loop = asyncio.get_running_loop()
    selector = getattr(loop, "_selector", None)
    assert selector is not None, f"premise: a selector event loop; got {type(loop).__name__}"
    real = selector.select
    seen: list[float | None] = []

    def recording(timeout: float | None = None) -> Any:
        seen.append(timeout)
        return real(timeout)

    selector.select = recording
    try:
        yield seen
    finally:
        del selector.select


_ASYNCIO_DIR = Path(asyncio.__file__).resolve().parent
TESTS_DIR = REPO / "tests"


@contextmanager
def rig_timers() -> Iterator[list[tuple[str, int, str, float]]]:
    """Record ``(file, line, qualified name, delay s)`` for every timer the running loop is asked to schedule from
    test code. ``asyncio.sleep(d)`` with ``d > 0``, ``call_later`` and ``wait_for``/``timeout`` all reach
    ``loop.call_at``; ``asyncio.sleep(0)`` never does. The origin is the innermost frame outside the ``asyncio``
    package, and only origins under ``tests/`` are kept (a timer the session itself arms is not the rig's)."""
    loop = asyncio.get_running_loop()
    real = loop.call_at
    seen: list[tuple[str, int, str, float]] = []

    def recording(when: float, callback: Any, *args: Any, context: Any = None) -> Any:
        f = sys._getframe(1)
        while f is not None and Path(f.f_code.co_filename).resolve().is_relative_to(_ASYNCIO_DIR):
            f = f.f_back
        if f is not None:
            path = Path(f.f_code.co_filename).resolve()
            if path.is_relative_to(TESTS_DIR):
                qual = f.f_code.co_qualname.replace(".<locals>", "")
                seen.append(
                    (path.relative_to(REPO).as_posix(), f.f_lineno, qual, when - loop.time())
                )
        return real(when, callback, *args, context=context)

    loop.call_at = recording  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        del loop.call_at


async def test_fake_rig_never_blocks_on_wall_time(tmp_path: Path) -> None:
    """Criterion 5 (red at the fork: ``Rig.pump`` slept 1 ms and the fakes' pollers 0.3-0.5 ms). During a whole
    fake-backend ``tool_rig`` voice turn (ask, then the turn's end), no code under ``tests/`` schedules a single
    event-loop timer, outside the real-thread waits in ``TIMER_ALLOWED``: no rig, fake or helper sleeps on wall time
    (``asyncio.sleep(d > 0)``, ``call_later``, ``call_at`` or a ``wait_for`` timeout). This is the assertion with
    teeth. The fakes' pollers spin with ``asyncio.sleep(0)`` for the whole turn, so the ready queue is never empty and
    the selector timeout is 0 whatever else sleeps; the selector check below is kept only as the coarse companion
    (it catches a wall-time wait from code outside ``tests/`` that leaves the loop idle). The turn is real: the
    answer is voiced and the click is dry-run."""
    script = [thought("the user wants the button"), answer(SPOKEN), tool(VALID_CLICK, canvas_idx=0)]
    async with tool_rig(tmp_path, script=script) as rig:
        with select_timeouts() as seen, rig_timers() as timers:
            asyncio.get_running_loop().call_later(0, lambda: None).cancel()
            canary = timers.pop() if timers else None
            assert canary is not None and canary[0] == "tests/unit/test_t0110_speed_guards.py", (
                f"premise: the recorder sees a timer scheduled from test code ({canary})"
            )
            await rig.ask("click the sign in button")
            await rig.turn_done()
        assert "I will click the Sign in button." in rig.tts.texts(), rig.tts.texts()
    assert rig.statuses() == ["proposed", "dry_run"], rig.statuses()
    assert len(seen) > 100, f"premise: the turn ran the loop ({len(seen)} selects)"
    offenders = sorted({(p, line, q) for p, line, q, _d in timers if (p, q) not in TIMER_ALLOWED})
    assert not offenders, (
        f"T-0110 criterion 5: test code scheduled {len(timers)} event-loop timers during a fake voice turn, "
        f"{len(offenders)} distinct sites outside TIMER_ALLOWED: {offenders[:8]} (delays s: "
        f"{sorted({round(d, 6) for *_x, d in timers})[:8]}); a rig, fake or helper sleeps on wall time"
    )
    waits = sorted({t for t in seen if t != 0}, key=lambda t: (t is None, t or 0.0))
    assert not waits, (
        f"T-0110 criterion 5: {sum(t != 0 for t in seen)} of {len(seen)} selector waits during a fake voice turn were "
        f"not zero (distinct: {waits[:8]}); a rig or fake poller still sleeps on wall time"
    )


async def test_real_module_rig_still_gets_wall_time(tmp_path: Path) -> None:
    """Criterion 5, the other direction (T-0057 criterion 13): the session over the real ``PipeWireAudioIO`` on
    ``FakeAudioDevice`` (``test_session_audio_seam[module]``'s rig, ``settle_audio`` = ``audio_device.settle``) still
    waits on wall time for the module's writer, and the turn completes with every committed source played."""
    from tests.fakes.audio_device import FakeAudioDevice, make_audio_io, settle

    clock = ScriptedClock()
    device = FakeAudioDevice(clock, buffer_ms=200.0)
    audio = make_audio_io(device)

    async def pump() -> None:
        await settle(device)

    sources = ("First sentence. ", "Second one. ")
    async with running(
        tmp_path,
        script=[answer(t) for t in sources],
        frames=[frame(0, "a")],
        clock=clock,
        audio=audio,
        settle_audio=pump,
        ms_per_word=100,
    ) as rig:
        with select_timeouts() as seen:
            await rig.ask("what is on this screen")
            await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        assert rig.state.history[-1].assistant_spoken.strip(), rig.state.history[-1]
    real_waits = [t for t in seen if t is not None and t > 0]
    assert real_waits, (
        "T-0110 criterion 5: the [module] rig no longer waits on wall time; PipeWireAudioIO's writer is a real "
        "thread or task and audio_device.settle must keep its real waits"
    )


#: The session-rig pump and the virtual-clock fakes' pollers: they must yield with ``asyncio.sleep(0)`` (part B).
ZERO_WAIT_SITES: dict[str, set[str]] = {
    "tests/unit/test_session_loop.py": {"Rig.pump"},
    "tests/unit/_hud_rig.py": {"Rig.pump"},
    "tests/unit/test_session_listening.py": {
        "OddFrameAudio.stt_blocks",
        "test_close_capture_is_called_once_after_the_capture_pump_is_cancelled",
    },
    "tests/fakes/stt.py": {"FakeStt._live", "FakeStt._replay"},
    "tests/fakes/audio.py": {
        "FakeAudioIO.playback_events",
        "FakeAudioIO.capture_frames",
        "FakeAudioIO.stt_blocks",
    },
}
#: Real-thread waits that stay real (T-0057 criterion 13), with their files.
REAL_WAIT_FILES = (
    "tests/fakes/audio_device.py",
    "tests/contract/test_audio_capture_conformance.py",
    "tests/contract/test_audio_late_consumer.py",
)
REAL_WAIT_SITES = {
    ("tests/fakes/audio_device.py", "EventSink.sync"),
    ("tests/fakes/audio_device.py", "settle"),
    ("tests/fakes/audio_device.py", "capture_settle"),
    ("tests/contract/test_audio_capture_conformance.py", "settle_sinks"),
    ("tests/contract/test_audio_late_consumer.py", "_build.advance_clock"),
}
#: ``(file, qualified name)`` of the test-code timers row 5 tolerates: exactly the real-thread waits row 6 exercises
#: and row 7 pins against the fork (T-0057 criterion 13), and nothing else. Matched on the scheduling frame's own
#: code object, so a wall-time sleep anywhere else in the same files (``audio_device.py`` included) is still caught.
TIMER_ALLOWED = frozenset(REAL_WAIT_SITES)


def asyncio_sleeps(source: str) -> list[tuple[str, int, str]]:
    """``(enclosing qualified name, line, argument)`` for every ``asyncio.sleep(...)`` *call* in the code. Parsed with
    ``ast``, so strings and comments never count."""
    out: list[tuple[str, int, str]] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                walk(child, [*stack, child.name])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "sleep"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "asyncio"
            ):
                arg = ast.unparse(child.args[0]) if child.args else "<none>"
                out.append((".".join(stack), child.lineno, arg))
            walk(child, stack)

    walk(ast.parse(source), [])
    return out


def _git_show(path: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "show", f"{FORK_SHA}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


HINT = (
    " (a source guard: it parses the code, so strings and comments do not count, but any asyncio.sleep call in these "
    "files does -- a new helper that legitimately waits on wall time belongs in REAL_WAIT_SITES, with a reason)"
)


def test_every_rig_poller_uses_the_shared_yield() -> None:
    """Criterion 5's source half. In the rig and fake-poller files, every ``asyncio.sleep`` is the shared yield
    ``asyncio.sleep(0)``, and every named site still yields (a poller that stopped yielding would spin the loop). The
    real-thread waits keep a non-zero wall wait, exactly as at ``FORK_SHA``."""
    offenders: list[str] = []
    for path, sites in ZERO_WAIT_SITES.items():
        found = asyncio_sleeps((REPO / path).read_text(encoding="utf-8"))
        offenders += [
            f"{path}:{line} {where}: asyncio.sleep({arg})"
            for where, line, arg in found
            if arg != "0"
        ]
        yielding = {where for where, _line, arg in found if arg == "0"}
        offenders += [
            f"{path} {site}: no asyncio.sleep(0) left" for site in sorted(sites - yielding)
        ]
    assert not offenders, (
        "T-0110 part B: a rig or fake poller waits on wall time:\n  "
        + "\n  ".join(offenders)
        + HINT
    )

    def real_sites(read: Callable[[str], str | None]) -> set[tuple[str, str, str]]:
        out: set[tuple[str, str, str]] = set()
        for path in REAL_WAIT_FILES:
            src = read(path)
            assert src is not None, path
            out |= {
                (path, where, arg)
                for where, _line, arg in asyncio_sleeps(src)
                if (path, where) in REAL_WAIT_SITES
            }
        return out

    now = real_sites(lambda p: (REPO / p).read_text(encoding="utf-8"))
    assert {(p, w) for p, w, _a in now} == REAL_WAIT_SITES, sorted(now)
    assert all(arg != "0" for _p, _w, arg in now), (
        f"T-0057 criterion 13: a real-thread wait became a yield: {now}"
    )
    if _git_show(REAL_WAIT_FILES[0]) is None:
        pytest.skip(f"{FORK_SHA[:7]} is not in this clone; the fork comparison is unavailable")
    assert now == real_sites(_git_show), (
        "T-0110 criterion 5: the real-thread waits moved since the fork"
    )


# =====================================================================================================
# Part C: the stub service closes fast
# =====================================================================================================


def test_stub_close_is_fast() -> None:
    """Criterion 8: 20 sequential ``StubService`` start-and-``close()`` cycles take < 1.0 s of wall time in total (at
    the fork each ``close()`` waited up to ``serve_forever``'s 0.5 s poll), and a closed stub's port refuses."""
    import urllib.error
    import urllib.request

    t0 = time.perf_counter()
    for _ in range(20):
        stub = StubService({"/health": (200, "ok")})
        stub.close()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, (
        f"T-0110 criterion 8: 20 sequential StubService closes took {elapsed:.2f} s (bound 1.0 s)"
    )
    with StubService({"/health": (200, "ok")}) as live:
        with urllib.request.urlopen(f"{live.base}/health", timeout=5) as resp:
            assert (resp.status, resp.read()) == (200, b"ok")
        base = live.base
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"{base}/health", timeout=2)
