"""T-0173 criteria 1-6, session side: playback terminals reach the machine on the session clock (P10).

docs/tasks/T-0173.md. W2-2 (``logs/t0160-web-smoke-abcf17d/turns/W2-2``): ``PipeWireAudioIO`` stamps
``playback.drained`` with the device's rendered position (4 760) while the session clock read 22 420, so R18 was
armed 2.7 s in the past and the next Tick closed the turn with ``hud.error "tool timeout"`` 23 ms after
``tool.execute``.

**How the lag is made.** The real ``PipeWireAudioIO`` over ``tests/fakes/audio_device.FakeAudioDevice`` on the rig's
``ScriptedClock`` (the T-0007 ``[module]`` rig, ``settle_audio = audio_device.settle``). The device renders only what it
is given, so its position does not move while nothing plays; every lag row first advances the session clock
``IDLE_LAG_MS`` (18 s) with the session idle and no audio, then runs the turn. ``tests/fakes/audio.py`` is not used for
the lag (it stamps the session clock, which is why nothing caught W2-2); it is used only by the zero-lag row.

Oracles, none of them the session's own bookkeeping:

* *when a terminal arrived* -> ``ReceiptAudio``: a delegating wrapper over the module that records
  ``clock.now_ms`` at the instant ``playback_events()`` hands each terminal to the session's pump (no await between
  the record and the hand-over);
* *what the machine was stepped with* -> ``RecordingMachine``'s log, and the session-log ``machine.step`` records
  (``session_ms`` is the session clock at the step), read back after ``aclose``;
* *R18 / the execute deadline* -> the constants written out from the task (15 000, 10 000) and the tick cadence
  (50 ms), never read from the machine's state.

Rows that need the restamp were written ``pending(task="T-0173")`` and activated at /implement T-0173 (they fail
until the restamp exists); the zero-lag row holds on the base.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), ``TurnPolicy(thinking=True, tool_loop=True)`` (8 steps),
tools ``dry_run`` (``_tools_switch_support.make_kit``), FakeTts 100 ms per word, typed turns, n = 1, deterministic per
row apart from the module's writer thread (real-time settle, as every ``[module]`` session row).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.fakes.audio_device import FakeAudioDevice, make_audio_io, settle
from tests.unit._tool_turn_support import WatchingExecutor, answer, tool, tool_rig
from tests.unit.test_session_loop import POLICY, RecordingAudio

#: R18 (T-0112) and the session's execute deadline (T-0114 ``TOOL_EXEC_TIMEOUT_MS``), written out from the task.
R18_MS = 15_000
EXEC_DEADLINE_MS = 10_000
#: The session tick cadence (``session.TICK_INTERVAL_MS``): "within one Tick" (criterion 2).
TICK_MS = 50
#: Criterion 6's bound.
P10_MS = 1_000
#: Silence before the turn: the device position stays at 0 while the session clock runs (criterion 2: lag >= 17 s).
IDLE_LAG_MS = 18_000
#: W2-2 shape: the first canvas arrives >= 15 s after ``llm.start``.
FIRST_CANVAS_MS = 15_500
MS_PER_WORD = 100

LOOP = TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=8)
QUESTION = "Go to example.com and read me the page heading."
PREFIX = "I will type example.com into the web address bar."
Q = '<|"|>'
TYPE_AT = (
    f"<|tool_call>call:type_at{{press_enter:true,text:{Q}example.com{Q},x:486,y:452}}<tool_call|>"
)
REPORT = "The heading is Example Domain."
LONG = "This answer is long enough to still be playing when the user cuts in on it. "
TERMINALS = (E.PlaybackStopped, E.PlaybackDrained)


# --- the audio: the real module over the fake device, plus receipt times ----------------------------------------


class ReceiptAudio:
    """Delegates everything to ``inner``; records ``(event, clock.now_ms)`` as each terminal is handed over."""

    def __init__(self, inner: Any, clock: Any, device: Any = None) -> None:
        self.inner = inner
        self.clock = clock
        self.device = device
        self.receipts: list[tuple[Any, int]] = []
        #: the device's written position once everything played so far has been handed over (rig pacing only)
        self.owed_ms = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        written = float(self.device.written_ms) if self.device is not None else 0.0
        self.owed_ms = max(self.owed_ms, written) + float(getattr(utterance, "duration_ms", 0) or 0)
        await self.inner.play(utterance, source=source)

    async def pause(self) -> None:
        await self.inner.pause()

    async def resume(self) -> None:
        await self.inner.resume()

    async def flush(self) -> int:
        self.owed_ms = 0.0  # the module drops what it had not written
        return int(await self.inner.flush())

    def mark_drain(self) -> None:
        self.inner.mark_drain()

    def played_ms(self) -> float:
        return float(self.inner.played_ms())

    async def playback_events(self) -> AsyncIterator[Any]:
        async for event in self.inner.playback_events():
            self.receipts.append((event, self.clock.now_ms))
            yield event


class FakeReceiptAudio(RecordingAudio):
    """The zero-lag row: ``FakeAudioIO`` (session-clock stamps), with the same receipt log."""

    def __init__(self, clock: Any) -> None:
        super().__init__(clock)
        self.receipts: list[tuple[Any, int]] = []

    async def playback_events(self) -> AsyncIterator[Any]:
        async for event in super().playback_events():
            self.receipts.append((event, self.clock.now_ms))
            yield event


class GatedExecutor(WatchingExecutor):
    """Answers only when the row opens the gate (or never): the 2 s executor of criterion 2."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def execute(self, inv: Any) -> Any:
        self.invocations.append(inv)
        if self.clock is not None:
            self.times.append(self.clock.now_ms)
        await self.gate.wait()
        from local_astra.contracts.toolcall import ToolOutcome

        return ToolOutcome(
            call_id=inv.call_id,
            status="dry_run",
            action=f"recorded {inv.name}",
            reason=None,
            resolved={},
        )


def lagged_audio(holder: dict[str, Any]) -> Any:
    def build(clock: Any) -> ReceiptAudio:
        device = FakeAudioDevice(clock, buffer_ms=200.0)
        holder["device"] = device
        audio = ReceiptAudio(make_audio_io(device), clock, device)
        holder["audio"] = audio
        return audio

    return build


async def with_device(rig: Any, holder: dict[str, Any]) -> None:
    """The ``[module]`` rig's real-time settle, spent only while there is audio to move: some of what was played is
    not yet written to the device, or the device still holds some. Idle and thinking stretches (most of a lag row's
    virtual time, with nothing to render) then cost no wall time. A row where this starves the writer only renders
    more slowly in virtual time; no assertion here depends on the audio's pace."""
    device, audio = holder["device"], holder["audio"]

    async def pump() -> None:
        if audio.owed_ms > device.written_ms + 0.5 or device.queued_ms() > 0.0:
            await settle(device)

    rig.settle_audio = pump


# --- driving --------------------------------------------------------------------------------------------------


async def idle_lag(rig: Any) -> None:
    await rig.advance(IDLE_LAG_MS, step=1_000)
    assert rig.state.state is TurnState.IDLE


async def type_question(rig: Any, text: str = QUESTION) -> None:
    assert rig.state.state is TurnState.IDLE
    rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=text))
    await rig.pump()


async def slow_first_canvas(rig: Any) -> int:
    """Hold the generate before its first commit for ``FIRST_CANVAS_MS`` after ``llm.start``; return llm.start."""
    rig.llm.hold()
    await type_question(rig)
    await rig.until(
        lambda: rig.state.state is TurnState.THINKING, "the typed turn never reached THINKING"
    )
    start = rig.state.llm_start_ms
    assert start is not None
    await rig.advance(FIRST_CANVAS_MS, step=500)
    assert rig.state.state is TurnState.THINKING, (
        f"premise: still thinking at +15 s; {rig.state.state}"
    )
    rig.llm.release()
    return int(start)


def records(rig: Any, kind: str = "machine.step") -> list[dict[str, Any]]:
    return [r for r in rig.log_records if r.get("kind") == kind]


def hud_errors(rig: Any) -> list[str]:
    return [c.message for c in rig.machine.commands(C.HudError)]


def step_records_with(rig: Any, command_type: str, **match: Any) -> list[dict[str, Any]]:
    out = []
    for r in records(rig):
        for c in r["data"]["commands"]:
            if c.get("type") == command_type and all(c.get(k) == v for k, v in match.items()):
                out.append(r)
    return out


async def w2_2_turn(tmp_path: Path, *, answer_after_ms: int | None = 2_000) -> dict[str, Any]:
    """Criterion 2's scenario. ``answer_after_ms=None``: the executor never answers (criterion 3, session half)."""
    holder: dict[str, Any] = {}
    executor = GatedExecutor()
    out: dict[str, Any] = {}
    async with tool_rig(
        tmp_path,
        policy=LOOP,
        script=[answer(PREFIX), tool(TYPE_AT)],
        executor=executor,
        audio_cls=lagged_audio(holder),
        ms_per_word=MS_PER_WORD,
    ) as rig:
        await with_device(rig, holder)
        rig.llm.script_steps([answer(REPORT)])
        await idle_lag(rig)
        out["llm_start"] = await slow_first_canvas(rig)
        await rig.until(
            lambda: bool(executor.invocations) or rig.state.state is TurnState.IDLE,
            "the call never executed and the turn never ended",
            limit_ms=20_000,
            step=10,
        )
        assert executor.invocations, (
            f"premise: the prefix drains and the call executes; turn ended first, hud.error {hud_errors(rig)}"
        )
        out["exec_ms"] = executor.times[0]
        if answer_after_ms is not None:
            await rig.advance(answer_after_ms, step=100)
            executor.gate.set()
        else:
            await rig.advance(EXEC_DEADLINE_MS + 200, step=250)
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "the turn never ended",
            limit_ms=30_000,
            step=20,
        )
        out["hud_errors"] = hud_errors(rig)
        out["transitions"] = rig.machine.transitions()
        out["log"] = rig.machine.log
        out["receipts"] = list(holder["audio"].receipts)
    out["records"] = records(rig)
    out["rig"] = rig
    return out


#: Criterion 2's scenario, run once per module (per xdist worker under ``--dist loadfile``) and read by every row
#: that only observes it: the rows assert on the recorded logs and never drive the rig themselves.
_SHARED: dict[str, dict[str, Any]] = {}


async def shared_w2_2_turn(factory: pytest.TempPathFactory) -> dict[str, Any]:
    if "w2_2" not in _SHARED:
        _SHARED["w2_2"] = await w2_2_turn(factory.mktemp("t0173_w2_2"))
    return _SHARED["w2_2"]


# =====================================================================================================
# criterion 1: the restamp
# =====================================================================================================


async def test_playback_terminal_is_restamped_on_the_session_clock(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Criterion 1: every terminal stepped carries the session time it arrived at; type, played_ms and played_text
    are the module's own."""
    out = await shared_w2_2_turn(tmp_path_factory)
    receipts = out["receipts"]
    stepped = [ev for ev, _, _ in out["log"] if isinstance(ev, TERMINALS)]
    assert receipts and len(stepped) == len(receipts), (
        f"premise: every terminal the module delivered reached the machine; delivered {len(receipts)}, "
        f"stepped {len(stepped)}"
    )
    for (module_ev, arrived), ev in zip(receipts, stepped, strict=True):
        assert (ev.type, ev.played_ms, ev.played_text) == (
            module_ev.type,
            module_ev.played_ms,
            module_ev.played_text,
        ), f"criterion 1: only t_ms is restamped; module {module_ev}, stepped {ev}"
        assert ev.t_ms == arrived, (
            f"criterion 1 / P10: {ev.type} must carry the session clock at arrival ({arrived}); it carries "
            f"{ev.t_ms} (the module's device stamp was {module_ev.t_ms})"
        )


# =====================================================================================================
# criterion 2: the W2-2 regression
# =====================================================================================================


async def test_w2_2_shape_no_spurious_tool_timeout(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Criterion 2: no ``hud.error "tool timeout"``; the result is accepted (``LlmContinue`` for step 2); the turn
    does not reach IDLE between ``tool.execute`` and the result."""
    out = await shared_w2_2_turn(tmp_path_factory)
    assert "tool timeout" not in out["hud_errors"], (
        f"criterion 2 (W2-2): a spurious R18 'tool timeout' closed the turn; hud.error {out['hud_errors']} "
        f"(drain stamped {[e.t_ms for e, _ in out['receipts']]} at session {[t for _, t in out['receipts']]})"
    )
    events = [ev for ev, _, _ in out["log"]]
    results = [i for i, ev in enumerate(events) if isinstance(ev, E.ToolResult)]
    executes = [
        i
        for i, (_, _, cmds) in enumerate(out["log"])
        if any(isinstance(c, C.ToolExecute) for c in cmds)
    ]
    assert executes and results, (
        f"premise: an execute and a result; executes {executes}, results {results}"
    )
    _, _, result_cmds = out["log"][results[0]]
    assert [c for c in result_cmds if isinstance(c, C.LlmContinue)], (
        f"criterion 2 / R12: the result is accepted and step 2 continues; commands {[c.type for c in result_cmds]}"
    )
    between = [st for _, st, _ in out["log"][executes[0] : results[0]]]
    assert TurnState.IDLE not in between, f"criterion 2: IDLE between execute and result: {between}"


async def test_tool_result_arm_is_session_time_plus_15s_under_device_lag(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Criterion 2: the ``ArmTimer("tool_result")`` record has ``at_ms == session_ms + 15 000`` within one Tick."""
    out = await shared_w2_2_turn(tmp_path_factory)
    recs = [
        (r["session_ms"], c["at_ms"])
        for r in out["records"]
        for c in r["data"]["commands"]
        if c.get("type") == "arm_timer" and c.get("name") == "tool_result"
    ]
    assert recs, "premise: R11 armed tool_result"
    session_ms, at_ms = recs[0]
    assert abs(at_ms - (session_ms + R18_MS)) <= TICK_MS, (
        f"criterion 2: tool_result armed at {at_ms}, want session_ms + 15 000 = {session_ms + R18_MS} (+-{TICK_MS}); "
        f"off by {at_ms - session_ms - R18_MS} ms"
    )


# =====================================================================================================
# criterion 3, session half: the execute deadline still wins
# =====================================================================================================


async def test_session_exec_deadline_still_wins_under_device_lag(tmp_path: Path) -> None:
    """Criterion 3: an executor that never answers gets ``ToolResult(status="timeout")`` at ``tool.execute + 10 000``
    session ms, and R18 does not fire first."""
    out = await w2_2_turn(tmp_path, answer_after_ms=None)
    events = [ev for ev, _, _ in out["log"]]
    timeouts = [ev for ev in events if isinstance(ev, E.ToolResult) and ev.status == "timeout"]
    assert "tool timeout" not in out["hud_errors"], (
        f"criterion 3: R18 fired before the session's 10 s execute deadline; hud.error {out['hud_errors']}"
    )
    assert timeouts and timeouts[0].t_ms == out["exec_ms"] + EXEC_DEADLINE_MS, (
        f"criterion 3: ToolResult(timeout) at tool.execute ({out['exec_ms']}) + 10 000; got "
        f"{[(e.status, e.t_ms) for e in events if isinstance(e, E.ToolResult)]}"
    )


# =====================================================================================================
# criterion 4: the interrupt snapshot
# =====================================================================================================


async def spoken_turn(tmp_path: Path, *, takeover: bool, text: str = LONG) -> dict[str, Any]:
    holder: dict[str, Any] = {}
    out: dict[str, Any] = {}
    async with tool_rig(
        tmp_path,
        kit=None,
        policy=POLICY,
        script=[answer(text)],
        audio_cls=lagged_audio(holder),
        ms_per_word=MS_PER_WORD,
    ) as rig:
        await with_device(rig, holder)
        await idle_lag(rig)
        await type_question(rig, "tell me something")
        if takeover:
            await rig.until(
                lambda: rig.state.state is TurnState.SPEAKING and rig.audio.played_ms() > 300,
                "the answer never started playing",
                limit_ms=10_000,
                step=10,
            )
            now = rig.clock.now_ms
            out["takeover_ms"] = now
            rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
            await rig.until(
                lambda: rig.state.state is not TurnState.INTERRUPTED,
                "stuck in INTERRUPTED",
                step=10,
            )
        else:
            await rig.until(
                lambda: rig.state.state is TurnState.IDLE, "the turn never ended", step=10
            )
        out["log"] = rig.machine.log
        out["receipts"] = list(holder["audio"].receipts)
    out["records"] = records(rig)
    out["rig"] = rig
    return out


async def test_interrupt_snapshot_uses_session_time(tmp_path: Path) -> None:
    """Criterion 4: INTERRUPTED + ``playback.stopped`` emits ``SnapshotFrame(at_ms=a)``, ``a`` the session time the
    terminal arrived at."""
    out = await spoken_turn(tmp_path, takeover=True)
    stops = [
        (i, ev) for i, (ev, _, _) in enumerate(out["log"]) if isinstance(ev, E.PlaybackStopped)
    ]
    assert stops, "premise: the barge-in flushed the run and playback.stopped reached the machine"
    i, _ = stops[0]
    snaps = [c.at_ms for c in out["log"][i][2] if isinstance(c, C.SnapshotFrame)]
    arrived = [t for ev, t in out["receipts"] if isinstance(ev, E.PlaybackStopped)]
    assert snaps and arrived, (
        f"premise: INTERRUPTED -> LISTENING with a snapshot; {snaps}, receipts {arrived}"
    )
    assert snaps[0] == arrived[0], (
        f"criterion 4: SnapshotFrame.at_ms is the session time the stop arrived ({arrived[0]}, takeover at "
        f"{out['takeover_ms']}); got {snaps[0]}"
    )


# =====================================================================================================
# criterion 5: the span
# =====================================================================================================


async def test_playback_stopped_span_not_before_first_sample(tmp_path: Path) -> None:
    """Criterion 5: ``turns.jsonl``'s ``playback.stopped`` is on the session clock: not before ``playback.first_sample``."""
    out = await spoken_turn(tmp_path, takeover=False, text="It is a short answer. ")
    spans = out["rig"].record("t1").spans
    assert "playback.first_sample" in spans and "playback.stopped" in spans, sorted(spans)
    assert spans["playback.stopped"][0] >= spans["playback.first_sample"][1], (
        f"criterion 5: playback.stopped {spans['playback.stopped']} is before playback.first_sample "
        f"{spans['playback.first_sample']} (device clock, not session clock)"
    )


# =====================================================================================================
# criterion 6: the one-clock invariant
# =====================================================================================================


async def test_every_stepped_event_is_on_the_session_clock(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Criterion 6 (P10): every ``machine.step`` record has ``0 <= session_ms - event.t_ms <= 1000``, ``scene.changed``
    only ``t_ms <= session_ms``."""
    out = await shared_w2_2_turn(tmp_path_factory)
    bad = []
    for r in out["records"]:
        ev = r["data"]["event"]
        lag = r["session_ms"] - ev["t_ms"]
        ok = lag >= 0 if ev["type"] == "scene.changed" else 0 <= lag <= P10_MS
        if not ok:
            bad.append((r["seq"], ev["type"], ev["t_ms"], r["session_ms"], lag))
    assert not bad, (
        f"criterion 6 / P10: events off the session clock (seq, type, t_ms, session_ms, lag): {bad}"
    )


# =====================================================================================================
# adversarial: attribution survives the restamp
# =====================================================================================================


async def test_attribution_survives_the_restamp_under_device_lag(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Adversarial (T-0092a criteria 1 and 3 under lag): (a) two runs in one turn (prefix, then report) both reach
    the machine restamped; (b) a turn closed by ``llm.error`` while SPEAKING has its run flushed, and that stale
    ``playback.stopped`` is dropped with its note, never stepped. Restamping *after* attribution (a copy whose ``id``
    is not in ``_attributed``) loses (b)."""
    out = await shared_w2_2_turn(tmp_path_factory)
    stepped = [ev for ev, _, _ in out["log"] if isinstance(ev, E.PlaybackDrained)]
    arrived = [t for ev, t in out["receipts"] if isinstance(ev, E.PlaybackDrained)]
    assert len(stepped) == 2 and len(arrived) == 2, (
        f"(a) two runs in one turn, both drains stepped; stepped {len(stepped)}, delivered {len(arrived)}"
    )
    assert [e.t_ms for e in stepped] == arrived, (
        f"(a) both drains restamped on arrival; stepped {[e.t_ms for e in stepped]}, arrived {arrived}"
    )

    holder: dict[str, Any] = {}
    async with tool_rig(
        tmp_path / "b",
        kit=None,
        policy=POLICY,
        script=[answer(LONG), answer("And a tail. ")],
        audio_cls=lagged_audio(holder),
        ms_per_word=MS_PER_WORD,
    ) as rig:
        await with_device(rig, holder)
        rig.llm.hold_after(1)
        await idle_lag(rig)
        await type_question(rig, "tell me something")
        await rig.until(
            lambda: rig.state.state is TurnState.SPEAKING and rig.audio.played_ms() > 300,
            "the answer never started playing",
            limit_ms=10_000,
            step=10,
        )
        rig.submit(
            E.LlmError(t_ms=rig.clock.now_ms, request_id=rig.state.active_generate, message="boom")
        )
        await rig.advance(2_000, step=20)
        assert rig.state.state is TurnState.IDLE
        delivered = [ev for ev, _ in holder["audio"].receipts if isinstance(ev, E.PlaybackStopped)]
        stops_stepped = [ev for ev in rig.machine.events() if isinstance(ev, E.PlaybackStopped)]
    notes = [r["data"] for r in rig.log_records if r.get("kind") == "note"]
    stale = [n for n in notes if n.get("what") == "stale playback terminal"]
    assert len(delivered) == 1, (
        f"premise: the session flushed the closed turn's run once; {delivered}"
    )
    assert not stops_stepped, (
        f"(b) T-0092a criterion 3: the stale stop reached the machine: {stops_stepped}"
    )
    assert len(stale) == 1 and stale[0]["run_turn"] == "t1" and stale[0]["current_turn"] is None, (
        f"(b) one stale-terminal note for t1's run; notes {notes}"
    )


# =====================================================================================================
# adversarial: zero lag (holds on the base, stays active)
# =====================================================================================================


async def test_zero_lag_stepped_terminals_are_unchanged(tmp_path: Path) -> None:
    """Adversarial: with ``FakeAudioIO`` (session-clock stamps, no lag) every terminal the machine is stepped with
    equals the one the audio delivered, field for field, and arrived at the time it carries: the restamp changes
    nothing when the clocks agree."""
    async with tool_rig(
        tmp_path,
        policy=LOOP,
        script=[answer(PREFIX), tool(TYPE_AT)],
        audio_cls=FakeReceiptAudio,
        ms_per_word=MS_PER_WORD,
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await rig.advance(IDLE_LAG_MS, step=1_000)
        await type_question(rig)
        await rig.turn_done(limit_ms=30_000)
        receipts = list(rig.audio.receipts)
        stepped = [ev for ev in rig.machine.events() if isinstance(ev, TERMINALS)]
    assert len(receipts) >= 2 and len(stepped) == len(receipts), (len(receipts), len(stepped))
    for (delivered, arrived), ev in zip(receipts, stepped, strict=True):
        assert delivered.t_ms == arrived, (
            f"premise: the fake stamps the arrival time; {delivered} at {arrived}"
        )
        assert ev.model_dump() == delivered.model_dump(), (
            f"zero lag: stepped {ev} != delivered {delivered}"
        )
    assert "tool timeout" not in hud_errors(rig)
