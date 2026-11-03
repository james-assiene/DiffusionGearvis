"""T-0057 criteria 3, 6, 7 and 8: the writer thread can be stopped, and stopping it is all it does.

``PipeWireAudioIO`` starts ``astra-audio-writer`` on the first ``play()`` and, before this task,
had no way to stop it at all: ``_writer_loop`` is an unconditional ``while True`` whose only exit
is ``WRITER_IDLE_EXIT_S = 2.0`` seconds of idleness, and nothing joins it. Measured on this tree
with the planner's probe: the thread outlives its test by **1.99 s** and makes **7 960** reads of
``FakeAudioDevice.queued_ms`` in that window (~4 000/s, the 0.2 ms *active* poll, because a row
that never flushes leaves ``_run`` set).

What these rows pin, and what they deliberately do not:

* **Criterion 3** -- an explicit stop seam that returns within 100 ms of real time, leaves no
  ``_feed_loop`` task pending, is idempotent, is safe on a module that never played, and does not
  deadlock against the lock ``_writer_loop`` takes on every pass (the landmine named in
  section Implementation notes: joining the thread while holding ``self._lock``).
* **Criterion 6 / architecture section 4** -- stopping is *not* ``flush()``. No ``PlaybackStopped``,
  no ``PlaybackDrained``, no change to ``played_ms``. A stop seam that ends the run would end turns
  mid-sentence and truncate history to audio the user did hear plus audio they did not.
* **Criterion 7** -- shortening ``WRITER_IDLE_EXIT_S`` is not the fix. A module nobody stopped
  still has its writer 100 ms after the last write; the cheap wrong fix (2.0 s -> 0.05 s) is red
  here, and is red in five named rows of ``test_audio_clock.py`` that need the writer to still be
  there when the virtual device clock moves.
* **Criterion 8** -- the seam is on the concrete class only; ``AudioIO`` is not widened.

The seam's *name* is the implementer's (criterion 3), so these rows go through
``tests/fakes/audio_device.audio_io_stop_seam``, which knows the candidate names and the arity.
Regime: ``FakeAudioDevice`` + ``VirtualClock``, no PortAudio, no PipeWire; the only real time
spent is the 100 ms gap of the criterion-7 fence and the bounds being measured.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from local_astra.contracts.protocols import AudioIO
from tests.conftest import VirtualClock
from tests.fakes.audio_device import (
    STOP_SEAM_NAMES,
    EventSink,
    FakeAudioDevice,
    PcmUtterance,
    audio_io_stop_seam,
    make_audio_io,
    maybe_await,
    play,
    play_out,
    played_ms,
    settle,
    tone,
)

#: Criterion 3(a): the join is bounded, measured with ``time.perf_counter`` (real time -- the
#: *device* clock is virtual and moves only when a row says so).
STOP_BUDGET_S = 0.100
#: Criterion 7: the gap a module nobody stopped must survive with its writer still alive. Shorter
#: than ``WRITER_IDLE_EXIT_S = 2.0`` by a wide margin, and longer than any poll in the module.
UNSTOPPED_GAP_S = 0.100
WRITER = "astra-audio-writer"

WHY_CLOSED = (
    f"{WRITER} thread(s) again. A play() or a mark_drain() after close() must start nothing:"
)

WHY_3 = (
    "T-0057 criterion 3: PipeWireAudioIO gains an explicit stop seam "
    f"(one of {list(STOP_SEAM_NAMES)}, taking no argument) that stops the writer thread"
)


def writers() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.is_alive() and t.name == WRITER}


def stop_seam(io: Any) -> Any:
    seam = audio_io_stop_seam(io)
    if seam is None:
        raise AssertionError(
            f"{type(io).__name__} has no stop seam: tried {list(STOP_SEAM_NAMES)}. {WHY_3}. "
            "Until it has one the writer's only exit is 2.0 s of idleness and every row that "
            "plays a tone hands the next row a live thread (docs/tasks/T-0057.md Regime)"
        )
    return seam


async def a_tone_was_played(device: FakeAudioDevice, ms: float = 200.0) -> Any:
    """A module with a run in flight and a writer thread alive: the state 48 rows end in."""
    io = make_audio_io(device)
    await play(io, PcmUtterance(samples=tone(ms), text="two hundred"), source="two hundred")
    await play_out(device, ms)
    assert writers(), (
        "precondition: playing a tone starts the writer thread; if it does not, this module is "
        "testing nothing (audio_io._ensure_writer)"
    )
    return io


# ---------------------------------------------------------------------------------------------
# Criterion 3 -- stopping is bounded, complete and idempotent.
# ---------------------------------------------------------------------------------------------


async def test_stop_joins_the_writer_within_100_ms() -> None:
    """Criterion 3(a): the seam returns in <= 100 ms and the thread it started is gone.

    Only threads this row started are judged: a writer another row leaked (which is exactly what
    happens until this task lands) must not decide this row's verdict.
    """
    device = FakeAudioDevice(VirtualClock())
    before = writers()
    io = await a_tone_was_played(device)
    mine = writers() - before
    assert mine, "precondition: this row started a writer of its own"

    start = time.perf_counter()
    await maybe_await(stop_seam(io)())
    elapsed = time.perf_counter() - start

    assert elapsed <= STOP_BUDGET_S, (
        f"the stop seam took {elapsed * 1000:.1f} ms, budget {STOP_BUDGET_S * 1000:.0f} ms "
        "(T-0057 criterion 3a, measured on the fake device)"
    )
    still_alive = sorted(t.name for t in mine if t.is_alive())
    assert still_alive == [], (
        f"the stop seam returned but {still_alive} is still running. Signalling the loop is not "
        "enough: criterion 3(a) says the thread is *joined*, because a thread that is merely "
        "asked to stop still reads the device until it notices"
    )


def _task_name(task: asyncio.Task[Any]) -> str:
    coro = task.get_coro()
    return str(getattr(coro, "__qualname__", coro))


def _feed_loop_tasks() -> list[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not current and "_feed_loop" in _task_name(t)]


async def test_stop_leaves_no_pending_asyncio_task() -> None:
    """Criterion 3(b): the second leak, which the thread guard cannot see.

    ``_ensure_feeder()`` creates a task running ``_feed_loop``, whose only exit is
    ``FEEDER_IDLE_EXIT_S = 5.0`` s with an empty queue; the planner's probe found 3 live tasks at
    the end of a one-tone test. A pending task is not a thread, so ``tests/conftest.py``'s guard
    is blind to it -- hence its own row (T-0057 Implementation notes, "the second leak").
    """
    device = FakeAudioDevice(VirtualClock())
    io = await a_tone_was_played(device)
    assert _feed_loop_tasks(), (
        "precondition: playing a tone starts the feeder task (audio_io._ensure_feeder)"
    )

    await maybe_await(stop_seam(io)())
    await asyncio.sleep(0)

    left = sorted(_task_name(t) for t in _feed_loop_tasks())
    assert left == [], (
        f"the stop seam left {left} pending. The task surfaces its exception at collection "
        "time, attached to whatever is running then -- which is the mechanism proposed for the "
        "PytestUnhandledThreadExceptionWarning that appears in one run and not the next "
        "(T-0057 criterion 3b)"
    )


async def test_stop_is_idempotent_and_safe_before_any_play() -> None:
    """Criterion 3(c): twice is the same as once, and a module that never played stops cleanly.

    The session is not wired to the seam in this task (T-0057, "Files this task needs"), so the
    first caller of it may well be a teardown that does not know whether the row ever played.
    """
    never_played = make_audio_io(FakeAudioDevice(VirtualClock()))
    await maybe_await(stop_seam(never_played)())
    await maybe_await(stop_seam(never_played)())

    device = FakeAudioDevice(VirtualClock())
    io = await a_tone_was_played(device)
    await maybe_await(stop_seam(io)())
    start = time.perf_counter()
    await maybe_await(stop_seam(io)())
    elapsed = time.perf_counter() - start
    assert elapsed <= STOP_BUDGET_S, (
        f"the second stop took {elapsed * 1000:.1f} ms: it waited for something that is already "
        "gone (criterion 3c)"
    )


async def test_stopping_mid_playback_does_not_deadlock_on_the_writers_lock() -> None:
    """The landmine: ``_writer_loop`` takes ``self._lock`` on every pass, so a seam that joins
    the thread *while holding it* deadlocks (T-0057 Contracts to respect, ruling A14).

    Called from a helper thread with a join timeout, so the failure is a red row and not a suite
    that hangs until the Stop hook's budget kills it. The device still has a run in flight and
    the writer is in its 0.2 ms active poll, which is when the lock is busiest.
    """
    device = FakeAudioDevice(VirtualClock())
    io = make_audio_io(device)
    await play(io, PcmUtterance(samples=tone(2000.0), text="two seconds"), source="two seconds")
    await play_out(device, 100.0)  # mid-run: 1.9 s still queued, the writer is topping up
    assert writers(), "precondition: the writer is running"

    done = threading.Event()
    failure: list[BaseException] = []
    took: list[float] = []

    def stop_it() -> None:
        started = time.perf_counter()
        try:
            result = stop_seam(io)()
            assert not asyncio.iscoroutine(result), (
                "an async stop seam cannot be called off the loop; criterion 3 is a lifecycle "
                "seam, and a deadlock on self._lock is the documented failure mode"
            )
        except BaseException as exc:
            failure.append(exc)
        finally:
            took.append(time.perf_counter() - started)
            done.set()

    caller = threading.Thread(target=stop_it, name="t0057-stop-caller")
    caller.start()
    # Waited on with the loop still turning, not with ``Event.wait``: a seam that hands the
    # cancellation of ``_feed_loop`` to the loop (``call_soon_threadsafe``) needs the loop to be
    # running to finish, and blocking it here would fail an implementation that is correct.
    deadline = time.monotonic() + 2.0
    while not done.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.002)
    finished = done.is_set()
    elapsed = took[0] if took else float("inf")
    caller.join(timeout=2.0)

    assert finished, (
        f"the stop seam did not return within 2.0 s while the writer was active: it is waiting "
        f"on a lock the writer holds on every pass (T-0057 Contracts to respect). Alive "
        f"writers: {sorted(t.name for t in writers())}"
    )
    if failure:
        raise failure[0]
    assert elapsed <= STOP_BUDGET_S, (
        f"stopping mid-playback took {elapsed * 1000:.1f} ms, budget {STOP_BUDGET_S * 1000:.0f} "
        "ms (criterion 3a); a bound that only holds when the writer is idle is not a bound"
    )


async def _close_from_another_thread(io: Any) -> None:
    """Call the stop seam off the row's loop, with the loop still turning.

    The only path on which ``close()`` runs while an event loop is alive, and therefore the only
    path on which a later ``play()`` can reach a feeder at all: the suite's other caller is the
    teardown in ``tests/conftest.py``, which runs after pytest-asyncio has closed the row's loop
    (ruling A15). Waited on by turning the loop rather than by ``Event.wait``, because a seam that
    hands the feeder's cancellation back with ``call_soon_threadsafe`` needs the loop to run.
    """
    done = threading.Event()
    failure: list[BaseException] = []

    def stop_it() -> None:
        try:
            result = stop_seam(io)()
            assert not asyncio.iscoroutine(result), (
                "an async stop seam cannot be called off the loop (criterion 3)"
            )
        except BaseException as exc:  # re-raised on the row's thread below
            failure.append(exc)
        finally:
            done.set()

    caller = threading.Thread(target=stop_it, name="t0057-stop-caller")
    caller.start()
    deadline = time.monotonic() + 2.0
    while not done.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.002)
    caller.join(timeout=2.0)
    if failure:
        raise failure[0]
    assert done.is_set(), "the stop seam did not return within 2.0 s (criterion 3a)"


async def test_playing_after_a_close_starts_no_writer_and_no_feeder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 3(c) on the reachable path: ``close()`` is *terminal*, not merely idempotent.

    Ruling A15: the ``_closed`` guards in ``play()`` and ``_ensure_writer()`` were unpinned --
    dropping either left all 81 rows green -- because the only caller of the seam in the suite is
    the teardown, which runs after the row's loop is shut, so nothing can queue anything
    afterwards and neither guard is reachable. Here the close comes from another thread while this
    row's loop is still turning, which is the shape a session teardown racing a late ``TtsSpeak``
    has, and then the module is asked to do the two things that restart its background work:
    ``mark_drain()`` reaches for the writer, and ``play()`` queues an utterance and starts a feeder
    for it.

    The writer half is judged on whether one is **started**, not on whether one is alive a
    scheduling point later. A resurrected writer finds the stop flag still set and exits at the top
    of its loop (ruling A13), so an aliveness check is a race that reports green on a box under
    load -- while the module is left holding a thread object it will never join, and on a tree
    where the flag is not sticky the same restart is a thread reading the device for two more
    seconds inside the next row.

    Both halves are asserted because they fail separately: without the ``_ensure_writer`` guard a
    writer is started after the join, and without the ``play()`` guard an ``_feed_loop`` task is
    stranded with nobody left to cancel it -- the leak the thread guard cannot see.
    """
    device = FakeAudioDevice(VirtualClock())
    before = writers()
    io = await a_tone_was_played(device)

    await _close_from_another_thread(io)
    await asyncio.sleep(0)
    assert [t.name for t in writers() - before if t.is_alive()] == [], (
        "precondition: the off-loop close joined the writer this row started"
    )

    # An oracle that cannot miss: every ``Thread.start()`` after the close, by name. Delegates,
    # so nothing else about threading changes for this row.
    started: list[str] = []
    original_start = threading.Thread.start

    def spy(self: threading.Thread) -> None:
        if self.name == WRITER:
            started.append(self.name)
        original_start(self)

    monkeypatch.setattr(threading.Thread, "start", spy)

    io.mark_drain()
    await play(io, PcmUtterance(samples=tone(100.0), text="too late"), source="too late")
    await asyncio.sleep(0)

    assert started == [], (
        f"a mark_drain() and a play() after close() started {len(started)} {WHY_CLOSED} "
        "close() is terminal: the module was joined and nobody is going to join it a second "
        "time (T-0057 criterion 3, ruling A15)"
    )
    alive = sorted(t.name for t in writers() - before if t.is_alive())
    assert alive == [], (
        f"a play() after close() left {alive} running: it will poll the device until its 2.0 s "
        "idle exit, inside whatever row runs next"
    )
    stranded = sorted(_task_name(t) for t in _feed_loop_tasks())
    assert stranded == [], (
        f"a play() after close() left {stranded} pending. There is no writer to render it and no "
        "second close() to cancel it, so the task surfaces its exception at collection time "
        "against an unrelated row (criterion 3b, ruling A13)"
    )


# ---------------------------------------------------------------------------------------------
# Criterion 6 / architecture section 4 -- stopping the writer is not flush(), and moves no number.
# ---------------------------------------------------------------------------------------------


async def test_stopping_the_writer_emits_no_terminal_event_and_moves_no_number() -> None:
    """Architecture section 4: ``playback.drained`` needs ``mark_drain()`` *and* the last sample,
    and ``played_ms`` is the device's rendered position. Neither is a lifecycle concern.

    A seam that ends the run to get the loop to exit would emit ``PlaybackStopped`` at teardown of
    every row that built a module -- the turn machine's ``SPEAKING -> IDLE`` edge fires on those
    events, so this is how a stop seam turns into a truncated turn.
    """
    device = FakeAudioDevice(VirtualClock())
    io = make_audio_io(device)
    sink = EventSink(io)
    await play(io, PcmUtterance(samples=tone(500.0), text="five hundred"), source="five hundred")
    await play_out(device, 200.0)
    before = await played_ms(io)
    underruns_before = io.underruns

    await maybe_await(stop_seam(io)())
    await sink.sync()
    terminal = sink.of_type(PlaybackStopped, PlaybackDrained)
    after = await played_ms(io)
    await sink.close()

    assert terminal == [], (
        f"stopping the writer emitted {terminal}. Stopping is not flush(): no PlaybackStopped, "
        "and PlaybackDrained still requires mark_drain() and the last sample (docs/02 section 4, "
        "T-0006 spec delta 1, T-0057 criterion 6)"
    )
    assert after == pytest.approx(before, abs=1e-6), (
        f"played_ms moved from {before:.3f} to {after:.3f} across the stop seam; it is the "
        "device's rendered position and nothing about the writer's lifetime may change it"
    )
    assert io.underruns == underruns_before, (
        f"underruns went from {underruns_before} to {io.underruns} across the stop seam; a run "
        "whose producer was stopped on purpose has not starved (T-0006 criterion 7)"
    )


async def test_the_stop_seam_does_not_stop_the_capture_half() -> None:
    """Criterion 6: ``close_capture()`` is T-0023's and keeps its own lifecycle.

    The two halves share one device object on purpose (one clock domain). A stop seam that closed
    the capture stream as well would end ``capture_frames()`` for a session that is still
    listening -- the barge-in path -- and would do it from a teardown.
    """
    device = FakeAudioDevice(VirtualClock())
    io = await a_tone_was_played(device)
    stops_before = device.capture_stops

    await maybe_await(stop_seam(io)())
    await settle(device)

    assert device.capture_stops == stops_before, (
        f"the stop seam stopped capture too ({stops_before} -> {device.capture_stops}); "
        "close_capture() is the capture half's own seam and belongs to T-0023 (criterion 6)"
    )


# ---------------------------------------------------------------------------------------------
# Criterion 7 -- the idle constant is load-bearing; shortening it is not the fix.
# ---------------------------------------------------------------------------------------------


async def test_an_unstopped_module_keeps_its_writer_across_a_gap() -> None:
    """A module nobody stopped still has its writer 100 ms after the last write.

    This is the fence under the cheap wrong fix. Setting ``WRITER_IDLE_EXIT_S`` to 0.05 s makes
    the leak census look fixed and turns five rows red elsewhere -- all four parametrisations of
    ``test_audio_clock.py::test_underrun_does_not_drain`` and
    ``test_audio_clock.py::test_no_drain_while_the_device_still_holds_the_tail`` -- because they
    need the writer to still be there when the virtual device clock moves. The signalled exit
    must be *signalled*: no caller, no exit.
    """
    device = FakeAudioDevice(VirtualClock())
    before = writers()
    await a_tone_was_played(device)
    mine = writers() - before

    time.sleep(UNSTOPPED_GAP_S)

    alive = {t for t in mine if t.is_alive()}
    assert alive, (
        f"the writer this row started exited on its own within {UNSTOPPED_GAP_S * 1000:.0f} ms "
        "of the last write. Nothing asked it to stop, so the exit came from the idle constant "
        "being shortened -- which is T-0057 criterion 7's wrong fix and turns five rows of "
        "test_audio_clock.py red (drain and underrun semantics, architecture section 4)"
    )


# ---------------------------------------------------------------------------------------------
# Criterion 8 -- the protocol is not widened.
# ---------------------------------------------------------------------------------------------

#: ``AudioIO`` as T-0007 left it. ``played_ms``/``mark_drain``/``playback_events`` are on it;
#: ``played_text``, ``playback_reference``, ``capture_frames`` and now the stop seam are not --
#: they live on the concrete class (T-0006 Owned files, T-0021 ruling).
AUDIO_IO_MEMBERS = frozenset(
    {
        "capture",
        "flush",
        "is_rendering",
        "mark_drain",
        "pause",
        "play",
        "playback_events",
        "played_ms",
        "resume",
    }
)


def test_the_stop_seam_is_not_added_to_the_audio_io_protocol() -> None:
    """Criterion 8: ``contracts/protocols.py`` is untouched, so every fake keeps conforming.

    Widening the protocol would make ``tests/fakes/audio.py::FakeAudioIO`` and every other
    implementation non-conforming until each grew a method it has no thread to stop.
    """
    members = frozenset(n for n in dir(AudioIO) if not n.startswith("_"))
    assert members == AUDIO_IO_MEMBERS, (
        f"the AudioIO protocol changed: added {sorted(members - AUDIO_IO_MEMBERS)}, removed "
        f"{sorted(AUDIO_IO_MEMBERS - members)}. T-0057 criterion 8 keeps the stop seam on the "
        "concrete class only"
    )
    named = sorted(members & set(STOP_SEAM_NAMES))
    assert named == [], f"the stop seam {named} was added to the AudioIO protocol (criterion 8)"
