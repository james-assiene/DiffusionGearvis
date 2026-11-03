"""T-0195 group B, the session half: log-only mic and recogniser diagnostics (B1-B7), through the session rig.

``docs/tasks/T-0195.md`` group B. **The surface these rows assume** (the task's names; the implementer owns the rest):

- ``note`` records written through ``Session._log``: ``{"what": "mic.level", "reason": "period" | "gate", ...}`` with
  exactly the B1 fields, and ``{"what": "mic.unmute_check", ...}`` with exactly the B4 fields;
- frame time is the capture frame's own ``t_ms`` (the STT-tap audio clock, ``frame.t_ms``), never the session clock.

**Readings** (test-author, stated in the report; the orchestrator rules):

- **B-R1, windows are half-open on frame time.** A window starts at its first frame; the frame with
  ``t_ms >= start + LEVEL_WINDOW_MS`` writes the row and is the *next* window's first frame. With 80 ms frames a period
  row covers exactly 125 frames, ``to - from == 9920``, and consecutive rows tile. The same holds for the gate row (the
  first frame under the new state writes it and starts the next window, so ``to_frame_t_ms`` is the frame before the
  ``mic.gate`` note's ``frame_t_ms``) and for the unmute check (from the unmute frame up to the frame before
  ``unmute + 5000``; with 80 ms frames that is ``unmute + 4960``).
- **B-R2, ``frames`` counts every frame of the window**, muted included; ``muted_frames`` is a subset.
- **B-R3, ``onsets`` counts the endpointer's ``("onset", ...)`` decisions** (the ``stt.decision`` rows of kind onset).
- **B-R4, "immediately before" is adjacency in the session log.** The ``mic.level`` gate row is the record right before
  each ``mic.gate`` record. A cut-short ``mic.unmute_check`` is written before that pair.
- **B-R5, the session start is not an unmute**: no ``mic.unmute_check`` before the first unmute.

**The rig.** ``tests/unit/_t0181_rig.mic_rig`` (read-only) with ``ShapedAudio`` below: the ``FakeAudioIO`` capture
(80 ms frames as virtual time passes), each frame's samples replaced by a 400 Hz sine whose RMS is the level the row
scripts (``level(t_ms)`` in dBFS; a whole number of periods per frame, so the frame RMS is the scripted level to float
precision), or by digital zeros, or by a frame holding ``nan``/``inf``. What it yields is recorded, packed exactly as the
pump packs it: the oracle for "the bytes ``stt.feed`` got are BASE's". The recogniser is a reactive ``FakeStt`` with
steps (one per 80 ms of fed audio, delivered ``lag_ms`` later), or ``StalledStt`` (B5(d): takes audio, says nothing,
its outbound queue grows).

Written ``pending(task="T-0195")``; activated 2026-10-29 when group B's implementation started. Every row is red at
BASE 51291e70 on a named ``AssertionError`` (no ``mic.level``/``mic.unmute_check`` note); the B6 rows are guards whose
BASE-red is only their premise that diagnostic rows exist.

Regime: CPU over fakes, ``ScriptedClock`` virtual time, ``FakeAudioIO`` capture starting with the session clock,
n = 1 deterministic per row. No measurement.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fakes.stt import FakeStt
from tests.unit._t0181_rig import MicRig, RecordingStt, mic_rig, reactive_stt
from tests.unit._tool_turn_support import answer
from tests.unit.test_session_loop import RecordingAudio

LEVEL = "mic.level"
CHECK = "mic.unmute_check"
GATE = "mic.gate"
FRAME_MS = 80
WINDOW_MS = 10_000
CHECK_MS = 5_000
LAG_MS = 500
REPLY = "Hi there. "

#: B1, verbatim
LEVEL_KEYS = {
    "what",
    "reason",
    "from_frame_t_ms",
    "to_frame_t_ms",
    "frames",
    "muted_frames",
    "zero_frames",
    "nonfinite_frames",
    "rms_dbfs",
    "peak_dbfs",
    "max_frame_rms_dbfs",
    "min_frame_rms_dbfs",
    "words_received",
    "words_estimated",
    "words_dropped_muted",
    "steps_received",
    "onsets",
    "pause_head_min",
    "pause_head_max",
    "stt",
}
#: B4, verbatim
CHECK_KEYS = {
    "what",
    "from_frame_t_ms",
    "to_frame_t_ms",
    "cut_short",
    "words_received",
    "words_estimated",
    "steps_received",
    "onsets",
    "max_frame_rms_dbfs",
    "stt",
}
DB_KEYS = ("rms_dbfs", "peak_dbfs", "max_frame_rms_dbfs", "min_frame_rms_dbfs")
#: B6: the BASE shapes (session.py at 51291e70: ``_on_decision`` and ``_mic_gate``)
DECISION_KEYS = {"decision", "kind", "t_eot_ms", "onset_ms", "p_end", "now_ms"}
GATE_MUTE_KEYS = {"what", "muted", "frame_t_ms"}
GATE_UNMUTE_KEYS = {"what", "muted", "frame_t_ms", "words_dropped"}

SILENCE = -100.0
SINE_HZ = 400
RATE = 24_000
N = 1920

#: a sample value the row scripts instead of a level
ZERO = "zero"
NAN = "nan"
INF = "inf"


# =====================================================================================================================
# the rig
# =====================================================================================================================


def tone(db: float) -> np.ndarray:
    """1920 samples of a 400 Hz sine whose RMS is ``db`` dBFS (amplitude ``sqrt(2) * 10**(db/20)``)."""
    amplitude = math.sqrt(2.0) * 10.0 ** (db / 20.0)
    t = np.arange(N, dtype=np.float64) / RATE
    return (amplitude * np.sin(2 * math.pi * SINE_HZ * t)).astype(np.float32)


class ShapedAudio(RecordingAudio):
    """``RecordingAudio`` whose STT frames carry the scripted signal ``level(t_ms)`` (dBFS, ``ZERO``, ``NAN``, ``INF``).
    The level is read when the frame is yielded. ``yielded`` keeps ``(bytes, t_ms)`` of every frame, packed as the
    pump packs it."""

    def __init__(self, clock: Any, level: Callable[[float], Any]) -> None:
        super().__init__(clock)
        self.level = level
        self.yielded: list[tuple[bytes, int]] = []
        self.levels: list[Any] = []

    def samples_for(self, t_ms: float) -> np.ndarray:
        spec = self.level(t_ms)
        self.levels.append(spec)
        if spec == ZERO:
            return np.zeros(N, dtype=np.float32)
        if spec in (NAN, INF):
            frame = tone(-30.0)
            frame[100] = math.nan if spec == NAN else math.inf
            return frame
        return tone(float(spec))

    async def stt_blocks(self) -> Any:
        async for frame in super().stt_blocks():
            shaped = dataclasses.replace(frame, samples=self.samples_for(frame.t_ms))
            pcm = np.ascontiguousarray(shaped.samples, dtype="<f4").tobytes()
            self.yielded.append((pcm, int(shaped.t_ms)))
            yield shaped


class ScriptedFrames(ShapedAudio):
    """Frames that do not wait for the session clock: ``count`` frames from ``origin_ms`` on the capture clock, one per
    event-loop turn, then nothing (the stream stays open). The session clock never moves unless the row moves it."""

    def __init__(
        self, clock: Any, level: Callable[[float], Any], origin_ms: int, count: int
    ) -> None:
        super().__init__(clock, level)
        self.origin_ms = origin_ms
        self.count = count

    async def stt_blocks(self) -> Any:
        for k in range(self.count):
            t_ms = float(self.origin_ms + FRAME_MS * k)
            samples = self.samples_for(t_ms)
            pcm = np.ascontiguousarray(samples, dtype="<f4").tobytes()
            self.yielded.append((pcm, int(t_ms)))
            yield _Frame(
                samples=samples, t_ms=t_ms, end_ms=t_ms + FRAME_MS, yielded_ms=t_ms + FRAME_MS
            )
            await asyncio.sleep(0)
        await asyncio.Event().wait()


@dataclasses.dataclass(frozen=True)
class _Frame:
    samples: np.ndarray
    t_ms: float
    end_ms: float
    yielded_ms: float


class StalledStt:
    """B5(d): a recogniser that takes audio and says nothing: no step, no word. Its outbound queue fills to 40 blocks
    and the rest is dropped, as ``MoshiSttClient.stats`` counts them (``queued_blocks``, ``dropped_blocks``)."""

    QUEUE = 40

    def __init__(self) -> None:
        self.fed: list[tuple[bytes, int]] = []
        self.closed = False
        self._done = asyncio.Event()

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        self.fed.append((pcm_24k_f32, t_ms))

    async def mark(self, t_eot_ms: int) -> int:
        return 1

    async def events(self) -> AsyncIterator[Any]:
        await self._done.wait()
        return
        yield  # pragma: no cover

    def stats(self) -> Mapping[str, int]:
        n = len(self.fed)
        return {
            "blocks": n,
            "queued_blocks": min(n, self.QUEUE),
            "dropped_blocks": max(0, n - self.QUEUE),
            "dropped_events": 0,
        }

    async def close(self) -> None:
        self.closed = True
        self._done.set()


class NoStatsStt:
    """Exactly the ``SttClient`` protocol over a reactive ``FakeStt``, with no ``stats()`` (B1: ``stt`` is null)."""

    def __init__(self, inner: FakeStt) -> None:
        self.inner = inner
        self.fed: list[tuple[bytes, int]] = []

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        self.fed.append((pcm_24k_f32, t_ms))
        await self.inner.feed(pcm_24k_f32, t_ms)

    async def mark(self, t_eot_ms: int) -> int:
        return await self.inner.mark(t_eot_ms)

    def events(self) -> Any:
        return self.inner.events()

    async def close(self) -> None:
        await self.inner.close()

    @property
    def closed(self) -> bool:
        return bool(self.inner.closed)


class SnapshotStt(RecordingStt):
    """``RecordingStt`` whose ``stats()`` carries the real-client fields the fake lacks, plus a serial number per
    call, so a row's ``stt`` can be matched to one snapshot ``stats()`` returned."""

    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.snapshots: list[dict[str, int]] = []
        #: the calling function's name per call (``_check_stt_drops`` asks every frame at BASE; the rows do not)
        self.callers: list[str] = []

    def stats(self) -> Any:
        self.callers.append(sys._getframe(1).f_code.co_name)
        snap = {
            **dict(self.inner.stats()),
            "queued_blocks": 0,
            "reconnects": 0,
            "word_delivery_lag_count": 0,
            "serial": len(self.snapshots) + 1,
        }
        self.snapshots.append(snap)
        return dict(snap)


def stt_kwargs(**extra: Any) -> dict[str, Any]:
    return {"lag_ms": LAG_MS, "steps": True, **extra}


def shaped(level: Callable[[float], Any]) -> Callable[[Any], ShapedAudio]:
    return lambda clock: ShapedAudio(clock, level)


def const(db: float) -> Callable[[float], Any]:
    return lambda _t: db


def notes_in_order(rig: MicRig) -> list[dict[str, Any]]:
    """Every record of the session log in order, as ``(kind, data)`` dicts (after ``aclose``)."""
    return [{"kind": r.get("kind"), **(r.get("data") or {})} for r in rig.log_records]


def levels(rig: MicRig, reason: str | None = None) -> list[dict[str, Any]]:
    rows = rig.notes(LEVEL)
    return [r for r in rows if reason is None or r.get("reason") == reason]


def require_rows(rig: MicRig, what: str = LEVEL) -> list[dict[str, Any]]:
    rows = rig.notes(what)
    assert rows, (
        f"T-0195 {'B1' if what == LEVEL else 'B4'}: the session wrote no {what!r} note; notes written: "
        f"{sorted({n.get('what') for n in rig.kind('note')})}"
    )
    return rows


def fed_t(rig: MicRig) -> list[int]:
    return [t for _, t in rig.rec_stt.fed]


def finite_or_none(row: Mapping[str, Any]) -> list[str]:
    """Keys of ``row`` (and of its ``stt``) holding a non-finite float."""
    bad = [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
    stt = row.get("stt")
    if isinstance(stt, dict):
        bad += [f"stt.{k}" for k, v in stt.items() if isinstance(v, float) and not math.isfinite(v)]
    return bad


# =====================================================================================================================
# B1: the period row
# =====================================================================================================================


async def test_period_row_every_ten_seconds_of_capture_time(tmp_path: Path) -> None:
    """B1. Part 1: 25 s of -30 dBFS capture. Two period rows, each 125 frames on the capture clock, tiling (B-R1),
    with exactly the B1 fields and the scripted level. Part 2 (adversarial): the session clock frozen and the capture
    clock 7 000 s ahead of it: 300 frames still give two period rows starting at the first frame and 10 000 ms later;
    a window kept on the session clock gives none (or rows at session-clock times)."""
    async with mic_rig(tmp_path / "a", audio=shaped(const(-30.0)), stt_kwargs=stt_kwargs()) as rig:
        await rig.advance(25_000, step=FRAME_MS)
        fed = fed_t(rig)
    rows = require_rows(rig)
    periods = [r for r in rows if r.get("reason") == "period"]
    t0 = fed[0]
    assert len(fed) >= 300, f"premise: 25 s of capture fed ~312 frames; {len(fed)}"
    got = [(r["from_frame_t_ms"], r["to_frame_t_ms"], r["frames"]) for r in periods]
    assert got == [(t0, t0 + 9_920, 125), (t0 + 10_000, t0 + 19_920, 125)], (
        f"B1/B-R1: one period row per 10 000 ms of capture time, 125 frames each, tiling from the first frame "
        f"({t0}); got {got}"
    )
    for r in periods:
        assert set(r) == LEVEL_KEYS, (
            f"B1: exactly the task's fields; extra {set(r) - LEVEL_KEYS}, missing {LEVEL_KEYS - set(r)}"
        )
        assert (r["muted_frames"], r["zero_frames"], r["nonfinite_frames"]) == (0, 0, 0), r
        for key in ("rms_dbfs", "max_frame_rms_dbfs", "min_frame_rms_dbfs"):
            assert abs(r[key] - (-30.0)) < 0.05, f"B1: {key} of a -30 dBFS tone; got {r[key]}"
        assert abs(r["peak_dbfs"] - (-30.0 + 20 * math.log10(math.sqrt(2)))) < 0.1, r["peak_dbfs"]
        assert 110 <= r["steps_received"] <= 130, (
            f"B1: the fake steps once per 80 ms of fed audio (~125 per window); got {r['steps_received']}"
        )
        assert r["pause_head_min"] == pytest.approx(0.9) and r["pause_head_max"] == pytest.approx(
            0.9
        ), (
            f"B1: the fake's silence head is 0.9 at every index; got {r['pause_head_min']}, {r['pause_head_max']}"
        )

    origin = 7_000_000
    async with mic_rig(
        tmp_path / "b",
        audio=lambda clock: ScriptedFrames(clock, const(-40.0), origin, 300),
        stt_kwargs=stt_kwargs(),
    ) as rig:
        clock_before = rig.clock.now_ms
        for _ in range(3_000):
            if len(rig.rec_stt.fed) >= 300:
                break
            await rig.pump(rounds=1)
        await rig.pump()
        clock_after = rig.clock.now_ms
        fed = fed_t(rig)
    assert clock_after == clock_before, "premise: the session clock did not move"
    assert len(fed) == 300 and fed[0] == origin, (
        f"premise: 300 frames from {origin}; {len(fed)}, {fed[:1]}"
    )
    periods = [r for r in rig.notes(LEVEL) if r.get("reason") == "period"]
    got = [(r["from_frame_t_ms"], r["to_frame_t_ms"], r["frames"]) for r in periods]
    assert got == [(origin, origin + 9_920, 125), (origin + 10_000, origin + 19_920, 125)], (
        f"B1 (adversarial): windows compare frame.t_ms with frame.t_ms only; with the session clock frozen at "
        f"{clock_after} and the capture clock at {origin}, got {got}"
    )


# =====================================================================================================================
# B2: the gate row
# =====================================================================================================================


async def test_gate_row_precedes_each_mic_gate_note_and_mic_gate_is_unchanged(
    tmp_path: Path,
) -> None:
    """B2. Mute after ~2 s, unmute after ~1.5 s muted, then 10.5 s live. Right before each ``mic.gate`` record sits a
    ``mic.level`` row with ``reason: "gate"`` closing the window at the frame before the switch (B-R1, B-R4); the
    mute's gate row is all live, the unmute's all muted with null levels; the next window starts at the switch frame;
    ``mic.gate`` keeps its BASE fields (T-0181 A10)."""
    async with mic_rig(tmp_path, audio=shaped(const(-30.0)), stt_kwargs=stt_kwargs()) as rig:
        await rig.advance(2_000, step=FRAME_MS)
        await rig.engage()
        await rig.advance(1_500, step=FRAME_MS)
        await rig.release()
        await rig.advance(10_500, step=FRAME_MS)
        fed = fed_t(rig)
    records = notes_in_order(rig)
    gates = [i for i, r in enumerate(records) if r.get("kind") == "note" and r.get("what") == GATE]
    assert len(gates) == 2, (
        f"premise: one mute and one unmute; mic.gate records {[records[i] for i in gates]}"
    )
    require_rows(rig)
    before = [records[i - 1] for i in gates]
    for i, prev in zip(gates, before, strict=True):
        assert (
            prev.get("kind") == "note"
            and prev.get("what") == LEVEL
            and prev.get("reason") == "gate"
        ), (
            f"B2/B-R4: the record right before {records[i]} must be the mic.level gate row; got {prev}"
        )
    mute_gate, unmute_gate = records[gates[0]], records[gates[1]]
    assert (
        set(mute_gate) - {"kind"} == GATE_MUTE_KEYS
        and set(unmute_gate) - {"kind"} == GATE_UNMUTE_KEYS
    ), f"B2/B6: mic.gate unchanged (T-0181 A10); got {mute_gate}, {unmute_gate}"
    m, u, t0 = mute_gate["frame_t_ms"], unmute_gate["frame_t_ms"], fed[0]
    closing_mute, closing_unmute = before
    assert (closing_mute["from_frame_t_ms"], closing_mute["to_frame_t_ms"]) == (t0, m - FRAME_MS), (
        f"B2/B-R1: the mute's gate row closes the window at the frame before the switch ({t0}..{m - FRAME_MS}); "
        f"got {closing_mute}"
    )
    assert closing_mute["frames"] == (m - t0) // FRAME_MS and closing_mute["muted_frames"] == 0, (
        closing_mute
    )
    assert abs(closing_mute["max_frame_rms_dbfs"] - (-30.0)) < 0.05, closing_mute
    assert (closing_unmute["from_frame_t_ms"], closing_unmute["to_frame_t_ms"]) == (
        m,
        u - FRAME_MS,
    ), (
        f"B2/B-R1: the unmute's gate row covers the muted span {m}..{u - FRAME_MS}; got {closing_unmute}"
    )
    n_muted = (u - m) // FRAME_MS
    assert (
        closing_unmute["frames"],
        closing_unmute["muted_frames"],
        closing_unmute["zero_frames"],
    ) == (
        n_muted,
        n_muted,
        0,
    ), (
        f"B3: every frame of the muted span counts as muted, none as a measured zero frame; got {closing_unmute}"
    )
    assert all(closing_unmute[k] is None for k in DB_KEYS), (
        f"B3: an all-muted window has null levels; {closing_unmute}"
    )
    after = [r for r in levels(rig, "period") if r["from_frame_t_ms"] >= u]
    assert after and after[0]["from_frame_t_ms"] == u and after[0]["to_frame_t_ms"] == u + 9_920, (
        f"B2/B-R1: the window the unmute opened starts at the unmute frame {u}; period rows after it {after}"
    )
    assert all(r["from_frame_t_ms"] != t0 or r["reason"] == "gate" for r in levels(rig)), (
        "B2: the gate row closed the first window; no period row may cover it again"
    )


# =====================================================================================================================
# B3: privacy
# =====================================================================================================================


async def test_muted_frames_are_not_measured(tmp_path: Path) -> None:
    """B3 (adversarial). The room is -50 dBFS while live; while muted the capture is -6 dBFS (someone talking with the
    mic muted). No row anywhere may carry a level above the live room: not a period row that spans the switch, not
    the unmute's gate row (null levels), not the unmute check. A window measured from the captured samples of muted
    frames reads -6; one measured after ``_mic_gate`` counts the muted frames as zero frames."""
    state = {"loud": False}
    audio = shaped(lambda _t: -6.0 if state["loud"] else -50.0)
    async with mic_rig(tmp_path, audio=audio, stt_kwargs=stt_kwargs()) as rig:
        await rig.advance(6_000, step=FRAME_MS)
        rig.send_mute(True)
        await rig.pump()
        state["loud"] = True
        rig.say(
            "muted words here", rig.clock.now_ms + 1_000
        )  # content-blind fake: released, dropped by the session
        await rig.advance(6_000, step=FRAME_MS)
        state["loud"] = False
        await rig.release()
        await rig.advance(9_000, step=FRAME_MS)
        loud_yields = sum(1 for spec in rig.audio.levels if spec == -6.0)
        zero_fed = sum(rig.rec_stt.zero())
    assert loud_yields >= 70 and zero_fed >= loud_yields, (
        f"premise: ~75 loud frames were captured while muted and all were fed as zeros; loud {loud_yields}, "
        f"zero-fed {zero_fed}"
    )
    rows = require_rows(rig) + rig.notes(CHECK)
    for r in rows:
        for key in DB_KEYS:
            v = r.get(key)
            assert v is None or v < -40.0, (
                f"B3: {key} = {v} in {r.get('what')}/{r.get('reason')}: muted audio was measured (live room is -50)"
            )
    gate_rows = levels(rig, "gate")
    unmute_row = gate_rows[-1]
    assert (
        unmute_row["muted_frames"] == unmute_row["frames"] >= 70 and unmute_row["zero_frames"] == 0
    ), (
        f"B3: the muted span counts toward muted_frames only (not zero_frames: measured after the gate); {unmute_row}"
    )
    assert all(unmute_row[k] is None for k in DB_KEYS), unmute_row
    dropped = rig.notes(GATE)[-1].get("words_dropped")
    assert dropped == 3, (
        f"premise: the three words spoken while muted were dropped by the session; {dropped}"
    )
    counted = [(r["words_received"], r["words_dropped_muted"]) for r in levels(rig)]
    assert sum(d for _, d in counted) == 3 and sum(w for w, _ in counted) >= 3, (
        f"B1/implementation notes: STT words are counted at _on_stt_event entry, before the mute drop, and the dropped "
        f"ones as words_dropped_muted; (words_received, words_dropped_muted) per row {counted}"
    )
    total_muted = sum(r["muted_frames"] for r in levels(rig))
    assert total_muted == zero_fed, (
        f"B3: every frame the mute fed as zeros is counted muted exactly once across the rows; {total_muted} vs "
        f"{zero_fed}"
    )
    # review F2: a 6 s mute is not an unmute; the only check opens at the unmute frame, and there is exactly one
    unmute_t = rig.notes(GATE)[-1]["frame_t_ms"]
    checks = rig.notes(CHECK)
    assert [c["from_frame_t_ms"] for c in checks] == [unmute_t], (
        f"B4: exactly one mic.unmute_check, from the unmute frame {unmute_t} (none opened by the mute); "
        f"got {[(c['from_frame_t_ms'], c['cut_short']) for c in checks]}"
    )
    assert checks[0]["cut_short"] is False, checks[0]
    # review F3: the recogniser keeps stepping on the zeros; steps delivered while muted are counted (one per 80 ms
    # of fed audio, the fake's rate), so the all-muted gate row's steps_received tracks its frame count
    frames, steps = unmute_row["frames"], unmute_row["steps_received"]
    assert abs(steps - frames) <= 8, (
        f"B1: steps delivered during the muted span are counted ({frames} muted frames, ~{frames} steps); got {steps}"
    )


# =====================================================================================================================
# B4: the unmute check
# =====================================================================================================================


async def test_unmute_check_after_five_seconds_and_cut_short_on_remute(tmp_path: Path) -> None:
    """B4. Mute, unmute at U, "Hello there" 400 ms later; a row at the first frame >= U + 5000 counting from U (two
    words, one onset, ~62 steps, the live level). Mute and unmute again at U2 and re-mute 2 s later: one row with
    ``cut_short: true`` closing at the frame before the re-mute, written before that mute's gate pair (B-R4). None
    before the first unmute (B-R5); one per unmute."""
    async with mic_rig(
        tmp_path, audio=shaped(const(-30.0)), stt_kwargs=stt_kwargs(), script=[answer(REPLY)]
    ) as rig:
        await rig.advance(1_000, step=FRAME_MS)
        await rig.engage()
        await rig.advance(2_000, step=FRAME_MS)
        u = await rig.release()
        rig.say("Hello there", u + 400)
        await rig.advance(6_000, step=FRAME_MS)
        await rig.engage()
        await rig.advance(500, step=FRAME_MS)
        u2 = await rig.release()
        await rig.advance(2_000, step=FRAME_MS)
        await rig.engage()
        await rig.advance(1_000, step=FRAME_MS)
        fed = fed_t(rig)
    checks = require_rows(rig, CHECK)
    records = notes_in_order(rig)
    gate_idx = [i for i, r in enumerate(records) if r.get("what") == GATE]
    check_idx = [i for i, r in enumerate(records) if r.get("what") == CHECK]
    assert len(gate_idx) == 5, (
        f"premise: mute, unmute, mute, unmute, mute; {[records[i] for i in gate_idx]}"
    )
    assert check_idx[0] > gate_idx[1], "B4/B-R5: no unmute check before the first unmute"
    assert len(checks) == 2, f"B4: one check per unmute; got {checks}"
    first, second = checks
    for c in checks:
        assert set(c) == CHECK_KEYS, (
            f"B4: exactly the task's fields; extra {set(c) - CHECK_KEYS}, missing {CHECK_KEYS - set(c)}"
        )
    last_before = max(t for t in fed if t < u + CHECK_MS)
    assert (first["from_frame_t_ms"], first["to_frame_t_ms"], first["cut_short"]) == (
        u,
        last_before,
        False,
    ), (
        f"B4/B-R1: from the unmute frame {u} to the frame before {u} + 5000 ({last_before}), not cut short; {first}"
    )
    assert (first["words_received"], first["words_estimated"], first["onsets"]) == (2, 0, 1), (
        f"B4/B-R3: 'Hello there' after the unmute: two words, none estimated, one onset; {first}"
    )
    assert 50 <= first["steps_received"] <= 66, (
        f"B4: ~62 steps in 5 s of fed audio; {first['steps_received']}"
    )
    assert abs(first["max_frame_rms_dbfs"] - (-30.0)) < 0.05, first
    remute = records[gate_idx[4]]["frame_t_ms"]
    assert records[gate_idx[3]]["frame_t_ms"] == u2
    assert (second["from_frame_t_ms"], second["to_frame_t_ms"], second["cut_short"]) == (
        u2,
        remute - FRAME_MS,
        True,
    ), (
        f"B4: a re-mute {remute - u2} ms after the unmute cuts the check short at the frame before it; {second}"
    )
    assert gate_idx[3] < check_idx[1] < gate_idx[4], (
        f"B4/B-R4: the cut-short row is written at the re-mute, before its mic.gate; check at {check_idx[1]}, "
        f"gates at {gate_idx}"
    )
    # review F1 (B-R4): the re-mute writes cut-short check, then the gate row, then mic.gate -- adjacent, in that order
    order = [
        (records[i].get("what"), records[i].get("reason"))
        for i in range(gate_idx[4] - 2, gate_idx[4] + 1)
    ]
    assert order == [(CHECK, None), (LEVEL, "gate"), (GATE, None)], (
        f"B2/B4/B-R4: at a re-mute the cut-short mic.unmute_check, then the mic.level gate row, then mic.gate; got {order}"
    )
    assert records[gate_idx[4] - 2].get("cut_short") is True, records[gate_idx[4] - 2]
    assert second["words_received"] == 0 and second["onsets"] == 0, second


# =====================================================================================================================
# B5: sufficiency
# =====================================================================================================================


def reading(rows: list[dict[str, Any]]) -> str:
    """The documented reading (task, desk procedure): which case a session's first two period rows show.

    - stalled: no steps, and the client's outbound backlog (``queued_blocks + dropped_blocks``) grows row to row;
    - healthy: words arrive;
    - quiet mic: no words and the loudest frame within 15 dB of the quietest (speech not above the room);
    - deaf recogniser: no words, steps continue, speech-level frames.
    """
    first, second = rows[0], rows[1]

    def backlog(r: dict[str, Any]) -> int:
        stt = r.get("stt") or {}
        return int(stt.get("queued_blocks", 0)) + int(stt.get("dropped_blocks", 0))

    if (
        first["steps_received"] == 0
        and second["steps_received"] == 0
        and backlog(second) > backlog(first)
    ):
        return "stalled"
    if first["words_received"] > 0:
        return "healthy"
    if first["max_frame_rms_dbfs"] - first["min_frame_rms_dbfs"] <= 15.0:
        return "quiet mic"
    return "deaf recogniser"


SPEECH = [(2_000, 3_200), (6_000, 7_000), (12_000, 13_000)]


def speech_level(room: float, voice: float) -> Callable[[float], Any]:
    return lambda t: voice if any(a <= t < b for a, b in SPEECH) else room


async def _scenario(
    path: Path, *, room: float, voice: float, words: bool, stalled: bool
) -> list[dict[str, Any]]:
    stt: Any = StalledStt() if stalled else reactive_stt(**stt_kwargs())
    async with mic_rig(
        path, audio=shaped(speech_level(room, voice)), stt=stt, script=[answer(REPLY)] * 3
    ) as rig:
        if words:
            for a, _ in SPEECH:
                rig.say("one two three four five", a)
        await rig.advance(21_000, step=FRAME_MS)
    rows = [r for r in rig.notes(LEVEL) if r.get("reason") == "period"]
    assert len(rows) >= 2, (
        f"T-0195 B1: two period rows in 21 s of capture; got {len(rows)} ({path.name}); notes "
        f"{sorted({n.get('what') for n in rig.kind('note')})}"
    )
    return rows


async def test_rows_tell_the_four_cases_apart(tmp_path: Path) -> None:
    """B5. Four scripted sessions, read with the documented reading above: (a) healthy (-55 room, -20 speech, words,
    steps); (b) quiet mic (-62 room, -55 'speech', steps, no words); (c) deaf recogniser (-55/-20, steps, no words);
    (d) stalled recogniser (-55/-20, no step, no word, backlog growing). Each reads as itself, and the facts the
    reading leans on are the ones B5 names."""
    a = await _scenario(tmp_path / "a", room=-55.0, voice=-20.0, words=True, stalled=False)
    b = await _scenario(tmp_path / "b", room=-62.0, voice=-55.0, words=False, stalled=False)
    c = await _scenario(tmp_path / "c", room=-55.0, voice=-20.0, words=False, stalled=False)
    d = await _scenario(tmp_path / "d", room=-55.0, voice=-20.0, words=False, stalled=True)
    got = {name: reading(rows) for name, rows in (("a", a), ("b", b), ("c", c), ("d", d))}
    assert got == {"a": "healthy", "b": "quiet mic", "c": "deaf recogniser", "d": "stalled"}, (
        f"B5: the rows do not tell the four cases apart; read {got}; first rows "
        f"{ {k: {f: v[0][f] for f in ('words_received', 'steps_received', 'max_frame_rms_dbfs', 'min_frame_rms_dbfs')} for k, v in (('a', a), ('b', b), ('c', c), ('d', d))} }"
    )
    assert a[0]["words_received"] >= 5 and 110 <= a[0]["steps_received"] <= 130, a[0]
    assert (
        abs(b[0]["max_frame_rms_dbfs"] - (-55.0)) < 0.05
        and abs(b[0]["min_frame_rms_dbfs"] - (-62.0)) < 0.05
    ), b[0]
    assert c[0]["words_received"] == 0 and c[0]["steps_received"] >= 110, c[0]
    assert abs(c[0]["max_frame_rms_dbfs"] - (-20.0)) < 0.05, c[0]
    assert (
        d[0]["steps_received"] == 0
        and d[1]["stt"]["dropped_blocks"] > d[0]["stt"]["dropped_blocks"]
    ), f"B5(d): stats() snapshots must show the backlog growing; {d[0]['stt']} -> {d[1]['stt']}"


# =====================================================================================================================
# B6: not model-facing
# =====================================================================================================================


async def test_feed_bytes_and_wire_identical_to_base(tmp_path: Path) -> None:
    """B6 (adversarial for "modifying the bytes fed to STT"). The capture holds tone, digital-zero, ``nan`` and ``inf``
    frames, then a muted span. ``stt.feed`` gets one call per captured frame, in order, with the frame's bytes exactly
    as captured (``nan`` and ``inf`` included) while live, and zeros of the same length while muted: BASE's pump
    (T-0181 A5). A typed question is answered with nothing of the diagnostics on the wire, in history or in
    ``turns.jsonl``; the non-finite frames are counted and no row carries a non-finite number. BASE-red only on the
    premise that diagnostic rows were written (the guard half is green at BASE)."""
    pattern = {0: -30.0, 1: ZERO, 2: NAN, 3: INF, 4: -45.0}

    def level(t: float) -> Any:
        return pattern[int(t // FRAME_MS) % 5]

    async with mic_rig(
        tmp_path, audio=shaped(level), stt_kwargs=stt_kwargs(), script=[answer(REPLY)]
    ) as rig:
        await rig.advance(4_000, step=FRAME_MS)
        await rig.type_question("what is on the screen")
        await rig.until(
            lambda: bool(rig.state.history),
            "premise: the typed question was answered",
            limit_ms=8_000,
        )
        await rig.engage()
        await rig.advance(2_000, step=FRAME_MS)
        await rig.release()
        await rig.advance(2_000, step=FRAME_MS)
        fed = list(rig.rec_stt.fed)
        yielded = list(rig.audio.yielded)
        history = list(rig.state.history)
    rows = require_rows(rig)
    assert len(fed) == len(yielded) and [t for _, t in fed] == [t for _, t in yielded], (
        f"B6: one feed per captured frame, same t_ms, same order; fed {len(fed)}, captured {len(yielded)}"
    )
    gates = rig.notes(GATE)
    m, u = gates[0]["frame_t_ms"], gates[1]["frame_t_ms"]
    changed = []
    for (pcm, t), (raw, _) in zip(fed, yielded, strict=True):
        want = bytes(len(raw)) if m <= t < u else raw
        if pcm != want:
            changed.append(t)
    assert not changed, (
        f"B6: the bytes fed to STT differ from BASE's at frames {changed[:10]} (of {len(changed)})"
    )
    for r in rows + rig.notes(CHECK):
        assert not finite_or_none(r), f"B1: non-finite values {finite_or_none(r)} in {r}"
    assert (
        sum(r["nonfinite_frames"] for r in rows) > 0 and sum(r["zero_frames"] for r in rows) > 0
    ), "B1: the nan/inf and digital-zero frames are counted"
    wire = rig.published_text()
    for word in (LEVEL, CHECK, "rms_dbfs", "muted_frames"):
        assert word not in wire, f"B6: {word!r} reached the wire"
    turns = "\n".join(rig.lines())
    assert turns and LEVEL not in turns and CHECK not in turns and "rms_dbfs" not in turns, (
        "B6: turns.jsonl carries no diagnostics"
    )
    assert len(history) == 1 and history[0].user_text == "what is on the screen", (
        f"B6: history {history}"
    )
    assert history[0].assistant_spoken.strip() == REPLY.strip(), (
        f"B6: only the spoken reply is stored; {history}"
    )
    assert LEVEL not in json.dumps([h.model_dump() for h in history]), (
        "B6: history carries no diagnostics"
    )
    for d in rig.kind("stt.decision"):
        assert set(d) == DECISION_KEYS, f"B6: stt.decision shape unchanged; {sorted(d)}"


async def test_no_text_or_pcm_in_diagnostic_rows(tmp_path: Path) -> None:
    """B6 (adversarial for "word text in a row"). Distinctive words are heard, an estimated word included; no
    ``mic.level`` or ``mic.unmute_check`` row carries their text, a list, or anything but numbers, booleans, null, the
    two strings B1 names and the ``stt`` mapping of numbers; each row is small (no PCM)."""
    secret = "Zanzibar quixotic marmalade"
    async with mic_rig(
        tmp_path, audio=shaped(const(-30.0)), stt_kwargs=stt_kwargs(), script=[answer(REPLY)] * 2
    ) as rig:
        await rig.advance(1_000, step=FRAME_MS)
        await rig.engage()
        await rig.advance(400, step=FRAME_MS)
        u = await rig.release()
        rig.say(secret, u + 300)
        rig.rec_stt.inner.script.append(FakeStt.open_word("Ophelia", u + 2_500, u + 2_500))
        await rig.advance(12_000, step=FRAME_MS)
    rows = require_rows(rig) + require_rows(rig, CHECK)
    assert sum(r["words_received"] for r in rig.notes(CHECK)) >= 3, (
        "premise: the words were heard after the unmute"
    )
    assert any(r.get("words_estimated", 0) >= 1 for r in rows), (
        "premise: the estimated word was counted"
    )
    for r in rows:
        text = json.dumps(r)
        for w in (*secret.split(), "Ophelia"):
            assert w.lower() not in text.lower(), f"B6: word text {w!r} in {r}"
        assert len(text) < 2_000, f"B6: a {len(text)}-byte diagnostic row (PCM?)"
        for k, v in r.items():
            if k == "stt":
                assert v is None or (
                    isinstance(v, dict) and all(isinstance(x, int | float) for x in v.values())
                ), f"B6: stt is a mapping of numbers; {v}"
            elif k == "what":
                assert v in (LEVEL, CHECK)
            elif k == "reason":
                assert v in ("period", "gate")
            else:
                assert v is None or isinstance(v, bool | int | float), (
                    f"B6: {k} = {v!r} in {r.get('what')}"
                )


# =====================================================================================================================
# B1: the stats snapshot
# =====================================================================================================================


async def test_stats_snapshot_present_or_null(tmp_path: Path) -> None:
    """B1/B4: ``stt`` is the full ``stats()`` mapping, one snapshot taken for the row, when the client has ``stats()``;
    ``null`` when it has none (and the rows are still written: a session over a client without ``stats()`` must not
    lose its diagnostics, nor its capture pump)."""
    snap = SnapshotStt(FakeStt(reactive=True, steps=True, lag_ms=LAG_MS))
    async with mic_rig(tmp_path / "a", audio=shaped(const(-30.0)), stt=snap) as rig:
        await rig.advance(1_000, step=FRAME_MS)
        await rig.engage()
        await rig.advance(400, step=FRAME_MS)
        await rig.release()
        await rig.advance(10_500, step=FRAME_MS)
        snapshots = list(snap.snapshots)
    rows = require_rows(rig) + require_rows(rig, CHECK)
    keys = set(snapshots[-1])
    for r in rows:
        stt = r["stt"]
        assert isinstance(stt, dict) and set(stt) == keys, (
            f"B1: stt is the full stats() mapping ({sorted(keys)}); got {stt}"
        )
        assert 1 <= stt["serial"] <= len(snapshots) and snapshots[stt["serial"] - 1] == stt, (
            f"B1: stt is one snapshot stats() returned, unaltered; got {stt}"
        )
    serials = [r["stt"]["serial"] for r in rows]
    assert len(set(serials)) == len(serials), f"B1: one stats() call per row; serials {serials}"
    # review A F2: exactly one stats() call per row. The BASE per-frame caller (``_check_stt_drops``, T-0007a A9)
    # also calls it, so the count is of every other call; a row that asks twice dies here.
    row_calls = [c for c in snap.callers if c != "_check_stt_drops"]
    assert "_check_stt_drops" in snap.callers, (
        "premise: the BASE per-frame stats() caller is still named so"
    )
    assert len(row_calls) == len(rows), (
        f"B1: stats() once per row; {len(rows)} rows, {len(row_calls)} calls outside _check_stt_drops "
        f"({sorted(set(row_calls))})"
    )

    bare = NoStatsStt(FakeStt(reactive=True, steps=True, lag_ms=LAG_MS))
    async with mic_rig(tmp_path / "b", audio=shaped(const(-30.0)), stt=bare) as rig:
        await rig.advance(1_000, step=FRAME_MS)
        await rig.mute(True)  # over the wire; ``engage`` needs a RecordingStt
        await rig.advance(400, step=FRAME_MS)
        await rig.mute(False)
        await rig.advance(10_500, step=FRAME_MS)
        fed = len(bare.fed)
        failures = list(rig.session.failures)
    assert not failures, f"B1: the capture pump died over a client without stats(): {failures}"
    assert fed >= 140, f"B1: the pump kept feeding; {fed} frames"
    rows = require_rows(rig) + require_rows(rig, CHECK)
    for r in rows:
        assert "stt" in r and r["stt"] is None, f"B1: stt is null without stats(); {r}"


# =====================================================================================================================
# Group B census amendment: the differentials' filter admits exactly the two diagnostic notes
# =====================================================================================================================


async def test_differential_filter_admits_exactly_the_diagnostic_notes(tmp_path: Path) -> None:
    """Census amendment (orchestrator ruling 2026-10-29). A real muted-and-unmuted session's log, through the shared
    funnel (``_t0177_pins.without_vision_frames``), loses exactly its ``mic.level`` and ``mic.unmute_check`` notes:
    every other record survives in order (``mic.gate``, ``mic.mute``, all the rest), and ``log.stopped.records`` drops
    by the number removed. A diagnostic note of the wrong shape (an extra key, a text value) is refused, not
    removed. Mutation: admitting ``mic.gate`` too reddens this row."""
    from tests.unit import _t0195_pins as pins
    from tests.unit._t0177_pins import without_vision_frames

    async with mic_rig(tmp_path, audio=shaped(const(-30.0)), stt_kwargs=stt_kwargs()) as rig:
        await rig.advance(1_000, step=FRAME_MS)
        await rig.engage()
        await rig.advance(400, step=FRAME_MS)
        await rig.release()
        await rig.advance(10_500, step=FRAME_MS)
    records = [{k: v for k, v in r.items() if k != "seq"} for r in rig.log_records]
    diag = [
        r for r in records if r.get("kind") == "note" and r["data"].get("what") in (LEVEL, CHECK)
    ]
    assert len(diag) >= 4, (
        f"premise: gate rows, a period row and an unmute check were written; {len(diag)}"
    )
    filtered = without_vision_frames(records)
    kept = [r for r in records if r not in diag]
    stopped = [r for r in kept if r.get("kind") == "log.stopped"]
    assert stopped, "premise: the log closed with log.stopped"
    want = [
        {**r, "data": {**r["data"], "records": r["data"]["records"] - len(diag)}}
        if r.get("kind") == "log.stopped"
        else r
        for r in kept
    ]
    assert filtered == want, (
        "census amendment: the filter removes exactly the mic.level/mic.unmute_check notes and recounts log.stopped; "
        f"kept {len(filtered)} of {len(records)} records, want {len(want)}"
    )
    whats = [r["data"].get("what") for r in filtered if r.get("kind") == "note"]
    assert whats.count(GATE) == 2 and "mic.mute" in whats, (
        f"census amendment: mic.gate/mic.mute kept; {whats}"
    )

    level = diag[0]["data"]
    for bad in (
        {**level, "text": "hello"},
        {**level, "reason": "hello there"},
        {**level, "stt": {"words": "hi"}},
    ):
        with pytest.raises(pins.DiagnosticPinError):
            pins.without_diagnostic_notes([{"kind": "note", "data": bad}])
    short = {k: v for k, v in level.items() if k != "peak_dbfs"}
    with pytest.raises(pins.DiagnosticPinError):
        pins.without_diagnostic_notes([{"kind": "note", "data": short}])
