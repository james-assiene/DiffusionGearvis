"""T-0182 group B through the session: no clip by default (B1/B2), one ``still_working`` in a long live tool loop (B4),
the rules between clips (B5) and a barge-in or a commit silences a playing clip (B7) (docs/tasks/T-0182.md, group B
and its amendments B-R1..B-R5). B9 (the turn line's ``extras.clips``) moved to T-0182c (B-R5): group B adds nothing to
turn lines, so these rows read clip plays off the machine's own ``PlayClip`` commands and the audio module's plays.

**Rigs.** ``test_session_loop.running`` (a conversational turn) and ``_tool_turn_support.tool_rig`` (a tool-loop turn,
``dry_run`` kit) with ``policy=`` the ``TurnPolicy`` ``cli`` builds (B-R4: ``ack_clip=False`` unless filler clips are
on; B4: in live with the loop on, ``still_working_after_ms = 20 000``; the session never asks which mode it is in).
``TurnPolicy()`` itself keeps BASE_B's ack (B-R4), so "defaults" here is ``test_t0182_clip_policy.cli_default``. The 20 s loop is real virtual time:
``RecordingToolExecutor(hang=True)`` holds every call until the session's own 10 s execute timeout
(``TOOL_EXEC_TIMEOUT_MS``, T-0114) answers it with ``ToolResult(timeout)``, and ``script_steps`` scripts the next
steps, so three calls run about 30 s. B7 runs the real ``PipeWireAudioIO`` on ``FakeAudioDevice`` (the device clock is
the rig's scripted clock): the ack asset is 2 s of the constant +0.5 (s16 16 384) at ``clips.clip_path(data_dir,
"ack")``, the TTS double's speech is the constant -0.25 (s16 -8 192), so every sample the device received is
attributed by its sign; "clip audio after the stop" is the count of > +0.375 samples in the blocks written after the
event, at 48 kHz.

**Spies.** ``filler_policy.decide`` and ``Session._interim_filler`` are wrapped (delegating to the real ones): B2 "off:
``_interim_filler`` is never called" is read off the second, and the first proves no filler decision is taken.

**Readings (reported):**

- B-R2's stale ack flag is pinned: with ``ack_clip=True`` in a silent live loop the machine emits no ``still_working``
  (the ack's ``clip_playing`` is never cleared), and the turn plays at most one ``still_working`` in total.
- B7's "≤ 300 ms" is read on the device's received blocks (an upper bound on what was heard after the event: the
  writer keeps one block ahead of the device clock).

Activated for ``/implement T-0182`` group B (the rows were pending through e1ca276f). At BASE_B (7646b8b6) ``TurnPolicy`` has neither B1/B4 field, so the
default and loop rows fail on ``clip_policy``'s named assertion (were the fields stubbed, BASE's machine would still
play the ack at every thinking turn's llm.start and its session run ``_interim_filler`` on every THINKING tick); the
B7 rows run BASE's own ack and fail on the clip audio heard after the stop (about 1.9 s of it).

Regime: CPU over the fakes, ``ScriptedClock`` (virtual time), thinking on, n = 1 and deterministic per row; the B7
rows wait on wall time for the audio writer thread (``audio_device.settle``), as ``test_t0110_speed_guards``'s module
rig does. No measurement.
"""

from __future__ import annotations

import typing
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_astra.conductor import session as session_mod
from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from local_astra.reflex import clips, filler_policy
from tests.fakes.audio_device import FakeAudioDevice, make_audio_io, settle
from tests.fakes.tool_executor import RecordingToolExecutor
from tests.fakes.tts import BYTES_PER_MS, FakeUtterance
from tests.unit._tool_turn_support import VALID_CLICK, ToolRig, tool, tool_rig
from tests.unit.test_session_loop import (
    RecordingTts,
    Rig,
    ScriptedClock,
    answer,
    frame,
    running,
    thought,
)
from tests.unit.test_t0182_clip_policy import cli_default, clip_policy

#: B4: ``Profile.still_working_after_s`` default (20) in ms, what ``cli`` sets in live with the loop on
LIVE_W = 20_000
#: T-0114 H-T114-4: the session's execute timeout, which answers a hanging call (written out from the task)
EXEC_TIMEOUT_MS = 10_000
QUESTION = "open the settings page"
REPORT = "The settings page is open."
#: two more valid v1 calls (click_at in range), written by hand
CLICK2 = "<|tool_call>call:click_at{x:120,y:640}<tool_call|>"
CLICK3 = "<|tool_call>call:click_at{x:900,y:40}<tool_call|>"
#: refused at detection (no v1 tool name): the not_finished ending (T-0178 B3)
NAMELESS = '<|tool_call>call:navigate{url:<|"|>https://example.org<|"|>}<tool_call|>'
ENDING = frozenset({"out_of_steps", "not_finished", "lost_thread"})

#: B7: the clip asset's and the speech's constant levels (s16), and the thresholds that attribute a device sample
CLIP_S16 = 16_384  # +0.5
SPEECH_S16 = -8_192  # -0.25
CLIP_ABOVE = 0.375
SPEECH_BELOW = -0.125
DEVICE_PER_MS = 48  # FakeAudioDevice renders 48 kHz
B7_BUDGET_MS = 300  # bench B4: onset to silence


# --- spies and helpers -----------------------------------------------------------------------------------------------


@dataclass
class Spies:
    decisions: list[str] = field(default_factory=list)
    interim: list[int] = field(default_factory=list)


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Spies:
    """``filler_policy.decide`` and ``Session._interim_filler``, wrapped; both delegate to the real ones."""
    found = Spies()
    real_decide = filler_policy.decide
    real_interim = session_mod.Session._interim_filler

    def decide(inputs: Any, cfg: Any = None) -> Any:
        out = real_decide(inputs, cfg)
        found.decisions.append(out)
        return out

    async def interim(self: Any, now: int) -> None:
        found.interim.append(now)
        await real_interim(self, now)

    monkeypatch.setattr(filler_policy, "decide", decide)
    monkeypatch.setattr(session_mod.Session, "_interim_filler", interim)
    return found


def install_assets(data_dir: Path, ms: int = 1000, level: int = 4000) -> None:
    """A rendered asset for every kind ``CLIP_TEXT`` registers (the render run's result), ``ms`` long."""
    for kind in session_mod.CLIP_TEXT:
        path = Path(clips.clip_path(data_dir, kind))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.full(24 * ms, level, dtype="<i2").tobytes())


def live_loop(**overrides: Any) -> TurnPolicy:
    """The ``TurnPolicy`` ``cli`` builds in live with the loop on and defaults (B4; B-R4: ack off), budget 8 unless
    overridden; ``ack_clip=True`` is the filler-clips-on cell."""
    overrides.setdefault("max_tool_steps", 8)
    return cli_default(tool_loop=True, still_working_after_ms=LIVE_W, **overrides)


def ack_on() -> TurnPolicy:
    """B2 on: ``TurnPolicy(thinking=True)``, which after B-R4 is ``ack_clip=True`` (BASE's ack) on both trees, so a B7
    row runs the same ack at BASE_B and after, and fails at BASE_B on B7 itself."""
    return TurnPolicy(thinking=True)


def machine_clips(rig: Any) -> list[tuple[E.Event, str]]:
    """Every ``PlayClip`` the machine emitted, with the event that caused it, in order."""
    return [
        (ev, c.kind) for ev, _, cmds in rig.machine.log for c in cmds if isinstance(c, C.PlayClip)
    ]


def llm_start_ms(rig: Any) -> int:
    starts = [
        ev.t_ms for ev, _, cmds in rig.machine.log if any(isinstance(c, C.LlmStart) for c in cmds)
    ]
    assert len(starts) == 1, f"premise: one llm.start; got {starts}"
    return starts[0]


def clip_kinds_played(rig: Any) -> list[str]:
    """What the audio module was handed with ``source=""`` (a clip), by kind, in order."""
    return [getattr(u, "kind", "?") for u in rig.audio.clip_plays()]


def assert_no_clip_text_anywhere(rig: Any) -> None:
    """No clip ever enters history, ``astra.spoken`` text or ``TtsSpeak.source`` (task Goal)."""
    texts = [t for t in session_mod.CLIP_TEXT.values() if t]
    sources = [c.source for c in rig.machine.commands(C.TtsSpeak)]
    spoken = [t for _, t, _ in rig.tts.requests]
    history = [h.assistant_spoken for h in rig.state.history]
    for where, values in (("TtsSpeak.source", sources), ("tts text", spoken), ("history", history)):
        bad = [v for v in values for t in texts if t in v]
        assert not bad, f"T-0182: clip text reached {where}: {bad}"


async def loop_turn(rig: ToolRig, *, limit_ms: int = 60_000) -> None:
    await rig.ask(QUESTION)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()),
        "the loop turn never ended",
        limit_ms=limit_ms,
        step=50,
    )


# =====================================================================================================================
# B1/B2: defaults play nothing and never enter the interim filler
# =====================================================================================================================


async def test_defaults_a_10s_conversational_turn_plays_no_clip(
    tmp_path: Path, spies: Spies
) -> None:
    """B1/B2 (defaults as ``cli`` resolves them, B-R4: ``ack_clip=False``; assets installed, as after the render
    run): a conversational turn held 10 s in THINKING plays no clip, the machine emits no ``PlayClip``,
    ``_interim_filler`` is never entered and ``decide`` is never asked."""
    install_assets(tmp_path)
    async with running(
        tmp_path, script=[answer("Here it is. ")], frames=[frame(0, "a")], policy=cli_default()
    ) as rig:
        rig.llm.hold()
        await rig.ask("what is on this screen")
        assert rig.state.state is TurnState.THINKING, "premise: the generate is held"
        await rig.advance(10_000, step=250)
        assert rig.state.state is TurnState.THINKING, "premise: 10 s in THINKING"
        assert machine_clips(rig) == [], (
            f"T-0182 B1: no PlayClip by default; got {[k for _, k in machine_clips(rig)]}"
        )
        assert clip_kinds_played(rig) == [], (
            f"T-0182 B1: no clip played; got {clip_kinds_played(rig)}"
        )
        assert spies.interim == [], (
            f"T-0182 B2: with filler clips off _interim_filler is never called; it was, at {spies.interim[:5]}"
        )
        assert spies.decisions == [], spies.decisions
        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
    assert clip_kinds_played(rig) == []
    assert rig.state.history[-1].assistant_spoken == "Here it is. "


async def test_ack_on_plays_the_ack_and_runs_the_interim_filler(
    tmp_path: Path, spies: Spies
) -> None:
    """B2 on (amendment B-R1, one switch: ``TurnPolicy(ack_clip=True)``), no asset installed: the machine's ack plays
    (a zero-length clip) and ``_interim_filler`` runs on THINKING ticks as at BASE_B."""
    async with running(
        tmp_path,
        script=[answer("Here it is. ")],
        frames=[frame(0, "a")],
        policy=clip_policy(ack_clip=True),
    ) as rig:
        rig.llm.hold()
        await rig.ask("what is on this screen")
        await rig.advance(5_000, step=250)
        assert [k for _, k in machine_clips(rig)] == ["ack"]
        assert spies.interim, "T-0182 B2: with filler clips on _interim_filler runs as at BASE_B"
        assert spies.decisions, "T-0182 B2: and asks filler_policy.decide"
        rig.llm.release()
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
    assert clip_kinds_played(rig)[0] == "ack", clip_kinds_played(rig)
    assert_no_clip_text_anywhere(rig)


# =====================================================================================================================
# B4: the t9 shape -- one still_working at 20 s in a long live loop
# =====================================================================================================================


async def test_t9_shape_one_still_working_at_20s(tmp_path: Path, spies: Spies) -> None:
    """B4 (the t9 shape): a live-loop turn whose three calls each hang until the 10 s execute timeout runs about
    30 s. Exactly one ``still_working`` plays: the machine emits it on the first ``Tick`` at or after llm.start +
    20 000 ms (every guard holds there), the audio module plays one clip, and nothing else plays. No clip text in
    history, ``TtsSpeak.source`` or speech; ``_interim_filler`` is never entered."""
    install_assets(tmp_path)
    executor = RecordingToolExecutor(hang=True)
    async with tool_rig(
        tmp_path,
        policy=live_loop(),
        executor=executor,
        script=[thought("open the menu first"), tool(VALID_CLICK)],
    ) as rig:
        rig.llm.script_steps([tool(CLICK2)], [tool(CLICK3)], [answer(REPORT)])
        await loop_turn(rig)
        t0 = llm_start_ms(rig)
        found = machine_clips(rig)
        assert [k for _, k in found] == ["still_working"], (
            f"T-0182 B4: one still_working in a long live loop turn; got {[(type(e).__name__, e.t_ms - t0, k) for e, k in found]}"
        )
        (ev, _), deadline = found[0], t0 + LIVE_W
        assert isinstance(ev, E.Tick) and ev.t_ms >= deadline, (
            f"T-0182 B4: on a Tick at or after llm.start + {LIVE_W} ms; got {type(ev).__name__} at +{ev.t_ms - t0}"
        )
        earlier = [e.t_ms for e in rig.machine.events(E.Tick) if deadline <= e.t_ms < ev.t_ms]
        assert earlier == [], (
            f"T-0182 B4: the first Tick where every guard holds; ticks at {[t - t0 for t in earlier]} played nothing"
        )
        assert len(executor.invocations) == 3, "premise: three calls ran (the turn passed 20 s)"
        assert clip_kinds_played(rig) == ["still_working"], clip_kinds_played(rig)
        assert spies.interim == [], (
            f"T-0182 B2: filler clips off; _interim_filler ran at {spies.interim[:5]}"
        )
        assert rig.state.history[-1].assistant_spoken == REPORT, rig.state.history[-1]
        assert_no_clip_text_anywhere(rig)


async def test_a_loop_turn_shorter_than_20s_plays_nothing(tmp_path: Path, spies: Spies) -> None:
    """B4, the other direction at session level: the same live-loop policy, one call that hangs until its 10 s
    timeout and a report: the turn ends before 20 s and plays no clip."""
    install_assets(tmp_path)
    executor = RecordingToolExecutor(hang=True)
    async with tool_rig(
        tmp_path,
        policy=live_loop(),
        executor=executor,
        script=[thought("open the menu first"), tool(VALID_CLICK)],
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await loop_turn(rig)
        t0 = llm_start_ms(rig)
        assert rig.clock.now_ms - t0 < LIVE_W, (
            f"premise: the turn ended before 20 s ({rig.clock.now_ms - t0})"
        )
        assert machine_clips(rig) == [] and clip_kinds_played(rig) == [], machine_clips(rig)


# =====================================================================================================================
# B5: still_working, then the ending clip: both play, in order (the "clips count as voiced" mutant)
# =====================================================================================================================


@pytest.mark.parametrize(
    ("last_step", "ending"),
    [
        pytest.param(CLICK3, "out_of_steps", id="out_of_steps"),
        pytest.param(NAMELESS, "not_finished", id="not_finished"),
    ],
)
async def test_b5_still_working_then_the_ending_clip(
    tmp_path: Path, spies: Spies, last_step: str, ending: str
) -> None:
    """B5 (H-T182-3): budget 2; calls 1 and 2 hang to their timeouts (the turn passes 20 s, ``still_working``
    plays); step 3 is the report step and ends with a valid call past the budget (``out_of_steps``) or a refused
    draft (``not_finished``). Both clips play, ``still_working`` first, through the playback queue; the loop report
    says the ending clip played; a clip never counts as voiced, so the honest ending is not suppressed (the mutant
    that counts the still_working as voiced loses the ending clip); no clip text anywhere."""
    install_assets(tmp_path)
    executor = RecordingToolExecutor(hang=True)
    async with tool_rig(
        tmp_path,
        policy=live_loop(max_tool_steps=2),
        executor=executor,
        script=[thought("open the menu first"), tool(VALID_CLICK)],
    ) as rig:
        rig.llm.script_steps([tool(CLICK2)], [tool(last_step)])
        await loop_turn(rig)
        kinds = [k for _, k in machine_clips(rig)]
        assert kinds == ["still_working", ending], (
            f"T-0182 B5: one progress clip, then the ending clip; got {kinds}"
        )
        assert clip_kinds_played(rig) == ["still_working", ending], clip_kinds_played(rig)
        assert_no_clip_text_anywhere(rig)
    record = rig.record("t1")
    assert record.extras.get("loop_report", {}).get("clip") is True, record.extras.get(
        "loop_report"
    )


# =====================================================================================================================
# B5 / B-R2: filler clips on in a live loop -- at most one still_working, and the stale ack flag pinned
# =====================================================================================================================


async def test_filler_clips_on_in_a_live_loop_at_most_one_still_working(
    tmp_path: Path, spies: Spies
) -> None:
    """B5 ("with filler_clips on in a live loop, at most one still_working per turn in total") and the B-R2 pin: with
    ``ack_clip=True`` the ack plays at llm.start and nothing in a silent tool loop clears ``clip_playing``, so the
    machine emits no ``still_working`` (accepted; a change here must be deliberate); the session's interim filler
    runs as at BASE_B; the turn plays at most one ``still_working`` in total."""
    install_assets(tmp_path)
    executor = RecordingToolExecutor(hang=True)
    async with tool_rig(
        tmp_path,
        policy=live_loop(ack_clip=True),
        executor=executor,
        script=[thought("open the menu first"), tool(VALID_CLICK)],
    ) as rig:
        rig.llm.script_steps([tool(CLICK2)], [tool(CLICK3)], [answer(REPORT)])
        await rig.ask(QUESTION)
        t0 = llm_start_ms(rig)
        await rig.until(
            lambda: rig.clock.now_ms >= t0 + LIVE_W + 500, "premise: 20 s", limit_ms=30_000
        )
        assert rig.state.state is TurnState.THINKING, "premise: the loop is still running at 20.5 s"
        assert rig.state.clip_playing is True, (
            "B-R2 pin: the ack's clip_playing is never cleared in a silent tool loop"
        )
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()),
            "the loop turn never ended",
            limit_ms=30_000,
        )
        machine = [k for _, k in machine_clips(rig)]
        assert machine == ["ack"], (
            f"B-R2 pin: the machine plays the ack and no still_working; got {machine}"
        )
        assert spies.interim, "T-0182 B2: with filler clips on _interim_filler runs as at BASE_B"
        played = clip_kinds_played(rig)
        assert played[0] == "ack" and played.count("still_working") <= 1, (
            f"T-0182 B5: at most one still_working per turn in total; got {played}"
        )
        assert set(played) <= {"ack", "still_working"}, played
        assert_no_clip_text_anywhere(rig)


# =====================================================================================================================
# B7: a stop silences the playing clip (real PipeWireAudioIO on FakeAudioDevice)
# =====================================================================================================================


class LevelUtterance(FakeUtterance):
    """``FakeUtterance`` whose audio is the constant ``SPEECH_S16`` instead of silence (same duration and chunks)."""

    async def audio(self) -> AsyncIterator[bytes]:
        sent = 0
        while sent < self.duration_ms and not self.cancelled:
            n = min(self.chunk_ms, self.duration_ms - sent)
            if self.first_audio_latency_ms_value is None:
                self.first_audio_latency_ms_value = 0.0
            self.chunks_sent += 1
            sent += n
            yield np.full(n * BYTES_PER_MS // 2, SPEECH_S16, dtype="<i2").tobytes()


class LevelTts(RecordingTts):
    async def speak(self, utterance_id: str, text: str, voice: str = "default") -> Any:
        self.requests.append((utterance_id, text, voice))
        u = LevelUtterance(utterance_id, text, self.ms_per_word)
        self.spoken.append(u)
        return u


def _ms(blocks: Sequence[np.ndarray], keep: typing.Callable[[np.ndarray], np.ndarray]) -> float:
    return sum(int(np.count_nonzero(keep(b))) for b in blocks) / DEVICE_PER_MS


def clip_ms(blocks: Sequence[np.ndarray]) -> float:
    return _ms(blocks, lambda b: b > CLIP_ABOVE)


def speech_ms(blocks: Sequence[np.ndarray]) -> float:
    return _ms(blocks, lambda b: b < SPEECH_BELOW)


def _ack_asset(data_dir: Path) -> None:
    """The ack asset: 2 s of +0.5, where ``Session._clip_pcm`` reads it (``clips.clip_path``)."""
    path = Path(clips.clip_path(data_dir, "ack"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.full(48_000, CLIP_S16, dtype="<i2").tobytes())


async def _module_rig(tmp_path: Path, script: Sequence[Any]) -> Any:
    clock = ScriptedClock()
    device = FakeAudioDevice(clock, buffer_ms=200.0)
    audio = make_audio_io(device)

    async def pump() -> None:
        await settle(device)

    ctx = running(
        tmp_path,
        script=list(script),
        frames=[frame(0, "a")],
        clock=clock,
        audio=audio,
        settle_audio=pump,
        tts=LevelTts(ms_per_word=200),
        policy=ack_on(),
    )
    return ctx, device


async def _clip_started(rig: Rig, device: FakeAudioDevice) -> int:
    """Ask, then let the ack play 100 ms on the device clock; returns the block index at that moment."""
    rig.llm.hold()
    await rig.ask("what is on this screen")
    assert rig.state.state is TurnState.THINKING, "premise: thinking, the generate held"
    assert [k for _, k in machine_clips(rig)] == ["ack"], "premise: the ack was emitted"
    await rig.advance(100, step=10)
    at = len(device.blocks)
    heard = clip_ms(device.blocks[:at])
    assert heard >= 50, f"premise: the 2 s ack is playing ({heard:.0f} ms of it so far)"
    return at


#: B7 chunk loop: an asset that is not a whole number of chunks, 2 s + 10 ms at 24 kHz (written out by hand), and what
#: reaches the 48 kHz device when nothing stops it: 2010 ms x 48 samples
WHOLE_TTS_SAMPLES = 48_240
WHOLE_DEVICE_SAMPLES = 2_010 * DEVICE_PER_MS
#: tolerance: half the 10 ms tail (240 device samples). The device's resampler rounds the clip's two edges below
#: ``CLIP_ABOVE`` (measured at 8f836adc+: 96 433, 47 samples short); dropping the tail loses 480 more.
WHOLE_TOLERANCE = 5 * DEVICE_PER_MS


async def test_b7_an_unstopped_clip_reaches_the_device_whole(tmp_path: Path) -> None:
    """B7 (the chunked clip must still play whole): with the generate held and nothing stopping it, an ack asset of
    2 s + 10 ms (a final piece shorter than a chunk) reaches the ``FakeAudioDevice`` whole: its sample count above
    ``CLIP_ABOVE`` is 2010 ms x 48 = 96 480, within 5 ms (half the tail). A loop that plays only the first pieces (100 ms) or drops
    the last partial piece (10 ms = 480 device samples) is red."""
    path = Path(clips.clip_path(tmp_path, "ack"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.full(WHOLE_TTS_SAMPLES, CLIP_S16, dtype="<i2").tobytes())
    ctx, device = await _module_rig(tmp_path, [answer("Here it is now. ")])
    async with ctx as rig:
        rig.llm.hold()
        await rig.ask("what is on this screen")
        assert rig.state.state is TurnState.THINKING, "premise: thinking, the generate held"
        assert [k for _, k in machine_clips(rig)] == ["ack"], "premise: the ack was emitted"
        await rig.advance(3_000, step=10)  # under interim_update_ms (4 s): no second clip
        assert [k for _, k in machine_clips(rig)] == ["ack"], "premise: one clip only"
        assert not any(isinstance(c, C.StopClip) for _, _, cmds in rig.machine.log for c in cmds), (
            "premise: nothing stopped the clip"
        )
        heard = sum(int(np.count_nonzero(b > CLIP_ABOVE)) for b in device.blocks)
        assert abs(heard - WHOLE_DEVICE_SAMPLES) <= WHOLE_TOLERANCE, (
            f"T-0182 B7: an unstopped clip plays whole; {heard} of {WHOLE_DEVICE_SAMPLES} device samples "
            f"({heard / DEVICE_PER_MS:.1f} of 2010 ms)"
        )


async def test_b7_a_barge_in_silences_the_playing_clip(tmp_path: Path) -> None:
    """B7 (bench B4 budget): a barge-in 100 ms into the 2 s ack (THINKING onset: ``StopClip``) leaves at most
    300 ms more clip audio on the device; at BASE_B the whole asset was already queued and about 1.9 s more plays."""
    _ack_asset(tmp_path)
    ctx, device = await _module_rig(tmp_path, [answer("Here it is now. ")])
    async with ctx as rig:
        at = await _clip_started(rig, device)
        await rig.onset()
        onset_cmds = [
            c for ev, _, cmds in rig.machine.log if isinstance(ev, E.UserSpeechOnset) for c in cmds
        ]
        assert any(isinstance(c, C.StopClip) for c in onset_cmds), (
            "premise: the THINKING onset emitted StopClip"
        )
        await rig.advance(1_000, step=10)
        after = clip_ms(device.blocks[at:])
        assert after <= B7_BUDGET_MS, (
            f"T-0182 B7: {after:.0f} ms of the clip played after the barge-in (budget {B7_BUDGET_MS} ms)"
        )


async def test_b7_a_commit_cuts_the_clip_and_its_answer_plays_in_full(tmp_path: Path) -> None:
    """B7 (amendment B-R1: "a commit cuts the playing clip while that commit's answer plays in full"): 100 ms into
    the 2 s ack the answer commits (THINKING→SPEAKING: ``StopClip``, then ``TtsSpeak``). At most 300 ms more clip
    audio plays, the answer's 800 ms (4 words x 200 ms) play whole, and history holds the answer."""
    _ack_asset(tmp_path)
    text = "Here it is now. "
    ctx, device = await _module_rig(tmp_path, [answer(text)])
    async with ctx as rig:
        at = await _clip_started(rig, device)
        rig.llm.release()
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "the turn never ended",
            limit_ms=6_000,
            step=10,
        )
        after = device.blocks[at:]
        assert clip_ms(after) <= B7_BUDGET_MS, (
            f"T-0182 B7: {clip_ms(after):.0f} ms of the clip played after the commit (budget {B7_BUDGET_MS} ms)"
        )
        want = 200 * len(text.split())
        assert speech_ms(after) >= want - 40, (
            f"T-0182 B7: the commit's answer plays in full; {speech_ms(after):.0f} of {want} ms reached the device"
        )
        assert rig.state.history[-1].assistant_spoken == text, rig.state.history[-1]
