"""T-0199: a voice turn's onset frame is looked up on the session clock, not the audio clock.

``docs/tasks/T-0199.md`` criteria 1-13. A voice ``UserSpeechOnset`` carries ``onset_ms`` on the **audio** clock (the
STT tap's capture position, from 0 at capture open); ``BrowserFrameSource`` stamps ``Frame.ts_ms`` on the **session**
clock. The machine emits ``SnapshotFrame(at_ms=onset_ms)``; after this task the session converts that time, for the
frame lookups of the snapshot a *voice* onset produced and for nothing else, with a mapping measured from the capture
frames ``_pump_capture`` already handles.

**The rig** (``onset_rig``): ``_tool_turn_support.tool_rig`` (tools off, or tools live with vision ``shared`` selected
over a socketless ``HudServer`` page, as in T-0177's ``vision_rig``) with three things swapped in through a ``Session``
subclass, the way ``vision_rig`` swaps its perception:

- the clock starts at ``ARM_MS`` = 11 400, so capture is armed at session 11 400;
- ``FakeAudioIO(capture_origin_ms=0)`` (T-0199's fake change): STT frame ``t_ms`` = elapsed since the arm, so audio
  time = session time - 11 400 ("behind"); or ``test_session_listening.OffsetAudio``, audio = session + 5 000 000
  ("ahead", criterion 5);
- ``FakeStt(reactive=True, steps=True, lag_ms=500)`` scripted **in audio time** (it releases a word only once the fed
  audio passes its end + lag), and a real ``BrowserFrameSource(now_ms=clock)`` wrapped by ``RecordingSource`` (pure
  delegation, plus a log of ``(source, at_ms, returned frame)``) that ingests ``AstraFrame``s at chosen session
  times through ``clock.call_at``.

HUD frames share all but the last hex digit of their phash (<= 4 bits apart, the scene threshold is 10), so the
source announces no scene change and every frame is its own identity.

**Expected values by hand.** The user speaks "what is on my screen" from audio 21 040 (behind) = session 32 440; with
200 ms per word the first word ends at audio 21 240 and is released when fed audio reaches 21 740, i.e. by the STT
frame [21 760, 21 840), handled at session 33 240 -- the delivery stamp ``t_ms`` is ~800 ms after the onset's session
time. The mapping errs late by the capture-to-handling latency, exact in the fake when a frame is handled at its
completion; the tolerance used is one STT frame (80 ms), so the expected onset frame is never within 80 ms of a frame
boundary (the nearest are 200 ms away).

**Seams assumed** (the task's names, ``docs/tasks/T-0199.md`` criteria 9 and 11): the session note
``{"what": "onset.clock", "onset_ms", "at_session_ms", "method": "capture"|"delivery"}``, one per voice-onset frame
lookup, read back from the real ``SessionLog`` (``rig.log_records``). The lookup time is read at the perception
boundary (``get_frame``'s ``at_ms``), never from session attributes.

**BASE-green guards** (deliberately green at BASE; their teeth are the mutants in the test-author's report):
``test_typed_turn_unmapped_with_offset_present`` (6), ``test_reflex_onset_span_and_events_keep_audio_time`` (12)
and ``test_scope_bytes_identical_base_to_tip`` (13). Every other row is red at BASE. The rows were
``pending(task="T-0199")`` until ``/implement T-0199`` started (activated 2026-10-29).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, ``FakeStt(reactive, lag 500 ms)``, 80 ms STT
frames, tools off or live (loop policy), n = 1 deterministic per row. No latency number here is quotable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocol import AstraFrame, AstraHello
from local_astra.contracts.speech import SttWordEvent
from local_astra.contracts.turn import TurnState
from local_astra.perception.browser_source import BrowserFrameSource
from tests import _git_range as gr
from tests.fakes.audio import FakeAudioIO
from tests.fakes.stt import FakeStt
from tests.unit import _tool_turn_support as tts
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import builder_url, image_urls, live_kit
from tests.unit._tool_turn_support import answer
from tests.unit.test_session_listening import DeafAudio, OffsetAudio
from tests.unit.test_session_live import _wire
from tests.unit.test_session_loop import RecordingAudio
from tests.unit.test_t0177_shared_frame import F0, VisionBrowser

REPO = Path(__file__).resolve().parents[2]

ARM_MS = 11_400
ONSET_AUDIO_MS = 21_040
ONSET_SESSION_MS = 32_440  # = ARM_MS + ONSET_AUDIO_MS (capture origin 0)
AHEAD_MS = 5_000_000  # OffsetAudio: audio = session + 5 000 000
LAG_MS = 500
MS_PER_WORD = 200
STT_FRAME_MS = 80
QUESTION = "what is on my screen"
NOTE = "onset.clock"
NOTE_KEYS = {"what", "onset_ms", "at_session_ms", "method"}
#: the phash of a scene change announced by hand (>= 10 bits from every HUD frame's)
FAR_PHASH = "ffffffffffff0000"
W, H = 1280, 720

REGIMES = ("behind", "ahead")
PATHS = ("hud", "shared")


def audio_onset(regime: str, session_ms: int = ONSET_SESSION_MS) -> int:
    """The audio time of a moment at ``session_ms`` (hand formula per regime)."""
    return session_ms - ARM_MS if regime == "behind" else session_ms + AHEAD_MS


# --- frames ---------------------------------------------------------------------------------------------------


def phash_for(i: int) -> str:
    return f"{'0' * 15}{i:x}"


@dataclass(frozen=True)
class Hud:
    """One HUD frame the page posts at session time ``ts``; ``frame`` is the ``Frame`` the source must store."""

    tag: str
    ts: int
    index: int

    @property
    def jpeg(self) -> bytes:
        return b"\xff\xd8" + f"T0199-{self.tag}|".encode() * 8

    @property
    def phash(self) -> str:
        return phash_for(self.index)

    def message(self) -> AstraFrame:
        return AstraFrame(
            source=Source.SCREEN,
            ts_ms=0,
            width=W,
            height=H,
            jpeg_b64=base64.b64encode(self.jpeg).decode(),
            phash=self.phash,
            title="the user's tab",
            app="firefox",
        )

    def frame(self) -> Frame:
        return Frame(
            source=Source.SCREEN,
            ts_ms=self.ts,
            width=W,
            height=H,
            jpeg=self.jpeg,
            phash=self.phash,
            title="the user's tab",
            app="firefox",
        )


def huds(*offsets: int, base: int = ONSET_SESSION_MS) -> dict[int, Hud]:
    """HUD frames at ``base + offset`` session time, keyed by offset."""
    return {off: Hud(tag=f"o{off}", ts=base + off, index=i + 1) for i, off in enumerate(offsets)}


class RecordingSource(BrowserFrameSource):
    """The real source, plus a log of every lookup (delegation only: nothing is faked)."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.calls: list[tuple[Source, int | None, Frame | None]] = []

    async def get_frame(self, source: Source, at_ms: int | None = None) -> Frame | None:
        found = await super().get_frame(source, at_ms)
        self.calls.append((source, at_ms, found))
        return found


# --- audio and STT ---------------------------------------------------------------------------------------------


def behind_audio(clock: Any) -> Any:
    return OriginAudio(clock, capture_origin_ms=0)


class OriginAudio(RecordingAudio):
    """``RecordingAudio`` over ``FakeAudioIO(capture_origin_ms=...)`` (T-0199's fake change)."""

    def __init__(self, clock: Any, *, capture_origin_ms: float) -> None:
        FakeAudioIO.__init__(self, clock, capture_origin_ms=capture_origin_ms)
        self.plays, self.pauses, self.calls, self.fail_play = [], [], [], False


#: criterion 10: the capture clock restarts at session ``RESTART_SESSION_MS``: frames from audio ``RESTART_AUDIO_MS``
#: on are stamped from 0 again (a device reopen), so the audio->session offset moves from 11 400 to 20 200.
RESTART_AUDIO_MS = 8_800
RESTART_SESSION_MS = ARM_MS + RESTART_AUDIO_MS  # 20 200


class RestartingAudio(OriginAudio):
    async def stt_blocks(self) -> Any:
        async for item in super().stt_blocks():
            if item.t_ms >= RESTART_AUDIO_MS:
                item = replace(
                    item, t_ms=item.t_ms - RESTART_AUDIO_MS, end_ms=item.end_ms - RESTART_AUDIO_MS
                )
            yield item


def reactive(start_ms: int | None, *, steps: bool = True) -> FakeStt:
    stt = FakeStt(reactive=True, steps=steps, lag_ms=LAG_MS)
    if start_ms is not None:
        stt.say(QUESTION, start_ms=start_ms, ms_per_word=MS_PER_WORD)
    return stt


class ShiftedWords:
    """Delegating ``SttClient`` whose words report ``start_ms`` shifted by ``shift`` (an audio time the fed audio
    has not reached, or one before capture began): what criterion 10's bounds exist for."""

    def __init__(self, inner: Any, shift: int) -> None:
        self.inner, self.shift = inner, shift

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def feed(self, pcm: bytes, t_ms: int) -> None:
        await self.inner.feed(pcm, t_ms)

    async def mark(self, t_eot_ms: int) -> int:
        return int(await self.inner.mark(t_eot_ms))

    def stats(self) -> Any:
        return self.inner.stats()

    async def close(self) -> None:
        await self.inner.close()

    async def events(self) -> AsyncIterator[Any]:
        async for ev in self.inner.events():
            if isinstance(ev, SttWordEvent):
                ev = ev.model_copy(
                    update={"start_ms": ev.start_ms + self.shift, "end_ms": ev.end_ms + self.shift}
                )
            yield ev


# --- the rig ---------------------------------------------------------------------------------------------------


@dataclass
class Obs:
    rig: Any
    source: RecordingSource
    log_records: list[dict[str, Any]] = field(default_factory=list)

    def notes(self) -> list[dict[str, Any]]:
        return [
            r["data"]
            for r in self.log_records
            if r.get("kind") == "note" and r["data"].get("what") == NOTE
        ]

    def voice_onsets(self) -> list[E.UserSpeechOnset]:
        return list(self.rig.machine.events(E.UserSpeechOnset))

    def lookups(self) -> list[list[tuple[Source, int | None, Frame | None]]]:
        """``get_frame`` calls grouped by snapshot (a snapshot asks one source, or every source under shared)."""
        groups: list[list[tuple[Source, int | None, Frame | None]]] = []
        for call in self.source.calls:
            if groups and call[0] not in {c[0] for c in groups[-1]} and call[1] == groups[-1][0][1]:
                groups[-1].append(call)
            else:
                groups.append([call])
        return groups

    def first_lookup(self) -> tuple[int | None, Frame | None]:
        groups = self.lookups()
        assert groups, "no perception.get_frame call: the onset snapshot never looked a frame up"
        group = groups[0]
        found = [f for _, _, f in group if f is not None]
        return group[0][1], (max(found, key=lambda f: f.ts_ms) if found else None)


@asynccontextmanager
async def onset_rig(
    tmp_path: Path,
    *,
    path: str = "hud",
    audio: Callable[[Any], Any] = behind_audio,
    stt: Any = None,
    frames: dict[int, Hud] | None = None,
    script: Any = None,
    before_start: Callable[[Any, RecordingSource], None] | None = None,
    start_ms: int = ARM_MS,
) -> AsyncIterator[Obs]:
    """``tool_rig`` with the clock at ``start_ms``, ``audio(clock)``, ``stt`` and a recording ``BrowserFrameSource``
    that ingests ``frames`` at their session times. ``path``: ``"hud"`` (tools off, the non-live ``get_frame``) or
    ``"shared"`` (tools live, vision ``shared`` selected over the HUD wire)."""
    real = tts.Session
    holder: dict[str, Any] = {}

    def make_audio(clock: Any) -> Any:
        clock.now_ms = start_ms
        return audio(clock)

    class Wrapped(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            clock = k["clock"]
            source = RecordingSource(now_ms=lambda: clock.now_ms)
            for hud in (frames or {}).values():
                clock.call_at(hud.ts, lambda h=hud: source.ingest(h.message()))
            if before_start is not None:
                before_start(clock, source)
            holder["source"] = source
            a = (*a[:2], stt if stt is not None else a[2], *a[3:5], source, *a[6:])
            super().__init__(*a, **k)

    tts.Session = Wrapped  # type: ignore[misc]
    browser = VisionBrowser([F0])
    obs: Obs | None = None
    try:
        async with tts.tool_rig(
            tmp_path,
            kit=live_kit(browser) if path == "shared" else None,
            policy=tsl.LOOP if path == "shared" else tts.POLICY,
            script=list(script) if script is not None else [answer("It is a page. ")],
            audio_cls=make_audio,
        ) as rig:
            tts.Session = real  # type: ignore[misc]
            source = holder["source"]
            rig.perception = source
            obs = Obs(rig=rig, source=source)
            # the capture pump's first iteration arms the fake's capture clock: let it run before any time passes,
            # so capture is armed at ``start_ms`` (the hand formulas above assume it)
            await rig.pump()
            assert rig.clock.now_ms == start_ms, "premise: capture armed at start_ms"
            if path == "shared":
                from local_astra.conductor.hud_server import HudServer

                browser.now_ms = lambda: rig.clock.now_ms
                server = HudServer(rig.session, frame_sink=source, port=0)
                page = server.open()
                page.receive(encode(AstraHello()))
                page.receive(json.dumps({"type": "astra.vision.select", "source": "shared"}))
                await rig.pump()
                rig.server, rig.page = server, page
            yield obs
    finally:
        tts.Session = real  # type: ignore[misc]
        if obs is not None:
            obs.log_records = list(getattr(obs.rig, "log_records", []))


async def to_onset(obs: Obs, *, preroll_to: int = ONSET_SESSION_MS - 1_200) -> E.UserSpeechOnset:
    """Advance in 400 ms steps to ``preroll_to``, then 40 ms steps until the machine has stepped a voice onset."""
    rig = obs.rig
    left = preroll_to - rig.clock.now_ms
    if left > 0:
        await rig.advance(left - left % 400, step=400)
    await rig.until(lambda: bool(obs.voice_onsets()), "no voice onset reached the machine", step=40)
    return obs.voice_onsets()[0]


async def to_turn_line(obs: Obs) -> None:
    rig = obs.rig
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and bool(rig.records()),
        "the voice turn never wrote its line",
        limit_ms=20_000,
        step=40,
    )


def gen_images(obs: Obs) -> list[str]:
    return image_urls(_wire(tsl.sent(obs.rig, "t1-gen")))


def assert_mapped(at: int | None, expected_session: int, what: str) -> None:
    assert at is not None and expected_session <= at < expected_session + STT_FRAME_MS, (
        f"{what}: the voice onset's frame is looked up at its session time {expected_session} (the mapping errs "
        f"late by at most one STT frame); it was looked up at {at}"
    )


async def voice_turn(
    tmp_path: Path, *, regime: str, path: str, frames: dict[int, Hud]
) -> tuple[Obs, E.UserSpeechOnset]:
    audio = behind_audio if regime == "behind" else OffsetAudio
    async with onset_rig(
        tmp_path, path=path, audio=audio, stt=reactive(audio_onset(regime)), frames=frames
    ) as obs:
        onset = await to_onset(obs)
        await to_turn_line(obs)
    return obs, onset


def assert_onset_frame(obs: Obs, onset: E.UserSpeechOnset, want: Hud | None, what: str) -> None:
    at, got = obs.first_lookup()
    assert_mapped(at, ONSET_SESSION_MS, what)
    assert onset.t_ms > ONSET_SESSION_MS + 200, (
        f"premise: the onset is delivered after the +200 frame (recogniser lag); delivered at {onset.t_ms}"
    )
    got_tag = None if got is None else got.phash
    want_tag = None if want is None else want.phash
    assert got_tag == want_tag, (
        f"{what}: the onset frame is {want and want.tag}; got {got and got.ts_ms}"
    )
    record = obs.rig.record("t1")
    assert record.extras.get("frames") == ([] if want is None else [want.phash]), (
        f"{what}: turn.frames; got {record.extras.get('frames')}"
    )
    images = gen_images(obs)
    assert images == ([] if want is None else [builder_url(want.frame())]), (
        f"{what}: the turn's first request carries the onset frame only; got {len(images)} image(s)"
    )


# =================================================================================================================
# criteria 1, 2: the regression
# =================================================================================================================


async def test_voice_onset_shared_live_gets_newest_frame_before_onset(tmp_path: Path) -> None:
    """Criterion 1 (James's session, live + shared): HUD frames at 1 Hz from 3 s before the onset's session time;
    the user speaks at audio 21 040 = session 32 440. F_k is the frame received at 32 440 (``ts_ms <= 32 440``, by
    hand); not blind; ``turn.frames == [F_k.phash]``. At BASE the lookup is at 21 040: no frame, blind."""
    frames = huds(-3000, -2000, -1000, 0, 1000)
    obs, onset = await voice_turn(tmp_path, regime="behind", path="shared", frames=frames)
    assert_onset_frame(obs, onset, frames[0], "criterion 1")
    assert obs.rig.record("t1").extras.get("frame_source") not in (None, "none"), (
        "criterion 1: not blind"
    )


async def test_voice_onset_hud_path_gets_newest_frame_before_onset(tmp_path: Path) -> None:
    """Criterion 2 (tools off, the non-live ``perception.get_frame``): same shape, the prompt's image is F_k."""
    frames = huds(-3000, -2000, -1000, 0, 1000)
    obs, onset = await voice_turn(tmp_path, regime="behind", path="hud", frames=frames)
    assert_onset_frame(obs, onset, frames[0], "criterion 2")


# =================================================================================================================
# criteria 3, 4, 5: never later, never earlier, both offset directions
# =================================================================================================================


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("regime", REGIMES)
async def test_frame_received_after_onset_never_used(
    tmp_path: Path, regime: str, path: str
) -> None:
    """Criterion 3 (adversarial), and 5 under ``ahead``: frames at onset -800 and +200 session ms; the +200 one is
    received ~600 ms before the onset is delivered (lag 500 ms). The -800 frame is the onset frame. Kills "use the
    delivery stamp ``t_ms``" (it picks +200) and, under ``ahead``, raw ``onset_ms`` (5 032 440 picks +200)."""
    frames = huds(-800, 200)
    obs, onset = await voice_turn(tmp_path, regime=regime, path=path, frames=frames)
    assert_onset_frame(obs, onset, frames[-800], f"criterion 3 ({regime}, {path})")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("regime", REGIMES)
async def test_frame_just_before_onset_is_used_not_an_older_one(
    tmp_path: Path, regime: str, path: str
) -> None:
    """Criterion 4 (adversarial), and 5 under ``ahead``: frames at onset -12 440 (session 20 000, at or before raw
    audio 21 040 under ``behind``), -2000 and -200. The -200 frame is the onset frame. Kills raw ``onset_ms`` under
    ``behind`` (it picks the session-20 000 frame) and a mapping that errs early by more than 200 ms."""
    frames = huds(-12_440, -2000, -200)
    obs, onset = await voice_turn(tmp_path, regime=regime, path=path, frames=frames)
    assert_onset_frame(obs, onset, frames[-200], f"criterion 4 ({regime}, {path})")


@pytest.mark.parametrize("path", PATHS)
async def test_audio_ahead_of_session_maps_too(tmp_path: Path, path: str) -> None:
    """Criterion 5 (criterion 1's shape with audio 5 000 000 ms ahead, ``OffsetAudio``): F_k is the frame received
    at the onset's session time. At BASE raw ``onset_ms`` (5 032 440) picks the newest frame received, +600."""
    frames = huds(-3000, -2000, -1000, 0, 600)
    obs, onset = await voice_turn(tmp_path, regime="ahead", path=path, frames=frames)
    assert onset.onset_ms == ONSET_SESSION_MS + AHEAD_MS, f"premise: audio onset; {onset}"
    assert_onset_frame(obs, onset, frames[0], f"criterion 5 ({path})")


# =================================================================================================================
# criteria 6, 7: typed turns and other snapshots are not mapped
# =================================================================================================================


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("regime", REGIMES)
async def test_typed_turn_unmapped_with_offset_present(
    tmp_path: Path, regime: str, path: str
) -> None:
    """Criterion 6 (negative, BASE-green guard): capture runs with the offset (a mapping exists), the user types at
    session S = 32 440, the message is handled at S + 150, frames at S - 300 and S + 100. The lookup is at S exactly,
    the frame is S - 300, and no ``onset.clock`` note is written. Kills "map every onset" under ``ahead`` (S maps to
    a negative time: blind) and any mapping without the delivery clamp under ``behind`` (S + 11 400: the +100 frame).
    """
    s = ONSET_SESSION_MS
    frames = huds(-300, 100)
    audio = behind_audio if regime == "behind" else OffsetAudio
    async with onset_rig(
        tmp_path, path=path, audio=audio, stt=reactive(None), frames=frames
    ) as obs:
        rig = obs.rig
        await rig.advance(s - 1_200 - rig.clock.now_ms, step=400)
        await rig.until(lambda: rig.clock.now_ms >= s + 150, "clock", step=50)
        rig.submit(E.UserText(t_ms=s, text=QUESTION))
        await to_turn_line(obs)
    at, got = obs.first_lookup()
    assert at == s, (
        f"criterion 6: a typed onset is looked up at the message's own session time {s}; got {at}"
    )
    assert got is not None and got.phash == frames[-300].phash, (
        f"criterion 6: the S-300 frame; got {got}"
    )
    assert obs.notes() == [], f"criterion 11: typed onsets write no {NOTE} note; got {obs.notes()}"


@pytest.mark.parametrize("regime", REGIMES)
async def test_scene_change_and_relisten_snapshots_unmapped(tmp_path: Path, regime: str) -> None:
    """Criterion 7 (adversarial) inside a voice turn whose onset *is* mapped (so keying on the turn's origin fails):

    - scene change: once the onset is stepped, ``SceneChanged(t_ms=now - 300)`` is submitted (an event handled late).
      Its lookup is at exactly ``now - 300``; frames at onset +300 and +600 make "map every ``SnapshotFrame``" visible
      under ``behind`` (clamped to now: the +600 frame) and under ``ahead`` (a negative time: nothing);
    - relisten: the reply is barged into by a session-clock onset (not a voice onset), the takeover interrupts, and
      ``INTERRUPTED -> LISTENING`` snapshots at the ``PlaybackStopped`` stamp: looked up at exactly that time.
    - exactly one ``onset.clock`` note (the voice onset's) in the whole run (criterion 11)."""
    frames = huds(-200, 300, 600)
    audio = behind_audio if regime == "behind" else OffsetAudio
    script = [answer("It is a long page with many words on it and more words after those. ")]
    async with onset_rig(
        tmp_path,
        audio=audio,
        stt=reactive(audio_onset(regime)),
        frames=frames,
        script=script,
    ) as obs:
        rig = obs.rig
        await to_onset(obs)
        await rig.until(lambda: rig.clock.now_ms >= ONSET_SESSION_MS + 900, "clock", step=20)
        # between the +300 and +600 frames, and at least 300 ms in the past when it is handled
        scene_at = ONSET_SESSION_MS + 450
        assert rig.clock.now_ms >= scene_at + 300, (
            f"premise: the scene event is handled late; now {rig.clock.now_ms}"
        )
        rig.submit(E.SceneChanged(t_ms=scene_at, source="screen", phash=FAR_PHASH))
        await rig.pump()
        await rig.until(lambda: rig.state.state is TurnState.SPEAKING, "no SPEAKING", step=40)
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: bool(rig.machine.events(E.PlaybackStopped)) and len(obs.lookups()) >= 3,
            "the barge-in never reached the relisten snapshot",
            step=40,
        )
    groups = obs.lookups()
    assert_mapped(groups[0][0][1], ONSET_SESSION_MS, "premise (the voice onset is mapped)")
    scene_group = groups[1]
    assert scene_group[0][1] == scene_at, (
        f"criterion 7: the scene-change snapshot is looked up at the machine's at_ms {scene_at}; got {scene_group[0][1]}"
    )
    assert scene_group[0][2] is not None and scene_group[0][2].phash == frames[300].phash, (
        f"criterion 7: the scene lookup returns the onset+300 frame; got {scene_group[0][2]}"
    )
    stop = rig.machine.events(E.PlaybackStopped)[-1]
    assert groups[2][0][1] == stop.t_ms, (
        f"criterion 7: the relisten snapshot is looked up at the PlaybackStopped stamp {stop.t_ms}; got {groups[2][0][1]}"
    )
    assert len(obs.notes()) == 1, f"criterion 11: one note, the voice onset's; got {obs.notes()}"


# =================================================================================================================
# criterion 8: the reset floor
# =================================================================================================================


def reset_at(ts: int) -> Callable[[Any, RecordingSource], None]:
    def install(clock: Any, source: RecordingSource) -> None:
        holder: dict[str, Any] = {}

        def fire() -> None:
            holder["session"].submit(E.SessionReset(t_ms=clock.now_ms))

        install.holder = holder  # type: ignore[attr-defined]
        clock.call_at(ts, fire)

    return install


async def floor_turn(tmp_path: Path, regime: str, frames: dict[int, Hud], reset_ms: int) -> Obs:
    audio = behind_audio if regime == "behind" else OffsetAudio
    hook = reset_at(reset_ms)
    async with onset_rig(
        tmp_path, audio=audio, stt=reactive(audio_onset(regime)), frames=frames, before_start=hook
    ) as obs:
        hook.holder["session"] = obs.rig.session  # type: ignore[attr-defined]
        await to_onset(obs)
        await to_turn_line(obs)
    return obs


@pytest.mark.parametrize("regime", REGIMES)
async def test_reset_floor_excludes_pre_reset_frame(tmp_path: Path, regime: str) -> None:
    """Criterion 8: the only frame before the onset is received at onset -600 and the session is reset at that same
    time (the frame is at or before the floor); a frame at +200 arrives before delivery. The onset frame is none:
    a blind turn, looked up at the mapped time. Kills a mapped lookup that skips the floor (the -600 frame), the
    delivery stamp and, under ``ahead``, raw ``onset_ms`` (both the +200 frame). A reset keeps the mapping
    (``RESET_KEEPS``): a cleared one falls back to the delivery stamp and picks +200."""
    frames = huds(-600, 200)
    obs = await floor_turn(tmp_path, regime, frames, ONSET_SESSION_MS - 600)
    at, _ = obs.first_lookup()
    assert_mapped(at, ONSET_SESSION_MS, "criterion 8")
    record = obs.rig.record("t1")
    assert record.extras.get("frames") == [], (
        f"criterion 8: blind turn; frames {record.extras.get('frames')}"
    )
    assert gen_images(obs) == [], "criterion 8: no image in the request"
    notes = obs.notes()
    assert len(notes) == 1 and notes[0].get("method") == "capture", f"criterion 11: {notes}"


@pytest.mark.parametrize("regime", REGIMES)
async def test_mapped_onset_before_floor_is_blind(tmp_path: Path, regime: str) -> None:
    """Criterion 8: the user starts speaking at 32 440, the session is reset at onset +100 (before the onset is
    delivered), frames at -200 (pre-floor) and +300 (post-floor, pre-delivery). The mapped onset is before the floor:
    blind. The delivery stamp (and raw ``onset_ms`` under ``ahead``) would show the +300 frame."""
    frames = huds(-200, 300)
    obs = await floor_turn(tmp_path, regime, frames, ONSET_SESSION_MS + 100)
    at, _ = obs.first_lookup()
    assert_mapped(at, ONSET_SESSION_MS, "criterion 8")
    record = obs.rig.record("t1")
    assert record.extras.get("frames") == [], (
        f"criterion 8: blind turn; frames {record.extras.get('frames')}"
    )
    notes = obs.notes()
    assert len(notes) == 1 and notes[0].get("method") == "capture", f"criterion 11: {notes}"


# =================================================================================================================
# criteria 9, 10: no mapping yet; bounds; restart
# =================================================================================================================


async def test_no_capture_seam_falls_back_to_delivery_stamp(tmp_path: Path) -> None:
    """Criterion 9: an audio object with no capture seam (``DeafAudio``), so no capture frame is ever handled; a
    script-mode recogniser delivers "what" (audio 21 040-21 240) as soon as the session starts at 33 240. Frames
    were received at 20 000 (at or before the raw audio onset), 32 640 and 33 040. The lookup is at the delivery
    stamp 33 240: the 33 040 frame, never the 20 000 one; one note ``method: "delivery"``."""
    start = 33_240
    frames = {0: Hud("raw", 20_000, 1), 1: Hud("mid", 32_640, 2), 2: Hud("late", 33_040, 3)}

    def ingest_before(clock: Any, source: RecordingSource) -> None:
        for hud in frames.values():
            clock.now_ms = hud.ts
            source.ingest(hud.message())
        clock.now_ms = start

    stt = FakeStt(
        script=[SttWordEvent(text="what", start_ms=ONSET_AUDIO_MS, end_ms=ONSET_AUDIO_MS + 200)]
    )
    async with onset_rig(
        tmp_path,
        audio=lambda clock: DeafAudio(RecordingAudio(clock)),
        stt=stt,
        before_start=ingest_before,
        start_ms=start,
    ) as obs:
        await obs.rig.until(lambda: bool(obs.voice_onsets()), "no voice onset", step=40)
        await obs.rig.pump()
        onset = obs.voice_onsets()[0]
    assert onset.onset_ms == ONSET_AUDIO_MS, f"premise: {onset}"
    at, got = obs.first_lookup()
    assert at == onset.t_ms, (
        f"criterion 9: no mapping: looked up at the delivery stamp {onset.t_ms}; got {at}"
    )
    assert got is not None and got.phash == frames[2].phash, (
        f"criterion 9: the 33 040 frame; got {got}"
    )
    notes = obs.notes()
    assert len(notes) == 1, f"criterion 9/11: one {NOTE} note; got {notes}"
    assert set(notes[0]) == NOTE_KEYS and notes[0]["method"] == "delivery", notes[0]
    assert (notes[0]["onset_ms"], notes[0]["at_session_ms"]) == (ONSET_AUDIO_MS, onset.t_ms), notes[
        0
    ]


@pytest.mark.parametrize(
    ("regime", "shift"),
    [("behind", 3_000), ("ahead", -(ONSET_SESSION_MS + AHEAD_MS - 1_000))],
    ids=["after-delivery", "before-zero"],
)
async def test_mapped_onset_never_after_delivery(tmp_path: Path, regime: str, shift: int) -> None:
    """Criterion 10 (bounds): the recogniser reports the first word's start ``shift`` ms off the audio it heard.
    ``after-delivery``: 24 040 maps to session 35 440, after the delivery stamp (~33 240): the lookup and the note
    are at most the delivery stamp. ``before-zero``: audio 1 000 under ``ahead`` maps to about -4 999 000: the
    lookup and the note are at least 0. Frames at onset -200 and +200 (both received before delivery)."""
    frames = huds(-200, 200)
    audio = behind_audio if regime == "behind" else OffsetAudio
    stt = ShiftedWords(reactive(audio_onset(regime)), shift)
    async with onset_rig(tmp_path, audio=audio, stt=stt, frames=frames) as obs:
        onset = await to_onset(obs)
        await obs.rig.pump()
    at, _ = obs.first_lookup()
    notes = obs.notes()
    assert len(notes) == 1 and notes[0].get("method") == "capture", f"criterion 11: {notes}"
    mapped = notes[0]["at_session_ms"]
    assert at == mapped, (
        f"criterion 11: the note states the lookup time; note {mapped}, lookup {at}"
    )
    assert 0 <= mapped <= onset.t_ms, (
        f"criterion 10: the mapped onset is within [0, delivery {onset.t_ms}]; got {mapped}"
    )
    if shift > 0:
        assert mapped == onset.t_ms, (
            f"criterion 10: clamped to the delivery stamp {onset.t_ms}; got {mapped}"
        )
    else:
        assert mapped == 0, f"criterion 10: clamped to 0; got {mapped}"


async def test_capture_clock_restart_followed(tmp_path: Path) -> None:
    """Criterion 10 (restart): capture armed at 11 400 with origin 0; at session 20 200 the capture clock restarts
    from 0 (offset 11 400 -> 20 200). The user speaks at audio 400 after the restart = session 20 600, delivered
    ~800 ms later. Frames at 19 000, 20 400 (-200) and 20 800 (+200). The onset frame is 20 400. A mapping that
    keeps the pre-restart offset looks up at 11 800 (blind); the delivery stamp picks 20 800."""
    onset_session = RESTART_SESSION_MS + 400
    frames = huds(-1600, -200, 200, base=onset_session)
    stt = reactive(None, steps=False)

    def say_after_restart(clock: Any, source: RecordingSource) -> None:
        clock.call_at(
            RESTART_SESSION_MS + 100,
            lambda: stt.say(QUESTION, start_ms=400, ms_per_word=MS_PER_WORD),
        )

    async with onset_rig(
        tmp_path,
        audio=lambda clock: RestartingAudio(clock, capture_origin_ms=0),
        stt=stt,
        frames=frames,
        before_start=say_after_restart,
    ) as obs:
        onset = await to_onset(obs, preroll_to=RESTART_SESSION_MS - 1_000)
        await obs.rig.pump()
    assert onset.onset_ms == 400, f"premise: the onset is the post-restart word; {onset}"
    at, got = obs.first_lookup()
    assert_mapped(at, onset_session, "criterion 10 (restart)")
    assert got is not None and got.phash == frames[-200].phash, (
        f"criterion 10: the 20 400 frame; got {got}"
    )


# =================================================================================================================
# criteria 11, 12
# =================================================================================================================


async def test_onset_clock_note_once_per_voice_onset(tmp_path: Path) -> None:
    """Criterion 11: one voice turn then one typed turn: exactly one ``onset.clock`` note, with exactly the four
    keys, ``onset_ms`` the audio onset (21 040), ``method: "capture"`` and ``at_session_ms`` the mapped time."""
    frames = huds(-200)
    script = [answer("It is a page. "), answer("Still a page. ")]
    async with onset_rig(
        tmp_path, stt=reactive(ONSET_AUDIO_MS), frames=frames, script=script
    ) as obs:
        await to_onset(obs)
        await to_turn_line(obs)
        rig = obs.rig
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="and now?"))
        await rig.until(lambda: len(rig.records()) >= 2, "the typed turn never ended", step=40)
    notes = obs.notes()
    assert len(notes) == 1, f"criterion 11: one note per voice onset, none for typed; got {notes}"
    note = notes[0]
    assert set(note) == NOTE_KEYS, f"criterion 11: the note's keys; got {sorted(note)}"
    assert (note["onset_ms"], note["method"]) == (ONSET_AUDIO_MS, "capture"), note
    assert_mapped(note["at_session_ms"], ONSET_SESSION_MS, "criterion 11 (at_session_ms)")


async def test_reflex_onset_span_and_events_keep_audio_time(tmp_path: Path) -> None:
    """Criterion 12 (negative, BASE-green guard): the ``reflex.onset`` span, the ``stt.decision`` row, the machine's
    ``UserSpeechOnset.onset_ms`` and ``MachineState.onset_ms`` stay on the audio clock (21 040), and
    ``assists.texts_for`` gets the machine's ``at_ms`` as at BASE (criterion 13, H-T199-3). The T-0181 mic fence's
    audio ``onset_ms`` is pinned by the T-0181 rows, unchanged."""
    frames = huds(-200)
    seen: list[Any] = []
    states: list[int | None] = []
    async with onset_rig(tmp_path, stt=reactive(ONSET_AUDIO_MS), frames=frames) as obs:
        rig = obs.rig
        texts_for = rig.session.assists.texts_for

        def spy(at_ms: Any) -> Any:
            seen.append(at_ms)
            return texts_for(at_ms)

        rig.session.assists.texts_for = spy
        await to_onset(obs)
        states.append(rig.state.onset_ms)
        await to_turn_line(obs)
    onset = obs.voice_onsets()[0]
    assert onset.onset_ms == ONSET_AUDIO_MS, onset
    assert states == [ONSET_AUDIO_MS], f"MachineState.onset_ms is audio time; got {states}"
    span = obs.rig.record("t1").spans.get("reflex.onset")
    assert span is not None and span[0] == ONSET_AUDIO_MS, (
        f"criterion 12: reflex.onset span; got {span}"
    )
    decisions = [r["data"] for r in obs.log_records if r.get("kind") == "stt.decision"]
    assert decisions and decisions[0]["onset_ms"] == ONSET_AUDIO_MS, decisions[:1]
    assert seen == [ONSET_AUDIO_MS], f"criterion 13: texts_for gets the BASE argument; got {seen}"


# =================================================================================================================
# criterion 13: scope
# =================================================================================================================

#: The commit T-0199 was planned at (docs/tasks/T-0199.md "Status"); a fixed literal, never a moving ref.
BASE = "333159f3"
#: T-0199's implementation tip: the second parent of its merge ``MERGE`` (closed at merge; re-pinned under T-0202 A,
#: whose A5 legitimately changes ``hud/src/tools/model.js`` after T-0199). A fixed literal, never a moving ref.
TIP = "bfd5bb8a"
#: The merge of T-0199 into main (``901024da^2 == bfd5bb8a``).
MERGE = "901024da"
#: sha256 of every file under the pinned paths at BASE (``git show 333159f3:<path> | sha256sum``, 2026-10-29).
#: Closed range (T-0202 A re-pin, docs/tasks/T-0202.md): the row compares these with the blobs at BASE and at TIP,
#: never with the working tree, so later tasks may edit these paths.
PINNED_SHA256: dict[str, str] = {
    "hud/README.md": "484a5717b9c9786698e59daa003edf9814fcd4007cf8d152b5149add7378062b",
    "hud/index.html": "4f63f5cb52296a4b491dca7f6538a91ff5bed184289b5ab0556dcbc23943cf24",
    "hud/package.json": "48e847860dd4f0907ce91e346dbb00a4a12213163ad3d80a953c343eb9a11979",
    "hud/src/ask/form.js": "15af9770b2dbe9f24c174e06e92d373ff592836280d49c354798c4a1997f2f07",
    "hud/src/ask/index.js": "165dac78569d6957b536529b9a8fcbe65cc75ee3d791968230dc214e2730bcdd",
    "hud/src/assists/extract.js": "44ddf19055f47a1440161e6d2ecb80089ac8765f81a4ef632734c4d1487496ba",
    "hud/src/assists/gate.js": "3c6031f17a480efbeb6ba704a0ceae4808a0e1608235db24e2d23005ce0e36d8",
    "hud/src/assists/index.js": "d450a6e5059ec07aa3586913c1000b0a643da83851836871696407c7763575c0",
    "hud/src/capture.js": "7f56b6171d6f0c435d16785011f2b05b993125fb6a4b7b2830387f7e35a95737",
    "hud/src/denoise/index.js": "4de546499bad2a83acbe4b17f301a4517b078b2791119f65e51e971ff924b0ff",
    "hud/src/denoise/model.js": "eda0a8e652370095afbb72cdd5e76163d433441c6d3a900b0d9ae869e9d8e76e",
    "hud/src/frames.js": "367a41598a3f9e82d55878895615fd6042adc4d7f5bd94f96d86b21b83e10f59",
    "hud/src/log/gate.js": "15910948112fac359fe575a3e70e96d9500eadc39cf1e65581b262fdbb8ffe3d",
    "hud/src/log/index.js": "f75e84d608511ccb76b9876a63c468de24a3c0bf772f6c2346c15ff3101ce499",
    "hud/src/main.js": "ceb8d93f9caabb0c7af75f78ad0239b9e8454a27edb524130d72a864b5252c31",
    "hud/src/mic.js": "bc6f9b485a0dfdd91f8df7ef768e044edfc0d2c6ee81aaf87fb63c0c95f4ac8f",
    "hud/src/panels.js": "80f19e509500dab19a4c817e8b5b37005c7c8278431d3535472bd98ef3c19b16",
    "hud/src/panels/spoken.js": "aee56e61bb81fe1c33aaf892a6d1dfd359b0c31cb01ece729a3d408571c434b2",
    "hud/src/panels/thinking.js": "2060321300e15972b037da425296aa60285d48777d33f496aba4e7cb7e1810cf",
    "hud/src/panels/thoughts.js": "cef9c14f6a5a3437d88050c9f84f193b39c8bddde74187dc51d553e7a6b919cd",
    "hud/src/panels/thread.js": "27c67a533a4de04ff9bd9df3417a1e794d13195095a9265f6ec3eaec6b0a69ba",
    "hud/src/panels/tool_call.js": "6e68626d82376b0537d14672057bfc93da97e097b210a7ebce538c5d7ae745f8",
    "hud/src/panels/transcript.js": "882ab488e5ee848647a9d0ca5ec4b46687e7598c1033668faa8bc49f8bd1e1ec",
    "hud/src/phash.js": "67e2d3034fea24c549c873b332203fc009b7cb030a1247d65842d62a0c584de8",
    "hud/src/protocol.js": "956f1d2683fdcdb1bbc0b25255ada5d4360eba27ede05cba73fa46d643d1c961",
    "hud/src/session/gate.js": "4058ef121bb6645857e0469c19da8e0b131e335e3c296a1faa3dc243c099ca6f",
    "hud/src/session/index.js": "0eb6a79b8ff27ff177b6b2640364934a93cb3ef4bb49a8c61684e6fdb6eedd66",
    "hud/src/state.js": "bf50972a0772e438c4080f544fa828655df3da8d4e5dbbc3dbeae9a0a3d5a691",
    "hud/src/stop.js": "20ba0b7f81d81159917aa39392b7867be66890ecc8fd3980c073f85ef5f20a7d",
    "hud/src/theme.js": "08db5dbc799f2bf461473e6508e6070e27993aacecb7c31bfde427d7ab4c2d32",
    "hud/src/tools/index.js": "b6a948cd6908859e5fa92975f1bee71eee242111bdf230fadb73d13121ea5e08",
    "hud/src/tools/model.js": "e5e0075b715a7e46e9427282c39d36bcfe548a2e82e673e592fb641dc1333c7c",
    "hud/src/ui/a2ui.js": "214e718b750e52188d60bdea9c60bbe2de6bf3f183c0876b90dad804322c2624",
    "hud/src/ui/action.js": "35ed3bee351f7482f839c79d3413e9ef4c7ab19a4b96fe6c38bc967b0e5ba749",
    "hud/src/ui/bridge.js": "f83124b3abaf49021ac5a7fd17445948a29b0d5d0f5044c0175ced1a67093755",
    "hud/src/ui/card.css": "fdcec3372cc95d2ffc4447644c7d1c2d9f97c1c56ea0de9a0ae2bd821e802e44",
    "hud/src/ui/cards_v2.js": "20359c43aecd7f03de46b43567c2ade5c9da1bdf1ebd38ba496b016926914116",
    "hud/src/ui/engine.js": "3440fbb903f52bf59cf98f17d1db358736c8fae64074726825525511e705753e",
    "hud/src/ui/fence.js": "f120c0351606ee4c955eba3e7691951d88817af84de738587f9e6f5bf6a06277",
    "hud/src/ui/index.js": "dbfbe3f8158f4d3005912c45f7219c45b583f3a3ec54a5d51faa8575cac2f3c2",
    "hud/src/ui/render.js": "4db6117cbdbfa22fdaca2b46f81286e1ffbefb911e10fbb2c7acefb637b34f4a",
    "hud/src/vision.js": "6a852fb0b32910002fedf4ce61f2effecb11546088814c7bec01247459e57b68",
    "hud/src/ws.js": "30bd8a44dbf7b11333da42d001d6d43695e807ca3449217f62137530c39aa83a",
    "hud/styles.css": "83ddea0bddb36eeee94e42085166e94bb3c397b495067c531e4600aede518aef",
    "local_astra/conductor/audio_capture.py": "62c9f4948fbaeee34b4473d91ece39d94ddd7b931c3c0d79220c06e1924e4330",
    "local_astra/conductor/audio_io.py": "d7958805a3ae0a8e063a844ef485b1a3ba9cd9569aa2a6420a4b2aedc4977718",
    "local_astra/conductor/stt_client.py": "8c4bffa9e32135a7f887b3d992697a987b064b18751a784f2e1247a794cf2441",
    "local_astra/conductor/turn_machine.py": "9944ee489b89bcb15720c5d130c735bfb1168145f5b730dc72b2a9433b41a266",
    "local_astra/contracts/__init__.py": "56f54b85df9d9c51f44f88c29ed383fb783c2c552f80d20c18463686c4ad8c34",
    "local_astra/contracts/base.py": "8a71b74d453c882d56e68d2118212778a0647725b0f97f0f3c1ab41d21cc7b0f",
    "local_astra/contracts/cd.py": "79e59ab4670011dea0d42724971074bfa2d0f00a29755d3c616e3e1a78a22b91",
    "local_astra/contracts/codec.py": "ec10fa7c84ec4845197321c37111b91a65ec263cc2fa2228baf9ff6fb3f92338",
    "local_astra/contracts/commands.py": "307a016fa2e3b10fe47fa0d41e6e3b24cf93e1d0ca34f1da47a5a86e2dd2dfaf",
    "local_astra/contracts/events.py": "1812fb3f191bff27d1461f92f7ecf294a1d7718a41003ccc9b333d128d789c34",
    "local_astra/contracts/llm.py": "80855e4a51637c0525bb4f8c39149ff80f46c5af8ddf5fe7929865d4c7d4fd5a",
    "local_astra/contracts/memory.py": "25e07d187b1d1b7ec44ebafc02734d7e6fd6dc1cc0b5b2904f9e0e519a9141ff",
    "local_astra/contracts/perception.py": "f053a559001c9cf265dc13fdc6883a8c15f8f2d5fe6cc44ff020dc8cb7913cab",
    "local_astra/contracts/protocol.py": "e51352a1bbf71210ce9e43f441a1416dc40b2a48ae1744902542fc3a65027724",
    "local_astra/contracts/protocols.py": "e43e12b6188f50f65cd95bdf2fa9f3eb1b0fcbef0fb8cd8aa747fc4a8a111343",
    "local_astra/contracts/speech.py": "da61f85786a30bf1733c8fe3015684daf0961cc40e53f69bf2abb75681565591",
    "local_astra/contracts/toolcall.py": "b9546244cc41bb4c731264090839d6ca2a0aef3fb473cf723e7c82f8266b3f47",
    "local_astra/contracts/turn.py": "6b51c0869077cc79edc7e8471b20702373b786f20cecf1124fe22a76b8e7ecd2",
    "local_astra/perception/assists.py": "bb0775457f807b827ef2c555f30de6c9e577a3dff6fb9ee835cb1fb2acef12b1",
    "local_astra/perception/browser_source.py": "01a98b55ec62f6f79bfc42546ab5235926de6894aeed7260dd4a83dd23225663",
}


def _blob_sha256(rev: str, path: str) -> str | None:
    """sha256 of ``path``'s bytes at commit ``rev`` (``git show``), or ``None`` if the path is absent there."""
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True, check=False
    )
    return hashlib.sha256(proc.stdout).hexdigest() if proc.returncode == 0 else None


def test_scope_bytes_identical_base_to_tip() -> None:
    """Criterion 13 (closed range BASE..TIP, re-pinned under T-0202 A): the machine, contracts, the frame source,
    assists, the STT client, the capture and audio modules and the HUD are byte-identical between T-0199's BASE
    and its implementation tip TIP (the second parent of MERGE). Each pin is first proven to be BASE's content, then
    TIP's blob must hash to it. Skips, naming the sha, when a commit is absent; never compares the working tree."""
    assert PINNED_SHA256, "the pin table is empty"
    gr.require_commits(BASE, TIP, MERGE)
    assert gr.is_ancestor(BASE, TIP), f"{BASE} is not an ancestor of {TIP}"
    parents = gr._git("rev-list", "--parents", "-n", "1", MERGE).stdout.split()[1:]
    tip_full = gr._git("rev-parse", "--verify", "--quiet", f"{TIP}^{{commit}}").stdout.strip()
    assert len(parents) == 2 and parents[1] == tip_full, (
        f"{TIP} is not the second parent of {MERGE}: {parents}"
    )
    not_base = [p for p, d in PINNED_SHA256.items() if _blob_sha256(BASE, p) != d]
    assert not_base == [], f"criterion 13: pins that are not {BASE}'s content: {not_base}"
    changed = [p for p, d in PINNED_SHA256.items() if _blob_sha256(TIP, p) != d]
    assert changed == [], f"criterion 13: files changed in {BASE}..{TIP}: {changed}"
