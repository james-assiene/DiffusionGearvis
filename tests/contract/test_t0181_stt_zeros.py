"""T-0181 C1 (and the fake's ``hears_audio`` contract): after an utterance, exact zeros produce no word, and a marker
issued during them is echoed (docs/tasks/T-0181.md C1, Regime (C), reading A-R6).

The mute (A5) feeds ``stt.feed`` all-zero frames in place of the captured ones. Its claim that "Gearvis hears
nothing" rests on the recogniser producing no ``Word`` from exact zeros, and its liveness (A5's "the drop-frames
mutation hangs in ENDPOINTING") on a marker still being echoed while only zeros arrive. The fast suite relies on
``FakeStt(reactive=True, hears_audio=True)`` for both; this module is the contract that the real moshi-server behaves
the same (``.claude/rules/tests.md``: a behaviour added to a fake needs a real-service test).

**The drive, identical for both backends:** ``tests/fixtures/audio/pocket_tts_sentence.wav`` (A-R6: 4.040 s, 24 kHz
s16le, "One, two, three, four, five.", standing in for Regime C's 2 s utterance) in 1920-sample (80 ms) f32le frames,
``t_ms`` from 0 on the fed-audio clock; then exact zeros (``+0.0`` in every sample) in 80 ms frames from
``ZEROS_START`` = 4040 ms: ``mark(T_EOT)`` (``T_EOT`` = 5080 ms, 13 zero frames in) is issued about 1 s into them, just
before the zero frame stamped ``T_EOT`` is fed; at least 3 s of zeros are fed in all
(38 frames, 3040 ms), and zeros keep being fed until the echo arrives or ``FLUSH_DEADLINE_MS`` of zeros have been fed
(the live server echoes a marker only while audio keeps arriving, T-0007a census A14).

**Assertions, identical for both backends:**

- (anti-vacuity) the utterance was heard: at least one word starting before ``ZEROS_START``;
- no ``Word`` whose ``start_ms`` is at or after ``ZEROS_START`` (C1: "no ``Word`` that starts after the zeros begin");
- exactly one ``SttFlushed`` with ``t_eot_ms == T_EOT``.

``fake``: the reactive fake with ``hears_audio=True``, scripted with the fixture's own word timings
(``pocket_tts_sentence.words.json``) and two *ghost* words whose audio falls inside the zeros (so a content-blind fake
would emit them); the fake must hear all five words, emit neither ghost, and count both in ``unheard``. The fake runs
on the fed-audio clock with no wall-clock pacing.

``real`` (``gpu``; C1, run by group C via ``/launch`` after Merge T-0181-a): ``MoshiSttClient`` against the pinned
moshi-server, fed at real-time pace, **n = 5** sequential runs on one boot, unseeded (moshi-server is deterministic for
a given input only if its sampling is greedy; the runs say whether it was), and **all 5 must pass**; the failure
message lists every run. Skips (never passes) unless the speech services answer their health checks.

Not ``pending``: neither arm depends on group A's session code (the fake's ``hears_audio`` is already in
``tests/fakes/stt.py``; the real arm tests the service), so a pending marker would turn the fake arm into an XPASS.

Regime (fake): in-process, fed-audio clock from 0, n = 1, deterministic. Regime (real): profile ``lean``, the pinned
moshi-server and ``stt-1b-en_fr`` (``services/versions.lock``), real-time pace, n = 5, a clean tree, HEAD recorded by the
launcher.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.speech import TTS_OUTPUT_FORMAT
from tests.contract.test_speech_clients_contract import (
    BLOCK_SAMPLES,
    _fixture_blocks,
    _skip_unless_services_are_up,
)
from tests.fakes import FakeStt
from tests.fakes.stt import word_event

pytestmark = pytest.mark.contract

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
WORDS = FIXTURES / "pocket_tts_sentence.words.json"
#: the fixture's duration (MANIFEST: 4.040 s), where the zeros begin on the fed-audio clock
ZEROS_START = 4040
FRAME_MS = 80
#: C1: "3 s of zeros in 80 ms frames" -- 38 frames, 3040 ms, the first whole-frame count covering 3 s
ZERO_FRAMES = 38
#: the marker is asked for about 1 s into the zeros, on the 80 ms frame grid (13 zero frames in: 5080 ms), just
#: before the zero frame stamped with it is fed
T_EOT = ZEROS_START + 13 * FRAME_MS
#: how long zeros keep coming while the echo is awaited (fed audio, from ZEROS_START)
FLUSH_DEADLINE_MS = 15_000
#: real backend: how many sequential runs, all of which must pass (C1)
REAL_RUNS = 5
#: fake backend: ghost words scripted inside the zeros (start, end) -- a content-blind recogniser would emit them
GHOSTS = (
    ("ghost", ZEROS_START + 400, ZEROS_START + 700),
    ("phantom", ZEROS_START + 1600, ZEROS_START + 1900),
)


@dataclass
class Run:
    events: list[Any] = field(default_factory=list)
    fed_zero_ms: int = 0

    def words(self) -> list[Any]:
        return [e for e in self.events if e.type == "word"]

    def flushed(self) -> list[Any]:
        return [e for e in self.events if e.type == "flushed"]

    def summary(self) -> str:
        return json.dumps(
            {
                "words": [(w.text, w.start_ms, w.end_ms) for w in self.words()],
                "flushed": [f.t_eot_ms for f in self.flushed()],
                "zeros_fed_ms": self.fed_zero_ms,
            }
        )


def fixture_frames() -> list[tuple[bytes, int]]:
    """The fixture as the STT tap hands it over: f32le frames of 1920 samples (the last one shorter), with ``t_ms``."""
    out: list[tuple[bytes, int]] = []
    t_ms = 0
    for block in _fixture_blocks():
        out.append((block, t_ms))
        t_ms += len(block) // 4 * 1000 // TTS_OUTPUT_FORMAT.sample_rate
    assert t_ms == ZEROS_START, f"premise: the fixture is 4040 ms (MANIFEST); fed {t_ms} ms"
    return out


ZERO = bytes(BLOCK_SAMPLES * 4)  # little-endian f32: four zero bytes are exactly +0.0


async def drive(client: Any, *, pace_s: float) -> Run:
    """The C1 drive over any ``SttClient``: the fixture, then zeros with a mark 1 s in, at least 3 s of them, and more
    until the echo or the deadline. ``pace_s`` is the wall time per frame (0.08 real time; 0 for the fake)."""
    run = Run()
    flushed = asyncio.Event()

    async def drain() -> None:
        async for event in client.events():
            run.events.append(event)
            if event.type == "flushed":
                flushed.set()

    async def tick() -> None:
        await asyncio.sleep(pace_s)
        for _ in range(3):
            await asyncio.sleep(0)

    task = asyncio.create_task(drain())
    try:
        for pcm, t_ms in fixture_frames():
            await client.feed(pcm, t_ms)
            await tick()
        t_ms = ZEROS_START
        n = 0
        while n < ZERO_FRAMES or (not flushed.is_set() and t_ms - ZEROS_START < FLUSH_DEADLINE_MS):
            if t_ms == T_EOT:
                await client.mark(T_EOT)
            await client.feed(ZERO, t_ms)
            t_ms += FRAME_MS
            n += 1
            run.fed_zero_ms = t_ms - ZEROS_START
            await tick()
        for _ in range(20):
            await tick()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return run


def check(run: Run, label: str) -> list[str]:
    """C1's three assertions; the reasons it fails (empty: it passed)."""
    problems: list[str] = []
    if not any(w.start_ms < ZEROS_START for w in run.words()):
        problems.append(f"{label}: the utterance was not heard (no word before {ZEROS_START} ms)")
    late = [(w.text, w.start_ms) for w in run.words() if w.start_ms >= ZEROS_START]
    if late:
        problems.append(f"{label}: words started inside the zeros (from {ZEROS_START} ms): {late}")
    if [f.t_eot_ms for f in run.flushed()] != [T_EOT]:
        problems.append(
            f"{label}: want one flushed(t_eot_ms={T_EOT}) for the mark issued during the zeros; got "
            f"{[f.t_eot_ms for f in run.flushed()]} after {run.fed_zero_ms} ms of zeros"
        )
    if run.fed_zero_ms < ZERO_FRAMES * FRAME_MS:
        problems.append(f"{label}: premise: at least 3 s of zeros fed; {run.fed_zero_ms} ms")
    return problems


def fake_client() -> FakeStt:
    spoken = json.loads(WORDS.read_text())["words"]
    script = [word_event(w["text"], w["start_ms"], w["end_ms"]) for w in spoken]
    script += [word_event(text, start, end) for text, start, end in GHOSTS]
    script.sort(key=lambda e: e.start_ms)
    return FakeStt(script, reactive=True, hears_audio=True)


async def test_c1_zeros_after_an_utterance_produce_no_word_and_a_mark_is_echoed(
    backend: str,
) -> None:
    """C1: the drive above; no word starts in the zeros, the utterance was heard, the mark issued during the zeros is
    echoed once with its ``t_eot_ms``. ``fake``: also every fixture word is heard and both ghosts are unheard.
    ``real``: n = 5 sequential runs, all must pass."""
    if backend == "fake":
        client = fake_client()
        run = await drive(client, pace_s=0.0)
        await client.close()
        problems = check(run, "fake")
        assert not problems, "\n".join(problems) + "\n" + run.summary()
        spoken = [w["text"] for w in json.loads(WORDS.read_text())["words"]]
        assert [w.text for w in run.words()] == spoken, (
            f"the fake hears every fixture word (their audio is not zero); {run.summary()}"
        )
        assert sorted(w.text for w in client.unheard) == sorted(g[0] for g in GHOSTS), (
            f"hears_audio: both ghosts were due and dropped as silence; unheard {client.unheard}"
        )
        return

    _skip_unless_services_are_up()
    from tests.fakes.moshi_stt_double import build_stt_client

    failures: list[str] = []
    summaries: list[str] = []
    for i in range(REAL_RUNS):
        client = build_stt_client()
        try:
            run = await drive(client, pace_s=FRAME_MS / 1000)
        finally:
            await client.close()
        summaries.append(f"run {i + 1}: {run.summary()}")
        failures += check(run, f"run {i + 1}")
    assert not failures, (
        f"T-0181 C1: {len(failures)} problem(s) over {REAL_RUNS} runs (all must pass):\n"
        + "\n".join(failures)
        + "\n"
        + "\n".join(summaries)
    )
