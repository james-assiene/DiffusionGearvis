#!/usr/bin/env python3
"""Rung 1/2 of the ladder, through the *production* session (T-0007 group B).

    uv run python scripts/demo/session_rung.py --image shot.png "what is on this screen?"
    uv run python scripts/demo/session_rung.py --listen --expect "what is the capital of france"

Unlike ``scripts/demo/ask.py`` — which talks to the engine directly on purpose — this drives the
real ``Session``: the real turn machine, speech gate, prompt builder, LLM client, filler policy,
ui gate and memory store, with the real vLLM / Pocket TTS / PipeWire clients wired in. That is the
point of the rung: the thing that answers is the thing under review, and it leaves one
``turns.jsonl`` line carrying its whole regime.

``--listen`` (rung 2, T-0032 group B) writes one **run directory** — ``--run-dir``, by default
``logs/rung2-<timestamp>`` — holding the artefacts ``tests/integration/test_session_listening_live.py``
reads (ruling A16 F2):

* ``turns.jsonl`` — the session's own turn lines, written straight into the run directory so it
  holds *this* run's turns and no earlier ones;
* ``rung.log`` — everything this script says, including one
  ``spoken=<what the operator said they would say> transcript=<what the session heard>`` line per
  utterance. ``spoken=`` is the operator's ``--expect`` sentence, stated *before* the run: the
  comparison is only evidence if the expected text cannot be written after hearing the answer;
* ``listening.json`` — per utterance ``eot_delay_ms`` (last word's audio time → ``E.SttEndOfTurn``
  submitted, read off the ``stt.end_of_turn`` span) and ``last_word_to_first_audio_ms``
  (``stt.last_word`` end → ``playback.first_sample``), each with the turn's whole ``Regime`` and
  the run's ``clock_offset_ms``. Both are **recorded, not thresholded**: n = 3 is not a B3 result
  (docs/03 §7);
* ``sockets.txt`` — ``ss -tunp`` lines belonging to this pid, sampled through the run
  (criterion 14: one non-loopback socket fails the task).

Two caveats that belong beside every number here. ``eot_delay_ms`` subtracts an *audio* time from
a *session-clock* time, and on hardware those origins differ (ruling A13 — the ADC zero is the
first PortAudio callback, the clock's zero is ``Session()``), so the run measures that gap once,
at the first fed frame, and carries it as ``clock_offset_ms`` on every row; the span is clamped at
zero upstream (``max(t_eot_ms, t_ms)`` in ``session.py``), so a **0** in that column means the two
times inverted, not that the decision was instant. And until T-0038 maps ``EndWord`` a word's
``end_ms`` equals its ``start_ms``, so "last word" means the last word's *start*.

Every endpoint comes from ``local_astra.config``, whose validator refuses anything but loopback
(docs/02 §9); nothing here names a URL. This script spends GPU time on an engine somebody else
started and claims no measurement, so it is deliberately ungated (T-0016 ruling D5) — harnesses
that measure live in ``scripts/gpu/`` and need a /launch token per run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, TextIO

from local_astra.conductor.audio_device import open_device
from local_astra.conductor.audio_io import PipeWireAudioIO
from local_astra.conductor.llm_client import VllmClient, VllmEndpoint
from local_astra.conductor.session import Session, SessionClock, SessionConfig
from local_astra.conductor.stt_client import MoshiSttClient
from local_astra.conductor.tts_client import PocketTtsClient
from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.config import Config, load
from local_astra.contracts import events as E
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.speech import SttEvent, SttStepEvent, SttWordEvent
from local_astra.memory.store import DEFAULT_DIR, SqliteMemory
from local_astra.telemetry.turns_log import TurnRecord

#: A question typed at the prompt still has to reach the machine as speech would: the rung scripts
#: the reflex/STT sequence of one utterance so the session runs its ordinary rows.
WORD_MS = 250

#: ``rung.log``, when a run directory is open: everything ``say`` prints, without the colour.
_LOG: TextIO | None = None


def open_run_log(path: Path) -> None:
    """Tee ``say`` into ``rung.log`` — the file the live test reads for its transcript lines."""
    global _LOG
    _LOG = path.open("w", encoding="utf-8")


def close_run_log() -> None:
    global _LOG
    if _LOG is not None:
        _LOG.close()
        _LOG = None


def say(message: str) -> None:
    print(f"\033[36m▸\033[0m {message}", file=sys.stderr, flush=True)
    if _LOG is not None:
        _LOG.write(f"{message}\n")
        _LOG.flush()


class FileFrames:
    """``PerceptionClient`` over one still image: rung 1's screen (docs/03 I1's canary fixture)."""

    def __init__(self, image: Path | None) -> None:
        self.frame: Frame | None = None
        #: a still image never changes scene; the list is here so the seam is the real one
        self.scene_events: list[object] = []
        if image is not None:
            jpeg = image.read_bytes()
            self.frame = Frame(
                source=Source.SCREEN,
                ts_ms=0,
                width=1280,
                height=720,
                jpeg=jpeg,
                phash=image.name,
                title=image.name,
                app="file",
            )

    async def get_frame(self, source: Source, at_ms: int | None = None) -> Frame | None:
        return self.frame

    async def clip(self, source: Source, t0_ms: int, t1_ms: int, fps: float = 1.0) -> list[Frame]:
        return [self.frame] if self.frame is not None else []

    async def events(self) -> AsyncIterator[object]:
        """A still image never changes scene, so this yields nothing and ends."""
        for event in self.scene_events:
            yield event


class PrsTap:
    """Delegating ``SttClient`` that prints the raw ``prs`` vector at the anchors that matter.

    T-0032 criterion 12's evidence: ``PAUSE_HEAD_INDEX`` is documented as *unverified* until a run
    reads the real vector (length, values at speech and at silence) off the box. Ruling A16 F2:
    the anchors are the ones the endpointer decides on, not the first 24 steps of the stream —
    which, on a run that starts before anyone speaks, are 24 vectors of pre-speech silence.

    What each anchor actually prints (ruling A17 (3) — the onset line used to claim something the
    code does not do):

    * **onset** — the most recent ``Step`` when the run's *first word is delivered*. Words arrive
      about one word-lag (≈ 0.5 s) after the audio they transcribe, and steps arrive in the same
      stream, so this vector is a step from *inside* the speech, not the silence-to-speech edge at
      ``start_ms``. That is still the useful half of criterion 12's pair: it is the **speech-side**
      vector, the one that must stay low on whichever head the endpointer reads.
    * **last word** — the most recent ``Step`` when the run's last word was delivered; also
      speech-side, one word-lag after the audio.
    * **end of turn** — the most recent ``Step`` at ``mark()``, i.e. the silence-side vector the
      decision was actually taken on. ``mark()`` is the anchor because the session awaits it once
      per end of turn, right after submitting ``E.SttEndOfTurn`` (criterion 5), so the decision is
      observable from a client wrapper without the rung submitting or inspecting a single machine
      event.

    Every line carries the step's own audio ``t_ms``, so the lag between an anchor's word time and
    its step time is visible in the log rather than argued about.

    The tap changes nothing on the wire — it forwards every event untouched — and only the rung
    mounts it, so the fast suite and ``astra session`` are unaffected.
    """

    def __init__(self, inner: MoshiSttClient, clock: SessionClock) -> None:
        self.inner = inner
        self.clock = clock
        #: ruling A13/A17 (2): session-clock time minus audio time on the very first fed frame —
        #: the gap between the two zeros (``Session()`` construction vs the first PortAudio
        #: callback) that criterion 13's ``eot_delay_ms`` silently contains. ``None`` until audio
        #: has been fed; measured once, never applied to anything here.
        self.clock_offset_ms: int | None = None
        self.utterances = 0
        self._latest: SttStepEvent | None = None
        self._onset_step: SttStepEvent | None = None
        self._last_word: tuple[SttWordEvent, SttStepEvent | None] | None = None
        self._run_open = False

    def _show(self, where: str, step: SttStepEvent | None) -> None:
        if step is None:
            say(f"prs at {where}: no Step seen yet (the server sent no pause heads)")
            return
        say(
            f"prs at {where} (n={len(step.pause_heads)}, audio {step.t_ms} ms): "
            f"{[round(p, 6) for p in step.pause_heads]}"
        )

    async def feed(self, pcm_24k_f32: bytes, t_ms: int) -> None:
        if self.clock_offset_ms is None:
            # Before the await, so the number is the offset at the first frame and not the offset
            # plus however long the first feed took.
            self.clock_offset_ms = self.clock.now_ms - t_ms
            say(
                f"clock_offset_ms={self.clock_offset_ms} (session clock − audio clock at the "
                "first fed frame; eot_delay_ms below contains it — ruling A13)"
            )
        await self.inner.feed(pcm_24k_f32, t_ms)

    async def mark(self, t_eot_ms: int) -> int:
        if self._last_word is not None:
            word, step = self._last_word
            self._show(
                f"utterance {self.utterances} last word ({word.text!r} at {word.end_ms} ms)", step
            )
        self._show(f"utterance {self.utterances} end of turn (t_eot={t_eot_ms} ms)", self._latest)
        self._run_open = False
        self._last_word = None
        return await self.inner.mark(t_eot_ms)

    async def close(self) -> None:
        await self.inner.close()

    async def events(self) -> AsyncIterator[SttEvent]:
        async for event in self.inner.events():
            if isinstance(event, SttStepEvent):
                self._latest = event
            elif isinstance(event, SttWordEvent):
                if not self._run_open:
                    self._run_open = True
                    self.utterances += 1
                    self._onset_step = self._latest
                    self._show(
                        f"utterance {self.utterances} onset "
                        f"({event.text!r} at {event.start_ms} ms)",
                        self._onset_step,
                    )
                self._last_word = (event, self._latest)
            yield event


async def sample_sockets(pid: int, seen: dict[str, None]) -> None:
    """Add this process's own ``ss -tunp`` lines to ``seen`` (criterion 14).

    One sample, called at the points where a socket can have appeared — after ``start()`` and
    after each utterance — rather than from a timer: the rung is inside the "no second timer
    mechanism" check of ``tests/unit/test_session_loop.py``, and a polling loop here would be a
    second clock in a file that must not have one. It records the lines verbatim and judges
    nothing: the live test, not this script, decides whether an address is loopback.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ss", "-tunp", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
    except FileNotFoundError:
        say("ss is not installed: sockets.txt will not be written (criterion 14 unproven)")
        return
    out, _ = await proc.communicate()
    for line in out.decode(errors="replace").splitlines():
        if f"pid={pid}," in line:
            seen.setdefault(" ".join(line.split()), None)


def write_sockets(run_dir: Path, pid: int, seen: dict[str, None]) -> None:
    if not seen:
        say(f"no socket line mentioned pid={pid}: sockets.txt is not written (nothing to show)")
        return
    path = run_dir / "sockets.txt"
    header = "Netid State Recv-Q Send-Q Local Peer Process"
    path.write_text("\n".join([header, *seen]) + "\n", encoding="utf-8")
    say(f"sockets.txt: {len(seen)} distinct lines for pid={pid}")


def listening_rows(
    run_dir: Path, spoken: list[str], heard: list[str], clock_offset_ms: int | None
) -> list[dict[str, Any]]:
    """Criterion 13, read off the turn lines the session itself wrote.

    ``eot_delay_ms`` is the ``stt.end_of_turn`` span: the last word's audio time → the clock time
    at which ``E.SttEndOfTurn`` was submitted (``_note_event`` stores exactly that pair). On CPU
    STT it is dominated by the ``Marker`` echo wait (3.0 s in the T-0007a census).
    ``last_word_to_first_audio_ms`` is ``stt.last_word`` end → ``playback.first_sample``, B3's
    second column. Neither is thresholded here — n = 3 is not a B3 result.

    **Neither is clamped here, but ``eot_delay_ms`` is clamped upstream** (ruling A17 (1)):
    ``session.py`` writes the span as ``(t_eot_ms, max(t_eot_ms, t_ms))``, so an inverted pair
    reads **0**, not a negative number. A 0 in this column is therefore not "instant": it means
    the clock time was at or behind the audio time, which is what the two zero origins of ruling
    A13 look like. ``clock_offset_ms`` — the session clock minus the audio clock at the first fed
    frame, measured once by ``PrsTap.feed`` — travels on every row so both columns can be debiased
    afterwards. It is recorded, never subtracted here: the raw span stays the evidence.
    """
    path = run_dir / "turns.jsonl"
    if not path.is_file():
        return []
    records = [TurnRecord.from_line(line) for line in path.read_text().splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "turn_id": record.turn_id,
            "outcome": record.outcome,
            "clock_offset_ms": clock_offset_ms,
            "spoken": spoken[index] if index < len(spoken) else None,
            "transcript": heard[index] if index < len(heard) else None,
        }
        eot = record.spans.get("stt.end_of_turn")
        if eot is not None:
            row["eot_delay_ms"] = eot[1] - eot[0]
        first_audio = record.latency_ms("stt.last_word", "playback.first_sample")
        if first_audio is not None:
            row["last_word_to_first_audio_ms"] = first_audio
        row["spans"] = {name: list(span) for name, span in record.spans.items()}
        row["extras"] = record.extras
        row["regime"] = record.regime.model_dump(mode="json")
        rows.append(row)
    return rows


def build(
    config: Config, data_dir: Path, args: argparse.Namespace, turns_path: Path | None = None
) -> tuple[Session, SessionClock]:
    services = config.active().services
    clock = SessionClock()
    machine = TurnMachine(TurnPolicy(thinking=config.active().diffusion.thinking))
    # T-0006 ruling A1: the playback module takes its device by injection. The rung names the sink
    # rather than letting the default pick one, because which sink is playing *is* part of the
    # regime this run reports (``--audio-path``, docs/01 §8: headset and speakers are not
    # comparable), and because a missing PortAudio then fails here with T-0006's own message
    # instead of somewhere inside the first utterance.
    audio = PipeWireAudioIO(open_device(sink=args.sink))
    session = Session(
        machine,
        # T-0032 ruling A10 (same defect as criterion 9(b) in cli.py): the endpoint comes from
        # the profile's ``Services``, whose validator is the loopback gate — not from
        # ``VllmEndpoint``'s default port, which a profile that moves vLLM would silently miss.
        VllmClient(VllmEndpoint(base_url=services.vllm)),
        # T-0007a ruling A5: both clients take the whole ``Services`` object, not one endpoint
        # string — that is the object whose validator refuses anything but loopback (docs/02 §9),
        # and passing a bare URL would put a second source of truth in front of it.
        PrsTap(MoshiSttClient(services), clock) if args.listen else MoshiSttClient(services),
        PocketTtsClient(services),
        audio,
        FileFrames(Path(args.image) if args.image else None),
        SqliteMemory(),
        clock=clock,
        config=SessionConfig(
            data_dir,
            # --listen writes the turn lines into the run directory, so the artefact the live test
            # reads holds this run's turns and nothing older (it asserts one line per utterance).
            turns_path=turns_path,
            profile=config.active(),
            audio_path=args.audio_path,
            source=Source.SCREEN,
        ),
    )
    return session, clock


async def typed_turn(session: Session, clock: SessionClock, question: str) -> None:
    """Drive one turn from a typed question: onset → words → end-of-turn → flushed."""
    onset = clock.now_ms
    session.submit(E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    for index, word in enumerate(question.split()):
        start = onset + index * WORD_MS
        session.submit(
            E.SttWord(t_ms=clock.now_ms, text=word, start_ms=start, end_ms=start + WORD_MS)
        )
    eot = onset + len(question.split()) * WORD_MS
    session.submit(E.SttEndOfTurn(t_ms=clock.now_ms, t_eot_ms=eot, p_end=0.9))
    session.submit(E.SttFlushed(t_ms=clock.now_ms, t_eot_ms=eot))


async def listen_turns(
    session: Session,
    args: argparse.Namespace,
    expect: list[str],
    heard: list[str],
    sockets: dict[str, None],
) -> int:
    """Rung 2: the session listens (T-0032 criterion 11). The rung submits **nothing**.

    Every event comes from the production path — the capture pump feeds ``stt_blocks()`` to the
    STT service, the endpointer turns the pause heads into onset and end of turn, and the
    ``Marker`` echo is what leaves ENDPOINTING (docs/02 §4 row 6). This function only waits for
    the turn lines and prints what was heard, so a transcript that does not match the spoken
    sentence is the session's answer and not the harness's.

    ``heard`` collects the transcripts as they arrive, so the caller can still write
    ``listening.json`` for the utterances that did land when a later one times out.

    ``expect[i]`` is what the operator said, before the run, they would say for utterance *i*;
    it is printed beside the transcript as ``spoken=... transcript=...``. The assistant's own
    answer is printed under a different key on purpose: ``spoken=`` in this file means the
    human's sentence.
    """
    say(f"--listen: speak {len(expect)} question(s); Ctrl-C to stop")
    for index, sentence in enumerate(expect, start=1):
        say(f"expected utterance {index}: {sentence}")
    say(
        "barge-in is not wired on this path (the self-speech gate silences the STT tap while we "
        "speak); T-0034 owns it"
    )
    while len(heard) < len(expect):
        session.turn_written.clear()
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=args.timeout_s)
        except TimeoutError:
            say(
                f"no turn within {args.timeout_s} s (state={session.state.state}); "
                f"{len(heard)}/{len(expect)} utterances"
            )
            return 1
        turn = session.state.history[-1] if session.state.history else None
        transcript = turn.user_text if turn is not None else ""
        heard.append(transcript)
        # The line the live test parses: one per utterance, the operator's sentence first.
        say(f"spoken={expect[len(heard) - 1]} transcript={transcript}")
        if turn is not None:
            say(f"answer: {turn.assistant_spoken}")
        await sample_sockets(os.getpid(), sockets)
    return 0


async def run(args: argparse.Namespace) -> int:
    config = load(Path(args.config) if args.config else None)
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DIR
    run_dir: Path | None = None
    turns_path: Path | None = None
    if args.listen:
        run_dir = Path(args.run_dir) if args.run_dir else Path("logs") / f"rung2-{_stamp()}"
        turns_path = run_dir / "turns.jsonl"
    expect = list(args.expect or [])
    if args.listen and not expect:
        # Never invent the operator's sentence: an unstated expectation is written as such and
        # the live test's 3-of-3 comparison fails on it, which is the honest outcome.
        expect = ["<unstated>"] * args.utterances
    if args.dry_run:
        # Opens no device, contacts no service, writes nothing — the check that runs on a box
        # with no PortAudio and no services (the artefact paths are printed, not created).
        say("--dry-run: no device is opened, no service is contacted, no turn is taken")
        say(f"profile={config.profile} data_dir={data_dir} audio_path={args.audio_path}")
        say(
            f"would write into: {run_dir if run_dir is not None else '(no run dir without --listen)'}"
        )
        for name in ("turns.jsonl", "rung.log", "listening.json", "sockets.txt"):
            say(f"  would write: {run_dir / name if run_dir is not None else '-'}")
        for index, sentence in enumerate(expect, start=1):
            say(f"  would expect utterance {index}: {sentence}")
        return 0
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        open_run_log(run_dir / "rung.log")
        say(f"run_dir={run_dir}")
        if not args.expect:
            say("--expect was not given: the spoken= column is '<unstated>' and cannot match")
    session, clock = build(config, data_dir, args, turns_path)
    say(f"profile={config.profile} data_dir={data_dir} turns={session.config.turns_path}")
    say(f"audio_path={args.audio_path} sink={args.sink or 'default'}")
    heard: list[str] = []
    sockets: dict[str, None] = {}
    await session.start()
    try:
        if args.listen:
            await sample_sockets(os.getpid(), sockets)
            return await listen_turns(session, args, expect, heard, sockets)
        session.turn_written.clear()
        await typed_turn(session, clock, args.question)
        try:
            await asyncio.wait_for(session.turn_written.wait(), timeout=args.timeout_s)
        except TimeoutError:
            say(f"the turn did not finish within {args.timeout_s} s (state={session.state.state})")
            return 1
        for turn in session.state.history:
            say(f"user: {turn.user_text}")
            say(f"spoken: {turn.assistant_spoken}")
    finally:
        await session.aclose()
        if run_dir is not None:
            # After aclose, so the last turn line is on disk before it is read back.
            write_sockets(run_dir, os.getpid(), sockets)
            tap = session.stt if isinstance(session.stt, PrsTap) else None
            offset = tap.clock_offset_ms if tap is not None else None
            say(f"clock_offset_ms={offset} (ruling A13: it is inside eot_delay_ms)")
            rows = listening_rows(run_dir, expect, heard, offset)
            (run_dir / "listening.json").write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8"
            )
            for row in rows:
                say(
                    f"utterance {row['turn_id']}: eot_delay_ms={row.get('eot_delay_ms')} "
                    f"last_word_to_first_audio_ms={row.get('last_word_to_first_audio_ms')} "
                    f"(recorded, not a B3 result; n=3)"
                )
            say(f"artefacts: {run_dir} (turns.jsonl, rung.log, listening.json, sockets.txt)")
            close_run_log()
    say(f"turns.jsonl: {session.config.turns_path}")
    return 0


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="what is on this screen?")
    parser.add_argument("--image", help="still image to answer about (rung 1)")
    parser.add_argument("--listen", action="store_true", help="spoken question (rung 2)")
    parser.add_argument("--config", help="astra.toml")
    parser.add_argument("--data-dir", help="where thumbnails and turns.jsonl go")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument(
        "--utterances",
        type=int,
        default=3,
        help="spoken questions to wait for (--listen); --expect overrides the count",
    )
    parser.add_argument(
        "--expect",
        action="append",
        metavar="SENTENCE",
        help="what you will say, in order, stated before the run (repeatable): it becomes the "
        "'spoken=' column the transcript is compared against (--listen)",
    )
    parser.add_argument(
        "--run-dir",
        help="where --listen writes turns.jsonl, rung.log, listening.json and sockets.txt "
        "(default logs/rung2-<timestamp>); this is ASTRA_LISTEN_RUN for the live test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build nothing and open no device: print what the run would write and exit",
    )
    parser.add_argument("--sink", help="PipeWire node name or id to play through")
    parser.add_argument(
        "--audio-path",
        default="speakers",
        help="what is actually playing, for the regime on the turns.jsonl line "
        "(headset | speakers+aec | loopback)",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
