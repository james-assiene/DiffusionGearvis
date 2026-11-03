"""T-0007a group B, the census rows: what moshi-server 0.6.4 *actually* sends on this box.

Criterion 12 ("wire census") and the open questions behind criteria 13–16 are **unknowns**, not
assertions. `tests/contract/test_speech_clients_contract.py`'s `[real]` rows pin what is certain
against the live service (a word arrives; `close()` ends the stream and is idempotent; the TTS body
is whole 24 kHz s16le frames). What no row has ever printed is the raw traffic, so five facts that
the rest of the system assumes are still assumptions:

* the word-timestamp **field name** and its **unit** — `tests/fakes/moshi_stt_double.py::word`
  spells it `start_time` in seconds because that is the upstream (delayed-streams-modeling)
  spelling, and `docs/tasks/T-0007a.md` ruling A1 says only the census can settle it. If the server
  spells it differently, every live word time is the client's fed-audio fallback and
  `stats()["estimated_word_times"]` is the tell;
* whether a **`Ready`** message arrives (ruling A8 keeps the mapping total meanwhile, so a `Ready`
  shows up as one `SttErrorEvent` naming an unknown type — a census item, not a failure);
* whether the fixture sentence comes back as **numerals** ("1") or **number words** ("one"), which
  decides whether T-0007's history and any WER eval can compare transcript text to fixture text at
  all;
* the **`Marker` echo shape** (does the echo carry the id back? any other field?);
* the **wall time** from the last fed block to the flushed event.

So each row here asserts only what the contract rows already proved is certain, and *records* the
rest: the raw msgpack messages are printed under ``-rA``.
Nothing here is a threshold and nothing here may become one without a task entry.

**Regime** (state it beside every number; none of these figures is comparable to anything else):
STT = kyutai/stt-1b-en_fr-candle in astra-moshi:0.6.4 on **CPU** (`ASTRA_STT_CPU=1`),
batch_size=2, **co-resident with vLLM**, box contended · audio in =
`tests/fixtures/audio/pocket_tts_sentence.wav` (24 kHz mono s16le, 4.040 s) fed at realtime pace
with 3.2 s of trailing silence added by the harness (the client pads nothing — T-0021's ruling) ·
**n = 1**, seed n/a · docs/03 §5's 0.5 ± 0.15 s word lag is a GPU-profile number and this is **not
comparable** to it.

**What the first census run found** (commit f114adf, log `logs/t0007a-group-b-census-f114adf.log`,
2/3 rows; the marker row failed on a harness bug in this file, fixed — see
`capture_one_utterance`). Recorded here so the next run has something to differ from, *not*
asserted anywhere: 100 inbound messages in 4 types — `Ready` once, `Step` x90 (keys
`step_idx`/`prs`/`buffered_pcm`), `Word` x5 (`text` + **`start_time` in seconds**, so ruling A1's
assumption holds and `tests/fakes/moshi_stt_double.py::word` is right), and **`EndWord`
{'stop_time': seconds} after each word** — a type nothing in the client maps today, which is why
`unknown_messages` was 5 and why every live word has `end_ms == start_ms` (the client invents no
duration — criterion 3 — and the stop time is sitting unused on the wire). Mapping `EndWord` and
`Ready` is **T-0038** (`docs/tasks/T-0038.md`), not this module's business: these rows record, they
do not gate. The transcript came back as **number words with punctuation** (`'one,' 'two,' …`), and
the server ran at RTF ≈ 1.3 on CPU (90 `Step`s = 7.2 s of model time for 10.4 s of audio fed).

**What the second run found** (commit 3e9813b, log `logs/t0007a-group-b-census-3e9813b.log`, 2/3):
no `Marker` echo in 15 s of waiting, and only 20 `Step`s in that window — 1.6 s of model time for
15 s of wall — against 90 `Step`s for the 10.4 s fed. **The server does not chew its backlog once
the audio inflow stops**: it had processed 7.2 s of 10.4 s and then near-stalled, so the marker,
which sits behind the unprocessed 3.2 s, was never reached. The harness now keeps the pump running
(see `capture_one_utterance`). It also means the earlier "the Marker echo is confirmed on this box"
reading of `tests/contract/test_speech_clients_contract.py`'s service row was **vacuous**:
`all(e.t_eot_ms >= 0 for e in flushed)` is true of an empty list and `drain.quiet(2.0)` returns on
two seconds of quiet whether or not a flush ever came. That row was fixed in the same commit as
this note; whether moshi-server 0.6.4 echoes a `Marker` at all is, as of 3e9813b, **still an open
census question** and docs/02 §2.3 / T-0003's `ENDPOINTING` row depend on the answer.

Run only through ``/launch``:
``uv run pytest tests/integration/test_speech_clients_live.py -q -m gpu -rA``
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import msgpack
import pytest

from local_astra.conductor.stt_client import SttConnection, _websocket_connect
from local_astra.contracts.speech import TTS_OUTPUT_FORMAT
from tests.contract.test_speech_clients_contract import (
    BLOCK_SAMPLES,
    FIXTURE,
    SENTENCE,
    _fixture_blocks,
    _skip_unless_services_are_up,
)
from tests.fakes.moshi_stt_double import build_stt_client

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

#: The fixture's duration in seconds, the oracle for "is this timestamp seconds or milliseconds?".
FIXTURE_S = 4.040
#: Trailing silence supplied by the harness (blocks of 80 ms), as in the contract module's live row.
TAIL_BLOCKS = 40
#: Realtime pace between fed blocks.
PACE_S = 0.08
#: How long the harness keeps feeding silence after `mark()` before calling the echo absent.
FLUSH_DEADLINE_S = 15.0

#: Fields a `Word` might carry a time in. The census prints whichever it finds — and prints the
#: whole message anyway, so a sixth spelling is visible rather than silently missing.
TIME_FIELD_CANDIDATES = ("start_time", "start", "start_s", "start_ms", "t", "time", "timestamp")


# --- capture ---------------------------------------------------------------------------------


@dataclass
class LiveCapture:
    """One live run: every raw inbound message, every mapped event, and two wall clocks."""

    raw_inbound: list[Any] = field(default_factory=list)
    #: `time.monotonic()` at which each `raw_inbound` message was read, same index.
    inbound_at: list[float] = field(default_factory=list)
    raw_outbound_types: list[str] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    marker_id: int = 0
    t_eot_ms: int = 0
    last_fed_at: float = 0.0
    marked_at: float = 0.0
    #: silence blocks fed *after* `mark()` to keep the model's clock moving (see the harness note).
    silence_blocks_after_mark: int = 0
    flushed_at: float | None = None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def words(self) -> list[Any]:
        return [e for e in self.events if e.type == "word"]

    @property
    def flushed(self) -> list[Any]:
        return [e for e in self.events if e.type == "flushed"]

    @property
    def inbound_types(self) -> list[str]:
        return [
            str(m.get("type", "<no type field>"))
            if isinstance(m, dict)
            else f"<{type(m).__name__}>"
            for m in self.raw_inbound
        ]

    @property
    def flush_latency_ms(self) -> float | None:
        if self.flushed_at is None:
            return None
        return (self.flushed_at - self.last_fed_at) * 1000.0

    def inbound_after_mark(self, type_name: str) -> list[tuple[float, Any]]:
        """`(seconds since mark(), message)` for every message of one type read after `mark()`.

        On CPU the server runs behind the audio (RTF ≈ 1.3 measured 2026-09-14), so the count of
        `Step`s here is how far behind it was when the marker was sent — the thing that decides
        whether a deadline is generous or lucky.
        """
        return [
            (at - self.marked_at, message)
            for at, message in zip(self.inbound_at, self.raw_inbound, strict=False)
            if at >= self.marked_at
            and isinstance(message, dict)
            and message.get("type") == type_name
        ]


class RecordingConnection:
    """The real connection, with every inbound frame decoded by the real codec and kept.

    The client maps messages into `SttEvent`s and the raw shape is lost; the census needs the raw
    shape, so it is tapped here at the transport seam the client already exposes (`connect=`)
    rather than by re-implementing the client's framing.
    """

    def __init__(self, inner: SttConnection, capture: LiveCapture) -> None:
        self._inner = inner
        self._log = capture.raw_inbound
        self._at = capture.inbound_at
        self._sent = capture.raw_outbound_types

    async def send(self, data: bytes) -> None:
        try:
            message = msgpack.unpackb(bytes(data))
            self._sent.append(str(message.get("type")) if isinstance(message, dict) else "?")
        except Exception as exc:  # a tap must never change what the client does
            self._sent.append(f"<undecodable: {type(exc).__name__}>")
        await self._inner.send(data)

    async def recv(self) -> Any:
        raw = await self._inner.recv()
        self._at.append(time.monotonic())
        try:
            self._log.append(msgpack.unpackb(bytes(raw)))
        except Exception as exc:
            self._log.append({"type": "<undecodable>", "error": f"{type(exc).__name__}: {exc}"})
        return raw

    async def close(self) -> None:
        await self._inner.close()


async def capture_one_utterance(
    *,
    connect_base: Callable[[str], Any] = _websocket_connect,
    services: Any = None,
    blocks: Sequence[bytes] | None = None,
) -> LiveCapture:
    """Feed the fixture at realtime pace, `mark()`, and record everything that came back.

    `connect_base`/`services` exist so the same capture path can be driven against
    `tests/fakes/moshi_stt_double.py`'s loopback double off the box; the live rows use the
    defaults, which is the production transport.

    **`events()` is drained by a task started before the first block is fed**, the way
    `tests/contract/test_speech_clients_contract.py`'s `EventDrain` does it. The first cut of this
    harness only started iterating after `mark()` and polled with
    ``asyncio.wait_for(iterator.__anext__(), 1.0)``: a `wait_for` timeout *cancels* the pending
    `__anext__`, which finalises the async generator, so the very next `__anext__` raised
    `StopAsyncIteration` and the capture ended ~3 s after `mark()` — while the CPU server, running
    at RTF ≈ 1.3 (90 `Step`s = 7.2 s processed for 10.4 s fed), was still catching up. It reported
    "no flush arrived" for a marker the server had not reached yet (census run f114adf). Nothing
    here may wait on `__anext__` with a timeout again.

    **Silence keeps being fed after `mark()`** until the flush arrives or the deadline expires.
    Census run 3e9813b waited the full 15 s in *pure silence on the wire* and saw 20 `Step`s —
    1.6 s of model time in 15 s of wall — after 90 `Step`s for the 10.4 s fed: the server does not
    chew its backlog once the audio inflow stops, so a marker sitting behind 3.2 s of unprocessed
    audio is never reached. Continuing to feed is not a workaround, it is the production shape: a
    live session's capture pump (T-0021) never stops feeding while a turn is open, and the
    delayed-stream model's clock only advances while audio arrives (`scripts/demo/_listen.py`'s
    measured fact). If the echo still does not arrive *with* continuous audio, that is the census
    answer — moshi-server 0.6.4 does not echo `Marker` in this configuration — and the row records
    it and fails loudly rather than quietly redefining the deadline.
    """
    capture = LiveCapture()

    async def connect(url: str) -> SttConnection:
        inner: SttConnection = await connect_base(url)
        return RecordingConnection(inner, capture)

    client = build_stt_client(services, connect=connect)
    flush_seen = asyncio.Event()

    async def drain() -> None:
        """One iterator over `events()`, from before the first feed until `close()` ends it."""
        async for event in client.events():
            capture.events.append(event)
            if event.type == "flushed" and capture.flushed_at is None:
                capture.flushed_at = time.monotonic()
                flush_seen.set()

    drain_task = asyncio.create_task(drain())
    try:
        t_ms = 0
        for block in blocks if blocks is not None else _fixture_blocks():
            await client.feed(block, t_ms)
            t_ms += len(block) // 4 * 1000 // TTS_OUTPUT_FORMAT.sample_rate
            await asyncio.sleep(PACE_S)
        for _ in range(TAIL_BLOCKS):  # the harness supplies the tail; the client pads nothing
            await client.feed(bytes(BLOCK_SAMPLES * 4), t_ms)
            t_ms += 80
            await asyncio.sleep(PACE_S)
        capture.last_fed_at = time.monotonic()
        capture.t_eot_ms = t_ms
        capture.marked_at = time.monotonic()
        capture.marker_id = await client.mark(t_ms)

        # Keep the pump running: feed silence at realtime pace until the flush arrives or the
        # deadline expires. Never wait on the generator (see the docstring), and never stop the
        # audio — a stopped stream stops the model's clock and the marker is never reached.
        deadline = time.monotonic() + FLUSH_DEADLINE_S
        while not flush_seen.is_set() and time.monotonic() < deadline:
            await client.feed(bytes(BLOCK_SAMPLES * 4), t_ms)
            t_ms += 80
            capture.silence_blocks_after_mark += 1
            await asyncio.sleep(PACE_S)
        capture.stats = dict(client.stats())
    finally:
        await client.close()  # ends events(), so the drain task finishes on its own
        await asyncio.wait_for(asyncio.gather(drain_task, return_exceptions=True), 5.0)
    return capture


_CACHE: dict[str, LiveCapture] = {}


async def live_capture() -> LiveCapture:
    """One live run shared by the rows below — 7 s of realtime audio is not worth feeding thrice."""
    _skip_unless_services_are_up()
    if "run" not in _CACHE:
        _CACHE["run"] = await capture_one_utterance()
    return _CACHE["run"]


# --- the census, as text -----------------------------------------------------------------------


def _example(messages: Sequence[Any]) -> str:
    return repr(messages[0]) if messages else "<none>"


def message_census(raw_inbound: Sequence[Any]) -> list[str]:
    """Criterion 12's table: every inbound type with its count, key set and one example value."""
    lines: list[str] = []
    order: list[str] = []
    by_type: dict[str, list[Any]] = {}
    for message in raw_inbound:
        name = (
            str(message.get("type", "<no type field>"))
            if isinstance(message, dict)
            else f"<{type(message).__name__}>"
        )
        if name not in by_type:
            by_type[name] = []
            order.append(name)
        by_type[name].append(message)
    lines.append(f"inbound messages: {len(raw_inbound)} in {len(order)} types")
    for name in order:
        group = by_type[name]
        keys: list[str] = []
        for message in group:
            if isinstance(message, dict):
                for key in message:
                    if str(key) not in keys:
                        keys.append(str(key))
        lines.append(f"  {name}: n={len(group)} keys={keys} example={_example(group)}")
    for expected in ("Word", "Step", "Marker", "EndWord", "Ready", "Error"):
        lines.append(f"  present? {expected}: {'yes' if expected in by_type else 'no'}")
    return lines


def timestamp_census(raw_inbound: Sequence[Any], audio_s: float) -> list[str]:
    """Which field (if any) a `Word` carries a time in, and whether it reads as s or ms.

    The unit is inferred from the audio actually fed: a time in seconds cannot exceed the audio's
    duration by much, while the same instant in milliseconds is ~1000x larger. The verdict is
    printed, never asserted — it is the input to ruling A1, not its conclusion.
    """
    words = [m for m in raw_inbound if isinstance(m, dict) and m.get("type") == "Word"]
    lines = [f"word messages: {len(words)}; audio fed: {audio_s:.3f} s"]
    if not words:
        return [*lines, "  no Word message: nothing to say about the timestamp field"]
    keys: list[str] = []
    for message in words:
        for key in message:
            if str(key) not in keys:
                keys.append(str(key))
    lines.append(f"  Word key set: {keys}")
    found = [k for k in keys if k in TIME_FIELD_CANDIDATES]
    numeric = [
        k
        for k in keys
        if any(isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool) for m in words)
    ]
    lines.append(f"  known time-field spellings present: {found or 'none'}")
    lines.append(f"  numeric fields (any spelling): {numeric or 'none'}")
    for key in numeric:
        values = [m[key] for m in words if isinstance(m.get(key), (int, float))]
        biggest = max(values)
        if biggest <= audio_s * 1.5:
            unit = "reads as SECONDS"
        elif biggest <= audio_s * 1500:
            unit = "reads as MILLISECONDS"
        else:
            unit = "neither s nor ms of the fed audio (token index? sample count?)"
        lines.append(f"  {key}: first={values[0]!r} max={biggest!r} -> {unit}")
    return lines


NUMBER_WORDS = ("one", "two", "three", "four", "five")


def numeral_census(texts: Sequence[str]) -> list[str]:
    """Numerals or number words for "one, two, three, four, five"? (criterion 13's open half.)"""
    folded = [t.strip().strip(".,!?").lower() for t in texts]
    digits = [t for t in folded if any(c.isdigit() for c in t)]
    spelled = [t for t in folded if t in NUMBER_WORDS]
    if digits and not spelled:
        verdict = "NUMERALS"
    elif spelled and not digits:
        verdict = "NUMBER WORDS"
    elif spelled and digits:
        verdict = "MIXED"
    else:
        verdict = "NEITHER (no numeral and no number word in the transcript)"
    return [
        f"fixture text: {SENTENCE!r} ({FIXTURE.name})",
        f"transcript words: {list(texts)}",
        f"joined: {' '.join(texts)!r}",
        f"digit-bearing: {digits} · spelled: {spelled}",
        f"numerals or number words? {verdict}",
    ]


def _report(name: str, lines: Sequence[str]) -> None:
    """Print one census block.

    ``print`` and not ``record_property``: this repo's ``junit_family`` is ``xunit2``, under which
    ``record_property`` records nothing and only warns ("record_property is incompatible with
    junit_family 'xunit2'", census run f114adf). The printed block is what ``-rA`` shows and what
    the Runs entry is pasted from.
    """
    print(f"\n--- {name} ---\n" + "\n".join(lines))


# --- the rows --------------------------------------------------------------------------------


async def test_the_wire_census_records_every_message_type_and_the_word_time_field() -> None:
    """Criterion 12, recorded: the raw traffic, the key sets, `Ready`, and ruling A1's field name.

    Certain (proved by the contract module's `[real]` rows on 2026-09-14): at least one message
    arrives and at least one is a `Word`. Everything else is printed for the report.
    """
    capture = await live_capture()

    _report("wire-census", message_census(capture.raw_inbound))
    _report(
        "word-timestamp-census",
        timestamp_census(capture.raw_inbound, FIXTURE_S + TAIL_BLOCKS * PACE_S),
    )
    _report(
        "client-stats",
        [
            f"outbound types sent: {sorted(set(capture.raw_outbound_types))}",
            f"stats(): {capture.stats}",
            "estimated_word_times > 0 means the client found no server timestamp and fell back "
            "to its fed-audio clock (criterion 3)",
        ],
    )
    _report(
        "regime",
        [
            "STT on CPU (ASTRA_STT_CPU=1), moshi-server 0.6.4, batch_size=2, vLLM co-resident,",
            f"audio={FIXTURE.name} 24 kHz mono s16le {FIXTURE_S:.3f} s at realtime pace + "
            f"{TAIL_BLOCKS * PACE_S:.1f} s harness silence, n=1, seed n/a.",
            "NOT comparable to docs/03 §5 (GPU profile) or to any other run.",
        ],
    )

    assert capture.raw_inbound, "no message at all arrived: the census below would be vacuous"
    assert any(t == "Word" for t in capture.inbound_types), (
        f"no Word message arrived; types were {sorted(set(capture.inbound_types))}"
    )


async def test_the_transcript_records_whether_numerals_or_number_words_come_back() -> None:
    """Criterion 13's open half: recorded, not gated (CPU, contended, n=1).

    Certain: words arrive as `word` events with non-negative, non-decreasing-ish times — the
    contract row already pins arrival and sign, so the only assertion here is that the transcript
    is not empty, which keeps the census from being vacuous.
    """
    capture = await live_capture()
    texts = [w.text for w in capture.words]

    _report("numeral-census", numeral_census(texts))
    _report(
        "word-events",
        [f"{w.start_ms:>6} ms .. {w.end_ms:>6} ms  {w.text!r}" for w in capture.words]
        or ["<no word event>"],
    )

    assert texts, (
        "no word event arrived from the live service, so there is no transcript to record "
        f"(events: {[e.type for e in capture.events]})"
    )


async def test_the_marker_echo_shape_and_the_flush_wall_time_are_recorded() -> None:
    """Criterion 12's marker question and the flush latency: recorded, and the row that fails.

    This is the one row here that can legitimately stay red. Criterion 4 is the client's contract —
    a `Marker` echo, and only an echo, becomes an `SttFlushedEvent` carrying the `t_eot_ms` that
    marker was asked for — and T-0003's `ENDPOINTING` row waits on it. Whether moshi-server 0.6.4
    on this box echoes at all is criterion 12's open question: run f114adf could not tell (the
    harness stopped early), run 3e9813b waited 15 s in silence and saw none, and the harness now
    keeps feeding so the model's clock keeps moving. If it still does not arrive, **the failure is
    the finding** and goes back to the orchestrator as a spec question — the client must not
    synthesise a flush to make this green (criterion 4's adversarial half).
    """
    capture = await live_capture()
    echoes = capture.inbound_after_mark("Marker")
    steps_after = capture.inbound_after_mark("Step")
    latency = capture.flush_latency_ms
    echo_ms = f"{echoes[0][0] * 1000:.0f} ms" if echoes else "<no echo>"

    _report(
        "marker-census",
        [
            f"client sent Marker id={capture.marker_id} for t_eot_ms={capture.t_eot_ms}",
            f"raw Marker echoes after mark(): {[m for _, m in echoes] or '<none>'}",
            f"echo key set: {sorted({str(k) for _, m in echoes for k in m})}",
            f"flushed events: {[e.model_dump() for e in capture.flushed]}",
            f"mark() -> Marker echo on the wire: {echo_ms}",
            f"silence blocks fed after mark(): {capture.silence_blocks_after_mark} "
            f"(= {capture.silence_blocks_after_mark * 80 / 1000:.1f} s at realtime pace; the pump "
            "never stops, as a live session's would not)",
            "last fed block -> flushed event: "
            + (f"{latency:.0f} ms" if latency is not None else "<no flush arrived>"),
            f"Steps read after mark(): {len(steps_after)} "
            f"(= {len(steps_after) * 80 / 1000:.1f} s of model time; the CPU server's backlog, "
            "which is why the deadline is 15 s and not 3)",
            "regime: CPU STT, vLLM co-resident, n=1 — not comparable to docs/03 §5 or to B6.",
        ],
    )

    assert capture.flushed, (
        "no flushed event arrived within "
        f"{FLUSH_DEADLINE_S}s of mark(); events were {[e.type for e in capture.events]}"
    )
    assert capture.flushed[0].t_eot_ms == capture.t_eot_ms, (
        "the flush must carry the t_eot the marker was asked for (criterion 4), got "
        f"{capture.flushed[0].t_eot_ms} for {capture.t_eot_ms}"
    )
