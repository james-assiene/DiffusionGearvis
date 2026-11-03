"""The transport's own tap: both wire directions, recorded by the code that moves the bytes.

T-0043 group A criterion 7. ``HudConnection.receive`` writes one ``wire.in`` per inbound frame —
*including* the ones it refuses, which are the interesting ones — and ``HudServer.publish`` writes
one ``wire.out`` per published message, whether or not a page is attached. The tap neither
consumes nor mutates: a frame still reaches ``BrowserFrameSource`` byte-identically.

The decode outcomes are the production codec's own branches (``tests/unit/test_hud_ingress.py``
is the module that pins them); this one asserts that each of them leaves a line in the log.

Regime: no socket (``server.open()``), ``ScriptedClock``, ``tests/fakes/``, no measurement.
"""

from __future__ import annotations

import base64
import json
import tracemalloc
from pathlib import Path
from typing import Any

from local_astra.contracts.codec import MAX_MESSAGE_BYTES, encode
from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.protocol import (
    AstraAssistToggle,
    AstraHello,
    AstraSpoken,
    AstraText,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from tests.unit._hud_rig import Recorder, astra_frame, require_symbol, running
from tests.unit._session_log_rig import kinds, make_log, of_kind, read_records, scan_for_pixels

#: Criterion 7's closed set for ``wire.in``'s ``outcome``.
OUTCOMES = {"astra", "unknown", "realtime", "malformed"}

#: A record must stay small whatever arrived: 4 KiB is two orders of magnitude under the 8 MiB
#: message the adversarial row sends, and well over any legitimate envelope.
RECORD_CAP_BYTES = 4096

#: What the tap may retain for the near-limit frame, over what the same frame costs a session
#: with no log: an envelope, a byte count and an outcome. 64 KiB is two orders of magnitude over
#: the record (433 bytes, measured) and two under the message.
TAP_RETAINED_CAP_BYTES = 64 * 1024

#: The marker the oversized frame's body starts with: a record that quotes any of the body quotes
#: this.
MARKER = "MARKERMARKER"

#: Built once, at import, so that no traced window below pays for the fixture itself. One byte
#: over ``MAX_MESSAGE_BYTES`` is what makes the codec refuse it; the tap must still record that it
#: arrived.
OVERSIZED = (
    '{"type":"astra.text","text":"' + MARKER + "x" * (MAX_MESSAGE_BYTES - len(MARKER)) + '"}'
)


class RecordingSink:
    def __init__(self) -> None:
        self.ingested: list[Any] = []

    def ingest(self, message: Any) -> Any:
        self.ingested.append(message)
        return None


def serve(rig: Any, sink: Any = None, **kwargs: Any) -> Any:
    """A ``HudServer`` over the rig's session with **no socket**: nothing is bound."""
    hud_server = require_symbol("HudServer")
    return hud_server(
        rig.session, frame_sink=sink if sink is not None else RecordingSink(), port=0, **kwargs
    )


async def test_every_inbound_frame_is_recorded_whatever_the_decoder_made_of_it(tmp_path):
    """Criterion 7: one ``wire.in`` per frame, for all five outcomes plus no_hello.

    The refused frames are the reason this tap exists: a page whose messages are all being
    dropped as ``no_hello`` or ``wrong_direction`` looks, from the session's side, exactly like a
    page that is sending nothing.
    """
    log = make_log(tmp_path / "logs")
    frames: list[str] = []
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig)
        conn = server.open()
        frames.append(encode(AstraHello()))
        frames.append(encode(AstraAssistToggle(flags=AssistFlags(page_text=True))))
        frames.append(encode(AstraSpoken(turn_id="t1", utterance_id="u1", text="hi")))
        frames.append('{"type":"astra.not.a.real.type","x":1}')
        frames.append('{"type":"session.update","session":{}}')
        frames.append("{this is not json")
        for raw in frames:
            conn.receive(raw)
        second = server.open()
        before_hello = encode(AstraAssistToggle(flags=AssistFlags(cells=True)))
        frames.append(before_hello)
        second.receive(before_hello)

    records = read_records(log.path)
    inbound = of_kind(records, "wire.in")
    assert len(inbound) == len(frames), (
        f"criterion 7: one wire.in per inbound frame, including the refused ones; {len(frames)} "
        f"frames arrived and the log holds {len(inbound)} (kinds={sorted(set(kinds(records)))})"
    )
    assert [r["data"]["n_bytes"] for r in inbound] == [len(f.encode()) for f in frames], (
        f"…each with the byte count that came off the wire; the log says "
        f"{[r['data']['n_bytes'] for r in inbound]}"
    )
    outcomes = [r["data"]["outcome"] for r in inbound]
    assert set(outcomes) <= OUTCOMES, (
        f"criterion 7 fixes the outcome vocabulary to {sorted(OUTCOMES)}; the log says {outcomes}"
    )
    assert outcomes[0] == "astra" and outcomes[1] == "astra" and outcomes[2] == "astra", (
        f"a hello, a routed toggle and a wrong-direction message all decoded as astra messages; "
        f"the log says {outcomes}"
    )
    assert outcomes[3] == "unknown", f"an unknown astra.* type; the log says {outcomes}"
    assert outcomes[4] == "realtime", f"a plain Realtime event; the log says {outcomes}"
    assert outcomes[5] == "malformed", f"bad JSON; the log says {outcomes}"
    assert outcomes[6] == "astra", (
        f"…and a well-formed message refused for arriving before the hello is still a frame that "
        f"arrived; the log says {outcomes}"
    )
    assert "astra.assist.toggle" in json.dumps(inbound[1]["data"]["msg"]), (
        f"criterion 7: the record carries the wire-form dump; it says {inbound[1]['data']!r:.300}"
    )
    assert isinstance(inbound[5]["data"]["msg"], str) and inbound[5]["data"]["msg"], (
        f"…or the decode error, for a frame that never became a message; it says "
        f"{inbound[5]['data']!r:.300}"
    )


async def test_the_byte_count_counts_bytes_and_not_characters(tmp_path):
    """Criterion 7: ``n_bytes`` is ``len(raw)`` *in bytes* — the wire's unit, not Python's.

    Every other fixture in this module is ASCII, where the two are the same number, so a tap that
    counted characters would pass all of them. It matters twice over: the byte count is what one
    compares against ``MAX_MESSAGE_BYTES`` when reading a log to explain a refused frame, and the
    codec's own limit counts UTF-8 bytes (``contracts/codec.decode``), so a log in characters
    would say a message was well under a limit it was over.
    """
    spoken = "météo — 数字 🙂 ½"
    log = make_log(tmp_path / "logs")
    inbound_raw = encode(AstraText(text=spoken))
    outbound = AstraSpoken(turn_id="t1", utterance_id="u1", text=spoken)
    assert len(inbound_raw) < len(inbound_raw.encode("utf-8")), (
        f"the fixture is multi-byte at all: {len(inbound_raw)} characters, "
        f"{len(inbound_raw.encode('utf-8'))} bytes"
    )
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig)
        conn = server.open()
        conn.receive(encode(AstraHello()))
        conn.receive(inbound_raw)
        server.publish(outbound)

    records = read_records(log.path)
    mine = [
        r
        for r in of_kind(records, "wire.in")
        if spoken in json.dumps(r["data"], ensure_ascii=False)
    ]
    assert len(mine) == 1, (
        f"the multi-byte frame was recorded once, with its text; the log holds "
        f"{[r['data'] for r in of_kind(records, 'wire.in')]!r:.400}"
    )
    assert mine[0]["data"]["n_bytes"] == len(inbound_raw.encode("utf-8")), (
        f"criterion 7: n_bytes is the frame's size on the wire "
        f"({len(inbound_raw.encode('utf-8'))} bytes); the record says "
        f"{mine[0]['data']['n_bytes']} — and {len(inbound_raw)} would be its length in "
        f"characters, which is what a tap that wrote len(raw) for a str reports"
    )
    out = [
        r
        for r in of_kind(records, "wire.out")
        if spoken in json.dumps(r["data"], ensure_ascii=False)
    ]
    assert len(out) == 1 and out[0]["data"]["n_bytes"] == len(encode(outbound).encode("utf-8")), (
        f"…and the same on the way out ({len(encode(outbound).encode('utf-8'))} bytes); the log "
        f"holds {[r['data'].get('n_bytes') for r in out]}"
    )


async def test_publish_is_recorded_even_when_no_page_is_attached(tmp_path):
    """Criterion 7: one ``wire.out`` per publish, connections or not.

    A session run with no page open (a rung, a headless evening) still logs what it would have
    shown — otherwise the log of the run one most wants to analyse is the emptiest.
    """
    log = make_log(tmp_path / "logs")
    message = AstraSpoken(turn_id="t-nobody", utterance_id="u1", text="nobody is listening")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig)
        assert not server.open().sent(), "a fresh connection has been sent nothing"
        server.publish(message)

    records = read_records(log.path)
    mine = [
        r
        for r in of_kind(records, "wire.out")
        if json.dumps(r["data"].get("msg")).find("nobody is listening") >= 0
    ]
    assert len(mine) == 1, (
        f"criterion 7: one wire.out per published message; the log holds {len(mine)} for this one "
        f"({[r['data'] for r in of_kind(records, 'wire.out')]!r:.400})"
    )
    assert mine[0]["data"]["n_bytes"] == len(encode(message).encode()), (
        f"…with the bytes it would put on the wire ({len(encode(message).encode())}); the record "
        f"says {mine[0]['data']['n_bytes']}"
    )
    assert mine[0]["data"].get("dropped") is not True, (
        f"…and it was not refused by anybody; the record says {mine[0]['data']!r:.300}"
    )


async def test_a_message_an_outbox_refused_is_recorded_as_dropped(tmp_path):
    """Criterion 7: an outbox refusal is recorded as a drop, never as a send.

    A one-message outbox and two spoken lines: the second cannot be dropped by the outbox's
    latest-wins rule (only ``astra.turn.state`` may be), so the connection is closed and the line
    never reaches the page. The log must not claim it was sent.
    """
    log = make_log(tmp_path / "logs")
    first = AstraSpoken(turn_id="t1", utterance_id="u1", text="the first line")
    second = AstraSpoken(turn_id="t1", utterance_id="u2", text="the second line")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig, outbox_size=1)
        conn = server.open()
        server.publish(first)
        server.publish(second)
        sent = [getattr(m, "text", None) for m in conn.sent()]

    assert "the second line" not in sent, (
        f"the rig itself must produce a refusal, or this row is vacuous; the connection holds "
        f"{sent}"
    )
    records = read_records(log.path)
    outbound = {
        r["data"]["msg"]["text"]: r["data"]
        for r in of_kind(records, "wire.out")
        if isinstance(r["data"].get("msg"), dict) and "text" in r["data"]["msg"]
    }
    assert outbound["the first line"].get("dropped") is not True, (
        f"the accepted message is recorded as sent; it says {outbound['the first line']!r}"
    )
    assert outbound["the second line"].get("dropped") is True, (
        f"criterion 7: the refused one carries dropped=true; it says "
        f"{outbound['the second line']!r}"
    )


async def test_the_tap_neither_consumes_nor_mutates_a_frame(tmp_path):
    """Criterion 7: frames still reach ``BrowserFrameSource`` byte-identically.

    The production shape: one ``BrowserFrameSource`` is both the session's perception client and
    the server's frame sink. The pixels the session sees after the tap must be the pixels the
    page posted — and the *log* must hold none of them (criterion 5).
    """
    from local_astra.perception.browser_source import BrowserFrameSource

    log = make_log(tmp_path / "logs")
    message = astra_frame("a")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        source = BrowserFrameSource(now_ms=lambda: rig.clock.now_ms)
        server = serve(rig, source)
        conn = server.open()
        conn.receive(encode(AstraHello()))
        conn.receive(encode(message))
        frame = await source.get_frame(message.source, None)

    assert frame is not None, "the frame reached the source at all"
    assert frame.jpeg == base64.b64decode(message.jpeg_b64), (
        "criterion 7: the tap neither consumes nor mutates — the source sees the posted pixels"
    )
    assert frame.phash == message.phash and frame.title == message.title, (
        f"…unchanged in every field; got {frame!r:.200}"
    )
    assert not scan_for_pixels(log.path), (
        f"criterion 5: and the log that recorded the frame holds no pixels; it contains "
        f"{scan_for_pixels(log.path)}"
    )
    inbound = of_kind(read_records(log.path), "wire.in")
    assert any(r["data"]["n_bytes"] == len(encode(message).encode()) for r in inbound), (
        f"…while still recording how big the frame was; the log says "
        f"{[r['data']['n_bytes'] for r in inbound]}"
    )


async def receive_oversized(tmp_path: Path, *, logging: bool) -> tuple[int, int, Any]:
    """Receive ``OVERSIZED`` once, traced. ``(retained, peak, log|None)``.

    ``OVERSIZED`` is built once at import, *outside* every traced window, so what
    ``tracemalloc`` counts here is what receiving it allocated — not the fixture. The two arms
    differ in one thing only: whether the session has a log for the tap to record into.
    """
    log = make_log(tmp_path / "logs") if logging else None
    kwargs: dict[str, Any] = {"session_log": log} if log is not None else {}
    async with running(tmp_path / "data", publish=Recorder(), **kwargs) as rig:
        server = serve(rig)
        conn = server.open()
        conn.receive(encode(AstraHello()))
        tracemalloc.start()
        conn.receive(OVERSIZED)
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return retained, peak, log


async def test_an_eight_mebibyte_message_contributes_a_byte_count_and_nothing_else(tmp_path):
    """Criterion 7, adversarial: the near-limit message the tap must not copy into the log.

    ``MAX_MESSAGE_BYTES`` is 8 MiB. A record that carried the body would put a second copy of it
    in the queue and a third on disk — the bounded-memory failure mode of criterion 3, arriving
    from the network rather than from the pump.

    Two measurements, because one of them can only be loose. ``contracts/codec.decode`` — which
    this task does not own and does not touch — evaluates ``len(raw.encode("utf-8"))`` for any
    ``str`` longer than a quarter of the limit, to count UTF-8 bytes rather than code points; that
    guard materialises one full copy of the message before any tap runs, so the *absolute* peak
    for this frame starts just over ``len(raw)`` whatever the tap does (8 388 700 bytes against a
    message of 8 388 639, measured on this box with no T-0043 code on the path). The bound that
    can be asserted absolutely is therefore the task's own wording — not the message size
    **twice** — and the sharp measurement is the differential below, which subtracts the codec and
    leaves the tap.
    """
    with_log_retained, with_log_peak, log = await receive_oversized(tmp_path / "on", logging=True)
    without_retained, without_peak, _ = await receive_oversized(tmp_path / "off", logging=False)
    assert log is not None
    raw = OVERSIZED

    assert with_log_peak < 2 * len(raw), (
        f"criterion 7: an oversized message is measured, not copied; receiving {len(raw)} bytes "
        f"peaked at {with_log_peak} bytes of new allocation, which is more than the message twice "
        f"— a record that holds the body alongside the codec's own size guard costs exactly that"
    )
    tap_cost = with_log_retained - without_retained
    assert tap_cost < TAP_RETAINED_CAP_BYTES, (
        f"criterion 7, differentially: receiving the same {len(raw)}-byte frame with a log "
        f"attached retained {with_log_retained} bytes against {without_retained} without one — "
        f"the tap itself kept {tap_cost} bytes, over the {TAP_RETAINED_CAP_BYTES} the envelope "
        f"and the byte count are worth. A body carried through the queue looks exactly like this "
        f"(peaks: {with_log_peak} with, {without_peak} without)"
    )
    lines = Path(log.path).read_text(encoding="utf-8").splitlines()
    oversized = [
        r
        for r in of_kind(read_records(log.path), "wire.in")
        if r["data"]["n_bytes"] == len(raw.encode())
    ]
    assert len(oversized) == 1, (
        f"the oversized frame is recorded once, with its size; the log holds "
        f"{[r['data']['n_bytes'] for r in of_kind(read_records(log.path), 'wire.in')]}"
    )
    assert oversized[0]["data"]["outcome"] == "malformed", (
        f"…and the outcome the codec gave it; the record says {oversized[0]['data']!r:.300}"
    )
    assert MARKER not in json.dumps(oversized[0]), (
        "criterion 7: never its body — not even the first bytes of it"
    )
    assert all(len(line) < RECORD_CAP_BYTES for line in lines), (
        f"…and no line of the log is over {RECORD_CAP_BYTES} bytes; the longest is "
        f"{max(len(line) for line in lines)}"
    )


async def test_a_session_with_no_log_records_nothing_and_still_serves(tmp_path):
    """Criterion 24, negative: with no log injected the transport behaves exactly as before.

    The tap is conditional on a log existing; a server over a session that is not logging must
    neither fail nor create a file anywhere.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    async with running(tmp_path / "data", publish=Recorder()) as rig:
        server = serve(rig)
        conn = server.open()
        conn.receive(encode(AstraHello()))
        conn.receive(encode(AstraTurnState(state=TurnState.IDLE, turn_id="t1")))
        server.publish(AstraSpoken(turn_id="t1", utterance_id="u1", text="hello"))
        assert getattr(rig.session, "session_log", "<no attribute>") is None, (
            "criterion 14: a session built without a log has none, and says so "
            "(Session.session_log is the attribute the transport's tap reads)"
        )
    assert not list(log_dir.iterdir()), (
        f"…and nothing was written anywhere; the directory holds {list(log_dir.iterdir())}"
    )


# --- T-0128c A7: the compact canvas row ---------------------------------------------------------
#
# Finding 3: one full ``wire.out`` per canvas frame is ~3.7 KB at ~25 Hz (~5 MB per minute of
# thinking). A7: the ``wire.out`` of an ``astra.canvas.step`` carries, under ``data["msg"]``, exactly
# ``COMPACT_KEYS``; every other type's ``wire.out`` is unchanged. Values by hand.

COMPACT_KEYS = {
    "type",
    "request_id",
    "canvas_idx",
    "step",
    "max_steps",
    "phase",
    "canvas_length",
    "n_committed",
    "n_kept",
    "n_stable",
}


async def _wire_out_of(tmp_path: Path, message: Any) -> dict[str, Any]:
    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        serve(rig).publish(message)
    out = of_kind(read_records(log.path), "wire.out")
    assert len(out) == 1, out
    return out[0]["data"]


async def test_a_canvas_steps_wire_out_is_the_compact_row(tmp_path):
    """A7: CL 8, canvas 1..8, argmax equal at p0,1,2,6 (4 positions), stable bits at p3,p4 = 0x18
    (2 positions, both where canvas != argmax, so "kept" counts 4 whether or not it excludes
    stable positions -- see the report)."""
    from local_astra.contracts.protocol import AstraCanvasStep

    frame = AstraCanvasStep(
        request_id="chatcmpl-x-1234abcd",
        canvas_idx=2,
        step=7,
        max_steps=48,
        canvas_length=8,
        canvas=(1, 2, 3, 4, 5, 6, 7, 8),
        argmax=(1, 2, 3, 40, 50, 60, 7, 80),
        stable="18",
        phase="denoise",
    )
    data = await _wire_out_of(tmp_path, frame)
    assert data["msg"] == {
        "type": "astra.canvas.step",
        "request_id": "chatcmpl-x-1234abcd",
        "canvas_idx": 2,
        "step": 7,
        "max_steps": 48,
        "phase": "denoise",
        "canvas_length": 8,
        "n_committed": 0,
        "n_kept": 4,
        "n_stable": 2,
    }, f"A7: the compact row, exactly: {data['msg']}"


async def test_a_real_cl256_frame_logs_under_a_kilobyte_and_no_rows(tmp_path):
    """A7: a recorded CL 256 frame (806da12 trajectory) -> no ``canvas``/``argmax``/``stable`` and
    the record under 1 KiB; a commit frame's ``n_committed`` is its id count."""
    from local_astra.contracts.protocol import AstraCanvasStep
    from tests.fakes.canvas_tap import load_trajectory

    frames = load_trajectory()["frames"]
    for frame in (frames[7], frames[8]):  # a denoise frame, then canvas 0's commit
        model = AstraCanvasStep.model_validate(frame)
        data = await _wire_out_of(tmp_path / frame["phase"], model)
        assert set(data["msg"]) == COMPACT_KEYS, sorted(data["msg"])
        assert len(json.dumps(data)) < 1024, len(json.dumps(data))
        if frame["phase"] == "commit":
            assert data["msg"]["n_committed"] == len(frame["committed_tokens"])


async def test_other_types_wire_out_is_unchanged_by_the_compact_canvas_row(tmp_path):
    """A7 (regression, holds today): non-canvas messages keep ``msg = model_dump(mode="json")``."""
    for message in (
        AstraSpoken(turn_id="t1", utterance_id="u1", text="hello"),
        AstraTurnState(state=TurnState.THINKING, turn_id="t1"),
    ):
        data = await _wire_out_of(tmp_path / message.type, message)
        assert data["msg"] == message.model_dump(mode="json"), data


async def test_a_canvas_frame_its_own_outbox_dropped_is_logged_dropped(tmp_path):
    """O-A3 (4) / T-0043 criterion 7: an outbox of 3 full of spoken lines; a canvas step arrives
    and the outbox drops it (a canvas message never costs the connection, A6). The page never gets
    it, so its ``wire.out`` says ``dropped: true`` -- the transport accepted it without closing,
    which is not the same as sending it. The spoken lines before it stay ``dropped`` false.

    Only the drop-on-arrival case is pinned: a queued frame later replaced or evicted already has
    its ``wire.out`` (see the report)."""
    from local_astra.contracts.protocol import AstraCanvasStep
    from tests.fakes.canvas_tap import denoise

    log = make_log(tmp_path / "logs")
    frame = AstraCanvasStep.model_validate(denoise("A", 0, 1))
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig, outbox_size=3)
        conn = server.open()
        for i in range(3):
            server.publish(AstraSpoken(turn_id="t1", utterance_id=f"u{i}", text=f"line {i}"))
        server.publish(frame)
        held = [m.type for m in conn.sent()]
        closed = conn.closed
    assert not closed and "astra.canvas.step" not in held, (
        f"premise: the outbox dropped the frame and kept the connection ({held}, closed={closed})"
    )
    out = of_kind(read_records(log.path), "wire.out")
    canvas = [r["data"] for r in out if r["data"]["msg"].get("type") == "astra.canvas.step"]
    lines = [r["data"] for r in out if r["data"]["msg"].get("type") == "astra.spoken"]
    assert len(canvas) == 1 and canvas[0]["dropped"] is True, (
        f"O-A3 (4): the frame its outbox dropped is logged dropped=true: {canvas}"
    )
    assert [d["dropped"] for d in lines] == [False, False, False], lines


async def test_a_canvas_frame_delivered_after_a_dropped_one_is_logged_not_dropped(tmp_path):
    """Reviewer 3 (1): outbox of 3 full of spoken lines; frame s1 is dropped (``dropped: true``);
    the queue drains (the writer sent it all); frame s2 is then queued -> its ``wire.out`` says
    ``dropped: false``. A "lost" flag that outlives its own offer would mark s2 dropped too."""
    from local_astra.contracts.protocol import AstraCanvasStep
    from tests.fakes.canvas_tap import denoise

    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig, outbox_size=3)
        conn = server.open()
        for i in range(3):
            server.publish(AstraSpoken(turn_id="t1", utterance_id=f"u{i}", text=f"line {i}"))
        server.publish(AstraCanvasStep.model_validate(denoise("A", 0, 1)))
        conn.outbox.drain()
        server.publish(AstraCanvasStep.model_validate(denoise("A", 0, 2)))
        held = [(m.type, getattr(m, "step", None)) for m in conn.outbox.pending()]
    assert held == [("astra.canvas.step", 2)], f"premise: s2 is queued after the drain: {held}"
    out = of_kind(read_records(log.path), "wire.out")
    canvas = {
        r["data"]["msg"]["step"]: r["data"]["dropped"]
        for r in out
        if r["data"]["msg"].get("type") == "astra.canvas.step"
    }
    assert canvas == {1: True, 2: False}, f"reviewer 3 (1): {canvas}"


async def test_a_denoise_dropped_at_the_canvas_cap_is_logged_dropped(tmp_path):
    """Reviewer 3 (3), the log half: outbox of 64 holding 8 commits (the canvas cap); a denoise
    frame arrives and is dropped itself -> its ``wire.out`` says ``dropped: true``."""
    from local_astra.contracts.protocol import AstraCanvasStep
    from tests.fakes.canvas_tap import commit, denoise

    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = serve(rig, outbox_size=64)
        server.open()
        for c in range(8):
            server.publish(AstraCanvasStep.model_validate(commit("A", c, 5)))
        server.publish(AstraCanvasStep.model_validate(denoise("B", 0, 1)))
    out = of_kind(read_records(log.path), "wire.out")
    rows = [r["data"] for r in out if r["data"]["msg"].get("type") == "astra.canvas.step"]
    assert [d["msg"]["phase"] for d in rows] == ["commit"] * 8 + ["denoise"], rows
    assert [d["dropped"] for d in rows] == [False] * 8 + [True], (
        f"reviewer 3 (3): the denoise dropped at the cap is logged dropped: "
        f"{[d['dropped'] for d in rows]}"
    )


async def test_the_compact_row_reads_stable_msb_first_and_kept_excludes_stable(tmp_path):
    """Reviewer 2 T2 (kills M7 LSB-first and M8 kept-includes-stable): CL 8, canvas == argmax at
    p0, p1, p5; stable = "c0" = 1100 0000, MSB first = positions 0 and 1.

    By hand: n_stable 2; n_kept = {0, 1, 5} minus stable {0, 1} = 1. LSB-first would read stable
    {6, 7} and n_kept 3; kept-including-stable would read n_kept 3."""
    from local_astra.contracts.protocol import AstraCanvasStep

    frame = AstraCanvasStep(
        request_id="r",
        canvas_idx=0,
        step=3,
        max_steps=48,
        canvas_length=8,
        canvas=(1, 2, 3, 4, 5, 6, 7, 8),
        argmax=(1, 2, 30, 40, 50, 6, 70, 80),
        stable="c0",
        phase="denoise",
    )
    data = await _wire_out_of(tmp_path, frame)
    assert (data["msg"]["n_stable"], data["msg"]["n_kept"]) == (2, 1), (
        f"T2: stable is MSB first and kept excludes stable: {data['msg']}"
    )
