"""T-0216b: a received frame never drops the connection or kills the session log.

- F1 (criterion 1): ``HudConnection.receive`` on a Realtime-typed frame nested in pydantic-core's
  serializer window (~255-992 deep) does not raise, the connection stays open and keeps serving,
  the frame is handled as before (``relay_ignored``), and its ``wire.in`` row is still written
  without the payload text.
- F2 (criterion 2): a frame carrying a lone surrogate is written as one valid UTF-8 JSON line with
  the surrogate escaped; the writer survives, later records reach disk, ``aclose`` does not raise.
- Negative (criterion 3): a record that serializes today is written byte-identically to main's
  serialization (``json.dumps(record, separators=(",", ":"), ensure_ascii=False)``, keys in
  ``seq, t_ms, wall, session_ms, kind, turn_id, data`` order).

Depth stays <= 990: the codec on main (before T-0216 merges) still raises on 100 000-deep frames,
which is T-0216's test, not this one.

Regime: no socket (``server.open()``), the T-0028 rig over ``tests/fakes/``, a real ``SessionLog``
in a tmp dir; no measurement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import AstraHello
from tests.unit._hud_rig import Recorder, require_symbol, running
from tests.unit._session_log_rig import StepClock, make_log, of_kind, read_records, settle


class RecordingSink:
    def __init__(self) -> None:
        self.ingested: list[Any] = []

    def ingest(self, message: Any) -> Any:
        self.ingested.append(message)
        return None


def _nested(shape: str, depth: int) -> str:
    if shape == "list":
        return '{"type":"x","a":' + "[" * depth + "]" * depth + "}"
    return '{"type":"x","a":' + '{"b":' * depth + "1" + "}" * depth + "}"


def _strict_lines(path: Path) -> list[str]:
    """Every line of the file, each checked to be strict UTF-8 and one JSON object."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")  # strict: an encoded surrogate (surrogatepass) fails here
    except UnicodeDecodeError as exc:
        raise AssertionError(f"criterion 2: the log is not valid UTF-8: {exc}") from None
    lines = text.splitlines()
    for n, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"line {n} is not JSON: {exc}; {line[:200]!r}") from None
        assert isinstance(obj, dict), f"line {n} is not an object"
    return lines


# --- F1: the serializer window -------------------------------------------------------------------


@pytest.mark.parametrize("depth", [256, 500, 900])
@pytest.mark.parametrize("shape", ["list", "dict"])
async def test_realtime_frame_nested_in_the_serializer_window_keeps_the_connection(
    tmp_path, shape: str, depth: int
):
    raw = _nested(shape, depth)
    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = require_symbol("HudServer")(rig.session, frame_sink=RecordingSink(), port=0)
        conn = server.open()
        before = dict(server.counts)
        try:
            conn.receive(raw)
        except Exception as e:
            raise AssertionError(
                f"criterion 1: receive is total but raised {type(e).__name__}: {str(e)[:120]}"
            ) from None

        assert not conn.closed and conn.close_reason is None, (
            f"criterion 1: the connection stays open; close_reason={conn.close_reason!r}"
        )
        # Handled as it is today: a Realtime-typed frame is counted relay_ignored, nothing else.
        delta = {k: server.counts[k] - before.get(k, 0) for k in server.counts}
        assert {k: v for k, v in delta.items() if v} == {"relay_ignored": 1}, (
            f"criterion 1: the frame is handled as today (one relay_ignored); delta={delta}"
        )
        # Still serving: a hello after the deep frame is answered.
        n_sent = len(conn.sent())
        try:
            conn.receive(encode(AstraHello()))
        except Exception as e:
            raise AssertionError(
                f"criterion 1: the hello after the deep frame raised {type(e).__name__}"
            ) from None
        assert conn.greeted and len(conn.sent()) > n_sent, (
            "criterion 1: after the deep frame the connection still takes a hello"
        )

    lines = _strict_lines(log.path)
    inbound = of_kind(read_records(log.path), "wire.in")
    assert len(inbound) == 2, (
        f"criterion 1: one wire.in per frame (deep frame + hello); got {len(inbound)}: "
        f"{[r['data'].get('outcome') for r in inbound]}"
    )
    row = inbound[0]["data"]
    assert row["outcome"] == "realtime", f"criterion 1: the deep frame's row says {row!r:.300}"
    assert row["n_bytes"] == len(raw.encode()), f"n_bytes {row['n_bytes']} != {len(raw)}"
    # T-0216b review T2 (coordinator ruling): the fallback row is the frame's type alone.
    assert row["msg"] == {"type": "x", "unserialisable": True}, (
        f"criterion 1: the fallback wire.in body; got {row['msg']!r:.300}"
    )
    assert inbound[1]["data"]["outcome"] == "astra", "the hello row follows"
    # No payload text: neither a bracket run nor a repeated ``{"b":`` chain, anywhere in the file.
    text = "\n".join(lines)
    assert not re.search(r"[\[\]{}]{50}", text), "criterion 1: a bracket run of the payload"
    # Any quoting (JSON, escaped JSON, Python repr): twenty ``{ b :`` links in a row is the payload.
    assert not re.search(r"(?:\{[^{}\w]{0,4}b[^{}\w]{0,4}:\s*){20}", text), (
        "criterion 1: the nested-dict payload text"
    )
    assert not re.search(r"\d{50}", text), "criterion 1: a long digit run"
    assert of_kind(read_records(log.path), "log.stopped"), "the log closed cleanly"


def _deep_ui_action(shape: str, depth: int) -> str:
    """``astra.ui.action`` whose ``context`` (``dict[str, Any]``, contracts/protocol.py) is deep."""
    if shape == "list":
        inner = '{"a":' + "[" * depth + "]" * depth + "}"
    else:
        inner = '{"b":' * depth + "1" + "}" * depth
    return '{"type":"astra.ui.action","surface_id":"s1","name":"press","context":' + inner + "}"


@pytest.mark.parametrize("depth", [256, 900])
@pytest.mark.parametrize("shape", ["list", "dict"])
async def test_astra_frame_nested_in_the_serializer_window_keeps_the_connection(
    tmp_path, shape: str, depth: int
):
    """T-0216b review T1: the AstraMessage branch of the wire.in tap, not only the Realtime one.

    Sent before the hello, so it is handled as today without reaching the session: one
    ``no_hello`` and one ``astra.error(code="no_hello")``.
    """
    raw = _deep_ui_action(shape, depth)
    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = require_symbol("HudServer")(rig.session, frame_sink=RecordingSink(), port=0)
        conn = server.open()
        before = dict(server.counts)
        try:
            conn.receive(raw)
        except Exception as e:
            raise AssertionError(
                f"criterion 1: receive is total but raised {type(e).__name__}: {str(e)[:120]}"
            ) from None
        assert not conn.closed and conn.close_reason is None, (
            f"criterion 1: the connection stays open; close_reason={conn.close_reason!r}"
        )
        delta = {k: server.counts[k] - before.get(k, 0) for k in server.counts}
        assert {k: v for k, v in delta.items() if v} == {"no_hello": 1}, (
            f"the ungreeted ui.action is handled as today (one no_hello); delta={delta}"
        )
        n_sent = len(conn.sent())
        conn.receive(encode(AstraHello()))
        assert conn.greeted and len(conn.sent()) > n_sent, "a later hello is answered"

    lines = _strict_lines(log.path)
    inbound = of_kind(read_records(log.path), "wire.in")
    assert len(inbound) == 2, f"one wire.in per frame; got {len(inbound)}"
    row = inbound[0]["data"]
    assert row["outcome"] == "astra" and row["msg"]["type"] == "astra.ui.action", (
        f"criterion 1: the deep frame's row says {row!r:.300}"
    )
    assert row["n_bytes"] == len(raw.encode())
    text = "\n".join(lines)
    assert not re.search(r"[\[\]{}]{50}", text), "criterion 1: a bracket run of the payload"
    assert not re.search(r"(?:\{[^{}\w]{0,4}b[^{}\w]{0,4}:\s*){20}", text), (
        "criterion 1: the nested-dict payload text"
    )


# --- F2: lone surrogates -------------------------------------------------------------------------

SURROGATE_FRAMES = {
    "realtime": '{"type":"x","a":"\\ud800"}',
    "astra_text": '{"type":"astra.text","text":"a\\ud800"}',
}


@pytest.mark.parametrize("variant", sorted(SURROGATE_FRAMES))
async def test_lone_surrogate_frame_is_logged_and_the_writer_survives(tmp_path, variant: str):
    raw = SURROGATE_FRAMES[variant]
    log = make_log(tmp_path / "logs")
    try:
        async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
            server = require_symbol("HudServer")(rig.session, frame_sink=RecordingSink(), port=0)
            conn = server.open()
            conn.receive(encode(AstraHello()))
            conn.receive(raw)
            await settle(log)
            for i in range(3):
                conn.receive('{"type":"x","b":' + str(i) + "}")
            await settle(log)
            assert not conn.closed, "the connection stays open"
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(
            f"criterion 2: the session (aclose) raised {type(e).__name__}: {str(e)[:160]}"
        ) from None

    lines = _strict_lines(log.path)
    records = read_records(log.path)
    inbound = of_kind(records, "wire.in")
    outcomes = [r["data"].get("outcome") for r in inbound]
    assert len(inbound) == 5, f"criterion 2: hello + surrogate + 3 later frames; got {outcomes}"
    # T-0216c re-pin: R-2. Since T-0216c the codec replaces a lone surrogate with U+FFFD at decode, so
    # the frame's row carries U+FFFD (not dropped), and no surrogate escape reaches disk at all. The
    # ``_emit`` escape path itself is pinned by ``test_lone_surrogate_record_is_written_escaped`` below.
    replaced_rows = [ln for ln in lines if '"wire.in"' in ln and "�" in ln]
    assert len(replaced_rows) == 1, (
        f"T-0216c re-pin: R-2: the surrogate frame's wire.in row is one line carrying U+FFFD; "
        f"found {len(replaced_rows)}"
    )
    assert "�" in json.dumps(inbound[1]["data"]["msg"], ensure_ascii=False), (
        f"T-0216c re-pin: R-2: the second wire.in row (the surrogate frame) carries U+FFFD; "
        f"got {inbound[1]['data']!r:.300}"
    )
    assert not any(re.search(r"\\ud[89a-f][0-9a-f]{2}", ln, re.I) for ln in lines), (
        "T-0216c re-pin: R-2: no lone-surrogate escape reaches the log once the codec replaces it"
    )
    # Every later frame reached disk, in order.
    later = [r["data"]["msg"] for r in inbound[2:]]
    # ``RealtimeEvent`` (contracts/protocol.py) is ``{type, payload}``; unchanged by this task.
    assert later == [{"type": "x", "payload": {"b": i}} for i in range(3)], (
        f"criterion 2: the three later frames reach disk in order; got {later!r:.300}"
    )
    assert records[-1]["kind"] == "log.stopped", "criterion 2: the writer ran to log.stopped"
    assert log.stats()["error"] is None, f"the log was disabled: {log.stats()['error']!r}"
    # T-0216b review T2: the byte count is what was written (the escaped line), log.stopped included.
    size = Path(log.path).stat().st_size
    assert log.stats()["bytes"] == size, (
        f"criterion 2: stats()['bytes'] {log.stats()['bytes']} != the file's {size} bytes"
    )


async def test_lone_surrogate_record_is_written_escaped(tmp_path):
    """T-0216c re-pin: R-2. A record handed to ``SessionLog.record`` directly with a lone surrogate
    (no codec on the way, so T-0216c's replacement does not apply) still takes ``_emit``'s escape
    path: one valid UTF-8 JSON line with the surrogate escaped, the writer alive for later records,
    and ``stats()['bytes']`` equal to the file size."""
    log = make_log(tmp_path / "logs", clock=StepClock())
    await log.start()
    try:
        log.record(
            "wire.in",
            {"n_bytes": 3, "outcome": "astra", "msg": {"type": "astra.text", "text": "a\ud800"}},
        )
        await settle(log)
        for i in range(3):
            log.record(
                "wire.in",
                {"n_bytes": 1, "outcome": "realtime", "msg": {"type": "x", "payload": {"b": i}}},
            )
        await settle(log)
    finally:
        await log.aclose()

    lines = _strict_lines(log.path)
    records = read_records(log.path)
    inbound = of_kind(records, "wire.in")
    assert len(inbound) == 4, (
        f"T-0216c re-pin: R-2: the surrogate record + 3 later; got {len(inbound)}"
    )
    escaped = [ln for ln in lines if '"wire.in"' in ln and re.search(r"\\ud800", ln, re.I)]
    assert len(escaped) == 1, (
        f"T-0216c re-pin: R-2: the direct record's line has \\ud800 escaped; found {len(escaped)}"
    )
    assert inbound[0]["data"]["msg"]["text"] == "a\ud800", (
        f"T-0216c re-pin: R-2: the escaped line round-trips to the record's text; "
        f"got {inbound[0]['data']!r:.200}"
    )
    later = [r["data"]["msg"] for r in inbound[1:]]
    assert later == [{"type": "x", "payload": {"b": i}} for i in range(3)], (
        f"T-0216c re-pin: R-2: the writer survives; later records reach disk in order; got {later!r:.300}"
    )
    assert records[-1]["kind"] == "log.stopped", (
        "T-0216c re-pin: R-2: the writer ran to log.stopped"
    )
    assert log.stats()["error"] is None, f"the log was disabled: {log.stats()['error']!r}"
    size = Path(log.path).stat().st_size
    assert log.stats()["bytes"] == size, (
        f"T-0216c re-pin: R-2: stats()['bytes'] {log.stats()['bytes']} != the file's {size} bytes"
    )


# --- criterion 3: byte identity --------------------------------------------------------------------

#: Representative records that serialize today: ASCII and non-ASCII text, nested containers,
#: numbers incl. the non-finite floats ``json.dumps`` writes by default, None/bools, empty data.
VALID = [
    ("wire.in", {"n_bytes": 12, "outcome": "astra", "msg": {"type": "astra.hello"}}),
    ("wire.in", {"n_bytes": 7, "outcome": "realtime", "msg": {"type": "x", "payload": {"b": 1}}}),
    ("wire.out", {"text": "café — 日本語 🙂", "quote": 'say "hi"\n\tthen\\stop'}),
    ("llm.commit", {"nested": [[1, [2, [3, {"k": [None, True, False]}]]]], "empty": {}}),
    ("machine.step", {"f": 1.5, "neg0": -0.0, "big": 1e300, "i": -(2**63), "nan": float("nan")}),
    ("tool.call", {"inf": float("inf"), "ninf": float("-inf"), "s": "  \x00\x1f"}),
    ("dropped", {}),
]


def _main_line(seq: int, parsed: dict[str, Any], kind: str, data: dict[str, Any]) -> str:
    """The line main writes: the record dict in its key order, dumped compact, non-ASCII raw."""
    record = {
        "seq": seq,
        "t_ms": parsed["t_ms"],
        "wall": parsed["wall"],
        "session_ms": 40 + seq,
        "kind": kind,
        "turn_id": f"t{seq}",
        "data": data,
    }
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)


async def test_valid_records_byte_identical(tmp_path):
    clock = StepClock(1000)
    log = make_log(tmp_path / "logs", clock=clock)
    await log.start()
    await settle(log)
    for n, (kind, data) in enumerate(VALID, start=1):
        clock.advance(5)
        log.record(kind, data, turn_id=f"t{n}", session_ms=40 + n)
    await settle(log)
    await log.aclose()

    raw_lines = Path(log.path).read_bytes().decode("utf-8").split("\n")
    assert raw_lines[-1] == "", "the file ends with a newline"
    body = raw_lines[1 : 1 + len(VALID)]  # skip log.started, stop before log.stopped
    assert len(body) == len(VALID), f"every record written; got {len(raw_lines) - 1} lines"
    for n, ((kind, data), line) in enumerate(zip(VALID, body, strict=True), start=1):
        parsed = json.loads(line)
        assert parsed["t_ms"] == 5 * n, f"t_ms from the injected clock; got {parsed['t_ms']}"
        expected = _main_line(n, parsed, kind, data)
        assert line == expected, (
            f"criterion 3: record {n} ({kind}) is not byte-identical to main's serialization:\n"
            f"  got      {line!r:.300}\n  expected {expected!r:.300}"
        )
