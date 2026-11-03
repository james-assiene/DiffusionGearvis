"""The session log's writer: the envelope, the bound, and the two ways it may not take a session down.

T-0043 group A, criteria 1-4 and 27. The seams are pinned in ``tests/unit/_session_log_rig.py``'s
docstring; nothing here re-implements the writer — the oracles are the task file's own vocabulary
(``CRITERION_2_KINDS``), a conservation law over offers, and an injected monotonic clock.

Regime for the timing rows: CPU only, no GPU resident, no service running, ``perf_counter_ns``
per call, the payload mix of docs/tasks/T-0043.md §6 (one ``astra.frame`` wire record with a
180 KiB JPEG, one ``machine.step`` with an ``SttStep``, one ``llm.commit`` with a 1.2 KiB canvas),
n as stated per row, seed n/a. A loaded box is the known confound; the failure message quotes
p50/p95/max so a flake can be told from a regression.
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.unit._session_log_rig import (
    CRITERION_2_KINDS,
    StepClock,
    container_sizes,
    jpeg,
    kinds,
    make_log,
    of_kind,
    pid,
    read_records,
    settle,
    symbol,
)

#: Criterion 27: the per-call ceiling on the pump, in milliseconds at the 95th percentile.
RECORD_P95_MS = 0.2


def mixed_payloads() -> list[tuple[str, dict[str, Any]]]:
    """The §6 payload mix: what a real turn actually offers, not three empty dicts."""
    from tests.unit._session_log_rig import b64

    return [
        (
            "wire.in",
            {
                "n_bytes": 246_000,
                "outcome": "astra",
                "msg": {
                    "type": "astra.frame",
                    "source": "screen",
                    "ts_ms": 12_000,
                    "width": 1280,
                    "height": 720,
                    "jpeg_b64": b64(jpeg(180 * 1024)),
                    "phash": "0f1e2d3c4b5a6978",
                },
            },
        ),
        (
            "machine.step",
            {
                "event": {"type": "stt.step", "t_ms": 12_040, "prs": [0.1, 0.2, 0.3, 0.4]},
                "state_before": "LISTENING",
                "state_after": "LISTENING",
                "commands": [],
            },
        ),
        (
            "llm.commit",
            {
                "request_id": "gen-1",
                "canvas_idx": 3,
                "segment": "answer",
                "text": "x" * 1200,
                "thought_open": False,
            },
        ),
    ]


def percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples_ms)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[int(len(ordered) * 0.95)]
    return p50, p95, ordered[-1]


async def test_the_envelope_is_the_same_on_every_record(tmp_path):
    """Criterion 2: seq, t_ms, wall, session_ms, kind, turn_id, data — on every line.

    ``seq`` is asserted to be exactly ``range(len(lines))``: "0-based, strictly increasing, no
    gaps" means the number is assigned when the record is *written*, which is what makes the
    ``dropped`` record of criterion 3 the only way a gap can be reported.
    """
    log = make_log(tmp_path)
    await log.start()
    log.record("note", {"n": 1})
    log.record("llm.commit", {"segment": "thought", "text": "hmm"}, turn_id="t1", session_ms=4200)
    await settle(log)
    await log.aclose()

    records = read_records(log.path)
    assert kinds(records)[0] == "log.started", (
        f"criterion 2: log.started is first; {kinds(records)}"
    )
    assert kinds(records)[-1] == "log.stopped", (
        f"criterion 2: log.stopped is last; {kinds(records)}"
    )
    assert [r["seq"] for r in records] == list(range(len(records))), (
        f"criterion 2: seq is 0-based, strictly increasing and gapless; got "
        f"{[r['seq'] for r in records]}"
    )
    for record in records:
        assert set(record) == {"seq", "t_ms", "wall", "session_ms", "kind", "turn_id", "data"}, (
            f"criterion 2 fixes the envelope; this record has {sorted(record)}"
        )
        assert isinstance(record["t_ms"], int), f"t_ms is an int; got {record['t_ms']!r}"
        stamped = datetime.fromisoformat(record["wall"])
        assert stamped.tzinfo is not None, (
            f"criterion 2: wall is ISO-8601 local time **with a UTC offset** — a timestamp "
            f"without one cannot be compared to anything; got {record['wall']!r}"
        )
    assert [r["t_ms"] for r in records] == sorted(r["t_ms"] for r in records), (
        f"criterion 2: t_ms is non-decreasing across the file; got {[r['t_ms'] for r in records]}"
    )
    commit = of_kind(records, "llm.commit")[0]
    assert commit["turn_id"] == "t1" and commit["session_ms"] == 4200, (
        f"criterion 2: turn_id and the session clock join a record to its turns.jsonl span; "
        f"got turn_id={commit['turn_id']!r} session_ms={commit['session_ms']!r}"
    )
    assert commit["data"] == {"segment": "thought", "text": "hmm"}, (
        f"…and the payload is the caller's, verbatim; got {commit['data']!r}"
    )
    note = of_kind(records, "note")[0]
    assert note["turn_id"] is None and note["session_ms"] is None, (
        f"criterion 2: unknown values are null, never invented; got {note!r}"
    )
    started = of_kind(records, "log.started")[0]
    assert started["data"].get("pid") == pid(), (
        f"criterion 17: log.started names the process that wrote the file; got {started['data']!r}"
    )
    stopped = of_kind(records, "log.stopped")[0]["data"]
    assert {"records", "dropped", "bytes"} <= set(stopped), (
        f"criterion 17: log.stopped carries the counters; got {stopped!r}"
    )
    assert stopped["dropped"] == 0 and stopped["records"] >= 3, (
        f"nothing overflowed and four records were offered; log.stopped says {stopped!r}"
    )
    size = Path(log.path).stat().st_size
    assert (
        abs(int(stopped["bytes"]) - size) <= len(Path(log.path).read_text().splitlines()[-1]) + 1
    ), (
        f"log.stopped's byte count is the file it is the last line of ({size} bytes on disk); "
        f"it says {stopped['bytes']}"
    )


async def test_t_ms_follows_the_injected_monotonic_clock(tmp_path):
    """Criterion 2: ``t_ms`` is milliseconds from a monotonic source, not a wall re-derivation.

    The writer is drained between the two offers, so the moment the record is offered and the
    moment it is written carry the same clock value — the assertion holds whichever of the two
    the implementation stamps, and a ``t_ms`` that is a constant, a counter or a wall-clock
    subtraction fails it.
    """
    clock = StepClock(1_000)
    log = make_log(tmp_path, clock=clock)
    await log.start()
    log.record("note", {"n": 1})
    await settle(log)
    clock.advance(250)
    log.record("note", {"n": 2})
    await settle(log)
    await log.aclose()

    notes = of_kind(read_records(log.path), "note")
    assert len(notes) == 2, f"both notes were written; got {kinds(read_records(log.path))}"
    assert notes[1]["t_ms"] - notes[0]["t_ms"] == 250, (
        f"criterion 2: 250 ms of the injected monotonic clock is 250 ms of t_ms; the file says "
        f"{notes[0]['t_ms']} then {notes[1]['t_ms']}"
    )


async def test_the_kind_vocabulary_is_closed_and_declared_once(tmp_path):
    """Criterion 2: ``KINDS`` is the task file's vocabulary, and nothing else is ever written.

    T-0043's sixteen **plus T-0054 criterion 5's ``speech.refused``**, which that task names as
    "the only new kind". The list lives in ``tests/unit/_session_log_rig.py``'s
    ``CRITERION_2_KINDS``, so this row moves with it by construction; T-0054's *Owned files*
    section names this pin so its red is read as the amendment it is and not as a regression
    (``.claude/rules/tests.md``, "Activation is three things in one commit").

    Why the set equality matters more than usual here: ``session_log.record()`` turns a kind
    outside ``KINDS`` into a complaining ``note`` (``session_log.py:405``) instead of raising, so
    a writer that emits ``speech.refused`` while ``KINDS`` does not list it loses the record
    *silently* — which is the exact failure shape T-0054 exists to end.
    """
    declared = symbol("KINDS")
    # T-0162 A-3 (pre-authorised pin move, amendment 2026-10-27): exactly one more kind, the
    # Restart's ``session.reset`` record.
    # T-0213 A2a-5 (named re-pin, A2a-1 commit): exactly one more kind, the applied Settings
    # change set's ``settings.change`` record.
    expected = {*CRITERION_2_KINDS, "session.reset", "settings.change"}
    assert set(declared) == expected, (
        "criterion 2: the closed vocabulary is declared in one place and is exactly the task "
        f"file's; missing={sorted(expected - set(declared))}, "
        f"extra={sorted(set(declared) - expected)}"
    )
    log = make_log(tmp_path)
    await log.start()
    log.record("wire.sideways", {"n": 1})
    log.record("note", {"n": 2})
    await settle(log)
    await log.aclose()

    written = read_records(log.path)
    assert all(r["kind"] in set(CRITERION_2_KINDS) for r in written), (
        f"criterion 2: a kind outside the vocabulary is a programming error, not a line; "
        f"the file holds {kinds(written)}"
    )
    assert of_kind(written, "note"), "…and the log kept working afterwards"


async def test_close_is_idempotent_and_a_second_log_takes_a_new_path(tmp_path):
    """Criterion 4 and criterion 17: closing twice writes one stop; reopening opens a new file.

    The second log is opened in the same directory in the same second of the same process, which
    is precisely when a ``<YYYYMMDD-HHMMSS>-<pid>`` name collides: two sessions must not share a
    file, or the second one's first record silently truncates the first one's evidence.
    """
    first = make_log(tmp_path)
    await first.start()
    first.record("note", {"which": "first"})
    await settle(first)
    await first.aclose()
    await first.aclose()

    records = read_records(first.path)
    assert kinds(records).count("log.stopped") == 1, (
        f"criterion 4: aclose is idempotent — one log.stopped, one drain; got {kinds(records)}"
    )

    second = make_log(tmp_path)
    await second.start()
    second.record("note", {"which": "second"})
    await settle(second)
    await second.aclose()

    assert Path(second.path) != Path(first.path), (
        f"criterion 17: a reopened log takes a new path; both are {first.path}"
    )
    assert of_kind(read_records(first.path), "note")[0]["data"] == {"which": "first"}, (
        "…and the first log's records are still in the first file"
    )


async def test_an_unserialisable_payload_becomes_an_error_record(tmp_path):
    """Criterion 3: ``record`` is total. A value that cannot be serialised names itself.

    The error record must name the kind that carried it *and* where in the payload it sat —
    "something failed to serialise" in a debug log is the same as nothing.
    """
    log = make_log(tmp_path)
    await log.start()
    log.record("llm.request", {"options": {"stop": object()}}, turn_id="t1")
    log.record("note", {"after": True})
    await settle(log)
    await log.aclose()

    records = read_records(log.path)
    errors = of_kind(records, "error")
    assert len(errors) == 1, (
        f"criterion 3: one error record for the record that could not be written; got "
        f"{kinds(records)}"
    )
    text = str(errors[0]["data"])
    assert "llm.request" in text, (
        f"criterion 3: the error record names the kind that carried the bad value; it says {text}"
    )
    assert "options" in text and "stop" in text, (
        f"criterion 3: …and the offending path inside the payload; it says {text}"
    )
    assert of_kind(records, "note"), "…and the log kept writing afterwards"
    assert [r["seq"] for r in records] == list(range(len(records))), (
        "…with no gap in seq: the error record replaced the bad one, it did not skip a number"
    )


async def test_record_is_synchronous_and_does_no_io(tmp_path, monkeypatch):
    """Criterion 3: ``record`` is not a coroutine, opens no file, and does not serialise.

    This is the property the whole design rests on: ``record`` is called from ``Session._handle``,
    from ``HudConnection.receive`` and from ``HudServer.publish``, all on the event loop, and a
    disk write on the pump is the defect no latency measurement would attribute to telemetry.

    Two things the first draft of this row got wrong, both of which made it pass over a ``record``
    that really did open, write and flush:

    * the log was never ``start()``ed, so every ``record`` call returned at its first guard
      (``self._handle is None``) without reaching the code the patch was watching for;
    * the patch *raised*, and ``record`` is total by criterion 3 — it catches ``Exception``, so
      the bomb was swallowed into an ``error`` record and the row stayed green.

    So: start the log first, and watch with spies that call through and are asserted afterwards
    rather than with bombs the code under test is required to swallow.
    """
    log = make_log(tmp_path)
    assert not inspect.iscoroutinefunction(log.record), (
        "criterion 3: record() is synchronous — the pump never awaits telemetry"
    )
    await log.start()
    assert Path(log.path).exists(), (
        "the log is open before the patch is armed — a record() that returns at its first guard "
        "proves nothing about I/O"
    )

    opened: list[str] = []
    dumped: list[str] = []
    real_open, real_path_open, real_dumps = builtins.open, Path.open, json.dumps

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(f"open({file!r})")
        return real_open(file, *args, **kwargs)

    def spy_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(f"Path.open({self!r})")
        return real_path_open(self, *args, **kwargs)

    def spy_dumps(obj: Any, *args: Any, **kwargs: Any) -> Any:
        dumped.append(type(obj).__name__)
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "open", spy_path_open)
    monkeypatch.setattr(json, "dumps", spy_dumps)
    # No ``await`` between here and ``undo``: the writer task is running, and *it* is allowed to
    # open nothing but is certainly allowed to serialise.
    assert log.record("note", {"n": 1}) is None, "record() returns nothing and never raises"
    assert log.record("wire.in", {"n_bytes": 12, "outcome": "malformed", "msg": "?"}) is None
    monkeypatch.undo()

    assert not opened, (
        f"criterion 3: record() performs no I/O, and the pump is what calls it; it called {opened}"
    )
    assert not dumped, (
        f"criterion 3: …and no json.dumps — the serialisation belongs to the writer task; "
        f"record() dumped {dumped}"
    )
    await settle(log)
    await log.aclose()
    records = read_records(log.path)
    assert [r["data"] for r in of_kind(records, "note")] == [{"n": 1}] and of_kind(
        records, "wire.in"
    ), (
        f"…and both records the row offered under the patch really did reach the file afterwards, "
        f"so this row watched a live offer path rather than a guard that returns early: the log "
        f"holds {kinds(records)}"
    )


async def test_record_snapshots_the_callers_object(tmp_path):
    """Criterion 3: the queue holds a snapshot, not a reference into the pump's own state."""
    log = make_log(tmp_path)
    await log.start()
    payload: dict[str, Any] = {"canvases": ["one"], "n": 1}
    log.record("note", payload)
    payload["canvases"].append("two")
    payload["n"] = 99
    await settle(log)
    await log.aclose()

    written = of_kind(read_records(log.path), "note")[0]["data"]
    assert written == {"canvases": ["one"], "n": 1}, (
        f"criterion 3: record() snapshots; the writer wrote what the caller mutated afterwards: "
        f"{written!r}"
    )


async def test_ten_thousand_offers_stay_bounded_and_the_gap_is_visible(tmp_path):
    """Criterion 3 and 27, adversarial: the writer never runs and the pump never notices.

    Regime: n=10 000 offers of a 1.2 KiB ``llm.commit`` payload, ``maxsize=4096``, the writer task
    never scheduled (no await between offers), CPU only, ``perf_counter_ns`` per call.

    The oracle for the ``dropped`` record is a conservation law, not a re-derivation of the
    writer's arithmetic: every ``note`` the test offered is either a line in the file or counted
    in a ``dropped`` record. A writer that forgets the marker, counts it twice, or resets the
    count loses the equality.
    """
    maxsize = 4096
    offers = 10_000
    log = make_log(tmp_path, maxsize=maxsize)
    await log.start()
    before = container_sizes(log)

    samples_ms: list[float] = []
    for n in range(offers):  # no await: the writer task cannot run in this loop
        started = time.perf_counter_ns()
        log.record("note", {"n": n, "text": "x" * 1200})
        samples_ms.append((time.perf_counter_ns() - started) / 1e6)

    assert log.qsize() <= maxsize, (
        f"criterion 3: the queue is bounded at maxsize={maxsize}; it holds {log.qsize()}"
    )
    after = container_sizes(log)
    grew = {
        name: (before.get(name, 0), size)
        for name, size in after.items()
        if size > max(before.get(name, 0), maxsize)
    }
    assert not grew, (
        f"criterion 3: memory is bounded — no other container may grow with the offers; {grew}"
    )
    p50, p95, worst = percentiles(samples_ms)
    assert p95 <= RECORD_P95_MS, (
        f"criterion 27: record() costs ≤ {RECORD_P95_MS} ms at p95 on the pump; over n={offers} "
        f"offers (1.2 KiB payload, writer never scheduled) p50={p50:.4f} ms p95={p95:.4f} ms "
        f"max={worst:.4f} ms"
    )

    await settle(log)
    log.record("note", {"marker": "after the drain"})
    await settle(log)
    await log.aclose()

    records = read_records(log.path)
    drops = of_kind(records, "dropped")
    assert drops, f"criterion 3: a gap is always visible in the file; kinds={set(kinds(records))}"
    written_notes = len(of_kind(records, "note"))
    counted = sum(int(r["data"]["count"]) for r in drops)
    assert written_notes + counted == offers + 1, (
        f"criterion 3: every offered record is either written or counted as dropped; "
        f"{written_notes} written + {counted} counted != {offers + 1} offered"
    )
    index = records.index(drops[0])
    assert records[index + 1]["kind"] == "note", (
        f"criterion 3: the dropped record immediately precedes the next successfully written "
        f"record; it is followed by {records[index + 1]['kind']!r}"
    )
    assert records[index + 1]["data"] == {"marker": "after the drain"}, (
        "…which is the record offered after the queue had room again"
    )
    assert drops[0]["data"]["since_seq"] == records[index - 1]["seq"], (
        f"criterion 3: since_seq names the last record written before the gap "
        f"({records[index - 1]['seq']}); the marker says {drops[0]['data']['since_seq']}"
    )
    assert [r["seq"] for r in records] == list(range(len(records))), (
        "…and seq itself still has no gaps: the dropped record is how a gap is reported"
    )


async def test_record_costs_less_than_the_budget_with_a_frame_in_the_payload(tmp_path):
    """Criterion 27 at the §6 regime: n=1000 calls of the mixed payload on a live log.

    The 180 KiB frame is the expensive one — redaction happens at offer time (§5: "redaction
    before the queue"), so this row is the one that fails if the redactor recomputes a perceptual
    hash or copies the base64 megabyte around instead of replacing it.
    """
    log = make_log(tmp_path)
    await log.start()
    payloads = mixed_payloads()
    samples_ms: list[float] = []
    for n in range(1000):
        kind, data = payloads[n % len(payloads)]
        started = time.perf_counter_ns()
        log.record(kind, data, turn_id="t1", session_ms=n)
        samples_ms.append((time.perf_counter_ns() - started) / 1e6)
        if n % 100 == 0:
            await asyncio.sleep(0)
    await settle(log)
    await log.aclose()

    p50, p95, worst = percentiles(samples_ms)
    assert p95 <= RECORD_P95_MS, (
        f"criterion 27: record() adds ≤ {RECORD_P95_MS} ms at p95 on the pump. Regime: n=1000, "
        f"payload mix of §6 (180 KiB JPEG wire.in, machine.step with an SttStep, 1.2 KiB "
        f"llm.commit), CPU only, perf_counter_ns. Measured p50={p50:.4f} ms p95={p95:.4f} ms "
        f"max={worst:.4f} ms"
    )


async def test_a_log_directory_that_is_a_file_disables_logging_and_says_so(tmp_path):
    """Criterion 4: an ``OSError`` from the file disables logging; it does not raise.

    ``--log-dir`` pointing at an existing *file* is the mistake a human makes at the shell, and
    it must cost the session nothing.
    """
    blocked = tmp_path / "sessions"
    blocked.write_text("this is a file, not a directory\n")

    log = make_log(blocked)
    await log.start()
    assert log.record("note", {"n": 1}) is None, (
        "criterion 4: record() is still total after the log disabled itself"
    )
    await settle(log)
    stats = log.stats()
    assert stats["error"], (
        f"criterion 4: the failure is reported, never swallowed; stats()={dict(stats)!r}"
    )
    assert isinstance(stats["error"], str) and "sessions" in str(stats["error"]), (
        f"…naming the path that could not be opened; stats()={dict(stats)!r}"
    )
    await log.aclose()
    assert blocked.read_text() == "this is a file, not a directory\n", (
        "…and the file that was in the way is untouched"
    )


def log_state_messages(recorder: Any) -> list[Any]:
    """Whatever the session published to say "this is the log's state now" (criterion 4).

    Group **B** adds the wire type (``astra.log.state`` in ``contracts/protocol.py``); group A
    owns the moment it is published. The assertion below is therefore about *how many* states a
    disabling produces, which is checkable without the type existing: the count is what a page
    reads, and two states for one failure is the bug (the panel would flicker between "on" and
    "off"). ``tests/unit/test_hud_ingress.py`` pins the type itself when group B lands.
    """
    return [
        message
        for message in recorder.messages
        if "log" in getattr(message, "type", type(message).__name__).lower()
    ]


async def test_a_session_whose_log_cannot_be_opened_still_completes_its_turn(tmp_path):
    """Criterion 4, adversarial: losing the log costs the session nothing at all.

    The log directory is an existing file, so the writer's first ``open`` raises. The turn under
    way must still complete, still speak, and still write its ``turns.jsonl`` line — with
    ``session_log: null``, because logging was off for that turn (criterion 14). A session that
    dies because its debug log could not be opened is worse than one with no log.
    """
    from tests.unit._hud_rig import Recorder, answer, frame, running

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory\n")
    log = make_log(blocked)
    recorder = Recorder()

    async with running(
        tmp_path / "data",
        script=[answer("Yes it is. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    ) as rig:
        await rig.ask("is it on")
        await rig.speak_out()
        lines = rig.lines()
        spoken = [m for m in recorder.messages if getattr(m, "type", "") == "astra.spoken"]

    assert len(lines) == 1, f"criterion 4: the turn still ends and still writes its line; {lines}"
    assert spoken, "…and it still spoke: the log's failure is not the turn's"
    from local_astra.telemetry.turns_log import TurnRecord

    record = TurnRecord.from_line(lines[0])
    assert record.session_log is None, (
        f"criterion 4 and 14: logging was off for this turn, so the line says so; it says "
        f"{record.session_log!r}"
    )
    assert log.stats()["error"], f"…and the log knows why it is off; stats()={dict(log.stats())!r}"
    states = log_state_messages(recorder)
    assert len(states) <= 1, (
        f"criterion 4: one disabling publishes one state, not a stream of them; the session "
        f"published {[getattr(m, 'type', m) for m in states]}"
    )
