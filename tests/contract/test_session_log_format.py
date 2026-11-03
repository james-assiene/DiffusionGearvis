"""The log file as a *format*: every line validates, the vocabulary is closed, JSONL is strict.

T-0043 group A criteria 1 and 2. The producer here is a real scripted session over the rig fakes
(``tests/unit/_hud_rig.py``), not a hand-written file: a format test whose input the test itself
wrote proves only that the test can write JSON.

Two properties beyond "it parses", and they are the ones a tool depends on:

* **flush per line** — a reader tailing the file while the session is still running sees each
  record before the next is written. That is what makes ``session_log_summary.py`` usable on a
  live session, and what a buffered writer silently breaks;
* **ungitted** — ``git check-ignore`` agrees the path is ignored. ``logs/`` is already in
  ``.gitignore``, so this is inheritance; a log directory moved elsewhere in a later task would
  commit every question the human ever asked.

Regime: no service, no GPU, ``ScriptedClock``, ``tests/fakes/``. This module has no ``backend``
parametrisation because there is no service behind it — the file is written by ``local_astra``
itself, and the "real" side of it is the very session the row runs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.unit._hud_rig import Recorder, answer, frame, running, thought, ui
from tests.unit._session_log_rig import (
    CRITERION_2_KINDS,
    SPEECH_REFUSED,
    kinds,
    make_log,
    of_kind,
    read_records,
    settle,
    symbol,
)

REPO = Path(__file__).resolve().parents[2]


async def logged_session(tmp_path: Path) -> Path:
    """One voice turn with a valid A2UI canvas, logged. Returns the log's path."""
    log = make_log(tmp_path / "logs")
    async with running(
        tmp_path / "data",
        script=[thought("thinking about it. "), answer("One half. "), ui()],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=Recorder(),
        session_log=log,
    ) as rig:
        await rig.ask("can you illustrate one half")
        await rig.speak_out()
    return Path(log.path)


async def test_every_line_of_a_real_session_validates_against_the_record_contract(tmp_path):
    """Criterion 2: the envelope is a contract, and every line of a real session satisfies it."""
    record_model = symbol("LogRecord")
    path = await logged_session(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 5, f"the session logged something at all; {len(lines)} lines"

    for n, line in enumerate(lines):
        try:
            record_model.model_validate_json(line)
        except Exception as exc:
            raise AssertionError(
                f"criterion 2: line {n} of {path.name} does not satisfy the record contract: "
                f"{exc}\nline={line[:400]}"
            ) from None

    written = read_records(path)
    outside = sorted({k for k in kinds(written) if k not in set(CRITERION_2_KINDS)})
    assert not outside, (
        f"criterion 2: kind comes from the closed vocabulary; this session wrote {outside}"
    )
    assert kinds(written)[0] == "log.started" and kinds(written)[-1] == "log.stopped", (
        f"criterion 2: the file opens and closes with its own markers; it reads {kinds(written)}"
    )


async def test_the_file_is_strict_jsonl(tmp_path):
    """Criterion 1: one JSON object per line, UTF-8, ``\\n``-terminated, no blank lines."""
    path = await logged_session(tmp_path)
    raw = path.read_bytes()
    assert raw.endswith(b"\n"), "criterion 1: every record line is \\n-terminated, the last too"
    assert b"\r" not in raw, "…with no carriage returns to confuse a line reader"
    text = raw.decode("utf-8")  # raises for anything that is not UTF-8
    for n, line in enumerate(text.split("\n")[:-1]):
        assert line.strip(), f"line {n} is blank; the format is strict JSONL"
        parsed = json.loads(line)
        assert isinstance(parsed, dict), f"line {n} is a {type(parsed).__name__}, not an object"
    assert path.suffix == ".jsonl", f"criterion 1: the name says what it is; got {path.name}"
    assert path.parent == tmp_path / "logs", (
        f"criterion 1: the file is created in the directory it was given (--log-dir); it is at "
        f"{path}"
    )


async def test_a_reader_tailing_the_file_sees_each_record_before_the_next(tmp_path):
    """Criterion 1: flushed per line, while the session is still running.

    The reader below is the ``tail -f`` a human runs during a live session: it holds the file
    open and reads what has arrived. A writer that buffers passes every other row in this module
    and fails this one.
    """
    log = make_log(tmp_path / "logs")
    await log.start()
    await settle(log)
    with Path(log.path).open(encoding="utf-8") as reader:
        reader.read()  # catch up past log.started
        seen: list[str] = []
        for n in range(3):
            log.record("note", {"n": n})
            await settle(log)
            chunk = reader.read()
            assert chunk.endswith("\n"), (
                f"criterion 1: record {n} is complete on disk before the next is written; the "
                f"reader saw {chunk!r}"
            )
            seen.extend(line for line in chunk.splitlines() if line)
            assert len(seen) == n + 1, (
                f"criterion 1: the reader has seen exactly the {n + 1} records offered so far; it "
                f"has {len(seen)} — a buffered writer shows them all at close, which is too late "
                f"for the session one is watching"
            )
            assert json.loads(seen[-1])["data"] == {"n": n}, (
                f"…and the newest one is the one just offered; it is {seen[-1]}"
            )
    await log.aclose()


def test_git_agrees_the_default_log_path_is_ungitted():
    """Criterion 1: ``git check-ignore`` says the path the log *actually* picks is ignored.

    The path comes from the implementation (``SessionLog`` over the default directory), not from
    a string written here: the point is that a log written on this box during a live session
    cannot be committed by accident, whichever directory the implementer chose. ``logs/`` is
    already gitignored, so the intended answer is inheritance — and a later move to a directory
    that is not ignored would commit every question the human ever asked.
    """
    from local_astra import supervisor

    default_dir = supervisor.log_dir() / "sessions"
    path = Path(make_log(default_dir).path)
    assert path.parent == default_dir, (
        f"criterion 1: the file lives in supervisor.log_dir()/sessions by default; it is at {path}"
    )
    relative = path.relative_to(REPO) if path.is_relative_to(REPO) else path
    result = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", str(relative)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"criterion 1: {relative} must be gitignored; git check-ignore exited "
        f"{result.returncode}: {result.stdout}{result.stderr}"
    )


# --- T-0054 criterion 5: the newest kind is a *record*, not a note ------------------------------
#
# Named in docs/tasks/T-0054.md's "Owned files" as a pin this task legitimately moves: the
# vocabulary rows above read ``CRITERION_2_KINDS`` from ``tests/unit/_session_log_rig.py``, which
# gains ``speech.refused``. Those rows are one-sided — they check that nothing *outside* the
# vocabulary is written — so adding a kind cannot redden them and cannot prove the kind is ever
# produced. This row is the other side, and it is in the *format* module on purpose: a kind the
# writer emits without ``KINDS`` listing it does not crash, it becomes a complaining ``note``
# (``session_log.py:405``), so "the record exists and validates as that kind" is a statement
# about the file format, not only about the session.


async def refusing_session(tmp_path: Path) -> Path:
    """One voice turn whose answer is structurally A2UI interface data. Returns the log's path.

    The text is the human's own recorded canvas 2 (``tests/fixtures/llm/flight_canvases.json``,
    turn ``t6`` of 2026-09-16), delivered the way it arrived that night — as one ``answer``
    commit. The voice guard (T-0052 criterion 8) refuses its first sentence today; what does not
    exist yet is any record that it did.
    """
    from tests.fakes.llm import flight_canvases

    log = make_log(tmp_path / "logs")
    async with running(
        tmp_path / "data",
        script=[answer(flight_canvases()["recorded"]["answer_piece_canvas2"])],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=Recorder(),
        session_log=log,
    ) as rig:
        await rig.ask("build me a flight booking interface")
        await rig.speak_out()
    return Path(log.path)


async def test_a_refusal_is_written_as_its_own_kind_and_validates_as_a_record(tmp_path):
    """T-0054 criterion 5: ``speech.refused`` is a real line of the file, not a smuggled note."""
    record_model = symbol("LogRecord")
    path = await refusing_session(tmp_path)
    written = read_records(path)

    refusals = of_kind(written, SPEECH_REFUSED)
    assert refusals, (
        "T-0054 criterion 5: the turn's SpeechGate refused a sentence (the voice guard has done "
        "so since T-0052) and no record says so. The session log holds "
        f"{sorted(set(kinds(written)))}. A count kept in a per-turn object dies with it."
    )

    for record in refusals:
        record_model.model_validate_json(json.dumps(record))
        assert record["kind"] in set(CRITERION_2_KINDS), (
            f"criterion 5: the kind is inside the closed vocabulary; got {record['kind']!r}. A "
            "kind the writer emits and KINDS does not list becomes a complaining note and the "
            "refusal is lost silently."
        )

    notes = [n for n in of_kind(written, "note") if "refus" in json.dumps(n).lower()]
    assert not notes, (
        "criterion 5: the refusal must be its own kind, not a note complaining that "
        f"'speech.refused' is unknown — session_log.KINDS has to gain it in the same diff. "
        f"Found {notes[:2]}"
    )


# --- T-0092c criterion 7: a tools-on session's file, ``tool.call`` included ---------------------------------


async def test_t0092c_a_tool_turns_log_validates_and_carries_tool_call_records(tmp_path):
    """T-0092c criterion 7: every line of a dry-run tool turn validates against the record contract, its kinds are
    inside the closed vocabulary (which gains ``tool.call``), and the ``tool.call`` records are there."""
    from tests.unit import _tool_turn_support as TT

    record_model = symbol("LogRecord")
    async with TT.tool_rig(tmp_path, script=[TT.answer(TT.SPOKEN), TT.tool(TT.VALID_CLICK)]) as rig:
        await rig.ask("click the sign in button")
        await rig.turn_done()
    written = rig.log_records
    for record in written:
        record_model.model_validate_json(json.dumps(record))
    outside = sorted({k for k in kinds(written) if k not in set(CRITERION_2_KINDS)})
    assert not outside, f"T-0092c criterion 7: kinds outside the vocabulary: {outside}"
    tool_calls = of_kind(written, "tool.call")
    assert [r["data"].get("status") for r in tool_calls] == ["proposed", "dry_run"], (
        f"T-0092c criterion 7: the tool.call records of the turn; got {[r['data'] for r in tool_calls]}"
    )
