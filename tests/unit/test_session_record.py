"""``turns.jsonl``: one line per turn, with the whole regime (T-0007 criterion 12).

Numbers are meaningless without the regime (CLAUDE.md), and the line is what ``astra bench``
reads (docs/03 §7). The record shape is ``telemetry/turns_log.py``'s and is not touched here;
what this task owns is that exactly one line is written per *opened* turn, on the turn-ending
row, with spans drawn only from ``telemetry/spans.py`` and the Q11 fields T-0010 ruling A6 named.
The API is fixed in ``test_session_loop.py``.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from pydantic import ValidationError

from local_astra.contracts.llm import Commit
from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.turn import TurnState
from local_astra.telemetry.spans import SPAN_NAMES
from local_astra.telemetry.turns_log import Regime, TurnRecord
from tests.unit.test_session_loop import (
    TurnsWriter,
    answer,
    frame,
    require_session,
    running,
)

CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"

#: A valid A2UI v0.9.1 canvas (the shape ``tests/unit/test_ui_gate.py`` uses): one surface, one
#: root, one child. Valid on purpose — ``ui_repairs`` must be 0 for input the fixer never touched.
UI_CANVAS = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"lesson_1","catalogId":"'
    + CATALOG_ID
    + '"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"lesson_1","components":['
    '{"id":"root","component":"Column","children":["a"]},'
    '{"id":"a","component":"Text","text":"one half"}]}}'
)

#: The values ``services/versions.lock`` pins; the regime must carry them, not placeholders.
LOCK_VLLM = "v0.29.0"
LOCK_MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


async def test_one_record_per_turn_with_a_complete_regime(tmp_path):
    """Criterion 12: one line, written on the turn-ending row, with a regime that can be compared.

    "Written on the turn-ending row, not on process exit" is asserted by reading the file while
    the session is still running — a writer that flushes in ``aclose()`` loses every line of a
    session that crashes, which is exactly the session whose line one wants.
    """
    async with running(
        tmp_path, script=[answer("Yes it is. ")], frames=[frame(0, "a")], ms_per_word=100
    ) as rig:
        await rig.ask("is it on")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        assert len(rig.lines()) == 1, "the line is written when the turn ends, not at exit"

        await rig.advance(500, step=250)
        rig.llm.script = [answer("No it is not. ")]
        await rig.ask("and now")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "turn two never ended")

        records = rig.records()
        assert [r.turn_id for r in records] == ["t1", "t2"], (
            f"one record per opened turn, in order; got {[r.turn_id for r in records]}"
        )
        assert len({r.session_id for r in records}) == 1, "both turns belong to one session"
        assert all(r.outcome == "spoken" for r in records), [r.outcome for r in records]

        record = records[0]
        assert record.unknown_spans() == [], (
            f"criterion 12: spans come only from telemetry/spans.py; {record.unknown_spans()} "
            "are not in SPAN_NAMES — adding one is a separate change the bench reader must learn"
        )
        required = {"llm.first_canvas", "tts.first_audio", "playback.first_sample", "llm.done"}
        assert required <= set(record.spans), (
            f"criterion 12/15: the turn's spans are missing {sorted(required - set(record.spans))}"
        )
        for name, (start, end) in record.spans.items():
            assert name in SPAN_NAMES and 0 <= start <= end, (name, start, end)

        regime = record.regime
        assert regime.profile == "lean"
        assert (regime.canvas_length, regime.max_denoising_steps) == (256, 48), (
            "the regime's diffusion settings come from the active profile (docs/02 §7)"
        )
        assert regime.thinking is True
        assert LOCK_VLLM in regime.vllm_version, (
            f"services/versions.lock pins {LOCK_VLLM}; the regime says {regime.vllm_version!r}"
        )
        assert regime.model == LOCK_MODEL, regime.model
        assert regime.stt and regime.tts, "the speech services are part of the regime too"
        assert regime.audio_path, "docs/01 §8: headset vs speakers changes every latency number"
        assert regime.git_commit == git_head(), (
            f"the line carries the commit it was produced at; got {regime.git_commit!r}"
        )
        assert regime.assist_flags == AssistFlags() and regime.assists == "pixels"


async def test_q11_fields_are_present_for_a_turn_that_produced_a_surface(tmp_path):
    """Criterion 12 / T-0010 ruling A6: ``ui_repairs``, ``ui_resamples``, ``ui_valid``.

    The canvas below is valid A2UI, so the fixer edits nothing and nothing is resampled: the row
    T-0010 C6 asked this task to write is ``(0, 0, True)``. The point of the criterion is that the
    three numbers exist at all — without them a demo that silently repaired every surface looks
    exactly like one that never had to.
    """
    script = [
        answer("Here it is. "),
        Commit(request_id="?", canvas_idx=1, segment="ui", text=UI_CANVAS),
    ]
    async with running(tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=100) as rig:
        await rig.ask("show me the surface")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        extras = rig.records()[-1].extras
        for key in ("ui_repairs", "ui_resamples", "ui_valid"):
            assert key in extras, f"criterion 12: extras must carry {key!r}; got {sorted(extras)}"
        assert extras["ui_valid"] is True, (
            f"the canvas is valid A2UI v0.9.1; the gate reported {extras['ui_valid']!r}"
        )
        assert extras["ui_repairs"] == 0, (
            f"valid input needs no repair; the gate counted {extras['ui_repairs']} edits"
        )
        assert extras["ui_resamples"] == 0, extras["ui_resamples"]
        assert "one half" not in "".join(t for _, t, _ in rig.tts.requests), (
            "the ui channel is shown, never spoken (CLAUDE.md: two output channels)"
        )


async def test_prefix_tokens_are_reported_as_they_come_back(tmp_path):
    """Criterion 12's other two extras, and the group-A half of criterion 19.

    ``PrefillHandle.prefix_tokens == 0`` means the engine did not tell us (no ``stream_options``),
    **not** an empty prefix and not an error: the line says ``prefix_tokens_reported=false`` and
    the turn runs normally. T-0020 is the task that uses the number.
    """
    async with running(
        tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")], prefix_tokens=1234
    ) as rig:
        await rig.ask("hello")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        extras = rig.records()[-1].extras
        assert extras.get("prefix_tokens") == 1234, extras
        assert extras.get("prefix_tokens_reported") is True, extras

    async with running(
        tmp_path / "zero", script=[answer("Yes. ")], frames=[frame(0, "a")], prefix_tokens=0
    ) as rig:
        await rig.ask("hello again")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "criterion 19: prefix_tokens == 0 is not an error and not an empty prefix",
        )
        record = rig.records()[-1]
        assert record.outcome == "spoken", record.outcome
        assert record.extras.get("prefix_tokens_reported") is False, record.extras


async def test_no_record_for_a_turn_that_never_opened(tmp_path):
    """Criterion 12, negative: no line for a session that never opened a turn.

    Ticks, scene changes and an assist toggle are not a turn. A writer that emits a line per
    *session* or per state change makes every count in the bench wrong.
    """
    async with running(tmp_path, script=[answer("Yes. ")], frames=[frame(0, "a")]) as rig:
        await rig.advance(3_000, step=250)
        assert rig.state.state is TurnState.IDLE
        assert rig.lines() == [], f"no turn opened, so no line; got {rig.lines()}"
    assert rig.lines() == [], "and closing the session does not invent one"


def test_the_turns_writer_appends_one_json_line_per_record(tmp_path):
    """``TurnsWriter``: append-only, one line per record, readable by ``TurnRecord.from_line``."""
    require_session()
    path = tmp_path / "turns.jsonl"
    regime = Regime(
        profile="lean",
        canvas_length=256,
        max_denoising_steps=48,
        thinking=True,
        vllm_version="vllm/vllm-openai:v0.29.0",
        model=LOCK_MODEL,
        stt="kyutai/stt-1b-en_fr",
        tts="pocket-tts",
        audio_path="null",
    )
    writer = TurnsWriter(path)
    for index in (1, 2):
        writer.write(
            TurnRecord(
                turn_id=f"t{index}",
                session_id="s",
                spans={"llm.done": (0, 1)},
                regime=regime,
                outcome="spoken",
            )
        )
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and [TurnRecord.from_line(line).turn_id for line in lines] == [
        "t1",
        "t2",
    ]
    assert all(json.loads(line)["regime"]["profile"] == "lean" for line in lines)


# --- T-0039 group A: how the question arrived (criterion 9) -------------------------------------
#
#   TurnRecord.input: Literal["voice", "text", "ui"] = "voice"
#
# On the *record* rather than on the ``Regime`` because one session mixes spoken, typed and
# clicked turns (docs/tasks/T-0039.md §5, ambiguity 2). The default is the load-bearing half:
# every line written before this task, and every spoken turn after it, is a voice turn and
# nothing names the field — a required field would make every existing turns.jsonl unreadable to
# the reader in local_astra/bench/.


def _regime() -> Regime:
    return Regime(
        profile="lean",
        canvas_length=256,
        max_denoising_steps=48,
        thinking=True,
        vllm_version="vllm/vllm-openai:v0.29.0",
        model=LOCK_MODEL,
        stt="kyutai/stt-1b-en_fr",
        tts="pocket-tts",
        audio_path="silent",
    )


def _record(regime: Regime, **kwargs) -> TurnRecord:
    return TurnRecord(
        turn_id="t1",
        session_id="s",
        spans={"llm.done": (0, 1)},
        regime=regime,
        outcome="spoken",
        **kwargs,
    )


def test_the_input_field_defaults_to_voice_and_is_serialised():
    """Criterion 9: the field exists, defaults to ``"voice"``, and travels on the line."""
    assert "input" in TurnRecord.model_fields, (
        'T-0039 criterion 9: TurnRecord gains input: Literal["voice","text","ui"] = "voice"; '
        f"it declares {sorted(TurnRecord.model_fields)}"
    )
    default = _record(_regime())
    assert default.input == "voice", (
        f"criterion 9: an unnamed input is a voice turn; the default is {default.input!r}"
    )
    assert json.loads(default.to_line())["input"] == "voice", (
        "criterion 9: …and it is on the line, not implied by its absence — a reader that has to "
        "guess would label every old line with whatever it guesses"
    )


@pytest.mark.parametrize("value", ["voice", "text", "ui"])
def test_a_line_round_trips_through_the_reader_with_every_input_value(value):
    """Criterion 9: the three values survive ``to_line``/``from_line`` and change nothing else."""
    assert "input" in TurnRecord.model_fields, "T-0039 criterion 9 (see the row above)"
    regime = _regime()
    record = _record(regime, input=value)
    back = TurnRecord.from_line(record.to_line())
    assert back == record, (
        f"criterion 9: a line with input={value!r} reads back unchanged in every field; "
        f"{back!r} != {record!r}"
    )
    assert back.input == value and back.regime == regime, (
        "criterion 9: the regime is otherwise unchanged (§5, ambiguity 2: no second copy of the "
        "field there)"
    )


def test_the_input_field_is_a_closed_set():
    """Criterion 9: a fourth value is a spec change, not a free-text label.

    ``input`` is what a later reader filters on; as a plain ``str`` the first typo (``"typed"``)
    would silently produce an empty filter rather than a loud error.
    """
    assert "input" in TurnRecord.model_fields, "T-0039 criterion 9 (see the row above)"
    with pytest.raises(ValidationError):
        _record(_regime(), input="typed")


# --- T-0043 criterion 14: the log path on the turn line -----------------------------------------

#: A ``turns.jsonl`` line written *before* this task (captured from a rig run at afeb1ef). It has
#: no ``session_log`` field, and a bench reader must still be able to read it.
LINE_BEFORE_T0043 = (
    '{"turn_id":"t1","session_id":"d5fae116225e","spans":{"reflex.onset":[0,0],'
    '"llm.prefill":[0,0],"stt.first_word":[0,200],"stt.last_word":[400,600],'
    '"stt.end_of_turn":[680,700],"llm.first_canvas":[830,860],"tts.first_audio":[860,860],'
    '"playback.first_sample":[860,860],"llm.done":[830,860],"playback.stopped":[1160,1160]},'
    '"events":[],"regime":{"profile":"lean","canvas_length":256,"max_denoising_steps":48,'
    '"thinking":true,"vllm_version":"vllm/vllm-openai:v0.29.0",'
    '"model":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","stt":"kyutai/stt-1b-en_fr",'
    '"tts":"pocket-tts@3.1.0","audio_path":"null","assists":"pixels",'
    '"assist_flags":{"page_text":false,"pdf_text":false,"cells":false,"at_spi":false},'
    '"git_commit":"afeb1ef2a973e1f017fad34b86401dbf4d0567cc","seed":null},"input":"voice",'
    '"outcome":"spoken","canvases":3,"thought_tokens":0,"answer_tokens":0,"thought_closed":null,'
    '"extras":{"prefix_tokens":1000,"prefix_tokens_reported":true,"stt_frames_fed":14,'
    '"stt_feed_errors":0,"capture_xruns":0,"capture_gated_blocks":0,"ui_repairs":0,'
    '"ui_resamples":0,"ui_valid":true}}'
)


def a_record(**overrides):
    """The smallest valid ``TurnRecord``; T-0043 adds exactly one optional field to it."""
    values = dict(
        turn_id="t1",
        session_id="s1",
        spans={},
        regime=Regime(
            profile="lean",
            canvas_length=256,
            max_denoising_steps=48,
            thinking=True,
            vllm_version=LOCK_VLLM,
            model=LOCK_MODEL,
            stt="kyutai/stt-1b-en_fr",
            tts="pocket-tts@3.1.0",
            audio_path="null",
        ),
        outcome="spoken",
    )
    values.update(overrides)
    return TurnRecord(**values)


def test_session_log_defaults_to_none_and_is_a_top_level_field():
    """T-0043 criterion 14: a top-level field, defaulting to ``None``, that serialises.

    Not an ``extras`` key: a bench reader must not have to guess where the log path is.
    """
    assert "session_log" in TurnRecord.model_fields, (
        f"criterion 14: TurnRecord gains session_log: str | None = None; its fields are "
        f"{sorted(TurnRecord.model_fields)}"
    )
    default = a_record()
    assert default.session_log is None, (
        f"criterion 14: it defaults to None (logging was off for that turn); got "
        f"{default.session_log!r}"
    )
    line = json.loads(default.to_line())
    assert "session_log" in line and line["session_log"] is None, (
        f"…and it is written on every line, as a top-level key; the line is {line}"
    )
    assert "session_log" not in line["extras"], "…never as an extras key (criterion 14)"

    path = "/home/user/demos/local_astra/logs/sessions/20260915-201500-4242.jsonl"
    carried = a_record(session_log=path)
    assert TurnRecord.from_line(carried.to_line()).session_log == path, (
        "criterion 14: the absolute path round-trips through the reader unchanged"
    )


def test_a_line_written_before_this_task_still_parses():
    """T-0043 criterion 14: every other field is unchanged and old lines still read.

    The line is a real one from before the field existed; ``astra bench`` reads a directory of
    them and must not start over because one task added a column.
    """
    assert "session_log" in TurnRecord.model_fields, (
        "criterion 14 has not been implemented yet (this row would pass vacuously otherwise)"
    )
    record = TurnRecord.from_line(LINE_BEFORE_T0043)
    assert record.session_log is None, (
        f"criterion 14: a line without the field reads as 'not logged'; got {record.session_log!r}"
    )
    assert record.turn_id == "t1" and record.outcome == "spoken" and record.canvases == 3, (
        "…and everything it did carry is unchanged"
    )
    assert record.extras["ui_valid"] is True and record.regime.profile == "lean"


async def test_a_logging_session_names_its_log_on_every_line(tmp_path):
    """T-0043 criterion 14: the absolute path of the log this turn was recorded in, per turn.

    Two turns in one session: both lines name the same file, absolutely, and the file is the one
    the session is writing.
    """
    from tests.unit._hud_rig import Recorder
    from tests.unit._hud_rig import answer as hud_answer
    from tests.unit._hud_rig import frame as hud_frame
    from tests.unit._hud_rig import running as hud_running
    from tests.unit._session_log_rig import make_log

    log = make_log(tmp_path / "logs")
    async with hud_running(
        tmp_path / "data",
        script=[hud_answer("Yes it is. ")],
        frames=[hud_frame(0, "a")],
        ms_per_word=100,
        publish=Recorder(),
        session_log=log,
    ) as rig:
        await rig.ask("is it on")
        await rig.speak_out()
        await rig.advance(500, step=250)
        rig.llm.script = [hud_answer("No it is not. ")]
        await rig.ask("and now")
        await rig.speak_out()
        lines = rig.lines()

    assert len(lines) == 2, f"two turns, two lines; got {len(lines)}"
    paths = [TurnRecord.from_line(line).session_log for line in lines]
    assert paths == [str(log.path), str(log.path)], (
        f"criterion 14: every line of a logged session names the log it was recorded in "
        f"({log.path}); the lines say {paths}"
    )
    assert all(p is not None and p.startswith("/") for p in paths), (
        f"…as an absolute path — a relative one is unresolvable from wherever bench runs; got "
        f"{paths}"
    )


async def test_a_session_without_a_log_writes_none(tmp_path):
    """T-0043 criterion 14, the other half: ``None`` when logging was off for that turn."""
    from tests.unit._hud_rig import Recorder
    from tests.unit._hud_rig import answer as hud_answer
    from tests.unit._hud_rig import running as hud_running

    assert "session_log" in TurnRecord.model_fields, (
        "criterion 14 has not been implemented yet (this row would pass vacuously otherwise)"
    )
    async with hud_running(
        tmp_path, script=[hud_answer("Yes it is. ")], ms_per_word=100, publish=Recorder()
    ) as rig:
        await rig.ask("is it on")
        await rig.speak_out()
        lines = rig.lines()

    assert len(lines) == 1 and TurnRecord.from_line(lines[0]).session_log is None, (
        f"criterion 14: a session with no log says so on its line; it says "
        f"{TurnRecord.from_line(lines[0]).session_log!r}"
    )


# --- T-0092b criteria 2 and 5: Regime.tools on every line, and "repair off" announced ---------------
#
# The kit is built by ``tests/unit/_tools_switch_support.py`` (its docstring fixes the assumed names).

from tests.unit import _tools_switch_support as T  # noqa: E402


async def _two_turn_regimes(tmp_path, kit) -> list[object]:
    rig = T.session_rig(tmp_path, kit=kit, script=[answer("One. ")])
    await rig.session.start()
    try:
        await T.one_typed_turn(rig, "first")
        await rig.advance(300)
        rig.llm.script = [answer("Two. ")]
        await T.one_typed_turn(rig, "second")
        await rig.until(lambda: len(rig.lines()) == 2, "turn two never wrote its line")
    finally:
        await rig.session.aclose()
    return [r.regime for r in rig.records()]


async def test_a_tools_off_turn_says_tools_off_and_carries_no_tool(tmp_path):
    """Criterion 5: with no kit the regime reads ``"off"`` and ``tool`` is ``None`` on every line."""
    rig = T.session_rig(tmp_path, script=[answer("Done. ")])
    await rig.session.start()
    try:
        await T.one_typed_turn(rig)
    finally:
        await rig.session.aclose()
    line = json.loads(rig.lines()[-1])
    assert line["regime"].get("tools") == "off", (
        f"T-0092b criterion 5: regime.tools is 'off'; {line['regime']}"
    )
    assert "tool" in line and line["tool"] is None, "T-0092b criterion 5: tool is written as null"


async def test_a_dry_run_session_writes_native_sha12_dry_run_on_every_turn(tmp_path):
    """Criterion 5: ``"native:<tools sha12>:dry_run"`` (sha12 of tools_v1.json = bbfc53ccdd80), every turn —
    tools are a session setting, so no turn changes it."""
    assert T.tools_sha256()[:12] == T.TOOLS_SHA12, "premise: tools_v1.json is the bbfc53cc… set"
    regimes = await _two_turn_regimes(tmp_path, T.make_kit())
    assert [getattr(r, "tools", "<absent>") for r in regimes] == [T.REGIME_ON, T.REGIME_ON]


async def test_repair_off_is_suffixed_on_every_turn_of_the_session(tmp_path):
    """Criteria 2 and 5: repair off appends ``":repair_off"``, and stays off for the life of the session."""
    regimes = await _two_turn_regimes(tmp_path, T.make_kit(repair_off="payloads stale: test"))
    assert [getattr(r, "tools", "<absent>") for r in regimes] == [
        T.REGIME_REPAIR_OFF,
        T.REGIME_REPAIR_OFF,
    ]


async def _log_records(tmp_path, kit):
    from local_astra.telemetry.session_log import SessionLog
    from tests.unit._session_log_rig import StepClock, read_records

    log = SessionLog(tmp_path / "logs", clock=StepClock())
    rig = T.session_rig(tmp_path, kit=kit, session_log=log)
    await rig.session.start()
    await rig.pump()
    await rig.session.aclose()
    return read_records(log.path)


async def test_repair_off_is_the_first_record_after_log_started(tmp_path):
    """Criterion 2: ``note {what: "tool repair off", reason}`` is the record right after ``log.started``."""
    reason = "manifest model_revision 0000 is not versions.lock's 2ea8372"
    records = await _log_records(tmp_path, T.make_kit(repair_off=reason))
    assert [r["kind"] for r in records[:2]] == ["log.started", "note"], (
        f"T-0092b criterion 2: the repair-off note follows log.started; the log begins "
        f"{[(r['kind'], r.get('data')) for r in records[:3]]}"
    )
    data = records[1]["data"]
    assert data.get("what") == "tool repair off" and data.get("reason") == reason, data


@pytest.mark.parametrize("repair", ["on", "tools off"])
async def test_no_repair_off_note_when_repair_is_on_or_tools_are_off(tmp_path, repair):
    """Criterion 2, the too-wide direction: the announcement is made only when repair really is off."""
    kit = T.make_kit() if repair == "on" else None
    records = await _log_records(tmp_path, kit)
    notes = [
        r for r in records if r["kind"] == "note" and r["data"].get("what") == "tool repair off"
    ]
    assert not notes, f"repair {repair}: no 'tool repair off' note; got {notes}"
    if kit is not None:
        assert records and records[0]["kind"] == "log.started", "premise: the log started"
