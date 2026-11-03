"""The fence lifecycle and the voice guard's refusals reach the log, the record and the wire.

T-0054 group A, criteria 1-12. The gap this module pins, in the task's words: today "a turn whose
surface was abandoned at the bound, a turn whose surface was reassembled from three canvases, and
a turn that never had a surface at all produce the *same* session log, the *same* ``turns.jsonl``
line and the *same* pane" — and a refusal leaves no trace outside the ``SpeechGate`` object,
which is discarded when the turn ends.

Everything here is a scripted turn over ``tests/fakes/`` with the ``ScriptedClock``, driven
through ``tests/unit/_hud_rig.py``'s ``Recorder``/``running`` helpers (read-only use — that file
is not this task's) and read back with ``tests/unit/_session_log_rig.py``'s ``read_records`` /
``of_kind``. The oracle for a fence value is always the ``Commit`` the fake backend was scripted
with; the oracle for a refusal is always an independent ``SpeechGate`` fed the same bytes.
``SpeechGate`` and ``llm_client`` are not this task's files (criterion 6) and are used here only
as oracles.

Regime: no service, no GPU, no timing claimed, n = 1 per row, CPU only.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from local_astra.conductor.speech_gate import MAX_REFUSED_SOURCES, SpeechGate
from local_astra.contracts import events as E
from local_astra.contracts.codec import decode, encode
from local_astra.contracts.llm import Commit
from local_astra.contracts.protocol import (
    ASTRA_MESSAGE_TYPES,
    CLIENT_TO_SERVER,
    SERVER_TO_CLIENT,
    AstraSpoken,
)
from tests.unit._hud_rig import (
    UI_CANVAS,
    Recorder,
    answer,
    frame,
    make_rig,
    require_symbol,
    running,
    thought,
)
from tests.unit._session_log_rig import (
    SPEECH_REFUSED,
    UI_FENCE_TYPE,
    fence_messages,
    make_log,
    of_kind,
    read_records,
    ui_fence_model,
)
from tests.unit._t0177_pins import without_vision_frames  # T-0177 B-P1

REPO = Path(__file__).resolve().parents[2]

#: The extras a turn line carried **before** this task, recorded on the branch base (commit
#: adb9b3b, this box, 2026-09-17) from the very rig these rows use. Criterion 3 says the line
#: "gains exactly three keys", which is a statement about a difference, so the baseline has to be
#: written down for the difference to exist.
EXTRAS_BEFORE_T0054 = frozenset(
    {
        "prefix_tokens",
        "prefix_tokens_reported",
        "stt_frames_fed",
        "stt_feed_errors",
        "capture_xruns",
        "capture_gated_blocks",
        "ui_repairs",
        "ui_resamples",
        "ui_valid",
    }
)

#: Criterion 3's three, in the task file's own spelling.
NEW_EXTRAS = frozenset({"ui_canvases", "ui_fence", "speech_refused"})

#: T-0059 criterion 20's four, added to every turn line after this task (ruling A15).
FRAME_EXTRAS = frozenset({"frames", "frames_kept", "scene_changes_seen", "history_images_evicted"})

#: T-0108 criterion 10's two, added to every turn line under that task's name (discounted).
T0108_EXTRAS = frozenset({"ui_valid_final", "ui_shown"})

#: T-0127 criterion 5's one, added to every turn line under that task's name (discounted; re-scope R1).
T0127_EXTRAS = frozenset({"ui_complete_final"})

#: Criterion 10's baseline: every message one ordinary spoken turn puts on the wire, in order,
#: recorded on the branch base (commit adb9b3b, this box, 2026-09-17) from
#: ``ordinary_turn`` below, over three consecutive runs that agreed exactly. "The cost of an
#: observability feature must be zero on the healthy path" is a claim about a trace, and this is
#: the trace it is a claim about. A row of this file re-records it; nothing else may.
ORDINARY_TURN_WIRE = (
    "astra.turn.state",
    "astra.transcript",
    "astra.transcript",
    "astra.transcript",
    "astra.transcript",
    "astra.transcript",
    "astra.turn.state",
    "astra.transcript",
    "astra.turn.state",
    "astra.turn.state",
    "astra.spoken",
    "astra.turn.state",
    "astra.spoken.played",
    "astra.turn.state",
)
#: T-0180 (narrowed, 2026-10-28): the trace is compared with messages of exactly this type left out; the session
#: publishes the turn's committed thought text in them, and nothing else about the trace may move.
THOUGHT_TYPE = "astra.thought"

#: sha256 of the two pure modules T-0054 criterion 6 requires a **zero diff** over, as T-0052
#: merged them. A later task that legitimately edits either updates these in its own PR — the
#: convention ``TURN_MACHINE_SHA256`` set in ``tests/unit/test_session_loop.py``.
#: T-0088a criterion 3 edits speech_gate.py (special-token trims and drops); re-pinned at ab462e5.
#: T-0224 edits speech_gate.py (repeat stop, spoken maths markup); re-pinned at bb6816fe, ef62d398 (R-6),
#: 67a26833 (R-7).
#: T-0232 edits speech_gate.py (instruction-leak refusal, word-sequence rule R-1); re-pinned at 5397cc2b,
#: 75900b83 (R-7b: forward floor 9 words, reverse floor 6).
SPEECH_GATE_SHA256 = "89c57c0e4261e52494588b7fccd0ec33d6e29fd02f04a062ac4e745519625470"
#: T-0088 implements the id-mode tools client in llm_client.py; re-pinned at 1bf503e, at 63c2493 (E11/E14) and at d7929f8 (E15).
#: T-0131 R2: A2 adds the visual-token budget to llm_client.py's request_body; re-pinned at 24db2e2.
#: T-0170: the orphan-close implicit thought and hold in llm_client.py; re-pinned at 71cc4f38.
#: T-0171: LlmEngineEmpty and the [DONE]/token_ids tracking in llm_client.py; re-pinned at 99eac20b.
LLM_CLIENT_SHA256 = "1e42a9404d6a3f8b502fdd7d13fe76867f736eae8d55bdd4f76e9cb80ad2585f"

QUESTION = "what is on this screen"


#: A sentence the voice guard refuses: it contains ``{`` and a quoted member of ``A2UI_KEYS``
#: (``"component"``), which is rule 1 of ``speech_gate._is_interface_data``. Balanced, so the
#: gate does not hold it, and terminated, so it is a sentence rather than a held clause.
def interface_sentence(tag: str) -> str:
    return f'{{"id":"{tag}","component":"Text","text":"{tag}"}} and so on. '


def ui_commit(
    canvas_idx: int,
    *,
    close: str | None,
    continued: bool = False,
    text: str = UI_CANVAS,
) -> Commit:
    """A ``ui`` commit with its fence lifecycle, as ``VllmClient`` emits one (T-0052 criterion 12).

    ``tests/unit/_hud_rig.py``'s ``ui()`` predates the two fields and is not this task's file, so
    the fence-carrying shape is built here.
    """
    return Commit(
        request_id="?",
        canvas_idx=canvas_idx,
        segment="ui",
        text=text,
        fence_close=close,
        fence_continued=continued,
    )


async def run_turn(
    tmp_path: Path,
    script: list[Commit],
    *,
    question: str = QUESTION,
    log: Any = None,
    recorder: Recorder | None = None,
    **kwargs: Any,
) -> tuple[Any, Recorder, Any]:
    """One scripted voice turn. Returns ``(turn record, recorder, rig)``."""
    recorder = recorder if recorder is not None else Recorder()
    extra = {"session_log": log} if log is not None else {}
    async with running(
        tmp_path,
        script=script,
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        **extra,
        **kwargs,
    ) as rig:
        await rig.ask(question)
        await rig.speak_out()
        records = rig.records()
    assert records, "the turn wrote no turns.jsonl line at all"
    return records[-1], recorder, rig


def commit_data(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r["data"] for r in of_kind(records, "llm.commit")]


# --- criterion 1: the llm.commit record carries the fence lifecycle ---------------------------


async def test_the_llm_commit_record_carries_the_fence_lifecycle(tmp_path) -> None:
    """Criterion 1: every commit's ``fence_close``/``fence_continued``, taken from the ``Commit``.

    The oracle is the script the fake backend was given — compared field by field against the
    records, never a literal copied from a run. ``session.py:1383-1401`` copies commit fields one
    by one, which is exactly why the two fields T-0052 added stop at the contract today.
    """
    script = [
        thought("Let me think about it. "),
        ui_commit(1, close="open"),
        ui_commit(2, close="inferred", continued=True),
        answer("There you go. "),
    ]
    log = make_log(tmp_path / "logs")
    await run_turn(tmp_path / "data", script, log=log)

    written = commit_data(read_records(Path(log.path)))
    assert len(written) == len(script), (
        f"criterion 10 of T-0043: one llm.commit record per commit; {len(written)} for "
        f"{len(script)} commits"
    )
    for expected, got in zip(script, written, strict=True):
        where = f"canvas {expected.canvas_idx} ({expected.segment})"
        assert got.get("canvas_idx") == expected.canvas_idx, f"records out of order at {where}"
        assert "fence_close" in got, (
            f"criterion 1: the llm.commit record gains fence_close (the FenceClose literal or "
            f"null) for {where}; the record holds {sorted(got)}"
        )
        assert "fence_continued" in got, (
            f"criterion 1: …and fence_continued (bool) for {where}; it holds {sorted(got)}"
        )
        assert got["fence_close"] == expected.fence_close, (
            f"criterion 1: {where} was committed with fence_close={expected.fence_close!r} and "
            f"the record says {got['fence_close']!r} — taken from the Commit, never re-derived"
        )
        assert got["fence_continued"] == expected.fence_continued, (
            f"criterion 1: {where} was committed with "
            f"fence_continued={expected.fence_continued!r}; the record says "
            f"{got['fence_continued']!r}"
        )


async def test_a_thought_or_answer_commit_records_a_null_fence(tmp_path) -> None:
    """Criterion 2, **adversarial**: a writer that stamps the last-seen value on every record.

    The script interleaves ``ui`` and ``answer`` commits, so a "last value wins" writer — which
    passes criterion 1 unharmed — puts ``"inferred"`` on a piece of spoken prose here. The record
    must say "this was never a fence", not "this fence ended cleanly".
    """
    script = [
        thought("Thinking. "),
        ui_commit(1, close="open"),
        answer("Here is the first half. "),
        ui_commit(2, close="inferred", continued=True),
        answer("And that is the rest. "),
    ]
    log = make_log(tmp_path / "logs")
    await run_turn(tmp_path / "data", script, log=log)

    written = commit_data(read_records(Path(log.path)))
    non_ui = [d for d in written if d.get("segment") != "ui"]
    assert len(non_ui) == 3, f"setup: three non-ui commits; the log holds {len(non_ui)}"
    for data in non_ui:
        assert data.get("fence_close") is None, (
            f"criterion 2: a {data['segment']} commit is not a fence and its record says null, "
            f"not the value of the ui commit beside it; canvas {data.get('canvas_idx')} says "
            f"{data.get('fence_close')!r}"
        )
        assert data.get("fence_continued") is False, (
            f"criterion 2: …and fence_continued is false on every non-ui commit; canvas "
            f"{data.get('canvas_idx')} says {data.get('fence_continued')!r}"
        )


# --- criteria 3 and 4: the turn line ----------------------------------------------------------


async def test_three_turns_read_differently_on_the_turn_line(tmp_path) -> None:
    """Criterion 4, the headline row: never existed / reassembled / abandoned are three states.

    Asserted as a set of three distinct tuples **and** individually, so "all three the same" and
    "one of them wrong" fail differently — a single set-equality assertion would report the two
    as the same failure and a reader could not tell which happened.
    """
    scripts = {
        "never existed": [answer("Nothing to show. ")],
        "reassembled": [
            ui_commit(1, close="open"),
            ui_commit(2, close="inferred", continued=True),
            answer("There you go. "),
        ],
        "abandoned": [
            ui_commit(1, close="open"),
            ui_commit(2, close="open", continued=True),
            ui_commit(3, close="abandoned", continued=True),
            answer("It got away from me. "),
        ],
    }
    expected = {
        "never existed": (0, None),
        "reassembled": (2, "inferred"),
        "abandoned": (3, "abandoned"),
    }

    seen: dict[str, tuple[Any, Any]] = {}
    for name, script in scripts.items():
        record, _, _ = await run_turn(tmp_path / name.replace(" ", "_"), script)
        seen[name] = (record.extras.get("ui_canvases"), record.extras.get("ui_fence"))

    for name, want in expected.items():
        assert seen[name] == want, (
            f"criterion 4: the {name!r} turn must read {want} as (ui_canvases, ui_fence) on its "
            f"own line; it reads {seen[name]}"
        )
    assert len(set(seen.values())) == 3, (
        "criterion 4: the three cases must be *distinguishable from the turn line alone*, which "
        f"is the whole point — they read {seen}"
    )


async def test_a_turn_with_no_surface_records_zero_canvases_and_a_null_fence(tmp_path) -> None:
    """Criterion 3's "never existed" leg, and the three keys on every line.

    ``ui_fence: null`` means "no fence was opened", never "the fence ended cleanly" — the
    distinction the whole task is about, and the one a reader of ``turns.jsonl`` will make
    without asking anyone.
    """
    record, _, _ = await run_turn(tmp_path, [thought("Thinking. "), answer("Nothing to show. ")])
    extras = record.extras

    missing = sorted(NEW_EXTRAS - set(extras))
    assert not missing, (
        f"criterion 3: extras gains exactly {sorted(NEW_EXTRAS)} on **every** turn line, beside "
        f"ui_repairs / ui_resamples / ui_valid because they are the same family and a bench "
        f"reader already reads that family from there (T-0054 D2). Missing: {missing}; the line "
        f"holds {sorted(extras)}"
    )
    assert set(extras) >= EXTRAS_BEFORE_T0054, (
        "criterion 3: the three are *added*; nothing that was on the line before this task may "
        f"leave it. Missing: {sorted(EXTRAS_BEFORE_T0054 - set(extras))}"
    )
    assert (
        set(extras) - EXTRAS_BEFORE_T0054 - FRAME_EXTRAS - T0108_EXTRAS - T0127_EXTRAS == NEW_EXTRAS
    ), (
        "criterion 3: exactly three keys, no more — a fourth is a field nobody documented and "
        f"``extras`` has no schema to discover it from (D2's stated cost). Still closed-world: "
        f"T-0059 criterion 20's four are named in FRAME_EXTRAS and discounted, so a *fifth* new "
        f"key reddens this. The line added "
        f"{sorted(set(extras) - EXTRAS_BEFORE_T0054 - FRAME_EXTRAS - T0108_EXTRAS - T0127_EXTRAS)}"
    )
    assert extras["ui_canvases"] == 0, (
        f"criterion 3: no ui commit means zero canvases; the line says {extras['ui_canvases']!r}"
    )
    assert extras["ui_fence"] is None, (
        "criterion 3: and a null fence — 'no fence was opened', which is not the same statement "
        f"as 'written' or 'inferred'. The line says {extras['ui_fence']!r}"
    )
    assert extras["speech_refused"] == 0, (
        f"criterion 3: an int >= 0 on every line, zero here; it says {extras['speech_refused']!r}"
    )


# --- criteria 5 and 7: the refusal ------------------------------------------------------------


def refusal_oracle(text: str) -> SpeechGate:
    """An independent ``SpeechGate`` fed the same bytes. Not this task's file; used as an oracle."""
    gate = SpeechGate()
    gate.feed(Commit(request_id="oracle", canvas_idx=0, segment="answer", text=text))
    gate.close()
    return gate


async def test_a_refusal_is_recorded_with_its_count_and_its_sources(tmp_path) -> None:
    """Criterion 5: the record carries the count and the sources, **byte-exact**.

    Byte-exact rather than a count alone because ``speech_refused: 1`` means the guard fired, not
    that it was right: a false positive (the guard silencing ordinary prose — T-0052 rulings
    A21/A14 measured how narrow the line is) is indistinguishable from a true refusal on the turn
    line, and the retained sources are the only way a human tells the two apart.
    """
    prose = "Here is a summary of what I built. "
    text = prose + interface_sentence("root")
    oracle = refusal_oracle(text)
    assert oracle.refused_count == 1, (
        f"setup: the guard must really refuse exactly one sentence of this text; it refused "
        f"{oracle.refused_count}"
    )

    log = make_log(tmp_path / "logs")
    record, _, _ = await run_turn(tmp_path / "data", [answer(text)], log=log)
    written = read_records(Path(log.path))

    refusals = of_kind(written, SPEECH_REFUSED)
    assert refusals, (
        "criterion 5: the gate refused a sentence and nothing recorded it. The log holds "
        f"{sorted({r['kind'] for r in written})}. A second line of defence firing means "
        "something upstream is wrong, and it must leave a trace outside the object that counted it."
    )
    data = refusals[-1]["data"]
    assert data.get("count") == oracle.refused_count, (
        f"criterion 5: count is the gate's running count ({oracle.refused_count}); the record "
        f"says {data.get('count')!r}"
    )
    assert data.get("sources") == oracle.refused_sources, (
        "criterion 5: sources are the gate's retained sources byte-exact — not normalised, not "
        f"truncated, not re-split. Expected {oracle.refused_sources!r}, got {data.get('sources')!r}"
    )
    assert data.get("truncated") is False, (
        "criterion 5: truncated is true when count > len(sources); one refusal is not truncated "
        f"and the record says {data.get('truncated')!r}"
    )
    assert record.extras.get("speech_refused") == oracle.refused_count, (
        f"criterion 3: the same number reaches the turn line; it says "
        f"{record.extras.get('speech_refused')!r}"
    )


async def test_the_ninth_refusal_is_counted_and_the_record_says_it_was_truncated(tmp_path) -> None:
    """Criterion 5's bound: nine refusals, ``count == 9``, eight sources, ``truncated is True``.

    ``MAX_REFUSED_SOURCES`` is imported rather than spelled, so this pins the *behaviour* at the
    bound without re-implementing it: past the bound refusals are still counted and only the
    transcript stops growing, so a pathological turn cannot grow the gate without limit.
    """
    text = "".join(interface_sentence(f"c{n}") for n in range(9))
    oracle = refusal_oracle(text)
    assert oracle.refused_count == 9 and len(oracle.refused_sources) == MAX_REFUSED_SOURCES == 8, (
        "setup: the text must produce nine refusals and eight retained sources before this row "
        f"means anything; the gate counted {oracle.refused_count} and kept "
        f"{len(oracle.refused_sources)}"
    )

    log = make_log(tmp_path / "logs")
    record, _, _ = await run_turn(tmp_path / "data", [answer(text)], log=log)
    refusals = of_kind(read_records(Path(log.path)), SPEECH_REFUSED)
    assert refusals, "criterion 5: nine refusals and not one record"
    data = refusals[-1]["data"]

    assert data.get("count") == 9, (
        f"criterion 5: every refusal is counted, past the bound too; the record says "
        f"{data.get('count')!r}"
    )
    assert len(data.get("sources") or []) == MAX_REFUSED_SOURCES, (
        f"criterion 5: the sources stop at MAX_REFUSED_SOURCES = {MAX_REFUSED_SOURCES}; the "
        f"record kept {len(data.get('sources') or [])}"
    )
    assert data.get("truncated") is True, (
        "criterion 5: truncated is true when count > len(sources) — without it a reader cannot "
        f"tell 'eight refusals' from 'eight of nine'. The record says {data.get('truncated')!r}"
    )
    assert record.extras.get("speech_refused") == 9, (
        f"criterion 3: the turn line carries the count, not the length of the list; it says "
        f"{record.extras.get('speech_refused')!r}"
    )


async def test_an_interrupted_turn_still_records_the_refusals_it_counted(tmp_path) -> None:
    """Criterion 7, **adversarial**: the gate dies with the turn.

    ``_sync_gate`` sets ``self._gate = None`` at IDLE and an interrupted turn's gate is *dropped,
    never closed* (T-0004 A9), so a tap that reads ``refused_count`` at ``_write_record`` time
    reads from an object that is already gone and writes 0. That failure is silent: the turn line
    is written, it is well formed, and the number on it is a lie. The count has to be accumulated
    onto ``_TurnLog`` as it changes.

    The barge-in is a second ``user.speech_onset`` while the turn is SPEAKING, which is what the
    machine turns into INTERRUPTED (docs/02 §4).
    """
    text = interface_sentence("root") + "And here is what I built for you, in words. "
    oracle = refusal_oracle(text)
    assert oracle.refused_count == 1, f"setup: one refusal expected, got {oracle.refused_count}"

    log = make_log(tmp_path / "logs")
    recorder = Recorder()
    async with running(
        tmp_path / "data",
        script=[answer(text)],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    ) as rig:
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(recorder.of(AstraSpoken)),
            "the turn never reached the speaking channel, so there was nothing to barge in on",
        )
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.pump()
        await rig.until(lambda: bool(rig.records()), "the interrupted turn never wrote its line")
        record = rig.records()[0]

    assert record.outcome == "interrupted", (
        f"setup: the turn must really have been cut off; its outcome is {record.outcome!r}"
    )
    assert record.extras.get("speech_refused") == 1, (
        "criterion 7: a refusal counted before the barge-in must already be on the turn's own "
        "tally when the line is written — the gate it was counted on is dropped at IDLE and a "
        f"reader that samples at turn end reads None and writes 0. The line says "
        f"{record.extras.get('speech_refused')!r}"
    )
    refusals = of_kind(read_records(Path(log.path)), SPEECH_REFUSED)
    assert refusals, (
        "criterion 7: …and its record is in the log. An interrupted turn is the case where a "
        "human most wants to know what the assistant was about to say."
    )
    assert refusals[-1]["data"].get("sources") == oracle.refused_sources, (
        f"criterion 5: with its evidence intact; got {refusals[-1]['data'].get('sources')!r}"
    )


#: A canvas that ends in interface data with **no** terminating ``.!?`` — the realistic shape,
#: because an A2UI canvas rarely ends in a sentence mark. ``SpeechGate`` holds an unterminated
#: trailing clause in ``pending`` and only judges it at ``close()``, so this text is refused on
#: the close path and never on the feed path.
UNTERMINATED_CANVAS = 'Here it is. {"id":"root","component":"Text","text":"hello"}'


async def test_a_refusal_that_happens_only_when_the_gate_closes_is_recorded(tmp_path) -> None:
    """Criteria 5, 3 and 9 on the **close** path, which no other row reaches.

    Every other refusal row builds its text with ``interface_sentence()``, which terminates the
    clause, so the gate judges it inside ``feed()`` and ``_note_refusals`` sees the rise on that
    call. A real canvas rarely ends in ``.!?``: the guard's verdict then arrives from
    ``gate.close()`` at turn end, and a tap that reads the gate *before* that block — the one-line
    move that leaves every other row in this module green — loses the count, the record and the
    notice in one go, silently, on a well-formed turn line (T-0054 ruling A17 F2).

    The regime is asserted first with an independent gate: nothing refused at ``feed``, one
    refusal at ``close``. Without that this row could quietly slide back onto the feed path if the
    guard's rules ever change, and would then pin nothing.
    """
    gate = SpeechGate()
    gate.feed(Commit(request_id="oracle", canvas_idx=0, segment="answer", text=UNTERMINATED_CANVAS))
    at_feed = gate.refused_count
    gate.close()
    assert (at_feed, gate.refused_count) == (0, 1), (
        "setup: this row exists to exercise the close path, so the guard must refuse nothing at "
        f"feed() and exactly one clause at close(); it refused {at_feed} then {gate.refused_count}"
    )

    log = make_log(tmp_path / "logs")
    record, recorder, _ = await run_turn(tmp_path / "data", [answer(UNTERMINATED_CANVAS)], log=log)

    assert record.extras.get("speech_refused") == 1, (
        "criterion 3: the guard refused the turn's last clause when it closed, and the turn line "
        f"carries that count; the line says {record.extras.get('speech_refused')!r}. A tap that "
        "reads the gate before close() writes 0 here and nothing else in this module notices."
    )
    refusals = of_kind(read_records(Path(log.path)), SPEECH_REFUSED)
    assert refusals, (
        "criterion 5: a refusal on the close path is a refusal — the second line of defence "
        "fired and the log must say so. Nothing was recorded."
    )
    data = refusals[-1]["data"]
    assert data.get("count") == 1, f"criterion 5: count is the gate's; the record says {data!r}"
    assert data.get("sources") == list(gate.refused_sources), (
        "criterion 5: sources are the gate's retained sources byte-exact — the held clause as it "
        f"was, not re-split. Expected {list(gate.refused_sources)!r}, got {data.get('sources')!r}"
    )
    assert data.get("truncated") is False, (
        f"criterion 5: one refusal is not truncated; the record says {data.get('truncated')!r}"
    )

    fences = fence_messages(recorder.messages)
    assert len(fences) == 1, (
        "criterion 9: the tuple moved once — from (None, 0, False, 0) to (None, 0, False, 1) — "
        "and that happened after the last canvas, so it is published once, at turn end. The wire "
        f"holds {[(m.close, m.canvases, m.continued, m.speech_refused) for m in fences]}"
    )
    assert fences[-1].speech_refused == 1, (
        "criterion 9: …carrying the refusal, which is the only notice the human gets that part "
        f"of the reply was never spoken. It says {fences[-1].speech_refused!r}"
    )
    assert fences[-1].canvases == 0 and fences[-1].close is None, (
        "criterion 15's refused-only row is exactly this shape: nothing shown, nothing closed, "
        f"something refused. The message says {(fences[-1].close, fences[-1].canvases)!r}"
    )


# --- criteria 9 and 10: the wire ---------------------------------------------------------------


async def ordinary_turn(tmp_path: Path, log: Any = None) -> tuple[Any, Recorder]:
    """One healthy spoken turn: a thought, a sentence, no surface, no refusal."""
    record, recorder, _ = await run_turn(
        tmp_path,
        [thought("thinking. "), answer("Here you go. ")],
        log=log,
    )
    return record, recorder


async def test_an_ordinary_turn_publishes_no_fence_message_and_writes_no_refusal_record(
    tmp_path,
) -> None:
    """Criterion 10, negative: the cost of an observability feature is zero on the healthy path.

    The wire trace is compared message-for-message against ``ORDINARY_TURN_WIRE``, recorded from
    this very turn on the branch base before any of this task existed (see that constant). A tap
    that publishes a fence notice "just in case", or one that publishes an all-zero message at
    turn end, turns this row red and nothing else.
    """
    log = make_log(tmp_path / "logs")
    record, recorder = await ordinary_turn(tmp_path / "data", log=log)

    assert not fence_messages(recorder.messages), (
        "criterion 10: a turn with no ui commit and no refusal publishes **zero** "
        f"astra.ui.fence messages; it published "
        f"{[m for m in recorder.messages if getattr(m, 'type', None) == UI_FENCE_TYPE]}"
    )
    # Two declared differences, and nothing else may move:
    # * T-0177 B-P1 (pre-authorised pin move): the published astra.vision.frame messages (B7), shape checked, are
    #   stripped;
    # * T-0180 (narrowed): ``astra.thought`` (the turn's committed thought text, a later task's message) is left
    #   out, by exact type.
    # Every other message is compared as recorded.
    wire = tuple(
        t
        for t in (
            getattr(m, "type", type(m).__name__)
            for m in without_vision_frames(list(recorder.messages))
        )
        if t != THOUGHT_TYPE
    )
    assert wire == ORDINARY_TURN_WIRE, (
        "criterion 10: the healthy turn's wire trace is message-for-message identical to the "
        "same turn before this task (recorded at the branch base, this module's "
        "ORDINARY_TURN_WIRE), astra.vision.frame (T-0177 B-P1) and astra.thought (T-0180) aside. "
        f"It is now {wire}; unfiltered {tuple(recorder.types())}"
    )
    written = read_records(Path(log.path))
    assert not of_kind(written, SPEECH_REFUSED), (
        "criterion 10: …and **zero** speech.refused records. Nothing was refused: the log holds "
        f"{of_kind(written, SPEECH_REFUSED)}"
    )
    assert record.extras.get("speech_refused") == 0 and record.extras.get("ui_canvases") == 0, (
        "criterion 3: the three keys are still on the line, reading zero and null — 'nothing "
        f"happened' is a fact and an absent key is not. The line says {record.extras}"
    )


async def test_the_fence_message_is_published_once_per_change_and_once_at_turn_end(
    tmp_path,
) -> None:
    """Criterion 9: published when the tuple changes, and never twice for the same tuple.

    Three legs, because the three failures are different implementations:

    * two ``ui`` commits with different canvases → two messages, the second the final tuple;
    * one ``ui`` commit → **one** message, not one plus a duplicate at turn end ("once more at
      turn end *if the last published tuple is not the final one*");
    * two ``ui`` commits of the *same* canvas with the same lifecycle → still one message, since
      ``canvases`` counts distinct ``canvas_idx`` values and the tuple therefore does not move.
    """
    _, two_canvases, _ = await run_turn(
        tmp_path / "two",
        [
            ui_commit(1, close="open"),
            ui_commit(2, close="inferred", continued=True),
            answer("There you go. "),
        ],
    )
    tuples = [
        (m.close, m.canvases, m.continued, m.speech_refused)
        for m in fence_messages(two_canvases.messages)
    ]
    assert len(tuples) == 2, (
        "criterion 9: one message per change of (close, canvases, continued, speech_refused) — "
        f"two ui commits change it twice. The wire holds {tuples}"
    )
    assert tuples[0] != tuples[1], (
        f"criterion 9: …and two identical tuples are never both published: {tuples}"
    )
    assert tuples[-1] == ("inferred", 2, True, 0), (
        f"criterion 9: the last message is the final tuple of the turn; it is {tuples[-1]}"
    )

    _, one_canvas, _ = await run_turn(
        tmp_path / "one", [ui_commit(1, close="written"), answer("There you go. ")]
    )
    once = fence_messages(one_canvas.messages)
    assert len(once) == 1, (
        "criterion 9, the de-duplication leg: the turn-end publish happens only if the last "
        "published tuple is not the final one. One ui commit and no refusal means one message, "
        f"not two; the wire holds {[(m.close, m.canvases, m.continued, m.speech_refused) for m in once]}"
    )

    _, same_canvas, _ = await run_turn(
        tmp_path / "same",
        [
            ui_commit(1, close="written"),
            ui_commit(1, close="written"),
            answer("There you go. "),
        ],
    )
    repeated = fence_messages(same_canvas.messages)
    assert len(repeated) == 1, (
        "criterion 9: two consecutive identical tuples are never both published — the second "
        "commit is the same canvas with the same lifecycle, so it says nothing new. The wire "
        f"holds {[(m.close, m.canvases, m.continued) for m in repeated]}"
    )
    assert repeated[0].canvases == 1, (
        "criterion 3: ui_canvases counts *distinct* canvas_idx values that produced a ui commit, "
        f"not commits; the message claims {repeated[0].canvases}"
    )


async def test_a_refusal_after_the_last_canvas_is_published_at_turn_end(tmp_path) -> None:
    """Criterion 9's "once more at turn end" leg, from the other side.

    The gate holds the last clause until ``close()``, so a refusal can land after the last ``ui``
    commit has already been published. Without the turn-end publish the page would keep the
    pre-refusal tuple forever, and the refusal would be visible only in the log — which is the
    half of the complaint this task exists to close.
    """
    text = "Here is what I built. " + interface_sentence("root")
    assert refusal_oracle(text).refused_count == 1, "setup: one refusal expected"

    _, recorder, _ = await run_turn(tmp_path, [ui_commit(1, close="written"), answer(text)])
    fences = fence_messages(recorder.messages)
    assert fences, "criterion 9: the turn showed a surface and refused a sentence and said neither"
    assert fences[-1].speech_refused == 1, (
        "criterion 9: the final tuple carries the refusal that arrived after the last canvas; "
        f"the wire holds {[(m.close, m.canvases, m.speech_refused) for m in fences]}"
    )
    assert len(fences) == 2, (
        "criterion 9's bound is one message per ui commit plus one per turn: one for the canvas, "
        f"one at turn end for the refusal. The wire holds {len(fences)}: "
        f"{[(m.close, m.canvases, m.speech_refused) for m in fences]}"
    )


async def test_a_late_commit_from_the_previous_request_is_not_tallied_on_the_next_turn(
    tmp_path,
) -> None:
    """Criteria 3 and 9, **adversarial**: the tally belongs to the turn that produced the canvas.

    ``_note_commit_fence`` ignores a commit whose generate id is not ``active_generate`` — the
    same commit ``_note_event`` refuses to gate (``_log_ui_not_gated``), so the turn line, the
    pane and the gate's verdict agree about what was shown. Without the guard a canvas the *last*
    request produced is counted on *this* turn's line and published as this turn's surface: the
    page would be told a surface exists for a turn that never had one, which is the exact class of
    lie this task exists to end (T-0054 ruling A17 F3).

    **Why the arrival is injected at the tap rather than scripted.** The fakes cannot deliver a
    commit that outlives its turn: the machine stays SPEAKING until ``llm.done``, so holding a
    generate open holds its turn open, and a barge-in cancels the generate task outright
    (``abort_request``). So turn 1's generate id is taken from turn 1's own ``llm.commit`` record
    — the session's id, never one invented here — and handed to the tap while turn 2 is really
    open, with turn 2's generate really live (``rig.llm.hold()`` keeps the session in THINKING, so
    ``active_generate`` is set and the two ids differ). Everything asserted afterwards is the turn
    line and the wire.

    The control call at the end is what stops this row passing for the wrong reason: the same tap,
    the same shape of commit, this turn's id — and it must tally. A row that only asserts absence
    would also pass if the tap were dead.
    """
    log = make_log(tmp_path / "logs")
    recorder = Recorder()
    rig = make_rig(
        tmp_path / "data",
        script=[answer("Here it is. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    )
    tally = getattr(rig.session, "_note_commit_fence", None)
    assert callable(tally), (
        "criteria 3 and 9: the session tallies a ui commit's fence where the Commit is — the "
        "lifecycle lives on the Commit and on nothing the machine sees — so there is a tap that "
        "takes the commit and the generate id that produced it. Session defines no "
        "_note_commit_fence, so this row has no seam to hand a late canvas to; it holds "
        f"{[n for n in dir(rig.session) if 'fence' in n]}"
    )
    await rig.session.start()
    try:
        await rig.ask(QUESTION)
        await rig.speak_out()
        first_turn = of_kind(read_records(Path(log.path)), "llm.commit")
        assert first_turn, "setup: turn 1 committed nothing, so it has no generate id to reuse"
        stale_generate = first_turn[0]["data"]["request_id"]

        # Turn 2, with its generate held open so it is the live one while the late canvas lands.
        rig.llm.hold()
        await rig.ask(QUESTION)
        live_generate = rig.state.active_generate
        assert live_generate and live_generate != stale_generate, (
            f"setup: turn 2 must have its own live generate; the session says "
            f"{live_generate!r} against turn 1's {stale_generate!r}"
        )

        tally(ui_commit(7, close="abandoned"), stale_generate)
        await rig.pump(2)
        assert not fence_messages(recorder.messages), (
            "criteria 9 and 10: a canvas from the previous request says nothing about this turn, "
            "so this turn publishes nothing — the page must not be shown a fence for a surface "
            f"this turn never had. The wire holds "
            f"{[(m.turn_id, m.close, m.canvases) for m in fence_messages(recorder.messages)]}"
        )

        # The control: the same tap, the same shape of commit, this turn's own generate id.
        tally(ui_commit(1, close="written"), live_generate)
        await rig.pump(2)
        live = fence_messages(recorder.messages)
        assert len(live) == 1 and live[0].canvases == 1 and live[0].close == "written", (
            "setup/criterion 9: this turn's own canvas is tallied and published once — if it is "
            "not, the row above proved nothing about the guard. The wire holds "
            f"{[(m.turn_id, m.close, m.canvases) for m in live]}"
        )

        rig.llm.release()
        await rig.speak_out()
        records = rig.records()
    finally:
        await rig.session.aclose()

    assert len(records) == 2, f"setup: two turn lines expected; the file holds {len(records)}"
    stale_turn, next_turn = records
    assert (next_turn.extras.get("ui_canvases"), next_turn.extras.get("ui_fence")) == (
        1,
        "written",
    ), (
        "criterion 3: ui_canvases counts the distinct canvases *this turn's* generate produced — "
        "one, the control — and ui_fence is that canvas's close. With the stale canvas counted "
        f"the line reads two canvases and a fence nobody on this turn opened; it reads "
        f"{(next_turn.extras.get('ui_canvases'), next_turn.extras.get('ui_fence'))}"
    )
    assert (stale_turn.extras.get("ui_canvases"), stale_turn.extras.get("ui_fence")) == (
        0,
        None,
    ), (
        "criterion 3: …and the turn that really owned the late canvas is already written and "
        f"unchanged — a record is not retroactively edited. It reads "
        f"{(stale_turn.extras.get('ui_canvases'), stale_turn.extras.get('ui_fence'))}"
    )


# --- criteria 8, 11 and 12: the contract, the outbox, the journal ------------------------------


def test_the_fence_message_is_server_to_client_only_and_has_no_replay_field() -> None:
    """Criterion 8: the model, its bounds, its direction, and the field it deliberately lacks.

    It has **no ``replay`` field** and is not journalled, because ``hud_server.py`` is T-0047's
    file (criterion 8, and "What stays invisible" item 1). A ``replay`` field here would be a
    promise the server cannot keep.
    """
    model = ui_fence_model()
    fields = {name: f for name, f in model.model_fields.items() if name != "type"}

    assert set(fields) == {"turn_id", "canvases", "close", "continued", "speech_refused"}, (
        "criterion 8 fixes the shape: AstraUiFence(turn_id, canvases, close=None, "
        f"continued=False, speech_refused=0). It has {sorted(fields)}"
    )
    assert "replay" not in model.model_fields, (
        "criterion 8: no replay field — this message is not journalled (TurnJournal.record "
        "returns early for it by construction) and a replay flag would claim otherwise"
    )
    assert model in ASTRA_MESSAGE_TYPES, (
        "criterion 8: the model joins ASTRA_MESSAGE_TYPES, which is what the codec round-trip "
        "test is parametrised over (contracts/protocol.py's own docstring)"
    )
    assert UI_FENCE_TYPE in set(SERVER_TO_CLIENT), (
        f"criterion 8: server->client; SERVER_TO_CLIENT holds {sorted(SERVER_TO_CLIENT)}"
    )
    assert UI_FENCE_TYPE not in set(CLIENT_TO_SERVER), (
        "criterion 8: …and never client->server — a page that sent one would be refused as "
        "wrong_direction"
    )
    assert not (set(CLIENT_TO_SERVER) & set(SERVER_TO_CLIENT)), (
        "criterion 8: the two direction sets stay disjoint"
    )
    every = {m.model_fields["type"].default for m in ASTRA_MESSAGE_TYPES}
    assert every == set(CLIENT_TO_SERVER) | set(SERVER_TO_CLIENT), (
        "criterion 8: …and jointly cover every model. Uncovered: "
        f"{sorted(every - (set(CLIENT_TO_SERVER) | set(SERVER_TO_CLIENT)))}; "
        f"named but modelless: {sorted((set(CLIENT_TO_SERVER) | set(SERVER_TO_CLIENT)) - every)}"
    )

    message = model(turn_id="t1", canvases=2, close="inferred", continued=True, speech_refused=1)
    assert decode(encode(message)) == message, (
        "criterion 8: it round-trips through the production codec unchanged"
    )
    for bad in ({"canvases": -1}, {"speech_refused": -1}, {"close": "salvaged"}, {"replay": True}):
        try:
            model(turn_id="t1", **{"canvases": 0, **bad})
        except Exception:
            continue
        raise AssertionError(
            f"criterion 8: extra='forbid' holds and the bounds are real; AstraUiFence accepted "
            f"{bad!r}. An unknown close value must be refused here, or the page is asked to "
            "render a state nobody defined."
        )


def test_the_fence_message_is_not_droppable() -> None:
    """Criterion 11: a full outbox closes the socket for it, exactly as for a surface.

    Only ``astra.turn.state`` may be dropped (T-0028 criterion 12) because it is idempotent and
    latest-wins. A dropped refusal notice is the failure this task exists to end, and making the
    fence droppable is T-0047's rule change in T-0047's file.
    """
    model = ui_fence_model()
    outbox_cls = require_symbol("Outbox")
    fence = model(turn_id="t1", canvases=1, close="abandoned")

    outbox = outbox_cls(maxsize=1)
    surface_like = outbox.offer(fence)
    assert surface_like is True, "setup: the first message fits in a bounded outbox of one"
    assert outbox.offer(fence) is False, (
        "criterion 11: a full outbox refuses an astra.ui.fence and the caller closes that "
        "connection — the droppable set stays {astra.turn.state}. It returned True, which means "
        "the notice was silently discarded."
    )
    assert outbox.dropped == 0, (
        f"criterion 11: nothing was dropped; the outbox counted {outbox.dropped}"
    )


def test_the_fence_message_is_not_journalled() -> None:
    """Criterion 8's second half: ``hud_server.py`` has zero diff, so the notice is not replayed.

    ``TurnJournal.record`` returns early for any message that is not one of its five journalled
    types, so this is a property of the *existing* code and the row's job is to keep it true —
    an implementer who reached into T-0047's file to journal the fence has left this task (and
    the consequence, a page that reloads mid-turn never seeing the notice, is stated in T-0054's
    "What stays invisible" item 1).
    """
    model = ui_fence_model()
    journal_cls = require_symbol("TurnJournal")
    journal = journal_cls()

    journal.record(model(turn_id="t1", canvases=2, close="inferred", continued=True))
    replayed = [getattr(m, "type", None) for m in journal.messages()]
    assert UI_FENCE_TYPE not in replayed, (
        "criterion 8: astra.ui.fence is not journalled and is never replayed; the journal would "
        f"hand a reconnecting page {replayed}"
    )


async def test_the_surface_gate_is_still_asked_with_final_false(tmp_path, monkeypatch) -> None:
    """Criterion 12 / D3: ``_gate_surface`` still calls ``UiGate.gate(..., final=False)``.

    Not cosmetic and not this task's call to make: ``validate_messages(..., final=True)`` promotes
    forward references from warnings to **errors** (``local_astra/ui/a2ui_schema.py:428``), so
    flipping it changes what the gate *accepts*, not only what it reports — with T-0010's
    progressive-render property and T-0052 ruling A6's seven-component surface at stake. A change
    to acceptance belongs in a task that owns ``ui_gate.py`` and can measure it.
    """
    rig = make_rig(
        tmp_path,
        script=[ui_commit(1, close="written"), answer("There you go. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=Recorder(),
    )
    seen: list[bool] = []
    real = rig.session.ui_gate.gate

    def spy(text: str, **kwargs: Any) -> Any:
        seen.append(bool(kwargs.get("final")))
        return real(text, **kwargs)

    monkeypatch.setattr(rig.session.ui_gate, "gate", spy)
    await rig.session.start()
    try:
        await rig.ask(QUESTION)
        await rig.speak_out()
    finally:
        await rig.session.aclose()

    assert seen, "setup: the turn's ui commit never reached the gate"
    assert not any(seen), (
        f"criterion 12 / D3: every UiGate.gate call of a turn passes final=False; this turn "
        f"passed {seen}. Flipping it is a different task (it changes what the gate accepts)."
    )


def test_the_pure_modules_this_task_may_not_touch_are_byte_identical() -> None:
    """Criterion 6: zero diff over ``speech_gate.py`` and ``llm_client.py``.

    "The tap reads the gate's counters from ``session.py``; it never calls into the gate to make
    it report, and ``_segment`` stays an adapter that hands the gate's output back unmodified"
    (criterion 6, quoting T-0052 criterion 10 verbatim). An implementer who wants to change the
    gate's API to make the tap easier has found a finding, not a licence.

    The two digests are of the files as T-0052 merged them; a later task that legitimately edits
    either updates the constant in its own PR, exactly as ``TURN_MACHINE_SHA256`` does in
    ``tests/unit/test_session_loop.py``.
    """
    expected = {
        "local_astra/conductor/speech_gate.py": SPEECH_GATE_SHA256,
        "local_astra/conductor/llm_client.py": LLM_CLIENT_SHA256,
    }
    for relative, digest in expected.items():
        got = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        assert got == digest, (
            f"criterion 6: {relative} must be byte-identical through this task (sha256 {got} != "
            f"{digest}). Both are pure modules: nothing logs from inside them, and the counters "
            "are read from session.py. Escalation trigger in T-0054's Blast radius."
        )


def test_the_session_does_not_log_from_inside_the_segmenter_adapter() -> None:
    """Criterion 6's other half: ``_segment`` hands the gate's output back unmodified.

    ``_segment`` runs inside ``machine.step`` and the machine checks
    ``"".join(sources) + remainder == text`` on every call; T-0052 ruling A20 measured what a
    mistake there costs — a silent turn, no log line, and a probe stuck in SPEAKING for 20 000 ms.
    So the tap may observe the counters, and it may not rewrite what ``_segment`` returns.
    """
    source = (REPO / "local_astra" / "conductor" / "session.py").read_text()
    tree = ast.parse(source)
    segment = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_segment"
        ),
        None,
    )
    assert segment is not None, (
        "session.py must still define _segment (machine.segment = self._segment)"
    )
    returns = [n for n in ast.walk(segment) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, "_segment must still return (sentences, pending)"
    for node in returns:
        rendered = ast.unparse(node)
        assert "gate.pending" in rendered and "s.source" in rendered and "s.text" in rendered, (
            "criterion 6: _segment returns the gate's own sentences and the gate's own pending "
            f"buffer, unmodified. It now returns {rendered!r} — a tap that filters or rewrites "
            "the sentences breaks the machine's conservation law, which fails silently."
        )
