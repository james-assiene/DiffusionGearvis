"""The taps: what a scripted turn leaves in the log, and the three states it must tell apart.

T-0043 group A, criteria 8-13 and 24. This is the module the task exists for: on 2026-09-15 the
model said it had illustrated something, nothing appeared, every ``turns.jsonl`` line said
``ui_valid: null``, and the three hypotheses — no ``ui`` segment was ever committed; one was
committed and the gate refused it; one was committed and the session dropped it before the gate —
could not be told apart. The headline rows below run one session per hypothesis and assert the
log reads differently in each.

Oracles, none of them the implementation: the machine's own call log (``RecordingMachine``) for
criterion 8, the fake backend's script for criterion 10, ``UiGate``'s own verdict for criterion
11, and — for criterion 24 — the same scripted turn run twice, with and without a log, compared
message by message on the wire.

Regime: ``ScriptedClock``, ``tests/fakes/``, no service, no device, no measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_astra.contracts import events as E
from local_astra.telemetry.turns_log import TurnRecord
from tests.unit._hud_rig import Recorder, answer, frame, running, thought, ui
from tests.unit._session_log_rig import kinds, make_log, of_kind, read_records

QUESTION = "can you illustrate one half"

#: A ``ui`` canvas the gate cannot repair into a single A2UI message (hypothesis (b)). The gate's
#: own answer to it is ``surface=None`` with a non-empty ``problems`` — asserted here, never
#: re-derived.
UNREPAIRABLE = "here is a diagram of one half, trust me"

#: What the log says about a turn's A2UI channel. Criterion 11: the log alone distinguishes the
#: three states. This function *reads* the log; it computes nothing the session computes.
NO_UI_SEGMENT = "no ui segment was committed"
COMMITTED_NO_SURFACE = "a ui segment was committed and no surface was emitted"
SURFACE_EMITTED = "a surface was emitted"


def ui_verdict(records: list[dict[str, Any]]) -> str:
    commits = [r for r in of_kind(records, "llm.commit") if r["data"].get("segment") == "ui"]
    gates = of_kind(records, "ui.gate")
    if not commits and not gates:
        return NO_UI_SEGMENT
    if any(g["data"].get("surface_emitted") for g in gates):
        return SURFACE_EMITTED
    return COMMITTED_NO_SURFACE


@dataclass
class Turn:
    """What one scripted, logged turn left behind."""

    records: list[dict[str, Any]] = field(default_factory=list)
    machine_log: list[Any] = field(default_factory=list)
    commits: list[Any] = field(default_factory=list)
    tts_texts: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    published: Recorder | None = None
    path: Path | None = None


async def logged_turn(
    tmp_path: Path,
    script: list[Any],
    *,
    question: str = QUESTION,
    rewrite_ui_request_id: str | None = None,
) -> Turn:
    """One voice turn over the rig with a ``SessionLog`` attached, read back after ``aclose``.

    ``rewrite_ui_request_id`` intercepts ``Session.submit`` between the ``_run_generate`` tap and
    the machine, which is exactly the shape of the silent drop in ``_note_event``: the backend
    delivered the commit (so criterion 10's record must exist) and the session then decided it
    belonged to a request that is no longer active.
    """
    log = make_log(tmp_path / "logs")
    recorder = Recorder()
    out = Turn(published=recorder)
    async with running(
        tmp_path / "data",
        script=script,
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=recorder,
        session_log=log,
    ) as rig:
        if rewrite_ui_request_id is not None:
            real = rig.session.submit

            def submit(event: Any) -> None:
                if isinstance(event, E.LlmCommit) and event.segment == "ui":
                    event = event.model_copy(update={"request_id": rewrite_ui_request_id})
                real(event)

            rig.session.submit = submit  # type: ignore[method-assign]
        await rig.ask(question)
        await rig.speak_out()
        out.machine_log = list(rig.machine.log)
        out.commits = list(rig.llm.script)
        out.tts_texts = list(rig.tts.texts())
        out.lines = rig.lines()
    out.path = Path(log.path)
    out.records = read_records(log.path)
    return out


async def test_a_scripted_turn_records_its_request_commits_steps_speech_and_record(tmp_path):
    """Criteria 8-13: the turn is reconstructible from the log alone, in order."""
    script = [thought("I should answer briefly. "), answer("One half. "), ui()]
    turn = await logged_turn(tmp_path, script)
    records = turn.records

    requests = of_kind(records, "llm.request")
    phases = {r["data"]["phase"] for r in requests}
    assert "generate" in phases and "prefill" in phases, (
        f"criterion 9: every LLM request is recorded before it is sent, prefills included; "
        f"phases={phases}"
    )
    generate = next(r for r in requests if r["data"]["phase"] == "generate")
    assert {"request_id", "options", "prompt", "drop_history_turns"} <= set(generate["data"]), (
        f"criterion 9 fixes the payload; got {sorted(generate['data'])}"
    )
    assert QUESTION in json.dumps(generate["data"]["prompt"]), (
        "criterion 9: the prompt's text parts are verbatim, so one can see what was asked"
    )

    commits = of_kind(records, "llm.commit")
    assert len(commits) == len(script), (
        f"criterion 10: one llm.commit per commit the backend yielded; the fake yielded "
        f"{len(script)} and the log holds {len(commits)} ({[c['data'].get('segment') for c in commits]})"
    )
    assert [c["data"]["segment"] for c in commits] == [c.segment for c in script], (
        "…in the order they arrived, with their segments"
    )
    assert [c["data"]["text"] for c in commits] == [c.text for c in script], (
        "…and their text untruncated"
    )
    assert commits[0]["data"]["text"] == script[0].text, (
        "criterion 6: the thought canvas is in the log verbatim — that is the point of the log"
    )

    steps = of_kind(records, "machine.step")
    assert len(steps) == len(turn.machine_log), (
        f"criterion 8: one machine.step record per step the machine took; the machine took "
        f"{len(turn.machine_log)} and the log holds {len(steps)}"
    )
    for record, (event, _state, cmds) in zip(steps, turn.machine_log, strict=True):
        data = record["data"]
        assert {"event", "state_before", "state_after", "commands"} <= set(data), (
            f"criterion 8 fixes the payload; got {sorted(data)}"
        )
        assert event.type in json.dumps(data["event"]), (
            f"…the event that was stepped ({event.type}); the record says {data['event']!r:.200}"
        )
        assert len(data["commands"]) == len(cmds), (
            f"…and every command it produced: the machine returned {[c.type for c in cmds]}, the "
            f"record holds {data['commands']!r:.300}"
        )
        text = json.dumps(data["commands"])
        for command in cmds:
            assert command.type in text, f"…including {command.type}; the record holds {text:.300}"
    assert any("turn_seq" in key for key in of_kind(records, "machine.step")[0]["data"]), (
        "criterion 8: turn_seq_before/after are in the payload — a step that opened a turn is "
        f"otherwise unreadable; got {sorted(of_kind(records, 'machine.step')[0]['data'])}"
    )

    utterances = of_kind(records, "tts.utterance")
    assert [u["data"]["text"] for u in utterances] == turn.tts_texts, (
        f"criterion 12: one tts.utterance per enqueued utterance, with what was said; the TTS "
        f"backend was asked for {turn.tts_texts} and the log holds "
        f"{[u['data'].get('text') for u in utterances]}"
    )
    assert all({"utterance_id", "source", "kind"} <= set(u["data"]) for u in utterances), (
        f"criterion 12 fixes the payload; got {sorted(utterances[0]['data'])}"
    )
    assert of_kind(records, "tts.played"), (
        f"criterion 12: the played report is recorded too; kinds={sorted(set(kinds(records)))}"
    )

    written = of_kind(records, "turn.record")
    assert len(written) == 1 and len(turn.lines) == 1, (
        f"criterion 13: one turn.record per turn line; {len(written)} records for "
        f"{len(turn.lines)} lines"
    )
    logged = TurnRecord.model_validate(written[0]["data"])
    on_disk = TurnRecord.from_line(turn.lines[0])
    assert logged == on_disk, (
        "criterion 13: the log carries the complete TurnRecord, so a log stands alone for "
        f"analysis; the log says {logged!r:.300} and turns.jsonl says {on_disk!r:.300}"
    )

    order = kinds(records)
    assert order.index("llm.request") < order.index("llm.commit") < order.index("turn.record"), (
        f"criterion 9: the request is recorded *before* it is sent; the file reads {order}"
    )
    assert of_kind(records, "llm.done"), "criterion 10: llm.done is recorded likewise"


async def test_the_log_alone_tells_the_three_ui_states_apart(tmp_path):
    """Criterion 11, the headline: (a) no ui segment, (b) refused, (c) emitted.

    One session per hypothesis. The verdicts must differ — a log in which all three read the same
    is exactly the ``ui_valid: null`` that made tonight's question unanswerable.
    """
    only_answer = await logged_turn(
        tmp_path / "a", [answer("Here is a diagram of one half. ")], question="illustrate a half"
    )
    refused = await logged_turn(
        tmp_path / "b",
        [answer("Here it is. "), ui(text=UNREPAIRABLE)],
        question="illustrate a half",
    )
    emitted = await logged_turn(
        tmp_path / "c", [answer("Here it is. "), ui()], question="illustrate a half"
    )

    assert ui_verdict(only_answer.records) == NO_UI_SEGMENT, (
        f"(a) the model never emitted a ui segment: no llm.commit with segment ui and no ui.gate "
        f"record; the log holds {sorted(set(kinds(only_answer.records)))}"
    )
    assert not of_kind(only_answer.records, "ui.gate"), (
        "…and nothing invents a gate verdict for a canvas that never existed"
    )

    gates = of_kind(refused.records, "ui.gate")
    assert len(gates) == 1, (
        f"(b) one gate verdict for the one ui commit; the log holds {len(gates)}"
    )
    verdict = gates[0]["data"]
    assert verdict["ok"] is False and verdict["surface_emitted"] is False, (
        f"(b) the gate refused the canvas; the record says {verdict!r}"
    )
    assert verdict["problems"], (
        f"(b) …with the gate's own problems list, which is the answer to 'why did nothing "
        f"appear'; the record says {verdict!r}"
    )
    assert verdict["raw_len"] == len(UNREPAIRABLE), (
        f"(b) …and the length of the canvas it judged ({len(UNREPAIRABLE)}); it says "
        f"{verdict.get('raw_len')!r}"
    )
    assert ui_verdict(refused.records) == COMMITTED_NO_SURFACE

    emitted_gates = of_kind(emitted.records, "ui.gate")
    assert len(emitted_gates) == 1 and emitted_gates[0]["data"]["surface_emitted"] is True, (
        f"(c) the surface went out; the record says {emitted_gates[0]['data'] if emitted_gates else None!r}"
    )
    assert emitted_gates[0]["data"]["surface_messages"] == 2, (
        f"(c) …carrying the message count of the surface published (the rig's canvas is a "
        f"createSurface plus an updateComponents); it says "
        f"{emitted_gates[0]['data']['surface_messages']!r}"
    )
    assert ui_verdict(emitted.records) == SURFACE_EMITTED

    verdicts = {
        ui_verdict(only_answer.records),
        ui_verdict(refused.records),
        ui_verdict(emitted.records),
    }
    assert len(verdicts) == 3, (
        f"criterion 11: the three states are distinguishable from the log alone; they read "
        f"{verdicts}"
    )


async def test_a_ui_commit_the_session_drops_is_still_in_the_log(tmp_path):
    """Criterion 10 and 11, adversarial: the silent-drop branch of ``_note_event``.

    The backend delivered a ``ui`` canvas; by the time the event reached the machine its
    ``request_id`` was no longer the active generate, so the gate never saw it and the turn line
    says ``ui_valid: null`` — tonight's symptom exactly. The log must say both things: the commit
    arrived, and it was not gated, with the reason. A log that is silent here lies by omission.
    """
    script = [answer("Here it is. "), ui()]
    turn = await logged_turn(tmp_path, script, rewrite_ui_request_id="stale-request-1")

    commits = of_kind(turn.records, "llm.commit")
    assert len(commits) == len(script), (
        f"criterion 10: the commit is recorded where it is received, before submit, so a commit "
        f"the session later drops is still in the log; {len(commits)} of {len(script)} are"
    )
    assert any(c["data"]["segment"] == "ui" for c in commits), (
        f"…including the ui one; the log holds {[c['data']['segment'] for c in commits]}"
    )

    gates = of_kind(turn.records, "ui.gate")
    assert len(gates) == 1, (
        f"criterion 11: a ui commit that is not gated still gets a ui.gate record saying so; the "
        f"log holds {len(gates)} (kinds={sorted(set(kinds(turn.records)))})"
    )
    data = gates[0]["data"]
    assert data.get("gated") is False, (
        f"…with gated=false, not a silent absence; the record says {data!r}"
    )
    assert isinstance(data.get("reason"), str) and data["reason"], (
        f"…and the reason it was not gated; the record says {data!r}"
    )
    assert "stale-request-1" in json.dumps(data) or "request" in data["reason"], (
        f"…which names the request mismatch that dropped it; the record says {data!r}"
    )
    assert TurnRecord.from_line(turn.lines[0]).extras["ui_valid"] is None, (
        "the turn line is unchanged — still the uninformative null; the log is what changed"
    )
    assert ui_verdict(turn.records) == COMMITTED_NO_SURFACE, (
        "criterion 11: and this reads as (b), not as (a): a canvas *was* committed"
    )


async def test_the_commit_is_in_the_log_before_the_machine_ever_sees_it(tmp_path):
    """Criterion 10: the tap sits *before* ``self.submit``, not beside it.

    The consequence — a commit the session drops is still in the log — has its own row above, but
    it holds for one particular drop (a stale ``request_id``). The ordering itself is what makes
    it hold for every future drop, and nothing else would notice if the two lines swapped: a tap
    after ``submit`` records the same commits for every turn that does not crash in between.

    The oracle is the offer order as seen from inside ``submit``: at the moment the machine is
    handed the commit, the log must already have been offered a ``llm.commit`` carrying that text.
    ``record`` is watched, not the file, because the writer task has not run yet at that instant —
    which is the whole point of criterion 3.
    """
    script = [thought("thinking. "), answer("One half. "), ui()]
    log = make_log(tmp_path / "logs")
    offered: list[tuple[str, Any]] = []
    at_submit: list[tuple[str, list[str]]] = []

    real_record = log.record

    def spy(kind: str, data: Any, **kwargs: Any) -> None:
        offered.append((kind, data))
        real_record(kind, data, **kwargs)

    log.record = spy  # type: ignore[method-assign]
    async with running(
        tmp_path / "data",
        script=list(script),
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=Recorder(),
        session_log=log,
    ) as rig:
        real_submit = rig.session.submit

        def submit(event: Any) -> None:
            if isinstance(event, E.LlmCommit):
                at_submit.append(
                    (event.text, [d.get("text") for k, d in offered if k == "llm.commit"])
                )
            real_submit(event)

        rig.session.submit = submit  # type: ignore[method-assign]
        await rig.ask(QUESTION)
        await rig.speak_out()

    assert [text for text, _ in at_submit] == [c.text for c in script], (
        f"the rig delivered every scripted commit to the machine; it saw "
        f"{[text for text, _ in at_submit]}"
    )
    late = [text for text, already in at_submit if text not in already]
    assert not late, (
        f"criterion 10: the llm.commit record is offered where the commit is received, *before* "
        f"self.submit — {late} reached the machine with nothing in the log yet, so a session that "
        f"dropped or died on that event would have lost it silently"
    )


async def test_an_exception_the_session_swallows_becomes_an_error_record(tmp_path):
    """Criterion 13: ``error`` records carry what the session swallows, with the repr.

    Two sites, because they swallow differently: ``_safely`` (a client call with no event of its
    own — the failure is appended to ``failures`` and the pump carries on) and ``_run_generate``'s
    except arm (the backend died mid-answer; the machine hears ``llm.error``). Before this task
    both were invisible after the fact: ``turns.jsonl`` says ``outcome`` and nothing about why.
    """
    log = make_log(tmp_path / "safely")
    async with running(
        tmp_path / "a", script=[answer("Yes. ")], session_log=log, publish=Recorder()
    ) as rig:

        async def refuse(t_eot_ms: int) -> int:
            raise RuntimeError("the marker never came back")

        rig.stt.mark = refuse  # type: ignore[method-assign]
        rig.session._on_decision(("end_of_turn", 900, 0.87), 1000)
        await rig.pump(rounds=8)

    errors = of_kind(read_records(log.path), "error")
    assert len(errors) == 1, (
        f"criterion 13: the failure _safely swallowed is one error record; the log holds "
        f"{[e['data'] for e in errors]}"
    )
    assert errors[0]["data"]["what"] == "stt.mark", (
        f"…naming the call that failed, which is the only thing that says *which* client died; "
        f"the record says {errors[0]['data']!r}"
    )
    assert "the marker never came back" in errors[0]["data"]["error"], (
        f"…and the exception's repr, not a swallowed 'something went wrong'; the record says "
        f"{errors[0]['data']!r}"
    )
    assert "RuntimeError" in errors[0]["data"]["error"], (
        f"…the type included (repr, not str); the record says {errors[0]['data']['error']!r}"
    )

    generate_log = make_log(tmp_path / "generate")
    async with running(
        tmp_path / "b",
        script=[answer("One half. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=Recorder(),
        session_log=generate_log,
    ) as rig:
        backend = rig.llm.generate

        async def explode(*args: Any, **kwargs: Any) -> Any:
            """The realistic shape: the stream dies after one canvas has been delivered."""
            async for commit in backend(*args, **kwargs):
                yield commit
            raise RuntimeError("the backend fell over mid-answer")

        rig.llm.generate = explode  # type: ignore[method-assign]
        await rig.ask(QUESTION)
        await rig.pump(rounds=12)
        lines = rig.lines()

    records = read_records(generate_log.path)
    generate_errors = [
        e for e in of_kind(records, "error") if e["data"].get("what") == "llm.generate"
    ]
    assert len(generate_errors) == 1, (
        f"criterion 13: the backend's failure is in the log at the generate arm; the log holds "
        f"{[e['data'].get('what') for e in of_kind(records, 'error')]} "
        f"(kinds={sorted(set(kinds(records)))})"
    )
    data = generate_errors[0]["data"]
    assert (
        "the backend fell over mid-answer" in data["error"] and "RuntimeError" in data["error"]
    ), f"…with the exception's repr; the record says {data!r}"
    assert data.get("request_id"), (
        f"…and the request it belonged to, so it joins the llm.request above it; it says {data!r}"
    )
    assert of_kind(records, "llm.request"), (
        "…and the request itself is in the log, so one can see what was sent before it failed"
    )
    order = kinds(records)
    assert order.index("llm.commit") < order.index("error"), (
        f"…after the canvas that did arrive, so the log says how far the answer got; the file "
        f"reads {order}"
    )
    assert not of_kind(records, "llm.done"), (
        "…and no llm.done, because the stream never finished — the two are not interchangeable"
    )
    assert lines, (
        "criterion 13: and the turn still ended and still wrote its line — an error record is "
        "telemetry, not a second failure"
    )


async def test_a_tts_failure_is_in_the_log_and_not_only_as_a_typed_event(tmp_path):
    """Criterion 13 and ruling A11: ``_speak_one``'s except arm owes an ``error`` record.

    ``tts.speak`` raising is caught in ``_speak_one`` and turned into ``E.TtsError``, which the
    log records as a ``machine.step``. That step says what the machine *did* — degrade, end the
    turn — and carries ``str(exc)``; it does not say what failed. The exception's type, the thing
    that tells a hung socket from a refused voice from a bug in the client, survives only in the
    repr, and the repr exists only if this arm records one. The human's first report from the desk
    was "the voice is sometimes broken"; this is the record that answers it.

    Both, not one instead of the other: the typed event must still reach the machine exactly as it
    does today (criterion 24 — logging is observation only).

    Asserted here is exactly what ruling A11 owes — the ``what`` string and the repr. The
    utterance id would let the record join the ``tts.utterance`` above it and is worth carrying,
    but the ruling does not demand it and neither does this row.
    """
    log = make_log(tmp_path / "logs")
    published = Recorder()
    async with running(
        tmp_path / "data",
        script=[answer("One half. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
        session_log=log,
    ) as rig:

        async def hang_up(utterance_id: str, text: str, voice: str = "default") -> Any:
            raise RuntimeError("the voice service hung up")

        rig.tts.speak = hang_up  # type: ignore[method-assign]
        await rig.ask(QUESTION)
        await rig.pump(rounds=12)

    records = read_records(log.path)
    steps = [
        r for r in of_kind(records, "machine.step") if r["data"]["event"].get("type") == "tts.error"
    ]
    assert len(steps) == 1, (
        f"the typed event still reached the machine, once, as it does today; the log holds "
        f"{[r['data']['event'].get('type') for r in of_kind(records, 'machine.step')]}"
    )

    failures = [r for r in of_kind(records, "error") if r["data"].get("what") == "tts.speak"]
    assert len(failures) >= 1, (
        f"criterion 13 (ruling A11): the exception _speak_one swallowed is an error record naming "
        f"the client call, as llm.generate, llm.prefill and stt.mark already are; the log holds "
        f"error records for {[r['data'].get('what') for r in of_kind(records, 'error')]} "
        f"(kinds={sorted(set(kinds(records)))})"
    )
    data = failures[0]["data"]
    assert "RuntimeError" in data["error"] and "the voice service hung up" in data["error"], (
        f"…carrying the exception's repr; the record says {data!r}"
    )
    assert "RuntimeError" not in json.dumps(steps[0]), (
        "…which is what the machine.step cannot give: E.TtsError carries str(exc), so the type "
        f"is lost there. The step says {steps[0]['data']['event']!r:.200}"
    )


async def test_the_hud_error_the_page_is_shown_is_an_error_record_too(tmp_path):
    """Criterion 13 and ruling A11: the ``hud.error`` command arm.

    Read first, then written for what is reachable. ``CommandExecutor.hud_error`` swallows nothing
    of its own — it calls ``publish_turn_error`` and returns — so what this arm carries is not a
    caught exception but the failure message the machine decided the page should be shown, which
    today lives *only* in the page's error strip and inside a command dump nested in a
    ``machine.step``. Neither survives a grep for ``kind == "error"``, which is how a session log
    is read (and how group C's ``session_log_summary.py`` will read it): a failure that is only
    discoverable by knowing which command type to look for inside which step is a failure the log
    does not report.

    ``E.SttError`` is the trigger because it reaches ``C.HudError`` from any state
    (``turn_machine._on_stt_error``) and because the session has no other tap on that path — so
    this row cannot pass on somebody else's record.
    """
    log = make_log(tmp_path / "logs")
    published = Recorder()
    message = "the transcriber closed the socket"
    async with running(
        tmp_path / "data", script=[answer("Yes. ")], session_log=log, publish=published
    ) as rig:
        rig.submit(E.SttError(t_ms=rig.clock.now_ms, message=message))
        await rig.pump(rounds=8)

    records = read_records(log.path)
    commanded = [
        r
        for r in of_kind(records, "machine.step")
        if any(c.get("type") == "hud.error" for c in r["data"]["commands"])
    ]
    assert len(commanded) == 1, (
        f"the machine produced one hud.error command for the failure; the log holds "
        f"{[[c.get('type') for c in r['data']['commands']] for r in of_kind(records, 'machine.step')]}"
    )
    assert any(message in getattr(m, "message", "") for m in published.messages), (
        f"…and the page was still shown it, unchanged (criterion 24); the page got "
        f"{published.types()}"
    )

    shown = [r for r in of_kind(records, "error") if r["data"].get("what") == "hud.error"]
    assert len(shown) >= 1, (
        f"criterion 13 (ruling A11): what the page is told went wrong is an error record, so a "
        f"reader who greps kind==error sees every failure of the session; the log holds error "
        f"records for {[r['data'].get('what') for r in of_kind(records, 'error')]} "
        f"(kinds={sorted(set(kinds(records)))})"
    )
    assert message in json.dumps(shown[0]["data"]), (
        f"…with the message the page was shown, verbatim; the record says {shown[0]['data']!r}"
    )


async def test_the_endpointers_decisions_are_recorded(tmp_path):
    """Criterion 12: ``stt.decision`` at the endpointer's decision, with the end-of-turn time.

    Driven at the tap site named in §5 (``Session._on_decision``) because the rig's STT fake
    replays events rather than audio; the listening path itself is T-0032's.
    """
    log = make_log(tmp_path / "logs")
    async with running(
        tmp_path / "data", script=[answer("Yes. ")], session_log=log, publish=Recorder()
    ) as rig:
        rig.session._on_decision(("onset", 400), 500)
        rig.session._on_decision(("end_of_turn", 900, 0.87), 1000)
        await rig.pump()

    records = read_records(log.path)
    decisions = of_kind(records, "stt.decision")
    assert len(decisions) == 2, (
        f"criterion 12: one record per decision the endpointer made; the log holds "
        f"{len(decisions)} (kinds={sorted(set(kinds(records)))})"
    )
    text = json.dumps(decisions[1]["data"])
    assert "900" in text, (
        f"…carrying the end-of-turn time the decision named; the record says {text}"
    )
    assert "end_of_turn" in text, f"…and which decision it was; the record says {text}"


async def test_logging_changes_nothing_a_user_hears_or_sees(tmp_path):
    """Criterion 24: byte-identical ``astra.*`` traffic with logging on and off.

    The same scripted turn twice, compared message by message on the wire (the production codec's
    bytes, not a summary). The ``turns.jsonl`` lines are compared too: identical except the new
    ``session_log`` field — and the session id, which is a fresh uuid per session.

    T-0180 B-R4 (2026-10-28): ``astra.thought``'s ``attempt_id`` is the engine id, which starts with the session's
    random id. Each session's own id (read from its turn line) is replaced by ``<session>`` in its wire bytes before
    the comparison; no message is dropped, so the thoughts are compared too.
    """
    script = [thought("thinking. "), answer("One half. "), ui()]
    logged = await logged_turn(tmp_path / "on", list(script))

    off = Recorder()
    async with running(
        tmp_path / "off" / "data",
        script=list(script),
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=off,
    ) as rig:
        await rig.ask(QUESTION)
        await rig.speak_out()
        off_lines = rig.lines()
        off_session = rig.session.session_id

    assert logged.published is not None
    assert len(logged.lines) == len(off_lines) == 1
    on_session = json.loads(logged.lines[0])["session_id"]
    assert json.loads(off_lines[0])["session_id"] == off_session, (
        "premise: the turn line names its session's id"
    )
    assert on_session and off_session and on_session != off_session, (
        f"premise: two sessions, two random ids: {on_session!r}, {off_session!r}"
    )
    # T-0180 B-R4: each session's own id becomes <session> in its own bytes; nothing else is rewritten or dropped
    on_wire = [m.replace(on_session, "<session>") for m in logged.published.wire()]
    off_wire = [m.replace(off_session, "<session>") for m in off.wire()]
    assert on_wire == off_wire, (
        "criterion 24: logging is observation only — the page sees the same bytes in the same "
        f"order. First difference: "
        f"{next((f'{a!r:.200} != {b!r:.200}' for a, b in zip(on_wire, off_wire, strict=False) if a != b), 'lengths differ')}; "
        f"{len(on_wire)} messages with logging on, {len(off_wire)} with it off"
    )

    assert len(logged.lines) == len(off_lines) == 1
    on_line = json.loads(logged.lines[0])
    off_line = json.loads(off_lines[0])
    path = on_line.pop("session_log", "<missing>")
    off_line.pop("session_log", None)
    for line in (on_line, off_line):
        line.pop("session_id")
    assert on_line == off_line, (
        "criterion 24: the turn line is identical except for the new field; differences: "
        f"{ {k: (on_line.get(k), off_line.get(k)) for k in set(on_line) | set(off_line) if on_line.get(k) != off_line.get(k)} }"
    )
    assert logged.path is not None and path == str(logged.path), (
        f"criterion 14: the line names the absolute path of the log this turn was recorded in; "
        f"it says {path!r}, the log is at {logged.path}"
    )


# --- T-0092c criterion 7: the ``tool.call`` kind -------------------------------------------------------------


async def test_t0092c_every_published_tool_call_is_one_tool_call_record_with_its_keys(tmp_path):
    """T-0092c criterion 7: ``session_log.KINDS`` gains ``tool.call``; one record per published ``astra.tool.call``,
    the same status in the same order, carrying ``{call_id, source, name, arguments, parse, schema_valid,
    query_problem, repair_ms, action, reason, resolved, spoken_prefix}``; no complaining ``note`` about the kind."""
    from tests.unit import _tool_turn_support as TT

    async with TT.tool_rig(
        tmp_path,
        script=[TT.answer(TT.SPOKEN), TT.tool(TT.REPAIRABLE_CLICK)],
        continuation=[TT.tool(TT.REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await rig.ask("click the sign in button")
        await rig.turn_done()
    published = [m.status for m in rig.tool_calls()]
    records = rig.tool_records()
    assert published, "T-0092c criterion 1: the turn published astra.tool.call"
    assert [r.get("status") for r in records] == published, (
        f"T-0092c criterion 7: one tool.call record per publish; records {[r.get('status') for r in records]}, "
        f"published {published}"
    )
    for r in records:
        missing = sorted(TT.TOOL_CALL_KEYS - set(r))
        assert not missing, f"T-0092c criterion 7: the tool.call record lacks {missing}"
        extra = sorted(set(r) - TT.TOOL_CALL_KEYS)
        assert not extra, (
            f"T-0092c criterion 7 / I13 (reviewer 2): the keys are exact; extra {extra}"
        )
    complaints = [n for n in rig.kind("note") if "tool.call" in json.dumps(n)]
    assert not complaints, (
        f"KINDS must list tool.call, or the record becomes a complaining note: {complaints}"
    )
