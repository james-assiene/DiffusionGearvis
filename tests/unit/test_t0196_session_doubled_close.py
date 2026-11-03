"""T-0196 criteria 4-8, session side: a live tool call committed with a doubled ``<tool_call|>`` close runs exactly as
the same call with one close, at step 1, at step >= 2 and recovered from an open thought; the decline and ask-first
refusals still win; nothing doubled reaches the model or the user (docs/tasks/T-0196.md).

Every row runs T-0111g's live session over the fakes through ``test_t0135_live_recovery.recovery_rig`` (one
``FakeLiveBrowser`` as executor and live browser, loop on, ``recover_thought_calls=True``: the live default since
T-0135b). **The oracle is the paired run**: the same turn whose call carries one close. The doubled run must equal it
in every section listed by ``capture`` (executions, the next step's request body byte for byte, ``tool.call`` records,
HUD tool cards, history, voiced text, ``astra.spoken`` messages, turn-line ``tool``/``extras``/``outcome``), and must
show no repair (no ``ToolRepairStart``, no repair continuation, no ``tokenize``, no ``tool_repair`` request).

Readings (reported to the orchestrator):

- Criterion 6 says "only ``tool.call.draft`` keeps the raw text", but a ``tool.call`` record of a valid call has no
  ``draft`` field at BASE (fields: status, call_id, source, name, arguments, parse, schema_valid, ...). The rows read
  it as: in the session log the doubled text appears only in ``llm.commit`` records (and in a ``tool.call`` record
  only under a ``draft`` key, should one exist); it **is** in the ``llm.commit`` (anti-vacuity).
- Turn lines are compared on ``tool``, ``extras`` and ``outcome`` (``session_id``, the log path and the virtual-clock
  spans are per run).
- ``test_recovered_repairable_still_refused`` and ``test_step2_close_then_prose_still_refused`` are negatives, green
  at BASE: guards, not pending rows.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, loop on (8 steps), live, 1280x800 frames,
repair on (step 1 only), ReGround off, thought-call recovery on, n = 1, deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.toolcall import live, loop
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0135_live_recovery as rec
from tests.unit._tool_turn_support import VALID_CLICK, answer, tool
from tests.unit.test_session_live import ASK, NO, step_body

CLOSE = "<tool_call|>"
DOUBLE = CLOSE + CLOSE
Q = '<|"|>'
#: still ``repairable`` after T-0196 (``schema invalid``: ``sideways`` is not in the enum), returned whole by
#: ``thought_call.recover`` (checked by ``test_t0135_live_recovery``'s premise)
SIDEWAYS = f"<|tool_call>call:scroll{{direction:{Q}sideways{Q}}}<tool_call|>"
DECLINED = rec.DECLINED
ASKED = rec.ASKED
H101C = "model text after call"


async def capture(
    tmp: Path,
    script: list[Any],
    steps: tuple[Any, ...],
    *,
    question: str = tsl.QUESTION,
    body_k: int | None,
) -> dict[str, Any]:
    """One live turn; every section the paired comparison reads."""
    tmp.mkdir(parents=True, exist_ok=True)
    browser = rec.browser3()
    async with rec.recovery_rig(tmp, browser, script=script, recover=True) as rig:
        if steps:
            rig.llm.script_steps(*steps)
        await rig.ask(question)
        await rig.turn_done("T-0196: the turn never ended", limit_ms=40_000)
        await rig.advance(300)
        # the step-k request, if it was sent (absent: the call did not run; the row's own assertions say so)
        sent_k = body_k is not None and len(tsl.steps_of(rig)) >= body_k - 1
        body = json.dumps(step_body(rig, body_k)) if sent_k else None
        tail = tsl.tail(tsl.step_request(rig, body_k)) if sent_k else None
        cards = [
            m.model_dump(mode="json") for m in rig.published.messages if m.type == "astra.tool.call"
        ]
        spoken_msgs = [
            m.model_dump(mode="json") for m in rig.published.messages if m.type == "astra.spoken"
        ]
        repair_starts = rig.commands(C.ToolRepairStart)
        repairs = rec.repair_requests(rig)
        tokenized = list(rig.llm.tokenized)
        history = tsl.history_text(rig)
        spoken = rig.spoken_sources()
    lines = [json.loads(x) for x in rig.lines()]
    return {
        "executed": rec.invocations(browser),
        "body": body,
        "tail": tail,
        "records": rig.tool_records(),
        "cards": cards,
        "history": history,
        "spoken": spoken,
        "astra.spoken": spoken_msgs,
        "lines": [{k: ln.get(k) for k in ("tool", "extras", "outcome")} for ln in lines],
        "repair": (repair_starts, len(repairs), tokenized, rec.tool_repair_logged(rig)),
        "log": rig.log_records,
        "raw_lines": "\n".join(rig.lines()),
    }


SECTIONS = ("executed", "body", "records", "cards", "history", "spoken", "astra.spoken", "lines")


def assert_same_as_single(doubled: dict[str, Any], single: dict[str, Any], what: str) -> None:
    for section in SECTIONS:
        assert doubled[section] == single[section], (
            f"T-0196 {what}: the doubled-close run differs from the single-close run in {section!r}\n"
            f"  single  {json.dumps(single[section], default=str)[:1200]}\n"
            f"  doubled {json.dumps(doubled[section], default=str)[:1200]}"
        )


def assert_no_repair(run: dict[str, Any], what: str) -> None:
    starts, n_repairs, tokenized, logged = run["repair"]
    assert (starts, n_repairs, tokenized, logged) == ([], 0, [], []), (
        f"T-0196 {what}: never repaired; ToolRepairStart {starts}, repair continuations {n_repairs}, "
        f"tokenized {tokenized}, tool_repair requests {logged}"
    )


def assert_nothing_doubled_reaches_model_or_user(run: dict[str, Any], what: str) -> None:
    """Criterion 6: no doubled close in history, voiced text, ``astra.spoken``, the continuation or the turn lines;
    in the continuation every ``<tool_call|>`` belongs to a rendered call; in the session log the raw text stays only
    in ``llm.commit`` (or a ``tool.call`` ``draft`` key)."""
    places = {
        "history": run["history"],
        "voiced": run["spoken"],
        "astra.spoken": json.dumps(run["astra.spoken"]),
        "turns.jsonl": run["raw_lines"],
        "cards": json.dumps(run["cards"]),
    }
    if run["tail"] is not None:
        places["continuation"] = run["tail"]
        places["request body"] = run["body"]
    bad = [
        where
        for where, text in places.items()
        if DOUBLE in text or DOUBLE in text.replace("\\u003c", "<")
    ]
    assert not bad, f"T-0196 criterion 6 ({what}): a doubled close reached {bad}"
    for where in ("history", "voiced", "astra.spoken"):
        assert CLOSE not in places[where], (
            f"T-0196 criterion 6 ({what}): a close tag reached {where}"
        )
    holders = []
    for r in run["log"]:
        data = r.get("data") or {}
        if r["kind"] == "llm.commit":
            continue
        if r["kind"] == "tool.call":
            data = {k: v for k, v in data.items() if k != "draft"}
        if DOUBLE in json.dumps(data):
            holders.append(r["kind"])
    assert holders == [], (
        f"T-0196 criterion 6 ({what}): the raw doubled text is logged under {holders}"
    )
    assert any(
        r["kind"] == "llm.commit" and DOUBLE in str((r.get("data") or {}).get("text", ""))
        for r in run["log"]
    ), f"premise ({what}): the doubled commit is in the session log verbatim"


def strip_rendered(tail: str, calls: list[tuple[str, dict[str, Any]]]) -> str:
    for name, args in calls:
        rendered = loop.render_call(name, args)
        assert rendered in tail, (
            f"T-0196 criterion 6: the continuation holds render_call({name}, {args})"
        )
        tail = tail.replace(rendered, "")
    return tail


# --------------------------------------------------------------------------------------------- criterion 4


async def test_step2_doubled_close_executes(tmp_path: Path) -> None:
    """Criterion 4: a step-2 ``CLICK2 + "<tool_call|>"`` executes with CLICK2's arguments, its ``tool.call`` records
    say ``proposed`` then ``executed`` with ``parse: "ok"`` and ``schema_valid``, no ``"repair: step 1 only"``, no
    repair; the loop continues and the step-3 request body equals the single-close run's byte for byte. At BASE it was
    refused "repair: step 1 only" and the turn ended."""
    runs = {}
    for name, call in (("single", tsl.CLICK2), ("doubled", tsl.CLICK2 + CLOSE)):
        runs[name] = await capture(
            tmp_path / name,
            [answer(tsl.S1), tool(VALID_CLICK)],
            ([answer(tsl.S2), tool(call)], [answer(tsl.S3)]),
            body_k=3,
        )
    d = runs["doubled"]
    assert d["executed"] == [("t1-call", tsl.CLICK_ARGS), ("t1-call-s2", tsl.CLICK2_ARGS)], d[
        "executed"
    ]
    recs = [r for r in d["records"] if r.get("call_id") == "t1-call-s2"]
    assert [(r["status"], r.get("parse"), r.get("schema_valid"), r["arguments"]) for r in recs] == [
        ("proposed", "ok", True, tsl.CLICK2_ARGS),
        ("executed", "ok", True, tsl.CLICK2_ARGS),
    ], recs
    assert all(r.get("reason") != "repair: step 1 only" for r in d["records"]), d["records"]
    assert_no_repair(d, "step 2")
    assert d["body"] == runs["single"]["body"], "T-0196 criterion 4: the step-3 body, byte for byte"
    assert_same_as_single(d, runs["single"], "step 2")


async def test_step1_doubled_close_no_repair(tmp_path: Path) -> None:
    """Criterion 5: the same text at step 1 is not repaired (no ``ToolRepairStart``, repair continuation or
    ``tokenize``), executes, records ``valid`` (``parse: "ok"``), and the step-2 request body and every other section
    equal the single-close run's."""
    runs = {}
    for name, call in (("single", VALID_CLICK), ("doubled", VALID_CLICK + CLOSE)):
        runs[name] = await capture(
            tmp_path / name, [answer(tsl.S1), tool(call)], ([answer(tsl.S3)],), body_k=2
        )
    d = runs["doubled"]
    assert d["executed"] == [("t1-call", tsl.CLICK_ARGS)], d["executed"]
    assert_no_repair(d, "step 1")
    line_tool = d["lines"][0]["tool"] or {}
    assert (line_tool.get("status"), line_tool.get("parse"), line_tool.get("repair_ms")) == (
        "executed",
        "ok",
        None,
    ), line_tool
    assert_same_as_single(d, runs["single"], "step 1")


# --------------------------------------------------------------------------------------------- criterion 6


@pytest.mark.parametrize("step", [1, 2])
async def test_no_duplicate_close_in_history_or_spoken(tmp_path: Path, step: int) -> None:
    """Criterion 6 (negative): in the runs of criteria 4 and 5, no doubled close and no ``<tool_call|>`` outside a
    rendered call reaches history, voiced text, ``astra.spoken``, the step continuation, the request body or the turn
    lines; the continuation's calls are ``loop.render_call(name, arguments)``; the raw text stays in ``llm.commit``.
    Red at BASE through the executions (the doubled call never ran)."""
    if step == 1:
        run = await capture(
            tmp_path, [answer(tsl.S1), tool(VALID_CLICK + CLOSE)], ([answer(tsl.S3)],), body_k=2
        )
        calls = [("click_at", tsl.CLICK_ARGS)]
    else:
        run = await capture(
            tmp_path,
            [answer(tsl.S1), tool(VALID_CLICK)],
            ([answer(tsl.S2), tool(tsl.CLICK2 + CLOSE)], [answer(tsl.S3)]),
            body_k=3,
        )
        calls = [("click_at", tsl.CLICK_ARGS), ("click_at", tsl.CLICK2_ARGS)]
    assert [a for _, a in run["executed"]] == [a for _, a in calls], (
        f"T-0196 criteria 4/5: the doubled-close call executes; {run['executed']}"
    )
    assert_nothing_doubled_reaches_model_or_user(run, f"step {step}")
    rest = strip_rendered(run["tail"], calls)
    assert CLOSE not in rest, (
        f"T-0196 criterion 6: a close outside a rendered call in the continuation: {rest!r}"
    )


# --------------------------------------------------------------------------------------------- criterion 7

SAFETY = {
    "decline": (NO, [answer(tsl.S1)], DECLINED, "user_declined"),
    "ask_first": (tsl.QUESTION, [answer(ASK)], ASKED, "asked_first"),
}


@pytest.mark.parametrize("case", sorted(SAFETY))
async def test_safety_refusals_still_win(tmp_path: Path, case: str) -> None:
    """Criterion 7 (negative; adversarial, safety must not be bypassed by the newly valid shape): with a declining
    transcript, or after a spoken prefix ending "?", a doubled-close call is refused with the decline / ask-first
    reason, never executed, never repaired -- exactly as the single-close call (records, cards, turn lines equal:
    at BASE the doubled draft was refused or repaired as an unparsed draft instead)."""
    question, prefix, reason, flag = SAFETY[case]
    runs = {}
    for name, call in (("single", VALID_CLICK), ("doubled", VALID_CLICK + CLOSE)):
        runs[name] = await capture(
            tmp_path / name, [*prefix, tool(call)], (), question=question, body_k=None
        )
    d = runs["doubled"]
    assert d["executed"] == [], f"T-0196 criterion 7: never executed; {d['executed']}"
    last = [r for r in d["records"] if r.get("call_id") == "t1-call"][-1]
    assert (last["status"], last["reason"], last.get(flag)) == ("refused", reason, True), last
    assert d["cards"] and (d["cards"][-1]["status"], d["cards"][-1]["reason"]) == (
        "refused",
        reason,
    ), d["cards"]
    assert_no_repair(d, case)
    assert_same_as_single(d, runs["single"], case)


# --------------------------------------------------------------------------------------------- criterion 8


@pytest.mark.parametrize("step", [1, 2])
async def test_recovered_doubled_close_executes(tmp_path: Path, step: int) -> None:
    """Criterion 8 (T-0135 A5/A6 amended; H-T196-1): an open thought ending in ``CALL + "<tool_call|>"``, recovery
    on, is ``valid``, executed, its records carry ``from_thought: True``, and the run equals the recovered
    single-close run (no ``THOUGHT_CALL_NO_REPAIR``)."""
    runs = {}
    for name, extra in (("single", ""), ("doubled", CLOSE)):
        if step == 1:
            script, steps, body_k = [rec.d5(VALID_CLICK + extra)], ([answer(tsl.S3)],), 2
        else:
            script = [answer(tsl.S1), tool(VALID_CLICK)]
            steps, body_k = ([rec.d5(tsl.CLICK2 + extra)], [answer(tsl.S3)]), 3
        runs[name] = await capture(tmp_path / name, script, steps, body_k=body_k)
    d = runs["doubled"]
    cid, args = ("t1-call", tsl.CLICK_ARGS) if step == 1 else ("t1-call-s2", tsl.CLICK2_ARGS)
    assert (cid, args) in d["executed"], (
        f"T-0196 criterion 8: the recovered call executes; {d['executed']}"
    )
    recs = [r for r in d["records"] if r.get("call_id") == cid]
    assert [r["status"] for r in recs] == ["proposed", "executed"], recs
    assert all(r.get(rec.FROM) is True for r in recs), recs
    assert_no_repair(d, f"recovered step {step}")
    assert_same_as_single(d, runs["single"], f"recovered step {step}")


@pytest.mark.parametrize("step", [1, 2])
async def test_recovered_repairable_still_refused(tmp_path: Path, step: int) -> None:
    """Criterion 8 (negative): a recovered draft that is still ``repairable`` (the ``sideways`` scroll, schema
    invalid) is refused ``THOUGHT_CALL_NO_REPAIR``, never repaired, never executed. Guard: green at BASE."""
    v = live.classify(SIDEWAYS)
    assert (v.status, v.reason) == ("repairable", "schema invalid"), ("premise", v)
    reason = rec.no_repair_reason()
    if step == 1:
        script, steps, cid = [rec.d5(SIDEWAYS)], ([answer(tsl.S3)],), "t1-call"
    else:
        script, steps = [answer(tsl.S1), tool(VALID_CLICK)], ([rec.d5(SIDEWAYS)], [answer(tsl.S3)])
        cid = "t1-call-s2"
    run = await capture(tmp_path, script, steps, body_k=None)
    assert cid not in [c for c, _ in run["executed"]], run["executed"]
    recs = [r for r in run["records"] if r.get("call_id") == cid]
    assert recs and recs[-1]["status"] == "refused" and recs[-1]["reason"] == reason, recs
    assert all(r.get(rec.FROM) is True for r in recs), recs
    assert_no_repair(run, f"recovered sideways step {step}")


# --------------------------------------------------------------------------------------------- adversarial


@pytest.mark.parametrize(
    "tail", [" ok", CLOSE + " ok", CLOSE + "."], ids=["word", "close_word", "close_dot"]
)
async def test_step2_close_then_prose_still_refused(tmp_path: Path, tail: str) -> None:
    """Adversarial (the rule too wide): a step-2 call followed by a close and then prose or punctuation is not the
    shape -- ``" ok"`` is refused "model text after call" (T-0102), ``"."`` stays ``repairable`` and is refused
    "repair: step 1 only" (T-0114); nothing executes for it and nothing is repaired. Guard: green at BASE."""
    text = tsl.CLICK2 + tail
    want = H101C if tail.endswith("ok") else "repair: step 1 only"  # by hand: T-0102, T-0114
    run = await capture(
        tmp_path,
        [answer(tsl.S1), tool(VALID_CLICK)],
        ([answer(tsl.S2), tool(text)], [answer(tsl.S3)]),
        body_k=None,
    )
    assert run["executed"] == [("t1-call", tsl.CLICK_ARGS)], run["executed"]
    recs = [r for r in run["records"] if r.get("call_id") == "t1-call-s2"]
    assert recs and recs[-1]["status"] == "refused" and recs[-1]["reason"] == want, recs
    assert_no_repair(run, "step-2 prose")
