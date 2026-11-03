"""T-0213 group A2-apply, part 2: what an applied (or refused) settings change leaves in the session log, the turn
line and the pages (docs/tasks/T-0213.md A2a-2, A2a-5, rulings R-8, R-14, R-15; "Tests to write -> Unit (A2-apply)
-> test_t0213_settings_log.py").

**The surface these rows assume** (the task's names; everything else is the implementer's):

- A2a-5: ``session_log.KINDS`` gains ``settings.change``; one record per **applied** set, whose data is exactly
  ``{rev, changes: {key: {old, new}}, source: "ui", turn_seq, applied_at, system_sha12_old, system_sha12_new}``, with
  ``applied_at`` one of ``"immediate"`` (applied when it arrived, at IDLE), ``"after_turn"`` (it waited for the turn
  in flight to end) or ``"after_stop"`` (``stop_turn=true`` stopped the turn in flight first).
- R-15: ``turn_seq`` is the machine's turn count when the set is applied (``MachineState.turn_seq``: 0 before any
  turn; 1 once t1 opened, so 1 for a set applied after t1 ends and 1 for a set held during t1).
- ``system_sha12_old``/``system_sha12_new``: sha256 of the rendered system message text, first 12 hex (A2r-4's
  definition), before and after the set. The oracles here are independent of the apply path: the hello state's
  ``system_sha12`` (A2-read, merged) and the system text of the requests the fake LLM actually received
  (``BudgetLlm``, which renders each body with the real ``VllmClient.request_body``).
- R-14 (and A2a-5, Q5): the user's instructions text is never written verbatim to ``settings.change``, to a note, to
  the ``wire.in`` record of the set or to the ``wire.out`` record of a settings state; those carry its sha12 and its
  length instead. **Reading LG-1:** the shape of that pair is not specified (``{"sha12": ..., "chars": ...}``, or
  two keys of the record), so the rows assert only that the sha12 of the text (sha256 of its UTF-8, 12 hex) and its
  length in code points occur in the record, and that no fragment of the text does. ``llm.request`` records are
  exempt (they log the prompt, as they log the user's own words); one row checks the text **is** in them, so the
  scan of the rest of the file is not vacuous.
- A2a-5: a refused set writes exactly one ``note`` whose data is ``{what: "settings refused", refusals}``;
  ``refusals`` is the refused list (``[{key: "*", reason: "stale"}]`` for a stale rev; ``validate``'s ``(key,
  reason)`` pairs for an invalid set). **Reading LG-2:** each refusal may be written as ``{key, reason}`` or as a
  ``[key, reason]`` pair; both are read.
- R-8: the next turn line (``turns.jsonl``) carries ``regime.settings_rev > 0``, ``regime.system_sha12`` (equal to
  sha256[:12] of that turn's request system text) and ``regime.instructions_edited``; a turn line written before any
  change has none of the three keys (R-8, merged; a premise here).
- A2a-2: every connected page receives the new state (``source: "ui"`` on the changed item); a page that connects
  later receives it in its hello.

The rig is part 1's (``tests/unit/test_t0213_apply_boundary.py``: a live session over ``FakeLiveBrowser``,
``Session(..., visual_tokens=1120)``, the ``BudgetLlm``, a socketless ``HudServer`` with a greeted page, the set sent
as raw JSON so that before A2a-1 the codec answers ``Unknown``). The session log is read after ``aclose``
(``rig.log_records``), because its writer is asynchronous.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, tools ``live`` (loop on, 8 steps, ReGround
off, 1280x800 JPEG frames), n = 1, deterministic per row. No measurement.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.telemetry import session_log
from tests.unit import test_session_tool_loop as tsl
from tests.unit._tool_turn_support import answer
from tests.unit.test_t0213_apply_boundary import (
    _TURN_OF,
    ANSWER_1,
    ANSWER_2,
    CANARY,
    CHECK_SAYS,
    NEW_BUDGET,
    OLD_BUDGET,
    QUESTION_1,
    QUESTION_2,
    SET,
    STATE,
    WINDOWS,
    _close_window,
    _open_window,
    _validate_oracle,
    _window_rig,
    item,
    last_state,
    refused_pairs,
    set_and_pump,
    settings_rig,
    sha12,
    states,
    typed_turn,
)

CHANGE = "settings.change"
REFUSED_NOTE = "settings refused"
#: fragments of the canary: a record that quotes a preview or a prefix of the text is caught too
CANARY_FRAGMENTS = ("ZQX-T0213-CANARY", "British English", "keep it short")
APPLIED_AT = ("immediate", "after_turn", "after_stop")
#: A2a-5's field list, typed from the criterion
CHANGE_FIELDS = {
    "rev",
    "changes",
    "source",
    "turn_seq",
    "applied_at",
    "system_sha12_old",
    "system_sha12_new",
}


# =====================================================================================================
# readers
# =====================================================================================================


def records(rig: Any, kind: str) -> list[dict[str, Any]]:
    """Whole records (``seq``, ``kind``, ``data``) of one kind, after ``aclose``."""
    return [r for r in rig.log_records if r.get("kind") == kind]


def changes_of(rig: Any) -> list[dict[str, Any]]:
    return [r["data"] for r in records(rig, CHANGE)]


def refused_notes(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if d.get("what") == REFUSED_NOTE]


def pairs_of(refusals: Any) -> list[tuple[str, str]]:
    """Reading LG-2: ``{key, reason}`` objects or ``[key, reason]`` pairs."""
    out: list[tuple[str, str]] = []
    for r in refusals:
        if isinstance(r, dict):
            out.append((r["key"], r["reason"]))
        else:
            key, reason = r
            out.append((key, reason))
    return out


def values(tree: Any) -> Iterator[Any]:
    """Every leaf and every key of a JSON tree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield k
            yield from values(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from values(v)
    else:
        yield tree


def holds_sha(tree: Any, digest: str) -> bool:
    return any(isinstance(v, str) and digest in v for v in values(tree))


def holds_int(tree: Any, n: int) -> bool:
    return any(isinstance(v, int) and not isinstance(v, bool) and v == n for v in values(tree))


def leaks(text: str) -> list[str]:
    return [f for f in CANARY_FRAGMENTS if f in text]


def assert_one_change(rig: Any) -> dict[str, Any]:
    got = changes_of(rig)
    assert got, (
        f"T-0213 A2a-5: an applied set writes one {CHANGE!r} record; none was written (the kind is not in "
        f"session_log.KINDS, or the set was not applied); kinds seen "
        f"{sorted({r.get('kind') for r in rig.log_records})}"
    )
    assert len(got) == 1, f"T-0213 A2a-5: one record per applied set; got {len(got)}: {got}"
    data = got[0]
    assert set(data) == CHANGE_FIELDS, (
        f"T-0213 A2a-5: the record's fields are {sorted(CHANGE_FIELDS)}; got {sorted(data)}"
    )
    assert data["source"] == "ui", data
    assert data["applied_at"] in APPLIED_AT, data
    return data


def system_text(rig_llm: Any, turn: str) -> str:
    sent = rig_llm.of_turn(turn)
    assert sent, f"premise: {turn} sent requests"
    return sent[0].system_text


def hello_state(page: Any) -> Any:
    found = states(page.sent())
    assert found, "premise: the greeted page's hello reply carries astra.settings.state (A2r-3)"
    return found[-1]


def t1_last_request_seq(rig: Any, turn: str = "t1") -> int:
    seqs = [
        r["seq"]
        for r in records(rig, "llm.request")
        if (m := _TURN_OF.search(str(r["data"].get("attempt_id", "")))) and m.group(1) == turn
    ]
    assert seqs, f"premise: {turn}'s llm.request records are in the log"
    return max(seqs)


def regime_of(rig: Any, turn: str) -> dict[str, Any]:
    lines = [x for x in rig.raw_lines() if x["turn_id"] == turn]
    assert len(lines) == 1, f"premise: one turn line for {turn}; got {len(lines)}"
    return dict(lines[0]["regime"])


# =====================================================================================================
# A2a-5: the kind
# =====================================================================================================


def test_settings_change_is_a_declared_kind():
    """A2a-5: ``session_log.KINDS`` gains ``settings.change`` (the log turns an undeclared kind into a complaining
    note, so a record written without it is lost)."""
    assert CHANGE in session_log.KINDS, f"T-0213 A2a-5: {CHANGE!r} is not in session_log.KINDS yet"


# =====================================================================================================
# A2a-5: one record per applied set, each applied_at
# =====================================================================================================


async def test_immediate_at_start_writes_one_record_with_turn_seq_zero(tmp_path):
    """A2a-5 / R-15: a set applied at IDLE before any turn writes one record: rev 1, ``changes`` exactly the changed
    keys with their old (the hello's values) and new values, ``source: "ui"``, ``turn_seq`` 0, ``applied_at:
    "immediate"``, ``system_sha12_old`` the hello's ``system_sha12`` and ``system_sha12_new`` the sha12 of t1's system
    text."""
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        hello = hello_state(rig.page)
        guard_old = item(hello, "repeat_guard").value
        assert isinstance(guard_old, bool), (
            f"premise: repeat_guard is a bool in live; {guard_old!r}"
        )
        assert item(hello, "visual_tokens").value == OLD_BUDGET, "premise: the rig's budget"
        mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET, "repeat_guard": not guard_old})
        assert last_state(rig, mark).rev == 1, "premise: the set was applied"
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        t1_text = system_text(llm, "t1")
    data = assert_one_change(rig)
    assert data["rev"] == 1, data
    assert data["changes"] == {
        "visual_tokens": {"old": OLD_BUDGET, "new": NEW_BUDGET},
        "repeat_guard": {"old": guard_old, "new": not guard_old},
    }, f"T-0213 A2a-5: changes is {{key: {{old, new}}}} for the changed keys; got {data['changes']}"
    assert data["turn_seq"] == 0, (
        f"T-0213 R-15: no turn has opened yet; turn_seq {data['turn_seq']}"
    )
    assert data["applied_at"] == "immediate", data
    assert data["system_sha12_old"] == hello.system_sha12, (data, hello.system_sha12)
    assert data["system_sha12_new"] == sha12(t1_text), (data["system_sha12_new"], sha12(t1_text))


async def test_immediate_after_a_turn_has_turn_seq_one_and_both_shas(tmp_path):
    """A2a-5 / R-15: after t1, a set of ``user_instructions`` at IDLE writes ``turn_seq`` 1, ``applied_at:
    "immediate"``, ``system_sha12_old`` = sha12(t1's system text) and ``system_sha12_new`` = sha12(t2's), which
    differ (the block was appended)."""
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        mark = await set_and_pump(rig, {"user_instructions": CANARY})
        assert last_state(rig, mark).rev == 1, "premise: the set was applied"
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t1_text, t2_text = system_text(llm, "t1"), system_text(llm, "t2")
    assert sha12(t1_text) != sha12(t2_text), "premise: the block changed the system text"
    data = assert_one_change(rig)
    assert (data["rev"], data["turn_seq"], data["applied_at"]) == (1, 1, "immediate"), data
    assert (data["system_sha12_old"], data["system_sha12_new"]) == (
        sha12(t1_text),
        sha12(t2_text),
    ), data
    assert list(data["changes"]) == ["user_instructions"], data["changes"]


async def test_two_applied_sets_write_two_records_in_order(tmp_path):
    """A2a-5: one record per applied set: two sets write two records, rev 1 then rev 2, the second's ``old`` being
    the first's ``new``; and the second's ``system_sha12_old`` is the first's ``system_sha12_new``."""
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
        mark = await set_and_pump(rig, {"visual_tokens": 280}, rev=1)
        assert last_state(rig, mark).rev == 2, "premise: both sets were applied"
    got = changes_of(rig)
    assert [d["rev"] for d in got] == [1, 2], f"T-0213 A2a-5: one record per applied set; {got}"
    assert got[0]["changes"] == {"visual_tokens": {"old": OLD_BUDGET, "new": NEW_BUDGET}}, got[0]
    assert got[1]["changes"] == {"visual_tokens": {"old": NEW_BUDGET, "new": 280}}, got[1]
    assert got[1]["system_sha12_old"] == got[0]["system_sha12_new"], got


@pytest.mark.parametrize("window", WINDOWS)
async def test_mid_turn_set_is_logged_after_the_turn(tmp_path, window):
    """A2a-5 / A2a-3 / R-15: a set that arrives during t1 (after its prefill started, between its steps, or during
    its completion check) writes exactly one record, ``applied_at: "after_turn"``, ``turn_seq`` 1, and that record
    comes after t1's last ``llm.request`` (it was applied at the boundary, not when it arrived). Its shas are t1's and
    t2's system texts."""
    gate = asyncio.Event()
    async with _window_rig(tmp_path, window, gate) as (rig, llm, browser):
        if window != "prefill":
            llm.script_steps([answer_s3()])
            if window == "completion_check":
                llm.script_checks([answer_check()])
        await _open_window(rig, llm, browser, window, gate)
        mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET, "user_instructions": CANARY})
        assert item(last_state(rig, mark), "visual_tokens").pending == NEW_BUDGET, (
            "premise: the change is pending mid-turn"
        )
        await _close_window(rig, llm, window, gate)
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t1_text, t2_text = system_text(llm, "t1"), system_text(llm, "t2")
    data = assert_one_change(rig)
    assert (data["rev"], data["applied_at"], data["turn_seq"]) == (1, "after_turn", 1), data
    assert data["changes"]["visual_tokens"] == {"old": OLD_BUDGET, "new": NEW_BUDGET}, data
    assert (data["system_sha12_old"], data["system_sha12_new"]) == (
        sha12(t1_text),
        sha12(t2_text),
    ), data
    (record,) = records(rig, CHANGE)
    assert record["seq"] > t1_last_request_seq(rig), (
        f"T-0213 A2a-3/A2a-5: the change is applied (and logged) after t1's last request; record seq "
        f"{record['seq']}, t1's last llm.request seq {t1_last_request_seq(rig)}"
    )


def answer_s3() -> Any:
    return answer(tsl.S3)


def answer_check() -> Any:
    return answer(CHECK_SAYS)


async def test_stop_turn_is_logged_after_stop(tmp_path):
    """A2a-5 / R-15: ``stop_turn=true`` while t1's click executes writes one record, ``applied_at: "after_stop"``,
    ``turn_seq`` 1."""
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer_s3()])
        await _open_window(rig, llm, browser, "between_steps", gate)
        mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET}, stop_turn=True)
        await rig.advance(500, step=40)
        gate.set()
        await rig.advance(1_000, step=40)
        assert last_state(rig, mark).rev == 1, "premise: the change was applied after the stop"
    data = assert_one_change(rig)
    assert (data["rev"], data["applied_at"], data["turn_seq"]) == (1, "after_stop", 1), data
    assert data["changes"] == {"visual_tokens": {"old": OLD_BUDGET, "new": NEW_BUDGET}}, data


async def test_stop_turn_at_idle_is_logged_immediate(tmp_path):
    """A2a-5 (the other side of ``after_stop``): ``stop_turn=true`` with no turn in flight stops nothing, so the
    record says ``applied_at: "immediate"``, not ``"after_stop"``."""
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET}, stop_turn=True)
        assert last_state(rig, mark).rev == 1, "premise: applied"
    data = assert_one_change(rig)
    assert (data["applied_at"], data["turn_seq"]) == ("immediate", 1), data


# =====================================================================================================
# R-14: the instructions text is never logged verbatim (except in llm.request)
# =====================================================================================================


async def test_instructions_text_is_logged_only_as_sha_and_length(tmp_path):
    """R-14 / A2a-5 (adversarial): ``user_instructions`` set to a canary at IDLE, then a turn under it. No fragment of
    the canary occurs anywhere in the session log file outside ``llm.request`` records (so not in
    ``settings.change``, a note, the set's ``wire.in`` or a state's ``wire.out``), nor in ``turns.jsonl``; the
    ``settings.change`` record, the set's ``wire.in`` and the rev-1 state's ``wire.out`` carry the canary's sha12
    and length instead. The canary **is** in t1's ``llm.request`` records (the scan is not vacuous)."""
    digest, length = sha12(CANARY), len(CANARY)
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        mark = await set_and_pump(rig, {"user_instructions": CANARY})
        assert last_state(rig, mark).rev == 1, "premise: applied"
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        log_path = Path(rig.log.path)
        turns_text = rig.turns_path.read_text(encoding="utf-8")
    exempt, scanned = [], []
    for n, line in enumerate(log_path.read_text(encoding="utf-8").splitlines()):
        (exempt if json.loads(line).get("kind") == "llm.request" else scanned).append((n, line))
    assert any(CANARY in line for _, line in exempt), (
        "premise: the canary reached the log through t1's llm.request (the prompt is logged verbatim)"
    )
    bad = [(n, json.loads(line)["kind"], leaks(line)) for n, line in scanned if leaks(line)]
    assert bad == [], (
        f"T-0213 R-14: the instructions text is never logged outside llm.request; found in (line, kind, "
        f"fragments) {bad}"
    )
    assert leaks(turns_text) == [], (
        "T-0213 A2a-5 (Q5): the turn lines carry the sha, never the text"
    )
    data = assert_one_change(rig)
    assert holds_sha(data["changes"]["user_instructions"]["new"], digest) and holds_int(
        data["changes"]["user_instructions"]["new"], length
    ), (
        f"T-0213 A2a-5: the new instructions are logged as sha12 {digest} and length {length}; {data['changes']}"
    )
    sets = [
        r["data"]
        for r in records(rig, "wire.in")
        if (r["data"].get("msg") or {}).get("type") == SET
    ]
    assert len(sets) == 1, (
        f"T-0213 A2a-1/R-14: one wire.in of the set, decoded as an astra.settings.set (not Unknown); "
        f"{[r['data'] for r in records(rig, 'wire.in')][-3:]}"
    )
    assert sets[0]["outcome"] == "astra", sets[0]
    assert holds_sha(sets[0], digest) and holds_int(sets[0], length), (
        f"T-0213 R-14: the set's wire.in carries the instructions' sha12 {digest} and length {length}; {sets[0]}"
    )
    outs = [
        r["data"]["msg"]
        for r in records(rig, "wire.out")
        if (r["data"].get("msg") or {}).get("type") == STATE and r["data"]["msg"].get("rev") == 1
    ]
    assert outs, "T-0213 A2a-2: the applied state was published (and logged as wire.out)"
    assert all(holds_sha(m, digest) and holds_int(m, length) for m in outs), (
        f"T-0213 R-14: the state's wire.out carries the instructions' sha12 {digest} and length {length}; {outs[0]}"
    )


@pytest.mark.parametrize("case", ["stale", "special-token"])
async def test_refused_instructions_are_not_logged_either(tmp_path, case):
    """R-14 (adversarial, the refusal path): a refused set carrying the canary (a stale rev; or the canary with a
    chat-template special token, refused by ``validate_instructions``) leaves no fragment of it outside
    ``llm.request``, in its ``wire.in``, its note or the state's ``wire.out``; the refusal is still noted."""
    text = CANARY if case == "stale" else CANARY + " <|think|>"
    rev = 5 if case == "stale" else 0
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        mark = await set_and_pump(rig, {"user_instructions": text}, rev=rev)
        assert last_state(rig, mark).refused, "premise: the set was refused"
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        log_text = Path(rig.log.path).read_text(encoding="utf-8")
    assert len(refused_notes(rig)) == 1, "premise: the refusal is noted"
    bad = [
        (json.loads(line)["kind"], leaks(line))
        for line in log_text.splitlines()
        if leaks(line) and json.loads(line).get("kind") != "llm.request"
    ]
    assert bad == [], f"T-0213 R-14: a refused set's instructions are not logged either; {bad}"
    sets = [
        r["data"]
        for r in records(rig, "wire.in")
        if (r["data"].get("msg") or {}).get("type") == SET
    ]
    assert len(sets) == 1 and holds_sha(sets[0], sha12(text)), (
        f"T-0213 R-14: the refused set's wire.in still carries the sha12 of its instructions; {sets}"
    )


# =====================================================================================================
# A2a-5: a refused set writes one note
# =====================================================================================================


REFUSALS = {
    "stale": ({"visual_tokens": NEW_BUDGET}, 3),
    "invalid": ({"visual_tokens": NEW_BUDGET, "repeat_guard": "yes"}, 0),
    "relaunch-key": ({"browser_web": False}, 0),
}


@pytest.mark.parametrize("case", list(REFUSALS))
async def test_a_refused_set_writes_exactly_one_note(tmp_path, case):
    """A2a-5: a refused set writes exactly one ``note {what: "settings refused", refusals}`` (``[("*", "stale")]``
    for a stale rev; ``validate``'s pairs for an invalid set) and no ``settings.change`` record."""
    changes, rev = REFUSALS[case]
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        want = [("*", "stale")] if case == "stale" else _validate_oracle(rig, changes)
        mark = await set_and_pump(rig, changes, rev=rev)
        state = last_state(rig, mark)
        assert sorted(refused_pairs(state)) == sorted(want), "premise: the state says refused"
    notes = refused_notes(rig)
    assert len(notes) == 1, f"T-0213 A2a-5: one note per refused set; got {notes}"
    assert set(notes[0]) == {"what", "refusals"}, (
        f"T-0213 A2a-5: the note is {{what, refusals}}; {notes[0]}"
    )
    assert sorted(pairs_of(notes[0]["refusals"])) == sorted(want), (notes[0], want)
    assert changes_of(rig) == [], (
        f"T-0213 A2a-5: nothing was applied, so no {CHANGE}; {changes_of(rig)}"
    )


async def test_an_applied_set_writes_no_refusal_note(tmp_path):
    """A2a-5 (the guard's other side): an applied set writes its ``settings.change`` and no ``settings refused``
    note; a stale set after it adds exactly one note and no second record."""
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
        await set_and_pump(rig, {"visual_tokens": 280}, rev=0)
    assert [d["rev"] for d in changes_of(rig)] == [1], (
        f"T-0213 A2a-5: the applied set wrote one {CHANGE} (rev 1) and the stale one none; {changes_of(rig)}"
    )
    notes = refused_notes(rig)
    assert len(notes) == 1 and pairs_of(notes[0]["refusals"]) == [("*", "stale")], notes


# =====================================================================================================
# R-8: the next turn line
# =====================================================================================================


async def test_the_next_turn_line_carries_the_settings_after_an_instructions_change(tmp_path):
    """A2a-5 / R-8: t1 (before any change) writes none of the three keys; after ``user_instructions`` is applied,
    t2's line has ``settings_rev`` 1, ``system_sha12`` = sha256[:12] of t2's request system text, and
    ``instructions_edited`` true."""
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        await set_and_pump(rig, {"user_instructions": CANARY})
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t2_text = system_text(llm, "t2")
        r1, r2 = regime_of(rig, "t1"), regime_of(rig, "t2")
    assert not {"settings_rev", "system_sha12", "instructions_edited"} & set(r1), (
        f"premise (R-8): an unchanged turn writes BASE's line; {sorted(r1)}"
    )
    got = (r2.get("settings_rev"), r2.get("system_sha12"), r2.get("instructions_edited"))
    assert got == (1, sha12(t2_text), True), (
        f"T-0213 A2a-5/R-8: t2's line carries (settings_rev, system_sha12, instructions_edited) = "
        f"(1, {sha12(t2_text)}, True); got {got}"
    )


async def test_the_next_turn_line_after_a_budget_change(tmp_path):
    """A2a-5 / R-8: a ``visual_tokens`` change alone moves ``settings_rev`` to 1 (so the keys are written), the
    ``system_sha12`` is t2's (equal to t1's: the budget is not prompt text) and ``instructions_edited`` is false."""
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t1_text, t2_text = system_text(llm, "t1"), system_text(llm, "t2")
        r2 = regime_of(rig, "t2")
    assert t1_text == t2_text, "premise: the budget does not change the system text"
    got = (r2.get("settings_rev"), r2.get("system_sha12"), r2.get("instructions_edited"))
    assert got == (1, sha12(t2_text), False), got


async def test_the_turn_line_after_an_instructions_reset(tmp_path):
    """A2a-5 / R-8: instructions set (rev 1) then reset to ``""`` (rev 2): t3's line has ``settings_rev`` 2 (still
    written, the rev is > 0), ``instructions_edited`` false and ``system_sha12`` equal to t1's system text's (BASE)."""
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        await set_and_pump(rig, {"user_instructions": CANARY})
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        await set_and_pump(rig, {"user_instructions": ""}, rev=1)
        await typed_turn(rig, "and now", "Still the same. ", 3)
        t1_text = system_text(llm, "t1")
        r3 = regime_of(rig, "t3")
    got = (r3.get("settings_rev"), r3.get("system_sha12"), r3.get("instructions_edited"))
    assert got == (2, sha12(t1_text), False), got


async def test_a_mid_turn_change_leaves_the_running_turns_line_on_the_old_settings(tmp_path):
    """A2a-3 / A2a-5 / R-8: a set arriving between t1's steps is applied after t1, so t1's line carries none of the
    three keys (rev 0, unchanged) and t2's carries rev 1 with t2's system sha."""
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer_s3()])
        await _open_window(rig, llm, browser, "between_steps", gate)
        await set_and_pump(rig, {"user_instructions": CANARY})
        await _close_window(rig, llm, "between_steps", gate)
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t2_text = system_text(llm, "t2")
        r1, r2 = regime_of(rig, "t1"), regime_of(rig, "t2")
    assert not {"settings_rev", "system_sha12", "instructions_edited"} & set(r1), (
        f"T-0213 A2a-3: t1 ran under rev 0 (applied after it), so its line is BASE's; {sorted(r1)}"
    )
    got = (r2.get("settings_rev"), r2.get("system_sha12"), r2.get("instructions_edited"))
    assert got == (1, sha12(t2_text), True), got


# =====================================================================================================
# A2a-2: every page, and a page that connects later
# =====================================================================================================


async def test_every_page_receives_the_state_and_a_later_page_gets_it_in_its_hello(tmp_path):
    """A2a-2: with two pages connected, an applied set reaches both as a state with rev 1 and ``source: "ui"`` on the
    changed item; a third page that connects afterwards gets the same in its hello's settings state (with the
    catalog, R-5: the last message of the reply), instructions included."""
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        other = rig.greet()
        marks = {id(p): len(p.sent()) for p in (rig.page, other)}
        await set_and_pump(rig, {"visual_tokens": NEW_BUDGET, "user_instructions": CANARY})
        got = {}
        for name, page in (("first", rig.page), ("second", other)):
            found = states(page.sent()[marks[id(page)] :])
            assert found, f"T-0213 A2a-2: the {name} page received no astra.settings.state"
            got[name] = found[-1]
        later = rig.greet()
        reply = later.sent()
    for name, state in got.items():
        vt = item(state, "visual_tokens")
        assert (state.rev, vt.value, vt.source) == (1, NEW_BUDGET, "ui"), (name, state.rev, vt)
        assert state.instructions == CANARY, name
    assert reply and getattr(reply[-1], "type", None) == STATE, (
        f"premise (R-5): the hello reply ends with the settings state; {[getattr(m, 'type', None) for m in reply]}"
    )
    hello = reply[-1]
    vt = item(hello, "visual_tokens")
    assert (hello.rev, vt.value, vt.source) == (1, NEW_BUDGET, "ui"), (hello.rev, vt)
    assert hello.instructions == CANARY and hello.catalog, (
        "T-0213 A2a-2: a later page sees the change"
    )


async def test_a_change_applied_after_the_turn_reaches_every_page(tmp_path):
    """A2a-2 / A2a-3: a set held during t1 is published to every page twice: at once (rev 0, the change pending) and
    when applied after t1 (rev 1, ``source: "ui"``, nothing pending)."""
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        other = rig.greet()
        llm.script_steps([answer_s3()])
        await _open_window(rig, llm, browser, "between_steps", gate)

        # No index mark: the socketless outbox moves the latest-wins astra.vision.frame to the back of
        # page.sent(), so an index taken before the set can shift. Mark by content instead: keep the states that
        # are not "rev 0 with nothing pending", and check none exists before the set.
        def marked(page: Any) -> list[Any]:
            return [
                s
                for s in states(page.sent())
                if s.rev != 0 or item(s, "visual_tokens").pending is not None
            ]

        before = {"first": marked(rig.page), "second": marked(other)}
        assert before == {"first": [], "second": []}, (
            f"premise: before the set no page has a pending or applied state; {before}"
        )
        await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
        await _close_window(rig, llm, "between_steps", gate)
        seen = {"first": marked(rig.page), "second": marked(other)}
    for name, found in seen.items():
        shape = [
            (s.rev, item(s, "visual_tokens").value, item(s, "visual_tokens").pending) for s in found
        ]
        assert (
            shape and shape[0] == (0, OLD_BUDGET, NEW_BUDGET) and shape[-1] == (1, NEW_BUDGET, None)
        ), f"T-0213 A2a-2/A2a-3: the {name} page sees pending then applied; {shape}"
        assert item(found[-1], "visual_tokens").source == "ui", name


# =====================================================================================================
# R-20 / R-16 / R-19: the corners of a set that must not leak (the reviewers' surviving mutants)
# =====================================================================================================

#: a canary no other text in the session holds; R-16's corners put it where a value, a key or a field name goes
LEAK = "ZQXR16LEAKCANARY"
#: R-19 (a): a refusal reason names the key, the value's type and its length, so it is short whatever the value is.
#: **Reading R20-1:** "bounded" is not given a number; 500 code points is far above any reason R-19 (a) describes
#: and far below the 300 kB value, so an implementation that quotes the value (or a prefix of it past 500) fails.
REASON_BOUND = 500
#: a value of about 300 kB (R-20: "a 300 kB string value")
HUGE = (LEAK + " ") * (300_000 // (len(LEAK) + 1))

#: R-16 / R-20 corners: the set's ``changes`` (each decodes as an ``astra.settings.set`` and is refused)
LEAK_SETS: dict[str, dict[str, Any]] = {
    "instructions-list": {"user_instructions": [LEAK]},
    "instructions-dict": {"user_instructions": {"t": LEAK}},
    "unknown-key": {LEAK: 1},
    "reground-string": {"reground": LEAK},
    "huge-instructions": {"user_instructions": HUGE},
    "huge-bool-value": {"repeat_guard": HUGE},
}


def _non_request_leaks(log_text: str, needle: str) -> list[str]:
    return [
        json.loads(line).get("kind")
        for line in log_text.splitlines()
        if needle in line and json.loads(line).get("kind") != "llm.request"
    ]


async def _leak_run(tmp_path: Path, raw: dict[str, Any]) -> tuple[Any, list[Any], str, str]:
    """Send ``raw`` as the page's frame at IDLE, pump, then one typed turn (so turns.jsonl has a line); return the
    rig, the states published after the frame, the session-log text and the turns.jsonl text (after ``aclose``)."""
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        mark = len(rig.published.messages)
        rig.page.receive(json.dumps(raw))
        await rig.pump()
        await rig.advance(40, step=20)
        answered = states(rig.published.messages[mark:])
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        turns_text = rig.turns_path.read_text(encoding="utf-8")
    log_text = Path(rig.log.path).read_text(encoding="utf-8")
    return rig, answered, log_text, turns_text


@pytest.mark.parametrize("case", list(LEAK_SETS))
async def test_r16_corner_sets_leak_nothing_and_are_answered(tmp_path, case):
    """R-16 / R-19 (a)(b) (adversarial): a set whose instructions are a list or a dict holding the canary, whose key is
    the canary, whose bool value is the canary, or whose value is a ~300 kB string of it is refused and answered with
    one ``astra.settings.state`` carrying a refusal; the canary is absent from every session-log line other than
    ``llm.request`` (``wire.in``, the note, the state's ``wire.out``) and from turns.jsonl; every refusal reason is
    bounded (reading R20-1) and quotes no part of the value or the key."""
    raw = {"type": SET, "rev": 0, "changes": LEAK_SETS[case]}
    rig, answered, log_text, turns_text = await _leak_run(tmp_path, raw)
    assert len(answered) == 1, (
        f"T-0213 A2a-2: the set is answered with one state; got {len(answered)}"
    )
    pairs = refused_pairs(answered[0])
    assert pairs and answered[0].rev == 0, (
        f"T-0213 A2a-2: the set is refused and changes nothing; {[(k[:40], r[:80]) for k, r in pairs]}"
    )
    for key, reason in pairs:
        assert LEAK not in key and LEAK not in reason, (
            f"T-0213 R-19 (a): a refusal never quotes a value or a key; ({key[:60]!r}, {reason[:120]!r})"
        )
        assert len(reason) <= REASON_BOUND, (
            f"T-0213 R-19 (a): the reason is bounded ({REASON_BOUND} code points); got {len(reason)}"
        )
    assert _non_request_leaks(log_text, LEAK) == [], (
        f"T-0213 R-16: the canary is in the session log outside llm.request, in kinds "
        f"{_non_request_leaks(log_text, LEAK)}"
    )
    assert LEAK not in turns_text, "T-0213 R-16: the scan includes turns.jsonl"
    assert len(refused_notes(rig)) == 1, "T-0213 A2a-5: the refusal is noted once"


async def test_r16_top_level_unknown_field_named_by_the_canary_is_not_logged(tmp_path):
    """R-16 / R-19 (c) (adversarial): an ``astra.settings.set`` with an unknown top-level field named by the canary is
    malformed (A2a-1); its ``wire.in`` record (the codec's error) names no pydantic ``loc`` field, so the canary is
    absent from every session-log line outside ``llm.request`` and from turns.jsonl; nothing is applied.

    **Reading R20-3:** R-20 says every set is answered with a state; a malformed frame does not decode to a set, so
    this row does not ask for a state (the transport answers ``astra.error``, T-0008)."""
    raw = {"type": SET, "rev": 0, "changes": {"visual_tokens": NEW_BUDGET}, LEAK: 1}
    rig, answered, log_text, turns_text = await _leak_run(tmp_path, raw)
    malformed = [
        r["data"] for r in records(rig, "wire.in") if r["data"].get("outcome") == "malformed"
    ]
    assert len(malformed) == 1, (
        f"premise (A2a-1): the frame is logged once as malformed; "
        f"{[r['data'].get('outcome') for r in records(rig, 'wire.in')]}"
    )
    assert _non_request_leaks(log_text, LEAK) == [], (
        f"T-0213 R-16 / R-19 (c): the malformed set's field name reached the log in kinds "
        f"{_non_request_leaks(log_text, LEAK)}"
    )
    assert LEAK not in turns_text, "T-0213 R-16: the scan includes turns.jsonl"
    assert all(s.rev == 0 for s in answered) and changes_of(rig) == [], "nothing was applied"


async def test_an_empty_key_is_refused_under_star_and_answered(tmp_path):
    """R-19 (d): ``{"changes": {"": 1}}`` is answered with one state whose refusal key is ``"*"`` (an empty refusal
    key cannot be carried, so it maps to ``"*"``) with a non-empty reason; rev stays 0 and one note is written."""
    rig, answered, _log_text, _turns = await _leak_run(
        tmp_path, {"type": SET, "rev": 0, "changes": {"": 1}}
    )
    assert len(answered) == 1, (
        f"T-0213 R-19 (d): every set is answered with a state; got {len(answered)}"
    )
    pairs = refused_pairs(answered[0])
    assert [k for k, _ in pairs] == ["*"] and all(r.strip() for _, r in pairs), pairs
    assert answered[0].rev == 0 and changes_of(rig) == []
    notes = refused_notes(rig)
    assert len(notes) == 1 and [k for k, _ in pairs_of(notes[0]["refusals"])] == ["*"], notes


# =====================================================================================================
# R-20: a no-op set, a duplicate set
# =====================================================================================================


@pytest.mark.parametrize(
    "changes", [{}, {"visual_tokens": OLD_BUDGET}], ids=["empty", "same-value"]
)
async def test_a_no_op_set_is_echoed_and_records_nothing(tmp_path, changes):
    """A2a-2 / A2a-5: an empty set, or one that names the value in force, is still answered with one state (rev 0,
    nothing refused, nothing pending) and writes no ``settings.change`` and no refusal note."""
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        mark = await set_and_pump(rig, changes)
        got = states(rig.published.messages[mark:])
    assert len(got) == 1, f"T-0213 A2a-2: a no-op set still gets its echo; got {len(got)} states"
    assert got[0].rev == 0 and not got[0].refused, (got[0].rev, got[0].refused)
    assert item(got[0], "visual_tokens").pending is None
    assert changes_of(rig) == [] and refused_notes(rig) == [], (changes_of(rig), refused_notes(rig))


async def test_the_same_set_twice_mid_turn_is_one_change(tmp_path):
    """A2a-3 / A2a-5: the same valid set sent twice while t1's click executes (both naming rev 0, the revision in
    force) leaves one pending change (both echoes show the same pending value, rev 0), and after t1 one
    ``settings.change`` record (rev 1, ``after_turn``); the rev in force is then 1, not 2."""
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer_s3()])
        await _open_window(rig, llm, browser, "between_steps", gate)
        first = last_state(rig, await set_and_pump(rig, {"visual_tokens": NEW_BUDGET}))
        second = last_state(rig, await set_and_pump(rig, {"visual_tokens": NEW_BUDGET}))
        await _close_window(rig, llm, "between_steps", gate)
        rev_after = rig.session.settings.rev
    for echo in (first, second):
        vt = item(echo, "visual_tokens")
        got = (echo.rev, vt.value, vt.pending, bool(echo.refused))
        assert got == (0, OLD_BUDGET, NEW_BUDGET, False), got
    got_changes = changes_of(rig)
    assert [(d["rev"], d["applied_at"]) for d in got_changes] == [(1, "after_turn")], (
        f"T-0213 A2a-5: one record for one change; {got_changes}"
    )
    assert rev_after == 1, rev_after


# =====================================================================================================
# R-20 / R-14: a mid-turn instructions change; the length in code points
# =====================================================================================================

#: five astral characters (U+1D11E, 4 UTF-8 bytes, 2 UTF-16 units each): 34 code points, 49 UTF-8 bytes, 39 UTF-16
ASTRAL = "ZQX-ASTRAL \U0001d11e\U0001d11e\U0001d11e\U0001d11e\U0001d11e answer in verse!!"
ASTRAL_FRAGMENTS = ("ZQX-ASTRAL", "answer in verse")


def test_astral_canary_premise():
    """The hand-counted lengths the next row asserts."""
    got = (len(ASTRAL), len(ASTRAL.encode("utf-8")), len(ASTRAL.encode("utf-16-le")) // 2)
    assert got == (34, 49, 39), got


async def test_mid_turn_instructions_are_redacted_and_counted_in_code_points(tmp_path):
    """R-14 / R-20 (adversarial): ``user_instructions`` set to an astral-character canary while t1's click executes:
    the pending echo's ``wire.out`` (rev 0, the item's pending value), the set's ``wire.in`` and the applied
    ``settings.change`` carry its sha12 and its length in code points (34), never its UTF-8 (49) or UTF-16 (39)
    length, and no fragment of the text occurs outside ``llm.request``; the page itself received the text."""
    gate = asyncio.Event()
    digest = sha12(ASTRAL)
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer_s3()])
        await _open_window(rig, llm, browser, "between_steps", gate)
        echo = last_state(rig, await set_and_pump(rig, {"user_instructions": ASTRAL}))
        assert echo.rev == 0 and item(echo, "user_instructions").pending == ASTRAL, (
            f"premise: the page sees the pending text itself; {item(echo, 'user_instructions')}"
        )
        await _close_window(rig, llm, "between_steps", gate)
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
    log_text = Path(rig.log.path).read_text(encoding="utf-8")
    bad = [k for f in ASTRAL_FRAGMENTS for k in _non_request_leaks(log_text, f)]
    assert bad == [], f"T-0213 R-14: the instructions text reached the log in kinds {bad}"
    pending_outs = [
        r["data"]["msg"]
        for r in records(rig, "wire.out")
        if (r["data"].get("msg") or {}).get("type") == STATE
        and r["data"]["msg"].get("rev") == 0
        and holds_sha(r["data"]["msg"], digest)
    ]
    assert pending_outs, (
        "T-0213 R-14: the pending state's wire.out carries the pending text's sha12"
    )
    sets = [
        r["data"]
        for r in records(rig, "wire.in")
        if (r["data"].get("msg") or {}).get("type") == SET
    ]
    assert len(sets) == 1, sets
    data = assert_one_change(rig)
    for what, tree in (
        ("pending wire.out", pending_outs[0]),
        ("wire.in", sets[0]),
        ("settings.change", data["changes"]["user_instructions"]["new"]),
    ):
        assert holds_sha(tree, digest) and holds_int(tree, 34), (
            f"T-0213 R-14: the {what} carries sha12 {digest} and the length in code points (34)"
        )
        assert not holds_int(tree, 49) and not holds_int(tree, 39), (
            f"T-0213 R-14: the {what}'s length is in code points, not UTF-8 bytes (49) or UTF-16 units (39)"
        )


# =====================================================================================================
# R-20: the turn line's grounding after a ReGround change
# =====================================================================================================

#: the rig's grounding as ``cli``'s ``grounding_of`` writes it for a live session with ReGround off at 1120 (T-0132
#: A4c; with ReGround off the pass-2 budget reads the main one)
GROUNDING_OFF = {
    "visual_tokens": OLD_BUDGET,
    "visual_tokens_from": "default",
    "reground": False,
    "reground_from": "default",
    "reground_visual_tokens": OLD_BUDGET,
    "reground_visual_tokens_from": "default",
}


async def test_turn_line_grounding_follows_a_reground_change(tmp_path):
    """A2a-4 / A2a-5 (T-0132 A4c): with the turn line's ``grounding`` given, a ReGround off -> on set applied after
    t1 (a pass-2 factory given, R-13a) leaves t1's line with the grounding as built and t2's with ``reground`` true
    and ``reground_visual_tokens`` 280, both ``_from: "ui"``; the main budget and its source are unchanged."""
    from tests.unit.test_t0213_apply_boundary import BudgetLlm

    async with settings_rig(tmp_path, session_kwargs={"grounding": dict(GROUNDING_OFF)}) as (
        rig,
        _llm,
        _browser,
    ):
        rig.session.init_settings(
            rig.session.settings, {}, {}, pass2_factory=lambda: BudgetLlm(visual_tokens=280)
        )
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        mark = await set_and_pump(rig, {"reground": True, "overview_frame": False})
        applied = last_state(rig, mark)
        assert applied.rev == 1, f"premise: applied; refused {refused_pairs(applied)}"
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        lines = {x["turn_id"]: x for x in rig.raw_lines()}
    assert lines["t1"]["extras"].get("grounding") == GROUNDING_OFF, lines["t1"]["extras"]
    want = {
        **GROUNDING_OFF,
        "reground": True,
        "reground_from": "ui",
        "reground_visual_tokens": 280,
        "reground_visual_tokens_from": "ui",
    }
    assert lines["t2"]["extras"].get("grounding") == want, (
        f"T-0213 A2a-5: t2's grounding reflects the change; {lines['t2']['extras'].get('grounding')}"
    )


# =====================================================================================================
# R-19 (e) / R-20: applied_at tells the truth
# =====================================================================================================


async def test_a_stop_turn_whose_stop_did_not_act_is_logged_after_turn(tmp_path):
    """R-19 (e) (adversarial): a ``stop_turn=true`` set while t1's click executes, followed at once by the page's own
    ``astra.turn.stop`` for t1 (queued behind the set, so it reaches the machine before the stop the set submits):
    the page's stop ends t1, the set's stop is stepped at IDLE and does nothing, so the change is logged
    ``applied_at: "after_turn"``, not ``"after_stop"``. The companion (the set's stop cut t1: ``after_stop``) is
    ``test_stop_turn_is_logged_after_stop``.

    **Reading R20-2:** R-20 says "``stop_turn`` when the turn already ended on its own"; a turn that ends on its own
    in the one pump hop between the set and its stop cannot be scheduled deterministically over the fakes, so the
    row uses the other way a stepped stop fails to act: another stop got there first. Either way the set's stop did
    not act."""
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer_s3()])
        await _open_window(rig, llm, browser, "between_steps", gate)
        since = len(rig.machine.log)
        body = {"type": SET, "rev": 0, "changes": {"visual_tokens": NEW_BUDGET}, "stop_turn": True}
        rig.page.receive(json.dumps(body))
        rig.page.receive(json.dumps({"type": "astra.turn.stop", "turn_id": "t1"}))
        await rig.pump()
        await rig.advance(500, step=40)
        gate.set()
        await rig.advance(1_000, step=40)
        stops = [(b, a) for b, ev, a in rig.machine.transitions()[since:] if ev.type == "user.stop"]
        assert len(stops) == 2 and stops[0][0] != stops[0][1] and stops[1][0] == stops[1][1], (
            f"premise: two stops stepped, the first (the page's) acted, the second (the set's) did not; {stops}"
        )
        assert rig.session.settings.rev == 1, "premise: applied"
    data = assert_one_change(rig)
    assert data["applied_at"] == "after_turn", (
        f"T-0213 R-19 (e): after_stop only when the set's stop acted; got {data['applied_at']}"
    )
