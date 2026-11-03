"""T-0120b: tool-loop compaction wired into the session (docs/tasks/T-0120b.md criteria 1-9; rulings H-T115-0,
H-T120b-0..5).

Every row runs over the session fakes (``tests/unit/_tool_turn_support.tool_rig``): ``FakeLlm`` scripted per step,
``FakeSearch`` with canned maximal rows, ``RecordingToolExecutor`` outcomes, the ``ScriptedClock``. Expected
continuations are joined **in the row** from ``render_call``/``render_response`` (T-0113 proved them against the
template oracle) with the hand-typed stubs below in place of the elided responses; ``loop.compact`` is only ever a
second, agreeing oracle, never the only one.

**The stubs, by hand.** ``SEARCH_STUB_5`` is 16 (``<|tool_response>``) + 23 (``response:google_search{``) + 5
(``note:``) + 5 + 51 (the note) + 5 + 1 (``,``) + 9 (``results:5``) + 1 (``}``) + 16 = 132 chars. ``TYPE_AT_STUB``
(the old note, status form) is 16 + 17 (``response:type_at{``) + 5 + 5 + 22 + 5 + 1 + 7 (``status:``) + 5 + 7 + 5 + 1
+ 16 = 112.

**The search loop (``_search_loop``).** Step 1 speaks ``PREFIX`` (20 chars) and searches ``library 1``; step k >= 2
withholds a 1 213-char text (embedded cut to ``MAX_STEP_TEXT`` = 1 000) and searches ``library k``; the last step
reports. Each call ``google_search{query:"library k"}`` renders to 69 chars; each canned result set (5 rows, snippets
over 300 chars) renders, capped by ``MAX_RESPONSE``, to 1 498 chars for k = 1 (its first title carries ``SENTINEL``)
and 1 496 for k >= 2 (premises asserted in ``_search_premises``). So step 1 is 1 587 chars, every later step 2 565;
eliding step 1 saves 1 366, any later one 1 364. Budget ``MAX_CONTINUATION`` = 14 000, target 14 000 * 2 // 3 =
9 333, ``keep_last`` 2:

    request   executed  uncompacted (stored)   compaction                              sent
    s2..s6    1..5      1 587 .. 11 847        none                                    as is
    s7        6         14 412 > 14 000        elide 0,1,2,3 (10 318 > 9 333 after 3)  8 954
    s8        7         8 954 + 2 565 = 11 519 none                                    11 519
    s9        8         11 519 + 2 565 = 14 084 elide 4,5 (0-3 are stubs; 4,5 only)     11 356

A ``target=MAX_CONTINUATION`` mutant stops at s7 after one elision (13 046 <= 14 000); a session that recompacts the
uncompacted list at s8 elides 0..4 (16 977 -> 10 155) and breaks the s7 prefix; a size check before compaction ends
``over_budget`` at s7 (14 412).

**Still over (``_still_over_loop``).** Eight ``type_at`` steps (a 5 000-char text argument, embedded cut to
``MAX_ARG_TEXT`` = 500: the call renders to 583 chars). Calls 1-6 return a 9-char dry-run action (the response renders
to 113 + 9 = 122 chars, 10 more than its stub), calls 7-8 a 3 000-char action (1 500). Step 1 is 20 + 583 + 122 =
725, steps 2-6 are 1 000 + 583 + 122 = 1 705, steps 7-8 1 000 + 583 + 1 500 = 3 083. After 7 executed: 725 + 5 * 1 705
+ 3 083 = 12 333 (sent, no compaction); after 8: 15 416, and eliding all six candidates saves 60: 15 356 > 14 000. So
s2..s8 are sent and s9 ends ``over_budget`` with one compaction.

**Mixed (``_mixed_loop``).** Step 1 speaks ``PREFIX`` and calls ``type_at`` (3 000-char action, 1 500-char response:
20 + 583 + 1 500 = 2 103 chars); steps 2-6 search (2 565 each); step 7 reports. After 6 executed: 14 928 -> elide 0
(saves 1 388: 13 540), 1, 2, 3 (1 364 each: 9 448, still over 9 333, candidates exhausted). s7 holds one
``TYPE_AT_STUB`` and three ``SEARCH_STUB_5``.

Readings (listed in the test-author's report):

- the request numbering is the machine's: "the step-k request" is ``t1-gen-sk``, sent after k - 1 executed steps;
  the task's "six maximal search steps ... the step-6 request" is read as six executed steps and the s7 request;
- the over_budget note's ``chars`` is the size of the compacted continuation (the one the size check measures);
- the sentinel may appear in the ``tool.result`` record of the call that returned it (that record is the result,
  logged when it arrived, T-0114) besides the ``llm.request`` prompts sent before it was elided;
- ``elided`` is a list of ints in ascending order.

Active since ``/implement T-0120b`` started: the rows that need the implementation are red until it lands. Rows
that hold at the base (the negative criteria 5's loop-off half, 7, 9, the no-note row and the fixture provenance)
are green at the base.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word unless a row says otherwise,
tools ``dry_run``, ``max_tool_steps`` 8, n = 1 and deterministic per row. No measurement.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import session as session_mod
from local_astra.contracts import events as E
from local_astra.contracts.toolcall import ToolOutcome
from local_astra.contracts.turn import TurnState
from local_astra.toolcall import loop
from local_astra.toolcall.loop import LoopStep, render_call, render_response
from tests import _git_range as gr
from tests.fakes.search import FakeSearch
from tests.fakes.tool_executor import RecordingToolExecutor
from tests.unit import _pre_t0155_persona as P
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0117_differential as t17
from tests.unit._tool_turn_support import VALID_CLICK, ToolRig, answer, tool, tool_rig
from tests.unit.test_session_tools_off_golden import normalise

REPO = Path(__file__).resolve().parents[2]

#: H-T120b-5: T-0121's merge commit, the differential and scope base. T-0121 changed only the tools-off answer path,
#: and its own differential stayed byte-identical on marker-free turns, so for these marker-free fixtures it is
#: code-equivalent to 791dc54; it is also the last code commit before this branch's fork point (4391cbc changes two
#: task files only). A literal, never a moving ref (.claude/rules/tests.md).
BASE = "12485258e5a867b56afa280c2d407f5cde78a769"
FIXTURE_DIR = REPO / "tests" / "fixtures" / "session" / "loop_compaction_base_1248525"
FIXTURE = FIXTURE_DIR / "base.json"
#: T-0094 "pin both ends": the reviewed tip of T-0120b (879335b, rebased onto main, both reviewers approved), so the
#: criterion 9/10 scope rows compare the closed range ``BASE..TIP`` and never the working tree. A literal.
TIP = "879335bd225ca199621d3cff6e2d157016e8290a"
#: Power pair for the scope predicate: T-0121's own range, known to change ``session.py``.
POWER_BASE = "791dc54"
POWER_TIP = BASE

SEARCH_NOTE = "earlier result omitted; search again if you need it"
OLD_NOTE = "earlier result omitted"
Q = '<|"|>'
SEARCH_STUB_5 = (
    '<|tool_response>response:google_search{note:<|"|>earlier result omitted; search again if you need it<|"|>,'
    "results:5}<tool_response|>"
)
TYPE_AT_STUB = (
    '<|tool_response>response:type_at{note:<|"|>earlier result omitted<|"|>,status:<|"|>dry_run<|"|>}'
    "<tool_response|>"
)
COMPACTED = "tool loop compacted"
NOTE_KEYS = frozenset({"what", "request_id", "step", "elided", "chars_before", "chars_after"})
#: T-0114 criterion 11's note, unchanged in form (criterion 1): its ``what`` and its keys at the base.
OVER_WHAT = "tool loop switched off: the continuation is over budget"
OVER_KEYS = frozenset({"what", "request_id", "step", "chars", "max_chars"})
OVER_MESSAGE = "tool loop: continuation over budget"

M = 14_000
TARGET = M * 2 // 3
PREFIX = tsl.PREFIX
REPORT = tsl.REPORT
SENTINEL = "SENTINEL-R1"
TYPE_ARGS = tsl.TYPE_ARGS
TYPE_CALL = tsl.TYPE_CALL


def _new(name: str) -> Any:
    """A T-0120b name from ``loop``, or ``NotImplementedError`` while it does not exist."""
    try:
        return getattr(loop, name)
    except AttributeError as exc:
        raise NotImplementedError(f"T-0120b: loop.{name} does not exist yet") from exc


# --------------------------------------------------------------------------- the fixture loops


def query(k: int) -> str:
    return f"library {k}"


def search_call(k: int) -> str:
    return f"<|tool_call>call:google_search{{query:{Q}{query(k)}{Q}}}<tool_call|>"


def rows(k: int) -> list[dict[str, str]]:
    return [
        {
            "title": (f"{SENTINEL} " if k == 1 and i == 0 else "") + f"Library {k}.{i} overview",
            "url": f"https://libs.example.org/{k}/{i}",
            "snippet": f"Result {k}.{i}: " + "lorem ipsum dolor " * 20,
        }
        for i in range(5)
    ]


def raw_text(k: int) -> str:
    """What the model says before call k: the voiced prefix at step 1, a withheld 1 213-char text after."""
    return PREFIX if k == 1 else f"Step {k} notes " + "w" * 1200


def search_step(k: int) -> tuple[str, str, Mapping[str, Any], Any]:
    """Executed step k as embedded: the bounded text, the call, the ``search_payload`` of the canned rows."""
    return (
        raw_text(k)[:1000],
        "google_search",
        {"query": query(k)},
        loop.search_payload(tsl.results_of(rows(k), query(k))),
    )


def type_step(k: int, action: str) -> tuple[str, str, Mapping[str, Any], Any]:
    args = {"press_enter": False, "text": "T" * 500, "x": 500, "y": 250}
    return (raw_text(k)[:1000], "type_at", args, loop.action_payload(action))


def hand_join(
    steps: Sequence[tuple[str, str, Mapping[str, Any], Any]], stubs: Mapping[int, str]
) -> str:
    """The continuation joined here: text + call + response per step, the responses at ``stubs`` replaced by the
    hand-typed stub given, leading whitespace dropped (the template's trim)."""
    return "".join(
        t + render_call(n, a) + (stubs[i] if i in stubs else render_response(n, p))
        for i, (t, n, a, p) in enumerate(steps)
    ).lstrip()


def stubbed(indices: Sequence[int], stub: str = SEARCH_STUB_5) -> dict[int, str]:
    return dict.fromkeys(indices, stub)


def _search_premises() -> None:
    """The fixture numbers of the module docstring; if these move, every hand number moves with them."""
    assert len(SEARCH_STUB_5) == 132 and len(TYPE_AT_STUB) == 112
    assert len(render_call("google_search", {"query": query(1)})) == 69
    assert [len(render_response("google_search", search_step(k)[3])) for k in range(1, 10)] == [
        1498
    ] + [1496] * 8
    assert SENTINEL in render_response("google_search", search_step(1)[3])
    assert len(render_call("type_at", type_step(1, "")[2])) == 583
    assert len(render_response("type_at", loop.action_payload("a" * 9))) == 122
    assert len(render_response("type_at", loop.action_payload("A" * 3000))) == 1500
    steps = [search_step(k) for k in range(1, 9)]
    assert [len(hand_join(steps[:n], {})) for n in range(1, 7)] == [
        1587,
        4152,
        6717,
        9282,
        11847,
        14412,
    ]
    # the s7 compaction: after three elisions still over the target, after four at 8 954
    assert len(hand_join(steps[:6], stubbed([0, 1, 2]))) == 10_318 > TARGET
    assert len(hand_join(steps[:6], stubbed([0, 1, 2, 3]))) == 8954
    assert (
        len(hand_join(steps[:6], stubbed([0]))) == 13_046 <= M
    )  # the target=MAX_CONTINUATION mutant
    assert len(hand_join(steps[:7], stubbed([0, 1, 2, 3]))) == 11_519 <= M
    assert len(hand_join(steps[:8], stubbed([0, 1, 2, 3]))) == 14_084 > M
    assert len(hand_join(steps[:8], stubbed([0, 1, 2, 3, 4]))) == 12_720 > TARGET
    assert len(hand_join(steps[:8], stubbed(range(6)))) == 11_356


async def _run(rig: ToolRig, limit_ms: int = 40_000) -> None:
    await rig.ask(tsl.QUESTION)
    await tsl.until_steps(rig, 1)
    await rig.turn_done(limit_ms=limit_ms)


async def _search_loop(tmp_path: Path, searches: int) -> ToolRig:
    """``searches`` executed searches, then the report (module docstring)."""
    search = FakeSearch({query(k): rows(k) for k in range(1, searches + 1)})
    async with tool_rig(
        tmp_path,
        policy=tsl.loop_policy(8),
        search=search,
        script=[answer(PREFIX), tool(search_call(1))],
    ) as rig:
        rig.llm.script_steps(
            *[[answer(raw_text(k)), tool(search_call(k))] for k in range(2, searches + 1)],
            [answer(REPORT)],
        )
        await _run(rig)
    return rig


def _outcome(cid: str, action: str) -> ToolOutcome:
    return ToolOutcome(call_id=cid, status="dry_run", action=action, reason=None, resolved={})


async def _still_over_loop(tmp_path: Path) -> ToolRig:
    actions = {tsl.call_id(k): ("a" * 9 if k <= 6 else "A" * 3000) for k in range(1, 10)}
    executor = RecordingToolExecutor({c: _outcome(c, a) for c, a in actions.items()})
    async with tool_rig(
        tmp_path,
        policy=tsl.loop_policy(8),
        executor=executor,
        script=[answer(PREFIX), tool(TYPE_CALL)],
    ) as rig:
        rig.llm.script_steps(*[[answer(raw_text(k)), tool(TYPE_CALL)] for k in range(2, 10)])
        await _run(rig)
    return rig


async def _mixed_loop(tmp_path: Path) -> ToolRig:
    executor = RecordingToolExecutor({tsl.call_id(1): _outcome(tsl.call_id(1), "A" * 3000)})
    search = FakeSearch({query(k): rows(k) for k in range(2, 7)})
    async with tool_rig(
        tmp_path,
        policy=tsl.loop_policy(8),
        executor=executor,
        search=search,
        script=[answer(PREFIX), tool(TYPE_CALL)],
    ) as rig:
        rig.llm.script_steps(
            *[[answer(raw_text(k)), tool(search_call(k))] for k in range(2, 7)],
            [answer(REPORT)],
        )
        await _run(rig)
    return rig


def compactions(rig: ToolRig) -> list[dict[str, Any]]:
    return tsl.notes(rig, COMPACTED)


def tool_compactions(rig: ToolRig, turn: str = "t1") -> Any:
    extras = rig.record(turn).extras
    assert "tool_compactions" in extras, (
        f"T-0120b criterion 5: a loop-on turn line carries tool_compactions; it holds {sorted(extras)}"
    )
    return extras["tool_compactions"]


def sent_ids(rig: ToolRig) -> list[int]:
    return [k for k in range(2, 12) if tsl.generates_for(rig, tsl.step_id(k))]


def llm_errors(rig: ToolRig, k: int) -> list[str]:
    return [e.message for e in rig.machine.events(E.LlmError) if e.request_id == tsl.step_id(k)]


# =====================================================================================================
# Criterion 1: compaction before the size check
# =====================================================================================================


async def test_compacts_before_over_budget(tmp_path):
    """Criterion 1. Six executed maximal searches are 14 412 chars uncompacted (over 14 000); compacted, 8 954. The
    s7 request is sent, carrying exactly the hand-joined continuation with steps 0-3 stubbed (and it equals
    ``step_prompt(base, compact(steps, 14 000))``); no ``LlmError`` for s7; the loop ends ``final``."""
    _search_premises()
    rig = await _search_loop(tmp_path, 6)
    steps = [search_step(k) for k in range(1, 7)]
    assert sent_ids(rig) == [2, 3, 4, 5, 6, 7], (
        f"T-0120b criterion 1: s2..s7 are sent; step requests {[r for r, _ in rig.llm.step_prompts]}; "
        f"notes {tsl.notes(rig)}"
    )
    assert llm_errors(rig, 7) == [], llm_errors(rig, 7)
    end = rig.record("t1").extras.get("tool_loop_end")
    assert end == "final" and end != "over_budget", end
    s7 = tsl.step_request(rig, 7)
    want = hand_join(steps, stubbed([0, 1, 2, 3]))
    assert len(want) == 8954
    assert tsl.tail(s7) == want, (
        f"T-0120b criterion 1: the s7 continuation is the compacted one ({len(tsl.tail(s7))} chars, want 8 954)"
    )
    pass1 = tsl.sent(rig, "t1-gen")
    oracle = loop.step_prompt(pass1, loop.compact([LoopStep(*s) for s in steps], M))
    assert s7 == oracle, (
        "T-0120b criterion 1: the s7 request is step_prompt(base, compact(steps, 14 000))"
    )
    assert REPORT in rig.spoken_sources()


def report_split(text: str, base: str) -> str:
    """T-0178 B1/B7b: ``text`` is ``base`` (the BASE continuation) followed by text that holds ``REPORT_FACT`` and no
    response or call marker; returns that appended text. Any separator is the implementer's."""
    from tests.unit.test_loop_report_fact import H_T178_3

    assert text.startswith(base), (
        f"T-0178 B1: the report step keeps BASE's continuation as its prefix ({len(text)})"
    )
    extra = text[len(base) :]
    assert H_T178_3 in extra, (
        f"T-0178 B1: REPORT_FACT follows the last tool response; got {extra[:200]!r}"
    )
    assert (
        "<|tool_response>" not in extra
        and "<tool_response|>" not in extra
        and "<|tool_call>" not in extra
    )
    return extra


# T-0178 B7b re-scope: s9 is the report step (max_tool_steps 8), so it carries REPORT_FACT after BASE's bytes
async def test_every_sent_continuation_fits(tmp_path):
    """Criterion 1/6 (the size check is on what is sent): over the eight-search loop every sent continuation is at
    most ``MAX_CONTINUATION``, and the uncompacted s7 and s9 continuations (14 412, 14 084) would not have been.
    T-0178 B7b: s9 is the report step, so its continuation is BASE's 11 356 chars followed by REPORT_FACT."""
    _search_premises()
    rig = await _search_loop(tmp_path, 8)
    assert sent_ids(rig) == list(range(2, 10)), [r for r, _ in rig.llm.step_prompts]
    sizes = {k: len(tsl.tail(tsl.step_request(rig, k))) for k in range(2, 10)}
    assert {k: v for k, v in sizes.items() if k != 9} == {
        2: 1587,
        3: 4152,
        4: 6717,
        5: 9282,
        6: 11847,
        7: 8954,
        8: 11519,
    }, sizes
    steps = [search_step(k) for k in range(1, 9)]
    base9 = hand_join(steps, stubbed(range(6)))
    assert len(base9) == 11356
    report_split(tsl.tail(tsl.step_request(rig, 9)), base9)
    assert max(sizes.values()) <= M


# =====================================================================================================
# Criterion 3: the compacted steps are kept (prefix stability)
# =====================================================================================================


async def test_compacted_steps_are_kept(tmp_path):
    """Criterion 3, adversarial against recompacting the uncompacted list each step. After the compaction at s7 the
    s8 request (11 519 chars, fits) starts with the s7 continuation byte for byte and equals the hand join with
    steps 0-3 still stubbed; one compaction in the turn. (A session recompacting the stored-but-uncompacted list
    at s8 elides 0..4 and sends 10 155 chars.)"""
    _search_premises()
    rig = await _search_loop(tmp_path, 7)
    assert sent_ids(rig) == list(range(2, 9)), [r for r, _ in rig.llm.step_prompts]
    s7, s8 = tsl.tail(tsl.step_request(rig, 7)), tsl.tail(tsl.step_request(rig, 8))
    assert s8.startswith(s7), (
        f"T-0120b criterion 3: the s8 continuation extends the s7 one (s7 {len(s7)}, s8 {len(s8)} chars)"
    )
    steps = [search_step(k) for k in range(1, 8)]
    assert s8 == hand_join(steps, stubbed([0, 1, 2, 3]))
    assert tool_compactions(rig) == 1
    assert len(compactions(rig)) == 1


# T-0178 B7b re-scope: s9 is the report step, so it is the hand join followed by REPORT_FACT
async def test_second_compaction_keeps_earlier_stubs(tmp_path):
    """Criterion 3, second half: a further compaction at s9 elides 4 and 5; every response elided at s7 (0-3) is
    still elided with identical bytes, so s7 and s9 share the hand-joined prefix of steps 0-3 stubbed, and s9 is the
    hand join with 0-5 stubbed (11 356). Two compactions. T-0178 B7b: s9 is the report step, so that hand join is
    followed by REPORT_FACT (``report_split``)."""
    _search_premises()
    rig = await _search_loop(tmp_path, 8)
    steps = [search_step(k) for k in range(1, 9)]
    head = hand_join(steps[:4], stubbed([0, 1, 2, 3]))
    s7, s9 = tsl.tail(tsl.step_request(rig, 7)), tsl.tail(tsl.step_request(rig, 9))
    assert s7.startswith(head) and s9.startswith(head)
    report_split(s9, hand_join(steps, stubbed(range(6))))
    assert tool_compactions(rig) == 2
    assert rig.record("t1").extras.get("tool_loop_end") == "final"


# =====================================================================================================
# Criterion 4: the note
# =====================================================================================================


async def test_note_fields(tmp_path):
    """Criterion 4. Two ``tool loop compacted`` notes, exactly these keys and values: s7 elides [0, 1, 2, 3] (14 412
    -> 8 954) and s9 the newly elided [4, 5] only (14 084 -> 11 356); ``chars_before`` and ``chars_after`` equal
    ``continuation_chars`` of the steps before and after; both fitting compactions have ``chars_before > 14 000 >=
    chars_after``. No note carries step or response text."""
    _search_premises()
    rig = await _search_loop(tmp_path, 8)
    found = compactions(rig)
    for n in found:
        assert set(n) == NOTE_KEYS, (
            f"T-0120b criterion 4: the note's keys are exactly {sorted(NOTE_KEYS)}; got {sorted(n)}"
        )
    assert found == [
        {
            "what": COMPACTED,
            "request_id": "t1-gen-s7",
            "step": 7,
            "elided": [0, 1, 2, 3],
            "chars_before": 14_412,
            "chars_after": 8954,
        },
        {
            "what": COMPACTED,
            "request_id": "t1-gen-s9",
            "step": 9,
            "elided": [4, 5],
            "chars_before": 14_084,
            "chars_after": 11_356,
        },
    ], f"T-0120b criterion 4: the compaction notes; got {found}"
    steps = [LoopStep(*search_step(k)) for k in range(1, 9)]
    assert loop.continuation_chars(steps[:6]) == 14_412
    assert loop.continuation_chars(loop.compact(steps[:6], M)) == 8954
    for n in found:
        assert n["chars_before"] > M >= n["chars_after"]
        blob = json.dumps(n)
        for text in (
            SENTINEL,
            OLD_NOTE,
            "<|tool_response>",
            "<|tool_call>",
            "www",
            "library",
            "lorem",
        ):
            assert text not in blob, (
                f"T-0120b criterion 4: the note carries no step or tool text ({text!r}): {n}"
            )


async def test_elided_text_leaves_the_log(tmp_path):
    """Criterion 4 (the note carries no tool-result text) and criterion 3 at the log: the sentinel planted in step
    1's result appears only in the ``tool.result`` record of that call and in the ``llm.request`` prompts of s2..s6,
    sent before it was elided; the s7, s8 and s9 prompts (which exist) do not hold it."""
    rig = await _search_loop(tmp_path, 8)
    holders: list[str] = []
    for r in rig.log_records:
        if SENTINEL not in json.dumps(r["data"], ensure_ascii=False):
            continue
        data = r["data"]
        holders.append(f"{r['kind']}:{data.get('request_id') or data.get('call_id')}")
    want = ["tool.result:t1-call"] + [f"llm.request:{tsl.step_id(k)}" for k in range(2, 7)]
    assert sorted(holders) == sorted(want), (
        f"T-0120b: the sentinel is in exactly {sorted(want)}; found in {sorted(holders)}"
    )
    requested = [r["request_id"] for r in rig.kind("llm.request") if r.get("phase") == "tool_step"]
    assert requested == [tsl.step_id(k) for k in range(2, 10)], requested


async def test_no_note_when_fitting(tmp_path):
    """Criteria 4 and 8 (negative; holds at the base): an S1-like loop (two searches of the default rows, then the
    report) fits every step, so no ``tool loop compacted`` note and no note that mentions compaction is written."""
    rig = await _fitting_search_loop(tmp_path)
    assert sent_ids(rig) == [2, 3], [r for r, _ in rig.llm.step_prompts]
    assert compactions(rig) == []
    assert [n for n in tsl.notes(rig) if "compact" in json.dumps(n)] == []


# =====================================================================================================
# Criterion 5: the turn line
# =====================================================================================================


async def _fitting_search_loop(tmp_path: Path) -> ToolRig:
    async with tool_rig(
        tmp_path,
        policy=tsl.LOOP,
        search=FakeSearch(),
        script=[answer(PREFIX), tool(tsl.SEARCH_CALL)],
    ) as rig:
        rig.llm.script_steps([answer("And the forecast. "), tool(search_call(2))], [answer(REPORT)])
        await _run(rig)
    return rig


async def test_tool_compactions_zero_on_fitting_loop(tmp_path):
    """Criterion 5 (H-T120b-3): a loop-on line where nothing was compacted carries ``tool_compactions: 0``, an int."""
    rig = await _fitting_search_loop(tmp_path)
    value = tool_compactions(rig)
    assert type(value) is int and value == 0, repr(value)


async def test_tool_compactions_counts_the_call_that_still_ends_over(tmp_path):
    """Criterion 5: the mixed loop compacts once (``tool_compactions == 1``); the still-over loop compacts once and
    ends ``over_budget`` (also 1: the count includes a compaction that did not save the loop)."""
    rig = await _mixed_loop(tmp_path / "mixed")
    assert tool_compactions(rig) == 1
    rig = await _still_over_loop(tmp_path / "over")
    assert tool_compactions(rig) == 1
    assert rig.record("t1").extras.get("tool_loop_end") == "over_budget"


async def test_tool_compactions_absent_loop_off(tmp_path):
    """Criterion 5 (negative; holds at the base): loop off, tools on (a tool turn and a plain turn) and tools off,
    no turn line and no ``turn.record`` log record carries ``tool_compactions``."""
    async with tool_rig(tmp_path / "on", script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        await rig.ask("click the sign in button")
        await rig.turn_done()
        await rig.advance(300)
        rig.llm.script = [answer("It says welcome back.")]
        await rig.ask("and what does it say now")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "turn 2"
        )
    async with tool_rig(tmp_path / "off", kit=None, script=[answer("Hello there.")]) as off:
        await off.ask("hello")
        await off.turn_done()
    for r in (rig, off):
        lines = [json.loads(line) for line in r.lines()]
        assert lines
        for line in lines:
            assert "tool_compactions" not in line["extras"], sorted(line["extras"])
        for rec in r.kind("turn.record"):
            assert "tool_compactions" not in rec.get("extras", {}), sorted(rec.get("extras", {}))


# =====================================================================================================
# Criterion 1 / T-0120 criterion 6: still over after compaction
# =====================================================================================================


async def test_still_over_ends_over_budget(tmp_path):
    """Criterion 1 (T-0120 criterion 6). Eight executed ``type_at`` steps are 15 416 chars; eliding all six
    candidates saves 60 (15 356 > 14 000). So s2..s8 are sent, s9 is not, ``LlmError("t1-gen-s9", ...)`` reaches the
    machine, the T-0114 note is written (its keys unchanged) after one compaction note, the HUD is told, the line
    says ``over_budget`` and ``tool_compactions == 1``."""
    _search_premises()
    steps = [type_step(k, "a" * 9 if k <= 6 else "A" * 3000) for k in range(1, 9)]
    assert len(hand_join(steps[:7], {})) == 12_333 <= M
    assert len(hand_join(steps, {})) == 15_416
    assert len(hand_join(steps, stubbed(range(6), TYPE_AT_STUB))) == 15_356 > M
    rig = await _still_over_loop(tmp_path)
    assert sent_ids(rig) == list(range(2, 9)), [r for r, _ in rig.llm.step_prompts]
    assert llm_errors(rig, 9) == [OVER_MESSAGE], llm_errors(rig, 9)
    found = compactions(rig)
    assert found == [
        {
            "what": COMPACTED,
            "request_id": "t1-gen-s9",
            "step": 9,
            "elided": [0, 1, 2, 3, 4, 5],
            "chars_before": 15_416,
            "chars_after": 15_356,
        }
    ], found
    over = tsl.notes(rig, OVER_WHAT)
    assert len(over) == 1 and set(over[0]) == OVER_KEYS, over
    # T-0178 B7b re-scope, tightened by amendment B1a ("the fact is appended, and the size check counts it";
    # mutant M19, the fact not counted, reports BASE's 15 356): s9 is the report step, so the note's chars is the
    # compacted 15 356 plus ``REPORT_SEPARATOR + REPORT_FACT`` (H-T178-3's 160 chars and the separator). Compacting
    # against the reduced target elides the same six candidates (all of them), so the compaction note is unchanged.
    from tests.unit.test_loop_report_fact import H_T178_3

    assert loop.REPORT_FACT == H_T178_3 and len(H_T178_3) == 160
    chars = over[0].get("chars")
    assert chars == 15_356 + len(loop.REPORT_SEPARATOR) + 160, (
        f"T-0178 B1a: the size check counts the fact; the over_budget note says {chars} chars"
    )
    assert {k: v for k, v in over[0].items() if k != "chars"} == {
        "what": OVER_WHAT,
        "request_id": "t1-gen-s9",
        "step": 9,
        "max_chars": M,
    }, over
    order = [n.get("what") for n in tsl.notes(rig) if n.get("what") in (COMPACTED, OVER_WHAT)]
    assert order == [COMPACTED, OVER_WHAT], order
    assert any(getattr(m, "type", "") == "astra.error" for m in rig.published.messages)
    assert rig.record("t1").extras.get("tool_loop_end") == "over_budget"
    assert tool_compactions(rig) == 1


# =====================================================================================================
# Criterion 2: the stub at session level
# =====================================================================================================


async def test_search_stub_invites_research(tmp_path):
    """Criterion 2 (H-T120b-0, H-T120b-1): the s7 request of the mixed loop holds the hand-typed 132-char search stub
    three times (steps 1-3) and the old-note ``type_at`` stub once (step 0), which says nothing about searching;
    the whole continuation is the hand join (9 448 chars)."""
    rig = await _mixed_loop(tmp_path)
    assert sent_ids(rig) == list(range(2, 8)), [r for r, _ in rig.llm.step_prompts]
    text = tsl.tail(tsl.step_request(rig, 7))
    steps = [type_step(1, "A" * 3000)] + [search_step(k) for k in range(2, 7)]
    want = hand_join(steps, {0: TYPE_AT_STUB, 1: SEARCH_STUB_5, 2: SEARCH_STUB_5, 3: SEARCH_STUB_5})
    assert len(want) == 9448
    assert text == want, f"T-0120b criterion 2: s7 is the hand join ({len(text)} chars, want 9 448)"
    assert text.count(SEARCH_STUB_5) == 3
    assert text.count(TYPE_AT_STUB) == 1
    action_stub = text[text.index("<|tool_response>response:type_at") :].split("<tool_response|>")[
        0
    ]
    assert "search" not in action_stub and OLD_NOTE in action_stub, action_stub
    assert _new("ELIDED_SEARCH_NOTE") == SEARCH_NOTE
    assert loop.ELIDED_NOTE == OLD_NOTE


# =====================================================================================================
# Criterion 6 is ``test_session_tool_loop.test_over_budget_ends_loop_and_publishes`` (re-scoped there).
# Criteria 7-8: byte-identical to the base on marker-free fixtures
# =====================================================================================================

#: what each scenario compares. Criterion 7: every session-log line, prompt, request body, publication, history row
#: and turn line. Criterion 8: every ``llm.request`` and ``note`` record, prompt, request body and turn line.
LOOP_OFF_SECTIONS = ("hud", "prompts", "requests", "log", "records", "history")
FITTING_SECTIONS = ("prompts", "requests", "log", "records")
FITTING_KINDS = ("llm.request", "note")

#: the files whose base blobs the fixture records (``git rev-parse <BASE>:<path>``)
PROVENANCE_PATHS = (
    "local_astra/conductor/session.py",
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/turn_machine.py",
    "tests/unit/_tool_turn_support.py",
    "tests/unit/test_session_loop.py",
    "tests/unit/test_session_tools_off_golden.py",
    "tests/unit/test_t0117_differential.py",
    "tests/fakes/llm.py",
    "tests/fakes/search.py",
)


def _sections(rig: ToolRig, tmp: Path) -> dict[str, Any]:
    out = t17._sections(rig, tmp, Path(rig.log.path), rig.session.session_id)
    history = [h.model_dump(mode="json") for h in rig.state.history]
    out["history"] = normalise(
        {"history": history}, rig.session.session_id, tmp, Path(rig.log.path)
    )["history"]
    return out


async def _loop_off_session(tmp: Path) -> dict[str, Any]:
    """Tools on, loop off: a plain turn, a tool turn, then a turn with history."""
    tmp.mkdir(parents=True, exist_ok=True)
    turns = (
        ("what is on the screen", (answer("A settings page with a sign in button."),)),
        ("click the sign in button", (answer(tsl.S1), tool(VALID_CLICK))),
        ("and what does it say now", (answer("It says welcome back."),)),
    )
    async with tool_rig(tmp, machine_cls=t17._PolicyMachine) as rig:
        for i, (question, script) in enumerate(turns):
            rig.llm.script = list(script)
            await rig.ask(question)
            await rig.until(
                lambda n=i + 1: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
                f"loop-off turn {i + 1} never ended",
            )
            await rig.advance(300)
    return _sections(rig, tmp)


async def _tools_off_session(tmp: Path) -> dict[str, Any]:
    """Tools off: a plain turn, then a turn with history."""
    tmp.mkdir(parents=True, exist_ok=True)
    turns = (("hello there", "Hello, how can I help?"), ("tell me more", "There is not much more."))
    async with tool_rig(tmp, kit=None, machine_cls=t17._PolicyMachine) as rig:
        for i, (question, text) in enumerate(turns):
            rig.llm.script = [answer(text)]
            await rig.ask(question)
            await rig.until(
                lambda n=i + 1: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
                f"tools-off turn {i + 1} never ended",
            )
            await rig.advance(300)
    return _sections(rig, tmp)


async def _fitting_session(tmp: Path) -> dict[str, Any]:
    """Loop on, every continuation fits: an S1-like three-step search turn, then a two-step action turn."""
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(
        tmp,
        machine_cls=t17._PolicyMachine,
        policy=tsl.LOOP,
        search=FakeSearch(),
        script=[answer(PREFIX), tool(tsl.SEARCH_CALL)],
    ) as rig:
        rig.llm.script_steps([answer("And the forecast. "), tool(search_call(2))], [answer(REPORT)])
        await tsl.loop_turn(rig, 3)
        await rig.advance(300)
        rig.llm.script = [answer(tsl.S1), tool(VALID_CLICK)]
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask("open the settings")
        await tsl.until_steps(rig, 1, turn="t2")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "t2 ended"
        )
        await rig.advance(300)
    return _sections(rig, tmp)


async def _capture(tmp: Path) -> dict[str, dict[str, Any]]:
    return {
        "loop_off": await _loop_off_session(tmp / "loop_off"),
        "tools_off": await _tools_off_session(tmp / "tools_off"),
        "fitting": await _fitting_session(tmp / "fitting"),
    }


def _select(scenario: str, sections: Mapping[str, Any]) -> dict[str, list[Any]]:
    if scenario == "fitting":
        out = {s: list(sections[s]) for s in FITTING_SECTIONS}
        out["log"] = [r for r in out["log"] if r["kind"] in FITTING_KINDS]
        return out
    return {s: list(sections[s]) for s in LOOP_OFF_SECTIONS}


def _item_hash(item: Any) -> str:
    return hashlib.sha256(
        json.dumps(item, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def _digest(captured: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    return {
        scenario: {
            s: [_item_hash(i) for i in items] for s, items in _select(scenario, secs).items()
        }
        for scenario, secs in captured.items()
    }


def _git_blob(path: Path) -> str:
    return subprocess.run(
        ["git", "hash-object", str(path)], capture_output=True, text=True, check=True
    ).stdout.strip()


def generate_base_fixture() -> str:
    """The fixture's JSON, captured from the tree this process imports. Run it from a detached worktree at ``BASE``
    (see the fixture's MANIFEST.md); it refuses unless ``local_astra`` and the harness come from that tree and the
    tree is ``BASE``."""
    import asyncio
    import tempfile

    import local_astra

    root = Path.cwd().resolve()
    origin = Path(local_astra.__file__).resolve()
    assert origin.is_relative_to(root), f"origin proof: local_astra from {origin}, not {root}"
    assert Path(tsl.__file__).resolve().is_relative_to(root), tsl.__file__
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head == BASE, f"the generator runs at {BASE}, not {head}"
    with tempfile.TemporaryDirectory() as tmp:
        captured = asyncio.run(_capture(Path(tmp)))
        again = asyncio.run(_capture(Path(tmp) / "again"))
    digest = _digest(captured)
    assert digest == _digest(again), "the capture is not deterministic over two runs"
    return json.dumps(
        {
            "base": BASE,
            "origin": str(origin.relative_to(root)),
            "blobs": {p: _git_blob(root / p) for p in PROVENANCE_PATHS},
            "digest": digest,
        },
        indent=1,
        sort_keys=True,
    )


def _fixture() -> dict[str, Any]:
    assert FIXTURE.is_file(), f"the base fixture {FIXTURE.relative_to(REPO)} is missing"
    data: dict[str, Any] = json.loads(FIXTURE.read_text())
    return data


def _strip_compactions(scenario: str, sections: dict[str, Any]) -> None:
    """Criterion 8: a loop-on turn line differs from the base only by ``tool_compactions: 0``."""
    for record in sections["records"]:
        extras = record["extras"]
        assert "tool_compactions" in extras, (
            f"T-0120b criterion 8: the loop-on line ({scenario}) carries tool_compactions; it holds {sorted(extras)}"
        )
        assert extras.pop("tool_compactions") == 0, (
            "T-0120b criterion 8: tool_compactions is 0 on a fitting loop"
        )


def _compare(scenario: str, sections: Mapping[str, Any], base: Mapping[str, list[str]]) -> None:
    now = _select(scenario, sections)
    assert set(now) == set(base), (sorted(now), sorted(base))
    for section, items in now.items():
        hashes = [_item_hash(i) for i in items]
        expected = base[section]
        first = next(
            (i for i, (a, b) in enumerate(zip(expected, hashes, strict=False)) if a != b),
            min(len(expected), len(hashes)) if len(expected) != len(hashes) else None,
        )
        shown = (
            json.dumps(items[first], sort_keys=True)[:1200]
            if first is not None and first < len(items)
            else None
        )
        assert hashes == expected, (
            f"T-0120b: {scenario} {section!r} differs from {BASE[:9]} (lengths {len(expected)} vs {len(hashes)}; "
            f"first differing index {first}; now: {shown})"
        )


def without_persona(sections: Mapping[str, Any], *, persona: str | None = None) -> dict[str, Any]:
    """T-0155 criterion 7: the observed sections with each system message's one persona sentence turned back into
    the pre-T-0155 one (raises on zero or more than one per message); the item hashes are then computed over the
    normalised items. The fixture's digests are never edited."""
    out, seen = P.normalise_system_messages(dict(sections), persona=persona)
    assert seen > 0, "premise: the capture holds system messages"
    # T-0162 C (pre-authorised pin move, amendment 2026-10-27): step/result on astra.tool.call only
    from tests.unit._t0162c_pins import without_tool_result_fields

    # T-0177 B-P1 (pre-authorised pin move): the published astra.vision.frame messages and their wire.out records,
    # shape checked, are the one declared HUD/log difference.
    from tests.unit._t0177_pins import without_vision_frames

    return dict(without_vision_frames(without_tool_result_fields(dict(out))))


async def test_persona_is_the_only_allowed_difference(tmp_path):
    """T-0155 criterion 7 (adversarial, either world): the loop-off session with its persona made Gearvis hashes to
    the base digests once normalised; with one more prompt character changed its prompts and request bodies do not;
    two persona sentences per system message raise."""
    base = _fixture()["digest"]["loop_off"]
    sections = await _loop_off_session(tmp_path / "loop_off")
    _compare("loop_off", without_persona(P.as_t0155(sections), persona=P.T0155_NAME), base)
    mutant = _select(
        "loop_off", without_persona(P.one_more_character(sections), persona=P.T0155_NAME)
    )
    for section in ("prompts", "requests"):
        assert [_item_hash(i) for i in mutant[section]] != base[section], (
            f"T-0155 criterion 7: a one-character prompt change outside the persona passes the {section!r} digest"
        )
    with pytest.raises(P.PersonaCountError):
        without_persona(P.two_personas(sections), persona=P.T0155_NAME)


async def test_loop_off_byte_identical_to_base(tmp_path):
    """Criterion 7 (negative; holds at the base): tools on with the loop off (a plain turn, a tool turn, a turn with
    history) and tools off (a plain turn, a turn with history): every session-log record, prompt, request body, HUD
    publication, history row and turn line is the base's, item for item (volatile fields normalised as the
    tools-off golden does, plus the log's pid and byte count)."""
    base = _fixture()["digest"]
    captured = {
        "loop_off": await _loop_off_session(tmp_path / "loop_off"),
        "tools_off": await _tools_off_session(tmp_path / "tools_off"),
    }
    for scenario, sections in captured.items():
        _compare(scenario, without_persona(sections), base[scenario])  # T-0155 criterion 7


async def test_fitting_loop_byte_identical_to_base(tmp_path):
    """Criterion 8: loop on, every continuation fits (an S1-like three-step search, a two-step action loop): every
    ``llm.request`` and ``note`` record, prompt and request body is the base's; each turn line is the base's plus
    ``tool_compactions: 0``; no ``tool loop compacted`` note."""
    base = _fixture()["digest"]
    sections = await _fitting_session(tmp_path / "fitting")
    assert [
        r for r in sections["log"] if r["kind"] == "note" and r["data"].get("what") == COMPACTED
    ] == []
    _strip_compactions("fitting", sections)
    _compare("fitting", without_persona(sections), base["fitting"])  # T-0155 criterion 7


def test_fixture_provenance():
    """The fixture names ``BASE`` and the base blob of each file it depends on, re-checked against
    ``git rev-parse <BASE>:<path>`` (fixed blob against fixed literal)."""
    gr.require_commits(BASE)
    data = _fixture()
    assert data["base"] == BASE
    assert data["origin"] == "local_astra/__init__.py", data["origin"]
    assert set(data["blobs"]) == set(PROVENANCE_PATHS)
    for path, blob in data["blobs"].items():
        want = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", f"{BASE}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert blob == want, f"{path}: the fixture's blob {blob} is not {BASE[:9]}'s {want}"
    assert set(data["digest"]) == {"loop_off", "tools_off", "fitting"}
    # premise: the scenarios ran what they say (three, two and two turn lines; nothing empty)
    lines = {scenario: len(secs["records"]) for scenario, secs in data["digest"].items()}
    assert lines == {"loop_off": 3, "tools_off": 2, "fitting": 2}, lines
    for scenario, secs in data["digest"].items():
        assert all(secs.values()), (scenario, {s: len(v) for s, v in secs.items()})


# =====================================================================================================
# Criterion 9: scope
# =====================================================================================================

UNCHANGED = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "local_astra/conductor/prompt_builder.py",
    "tests/unit/test_toolcall_loop_render.py",
    "tests/unit/test_toolcall_loop_injection.py",
    "tests/cd/unit/test_toolcall_loop_oracle.py",
    "tests/unit/test_session_tools_off_golden.py",
    "docs/cd",
)
#: (path, name, value): the constants criterion 9 keeps, their values typed from the tasks (H-T114-4, T-0113, H-T120-1)
CONSTANTS = (
    ("local_astra/conductor/session.py", "MAX_CONTINUATION", 14_000),
    ("local_astra/conductor/session.py", "MAX_STEP_TEXT", 1000),
    ("local_astra/conductor/session.py", "MAX_ARG_TEXT", 500),
    ("local_astra/toolcall/loop.py", "MAX_RESPONSE", 1500),
    ("local_astra/toolcall/loop.py", "COMPACT_KEEP_LAST", 2),
)
T0115A = "scripts/gpu/t0115a_loop_requests.py"


def _changed_since_base(*paths: str) -> list[str]:
    """Paths under ``paths`` that differ between the commits ``BASE`` and ``TIP`` (closed range, T-0094)."""
    proc = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", BASE, TIP, "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return sorted(proc.stdout.split())


def _module_constants(source: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in ast.parse(source).body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        value = getattr(node, "value", None)
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant):
            out[target.id] = value.value
    return out


def test_scope_unchanged():
    """Criterion 9 (negative; holds at the base): ``turn_machine.py``, ``contracts/**``, ``prompt_builder.py``, the
    T-0113 rows (so ``render_call``/``search_payload``/``action_payload``/``status_payload``/``step_prompt`` keep
    their outputs), the tools-off golden module and ``docs/cd`` are unchanged over ``BASE..TIP``; the five bounds keep
    their values at both ends; the T-0115a harness changes at most in its ``E_BUDGET`` line and comments (criterion
    10). The power twin shows the predicate sees T-0121's ``session.py`` change. ``loop.py``'s import rule is
    ``test_toolcall_loop_compact.test_imports_and_purity``; the default target is
    ``test_toolcall_loop_compact.test_default_target_is_budget_times_two_floordiv_three``."""
    gr.require_commits(BASE, TIP, POWER_BASE)
    assert gr.is_ancestor(BASE, TIP), f"{BASE[:9]} is not an ancestor of {TIP[:9]}"
    assert _changed_since_base(*UNCHANGED) == []
    assert gr.changed(POWER_BASE, POWER_TIP, "local_astra/conductor/session.py") == [
        "local_astra/conductor/session.py"
    ]
    for path, name, value in CONSTANTS:
        base_src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASE}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert _module_constants(base_src).get(name) == value, (path, name)
        tip_src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{TIP}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert _module_constants(tip_src).get(name) == value, (
            f"T-0120b criterion 9: {path} {name} is {value}"
        )
    assert session_mod.MAX_CONTINUATION == M
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", "-U0", BASE, TIP, "--", T0115A],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changed = [
        line[1:]
        for line in diff.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    bad = [
        line
        for line in changed
        if not (line.startswith("E_BUDGET = ") or line.lstrip().startswith("#") or not line.strip())
    ]
    assert bad == [], f"T-0120b criterion 10: {T0115A} changes only in E_BUDGET; other lines {bad}"
