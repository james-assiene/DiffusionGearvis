"""T-0135 group A2 (A4-A9, session side): a tool call left at the tail of an unclosed thought is recovered by the live
judge stage when ``Session(..., recover_thought_calls=True)``, and judged like any other call (docs/tasks/T-0135.md).

Every row runs T-0111g's live session over the fakes (``_live_browser_support.feedback_rig``: loop on, one
``FakeLiveBrowser`` as executor and live browser, the HUD sentinel frame), built through ``recovery_rig`` below, which
passes ``recover_thought_calls`` only when the row gives it (``ABSENT``: the keyword is not passed, the BASE
session). A session without the keyword is a named ``AssertionError`` (A4's pinned seam), never a ``TypeError``.

**The interfaces these rows pin** (from A4-A8; listed in the test-author's report):

- ``Session(..., recover_thought_calls: bool | None = None)``; ``True`` with a kit whose ``mode`` is not ``"live"``
  raises ``ValueError`` whose message names ``live``;
- ``session.THOUGHT_CALL_NO_REPAIR: str``, the refusal reason of a recovered ``repairable`` call (A6), distinct from
  T-0114's ``"repair: step 1 only"``, T-0107a's and T-0107e's reasons and T-0092b's ``"repair off: ..."``;
- ``tool.call`` records of a recovered call carry ``from_thought: True`` (every record of that call, refused ones
  included); no other record carries the key;
- the turn line (``TurnRecord.extras``) of a live turn carries ``recover_thought_calls: <bool>`` when the keyword is
  given (``False`` or ``True``), and no such key when it is ``None`` or not passed.

**The recovered call's expected path** is BASE's answer-channel path, written out by hand: the step-k+1 continuation
is ``loop.step_prompt`` over the step-1 prompt as sent with that step's ``step_text`` (``""`` for a step whose only
commit is a thought: the answer channel with thoughts removed) and the executed payload ``{action, screenshot,
status}`` (the fake's action wording, ``clicked at (x_px, y_px)`` with ``x * 1280 // 1000``, ``y * 800 // 1000``).
``test_recovered_follows_answer_channel_path`` compares a recovered run with the same turn whose call came on the
answer channel instead, section by section.

Readings (reported as ambiguities):

- A7 lists ``machine.events()``, but at BASE every ``thought`` commit is submitted to the machine as an
  ``LlmCommit(segment="thought")`` (A9: "thought commits are still yielded and logged as before"; probed at BASE:
  the sentinel reaches ``machine.events()`` only through that event). The rows read A7 as: no machine event other
  than the request's own thought ``LlmCommit`` carries a character of the thought block;
- A4's "a tool kit whose mode is not live" is pinned for a ``dry_run`` kit; tools off (no kit) is not pinned;
- A8 is pinned on live turn lines only; whether a ``dry_run`` session given ``False`` writes the key is not pinned;
- "never fires on prefill or captioner requests" (A5) has no row: neither request's commits reach the judge stage.

Every row fails at BASE 8887909 on the missing keyword (or, where the rig is not reached, on a missing name).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on (step 1 only), ReGround off unless a row says otherwise, n = 1, deterministic.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.toolcall import live
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit._hud_rig import split_thoughts
from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame
from tests.unit._tool_turn_support import REPAIRABLE_CLICK, VALID_CLICK, answer, thought, tool
from tests.unit.test_session_live import (
    ACT1,
    ASK,
    NO,
    executed,
    step_body,
    want_tail,
)
from tests.unit.test_session_reground import P1, click, frames, regrounds
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

Q = '<|"|>'
KEY = "recover_thought_calls"
FROM = "from_thought"
#: the T-0132 D-5 call (``sessions/20261022-204803-1975518.jsonl``, ``t1-gen-s2``), verbatim
TYPE_D5 = f"<|tool_call>call:type_at{{press_enter:true,text:{Q}ada@example.com{Q},x:305,y:448}}<tool_call|>"
TYPE_ARGS = {"press_enter": True, "text": "ada@example.com", "x": 305, "y": 448}
#: the fake's wording for (305, 448) on 1280x800: 305 * 1280 // 1000 = 390, 448 * 800 // 1000 = 358
ACT_TYPE = "clicked at (390, 358)"
#: a recovered call that stays ``repairable`` ("schema invalid": ``sideways`` is not in the enum), returned whole by
#: ``thought_call.recover``. T-0196 moved this premise: it was the doubled close of ``...-210810-...`` (S5),
#: ``VALID_CLICK + "<tool_call|>"``, which T-0196 judges ``valid`` (``test_recovered_doubled_close_is_valid``).
STILL_REPAIRABLE = f"<|tool_call>call:scroll{{direction:{Q}sideways{Q}}}<tool_call|>"
DOUBLED = VALID_CLICK + "<tool_call|>"
CLICK2, CLICK2_ARGS = tsl.CLICK2, tsl.CLICK2_ARGS
CLICK3, CLICK3_ARGS = tsl.CLICK3, tsl.CLICK3_ARGS
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
SENT = "ZQX-T0135-THOUGHT"
PLAN = "the email field sits left of the button so I type there"
DECLINED = "the user said no"
ASKED = "asked the user first"
#: the reasons a recovered repairable refusal must not reuse (T-0114, T-0107e, T-0107a)
OTHER_REASONS = ("repair: step 1 only", DECLINED, ASKED)

ABSENT: Any = object()


# --------------------------------------------------------------------------------------------- the rig


def session_mod() -> Any:
    return importlib.import_module("local_astra.conductor.session")


def no_repair_reason() -> str:
    value = getattr(session_mod(), "THOUGHT_CALL_NO_REPAIR", None)
    assert isinstance(value, str) and value.strip(), (
        "T-0135 A6: session.THOUGHT_CALL_NO_REPAIR (a non-empty str reason constant) does not exist yet"
    )
    assert value not in OTHER_REASONS and not value.startswith("repair off:"), (
        f"T-0135 A6: THOUGHT_CALL_NO_REPAIR is a new reason, not an existing one; got {value!r}"
    )
    return value


def require_keyword() -> None:
    params = inspect.signature(session_mod().Session.__init__).parameters
    assert KEY in params, (
        "T-0135 A4: Session(..., recover_thought_calls: bool | None = None) does not exist yet"
    )


@asynccontextmanager
async def recovery_rig(
    tmp_path: Path,
    browser: Any,
    *,
    script: Any,
    recover: Any = True,
    reground: bool = False,
    kit: Any = None,
    machine_cls: Any = None,
) -> AsyncIterator[Any]:
    """T-0111g's live session with ``recover_thought_calls=recover`` (``ABSENT``: not passed)."""
    kwargs: dict[str, Any] = {}
    if recover is not ABSENT:
        require_keyword()
        kwargs[KEY] = recover
    if reground:
        kwargs["reground"] = True
    async with feedback_rig(
        tmp_path,
        browser,
        script=script,
        session_kwargs=kwargs,
        reground_llm=reground,
        kit=kit,
        machine_cls=machine_cls,
    ) as rig:
        yield rig


def browser3() -> FakeLiveBrowser:
    return FakeLiveBrowser(
        [live_frame("onset")], post_frames=[live_frame("after-1"), live_frame("after-2")]
    )


def d5(call: str = TYPE_D5, *, words: str = PLAN) -> Commit:
    """The D-5 shape: one open thought whose text ends in a complete native call."""
    return thought(f"{SENT} {words}. {call}", open_=True)


async def run_turn(rig: Any, steps: tuple[Any, ...], question: str = tsl.QUESTION) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(question)
    await rig.turn_done("T-0135: the turn never ended", limit_ms=40_000)
    await rig.advance(300)


def records_of(rig: Any, call_id: str) -> list[dict[str, Any]]:
    return [r for r in rig.tool_records() if r.get("call_id") == call_id]


def invocations(browser: FakeLiveBrowser) -> list[tuple[str, dict[str, Any]]]:
    return [(i.call_id, dict(i.arguments)) for i in browser.invocations]


def no_from_thought(rig: Any) -> None:
    carrying = [r for r in rig.tool_records() if FROM in r]
    assert not carrying, f"T-0135 A6: from_thought on a call that was not recovered: {carrying}"


def repair_requests(rig: Any) -> list[Any]:
    """H-T84-2 repair continuations (a continuation whose final assistant text ends in the open call tag)."""
    return [
        p
        for _, p in rig.llm.gen_prompts
        if p.continue_final_message and tsl.tail(p).endswith("<|tool_call>")
    ]


def tool_repair_logged(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if d.get("phase") == "tool_repair"]


# =====================================================================================================
# A5/A6: the recovered call executes
# =====================================================================================================

#: (step-1 script, step scripts, recovered call id, its arguments, executed calls, the next request's steps)
RECOVERED: dict[str, Any] = {
    # step 2 holds only the D-5 thought (the T-0132 shape)
    "step2": (
        [answer(S1), tool(VALID_CLICK)],
        ([d5()], [answer(S3)]),
        "t1-call-s2",
        TYPE_ARGS,
        [("t1-call", tsl.CLICK_ARGS), ("t1-call-s2", TYPE_ARGS)],
        [
            (S1, "click_at", tsl.CLICK_ARGS, executed(ACT1)),
            ("", "type_at", TYPE_ARGS, executed(ACT_TYPE)),
        ],
    ),
    # the open block is two commits: recover() sees their texts concatenated in order (A1)
    "step2_split": (
        [answer(S1), tool(VALID_CLICK)],
        (
            [
                thought(f"{SENT} {PLAN}. <|tool_call>call:type_at{{press_enter:true,", open_=True),
                thought(f"text:{Q}ada@example.com{Q},x:305,y:448}}<tool_call|>\n", open_=True),
            ],
            [answer(S3)],
        ),
        "t1-call-s2",
        TYPE_ARGS,
        [("t1-call", tsl.CLICK_ARGS), ("t1-call-s2", TYPE_ARGS)],
        [
            (S1, "click_at", tsl.CLICK_ARGS, executed(ACT1)),
            ("", "type_at", TYPE_ARGS, executed(ACT_TYPE)),
        ],
    ),
    # a closed thought before the open block does not stop it (the guard-too-wide twin of O-4)
    "step2_after_closed_thought": (
        [answer(S1), tool(VALID_CLICK)],
        ([thought("I looked at the page."), d5()], [answer(S3)]),
        "t1-call-s2",
        TYPE_ARGS,
        [("t1-call", tsl.CLICK_ARGS), ("t1-call-s2", TYPE_ARGS)],
        [
            (S1, "click_at", tsl.CLICK_ARGS, executed(ACT1)),
            ("", "type_at", TYPE_ARGS, executed(ACT_TYPE)),
        ],
    ),
    # two tags in the open block: an earlier complete call quoted mid-thought, the D-5 call at the tail; the text
    # from the *last* tag is judged (mutation 1: from the first, the text holds two calls)
    "step2_two_tags": (
        [answer(S1), tool(VALID_CLICK)],
        ([d5(words=f"{PLAN}, not {CLICK3} as I first thought")], [answer(S3)]),
        "t1-call-s2",
        TYPE_ARGS,
        [("t1-call", tsl.CLICK_ARGS), ("t1-call-s2", TYPE_ARGS)],
        [
            (S1, "click_at", tsl.CLICK_ARGS, executed(ACT1)),
            ("", "type_at", TYPE_ARGS, executed(ACT_TYPE)),
        ],
    ),
    # A5: the turn's ``generate`` (step 1) is recovered too
    "step1": (
        [d5(VALID_CLICK)],
        ([answer(S3)],),
        "t1-call",
        tsl.CLICK_ARGS,
        [("t1-call", tsl.CLICK_ARGS)],
        [("", "click_at", tsl.CLICK_ARGS, executed(ACT1))],
    ),
}


@pytest.mark.parametrize("case", sorted(RECOVERED))
async def test_recovered_call_executes(tmp_path: Path, case: str) -> None:
    """A5/A6 (mutations 5, 7): a request whose only commits are an open thought ending in a tail call executes that
    call through the live path: its ``tool.call`` records carry ``from_thought: true`` (``proposed`` then
    ``executed``, ``source: draft``, its own empty ``spoken_prefix``), the next step request is sent with
    ``step_text == ""`` and the rendered call and payload (the hand oracle), and the turn line has
    ``recover_thought_calls: true``."""
    script, steps, cid, args, calls, oracle_steps = RECOVERED[case]
    assert live.classify(TYPE_D5).arguments == TYPE_ARGS, "premise: the D-5 call is valid"
    browser = browser3()
    async with recovery_rig(tmp_path, browser, script=script, recover=True) as rig:
        await run_turn(rig, steps)
        k_next = len(oracle_steps) + 1
        assert len(tsl.steps_of(rig)) == k_next - 1, (
            f"T-0135 A6: the step-{k_next} request is sent after the recovered call; step requests "
            f"{[r for r, _ in rig.llm.step_prompts]}"
        )
        tail_now = tsl.tail(tsl.step_request(rig, k_next))
        tail_want = want_tail(rig, oracle_steps)
        spoken = rig.spoken_sources()
        extras = dict(rig.record("t1").extras)
        line_tool = dict(rig.record("t1").tool or {})
    assert invocations(browser) == calls, (
        f"T-0135 A5: the recovered {cid} executes its arguments verbatim; executed {invocations(browser)}"
    )
    assert tail_now == tail_want, (
        f"T-0135 A6/A7: the step-{k_next} continuation embeds step_text '' and render_call, as for an "
        f"answer-channel call\n  want {tail_want[-400:]!r}\n  got  {tail_now[-400:]!r}"
    )
    recs = records_of(rig, cid)
    assert [r.get("status") for r in recs] == ["proposed", "executed"], recs
    for r in recs:
        assert r.get(FROM) is True, (
            f"T-0135 A6: every record of the recovered call has from_thought: {r}"
        )
        assert (r["source"], r["name"], r["arguments"], r["spoken_prefix"]) == (
            "draft",
            oracle_steps[-1][1],
            args,
            "",
        ), r
    others = [r for r in rig.tool_records() if r.get("call_id") != cid]
    assert all(FROM not in r for r in others), f"T-0135 A6: from_thought only on {cid}: {others}"
    if cid == "t1-call":
        assert line_tool.get(FROM) is True, f"the turn line's step-1 tool record: {line_tool}"
    assert extras.get(KEY) is True, (
        f"T-0135 A8: the turn line says recover_thought_calls: true; {extras}"
    )
    assert (extras.get("tool_steps_executed"), extras.get("tool_loop_end")) == (
        len(calls),
        "final",
    ), extras
    assert spoken == ("" if cid == "t1-call" else S1) + S3, f"no thought word voiced; {spoken!r}"


async def test_recovered_follows_answer_channel_path(tmp_path: Path) -> None:
    """A6 ("exactly the answer-channel call's path"): the D-5 step 2 recovered, and the same turn whose step 2 is
    the same call on the answer channel, give the same executions, step-3 request body, HUD messages, history,
    voiced text, turn line and ``tool.call`` records (``from_thought`` removed), and the same session-log kinds
    (A8: no new log kind or span)."""
    runs: dict[str, dict[str, Any]] = {}
    for name, step2 in (("thought", [d5()]), ("answer", [tool(TYPE_D5)])):
        browser = browser3()
        async with recovery_rig(
            tmp_path / name,
            browser,
            script=[answer(S1), tool(VALID_CLICK)],
            recover=True,
            machine_cls=_PolicyMachine,
        ) as rig:
            await run_turn(rig, (step2, [answer(S3)]))
            body3 = step_body(rig, 3)
            session_id, log_path = rig.session.session_id, Path(rig.log.path)
        sec = _with_history(rig, tmp_path / name, session_id, log_path)
        runs[name] = {
            "executed": invocations(browser),
            "body3": body3,
            "hud": sec["hud"],
            "history": sec["history"],
            "speaks": [c.model_dump(mode="json") for c in rig.speaks()],
            # the turn line's thought counters legitimately differ (one run has a thought); extras, the step-1
            # ``tool`` record and the span names do not
            "extras": [rec["extras"] for rec in sec["records"]],
            "line tool": [rec["tool"] for rec in sec["records"]],
            "span names": [sorted(rec["spans"]) for rec in sec["records"]],
            "tool.call": [
                {k: v for k, v in r["data"].items() if k != FROM}
                for r in sec["log"]
                if r["kind"] == "tool.call"
            ],
            "kinds": sorted({r["kind"] for r in sec["log"]}),
        }
        if name == "thought":
            flagged = [
                r["data"] for r in sec["log"] if r["kind"] == "tool.call" and FROM in r["data"]
            ]
            assert flagged and all(r["call_id"] == "t1-call-s2" for r in flagged), (
                f"T-0135 A6: the recovered call's records carry from_thought; {flagged}"
            )
    a, b = runs["thought"], runs["answer"]
    assert b["executed"] == [("t1-call", tsl.CLICK_ARGS), ("t1-call-s2", TYPE_ARGS)], "premise"
    for section in a:
        _assert_equal(
            b[section], a[section], f"T-0135 A6 recovered vs answer-channel {section!r} (not BASE):"
        )


# =====================================================================================================
# A4/A8: off and None stay silent
# =====================================================================================================


@pytest.mark.parametrize("recover", [False, None], ids=["false", "none"])
async def test_off_and_none_stay_silent(tmp_path: Path, recover: bool | None) -> None:
    """A4/A8/A9 (mutation 7): the D-5 script with ``False`` or ``None`` ends ``final`` after step 2, executes only
    step 1, sends no step-3 request and writes no ``from_thought``; the turn line has ``recover_thought_calls:
    false`` for ``False`` and no such key for ``None``."""
    browser = browser3()
    async with recovery_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], recover=recover
    ) as rig:
        await run_turn(rig, ([d5()], [answer(S3)]))
        (done2,) = tsl.dones(rig, "t1-gen-s2")
        extras = dict(rig.record("t1").extras)
        n_steps = len(tsl.steps_of(rig))
    assert done2.tool is None, f"T-0135 A4: no call reaches the machine; {done2.tool}"
    assert invocations(browser) == [("t1-call", tsl.CLICK_ARGS)], invocations(browser)
    assert n_steps == 1, "no step-3 request"
    no_from_thought(rig)
    assert (extras.get("tool_steps_executed"), extras.get("tool_loop_end")) == (1, "final"), extras
    if recover is None:
        assert KEY not in extras, f"T-0135 A8: None adds no key; {extras.get(KEY)!r}"
    else:
        assert KEY in extras and extras[KEY] is False, (
            f"T-0135 A8: a given keyword writes recover_thought_calls: false; extras keys {sorted(extras)}"
        )


def _drop_key(tree: Any, path: tuple[str, ...] = ()) -> tuple[Any, list[tuple[str, ...]]]:
    """``tree`` without any ``recover_thought_calls`` key, and the paths it was removed from."""
    found: list[tuple[str, ...]] = []
    if isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            if k == KEY:
                found.append(path)
                continue
            out[k], sub = _drop_key(v, (*path, str(k)))
            found += sub
        return out, found
    if isinstance(tree, list):
        items = []
        for i, v in enumerate(tree):
            x, sub = _drop_key(v, (*path, str(i)))
            items.append(x)
            found += sub
        return items, found
    if isinstance(tree, str) and KEY in tree:
        found.append((*path, "<in a string>"))
    return tree, found


async def _two_live_turns(tmp: Path, recover: Any) -> dict[str, Any]:
    """The D-5 loop turn, then a toolless live turn; every section (T-0121's capture)."""
    tmp.mkdir(parents=True, exist_ok=True)
    async with recovery_rig(
        tmp,
        browser3(),
        script=[answer(S1), tool(VALID_CLICK)],
        recover=recover,
        machine_cls=_PolicyMachine,
    ) as rig:
        rig.llm.script_steps([d5()], [answer(S3)])
        await rig.ask(tsl.QUESTION)
        await t0107a._until_turns(rig, 1, "the D-5 turn")
        rig.llm.script = [answer("It is a small shop. ")]
        await rig.ask("what is on the page")
        await t0107a._until_turns(rig, 2, "the toolless live turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    return _with_history(rig, tmp, session_id, log_path)


async def test_false_differs_from_absent_only_by_key(tmp_path: Path) -> None:
    """A8/A9 ("with ``False``, the only difference is A8's key"; every live turn line): the D-5 turn and a toolless
    live turn with ``False`` equal the same session without the keyword in every section (machine steps, HUD,
    prompts, request bodies, session-log records, turn lines, history) once ``recover_thought_calls`` is removed; the
    key sits only under turn-line ``extras`` (and ``turn.record``'s copy), ``False`` on both lines. ``None`` adds
    it nowhere."""
    off = await _two_live_turns(tmp_path / "false", False)
    absent = await _two_live_turns(tmp_path / "absent", ABSENT)
    none = await _two_live_turns(tmp_path / "none", None)
    off_clean, where = _drop_key(off)
    _, where_absent = _drop_key(absent)
    _, where_none = _drop_key(none)
    assert where_absent == [], f"premise: no key without the keyword; {where_absent}"
    assert where_none == [], f"T-0135 A8: None adds no key; found at {where_none}"
    assert where and all(p[-1] == "extras" for p in where), (
        f"T-0135 A8: the key lives only in turn-line extras; found at {where}"
    )
    assert [rec["extras"].get(KEY) for rec in off["records"]] == [False, False], (
        "T-0135 A8: every live turn line, the toolless one too, says recover_thought_calls: false"
    )
    for section in t0107a.SECTIONS:
        _assert_equal(
            absent[section], off_clean[section], f"T-0135 A9 False vs absent {section!r}:"
        )
        _assert_equal(absent[section], none[section], f"T-0135 A9 None vs absent {section!r}:")


# =====================================================================================================
# A7: never voiced, never stored
# =====================================================================================================


async def test_thought_never_voiced_or_stored(tmp_path: Path) -> None:
    """A7 (adversarial; H4, P6', L5, L6): the sentinel and the plan words before the tag appear in no ``TtsSpeak``,
    audio play, history entry, HUD message, ``Episode``, turn line, step-3 request body or prompt, and in no machine
    event except the step's own thought ``LlmCommit`` (A9; see the module docstring). The thought stays in the
    session log's ``llm.commit`` verbatim; the call reaches the step-3 request only as ``render_call``.

    Narrowed by T-0180 for the HUD only (R-1, B-R5): the recovered thought may appear as the ``text`` of an
    ``astra.thought``, and nowhere else on the HUD (every other message, and every other field of
    ``astra.thought``, is searched as before). Anti-vacuity: it **is** in the step-2 ``astra.thought`` text."""
    browser = browser3()
    async with recovery_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], recover=True
    ) as rig:
        await run_turn(rig, ([d5()], [answer(S3)]))
        assert [c for c, _ in invocations(browser)] == ["t1-call", "t1-call-s2"], (
            f"premise (T-0135 A5): the recovered call executed; {invocations(browser)}"
        )
        prompt3 = tsl.step_request(rig, 3)
        body3 = json.dumps(step_body(rig, 3))
        events = list(rig.machine.events())
        # T-0180 B-R5: the HUD searched is every message but astra.thought's ``text`` field
        split = split_thoughts(rig.published.messages)
        places = {
            "TtsSpeak": json.dumps([c.model_dump(mode="json") for c in rig.speaks()]),
            "audio": repr(getattr(rig.audio, "plays", "")),
            "history": tsl.history_text(rig),
            "HUD": split.elsewhere,
            "Episode": repr(rig.memory.episodes),
            "turns.jsonl": "\n".join(rig.lines()),
            "step-3 body": body3,
            "step-3 prompt": prompt3.model_dump_json(),
        }
    for secret in (SENT, PLAN):
        bad = [where for where, text in places.items() if secret in text]
        assert not bad, f"T-0135 A7: {secret!r} reached {bad}"
        carrying = [e for e in events if secret in json.dumps(e.model_dump(mode="json"))]
        stray = [
            e
            for e in carrying
            if not (
                isinstance(e, E.LlmCommit)
                and e.segment == "thought"
                and e.request_id == "t1-gen-s2"
            )
        ]
        assert not stray, (
            f"T-0135 A7: a machine event other than the thought commit carries it: {stray}"
        )
    assert "<|channel>" not in tsl.tail(prompt3) and "type_at" in tsl.tail(prompt3), "premise"
    logged = [d for d in rig.kind("llm.commit") if d.get("segment") == "thought"]
    assert any(SENT in d["text"] and d["text"].endswith(TYPE_D5) for d in logged), (
        "T-0135 A7: the thought commit stays in the session log verbatim"
    )
    # T-0180 B1/B6 anti-vacuity (B-R5): the step-2 thought commit is shown, as step 2's own block
    shown = [text for step, kind, _, text in split.thoughts if step == 2 and kind == "step"]
    assert any(SENT in text and PLAN in text for text in shown), (
        "T-0180 B1 (narrowed A7): the recovered step-2 thought is in a step=2, kind='step' astra.thought text; "
        f"published thoughts {split.thoughts}"
    )


# =====================================================================================================
# A6: repairable recovered calls are refused, never repaired
# =====================================================================================================


@pytest.mark.parametrize("step", [1, 2])
async def test_repairable_refused_not_repaired(tmp_path: Path, step: int) -> None:
    """A6 (P5'; mutation 4, the repair at step 1): a recovered ``repairable`` call (the ``sideways`` scroll; the doubled close until T-0196) is refused
    with ``THOUGHT_CALL_NO_REPAIR`` at step 1 and at step 2 (ahead of T-0114's "repair: step 1 only"): no
    ``ToolRepairStart``, no repair request, no tokenize, no ``tool_repair`` record, nothing executed for it, its
    records carry ``from_thought``, and ``tool_loop_end == "refused"``."""
    verdict = live.classify(STILL_REPAIRABLE)
    assert (verdict.status, verdict.reason) == ("repairable", "schema invalid"), verdict
    reason = no_repair_reason()
    browser = browser3()
    if step == 1:
        script, steps, cid, mid = [d5(STILL_REPAIRABLE)], ([answer(S3)],), "t1-call", "t1-gen"
    else:
        script, steps = [answer(S1), tool(VALID_CLICK)], ([d5(STILL_REPAIRABLE)], [answer(S3)])
        cid, mid = "t1-call-s2", "t1-gen-s2"
    async with recovery_rig(tmp_path, browser, script=script, recover=True) as rig:
        await run_turn(rig, steps)
        (done,) = tsl.dones(rig, mid)
        cards = rig.tool_calls(cid)
        repair_starts = rig.commands(C.ToolRepairStart)
        repairs = repair_requests(rig)
        extras = dict(rig.record("t1").extras)
    assert done.tool is not None and (done.tool.call_id, done.tool.status, done.tool.reason) == (
        cid,
        "refused",
        reason,
    ), f"T-0135 A6: the recovered repairable draft as the machine sees it: {done.tool}"
    assert cards and (cards[-1].status, cards[-1].reason) == ("refused", reason), cards
    assert repair_starts == [] and repairs == [] and rig.llm.tokenized == [], (
        f"T-0135 A6: never repaired; ToolRepairStart {repair_starts}, repairs {len(repairs)}, "
        f"tokenized {rig.llm.tokenized}"
    )
    assert tool_repair_logged(rig) == [], "no tool_repair request in the log"
    assert cid not in [c for c, _ in invocations(browser)], invocations(browser)
    recs = records_of(rig, cid)
    assert recs and all(r.get(FROM) is True for r in recs), recs
    assert recs[-1]["status"] == "refused" and recs[-1]["reason"] == reason, recs[-1]
    assert extras.get("tool_loop_end") == "refused", extras


def test_recovered_doubled_close_is_valid() -> None:
    """T-0196 criterion 8 (A5/A6 amended): the doubled close of ``...-210810-...``, recovered whole from the open
    thought, is ``valid`` with its single-close arguments (the session rows: ``test_t0196_session_doubled_close``)."""
    from local_astra.toolcall import thought_call

    got = thought_call.recover(f"{SENT} {PLAN}. {DOUBLED}")
    assert got == DOUBLED, got
    v = live.classify(got)
    assert (v.status, v.name, v.arguments, v.reason) == (
        "valid",
        "click_at",
        tsl.CLICK_ARGS,
        None,
    ), v


# =====================================================================================================
# A6: decline and ask-first apply first
# =====================================================================================================

#: (question, step-1 script, reason, the record's own flag)
SAFETY: dict[str, tuple[str, list[Commit], str, str]] = {
    "decline": (NO, [d5(VALID_CLICK)], DECLINED, "user_declined"),
    "decline_repairable": (NO, [d5(STILL_REPAIRABLE)], DECLINED, "user_declined"),
    "ask_first": (tsl.QUESTION, [answer(ASK), d5(VALID_CLICK)], ASKED, "asked_first"),
    "ask_first_repairable": (
        tsl.QUESTION,
        [answer(ASK), d5(STILL_REPAIRABLE)],
        ASKED,
        "asked_first",
    ),
}


@pytest.mark.parametrize("case", sorted(SAFETY))
async def test_decline_and_ask_first_still_win(tmp_path: Path, case: str) -> None:
    """A6 (mutation 8): a "no" transcript (T-0107e) and a spoken prefix ending "?" (T-0107a) refuse the recovered
    call with their own reasons, ahead of ``THOUGHT_CALL_NO_REPAIR`` for a repairable one; nothing executes, nothing
    is repaired, the records carry ``from_thought`` and their own flag, and the loop ends ``refused``."""
    question, script, reason, flag = SAFETY[case]
    browser = browser3()
    async with recovery_rig(tmp_path, browser, script=script, recover=True) as rig:
        await rig.ask(question)
        await rig.turn_done("T-0135: the turn never ended", limit_ms=40_000)
        await rig.advance(300)
        cards = rig.tool_calls("t1-call")
        repairs = repair_requests(rig)
        extras = dict(rig.record("t1").extras)
    assert browser.executes() == 0, f"T-0135 A6: never executed; {browser.calls}"
    assert cards and (cards[-1].status, cards[-1].reason) == ("refused", reason), (
        f"T-0135 A6: the recovered call is refused with {reason!r}; cards {cards}"
    )
    assert repairs == [] and rig.commands(C.ToolRepairStart) == [], "never repaired"
    recs = records_of(rig, "t1-call")
    assert recs and all(r.get(FROM) is True for r in recs), recs
    assert recs[-1].get(flag) is True and recs[-1]["reason"] == reason, recs[-1]
    assert extras.get("tool_loop_end") == "refused", extras


# =====================================================================================================
# A9: an answer-channel call is unchanged
# =====================================================================================================

ANSWER_CHANNEL: dict[str, list[Commit]] = {
    # a closed thought quoting a tagged call, then a real answer-channel call
    "closed_thought_quotes": [thought(f"{SENT} I could {CLICK3}"), answer(S2), tool(CLICK2)],
    # an open thought ending in a complete call, and an answer-channel call kept by the same request
    "open_thought_then_tool": [d5(CLICK3), tool(CLICK2)],
}


@pytest.mark.parametrize("case", sorted(ANSWER_CHANNEL))
async def test_answer_channel_call_unchanged(tmp_path: Path, case: str) -> None:
    """A9 (adversarial; mutations 3 and 9, the thought's call preferred over the answer's): with ``True``, a step 2
    that kept an answer-channel ``tool`` commit executes that call (CLICK2, never the thought's CLICK3), writes no
    ``from_thought``, and every section equals the same session without the keyword except the turn-line key."""
    runs = {}
    for name, recover in (("true", True), ("absent", ABSENT)):
        browser = browser3()
        async with recovery_rig(
            tmp_path / name,
            browser,
            script=[answer(S1), tool(VALID_CLICK)],
            recover=recover,
            machine_cls=_PolicyMachine,
        ) as rig:
            await run_turn(rig, (ANSWER_CHANNEL[case], [answer(S3)]))
            session_id, log_path = rig.session.session_id, Path(rig.log.path)
        no_from_thought(rig)
        runs[name] = (
            invocations(browser),
            _with_history(rig, tmp_path / name, session_id, log_path),
        )
    (calls_t, sec_t), (calls_a, sec_a) = runs["true"], runs["absent"]
    assert calls_t == calls_a == [("t1-call", tsl.CLICK_ARGS), ("t1-call-s2", CLICK2_ARGS)], (
        f"T-0135 A9: the answer's call executes, not the thought's; True {calls_t}, absent {calls_a}"
    )
    clean, where = _drop_key(sec_t)
    assert where and all(p[-1] == "extras" for p in where), where
    for section in t0107a.SECTIONS:
        _assert_equal(
            sec_a[section], clean[section], f"T-0135 A9 {case} True vs absent {section!r}:"
        )


# =====================================================================================================
# A1/A5: what never fires
# =====================================================================================================

NOT_RECOVERED: dict[str, Commit | list[Commit]] = {
    # T-0102 spirit: text after the close
    "prose_after_close": d5(f"{TYPE_D5} Actually, I should scroll."),
    # O-4: a call split across a closed thought and the open block is never stitched
    "stitched_closed_then_open": [
        thought(f"{SENT} <|tool_call>call:type_at{{press_enter:true,text:{Q}ada@example.com{Q},"),
        thought("x:305,y:448}<tool_call|>", open_=True),
    ],
    # a complete call in an earlier closed thought, then an open thought with no tag
    "call_in_closed_then_open_prose": [
        thought(f"{SENT} {TYPE_D5}"),
        thought("Now I wait for the page.", open_=True),
    ],
    # ``thought_open`` is per canvas: the final non-tool commit is an answer commit (A5(c) reads the segment)
    "open_answer_last": [d5(), tsl.open_answer("I will type the address.")],
    # O-4: open, closed, open -- the trailing open run is only the last commit, which holds no tag; stitching the
    # first open commit across the closed one would give a complete call (reviewer A, mutant: ``else: clear()`` ->
    # ``elif commit.segment != "thought": clear()``)
    "open_closed_open": [
        thought(
            f"<|tool_call>call:type_at{{press_enter:true,text:{Q}ada@example.com{Q},", open_=True
        ),
        thought(" done looking."),
        thought("x:305,y:448}<tool_call|>", open_=True),
    ],
    # A5(c) reads the segment: a whitespace-only answer commit flagged ``thought_open`` ends the request, so the
    # final non-tool commit is not a thought (reviewer A, mutant: keying on ``thought_open`` alone, which would
    # append " " to the block and still recover, the tail being whitespace)
    "open_answer_whitespace_last": [d5(), tsl.open_answer(" ")],
    # an open thought holding the tag but no close
    "no_close": thought(
        f"{SENT} <|tool_call>call:type_at{{press_enter:true,x:305,y:448}}", open_=True
    ),
}


@pytest.mark.parametrize("case", sorted(NOT_RECOVERED))
async def test_tail_prose_not_recovered(tmp_path: Path, case: str) -> None:
    """A1/A5 (mutations 1 and 2; O-4): with ``True``, a step 2 whose open thought has prose after the close, a call
    stitched from a closed thought and the open block, a call only in an earlier closed thought, an open-flagged
    answer commit last, or a tag with no close, ends ``final`` with no execution after step 1 and no
    ``from_thought``; the turn line still says ``recover_thought_calls: true``."""
    step2 = NOT_RECOVERED[case]
    step2 = step2 if isinstance(step2, list) else [step2]
    browser = browser3()
    async with recovery_rig(
        tmp_path, browser, script=[answer(S1), tool(VALID_CLICK)], recover=True
    ) as rig:
        await run_turn(rig, (step2, [answer(S3)]))
        (done2,) = tsl.dones(rig, "t1-gen-s2")
        extras = dict(rig.record("t1").extras)
    assert done2.tool is None, f"T-0135 A1/A5: no call is recovered; {done2.tool}"
    assert invocations(browser) == [("t1-call", tsl.CLICK_ARGS)], invocations(browser)
    no_from_thought(rig)
    assert (extras.get("tool_loop_end"), extras.get(KEY)) == ("final", True), extras


async def test_not_on_pass2_or_repair(tmp_path: Path) -> None:
    """A5 (mutation 6): with ReGround on, a pass-2 response shaped like D-5 (an open thought ending in a valid call)
    still gives ``pass2_unparseable`` and pass 1's (500, 250) executes, as without the keyword; and an H-T84-2
    repair whose response is that shape gives the same records and executions as without the keyword. No record
    carries ``from_thought`` in either."""
    pass2 = [d5(click(900, 100))]
    seen: dict[str, Any] = {}
    for name, recover in (("true", True), ("absent", ABSENT)):
        browser = frames()
        async with recovery_rig(
            tmp_path / f"p2-{name}",
            browser,
            script=[answer(S1), tool(VALID_CLICK)],
            recover=recover,
            reground=True,
        ) as rig:
            rig.llm.script_pass2(pass2)
            await run_turn(rig, ([answer(S3)],))
            assert rig.llm.pass2_ids, "premise: pass 2 ran"
        no_from_thought(rig)
        seen[name] = (
            [r["outcome"] for r in regrounds(rig)],
            invocations(browser),
            rig.tool_records(),
        )
    assert seen["true"][0] == ["pass2_unparseable"], (
        f"T-0135 A5: pass 2 is never recovered; {seen['true'][0]}"
    )
    assert seen["true"][1] == [("t1-call", P1)], f"pass 1's arguments execute; {seen['true'][1]}"
    assert seen["true"] == seen["absent"], "T-0135 A5: pass 2 as without the keyword"

    repaired: dict[str, Any] = {}
    for name, recover in (("true", True), ("absent", ABSENT)):
        browser = browser3()
        async with recovery_rig(
            tmp_path / f"rp-{name}",
            browser,
            script=[
                answer(S1),
                tool(REPAIRABLE_CLICK),
            ],
            recover=recover,
        ) as rig:
            rig.llm.script_continuation(d5(VALID_CLICK))
            rig.llm.script_steps([answer(S3)])
            await rig.ask(tsl.QUESTION)
            await rig.turn_done("T-0135: the repair turn never ended", limit_ms=40_000)
            await rig.advance(300)
            assert repair_requests(rig), "premise: the H-T84-2 repair was sent"
        no_from_thought(rig)
        repaired[name] = (invocations(browser), rig.tool_records())
    assert repaired["true"] == repaired["absent"], (
        f"T-0135 A5: a repair's D-5-shaped response is not recovered; True {repaired['true']}, "
        f"absent {repaired['absent']}"
    )


async def test_aborted_request_not_recovered(tmp_path: Path) -> None:
    """A5(a): a takeover after the D-5 thought commit was delivered but before the request ended aborts it, and
    nothing executes for it, no ``tool.call`` is written; the positive control (the same step-1 script, not
    aborted) executes the recovered call."""
    first = d5(VALID_CLICK)
    browser = browser3()
    async with recovery_rig(
        tmp_path / "abort", browser, script=[first, thought(" more", open_=True)], recover=True
    ) as rig:
        rig.llm.hold_after(1)
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: any(
                isinstance(e, E.LlmCommit) and e.segment == "thought" and SENT in e.text
                for e in rig.machine.events()
            ),
            "premise: the D-5 thought commit was delivered",
            limit_ms=6_000,
        )
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        rig.submit(E.UserTakeover(t_ms=now + 1, onset_ms=now))
        await rig.until(lambda: rig.state.turn_id == "t2", "the aborted turn never closed")
        rig.llm.release()
        rig.llm.script = [answer("Okay. ")]
        await rig.words("stop that")
        await rig.end_of_turn()
        await rig.until(lambda: len(rig.lines()) >= 2, "t2 never ended", limit_ms=20_000)
        await rig.advance(2_000)
        assert any(tsl._engine("t1-gen").search(rid) for rid in rig.llm.aborted), (
            f"premise: the t1 request was aborted; aborted {rig.llm.aborted}"
        )
    assert browser.executes() == 0, (
        f"T-0135 A5(a): an aborted request is not recovered; {browser.calls}"
    )
    assert [r for r in rig.tool_records() if r.get("call_id", "").startswith("t1-")] == []
    control = browser3()
    async with recovery_rig(tmp_path / "control", control, script=[first], recover=True) as rig:
        await run_turn(rig, ([answer(S3)],))
    assert invocations(control) == [("t1-call", tsl.CLICK_ARGS)], (
        f"control (T-0135 A5): not aborted, the same thought is recovered; {invocations(control)}"
    )


# =====================================================================================================
# A4: live only
# =====================================================================================================


async def test_non_live_refused(tmp_path: Path) -> None:
    """A4 (``.claude/rules/conductor.md``: no silent no-op): ``True`` with a ``dry_run`` kit raises ``ValueError``
    naming ``live``; the guard-too-wide twins: ``False`` and ``None`` with the same kit, and ``True`` with the live
    kit, build a session."""
    require_keyword()
    with pytest.raises(ValueError, match="live"):
        async with recovery_rig(
            tmp_path / "dry-true", FakeLiveBrowser(), script=[], recover=True, kit=T.make_kit()
        ):
            pass
    for name, recover in (("dry-false", False), ("dry-none", None)):
        async with recovery_rig(
            tmp_path / name, FakeLiveBrowser(), script=[], recover=recover, kit=T.make_kit()
        ) as rig:
            assert rig.session.tool_kit.mode == "dry_run", "premise: the dry_run kit"
    async with recovery_rig(
        tmp_path / "live-true", FakeLiveBrowser(), script=[], recover=True
    ) as rig:
        assert rig.session.tool_kit.mode == "live", "premise: the live kit"
