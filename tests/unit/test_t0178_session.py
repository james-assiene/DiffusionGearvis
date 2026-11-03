"""T-0178 group B through the session: the report step's fact, the withheld dead text, the system clips and the turn
line (docs/tasks/T-0178.md B1, B2-B5 at session level, B7, B8; amendment B4a: two clip kinds).

Every row runs over the session fakes (``tests/unit/_tool_turn_support.tool_rig``: ``FakeLlm`` with one script per step,
the recording executor, ``RecordingAudio``, the ``ScriptedClock``). Expected request bytes are computed in the row with
``loop.step_prompt`` and the payload builders (T-0113 proved them against the template oracle), exactly as
``test_session_tool_loop.py`` does; nothing is copied from an implementation.

What the rows fix where the task is silent (in the test-author's report):

- **REPORT_FACT placement** is pinned as behaviour, not as a helper: the report step's last message text *starts with*
  BASE's continuation byte for byte, and what follows holds ``REPORT_FACT`` (stripped), no ``<|tool_response>``, no
  ``<tool_response|>`` and no call. Every other field of the prompt and every earlier message equals BASE's. Any
  separator between the two is the implementer's.
- **The clip's text** is read off what the session hands the audio device (``RecordingAudio.clip_plays()``, played
  with ``source=""``): ``.text`` is the ruled sentence for its kind. The audio is not pinned (no clip asset exists;
  H-T179-3 waits on James), and the clip must never reach history or a ``tts.utterance`` record.
- **``extras.loop_report``** is read as B7 states it: ``{"fact": bool, "withheld": "budget" | "refused" |
  "announced" | None, "clip": bool}``, written (amendment B7a) only on a loop-on turn where the fact was sent,
  something was withheld or a clip played; absent on every other turn line and with the loop off.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, n = 1 and
deterministic per row; no measurement.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_astra.conductor import session as session_mod
from local_astra.contracts import commands as C
from local_astra.contracts.llm import Prompt
from local_astra.toolcall import loop
from tests.fakes.search import FakeSearch
from tests.unit import test_session_tool_loop as tsl
from tests.unit._tool_turn_support import VALID_CLICK, answer, final_message, tool, tool_rig
from tests.unit.test_loop_report_fact import H_T178_3
from tests.unit.test_session_loop import POLICY, RecordingMachine
from tests.unit.test_session_tool_loop import (
    CLICK2,
    CLICK2_ARGS,
    CLICK3,
    CLICK_ARGS,
    RECORDED_CLICK,
    S1,
    S2,
    loop_policy,
    loop_turn,
    oracle,
    sent,
    step_request,
)

OUT_OF_STEPS_TEXT = "I ran out of steps before I could finish."  # H-T178-5, B2
NOT_FINISHED_TEXT = "I could not finish that."  # H-T178-5, B3
REPORT = "I opened the menu and the settings item, but the page stopped responding."
DEAD = "I will click the search result."
RESP_OPEN, RESP_CLOSE = "<|tool_response>", "<tool_response|>"


def tail(prompt: Prompt) -> str:
    return prompt.messages[-1].text()


def requests(rig: Any) -> list[dict[str, Any]]:
    return rig.kind("llm.request")


#: amendment B4a: the two system clip kinds (the machine's own "ack" filler is not one)
SYSTEM_CLIPS = ("out_of_steps", "not_finished")


def system_clips(rig: Any) -> list[Any]:
    return [u for u in rig.audio.clip_plays() if getattr(u, "kind", None) in SYSTEM_CLIPS]


def clip_texts(rig: Any) -> list[str]:
    """The text of every system clip handed to the audio device (``source=""``)."""
    return [str(getattr(u, "text", "")) for u in system_clips(rig)]


def voiced(rig: Any) -> str:
    return rig.spoken_sources()


def utterance_sources(rig: Any) -> list[str]:
    return [str(r.get("source", "")) for r in rig.kind("tts.utterance")]


def assert_fact_appended(got: Prompt, base: Prompt, what: str) -> None:
    """B1: ``got`` is ``base`` with text appended to its continuation, and that text holds REPORT_FACT."""
    assert got.model_copy(update={"messages": got.messages[:-1]}) == base.model_copy(
        update={"messages": base.messages[:-1]}
    ), f"T-0178 B1 ({what}): only the continuation may change"
    assert got.messages[-1].role == base.messages[-1].role == "assistant"
    t, b = tail(got), tail(base)
    assert t.startswith(b), (
        f"T-0178 B1 ({what}): the fact is append-only; BASE's continuation is not a prefix:\n"
        f"base ...{b[-200:]!r}\ngot  ...{t[len(b) - 50 : len(b) + 200]!r}"
    )
    extra = t[len(b) :]
    assert H_T178_3 in extra, (
        f"T-0178 B1 ({what}): REPORT_FACT after the last tool response; got {extra!r}"
    )
    for marker in (RESP_OPEN, RESP_CLOSE, "<|tool_call>"):
        assert marker not in extra, (
            f"T-0178 B1 ({what}): the fact sits outside any payload; {extra!r}"
        )


# --- B1 ----------------------------------------------------------------------------------------------------------------


async def test_report_fact_only_on_report_step(tmp_path):
    """B1: at ``max_tool_steps=2`` the step-3 request is BASE's ``step_prompt`` followed by REPORT_FACT (every earlier
    byte BASE's); step 2 is byte-identical to BASE; the ``llm.request`` record of step 3 alone has
    ``report_fact: true``, every other record lacks the key. The report answer (no call) is voiced as today (B5)."""
    async with tool_rig(
        tmp_path, policy=loop_policy(2), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(S2), tool(CLICK2)], [answer(REPORT)])
        await loop_turn(rig, 3)
        pass1 = sent(rig, "t1-gen")
        s1 = (S1, "click_at", CLICK_ARGS, RECORDED_CLICK)
        s2 = (S2, "click_at", CLICK2_ARGS, RECORDED_CLICK)
        assert step_request(rig, 2) == oracle(pass1, [s1]), (
            "T-0178 B1: step 2 is byte-identical to BASE"
        )
        assert_fact_appended(step_request(rig, 3), oracle(pass1, [s1, s2]), "step 3")
        assert voiced(rig) == S1 + REPORT, voiced(rig)
        assert clip_texts(rig) == [], clip_texts(rig)
    marked = [
        (r.get("request_id"), r.get("report_fact")) for r in requests(rig) if "report_fact" in r
    ]
    assert marked == [("t1-gen-s3", True)], f"T-0178 B1: report_fact only on step 3; {marked}"
    # B1 negative (guard width; mutation: the fact on every step >= 2): the same turn at the default 8 sends no request
    # with REPORT_FACT and no record with ``report_fact``. In this row (not its own) because it holds at BASE, and a
    # pending row that passes is an XPASS; its own premise is the positive half above.
    async with tool_rig(
        tmp_path / "eight", policy=loop_policy(8), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(S2), tool(CLICK2)], [answer(REPORT)])
        await loop_turn(rig, 3)
        pass1 = sent(rig, "t1-gen")
        s1 = (S1, "click_at", CLICK_ARGS, RECORDED_CLICK)
        s2 = (S2, "click_at", CLICK2_ARGS, RECORDED_CLICK)
        assert step_request(rig, 3) == oracle(pass1, [s1, s2])
    assert not [r for r in requests(rig) if "report_fact" in r]


async def test_report_fact_survives_compaction(tmp_path, monkeypatch):
    """B1 with compaction (implementation notes: compaction keeps the last two responses, which hold the fact): with
    ``MAX_CONTINUATION`` small enough that the report step's continuation is compacted, that request is still the
    compacted continuation followed by REPORT_FACT, and its record says ``report_fact: true``.

    Premise, from the rendered lengths (``loop.continuation_chars`` of the fixture search step, 486 chars each):
    three searches are 1458 chars, over the patched budget 1400, so step 4 compacts by eliding step 1's response
    (the last two kept; the stub saves 268, so 1190 chars remain); two searches (972) fit, so steps 2-3 are not
    compacted. With the fact (161 chars of H-T178-3 plus any separator) the compacted step 4 is about 1351, inside
    1400 whether or not the size check counts the fact."""
    monkeypatch.setattr(session_mod, "MAX_CONTINUATION", 1400)
    search = tsl.SEARCH_CALL
    async with tool_rig(
        tmp_path,
        policy=loop_policy(3),
        script=[answer("Searching."), tool(search)],
        search=FakeSearch(),
    ) as rig:
        rig.llm.script_steps(
            [answer("Searching."), tool(search)],
            [answer("Searching."), tool(search)],
            [answer(REPORT)],
        )
        await loop_turn(rig, 4)
        got = step_request(rig, 4)
        assert tail(got).rstrip().endswith(H_T178_3), tail(got)[-300:]
        assert tail(got).count(H_T178_3) == 1
        assert tail(got).count(RESP_CLOSE) == 3, "premise: three responses (one a stub)"
        assert loop.ELIDED_SEARCH_NOTE in tail(got), "premise: step 1's response was elided"
    compacted = [n for n in rig.kind("note") if n.get("what") == "tool loop compacted"]
    assert [n.get("step") for n in compacted] == [4], (
        f"premise: compaction ran at step 4 only; {compacted}"
    )
    marked = [r.get("request_id") for r in requests(rig) if r.get("report_fact") is True]
    assert marked == ["t1-gen-s4"], marked
    extras = rig.record("t1").extras
    assert extras.get("tool_loop_end") != "over_budget", extras


# --- B1a: the report fact always fits -----------------------------------------------------------------------------

#: reviewer A's probe, by hand (the rendered lengths are T-0120b's fixture numbers, ``test_session_loop_compaction``):
#: step 1 speaks ``PREFIX`` and searches (1 587 chars); steps 2-8 each withhold a 201-char text ("Step k notes " plus
#: 188 ``w``) and search (201 + 69 + 1 496 = 1 766). After 8 executed searches the report step's BASE continuation is
#: 1 587 + 7 * 1 766 = 13 949 chars: within ``MAX_CONTINUATION`` (14 000), so BASE sends it uncompacted, and inside
#: B1a's window 13 839-14 000 (14 000 - 161 < 13 949; the fact is 160 chars, plus any separator), so appending the fact
#: without compacting first gives about 14 111 > 14 000 and ``over_budget`` (f8e26760). Under B1a the compaction target
#: is 14 000 - len(separator + fact) (13 838 with a two-char separator), so s9 compacts to ``budget * 2 // 3`` (9 225):
#: eliding steps 0-2 leaves 9 855, steps 0-3 leave 8 491 (any separator up to 600 chars gives the same elisions).
PROBE_PAD = 188
PROBE_CHARS = 13_949
PROBE_COMPACTED = 8_491


def _probe_step(k: int) -> tuple[Any, ...]:
    from tests.unit import test_session_loop_compaction as cmp

    text, name, args, payload = cmp.search_step(k)
    return (text if k == 1 else f"Step {k} notes " + "w" * PROBE_PAD, name, args, payload)


async def test_report_fact_fits_after_compaction_at_the_window(tmp_path):
    """Amendment B1a (reviewer A's probe; adversarial: no new ``over_budget`` because of the fact). Seven maximal
    searches plus one more at budget 8 leave a 13 949-char continuation at the report step s9: BASE would send it, so
    under B1a it is sent, carrying REPORT_FACT, with ``report_fact: true``. The session compacts against
    ``MAX_CONTINUATION - len(REPORT_SEPARATOR + REPORT_FACT)``, so the sent continuation is the hand join with steps
    0-3 stubbed (8 491 chars) followed by the fact, and the whole of it, fact included, is within 14 000. The loop ends
    ``final``, the report is voiced and ``loop_report.fact`` is true."""
    from tests.unit import test_session_loop_compaction as cmp

    steps = [_probe_step(k) for k in range(1, 9)]
    assert len(cmp.hand_join(steps, {})) == PROBE_CHARS, "premise: the probe's BASE continuation"
    assert 14_000 - len(H_T178_3) - 1 < PROBE_CHARS <= 14_000, "premise: inside B1a's window"
    assert len(cmp.hand_join(steps[:7], {})) == 12_183 <= 14_000, "premise: s2..s8 never compact"
    assert len(cmp.hand_join(steps, cmp.stubbed([0, 1, 2]))) == 9_855
    want_base = cmp.hand_join(steps, cmp.stubbed([0, 1, 2, 3]))
    assert len(want_base) == PROBE_COMPACTED
    search = FakeSearch({cmp.query(k): cmp.rows(k) for k in range(1, 9)})
    async with tool_rig(
        tmp_path,
        policy=loop_policy(8),
        search=search,
        script=[answer(tsl.PREFIX), tool(cmp.search_call(1))],
    ) as rig:
        rig.llm.script_steps(
            *[[answer(_probe_step(k)[0]), tool(cmp.search_call(k))] for k in range(2, 9)],
            [answer(REPORT)],
        )
        await rig.ask(tsl.QUESTION)
        await tsl.until_steps(rig, 1)
        await rig.turn_done(limit_ms=60_000)
    extras = rig.record("t1").extras
    assert extras.get("tool_loop_end") == "final", (
        f"T-0178 B1a: the fact never makes a sendable report step over budget; tool_loop_end "
        f"{extras.get('tool_loop_end')!r}, notes {tsl.notes(rig)}"
    )
    assert cmp.sent_ids(rig) == list(range(2, 10)), [r for r, _ in rig.llm.step_prompts]
    assert [n["step"] for n in cmp.compactions(rig)] == [9], cmp.compactions(rig)
    note = cmp.compactions(rig)[0]
    assert (note["elided"], note["chars_before"], note["chars_after"]) == (
        [0, 1, 2, 3],
        PROBE_CHARS,
        PROBE_COMPACTED,
    ), f"T-0178 B1a: s9 compacts against the reduced target; {note}"
    s9 = tail(step_request(rig, 9))
    cmp.report_split(s9, want_base)
    assert len(s9) <= 14_000, f"T-0178 B1a: the size check counts the fact; sent {len(s9)} chars"
    marked = [r.get("request_id") for r in requests(rig) if r.get("report_fact") is True]
    assert marked == ["t1-gen-s9"], marked
    assert extras.get("loop_report", {}).get("fact") is True, extras.get("loop_report")
    assert REPORT in voiced(rig), voiced(rig)


async def test_report_fact_flag_false_when_the_report_step_is_not_sent(tmp_path):
    """Amendment B1a, "the flag reflects what was sent" (mutation: the flag set before the size check). T-0120b's
    still-over loop (eight ``type_at`` steps, 15 356 chars after compaction) ends ``over_budget`` at the report step
    s9, so no request carrying the fact was sent: no ``llm.request`` record has ``report_fact``, and the turn line's
    ``loop_report``, if written at all (B7a), has ``fact: false``."""
    from tests.unit import test_session_loop_compaction as cmp

    rig = await cmp._still_over_loop(tmp_path)
    extras = rig.record("t1").extras
    assert extras.get("tool_loop_end") == "over_budget", "premise: the report step is not sent"
    assert cmp.sent_ids(rig) == list(range(2, 9)), "premise: s9 was never sent"
    assert not [r for r in requests(rig) if "report_fact" in r], (
        "T-0178 B1a: report_fact only on a request that was sent"
    )
    report = extras.get("loop_report")
    assert report is None or report.get("fact") is False, (
        f"T-0178 B1a: loop_report.fact is true only when the fact was sent; {report}"
    )


# --- B2-B5 at session level ---------------------------------------------------------------------------------------


async def test_replay_t9_shape(tmp_path):
    """B2 (the desk session's t9, seq 7230-8590): 9 requests at budget 8; step 1 voices its prefix, steps 2-8 each
    click, step 9 announces a reload and drafts a click past the budget. Only the step-1 prefix is voiced, "I will
    try to reload the page…" is never a ``tts.utterance`` source, the call is cancelled ``step_budget`` (the HUD card
    keeps that reason, B8), one ``out_of_steps`` clip is played with its ruled text, and history holds the prefix."""
    prefix = "I will open the chapter."
    dead = "I will try to reload the page to ensure the content loads correctly."
    steps = [[answer(f"Page {k} is open."), tool(CLICK2)] for k in range(2, 9)]
    async with tool_rig(
        tmp_path, policy=loop_policy(8), script=[answer(prefix), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps(*steps, [answer(dead), tool(CLICK3)])
        await loop_turn(rig, 9, limit_ms=60_000)
        cancels = [(c.call_id, c.reason) for c in rig.commands(C.ToolCancel)]
        assert cancels == [("t1-call-s9", "step_budget")], cancels
        card = final_message(rig, "t1-call-s9")
        assert (card.status, card.reason) == ("cancelled", "step_budget"), card
        assert voiced(rig) == prefix, f"T-0178 B2: voiced {voiced(rig)!r}"
        assert clip_texts(rig) == [OUT_OF_STEPS_TEXT], clip_texts(rig)
        history = json.dumps([h.model_dump(mode="json") for h in rig.state.history])
        assert dead not in history and OUT_OF_STEPS_TEXT not in history, history
    assert all("reload" not in s for s in utterance_sources(rig)), utterance_sources(rig)
    assert len([r for r in requests(rig) if r.get("request_id", "").startswith("t1-gen")]) == 9
    extras = rig.record("t1").extras
    assert (extras.get("tool_steps_executed"), extras.get("tool_loop_end")) == (8, "budget"), extras


async def test_announcement_split_across_commits(tmp_path):
    """Adversarial (B2): the report step's dead text arrives over three answer commits, one of them before the call
    commit, and the call is past the budget: none of the three pieces is voiced or stored, the turn ends with the
    clip."""
    pieces = ["I will ", "click the ", "search result."]
    async with tool_rig(
        tmp_path, policy=loop_policy(2), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps(
            [answer(S2), tool(CLICK2)],
            [
                answer(pieces[0]),
                answer(pieces[1], canvas_idx=1),
                answer(pieces[2], canvas_idx=2),
                tool(CLICK3),
            ],
        )
        await loop_turn(rig, 3)
        assert voiced(rig) == S1, voiced(rig)
        assert clip_texts(rig) == [OUT_OF_STEPS_TEXT], clip_texts(rig)
        history = json.dumps([h.model_dump(mode="json") for h in rig.state.history])
    for p in ("click the", "search result"):
        assert all(p not in s for s in utterance_sources(rig)), (p, utterance_sources(rig))
        assert p not in history, history


async def test_malformed_step_withheld_with_not_finished_clip(tmp_path):
    """B3 + B4a at session level: a repairable step-2 draft reaches the machine ``refused`` ("repair: step 1 only");
    its text is withheld (no ``TtsSpeak``, no ``tts.utterance``), the ``not_finished`` clip is played with "I could
    not finish that.", the loop ends ``refused``."""
    from tests.unit._tool_turn_support import REPAIRABLE_CLICK

    said = "I could not find it."
    async with tool_rig(
        tmp_path, policy=loop_policy(8), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(said), tool(REPAIRABLE_CLICK)])
        await loop_turn(rig, 2)
        assert voiced(rig) == S1, voiced(rig)
        assert clip_texts(rig) == [NOT_FINISHED_TEXT], clip_texts(rig)
    assert all(said not in s for s in utterance_sources(rig))
    assert rig.record("t1").extras.get("tool_loop_end") == "refused"


#: amendment B3a: (forced-step text, voiced?, clip texts) by hand; the t7 announcement is T-0176's fixture
B3A_CASES = {
    "answer": ("The latest deluxe volume is volume four, listed at forty dollars.", True, []),
    "announcement": (
        "I will search for the latest volume of the Vinland Saga Deluxe edition available on Amazon.",
        False,
        [NOT_FINISHED_TEXT],
    ),
}


@pytest.mark.parametrize("case", list(B3A_CASES))
async def test_forced_search_refusal_voices_an_answer_and_withholds_an_announcement(tmp_path, case):
    """Amendment B3a through the session, on T-0176's rig (``test_search_withdrawal.wrig``): searches 1-5 run, step
    6's search is refused at execution (search withdrawn), and step 7, the forced answer step, says ``text`` and then
    drafts ``google_search`` again, which T-0176 refuses before execution with ``SEARCH_CAP_CARD``. A real answer is
    voiced and no clip plays; an announcement-only text is withheld (no ``tts.utterance`` source holds it) and the
    ``not_finished`` clip plays "I could not finish that."."""
    from local_astra.toolcall.announced_call import announces
    from tests.unit import test_search_withdrawal as sw

    text, spoken, want_clips = B3A_CASES[case]
    assert (announces(text) is None) is spoken, "premise: the case's announcement shape"
    steps = [
        *sw.search_steps(2, 6),
        [answer(text), tool(sw.search_call(sw.QN[6]))],
        [answer(REPORT)],
    ]
    async with sw.wrig(tmp_path, FakeSearch(), steps=steps) as rig:
        await sw.run_turn(rig, 6)
        got_voiced = voiced(rig)
        got_clips = clip_texts(rig)
    sw.assert_forced_refusal(rig, 7, sw.CAP_CARD, f"B3a {case}")
    assert (text in got_voiced) is spoken, (
        f"T-0178 B3a ({case}): the forced step's text voiced={spoken}; voiced {got_voiced!r}"
    )
    assert got_voiced.startswith(sw.PREFIX), got_voiced
    assert got_clips == want_clips, f"T-0178 B3a ({case}): clips {got_clips}"
    if not spoken:
        assert all("I will search" not in s for s in utterance_sources(rig)), utterance_sources(rig)
    assert REPORT not in got_voiced, "premise: the turn ended at the forced step"


# --- B7 ---------------------------------------------------------------------------------------------------------------

#: case -> (budget, step scripts after step 1, steps, expected loop_report, expected tool_loop_end), by hand from B7
EXTRAS_CASES: dict[str, tuple[int, list[list[Any]], int, dict[str, Any], str]] = {
    "budget": (
        2,
        [[answer(S2), tool(CLICK2)], [answer(DEAD), tool(CLICK3)]],
        3,
        {"fact": True, "withheld": "budget", "clip": True},
        "budget",
    ),
    "report": (
        2,
        [[answer(S2), tool(CLICK2)], [answer(REPORT)]],
        3,
        {"fact": True, "withheld": None, "clip": False},
        "final",
    ),
    "announced": (
        2,
        [[answer(S2), tool(CLICK2)], [answer(DEAD)]],
        3,
        {"fact": True, "withheld": "announced", "clip": True},
        "final",
    ),
}


@pytest.mark.parametrize("case", list(EXTRAS_CASES))
async def test_loop_report_extras(tmp_path, case):
    """B7/B7a: a loop-on turn that reached the report step or withheld text has ``extras.loop_report``;
    ``tool_loop_end`` keeps ``budget``/``final`` as at BASE; the clip is played exactly when ``clip`` is true."""
    budget, scripts, n, want, end = EXTRAS_CASES[case]
    async with tool_rig(
        tmp_path, policy=loop_policy(budget), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps(*scripts)
        await loop_turn(rig, n)
        played = len(system_clips(rig))
    extras = rig.record("t1").extras
    assert extras.get("loop_report") == want, f"T-0178 B7 ({case}): {extras.get('loop_report')!r}"
    assert extras.get("tool_loop_end") == end, extras
    assert played == (1 if want["clip"] else 0), played


async def test_loop_report_extras_refused(tmp_path):
    """B7: a malformed step-2 draft gives ``{"fact": false, "withheld": "refused", "clip": true}`` and
    ``tool_loop_end == "refused"``."""
    from tests.unit._tool_turn_support import REPAIRABLE_CLICK

    async with tool_rig(
        tmp_path, policy=loop_policy(8), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer("I could not find it."), tool(REPAIRABLE_CLICK)])
        await loop_turn(rig, 2)
    extras = rig.record("t1").extras
    assert extras.get("loop_report") == {"fact": False, "withheld": "refused", "clip": True}, extras
    assert extras.get("tool_loop_end") == "refused", extras


async def test_loop_off_no_extras(tmp_path):
    """B7/B7a negatives: with the loop off (tools on, ``TurnPolicy(thinking=True)``), no ``loop_report`` key and no
    ``report_fact`` key anywhere; with the loop on, a turn that never reaches the report step, withholds nothing and
    plays no clip has no ``loop_report`` either. The last block is the premise that makes both negatives non-vacuous:
    the same rig with a report step does carry the key."""
    async with tool_rig(tmp_path, policy=POLICY, script=[answer(S1), tool(VALID_CLICK)]) as rig:
        await rig.ask("open the menu")
        await rig.turn_done()
    extras = rig.record("t1").extras
    assert "loop_report" not in extras and "tool_loop_end" not in extras, extras
    assert not [r for r in requests(rig) if "report_fact" in r]
    # B7a negative: loop on, a 2-step turn that never reaches the report step, withholds nothing and plays no clip
    # writes no key either (its turn line is BASE's)
    async with tool_rig(
        tmp_path / "quiet", policy=loop_policy(8), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await loop_turn(rig, 2)
    quiet = rig.record("t1").extras
    assert "loop_report" not in quiet and quiet.get("tool_loop_end") == "final", quiet
    # premise of both negatives: the same rig with a report step carries the key (else this row is vacuous)
    async with tool_rig(
        tmp_path / "on", policy=loop_policy(1), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await loop_turn(rig, 2)
    assert rig.record("t1").extras.get("loop_report") == {
        "fact": True,
        "withheld": None,
        "clip": False,
    }


# --- the ending clip's run (coordinator ruling on "a clip never outlives its turn", 2026-10-28) ------------------------

#: budget 1, thinking on (the policy every loop rig uses)
NO_ACK = dict(thinking=True, tool_loop=True, max_tool_steps=1)
#: turn 2's answer: one whitespace-only sentence, which the speech gate latches to ``TtsSpeak(text="")``: the machine
#: goes SPEAKING and nothing is synthesised or played (probed on this rig: one empty speak, zero plays). With tools on
#: the answer cleaner drops ``<|think|>`` literals, so T-0007's ``<|think|>`` latch does not apply here.
LATCHED = [answer(" ")]


class NoAckMachine(RecordingMachine):
    """The production machine with its ``PlayClip(kind="ack")`` commands removed (thinking on always plays one at
    THINKING, criterion 7), so a turn can play nothing at all. Every other command and the state are the machine's."""

    def step(self, state: Any, event: Any) -> Any:
        new, cmds = super().step(state, event)
        return new, [c for c in cmds if not (isinstance(c, C.PlayClip) and c.kind == "ack")]


async def _budget_end_then(rig: Any, second: list[Any]) -> None:
    """Turn 1 ends at the budget with the ``out_of_steps`` clip (B2/B4); then turn 2 is asked with ``second``."""
    rig.llm.script_steps([answer(DEAD), tool(CLICK3)])
    await loop_turn(rig, 2)
    assert clip_texts(rig) == [OUT_OF_STEPS_TEXT], (
        f"premise: turn 1 played the ending clip; {clip_texts(rig)}"
    )
    rig.llm.script = list(second)
    await rig.ask("and now")


async def test_turn_after_an_ending_clip_that_plays_nothing_still_ends(tmp_path):
    """T-0178 (implementer's ``_maybe_mark_drain`` guard, f8e26760): the ending clip plays in a run minted after the
    machine is IDLE (a turn-less run). If the next turn plays nothing at all (no ack, every sentence latched to ""),
    a device drain would be that old run's terminal, dropped as stale (T-0092a), and the turn would sit in SPEAKING.
    The session must drain the turn itself: turn 2 reaches IDLE with nothing played and an empty history entry.
    (Mutant without the guard: turn 2 stays in SPEAKING; measured in the test-author's report.)"""
    from local_astra.conductor.turn_machine import TurnPolicy
    from local_astra.contracts.turn import TurnState

    async with tool_rig(
        tmp_path,
        policy=TurnPolicy(**NO_ACK),
        script=[answer(S1), tool(VALID_CLICK)],
        machine_cls=NoAckMachine,
    ) as rig:
        await _budget_end_then(rig, LATCHED)
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and rig.state.turn_id == "t2",
            "T-0178: a turn that plays nothing after an ending clip must still end (it sat in SPEAKING)",
            limit_ms=10_000,
        )
        plays_t2 = [
            u for u, source in rig.audio.plays if getattr(u, "utterance_id", "").startswith("t2")
        ]
        assert [getattr(u, "kind", None) for u in plays_t2 if source_is_clip(rig, u)] == [], (
            f"premise: turn 2 played no clip (no ack); {plays_t2}"
        )
        # premise: turn 2 synthesised nothing (``RecordingTts.requests`` holds ``(utterance_id, text, voice)``); turn 1's
        # synthesis is in the same journal, so the filter is shown to select by utterance id and not to be empty by
        # construction
        by_turn = [(str(uid), text) for uid, text, _voice in rig.tts.requests]
        assert any(uid.startswith("t1") and text for uid, text in by_turn), (
            f"premise: turn 1's synthesis is journalled under t1 ids; {by_turn}"
        )
        synthesised_t2 = [(uid, text) for uid, text in by_turn if uid.startswith("t2")]
        assert synthesised_t2 == [], f"premise: turn 2 synthesised nothing; {synthesised_t2}"
        assert rig.state.history[-1].assistant_spoken == "", rig.state.history[-1]


def source_is_clip(rig: Any, utterance: Any) -> bool:
    return any(u is utterance and source == "" for u, source in rig.audio.plays)


async def test_ending_clip_run_is_turn_less_and_the_next_turns_first_play_flushes_it(tmp_path):
    """Ruling (coordinator, 2026-10-28) on "a clip never outlives its turn": the ending clip plays just after the turn
    closes, never inside the next one. Pinned as: the machine emits the clip in the very tuple that takes the turn to
    IDLE (so its run belongs to no turn), and the next turn's first play is preceded by a flush of that run (G7), so the
    clip never overlaps the next turn's audio."""
    from local_astra.conductor.turn_machine import TurnPolicy
    from local_astra.contracts.turn import TurnState

    async with tool_rig(
        tmp_path,
        policy=TurnPolicy(**NO_ACK),
        script=[answer(S1), tool(VALID_CLICK)],
        machine_cls=NoAckMachine,
    ) as rig:
        await _budget_end_then(rig, [answer("Here it is.")])
        await rig.turn_done("turn 2 never ended")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and rig.state.turn_id == "t2", "turn 2 ends"
        )
        closing = [
            (st, cmds)
            for _, st, cmds in rig.machine.log
            if any(isinstance(c, C.PlayClip) and c.kind == "out_of_steps" for c in cmds)
        ]
        assert len(closing) == 1 and closing[0][0] is TurnState.IDLE, (
            f"the ending clip is emitted in the tuple that ends the turn; {closing}"
        )
        calls = rig.audio.calls
        clip_at = [
            i
            for i, (u, s) in enumerate(rig.audio.plays)
            if getattr(u, "kind", None) == "out_of_steps"
        ]
        assert len(clip_at) == 1, rig.audio.plays
        # the play index of the first turn-2 utterance, mapped onto the ``calls`` journal (one "play" per play)
        play_calls = [i for i, c in enumerate(calls) if c == "play"]
        clip_call = play_calls[clip_at[0]]
        next_call = play_calls[clip_at[0] + 1] if len(play_calls) > clip_at[0] + 1 else None
        assert next_call is not None, f"premise: turn 2 played its answer; calls {calls}"
        assert "flush" in calls[clip_call + 1 : next_call], (
            f"G7: the next turn's first play flushes the clip's run first; calls {calls}"
        )
