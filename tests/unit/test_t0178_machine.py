"""T-0178 group B, the pure machine: R15/R16 rewritten so that a step >= 2 whose call will not run is never voiced, and a
fixed clip says the assistant stopped (docs/tasks/T-0178.md B2-B5; ruling H-T178-1: one report step, a fixed clip as
the fallback, D16 unchanged).

Every expectation is written by hand from the task's criteria. Where a criterion says "as at BASE" or "byte for byte as
at BASE" (B3's policy refusals, B5's no-call report answer), the oracle is BASE's own machine: ``turn_machine.py`` read
from the literal commit ``BASE_SHA`` (main at T-0178's branch point) with ``git show`` and executed as a separate
module, and the row compares the ``(state, commands)`` of every step of the same event sequence on both machines. The
BASE copy is never this tree's file, so an implementation that changed those rows cannot hide behind it; the row skips
(never passes) when the commit is absent.

The surface these rows assume where the task is silent (in the test-author's report):

- the clip is ``C.PlayClip`` with ``kind == "out_of_steps"`` after a budget end or a report-step announcement (B2,
  B5) and ``kind == "not_finished"`` after a malformed draft (B3; amendment B4a: two kinds). The clip's *text* is
  pinned at session level (``test_t0178_session.py``); the audio is not (no clip asset exists, H-T179-3);
- "the turn completes as the empty-step completion does today" (B4) is read as: when nothing of the turn is queued or
  playing, the same tuple that withholds emits the clip, ``CommitHistory(user_only=False)``, ``ScheduleSummary`` and
  goes IDLE (the empty step's tuple at BASE, ``_finish_step``'s ``not held_text`` branch).

Dropped by amendment B4b: ``test_clip_after_queued_speech_waits_for_drain`` (unreachable in the pure machine: R11 at
step >= 2 executes only after the drain, and R14 voices nothing before the step's ``llm.done``). L10 in
``test_turn_invariants.py`` pins the reachable part (the turn reaches IDLE, at most one system clip).

Regime: CPU, pure, the ``clock`` fixture (virtual time); no measurement.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from local_astra.conductor.turn_machine import TurnMachine
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import HistoryTurn, TurnState
from local_astra.toolcall.ask_first import ASKED_FIRST_REASON
from local_astra.toolcall.user_declines import USER_DECLINED_REASON
from tests import _git_range as gr
from tests.unit.test_turn_tool_loop_rows import (
    USER,
    cancels,
    drain,
    of,
    result,
    say,
    sources,
)
from tests.unit.test_turn_tool_rows import done, draft, step, to_thinking

REPO = Path(__file__).resolve().parents[2]
#: main at T-0178's branch point (docs/tasks/T-0178.md; the worktree branched from it). A literal, never a moving ref.
BASE_SHA = "0fd25414"
#: amendment B4a: the budget clip (B2, B5) and the malformed-draft clip (B3)
CLIP_KIND = "out_of_steps"
NOT_FINISHED = "not_finished"
SYSTEM_CLIPS = (CLIP_KIND, NOT_FINISHED)
#: B3's reasons, as the session hands them to the machine (``session._judge_tool``); typed by hand from the census
MALFORMED = ("repair: step 1 only", "combo not in enum", "no call prefix")


# --- the BASE oracle -------------------------------------------------------------------------------------------------

_BASE: ModuleType | None = None


def base_module() -> ModuleType:
    """``local_astra/conductor/turn_machine.py`` at ``BASE_SHA``, executed as its own module."""
    global _BASE
    if _BASE is not None:
        return _BASE
    gr.require_commits(BASE_SHA)
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE_SHA}:local_astra/conductor/turn_machine.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    name = "_t0178_base_turn_machine"
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses read the defining module
    exec(compile(src, f"<{BASE_SHA}:turn_machine.py>", "exec"), mod.__dict__)
    _BASE = mod
    return mod


def machines(**policy):
    """(this tree's machine, BASE's machine) on the same loop policy."""
    from tests.unit.test_turn_invariants import loop_policy

    base = base_module()
    now = TurnMachine(loop_policy(tool_loop=True, **policy))
    old = base.TurnMachine(base.TurnPolicy(thinking=True, tool_loop=True, **policy))
    return now, old


def clips(cmds) -> list:
    """The system clips (amendment B4a: either kind) among ``cmds``."""
    return [c for c in cmds if isinstance(c, C.PlayClip) and c.kind in SYSTEM_CLIPS]


def clip_kinds(cmds) -> list[str]:
    return [c.kind for c in clips(cmds)]


def test_clip_kinds_gain_out_of_steps_and_not_finished():
    """B4a (owned files: ``contracts/commands.py``): ``ClipKind`` gains exactly ``"out_of_steps"`` and
    ``"not_finished"``; ``PlayClip`` takes both; the existing kinds stay."""
    import typing

    kinds = set(typing.get_args(C.ClipKind))
    # T-0179 B-6 re-pin (after ebaf4248): T-0179 adds "lost_thread", and only that
    # T-0192 B1 re-pin (before implementation): T-0192 adds "at_checkout", and only that
    assert kinds == {"ack", "hmm", "still_working", *SYSTEM_CLIPS, "lost_thread", "at_checkout"}, (
        f"T-0178 B4a / T-0179 B-6 / T-0192 B1: ClipKind gains {SYSTEM_CLIPS}, lost_thread and at_checkout; "
        f"it is {sorted(kinds)}"
    )
    for kind in SYSTEM_CLIPS:
        assert C.PlayClip(kind=kind).kind == kind


# --- a scripted loop turn, driven on one machine --------------------------------------------------------------------


def to_report_step(m, clock, *, budget: int, prefix: str = "A."):
    """Step 1 voices ``prefix`` and drafts c1; each further step drafts a valid call with no text until ``budget``
    calls have executed; returns the state at the start of step ``budget + 1`` (the report step) and every command."""
    log: list = []
    s = to_thinking(m, clock)
    if prefix:
        s, cmds = say(m, s, clock, prefix, "t1-gen")
        log += cmds
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen", draft("valid", call_id="c1")))
    log += cmds
    if s.state is TurnState.SPEAKING:
        s, cmds = drain(m, s, clock, prefix)
        log += cmds
    for k in range(2, budget + 1):
        s, cmds = result(m, s, clock, f"c{k - 1}")
        log += cmds
        s, cmds = step(
            m, s, done(clock.advance(10), f"t1-gen-s{k}", draft("valid", call_id=f"c{k}"))
        )
        log += cmds
    s, cmds = result(m, s, clock, f"c{budget}")
    log += cmds
    assert s.active_generate == f"t1-gen-s{budget + 1}", f"premise: the report step started; {log}"
    return s, log


def run(m, events_of, clock):
    """Feed ``events_of(clock)`` (a list of zero-arg event factories) and return [(state, cmds)] per event."""
    s = m.initial()
    out = []
    for make in events_of:
        s, cmds = m.step(s, make())
        out.append((s, list(cmds)))
    return out


# --- B2 / R16: a cancelled call's text is never voiced ---------------------------------------------------------------


def test_r16_withholds_step_text(clock):
    """B2 + B4: ``max_tool_steps=2``; the report step (step 3) says "I will click X." and drafts a valid call. The
    machine cancels it ``step_budget`` as today, voices none of the step's text (no ``TtsSpeak`` sources it, now or
    later), plays one ``PlayClip(out_of_steps)``, and the turn completes: IDLE, ``CommitHistory(user_only=False)``,
    ``ScheduleSummary``, history holding only the earlier played "A."."""
    m = TurnMachine(machines(max_tool_steps=2)[0].policy)
    s, _ = to_report_step(m, clock, budget=2)
    s, said = say(m, s, clock, "I will click X.", "t1-gen-s3")
    assert not of(said, C.TtsSpeak), said
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s3", draft("valid", call_id="c3")))
    assert cancels(cmds) == [("c3", "step_budget")], cmds
    assert not of(cmds, C.ToolExecute), cmds
    assert "click X" not in sources(cmds), (
        f"T-0178 B2: the cancelled step's text was voiced: {cmds}"
    )
    assert not of(cmds, C.TtsSpeak), f"T-0178 B2: nothing is voiced at the cancelled step: {cmds}"
    assert clip_kinds(cmds) == [CLIP_KIND], f"T-0178 B4: one PlayClip(out_of_steps); got {cmds}"
    assert s.state is TurnState.IDLE, f"T-0178 B4: the turn completes; went {s.state.value}: {cmds}"
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)], cmds
    assert of(cmds, C.ScheduleSummary), cmds
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A."), s.history
    # a late drain (the clip's run) voices nothing and stores nothing more
    after, cmds = drain(m, s, clock, "")
    assert not of(cmds, C.TtsSpeak) and after.history == s.history, cmds


def test_r16_at_the_default_budget_nine_requests_withhold_the_ninth(clock):
    """B2 at the default 8 (the t9/t11 shape): 8 executes, the 9th step's announcement withheld, one clip, IDLE."""
    from tests.unit.test_turn_invariants import loop_policy

    m = TurnMachine(loop_policy(tool_loop=True))
    s, log = to_report_step(m, clock, budget=8)
    assert [c.call_id for c in of(log, C.ToolExecute)] == [f"c{k}" for k in range(1, 9)]
    dead = "I will try to reload the page to ensure the content loads correctly."
    s, _ = say(m, s, clock, dead, "t1-gen-s9")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s9", draft("valid", call_id="c9")))
    assert cancels(cmds) == [("c9", "step_budget")]
    assert dead not in sources(log + cmds) and "reload" not in sources(cmds), cmds
    assert clip_kinds(cmds) == [CLIP_KIND] and s.state is TurnState.IDLE, cmds
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A.")


def test_r16_with_nothing_spoken_before_still_completes_with_one_clip(clock):
    """B2/B4 with an empty step 1 (the call ran from THINKING): the clip is still the only thing played and the turn
    reaches IDLE; nothing of the cancelled step is stored."""
    m = TurnMachine(machines(max_tool_steps=1)[0].policy)
    s, _ = to_report_step(m, clock, budget=1, prefix="")
    s, _ = say(m, s, clock, "I will click X.", "t1-gen-s2")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2", draft("valid", call_id="c2")))
    assert cancels(cmds) == [("c2", "step_budget")]
    assert not of(cmds, C.TtsSpeak) and clip_kinds(cmds) == [CLIP_KIND], cmds
    assert s.state is TurnState.IDLE
    assert all("click X" not in h.assistant_spoken for h in s.history), s.history


# --- B3: a malformed draft's text is never voiced --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["I will click X.", "I could not find it."], ids=["announce", "plain"]
)
@pytest.mark.parametrize(
    ("status", "reason"),
    [*(("refused", r) for r in MALFORMED), ("repairable", "schema")],
    ids=[*MALFORMED, "repairable"],
)
def test_refused_draft_step_withheld(clock, status, reason, text):
    """B3 (H-T178-8: the whole step text): at step 2 with budget left, a draft that reached the machine ``refused``
    (with a non-policy reason) or ``repairable`` is cancelled ``refused`` as today; its text is withheld and, nothing
    else being voiced, one clip plays and the turn completes with only step 1's "A." stored."""
    from tests.unit.test_turn_tool_loop_rows import at_step2, loop

    m = loop()
    s, _ = at_step2(m, clock)
    s, _ = say(m, s, clock, text, "t1-gen-s2")
    s, cmds = step(
        m, s, done(clock.advance(10), "t1-gen-s2", draft(status, call_id="c2", reason=reason))
    )
    assert cancels(cmds) == [("c2", "refused")], cmds
    assert not of(cmds, C.ToolExecute) and not of(cmds, C.ToolRepairStart), cmds
    assert not of(cmds, C.TtsSpeak), f"T-0178 B3: the malformed step's text was voiced: {cmds}"
    assert clip_kinds(cmds) == [NOT_FINISHED], (
        f"T-0178 B4a: one PlayClip(not_finished) after a B3 withholding; {cmds}"
    )
    assert s.state is TurnState.IDLE, s.state
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A."), s.history


def _policy_turn(reason: str, said: str, *, step2: bool = True):
    """Event factories: step 1 "A." + c1, drain, result, step 2 says ``said`` and drafts a call refused for ``reason``."""

    def events(clock):
        from tests.unit.test_turn_tool_rows import POLICY

        t = {"now": 0}

        def at(ms):
            t["now"] += ms
            return t["now"]

        evs = []
        onset = 1000
        t["now"] = onset
        evs.append(lambda: E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
        for i, w in enumerate(["open", "the", "menu"]):
            start = onset + i * 250
            evs.append(
                lambda start=start, w=w: E.SttWord(
                    t_ms=start + 400, text=w, start_ms=start, end_ms=start + 200
                )
            )
        evs += [
            lambda: E.SttEndOfTurn(t_ms=2900, t_eot_ms=2850, p_end=1.0),
            lambda: E.SttFlushed(t_ms=3000, t_eot_ms=2850),
            lambda: E.Tick(t_ms=3000 + POLICY.eot_settle_ms),
            lambda: E.LlmCommit(
                t_ms=4000, request_id="t1-gen", canvas_idx=0, segment="answer", text="A."
            ),
            lambda: done(4010, "t1-gen", draft("valid", call_id="c1")),
            lambda: E.PlaybackDrained(t_ms=4100, played_ms=400, played_text="A."),
        ]
        from tests.unit.test_turn_invariants import tool_result_event

        evs += [
            lambda: tool_result_event(4200, "c1", "dry_run"),
            lambda: E.LlmCommit(
                t_ms=4500, request_id="t1-gen-s2", canvas_idx=0, segment="answer", text=said
            ),
            lambda: done(4510, "t1-gen-s2", draft("refused", call_id="c2", reason=reason)),
            lambda: E.PlaybackDrained(t_ms=6000, played_ms=900, played_text=said),
        ]
        return evs

    return events


def _differential(policy: dict, events) -> list:
    now, old = machines(**policy)
    got = run(now, events(None), None)
    want = run(old, events(None), None)
    for i, ((s1, c1), (s0, c0)) in enumerate(zip(got, want, strict=True)):
        assert c1 == c0, f"event {i}: commands differ from BASE {BASE_SHA}:\nnow  {c1}\nbase {c0}"
        assert s1 == s0, f"event {i}: state differs from BASE {BASE_SHA}"
    return got


@pytest.mark.parametrize(
    ("reason", "said"),
    [
        ("asked the user first", "Should I delete the file?"),
        ("the user said no", "Alright, I will leave it."),
    ],
    ids=["ask-first", "declined"],
)
def test_policy_refusals_still_voice(reason, said):
    """B3 negative (adversarial; mutation: every refused step withheld): an ask-first refusal voices its question
    (T-0107a), a user-declined one the model's answer (T-0107e), every command and state byte-identical to BASE.
    The reasons are typed literals equal to the session's constants (the machine compares plain strings, B3)."""
    assert reason in (ASKED_FIRST_REASON, USER_DECLINED_REASON), "premise: a policy reason"
    got = _differential({}, _policy_turn(reason, said))
    voiced = "".join(c.source for _, cmds in got for c in cmds if isinstance(c, C.TtsSpeak))
    assert voiced == "A." + said, voiced
    assert not any(clips(cmds) for _, cmds in got)
    assert got[-1][0].state is TurnState.IDLE


# --- B3a: T-0176's forced-search refusal ------------------------------------------------------------------------------

#: amendment B3a: the card reasons ``session._search_forced`` hands the machine for a search drafted at the forced
#: answer step (T-0176 criterion 7, H-T176-1), typed by hand from T-0176; equality with the session's constants is a
#: premise checked in each row (the machine compares plain strings)
SEARCH_CAP_REASONS = (
    "Gearvis limit: 5 searches per question",
    "Gearvis limit: 3 failed searches per question",
)
#: a real answer drawn from what was found (``announces`` is None) and the t7 announcement (``announces`` is not None)
FOUND_ANSWER = "The latest deluxe volume is volume four, listed at forty dollars."
T7_ANNOUNCEMENT = (
    "I will search for the latest volume of the Vinland Saga Deluxe edition available on Amazon."
)


def _search_cap_premises(reason: str) -> None:
    from local_astra.conductor import session as session_mod
    from local_astra.toolcall.announced_call import announces

    assert (session_mod.SEARCH_CAP_CARD, session_mod.SEARCH_FAILED_CAP_CARD) == SEARCH_CAP_REASONS
    assert reason in SEARCH_CAP_REASONS
    assert announces(FOUND_ANSWER) is None, "premise: the answer is not an announcement"
    assert announces(T7_ANNOUNCEMENT) is not None, "premise: the t7 sentence is an announcement"


def test_machine_search_cap_reasons_track_the_session_cards():
    """Amendment B3a drift guard: the machine keeps the two search-cap reasons as plain strings (it stays pure), copies
    of ``session.SEARCH_CAP_CARD`` and ``SEARCH_FAILED_CAP_CARD``. If the session's cards change (a different
    ``MAX_SEARCHES_PER_TURN``, a reworded card) and the machine's copies do not, the forced-step refusal would silently
    fall back to B3's withholding. The set is exactly the two cards, with no duplicate or extra entry."""
    from local_astra.conductor import session as session_mod
    from local_astra.conductor import turn_machine

    got = turn_machine._SEARCH_CAP_REASONS
    want = {session_mod.SEARCH_CAP_CARD, session_mod.SEARCH_FAILED_CAP_CARD}
    assert len(want) == 2, f"premise: two distinct session cards; {want}"
    assert len(got) == 2 and set(got) == want, (
        f"T-0178 B3a: turn_machine._SEARCH_CAP_REASONS {got!r} drifted from the session's cards {sorted(want)}"
    )


@pytest.mark.parametrize("reason", SEARCH_CAP_REASONS, ids=["cap", "failed-cap"])
def test_forced_search_after_a_real_answer_is_voiced(reason):
    """Amendment B3a, plain answer: the forced answer step says what was found, then drafts ``google_search``, which
    reaches the machine refused with a search-cap reason. The text is voiced as at BASE (the model answered "with what
    you found"): every command and state byte-identical to BASE's machine, the voiced text is "A." plus the answer,
    no system clip plays, and the turn reaches IDLE. (At f8e26760 the reason is not a policy reason, so the answer is
    withheld and ``not_finished`` plays: this row is red there.)"""
    _search_cap_premises(reason)
    got = _differential({}, _policy_turn(reason, FOUND_ANSWER))
    voiced = "".join(c.source for _, cmds in got for c in cmds if isinstance(c, C.TtsSpeak))
    assert voiced == "A." + FOUND_ANSWER, (
        f"T-0178 B3a: the forced step's answer is voiced; {voiced!r}"
    )
    assert not any(clips(cmds) for _, cmds in got), (
        f"T-0178 B3a: no clip after a voiced forced-step answer; {[c for _, cs in got for c in clips(cs)]}"
    )
    assert any(C.ToolCancel(call_id="c2", reason="refused") in cmds for _, cmds in got), (
        "premise: the refused search is cancelled as today"
    )
    assert got[-1][0].state is TurnState.IDLE


@pytest.mark.parametrize("reason", SEARCH_CAP_REASONS, ids=["cap", "failed-cap"])
def test_forced_search_after_an_announcement_is_withheld(clock, reason):
    """Amendment B3a, announcement only (the t7 shape; guard width, mutation: the search-cap reasons added to the
    policy reasons, so every forced-step text is voiced): the forced step says only "I will search for …" and drafts
    a search refused with a search-cap reason. It is cancelled ``refused``, its text is withheld, one
    ``PlayClip(not_finished)`` plays, and the turn completes with only step 1's "A." stored."""
    from tests.unit.test_turn_tool_loop_rows import at_step2, loop

    _search_cap_premises(reason)
    m = loop()
    s, _ = at_step2(m, clock)
    s, said = say(m, s, clock, T7_ANNOUNCEMENT, "t1-gen-s2")
    s, cmds = step(
        m, s, done(clock.advance(10), "t1-gen-s2", draft("refused", call_id="c2", reason=reason))
    )
    assert cancels(cmds) == [("c2", "refused")], cmds
    assert "I will search" not in sources(said + cmds), (
        f"T-0178 B3a: the forced step's announcement was voiced: {cmds}"
    )
    assert not of(cmds, C.TtsSpeak), cmds
    assert clip_kinds(cmds) == [NOT_FINISHED], (
        f"T-0178 B3a: one PlayClip(not_finished) after a withheld forced-step announcement; {cmds}"
    )
    assert s.state is TurnState.IDLE, s.state
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A."), s.history


# --- B5: the report answer -------------------------------------------------------------------------------------------


def _report_turn(said: str, budget: int):
    """Event factories: step 1 "A." + c1, drain, then valid calls until ``budget`` executed, then the report step says
    ``said`` and ends with no call, then the drain of what it said."""

    def events(_clock):
        base = _policy_turn("unused", "")(None)[:11]  # up to and including the c1 result
        from tests.unit.test_turn_invariants import tool_result_event

        evs = list(base)
        t = 5000
        for k in range(2, budget + 1):
            evs.append(lambda k=k, t=t: done(t, f"t1-gen-s{k}", draft("valid", call_id=f"c{k}")))
            evs.append(lambda k=k, t=t: tool_result_event(t + 50, f"c{k}", "dry_run"))
            t += 100
        r = f"t1-gen-s{budget + 1}"
        evs += [
            lambda: E.LlmCommit(
                t_ms=t + 300, request_id=r, canvas_idx=0, segment="answer", text=said
            ),
            lambda: done(t + 310, r),
            lambda: E.PlaybackDrained(t_ms=t + 2000, played_ms=900, played_text=said),
        ]
        return evs

    return events


def test_report_answer_without_call_voiced():
    """B5 negative: a report step (step 3 at budget 2) that ends with no call and reports voices its committed text
    exactly as at BASE (every command and state), and plays no clip."""
    said = "I opened the menu and found the settings, but the page did not load further."
    got = _differential({"max_tool_steps": 2}, _report_turn(said, 2))
    voiced = "".join(c.source for _, cmds in got for c in cmds if isinstance(c, C.TtsSpeak))
    assert voiced == "A." + said, voiced
    assert not any(clips(cmds) for _, cmds in got)
    assert got[-1][0].history[-1] == HistoryTurn(user_text=USER, assistant_spoken=f"A. {said}")


def test_report_announcement_withheld(clock):
    """B5: a report step that ends with no call but only announces an action ("I will click the settings item.";
    ``announces`` is not None) is withheld as in B3, followed by the clip; the turn completes with "A." stored."""
    from local_astra.toolcall.announced_call import announces

    said = "I will click the settings item."
    assert announces(said) is not None, "premise: the text is an announcement"
    m = TurnMachine(machines(max_tool_steps=2)[0].policy)
    s, _ = to_report_step(m, clock, budget=2)
    s, _ = say(m, s, clock, said, "t1-gen-s3")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s3"))
    assert not of(cmds, C.TtsSpeak), f"T-0178 B5: the announcement was voiced: {cmds}"
    assert not of(cmds, C.ToolCancel) and not of(cmds, C.ToolExecute), cmds
    assert clip_kinds(cmds) == [CLIP_KIND] and s.state is TurnState.IDLE, cmds  # B5: "the B2 clip"
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A.")


def test_announcement_before_the_report_step_is_voiced_as_at_base():
    """B5 guard width (mutation: announcements withheld at every step >= 2): with budget left (step 2 of 8), a no-call
    announcement is the T-0172 re-ask's business, not this task's; the machine voices it exactly as at BASE."""
    said = "I will click the settings item."
    got = _differential({"max_tool_steps": 8}, _report_turn(said, 1))
    voiced = "".join(c.source for _, cmds in got for c in cmds if isinstance(c, C.TtsSpeak))
    assert voiced == "A." + said and not any(clips(cmds) for _, cmds in got), voiced


def test_silent_report_step_plays_no_clip(clock):
    """B4 guard width: a report step with no text and no call withholds nothing, so no clip (the empty-step
    completion as today)."""
    m = TurnMachine(machines(max_tool_steps=2)[0].policy)
    s, _ = to_report_step(m, clock, budget=2)
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s3"))
    assert s.state is TurnState.IDLE and not clips(cmds), cmds
