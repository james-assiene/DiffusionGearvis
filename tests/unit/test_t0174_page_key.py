"""T-0174 group B2, B2-1/B2-2/B2-4: the ``page`` key on a live step's settled response, tied to v2
(docs/tasks/T-0174.md).

**The surface these rows pin** (the task's names; readings stated where it is silent):

- ``Session(..., page_fact: bool = False)`` (keyword, default ``False``). With ``page_fact=True``, an **executed**
  live action whose ``resolved`` carries a ``bool`` ``navigated`` settles with BASE's payload plus
  ``page: "new page loaded"`` (``True``) or ``page: "no new page loaded"`` (``False``), then B8 adds ``screenshot``.
  No key when ``navigated`` is absent or not a ``bool`` (``None``, ``0``, ``1``, ``"true"``, ``"false"``: an int is
  not a bool here, so neither a truthiness test nor an ``is not None`` test passes), and none for ``refused`` or
  ``failed`` outcomes even when their ``resolved`` carries a ``bool`` (B2-1 says "an executed live action").
- The tool is not read: ``click_at``, ``type_at`` and ``press_key`` each get the key from ``resolved`` alone (B1
  decides which tools carry ``navigated``; the fake here scripts it per call id).
- B2-2: the rendered entry is ``page:<|"|>…<|"|>`` and sorts between ``action`` and ``repeat_of_step``/``screenshot``
  under ``loop._entries``' dictsort; the rows check the literal rendered response (typed by hand), so a key that
  sorted after ``screenshot`` (or was spliced into the text after it) fails. The T-0137/T-0138 harness
  ``_SCREENSHOT`` regexes (copied below, byte for byte, with their ``_RESPONSE_BLOCK``) extract the same value from a
  v2 continuation as from the same continuation without ``page``. ``session.py:4165`` (the label reads
  ``settled.get("screenshot")``) and T-0137's ``_guard_outcome`` are checked by behaviour: with ``page_fact`` on, the
  whole step request equals the off run's once the ``page`` entries are removed (the label, T-0136's clause, the
  images), and the guard's ``repeat_of_step`` fact and its stop are BASE's (the fact never carries ``page``).
- B2-4: the default (no keyword) is BASE even when ``resolved`` carries ``navigated`` (active row: it holds at BASE);
  ``page_fact=False`` behaves the same (pending: the keyword does not exist at BASE).
- B2-1 cli: ``Session(page_fact=True)`` iff the resolved live guidance is ``v2`` (``cli.live_guidance_of``); for
  ``v1`` (default, flag, profile) and outside live the keyword is absent or ``False`` (those rows hold at BASE and
  are active). Reading (reported): "absent or ``False``" is all B2-1 says; group A's A6 differential
  (``test_t0174_v1_identity.py``) compares every ``Session`` keyword ``cli`` passes for v1 with BASE's, so it is the
  stricter fence (an explicit ``page_fact=False`` for v1 would fail there, not here).

Expected payloads are hand-written dicts rendered through ``loop.step_prompt`` (T-0113's oracle, as
``test_t0137_session``), plus literal rendered strings typed by hand; never a call into the code under test.

Group B2. Every new-behaviour row was written ahead as ``pending(task="T-0174")`` and activated for ``/implement
T-0174`` group B2 (markers removed); at BASE each fails on a named ``AssertionError`` (``Session(..., page_fact=...)`` does not exist, or the key is absent).
``test_default_is_base``, ``test_cli_page_fact[default|flag-v1|profile-v1|...v1-over-v2]`` and
``test_cli_no_page_fact_outside_live`` hold at BASE and are active.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps), live,
1280x800 frames, repair on, T-0134c's label on, echo ``model``, ReGround off; n = 1, deterministic. BASE = 284b2e0f.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.toolcall import ToolOutcome
from tests.unit import test_t0137_session as t137
from tests.unit._live_browser_support import (
    failed,
    feedback_rig,
    jpeg_frame,
    live_feedback,
    refused,
    session_mod,
    turn_logs,
)
from tests.unit._tool_turn_support import answer, patch_bound, tool
from tests.unit.test_session_live import card, everything_logged, step_tail, want_tail

Q = '<|"|>'
S = t137.S
REPORT = t137.REPORT
CLICK, CLICK_ARGS, ACT = t137.CLICK, t137.CLICK_ARGS, t137.ACT
TYPE = t137.type_call(500, 250, "hi", True)
TYPE_ARGS = {"press_enter": True, "text": "hi", "x": 500, "y": 250}
PRESS = t137.press("Enter")
PRESS_ARGS = {"combo": "Enter"}
#: tool -> (the call text, its parsed arguments, the fake's action string, echo "model"; typed by hand)
CALLS: dict[str, tuple[str, dict[str, Any], str]] = {
    "click_at": (CLICK, CLICK_ARGS, ACT),
    "type_at": (TYPE, TYPE_ARGS, "clicked at (500, 250)"),
    "press_key": (PRESS, PRESS_ARGS, "pressed Enter"),
}
NEW = "new page loaded"
NO_NEW = "no new page loaded"
#: the screen key P and its captures (T-0137's): ``same()`` is B8 ``unchanged``, ``other_bytes`` is ``updated``
same, other_bytes = t137.same, t137.other_bytes

#: copied byte for byte from scripts/gpu/t0137_repeat_guard_ab.py:2359-2360 and scripts/gpu/t0138_suite_ab.py:1961-1962
#: (identical in both)
_RESPONSE_BLOCK = re.compile(r"<\|tool_response>.*?<tool_response\|>", re.DOTALL)
_SCREENSHOT = re.compile(r'\bscreenshot:<\|"\|>(updated|unchanged)<\|"\|>')
#: one rendered ``page`` entry (B2-1's two values), with the comma that follows it (``screenshot`` always does)
_PAGE_ENTRY = re.compile(r'page:<\|"\|>(?:new page loaded|no new page loaded)<\|"\|>,')


#: the step id the fake's call ids carry: step 1 is ``t1-call``, step k is ``t1-call-s<k>``
def cid(step: int) -> str:
    return "t1-call" if step == 1 else f"t1-call-s{step}"


# --------------------------------------------------------------------------------------------- seams


def require_page_fact() -> None:
    params = inspect.signature(session_mod().Session.__init__).parameters
    assert "page_fact" in params, "T-0174 B2-1: Session(..., page_fact=...) does not exist yet"


def page_rig(
    tmp: Path,
    browser: Any,
    *,
    script: Any,
    page_fact: bool | None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """``feedback_rig`` with the T-0134c label on and ``Session(..., page_fact=<page_fact>)``; ``None`` passes no
    keyword (BASE). A missing keyword is a named ``AssertionError``."""
    kw: dict[str, Any] = {"live_feedback": live_feedback(screen_label=True, page_extent=False)}
    if page_fact is not None:
        require_page_fact()
        kw["page_fact"] = page_fact
    kw.update(extra or {})
    return feedback_rig(tmp, browser, script=script, session_kwargs=kw)


_MISSING = object()


def script_nav(
    browser: Any, step: int, value: object = _MISSING, *, outcome: str = "executed"
) -> None:
    """Step ``step``'s call settles through the fake's own rules, with ``navigated: value`` merged into ``resolved``
    (``_MISSING``: no key). ``outcome`` ``failed``/``refused``: that outcome instead, its ``resolved`` carrying
    ``navigated`` (adversarial: a bool there must not give a key)."""

    def handle(fb: Any, inv: Any) -> ToolOutcome:
        if outcome == "failed":
            out = failed(inv, "TimeoutError during " + inv.name)
        elif outcome == "refused":
            out = refused(inv, "the element is covered")
        else:
            out = fb.perform(inv)
        if value is _MISSING:
            return out
        assert out.status == outcome, f"premise: the fake settled {inv.call_id} as {out.status}"
        return out.model_copy(update={"resolved": {**out.resolved, "navigated": value}})

    browser.handlers[cid(step)] = handle


def ex(
    action: str, updated: bool, page: str | None = None, fact: dict[str, Any] | None = None
) -> dict[str, Any]:
    """An executed step's settled payload, by hand: BASE's ``{action, status}``, B8's ``screenshot``, and B2-1's
    ``page`` (and T-0137's fact) when given."""
    out: dict[str, Any] = {
        "action": action,
        "screenshot": "updated" if updated else "unchanged",
        "status": "executed",
    }
    if page is not None:
        out["page"] = page
    if fact is not None:
        out["repeat_of_step"] = fact
    return out


def rendered(name: str, action: str, page: str, screenshot: str) -> str:
    """B2-2's rendered response, typed by hand (``action``, ``page``, ``screenshot``, ``status``: dictsort order)."""
    return (
        f"<|tool_response>response:{name}{{action:{Q}{action}{Q},page:{Q}{page}{Q},"
        f"screenshot:{Q}{screenshot}{Q},status:{Q}executed{Q}}}<tool_response|>"
    )


def harness_screenshots(text: str) -> list[str]:
    """The T-0137/T-0138 harness extraction, applied per tool-response block (the harness reads the newest block;
    every block here, so a change in any of them shows)."""
    return [(_SCREENSHOT.findall(b) or ["<none>"])[-1] for b in _RESPONSE_BLOCK.findall(text)]


def scrub(obj: Any) -> Any:
    """``obj`` with every rendered ``page`` entry removed from every string (for the on/off differential)."""
    if isinstance(obj, str):
        return _PAGE_ENTRY.sub("", obj)
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def step_dump(rig: Any, k: int) -> Any:
    from tests.unit import test_session_tool_loop as tsl

    return tsl.step_request(rig, k).model_dump(mode="json")


async def two_steps(rig: Any, call: str, *, k: int = 2) -> None:
    """Step 1's call is in the rig's script; steps 2..k-1 repeat ``call``; step k answers (``k`` requests)."""
    steps = [(S[i], call) for i in range(1, k - 1)] + [(REPORT, "")]
    await t137.run(rig, t137.steps_of(steps))


# =====================================================================================================
# B2-1: the keyword and the values
# =====================================================================================================


def test_keyword_default_false() -> None:
    """B2-1: ``Session.__init__`` takes ``page_fact`` as a keyword whose default is ``False``."""
    require_page_fact()
    p = inspect.signature(session_mod().Session.__init__).parameters["page_fact"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False, (
        f"T-0174 B2-1: Session(*, page_fact: bool = False); got {p.kind} default {p.default!r}"
    )


@pytest.mark.parametrize("name", list(CALLS))
async def test_page_values(tmp_path: Path, name: str) -> None:
    """B2-1: step 1 executes with ``navigated: True``, step 2 the same call with ``navigated: False``; the step-3
    continuation carries step 1's payload with ``page: "new page loaded"`` and step 2's with ``page: "no new page
    loaded"`` (each beside B8's ``screenshot``: both captures change the bytes, ``updated``)."""
    call, args, action = CALLS[name]
    browser = t137.browser_of([other_bytes("a"), other_bytes("b")])
    script_nav(browser, 1, True)
    script_nav(browser, 2, False)
    async with page_rig(
        tmp_path, browser, script=[answer(S[0]), tool(call)], page_fact=True
    ) as rig:
        await two_steps(rig, call, k=3)
        assert t137.dispatched(browser) == 2, f"premise: both calls executed; {browser.calls}"
        want = [
            (S[0], name, args, ex(action, True, NEW)),
            (S[1], name, args, ex(action, True, NO_NEW)),
        ]
        got = step_tail(rig, 3)
        assert got == want_tail(rig, want), (
            f"T-0174 B2-1 ({name}): the step-3 continuation carries page per step; got ...{got[-700:]}"
        )


async def test_rendered_order(tmp_path: Path) -> None:
    """B2-2: the rendered response is exactly ``action, page, screenshot, status`` (typed by hand): ``page`` sorts
    before ``screenshot``, and the screenshot entry is BASE's bytes."""
    browser = t137.browser_of([same()])
    script_nav(browser, 1, False)
    async with page_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], page_fact=True
    ) as rig:
        await two_steps(rig, CLICK)
        got = step_tail(rig, 2)
        want = rendered("click_at", ACT, NO_NEW, "unchanged")
        assert got.count(want) == 1, (
            f"T-0174 B2-2: the response renders action, page, screenshot, status in that order; want {want!r}; "
            f"got ...{got[-500:]}"
        )


async def test_absent_navigated_no_key(tmp_path: Path) -> None:
    """B2-1 (adversarial: never a guessed ``"no new page loaded"``): with ``page_fact=True`` and no ``navigated`` in
    ``resolved`` (B1-1b's pending navigation, a driver without the capability), the payload is BASE's: no ``page``
    key, and the words appear nowhere in the request."""
    browser = t137.browser_of([other_bytes("a")])
    async with page_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], page_fact=True
    ) as rig:
        await two_steps(rig, CLICK)
        assert cid(1) not in browser.handlers and not browser.resolved_extra, (
            "premise: no navigated"
        )
        got = step_tail(rig, 2)
        assert got == want_tail(rig, [(S[0], "click_at", CLICK_ARGS, ex(ACT, True))]), (
            f"T-0174 B2-1: no navigated, no page key; got ...{got[-500:]}"
        )
        assert "page loaded" not in got


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", "no new page loaded"], ids=repr)
async def test_non_bool_navigated_no_key(tmp_path: Path, value: object) -> None:
    """B2-1 (adversarial): a ``navigated`` that is not a ``bool`` gives no key (``0``/``1`` included: a truthiness
    or an ``is not None`` test would emit one)."""
    browser = t137.browser_of([other_bytes("a")])
    script_nav(browser, 1, value)
    async with page_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], page_fact=True
    ) as rig:
        await two_steps(rig, CLICK)
        got = step_tail(rig, 2)
        assert got == want_tail(rig, [(S[0], "click_at", CLICK_ARGS, ex(ACT, True))]), (
            f"T-0174 B2-1: navigated={value!r} is not a bool, so no page key; got ...{got[-500:]}"
        )


@pytest.mark.parametrize("outcome", ["failed", "refused"])
async def test_not_executed_no_key(tmp_path: Path, outcome: str) -> None:
    """B2-1 ("an executed live action"): a ``failed`` or ``refused`` outcome whose ``resolved`` carries
    ``navigated: True`` settles with BASE's status payload plus B8's ``screenshot``, and no ``page``."""
    browser = t137.browser_of([], frames=[t137.F0, other_bytes("fresh")])
    script_nav(browser, 1, True, outcome=outcome)
    async with page_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], page_fact=True
    ) as rig:
        await two_steps(rig, CLICK)
        assert t137.statuses(rig, cid(1)) == [outcome], "premise"
        got = step_tail(rig, 2)
        assert "page" not in [
            m.group(1) for m in re.finditer(r"[{,](\w+):", _RESPONSE_BLOCK.findall(got)[-1])
        ], f"T-0174 B2-1: a {outcome} outcome gets no page key; got ...{got[-500:]}"
        assert "page loaded" not in got


# =====================================================================================================
# B2-2: the screenshot key untouched
# =====================================================================================================


@pytest.mark.parametrize("post", ["updated", "unchanged"])
async def test_screenshot_extraction_equal(tmp_path: Path, post: str) -> None:
    """B2-2: the same script with ``page_fact=True`` and without the keyword: the copied harness ``_SCREENSHOT``
    extracts the same value from every response block of the step-3 request (``post`` by hand), and the whole step-2
    and step-3 requests are equal once the ``page`` entries are removed (the T-0134 label that reads
    ``screenshot``, the images, every other byte). Premise: the on run carries the two entries."""
    frames = [other_bytes("a"), other_bytes("b")] if post == "updated" else [same(), same()]
    runs: dict[str, Any] = {}
    for mode, flag in (("on", True), ("off", None)):
        browser = t137.browser_of(list(frames))
        script_nav(browser, 1, True)
        script_nav(browser, 2, False)
        async with page_rig(
            tmp_path / mode, browser, script=[answer(S[0]), tool(CLICK)], page_fact=flag
        ) as rig:
            await two_steps(rig, CLICK, k=3)
            runs[mode] = (step_tail(rig, 3), step_dump(rig, 2), step_dump(rig, 3))
    on_tail, off_tail = runs["on"][0], runs["off"][0]
    assert len(_PAGE_ENTRY.findall(on_tail)) == 2, (
        f"T-0174 B2-1: two page entries; ...{on_tail[-700:]}"
    )
    assert harness_screenshots(on_tail) == harness_screenshots(off_tail) == [post, post], (
        f"T-0174 B2-2: the harness regex reads {harness_screenshots(on_tail)} with page, "
        f"{harness_screenshots(off_tail)} without; want [{post!r}, {post!r}]"
    )
    for i, k in ((1, 2), (2, 3)):
        assert scrub(runs["on"][i]) == runs["off"][i], (
            f"T-0174 B2-2: the step-{k} request with page_fact differs from BASE's beyond the page entries"
        )


async def test_repeat_fact_unchanged(tmp_path: Path) -> None:
    """B2-2 / T-0137 B-6: with ``repeat_guard`` and ``page_fact`` on, a repeated click (both captures unchanged,
    ``navigated: False``) carries BASE's fact ``{step: 1, action, screenshot: "unchanged"}``: the fact never carries
    ``page``; each executed payload carries ``page`` beside it."""
    browser = t137.browser_of([same(), same()])
    script_nav(browser, 1, False)
    script_nav(browser, 2, False)
    async with page_rig(
        tmp_path,
        browser,
        script=[answer(S[0]), tool(CLICK)],
        page_fact=True,
        extra={"repeat_guard": True},
    ) as rig:
        await two_steps(rig, CLICK, k=3)
        want = [
            (S[0], "click_at", CLICK_ARGS, ex(ACT, False, NO_NEW)),
            (S[1], "click_at", CLICK_ARGS, ex(ACT, False, NO_NEW, t137.fact(1, ACT, "unchanged"))),
        ]
        got = step_tail(rig, 3)
        assert got == want_tail(rig, want), (
            f"T-0174 B2-2: the repeat fact is T-0137's, page beside it; got ...{got[-700:]}"
        )
        assert t137.facts_in(got) == [{"action": ACT, "screenshot": "unchanged", "step": 1}], (
            t137.facts_in(got)
        )


async def test_repeat_stop_unchanged(tmp_path: Path) -> None:
    """B2-2 / T-0137 B-5 (``_guard_outcome`` reads ``screenshot``, never ``page``): four identical clicks on an
    unchanging screen, each executed one with ``navigated: False``. Exactly as T-0137's row: three dispatched, the
    fourth refused with A-1's reason and no ``page`` (refused); the executed three carry ``"no new page loaded"``."""
    browser = t137.browser_of([same(), same(), same()])
    for k in (1, 2, 3):
        script_nav(browser, k, False)
    async with page_rig(
        tmp_path,
        browser,
        script=[answer(S[0]), tool(CLICK)],
        page_fact=True,
        extra={"repeat_guard": True},
    ) as rig:
        await t137.run(
            rig, t137.steps_of([(S[1], CLICK), (S[2], CLICK), (S[3], CLICK), (REPORT, "")])
        )
        assert t137.dispatched(browser) == 3, f"T-0137 B-5 under page_fact; {browser.calls}"
        want = [
            (S[0], "click_at", CLICK_ARGS, ex(ACT, False, NO_NEW)),
            (S[1], "click_at", CLICK_ARGS, ex(ACT, False, NO_NEW, t137.fact(1, ACT, "unchanged"))),
            (S[2], "click_at", CLICK_ARGS, ex(ACT, False, NO_NEW, t137.fact(2, ACT, "unchanged"))),
            (
                S[3],
                "click_at",
                CLICK_ARGS,
                t137.refusal(t137.REPEAT_REASON, t137.fact(3, ACT, "unchanged")),
            ),
        ]
        got = step_tail(rig, 5)
        assert got == want_tail(rig, want), (
            f"T-0174 B2-2: the step-5 continuation; got ...{got[-900:]}"
        )
        c = card(rig, cid(4))
        assert (c.status, c.reason) == ("refused", t137.REPEAT_REASON), c


async def test_t0136_outcome_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B2-2 / T-0136: with ``action_outcome`` on (``outcome_of_jpeg`` patched to ``changed near your click``) and
    ``page_fact`` on vs no keyword, the measured outcomes are the same and the step-2/step-3 requests (T-0136's
    clause in the label included) are equal once the ``page`` entries are removed."""
    ao = t137._action_outcome()
    near = ao.Outcome(kind="near", region=None, changed_cells=12, cells=200)
    assert near.text == "changed near your click", near.text
    patch_bound(monkeypatch, ao, "outcome_of_jpeg", lambda *a, **k: near)
    spy = turn_logs(monkeypatch)
    runs: dict[str, Any] = {}
    for mode, flag in (("on", True), ("off", None)):
        post = [
            jpeg_frame(f"after-{i}", seed=i).model_copy(update={"phash": t137.P}) for i in (1, 2)
        ]
        onset = jpeg_frame("onset", seed=0).model_copy(update={"phash": t137.P})
        browser = t137.browser_of(post, frames=[onset])
        script_nav(browser, 1, True)
        script_nav(browser, 2, False)
        async with page_rig(
            tmp_path / mode,
            browser,
            script=[answer(S[0]), tool(CLICK)],
            page_fact=flag,
            extra={"action_outcome": True},
        ) as rig:
            await two_steps(rig, CLICK, k=3)
            kinds = {k: o.kind for k, o in spy.by_turn("t1").action_outcomes.items()}
            runs[mode] = (kinds, step_dump(rig, 2), step_dump(rig, 3), step_tail(rig, 3))
    assert runs["on"][0] == runs["off"][0] == {1: "near", 2: "near"}, (
        runs["on"][0],
        runs["off"][0],
    )
    assert len(_PAGE_ENTRY.findall(runs["on"][3])) == 2, "premise: the on run carries page"
    assert "changed near your click" in str(runs["off"][2]), (
        "premise: T-0136's clause is in the request"
    )
    for i, k in ((1, 2), (2, 3)):
        assert scrub(runs["on"][i]) == runs["off"][i], (
            f"T-0174 B2-2: T-0136's step-{k} request changed beyond the page entries"
        )


# =====================================================================================================
# B2-4: off and default are BASE
# =====================================================================================================


async def test_default_is_base(tmp_path: Path) -> None:
    """B2-4 (active; holds at BASE): no ``page_fact`` keyword, ``navigated`` ``True`` then ``False`` in ``resolved``:
    the step-3 continuation is BASE's (no ``page``), and the words appear nowhere in the session log."""
    browser = t137.browser_of([other_bytes("a"), other_bytes("b")])
    script_nav(browser, 1, True)
    script_nav(browser, 2, False)
    async with page_rig(
        tmp_path, browser, script=[answer(S[0]), tool(CLICK)], page_fact=None
    ) as rig:
        await two_steps(rig, CLICK, k=3)
        want = [
            (S[0], "click_at", CLICK_ARGS, ex(ACT, True)),
            (S[1], "click_at", CLICK_ARGS, ex(ACT, True)),
        ]
        got = step_tail(rig, 3)
        assert got == want_tail(rig, want), f"T-0174 B2-4: default is BASE; got ...{got[-600:]}"
    assert rig.log_records, "premise: the session log was read back"
    assert "page loaded" not in everything_logged(rig)


async def test_explicit_false_is_base(tmp_path: Path) -> None:
    """B2-4: ``page_fact=False`` gives the default's bytes: the step-2 and step-3 requests equal the no-keyword
    run's (no ``page``) with ``navigated`` in ``resolved``."""
    runs: dict[str, Any] = {}
    for mode, flag in (("false", False), ("none", None)):
        browser = t137.browser_of([other_bytes("a"), other_bytes("b")])
        script_nav(browser, 1, True)
        script_nav(browser, 2, False)
        async with page_rig(
            tmp_path / mode, browser, script=[answer(S[0]), tool(CLICK)], page_fact=flag
        ) as rig:
            await two_steps(rig, CLICK, k=3)
            runs[mode] = (step_dump(rig, 2), step_dump(rig, 3))
    assert runs["false"] == runs["none"], "T-0174 B2-4: page_fact=False is BASE"
    assert "page loaded" not in str(runs["false"])


# =====================================================================================================
# B2-1: cli passes page_fact=True iff the resolved guidance is v2
# =====================================================================================================

FLAG, FIELD = "--live-guidance", "live_guidance"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
#: case -> (flags, profile body after LIVE_PROFILE, version); as test_t0174_cli_guidance.SOURCES, typed here
SOURCES: dict[str, tuple[list[str], str, str]] = {
    "default": (
        [],
        "",
        "v3",
    ),  # T-0191 C3: the web-mode default is v3 (v2 since T-0174 D1, v1 in group B2), so page_fact=True
    "flag-v1": ([FLAG, "v1"], "", "v1"),
    "flag-v2": ([FLAG, "v2"], "", "v2"),
    "profile-v1": ([], f'{FIELD} = "v1"\n', "v1"),
    "profile-v2": ([], f'{FIELD} = "v2"\n', "v2"),
    "flag-v1-over-profile-v2": ([FLAG, "v1"], f'{FIELD} = "v2"\n', "v1"),
    "flag-v2-over-profile-v1": ([FLAG, "v2"], f'{FIELD} = "v1"\n', "v2"),
}


@pytest.mark.parametrize(
    "case",
    [c for c in SOURCES],
)
def test_cli_page_fact(tmp_path: Path, capsys: Any, monkeypatch: Any, case: str) -> None:
    """B2-1 (H-T174-2): a live session built by ``cli`` gets ``page_fact=True`` iff the resolved live guidance is
    ``v2`` (or ``v3``, T-0191 A4 amends B2-1); with ``v1`` (flag, profile) the keyword is absent or ``False`` (v1
    cases active: BASE). ``default`` re-scoped by T-0191 C3: the session gets ``LIVE_GUIDANCE["v3"]`` (web mode)."""
    from tests.unit import _t0202_pins as t0202
    from tests.unit.test_cli_t0131 import build as build_mode

    flags, body, version = SOURCES[case]
    # T-0202 D3 re-pin: the wait tool pinned off, so tool_guidance is LIVE_GUIDANCE[version] itself (``is``)
    calls = build_mode(
        tmp_path, capsys, monkeypatch, "off", *t0202.pin_off(flags, body), body=LIVE_PROFILE + body
    )
    kw = calls.session_kwargs[-1]
    from local_astra.conductor import prompt_builder as pb

    want_guidance = getattr(pb, "LIVE_GUIDANCE", {}).get(version)
    assert want_guidance is not None and kw.get("tool_guidance") is want_guidance, (
        f"premise ({case}): the session got LIVE_GUIDANCE[{version!r}] (group A)"
    )
    if version in ("v2", "v3"):
        assert kw.get("page_fact") is True, (
            f"T-0174 B2-1 ({case}): cli passes page_fact=True with {version}; got {kw.get('page_fact', '<absent>')!r}"
        )
    else:
        assert kw.get("page_fact", False) is False, (
            f"T-0174 B2-1 ({case}): no page fact with v1; got page_fact={kw.get('page_fact')!r}"
        )


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_cli_no_page_fact_outside_live(
    tmp_path: Path, capsys: Any, monkeypatch: Any, mode: str
) -> None:
    """B2-1 (active): outside live the session gets no page fact (the resolved guidance is ``None``)."""
    from tests.unit.test_cli_t0131 import build as build_mode

    calls = build_mode(tmp_path, capsys, monkeypatch, mode)
    kw = calls.session_kwargs[-1]
    assert kw.get("page_fact", False) is False, f"T-0174 B2-1 ({mode}): {kw.get('page_fact')!r}"


# =====================================================================================================
# B2-4: the frozen paths over B2's closed range (A6's style, test_t0174_v1_identity)
# =====================================================================================================

#: B2's BASE: 284b2e0f (main after Merge T-0174-b1 plus PROGRESS, the fork point of T-0174-b2-page-key). A literal.
B2_BASE = "284b2e0f"
#: B2's implementation tip: b7648b7f ("T-0174 group B2: page key on executed live steps, tied to guidance v2"). A
#: literal, bumped by the test-author if the implementation gains a commit.
B2_TIP = "b7648b7f"
#: power: group A's implementation commit 31b91d7d and its parent 87172bda (``prompt_builder.py`` changed there)
B2_POWER = ("87172bda", "31b91d7d")
#: B2-4's byte-identical list, written out from the criterion
B2_FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/contracts",
    "local_astra/toolcall",
)


def test_b2_frozen_paths_over_closed_range() -> None:
    """B2-4 (active): ``turn_machine.py``, ``speech_gate.py``, ``prompt_builder.py``, ``contracts/**`` and
    ``toolcall/**`` are byte-identical over ``B2_BASE..B2_TIP`` (two literals, never the working tree), and each exists
    at B2_BASE. Power: the same predicate sees group A's ``prompt_builder.py`` change in 87172bda..31b91d7d."""
    from tests import _git_range as gr

    gr.require_commits(B2_BASE, B2_TIP, *B2_POWER)
    assert gr.is_ancestor(B2_BASE, B2_TIP), f"{B2_BASE} is not an ancestor of {B2_TIP}"
    listed = gr._git("ls-tree", "-r", "--name-only", B2_BASE, "--", *B2_FROZEN).stdout.split()
    for p in B2_FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at {B2_BASE}"
        )
    changed = gr.changed(B2_BASE, B2_TIP, *B2_FROZEN)
    assert changed == [], f"T-0174 B2-4: frozen paths differ in {B2_BASE}..{B2_TIP}: {changed}"
    assert "local_astra/conductor/prompt_builder.py" in gr.changed(
        B2_POWER[0], B2_POWER[1], *B2_FROZEN
    ), "power"
